"""Independent Spec-1 verification: multi-timeframe registry/resampling, timeframe
aware annualisation + UTC settlement, IBKR/data API timeframe propagation, and the
live-NLV / cached-Parquet engine binding.

Verification-only suite for branch ``feat/spec1-multi-timeframe`` (tasks 67-72);
no production edits unless a real bug is found (reported first). Everything is
hermetic: synthetic frames, ``tmp_path`` Parquet caches, fake IBKR/Databento
providers — no network and no live TWS.

Sections
--------
1. Timeframe registry (exact 8 specs) + look-ahead-safe resampler + repository
   auto-resample/cache round-trip.
2. Timeframe annualisation (metrics + strategies/risk) and UTC-calendar-day daily
   settlement / daily-loss resets with intraday bars.
3+): IBKR backfill errors (task 68), API timeframe propagation, EngineMarketService
   live NLV / cached-Parquet term structure, frontend contract sanity.
"""

from __future__ import annotations

import dataclasses
import math
from collections import Counter
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
from configs.instruments import InstrumentsConfig, load_instruments
from configs.settings import Settings
from fastapi.testclient import TestClient
from src.analytics.metrics import (
    PerformanceMetrics,
    compute_metrics,
    resolve_periods_per_year,
)
from src.backtest.engine import EventBacktester, SimpleContractTranslator
from src.backtest.margin_simulator import (
    DailySettlementClock,
    MarginSimulator,
    is_new_utc_day,
    utc_calendar_day,
)
from src.core.types import MarketState, PipelineResult, WeightVector
from src.data import schema as s
from src.data.resampler import VWAP_COLUMN, resample_bars
from src.data.storage import ParquetRepository
from src.data.timeframes import (
    TIMEFRAMES,
    bars_per_year,
    base_timeframe,
    databento_base_schema,
    get_timeframe,
    ibkr_bar_size,
    schema_for_timeframe,
    supported_timeframes,
    timeframe_seconds,
)
from src.risk.quant_risk_overlay import QuantRiskConfig, QuantRiskOverlay
from src.strategies._numeric import (
    annualization_factor_for,
    bars_per_year_for,
)
from src.strategies.allocators.risk_parity import RiskParityAllocator
from src.strategies.allocators.volatility_target import VolatilityTargetAllocator
from src.strategies.timing.tsmom import TSMOMTiming

from tests.fixtures.market import (
    make_bars_frame,
    make_contract_targets,
    make_market_state,
    make_weight_vector,
)

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"
INSTRUMENTS_PATH = CONFIGS_DIR / "instruments.yaml"
BASE = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
FIXED_CLOCK = datetime(2026, 11, 3, 12, 0, tzinfo=UTC)

EXPECTED_TIMEFRAMES: dict[str, dict[str, object]] = {
    "1m": {
        "every": "1m",
        "ibkr": "1 min",
        "databento": "ohlcv-1m",
        "bars_per_year": 347_760,
        "resample_from": None,
        "seconds": 60,
        "schema": "ohlcv-1m",
    },
    "5m": {
        "every": "5m",
        "ibkr": "5 mins",
        "databento": "ohlcv-1m",
        "bars_per_year": 69_552,
        "resample_from": "1m",
        "seconds": 300,
        "schema": "ohlcv-5m",
    },
    "15m": {
        "every": "15m",
        "ibkr": "15 mins",
        "databento": "ohlcv-1m",
        "bars_per_year": 23_184,
        "resample_from": "1m",
        "seconds": 900,
        "schema": "ohlcv-15m",
    },
    "30m": {
        "every": "30m",
        "ibkr": "30 mins",
        "databento": "ohlcv-1m",
        "bars_per_year": 11_592,
        "resample_from": "1m",
        "seconds": 1_800,
        "schema": "ohlcv-30m",
    },
    "1h": {
        "every": "1h",
        "ibkr": "1 hour",
        "databento": "ohlcv-1h",
        "bars_per_year": 5_796,
        "resample_from": "1m",
        "seconds": 3_600,
        "schema": "ohlcv-1h",
    },
    "4h": {
        "every": "4h",
        "ibkr": "4 hours",
        "databento": "ohlcv-1h",
        "bars_per_year": 1_449,
        "resample_from": "1h",
        "seconds": 14_400,
        "schema": "ohlcv-4h",
    },
    "1d": {
        "every": "1d",
        "ibkr": "1 day",
        "databento": "ohlcv-1d",
        "bars_per_year": 252,
        "resample_from": None,
        "seconds": 86_400,
        "schema": "ohlcv-1d",
    },
    "1w": {
        "every": "1w",
        "ibkr": "1 week",
        "databento": "ohlcv-1d",
        "bars_per_year": 52,
        "resample_from": "1d",
        "seconds": 604_800,
        "schema": "ohlcv-1w",
    },
}


@pytest.fixture(scope="module")
def master() -> InstrumentsConfig:
    return load_instruments(INSTRUMENTS_PATH)


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        environment="backtest",
        data_dir=tmp_path / "data",
        parquet_dir=tmp_path / "data" / "parquet",
        duckdb_path=tmp_path / "data" / "spec1.duckdb",
        report_dir=tmp_path / "artifacts",
        configs_dir=CONFIGS_DIR,
        instruments_path=INSTRUMENTS_PATH,
        telemetry_interval_seconds=0.02,
        api_enable_ibkr=False,
        ibkr_paper=True,
    )


def _minute_bars(
    *,
    roots: tuple[str, ...] = ("ES",),
    periods: int = 60,
    start: datetime | None = None,
) -> pl.DataFrame:
    """Canonical 1-minute bars starting at 00:01 UTC (clean right-boundary windows)."""
    return make_bars_frame(
        roots=roots,
        periods=periods,
        start=start or (datetime(2026, 1, 1, 0, 1, tzinfo=UTC)),
        interval_minutes=1,
        instrument_master=load_instruments(INSTRUMENTS_PATH),
    )


# ---------------------------------------------------------------------------
# 1. Registry + resampler + repository timeframe cache (task 67)
# ---------------------------------------------------------------------------


def test_registry_exact_eight_timeframes() -> None:
    assert supported_timeframes() == tuple(EXPECTED_TIMEFRAMES)
    assert len(TIMEFRAMES) == 8
    for spec in TIMEFRAMES:
        expected = EXPECTED_TIMEFRAMES[spec.name]
        assert spec.every == expected["every"]
        assert spec.ibkr_bar_size == expected["ibkr"]
        assert spec.databento_base_schema == expected["databento"]
        assert spec.bars_per_year == expected["bars_per_year"]
        assert spec.resample_from == expected["resample_from"]
        assert spec.seconds == expected["seconds"]
        assert spec.schema == expected["schema"]
        # Helper wrappers agree with the spec.
        assert bars_per_year(spec.name) == expected["bars_per_year"]
        assert ibkr_bar_size(spec.name) == expected["ibkr"]
        assert databento_base_schema(spec.name) == expected["databento"]
        assert schema_for_timeframe(spec.name) == expected["schema"]
        assert base_timeframe(spec.name) == expected["resample_from"]
        assert timeframe_seconds(spec.name) == expected["seconds"]
    # Aliases/whitespace normalise; unknown names are actionable ValueErrors.
    assert get_timeframe("1H").name == "1h"
    assert get_timeframe(" 5min ").name == "5m"
    assert get_timeframe("daily").name == "1d"
    with pytest.raises(ValueError):
        get_timeframe("2h")
    with pytest.raises(ValueError):
        bars_per_year("nope")


@pytest.mark.parametrize(
    ("target", "every_seconds"),
    [
        ("5m", 300),
        ("15m", 900),
        ("30m", 1_800),
        ("1h", 3_600),
        ("4h", 14_400),
        ("1d", 86_400),
        ("1w", 604_800),
    ],
)
def test_resample_aggregates_ohlcv_vwap_and_right_labels(target: str, every_seconds: int) -> None:
    bars = _minute_bars(periods=60)
    # Give every source bar a deterministic volume/vwap so the aggregation is checkable.
    bars = bars.with_columns(
        pl.int_range(0, pl.len()).alias("_i"),
    ).with_columns(
        (pl.col("volume") + 1.0).alias("volume"),
        (pl.col("close") + pl.col("_i") * 0.01).alias(VWAP_COLUMN),
        (pl.lit(1000.0) + pl.col("_i")).alias("open_interest"),
    )
    resampled = resample_bars(bars, target)
    assert not resampled.is_empty()
    assert resampled.height == expected_groups(bars, target, every_seconds)
    # Canonical schema + sort order + vwap column.
    s.ensure_bars_schema(resampled.drop(VWAP_COLUMN))
    assert VWAP_COLUMN in resampled.columns
    assert resampled.schema[s.TIMESTAMP_COLUMN] == pl.Datetime("us", "UTC")
    keys = list(s.BAR_KEY_COLUMNS)
    assert resampled.select(keys).equals(resampled.sort(keys).select(keys))
    # Every output label is the right boundary of its window: label >= all source
    # timestamps in that window and the window is (label - every, label].
    interval = timedelta(seconds=every_seconds)
    timestamps = set(bars.get_column(s.TIMESTAMP_COLUMN).to_list())
    grouped_volumes: Counter[datetime] = Counter()
    grouped_open: dict[datetime, float] = {}
    grouped_close: dict[datetime, float] = {}
    for row in bars.iter_rows(named=True):
        stamp = row[s.TIMESTAMP_COLUMN]
        label = _right_label(stamp, every_seconds)
        grouped_volumes[label] += float(row["volume"])
        grouped_open.setdefault(label, float(row["open"]))
        grouped_close[label] = float(row["close"])
    for row in resampled.iter_rows(named=True):
        label = row[s.TIMESTAMP_COLUMN]
        window_start = label - interval
        in_window = [stamp for stamp in timestamps if window_start < stamp <= label]
        assert in_window, f"no source bars in window for label {label}"
        assert all(stamp <= label for stamp in in_window), "look-ahead leaked"
        assert row["volume"] == pytest.approx(grouped_volumes[label])
        assert row["open"] == pytest.approx(grouped_open[label])
        assert row["close"] == pytest.approx(grouped_close[label])
        assert row[VWAP_COLUMN] > 0.0
    # OHLC invariants per resampled bar.
    assert (
        resampled.filter(
            (pl.col("high") < pl.col("low"))
            | (pl.col("high") < pl.col("open"))
            | (pl.col("high") < pl.col("close"))
            | (pl.col("low") > pl.col("open"))
            | (pl.col("low") > pl.col("close"))
        ).height
        == 0
    )


def expected_groups(bars: pl.DataFrame, target: str, every_seconds: int) -> int:
    """Number of distinct right-boundary windows containing at least one bar."""
    labels = {
        _right_label(stamp, every_seconds)
        for stamp in bars.get_column(s.TIMESTAMP_COLUMN).to_list()
    }
    return len(labels)


def _right_label(stamp: datetime, every_seconds: int) -> datetime:
    """Right-closed window label for a source timestamp (UTC)."""
    if every_seconds == 604_800:  # Polars weekly windows start on Monday 00:00 UTC.
        monday = datetime.combine(
            stamp.date() - timedelta(days=stamp.weekday()), datetime.min.time(), tzinfo=UTC
        )
        return monday + timedelta(weeks=1)
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    seconds = int((stamp - epoch).total_seconds())
    bucket = every_seconds * math.ceil(seconds / every_seconds)
    return epoch + timedelta(seconds=bucket)


def test_resampler_volume_weighted_vwap_and_zero_volume_fallback() -> None:
    bars = _minute_bars(periods=10)
    bars = bars.with_columns(
        pl.lit(2.0).alias("volume"),
        (pl.col("close") + 10.0).alias(VWAP_COLUMN),
    )
    resampled = resample_bars(bars, "5m")
    expected_vwap = float(
        bars.filter(
            (pl.col(s.TIMESTAMP_COLUMN) > datetime(2026, 1, 1, 0, 0, tzinfo=UTC))
            & (pl.col(s.TIMESTAMP_COLUMN) <= datetime(2026, 1, 1, 0, 5, tzinfo=UTC))
        )
        .select((pl.col(VWAP_COLUMN) * pl.col("volume")).sum() / pl.col("volume").sum())
        .item()
    )
    first = resampled.filter(pl.col(s.TIMESTAMP_COLUMN) == datetime(2026, 1, 1, 0, 5, tzinfo=UTC))
    assert first[VWAP_COLUMN][0] == pytest.approx(expected_vwap)
    # Zero total volume -> fallback to the window's last close, never NaN.
    zero = bars.with_columns(pl.lit(0.0).alias("volume"))
    zero_resampled = resample_bars(zero, "5m")
    assert zero_resampled[VWAP_COLUMN].to_list() == pytest.approx(zero_resampled["close"].to_list())


def test_resampler_rejects_finer_target_and_handles_empty() -> None:
    bars = _minute_bars(periods=10)
    with pytest.raises(ValueError, match="finer"):
        resample_bars(bars, "1m", source_timeframe="5m")
    empty = bars.clear()
    result = resample_bars(empty, "5m", source_timeframe="1m")
    assert result.is_empty()
    assert set(result.columns) == {*s.BAR_COLUMNS, VWAP_COLUMN}
    # Minimal single-bar frame still works (one group).
    single = bars.head(1)
    one = resample_bars(single, "5m", source_timeframe="1m")
    assert one.height == 1


def test_repository_timeframe_auto_resample_and_roundtrip(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    repository = ParquetRepository(settings)
    minute = _minute_bars(roots=("ES",), periods=120)
    written = repository.write_timeframe("1m", minute)
    assert written == minute.height
    assert repository.available_roots("ohlcv-1m") == ("ES",)
    # No 5m partition yet: read_timeframe auto-resamples from 1m.
    assert repository.available_roots("ohlcv-5m") == ()
    auto = repository.read_timeframe("5m", roots=("ES",), persist=True)
    assert not auto.is_empty()
    assert auto.height == 24  # 120 one-minute bars -> 24 five-minute windows
    assert repository.available_roots("ohlcv-5m") == ("ES",), "persist=True must cache 5m"
    # Explicit target partition round-trips and survives a fresh read.
    explicit = repository.read_timeframe("5m", roots=("ES",))
    assert explicit.height == auto.height
    assert explicit.equals(auto)
    # Writing an explicit target partition is honoured and read back unchanged.
    repository.delete_root("ES", schema="ohlcv-5m")
    repository.write_timeframe("5m", auto)
    direct = repository.read_timeframe("5m", roots=("ES",))
    assert direct.height == auto.height
    # Empty target+base yields a canonical empty frame, not an exception.
    empty = repository.read_timeframe("4h", roots=("NQ",))
    assert empty.is_empty()
    assert set(empty.columns) >= set(s.BAR_COLUMNS)


# ---------------------------------------------------------------------------
# 2. Timeframe annualisation + UTC daily reset (tasks 69/70)
# ---------------------------------------------------------------------------


def _state_with_returns(
    timeframe: str,
    *,
    returns: np.ndarray | None = None,
    seed: int = 7,
) -> MarketState:
    state = make_market_state(("ES", "NQ"), seed=seed, returns_length=260)
    if returns is not None:
        state = dataclasses.replace(state, returns={"ES": returns, "NQ": returns})
    # Remove the covariance so the realised-returns annualisation path (the one
    # that depends on state.timeframe) is exercised, not the covariance diagonal.
    return dataclasses.replace(
        state, timeframe=timeframe, covariance=None, covariance_instruments=()
    )


def test_annualization_helpers_exact_values() -> None:
    for timeframe, expected in EXPECTED_TIMEFRAMES.items():
        assert bars_per_year_for(timeframe) == float(expected["bars_per_year"])
        assert annualization_factor_for(timeframe) == pytest.approx(
            math.sqrt(float(expected["bars_per_year"]))
        )
    assert bars_per_year_for(None) == 252.0
    assert annualization_factor_for(None) == pytest.approx(math.sqrt(252.0))
    assert bars_per_year_for("1d") == 252.0


def test_market_state_builder_timeframe_covariance_annualization(
    master: InstrumentsConfig,
) -> None:
    from src.data.market_state_builder import MarketStateBuilder

    frame = make_bars_frame(
        roots=("ES", "NQ"), periods=300, start=FIXED_CLOCK - timedelta(days=400)
    )
    as_of = frame.get_column(s.TIMESTAMP_COLUMN).max()
    daily = MarketStateBuilder(instrument_master=master, timeframe="1d").build(
        frame, as_of, equity=1_000_000.0
    )
    intraday = MarketStateBuilder(instrument_master=master, timeframe="5m").build(
        frame, as_of, equity=1_000_000.0
    )
    assert daily.timeframe == "1d" and intraday.timeframe == "5m"
    assert daily.covariance is not None and intraday.covariance is not None
    expected_ratio = float(EXPECTED_TIMEFRAMES["5m"]["bars_per_year"]) / 252.0
    assert float(np.diag(intraday.covariance)[0]) == pytest.approx(
        float(np.diag(daily.covariance)[0]) * expected_ratio, rel=1e-9
    )
    assert intraday.timeframe in {spec.name for spec in TIMEFRAMES}


def test_metrics_vol_sharpe_scale_with_sqrt_bars_per_year() -> None:
    rng = np.random.default_rng(11)
    returns = rng.normal(0.0, 0.01, size=500)
    equity = np.concatenate(([1_000_000.0], 1_000_000.0 * np.cumprod(1.0 + returns)))
    daily = compute_metrics(equity, timeframe="1d")
    assert isinstance(daily, PerformanceMetrics)
    assert daily.periods_per_year == 252
    assert daily.timeframe == "1d"
    for timeframe in ("1m", "5m", "15m", "30m", "1h", "4h", "1w"):
        ppy = int(EXPECTED_TIMEFRAMES[timeframe]["bars_per_year"])
        other = compute_metrics(equity, timeframe=timeframe)
        assert other.periods_per_year == ppy
        assert other.timeframe == timeframe
        expected_ratio = math.sqrt(ppy / 252.0)
        assert other.annualized_volatility == pytest.approx(
            daily.annualized_volatility * expected_ratio, rel=1e-9
        )
        assert other.sharpe == pytest.approx(daily.sharpe * expected_ratio, rel=1e-9)
        assert other.sortino == pytest.approx(daily.sortino * expected_ratio, rel=1e-9)
    # Explicit periods_per_year wins over the timeframe label (legacy contract).
    assert resolve_periods_per_year("5m", periods_per_year=1000) == 1000
    assert resolve_periods_per_year("5m") == 69_552
    override = compute_metrics(equity, timeframe="5m", periods_per_year=252)
    assert override.periods_per_year == 252
    assert override.annualized_volatility == pytest.approx(daily.annualized_volatility)


def test_volatility_target_and_risk_parity_use_state_timeframe() -> None:
    base = make_market_state(("ES", "NQ"), seed=5)
    # Force the realised-returns covariance path so the annualisation factor of
    # the state's timeframe is what drives the scaling.
    no_covariance = dataclasses.replace(base, covariance=None, covariance_instruments=())
    daily = dataclasses.replace(no_covariance, timeframe="1d")
    intraday = dataclasses.replace(no_covariance, timeframe="5m")
    vol_target = VolatilityTargetAllocator(target_annual_volatility=0.12)
    daily_weights = vol_target.allocate(("ES", "NQ"), daily)
    intraday_weights = vol_target.allocate(("ES", "NQ"), intraday)
    ratio = math.sqrt(252.0 / float(EXPECTED_TIMEFRAMES["5m"]["bars_per_year"]))
    for root in ("ES", "NQ"):
        assert intraday_weights.weights[root] == pytest.approx(
            daily_weights.weights[root] * ratio, rel=1e-6
        ), "intraday annualised vol must shrink the target-weights scale"
    risk_parity = RiskParityAllocator(target_annual_volatility=0.12)
    daily_rp = risk_parity.allocate(("ES", "NQ"), daily)
    intraday_rp = risk_parity.allocate(("ES", "NQ"), intraday)
    for root in ("ES", "NQ"):
        assert intraday_rp.weights[root] == pytest.approx(daily_rp.weights[root] * ratio, rel=1e-6)


def test_tsmom_scales_by_state_timeframe_volatility() -> None:
    rng = np.random.default_rng(3)
    returns = rng.normal(0.0005, 0.002, size=200)
    overlay = TSMOMTiming(
        lookback_fast=10,
        lookback_slow=40,
        vol_lookback=20,
        target_volatility=0.12,
        min_scale=0.0,
        max_scale=100.0,
        require_slow_agreement=False,
    )
    daily = _state_with_returns("1d", returns=returns)
    intraday = _state_with_returns("5m", returns=returns)
    daily_factor = overlay.scaling_factor(daily, "ES")
    intraday_factor = overlay.scaling_factor(intraday, "ES")
    assert daily_factor > 0.0 and intraday_factor > 0.0
    expected_ratio = math.sqrt(252.0 / float(EXPECTED_TIMEFRAMES["5m"]["bars_per_year"]))
    assert intraday_factor / daily_factor == pytest.approx(expected_ratio, rel=1e-9)
    weights = make_weight_vector({"ES": 0.5})
    assert overlay.adjust_timing(weights, daily).weights["ES"] == pytest.approx(0.5 * daily_factor)


def test_quant_risk_config_annualization_from_timeframe() -> None:
    daily = QuantRiskConfig()
    assert daily.annualization_factor == 252
    for timeframe in ("1m", "5m", "1h", "4h", "1w"):
        config = QuantRiskConfig(timeframe=timeframe)
        assert config.annualization_factor == int(EXPECTED_TIMEFRAMES[timeframe]["bars_per_year"])
    explicit = QuantRiskConfig(timeframe="5m", annualization_factor=7.0)
    assert explicit.annualization_factor == 7.0, "explicit override must win"
    with pytest.raises(ValueError):
        QuantRiskConfig(timeframe="2h")


def test_quant_overlay_utc_daily_loss_reset_with_intraday_bars() -> None:
    config = QuantRiskConfig(
        timeframe="5m",
        daily_max_loss_pct=0.03,
        drawdown_limit_pct=0.90,
        trailing_stop_pct=0.90,
        daily_loss_kill_switch_enabled=True,
    )
    overlay = QuantRiskOverlay(config=config, instruments=load_instruments(INSTRUMENTS_PATH))
    day1 = datetime(2026, 1, 5, 14, 0, tzinfo=UTC)
    overlay.update_equity(1_000_000.0, day1)
    baseline = overlay.persisted_state.starting_daily_nlv
    assert baseline == pytest.approx(1_000_000.0)
    # 20 same-day intraday bars never reset the daily baseline.
    for index in range(1, 21):
        overlay.update_equity(1_000_000.0 - index * 100.0, day1 + timedelta(minutes=5 * index))
    assert overlay.persisted_state.starting_daily_nlv == pytest.approx(baseline)
    assert overlay.daily_loss_latched is False
    # A 3% same-day loss latches the daily breaker and flattens the book.
    overlay.update_equity(960_000.0, day1 + timedelta(hours=2))
    assert overlay.daily_loss_latched is True
    latched_state = dataclasses.replace(
        _state_with_returns("5m"),
        timestamp=day1 + timedelta(hours=2),
        equity=960_000.0,
    )
    flat = overlay.apply_risk(
        make_weight_vector({"ES": 0.5, "NQ": 0.5}),
        latched_state,
    )
    assert flat.is_flat()
    # First bar of the next UTC day re-bases the baseline and clears the latch.
    day2 = datetime(2026, 1, 6, 0, 5, tzinfo=UTC)
    overlay.update_equity(960_000.0, day2)
    assert overlay.daily_loss_latched is False
    assert overlay.persisted_state.starting_daily_nlv == pytest.approx(960_000.0)
    restored = overlay.apply_risk(
        make_weight_vector({"ES": 0.5, "NQ": 0.5}),
        dataclasses.replace(_state_with_returns("5m"), timestamp=day2, equity=960_000.0),
    )
    assert not restored.is_flat(), "the next UTC day must allow trading again"
    # Portfolio HWM survives the roll; a latched kill switch also survives.
    assert overlay.persisted_state.portfolio_high_water_mark == pytest.approx(1_000_000.0)
    overlay.engage_kill_switch("operator")
    overlay.update_equity(960_000.0, datetime(2026, 1, 7, 0, 5, tzinfo=UTC))
    kill_state = dataclasses.replace(
        _state_with_returns("5m"),
        timestamp=datetime(2026, 1, 7, 0, 5, tzinfo=UTC),
        equity=960_000.0,
    )
    assert overlay.apply_risk(make_weight_vector({"ES": 0.5}), kill_state).is_flat(), (
        "operator kill switch must survive a session roll"
    )


def test_margin_settlement_utc_day_helpers() -> None:
    first = datetime(2026, 1, 5, 10, 0, tzinfo=UTC)
    same_day = datetime(2026, 1, 5, 15, 0, tzinfo=UTC)
    next_day = datetime(2026, 1, 6, 0, 30, tzinfo=UTC)
    assert utc_calendar_day(first) == date(2026, 1, 5)
    assert is_new_utc_day(first, same_day) is False
    assert is_new_utc_day(same_day, next_day) is True
    assert is_new_utc_day(None, first) is True
    clock = DailySettlementClock()
    assert clock.due(first) is True
    clock.mark(first)
    assert clock.due(same_day) is False
    assert clock.settlements == 1
    assert clock.due(next_day) is True
    clock.reset()
    assert clock.last_settled_day is None and clock.settlements == 0


class _CountingMarginSimulator(MarginSimulator):
    def __init__(self, instruments: InstrumentsConfig) -> None:
        super().__init__(instruments)
        self.settle_timestamps: list[datetime] = []

    def settle(
        self,
        timestamp: datetime,
        *,
        cash: float,
        positions: Any,
        settlement_prices: Any = None,
    ) -> Any:
        self.settle_timestamps.append(timestamp)
        return super().settle(
            timestamp,
            cash=cash,
            positions=positions,
            settlement_prices=settlement_prices,
        )


class _ConstantPipeline:
    name = "constant-weights"

    def __init__(self, weights: dict[str, float]) -> None:
        self._weights = WeightVector(dict(weights))

    def run(self, state: MarketState) -> PipelineResult:
        return PipelineResult(
            timestamp=state.timestamp,
            universe=state.ids,
            base_weights=self._weights,
            timing_weights=self._weights,
            risk_weights=self._weights,
            final_weights=self._weights,
            targets=make_contract_targets({"ES": 2}),
        )


def test_event_backtester_settles_once_per_utc_day_intraday(
    master: InstrumentsConfig,
) -> None:
    timestamps = [
        BASE,
        BASE + timedelta(hours=1),
        BASE + timedelta(hours=2),
        BASE + timedelta(days=1),
        BASE + timedelta(days=1, hours=1),
    ]
    states = [
        make_market_state(("ES",), timestamp=stamp, equity=1_000_000.0) for stamp in timestamps
    ]
    spy = _CountingMarginSimulator(master)
    engine = EventBacktester(
        _ConstantPipeline({"ES": 0.30}),
        SimpleContractTranslator(master),
        instrument_master=master,
        initial_equity=1_000_000.0,
        margin_simulator=spy,
        timeframe="1h",
    )
    result = engine.run(states=states)
    # Same-day intraday bars settle at most once; the next UTC day settles again.
    days = [stamp.date() for stamp in spy.settle_timestamps]
    assert len(days) == len(set(days)), "double settlement on the same UTC day"
    assert all(stamp.date() <= max(days) for stamp in spy.settle_timestamps)
    result.check_accounting_identity()  # type: ignore[attr-defined]
    # Daily bars keep the legacy per-bar settlement (1 bar == 1 session).
    daily_states = [
        make_market_state(("ES",), timestamp=BASE + timedelta(days=offset), equity=1_000_000.0)
        for offset in range(3)
    ]
    daily_spy = _CountingMarginSimulator(master)
    daily_engine = EventBacktester(
        _ConstantPipeline({"ES": 0.30}),
        SimpleContractTranslator(master),
        instrument_master=master,
        initial_equity=1_000_000.0,
        margin_simulator=daily_spy,
        timeframe="1d",
    )
    daily_engine.run(states=daily_states)
    per_day = Counter(stamp.date() for stamp in daily_spy.settle_timestamps)
    assert any(count >= 2 for count in per_day.values())


# ---------------------------------------------------------------------------
# 3. IBKR backfill / data API timeframe + exact error contract (task 68)
# ---------------------------------------------------------------------------


class _FakeDatabentoLoader:
    """DataService loader double: metadata estimate + base-timeframe ingest."""

    def __init__(self, *, cost: float = 5.0, billable: int = 2_000_000) -> None:
        self.cost = cost
        self.billable = billable
        self.estimate_calls: list[str] = []
        self.ingest_calls: list[dict[str, object]] = []

    def estimate_cost_usd(self, roots: Any, start: Any, end: Any, schema: str) -> Any:
        from src.data.databento_loader import CostEstimate

        self.estimate_calls.append(schema)
        return CostEstimate(
            roots=tuple(str(root) for root in roots),
            schema=schema,
            start=start,
            end=end,
            estimated_cost_usd=self.cost,
            billable_size_bytes=self.billable,
            billable_size_megabytes=self.billable / 1_000_000.0,
        )

    def ingest(
        self,
        roots: Any,
        start: Any,
        end: Any,
        schema: str,
        *,
        allow_over_budget: bool = False,
    ) -> pl.DataFrame:
        self.ingest_calls.append(
            {"roots": list(roots), "schema": schema, "allow_over_budget": allow_over_budget}
        )
        return _minute_bars(roots=tuple(str(root) for root in roots), periods=60, start=start)


class _FakeIBKRFetcher:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.fetch_calls: list[tuple[list[str], datetime, datetime]] = []

    def fetch_bars_sync(self, roots: Any, start: datetime, end: datetime) -> pl.DataFrame:
        self.fetch_calls.append((list(roots), start, end))
        if self.fail:
            raise RuntimeError("connection refused")
        return _minute_bars(roots=tuple(str(root) for root in roots), periods=60, start=start)


@contextmanager
def _data_app(
    tmp_path: Path,
    *,
    loader: _FakeDatabentoLoader | None = None,
    fetcher: _FakeIBKRFetcher | None = None,
    ibkr_available: bool = True,
    seed: bool = False,
) -> Any:
    from src.api.data_service import DataService
    from src.api.main import create_app as _create_app

    settings = _settings(tmp_path)
    if seed:
        repository = ParquetRepository(settings)
        repository.write_bars(_minute_bars(roots=("ES",), periods=30))
    app = _create_app(settings, clock=lambda: FIXED_CLOCK)
    with TestClient(app) as client:
        service = DataService(
            settings=settings,
            repository=ParquetRepository(settings),
            instrument_master=load_instruments(INSTRUMENTS_PATH),
            clock=lambda: FIXED_CLOCK,
            databento_loader_provider=(lambda: loader) if loader is not None else None,
            ibkr_fetcher_provider=(lambda timeframe="1d": fetcher) if fetcher is not None else None,
            ibkr_available_provider=lambda: ibkr_available,
        )
        app.state.data_service = service
        yield client, service, settings


def test_api_inventory_timeframe_fields(tmp_path: Path) -> None:
    with _data_app(tmp_path, seed=True) as (client, _service, _settings):
        response = client.get("/api/data/inventory?timeframe=5m")
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["timeframe"] == "5m"
        assert body["schema"] == "ohlcv-5m"
        assert body["base_schema"] == "ohlcv-1m"
        assert body["roots"], "inventory must still enumerate every master root"
        # 1m base cache is not counted as 5m coverage.
        assert body["totals"]["cached_roots"] == 0
        # Bad timeframe on inventory must be an actionable 422 (not a 500).
        bad = client.get("/api/data/inventory?timeframe=2h")
        assert bad.status_code == 422, bad.text


def test_api_estimate_cost_timeframe_and_errors(tmp_path: Path) -> None:
    loader = _FakeDatabentoLoader(cost=3.5)
    with _data_app(tmp_path, loader=loader) as (client, _service, _settings):
        ibkr = client.post(
            "/api/data/estimate-cost",
            json={
                "roots": ["ES"],
                "start": "2026-01-01",
                "end": "2026-01-10",
                "source": "ibkr",
                "timeframe": "5m",
            },
        )
        assert ibkr.status_code == 200, ibkr.text
        assert ibkr.json()["estimated_cost_usd"] == 0.0
        assert ibkr.json()["source"] == "ibkr"

        databento = client.post(
            "/api/data/estimate-cost",
            json={
                "roots": ["ES", "NQ"],
                "start": "2026-01-01",
                "end": "2026-01-10",
                "source": "databento",
                "timeframe": "5m",
            },
        )
        assert databento.status_code == 200, databento.text
        body = databento.json()
        assert body["timeframe"] == "5m"
        assert body["schema"] == "ohlcv-5m"
        assert body["base_schema"] == "ohlcv-1m"
        assert body["estimated_cost_usd"] == pytest.approx(3.5)
        assert loader.estimate_calls == ["ohlcv-1m"], "estimate uses the base Databento schema"

        unknown = client.post(
            "/api/data/estimate-cost",
            json={
                "roots": ["NOPE"],
                "start": "2026-01-01",
                "end": "2026-01-10",
                "source": "ibkr",
            },
        )
        assert unknown.status_code == 422
        bad_timeframe = client.post(
            "/api/data/estimate-cost",
            json={
                "roots": ["ES"],
                "start": "2026-01-01",
                "end": "2026-01-10",
                "source": "ibkr",
                "timeframe": "2h",
            },
        )
        assert bad_timeframe.status_code == 422

    # No key + no provider: Databento estimate must be a structured 400.
    with _data_app(tmp_path / "nokey") as (client, _service, _settings):
        response = client.post(
            "/api/data/estimate-cost",
            json={
                "roots": ["ES"],
                "start": "2026-01-01",
                "end": "2026-01-10",
                "source": "databento",
            },
        )
        assert response.status_code == 400, response.text
        detail = response.json()["detail"]
        assert detail["error"] == "databento_api_key_missing"
        assert "FUTURESRLX_DATABENTO_API_KEY" in detail["message"]


def test_api_ingest_databento_budget_402_and_timeframe_resample(tmp_path: Path) -> None:
    loader = _FakeDatabentoLoader(cost=5.0)
    with _data_app(tmp_path, loader=loader) as (client, _service, _settings):
        refused = client.post(
            "/api/data/ingest",
            json={
                "source": "databento",
                "roots": ["ES", "NQ"],
                "start": "2026-01-01",
                "end": "2026-01-10",
                "timeframe": "5m",
            },
        )
        assert refused.status_code == 402, refused.text
        detail = refused.json()["detail"]
        assert detail["error"] == "databento_budget_exceeded"
        assert detail["message"] == (
            "Databento cost guard exceeded ($5.00 > $2.00 cap); set "
            "allow_over_budget=true and confirm=true to override, or raise "
            "FUTURESRLX_DATABENTO_MAX_COST_USD_PER_JOB"
        )
        assert detail["estimated_cost_usd"] == pytest.approx(5.0)
        assert detail["max_cost_usd"] == pytest.approx(2.0)
        assert detail["override_allowed"] is True
        assert loader.ingest_calls == [], "refused Databento job must never download"

        allowed = client.post(
            "/api/data/ingest",
            json={
                "source": "databento",
                "roots": ["ES", "NQ"],
                "start": "2026-01-01",
                "end": "2026-01-10",
                "timeframe": "5m",
                "allow_over_budget": True,
                "confirm": True,
            },
        )
        assert allowed.status_code == 200, allowed.text
        body = allowed.json()
        assert body["timeframe"] == "5m"
        assert body["base_schema"] == "ohlcv-1m"
        assert body["resampled"] is True
        assert body["bars_written"] > 0
        assert body["cost_estimate"]["estimated_cost_usd"] == pytest.approx(5.0)
        assert loader.ingest_calls and loader.ingest_calls[0]["schema"] == "ohlcv-1m"
        repository = _service.repository
        assert repository.available_roots("ohlcv-5m") == ("ES", "NQ")
        by_root = {row.root: row for row in repository.inventory("ohlcv-5m")}
        assert by_root["ES"].source == "databento"
        assert by_root["ES"].cached_bars > 0


def test_api_ingest_ibkr_unreachable_503_and_success(tmp_path: Path) -> None:
    fetcher = _FakeIBKRFetcher()
    with _data_app(tmp_path / "down", fetcher=fetcher, ibkr_available=False) as (
        client,
        _service,
        settings,
    ):
        refused = client.post(
            "/api/data/ingest",
            json={
                "source": "ibkr",
                "roots": ["ES"],
                "start": "2026-01-01",
                "end": "2026-01-10",
                "timeframe": "5m",
            },
        )
        assert refused.status_code == 503, refused.text
        detail = refused.json()["detail"]
        assert detail["error"] == "ibkr_unreachable"
        assert f"{settings.ibkr_host}:{settings.ibkr_port}" in detail["message"]
        assert "start TWS or IB Gateway" in detail["message"]
        assert fetcher.fetch_calls == [], "unreachable IBKR must not attempt a fetch"

    with _data_app(
        tmp_path / "fetchfail", fetcher=_FakeIBKRFetcher(fail=True), ibkr_available=True
    ) as (client, _service, _settings):
        failed = client.post(
            "/api/data/ingest",
            json={
                "source": "ibkr",
                "roots": ["ES"],
                "start": "2026-01-01",
                "end": "2026-01-10",
                "timeframe": "5m",
            },
        )
        assert failed.status_code == 503, failed.text
        assert failed.json()["detail"]["error"] == "ibkr_unreachable"

    fetcher_ok = _FakeIBKRFetcher()
    with _data_app(tmp_path / "up", fetcher=fetcher_ok, ibkr_available=True) as (
        client,
        service,
        _settings,
    ):
        response = client.post(
            "/api/data/ingest",
            json={
                "source": "ibkr",
                "roots": ["ES", "NQ"],
                "start": "2026-01-01",
                "end": "2026-01-10",
                "timeframe": "5m",
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["source"] == "ibkr"
        assert body["timeframe"] == "5m"
        assert body["base_schema"] == "ohlcv-1m"
        assert body["resampled"] is True
        assert body["bars_written"] > 0
        assert body["cost_estimate"] is None
        assert fetcher_ok.fetch_calls
        by_root = {row.root: row for row in service.repository.inventory("ohlcv-5m")}
        assert by_root["ES"].source == "ibkr"
        assert by_root["NQ"].source == "ibkr"


# ---------------------------------------------------------------------------
# 4. Timeframe propagation into backtests / Optuna (tasks 68/70)
# ---------------------------------------------------------------------------


def _strategy_with_roots(tmp_path: Path) -> Path:
    configs = tmp_path / "configs"
    strategies = configs / "strategies"
    strategies.mkdir(parents=True, exist_ok=True)
    text = (CONFIGS_DIR / "strategies" / "tsmom_vol_target.yaml").read_text(encoding="utf-8")
    text = text.replace(
        "    min_history_days: 252\n",
        "    min_history_days: 252\n    roots: ES,NQ,CL\n",
        1,
    )
    (strategies / "tsmom_vol_target.yaml").write_text(text, encoding="utf-8")
    return configs


def _app_settings(tmp_path: Path) -> Settings:
    return _settings(tmp_path).model_copy(update={"configs_dir": _strategy_with_roots(tmp_path)})


def _seed_daily_cache(settings: Settings) -> None:
    repository = ParquetRepository(settings)
    repository.write_bars(
        make_bars_frame(
            roots=("ES", "NQ", "CL"),
            periods=400,
            start=FIXED_CLOCK - timedelta(days=600),
            instrument_master=load_instruments(INSTRUMENTS_PATH),
        )
    )


def _seed_5m_cache(settings: Settings) -> None:
    repository = ParquetRepository(settings)
    minute = _minute_bars(
        roots=("ES", "NQ", "CL"), periods=1_200, start=FIXED_CLOCK - timedelta(days=2)
    )
    repository.write_timeframe("5m", resample_bars(minute, "5m"))


@contextmanager
def _app(tmp_path: Path, *, seed: str | None = None) -> Any:
    from src.api.main import create_app as _create_app

    settings = _app_settings(tmp_path)
    if seed == "daily":
        _seed_daily_cache(settings)
    elif seed == "5m":
        _seed_5m_cache(settings)
    app = _create_app(settings, clock=lambda: FIXED_CLOCK)
    with TestClient(app) as client:
        yield client, settings


def _backtest_payload(
    data_mode: str, timeframe: str, *, start: date, end: date
) -> dict[str, object]:
    return {
        "strategy_id": "tsmom_vol_target",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "initial_equity": 1_000_000.0,
        "walk_forward": False,
        "data_mode": data_mode,
        "timeframe": timeframe,
    }


def test_backtest_cached_real_per_timeframe(tmp_path: Path) -> None:
    daily_start = (FIXED_CLOCK - timedelta(days=300)).date()
    daily_end = (FIXED_CLOCK - timedelta(days=30)).date()
    with _app(tmp_path / "daily", seed="daily") as (client, _settings):
        response = client.post(
            "/api/backtests",
            json=_backtest_payload("cached_real", "1d", start=daily_start, end=daily_end),
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["timeframe"] == "1d"
        assert body["data_origin"] == "real_parquet"
        assert body["cached_bars"]["timeframe"] == "1d"
        assert body["cached_bars"]["schema"] == "ohlcv-1d"

    intraday_start = (FIXED_CLOCK - timedelta(days=3)).date()
    intraday_end = (FIXED_CLOCK - timedelta(days=1)).date()
    with _app(tmp_path / "5m", seed="5m") as (client, _settings):
        response = client.post(
            "/api/backtests",
            json=_backtest_payload("cached_real", "5m", start=intraday_start, end=intraday_end),
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["timeframe"] == "5m"
        assert body["data_origin"] == "real_parquet"
        assert body["cached_bars"]["timeframe"] == "5m"
        assert body["cached_bars"]["schema"] == "ohlcv-5m"
        assert body["cached_bars"]["bar_count"] >= 200
        # List/detail expose the same timeframe provenance immediately.
        listed = {row["id"]: row for row in client.get("/api/backtests").json()["runs"]}
        assert listed[body["id"]]["timeframe"] == "5m"
        detail = client.get(f"/api/backtests/{body['id']}").json()
        assert detail["cached_bars"]["timeframe"] == "5m"


def test_backtest_auto_resamples_finer_cache_for_timeframe(tmp_path: Path) -> None:
    start = (FIXED_CLOCK - timedelta(days=3)).date()
    end = (FIXED_CLOCK - timedelta(days=1)).date()
    with _app(tmp_path) as (client, settings):
        # Only the 1m base cache exists; auto/5m must auto-resample it.
        repository = ParquetRepository(settings)
        repository.write_timeframe(
            "1m",
            _minute_bars(
                roots=("ES", "NQ", "CL"), periods=1_200, start=FIXED_CLOCK - timedelta(days=2)
            ),
        )
        response = client.post(
            "/api/backtests",
            json=_backtest_payload("auto", "5m", start=start, end=end),
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["data_origin"] == "real_parquet"
        assert body["timeframe"] == "5m"
        assert body["cached_bars"]["timeframe"] == "5m"
        assert body["cached_bars"]["schema"] == "ohlcv-5m"


def test_backtest_synthetic_timeframe_annualization_propagates(tmp_path: Path) -> None:
    start = (FIXED_CLOCK - timedelta(days=300)).date()
    end = (FIXED_CLOCK - timedelta(days=30)).date()
    with _app(tmp_path) as (client, _settings):
        daily = client.post(
            "/api/backtests",
            json=_backtest_payload("synthetic_offline", "1d", start=start, end=end),
        ).json()
        intraday = client.post(
            "/api/backtests",
            json=_backtest_payload("synthetic_offline", "5m", start=start, end=end),
        ).json()
        assert daily["timeframe"] == "1d" and intraday["timeframe"] == "5m"
        assert daily["data_origin"] == "synthetic_fixture"
        assert intraday["data_origin"] == "synthetic_fixture"
        assert daily["cached_bars"] is None and intraday["cached_bars"] is None
        # The timeframe reaches the pipeline/risk path: the 5m states are built with
        # intraday-annualised covariance, so the risk overlay sizes a different book
        # and the resulting metric set differs from the daily run (both finite).
        assert math.isfinite(float(intraday["metrics"]["annual_volatility"]))
        assert intraday["metrics"] != daily["metrics"], (
            "request timeframe must propagate into the states/risk decisions"
        )
        assert intraday["metrics"]["annual_volatility"] != pytest.approx(
            daily["metrics"]["annual_volatility"]
        )


def test_backtest_cached_real_5m_empty_is_422(tmp_path: Path) -> None:
    start = (FIXED_CLOCK - timedelta(days=3)).date()
    end = (FIXED_CLOCK - timedelta(days=1)).date()
    with _app(tmp_path) as (client, _settings):
        response = client.post(
            "/api/backtests",
            json=_backtest_payload("cached_real", "5m", start=start, end=end),
        )
        assert response.status_code == 422, response.text
        assert "no sufficient cached real" in response.json()["detail"]


def test_optimize_timeframe_metadata_and_errors(tmp_path: Path) -> None:
    with _app(tmp_path / "daily", seed="daily") as (client, _settings):
        response = client.post(
            "/api/optimize",
            json={
                "base_strategy_id": "tsmom_vol_target",
                "n_trials": 2,
                "wf_splits": 3,
                "data_mode": "cached_real",
                "timeframe": "1d",
            },
        )
        assert response.status_code == 200, response.text
        study = response.json()
        assert study["timeframe"] == "1d"
        assert study["data_origin"] == "real_parquet"
        assert study["cached_bars"]["timeframe"] == "1d"
        listed = {row["study_id"]: row for row in client.get("/api/optimize").json()}
        assert listed[study["study_id"]]["timeframe"] == "1d"

    with _app(tmp_path / "synthetic") as (client, _settings):
        synthetic = client.post(
            "/api/optimize",
            json={
                "base_strategy_id": "tsmom_vol_target",
                "n_trials": 2,
                "wf_splits": 3,
                "data_mode": "synthetic_offline",
                "timeframe": "5m",
            },
        )
        assert synthetic.status_code == 200, synthetic.text
        body = synthetic.json()
        assert body["timeframe"] == "5m"
        assert body["data_origin"] == "synthetic_fixture"
        assert body["cached_bars"] is None

        missing = client.post(
            "/api/optimize",
            json={
                "base_strategy_id": "tsmom_vol_target",
                "n_trials": 2,
                "wf_splits": 3,
                "data_mode": "cached_real",
                "timeframe": "5m",
            },
        )
        assert missing.status_code == 422, missing.text
        assert "no sufficient cached real" in missing.json()["detail"]


# ---------------------------------------------------------------------------
# 5. EngineMarketService: live NLV binding + cached Parquet first (task 71)
# ---------------------------------------------------------------------------


class _NLExecution:
    """Fake execution adapter exposing a live account snapshots and NLV."""

    def __init__(self, net_liquidation: float = 1_000_000.0) -> None:
        self._nlv = net_liquidation
        self.ibkr_enabled = True
        self.connected = True

    async def account(self) -> Any:
        from src.api.schemas import AccountSnapshotResponse

        return AccountSnapshotResponse(
            timestamp=FIXED_CLOCK,
            net_liquidation=self._nlv,
            cash_balance=self._nlv,
            buying_power=self._nlv,
            initial_margin_used=50_000.0,
            maintenance_margin_used=45_000.0,
            excess_liquidity=self._nlv - 50_000.0,
            margin_utilization=0.05,
            day_pnl=0.0,
            day_pnl_pct=0.0,
            gross_leverage=0.0,
            net_exposure=0.0,
            source="engine",
        )

    async def positions(self) -> dict[str, Any]:
        return {}


def _offline_market(master: InstrumentsConfig) -> Any:
    from src.api import fixtures as api_fixtures

    return api_fixtures.build_offline_market(master, clock=lambda: FIXED_CLOCK)


def test_engine_market_live_nlv_matches_positions_view(tmp_path: Path) -> None:
    from src.api.live import LiveStateService
    from src.api.main import create_app as _create_app
    from src.api.services import EngineMarketService, build_container

    settings = _settings(tmp_path)
    master = load_instruments(INSTRUMENTS_PATH)
    container = build_container(settings, clock=lambda: FIXED_CLOCK)
    execution = _NLExecution(1_000_000.0)
    live = LiveStateService(
        settings=settings,
        clock=lambda: FIXED_CLOCK,
        execution=execution,
        instrument_master=master,
        market=container.market,
    )
    engine = EngineMarketService(
        market=_offline_market(master),
        settings=settings,
        catalog=container.strategies,
        clock=lambda: FIXED_CLOCK,
        equity_provider=lambda: 1_000_000.0,
    )
    container = dataclasses.replace(container, execution=execution, live=live, market=engine)
    app = _create_app(settings, clock=lambda: FIXED_CLOCK, container=container)
    with TestClient(app) as client:
        positions = client.get("/api/positions")
        assert positions.status_code == 200, positions.text
        nlv = positions.json()["net_liquidation"]
        assert nlv == pytest.approx(1_000_000.0)
        assert engine.latest_state().equity == pytest.approx(nlv)
        assert engine.equity_source == "live-account"
        assert engine.data_backing["equity_source"] == "live-account"
        pipeline = client.get("/api/pipeline/latest")
        assert pipeline.status_code == 200, pipeline.text
        body = pipeline.json()
        assert body["data_source"] == "engine"
        messages = " ".join(
            f"{entry.get('detail', '')} {entry.get('value', '')}" for entry in body["diagnostics"]
        )
        assert "1,000,000" in messages
        assert "live-account" in messages
        assert "4,000,000" not in messages, "the $4M fixture equity must not leak when live"
        # Every published notional is valued on the same ~$1M book (never $4M).
        assert all(
            abs(float(row.get("notional") or 0.0)) < 4_000_000.0 for row in body["instruments"]
        )


def test_engine_market_fixture_fallback_without_provider(tmp_path: Path) -> None:
    from src.api.services import INITIAL_EQUITY, EngineMarketService, build_container

    settings = _settings(tmp_path)
    master = load_instruments(INSTRUMENTS_PATH)
    container = build_container(settings, clock=lambda: FIXED_CLOCK)
    engine = EngineMarketService(
        market=_offline_market(master),
        settings=settings,
        catalog=container.strategies,
        clock=lambda: FIXED_CLOCK,
    )
    assert engine.equity_source == "fixture"
    assert engine.latest_state().equity == pytest.approx(INITIAL_EQUITY)
    assert engine.data_origin == "fixture"
    response = engine.pipeline()
    assert response.data_source == "engine"
    messages = " ".join(f"{entry.detail} {entry.value}" for entry in response.diagnostics)
    assert "4,000,000" in messages or "fixture" in messages.lower()


def test_engine_market_cached_parquet_first_and_term_structure(tmp_path: Path) -> None:
    from src.api.main import create_app as _create_app
    from src.api.services import EngineMarketService, build_container

    settings = _settings(tmp_path)
    master = load_instruments(INSTRUMENTS_PATH)
    repository = ParquetRepository(settings)
    cached = make_bars_frame(
        roots=("ES", "NQ"),
        periods=300,
        start=FIXED_CLOCK - timedelta(days=400),
        instrument_master=master,
    )
    repository.write_bars(cached)
    container = build_container(settings, clock=lambda: FIXED_CLOCK)
    engine = EngineMarketService(
        market=_offline_market(master),
        settings=settings,
        catalog=container.strategies,
        clock=lambda: FIXED_CLOCK,
    )
    assert engine.data_origin == "cached_parquet"
    backing = engine.data_backing
    assert set(backing["cached_roots"]) == {"ES", "NQ"}
    assert backing["partial"] is True, "uncovered master roots must be attributed as fallback"
    latest_timestamp = cached.get_column(s.TIMESTAMP_COLUMN).max()
    state = engine.latest_state()
    assert state.timestamp == latest_timestamp
    term = engine.term_structure()
    es_curve = next(curve for curve in term.curves if curve.root == "ES")
    last_close_es = float(
        cached.filter(pl.col(s.ROOT_SYMBOL_COLUMN) == "ES")
        .sort(s.TIMESTAMP_COLUMN)
        .get_column(s.CLOSE_COLUMN)[-1]
    )
    assert es_curve.contracts[0].price == pytest.approx(last_close_es, rel=1e-2)
    # API term-structure publishes the additive backing headers; pipeline is engine-backed.
    container = dataclasses.replace(container, market=engine)
    app = _create_app(settings, clock=lambda: FIXED_CLOCK, container=container)
    with TestClient(app) as client:
        response = client.get("/api/term-structure")
        assert response.status_code == 200, response.text
        assert response.headers["X-Data-Origin"] == "cached_parquet"
        assert response.headers["X-Data-Cached-Roots"] == "2"
        pipeline = client.get("/api/pipeline/latest")
        assert pipeline.status_code == 200
        assert pipeline.json()["data_source"] == "engine"


def test_engine_market_partial_cache_and_no_cache_fallback(tmp_path: Path) -> None:
    from src.api.services import EngineMarketService, build_container

    master = load_instruments(INSTRUMENTS_PATH)
    # Only ES cached: cached origin with explicit fallback attribution.
    partial_settings = _settings(tmp_path / "partial")
    ParquetRepository(partial_settings).write_bars(
        make_bars_frame(
            roots=("ES",),
            periods=250,
            start=FIXED_CLOCK - timedelta(days=300),
            instrument_master=master,
        )
    )
    partial_engine = EngineMarketService(
        market=_offline_market(master),
        settings=partial_settings,
        catalog=build_container(partial_settings, clock=lambda: FIXED_CLOCK).strategies,
        clock=lambda: FIXED_CLOCK,
    )
    assert partial_engine.data_origin == "cached_parquet"
    partial = partial_engine.data_backing
    assert "ES" in partial["cached_roots"]
    assert partial["fallback_roots"], "missing roots must be explicitly attributed"
    # No cache at all: deterministic fixture backing.
    empty_engine = EngineMarketService(
        market=_offline_market(master),
        settings=_settings(tmp_path / "empty"),
        catalog=build_container(
            _settings(tmp_path / "empty"), clock=lambda: FIXED_CLOCK
        ).strategies,
        clock=lambda: FIXED_CLOCK,
    )
    assert empty_engine.data_origin == "fixture"
    assert empty_engine.data_backing["cached_roots"] == []


# ---------------------------------------------------------------------------
# 6. Frontend contract sanity (task 72 consumes additive fields)
# ---------------------------------------------------------------------------


def test_frontend_contract_fields_present(tmp_path: Path) -> None:
    with _app(tmp_path / "synthetic") as (client, _settings):
        start = (FIXED_CLOCK - timedelta(days=300)).date()
        end = (FIXED_CLOCK - timedelta(days=30)).date()
        run = client.post(
            "/api/backtests",
            json=_backtest_payload("synthetic_offline", "5m", start=start, end=end),
        ).json()
        assert set(run) >= {"timeframe", "data_origin", "data_mode", "cached_bars"}
        assert isinstance(run["timeframe"], str)
        listed = client.get("/api/backtests").json()["runs"]
        assert any(row["timeframe"] == "5m" for row in listed)

    with _data_app(tmp_path / "data") as (client, _service, _settings):
        inventory = client.get("/api/data/inventory?timeframe=1h").json()
        assert inventory["timeframe"] == "1h"
        assert inventory["base_schema"] == "ohlcv-1h"
        assert inventory["schema"] == "ohlcv-1h"
        estimate = client.post(
            "/api/data/estimate-cost",
            json={
                "roots": ["ES"],
                "start": "2026-01-01",
                "end": "2026-01-10",
                "source": "ibkr",
                "timeframe": "1w",
            },
        ).json()
        assert estimate["timeframe"] == "1w"
        assert estimate["base_schema"] == "ohlcv-1d"
        assert estimate["schema"] == "ohlcv-1w"
