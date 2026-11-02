"""Phase-8 tests: multi-source hydration fallback, IBKR history, offline runner, lazy imports.

Everything here is offline and deterministic: vendor clients/frames are injected
fakes, the CLI subprocesses run against temporary data directories with every
``FUTURESRLX_*`` variable stripped, and the offline simulation window is a fixed
constant (:data:`src.execution.runner.OFFLINE_SIMULATION_END`).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import textwrap
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest
from configs.instruments import InstrumentsConfig
from configs.settings import Settings
from src.backtest.simulated_broker import SimulatedBroker
from src.core.types import ContractTargetSet, MarketState, OrderStatus
from src.data import schema as s
from src.data.ibkr_data import HISTORICAL_MAX_REQUEST_DAYS, IBKRHistoricalBarFetcher
from src.data.storage import ParquetRepository
from src.execution.hydrator import (
    HYDRATION_SOURCE_HINTS,
    HYDRATION_SOURCES,
    MIN_REQUIRED_BARS,
    ColdStartHydrator,
    HydrationError,
    MultiSourceBackfill,
    add_neutral_term_structure,
    build_backfill_chain,
    offline_backfill,
)
from src.execution.runner import (
    OFFLINE_DEFAULT_ROOTS,
    OFFLINE_INITIAL_CASH,
    OFFLINE_SIMULATION_END,
    OfflineMarketStateStreamer,
    OfflineTargetBroker,
    _run_offline,
    build_hydration_chain,
    describe_hydration_sources,
)

from tests.fixtures.market import (
    make_bars_frame,
    make_contract_targets,
    make_market_state,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
START = datetime(2024, 1, 2, 21, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Deterministic fakes / helpers
# ---------------------------------------------------------------------------


class FakeBackfill:
    """Records backfill calls; returns a frame for the requested roots."""

    def __init__(
        self,
        name: str,
        *,
        only: tuple[str, ...] | None = None,
        error: BaseException | None = None,
        frame: pl.DataFrame | None = None,
        empty: bool = False,
    ) -> None:
        self.name = name
        self.only = only
        self.error = error
        self.frame = frame
        self.empty = empty
        self.calls: list[tuple[tuple[str, ...], datetime, datetime]] = []

    @property
    def called(self) -> bool:
        return bool(self.calls)

    def __call__(self, roots: list[str], start: datetime, end: datetime) -> pl.DataFrame:
        self.calls.append((tuple(roots), start, end))
        if self.error is not None:
            raise self.error
        if self.frame is not None:
            return self.frame
        if self.empty:
            return s.empty_bars_frame()
        selected = tuple(root for root in roots if self.only is None or root in self.only)
        if not selected:
            return s.empty_bars_frame()
        return make_bars_frame(roots=selected, periods=3)


def settings_with(tmp_path: Path, **overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "environment": "backtest",
        "data_dir": tmp_path / "data",
        "parquet_dir": tmp_path / "data" / "parquet",
        "duckdb_path": tmp_path / "data" / "test.duckdb",
        "report_dir": tmp_path / "artifacts",
        "instruments_path": REPO_ROOT / "configs" / "instruments.yaml",
    }
    base.update(overrides)
    return Settings(**base)


def _bar(timestamp: datetime, price: float, **overrides: Any) -> SimpleNamespace:
    values: dict[str, Any] = {
        "date": timestamp,
        "open": price,
        "high": price * 1.01,
        "low": price * 0.99,
        "close": price * 1.005,
        "volume": 1_500.0,
        "average": price,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class FakeHistoricalClient:
    """In-memory ``ib_async.IB`` historical-data double (no network)."""

    def __init__(
        self,
        bars_by_root: dict[str, list[SimpleNamespace]] | None = None,
        *,
        connect_error: BaseException | None = None,
        fail_roots: tuple[str, ...] = (),
    ) -> None:
        self.bars_by_root = bars_by_root or {}
        self.connect_error = connect_error
        self.fail_roots = fail_roots
        self.connected = False
        self.connect_calls: list[tuple[str, int, int, float]] = []
        self.disconnect_calls = 0
        self.requests: list[tuple[str, datetime, str, str, int]] = []
        self.qualified = 0

    def isConnected(self) -> bool:  # noqa: N802 - mirrors ib_async
        return self.connected

    async def connectAsync(  # noqa: N802 - mirrors ib_async
        self, host: str, port: int, client_id: int, timeout: float
    ) -> None:
        self.connect_calls.append((host, port, client_id, timeout))
        if self.connect_error is not None:
            raise self.connect_error
        self.connected = True

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self.connected = False

    async def qualifyContractsAsync(self, contract: object) -> list[object]:  # noqa: N802
        self.qualified += 1
        return [contract]

    async def reqHistoricalDataAsync(  # noqa: N802 - mirrors ib_async
        self,
        contract: object,
        end: datetime,
        duration: str,
        bar_size: str,
        what_to_show: str,
        use_rth: bool,
        format_date: int,
        keep_up_to_date: bool,
    ) -> list[SimpleNamespace]:
        root = str(getattr(contract, "symbol", ""))
        self.requests.append((root, end, duration, bar_size, format_date))
        if root in self.fail_roots:
            return []
        return list(self.bars_by_root.get(root, []))


def make_ibkr_fetcher(
    settings: Settings,
    client: FakeHistoricalClient,
) -> IBKRHistoricalBarFetcher:
    return IBKRHistoricalBarFetcher(
        settings,
        ib_factory=lambda: client,
        contract_factory=lambda root: SimpleNamespace(symbol=root),
    )


def cli_environment(tmp_path: Path) -> dict[str, str]:
    """Hermetic subprocess environment pointing every writable path at ``tmp_path``."""
    env = {key: value for key, value in os.environ.items() if not key.startswith("FUTURESRLX_")}
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["FUTURESRLX_DATA_DIR"] = str(tmp_path / "data")
    env["FUTURESRLX_PARQUET_DIR"] = str(tmp_path / "data" / "parquet")
    env["FUTURESRLX_DUCKDB_PATH"] = str(tmp_path / "data" / "runner.duckdb")
    env["FUTURESRLX_REPORT_DIR"] = str(tmp_path / "artifacts")
    return env


def run_cli(
    *args: str,
    env: dict[str, str],
    timeout: float = 240.0,
) -> subprocess.CompletedProcess[str]:
    """Run ``python -m src.execution.runner`` with ``-W error::RuntimeWarning``."""
    return subprocess.run(
        [sys.executable, "-W", "error::RuntimeWarning", "-m", "src.execution.runner", *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


# ---------------------------------------------------------------------------
# Multi-source fallback chain
# ---------------------------------------------------------------------------


def test_auto_chain_order_prefers_databento_then_ibkr_then_offline() -> None:
    databento = FakeBackfill("databento")
    ibkr = FakeBackfill("ibkr")
    offline = FakeBackfill("offline")
    chain = build_backfill_chain(
        databento=databento,
        ibkr=ibkr,
        offline=offline,
        source="auto",
        offline_allowed=True,
    )
    assert isinstance(chain, MultiSourceBackfill)
    assert chain.sources == ("databento", "ibkr", "offline")
    assert len(chain) == 3 and bool(chain)
    frame = chain(("ES", "NQ"), START, START + timedelta(days=5))
    assert databento.called and not ibkr.called and not offline.called
    assert set(frame[s.ROOT_SYMBOL_COLUMN].unique().to_list()) == {"ES", "NQ"}
    # The chain adds the neutral second expiration expected by the hydrator.
    per_moment = frame.group_by([s.ROOT_SYMBOL_COLUMN, s.TIMESTAMP_COLUMN]).agg(
        pl.col(s.EXPIRATION_COLUMN).n_unique().alias("expirations")
    )
    assert per_moment["expirations"].unique().to_list() == [2]


def test_auto_skips_unconfigured_databento_and_uses_ibkr() -> None:
    ibkr = FakeBackfill("ibkr")
    offline = FakeBackfill("offline")
    chain = build_backfill_chain(ibkr=ibkr, offline=offline, source="auto", offline_allowed=True)
    assert chain.sources == ("ibkr", "offline")
    frame = chain(("ES", "NQ"), START, START + timedelta(days=5))
    assert ibkr.called and not offline.called
    assert frame.height > 0


def test_auto_falls_back_to_offline_when_vendors_fail(
    caplog: pytest.LogCaptureFixture,
) -> None:
    databento = FakeBackfill("databento", error=RuntimeError("missing API key"))
    ibkr = FakeBackfill("ibkr", error=ConnectionError("TWS unreachable"))
    offline = FakeBackfill("offline")
    chain = build_backfill_chain(
        databento=databento,
        ibkr=ibkr,
        offline=offline,
        source="auto",
        offline_allowed=True,
    )
    with caplog.at_level("WARNING"):
        frame = chain(("ES",), START, START + timedelta(days=5))
    assert databento.called and ibkr.called and offline.called
    assert frame.height > 0
    messages = [record.getMessage() for record in caplog.records]
    assert any("missing API key" in message for message in messages)
    assert any("TWS unreachable" in message for message in messages)


def test_auto_never_uses_offline_unless_explicitly_allowed() -> None:
    databento = FakeBackfill("databento", error=RuntimeError("no key"))
    ibkr = FakeBackfill("ibkr", error=RuntimeError("TWS down"))
    offline = FakeBackfill("offline")
    chain = build_backfill_chain(
        databento=databento,
        ibkr=ibkr,
        offline=offline,
        source="auto",
        offline_allowed=False,
    )
    assert chain.sources == ("databento", "ibkr")
    with pytest.raises(HydrationError) as excinfo:
        chain(("ES",), START, START + timedelta(days=5))
    assert "(1) rerun with --offline" in str(excinfo.value)
    assert not offline.called


def test_auto_with_no_available_source_fails_closed_with_all_hints() -> None:
    chain = build_backfill_chain(source="auto", offline_allowed=False)
    assert chain.sources == ()
    assert len(chain) == 0 and not chain
    with pytest.raises(HydrationError) as excinfo:
        chain(("ES",), START, START + timedelta(days=5))
    message = str(excinfo.value)
    assert "no hydration backfill source is available" in message
    for hint in ("--offline", "IBKR TWS", "FUTURESRLX_ENVIRONMENT", "IBKR_PORT"):
        assert hint in message
    assert "FUTURESRLX_DATABENTO_API_KEY" in message
    assert HYDRATION_SOURCE_HINTS in message


def test_forced_source_unavailable_fails_closed() -> None:
    with pytest.raises(HydrationError) as databento_error:
        build_backfill_chain(source="databento")
    assert "FUTURESRLX_DATABENTO_API_KEY" in str(databento_error.value)

    with pytest.raises(HydrationError) as ibkr_error:
        build_backfill_chain(source="ibkr")
    message = str(ibkr_error.value)
    assert "FUTURESRLX_ENVIRONMENT" in message
    assert "IBKR_PORT" in message
    assert "TWS" in message

    with pytest.raises(HydrationError) as offline_error:
        build_backfill_chain(source="offline")
    assert "--offline" in str(offline_error.value)

    with pytest.raises(HydrationError) as unknown:
        build_backfill_chain(source="bloomberg")
    assert "unknown hydration source" in str(unknown.value)
    assert "auto" in str(unknown.value)


def test_forced_source_uses_only_that_vendor() -> None:
    databento = FakeBackfill("databento")
    ibkr = FakeBackfill("ibkr")
    offline = FakeBackfill("offline")
    chain = build_backfill_chain(
        databento=databento,
        ibkr=ibkr,
        offline=offline,
        source="ibkr",
    )
    assert chain.sources == ("ibkr",)
    frame = chain(("ES",), START, START + timedelta(days=5))
    assert ibkr.called and not databento.called and not offline.called
    assert frame.height > 0


def test_partial_coverage_falls_back_per_root() -> None:
    databento = FakeBackfill("databento", only=("ES",))
    ibkr = FakeBackfill("ibkr", only=("NQ",))
    chain = build_backfill_chain(
        databento=databento, ibkr=ibkr, source="auto", offline_allowed=False
    )
    frame = chain(("ES", "NQ"), START, START + timedelta(days=5))
    assert set(frame[s.ROOT_SYMBOL_COLUMN].unique().to_list()) == {"ES", "NQ"}
    assert databento.calls[0][0] == ("ES", "NQ")
    # The second source is only asked for the still-missing root.
    assert ibkr.calls[0][0] == ("NQ",)


def test_chain_falls_through_empty_and_malformed_frames() -> None:
    empty = FakeBackfill("databento", empty=True)
    malformed_frame = pl.DataFrame({"timestamp_utc": [START]})
    malformed = FakeBackfill("ibkr", frame=malformed_frame)
    offline = FakeBackfill("offline")
    chain = build_backfill_chain(
        databento=empty,
        ibkr=malformed,
        offline=offline,
        source="auto",
        offline_allowed=True,
    )
    frame = chain(("ES",), START, START + timedelta(days=5))
    assert offline.called
    assert frame.height > 0

    empty_chain = build_backfill_chain(
        databento=FakeBackfill("databento", empty=True),
        ibkr=FakeBackfill("ibkr", empty=True),
        source="auto",
        offline_allowed=False,
    )
    with pytest.raises(HydrationError) as excinfo:
        empty_chain(("ES",), START, START + timedelta(days=5))
    assert "returned no rows" in str(excinfo.value)


def test_chain_requires_at_least_one_root() -> None:
    chain = build_backfill_chain(databento=FakeBackfill("databento"), source="databento")
    with pytest.raises(HydrationError):
        chain((), START, START + timedelta(days=5))


# ---------------------------------------------------------------------------
# Neutral term structure / offline source
# ---------------------------------------------------------------------------


def test_add_neutral_term_structure_is_flat_and_idempotent() -> None:
    frame = make_bars_frame(roots=("ES", "NQ"), periods=4)
    augmented = add_neutral_term_structure(frame)
    per_moment = augmented.group_by([s.ROOT_SYMBOL_COLUMN, s.TIMESTAMP_COLUMN]).agg(
        pl.col(s.EXPIRATION_COLUMN).n_unique().alias("expirations")
    )
    assert per_moment["expirations"].unique().to_list() == [2]
    assert augmented.height == frame.height * 2
    # Already augmented data is returned unchanged (no exponential growth).
    assert add_neutral_term_structure(augmented).equals(augmented)
    # The synthetic next expiry is exactly NEUTRAL days after the real one.
    front = frame.select(s.ROOT_SYMBOL_COLUMN, s.TIMESTAMP_COLUMN, s.EXPIRATION_COLUMN)
    next_rows = (
        augmented.filter(pl.col(s.INSTRUMENT_ID_COLUMN).str.ends_with("_NEXT"))
        .select(s.ROOT_SYMBOL_COLUMN, s.TIMESTAMP_COLUMN, s.EXPIRATION_COLUMN)
        .rename({s.EXPIRATION_COLUMN: "_next_expiration"})
    )
    joined = front.join(next_rows, on=[s.ROOT_SYMBOL_COLUMN, s.TIMESTAMP_COLUMN])
    assert joined.height == frame.height
    offsets = joined.select(
        (pl.col("_next_expiration") - pl.col(s.EXPIRATION_COLUMN))
        .dt.total_days()
        .alias("offset_days")
    )
    assert offsets["offset_days"].unique().to_list() == [90]


def test_add_neutral_term_structure_rejects_missing_expiration() -> None:
    frame = make_bars_frame(roots=("ES",), periods=2).with_columns(
        pl.lit(None, dtype=pl.Date).alias(s.EXPIRATION_COLUMN)
    )
    with pytest.raises(HydrationError) as excinfo:
        add_neutral_term_structure(frame)
    assert "no expiration_date" in str(excinfo.value)


def test_offline_backfill_is_deterministic_canonical_and_warm() -> None:
    end = OFFLINE_SIMULATION_END
    start = end - timedelta(days=400)
    frame = offline_backfill(("ES", "NQ"), start, end + timedelta(microseconds=1))
    again = offline_backfill(("ES", "NQ"), start, end + timedelta(microseconds=1))
    assert frame.equals(again)
    assert set(s.BAR_COLUMNS).issubset(frame.columns)
    counts = frame.group_by(s.ROOT_SYMBOL_COLUMN).len()
    assert counts["len"].min() >= MIN_REQUIRED_BARS * 2  # front + neutral next leg
    from src.data.feature_processor import FeatureProcessor

    features = FeatureProcessor().compute(frame)
    final = (
        features.sort(s.TIMESTAMP_COLUMN).group_by(s.ROOT_SYMBOL_COLUMN, maintain_order=True).last()
    )
    for record in final.iter_rows(named=True):
        for feature in s.REQUIRED_FEATURES:
            value = record.get(feature)
            assert value is not None, (record[s.ROOT_SYMBOL_COLUMN], feature)
            assert value == pytest.approx(value)  # finite (NaN fails this)
    assert final[s.TERM_SLOPE].to_list() == [0.0, 0.0]


def test_offline_backfill_validates_window() -> None:
    with pytest.raises(HydrationError):
        offline_backfill(("ES",), START, START)


# ---------------------------------------------------------------------------
# IBKR historical daily bars
# ---------------------------------------------------------------------------


async def test_ibkr_historical_parser_returns_canonical_bars(settings: Settings) -> None:
    client = FakeHistoricalClient(
        {
            "ES": [
                _bar(datetime(2024, 1, 2, 21, 0, tzinfo=UTC), 4_800.0),
                _bar(datetime(2024, 1, 3, 21, 0, tzinfo=UTC), 4_810.0),
            ],
            "NQ": [_bar(datetime(2024, 1, 2, 21, 0, tzinfo=UTC), 16_800.0)],
        }
    )
    fetcher = make_ibkr_fetcher(settings, client)
    frame = await fetcher.fetch_bars(
        ("ES", "NQ"),
        datetime(2024, 1, 1, 21, 0, tzinfo=UTC),
        datetime(2024, 1, 5, 21, 0, tzinfo=UTC),
    )
    assert list(frame.columns[: len(s.BAR_COLUMNS)]) == list(s.BAR_COLUMNS)
    assert frame[s.TIMESTAMP_COLUMN].dtype == pl.Datetime("us", "UTC")
    assert frame.height == 3
    assert set(frame[s.ROOT_SYMBOL_COLUMN].to_list()) == {"ES", "NQ"}
    assert all(isinstance(value, date) for value in frame[s.EXPIRATION_COLUMN].to_list())
    assert frame[s.TIMESTAMP_COLUMN].is_sorted()
    assert client.connect_calls == [
        (
            settings.ibkr_host,
            settings.ibkr_port,
            settings.ibkr_client_id,
            settings.ibkr_connect_timeout_seconds,
        )
    ]
    assert client.disconnect_calls == 1
    assert client.requests and client.requests[0][3] == "1 day"
    assert client.requests[0][4] == 2  # formatDate=2 (UTC)


async def test_ibkr_historical_chunks_long_windows(settings: Settings) -> None:
    client = FakeHistoricalClient({"ES": [_bar(datetime(2024, 1, 2, 21, 0, tzinfo=UTC), 4_800.0)]})
    fetcher = make_ibkr_fetcher(settings, client)
    end = datetime(2024, 6, 1, 21, 0, tzinfo=UTC)
    start = end - timedelta(days=HISTORICAL_MAX_REQUEST_DAYS * 2 + 10)
    frame = await fetcher.fetch_bars(("ES",), start, end)
    assert frame.height == 1  # duplicated across windows, de-duplicated by bar key
    durations = [request[2] for request in client.requests]
    assert len(durations) >= 2
    assert all(duration.endswith(" D") for duration in durations)
    assert sum(int(duration.split(" ")[0]) for duration in durations) >= (end - start).days


def test_ibkr_historical_sync_bridge_from_sync_context(settings: Settings) -> None:
    client = FakeHistoricalClient({"ES": [_bar(datetime(2024, 1, 2, 21, 0, tzinfo=UTC), 4_800.0)]})
    fetcher = make_ibkr_fetcher(settings, client)
    frame = fetcher.fetch_bars_sync(
        ("ES",), datetime(2024, 1, 1, 21, 0, tzinfo=UTC), datetime(2024, 1, 5, 21, 0, tzinfo=UTC)
    )
    assert frame.height == 1


async def test_ibkr_historical_sync_bridge_from_running_loop(settings: Settings) -> None:
    client = FakeHistoricalClient({"ES": [_bar(datetime(2024, 1, 2, 21, 0, tzinfo=UTC), 4_800.0)]})
    fetcher = make_ibkr_fetcher(settings, client)
    frame = await asyncio.to_thread(
        fetcher.fetch_bars_sync,
        ("ES",),
        datetime(2024, 1, 1, 21, 0, tzinfo=UTC),
        datetime(2024, 1, 5, 21, 0, tzinfo=UTC),
    )
    assert frame.height == 1
    assert client.disconnect_calls == 1


async def test_ibkr_historical_empty_response_is_actionable(settings: Settings) -> None:
    fetcher = make_ibkr_fetcher(settings, FakeHistoricalClient())
    with pytest.raises(RuntimeError) as excinfo:
        await fetcher.fetch_bars(
            ("ES",),
            datetime(2024, 1, 1, 21, 0, tzinfo=UTC),
            datetime(2024, 1, 5, 21, 0, tzinfo=UTC),
        )
    message = str(excinfo.value)
    assert "IBKR returned no daily historical bars" in message
    assert "TWS/IB Gateway" in message
    assert "FUTURESRLX_ENVIRONMENT" in message
    assert "FUTURESRLX_IBKR_PORT" in message


async def test_ibkr_historical_connect_failure_is_actionable(settings: Settings) -> None:
    client = FakeHistoricalClient(connect_error=ConnectionError("refused"))
    fetcher = make_ibkr_fetcher(settings, client)
    with pytest.raises(RuntimeError) as excinfo:
        await fetcher.fetch_bars(
            ("ES",),
            datetime(2024, 1, 1, 21, 0, tzinfo=UTC),
            datetime(2024, 1, 5, 21, 0, tzinfo=UTC),
        )
    message = str(excinfo.value)
    assert "cannot connect to TWS/IB Gateway" in message
    assert f"{settings.ibkr_host}:{settings.ibkr_port}" in message
    assert "refused" in message
    assert client.disconnect_calls == 0


async def test_ibkr_historical_ignores_inconsistent_ohlc(settings: Settings) -> None:
    client = FakeHistoricalClient(
        {
            "ES": [
                _bar(datetime(2024, 1, 2, 21, 0, tzinfo=UTC), 4_800.0, high=4_000.0),
                _bar(datetime(2024, 1, 3, 21, 0, tzinfo=UTC), 4_810.0),
            ]
        }
    )
    fetcher = make_ibkr_fetcher(settings, client)
    frame = await fetcher.fetch_bars(
        ("ES",),
        datetime(2024, 1, 1, 21, 0, tzinfo=UTC),
        datetime(2024, 1, 5, 21, 0, tzinfo=UTC),
    )
    assert frame.height == 1


# ---------------------------------------------------------------------------
# Settings-aware source resolution
# ---------------------------------------------------------------------------


def test_describe_hydration_sources_reports_availability(tmp_path: Path) -> None:
    backtest = settings_with(tmp_path)
    description = describe_hydration_sources(backtest)
    assert description["requested"] == "auto"
    assert description["order"] == []
    assert description["available"] == {
        "databento": False,
        "ibkr": False,
        "offline": False,
    }
    reasons = description["unavailable_reasons"]
    assert isinstance(reasons, dict)
    assert "FUTURESRLX_DATABENTO_API_KEY" in reasons["databento"]
    assert "FUTURESRLX_ENVIRONMENT" in reasons["ibkr"]
    assert "--offline" in reasons["offline"]
    hints = description["hints"]
    assert isinstance(hints, str) and "--offline" in hints

    offline = describe_hydration_sources(backtest, offline_allowed=True)
    assert offline["order"] == ["offline"]
    assert offline["offline_allowed"] is True


def test_build_hydration_chain_resolves_full_auto_order(tmp_path: Path) -> None:
    settings = settings_with(tmp_path, environment="paper", databento_api_key="test-key")
    master = settings.instruments
    repository = ParquetRepository(settings)
    description = describe_hydration_sources(settings, offline_allowed=True)
    assert description["order"] == ["databento", "ibkr", "offline"]
    chain = build_hydration_chain(
        settings,
        repository=repository,
        instrument_master=master,
        offline_allowed=True,
    )
    assert chain.sources == ("databento", "ibkr", "offline")


def test_build_hydration_chain_accepts_injected_fakes(tmp_path: Path) -> None:
    settings = settings_with(tmp_path)
    master = settings.instruments
    databento = FakeBackfill("databento")
    ibkr = FakeBackfill("ibkr")
    offline = FakeBackfill("offline")
    chain = build_hydration_chain(
        settings,
        repository=ParquetRepository(settings),
        instrument_master=master,
        source="auto",
        offline_allowed=True,
        databento_fetch=databento,
        ibkr_fetch=ibkr,
        offline_fetch=offline,
    )
    assert chain.sources == ("databento", "ibkr", "offline")
    frame = chain(("ES",), START, START + timedelta(days=5))
    assert databento.called and frame.height > 0


def test_build_hydration_chain_forced_source_fails_closed(tmp_path: Path) -> None:
    settings = settings_with(tmp_path)
    with pytest.raises(HydrationError) as excinfo:
        build_hydration_chain(
            settings,
            repository=ParquetRepository(settings),
            instrument_master=settings.instruments,
            source="databento",
        )
    assert "FUTURESRLX_DATABENTO_API_KEY" in str(excinfo.value)
    assert "--offline" in str(excinfo.value)


def test_hydration_source_selector_matches_settings_field(tmp_path: Path) -> None:
    settings = settings_with(tmp_path, hydration_source="offline")
    description = describe_hydration_sources(settings)
    assert description["requested"] == "offline"
    assert description["order"] == ["offline"]
    assert "offline" in HYDRATION_SOURCES


# ---------------------------------------------------------------------------
# Offline streamer / broker / hydration
# ---------------------------------------------------------------------------


def test_offline_hydration_warm_states_end_at_simulation_end(
    settings: Settings,
) -> None:
    end = OFFLINE_SIMULATION_END
    repository = ParquetRepository(settings)
    chain = build_backfill_chain(offline=offline_backfill, source="offline")
    hydrator = ColdStartHydrator(
        repository=repository,
        lookback_days=settings.hydration_lookback_days,
        backfill=chain,
        clock=lambda: end,
    )
    result = hydrator.hydrate(("ES", "NQ"), end, equity=OFFLINE_INITIAL_CASH)
    assert result.backfilled is True
    assert result.final_state.timestamp == end
    assert hydrator.is_warm(result.final_state)
    assert set(result.roots) == {"ES", "NQ"}
    assert all(count >= MIN_REQUIRED_BARS for count in result.coverage.values())


def test_offline_streamer_replays_warm_states_and_marks_market(
    settings: Settings, instrument_master: InstrumentsConfig
) -> None:
    end = OFFLINE_SIMULATION_END
    start = end - timedelta(days=settings.hydration_lookback_days)
    roots = ("ES", "GC")
    simulated = SimulatedBroker(
        instrument_master, initial_cash=OFFLINE_INITIAL_CASH, timestamp=start
    )
    broker = OfflineTargetBroker(simulated_broker=simulated, instrument_master=instrument_master)
    streamer = OfflineMarketStateStreamer(
        roots=roots,
        start=start,
        end=end,
        equity=OFFLINE_INITIAL_CASH,
        instrument_master=instrument_master,
        replay_states=3,
        on_state=broker.update_market,
    )

    async def collect() -> list[MarketState]:
        return [state async for state in streamer.stream()]

    states = asyncio.run(collect())
    assert len(states) == 3
    assert [state.timestamp for state in states] == sorted(state.timestamp for state in states)
    assert states[-1].timestamp == end
    for state in states:
        assert state.covariance is not None
        for root in roots:
            snapshot = state.snapshot(root)
            for feature in s.REQUIRED_FEATURES:
                assert feature in snapshot.features
        assert state.equity == OFFLINE_INITIAL_CASH
    assert broker.last_market_state is states[-1]

    async def trade() -> list[Any]:
        assert await broker.get_positions() == {}
        receipts = await broker.submit_targets(
            make_contract_targets({"ES": 1}), {}, states[-1].timestamp
        )
        assert [receipt.status for receipt in receipts] == [OrderStatus.FILLED]
        positions = await broker.get_positions()
        assert positions["ES"].quantity == 1
        flat = await broker.submit_targets(
            ContractTargetSet.empty(), {"ES": 1}, states[-1].timestamp
        )
        assert [receipt.status for receipt in flat] == [OrderStatus.FILLED]
        return await broker.get_positions()

    remaining = asyncio.run(trade())
    assert remaining == {} or all(position.quantity == 0 for position in remaining.values())


def test_offline_streamer_validation_and_connection_flags(
    settings: Settings, instrument_master: InstrumentsConfig
) -> None:
    end = OFFLINE_SIMULATION_END
    start = end - timedelta(days=10)
    with pytest.raises(ValueError):
        OfflineMarketStateStreamer(
            roots=(),
            start=start,
            end=end,
            equity=OFFLINE_INITIAL_CASH,
            instrument_master=instrument_master,
        )
    with pytest.raises(ValueError):
        OfflineMarketStateStreamer(
            roots=("ES",),
            start=end,
            end=start,
            equity=OFFLINE_INITIAL_CASH,
            instrument_master=instrument_master,
        )
    with pytest.raises(ValueError):
        OfflineMarketStateStreamer(
            roots=("ES",),
            start=start,
            end=end,
            equity=OFFLINE_INITIAL_CASH,
            instrument_master=instrument_master,
            replay_states=0,
        )
    streamer = OfflineMarketStateStreamer(
        roots=("ES",),
        start=start,
        end=end,
        equity=OFFLINE_INITIAL_CASH,
        instrument_master=instrument_master,
    )
    assert streamer.roots == ("ES",)
    assert streamer.replay_states > 0
    assert streamer.is_connected is False
    asyncio.run(streamer.connect())
    assert streamer.is_connected is True
    asyncio.run(streamer.disconnect())
    assert streamer.is_connected is False


def test_offline_runner_submits_orders_end_to_end(settings: Settings) -> None:
    args = argparse.Namespace(
        strategy=None,
        check=False,
        offline=True,
        backfill_source=None,
        roots=None,
        max_states=30,
        log_level=None,
    )
    status = _run_offline(settings, args)
    assert status.hydrated is True
    assert status.states_processed == 30
    assert status.halted is False
    assert status.locked is False
    assert status.orders_submitted >= 1
    assert status.orders_rejected == 0
    assert status.started_at == OFFLINE_SIMULATION_END
    assert status.roots == tuple(sorted(OFFLINE_DEFAULT_ROOTS))


# ---------------------------------------------------------------------------
# CLI subprocess checks (offline / no-source / check / lazy imports)
# ---------------------------------------------------------------------------


def test_cli_no_source_exits_2_with_actionable_hints(tmp_path: Path) -> None:
    proc = run_cli("--max-states", "1", env=cli_environment(tmp_path))
    assert proc.returncode == 2, proc.stderr
    assert "Traceback" not in proc.stderr
    stderr = proc.stderr
    for hint in (
        "--offline",
        "IBKR TWS",
        "FUTURESRLX_ENVIRONMENT",
        "IBKR_PORT",
        "FUTURESRLX_DATABENTO_API_KEY",
    ):
        assert hint in stderr
    assert "no hydration backfill source" in stderr


def test_cli_forced_databento_without_key_exits_2_cleanly(tmp_path: Path) -> None:
    proc = run_cli(
        "--backfill-source", "databento", "--max-states", "1", env=cli_environment(tmp_path)
    )
    assert proc.returncode == 2, proc.stderr
    assert "Traceback" not in proc.stderr
    assert "FUTURESRLX_DATABENTO_API_KEY" in proc.stderr
    assert "--offline" in proc.stderr


def test_cli_check_reports_offline_and_backfill_flags(tmp_path: Path) -> None:
    env = cli_environment(tmp_path)
    proc = run_cli("--check", env=env)
    assert proc.returncode == 0, proc.stderr
    assert "RuntimeWarning" not in proc.stderr
    assert "Traceback" not in proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["offline"] is False
    assert payload["backfill_source"] == "auto"
    assert payload["hydration_sources"]["order"] == []
    assert payload["hydration_backfill_chain"] == []
    assert payload["hydration_backfill_ready"] is False

    offline = run_cli("--check", "--offline", "--backfill-source", "offline", env=env)
    assert offline.returncode == 0, offline.stderr
    offline_payload = json.loads(offline.stdout)
    assert offline_payload["offline"] is True
    assert offline_payload["backfill_source"] == "offline"
    assert offline_payload["hydration_backfill_chain"] == ["offline"]
    assert offline_payload["hydration_backfill_ready"] is True

    # ``--offline`` alone implies the synthetic source in the check summary too.
    implied = run_cli("--check", "--offline", env=env)
    assert implied.returncode == 0, implied.stderr
    implied_payload = json.loads(implied.stdout)
    assert implied_payload["offline"] is True
    assert implied_payload["backfill_source"] == "offline"
    assert implied_payload["hydration_backfill_chain"] == ["offline"]


def test_cli_offline_max_states_2_runs_end_to_end(tmp_path: Path) -> None:
    proc = run_cli("--offline", "--max-states", "2", env=cli_environment(tmp_path))
    assert proc.returncode == 0, proc.stderr
    assert "RuntimeWarning" not in proc.stderr
    assert "Traceback" not in proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["hydrated"] is True
    assert payload["states_processed"] == 2
    assert payload["halted"] is False
    assert payload["locked"] is False
    assert payload["started_at"] == OFFLINE_SIMULATION_END.isoformat()
    # No credentials were provided: the run must be fully synthetic/offline.
    assert payload["roots"] == sorted(OFFLINE_DEFAULT_ROOTS)


def test_cli_offline_conflicting_backfill_source_exits_2(tmp_path: Path) -> None:
    proc = run_cli("--offline", "--backfill-source", "databento", env=cli_environment(tmp_path))
    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr
    assert "cannot be combined" in proc.stderr


_LAZY_IMPORT_SCRIPT = textwrap.dedent(
    """
    import sys

    import src.execution

    assert "src.execution.runner" not in sys.modules, sorted(
        name for name in sys.modules if name.startswith("src.execution")
    )
    assert "TradingRunner" in dir(src.execution)
    assert "OrderManagementSystem" in dir(src.execution)
    from src.execution import (
        ColdStartHydrator,
        MultiSourceBackfill,
        OrderManagementSystem,
        PositionReconciler,
        RunnerStatus,
        TargetBroker,
        TradingRunner,
        main,
    )
    assert "src.execution.runner" in sys.modules
    assert TradingRunner.__name__ == "TradingRunner"
    assert RunnerStatus.__name__ == "RunnerStatus"
    assert callable(main)
    assert issubclass(ColdStartHydrator, object)
    assert MultiSourceBackfill is not None
    assert PositionReconciler is not None
    assert TargetBroker is not None
    assert OrderManagementSystem is not None
    print("lazy-ok")
    """
)


def test_execution_package_lazily_imports_the_runner(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, "-W", "error::RuntimeWarning", "-c", _LAZY_IMPORT_SCRIPT],
        cwd=REPO_ROOT,
        env=cli_environment(tmp_path),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "lazy-ok" in proc.stdout
    assert "RuntimeWarning" not in proc.stderr


def test_runner_cli_module_import_has_no_runpy_warning(tmp_path: Path) -> None:
    proc = run_cli("--check", env=cli_environment(tmp_path))
    assert proc.returncode == 0
    assert "RuntimeWarning" not in proc.stderr
    assert "found in sys.modules" not in proc.stderr


# ---------------------------------------------------------------------------
# HydrationResult / MarketState hygiene
# ---------------------------------------------------------------------------


def test_make_market_state_fixture_is_warm_for_chain_output(
    settings: Settings,
) -> None:
    # Guard: the hydration warm-up contract used by the chain matches the fixtures.
    end = OFFLINE_SIMULATION_END
    repository = ParquetRepository(settings)
    chain = build_backfill_chain(offline=offline_backfill, source="offline")
    hydrator = ColdStartHydrator(
        repository=repository, lookback_days=400, backfill=chain, clock=lambda: end
    )
    hydrator.hydrate(("ES",), end, equity=OFFLINE_INITIAL_CASH)
    warm = make_market_state(roots=("ES",))
    assert hydrator.is_warm(warm)
