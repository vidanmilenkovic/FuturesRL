"""Unit tests for Spec-1 timeframe-aware annualisation and UTC daily risk resets.

The canonical registry (``src.data.timeframes``) is the single source of truth for
``bars_per_year``; strategies/risk must annualise per-bar statistics with it while
preserving the exact historical 252-trading-day behaviour for ``"1d"`` states.
"""

from __future__ import annotations

import dataclasses
import math
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from configs.strategy import BacktestSpec
from pydantic import ValidationError
from src.core.types import MarketState
from src.data.timeframes import bars_per_year, supported_timeframes
from src.risk.quant_risk_overlay import QuantRiskConfig, QuantRiskOverlay
from src.strategies._numeric import (
    annualized_volatility,
    bars_per_year_for,
    covariance_matrix,
    portfolio_volatility,
)
from src.strategies.allocators import RiskParityAllocator, VolatilityTargetAllocator
from src.strategies.timing import TSMOMTiming

from tests.fixtures.market import FIXED_TIMESTAMP, make_market_state

TARGET_VOL = 0.12
BAR_VOL = 0.01


def _state(timeframe: str = "1d", *, seed: int = 7) -> MarketState:
    """Two-root state whose per-bar returns have a fixed 1% volatility."""
    base = make_market_state(roots=("ES", "NQ"), with_covariance=False, seed=seed)
    rng = np.random.default_rng(seed)
    series = {root: rng.normal(0.001, BAR_VOL, size=200) for root in base.ids}
    return dataclasses.replace(base, timeframe=timeframe, returns=series)


# ---------------------------------------------------------------------------
# Registry plumbing / defaults
# ---------------------------------------------------------------------------


def test_bars_per_year_helper_and_daily_default_are_unchanged() -> None:
    assert bars_per_year_for(None) == 252.0
    assert bars_per_year_for("1d") == 252.0
    assert bars_per_year_for("1w") == 52.0
    for timeframe in supported_timeframes():
        assert bars_per_year_for(timeframe) == float(bars_per_year(timeframe))


@pytest.mark.parametrize("timeframe", supported_timeframes())
def test_marketstate_timeframe_defaults_and_registry_validation(timeframe: str) -> None:
    state = _state(timeframe)
    assert state.timeframe == timeframe
    assert "timeframe" in {field.name for field in dataclasses.fields(MarketState)}


def test_marketstate_timeframe_normalises_aliases_and_rejects_unknown() -> None:
    state = _state("1h")
    assert dataclasses.replace(state, timeframe=" 1H ").timeframe == "1h"
    assert dataclasses.replace(state, timeframe="5min").timeframe == "5m"
    with pytest.raises(ValueError, match="unsupported timeframe"):
        dataclasses.replace(state, timeframe="2h")
    assert make_market_state().timeframe == "1d"


def test_backtest_spec_timeframe_default_validation_and_dump() -> None:
    default = BacktestSpec(
        start=FIXED_TIMESTAMP.date(), end=FIXED_TIMESTAMP.date() + timedelta(days=30)
    )
    assert default.timeframe == "1d"
    assert default.model_dump()["timeframe"] == "1d"
    configured = BacktestSpec(
        start=FIXED_TIMESTAMP.date(),
        end=FIXED_TIMESTAMP.date() + timedelta(days=30),
        timeframe=" 5M ",
    )
    assert configured.timeframe == "5m"
    assert configured.model_dump()["timeframe"] == "5m"
    with pytest.raises(ValidationError, match="timeframe"):
        BacktestSpec(
            start=FIXED_TIMESTAMP.date(),
            end=FIXED_TIMESTAMP.date() + timedelta(days=30),
            timeframe="2h",
        )


# ---------------------------------------------------------------------------
# Annualisation scaling (volatility / vol-target / risk-parity / TSMOM)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("timeframe", ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"])
def test_annualized_volatility_scales_with_bars_per_year(timeframe: str) -> None:
    state = _state(timeframe)
    expected_factor = math.sqrt(bars_per_year(timeframe))
    for root in state.ids:
        # The helper estimates from the trailing ``lookback`` (default 60) bars.
        realised_bar_vol = float(np.std(state.returns[root][-60:], ddof=1))
        assert annualized_volatility(state, root) == pytest.approx(
            realised_bar_vol * expected_factor, rel=1e-9
        )


def test_1d_annualization_is_the_historical_sqrt_252_behaviour() -> None:
    state = _state("1d")
    realised_bar_vol = float(np.std(state.returns["ES"][-60:], ddof=1))
    assert annualized_volatility(state, "ES") == pytest.approx(
        realised_bar_vol * math.sqrt(252.0), rel=1e-12
    )


def test_volatility_target_hits_annual_target_across_timeframes() -> None:
    allocator = VolatilityTargetAllocator(
        target_annual_volatility=TARGET_VOL, max_gross_leverage=10.0
    )
    gross: dict[str, float] = {}
    for timeframe in ("1d", "1h", "5m"):
        state = _state(timeframe)
        weights = allocator.allocate(state.ids, state)
        gross[timeframe] = weights.gross
        covariance = covariance_matrix(state, state.ids)
        realised = portfolio_volatility(
            np.array([weights[root] for root in state.ids], dtype=np.float64), covariance
        )
        assert realised == pytest.approx(TARGET_VOL, rel=1e-6)
    # Same per-bar returns, so gross must shrink by sqrt(bars/year) for intraday.
    assert gross["1d"] / gross["1h"] == pytest.approx(
        math.sqrt(bars_per_year("1h") / 252.0), rel=1e-6
    )
    assert gross["1d"] / gross["5m"] == pytest.approx(
        math.sqrt(bars_per_year("5m") / 252.0), rel=1e-6
    )


def test_risk_parity_vol_target_scales_with_timeframe() -> None:
    allocator = RiskParityAllocator(target_annual_volatility=TARGET_VOL, max_gross_leverage=10.0)
    daily = allocator.allocate(_state("1d").ids, _state("1d"))
    intraday = allocator.allocate(_state("5m").ids, _state("5m"))
    assert daily.gross / intraday.gross == pytest.approx(
        math.sqrt(bars_per_year("5m") / 252.0), rel=1e-6
    )


def test_tsmom_volatility_scaling_uses_timeframe() -> None:
    overlay = TSMOMTiming(
        lookback_fast=5,
        lookback_slow=20,
        vol_lookback=20,
        target_volatility=0.10,
        min_scale=0.0,
        max_scale=10.0,
    )

    def factor(state: MarketState) -> float:
        return overlay.scaling_factor(state, "ES")

    daily = factor(_state("1d"))
    intraday = factor(_state("5m"))
    assert abs(daily) > abs(intraday)
    assert abs(daily) / abs(intraday) == pytest.approx(
        math.sqrt(bars_per_year("5m") / 252.0), rel=1e-6
    )


# ---------------------------------------------------------------------------
# Quant risk overlay: timeframe covariance annualisation
# ---------------------------------------------------------------------------


def test_quant_risk_config_derives_annualization_factor_from_timeframe() -> None:
    assert QuantRiskConfig().annualization_factor == 252
    assert QuantRiskConfig(timeframe="5m").annualization_factor == bars_per_year("5m")
    assert QuantRiskConfig(timeframe="1h").annualization_factor == bars_per_year("1h")
    # An explicit non-default factor is an override and survives the timeframe.
    assert QuantRiskConfig(timeframe="5m", annualization_factor=365).annualization_factor == 365
    with pytest.raises(ValidationError, match="timeframe"):
        QuantRiskConfig(timeframe="2h")


def test_quant_risk_overlay_vol_target_uses_state_timeframe() -> None:
    def config(timeframe: str) -> QuantRiskConfig:
        return QuantRiskConfig(
            timeframe=timeframe,
            max_gross_leverage=50.0,
            max_single_instrument_weight=50.0,
            max_sector_weight=50.0,
            min_portfolio_volatility=1e-9,
        )

    from src.core.types import WeightVector

    weights = WeightVector({"ES": 0.5})
    daily_weights = QuantRiskOverlay(config("1d")).apply_risk(weights, _state("1d"))
    intraday_weights = QuantRiskOverlay(config("5m")).apply_risk(weights, _state("5m"))
    assert daily_weights.gross > intraday_weights.gross
    # Same per-bar returns: annualised sigma scales by sqrt(bars_per_year), so the
    # vol-target scale shrinks by the same factor.
    assert daily_weights.gross / intraday_weights.gross == pytest.approx(
        math.sqrt(bars_per_year("5m") / 252.0), rel=1e-6
    )


# ---------------------------------------------------------------------------
# UTC calendar-day daily loss / counters
# ---------------------------------------------------------------------------


def _utc(*args: int) -> datetime:
    return datetime(*args, tzinfo=UTC)


def test_daily_loss_accumulates_intraday_and_resets_on_utc_day_boundary() -> None:
    overlay = QuantRiskOverlay(QuantRiskConfig(daily_max_loss_pct=0.03))
    day1 = _utc(2026, 1, 5)
    overlay.update_equity(1_000_000.0, day1.replace(hour=9, minute=30))
    # Two intraday bars 2.0% + 1.2% down: no reset between them, latch triggers.
    overlay.update_equity(980_000.0, day1.replace(hour=10, minute=30))
    assert overlay.daily_loss_latched is False
    overlay.update_equity(968_000.0, day1.replace(hour=15, minute=59))
    assert overlay.daily_loss_latched is True
    snapshot = overlay.snapshot_state()
    assert snapshot["starting_daily_equity"] == 1_000_000.0
    assert snapshot["session_date"] == "2026-01-05"
    assert snapshot["portfolio_high_water_mark"] == 1_000_000.0

    # First bar of the next UTC calendar day re-bases the daily NLV and clears the
    # latch; the portfolio HWM survives (drawdown history is not reset).
    overlay.update_equity(967_000.0, _utc(2026, 1, 6, 0, 5))
    assert overlay.daily_loss_latched is False
    snapshot = overlay.snapshot_state()
    assert snapshot["session_date"] == "2026-01-06"
    assert snapshot["starting_daily_equity"] == 967_000.0
    assert snapshot["portfolio_high_water_mark"] == 1_000_000.0

    # The same-day accumulation rule still applies on the new day.
    overlay.update_equity(935_000.0, _utc(2026, 1, 6, 9, 35))
    assert overlay.daily_loss_latched is True
    overlay.update_equity(935_500.0, _utc(2026, 1, 6, 9, 40))
    assert overlay.daily_loss_latched is True
    assert overlay.snapshot_state()["starting_daily_equity"] == 967_000.0


def test_many_intraday_bars_do_not_reset_daily_counters() -> None:
    overlay = QuantRiskOverlay(QuantRiskConfig(daily_max_loss_pct=0.05))
    start = _utc(2026, 3, 2, 8, 0)
    overlay.update_equity(1_000_000.0, start)
    # 20 intraday bars within the same UTC day: the daily baseline must not move.
    for minute in range(0, 20):
        overlay.update_equity(1_000_000.0 - 1_000.0 * minute, start + timedelta(minutes=5 * minute))
        assert overlay.snapshot_state()["starting_daily_equity"] == 1_000_000.0
        assert overlay.daily_loss_latched is False
    # Bar 21 crosses 5% down -> latches from the same daily baseline.
    overlay.update_equity(949_000.0, start + timedelta(minutes=5 * 20))
    assert overlay.daily_loss_latched is True
