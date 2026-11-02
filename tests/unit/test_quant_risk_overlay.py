"""Unit tests for the Tier B quantitative futures risk overlay.

Every test is offline and deterministic. Numeric expectations are derived from
the synthetic fixture covariance (annualised sector vols 18% equity, 35% energy,
6% rates, 15% metals with clustered correlations) or hand-computed.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from datetime import UTC, datetime

import numpy as np
import pytest
from configs.settings import RiskSettings
from src.core.interfaces import RiskOverlay
from src.core.types import MarketState, RoundingMode, WeightVector
from src.risk.quant_risk_overlay import QuantRiskConfig, QuantRiskOverlay

from tests.fixtures.market import (
    FIXED_TIMESTAMP,
    make_market_state,
    make_snapshot,
    make_weight_vector,
)

NEXT_DAY = datetime(2024, 1, 3, 21, 0, tzinfo=UTC)


def _config(**overrides: float | bool) -> QuantRiskConfig:
    """Permissive config: only the control under test is isolated by default."""
    base: dict[str, float | bool] = {
        "volatility_targeting_enabled": False,
        "max_single_instrument_weight": 10.0,
        "max_sector_weight": 10.0,
        "max_gross_leverage": 10.0,
        "trailing_stop_pct": 0.99,
        "drawdown_limit_pct": 0.99,
        "daily_max_loss_pct": 1.0,
    }
    base.update(overrides)
    return QuantRiskConfig(**base)  # type: ignore[arg-type]


def _overlay(**overrides: float | bool) -> QuantRiskOverlay:
    cvars = dict(overrides)
    config = _config(**cvars)
    return QuantRiskOverlay(config=config, clock=lambda: FIXED_TIMESTAMP)


def _state(
    *,
    equity: float = 1_000_000.0,
    prices: dict[str, float] | None = None,
    with_covariance: bool = True,
    with_returns: bool = True,
    timestamp: datetime = FIXED_TIMESTAMP,
) -> MarketState:
    return make_market_state(
        equity=equity,
        prices=prices or {},
        with_covariance=with_covariance,
        with_returns=with_returns,
        timestamp=timestamp,
    )


def _expected_vol_target(
    weights: dict[str, float], state: MarketState, target: float
) -> dict[str, float]:
    ids = tuple(weights)
    covariance = state.covariance_submatrix(ids)
    assert covariance is not None
    vector = np.array([weights[i] for i in ids], dtype=np.float64)
    sigma = math.sqrt(float(vector @ covariance @ vector))
    scale = target / sigma
    return {root: value * scale for root, value in weights.items()}


def _expected_ewma_vol(series: np.ndarray, halflife: float, annualization: int) -> float:
    length = series.size
    decay = 0.5 ** (1.0 / halflife)
    raw = decay ** np.arange(length - 1, -1, -1, dtype=np.float64)
    weights = raw / raw.sum()
    mean = float(weights @ series)
    centered = series - mean
    variance = float(np.sum(weights * centered * centered)) * annualization
    return math.sqrt(variance)


# ---------------------------------------------------------------------------
# Ex-ante volatility targeting
# ---------------------------------------------------------------------------


def test_volatility_target_single_leg() -> None:
    """ES carries 18% annualised vol; a 12% target scales the weight by 12/18."""
    overlay = _overlay(target_annual_volatility=0.12, volatility_targeting_enabled=True)
    state = _state()
    result = overlay.apply_risk(make_weight_vector({"ES": 1.0}), state)
    assert result["ES"] == pytest.approx(0.12 / 0.18, rel=1e-12)


def test_volatility_target_uses_full_covariance() -> None:
    overlay = _overlay(target_annual_volatility=0.12, volatility_targeting_enabled=True)
    state = _state()
    weights = {"ES": 0.5, "NQ": -0.25, "CL": 0.15}
    expected = _expected_vol_target(weights, state, 0.12)
    result = overlay.apply_risk(make_weight_vector(weights), state)
    for root, value in expected.items():
        assert result[root] == pytest.approx(value, rel=1e-12)


def test_volatility_target_falls_back_to_returns_ewma() -> None:
    """With no covariance matrix the overlay estimates EWMA covariance from returns."""
    overlay = _overlay(
        target_annual_volatility=0.12, ewma_halflife_days=20.0, volatility_targeting_enabled=True
    )
    state = _state(with_covariance=False)
    series = np.asarray(state.returns["ES"], dtype=np.float64)
    sigma = _expected_ewma_vol(series, 20.0, 252)
    result = overlay.apply_risk(make_weight_vector({"ES": 1.0}), state)
    assert result["ES"] == pytest.approx(0.12 / sigma, rel=1e-12)


def test_zero_covariance_is_ignored_not_fatal() -> None:
    overlay = _overlay()
    state = replace(
        _state(),
        covariance=np.zeros((5, 5), dtype=np.float64),
        covariance_instruments=("ES", "NQ", "CL", "GC", "ZN"),
        returns={},
    )
    weights = make_weight_vector({"ES": 0.5, "NQ": -0.25})
    result = overlay.apply_risk(weights, state)
    assert result["ES"] == pytest.approx(0.5)
    assert result["NQ"] == pytest.approx(-0.25)
    assert all(math.isfinite(value) for value in result.as_dict().values())


def test_negative_definite_covariance_falls_back_to_diagonal() -> None:
    """A non-PSD covariance matrix must never produce NaN/inf weights."""
    overlay = _overlay()
    state = _state(with_returns=False)
    covariance = np.full((5, 5), -1.0, dtype=np.float64)
    state = replace(
        state,
        covariance=covariance,
        covariance_instruments=("ES", "NQ", "CL", "GC", "ZN"),
        returns={},
    )
    weights = make_weight_vector({"ES": 0.4, "CL": 0.2})
    result = overlay.apply_risk(weights, state)
    assert result["ES"] == pytest.approx(0.4)
    assert result["CL"] == pytest.approx(0.2)


def test_volatility_targeting_disabled_leaves_weights_unchanged() -> None:
    overlay = _overlay(volatility_targeting_enabled=False)
    result = overlay.apply_risk(make_weight_vector({"ES": 0.5}), _state())
    assert result["ES"] == pytest.approx(0.5)


def test_volatility_target_never_produces_non_finite_weights() -> None:
    overlay = _overlay(target_annual_volatility=0.12, volatility_targeting_enabled=True)
    covariance = np.eye(5, dtype=np.float64) * 1e-30
    state = replace(
        _state(),
        covariance=covariance,
        covariance_instruments=("ES", "NQ", "CL", "GC", "ZN"),
        returns={},
    )
    # Tiny volatilities are below min_portfolio_volatility: no explosive scaling.
    result = overlay.apply_risk(make_weight_vector({"ES": 0.5, "NQ": 0.5}), state)
    assert math.isfinite(result["ES"])
    assert math.isfinite(result["NQ"])


# ---------------------------------------------------------------------------
# Leverage / concentration caps
# ---------------------------------------------------------------------------


def test_gross_leverage_cap_scales_all_legs() -> None:
    overlay = _overlay(volatility_targeting_enabled=False, max_gross_leverage=2.0)
    result = overlay.apply_risk(make_weight_vector({"ES": 2.0, "NQ": 2.0}), _state())
    assert result["ES"] == pytest.approx(1.0)
    assert result["NQ"] == pytest.approx(1.0)
    assert result.gross == pytest.approx(2.0)


def test_single_instrument_cap_clips() -> None:
    overlay = _overlay(volatility_targeting_enabled=False, max_single_instrument_weight=0.4)
    result = overlay.apply_risk(make_weight_vector({"ES": 0.8, "CL": -0.2}), _state())
    assert result["ES"] == pytest.approx(0.4)
    assert result["CL"] == pytest.approx(-0.2)


def test_sector_cap_scales_equity_cluster_proportionally() -> None:
    """ES + NQ share the equity beta sector: their gross is capped at 0.4."""
    overlay = _overlay(
        volatility_targeting_enabled=False,
        max_sector_weight=0.4,
        max_single_instrument_weight=10.0,
    )
    result = overlay.apply_risk(make_weight_vector({"ES": 0.4, "NQ": 0.4, "CL": 0.2}), _state())
    assert result["ES"] == pytest.approx(0.2)
    assert result["NQ"] == pytest.approx(0.2)
    # Energy is a different sector and stays untouched.
    assert result["CL"] == pytest.approx(0.2)


def test_sector_cap_uses_absolute_weights_for_shorts() -> None:
    overlay = _overlay(
        volatility_targeting_enabled=False,
        max_sector_weight=0.4,
        max_single_instrument_weight=10.0,
    )
    result = overlay.apply_risk(make_weight_vector({"ES": -0.5, "NQ": 0.3}), _state())
    assert result["ES"] == pytest.approx(-0.25)
    assert result["NQ"] == pytest.approx(0.15)


def test_unknown_root_raises_value_error() -> None:
    overlay = _overlay()
    with pytest.raises(ValueError, match="XX"):
        overlay.apply_risk(make_weight_vector({"XX": 0.1}), _state())


def test_empty_and_flat_weights_return_flat_vector() -> None:
    overlay = _overlay()
    state = _state()
    assert overlay.apply_risk(WeightVector.empty(), state) == WeightVector.empty()
    flat = overlay.apply_risk(make_weight_vector({"ES": 0.0}), state)
    assert flat.is_flat()
    assert set(flat.as_dict()) == {"ES"}


# ---------------------------------------------------------------------------
# Trailing stops
# ---------------------------------------------------------------------------


def test_instrument_trailing_stop_flattens_loser_until_reset() -> None:
    overlay = _overlay(trailing_stop_pct=0.10)
    overlay.update_equity(1_000_000.0, timestamp=FIXED_TIMESTAMP)
    assert overlay.apply_risk(make_weight_vector({"ES": 0.5}), _state())["ES"] == pytest.approx(0.5)

    crashed = _state(prices={"ES": 4800.0 * 0.89})
    stopped = overlay.apply_risk(make_weight_vector({"ES": 0.5}), crashed)
    assert stopped["ES"] == 0.0
    assert "ES" in overlay.snapshot_state()["stopped_roots"]

    # Still flat on the recovery bounce until the next session roll.
    recovered = _state(prices={"ES": 4800.0})
    assert overlay.apply_risk(make_weight_vector({"ES": 0.5}), recovered)["ES"] == 0.0

    overlay.reset_day(1_000_000.0, timestamp=FIXED_TIMESTAMP)
    assert overlay.apply_risk(make_weight_vector({"ES": 0.5}), recovered)["ES"] == pytest.approx(
        0.5
    )


def test_short_trailing_stop_flattens_on_adverse_rally() -> None:
    overlay = _overlay(trailing_stop_pct=0.10)
    overlay.update_equity(1_000_000.0, timestamp=FIXED_TIMESTAMP)
    assert overlay.apply_risk(make_weight_vector({"ES": -0.5}), _state())["ES"] == pytest.approx(
        -0.5
    )
    rallied = _state(prices={"ES": 4800.0 * 1.12})
    assert overlay.apply_risk(make_weight_vector({"ES": -0.5}), rallied)["ES"] == 0.0


def test_portfolio_trailing_stop_flattens_everything() -> None:
    overlay = _overlay(trailing_stop_pct=0.10, drawdown_limit_pct=0.99)
    overlay.update_equity(1_000_000.0, timestamp=FIXED_TIMESTAMP)
    overlay.update_equity(890_000.0, timestamp=FIXED_TIMESTAMP)
    state = _state(equity=890_000.0)
    result = overlay.apply_risk(make_weight_vector({"ES": 0.5, "CL": 0.2}), state)
    assert result.is_flat()
    # Recovery above the trailing threshold re-enables risk taking.
    overlay.update_equity(1_000_000.0, timestamp=FIXED_TIMESTAMP)
    restored = overlay.apply_risk(make_weight_vector({"ES": 0.5, "CL": 0.2}), _state())
    assert restored["ES"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Daily max-loss kill switch / drawdown limit
# ---------------------------------------------------------------------------


def test_daily_max_loss_kill_switch_latches_and_resets() -> None:
    overlay = _overlay(daily_max_loss_pct=0.03, trailing_stop_pct=0.99, drawdown_limit_pct=0.99)
    overlay.update_equity(1_000_000.0, timestamp=FIXED_TIMESTAMP)
    overlay.update_equity(970_000.0, timestamp=FIXED_TIMESTAMP)
    assert overlay.kill_switch_active is True
    state = _state(equity=970_000.0)
    weights = make_weight_vector({"ES": 0.5, "NQ": 0.25})
    assert overlay.apply_risk(weights, state).is_flat()

    # Even a recovery cannot un-latch the kill switch within the session.
    overlay.update_equity(1_000_000.0, timestamp=FIXED_TIMESTAMP)
    assert overlay.kill_switch_active is True
    assert overlay.apply_risk(weights, _state()).is_flat()

    overlay.reset_day(1_000_000.0, timestamp=FIXED_TIMESTAMP)
    assert overlay.kill_switch_active is False
    restored = overlay.apply_risk(weights, _state())
    assert restored["ES"] == pytest.approx(0.5)


def test_session_roll_clears_daily_kill_switch() -> None:
    overlay = _overlay(daily_max_loss_pct=0.03, trailing_stop_pct=0.99, drawdown_limit_pct=0.99)
    overlay.update_equity(1_000_000.0, timestamp=FIXED_TIMESTAMP)
    overlay.update_equity(960_000.0, timestamp=FIXED_TIMESTAMP)
    assert overlay.kill_switch_active is True
    next_session = _state(equity=1_000_000.0, timestamp=NEXT_DAY)
    restored = overlay.apply_risk(make_weight_vector({"ES": 0.5}), next_session)
    assert overlay.kill_switch_active is False
    assert restored["ES"] == pytest.approx(0.5)


def test_drawdown_limit_flattens_while_breached() -> None:
    overlay = _overlay(
        drawdown_limit_pct=0.20,
        trailing_stop_pct=0.99,
        daily_max_loss_pct=1.0,
    )
    overlay.update_equity(1_000_000.0, timestamp=FIXED_TIMESTAMP)
    overlay.update_equity(790_000.0, timestamp=FIXED_TIMESTAMP)
    weights = make_weight_vector({"ES": 0.5})
    assert overlay.apply_risk(weights, _state(equity=790_000.0)).is_flat()
    # A recovery back inside the drawdown limit re-enables risk taking.
    overlay.update_equity(850_000.0, timestamp=FIXED_TIMESTAMP)
    assert not overlay.apply_risk(weights, _state(equity=850_000.0)).is_flat()


def test_reset_day_requires_equity_value() -> None:
    overlay = QuantRiskOverlay(config=_config(), clock=lambda: FIXED_TIMESTAMP)
    with pytest.raises(ValueError, match="equity"):
        overlay.reset_day()


def test_invalid_equity_updates_rejected() -> None:
    overlay = _overlay()
    for bad in (0.0, -5.0, math.inf, math.nan):
        with pytest.raises(ValueError):
            overlay.update_equity(bad, timestamp=FIXED_TIMESTAMP)
    with pytest.raises(ValueError, match="timezone-aware"):
        overlay.update_equity(1_000_000.0, timestamp=datetime(2024, 1, 2, 21, 0))


def test_manual_reset_kill_switch() -> None:
    overlay = _overlay(daily_max_loss_pct=0.03, trailing_stop_pct=0.99, drawdown_limit_pct=0.99)
    overlay.update_equity(1_000_000.0, timestamp=FIXED_TIMESTAMP)
    overlay.update_equity(970_000.0, timestamp=FIXED_TIMESTAMP)
    assert overlay.kill_switch_active
    overlay.reset_kill_switch()
    assert not overlay.kill_switch_active


# ---------------------------------------------------------------------------
# Robustness / defensive paths
# ---------------------------------------------------------------------------


def test_default_constructor_and_portfolio_checks_without_equity() -> None:
    overlay = QuantRiskOverlay(volatility_targeting_enabled=False)
    assert overlay.config.target_annual_volatility == 0.12
    state = _state(equity=None)
    result = overlay.apply_risk(make_weight_vector({"ES": 0.5}), state)
    # Default config: single-instrument cap 0.5, sector cap 0.4.
    assert result["ES"] == pytest.approx(0.4)


def test_session_roll_without_equity_information_is_safe() -> None:
    overlay = _overlay()
    overlay.apply_risk(make_weight_vector({"ES": 0.4}), _state(equity=None))
    next_session = _state(equity=None, timestamp=NEXT_DAY)
    # No equity was ever observed: the session roll must simply not rebase.
    result = overlay.apply_risk(make_weight_vector({"ES": 0.4}), next_session)
    assert result["ES"] == pytest.approx(0.4)


def test_daily_loss_switch_can_be_disabled() -> None:
    overlay = _overlay(
        daily_loss_kill_switch_enabled=False,
        daily_max_loss_pct=0.03,
        trailing_stop_pct=0.99,
        drawdown_limit_pct=0.99,
    )
    overlay.update_equity(1_000_000.0, timestamp=FIXED_TIMESTAMP)
    overlay.update_equity(900_000.0, timestamp=FIXED_TIMESTAMP)
    assert overlay.kill_switch_active is False
    assert overlay.apply_risk(make_weight_vector({"ES": 0.5}), _state(equity=900_000.0))["ES"] == (
        pytest.approx(0.5)
    )


def test_constant_returns_produce_zero_volatility_and_no_scaling() -> None:
    overlay = _overlay(target_annual_volatility=0.12, volatility_targeting_enabled=True)
    state = _state(with_covariance=False)
    constant = {
        root: np.zeros(series.size, dtype=np.float64) for root, series in state.returns.items()
    }
    state = replace(state, returns=constant)
    result = overlay.apply_risk(make_weight_vector({"ES": 0.5}), state)
    assert result["ES"] == pytest.approx(0.5)


def test_zero_weight_entry_is_preserved_as_flat() -> None:
    overlay = _overlay()
    result = overlay.apply_risk(make_weight_vector({"ES": 0.0, "CL": 0.2}), _state())
    assert result["ES"] == 0.0
    assert result["CL"] == pytest.approx(0.2)


def test_root_without_market_snapshot_is_left_untouched() -> None:
    overlay = _overlay()
    state = MarketState(
        timestamp=FIXED_TIMESTAMP,
        instruments={"ES": make_snapshot("ES")},
        equity=1_000_000.0,
    )
    result = overlay.apply_risk(make_weight_vector({"ES": 0.2, "ZN": 0.1}), state)
    assert result["ES"] == pytest.approx(0.2)
    assert result["ZN"] == pytest.approx(0.1)


def test_covariance_quadratic_none_falls_back_to_ewma() -> None:
    overlay = _overlay(target_annual_volatility=0.12, volatility_targeting_enabled=True)
    state = _state()
    state = replace(
        state,
        covariance=np.zeros((5, 5), dtype=np.float64),
        covariance_instruments=("ES", "NQ", "CL", "GC", "ZN"),
    )
    series = np.asarray(state.returns["ES"], dtype=np.float64)
    sigma = _expected_ewma_vol(series, 20.0, 252)
    result = overlay.apply_risk(make_weight_vector({"ES": 1.0}), state)
    assert result["ES"] == pytest.approx(0.12 / sigma, rel=1e-12)


def test_non_psd_covariance_uses_diagonal_volatility_fallback() -> None:
    overlay = _overlay(target_annual_volatility=0.12, volatility_targeting_enabled=True)
    state = _state()
    covariance = np.array([[0.04, 0.10], [0.10, 0.04]], dtype=np.float64)
    state = replace(
        state,
        covariance=covariance,
        covariance_instruments=("ES", "NQ"),
        returns={},
    )
    # w^T Sigma w < 0 for opposing legs, so the diagonal fallback is used.
    expected_sigma = math.sqrt(0.25 * 0.04 + 0.25 * 0.04)
    result = overlay.apply_risk(make_weight_vector({"ES": 0.5, "NQ": -0.5}), state)
    # Single-instrument cap would clip; relax it.
    overlay = _overlay(
        target_annual_volatility=0.12,
        volatility_targeting_enabled=True,
        max_single_instrument_weight=10.0,
    )
    result = overlay.apply_risk(make_weight_vector({"ES": 0.5, "NQ": -0.5}), state)
    assert result["ES"] == pytest.approx(0.5 * 0.12 / expected_sigma)
    assert result["NQ"] == pytest.approx(-0.5 * 0.12 / expected_sigma)


def test_ewma_missing_return_series_falls_through_to_diagonal() -> None:
    overlay = _overlay(target_annual_volatility=0.12, volatility_targeting_enabled=True)
    state = _state(with_covariance=False)
    partial_returns = {"ES": state.returns["ES"]}
    state = replace(state, returns=partial_returns)
    result = overlay.apply_risk(make_weight_vector({"ES": 0.5, "NQ": 0.5}), state)
    assert result["ES"] == pytest.approx(0.5)
    assert result["NQ"] == pytest.approx(0.5)


def test_ewma_series_too_short_falls_through() -> None:
    overlay = _overlay(target_annual_volatility=0.12, volatility_targeting_enabled=True)
    state = _state(with_covariance=False)
    short_returns = {
        root: np.asarray(series, dtype=np.float64)[:1] for root, series in state.returns.items()
    }
    state = replace(state, returns=short_returns)
    result = overlay.apply_risk(make_weight_vector({"ES": 0.5}), state)
    assert result["ES"] == pytest.approx(0.5)


def test_diagonal_fallback_mixes_covariance_and_returns() -> None:
    overlay = _overlay(target_annual_volatility=0.12, volatility_targeting_enabled=True)
    state = _state()
    state = replace(
        state,
        covariance=np.array([[0.04]], dtype=np.float64),
        covariance_instruments=("ES",),
        returns={"NQ": state.returns["NQ"]},
    )
    nq_daily_var = float(np.var(np.asarray(state.returns["NQ"], dtype=np.float64), ddof=1))
    expected_sigma = math.sqrt(0.25 * 0.04 + 0.25 * nq_daily_var * 252.0)
    result = overlay.apply_risk(make_weight_vector({"ES": 0.5, "NQ": 0.5}), state)
    assert result["ES"] == pytest.approx(0.5 * 0.12 / expected_sigma)
    assert result["NQ"] == pytest.approx(0.5 * 0.12 / expected_sigma)


def test_diagonal_fallback_returns_none_when_a_vol_is_unknown() -> None:
    overlay = _overlay(target_annual_volatility=0.12, volatility_targeting_enabled=True)
    state = _state(with_covariance=False)
    state = replace(state, returns={"ES": state.returns["ES"]})
    result = overlay.apply_risk(make_weight_vector({"ES": 0.5, "ZN": 0.5}), state)
    assert result["ES"] == pytest.approx(0.5)
    assert result["ZN"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Configuration, protocol, determinism and state snapshot
# ---------------------------------------------------------------------------


def test_overlay_satisfies_risk_overlay_protocol() -> None:
    assert isinstance(_overlay(), RiskOverlay)


def test_config_from_risk_settings() -> None:
    settings = RiskSettings()
    config = QuantRiskConfig.from_risk_settings(settings)
    assert config.target_annual_volatility == settings.target_annual_volatility
    assert config.max_gross_leverage == settings.max_gross_leverage
    assert config.max_sector_weight == settings.max_sector_weight
    assert config.daily_max_loss_pct == settings.daily_max_loss_pct
    overlay = QuantRiskOverlay(config=settings)
    assert overlay.config.max_single_instrument_weight == settings.max_single_instrument_weight


def test_explicit_parameter_overrides_config() -> None:
    overlay = QuantRiskOverlay(
        QuantRiskConfig(target_annual_volatility=0.10),
        target_annual_volatility=0.20,
        max_gross_leverage=2.5,
    )
    assert overlay.config.target_annual_volatility == 0.20
    assert overlay.config.max_gross_leverage == 2.5


def test_invalid_config_object_rejected() -> None:
    with pytest.raises(TypeError, match="QuantRiskConfig"):
        QuantRiskOverlay(config=object())  # type: ignore[arg-type]


def test_invalid_pydantic_config_rejected() -> None:
    with pytest.raises(Exception, match="greater than 0"):
        QuantRiskConfig(target_annual_volatility=0.0)


def test_deterministic_results() -> None:
    state = _state()
    weights = make_weight_vector({"ES": 0.5, "NQ": -0.2, "CL": 0.1})
    first = _overlay().apply_risk(weights, state)
    second = _overlay().apply_risk(weights, state)
    assert first == second


def test_snapshot_state_is_json_serialisable() -> None:
    overlay = _overlay(daily_max_loss_pct=0.03)
    overlay.update_equity(1_000_000.0, timestamp=FIXED_TIMESTAMP)
    overlay.update_equity(970_000.0, timestamp=FIXED_TIMESTAMP)
    snapshot = overlay.snapshot_state()
    json.dumps(snapshot)  # must be strictly JSON-serialisable
    assert snapshot["kill_switch_active"] is True
    assert snapshot["starting_daily_equity"] == pytest.approx(1_000_000.0)
    assert snapshot["daily_pnl_pct"] == pytest.approx(-0.03)
    assert snapshot["session_date"] == "2024-01-02"


def test_reset_clears_all_state() -> None:
    overlay = _overlay(daily_max_loss_pct=0.03)
    overlay.update_equity(1_000_000.0, timestamp=FIXED_TIMESTAMP)
    overlay.update_equity(950_000.0, timestamp=FIXED_TIMESTAMP)
    overlay.reset()
    snapshot = overlay.snapshot_state()
    assert snapshot["equity"] is None
    assert snapshot["kill_switch_active"] is False
    assert overlay.kill_switch_active is False


def test_weights_are_reindexed_to_canonical_roots() -> None:
    overlay = _overlay(volatility_targeting_enabled=False)
    result = overlay.apply_risk(make_weight_vector({"es": 0.4, "cl": 0.2}), _state())
    assert set(result.as_dict()) == {"ES", "CL"}
    assert result["ES"] == pytest.approx(0.4)


def test_name_property_and_rounding_mode_import() -> None:
    overlay = _overlay()
    # Pipeline stage registry name: must match configs/strategies/*.yaml exactly.
    assert overlay.name == "quant_risk"
    assert RoundingMode.NEAREST.value == "nearest"
