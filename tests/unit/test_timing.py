"""Unit tests for the timing / term-structure overlays (``T_t``)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from src.core.interfaces import TimingOverlay
from src.core.types import InstrumentId, MarketState, WeightVector
from src.strategies._params import ParamError
from src.strategies.timing import (
    TIMING_REGISTRY,
    CarryFilter,
    CompositeTiming,
    KAMATiming,
    TSMOMTiming,
    build_timing,
    kaufman_adaptive_moving_average,
)

from tests.fixtures.market import make_market_state


def _constant_returns(state: MarketState, daily: float = 0.005) -> MarketState:
    series = {root: np.full(260, daily, dtype=np.float64) for root in state.instruments}
    return dataclasses.replace(state, returns=series)


def _state_for(roots: tuple[str, ...], state: MarketState | None = None) -> MarketState:
    return state if state is not None else make_market_state(roots=roots)


class _RecorderOverlay:
    def __init__(self, name: str, factor: float, calls: list[str]) -> None:
        self._name = name
        self._factor = factor
        self._calls = calls

    @property
    def name(self) -> str:
        return self._name

    def adjust_timing(self, weights: WeightVector, state: MarketState) -> WeightVector:
        del state
        self._calls.append(self._name)
        return weights.scaled(self._factor)


# ---------------------------------------------------------------------------
# TSMOM
# ---------------------------------------------------------------------------


def test_tsmom_follows_agreeing_fast_and_slow_returns() -> None:
    state = _constant_returns(make_market_state(roots=("ES", "NQ")), daily=0.002)
    weights = WeightVector({"ES": 0.3, "NQ": 0.2})
    overlay = TSMOMTiming()
    timing = overlay.adjust_timing(weights, state)
    assert overlay.name == "tsmom"
    assert timing["ES"] == pytest.approx(0.3)
    assert timing["NQ"] == pytest.approx(0.2)
    downside = _constant_returns(make_market_state(roots=("ES", "NQ")), daily=-0.002)
    flipped = overlay.adjust_timing(weights, downside)
    assert flipped["ES"] == pytest.approx(-0.3)
    assert flipped["NQ"] == pytest.approx(-0.2)


def test_tsmom_disagreement_flattens_and_can_disable_flipping() -> None:
    state = make_market_state(roots=("ES",))
    rng = np.random.default_rng(0)
    values = rng.normal(0.0, 0.001, size=260)
    values[:239] = -0.005
    values[-21:] = 0.02
    disagreement = dataclasses.replace(state, returns={"ES": values})
    overlay = TSMOMTiming()
    assert overlay.momentum(disagreement, "ES")[0] > 0.0
    assert overlay.momentum(disagreement, "ES")[1] < 0.0
    assert overlay.adjust_timing(WeightVector({"ES": 1.0}), disagreement)["ES"] == 0.0
    no_flip = TSMOMTiming(allow_flip=False)
    downside = _constant_returns(state, daily=-0.002)
    assert no_flip.adjust_timing(WeightVector({"ES": 1.0}), downside)["ES"] == 0.0


def test_tsmom_volatility_scaling() -> None:
    state = make_market_state(roots=("ES", "NQ"))
    covariance = np.diag([0.04, 0.09])
    priced = dataclasses.replace(state, covariance=covariance, covariance_instruments=state.ids)
    rising = _constant_returns(priced, daily=0.002)
    overlay = TSMOMTiming(target_volatility=0.12)
    timing = overlay.adjust_timing(WeightVector({"ES": 1.0, "NQ": 1.0}), rising)
    assert timing["ES"] == pytest.approx(0.12 / 0.2)
    assert timing["NQ"] == pytest.approx(0.12 / 0.3)
    clamped = TSMOMTiming(target_volatility=1.0, max_scale=2.0)
    timing = clamped.adjust_timing(WeightVector({"ES": 1.0, "NQ": 1.0}), rising)
    assert timing["ES"] == pytest.approx(2.0)
    floored = TSMOMTiming(target_volatility=1e-6, min_scale=0.25)
    assert floored.scaling_factor(rising, "ES") == pytest.approx(0.25)


def test_tsmom_feature_fallbacks_and_empty_weights() -> None:
    state = make_market_state(
        roots=("ES", "NQ"),
        with_returns=False,
        features={
            "ES": {"tsmom_21": 0.05, "tsmom_126": 0.10, "ret_21d": 0.05, "ret_126d": 0.10},
            "NQ": {"tsmom_21": -0.05, "tsmom_126": -0.10, "ret_21d": -0.05, "ret_126d": -0.10},
        },
    )
    overlay = TSMOMTiming()
    timing = overlay.adjust_timing(WeightVector({"ES": 1.0, "NQ": 1.0}), state)
    assert timing["ES"] == pytest.approx(1.0)
    assert timing["NQ"] == pytest.approx(-1.0)
    assert overlay.adjust_timing(WeightVector.empty(), state) == WeightVector.empty()


def test_tsmom_without_any_history_is_neutral() -> None:
    state = make_market_state(roots=("ES",), with_returns=False)
    blanked = dataclasses.replace(
        state,
        instruments={
            root: dataclasses.replace(snapshot, features={})
            for root, snapshot in state.instruments.items()
        },
    )
    overlay = TSMOMTiming()
    assert overlay.momentum(blanked, "ES") == (0.0, 0.0)
    assert overlay.adjust_timing(WeightVector({"ES": 1.0}), blanked)["ES"] == 0.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"lookback_fast": 0}, "lookback"),
        ({"lookback_slow": 0}, "lookback"),
        ({"lookback_fast": 50, "lookback_slow": 50}, "lookback_fast"),
        ({"vol_lookback": 1}, "vol_lookback"),
        ({"target_volatility": 0.0}, "target_volatility"),
        ({"min_scale": -0.1}, "min_scale"),
        ({"min_scale": 2.0, "max_scale": 1.0}, "max_scale"),
    ],
)
def test_tsmom_rejects_invalid_arguments(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        TSMOMTiming(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# KAMA
# ---------------------------------------------------------------------------


def test_kaufman_adaptive_moving_average_behaviour() -> None:
    rising = np.linspace(100.0, 200.0, 60, dtype=np.float64)
    kama = kaufman_adaptive_moving_average(rising)
    assert kama.shape == rising.shape
    assert kama[-1] < rising[-1]
    assert kama[-1] > rising[0]
    assert np.all(np.diff(kama) >= 0.0)
    assert kaufman_adaptive_moving_average(np.zeros(0)).size == 0
    short = kaufman_adaptive_moving_average(np.array([1.0, 2.0]), window=10)
    assert np.allclose(short, np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="window"):
        kaufman_adaptive_moving_average(rising, window=0)
    with pytest.raises(ValueError, match="fast_period"):
        kaufman_adaptive_moving_average(rising, fast_period=0)
    with pytest.raises(ValueError, match="fast_period must be smaller"):
        kaufman_adaptive_moving_average(rising, fast_period=30, slow_period=10)


def test_kama_timing_keeps_uptrends_and_flattens_downtrends() -> None:
    state = make_market_state(roots=("ES",))
    overlay = KAMATiming()
    uptrend = _constant_returns(state, daily=0.03)
    upside = overlay.adjust_timing(WeightVector({"ES": 1.0}), uptrend)["ES"]
    assert upside > 0.9
    downtrend = _constant_returns(state, daily=-0.03)
    assert overlay.adjust_timing(WeightVector({"ES": 1.0}), downtrend)["ES"] == pytest.approx(0.0)
    flipping = KAMATiming(allow_flip=True)
    assert flipping.adjust_timing(WeightVector({"ES": 1.0}), downtrend)["ES"] < 0.0
    assert overlay.name == "kama"


def test_kama_timing_feature_fallback_and_no_data() -> None:
    state = make_market_state(
        roots=("ES",),
        with_returns=False,
        features={"ES": {"ret_126d": 0.5}},
    )
    assert KAMATiming().scaling_factor(state, "ES") == pytest.approx(1.0)
    bearish = make_market_state(
        roots=("ES",), with_returns=False, features={"ES": {"ret_126d": -0.5}}
    )
    assert KAMATiming().scaling_factor(bearish, "ES") == pytest.approx(0.0)
    blanked = dataclasses.replace(
        state,
        instruments={"ES": dataclasses.replace(state.instruments["ES"], features={})},
        returns={},
    )
    assert KAMATiming().agreement(blanked, "ES") == 0.0
    assert KAMATiming().adjust_timing(WeightVector.empty(), state) == WeightVector.empty()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"window": 0}, "window"),
        ({"fast_period": 0}, "fast_period"),
        ({"slow_period": 0}, "slow_period"),
        ({"fast_period": 40, "slow_period": 10}, "fast_period"),
        ({"trend_band": 0.0}, "trend_band"),
        ({"feature_band": 0.0}, "feature_band"),
    ],
)
def test_kama_rejects_invalid_arguments(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        KAMATiming(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Carry filter
# ---------------------------------------------------------------------------


def test_carry_filter_zeroes_negative_carry_legs() -> None:
    state = make_market_state(roots=("ES", "CL"))
    overlay = CarryFilter()
    weights = WeightVector({"ES": 1.0, "CL": 1.0})
    timing = overlay.adjust_timing(weights, state)
    assert overlay.name == "carry_filter"
    assert overlay.mode == "zero"
    # Fixture: equity index pays positive annualised carry, energy is in contango.
    assert overlay.carry_bps(state, "ES") > 0.0
    assert overlay.carry_bps(state, "CL") < 0.0
    assert timing["ES"] == pytest.approx(1.5)
    assert timing["CL"] == pytest.approx(0.0)


def test_carry_filter_scale_mode_and_threshold() -> None:
    state = make_market_state(roots=("ES",))
    scaled = CarryFilter(mode="scale", carry_weight=1.0, carry_scale_bps=100.0)
    assert scaled.scaling_factor(state, "ES") == pytest.approx(1.5)
    mildly_negative = dataclasses.replace(
        state,
        instruments={
            "ES": dataclasses.replace(
                state.instruments["ES"], features={"carry_annualized": -50e-4}
            )
        },
    )
    assert scaled.scaling_factor(mildly_negative, "ES") == pytest.approx(
        1.0 + np.tanh(-0.5), rel=1e-6
    )
    assert CarryFilter().scaling_factor(mildly_negative, "ES") == 0.0
    strict = CarryFilter(min_carry_bps=10_000.0)
    assert strict.scaling_factor(state, "ES") == 0.0
    neutral_weight = CarryFilter(carry_weight=0.0)
    assert neutral_weight.scaling_factor(state, "ES") == pytest.approx(1.0)


def test_carry_filter_term_slope_fallback_and_missing_features() -> None:
    state = make_market_state(roots=("ES",))
    slope_only = dataclasses.replace(
        state,
        instruments={
            "ES": dataclasses.replace(state.instruments["ES"], features={"term_slope": 0.01})
        },
    )
    assert CarryFilter().carry_bps(slope_only, "ES") == pytest.approx(365.0 / 30.0 * 1e4 * 0.01)
    stripped = dataclasses.replace(
        state,
        instruments={"ES": dataclasses.replace(state.instruments["ES"], features={})},
    )
    assert CarryFilter().carry_bps(stripped, "ES") is None
    assert CarryFilter().scaling_factor(stripped, "ES") == pytest.approx(1.0)
    assert CarryFilter().carry_bps(state, "UNKNOWN") is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"carry_weight": -1.0}, "carry_weight"),
        ({"carry_scale_bps": 0.0}, "carry_scale_bps"),
        ({"max_scale": 0.5}, "max_scale"),
        ({"mode": "flip"}, "carry filter mode"),
    ],
)
def test_carry_filter_rejects_invalid_arguments(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        CarryFilter(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Composite timing + registry
# ---------------------------------------------------------------------------


def test_composite_timing_applies_components_in_order() -> None:
    calls: list[str] = []
    state = make_market_state()
    overlay = CompositeTiming(
        [_RecorderOverlay("first", 2.0, calls), _RecorderOverlay("second", 0.5, calls)]
    )
    assert overlay.name == "composite"
    assert overlay.adjust_timing(WeightVector({"ES": 1.0}), state)["ES"] == pytest.approx(1.0)
    assert calls == ["first", "second"]
    assert len(overlay.components) == 2
    with pytest.raises(ValueError, match="at least one"):
        CompositeTiming([])
    with pytest.raises(TypeError, match="TimingOverlay protocol"):
        CompositeTiming([object()])  # type: ignore[list-item]


def test_build_timing_registry_and_yaml_params() -> None:
    assert set(TIMING_REGISTRY) == {"tsmom", "kama", "carry_filter", "composite"}
    state = make_market_state()
    weights = WeightVector.equal_weight(state.ids, 1.0)
    tsmom = build_timing("tsmom", {"lookback_fast": 21, "lookback_slow": 126, "vol_lookback": 20})
    assert isinstance(tsmom, TSMOMTiming)
    assert isinstance(tsmom.adjust_timing(weights, state), WeightVector)
    kama = build_timing("kama", {"window": 10, "fast_period": 2, "slow_period": 30})
    assert isinstance(kama, KAMATiming)
    carry = build_timing("carry_filter", {"min_carry_bps": 0.0, "carry_weight": 1.0})
    assert isinstance(carry, CarryFilter)
    assert isinstance(carry, TimingOverlay)
    composite = build_timing("composite", {"components": "tsmom,kama"})
    assert isinstance(composite, CompositeTiming)
    assert isinstance(composite.adjust_timing(weights, state), WeightVector)
    with pytest.raises(ValueError, match="unknown timing overlay"):
        build_timing("does_not_exist")
    with pytest.raises(ParamError, match="unknown parameter"):
        build_timing("tsmom", {"lookbacks": 21})
    with pytest.raises(ValueError, match="components"):
        build_timing("composite", {})
    with pytest.raises(ValueError, match="cannot contain a composite"):
        build_timing("composite", {"components": "composite"})


def test_timing_overlays_never_emit_non_finite_weights() -> None:
    state = make_market_state()
    weights = WeightVector({"ES": 0.1, "NQ": -0.2, "CL": 0.0, "GC": 0.05, "ZN": -0.05})
    for name, params in (
        ("tsmom", {}),
        ("kama", {}),
        ("carry_filter", {}),
        ("composite", {"components": "tsmom,kama,carry_filter"}),
    ):
        adjusted = build_timing(name, params).adjust_timing(weights, state)
        assert all(np.isfinite(value) for value in adjusted.as_dict().values())
        assert set(adjusted.as_dict()) <= set(weights.as_dict())


def test_instrument_id_type_alias_is_used() -> None:
    root: InstrumentId = "ES"
    assert root in make_market_state().instruments
