"""Unit tests for the mean-reversion timing overlay (Phase-10 Part B1)."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from src.core.types import MarketState, WeightVector
from src.strategies._params import ParamError
from src.strategies.timing import (
    MeanReversionTiming,
    bollinger_bands,
    build_mean_reversion_timing,
    build_timing,
    wilder_rsi,
)

from tests.fixtures.market import make_market_state


def _constant_returns_state(daily: float, roots: tuple[str, ...] = ("ES",)) -> MarketState:
    state = make_market_state(roots=roots)
    return dataclasses.replace(
        state,
        returns={root: np.full(260, daily, dtype=np.float64) for root in state.instruments},
    )


def _state_with_features(
    roots: tuple[str, ...], features: dict[str, dict[str, float]]
) -> MarketState:
    return make_market_state(roots=roots, with_returns=False, features=features)


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def test_wilder_rsi_handles_trends_flats_and_short_history() -> None:
    rising = np.linspace(100.0, 200.0, 40)
    falling = np.linspace(200.0, 100.0, 40)
    flat = np.full(40, 100.0)
    assert wilder_rsi(rising) == pytest.approx(100.0)
    assert wilder_rsi(falling) == pytest.approx(0.0)
    assert wilder_rsi(flat) == pytest.approx(50.0)
    assert wilder_rsi(np.array([100.0, 101.0])) is None
    with pytest.raises(ValueError, match="period"):
        wilder_rsi(rising, period=1)


def test_bollinger_bands_math_and_short_history() -> None:
    window = np.arange(100.0, 125.0)
    last, lower, upper = bollinger_bands(window, 20, 2.0)
    assert last == pytest.approx(124.0)
    middle = float(np.mean(window[-20:]))
    deviation = float(np.std(window[-20:], ddof=0))
    assert lower == pytest.approx(middle - 2.0 * deviation)
    assert upper == pytest.approx(middle + 2.0 * deviation)
    assert bollinger_bands(window[:5], 20, 2.0) is None
    with pytest.raises(ValueError, match="window"):
        bollinger_bands(window, 1, 2.0)


# ---------------------------------------------------------------------------
# Signals / scaling
# ---------------------------------------------------------------------------


def test_oversold_state_boosts_longs_and_reduces_shorts() -> None:
    overlay = MeanReversionTiming()
    state = _constant_returns_state(-0.02)
    assert overlay.rsi(state, "ES") is not None
    assert overlay.rsi(state, "ES") < 25.0
    assert overlay.signal(state, "ES") == 1
    assert overlay.name == "mean_reversion"
    long_weights = overlay.adjust_timing(WeightVector({"ES": 1.0}), state)
    short_weights = overlay.adjust_timing(WeightVector({"ES": -1.0}), state)
    assert long_weights["ES"] == pytest.approx(1.5)
    assert short_weights["ES"] == pytest.approx(-0.5)
    assert overlay.scaling_factor(state, "ES", 0.0) == pytest.approx(1.0)


def test_overbought_state_reduces_longs_and_boosts_shorts() -> None:
    overlay = MeanReversionTiming()
    state = _constant_returns_state(0.02)
    assert overlay.rsi(state, "ES") > 75.0
    assert overlay.signal(state, "ES") == -1
    long_weights = overlay.adjust_timing(WeightVector({"ES": 1.0}), state)
    short_weights = overlay.adjust_timing(WeightVector({"ES": -1.0}), state)
    assert long_weights["ES"] == pytest.approx(0.5)
    assert short_weights["ES"] == pytest.approx(-1.5)


def test_turbulence_veto_freezes_the_overlay() -> None:
    overlay = MeanReversionTiming(turbulence_veto_threshold=0.95)
    base = _constant_returns_state(-0.02)
    vetoed = dataclasses.replace(
        base,
        instruments={
            "ES": dataclasses.replace(base.instruments["ES"], features={"turbulence": 0.99})
        },
    )
    assert overlay.signal(vetoed, "ES") == 0
    assert overlay.adjust_timing(WeightVector({"ES": 1.0}), vetoed)["ES"] == pytest.approx(1.0)
    # A higher threshold un-vetoes the same state.
    relaxed = MeanReversionTiming(turbulence_veto_threshold=1.5)
    assert relaxed.signal(vetoed, "ES") == 1


def test_default_fixture_is_bounded_and_deterministic() -> None:
    overlay = MeanReversionTiming()
    state = make_market_state()
    weights = WeightVector.equal_weight(state.ids, 1.0)
    first = overlay.adjust_timing(weights, state)
    second = overlay.adjust_timing(weights, state)
    assert first == second
    for root in state.ids:
        factor = first.get(root) / weights[root]
        assert 0.0 <= factor <= 1.0 + overlay.reversion_strength + 1e-12
        assert math.isfinite(first[root])


def test_feature_fallbacks_for_rsi_and_bollinger() -> None:
    overlay = MeanReversionTiming()
    cheap = _state_with_features(
        ("ES",), {"ES": {"rsi": 50.0, "bb_lower": 4_900.0, "bb_upper": 5_100.0, "turbulence": 0.1}}
    )
    # ES mark is 4,800 in the fixtures -> below the lower band -> oversold by price.
    assert overlay.signal(cheap, "ES") == 1
    rich = _state_with_features(
        ("ES",), {"ES": {"rsi": 50.0, "bb_lower": 4_000.0, "bb_upper": 4_100.0, "turbulence": 0.1}}
    )
    assert overlay.signal(rich, "ES") == -1
    neutral = _state_with_features(
        ("ES",), {"ES": {"rsi": 50.0, "bb_lower": 4_000.0, "bb_upper": 5_000.0, "turbulence": 0.1}}
    )
    assert overlay.signal(neutral, "ES") == 0
    rsi_only = _state_with_features(("ES",), {"ES": {"rsi": 90.0, "turbulence": 0.1}})
    assert overlay.signal(rsi_only, "ES") == -1


def test_signal_uses_only_point_in_time_state_fields() -> None:
    """The overlay reads only the state's trailing returns/features (no look-ahead)."""
    overlay = MeanReversionTiming()
    state = make_market_state(roots=("ES",), returns_length=260)
    # Equity/metadata/covariance must not influence the timing signal.
    other = dataclasses.replace(
        state,
        equity=5_000_000.0,
        covariance=None,
        covariance_instruments=(),
        metadata={"unrelated": 1.0},
    )
    assert overlay.signal(state, "ES") == overlay.signal(other, "ES")
    # The Bollinger bands depend only on the trailing window, never older bars.
    base = np.asarray(state.returns["ES"], dtype=np.float64)
    altered = np.concatenate([np.full(200, 0.5), base[-60:]])
    a = dataclasses.replace(state, returns={"ES": base})
    b = dataclasses.replace(state, returns={"ES": altered})
    bands_a = overlay.bands(a, "ES")
    bands_b = overlay.bands(b, "ES")
    assert bands_a is not None and bands_b is not None
    last_a, lower_a, upper_a = bands_a
    last_b, lower_b, upper_b = bands_b
    # The reconstructed price path differs by a constant scale factor; the band
    # geometry (and therefore the price<lower / price>upper signals) is identical.
    assert lower_a / last_a == pytest.approx(lower_b / last_b)
    assert upper_a / last_a == pytest.approx(upper_b / last_b)
    # A flat flat path produces no reversion action (neutral signal).
    flat = dataclasses.replace(state, returns={"ES": np.zeros(260, dtype=np.float64)})
    assert overlay.signal(flat, "ES") == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"rsi_period": 1}, "rsi_period"),
        ({"bollinger_window": 1}, "bollinger_window"),
        ({"oversold_threshold": -1.0}, "oversold_threshold"),
        ({"oversold_threshold": 60.0}, "oversold_threshold"),
        ({"overbought_threshold": 40.0}, "overbought_threshold"),
        ({"overbought_threshold": 120.0}, "overbought_threshold"),
        ({"bollinger_std": 0.0}, "bollinger_std"),
        ({"turbulence_veto_threshold": 0.0}, "turbulence_veto_threshold"),
        ({"reversion_strength": 1.5}, "reversion_strength"),
    ],
)
def test_mean_reversion_rejects_invalid_arguments(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        MeanReversionTiming(**kwargs)  # type: ignore[arg-type]


def test_build_mean_reversion_from_params_and_registry() -> None:
    overlay = build_timing(
        "mean_reversion",
        {
            "rsi_period": 10,
            "oversold_threshold": 30,
            "overbought_threshold": 70,
            "bollinger_window": 15,
            "bollinger_std": 1.5,
            "turbulence_veto_threshold": 2.0,
            "reversion_strength": 0.25,
        },
    )
    assert isinstance(overlay, MeanReversionTiming)
    assert overlay.name == "mean_reversion"
    state = _constant_returns_state(-0.02)
    assert overlay.signal(state, "ES") == 1
    assert overlay.adjust_timing(WeightVector({"ES": 1.0}), state)["ES"] == pytest.approx(1.25)
    assert isinstance(build_mean_reversion_timing({}), MeanReversionTiming)
    with pytest.raises(ParamError, match="unknown parameter"):
        build_timing("mean_reversion", {"rsi": 14})
