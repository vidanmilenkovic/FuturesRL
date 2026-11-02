"""Unit tests for the universe selectors (``S_t``)."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np
import pytest
from configs.instruments import InstrumentsConfig
from src.core.interfaces import UniverseSelector
from src.core.types import InstrumentId, MarketState, WeightVector
from src.strategies._params import ParamError
from src.strategies.selectors import (
    SELECTOR_REGISTRY,
    CompositeSelector,
    LiquiditySelector,
    RegimeSelector,
    build_selector,
)

from tests.fixtures.market import make_market_state


def _returns_state(
    state: MarketState,
    series: dict[InstrumentId, np.ndarray[tuple[int], np.dtype[np.float64]]],
) -> MarketState:
    return dataclasses.replace(state, returns=series)


def _negative_returns(n: int, *, seed: int = 3) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    rng = np.random.default_rng(seed)
    return rng.normal(-0.01, 0.004, size=n)


class _FixedSelector:
    """Minimal structural UniverseSelector used to test composition."""

    def __init__(self, name: str, instruments: Sequence[InstrumentId]) -> None:
        self._name = name
        self._instruments = tuple(instruments)

    @property
    def name(self) -> str:
        return self._name

    def select(self, state: MarketState) -> Sequence[InstrumentId]:
        del state
        return self._instruments


# ---------------------------------------------------------------------------
# Liquidity selector
# ---------------------------------------------------------------------------


def test_liquidity_selector_ranks_by_dollar_volume() -> None:
    state = make_market_state()
    selector = LiquiditySelector()
    selected = selector.select(state)
    dollar_volumes = selector.dollar_volume(state)
    assert set(selected) == set(state.instruments)
    assert [dollar_volumes[root] for root in selected] == sorted(
        dollar_volumes.values(), reverse=True
    )
    # NQ (~4.2bn) is the largest dollar volume in the fixture, CL (~18m) the smallest.
    assert selected[0] == "NQ"
    assert selected[-1] == "CL"


def test_liquidity_selector_applies_filters_and_top_n() -> None:
    state = make_market_state()
    selector = LiquiditySelector(
        min_avg_dollar_volume=50_000_000.0,
        min_open_interest=1_000_000.0,
        top_n=2,
        min_history_days=100,
    )
    selected = selector.select(state)
    assert len(selected) == 2
    assert selected == ["NQ", "ES"]


def test_liquidity_selector_min_history_days() -> None:
    state = make_market_state(returns_length=40)
    assert LiquiditySelector(min_history_days=252).select(state) == []
    assert len(LiquiditySelector(min_history_days=10).select(state)) == len(state.instruments)
    # A missing return series means "unknown history" and is not filtered out.
    no_returns = make_market_state(with_returns=False)
    assert len(LiquiditySelector(min_history_days=252).select(no_returns)) == len(
        no_returns.instruments
    )


def test_liquidity_selector_is_deterministic_with_ties() -> None:
    state = make_market_state()
    # Equalise volume, open interest *and* the mark so the ranking falls through
    # to the root symbol tie-break and stays fully deterministic.
    equalised = dataclasses.replace(
        state,
        instruments={
            root: dataclasses.replace(
                snapshot,
                price=100.0,
                bid=99.99,
                ask=100.01,
                volume=1_000.0,
                open_interest=5_000.0,
            )
            for root, snapshot in state.instruments.items()
        },
    )
    selector = LiquiditySelector()
    assert selector.select(equalised) == sorted(equalised.instruments)
    assert selector.select(equalised) == selector.select(equalised)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_avg_dollar_volume": -1.0}, "min_avg_dollar_volume"),
        ({"min_open_interest": -1.0}, "min_open_interest"),
        ({"top_n": 0}, "top_n"),
        ({"min_history_days": -1}, "min_history_days"),
    ],
)
def test_liquidity_selector_rejects_invalid_arguments(
    kwargs: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        LiquiditySelector(**kwargs)  # type: ignore[arg-type]


def test_build_liquidity_selector_from_yaml_params(instrument_master: InstrumentsConfig) -> None:
    selector = build_selector(
        "liquidity",
        {
            "min_avg_dollar_volume": 50_000_000.0,
            "min_open_interest": 5_000.0,
            "top_n": 10,
            "min_history_days": 252,
        },
        instrument_master,
    )
    assert isinstance(selector, LiquiditySelector)
    assert isinstance(selector, UniverseSelector)
    assert selector.name == "liquidity"
    assert selector.select(make_market_state()) == ["NQ", "ES", "GC"]


def test_build_selector_rejects_unknown_name_and_params(
    instrument_master: InstrumentsConfig,
) -> None:
    with pytest.raises(ValueError, match="unknown universe selector"):
        build_selector("does_not_exist", {}, instrument_master)
    with pytest.raises(ParamError, match="unknown parameter"):
        build_selector("liquidity", {"topn": 3}, instrument_master)
    with pytest.raises(ParamError, match="expected a number"):
        build_selector("liquidity", {"min_avg_dollar_volume": "not-a-number"}, instrument_master)


def test_selector_registry_contents() -> None:
    assert set(SELECTOR_REGISTRY) == {"liquidity", "regime", "composite"}


# ---------------------------------------------------------------------------
# Regime selector
# ---------------------------------------------------------------------------


def test_regime_selector_neutral_fixture_is_not_risk_off() -> None:
    selector = RegimeSelector()
    diagnosis = selector.diagnose(make_market_state())
    assert diagnosis.risk_off is False
    assert diagnosis.slow_risk_off is False
    assert diagnosis.fast_risk_off is False
    assert 0.0 <= diagnosis.breadth_negative <= 1.0
    assert np.isfinite(diagnosis.macro_volatility)
    state = make_market_state()
    selected = selector.select(state)
    assert set(selected) == set(state.instruments)
    assert selector.select(state) == selected


def test_regime_selector_detects_slow_risk_off() -> None:
    state = make_market_state(roots=("ES", "NQ", "ZN"))
    series = {
        root: _negative_returns(260, seed=index) for index, root in enumerate(state.instruments)
    }
    shocked = _returns_state(state, series)
    selector = RegimeSelector(macro_vol_threshold=0.10, breadth_threshold=0.6, min_universe=2)
    diagnosis = selector.diagnose(shocked)
    assert diagnosis.slow_risk_off is True
    assert diagnosis.risk_off is True
    # All trends are negative: the selector de-risks to the least-negative roots.
    assert len(selector.select(shocked)) == 2


def test_regime_selector_detects_fast_three_day_shock() -> None:
    state = make_market_state(roots=("ES", "NQ", "ZN"))
    rng = np.random.default_rng(11)
    series = {}
    for root in state.instruments:
        values = rng.normal(0.001, 0.002, size=260)
        values[-3:] = -0.05
        series[root] = values
    shocked = _returns_state(state, series)
    selector = RegimeSelector()
    diagnosis = selector.diagnose(shocked)
    assert diagnosis.fast_risk_off is True
    assert diagnosis.mean_shock < 0.0
    assert diagnosis.shock_zscore < -3.0
    assert diagnosis.risk_off is True


def test_regime_selector_forced_risk_off_via_regime_and_metadata() -> None:
    state = make_market_state()
    selector = RegimeSelector()
    assert selector.diagnose(dataclasses.replace(state, regime="risk_off")).risk_off is True
    assert selector.diagnose(dataclasses.replace(state, regime="crisis")).forced_risk_off is True
    assert selector.diagnose(dataclasses.replace(state, regime="neutral")).forced_risk_off is False
    locked = dataclasses.replace(state, metadata={**dict(state.metadata), "macro_lockout": True})
    assert selector.diagnose(locked).forced_risk_off is True


def test_regime_selector_excludes_risk_off_roots_and_sectors(
    instrument_master: InstrumentsConfig,
) -> None:
    state = dataclasses.replace(make_market_state(), regime="risk_off")
    top_level = RegimeSelector(risk_off_roots=("CL",), filter_negative_trend=False)
    assert "CL" not in top_level.select(state)
    sector_based = RegimeSelector(
        instrument_master=instrument_master,
        risk_off_sectors=("energy",),
        filter_negative_trend=False,
    )
    assert "CL" not in sector_based.select(state)
    everything = RegimeSelector(
        risk_off_roots=tuple(state.instruments), filter_negative_trend=False
    )
    assert everything.select(state) == []


def test_regime_selector_min_universe_and_top_n() -> None:
    state = dataclasses.replace(make_market_state(), regime="risk_off")
    selector = RegimeSelector(filter_negative_trend=True, min_universe=3, top_n=2)
    selected = selector.select(state)
    assert len(selected) == 2  # truncated by top_n


def test_regime_selector_require_both() -> None:
    state = make_market_state(roots=("ES", "NQ", "ZN"))
    series = {
        root: _negative_returns(260, seed=index) for index, root in enumerate(state.instruments)
    }
    shocked = _returns_state(state, series)
    selector = RegimeSelector(
        macro_vol_threshold=0.10, shock_return_threshold=0.10, require_both=True
    )
    diagnosis = selector.diagnose(shocked)
    # Slow risk-off is true, but the 3-day shock is not: require_both keeps the regime neutral.
    assert diagnosis.slow_risk_off is True
    assert diagnosis.fast_risk_off is False
    assert diagnosis.risk_off is False


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"trend_lookback_days": 1}, "trend_lookback_days"),
        ({"shock_lookback_days": 0}, "shock_lookback_days"),
        ({"vol_lookback_days": 2}, "vol_lookback_days"),
        ({"shock_z_threshold": 0.0}, "shock_z_threshold"),
        ({"shock_return_threshold": -0.1}, "shock_return_threshold"),
        ({"macro_vol_threshold": 0.0}, "macro_vol_threshold"),
        ({"breadth_threshold": 1.5}, "breadth_threshold"),
        ({"min_universe": 0}, "min_universe"),
        ({"top_n": 0}, "top_n"),
    ],
)
def test_regime_selector_rejects_invalid_arguments(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        RegimeSelector(**kwargs)  # type: ignore[arg-type]


def test_build_regime_selector_from_params(instrument_master: InstrumentsConfig) -> None:
    selector = build_selector(
        "regime",
        {
            "trend_lookback_days": 126,
            "shock_z_threshold": 2.5,
            "risk_off_roots": "CL,GC",
            "risk_off_sectors": "energy",
            "filter_negative_trend": "false",
            "top_n": 4,
        },
        instrument_master,
    )
    assert isinstance(selector, RegimeSelector)
    state = dataclasses.replace(make_market_state(), regime="risk_off")
    selected = selector.select(state)
    assert "CL" not in selected
    assert "GC" not in selected
    assert len(selected) <= 4
    with pytest.raises(ParamError, match="unknown parameter"):
        build_selector("regime", {"shockz": 3.0}, instrument_master)


# ---------------------------------------------------------------------------
# Composite selector
# ---------------------------------------------------------------------------


def _composite_fixture() -> MarketState:
    return make_market_state()


def test_composite_selector_intersection_and_order() -> None:
    state = _composite_fixture()
    first = _FixedSelector("first", ("ES", "NQ", "CL"))
    second = _FixedSelector("second", ("NQ", "CL", "ZN"))
    composite = CompositeSelector([first, second])
    assert composite.mode == "intersection"
    assert composite.select(state) == ["NQ", "CL"]
    ordered = CompositeSelector([first, second], order=("CL", "NQ", "ES"))
    assert ordered.select(state) == ["CL", "NQ"]


def test_composite_selector_union() -> None:
    state = _composite_fixture()
    first = _FixedSelector("first", ("ES", "NQ"))
    second = _FixedSelector("second", ("NQ", "CL"))
    union = CompositeSelector([first, second], mode="union")
    assert union.select(state) == ["ES", "NQ", "CL"]
    assert union.components == (first, second)


def test_composite_selector_validation() -> None:
    with pytest.raises(ValueError, match="at least one"):
        CompositeSelector([])
    with pytest.raises(ValueError, match="composite mode"):
        CompositeSelector([_FixedSelector("a", ("ES",))], mode="xor")
    with pytest.raises(TypeError, match="UniverseSelector protocol"):
        CompositeSelector([object()])  # type: ignore[list-item]


def test_build_composite_selector(instrument_master: InstrumentsConfig) -> None:
    selector = build_selector(
        "composite",
        {"components": "liquidity,regime", "mode": "intersection", "order": "ES,NQ,CL,GC,ZN"},
        instrument_master,
    )
    assert isinstance(selector, CompositeSelector)
    state = make_market_state()
    selected = selector.select(state)
    liquidity = build_selector("liquidity", {}, instrument_master).select(state)
    assert set(selected) == set(liquidity)
    assert selected == [root for root in ("ES", "NQ", "CL", "GC", "ZN") if root in set(liquidity)]
    with pytest.raises(ValueError, match="components"):
        build_selector("composite", {}, instrument_master)
    with pytest.raises(ValueError, match="cannot contain a composite"):
        build_selector("composite", {"components": "composite"}, instrument_master)


def test_composite_selector_weight_vector_type_is_unrelated() -> None:
    # Sanity: selectors never return WeightVector instances (contract check).
    assert not isinstance(_FixedSelector("a", ("ES",)).select(_composite_fixture()), WeightVector)
