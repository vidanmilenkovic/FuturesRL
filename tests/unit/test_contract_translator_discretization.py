"""Unit tests for :meth:`ContractTranslator.discretization_report`.

The expectations are hand-computed from the committed instrument master and the
deterministic fixtures:

* ES = 4,800 x 50 = 240,000 USD per contract; MES = 4,800 x 5 = 24,000 USD.
* NQ = 16,800 x 20 = 336,000 USD per contract; MNQ = 16,800 x 2 = 33,600 USD.
* ZN = 111.5 x 1,000 = 111,500 USD per contract (no micro pair).

Every test is offline and deterministic.
"""

from __future__ import annotations

import dataclasses
import math

import pytest
from configs.instruments import load_instruments
from hypothesis import given, settings
from hypothesis import strategies as st
from src.core.types import (
    ContractTarget,
    ContractTargetSet,
    MarketState,
    RoundingMode,
    WeightVector,
)
from src.risk.contract_translator import (
    ContractTranslator,
    DiscretizationReport,
    MicroRoute,
)

from tests.fixtures.market import (
    make_contract_targets,
    make_market_state,
    make_weight_vector,
)

#: Contract notionals implied by the committed instrument master.
ES_CONTRACT_USD: float = 240_000.0
MES_CONTRACT_USD: float = 24_000.0
NQ_CONTRACT_USD: float = 336_000.0
MNQ_CONTRACT_USD: float = 33_600.0
ZN_CONTRACT_USD: float = 111_500.0


def _translator(**kwargs: object) -> ContractTranslator:
    defaults: dict[str, object] = {"rounding_mode": RoundingMode.NEAREST}
    defaults.update(kwargs)
    return ContractTranslator(**defaults)  # type: ignore[arg-type]


def _state(
    roots: tuple[str, ...] = ("ES", "NQ", "ZN"),
    *,
    equity: float | None = 1_000_000.0,
    account_currency: str = "USD",
    fx_rates: dict[str, float] | None = None,
) -> MarketState:
    return make_market_state(
        roots=roots,
        equity=equity,
        with_covariance=False,
        with_returns=False,
        account_currency=account_currency,
        fx_rates=fx_rates,
    )


def _report(
    weight_map: dict[str, float],
    *,
    roots: tuple[str, ...] = ("ES", "NQ", "ZN"),
    equity: float = 1_000_000.0,
    translator: ContractTranslator | None = None,
    account_currency: str = "USD",
    fx_rates: dict[str, float] | None = None,
) -> tuple[ContractTranslator, MarketState, WeightVector, ContractTargetSet, DiscretizationReport]:
    engine = translator if translator is not None else _translator()
    state = _state(roots, equity=equity, account_currency=account_currency, fx_rates=fx_rates)
    weights = make_weight_vector(weight_map)
    targets = engine.translate(weights, state)
    report = engine.discretization_report(weights, targets, state)
    return engine, state, weights, targets, report


# ---------------------------------------------------------------------------
# Exact hand-computed tracking error
# ---------------------------------------------------------------------------


def test_all_standard_exact_replication_has_zero_tracking_error() -> None:
    """Whole standard contracts replicate the target exactly: tracking error 0."""
    _, _, _, targets, report = _report({"ES": 0.24, "ZN": 0.1115})
    assert targets.quantities == {"ES": 1, "ZN": 1}
    assert report.tracking_error == pytest.approx(0.0, abs=1e-15)
    assert report.tracking_error_pct == pytest.approx(0.0, abs=1e-13)
    assert dict(report.target_weights) == pytest.approx({"ES": 0.24, "ZN": 0.1115})
    assert dict(report.achieved_weights) == pytest.approx({"ES": 0.24, "ZN": 0.1115})
    assert report.micro_routes == ()
    # (15,000 ES + 2,200 ZN initial margin) / 1,000,000.
    assert report.margin_utilization == pytest.approx(17_200.0 / 1_000_000.0)


def test_standard_plus_micro_residual_hand_computed() -> None:
    """ES 0.30 -> 1 ES + 3 MES: achieved 0.24 + 0.072, tracking error 0.066."""
    _, _, _, targets, report = _report({"ES": 0.30})
    assert targets.quantities == {"ES": 1, "MES": 3}
    assert dict(report.target_weights) == pytest.approx({"ES": 0.30, "MES": 0.0})
    assert dict(report.achieved_weights) == pytest.approx(
        {"ES": 240_000.0 / 1_000_000.0, "MES": 3 * MES_CONTRACT_USD / 1_000_000.0}
    )
    expected_error = (abs(0.30 - 0.24) + abs(0.0 - 0.072)) / 2.0
    assert report.tracking_error == pytest.approx(expected_error)
    assert report.tracking_error_pct == pytest.approx(expected_error * 100.0)
    assert report.micro_routes == (
        MicroRoute(
            family="ES",
            standard_root="ES",
            standard_quantity=1,
            micro_root="MES",
            micro_quantity=3,
        ),
    )
    assert report.margin_utilization == pytest.approx((15_000.0 + 3 * 1_500.0) / 1_000_000.0)


def test_floor_vs_nearest_rounding_changes_tracking_error() -> None:
    """At 200k equity ES 0.30 needs 2 (FLOOR) or 3 (NEAREST) MES contracts."""
    floor_engine = _translator(rounding_mode=RoundingMode.FLOOR)
    _, _, _, floor_targets, floor_report = _report(
        {"ES": 0.30}, roots=("ES",), equity=200_000.0, translator=floor_engine
    )
    assert floor_targets.quantities == {"MES": 2}
    # target 0.30 vs achieved 2 x 24,000 / 200,000 = 0.24 -> (0.30 + 0.24) / 2.
    assert floor_report.tracking_error == pytest.approx(0.27)
    assert floor_report.micro_routes[0].standard_quantity == 0
    assert floor_report.micro_routes[0].micro_quantity == 2

    nearest_engine = _translator(rounding_mode=RoundingMode.NEAREST)
    _, _, _, nearest_targets, nearest_report = _report(
        {"ES": 0.30}, roots=("ES",), equity=200_000.0, translator=nearest_engine
    )
    assert nearest_targets.quantities == {"MES": 3}
    # target 0.30 vs achieved 3 x 24,000 / 200,000 = 0.36 -> (0.30 + 0.36) / 2.
    assert nearest_report.tracking_error == pytest.approx(0.33)
    assert nearest_report.tracking_error > floor_report.tracking_error


def test_short_weights_report_negative_achieved_legs() -> None:
    """A short target produces negative achieved weights and same-size error."""
    _, _, _, targets, report = _report({"ES": -0.30})
    assert targets.quantities == {"ES": -1, "MES": -3}
    assert dict(report.target_weights) == pytest.approx({"ES": -0.30, "MES": 0.0})
    assert dict(report.achieved_weights) == pytest.approx({"ES": -0.24, "MES": -0.072})
    assert report.tracking_error == pytest.approx((0.06 + 0.072) / 2.0)
    route = report.micro_routes[0]
    assert route.standard_quantity == -1
    assert route.micro_quantity == -3


def test_both_standard_and_micro_weighted_family_report() -> None:
    """ES 0.30 + MES 0.10 nets to 1 ES + 7 MES: tracking error 0.064."""
    _, _, _, targets, report = _report({"ES": 0.30, "MES": 0.10})
    assert targets.quantities == {"ES": 1, "MES": 7}
    assert dict(report.target_weights) == pytest.approx({"ES": 0.30, "MES": 0.10})
    assert dict(report.achieved_weights) == pytest.approx({"ES": 0.24, "MES": 0.168})
    assert report.tracking_error == pytest.approx((0.06 + 0.068) / 2.0)
    assert report.micro_routes == (
        MicroRoute(
            family="ES",
            standard_root="ES",
            standard_quantity=1,
            micro_root="MES",
            micro_quantity=7,
        ),
    )
    assert report.margin_utilization == pytest.approx((15_000.0 + 7 * 1_500.0) / 1_000_000.0)


def test_micro_only_target_is_routed_and_has_native_family() -> None:
    """An explicit MES weight is reported as a family route with zero standard leg."""
    _, _, _, targets, report = _report({"MES": 0.072})
    assert targets.quantities == {"MES": 3}
    assert report.tracking_error == pytest.approx(0.0, abs=1e-15)
    assert report.micro_routes == (
        MicroRoute(
            family="ES",
            standard_root="ES",
            standard_quantity=0,
            micro_root="MES",
            micro_quantity=3,
        ),
    )
    assert report.margin_utilization == pytest.approx(3 * 1_500.0 / 1_000_000.0)


def test_micro_routing_lead_repro_nq_below_one_contract() -> None:
    """NQ 0.20 at 1M routes to 6 MNQ: target 0.20 vs achieved 0.2016."""
    _, _, _, targets, report = _report({"NQ": 0.20}, roots=("NQ",))
    assert targets.quantities == {"MNQ": 6}
    assert dict(report.target_weights) == pytest.approx({"NQ": 0.20, "MNQ": 0.0})
    assert dict(report.achieved_weights) == pytest.approx(
        {"NQ": 0.0, "MNQ": 6 * MNQ_CONTRACT_USD / 1_000_000.0}
    )
    assert report.tracking_error == pytest.approx((0.20 + 0.2016) / 2.0)
    assert report.micro_routes[0].standard_quantity == 0


def test_eur_account_conversion_uses_usd_equity() -> None:
    """100,000 EUR at 1.20 = 120,000 USD sized equity for a EUR-based report."""
    translator = _translator(
        rounding_mode=RoundingMode.FLOOR, account_currency="EUR", fx_rates={"EUR": 1.2}
    )
    _, _, _, targets, report = _report(
        {"ES": 2.5},
        roots=("ES",),
        equity=100_000.0,
        translator=translator,
        account_currency="EUR",
        fx_rates={"EUR": 1.2},
    )
    assert targets.quantities == {"ES": 1, "MES": 2}
    # Achieved notionals: 240,000 + 48,000 USD over 120,000 USD equity.
    assert dict(report.achieved_weights) == pytest.approx({"ES": 2.0, "MES": 0.4})
    assert report.tracking_error == pytest.approx((abs(2.5 - 2.0) + 0.4) / 2.0)
    assert report.tracking_error_pct == pytest.approx(45.0)
    assert report.margin_utilization == pytest.approx(18_000.0 / 120_000.0)


def test_eur_report_matches_per_call_fx_overrides() -> None:
    """Per-call FX overrides on the report agree with translator-level overrides."""
    state = _state(roots=("ES",), equity=100_000.0, account_currency="EUR", fx_rates={"EUR": 1.2})
    engine = _translator(rounding_mode=RoundingMode.FLOOR)
    weights = make_weight_vector({"ES": 2.5})
    targets = engine.translate(weights, state, account_currency="EUR", fx_rates={"EUR": 1.2})
    report = engine.discretization_report(
        weights,
        targets,
        state,
        account_currency="EUR",
        fx_rates={"EUR": 1.2},
    )
    assert report.tracking_error == pytest.approx(0.45)
    assert report.margin_utilization == pytest.approx(0.15)


def test_non_usd_account_missing_rate_fails_closed() -> None:
    """A non-USD account without an FX rate raises before reporting (fail closed)."""
    state = _state(roots=("ES",), equity=100_000.0, account_currency="EUR")
    engine = _translator()
    weights = make_weight_vector({"ES": 0.5})
    targets = engine.translate(weights, state, account_currency="USD")
    with pytest.raises(KeyError, match="EUR"):
        engine.discretization_report(weights, targets, state)


def test_discretization_report_margin_utilization_matches_translator_method() -> None:
    engine, state, _weights, targets, report = _report({"ES": 0.30, "ZN": 0.05})
    expected = engine.margin_utilization(targets, state.equity or 0.0)
    assert report.margin_utilization == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Union-of-roots semantics and degeneracies
# ---------------------------------------------------------------------------


def test_empty_weights_return_zero_report_without_equity() -> None:
    """Fully flat inputs need neither equity nor FX data and report all zeros."""
    engine = _translator()
    state = _state(roots=("ES",), equity=None)
    weights = WeightVector.empty()
    targets = engine.translate(weights, state)
    report = engine.discretization_report(weights, targets, state)
    assert report.tracking_error == 0.0
    assert report.tracking_error_pct == 0.0
    assert report.margin_utilization == 0.0
    assert dict(report.target_weights) == {}
    assert dict(report.achieved_weights) == {}
    assert report.micro_routes == ()


def test_explicitly_zero_weights_are_flat() -> None:
    engine = _translator()
    state = _state(roots=("ES",))
    weights = WeightVector({"ES": 0.0, "ZN": 0.0})
    report = engine.discretization_report(weights, ContractTargetSet.empty(), state)
    assert report.tracking_error == 0.0
    assert dict(report.target_weights) == {}
    assert dict(report.achieved_weights) == {}


def test_target_root_without_achieved_leg_counts_as_zero_achieved() -> None:
    """A target below the discrete threshold achieves 0: error is |w| / 2."""
    _, _, _, targets, report = _report(
        {"ZN": 0.08}, translator=_translator(rounding_mode=RoundingMode.FLOOR)
    )
    assert targets.quantities == {}
    assert dict(report.target_weights) == pytest.approx({"ZN": 0.08})
    assert dict(report.achieved_weights) == {"ZN": 0.0}
    assert report.tracking_error == pytest.approx(0.04)
    assert report.tracking_error_pct == pytest.approx(4.0)
    assert report.margin_utilization == 0.0
    assert report.micro_routes == ()


def test_achieved_leg_without_target_counts_as_zero_target() -> None:
    """An achieved leg with no explicit weight gets a 0.0 target weight."""
    engine = _translator()
    state = _state(roots=("ES",))
    targets = make_contract_targets({"ES": 1})
    report = engine.discretization_report(WeightVector.empty(), targets, state)
    assert dict(report.target_weights) == {"ES": 0.0}
    assert dict(report.achieved_weights) == pytest.approx({"ES": ES_CONTRACT_USD / 1_000_000.0})
    assert report.tracking_error == pytest.approx(0.12)
    assert set(report.target_weights) == set(report.achieved_weights)
    assert report.micro_routes == ()


def test_zero_quantity_target_contributes_zero_exposure() -> None:
    engine = _translator()
    state = _state(roots=("ES",))
    targets = ContractTargetSet(
        (ContractTarget(instrument_id="ES", quantity=0, price=4800.0, multiplier=50.0),)
    )
    report = engine.discretization_report(make_weight_vector({"ES": 0.1}), targets, state)
    assert dict(report.achieved_weights) == {"ES": 0.0}
    assert report.tracking_error == pytest.approx(0.05)
    assert report.micro_routes == ()


def test_unknown_target_root_raises_value_error_naming_root() -> None:
    engine = _translator()
    state = _state(roots=("ES",))
    weights = make_weight_vector({"ES": 0.1})
    targets = ContractTargetSet(
        (
            ContractTarget(
                instrument_id="XX",
                quantity=1,
                price=100.0,
                multiplier=1.0,
            ),
        )
    )
    with pytest.raises(ValueError, match="XX"):
        engine.discretization_report(weights, targets, state)


def test_report_is_deterministic() -> None:
    engine, state, weights, targets, first = _report({"ES": 0.30, "NQ": 0.20})
    second = engine.discretization_report(weights, targets, state)
    assert first == second
    assert first.tracking_error == second.tracking_error


def test_report_maps_are_frozen_and_keys_consistent() -> None:
    _, _, _, _, report = _report({"ES": 0.30, "NQ": 0.20})
    assert set(report.target_weights) == set(report.achieved_weights)
    assert isinstance(report.micro_routes, tuple)
    with pytest.raises(TypeError):
        report.target_weights["ES"] = 0.1  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.tracking_error = 0.0  # type: ignore[misc]
    assert report.tracking_error_pct == pytest.approx(report.tracking_error * 100.0)


def test_report_construction_validation() -> None:
    with pytest.raises(ValueError):
        DiscretizationReport(
            tracking_error=-0.1, tracking_error_pct=0.0, target_weights={}, achieved_weights={}
        )
    with pytest.raises(ValueError):
        DiscretizationReport(
            tracking_error=0.1, tracking_error_pct=math.inf, target_weights={}, achieved_weights={}
        )
    with pytest.raises(ValueError):
        DiscretizationReport(
            tracking_error=0.1,
            tracking_error_pct=10.0,
            target_weights={"": 1.0},
            achieved_weights={},
        )
    with pytest.raises(ValueError):
        DiscretizationReport(
            tracking_error=0.1,
            tracking_error_pct=10.0,
            target_weights={"ES": math.nan},
            achieved_weights={},
        )
    with pytest.raises(TypeError):
        DiscretizationReport(
            tracking_error=0.0,
            tracking_error_pct=0.0,
            target_weights={},
            achieved_weights={},
            micro_routes=("not-a-route",),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError):
        MicroRoute(
            family="", standard_root="ES", standard_quantity=1, micro_root="MES", micro_quantity=1
        )
    with pytest.raises(TypeError):
        MicroRoute(
            family="ES",
            standard_root="ES",
            standard_quantity=True,  # type: ignore[arg-type]
            micro_root="MES",
            micro_quantity=1,
        )


def test_family_without_micro_pair_has_no_route() -> None:
    _, _, _, targets, report = _report({"ZN": 0.223})
    assert targets.quantities == {"ZN": 2}
    assert report.micro_routes == ()
    assert report.tracking_error == pytest.approx(0.0, abs=1e-15)


# ---------------------------------------------------------------------------
# Hypothesis properties
# ---------------------------------------------------------------------------


_NON_ZERO_WEIGHT = st.one_of(
    st.floats(min_value=0.01, max_value=0.9, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-0.9, max_value=-0.01, allow_nan=False, allow_infinity=False),
)


@settings(max_examples=60, deadline=None)
@given(
    equity=st.floats(
        min_value=2_000_000.0, max_value=20_000_000.0, allow_nan=False, allow_infinity=False
    ),
    weight=_NON_ZERO_WEIGHT,
    rounding_mode=st.sampled_from(list(RoundingMode)),
)
def test_property_non_routed_tracking_error_equals_analytic_deviation(
    equity: float, weight: float, rounding_mode: RoundingMode
) -> None:
    """Non-routed ZN: the report equals the analytic weight deviation / 2.

    Also proves ``0 <= tracking_error <= 1`` for a fully-invested single leg
    (|w| <= 0.9 and a half-contract rounding bound below 0.06 at 2M+ equity).
    """
    state = make_market_state(
        roots=("ZN",), equity=equity, with_covariance=False, with_returns=False
    )
    translator = _translator(rounding_mode=rounding_mode, enable_micro_routing=False)
    weights = make_weight_vector({"ZN": weight})
    targets = translator.translate(weights, state)
    report = translator.discretization_report(weights, targets, state)

    target = targets.get("ZN")
    achieved = (
        0.0 if target is None else target.quantity * target.price * target.multiplier / equity
    )
    assert report.tracking_error == pytest.approx(abs(weight - achieved) / 2.0, abs=1e-12)
    assert 0.0 <= report.tracking_error <= 1.0
    assert report.tracking_error_pct == pytest.approx(report.tracking_error * 100.0, abs=1e-12)
    assert dict(report.target_weights) == {"ZN": weight}
    assert report.achieved_weights["ZN"] == pytest.approx(achieved, abs=1e-12)
    assert report.micro_routes == ()
    assert report.margin_utilization == pytest.approx(targets.total_initial_margin / equity)


@settings(max_examples=50, deadline=None)
@given(
    equity=st.floats(
        min_value=5_000_000.0, max_value=50_000_000.0, allow_nan=False, allow_infinity=False
    ),
    es=st.floats(min_value=-0.3, max_value=0.3, allow_nan=False, allow_infinity=False),
    nq=st.floats(min_value=-0.3, max_value=0.3, allow_nan=False, allow_infinity=False),
    zn=st.floats(min_value=-0.3, max_value=0.3, allow_nan=False, allow_infinity=False),
    rounding_mode=st.sampled_from(list(RoundingMode)),
)
def test_property_multi_root_maps_routes_and_bounds(
    equity: float, es: float, nq: float, zn: float, rounding_mode: RoundingMode
) -> None:
    """Maps stay key-consistent, routes mirror achieved contracts and error <= 1."""
    weights_map = {"ES": es, "NQ": nq, "ZN": zn}
    state = make_market_state(
        roots=("ES", "NQ", "ZN"),
        equity=equity,
        with_covariance=False,
        with_returns=False,
    )
    translator = _translator(rounding_mode=rounding_mode)
    weights = make_weight_vector(weights_map)
    targets = translator.translate(weights, state)
    report = translator.discretization_report(weights, targets, state)
    master = load_instruments()

    assert set(report.target_weights) == set(report.achieved_weights)
    assert 0.0 <= report.tracking_error <= 1.0
    assert report.tracking_error_pct == pytest.approx(report.tracking_error * 100.0, abs=1e-12)
    expected_error = 0.5 * math.fsum(
        abs(report.target_weights[root] - report.achieved_weights[root])
        for root in report.target_weights
    )
    assert report.tracking_error == pytest.approx(expected_error, abs=1e-12)
    assert report.margin_utilization <= translator.max_margin_utilization + 1e-9
    quantities = dict(targets.quantities)
    for route in report.micro_routes:
        assert route.family != route.micro_root
        assert route.standard_root == route.family
        assert master.get(route.family).micro_root == route.micro_root
        assert route.standard_quantity == quantities.get(route.standard_root, 0)
        assert route.micro_quantity == quantities.get(route.micro_root, 0)
        assert route.micro_quantity != 0
    for root, achieved_weight in report.achieved_weights.items():
        if root in ("ES", "NQ", "ZN"):
            continue
        assert root in {"MES", "MNQ"}
        assert achieved_weight != 0.0
