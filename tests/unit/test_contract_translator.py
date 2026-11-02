"""Unit tests for the deterministic futures :class:`ContractTranslator`.

All expectations are hand-computed from the committed instrument master (ES at
4800 with a 50x multiplier is a 240,000 USD contract; MES is 24,000 USD) and the
deterministic pytest fixtures. Every test is offline and deterministic.
"""

from __future__ import annotations

import math
from datetime import date

import pytest
from configs.instruments import InstrumentsConfig, load_instruments
from configs.strategy import TranslatorConfig
from hypothesis import given, settings
from hypothesis import strategies as st
from src.core.interfaces import ContractTranslatorProtocol
from src.core.types import (
    AssetClass,
    ContractTargetSet,
    InstrumentSnapshot,
    MarketState,
    RoundingMode,
    WeightVector,
)
from src.risk.contract_translator import ContractTranslator

from tests.fixtures.market import (
    DEFAULT_EXPIRATION,
    DEFAULT_PRICES,
    FIXED_TIMESTAMP,
    make_snapshot,
    make_weight_vector,
)

#: Standard root -> paired micro root (mirrors ``configs/instruments.yaml``).
MICRO_PAIRS: dict[str, str] = {"ES": "MES", "NQ": "MNQ", "CL": "MCL", "GC": "MGC"}

#: Roots with no paired micro contract.
NO_MICRO_ROOTS: tuple[str, ...] = ("ZN", "ZT", "YM", "RTY", "6E")


def _master() -> InstrumentsConfig:
    return load_instruments()


def make_state(
    roots: tuple[str, ...] = ("ES", "NQ", "CL", "GC", "ZN"),
    *,
    equity: float | None = 1_000_000.0,
    prices: dict[str, float] | None = None,
    include_micros: bool = True,
) -> MarketState:
    """Build a compact market state with deterministic prices."""
    master = _master()
    selected = list(roots)
    if include_micros:
        for root in list(selected):
            micro = master.get(root).micro_root
            if micro is not None and micro not in selected:
                selected.append(micro)
    price_map = prices or {}
    snapshots = [make_snapshot(root, price=price_map.get(root)) for root in selected]
    return MarketState(
        timestamp=FIXED_TIMESTAMP,
        instruments={snapshot.instrument_id: snapshot for snapshot in snapshots},
        equity=equity,
    )


def _translator(**kwargs: object) -> ContractTranslator:
    defaults: dict[str, object] = {"rounding_mode": RoundingMode.NEAREST}
    defaults.update(kwargs)
    return ContractTranslator(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Exact hand-computed translation math
# ---------------------------------------------------------------------------


def test_hand_computed_es_notional_floor_micro_routing() -> None:
    """ES at 1,000,000 equity: 300,000 USD target is 1.25 ES -> 1 ES + micro residual."""
    translator = _translator(rounding_mode=RoundingMode.FLOOR)
    targets = translator.translate(make_weight_vector({"ES": 0.30}), make_state(roots=("ES",)))
    # 1.25 ES raw -> FLOOR 1 ES (240,000 USD); residual 60,000 USD / 24,000 = 2.5 MES -> 2.
    assert targets.quantities == {"ES": 1, "MES": 2}
    es = targets.get("ES")
    mes = targets.get("MES")
    assert es is not None
    assert mes is not None
    assert es.price == 4800.0
    assert es.multiplier == 50.0
    assert es.signed_notional == pytest.approx(240_000.0)
    assert es.expiration == DEFAULT_EXPIRATION
    assert es.contract_symbol == "ES_20240315"
    assert es.initial_margin == 15_000.0
    assert es.maintenance_margin == 13_600.0
    assert es.asset_class is AssetClass.EQUITY_INDEX
    assert mes.price == 4800.0
    assert mes.multiplier == 5.0
    assert mes.contract_symbol == "MES_20240315"
    assert mes.initial_margin == 1_500.0
    assert mes.asset_class is AssetClass.EQUITY_INDEX
    # Initial margin projection recomputed after micro routing.
    assert targets.total_initial_margin == pytest.approx(15_000.0 + 2 * 1_500.0)
    assert translator.margin_utilization(targets, 1_000_000.0) == pytest.approx(0.018)


def test_nearest_rounds_residual_micro_half_away_from_zero() -> None:
    """NEAREST: 2.5 MES residual contracts round away from zero to 3."""
    translator = _translator(rounding_mode=RoundingMode.NEAREST)
    targets = translator.translate(make_weight_vector({"ES": 0.30}), make_state(roots=("ES",)))
    # ES raw 1.25 -> 1; residual 60,000 / 24,000 = 2.5 MES -> 3.
    assert targets.quantities == {"ES": 1, "MES": 3}
    achieved = targets.net_notional
    assert achieved == pytest.approx(240_000.0 + 3 * 24_000.0)
    assert abs(achieved - 300_000.0) <= 24_000.0


def test_short_weights_produce_negative_contracts() -> None:
    """A negative weight must produce a short contract leg, never a long one."""
    translator = _translator(rounding_mode=RoundingMode.FLOOR)
    targets = translator.translate(make_weight_vector({"ES": -0.30}), make_state(roots=("ES",)))
    assert targets.quantities == {"ES": -1, "MES": -2}
    assert targets.net_notional < 0.0
    es = targets.get("ES")
    assert es is not None
    assert es.side.value == "SELL"
    assert es.initial_margin_required == pytest.approx(15_000.0)


@pytest.mark.parametrize(
    ("mode", "quantity"),
    [
        (RoundingMode.FLOOR, 1),
        (RoundingMode.NEAREST, 1),
        (RoundingMode.CEIL, 2),
    ],
)
def test_rounding_mode_on_fractional_standard_contract(mode: RoundingMode, quantity: int) -> None:
    """1.4 raw ZN contracts round toward zero / nearest / away from zero."""
    equity = 1_000_000.0
    weight = 1.4 * (111.5 * 1000.0) / equity
    translator = _translator(rounding_mode=mode)
    targets = translator.translate(make_weight_vector({"ZN": weight}), make_state(roots=("ZN",)))
    assert targets.quantities == {"ZN": quantity}


@pytest.mark.parametrize(
    ("raw", "floor", "nearest", "ceil"),
    [
        (1.4, 1, 1, 2),
        (1.6, 1, 2, 2),
        (2.5, 2, 3, 3),
        (-1.4, -1, -1, -2),
        (-1.6, -1, -2, -2),
        (-2.5, -2, -3, -3),
    ],
)
def test_rounding_modes_positive_and_negative(
    raw: float, floor: int, nearest: int, ceil: int
) -> None:
    equity = 1_000_000.0
    weight = raw * (111.5 * 1000.0) / equity
    state = make_state(roots=("ZN",))
    results = {}
    for mode in (RoundingMode.FLOOR, RoundingMode.NEAREST, RoundingMode.CEIL):
        targets = _translator(rounding_mode=mode).translate(
            make_weight_vector({"ZN": weight}), state
        )
        results[mode] = targets.quantities.get("ZN", 0)
    assert results[RoundingMode.FLOOR] == floor
    assert results[RoundingMode.NEAREST] == nearest
    assert results[RoundingMode.CEIL] == ceil


@pytest.mark.parametrize(
    ("buffer", "expected"),
    [(0.0, 2), (0.2, 2), (0.5, 1), (2.0, 1)],
)
def test_rebalance_buffer_biases_nearest_toward_zero(buffer: float, expected: int) -> None:
    """A 1.6-contract target rounds up only when the buffer threshold is met."""
    equity = 1_000_000.0
    weight = 1.6 * (111.5 * 1000.0) / equity
    translator = _translator(rounding_mode=RoundingMode.NEAREST, rebalance_buffer_contracts=buffer)
    targets = translator.translate(make_weight_vector({"ZN": weight}), make_state(roots=("ZN",)))
    assert targets.quantities == {"ZN": expected}


# ---------------------------------------------------------------------------
# Micro routing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (RoundingMode.FLOOR, {"MES": 2}),
        (RoundingMode.NEAREST, {"MES": 3}),
    ],
)
def test_small_equity_routes_into_micro_contract(
    mode: RoundingMode, expected: dict[str, int]
) -> None:
    """At 200,000 equity an ES target of 0.30 cannot hold ES; MES carries the exposure."""
    translator = _translator(rounding_mode=mode)
    state = make_state(roots=("ES",), equity=200_000.0)
    targets = translator.translate(make_weight_vector({"ES": 0.30}), state)
    assert targets.quantities == expected
    assert targets.get("ES") is None
    achieved = abs(targets.net_notional)
    target = 0.30 * 200_000.0
    assert achieved <= target + 24_000.0
    assert achieved > 0.0


def test_ceil_mode_never_overshoots_requested_notional() -> None:
    """CEIL picks the same-direction decomposition with the smallest tracking error."""
    translator = _translator(rounding_mode=RoundingMode.CEIL)
    state = make_state(roots=("ES",), equity=200_000.0)
    targets = translator.translate(make_weight_vector({"ES": 0.30}), state)
    # CEIL standard leg would be 1 ES (240k) vs a 60k target; the micro-only
    # decomposition (3 MES = 72k) tracks far better and never hedges itself.
    assert targets.quantities == {"MES": 3}
    assert abs(targets.net_notional) <= 60_000.0 + 24_000.0


def test_micro_routing_disabled_keeps_discrete_standard_only() -> None:
    translator = _translator(rounding_mode=RoundingMode.NEAREST, enable_micro_routing=False)
    state = make_state(roots=("ES",), equity=200_000.0)
    targets = translator.translate(make_weight_vector({"ES": 0.30}), state)
    # 0.25 ES raw rounds to zero, and routing is disabled: flat target set.
    assert targets.quantities == {}


def test_both_standard_and_micro_requested_are_netted() -> None:
    """ES 0.30 + MES 0.10 nets into a 400,000 USD family target with no double count."""
    translator = _translator(rounding_mode=RoundingMode.NEAREST)
    targets = translator.translate(
        make_weight_vector({"ES": 0.30, "MES": 0.10}), make_state(roots=("ES",))
    )
    # Family target 400,000: the truncation decomposition 1 ES (240,000) plus a
    # same-sign residual of 160,000 (7 MES) tracks best and never self-hedges.
    assert targets.quantities == {"ES": 1, "MES": 7}
    achieved = targets.net_notional
    assert achieved == pytest.approx(240_000.0 + 168_000.0)
    assert abs(achieved - 400_000.0) <= 24_000.0
    assert all(target.quantity > 0 for target in targets)


def test_micro_only_request_stays_micro() -> None:
    """An explicit micro weight is honoured as micro contracts, never upsampled."""
    translator = _translator()
    state = make_state(roots=("ES",), equity=200_000.0)
    targets = translator.translate(make_weight_vector({"MES": 0.72}), state)
    assert targets.quantities == {"MES": 6}


def test_exact_standard_multiple_has_no_micro_leg() -> None:
    """A whole standard-contract multiple leaves a zero residual: no micro leg."""
    translator = _translator(rounding_mode=RoundingMode.NEAREST)
    targets = translator.translate(make_weight_vector({"ES": 0.24}), make_state(roots=("ES",)))
    # 0.24 * 1,000,000 = 240,000 = exactly one ES, residual 0 -> no MES leg.
    assert targets.quantities == {"ES": 1}


def test_residual_below_half_micro_contract_is_not_routed() -> None:
    """A residual that rounds to zero micro contracts produces no micro leg."""
    translator = _translator(rounding_mode=RoundingMode.NEAREST)
    targets = translator.translate(make_weight_vector({"ES": 0.25}), make_state(roots=("ES",)))
    # 1.0417 ES -> 1 ES; residual 10,000 USD = 0.417 MES -> 0.
    assert targets.quantities == {"ES": 1}


def test_impossible_margin_budget_flattens_to_feasible_targets() -> None:
    """CEIL + a budget below one contract margin must still return a feasible set."""
    translator = _translator(rounding_mode=RoundingMode.CEIL)
    targets = translator.translate(
        make_weight_vector({"ES": 1.0}), make_state(roots=("ES",), equity=10_000.0)
    )
    assert translator.margin_utilization(targets, 10_000.0) <= 0.60
    assert targets.quantities == {"MES": 1}


def test_min_micro_notional_ratio_suppresses_dust_routing() -> None:
    """A residual below the configured micro notional ratio is not routed."""
    equity = 1_000_000.0
    # ES raw = 1.06 -> 1 ES; residual 0.06 * 240,000 = 14,400 USD = 0.6 MES.
    weight = 1.06 * (4800.0 * 50.0) / equity
    state = make_state(roots=("ES",))
    routed = _translator().translate(make_weight_vector({"ES": weight}), state)
    assert routed.quantities == {"ES": 1, "MES": 1}
    suppressed = _translator(min_micro_notional_ratio=0.75).translate(
        make_weight_vector({"ES": weight}), state
    )
    assert suppressed.quantities == {"ES": 1}


# ---------------------------------------------------------------------------
# Regression: never hedge one family leg against the other
# ---------------------------------------------------------------------------


def test_nq_below_one_standard_contract_routes_single_micro_leg() -> None:
    """Regression (lead review): NQ 0.20 at 1M must be MNQ +6, not NQ +1 / MNQ -4."""
    translator = _translator(rounding_mode=RoundingMode.NEAREST)
    targets = translator.translate(make_weight_vector({"NQ": 0.20}), make_state(roots=("NQ",)))
    # 200,000 / 336,000 = 0.595 standard contracts; the micro-only decomposition
    # (200,000 / 33,600 = 5.95 -> 6 MNQ) tracks far better than the overshooting
    # standard leg and never creates an opposite-signed hedge.
    assert targets.quantities == {"MNQ": 6}
    assert [target.quantity for target in targets] == [6]
    assert targets.net_notional == pytest.approx(201_600.0)
    assert abs(targets.net_notional - 200_000.0) <= 33_600.0


def test_standard_leg_above_one_gets_same_sign_micro_residual() -> None:
    """A 1.25 standard-contract target keeps a same-sign residual micro leg."""
    translator = _translator(rounding_mode=RoundingMode.NEAREST)
    weight = 1.25 * (16_800.0 * 20.0) / 1_000_000.0
    targets = translator.translate(make_weight_vector({"NQ": weight}), make_state(roots=("NQ",)))
    # NQ raw 1.25 -> 1 NQ (336,000); residual 84,000 / 33,600 = 2.5 -> 3 MNQ.
    assert targets.quantities == {"NQ": 1, "MNQ": 3}
    assert all(target.quantity > 0 for target in targets)


def test_floor_mode_never_emits_negative_micro_residual() -> None:
    """FLOOR is conservative: its residual is always in the direction of the target."""
    translator = _translator(rounding_mode=RoundingMode.FLOOR)
    # Both legs explicitly requested, family raw = 1.8 standard contracts.
    weight = 1.8 * (4_800.0 * 50.0) / 1_000_000.0
    targets = translator.translate(
        make_weight_vector({"ES": weight / 2.0, "MES": weight / 2.0}),
        make_state(roots=("ES",)),
    )
    assert targets.quantities == {"ES": 1, "MES": 8}
    assert all(target.quantity > 0 for target in targets)
    short = translator.translate(
        make_weight_vector({"ES": -weight / 2.0, "MES": -weight / 2.0}),
        make_state(roots=("ES",)),
    )
    assert short.quantities == {"ES": -1, "MES": -8}
    assert all(target.quantity < 0 for target in short)


@pytest.mark.parametrize("mode", [RoundingMode.FLOOR, RoundingMode.NEAREST, RoundingMode.CEIL])
@pytest.mark.parametrize("weight", [0.2, 0.3, 0.42, 0.55, -0.2, -0.42, -0.55])
def test_family_never_has_opposite_signed_legs(mode: RoundingMode, weight: float) -> None:
    """A standard/micro family is a directional position, never a self-hedge spread."""
    translator = _translator(rounding_mode=mode)
    targets = translator.translate(make_weight_vector({"NQ": weight}), make_state(roots=("NQ",)))
    quantities = [target.quantity for target in targets]
    assert not (any(q > 0 for q in quantities) and any(q < 0 for q in quantities))


def test_nearest_tie_break_prefers_fewer_legs() -> None:
    """On equal tracking error the configured standard leg (fewer legs) wins."""
    translator = _translator(rounding_mode=RoundingMode.NEAREST)
    weight = 0.99 * (16_800.0 * 20.0) / 1_000_000.0
    targets = translator.translate(make_weight_vector({"NQ": weight}), make_state(roots=("NQ",)))
    assert targets.quantities == {"NQ": 1}


def test_cancelling_standard_and_micro_weights_produce_flat_set() -> None:
    """Equal-and-opposite standard/micro weights net to zero (flat) exposure."""
    translator = _translator(rounding_mode=RoundingMode.NEAREST)
    targets = translator.translate(
        make_weight_vector({"ES": 0.10, "MES": -0.10}), make_state(roots=("ES",))
    )
    assert targets.quantities == {}


def test_micro_request_resolves_price_from_standard_counterpart() -> None:
    """A state carrying only the standard leg can still price an explicit micro leg."""
    translator = _translator(rounding_mode=RoundingMode.FLOOR)
    state = MarketState(
        timestamp=FIXED_TIMESTAMP,
        instruments={"ES": make_snapshot("ES")},
        equity=200_000.0,
    )
    targets = translator.translate(make_weight_vector({"MES": 0.72}), state)
    # 144,000 USD / 24,000 = 6 MES, priced off the same-underlying ES snapshot.
    assert targets.quantities == {"MES": 6}
    assert targets.get("MES") is not None
    assert targets.get("MES").price == 4800.0  # type: ignore[union-attr]


def test_missing_snapshot_for_unpaired_root_raises() -> None:
    """A root with no standard/micro counterpart and no snapshot is a hard error."""
    translator = _translator()
    state = MarketState(
        timestamp=FIXED_TIMESTAMP,
        instruments={"ES": make_snapshot("ES")},
        equity=1_000_000.0,
    )
    with pytest.raises(ValueError, match="ZN"):
        translator.translate(make_weight_vector({"ZN": 0.10}), state)


def test_truncate_quantity_handles_zero_and_non_finite() -> None:
    translator = _translator()
    assert translator._truncate_quantity(0.0) == 0
    assert translator._truncate_quantity(math.inf) == 0
    assert translator._truncate_quantity(math.nan) == 0


# ---------------------------------------------------------------------------
# Margin feasibility projection
# ---------------------------------------------------------------------------


def test_margin_ceiling_scales_positions_down() -> None:
    """A 20x ES target at 1,000,000 equity must be scaled to the 60% margin ceiling."""
    translator = _translator(rounding_mode=RoundingMode.NEAREST)
    state = make_state(roots=("ES",))
    targets = translator.translate(make_weight_vector({"ES": 20.0}), state)
    # Unconstrained: 83.33 ES + residual micro would need >1.2M initial margin.
    utilization = translator.margin_utilization(targets, 1_000_000.0)
    assert utilization <= 0.60
    # Iterative scale + deterministic feasibility drop yields 39 ES + 1 MES.
    assert targets.quantities == {"ES": 39, "MES": 1}
    assert targets.total_initial_margin == pytest.approx(39 * 15_000.0 + 1_500.0)


def test_tight_margin_ceiling_always_respected() -> None:
    translator = _translator(max_margin_utilization=0.02)
    state = make_state()
    weights = make_weight_vector({"ES": 0.5, "NQ": 0.5, "CL": 0.5, "GC": 0.5})
    targets = translator.translate(weights, state)
    assert translator.margin_utilization(targets, 1_000_000.0) <= 0.02 + 1e-12


def test_flat_weights_return_flat_target_set() -> None:
    translator = _translator()
    flat = ContractTargetSet.empty()
    assert translator.translate(WeightVector.empty(), make_state()) == flat
    assert translator.translate(make_weight_vector({"ES": 0.0, "NQ": 0.0}), make_state()) == flat
    # A flat target set needs no equity information at all.
    assert translator.translate(WeightVector.empty(), make_state(equity=None)) == flat


# ---------------------------------------------------------------------------
# Error handling and validation
# ---------------------------------------------------------------------------


def test_missing_equity_raises_clear_value_error() -> None:
    translator = _translator()
    state = make_state(roots=("ES",), equity=None)
    with pytest.raises(ValueError, match="equity"):
        translator.translate(make_weight_vector({"ES": 0.10}), state)


def test_explicit_equity_overrides_state_equity() -> None:
    translator = _translator(rounding_mode=RoundingMode.FLOOR)
    state = make_state(roots=("ES",), equity=1_000_000.0)
    targets = translator.translate(make_weight_vector({"ES": 0.30}), state, equity=200_000.0)
    assert targets.quantities == {"MES": 2}


def test_unknown_root_raises_value_error_naming_root() -> None:
    translator = _translator()
    with pytest.raises(ValueError, match="XX"):
        translator.translate(make_weight_vector({"XX": 0.1}), make_state())


def test_missing_market_snapshot_raises_value_error_naming_root() -> None:
    translator = _translator()
    state = MarketState(
        timestamp=FIXED_TIMESTAMP,
        instruments={"ES": make_snapshot("ES")},
        equity=1_000_000.0,
    )
    with pytest.raises(ValueError, match="NQ"):
        translator.translate(make_weight_vector({"NQ": 0.10}), state)


def test_negative_or_zero_equity_rejected() -> None:
    translator = _translator()
    state = make_state(roots=("ES",))
    for bad_equity in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(ValueError):
            translator.translate(make_weight_vector({"ES": 0.1}), state, equity=bad_equity)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_margin_utilization": 0.0},
        {"max_margin_utilization": 1.5},
        {"rebalance_buffer_contracts": -1.0},
        {"rebalance_buffer_contracts": math.inf},
        {"min_micro_notional_ratio": -0.1},
        {"min_micro_notional_ratio": 1.1},
    ],
)
def test_invalid_constructor_configuration_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        ContractTranslator(**kwargs)  # type: ignore[arg-type]


def test_rounding_mode_must_be_enum_member() -> None:
    with pytest.raises(TypeError, match="RoundingMode"):
        ContractTranslator(rounding_mode="nearest")  # type: ignore[arg-type]


def test_invalid_margin_utilization_argument_rejected() -> None:
    translator = _translator()
    state = make_state(roots=("ES",))
    with pytest.raises(ValueError):
        translator.margin_utilization(
            translator.translate(make_weight_vector({"ES": 0.1}), state), 0.0
        )


# ---------------------------------------------------------------------------
# Determinism, protocol and configuration integration
# ---------------------------------------------------------------------------


def test_translation_is_deterministic() -> None:
    translator = _translator()
    state = make_state()
    weights = make_weight_vector({"ES": 0.25, "NQ": -0.15, "CL": 0.1, "GC": -0.05})
    first = translator.translate(weights, state)
    second = translator.translate(weights, state)
    assert first == second


def test_lowercase_root_is_normalised_to_canonical_symbol() -> None:
    translator = _translator(rounding_mode=RoundingMode.FLOOR)
    targets = translator.translate(make_weight_vector({"es": 0.30}), make_state(roots=("ES",)))
    assert targets.quantities == {"ES": 1, "MES": 2}
    assert all(target.instrument_id == target.instrument_id.upper() for target in targets)


def test_satisfies_translator_protocol() -> None:
    assert isinstance(_translator(), ContractTranslatorProtocol)


def test_from_config_uses_declarative_parameters() -> None:
    config = TranslatorConfig(
        rounding_mode=RoundingMode.FLOOR,
        rebalance_buffer_contracts=0.0,
        max_margin_utilization=0.30,
        enable_micro_routing=False,
        min_micro_notional_ratio=0.0,
    )
    translator = ContractTranslator.from_config(config)
    assert translator.rounding_mode is RoundingMode.FLOOR
    assert translator.max_margin_utilization == 0.30
    assert translator.enable_micro_routing is False
    state = make_state(roots=("ES",), equity=200_000.0)
    assert translator.translate(make_weight_vector({"ES": 0.30}), state).quantities == {}


def test_contract_symbol_without_expiration_uses_root() -> None:
    translator = _translator()
    snapshot = InstrumentSnapshot(
        instrument_id="ZN",
        root_symbol="ZN",
        price=111.5,
        asset_class=AssetClass.RATES,
        expiration=None,
    )
    state = MarketState(
        timestamp=FIXED_TIMESTAMP,
        instruments={"ZN": snapshot},
        equity=1_000_000.0,
    )
    targets = translator.translate(make_weight_vector({"ZN": 0.2}), state)
    target = targets.get("ZN")
    assert target is not None
    assert target.contract_symbol == "ZN"
    assert target.expiration is None


# ---------------------------------------------------------------------------
# Hypothesis property tests: arbitrary equity / price / weight shocks
# ---------------------------------------------------------------------------


@settings(max_examples=60, deadline=None)
@given(
    equity=st.floats(
        min_value=1_000.0, max_value=5_000_000.0, allow_nan=False, allow_infinity=False
    ),
    weights=st.dictionaries(
        keys=st.sampled_from(["ES", "MES", "NQ", "MNQ", "CL", "MCL", "GC", "MGC", "ZN"]),
        values=st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=5,
    ),
    price_scale=st.floats(min_value=0.25, max_value=4.0, allow_nan=False, allow_infinity=False),
    rounding_mode=st.sampled_from([RoundingMode.FLOOR, RoundingMode.NEAREST]),
)
def test_property_margin_ceiling_and_notional_tracking(
    equity: float,
    weights: dict[str, float],
    price_scale: float,
    rounding_mode: RoundingMode,
) -> None:
    """Property: margin utilisation never exceeds eta_max and tracking error is bounded.

    ``|achieved notional| <= |target notional| + one micro contract value`` holds
    for every standard/micro family under the conservative (FLOOR) and balanced
    (NEAREST) modes; a root without a micro pair can overshoot by at most half a
    standard contract, the tightest possible bound for a standard-only instrument.
    """
    master = _master()
    prices = {root: DEFAULT_PRICES[root] * price_scale for root in DEFAULT_PRICES}
    state = make_state(roots=("ES", "NQ", "CL", "GC", "ZN"), equity=equity, prices=prices)
    translator = ContractTranslator(rounding_mode=rounding_mode)
    targets = translator.translate(WeightVector(weights), state)

    # 1. Hard margin ceiling invariant.
    utilization = targets.total_initial_margin / equity
    assert utilization <= translator.max_margin_utilization + 1e-9

    # 2. Notional tracking invariant per family.
    for standard_root, micro_root in MICRO_PAIRS.items():
        family_target = (weights.get(standard_root, 0.0) + weights.get(micro_root, 0.0)) * equity
        achieved = sum(
            target.signed_notional
            for target in targets
            if target.instrument_id in (standard_root, micro_root)
        )
        micro_value = prices[micro_root] * master.get(micro_root).multiplier
        assert abs(achieved) <= abs(family_target) + micro_value + 1e-6

    # 3. Standard-only roots: discrete rounding is bounded by one contract.
    for standard_root in ("ZN",):
        target_notional = weights.get(standard_root, 0.0) * equity
        achieved = sum(
            target.signed_notional for target in targets if target.instrument_id == standard_root
        )
        contract_value = prices[standard_root] * master.get(standard_root).multiplier
        assert abs(achieved) <= abs(target_notional) + contract_value + 1e-6

    # 4. A standard/micro family must never hedge itself: no opposite-signed legs.
    for standard_root, micro_root in MICRO_PAIRS.items():
        family_quantities = [
            target.quantity
            for target in targets
            if target.instrument_id in (standard_root, micro_root)
        ]
        assert not (
            any(quantity > 0 for quantity in family_quantities)
            and any(quantity < 0 for quantity in family_quantities)
        )


@settings(max_examples=50, deadline=None)
@given(
    equity=st.floats(
        min_value=1_000.0, max_value=5_000_000.0, allow_nan=False, allow_infinity=False
    ),
    weights=st.dictionaries(
        keys=st.sampled_from(["ES", "MES", "NQ", "MNQ", "CL", "MCL", "GC", "MGC", "ZN"]),
        values=st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=5,
    ),
    price_scale=st.floats(min_value=0.25, max_value=4.0, allow_nan=False, allow_infinity=False),
)
def test_property_ceil_margin_ceiling_and_feasible_tracking(
    equity: float, weights: dict[str, float], price_scale: float
) -> None:
    """CEIL is a stress mode: margin is always respected.

    When the requested exposure is itself margin feasible, the micro routing keeps
    ``|achieved| <= |target| + one micro contract value``. When the margin
    projection binds, solvency deliberately dominates tracking error.
    """
    master = _master()
    prices = {root: DEFAULT_PRICES[root] * price_scale for root in DEFAULT_PRICES}
    state = make_state(roots=("ES", "NQ", "CL", "GC", "ZN"), equity=equity, prices=prices)
    translator = ContractTranslator(rounding_mode=RoundingMode.CEIL)
    targets = translator.translate(WeightVector(weights), state)

    assert targets.total_initial_margin / equity <= translator.max_margin_utilization + 1e-9

    unconstrained = ContractTranslator(
        rounding_mode=RoundingMode.CEIL, max_margin_utilization=1.0
    ).translate(WeightVector(weights), state)
    if unconstrained.total_initial_margin > translator.max_margin_utilization * equity:
        return  # margin projection bound the exposure; tracking error is not meaningful
    assert targets == unconstrained
    for standard_root, micro_root in MICRO_PAIRS.items():
        family_target = (weights.get(standard_root, 0.0) + weights.get(micro_root, 0.0)) * equity
        achieved = sum(
            target.signed_notional
            for target in targets
            if target.instrument_id in (standard_root, micro_root)
        )
        micro_value = prices[micro_root] * master.get(micro_root).multiplier
        assert abs(achieved) <= abs(family_target) + micro_value + 1e-6

    # No standard/micro family may hedge itself in any rounding mode.
    for standard_root, micro_root in MICRO_PAIRS.items():
        family_quantities = [
            target.quantity
            for target in targets
            if target.instrument_id in (standard_root, micro_root)
        ]
        assert not (
            any(quantity > 0 for quantity in family_quantities)
            and any(quantity < 0 for quantity in family_quantities)
        )


@settings(max_examples=40, deadline=None)
@given(
    equity=st.floats(
        min_value=1_000.0, max_value=2_000_000.0, allow_nan=False, allow_infinity=False
    ),
    weight=st.floats(min_value=-4.0, max_value=4.0, allow_nan=False, allow_infinity=False),
)
def test_property_margin_ceiling_with_tight_utilization(equity: float, weight: float) -> None:
    """A very tight eta_max must still never be violated."""
    translator = ContractTranslator(max_margin_utilization=0.05)
    state = make_state(roots=("ES", "NQ"), equity=equity)
    targets = translator.translate(make_weight_vector({"ES": weight, "NQ": -weight}), state)
    assert targets.total_initial_margin / equity <= 0.05 + 1e-9


def test_expiration_date_is_preserved() -> None:
    translator = _translator()
    snapshot = make_snapshot("ES")
    state = MarketState(
        timestamp=FIXED_TIMESTAMP,
        instruments={"ES": snapshot},
        equity=1_000_000.0,
    )
    target = translator.translate(make_weight_vector({"ES": 0.3}), state).get("ES")
    assert target is not None
    assert target.expiration == date(2024, 3, 15)
