"""Unit tests for the tick-aware futures transaction-cost model."""

from __future__ import annotations

import math
from collections.abc import Mapping

import pytest
from configs.instruments import InstrumentsConfig
from src.backtest.cost_model import (
    ASSET_CLASS_FEE_SCHEDULES,
    DEFAULT_FEE_SCHEDULE,
    EXCHANGE_FEE_SCHEDULES,
    CostBreakdown,
    CostModel,
    FeeSchedule,
    RollCostBreakdown,
    SlippageModel,
    default_fee_schedule,
)
from src.core.types import AssetClass


@pytest.fixture()
def cost_model(instrument_master: InstrumentsConfig) -> CostModel:
    """A deterministic cost model with zero impact (pure spread/fee assertions)."""
    return CostModel(
        instrument_master=instrument_master,
        impact_coefficient=0.0,
        base_slippage_ticks=0.0,
    )


def test_fee_schedule_tiers_and_fees() -> None:
    schedule = FeeSchedule(
        commission_tiers=((0, 1.0), (100, 0.8), (1_000, 0.5)),
        exchange_fee=1.0,
        clearing_fee=0.25,
        nfa_fee=0.05,
    )
    assert schedule.fee_per_contract == pytest.approx(1.30)
    assert schedule.commission_rate(0) == pytest.approx(1.0)
    assert schedule.commission_rate(99) == pytest.approx(1.0)
    assert schedule.commission_rate(100) == pytest.approx(0.8)
    assert schedule.commission_rate(1_000) == pytest.approx(0.5)
    assert schedule.commission_rate(50_000) == pytest.approx(0.5)
    assert schedule.commission_for(3, monthly_volume=100) == pytest.approx(2.4)
    assert schedule.fees_for(4) == pytest.approx(5.2)
    assert schedule.to_dict()["fee_per_contract"] == pytest.approx(1.30)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"commission_tiers": ()},
        {"commission_tiers": ((1, 1.0),)},
        {"commission_tiers": ((0, 1.0), (0, 0.5))},
        {"commission_tiers": ((0, -1.0),)},
        {"commission_tiers": ((0, math.nan),)},
        {"exchange_fee": -1.0},
        {"nfa_fee": math.inf},
    ],
)
def test_fee_schedule_validation(kwargs: Mapping[str, object]) -> None:
    with pytest.raises((ValueError, TypeError)):
        FeeSchedule(**kwargs)  # type: ignore[arg-type]


def test_fee_schedule_quantity_validation() -> None:
    schedule = FeeSchedule()
    with pytest.raises(TypeError):
        schedule.commission_rate(1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        schedule.commission_rate(-1)
    with pytest.raises(ValueError):
        schedule.commission_for(-1)
    with pytest.raises(ValueError):
        schedule.fees_for(-1)


def test_default_fee_schedules_per_class_and_exchange() -> None:
    assert set(ASSET_CLASS_FEE_SCHEDULES) == set(AssetClass)
    assert (
        default_fee_schedule(AssetClass.EQUITY_INDEX)
        is ASSET_CLASS_FEE_SCHEDULES[AssetClass.EQUITY_INDEX]
    )
    assert default_fee_schedule(AssetClass.UNKNOWN) is ASSET_CLASS_FEE_SCHEDULES[AssetClass.UNKNOWN]
    # Exchange-specific schedules exist for the CME group venues.
    assert set(EXCHANGE_FEE_SCHEDULES) == {"CME", "CBOT", "NYMEX", "COMEX"}
    assert DEFAULT_FEE_SCHEDULE.fee_per_contract > 0.0


def test_fee_schedule_resolution(instrument_master: InstrumentsConfig) -> None:
    explicit = FeeSchedule(commission_tiers=((0, 9.0),), exchange_fee=9.0)
    low_volume_model = CostModel(
        instrument_master=instrument_master,
        fee_schedules={"ES": explicit},
        monthly_volume=0,
    )
    assert low_volume_model.fee_schedule("ES") is explicit
    candidate = low_volume_model.transaction_cost("ES", 2, 4800.0, 0.25, 50.0, 250_000.0)
    assert candidate.commission == pytest.approx(18.0)
    assert candidate.fees == pytest.approx((9.0 + 0.30 + 0.02) * 2)

    class_override = FeeSchedule(commission_tiers=((0, 1.0),), exchange_fee=0.5)
    class_model = CostModel(
        instrument_master=instrument_master,
        fee_schedules={"equity_index": class_override},
    )
    assert class_model.fee_schedule("ES") is class_override
    exchange_model = CostModel(
        instrument_master=instrument_master,
        fee_schedules={"CME": class_override},
    )
    assert exchange_model.fee_schedule("ES") is class_override
    # Unknown root without a master falls back to the generic default schedule.
    bare_model = CostModel()
    assert bare_model.fee_schedule("ZZZ") == DEFAULT_FEE_SCHEDULE
    assert bare_model.commission_per_contract("ZZZ") == pytest.approx(0.85)
    assert bare_model.fee_per_contract("ZZZ") == pytest.approx(
        DEFAULT_FEE_SCHEDULE.fee_per_contract
    )


def test_commission_tiers_by_monthly_volume(cost_model: CostModel) -> None:
    assert cost_model.commission_per_contract("ES", 0) == pytest.approx(0.85)
    assert cost_model.commission_per_contract("ES", 1_000) == pytest.approx(0.65)
    assert cost_model.commission_per_contract("ES", 10_000) == pytest.approx(0.45)


def test_transaction_cost_components_es(cost_model: CostModel) -> None:
    breakdown = cost_model.transaction_cost(
        "ES", 2, 4800.0, tick_size=0.25, multiplier=50.0, bar_volume=250_000.0
    )
    # Minimum one-tick full spread -> half a tick per side.
    assert breakdown.spread_ticks == pytest.approx(0.5)
    assert breakdown.slippage_ticks == pytest.approx(0.0)
    assert breakdown.spread_cost == pytest.approx(0.5 * 0.25 * 50.0 * 2)
    assert breakdown.slippage_cost == pytest.approx(0.0)
    assert breakdown.commission == pytest.approx(1.70)
    assert breakdown.fees == pytest.approx((1.38 + 0.30 + 0.02) * 2)
    assert breakdown.total == pytest.approx(
        breakdown.spread_cost + breakdown.slippage_cost + breakdown.commission + breakdown.fees
    )
    assert breakdown.funding == 0.0
    # The same call is deterministic.
    again = cost_model.transaction_cost(
        "ES", 2, 4800.0, tick_size=0.25, multiplier=50.0, bar_volume=250_000.0
    )
    assert again == breakdown


def test_quoted_spread_and_limit_orders(cost_model: CostModel) -> None:
    wide = cost_model.transaction_cost(
        "ES",
        1,
        4800.0,
        0.25,
        50.0,
        250_000.0,
        quoted_spread_ticks=4.0,
    )
    assert wide.spread_ticks == pytest.approx(2.0)
    tight = cost_model.transaction_cost(
        "ES",
        1,
        4800.0,
        0.25,
        50.0,
        250_000.0,
        quoted_spread_ticks=0.2,
    )
    assert tight.spread_ticks == pytest.approx(0.5)  # floored at one tick
    limit = cost_model.transaction_cost("ES", 1, 4800.0, 0.25, 50.0, 250_000.0, limit_order=True)
    assert limit.spread_cost == pytest.approx(0.0)
    assert limit.slippage_cost == pytest.approx(0.0)
    assert limit.total == pytest.approx(limit.commission + limit.fees)
    with pytest.raises(ValueError):
        cost_model.transaction_cost(
            "ES", 1, 4800.0, 0.25, 50.0, 250_000.0, quoted_spread_ticks=-1.0
        )


def test_bag_combo_costs(cost_model: CostModel) -> None:
    regular = cost_model.transaction_cost("ES", 2, 4800.0, 0.25, 50.0, 250_000.0)
    bag = cost_model.transaction_cost("ES", 2, 4800.0, 0.25, 50.0, 250_000.0, is_bag=True)
    assert bag.commission < regular.commission
    assert bag.spread_ticks == pytest.approx(regular.spread_ticks * cost_model.bag_spread_factor)


def test_volume_impact_models(instrument_master: InstrumentsConfig) -> None:
    square = CostModel(
        instrument_master=instrument_master,
        slippage_model=SlippageModel.SQUARE_ROOT,
        impact_coefficient=10.0,
        max_impact_ticks=100.0,
        base_slippage_ticks=0.0,
        min_spread_ticks=0.0,
    )
    linear = CostModel(
        instrument_master=instrument_master,
        slippage_model=SlippageModel.LINEAR,
        impact_coefficient=10.0,
        max_impact_ticks=100.0,
        base_slippage_ticks=0.0,
        min_spread_ticks=0.0,
    )
    small = square.estimated_impact_ticks(100, 100_000.0)
    large = square.estimated_impact_ticks(10_000, 100_000.0)
    assert small < large
    assert small == pytest.approx(10.0 * math.sqrt(100 / 100_000.0))
    assert linear.estimated_impact_ticks(1_000, 100_000.0) == pytest.approx(10.0 * 0.01)
    assert square.estimated_impact_ticks(1_000, 0.0) == pytest.approx(10.0)
    capped = CostModel(
        instrument_master=instrument_master,
        impact_coefficient=100.0,
        max_impact_ticks=4.0,
        base_slippage_ticks=0.5,
    )
    assert capped.estimated_impact_ticks(10_000, 1.0) == pytest.approx(4.5)
    assert capped.estimated_impact_ticks(1, 1.0, extra_slippage_ticks=2.0) == pytest.approx(
        4.0 + 0.5 + 2.0
    )


def test_slippage_is_monotonic_and_capped(instrument_master: InstrumentsConfig) -> None:
    model = CostModel(
        instrument_master=instrument_master,
        impact_coefficient=8.0,
        max_impact_ticks=5.0,
    )
    previous = 0.0
    for quantity in (1, 10, 100, 1_000):
        breakdown = model.transaction_cost("CL", quantity, 72.5, 0.01, 1_000.0, bar_volume=10_000.0)
        assert breakdown.slippage_ticks >= previous
        previous = breakdown.slippage_ticks
    assert previous <= 5.0 + model.base_slippage_ticks + 1e-12


def test_commission_bps_and_extra_fees(instrument_master: InstrumentsConfig) -> None:
    model = CostModel(instrument_master=instrument_master, commission_bps=2.0)
    plain = CostModel(instrument_master=instrument_master)
    priced = model.transaction_cost("ES", 1, 4800.0, 0.25, 50.0, 250_000.0)
    base = plain.transaction_cost("ES", 1, 4800.0, 0.25, 50.0, 250_000.0)
    assert priced.commission == pytest.approx(base.commission + 1 * 4800.0 * 50.0 * 2.0 / 10_000.0)
    penalised = plain.transaction_cost(
        "ES", 1, 4800.0, 0.25, 50.0, 250_000.0, extra_fee_per_contract=10.0
    )
    assert penalised.fees == pytest.approx(base.fees + 10.0)
    assert penalised.fee_per_contract == pytest.approx(base.fee_per_contract + 10.0)


def test_zero_quantity_returns_zero_breakdown(cost_model: CostModel) -> None:
    breakdown = cost_model.transaction_cost("ES", 0, 4800.0, 0.25, 50.0, 250_000.0)
    assert breakdown.total == 0.0
    assert breakdown.to_dict()["total"] == 0.0
    zero = CostBreakdown.zero("ES", price=4800.0, tick_size=0.25, multiplier=50.0)
    assert zero.total == 0.0
    assert zero.quantity == 0


def test_round_trip_cost_is_two_legs(cost_model: CostModel) -> None:
    single = cost_model.transaction_cost("ES", 2, 4800.0, 0.25, 50.0, 250_000.0)
    round_trip = cost_model.round_trip_cost("ES", 2, 4800.0, 0.25, 50.0, 250_000.0)
    assert round_trip.total == pytest.approx(single.total * 2.0)
    assert round_trip.spread_cost == pytest.approx(single.spread_cost * 2.0)
    assert round_trip.commission == pytest.approx(single.commission * 2.0)
    assert round_trip.price == pytest.approx(4800.0)


def test_roll_cost_math_and_bag_discount(cost_model: CostModel) -> None:
    plain = cost_model.roll_cost(
        "ES", 2, 4810.0, 4812.0, tick_size=0.25, multiplier=50.0, bar_volume=250_000.0
    )
    assert isinstance(plain, RollCostBreakdown)
    assert plain.close_leg.price == pytest.approx(4810.0)
    assert plain.open_leg.price == pytest.approx(4812.0)
    assert plain.total == pytest.approx(plain.close_leg.total + plain.open_leg.total)
    assert plain.bag_discount == pytest.approx(0.0)
    assert plain.total_before_bag_discount == pytest.approx(plain.total)
    assert plain.funding == 0.0

    bag = cost_model.roll_cost(
        "ES",
        2,
        4810.0,
        4812.0,
        tick_size=0.25,
        multiplier=50.0,
        bar_volume=250_000.0,
        use_bag=True,
    )
    assert bag.bag_discount > 0.0
    assert bag.total < plain.total
    assert bag.total_before_bag_discount == pytest.approx(plain.total)
    assert bag.commission < plain.commission
    assert bag.spread_cost < plain.spread_cost
    assert bag.to_dict()["bag_discount"] == pytest.approx(bag.bag_discount)


def test_forced_liquidation_cost_is_punitive(instrument_master: InstrumentsConfig) -> None:
    model = CostModel(instrument_master=instrument_master)
    regular = model.transaction_cost("ES", 2, 4800.0, 0.25, 50.0, 250_000.0)
    forced = model.forced_liquidation_cost("ES", 2, 4800.0, 0.25, 50.0, 250_000.0)
    assert forced.slippage_ticks > regular.slippage_ticks
    assert forced.fees > regular.fees
    explicit = model.forced_liquidation_cost(
        "ES",
        2,
        4800.0,
        0.25,
        50.0,
        250_000.0,
        penalty_ticks=1.0,
        penalty_fee_per_contract=1.0,
    )
    assert explicit.fees == pytest.approx(regular.fees + 2.0)
    assert explicit.slippage_ticks == pytest.approx(regular.slippage_ticks + 1.0)


def test_breakdown_helpers(cost_model: CostModel) -> None:
    first = cost_model.transaction_cost("ES", 1, 4800.0, 0.25, 50.0, 250_000.0)
    second = cost_model.transaction_cost("ES", 3, 4810.0, 0.25, 50.0, 250_000.0)
    combined = first.combined(second)
    assert combined.quantity == 4
    assert combined.total == pytest.approx(first.total + second.total)
    scaled = first.scaled(2.0)
    assert scaled.total == pytest.approx(first.total * 2.0)
    assert scaled.quantity == first.quantity
    with pytest.raises(ValueError):
        first.combined(cost_model.transaction_cost("NQ", 1, 16800.0, 0.25, 20.0, 1.0))
    with pytest.raises(ValueError):
        first.scaled(-1.0)
    ticks = first.total_ticks_per_contract
    assert ticks > first.execution_cost_ticks
    zero = CostBreakdown.zero("ES", price=4800.0, tick_size=0.25, multiplier=50.0)
    assert zero.total_ticks_per_contract == 0.0


def test_cost_breakdown_validation() -> None:
    base = {
        "root": "ES",
        "quantity": 1,
        "price": 4800.0,
        "tick_size": 0.25,
        "multiplier": 50.0,
    }
    with pytest.raises(ValueError):
        CostBreakdown(root="", **{k: v for k, v in base.items() if k != "root"})
    with pytest.raises(TypeError):
        CostBreakdown(**{**base, "quantity": 1.5})  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        CostBreakdown(**{**base, "quantity": -1})
    with pytest.raises(ValueError):
        CostBreakdown(**{**base, "tick_size": 0.0})
    with pytest.raises(ValueError):
        CostBreakdown(**{**base, "commission": math.inf})


def test_cost_model_validation() -> None:
    with pytest.raises(TypeError):
        CostModel(slippage_model="sqrt")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        CostModel(impact_coefficient=-1.0)
    with pytest.raises(ValueError):
        CostModel(min_spread_ticks=-1.0)
    with pytest.raises(ValueError):
        CostModel(bag_spread_factor=1.5)
    with pytest.raises(TypeError):
        CostModel(monthly_volume=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        CostModel(monthly_volume=-1)


def test_transaction_cost_validation(cost_model: CostModel) -> None:
    with pytest.raises(ValueError):
        cost_model.transaction_cost("", 1, 4800.0, 0.25, 50.0, 1.0)
    with pytest.raises(TypeError):
        cost_model.transaction_cost("ES", 1.5, 4800.0, 0.25, 50.0, 1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        cost_model.transaction_cost("ES", -1, 4800.0, 0.25, 50.0, 1.0)
    with pytest.raises(ValueError):
        cost_model.transaction_cost("ES", 1, 0.0, 0.25, 50.0, 1.0)
    with pytest.raises(ValueError):
        cost_model.transaction_cost("ES", 1, 4800.0, 0.0, 50.0, 1.0)
    with pytest.raises(ValueError):
        cost_model.transaction_cost("ES", 1, 4800.0, 0.25, 0.0, 1.0)
    with pytest.raises(ValueError):
        cost_model.transaction_cost("ES", 1, 4800.0, 0.25, 50.0, -1.0)
    with pytest.raises(ValueError):
        cost_model.transaction_cost("ES", 1, 4800.0, 0.25, 50.0, 1.0, monthly_volume=-1)


def test_roll_and_forced_validation(cost_model: CostModel) -> None:
    with pytest.raises(ValueError):
        cost_model.roll_cost("ES", 1, 4800.0, 0.0, 0.25, 50.0, 1.0)
    with pytest.raises(ValueError):
        cost_model.roll_cost("ES", -1, 4800.0, 4801.0, 0.25, 50.0, 1.0)
    with pytest.raises(ValueError):
        cost_model.roll_cost("ES", 1, 4800.0, 4801.0, 0.25, 50.0, 1.0, monthly_volume=-5)


def test_participation_rate_validation(cost_model: CostModel) -> None:
    assert cost_model.participation_rate(10, 0.0) == pytest.approx(1.0)
    assert cost_model.participation_rate(10, 100.0) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        cost_model.participation_rate(-1, 100.0)
    with pytest.raises(ValueError):
        cost_model.participation_rate(1, -1.0)
    with pytest.raises(ValueError):
        cost_model.estimated_impact_ticks(1, 1.0, extra_slippage_ticks=-1.0)


def test_cost_model_metadata(cost_model: CostModel) -> None:
    payload = cost_model.to_dict()
    assert payload["slippage_model"] == "sqrt"
    assert payload["min_spread_ticks"] == pytest.approx(1.0)
    assert payload["monthly_volume"] == 0
    assert cost_model.instrument_master is not None
