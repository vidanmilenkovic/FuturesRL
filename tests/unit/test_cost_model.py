"""Unit tests for the tick-aware futures transaction-cost model."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml
from configs.instruments import DEFAULT_INSTRUMENTS_PATH, InstrumentsConfig
from configs.settings import Settings
from src.backtest.cost_model import (
    ASSET_CLASS_FEE_SCHEDULES,
    CALIBRATED_COSTS_ENV_VAR,
    DEFAULT_CALIBRATION_MAX_AGE_DAYS,
    DEFAULT_FEE_SCHEDULE,
    EXCHANGE_FEE_SCHEDULES,
    CalibratedCostOverride,
    CalibrationError,
    CalibrationProvenance,
    CostBreakdown,
    CostModel,
    FeeSchedule,
    RollCostBreakdown,
    SlippageModel,
    default_calibrated_costs_path,
    default_fee_schedule,
    load_calibrated_costs,
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
    assert breakdown.fees == pytest.approx(1.40 * 2)
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


# ---------------------------------------------------------------------------
# De-bloat/realism: per-instrument fees, roll slippage and calibrated overrides
# ---------------------------------------------------------------------------


def _cost(model: CostModel, root: str, quantity: int = 1) -> CostBreakdown:
    return model.transaction_cost(root, quantity, 4_800.0, 0.25, 50.0, 250_000.0)


def _write_calibration(
    path: Path,
    roots: Mapping[str, object],
    *,
    calibrated_at: datetime | None = None,
    account_id: str = "DU1234567",
    warnings: list[str] | None = None,
    schema_version: int = 1,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": schema_version,
        "account_id": account_id,
        "calibrated_at": (calibrated_at or datetime.now(UTC)).isoformat(),
        "roots": dict(roots),
        "warnings": list(warnings or []),
    }
    path.write_text(json.dumps(payload))
    return path


def _root_override(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "broker_commission_per_contract": 0.42,
        "exchange_and_reg_fee_per_contract": 1.50,
        "initial_margin": 15_234.5,
        "maintenance_margin": 13_870.2,
        "commission_source": "ibkr_whatif",
        "margin_source": "ibkr_whatif",
    }
    values.update(overrides)
    return values


def test_all_instrument_master_roots_have_explicit_fee_fields(
    instrument_master: InstrumentsConfig,
) -> None:
    roots = instrument_master.roots()
    assert len(roots) == 45
    raw = yaml.safe_load(DEFAULT_INSTRUMENTS_PATH.read_text())
    raw_instruments = raw["instruments"]
    for root in roots:
        spec = instrument_master.get(root)
        assert spec.broker_commission_per_contract >= 0.0
        assert spec.exchange_and_reg_fee_per_contract >= 0.0
        assert 0.0 <= spec.default_slippage_ticks <= 1.0
        node = raw_instruments[root]
        assert "broker_commission_per_contract" in node
        assert "exchange_and_reg_fee_per_contract" in node
        assert "default_slippage_ticks" in node
        # Class realism: standards pay ~0.85 broker, micros are cheaper.
        if spec.is_micro:
            assert spec.broker_commission_per_contract <= 0.25
        else:
            assert spec.broker_commission_per_contract == pytest.approx(0.85)


def test_standard_vs_micro_exact_commission_and_fees(
    cost_model: CostModel,
) -> None:
    es = _cost(cost_model, "ES")
    mes = _cost(cost_model, "MES")
    assert es.commission_per_contract == pytest.approx(0.85)
    assert es.fee_per_contract == pytest.approx(1.40)
    assert mes.commission_per_contract == pytest.approx(0.25)
    assert mes.fee_per_contract == pytest.approx(0.37)
    assert mes.total < es.total
    cl = _cost(cost_model, "CL")
    mcl = _cost(cost_model, "MCL")
    assert cl.fee_per_contract == pytest.approx(1.52)
    assert mcl.commission_per_contract == pytest.approx(0.25)
    assert mcl.fee_per_contract == pytest.approx(0.52)
    # Micro roots are flat-fee, not volume-tiered like the standard contract.
    assert cost_model.commission_per_contract("MES", 50_000) == pytest.approx(0.25)
    assert cost_model.fee_schedule("MES") is not cost_model.fee_schedule("ES")


def test_default_slippage_ticks_per_instrument(cost_model: CostModel) -> None:
    assert cost_model.default_slippage_ticks("ES") == pytest.approx(0.5)
    assert cost_model.default_slippage_ticks("MES") == pytest.approx(1.0)
    assert cost_model.default_slippage_ticks("ZZZ") == 0.0


def test_roll_slippage_multiplier_applies_only_to_roll_legs(
    instrument_master: InstrumentsConfig,
) -> None:
    model = CostModel(
        instrument_master=instrument_master,
        impact_coefficient=8.0,
        min_spread_ticks=0.0,
        base_slippage_ticks=0.0,
    )
    normal = model.transaction_cost("ES", 100, 4_800.0, 0.25, 50.0, bar_volume=1_000.0)
    assert normal.slippage_ticks > 0.0
    roll = model.roll_cost(
        "ES", 100, 4_800.0, 4_802.0, tick_size=0.25, multiplier=50.0, bar_volume=1_000.0
    )
    assert roll.close_leg.slippage_ticks == pytest.approx(normal.slippage_ticks * 1.5)
    assert roll.open_leg.slippage_ticks == pytest.approx(normal.slippage_ticks * 1.5)
    assert roll.close_leg.spread_ticks == pytest.approx(normal.spread_ticks)
    assert roll.to_dict()["roll_slippage_multiplier"] == pytest.approx(1.5)
    # Normal entries remain exactly unchanged.
    again = model.transaction_cost("ES", 100, 4_800.0, 0.25, 50.0, bar_volume=1_000.0)
    assert again == normal
    # Direct is_roll flag and a custom multiplier.
    direct = model.transaction_cost(
        "ES", 100, 4_800.0, 0.25, 50.0, bar_volume=1_000.0, is_roll=True
    )
    assert direct.slippage_ticks == pytest.approx(normal.slippage_ticks * 1.5)
    doubled = CostModel(
        instrument_master=instrument_master,
        impact_coefficient=8.0,
        min_spread_ticks=0.0,
        base_slippage_ticks=0.0,
        roll_slippage_multiplier=2.0,
    )
    doubled_roll = doubled.roll_cost(
        "ES", 100, 4_800.0, 4_802.0, tick_size=0.25, multiplier=50.0, bar_volume=1_000.0
    )
    assert doubled_roll.close_leg.slippage_ticks == pytest.approx(normal.slippage_ticks * 2.0)
    with pytest.raises(ValueError):
        CostModel(roll_slippage_multiplier=-0.1)


def test_load_calibrated_costs_reader_contract(tmp_path: Path) -> None:
    path = _write_calibration(
        tmp_path / "ibkr_calibrated_costs.json",
        {"ES": _root_override(), "NQ": _root_override(broker_commission_per_contract=0.5)},
        account_id="DU7654321",
        warnings=["MCL: whatIfOrder failed"],
    )
    payload = load_calibrated_costs(path)
    assert payload.account_id == "DU7654321"
    assert payload.schema_version == 1
    assert payload.calibrated_at.tzinfo is not None
    assert payload.path == str(path)
    assert payload.warnings == ("MCL: whatIfOrder failed",)
    es = payload.roots["ES"]
    assert isinstance(es, CalibratedCostOverride)
    assert es.broker_commission_per_contract == pytest.approx(0.42)
    assert es.exchange_and_reg_fee_per_contract == pytest.approx(1.50)
    assert es.initial_margin == pytest.approx(15_234.5)
    assert es.maintenance_margin == pytest.approx(13_870.2)
    assert es.commission_source == "ibkr_whatif"
    assert payload.roots["NQ"].broker_commission_per_contract == pytest.approx(0.5)


def test_cost_model_applies_calibration_and_reports_provenance(
    tmp_path: Path, instrument_master: InstrumentsConfig
) -> None:
    path = _write_calibration(
        tmp_path / "cache" / "ibkr_calibrated_costs.json", {"ES": _root_override()}
    )
    model = CostModel(instrument_master=instrument_master, calibrated_costs_path=path)
    assert model.calibration is not None
    assert model.calibration_path == path
    assert model.calibration_error is None
    assert model.commission_per_contract("ES") == pytest.approx(0.42)
    assert model.commission_per_contract("ES", 50_000) == pytest.approx(0.42)
    assert model.fee_per_contract("ES") == pytest.approx(1.50)
    cost = _cost(model, "ES", 2)
    assert cost.commission_per_contract == pytest.approx(0.42)
    assert cost.fee_per_contract == pytest.approx(1.50)
    assert model.fee_provenance("ES") is CalibrationProvenance.IBKR_CALIBRATED
    assert model.fee_provenance("NQ") is CalibrationProvenance.YAML_DEFAULT
    assert model.calibrated_margins("ES") == pytest.approx((15_234.5, 13_870.2))
    assert model.margin_provenance("ES") is CalibrationProvenance.IBKR_CALIBRATED
    assert model.margin_provenance("NQ") is CalibrationProvenance.YAML_DEFAULT
    metadata = model.to_dict()["calibration"]
    assert metadata["source"] == "ibkr_calibrated"  # type: ignore[index]
    assert metadata["applied_roots"] == ["ES"]  # type: ignore[index]
    assert metadata["account_id"] == "DU1234567"  # type: ignore[index]


def test_calibration_absent_corrupt_stale_and_unsupported_fall_back(
    tmp_path: Path, instrument_master: InstrumentsConfig
) -> None:
    absent = CostModel(
        instrument_master=instrument_master, calibrated_costs_path=tmp_path / "missing.json"
    )
    assert absent.calibration is None
    assert absent.calibration_error is None
    assert absent.fee_provenance("ES") is CalibrationProvenance.YAML_DEFAULT
    assert absent.fee_per_contract("ES") == pytest.approx(1.40)

    corrupt_path = tmp_path / "corrupt.json"
    corrupt_path.write_text("{not json")
    corrupt = CostModel(instrument_master=instrument_master, calibrated_costs_path=corrupt_path)
    assert corrupt.calibration is None
    assert corrupt.calibration_error is not None and "not valid JSON" in corrupt.calibration_error
    assert corrupt.fee_per_contract("ES") == pytest.approx(1.40)

    stale_path = _write_calibration(
        tmp_path / "stale.json",
        {"ES": _root_override()},
        calibrated_at=datetime.now(UTC) - timedelta(days=DEFAULT_CALIBRATION_MAX_AGE_DAYS + 1),
    )
    stale = CostModel(instrument_master=instrument_master, calibrated_costs_path=stale_path)
    assert stale.calibration is None
    assert stale.calibration_error is not None and "stale" in stale.calibration_error

    unsupported_path = _write_calibration(
        tmp_path / "unsupported.json", {"ES": _root_override()}, schema_version=2
    )
    unsupported = CostModel(
        instrument_master=instrument_master, calibrated_costs_path=unsupported_path
    )
    assert unsupported.calibration is None
    assert unsupported.calibration_error is not None
    assert "schema_version" in unsupported.calibration_error


def test_calibration_invalid_roots_are_skipped_with_warnings(
    tmp_path: Path, instrument_master: InstrumentsConfig
) -> None:
    path = _write_calibration(
        tmp_path / "partial.json",
        {
            "ES": _root_override(),
            "NQ": {"exchange_and_reg_fee_per_contract": 1.4},  # missing broker
            "": {},
        },
    )
    payload = load_calibrated_costs(path)
    assert set(payload.roots) == {"ES"}
    assert any("NQ" in warning for warning in payload.warnings)
    model = CostModel(instrument_master=instrument_master, calibrated_costs_path=path)
    assert model.commission_per_contract("ES") == pytest.approx(0.42)
    assert model.fee_provenance("NQ") is CalibrationProvenance.YAML_DEFAULT
    assert model.fee_per_contract("NQ") == pytest.approx(1.40)


def test_no_usable_roots_and_missing_calibrated_at_are_rejected(
    tmp_path: Path, instrument_master: InstrumentsConfig
) -> None:
    empty_path = _write_calibration(tmp_path / "empty.json", {})
    with pytest.raises(CalibrationError, match="no usable root overrides"):
        load_calibrated_costs(empty_path)
    model = CostModel(instrument_master=instrument_master, calibrated_costs_path=empty_path)
    assert model.calibration is None and model.calibration_error is not None

    missing_timestamp = tmp_path / "no_ts.json"
    missing_timestamp.write_text(json.dumps({"roots": {"ES": _root_override()}}))
    with pytest.raises(CalibrationError, match="calibrated_at"):
        load_calibrated_costs(missing_timestamp)
    with pytest.raises(CalibrationError, match="cannot read"):
        load_calibrated_costs(tmp_path / "absent.json")


def test_partial_margins_apply_fees_but_not_margin_provenance(
    tmp_path: Path, instrument_master: InstrumentsConfig
) -> None:
    path = _write_calibration(
        tmp_path / "margins.json",
        {"ES": _root_override(maintenance_margin=None)},
    )
    payload = load_calibrated_costs(path)
    assert payload.roots["ES"].initial_margin == pytest.approx(15_234.5)
    assert payload.roots["ES"].maintenance_margin is None
    model = CostModel(instrument_master=instrument_master, calibrated_costs_path=path)
    assert model.fee_provenance("ES") is CalibrationProvenance.IBKR_CALIBRATED
    assert model.calibrated_margins("ES") is None
    assert model.margin_provenance("ES") is CalibrationProvenance.YAML_DEFAULT


def test_calibration_max_age_none_and_future_timestamps(
    tmp_path: Path, instrument_master: InstrumentsConfig
) -> None:
    old = _write_calibration(
        tmp_path / "old.json",
        {"ES": _root_override()},
        calibrated_at=datetime.now(UTC) - timedelta(days=365),
    )
    payload = load_calibrated_costs(old, max_age_days=None)
    assert "ES" in payload.roots
    future = _write_calibration(
        tmp_path / "future.json",
        {"ES": _root_override()},
        calibrated_at=datetime.now(UTC) + timedelta(days=1),
    )
    model = CostModel(instrument_master=instrument_master, calibrated_costs_path=future)
    assert model.calibration is not None
    assert model.commission_per_contract("ES") == pytest.approx(0.42)


def test_reload_calibration_picks_up_new_and_removed_files(
    tmp_path: Path, instrument_master: InstrumentsConfig
) -> None:
    path = _write_calibration(tmp_path / "reload.json", {"ES": _root_override()})
    model = CostModel(instrument_master=instrument_master, calibrated_costs_path=path)
    assert model.commission_per_contract("ES") == pytest.approx(0.42)
    _write_calibration(path, {"ES": _root_override(broker_commission_per_contract=0.55)})
    model.reload_calibration()
    assert model.commission_per_contract("ES") == pytest.approx(0.55)
    path.unlink()
    model.reload_calibration()
    assert model.calibration is None
    assert model.commission_per_contract("ES") == pytest.approx(0.85)


def test_calibration_override_beats_yaml_but_explicit_fee_schedules_win(
    tmp_path: Path, instrument_master: InstrumentsConfig
) -> None:
    path = _write_calibration(tmp_path / "override.json", {"ES": _root_override()})
    explicit = FeeSchedule(commission_tiers=((0, 9.0),), exchange_fee=9.0)
    model = CostModel(
        instrument_master=instrument_master,
        calibrated_costs_path=path,
        fee_schedules={"ES": explicit},
    )
    assert model.fee_schedule("ES") is explicit
    assert model.fee_provenance("ES") is CalibrationProvenance.YAML_DEFAULT


def test_default_calibrated_costs_path_respects_settings_and_env(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    instrument_master: InstrumentsConfig,
) -> None:
    assert default_calibrated_costs_path(settings) == (
        settings.data_dir / "cache" / "ibkr_calibrated_costs.json"
    )
    override = tmp_path / "env-calibration.json"
    monkeypatch.setenv(CALIBRATED_COSTS_ENV_VAR, str(override))
    assert default_calibrated_costs_path(settings) == override
    _write_calibration(override, {"ES": _root_override(broker_commission_per_contract=0.33)})
    model = CostModel(instrument_master=instrument_master)
    assert model.calibration_path == override
    assert model.commission_per_contract("ES") == pytest.approx(0.33)
