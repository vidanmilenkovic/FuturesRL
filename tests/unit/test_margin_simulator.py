"""Unit tests for the futures margin, settlement and forced-liquidation simulator."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from configs.instruments import InstrumentsConfig
from src.backtest.margin_simulator import (
    ForcedLiquidationPlan,
    MarginCall,
    MarginRequirements,
    MarginSimulator,
    SettlementResult,
)
from src.core.types import AccountSnapshot, Position

TIMESTAMP = datetime(2024, 1, 2, 21, 0, tzinfo=UTC)


def make_position(
    root: str = "ES",
    *,
    quantity: int = 2,
    average_price: float = 4800.0,
    market_price: float = 4800.0,
    multiplier: float = 50.0,
    initial_margin: float = 15_000.0,
    maintenance_margin: float = 13_600.0,
    expiration: date = date(2024, 3, 15),
) -> Position:
    return Position(
        instrument_id=root,
        quantity=quantity,
        average_price=average_price,
        market_price=market_price,
        multiplier=multiplier,
        expiration=expiration,
        initial_margin=initial_margin,
        maintenance_margin=maintenance_margin,
        contract_symbol=f"{root}_{expiration:%Y%m%d}",
    )


def test_margin_requirements_and_equity() -> None:
    simulator = MarginSimulator()
    positions = {"ES": make_position(), "NQ": make_position("NQ", quantity=-1)}
    requirements = simulator.margin_requirements(positions)
    assert isinstance(requirements, MarginRequirements)
    assert requirements.initial_margin == pytest.approx(15_000.0 * 2 + 15_000.0)
    assert requirements.maintenance_margin == pytest.approx(13_600.0 * 2 + 13_600.0)
    assert requirements.utilisation == pytest.approx(13_600.0 / 15_000.0)
    assert simulator.unrealized_pnl(positions) == pytest.approx(0.0)
    assert simulator.equity(1_000_000.0, positions) == pytest.approx(1_000_000.0)
    moved = {"ES": make_position(market_price=4810.0)}
    assert simulator.unrealized_pnl(moved) == pytest.approx(2 * 10.0 * 50.0)
    assert simulator.equity(1_000_000.0, moved) == pytest.approx(1_001_000.0)
    empty = simulator.margin_requirements({})
    assert empty.initial_margin == 0.0
    assert empty.maintenance_margin == 0.0
    assert empty.utilisation == 0.0


def test_margin_falls_back_to_instrument_master(
    instrument_master: InstrumentsConfig,
) -> None:
    simulator = MarginSimulator(instrument_master)
    position = make_position(initial_margin=0.0, maintenance_margin=0.0)
    requirements = simulator.margin_requirements({"ES": position})
    assert requirements.initial_margin == pytest.approx(2 * 15_000.0)
    assert requirements.maintenance_margin == pytest.approx(2 * 13_600.0)


def test_maintenance_ratio_fallback_and_clamp() -> None:
    simulator = MarginSimulator(maintenance_ratio=0.5)
    unknown = make_position("ZZZ", initial_margin=1_000.0, maintenance_margin=0.0)
    requirements = simulator.margin_requirements({"ZZZ": unknown})
    assert requirements.initial_margin == pytest.approx(2_000.0)
    assert requirements.maintenance_margin == pytest.approx(1_000.0)
    inverted = make_position("ZZZ", initial_margin=1_000.0, maintenance_margin=2_000.0)
    requirements = simulator.margin_requirements({"ZZZ": inverted})
    assert requirements.maintenance_margin == pytest.approx(2_000.0)
    assert requirements.maintenance_margin <= requirements.initial_margin


def test_position_key_validation() -> None:
    simulator = MarginSimulator()
    with pytest.raises(ValueError):
        simulator.margin_requirements({"NQ": make_position("ES")})
    with pytest.raises(ValueError):
        simulator.equity(1_000_000.0, {"NQ": make_position("ES")})
    with pytest.raises(ValueError):
        simulator.check_margin_call(
            TIMESTAMP, equity=1_000_000.0, positions={"NQ": make_position("ES")}
        )


def test_account_snapshot() -> None:
    simulator = MarginSimulator()
    positions = {"ES": make_position(market_price=4810.0)}
    snapshot = simulator.account_snapshot(TIMESTAMP, cash=1_000_000.0, positions=positions)
    assert isinstance(snapshot, AccountSnapshot)
    assert snapshot.net_liquidation == pytest.approx(1_001_000.0)
    assert snapshot.cash_balance == pytest.approx(1_000_000.0)
    assert snapshot.initial_margin_used == pytest.approx(30_000.0)
    assert snapshot.maintenance_margin_used == pytest.approx(27_200.0)
    assert snapshot.excess_liquidity == pytest.approx(1_001_000.0 - 30_000.0)
    assert snapshot.buying_power == pytest.approx(1_001_000.0 - 30_000.0)
    assert snapshot.margin_utilization == pytest.approx(30_000.0 / 1_001_000.0)
    assert simulator.utilisation(snapshot) == pytest.approx(snapshot.margin_utilization)


def test_account_snapshot_rejects_depleted_account() -> None:
    simulator = MarginSimulator()
    position = make_position(market_price=1_000.0)
    with pytest.raises(ValueError, match="non-positive"):
        simulator.account_snapshot(TIMESTAMP, cash=0.0, positions={"ES": position})


def test_daily_settlement_conserves_equity_and_moves_cash() -> None:
    simulator = MarginSimulator()
    position = make_position(market_price=4800.0)
    result = simulator.settle(
        TIMESTAMP,
        cash=1_000_000.0,
        positions={"ES": position},
        settlement_prices={"ES": 4810.0},
    )
    assert isinstance(result, SettlementResult)
    assert result.settlement_pnl == pytest.approx(2 * 10.0 * 50.0)
    assert result.cash_after == pytest.approx(1_001_000.0)
    assert result.cash_before == pytest.approx(1_000_000.0)
    # The settlement identity: value is moved from unrealised to realised cash.
    assert result.equity_after == pytest.approx(result.equity_before)
    assert result.equity_after == pytest.approx(1_001_000.0)
    settled = result.positions[0]
    assert settled.average_price == pytest.approx(4810.0)
    assert settled.market_price == pytest.approx(4810.0)
    assert settled.quantity == 2
    assert settled.unrealized_pnl == pytest.approx(0.0)
    assert result.initial_margin_used == pytest.approx(30_000.0)
    assert result.maintenance_margin_used == pytest.approx(27_200.0)
    assert result.excess_liquidity == pytest.approx(1_001_000.0 - 30_000.0)
    assert result.margin_utilization == pytest.approx(30_000.0 / 1_001_000.0)
    payload = result.to_dict()
    assert payload["settlement_pnl"] == pytest.approx(1_000.0)
    assert payload["equity_after"] == pytest.approx(1_001_000.0)


def test_settlement_short_position_and_missing_prices() -> None:
    simulator = MarginSimulator()
    short = make_position(quantity=-2, average_price=4800.0, market_price=4790.0)
    result = simulator.settle(TIMESTAMP, cash=1_000_000.0, positions={"ES": short})
    # No price supplied: mark to the position's own market price.
    assert result.settlement_pnl == pytest.approx(2 * 10.0 * 50.0)
    assert result.cash_after == pytest.approx(1_001_000.0)
    empty = simulator.settle(TIMESTAMP, cash=500.0, positions={})
    assert empty.settlement_pnl == 0.0
    assert empty.positions == ()
    assert empty.equity_after == pytest.approx(500.0)


def test_settlement_validation() -> None:
    simulator = MarginSimulator()
    position = make_position()
    with pytest.raises(ValueError):
        simulator.settle(
            TIMESTAMP,
            cash=1_000_000.0,
            positions={"ES": position},
            settlement_prices={"ES": -1.0},
        )
    with pytest.raises(ValueError):
        simulator.settle(TIMESTAMP, cash=float("nan"), positions={"ES": position})


def test_margin_call_detection() -> None:
    simulator = MarginSimulator()
    positions = {"ES": make_position(quantity=2)}
    healthy = simulator.check_margin_call(TIMESTAMP, equity=100_000.0, positions=positions)
    assert isinstance(healthy, MarginCall)
    assert healthy.triggered is False
    assert healthy.deficit == 0.0
    assert healthy.positions == tuple(positions.values())
    breach = simulator.check_margin_call(TIMESTAMP, equity=20_000.0, positions=positions)
    assert breach.triggered is True
    assert breach.deficit == pytest.approx(27_200.0 - 20_000.0)
    assert breach.maintenance_margin_used == pytest.approx(27_200.0)
    assert breach.utilisation == pytest.approx(30_000.0 / 20_000.0)
    assert "maintenance margin" in breach.reason
    flat = simulator.check_margin_call(TIMESTAMP, equity=0.0, positions={})
    assert flat.triggered is False
    zero_equity = simulator.check_margin_call(TIMESTAMP, equity=0.0, positions=positions)
    assert zero_equity.triggered is True
    assert zero_equity.utilisation == 0.0
    buffered = MarginSimulator(margin_call_buffer=0.10)
    assert (
        buffered.check_margin_call(TIMESTAMP, equity=29_000.0, positions=positions).triggered
        is True
    )
    assert (
        simulator.check_margin_call(TIMESTAMP, equity=29_000.0, positions=positions).triggered
        is False
    )
    assert breach.to_dict()["triggered"] is True


def test_forced_liquidation_plan(instrument_master: InstrumentsConfig) -> None:
    simulator = MarginSimulator(instrument_master)
    positions = {"ES": make_position(quantity=2)}
    call = simulator.check_margin_call(TIMESTAMP, equity=10_000.0, positions=positions)
    plan = simulator.forced_liquidation_plan(call)
    assert isinstance(plan, ForcedLiquidationPlan)
    assert plan.positions == call.positions
    assert plan.estimated_slippage_cost == pytest.approx(2.0 * 0.25 * 50.0 * 2)
    assert plan.estimated_penalty_fee == pytest.approx(15.0 * 2)
    assert plan.total_estimated_cost == pytest.approx(
        plan.estimated_slippage_cost + plan.estimated_penalty_fee
    )
    assert plan.to_dict()["total_estimated_cost"] == pytest.approx(plan.total_estimated_cost)
    with pytest.raises(ValueError):
        simulator.forced_liquidation_plan(
            simulator.check_margin_call(TIMESTAMP, equity=1_000_000.0, positions=positions)
        )
    # Without an instrument master the tick size is unknown -> no slippage estimate.
    bare = MarginSimulator()
    bare_plan = bare.forced_liquidation_plan(
        bare.check_margin_call(TIMESTAMP, equity=0.0, positions=positions)
    )
    assert bare_plan.estimated_slippage_cost == pytest.approx(0.0)
    assert bare_plan.estimated_penalty_fee == pytest.approx(30.0)


def test_margin_simulator_validation() -> None:
    with pytest.raises(ValueError):
        MarginSimulator(maintenance_ratio=1.5)
    with pytest.raises(ValueError):
        MarginSimulator(maintenance_ratio=float("nan"))
    with pytest.raises(ValueError):
        MarginSimulator(penalty_ticks=-1.0)
    with pytest.raises(ValueError):
        MarginSimulator(penalty_fee_per_contract=-1.0)
    with pytest.raises(ValueError):
        MarginSimulator(margin_call_buffer=-1.0)
    with pytest.raises(ValueError):
        MarginRequirements(initial_margin=-1.0)
    with pytest.raises(ValueError):
        MarginRequirements(maintenance_margin=float("inf"))
    payload = MarginSimulator().to_dict()
    assert payload["maintenance_ratio"] == pytest.approx(0.90)
    assert payload["penalty_ticks"] == pytest.approx(2.0)
