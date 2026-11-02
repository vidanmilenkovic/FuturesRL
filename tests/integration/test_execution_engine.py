"""End-to-end execution-engine integration test (offline fake IBKR venue).

Exercises the real component boundaries together: the deterministic
``ContractTranslator`` producing targets, the OMS pre-trade gate + submission,
order/fill aggregation, the connection manager's contract cache, the native BAG
roll and the reconciliation / kill-switch loop.  Every venue interaction is
served by the in-memory fake transport, so the test is deterministic and
network-free.
"""

from __future__ import annotations

from datetime import date

import pytest
from configs.settings import Settings
from src.core.types import (
    ContractTargetSet,
    MarketState,
    OrderStatus,
    RoundingMode,
    WeightVector,
)
from src.execution.ibkr_client import MarginImpact, VenuePosition
from src.execution.ibkr_roller import ContractRoller, select_roll_expirations
from src.execution.oms import OrderManagementSystem, PreTradeRiskGate
from src.execution.reconciler import PositionReconciler

from tests.fixtures.market import DEFAULT_PRICES
from tests.unit.test_ibkr_client import FakeClock, FakeIBClient, make_manager

pytestmark = pytest.mark.integration


def _price_provider(root: str) -> float | None:
    return DEFAULT_PRICES.get(root)


async def _make_execution_stack(
    settings: Settings, client: FakeIBClient, clock: FakeClock
) -> tuple[OrderManagementSystem, PreTradeRiskGate]:
    manager = make_manager(settings, client, clock, price_provider=_price_provider)
    await manager.connect()

    async def account_provider() -> object:
        return await manager.fetch_account()

    gate = PreTradeRiskGate(
        settings=settings,
        what_if=manager.what_if_order,
        account_provider=account_provider,  # type: ignore[arg-type]
        price_provider=_price_provider,
        instruments=settings.instruments,
        clock=clock,
        rate_capacity=100.0,
    )
    oms = OrderManagementSystem(
        manager=manager,
        settings=settings,
        clock=clock,
        sleep=clock.sleep,
        gate=gate,
    )
    return oms, gate


def _targets_from_translator(weights: WeightVector, market_state: MarketState) -> ContractTargetSet:
    translator_module = pytest.importorskip("src.risk.contract_translator")
    try:
        translator = translator_module.ContractTranslator(
            rounding_mode=RoundingMode.NEAREST,
            max_margin_utilization=0.60,
            enable_micro_routing=True,
        )
    except TypeError:  # pragma: no cover - broader constructor signature
        translator = translator_module.ContractTranslator()
    targets = translator.translate(weights, market_state)
    assert isinstance(targets, ContractTargetSet)
    return targets


async def test_execution_engine_end_to_end(
    settings: Settings,
    market_state: MarketState,
    weights: WeightVector,
) -> None:
    clock = FakeClock()
    client = FakeIBClient()
    oms, gate = await _make_execution_stack(settings, client, clock)
    manager = oms.manager
    targets = _targets_from_translator(weights, market_state)
    assert targets, "translator must produce at least one contract target"

    # -- 1. rebalance: deterministic deltas through the mandatory pre-trade gate
    receipts = await oms.submit_targets(
        targets, current_positions={}, timestamp=market_state.timestamp
    )
    assert receipts
    assert all(receipt.status is OrderStatus.FILLED for receipt in receipts)
    expected = {target.instrument_id: target.quantity for target in targets if target.quantity}
    submitted_ids = {request.order_ref for _, request in client.placed_orders}
    assert submitted_ids == {
        order.order_id for order in oms.orders_from_targets(targets, {}, market_state.timestamp)
    }
    assert len(oms.all_fills()) == len(receipts)
    # the same rebalance at the same timestamp is idempotent and does not re-fire
    before = len(client.placed_orders)
    await oms.submit_targets(targets, current_positions={}, timestamp=market_state.timestamp)
    assert len(client.placed_orders) == before
    # gate audit trail recorded the margin utilisation of every accepted order
    assert all(decision.accepted for decision in gate.decisions)
    assert gate.decisions[-1].initial_margin_utilization is not None

    # -- 2. broker positions match the internal expectation after the fills
    client.position_rows = [
        VenuePosition(
            root_symbol=target.instrument_id,
            quantity=target.quantity,
            avg_cost=target.quantity * target.price * target.multiplier,
            multiplier=target.multiplier,
            expiration=target.expiration,
        )
        for target in targets
        if target.quantity
    ]
    reconciler = PositionReconciler(
        expected_positions=expected,
        broker_positions=manager.fetch_positions,
        clock=clock,
        sleep=clock.sleep,
    )
    assert await reconciler.reconcile()
    assert not reconciler.halted

    # -- 3. native BAG roll of the front ES contract
    front, back = select_roll_expirations(
        settings.instruments.get("ES"), market_state.timestamp.date()
    )
    assert front < back
    roller = ContractRoller(
        manager,
        instrument_master=settings.instruments,
        clock=clock,
        sleep=clock.sleep,
    )
    es_quantity = max(1, abs(expected.get("ES", 0)))
    roll = await roller.execute_roll("ES", es_quantity, as_of=market_state.timestamp.date())
    assert roll.bag_used
    assert roll.complete
    assert client.placed_bags
    assert client.placed_bags[0].ratio_balanced
    assert [leg.action.value for leg in client.placed_bags[0].legs] == ["SELL", "BUY"]

    # -- 4. unapproved drift halts rebalancing; approved drift releases it
    client.position_rows[0] = VenuePosition(
        root_symbol=client.position_rows[0].root_symbol,
        quantity=client.position_rows[0].quantity + 1,
        avg_cost=client.position_rows[0].avg_cost,
        multiplier=client.position_rows[0].multiplier,
    )
    assert not await reconciler.reconcile()
    assert reconciler.halted
    drifted_instrument = client.position_rows[0].root_symbol
    reconciler.approve_drift({drifted_instrument: 1})
    assert await reconciler.reconcile()

    # -- 5. emergency flatten: cancel working orders, close everything, lock
    clock.advance(60.0)
    client.margin_impact = MarginImpact(
        initial_margin_after=1_000.0, maintenance_margin_after=500.0
    )
    flatten_receipts = await oms.flatten_all()
    closes = flatten_receipts[-len(client.position_rows) :]
    assert all(receipt.status is OrderStatus.FILLED for receipt in closes)
    assert oms.is_locked
    assert gate.lock_reason == "emergency flatten_all"
    oms.reset_kill_switch()
    assert not oms.is_locked

    # -- 6. contract resolution was cached: no duplicate (root, month) requests
    keys = [
        (request.root, request.last_trade_date_or_contract_month)
        for request in client.qualify_requests
    ]
    assert len(keys) == len(set(keys))
    assert manager.connection_attempts == 1
    await oms.disconnect()


async def test_execution_engine_roll_falls_back_to_legs_on_unsupported_bag(
    settings: Settings, market_state: MarketState
) -> None:
    from src.execution.ibkr_client import IBKRBagUnsupportedError

    clock = FakeClock()
    client = FakeIBClient()
    manager = make_manager(settings, client, clock, price_provider=_price_provider)
    await manager.connect()
    roller = ContractRoller(
        manager,
        instrument_master=settings.instruments,
        clock=clock,
        sleep=clock.sleep,
    )
    client.bag_raises = IBKRBagUnsupportedError("BAG unsupported on this gateway")
    as_of: date = market_state.timestamp.date()
    execution = await roller.execute_roll("CL", -2, as_of=as_of)
    assert not execution.bag_used
    assert execution.complete
    assert [request.action for _, request in client.placed_orders] == ["BUY", "SELL"]


async def test_execution_engine_gate_rejects_margin_breach(
    settings: Settings, market_state: MarketState
) -> None:
    clock = FakeClock()
    client = FakeIBClient()
    oms, _gate = await _make_execution_stack(settings, client, clock)
    client.margin_impact = MarginImpact(
        initial_margin_after=950_000.0, maintenance_margin_after=930_000.0
    )
    targets = _targets_from_translator(WeightVector.from_mapping({"ES": -0.10}), market_state)
    receipts = await oms.submit_targets(
        targets, current_positions={}, timestamp=market_state.timestamp
    )
    assert receipts
    assert all(receipt.status is OrderStatus.REJECTED for receipt in receipts)
    assert all("margin limit" in receipt.message for receipt in receipts)
    assert client.placed_orders == []
