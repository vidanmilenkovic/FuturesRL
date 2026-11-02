"""Unit tests for the OMS, the deterministic delta logic and the pre-trade gate.

All IBKR interaction is faked through ``tests.unit.test_ibkr_client.FakeIBClient``;
the tests are offline, deterministic and cover the mandatory ordering of the
pre-trade checks, idempotent order ids, brackets, kill-switch and fill
aggregation.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, cast

import pytest
from configs.settings import RiskSettings, Settings
from src.core.interfaces import BrokerAdapter
from src.core.interfaces import PreTradeRiskGate as PreTradeRiskGateProtocol
from src.core.types import (
    ContractTargetSet,
    OrderEvent,
    OrderReceipt,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from src.execution.ibkr_client import (
    IBKRDataError,
    IBOrderRequest,
    MarginImpact,
    VenueOrderSnapshot,
    VenuePosition,
)
from src.execution.oms import (
    OrderManagementSystem,
    PreTradeRiskGate,
    build_orders_from_targets,
    deterministic_order_id,
)
from src.risk.state_store import InMemoryRiskStateStore, PersistedRiskState, RateLimiterState

from tests.fixtures.market import (
    FIXED_TIMESTAMP,
    make_account_snapshot,
    make_contract_target,
    make_contract_targets,
)
from tests.unit.test_ibkr_client import FakeClock, FakeIBClient, filled_trade, make_manager

BUY = OrderSide.BUY
SELL = OrderSide.SELL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture()
def fake_client() -> FakeIBClient:
    return FakeIBClient()


class FakeMonotonic:
    """Injectable monotonic clock for token-bucket / duplicate-window tests."""

    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def make_event(
    instrument_id: str = "ES",
    *,
    side: OrderSide = BUY,
    quantity: int = 1,
    order_id: str = "order-1",
    order_type: OrderType = OrderType.MARKET,
    limit_price: float | None = None,
    stop_price: float | None = None,
    timestamp: datetime = FIXED_TIMESTAMP,
) -> OrderEvent:
    return OrderEvent(
        order_id=order_id,
        timestamp=timestamp,
        instrument_id=instrument_id,
        side=side,
        quantity=quantity,
        order_type=order_type,
        time_in_force=TimeInForce.DAY,
        limit_price=limit_price,
        stop_price=stop_price,
    )


def make_gate(
    settings: Settings,
    clock: FakeClock,
    *,
    margin_impact: MarginImpact | None = None,
    price_provider: Any = None,
    monotonic: Any = None,
    max_order_notional: float | None = None,
    rate_capacity: float | None = None,
    what_if: Any = None,
    rate_limiter_state: RateLimiterState | None = None,
    wall_clock: Any = None,
    persist_rate_limiter: Any = None,
) -> PreTradeRiskGate:
    impact = margin_impact or MarginImpact(
        initial_margin_after=135_000.0, maintenance_margin_after=123_600.0
    )

    async def default_what_if(_order: OrderEvent) -> MarginImpact:
        return impact

    async def account_provider() -> Any:
        return make_account_snapshot()

    return PreTradeRiskGate(
        settings=settings,
        what_if=what_if if what_if is not None else default_what_if,
        account_provider=account_provider,
        price_provider=price_provider if price_provider is not None else lambda _root: 4800.0,
        instruments=settings.instruments,
        clock=clock,
        monotonic=monotonic,
        max_order_notional=max_order_notional,
        rate_capacity=rate_capacity,
        rate_limiter_state=rate_limiter_state,
        wall_clock=wall_clock,
        persist_rate_limiter=persist_rate_limiter,
    )


def make_oms(
    settings: Settings,
    client: FakeIBClient,
    clock: FakeClock,
    *,
    prices: dict[str, float] | None = None,
    gate: PreTradeRiskGate | None = None,
    state_store: Any = None,
    wall_clock: Any = None,
) -> tuple[OrderManagementSystem, Any]:
    price_map = prices or {"ES": 4800.0, "NQ": 16800.0, "CL": 72.5, "GC": 2050.0, "ZN": 111.5}
    manager = make_manager(settings, client, clock, price_provider=lambda root: price_map.get(root))
    oms = OrderManagementSystem(
        manager=manager,
        settings=settings,
        clock=clock,
        sleep=clock.sleep,
        gate=gate,
        state_store=state_store,
        wall_clock=wall_clock,
    )
    return oms, manager


class BrokenGate:
    """Gate double whose ``check`` crashes, exercising the OMS safety net."""

    def __init__(self) -> None:
        self.is_locked = False
        self.lock_reason = ""

    async def check(self, order: OrderEvent, *, emergency: bool = False) -> OrderReceipt | None:
        raise RuntimeError("gate exploded")


# ---------------------------------------------------------------------------
# Deterministic target -> delta translation
# ---------------------------------------------------------------------------


def test_orders_from_targets_is_deterministic_and_propagates_contracts() -> None:
    targets = make_contract_targets({"ES": 2, "NQ": 1, "CL": -3})
    current = {"ES": 1, "NQ": 0, "GC": 4}
    orders = build_orders_from_targets(targets, current, FIXED_TIMESTAMP)
    by_instrument = {order.instrument_id: order for order in orders}
    assert set(by_instrument) == {"ES", "NQ", "CL", "GC"}
    assert by_instrument["ES"].side is BUY
    assert by_instrument["ES"].quantity == 1
    assert by_instrument["ES"].order_id.startswith("ord-")
    assert by_instrument["ES"].contract_symbol == "ES_TEST"
    assert by_instrument["ES"].expiration == date(2024, 3, 15)
    assert by_instrument["CL"].side is SELL
    assert by_instrument["CL"].quantity == 3
    assert by_instrument["GC"].side is SELL
    assert by_instrument["GC"].quantity == 4
    assert by_instrument["NQ"].tag == "rebalance"
    again = build_orders_from_targets(targets, current, FIXED_TIMESTAMP)
    assert [order.order_id for order in orders] == [order.order_id for order in again]
    next_second = FIXED_TIMESTAMP.replace(microsecond=500_000)
    assert [
        order.order_id for order in build_orders_from_targets(targets, current, next_second)
    ] == [order.order_id for order in orders]  # same one-second bucket
    later = build_orders_from_targets(targets, current, FIXED_TIMESTAMP.replace(second=1))
    assert later[0].order_id != orders[0].order_id


def test_orders_from_targets_skips_zero_deltas_and_closes_unlisted_positions() -> None:
    targets = make_contract_targets({"ES": 2})
    assert build_orders_from_targets(targets, {"ES": 2}, FIXED_TIMESTAMP) == []
    orders = build_orders_from_targets(targets, {"ES": 2, "NQ": -3}, FIXED_TIMESTAMP)
    assert len(orders) == 1
    assert orders[0].instrument_id == "NQ"
    assert orders[0].side is BUY
    assert orders[0].quantity == 3


def test_orders_from_targets_supports_synthetic_flatten() -> None:
    current = {"ES": 2, "CL": -3}
    flat = build_orders_from_targets(
        ContractTargetSet.empty(), current, FIXED_TIMESTAMP, flatten=True, tag="flatten"
    )
    assert [(order.instrument_id, order.side, order.quantity) for order in flat] == [
        ("CL", BUY, 3),
        ("ES", SELL, 2),
    ]
    assert all(order.tag == "flatten" for order in flat)
    # An empty target set without the explicit flag also flattens the book.
    implicit = build_orders_from_targets(ContractTargetSet.empty(), current, FIXED_TIMESTAMP)
    assert [(order.instrument_id, order.side) for order in implicit] == [
        ("CL", BUY),
        ("ES", SELL),
    ]


def test_orders_from_targets_honours_explicit_zero_target() -> None:
    zero_target = ContractTargetSet((make_contract_target("ES", 0),))
    orders = build_orders_from_targets(zero_target, {"ES": 2}, FIXED_TIMESTAMP)
    assert len(orders) == 1
    assert orders[0].side is SELL
    assert orders[0].quantity == 2


def test_deterministic_order_id_properties() -> None:
    base = deterministic_order_id(
        instrument_id="ES", side=BUY, quantity=2, timestamp=FIXED_TIMESTAMP
    )
    assert base == deterministic_order_id(
        instrument_id="ES", side=BUY, quantity=2, timestamp=FIXED_TIMESTAMP
    )
    assert base != deterministic_order_id(
        instrument_id="ES", side=SELL, quantity=2, timestamp=FIXED_TIMESTAMP
    )
    assert base != deterministic_order_id(
        instrument_id="NQ", side=BUY, quantity=2, timestamp=FIXED_TIMESTAMP
    )
    assert len(base) <= 40  # IBKR orderRef length limit


# ---------------------------------------------------------------------------
# IBOrderRequest wire mapping
# ---------------------------------------------------------------------------


def test_ib_order_request_maps_every_order_type() -> None:
    market = IBOrderRequest.from_order_event(make_event(order_type=OrderType.MARKET))
    assert (market.order_type, market.limit_price, market.algo_strategy) == ("MKT", None, None)
    limit = IBOrderRequest.from_order_event(
        make_event(order_type=OrderType.LIMIT, limit_price=4800.0)
    )
    assert (limit.order_type, limit.limit_price) == ("LMT", 4800.0)
    stop = IBOrderRequest.from_order_event(
        make_event(side=SELL, order_type=OrderType.STOP, stop_price=4700.0)
    )
    assert (stop.order_type, stop.stop_price, stop.action) == ("STP", 4700.0, "SELL")
    stop_limit = IBOrderRequest.from_order_event(
        make_event(order_type=OrderType.STOP_LIMIT, limit_price=4700.0, stop_price=4690.0)
    )
    assert stop_limit.order_type == "STP LMT"
    adaptive = IBOrderRequest.from_order_event(make_event(order_type=OrderType.ADAPTIVE))
    assert adaptive.order_type == "MKT"
    assert adaptive.algo_strategy == "Adaptive"
    assert adaptive.algo_params == (("adaptivePriority", "Normal"),)
    adaptive_limit = IBOrderRequest.from_order_event(
        make_event(order_type=OrderType.ADAPTIVE, limit_price=4800.0)
    )
    assert (adaptive_limit.order_type, adaptive_limit.algo_strategy) == ("LMT", "Adaptive")
    twap = IBOrderRequest.from_order_event(make_event(order_type=OrderType.TWAP))
    assert (twap.algo_strategy, twap.algo_params) == ("TWAP", (("strategyType", "TWAP"),))
    vwap = IBOrderRequest.from_order_event(make_event(order_type=OrderType.VWAP))
    assert vwap.algo_strategy == "VWAP"
    with pytest.raises(IBKRDataError, match="ContractRoller"):
        IBOrderRequest.from_order_event(make_event(order_type=OrderType.CALENDAR_SPREAD))


def test_ib_order_request_validates_fields() -> None:
    with pytest.raises(ValueError):
        IBOrderRequest(order_ref="", action="BUY", quantity=1, order_type="MKT")
    with pytest.raises(ValueError):
        IBOrderRequest(order_ref="r", action="HOLD", quantity=1, order_type="MKT")
    with pytest.raises(ValueError):
        IBOrderRequest(order_ref="r", action="BUY", quantity=0, order_type="MKT")
    with pytest.raises(ValueError):
        IBOrderRequest(order_ref="r", action="BUY", quantity=1, order_type="MKT", limit_price=-1)


# ---------------------------------------------------------------------------
# Pre-trade risk gate
# ---------------------------------------------------------------------------


async def test_gate_kill_switch_blocks_and_resets(settings: Settings, clock: FakeClock) -> None:
    gate = make_gate(settings, clock)
    order = make_event()
    gate.lock("daily loss limit")
    rejection = await gate.check(order)
    assert rejection is not None
    assert rejection.status is OrderStatus.REJECTED
    assert "kill switch" in rejection.message
    assert gate.is_locked
    assert gate.rejections and gate.rejections[-1].order_id == order.order_id
    gate.reset_kill_switch()
    assert not gate.is_locked
    assert await gate.check(order) is None
    assert gate.decisions[-1].accepted


async def test_gate_fat_finger_contract_limit(settings: Settings, clock: FakeClock) -> None:
    gate = make_gate(settings, clock)
    oversized = make_event(quantity=settings.risk.max_order_contracts + 1)
    rejection = await gate.check(oversized)
    assert rejection is not None
    assert "fat-finger" in rejection.message
    assert "max_order_contracts" in rejection.message


async def test_gate_fat_finger_notional_cap(settings: Settings, clock: FakeClock) -> None:
    gate = make_gate(settings, clock, max_order_notional=100_000.0)
    limit_order = make_event(order_type=OrderType.LIMIT, limit_price=4800.0)
    rejection = await gate.check(limit_order)  # 1 * 4800 * 50 = 240,000 > 100,000
    assert rejection is not None
    assert "notional" in rejection.message
    generous = make_gate(settings, clock, max_order_notional=500_000.0)
    assert await generous.check(limit_order) is None


async def test_gate_notional_cap_defaults_to_gross_leverage(
    settings: Settings, clock: FakeClock
) -> None:
    gate = make_gate(settings, clock)
    # 1 ES @ 4800 * 50 = 240k << 3.0 * 1,000,000 NLV
    assert await gate.check(make_event()) is None


async def test_gate_requires_a_mark_price_for_market_orders(
    settings: Settings, clock: FakeClock
) -> None:
    gate_no_provider = PreTradeRiskGate(
        settings=settings,
        what_if=pass_what_if,
        account_provider=account_provider,
        price_provider=None,
        instruments=settings.instruments,
        clock=clock,
    )
    rejection = await gate_no_provider.check(make_event())
    assert rejection is not None
    assert "no mark price" in rejection.message
    # A limit price resolves it without any market data.
    limit_order = make_event(order_type=OrderType.LIMIT, limit_price=4800.0)
    assert await gate_no_provider.check(limit_order) is None


async def pass_what_if(_order: OrderEvent) -> MarginImpact:
    """Margin simulation stub that always passes the utilisation limit."""
    return MarginImpact(initial_margin_after=1000.0, maintenance_margin_after=500.0)


async def account_provider() -> Any:
    return make_account_snapshot()


async def test_gate_rejects_unknown_instrument(settings: Settings, clock: FakeClock) -> None:
    gate = make_gate(settings, clock, price_provider=lambda _root: 100.0)
    rejection = await gate.check(make_event("ZZZ"))
    assert rejection is not None
    assert "instrument master" in rejection.message


async def test_gate_rate_limiter_token_bucket(settings: Settings, clock: FakeClock) -> None:
    limited = Settings(
        environment="backtest",
        instruments_path=settings.instruments_path,
        risk=RiskSettings(order_rate_limit_per_second=1.0, duplicate_order_window_seconds=0.0),
    )
    monotonic = FakeMonotonic()
    gate = make_gate(limited, clock, monotonic=monotonic, rate_capacity=1.0)
    assert await gate.check(make_event("ES", order_id="a")) is None
    blocked = await gate.check(make_event("NQ", order_id="b"))
    assert blocked is not None
    assert "rate limit" in blocked.message
    monotonic.advance(1.0)
    assert await gate.check(make_event("NQ", order_id="c")) is None


async def test_gate_duplicate_order_window(settings: Settings, clock: FakeClock) -> None:
    duplicated = Settings(
        environment="backtest",
        instruments_path=settings.instruments_path,
        risk=RiskSettings(order_rate_limit_per_second=100.0, duplicate_order_window_seconds=5.0),
    )
    monotonic = FakeMonotonic()
    gate = make_gate(duplicated, clock, monotonic=monotonic)
    assert await gate.check(make_event(order_id="a")) is None
    duplicate = await gate.check(make_event(order_id="b"))
    assert duplicate is not None
    assert "duplicate order" in duplicate.message
    monotonic.advance(5.5)
    assert await gate.check(make_event(order_id="c")) is None


async def test_gate_margin_utilisation_limit(settings: Settings, clock: FakeClock) -> None:
    heavy = MarginImpact(initial_margin_after=900_000.0, maintenance_margin_after=800_000.0)
    gate = make_gate(settings, clock, margin_impact=heavy)
    rejection = await gate.check(make_event())
    assert rejection is not None
    assert "margin limit" in rejection.message
    assert "90.00%" in rejection.message


async def test_gate_rejects_incomplete_margin_simulation(
    settings: Settings, clock: FakeClock
) -> None:
    gate = make_gate(settings, clock, margin_impact=MarginImpact())
    rejection = await gate.check(make_event())
    assert rejection is not None
    assert "incomplete" in rejection.message


async def test_gate_rejects_when_what_if_fails(settings: Settings, clock: FakeClock) -> None:
    async def failing(_order: OrderEvent) -> MarginImpact:
        raise ConnectionError("TWS lost")

    gate = make_gate(settings, clock, what_if=failing)
    rejection = await gate.check(make_event())
    assert rejection is not None
    assert "whatIfOrder" in rejection.message
    assert "TWS lost" in rejection.message


async def test_gate_records_margin_utilisation_on_acceptance(
    settings: Settings, clock: FakeClock
) -> None:
    gate = make_gate(settings, clock)
    assert await gate.check(make_event()) is None
    decision = gate.decisions[-1]
    assert decision.accepted
    assert decision.initial_margin_utilization == pytest.approx(0.135)
    assert decision.maintenance_margin_utilization == pytest.approx(0.1236)


async def test_gate_emergency_mode_bypasses_lock_but_keeps_margin(
    settings: Settings, clock: FakeClock
) -> None:
    gate = make_gate(settings, clock)
    gate.lock("flattening")
    order = make_event(quantity=settings.risk.max_order_contracts * 2)
    assert await gate.check(order) is not None  # locked
    # emergency: lock and fat-finger are skipped, margin simulation still runs
    assert await gate.check(order, emergency=True) is None
    heavy = make_gate(
        settings,
        clock,
        margin_impact=MarginImpact(
            initial_margin_after=990_000.0, maintenance_margin_after=980_000.0
        ),
    )
    heavy.lock("flattening")
    emergency_rejection = await heavy.check(order, emergency=True)
    assert emergency_rejection is not None
    assert "margin limit" in emergency_rejection.message


async def test_gate_rejects_when_account_fetch_fails(settings: Settings, clock: FakeClock) -> None:
    async def failing_account() -> Any:
        raise ConnectionError("account unavailable")

    gate = PreTradeRiskGate(
        settings=settings,
        what_if=pass_what_if,
        account_provider=failing_account,
        price_provider=lambda _root: 4800.0,
        instruments=settings.instruments,
        clock=clock,
    )
    rejection = await gate.check(make_event())
    assert rejection is not None
    assert "cannot fetch account" in rejection.message


# ---------------------------------------------------------------------------
# OMS submission, tracking and brackets
# ---------------------------------------------------------------------------


async def test_submit_order_tracks_status_and_aggregates_fills(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    oms, manager = make_oms(settings, fake_client, clock)
    await manager.connect()
    receipt = await oms.submit_order(make_event(quantity=2))
    assert receipt.status is OrderStatus.FILLED
    assert receipt.filled_quantity == 2
    assert receipt.average_price == pytest.approx(4800.0)
    assert receipt.commission == pytest.approx(1.0)
    assert oms.order_status("order-1") is OrderStatus.FILLED
    assert oms.fills("order-1")[0].quantity == 2
    assert oms.all_fills()[0].instrument_id == "ES"
    assert oms.order_receipt("order-1") is not None
    assert oms.open_order_ids == ()
    assert fake_client.placed_orders[0][1].order_ref == "order-1"


async def test_submit_order_is_idempotent_after_execution(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    oms, manager = make_oms(settings, fake_client, clock)
    await manager.connect()
    order = make_event()
    await oms.submit_order(order)
    await oms.submit_order(order)  # retry must not re-transmit a filled order
    assert len(fake_client.placed_orders) == 1


async def test_submit_order_blocked_by_kill_switch(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    gate = make_gate(settings, clock)
    gate.lock("manual")
    oms, manager = make_oms(settings, fake_client, clock, gate=gate)
    await manager.connect()
    receipt = await oms.submit_order(make_event())
    assert receipt.status is OrderStatus.REJECTED
    assert "kill switch" in receipt.message
    assert fake_client.placed_orders == []


async def test_submit_order_handles_transport_failure(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    oms, manager = make_oms(settings, fake_client, clock)
    await manager.connect()
    fake_client.place_raises = ConnectionError("socket closed")
    receipt = await oms.submit_order(make_event())
    assert receipt.status is OrderStatus.REJECTED
    assert "order transmission failed" in receipt.message


async def test_submit_order_survives_crashing_gate(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    oms, manager = make_oms(settings, fake_client, clock, gate=cast(PreTradeRiskGate, BrokenGate()))
    await manager.connect()
    receipt = await oms.submit_order(make_event())
    assert receipt.status is OrderStatus.REJECTED
    assert "pre-trade gate failure" in receipt.message
    assert fake_client.placed_orders == []


async def test_submit_bracket_creates_native_stop_child(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    oms, manager = make_oms(settings, fake_client, clock)
    await manager.connect()
    entry = make_event(quantity=2)
    receipts = await oms.submit_bracket(entry, 4700.0)
    assert [receipt.status for receipt in receipts] == [OrderStatus.FILLED, OrderStatus.FILLED]
    assert receipts[1].order_id == f"{entry.order_id}-sl"
    _contract, parent, child = fake_client.brackets[0]
    assert parent.order_ref == entry.order_id
    assert parent.order_type == "MKT"
    assert child.order_ref == f"{entry.order_id}-sl"
    assert child.order_type == "STP"
    assert child.stop_price == pytest.approx(4700.0)
    assert child.action == "SELL"
    assert oms.order_status(child.order_ref) is OrderStatus.FILLED
    assert len(oms.all_fills()) == 2
    single = await oms.submit_order(make_event(order_id="bracket-2"), stop_price=4700.0)
    assert single.status is OrderStatus.FILLED
    assert fake_client.brackets[1][2].order_ref == "bracket-2-sl"


async def test_submit_bracket_cancels_entry_when_stop_child_rejected(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    oms, manager = make_oms(settings, fake_client, clock)
    await manager.connect()
    entry = make_event(quantity=1)
    fake_client.trade_overrides[f"{entry.order_id}-sl"] = filled_trade(
        f"{entry.order_id}-sl", status=OrderStatus.REJECTED, message="stop rejected"
    )
    receipts = await oms.submit_bracket(entry, 4700.0)
    assert receipts[0].status is OrderStatus.FILLED
    assert receipts[1].status is OrderStatus.REJECTED
    assert entry.order_id in fake_client.cancelled


async def test_submit_bracket_stop_gate_rejection_places_nothing(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    gate = make_gate(settings, clock, max_order_notional=100_000.0)
    oms, manager = make_oms(settings, fake_client, clock, gate=gate)
    await manager.connect()
    # both the entry and the stop child exceed the notional cap, so the gate
    # rejects the bracket before anything is transmitted
    receipts = await oms.submit_bracket(
        make_event(order_type=OrderType.LIMIT, limit_price=4800.0), 4700.0
    )
    assert all(receipt.status is OrderStatus.REJECTED for receipt in receipts)
    assert fake_client.brackets == []


async def test_submit_targets_batches_delta_orders(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    oms, manager = make_oms(settings, fake_client, clock)
    await manager.connect()
    targets = make_contract_targets({"ES": 2, "NQ": 1})
    receipts = await oms.submit_targets(targets, {"ES": 1}, FIXED_TIMESTAMP)
    assert len(receipts) == 2
    assert all(receipt.status is OrderStatus.FILLED for receipt in receipts)
    submitted = {request.order_ref for _, request in fake_client.placed_orders}
    assert len(submitted) == 2
    expected_ids = {
        order.order_id for order in oms.orders_from_targets(targets, {"ES": 1}, FIXED_TIMESTAMP)
    }
    assert submitted == expected_ids


async def test_submit_targets_with_stop_prices_uses_brackets(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    oms, manager = make_oms(settings, fake_client, clock)
    await manager.connect()
    targets = make_contract_targets({"ES": 1})
    receipts = await oms.submit_targets(targets, {}, FIXED_TIMESTAMP, stop_prices={"ES": 4700.0})
    assert len(receipts) == 2
    assert len(fake_client.brackets) == 1


async def test_submit_targets_flatten_flag(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    oms, manager = make_oms(settings, fake_client, clock)
    await manager.connect()
    receipts = await oms.submit_targets(
        make_contract_targets({"ES": 1}),
        {"ES": 2, "CL": -1},
        FIXED_TIMESTAMP,
        flatten=True,
    )
    assert len(receipts) == 2
    sides = {(request.order_ref, request.action) for _, request in fake_client.placed_orders}
    assert len(sides) == 2


async def test_flatten_orders_helper(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    oms, _manager = make_oms(settings, fake_client, clock)
    orders = oms.flatten_orders({"ES": 2}, FIXED_TIMESTAMP)
    assert orders[0].side is SELL
    assert orders[0].tag == "flatten"


async def test_cancel_order_and_cancel_all(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    oms, manager = make_oms(settings, fake_client, clock)
    await manager.connect()
    await oms.submit_order(make_event(order_id="working"))
    fake_client.working_orders = [
        VenueOrderSnapshot(
            order_ref="working",
            status=OrderStatus.SUBMITTED,
            instrument_id="ES",
            side=BUY,
            quantity=1,
        )
    ]
    cancelled = await oms.cancel_all()
    assert cancelled[0].status is OrderStatus.CANCELLED
    assert "working" in fake_client.cancelled
    missing = await oms.cancel_order("nope")
    assert missing.status is OrderStatus.UNKNOWN


async def test_submit_order_never_retransmits_a_working_order(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    oms, manager = make_oms(settings, fake_client, clock)
    await manager.connect()
    fake_client.place_status = OrderStatus.SUBMITTED
    order = make_event()
    first = await oms.submit_order(order)
    assert first.status is OrderStatus.SUBMITTED
    second = await oms.submit_order(order)  # retry of a live order: no re-fire
    assert second.status is OrderStatus.SUBMITTED
    assert len(fake_client.placed_orders) == 1


async def test_cancel_order_updates_tracked_status(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    oms, manager = make_oms(settings, fake_client, clock)
    await manager.connect()
    fake_client.place_status = OrderStatus.SUBMITTED
    await oms.submit_order(make_event(order_id="live"))
    fake_client.working_orders = [
        VenueOrderSnapshot(
            order_ref="live",
            status=OrderStatus.SUBMITTED,
            instrument_id="ES",
            side=BUY,
            quantity=1,
        )
    ]
    receipt = await oms.cancel_order("live")
    assert receipt.status is OrderStatus.CANCELLED
    assert oms.order_status("live") is OrderStatus.CANCELLED


async def test_wait_for_completion_sleeps_between_polls(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    oms, manager = make_oms(settings, fake_client, clock)
    await manager.connect()
    fake_client.place_status = OrderStatus.SUBMITTED
    await oms.submit_order(make_event(order_id="polled"))
    polls = {"count": 0}

    async def sequenced(_order_ref: str) -> Any:
        polls["count"] += 1
        if polls["count"] == 1:
            return None
        return filled_trade("polled", quantity=1)

    fake_client.get_trade = sequenced  # type: ignore[method-assign]
    receipt = await oms.wait_for_completion("polled", timeout_seconds=5.0)
    assert receipt.status is OrderStatus.FILLED
    assert polls["count"] == 2
    assert clock.sleeps  # poll interval went through the injected sleep


async def test_wait_for_completion_polls_until_filled(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    oms, manager = make_oms(settings, fake_client, clock)
    await manager.connect()
    fake_client.place_status = OrderStatus.SUBMITTED
    receipt = await oms.submit_order(make_event(order_id="slow"))
    assert receipt.status is OrderStatus.SUBMITTED
    fake_client.get_trade_result = filled_trade("slow", quantity=1)
    completed = await oms.wait_for_completion("slow", timeout_seconds=1.0)
    assert completed.status is OrderStatus.FILLED
    assert completed.filled_quantity == 1


async def test_wait_for_completion_times_out_deterministically(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    oms, manager = make_oms(settings, fake_client, clock)
    await manager.connect()
    fake_client.place_status = OrderStatus.SUBMITTED
    await oms.submit_order(make_event(order_id="stuck"))
    receipt = await oms.wait_for_completion("stuck", timeout_seconds=0.0)
    assert receipt.status is OrderStatus.SUBMITTED
    assert "timed out" in receipt.message
    with pytest.raises(KeyError):
        await oms.wait_for_completion("unknown")


# ---------------------------------------------------------------------------
# Emergency kill-switch
# ---------------------------------------------------------------------------


async def test_flatten_all_cancels_closes_and_locks(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    oms, manager = make_oms(settings, fake_client, clock)
    await manager.connect()
    fake_client.position_rows = [
        VenuePosition(root_symbol="ES", quantity=2, avg_cost=240_000.0, multiplier=50.0),
        VenuePosition(root_symbol="CL", quantity=-3, avg_cost=72_500.0, multiplier=1000.0),
    ]
    fake_client.working_orders = [
        VenueOrderSnapshot(
            order_ref="open-1",
            status=OrderStatus.SUBMITTED,
            instrument_id="ES",
            side=BUY,
            quantity=1,
        )
    ]
    fake_client.margin_impact = MarginImpact(
        initial_margin_after=1_000.0, maintenance_margin_after=500.0
    )
    receipts = await oms.flatten_all()
    assert receipts[0].status is OrderStatus.CANCELLED
    closes = receipts[1:]
    assert len(closes) == 2
    assert all(receipt.status is OrderStatus.FILLED for receipt in closes)
    assert oms.is_locked
    assert "flatten_all" in oms.gate.lock_reason
    actions = {request.order_ref: request.action for _, request in fake_client.placed_orders}
    assert sorted(actions.values()) == ["BUY", "SELL"]
    oms.reset_kill_switch()
    assert not oms.is_locked


async def test_flatten_all_with_no_positions_still_locks(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    oms, manager = make_oms(settings, fake_client, clock)
    await manager.connect()
    receipts = await oms.flatten_all()
    assert receipts == []
    assert oms.is_locked


async def test_flatten_all_closes_even_when_already_locked(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    gate = make_gate(settings, clock)
    gate.lock("already locked")
    oms, manager = make_oms(settings, fake_client, clock, gate=gate)
    await manager.connect()
    fake_client.position_rows = [
        VenuePosition(root_symbol="ES", quantity=1, avg_cost=240_000.0, multiplier=50.0),
    ]
    fake_client.margin_impact = MarginImpact(
        initial_margin_after=1_000.0, maintenance_margin_after=500.0
    )
    receipts = await oms.flatten_all()
    assert receipts[-1].status is OrderStatus.FILLED
    assert len(fake_client.placed_orders) == 1


async def test_flatten_all_locks_when_position_fetch_fails(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    oms, manager = make_oms(settings, fake_client, clock)
    await manager.connect()

    async def failing_positions() -> Any:
        raise ConnectionError("positions unavailable")

    manager.fetch_positions = failing_positions  # type: ignore[method-assign]
    receipts = await oms.flatten_all()
    assert oms.is_locked  # the kill-switch engages even when the venue is down
    assert receipts[0].status is OrderStatus.REJECTED
    assert "flatten position fetch failed" in receipts[0].message


async def test_flatten_all_reports_cancel_failure(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    oms, manager = make_oms(settings, fake_client, clock)
    await manager.connect()

    async def failing_open_orders() -> Any:
        raise ConnectionError("socket gone")

    manager.open_orders = failing_open_orders  # type: ignore[method-assign]
    receipts = await oms.flatten_all()
    assert receipts[0].status is OrderStatus.REJECTED
    assert "cancel_all failed" in receipts[0].message
    assert oms.is_locked


# ---------------------------------------------------------------------------
# Protocol conformance / API surface
# ---------------------------------------------------------------------------


async def test_oms_satisfies_broker_adapter_protocol(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    oms, manager = make_oms(settings, fake_client, clock)
    await manager.connect()
    assert oms.name == "ibkr-execution"
    assert oms.is_connected
    assert isinstance(oms, BrokerAdapter)
    assert isinstance(oms.gate, PreTradeRiskGateProtocol)
    account = await oms.get_account()
    assert account.net_liquidation == pytest.approx(1_000_000.0)
    assert await oms.get_positions() == {}
    await oms.disconnect()
    assert not manager.is_connected


def test_risk_settings_gate_defaults(settings: Settings) -> None:
    gate = make_gate(settings, FakeClock())
    assert gate.is_locked is False
    assert gate.lock_reason == ""
    assert gate.rejections == ()
    gate.reset_kill_switch()
    assert not gate.is_locked


# ---------------------------------------------------------------------------
# Persisted rate limiter (Phase 7 upgrade 2, OMS side)
# ---------------------------------------------------------------------------


class FakeWallClock:
    """Injectable wall clock in epoch seconds for token-bucket refill tests."""

    def __init__(self, start: float = 1_000.0) -> None:
        self.value = start

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def limited_settings(settings: Settings) -> Settings:
    """Settings with a 1 order/s bucket and no duplicate window (isolates refill)."""
    return Settings(
        environment="backtest",
        instruments_path=settings.instruments_path,
        risk=RiskSettings(order_rate_limit_per_second=1.0, duplicate_order_window_seconds=0.0),
    )


async def test_gate_seeds_persisted_bucket_and_refills_by_wall_clock(
    settings: Settings, clock: FakeClock
) -> None:
    limited = limited_settings(settings)
    wall = FakeWallClock(1_000.0)
    persisted_snapshots: list[RateLimiterState] = []
    seeded = RateLimiterState(tokens=0.0, capacity=1.0, last_refill_epoch_seconds=wall.value)
    gate = make_gate(
        limited,
        clock,
        rate_capacity=1.0,
        wall_clock=wall,
        rate_limiter_state=seeded,
        persist_rate_limiter=persisted_snapshots.append,
    )
    assert gate.persistent_rate_limiter
    blocked = await gate.check(make_event("ES", order_id="a"))
    assert blocked is not None and "rate limit" in blocked.message
    wall.advance(1.0)
    assert await gate.check(make_event("NQ", order_id="b")) is None
    snapshot = gate.rate_limiter_state()
    assert snapshot is not None
    assert snapshot.tokens == pytest.approx(0.0)
    assert snapshot.capacity == pytest.approx(1.0)
    assert snapshot.last_refill_epoch_seconds == pytest.approx(wall.value)
    assert len(persisted_snapshots) == 1
    assert persisted_snapshots[0].tokens == pytest.approx(0.0)


async def test_persisted_token_bucket_restores_throttling_after_oms_restart(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    limited = limited_settings(settings)
    wall = FakeWallClock(1_000.0)
    store = InMemoryRiskStateStore()
    oms, manager = make_oms(limited, fake_client, clock, state_store=store, wall_clock=wall)
    await manager.connect()
    first = await oms.submit_order(make_event("ES", quantity=1, order_id="o-1"))
    assert first.status is OrderStatus.FILLED
    throttled = await oms.submit_order(make_event("NQ", quantity=1, order_id="o-2"))
    assert throttled.status is OrderStatus.REJECTED
    assert "rate limit" in throttled.message
    persisted = store.load()
    assert persisted is not None and persisted.rate_limiter is not None
    assert persisted.rate_limiter.tokens == pytest.approx(0.0)
    # Restart: a brand-new OMS with the same durable bucket must not get a full
    # bucket back.
    restarted, restarted_manager = make_oms(
        limited, fake_client, clock, state_store=store, wall_clock=wall
    )
    await restarted_manager.connect()
    still_throttled = await restarted.submit_order(make_event("CL", quantity=1, order_id="o-3"))
    assert still_throttled.status is OrderStatus.REJECTED
    assert "rate limit" in still_throttled.message
    # After the wall clock earned exactly one token, the order passes again.
    wall.advance(1.0)
    accepted = await restarted.submit_order(make_event("GC", quantity=1, order_id="o-4"))
    assert accepted.status is OrderStatus.FILLED
    after = store.load()
    assert after is not None and after.rate_limiter is not None
    assert after.rate_limiter.tokens == pytest.approx(0.0)
    assert after.rate_limiter.last_refill_epoch_seconds == pytest.approx(wall.value)


async def test_oms_seeds_the_bucket_from_the_persisted_anchor_and_elapsed_time(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    limited = limited_settings(settings)
    wall = FakeWallClock(1_000.0)
    seeded = PersistedRiskState(
        rate_limiter=RateLimiterState(
            tokens=0.0, capacity=1.0, last_refill_epoch_seconds=wall.value - 0.5
        )
    )
    store = InMemoryRiskStateStore(seeded)
    oms, manager = make_oms(limited, fake_client, clock, state_store=store, wall_clock=wall)
    await manager.connect()
    # Half a second elapsed while the process was down: not yet a full token.
    throttled = await oms.submit_order(make_event("ES", order_id="a"))
    assert throttled.status is OrderStatus.REJECTED
    assert "rate limit" in throttled.message
    wall.advance(0.5)
    assert (await oms.submit_order(make_event("NQ", order_id="b"))).status is OrderStatus.FILLED


async def test_kill_switch_lock_persists_across_restart(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    store = InMemoryRiskStateStore()
    oms, manager = make_oms(settings, fake_client, clock, state_store=store)
    await manager.connect()
    oms.lock_kill_switch("daily max loss")
    persisted = store.load()
    assert persisted is not None
    assert persisted.kill_switch_locked is True
    assert persisted.kill_switch_reason == "daily max loss"
    # A restarted process restores the latch and refuses non-emergency orders.
    restarted, restarted_manager = make_oms(settings, fake_client, clock, state_store=store)
    await restarted_manager.connect()
    assert restarted.is_locked
    assert restarted.gate.lock_reason == "daily max loss"
    assert restarted.persisted_state is not None
    rejected = await restarted.submit_order(make_event("ES", order_id="after-restart"))
    assert rejected.status is OrderStatus.REJECTED
    assert "kill switch" in rejected.message
    assert not fake_client.placed_orders


async def test_flatten_all_persists_the_lock_and_preserves_other_fields(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    initial = PersistedRiskState(
        portfolio_high_water_mark=1_200_000.0,
        instrument_high_water_marks={"ES": 4_900.0},
        extra={"risk_overlay": "hwm"},
    )
    store = InMemoryRiskStateStore(initial)
    oms, manager = make_oms(settings, fake_client, clock, state_store=store)
    await manager.connect()
    fake_client.position_rows = [
        VenuePosition(root_symbol="ES", quantity=2, avg_cost=240_000.0, multiplier=50.0)
    ]
    fake_client.margin_impact = MarginImpact(
        initial_margin_after=1_000.0, maintenance_margin_after=500.0
    )
    receipts = await oms.flatten_all()
    assert receipts[-1].status is OrderStatus.FILLED
    locked = store.load()
    assert locked is not None
    assert locked.kill_switch_locked is True
    assert "flatten_all" in locked.kill_switch_reason
    assert locked.portfolio_high_water_mark == pytest.approx(1_200_000.0)
    assert dict(locked.instrument_high_water_marks) == {"ES": 4_900.0}
    assert dict(locked.extra) == {"risk_overlay": "hwm"}
    # Operator reset persists the released latch, still without clobbering HWM.
    oms.reset_kill_switch()
    released = store.load()
    assert released is not None
    assert released.kill_switch_locked is False
    assert released.kill_switch_reason == ""
    assert released.portfolio_high_water_mark == pytest.approx(1_200_000.0)
    assert dict(released.extra) == {"risk_overlay": "hwm"}


async def test_state_store_none_keeps_the_legacy_in_memory_behavior(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    oms, manager = make_oms(settings, fake_client, clock)
    assert oms.persisted_state is None
    assert oms.gate.persistent_rate_limiter is False
    assert oms.gate.rate_limiter_state() is None
    await manager.connect()
    oms.lock_kill_switch("manual")
    assert oms.is_locked
    oms.reset_kill_switch()
    assert not oms.is_locked
    first = await oms.submit_order(make_event("ES", order_id="a"))
    assert first.status is OrderStatus.FILLED


async def test_rate_limiter_snapshot_capacity_follows_configured_settings(
    settings: Settings, clock: FakeClock
) -> None:
    limited = limited_settings(settings)
    wall = FakeWallClock(500.0)
    gate = make_gate(
        limited,
        clock,
        rate_capacity=4.0,
        wall_clock=wall,
        rate_limiter_state=RateLimiterState(
            tokens=1.0, capacity=2.0, last_refill_epoch_seconds=wall.value
        ),
    )
    snapshot = gate.rate_limiter_state()
    assert snapshot is not None
    assert snapshot.capacity == pytest.approx(4.0)
    assert snapshot.tokens == pytest.approx(1.0)
    assert await gate.check(make_event("ES", order_id="a")) is None
    assert gate.rate_limiter_state() is not None
    assert gate.rate_limiter_state().tokens == pytest.approx(0.0)  # type: ignore[union-attr]
