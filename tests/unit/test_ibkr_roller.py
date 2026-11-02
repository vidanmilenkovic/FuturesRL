"""Unit tests for the BAG calendar-spread contract roller (offline, fake venue)."""

from __future__ import annotations

from datetime import date

import pytest
from configs.settings import Settings
from src.core.types import OrderStatus
from src.execution.ibkr_client import (
    IBKRBagUnsupportedError,
    VenueContract,
    VenueTrade,
)
from src.execution.ibkr_roller import (
    ContractRoller,
    RollPlan,
    select_roll_expirations,
)

from tests.unit.test_ibkr_client import FakeClock, FakeIBClient, filled_trade, make_manager

#: Deterministic roll reference date: 2024-01-02 (front ES = 2024-03-15).
AS_OF = date(2024, 1, 2)


@pytest.fixture()
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture()
def fake_client() -> FakeIBClient:
    return FakeIBClient()


def make_roller(
    settings: Settings,
    client: FakeIBClient,
    clock: FakeClock,
    *,
    wait_timeout_seconds: float = 10.0,
) -> ContractRoller:
    manager = make_manager(settings, client, clock)
    return ContractRoller(
        manager,
        instrument_master=settings.instruments,
        clock=clock,
        sleep=clock.sleep,
        wait_timeout_seconds=wait_timeout_seconds,
    )


# ---------------------------------------------------------------------------
# Deterministic contract-month selection
# ---------------------------------------------------------------------------


def test_select_roll_expirations_uses_month_codes(settings: Settings) -> None:
    master = settings.instruments
    assert select_roll_expirations(master.get("ES"), AS_OF) == (
        date(2024, 3, 15),
        date(2024, 6, 21),
    )
    assert select_roll_expirations(master.get("CL"), AS_OF) == (
        date(2024, 1, 19),
        date(2024, 2, 16),
    )
    assert select_roll_expirations(master.get("GC"), AS_OF) == (
        date(2024, 2, 16),
        date(2024, 4, 19),
    )


def test_select_roll_expirations_skips_contracts_inside_roll_window(settings: Settings) -> None:
    es = settings.instruments.get("ES")  # days_before_fnd == 5
    assert select_roll_expirations(es, date(2024, 3, 9)) == (
        date(2024, 3, 15),
        date(2024, 6, 21),
    )
    assert select_roll_expirations(es, date(2024, 3, 12)) == (
        date(2024, 6, 21),
        date(2024, 9, 20),
    )
    zn = settings.instruments.get("ZN")  # days_before_fnd == 10
    assert select_roll_expirations(zn, date(2024, 3, 4)) == (
        date(2024, 3, 15),
        date(2024, 6, 21),
    )
    assert select_roll_expirations(zn, date(2024, 3, 5)) == (
        date(2024, 6, 21),
        date(2024, 9, 20),
    )


# ---------------------------------------------------------------------------
# Plan / BAG construction
# ---------------------------------------------------------------------------


async def test_plan_roll_resolves_front_and_next_contracts(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    roller = make_roller(settings, fake_client, clock)
    await roller.manager.connect()
    plan = await roller.plan_roll("ES", 2, as_of=AS_OF)
    assert plan.root == "ES"
    assert plan.quantity == 2
    assert plan.front.expiration == date(2024, 3, 15)
    assert plan.back.expiration == date(2024, 6, 21)
    assert plan.front.con_id != plan.back.con_id
    assert plan.front.last_trade_date_or_contract_month == "202403"
    assert plan.back.last_trade_date_or_contract_month == "202406"
    assert plan.spread_action.value == "BUY"
    assert plan.front_action.value == "SELL"
    assert plan.back_action.value == "BUY"
    assert plan.absolute_quantity == 2
    assert plan.order_ref.startswith("r-")
    again = await roller.plan_roll("ES", 2, as_of=AS_OF)
    assert again.order_ref == plan.order_ref  # deterministic idempotent roll id


async def test_plan_roll_short_position_mirrors_direction(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    roller = make_roller(settings, fake_client, clock)
    await roller.manager.connect()
    plan = await roller.plan_roll("CL", -3, as_of=AS_OF)
    assert plan.spread_action.value == "SELL"
    assert plan.front_action.value == "BUY"
    assert plan.back_action.value == "SELL"
    assert plan.absolute_quantity == 3


async def test_plan_roll_rejects_zero_quantity_and_unknown_root(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    roller = make_roller(settings, fake_client, clock)
    await roller.manager.connect()
    with pytest.raises(ValueError):
        await roller.plan_roll("ES", 0)
    with pytest.raises(Exception, match="unknown futures root"):
        await roller.plan_roll("NOPE", 1)


def test_build_bag_spec_structure(settings: Settings) -> None:
    roller = make_roller(settings, FakeIBClient(), FakeClock())
    plan = RollPlan(
        root="ES",
        quantity=2,
        front=VenueContract(
            root="ES",
            con_id=101,
            exchange="CME",
            last_trade_date_or_contract_month="202403",
            multiplier=50.0,
            expiration=date(2024, 3, 15),
        ),
        back=VenueContract(
            root="ES",
            con_id=202,
            exchange="CME",
            last_trade_date_or_contract_month="202406",
            multiplier=50.0,
            expiration=date(2024, 6, 21),
        ),
        order_ref="r-abc",
    )
    bag = roller.build_bag_spec(plan)
    assert bag.root == "ES"
    assert bag.exchange == "CME"
    assert bag.currency == "USD"
    assert bag.order.order_ref == "r-abc"
    assert bag.order.action == "BUY"
    assert bag.order.quantity == 2
    assert bag.order.order_type == "MKT"
    assert len(bag.legs) == 2
    assert bag.legs[0].con_id == 101
    assert bag.legs[0].action.value == "SELL"
    assert bag.legs[0].ratio == 1
    assert bag.legs[0].exchange == "CME"
    assert bag.legs[1].con_id == 202
    assert bag.legs[1].action.value == "BUY"
    assert bag.ratio_balanced


def test_bag_spec_validation(settings: Settings) -> None:
    roller = make_roller(settings, FakeIBClient(), FakeClock())
    plan = RollPlan(
        root="ES",
        quantity=1,
        front=VenueContract(
            root="ES",
            con_id=1,
            exchange="CME",
            last_trade_date_or_contract_month="202403",
            multiplier=50.0,
            expiration=date(2024, 3, 15),
        ),
        back=VenueContract(
            root="ES",
            con_id=2,
            exchange="CME",
            last_trade_date_or_contract_month="202406",
            multiplier=50.0,
            expiration=date(2024, 6, 21),
        ),
        order_ref="r-1",
    )
    bag = roller.build_bag_spec(plan)
    with pytest.raises(ValueError):
        type(bag)(
            root="ES",
            exchange="CME",
            legs=(bag.legs[0],),
            order=bag.order,
        )
    with pytest.raises(ValueError):
        type(bag)(
            root="ES",
            exchange="CME",
            legs=(
                bag.legs[0],
                type(bag.legs[1])(
                    con_id=2,
                    action=bag.legs[1].action,
                    ratio=1,
                    exchange="CME",
                ),
            ),
            order=type(bag.order)(
                order_ref="r-1", action="BUY", quantity=1, order_type="STP", stop_price=1.0
            ),
        )


def test_roll_plan_validation() -> None:
    front = VenueContract(
        root="ES",
        con_id=1,
        exchange="CME",
        last_trade_date_or_contract_month="202403",
        multiplier=50.0,
        expiration=date(2024, 3, 15),
    )
    back = VenueContract(
        root="ES",
        con_id=2,
        exchange="CME",
        last_trade_date_or_contract_month="202406",
        multiplier=50.0,
        expiration=date(2024, 6, 21),
    )
    with pytest.raises(ValueError):
        RollPlan(root="ES", quantity=0, front=front, back=back, order_ref="r")
    with pytest.raises(ValueError):
        RollPlan(root="ES", quantity=1, front=front, back=front, order_ref="r")
    no_expiry = VenueContract(
        root="ES",
        con_id=3,
        exchange="CME",
        last_trade_date_or_contract_month="202409",
        multiplier=50.0,
    )
    with pytest.raises(ValueError):
        RollPlan(root="ES", quantity=1, front=front, back=no_expiry, order_ref="r")


# ---------------------------------------------------------------------------
# BAG execution
# ---------------------------------------------------------------------------


async def test_execute_roll_via_native_bag(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    roller = make_roller(settings, fake_client, clock)
    await roller.manager.connect()
    execution = await roller.execute_roll("ES", 2, as_of=AS_OF)
    assert execution.complete
    assert execution.bag_used
    assert execution.fills
    assert execution.receipts[0].status.value == "FILLED"
    assert execution.receipts[0].filled_quantity == 2
    assert execution.filled_quantity == 2
    assert len(fake_client.placed_bags) == 1
    bag = fake_client.placed_bags[0]
    assert bag.root == "ES"
    assert bag.order.action == "BUY"
    assert [leg.action.value for leg in bag.legs] == ["SELL", "BUY"]


async def test_execute_roll_short_position_uses_sell_spread(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    roller = make_roller(settings, fake_client, clock)
    await roller.manager.connect()
    execution = await roller.execute_roll("ES", -2, as_of=AS_OF)
    assert execution.complete
    bag = fake_client.placed_bags[0]
    assert bag.order.action == "SELL"
    assert [leg.action.value for leg in bag.legs] == ["BUY", "SELL"]


async def test_execute_roll_waits_for_non_terminal_bag(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    roller = make_roller(settings, fake_client, clock)
    await roller.manager.connect()
    fake_client.place_status = OrderStatus.SUBMITTED
    fake_client.get_trade_result = filled_trade("ignored", quantity=2)
    execution = await roller.execute_roll("ES", 2, as_of=AS_OF)
    assert fake_client.get_trade_result is not None
    assert execution.complete


async def test_execute_roll_times_out_on_stuck_bag(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    roller = make_roller(settings, fake_client, clock, wait_timeout_seconds=0.0)
    await roller.manager.connect()
    fake_client.place_status = OrderStatus.SUBMITTED
    execution = await roller.execute_roll("ES", 2, as_of=AS_OF)
    assert not execution.complete
    assert "not complete" in execution.message


async def test_execute_roll_reports_partial_bag_fill(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    roller = make_roller(settings, fake_client, clock)
    await roller.manager.connect()
    fake_client.bag_result = VenueTrade(
        order_ref="ignored",
        status=OrderStatus.PARTIALLY_FILLED,
        filled_quantity=1,
        average_price=4800.0,
        message="partial",
    )
    execution = await roller.execute_roll("ES", 2, as_of=AS_OF, wait=False)
    assert not execution.complete
    assert "partial fill 1/2" in execution.message


async def test_cancelled_bag_is_reported_not_retried(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    roller = make_roller(settings, fake_client, clock)
    await roller.manager.connect()
    fake_client.bag_result = VenueTrade(
        order_ref="ignored",
        status=OrderStatus.CANCELLED,
        filled_quantity=0,
        message="cancelled by operator",
    )
    execution = await roller.execute_roll("ES", 2, as_of=AS_OF, wait=False)
    assert not execution.complete
    assert execution.bag_used
    assert "roll cancelled after filling 0/2" in execution.message


async def test_execute_roll_rejects_filled_but_short_bag(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    roller = make_roller(settings, fake_client, clock)
    await roller.manager.connect()
    fake_client.bag_result = filled_trade("ignored", quantity=1, status=OrderStatus.FILLED)
    execution = await roller.execute_roll("ES", 2, as_of=AS_OF)
    assert not execution.complete
    assert "filled=1/2" in execution.message


# ---------------------------------------------------------------------------
# Two-legged fallback
# ---------------------------------------------------------------------------


async def test_bag_unsupported_error_falls_back_to_two_legs(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    roller = make_roller(settings, fake_client, clock)
    await roller.manager.connect()
    fake_client.bag_raises = IBKRBagUnsupportedError("BAG not supported on this gateway")
    execution = await roller.execute_roll("ES", 2, as_of=AS_OF)
    assert execution.complete
    assert not execution.bag_used
    assert len(execution.receipts) == 2
    assert all(receipt.status.value == "FILLED" for receipt in execution.receipts)
    assert len(fake_client.placed_orders) == 2
    (first_contract, first), (second_contract, second) = fake_client.placed_orders
    assert first.action == "SELL"  # close the long front contract first
    assert first.order_ref.endswith("-f")
    assert first_contract.last_trade_date_or_contract_month == "202403"
    assert second.action == "BUY"
    assert second.order_ref.endswith("-b")
    assert second_contract.last_trade_date_or_contract_month == "202406"


async def test_rejected_bag_with_unsupported_message_falls_back(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    roller = make_roller(settings, fake_client, clock)
    await roller.manager.connect()
    fake_client.bag_result = VenueTrade(
        order_ref="ignored",
        status=OrderStatus.REJECTED,
        message="Combo/BAG calendar spread orders are unsupported for this account",
    )
    execution = await roller.execute_roll("ES", -1, as_of=AS_OF)
    assert execution.complete
    assert not execution.bag_used
    assert [request.action for _, request in fake_client.placed_orders] == ["BUY", "SELL"]


async def test_leg_fallback_does_not_open_back_leg_on_partial_front_fill(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    roller = make_roller(settings, fake_client, clock)
    await roller.manager.connect()
    plan = await roller.plan_roll("ES", 2, as_of=AS_OF)
    fake_client.bag_raises = IBKRBagUnsupportedError("unsupported")
    fake_client.trade_overrides[f"{plan.order_ref}-f"] = filled_trade(
        f"{plan.order_ref}-f",
        quantity=1,
        status=OrderStatus.PARTIALLY_FILLED,
    )
    execution = await roller.execute_roll("ES", 2, as_of=AS_OF)
    assert not execution.complete
    assert len(fake_client.placed_orders) == 1  # back leg must not be sent
    assert "back leg not sent" in execution.message


async def test_leg_fallback_reports_partial_back_leg(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    roller = make_roller(settings, fake_client, clock)
    await roller.manager.connect()
    plan = await roller.plan_roll("ES", 2, as_of=AS_OF)
    fake_client.bag_raises = IBKRBagUnsupportedError("unsupported")
    fake_client.trade_overrides[f"{plan.order_ref}-b"] = filled_trade(
        f"{plan.order_ref}-b",
        quantity=1,
        status=OrderStatus.PARTIALLY_FILLED,
    )
    execution = await roller.execute_roll("ES", 2, as_of=AS_OF)
    assert not execution.complete
    assert len(fake_client.placed_orders) == 2
    assert "partial fill 1/2" in execution.message


async def test_leg_fallback_handles_transport_failure(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    roller = make_roller(settings, fake_client, clock)
    await roller.manager.connect()
    fake_client.bag_raises = IBKRBagUnsupportedError("unsupported")
    fake_client.place_raises = ConnectionError("socket closed")
    execution = await roller.execute_roll("ES", 2, as_of=AS_OF)
    assert not execution.complete
    assert "front leg did not fully fill" in execution.message
    assert execution.receipts[0].status.value == "REJECTED"


async def test_unexpected_bag_failure_is_reported(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    roller = make_roller(settings, fake_client, clock)
    await roller.manager.connect()
    fake_client.bag_raises = RuntimeError("gateway exploded")
    execution = await roller.execute_roll("ES", 2, as_of=AS_OF)
    assert not execution.complete
    assert execution.bag_used
    assert execution.receipts[0].status.value == "REJECTED"
    assert "BAG submission failed" in execution.message


async def test_execute_roll_rejects_zero_quantity(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    roller = make_roller(settings, fake_client, clock)
    await roller.manager.connect()
    with pytest.raises(ValueError):
        await roller.execute_roll("ES", 0)


def test_roller_exposes_instrument_master(settings: Settings) -> None:
    roller = make_roller(settings, FakeIBClient(), FakeClock())
    assert roller.instruments.has("ES")
