"""Unit tests for the async IBKR connection/state manager.

Every test is offline and deterministic: a :class:`FakeIBClient` implements the
pure :class:`IBClientLike` transport contract, the clock and sleep are injected,
and no test ever touches TWS, IB Gateway or the network.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import UTC, date, datetime, time, timedelta
from types import SimpleNamespace
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from configs.settings import Settings
from src.core.types import OrderEvent, OrderSide, OrderStatus
from src.execution.ibkr_client import (
    IBKRConnectionError,
    IBKRConnectionManager,
    IBKRDataError,
    IBOrderRequest,
    MarginImpact,
    VenueAccountValue,
    VenueContract,
    VenueFill,
    VenueOrderSnapshot,
    VenuePosition,
    VenueTrade,
    active_contract_expiration,
    contract_month_string,
    map_order_status,
    margin_impact_from_order_state,
    normalize_expiration,
    parse_contract_month,
    third_friday,
    upcoming_contract_expirations,
)

EASTERN: ZoneInfo = ZoneInfo("America/New_York")
FIXED: datetime = datetime(2024, 1, 2, 21, 0, tzinfo=UTC)  # 16:00 New York


# ---------------------------------------------------------------------------
# Deterministic fakes
# ---------------------------------------------------------------------------


class FakeClock:
    """Injectable wall clock whose async ``sleep`` advances time deterministically."""

    def __init__(self, start: datetime = FIXED) -> None:
        self.now = start
        self.sleeps: list[float] = []

    def __call__(self) -> datetime:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now = self.now + timedelta(seconds=seconds)

    def advance(self, seconds: float) -> None:
        self.now = self.now + timedelta(seconds=seconds)


def make_account_rows(
    *,
    net_liquidation: str = "1000000.00",
    cash: str = "800000.00",
    initial_margin: str = "120000.00",
    maintenance_margin: str = "110000.00",
    excess_liquidity: str = "880000.00",
    currency: str = "USD",
) -> list[VenueAccountValue]:
    return [
        VenueAccountValue("NetLiquidation", net_liquidation, currency),
        VenueAccountValue("TotalCashValue", cash, currency),
        VenueAccountValue("BuyingPower", "2000000.00", currency),
        VenueAccountValue("InitMarginReq", initial_margin, currency),
        VenueAccountValue("MaintMarginReq", maintenance_margin, currency),
        VenueAccountValue("ExcessLiquidity", excess_liquidity, currency),
    ]


def filled_trade(
    order_ref: str,
    *,
    quantity: int = 1,
    price: float = 4800.0,
    side: OrderSide = OrderSide.BUY,
    commission: float = 1.0,
    status: OrderStatus = OrderStatus.FILLED,
    message: str = "",
) -> VenueTrade:
    return VenueTrade(
        order_ref=order_ref,
        status=status,
        filled_quantity=quantity,
        average_price=price,
        commission=commission,
        message=message,
        fills=(
            VenueFill(
                order_ref=order_ref,
                side=side,
                quantity=quantity,
                price=price,
                timestamp=FIXED,
                commission=commission,
            ),
        ),
    )


class FakeIBClient:
    """In-memory implementation of :class:`IBClientLike` used across execution tests."""

    def __init__(self) -> None:
        self.connected = False
        self.connect_calls = 0
        self.disconnect_calls = 0
        self.fail_connects = 0
        self.account_rows: list[VenueAccountValue] = make_account_rows()
        self.position_rows: list[VenuePosition] = []
        self.margin_impact = MarginImpact(
            initial_margin_before=120_000.0,
            maintenance_margin_before=110_000.0,
            initial_margin_change=15_000.0,
            maintenance_margin_change=13_600.0,
            initial_margin_after=135_000.0,
            maintenance_margin_after=123_600.0,
        )
        self.qualify_requests: list[Any] = []
        self.what_if_calls: list[tuple[VenueContract, IBOrderRequest]] = []
        self.placed_orders: list[tuple[VenueContract, IBOrderRequest]] = []
        self.brackets: list[tuple[VenueContract, IBOrderRequest, IBOrderRequest]] = []
        self.placed_bags: list[Any] = []
        self.cancelled: list[str] = []
        self.working_orders: list[VenueOrderSnapshot] = []
        self.trade_overrides: dict[str, VenueTrade] = {}
        self.place_status: OrderStatus = OrderStatus.FILLED
        self.get_trade_result: VenueTrade | None = None
        self.bag_result: VenueTrade | None = None
        self.bag_raises: BaseException | None = None
        self.place_raises: BaseException | None = None
        self.next_con_id = 100
        self._con_ids: dict[tuple[str, str], int] = {}
        self.fail_contracts: set[tuple[str, str]] = set()

    # -- protocol -----------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self.connected

    async def connect(self, *, host: str, port: int, client_id: int, timeout: float) -> None:
        self.connect_calls += 1
        if self.fail_connects > 0:
            self.fail_connects -= 1
            raise ConnectionError("simulated connect failure")
        self.connected = True

    async def disconnect(self) -> None:
        self.disconnect_calls += 1
        self.connected = False

    async def account_values(self) -> Sequence[VenueAccountValue]:
        return list(self.account_rows)

    async def positions(self) -> Sequence[VenuePosition]:
        return list(self.position_rows)

    async def qualify_contract(self, request: Any) -> VenueContract:
        self.qualify_requests.append(request)
        key = (request.root, request.last_trade_date_or_contract_month)
        if key in self.fail_contracts:
            raise IBKRDataError(
                f"IBKR could not qualify {request.root} {request.last_trade_date_or_contract_month}"
            )
        con_id = self._con_ids.get(key)
        if con_id is None:
            con_id = self.next_con_id
            self.next_con_id += 1
            self._con_ids[key] = con_id
        return VenueContract(
            root=request.root,
            con_id=con_id,
            exchange=request.exchange,
            last_trade_date_or_contract_month=request.last_trade_date_or_contract_month,
            multiplier=request.multiplier,
            expiration=request.expiration,
            local_symbol=f"{request.root}{request.last_trade_date_or_contract_month}",
            currency=request.currency,
        )

    async def what_if_order(self, contract: VenueContract, order: IBOrderRequest) -> MarginImpact:
        self.what_if_calls.append((contract, order))
        return self.margin_impact

    def _trade_for(self, order: IBOrderRequest, contract: VenueContract) -> VenueTrade:
        override = self.trade_overrides.get(order.order_ref)
        if override is not None:
            return override
        price = order.limit_price if order.limit_price is not None else 4800.0
        side = OrderSide(order.action)
        return filled_trade(
            order.order_ref,
            quantity=order.quantity,
            price=price,
            side=side,
            status=self.place_status,
        )

    async def place_order(self, contract: VenueContract, order: IBOrderRequest) -> VenueTrade:
        self.placed_orders.append((contract, order))
        if self.place_raises is not None:
            raise self.place_raises
        return self._trade_for(order, contract)

    async def place_bracket(
        self,
        contract: VenueContract,
        parent: IBOrderRequest,
        child: IBOrderRequest,
    ) -> tuple[VenueTrade, VenueTrade]:
        self.brackets.append((contract, parent, child))
        return (self._trade_for(parent, contract), self._trade_for(child, contract))

    async def place_bag_order(self, bag: Any) -> VenueTrade:
        self.placed_bags.append(bag)
        if self.bag_raises is not None:
            raise self.bag_raises
        if self.bag_result is not None:
            return self.bag_result
        request = bag.order
        return filled_trade(
            request.order_ref,
            quantity=request.quantity,
            side=OrderSide(request.action),
            status=self.place_status,
        )

    async def open_orders(self) -> Sequence[VenueOrderSnapshot]:
        return list(self.working_orders)

    async def cancel_order(self, order_ref: str) -> bool:
        self.cancelled.append(order_ref)
        return any(snapshot.order_ref == order_ref for snapshot in self.working_orders)

    async def get_trade(self, order_ref: str) -> VenueTrade | None:
        if self.get_trade_result is not None:
            return self.get_trade_result
        return self.trade_overrides.get(order_ref)


def make_manager(
    settings: Settings,
    client: FakeIBClient,
    clock: FakeClock,
    *,
    heartbeat_enabled: bool = False,
    **kwargs: Any,
) -> IBKRConnectionManager:
    return IBKRConnectionManager(
        settings,
        client_factory=lambda: client,
        clock=clock,
        sleep=clock.sleep,
        random_source=lambda: 0.5,
        heartbeat_enabled=heartbeat_enabled,
        **kwargs,
    )


@pytest.fixture()
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture()
def fake_client() -> FakeIBClient:
    return FakeIBClient()


# ---------------------------------------------------------------------------
# Lifecycle / reconnection
# ---------------------------------------------------------------------------


async def test_connect_and_disconnect_lifecycle(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    manager = make_manager(settings, fake_client, clock)
    assert not manager.is_connected
    await manager.connect()
    assert manager.is_connected
    assert fake_client.connect_calls == 1
    assert manager.connection_attempts == 1
    await manager.connect()  # idempotent while connected
    assert fake_client.connect_calls == 1
    await manager.disconnect()
    assert not manager.is_connected
    assert fake_client.disconnect_calls == 1
    with pytest.raises(IBKRConnectionError):
        _ = manager.client


async def test_async_context_manager(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    manager = make_manager(settings, fake_client, clock)
    async with manager as connected:
        assert connected is manager
        assert connected.is_connected
    assert not manager.is_connected


async def test_connect_retries_with_exponential_backoff(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    fake_client.fail_connects = 2
    manager = make_manager(settings, fake_client, clock)
    await manager.connect()
    assert manager.is_connected
    assert fake_client.connect_calls == 3
    # jitter factor is exactly 1.0 with random_source == 0.5
    assert clock.sleeps == [
        settings.ibkr_reconnect_initial_delay_seconds,
        settings.ibkr_reconnect_initial_delay_seconds * 2,
    ]
    assert manager.connection_attempts == 3
    assert manager.last_error is None


async def test_backoff_is_capped_and_jittered(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    capped = Settings(
        environment="backtest",
        instruments_path=settings.instruments_path,
        ibkr_reconnect_initial_delay_seconds=1.0,
        ibkr_reconnect_max_delay_seconds=3.0,
        ibkr_reconnect_max_attempts=5,
    )
    fake_client.fail_connects = 3
    manager = IBKRConnectionManager(
        capped,
        client_factory=lambda: fake_client,
        clock=clock,
        sleep=clock.sleep,
        random_source=lambda: 0.0,
        heartbeat_enabled=False,
    )
    await manager.connect()
    assert fake_client.connect_calls == 4
    # base delays 1, 2, 3 (capped) each scaled by (1 - jitter) == 0.9
    assert clock.sleeps == pytest.approx([0.9, 1.8, 2.7])
    assert manager.last_error is None


async def test_connect_raises_after_max_attempts(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    fake_client.fail_connects = 100
    manager = make_manager(settings, fake_client, clock)
    with pytest.raises(IBKRConnectionError, match="after 10 attempts"):
        await manager.connect()
    assert fake_client.connect_calls == settings.ibkr_reconnect_max_attempts
    assert len(clock.sleeps) == settings.ibkr_reconnect_max_attempts - 1
    assert manager.last_error is not None
    assert "ConnectionError" in manager.last_error


async def test_heartbeat_tick_reconnects_lost_session(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    manager = make_manager(settings, fake_client, clock)
    await manager.connect()
    fake_client.connected = False
    assert await manager.heartbeat_tick()
    assert manager.is_connected
    assert manager.reconnect_count == 1
    assert fake_client.connect_calls == 2


async def test_heartbeat_tick_noop_when_healthy(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    manager = make_manager(settings, fake_client, clock)
    await manager.connect()
    assert await manager.heartbeat_tick()
    assert manager.reconnect_count == 0
    assert manager.last_heartbeat_at == clock.now


async def test_heartbeat_loop_detects_lost_connection(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    async def yielding_sleep(seconds: float) -> None:
        clock.sleeps.append(seconds)
        await asyncio.sleep(0)
        clock.advance(seconds)
        fake_client.connected = False

    manager = IBKRConnectionManager(
        settings,
        client_factory=lambda: fake_client,
        clock=clock,
        sleep=yielding_sleep,
        random_source=lambda: 0.5,
        heartbeat_enabled=True,
    )
    await manager.connect()
    for _ in range(6):
        await asyncio.sleep(0)
    assert manager.reconnect_count >= 1
    assert fake_client.connect_calls >= 2
    await manager.disconnect()
    assert not manager.is_connected


# ---------------------------------------------------------------------------
# Nightly server-reset window
# ---------------------------------------------------------------------------


def _et(y: int, mo: int, d: int, h: int, mi: int = 0) -> datetime:
    return datetime(y, mo, d, h, mi, tzinfo=EASTERN)


async def test_nightly_reset_window_detection(
    settings: Settings, fake_client: FakeIBClient
) -> None:
    clock = FakeClock(_et(2024, 1, 3, 23, 50))
    manager = make_manager(settings, fake_client, clock)
    assert manager.nightly_reset_start.isoformat() == "23:45:00"
    assert manager.nightly_reset_end.isoformat() == "00:45:00"
    assert manager.in_nightly_reset_window()
    assert manager.seconds_until_reset_window_end() == pytest.approx(55 * 60)
    clock.now = _et(2024, 1, 4, 0, 30)
    assert manager.in_nightly_reset_window()
    assert manager.seconds_until_reset_window_end() == pytest.approx(15 * 60)
    clock.now = _et(2024, 1, 3, 23, 0)
    assert not manager.in_nightly_reset_window()
    assert manager.seconds_until_reset_window_end() == 0.0
    clock.now = _et(2024, 1, 4, 0, 50)
    assert not manager.in_nightly_reset_window()
    clock.now = _et(2024, 1, 4, 12, 0)
    assert not manager.in_nightly_reset_window()


async def test_nightly_reset_window_handles_dst(
    settings: Settings, fake_client: FakeIBClient
) -> None:
    # 2024-07-01 is EDT (UTC-4): 03:50 UTC == 23:50 New York.
    clock = FakeClock(datetime(2024, 7, 2, 3, 50, tzinfo=UTC))
    manager = make_manager(settings, fake_client, clock)
    assert manager.in_nightly_reset_window()
    assert manager.seconds_until_reset_window_end() == pytest.approx(55 * 60)


async def test_nightly_reset_window_custom_hours(
    settings: Settings, fake_client: FakeIBClient
) -> None:
    from datetime import time

    clock = FakeClock(_et(2024, 1, 3, 22, 10))
    manager = make_manager(
        settings,
        fake_client,
        clock,
        nightly_reset_start=time(22, 0),
        nightly_reset_end=time(22, 30),
    )
    assert manager.in_nightly_reset_window()
    assert manager.seconds_until_reset_window_end() == pytest.approx(20 * 60)
    clock.now = _et(2024, 1, 3, 22, 30)
    assert not manager.in_nightly_reset_window()


async def test_client_id_rotates_across_reconnects(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    """A wedged TWS client id must not prevent reconnection with a fresh id."""
    manager = make_manager(settings, fake_client, clock)
    await manager.connect()
    first = manager.active_client_id
    assert first == settings.ibkr_client_id
    manager.invalidate("simulated wedge")
    assert await manager.ensure_connected()
    assert manager.active_client_id == settings.ibkr_client_id + 1


async def test_connect_waits_for_nightly_reset_window(
    settings: Settings, fake_client: FakeIBClient
) -> None:
    clock = FakeClock(_et(2024, 1, 3, 23, 50))
    manager = make_manager(settings, fake_client, clock)
    await manager.connect()
    assert clock.sleeps[0] == pytest.approx(55 * 60)
    assert fake_client.connect_calls == 1
    assert manager.is_connected
    assert clock.now == _et(2024, 1, 4, 0, 45)


# ---------------------------------------------------------------------------
# Account / positions
# ---------------------------------------------------------------------------


async def test_fetch_account_maps_tags(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    fake_client.account_rows = [
        *make_account_rows(),
        VenueAccountValue("NetLiquidation", "999999999.00", "EUR"),
        VenueAccountValue("NetLiquidation", "", "USD"),
    ]
    manager = make_manager(settings, fake_client, clock)
    await manager.connect()
    snapshot = await manager.fetch_account()
    assert snapshot.net_liquidation == 1_000_000.0
    assert snapshot.cash_balance == 800_000.0
    assert snapshot.buying_power == 2_000_000.0
    assert snapshot.initial_margin_used == 120_000.0
    assert snapshot.maintenance_margin_used == 110_000.0
    assert snapshot.excess_liquidity == 880_000.0
    assert snapshot.margin_utilization == pytest.approx(0.12)
    assert snapshot.timestamp == clock.now


async def test_fetch_account_accepts_account_base_currency_rows(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    """EUR-base IBKR accounts must not be rejected as if only USD rows counted."""
    eur_settings = settings.model_copy(update={"account_currency": "EUR"})
    fake_client.account_rows = [
        VenueAccountValue("NetLiquidation", "1003547.18", "EUR"),
        VenueAccountValue("TotalCashValue", "1001970.93", "EUR"),
        VenueAccountValue("BuyingPower", "4007883.72", "EUR"),
        VenueAccountValue("NetLiquidation", "1.00", "GBP"),
    ]
    manager = make_manager(eur_settings, fake_client, clock)
    await manager.connect()
    snapshot = await manager.fetch_account()
    assert snapshot.net_liquidation == pytest.approx(1_003_547.18)
    assert snapshot.cash_balance == pytest.approx(1_001_970.93)
    assert snapshot.buying_power == pytest.approx(4_007_883.72)


async def test_fetch_account_uses_ledger_currency_when_config_mismatches(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    """EUR-base IBKR accounts must work even if account_currency says USD."""
    assert settings.account_currency.upper() == "USD"
    fake_client.account_rows = [
        VenueAccountValue("NetLiquidation", "1003691.28", "EUR"),
        VenueAccountValue("TotalCashValue", "1001957.62", "EUR"),
        VenueAccountValue("BuyingPower", "4007883.72", "EUR"),
    ]
    manager = make_manager(settings, fake_client, clock)
    await manager.connect()
    snapshot = await manager.fetch_account()
    assert snapshot.net_liquidation == pytest.approx(1_003_691.28)
    assert snapshot.cash_balance == pytest.approx(1_001_957.62)
    assert snapshot.buying_power == pytest.approx(4_007_883.72)


async def test_fetch_account_requires_net_liquidation(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    fake_client.account_rows = [VenueAccountValue("TotalCashValue", "1000.00", "USD")]
    manager = make_manager(settings, fake_client, clock)
    await manager.connect()
    with pytest.raises(IBKRDataError, match="NetLiquidation"):
        await manager.fetch_account()


async def test_fetch_positions_maps_multiplier_cost_basis_and_marks(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    fake_client.position_rows = [
        VenuePosition(root_symbol="ES", quantity=2, avg_cost=240_000.0, multiplier=50.0),
        VenuePosition(root_symbol="cl", quantity=-1, avg_cost=72_500.0, multiplier=1000.0),
        VenuePosition(root_symbol="GC", quantity=0, avg_cost=0.0),
    ]
    manager = make_manager(settings, fake_client, clock, price_provider=lambda root: 4810.0)
    await manager.connect()
    positions = await manager.fetch_positions()
    assert set(positions) == {"ES", "CL"}
    es = positions["ES"]
    assert es.quantity == 2
    assert es.average_price == pytest.approx(4800.0)
    assert es.market_price == pytest.approx(4810.0)
    assert es.multiplier == pytest.approx(50.0)
    assert es.initial_margin == pytest.approx(15_000.0)
    assert es.expiration == date(2024, 3, 15)  # front ES month on 2024-01-02
    assert positions["CL"].quantity == -1
    assert positions["CL"].average_price == pytest.approx(72.5)


async def test_fetch_positions_prefers_explicit_prices_and_falls_back_to_cost(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    fake_client.position_rows = [
        VenuePosition(root_symbol="ES", quantity=1, avg_cost=240_000.0, multiplier=50.0),
    ]
    manager = make_manager(settings, fake_client, clock)
    await manager.connect()
    explicit = await manager.fetch_positions(prices={"ES": 4900.0})
    assert explicit["ES"].market_price == pytest.approx(4900.0)
    fallback = await manager.fetch_positions()
    assert fallback["ES"].market_price == pytest.approx(4800.0)
    assert fallback["ES"].average_price == pytest.approx(4800.0)


async def test_fetch_positions_handles_unknown_root_and_rejects_zero_price(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    fake_client.position_rows = [
        VenuePosition(root_symbol="ZZZ", quantity=1, avg_cost=500.0, multiplier=2.0),
    ]
    manager = make_manager(settings, fake_client, clock)
    await manager.connect()
    positions = await manager.fetch_positions()
    assert positions["ZZZ"].multiplier == pytest.approx(2.0)
    assert positions["ZZZ"].average_price == pytest.approx(250.0)
    assert positions["ZZZ"].expiration is None
    fake_client.position_rows = [VenuePosition(root_symbol="ZZZ", quantity=1, avg_cost=0.0)]
    with pytest.raises(IBKRDataError, match="no price"):
        await manager.fetch_positions()
    fake_client.position_rows = [VenuePosition(root_symbol="", quantity=1, avg_cost=1.0)]
    with pytest.raises(IBKRDataError, match="without a contract symbol"):
        await manager.fetch_positions()


# ---------------------------------------------------------------------------
# Contract resolution
# ---------------------------------------------------------------------------


async def test_resolve_contract_uses_instrument_master_and_caches(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    manager = make_manager(settings, fake_client, clock)
    await manager.connect()
    contract = await manager.resolve_contract("es")
    assert contract.root == "ES"
    assert contract.exchange == "CME"
    assert contract.last_trade_date_or_contract_month == "202403"
    assert contract.con_id == 100
    assert fake_client.qualify_requests[0].exchange == "CME"
    assert fake_client.qualify_requests[0].multiplier == pytest.approx(50.0)
    again = await manager.resolve_contract("ES", date(2024, 3, 15))
    assert again.con_id == contract.con_id
    assert len(fake_client.qualify_requests) == 1  # cached by (root, contract month)


@pytest.mark.parametrize(
    ("expiration", "expected_month"),
    [
        (date(2024, 6, 21), "202406"),
        ("202412", "202412"),
        ("2024-12-20", "202412"),
        ("20240315", "202403"),
    ],
)
async def test_resolve_contract_accepts_explicit_expirations(
    settings: Settings,
    fake_client: FakeIBClient,
    clock: FakeClock,
    expiration: object,
    expected_month: str,
) -> None:
    manager = make_manager(settings, fake_client, clock)
    await manager.connect()
    contract = await manager.resolve_contract("ES", expiration)  # type: ignore[arg-type]
    assert contract.last_trade_date_or_contract_month == expected_month


async def test_resolve_contract_unknown_root_raises(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    manager = make_manager(settings, fake_client, clock)
    await manager.connect()
    with pytest.raises(IBKRDataError, match="unknown futures root"):
        await manager.resolve_contract("NOPE")


async def test_resolve_contract_falls_forward_and_uses_fx_symbol(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    manager = make_manager(settings, fake_client, clock)
    await manager.connect()
    fake_client.fail_contracts.add(("CL", "202401"))
    contract = await manager.resolve_contract("CL")
    assert contract.last_trade_date_or_contract_month == "202402"
    assert [
        request.last_trade_date_or_contract_month for request in fake_client.qualify_requests
    ] == [
        "202401",
        "202402",
    ]

    fx = await manager.resolve_contract("6E")
    assert fx.last_trade_date_or_contract_month == "202403"
    assert fake_client.qualify_requests[-1].symbol == "EUR"


# ---------------------------------------------------------------------------
# whatIfOrder margin simulation
# ---------------------------------------------------------------------------


async def test_what_if_order_returns_margin_impact(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    manager = make_manager(settings, fake_client, clock)
    await manager.connect()
    order = OrderEvent.market(
        order_id="o-1",
        timestamp=clock.now,
        instrument_id="ES",
        side=OrderSide.BUY,
        quantity=2,
    )
    impact = await manager.what_if_order(order)
    assert impact is fake_client.margin_impact
    assert impact.projected_initial_margin == pytest.approx(135_000.0)
    assert impact.projected_maintenance_margin == pytest.approx(123_600.0)
    contract, request = fake_client.what_if_calls[0]
    assert contract.last_trade_date_or_contract_month == "202403"
    assert request.order_type == "MKT"
    assert request.action == "BUY"
    assert request.quantity == 2


async def test_what_if_order_uses_injected_provider(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    expected = MarginImpact(initial_margin_after=1.0, maintenance_margin_after=0.5)
    manager = make_manager(
        settings,
        fake_client,
        clock,
        what_if_provider=lambda _order: _async_value(expected),
    )
    await manager.connect()
    order = OrderEvent.market(
        order_id="o-1",
        timestamp=clock.now,
        instrument_id="ES",
        side=OrderSide.SELL,
        quantity=1,
    )
    assert await manager.what_if_order(order) is expected
    assert fake_client.what_if_calls == []


async def _async_value(value: Any) -> Any:
    return value


async def test_what_if_requires_connection(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    manager = make_manager(settings, fake_client, clock)
    order = OrderEvent.market(
        order_id="o-1",
        timestamp=clock.now,
        instrument_id="ES",
        side=OrderSide.BUY,
        quantity=1,
    )
    with pytest.raises(IBKRConnectionError):
        await manager.what_if_order(order)


# ---------------------------------------------------------------------------
# Order/transport primitives exposed by the manager
# ---------------------------------------------------------------------------


async def test_manager_order_primitives_delegate_to_transport(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    manager = make_manager(settings, fake_client, clock)
    await manager.connect()
    contract = await manager.resolve_contract("ES")
    request = IBOrderRequest(order_ref="r-1", action="BUY", quantity=1, order_type="MKT")
    trade = await manager.place_order(contract, request)
    assert trade.order_ref == "r-1"
    assert fake_client.placed_orders[-1] == (contract, request)
    parent = IBOrderRequest(order_ref="p-1", action="BUY", quantity=1, order_type="MKT")
    child = IBOrderRequest(
        order_ref="p-1-sl", action="SELL", quantity=1, order_type="STP", stop_price=4700.0
    )
    parent_trade, child_trade = await manager.place_bracket(contract, parent, child)
    assert parent_trade.status is OrderStatus.FILLED
    assert child_trade.order_ref == "p-1-sl"
    fake_client.working_orders = [
        VenueOrderSnapshot(
            order_ref="p-1",
            status=OrderStatus.SUBMITTED,
            instrument_id="ES",
            side=OrderSide.BUY,
            quantity=1,
        )
    ]
    assert len(await manager.open_orders()) == 1
    assert await manager.cancel_order("p-1") is True
    assert fake_client.cancelled == ["p-1"]
    fake_client.get_trade_result = filled_trade("r-1")
    assert (await manager.get_trade("r-1")) is fake_client.get_trade_result
    account = await manager.get_account()
    assert account.net_liquidation == pytest.approx(1_000_000.0)
    assert (await manager.get_positions()) == {}


async def test_set_price_provider(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    manager = make_manager(settings, fake_client, clock)
    await manager.connect()
    fake_client.position_rows = [
        VenuePosition(root_symbol="ES", quantity=1, avg_cost=240_000.0, multiplier=50.0)
    ]
    manager.set_price_provider(lambda _root: 5000.0)
    positions = await manager.fetch_positions()
    assert positions["ES"].market_price == pytest.approx(5000.0)


async def test_manager_properties(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    manager = make_manager(settings, fake_client, clock)
    assert manager.name == "ibkr"
    assert manager.instruments.has("ES")
    assert manager.price_provider("ES") is None
    manager.set_what_if_provider(None)
    await manager.connect()
    assert manager.connection_attempts == 1


# ---------------------------------------------------------------------------
# Pure calendar / parsing helpers
# ---------------------------------------------------------------------------


def test_third_friday_and_contract_months(settings: Settings) -> None:
    master = settings.instruments
    assert third_friday(2024, 3) == date(2024, 3, 15)
    assert third_friday(2024, 6) == date(2024, 6, 21)
    assert third_friday(2024, 1) == date(2024, 1, 19)
    assert contract_month_string(date(2024, 3, 15)) == "202403"
    es = master.get("ES")
    assert upcoming_contract_expirations(es, date(2024, 1, 2), count=2) == (
        date(2024, 3, 15),
        date(2024, 6, 21),
    )
    assert active_contract_expiration(es, date(2024, 3, 12)) == date(2024, 6, 21)
    assert active_contract_expiration(es, date(2024, 3, 9)) == date(2024, 3, 15)
    gc = master.get("GC")
    assert upcoming_contract_expirations(gc, date(2024, 1, 2), count=2) == (
        date(2024, 2, 16),
        date(2024, 4, 19),
    )


def test_normalize_expiration_variants() -> None:
    assert normalize_expiration(None) == (None, None)
    assert normalize_expiration("") == (None, None)
    assert normalize_expiration(date(2024, 3, 15)) == (date(2024, 3, 15), "202403")
    assert normalize_expiration("202403") == (date(2024, 3, 1), "202403")
    assert normalize_expiration("20240315") == (date(2024, 3, 15), "202403")
    assert normalize_expiration("2024-03-15") == (date(2024, 3, 15), "202403")
    with pytest.raises(IBKRDataError):
        normalize_expiration("not-a-date")
    with pytest.raises(TypeError):
        normalize_expiration(datetime(2024, 3, 15, tzinfo=UTC))  # type: ignore[arg-type]


def test_parse_contract_month_and_status_mapping() -> None:
    assert parse_contract_month("20240315") == date(2024, 3, 15)
    assert parse_contract_month("202403") == date(2024, 3, 1)
    assert parse_contract_month("") is None
    assert parse_contract_month(None) is None
    assert parse_contract_month(object()) is None
    assert parse_contract_month("garbage") is None
    cases: dict[str, OrderStatus] = {
        "PendingSubmit": OrderStatus.PENDING,
        "ApiPending": OrderStatus.PENDING,
        "PreSubmitted": OrderStatus.SUBMITTED,
        "Submitted": OrderStatus.SUBMITTED,
        "PendingCancel": OrderStatus.SUBMITTED,
        "Filled": OrderStatus.FILLED,
        "Cancelled": OrderStatus.CANCELLED,
        "ApiCancelled": OrderStatus.CANCELLED,
        "Inactive": OrderStatus.REJECTED,
        "SomethingNew": OrderStatus.UNKNOWN,
    }
    for raw, expected in cases.items():
        assert map_order_status(raw) is expected
    assert map_order_status(None) is OrderStatus.UNKNOWN
    assert map_order_status(OrderStatus.FILLED) is OrderStatus.FILLED


def test_margin_impact_from_order_state_parses_sentinels() -> None:
    state = SimpleNamespace(
        initMarginBefore="120000.0",
        maintMarginBefore="110000.0",
        initMarginChange="15000.0",
        maintMarginChange="13600.0",
        initMarginAfter="1.7976931348623157E308",
        maintMarginAfter="",
        warningText="order exceeds margin cushion",
    )
    impact = margin_impact_from_order_state(state)
    assert impact.initial_margin_before == pytest.approx(120_000.0)
    assert impact.initial_margin_after is None
    assert impact.maintenance_margin_after is None
    assert impact.projected_initial_margin == pytest.approx(135_000.0)
    assert impact.projected_maintenance_margin == pytest.approx(123_600.0)
    assert impact.is_complete
    assert impact.warning == "order exceeds margin cushion"
    assert impact.utilization(1_000_000.0) == pytest.approx((0.135, 0.1236))
    blank = margin_impact_from_order_state(SimpleNamespace())
    assert not blank.is_complete
    assert blank.projected_initial_margin is None
    assert blank.projected_maintenance_margin is None
    assert blank.utilization(0.0) == (None, None)
    assert blank.warning == ""


def test_margin_impact_after_fields_take_precedence() -> None:
    impact = MarginImpact(
        initial_margin_before=1.0,
        initial_margin_change=2.0,
        initial_margin_after=10.0,
        maintenance_margin_after=9.0,
    )
    assert impact.projected_initial_margin == pytest.approx(10.0)
    assert impact.projected_maintenance_margin == pytest.approx(9.0)


# ---------------------------------------------------------------------------
# Live ib_async adapter mapping (offline: pure object construction only)
# ---------------------------------------------------------------------------


def test_ib_async_adapter_maps_contract_and_orders() -> None:
    pytest.importorskip("ib_async")
    from src.execution.ibkr_client import (
        BagLeg,
        BagOrderSpec,
        ContractRequest,
        _IbAsyncIBClient,
    )

    adapter = _IbAsyncIBClient()
    request = ContractRequest(
        root="ES",
        last_trade_date_or_contract_month="202403",
        exchange="CME",
        multiplier=50.0,
        expiration=date(2024, 3, 15),
    )
    future = adapter._build_contract(request)
    assert future.symbol == "ES"
    assert future.lastTradeDateOrContractMonth == "202403"
    assert future.exchange == "CME"
    # IBKR rejects qualification requests that carry a multiplier; let the
    # resolved contract supply it instead.
    assert future.multiplier in ("", None)

    fx_request = ContractRequest(
        root="6E",
        last_trade_date_or_contract_month="202612",
        exchange="CME",
        multiplier=125000.0,
        symbol="EUR",
    )
    assert adapter._build_contract(fx_request).symbol == "EUR"

    order = adapter._to_ib_order(
        IBOrderRequest(
            order_ref="ord-1",
            action="BUY",
            quantity=3,
            order_type="LMT",
            limit_price=4800.0,
            algo_strategy="Adaptive",
            algo_params=(("adaptivePriority", "Normal"),),
        )
    )
    assert order.action == "BUY"
    assert order.totalQuantity == 3
    assert order.orderType == "LMT"
    assert order.lmtPrice == 4800.0
    assert order.orderRef == "ord-1"
    assert order.algoStrategy == "Adaptive"
    assert order.algoParams[0].tag == "adaptivePriority"

    venue = VenueContract(
        root="ES",
        con_id=123,
        exchange="CME",
        last_trade_date_or_contract_month="202403",
        multiplier=50.0,
        expiration=date(2024, 3, 15),
    )
    qualified = adapter._contract_for(venue)
    assert qualified.conId == 123
    assert adapter._contract_for(venue) is qualified  # cached

    bag = BagOrderSpec(
        root="ES",
        exchange="CME",
        legs=(
            BagLeg(con_id=123, action=OrderSide.SELL, exchange="CME"),
            BagLeg(con_id=456, action=OrderSide.BUY, exchange="CME"),
        ),
        order=IBOrderRequest(order_ref="r-1", action="BUY", quantity=2, order_type="MKT"),
    )
    combo = adapter._build_bag(bag)
    assert combo.secType == "BAG"
    assert combo.symbol == "ES"
    assert len(combo.comboLegs) == 2
    assert combo.comboLegs[0].conId == 123
    assert combo.comboLegs[0].action == "SELL"
    assert combo.comboLegs[1].action == "BUY"


def test_ib_async_adapter_normalises_trade_and_state() -> None:
    pytest.importorskip("ib_async")
    from src.execution.ibkr_client import _IbAsyncIBClient

    adapter = _IbAsyncIBClient()
    trade = SimpleNamespace(
        order=SimpleNamespace(orderRef="ord-9"),
        orderStatus=SimpleNamespace(status="Filled", filled=2.0, avgFillPrice=4810.0, whyHeld=""),
        fills=[
            SimpleNamespace(
                execution=SimpleNamespace(side="BOT", shares=2.0, price=4810.0),
                commissionReport=SimpleNamespace(commission=4.5),
                time=datetime(2024, 1, 2, 21, 0, tzinfo=UTC),
            )
        ],
    )
    venue = adapter._venue_trade(trade)
    assert venue.order_ref == "ord-9"
    assert venue.status is OrderStatus.FILLED
    assert venue.filled_quantity == 2
    assert venue.average_price == pytest.approx(4810.0)
    assert venue.commission == pytest.approx(4.5)
    assert venue.fills[0].side is OrderSide.BUY
    assert venue.is_terminal


# ---------------------------------------------------------------------------
# Phase-10: resilient liveness / reconnect surface (task-22)
# ---------------------------------------------------------------------------


async def test_ensure_connected_is_bounded_and_never_raises(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    manager = make_manager(settings, fake_client, clock)
    fake_client.fail_connects = 1

    # A single bounded probe: one transport attempt, no exception, error recorded.
    assert await manager.ensure_connected(max_attempts=1) is False
    assert fake_client.connect_calls == 1
    assert manager.last_error is not None
    assert manager.is_connected is False

    # Once TWS is reachable again the next probe reconnects.
    assert await manager.ensure_connected(max_attempts=1) is True
    assert manager.is_connected is True
    assert manager.last_error is None


async def test_ensure_connected_skips_nightly_reset_window(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    # FIXED is 16:00 New York -> inside a 23:45-00:45 reset window only when the
    # window is configured to cover it; use an explicit full-day window instead.
    manager = make_manager(
        settings,
        fake_client,
        clock,
        nightly_reset_start=time(15, 0),
        nightly_reset_end=time(17, 0),
    )
    assert manager.in_nightly_reset_window() is True
    assert await manager.ensure_connected(max_attempts=1) is False
    assert fake_client.connect_calls == 0
    assert manager.last_error is not None
    assert "nightly reset window" in manager.last_error


async def test_invalidate_marks_session_dead_and_heartbeat_reconnects(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    manager = make_manager(settings, fake_client, clock)
    await manager.connect()
    assert manager.is_connected is True

    # Simulated TWS restart: the SDK socket dies and a request path invalidates it.
    fake_client.connected = False
    manager.invalidate("connection reset by peer")
    assert manager.is_connected is False
    assert manager.last_error == "connection reset by peer"
    assert manager.is_reconnecting is False

    # The liveness layer re-establishes the session (heartbeat_tick path).
    assert await manager.heartbeat_tick() is True
    assert manager.is_connected is True
    assert manager.is_reconnecting is False
    assert manager.last_error is None
    await manager.disconnect()


async def test_ensure_heartbeat_restarts_a_dead_monitor(
    settings: Settings, fake_client: FakeIBClient, clock: FakeClock
) -> None:
    async def _yielding_sleep(seconds: float) -> None:
        clock.advance(seconds)
        await asyncio.sleep(0)

    manager = IBKRConnectionManager(
        settings,
        client_factory=lambda: fake_client,
        clock=clock,
        sleep=_yielding_sleep,
        heartbeat_enabled=True,
    )
    await manager.connect()
    task = manager._heartbeat_task
    assert task is not None
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert manager.heartbeat_running is False
    assert manager.ensure_heartbeat() is True
    assert manager.heartbeat_running is True
    await manager.disconnect()
