"""Phase-10 unit tests: IBKR reconnect + honest last-known-good book (task-22).

Every test is offline and deterministic: the real :class:`IBKRConnectionManager`
talks to the in-memory :class:`FakeIBClient` transport, the clock/sleep are
injected and no test touches TWS, IB Gateway or the network.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime
from typing import Any, cast

import pytest
from configs.settings import Settings
from src.api import fixtures
from src.api.schemas import (
    AccountSnapshotResponse,
    HealthResponse,
    PositionRow,
)
from src.api.services import (
    IBKRBookUnavailableError,
    IBKRExecutionService,
    build_container,
)
from src.api.telemetry import TelemetryBroadcaster
from src.core.types import AssetClass
from src.execution.ibkr_client import IBKRConnectionManager, VenuePosition

from tests.unit.test_ibkr_client import FakeClock, FakeIBClient, make_account_rows

FIXED_TIMESTAMP = datetime(2026, 11, 2, 17, 0, tzinfo=UTC)

#: Live IBKR book values that are deliberately different from the deterministic
#: simulated broker's seeded demo book (~3.99M NLV), so any accidental fallback
#: is visible in the assertions.
LIVE_NLV = 2_000_000.0
LIVE_QUANTITY = 7


@pytest.fixture()
def clock() -> FakeClock:
    return FakeClock(FIXED_TIMESTAMP)


@pytest.fixture()
def fake_client() -> FakeIBClient:
    client = FakeIBClient()
    client.account_rows = make_account_rows(net_liquidation=str(LIVE_NLV))
    client.position_rows = [
        VenuePosition(
            root_symbol="MES",
            quantity=LIVE_QUANTITY,
            avg_cost=4_800.0 * 5.0,
            multiplier=5.0,
            expiration=date(2026, 12, 18),
            local_symbol="MESZ6",
        )
    ]
    return client


@pytest.fixture()
def ibkr_settings(settings: Settings) -> Settings:
    """Single bounded connect attempt + short timeout (offline reconnect tests)."""
    return settings.model_copy(
        update={
            "ibkr_connect_timeout_seconds": 0.05,
            "ibkr_reconnect_max_attempts": 1,
            "ibkr_heartbeat_interval_seconds": 0.05,
        }
    )


@pytest.fixture()
def market(settings: Settings, clock: FakeClock) -> fixtures.OfflineMarket:
    return fixtures.build_offline_market(settings.instruments, clock=clock)


def _manager(settings: Settings, client: FakeIBClient, clock: FakeClock) -> IBKRConnectionManager:
    return IBKRConnectionManager(
        settings,
        client_factory=lambda: client,
        clock=clock,
        sleep=clock.sleep,
        random_source=lambda: 0.5,
        heartbeat_enabled=False,
    )


def _service(
    ibkr_settings: Settings,
    market: fixtures.OfflineMarket,
    clock: FakeClock,
    manager: IBKRConnectionManager,
) -> IBKRExecutionService:
    return IBKRExecutionService(
        settings=ibkr_settings,
        market=market,
        clock=clock,
        connection_manager=manager,
    )


# ---------------------------------------------------------------------------
# Last-known-good book (never simulated)
# ---------------------------------------------------------------------------


async def test_account_reports_live_ibkr_book(
    ibkr_settings: Settings,
    market: fixtures.OfflineMarket,
    clock: FakeClock,
    fake_client: FakeIBClient,
) -> None:
    service = _service(ibkr_settings, market, clock, _manager(ibkr_settings, fake_client, clock))
    assert service.ibkr_enabled is True
    assert service.source_name == "engine"
    assert service.connected is False

    assert await service.connect() is True
    account = await service.account()

    assert account.net_liquidation == pytest.approx(LIVE_NLV)
    assert account.source == "engine"
    assert account.stale is False
    assert account.stale_reason is None
    assert [row.quantity for row in account.positions] == [LIVE_QUANTITY]
    assert service.connected is True
    assert service.has_last_good_book is True
    assert service.last_book_at == clock.now
    assert service.last_error is None
    await service.disconnect()


async def test_disconnected_serves_last_good_stale_book(
    ibkr_settings: Settings,
    market: fixtures.OfflineMarket,
    clock: FakeClock,
    fake_client: FakeIBClient,
) -> None:
    manager = _manager(ibkr_settings, fake_client, clock)
    service = _service(ibkr_settings, market, clock, manager)
    assert await service.connect() is True
    fresh = await service.account()
    assert fresh.stale is False

    # Simulated TWS restart: socket dies, background reconnect cannot reach it yet.
    fake_client.connected = False
    fake_client.fail_connects = 50
    manager.invalidate("connection reset by peer")
    stale = await service.account()

    assert stale.stale is True
    assert stale.source == "engine"
    assert stale.net_liquidation == pytest.approx(LIVE_NLV)
    assert stale.positions and stale.positions[0].quantity == LIVE_QUANTITY
    assert stale.stale_reason is not None
    assert "last-known-good" in stale.stale_reason
    # Never the simulated book.
    assert stale.net_liquidation != fixtures.DEFAULT_INITIAL_EQUITY

    positions = await service.positions()
    assert positions["MES"].quantity == LIVE_QUANTITY
    await service.stop_background()
    await service.disconnect()


async def test_never_connected_raises_instead_of_simulated(
    ibkr_settings: Settings,
    market: fixtures.OfflineMarket,
    clock: FakeClock,
    fake_client: FakeIBClient,
) -> None:
    fake_client.fail_connects = 50
    service = _service(ibkr_settings, market, clock, _manager(ibkr_settings, fake_client, clock))
    assert service.has_last_good_book is False

    with pytest.raises(IBKRBookUnavailableError, match="no live account book"):
        await service.account()
    with pytest.raises(IBKRBookUnavailableError, match="no live account book"):
        await service.positions()
    await service.stop_background()


async def test_fetch_failure_invalidates_and_serves_stale(
    ibkr_settings: Settings,
    market: fixtures.OfflineMarket,
    clock: FakeClock,
    fake_client: FakeIBClient,
) -> None:
    manager = _manager(ibkr_settings, fake_client, clock)
    service = _service(ibkr_settings, market, clock, manager)
    assert await service.connect() is True
    assert (await service.account()).stale is False

    async def _boom() -> Any:
        raise ConnectionError("socket reset mid-fetch")

    fake_client.account_values = _boom  # type: ignore[method-assign]
    fake_client.fail_connects = 50
    stale = await service.account()

    assert stale.stale is True
    assert stale.net_liquidation == pytest.approx(LIVE_NLV)
    assert manager.is_connected is False
    assert service.last_error is not None
    assert "ConnectionError" in service.last_error
    await service.stop_background()
    await service.disconnect()


# ---------------------------------------------------------------------------
# Reconnect / liveness
# ---------------------------------------------------------------------------


async def test_reconnects_after_tws_returns(
    ibkr_settings: Settings,
    market: fixtures.OfflineMarket,
    clock: FakeClock,
    fake_client: FakeIBClient,
) -> None:
    fake_client.fail_connects = 50
    service = _service(ibkr_settings, market, clock, _manager(ibkr_settings, fake_client, clock))
    assert await service.connect() is False
    assert service.connected is False
    assert service.last_error is not None

    fake_client.fail_connects = 0
    assert await service.ensure_connected() is True
    assert service.connected is True
    assert service.last_error is None
    account = await service.account()
    assert account.stale is False
    assert account.net_liquidation == pytest.approx(LIVE_NLV)
    await service.disconnect()


async def test_ensure_connected_is_throttled(
    ibkr_settings: Settings,
    market: fixtures.OfflineMarket,
    clock: FakeClock,
    fake_client: FakeIBClient,
) -> None:
    fake_client.fail_connects = 500
    service = _service(ibkr_settings, market, clock, _manager(ibkr_settings, fake_client, clock))
    assert await service.ensure_connected() is False
    attempts = fake_client.connect_calls
    assert attempts == 1
    # Within the probe interval a second request-path probe is a no-op.
    assert await service.ensure_connected() is False
    assert fake_client.connect_calls == attempts
    await service.stop_background()


async def test_background_liveness_reconnects_automatically(
    ibkr_settings: Settings,
    market: fixtures.OfflineMarket,
    clock: FakeClock,
    fake_client: FakeIBClient,
) -> None:
    fake_client.fail_connects = 1
    manager = _manager(ibkr_settings, fake_client, clock)
    service = _service(ibkr_settings, market, clock, manager)
    service._probe_interval = 0.01
    assert await service.connect() is False
    assert service.connected is False

    await service.start_background()
    try:
        for _ in range(400):
            if service.connected:
                break
            await asyncio.sleep(0.01)
        assert service.connected is True
        assert service.reconnecting is False
        account = await service.account()
        assert account.stale is False
        assert account.net_liquidation == pytest.approx(LIVE_NLV)
    finally:
        await service.stop_background()
    assert service._liveness_task is None
    await service.disconnect()


async def test_flatten_never_falls_back_to_simulated(
    ibkr_settings: Settings,
    market: fixtures.OfflineMarket,
    clock: FakeClock,
    fake_client: FakeIBClient,
) -> None:
    fake_client.fail_connects = 500
    service = _service(ibkr_settings, market, clock, _manager(ibkr_settings, fake_client, clock))
    with pytest.raises(IBKRBookUnavailableError, match="simulated"):
        await service.flatten_all()
    await service.stop_background()


# ---------------------------------------------------------------------------
# Telemetry: never overwrite a live book with simulated positions
# ---------------------------------------------------------------------------


def _position_row(quantity: int = LIVE_QUANTITY) -> PositionRow:
    return PositionRow(
        instrument_id="MES",
        root_symbol="MES",
        asset_class=AssetClass.EQUITY_INDEX,
        contract_symbol="MESZ6",
        expiration=date(2026, 12, 18),
        days_to_roll=45,
        quantity=quantity,
        average_price=4_800.0,
        market_price=4_800.0,
        multiplier=5.0,
        notional=quantity * 4_800.0 * 5.0,
        weight=0.084,
        unrealized_pnl=0.0,
        initial_margin=quantity * 1_500.0,
        maintenance_margin=quantity * 1_400.0,
    )


def _account(net_liquidation: float = LIVE_NLV) -> AccountSnapshotResponse:
    return AccountSnapshotResponse(
        timestamp=FIXED_TIMESTAMP,
        net_liquidation=net_liquidation,
        cash_balance=net_liquidation * 0.5,
        buying_power=net_liquidation * 2.0,
        initial_margin_used=net_liquidation * 0.10,
        maintenance_margin_used=net_liquidation * 0.09,
        excess_liquidity=net_liquidation * 0.90,
        margin_utilization=0.10,
        day_pnl=0.0,
        day_pnl_pct=0.0,
        gross_leverage=0.084,
        net_exposure=net_liquidation * 0.084,
        positions=[_position_row()],
        source="engine",
    )


class _FakeExecution:
    def __init__(self, *, ibkr_enabled: bool) -> None:
        self.ibkr_enabled = ibkr_enabled
        self.account_result: AccountSnapshotResponse | None = _account()
        self.error: Exception | None = None
        self.account_calls = 0

    async def account(self) -> AccountSnapshotResponse:
        self.account_calls += 1
        if self.error is not None:
            raise self.error
        if self.account_result is None:  # pragma: no cover - defensive
            raise IBKRBookUnavailableError("no book")
        return self.account_result


def _broadcaster(execution: _FakeExecution, settings: Settings) -> TelemetryBroadcaster:
    return TelemetryBroadcaster(
        settings=settings,
        clock=lambda: FIXED_TIMESTAMP,
        execution=cast(Any, execution),
        risk=cast(Any, object()),
        market=cast(Any, object()),
    )


async def test_telemetry_never_publishes_simulated_book_for_ibkr(
    settings: Settings,
) -> None:
    execution = _FakeExecution(ibkr_enabled=True)
    execution.error = IBKRBookUnavailableError("no live account book")
    broadcaster = _broadcaster(execution, settings)

    # No honest book: account frames degrade to heartbeats (never simulated data).
    for kind in ("pnl", "margin", "positions"):
        envelope = await broadcaster._build_envelope(kind)
        assert envelope.kind == "heartbeat"

    # The live IBKR book arrives: account frames now carry it.
    execution.error = None
    envelope = await broadcaster._build_envelope("positions")
    assert envelope.kind == "positions"
    assert envelope.payload.positions[0].quantity == LIVE_QUANTITY
    assert envelope.payload.gross_leverage == pytest.approx(0.084)

    # A later refresh failure keeps the last-known-good IBKR book in the stream.
    execution.error = IBKRBookUnavailableError("TWS restarted")
    broadcaster._account_refreshed_at = None
    envelope = await broadcaster._build_envelope("positions")
    assert envelope.kind == "positions"
    assert envelope.payload.positions[0].quantity == LIVE_QUANTITY
    assert envelope.payload.gross_leverage == pytest.approx(0.084)


async def test_telemetry_offline_execution_keeps_existing_cache(
    settings: Settings,
) -> None:
    execution = _FakeExecution(ibkr_enabled=False)
    broadcaster = _broadcaster(execution, settings)
    for kind in ("pnl", "margin", "positions", "positions"):
        await broadcaster._build_envelope(kind)
    # Deterministic offline book is fetched exactly once, as before.
    assert execution.account_calls == 1


# ---------------------------------------------------------------------------
# Container health: dynamically derived IBKR provenance
# ---------------------------------------------------------------------------


class _FakeIBKRSurface:
    def __init__(
        self,
        *,
        ibkr_enabled: bool,
        connected: bool,
        reconnecting: bool = False,
        last_error: str | None = None,
    ) -> None:
        self.ibkr_enabled = ibkr_enabled
        self.connected = connected
        self.reconnecting = reconnecting
        self.last_error = last_error


def _health_for(surface: object, settings: Settings) -> HealthResponse:
    container = build_container(settings, clock=lambda: FIXED_TIMESTAMP)
    container.execution = cast(Any, surface)
    return container.health()


def test_health_offline_process_reports_ibkr_disabled(settings: Settings) -> None:
    health = _health_for(_FakeIBKRSurface(ibkr_enabled=False, connected=False), settings)
    assert health.offline_mode is True
    assert health.ibkr_enabled is False
    assert health.ibkr_connected is False
    assert health.ibkr_reconnecting is False
    assert health.ibkr_last_error is None


def test_health_connected_ibkr_process_is_online(settings: Settings) -> None:
    health = _health_for(_FakeIBKRSurface(ibkr_enabled=True, connected=True), settings)
    assert health.offline_mode is False
    assert health.status == "ok"
    assert health.ibkr_enabled is True
    assert health.ibkr_connected is True
    assert health.ibkr_reconnecting is False
    assert health.ibkr_last_error is None


def test_health_disconnected_ibkr_process_is_degraded_and_honest(settings: Settings) -> None:
    health = _health_for(
        _FakeIBKRSurface(
            ibkr_enabled=True,
            connected=False,
            reconnecting=True,
            last_error="TimeoutError: TWS 127.0.0.1:7497 unavailable",
        ),
        settings,
    )
    assert health.offline_mode is True
    assert health.status == "degraded"
    assert health.ibkr_enabled is True
    assert health.ibkr_connected is False
    assert health.ibkr_reconnecting is True
    assert health.ibkr_last_error is not None
    assert "TWS" in health.ibkr_last_error
