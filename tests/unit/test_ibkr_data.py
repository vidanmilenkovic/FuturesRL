"""Unit tests for the IBKR market-data streamer.

The streamer is driven entirely through an injectable fake ``ib_async`` client:
no TWS/Gateway socket, no network, no real contract resolution. Tests cover
lazy-SDK errors, exponential backoff, subscriptions, bar/quote/tick/book ingest,
the async stream generator, heartbeat reconnection and the event handlers.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest
from configs.settings import Settings
from src.core.interfaces import MarketDataStreamer
from src.data.ibkr_data import (
    BAR_INTERVAL_SECONDS,
    MARKET_DEPTH_LEVELS,
    IBKRMarketDataStreamer,
)

NOW = datetime(2024, 1, 2, 21, 0, tzinfo=UTC)


class FakeSubscription:
    """Stand-in for an ``ib_async`` subscription object (with a request id)."""

    def __init__(self, req_id: int) -> None:
        self.reqId = req_id


class FakeIB:
    """Deterministic fake of the ``ib_async.IB`` subset used by the streamer."""

    def __init__(self, failures: int = 0) -> None:
        self.failures = failures
        self.connected = False
        self.connect_calls = 0
        self.disconnect_calls = 0
        self.subscriptions: dict[str, list[FakeSubscription]] = {}
        self.cancellations: list[str] = []
        self._next_req_id = 1

    def isConnected(self) -> bool:  # noqa: N802 - mirrors the ib_async API
        return self.connected

    async def connectAsync(self, host: str, port: int, client_id: int, timeout: float) -> None:  # noqa: N802
        del host, port, client_id, timeout
        self.connect_calls += 1
        if self.connect_calls <= self.failures:
            raise ConnectionError("connection refused")
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False
        self.disconnect_calls += 1

    def _subscribe(self, kind: str) -> FakeSubscription:
        subscription = FakeSubscription(self._next_req_id)
        self._next_req_id += 1
        self.subscriptions.setdefault(kind, []).append(subscription)
        return subscription

    def reqRealTimeBars(  # noqa: N802
        self, contract: object, bar_size: int, what_to_show: str, use_rth: bool
    ) -> FakeSubscription:
        del contract
        assert bar_size == BAR_INTERVAL_SECONDS
        assert what_to_show == "TRADES"
        del use_rth
        return self._subscribe("bars")

    def reqTickByTickData(  # noqa: N802
        self, contract: object, tick_type: str, number_of_ticks: int, ignore_size: bool
    ) -> FakeSubscription:
        del contract, number_of_ticks, ignore_size
        assert tick_type == "AllLast"
        return self._subscribe("ticks")

    def reqMktData(  # noqa: N802
        self, contract: object, generic_ticks: str, snapshot: bool, regulatory: bool
    ) -> FakeSubscription:
        del contract, generic_ticks, snapshot, regulatory
        return self._subscribe("quotes")

    def reqMktDepth(  # noqa: N802
        self, contract: object, num_rows: int, is_smart_depth: bool
    ) -> FakeSubscription:
        del contract, is_smart_depth
        assert num_rows == MARKET_DEPTH_LEVELS
        return self._subscribe("depth")

    def cancelRealTimeBars(self, bars: object) -> None:  # noqa: N802
        self.cancellations.append("bars")

    def cancelTickByTickData(self, contract: object, tick_type: str) -> None:  # noqa: N802
        self.cancellations.append("ticks")

    def cancelMktData(self, contract: object) -> None:  # noqa: N802
        self.cancellations.append("quotes")

    def cancelMktDepth(self, contract: object, is_smart_depth: bool) -> None:  # noqa: N802
        self.cancellations.append("depth")


def contract_factory(root: str) -> object:
    """Fake contract resolver (no ib_async import)."""
    return SimpleNamespace(symbol=root, localSymbol=f"{root}Z4")


class Harness:
    """Bundles a streamer, its fake client and the recorded backoff sleeps."""

    def __init__(
        self,
        settings: Settings,
        *,
        roots: tuple[str, ...] = ("ES", "NQ"),
        failures: int = 0,
        max_attempts: int | None = None,
    ) -> None:
        updates: dict[str, Any] = {}
        if max_attempts is not None:
            updates["ibkr_reconnect_max_attempts"] = max_attempts
        self.settings = settings.model_copy(update=updates) if updates else settings
        self.fake = FakeIB(failures)
        self.sleeps: list[float] = []
        self.clients: list[FakeIB] = []

        async def fake_sleep(seconds: float) -> None:
            self.sleeps.append(seconds)

        def factory() -> FakeIB:
            client = FakeIB(0) if self.clients else self.fake
            self.clients.append(client)
            return client

        self.streamer = IBKRMarketDataStreamer(
            self.settings,
            roots=roots,
            ib_factory=factory,
            contract_factory=contract_factory,
            sleep=fake_sleep,
        )


def build_harness(settings: Settings, **kwargs: Any) -> Harness:
    return Harness(settings, **kwargs)


# ---------------------------------------------------------------------------
# Lazy SDK / connection
# ---------------------------------------------------------------------------


async def test_missing_sdk_raises_actionable_error(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "ib_async", None)
    streamer = IBKRMarketDataStreamer(settings, roots=("ES",))
    with pytest.raises(RuntimeError, match="ib_async SDK is not installed"):
        await streamer.connect()
    assert streamer.is_connected is False
    assert isinstance(streamer, MarketDataStreamer)


async def test_missing_sdk_when_resolving_contracts(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "ib_async", None)
    fake = FakeIB()
    streamer = IBKRMarketDataStreamer(settings, roots=("ES",), ib_factory=lambda: fake)
    with pytest.raises(RuntimeError, match="ib_async SDK is not installed"):
        await streamer.connect()
    assert fake.connect_calls == 1


async def test_connect_backoff_then_success(settings: Settings) -> None:
    harness = build_harness(settings, failures=1)
    await harness.streamer.connect()
    assert harness.streamer.is_connected is True
    assert harness.fake.connect_calls == 2
    assert harness.sleeps == [1.0]
    assert harness.streamer.backoff_delays == (1.0,)
    assert len(harness.fake.subscriptions["bars"]) == 2
    assert len(harness.fake.subscriptions["ticks"]) == 2
    assert len(harness.fake.subscriptions["quotes"]) == 2
    assert len(harness.fake.subscriptions["depth"]) == 2


async def test_connect_exhausts_attempts(settings: Settings) -> None:
    harness = build_harness(settings, failures=99, max_attempts=3)
    with pytest.raises(RuntimeError, match="failed after 3 attempt"):
        await harness.streamer.connect()
    assert harness.fake.connect_calls == 3
    assert harness.sleeps == [1.0, 2.0]
    assert harness.streamer.is_connected is False


async def test_connect_backoff_is_capped(settings: Settings) -> None:
    capped = settings.model_copy(
        update={
            "ibkr_reconnect_initial_delay_seconds": 2.0,
            "ibkr_reconnect_max_delay_seconds": 3.0,
            "ibkr_reconnect_max_attempts": 4,
        }
    )
    fake = FakeIB(failures=99)
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    streamer = IBKRMarketDataStreamer(
        capped,
        roots=("ES",),
        ib_factory=lambda: fake,
        contract_factory=contract_factory,
        sleep=fake_sleep,
    )
    with pytest.raises(RuntimeError):
        await streamer.connect()
    assert sleeps == [2.0, 3.0, 3.0]
    assert streamer.backoff_delays == (2.0, 3.0, 3.0)


async def test_disconnect_before_connect_is_safe(settings: Settings) -> None:
    harness = build_harness(settings)
    await harness.streamer.disconnect()
    assert harness.streamer.is_connected is False
    generator = harness.streamer.stream()
    with pytest.raises(StopAsyncIteration):
        await anext(generator)


# ---------------------------------------------------------------------------
# Ingest -> MarketState
# ---------------------------------------------------------------------------


async def test_ingest_bar_quote_tick_and_book(settings: Settings) -> None:
    harness = build_harness(settings)
    await harness.streamer.connect()
    streamer = harness.streamer
    assert streamer.latest_state is None

    quote_state = streamer.ingest_quote(
        "ES", timestamp=NOW, bid=4800.0, ask=4800.25, bid_size=10.0, ask_size=8.0
    )
    assert quote_state.ids == ("ES",)
    assert quote_state.price("ES") == pytest.approx(4800.125)
    assert quote_state.snapshot("ES").bid == 4800.0
    assert quote_state.snapshot("ES").ask == 4800.25
    assert quote_state.feature("ES", "l1_imbalance") == pytest.approx((10 - 8) / 18)

    bar_state = streamer.ingest_bar(
        "ES",
        timestamp=NOW + timedelta(seconds=5),
        open=4800.0,
        high=4801.0,
        low=4799.0,
        close=4800.5,
        volume=120.0,
        open_interest=1_000.0,
    )
    assert bar_state.snapshot("ES").volume == 120.0
    assert bar_state.snapshot("ES").open_interest == 1_000.0
    assert bar_state.snapshot("ES").settlement == 4800.5

    book_state = streamer.ingest_book(
        "ES",
        timestamp=NOW + timedelta(seconds=10),
        bids=[(4800.0, 10.0), (4799.75, 20.0)],
        asks=[(4800.25, 8.0), (4800.5, 5.0)],
    )
    assert book_state.feature("ES", "bid_depth") == pytest.approx(30.0)
    assert book_state.feature("ES", "ask_depth") == pytest.approx(13.0)
    assert book_state.feature("ES", "l2_imbalance") == pytest.approx((30 - 13) / 43)

    tick_state = streamer.ingest_tick(
        "NQ", timestamp=NOW + timedelta(seconds=11), price=16_800.0, size=2.0
    )
    assert tick_state.ids == ("ES", "NQ")
    assert tick_state.price("NQ") == pytest.approx(16_800.0)
    assert tick_state.feature("NQ", "last_size") == pytest.approx(2.0)
    assert streamer.latest_state is tick_state

    metadata = tick_state.metadata
    assert metadata["source"] == "ibkr"
    assert metadata["bar_interval_seconds"] == pytest.approx(5.0)
    assert metadata["environment"] == harness.settings.environment


async def test_ingest_validation(settings: Settings) -> None:
    harness = build_harness(settings)
    await harness.streamer.connect()
    streamer = harness.streamer
    with pytest.raises(ValueError, match="crossed L1 quote"):
        streamer.ingest_quote("ES", timestamp=NOW, bid=101.0, ask=100.0)
    with pytest.raises(ValueError, match="not streamed"):
        streamer.ingest_bar("GC", timestamp=NOW, open=1.0, high=1.0, low=1.0, close=1.0)
    with pytest.raises(ValueError, match="non-finite"):
        streamer.ingest_bar(
            "ES",
            timestamp=NOW,
            open=float("nan"),
            high=1.0,
            low=1.0,
            close=1.0,
        )
    with pytest.raises(RuntimeError, match="no market data"):
        streamer._emit(NOW)


async def test_emit_skips_roots_without_data(settings: Settings) -> None:
    harness = build_harness(settings)
    await harness.streamer.connect()
    state = harness.streamer.ingest_tick("NQ", timestamp=NOW, price=16_800.0)
    assert state.ids == ("NQ",)


# ---------------------------------------------------------------------------
# Async stream + disconnect
# ---------------------------------------------------------------------------


async def test_stream_yields_states_and_terminates_on_disconnect(settings: Settings) -> None:
    harness = build_harness(settings)
    await harness.streamer.connect()
    generator = harness.streamer.stream()
    harness.streamer.ingest_tick("ES", timestamp=NOW, price=4800.0)
    first = await asyncio.wait_for(anext(generator), timeout=1.0)
    assert first.price("ES") == pytest.approx(4800.0)

    await harness.streamer.disconnect()
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(anext(generator), timeout=1.0)
    assert harness.streamer.is_connected is False
    # 4 subscriptions per root x 2 roots.
    assert len(harness.fake.cancellations) == 8
    assert harness.fake.disconnect_calls == 1
    await harness.streamer.disconnect()  # idempotent


async def test_connect_drains_stale_stream_terminator(settings: Settings) -> None:
    harness = build_harness(settings)
    await harness.streamer.connect()
    harness.streamer.ingest_tick("ES", timestamp=NOW, price=4800.0)
    await harness.streamer.disconnect()
    await harness.streamer.connect()
    generator = harness.streamer.stream()
    harness.streamer.ingest_tick("ES", timestamp=NOW + timedelta(seconds=1), price=4801.0)
    state = await asyncio.wait_for(anext(generator), timeout=1.0)
    assert state.price("ES") == pytest.approx(4801.0)


async def test_heartbeat_reconnects_after_connection_loss(settings: Settings) -> None:
    harness = build_harness(settings)
    await harness.streamer.connect()
    assert await harness.streamer.heartbeat() is True
    assert len(harness.clients) == 1

    harness.clients[0].connected = False
    assert await harness.streamer.heartbeat() is True
    assert len(harness.clients) == 2
    assert harness.streamer.is_connected is True
    await harness.streamer.disconnect()


async def test_reconnect_replaces_client(settings: Settings) -> None:
    harness = build_harness(settings)
    await harness.streamer.connect()
    first = harness.streamer._client
    await harness.streamer.reconnect()
    assert harness.streamer._client is not first
    assert harness.streamer.is_connected is True
    assert len(harness.clients) == 2
    await harness.streamer.disconnect()


# ---------------------------------------------------------------------------
# Event handlers / root resolution
# ---------------------------------------------------------------------------


async def test_realtime_bar_event_handler(settings: Settings) -> None:
    harness = build_harness(settings)
    await harness.streamer.connect()
    bars = SimpleNamespace(contract=SimpleNamespace(symbol="ESZ4"))
    bar = SimpleNamespace(time=NOW, open=4800.0, high=4801.0, low=4799.0, close=4800.5, volume=42.0)
    harness.streamer._on_realtime_bar_event(bars, bar)
    latest = harness.streamer.latest_state
    assert latest is not None
    assert latest.price("ES") == pytest.approx(4800.5)
    assert latest.snapshot("ES").volume == 42.0
    # Malformed payloads are ignored rather than crashing the event loop.
    harness.streamer._on_realtime_bar_event()
    harness.streamer._on_realtime_bar_event((bars, bar))


async def test_tick_by_tick_event_handler_uses_request_mapping(settings: Settings) -> None:
    harness = build_harness(settings)
    await harness.streamer.connect()
    tick_subscription = harness.fake.subscriptions["ticks"][1]  # NQ (sorted roots)
    harness.streamer._on_tick_by_tick_event(
        tick_subscription.reqId, "AllLast", NOW, 16_800.0, 3.0, None, "CME", ""
    )
    latest = harness.streamer.latest_state
    assert latest is not None
    assert latest.price("NQ") == pytest.approx(16_800.0)
    harness.streamer._on_tick_by_tick_event(1, "AllLast")


async def test_pending_tickers_and_depth_handlers(settings: Settings) -> None:
    harness = build_harness(settings)
    await harness.streamer.connect()
    streamer = harness.streamer

    quote_ticker = SimpleNamespace(
        contract=SimpleNamespace(symbol="ESZ4", localSymbol="ESZ4"),
        last=4800.0,
        bid=4799.75,
        ask=4800.25,
        bidSize=10.0,
        askSize=12.0,
        volume=1_000.0,
        domBids=None,
        domAsks=None,
    )
    streamer._on_pending_tickers_event([quote_ticker])
    assert streamer.latest_state is not None
    assert streamer.latest_state.snapshot("ES").ask == 4800.25

    depth_ticker = SimpleNamespace(
        contract=SimpleNamespace(symbol="ESZ4"),
        last=None,
        bid=None,
        ask=None,
        volume=None,
        domBids=[SimpleNamespace(price=4800.0, size=5.0), SimpleNamespace(price=4799.75, size=5.0)],
        domAsks=[SimpleNamespace(price=4800.25, size=5.0)],
    )
    streamer._on_market_depth_event(depth_ticker)
    assert streamer.latest_state is not None
    assert streamer.latest_state.feature("ES", "l2_imbalance") == pytest.approx(1.0 / 3.0)

    trade_ticker = SimpleNamespace(
        contract=SimpleNamespace(symbol="NQZ4"),
        last=16_805.0,
        bid=None,
        ask=None,
        volume=7.0,
        domBids=None,
        domAsks=None,
    )
    streamer._on_pending_tickers_event([trade_ticker])
    assert streamer.latest_state is not None
    assert streamer.latest_state.price("NQ") == pytest.approx(16_805.0)

    streamer._on_pending_tickers_event([SimpleNamespace(contract=None)])
    streamer._on_market_depth_event()


async def test_root_resolution(settings: Settings) -> None:
    harness = build_harness(settings)
    streamer = harness.streamer
    assert streamer._root_from_symbol("ESZ4") == "ES"
    assert streamer._root_from_symbol("ES") == "ES"
    assert streamer._root_from_symbol("ZZZ") is None
    assert streamer._root_for(SimpleNamespace(symbol="NQZ4")) == "NQ"
    assert streamer._root_for(SimpleNamespace(localSymbol="ESZ4")) == "ES"
    assert streamer._root_for(SimpleNamespace(symbol="???")) is None
    assert streamer._root_for(None) is None
    assert streamer.tick_size == pytest.approx(0.25)  # both ES and NQ trade in 0.25 ticks
