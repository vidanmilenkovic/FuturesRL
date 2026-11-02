"""IBKR paper/live market-data streamer built on ``ib_async``.

The streamer is the live/paper implementation of
:class:`src.core.interfaces.MarketDataStreamer`. It subscribes to 5-second
real-time bars, tick-by-tick trades and L1/L2 market depth for every configured
futures root, folds them into immutable :class:`~src.core.types.MarketState`
snapshots and exposes them through an async generator.

Design notes
------------
* ``ib_async`` is imported **lazily** (only when a real client/contract is
  created). A missing SDK raises an actionable :class:`RuntimeError` instead of
  crashing at import time — the data package stays importable for offline tests.
* The IB client and the contract resolver are **injectable** (``ib_factory=`` /
  ``contract_factory=``) so unit tests exercise connect/backoff/ingest/stream
  logic against a deterministic fake and never touch a TWS/Gateway socket.
* ``connect()`` retries with exponential backoff (initial delay, doubling, capped
  at the configured maximum) and ``reconnect()``/``heartbeat()`` support an
  external supervision loop. The sleep function is injectable so backoff tests
  run instantly.
* The streamer never fabricates a vendor feed: it only emits states built from
  data actually received from the socket.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Sequence
from datetime import UTC, datetime
from functools import partial
from typing import Any, Protocol, cast

from configs.instruments import InstrumentsConfig, InstrumentSpec, load_instruments
from configs.settings import Settings, get_settings

from src.core.types import (
    AssetClass,
    InstrumentId,
    InstrumentSnapshot,
    MarketState,
    MetadataValue,
    ensure_utc,
)
from src.data.contracts import parse_contract_symbol
from src.data.schema import validate_roots

__all__ = ["IBKRMarketDataStreamer"]

#: IBKR real-time bar size in seconds (a "5-second bar" feed).
BAR_INTERVAL_SECONDS: int = 5

#: Number of market-depth levels requested per root (L2).
MARKET_DEPTH_LEVELS: int = 10

#: Default futures universe streamed when none is configured.
DEFAULT_STREAM_ROOTS: tuple[InstrumentId, ...] = ("ES", "NQ", "CL", "GC", "ZN")

type SleepFunction = Callable[[float], Awaitable[None]]


class _IBClient(Protocol):
    """The subset of the ``ib_async.IB`` API this streamer depends on.

    The method names intentionally mirror the ``ib_async`` camelCase API; they
    are only ever reached through the injected client / real ``ib_async.IB``.
    """

    def isConnected(self) -> bool:  # noqa: N802
        """Whether the TWS/Gateway session is currently connected."""
        ...

    async def connectAsync(  # noqa: N802
        self, host: str, port: int, client_id: int, timeout: float
    ) -> None:
        """Open the TWS/Gateway session."""
        ...

    def disconnect(self) -> None:
        """Close the TWS/Gateway session."""
        ...

    def reqRealTimeBars(  # noqa: N802
        self, contract: object, bar_size: int, what_to_show: str, use_rth: bool
    ) -> object:
        """Subscribe to real-time bars for ``contract``."""
        ...

    def cancelRealTimeBars(self, bars: object) -> None:  # noqa: N802
        """Cancel a real-time bar subscription."""
        ...

    def reqTickByTickData(  # noqa: N802
        self, contract: object, tick_type: str, number_of_ticks: int, ignore_size: bool
    ) -> object:
        """Subscribe to tick-by-tick trades."""
        ...

    def cancelTickByTickData(  # noqa: N802
        self, contract: object, tick_type: str
    ) -> None:
        """Cancel a tick-by-tick subscription."""
        ...

    def reqMktData(  # noqa: N802
        self, contract: object, generic_ticks: str, snapshot: bool, regulatory: bool
    ) -> object:
        """Subscribe to L1 market data."""
        ...

    def cancelMktData(self, contract: object) -> None:  # noqa: N802
        """Cancel an L1 subscription."""
        ...

    def reqMktDepth(  # noqa: N802
        self, contract: object, num_rows: int, is_smart_depth: bool
    ) -> object:
        """Subscribe to L2 market depth."""
        ...

    def cancelMktDepth(self, contract: object, is_smart_depth: bool) -> None:  # noqa: N802
        """Cancel an L2 subscription."""
        ...


class _BarState:
    """Latest known market data for one root."""

    __slots__ = (
        "ask",
        "ask_size",
        "asks",
        "bid",
        "bid_size",
        "bids",
        "last_price",
        "last_size",
        "open_interest",
        "volume",
    )

    def __init__(self) -> None:
        self.last_price: float | None = None
        self.last_size: float = 0.0
        self.bid: float | None = None
        self.ask: float | None = None
        self.bid_size: float = 0.0
        self.ask_size: float = 0.0
        self.volume: float = 0.0
        self.open_interest: float = 0.0
        self.bids: list[tuple[float, float]] = []
        self.asks: list[tuple[float, float]] = []

    def mid(self) -> float | None:
        """Mid price when a two-sided quote exists, else the last trade."""
        if self.bid is not None and self.ask is not None:
            return 0.5 * (self.bid + self.ask)
        return self.last_price

    def mark(self) -> float | None:
        """Best available mark price."""
        return self.mid() or self.last_price


def _default_sleep(seconds: float) -> Awaitable[None]:
    """Default (real) backoff sleep: ``asyncio.sleep``."""
    return asyncio.sleep(seconds)


def _default_ib_factory() -> _IBClient:
    """Create a real ``ib_async.IB`` client (lazy import, actionable error)."""
    try:
        import ib_async
    except ImportError as exc:  # pragma: no cover - SDK installed in CI
        raise RuntimeError(
            "The ib_async SDK is not installed. Install the optional live extra "
            "(`pip install 'futuresrl-x[live]'` or `pip install ib_async`) before "
            "connecting to IBKR; offline/backtest code must not need it."
        ) from exc
    return cast(_IBClient, ib_async.IB())


class IBKRMarketDataStreamer:
    """Async IBKR market-data streamer implementing ``MarketDataStreamer``."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        roots: Sequence[InstrumentId] | None = None,
        instrument_master: InstrumentsConfig | None = None,
        ib_factory: Callable[[], _IBClient] | None = None,
        contract_factory: Callable[[InstrumentId], object] | None = None,
        sleep: SleepFunction | None = None,
        queue_maxsize: int = 1024,
    ) -> None:
        self._settings = settings if settings is not None else get_settings()
        self._roots = validate_roots(tuple(roots) if roots is not None else DEFAULT_STREAM_ROOTS)
        self._master = instrument_master
        self._ib_factory = ib_factory if ib_factory is not None else _default_ib_factory
        self._contract_factory = contract_factory
        self._sleep: SleepFunction = sleep if sleep is not None else _default_sleep
        self._queue: asyncio.Queue[MarketState | None] = asyncio.Queue(maxsize=queue_maxsize)
        self._stream_closed = False
        self._client: _IBClient | None = None
        self._states: dict[InstrumentId, _BarState] = {root: _BarState() for root in self._roots}
        self._latest_state: MarketState | None = None
        self._subscriptions: dict[InstrumentId, tuple[object, object, object, object]] = {}
        self._subscription_roots: dict[int, InstrumentId] = {}
        self._backoff_delays: list[float] = []

    # -- introspection ---------------------------------------------------------

    @property
    def roots(self) -> tuple[InstrumentId, ...]:
        """Futures roots streamed by this instance."""
        return self._roots

    @property
    def settings(self) -> Settings:
        """Runtime settings (host/port/backoff) driving this streamer."""
        return self._settings

    @property
    def is_connected(self) -> bool:
        """Whether the underlying IBKR session is connected."""
        return self._client is not None and bool(self._client.isConnected())

    @property
    def latest_state(self) -> MarketState | None:
        """Most recently emitted market state (``None`` before the first event)."""
        return self._latest_state

    @property
    def backoff_delays(self) -> tuple[float, ...]:
        """Backoff delays (seconds) used during the last failed connect attempts."""
        return tuple(self._backoff_delays)

    @property
    def instrument_master(self) -> InstrumentsConfig:
        """Instrument master used for IBKR exchange/currency resolution."""
        if self._master is None:
            self._master = load_instruments(self._settings.instruments_path)
        return self._master

    # -- connection lifecycle --------------------------------------------------

    def _create_client(self) -> _IBClient:
        """Create the injected/real IB client (SDK import happens here, lazily)."""
        return self._ib_factory()

    async def connect(self) -> None:
        """Connect with exponential backoff and subscribe to every root."""
        client = self._client if self._client is not None else self._create_client()
        await self._connect_with_backoff(client)
        self._client = client
        self._subscribe(client)
        self._attach_events(client)
        self._stream_closed = False
        # A previous disconnect may have enqueued the termination sentinel.
        while not self._queue.empty():
            with contextlib.suppress(asyncio.QueueEmpty):
                self._queue.get_nowait()

    async def _connect_with_backoff(self, client: _IBClient) -> None:
        """Retry ``connectAsync`` with exponential backoff."""
        self._backoff_delays.clear()
        initial = max(self._settings.ibkr_reconnect_initial_delay_seconds, 0.0)
        maximum = max(self._settings.ibkr_reconnect_max_delay_seconds, initial)
        attempts = max(1, self._settings.ibkr_reconnect_max_attempts)
        delay = initial
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                await client.connectAsync(
                    self._settings.ibkr_host,
                    self._settings.ibkr_port,
                    self._settings.ibkr_client_id,
                    self._settings.ibkr_connect_timeout_seconds,
                )
            except Exception as exc:
                last_error = exc
            else:
                if client.isConnected():
                    return
                last_error = RuntimeError("IB client reported disconnected after connectAsync")
            if attempt < attempts - 1:
                self._backoff_delays.append(delay)
                await self._sleep(delay)
                delay = min(delay * 2.0, maximum)
        raise RuntimeError(
            f"IBKR connection failed after {attempts} attempt(s) to "
            f"{self._settings.ibkr_host}:{self._settings.ibkr_port}: {last_error}"
        ) from last_error

    async def reconnect(self) -> None:
        """Tear down the current session (if any) and reconnect with backoff."""
        if self._client is not None:
            self._cancel_subscriptions(self._client)
            with contextlib.suppress(Exception):
                self._client.disconnect()
            self._client = None
        self._subscriptions.clear()
        self._subscription_roots.clear()
        await self.connect()

    async def disconnect(self) -> None:
        """Cancel every subscription and close the session (terminates ``stream``)."""
        client = self._client
        if client is not None:
            self._cancel_subscriptions(client)
            with contextlib.suppress(Exception):
                client.disconnect()
        self._client = None
        self._subscriptions.clear()
        self._subscription_roots.clear()
        if not self._stream_closed:
            self._stream_closed = True
            self._queue.put_nowait(None)

    async def heartbeat(self) -> bool:
        """Supervision hook: reconnect when the session dropped; return health."""
        if self.is_connected:
            return True
        await self.reconnect()
        return self.is_connected

    @property
    def tick_size(self) -> float:
        """Smallest configured tick size (diagnostic helper for book features)."""
        sizes = [
            spec.tick_size
            for spec in (self._spec_or_none(root) for root in self._roots)
            if spec is not None
        ]
        return min(sizes) if sizes else 0.0

    # -- subscriptions / event plumbing ---------------------------------------

    def _contract_for(self, root: InstrumentId) -> object:
        """Resolve the IBKR contract for ``root`` (injected or real ``ib_async.Future``)."""
        if self._contract_factory is not None:
            return self._contract_factory(root)
        try:
            import ib_async
        except ImportError as exc:  # pragma: no cover - SDK installed in CI
            raise RuntimeError(
                "The ib_async SDK is not installed; install the optional live extra "
                "(`pip install 'futuresrl-x[live]'`) or inject a contract_factory."
            ) from exc
        spec = self._spec_or_none(root)
        exchange = spec.ibkr_exchange if spec is not None else "CME"
        currency = spec.currency if spec is not None else "USD"
        return ib_async.Future(symbol=root, exchange=exchange, currency=currency)

    def _subscribe(self, client: _IBClient) -> None:
        """Subscribe to 5-second bars, tick-by-tick trades, L1 quotes and L2 depth."""
        for root in self._roots:
            contract = self._contract_for(root)
            bars = client.reqRealTimeBars(contract, BAR_INTERVAL_SECONDS, "TRADES", False)
            ticks = client.reqTickByTickData(contract, "AllLast", 0, False)
            quotes = client.reqMktData(contract, "", False, False)
            depth = client.reqMktDepth(contract, MARKET_DEPTH_LEVELS, False)
            self._subscriptions[root] = (bars, ticks, quotes, depth)
            for subscription in (bars, ticks, quotes, depth):
                req_id = getattr(subscription, "reqId", None)
                if isinstance(req_id, int):
                    self._subscription_roots[req_id] = root

    def _cancel_subscriptions(self, client: _IBClient) -> None:
        """Best-effort cancellation of every subscription of ``client``."""
        for root, subscriptions in self._subscriptions.items():
            bars = subscriptions[0]
            contract = self._contract_for(root)
            cancellations: list[Callable[[], None]] = [
                partial(client.cancelRealTimeBars, bars),
                partial(client.cancelTickByTickData, contract, "AllLast"),
                partial(client.cancelMktData, contract),
                partial(client.cancelMktDepth, contract, False),
            ]
            for cancel in cancellations:
                with contextlib.suppress(Exception):
                    cancel()

    def _attach_events(self, client: _IBClient) -> None:
        """Attach the ib_async event handlers (no-op for clients without events)."""
        handlers: tuple[tuple[str, Callable[..., None]], ...] = (
            ("realtimeBarEvent", self._on_realtime_bar_event),
            ("tickByTickAllLastEvent", self._on_tick_by_tick_event),
            ("pendingTickersEvent", self._on_pending_tickers_event),
            ("updateMktDepthEvent", self._on_market_depth_event),
            ("updateMktDepthL2Event", self._on_market_depth_event),
        )
        for name, handler in handlers:
            signal = getattr(client, name, None)
            if signal is None:
                continue
            try:
                signal += handler
            except TypeError:  # pragma: no cover - fake clients without Event support
                continue

    # -- public ingest API (event handlers + deterministic fakes in tests) -----

    def ingest_bar(
        self,
        root: InstrumentId,
        *,
        timestamp: datetime,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: float = 0.0,
        open_interest: float = 0.0,
    ) -> MarketState:
        """Fold one real-time bar into the state and emit it."""
        normalised_root = self._validate_root(root)
        moment = ensure_utc(timestamp)
        state = self._state(normalised_root)
        # The real-time bar payload is trusted for OHLC shape (IBKR guarantees
        # high >= max(open, close) >= min(open, close) >= low); the close is the
        # mark used by the snapshot, so only finite positive marks are accepted.
        for name, value in (("open", open), ("high", high), ("low", low), ("close", close)):
            if not math.isfinite(float(value)):
                raise ValueError(f"non-finite {name} in {normalised_root!r} real-time bar")
        state.last_price = float(close)
        state.volume = float(volume)
        state.open_interest = float(open_interest)
        return self._emit(moment)

    def ingest_tick(
        self,
        root: InstrumentId,
        *,
        timestamp: datetime,
        price: float,
        size: float = 0.0,
    ) -> MarketState:
        """Fold one tick-by-tick trade into the state and emit it."""
        normalised_root = self._validate_root(root)
        moment = ensure_utc(timestamp)
        state = self._state(normalised_root)
        state.last_price = float(price)
        state.last_size = float(size)
        return self._emit(moment)

    def ingest_quote(
        self,
        root: InstrumentId,
        *,
        timestamp: datetime,
        bid: float,
        ask: float,
        bid_size: float = 0.0,
        ask_size: float = 0.0,
    ) -> MarketState:
        """Fold one L1 quote into the state and emit it."""
        normalised_root = self._validate_root(root)
        moment = ensure_utc(timestamp)
        if bid > ask:
            raise ValueError(f"crossed L1 quote for {normalised_root!r}: bid {bid} > ask {ask}")
        state = self._state(normalised_root)
        state.bid = float(bid)
        state.ask = float(ask)
        state.bid_size = float(bid_size)
        state.ask_size = float(ask_size)
        return self._emit(moment)

    def ingest_book(
        self,
        root: InstrumentId,
        *,
        timestamp: datetime,
        bids: Sequence[tuple[float, float]],
        asks: Sequence[tuple[float, float]],
    ) -> MarketState:
        """Fold one L2 depth snapshot into the state and emit it."""
        normalised_root = self._validate_root(root)
        moment = ensure_utc(timestamp)
        state = self._state(normalised_root)
        state.bids = [(float(price), float(size)) for price, size in bids]
        state.asks = [(float(price), float(size)) for price, size in asks]
        if state.bids:
            best_bid = max(price for price, _ in state.bids)
            state.bid = best_bid
            state.bid_size = sum(size for price, size in state.bids if price == best_bid)
        if state.asks:
            best_ask = min(price for price, _ in state.asks)
            state.ask = best_ask
            state.ask_size = sum(size for price, size in state.asks if price == best_ask)
        return self._emit(moment)

    # -- async stream ----------------------------------------------------------

    def stream(self) -> AsyncIterator[MarketState]:
        """Yield immutable market states as IBKR events arrive."""
        return self._stream()

    async def _stream(self) -> AsyncIterator[MarketState]:
        queue = self._queue
        while True:
            state = await queue.get()
            if state is None:
                return
            yield state

    # -- state assembly --------------------------------------------------------

    def _validate_root(self, root: InstrumentId) -> InstrumentId:
        normalised = root.strip().upper()
        if normalised not in self._states:
            raise ValueError(f"root {root!r} is not streamed; configured roots {self._roots!r}")
        return normalised

    def _state(self, root: InstrumentId) -> _BarState:
        return self._states[root]

    def _emit(self, timestamp: datetime) -> MarketState:
        """Build a market state from every known root and publish it."""
        snapshots: list[InstrumentSnapshot] = []
        for root in self._roots:
            snapshot = self._snapshot(root)
            if snapshot is not None:
                snapshots.append(snapshot)
        if not snapshots:
            raise RuntimeError("no market data has been received yet; cannot build a state")
        stamp = ensure_utc(timestamp)
        metadata: dict[str, MetadataValue] = {
            "source": "ibkr",
            "bar_interval_seconds": float(BAR_INTERVAL_SECONDS),
            "environment": self._settings.environment,
        }
        state = MarketState.from_snapshots(stamp, snapshots, metadata=metadata)
        self._latest_state = state
        if self._stream_closed:
            self._stream_closed = False
        try:
            self._queue.put_nowait(state)
        except asyncio.QueueFull:  # pragma: no cover - backpressure policy
            with contextlib.suppress(asyncio.QueueEmpty):
                self._queue.get_nowait()
            self._queue.put_nowait(state)
        return state

    def _snapshot(self, root: InstrumentId) -> InstrumentSnapshot | None:
        """Build the current snapshot for ``root`` (``None`` before any data)."""
        state = self._states[root]
        price = state.mark()
        if price is None or not math.isfinite(price) or price <= 0:
            return None
        bid = state.bid
        ask = state.ask
        if bid is not None and ask is not None and bid > ask:
            bid = ask = None
        spec = self._spec_or_none(root)
        features = self._features(state, spec)
        return InstrumentSnapshot(
            instrument_id=root,
            root_symbol=root,
            price=float(price),
            asset_class=spec.asset_class if spec is not None else AssetClass.UNKNOWN,
            bid=bid,
            ask=ask,
            volume=max(state.volume, 0.0),
            open_interest=max(state.open_interest, 0.0),
            settlement=state.last_price,
            features=features,
        )

    def _features(self, state: _BarState, spec: InstrumentSpec | None) -> dict[str, float]:
        """Book/tick features attached to the snapshot (finite values only)."""
        features: dict[str, float] = {}
        mid = state.mid()
        if mid is not None:
            features["mid"] = float(mid)
        if state.bid is not None and state.ask is not None:
            features["spread"] = float(state.ask - state.bid)
            features["l1_imbalance"] = _imbalance(state.bid_size, state.ask_size)
        if state.bids and state.asks:
            bid_depth = sum(size for _, size in state.bids)
            ask_depth = sum(size for _, size in state.asks)
            features["l2_imbalance"] = _imbalance(bid_depth, ask_depth)
            features["bid_depth"] = float(bid_depth)
            features["ask_depth"] = float(ask_depth)
        if spec is not None:
            features["multiplier"] = float(spec.multiplier)
            features["tick_size"] = float(spec.tick_size)
        features["last_size"] = float(state.last_size)
        return {name: value for name, value in features.items() if math.isfinite(value)}

    def _spec_or_none(self, root: InstrumentId) -> InstrumentSpec | None:
        master = self.instrument_master
        return master.get(root) if master.has(root) else None

    # -- ib_async event handlers ----------------------------------------------

    def _on_realtime_bar_event(self, *args: object) -> None:
        """Handle ``realtimeBarEvent(bars, bar)``."""
        bars = args[0] if args else None
        bar = args[1] if len(args) > 1 else None
        if bar is None and isinstance(bars, tuple) and len(bars) == 2:
            bars, bar = bars
        if bar is None:
            return
        root = self._root_for(getattr(bars, "contract", None))
        if root is None:
            return
        self.ingest_bar(
            root,
            timestamp=cast(datetime, getattr(bar, "time", datetime.now(UTC))),
            open=float(getattr(bar, "open", 0.0)),
            high=float(getattr(bar, "high", 0.0)),
            low=float(getattr(bar, "low", 0.0)),
            close=float(getattr(bar, "close", 0.0)),
            volume=float(getattr(bar, "volume", 0.0)),
        )

    def _on_tick_by_tick_event(self, *args: object) -> None:
        """Handle ``tickByTickAllLastEvent(reqId, tickType, time, price, size, ...)``."""
        if len(args) < 5:
            return
        req_id, _tick_type, time, price, size = args[:5]
        root = self._subscription_roots.get(req_id) if isinstance(req_id, int) else None
        if root is None:
            if len(self._roots) != 1:
                return
            root = self._roots[0]
        moment = time if isinstance(time, datetime) else datetime.now(UTC)
        self.ingest_tick(
            root, timestamp=moment, price=float(cast(float, price)), size=float(cast(float, size))
        )

    def _on_pending_tickers_event(self, *args: object) -> None:
        """Handle ``pendingTickersEvent(tickers)``: L1 quote/trade refresh."""
        tickers = args[0] if args else None
        if tickers is None:
            return
        for ticker in cast(Iterable[Any], tickers):
            root = self._root_for(getattr(ticker, "contract", None))
            if root is None:
                continue
            last = getattr(ticker, "last", None)
            bid = getattr(ticker, "bid", None)
            ask = getattr(ticker, "ask", None)
            volume = getattr(ticker, "volume", None)
            bids = getattr(ticker, "domBids", None)
            asks = getattr(ticker, "domAsks", None)
            moment = datetime.now(UTC)
            if bids and asks:
                self.ingest_book(
                    root,
                    timestamp=moment,
                    bids=[(float(entry.price), float(entry.size)) for entry in bids],
                    asks=[(float(entry.price), float(entry.size)) for entry in asks],
                )
            elif (
                isinstance(bid, (int, float))
                and isinstance(ask, (int, float))
                and bid > 0
                and ask > 0
            ):
                self.ingest_quote(
                    root,
                    timestamp=moment,
                    bid=float(bid),
                    ask=float(ask),
                    bid_size=float(getattr(ticker, "bidSize", 0.0) or 0.0),
                    ask_size=float(getattr(ticker, "askSize", 0.0) or 0.0),
                )
            elif isinstance(last, (int, float)) and last > 0:
                self.ingest_tick(
                    root,
                    timestamp=moment,
                    price=float(last),
                    size=float(volume or 0.0),
                )

    def _on_market_depth_event(self, *args: object) -> None:
        """Handle ``updateMktDepthEvent(ticker, position, ...)`` by refreshing the book."""
        ticker = args[0] if args else None
        if ticker is None:
            return
        self._on_pending_tickers_event([ticker])

    def _root_for(self, contract: object) -> InstrumentId | None:
        """Map an IBKR contract/ticker back to a configured root."""
        if contract is None:
            return None
        for attribute in ("symbol", "localSymbol"):
            value = getattr(contract, attribute, None)
            if isinstance(value, str) and value:
                root = self._root_from_symbol(value)
                if root is not None:
                    return root
        return None

    def _root_from_symbol(self, symbol: str) -> InstrumentId | None:
        """Resolve a raw IBKR symbol (``"ESZ4"``) to a configured root."""
        normalised = symbol.strip().upper()
        if normalised in self._states:
            return normalised
        parsed = parse_contract_symbol(normalised, roots=self._roots, reference=datetime.now(UTC))
        if parsed is not None and parsed[0] in self._states:
            return parsed[0]
        for root in sorted(self._roots, key=len, reverse=True):
            if normalised.startswith(root):
                return root
        return None


def _imbalance(bid_size: float, ask_size: float) -> float:
    """Normalised order-book imbalance in ``[-1, 1]`` (0 when empty)."""
    total = bid_size + ask_size
    if total <= 0:
        return 0.0
    return (bid_size - ask_size) / total
