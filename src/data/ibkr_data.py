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
import inspect
import logging
import math
import threading
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from functools import partial
from typing import Any, Protocol, cast

import polars as pl
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
from src.data import schema as s
from src.data.contracts import next_contract_expiration, parse_contract_symbol
from src.data.ibkr_contracts import FrontFuture, build_front_contract, qualify_front_future
from src.data.schema import validate_roots

__all__ = ["IBKRHistoricalBarFetcher", "IBKRMarketDataStreamer"]

logger = logging.getLogger(__name__)

#: IBKR real-time bar size in seconds (a "5-second bar" feed).
BAR_INTERVAL_SECONDS: int = 5

#: Number of market-depth levels requested per root (L2).
MARKET_DEPTH_LEVELS: int = 10

#: Default futures universe streamed when none is configured.
DEFAULT_STREAM_ROOTS: tuple[InstrumentId, ...] = ("ES", "NQ", "CL", "GC", "ZN")

type SleepFunction = Callable[[float], Awaitable[None]]

#: Injectable delayed/historical fallback feed for roots whose real-time
#: subscriptions are not permitted: (roots, start, end) -> bar-like rows/frame.
type DelayedBarSource = Callable[[Sequence[InstrumentId], datetime, datetime], Awaitable[object]]

#: IBKR error codes meaning "no/insufficient market-data permission" for the
#: requested tick/depth subscriptions (10189 no permissions, 10090 partial
#: subscription, 354 not subscribed).
MARKET_DATA_PERMISSION_ERROR_CODES: frozenset[int] = frozenset({10189, 10090, 354})

#: IBKR error strings that accompany the permission codes above.
_MARKET_DATA_PERMISSION_TOKENS: tuple[str, ...] = (
    "10189",
    "10090",
    "354",
    "market data permission",
    "market-data permission",
    "not subscribed",
    "no market data permissions",
)

#: Default interval (seconds) between delayed/historical fallback polls.
DEFAULT_DELAYED_POLL_INTERVAL_SECONDS: float = 60.0
_DELAYED_HISTORICAL_DURATION = "1 D"
_DELAYED_HISTORICAL_BAR_SIZE = "1 min"


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

    async def qualifyContractsAsync(self, *contracts: object) -> object:  # noqa: N802
        """Resolve contracts to qualified versions carrying a ``conId``.

        Optional at runtime: duck-typed/legacy clients without this method fall
        back to their raw contracts, while a real ``ib_async.IB`` always qualifies
        before any ``reqRealTimeBars``/``reqTickByTickData``/``reqMktData``/
        ``reqMktDepth`` call (unqualified IBKR contracts cannot be hashed).
        """
        ...

    def reqMarketDataType(self, market_data_type: int) -> None:  # noqa: N802
        """Select the IBKR market-data type (1 live, 2 frozen, 3 delayed, 4 delayed-frozen)."""
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


def _has_con_id(contract: object) -> bool:
    """Whether ``contract`` carries an IBKR contract id (``conId``/``con_id``)."""
    con_id = getattr(contract, "conId", None)
    if con_id is None:
        con_id = getattr(contract, "con_id", None)
    return isinstance(con_id, int) and not isinstance(con_id, bool) and con_id > 0


def _first_qualified_contract(result: object) -> object | None:
    """Pick the first qualified contract from an ``ib_async`` qualification result.

    ``qualifyContractsAsync`` returns a list with one qualified contract per
    requested contract (fewer/none when resolution fails); single-contract
    duck-typed fakes are accepted too.
    """
    if result is None:
        return None
    candidates: tuple[object, ...] = (
        tuple(result) if isinstance(result, (list, tuple)) else (result,)
    )
    for candidate in candidates:
        if candidate is not None and _has_con_id(candidate):
            return candidate
    return None


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
            "(`pip install 'futuresrl[live]'` or `pip install ib_async`) before "
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
        clock: Callable[[], datetime] | None = None,
        delayed_poll_interval_seconds: float | None = DEFAULT_DELAYED_POLL_INTERVAL_SECONDS,
        delayed_bar_source: DelayedBarSource | None = None,
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
        self._contracts: dict[InstrumentId, object] = {}
        self._front_futures: dict[InstrumentId, FrontFuture] = {}
        self._subscriptions: dict[InstrumentId, tuple[object, object, object, object]] = {}
        self._subscription_roots: dict[int, InstrumentId] = {}
        self._backoff_delays: list[float] = []
        self._clock: Callable[[], datetime] = (
            clock if clock is not None else lambda: datetime.now(UTC)
        )
        self._contract_cache: dict[tuple[str, str], FrontFuture] = {}
        self._degraded_roots: set[InstrumentId] = set()
        self._fallback_reasons: dict[InstrumentId, str] = {}
        self._delayed_poll_interval = delayed_poll_interval_seconds
        self._delayed_bar_source = delayed_bar_source
        self._delayed_poll_task: asyncio.Task[None] | None = None
        self._delayed_polling_unsupported = False
        self._requested_market_data_type: int | None = None
        self._market_data_type_error: str | None = None

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
    def fallback_active(self) -> bool:
        """Whether any root is degraded to delayed/historical polling."""
        return bool(self._degraded_roots)

    @property
    def fallback_roots(self) -> tuple[InstrumentId, ...]:
        """Roots currently served through the delayed/historical fallback."""
        return tuple(sorted(self._degraded_roots))

    def diagnostics(self) -> dict[str, object]:
        """JSON-friendly market-data diagnostics (fallback status, market-data type)."""
        polling = self._delayed_poll_task is not None and not self._delayed_poll_task.done()
        return {
            "connected": self.is_connected,
            "market_data_type": {
                "configured": int(self._settings.ibkr_market_data_type),
                "requested": self._requested_market_data_type,
                "error": self._market_data_type_error,
            },
            "market_data_fallback": {
                "active": bool(self._degraded_roots),
                "roots": sorted(self._degraded_roots),
                "reasons": dict(self._fallback_reasons),
                "polling": polling,
                "poll_interval_seconds": self._delayed_poll_interval,
                "unsupported": self._delayed_polling_unsupported,
            },
            "subscriptions": sorted(self._subscriptions),
            "qualified_contracts": {
                root: front.to_dict() for root, front in sorted(self._front_futures.items())
            },
        }

    def _now(self) -> datetime:
        """Current timestamp from the injectable clock, normalised to UTC."""
        moment = self._clock()
        if not isinstance(moment, datetime):
            raise TypeError(f"clock must return a datetime, got {type(moment)!r}")
        return ensure_utc(moment)

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
        self._reset_fallback_state()
        self._front_futures.clear()
        await self._request_market_data_type(client)
        # Attach error handlers before subscribing so asynchronous IBKR permission
        # errors emitted during subscription setup are already routed to fallback.
        self._attach_events(client)
        try:
            await self._subscribe(client)
        except Exception:
            # A subscription failure must not leave a half-initialised session:
            # tear the connection down so a later retry/heartbeat starts cleanly.
            self._stop_fallback_polling()
            self._reset_fallback_state()
            self._client = None
            self._contracts.clear()
            self._front_futures.clear()
            self._subscriptions.clear()
            self._subscription_roots.clear()
            with contextlib.suppress(Exception):
                client.disconnect()
            raise
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
        self._stop_fallback_polling()
        self._reset_fallback_state()
        self._front_futures.clear()
        if self._client is not None:
            self._cancel_subscriptions(self._client)
            with contextlib.suppress(Exception):
                self._client.disconnect()
            self._client = None
        self._contracts.clear()
        self._subscriptions.clear()
        self._subscription_roots.clear()
        await self.connect()

    async def disconnect(self) -> None:
        """Cancel every subscription and close the session (terminates ``stream``)."""
        self._stop_fallback_polling()
        self._reset_fallback_state()
        self._front_futures.clear()
        client = self._client
        if client is not None:
            self._cancel_subscriptions(client)
            with contextlib.suppress(Exception):
                client.disconnect()
        self._client = None
        self._contracts.clear()
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

    def _contract_for(self, root: InstrumentId, *, reference: datetime | None = None) -> object:
        """Resolve the explicit front-month IBKR contract for ``root``.

        Never returns a bare ``Future``: the contract always carries an explicit
        ``lastTradeDateOrContractMonth`` derived from the instrument roll rules
        (or the injected ``contract_factory`` for tests/legacy adapters).
        """
        contract, _month, _expiration = build_front_contract(
            root,
            reference=reference if reference is not None else self._now(),
            instrument_master=self._master,
            raw_contract_factory=self._contract_factory,
        )
        return contract

    async def _subscribe(self, client: _IBClient) -> None:
        """Qualify every front-month contract first, then subscribe per root.

        Contracts are resolved through :func:`src.data.ibkr_contracts.qualify_front_future`
        (explicit contract month + conId verification + per-month cache), never a
        bare ``Future``. If a root cannot be qualified it is skipped with an
        actionable warning; if no root qualifies an aggregate error is raised.

        Real-time tick/depth requests may be rejected for account/market-data
        permission reasons (synchronous exceptions or asynchronous IBKR error
        codes 10189/10090/354). Those roots are **not** allowed to crash the
        streamer: their paid subscriptions are cancelled, the root is marked
        degraded and served through delayed/historical polling instead while the
        remaining roots keep their real-time subscriptions.
        """
        failures: list[str] = []
        qualified: dict[InstrumentId, FrontFuture] = {}
        for root in self._roots:
            try:
                front = await qualify_front_future(
                    client,
                    root,
                    reference=self._now(),
                    cache=self._contract_cache,
                    instrument_master=self._master,
                    raw_contract_factory=self._contract_factory,
                )
            except RuntimeError as exc:
                failures.append(str(exc))
                logger.warning("%s; skipping market-data subscription", exc)
                continue
            qualified[root] = front
            self._front_futures[root] = front
            self._contracts[root] = front.contract
        if not qualified:
            raise RuntimeError(
                "IBKR could not qualify any market-data contract; no subscriptions were "
                "made. " + " | ".join(failures)
            )
        for root, front in qualified.items():
            created: list[tuple[str, object]] = []
            try:
                bars = client.reqRealTimeBars(front.contract, BAR_INTERVAL_SECONDS, "TRADES", False)
                created.append(("bars", bars))
                ticks = client.reqTickByTickData(front.contract, "AllLast", 0, False)
                created.append(("ticks", ticks))
                quotes = client.reqMktData(front.contract, "", False, False)
                created.append(("quotes", quotes))
                depth = client.reqMktDepth(front.contract, MARKET_DEPTH_LEVELS, False)
                created.append(("depth", depth))
            except Exception as exc:
                self._cancel_created_subscriptions(client, root, created)
                reason = (
                    f"market-data permission denied for {root!r}: {exc}"
                    if _is_market_data_permission_error(exc)
                    else f"market-data subscription failed for {root!r}: {exc}"
                )
                logger.warning("%s; switching %r to delayed/historical polling", reason, root)
                self._degrade_root(root, reason)
                continue
            self._subscriptions[root] = (bars, ticks, quotes, depth)
            for subscription in (bars, ticks, quotes, depth):
                req_id = getattr(subscription, "reqId", None)
                if isinstance(req_id, int):
                    self._subscription_roots[req_id] = root
        if self._degraded_roots:
            self._ensure_fallback_polling()

    async def _request_market_data_type(self, client: _IBClient) -> None:
        """Request the configured IBKR market-data type on connect (fake-safe)."""
        configured = int(self._settings.ibkr_market_data_type)
        setter = getattr(client, "reqMarketDataType", None)
        self._market_data_type_error = None
        if not callable(setter):
            logger.debug(
                "IBKR client has no reqMarketDataType support; keeping the session default"
            )
            self._requested_market_data_type = configured
            return
        try:
            result = setter(configured)
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            self._market_data_type_error = f"{type(exc).__name__}: {exc}"
            logger.warning("IBKR reqMarketDataType(%d) failed: %s", configured, exc)
            return
        self._requested_market_data_type = configured

    # -- delayed / historical fallback polling --------------------------------

    async def poll_delayed_roots(
        self, roots: Sequence[InstrumentId] | None = None
    ) -> dict[str, int]:
        """Poll one delayed/historical snapshot per degraded root.

        Each polled bar is fed through the existing :meth:`ingest_bar` /
        :meth:`ingest_quote` path, so degraded roots keep producing
        :class:`~src.core.types.MarketState` updates without real-time
        subscriptions. Returns ``{root: rows_ingested}``.
        """
        client = self._client
        if client is None:
            return {}
        targets = tuple(sorted(self._degraded_roots)) if roots is None else tuple(roots)
        ingested: dict[str, int] = {}
        for root in targets:
            try:
                rows = await self._fetch_delayed_rows(client, root)
            except Exception as exc:
                logger.warning("IBKR delayed fallback poll failed for %r: %s", root, exc)
                continue
            latest = _latest_delayed_row(rows)
            if latest is None:
                ingested[root] = 0
                continue
            self._ingest_delayed_row(root, latest)
            ingested[root] = 1
        return ingested

    async def _fetch_delayed_rows(self, client: _IBClient, root: InstrumentId) -> list[object]:
        """Fetch delayed/historical rows for ``root`` (injected feed or IBKR history)."""
        end = self._now()
        start = end - timedelta(days=1)
        if self._delayed_bar_source is not None:
            result = await self._delayed_bar_source((root,), start, end)
            return _normalise_delayed_rows(result)
        contract = self._contracts.get(root)
        if contract is None:
            front = self._front_futures.get(root)
            contract = front.contract if front is not None else self._contract_for(root)
        history = getattr(client, "reqHistoricalDataAsync", None)
        if not callable(history):
            if not self._delayed_polling_unsupported:
                self._delayed_polling_unsupported = True
                logger.warning(
                    "IBKR delayed fallback for %r is unavailable: the client has no "
                    "reqHistoricalDataAsync and no delayed_bar_source was injected",
                    root,
                )
            return []
        try:
            result = history(
                contract,
                end,
                _DELAYED_HISTORICAL_DURATION,
                _DELAYED_HISTORICAL_BAR_SIZE,
                "TRADES",
                False,
                2,
                False,
            )
            if inspect.isawaitable(result):
                result = await result
        except Exception as exc:
            logger.warning("IBKR delayed history request failed for %r: %s", root, exc)
            return []
        return _normalise_delayed_rows(result)

    def _ingest_delayed_row(self, root: InstrumentId, row: object) -> None:
        """Feed one delayed/historical row through the existing ingest path."""
        timestamp = _bar_timestamp(_row_field(row, "timestamp_utc", "timestamp", "date", "time"))
        timestamp = timestamp if timestamp is not None else self._now()
        close = _positive_float(_row_field(row, "close", "price", "last"))
        if close is None:
            return
        bid = _positive_float(_row_field(row, "bid_px", "bid", "bidPrice"))
        ask = _positive_float(_row_field(row, "ask_px", "ask", "askPrice"))
        if bid is not None and ask is not None and bid <= ask:
            self.ingest_quote(
                root,
                timestamp=timestamp,
                bid=bid,
                ask=ask,
                bid_size=_finite_float(_row_field(row, "bid_sz", "bidSize")) or 0.0,
                ask_size=_finite_float(_row_field(row, "ask_sz", "askSize")) or 0.0,
            )
        open_price = _positive_float(_row_field(row, "open")) or close
        high = max(_positive_float(_row_field(row, "high")) or close, open_price, close)
        low = min(_positive_float(_row_field(row, "low")) or close, open_price, close)
        volume = _finite_float(_row_field(row, "volume")) or 0.0
        try:
            self.ingest_bar(
                root,
                timestamp=timestamp,
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=max(volume, 0.0),
            )
        except ValueError as exc:
            logger.debug("ignoring unusable delayed bar for %r: %s", root, exc)

    def _degrade_root(self, root: InstrumentId, reason: str) -> None:
        """Mark ``root`` degraded, cancel its subscriptions and start fallback polling."""
        self._fallback_reasons.setdefault(root, reason)
        if root in self._degraded_roots:
            return
        self._degraded_roots.add(root)
        client = self._client
        if client is not None:
            self._cancel_root_subscriptions(client, root)
        logger.warning(
            "IBKR market data degraded for root %r (%s); serving it via delayed/historical polling",
            root,
            reason,
        )
        self._ensure_fallback_polling()

    def _reset_fallback_state(self) -> None:
        """Reset degraded/fallback state (called on connect/disconnect)."""
        self._stop_fallback_polling()
        self._degraded_roots.clear()
        self._fallback_reasons.clear()
        self._delayed_polling_unsupported = False
        self._requested_market_data_type = None
        self._market_data_type_error = None

    def _ensure_fallback_polling(self) -> None:
        """Start the background delayed/historical polling loop when possible."""
        interval = self._delayed_poll_interval
        if not self._degraded_roots or interval is None or interval <= 0.0:
            return
        if self._delayed_poll_task is not None and not self._delayed_poll_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._delayed_poll_task = loop.create_task(self._delayed_poll_loop())

    async def _delayed_poll_loop(self) -> None:
        """Background loop polling degraded roots until disconnect/recovery."""
        interval = float(self._delayed_poll_interval or 0.0)
        try:
            while self._client is not None and self._degraded_roots:
                try:
                    await self.poll_delayed_roots()
                except Exception as exc:
                    logger.warning("IBKR delayed fallback poll loop error: %s", exc)
                await self._sleep(interval)
        finally:
            self._delayed_poll_task = None

    def _stop_fallback_polling(self) -> None:
        """Cancel the background delayed/historical polling task (idempotent)."""
        task = self._delayed_poll_task
        self._delayed_poll_task = None
        if task is not None and not task.done():
            task.cancel()

    def _cancel_created_subscriptions(
        self,
        client: _IBClient,
        root: InstrumentId,
        created: Sequence[tuple[str, object]],
    ) -> None:
        """Best-effort cancellation of subscriptions created before a failure."""
        front = self._front_futures.get(root)
        contract = self._contracts.get(root) or (
            front.contract if front is not None else self._contract_for(root)
        )
        for kind, handle in created:
            with contextlib.suppress(Exception):
                if kind == "bars":
                    client.cancelRealTimeBars(handle)
                elif kind == "ticks":
                    client.cancelTickByTickData(contract, "AllLast")
                elif kind == "quotes":
                    client.cancelMktData(contract)
                elif kind == "depth":
                    client.cancelMktDepth(contract, False)

    def _cancel_root_subscriptions(self, client: _IBClient, root: InstrumentId) -> None:
        """Cancel every real-time subscription of one degraded root."""
        subscriptions = self._subscriptions.pop(root, None)
        if subscriptions is not None:
            front = self._front_futures.get(root)
            contract = self._contracts.get(root) or (
                front.contract if front is not None else self._contract_for(root)
            )
            self._cancel_handles(client, subscriptions, contract)
        for req_id, mapped_root in list(self._subscription_roots.items()):
            if mapped_root == root:
                self._subscription_roots.pop(req_id, None)

    def _cancel_handles(
        self,
        client: _IBClient,
        subscriptions: tuple[object, object, object, object],
        contract: object,
    ) -> None:
        """Cancel the four real-time subscriptions of one root (best effort)."""
        bars = subscriptions[0]
        cancellations: list[Callable[[], None]] = [
            partial(client.cancelRealTimeBars, bars),
            partial(client.cancelTickByTickData, contract, "AllLast"),
            partial(client.cancelMktData, contract),
            partial(client.cancelMktDepth, contract, False),
        ]
        for cancel in cancellations:
            with contextlib.suppress(Exception):
                cancel()

    def _on_error_event(self, *args: object) -> None:
        """Handle IBKR ``errorEvent`` permission errors by degrading the root.

        ``ib_async`` emits ``errorEvent(reqId, errorCode, errorString, contract)``;
        codes 10189/10090/354 mean the tick/depth subscription is not permitted and
        must switch that root to the delayed/historical fallback instead of
        crashing the streamer.
        """
        if len(args) < 2:
            return
        if len(args) >= 3:
            req_id_raw, code_raw, message = args[0], args[1], args[2]
            contract = args[3] if len(args) > 3 else None
        else:
            req_id_raw, code_raw, message, contract = None, args[0], args[1], None
        try:
            code = int(cast(int, code_raw))
        except (TypeError, ValueError):
            return
        if code not in MARKET_DATA_PERMISSION_ERROR_CODES:
            return
        req_id = req_id_raw if isinstance(req_id_raw, int) else None
        root = self._subscription_roots.get(req_id) if req_id is not None else None
        if root is None and contract is not None:
            root = self._root_for(contract)
        if root is None or root not in self._states:
            return
        self._degrade_root(root, f"IBKR error {code}: {message}")

    def _cancel_subscriptions(self, client: _IBClient) -> None:
        """Best-effort cancellation of every subscription of ``client``."""
        for root, subscriptions in list(self._subscriptions.items()):
            front = self._front_futures.get(root)
            contract = self._contracts.get(root) or (
                front.contract if front is not None else self._contract_for(root)
            )
            self._cancel_handles(client, subscriptions, contract)

    def _attach_events(self, client: _IBClient) -> None:
        """Attach the ib_async event handlers (no-op for clients without events)."""
        handlers: tuple[tuple[str, Callable[..., None]], ...] = (
            ("realtimeBarEvent", self._on_realtime_bar_event),
            ("tickByTickAllLastEvent", self._on_tick_by_tick_event),
            ("pendingTickersEvent", self._on_pending_tickers_event),
            ("updateMktDepthEvent", self._on_market_depth_event),
            ("updateMktDepthL2Event", self._on_market_depth_event),
            ("errorEvent", self._on_error_event),
            ("error", self._on_error_event),
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


def _row_field(row: object, *names: str) -> object:
    """Read the first non-null field from a mapping or object row."""
    if isinstance(row, Mapping):
        for name in names:
            value = row.get(name)
            if value is not None:
                return value
        return None
    for name in names:
        value = getattr(row, name, None)
        if value is not None:
            return value
    return None


def _normalise_delayed_rows(result: object) -> list[object]:
    """Normalise a delayed/historical polling result into a list of row objects."""
    if result is None:
        return []
    if isinstance(result, pl.DataFrame):
        return list(result.iter_rows(named=True))
    if isinstance(result, (list, tuple)):
        return list(result)
    try:
        return list(cast(Iterable[object], result))
    except TypeError:
        return [result]


def _latest_delayed_row(rows: Sequence[object]) -> object | None:
    """Return the newest parseable row of a delayed/historical poll (or ``None``)."""
    latest: object | None = None
    latest_timestamp: datetime | None = None
    for row in rows:
        timestamp = _bar_timestamp(_row_field(row, "timestamp_utc", "timestamp", "date", "time"))
        if timestamp is None:
            if latest is None:
                latest = row
            continue
        if latest_timestamp is None or timestamp >= latest_timestamp:
            latest = row
            latest_timestamp = timestamp
    return latest


def _is_market_data_permission_error(exc: BaseException) -> bool:
    """Whether an IBKR subscription failure is a market-data permission error."""
    code = getattr(exc, "code", None)
    if (
        isinstance(code, int)
        and not isinstance(code, bool)
        and code in MARKET_DATA_PERMISSION_ERROR_CODES
    ):
        return True
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(token.lower() in text for token in _MARKET_DATA_PERMISSION_TOKENS)


# ---------------------------------------------------------------------------
# Historical daily bars (Phase 8 cold-start hydration fallback)
# ---------------------------------------------------------------------------

#: IBKR bar size used for cold-start hydration history.
HISTORICAL_BAR_SIZE: str = "1 day"

#: Maximum calendar span requested per IBKR historical call (IB limits long
#: durations); longer windows are fetched in chunks and de-duplicated.
HISTORICAL_MAX_REQUEST_DAYS: int = 365

#: ``formatDate=2`` makes IBKR return UTC epoch-based bar timestamps.
_IBKR_FORMAT_DATE_UTC: int = 2

#: Placeholder expiration for a continuous-futures series whose contract has no
#: single expiry (the hydrator adds a neutral next leg for the term structure).
_CONTINUOUS_PLACEHOLDER_EXPIRATION: date = date(2099, 12, 31)


class _HistoricalIBClient(Protocol):
    """The subset of the ``ib_async.IB`` API the historical fetcher depends on.

    Method names mirror the ``ib_async`` camelCase API; they are only ever
    reached through the injected client / a real ``ib_async.IB``.
    """

    def isConnected(self) -> bool:  # noqa: N802
        """Whether the TWS/IB Gateway session is currently connected."""
        ...

    async def connectAsync(  # noqa: N802
        self, host: str, port: int, client_id: int, timeout: float
    ) -> None:
        """Open the TWS/Gateway session."""
        ...

    def disconnect(self) -> None:
        """Close the TWS/Gateway session."""
        ...

    async def reqHistoricalDataAsync(  # noqa: N802
        self,
        contract: object,
        end_date_time: datetime,
        duration: str,
        bar_size: str,
        what_to_show: str,
        use_rth: bool,
        format_date: int,
        keep_up_to_date: bool,
    ) -> Iterable[object]:
        """Request historical bars, returning IBKR ``BarData``-like rows."""
        ...


def _finite_float(value: object) -> float | None:
    """Coerce an IBKR numeric attribute to a finite ``float`` (``None`` otherwise)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(cast("Any", value))
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _positive_float(value: object) -> float | None:
    """Coerce an IBKR price attribute to a finite strictly-positive ``float``."""
    result = _finite_float(value)
    return result if result is not None and result > 0.0 else None


def _bar_timestamp(value: object) -> datetime | None:
    """Normalise an IBKR bar date (``datetime``/``date``/epoch-ish) to UTC."""
    if isinstance(value, datetime):
        return ensure_utc(value) if value.tzinfo is not None else value.replace(tzinfo=UTC)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=UTC)
    return None


class IBKRHistoricalBarFetcher:
    """Fetch canonical daily historical bars from IBKR (TWS / IB Gateway).

    This is the cold-start hydration fallback used when no Databento API key is
    configured. It requests ``1 day`` TRADES bars through the lazily imported
    ``ib_async`` SDK (``reqHistoricalDataAsync``), parsing IBKR's ``BarData``
    rows into the canonical :mod:`src.data.schema` bar frame and stamping a
    contract expiration so the feature processor can derive term structure.

    Design rules
    ------------
    * **No network in tests**: the client and the contract resolution are both
      injectable (``ib_factory`` / ``contract_factory``); the SDK is only
      imported when the default factory is actually invoked.
    * **Sync bridge**: :meth:`fetch_bars_sync` is a :data:`BackfillFn`-compatible
      entry point. Because hydration is called synchronously *inside* the async
      runner loop, it executes the async fetch on a dedicated worker thread with
      its own event loop instead of nesting ``asyncio.run``.
    * **Chunked windows**: long lookbacks (e.g. 400 days) are split into
      ``HISTORICAL_MAX_REQUEST_DAYS`` windows because IBKR limits the duration of
      a single historical request.
    * **Fail closed**: connection failures and empty responses raise an
      actionable ``RuntimeError`` naming TWS/IB Gateway, ``FUTURESRL_ENVIRONMENT``
      and ``FUTURESRL_IBKR_PORT`` so the hydration chain can fall back cleanly.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        instrument_master: InstrumentsConfig | None = None,
        ib_factory: Callable[[], object] | None = None,
        contract_factory: Callable[[InstrumentId], object] | None = None,
        clock: Callable[[], datetime] | None = None,
        bar_size: str = HISTORICAL_BAR_SIZE,
        what_to_show: str = "TRADES",
        use_rth: bool = False,
        max_request_days: int = HISTORICAL_MAX_REQUEST_DAYS,
    ) -> None:
        if not bar_size:
            raise ValueError("bar_size must be a non-empty string")
        if not what_to_show:
            raise ValueError("what_to_show must be a non-empty string")
        if max_request_days < 1:
            raise ValueError("max_request_days must be >= 1")
        self._settings = settings if settings is not None else get_settings()
        self._master = instrument_master
        self._ib_factory = ib_factory
        self._contract_factory = contract_factory
        self._clock: Callable[[], datetime] = (
            clock if clock is not None else lambda: datetime.now(UTC)
        )
        self._bar_size = bar_size
        self._what_to_show = what_to_show
        self._use_rth = use_rth
        self._max_request_days = max_request_days

    # -- introspection ---------------------------------------------------------

    @property
    def settings(self) -> Settings:
        """Runtime settings (IBKR host/port/client id and credentials)."""
        return self._settings

    @property
    def instrument_master(self) -> InstrumentsConfig:
        """Instrument master used for exchange/currency and roll resolution."""
        if self._master is None:
            self._master = load_instruments(self._settings.instruments_path)
        return self._master

    def _spec_or_none(self, root: InstrumentId) -> InstrumentSpec | None:
        master = self.instrument_master
        return master.get(root) if master.has(root) else None

    # -- client / contract resolution -----------------------------------------

    def _create_client(self) -> _HistoricalIBClient:
        """Create the injected/real IB client (SDK import happens here, lazily)."""
        factory = self._ib_factory
        if factory is not None:
            return cast("_HistoricalIBClient", factory())
        try:
            import ib_async
        except ImportError as exc:  # pragma: no cover - SDK installed in CI
            raise RuntimeError(
                "The ib_async SDK is not installed; install the optional live extra "
                "(`pip install 'futuresrl[live]'` or `pip install ib_async`) before "
                "fetching IBKR historical data."
            ) from exc
        return cast("_HistoricalIBClient", ib_async.IB())

    def _contract_for(self, root: InstrumentId, *, reference: datetime | None = None) -> object:
        """Resolve the explicit front-month IBKR contract for historical data.

        Never uses a continuous/bare ``ContFuture``: the contract carries an
        explicit ``lastTradeDateOrContractMonth`` derived from the roll rules (or
        the injected ``contract_factory`` for tests).
        """
        contract, _month, _expiration = build_front_contract(
            root,
            reference=reference if reference is not None else self._clock(),
            instrument_master=self._master,
            raw_contract_factory=self._contract_factory,
        )
        return contract

    async def _qualify(self, client: _HistoricalIBClient, contract: object) -> object:
        """Qualify the contract when the client supports it (best effort)."""
        qualify = getattr(client, "qualifyContractsAsync", None)
        if not callable(qualify):
            return contract
        try:
            result = await cast("Callable[[object], Awaitable[object]]", qualify)(contract)
        except Exception as exc:  # pragma: no cover - fake clients never raise here
            logger.warning("IBKR contract qualification failed for history request: %s", exc)
            return contract
        try:
            items = list(cast("Iterable[object]", result))
        except TypeError:
            return contract
        valid = [item for item in items if item is not None]
        return valid[0] if valid else contract

    def _expiration_for(self, root: InstrumentId, start: datetime) -> date:
        """Representative contract expiration used to key the historical bars."""
        spec = self._spec_or_none(root)
        if spec is None:
            return _CONTINUOUS_PLACEHOLDER_EXPIRATION
        try:
            return next_contract_expiration(spec.roll.contract_months, start)
        except Exception:  # pragma: no cover - defensive against bad roll config
            return _CONTINUOUS_PLACEHOLDER_EXPIRATION

    # -- windows / parsing -----------------------------------------------------

    def _windows(self, start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
        """Split ``[start, end)`` into IBKR-sized historical request windows."""
        windows: list[tuple[datetime, datetime]] = []
        cursor = end
        while cursor > start:
            window_start = max(start, cursor - timedelta(days=self._max_request_days))
            windows.append((window_start, cursor))
            cursor = window_start
        return windows

    @staticmethod
    def _duration_days(window: tuple[datetime, datetime]) -> int:
        window_start, window_end = window
        seconds = max((window_end - window_start).total_seconds(), 1.0)
        return max(1, math.ceil(seconds / 86_400.0))

    def _row(self, root: InstrumentId, expiration: date, bar: object) -> dict[str, object] | None:
        """Parse one IBKR ``BarData`` row into a canonical bar dict."""
        timestamp = _bar_timestamp(getattr(bar, "date", None))
        if timestamp is None:
            return None
        open_price = _positive_float(getattr(bar, "open", None))
        high_price = _positive_float(getattr(bar, "high", None))
        low_price = _positive_float(getattr(bar, "low", None))
        close_price = _positive_float(getattr(bar, "close", None))
        if None in (open_price, high_price, low_price, close_price):
            return None
        assert open_price is not None
        assert high_price is not None
        assert low_price is not None
        assert close_price is not None
        if not (
            low_price <= min(open_price, close_price) <= max(open_price, close_price) <= high_price
        ):
            logger.debug("ignoring inconsistent IBKR OHLC bar for %s at %s", root, timestamp)
            return None
        volume = _finite_float(getattr(bar, "volume", None)) or 0.0
        settlement = _positive_float(getattr(bar, "average", None))
        return {
            s.TIMESTAMP_COLUMN: timestamp,
            s.INSTRUMENT_ID_COLUMN: root,
            s.ROOT_SYMBOL_COLUMN: root,
            s.EXPIRATION_COLUMN: expiration,
            s.OPEN_COLUMN: open_price,
            s.HIGH_COLUMN: high_price,
            s.LOW_COLUMN: low_price,
            s.CLOSE_COLUMN: close_price,
            s.VOLUME_COLUMN: max(volume, 0.0),
            s.OPEN_INTEREST_COLUMN: 0.0,
            s.SETTLEMENT_COLUMN: settlement,
        }

    # -- fetching --------------------------------------------------------------

    async def fetch_bars(
        self,
        roots: Sequence[InstrumentId],
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """Fetch canonical daily bars for ``roots`` in ``[start, end]`` from IBKR.

        Raises
        ------
        RuntimeError
            When TWS/IB Gateway is unreachable or IBKR returns no bars; the
            message names the environment/port configuration so an operator can
            fix it (and the hydration chain can fall back).
        """
        normalised = validate_roots(tuple(roots))
        if end <= start:
            raise ValueError(
                f"IBKR historical window must be increasing, got [{start.isoformat()}, "
                f"{end.isoformat()}]"
            )
        client = self._create_client()
        connected_here = False
        rows: list[dict[str, object]] = []
        try:
            if not client.isConnected():
                try:
                    await client.connectAsync(
                        self._settings.ibkr_host,
                        self._settings.ibkr_port,
                        self._settings.ibkr_client_id,
                        self._settings.ibkr_connect_timeout_seconds,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        "IBKR historical data unavailable: cannot connect to TWS/IB Gateway at "
                        f"{self._settings.ibkr_host}:{self._settings.ibkr_port} "
                        f"({type(exc).__name__}: {exc}). Start TWS or IB Gateway and set "
                        "FUTURESRL_ENVIRONMENT=paper|live plus FUTURESRL_IBKR_PORT "
                        "(paper 7497/4002, live 7496/4001)."
                    ) from exc
                connected_here = True
            for root in normalised:
                contract = await self._qualify(
                    client, self._contract_for(root, reference=self._clock())
                )
                expiration = self._expiration_for(root, start)
                for window in self._windows(start, end):
                    bars = await client.reqHistoricalDataAsync(
                        contract,
                        window[1],
                        f"{self._duration_days(window)} D",
                        self._bar_size,
                        self._what_to_show,
                        self._use_rth,
                        _IBKR_FORMAT_DATE_UTC,
                        False,
                    )
                    for bar in bars:
                        row = self._row(root, expiration, bar)
                        if row is not None:
                            rows.append(row)
        finally:
            if connected_here:
                with contextlib.suppress(Exception):
                    client.disconnect()
        if not rows:
            raise RuntimeError(
                "IBKR returned no daily historical bars for roots "
                f"{list(normalised)!r} between {start.isoformat()} and {end.isoformat()}; "
                "check that TWS/IB Gateway is running, FUTURESRL_ENVIRONMENT is paper|live and "
                "FUTURESRL_IBKR_PORT matches it, and that the account is entitled to the "
                "requested futures history"
            )
        frame = pl.DataFrame(rows)
        validated = s.ensure_bars_schema(frame, context="ibkr-historical")
        return s.dedupe_frame(validated, s.BAR_KEY_COLUMNS, keep="last")

    def fetch_bars_sync(
        self,
        roots: Sequence[InstrumentId],
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """Synchronous :data:`BackfillFn` wrapper around :meth:`fetch_bars`.

        Runs the async fetch directly when no event loop is active, otherwise
        bridges it through a dedicated worker thread (hydration is invoked
        synchronously from inside the async runner loop, where nesting
        ``asyncio.run`` is not allowed).
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.fetch_bars(roots, start, end))
        outcome: list[pl.DataFrame] = []
        failures: list[BaseException] = []

        def _worker() -> None:
            try:
                outcome.append(asyncio.run(self.fetch_bars(roots, start, end)))
            except BaseException as exc:
                failures.append(exc)

        worker = threading.Thread(target=_worker, name="futuresrl-ibkr-history", daemon=True)
        worker.start()
        worker.join()
        if failures:
            raise failures[0]
        if not outcome:  # pragma: no cover - the worker always appends or fails
            raise RuntimeError("IBKR historical fetch produced no result")
        return outcome[0]
