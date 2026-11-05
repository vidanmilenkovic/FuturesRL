"""Asynchronous IBKR connection/state manager for the FuturesRL execution engine.

This module is the *only* place in ``src/execution`` that knows about the IBKR
wire protocol.  It is deliberately split into three layers:

``Venue*`` dataclasses / ``IBClientLike`` protocol
    Pure-Python, zero-dependency transport contract.  Unit tests inject a fake
    client implementing :class:`IBClientLike`; production code never needs the
    IBKR SDK to be importable in order to be type-checked or tested.

:class:`IBKRConnectionManager`
    Owns the async lifecycle: connect/disconnect under a lock, heartbeat
    monitoring, exponential backoff with jitter for reconnects and the IBKR
    nightly server-reset window (23:45-00:45 New York time by default, fully
    configurable).  It also exposes the account/position/contract/order
    primitives the OMS, roller and reconciler consume.

``_IbAsyncIBClient``
    The concrete ``ib_async`` adapter.  ``ib_async`` is imported lazily inside
    the constructor only, so importing :mod:`src.execution` never requires the
    optional live-trading dependency and unit tests stay completely offline.

Design rules
------------
* One ``asyncio.Lock`` protects every connection-state transition, so two
  concurrent rebalances can never race a reconnect.
* Every blocking wait goes through an injected ``sleep`` callable and every
  timestamp through an injected ``clock`` callable; tests are deterministic.
* The nightly reset window is computed in ``America/New_York`` (DST aware),
  never in the host timezone.
* Margin figures are parsed from IBKR's sentinel-laden ``OrderState`` strings
  and reported as :class:`MarginImpact`; missing margin data is ``None``, never
  silently zero (the pre-trade gate treats ``None`` as a hard rejection).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Any, Protocol, runtime_checkable
from zoneinfo import ZoneInfo

from configs.instruments import InstrumentsConfig, InstrumentSpec
from configs.settings import Settings

from src.core.types import (
    AccountSnapshot,
    InstrumentId,
    OrderEvent,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    ensure_utc,
)

logger = logging.getLogger(__name__)

#: Timezone used for every IBKR scheduled-maintenance / session computation.
NEW_YORK: ZoneInfo = ZoneInfo("America/New_York")

#: Callable resolving the latest known mark price for a futures root.
type PriceProvider = Callable[[InstrumentId], float | None]

#: Injectable wall clock (timezone-aware UTC expected).
type Clock = Callable[[], datetime]

#: Injectable async sleep.
type Sleeper = Callable[[float], Awaitable[None]]

#: IBKR's "not set" sentinel for float fields.
_IBKR_UNSET: float = 1.7976931348623157e308

_FRIDAY: int = 4


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class IBKRError(RuntimeError):
    """Base class for every IBKR execution-layer failure."""


class IBKRConnectionError(IBKRError):
    """The TWS / IB Gateway session could not be established."""


class IBKRDataError(IBKRError):
    """IBKR returned state that cannot be mapped onto the core contracts."""


class IBKRBagUnsupportedError(IBKRError):
    """The venue rejected a native ``BAG`` calendar-spread order."""


# ---------------------------------------------------------------------------
# Pure venue transport contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VenueAccountValue:
    """One ``accountValues()`` row, string-typed exactly as IBKR sends it."""

    tag: str
    value: str
    currency: str = ""
    account: str = ""


@dataclass(frozen=True, slots=True)
class VenuePosition:
    """One ``positions()`` row normalised to primitive Python types."""

    root_symbol: str
    quantity: int
    avg_cost: float
    multiplier: float | None = None
    expiration: date | None = None
    local_symbol: str | None = None


@dataclass(frozen=True, slots=True)
class ContractRequest:
    """Request to resolve one abstract futures contract against IBKR."""

    root: str
    last_trade_date_or_contract_month: str
    exchange: str
    multiplier: float
    expiration: date | None = None
    currency: str = "USD"

    def __post_init__(self) -> None:
        if not self.root:
            raise ValueError("root must be non-empty")
        if not self.last_trade_date_or_contract_month:
            raise ValueError("last_trade_date_or_contract_month must be non-empty")
        if not self.exchange:
            raise ValueError("exchange must be non-empty")
        if self.multiplier <= 0:
            raise ValueError("multiplier must be strictly positive")


@dataclass(frozen=True, slots=True)
class VenueContract:
    """A qualified IBKR contract (``conId`` resolved)."""

    root: str
    con_id: int
    exchange: str
    last_trade_date_or_contract_month: str
    multiplier: float
    expiration: date | None = None
    local_symbol: str | None = None
    currency: str = "USD"

    def __post_init__(self) -> None:
        if not self.root:
            raise ValueError("root must be non-empty")
        if self.con_id <= 0:
            raise ValueError(f"conId must be strictly positive, got {self.con_id}")
        if self.multiplier <= 0:
            raise ValueError("multiplier must be strictly positive")


@dataclass(frozen=True, slots=True)
class MarginImpact:
    """IBKR ``whatIfOrder`` margin projection for one hypothetical order."""

    initial_margin_before: float | None = None
    maintenance_margin_before: float | None = None
    initial_margin_change: float | None = None
    maintenance_margin_change: float | None = None
    initial_margin_after: float | None = None
    maintenance_margin_after: float | None = None
    warning: str = ""

    @property
    def projected_initial_margin(self) -> float | None:
        """Post-trade initial margin, or ``None`` when IBKR did not report it."""
        if self.initial_margin_after is not None:
            return self.initial_margin_after
        if self.initial_margin_before is not None and self.initial_margin_change is not None:
            return self.initial_margin_before + self.initial_margin_change
        return None

    @property
    def projected_maintenance_margin(self) -> float | None:
        """Post-trade maintenance margin, or ``None`` when unreported."""
        if self.maintenance_margin_after is not None:
            return self.maintenance_margin_after
        if (
            self.maintenance_margin_before is not None
            and self.maintenance_margin_change is not None
        ):
            return self.maintenance_margin_before + self.maintenance_margin_change
        return None

    @property
    def is_complete(self) -> bool:
        """Whether both projected initial *and* maintenance margin are known."""
        return self.projected_initial_margin is not None and (
            self.projected_maintenance_margin is not None
        )

    def utilization(self, net_liquidation: float) -> tuple[float | None, float | None]:
        """Return ``(initial, maintenance)`` projected utilisation fractions."""
        if net_liquidation <= 0:
            return (None, None)
        initial = self.projected_initial_margin
        maintenance = self.projected_maintenance_margin
        return (
            None if initial is None else initial / net_liquidation,
            None if maintenance is None else maintenance / net_liquidation,
        )


@dataclass(frozen=True, slots=True)
class VenueFill:
    """One normalised execution report."""

    order_ref: str
    side: OrderSide
    quantity: int
    price: float
    timestamp: datetime | None = None
    commission: float = 0.0


@dataclass(frozen=True, slots=True)
class VenueTrade:
    """Normalised snapshot of one IBKR ``Trade`` (order + status + fills)."""

    order_ref: str
    status: OrderStatus
    filled_quantity: int = 0
    average_price: float | None = None
    commission: float = 0.0
    message: str = ""
    fills: tuple[VenueFill, ...] = ()

    @property
    def is_terminal(self) -> bool:
        """Whether the order reached a terminal lifecycle state."""
        return self.status in (
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
        )


@dataclass(frozen=True, slots=True)
class VenueOrderSnapshot:
    """One open-order row used by ``cancel_all`` and telemetry."""

    order_ref: str
    status: OrderStatus
    instrument_id: InstrumentId
    side: OrderSide
    quantity: int
    filled_quantity: int = 0


# ---------------------------------------------------------------------------
# IBKR order wire mapping (pure data; the live adapter turns it into an Order)
# ---------------------------------------------------------------------------

#: IBKR algo strategy plus default algo parameters for each supported algo.
_ALGO_STRATEGIES: Mapping[OrderType, tuple[str, tuple[tuple[str, str], ...]]] = {
    OrderType.ADAPTIVE: ("Adaptive", (("adaptivePriority", "Normal"),)),
    OrderType.TWAP: ("TWAP", (("strategyType", "TWAP"),)),
    OrderType.VWAP: ("VWAP", (("strategyType", "VWAP"),)),
}

#: Core order type -> IBKR ``orderType`` for plain (non-algo) orders.
_PLAIN_ORDER_TYPES: Mapping[OrderType, str] = {
    OrderType.MARKET: "MKT",
    OrderType.LIMIT: "LMT",
    OrderType.STOP: "STP",
    OrderType.STOP_LIMIT: "STP LMT",
}


@dataclass(frozen=True, slots=True)
class IBOrderRequest:
    """SDK-free description of one IBKR order.

    The live adapter maps this onto ``ib_async.Order``; unit tests inspect the
    dataclass directly and never need the IBKR SDK.
    """

    order_ref: str
    action: str
    quantity: int
    order_type: str
    time_in_force: str = "DAY"
    limit_price: float | None = None
    stop_price: float | None = None
    algo_strategy: str | None = None
    algo_params: tuple[tuple[str, str], ...] = ()
    transmit: bool = True
    good_till_date: str | None = None

    def __post_init__(self) -> None:
        if not self.order_ref:
            raise ValueError("order_ref must be non-empty")
        if self.action not in ("BUY", "SELL"):
            raise ValueError(f"action must be BUY or SELL, got {self.action!r}")
        if self.quantity <= 0:
            raise ValueError("quantity must be strictly positive")
        if not self.order_type:
            raise ValueError("order_type must be non-empty")
        if self.limit_price is not None and self.limit_price <= 0:
            raise ValueError("limit_price must be strictly positive")
        if self.stop_price is not None and self.stop_price <= 0:
            raise ValueError("stop_price must be strictly positive")

    @property
    def is_algo(self) -> bool:
        """Whether this request targets an IBKR algorithmic order."""
        return self.algo_strategy is not None

    @classmethod
    def from_order_event(cls, order: OrderEvent) -> IBOrderRequest:
        """Translate a core :class:`~src.core.types.OrderEvent` to an IBKR order.

        ``MKT``/``LMT``/``STP``/``STP LMT`` map directly; ``IBALGO`` becomes an
        Adaptive algo order and ``TWAP``/``VWAP`` become the corresponding IB
        algo strategies.  ``BAG`` spreads are contract-level and must be built
        by :class:`~src.execution.ibkr_roller.ContractRoller`, so an
        ``OrderEvent`` of that type is rejected here instead of being silently
        mis-transmitted.
        """
        if order.order_type is OrderType.CALENDAR_SPREAD:
            raise IBKRDataError(
                "CALENDAR_SPREAD orders are BAG combo orders and must be submitted via "
                "ContractRoller.execute_roll(), not OrderManagementSystem.submit_order()"
            )
        limit_price = order.limit_price
        algo_strategy = None
        algo_params: tuple[tuple[str, str], ...] = ()
        configured = _ALGO_STRATEGIES.get(order.order_type)
        if configured is not None:
            algo_strategy, algo_params = configured
            order_type = "LMT" if limit_price is not None else "MKT"
        else:
            order_type = _PLAIN_ORDER_TYPES[order.order_type]
        return cls(
            order_ref=order.order_id,
            action=order.side.value,
            quantity=order.quantity,
            order_type=order_type,
            time_in_force=order.time_in_force.value,
            limit_price=limit_price,
            stop_price=order.stop_price,
            algo_strategy=algo_strategy,
            algo_params=algo_params,
        )


@dataclass(frozen=True, slots=True)
class BagLeg:
    """One leg of a native IBKR calendar-spread ``BAG`` combo."""

    con_id: int
    action: OrderSide
    ratio: int = 1
    exchange: str = ""

    def __post_init__(self) -> None:
        if self.con_id <= 0:
            raise ValueError("con_id must be strictly positive")
        if self.ratio <= 0:
            raise ValueError("ratio must be strictly positive")
        if not self.exchange:
            raise ValueError("exchange must be non-empty")


@dataclass(frozen=True, slots=True)
class BagOrderSpec:
    """SDK-free description of a native calendar-spread combo order."""

    root: str
    exchange: str
    legs: tuple[BagLeg, ...]
    order: IBOrderRequest
    currency: str = "USD"

    def __post_init__(self) -> None:
        if not self.root:
            raise ValueError("root must be non-empty")
        if not self.exchange:
            raise ValueError("exchange must be non-empty")
        if len(self.legs) < 2:
            raise ValueError("a BAG calendar spread requires at least two legs")
        if any(leg.con_id <= 0 for leg in self.legs):
            raise ValueError("every BAG leg requires a resolved conId")
        if abs(sum(leg.ratio for leg in self.legs)) == 0:
            raise ValueError("BAG leg ratios must not sum to zero")
        if self.order.order_type not in ("MKT", "LMT"):
            raise ValueError("BAG orders must be priced as MKT or LMT at the order level")

    @property
    def ratio_balanced(self) -> bool:
        """Whether the combo is ratio-balanced (1:1 calendar spread)."""
        buys = sum(leg.ratio for leg in self.legs if leg.action is OrderSide.BUY)
        sells = sum(leg.ratio for leg in self.legs if leg.action is OrderSide.SELL)
        return buys == sells


# ---------------------------------------------------------------------------
# Transport protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class IBClientLike(Protocol):
    """The narrow, SDK-free IBKR transport used by the execution layer.

    A fake implementation is a handful of async methods returning the
    ``Venue*`` dataclasses; the live implementation is ``_IbAsyncIBClient``.
    """

    @property
    def is_connected(self) -> bool:
        """Whether the underlying TWS/IB Gateway session is alive."""
        ...

    async def connect(self, *, host: str, port: int, client_id: int, timeout: float) -> None:
        """Open the gateway session."""
        ...

    async def disconnect(self) -> None:
        """Close the gateway session."""
        ...

    async def account_values(self) -> Sequence[VenueAccountValue]:
        """Return the cached account values."""
        ...

    async def positions(self) -> Sequence[VenuePosition]:
        """Return the cached account positions."""
        ...

    async def qualify_contract(self, request: ContractRequest) -> VenueContract:
        """Resolve an abstract contract request into a qualified ``conId``."""
        ...

    async def what_if_order(self, contract: VenueContract, order: IBOrderRequest) -> MarginImpact:
        """Run IBKR ``whatIfOrder`` for a hypothetical order."""
        ...

    async def place_order(self, contract: VenueContract, order: IBOrderRequest) -> VenueTrade:
        """Transmit a single (non-bracket) order."""
        ...

    async def place_bracket(
        self,
        contract: VenueContract,
        parent: IBOrderRequest,
        child: IBOrderRequest,
    ) -> tuple[VenueTrade, VenueTrade]:
        """Transmit an entry order with a native exchange stop-loss child."""
        ...

    async def place_bag_order(self, bag: BagOrderSpec) -> VenueTrade:
        """Transmit a native ``BAG`` calendar-spread combo order."""
        ...

    async def open_orders(self) -> Sequence[VenueOrderSnapshot]:
        """Return every currently working order."""
        ...

    async def cancel_order(self, order_ref: str) -> bool:
        """Cancel one order by our client order id; ``False`` when not found."""
        ...

    async def get_trade(self, order_ref: str) -> VenueTrade | None:
        """Return the latest normalised state of one order."""
        ...


# ---------------------------------------------------------------------------
# Contract-month calendar helpers (shared with the roller)
# ---------------------------------------------------------------------------


def third_friday(year: int, month: int) -> date:
    """Return the third Friday of ``month`` (standard CME contract expiry)."""
    first = date(year, month, 1)
    offset = (_FRIDAY - first.weekday()) % 7
    return first + timedelta(days=offset + 14)


def contract_month_string(expiration: date) -> str:
    """Format a date as IBKR's six-digit ``YYYYMM`` contract month."""
    return f"{expiration.year:04d}{expiration.month:02d}"


def normalize_expiration(expiration: date | str | None) -> tuple[date | None, str | None]:
    """Normalise a date/``YYYYMM``/``YYYYMMDD`` expiry into ``(date, YYYYMM)``."""
    if expiration is None:
        return (None, None)
    if isinstance(expiration, datetime):
        raise TypeError("expiration must be a date, not a datetime")
    if isinstance(expiration, date):
        return (expiration, contract_month_string(expiration))
    raw = expiration.strip()
    if not raw:
        return (None, None)
    if raw.isdigit() and len(raw) == 6:
        year, month = int(raw[:4]), int(raw[4:])
        parsed = date(year, month, 1)
        return (parsed, contract_month_string(parsed))
    if raw.isdigit() and len(raw) == 8:
        parsed = date(int(raw[:4]), int(raw[4:6]), int(raw[6:]))
        return (parsed, contract_month_string(parsed))
    try:
        parsed = date.fromisoformat(raw)
    except ValueError as exc:
        raise IBKRDataError(f"cannot parse expiration {expiration!r}") from exc
    return (parsed, contract_month_string(parsed))


def _month_candidates(spec: InstrumentSpec, as_of: date, count: int) -> tuple[date, ...]:
    """Return the next ``count`` listed contract expiries after ``as_of``."""
    allowed = set(spec.roll.contract_months)
    year, month = as_of.year, as_of.month
    found: list[date] = []
    guard = 0
    while len(found) < count:
        guard += 1
        if guard > 240:  # pragma: no cover - defensive: 20 years of empty months
            raise IBKRDataError(f"no contract months listed for {spec.root}")
        if month in allowed:
            expiry = third_friday(year, month)
            if expiry >= as_of:
                found.append(expiry)
        month += 1
        if month > 12:
            month = 1
            year += 1
    return tuple(found)


def upcoming_contract_expirations(
    spec: InstrumentSpec, as_of: date, *, count: int = 2
) -> tuple[date, ...]:
    """The next ``count`` contract expiries implied by the roll rules."""
    return _month_candidates(spec, as_of, count)


def active_contract_expiration(spec: InstrumentSpec, as_of: date) -> date:
    """Front-month expiry: the first contract still outside its roll window."""
    days = spec.roll.days_before_fnd
    for expiry in _month_candidates(spec, as_of, count=8):
        if expiry - timedelta(days=days) > as_of:
            return expiry
    raise IBKRDataError(f"no active contract found for {spec.root}")  # pragma: no cover


# ---------------------------------------------------------------------------
# Connection manager
# ---------------------------------------------------------------------------


def _no_price(_root: InstrumentId) -> float | None:
    return None


_DEFAULT_PRICE_PROVIDER: PriceProvider = _no_price


def _utc_now() -> datetime:
    return datetime.now(UTC)


async def _asyncio_sleep(seconds: float) -> None:
    if seconds > 0:
        await asyncio.sleep(seconds)


class IBKRConnectionManager:
    """Async lifecycle + state manager for the IBKR execution engine.

    The manager guarantees that every public coroutine either talks to a live
    session or raises :class:`IBKRConnectionError`; it never silently degrades.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        instrument_master: InstrumentsConfig | None = None,
        client_factory: Callable[[], IBClientLike] | None = None,
        clock: Clock | None = None,
        sleep: Sleeper | None = None,
        random_source: Callable[[], float] | None = None,
        nightly_reset_start: time | None = None,
        nightly_reset_end: time | None = None,
        price_provider: PriceProvider | None = None,
        what_if_provider: Callable[[OrderEvent], Awaitable[MarginImpact]] | None = None,
        reconnect_jitter: float = 0.1,
        heartbeat_enabled: bool = True,
    ) -> None:
        if not 0.0 <= reconnect_jitter < 1.0:
            raise ValueError("reconnect_jitter must be within [0, 1)")
        self._settings = settings
        self._instruments = (
            instrument_master if instrument_master is not None else settings.instruments
        )
        self._client_factory = client_factory
        self._clock: Clock = clock if clock is not None else _utc_now
        self._sleep: Sleeper = sleep if sleep is not None else _asyncio_sleep
        self._random = random_source if random_source is not None else _random01
        self._reset_start = nightly_reset_start
        self._reset_end = nightly_reset_end
        self._default_reset_start = time(self._settings.ibkr_nightly_reset_start_hour_et % 24, 45)
        self._default_reset_end = time(self._settings.ibkr_nightly_reset_end_hour_et % 24, 45)
        self._price_provider: PriceProvider = (
            price_provider if price_provider is not None else _DEFAULT_PRICE_PROVIDER
        )
        self._what_if_provider = what_if_provider
        self._jitter = reconnect_jitter
        self._heartbeat_enabled = heartbeat_enabled
        self._lock = asyncio.Lock()
        self._client: IBClientLike | None = None
        self._connected = False
        self._closing = False
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._stale_clients: list[IBClientLike] = []
        self._contract_cache: dict[tuple[str, str], VenueContract] = {}
        self._connection_attempts = 0
        self._reconnects = 0
        self._last_error: str | None = None
        self._last_heartbeat: datetime | None = None
        # TWS can keep a stale "accepted" session for a client id after a TWS
        # restart, rejecting/accepting reconnects without streaming account data.
        # Rotating the client id on every connect/attempt guarantees a clean
        # subscription while keeping the configured id as the base.
        self._client_id_offset = 0
        self._active_client_id: int | None = None
        self._reconnecting: bool = False

    # -- introspection ---------------------------------------------------------

    @property
    def name(self) -> str:
        """Broker identifier used by telemetry."""
        return "ibkr"

    @property
    def is_connected(self) -> bool:
        """Whether a live session is currently established."""
        client = self._client
        return bool(self._connected and client is not None and client.is_connected)

    @property
    def connection_attempts(self) -> int:
        """Number of connect attempts performed so far (monotonic)."""
        return self._connection_attempts

    @property
    def reconnect_count(self) -> int:
        """Number of heartbeat-triggered reconnects so far."""
        return self._reconnects

    @property
    def active_client_id(self) -> int | None:
        """IBKR client id currently in use (rotated across reconnects)."""
        return self._active_client_id

    def _next_client_id(self) -> int:
        """Return the next client id, rotating within a bounded window."""
        client_id = self._settings.ibkr_client_id + (self._client_id_offset % 8)
        self._client_id_offset += 1
        return client_id

    @property
    def last_error(self) -> str | None:
        """Human-readable description of the last connection failure."""
        return self._last_error

    @property
    def last_heartbeat_at(self) -> datetime | None:
        """Timestamp of the last successful heartbeat."""
        return self._last_heartbeat

    @property
    def is_reconnecting(self) -> bool:
        """Whether a reconnect attempt is currently in flight."""
        return self._reconnecting

    @property
    def heartbeat_running(self) -> bool:
        """Whether the background heartbeat/liveness task is alive."""
        task = self._heartbeat_task
        return task is not None and not task.done()

    @property
    def instruments(self) -> InstrumentsConfig:
        """The instrument master backing contract resolution."""
        return self._instruments

    @property
    def client(self) -> IBClientLike:
        """The connected client, raising :class:`IBKRConnectionError` if absent."""
        return self._require_client()

    @property
    def price_provider(self) -> PriceProvider:
        """Mark-price provider used to value positions and fat-finger checks."""
        return self._price_provider

    def set_price_provider(self, provider: PriceProvider) -> None:
        """Install the mark-price provider (e.g. from the market-data layer)."""
        self._price_provider = provider

    def set_what_if_provider(
        self, provider: Callable[[OrderEvent], Awaitable[MarginImpact]] | None
    ) -> None:
        """Override the margin simulation callable (tests / custom venues)."""
        self._what_if_provider = provider

    def _require_client(self) -> IBClientLike:
        client = self._client
        if client is None or not self._connected:
            raise IBKRConnectionError("IBKR connection is not established")
        return client

    # -- lifecycle -------------------------------------------------------------

    async def __aenter__(self) -> IBKRConnectionManager:
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        del exc_type, exc, traceback
        await self.disconnect()

    async def connect(self, *, max_attempts: int | None = None) -> None:
        """Connect to TWS / IB Gateway, retrying with exponential backoff.

        ``max_attempts`` overrides ``settings.ibkr_reconnect_max_attempts`` for a
        single bounded probe (request-path / background liveness reconnects must
        never block the event loop for the full production retry budget).
        """
        if self.is_connected:
            return
        async with self._lock:
            await self._connect_locked(max_attempts=max_attempts)

    async def ensure_connected(self, *, max_attempts: int = 1) -> bool:
        """Best-effort reconnect that never raises.

        Returns ``True`` when a live session is (re)established. Inside the IBKR
        nightly reset window the probe is skipped immediately (the venue is
        intentionally unreachable) instead of sleeping until the window closes.
        """
        if self.is_connected:
            return True
        if self.in_nightly_reset_window():
            self._last_error = "inside the IBKR nightly reset window"
            return False
        try:
            await self.connect(max_attempts=max_attempts)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            return False
        return self.is_connected

    def invalidate(self, reason: str) -> None:
        """Mark the current transport as dead after a request-path failure.

        Drops the client reference and the connected flag so the heartbeat /
        liveness probe re-establishes a fresh session. The wedged client is queued
        for best-effort disposal on the next reconnect/heartbeat tick: the manager
        never blocks the caller waiting for a stuck socket teardown.
        """
        self._last_error = reason or "IBKR session invalidated"
        self._connected = False
        client = self._client
        self._client = None
        if client is not None:
            self._stale_clients.append(client)

    async def _dispose_stale_clients(self) -> None:
        clients, self._stale_clients = self._stale_clients, []
        for client in clients:
            try:
                await client.disconnect()
            except Exception:  # pragma: no cover - best-effort socket teardown
                logger.debug("failed to dispose a stale IBKR client", exc_info=True)

    def ensure_heartbeat(self) -> bool:
        """Restart the heartbeat task when a live session lost its monitor."""
        if not self._connected or self._closing or self.heartbeat_running:
            return False
        self._start_heartbeat()
        return self.heartbeat_running

    async def disconnect(self) -> None:
        """Cancel the heartbeat and tear down the gateway session."""
        async with self._lock:
            await self._disconnect_locked()

    async def _disconnect_locked(self) -> None:
        self._closing = True
        task = self._heartbeat_task
        self._heartbeat_task = None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        client = self._client
        self._client = None
        self._connected = False
        self._contract_cache.clear()
        await self._dispose_stale_clients()
        if client is not None:
            try:
                await client.disconnect()
            except Exception:  # pragma: no cover - best-effort teardown
                logger.exception("error while disconnecting from IBKR")
        self._closing = False

    def _make_client(self) -> IBClientLike:
        factory = self._client_factory
        if factory is not None:
            return factory()
        return _IbAsyncIBClient()

    async def _connect_locked(self, *, max_attempts: int | None = None) -> None:
        if self.is_connected:
            return
        self._closing = False
        self._reconnecting = True
        await self._dispose_stale_clients()
        try:
            settings = self._settings
            attempts = max(
                1,
                settings.ibkr_reconnect_max_attempts if max_attempts is None else max_attempts,
            )
            last_exc: BaseException | None = None
            for attempt in range(1, attempts + 1):
                if self.in_nightly_reset_window():
                    await self._wait_for_reset_window()
                self._connection_attempts += 1
                client = self._client if self._client is not None else self._make_client()
                client_id = self._next_client_id()
                try:
                    await client.connect(
                        host=settings.ibkr_host,
                        port=settings.ibkr_port,
                        client_id=client_id,
                        timeout=settings.ibkr_connect_timeout_seconds,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    last_exc = exc
                    self._last_error = f"{type(exc).__name__}: {exc}"
                    self._client = None
                    self._connected = False
                    try:
                        await client.disconnect()
                    except Exception:  # pragma: no cover - best-effort teardown
                        logger.debug("failed to dispose a failed IBKR client", exc_info=True)
                    logger.warning(
                        "IBKR connection attempt %d/%d failed: %s",
                        attempt,
                        attempts,
                        self._last_error,
                    )
                    if attempt >= attempts:
                        break
                    await self._sleep(self._backoff_delay(attempt))
                    continue
                self._client = client
                self._connected = True
                self._active_client_id = client_id
                self._last_error = None
                self._last_heartbeat = self._clock()
                self._start_heartbeat()
                logger.info(
                    "connected to IBKR at %s:%d (clientId=%d)",
                    settings.ibkr_host,
                    settings.ibkr_port,
                    client_id,
                )
                return
            raise IBKRConnectionError(
                f"could not connect to IBKR at {settings.ibkr_host}:{settings.ibkr_port} after "
                f"{attempts} attempts (last error: {self._last_error})"
            ) from last_exc
        finally:
            self._reconnecting = False

    def _backoff_delay(self, attempt: int) -> float:
        """Exponential backoff with symmetric jitter, capped by settings."""
        settings = self._settings
        base = settings.ibkr_reconnect_initial_delay_seconds
        cap = settings.ibkr_reconnect_max_delay_seconds
        exponential = min(cap, base * (2.0 ** (attempt - 1)))
        factor = 1.0 - self._jitter + 2.0 * self._jitter * self._random()
        return max(0.0, exponential * factor)

    # -- nightly IBKR server reset window -------------------------------------

    @property
    def nightly_reset_start(self) -> time:
        """Start of the IBKR nightly reset window (New York local time)."""
        return self._reset_start if self._reset_start is not None else self._default_reset_start

    @property
    def nightly_reset_end(self) -> time:
        """End of the IBKR nightly reset window (New York local time)."""
        return self._reset_end if self._reset_end is not None else self._default_reset_end

    def in_nightly_reset_window(self, now: datetime | None = None) -> bool:
        """Whether ``now`` falls inside the IBKR nightly server-reset window."""
        moment = ensure_utc(now if now is not None else self._clock())
        local = moment.astimezone(NEW_YORK)
        current = local.time().replace(tzinfo=None)
        start = self.nightly_reset_start
        end = self.nightly_reset_end
        if start == end:
            return False
        if start < end:
            return start <= current < end
        return current >= start or current < end

    def seconds_until_reset_window_end(self, now: datetime | None = None) -> float:
        """Seconds from ``now`` until the nightly reset window closes."""
        moment = ensure_utc(now if now is not None else self._clock())
        if not self.in_nightly_reset_window(moment):
            return 0.0
        local = moment.astimezone(NEW_YORK)
        start = self.nightly_reset_start
        end = self.nightly_reset_end
        current = local.time().replace(tzinfo=None)
        end_day = local.date()
        if start >= end and current >= start:
            end_day = end_day + timedelta(days=1)
        end_at = datetime.combine(end_day, end, tzinfo=NEW_YORK)
        return max(0.0, (end_at - local).total_seconds())

    async def _wait_for_reset_window(self) -> None:
        delay = self.seconds_until_reset_window_end()
        logger.warning(
            "inside IBKR nightly reset window (%.0fs remaining); delaying connection", delay
        )
        await self._sleep(delay)

    # -- heartbeat / reconnect ------------------------------------------------

    def _start_heartbeat(self) -> None:
        if not self._heartbeat_enabled:
            return
        task = self._heartbeat_task
        if task is not None and not task.done():
            return
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop(), name="ibkr-heartbeat")

    async def _heartbeat_loop(self) -> None:
        interval = self._settings.ibkr_heartbeat_interval_seconds
        try:
            while not self._closing:
                await self._sleep(interval)
                try:
                    await self.heartbeat_tick()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("IBKR heartbeat reconnection failed")
        except asyncio.CancelledError:
            raise
        finally:
            self._heartbeat_task = None

    async def heartbeat_tick(self) -> bool:
        """Verify liveness once, reconnecting (with backoff) when it is lost."""
        await self._dispose_stale_clients()
        client = self._client
        if client is not None and client.is_connected:
            self._last_heartbeat = self._clock()
            return True
        logger.warning("IBKR heartbeat lost; reconnecting")
        async with self._lock:
            if self.is_connected:
                self._last_heartbeat = self._clock()
                return True
            self._reconnects += 1
            # One bounded attempt per heartbeat tick: a dead TWS keeps the event
            # loop responsive and the session is retried on the next interval
            # instead of sleeping through the full production backoff budget.
            await self._connect_locked(max_attempts=1)
        return self.is_connected

    # -- account / positions ---------------------------------------------------

    async def fetch_account(self) -> AccountSnapshot:
        """Fetch the current account valuation and margin consumption."""
        client = self._require_client()
        rows = await client.account_values()
        # IBKR reports the ledger in the account's base currency (EUR on many
        # European paper accounts) plus optional per-currency rows. Accept the
        # configured account currency, USD and the currency-less rows, preferring
        # the configured base currency, then USD.
        base_currency = self._settings.account_currency.upper()
        per_currency: dict[str, dict[str, float]] = {}
        for row in rows:
            parsed = _optional_float(row.value)
            if parsed is not None:
                per_currency.setdefault((row.currency or "").upper(), {})[row.tag] = parsed
        # Prefer the configured base currency, then USD/currency-less rows. IBKR
        # reports the ledger in the account's *actual* base currency (e.g. EUR),
        # so if the configuration disagrees fall back to whichever currency has
        # a NetLiquidation row instead of returning a 503.
        preferred = (
            base_currency if "NetLiquidation" in per_currency.get(base_currency, {}) else None
        )
        if preferred is None:
            preferred = next(
                (
                    currency
                    for currency in ("USD", "")
                    if "NetLiquidation" in per_currency.get(currency, {})
                ),
                None,
            )
        if preferred is None:
            preferred = next(
                (
                    currency
                    for currency, values in per_currency.items()
                    if "NetLiquidation" in values
                ),
                None,
            )
        if preferred is not None and preferred != base_currency:
            logger.warning(
                "IBKR reports its account ledger in %s but account_currency=%s; "
                "using the broker's ledger currency. Set FUTURESRL_ACCOUNT_CURRENCY=%s "
                "and FUTURESRL_FX_RATES so USD notional sizing stays correct.",
                preferred or "(default)",
                base_currency,
                preferred or "USD",
            )
        tags: dict[str, float] = dict(per_currency.get(preferred or "", {}))
        net_liquidation = tags.get("NetLiquidation")
        if net_liquidation is None or net_liquidation <= 0.0:
            raise IBKRDataError(
                "IBKR account does not report a positive NetLiquidation value "
                f"(tags received: {sorted(tags)})"
            )
        return AccountSnapshot(
            timestamp=self._clock(),
            net_liquidation=net_liquidation,
            cash_balance=tags.get("TotalCashValue", 0.0),
            buying_power=tags.get("BuyingPower", 0.0),
            initial_margin_used=tags.get("InitMarginReq", 0.0),
            maintenance_margin_used=tags.get("MaintMarginReq", 0.0),
            excess_liquidity=tags.get("ExcessLiquidity", 0.0),
        )

    async def fetch_positions(
        self, prices: Mapping[InstrumentId, float] | None = None
    ) -> Mapping[InstrumentId, Position]:
        """Fetch broker positions as core :class:`~src.core.types.Position` objects.

        IBKR's ``avgCost`` for futures includes the contract multiplier; the
        per-contract average price is therefore ``avgCost / multiplier``.  The
        mark price is taken from ``prices`` first, then from the configured
        price provider, and finally falls back to cost basis when no market
        price is available yet (documented, deterministic fallback).
        """
        client = self._require_client()
        rows = await client.positions()
        overrides = dict(prices or {})
        positions: dict[InstrumentId, Position] = {}
        for row in rows:
            quantity = round(row.quantity)
            if quantity == 0:
                continue
            root = row.root_symbol.strip().upper()
            if not root:
                raise IBKRDataError("IBKR returned a position without a contract symbol")
            spec = self._instruments.get(root) if self._instruments.has(root) else None
            multiplier = spec.multiplier if spec is not None else (row.multiplier or 1.0)
            if multiplier <= 0:
                raise IBKRDataError(f"invalid multiplier for position {root!r}")
            average_price = row.avg_cost / multiplier if row.avg_cost > 0 else 0.0
            mark = overrides.get(root)
            if mark is None:
                mark = self._price_provider(root)
            if mark is None or mark <= 0:
                mark = average_price
            if mark <= 0:
                raise IBKRDataError(f"cannot value position {root!r}: no price available")
            if average_price <= 0:
                average_price = mark
            expiration = row.expiration
            if expiration is None and spec is not None:
                expiration = active_contract_expiration(spec, self._local_date())
            positions[root] = Position(
                instrument_id=root,
                quantity=quantity,
                average_price=average_price,
                market_price=mark,
                multiplier=multiplier,
                expiration=expiration,
                initial_margin=spec.margins.initial if spec is not None else 0.0,
                maintenance_margin=spec.margins.maintenance if spec is not None else 0.0,
                contract_symbol=row.local_symbol,
            )
        return positions

    # -- contract resolution ---------------------------------------------------

    def _local_date(self) -> date:
        return self._clock().astimezone(NEW_YORK).date()

    async def resolve_contract(
        self, root: str, expiration: date | str | None = None
    ) -> VenueContract:
        """Resolve an abstract futures root/expiry into a qualified contract."""
        try:
            spec = self._instruments.get(root)
        except KeyError as exc:
            raise IBKRDataError(str(exc)) from exc
        expiration_date, month_string = normalize_expiration(expiration)
        if month_string is None or expiration_date is None:
            expiration_date = active_contract_expiration(spec, self._local_date())
            month_string = contract_month_string(expiration_date)
        cache_key = (spec.root, month_string)
        cached = self._contract_cache.get(cache_key)
        if cached is not None:
            return cached
        client = self._require_client()
        request = ContractRequest(
            root=spec.root,
            last_trade_date_or_contract_month=month_string,
            exchange=spec.ibkr_exchange,
            multiplier=spec.multiplier,
            expiration=expiration_date,
            currency=spec.currency,
        )
        qualified = await client.qualify_contract(request)
        self._contract_cache[cache_key] = qualified
        return qualified

    # -- margin simulation / order primitives ----------------------------------

    async def what_if_order(self, order: OrderEvent) -> MarginImpact:
        """Run the mandatory pre-trade IBKR ``whatIfOrder`` margin simulation."""
        provider = self._what_if_provider
        if provider is not None:
            return await provider(order)
        contract = await self.resolve_contract(order.instrument_id, order.expiration)
        request = IBOrderRequest.from_order_event(order)
        client = self._require_client()
        return await client.what_if_order(contract, request)

    async def place_order(self, contract: VenueContract, order: IBOrderRequest) -> VenueTrade:
        """Transmit one single-leg order through the connected session."""
        client = self._require_client()
        return await client.place_order(contract, order)

    async def place_bracket(
        self, contract: VenueContract, parent: IBOrderRequest, child: IBOrderRequest
    ) -> tuple[VenueTrade, VenueTrade]:
        """Transmit an entry + native stop-loss child bracket."""
        client = self._require_client()
        return await client.place_bracket(contract, parent, child)

    async def place_bag_order(self, bag: BagOrderSpec) -> VenueTrade:
        """Transmit a native ``BAG`` calendar-spread combo order."""
        client = self._require_client()
        return await client.place_bag_order(bag)

    async def open_orders(self) -> Sequence[VenueOrderSnapshot]:
        """Return every order currently working at the venue."""
        client = self._require_client()
        return await client.open_orders()

    async def cancel_order(self, order_ref: str) -> bool:
        """Cancel one order by our client order id."""
        client = self._require_client()
        return await client.cancel_order(order_ref)

    async def get_trade(self, order_ref: str) -> VenueTrade | None:
        """Fetch the latest normalised state for one order id."""
        client = self._require_client()
        return await client.get_trade(order_ref)

    # -- BrokerAdapter-compatible aliases --------------------------------------

    async def get_account(self) -> AccountSnapshot:
        """Alias of :meth:`fetch_account` (BrokerAdapter naming)."""
        return await self.fetch_account()

    async def get_positions(
        self, prices: Mapping[InstrumentId, float] | None = None
    ) -> Mapping[InstrumentId, Position]:
        """Alias of :meth:`fetch_positions` (BrokerAdapter naming)."""
        return await self.fetch_positions(prices)


# ---------------------------------------------------------------------------
# Pure parsing / mapping helpers
# ---------------------------------------------------------------------------


def _random01() -> float:
    import random

    return random.random()


def _optional_float(value: object) -> float | None:
    """Parse an IBKR numeric string, mapping blanks/sentinels to ``None``."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        candidate = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            candidate = float(text)
        except ValueError:
            return None
    else:
        return None
    if not math.isfinite(candidate) or abs(candidate) >= _IBKR_UNSET:
        return None
    return candidate


def parse_contract_month(month_string: object) -> date | None:
    """Best-effort parse of an IBKR ``lastTradeDateOrContractMonth`` value."""
    if not isinstance(month_string, str):
        return None
    raw = month_string.strip()
    if not raw:
        return None
    try:
        return normalize_expiration(raw)[0]
    except IBKRDataError:
        return None


def margin_impact_from_order_state(state: object) -> MarginImpact:
    """Convert an IBKR ``OrderState``-like object into :class:`MarginImpact`."""

    def read(name: str) -> float | None:
        return _optional_float(getattr(state, name, None))

    warning = getattr(state, "warningText", "")
    return MarginImpact(
        initial_margin_before=read("initMarginBefore"),
        maintenance_margin_before=read("maintMarginBefore"),
        initial_margin_change=read("initMarginChange"),
        maintenance_margin_change=read("maintMarginChange"),
        initial_margin_after=read("initMarginAfter"),
        maintenance_margin_after=read("maintMarginAfter"),
        warning=warning if isinstance(warning, str) else "",
    )


_STATUS_MAP: Mapping[str, OrderStatus] = {
    "PendingSubmit": OrderStatus.PENDING,
    "ApiPending": OrderStatus.PENDING,
    "PendingCancel": OrderStatus.SUBMITTED,
    "PreSubmitted": OrderStatus.SUBMITTED,
    "Submitted": OrderStatus.SUBMITTED,
    "Filled": OrderStatus.FILLED,
    "Cancelled": OrderStatus.CANCELLED,
    "ApiCancelled": OrderStatus.CANCELLED,
    "Inactive": OrderStatus.REJECTED,
    "Rejected": OrderStatus.REJECTED,
}


def map_order_status(status: object) -> OrderStatus:
    """Map an IBKR status string onto the core :class:`OrderStatus`."""
    if isinstance(status, OrderStatus):
        return status
    if not isinstance(status, str):
        return OrderStatus.UNKNOWN
    return _STATUS_MAP.get(status.strip(), OrderStatus.UNKNOWN)


# ---------------------------------------------------------------------------
# ib_async adapter (lazy import; excluded from coverage / unit tests)
# ---------------------------------------------------------------------------


class _IbAsyncIBClient:
    """Concrete ``ib_async`` transport.

    ``ib_async`` is imported inside ``__init__`` (never at module import time)
    so the execution package stays importable without the live SDK installed.
    """

    def __init__(self) -> None:  # pragma: no cover - requires the live SDK
        from ib_async import IB

        self._ib: Any = IB()
        self._contracts: dict[int, Any] = {}
        self._requested_accounts: set[str] = set()
        self._account_tasks: set[asyncio.Future[Any]] = set()
        # TWS can wedge a reconnected client's account-update stream; serialise
        # ledger reads and prefer the request/response account-summary API.
        self._account_lock = asyncio.Lock()

    @property
    def is_connected(self) -> bool:
        return bool(self._ib.isConnected())  # pragma: no cover - SDK round trip

    async def connect(self, *, host: str, port: int, client_id: int, timeout: float) -> None:
        await self._ib.connectAsync(  # pragma: no cover - SDK round trip
            host=host, port=port, clientId=client_id, timeout=timeout
        )

    async def _request_account_updates_once(self) -> None:  # pragma: no cover
        """Request account updates for every account not requested yet.

        TWS only streams the ledger (NetLiquidation, margins, ...) after a
        per-account update subscription. Accounts may not be populated at the
        instant the socket opens, so this is retried from ``account_values``
        until the account codes are known.
        """
        accounts_method = getattr(self._ib, "managedAccounts", None)
        accounts = list(accounts_method() or []) if callable(accounts_method) else []
        request = getattr(self._ib, "reqAccountUpdatesAsync", None)
        if not callable(request):
            return
        loop = asyncio.get_running_loop()
        for account in accounts:
            if account in self._requested_accounts:
                continue
            self._requested_accounts.add(account)
            try:
                pending = request(account)
                future = asyncio.ensure_future(pending, loop=loop)
            except Exception:  # pragma: no cover - defensive SDK behaviour
                self._requested_accounts.discard(account)
                logger.debug("IBKR account-update subscription failed", exc_info=True)
                continue
            self._account_tasks.add(future)
            future.add_done_callback(self._account_tasks.discard)

    async def disconnect(self) -> None:  # pragma: no cover - SDK round trip
        self._ib.disconnect()

    async def _summary_rows(self) -> list[VenueAccountValue]:
        """Request/response ledger via IBKR's account-summary API."""
        summary = getattr(self._ib, "accountSummaryAsync", None)
        if not callable(summary):
            return []
        try:
            values = await asyncio.wait_for(summary(), timeout=8.0)
        except TimeoutError:
            logger.debug("IBKR account-summary request timed out")
            return []
        except Exception:  # pragma: no cover - defensive SDK behaviour
            logger.debug("IBKR account-summary request failed", exc_info=True)
            return []
        return [
            VenueAccountValue(
                tag=str(value.tag),
                value=str(value.value),
                currency=str(getattr(value, "currency", "")),
                account=str(getattr(value, "account", "")),
            )
            for value in (values or [])
        ]

    async def account_values(self) -> Sequence[VenueAccountValue]:  # pragma: no cover
        async with self._account_lock:
            # 1) accountSummary is a request/response round-trip and keeps working
            #    when the streaming account-update subscription is wedged after a
            #    TWS restart / duplicate client-id handshake.
            summary_rows = await self._summary_rows()
            rows = [
                VenueAccountValue(
                    tag=value.tag,
                    value=str(value.value),
                    currency=str(value.currency),
                    account=str(value.account),
                )
                for value in self._ib.accountValues()
            ]
            if any(row.tag == "NetLiquidation" for row in summary_rows):
                # Merge: cached rows provide extra tags, summary is authoritative.
                merged = {row.tag: row for row in rows}
                merged.update({row.tag: row for row in summary_rows})
                return list(merged.values())

            # 2) Fallback: subscribe and poll the streaming cache briefly.
            await self._request_account_updates_once()
            loop = asyncio.get_running_loop()
            deadline = loop.time() + 5.0
            while loop.time() < deadline:
                rows = [
                    VenueAccountValue(
                        tag=value.tag,
                        value=str(value.value),
                        currency=str(value.currency),
                        account=str(value.account),
                    )
                    for value in self._ib.accountValues()
                ]
                if any(row.tag == "NetLiquidation" for row in rows):
                    return rows
                await asyncio.sleep(0.2)
            return rows

    async def positions(self) -> Sequence[VenuePosition]:  # pragma: no cover
        rows: list[VenuePosition] = []
        for position in self._ib.positions():
            contract = position.contract
            rows.append(
                VenuePosition(
                    root_symbol=str(contract.symbol),
                    quantity=round(float(position.position)),
                    avg_cost=float(position.avgCost),
                    multiplier=_optional_float(contract.multiplier),
                    expiration=parse_contract_month(contract.lastTradeDateOrContractMonth),
                    local_symbol=str(contract.localSymbol) or None,
                )
            )
        return rows

    def _build_contract(self, request: ContractRequest) -> Any:
        from ib_async import Future

        return Future(
            symbol=request.root,
            lastTradeDateOrContractMonth=request.last_trade_date_or_contract_month,
            exchange=request.exchange,
            multiplier=_format_multiplier(request.multiplier),
            currency=request.currency,
        )

    def _contract_for(self, venue: VenueContract) -> Any:
        cached = self._contracts.get(venue.con_id)
        if cached is not None:
            return cached
        from ib_async import Future

        contract = Future(
            symbol=venue.root,
            lastTradeDateOrContractMonth=venue.last_trade_date_or_contract_month,
            exchange=venue.exchange,
            multiplier=_format_multiplier(venue.multiplier),
            currency=venue.currency,
        )
        contract.conId = venue.con_id
        self._contracts[venue.con_id] = contract
        return contract

    async def qualify_contract(self, request: ContractRequest) -> VenueContract:  # pragma: no cover
        contract = self._build_contract(request)
        qualified = await self._ib.qualifyContractsAsync(contract)
        resolved = qualified[0] if qualified else None
        if resolved is None or not getattr(resolved, "conId", 0):
            raise IBKRDataError(
                f"IBKR could not qualify {request.root} "
                f"{request.last_trade_date_or_contract_month} on {request.exchange}"
            )
        self._contracts[int(resolved.conId)] = resolved
        return VenueContract(
            root=str(resolved.symbol) or request.root,
            con_id=int(resolved.conId),
            exchange=str(resolved.exchange) or request.exchange,
            last_trade_date_or_contract_month=str(resolved.lastTradeDateOrContractMonth)
            or request.last_trade_date_or_contract_month,
            multiplier=_optional_float(resolved.multiplier) or request.multiplier,
            expiration=request.expiration,
            local_symbol=str(resolved.localSymbol) or None,
            currency=str(resolved.currency) or request.currency,
        )

    def _to_ib_order(self, request: IBOrderRequest) -> Any:
        from ib_async import Order, TagValue

        order = Order(
            action=request.action,
            totalQuantity=request.quantity,
            orderType=request.order_type,
            tif=request.time_in_force,
            orderRef=request.order_ref,
            transmit=request.transmit,
        )
        if request.limit_price is not None:
            order.lmtPrice = request.limit_price
        if request.stop_price is not None:
            order.auxPrice = request.stop_price
        if request.good_till_date is not None:
            order.goodTillDate = request.good_till_date
        if request.algo_strategy is not None:
            order.algoStrategy = request.algo_strategy
            order.algoParams = [TagValue(key, value) for key, value in request.algo_params]
        return order

    async def what_if_order(
        self, contract: VenueContract, order: IBOrderRequest
    ) -> MarginImpact:  # pragma: no cover - SDK round trip
        state = await self._ib.whatIfOrderAsync(
            self._contract_for(contract), self._to_ib_order(order)
        )
        return margin_impact_from_order_state(state)

    def _venue_trade(self, trade: Any) -> VenueTrade:
        order_ref = str(trade.order.orderRef or "")
        status = map_order_status(trade.orderStatus.status)
        fills: list[VenueFill] = []
        for fill in trade.fills:
            execution = fill.execution
            side = OrderSide.BUY if str(execution.side).upper() == "BOT" else OrderSide.SELL
            timestamp = fill.time if isinstance(fill.time, datetime) else None
            if timestamp is not None and timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=UTC)
            commission = _optional_float(getattr(fill.commissionReport, "commission", None)) or 0.0
            fills.append(
                VenueFill(
                    order_ref=order_ref,
                    side=side,
                    quantity=round(abs(float(execution.shares))),
                    price=float(execution.price),
                    timestamp=timestamp,
                    commission=commission,
                )
            )
        average_price = _optional_float(trade.orderStatus.avgFillPrice)
        return VenueTrade(
            order_ref=order_ref,
            status=status,
            filled_quantity=round(float(trade.orderStatus.filled or 0.0)),
            average_price=average_price if average_price and average_price > 0 else None,
            commission=sum(fill.commission for fill in fills),
            message=str(getattr(trade.orderStatus, "whyHeld", "") or ""),
            fills=tuple(fills),
        )

    async def place_order(
        self, contract: VenueContract, order: IBOrderRequest
    ) -> VenueTrade:  # pragma: no cover - SDK round trip
        trade = self._ib.placeOrder(self._contract_for(contract), self._to_ib_order(order))
        return self._venue_trade(trade)

    async def place_bracket(
        self,
        contract: VenueContract,
        parent: IBOrderRequest,
        child: IBOrderRequest,
    ) -> tuple[VenueTrade, VenueTrade]:  # pragma: no cover - SDK round trip
        ib_contract = self._contract_for(contract)
        parent_order = self._to_ib_order(parent)
        parent_order.transmit = False
        parent_trade = self._ib.placeOrder(ib_contract, parent_order)
        child_order = self._to_ib_order(child)
        child_order.parentId = parent_trade.order.orderId
        child_order.transmit = True
        child_trade = self._ib.placeOrder(ib_contract, child_order)
        return (self._venue_trade(parent_trade), self._venue_trade(child_trade))

    async def place_bag_order(self, bag: BagOrderSpec) -> VenueTrade:  # pragma: no cover
        trade = self._ib.placeOrder(self._build_bag(bag), self._to_ib_order(bag.order))
        return self._venue_trade(trade)

    def _build_bag(self, bag: BagOrderSpec) -> Any:
        from ib_async import Bag, ComboLeg

        contract = Bag(  # type: ignore[no-untyped-call]
            symbol=bag.root, exchange=bag.exchange, currency=bag.currency
        )
        contract.comboLegs = [
            ComboLeg(
                conId=leg.con_id,
                ratio=leg.ratio,
                action=leg.action.value,
                exchange=leg.exchange,
            )
            for leg in bag.legs
        ]
        return contract

    async def open_orders(self) -> Sequence[VenueOrderSnapshot]:  # pragma: no cover
        rows: list[VenueOrderSnapshot] = []
        for trade in self._ib.openTrades():
            order = trade.order
            rows.append(
                VenueOrderSnapshot(
                    order_ref=str(order.orderRef or ""),
                    status=map_order_status(trade.orderStatus.status),
                    instrument_id=str(trade.contract.symbol),
                    side=OrderSide.BUY if str(order.action).upper() == "BUY" else OrderSide.SELL,
                    quantity=round(float(order.totalQuantity)),
                    filled_quantity=round(float(trade.orderStatus.filled or 0.0)),
                )
            )
        return rows

    async def cancel_order(self, order_ref: str) -> bool:  # pragma: no cover
        for trade in self._ib.openTrades():
            if str(trade.order.orderRef or "") == order_ref:
                self._ib.cancelOrder(trade.order)
                return True
        return False

    async def get_trade(self, order_ref: str) -> VenueTrade | None:  # pragma: no cover
        for trade in reversed(self._ib.trades()):
            if str(trade.order.orderRef or "") == order_ref:
                return self._venue_trade(trade)
        return None


def _format_multiplier(multiplier: float) -> str:
    """Format a multiplier for IBKR (integer contracts must not become ``50.0``)."""
    if float(multiplier).is_integer():
        return str(int(multiplier))
    return f"{multiplier:g}"
