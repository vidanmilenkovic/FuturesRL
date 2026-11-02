"""Order Management System and mandatory pre-trade risk gate for IBKR execution.

The OMS is the single choke point between an approved target portfolio and the
live order book.  It has three responsibilities:

:class:`PreTradeRiskGate`
    An independent hard gate every order must clear *in this order*: the
    kill-switch/lock state, fat-finger limits (max contracts per order and max
    notional), duplicate / rapid-fire rate limiting (token bucket) and a
    mandatory IBKR ``whatIfOrder`` margin simulation compared against
    :attr:`~configs.settings.RiskSettings.max_margin_utilization`.  There is no
    bypass flag: an order that cannot be margin-simulated is rejected.

:func:`orders_from_targets` / :meth:`OrderManagementSystem.orders_from_targets`
    Deterministically translate a :class:`~src.core.types.ContractTargetSet`
    into the delta :class:`~src.core.types.OrderEvent` list, with idempotent
    ``order_id`` values (UUID5 of root/side/quantity/expiry/one-second bucket)
    so a retried rebalance at the same timestamp can never double-fire.

:class:`OrderManagementSystem`
    Order submission (MKT / LMT / IBALGO adaptive / TWAP / VWAP), bracket
    orders (entry + native exchange stop-loss child), batch target submission
    through the gate, order-status tracking, fill aggregation and the emergency
    ``flatten_all()`` / ``reset_kill_switch()`` kill-switch primitives.

The OMS talks to :class:`~src.execution.ibkr_client.IBKRConnectionManager`; it
never imports ``ib_async`` itself and is fully unit-testable with a fake client.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from configs.instruments import InstrumentsConfig
from configs.settings import Settings

from src.core.types import (
    AccountSnapshot,
    ContractTarget,
    ContractTargetSet,
    Fill,
    InstrumentId,
    OrderEvent,
    OrderReceipt,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    TimeInForce,
    ensure_utc,
)
from src.execution.ibkr_client import (
    IBKRConnectionManager,
    IBOrderRequest,
    MarginImpact,
    PriceProvider,
    VenueTrade,
)
from src.risk.state_store import PersistedRiskState, RateLimiterState, RiskStateStore

if TYPE_CHECKING:
    from src.execution.ibkr_client import Clock, Sleeper

logger = logging.getLogger(__name__)

#: Fixed UUID5 namespace for deterministic client order ids.
ORDER_ID_NAMESPACE: uuid.UUID = uuid.UUID("c9fb1c0e-6d0b-5c56-9f4f-8f7c0e7c2b11")

#: One second granularity for the deterministic order-id timestamp bucket.
_ORDER_ID_BUCKET_SECONDS: int = 1


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _wall_clock_now() -> float:
    """Wall-clock epoch seconds (the rate-limiter's restart-safe anchor)."""
    return time.time()


async def _asyncio_sleep(seconds: float) -> None:
    if seconds > 0:
        await asyncio.sleep(seconds)


def deterministic_order_id(
    *,
    instrument_id: InstrumentId,
    side: OrderSide,
    quantity: int,
    timestamp: datetime,
    contract_symbol: str | None = None,
    expiration: object | None = None,
    tag: str = "",
) -> str:
    """Return the deterministic, idempotent client order id for one delta.

    The id is the UUID5 of a canonical client-order key:
    ``root | contract symbol | expiry | side | quantity | second bucket``.
    Re-running the same rebalance at the same timestamp therefore produces the
    exact same ids (safe retries, no duplicate fills), while a later rebalance
    bucket produces new ids.
    """
    moment = ensure_utc(timestamp)
    bucket = moment.replace(microsecond=0)
    exp_key = ""
    if expiration is not None:
        exp_key = expiration.isoformat() if hasattr(expiration, "isoformat") else str(expiration)
    key = "|".join(
        (
            instrument_id,
            contract_symbol or "",
            exp_key,
            side.value,
            str(int(quantity)),
            bucket.isoformat(),
            tag,
        )
    )
    return f"ord-{uuid.uuid5(ORDER_ID_NAMESPACE, key).hex}"


def _delta_sign(delta: int) -> OrderSide:
    return OrderSide.BUY if delta > 0 else OrderSide.SELL


# ---------------------------------------------------------------------------
# Target -> order translation
# ---------------------------------------------------------------------------


def build_orders_from_targets(
    targets: ContractTargetSet,
    current_positions: Mapping[InstrumentId, int],
    timestamp: datetime,
    *,
    flatten: bool = False,
    tag: str = "rebalance",
) -> list[OrderEvent]:
    """Translate target contract quantities into the delta order list.

    ``delta = target - current``; zero deltas are skipped.  Instruments with a
    current position but no target are closed (the target set is the *full*
    target book), which also makes ``flatten=True`` (or an empty target set) a
    complete synthetic flatten.  Order ids are deterministic and idempotent.
    """
    moment = ensure_utc(timestamp)
    current = {key: int(value) for key, value in current_positions.items() if int(value) != 0}
    order: list[InstrumentId] = []
    positioned: dict[InstrumentId, ContractTarget] = {}
    if not flatten:
        for contract_target in targets:
            positioned[contract_target.instrument_id] = contract_target
            order.append(contract_target.instrument_id)
    for instrument_id in sorted(set(current) - set(positioned)):
        order.append(instrument_id)
    orders: list[OrderEvent] = []
    for instrument_id in order:
        target: ContractTarget | None = None if flatten else positioned.get(instrument_id)
        target_quantity = 0 if target is None else int(target.quantity)
        delta = target_quantity - current.get(instrument_id, 0)
        if delta == 0:
            continue
        side = _delta_sign(delta)
        quantity = abs(delta)
        contract_symbol = target.contract_symbol if target is not None else None
        expiration = target.expiration if target is not None else None
        orders.append(
            OrderEvent(
                order_id=deterministic_order_id(
                    instrument_id=instrument_id,
                    side=side,
                    quantity=quantity,
                    timestamp=moment,
                    contract_symbol=contract_symbol,
                    expiration=expiration,
                    tag=tag,
                ),
                timestamp=moment,
                instrument_id=instrument_id,
                side=side,
                quantity=quantity,
                order_type=OrderType.MARKET,
                time_in_force=TimeInForce.DAY,
                contract_symbol=contract_symbol,
                expiration=expiration,
                tag=tag,
            )
        )
    return orders


# ---------------------------------------------------------------------------
# Pre-trade risk gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GateDecision:
    """Recorded outcome of one pre-trade gate evaluation (audit trail)."""

    order_id: str
    accepted: bool
    message: str = ""
    initial_margin_utilization: float | None = None
    maintenance_margin_utilization: float | None = None


class PreTradeRiskGate:
    """Independent, mandatory pre-trade hard gate (implements ``PreTradeRiskGate``).

    Check order is fixed and non-bypassable:

    1. kill-switch / lock state,
    2. fat-finger: max contracts per order and max order notional,
    3. duplicate / rapid-fire rate limiter (token bucket, injectable clock),
    4. mandatory IBKR ``whatIfOrder`` margin simulation versus
       ``RiskSettings.max_margin_utilization``.

    ``emergency=True`` is reserved for :meth:`OrderManagementSystem.flatten_all`
    and only skips the lock, size and rate-limit checks (a closing order must
    never be blocked by the kill-switch it is enforcing); the margin simulation
    remains mandatory.

    Persisted token bucket (Phase 7)
    --------------------------------
    The rapid-fire token bucket can be anchored to *wall-clock* epoch seconds
    instead of the monotonic clock and seeded from a
    :class:`~src.risk.state_store.RateLimiterState` snapshot. A container restart
    then refills exactly the tokens earned while the process was down instead of
    silently resetting the bucket to full capacity (which would let a restart
    double the allowed order rate). Every accepted order persists the bucket
    snapshot through ``persist_rate_limiter``. When neither a snapshot nor a
    persistence callback is supplied, the original monotonic in-memory behaviour
    is preserved exactly.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        what_if: Callable[[OrderEvent], Awaitable[MarginImpact]],
        account_provider: Callable[[], Awaitable[AccountSnapshot]],
        price_provider: PriceProvider | None = None,
        instruments: InstrumentsConfig | None = None,
        clock: Clock | None = None,
        monotonic: Callable[[], float] | None = None,
        max_order_notional: float | None = None,
        rate_capacity: float | None = None,
        rate_limiter_state: RateLimiterState | None = None,
        wall_clock: Callable[[], float] | None = None,
        persist_rate_limiter: Callable[[RateLimiterState], None] | None = None,
    ) -> None:
        if max_order_notional is not None and max_order_notional <= 0:
            raise ValueError("max_order_notional must be strictly positive")
        self._settings = settings
        self._what_if = what_if
        self._account_provider = account_provider
        self._price_provider = price_provider
        self._instruments = instruments if instruments is not None else settings.instruments
        self._clock: Clock = clock if clock is not None else _utc_now
        self._monotonic: Callable[[], float] = monotonic if monotonic is not None else _monotonic
        self._wall_clock: Callable[[], float] = (
            wall_clock if wall_clock is not None else _wall_clock_now
        )
        self._persist_rate_limiter = persist_rate_limiter
        self._persistent_bucket = False
        self._max_order_notional = max_order_notional
        rate = self._settings.risk.order_rate_limit_per_second
        self._rate_capacity = rate_capacity if rate_capacity is not None else max(1.0, rate)
        self._tokens: float | None = None
        self._last_refill: float = 0.0
        self._recent_orders: dict[tuple[object, ...], float] = {}
        self._locked = False
        self._lock_reason = ""
        self.decisions: list[GateDecision] = []
        if rate_limiter_state is not None or persist_rate_limiter is not None:
            self.configure_rate_limiter(
                rate_limiter_state,
                wall_clock=self._wall_clock,
                persist=persist_rate_limiter,
            )

    # -- persisted rate limiter -------------------------------------------------

    @property
    def persistent_rate_limiter(self) -> bool:
        """Whether the token bucket is wall-clock anchored and restart-safe."""
        return self._persistent_bucket

    def configure_rate_limiter(
        self,
        state: RateLimiterState | None = None,
        *,
        wall_clock: Callable[[], float] | None = None,
        persist: Callable[[RateLimiterState], None] | None = None,
    ) -> None:
        """Enable the restart-safe token bucket, seeded from ``state``.

        ``state`` is a snapshot persisted by a previous process; its tokens are
        refilled from the stored wall-clock anchor by the configured
        ``order_rate_limit_per_second``. ``None`` starts from a full bucket
        anchored at the current wall clock. The configured ``rate_capacity`` is
        always authoritative (a changed setting must take effect after restart).
        """
        if wall_clock is not None:
            self._wall_clock = wall_clock
        if persist is not None:
            self._persist_rate_limiter = persist
        self._persistent_bucket = True
        now = self._wall_clock()
        if state is not None:
            self._tokens = min(float(self._rate_capacity), max(0.0, float(state.tokens)))
            self._last_refill = float(state.last_refill_epoch_seconds)
        else:
            self._tokens = float(self._rate_capacity)
            self._last_refill = now

    def rate_limiter_state(self) -> RateLimiterState | None:
        """Current token-bucket snapshot (``None`` when not persistent)."""
        if not self._persistent_bucket or self._tokens is None:
            return None
        return RateLimiterState(
            tokens=min(float(self._rate_capacity), max(0.0, self._tokens)),
            capacity=float(self._rate_capacity),
            last_refill_epoch_seconds=float(self._last_refill),
        )

    # -- kill switch -----------------------------------------------------------

    @property
    def is_locked(self) -> bool:
        """Whether the kill-switch currently blocks new risk-increasing orders."""
        return self._locked

    @property
    def lock_reason(self) -> str:
        """Human-readable reason for the current lock (empty when unlocked)."""
        return self._lock_reason

    @property
    def rejections(self) -> tuple[GateDecision, ...]:
        """Every rejected gate decision recorded so far."""
        return tuple(decision for decision in self.decisions if not decision.accepted)

    def lock(self, reason: str = "kill switch engaged") -> None:
        """Engage the kill-switch: no new order passes the gate."""
        self._locked = True
        self._lock_reason = reason
        logger.critical("pre-trade kill switch engaged: %s", reason)

    def unlock(self, reason: str = "manual reset") -> None:
        """Release the kill-switch (operator / API manual reset)."""
        self._locked = False
        self._lock_reason = ""
        logger.warning("pre-trade kill switch released: %s", reason)

    def reset_kill_switch(self) -> None:
        """Alias of :meth:`unlock` for the API/telemetry layer."""
        self.unlock("reset_kill_switch")

    # -- gate ------------------------------------------------------------------

    async def check(self, order: OrderEvent, *, emergency: bool = False) -> OrderReceipt | None:
        """Return ``None`` when ``order`` passes, otherwise a rejection receipt."""
        if not emergency and self._locked:
            return self._reject(order, f"kill switch engaged: {self._lock_reason}")
        if not emergency:
            fat_finger = self._check_fat_finger(order)
            if fat_finger is not None:
                return fat_finger
        account: AccountSnapshot | None = None
        notional_rejection, account = await self._check_notional(order, emergency)
        if notional_rejection is not None:
            return notional_rejection
        if not emergency:
            rate_rejection = self._check_rate_limit(order)
            if rate_rejection is not None:
                return rate_rejection
        margin_rejection, account, utilization = await self._check_margin(order, account)
        if margin_rejection is not None:
            return margin_rejection
        self._recent_orders[self._order_key(order)] = self._monotonic()
        initial_util, maintenance_util = utilization
        self.decisions.append(
            GateDecision(
                order_id=order.order_id,
                accepted=True,
                initial_margin_utilization=initial_util,
                maintenance_margin_utilization=maintenance_util,
            )
        )
        return None

    # -- individual checks -----------------------------------------------------

    def _check_fat_finger(self, order: OrderEvent) -> OrderReceipt | None:
        risk = self._settings.risk
        if order.quantity > risk.max_order_contracts:
            return self._reject(
                order,
                f"fat-finger: order quantity {order.quantity} exceeds max_order_contracts "
                f"{risk.max_order_contracts}",
            )
        return None

    def _notional_price(self, order: OrderEvent) -> float | None:
        if order.limit_price is not None:
            return order.limit_price
        if order.stop_price is not None:
            return order.stop_price
        provider = self._price_provider
        if provider is None:
            return None
        price = provider(order.instrument_id)
        if price is None or price <= 0:
            return None
        return price

    async def _check_notional(
        self, order: OrderEvent, emergency: bool
    ) -> tuple[OrderReceipt | None, AccountSnapshot | None]:
        if emergency:
            return (None, None)
        if not self._instruments.has(order.instrument_id):
            return (
                self._reject(
                    order,
                    f"fat-finger: no instrument master specification for {order.instrument_id!r}",
                ),
                None,
            )
        spec = self._instruments.get(order.instrument_id)
        price = self._notional_price(order)
        if price is None:
            return (
                self._reject(
                    order,
                    f"fat-finger: no mark price available to verify the order notional for "
                    f"{order.instrument_id!r}; wire a price provider or use a limit order",
                ),
                None,
            )
        notional = abs(order.quantity) * price * spec.multiplier
        account: AccountSnapshot | None = None
        cap = self._max_order_notional
        if cap is None:
            try:
                account = await self._account_provider()
            except Exception as exc:
                return (
                    self._reject(
                        order,
                        f"cannot fetch account state for the notional check: "
                        f"{type(exc).__name__}: {exc}",
                    ),
                    None,
                )
            cap = self._settings.risk.max_gross_leverage * account.net_liquidation
        if notional > cap:
            return (
                self._reject(
                    order,
                    f"fat-finger: order notional {notional:,.2f} exceeds the per-order "
                    f"notional cap {cap:,.2f}",
                ),
                account,
            )
        return (None, account)

    def _order_key(self, order: OrderEvent) -> tuple[object, ...]:
        return (
            order.instrument_id,
            order.side,
            order.quantity,
            order.order_type,
            order.limit_price,
            order.stop_price,
        )

    def _check_rate_limit(self, order: OrderEvent) -> OrderReceipt | None:
        now = self._monotonic()
        window = self._settings.risk.duplicate_order_window_seconds
        key = self._order_key(order)
        last_seen = self._recent_orders.get(key)
        if last_seen is not None and window > 0.0 and (now - last_seen) < window:
            return self._reject(
                order,
                f"duplicate order: an identical {order.instrument_id} order was accepted "
                f"{now - last_seen:.3f}s ago (window {window:.3f}s)",
            )
        rate = self._settings.risk.order_rate_limit_per_second
        if self._persistent_bucket:
            epoch = self._wall_clock()
            if self._tokens is None:  # pragma: no cover - configure() seeds the bucket
                self._tokens = float(self._rate_capacity)
                self._last_refill = epoch
            else:
                elapsed = max(0.0, epoch - self._last_refill)
                self._tokens = min(self._rate_capacity, self._tokens + elapsed * rate)
                self._last_refill = epoch
            if self._tokens < 1.0:
                return self._reject(
                    order,
                    f"rate limit: token bucket exhausted ({rate:g} orders/s configured)",
                )
            self._tokens -= 1.0
            persist = self._persist_rate_limiter
            if persist is not None:
                snapshot = self.rate_limiter_state()
                if snapshot is not None:
                    persist(snapshot)
            return None
        if self._tokens is None:
            self._tokens = self._rate_capacity
            self._last_refill = now
        else:
            elapsed = max(0.0, now - self._last_refill)
            self._tokens = min(self._rate_capacity, self._tokens + elapsed * rate)
            self._last_refill = now
        if self._tokens < 1.0:
            return self._reject(
                order,
                f"rate limit: token bucket exhausted ({rate:g} orders/s configured)",
            )
        self._tokens -= 1.0
        return None

    async def _check_margin(
        self, order: OrderEvent, account: AccountSnapshot | None
    ) -> tuple[OrderReceipt | None, AccountSnapshot | None, tuple[float | None, float | None]]:
        empty: tuple[float | None, float | None] = (None, None)
        try:
            impact = await self._what_if(order)
        except Exception as exc:
            return (
                self._reject(
                    order,
                    f"mandatory IBKR whatIfOrder margin simulation failed: "
                    f"{type(exc).__name__}: {exc}",
                ),
                account,
                empty,
            )
        if not impact.is_complete:
            return (
                self._reject(
                    order,
                    "mandatory IBKR whatIfOrder margin simulation returned incomplete "
                    "initial/maintenance margin data; order rejected",
                ),
                account,
                empty,
            )
        if account is None:
            try:
                account = await self._account_provider()
            except Exception as exc:
                return (
                    self._reject(
                        order,
                        f"cannot fetch account state for margin utilisation check: "
                        f"{type(exc).__name__}: {exc}",
                    ),
                    account,
                    empty,
                )
        initial_util, maintenance_util = impact.utilization(account.net_liquidation)
        limit = self._settings.risk.max_margin_utilization
        if initial_util is None or maintenance_util is None:
            return (
                self._reject(order, "margin utilisation could not be computed"),
                account,
                empty,
            )
        if initial_util > limit or maintenance_util > limit:
            return (
                self._reject(
                    order,
                    f"margin limit: projected utilisation initial={initial_util:.2%} "
                    f"maintenance={maintenance_util:.2%} exceeds max_margin_utilization "
                    f"{limit:.2%}",
                ),
                account,
                (initial_util, maintenance_util),
            )
        return (None, account, (initial_util, maintenance_util))

    def _reject(self, order: OrderEvent, message: str) -> OrderReceipt:
        receipt = OrderReceipt(
            order_id=order.order_id,
            status=OrderStatus.REJECTED,
            message=message,
            timestamp=self._clock(),
        )
        self.decisions.append(
            GateDecision(order_id=order.order_id, accepted=False, message=message)
        )
        logger.warning("pre-trade gate rejected %s: %s", order.order_id, message)
        return receipt


def _monotonic() -> float:
    import time

    return time.monotonic()


# ---------------------------------------------------------------------------
# Managed order state
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ManagedOrder:
    """Internal lifecycle record for one submitted order."""

    event: OrderEvent
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    average_price: float | None = None
    commission: float = 0.0
    message: str = ""
    receipt: OrderReceipt | None = None
    fills: list[Fill] = field(default_factory=list)
    venue_trade: VenueTrade | None = None

    @property
    def is_terminal(self) -> bool:
        """Whether the order reached a terminal lifecycle state."""
        return self.status in (
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
        )


# ---------------------------------------------------------------------------
# Order Management System
# ---------------------------------------------------------------------------


class OrderManagementSystem:
    """IBKR order management system implementing the ``BrokerAdapter`` surface."""

    def __init__(
        self,
        *,
        manager: IBKRConnectionManager,
        settings: Settings,
        instrument_master: InstrumentsConfig | None = None,
        gate: PreTradeRiskGate | None = None,
        clock: Clock | None = None,
        sleep: Sleeper | None = None,
        monotonic: Callable[[], float] | None = None,
        price_provider: PriceProvider | None = None,
        poll_interval_seconds: float = 0.05,
        state_store: RiskStateStore | None = None,
        wall_clock: Callable[[], float] | None = None,
    ) -> None:
        self._manager = manager
        self._settings = settings
        self._instruments = (
            instrument_master if instrument_master is not None else manager.instruments
        )
        self._clock: Clock = clock if clock is not None else _utc_now
        self._sleep: Sleeper = sleep if sleep is not None else _asyncio_sleep
        self._monotonic: Callable[[], float] = monotonic if monotonic is not None else _monotonic
        self._wall_clock: Callable[[], float] = (
            wall_clock if wall_clock is not None else _wall_clock_now
        )
        self._poll_interval = poll_interval_seconds
        self._orders: dict[str, ManagedOrder] = {}
        if gate is not None:
            self._gate = gate
        else:
            self._gate = PreTradeRiskGate(
                settings=settings,
                what_if=manager.what_if_order,
                account_provider=manager.fetch_account,
                price_provider=price_provider
                if price_provider is not None
                else manager.price_provider,
                instruments=self._instruments,
                clock=self._clock,
                monotonic=self._monotonic,
            )
        self._state_store = state_store
        self._persisted: PersistedRiskState | None = None
        if state_store is not None:
            # Fail closed: an unreadable risk state must abort startup rather than
            # silently re-arm a kill switch/bucket that was already consumed.
            persisted = state_store.load()
            self._persisted = persisted
            if persisted is not None and persisted.kill_switch_locked:
                self._gate.lock(persisted.kill_switch_reason or "persisted kill switch engaged")
            self._gate.configure_rate_limiter(
                persisted.rate_limiter if persisted is not None else None,
                wall_clock=self._wall_clock,
                persist=self._persist_rate_limiter,
            )

    # -- persisted risk state ---------------------------------------------------

    @property
    def persisted_state(self) -> PersistedRiskState | None:
        """Last durable risk state observed/persisted by the OMS."""
        return self._persisted

    def _persist_rate_limiter(self, bucket: RateLimiterState) -> None:
        """Persist the accepted-order token bucket (fail closed on I/O errors)."""
        self._update_persisted(lambda state: state.updated(rate_limiter=bucket))

    def _update_persisted(
        self,
        mutator: Callable[[PersistedRiskState], PersistedRiskState],
        *,
        raise_on_error: bool = True,
    ) -> PersistedRiskState | None:
        """Load-modify-save the durable risk state, preserving unrelated fields.

        Another process (the risk overlay) persists high-water marks and the
        daily-loss latch; re-reading before writing keeps those fields intact.
        """
        store = self._state_store
        if store is None:
            return None
        try:
            current = store.load()
            base = current if current is not None else PersistedRiskState()
            updated = mutator(base).updated(updated_at=self._clock())
            store.save(updated)
        except Exception:
            logger.exception("could not persist risk state")
            if raise_on_error:
                raise
            return None
        self._persisted = updated
        return updated

    def lock_kill_switch(self, reason: str = "kill switch engaged") -> None:
        """Engage the kill-switch and persist the latch across restarts."""
        self._gate.lock(reason)
        if self._state_store is not None:
            self._update_persisted(lambda state: state.locked(reason), raise_on_error=False)

    def reset_kill_switch(self) -> None:
        """Manually release the kill-switch and persist the released latch."""
        self._gate.reset_kill_switch()
        if self._state_store is not None:
            self._update_persisted(lambda state: state.unlocked(), raise_on_error=False)

    # -- introspection ---------------------------------------------------------

    @property
    def name(self) -> str:
        """Broker identifier (``BrokerAdapter.name``)."""
        return "ibkr-execution"

    @property
    def is_connected(self) -> bool:
        """Whether the underlying IBKR session is alive (``BrokerAdapter``)."""
        return self._manager.is_connected

    @property
    def manager(self) -> IBKRConnectionManager:
        """The underlying connection manager."""
        return self._manager

    @property
    def gate(self) -> PreTradeRiskGate:
        """The pre-trade risk gate guarding every submission."""
        return self._gate

    @property
    def is_locked(self) -> bool:
        """Whether the kill-switch is currently engaged."""
        return self._gate.is_locked

    @property
    def orders(self) -> Mapping[str, ManagedOrder]:
        """Read-only view of every managed order."""
        return dict(self._orders)

    @property
    def open_order_ids(self) -> tuple[str, ...]:
        """Ids of orders that are not terminal yet."""
        return tuple(order_id for order_id, order in self._orders.items() if not order.is_terminal)

    def order_status(self, order_id: str) -> OrderStatus:
        """Current lifecycle status for ``order_id`` (``UNKNOWN`` if untracked)."""
        managed = self._orders.get(order_id)
        return managed.status if managed is not None else OrderStatus.UNKNOWN

    def order_receipt(self, order_id: str) -> OrderReceipt | None:
        """Latest receipt for ``order_id`` (``None`` if untracked)."""
        managed = self._orders.get(order_id)
        return None if managed is None else managed.receipt

    def fills(self, order_id: str) -> list[Fill]:
        """Aggregated fills for one order."""
        managed = self._orders.get(order_id)
        return list(managed.fills) if managed is not None else []

    def all_fills(self) -> list[Fill]:
        """Every aggregated fill across all managed orders, in submission order."""
        aggregated: list[Fill] = []
        for managed in self._orders.values():
            aggregated.extend(managed.fills)
        return aggregated

    # -- BrokerAdapter lifecycle ----------------------------------------------

    async def connect(self) -> None:
        """Connect the underlying IBKR session."""
        await self._manager.connect()

    async def disconnect(self) -> None:
        """Disconnect the underlying IBKR session."""
        await self._manager.disconnect()

    async def get_account(self) -> AccountSnapshot:
        """Account snapshot (``BrokerAdapter``)."""
        return await self._manager.fetch_account()

    async def get_positions(self) -> Mapping[InstrumentId, Position]:
        """Broker positions (``BrokerAdapter``)."""
        return await self._manager.fetch_positions()

    # -- target -> order translation ------------------------------------------

    def orders_from_targets(
        self,
        targets: ContractTargetSet,
        current_positions: Mapping[InstrumentId, int],
        timestamp: datetime,
        *,
        flatten: bool = False,
        tag: str = "rebalance",
    ) -> list[OrderEvent]:
        """Deterministic delta orders for ``targets`` versus ``current_positions``."""
        return build_orders_from_targets(
            targets,
            current_positions,
            timestamp,
            flatten=flatten,
            tag=tag,
        )

    def flatten_orders(
        self,
        current_positions: Mapping[InstrumentId, int],
        timestamp: datetime,
    ) -> list[OrderEvent]:
        """Synthetic flatten target: market-close every non-zero position."""
        return self.orders_from_targets(
            ContractTargetSet.empty(),
            current_positions,
            timestamp,
            flatten=True,
            tag="flatten",
        )

    # -- submission ------------------------------------------------------------

    async def submit_order(
        self,
        order: OrderEvent,
        *,
        stop_price: float | None = None,
        emergency: bool = False,
    ) -> OrderReceipt:
        """Submit one order through the pre-trade gate (BrokerAdapter)."""
        if stop_price is not None:
            receipts = await self.submit_bracket(order, stop_price, emergency=emergency)
            return receipts[0]
        return await self._submit_single(order, emergency=emergency)

    async def submit_bracket(
        self, order: OrderEvent, stop_price: float, *, emergency: bool = False
    ) -> list[OrderReceipt]:
        """Submit an entry order plus a native exchange stop-loss child order."""
        if stop_price <= 0:
            raise ValueError("stop_price must be strictly positive")
        child = self._bracket_child(order, stop_price)
        parent_record = self._track(order)
        child_record = self._track(child)
        rejected = False
        for record in (parent_record, child_record):
            rejection = await self._gate.check(record.event, emergency=emergency)
            if rejection is not None:
                record.status = OrderStatus.REJECTED
                record.receipt = rejection
                record.message = rejection.message
                rejected = True
        if rejected:
            # A bracket is atomic: if either leg fails the gate, nothing is sent.
            for record in (parent_record, child_record):
                if not record.is_terminal:
                    record.status = OrderStatus.REJECTED
                    record.message = "bracket aborted: sibling order failed the pre-trade gate"
                    record.receipt = OrderReceipt(
                        order_id=record.event.order_id,
                        status=OrderStatus.REJECTED,
                        message=record.message,
                        timestamp=self._clock(),
                    )
            return [self._snapshot(parent_record), self._snapshot(child_record)]
        try:
            contract = await self._manager.resolve_contract(order.instrument_id, order.expiration)
            parent_request = IBOrderRequest.from_order_event(order)
            child_request = IBOrderRequest.from_order_event(child)
            parent_trade, child_trade = await self._manager.place_bracket(
                contract, parent_request, child_request
            )
        except Exception as exc:
            message = f"bracket submission failed: {type(exc).__name__}: {exc}"
            logger.exception("bracket submission failed for %s", order.order_id)
            for record in (parent_record, child_record):
                record.status = OrderStatus.REJECTED
                record.message = message
                record.receipt = OrderReceipt(
                    order_id=record.event.order_id,
                    status=OrderStatus.REJECTED,
                    message=message,
                    timestamp=self._clock(),
                )
            return [self._snapshot(parent_record), self._snapshot(child_record)]
        self._apply_trade(parent_record, parent_trade)
        self._apply_trade(child_record, child_trade)
        if child_record.status is OrderStatus.REJECTED:
            logger.error(
                "stop-loss child %s rejected; cancelling entry %s",
                child.order_id,
                order.order_id,
            )
            await self._safe_cancel(order.order_id)
        return [self._snapshot(parent_record), self._snapshot(child_record)]

    async def submit_orders(
        self, orders: Sequence[OrderEvent], *, emergency: bool = False
    ) -> list[OrderReceipt]:
        """Submit a batch sequentially, preserving order (rolls are two-legged)."""
        receipts: list[OrderReceipt] = []
        for order in orders:
            receipts.append(await self._submit_single(order, emergency=emergency))
        return receipts

    async def submit_targets(
        self,
        targets: ContractTargetSet,
        current_positions: Mapping[InstrumentId, int],
        timestamp: datetime,
        *,
        flatten: bool = False,
        stop_prices: Mapping[InstrumentId, float] | None = None,
    ) -> list[OrderReceipt]:
        """Translate targets into delta orders and submit them through the gate."""
        orders = self.orders_from_targets(targets, current_positions, timestamp, flatten=flatten)
        stops = stop_prices or {}
        receipts: list[OrderReceipt] = []
        for order in orders:
            stop_price = stops.get(order.instrument_id)
            if stop_price is not None:
                receipts.extend(await self.submit_bracket(order, stop_price))
            else:
                receipts.append(await self._submit_single(order))
        return receipts

    async def wait_for_completion(
        self, order_id: str, *, timeout_seconds: float = 5.0
    ) -> OrderReceipt:
        """Poll the venue until ``order_id`` is terminal (or the timeout expires)."""
        managed = self._orders.get(order_id)
        if managed is None:
            raise KeyError(f"unknown order id {order_id!r}")
        deadline = self._monotonic() + timeout_seconds
        while True:
            trade = await self._manager.get_trade(order_id)
            if trade is not None:
                self._apply_trade(managed, trade)
            if managed.is_terminal:
                break
            if self._monotonic() >= deadline:
                managed.message = (
                    f"timed out after {timeout_seconds:g}s waiting for {order_id} to complete"
                )
                break
            await self._sleep(self._poll_interval)
        return self._snapshot(managed)

    # -- cancellation / kill switch -------------------------------------------

    async def cancel_order(self, order_id: str) -> OrderReceipt:
        """Cancel one order by client order id."""
        cancelled = await self._manager.cancel_order(order_id)
        managed = self._orders.get(order_id)
        status = OrderStatus.CANCELLED if cancelled else OrderStatus.UNKNOWN
        message = "cancelled" if cancelled else "order not working at venue"
        if managed is not None and not managed.is_terminal:
            managed.status = status
            managed.message = message
        receipt = OrderReceipt(
            order_id=order_id,
            status=status,
            filled_quantity=managed.filled_quantity if managed is not None else 0,
            message=message,
            timestamp=self._clock(),
        )
        if managed is not None:
            managed.receipt = receipt
        return receipt

    async def cancel_all(self) -> list[OrderReceipt]:
        """Cancel every order currently working at the venue."""
        snapshots = await self._manager.open_orders()
        receipts: list[OrderReceipt] = []
        for snapshot in snapshots:
            receipts.append(await self.cancel_order(snapshot.order_ref))
        return receipts

    async def flatten_all(self) -> list[OrderReceipt]:
        """Emergency kill-switch: cancel all, market-close everything, then lock."""
        receipts: list[OrderReceipt] = []
        try:
            receipts.extend(await self.cancel_all())
        except Exception as exc:
            logger.exception("flatten_all: cancel_all failed")
            receipts.append(
                OrderReceipt(
                    order_id="cancel-all",
                    status=OrderStatus.REJECTED,
                    message=f"cancel_all failed: {type(exc).__name__}: {exc}",
                    timestamp=self._clock(),
                )
            )
        self.lock_kill_switch("emergency flatten_all")
        timestamp = self._clock()
        try:
            positions = await self._manager.fetch_positions()
        except Exception as exc:
            logger.exception("flatten_all: position fetch failed after kill-switch engagement")
            receipts.append(
                OrderReceipt(
                    order_id="flatten-positions",
                    status=OrderStatus.REJECTED,
                    message=f"flatten position fetch failed: {type(exc).__name__}: {exc}",
                    timestamp=timestamp,
                )
            )
            return receipts
        for position in positions.values():
            if position.quantity == 0:
                continue
            side = OrderSide.SELL if position.quantity > 0 else OrderSide.BUY
            quantity = abs(position.quantity)
            order = OrderEvent(
                order_id=deterministic_order_id(
                    instrument_id=position.instrument_id,
                    side=side,
                    quantity=quantity,
                    timestamp=timestamp,
                    contract_symbol=position.contract_symbol,
                    expiration=position.expiration,
                    tag="flatten",
                ),
                timestamp=timestamp,
                instrument_id=position.instrument_id,
                side=side,
                quantity=quantity,
                order_type=OrderType.MARKET,
                time_in_force=TimeInForce.DAY,
                contract_symbol=position.contract_symbol,
                expiration=position.expiration,
                tag="flatten",
            )
            receipts.append(await self._submit_single(order, emergency=True))
        return receipts

    # -- internals -------------------------------------------------------------

    def _track(self, order: OrderEvent) -> ManagedOrder:
        record = self._orders.get(order.order_id)
        if record is None:
            record = ManagedOrder(event=order)
            self._orders[order.order_id] = record
        return record

    def _bracket_child(self, order: OrderEvent, stop_price: float) -> OrderEvent:
        side = OrderSide.SELL if order.side is OrderSide.BUY else OrderSide.BUY
        return OrderEvent(
            order_id=f"{order.order_id}-sl",
            timestamp=order.timestamp,
            instrument_id=order.instrument_id,
            side=side,
            quantity=order.quantity,
            order_type=OrderType.STOP,
            time_in_force=order.time_in_force,
            stop_price=stop_price,
            contract_symbol=order.contract_symbol,
            expiration=order.expiration,
            parent_order_id=order.order_id,
            tag="bracket-stop",
        )

    async def _submit_single(self, order: OrderEvent, *, emergency: bool = False) -> OrderReceipt:
        record = self._track(order)
        if record.venue_trade is not None or record.is_terminal:
            # Idempotent retry: an order that already reached the venue (or is
            # finished) is never transmitted a second time.  Only gate-rejected
            # orders (no venue trade) may be retried by the caller.
            return self._snapshot(record)
        try:
            rejection = await self._gate.check(order, emergency=emergency)
        except Exception as exc:
            message = f"pre-trade gate failure: {type(exc).__name__}: {exc}"
            logger.exception("pre-trade gate crashed for %s", order.order_id)
            record.status = OrderStatus.REJECTED
            record.message = message
            record.receipt = OrderReceipt(
                order_id=order.order_id,
                status=OrderStatus.REJECTED,
                message=message,
                timestamp=self._clock(),
            )
            return self._snapshot(record)
        if rejection is not None:
            record.status = rejection.status
            record.message = rejection.message
            record.receipt = rejection
            return rejection
        try:
            contract = await self._manager.resolve_contract(order.instrument_id, order.expiration)
            request = IBOrderRequest.from_order_event(order)
            trade = await self._manager.place_order(contract, request)
        except Exception as exc:
            message = f"order transmission failed: {type(exc).__name__}: {exc}"
            logger.exception("order %s transmission failed", order.order_id)
            record.status = OrderStatus.REJECTED
            record.message = message
            record.receipt = OrderReceipt(
                order_id=order.order_id,
                status=OrderStatus.REJECTED,
                message=message,
                timestamp=self._clock(),
            )
            return self._snapshot(record)
        self._apply_trade(record, trade)
        return self._snapshot(record)

    async def _safe_cancel(self, order_id: str) -> None:
        try:
            await self._manager.cancel_order(order_id)
        except Exception:  # pragma: no cover - defensive: entry may not be working
            logger.exception("failed to cancel order %s", order_id)

    def _apply_trade(self, record: ManagedOrder, trade: VenueTrade) -> None:
        record.venue_trade = trade
        record.status = trade.status
        record.filled_quantity = trade.filled_quantity
        record.average_price = trade.average_price
        record.commission = trade.commission
        if trade.message:
            record.message = trade.message
        fills: list[Fill] = []
        for venue_fill in trade.fills:
            timestamp = venue_fill.timestamp if venue_fill.timestamp is not None else self._clock()
            fills.append(
                Fill(
                    order_id=record.event.order_id,
                    instrument_id=record.event.instrument_id,
                    side=venue_fill.side,
                    quantity=venue_fill.quantity,
                    price=venue_fill.price,
                    timestamp=timestamp,
                    commission=venue_fill.commission,
                )
            )
        record.fills = fills
        record.receipt = self._snapshot(record)

    def _snapshot(self, record: ManagedOrder) -> OrderReceipt:
        return OrderReceipt(
            order_id=record.event.order_id,
            status=record.status,
            filled_quantity=record.filled_quantity,
            average_price=record.average_price,
            commission=record.commission,
            message=record.message,
            timestamp=self._clock(),
        )
