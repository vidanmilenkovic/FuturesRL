"""Automated futures contract-roll execution via native IBKR BAG calendar spreads.

Rolling a futures position by sending two independent leg orders exposes the
desk to leg-to-leg slippage: the front leg can fill while the back leg runs
away.  IBKR supports native calendar-spread combo orders (``secType="BAG"`` with
one combo leg per expiry), which execute atomically at a single spread price.
:class:`ContractRoller`:

* resolves the front and next contract expiry from the instrument master's roll
  rules (contract month codes + ``days_before_fnd``), deterministically;
* builds the ratio-1:1 ``BAG`` combo (sell front / buy back for a long roll and
  the mirror for a short roll);
* submits it through the connection manager and verifies the fill;
* falls back to two sequential leg orders when the venue reports ``BAG`` as
  unsupported (or rejects the combo), still verifying the first leg before
  opening the second so a failed roll can never leave an unhedged double
  position;
* reports partial fills explicitly instead of assuming the roll completed.

The roller never imports ``ib_async``: the pure :class:`BagOrderSpec` /
:class:`~src.execution.ibkr_client.IBOrderRequest` dataclasses are converted by
the connection manager's transport adapter, so every test here is offline.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

from configs.instruments import InstrumentsConfig, InstrumentSpec

from src.core.types import (
    Fill,
    InstrumentId,
    OrderReceipt,
    OrderSide,
    OrderStatus,
    ensure_utc,
)
from src.execution.ibkr_client import (
    BagLeg,
    BagOrderSpec,
    IBKRBagUnsupportedError,
    IBKRConnectionManager,
    IBKRDataError,
    IBOrderRequest,
    VenueContract,
    VenueTrade,
    upcoming_contract_expirations,
)

if TYPE_CHECKING:
    from src.execution.ibkr_client import Clock, Sleeper

logger = logging.getLogger(__name__)

#: Fixed UUID5 namespace for deterministic roll order references.
ROLL_ID_NAMESPACE: uuid.UUID = uuid.UUID("5c1b2e42-9a8a-5d3f-b7d2-1f4e2f4f8d07")

#: Error text fragments that indicate the venue cannot handle BAG combos.
_BAG_UNSUPPORTED_HINTS: tuple[str, ...] = (
    "bag",
    "combo",
    "calendar spread",
    "unsupported",
    "not supported",
    "invalid order type",
)


def _utc_now() -> datetime:
    return datetime.now(UTC)


async def _asyncio_sleep(seconds: float) -> None:
    if seconds > 0:
        await asyncio.sleep(seconds)


def _monotonic() -> float:
    return time.monotonic()


# ---------------------------------------------------------------------------
# Deterministic roll scheduling
# ---------------------------------------------------------------------------


def select_roll_expirations(
    spec: InstrumentSpec, as_of: date, *, lookahead: int = 8
) -> tuple[date, date]:
    """Return ``(front, next)`` contract expiries for a roll on ``as_of``.

    The front contract is the first listed contract month that is still outside
    its mandatory roll window (``days_before_fnd`` before expiry); the next
    contract is the following listed month.  Selection is pure calendar
    arithmetic from the instrument master, hence deterministic.
    """
    candidates = upcoming_contract_expirations(spec, as_of, count=lookahead)
    if len(candidates) < 2:  # pragma: no cover - instrument master always lists >= 2 months
        raise IBKRDataError(f"instrument {spec.root} lists fewer than two contract months")
    days = spec.roll.days_before_fnd
    for index, expiry in enumerate(candidates[:-1]):
        if expiry - timedelta(days=days) > as_of:
            return (expiry, candidates[index + 1])
    raise IBKRDataError(  # pragma: no cover - requires ten months inside the roll window
        f"no rollable contract pair found for {spec.root} on {as_of}"
    )


def _roll_order_ref(
    *,
    root: str,
    front: VenueContract,
    back: VenueContract,
    quantity: int,
    timestamp: datetime,
    suffix: str = "",
) -> str:
    moment = ensure_utc(timestamp).replace(microsecond=0)
    key = "|".join(
        (
            root,
            str(front.con_id),
            str(back.con_id),
            str(int(quantity)),
            moment.isoformat(),
        )
    )
    digest = uuid.uuid5(ROLL_ID_NAMESPACE, key).hex
    return f"r-{digest[:32]}{suffix}"


# ---------------------------------------------------------------------------
# Roll plan / execution containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RollPlan:
    """Everything needed to execute one roll: both contracts and direction."""

    root: str
    quantity: int
    front: VenueContract
    back: VenueContract
    order_ref: str

    def __post_init__(self) -> None:
        if self.quantity == 0:
            raise ValueError("roll quantity must be non-zero")
        if self.front.con_id == self.back.con_id:
            raise ValueError("front and back contracts must differ")
        if self.front.expiration is None or self.back.expiration is None:
            raise ValueError("roll contracts must carry expirations")
        if self.back.expiration <= self.front.expiration:  # pragma: no cover - calendar guard
            raise ValueError("back contract must expire after the front contract")

    @property
    def spread_action(self) -> OrderSide:
        """Action of the combo order: BUY the spread for a long roll."""
        return OrderSide.BUY if self.quantity > 0 else OrderSide.SELL

    @property
    def front_action(self) -> OrderSide:
        """Action of the front leg (close the existing position)."""
        return OrderSide.SELL if self.quantity > 0 else OrderSide.BUY

    @property
    def back_action(self) -> OrderSide:
        """Action of the back leg (open the next position)."""
        return OrderSide.BUY if self.quantity > 0 else OrderSide.SELL

    @property
    def absolute_quantity(self) -> int:
        """Unsigned roll size."""
        return abs(self.quantity)


@dataclass(frozen=True, slots=True)
class RollExecution:
    """Outcome of one roll attempt."""

    plan: RollPlan
    bag_used: bool
    receipts: tuple[OrderReceipt, ...] = ()
    fills: tuple[Fill, ...] = ()
    complete: bool = False
    message: str = ""

    @property
    def filled_quantity(self) -> int:
        """Total absolute quantity filled across every receipt."""
        return sum(receipt.filled_quantity for receipt in self.receipts)


# ---------------------------------------------------------------------------
# Roller
# ---------------------------------------------------------------------------


class ContractRoller:
    """Executes automated front-to-next contract rolls through IBKR."""

    def __init__(
        self,
        manager: IBKRConnectionManager,
        *,
        instrument_master: InstrumentsConfig | None = None,
        clock: Clock | None = None,
        sleep: Sleeper | None = None,
        monotonic: Callable[[], float] | None = None,
        wait_timeout_seconds: float = 10.0,
        poll_interval_seconds: float = 0.05,
    ) -> None:
        self._manager = manager
        self._instruments = (
            instrument_master if instrument_master is not None else manager.instruments
        )
        self._clock: Clock = clock if clock is not None else _utc_now
        self._sleep: Sleeper = sleep if sleep is not None else _asyncio_sleep
        self._monotonic: Callable[[], float] = monotonic if monotonic is not None else _monotonic
        self._wait_timeout = wait_timeout_seconds
        self._poll_interval = poll_interval_seconds

    @property
    def manager(self) -> IBKRConnectionManager:
        """The connection manager used to resolve and execute roll orders."""
        return self._manager

    @property
    def instruments(self) -> InstrumentsConfig:
        """The instrument master used for roll scheduling."""
        return self._instruments

    def _as_of_date(self, as_of: date | None) -> date:
        if as_of is not None:
            return as_of
        return ensure_utc(self._clock()).date()

    async def plan_roll(
        self, root: InstrumentId, quantity: int, *, as_of: date | None = None
    ) -> RollPlan:
        """Resolve the front/next contracts and build the deterministic roll plan."""
        if quantity == 0:
            raise ValueError("quantity must be non-zero to plan a roll")
        try:
            spec = self._instruments.get(root)
        except KeyError as exc:
            raise IBKRDataError(str(exc)) from exc
        front_expiry, back_expiry = select_roll_expirations(spec, self._as_of_date(as_of))
        front = await self._manager.resolve_contract(spec.root, front_expiry)
        back = await self._manager.resolve_contract(spec.root, back_expiry)
        return RollPlan(
            root=spec.root,
            quantity=int(quantity),
            front=front,
            back=back,
            order_ref=_roll_order_ref(
                root=spec.root,
                front=front,
                back=back,
                quantity=int(quantity),
                timestamp=self._clock(),
            ),
        )

    def build_bag_spec(self, plan: RollPlan) -> BagOrderSpec:
        """Build the ratio-balanced native ``BAG`` calendar-spread order."""
        legs = (
            BagLeg(
                con_id=plan.front.con_id,
                action=plan.front_action,
                ratio=1,
                exchange=plan.front.exchange,
            ),
            BagLeg(
                con_id=plan.back.con_id,
                action=plan.back_action,
                ratio=1,
                exchange=plan.back.exchange,
            ),
        )
        request = IBOrderRequest(
            order_ref=plan.order_ref,
            action=plan.spread_action.value,
            quantity=plan.absolute_quantity,
            order_type="MKT",
            time_in_force="DAY",
        )
        return BagOrderSpec(
            root=plan.root,
            exchange=plan.front.exchange,
            legs=legs,
            order=request,
            currency=plan.front.currency,
        )

    async def execute_roll(
        self,
        root: InstrumentId,
        quantity: int,
        *,
        as_of: date | None = None,
        wait: bool = True,
    ) -> RollExecution:
        """Execute a native BAG roll, falling back to two sequential leg orders."""
        plan = await self.plan_roll(root, quantity, as_of=as_of)
        bag = self.build_bag_spec(plan)
        try:
            trade = await self._manager.place_bag_order(bag)
        except IBKRBagUnsupportedError as exc:
            logger.warning("IBKR rejected the BAG combo for %s: %s; using leg fallback", root, exc)
            return await self._execute_legs(plan, reason=str(exc))
        except Exception as exc:
            logger.exception("BAG roll submission failed for %s", root)
            receipt = _rejected_receipt(
                plan.order_ref, f"BAG submission failed: {exc}", self._clock()
            )
            return RollExecution(
                plan=plan,
                bag_used=True,
                receipts=(receipt,),
                complete=False,
                message=receipt.message,
            )
        if wait and not trade.is_terminal:
            trade = await self._await_trade(plan.order_ref, trade)
        if _bag_unsupported(trade):
            logger.warning(
                "IBKR reported BAG unsupported for %s (%s); using leg fallback",
                root,
                trade.message,
            )
            return await self._execute_legs(plan, reason=trade.message)
        receipt = self._receipt(plan.order_ref, trade)
        fills = self._fills(plan.root, trade)
        expected = plan.absolute_quantity
        complete = trade.status is OrderStatus.FILLED and trade.filled_quantity >= expected
        message = "" if complete else _incomplete_message(trade, expected)
        return RollExecution(
            plan=plan,
            bag_used=True,
            receipts=(receipt,),
            fills=tuple(fills),
            complete=complete,
            message=message,
        )

    async def _execute_legs(self, plan: RollPlan, *, reason: str = "") -> RollExecution:
        """Two sequential leg orders (front close first, then back open)."""
        receipts: list[OrderReceipt] = []
        fills: list[Fill] = []
        front_request = self._leg_request(plan, suffix="-f", action=plan.front_action)
        front_trade = await self._place_leg(front_request, plan.front)
        receipts.append(self._receipt(front_request.order_ref, front_trade))
        fills.extend(self._fills(plan.root, front_trade))
        if (
            front_trade.status is not OrderStatus.FILLED
            or front_trade.filled_quantity < plan.absolute_quantity
        ):
            message = _incomplete_message(front_trade, plan.absolute_quantity)
            return RollExecution(
                plan=plan,
                bag_used=False,
                receipts=tuple(receipts),
                fills=tuple(fills),
                complete=False,
                message=f"front leg did not fully fill ({message}); back leg not sent "
                + (f"[fallback reason: {reason}]" if reason else ""),
            )
        back_request = self._leg_request(plan, suffix="-b", action=plan.back_action)
        back_trade = await self._place_leg(back_request, plan.back)
        receipts.append(self._receipt(back_request.order_ref, back_trade))
        fills.extend(self._fills(plan.root, back_trade))
        complete = (
            back_trade.status is OrderStatus.FILLED
            and back_trade.filled_quantity >= plan.absolute_quantity
        )
        message = "" if complete else _incomplete_message(back_trade, plan.absolute_quantity)
        return RollExecution(
            plan=plan,
            bag_used=False,
            receipts=tuple(receipts),
            fills=tuple(fills),
            complete=complete,
            message=message,
        )

    def _leg_request(self, plan: RollPlan, *, suffix: str, action: OrderSide) -> IBOrderRequest:
        return IBOrderRequest(
            order_ref=f"{plan.order_ref}{suffix}",
            action=action.value,
            quantity=plan.absolute_quantity,
            order_type="MKT",
            time_in_force="DAY",
        )

    async def _place_leg(self, request: IBOrderRequest, contract: VenueContract) -> VenueTrade:
        try:
            return await self._manager.place_order(contract, request)
        except Exception as exc:
            logger.exception("roll leg %s failed", request.order_ref)
            return VenueTrade(
                order_ref=request.order_ref,
                status=OrderStatus.REJECTED,
                message=str(exc),
            )

    async def _await_trade(self, order_ref: str, initial: VenueTrade) -> VenueTrade:
        deadline = self._monotonic() + self._wait_timeout
        trade = initial
        while not trade.is_terminal:
            if self._monotonic() >= deadline:
                return trade
            await self._sleep(self._poll_interval)
            latest = await self._manager.get_trade(order_ref)
            if latest is None:
                continue
            trade = latest
        return trade

    def _receipt(self, order_ref: str, trade: VenueTrade) -> OrderReceipt:
        return OrderReceipt(
            order_id=order_ref,
            status=trade.status,
            filled_quantity=trade.filled_quantity,
            average_price=trade.average_price,
            commission=trade.commission,
            message=trade.message,
            timestamp=self._clock(),
        )

    def _fills(self, instrument_id: InstrumentId, trade: VenueTrade) -> list[Fill]:
        fills: list[Fill] = []
        for venue_fill in trade.fills:
            timestamp = venue_fill.timestamp if venue_fill.timestamp is not None else self._clock()
            fills.append(
                Fill(
                    order_id=trade.order_ref,
                    instrument_id=instrument_id,
                    side=venue_fill.side,
                    quantity=venue_fill.quantity,
                    price=venue_fill.price,
                    timestamp=timestamp,
                    commission=venue_fill.commission,
                )
            )
        return fills


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rejected_receipt(order_id: str, message: str, timestamp: datetime) -> OrderReceipt:
    return OrderReceipt(
        order_id=order_id,
        status=OrderStatus.REJECTED,
        message=message,
        timestamp=timestamp,
    )


def _bag_unsupported(trade: VenueTrade) -> bool:
    if trade.status not in (OrderStatus.REJECTED, OrderStatus.CANCELLED):
        return False
    message = trade.message.lower()
    return any(hint in message for hint in _BAG_UNSUPPORTED_HINTS)


def _incomplete_message(trade: VenueTrade, expected: int) -> str:
    if trade.status is OrderStatus.REJECTED:
        return f"roll rejected: {trade.message}"
    if trade.status is OrderStatus.CANCELLED:
        return f"roll cancelled after filling {trade.filled_quantity}/{expected}"
    if trade.status is OrderStatus.PARTIALLY_FILLED:
        return f"partial fill {trade.filled_quantity}/{expected}"
    return (
        f"roll not complete: status={trade.status.value} filled={trade.filled_quantity}/{expected}"
    )
