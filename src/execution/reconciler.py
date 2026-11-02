"""Periodic internal-versus-IBKR position/cash state reconciliation.

Live futures execution is only safe while the OMS's internal view of the book
matches the broker's.  :class:`PositionReconciler` compares:

* the *expected* positions (injected provider, typically the OMS/translator's
  target book net of known fills) against ``ib.positions()``; and
* the broker's ``cash_balance`` against the internally expected cash.

Any **unapproved** non-zero contract drift halts automated rebalancing
(``halted=True``) and fires an alert callback; the halt is sticky until an
operator calls :meth:`PositionReconciler.resume` (or ``auto_resume=True`` and a
clean reconciliation clears it).  Contracts are indivisible, so the position
tolerance defaults to zero: a one-lot drift is a real drift.

Open-order-aware drift (Phase 7)
--------------------------------
``ib.positions()`` only reflects **filled** quantity.  An order that is still
working (open trade, partially filled) legitimately makes the broker's reported
position differ from the target book for a few seconds, and marking that as an
unapproved drift would halt trading on every ordinary rebalance.  When a
``working_orders`` provider is injected the reconciler therefore compares
*effective* positions::

    effective_position = broker_position + sum(signed_remaining of working orders)

and only flags residual drift.  If the residual drift can still be explained by
an in-flight order (the newest relevant working order was updated less than
``reconcile_grace_seconds`` ago) the pass is **not** halted: it reports a note
and records ``last_grace_active=True`` / ``last_working_deltas``.  A stale
in-flight order (older than the grace window) is reported as real drift because
a working order that has not progressed for longer than the grace window is
itself an anomaly.  With ``working_orders=None`` the original behaviour is
preserved exactly (no provider call, no grace window).

The reconciler is fully offline-testable: every I/O source is an injected
callable and both clock and sleep are injectable, so the async ``run_forever``
loop is deterministic under test.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from types import MappingProxyType
from typing import TYPE_CHECKING

from src.core.types import (
    AccountSnapshot,
    InstrumentId,
    Position,
    ReconcileReport,
    ensure_utc,
)

if TYPE_CHECKING:
    from src.execution.ibkr_client import Clock, Sleeper

logger = logging.getLogger(__name__)

#: Expected positions: a static mapping or a callable returning one.
type ExpectedPositions = Mapping[InstrumentId, int] | Callable[[], Mapping[InstrumentId, int]]

#: Broker positions provider (typically ``IBKRConnectionManager.fetch_positions``).
type PositionsProvider = Callable[[], Awaitable[Mapping[InstrumentId, Position]]]

#: Broker account provider (typically ``IBKRConnectionManager.fetch_account``).
type AccountProvider = Callable[[], Awaitable[AccountSnapshot]]

#: Alert sink invoked for every unapproved drift report.
type AlertCallback = Callable[[ReconcileReport], None]

#: Expected cash: static value or callable (fills/commission-aware).
type ExpectedCash = float | Callable[[], float]

#: Working-order snapshot provider (``IBKRConnectionManager.open_orders`` analogue).
type WorkingOrdersProvider = Callable[[], Awaitable[Sequence["WorkingOrderSnapshot"]]]


@dataclass(frozen=True, slots=True)
class WorkingOrderSnapshot:
    """One in-flight order as observed at the venue.

    ``signed_remaining`` is the signed quantity still working at the exchange
    (``+`` buy, ``-`` sell); a partial fill contributes only its remaining
    quantity, because the filled part is already reflected in the broker
    position.  ``updated_at`` is the venue timestamp of the last order-state
    transition and drives the reconciliation grace window.
    """

    order_id: str
    instrument_id: InstrumentId
    signed_remaining: int
    updated_at: datetime
    status: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.order_id, str) or not self.order_id:
            raise ValueError("order_id must be a non-empty string")
        if not isinstance(self.instrument_id, str) or not self.instrument_id:
            raise ValueError("instrument_id must be a non-empty string")
        if isinstance(self.signed_remaining, bool) or not isinstance(self.signed_remaining, int):
            raise TypeError("signed_remaining must be an int")
        object.__setattr__(self, "updated_at", ensure_utc(self.updated_at))
        if not isinstance(self.status, str):
            raise TypeError("status must be a string")


def _utc_now() -> datetime:
    return datetime.now(UTC)


async def _asyncio_sleep(seconds: float) -> None:
    if seconds > 0:
        await asyncio.sleep(seconds)


class PositionReconciler:
    """Compares expected state against broker state and halts on drift.

    Implements the :class:`~src.core.interfaces.PositionReconciler` protocol
    (``async reconcile() -> bool``) while exposing the richer telemetry
    (``halted``, ``last_report``, ``last_cash_drift``) the API layer consumes.
    """

    def __init__(
        self,
        *,
        expected_positions: ExpectedPositions,
        broker_positions: PositionsProvider,
        expected_cash: ExpectedCash | None = None,
        broker_account: AccountProvider | None = None,
        cash_tolerance: float = 1.0,
        position_tolerance: int = 0,
        alert: AlertCallback | None = None,
        interval_seconds: float = 30.0,
        clock: Clock | None = None,
        sleep: Sleeper | None = None,
        stop_event: asyncio.Event | None = None,
        auto_resume: bool = False,
        working_orders: WorkingOrdersProvider | None = None,
        reconcile_grace_seconds: float = 30.0,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be strictly positive")
        if cash_tolerance < 0:
            raise ValueError("cash_tolerance must be non-negative")
        if position_tolerance < 0:
            raise ValueError("position_tolerance must be non-negative")
        if reconcile_grace_seconds < 0:
            raise ValueError("reconcile_grace_seconds must be non-negative")
        if (broker_account is None) != (expected_cash is None):
            raise ValueError(
                "broker_account and expected_cash must be provided together "
                "(cash reconciliation is all-or-nothing)"
            )
        self._expected_positions = expected_positions
        self._broker_positions = broker_positions
        self._expected_cash = expected_cash
        self._broker_account = broker_account
        self._cash_tolerance = cash_tolerance
        self._position_tolerance = position_tolerance
        self._alert = alert
        self._interval = interval_seconds
        self._clock: Clock = clock if clock is not None else _utc_now
        self._sleep: Sleeper = sleep if sleep is not None else _asyncio_sleep
        self._stop_event = stop_event if stop_event is not None else asyncio.Event()
        self._auto_resume = auto_resume
        self._working_orders = working_orders
        self._grace_seconds = float(reconcile_grace_seconds)
        self._approved_drift: dict[InstrumentId, int] = {}
        self._halted = False
        self._last_report: ReconcileReport | None = None
        self._reports: list[ReconcileReport] = []
        self._last_cash_drift: float | None = None
        self._last_working_deltas: dict[InstrumentId, int] = {}
        self._last_grace_active = False
        self._running = False

    # -- introspection ---------------------------------------------------------

    @property
    def halted(self) -> bool:
        """Whether automated rebalancing is halted due to unapproved drift."""
        return self._halted

    @property
    def last_report(self) -> ReconcileReport | None:
        """Most recent reconciliation report."""
        return self._last_report

    @property
    def reports(self) -> tuple[ReconcileReport, ...]:
        """Full reconciliation audit trail."""
        return tuple(self._reports)

    @property
    def last_cash_drift(self) -> float | None:
        """Most recent ``broker_cash - expected_cash`` difference."""
        return self._last_cash_drift

    @property
    def interval_seconds(self) -> float:
        """Configured reconciliation period."""
        return self._interval

    @property
    def is_running(self) -> bool:
        """Whether the ``run_forever`` loop is active."""
        return self._running

    @property
    def reconcile_grace_seconds(self) -> float:
        """Grace window during which in-flight orders may explain position drift."""
        return self._grace_seconds

    @property
    def last_working_deltas(self) -> Mapping[InstrumentId, int]:
        """Signed in-flight quantity per root observed in the latest pass."""
        return MappingProxyType(dict(self._last_working_deltas))

    @property
    def last_grace_active(self) -> bool:
        """Whether the latest pass suppressed drift inside the grace window."""
        return self._last_grace_active

    # -- operator controls -----------------------------------------------------

    def approve_drift(self, drift: Mapping[InstrumentId, int]) -> None:
        """Approve a known signed drift (broker minus expected) for the next pass."""
        for instrument_id, quantity in drift.items():
            if int(quantity) != 0:
                self._approved_drift[instrument_id] = int(quantity)
            else:
                self._approved_drift.pop(instrument_id, None)
        logger.warning("approved position drift: %s", dict(self._approved_drift))

    def clear_approved_drift(self) -> None:
        """Forget every approved drift allowance."""
        self._approved_drift.clear()

    def resume(self) -> None:
        """Release the halt after an operator review."""
        if self._halted:
            logger.warning("reconciliation halt released by operator")
        self._halted = False
        self._approved_drift.clear()

    def stop(self) -> None:
        """Ask the ``run_forever`` loop to stop after its current iteration."""
        self._stop_event.set()

    # -- reconciliation --------------------------------------------------------

    def _resolve_expected(self) -> Mapping[InstrumentId, int]:
        provider = self._expected_positions
        if callable(provider):
            return provider()
        return provider

    def _resolve_expected_cash(self) -> float | None:
        expected = self._expected_cash
        if expected is None:
            return None
        if callable(expected):
            return expected()
        return expected

    async def reconcile(self) -> bool:
        """Reconcile once; return ``True`` when internal and broker state agree."""
        expected_raw = self._resolve_expected()
        expected = {
            instrument_id: int(quantity)
            for instrument_id, quantity in expected_raw.items()
            if int(quantity) != 0
        }
        resolved_working = await self._resolve_working_orders()
        if resolved_working is None:
            # The provider failed: the pass is failed closed (halt + report).
            return False
        working_deltas, working_newest = resolved_working
        broker = await self._broker_positions()
        actual: dict[InstrumentId, int] = {}
        for instrument_id, position in broker.items():
            quantity = int(position.quantity)
            if quantity != 0:
                actual[instrument_id] = quantity
        effective = self._effective_positions(actual, working_deltas)
        now = self._clock()
        drift: dict[InstrumentId, int] = {}
        suppressed: dict[InstrumentId, int] = {}
        grace_active = False
        for instrument_id in sorted(set(expected) | set(actual) | set(working_deltas)):
            difference = effective.get(instrument_id, 0) - expected.get(instrument_id, 0)
            allowed = self._approved_drift.get(instrument_id, 0)
            if abs(difference - allowed) <= self._position_tolerance:
                continue
            if self._within_grace(instrument_id, working_newest, now):
                suppressed[instrument_id] = difference
                grace_active = True
                continue
            drift[instrument_id] = difference
        self._last_grace_active = grace_active
        cash_drift: float | None = None
        cash_message = ""
        if self._broker_account is not None:
            account = await self._broker_account()
            expected_cash = self._resolve_expected_cash()
            if expected_cash is not None:
                cash_drift = account.cash_balance - expected_cash
                self._last_cash_drift = cash_drift
                if abs(cash_drift) > self._cash_tolerance:
                    cash_message = (
                        f"cash drift {cash_drift:+,.2f} exceeds tolerance "
                        f"{self._cash_tolerance:,.2f}"
                    )
        ok = not drift and not cash_message
        if ok:
            if self._auto_resume and self._halted:
                logger.warning("reconciliation clean; auto-resuming automated rebalancing")
                self._halted = False
                self._approved_drift.clear()
            if suppressed:
                message = (
                    "reconciled within the grace window: in-flight working orders may still "
                    "explain "
                    + ", ".join(
                        f"{instrument_id} {difference:+d} contract(s)"
                        for instrument_id, difference in sorted(suppressed.items())
                    )
                    + f" (newest order < {self._grace_seconds:g}s old)"
                )
            else:
                message = "reconciled: internal and broker state agree"
        else:
            self._halted = True
            parts: list[str] = []
            if drift:
                if self._working_orders is None:
                    parts.append(
                        "unapproved position drift: "
                        + ", ".join(
                            f"{instrument_id} expected {expected.get(instrument_id, 0)} "
                            f"actual {actual.get(instrument_id, 0)}"
                            for instrument_id in sorted(drift)
                        )
                    )
                else:
                    parts.append(
                        "unapproved position drift: "
                        + ", ".join(
                            f"{instrument_id} expected {expected.get(instrument_id, 0)} "
                            f"broker {actual.get(instrument_id, 0)} "
                            f"working {working_deltas.get(instrument_id, 0):+d} "
                            f"effective {effective.get(instrument_id, 0)}"
                            for instrument_id in sorted(drift)
                        )
                    )
            if cash_message:
                parts.append(cash_message)
            message = "; ".join(parts)
        report = self._make_report(
            ok=ok,
            expected=expected,
            actual=actual,
            drift=drift,
            message=message,
        )
        self._publish(report)
        return ok

    async def run_forever(self, interval_seconds: float | None = None) -> None:
        """Reconcile every ``interval_seconds`` until :meth:`stop` is called."""
        interval = self._interval if interval_seconds is None else interval_seconds
        if interval <= 0:
            raise ValueError("interval_seconds must be strictly positive")
        self._running = True
        try:
            while not self._stop_event.is_set():
                try:
                    await self.reconcile()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self._handle_failure(exc)
                if self._stop_event.is_set():
                    break
                await self._sleep(interval)
        finally:
            self._running = False

    # -- internals -------------------------------------------------------------

    async def _resolve_working_orders(
        self,
    ) -> tuple[dict[InstrumentId, int], dict[InstrumentId, datetime]] | None:
        """Fetch and aggregate in-flight working orders per root.

        Returns ``(signed_delta_per_root, newest_update_per_root)``, or ``None``
        when the provider failed (the pass is then failed closed exactly like a
        broker-positions failure).
        """
        self._last_working_deltas = {}
        provider = self._working_orders
        if provider is None:
            return ({}, {})
        try:
            snapshots = await provider()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._handle_failure(exc)
            return None
        deltas: dict[InstrumentId, int] = {}
        newest: dict[InstrumentId, datetime] = {}
        for snapshot in snapshots:
            if snapshot.signed_remaining == 0:
                continue
            instrument_id = snapshot.instrument_id
            deltas[instrument_id] = deltas.get(instrument_id, 0) + int(snapshot.signed_remaining)
            previous = newest.get(instrument_id)
            if previous is None or snapshot.updated_at > previous:
                newest[instrument_id] = snapshot.updated_at
        self._last_working_deltas = dict(deltas)
        return (deltas, newest)

    @staticmethod
    def _effective_positions(
        actual: Mapping[InstrumentId, int], working_deltas: Mapping[InstrumentId, int]
    ) -> dict[InstrumentId, int]:
        """Broker positions plus the signed remaining quantity of in-flight orders."""
        effective = dict(actual)
        for instrument_id, delta in working_deltas.items():
            effective[instrument_id] = effective.get(instrument_id, 0) + int(delta)
        return effective

    def _within_grace(
        self,
        instrument_id: InstrumentId,
        newest_update: Mapping[InstrumentId, datetime],
        now: datetime,
    ) -> bool:
        """Whether an in-flight order for ``instrument_id`` is inside the grace window."""
        if self._working_orders is None:
            return False
        updated_at = newest_update.get(instrument_id)
        if updated_at is None:
            return False
        age = (now - updated_at).total_seconds()
        # A small negative age is venue/local clock skew and is treated as fresh;
        # an order timestamped far in the future is not evidence of progress.
        return -self._grace_seconds <= age < self._grace_seconds

    def _handle_failure(self, exc: BaseException) -> None:
        self._halted = True
        self._last_grace_active = False
        message = f"reconciliation failed: {type(exc).__name__}: {exc}"
        logger.exception("reconciliation failure")
        report = ReconcileReport(
            timestamp=self._clock(),
            ok=False,
            halted=True,
            message=message,
        )
        self._publish(report)

    def _make_report(
        self,
        *,
        ok: bool,
        expected: Mapping[InstrumentId, int],
        actual: Mapping[InstrumentId, int],
        drift: Mapping[InstrumentId, int],
        message: str,
    ) -> ReconcileReport:
        return ReconcileReport(
            timestamp=self._clock(),
            ok=ok,
            expected_positions=dict(expected),
            actual_positions=dict(actual),
            drift=dict(drift),
            halted=self._halted,
            message=message,
        )

    def _publish(self, report: ReconcileReport) -> None:
        self._last_report = report
        self._reports.append(report)
        if not report.ok:
            logger.error("reconciliation drift: %s", report.message)
            alert = self._alert
            if alert is not None:
                try:
                    alert(report)
                except Exception:  # pragma: no cover - alert sink must never crash the loop
                    logger.exception("reconciliation alert callback failed")
