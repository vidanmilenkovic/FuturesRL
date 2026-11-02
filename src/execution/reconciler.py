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

The reconciler is fully offline-testable: every I/O source is an injected
callable and both clock and sleep are injectable, so the async ``run_forever``
loop is deterministic under test.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from src.core.types import (
    AccountSnapshot,
    InstrumentId,
    Position,
    ReconcileReport,
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
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be strictly positive")
        if cash_tolerance < 0:
            raise ValueError("cash_tolerance must be non-negative")
        if position_tolerance < 0:
            raise ValueError("position_tolerance must be non-negative")
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
        self._approved_drift: dict[InstrumentId, int] = {}
        self._halted = False
        self._last_report: ReconcileReport | None = None
        self._reports: list[ReconcileReport] = []
        self._last_cash_drift: float | None = None
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
        broker = await self._broker_positions()
        actual: dict[InstrumentId, int] = {}
        for instrument_id, position in broker.items():
            quantity = int(position.quantity)
            if quantity != 0:
                actual[instrument_id] = quantity
        drift: dict[InstrumentId, int] = {}
        for instrument_id in sorted(set(expected) | set(actual)):
            difference = actual.get(instrument_id, 0) - expected.get(instrument_id, 0)
            allowed = self._approved_drift.get(instrument_id, 0)
            if abs(difference - allowed) > self._position_tolerance:
                drift[instrument_id] = difference
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
            message = "reconciled: internal and broker state agree"
        else:
            self._halted = True
            parts: list[str] = []
            if drift:
                parts.append(
                    "unapproved position drift: "
                    + ", ".join(
                        f"{instrument_id} expected {expected.get(instrument_id, 0)} "
                        f"actual {actual.get(instrument_id, 0)}"
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

    def _handle_failure(self, exc: BaseException) -> None:
        self._halted = True
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
