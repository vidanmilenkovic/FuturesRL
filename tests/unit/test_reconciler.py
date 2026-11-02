"""Unit tests for periodic internal-vs-broker reconciliation (offline, deterministic)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime, timedelta

import pytest
from src.core.interfaces import PositionReconciler as PositionReconcilerProtocol
from src.core.types import AccountSnapshot, Position
from src.execution.reconciler import PositionReconciler, WorkingOrderSnapshot

from tests.fixtures.market import FIXED_TIMESTAMP, make_account_snapshot
from tests.unit.test_ibkr_client import FakeClock


def make_position(
    instrument_id: str, quantity: int, *, price: float = 4800.0, multiplier: float = 50.0
) -> Position:
    return Position(
        instrument_id=instrument_id,
        quantity=quantity,
        average_price=price,
        market_price=price,
        multiplier=multiplier,
    )


def make_positions(*rows: tuple[str, int]) -> dict[str, Position]:
    return {
        instrument_id: make_position(instrument_id, quantity) for instrument_id, quantity in rows
    }


def make_reconciler(
    clock: FakeClock,
    *,
    expected: object = None,
    broker_positions: Callable[[], Awaitable[dict[str, Position]]] | None = None,
    expected_cash: float | Callable[[], float] | None = None,
    broker_account: Callable[[], Awaitable[AccountSnapshot]] | None = None,
    alert: Callable[[object], None] | None = None,
    **kwargs: object,
) -> PositionReconciler:
    expected_positions = expected if expected is not None else {"ES": 2}
    provider = broker_positions if broker_positions is not None else static_positions(("ES", 2))
    return PositionReconciler(
        expected_positions=expected_positions,  # type: ignore[arg-type]
        broker_positions=provider,
        expected_cash=expected_cash,
        broker_account=broker_account,
        alert=alert,  # type: ignore[arg-type]
        clock=clock,
        sleep=clock.sleep,
        **kwargs,  # type: ignore[arg-type]
    )


def static_positions(*rows: tuple[str, int]) -> Callable[[], Awaitable[dict[str, Position]]]:
    async def provider() -> dict[str, Position]:
        return make_positions(*rows)

    return provider


def failing_positions() -> Callable[[], Awaitable[dict[str, Position]]]:
    async def provider() -> dict[str, Position]:
        raise ConnectionError("positions unavailable")

    return provider


@pytest.fixture()
def clock() -> FakeClock:
    return FakeClock(FIXED_TIMESTAMP)


# ---------------------------------------------------------------------------
# Single reconciliation passes
# ---------------------------------------------------------------------------


async def test_reconcile_clean_state(clock: FakeClock) -> None:
    reconciler = make_reconciler(
        clock,
        expected={"ES": 2, "CL": -1},
        broker_positions=static_positions(("ES", 2), ("CL", -1)),
    )
    assert await reconciler.reconcile()
    report = reconciler.last_report
    assert report is not None
    assert report.ok
    assert not report.halted
    assert report.drift == {}
    assert report.expected_positions == {"ES": 2, "CL": -1}
    assert report.actual_positions == {"ES": 2, "CL": -1}
    assert report.timestamp == clock.now
    assert "agree" in report.message
    assert len(reconciler.reports) == 1
    assert not reconciler.halted


async def test_reconcile_detects_unapproved_drift_and_alerts(clock: FakeClock) -> None:
    alerts: list[object] = []
    reconciler = make_reconciler(
        clock,
        expected={"ES": 2},
        broker_positions=static_positions(("ES", 3)),
        alert=alerts.append,
    )
    assert not await reconciler.reconcile()
    assert reconciler.halted
    report = reconciler.last_report
    assert report is not None
    assert not report.ok
    assert report.halted
    assert report.drift == {"ES": 1}
    assert "unapproved position drift" in report.message
    assert "expected 2 actual 3" in report.message
    assert len(alerts) == 1
    assert alerts[0] is report


async def test_reconcile_detects_missing_and_extra_positions(clock: FakeClock) -> None:
    reconciler = make_reconciler(
        clock,
        expected={"ES": 2, "NQ": 1},
        broker_positions=static_positions(("CL", -2)),
    )
    assert not await reconciler.reconcile()
    assert reconciler.last_report is not None
    assert reconciler.last_report.drift == {"CL": -2, "ES": -2, "NQ": -1}


async def test_reconcile_ignores_zero_broker_positions(clock: FakeClock) -> None:
    reconciler = make_reconciler(
        clock,
        expected={"ES": 2},
        broker_positions=static_positions(("ES", 2), ("GC", 0)),
    )
    assert await reconciler.reconcile()
    assert reconciler.last_report is not None
    assert "GC" not in reconciler.last_report.actual_positions


async def test_approved_drift_is_tolerated_until_cleared(clock: FakeClock) -> None:
    reconciler = make_reconciler(
        clock,
        expected={"ES": 2},
        broker_positions=static_positions(("ES", 3)),
    )
    reconciler.approve_drift({"ES": 1})
    assert await reconciler.reconcile()
    assert not reconciler.halted
    reconciler.clear_approved_drift()
    assert not await reconciler.reconcile()
    assert reconciler.halted


async def test_position_tolerance_allows_small_integer_drift(clock: FakeClock) -> None:
    tolerated = make_reconciler(
        clock,
        expected={"ES": 2},
        broker_positions=static_positions(("ES", 3)),
        position_tolerance=1,
    )
    assert await tolerated.reconcile()
    strict = make_reconciler(
        clock,
        expected={"ES": 2},
        broker_positions=static_positions(("ES", 3)),
    )
    assert not await strict.reconcile()


async def test_cash_reconciliation(clock: FakeClock) -> None:
    async def account(cash: float) -> AccountSnapshot:
        return AccountSnapshot(
            timestamp=FIXED_TIMESTAMP,
            net_liquidation=1_000_000.0,
            cash_balance=cash,
        )

    async def clean_account() -> AccountSnapshot:
        return await account(500_000.0)

    async def drifted_account() -> AccountSnapshot:
        return await account(500_100.0)

    clean = make_reconciler(
        clock,
        broker_account=clean_account,
        expected_cash=500_000.0,
    )
    assert await clean.reconcile()
    assert clean.last_cash_drift == pytest.approx(0.0)

    drifted = make_reconciler(
        clock,
        broker_account=drifted_account,
        expected_cash=500_000.0,
        cash_tolerance=1.0,
    )
    assert not await drifted.reconcile()
    assert drifted.halted
    assert drifted.last_cash_drift == pytest.approx(100.0)
    assert drifted.last_report is not None
    assert "cash drift" in drifted.last_report.message

    tolerated = make_reconciler(
        clock,
        broker_account=drifted_account,
        expected_cash=500_000.0,
        cash_tolerance=100.0,
    )
    assert await tolerated.reconcile()
    assert not tolerated.halted


async def test_dynamic_expected_positions_provider(clock: FakeClock) -> None:
    state: dict[str, int] = {"ES": 2}

    def expected_provider() -> dict[str, int]:
        return dict(state)

    reconciler = make_reconciler(
        clock,
        expected=expected_provider,
        broker_positions=static_positions(("ES", 2)),
    )
    assert await reconciler.reconcile()
    state["ES"] = 1
    assert not await reconciler.reconcile()
    assert reconciler.last_report is not None
    assert reconciler.last_report.drift == {"ES": 1}


# ---------------------------------------------------------------------------
# Halt lifecycle
# ---------------------------------------------------------------------------


async def test_halt_is_sticky_until_operator_resume(clock: FakeClock) -> None:
    positions: dict[str, int] = {"ES": 3}

    async def broker() -> dict[str, Position]:
        return make_positions(("ES", positions["ES"]))

    reconciler = make_reconciler(clock, expected={"ES": 2}, broker_positions=broker)
    assert not await reconciler.reconcile()
    assert reconciler.halted
    positions["ES"] = 2
    assert await reconciler.reconcile()  # clean again ...
    assert reconciler.halted  # ... but the halt is sticky
    reconciler.resume()
    assert not reconciler.halted
    assert await reconciler.reconcile()


async def test_auto_resume_clears_halt_on_clean_reconcile(clock: FakeClock) -> None:
    positions: dict[str, int] = {"ES": 3}

    async def broker() -> dict[str, Position]:
        return make_positions(("ES", positions["ES"]))

    reconciler = make_reconciler(
        clock, expected={"ES": 2}, broker_positions=broker, auto_resume=True
    )
    assert not await reconciler.reconcile()
    assert reconciler.halted
    positions["ES"] = 2
    assert await reconciler.reconcile()
    assert not reconciler.halted


# ---------------------------------------------------------------------------
# Async loop
# ---------------------------------------------------------------------------


async def test_run_forever_reconciles_until_stopped(clock: FakeClock) -> None:
    calls = {"count": 0}
    reconciler = make_reconciler(
        clock,
        expected={"ES": 2},
        broker_positions=static_positions(("ES", 2)),
    )

    async def stop_after_two(_seconds: float) -> None:
        calls["count"] += 1
        clock.sleeps.append(_seconds)
        if calls["count"] >= 2:
            reconciler.stop()

    reconciler._sleep = stop_after_two  # type: ignore[attr-defined]
    await reconciler.run_forever(interval_seconds=7.5)
    assert calls["count"] == 2
    assert len(reconciler.reports) == 2
    assert clock.sleeps == [7.5, 7.5]
    assert not reconciler.is_running


async def test_run_forever_halts_and_alerts_on_broker_failure(clock: FakeClock) -> None:
    alerts: list[object] = []
    reconciler = make_reconciler(
        clock,
        broker_positions=failing_positions(),
        alert=alerts.append,
    )

    async def stop_after_one(_seconds: float) -> None:
        reconciler.stop()

    reconciler._sleep = stop_after_one  # type: ignore[attr-defined]
    await reconciler.run_forever(interval_seconds=1.0)
    assert reconciler.halted
    assert reconciler.last_report is not None
    assert not reconciler.last_report.ok
    assert "reconciliation failed" in reconciler.last_report.message
    assert len(alerts) == 1


async def test_run_forever_rejects_non_positive_interval(clock: FakeClock) -> None:
    reconciler = make_reconciler(
        clock,
        expected={"ES": 2},
        broker_positions=static_positions(("ES", 2)),
    )
    with pytest.raises(ValueError):
        await reconciler.run_forever(interval_seconds=0.0)


async def test_run_forever_is_cancellable(clock: FakeClock) -> None:
    import asyncio

    reconciler = make_reconciler(
        clock,
        expected={"ES": 2},
        broker_positions=static_positions(("ES", 2)),
    )

    async def real_sleep(seconds: float) -> None:
        await asyncio.sleep(seconds)

    reconciler._sleep = real_sleep  # type: ignore[attr-defined]
    task = asyncio.create_task(reconciler.run_forever(interval_seconds=1000.0))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not reconciler.is_running
    assert reconciler.reports


async def test_run_forever_with_preset_stop_event(clock: FakeClock) -> None:
    reconciler = make_reconciler(
        clock,
        expected={"ES": 2},
        broker_positions=static_positions(("ES", 2)),
    )
    reconciler.stop()
    await reconciler.run_forever()
    assert reconciler.reports == ()
    assert reconciler.interval_seconds == 30.0


# ---------------------------------------------------------------------------
# Validation / protocol conformance
# ---------------------------------------------------------------------------


def test_reconciler_validation(clock: FakeClock) -> None:
    with pytest.raises(ValueError):
        make_reconciler(clock, interval_seconds=0.0)
    with pytest.raises(ValueError):
        make_reconciler(clock, cash_tolerance=-1.0)
    with pytest.raises(ValueError):
        make_reconciler(clock, position_tolerance=-1)
    with pytest.raises(ValueError):
        make_reconciler(clock, expected_cash=1000.0)  # account provider missing
    with pytest.raises(ValueError):
        make_reconciler(
            clock,
            broker_account=_make_account_provider(),
        )  # expected_cash missing
    with pytest.raises(ValueError):
        make_reconciler(clock, interval_seconds=-1.0)


def test_resume_is_noop_when_not_halted(clock: FakeClock) -> None:
    reconciler = make_reconciler(
        clock,
        expected={"ES": 2},
        broker_positions=static_positions(("ES", 2)),
    )
    reconciler.resume()
    assert not reconciler.halted


async def test_reconciler_satisfies_protocol(clock: FakeClock) -> None:
    reconciler = make_reconciler(
        clock,
        expected={"ES": 2},
        broker_positions=static_positions(("ES", 2)),
    )
    assert isinstance(reconciler, PositionReconcilerProtocol)
    assert await reconciler.reconcile()
    assert await reconciler.reconcile()
    assert len(reconciler.reports) == 2


def _make_account_provider() -> Callable[[], Awaitable[AccountSnapshot]]:
    async def provider() -> AccountSnapshot:
        return make_account_snapshot()

    return provider


async def test_reconcile_uses_injected_clock_timestamp(clock: FakeClock) -> None:
    clock.now = datetime(2024, 6, 3, 14, 30, tzinfo=UTC)
    reconciler = make_reconciler(
        clock,
        expected={"ES": 1},
        broker_positions=static_positions(("ES", 1)),
    )
    await reconciler.reconcile()
    assert reconciler.last_report is not None
    assert reconciler.last_report.timestamp == datetime(2024, 6, 3, 14, 30, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Open-order-aware drift (Phase 7)
# ---------------------------------------------------------------------------


def working_orders(
    *snapshots: WorkingOrderSnapshot,
) -> Callable[[], Awaitable[Sequence[WorkingOrderSnapshot]]]:
    """Static working-order provider returning ``snapshots``."""

    async def provider() -> Sequence[WorkingOrderSnapshot]:
        return snapshots

    return provider


def order(
    instrument_id: str,
    signed_remaining: int,
    *,
    updated_at: datetime,
    order_id: str = "wo-1",
) -> WorkingOrderSnapshot:
    return WorkingOrderSnapshot(
        order_id=order_id,
        instrument_id=instrument_id,
        signed_remaining=signed_remaining,
        updated_at=updated_at,
        status="Submitted",
    )


async def test_open_order_exactly_explains_drift(clock: FakeClock) -> None:
    reconciler = make_reconciler(
        clock,
        expected={"ES": 2},
        broker_positions=static_positions(("ES", 1)),
        working_orders=working_orders(order("ES", 1, updated_at=clock.now)),
    )
    assert await reconciler.reconcile()
    assert not reconciler.halted
    assert reconciler.last_working_deltas == {"ES": 1}
    assert reconciler.last_grace_active is False
    report = reconciler.last_report
    assert report is not None and report.ok and report.drift == {}
    assert "agree" in report.message


async def test_partial_fill_remaining_explains_drift(clock: FakeClock) -> None:
    # 2 of 4 contracts already filled: broker shows 1 held, 2 still working.
    reconciler = make_reconciler(
        clock,
        expected={"ES": 3},
        broker_positions=static_positions(("ES", 1)),
        working_orders=working_orders(order("ES", 2, updated_at=clock.now, order_id="wo-partial")),
    )
    assert await reconciler.reconcile()
    assert not reconciler.halted
    assert reconciler.last_working_deltas == {"ES": 2}
    assert reconciler.last_report is not None
    assert reconciler.last_report.drift == {}


async def test_multiple_working_orders_accumulate(clock: FakeClock) -> None:
    reconciler = make_reconciler(
        clock,
        expected={"ES": 3},
        broker_positions=static_positions(),
        working_orders=working_orders(
            order("ES", 1, updated_at=clock.now, order_id="wo-a"),
            order("ES", 2, updated_at=clock.now, order_id="wo-b"),
        ),
    )
    assert await reconciler.reconcile()
    assert reconciler.last_working_deltas == {"ES": 3}
    assert not reconciler.halted


async def test_zero_remaining_working_order_is_ignored(clock: FakeClock) -> None:
    reconciler = make_reconciler(
        clock,
        expected={"ES": 2},
        broker_positions=static_positions(("ES", 2)),
        working_orders=working_orders(order("ES", 0, updated_at=clock.now)),
    )
    assert await reconciler.reconcile()
    assert reconciler.last_working_deltas == {}
    assert reconciler.last_grace_active is False


async def test_grace_window_suppresses_unexplained_drift(clock: FakeClock) -> None:
    reconciler = make_reconciler(
        clock,
        expected={"ES": 2},
        broker_positions=static_positions(("ES", 1)),
        working_orders=working_orders(order("ES", 2, updated_at=clock.now - timedelta(seconds=10))),
        reconcile_grace_seconds=30.0,
    )
    assert await reconciler.reconcile()
    assert not reconciler.halted
    assert reconciler.last_grace_active is True
    assert reconciler.last_working_deltas == {"ES": 2}
    report = reconciler.last_report
    assert report is not None
    assert report.ok and report.drift == {}
    assert "grace window" in report.message
    assert reconciler.reconcile_grace_seconds == 30.0


async def test_stale_working_order_beyond_grace_is_drift(clock: FakeClock) -> None:
    alerts: list[object] = []
    reconciler = make_reconciler(
        clock,
        expected={"ES": 2},
        broker_positions=static_positions(("ES", 1)),
        working_orders=working_orders(order("ES", 2, updated_at=clock.now - timedelta(seconds=60))),
        reconcile_grace_seconds=30.0,
        alert=alerts.append,
    )
    assert not await reconciler.reconcile()
    assert reconciler.halted
    assert reconciler.last_grace_active is False
    report = reconciler.last_report
    assert report is not None
    assert report.drift == {"ES": 1}
    assert "working +2" in report.message
    assert "effective 3" in report.message
    assert len(alerts) == 1


async def test_wrong_direction_working_delta_is_drift(clock: FakeClock) -> None:
    reconciler = make_reconciler(
        clock,
        expected={"ES": 2},
        broker_positions=static_positions(("ES", 1)),
        working_orders=working_orders(order("ES", -1, updated_at=clock.now)),
        reconcile_grace_seconds=0.0,
    )
    assert not await reconciler.reconcile()
    assert reconciler.halted
    report = reconciler.last_report
    assert report is not None
    assert report.drift == {"ES": -2}
    assert "working -1" in report.message


async def test_working_orders_provider_failure_halts(clock: FakeClock) -> None:
    alerts: list[object] = []

    async def failing() -> Sequence[WorkingOrderSnapshot]:
        raise ConnectionError("open orders unavailable")

    reconciler = make_reconciler(
        clock,
        expected={"ES": 2},
        broker_positions=static_positions(("ES", 2)),
        working_orders=failing,
        alert=alerts.append,
    )
    assert not await reconciler.reconcile()
    assert reconciler.halted
    assert reconciler.last_working_deltas == {}
    report = reconciler.last_report
    assert report is not None
    assert not report.ok
    assert "reconciliation failed" in report.message
    assert "open orders unavailable" in report.message
    assert len(alerts) == 1


async def test_run_forever_halts_on_working_orders_provider_failure(clock: FakeClock) -> None:
    async def failing() -> Sequence[WorkingOrderSnapshot]:
        raise ConnectionError("open orders unavailable")

    async def stop_after_one(_seconds: float) -> None:
        reconciler.stop()

    reconciler = make_reconciler(
        clock,
        working_orders=failing,
    )
    reconciler._sleep = stop_after_one  # type: ignore[attr-defined]
    await reconciler.run_forever(1.0)
    assert reconciler.halted
    assert reconciler.last_report is not None
    assert not reconciler.last_report.ok


async def test_approved_drift_uses_effective_positions(clock: FakeClock) -> None:
    reconciler = make_reconciler(
        clock,
        expected={"ES": 2},
        broker_positions=static_positions(("ES", 2)),
        working_orders=working_orders(order("ES", 1, updated_at=clock.now)),
        reconcile_grace_seconds=0.0,
    )
    reconciler.approve_drift({"ES": 1})
    assert await reconciler.reconcile()
    assert not reconciler.halted


async def test_no_provider_preserves_legacy_semantics(clock: FakeClock) -> None:
    reconciler = make_reconciler(
        clock, expected={"ES": 2}, broker_positions=static_positions(("ES", 3))
    )
    assert not await reconciler.reconcile()
    assert reconciler.last_working_deltas == {}
    assert reconciler.last_grace_active is False
    report = reconciler.last_report
    assert report is not None
    assert report.drift == {"ES": 1}
    assert "actual 3" in report.message
    assert "working" not in report.message


def test_reconcile_grace_seconds_validation(clock: FakeClock) -> None:
    with pytest.raises(ValueError):
        make_reconciler(
            clock,
            expected={"ES": 2},
            broker_positions=static_positions(("ES", 2)),
            reconcile_grace_seconds=-1.0,
        )


def test_working_order_snapshot_validation() -> None:
    with pytest.raises(ValueError):
        WorkingOrderSnapshot(
            order_id="", instrument_id="ES", signed_remaining=1, updated_at=FIXED_TIMESTAMP
        )
    with pytest.raises(ValueError):
        WorkingOrderSnapshot(
            order_id="a", instrument_id="", signed_remaining=1, updated_at=FIXED_TIMESTAMP
        )
    with pytest.raises(TypeError):
        WorkingOrderSnapshot(
            order_id="a",
            instrument_id="ES",
            signed_remaining=True,  # type: ignore[arg-type]
            updated_at=FIXED_TIMESTAMP,
        )
    with pytest.raises(ValueError):
        WorkingOrderSnapshot(
            order_id="a",
            instrument_id="ES",
            signed_remaining=1,
            updated_at=datetime(2024, 1, 2, 21, 0),  # naive
        )
    snapshot = WorkingOrderSnapshot(
        order_id="a",
        instrument_id="ES",
        signed_remaining=-2,
        updated_at=datetime(2024, 1, 2, 21, 0, tzinfo=UTC),
    )
    assert snapshot.updated_at.tzinfo is not None
