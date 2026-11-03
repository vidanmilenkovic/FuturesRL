"""v1.5.1 (task-38) unit tests: IBKR live NLV history + ex-ante attribution.

Offline and deterministic: the persisted history store, the IBKR book refresh and
the LiveStateService enrichment are exercised with injected clocks/clients and
synthetic market states. No network, no TWS.
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from configs.settings import Settings
from src.api import fixtures
from src.api.live import (
    EQUITY_HISTORY_FILE_NAME,
    MIN_LIVE_CURVE_POINTS,
    LiveEquityHistoryStore,
    LiveStateService,
    ex_ante_portfolio_volatility,
)
from src.api.schemas import AccountSnapshotResponse, EquityPoint, PositionRow
from src.api.services import IBKRExecutionService
from src.core.types import AssetClass
from src.execution.ibkr_client import IBKRConnectionManager

from tests.fixtures.market import FIXED_TIMESTAMP, make_market_state
from tests.unit.test_ibkr_client import FakeClock, FakeIBClient, make_account_rows

LIVE_NLV = 1_234_567.89


# ---------------------------------------------------------------------------
# Fakes / helpers
# ---------------------------------------------------------------------------


class MutableClock:
    """Deterministic wall clock with explicit advancement."""

    def __init__(self, start: datetime = FIXED_TIMESTAMP) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now = self.now + timedelta(seconds=seconds)


class FakeMarket:
    """Minimal MarketDataService stand-in exposing ``latest_state``."""

    def __init__(self, state: object | None) -> None:
        self._state = state
        self.calls = 0

    def latest_state(self) -> object:
        self.calls += 1
        if self._state is None:
            raise RuntimeError("no market state")
        return self._state


def _point(minutes: int, value: float = 1_000_000.0) -> EquityPoint:
    return EquityPoint(time=FIXED_TIMESTAMP + timedelta(minutes=minutes), value=value)


def _position_row(
    root: str = "ES",
    *,
    weight: float = 0.30,
    quantity: int = 2,
    market_price: float = 4_800.0,
    multiplier: float = 50.0,
) -> PositionRow:
    notional = quantity * market_price * multiplier
    return PositionRow(
        instrument_id=root,
        root_symbol=root,
        asset_class=AssetClass.EQUITY_INDEX,
        contract_symbol=f"{root}Z6",
        expiration=date(2026, 12, 18),
        days_to_roll=45,
        quantity=quantity,
        average_price=market_price,
        market_price=market_price,
        multiplier=multiplier,
        notional=notional,
        weight=weight,
        unrealized_pnl=0.0,
        initial_margin=1_500.0,
        maintenance_margin=1_400.0,
    )


def _snapshot(
    *,
    positions: list[PositionRow] | None = None,
    curve: list[EquityPoint] | None = None,
    net_liquidation: float = 1_000_000.0,
) -> AccountSnapshotResponse:
    rows = positions if positions is not None else []
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
        gross_leverage=0.3,
        net_exposure=net_liquidation * 0.3,
        equity_curve=curve if curve is not None else [_point(-120), _point(-60)],
        positions=rows,
        source="engine",
    )


class FakeIBKRExecution:
    """Execution adapter stand-in for the LiveStateService attribution tests."""

    def __init__(
        self,
        snapshot: AccountSnapshotResponse,
        points: list[EquityPoint] | None = None,
        *,
        ibkr_enabled: bool = True,
    ) -> None:
        self._snapshot = snapshot
        self._points = list(points or [])
        self.ibkr_enabled = ibkr_enabled
        self.source_name = "engine" if ibkr_enabled else "simulated"

    async def account(self) -> AccountSnapshotResponse:
        return self._snapshot

    def live_equity_points(self) -> tuple[EquityPoint, ...]:
        return tuple(self._points)


def _live_service(
    settings: Settings,
    execution: FakeIBKRExecution,
    market_state: object | None,
) -> LiveStateService:
    return LiveStateService(
        settings=settings,
        clock=lambda: FIXED_TIMESTAMP,
        execution=execution,
        instrument_master=settings.instruments,
        market=cast(Any, FakeMarket(market_state)),
    )


# ---------------------------------------------------------------------------
# Durable live IBKR NLV history
# ---------------------------------------------------------------------------


def test_history_append_reload_and_dedupe(tmp_path: Path) -> None:
    clock = MutableClock()
    path = tmp_path / "ibkr_equity_history.json"
    store = LiveEquityHistoryStore(path, clock=clock)

    assert store.load() == []
    assert [point.value for point in store.append(_point(0, 1_000_000.0))] == [1_000_000.0]
    store.append(_point(1, 1_010_000.0))

    # Same timestamp dedupes (latest observed value wins, no duplicate row).
    store.append(_point(0, 999_000.0))
    points = store.load()
    assert len(points) == 2
    assert points[0].value == pytest.approx(999_000.0)
    assert points[1].value == pytest.approx(1_010_000.0)

    # A fresh store instance reloads the persisted document.
    reloaded = LiveEquityHistoryStore(path, clock=clock)
    assert [(point.time, point.value) for point in reloaded.load()] == [
        (point.time, point.value) for point in points
    ]
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert len(payload["points"]) == 2
    assert not list(tmp_path.glob("*.tmp"))


def test_history_is_bounded_by_point_count(tmp_path: Path) -> None:
    clock = MutableClock()
    store = LiveEquityHistoryStore(tmp_path / "history.json", clock=clock, max_points=5)
    for index in range(8):
        clock.advance(60)
        store.append(EquityPoint(time=clock(), value=1_000_000.0 + index))
    points = store.load()
    assert len(points) == 5
    assert [point.value for point in points] == [
        1_000_003.0,
        1_000_004.0,
        1_000_005.0,
        1_000_006.0,
        1_000_007.0,
    ]


def test_history_is_bounded_by_age(tmp_path: Path) -> None:
    clock = MutableClock()
    store = LiveEquityHistoryStore(tmp_path / "history.json", clock=clock, max_age_days=30.0)
    stale = EquityPoint(time=FIXED_TIMESTAMP - timedelta(days=40), value=1_000_000.0)
    store.append(stale)
    assert store.load() == []
    clock.advance(60)
    store.append(EquityPoint(time=clock(), value=2_000_000.0))
    assert [point.value for point in store.load()] == [2_000_000.0]


def test_history_corruption_is_quarantined_not_fatal(tmp_path: Path) -> None:
    clock = MutableClock()
    path = tmp_path / "history.json"
    path.write_text("{ this is not valid json", encoding="utf-8")
    store = LiveEquityHistoryStore(path, clock=clock)
    assert store.load() == []
    quarantined = list(tmp_path.glob("history.json.corrupt-*"))
    assert len(quarantined) == 1
    # Appending after quarantine recovers with a clean document.
    store.append(_point(0, 1_500_000.0))
    assert [point.value for point in store.load()] == [1_500_000.0]


def test_history_accepts_a_directory_path(tmp_path: Path) -> None:
    store = LiveEquityHistoryStore(tmp_path, clock=MutableClock())
    assert store.path.name == EQUITY_HISTORY_FILE_NAME
    store.clear()
    assert store.path.name == EQUITY_HISTORY_FILE_NAME


async def test_ibkr_service_persists_observed_nlv_points(
    settings: Settings, tmp_path: Path
) -> None:
    clock = FakeClock(FIXED_TIMESTAMP)
    client = FakeIBClient()
    client.account_rows = make_account_rows(net_liquidation=str(LIVE_NLV))
    store = LiveEquityHistoryStore(tmp_path / "ibkr_equity_history.json", clock=clock)
    manager = IBKRConnectionManager(
        settings,
        client_factory=lambda: client,
        clock=clock,
        sleep=clock.sleep,
        random_source=lambda: 0.5,
        heartbeat_enabled=False,
    )
    market = fixtures.build_offline_market(settings.instruments, clock=clock)
    service = IBKRExecutionService(
        settings=settings,
        market=market,
        clock=clock,
        connection_manager=manager,
        equity_history_store=store,
    )

    assert await service.connect() is True
    await service.account()
    clock.advance(60)
    await service.account()
    clock.advance(60)
    await service.account()

    points = service.live_equity_points
    assert len(points) == 3
    assert [point.value for point in points] == pytest.approx([LIVE_NLV, LIVE_NLV, LIVE_NLV])
    assert service.equity_history_path == store.path
    await service.disconnect()


# ---------------------------------------------------------------------------
# LiveStateService attribution: curve provenance + ex-ante volatility
# ---------------------------------------------------------------------------


async def test_below_threshold_curve_is_marked_simulated_backfill(settings: Settings) -> None:
    backfill = [_point(-180), _point(-120), _point(-60)]
    snapshot = _snapshot(curve=backfill)
    live_points = [_point(index, 1_100_000.0) for index in range(MIN_LIVE_CURVE_POINTS - 2)]
    service = _live_service(settings, FakeIBKRExecution(snapshot, live_points), make_market_state())

    response = await service.account()

    assert response.equity_curve_source == "simulated-backfill"
    assert response.equity_curve_live_points == len(live_points)
    # The existing synthetic backfill curve is kept (never relabelled live).
    assert [point.value for point in response.equity_curve] == [point.value for point in backfill]


async def test_threshold_curve_is_marked_ibkr_live(settings: Settings) -> None:
    backfill = [_point(-180), _point(-120), _point(-60)]
    live_points = [_point(index, 1_200_000.0 + index) for index in range(MIN_LIVE_CURVE_POINTS + 2)]
    service = _live_service(
        settings,
        FakeIBKRExecution(_snapshot(curve=backfill), live_points),
        make_market_state(),
    )

    response = await service.account()

    assert response.equity_curve_source == "ibkr-live"
    assert response.equity_curve_live_points == len(live_points)
    assert len(response.equity_curve) == len(live_points)
    assert [point.value for point in response.equity_curve] == [
        point.value for point in live_points
    ]


async def test_flat_ibkr_book_has_zero_portfolio_volatility(settings: Settings) -> None:
    service = _live_service(settings, FakeIBKRExecution(_snapshot(positions=[])), None)
    response = await service.account()
    assert response.portfolio_volatility == 0.0
    assert response.portfolio_volatility_source == "ex-ante-book"


async def test_portfolio_volatility_uses_book_weights_and_covariance(
    settings: Settings,
) -> None:
    state = make_market_state(roots=("ES", "NQ"), with_covariance=True)
    weights = {"ES": 0.30, "NQ": -0.20}
    positions = [
        _position_row("ES", weight=weights["ES"]),
        _position_row("NQ", weight=weights["NQ"]),
    ]
    service = _live_service(settings, FakeIBKRExecution(_snapshot(positions=positions)), state)

    response = await service.account()

    covariance = state.covariance_submatrix(("ES", "NQ"))
    assert covariance is not None
    vector = np.array([weights["ES"], weights["NQ"]], dtype=np.float64)
    expected = math.sqrt(float(vector @ covariance @ vector))
    assert response.portfolio_volatility == pytest.approx(expected)
    assert response.portfolio_volatility_source == "ex-ante-book"
    # Annualised fraction, not a period return figure.
    assert 0.0 < cast(float, response.portfolio_volatility) < 1.0


async def test_portfolio_volatility_unavailable_without_covariance(settings: Settings) -> None:
    state = make_market_state(roots=("ES", "NQ"), with_covariance=False)
    positions = [_position_row("ES", weight=0.30)]
    service = _live_service(settings, FakeIBKRExecution(_snapshot(positions=positions)), state)
    response = await service.account()
    assert response.portfolio_volatility is None
    assert response.portfolio_volatility_source == "unavailable"


async def test_offline_account_keeps_existing_semantics(settings: Settings) -> None:
    snapshot = _snapshot(curve=[_point(-120), _point(-60)])
    offline_execution = FakeIBKRExecution(
        snapshot, [_point(index) for index in range(6)], ibkr_enabled=False
    )
    service = _live_service(settings, offline_execution, make_market_state())

    response = await service.account()

    assert response.source == "simulated"
    assert response.equity_curve_source == "fixture"
    assert response.equity_curve_live_points == 0
    assert response.portfolio_volatility is None
    assert response.portfolio_volatility_source is None
    assert [point.value for point in response.equity_curve] == [
        point.value for point in snapshot.equity_curve
    ]


def test_ex_ante_portfolio_volatility_helper() -> None:
    assert ex_ante_portfolio_volatility({}, None) == (0.0, "ex-ante-book")
    assert ex_ante_portfolio_volatility({"ES": 0.0}, make_market_state()) == (
        0.0,
        "ex-ante-book",
    )
    state = make_market_state(roots=("ES",), with_covariance=True)
    variance = float(state.covariance_submatrix(("ES",))[0, 0])  # type: ignore[index]
    volatility, source = ex_ante_portfolio_volatility({"ES": 0.5}, state)
    assert source == "ex-ante-book"
    assert volatility == pytest.approx(math.sqrt(0.25 * variance))
    missing, missing_source = ex_ante_portfolio_volatility(
        {"ES": 0.5}, make_market_state(roots=("ES",), with_covariance=False)
    )
    assert missing is None
    assert missing_source == "unavailable"
    unknown_root = make_market_state(roots=("ES",), with_covariance=True)
    assert ex_ante_portfolio_volatility({"ZZ": 0.5}, unknown_root) == (None, "unavailable")


async def test_market_provider_failure_marks_volatility_unavailable(settings: Settings) -> None:
    positions = [_position_row("ES", weight=0.30)]
    execution = FakeIBKRExecution(_snapshot(positions=positions))
    service = LiveStateService(
        settings=settings,
        clock=lambda: FIXED_TIMESTAMP,
        execution=execution,
        instrument_master=settings.instruments,
        market=cast(Any, FakeMarket(None)),  # latest_state() raises
    )

    response = await service.account()
    assert response.portfolio_volatility is None
    assert response.portfolio_volatility_source == "unavailable"
