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
    MIN_LIVE_CURVE_DAYS,
    MIN_LIVE_CURVE_POINTS,
    LiveEquityHistoryStore,
    LiveStateService,
    ex_ante_portfolio_volatility,
)
from src.api.schemas import AccountSnapshotResponse, EquityPoint, PositionRow
from src.api.services import IBKRBookUnavailableError, IBKRExecutionService
from src.api.telemetry import TelemetryBroadcaster
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


def _daily_point(days_ago: int, value: float = 1_000_000.0) -> EquityPoint:
    return EquityPoint(time=FIXED_TIMESTAMP - timedelta(days=days_ago), value=value)


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
    # Same UTC day -> one distinct day; intraday view returns that day's points.
    assert store.distinct_utc_days() == 1
    assert [point.value for point in store.intraday_points()] == [point.value for point in points]


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
    assert response.equity_curve_live_days == 1
    assert len(response.equity_curve_intraday) == len(live_points)
    # The existing synthetic backfill curve is kept (never relabelled live).
    assert [point.value for point in response.equity_curve] == [point.value for point in backfill]


async def test_same_day_burst_does_not_promote_the_live_curve(settings: Settings) -> None:
    """42 same-day live points are NOT enough: the 6M curve stays backfilled."""
    backfill = [_point(-180), _point(-120), _point(-60)]
    live_points = [_point(minutes, 1_200_000.0 + minutes) for minutes in range(42)]
    service = _live_service(
        settings,
        FakeIBKRExecution(_snapshot(curve=backfill), live_points),
        make_market_state(),
    )

    response = await service.account()

    assert response.equity_curve_source == "simulated-backfill"
    assert response.equity_curve_live_points == 42
    assert response.equity_curve_live_days == 1
    # 1D plotting gets all 42 real intraday points even though the 6M curve is backfilled.
    assert len(response.equity_curve_intraday) == 42
    assert [point.value for point in response.equity_curve_intraday] == [
        point.value for point in live_points
    ]
    # The synthetic backfill curve is preserved.
    assert [point.value for point in response.equity_curve] == [point.value for point in backfill]


async def test_five_distinct_days_promote_the_live_curve(settings: Settings) -> None:
    backfill = [_point(-180), _point(-120), _point(-60)]
    live_points = [_daily_point(days_ago, 1_200_000.0 + days_ago) for days_ago in range(5, 0, -1)]
    live_points.append(_point(1, 1_300_000.0))  # today's first intraday point
    service = _live_service(
        settings,
        FakeIBKRExecution(_snapshot(curve=backfill), live_points),
        make_market_state(),
    )

    response = await service.account()

    assert response.equity_curve_source == "ibkr-live"
    assert response.equity_curve_live_points == len(live_points)
    assert response.equity_curve_live_days == MIN_LIVE_CURVE_DAYS + 1
    # Live mode plots the real recorded points only (no synthetic spline prepended).
    assert len(response.equity_curve) == len(live_points)
    assert [point.value for point in response.equity_curve] == [
        point.value for point in live_points
    ]
    # Intraday curve is the latest recorded UTC day only.
    assert len(response.equity_curve_intraday) == 1
    assert response.equity_curve_intraday[0].value == pytest.approx(1_300_000.0)


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


# ---------------------------------------------------------------------------
# v1.5.2 WS telemetry: IBKR frames use the enriched book, no demo jitter
# ---------------------------------------------------------------------------


def _broadcaster(
    settings: Settings,
    execution: FakeIBKRExecution,
    provider: object | None,
) -> TelemetryBroadcaster:
    return TelemetryBroadcaster(
        settings=settings,
        clock=lambda: FIXED_TIMESTAMP,
        execution=cast(Any, execution),
        risk=cast(Any, object()),
        market=cast(Any, object()),
        account_provider=cast(Any, provider),
    )


async def test_telemetry_ibkr_frames_use_enriched_flat_book_without_jitter(
    settings: Settings,
) -> None:
    enriched = _snapshot(positions=[], curve=[]).model_copy(
        update={
            "portfolio_volatility": 0.0,
            "portfolio_volatility_source": "ex-ante-book",
            # Deliberately leftover/stale margin figures: a flat IBKR book must
            # still publish exactly 0.0 margin/util/volatility in the frames.
            "initial_margin_used": 3_000.0,
            "maintenance_margin_used": 2_700.0,
            "margin_utilization": 0.003,
            "equity_curve_source": "simulated-backfill",
            "equity_curve_live_points": 42,
            "equity_curve_live_days": 1,
            "equity_curve_intraday": [_point(minutes) for minutes in range(42)],
        }
    )
    execution = FakeIBKRExecution(_snapshot())  # raw fallback must NOT be used
    provider_calls: list[int] = []

    async def provider() -> AccountSnapshotResponse:
        provider_calls.append(1)
        return enriched

    broadcaster = _broadcaster(settings, execution, provider)

    pnl = await broadcaster._build_envelope("pnl")
    assert pnl.kind == "pnl"
    # No demo jitter: the frame carries the exact enriched book values.
    assert pnl.payload.net_liquidation == pytest.approx(enriched.net_liquidation)
    assert pnl.payload.day_pnl == pytest.approx(enriched.day_pnl)
    assert pnl.payload.point.value == pytest.approx(enriched.net_liquidation)
    assert pnl.payload.portfolio_volatility == 0.0
    assert pnl.payload.initial_margin_used == 0.0
    assert pnl.payload.margin_utilization == 0.0
    assert pnl.payload.open_positions == 0

    margin = await broadcaster._build_envelope("margin")
    assert margin.kind == "margin"
    assert margin.payload.initial_margin_used == 0.0
    assert margin.payload.maintenance_margin_used == 0.0
    assert margin.payload.margin_utilization == 0.0
    assert margin.payload.portfolio_volatility == 0.0
    assert margin.payload.open_positions == 0
    assert provider_calls, "IBKR frames must read the enriched LiveStateService account"


async def test_telemetry_ibkr_frames_carry_open_positions_count(settings: Settings) -> None:
    positions = [_position_row("ES", weight=0.3), _position_row("NQ", weight=-0.2)]
    enriched = _snapshot(positions=positions).model_copy(
        update={"portfolio_volatility": 0.137, "portfolio_volatility_source": "ex-ante-book"}
    )

    async def provider() -> AccountSnapshotResponse:
        return enriched

    broadcaster = _broadcaster(settings, FakeIBKRExecution(_snapshot()), provider)
    pnl = await broadcaster._build_envelope("pnl")
    margin = await broadcaster._build_envelope("margin")
    assert pnl.payload.open_positions == 2
    assert pnl.payload.portfolio_volatility == pytest.approx(0.137)
    assert margin.payload.open_positions == 2
    assert margin.payload.portfolio_volatility == pytest.approx(0.137)


async def test_telemetry_ibkr_without_book_emits_heartbeat(settings: Settings) -> None:
    execution = FakeIBKRExecution(_snapshot())

    async def unavailable() -> AccountSnapshotResponse:
        raise IBKRBookUnavailableError("IBKR session unavailable; no live account book")

    broadcaster = _broadcaster(settings, execution, unavailable)
    for kind in ("pnl", "margin", "positions"):
        envelope = await broadcaster._build_envelope(kind)
        assert envelope.kind == "heartbeat"


async def test_telemetry_offline_frames_keep_jitter_and_default_attribution(
    settings: Settings,
) -> None:
    snapshot = _snapshot()
    execution = FakeIBKRExecution(snapshot, ibkr_enabled=False)
    broadcaster = _broadcaster(settings, execution, None)

    pnl = await broadcaster._build_envelope("pnl")
    margin = await broadcaster._build_envelope("margin")
    assert pnl.kind == "pnl"
    # Offline deterministic demo jitter is unchanged.
    assert pnl.payload.net_liquidation != pytest.approx(snapshot.net_liquidation)
    assert pnl.payload.portfolio_volatility is None
    assert pnl.payload.open_positions == 0
    assert margin.payload.portfolio_volatility is None
