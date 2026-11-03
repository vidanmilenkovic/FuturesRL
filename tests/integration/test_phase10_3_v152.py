"""Independent v1.5.2 verification: IBKR telemetry/equity provenance + legacy purge.

Verification-only suite for branch ``fix/v1.5.2-state-sync`` (tasks 42/43/44); no
production edits. It pins the v1.5.2 contracts:

1. **WS telemetry**: ``pnl``/``margin``/``positions`` frames source the SAME
   enriched IBKR account as ``GET /api/positions`` (injected ``account_provider``);
   flat IBKR book -> ``initial_margin_used``/``margin_utilization``/
   ``portfolio_volatility`` exactly ``0.0`` and ``open_positions == 0``; no demo
   PnL jitter on IBKR frames; last-known-good book survives provider failures;
   IBKR enabled without a book emits heartbeat (never simulated); offline frames
   keep the historical jitter/behaviour.
2. **Equity provenance**: ``LiveEquityHistoryStore`` same-day bursts stay
   ``simulated-backfill`` with ``live_days == 1`` and an intraday curve; the
   curve becomes ``ibkr-live`` ONLY when history spans >=5 distinct UTC days, and
   then ``equity_curve`` contains real recorded points only (no synthetic
   spline); offline unchanged.
3. **Optimizer legacy detection**: fabricated old equal-weight/mean-variance +
   structural studies are flagged with reasons; a clean curated 6-param study is
   valid.
4. **API startup purge/quarantine**: legacy + fixture studies are moved to
   ``<optimizations>/legacy/`` with reasons, excluded from list/detail/apply
   (404), corrupt files are quarantined safely, valid artifacts stay
   byte-canonical, and restarts are idempotent; the Studio study list (GET
   /api/optimize) is clean.

Everything is offline and deterministic: fakes + ``tmp_path`` only, never TWS.
"""

from __future__ import annotations

import asyncio
import json
import math
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest
from configs.instruments import InstrumentsConfig, load_instruments
from configs.settings import Settings
from configs.strategy import load_strategy_config
from fastapi.testclient import TestClient
from src.api.live import (
    EQUITY_HISTORY_FILE_NAME,
    MIN_LIVE_CURVE_DAYS,
    LiveEquityHistoryStore,
    LiveStateService,
    ex_ante_portfolio_volatility,
)
from src.api.main import create_app
from src.api.optimize_service import OptimizationService, OptimizationStudy
from src.api.schemas import (
    AccountSnapshotResponse,
    EquityPoint,
    MarginTelemetryPayload,
    PnlTelemetryPayload,
    PositionRow,
    PositionsTelemetryPayload,
    TelemetryEnvelope,
)
from src.api.services import Container, build_container
from src.api.telemetry import TelemetryBroadcaster
from src.backtest.optimizer import (
    default_search_space,
    is_legacy_search_space,
    validate_persisted_optimization,
)
from src.core.types import AssetClass

from tests.fixtures.market import make_market_state

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
INSTRUMENTS_PATH = REPO_ROOT / "configs" / "instruments.yaml"
STRATEGY_DIR = REPO_ROOT / "configs" / "strategies"

FIXED_CLOCK = datetime(2026, 11, 3, 12, 0, tzinfo=UTC)
TELEMETRY_INTERVAL = 0.01
FLAT_EQUITY = 1_000_000.0

#: Curated Optuna default-space size (v1.5.3 removed carry -> exactly five).
EXPECTED_CURATED_COUNT = 5


# ---------------------------------------------------------------------------
# Fixtures / shared helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def master() -> InstrumentsConfig:
    return load_instruments(INSTRUMENTS_PATH)


def _settings(tmp_path: Path, *, telemetry_interval: float = TELEMETRY_INTERVAL) -> Settings:
    return Settings(
        environment="backtest",
        data_dir=tmp_path / "data",
        parquet_dir=tmp_path / "data" / "parquet",
        duckdb_path=tmp_path / "data" / "v152.duckdb",
        report_dir=tmp_path / "artifacts",
        instruments_path=INSTRUMENTS_PATH,
        telemetry_interval_seconds=telemetry_interval,
    )


class _Clock:
    """Mutable deterministic clock (telemetry cache cadence + store trims)."""

    def __init__(self, now: datetime = FIXED_CLOCK) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now = self.now + timedelta(seconds=seconds)


def _position_row(root: str = "ES", *, weight: float = 0.24, quantity: int = 1) -> PositionRow:
    return PositionRow(
        instrument_id=root,
        root_symbol=root,
        asset_class=AssetClass.EQUITY_INDEX,
        contract_symbol=f"{root}_20240315",
        expiration=None,
        days_to_roll=None,
        quantity=quantity,
        average_price=4800.0,
        market_price=4800.0,
        multiplier=50.0,
        notional=weight * FLAT_EQUITY,
        weight=weight,
        unrealized_pnl=0.0,
        initial_margin=15_000.0,
        maintenance_margin=13_600.0,
    )


def _account_snapshot(
    positions: list[PositionRow] | None = None,
    *,
    curve: list[EquityPoint] | None = None,
    net_liquidation: float = FLAT_EQUITY,
    initial_margin_used: float = 0.0,
    margin_utilization: float = 0.0,
    portfolio_volatility: float | None = 0.0,
) -> AccountSnapshotResponse:
    rows = list(positions or [])
    return AccountSnapshotResponse(
        timestamp=FIXED_CLOCK,
        net_liquidation=net_liquidation,
        cash_balance=net_liquidation,
        buying_power=net_liquidation * 2.0,
        initial_margin_used=initial_margin_used,
        maintenance_margin_used=initial_margin_used * 0.9,
        excess_liquidity=net_liquidation - initial_margin_used,
        margin_utilization=margin_utilization,
        day_pnl=0.0,
        day_pnl_pct=0.0,
        gross_leverage=sum(abs(row.weight) for row in rows),
        net_exposure=sum(row.weight for row in rows),
        equity_curve=list(curve or []),
        positions=rows,
        source="engine",
        equity_curve_source="fixture",
        equity_curve_live_points=0,
        equity_curve_live_days=0,
        equity_curve_intraday=[],
        portfolio_volatility=portfolio_volatility,
        portfolio_volatility_source="ex-ante-book" if portfolio_volatility is not None else None,
    )


class _FakeExecution:
    """Minimal execution adapter (IBKR flag + deterministic account book)."""

    def __init__(
        self,
        *,
        ibkr_enabled: bool,
        account: AccountSnapshotResponse | Exception,
        points: tuple[EquityPoint, ...] = (),
    ) -> None:
        self.ibkr_enabled = ibkr_enabled
        self.source_name = "engine"
        self.connected = True
        self._account = account
        self._points = points
        self.account_calls = 0

    async def account(self) -> AccountSnapshotResponse:
        self.account_calls += 1
        if isinstance(self._account, Exception):
            raise self._account
        return self._account

    @property
    def live_equity_points(self) -> tuple[EquityPoint, ...]:
        return self._points


class _FakeMarket:
    """Minimal MarketDataService stand-in."""

    def __init__(self, state: Any | None = None) -> None:
        self._state = state

    def latest_state(self) -> Any | None:
        return self._state


def _broadcaster(
    tmp_path: Path,
    *,
    execution: _FakeExecution,
    clock: Any,
    account_provider: Any | None = None,
) -> TelemetryBroadcaster:
    return TelemetryBroadcaster(
        settings=_settings(tmp_path),
        clock=clock,
        execution=cast(Any, execution),
        risk=cast(Any, object()),
        market=cast(Any, object()),
        account_provider=account_provider,
    )


def _frame(broadcaster: TelemetryBroadcaster, kind: str) -> TelemetryEnvelope:
    return asyncio.run(broadcaster._build_envelope(cast(Any, kind)))


# ---------------------------------------------------------------------------
# 1. WS telemetry: IBKR frames source the enriched account, no jitter
# ---------------------------------------------------------------------------


def _flat_ibkr_account() -> AccountSnapshotResponse:
    return _account_snapshot(
        [],
        curve=[],
        net_liquidation=FLAT_EQUITY,
        initial_margin_used=0.0,
        margin_utilization=0.0,
        portfolio_volatility=0.0,
    )


def _simulated_account(*, portfolio_volatility: float | None = 0.119) -> AccountSnapshotResponse:
    return _account_snapshot(
        [_position_row("ES", weight=0.24)],
        curve=[EquityPoint(time=FIXED_CLOCK, value=500_000.0)],
        net_liquidation=500_000.0,
        initial_margin_used=3_000.0,
        margin_utilization=0.006,
        portfolio_volatility=portfolio_volatility,
    )


def test_telemetry_ibkr_frames_use_enriched_account_no_jitter(tmp_path: Path) -> None:
    raw = _simulated_account()  # deliberately different simulated book
    enriched = _flat_ibkr_account()
    provider_calls = 0

    async def provider() -> AccountSnapshotResponse:
        nonlocal provider_calls
        provider_calls += 1
        return enriched

    execution = _FakeExecution(ibkr_enabled=True, account=raw)
    broadcaster = _broadcaster(
        tmp_path, execution=execution, clock=_Clock(), account_provider=provider
    )

    pnl_frames = [_frame(broadcaster, "pnl") for _ in range(6)]
    for envelope in pnl_frames:
        assert envelope.kind in {"pnl", "heartbeat"}
        payload = cast(PnlTelemetryPayload, envelope.payload)
        assert isinstance(payload, PnlTelemetryPayload)
        assert payload.net_liquidation == pytest.approx(FLAT_EQUITY)
        assert payload.day_pnl == pytest.approx(0.0)
        assert payload.portfolio_volatility == 0.0
        assert payload.initial_margin_used == 0.0
        assert payload.margin_utilization == 0.0
        assert payload.open_positions == 0
    # No demo jitter on IBKR frames: every tick is identical.
    assert {frame.payload.model_dump()["net_liquidation"] for frame in pnl_frames} == {FLAT_EQUITY}
    # The raw simulated book (500k / 3k margin / 11.9% vol) never leaks into WS.
    assert all(
        frame.payload.model_dump().get("net_liquidation") != 500_000.0 for frame in pnl_frames
    )
    assert provider_calls >= 1
    assert execution.account_calls == 0, (
        "IBKR frames must use the enriched provider, not raw execution"
    )

    margin = cast(MarginTelemetryPayload, _frame(broadcaster, "margin").payload)
    assert margin.initial_margin_used == 0.0
    assert margin.maintenance_margin_used == 0.0
    assert margin.margin_utilization == 0.0
    assert margin.portfolio_volatility == 0.0
    assert margin.open_positions == 0

    positions = cast(PositionsTelemetryPayload, _frame(broadcaster, "positions").payload)
    assert positions.positions == []
    assert positions.gross_leverage == 0.0
    assert positions.net_exposure == 0.0


def test_telemetry_ibkr_last_known_good_survives_provider_failure(tmp_path: Path) -> None:
    clock = _Clock()
    enriched = _flat_ibkr_account()
    state = {"fail": False}

    async def provider() -> AccountSnapshotResponse:
        if state["fail"]:
            raise RuntimeError("IBKR session dropped")
        return enriched

    execution = _FakeExecution(ibkr_enabled=True, account=RuntimeError("no book"))
    broadcaster = _broadcaster(
        tmp_path, execution=execution, clock=clock, account_provider=provider
    )

    first = _frame(broadcaster, "pnl")
    assert first.kind == "pnl"
    clock.advance(5.0)  # past the refresh cadence -> provider is retried
    state["fail"] = True
    second = _frame(broadcaster, "pnl")
    assert second.kind == "pnl", (
        "a last-known-good IBKR book must keep streaming, not degrade to heartbeat"
    )
    payload = cast(PnlTelemetryPayload, second.payload)
    assert payload.net_liquidation == pytest.approx(FLAT_EQUITY)
    assert payload.portfolio_volatility == 0.0


def test_telemetry_ibkr_without_book_emits_heartbeat_never_simulated(tmp_path: Path) -> None:
    async def provider() -> AccountSnapshotResponse:
        raise RuntimeError("IBKR enabled but not connected")

    execution = _FakeExecution(ibkr_enabled=True, account=RuntimeError("not connected"))
    broadcaster = _broadcaster(
        tmp_path, execution=execution, clock=_Clock(), account_provider=provider
    )

    for kind in ("pnl", "margin", "positions"):
        envelope = _frame(broadcaster, kind)
        assert envelope.kind == "heartbeat", (
            f"{kind} must degrade to heartbeat without an honest book"
        )
        assert envelope.payload.model_dump()["status"] == "ok"


def test_telemetry_offline_frames_unchanged(tmp_path: Path) -> None:
    account = _simulated_account(portfolio_volatility=None)
    execution = _FakeExecution(ibkr_enabled=False, account=account)

    async def unused_provider() -> AccountSnapshotResponse:
        raise AssertionError("offline telemetry must not use the IBKR account provider")

    broadcaster = _broadcaster(
        tmp_path, execution=execution, clock=_Clock(), account_provider=unused_provider
    )
    envelope = _frame(broadcaster, "pnl")
    assert envelope.kind == "pnl"
    payload = cast(PnlTelemetryPayload, envelope.payload)
    # The historical deterministic demo jitter is preserved offline (sequence 1).
    expected_jitter = 180.0 * math.sin(1.0 * 0.7)
    assert payload.net_liquidation == pytest.approx(round(500_000.0 + expected_jitter, 2))
    assert payload.net_liquidation != pytest.approx(500_000.0)
    assert payload.portfolio_volatility is None  # additive fields default offline
    assert payload.initial_margin_used == pytest.approx(3_000.0)
    assert payload.margin_utilization == pytest.approx(0.006)
    assert payload.open_positions == 1

    margin = cast(MarginTelemetryPayload, _frame(broadcaster, "margin").payload)
    assert margin.initial_margin_used == pytest.approx(3_000.0)
    assert margin.margin_utilization == pytest.approx(0.006)
    assert margin.portfolio_volatility is None
    assert margin.open_positions == 1

    positions = cast(PositionsTelemetryPayload, _frame(broadcaster, "positions").payload)
    assert len(positions.positions) == 1


def test_ws_telemetry_ibkr_frames_end_to_end(tmp_path: Path) -> None:
    """WebSocket end-to-end: flat IBKR book frames are exact and jitter-free."""
    settings = _settings(tmp_path)
    clock = _Clock()
    execution = _FakeExecution(ibkr_enabled=True, account=_flat_ibkr_account())
    container: Container = build_container(settings, clock=clock)
    live = LiveStateService(
        settings=settings,
        clock=clock,
        execution=execution,
        instrument_master=settings.instruments,
        market=_FakeMarket(None),
    )
    telemetry = TelemetryBroadcaster(
        settings=settings,
        clock=clock,
        execution=cast(Any, execution),
        risk=container.risk,
        market=container.market,
        account_provider=live.account,
    )
    container = replace(container, execution=cast(Any, execution), live=live, telemetry=telemetry)
    app = create_app(settings, clock=clock, container=container)

    seen: dict[str, list[dict[str, Any]]] = {}
    with TestClient(app) as client, client.websocket_connect("/ws/telemetry") as websocket:
        for _ in range(120):
            frame = websocket.receive_json()
            seen.setdefault(frame["kind"], []).append(frame)
            if {"pnl", "margin", "positions"} <= set(seen):
                break
    assert {"pnl", "margin", "positions"} <= set(seen), f"missing WS frame kinds: {set(seen)}"
    pnl_values = {frame["payload"]["net_liquidation"] for frame in seen["pnl"]}
    assert pnl_values == {FLAT_EQUITY}, f"IBKR WS pnl frames jittered: {pnl_values}"
    for frame in seen["pnl"]:
        payload = frame["payload"]
        assert payload["initial_margin_used"] == 0.0
        assert payload["margin_utilization"] == 0.0
        assert payload["portfolio_volatility"] == 0.0
        assert payload["open_positions"] == 0
    for frame in seen["margin"]:
        payload = frame["payload"]
        assert payload["initial_margin_used"] == 0.0
        assert payload["margin_utilization"] == 0.0
        assert payload["portfolio_volatility"] == 0.0
        assert payload["open_positions"] == 0
    for frame in seen["positions"]:
        assert frame["payload"]["positions"] == []


# ---------------------------------------------------------------------------
# 2. Equity provenance: distinct UTC days + intraday curve
# ---------------------------------------------------------------------------


def _same_day_points(count: int = 42) -> tuple[EquityPoint, ...]:
    return tuple(
        EquityPoint(
            time=FIXED_CLOCK - timedelta(minutes=count - index),
            value=FLAT_EQUITY + index,
        )
        for index in range(count)
    )


def _points_across_days(days: int, per_day: int = 2) -> tuple[EquityPoint, ...]:
    points: list[EquityPoint] = []
    for day in range(days):
        day_start = FIXED_CLOCK - timedelta(days=day)
        for index in range(per_day):
            points.append(
                EquityPoint(
                    time=day_start - timedelta(minutes=index),
                    value=FLAT_EQUITY + day * 1_000.0 + index,
                )
            )
    return tuple(sorted(points, key=lambda point: point.time))


def _history_store(tmp_path: Path) -> LiveEquityHistoryStore:
    return LiveEquityHistoryStore(
        tmp_path / "data" / EQUITY_HISTORY_FILE_NAME,
        clock=lambda: FIXED_CLOCK,
    )


def test_live_history_distinct_days_and_intraday_helpers(tmp_path: Path) -> None:
    store = _history_store(tmp_path)
    assert store.distinct_utc_days() == 0
    assert store.intraday_points() == []
    for point in _same_day_points(42):
        store.append(point)
    assert store.distinct_utc_days() == 1
    intraday = store.intraday_points()
    assert len(intraday) == 42
    assert [point.time for point in intraday] == [point.time for point in _same_day_points(42)]

    other = _history_store(tmp_path / "other")
    for point in _points_across_days(days=5, per_day=2):
        other.append(point)
    assert other.distinct_utc_days() == 5
    assert len(other.intraday_points()) == 2  # latest recorded UTC day only


def _attribution_client(
    tmp_path: Path,
    *,
    execution: _FakeExecution,
    market_state: Any | None = None,
) -> TestClient:
    settings = _settings(tmp_path)
    clock = lambda: FIXED_CLOCK  # noqa: E731 - deterministic fixture clock
    container = build_container(settings, clock=clock)
    live = LiveStateService(
        settings=settings,
        clock=clock,
        execution=execution,
        instrument_master=settings.instruments,
        market=_FakeMarket(market_state),
    )
    container = replace(container, execution=cast(Any, execution), live=live)
    app = create_app(settings, clock=clock, container=container)
    return TestClient(app)


def test_positions_same_day_burst_is_backfill_with_intraday(tmp_path: Path) -> None:
    points = _same_day_points(42)
    synthetic = [
        EquityPoint(time=FIXED_CLOCK - timedelta(days=30), value=900_000.0),
        EquityPoint(time=FIXED_CLOCK - timedelta(days=15), value=950_000.0),
        EquityPoint(time=FIXED_CLOCK, value=FLAT_EQUITY),
    ]
    execution = _FakeExecution(
        ibkr_enabled=True, account=_account_snapshot([], curve=synthetic), points=points
    )
    with _attribution_client(tmp_path, execution=execution) as client:
        body = client.get("/api/positions").json()
    assert body["equity_curve_source"] == "simulated-backfill"
    assert body["equity_curve_live_points"] == 42
    assert body["equity_curve_live_days"] == 1
    assert len(body["equity_curve_intraday"]) == 42
    assert [point["value"] for point in body["equity_curve"]] == [
        pytest.approx(point.value) for point in synthetic
    ], "same-day burst must keep the synthetic backfill multi-day curve"


def test_positions_five_distinct_days_promote_to_ibkr_live(tmp_path: Path) -> None:
    points = _points_across_days(days=5, per_day=2)
    synthetic = [EquityPoint(time=FIXED_CLOCK - timedelta(days=30), value=900_000.0)]
    execution = _FakeExecution(
        ibkr_enabled=True,
        account=_account_snapshot([], curve=synthetic),
        points=points,
    )
    with _attribution_client(tmp_path, execution=execution) as client:
        body = client.get("/api/positions").json()
    assert body["equity_curve_source"] == "ibkr-live"
    assert body["equity_curve_live_points"] == len(points)
    assert body["equity_curve_live_days"] == 5
    assert [point["value"] for point in body["equity_curve"]] == [
        pytest.approx(point.value) for point in points
    ], "live mode must contain ONLY real recorded points (synthetic spline removed)"
    assert len(body["equity_curve"]) == len(points)
    intraday = [point["value"] for point in body["equity_curve_intraday"]]
    assert intraday == [pytest.approx(point.value) for point in points[-2:]]


def test_positions_four_distinct_days_stay_backfill(tmp_path: Path) -> None:
    points = _points_across_days(days=4, per_day=2)
    synthetic = [EquityPoint(time=FIXED_CLOCK - timedelta(days=30), value=900_000.0)]
    execution = _FakeExecution(
        ibkr_enabled=True, account=_account_snapshot([], curve=synthetic), points=points
    )
    with _attribution_client(tmp_path, execution=execution) as client:
        body = client.get("/api/positions").json()
    assert MIN_LIVE_CURVE_DAYS == 5
    assert body["equity_curve_source"] == "simulated-backfill"
    assert body["equity_curve_live_days"] == 4
    assert body["equity_curve_live_points"] == len(points)
    assert len(body["equity_curve_intraday"]) == 2
    assert [point["value"] for point in body["equity_curve"]] == [
        pytest.approx(point.value) for point in synthetic
    ]


def test_positions_offline_provenance_unchanged(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path), clock=lambda: FIXED_CLOCK)
    with TestClient(app) as client:
        response = client.get("/api/positions")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["equity_curve_source"] == "fixture"
    assert body["equity_curve_live_points"] == 0
    assert body["equity_curve_live_days"] == 0
    assert body["equity_curve_intraday"] == []
    assert body["portfolio_volatility"] is None


def test_positions_ex_ante_volatility_with_live_book(tmp_path: Path) -> None:
    state = make_market_state(roots=("ES", "NQ"))
    positions = [_position_row("ES", weight=0.3), _position_row("NQ", weight=0.2, quantity=2)]
    points = _points_across_days(days=5, per_day=2)
    execution = _FakeExecution(
        ibkr_enabled=True, account=_account_snapshot(positions), points=points
    )
    with _attribution_client(tmp_path, execution=execution, market_state=state) as client:
        body = client.get("/api/positions").json()
    assert body["equity_curve_source"] == "ibkr-live"
    volatility = body["portfolio_volatility"]
    assert isinstance(volatility, float) and math.isfinite(volatility) and volatility > 0.0
    assert body["portfolio_volatility_source"] == "ex-ante-book"
    expected, source = ex_ante_portfolio_volatility({"ES": 0.3, "NQ": 0.2}, state)
    assert source == "ex-ante-book"
    assert expected is not None
    assert volatility == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# 3. Optimizer legacy detection
# ---------------------------------------------------------------------------


def _clean_curated_payload(base_id: str = "tsmom_vol_target") -> dict[str, Any]:
    config = load_strategy_config(STRATEGY_DIR / f"{base_id}.yaml")
    space = [
        {"param_path": entry.param_path, "type": entry.type, "low": entry.low, "high": entry.high}
        for entry in default_search_space(config)
    ]
    return {
        "base_strategy_id": base_id,
        "seed": 7,
        "request": {"base_strategy_id": base_id, "search_space": space},
        "best_params": {space[0]["param_path"]: space[0]["low"]},
        "trials": [{"params": {space[0]["param_path"]: space[0]["low"]}}],
    }


def test_optimizer_clean_curated_study_is_valid() -> None:
    payload = _clean_curated_payload("tsmom_vol_target")
    assert len(payload["request"]["search_space"]) == EXPECTED_CURATED_COUNT
    assert validate_persisted_optimization(payload) == []
    assert is_legacy_search_space(payload) is False


def test_optimizer_legacy_tsmom_study_is_flagged_with_reasons() -> None:
    payload = {
        "base_strategy_id": "tsmom_vol_target",
        "seed": 7,
        "request": {
            "base_strategy_id": "tsmom_vol_target",
            "search_space": [
                {"param_path": "allocator.params.risk_aversion"},  # mean_variance
                {"param_path": "allocator.params.gross_leverage"},  # equal_weight
                {"param_path": "timing.0.params.rsi_period"},  # inactive overlay param
                {"param_path": "universe.params.top_n"},  # structural / hygiene
            ],
        },
        "best_params": {"universe.params.top_n": 8},
        "trials": [],
    }
    reasons = validate_persisted_optimization(payload)
    assert reasons, "legacy inactive/structural study must be flagged"
    joined = " ".join(reasons)
    assert "universe.params.top_n" in joined
    assert "non-tunable structural/hygiene" in joined
    assert "allocator.params.risk_aversion" in joined or "inactive" in joined
    assert is_legacy_search_space(payload) is True


def test_optimizer_legacy_drl_study_is_flagged_with_reasons() -> None:
    payload = {
        "base_strategy_id": "drl_ppo_llm_risk",
        "seed": 7,
        "request": {
            "base_strategy_id": "drl_ppo_llm_risk",
            "search_space": [
                {"param_path": "allocator.params.risk_aversion"},
                {"param_path": "allocator.params.max_weight"},
                {"param_path": "universe.params.top_n"},
            ],
        },
        "best_params": {"universe.params.top_n": 5},
        "trials": [],
    }
    reasons = validate_persisted_optimization(payload)
    assert reasons
    assert any("inactive" in reason or "unknown" in reason for reason in reasons)
    assert any("non-tunable structural" in reason for reason in reasons)


def test_optimizer_unresolvable_base_is_handled_conservatively() -> None:
    payload = {
        "base_strategy_id": "does_not_exist_strategy",
        "seed": 7,
        "request": {
            "base_strategy_id": "does_not_exist_strategy",
            "search_space": [
                {"param_path": "foo.params.imaginary"},
                {"param_path": "universe.params.top_n"},
            ],
        },
    }
    reasons = validate_persisted_optimization(payload)
    assert any("unknown stage" in reason for reason in reasons)
    assert any("non-tunable structural" in reason for reason in reasons)


# ---------------------------------------------------------------------------
# 4. API startup purge / quarantine of legacy optimization artifacts
# ---------------------------------------------------------------------------


def _artifacts_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "artifacts" / "optimizations"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _study_payload(
    study_id: str,
    *,
    base_id: str = "tsmom_vol_target",
    provenance: str | None = None,
    search_space: list[dict[str, Any]] | None = None,
    best_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    study = OptimizationStudy.model_validate(
        {
            "study_id": study_id,
            "base_strategy_id": base_id,
            "objective_metric": "sharpe",
            "n_trials": 6,
            "completed_trials": 6,
            "created_at": FIXED_CLOCK.isoformat(),
            "best_params": dict(best_params or {}),
            "best_value": 1.25,
        }
    )
    payload = study.model_dump(mode="json")
    # ``OptimizationStudy`` defaults ``provenance="fixture"``; engine-canonical
    # payloads must not carry the API-owned provenance tag (the reload classifies
    # them from ``request``/``seed`` instead).
    payload.pop("provenance", None)
    if provenance is not None:
        payload["provenance"] = provenance
    else:  # optimizer-canonical engine study
        payload["seed"] = 7
        payload["request"] = {"base_strategy_id": base_id, "search_space": list(search_space or [])}
    return payload


def _write_studies(directory: Path) -> dict[str, Path]:
    tsmom = load_strategy_config(STRATEGY_DIR / "tsmom_vol_target.yaml")
    clean_space = [
        {"param_path": entry.param_path, "type": entry.type, "low": entry.low, "high": entry.high}
        for entry in default_search_space(tsmom)
    ]
    legacy_space = [
        {"param_path": "allocator.params.risk_aversion"},
        {"param_path": "allocator.params.gross_leverage"},
        {"param_path": "universe.params.top_n"},
    ]
    payloads = {
        "clean-engine": _study_payload(
            "clean-engine",
            search_space=clean_space,
            best_params={"timing.0.params.lookback_fast": 21},
        ),
        "legacy-engine": _study_payload(
            "legacy-engine", search_space=legacy_space, best_params={"universe.params.top_n": 8}
        ),
        "clean-fixture": _study_payload(
            "clean-fixture", provenance="fixture", best_params={"timing.0.params.lookback_fast": 21}
        ),
        "legacy-fixture": _study_payload(
            "legacy-fixture", provenance="fixture", best_params={"universe.params.top_n": 8}
        ),
    }
    paths: dict[str, Path] = {}
    for study_id, payload in payloads.items():
        path = directory / f"{study_id}.json"
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        paths[study_id] = path
    corrupt = directory / "corrupt.json"
    corrupt.write_text("{this is not valid json", encoding="utf-8")
    paths["corrupt"] = corrupt
    return paths


def _quarantined(service: OptimizationService) -> dict[str, Any]:
    return {item.study_id: item for item in service.quarantined_studies}


def test_startup_purge_quarantines_legacy_studies(tmp_path: Path) -> None:
    directory = _artifacts_dir(tmp_path)
    paths = _write_studies(directory)
    clean_before = {
        study_id: paths[study_id].read_bytes() for study_id in ("clean-engine", "clean-fixture")
    }
    settings = _settings(tmp_path)
    app = create_app(settings, clock=lambda: FIXED_CLOCK)
    with TestClient(app) as client:
        listed = client.get("/api/optimize")
        assert listed.status_code == 200, listed.text
        ids = {row["study_id"] for row in listed.json()}
        assert ids == {"clean-engine", "clean-fixture"}, f"Studio study list not clean: {ids}"

        service = cast(OptimizationService, app.state.optimize_service)
        quarantined = _quarantined(service)
        assert {"legacy-engine", "legacy-fixture"} <= set(quarantined)
        for study_id in ("legacy-engine", "legacy-fixture"):
            reasons = quarantined[study_id].reasons
            assert reasons, f"{study_id} quarantined without a reason"
            target = quarantined[study_id].quarantine_path
            assert target.exists() and target.parent == service.legacy_dir
            assert target.name.startswith(f"{study_id}.json")
            assert not (directory / f"{study_id}.json").exists()
        legacy_reasons = " ".join(
            reason
            for study_id in ("legacy-engine", "legacy-fixture")
            for reason in quarantined[study_id].reasons
        )
        assert "top_n" in legacy_reasons or "non-tunable" in legacy_reasons

        # Corrupt payload is quarantined safely (renamed) and never loaded.
        assert not paths["corrupt"].exists()
        assert list(directory.glob("corrupt.json.corrupt-*")), "corrupt artifact not quarantined"
        assert any("corrupt" in warning for warning in service.warnings)

        # Quarantined ids are not selectable: detail + apply return 404.
        assert client.get("/api/optimize/legacy-engine").status_code == 404
        assert client.post("/api/optimize/legacy-engine/apply", json={}).status_code == 404
        assert client.get("/api/optimize/legacy-fixture").status_code == 404
        assert client.post("/api/optimize/legacy-fixture/apply", json={}).status_code == 404
        # Clean studies remain fully selectable.
        for study_id in ("clean-engine", "clean-fixture"):
            detail = client.get(f"/api/optimize/{study_id}")
            assert detail.status_code == 200, detail.text
            assert detail.json()["study_id"] == study_id
        # Valid artifacts were not rewritten (byte-canonical).
        for study_id in ("clean-engine", "clean-fixture"):
            assert paths[study_id].read_bytes() == clean_before[study_id]

    # Restart is idempotent: nothing new to quarantine, the same clean set.
    restart_app = create_app(_settings(tmp_path), clock=lambda: FIXED_CLOCK)
    with TestClient(restart_app) as client:
        listed = client.get("/api/optimize")
        assert {row["study_id"] for row in listed.json()} == {"clean-engine", "clean-fixture"}
        restart_service = cast(OptimizationService, restart_app.state.optimize_service)
        assert restart_service.quarantined_count == 0, "restart re-quarantined artifacts"
        legacy_files = list((directory / "legacy").glob("*.json"))
        assert len(legacy_files) == 2, f"unexpected legacy dir content: {legacy_files}"
        assert client.get("/api/optimize/legacy-engine").status_code == 404
