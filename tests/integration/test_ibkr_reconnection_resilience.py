"""Phase-10 integration tests: API-owned IBKR session resilience (task-22).

A deterministic :class:`FakeIBKRManager` is injected in place of the real
``IBKRConnectionManager`` so the whole FastAPI lifespan can be exercised offline:
startup with TWS down, background reconnect when TWS returns, /health provenance,
and last-known-good /api/positions serving while the socket is lost.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Iterator, Mapping
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest
import src.execution.ibkr_client as ibkr_client_module
from configs.instruments import InstrumentsConfig
from configs.settings import Settings
from fastapi.testclient import TestClient
from src.api.live import (
    EQUITY_HISTORY_FILE_NAME,
    MIN_LIVE_CURVE_DAYS,
    MIN_LIVE_CURVE_POINTS,
    LiveEquityHistoryStore,
)
from src.api.main import create_app
from src.api.schemas import EquityPoint
from src.api.services import IBKRExecutionService
from src.core.types import AccountSnapshot, InstrumentId, Position
from src.execution.ibkr_client import MarginImpact

pytestmark = pytest.mark.integration

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
FIXED_TIMESTAMP = datetime(2026, 11, 2, 17, 0, tzinfo=UTC)

#: Live IBKR values deliberately different from the simulated demo book (~3.99M).
LIVE_NLV = 2_000_000.0
LIVE_QUANTITY = 7


class FakeIBKRManager:
    """Offline stand-in for :class:`~src.execution.ibkr_client.IBKRConnectionManager`."""

    def __init__(self, master: InstrumentsConfig) -> None:
        self.instruments = master
        self.is_connected = False
        self.is_reconnecting = False
        self.last_error: str | None = None
        self.heartbeat_running = False
        self.connect_calls = 0
        self.ensure_calls = 0
        self.fetch_calls = 0
        self.fail_connect = True
        self.account = AccountSnapshot(
            timestamp=FIXED_TIMESTAMP,
            net_liquidation=LIVE_NLV,
            cash_balance=LIVE_NLV * 0.5,
            buying_power=LIVE_NLV * 2.0,
            initial_margin_used=LIVE_NLV * 0.10,
            maintenance_margin_used=LIVE_NLV * 0.09,
            excess_liquidity=LIVE_NLV * 0.90,
        )
        self.positions: dict[InstrumentId, Position] = {
            "MES": Position(
                instrument_id="MES",
                quantity=LIVE_QUANTITY,
                average_price=4_800.0,
                market_price=4_800.0,
                multiplier=5.0,
                expiration=date(2026, 12, 18),
                initial_margin=1_500.0,
                maintenance_margin=1_400.0,
                contract_symbol="MESZ6",
            )
        }

    # -- OMS dependencies ------------------------------------------------------

    @property
    def price_provider(self) -> Any:
        return lambda _root: None

    async def what_if_order(self, _order: object) -> MarginImpact:
        return MarginImpact(
            initial_margin_before=0.0,
            maintenance_margin_before=0.0,
            initial_margin_change=0.0,
            maintenance_margin_change=0.0,
            initial_margin_after=0.0,
            maintenance_margin_after=0.0,
        )

    # -- connection lifecycle --------------------------------------------------

    async def connect(self) -> None:
        self.connect_calls += 1
        if self.fail_connect:
            self.last_error = "TimeoutError: TWS 127.0.0.1:7497 unavailable"
            raise ConnectionError(self.last_error)
        self.is_connected = True
        self.last_error = None

    async def ensure_connected(self, *, max_attempts: int = 1) -> bool:
        del max_attempts
        self.ensure_calls += 1
        if self.fail_connect:
            self.last_error = "TimeoutError: TWS 127.0.0.1:7497 unavailable"
            return False
        self.is_connected = True
        self.last_error = None
        return True

    async def disconnect(self) -> None:
        self.is_connected = False

    def invalidate(self, reason: str) -> None:
        self.is_connected = False
        self.last_error = reason

    def ensure_heartbeat(self) -> bool:
        return False

    # -- book ------------------------------------------------------------------

    async def fetch_account(self) -> AccountSnapshot:
        if not self.is_connected:
            raise ConnectionError("IBKR connection is not established")
        # Real IBKR observations carry the fetch wall-clock; keep them distinct so
        # the persisted live-equity history grows instead of deduping one point.
        self.fetch_calls += 1
        return dataclasses.replace(
            self.account, timestamp=FIXED_TIMESTAMP + timedelta(seconds=self.fetch_calls)
        )

    async def fetch_positions(
        self, prices: Mapping[InstrumentId, float] | None = None
    ) -> Mapping[InstrumentId, Position]:
        del prices
        if not self.is_connected:
            raise ConnectionError("IBKR connection is not established")
        return dict(self.positions)


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        environment="backtest",
        data_dir=tmp_path / "data",
        parquet_dir=tmp_path / "data" / "parquet",
        duckdb_path=tmp_path / "data" / "api.duckdb",
        report_dir=tmp_path / "artifacts",
        instruments_path=REPO_ROOT / "configs" / "instruments.yaml",
        telemetry_interval_seconds=0.05,
        api_cors_origins=("http://localhost:3080",),
        api_enable_ibkr=True,
        ibkr_connect_timeout_seconds=0.05,
        ibkr_reconnect_max_attempts=1,
        ibkr_heartbeat_interval_seconds=0.05,
    )


def _wait_for(predicate: Any, *, attempts: int = 400, interval: float = 0.01) -> bool:
    for _ in range(attempts):
        if predicate():
            return True
        time.sleep(interval)
    return False


@pytest.fixture()
def fake_manager(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[FakeIBKRManager]:
    settings = _settings(tmp_path)
    manager = FakeIBKRManager(settings.instruments)
    monkeypatch.setattr(
        ibkr_client_module, "IBKRConnectionManager", lambda *args, **kwargs: manager
    )
    monkeypatch.setattr(IBKRExecutionService, "_MIN_PROBE_INTERVAL_SECONDS", 0.01)
    yield manager


def test_reconnect_and_stale_book_end_to_end(fake_manager: FakeIBKRManager, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with TestClient(create_app(settings, clock=lambda: FIXED_TIMESTAMP)) as client:
        # --- startup with TWS down: honest offline, never simulated positions ---
        health = client.get("/health").json()
        assert health["ibkr_enabled"] is True
        assert health["ibkr_connected"] is False
        assert health["offline_mode"] is True
        assert health["ibkr_last_error"]
        assert health["status"] == "degraded"

        unavailable = client.get("/api/positions")
        assert unavailable.status_code == 503
        assert "no live account book" in unavailable.json()["detail"]

        # --- TWS comes back: the background liveness probe reconnects ----------
        fake_manager.fail_connect = False
        assert _wait_for(lambda: client.get("/health").json()["ibkr_connected"] is True)
        health = client.get("/health").json()
        assert health["offline_mode"] is False
        assert health["ibkr_enabled"] is True
        assert health["ibkr_reconnecting"] is False
        assert health["ibkr_last_error"] is None

        live = client.get("/api/positions")
        assert live.status_code == 200
        body = live.json()
        assert body["net_liquidation"] == pytest.approx(LIVE_NLV)
        assert body["stale"] is False
        assert body["stale_reason"] is None
        assert body["source"] == "engine"
        assert [row["quantity"] for row in body["positions"]] == [LIVE_QUANTITY]
        assert body["positions"][0]["instrument_id"] == "MES"

        # Live OMS order/fill rows are engine-sourced, never labelled simulated.
        working = client.get("/api/orders/working").json()
        assert working["source"] == "engine"

        # --- TWS flaps again: last-known-good book is served as stale ----------
        fake_manager.is_connected = False
        fake_manager.fail_connect = True
        assert _wait_for(lambda: client.get("/health").json()["ibkr_connected"] is False)
        stale = client.get("/api/positions")
        assert stale.status_code == 200
        body = stale.json()
        assert body["stale"] is True
        assert body["stale_reason"]
        assert "last-known-good" in body["stale_reason"]
        assert body["source"] == "engine"
        assert body["net_liquidation"] == pytest.approx(LIVE_NLV)
        assert [row["quantity"] for row in body["positions"]] == [LIVE_QUANTITY]
        # The simulated demo book must never appear.
        assert body["net_liquidation"] != pytest.approx(3_990_000.0, rel=0.05)

        # --- TWS returns again: the same service heals without a restart ------
        fake_manager.fail_connect = False
        assert _wait_for(lambda: client.get("/health").json()["ibkr_connected"] is True)
        healed = client.get("/api/positions").json()
        assert healed["stale"] is False
        assert healed["net_liquidation"] == pytest.approx(LIVE_NLV)


def test_single_probe_interval_never_busy_loops(
    fake_manager: FakeIBKRManager, tmp_path: Path
) -> None:
    """The liveness probe must not call the transport more than once per interval."""
    settings = _settings(tmp_path)
    with TestClient(create_app(settings, clock=lambda: FIXED_TIMESTAMP)) as client:
        assert _wait_for(lambda: client.get("/health").json()["ibkr_enabled"] is True)
        time.sleep(0.25)
        # ~5 probes in 250ms at a 50ms cadence, far below a busy loop (~thousands).
        assert fake_manager.ensure_calls <= 20
        assert fake_manager.ensure_calls >= 1


def test_same_day_live_points_stay_backfilled_with_intraday_curve(
    fake_manager: FakeIBKRManager, tmp_path: Path
) -> None:
    """42 same-day live NLV points must NOT promote the 6M chart to live."""
    settings = _settings(tmp_path).model_copy(update={"telemetry_interval_seconds": 3600.0})
    history_path = settings.data_dir / EQUITY_HISTORY_FILE_NAME
    seed_store = LiveEquityHistoryStore(history_path, clock=lambda: FIXED_TIMESTAMP)
    for index in range(3):
        seed_store.append(
            EquityPoint(time=FIXED_TIMESTAMP - timedelta(minutes=3 - index), value=LIVE_NLV)
        )
    with TestClient(create_app(settings, clock=lambda: FIXED_TIMESTAMP)) as client:
        fake_manager.fail_connect = False
        assert _wait_for(lambda: client.get("/health").json()["ibkr_connected"] is True)

        body = client.get("/api/positions").json()
        # Same UTC day only: the synthetic 6M backfill curve is kept (badged), while
        # the 1D chart receives the real intraday points.
        assert body["equity_curve_source"] == "simulated-backfill"
        assert body["equity_curve_live_days"] == 1
        assert 1 <= body["equity_curve_live_points"] < MIN_LIVE_CURVE_POINTS
        assert body["equity_curve_intraday"]
        assert len(body["equity_curve_intraday"]) == body["equity_curve_live_points"]
        assert all(
            item["value"] == pytest.approx(LIVE_NLV) for item in body["equity_curve_intraday"]
        )

        # Even many more same-day observations never flip the multi-day curve live.
        for _ in range(MIN_LIVE_CURVE_POINTS * 3):
            body = client.get("/api/positions").json()
        assert body["equity_curve_live_days"] == 1
        assert body["equity_curve_source"] == "simulated-backfill"
        assert body["equity_curve_live_points"] >= MIN_LIVE_CURVE_POINTS
        assert len(body["equity_curve_intraday"]) == body["equity_curve_live_points"]


def test_multi_day_live_history_promotes_real_curve(
    fake_manager: FakeIBKRManager, tmp_path: Path
) -> None:
    """>=5 distinct UTC days promote the curve to real recorded points only."""
    settings = _settings(tmp_path).model_copy(update={"telemetry_interval_seconds": 3600.0})
    history_path = settings.data_dir / EQUITY_HISTORY_FILE_NAME
    seed_store = LiveEquityHistoryStore(history_path, clock=lambda: FIXED_TIMESTAMP)
    for days_ago in range(6, 0, -1):
        seed_store.append(
            EquityPoint(time=FIXED_TIMESTAMP - timedelta(days=days_ago), value=LIVE_NLV)
        )
    with TestClient(create_app(settings, clock=lambda: FIXED_TIMESTAMP)) as client:
        fake_manager.fail_connect = False
        assert _wait_for(lambda: client.get("/health").json()["ibkr_connected"] is True)

        body = client.get("/api/positions").json()
        assert body["equity_curve_source"] == "ibkr-live"
        assert body["equity_curve_live_days"] >= MIN_LIVE_CURVE_DAYS
        assert body["equity_curve_live_points"] >= 7
        # Live mode = real recorded points only (no synthetic spline prepended).
        assert len(body["equity_curve"]) == body["equity_curve_live_points"]
        assert all(item["value"] == pytest.approx(LIVE_NLV) for item in body["equity_curve"])
        curve_times = [item["time"] for item in body["equity_curve"]]
        assert curve_times == sorted(curve_times)
        # 1D curve only contains the latest recorded UTC day (today's fetch).
        intraday = body["equity_curve_intraday"]
        assert intraday
        latest_day = curve_times[-1][:10]
        assert all(item["time"][:10] == latest_day for item in intraday)

        # Portfolio volatility is ex-ante from the live book weights/covariance.
        assert body["portfolio_volatility_source"] in {"ex-ante-book", "unavailable"}
        if body["portfolio_volatility_source"] == "ex-ante-book":
            assert body["portfolio_volatility"] is not None
            assert 0.0 < body["portfolio_volatility"] < 1.0


def _assert_same_optional_float(left: object, right: object) -> None:
    if right is None:
        assert left is None
        return
    assert left is not None
    assert float(cast(float, left)) == pytest.approx(float(cast(float, right)), rel=1e-9, abs=1e-9)


def test_websocket_ibkr_frames_match_rest_enriched_book(
    fake_manager: FakeIBKRManager, tmp_path: Path
) -> None:
    """WS pnl/margin frames must carry the same enriched IBKR book as REST (no jitter)."""
    settings = _settings(tmp_path)
    with TestClient(create_app(settings, clock=lambda: FIXED_TIMESTAMP)) as client:
        fake_manager.fail_connect = False
        assert _wait_for(lambda: client.get("/health").json()["ibkr_connected"] is True)
        rest = client.get("/api/positions").json()
        assert rest["source"] == "engine"

        seen: dict[str, dict[str, object]] = {}
        with client.websocket_connect("/ws/telemetry") as websocket:
            websocket.receive_json()  # immediate heartbeat
            for _ in range(240):
                frame = websocket.receive_json()
                if frame.get("type") != "telemetry":
                    continue
                kind = str(frame.get("kind"))
                if kind in {"pnl", "margin", "positions"}:
                    seen.setdefault(kind, frame["payload"])
                if {"pnl", "margin", "positions"} <= set(seen):
                    break
    assert {"pnl", "margin", "positions"} <= set(seen), "telemetry cycle incomplete"
    pnl = seen["pnl"]
    margin = seen["margin"]
    positions_frame = seen["positions"]

    # No demo jitter: the frame NLV/point match the REST IBKR book exactly.
    assert float(cast(float, pnl["net_liquidation"])) == pytest.approx(
        rest["net_liquidation"], abs=0.01
    )
    point = cast(dict[str, object], pnl["point"])
    assert float(cast(float, point["value"])) == pytest.approx(rest["net_liquidation"], abs=0.01)
    assert int(cast(int, pnl["open_positions"])) == len(rest["positions"])
    _assert_same_optional_float(pnl["portfolio_volatility"], rest["portfolio_volatility"])

    # Margin frame matches the same enriched book (flat-book values are exact 0.0).
    assert float(cast(float, margin["initial_margin_used"])) == pytest.approx(
        rest["initial_margin_used"]
    )
    assert float(cast(float, margin["margin_utilization"])) == pytest.approx(
        rest["margin_utilization"]
    )
    assert int(cast(int, margin["open_positions"])) == len(rest["positions"])
    _assert_same_optional_float(margin["portfolio_volatility"], rest["portfolio_volatility"])

    # Position frame carries the same book rows as REST.
    ws_positions = cast(list[dict[str, object]], positions_frame["positions"])
    assert [(row["instrument_id"], row["quantity"]) for row in ws_positions] == [
        (row["instrument_id"], row["quantity"]) for row in rest["positions"]
    ]
