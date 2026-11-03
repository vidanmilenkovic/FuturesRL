"""Phase-10 integration tests: API-owned IBKR session resilience (task-22).

A deterministic :class:`FakeIBKRManager` is injected in place of the real
``IBKRConnectionManager`` so the whole FastAPI lifespan can be exercised offline:
startup with TWS down, background reconnect when TWS returns, /health provenance,
and last-known-good /api/positions serving while the socket is lost.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest
import src.execution.ibkr_client as ibkr_client_module
from configs.instruments import InstrumentsConfig
from configs.settings import Settings
from fastapi.testclient import TestClient
from src.api.main import create_app
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
        return self.account

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
