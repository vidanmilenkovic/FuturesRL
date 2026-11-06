"""Workstation UX: IBKR port discovery + guarded connection switch API tests."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from configs.settings import Settings
from fastapi.testclient import TestClient
from src.api.main import create_app
from src.api.routes import runner as runner_routes

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXED = datetime(2026, 11, 6, 12, 0, tzinfo=UTC)
TELEMETRY_INTERVAL = 0.02


def _settings(tmp_path: Path, *, environment: str = "backtest", port: int = 7497) -> Settings:
    configs = tmp_path / "configs"
    strategies = configs / "strategies"
    strategies.mkdir(parents=True, exist_ok=True)
    for source in sorted((REPO_ROOT / "configs" / "strategies").glob("*.yaml")):
        shutil.copyfile(source, strategies / source.name)
    return Settings(
        environment=environment,
        configs_dir=configs,
        data_dir=tmp_path / "data",
        parquet_dir=tmp_path / "data" / "parquet",
        duckdb_path=tmp_path / "data" / "api.duckdb",
        report_dir=tmp_path / "artifacts",
        instruments_path=REPO_ROOT / "configs" / "instruments.yaml",
        ibkr_port=port,
        ibkr_client_id=42,
        telemetry_interval_seconds=TELEMETRY_INTERVAL,
        api_cors_origins=("http://localhost:3080",),
    )


def _app(settings: Settings) -> TestClient:
    return TestClient(create_app(settings, clock=lambda: FIXED))


@pytest.fixture()
def client(tmp_path: Path) -> Iterator[TestClient]:
    with _app(_settings(tmp_path)) as test_client:
        yield test_client


def test_ibkr_ports_probe_contract(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_probe(
        host: str, port: int, *, timeout: float = 0.25
    ) -> tuple[bool, float | None]:
        del host, timeout
        if port in {7497, 7496}:
            return True, 11.5
        return False, None

    monkeypatch.setattr(runner_routes, "probe_ibkr_port", fake_probe)
    response = client.get("/api/runner/ibkr-ports")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["host"] == "127.0.0.1"
    assert [row["port"] for row in body["ports"]] == [7497, 4002, 7496, 4001]
    by_port = {row["port"]: row for row in body["ports"]}
    assert by_port[7497]["mode"] == "paper" and by_port[7497]["gateway"] == "tws"
    assert by_port[4002]["gateway"] == "ib_gateway"
    assert by_port[7496]["mode"] == "live"
    assert by_port[7497]["reachable"] is True
    assert by_port[7497]["latency_ms"] == 11.5
    assert by_port[4002]["reachable"] is False and by_port[4002]["latency_ms"] is None
    assert body["active"] == {
        "port": 7497,
        "client_id": 42,
        "environment": "backtest",
        "mode": "paper",
        "gateway": "tws",
    }
    assert isinstance(body["warnings"], list)
    assert body["probed_at"]


def test_ibkr_switch_paper_reports_restart_required(client: TestClient) -> None:
    response = client.post(
        "/api/runner/ibkr-connection",
        json={"port": 4002, "client_id": 43},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "restart_required"
    assert body["applied"] is False
    assert body["restart_required"] is True
    assert body["requested"]["port"] == 4002
    assert body["requested"]["mode"] == "paper"
    assert body["requested"]["client_id"] == 43
    # Never claims the switch happened: active remains the configured connection.
    assert body["active"]["port"] == 7497
    assert "NOT switched" in body["message"]


def test_ibkr_switch_live_confirmation_and_paper_environment(tmp_path: Path) -> None:
    with _app(_settings(tmp_path / "backtest")) as client:
        wrong = client.post(
            "/api/runner/ibkr-connection",
            json={"port": 7496, "confirmation": "live"},
        )
        assert wrong.status_code == 400
        assert "LIVE" in wrong.json()["detail"]

        valid = client.post(
            "/api/runner/ibkr-connection",
            json={"port": 7496, "confirmation": "LIVE"},
        )
        assert valid.status_code == 200, valid.text
        assert valid.json()["status"] == "restart_required"

    paper_settings = _settings(tmp_path / "paper", environment="paper", port=7497)
    with _app(paper_settings) as paper_client:
        refused = paper_client.post(
            "/api/runner/ibkr-connection",
            json={"port": 7496, "confirmation": "LIVE"},
        )
        assert refused.status_code == 409, refused.text
        assert "paper environment" in refused.json()["detail"]


def test_ibkr_switch_applied_when_adapter_supports_runtime_reconnect(
    client: TestClient,
) -> None:
    class FakeExecution:
        async def switch_connection(self, port: int, client_id: int) -> bool:
            return port == 4002 and client_id == 43

    client.app.state.container.execution = FakeExecution()  # type: ignore[attr-defined]
    response = client.post(
        "/api/runner/ibkr-connection",
        json={"port": 4002, "client_id": 43},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "applied"
    assert body["applied"] is True
    assert body["restart_required"] is False
    assert body["active"]["port"] == 4002
    assert body["active"]["client_id"] == 43
    assert body["active"]["mode"] == "paper"


def test_ibkr_switch_invalid_port_is_422(client: TestClient) -> None:
    response = client.post("/api/runner/ibkr-connection", json={"port": 1234, "client_id": 42})
    assert response.status_code == 422
