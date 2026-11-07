"""IBKR whatIf cost/margin calibration endpoint + persisted override tests."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from configs.settings import Settings
from fastapi.testclient import TestClient
from src.api import fixtures
from src.api.main import create_app
from src.api.services import IBKRExecutionService, calibrated_costs_path
from src.execution.ibkr_client import MarginImpact

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXED = datetime(2026, 11, 7, 12, 0, tzinfo=UTC)
TELEMETRY_INTERVAL = 0.02


def _settings(tmp_path: Path) -> Settings:
    configs = tmp_path / "configs"
    strategies = configs / "strategies"
    strategies.mkdir(parents=True, exist_ok=True)
    for source in sorted((REPO_ROOT / "configs" / "strategies").glob("*.yaml")):
        shutil.copyfile(source, strategies / source.name)
    return Settings(
        environment="paper",
        configs_dir=configs,
        data_dir=tmp_path / "data",
        parquet_dir=tmp_path / "data" / "parquet",
        duckdb_path=tmp_path / "data" / "api.duckdb",
        report_dir=tmp_path / "artifacts",
        instruments_path=REPO_ROOT / "configs" / "instruments.yaml",
        ibkr_account="DU1234567",
        ibkr_port=7497,
        ibkr_client_id=42,
        telemetry_interval_seconds=TELEMETRY_INTERVAL,
        api_cors_origins=("http://localhost:3080",),
    )


class FakeIBKRManager:
    """Minimal connection-manager surface for whatIf calibration tests."""

    def __init__(self, *, connected: bool = True, failures: set[str] | None = None) -> None:
        self._connected = connected
        self._failures = failures or set()
        self.what_if_calls: list[str] = []
        self.place_order_calls = 0
        self.last_error: str | None = None
        self.is_reconnecting = False

    @property
    def instruments(self) -> Any:
        return self._master

    def set_master(self, master: Any) -> None:
        self._master = master

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def price_provider(self) -> None:
        return None

    async def fetch_account(self) -> Any:
        raise RuntimeError("fetch_account is not part of the calibration test path")

    async def what_if_order(self, order: Any) -> MarginImpact:
        self.what_if_calls.append(order.instrument_id)
        if order.instrument_id in self._failures:
            raise RuntimeError("whatIf rejected for test")
        return MarginImpact(
            initial_margin_before=10_000.0,
            maintenance_margin_before=9_000.0,
            initial_margin_change=1_234.5,
            maintenance_margin_change=1_000.0,
            initial_margin_after=11_234.5,
            maintenance_margin_after=10_000.0,
            commission=2.25,
            commission_currency="USD",
        )

    async def place_order(self, *args: object, **kwargs: object) -> None:
        self.place_order_calls += 1
        raise AssertionError("calibration must never submit an order")


def _app(settings: Settings) -> TestClient:
    return TestClient(create_app(settings, clock=lambda: FIXED))


@pytest.fixture()
def client(tmp_path: Path) -> Iterator[TestClient]:
    with _app(_settings(tmp_path)) as test_client:
        yield test_client


def _fake_execution_response(service: IBKRExecutionService, roots: list[str] | None) -> Any:
    return service.calibrate_costs(roots)


def test_calibration_route_refuses_simulated_adapter(client: TestClient) -> None:
    response = client.post("/api/runner/calibrate-costs", json={"roots": ["ES"]})
    assert response.status_code == 409, response.text
    assert "IBKR" in response.json()["detail"]

    status = client.get("/api/runner/calibrated-costs")
    assert status.status_code == 200, status.text
    body = status.json()
    assert body["available"] is False
    assert body["source"] == "none"
    assert body["roots"] == {}


def test_calibration_service_read_only_persists_exact_contract(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    manager = FakeIBKRManager(failures={"NQ"})
    master = settings.instruments
    manager.set_master(master)
    market = fixtures.build_offline_market(master, clock=lambda: FIXED)
    service = IBKRExecutionService(
        settings=settings,
        market=market,
        clock=lambda: FIXED,
        connection_manager=manager,  # type: ignore[arg-type]
    )

    import asyncio

    result = asyncio.run(service.calibrate_costs(["ES", "NQ"]))
    assert result["status"] == "partial"
    assert result["source"] == "ibkr"
    assert result["account_id"] == "DU1234567"
    assert set(result["roots"]) == {"ES"}
    entry = result["roots"]["ES"]
    assert entry["commission_source"] == "ibkr_whatif"
    assert entry["margin_source"] == "ibkr_whatif"
    assert entry["broker_commission_per_contract"] + entry[
        "exchange_and_reg_fee_per_contract"
    ] == pytest.approx(2.25)
    assert entry["initial_margin"] == pytest.approx(1_234.5)
    assert entry["maintenance_margin"] == pytest.approx(1_000.0)
    assert any("NQ" in warning for warning in result["warnings"])
    assert manager.what_if_calls == ["ES", "NQ"]
    assert manager.place_order_calls == 0  # no submission ever

    path = calibrated_costs_path(settings)
    assert path.is_file()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["account_id"] == "DU1234567"
    assert payload["calibrated_at"]
    assert set(payload["roots"]) == {"ES"}
    assert payload["warnings"]
    assert payload["roots"]["ES"]["initial_margin"] == pytest.approx(1_234.5)


def test_calibrated_costs_get_and_route_success(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    manager = FakeIBKRManager()
    master = settings.instruments
    manager.set_master(master)
    market = fixtures.build_offline_market(master, clock=lambda: FIXED)
    service = IBKRExecutionService(
        settings=settings,
        market=market,
        clock=lambda: FIXED,
        connection_manager=manager,  # type: ignore[arg-type]
    )
    with _app(settings) as client:
        client.app.state.container.execution = service  # type: ignore[attr-defined]
        calibrated = client.post("/api/runner/calibrate-costs", json={"roots": ["ES", "NQ"]})
        assert calibrated.status_code == 200, calibrated.text
        body = calibrated.json()
        assert body["status"] == "calibrated"
        assert body["source"] == "ibkr"
        assert set(body["roots"]) == {"ES", "NQ"}
        assert body["path"] == str(calibrated_costs_path(settings))

        status = client.get("/api/runner/calibrated-costs")
        assert status.status_code == 200, status.text
        status_body = status.json()
        assert status_body["available"] is True
        assert status_body["source"] == "ibkr"
        assert status_body["account_id"] == "DU1234567"
        assert set(status_body["roots"]) == {"ES", "NQ"}
        assert status_body["age_seconds"] is not None


def test_calibration_unknown_root_is_422(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    manager = FakeIBKRManager()
    manager.set_master(settings.instruments)
    market = fixtures.build_offline_market(settings.instruments, clock=lambda: FIXED)
    service = IBKRExecutionService(
        settings=settings,
        market=market,
        clock=lambda: FIXED,
        connection_manager=manager,  # type: ignore[arg-type]
    )
    with _app(settings) as client:
        client.app.state.container.execution = service  # type: ignore[attr-defined]
        response = client.post("/api/runner/calibrate-costs", json={"roots": ["NOPE"]})
        assert response.status_code == 422


def test_calibration_path_honours_operator_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    custom = tmp_path / "custom-calibration.json"
    monkeypatch.setenv("FUTURESRL_IBKR_CALIBRATED_COSTS_PATH", str(custom))
    settings = _settings(tmp_path)
    assert calibrated_costs_path(settings) == custom
