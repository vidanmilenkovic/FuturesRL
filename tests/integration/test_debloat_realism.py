"""Independent verification of the de-bloat / realism batch (tasks 84-88).

Verification-only suite for branch ``feat/debloat-cost-calibration-realism``;
no production edits unless a real bug is found (reported first). Hermetic: fake
IBKR whatIf manager, tmp payloads/caches, offline settings — no network, no TWS,
no git commands, no ``.env`` edits.

Sections
--------
1. Docker/mock de-bloat hygiene (compose services, stale infra claims, mock.ts).
2. DRL PPO/CPPO-only, seasonality removal and the event-bus keep decision.
3. 45-root fee schedules, exact micro costs, roll multiplier, calibrated loader.
4. IBKR whatIf calibration endpoint (persisted exact contract, honest errors).
5. Stress-regime margins, calibrated overrides and cost-breakdown transparency.
6. README Task 6 (architecture/quick start/docker tree) contract.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
import yaml
from configs.instruments import load_instruments
from configs.settings import Settings
from fastapi.testclient import TestClient
from src.api import fixtures
from src.api.main import create_app
from src.api.services import IBKRExecutionService, calibrated_costs_path
from src.backtest.cost_model import (
    CALIBRATED_COSTS_ENV_VAR,
    DEFAULT_CALIBRATION_MAX_AGE_DAYS,
    CalibrationProvenance,
    CostModel,
)
from src.backtest.engine import EventBacktester, SimpleContractTranslator
from src.backtest.margin_simulator import MarginSimulator
from src.core.types import PipelineResult, Position, WeightVector
from src.execution.ibkr_client import MarginImpact

from tests.fixtures.market import make_bars_frame, make_market_state

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
INSTRUMENTS_PATH = REPO_ROOT / "configs" / "instruments.yaml"
FIXED = datetime(2026, 11, 7, 12, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def master() -> Any:
    return load_instruments(INSTRUMENTS_PATH)


@pytest.fixture(scope="module")
def yaml_roots() -> dict[str, dict[str, Any]]:
    payload = yaml.safe_load(INSTRUMENTS_PATH.read_text(encoding="utf-8"))
    return dict(payload["instruments"])


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    configs = tmp_path / "configs"
    strategies = configs / "strategies"
    strategies.mkdir(parents=True, exist_ok=True)
    for source in sorted((REPO_ROOT / "configs" / "strategies").glob("*.yaml")):
        shutil.copyfile(source, strategies / source.name)
    macro = REPO_ROOT / "configs" / "macro_calendar.yaml"
    if macro.is_file():
        shutil.copyfile(macro, configs / macro.name)
    values: dict[str, object] = {
        "environment": "paper",
        "configs_dir": configs,
        "data_dir": tmp_path / "data",
        "parquet_dir": tmp_path / "data" / "parquet",
        "duckdb_path": tmp_path / "data" / "debloat.duckdb",
        "report_dir": tmp_path / "artifacts",
        "instruments_path": INSTRUMENTS_PATH,
        "ibkr_account": "DU1234567",
        "ibkr_port": 7497,
        "ibkr_client_id": 42,
        "api_enable_ibkr": False,
        "telemetry_interval_seconds": 0.02,
        "api_cors_origins": ("http://localhost:3100",),
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


class _ConstantPipeline:
    name = "constant"

    def __init__(self, weights: dict[str, float]) -> None:
        self._weights = WeightVector(dict(weights))

    def run(self, state: Any) -> PipelineResult:
        return PipelineResult(
            timestamp=state.timestamp,
            universe=state.ids,
            base_weights=self._weights,
            timing_weights=self._weights,
            risk_weights=self._weights,
            final_weights=self._weights,
        )


# ---------------------------------------------------------------------------
# 1. Docker / mock de-bloat hygiene
# ---------------------------------------------------------------------------


def _compose() -> dict[str, Any]:
    return yaml.safe_load((REPO_ROOT / "docker" / "docker-compose.yml").read_text(encoding="utf-8"))


def test_docker_compose_is_debloated() -> None:
    compose = _compose()
    services = set(compose["services"])
    assert services == {"ib-gateway", "backend", "frontend"}
    text = json.dumps(compose).lower()
    for stale in ("timescaledb", "timescale", "redis", "postgres"):
        assert stale not in text, f"docker-compose still references {stale}"
    assert "backend" in compose["services"]
    assert "frontend" in compose["services"]
    # The optional IB Gateway container is headless TWS; no database/message broker.
    gateway = compose["services"]["ib-gateway"]
    assert "image" in gateway
    compose_text = (REPO_ROOT / "docker" / "docker-compose.yml").read_text(encoding="utf-8")
    for stale in ("timescaledb", "redis", "postgres"):
        assert stale not in compose_text.lower()


def test_env_and_docs_do_not_claim_removed_infrastructure() -> None:
    env_example = (REPO_ROOT / ".env.example").read_text(encoding="utf-8").lower()
    for stale in ("timescaledb", "timescale", "redis", "postgres"):
        assert stale not in env_example
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8").lower()
    for stale in ("mock.ts", "mock data", "timescaledb", "redis", "postgres"):
        assert stale not in readme, f"README still mentions {stale!r}"
    for doc in (REPO_ROOT / "docs").glob("*.md"):
        text = doc.read_text(encoding="utf-8", errors="replace").lower()
        assert "timescaledb" not in text, f"{doc.name} still claims TimescaleDB"
        for stale in ("mock.ts", "postgres"):
            assert stale not in text, f"{doc.name} still mentions {stale}"
    docker_readme = (REPO_ROOT / "docker" / "README.md").read_text(encoding="utf-8").lower()
    assert "no external database" in docker_readme
    assert "no external database, redis" in docker_readme or "no external database" in docker_readme


def test_frontend_mock_module_is_gone_and_fallbacks_are_honest() -> None:
    assert not (REPO_ROOT / "frontend" / "lib" / "mock.ts").exists()
    source_files = [
        path
        for path in (REPO_ROOT / "frontend").rglob("*.ts*")
        if not {"node_modules", ".next"} & set(path.parts)
    ]
    for path in source_files:
        text = path.read_text(encoding="utf-8", errors="replace")
        assert "@/lib/mock" not in text, f"{path.name} still imports the mock module"
        assert 'from "./mock"' not in text
    api = (REPO_ROOT / "frontend" / "lib" / "api.ts").read_text(encoding="utf-8")
    # Fallbacks are honest: schema-default/empty payload plus an error + demo source.
    assert 'source: "demo"' in api
    assert "error:" in api
    assert 'from "./offline"' in api or 'from "./offline"' in api
    offline = (REPO_ROOT / "frontend" / "lib" / "offline.ts").read_text(encoding="utf-8")
    assert "no synthetic prices" in offline.lower()
    assert "empty" in offline.lower()
    shell = (REPO_ROOT / "frontend" / "components" / "shell" / "app-shell.tsx").read_text(
        encoding="utf-8"
    )
    assert "api-offline-banner" in shell


# ---------------------------------------------------------------------------
# 2. DRL PPO/CPPO-only, seasonality removal, event bus audit
# ---------------------------------------------------------------------------


def test_drl_supports_only_ppo_and_cppo() -> None:
    from src.api.drl_service import DRL_ALGORITHMS
    from src.strategies.allocators.drl import SUPPORTED_ALGORITHMS
    from src.strategies.parameter_catalog import catalog_dict
    from src.strategies.train_drl import normalise_algorithm

    assert tuple(SUPPORTED_ALGORITHMS) == ("PPO", "CPPO")
    assert tuple(DRL_ALGORITHMS) == ("PPO", "CPPO")
    assert normalise_algorithm("ppo") == "PPO"
    assert normalise_algorithm("Cppo") == "CPPO"
    for removed in ("SAC", "TD3", "A2C"):
        with pytest.raises(ValueError):
            normalise_algorithm(removed)
    payload = catalog_dict(load_instruments(INSTRUMENTS_PATH))
    allocator = next(stage for stage in payload["stages"] if stage["id"] == "allocator")
    assert allocator["choices"] and "drl" in {choice["name"] for choice in allocator["choices"]}
    drl_algorithm = next(
        entry for entry in payload["parameters"] if entry["key"] == "allocator.drl.algorithm"
    )
    assert list(drl_algorithm["choices"]) == ["PPO", "CPPO"]
    for stage in payload["stages"]:
        text = json.dumps(stage).lower()
        for removed in ("sac", "td3", "a2c"):
            assert removed not in text
    # Frontend DRL panel is contract-limited to the same two options.
    drl_panel = (REPO_ROOT / "frontend" / "components" / "studio" / "drl-panel.tsx").read_text(
        encoding="utf-8"
    )
    assert "PPO" in drl_panel and "CPPO" in drl_panel
    for removed in ("SAC", "TD3", "A2C"):
        assert removed not in drl_panel


def test_seasonality_is_removed_everywhere() -> None:
    assert not (REPO_ROOT / "src" / "strategies" / "timing" / "seasonality.py").exists()
    assert not (REPO_ROOT / "tests" / "unit" / "test_timing_seasonality.py").exists()
    for path in (REPO_ROOT / "src").rglob("*.py"):
        assert "seasonality" not in path.read_text(encoding="utf-8", errors="replace").lower(), (
            f"{path} still references seasonality"
        )
    for path in (REPO_ROOT / "frontend").rglob("*.ts*"):
        if {"node_modules", ".next"} & set(path.parts):
            continue
        assert "seasonality" not in path.read_text(encoding="utf-8", errors="replace").lower()
    from src.strategies.parameter_catalog import catalog_dict

    payload = catalog_dict(load_instruments(INSTRUMENTS_PATH))
    timing = next(stage for stage in payload["stages"] if stage["id"] == "timing")
    assert {choice["name"] for choice in timing["choices"]} == {
        "tsmom",
        "kama",
        "carry_filter",
        "mean_reversion",
        "composite",
    }


def test_event_bus_is_kept_because_runner_and_telemetry_use_it() -> None:
    """Audit decision: EventBus is live infrastructure, so it must NOT be deleted."""
    bus = REPO_ROOT / "src" / "core" / "event_bus.py"
    assert bus.is_file()
    runner = (REPO_ROOT / "src" / "execution" / "runner.py").read_text(encoding="utf-8")
    telemetry = (REPO_ROOT / "src" / "api" / "telemetry.py").read_text(encoding="utf-8")
    assert "from src.core.event_bus import" in runner or "core.event_bus" in runner
    assert "EventBus" in telemetry or "core.event_bus" in telemetry
    core_init = (REPO_ROOT / "src" / "core" / "__init__.py").read_text(encoding="utf-8")
    assert "EventBus" in core_init


# ---------------------------------------------------------------------------
# 3. 45-root fees, micro routing, roll multiplier, calibrated loader
# ---------------------------------------------------------------------------


def test_all_45_roots_have_realistic_fee_fields(
    master: Any, yaml_roots: dict[str, dict[str, Any]]
) -> None:
    assert len(yaml_roots) == len(master.roots()) == 45
    for root in master.roots():
        raw = yaml_roots[root]
        for field in (
            "broker_commission_per_contract",
            "exchange_and_reg_fee_per_contract",
            "default_slippage_ticks",
        ):
            assert field in raw, f"{root} YAML missing {field}"
            assert float(raw[field]) > 0.0
        spec = master.get(root)
        assert spec.broker_commission_per_contract > 0.0
        assert spec.exchange_and_reg_fee_per_contract > 0.0
        assert spec.default_slippage_ticks > 0.0
        assert spec.broker_commission_per_contract <= 5.0
        assert spec.exchange_and_reg_fee_per_contract <= 10.0


def test_exact_standard_vs_micro_costs(master: Any) -> None:
    model = CostModel(instrument_master=master)
    es = model.fee_schedule("ES")
    mes = model.fee_schedule("MES")
    assert es.commission_rate(0) == pytest.approx(0.85)
    assert es.fee_per_contract == pytest.approx(1.40)
    assert mes.commission_rate(0) == pytest.approx(0.25)
    assert mes.fee_per_contract == pytest.approx(0.37)
    cl = model.fee_schedule("CL")
    mcl = model.fee_schedule("MCL")
    assert cl.commission_rate(0) == pytest.approx(0.85)
    assert cl.fee_per_contract == pytest.approx(1.52)
    assert mcl.commission_rate(0) == pytest.approx(0.25)
    assert mcl.fee_per_contract == pytest.approx(0.52)
    # The cost model charges the actual traded instrument, not its standard root.
    micro_cost = model.transaction_cost(
        "MES", 1, 5_000.0, 0.25, 5.0, bar_volume=10_000.0, monthly_volume=0
    )
    standard_cost = model.transaction_cost(
        "ES", 1, 5_000.0, 0.25, 50.0, bar_volume=10_000.0, monthly_volume=0
    )
    assert micro_cost.commission == pytest.approx(0.25)
    assert micro_cost.fees == pytest.approx(0.37)
    assert standard_cost.commission == pytest.approx(0.85)
    assert standard_cost.fees == pytest.approx(1.40)
    assert micro_cost.commission < standard_cost.commission
    assert model.commission_per_contract("MES") == pytest.approx(0.25)
    assert model.fee_schedule("MCL").commission_rate(0) == pytest.approx(0.25)


def test_roll_slippage_multiplier_applies_only_to_roll_legs(master: Any) -> None:
    model = CostModel(instrument_master=master, roll_slippage_multiplier=1.5)
    assert model.roll_slippage_multiplier == pytest.approx(1.5)
    base = CostModel(instrument_master=master, roll_slippage_multiplier=1.0)
    kwargs = {
        "root": "ES",
        "quantity": 2,
        "price": 5_000.0,
        "tick_size": 0.25,
        "multiplier": 50.0,
        "bar_volume": 10_000.0,
        "monthly_volume": 0,
    }
    normal = model.transaction_cost(**kwargs)
    roll = model.transaction_cost(**kwargs, is_roll=True)
    baseline_roll = base.transaction_cost(**kwargs, is_roll=True)
    assert roll.slippage_ticks == pytest.approx(normal.slippage_ticks * 1.5)
    assert roll.spread_cost == pytest.approx(normal.spread_cost)
    assert roll.commission == pytest.approx(normal.commission)
    assert roll.fees == pytest.approx(normal.fees)
    assert roll.slippage_cost > baseline_roll.slippage_cost
    assert roll.total > normal.total


def _write_calibration(
    path: Path,
    *,
    roots: dict[str, dict[str, float]] | None = None,
    calibrated_at: datetime | None = None,
) -> Path:
    payload = {
        "schema_version": 1,
        "account_id": "DU-CAL-1",
        "calibrated_at": (calibrated_at or (datetime.now(UTC) - timedelta(hours=1))).isoformat(),
        "roots": roots
        if roots is not None
        else {
            "ES": {
                "broker_commission_per_contract": 0.40,
                "exchange_and_reg_fee_per_contract": 1.30,
                "initial_margin": 15_500.0,
                "maintenance_margin": 14_000.0,
                "commission_source": "ibkr_whatif",
                "margin_source": "ibkr_whatif",
            },
            "NQ": {
                "broker_commission_per_contract": 0.41,
                "exchange_and_reg_fee_per_contract": 1.31,
                "initial_margin": 16_500.0,
                "maintenance_margin": 15_000.0,
                "commission_source": "ibkr_whatif",
                "margin_source": "ibkr_whatif",
            },
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_calibrated_loader_applies_overrides_and_provenance(tmp_path: Path, master: Any) -> None:
    path = _write_calibration(tmp_path / "cache" / "ibkr_calibrated_costs.json")
    model = CostModel(instrument_master=master, calibrated_costs_path=path)
    assert model.calibration is not None
    assert model.calibration.account_id == "DU-CAL-1"
    schedule = model.fee_schedule("ES")
    assert schedule.commission_rate(0) == pytest.approx(0.40)
    assert schedule.fee_per_contract == pytest.approx(1.30)
    assert model.fee_provenance("ES") is CalibrationProvenance.IBKR_CALIBRATED
    assert model.margin_provenance("ES") is CalibrationProvenance.IBKR_CALIBRATED
    assert model.calibrated_margins("ES") == (15_500.0, 14_000.0)
    metadata = model.calibration_metadata()
    assert metadata["source"] == "ibkr_calibrated"
    assert metadata["account_id"] == "DU-CAL-1"
    assert set(metadata["applied_roots"]) == {"ES", "NQ"}
    # Uncalibrated roots keep the YAML defaults/provenance.
    gc = model.fee_schedule("GC")
    assert model.fee_provenance("GC") is CalibrationProvenance.YAML_DEFAULT
    assert gc.commission_rate(0) == pytest.approx(0.85)
    # Environment override wires the default CostModel path to the file.
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setenv(CALIBRATED_COSTS_ENV_VAR, str(path))
        env_model = CostModel(instrument_master=master)
        assert env_model.calibration is not None
        assert env_model.fee_schedule("ES").commission_rate(0) == pytest.approx(0.40)
    finally:
        monkeypatch.undo()


def test_calibrated_loader_falls_back_on_missing_corrupt_or_stale(
    tmp_path: Path, master: Any
) -> None:
    missing = CostModel(
        instrument_master=master, calibrated_costs_path=tmp_path / "cache" / "missing.json"
    )
    assert missing.calibration is None
    assert missing.fee_schedule("ES").commission_rate(0) == pytest.approx(0.85)
    assert missing.fee_provenance("ES") is CalibrationProvenance.YAML_DEFAULT

    corrupt_path = tmp_path / "cache" / "corrupt.json"
    corrupt_path.parent.mkdir(parents=True, exist_ok=True)
    corrupt_path.write_text("{not-json", encoding="utf-8")
    corrupt = CostModel(instrument_master=master, calibrated_costs_path=corrupt_path)
    assert corrupt.calibration is None
    assert corrupt.calibration_error
    assert "not valid JSON" in corrupt.calibration_error
    assert corrupt.fee_provenance("ES") is CalibrationProvenance.YAML_DEFAULT
    assert corrupt.fee_schedule("ES").commission_rate(0) == pytest.approx(0.85)

    stale_path = _write_calibration(
        tmp_path / "cache" / "stale.json",
        calibrated_at=datetime.now(UTC) - timedelta(days=DEFAULT_CALIBRATION_MAX_AGE_DAYS + 10),
    )
    stale = CostModel(instrument_master=master, calibrated_costs_path=stale_path)
    assert stale.calibration is None
    assert stale.calibration_error and "stale" in stale.calibration_error
    assert stale.fee_provenance("ES") is CalibrationProvenance.YAML_DEFAULT


# ---------------------------------------------------------------------------
# 4. IBKR whatIf calibration endpoint
# ---------------------------------------------------------------------------


class _FakeWhatIfManager:
    def __init__(self, *, connected: bool = True, failures: set[str] | None = None) -> None:
        self._connected = connected
        self._failures = failures or set()
        self.what_if_calls: list[str] = []
        self.place_order_calls = 0
        self.last_error: str | None = None
        self.is_reconnecting = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def price_provider(self) -> None:
        return None

    async def fetch_account(self) -> Any:
        raise RuntimeError("account fetching is not part of the calibration path")

    async def what_if_order(self, order: Any) -> MarginImpact:
        self.what_if_calls.append(order.instrument_id)
        if order.instrument_id in self._failures:
            raise RuntimeError("whatIf rejected")
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
        raise AssertionError("calibration must never submit orders")

    async def place_bracket(self, *args: object, **kwargs: object) -> None:
        self.place_order_calls += 1
        raise AssertionError("calibration must never submit orders")


def _calibration_service(settings: Settings, manager: _FakeWhatIfManager) -> IBKRExecutionService:
    master = settings.instruments
    market = fixtures.build_offline_market(master, clock=lambda: FIXED)
    return IBKRExecutionService(
        settings=settings,
        market=market,
        clock=lambda: FIXED,
        connection_manager=manager,  # type: ignore[arg-type]
    )


@contextmanager
def _calibration_app(settings: Settings, service: IBKRExecutionService) -> Iterator[TestClient]:
    with TestClient(create_app(settings, clock=lambda: FIXED)) as client:
        client.app.state.container.execution = service  # type: ignore[attr-defined]
        yield client


def test_calibration_endpoint_success_persists_exact_contract(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    manager = _FakeWhatIfManager()
    service = _calibration_service(settings, manager)
    with _calibration_app(settings, service) as client:
        response = client.post("/api/runner/calibrate-costs", json={"roots": ["ES", "NQ"]})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["status"] == "calibrated"
        assert body["source"] == "ibkr"
        assert body["account_id"] == "DU1234567"
        assert set(body["roots"]) == {"ES", "NQ"}
        assert body["path"] == str(calibrated_costs_path(settings))
        entry = body["roots"]["ES"]
        assert entry["commission_source"] == "ibkr_whatif"
        assert entry["margin_source"] == "ibkr_whatif"
        assert entry["broker_commission_per_contract"] + entry[
            "exchange_and_reg_fee_per_contract"
        ] == pytest.approx(2.25)
        assert entry["initial_margin"] == pytest.approx(1_234.5)
        assert entry["maintenance_margin"] == pytest.approx(1_000.0)
        assert manager.what_if_calls == ["ES", "NQ"]
        assert manager.place_order_calls == 0, "calibration must be read-only"

        path = calibrated_costs_path(settings)
        assert path.is_file()
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["schema_version"] == 1
        assert payload["account_id"] == "DU1234567"
        assert payload["calibrated_at"]
        assert set(payload["roots"]) == {"ES", "NQ"}
        assert payload["roots"]["ES"]["initial_margin"] == pytest.approx(1_234.5)

        status = client.get("/api/runner/calibrated-costs")
        assert status.status_code == 200
        status_body = status.json()
        assert status_body["available"] is True
        assert status_body["source"] == "ibkr"
        assert status_body["account_id"] == "DU1234567"
        assert set(status_body["roots"]) == {"ES", "NQ"}
        assert status_body["age_seconds"] is not None
        # The persisted file is loader-readable by the backtest layer.
        model = CostModel(
            instrument_master=load_instruments(INSTRUMENTS_PATH), calibrated_costs_path=path
        )
        assert model.calibration is not None
        assert model.calibrated_margins("ES") == (1_234.5, 1_000.0)
        assert model.fee_provenance("ES") is CalibrationProvenance.IBKR_CALIBRATED
        simulator = MarginSimulator(calibrated_costs_path=path)
        assert simulator.calibrated_margins["ES"].initial_margin == pytest.approx(1_234.5)


def test_calibration_endpoint_partial_and_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path / "partial")
    manager = _FakeWhatIfManager(failures={"NQ"})
    service = _calibration_service(settings, manager)
    with _calibration_app(settings, service) as client:
        response = client.post("/api/runner/calibrate-costs", json={"roots": ["ES", "NQ"]})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["status"] == "partial"
        assert set(body["roots"]) == {"ES"}
        assert any("NQ" in warning for warning in body["warnings"])
        assert calibrated_costs_path(settings).is_file()

    from src.backtest.cost_model import CALIBRATED_COSTS_ENV_VAR

    monkeypatch.delenv(CALIBRATED_COSTS_ENV_VAR, raising=False)
    all_failed = _settings(tmp_path / "unavailable")
    failing = _FakeWhatIfManager(failures={"ES", "NQ"})
    failed_service = _calibration_service(all_failed, failing)
    with _calibration_app(all_failed, failed_service) as client:
        response = client.post("/api/runner/calibrate-costs", json={"roots": ["ES", "NQ"]})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["status"] == "unavailable"
        assert body["roots"] == {}
        assert body["path"] is None
        assert not calibrated_costs_path(all_failed).is_file()
        status = client.get("/api/runner/calibrated-costs").json()
        assert status["available"] is False
        assert status["source"] == "none"


def test_calibration_endpoint_disconnected_and_unknown_root(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with TestClient(create_app(settings, clock=lambda: FIXED)) as client:
        refused = client.post("/api/runner/calibrate-costs", json={"roots": ["ES"]})
        assert refused.status_code == 409, refused.text
        assert "IBKR" in refused.json()["detail"]

    manager = _FakeWhatIfManager()
    service = _calibration_service(settings, manager)
    with _calibration_app(settings, service) as client:
        unknown = client.post("/api/runner/calibrate-costs", json={"roots": ["NOPE"]})
        assert unknown.status_code == 422, unknown.text
        assert "NOPE" in unknown.json()["detail"]


# ---------------------------------------------------------------------------
# 5. Stress margins, calibrated overrides, cost breakdown
# ---------------------------------------------------------------------------


def _position(root: str, *, quantity: int = 2) -> Position:
    return Position(
        instrument_id=root,
        quantity=quantity,
        average_price=4_800.0,
        market_price=4_800.0,
        multiplier=50.0,
        initial_margin=10_000.0,
        maintenance_margin=9_000.0,
        expiration=date(2024, 3, 15),
    )


def test_stress_margin_multiplier_high_vol_only() -> None:
    simulator = MarginSimulator(stress_margin_multiplier=1.35, stress_vol_threshold=0.30)
    simulator.set_volatilities({"ES": 0.20})
    normal = simulator.margin_requirements({"ES": _position("ES")})
    assert normal.initial_margin == pytest.approx(20_000.0)
    assert normal.maintenance_margin == pytest.approx(18_000.0)
    assert normal.stress_applied is False
    assert normal.stress_multiplier == pytest.approx(1.0)

    simulator.set_volatilities({"ES": 0.45})
    stressed = simulator.margin_requirements({"ES": _position("ES")})
    assert stressed.initial_margin == pytest.approx(20_000.0 * 1.35)
    assert stressed.maintenance_margin == pytest.approx(18_000.0 * 1.35)
    assert stressed.stress_applied is True
    assert stressed.stress_multiplier == pytest.approx(1.35)
    assert stressed.stress_roots == ("ES",)
    assert stressed.to_dict()["stress_multiplier"] == pytest.approx(1.35)
    # Normal/other roots remain unchanged.
    simulator.set_volatilities({"ES": 0.45, "NQ": 0.10})
    mixed = simulator.margin_requirements({"ES": _position("ES"), "NQ": _position("NQ")})
    assert mixed.stress_roots == ("ES",)


def test_calibrated_margins_apply_before_stress(tmp_path: Path, master: Any) -> None:
    path = _write_calibration(tmp_path / "cache" / "ibkr_calibrated_costs.json")
    simulator = MarginSimulator(
        instrument_master=master, calibrated_costs_path=path, stress_vol_threshold=0.30
    )
    calibrated = simulator.margin_requirements({"ES": _position("ES")})
    assert calibrated.initial_margin == pytest.approx(15_500.0 * 2)
    assert calibrated.maintenance_margin == pytest.approx(14_000.0 * 2)
    assert calibrated.calibrated is True
    assert calibrated.calibrated_roots == ("ES",)
    simulator.set_volatilities({"ES": 0.50})
    stressed = simulator.margin_requirements({"ES": _position("ES")})
    assert stressed.initial_margin == pytest.approx(15_500.0 * 2 * 1.35)
    assert stressed.calibrated is True
    assert stressed.stress_applied is True
    # Corrupt/missing calibration falls back to YAML margins cleanly.
    bad = tmp_path / "cache" / "bad.json"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("{bad", encoding="utf-8")
    fallback = MarginSimulator(instrument_master=master, calibrated_costs_path=bad)
    assert fallback.calibrated_margins == {}
    assert fallback.margin_requirements({"ES": _position("ES")}).calibrated is False


def _run_engine_with_rolls(master: Any, multiplier: float) -> Any:
    first = make_bars_frame(("ES",), periods=3, expiration=date(2024, 3, 15), seed=3)
    second = make_bars_frame(
        ("ES",),
        periods=3,
        start=first["timestamp_utc"][0] + timedelta(days=3),
        expiration=date(2024, 6, 21),
        seed=3,
    )
    bars = pl.concat([first, second]).sort(["timestamp_utc", "root_symbol"])
    cost_model = CostModel(instrument_master=master, roll_slippage_multiplier=multiplier)
    engine = EventBacktester(
        _ConstantPipeline({"ES": 0.50}),
        SimpleContractTranslator(master),
        instrument_master=master,
        cost_model=cost_model,
        initial_equity=1_000_000.0,
    )
    return engine.run(bars=bars)


def test_backtest_result_cost_breakdown_and_roll_multiplier(master: Any) -> None:
    baseline = _run_engine_with_rolls(master, 1.0)
    multiplied = _run_engine_with_rolls(master, 1.5)
    assert len(baseline.roll_events) == len(multiplied.roll_events) == 1
    assert multiplied.cost_breakdown["roll_slippage"] > baseline.cost_breakdown["roll_slippage"]
    assert multiplied.roll_cost_total > baseline.roll_cost_total
    breakdown = multiplied.cost_breakdown
    for key in (
        "broker_commissions",
        "exchange_and_regulatory_fees",
        "slippage",
        "funding",
        "roll_cost_total",
        "roll_commissions",
        "roll_exchange_and_regulatory_fees",
        "roll_slippage",
        "non_roll_commissions",
        "non_roll_exchange_and_regulatory_fees",
        "non_roll_slippage",
        "total_commissions",
        "total_execution_costs",
    ):
        assert key in breakdown, f"missing cost breakdown field {key}"
    assert breakdown["total_costs"] == pytest.approx(multiplied.total_costs)
    assert breakdown["broker_commissions"] == pytest.approx(multiplied.commissions)
    assert breakdown["exchange_and_regulatory_fees"] == pytest.approx(multiplied.fees)
    assert breakdown["slippage"] == pytest.approx(multiplied.slippage)
    assert breakdown["roll_commissions"] + breakdown[
        "roll_exchange_and_regulatory_fees"
    ] + breakdown["roll_slippage"] == pytest.approx(multiplied.roll_cost_total)
    assert "cost_breakdown" in multiplied.to_dict()
    multiplied.check_accounting_identity()


def test_event_engine_reports_stress_margins(master: Any) -> None:
    high_vol = make_market_state(("ES",), equity=1_000_000.0)
    base_state = dataclasses_replace_cov(high_vol, volatility=0.45)
    states = [
        dataclasses_replace_timestamp(base_state, timestamp=base_state.timestamp) for _ in range(1)
    ]
    engine = EventBacktester(
        _ConstantPipeline({"ES": 0.30}),
        SimpleContractTranslator(master),
        instrument_master=master,
        initial_equity=1_000_000.0,
    )
    result = engine.run(states=states)
    assert result.margin_stress_applied is True
    assert result.margin_stress_multiplier == pytest.approx(1.35)
    assert result.margin_stress_roots == ("ES",)
    assert result.to_dict()["margin_stress_applied"] is True


def dataclasses_replace_cov(state: Any, *, volatility: float) -> Any:
    from dataclasses import replace

    return replace(
        state,
        covariance=np.array([[volatility * volatility]]),
        covariance_instruments=("ES",),
    )


def dataclasses_replace_timestamp(state: Any, *, timestamp: datetime) -> Any:
    from dataclasses import replace

    return replace(state, timestamp=timestamp)


def test_api_tearsheet_exposes_cost_breakdown(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    start = (FIXED - timedelta(days=300)).date()
    end = (FIXED - timedelta(days=30)).date()
    with TestClient(create_app(settings, clock=lambda: FIXED)) as client:
        run = client.post(
            "/api/backtests",
            json={
                "strategy_id": "tsmom_vol_target",
                "start": start.isoformat(),
                "end": end.isoformat(),
                "data_mode": "synthetic_offline",
                "timeframe": "1d",
            },
        )
        assert run.status_code == 200, run.text
        body = run.json()
        tearsheet = client.get(f"/api/backtests/{body['id']}/tearsheet")
        assert tearsheet.status_code == 200, tearsheet.text
        sheet = tearsheet.json()
        breakdown = sheet.get("cost_breakdown") or (body.get("cost_breakdown") or {})
        assert breakdown, "tearsheet/run must expose the execution-cost breakdown"
        for key in (
            "broker_commissions",
            "exchange_and_regulatory_fees",
            "slippage",
            "roll_cost_total",
            "total_costs",
        ):
            assert key in breakdown, f"API cost breakdown missing {key}"
            assert float(breakdown[key]) >= 0.0


# ---------------------------------------------------------------------------
# 6. README Task 6 (architecture / quick start / docker tree)
# ---------------------------------------------------------------------------


def test_readme_task6_contract() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    architecture = readme.split("## Architecture", 1)[1].split("## ", 1)[0]
    assert "WIP" not in architecture and "TODO" not in architecture
    assert "Data Layer" in architecture
    assert "Strategy Layer" in architecture
    assert "Execution & OMS Layer" in architecture
    for row in ("| Data |", "| Strategy |", "| Risk & Translation |", "| Execution & OMS |"):
        assert row in architecture, f"architecture table missing {row}"
    quick = readme.split("## Quick Start", 1)[1].split("## ", 1)[0]
    assert "```bash" in quick
    assert "cd FuturesRL" in quick
    assert quick.count("```") >= 2
    assert "docker/                      #" in readme or "docker/                      #" in readme
    lowered = readme.lower()
    for stale in ("mock.ts", "timescaledb", "redis", "postgres"):
        assert stale not in lowered
