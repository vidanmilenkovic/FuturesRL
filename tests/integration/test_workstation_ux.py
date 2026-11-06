"""Independent verification of the Workstation UX batch (tasks 78-82).

Verification-only suite for branch ``feat/workstation-ux-llm-root-diagnostics``;
no production edits unless a real bug is found (reported first). Everything is
hermetic: fake LLM clients, fake IBKR TCP probes, tmp configs/Parquet, offline
provider settings — no network, no TWS, no git commands and no ``.env`` edits.

Sections
--------
1. LLM provider status/ping endpoints + additive ``/health.llm_status``.
2. 45-root query scopes for term structure, pipeline and Risk & LLM state.
3. IBKR port discovery (7497/4002/7496/4001) + guarded connection switch.
4. Composite timing serialization + parameter-catalog metadata.
5. Frontend contract sanity for the new typed fields/UI hooks.
6. Test hygiene: no ``test_spec``/``test_phase`` files and the OSS guard intact.
"""

from __future__ import annotations

import asyncio
import dataclasses
import shutil
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from configs.instruments import InstrumentsConfig, load_instruments
from configs.settings import Settings
from configs.strategy import StageConfig, StrategyConfig, TranslatorConfig, parse_strategy_config
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError
from src.api.llm_status import LLMStatusService
from src.api.main import create_app
from src.api.routes import runner as runner_routes
from src.api.services import build_container
from src.data.storage import ParquetRepository
from src.strategies.parameter_catalog import (
    ADVANCED_PARAMETER_KEYS,
    ALLOCATOR_FORMULAS,
    catalog_dict,
)
from src.strategies.pipeline import build_pipeline

from tests.fixtures.market import make_bars_frame, make_market_state

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
INSTRUMENTS_PATH = REPO_ROOT / "configs" / "instruments.yaml"
FIXED = datetime(2026, 11, 6, 12, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def master() -> InstrumentsConfig:
    return load_instruments(INSTRUMENTS_PATH)


def _base_settings(tmp_path: Path, **overrides: object) -> Settings:
    configs = tmp_path / "configs"
    strategies = configs / "strategies"
    strategies.mkdir(parents=True, exist_ok=True)
    for source in sorted((REPO_ROOT / "configs" / "strategies").glob("*.yaml")):
        shutil.copyfile(source, strategies / source.name)
    macro = REPO_ROOT / "configs" / "macro_calendar.yaml"
    if macro.is_file():
        shutil.copyfile(macro, configs / macro.name)
    values: dict[str, object] = {
        "environment": "backtest",
        "configs_dir": configs,
        "data_dir": tmp_path / "data",
        "parquet_dir": tmp_path / "data" / "parquet",
        "duckdb_path": tmp_path / "data" / "ux.duckdb",
        "report_dir": tmp_path / "artifacts",
        "instruments_path": INSTRUMENTS_PATH,
        "telemetry_interval_seconds": 0.02,
        "api_enable_ibkr": False,
        "api_cors_origins": ("http://localhost:3100",),
        "ibkr_host": "127.0.0.1",
        "ibkr_port": 7497,
        "ibkr_client_id": 42,
        "llm_provider": "offline",
        "llm_model": "gpt-4o-mini",
        "llm_api_key": None,
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


@dataclasses.dataclass
class _Harness:
    client: TestClient
    settings: Settings
    container: Any


def _app(settings: Settings) -> Iterator[TestClient]:
    return TestClient(create_app(settings, clock=lambda: FIXED))


class _FakeLLMClient:
    def __init__(
        self, *, response: str | None = '{"ok": true}', error: BaseException | None = None
    ) -> None:
        self.response = response
        self.error = error
        self.calls: list[str] = []

    async def complete(
        self, prompt: str, *, system_prompt: str | None = None, temperature: float = 0.0
    ) -> str:
        del system_prompt, temperature
        self.calls.append(prompt)
        if self.error is not None:
            raise self.error
        return self.response or ""


class _SlowLLMClient:
    async def complete(self, *_: object, **__: object) -> str:
        await asyncio.sleep(5.0)
        return '{"ok": true}'


def _llm_client(
    settings: Settings,
    *,
    client: object | None = None,
    monotonic: list[float] | None = None,
    cache_entries: int | None = None,
    timeout: float = 5.0,
) -> LLMStatusService:
    values = iter(monotonic or [1.0, 1.25])
    return LLMStatusService(
        settings=settings,
        clock=lambda: FIXED,
        monotonic=lambda: next(values),
        client_factory=(lambda: client) if client is not None else None,
        ping_timeout_seconds=timeout,
        cache_entries_provider=(lambda: cache_entries) if cache_entries is not None else None,
    )


def _with_llm_service(settings: Settings, service: LLMStatusService) -> TestClient:
    container = build_container(settings, clock=lambda: FIXED)
    patched = dataclasses.replace(container, llm_status=service)
    return TestClient(create_app(settings, clock=lambda: FIXED, container=patched))


# ---------------------------------------------------------------------------
# 1. LLM provider status / ping / health
# ---------------------------------------------------------------------------


def test_llm_status_offline_stub_ping_is_network_free(tmp_path: Path) -> None:
    settings = _base_settings(tmp_path)
    fake = _FakeLLMClient()
    with _with_llm_service(settings, _llm_client(settings, client=fake, cache_entries=3)) as client:
        status = client.get("/api/risk/llm-status")
        assert status.status_code == 200, status.text
        body = status.json()
        assert body["provider"] == "offline"
        assert body["model"] == "gpt-4o-mini"
        assert body["mode"] == "offline_stub"
        assert body["status"] == "offline_stub"
        assert body["api_key_configured"] is False
        assert body["last_latency_ms"] is None
        assert body["last_checked_at"] is None
        assert body["cache_entries"] == 3
        assert body["error_detail"] is None
        # Read-only status must not ping.
        assert fake.calls == []
        ping = client.post("/api/risk/llm-ping")
        assert ping.status_code == 200, ping.text
        pinged = ping.json()
        assert pinged["mode"] == "offline_stub"
        assert pinged["status"] == "offline_stub"
        assert pinged["last_checked_at"] is not None
        assert fake.calls == [], "offline stub must never touch the network"


def test_llm_ping_online_success_and_health_field(tmp_path: Path) -> None:
    settings = _base_settings(
        tmp_path, llm_provider="openai", llm_api_key=SecretStr("sk-test"), llm_model="gpt-4o-mini"
    )
    fake = _FakeLLMClient()
    with _with_llm_service(settings, _llm_client(settings, client=fake)) as client:
        before = client.get("/api/risk/llm-status").json()
        assert before["mode"] == "live_api"
        assert before["status"] == "unreachable"  # honest "not pinged yet"
        assert before["api_key_configured"] is True
        response = client.post("/api/risk/llm-ping")
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["status"] == "online"
        assert body["last_latency_ms"] == pytest.approx(250.0)
        assert body["last_checked_at"] is not None
        assert body["error_detail"] is None
        assert fake.calls and "Health check" in fake.calls[0]
        health = client.get("/health").json()
        assert health["llm_status"] == body


def test_llm_ping_auth_and_unreachable_classification(tmp_path: Path) -> None:
    settings = _base_settings(
        tmp_path, llm_provider="openai", llm_api_key=SecretStr("sk-bad"), llm_model="gpt-4o-mini"
    )
    auth = _FakeLLMClient(error=RuntimeError("401 Unauthorized: invalid api key"))
    with _with_llm_service(settings, _llm_client(settings, client=auth)) as client:
        body = client.post("/api/risk/llm-ping").json()
        assert body["status"] == "auth_error"
        assert "401" in (body["error_detail"] or "")
        assert body["last_latency_ms"] is None

    unreachable = _FakeLLMClient(error=ConnectionError("network is unreachable"))
    with _with_llm_service(settings, _llm_client(settings, client=unreachable)) as client:
        body = client.post("/api/risk/llm-ping").json()
        assert body["status"] == "unreachable"
        assert body["error_detail"]

    with _with_llm_service(
        settings, _llm_client(settings, client=_SlowLLMClient(), timeout=0.01)
    ) as client:
        body = client.post("/api/risk/llm-ping").json()
        assert body["status"] == "unreachable"
        assert "timed out" in (body["error_detail"] or "").lower()


def test_llm_status_missing_key_is_honest_offline_stub(tmp_path: Path) -> None:
    settings = _base_settings(tmp_path)
    missing_key = settings.model_copy(update={"llm_provider": "openai", "llm_api_key": None})
    service = _llm_client(missing_key)
    assert service.mode == "offline_stub"
    assert service.api_key_configured is False
    with _with_llm_service(settings, service) as client:
        body = client.get("/api/risk/llm-status").json()
        assert body["provider"] == "openai"
        assert body["mode"] == "offline_stub"
        assert body["status"] == "offline_stub"
        assert body["api_key_configured"] is False
        pinged = client.post("/api/risk/llm-ping").json()
        assert pinged["status"] == "offline_stub"
        assert "API key" in (pinged["error_detail"] or "")


# ---------------------------------------------------------------------------
# 2. 45-root term structure / pipeline / risk scopes
# ---------------------------------------------------------------------------


def test_term_structure_default_vs_all_45_roots(tmp_path: Path, master: InstrumentsConfig) -> None:
    with _app(_base_settings(tmp_path)) as client:
        default = client.get("/api/term-structure")
        assert default.status_code == 200, default.text
        default_body = default.json()
        assert default_body["scope"] == "strategy"
        assert 0 < len(default_body["curves"]) < 45
        all_roots = client.get("/api/term-structure?scope=all")
        assert all_roots.status_code == 200, all_roots.text
        body = all_roots.json()
        assert body["scope"] == "all"
        assert len(body["selected_roots"]) == 45
        assert len(body["curves"]) == 45
        assert sorted(master.roots()) == body["selected_roots"]
        assert {curve["root"] for curve in body["curves"]} == set(body["selected_roots"])
        assert body["data_origin"] == "deterministic_instrument_master"
        assert all_roots.headers["X-Data-Origin"] == body["data_origin"]
        assert all_roots.headers["X-Data-Roots"] == "45"
        assert all_roots.headers["X-Scope-Roots"].count(",") == 44


def test_term_structure_sector_and_root_filters(tmp_path: Path, master: InstrumentsConfig) -> None:
    with _app(_base_settings(tmp_path)) as client:
        energy = client.get("/api/term-structure?sector=energy")
        assert energy.status_code == 200, energy.text
        body = energy.json()
        assert body["sector"] == "energy"
        assert body["selected_roots"]
        for root in body["selected_roots"]:
            assert master.get(root).sector.lower() == "energy"
        roots = client.get("/api/term-structure?roots=ES,NQ")
        assert roots.status_code == 200, roots.text
        assert roots.json()["selected_roots"] == ["ES", "NQ"]
        assert {curve["root"] for curve in roots.json()["curves"]} == {"ES", "NQ"}
        for bad in (
            "/api/term-structure?roots=ZZZ",
            "/api/term-structure?sector=widgets",
            "/api/term-structure?scope=everything",
        ):
            response = client.get(bad)
            assert response.status_code == 422, (bad, response.text)


def test_term_structure_cached_scope_uses_parquet(tmp_path: Path) -> None:
    settings = _base_settings(tmp_path)
    repository = ParquetRepository(settings)
    repository.write_bars(
        make_bars_frame(
            roots=("ES", "NQ"),
            periods=120,
            start=FIXED.replace(tzinfo=UTC) - __import__("datetime").timedelta(days=600),
            instrument_master=load_instruments(INSTRUMENTS_PATH),
        )
    )
    with _app(settings) as client:
        response = client.get("/api/term-structure?scope=cached")
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["scope"] == "cached"
        assert set(body["selected_roots"]) == {"ES", "NQ"}
        assert body["data_origin"] in {"cached_parquet", "mixed_cached_and_deterministic"}
        assert len(body["curves"]) == 2


def test_pipeline_scopes_cover_all_roots_and_filters(
    tmp_path: Path, master: InstrumentsConfig
) -> None:
    with _app(_base_settings(tmp_path)) as client:
        default = client.get("/api/pipeline/latest")
        assert default.status_code == 200, default.text
        assert default.json()["scope"] == "strategy"
        assert 0 < len(default.json()["instruments"]) < 45
        all_roots = client.get("/api/pipeline/latest?scope=all")
        assert all_roots.status_code == 200, all_roots.text
        body = all_roots.json()
        assert body["scope"] == "all"
        assert len(body["selected_roots"]) == 45
        assert len(body["instruments"]) == 45
        assert {row["instrument_id"] for row in body["instruments"]} == set(body["selected_roots"])
        assert body["data_origin"] == "deterministic_instrument_master"
        metals = client.get("/api/pipeline/latest?sector=metals")
        assert metals.status_code == 200, metals.text
        assert all(
            master.get(root).sector.lower() == "metals" for root in metals.json()["selected_roots"]
        )
        explicit = client.get("/api/pipeline/latest?roots=ES,CL")
        assert explicit.status_code == 200, explicit.text
        assert explicit.json()["selected_roots"] == ["ES", "CL"]
        assert len(explicit.json()["instruments"]) == 2
        for bad in (
            "/api/pipeline/latest?roots=ZZZ",
            "/api/pipeline/latest?sector=widgets",
            "/api/pipeline/latest?scope=everything",
        ):
            assert client.get(bad).status_code == 422, bad


def test_risk_state_scopes_cover_all_roots_and_filters(
    tmp_path: Path, master: InstrumentsConfig
) -> None:
    with _app(_base_settings(tmp_path)) as client:
        default = client.get("/api/risk/state")
        assert default.status_code == 200, default.text
        assert default.json()["scope"] == "strategy"
        assert 0 < len(default.json()["assessments"]) < 45
        all_roots = client.get("/api/risk/state?scope=all")
        assert all_roots.status_code == 200, all_roots.text
        body = all_roots.json()
        assert body["scope"] == "all"
        assert len(body["selected_roots"]) == 45
        assert len(body["assessments"]) == 45
        assert body["derived_roots"], "roots without a live/cached snapshot must be attributed"
        assert set(body["derived_roots"]) <= set(body["selected_roots"])
        assert body["assessment_source"] == "mixed"
        assert all(row["instrument_id"] in body["selected_roots"] for row in body["assessments"])
        for row in body["assessments"]:
            if row["instrument_id"] in body["derived_roots"]:
                assert row["source"] == "derived-neutral"
        energy = client.get("/api/risk/state?sector=energy")
        assert energy.status_code == 200, energy.text
        assert energy.json()["sector"] == "energy"
        assert energy.json()["selected_roots"]
        assert all(
            master.get(root).sector.lower() == "energy" for root in energy.json()["selected_roots"]
        )
        explicit = client.get("/api/risk/state?roots=ES,CL")
        assert explicit.status_code == 200, explicit.text
        assert explicit.json()["selected_roots"] == ["CL", "ES"]  # risk scopes are sorted
        assert len(explicit.json()["assessments"]) == 2
        for bad in (
            "/api/risk/state?roots=ZZZ",
            "/api/risk/state?sector=widgets",
            "/api/risk/state?scope=nope",
        ):
            assert client.get(bad).status_code == 422, bad


def test_risk_cached_scope_reports_cached_roots(tmp_path: Path) -> None:
    settings = _base_settings(tmp_path)
    ParquetRepository(settings).write_bars(
        make_bars_frame(
            roots=("ES", "NQ"),
            periods=120,
            start=FIXED.replace(tzinfo=UTC) - __import__("datetime").timedelta(days=600),
            instrument_master=load_instruments(INSTRUMENTS_PATH),
        )
    )
    with _app(settings) as client:
        response = client.get("/api/risk/state?scope=cached")
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["scope"] == "cached"
        assert set(body["selected_roots"]) == {"ES", "NQ"}
        assert len(body["assessments"]) == 2


# ---------------------------------------------------------------------------
# 3. IBKR port discovery + guarded connection switch
# ---------------------------------------------------------------------------


def _fake_probe_factory(reachable: set[int]):
    async def fake_probe(
        host: str, port: int, *, timeout: float = 0.25
    ) -> tuple[bool, float | None]:
        del host, timeout
        return (True, 11.5) if port in reachable else (False, None)

    return fake_probe


def test_ibkr_ports_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner_routes, "probe_ibkr_port", _fake_probe_factory({7497, 4002}))
    with _app(_base_settings(tmp_path)) as client:
        response = client.get("/api/runner/ibkr-ports")
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["host"] == "127.0.0.1"
        assert [row["port"] for row in body["ports"]] == [7497, 4002, 7496, 4001]
        by_port = {row["port"]: row for row in body["ports"]}
        assert by_port[7497]["mode"] == "paper" and by_port[7497]["gateway"] == "tws"
        assert by_port[4002]["mode"] == "paper" and by_port[4002]["gateway"] == "ib_gateway"
        assert by_port[7496]["mode"] == "live" and by_port[7496]["gateway"] == "tws"
        assert by_port[4001]["mode"] == "live" and by_port[4001]["gateway"] == "ib_gateway"
        assert by_port[7497]["reachable"] is True
        assert by_port[7497]["latency_ms"] == pytest.approx(11.5)
        assert by_port[7496]["reachable"] is False
        assert by_port[7496]["latency_ms"] is None
        assert body["active"] == {
            "port": 7497,
            "client_id": 42,
            "environment": "backtest",
            "mode": "paper",
            "gateway": "tws",
        }
        assert body["warnings"] == []


def test_ibkr_ports_warn_when_none_reachable_or_unknown_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner_routes, "probe_ibkr_port", _fake_probe_factory(set()))
    with _app(_base_settings(tmp_path)) as client:
        body = client.get("/api/runner/ibkr-ports").json()
        assert all(row["reachable"] is False for row in body["ports"])
        assert any("no reachable IBKR" in warning for warning in body["warnings"])
    with _app(_base_settings(tmp_path / "unknown", ibkr_port=7498)) as client:
        body = client.get("/api/runner/ibkr-ports").json()
        assert body["active"]["port"] == 7498
        assert body["active"]["mode"] is None
        assert any("not a documented IBKR" in warning for warning in body["warnings"])


def test_ibkr_paper_switch_is_guarded_honest_and_order_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner_routes, "probe_ibkr_port", _fake_probe_factory({4002}))
    with _app(_base_settings(tmp_path)) as client:
        response = client.post("/api/runner/ibkr-connection", json={"port": 4002, "client_id": 43})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["requested"] == {
            "port": 4002,
            "mode": "paper",
            "gateway": "ib_gateway",
            "client_id": 43,
        }
        assert body["applied"] is False
        assert body["restart_required"] is True
        assert body["status"] == "restart_required"
        assert "NOT switched" in body["message"]
        assert body["active"]["port"] == 7497, "the response must not fake an applied switch"
        assert any("restart the API" in warning for warning in body["warnings"])


def test_ibkr_live_switch_refused_in_paper_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner_routes, "probe_ibkr_port", _fake_probe_factory({7496}))
    with _app(_base_settings(tmp_path, environment="paper")) as client:
        response = client.post(
            "/api/runner/ibkr-connection",
            json={"port": 7496, "confirmation": "LIVE"},
        )
        assert response.status_code == 409, response.text
        assert "paper environment" in response.json()["detail"]
        assert response.json()["detail"]


def test_ibkr_live_switch_requires_exact_live_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner_routes, "probe_ibkr_port", _fake_probe_factory({7496}))
    with _app(_base_settings(tmp_path)) as client:
        missing = client.post("/api/runner/ibkr-connection", json={"port": 7496})
        assert missing.status_code == 400
        assert '"LIVE"' in missing.json()["detail"]
        wrong = client.post(
            "/api/runner/ibkr-connection", json={"port": 7496, "confirmation": "live"}
        )
        assert wrong.status_code == 400
        ok = client.post("/api/runner/ibkr-connection", json={"port": 7496, "confirmation": "LIVE"})
        assert ok.status_code == 200, ok.text
        body = ok.json()
        assert body["requested"]["mode"] == "live"
        assert body["requested"]["gateway"] == "tws"
        assert body["applied"] is False and body["restart_required"] is True
        invalid_port = client.post("/api/runner/ibkr-connection", json={"port": 1234})
        assert invalid_port.status_code == 422
        invalid_client = client.post(
            "/api/runner/ibkr-connection", json={"port": 4002, "client_id": 5000}
        )
        assert invalid_client.status_code == 422


def test_ibkr_switch_applies_only_when_adapter_reconnects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner_routes, "probe_ibkr_port", _fake_probe_factory({4002}))

    class _SwitchingExecution:
        ibkr_enabled = False
        connected = False
        reconnecting = False

        def __init__(self, result: object) -> None:
            self.result = result
            self.calls: list[tuple[int, int]] = []

        async def switch_connection(self, port: int, client_id: int) -> object:
            self.calls.append((port, client_id))
            return self.result

    settings = _base_settings(tmp_path)
    for result, expected_status in ((True, "applied"), (False, "restart_required")):
        execution = _SwitchingExecution(result)
        container = build_container(settings, clock=lambda: FIXED)
        patched = dataclasses.replace(container, execution=execution)
        with TestClient(create_app(settings, clock=lambda: FIXED, container=patched)) as client:
            response = client.post(
                "/api/runner/ibkr-connection", json={"port": 4002, "client_id": 41}
            )
            assert response.status_code == 200, response.text
            body = response.json()
            assert body["status"] == expected_status
            assert body["applied"] is (expected_status == "applied")
            assert body["requested"]["client_id"] == 41
            assert execution.calls == [(4002, 41)]


# ---------------------------------------------------------------------------
# 4. Composite timing serialization + catalog metadata
# ---------------------------------------------------------------------------


def _modular_strategy() -> StrategyConfig:
    return StrategyConfig(
        id="ui_modular_composite",
        description="Independent modular composite verification.",
        universe=StageConfig(
            name="composite",
            params={
                "components": "liquidity,regime",
                "mode": "intersection",
                "roots": "ES,NQ,CL,GC",
            },
        ),
        allocator=StageConfig(name="volatility_target", params={"target_annual_volatility": 0.12}),
        timing=(
            StageConfig(name="tsmom", params={"lookback_fast": 12, "lookback_slow": 90}),
            StageConfig(name="kama", params={"window": 14, "fast_period": 2, "slow_period": 30}),
            StageConfig(name="carry_filter", params={"min_carry_bps": -40.0}),
        ),
        risk=(StageConfig(name="quant_risk", params={"vol_target": 0.10}),),
        translator=TranslatorConfig(),
    )


def test_composite_timing_stack_serializes_and_builds_pipeline(master: InstrumentsConfig) -> None:
    strategy = _modular_strategy()
    timing = {stage.name: dict(stage.params) for stage in strategy.timing}
    assert len(strategy.timing) == 3
    assert timing["tsmom"]["lookback_fast"] == 12
    assert timing["kama"]["window"] == 14
    assert timing["carry_filter"]["min_carry_bps"] == -40.0
    reparsed = parse_strategy_config(strategy.model_dump())
    assert reparsed.timing == strategy.timing
    pipeline = build_pipeline(strategy, master)
    result = pipeline.run(make_market_state(roots=("ES", "NQ", "CL", "GC")))
    assert set(result.final_weights.weights) <= set(result.universe)
    with pytest.raises(ValidationError):
        StageConfig(name="tsmom", params={"lookback_fast": {"nested": 1}})  # type: ignore[dict-item]


def test_catalog_exposes_formulas_and_advanced_params(master: InstrumentsConfig) -> None:
    payload = catalog_dict(master)
    formulas = payload["formulas"]
    advanced = set(payload["advanced_parameter_keys"])
    assert set(formulas) == set(ALLOCATOR_FORMULAS)
    assert advanced == set(ADVANCED_PARAMETER_KEYS) and advanced
    parameters = {entry["key"]: entry for entry in payload["parameters"]}
    for key in advanced:
        assert parameters[key]["advanced"] is True
    for entry in payload["parameters"]:
        if entry["formula"] is not None:
            assert entry["formula"] in formulas.values()


def test_catalog_api_serves_additive_formula_and_advanced_metadata(tmp_path: Path) -> None:
    with _app(_base_settings(tmp_path)) as client:
        response = client.get("/api/strategies/schema/catalog")
        assert response.status_code == 200, response.text
        body = response.json()
        parameters = body["parameters"]
        assert parameters
        assert any(parameter.get("formula") for parameter in parameters)
        assert any(parameter["advanced"] is True for parameter in parameters)
        assert body["formulas"], "API must relay the catalog allocator formulas to the UI"
        assert body["advanced_parameter_keys"] == sorted(ADVANCED_PARAMETER_KEYS) or set(
            body["advanced_parameter_keys"]
        ) == set(ADVANCED_PARAMETER_KEYS)
        advanced_keys = set(body["advanced_parameter_keys"])
        advanced_rows = {row["key"]: row for row in parameters if row["advanced"] is True}
        assert advanced_rows
        assert set(advanced_rows) <= advanced_keys


def test_composite_strategy_can_be_saved_via_studio_api(tmp_path: Path) -> None:
    with _app(_base_settings(tmp_path)) as client:
        payload = _modular_strategy().model_dump(mode="json")
        response = client.post("/api/strategies", json=payload)
        assert response.status_code in {200, 201}, response.text
        created = response.json()
        assert created["id"] == "ui_modular_composite"
        assert [stage["name"] for stage in created["timing"]] == ["tsmom", "kama", "carry_filter"]
        listed = {row["id"] for row in client.get("/api/strategies").json()}
        assert "ui_modular_composite" in listed
        duplicate = client.post("/api/strategies", json=payload)
        assert duplicate.status_code == 409


# ---------------------------------------------------------------------------
# 5. Frontend contract sanity (static typed-contract assertions)
# ---------------------------------------------------------------------------


def _read_frontend(relative: str) -> str:
    return (REPO_ROOT / "frontend" / relative).read_text(encoding="utf-8")


def test_frontend_llm_pill_and_risk_contract() -> None:
    types = _read_frontend("lib/types.ts")
    for token in ("LLMStatusResponse", '"offline_stub"', '"online"', "api_key_configured"):
        assert token in types, f"frontend types missing {token!r}"
    shell = _read_frontend("components/shell/app-shell.tsx")
    assert "LLMStatusPill" in shell
    pill = _read_frontend("components/shell/llm-status-pill.tsx")
    assert "llm-status" in pill and "llm-ping" in pill
    risk = _read_frontend("components/risk/risk-view.tsx")
    for token in ("one-way brake", "Confidence", "Dispersion", "weight multiplier", "Regime"):
        assert token in risk, f"risk view missing {token!r}"
    api = _read_frontend("lib/api.ts")
    assert "/api/risk/llm-status" in api and "/api/risk/llm-ping" in api


def test_frontend_root_filters_and_curve_modes() -> None:
    filter_bar = _read_frontend("components/filters/root-scope-filter.tsx")
    for token in ("Active Strategy", "Cached Parquet", "All 45", "sector"):
        assert token in filter_bar, f"root filter missing {token!r}"
    types = _read_frontend("lib/types.ts")
    for token in ("RootScope", "selected_roots"):
        assert token in types
    term = _read_frontend("components/term-structure/term-structure-view.tsx")
    for token in (
        "Normalized Basis % vs. F1",
        "Annualized Roll Yield %",
        "Raw Contract Price ($)",
        "curve-spread-header",
        "curve-zero-reference",
    ):
        assert token in term or token in _read_frontend("components/charts/plots.tsx")
    plots = _read_frontend("components/charts/plots.tsx")
    assert "NORMALIZED_BASIS_MIN_HALF_SPAN = 1.5" in plots
    assert "RAW_PRICE_MIN_HALF_SPAN_PCT" in plots


def test_frontend_manual_builder_and_ibkr_operator_contract() -> None:
    builder = _read_frontend("components/studio/manual-builder.tsx")
    for token in ("Modular Composite Stack", "Save as Custom Preset", "localStorage", "advanced"):
        assert token in builder, f"manual builder missing {token!r}"
    api = _read_frontend("lib/api.ts")
    assert "/api/runner/ibkr-ports" in api and "/api/runner/ibkr-connection" in api
    manager = _read_frontend("components/shell/ibkr-connection-manager.tsx")
    for token in ("LIVE_PORTS", "7496", "4001", "getIbkrPorts", "switchIbkrConnection", '"LIVE"'):
        assert token in manager, f"IBKR manager missing {token!r}"


# ---------------------------------------------------------------------------
# 6. Test hygiene / OSS guard
# ---------------------------------------------------------------------------


def test_no_spec_or_phase_named_tests_remain() -> None:
    tests_root = REPO_ROOT / "tests"
    offenders = [
        path
        for path in tests_root.rglob("*.py")
        if "test_spec" in path.name or "test_phase" in path.name
    ]
    assert not offenders, f"spec/phase-named tests must not remain: {offenders}"
    assert (tests_root / "integration" / "test_open_source_release.py").is_file(), (
        "the OSS public-repo guard must remain in place"
    )
