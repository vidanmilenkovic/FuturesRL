"""Phase-10 API integration tests: Strategy Studio, Optuna studies, bridge pipeline.

Fully offline and deterministic: the app runs against a temporary config/artifact
root, the optimizer is either the real deterministic-backend implementation or an
injected fixture service, and the runner bridge is a synthetic snapshot written
through the production ``RunnerBridgeStore`` contract.
"""

from __future__ import annotations

import copy
import json
import shutil
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from configs.settings import Settings
from fastapi.testclient import TestClient
from src.api.main import create_app
from src.api.optimize_service import OptimizationService
from src.api.routes.optimize import router as optimize_router
from src.api.studio_service import StrategyStudioService

pytestmark = pytest.mark.integration

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
FIXED_TIMESTAMP = datetime(2026, 11, 2, 17, 0, tzinfo=UTC)
TELEMETRY_INTERVAL = 0.02

EXPECTED_PRESET_IDS = {
    "trend_following_tsmom_kama",
    "commodity_carry_term_structure",
    "panic_mean_reversion",
    "seasonal_commodity_cycle",
    "drl_neural_policy_llm_risk",
}


def _settings(tmp_path: Path) -> Settings:
    """Hermetic settings with a private config tree and artifact dirs."""
    configs_dir = tmp_path / "configs"
    strategies_dir = configs_dir / "strategies"
    strategies_dir.mkdir(parents=True, exist_ok=True)
    for source in sorted((REPO_ROOT / "configs" / "strategies").glob("*.yaml")):
        shutil.copyfile(source, strategies_dir / source.name)
    return Settings(
        environment="backtest",
        configs_dir=configs_dir,
        data_dir=tmp_path / "data",
        parquet_dir=tmp_path / "data" / "parquet",
        duckdb_path=tmp_path / "data" / "api.duckdb",
        report_dir=tmp_path / "artifacts",
        instruments_path=REPO_ROOT / "configs" / "instruments.yaml",
        telemetry_interval_seconds=TELEMETRY_INTERVAL,
        api_cors_origins=("http://localhost:3080",),
    )


def _app(
    settings: Settings,
    *,
    optimize_service: OptimizationService | None = None,
) -> TestClient:
    app = create_app(settings, clock=lambda: FIXED_TIMESTAMP)
    app.include_router(optimize_router, prefix="/api")
    if optimize_service is not None:
        app.state.studio_service = optimize_service.studio  # type: ignore[attr-defined]
        app.state.optimize_service = optimize_service
    return TestClient(app)


def _fixture_optimize_service(settings: Settings) -> OptimizationService:
    studio = StrategyStudioService(settings=settings, clock=lambda: FIXED_TIMESTAMP)
    return OptimizationService(
        settings=settings,
        studio=studio,
        clock=lambda: FIXED_TIMESTAMP,
        optimizer_provider=lambda: None,
    )


@pytest.fixture()
def client(tmp_path: Path) -> Iterator[TestClient]:
    with _app(_settings(tmp_path)) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# Strategy Studio catalog
# ---------------------------------------------------------------------------


def test_catalog_exposes_parameters_stages_and_five_presets(client: TestClient) -> None:
    response = client.get("/api/strategies/schema/catalog")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["version"] == 1
    assert body["stage_ids"] == ["universe", "allocator", "timing", "risk", "translator"]
    assert len(body["stages"]) == 5
    assert len(body["parameters"]) >= 20
    assert {preset["id"] for preset in body["presets"]} == EXPECTED_PRESET_IDS
    assert body["source"] in {"src.strategies.parameter_catalog", "fallback"}
    for parameter in body["parameters"][:10]:
        assert {"key", "stage", "type", "default", "optuna_tunable"} <= set(parameter)
        assert parameter["type"] in {"int", "float", "bool", "select", "multiselect"}
    for preset in body["presets"]:
        config = preset["config"]
        assert config["id"]
        assert "universe" in config and "allocator" in config
        assert preset["strategy"] == preset["config"]
        assert preset["name"]
        assert preset["hypothesis"]

    # Additive v1.5.1 metadata: curated Optuna defaults + tunable hygiene flags.
    parameters = body["parameters"]
    assert all(isinstance(parameter["optuna_default_selected"], bool) for parameter in parameters)
    parameters_by_key = {parameter["key"]: parameter for parameter in parameters}
    curated = sorted(
        parameter["key"] for parameter in parameters if parameter["optuna_default_selected"]
    )
    assert len(curated) == 6, curated
    assert set(curated) == {
        "allocator.volatility_target.target_annual_volatility",
        "risk.quant_risk.trailing_stop_pct",
        "timing.carry_filter.min_carry_bps",
        "timing.kama.window",
        "timing.tsmom.lookback_fast",
        "timing.tsmom.lookback_slow",
    }
    for key in curated:
        assert parameters_by_key[key]["optuna_tunable"] is True
    hygiene_keys = [
        "universe.liquidity.min_avg_dollar_volume",
        "universe.liquidity.min_history_days",
        "universe.liquidity.top_n",
        "risk.llm_risk.sample_count",
        "risk.llm_risk.cache_ttl_seconds",
    ]
    macro_keys = [
        key for key in parameters_by_key if key.startswith("risk.llm_risk.macro_lockout_")
    ]
    assert macro_keys, "expected macro-lockout hygiene parameters"
    for key in (*hygiene_keys, *macro_keys):
        assert key in parameters_by_key, key
        assert parameters_by_key[key]["optuna_tunable"] is False, key
        assert parameters_by_key[key]["optuna_default_selected"] is False, key

    # Additive Studio UX metadata: stage dropdown choices passed through unchanged.
    stage_choices = body["stage_choices"]
    assert stage_choices["universe"] == ["liquidity", "regime", "composite"]
    assert stage_choices["allocator"] == [
        "volatility_target",
        "risk_parity",
        "equal_weight",
        "mean_variance",
        "drl",
    ]
    assert set(stage_choices) >= {"universe", "allocator", "timing", "risk", "translator"}
    assert all(stage_choices[key] for key in stage_choices)
    stages_by_id = {stage["id"]: stage for stage in body["stages"]}
    assert stages_by_id["universe"]["choices"]
    first_choice = stages_by_id["universe"]["choices"][0]
    assert {"name", "label", "description", "parameter_keys"} <= set(first_choice)
    assert [choice["name"] for choice in stages_by_id["universe"]["choices"]] == stage_choices[
        "universe"
    ]

    # Additive Studio UX metadata: sector-grouped instrument universe explorer.
    universe = body["universe"]
    assert set(universe) >= {"sectors", "roots"}
    sector_ids = [sector["id"] for sector in universe["sectors"]]
    expected_groups = ["equities", "rates", "energy", "metals", "fx", "agri", "crypto"]
    assert sector_ids[0] == "all"
    assert sector_ids[1:8] == expected_groups
    all_sector = universe["sectors"][0]
    assert all_sector["count"] == len(universe["roots"]) == len(all_sector["roots"])
    assert len(universe["roots"]) >= 40
    groups = {sector["id"]: sector for sector in universe["sectors"][1:]}
    for group in expected_groups:
        assert groups[group]["count"] > 0
        assert groups[group]["count"] == len(groups[group]["roots"])
        roots = [row["root"] for row in groups[group]["roots"]]
        assert len(set(roots)) == len(roots)
        assert {row["sector_group"] for row in groups[group]["roots"]} == {group}
    for root in universe["roots"]:
        assert {
            "root",
            "name",
            "asset_class",
            "sector",
            "sector_group",
            "exchange",
            "multiplier",
            "tick_size",
            "micro_root",
            "standard_root",
            "currency",
        } <= set(root)
        assert root["sector_group"] in groups
    # The whole catalog JSON round-trips (typed models serialise cleanly).
    assert json.loads(json.dumps(body)) == body


def test_catalog_universe_agrees_with_instruments_endpoint(client: TestClient) -> None:
    catalog = client.get("/api/strategies/schema/catalog").json()
    instruments = client.get("/api/instruments").json()
    universe = catalog["universe"]
    catalog_roots = {row["root"] for row in universe["roots"]}
    instrument_roots = {row["root"] for row in instruments["roots"]}
    assert catalog_roots == instrument_roots
    assert len(universe["roots"]) == instruments["count"]
    catalog_groups = {
        sector["id"]: sector["count"] for sector in universe["sectors"] if sector["id"] != "all"
    }
    for sector in instruments["sectors"]:
        assert catalog_groups.get(sector["id"]) == sector["count"]
    by_root = {row["root"]: row for row in universe["roots"]}
    for row in instruments["roots"]:
        if row["micro_root"]:
            assert by_root[row["micro_root"]]["standard_root"] == row["root"]
    assert json.loads(json.dumps(universe)) == universe


def test_create_strategy_from_preset_persists_yaml(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with _app(settings) as client:
        catalog = client.get("/api/strategies/schema/catalog").json()
        preset = catalog["presets"][0]
        payload = copy.deepcopy(preset["strategy"])
        payload["id"] = "studio_trend_test"
        created = client.post("/api/strategies", json=payload)
        assert created.status_code == 200, created.text
        body = created.json()
        assert body["id"] == "studio_trend_test"
        assert body["universe"]["name"]
        target = settings.strategy_artifacts_dir / "studio_trend_test.yaml"
        assert target.is_file()
        assert body["updated_at"]
        imported = json.loads(client.get("/api/strategies/studio_trend_test").text)
        assert imported["id"] == "studio_trend_test"
        assert "studio_trend_test" in {row["id"] for row in client.get("/api/strategies").json()}
        duplicate = client.post("/api/strategies", json=payload)
        assert duplicate.status_code == 409

        # Save & Run in Backtest Lab: the new id is registered in the in-memory
        # catalog immediately (no restart / no committed-YAML mutation).
        run = client.post(
            "/api/backtests",
            json={
                "strategy_id": "studio_trend_test",
                "start": "2025-01-02",
                "end": "2025-06-30",
                "initial_equity": 1_000_000.0,
                "rebalance_frequency": "W-FRI",
                "walk_forward": False,
                "random_seed": 7,
            },
        )
        assert run.status_code == 200, run.text
        assert run.json()["strategy_id"] == "studio_trend_test"


def test_create_strategy_accepts_yaml_body_and_rejects_unsafe_input(
    client: TestClient,
) -> None:
    yaml_body = """
id: studio_yaml_created
description: created from a YAML document
universe:
  name: liquidity
  params: {top_n: 5}
allocator:
  name: volatility_target
  params: {target_annual_volatility: 0.12}
timing:
  - name: tsmom
    params: {lookback_fast: 21, lookback_slow: 126}
risk:
  - name: quant_risk
    params: {vol_target: 0.12}
translator:
  rounding_mode: nearest
  max_margin_utilization: 0.6
"""
    created = client.post(
        "/api/strategies", content=yaml_body, headers={"content-type": "application/x-yaml"}
    )
    assert created.status_code == 200, created.text
    assert created.json()["id"] == "studio_yaml_created"

    traversal = client.post(
        "/api/strategies",
        json={
            "id": "../escape",
            "universe": {"name": "liquidity"},
            "allocator": {"name": "equal_weight"},
        },
    )
    assert traversal.status_code == 400
    assert "invalid strategy id" in traversal.json()["detail"]

    bad_slug = client.post(
        "/api/strategies",
        json={
            "id": "Bad-Id!",
            "universe": {"name": "liquidity"},
            "allocator": {"name": "equal_weight"},
        },
    )
    assert bad_slug.status_code == 400

    invalid_config = client.post(
        "/api/strategies",
        json={"id": "studio_missing_universe", "allocator": {"name": "equal_weight"}},
    )
    assert invalid_config.status_code == 422

    invalid_yaml = client.post(
        "/api/strategies",
        content="universe: [",
        headers={"content-type": "application/x-yaml"},
    )
    assert invalid_yaml.status_code == 422
    assert "invalid YAML" in invalid_yaml.json()["detail"]


def test_update_builtin_strategy_is_overlay_and_validated(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    committed = settings.configs_dir / "strategies" / "tsmom_vol_target.yaml"
    committed_before = committed.read_text(encoding="utf-8")
    with _app(settings) as client:
        updated = client.put(
            "/api/strategies/tsmom_vol_target",
            json={"description": "edited by the phase-10 API test"},
        )
        assert updated.status_code == 200, updated.text
        assert updated.json()["description"] == "edited by the phase-10 API test"
        detail = client.get("/api/strategies/tsmom_vol_target").json()
        assert detail["description"] == "edited by the phase-10 API test"
        # The committed YAML stays immutable in the backtest environment; an
        # overlay under artifacts/strategies carries the edit instead.
        assert committed.read_text(encoding="utf-8") == committed_before
        assert (settings.strategy_artifacts_dir / "tsmom_vol_target.yaml").is_file()

        invalid = client.put("/api/strategies/tsmom_vol_target", json={"universe": {"params": {}}})
        assert invalid.status_code == 422
        unknown = client.put("/api/strategies/does_not_exist", json={"description": "nope"})
        assert unknown.status_code == 404
        mismatch = client.put("/api/strategies/tsmom_vol_target", json={"id": "other_strategy"})
        assert mismatch.status_code == 422
        unsafe = client.put("/api/strategies/..", json={"description": "nope"})
        assert unsafe.status_code in {400, 404}


# ---------------------------------------------------------------------------
# Optuna optimization endpoints (fixture provenance)
# ---------------------------------------------------------------------------


def _optimize_payload(trials: int = 5, splits: int = 3) -> dict[str, Any]:
    return {
        "base_strategy_id": "tsmom_vol_target",
        "objective_metric": "sharpe",
        "n_trials": trials,
        "walk_forward": True,
        "wf_splits": splits,
        "purge_bars": 3,
        "embargo_bars": 2,
        "seed": 11,
        "search_space": [
            {
                "param_path": "allocator.params.target_annual_volatility",
                "type": "float",
                "low": 0.08,
                "high": 0.16,
                "step": 0.02,
            },
            {
                "param_path": "timing.1.params.fast_period",
                "type": "int",
                "low": 2,
                "high": 6,
                "step": 1,
            },
        ],
    }


def test_fixture_optimization_study_persists_and_reloads(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    service = _fixture_optimize_service(settings)
    with _app(settings, optimize_service=service) as client:
        response = client.post("/api/optimize", json=_optimize_payload())
        assert response.status_code == 200, response.text
        study = response.json()
        assert study["provenance"] == "fixture"
        assert study["study_id"].startswith("opt-")
        assert study["n_trials"] == 5
        assert study["trials"] and len(study["trials"]) == 5
        assert study["best_params"]
        assert study["best_strategy_config"]["id"] == "tsmom_vol_target"
        assert study["message"]
        artifact = settings.optimization_artifacts_dir / f"{study['study_id']}.json"
        assert artifact.is_file()
        study_id = study["study_id"]

        listed = client.get("/api/optimize").json()
        assert study_id in {row["study_id"] for row in listed}
        detail = client.get(f"/api/optimize/{study_id}")
        assert detail.status_code == 200
        assert len(detail.json()["trials"]) == 5
        assert client.get("/api/optimize/unknown-study").status_code == 404

    # A brand-new app instance reloads the persisted study from disk.
    with _app(settings) as second:
        listed = second.get("/api/optimize").json()
        assert study_id in {row["study_id"] for row in listed}
        reloaded = second.get(f"/api/optimize/{study_id}")
        assert reloaded.status_code == 200
        assert reloaded.json()["provenance"] == "fixture"


def test_optimization_validation_errors(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with _app(settings, optimize_service=_fixture_optimize_service(settings)) as client:
        unknown = client.post(
            "/api/optimize", json={**_optimize_payload(), "base_strategy_id": "nope"}
        )
        assert unknown.status_code == 404
        assert (
            client.post("/api/optimize", json={**_optimize_payload(), "n_trials": 1}).status_code
            == 422
        )
        assert (
            client.post(
                "/api/optimize", json={**_optimize_payload(), "objective_metric": "bogus"}
            ).status_code
            == 422
        )
        default_space = client.post(
            "/api/optimize",
            json={"base_strategy_id": "tsmom_vol_target", "n_trials": 3, "wf_splits": 2},
        )
        assert default_space.status_code == 200, default_space.text
        assert len(default_space.json()["trials"]) == 3


def test_apply_optimization_creates_tuned_strategy(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with _app(settings, optimize_service=_fixture_optimize_service(settings)) as client:
        study = client.post("/api/optimize", json=_optimize_payload(trials=4)).json()
        study_id = study["study_id"]
        applied = client.post(
            f"/api/optimize/{study_id}/apply",
            json={"new_strategy_id": "studio_optimized"},
        )
        assert applied.status_code == 200, applied.text
        body = applied.json()
        assert body["strategy"]["id"] == "studio_optimized"
        assert body["best_params"] == study["best_params"]
        target = settings.strategy_artifacts_dir / "studio_optimized.yaml"
        assert target.is_file()
        assert client.get("/api/strategies/studio_optimized").status_code == 200

        duplicate = client.post(
            f"/api/optimize/{study_id}/apply", json={"new_strategy_id": "studio_optimized"}
        )
        assert duplicate.status_code == 409
        assert client.post("/api/optimize/unknown-study/apply", json={}).status_code == 404
        traversal = client.post(
            f"/api/optimize/{study_id}/apply", json={"new_strategy_id": "../escape"}
        )
        assert traversal.status_code == 400


def test_optimization_accepts_catalog_parameter_keys(tmp_path: Path) -> None:
    """The Studio sends catalog keys; the API resolves them to concrete paths."""
    settings = _settings(tmp_path)
    payload = _optimize_payload(trials=3, splits=2)
    payload["search_space"] = [
        {
            "param_path": "timing.tsmom.lookback_fast",
            "type": "int",
            "low": 15,
            "high": 30,
            "step": 5,
        }
    ]
    with _app(settings, optimize_service=_fixture_optimize_service(settings)) as client:
        response = client.post("/api/optimize", json=payload)
        assert response.status_code == 200, response.text
        study = response.json()
        resolved = list(study["best_params"])
        assert resolved, study
        assert all(path.startswith("timing.") and "params." in path for path in resolved)
        applied = client.post(
            f"/api/optimize/{study['study_id']}/apply",
            json={"new_strategy_id": "studio_catalog_key_opt"},
        )
        assert applied.status_code == 200, applied.text


def test_real_optimizer_study_and_verification_apply(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with _app(settings) as client:
        response = client.post(
            "/api/optimize",
            json=_optimize_payload(trials=2, splits=2),
        )
        assert response.status_code == 200, response.text
        study = response.json()
        assert study["provenance"] == "engine", study.get("message")
        assert study["trials"] and len(study["trials"]) == 2
        assert study["best_params"]
        assert study["fold_count"] == 2
        assert study["artifact_path"]
        # The optimizer's canonical artifact must stay reloadable untouched: the API
        # never rewrites it (provenance is inferred when reading it back).
        artifact = Path(study["artifact_path"])
        assert artifact.is_file()
        raw = json.loads(artifact.read_text(encoding="utf-8"))
        assert {"request", "seed", "study_id"} <= set(raw)
        assert "provenance" not in raw or raw["provenance"] == "engine"
        optimizer_module = pytest.importorskip("src.backtest.optimizer")
        canonical = optimizer_module.load_optimization(
            study["study_id"], output_dir=settings.optimization_artifacts_dir
        )
        assert canonical.study_id == study["study_id"]
        assert canonical.best_params == study["best_params"]
        applied = client.post(
            f"/api/optimize/{study['study_id']}/apply",
            json={
                "new_strategy_id": "studio_opt_verified",
                "verification_backtest": True,
                "start": "2025-01-02",
                "end": "2025-06-30",
                "initial_equity": 1_000_000.0,
            },
        )
        assert applied.status_code == 200, applied.text
        body = applied.json()
        assert body["strategy"]["id"] == "studio_opt_verified"
        verification = body["verification_backtest"]
        assert verification is not None
        assert verification["status"] == "completed", verification
        assert verification["run"]["id"]
        assert verification["run"]["strategy_id"] == "studio_opt_verified"

        # The tuned id is now runnable directly too (Save & Run path).
        rerun = client.post(
            "/api/backtests",
            json={
                "strategy_id": "studio_opt_verified",
                "start": "2025-01-02",
                "end": "2025-06-30",
                "initial_equity": 1_000_000.0,
                "rebalance_frequency": "W-FRI",
                "walk_forward": False,
                "random_seed": 7,
            },
        )
        assert rerun.status_code == 200, rerun.text
        assert rerun.json()["strategy_id"] == "studio_opt_verified"

    # A restarted API process reloads the optimizer-canonical artifact and infers
    # its real-engine provenance (no API-side rewrite was needed).
    with _app(settings) as restarted:
        detail = restarted.get(f"/api/optimize/{study['study_id']}")
        assert detail.status_code == 200, detail.text
        reloaded = detail.json()
        assert reloaded["provenance"] == "engine"
        assert len(reloaded["trials"]) == 2
        assert reloaded["best_params"] == study["best_params"]


def test_mixed_artifact_directory_keeps_provenance_separate(tmp_path: Path) -> None:
    """Canonical engine artifacts and offline fixture studies coexist in one dir.

    The API branches on the payload shape (canonical artifacts carry ``request``/
    ``seed``; fixture studies carry explicit ``provenance``) and never routes a
    fixture payload through the engine's strict ``load_optimization`` loader.
    """
    settings = _settings(tmp_path)
    with _app(settings, optimize_service=_fixture_optimize_service(settings)) as fixture_client:
        fixture_study = fixture_client.post(
            "/api/optimize", json=_optimize_payload(trials=3)
        ).json()
        assert fixture_study["provenance"] == "fixture"
    with _app(settings) as engine_client:
        engine_study = engine_client.post(
            "/api/optimize", json=_optimize_payload(trials=2, splits=2)
        ).json()
        assert engine_study["provenance"] == "engine", engine_study.get("message")
    assert {path.stem for path in settings.optimization_artifacts_dir.glob("*.json")} >= {
        fixture_study["study_id"],
        engine_study["study_id"],
    }

    with _app(settings) as restarted:
        rows = {row["study_id"]: row for row in restarted.get("/api/optimize").json()}
        assert rows[fixture_study["study_id"]]["provenance"] == "fixture"
        assert rows[engine_study["study_id"]]["provenance"] == "engine"
        fixture_detail = restarted.get(f"/api/optimize/{fixture_study['study_id']}")
        assert fixture_detail.status_code == 200
        assert len(fixture_detail.json()["trials"]) == 3

    optimizer_module = pytest.importorskip("src.backtest.optimizer")
    canonical = optimizer_module.load_optimization(
        engine_study["study_id"], output_dir=settings.optimization_artifacts_dir
    )
    assert canonical.study_id == engine_study["study_id"]
    assert canonical.best_params == engine_study["best_params"]
    # The engine loader is intentionally strict: fixture payloads are API-only and
    # must never be fed to it (the API never does; this documents the boundary).
    with pytest.raises(ValueError):
        optimizer_module.load_optimization(
            fixture_study["study_id"], output_dir=settings.optimization_artifacts_dir
        )


# ---------------------------------------------------------------------------
# Runner-bridge pipeline parity (Part A1)
# ---------------------------------------------------------------------------


def _bridge_row_from_engine(row: dict[str, Any]) -> dict[str, Any]:
    driver = row.get("driver") or row["reason"]
    return {
        "instrument_id": row["instrument_id"],
        "root": row["instrument_id"],
        "asset_class": row["asset_class"],
        "selected": row["selected"],
        "weight_base": row["weight_base"],
        "weight_timing": row["weight_timing"],
        "weight_risk": row["weight_risk"],
        "weight_final": row["weight_final"],
        "weight_actual": row["weight_actual"],
        "delta_timing": row["delta_timing"],
        "delta_risk": row["delta_risk"],
        "target_contracts": row["target_contracts"],
        "notional": row["notional"],
        "driver": driver,
        "regime": row.get("regime"),
        "reason": driver,
    }


def _matrix_fields() -> tuple[str, ...]:
    return (
        "weight_base",
        "weight_timing",
        "weight_risk",
        "weight_final",
        "weight_actual",
        "delta_timing",
        "delta_risk",
        "target_contracts",
        "notional",
    )


def _assert_matrix_parity(mapped: dict[str, Any], engine_rows: list[dict[str, Any]]) -> None:
    assert mapped["data_source"] == "runner-bridge"
    assert [row["instrument_id"] for row in mapped["instruments"]] == sorted(
        row["instrument_id"] for row in engine_rows
    )
    by_id = {row["instrument_id"]: row for row in mapped["instruments"]}
    for expected in engine_rows:
        got = by_id[expected["instrument_id"]]
        assert got["selected"] is expected["selected"]
        assert got["asset_class"] == expected["asset_class"]
        for field in _matrix_fields():
            assert got[field] == pytest.approx(expected[field], abs=5e-8), field
        assert got["driver"] == (expected.get("driver") or expected["reason"])
        assert got["reason"] == got["driver"]
        assert got["regime"] == expected.get("regime")


def test_bridge_pipeline_matches_engine_weight_matrix(tmp_path: Path) -> None:
    bridge_module = pytest.importorskip("src.execution.runner_bridge")
    from src.api.bridge_pipeline import BridgePipelineService, pipeline_from_bridge_payload

    settings = _settings(tmp_path)
    with _app(settings) as first:
        engine = first.get("/api/pipeline/latest").json()
    engine_rows = engine["instruments"]
    assert engine_rows
    assert {"driver", "regime"} <= set(engine_rows[0])

    payload: dict[str, object] = dict(engine)  # flat scalar summary keys reuse
    payload["instruments"] = [_bridge_row_from_engine(row) for row in engine_rows]
    payload["micro_routing"] = [
        {
            "family": "equity",
            "standard_root": "ES",
            "standard_quantity": 1,
            "micro_root": "MES",
            "micro_quantity": 4,
        }
    ]
    payload["tracking_error_pct"] = 1.25
    payload["tracking_error_source"] = "translator"

    mapped = pipeline_from_bridge_payload(
        payload,
        master=settings.instruments,
        timestamp=FIXED_TIMESTAMP,
        sequence=9,
    )
    assert mapped is not None
    assert mapped.tracking_error_pct == pytest.approx(1.25)
    assert mapped.tracking_error_source == "translator"
    assert len(mapped.micro_routing) == 1
    assert mapped.micro_routing[0].micro_root == "MES"
    bridge_rows = [
        bridge_module.BridgePipelineRow(
            **{key: value for key, value in row.items() if key != "reason"}
        )
        for row in payload["instruments"]
    ]
    store = bridge_module.RunnerBridgeStore(settings)
    store.save(
        bridge_module.RunnerBridgeSnapshot(
            schema_version=bridge_module.BRIDGE_SCHEMA_VERSION,
            runner_id="runner-phase10-new",
            mode="offline",
            sequence=10,
            updated_at=FIXED_TIMESTAMP - timedelta(seconds=1),
            stale_after_seconds=3600.0,
            pipeline={
                "universe": ",".join(engine["universe"]),
                "tracking_error_pct": 1.25,
            },
            instruments=tuple(bridge_rows),
            micro_routing=(
                bridge_module.BridgeMicroRouting(
                    family="equity",
                    standard_root="ES",
                    standard_quantity=1,
                    micro_root="MES",
                    micro_quantity=4,
                ),
            ),
            tracking_error_pct=1.25,
            tracking_error_source="translator",
        )
    )

    # Endpoint consumes the fresh bridge matrix and renders it identically to the
    # engine waterfall (Part A1 parity).
    with _app(settings) as client:
        bridged = client.get("/api/pipeline/latest").json()
        _assert_matrix_parity(bridged, engine_rows)
        assert bridged["universe"] == engine["universe"]
        assert bridged["tracking_error_pct"] == pytest.approx(1.25)
        assert bridged["tracking_error_source"] == "translator"
        assert bridged["micro_routing"] == [
            {
                "family": "equity",
                "standard_root": "ES",
                "standard_quantity": 1,
                "micro_root": "MES",
                "micro_quantity": 4,
            }
        ]

    # Honest fallback: a Phase-9 snapshot without per-instrument rows never yields
    # fabricated weight rows (compact bridge summary has an empty matrix).
    store.save(
        bridge_module.RunnerBridgeSnapshot(
            schema_version=bridge_module.BRIDGE_SCHEMA_VERSION,
            runner_id="runner-phase10-old",
            mode="offline",
            sequence=3,
            updated_at=FIXED_TIMESTAMP - timedelta(seconds=1),
            stale_after_seconds=3600.0,
            pipeline={"universe": "ES,NQ"},
        )
    )
    service = BridgePipelineService(
        settings=settings, clock=lambda: FIXED_TIMESTAMP, instrument_master=settings.instruments
    )
    assert service.pipeline() is None
    assert (
        pipeline_from_bridge_payload(
            {"universe": "ES,NQ"}, master=settings.instruments, timestamp=FIXED_TIMESTAMP
        )
        is None
    )
    with _app(settings) as client:
        fallback = client.get("/api/pipeline/latest").json()
        assert fallback["data_source"] == "runner-bridge"
        assert fallback["instruments"] == []
