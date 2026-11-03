"""QoL artifact-management API tests: deletes, strategy clone/rename/YAML, exports.

Hermetic: every test uses a temporary config/artifact root copied from the committed
strategies, so committed YAMLs are never mutated.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml
from configs.settings import Settings
from fastapi.testclient import TestClient
from src.api.main import create_app
from src.api.optimize_service import OptimizationService
from src.api.routes.optimize import router as optimize_router
from src.api.studio_service import StrategyStudioService

pytestmark = pytest.mark.integration

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
FIXED_TIMESTAMP = datetime(2026, 11, 3, 17, 0, tzinfo=UTC)
TELEMETRY_INTERVAL = 0.02


def _settings(tmp_path: Path) -> Settings:
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


def _development_settings(tmp_path: Path) -> Settings:
    """Hermetic development-mode settings (new strategies persist to configs)."""
    return _settings(tmp_path).model_copy(update={"environment": "development"})


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


def _strategy_payload(strategy_id: str) -> dict[str, Any]:
    payload = yaml.safe_load(
        (REPO_ROOT / "configs" / "strategies" / "tsmom_vol_target.yaml").read_text(encoding="utf-8")
    )
    payload = json.loads(json.dumps(payload, default=str))
    payload["id"] = strategy_id
    payload["description"] = f"QoL test strategy {strategy_id}"
    return payload


# ---------------------------------------------------------------------------
# Strategy clone / rename / YAML export / delete
# ---------------------------------------------------------------------------


def test_strategy_clone_rename_export_and_delete(client: TestClient, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with _app(settings) as local:
        created = local.post("/api/strategies", json=_strategy_payload("qol_custom_base"))
        assert created.status_code == 200, created.text
        assert created.json()["builtin"] is False
        assert created.json()["path"].endswith("qol_custom_base.yaml")
        builtin_view = local.get("/api/strategies/tsmom_vol_target").json()
        assert builtin_view["builtin"] is True
        assert builtin_view["path"].endswith("tsmom_vol_target.yaml")
        overlay = settings.strategy_artifacts_dir
        assert (overlay / "qol_custom_base.yaml").is_file()

        cloned = local.post(
            "/api/strategies/qol_custom_base/clone",
            json={"new_strategy_id": "qol_custom_clone"},
        )
        assert cloned.status_code == 200, cloned.text
        assert cloned.json()["id"] == "qol_custom_clone"
        assert (overlay / "qol_custom_clone.yaml").is_file()
        assert local.get("/api/strategies/qol_custom_clone").status_code == 200
        duplicate_clone = local.post(
            "/api/strategies/qol_custom_base/clone",
            json={"new_strategy_id": "qol_custom_clone"},
        )
        assert duplicate_clone.status_code == 409
        assert (
            local.post(
                "/api/strategies/does_not_exist/clone",
                json={"new_strategy_id": "qol_x"},
            ).status_code
            == 404
        )

        downloaded = local.get("/api/strategies/qol_custom_clone/yaml")
        assert downloaded.status_code == 200
        assert downloaded.headers["content-type"].startswith("text/yaml")
        assert 'filename="qol_custom_clone.yaml"' in downloaded.headers["content-disposition"]
        assert "id: qol_custom_clone" in downloaded.text

        renamed = local.post(
            "/api/strategies/qol_custom_clone/rename",
            json={"new_strategy_id": "qol_custom_renamed"},
        )
        assert renamed.status_code == 200, renamed.text
        assert renamed.json()["id"] == "qol_custom_renamed"
        assert (overlay / "qol_custom_renamed.yaml").is_file()
        assert not (overlay / "qol_custom_clone.yaml").exists()
        assert local.get("/api/strategies/qol_custom_clone").status_code == 404
        listed = {row["id"] for row in local.get("/api/strategies").json()}
        assert "qol_custom_renamed" in listed and "qol_custom_clone" not in listed
        # Rename refreshes the runtime catalog too (available to the Backtest Lab).
        available = set(local.get("/api/backtests").json()["available_strategies"])
        assert "qol_custom_renamed" in available and "qol_custom_clone" not in available

        # Built-ins are protected from rename/delete; clone is allowed.
        assert (
            local.post(
                "/api/strategies/tsmom_vol_target/rename",
                json={"new_strategy_id": "renamed_builtin"},
            ).status_code
            == 409
        )
        assert local.delete("/api/strategies/tsmom_vol_target").status_code == 409
        assert (
            local.post(
                "/api/strategies/tsmom_vol_target/clone",
                json={"new_strategy_id": "qol_builtin_clone"},
            ).status_code
            == 200
        )

        deleted = local.delete("/api/strategies/qol_custom_renamed")
        assert deleted.status_code == 200, deleted.text
        receipt = deleted.json()
        assert receipt["id"] == "qol_custom_renamed"
        assert receipt["deleted"] is True
        assert receipt["artifact_deleted"] is True
        assert receipt["path"].endswith("qol_custom_renamed.yaml")
        assert not (overlay / "qol_custom_renamed.yaml").exists()
        assert local.get("/api/strategies/qol_custom_renamed").status_code == 404
        assert "qol_custom_renamed" not in {
            row["id"] for row in local.get("/api/strategies").json()
        }
        assert "qol_custom_renamed" not in set(
            local.get("/api/backtests").json()["available_strategies"]
        )
        assert local.delete("/api/strategies/qol_custom_renamed").status_code == 404


# ---------------------------------------------------------------------------
# Optimization study deletion
# ---------------------------------------------------------------------------


def test_custom_strategy_registry_survives_restart(tmp_path: Path) -> None:
    """Custom strategies written to configs/ remain deletable after an API restart."""
    settings = _development_settings(tmp_path)
    registry_path = settings.strategy_artifacts_dir / ".custom_ids.json"
    with _app(settings) as local:
        created = local.post("/api/strategies", json=_strategy_payload("qol_dev_custom"))
        assert created.status_code == 200, created.text
        cloned = local.post(
            "/api/strategies/qol_dev_custom/clone",
            json={"new_strategy_id": "qol_dev_clone"},
        )
        assert cloned.status_code == 200, cloned.text
        renamed = local.post(
            "/api/strategies/qol_dev_clone/rename",
            json={"new_strategy_id": "qol_dev_renamed"},
        )
        assert renamed.status_code == 200, renamed.text
        configs_dir = settings.configs_dir / "strategies"
        assert (configs_dir / "qol_dev_custom.yaml").is_file()
        assert (configs_dir / "qol_dev_renamed.yaml").is_file()
        assert not (configs_dir / "qol_dev_clone.yaml").exists()
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        registered = set(registry["custom_ids"])
        assert {"qol_dev_custom", "qol_dev_renamed"} <= registered
        assert "qol_dev_clone" not in registered

    # Simulated API restart: a fresh app/service discovers the same configs tree but
    # loads the durable custom registry, so custom strategies stay deletable while
    # the committed built-in is still protected.
    with _app(settings) as restarted:
        assert restarted.delete("/api/strategies/tsmom_vol_target").status_code == 409
        deleted_renamed = restarted.delete("/api/strategies/qol_dev_renamed")
        assert deleted_renamed.status_code == 200, deleted_renamed.text
        assert deleted_renamed.json()["artifact_deleted"] is True
        deleted_custom = restarted.delete("/api/strategies/qol_dev_custom")
        assert deleted_custom.status_code == 200, deleted_custom.text
        assert not (settings.configs_dir / "strategies" / "qol_dev_custom.yaml").exists()

    assert not {
        "qol_dev_custom",
        "qol_dev_renamed",
    } & set(json.loads(registry_path.read_text(encoding="utf-8"))["custom_ids"])


def test_optimization_study_delete(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    service = _fixture_optimize_service(settings)
    with _app(settings, optimize_service=service) as local:
        response = local.post(
            "/api/optimize",
            json={"base_strategy_id": "tsmom_vol_target", "n_trials": 2, "wf_splits": 2},
        )
        assert response.status_code == 200, response.text
        study_id = response.json()["study_id"]
        artifact = settings.optimization_artifacts_dir / f"{study_id}.json"
        assert artifact.is_file()
        assert local.get(f"/api/optimize/{study_id}").status_code == 200

        deleted = local.delete(f"/api/optimize/{study_id}")
        assert deleted.status_code == 200, deleted.text
        receipt = deleted.json()
        assert receipt["id"] == study_id and receipt["deleted"] is True
        assert receipt["artifact_deleted"] is True
        assert not artifact.exists()
        assert local.get(f"/api/optimize/{study_id}").status_code == 404
        listed = {row["study_id"] for row in local.get("/api/optimize").json()}
        assert study_id not in listed
        assert local.delete(f"/api/optimize/{study_id}").status_code == 404


# ---------------------------------------------------------------------------
# Backtest snapshot / trades / exports / deletion
# ---------------------------------------------------------------------------


def _run_short_backtest(client: TestClient) -> dict[str, Any]:
    response = client.post(
        "/api/backtests",
        json={
            "strategy_id": "tsmom_vol_target",
            "start": "2025-01-02",
            "end": "2025-03-31",
            "initial_equity": 1_000_000.0,
            "rebalance_frequency": "W-FRI",
            "walk_forward": False,
            "random_seed": 7,
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_backtest_snapshot_trades_exports_and_delete(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with _app(settings) as local:
        summary = _run_short_backtest(local)
        run_id = summary["id"]
        assert summary["strategy_config"] is not None
        assert summary["strategy_config"]["id"] == "tsmom_vol_target"

        tearsheet = local.get(f"/api/backtests/{run_id}/tearsheet")
        assert tearsheet.status_code == 200
        trades = tearsheet.json()["trades"]
        assert isinstance(trades, list)
        assert trades, "expected the real engine run to execute trades"
        assert {
            "timestamp",
            "order_id",
            "instrument_id",
            "side",
            "quantity",
            "price",
            "notional",
            "commissions",
            "slippage",
            "kind",
        } <= set(trades[0])

        exported = local.get(f"/api/backtests/{run_id}/export.json")
        assert exported.status_code == 200
        assert exported.headers["content-type"].startswith("application/json")
        assert 'filename="' + run_id + '-export.json"' in exported.headers["content-disposition"]
        payload = exported.json()
        assert payload["summary"]["id"] == run_id
        assert payload["summary"]["strategy_config"]["id"] == "tsmom_vol_target"
        assert payload["tearsheet"]["trades"] == trades
        assert payload["strategy_config"] == summary["strategy_config"]

        csv_response = local.get(f"/api/backtests/{run_id}/trades.csv")
        assert csv_response.status_code == 200
        assert csv_response.headers["content-type"].startswith("text/csv")
        assert 'filename="' + run_id + '-trades.csv"' in csv_response.headers["content-disposition"]
        lines = csv_response.text.strip().splitlines()
        assert lines[0].startswith("timestamp,order_id,instrument_id,side,quantity,price")
        assert len(lines) == len(trades) + 1

        # Snapshot survives a restart (persisted artifact is loaded with strategy_config).
        with _app(settings) as restarted:
            reloaded = restarted.get(f"/api/backtests/{run_id}").json()
            assert reloaded["strategy_config"]["id"] == "tsmom_vol_target"
            assert restarted.get(f"/api/backtests/{run_id}/tearsheet").json()["trades"] == trades

        artifact = settings.backtest_artifacts_dir / f"{run_id}.json"
        assert artifact.is_file()
        deleted = local.delete(f"/api/backtests/{run_id}")
        assert deleted.status_code == 200, deleted.text
        receipt = deleted.json()
        assert receipt["id"] == run_id and receipt["deleted"] is True
        assert receipt["artifact_deleted"] is True
        assert not artifact.exists()
        assert local.get(f"/api/backtests/{run_id}").status_code == 404
        assert local.get(f"/api/backtests/{run_id}/tearsheet").status_code == 404
        assert local.get(f"/api/backtests/{run_id}/export.json").status_code == 404
        assert local.get(f"/api/backtests/{run_id}/trades.csv").status_code == 404
        assert run_id not in {row["id"] for row in local.get("/api/backtests").json()["runs"]}
        assert local.delete(f"/api/backtests/{run_id}").status_code == 404


def test_walk_forward_run_exports_trades(tmp_path: Path) -> None:
    """Walk-forward tear sheets expose the combined OOS trade tape + config snapshot."""
    settings = _settings(tmp_path)
    with _app(settings) as local:
        response = local.post(
            "/api/backtests",
            json={
                "strategy_id": "tsmom_vol_target",
                "start": "2025-01-02",
                "end": "2025-06-30",
                "initial_equity": 1_000_000.0,
                "rebalance_frequency": "W-FRI",
                "walk_forward": True,
                "wf_splits": 2,
                "random_seed": 7,
            },
        )
        assert response.status_code == 200, response.text
        summary = response.json()
        assert summary["kind"] == "walk_forward"
        assert summary["strategy_config"]["id"] == "tsmom_vol_target"
        tearsheet = local.get(f"/api/backtests/{summary['id']}/tearsheet").json()
        assert tearsheet["folds"]
        trades = tearsheet["trades"]
        assert trades, "expected combined out-of-sample walk-forward trades"
        csv_response = local.get(f"/api/backtests/{summary['id']}/trades.csv")
        assert csv_response.status_code == 200
        assert len(csv_response.text.strip().splitlines()) == len(trades) + 1


def test_legacy_backtest_artifacts_tolerate_missing_snapshot_and_trades(tmp_path: Path) -> None:
    """Old artifacts without ``strategy_config``/``trades`` still load (null/[])."""
    settings = _settings(tmp_path)
    with _app(settings) as local:
        summary = _run_short_backtest(local)
        run_id = summary["id"]
        artifact = settings.backtest_artifacts_dir / f"{run_id}.json"
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        payload["summary"].pop("strategy_config", None)
        payload["tearsheet"].pop("trades", None)
        artifact.write_text(json.dumps(payload), encoding="utf-8")

    with _app(settings) as restarted:
        legacy = restarted.get(f"/api/backtests/{run_id}")
        assert legacy.status_code == 200
        assert legacy.json()["strategy_config"] is None
        legacy_tearsheet = restarted.get(f"/api/backtests/{run_id}/tearsheet")
        assert legacy_tearsheet.status_code == 200
        assert legacy_tearsheet.json()["trades"] == []
        assert restarted.get(f"/api/backtests/{run_id}/trades.csv").text.splitlines() == [
            "timestamp,order_id,instrument_id,side,quantity,price,notional,commissions,"
            "slippage,kind"
        ]
