"""Integration tests for the Phase-10 ``/api/drl`` routes (B3b).

Fully offline/deterministic: the DRL trainer is a fake callable that writes the
same artifact trio as the real trainer (``<name>.zip`` + ``.meta.json`` +
``.reward_curve.json``), so these tests cover request validation, persistence and
registry parsing without importing the SB3/torch stack or training a model.
"""

from __future__ import annotations

import json
import os
import types
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from configs.settings import Settings
from fastapi import FastAPI
from fastapi.testclient import TestClient
from src.api.drl_service import (
    MAX_SYNC_TIMESTEPS,
    DRLModelListResponse,
    DRLService,
    DRLTrainResponse,
)
from src.api.routes.drl import router

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
#: Deterministic clock for ``generated_at`` / default timestamps.
FIXED_TIMESTAMP = datetime(2026, 11, 2, 17, 0, tzinfo=UTC)


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        environment="backtest",
        data_dir=tmp_path / "data",
        parquet_dir=tmp_path / "data" / "parquet",
        duckdb_path=tmp_path / "data" / "drl.duckdb",
        report_dir=tmp_path / "artifacts",
        instruments_path=REPO_ROOT / "configs" / "instruments.yaml",
    )


class FakeDRLTrainer:
    """Deterministic stand-in for ``src.strategies.train_drl.train_drl``."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(dict(kwargs))
        output_dir = Path(str(kwargs["output_dir"]))
        checkpoint_name = str(kwargs["checkpoint_name"])
        algorithm = str(kwargs["algorithm"])
        timesteps = int(str(kwargs["timesteps"]))
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = output_dir / f"{checkpoint_name}.zip"
        checkpoint.write_bytes(b"fake-sb3-checkpoint-bytes")
        curve = {
            "window_steps": 1,
            "points": [
                {"timestep": 2, "mean_reward": 0.002},
                {"timestep": 4, "mean_reward": 0.004},
            ],
            "window_mean_rewards": [0.002, 0.004],
            "episode_rewards": [0.01],
            "episode_count": 1,
            "env_steps": 4,
            "samples": 2,
            "final_mean_reward": 0.004,
            "best_mean_reward": 0.004,
        }
        curve_path = Path(f"{checkpoint}.reward_curve.json")
        curve_path.write_text(json.dumps(curve), encoding="utf-8")
        learning_rate = kwargs.get("learning_rate")
        gamma = kwargs.get("gamma")
        penalty = kwargs.get("llm_reward_penalty_lambda")
        seed = kwargs.get("seed")
        periods = kwargs.get("periods")
        effective_learning_rate = 3e-4 if learning_rate is None else float(str(learning_rate))
        effective_gamma = 0.99 if gamma is None else float(str(gamma))
        effective_penalty = 0.5 if penalty is None else float(str(penalty))
        effective_seed = 7 if seed is None else int(str(seed))
        effective_periods = 512 if periods is None else int(str(periods))
        sb3_algorithm = "PPO" if algorithm == "CPPO" else algorithm
        hyperparameters: dict[str, object] = {
            "algorithm": algorithm,
            "sb3_algorithm": sb3_algorithm,
            "policy": "MlpPolicy",
            "total_timesteps": timesteps,
            "learning_rate": effective_learning_rate,
            "gamma": effective_gamma,
            "llm_reward_penalty_lambda": effective_penalty,
            "seed": effective_seed,
            "device": str(kwargs["device"]),
            "reward_curve_windows": 20,
        }
        summary: dict[str, object] = {
            "status": "ok",
            "strategy_id": "drl_ppo_llm_risk",
            "strategy_path": str(kwargs["strategy"]),
            "algorithm": algorithm,
            "sb3_algorithm": sb3_algorithm,
            "policy": "MlpPolicy",
            "total_timesteps": timesteps,
            "learning_rate": effective_learning_rate,
            "gamma": effective_gamma,
            "llm_reward_penalty_lambda": effective_penalty,
            "seed": effective_seed,
            "device": str(kwargs["device"]),
            "checkpoint_name": checkpoint_name,
            "model_path": str(checkpoint),
            "meta_path": f"{checkpoint}.meta.json",
            "reward_curve_path": str(curve_path),
            "universe": ["ES", "NQ", "ZN"],
            "feature_names": ["ret_1d", "term_slope", "carry_annualized"],
            "feature_fallbacks": [],
            "observation_shape": [10],
            "action_shape": [3],
            "offline": True,
            "periods": effective_periods,
            "usable_rows": 128,
            "initial_equity": 1_000_000.0,
            "commission_bps": 3.0,
            "slippage_ticks": 1.0,
            "trained_at": FIXED_TIMESTAMP.isoformat(),
            "duration_seconds": 0.5,
            "total_duration_seconds": 0.75,
            "sb3_version": "2.9.2-test",
            "git_commit": "cafebabecafebabe",
            "hyperparameters": hyperparameters,
            "reward_curve": curve,
        }
        Path(f"{checkpoint}.meta.json").write_text(json.dumps(summary), encoding="utf-8")
        return summary


def _write_artifact(
    models_dir: Path,
    name: str,
    *,
    trained_at: datetime,
    metadata: dict[str, object] | None = None,
    write_meta: bool = True,
    corrupt_meta: bool = False,
) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = models_dir / f"{name}.zip"
    checkpoint.write_bytes(b"fake-model")
    if corrupt_meta:
        Path(f"{checkpoint}.meta.json").write_text("{not valid json", encoding="utf-8")
    elif write_meta:
        payload = dict(metadata or {})
        payload.setdefault("trained_at", trained_at.isoformat())
        payload.setdefault("algorithm", "PPO")
        Path(f"{checkpoint}.meta.json").write_text(json.dumps(payload), encoding="utf-8")
    return checkpoint


@pytest.fixture()
def trainer() -> FakeDRLTrainer:
    return FakeDRLTrainer()


@pytest.fixture()
def service(tmp_path: Path, trainer: FakeDRLTrainer) -> DRLService:
    return DRLService(settings=_settings(tmp_path), clock=lambda: FIXED_TIMESTAMP, train_fn=trainer)


@pytest.fixture()
def client(service: DRLService) -> Iterator[TestClient]:
    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.state.drl_service = service
    with TestClient(app) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# GET /api/drl/models
# ---------------------------------------------------------------------------


def test_models_empty_registry(client: TestClient, service: DRLService) -> None:
    response = client.get("/api/drl/models")
    assert response.status_code == 200
    body = response.json()
    assert body["models"] == []
    assert body["count"] == 0
    assert body["models_dir"] == str(service.models_dir)
    assert body["generated_at"].startswith("2026-11-02T17:00:00")


def test_models_lists_and_parses_artifacts(
    client: TestClient, service: DRLService, tmp_path: Path
) -> None:
    models_dir = service.models_dir
    newer_meta: dict[str, object] = {
        "status": "ok",
        "algorithm": "PPO",
        "sb3_algorithm": "PPO",
        "strategy_id": "drl_ppo_llm_risk",
        "policy": "MlpPolicy",
        "total_timesteps": 64,
        "universe": ["ES", "NQ"],
        "feature_names": ["ret_1d", "term_slope"],
        "git_commit": "deadbeef" * 5,
        "offline": True,
        "trained_at": (FIXED_TIMESTAMP - timedelta(hours=1)).isoformat(),
        "reward_curve": {
            "window_steps": 2,
            "samples": 4,
            "episode_count": 2,
            "env_steps": 200,
            "final_mean_reward": 0.012,
            "best_mean_reward": 0.03,
        },
        "reward_curve_path": str(models_dir / "newer_model.zip.reward_curve.json"),
    }
    _write_artifact(models_dir, "newer_model", trained_at=FIXED_TIMESTAMP, metadata=newer_meta)
    older = _write_artifact(
        models_dir,
        "older_model",
        trained_at=FIXED_TIMESTAMP - timedelta(days=3),
        metadata={"algorithm": "SAC", "total_timesteps": 32, "universe": ["CL"]},
    )
    mtime = FIXED_TIMESTAMP.timestamp() - 2 * 86_400
    os.utime(older, (mtime, mtime))
    no_meta = _write_artifact(
        models_dir, "no_meta", trained_at=FIXED_TIMESTAMP - timedelta(days=2), write_meta=False
    )
    os.utime(no_meta, (FIXED_TIMESTAMP.timestamp() - 5 * 86_400,) * 2)
    corrupt = _write_artifact(
        models_dir,
        "corrupt_meta",
        trained_at=FIXED_TIMESTAMP - timedelta(days=4),
        corrupt_meta=True,
    )
    os.utime(corrupt, (FIXED_TIMESTAMP.timestamp() - 6 * 86_400,) * 2)

    response = client.get("/api/drl/models")
    assert response.status_code == 200
    body = DRLModelListResponse.model_validate(response.json())
    assert body.count == 4
    names = [model.checkpoint_name for model in body.models]
    # Registry sorts newest first using trained_at (fallback: file mtime).
    assert names == ["newer_model", "older_model", "no_meta", "corrupt_meta"]

    newer = body.models[0]
    assert newer.metadata_available is True
    assert newer.metadata_error is None
    assert newer.algorithm == "PPO"
    assert newer.total_timesteps == 64
    assert newer.universe == ["ES", "NQ"]
    assert newer.features == ["ret_1d", "term_slope"]
    assert newer.git_commit == "deadbeef" * 5
    assert newer.offline is True
    assert newer.created_at is not None
    assert newer.created_at == FIXED_TIMESTAMP - timedelta(hours=1)
    assert newer.reward_curve is not None
    assert newer.reward_curve.samples == 4
    assert newer.reward_curve.episode_count == 2
    assert newer.reward_curve.final_mean_reward == pytest.approx(0.012)
    assert newer.reward_curve.path is not None
    assert newer.file_size_bytes == len(b"fake-model")
    assert newer.meta_path is not None

    older_model = body.models[1]
    assert older_model.algorithm == "SAC"
    assert older_model.total_timesteps == 32
    assert older_model.universe == ["CL"]
    assert older_model.reward_curve is None

    no_meta_model = body.models[2]
    assert no_meta_model.metadata_available is False
    assert no_meta_model.metadata_error is not None
    assert "missing" in no_meta_model.metadata_error.lower()
    assert no_meta_model.algorithm is None
    assert no_meta_model.created_at is not None  # filesystem mtime fallback

    corrupt_model = body.models[3]
    assert corrupt_model.metadata_available is False
    assert corrupt_model.metadata_error is not None
    assert "unreadable" in corrupt_model.metadata_error.lower()
    assert body.generated_at == FIXED_TIMESTAMP


# ---------------------------------------------------------------------------
# POST /api/drl/train
# ---------------------------------------------------------------------------


def test_train_persists_artifacts_and_registry_lists_them(
    client: TestClient,
    trainer: FakeDRLTrainer,
    service: DRLService,
    tmp_path: Path,
) -> None:
    payload = {
        "timesteps": 8,
        "periods": 160,
        "checkpoint_name": "api_unit_ppo",
        "learning_rate": 0.0005,
        "gamma": 0.95,
        "llm_reward_penalty_lambda": 0.2,
        "seed": 5,
    }
    response = client.post("/api/drl/train", json=payload)
    assert response.status_code == 200, response.text
    body = DRLTrainResponse.model_validate(response.json())
    assert body.status == "ok"
    assert body.checkpoint_name == "api_unit_ppo"
    assert body.algorithm == "PPO"
    assert body.sb3_algorithm == "PPO"
    assert body.total_timesteps == 8
    assert body.learning_rate == pytest.approx(0.0005)
    assert body.gamma == pytest.approx(0.95)
    assert body.llm_reward_penalty_lambda == pytest.approx(0.2)
    assert body.seed == 5
    assert body.model_path.endswith("api_unit_ppo.zip")
    assert body.meta_path.endswith("api_unit_ppo.zip.meta.json")
    assert body.reward_curve_path is not None
    assert body.reward_curve.samples == 2
    assert body.reward_curve.final_mean_reward == pytest.approx(0.004)
    assert body.hyperparameters["learning_rate"] == pytest.approx(0.0005)
    assert body.universe == ["ES", "NQ", "ZN"]
    assert body.feature_fallbacks == []
    assert body.trained_at == FIXED_TIMESTAMP

    # The fake trainer received the request parameters and the service models dir.
    assert len(trainer.calls) == 1
    call = trainer.calls[0]
    assert str(call["strategy"]).endswith("drl_ppo_llm_risk.yaml")
    assert call["algorithm"] == "PPO"
    assert call["timesteps"] == 8
    assert call["checkpoint_name"] == "api_unit_ppo"
    assert Path(str(call["output_dir"])) == service.models_dir
    assert call["offline"] is True

    # Persistence: the artifact trio exists and a *new* service instance sees it.
    models_dir = service.models_dir
    for name in (
        "api_unit_ppo.zip",
        "api_unit_ppo.zip.meta.json",
        "api_unit_ppo.zip.reward_curve.json",
    ):
        assert (models_dir / name).is_file()
    reloaded = DRLService(settings=_settings(tmp_path), clock=lambda: FIXED_TIMESTAMP).list_models()
    assert reloaded.count == 1
    assert reloaded.models[0].checkpoint_name == "api_unit_ppo"
    assert reloaded.models[0].metadata_available is True


def test_train_default_checkpoint_name_and_cppo_mapping(
    client: TestClient, trainer: FakeDRLTrainer, service: DRLService
) -> None:
    response = client.post("/api/drl/train", json={"algorithm": "CPPO", "timesteps": 4})
    assert response.status_code == 200, response.text
    body = DRLTrainResponse.model_validate(response.json())
    assert body.checkpoint_name == "drl_ppo_llm_risk_cppo"
    assert body.algorithm == "CPPO"
    assert body.sb3_algorithm == "PPO"
    assert body.hyperparameters["sb3_algorithm"] == "PPO"
    assert (service.models_dir / "drl_ppo_llm_risk_cppo.zip").is_file()
    assert trainer.calls[0]["algorithm"] == "CPPO"


@pytest.mark.parametrize(
    ("payload", "expected_status"),
    [
        ({"algorithm": "DQN"}, 422),
        ({"timesteps": 0}, 422),
        ({"timesteps": MAX_SYNC_TIMESTEPS + 1}, 422),
        ({"learning_rate": 0}, 422),
        ({"gamma": 1.5}, 422),
        ({"llm_reward_penalty_lambda": -1}, 422),
        ({"checkpoint_name": "../evil"}, 422),
        ({"checkpoint_name": "bad name"}, 422),
        ({"device": "   "}, 422),
        ({"offline": False}, 422),
        ({"strategy_id": "tsmom_vol_target"}, 422),
        ({"strategy_id": "does_not_exist"}, 404),
        ({"unexpected_field": 1}, 422),
    ],
)
def test_train_validation_errors_never_call_the_trainer(
    client: TestClient,
    trainer: FakeDRLTrainer,
    payload: dict[str, object],
    expected_status: int,
) -> None:
    response = client.post("/api/drl/train", json=payload)
    assert response.status_code == expected_status, response.text
    assert trainer.calls == []


def test_train_response_exposes_reward_curve_for_frontend(
    client: TestClient,
) -> None:
    response = client.post("/api/drl/train", json={"timesteps": 16})
    assert response.status_code == 200
    curve = response.json()["reward_curve"]
    assert set(curve) >= {
        "window_steps",
        "points",
        "window_mean_rewards",
        "episode_rewards",
        "episode_count",
        "env_steps",
        "samples",
        "final_mean_reward",
        "best_mean_reward",
    }
    assert isinstance(curve["points"], list)
    for point in curve["points"]:
        assert set(point) == {"timestep", "mean_reward"}


# ---------------------------------------------------------------------------
# Dependency resolution / service construction
# ---------------------------------------------------------------------------


def test_dependency_falls_back_to_container_settings(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.state.container = types.SimpleNamespace(settings=settings)
    with TestClient(app) as client:
        response = client.get("/api/drl/models")
    assert response.status_code == 200
    body = response.json()
    assert body["models_dir"] == str(settings.drl_models_dir)
    assert body["count"] == 0


def test_service_models_dir_override(tmp_path: Path) -> None:
    custom = tmp_path / "custom_models"
    _write_artifact(custom, "custom", trained_at=FIXED_TIMESTAMP)
    service = DRLService(
        settings=_settings(tmp_path),
        models_dir=custom,
        clock=lambda: FIXED_TIMESTAMP,
    )
    listing = service.list_models()
    assert service.models_dir == custom
    assert listing.count == 1
    assert listing.models[0].checkpoint_name == "custom"
