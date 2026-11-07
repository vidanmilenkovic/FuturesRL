"""DRL checkpoint registry and offline training service (``/api/drl`` backend).

The service is deliberately standalone: it is constructed from
:class:`~configs.settings.Settings` / artifact paths only, owns no FastAPI state
and does not touch the live execution container. The REST router
(:mod:`src.api.routes.drl`) is mounted by the app factory through
``src/api/routes/__init__.py``; the module-level dependency
:func:`get_drl_service` resolves a service from, in priority order:

1. ``app.state.drl_service`` (tests / embedding),
2. the app container's ``settings``,
3. the process-wide :func:`~configs.settings.get_settings` singleton.

Training is the same callable the CLI uses
(:func:`src.strategies.train_drl.train_drl`), executed in a worker thread by the
route so the event loop is never blocked. The registry root is
``settings.drl_models_dir`` (``<report_dir>/models``, default ``artifacts/models``)
and can be overridden per service instance (tests/``tmp_path``).

REST contract
-------------
``GET /api/drl/models`` -> :class:`DRLModelListResponse`: every
``<models_dir>/*.zip`` checkpoint with the parsed ``<name>.zip.meta.json``
(algorithm, timesteps, universe roots, features, created timestamp, git commit,
reward-curve summary) and filesystem fallbacks for missing/corrupt sidecars.

``POST /api/drl/train`` -> :class:`DRLTrainResponse`: validates the algorithm,
bounded ``timesteps`` and checkpoint name, trains the strategy's ``drl:`` section
offline (synchronously for a small budget), persists ``<name>.zip`` +
``<name>.zip.meta.json`` + ``<name>.zip.reward_curve.json`` and returns the
training summary with the reward curve. Errors: ``404`` unknown strategy id,
``422`` invalid request / strategy without a ``drl:`` section, ``500`` training
infrastructure failure.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal, cast

from configs.settings import Settings, get_settings
from fastapi import Depends, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "DEFAULT_TRAIN_TIMESTEPS",
    "DRL_ALGORITHMS",
    "MAX_SYNC_TIMESTEPS",
    "DRLModelListResponse",
    "DRLModelSummary",
    "DRLRewardCurve",
    "DRLRewardCurveSummary",
    "DRLRewardPoint",
    "DRLService",
    "DRLTrainRequest",
    "DRLTrainResponse",
    "DrlServiceDep",
    "get_drl_service",
]

#: Supported Stable-Baselines3 algorithms (CPPO trains as risk-shaped PPO).
DRL_ALGORITHMS: tuple[str, ...] = ("PPO", "CPPO")
type DRLAlgorithm = Literal["PPO", "CPPO"]

#: Default training budget for an API-triggered smoke run.
DEFAULT_TRAIN_TIMESTEPS: int = 128
#: Maximum timesteps accepted by the synchronous ``POST /api/drl/train`` endpoint.
MAX_SYNC_TIMESTEPS: int = 20_000
#: Checkpoint stem contract (must stay in sync with ``src.strategies.train_drl``).
_CHECKPOINT_NAME_RE: re.Pattern[str] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_META_SUFFIX = ".meta.json"
_REWARD_CURVE_SUFFIX = ".reward_curve.json"

#: Signature of the (injectable) training callable.
type TrainFn = Callable[..., dict[str, object]]


class _ResponseModel(BaseModel):
    """Response model: tolerant of additive backend fields (frontend safety)."""

    model_config = ConfigDict(extra="ignore")


class _RequestModel(BaseModel):
    """Request model: unknown fields are rejected (frozen JSON contract)."""

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Reward-curve models
# ---------------------------------------------------------------------------


class DRLRewardPoint(_ResponseModel):
    """One reward-curve sample (mean reward of a contiguous step window)."""

    timestep: int
    mean_reward: float


class DRLRewardCurve(_ResponseModel):
    """Bounded reward curve recorded during training.

    ``points``/``window_mean_rewards`` are capped (the trainer down-samples to the
    requested number of windows), ``episode_rewards`` keeps at most the last 100
    completed-episode returns while ``episode_count`` reports the true total.
    """

    window_steps: int = 1
    points: list[DRLRewardPoint] = Field(default_factory=list)
    window_mean_rewards: list[float] = Field(default_factory=list)
    episode_rewards: list[float] = Field(default_factory=list)
    episode_count: int = 0
    env_steps: int = 0
    samples: int = 0
    final_mean_reward: float | None = None
    best_mean_reward: float | None = None


class DRLRewardCurveSummary(_ResponseModel):
    """Compact reward-curve statistics attached to a registry entry."""

    window_steps: int | None = None
    samples: int = 0
    episode_count: int = 0
    env_steps: int | None = None
    final_mean_reward: float | None = None
    best_mean_reward: float | None = None
    path: str | None = None


# ---------------------------------------------------------------------------
# Registry models
# ---------------------------------------------------------------------------


class DRLModelSummary(_ResponseModel):
    """One trained checkpoint discovered in ``settings.drl_models_dir``."""

    checkpoint_name: str
    model_path: str
    meta_path: str | None = None
    metadata_available: bool = False
    metadata_error: str | None = None
    file_size_bytes: int = Field(default=0, ge=0)
    algorithm: str | None = None
    sb3_algorithm: str | None = None
    strategy_id: str | None = None
    policy: str | None = None
    total_timesteps: int | None = None
    learning_rate: float | None = None
    gamma: float | None = None
    llm_reward_penalty_lambda: float | None = None
    seed: int | None = None
    device: str | None = None
    offline: bool | None = None
    universe: list[str] = Field(default_factory=list)
    features: list[str] = Field(default_factory=list)
    created_at: datetime | None = None
    git_commit: str | None = None
    reward_curve: DRLRewardCurveSummary | None = None


class DRLModelListResponse(_ResponseModel):
    """``GET /api/drl/models`` payload."""

    models: list[DRLModelSummary] = Field(default_factory=list)
    count: int = 0
    models_dir: str
    generated_at: datetime


# ---------------------------------------------------------------------------
# Training request / response models
# ---------------------------------------------------------------------------


class DRLTrainRequest(_RequestModel):
    """``POST /api/drl/train`` request body."""

    strategy_id: str = Field(
        default="drl_ppo_llm_risk",
        min_length=1,
        max_length=128,
        description="Strategy id from configs/strategies/*.yaml (must have a drl: section).",
    )
    algorithm: DRLAlgorithm = Field(
        default="PPO", description="Stable-Baselines3 algorithm (CPPO trains as risk-shaped PPO)."
    )
    timesteps: int = Field(
        default=DEFAULT_TRAIN_TIMESTEPS,
        ge=1,
        le=MAX_SYNC_TIMESTEPS,
        description=(
            "Synchronous training budget (bounded; this endpoint is meant for offline "
            f"smoke runs, max {MAX_SYNC_TIMESTEPS} timesteps)."
        ),
    )
    learning_rate: float | None = Field(default=None, gt=0.0, le=1.0)
    gamma: float | None = Field(default=None, ge=0.0, le=1.0)
    llm_reward_penalty_lambda: float | None = Field(default=None, ge=0.0)
    checkpoint_name: str | None = Field(
        default=None,
        max_length=64,
        description=(
            "Checkpoint stem inside the models directory (default: "
            "<strategy_id>_<algorithm>); letters/digits/._- only, no traversal."
        ),
    )
    seed: int | None = Field(default=None, ge=0)
    device: str = Field(default="cpu", min_length=1, description="SB3 device, e.g. cpu/cuda.")
    periods: int | None = Field(
        default=None,
        ge=2,
        le=100_000,
        description="Offline synthetic daily bars per root (default: trainer default).",
    )
    offline: bool = Field(
        default=True,
        description="Only deterministic offline training is supported; false is rejected.",
    )

    @field_validator("checkpoint_name")
    @classmethod
    def _valid_checkpoint_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not _CHECKPOINT_NAME_RE.fullmatch(cleaned) or ".." in cleaned:
            raise ValueError(
                "checkpoint_name must be 1-64 characters of letters, digits, '.', '_' or '-', "
                "start with a letter/digit and contain no '..' path traversal"
            )
        return cleaned

    @field_validator("device")
    @classmethod
    def _valid_device(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("device must be a non-empty Stable-Baselines3 device string")
        return cleaned


class DRLTrainResponse(_ResponseModel):
    """``POST /api/drl/train`` payload: the full training summary + reward curve."""

    status: Literal["ok"] = "ok"
    strategy_id: str
    strategy_path: str
    algorithm: str
    sb3_algorithm: str
    policy: str
    total_timesteps: int
    learning_rate: float
    gamma: float
    llm_reward_penalty_lambda: float
    seed: int
    device: str
    checkpoint_name: str
    model_path: str
    meta_path: str
    reward_curve_path: str | None = None
    universe: list[str] = Field(default_factory=list)
    feature_names: list[str] = Field(default_factory=list)
    feature_fallbacks: list[str] = Field(default_factory=list)
    observation_shape: list[int] = Field(default_factory=list)
    action_shape: list[int] = Field(default_factory=list)
    offline: bool = True
    periods: int = 0
    usable_rows: int = 0
    initial_equity: float = 0.0
    commission_bps: float = 0.0
    slippage_ticks: float = 0.0
    trained_at: datetime
    duration_seconds: float = 0.0
    total_duration_seconds: float = 0.0
    sb3_version: str = "unknown"
    git_commit: str | None = None
    hyperparameters: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    reward_curve: DRLRewardCurve = Field(default_factory=DRLRewardCurve)


# ---------------------------------------------------------------------------
# Metadata parsing helpers
# ---------------------------------------------------------------------------


def _meta_str(payload: dict[str, object], key: str) -> str | None:
    value = payload.get(key)
    return value if isinstance(value, str) and value else None


def _meta_int(payload: dict[str, object], key: str) -> int | None:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _meta_float(payload: dict[str, object], key: str) -> float | None:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _meta_bool(payload: dict[str, object], key: str) -> bool | None:
    value = payload.get(key)
    return value if isinstance(value, bool) else None


def _meta_str_list(payload: dict[str, object], key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [item for item in value if isinstance(item, str)]


def _meta_timestamp(payload: dict[str, object], key: str) -> datetime | None:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(UTC)


def _reward_curve_summary(
    metadata: dict[str, object], fallback_path: Path
) -> DRLRewardCurveSummary | None:
    """Build the compact curve summary from metadata (or the sidecar file)."""
    raw_path = _meta_str(metadata, "reward_curve_path")
    curve_path = Path(raw_path) if raw_path is not None else fallback_path
    curve: dict[str, object] | None = None
    raw_curve = metadata.get("reward_curve")
    if isinstance(raw_curve, dict):
        curve = {str(key): value for key, value in raw_curve.items()}
    elif curve_path.is_file():
        try:
            payload = json.loads(curve_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = None
        if isinstance(payload, dict):
            curve = {str(key): value for key, value in payload.items()}
    if curve is None:
        return None
    return DRLRewardCurveSummary(
        window_steps=_meta_int(curve, "window_steps"),
        samples=_meta_int(curve, "samples") or 0,
        episode_count=_meta_int(curve, "episode_count") or 0,
        env_steps=_meta_int(curve, "env_steps"),
        final_mean_reward=_meta_float(curve, "final_mean_reward"),
        best_mean_reward=_meta_float(curve, "best_mean_reward"),
        path=str(curve_path),
    )


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class DRLService:
    """DRL checkpoint registry + offline training service.

    Parameters
    ----------
    settings:
        Process settings; defaults to the cached :func:`get_settings` singleton.
    models_dir:
        Registry/training artifact root (default: ``settings.drl_models_dir``).
    strategies_dir:
        Directory holding strategy YAMLs (default: ``settings.strategies_dir``).
    clock:
        Injectable UTC clock (deterministic list ``generated_at`` in tests).
    train_fn:
        Injectable training callable (default: ``src.strategies.train_drl.train_drl``,
        imported lazily so importing the API never imports the DRL stack).
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        models_dir: str | Path | None = None,
        strategies_dir: str | Path | None = None,
        clock: Callable[[], datetime] | None = None,
        train_fn: TrainFn | None = None,
    ) -> None:
        self._settings = settings if settings is not None else get_settings()
        resolved_models_dir = (
            models_dir if models_dir is not None else self._settings.drl_models_dir
        )
        self._models_dir = Path(resolved_models_dir)
        resolved_strategies = (
            strategies_dir if strategies_dir is not None else self._settings.strategies_dir
        )
        self._strategies_dir = Path(resolved_strategies)
        self._clock = clock or (lambda: datetime.now(UTC))
        self._train_fn = train_fn

    # -- observability ---------------------------------------------------------

    @property
    def models_dir(self) -> Path:
        """Registry/training artifact root."""
        return self._models_dir

    @property
    def strategies_dir(self) -> Path:
        """Directory holding the strategy YAMLs."""
        return self._strategies_dir

    def _now(self) -> datetime:
        moment = self._clock()
        if moment.tzinfo is None or moment.utcoffset() is None:
            raise ValueError("clock must return a timezone-aware datetime")
        return moment.astimezone(UTC)

    # -- checkpoint registry ---------------------------------------------------

    def list_models(self) -> DRLModelListResponse:
        """List every ``*.zip`` checkpoint in the registry root (newest first)."""
        models: list[DRLModelSummary] = []
        if self._models_dir.is_dir():
            for checkpoint in sorted(self._models_dir.glob("*.zip")):
                models.append(self._summarise_checkpoint(checkpoint))
        models.sort(
            key=lambda summary: (
                summary.created_at if summary.created_at is not None else _EPOCH,
                summary.checkpoint_name,
            ),
            reverse=True,
        )
        return DRLModelListResponse(
            models=models,
            count=len(models),
            models_dir=str(self._models_dir),
            generated_at=self._now(),
        )

    def _summarise_checkpoint(self, checkpoint: Path) -> DRLModelSummary:
        meta_path = Path(f"{checkpoint}{_META_SUFFIX}")
        try:
            file_size = checkpoint.stat().st_size
        except OSError:
            file_size = 0
        metadata: dict[str, object] = {}
        metadata_error: str | None = None
        metadata_available = False
        if meta_path.is_file():
            try:
                payload = json.loads(meta_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    metadata = {str(key): value for key, value in payload.items()}
                    metadata_available = True
                else:
                    metadata_error = "metadata payload is not a JSON object"
            except (OSError, ValueError) as exc:
                metadata_error = f"unreadable metadata sidecar: {exc}"
        else:
            metadata_error = "metadata sidecar missing"
        created_at = _meta_timestamp(metadata, "trained_at")
        if created_at is None:
            try:
                created_at = datetime.fromtimestamp(checkpoint.stat().st_mtime, tz=UTC)
            except OSError:
                created_at = None
        curve_path = Path(
            _meta_str(metadata, "reward_curve_path") or f"{checkpoint}{_REWARD_CURVE_SUFFIX}"
        )
        return DRLModelSummary(
            checkpoint_name=checkpoint.stem,
            model_path=str(checkpoint),
            meta_path=str(meta_path) if meta_path.is_file() else None,
            metadata_available=metadata_available,
            metadata_error=metadata_error,
            file_size_bytes=file_size,
            algorithm=_meta_str(metadata, "algorithm"),
            sb3_algorithm=_meta_str(metadata, "sb3_algorithm"),
            strategy_id=_meta_str(metadata, "strategy_id"),
            policy=_meta_str(metadata, "policy"),
            total_timesteps=_meta_int(metadata, "total_timesteps"),
            learning_rate=_meta_float(metadata, "learning_rate"),
            gamma=_meta_float(metadata, "gamma"),
            llm_reward_penalty_lambda=_meta_float(metadata, "llm_reward_penalty_lambda"),
            seed=_meta_int(metadata, "seed"),
            device=_meta_str(metadata, "device"),
            offline=_meta_bool(metadata, "offline"),
            universe=_meta_str_list(metadata, "universe"),
            features=_meta_str_list(metadata, "feature_names"),
            created_at=created_at,
            git_commit=_meta_str(metadata, "git_commit"),
            reward_curve=_reward_curve_summary(metadata, curve_path),
        )

    # -- training --------------------------------------------------------------

    def train(self, request: DRLTrainRequest) -> DRLTrainResponse:
        """Validate and run one offline training request synchronously.

        Raises
        ------
        KeyError
            Unknown ``strategy_id`` (route -> ``404``).
        ValueError
            Invalid request semantics (non-offline, strategy without ``drl:``
            section, bad checkpoint name; route -> ``422``).
        RuntimeError
            Training infrastructure/argument failure (route -> ``500``).
        """
        if request.offline is not True:
            raise ValueError(
                "only deterministic offline training is supported; pass offline=true "
                "(no Databento/IBKR credentials are used by this endpoint)"
            )
        if request.timesteps < 1 or request.timesteps > MAX_SYNC_TIMESTEPS:
            raise ValueError(
                f"timesteps must be within [1, {MAX_SYNC_TIMESTEPS}], got {request.timesteps}"
            )
        checkpoint_name = self._resolve_checkpoint_name(request)
        strategy_path = self._resolve_strategy_path(request.strategy_id)
        train_fn = self._resolve_train_fn()
        summary = train_fn(
            strategy=strategy_path,
            algorithm=request.algorithm,
            timesteps=request.timesteps,
            learning_rate=request.learning_rate,
            gamma=request.gamma,
            llm_reward_penalty_lambda=request.llm_reward_penalty_lambda,
            checkpoint_name=checkpoint_name,
            output_dir=self._models_dir,
            offline=True,
            seed=request.seed,
            device=request.device,
            periods=request.periods,
        )
        return DRLTrainResponse.model_validate(summary)

    def _resolve_checkpoint_name(self, request: DRLTrainRequest) -> str:
        candidate = (
            request.checkpoint_name.strip()
            if request.checkpoint_name is not None
            else f"{request.strategy_id}_{request.algorithm.lower()}"
        )
        if not _CHECKPOINT_NAME_RE.fullmatch(candidate) or ".." in candidate:
            raise ValueError(
                "checkpoint_name must be 1-64 characters of letters, digits, '.', '_' or '-', "
                "start with a letter/digit and contain no '..' path traversal"
            )
        return candidate

    def _resolve_strategy_path(self, strategy_id: str) -> Path:
        from configs.strategy import load_strategy_config

        directory = self._strategies_dir
        if not directory.is_dir():
            raise RuntimeError(f"strategy directory not found: {directory}")
        known: list[str] = []
        for path in sorted(directory.glob("*.yaml")):
            try:
                config = load_strategy_config(path)
            except (OSError, ValueError) as exc:
                raise RuntimeError(f"invalid strategy config {path}: {exc}") from exc
            known.append(config.id)
            if config.id != strategy_id:
                continue
            if config.drl is None:
                raise ValueError(
                    f"strategy {strategy_id!r} has no 'drl:' section; add algorithm/policy/"
                    "total_timesteps/learning_rate/gamma/seed to its YAML before training"
                )
            return path
        raise KeyError(f"unknown strategy {strategy_id!r}; available: {sorted(known)!r}")

    def _resolve_train_fn(self) -> TrainFn:
        if self._train_fn is not None:
            return self._train_fn
        from src.strategies import train_drl as trainer_module

        return cast("TrainFn", trainer_module.train_drl)


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------


def get_drl_service(request: Request) -> DRLService:
    """Resolve the DRL service for a request (state override -> container -> settings)."""
    service = getattr(request.app.state, "drl_service", None)
    if isinstance(service, DRLService):
        return service
    container = getattr(request.app.state, "container", None)
    settings = getattr(container, "settings", None)
    if not isinstance(settings, Settings):
        settings = get_settings()
    return DRLService(settings=settings)


DrlServiceDep = Annotated[DRLService, Depends(get_drl_service)]
