"""Walk-forward optimization studies exposed by the Strategy Studio API.

The heavy lifting belongs to ``src.backtest.optimizer`` (backtest-eng): a synchronous
``WalkForwardOptimizer`` that scores every trial on purged/embargoed out-of-sample
walk-forward folds and persists a typed ``OptimizationResult``. This module is the
API-side service:

* translates the Studio's ``OptimizeRequest`` into the optimizer's request model
  (resolving catalog parameter keys to concrete dotted paths through the strategy
  layer when available),
* runs the optimizer in a worker thread from the route (synchronous engine),
* normalises the result into the API study contract, persists it atomically under
  ``settings.optimization_artifacts_dir/<study_id>.json`` and reloads studies on
  construction so a restart keeps the whole history,
* falls back to a deterministic offline *fixture* study when the optimizer module is
  not deployed (``provenance="fixture"`` + explicit message), so the Studio always
  works offline,
* applies the best parameters to the base strategy YAML via
  :class:`~src.api.studio_service.StrategyStudioService` and optionally triggers a
  verification backtest through an injected runner.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal, cast

import numpy as np
from configs.settings import Settings
from configs.strategy import StrategyConfig
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.api.schemas import StrategyConfigView
from src.api.studio_service import (
    StrategyStudioError,
    StrategyStudioService,
    apply_dotted_params,
    resolve_optimizer_path,
)

logger = logging.getLogger(__name__)

__all__ = [
    "OptimizationApplyRequest",
    "OptimizationApplyResponse",
    "OptimizationError",
    "OptimizationService",
    "OptimizationStudy",
    "OptimizationStudySummary",
    "OptimizeParameterRange",
    "OptimizeRequest",
    "TrialView",
]

#: Default optimizer module (backtest-eng task-26).
OPTIMIZER_MODULE = "src.backtest.optimizer"

#: Maximum accepted study payload size.
MAX_STUDY_PAYLOAD_BYTES: int = 8 * 1024 * 1024

#: Valid study id: optimizer-produced ids (``<strategy>-<objective>-Nt-sK``) and
#: API fixture ids (``opt-...``). Path separators / traversal are rejected below.
_STUDY_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,160}$")

type ObjectiveMetric = Literal["sharpe", "sortino", "calmar", "cagr", "rachev", "return_on_margin"]
type RangeType = Literal["int", "float", "categorical"]
type ScalarValue = bool | int | float | str
type Clock = Callable[[], datetime]

#: Objective names accepted by the optimizer's request model.
_OBJECTIVE_ALIASES: Mapping[str, str] = {
    "return_on_margin": "return_on_margin",
    "rom": "return_on_margin",
}


def _utc_now() -> datetime:
    return datetime.now(UTC)


class OptimizationError(RuntimeError):
    """A rejected/failed optimization request (with an HTTP status hint)."""

    def __init__(self, message: str, *, status_code: int = 422) -> None:
        super().__init__(message)
        self.status_code = status_code


# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------


class _StudyModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class OptimizeParameterRange(_StudyModel):
    """One search-space dimension (mirrors the optimizer's ``ParameterRange``)."""

    param_path: str
    type: RangeType = "float"
    low: float | None = None
    high: float | None = None
    step: float | None = None
    choices: list[ScalarValue] = Field(default_factory=list)


class OptimizeRequest(_StudyModel):
    """``POST /api/optimize`` body."""

    base_strategy_id: str
    objective_metric: ObjectiveMetric = "sharpe"
    n_trials: int = Field(default=25, ge=2, le=200)
    walk_forward: bool = True
    wf_splits: int = Field(default=4, ge=2, le=12)
    purge_bars: int = Field(default=5, ge=0)
    embargo_bars: int = Field(default=5, ge=0)
    drawdown_penalty_threshold: float = Field(default=0.20, ge=0.0, le=1.0)
    search_space: list[OptimizeParameterRange] = Field(default_factory=list)
    seed: int = 7


class TrialView(_StudyModel):
    """One evaluated trial in the persisted study."""

    trial_number: int = Field(ge=0)
    params: dict[str, ScalarValue] = Field(default_factory=dict)
    state: str = "COMPLETE"
    objective_value: float | None = None
    raw_objective: float | None = None
    oos_sharpe: float | None = None
    oos_sortino: float | None = None
    oos_calmar: float | None = None
    oos_cagr: float | None = None
    oos_rachev: float | None = None
    oos_return_on_margin: float | None = None
    max_dd: float | None = None
    duration_ms: float | None = None
    fold_count: int | None = None
    error: str | None = None


class OptimizationStudySummary(_StudyModel):
    """List-view row for one persisted optimization study."""

    study_id: str
    base_strategy_id: str
    objective_metric: str
    direction: str = "maximize"
    n_trials: int = 0
    completed_trials: int = 0
    best_trial_number: int | None = None
    best_params: dict[str, ScalarValue] = Field(default_factory=dict)
    best_value: float | None = None
    best_raw_value: float | None = None
    drawdown_penalty_threshold: float = 0.20
    created_at: datetime
    duration_ms: float | None = None
    fold_count: int | None = None
    walk_forward: bool = True
    backend: str = "deterministic"
    artifact_path: str | None = None
    schema_version: int = 1
    #: ``engine`` = real WalkForwardOptimizer; ``fixture`` = deterministic fallback.
    provenance: Literal["engine", "fixture"] = "fixture"
    message: str = ""


class OptimizationStudy(OptimizationStudySummary):
    """Full detail for ``GET /api/optimize/{study_id}``."""

    trials: list[TrialView] = Field(default_factory=list)
    param_importances: dict[str, float] = Field(default_factory=dict)
    best_strategy_config: dict[str, object] | None = None


class OptimizationApplyRequest(_StudyModel):
    """``POST /api/optimize/{study_id}/apply`` body."""

    new_strategy_id: str | None = None
    verification_backtest: bool = False
    start: date | None = None
    end: date | None = None
    initial_equity: float | None = Field(default=None, gt=0.0)


class OptimizationApplyResponse(_StudyModel):
    """Result of applying a study's best parameters as a new strategy."""

    study_id: str
    strategy: object
    best_params: dict[str, ScalarValue] = Field(default_factory=dict)
    verification_backtest: dict[str, object] | None = None
    message: str = ""


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _OptimizerOutcome:
    payload: Mapping[str, object]
    artifact_path: Path | None


class OptimizationService:
    """Persist/list/get/run/apply WalkForwardOptimizer studies for the Studio."""

    def __init__(
        self,
        *,
        settings: Settings,
        studio: StrategyStudioService,
        clock: Clock | None = None,
        output_dir: Path | None = None,
        optimizer_provider: Callable[[], object] | None = None,
        backtest_runner: Callable[
            [OptimizationApplyRequest, OptimizationStudy, StrategyConfigView],
            Mapping[str, object] | None,
        ]
        | None = None,
    ) -> None:
        self._settings = settings
        self._studio = studio
        self._clock: Clock = clock if clock is not None else _utc_now
        default_dir = getattr(
            settings, "optimization_artifacts_dir", settings.report_dir / "optimizations"
        )
        self._output_dir = Path(output_dir) if output_dir is not None else Path(default_dir)
        self._optimizer_provider = optimizer_provider
        self._backtest_runner = backtest_runner
        self._studies: dict[str, OptimizationStudy] = {}
        self._warnings: list[str] = []
        self.reload()

    # -- introspection ---------------------------------------------------------

    @property
    def warnings(self) -> tuple[str, ...]:
        """Non-fatal study-store warnings (corrupt files, write failures)."""
        return tuple(self._warnings)

    @property
    def studio(self) -> StrategyStudioService:
        """Strategy Studio service used for base configs and tuned-strategy writes."""
        return self._studio

    @property
    def output_dir(self) -> Path:
        """Directory holding ``<study_id>.json`` study payloads."""
        return self._output_dir

    @property
    def settings(self) -> Settings:
        """Runtime settings the service was built from."""
        return self._settings

    # -- persistence -----------------------------------------------------------

    def _study_path(self, study_id: str) -> Path:
        if (
            not _STUDY_ID_PATTERN.match(study_id)
            or ".." in study_id
            or "/" in study_id
            or "\\" in study_id
        ):
            raise OptimizationError(f"invalid study id {study_id!r}", status_code=400)
        target = (self._output_dir / f"{study_id}.json").resolve()
        resolved_base = self._output_dir.resolve()
        if resolved_base not in target.parents:
            raise OptimizationError(
                "refusing to access a path outside the study directory", status_code=400
            )
        return target

    def reload(self) -> int:
        """(Re)load persisted studies from disk; corrupt files are quarantined."""
        self._studies.clear()
        directory = self._output_dir
        if not directory.is_dir():
            return 0
        for path in sorted(directory.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                self._quarantine(path, exc)
                continue
            if not isinstance(payload, Mapping):
                self._quarantine(path, ValueError("study payload is not a JSON object"))
                continue
            candidate = dict(payload)
            if "provenance" not in candidate:
                # Optimizer-canonical artifacts carry ``request``/``seed`` and no API
                # provenance tag; the API marks them as real-engine studies.
                candidate["provenance"] = (
                    "engine" if ("request" in candidate or "seed" in candidate) else "fixture"
                )
            try:
                study = OptimizationStudy.model_validate(candidate)
            except (ValidationError, TypeError) as exc:
                self._quarantine(path, exc)
                continue
            self._studies[study.study_id] = study
        return len(self._studies)

    def _quarantine(self, path: Path, exc: BaseException) -> None:
        warning = f"corrupt optimization artifact {path.name}: {type(exc).__name__}: {exc}; skipped"
        logger.warning(warning)
        self._warnings.append(warning)
        try:
            path.replace(path.with_name(f"{path.name}.corrupt-{int(self._clock().timestamp())}"))
        except OSError:  # pragma: no cover - best-effort quarantine
            logger.exception("failed to quarantine corrupt optimization artifact %s", path)

    def _persist(self, study: OptimizationStudy) -> Path:
        try:
            directory = self._output_dir
            directory.mkdir(parents=True, exist_ok=True)
            target = self._study_path(study.study_id)
            payload = json.dumps(study.model_dump(mode="json"), sort_keys=True).encode("utf-8")
            if len(payload) > MAX_STUDY_PAYLOAD_BYTES:
                raise OptimizationError("optimization study payload is too large", status_code=413)
            fd, temp_name = tempfile.mkstemp(
                prefix=f".{study.study_id}.", suffix=".json.tmp", dir=str(directory)
            )
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                Path(temp_name).replace(target)
            except BaseException:
                Path(temp_name).unlink(missing_ok=True)
                raise
            return target
        except OptimizationError:
            raise
        except Exception as exc:  # the artifact store must never wedge the API
            warning = (
                f"could not persist optimization {study.study_id}: {type(exc).__name__}: {exc}"
            )
            logger.warning(warning)
            self._warnings.append(warning)
            return self._output_dir / f"{study.study_id}.json"

    # -- queries ---------------------------------------------------------------

    def list_studies(self) -> list[OptimizationStudySummary]:
        """Persisted studies, newest first."""
        summaries = [
            OptimizationStudySummary.model_validate(study.model_dump(mode="json"))
            for study in self._studies.values()
        ]
        return sorted(summaries, key=lambda summary: summary.created_at, reverse=True)

    def get_study(self, study_id: str) -> OptimizationStudy:
        """One study or :class:`OptimizationError` (404 when unknown)."""
        self._study_path(study_id)  # validates the id/path
        study = self._studies.get(study_id)
        if study is None:
            raise OptimizationError(f"unknown optimization study {study_id!r}", status_code=404)
        return study

    # -- run -------------------------------------------------------------------

    def _optimizer_module(self) -> object | None:
        if self._optimizer_provider is not None:
            try:
                return self._optimizer_provider()
            except Exception:  # pragma: no cover - defensive injection
                return None
        try:
            import importlib

            return importlib.import_module(OPTIMIZER_MODULE)
        except Exception:
            return None

    def run(self, request: OptimizeRequest) -> OptimizationStudy:
        """Run one synchronous study (real optimizer or deterministic fallback)."""
        base = self._studio.core_config(request.base_strategy_id)
        if base is None:
            raise OptimizationError(
                f"unknown base strategy {request.base_strategy_id!r}", status_code=404
            )
        base_definition = base.model_dump(mode="json")
        resolved_request = self._resolve_search_space(request, base_definition)
        study: OptimizationStudy | None = None
        message = ""
        module = self._optimizer_module()
        if module is not None:
            try:
                outcome = self._run_with_optimizer(module, base, resolved_request)
                study = _study_from_payload(
                    outcome.payload,
                    provenance="engine",
                    artifact_path=outcome.artifact_path,
                )
                if outcome.artifact_path is None or not outcome.artifact_path.is_file():
                    # Last-resort API persistence only when the optimizer did not
                    # write its canonical artifact (e.g. persist=False deployments).
                    study = study.model_copy(update={"artifact_path": str(self._persist(study))})
            except Exception as exc:
                message = f"optimizer unavailable, deterministic fixture study returned: {type(exc).__name__}: {exc}"
                logger.warning("WalkForwardOptimizer failed: %s", exc)
        else:
            message = (
                f"{OPTIMIZER_MODULE} is not deployed; deterministic fixture study returned "
                "(no Optuna/backtest evaluation was performed)"
            )
        if study is None:
            study = self._fixture_study(resolved_request, base, message)
            study = study.model_copy(update={"artifact_path": str(self._persist(study))})
        else:
            # The optimizer already persisted its canonical artifact; the API must
            # not rewrite it (``src.backtest.optimizer.load_optimization`` must keep
            # working on the same file).
            study = study.model_copy(update={"message": message})
        self._studies[study.study_id] = study
        return study

    def _resolve_search_space(
        self, request: OptimizeRequest, base_definition: Mapping[str, object]
    ) -> OptimizeRequest:
        """Resolve catalog keys (``timing.tsmom.lookback``) to concrete dotted paths."""
        if not request.search_space:
            return request
        resolved: list[OptimizeParameterRange] = []
        for dimension in request.search_space:
            path = dimension.param_path
            try:
                path = resolve_optimizer_path(base_definition, path)
            except Exception:  # pragma: no cover - defensive adapter
                path = dimension.param_path
            resolved.append(dimension.model_copy(update={"param_path": path}))
        return request.model_copy(update={"search_space": resolved})

    def _run_with_optimizer(
        self,
        module: object,
        config: StrategyConfig,
        request: OptimizeRequest,
    ) -> _OptimizerOutcome:
        optimizer_class = getattr(module, "WalkForwardOptimizer", None)
        if not callable(optimizer_class):
            raise OptimizationError(f"{OPTIMIZER_MODULE} does not expose WalkForwardOptimizer")
        request_class = getattr(module, "OptimizationRequest", None)
        payload = request.model_dump(mode="json")
        payload["base_strategy_id"] = config.id
        payload["objective_metric"] = _OBJECTIVE_ALIASES.get(
            payload["objective_metric"], payload["objective_metric"]
        )
        # The optimizer's request model is strict: numeric ranges must not carry
        # ``choices`` and it has no ``seed`` field (the seed is a constructor arg).
        payload.pop("seed", None)
        payload["search_space"] = _engine_search_space(payload.get("search_space"))
        if not payload.get("search_space"):
            default_builder = getattr(module, "default_search_space", None)
            if callable(default_builder):
                try:
                    defaults = default_builder(config)
                except Exception:  # pragma: no cover - defensive adapter
                    defaults = None
                if defaults:
                    payload["search_space"] = [
                        _object_mapping(item) for item in cast(Sequence[object], defaults)
                    ]
        if not payload.get("search_space"):
            raise OptimizationError("optimization request requires a non-empty search_space")
        if callable(request_class):
            engine_request = request_class.model_validate(payload)
        else:
            engine_request = payload
        import inspect

        try:
            signature = inspect.signature(optimizer_class)
        except (TypeError, ValueError):  # pragma: no cover - builtins only
            signature = None
        candidates: dict[str, object] = {
            "config": config,
            "request": engine_request,
            "instrument_master": self._settings.instruments,
            "seed": request.seed,
            "output_dir": self._output_dir,
            # API-triggered studies are synchronous/offline-friendly: the optimizer
            # may use its deterministic offline bars/pipeline fallbacks.
            "offline": True,
        }
        kwargs: dict[str, object] = {}
        if signature is not None:
            accepts_var_kwargs = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
            for name, parameter in signature.parameters.items():
                if name == "self":
                    continue
                if name in candidates:
                    kwargs[name] = candidates[name]
                elif parameter.kind is inspect.Parameter.VAR_KEYWORD and accepts_var_kwargs:
                    continue
                elif parameter.default is inspect.Parameter.empty and parameter.kind in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                ):
                    raise OptimizationError(
                        f"WalkForwardOptimizer requires unsupported parameter {name!r}"
                    )
        optimizer = optimizer_class(**kwargs)
        result = optimizer.optimize()
        result_payload = _result_payload(result)
        result_payload.setdefault("base_strategy_id", config.id)
        result_payload.setdefault("created_at", self._clock().isoformat())
        artifact = result_payload.get("artifact_path")
        artifact_path = Path(str(artifact)) if isinstance(artifact, str) and artifact else None
        if artifact_path is None or not artifact_path.is_file():
            writer = getattr(module, "write_optimization_artifact", None)
            study_id = str(result_payload.get("study_id") or config.id)
            if callable(writer):
                try:
                    artifact_path = Path(writer(result, self._output_dir / f"{study_id}.json"))
                    result_payload["artifact_path"] = str(artifact_path)
                except Exception as exc:  # pragma: no cover - defensive writer
                    logger.warning("could not write canonical optimization artifact: %s", exc)
        study = _study_from_payload(
            result_payload, provenance="engine", artifact_path=artifact_path
        )
        return _OptimizerOutcome(payload=study.model_dump(mode="json"), artifact_path=artifact_path)

    # -- deterministic fixture fallback ---------------------------------------

    def _fixture_search_space(self, request: OptimizeRequest) -> list[OptimizeParameterRange]:
        if request.search_space:
            return list(request.search_space)
        catalog = self._studio.catalog()
        spaces: list[OptimizeParameterRange] = []
        for spec in catalog.parameters:
            if not spec.optuna_tunable or spec.type not in {"int", "float"}:
                continue
            low = spec.min if spec.min is not None else 0.0
            high = (
                spec.max
                if spec.max is not None
                else (low + 1.0 if spec.type == "float" else low + 10.0)
            )
            if high <= low:
                high = low + 1.0
            spaces.append(
                OptimizeParameterRange(
                    param_path=spec.key,
                    type="int" if spec.type == "int" else "float",
                    low=low,
                    high=high,
                    step=spec.step,
                )
            )
            if len(spaces) >= 6:
                break
        if spaces:
            return spaces
        return [
            OptimizeParameterRange(
                param_path="allocator.params.target_annual_volatility",
                type="float",
                low=0.05,
                high=0.25,
                step=0.01,
            )
        ]

    def _sample_range(
        self, dimension: OptimizeParameterRange, rng: np.random.Generator
    ) -> ScalarValue:
        if dimension.type == "categorical":
            choices = list(dimension.choices)
            if not choices:
                return "default"
            return choices[int(rng.integers(0, len(choices)))]
        low = dimension.low if dimension.low is not None else 0.0
        high = dimension.high if dimension.high is not None else low + 1.0
        if high <= low:
            high = low + 1.0
        draw = float(rng.uniform(low, high))
        step = dimension.step
        if dimension.type == "int":
            return round(draw)
        if step is not None and step > 0.0:
            return round(low + round((draw - low) / step) * step, 8)
        return round(draw, 8)

    def _fixture_study(
        self,
        request: OptimizeRequest,
        base: StrategyConfig,
        message: str,
    ) -> OptimizationStudy:
        spaces = self._fixture_search_space(request)
        seed_material = (
            f"{request.base_strategy_id}|{request.objective_metric}|{request.n_trials}|"
            f"{request.wf_splits}|{request.seed}|{len(spaces)}"
        )
        digest = hashlib.sha1(seed_material.encode("utf-8")).hexdigest()[:8]
        rng = np.random.default_rng(request.seed + int(digest, 16) % 100_003)
        trials: list[TrialView] = []
        fold_count = request.wf_splits if request.walk_forward else 1
        for trial_number in range(request.n_trials):
            params: dict[str, ScalarValue] = {
                dimension.param_path: self._sample_range(dimension, rng) for dimension in spaces
            }
            raw = self._fixture_objective(params, request, rng)
            max_dd = max(
                0.0, min(1.0, 0.04 + 0.35 * abs(math.sin(trial_number * 0.7 + request.seed)))
            )
            penalty = max(0.0, max_dd - request.drawdown_penalty_threshold)
            objective = raw - penalty
            trials.append(
                TrialView(
                    trial_number=trial_number,
                    params=params,
                    state="COMPLETE",
                    objective_value=round(objective, 8),
                    raw_objective=round(raw, 8),
                    oos_sharpe=round(raw, 6),
                    oos_sortino=round(raw * 1.15, 6),
                    oos_calmar=round(raw * 0.9, 6),
                    oos_cagr=round(raw * 0.08, 8),
                    oos_rachev=round(1.0 + raw * 0.25, 6),
                    oos_return_on_margin=round(raw * 0.2, 8),
                    max_dd=round(max_dd, 8),
                    duration_ms=round(4.0 + float(rng.uniform(0.0, 6.0)), 3),
                    fold_count=fold_count,
                )
            )
        best = max(trials, key=lambda trial: trial.objective_value or -math.inf)
        best_config: dict[str, object] | None
        try:
            tuned = apply_dotted_params(base, best.params)
            best_config = tuned.model_dump(mode="json")
        except StrategyStudioError:
            best_config = None
        importances = {
            dimension.param_path: round(float(abs(math.sin(index + 1.0 + request.seed))), 6)
            for index, dimension in enumerate(spaces)
        }
        timestamp = self._clock()
        study_id = self._allocate_study_id(f"opt-{timestamp:%Y%m%dT%H%M%S}-{digest}")
        return OptimizationStudy(
            study_id=study_id,
            base_strategy_id=base.id,
            objective_metric=request.objective_metric,
            n_trials=request.n_trials,
            completed_trials=len(trials),
            best_trial_number=best.trial_number,
            best_params=dict(best.params),
            best_value=best.objective_value,
            best_raw_value=best.raw_objective,
            drawdown_penalty_threshold=request.drawdown_penalty_threshold,
            created_at=timestamp,
            duration_ms=round(float(sum(trial.duration_ms or 0.0 for trial in trials)), 3),
            fold_count=fold_count,
            walk_forward=request.walk_forward,
            backend="deterministic-fixture",
            artifact_path=None,
            schema_version=1,
            provenance="fixture",
            message=message,
            trials=trials,
            param_importances=importances,
            best_strategy_config=best_config,
        )

    def _fixture_objective(
        self, params: Mapping[str, ScalarValue], request: OptimizeRequest, rng: np.random.Generator
    ) -> float:
        del rng
        influence = 0.0
        for index, value in enumerate(params.values()):
            if isinstance(value, bool):
                numeric = 1.0 if value else 0.0
            elif isinstance(value, int | float):
                numeric = float(value)
            else:
                numeric = 0.05 * len(str(value))
            influence += 0.15 * math.sin(numeric * (index + 1) + request.seed)
        base = 0.55 + 0.2 * math.sin(request.seed * 0.7)
        return float(base + influence)

    def _allocate_study_id(self, candidate: str) -> str:
        study_id = candidate
        suffix = 1
        while study_id in self._studies or self._study_path(study_id).exists():
            suffix += 1
            study_id = f"{candidate}-{suffix}"
        return study_id

    # -- apply -----------------------------------------------------------------

    def apply(
        self,
        study_id: str,
        request: OptimizationApplyRequest | None = None,
        *,
        backtest_runner: Callable[
            [OptimizationApplyRequest, OptimizationStudy, StrategyConfigView],
            Mapping[str, object] | None,
        ]
        | None = None,
    ) -> OptimizationApplyResponse:
        """Apply a study's best parameters as a new strategy (hermetic YAML write)."""
        resolved_request = request or OptimizationApplyRequest()
        study = self.get_study(study_id)
        if not study.best_params:
            raise OptimizationError(
                f"optimization study {study_id!r} has no best parameters", status_code=409
            )
        try:
            view, config = self._studio.apply_best_params(
                study.base_strategy_id,
                study.best_params,
                strategy_id=resolved_request.new_strategy_id,
            )
        except StrategyStudioError as exc:
            raise OptimizationError(str(exc), status_code=exc.status_code) from exc
        verification: Mapping[str, object] | None = None
        message = (
            f"applied {len(study.best_params)} best parameter(s) from {study.study_id} "
            f"to new strategy {config.id!r}"
        )
        runner = backtest_runner or self._backtest_runner
        if resolved_request.verification_backtest and runner is not None:
            try:
                verification = runner(resolved_request, study, view)
                if verification is not None:
                    message += "; verification backtest started"
            except Exception as exc:
                verification = {"error": f"{type(exc).__name__}: {exc}"}
                message += "; verification backtest failed"
        return OptimizationApplyResponse(
            study_id=study.study_id,
            strategy=view,
            best_params=dict(study.best_params),
            verification_backtest=dict(verification) if verification is not None else None,
            message=message,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result_payload(result: object) -> dict[str, object]:
    """Normalise an ``OptimizationResult``-like object into a JSON mapping."""
    to_dict = getattr(result, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return dict(cast(Mapping[str, object], payload))
    model_dump = getattr(result, "model_dump", None)
    if callable(model_dump):
        payload = model_dump(mode="json")
        if isinstance(payload, Mapping):
            return dict(cast(Mapping[str, object], payload))
    fields = (
        "study_id",
        "base_strategy_id",
        "objective_metric",
        "direction",
        "n_trials",
        "completed_trials",
        "best_trial_number",
        "best_params",
        "best_value",
        "best_raw_value",
        "best_strategy_config",
        "trials",
        "param_importances",
        "drawdown_penalty_threshold",
        "created_at",
        "duration_ms",
        "fold_count",
        "walk_forward",
        "backend",
        "artifact_path",
        "schema_version",
    )
    payload = {}
    for field in fields:
        if hasattr(result, field):
            payload[field] = getattr(result, field)
    return payload


def _engine_search_space(payload: object) -> list[dict[str, object]]:
    """Normalise API search-space dimensions for the optimizer's strict model."""
    if not isinstance(payload, Sequence) or isinstance(payload, str | bytes):
        return []
    dimensions: list[dict[str, object]] = []
    for entry in cast(Sequence[object], payload):
        mapping = _object_mapping(entry)
        param_path = mapping.get("param_path")
        if not isinstance(param_path, str) or not param_path:
            continue
        range_type = str(mapping.get("type") or "float")
        dimension: dict[str, object] = {"param_path": param_path, "type": range_type}
        if range_type == "categorical":
            choices = mapping.get("choices")
            dimension["choices"] = [
                choice for choice in _sequence_values(choices) if isinstance(choice, str)
            ]
        else:
            for key in ("low", "high", "step"):
                value = mapping.get(key)
                if isinstance(value, int | float) and not isinstance(value, bool):
                    dimension[key] = float(value)
        dimensions.append(dimension)
    return dimensions


def _sequence_values(payload: object) -> list[object]:
    if isinstance(payload, Sequence) and not isinstance(payload, str | bytes | bytearray):
        return list(cast(Sequence[object], payload))
    return []


def _study_from_payload(
    payload: Mapping[str, object],
    *,
    provenance: Literal["engine", "fixture"],
    artifact_path: Path | None = None,
) -> OptimizationStudy:
    payload = dict(payload)
    payload["provenance"] = payload.get("provenance", provenance)
    if payload.get("artifact_path") is None and artifact_path is not None:
        payload["artifact_path"] = str(artifact_path)
    payload.setdefault("schema_version", 1)
    payload.setdefault("best_params", {})
    trials = payload.get("trials")
    normalised_trials: list[dict[str, object]] = []
    if isinstance(trials, Sequence) and not isinstance(trials, str | bytes):
        for entry in cast(Sequence[object], trials):
            mapping = entry if isinstance(entry, Mapping) else _object_mapping(entry)
            if mapping:
                normalised_trials.append(dict(cast(Mapping[str, object], mapping)))
    payload["trials"] = normalised_trials
    if isinstance(payload.get("created_at"), str):
        try:
            payload["created_at"] = datetime.fromisoformat(
                str(payload["created_at"]).replace("Z", "+00:00")
            )
        except ValueError:
            payload["created_at"] = _utc_now()
    try:
        return OptimizationStudy.model_validate(payload)
    except ValidationError as exc:
        raise OptimizationError(f"optimizer returned an invalid study payload: {exc}") from exc


def _object_mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return cast(Mapping[str, object], value)
    for attribute in ("model_dump", "to_dict"):
        dumper = getattr(value, attribute, None)
        if callable(dumper):
            try:
                dumped = dumper()
            except Exception:  # pragma: no cover - defensive adapter
                return {}
            return cast(Mapping[str, object], dumped) if isinstance(dumped, Mapping) else {}
    return {}
