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
    n_trades: int = 0
    avg_gross_exposure: float = 0.0
    zero_activity: bool = False
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
    #: Kept for backwards compatibility; validation now defaults to on and apply
    #: always triggers+persists a walk-forward verification when a runner exists.
    verification_backtest: bool = True
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
    #: Additive v1.5.3 response fields: visible save + persisted verification run.
    saved_strategy_id: str = ""
    strategy_path: str = ""
    backtest_run_id: str | None = None


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


#: Parameter names that are structural/data-hygiene (never tunable). Used by the
#: graceful fallback validator when ``src.backtest.optimizer``'s legacy validator
#: is not importable (e.g. partially deployed package).
_STRUCTURAL_PARAM_NAMES: frozenset[str] = frozenset(
    {
        "min_avg_dollar_volume",
        "min_open_interest",
        "top_n",
        "min_history_days",
        "min_universe",
        "filter_negative_trend",
        "require_both",
        "risk_off_roots",
        "risk_off_sectors",
        "mode",
        "components",
        "normalize_observations",
        "volatility_targeting_enabled",
        "daily_loss_kill_switch_enabled",
        "sample_count",
        "cache_ttl_seconds",
        "macro_lockout_minutes_before",
        "macro_lockout_minutes_after",
        "macro_lockout_action",
        "macro_lockout_leverage_cap",
        "rounding_mode",
        "rebalance_buffer_contracts",
        "enable_micro_routing",
        "min_micro_notional_ratio",
    }
)


@dataclass(frozen=True, slots=True)
class QuarantinedStudy:
    """One persisted study moved to ``<optimizations>/legacy/`` at startup."""

    study_id: str
    source_path: Path
    quarantine_path: Path
    reasons: tuple[str, ...]


def _path_value(definition: object, path: str) -> tuple[bool, object]:
    """Walk a dotted path (dict keys / list indices) and report whether it exists."""
    cursor: object = definition
    for segment in path.split("."):
        if isinstance(cursor, list):
            if not segment.isdigit() or int(segment) >= len(cursor):
                return (False, None)
            cursor = cursor[int(segment)]
            continue
        if not isinstance(cursor, Mapping):
            return (False, None)
        mapping = cast(Mapping[str, object], cursor)
        if segment not in mapping:
            return (False, None)
        cursor = mapping[segment]
    return (True, cursor)


def _parameter_paths(payload: Mapping[str, object]) -> tuple[str, ...]:
    """Collect every tuned/search-space parameter path mentioned by a study payload."""
    paths: list[str] = []

    def add(value: object) -> None:
        if isinstance(value, str) and value.strip() and value.strip() not in paths:
            paths.append(value.strip())

    best_params = payload.get("best_params")
    if isinstance(best_params, Mapping):
        for key in cast(Mapping[object, object], best_params):
            add(key)
    request = payload.get("request")
    if isinstance(request, Mapping):
        search_space = cast(Mapping[str, object], request).get("search_space")
        for entry in _sequence_items(search_space):
            mapping = _mapping_items(entry)
            add(mapping.get("param_path"))
    trials = payload.get("trials")
    for entry in _sequence_items(trials):
        mapping = _mapping_items(entry)
        params = mapping.get("params")
        if isinstance(params, Mapping):
            for key in cast(Mapping[object, object], params):
                add(key)
        for key in ("search_space", "param_path"):
            value = mapping.get(key)
            if isinstance(value, str):
                add(value)
    return tuple(paths)


def _sequence_items(value: object) -> list[object]:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return list(cast(Sequence[object], value))
    return []


def _mapping_items(value: object) -> Mapping[str, object]:
    return cast(Mapping[str, object], value) if isinstance(value, Mapping) else {}


def _base_stage_names(definition: Mapping[str, object]) -> Mapping[str, frozenset[str]]:
    universe = _mapping_items(definition.get("universe"))
    allocator = _mapping_items(definition.get("allocator"))
    timing = [
        str(_mapping_items(stage).get("name") or "")
        for stage in _sequence_items(definition.get("timing"))
    ]
    risk = [
        str(_mapping_items(stage).get("name") or "")
        for stage in _sequence_items(definition.get("risk"))
    ]
    return {
        "universe": frozenset({str(universe.get("name") or "")}),
        "allocator": frozenset({str(allocator.get("name") or "")}),
        "timing": frozenset(name for name in timing if name),
        "risk": frozenset(name for name in risk if name),
    }


def _catalog_key_for_path(path: str) -> str | None:
    """Best-effort mapping of a concrete dotted path to the catalog's key format."""
    parts = path.split(".")
    if not parts:
        return None
    head = parts[0]
    if head in {"universe", "allocator"}:
        if len(parts) == 2:
            return f"{head}.{parts[1]}"
        if len(parts) >= 3 and parts[1] == "params":
            return None  # needs the active overlay name from the strategy config
        if len(parts) >= 3:
            return f"{head}.{parts[1]}.{parts[2]}"
        return None
    if head in {"timing", "risk"}:
        if len(parts) >= 4 and parts[2] == "params":
            return f"{head}.{parts[1]}.{parts[3]}"
        if len(parts) >= 3:
            return f"{head}.{parts[1]}.{parts[2]}"
        return None
    if head == "translator":
        return f"translator.{parts[1]}" if len(parts) >= 2 else None
    if head == "drl":
        return f"allocator.drl.{parts[1]}" if len(parts) >= 2 else None
    return None


def _catalog_validation_context(
    definition: Mapping[str, object],
) -> tuple[frozenset[str] | None, frozenset[str]]:
    """``(active_parameter_keys, non_tunable_keys)`` from the strategy catalog."""
    try:
        import importlib

        module = importlib.import_module("src.strategies.parameter_catalog")
    except Exception:
        return (None, _STRUCTURAL_PARAM_NAMES)
    active_keys: frozenset[str] | None = None
    active_builder = getattr(module, "active_parameter_keys", None)
    if callable(active_builder):
        try:
            keys = active_builder(dict(definition))
        except Exception:
            keys = None
        if keys is not None:
            active_keys = frozenset(str(key) for key in keys)
    non_tunable = getattr(module, "NON_TUNABLE_PARAMETER_KEYS", None)
    if non_tunable is not None:
        try:
            return (active_keys, frozenset(str(key) for key in non_tunable))
        except TypeError:  # pragma: no cover - defensive adapter
            pass
    return (active_keys, _STRUCTURAL_PARAM_NAMES)


def fallback_legacy_reasons(
    payload: Mapping[str, object],
    base_definition: Mapping[str, object] | None,
) -> tuple[str, ...]:
    """Fallback legacy classifier used when the optimizer validator is unavailable.

    Mirrors the optimizer helper's contract: a study is legacy when it tunes a
    parameter that is inactive for its base strategy (e.g. ``equal_weight`` or
    ``mean_variance`` params on a TSMOM/DRL base), structural/data-hygiene
    (``top_n``/``sample_count``/...), or impossible for the resolved base config;
    unknown/unresolvable base strategies are legacy too.
    """
    base_id = payload.get("base_strategy_id")
    if not isinstance(base_id, str) or not base_id.strip():
        return ("missing base_strategy_id",)
    if base_definition is None:
        return (f"unknown base strategy {base_id!r}",)
    stage_names = _base_stage_names(base_definition)
    active_keys, non_tunable_keys = _catalog_validation_context(base_definition)
    structural_names = _STRUCTURAL_PARAM_NAMES | {
        key.rsplit(".", 1)[-1] for key in non_tunable_keys if "." in key
    }
    reasons: list[str] = []
    for path in _parameter_paths(payload):
        exists, _value = _path_value(base_definition, path)
        last_param = path.rsplit(".", 1)[-1]
        if exists:
            if last_param in structural_names:
                reasons.append(f"non-tunable structural parameter {path!r}")
            continue
        catalog_key = _catalog_key_for_path(path)
        if catalog_key is not None and catalog_key in non_tunable_keys:
            reasons.append(f"non-tunable structural parameter {path!r}")
            continue
        if catalog_key is not None and active_keys is not None and catalog_key in active_keys:
            continue
        if last_param in structural_names:
            reasons.append(f"non-tunable structural parameter {path!r}")
            continue
        # Inactive overlay detection: stage token present but not selected.
        parts = path.split(".")
        stage = parts[0]
        if stage in {"timing", "risk"} and len(parts) >= 4 and parts[2] == "params":
            # ``timing.<index>.params.<param>``: index exists but param absent.
            reasons.append(f"inactive or unknown parameter {path!r}")
            continue
        if stage in {"universe", "allocator"} and len(parts) >= 3:
            overlay = parts[1]
            if overlay not in stage_names.get(stage, frozenset()):
                reasons.append(f"inactive {stage} overlay {overlay!r} in {path!r}")
                continue
        if stage in {"timing", "risk"} and len(parts) == 3:
            overlay = parts[1]
            if overlay not in stage_names.get(stage, frozenset()):
                reasons.append(f"inactive {stage} overlay {overlay!r} in {path!r}")
                continue
        reasons.append(f"unknown or inactive parameter path {path!r}")
    return tuple(dict.fromkeys(reasons))


def _normalise_validation_outcome(outcome: object) -> tuple[str, ...] | None:
    """Normalise the optimizer validator's return value (``None`` = unknown shape)."""
    if outcome is None:
        return ()
    if isinstance(outcome, bool):
        return () if outcome else ("legacy study flagged by the optimizer validator",)
    if isinstance(outcome, str):
        return (outcome,)
    if isinstance(outcome, Sequence) and not isinstance(outcome, str | bytes | bytearray):
        return tuple(str(item) for item in cast(Sequence[object], outcome) if str(item))
    reasons_attr = getattr(outcome, "reasons", None)
    if reasons_attr is not None:
        return _normalise_validation_outcome(reasons_attr)
    valid_attr = getattr(outcome, "valid", None)
    if isinstance(valid_attr, bool):
        return () if valid_attr else ("legacy study flagged by the optimizer validator",)
    mapping = _mapping_items(outcome)
    if mapping:
        for key in ("reasons", "legacy_reasons", "errors"):
            if key in mapping:
                return _normalise_validation_outcome(mapping[key])
        for key in ("valid", "is_valid"):
            if isinstance(mapping.get(key), bool):
                return () if mapping[key] else ("legacy study flagged by the optimizer validator",)
        if "is_legacy" in mapping and mapping["is_legacy"] is True:
            return ("legacy study flagged by the optimizer validator",)
    return None


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
        self._quarantined: list[QuarantinedStudy] = []
        self._warnings: list[str] = []
        self._legacy_dir = self._output_dir / "legacy"
        self.reload()

    # -- introspection ---------------------------------------------------------

    @property
    def warnings(self) -> tuple[str, ...]:
        """Non-fatal study-store warnings (corrupt files, write failures)."""
        return tuple(self._warnings)

    @property
    def quarantined_count(self) -> int:
        """Number of legacy/invalid studies quarantined during the last reload."""
        return len(self._quarantined)

    @property
    def quarantined_studies(self) -> tuple[QuarantinedStudy, ...]:
        """Details (source/quarantine path + reasons) of the last startup purge."""
        return tuple(self._quarantined)

    @property
    def legacy_dir(self) -> Path:
        """Directory receiving legacy/invalid study artifacts."""
        return self._legacy_dir

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

    # -- deletion --------------------------------------------------------------

    def delete_study(self, study_id: str) -> bool:
        """Delete one study from memory + its canonical top-level artifact.

        Returns whether a persisted artifact file was removed. Raises
        :class:`OptimizationError` (404) for unknown ids and validates the id against
        traversal before touching the filesystem. Legacy quarantine files are left
        untouched (they are already archived).
        """
        self._study_path(study_id)  # validates id/path before any deletion
        study = self._studies.pop(study_id, None)
        if study is None:
            raise OptimizationError(f"unknown optimization study {study_id!r}", status_code=404)
        path = self._study_path(study_id)
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return False
        except OSError:  # pragma: no cover - best-effort artifact cleanup
            warning = f"could not delete optimization artifact {path.name}"
            logger.warning(warning)
            self._warnings.append(warning)
            return False

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
        """(Re)load persisted studies, quarantining legacy/invalid artifacts.

        Every top-level ``<optimizations>/<study_id>.json`` (canonical engine
        artifacts *and* API fixture studies) is validated against its base
        strategy's active/tunable parameter space. Invalid/legacy studies are
        atomically moved to ``<optimizations>/legacy/`` with the reasons logged and
        exposed via :attr:`quarantined_studies`; they are excluded from
        list/detail/apply. Canonical v1.5.1 artifacts are never rewritten (their
        bytes stay identical). The pass is idempotent and never raises on corrupt
        or legacy files.
        """
        self._studies.clear()
        self._quarantined.clear()
        directory = self._output_dir
        if not directory.is_dir():
            return 0
        loaded = 0
        for path in sorted(directory.glob("*.json")):
            if path.name.endswith(".legacy.json"):
                # Already-quarantined single-file variant from an older deployment.
                continue
            if path.is_dir():  # pragma: no cover - defensive: glob returns files
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                self._quarantine(path, exc)
                continue
            if not isinstance(payload, Mapping):
                self._quarantine(path, ValueError("study payload is not a JSON object"))
                continue
            raw = cast(Mapping[str, object], payload)
            candidate = dict(raw)
            if "provenance" not in candidate:
                # Optimizer-canonical artifacts carry ``request``/``seed`` and no API
                # provenance tag; the API marks them as real-engine studies.
                candidate["provenance"] = (
                    "engine" if ("request" in candidate or "seed" in candidate) else "fixture"
                )
            reasons = self._legacy_reasons(raw)
            if reasons:
                self._quarantine_legacy(path, reasons)
                continue
            try:
                study = OptimizationStudy.model_validate(candidate)
            except (ValidationError, TypeError) as exc:
                self._quarantine(path, exc)
                continue
            self._studies[study.study_id] = study
            loaded += 1
        if self._quarantined:
            logger.warning(
                "optimization startup: quarantined %d legacy study artifact(s), loaded %d "
                "valid from %s; reasons: %s",
                len(self._quarantined),
                loaded,
                directory,
                "; ".join(f"{item.study_id}: {item.reasons[0]}" for item in self._quarantined[:5]),
            )
        else:
            logger.info(
                "optimization startup: loaded %d optimization study artifact(s) from %s",
                loaded,
                directory,
            )
        return loaded

    @staticmethod
    def _is_fixture_study(payload: Mapping[str, object]) -> bool:
        """Whether a payload is an API fixture study (engine validator skips those)."""
        if "provenance" in payload:
            return True
        backend = str(payload.get("backend") or "")
        if backend and backend not in {"optuna", "deterministic"}:
            return True
        return "request" not in payload and "seed" not in payload

    def _legacy_reasons(self, payload: Mapping[str, object]) -> tuple[str, ...]:
        """Validate one persisted study against its base strategy's active space."""
        base_id = payload.get("base_strategy_id")
        base_definition: Mapping[str, object] | None = None
        base_config: StrategyConfig | None = None
        if isinstance(base_id, str) and base_id.strip():
            definition = self._studio.strategy_definition(base_id.strip())
            if definition is not None:
                base_definition = definition
            base_config = self._studio.core_config(base_id.strip())
        module = self._optimizer_module()
        validator = (
            getattr(module, "validate_persisted_optimization", None) if module is not None else None
        )
        # The optimizer helper deliberately returns [] for API fixture payloads
        # (``provenance`` / non-canonical backend / no request+seed). The API owns
        # fixture hygiene, so fixtures always use the API-side validator below.
        if callable(validator) and not self._is_fixture_study(payload):
            outcome: object = None
            attempts: tuple[tuple[tuple[object, ...], dict[str, object]], ...] = (
                ((payload,), {"config": base_config}),
                ((payload,), {}),
                ((payload, base_config), {}),
            )
            for args, kwargs in attempts:
                try:
                    outcome = validator(*args, **kwargs)
                except TypeError:
                    continue
                except Exception as exc:
                    logger.warning("optimizer legacy validator failed for %s: %s", base_id, exc)
                    outcome = None
                break
            reasons = _normalise_validation_outcome(outcome)
            if reasons is not None:
                return reasons
        return fallback_legacy_reasons(payload, base_definition)

    def _quarantine_legacy(self, path: Path, reasons: Sequence[str]) -> QuarantinedStudy:
        """Move one legacy study into ``<optimizations>/legacy/`` (atomic, unique)."""
        legacy_dir = self._legacy_dir
        target = legacy_dir / path.name
        try:
            legacy_dir.mkdir(parents=True, exist_ok=True)
            if target.exists() or target.resolve() == path.resolve():
                stamp = int(self._clock().timestamp())
                target = legacy_dir / f"{path.stem}.legacy-{stamp}{path.suffix}"
                counter = 1
                while target.exists():
                    counter += 1
                    target = legacy_dir / f"{path.stem}.legacy-{stamp}-{counter}{path.suffix}"
            path.replace(target)
        except OSError as exc:  # pragma: no cover - best-effort quarantine
            warning = f"could not quarantine legacy optimization artifact {path.name}: {exc}"
            logger.exception(warning)
            self._warnings.append(warning)
            return QuarantinedStudy(
                study_id=path.stem,
                source_path=path,
                quarantine_path=path,
                reasons=tuple(reasons),
            )
        study_id = path.stem
        item = QuarantinedStudy(
            study_id=study_id,
            source_path=path,
            quarantine_path=target,
            reasons=tuple(reasons),
        )
        warning = (
            f"legacy optimization artifact {path.name} quarantined to {target.name}: "
            + "; ".join(reasons[:3])
        )
        logger.warning(warning)
        self._warnings.append(warning)
        self._quarantined.append(item)
        return item

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
            tuned = apply_dotted_params(base, best.params, allow_new_params=True)
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
        """Apply best params as a new visible strategy and verify it end-to-end.

        The tuned strategy is persisted (committed ``configs/strategies`` in
        development/paper/live, ``artifacts/strategies`` overlay in test/backtest)
        and the in-memory catalog is refreshed before the verification run, so the
        new id is immediately visible via ``GET /api/strategies`` and runnable via
        ``POST /api/backtests``. A walk-forward verification backtest is always
        triggered when a runner is available (the request flag is retained only for
        backwards compatibility) and its persisted run id is returned.
        """
        resolved_request = request or OptimizationApplyRequest()
        study = self.get_study(study_id)
        if not study.best_params:
            raise OptimizationError(
                f"optimization study {study_id!r} has no best parameters", status_code=409
            )
        try:
            saved = self._studio.save_best_params(
                study.base_strategy_id,
                study.best_params,
                strategy_id=resolved_request.new_strategy_id,
            )
        except StrategyStudioError as exc:
            raise OptimizationError(str(exc), status_code=exc.status_code) from exc
        view, config = saved.view, saved.config
        verification: Mapping[str, object] | None = None
        message = (
            f"applied {len(study.best_params)} best parameter(s) from {study.study_id} "
            f"to new strategy {config.id!r}"
        )
        runner = backtest_runner or self._backtest_runner
        if runner is None:
            verification = {
                "status": "skipped",
                "message": "verification backtest runner unavailable",
            }
            message += "; verification walk-forward backtest skipped (no runner)"
        else:
            try:
                outcome = runner(resolved_request, study, view)
                if outcome is None:
                    verification = {
                        "status": "skipped",
                        "message": "verification backtest runner returned no result",
                    }
                    message += "; verification walk-forward backtest skipped (no result)"
                else:
                    verification = dict(outcome)
                    status = str(verification.get("status") or "started")
                    run_id = _verification_run_id(verification)
                    message += f"; verification walk-forward backtest {status}"
                    if run_id:
                        message += f" (run {run_id})"
            except Exception as exc:
                verification = {"error": f"{type(exc).__name__}: {exc}"}
                message += "; verification walk-forward backtest failed"
        return OptimizationApplyResponse(
            study_id=study.study_id,
            strategy=view,
            best_params=dict(study.best_params),
            verification_backtest=dict(verification) if verification is not None else None,
            message=message,
            saved_strategy_id=config.id,
            strategy_path=str(saved.path),
            backtest_run_id=_verification_run_id(verification),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _verification_run_id(verification: Mapping[str, object] | None) -> str | None:
    """Extract the persisted backtest run id from a verification-runner payload."""
    if verification is None:
        return None
    candidates: list[object] = []
    run = verification.get("run")
    if isinstance(run, Mapping):
        run_mapping = cast(Mapping[str, object], run)
        candidates.extend(
            [run_mapping.get("id"), run_mapping.get("run_id"), run_mapping.get("backtest_run_id")]
        )
    candidates.extend([verification.get("backtest_run_id"), verification.get("run_id")])
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


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
