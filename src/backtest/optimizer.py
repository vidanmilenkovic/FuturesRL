"""Walk-forward strategy optimizer with Optuna and a deterministic fallback.

``WalkForwardOptimizer`` tunes a declarative :class:`~configs.strategy.StrategyConfig`
against **out-of-sample walk-forward folds only**: every trial is evaluated through
the existing public :func:`src.backtest.engine.run_walk_forward` (purged/embargoed
folds), never on the in-sample concatenated window. The objective is the mean
out-of-sample metric across folds, optionally reduced by a drawdown penalty when
the OOS max drawdown exceeds ``drawdown_penalty_threshold``.

Two optimisation backends are supported:

``optuna``
    TPE sampler (``optuna.samplers.TPESampler``) when Optuna is installed. The
    import is lazy: importing this module never imports Optuna.
``deterministic``
    A built-in Latin-Hypercube design with seeded stratified permutations. It is
    offline, reproducible and used automatically when Optuna is unavailable
    (e.g. CI / the quality gate), so the optimizer remains fully testable.

Studies are persisted atomically to
``<settings.optimization_artifacts_dir>/<study_id>.json`` (``artifacts/optimizations``
by default) and can be reloaded by the API through :func:`load_optimization`.

CLI
---
``python -m src.backtest.optimizer --strategy configs/strategies/tsmom_vol_target.yaml \
    --trials 20 --objective sharpe --offline --output artifacts/opt_tsmom.json``
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
from configs.instruments import InstrumentsConfig, load_instruments
from configs.strategy import StrategyConfig, load_strategy_config
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from src.backtest.engine import (
    BacktestMode,
    WalkForwardReport,
    build_states_from_bars,
    run_from_config,
    run_walk_forward,
)
from src.core.types import MarketState
from src.data.schema import ensure_bars_schema

#: Scalar parameter value that can be embedded in a strategy config / JSON.
type ParamValue = int | float | str | bool

#: Supported optimisation objectives (all are "higher is better").
type ObjectiveMetric = Literal["sharpe", "sortino", "calmar", "cagr", "rachev", "return_on_margin"]

#: All supported objective metric names, in display order.
OBJECTIVE_METRICS: tuple[str, ...] = (
    "sharpe",
    "sortino",
    "calmar",
    "cagr",
    "rachev",
    "return_on_margin",
)

#: Penalised objective returned to Optuna for a trial that could not be evaluated.
_FAILED_OBJECTIVE: float = -1.0e12

#: Default number of optimisation trials.
DEFAULT_TRIAL_COUNT: int = 25

#: Schema version of the persisted optimisation artifact / API payload.
OPTIMIZATION_SCHEMA_VERSION: int = 1


class OptimizationBackend(StrEnum):
    """Optimisation backend actually used for a study."""

    OPTUNA = "optuna"
    DETERMINISTIC = "deterministic"


class TrialState(StrEnum):
    """Outcome of a single optimisation trial."""

    COMPLETE = "COMPLETE"
    FAIL = "FAIL"


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class ParameterRange(BaseModel):
    """A single tunable strategy parameter inside the declared search space.

    ``param_path`` is a dotted path into the serialised ``StrategyConfig``:
    ``allocator.params.target_annual_volatility``,
    ``timing.0.params.lookback_fast`` (list index) or
    ``timing.tsmom.params.lookback_fast`` (stage name),
    ``risk.1.params.max_single_instrument_weight``,
    ``translator.max_margin_utilization``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    param_path: str = Field(min_length=1)
    type: Literal["int", "float", "categorical"]
    low: float | None = None
    high: float | None = None
    step: float | None = None
    choices: tuple[ParamValue, ...] | None = None

    @field_validator("param_path")
    @classmethod
    def _clean_param_path(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("param_path must be non-empty")
        parts = cleaned.split(".")
        if any(not part.strip() for part in parts):
            raise ValueError(f"param_path {value!r} must not contain empty segments")
        return cleaned

    @model_validator(mode="after")
    def _validate_shape(self) -> ParameterRange:
        if self.type == "categorical":
            if self.choices is None or len(self.choices) == 0:
                raise ValueError("categorical parameters require a non-empty 'choices'")
            if len(set(self.choices)) != len(self.choices):
                raise ValueError(f"duplicate categorical choices for {self.param_path!r}")
            if self.low is not None or self.high is not None or self.step is not None:
                raise ValueError("categorical parameters must not define low/high/step")
            return self
        if self.choices is not None:
            raise ValueError("numeric parameters must not define 'choices'")
        if self.low is None or self.high is None:
            raise ValueError(f"numeric parameter {self.param_path!r} requires low and high")
        for name in ("low", "high"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{self.param_path!r}: {name} must be finite")
        if float(self.low) >= float(self.high):
            raise ValueError(f"{self.param_path!r}: low must be strictly below high")
        if self.step is not None:
            if not math.isfinite(float(self.step)) or float(self.step) <= 0.0:
                raise ValueError(f"{self.param_path!r}: step must be finite and positive")
            if self.type == "int" and float(self.step) != int(float(self.step)):
                raise ValueError(f"{self.param_path!r}: int step must be a whole number")
        if self.type == "int":
            for name in ("low", "high"):
                value = float(getattr(self, name))
                if value != int(value):
                    raise ValueError(f"{self.param_path!r}: int bounds must be whole numbers")
        return self

    def numeric_bounds(self) -> tuple[float, float]:
        """Validated ``(low, high)`` bounds for a numeric parameter."""
        if self.low is None or self.high is None:
            raise ValueError(f"{self.param_path!r} is not a numeric parameter")
        return float(self.low), float(self.high)


class OptimizationRequest(BaseModel):
    """Declarative optimisation request (API + CLI contract)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    base_strategy_id: str = Field(min_length=1)
    objective_metric: ObjectiveMetric = "sharpe"
    n_trials: int = Field(default=DEFAULT_TRIAL_COUNT, ge=2, le=200)
    walk_forward: bool = True
    wf_splits: int = Field(default=4, ge=2, le=20)
    purge_bars: int = Field(default=5, ge=0)
    embargo_bars: int = Field(default=5, ge=0)
    drawdown_penalty_threshold: float = Field(default=0.20, ge=0.0, le=1.0)
    search_space: tuple[ParameterRange, ...] = Field(min_length=1)

    @field_validator("base_strategy_id")
    @classmethod
    def _clean_base_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("base_strategy_id must be non-empty")
        return cleaned

    @model_validator(mode="after")
    def _unique_param_paths(self) -> OptimizationRequest:
        paths = [parameter.param_path for parameter in self.search_space]
        if len(set(paths)) != len(paths):
            raise ValueError("search_space contains duplicate param_path entries")
        return self


class TrialSummary(BaseModel):
    """Result of one optimisation trial (OOS metrics + timing + params)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trial_number: int = Field(ge=0)
    params: dict[str, ParamValue] = Field(default_factory=dict)
    state: TrialState = TrialState.COMPLETE
    objective_value: float = 0.0
    raw_objective: float = 0.0
    oos_sharpe: float = 0.0
    oos_sortino: float = 0.0
    oos_calmar: float = 0.0
    oos_cagr: float = 0.0
    oos_rachev: float = 0.0
    oos_return_on_margin: float = 0.0
    max_dd: float = 0.0
    fold_count: int = Field(default=0, ge=0)
    duration_ms: float = Field(default=0.0, ge=0.0)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable trial payload."""
        return self.model_dump(mode="json")


class OptimizationResult(BaseModel):
    """Persisted, JSON-serialisable result of one walk-forward optimisation study."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = OPTIMIZATION_SCHEMA_VERSION
    study_id: str
    base_strategy_id: str
    objective_metric: ObjectiveMetric
    direction: Literal["maximize"] = "maximize"
    n_trials: int
    completed_trials: int
    failed_trials: int
    best_trial_number: int | None
    best_params: dict[str, ParamValue]
    best_value: float | None
    best_raw_value: float | None
    best_strategy_config: dict[str, Any]
    trials: tuple[TrialSummary, ...]
    param_importances: dict[str, float]
    drawdown_penalty_threshold: float
    walk_forward: bool
    fold_count: int
    backend: OptimizationBackend
    seed: int
    created_at: datetime
    duration_ms: float
    artifact_path: str | None = None
    request: OptimizationRequest

    @field_validator("created_at")
    @classmethod
    def _aware_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware UTC")
        return value.astimezone(UTC)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable study payload (API / CLI contract)."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> OptimizationResult:
        """Validate a persisted study payload back into a typed result."""
        return cls.model_validate(dict(payload))


# ---------------------------------------------------------------------------
# Dotted-path parameter application
# ---------------------------------------------------------------------------


def _split_param_path(param_path: str) -> list[str]:
    parts = [part.strip() for part in param_path.split(".")]
    if any(not part for part in parts):
        raise ValueError(f"invalid parameter path {param_path!r}")
    return parts


def _descend(cursor: Any, part: str, param_path: str) -> Any:
    if isinstance(cursor, Mapping):
        if part not in cursor:
            raise ValueError(f"unknown parameter path {param_path!r}: missing {part!r}")
        return cursor[part]
    if isinstance(cursor, (list, tuple)):
        if part.isdigit():
            index = int(part)
            if not 0 <= index < len(cursor):
                raise ValueError(
                    f"unknown parameter path {param_path!r}: index {index} out of range"
                )
            return cursor[index]
        matches = [
            item for item in cursor if isinstance(item, Mapping) and item.get("name") == part
        ]
        if len(matches) != 1:
            raise ValueError(f"unknown parameter path {param_path!r}: stage {part!r} is not unique")
        return matches[0]
    raise ValueError(f"unknown parameter path {param_path!r}: {part!r} is not a container")


def _set_leaf(cursor: Any, part: str, value: ParamValue, param_path: str) -> None:
    if not isinstance(cursor, dict):
        raise ValueError(f"unknown parameter path {param_path!r}: {part!r} is not a mapping")
    if part not in cursor:
        raise ValueError(f"unknown parameter path {param_path!r}: missing {part!r}")
    cursor[part] = value


def _get_dotted(payload: Mapping[str, Any], param_path: str) -> Any:
    cursor: Any = payload
    for part in _split_param_path(param_path):
        cursor = _descend(cursor, part, param_path)
    return cursor


def _set_dotted(payload: dict[str, Any], param_path: str, value: ParamValue) -> None:
    parts = _split_param_path(param_path)
    cursor: Any = payload
    for part in parts[:-1]:
        cursor = _descend(cursor, part, param_path)
    _set_leaf(cursor, parts[-1], value, param_path)


def apply_parameters(config: StrategyConfig, params: Mapping[str, ParamValue]) -> StrategyConfig:
    """Apply sampled parameters to a copy of ``config`` by dotted path.

    The modified payload is re-validated through :class:`StrategyConfig`, so an
    invalid combination raises ``ValueError`` instead of producing a broken
    strategy.
    """
    raw: dict[str, Any] = config.model_dump(mode="python")
    for param_path, value in params.items():
        _set_dotted(raw, param_path, value)
    try:
        return StrategyConfig.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(
            f"sampled parameters {dict(params)!r} produced an invalid StrategyConfig: {exc}"
        ) from exc


def _validate_search_space(config: StrategyConfig, search_space: Sequence[ParameterRange]) -> None:
    """Fail fast when a search-space path does not exist or has the wrong type."""
    raw = config.model_dump(mode="python")
    for parameter in search_space:
        existing = _get_dotted(raw, parameter.param_path)
        if parameter.type == "categorical":
            if isinstance(existing, (dict, list)) or existing is None:
                raise ValueError(
                    f"categorical parameter {parameter.param_path!r} does not address a "
                    "scalar config value"
                )
        elif isinstance(existing, bool) or not isinstance(existing, (int, float)):
            raise ValueError(
                f"numeric parameter {parameter.param_path!r} does not address a "
                f"numeric config value (found {type(existing).__name__})"
            )


# ---------------------------------------------------------------------------
# Deterministic Latin-Hypercube fallback sampler
# ---------------------------------------------------------------------------


def _numeric_sample(parameter: ParameterRange, unit: float) -> ParamValue:
    low, high = parameter.numeric_bounds()
    if parameter.step is not None:
        step = float(parameter.step)
        levels = max(1, math.floor((high - low) / step + 1.0e-9) + 1)
        level = min(levels - 1, math.floor(unit * levels))
        value = low + level * step
    else:
        value = low + unit * (high - low)
    clipped = min(max(value, low), high)
    if parameter.type == "int":
        return round(clipped)
    return float(round(clipped, 12))


def deterministic_lhs_parameter_sets(
    search_space: Sequence[ParameterRange],
    n_trials: int,
    *,
    seed: int = 7,
) -> tuple[dict[str, ParamValue], ...]:
    """Deterministic Latin-Hypercube design over the declared search space.

    Each numeric dimension is stratified into ``n_trials`` equal-probability bins
    with a seeded permutation; categorical dimensions use a seeded permutation of
    a balanced choice cycle. The result depends only on ``(search_space, n_trials,
    seed)``, never on wall-clock time or random OS state.
    """
    if isinstance(n_trials, bool) or not isinstance(n_trials, int) or n_trials < 1:
        raise ValueError("n_trials must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not search_space:
        raise ValueError("search_space must not be empty")
    rng = np.random.default_rng(seed)
    index_columns: dict[str, np.ndarray[Any, np.dtype[np.int64]]] = {}
    numeric_columns: dict[str, np.ndarray[Any, np.dtype[np.float64]]] = {}
    for parameter in search_space:
        if parameter.type == "categorical":
            choices = parameter.choices
            if choices is None or len(choices) == 0:
                raise ValueError(f"categorical parameter {parameter.param_path!r} has no choices")
            base = np.arange(n_trials, dtype=np.int64) % len(choices)
            permutation = rng.permutation(n_trials).astype(np.int64)
            index_columns[parameter.param_path] = base[permutation]
        else:
            strata = np.arange(n_trials, dtype=np.float64)
            permutation = rng.permutation(n_trials)
            numeric_columns[parameter.param_path] = (strata[permutation] + 0.5) / n_trials
    parameter_sets: list[dict[str, ParamValue]] = []
    for trial_index in range(n_trials):
        params: dict[str, ParamValue] = {}
        for parameter in search_space:
            choices = parameter.choices
            if parameter.type == "categorical" and choices is not None:
                position = int(index_columns[parameter.param_path][trial_index])
                params[parameter.param_path] = choices[position]
            else:
                unit = float(numeric_columns[parameter.param_path][trial_index])
                params[parameter.param_path] = _numeric_sample(parameter, unit)
        parameter_sets.append(params)
    return tuple(parameter_sets)


def _encode_parameter(value: ParamValue, parameter: ParameterRange) -> float:
    if parameter.type == "categorical":
        choices = parameter.choices or ()
        try:
            return float(choices.index(value))
        except ValueError:
            return -1.0
    return float(value)


def deterministic_param_importances(
    search_space: Sequence[ParameterRange],
    trials: Sequence[TrialSummary],
) -> dict[str, float]:
    """Deterministic parameter-importance estimate (|Pearson correlation|, normalised)."""
    completed = [trial for trial in trials if trial.state is TrialState.COMPLETE]
    paths = [parameter.param_path for parameter in search_space]
    if len(completed) < 2:
        return dict.fromkeys(paths, 0.0)
    objectives = np.array([trial.objective_value for trial in completed], dtype=np.float64)
    if float(np.std(objectives)) == 0.0:
        return dict.fromkeys(paths, 0.0)
    raw_importances: dict[str, float] = {}
    for parameter in search_space:
        encoded = np.array(
            [
                _encode_parameter(trial.params.get(parameter.param_path, 0), parameter)
                for trial in completed
            ],
            dtype=np.float64,
        )
        if float(np.std(encoded)) == 0.0:
            raw_importances[parameter.param_path] = 0.0
            continue
        correlation = np.corrcoef(encoded, objectives)[0, 1]
        raw_importances[parameter.param_path] = (
            float(abs(correlation)) if math.isfinite(float(correlation)) else 0.0
        )
    total = sum(raw_importances.values())
    if total <= 0.0:
        return raw_importances
    return {path: value / total for path, value in raw_importances.items()}


# ---------------------------------------------------------------------------
# Optimization helpers
# ---------------------------------------------------------------------------


def _load_optuna_module() -> Any | None:
    """Lazily import Optuna; ``None`` when the optional dependency is absent."""
    try:
        return importlib.import_module("optuna")
    except ImportError:  # pragma: no cover - exercised when Optuna is installed
        return None


def _mean_aggregate_metric(report: WalkForwardReport, metric: str) -> float:
    """Read ``report.aggregate[metric]["mean"]`` defensively."""
    aggregate: Any = report.aggregate
    if not isinstance(aggregate, Mapping):
        return 0.0
    stats = aggregate.get(metric)
    if not isinstance(stats, Mapping):
        return 0.0
    mean = stats.get("mean")
    return float(mean) if isinstance(mean, (int, float)) else 0.0


def _mean_fold_metric(report: WalkForwardReport, metric: str) -> float:
    """Mean of a per-fold metric (used for metrics not aggregated by the engine)."""
    values: list[float] = []
    for fold in report.folds:
        metrics: Any = fold.metrics
        if isinstance(metrics, Mapping):
            value = metrics.get(metric)
            if isinstance(value, (int, float)):
                values.append(float(value))
    if not values:
        return 0.0
    return float(np.mean(np.array(values, dtype=np.float64)))


def _extract_oos_metrics(report: WalkForwardReport) -> dict[str, float]:
    """Map a walk-forward report to the six supported OOS objective metrics."""
    return {
        "sharpe": _mean_aggregate_metric(report, "sharpe"),
        "sortino": _mean_aggregate_metric(report, "sortino"),
        "calmar": _mean_aggregate_metric(report, "calmar"),
        "cagr": _mean_aggregate_metric(report, "cagr"),
        "rachev": _mean_fold_metric(report, "rachev_ratio_05"),
        "return_on_margin": _mean_aggregate_metric(report, "return_on_max_initial_margin"),
        "max_drawdown": _mean_aggregate_metric(report, "max_drawdown"),
    }


def _sanitize_study_id(study_id: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in "._-" else "_" for character in study_id
    )
    cleaned = cleaned.strip("._-")
    return cleaned or "optimization"


def _default_study_id(request: OptimizationRequest, seed: int) -> str:
    return _sanitize_study_id(
        f"{request.base_strategy_id}-{request.objective_metric}-{request.n_trials}t-s{seed}"
    )


# ---------------------------------------------------------------------------
# Persistence (atomic JSON artifacts)
# ---------------------------------------------------------------------------


def _default_output_dir() -> Path:
    from configs.settings import get_settings

    return get_settings().optimization_artifacts_dir


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    """Atomically write ``payload`` to ``path`` (temp file + ``os.replace``)."""
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{resolved.name}.", suffix=".tmp", dir=str(resolved.parent)
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(resolved)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return resolved


def write_optimization_artifact(result: OptimizationResult, path: Path | str) -> Path:
    """Atomically persist a study result to ``path`` and return the written path."""
    return _atomic_write_json(Path(path), result.to_dict())


def load_optimization(
    path_or_study_id: Path | str,
    *,
    output_dir: Path | str | None = None,
) -> OptimizationResult:
    """Load a persisted study by path or by ``study_id`` from the artifact directory."""
    candidate = Path(path_or_study_id)
    if not candidate.is_file():
        base = Path(output_dir) if output_dir is not None else _default_output_dir()
        candidate = base / f"{path_or_study_id}.json"
    if not candidate.is_file():
        raise FileNotFoundError(f"optimization artifact not found: {candidate}")
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid optimization artifact JSON at {candidate}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"optimization artifact at {candidate} must be a JSON object")
    try:
        result = OptimizationResult.from_dict(payload)
    except ValidationError as exc:
        raise ValueError(f"invalid optimization artifact at {candidate}: {exc}") from exc
    if result.artifact_path is None:
        result = result.model_copy(update={"artifact_path": str(candidate)})
    return result


# ---------------------------------------------------------------------------
# Default search space
# ---------------------------------------------------------------------------


def default_search_space(config: StrategyConfig) -> tuple[ParameterRange, ...]:
    """Derive a sensible, config-valid search space for the Studio/CLI optimizer."""
    ranges: list[ParameterRange] = []

    def add_allocator(name: str, low: float, high: float, step: float) -> None:
        if name in config.allocator.params:
            ranges.append(
                ParameterRange(
                    param_path=f"allocator.params.{name}",
                    type="float",
                    low=low,
                    high=high,
                    step=step,
                )
            )

    add_allocator("target_annual_volatility", 0.05, 0.25, 0.01)
    timing_ranges: dict[str, tuple[int, int, int]] = {
        "lookback_fast": (10, 40, 5),
        "lookback_slow": (60, 200, 20),
        "vol_lookback": (10, 60, 5),
        "window": (5, 30, 5),
        "fast_period": (2, 10, 1),
        "slow_period": (15, 60, 5),
    }
    if config.timing:
        first_timing = config.timing[0]
        for name, (low, high, step) in timing_ranges.items():
            if name in first_timing.params:
                ranges.append(
                    ParameterRange(
                        param_path=f"timing.0.params.{name}",
                        type="int",
                        low=low,
                        high=high,
                        step=step,
                    )
                )
    float_risk_ranges: dict[str, tuple[float, float, float]] = {
        "max_single_instrument_weight": (0.10, 0.50, 0.05),
        "max_sector_weight": (0.20, 0.60, 0.05),
        "vol_target": (0.05, 0.25, 0.01),
    }
    for index, stage in enumerate(config.risk):
        for name, (low_f, high_f, step_f) in float_risk_ranges.items():
            if name in stage.params:
                ranges.append(
                    ParameterRange(
                        param_path=f"risk.{index}.params.{name}",
                        type="float",
                        low=low_f,
                        high=high_f,
                        step=step_f,
                    )
                )
    ranges.append(
        ParameterRange(
            param_path="translator.max_margin_utilization",
            type="float",
            low=0.40,
            high=0.80,
            step=0.05,
        )
    )
    if not ranges:
        raise ValueError(
            f"cannot derive a default search space for strategy {config.id!r}: "
            "declare explicit ParameterRange entries"
        )
    return tuple(ranges)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


class WalkForwardOptimizer:
    """Optimise a declarative strategy strictly on OOS walk-forward folds.

    Parameters
    ----------
    config:
        Base declarative strategy config (loaded from ``configs/strategies``).
    request:
        Validated optimisation request (objective, trial count, walk-forward
        guardrails, search space).
    instrument_master, bars, states:
        Optional data window. When ``bars``/``states`` are omitted the optimizer
        uses the deterministic offline bar source and caches one point-in-time
        state path for the whole study (so state construction cost is paid once,
        not once per trial).
    seed:
        Seed for the Optuna TPE sampler / deterministic Latin-Hypercube design.
    use_optuna:
        ``None`` auto-detects Optuna, ``False`` forces the deterministic fallback
        (offline/CI), ``True`` requires Optuna (raises if it is not installed).
    output_dir:
        Directory for persisted studies; defaults to
        ``settings.optimization_artifacts_dir``.
    persist:
        Whether :meth:`optimize` writes the study JSON artifact.
    mode:
        Backtest mode for fold evaluation (default vectorised for fast sweeps).
    offline:
        Allow the CLI's deterministic offline pipeline/translator fallbacks.
    pipeline, translator:
        Optional injected FinRL-X pipeline / contract translator.
    backtest_runner:
        Optional injected walk-forward runner (must return a
        :class:`~src.backtest.engine.WalkForwardReport`). Defaults to the public
        :func:`src.backtest.engine.run_walk_forward`; tests use this to spy on
        OOS-only evaluation.
    drawdown_penalty_coefficient:
        Objective reduction per unit of OOS drawdown above the threshold.
    study_id:
        Optional explicit study identifier (default: deterministic request hash).
    """

    def __init__(
        self,
        config: StrategyConfig,
        *,
        request: OptimizationRequest,
        instrument_master: InstrumentsConfig | None = None,
        bars: pl.DataFrame | None = None,
        states: Sequence[MarketState] | None = None,
        seed: int = 7,
        use_optuna: bool | None = None,
        output_dir: Path | str | None = None,
        persist: bool = True,
        mode: BacktestMode | None = None,
        offline: bool = False,
        pipeline: Any | None = None,
        translator: Any | None = None,
        backtest_runner: Callable[..., WalkForwardReport] | None = None,
        drawdown_penalty_coefficient: float = 1.0,
        study_id: str | None = None,
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        if not math.isfinite(drawdown_penalty_coefficient) or drawdown_penalty_coefficient < 0.0:
            raise ValueError("drawdown_penalty_coefficient must be finite and non-negative")
        if not isinstance(persist, bool):
            raise TypeError("persist must be a bool")
        if mode is not None and not isinstance(mode, BacktestMode):
            raise TypeError("mode must be a BacktestMode or None")
        if not isinstance(offline, bool):
            raise TypeError("offline must be a bool")
        if config.id != request.base_strategy_id:
            raise ValueError(
                f"request base_strategy_id {request.base_strategy_id!r} does not match "
                f"config id {config.id!r}"
            )
        _validate_search_space(config, request.search_space)
        optuna_module = _load_optuna_module()
        if use_optuna is True and optuna_module is None:
            raise RuntimeError(
                "Optuna is not installed; install the 'optuna' dependency or let "
                "use_optuna=None select the deterministic fallback"
            )
        resolved_use_optuna = (optuna_module is not None) if use_optuna is None else use_optuna
        self._config = config
        self._request = request
        self._instrument_master = (
            instrument_master if instrument_master is not None else load_instruments()
        )
        self._bars = bars
        self._states = list(states) if states is not None else None
        self._seed = seed
        self._optuna = optuna_module if resolved_use_optuna else None
        self._backend = (
            OptimizationBackend.OPTUNA if resolved_use_optuna else OptimizationBackend.DETERMINISTIC
        )
        self._output_dir = Path(output_dir) if output_dir is not None else _default_output_dir()
        self._persist = persist
        self._mode = mode if mode is not None else BacktestMode.VECTORIZED
        self._offline = offline
        self._pipeline = pipeline
        self._translator = translator
        self._backtest_runner = backtest_runner
        self._penalty_coefficient = float(drawdown_penalty_coefficient)
        self._study_id = study_id if study_id is not None else _default_study_id(request, seed)
        self._window_bars: pl.DataFrame | None = None
        self._window_states: list[MarketState] | None = None

    # -- public properties -----------------------------------------------------

    @property
    def study_id(self) -> str:
        """Identifier under which the study is persisted."""
        return self._study_id

    @property
    def backend(self) -> OptimizationBackend:
        """Optimisation backend selected for this study."""
        return self._backend

    @property
    def request(self) -> OptimizationRequest:
        """Declared optimisation request."""
        return self._request

    @property
    def config(self) -> StrategyConfig:
        """Base strategy config being optimised."""
        return self._config

    def _initial_equity(self) -> float:
        spec = self._config.backtest
        return spec.initial_equity if spec is not None else 1_000_000.0

    # -- optimisation loop -----------------------------------------------------

    def optimize(
        self,
        *,
        backtest_runner: Callable[..., WalkForwardReport] | None = None,
    ) -> OptimizationResult:
        """Run the study and return the persisted, typed result.

        ``backtest_runner`` optionally overrides the injected walk-forward runner
        for this study only (the API/tests can pass a spy); it must return a
        :class:`~src.backtest.engine.WalkForwardReport`.
        """
        if backtest_runner is not None and not callable(backtest_runner):
            raise TypeError("backtest_runner must be callable or None")
        previous_runner = self._backtest_runner
        if backtest_runner is not None:
            self._backtest_runner = backtest_runner
        try:
            return self._optimize_impl()
        finally:
            self._backtest_runner = previous_runner

    def _optimize_impl(self) -> OptimizationResult:
        started = time.perf_counter()
        self._window_bars, self._window_states = self._prepare_window()
        optuna_study: Any | None = None
        if self._backend is OptimizationBackend.OPTUNA:
            trials, optuna_study = self._run_optuna()
        else:
            trials = self._run_deterministic()
        completed = [trial for trial in trials if trial.state is TrialState.COMPLETE]
        if not completed:
            first_error = next(
                (trial.error for trial in trials if trial.error is not None), "unknown error"
            )
            raise RuntimeError(f"all {len(trials)} optimization trials failed: {first_error}")
        best = max(completed, key=lambda trial: (trial.objective_value, -trial.trial_number))
        best_config = apply_parameters(self._config, best.params)
        importances = self._param_importances(trials, optuna_study)
        duration_ms = (time.perf_counter() - started) * 1000.0
        artifact_path = str(self._output_dir / f"{self._study_id}.json") if self._persist else None
        result = OptimizationResult(
            study_id=self._study_id,
            base_strategy_id=self._request.base_strategy_id,
            objective_metric=self._request.objective_metric,
            n_trials=self._request.n_trials,
            completed_trials=len(completed),
            failed_trials=len(trials) - len(completed),
            best_trial_number=best.trial_number,
            best_params=dict(best.params),
            best_value=best.objective_value,
            best_raw_value=best.raw_objective,
            best_strategy_config=best_config.model_dump(mode="json"),
            trials=tuple(trials),
            param_importances=importances,
            drawdown_penalty_threshold=self._request.drawdown_penalty_threshold,
            walk_forward=self._request.walk_forward,
            fold_count=best.fold_count,
            backend=self._backend,
            seed=self._seed,
            created_at=datetime.now(UTC),
            duration_ms=duration_ms,
            artifact_path=artifact_path,
            request=self._request,
        )
        if artifact_path is not None:
            write_optimization_artifact(result, artifact_path)
        return result

    def _prepare_window(self) -> tuple[pl.DataFrame | None, list[MarketState] | None]:
        """Resolve/cache the data window and point-in-time states once per study."""
        bars = self._bars
        states = list(self._states) if self._states is not None else None
        if bars is None and states is None:
            from src.backtest import engine as engine_module

            bars = engine_module._offline_bars(self._config, self._instrument_master)
        if states is None and bars is not None:
            states = build_states_from_bars(
                bars,
                self._instrument_master,
                initial_equity=self._initial_equity(),
            )
        return bars, states

    def _run_deterministic(self) -> list[TrialSummary]:
        parameter_sets = deterministic_lhs_parameter_sets(
            self._request.search_space, self._request.n_trials, seed=self._seed
        )
        return [
            self._evaluate_params(params, trial_index)
            for trial_index, params in enumerate(parameter_sets)
        ]

    def _run_optuna(self) -> tuple[list[TrialSummary], Any]:
        optuna = self._optuna
        if optuna is None:  # pragma: no cover - constructor guarantees it
            raise RuntimeError("Optuna backend selected but optuna is not available")
        sampler = optuna.samplers.TPESampler(seed=self._seed)
        study = optuna.create_study(
            direction="maximize", sampler=sampler, study_name=self._study_id
        )
        summaries: list[TrialSummary] = []

        def objective(trial: Any) -> float:
            params = self._sample_optuna_parameters(trial)
            summary = self._evaluate_params(params, int(trial.number))
            summaries.append(summary)
            if summary.state is TrialState.FAIL:
                return _FAILED_OBJECTIVE
            return summary.objective_value

        study.optimize(objective, n_trials=self._request.n_trials)
        return summaries, study

    def _sample_optuna_parameters(self, trial: Any) -> dict[str, ParamValue]:
        params: dict[str, ParamValue] = {}
        for parameter in self._request.search_space:
            if parameter.type == "categorical":
                choices = list(parameter.choices or ())
                params[parameter.param_path] = trial.suggest_categorical(
                    parameter.param_path, choices
                )
            elif parameter.type == "int":
                low, high = parameter.numeric_bounds()
                step = int(float(parameter.step)) if parameter.step is not None else 1
                params[parameter.param_path] = int(
                    trial.suggest_int(parameter.param_path, int(low), int(high), step=step)
                )
            else:
                low, high = parameter.numeric_bounds()
                if parameter.step is not None:
                    params[parameter.param_path] = float(
                        trial.suggest_float(
                            parameter.param_path, low, high, step=float(parameter.step)
                        )
                    )
                else:
                    params[parameter.param_path] = float(
                        trial.suggest_float(parameter.param_path, low, high)
                    )
        return params

    def _evaluate_params(self, params: Mapping[str, ParamValue], trial_number: int) -> TrialSummary:
        started = time.perf_counter()
        try:
            trial_config = apply_parameters(self._config, params)
            metrics, fold_count = self._evaluate_config(trial_config)
        except Exception as exc:
            return TrialSummary(
                trial_number=trial_number,
                params=dict(params),
                state=TrialState.FAIL,
                objective_value=_FAILED_OBJECTIVE,
                duration_ms=(time.perf_counter() - started) * 1000.0,
                error=f"{type(exc).__name__}: {exc}",
            )
        raw_objective = metrics[self._request.objective_metric]
        penalty = (
            max(
                0.0,
                metrics["max_drawdown"] - self._request.drawdown_penalty_threshold,
            )
            * self._penalty_coefficient
        )
        return TrialSummary(
            trial_number=trial_number,
            params=dict(params),
            state=TrialState.COMPLETE,
            objective_value=raw_objective - penalty,
            raw_objective=raw_objective,
            oos_sharpe=metrics["sharpe"],
            oos_sortino=metrics["sortino"],
            oos_calmar=metrics["calmar"],
            oos_cagr=metrics["cagr"],
            oos_rachev=metrics["rachev"],
            oos_return_on_margin=metrics["return_on_margin"],
            max_dd=metrics["max_drawdown"],
            fold_count=fold_count,
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

    def _evaluate_config(self, config: StrategyConfig) -> tuple[dict[str, float], int]:
        if self._request.walk_forward:
            return self._evaluate_walk_forward(config)
        return self._evaluate_in_sample(config)

    def _evaluate_walk_forward(self, config: StrategyConfig) -> tuple[dict[str, float], int]:
        """Score a trial ONLY on purged/embargoed out-of-sample folds."""
        runner = self._backtest_runner if self._backtest_runner is not None else run_walk_forward
        report = runner(
            config,
            bars=self._window_bars,
            states=self._window_states,
            splits=self._request.wf_splits,
            purge_bars=self._request.purge_bars,
            embargo_bars=self._request.embargo_bars,
            initial_equity=self._initial_equity(),
            mode=self._mode,
            instrument_master=self._instrument_master,
            pipeline=self._pipeline,
            translator=self._translator,
            offline=self._offline,
        )
        return _extract_oos_metrics(report), report.fold_count

    def _evaluate_in_sample(self, config: StrategyConfig) -> tuple[dict[str, float], int]:
        """Opt-in in-sample evaluation (``walk_forward=False``; never the default)."""
        from src.analytics.metrics import compute_metrics_from_result

        result = run_from_config(
            config,
            bars=self._window_bars,
            states=self._window_states,
            pipeline=self._pipeline,
            translator=self._translator,
            instrument_master=self._instrument_master,
            mode=self._mode,
        )
        metrics = compute_metrics_from_result(result)
        return {
            "sharpe": metrics.sharpe,
            "sortino": metrics.sortino,
            "calmar": metrics.calmar,
            "cagr": metrics.cagr,
            "rachev": metrics.rachev_ratio_05,
            "return_on_margin": metrics.return_on_max_initial_margin,
            "max_drawdown": metrics.max_drawdown,
        }, 1

    def _param_importances(
        self, trials: Sequence[TrialSummary], optuna_study: Any | None
    ) -> dict[str, float]:
        if self._backend is OptimizationBackend.OPTUNA and optuna_study is not None:
            optuna = self._optuna
            if optuna is not None:
                try:
                    raw = optuna.importance.get_param_importances(optuna_study)
                    importances = {
                        str(name): float(value)
                        for name, value in raw.items()
                        if math.isfinite(float(value)) and float(value) >= 0.0
                    }
                    if importances:
                        return importances
                except Exception:
                    pass
        return deterministic_param_importances(self._request.search_space, trials)


# ---------------------------------------------------------------------------
# Module-level convenience API
# ---------------------------------------------------------------------------


def run_optimization(
    config: StrategyConfig,
    request: OptimizationRequest,
    *,
    instrument_master: InstrumentsConfig | None = None,
    bars: pl.DataFrame | None = None,
    states: Sequence[MarketState] | None = None,
    seed: int = 7,
    use_optuna: bool | None = None,
    output_dir: Path | str | None = None,
    persist: bool = True,
    mode: BacktestMode | None = None,
    offline: bool = False,
    pipeline: Any | None = None,
    translator: Any | None = None,
    backtest_runner: Callable[..., WalkForwardReport] | None = None,
    drawdown_penalty_coefficient: float = 1.0,
    study_id: str | None = None,
) -> OptimizationResult:
    """Run one walk-forward optimisation study (convenience wrapper)."""
    optimizer = WalkForwardOptimizer(
        config,
        request=request,
        instrument_master=instrument_master,
        bars=bars,
        states=states,
        seed=seed,
        use_optuna=use_optuna,
        output_dir=output_dir,
        persist=persist,
        mode=mode,
        offline=offline,
        pipeline=pipeline,
        translator=translator,
        backtest_runner=backtest_runner,
        drawdown_penalty_coefficient=drawdown_penalty_coefficient,
        study_id=study_id,
    )
    return optimizer.optimize()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _arg_trial_count(value: str) -> int:
    """Argparse type: 2 <= n_trials <= 200."""
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be an integer between 2 and 200") from exc
    if not 2 <= parsed <= 200:
        raise argparse.ArgumentTypeError("must be between 2 and 200")
    return parsed


def _arg_at_least_two(value: str) -> int:
    """Argparse type: integer >= 2."""
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be an integer >= 2") from exc
    if parsed < 2:
        raise argparse.ArgumentTypeError("must be >= 2")
    return parsed


def _arg_non_negative_int(value: str) -> int:
    """Argparse type: integer >= 0."""
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for walk-forward strategy optimisation (offline capable)."""
    parser = argparse.ArgumentParser(
        prog="futuresrl-optimize",
        description=(
            "Optimise a FuturesRL-X strategy strictly on purged/embargoed walk-forward "
            "out-of-sample folds (Optuna when installed, deterministic LHS fallback)."
        ),
    )
    parser.add_argument(
        "--strategy",
        required=True,
        type=Path,
        help="Path to a declarative strategy YAML",
    )
    parser.add_argument(
        "--trials",
        type=_arg_trial_count,
        default=DEFAULT_TRIAL_COUNT,
        metavar="N",
        help="Number of optimisation trials (2..200, default 25)",
    )
    parser.add_argument(
        "--objective",
        choices=list(OBJECTIVE_METRICS),
        default="sharpe",
        help="Out-of-sample objective metric (default: sharpe)",
    )
    parser.add_argument(
        "--splits",
        type=_arg_at_least_two,
        default=4,
        metavar="N",
        help="Walk-forward folds (default 4)",
    )
    parser.add_argument(
        "--purge-bars",
        type=_arg_non_negative_int,
        default=5,
        metavar="N",
        help="Bars purged before each OOS test window (default 5)",
    )
    parser.add_argument(
        "--embargo-bars",
        type=_arg_non_negative_int,
        default=5,
        metavar="N",
        help="Bars embargoed after each OOS test window (default 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Deterministic sampler seed (default 7)",
    )
    parser.add_argument(
        "--bars",
        type=Path,
        default=None,
        help="Optional Parquet file with canonical OHLCV bars (offline source otherwise)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Allow deterministic offline pipeline/translator fallbacks (no network)",
    )
    parser.add_argument(
        "--in-sample",
        action="store_true",
        help="Opt-in in-sample evaluation instead of walk-forward OOS (not recommended)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output JSON path (default: persisted study artifact path)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Study artifact directory (default: settings.optimization_artifacts_dir)",
    )
    args = parser.parse_args(argv)
    try:
        config = load_strategy_config(args.strategy)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
        raise AssertionError("unreachable") from exc
    bars: pl.DataFrame | None = None
    if args.bars is not None:
        try:
            bars = ensure_bars_schema(pl.read_parquet(args.bars), context="optimizer_bars")
        except Exception as exc:
            parser.error(f"failed to load bars from {args.bars}: {exc}")
            raise AssertionError("unreachable") from exc
    try:
        request = OptimizationRequest(
            base_strategy_id=config.id,
            objective_metric=args.objective,
            n_trials=args.trials,
            walk_forward=not args.in_sample,
            wf_splits=args.splits,
            purge_bars=args.purge_bars,
            embargo_bars=args.embargo_bars,
            search_space=default_search_space(config),
        )
    except (ValidationError, ValueError) as exc:
        parser.error(f"invalid optimization request: {exc}")
        raise AssertionError("unreachable") from exc
    try:
        result = run_optimization(
            config,
            request,
            bars=bars,
            seed=args.seed,
            output_dir=args.output_dir,
            mode=BacktestMode.VECTORIZED,
            offline=args.offline,
        )
    except Exception as exc:
        print(f"error: optimization failed: {exc}", file=sys.stderr)
        return 1
    output = (
        args.output
        if args.output is not None
        else (Path(result.artifact_path) if result.artifact_path is not None else None)
    )
    if output is not None:
        write_optimization_artifact(result, output)
    summary = {
        "study_id": result.study_id,
        "base_strategy_id": result.base_strategy_id,
        "objective_metric": result.objective_metric,
        "direction": result.direction,
        "n_trials": result.n_trials,
        "completed_trials": result.completed_trials,
        "failed_trials": result.failed_trials,
        "best_trial_number": result.best_trial_number,
        "best_value": result.best_value,
        "best_raw_value": result.best_raw_value,
        "best_params": result.best_params,
        "backend": result.backend.value,
        "fold_count": result.fold_count,
        "artifact_path": result.artifact_path,
        "output": str(output) if output is not None else None,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
