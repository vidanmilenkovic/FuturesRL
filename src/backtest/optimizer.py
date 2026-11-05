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
import contextlib
import importlib
import json
import math
import os
import sys
import tempfile
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
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

#: Hard objective assigned to flat/zero-activity trials (never crowned as best).
HARD_INVALID_OBJECTIVE: float = -10.0

#: Minimum offline bar window for walk-forward optimisation/warm-up.
MIN_OFFLINE_BARS: int = 750

#: Default warm-up (annual indicator history) used to size the offline window.
_MIN_INDICATOR_WARMUP_BARS: int = 252

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
    min_trades: int = Field(default=3, ge=0)
    min_avg_gross_exposure: float = Field(default=0.01, ge=0.0, le=1.0)
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
    n_trades: int = Field(default=0, ge=0)
    avg_gross_exposure: float = Field(default=0.0, ge=0.0)
    zero_activity: bool = False
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


def _set_leaf(
    cursor: Any,
    part: str,
    value: ParamValue,
    param_path: str,
    *,
    allow_new: bool,
) -> None:
    if not isinstance(cursor, dict):
        raise ValueError(f"unknown parameter path {param_path!r}: {part!r} is not a mapping")
    if part not in cursor and not allow_new:
        raise ValueError(f"unknown parameter path {param_path!r}: missing {part!r}")
    cursor[part] = value


def _get_dotted(payload: Mapping[str, Any], param_path: str) -> Any:
    cursor: Any = payload
    for part in _split_param_path(param_path):
        cursor = _descend(cursor, part, param_path)
    return cursor


def _set_dotted(
    payload: dict[str, Any],
    param_path: str,
    value: ParamValue,
    *,
    allow_new: bool = False,
) -> None:
    parts = _split_param_path(param_path)
    cursor: Any = payload
    for part in parts[:-1]:
        cursor = _descend(cursor, part, param_path)
    _set_leaf(cursor, parts[-1], value, param_path, allow_new=allow_new)


def apply_parameters(
    config: StrategyConfig,
    params: Mapping[str, ParamValue],
    *,
    allow_new_params: bool = False,
) -> StrategyConfig:
    """Apply sampled parameters to a copy of ``config`` by dotted path.

    The modified payload is re-validated through :class:`StrategyConfig`, so an
    invalid combination raises ``ValueError`` instead of producing a broken
    strategy. ``allow_new_params`` permits catalogued stage parameters that the
    base config omits (e.g. ``risk.quant_risk.trailing_stop_pct``): the optimizer
    uses it for catalog-validated defaults, while the public default stays strict.
    """
    raw: dict[str, Any] = config.model_dump(mode="python")
    for param_path, value in params.items():
        _set_dotted(raw, param_path, value, allow_new=allow_new_params)
    try:
        return StrategyConfig.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(
            f"sampled parameters {dict(params)!r} produced an invalid StrategyConfig: {exc}"
        ) from exc


#: Curated *active-core* catalog keys used for the default optimizer search space.
#: This is the Studio's "high-signal hypothesis" set: the active allocator's
#: volatility target plus the active timing overlays' core signal parameters and
#: the quant-risk trailing stop. Structural/hygiene params, inactive overlays,
#: arbitrary risk caps and translator plumbing are deliberately excluded.
CORE_OPTIMIZER_KEYS: tuple[str, ...] = (
    "allocator.volatility_target.target_annual_volatility",
    "allocator.risk_parity.target_annual_volatility",
    "timing.tsmom.lookback_fast",
    "timing.tsmom.lookback_slow",
    "timing.kama.window",
    "timing.mean_reversion.rsi_period",
    "timing.mean_reversion.oversold_threshold",
    "timing.mean_reversion.overbought_threshold",
    "timing.mean_reversion.turbulence_veto_threshold",
    "timing.mean_reversion.reversion_strength",
    "risk.quant_risk.trailing_stop_pct",
)

#: Defensive blocklist of structural/data-hygiene leaf params (also marked
#: ``optuna_tunable=False`` by the strategy catalog; duplicated here so the
#: optimizer still filters them when the catalog module is unavailable).
_UNIVERSE_HYGIENE_PARAMS: frozenset[str] = frozenset(
    {"min_avg_dollar_volume", "min_open_interest", "min_history_days", "top_n"}
)
_LLM_STRUCTURAL_PARAMS: frozenset[str] = frozenset(
    {
        "sample_count",
        "cache_ttl_seconds",
        "macro_lockout_minutes_before",
        "macro_lockout_minutes_after",
        "macro_lockout_action",
        "macro_lockout_leverage_cap",
        "semantic_variance_max",
    }
)

_MISSING: object = object()


@dataclass(frozen=True, slots=True)
class _CatalogSearchContext:
    """Active catalog bindings for one strategy config.

    Built from strategy-eng's catalog helpers when ``src.strategies`` is
    importable; the optimizer degrades gracefully to its built-in curated bounds
    when it is not.
    """

    key_to_path: Mapping[str, str]
    numeric_ranges: Mapping[str, ParameterRange]
    active_tunable_paths: frozenset[str]
    injectable_paths: frozenset[str]
    default_selected_keys: tuple[str, ...]


def _path_value(strategy: Mapping[str, Any], param_path: str) -> Any:
    """Value at a dotted path, or the ``_MISSING`` sentinel when absent."""
    try:
        return _get_dotted(strategy, param_path)
    except ValueError:
        return _MISSING


def _is_structural_parameter(param_path: str) -> bool:
    """Whether a dotted path addresses a structural/data-hygiene parameter."""
    parts = param_path.split(".")
    leaf = parts[-1].lower() if parts else ""
    if leaf in _LLM_STRUCTURAL_PARAMS:
        return True
    return bool(parts and parts[0] == "universe" and leaf in _UNIVERSE_HYGIENE_PARAMS)


def _normalise_param_path(strategy: Mapping[str, Any], param_path: str) -> str:
    """Normalise ``stage.stage_name.param`` catalog keys to dotted config paths."""
    cleaned = param_path.strip()
    parts = cleaned.split(".")
    if len(parts) < 3:
        return cleaned
    stage = parts[0]
    if stage in {"universe", "allocator"}:
        block = strategy.get(stage)
        if isinstance(block, Mapping):
            name = str(block.get("name", "")).strip().lower()
            if parts[1].strip().lower() == name and parts[2] != "params":
                return ".".join([stage, "params", *parts[2:]])
        return cleaned
    if stage in {"timing", "risk"}:
        entries = strategy.get(stage)
        if not isinstance(entries, (list, tuple)) or parts[1].isdigit():
            return cleaned
        wanted = parts[1].strip().lower()
        for index, entry in enumerate(entries):
            if isinstance(entry, Mapping) and str(entry.get("name", "")).strip().lower() == wanted:
                tail = parts[2:]
                if tail and tail[0] == "params":
                    tail = tail[1:]
                return ".".join([stage, str(index), "params", *tail])
    return cleaned


def _catalog_search_context(config: StrategyConfig) -> _CatalogSearchContext | None:
    """Active/tunable catalog bindings for ``config`` (``None`` when unavailable)."""
    try:
        catalog_module: Any = importlib.import_module("src.strategies.parameter_catalog")
    except ImportError:  # pragma: no cover - graceful fallback path
        return None
    strategy = config.model_dump(mode="python")
    try:
        catalog = catalog_module.parameter_catalog()
        resolver = getattr(catalog_module, "resolve_param_path", None)
        parameters_fn = getattr(catalog_module, "parameters_for_strategy", None)
        if callable(parameters_fn):
            active_specs: Sequence[Any] = tuple(parameters_fn(strategy))
        else:  # pragma: no cover - compatibility with older catalog revisions
            active_keys = set(catalog_module.parameter_keys_for_strategy(strategy))
            active_specs = tuple(spec for spec in catalog.parameters if spec.key in active_keys)
        defaults_fn = getattr(catalog_module, "default_selected_parameter_keys", None)
        if callable(defaults_fn):
            default_selected = tuple(str(key) for key in defaults_fn(strategy))
        else:  # pragma: no cover - compatibility with older catalog revisions
            default_selected = tuple(
                str(spec.key)
                for spec in active_specs
                if bool(getattr(spec, "optuna_default_selected", False))
            )
    except Exception:  # pragma: no cover - defensive: catalog must never block tuning
        return None
    if not callable(resolver):
        return None
    key_to_path: dict[str, str] = {}
    numeric_ranges: dict[str, ParameterRange] = {}
    active_tunable_paths: set[str] = set()
    injectable_paths: set[str] = set()
    for spec in active_specs:
        if not bool(getattr(spec, "optuna_tunable", False)):
            continue
        key = str(getattr(spec, "key", ""))
        if not key:
            continue
        spec_type = getattr(getattr(spec, "type", None), "value", None)
        if spec_type not in {"int", "float", "bool", "select"}:
            continue
        try:
            path = resolver(strategy, key)
        except (KeyError, ValueError):
            continue
        path = _normalise_param_path(strategy, str(path))
        if _is_structural_parameter(path):
            continue
        missing = _path_value(strategy, path) is _MISSING
        if missing:
            # A catalogued stage parameter that the base config omits can still
            # be tuned: applying it adds the key to the stage's ``params`` dict.
            parent_path = ".".join(path.split(".")[:-1])
            try:
                parent = _get_dotted(strategy, parent_path)
            except ValueError:
                continue
            if not isinstance(parent, Mapping) or ".params." not in f".{path}.":
                continue
            injectable_paths.add(path)
        key_to_path[key] = path
        active_tunable_paths.add(path)
        if spec_type not in {"int", "float"}:
            continue
        minimum = getattr(spec, "min", None)
        maximum = getattr(spec, "max", None)
        step = getattr(spec, "step", None)
        if minimum is None or maximum is None:
            continue
        try:
            low = float(minimum)
            high = float(maximum)
            step_value = float(step) if step is not None else None
        except (TypeError, ValueError):  # pragma: no cover - defensive catalog data
            continue
        if not math.isfinite(low) or not math.isfinite(high) or low >= high:
            continue
        range_type: Literal["int", "float"] = "int" if spec_type == "int" else "float"
        if range_type == "int" and step_value is None:
            step_value = 1.0
        numeric_ranges[path] = ParameterRange(
            param_path=path,
            type=range_type,
            low=low,
            high=high,
            step=step_value,
        )
    return _CatalogSearchContext(
        key_to_path=key_to_path,
        numeric_ranges=numeric_ranges,
        active_tunable_paths=frozenset(active_tunable_paths),
        injectable_paths=frozenset(injectable_paths),
        default_selected_keys=default_selected,
    )


def _fallback_default_ranges(config: StrategyConfig) -> list[ParameterRange]:
    """Built-in curated default ranges used when the strategy catalog is unavailable."""
    strategy = config.model_dump(mode="python")
    ranges: list[ParameterRange] = []
    allocator = strategy.get("allocator")
    if isinstance(allocator, Mapping):
        name = str(allocator.get("name", "")).strip().lower()
        params = allocator.get("params")
        if (
            name in {"volatility_target", "risk_parity"}
            and isinstance(params, Mapping)
            and "target_annual_volatility" in params
        ):
            ranges.append(
                ParameterRange(
                    param_path="allocator.params.target_annual_volatility",
                    type="float",
                    low=0.01,
                    high=0.6,
                    step=0.01,
                )
            )
    overlay_bounds: Mapping[str, tuple[tuple[str, str, float, float, float], ...]] = {
        "tsmom": (
            ("lookback_fast", "int", 2.0, 126.0, 1.0),
            ("lookback_slow", "int", 21.0, 504.0, 1.0),
        ),
        "kama": (("window", "int", 2.0, 50.0, 1.0),),
        "mean_reversion": (
            ("rsi_period", "int", 2.0, 50.0, 1.0),
            ("oversold_threshold", "float", 1.0, 49.0, 1.0),
            ("overbought_threshold", "float", 51.0, 99.0, 1.0),
            ("turbulence_veto_threshold", "float", 0.05, 10.0, 0.05),
            ("reversion_strength", "float", 0.0, 1.0, 0.05),
        ),
    }
    timing = strategy.get("timing")
    if isinstance(timing, (list, tuple)):
        for index, entry in enumerate(timing):
            if not isinstance(entry, Mapping):
                continue
            name = str(entry.get("name", "")).strip().lower()
            params = entry.get("params")
            if not isinstance(params, Mapping):
                continue
            for param_name, type_name, low, high, step in overlay_bounds.get(name, ()):
                if param_name not in params:
                    continue
                range_type: Literal["int", "float"] = "int" if type_name == "int" else "float"
                ranges.append(
                    ParameterRange(
                        param_path=f"timing.{index}.params.{param_name}",
                        type=range_type,
                        low=low,
                        high=high,
                        step=step,
                    )
                )
    risk = strategy.get("risk")
    if isinstance(risk, (list, tuple)):
        for index, entry in enumerate(risk):
            if not isinstance(entry, Mapping):
                continue
            if str(entry.get("name", "")).strip().lower() != "quant_risk":
                continue
            params = entry.get("params")
            if isinstance(params, Mapping) and "trailing_stop_pct" in params:
                ranges.append(
                    ParameterRange(
                        param_path=f"risk.{index}.params.trailing_stop_pct",
                        type="float",
                        low=0.01,
                        high=0.9,
                        step=0.01,
                    )
                )
    return ranges


def _generic_active_ranges(context: _CatalogSearchContext) -> list[ParameterRange]:
    """Generic active numeric fallback (allocator/timing first, capped at six)."""
    preferred = [
        parameter
        for path, parameter in context.numeric_ranges.items()
        if path.startswith(("allocator.", "timing."))
    ]
    remaining = [
        parameter
        for path, parameter in context.numeric_ranges.items()
        if not path.startswith(("allocator.", "timing."))
    ]
    return (preferred + remaining)[:6]


def filter_search_space_for_config(
    search_space: Sequence[ParameterRange],
    config: StrategyConfig,
) -> tuple[ParameterRange, ...]:
    """Drop inactive, non-tunable and structural parameters from an explicit space.

    Used by :class:`WalkForwardOptimizer` for UI-supplied search spaces so a stale
    Studio selection can never evaluate parameters whose stage/overlay is not
    active in the base strategy. Paths are normalised from catalog-key form
    (``timing.tsmom.lookback_fast``) to the concrete config form, order is kept
    and duplicates are removed. Raises an actionable ``ValueError`` when nothing
    tunable remains.
    """
    if not search_space:
        raise ValueError(
            "optimizer search space is empty; select at least one active tunable parameter"
        )
    strategy = config.model_dump(mode="python")
    context = _catalog_search_context(config)
    active_paths = context.active_tunable_paths if context is not None else None
    filtered: list[ParameterRange] = []
    seen: set[str] = set()
    dropped: list[str] = []
    for parameter in search_space:
        path = _normalise_param_path(strategy, parameter.param_path)
        if path in seen:
            continue
        existing = _path_value(strategy, path)
        injectable = context is not None and path in context.injectable_paths
        keep = (existing is not _MISSING or injectable) and not _is_structural_parameter(path)
        if keep and active_paths is not None:
            keep = path in active_paths
        if keep and not injectable:
            if parameter.type == "categorical":
                keep = not isinstance(existing, (dict, list, tuple)) and existing is not None
            else:
                keep = isinstance(existing, (int, float)) and not isinstance(existing, bool)
        if not keep:
            dropped.append(parameter.param_path)
            continue
        seen.add(path)
        filtered.append(parameter.model_copy(update={"param_path": path}))
    if not filtered:
        hint = ""
        if active_paths:
            preview = sorted(active_paths)[:8]
            hint = f" Active tunable paths include: {preview}."
        raise ValueError(
            f"optimizer search space has no active tunable parameters for strategy "
            f"{config.id!r}; dropped inactive/non-tunable paths {sorted(set(dropped))}.{hint}"
        )
    return tuple(filtered)


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


def _finite_float(value: object, default: float = 0.0) -> float:
    """Best-effort finite float coercion."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    number = float(value)
    return number if math.isfinite(number) else default


def _avg_gross_exposure_from_result(result: object) -> float:
    """Mean ``gross_exposure / equity`` over a result's equity curve (or 0.0)."""
    equity_curve = getattr(result, "equity_curve", None)
    if not isinstance(equity_curve, pl.DataFrame):
        return 0.0
    columns = set(equity_curve.columns)
    if not {"equity", "gross_exposure"} <= columns:
        return 0.0
    equities = equity_curve.get_column("equity").to_numpy().astype(np.float64)
    gross = equity_curve.get_column("gross_exposure").to_numpy().astype(np.float64)
    if equities.size == 0 or gross.size != equities.size:
        return 0.0
    valid = np.isfinite(equities) & np.isfinite(gross) & (equities > 0.0) & (gross >= 0.0)
    if not bool(np.any(valid)):
        return 0.0
    return max(0.0, float(np.mean(gross[valid] / equities[valid])))


def _trade_count_from_report(report: object) -> int:
    """Total executed trades in a walk-forward report (robust to fake reports)."""
    total = getattr(report, "total_trades", None)
    if isinstance(total, int) and not isinstance(total, bool):
        return max(0, total)
    folds = getattr(report, "folds", ())
    if isinstance(folds, Sequence) and not isinstance(folds, (str, bytes)):
        summed = 0
        for fold in folds:
            value = getattr(fold, "n_trades", None)
            if isinstance(value, int) and not isinstance(value, bool):
                summed += max(0, value)
        if summed:
            return summed
    combined = getattr(report, "combined_result", None)
    value = getattr(combined, "n_trades", None) if combined is not None else None
    return max(0, value) if isinstance(value, int) and not isinstance(value, bool) else 0


def _activity_from_report(report: object) -> tuple[int, float]:
    """``(n_trades, avg_gross_exposure)`` for a walk-forward report."""
    n_trades = _trade_count_from_report(report)
    direct = getattr(report, "avg_gross_exposure", None)
    if isinstance(direct, (int, float)) and not isinstance(direct, bool):
        exposure = _finite_float(direct)
        if exposure > 0.0:
            return n_trades, max(0.0, exposure)
    combined = getattr(report, "combined_result", None)
    if combined is not None:
        return n_trades, _avg_gross_exposure_from_result(combined)
    return n_trades, max(0.0, _finite_float(direct))


def _activity_from_backtest_result(result: object) -> tuple[int, float]:
    """``(n_trades, avg_gross_exposure)`` for a single ``BacktestResult``."""
    raw_trades = getattr(result, "n_trades", None)
    n_trades = (
        max(0, raw_trades)
        if isinstance(raw_trades, int) and not isinstance(raw_trades, bool)
        else 0
    )
    return n_trades, _avg_gross_exposure_from_result(result)


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
# Persisted-study validation (legacy / inactive search spaces)
# ---------------------------------------------------------------------------

#: Top-level strategy stages understood by the declarative config/catalog.
_KNOWN_STAGES: frozenset[str] = frozenset({"universe", "allocator", "timing", "risk", "translator"})


def _optimization_payload(result_or_dict: OptimizationResult | Mapping[str, Any]) -> dict[str, Any]:
    """Normalise a typed result or a raw ``json.loads`` payload into a dict."""
    if isinstance(result_or_dict, OptimizationResult):
        return result_or_dict.model_dump(mode="json")
    if isinstance(result_or_dict, Mapping):
        return dict(result_or_dict)
    raise TypeError(
        "validate_persisted_optimization accepts an OptimizationResult or a mapping payload"
    )


def _looks_like_fixture_payload(payload: Mapping[str, Any]) -> bool:
    """API-owned fixture studies are not canonical engine studies and are skipped.

    Fixture payloads carry an explicit ``provenance`` tag, a non-engine backend
    (e.g. ``deterministic-fixture``) and no ``request``/``seed``; they are owned by
    the API model and must never be reported as legacy engine studies.
    """
    if "provenance" in payload:
        return True
    backend = payload.get("backend")
    engine_backends = {OptimizationBackend.OPTUNA.value, OptimizationBackend.DETERMINISTIC.value}
    if isinstance(backend, str) and backend not in engine_backends:
        return True
    request = payload.get("request")
    return not isinstance(request, Mapping) and "seed" not in payload


def _extract_base_strategy_id(payload: Mapping[str, Any]) -> str | None:
    """Best-effort ``base_strategy_id`` lookup (top-level or nested request)."""
    value = payload.get("base_strategy_id")
    if not isinstance(value, str):
        request = payload.get("request")
        if isinstance(request, Mapping):
            value = request.get("base_strategy_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _collect_persisted_parameter_paths(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Unique parameter paths recorded in the search space, best params and trials."""
    collected: list[str] = []

    def add(value: object) -> None:
        if isinstance(value, str) and value.strip():
            collected.append(value.strip())

    request = payload.get("request")
    if isinstance(request, Mapping):
        search_space = request.get("search_space")
        if isinstance(search_space, Sequence) and not isinstance(search_space, (str, bytes)):
            for entry in search_space:
                if isinstance(entry, Mapping):
                    add(entry.get("param_path"))
                elif isinstance(entry, str):
                    add(entry)
    best_params = payload.get("best_params")
    if isinstance(best_params, Mapping):
        for key in best_params:
            add(key)
    trials = payload.get("trials")
    if isinstance(trials, Sequence) and not isinstance(trials, (str, bytes)):
        for trial in trials:
            if not isinstance(trial, Mapping):
                continue
            params = trial.get("params")
            if isinstance(params, Mapping):
                for key in params:
                    add(key)
    seen: set[str] = set()
    unique: list[str] = []
    for path in collected:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return tuple(unique)


def _resolve_base_strategy_config(base_strategy_id: str) -> StrategyConfig | None:
    """Resolve a base strategy config by id (committed YAML, preset, custom YAML)."""
    repo_root = Path(__file__).resolve().parents[2]
    configs_dir = repo_root / "configs" / "strategies"
    direct = configs_dir / f"{base_strategy_id}.yaml"
    candidates: list[Path] = []
    if direct.is_file():
        candidates.append(direct)
    if configs_dir.is_dir():
        candidates.extend(sorted(path for path in configs_dir.glob("*.yaml") if path != direct))
    for path in candidates:
        try:
            config = load_strategy_config(path)
        except (FileNotFoundError, ValueError, OSError):
            continue
        if config.id == base_strategy_id:
            return config
    try:
        catalog_module: Any = importlib.import_module("src.strategies.parameter_catalog")
        catalog = catalog_module.parameter_catalog()
        for preset in catalog.presets:
            if str(getattr(preset, "id", "")) == base_strategy_id:
                resolved = preset.to_config()
                if isinstance(resolved, StrategyConfig):
                    return resolved
    except Exception:  # pragma: no cover - catalog must never block startup validation
        pass
    try:
        from configs.settings import get_settings

        custom_path = get_settings().strategy_artifacts_dir / f"{base_strategy_id}.yaml"
        if custom_path.is_file():
            return load_strategy_config(custom_path)
    except Exception:  # pragma: no cover - optional settings/custom-strategy lookup
        pass
    return None


def validate_persisted_optimization(
    result_or_dict: OptimizationResult | Mapping[str, Any],
    config: StrategyConfig | None = None,
) -> list[str]:
    """Validate a persisted optimizer study against its base strategy.

    Returns an empty list when the study is a valid current-engine study, or a
    sorted list of human-readable legacy/quarantine reasons when its recorded
    search space (or trial/best parameter paths) references an inactive
    stage/overlay, a non-tunable structural/data-hygiene path, or a parameter that
    is absent from the base strategy.

    Parameters
    ----------
    result_or_dict:
        A typed :class:`OptimizationResult` (from :func:`load_optimization`) or the
        raw ``json.loads`` payload mapping. API-owned fixture payloads (explicit
        ``provenance`` / non-engine backend / no ``request``+``seed``) are not
        canonical engine studies and return ``[]``.
    config:
        Optional pre-resolved base strategy config. When omitted (or when its id
        does not match ``base_strategy_id``) the helper resolves the base strategy
        itself from the committed YAMLs, catalog presets, or the custom strategy
        artifact directory.

    The helper is pure: it never mutates the payload or writes/quarantines files.
    Unresolvable base strategies are handled conservatively: non-tunable
    structural paths and unknown stages are flagged, while known-stage paths that
    cannot be verified are accepted (custom/undeployed strategies are not legacy).
    """
    payload = _optimization_payload(result_or_dict)
    if _looks_like_fixture_payload(payload):
        return []
    base_strategy_id = _extract_base_strategy_id(payload)
    if base_strategy_id is None:
        # Non-canonical payload without a base id: nothing engine-specific to check.
        return []
    parameter_paths = _collect_persisted_parameter_paths(payload)
    if not parameter_paths:
        return []
    resolved_config: StrategyConfig | None = None
    if config is not None and config.id == base_strategy_id:
        resolved_config = config
    else:
        resolved_config = _resolve_base_strategy_config(base_strategy_id)
    reasons: list[str] = []
    if resolved_config is None:
        for raw_path in parameter_paths:
            path = raw_path.strip()
            parts = path.split(".")
            stage = parts[0] if parts else ""
            if _is_structural_parameter(path):
                reasons.append(f"non-tunable structural/hygiene parameter path {raw_path!r}")
            elif stage not in _KNOWN_STAGES:
                reasons.append(
                    f"unknown stage in parameter path {raw_path!r} "
                    f"(base strategy {base_strategy_id!r} could not be resolved)"
                )
        return sorted(set(reasons))
    strategy = resolved_config.model_dump(mode="python")
    context = _catalog_search_context(resolved_config)
    active_paths = context.active_tunable_paths if context is not None else None
    for raw_path in parameter_paths:
        path = _normalise_param_path(strategy, raw_path)
        if _is_structural_parameter(path):
            reasons.append(f"non-tunable structural/hygiene parameter path {raw_path!r}")
            continue
        existing = _path_value(strategy, path)
        if existing is _MISSING:
            # Catalog-active parameters omitted by the committed YAML are still
            # tuneable (the optimizer injects them), so they are not legacy.
            if active_paths is not None and path in active_paths:
                continue
            reasons.append(
                f"inactive or unknown parameter path {raw_path!r} "
                f"for base strategy {base_strategy_id!r}"
            )
            continue
        if active_paths is not None and path not in active_paths:
            reasons.append(
                f"inactive parameter path {raw_path!r} for base strategy {base_strategy_id!r}"
            )
    return sorted(set(reasons))


def is_legacy_search_space(
    result_or_dict: OptimizationResult | Mapping[str, Any],
    config: StrategyConfig | None = None,
) -> bool:
    """Whether a persisted optimizer study should be quarantined as legacy."""
    return bool(validate_persisted_optimization(result_or_dict, config))


# ---------------------------------------------------------------------------
# Default search space
# ---------------------------------------------------------------------------


#: Curated default bounds for the active-core parameters: the catalog bounds are
#: clipped to classic signal horizons so a fresh Studio/CLI study is useful on
#: the deterministic offline window instead of sampling lookbacks longer than the
#: available history. Explicit UI search spaces keep the user's own bounds.
_CURATED_DEFAULT_BOUNDS: Mapping[str, tuple[float | None, float | None]] = {
    "target_annual_volatility": (0.05, 0.30),
    "lookback_fast": (None, 63.0),
    "lookback_slow": (64.0, 252.0),
    "window": (5.0, 30.0),
    "trailing_stop_pct": (0.03, 0.30),
}


def _apply_curated_default_bounds(ranges: list[ParameterRange]) -> list[ParameterRange]:
    """Clip catalog bounds of curated defaults to sensible signal horizons."""
    clipped: list[ParameterRange] = []
    for parameter in ranges:
        leaf = parameter.param_path.split(".")[-1]
        lower_cap, upper_cap = _CURATED_DEFAULT_BOUNDS.get(leaf, (None, None))
        if lower_cap is None and upper_cap is None:
            clipped.append(parameter)
            continue
        low, high = parameter.numeric_bounds()
        new_low = max(low, lower_cap) if lower_cap is not None else low
        new_high = min(high, upper_cap) if upper_cap is not None else high
        if new_low >= new_high:
            clipped.append(parameter)
            continue
        clipped.append(
            parameter.model_copy(update={"low": float(new_low), "high": float(new_high)})
        )
    return clipped


def _constrain_default_ranges(ranges: list[ParameterRange]) -> list[ParameterRange]:
    """Apply cross-parameter constraints to the curated default ranges.

    The catalog exposes independent TSMOM fast/slow lookback bounds; sampling them
    independently frequently produces ``lookback_fast >= lookback_slow`` (an
    invalid TSMOM definition). Partition the two ranges at a deterministic boundary
    so the default space only ever samples valid fast/slow pairs.
    """
    by_path = {parameter.param_path: parameter for parameter in ranges}
    fast = by_path.get("timing.0.params.lookback_fast")
    slow = by_path.get("timing.0.params.lookback_slow")
    if fast is None or slow is None or fast.low is None or fast.high is None:
        return ranges
    if slow.low is None or slow.high is None:
        return ranges
    boundary = max(
        int(fast.low),
        min(int(fast.high), int((float(fast.high) + float(slow.low)) / 2)),
    )
    slow_low = max(int(slow.low), boundary + 1)
    if boundary < int(fast.low) or slow_low >= int(slow.high):
        return ranges
    constrained: list[ParameterRange] = []
    for parameter in ranges:
        if parameter.param_path == fast.param_path:
            constrained.append(parameter.model_copy(update={"high": float(boundary)}))
        elif parameter.param_path == slow.param_path:
            constrained.append(parameter.model_copy(update={"low": float(slow_low)}))
        else:
            constrained.append(parameter)
    return constrained


def default_search_space(config: StrategyConfig) -> tuple[ParameterRange, ...]:
    """Derive the curated, ACTIVE-ONLY default search space for the optimizer.

    Only high-signal parameters of the strategy's *active* stages are included:
    the active allocator's volatility target, the active timing overlays' core
    signal parameters (TSMOM fast/slow, KAMA window, carry threshold,
    mean-reversion thresholds) and the quant-risk trailing stop. Structural/data
    hygiene parameters, inactive overlays, arbitrary risk caps and translator
    plumbing are deliberately excluded; the returned space is passed through
    :func:`filter_search_space_for_config` so it can never contain a parameter
    that is absent from the config.

    Bounds come from strategy-eng's parameter catalog when it is importable, with
    a built-in curated fallback otherwise.
    """
    context = _catalog_search_context(config)
    ranges: list[ParameterRange] = []
    if context is not None:
        for key in CORE_OPTIMIZER_KEYS:
            path = context.key_to_path.get(key)
            parameter = context.numeric_ranges.get(path) if path is not None else None
            if parameter is not None and parameter not in ranges:
                ranges.append(parameter)
        if not ranges:
            for key in context.default_selected_keys:
                path = context.key_to_path.get(key)
                parameter = context.numeric_ranges.get(path) if path is not None else None
                if parameter is not None and parameter not in ranges:
                    ranges.append(parameter)
        if not ranges:
            ranges = _generic_active_ranges(context)
    else:
        ranges = _fallback_default_ranges(config)
    if not ranges:
        raise ValueError(
            f"cannot derive an active default search space for strategy {config.id!r}: "
            "no curated/active numeric tunables were found; declare explicit "
            "ParameterRange entries"
        )
    curated = _apply_curated_default_bounds(ranges)
    return filter_search_space_for_config(tuple(_constrain_default_ranges(curated)), config)


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
        filtered_search_space = filter_search_space_for_config(request.search_space, config)
        optuna_module = _load_optuna_module()
        if use_optuna is True and optuna_module is None:
            raise RuntimeError(
                "Optuna is not installed; install the 'optuna' dependency or let "
                "use_optuna=None select the deterministic fallback"
            )
        resolved_use_optuna = (optuna_module is not None) if use_optuna is None else use_optuna
        self._config = config
        self._request = request.model_copy(update={"search_space": filtered_search_space})
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
        # Defensive re-filter: an explicit UI/WS search space may contain stale or
        # inactive parameters; never evaluate those.
        self._request = self._request.model_copy(
            update={
                "search_space": filter_search_space_for_config(
                    self._request.search_space, self._config
                )
            }
        )
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
        # Flat/low-exposure trials must never be crowned as best. If every
        # completed trial is inactive we still persist the study (for diagnosis)
        # with best_trial_number=None and empty best_params/best_strategy_config.
        active = [trial for trial in completed if not trial.zero_activity]
        best = (
            max(active, key=lambda trial: (trial.objective_value, -trial.trial_number))
            if active
            else None
        )
        best_config = (
            apply_parameters(self._config, best.params, allow_new_params=True)
            if best is not None
            else None
        )
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
            best_trial_number=best.trial_number if best is not None else None,
            best_params=dict(best.params) if best is not None else {},
            best_value=best.objective_value if best is not None else None,
            best_raw_value=best.raw_objective if best is not None else None,
            best_strategy_config=(
                best_config.model_dump(mode="json") if best_config is not None else {}
            ),
            trials=tuple(trials),
            param_importances=importances,
            drawdown_penalty_threshold=self._request.drawdown_penalty_threshold,
            walk_forward=self._request.walk_forward,
            fold_count=best.fold_count if best is not None else 0,
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

            bars = engine_module._offline_bars(
                self._config,
                self._instrument_master,
                periods=self._offline_bar_periods(),
            )
        if states is None and bars is not None:
            states = build_states_from_bars(
                bars,
                self._instrument_master,
                initial_equity=self._initial_equity(),
            )
        return bars, states

    def _warmup_periods(self) -> int:
        """Indicator warm-up required before the first OOS test bar."""
        warmup = _MIN_INDICATOR_WARMUP_BARS
        for parameter in self._request.search_space:
            if parameter.type == "categorical" or parameter.high is None:
                continue
            leaf = parameter.param_path.split(".")[-1].lower()
            if "lookback" in leaf or leaf == "window":
                warmup = max(warmup, int(parameter.high))
        return max(1, warmup)

    def _offline_bar_periods(self) -> int:
        """Offline bar count for optimisation: >=750, enough folds and warm-up.

        Uses the v1.5.3 formula ``max(750, 150*wf_splits + 2*(purge+embargo) + 300)``
        and additionally guarantees the derived walk-forward training block is at
        least one indicator warm-up long (``purge + (splits+1)*warmup``), so
        lookback indicators are warm across the train->test boundary. In-sample
        studies keep the previous 260-bar offline behaviour.
        """
        if not self._request.walk_forward:
            return max(260, _MIN_INDICATOR_WARMUP_BARS)
        splits = self._request.wf_splits
        bars = 150 * splits + 2 * (self._request.purge_bars + self._request.embargo_bars) + 300
        warmup_window = self._request.purge_bars + (splits + 1) * self._warmup_periods()
        return max(MIN_OFFLINE_BARS, 260, bars, warmup_window)

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
        # Optuna logs every trial at INFO to stderr by default; keep CLI runs
        # quiet (zero-warning contract) while retaining genuine warnings/errors.
        logging_module = getattr(optuna, "logging", None)
        if logging_module is not None:
            with contextlib.suppress(Exception):  # pragma: no cover - optuna API guard
                logging_module.set_verbosity(logging_module.WARNING)
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
            trial_config = apply_parameters(self._config, params, allow_new_params=True)
            metrics, fold_count, n_trades, avg_gross_exposure = self._evaluate_config(trial_config)
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
        zero_activity = (
            n_trades < self._request.min_trades
            or avg_gross_exposure < self._request.min_avg_gross_exposure
        )
        return TrialSummary(
            trial_number=trial_number,
            params=dict(params),
            state=TrialState.COMPLETE,
            # Flat/low-exposure trials get a hard constant objective and are
            # excluded from best-trial selection (raw metrics are still recorded).
            objective_value=HARD_INVALID_OBJECTIVE if zero_activity else raw_objective - penalty,
            raw_objective=raw_objective,
            oos_sharpe=metrics["sharpe"],
            oos_sortino=metrics["sortino"],
            oos_calmar=metrics["calmar"],
            oos_cagr=metrics["cagr"],
            oos_rachev=metrics["rachev"],
            oos_return_on_margin=metrics["return_on_margin"],
            max_dd=metrics["max_drawdown"],
            fold_count=fold_count,
            n_trades=n_trades,
            avg_gross_exposure=avg_gross_exposure,
            zero_activity=zero_activity,
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

    def _evaluate_config(self, config: StrategyConfig) -> tuple[dict[str, float], int, int, float]:
        if self._request.walk_forward:
            return self._evaluate_walk_forward(config)
        return self._evaluate_in_sample(config)

    def _evaluate_walk_forward(
        self, config: StrategyConfig
    ) -> tuple[dict[str, float], int, int, float]:
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
        n_trades, avg_gross_exposure = _activity_from_report(report)
        return _extract_oos_metrics(report), report.fold_count, n_trades, avg_gross_exposure

    def _evaluate_in_sample(
        self, config: StrategyConfig
    ) -> tuple[dict[str, float], int, int, float]:
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
        n_trades, avg_gross_exposure = _activity_from_backtest_result(result)
        return (
            {
                "sharpe": metrics.sharpe,
                "sortino": metrics.sortino,
                "calmar": metrics.calmar,
                "cagr": metrics.cagr,
                "rachev": metrics.rachev_ratio_05,
                "return_on_margin": metrics.return_on_max_initial_margin,
                "max_drawdown": metrics.max_drawdown,
            },
            1,
            n_trades,
            avg_gross_exposure,
        )

    def _param_importances(
        self, trials: Sequence[TrialSummary], optuna_study: Any | None
    ) -> dict[str, float]:
        if self._backend is OptimizationBackend.OPTUNA and optuna_study is not None:
            optuna = self._optuna
            if optuna is not None:
                try:
                    with warnings.catch_warnings():
                        # Optuna warns when quantile targets select identical
                        # trial sets (small studies); deterministic fallback is
                        # still valid, so keep the CLI warning-free.
                        warnings.simplefilter("ignore", UserWarning)
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
            "Optimise a FuturesRL strategy strictly on purged/embargoed walk-forward "
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
