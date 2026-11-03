"""Strategy Studio service: parameter catalog, preset templates and YAML persistence.

The Strategy Studio lets an operator compose a FinRL-X pipeline from a typed
parameter catalog (owned by ``src.strategies.parameter_catalog``), load one of the
built-in hypothesis presets and save the result as a declarative strategy YAML.

Safety contract
---------------
* Strategy ids are strict slugs (``^[a-z][a-z0-9_]{0,63}$``); traversal attempts
  (``..``, separators, absolute paths, NUL bytes) are rejected before any
  filesystem operation and every resolved target is re-checked to live inside its
  base directory (defence in depth against symlink/encoding tricks).
* Payloads are validated through the frozen :mod:`configs.strategy` models before
  a single byte is written; a strategy that would not build a FinRL-X pipeline is
  never persisted.
* Writes are atomic (temp file + ``os.replace``) and deterministic, so a reader
  never observes a partial strategy.
* Built-in strategies (``configs/strategies/<id>.yaml``) are only overwritten in
  a real runtime environment; backtest/test environments write their overrides to
  ``settings.strategy_artifacts_dir`` instead, keeping the committed configs
  immutable during CI.

The parameter catalog is consumed lazily: when
``src.strategies.parameter_catalog`` is importable its typed registry
(``catalog_dict()``) and the five hypothesis presets are used verbatim; otherwise
a deterministic built-in fallback keeps the Studio fully usable and the response's
``source``/``preset_source`` fields report the provenance honestly.
"""

from __future__ import annotations

import logging
import math
import os
import re
import tempfile
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Literal, cast

import yaml
from configs.instruments import (
    InstrumentsConfig,
    InstrumentSpec,
    clear_instruments_cache,
    load_instruments,
)
from configs.settings import Settings
from configs.strategy import StrategyConfig, parse_strategy_config
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.api.schemas import StageConfig, StrategyConfigView

logger = logging.getLogger(__name__)

__all__ = [
    "CatalogStage",
    "ParameterSpec",
    "ParameterType",
    "SavedStrategy",
    "StrategyCatalogResponse",
    "StrategyPreset",
    "StrategyStudioError",
    "StrategyStudioService",
    "apply_dotted_params",
    "fallback_parameter_catalog",
    "fallback_presets",
    "load_parameter_catalog",
    "load_presets",
]

#: Maximum accepted strategy YAML/JSON payload size (defensive bound).
MAX_STRATEGY_PAYLOAD_BYTES: int = 256 * 1024

#: Strict strategy id slug (also protects against path traversal).
_STRATEGY_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")

#: Runtime environments in which built-in YAMLs may be edited in place.
_BUILTIN_WRITE_ENVIRONMENTS: frozenset[str] = frozenset({"development", "paper", "live"})

#: Parameter catalog module owned by ``src.strategies``.
PARAMETER_CATALOG_MODULE = "src.strategies.parameter_catalog"

type ParameterType = Literal["int", "float", "bool", "select", "multiselect"]
type ScalarChoice = bool | int | float | str
type Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(UTC)


class StrategyStudioError(ValueError):
    """A strategy-studio request that must be rejected (with an HTTP hint)."""

    def __init__(self, message: str, *, status_code: int = 422) -> None:
        super().__init__(message)
        self.status_code = status_code


# ---------------------------------------------------------------------------
# Catalog / response models (local until the shared schema file is free)
# ---------------------------------------------------------------------------


class _StudioModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class ParameterSpec(_StudioModel):
    """One editable/tunable pipeline parameter shown by the Studio."""

    key: str
    stage: str
    stage_name: str = ""
    param: str = ""
    label: str = ""
    description: str = ""
    type: ParameterType = "float"
    default: ScalarChoice | list[ScalarChoice] | None = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: list[ScalarChoice] = Field(default_factory=list)
    unit: str | None = None
    optuna_tunable: bool = False
    #: Additive v1.5.1 flag: true only for the curated parameters pre-selected in
    #: the Studio's Optuna search-space defaults (exactly five on the live catalog;
    #: v1.5.3 removed ``timing.carry_filter.min_carry_bps`` from the global preselect).
    optuna_default_selected: bool = False
    log_scale: bool = False
    yaml_encoding: str = "scalar"
    #: Cardinality bounds for multiselect params (``None`` for scalar params).
    min_items: int | None = None
    max_items: int | None = None
    #: Binding location: ``stage_params`` | ``translator`` | ``drl_block``.
    location: str = "stage_params"
    #: Concrete dotted path hint inside a serialised strategy (best-effort without
    #: a base strategy; ``resolve_optimizer_path`` computes the exact list index).
    path: str = ""

    def as_dict(self) -> dict[str, object]:
        """JSON-serialisable representation."""
        return self.model_dump(mode="json")


class StageChoiceOption(_StudioModel):
    """One selectable option (implementation) inside a pipeline stage."""

    name: str
    label: str = ""
    description: str = ""
    parameter_keys: list[str] = Field(default_factory=list)


class CatalogStage(_StudioModel):
    """One Studio stage section with its parameter keys and dropdown options."""

    id: str
    label: str
    description: str = ""
    parameter_keys: list[str] = Field(default_factory=list)
    #: Additive Studio UX metadata: selectable implementations of this stage.
    choices: list[StageChoiceOption] = Field(default_factory=list)


class UniverseRoot(_StudioModel):
    """One instrument-master root mirrored into the Studio universe explorer."""

    root: str
    name: str
    asset_class: str
    sector: str
    sector_group: str
    exchange: str
    multiplier: float
    tick_size: float
    micro_root: str | None = None
    standard_root: str | None = None
    currency: str = "USD"


class UniverseSector(_StudioModel):
    """One sector lane of the Studio universe explorer (``all`` + seven sectors)."""

    id: str
    label: str
    description: str = ""
    count: int = Field(default=0, ge=0)
    roots: list[UniverseRoot] = Field(default_factory=list)


class UniverseCatalog(_StudioModel):
    """Sector-grouped instrument master metadata (mirrors universe_metadata().to_dict())."""

    sectors: list[UniverseSector] = Field(default_factory=list)
    roots: list[UniverseRoot] = Field(default_factory=list)


class StrategyPreset(_StudioModel):
    """A named hypothesis template: a complete ``StrategyConfig`` payload."""

    id: str
    name: str
    description: str = ""
    hypothesis: str = ""
    tags: list[str] = Field(default_factory=list)
    parameter_keys: list[str] = Field(default_factory=list)
    #: Full ``StrategyConfig`` dict, POST-ready (canonical catalog key: ``strategy``).
    strategy: dict[str, object] = Field(default_factory=dict)
    #: Alias of ``strategy`` kept for the earlier API draft / Studio clients.
    config: dict[str, object] = Field(default_factory=dict)


class StrategyCatalogResponse(_StudioModel):
    """``GET /api/strategies/schema/catalog`` payload consumed by the Studio UI."""

    version: int = 1
    stages: list[CatalogStage] = Field(default_factory=list)
    stage_ids: list[str] = Field(default_factory=list)
    #: Additive Studio UX metadata: ``{stage_id: [option_name, ...]}``.
    stage_choices: dict[str, list[str]] = Field(default_factory=dict)
    #: Additive Studio UX metadata: sector-grouped instrument universe explorer data.
    universe: UniverseCatalog = Field(default_factory=UniverseCatalog)
    parameters: list[ParameterSpec] = Field(default_factory=list)
    presets: list[StrategyPreset] = Field(default_factory=list)
    source: str = "fallback"
    preset_source: str = "fallback"
    generated_at: datetime


# ---------------------------------------------------------------------------
# Static fallback catalog + presets
# ---------------------------------------------------------------------------


def _backtest_block() -> dict[str, object]:
    return {
        "start": "2024-01-02",
        "end": "2025-12-31",
        "initial_equity": 1_000_000.0,
        "rebalance_frequency": "W-FRI",
        "random_seed": 7,
    }


def _translator_block(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "rounding_mode": "nearest",
        "rebalance_buffer_contracts": 0.0,
        "max_margin_utilization": 0.60,
        "enable_micro_routing": True,
    }
    payload.update(overrides)
    return payload


#: Static fallback dropdown options when the strategy layer is not importable.
_FALLBACK_STAGE_CHOICES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "universe": ("liquidity", "regime", "composite"),
        "allocator": (
            "volatility_target",
            "risk_parity",
            "equal_weight",
            "mean_variance",
            "drl",
        ),
        "timing": (
            "tsmom",
            "kama",
            "carry_filter",
            "mean_reversion",
            "seasonality",
            "composite",
        ),
        "risk": ("quant_risk", "llm_risk"),
        "translator": ("translator",),
    }
)


def _fallback_stage_choices() -> dict[str, list[str]]:
    return {stage: list(options) for stage, options in _FALLBACK_STAGE_CHOICES.items()}


def _stage_specs() -> list[CatalogStage]:
    definitions = (
        ("universe", "Universe · S_t", "Root eligibility filters."),
        ("allocator", "Allocator · A_t", "Weight generation."),
        ("timing", "Timing & Signals · T_t", "Signal overlays."),
        ("risk", "Risk & LLM · R_t", "Risk projection."),
        ("translator", "Contract Translator", "w_t -> N_i,t rounding/margin."),
    )
    return [
        CatalogStage(
            id=stage_id,
            label=label,
            description=description,
            choices=[
                StageChoiceOption(name=name, label=name.replace("_", " ").title())
                for name in _FALLBACK_STAGE_CHOICES.get(stage_id, ())
            ],
        )
        for stage_id, label, description in definitions
    ]


def _fallback_spec(
    key: str,
    stage: str,
    stage_name: str,
    param: str,
    *,
    type_: ParameterType = "float",
    default: ScalarChoice | list[ScalarChoice] | None = None,
    minimum: float | None = None,
    maximum: float | None = None,
    step: float | None = None,
    choices: Sequence[ScalarChoice] = (),
    label: str = "",
    description: str = "",
    unit: str | None = None,
    optuna_tunable: bool = False,
    optuna_default_selected: bool = False,
    yaml_encoding: str = "scalar",
) -> ParameterSpec:
    return ParameterSpec(
        key=key,
        stage=stage,
        stage_name=stage_name,
        param=param,
        label=label or param.replace("_", " ").title(),
        description=description,
        type=type_,
        default=default,
        min=minimum,
        max=maximum,
        step=step,
        choices=list(choices),
        unit=unit,
        optuna_tunable=optuna_tunable,
        optuna_default_selected=optuna_default_selected,
        yaml_encoding=yaml_encoding,
        path=f"{stage}.params.{param}",
    )


def fallback_parameter_catalog() -> list[ParameterSpec]:
    """Deterministic fallback catalog (used only when the strategy layer is absent)."""
    specs: list[ParameterSpec] = [
        _fallback_spec(
            "universe.liquidity.top_n",
            "universe",
            "liquidity",
            "top_n",
            type_="int",
            default=6,
            minimum=1,
            maximum=20,
            step=1,
            description="S_t: keep only the N most liquid eligible roots.",
            optuna_tunable=False,
        ),
        _fallback_spec(
            "universe.liquidity.min_volume",
            "universe",
            "liquidity",
            "min_volume",
            default=100_000.0,
            minimum=0.0,
            maximum=10_000_000.0,
            step=10_000.0,
            description="S_t: minimum average daily volume filter.",
            unit="contracts",
            optuna_tunable=False,
        ),
        _fallback_spec(
            "allocator.volatility_target.target_annual_volatility",
            "allocator",
            "volatility_target",
            "target_annual_volatility",
            default=0.12,
            minimum=0.02,
            maximum=0.40,
            step=0.01,
            description="A_t: volatility-targeting portfolio risk budget.",
            optuna_tunable=True,
            optuna_default_selected=True,
        ),
        _fallback_spec(
            "allocator.risk_parity.lookback_days",
            "allocator",
            "risk_parity",
            "lookback_days",
            type_="int",
            default=60,
            minimum=20,
            maximum=252,
            step=1,
            description="A_t: covariance lookback for equal-risk contribution.",
            unit="days",
            optuna_tunable=True,
        ),
        _fallback_spec(
            "timing.tsmom.lookback",
            "timing",
            "tsmom",
            "lookback",
            type_="int",
            default=126,
            minimum=10,
            maximum=252,
            step=1,
            description="T_t: TSMOM momentum lookback in trading days.",
            unit="days",
            optuna_tunable=True,
            optuna_default_selected=True,
        ),
        _fallback_spec(
            "timing.mean_reversion.rsi_period",
            "timing",
            "mean_reversion",
            "rsi_period",
            type_="int",
            default=14,
            minimum=5,
            maximum=50,
            step=1,
            description="T_t: RSI period for the panic mean-reversion overlay.",
            unit="bars",
            optuna_tunable=True,
            optuna_default_selected=True,
        ),
        _fallback_spec(
            "timing.mean_reversion.oversold_threshold",
            "timing",
            "mean_reversion",
            "oversold_threshold",
            default=25.0,
            minimum=5.0,
            maximum=45.0,
            step=1.0,
            description="T_t: RSI level that boosts long exposure.",
            optuna_tunable=True,
        ),
        _fallback_spec(
            "timing.seasonality.active_months",
            "timing",
            "seasonality",
            "active_months",
            type_="multiselect",
            choices=tuple(range(1, 13)),
            default=None,
            description="T_t: UTC calendar months in which the seasonal bias is active.",
            yaml_encoding="comma_ints",
        ),
        _fallback_spec(
            "risk.volatility_target.vol_target",
            "risk",
            "volatility_target",
            "vol_target",
            default=0.12,
            minimum=0.02,
            maximum=0.40,
            step=0.01,
            description="R_t: ex-ante portfolio volatility target.",
            optuna_tunable=True,
            optuna_default_selected=True,
        ),
        _fallback_spec(
            "risk.quant_risk.max_single_instrument_weight",
            "risk",
            "quant_risk",
            "max_single_instrument_weight",
            default=0.50,
            minimum=0.05,
            maximum=1.0,
            step=0.05,
            description="R_t: concentration cap per root.",
            optuna_tunable=True,
            optuna_default_selected=True,
        ),
        _fallback_spec(
            "risk.llm_risk.llm_confidence_threshold",
            "risk",
            "llm_risk",
            "llm_confidence_threshold",
            default=0.70,
            minimum=0.0,
            maximum=1.0,
            step=0.01,
            description="R_t: τ minimum LLM confidence before a semantic signal is used.",
            optuna_tunable=True,
        ),
        _fallback_spec(
            "translator.rounding_mode",
            "translator",
            "translator",
            "rounding_mode",
            type_="select",
            default="nearest",
            choices=("floor", "nearest", "ceil"),
            description="N_i,t: fractional contract rounding policy.",
        ),
        _fallback_spec(
            "translator.max_margin_utilization",
            "translator",
            "translator",
            "max_margin_utilization",
            default=0.60,
            minimum=0.10,
            maximum=0.90,
            step=0.01,
            description="Hard margin cushion ceiling η_max.",
            optuna_tunable=True,
        ),
    ]
    return specs


def fallback_presets() -> list[StrategyPreset]:
    """The five Phase-10 hypothesis presets (fallback copy of the strategy layer's)."""
    trend = {
        "id": "trend_following_tsmom_kama",
        "description": "Multi-root TSMOM trend following with KAMA confirmation.",
        "tags": ["trend", "tsmom", "kama"],
        "universe": {"name": "liquidity", "params": {"top_n": 6}},
        "allocator": {"name": "volatility_target", "params": {"target_annual_volatility": 0.12}},
        "timing": [
            {"name": "tsmom", "params": {"lookback": 126}},
            {"name": "kama", "params": {"fast": 2, "slow": 30}},
        ],
        "risk": [
            {"name": "quant_risk", "params": {"mode": "full"}},
            {"name": "llm_risk", "params": {"mode": "enforce"}},
        ],
        "translator": _translator_block(),
        "backtest": _backtest_block(),
    }
    carry = {
        "id": "commodity_carry_term_structure",
        "description": "Carry and term-structure carry harvesting across futures curves.",
        "tags": ["carry", "term-structure", "roll-yield"],
        "universe": {"name": "liquidity", "params": {"top_n": 8}},
        "allocator": {"name": "risk_parity", "params": {"lookback_days": 60}},
        "timing": [
            {"name": "carry_filter", "params": {"min_carry": 0.0}},
            {"name": "tsmom", "params": {"lookback": 63}},
        ],
        "risk": [{"name": "quant_risk", "params": {"mode": "full"}}],
        "translator": _translator_block(),
        "backtest": _backtest_block(),
    }
    mean_reversion = {
        "id": "panic_mean_reversion",
        "description": "Panic mean-reversion on RSI oversold/Bollinger bands with turbulence veto.",
        "tags": ["mean-reversion", "rsi", "bollinger"],
        "universe": {"name": "liquidity", "params": {"top_n": 6}},
        "allocator": {"name": "risk_parity", "params": {"lookback_days": 40}},
        "timing": [
            {
                "name": "mean_reversion",
                "params": {
                    "rsi_period": 14,
                    "oversold_threshold": 25,
                    "overbought_threshold": 75,
                    "bollinger_window": 20,
                    "bollinger_std": 2.0,
                    "reversion_strength": 0.5,
                },
            }
        ],
        "risk": [{"name": "quant_risk", "params": {"mode": "full"}}],
        "translator": _translator_block(),
        "backtest": _backtest_block(),
    }
    seasonal = {
        "id": "seasonal_commodity_cycle",
        "description": "Seasonal commodity cycle tilts with an out-of-season de-risking scale.",
        "tags": ["seasonality", "commodities", "calendar"],
        "universe": {"name": "liquidity", "params": {"top_n": 8}},
        "allocator": {"name": "volatility_target", "params": {"target_annual_volatility": 0.10}},
        "timing": [
            {
                "name": "seasonality",
                "params": {"active_months": "1,2,3,4,5,9,10,11,12", "out_of_season_scale": 0.5},
            }
        ],
        "risk": [{"name": "quant_risk", "params": {"mode": "full"}}],
        "translator": _translator_block(),
        "backtest": _backtest_block(),
    }
    drl = {
        "id": "drl_neural_policy_llm_risk",
        "description": "DRL policy allocator with the LLM Risk-First structural constraint engine.",
        "tags": ["drl", "ppo", "llm-risk"],
        "universe": {"name": "liquidity", "params": {"top_n": 6}},
        "allocator": {"name": "drl", "params": {"algorithm": "PPO"}},
        "timing": [{"name": "tsmom", "params": {"lookback": 126}}],
        "risk": [
            {"name": "quant_risk", "params": {"mode": "full"}},
            {"name": "llm_risk", "params": {"mode": "enforce"}},
        ],
        "translator": _translator_block(),
        "backtest": _backtest_block(),
        "drl": {
            "algorithm": "PPO",
            "policy": "MlpPolicy",
            "total_timesteps": 100_000,
            "learning_rate": 0.0003,
            "gamma": 0.99,
            "random_seed": 7,
        },
    }
    templates: tuple[tuple[str, str, str, dict[str, object], tuple[str, ...]], ...] = cast(
        tuple[tuple[str, str, str, dict[str, object], tuple[str, ...]], ...],
        (
            (
                "trend_following_tsmom_kama",
                "Trend Following (TSMOM + KAMA)",
                "Prices trend; ride multi-month momentum with adaptive smoothing confirmation.",
                trend,
                ("timing.tsmom.lookback", "universe.liquidity.top_n"),
            ),
            (
                "commodity_carry_term_structure",
                "Commodity Carry & Term Structure",
                "Futures curves embed roll yield; harvest backwardation and scale contango.",
                carry,
                ("allocator.risk_parity.lookback_days",),
            ),
            (
                "panic_mean_reversion",
                "Panic Mean-Reversion (RSI + Bollinger)",
                "Liquidity shocks overshoot; buy panic dislocations when turbulence is not extreme.",
                mean_reversion,
                (
                    "timing.mean_reversion.rsi_period",
                    "timing.mean_reversion.oversold_threshold",
                ),
            ),
            (
                "seasonal_commodity_cycle",
                "Seasonal Commodity Cycle",
                "Commodity returns exhibit calendar seasonality; tilt into seasonal months only.",
                seasonal,
                ("timing.seasonality.active_months",),
            ),
            (
                "drl_neural_policy_llm_risk",
                "DRL Neural Policy + LLM Risk",
                "A learned policy can adapt allocation, but semantic macro risk must veto it.",
                drl,
                ("risk.llm_risk.llm_confidence_threshold",),
            ),
        ),
    )
    presets: list[StrategyPreset] = []
    for preset_id, name, hypothesis, payload, keys in templates:
        config = dict(payload)
        config["id"] = preset_id
        presets.append(
            StrategyPreset(
                id=preset_id,
                name=name,
                description=str(payload.get("description", "")),
                hypothesis=hypothesis,
                tags=[str(tag) for tag in cast(Sequence[object], payload.get("tags", ()))],
                parameter_keys=list(keys),
                strategy=config,
                config=config,
            )
        )
    return presets


# ---------------------------------------------------------------------------
# Parameter-catalog adapter (consumes src.strategies.parameter_catalog)
# ---------------------------------------------------------------------------


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


def _pick(mapping: Mapping[str, object], *names: str) -> object:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return None


def _as_scalar_choice(value: object) -> ScalarChoice | None:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    return None


def _sequence_entries(payload: object) -> list[object]:
    if isinstance(payload, Sequence) and not isinstance(payload, str | bytes):
        return list(cast(Sequence[object], payload))
    if isinstance(payload, Mapping):
        mapping = _object_mapping(payload)
        for key in ("parameters", "params", "entries", "items", "catalog", "presets", "stages"):
            if key in mapping:
                return _sequence_entries(mapping[key])
    provider = getattr(payload, "parameters", None)
    if provider is not None and not callable(provider):
        return _sequence_entries(provider)
    return []


_TYPE_ALIASES: Mapping[str, ParameterType] = {
    "int": "int",
    "integer": "int",
    "float": "float",
    "number": "float",
    "double": "float",
    "bool": "bool",
    "boolean": "bool",
    "select": "select",
    "enum": "select",
    "multiselect": "multiselect",
    "multi_select": "multiselect",
}


def _coerce_parameter(entry: object) -> ParameterSpec | None:
    if isinstance(entry, str):
        return None
    mapping = _object_mapping(entry)
    if not mapping:
        return None
    key = str(_pick(mapping, "key", "name", "id") or "").strip()
    if not key:
        return None
    stage = str(_pick(mapping, "stage", "group", "section") or "universe")
    stage_name = str(_pick(mapping, "stage_name", "sub_stage", "overlay") or "")
    param = str(_pick(mapping, "param", "yaml_param", "parameter") or key.split(".")[-1])
    param_type = _TYPE_ALIASES.get(
        str(_pick(mapping, "type", "param_type", "kind") or "float").strip().lower(), "float"
    )
    choices: list[ScalarChoice] = []
    raw_choices = _pick(mapping, "choices", "options", "allowed_values")
    for choice in _sequence_entries(raw_choices):
        scalar = _as_scalar_choice(choice)
        if scalar is not None:
            choices.append(scalar)
    raw_default = _pick(mapping, "default", "value")
    if isinstance(raw_default, Sequence) and not isinstance(raw_default, str | bytes):
        default: ScalarChoice | list[ScalarChoice] | None = [
            scalar
            for scalar in (_as_scalar_choice(item) for item in cast(Sequence[object], raw_default))
            if scalar is not None
        ]
    else:
        default = _as_scalar_choice(raw_default)

    def number(name: str, *aliases: str) -> float | None:
        value = _pick(mapping, name, *aliases)
        if isinstance(value, bool) or not isinstance(value, int | float):
            return None
        result = float(value)
        return result if math.isfinite(result) else None

    tunable = bool(_pick(mapping, "optuna_tunable", "tunable", "optimizable") or False)
    selected = bool(
        _pick(
            mapping,
            "optuna_default_selected",
            "optuna_default",
            "default_selected",
        )
        or False
    )
    return ParameterSpec(
        key=key,
        stage=stage,
        stage_name=stage_name,
        param=param,
        label=str(
            _pick(mapping, "label", "title", "display_name") or param.replace("_", " ").title()
        ),
        description=str(_pick(mapping, "description", "help", "hypothesis") or ""),
        type=param_type,
        default=default,
        min=number("min", "minimum", "low"),
        max=number("max", "maximum", "high"),
        step=number("step", "increment"),
        choices=choices,
        unit=cast("str | None", _pick(mapping, "unit", "suffix")),
        optuna_tunable=tunable,
        optuna_default_selected=selected,
        log_scale=bool(_pick(mapping, "log_scale", "log") or False),
        yaml_encoding=str(_pick(mapping, "yaml_encoding", "encoding") or "scalar"),
        min_items=(
            int(cast(float, value))
            if (value := _pick(mapping, "min_items", "minimum_items")) is not None
            and isinstance(value, int | float)
            and not isinstance(value, bool)
            else None
        ),
        max_items=(
            int(cast(float, value))
            if (value := _pick(mapping, "max_items", "maximum_items")) is not None
            and isinstance(value, int | float)
            and not isinstance(value, bool)
            else None
        ),
        location=str(_pick(mapping, "location", "binding") or "stage_params"),
        path=str(_pick(mapping, "path", "param_path") or f"{stage}.params.{param}"),
    )


def _load_catalog_module() -> object | None:
    try:
        import importlib

        return importlib.import_module(PARAMETER_CATALOG_MODULE)
    except Exception:
        return None


def _catalog_payload(
    module: object, instrument_master: InstrumentsConfig | None = None
) -> Mapping[str, object] | None:
    """``catalog_dict(...)`` payload, built for the current instrument master.

    The strategy layer rebuilds the catalog per instrument-master fingerprint, so
    passing the freshly loaded master keeps universe/root multiselect choices in
    sync with ``configs/instruments.yaml`` (important for a long-running API after
    data-eng adds roots).
    """
    for name in ("catalog_dict", "catalog_json"):
        provider = getattr(module, name, None)
        if not callable(provider):
            continue
        attempts: list[tuple[object, ...]] = (
            [(instrument_master,), ()] if instrument_master is not None else [()]
        )
        for arguments in attempts:
            try:
                payload = provider(*arguments)
            except TypeError:
                continue
            except Exception:  # pragma: no cover - defensive adapter
                payload = None
                break
            mapping = _object_mapping(payload)
            if mapping:
                return mapping
    provider = getattr(module, "parameter_catalog", None)
    if callable(provider):
        for arguments in [(instrument_master,), ()] if instrument_master is not None else [()]:
            try:
                candidate = provider(*arguments)
            except TypeError:
                continue
            except Exception:  # pragma: no cover - defensive adapter
                candidate = None
                break
            dumper = getattr(candidate, "to_dict", None)
            if callable(dumper):
                mapping = _object_mapping(dumper())
                if mapping:
                    return mapping
    for name in ("CATALOG", "catalog", "PARAMETER_CATALOG"):
        candidate = getattr(module, name, None)
        if candidate is None:
            continue
        dumper = getattr(candidate, "to_dict", None)
        if callable(dumper):
            try:
                mapping = _object_mapping(dumper())
            except Exception:  # pragma: no cover - defensive adapter
                continue
            if mapping:
                return mapping
    return None


def load_parameter_catalog(
    instrument_master: InstrumentsConfig | None = None,
) -> tuple[list[ParameterSpec], str]:
    """Load the strategy layer's typed catalog, falling back deterministically."""
    module = _load_catalog_module()
    if module is not None:
        payload = _catalog_payload(module, instrument_master)
        if payload is not None:
            specs = [
                spec
                for spec in (
                    _coerce_parameter(entry)
                    for entry in _sequence_entries(payload.get("parameters"))
                )
                if spec is not None
            ]
            if specs:
                return specs, PARAMETER_CATALOG_MODULE
    return fallback_parameter_catalog(), "fallback"


def _stage_options(mapping: Mapping[str, object]) -> list[StageChoiceOption]:
    options: list[StageChoiceOption] = []
    for entry in _sequence_entries(mapping.get("choices")):
        option = _object_mapping(entry)
        name = str(_pick(option, "name", "id", "key") or "").strip()
        if not name:
            continue
        options.append(
            StageChoiceOption(
                name=name,
                label=str(_pick(option, "label", "title") or name.replace("_", " ").title()),
                description=str(_pick(option, "description") or ""),
                parameter_keys=[
                    str(item)
                    for item in _sequence_entries(_pick(option, "parameter_keys") or ())
                    if isinstance(item, str)
                ],
            )
        )
    return options


def _stages_from_payload(payload: Mapping[str, object]) -> list[CatalogStage]:
    stages: list[CatalogStage] = []
    for entry in _sequence_entries(payload.get("stages")):
        mapping = _object_mapping(entry)
        stage_id = str(_pick(mapping, "id", "key", "name") or "").strip()
        if not stage_id:
            continue
        stages.append(
            CatalogStage(
                id=stage_id,
                label=str(_pick(mapping, "label", "name") or stage_id.title()),
                description=str(_pick(mapping, "description") or ""),
                parameter_keys=[
                    str(item)
                    for item in _sequence_entries(_pick(mapping, "parameter_keys") or ())
                    if isinstance(item, str)
                ],
                choices=_stage_options(mapping),
            )
        )
    return stages


def _stage_choices_from_payload(payload: Mapping[str, object]) -> dict[str, list[str]]:
    raw = payload.get("stage_choices")
    choices: dict[str, list[str]] = {}
    if isinstance(raw, Mapping):
        for stage_id, values in cast(Mapping[object, object], raw).items():
            key = str(stage_id).strip()
            if not key:
                continue
            names: list[str] = []
            for entry in _sequence_entries(values):
                if isinstance(entry, str):
                    names.append(entry)
                    continue
                mapping = _object_mapping(entry)
                name = str(_pick(mapping, "name", "id", "key") or "").strip()
                if name:
                    names.append(name)
            if names:
                choices[key] = names
    return choices


def _universe_root(entry: object) -> UniverseRoot | None:
    mapping = _object_mapping(entry)
    root = str(_pick(mapping, "root", "instrument_id", "symbol") or "").strip().upper()
    if not root:
        return None
    multiplier = _pick(mapping, "multiplier")
    tick_size = _pick(mapping, "tick_size")
    return UniverseRoot(
        root=root,
        name=str(_pick(mapping, "name", "description") or root),
        asset_class=str(_pick(mapping, "asset_class") or "unknown"),
        sector=str(_pick(mapping, "sector") or "").lower(),
        sector_group=str(_pick(mapping, "sector_group") or "").lower(),
        exchange=str(_pick(mapping, "exchange") or ""),
        multiplier=float(multiplier) if isinstance(multiplier, int | float) else 0.0,
        tick_size=float(tick_size) if isinstance(tick_size, int | float) else 0.0,
        micro_root=cast("str | None", _pick(mapping, "micro_root")),
        standard_root=cast("str | None", _pick(mapping, "standard_root")),
        currency=str(_pick(mapping, "currency") or "USD"),
    )


def _universe_roots(entries: object) -> list[UniverseRoot]:
    roots: list[UniverseRoot] = []
    for entry in _sequence_entries(entries):
        root = _universe_root(entry)
        if root is not None:
            roots.append(root)
    return roots


def _universe_from_payload(payload: Mapping[str, object]) -> UniverseCatalog:
    raw = _pick(payload, "universe")
    universe = _object_mapping(raw)
    sectors: list[UniverseSector] = []
    for entry in _sequence_entries(universe.get("sectors")):
        mapping = _object_mapping(entry)
        sector_id = str(_pick(mapping, "id", "key", "name") or "").strip().lower()
        if not sector_id:
            continue
        roots = _universe_roots(mapping.get("roots"))
        sectors.append(
            UniverseSector(
                id=sector_id,
                label=str(_pick(mapping, "label", "name") or sector_id.title()),
                description=str(_pick(mapping, "description") or ""),
                count=len(roots)
                if not isinstance(_pick(mapping, "count"), int)
                else int(cast(int, _pick(mapping, "count"))),
                roots=roots,
            )
        )
    flat = _universe_roots(universe.get("roots"))
    if not flat:
        flat = [root for sector in sectors for root in sector.roots]
    return UniverseCatalog(sectors=sectors, roots=flat)


#: Canonical Studio sector groups for the master-derived fallback universe.
_SECTOR_GROUP_ORDER: tuple[str, ...] = (
    "equities",
    "rates",
    "energy",
    "metals",
    "fx",
    "agri",
    "crypto",
)
_SECTOR_GROUP_LABELS: Mapping[str, str] = {
    "equities": "Equities",
    "rates": "Rates",
    "energy": "Energy",
    "metals": "Metals",
    "fx": "FX",
    "agri": "Agriculture",
    "crypto": "Crypto",
}
_SECTOR_GROUP_ALIASES: Mapping[str, str] = {
    "equity": "equities",
    "equities": "equities",
    "equity_index": "equities",
    "index": "equities",
    "rates": "rates",
    "rate": "rates",
    "interest_rates": "rates",
    "energy": "energy",
    "metals": "metals",
    "metal": "metals",
    "fx": "fx",
    "forex": "fx",
    "currency": "fx",
    "agri": "agri",
    "agriculture": "agri",
    "grains": "agri",
    "crypto": "crypto",
    "cryptocurrency": "crypto",
}


def _sector_group_for(spec: InstrumentSpec) -> str:
    raw = str(getattr(spec, "sector", "") or "").strip().lower()
    resolved = _SECTOR_GROUP_ALIASES.get(raw)
    if resolved is not None:
        return resolved
    asset_class = str(spec.asset_class).lower()
    return _SECTOR_GROUP_ALIASES.get(asset_class, "other")


def _universe_from_master(master: InstrumentsConfig) -> UniverseCatalog:
    """Fallback universe metadata derived directly from the instrument master."""
    groups: dict[str, list[UniverseRoot]] = {sector: [] for sector in _SECTOR_GROUP_ORDER}
    for spec in master.instruments.values():
        group = _sector_group_for(spec)
        groups.setdefault(group, []).append(
            UniverseRoot(
                root=spec.root,
                name=spec.name,
                asset_class=str(spec.asset_class),
                sector=str(spec.sector).lower(),
                sector_group=group,
                exchange=spec.exchange,
                multiplier=spec.multiplier,
                tick_size=spec.tick_size,
                micro_root=spec.micro_root,
                standard_root=spec.standard_root,
                currency=spec.currency,
            )
        )
    flat = sorted((root for roots in groups.values() for root in roots), key=lambda row: row.root)
    sectors = [
        UniverseSector(
            id="all",
            label="All",
            description="Every futures root in the instrument master",
            count=len(flat),
            roots=flat,
        )
    ]
    for sector in (*_SECTOR_GROUP_ORDER, "other"):
        roots = sorted(groups.get(sector, []), key=lambda row: row.root)
        if sector == "other" and not roots:
            continue
        sectors.append(
            UniverseSector(
                id=sector,
                label=_SECTOR_GROUP_LABELS.get(sector, sector.title()),
                description="",
                count=len(roots),
                roots=roots,
            )
        )
    return UniverseCatalog(sectors=sectors, roots=flat)


#: Path -> (mtime, master) cache: catalog universe follows the instrument YAML live.
_MASTER_CACHE: dict[str, tuple[float, InstrumentsConfig]] = {}
_MASTER_LOCK = threading.Lock()


def load_current_instrument_master(settings: Settings) -> InstrumentsConfig:
    """Instrument master, refreshed when the YAML changed on disk."""
    path = Path(settings.instruments_path or (settings.configs_dir / "instruments.yaml"))
    try:
        mtime = path.stat().st_mtime
    except OSError:  # pragma: no cover - committed master always exists
        return settings.instruments
    key = str(path.resolve())
    with _MASTER_LOCK:
        cached = _MASTER_CACHE.get(key)
        if cached is not None and cached[0] == mtime:
            return cached[1]
        clear_instruments_cache()
        master = load_instruments(path)
        _MASTER_CACHE[key] = (mtime, master)
        return master


def load_stages(instrument_master: InstrumentsConfig | None = None) -> list[CatalogStage]:
    """Load stage metadata (with dropdown options) from the strategy layer."""
    module = _load_catalog_module()
    if module is not None:
        payload = _catalog_payload(module, instrument_master)
        if payload is not None:
            stages = _stages_from_payload(payload)
            if stages:
                return stages
    return _stage_specs()


def load_stage_choices(
    instrument_master: InstrumentsConfig | None = None,
) -> dict[str, list[str]]:
    """Load ``{stage_id: [option_name, ...]}`` from the strategy layer."""
    module = _load_catalog_module()
    if module is not None:
        payload = _catalog_payload(module, instrument_master)
        if payload is not None:
            choices = _stage_choices_from_payload(payload)
            if choices:
                return choices
    return _fallback_stage_choices()


def load_universe(instrument_master: InstrumentsConfig | None = None) -> UniverseCatalog:
    """Load sector-grouped instrument metadata from the strategy layer or master."""
    module = _load_catalog_module()
    if module is not None:
        payload = _catalog_payload(module, instrument_master)
        if payload is not None:
            universe = _universe_from_payload(payload)
            if universe.roots:
                return universe
    master = instrument_master if instrument_master is not None else load_instruments()
    return _universe_from_master(master)


def load_presets(
    instrument_master: InstrumentsConfig | None = None,
) -> tuple[list[StrategyPreset], str]:
    """Load the strategy layer's hypothesis presets, falling back deterministically."""
    module = _load_catalog_module()
    if module is not None:
        payload = _catalog_payload(module, instrument_master)
        if payload is not None:
            presets = _presets_from_entries(_sequence_entries(payload.get("presets")))
            if presets:
                return presets, PARAMETER_CATALOG_MODULE
        for name in (
            "presets",
            "hypothesis_templates",
            "strategy_presets",
            "default_presets",
            "build_presets",
        ):
            provider = getattr(module, name, None)
            if provider is None:
                continue
            candidate: object = provider() if callable(provider) else provider
            presets = _presets_from_entries(_sequence_entries(candidate))
            if presets:
                return presets, PARAMETER_CATALOG_MODULE
    return fallback_presets(), "fallback"


def _presets_from_entries(entries: Sequence[object]) -> list[StrategyPreset]:
    presets: list[StrategyPreset] = []
    for entry in entries:
        mapping = _object_mapping(entry)
        config = _object_mapping(_pick(mapping, "strategy", "config", "payload") or {})
        preset_id = str(_pick(mapping, "id", "key", "name") or "").strip().lower()
        if not preset_id or not config:
            continue
        presets.append(
            StrategyPreset(
                id=preset_id,
                name=str(
                    _pick(mapping, "name", "label", "title") or preset_id.replace("_", " ").title()
                ),
                description=str(_pick(mapping, "description") or ""),
                hypothesis=str(_pick(mapping, "hypothesis", "rationale") or ""),
                tags=[
                    str(tag)
                    for tag in _sequence_entries(_pick(mapping, "tags") or ())
                    if isinstance(tag, str)
                ],
                parameter_keys=[
                    str(item)
                    for item in _sequence_entries(_pick(mapping, "parameter_keys") or ())
                    if isinstance(item, str)
                ],
                strategy=dict(config),
                config=dict(config),
            )
        )
    return presets


# ---------------------------------------------------------------------------
# Dotted-path parameter application (Optuna best_params -> StrategyConfig)
# ---------------------------------------------------------------------------


def _config_mapping(config: StrategyConfig) -> dict[str, object]:
    return config.model_dump(mode="json")


def resolve_optimizer_path(strategy: Mapping[str, object], key_or_path: str) -> str:
    """Resolve a catalog parameter key to a concrete dotted path in ``strategy``.

    Uses the strategy layer's ``resolve_param_path`` when importable (it knows the
    actual timing/risk list indices); otherwise catalog keys are passed through as
    concrete dotted paths and validated by :func:`apply_dotted_params`.
    """
    module = _load_catalog_module()
    resolver = getattr(module, "resolve_param_path", None) if module is not None else None
    if callable(resolver):
        try:
            resolved = resolver(dict(strategy), key_or_path)
        except Exception:
            resolved = None
        if isinstance(resolved, str) and resolved:
            return resolved
    return key_or_path


def apply_dotted_params(
    config: StrategyConfig,
    params: Mapping[str, object],
    *,
    strategy_id: str | None = None,
    allow_new_params: bool = False,
) -> StrategyConfig:
    """Apply ``{"timing.1.params.lookback": 90}`` style params and re-validate.

    Lists are addressed by integer segments (``timing.1.params``); scalar stages
    accept a direct stage prefix (``translator.max_margin_utilization``). Unknown
    paths raise :class:`StrategyStudioError` instead of silently dropping a tuned
    parameter. When ``allow_new_params`` is true, a leaf nested directly inside a
    stage's ``params`` mapping may be created (the optimizer tunes catalog-default
    parameters such as ``quant_risk.trailing_stop_pct`` that the committed YAML may
    omit); unknown stages/indices and non-params paths still raise.
    """
    payload = _config_mapping(config)
    for path, value in params.items():
        if not isinstance(path, str) or not path.strip():
            raise StrategyStudioError(f"invalid parameter path {path!r}")
        segments = path.strip().split(".")
        cursor: object = payload
        for index, segment in enumerate(segments):
            last = index == len(segments) - 1
            if isinstance(cursor, list):
                if not segment.isdigit() or int(segment) >= len(cursor):
                    raise StrategyStudioError(f"parameter path {path!r} does not exist")
                position = int(segment)
                if last:
                    cursor[position] = value
                else:
                    cursor = cursor[position]
                continue
            if not isinstance(cursor, dict):
                raise StrategyStudioError(f"parameter path {path!r} does not exist")
            mapping = cast(dict[str, object], cursor)
            if segment not in mapping:
                if allow_new_params and last and index > 0 and segments[index - 1] == "params":
                    mapping[segment] = value
                    break
                raise StrategyStudioError(
                    f"parameter path {path!r} does not exist in the base strategy"
                )
            if last:
                mapping[segment] = value
            else:
                cursor = mapping[segment]
    if strategy_id is not None:
        payload["id"] = strategy_id
    try:
        return parse_strategy_config(payload)
    except (ValidationError, ValueError, TypeError) as exc:
        raise StrategyStudioError(
            f"tuned parameters are not a valid strategy config: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SavedStrategy:
    """A validated strategy config persisted through the Studio service.

    ``path`` is the exact YAML written: in test/backtest environments this is the
    ``artifacts/strategies`` overlay, while development/paper/live environments
    write new strategies to the committed ``configs/strategies`` tree so they are
    visible in the Studio immediately.
    """

    view: StrategyConfigView
    config: StrategyConfig
    path: Path

    @property
    def strategy_id(self) -> str:
        """Persisted strategy id (same as ``config.id``)."""
        return self.config.id


@dataclass(frozen=True, slots=True)
class _DiscoveredStrategy:
    path: Path
    config: StrategyConfig
    builtin: bool


class StrategyStudioService:
    """Parameter catalog + strategy YAML persistence for the Strategy Studio."""

    def __init__(
        self,
        *,
        settings: Settings,
        clock: Clock | None = None,
        parameter_catalog_provider: Callable[[], object] | None = None,
        preset_provider: Callable[[], object] | None = None,
        builtin_dir: Path | None = None,
        custom_dir: Path | None = None,
        catalog: object | None = None,
    ) -> None:
        self._settings = settings
        self._clock: Clock = clock if clock is not None else _utc_now
        self._parameter_provider = parameter_catalog_provider
        self._preset_provider = preset_provider
        self._catalog = catalog
        self._builtin_dir = builtin_dir or (settings.configs_dir / "strategies")
        default_custom = getattr(
            settings, "strategy_artifacts_dir", settings.report_dir / "strategies"
        )
        self._custom_dir = custom_dir or Path(default_custom)
        self._builtin_writes = settings.environment in _BUILTIN_WRITE_ENVIRONMENTS

    # -- paths / safety --------------------------------------------------------

    @property
    def builtin_dir(self) -> Path:
        """Directory of committed built-in strategy YAMLs."""
        return self._builtin_dir

    @property
    def custom_dir(self) -> Path:
        """Directory of API-created custom strategy YAMLs."""
        return self._custom_dir

    @property
    def builtin_writes_enabled(self) -> bool:
        """Whether built-in YAMLs may be overwritten in this environment."""
        return self._builtin_writes

    @staticmethod
    def validate_strategy_id(strategy_id: object) -> str:
        """Validate + normalise a strategy id, rejecting traversal/separators."""
        if not isinstance(strategy_id, str):
            raise StrategyStudioError("strategy id must be a string", status_code=400)
        candidate = strategy_id.strip()
        if not _STRATEGY_ID_PATTERN.match(candidate):
            raise StrategyStudioError(
                "invalid strategy id: must match ^[a-z][a-z0-9_]{0,63}$ "
                "(lower-case letters, digits and underscores; no path separators or '..')",
                status_code=400,
            )
        return candidate

    def _safe_target(self, base: Path, strategy_id: str) -> Path:
        target = (base / f"{strategy_id}.yaml").resolve()
        resolved_base = base.resolve()
        if resolved_base not in target.parents:
            raise StrategyStudioError(
                "refusing to write outside the strategy directory", status_code=400
            )
        return target

    def _refresh_catalog(self, config: StrategyConfig) -> None:
        """Mirror a persisted config into the runtime strategy catalog (in memory).

        Keeps a newly created/tuned strategy immediately runnable by the backtest
        service without restarting the API. Disk state (committed built-ins) is
        never mutated here; only the in-memory registry changes.
        """
        upsert = getattr(self._catalog, "upsert", None)
        if not callable(upsert):
            return
        try:
            upsert(config)
        except Exception:  # pragma: no cover - defensive: catalog must not wedge writes
            logger.exception("strategy catalog refresh failed for %s", config.id)

    # -- catalog ---------------------------------------------------------------

    def catalog(self) -> StrategyCatalogResponse:
        """Typed parameter catalog, stage/universe metadata and the five presets.

        The third-party strategy layer is queried with the *current* instrument
        master (mtime-refreshed), so the universe explorer and root/sector
        multiselect choices follow ``configs/instruments.yaml`` without an API
        restart. Additive Studio-UX fields (``stage_choices``, ``stages[].choices``,
        ``universe``) are passed through unchanged when the layer provides them;
        otherwise they are derived from the instrument master / static fallback.
        """
        master = load_current_instrument_master(self._settings)
        module = _load_catalog_module()
        payload = _catalog_payload(module, master) if module is not None else None
        if self._parameter_provider is not None:
            specs = self._specs_from_payload(self._parameter_provider())
            source = "injected"
        elif payload is not None:
            specs = [
                spec
                for spec in (
                    _coerce_parameter(entry)
                    for entry in _sequence_entries(payload.get("parameters"))
                )
                if spec is not None
            ]
            if specs:
                source = PARAMETER_CATALOG_MODULE
            else:  # pragma: no cover - defensive: catalog without parameters
                specs, source = fallback_parameter_catalog(), "fallback"
        else:
            specs, source = fallback_parameter_catalog(), "fallback"

        if self._preset_provider is not None:
            presets = _presets_from_entries(_sequence_entries(self._preset_provider()))
            preset_source = "injected"
        elif payload is not None:
            presets = _presets_from_entries(_sequence_entries(payload.get("presets")))
            if presets:
                preset_source = PARAMETER_CATALOG_MODULE
            else:  # pragma: no cover - defensive: catalog without presets
                presets, preset_source = fallback_presets(), "fallback"
        else:
            presets, preset_source = fallback_presets(), "fallback"

        stages = _stages_from_payload(payload) if payload is not None else []
        if not stages:
            stages = _stage_specs()
        stage_choices = _stage_choices_from_payload(payload) if payload is not None else {}
        if not stage_choices:
            stage_choices = _fallback_stage_choices()
        universe = _universe_from_payload(payload) if payload is not None else UniverseCatalog()
        if not universe.roots:
            universe = _universe_from_master(master)
        return StrategyCatalogResponse(
            version=1,
            stages=stages,
            stage_ids=[stage.id for stage in stages],
            stage_choices=stage_choices,
            universe=universe,
            parameters=specs,
            presets=presets,
            source=source,
            preset_source=preset_source,
            generated_at=self._clock(),
        )

    @staticmethod
    def _specs_from_payload(payload: object) -> list[ParameterSpec]:
        specs: list[ParameterSpec] = []
        for entry in _sequence_entries(payload):
            spec = _coerce_parameter(entry)
            if spec is not None:
                specs.append(spec)
        return specs

    # -- discovery -------------------------------------------------------------

    def _parse_file(self, path: Path) -> StrategyConfig | None:
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            return None
        if not isinstance(payload, Mapping):
            return None
        try:
            return parse_strategy_config(dict(cast(Mapping[str, object], payload)))
        except (ValidationError, ValueError, TypeError):
            return None

    def _discover(self) -> dict[str, _DiscoveredStrategy]:
        discovered: dict[str, _DiscoveredStrategy] = {}
        for directory, builtin in ((self._builtin_dir, True), (self._custom_dir, False)):
            if not directory.is_dir():
                continue
            for path in sorted(directory.glob("*.yaml")):
                config = self._parse_file(path)
                if config is None:
                    continue
                discovered[config.id] = _DiscoveredStrategy(
                    path=path, config=config, builtin=builtin
                )
        return discovered

    def _find(self, strategy_id: str) -> _DiscoveredStrategy | None:
        return self._discover().get(strategy_id)

    # -- views -----------------------------------------------------------------

    @staticmethod
    def _stage_view(stage: object) -> StageConfig:
        mapping = _object_mapping(stage)
        params: dict[str, bool | int | float | str] = {}
        raw_params = mapping.get("params")
        if isinstance(raw_params, Mapping):
            for key, value in cast(Mapping[object, object], raw_params).items():
                if isinstance(value, bool | int | float | str):
                    params[str(key)] = value
        return StageConfig(name=str(mapping.get("name") or ""), params=params)

    def view_for(self, config: StrategyConfig, *, target: Path | None = None) -> StrategyConfigView:
        """API view of a validated strategy config."""
        if target is not None and target.is_file():
            updated_at = datetime.fromtimestamp(target.stat().st_mtime, tz=UTC)
        else:
            found = self._find(config.id)
            updated_at = (
                datetime.fromtimestamp(found.path.stat().st_mtime, tz=UTC)
                if found is not None
                else self._clock()
            )
        return StrategyConfigView(
            id=config.id,
            description=config.description,
            tags=list(config.tags),
            universe=self._stage_view(config.universe),
            allocator=self._stage_view(config.allocator),
            timing=[self._stage_view(stage) for stage in config.timing],
            risk=[self._stage_view(stage) for stage in config.risk],
            updated_at=updated_at,
        )

    # -- CRUD ------------------------------------------------------------------

    def list_configs(self) -> list[StrategyConfigView]:
        """Every strategy (built-ins + custom overlays), sorted by id."""
        discovered = self._discover()
        return [self.view_for(discovered[key].config) for key in sorted(discovered)]

    def get_config(self, strategy_id: str) -> StrategyConfigView | None:
        """One strategy view or ``None``."""
        core = self.core_config(strategy_id)
        return None if core is None else self.view_for(core)

    def core_config(self, strategy_id: str) -> StrategyConfig | None:
        """Validated strategy config or ``None`` (used by optimizer/backtests)."""
        try:
            key = self.validate_strategy_id(strategy_id)
        except StrategyStudioError:
            return None
        found = self._find(key)
        return None if found is None else found.config

    def strategy_definition(self, strategy_id: str) -> dict[str, object] | None:
        """Serialised YAML-ready config dict for ``strategy_id`` (for the optimizer)."""
        config = self.core_config(strategy_id)
        return None if config is None else _config_mapping(config)

    def create_strategy(self, payload: Mapping[str, object]) -> StrategyConfigView:
        """Validate a full StrategyConfig payload and persist it as a new YAML."""
        if len(repr(payload).encode("utf-8")) > MAX_STRATEGY_PAYLOAD_BYTES:
            raise StrategyStudioError("strategy payload is too large", status_code=413)
        raw_id = payload.get("id")
        if raw_id is None:
            raise StrategyStudioError("strategy payload requires an 'id'", status_code=422)
        strategy_id = self.validate_strategy_id(raw_id)
        if self._find(strategy_id) is not None:
            raise StrategyStudioError(
                f"strategy {strategy_id!r} already exists; use PUT to update it",
                status_code=409,
            )
        config = self._validate_payload({**dict(payload), "id": strategy_id})
        target = self._write_config(config, builtin=False)
        self._refresh_catalog(config)
        return self.view_for(config, target=target)

    def update_strategy(
        self, strategy_id: str, payload: Mapping[str, object]
    ) -> StrategyConfigView:
        """Merge a partial/full payload into an existing strategy and persist it."""
        key = self.validate_strategy_id(strategy_id)
        current = self._find(key)
        if current is None:
            raise StrategyStudioError(f"unknown strategy {key!r}", status_code=404)
        if payload.get("id") is not None:
            payload_id = str(payload.get("id")).strip().lower()
            if payload_id != key:
                raise StrategyStudioError(
                    f"strategy id in payload ({payload_id!r}) does not match the path ({key!r})",
                    status_code=422,
                )
        merged = _config_mapping(current.config)
        for field, value in payload.items():
            if field != "id":
                merged[field] = value
        config = self._validate_payload({**merged, "id": key})
        target = self._write_config(config, builtin=current.builtin)
        self._refresh_catalog(config)
        return self.view_for(config, target=target)

    def _validate_payload(self, payload: Mapping[str, object]) -> StrategyConfig:
        try:
            return parse_strategy_config(dict(payload))
        except (ValidationError, ValueError, TypeError) as exc:
            raise StrategyStudioError(f"invalid strategy config: {exc}") from exc

    def save_config(self, config: StrategyConfig, *, builtin: bool = False) -> Path:
        """Persist a validated config (atomic) and return the YAML path."""
        return self._write_config(config, builtin=builtin)

    def _write_config(self, config: StrategyConfig, *, builtin: bool) -> Path:
        """Persist one config atomically and return its YAML path.

        In development/paper/live the target is ``configs/strategies/<id>.yaml`` for
        every write (including new Studio-authored/optimizer-applied strategies), so
        saved strategies are visible in the committed config tree. Test/backtest
        environments keep writing ``artifacts/strategies`` overlays (and never touch
        committed built-ins).
        """
        self.validate_strategy_id(config.id)
        _ = builtin  # target directory is environment-driven, not caller-driven
        use_builtin = self._builtin_writes
        base = self._builtin_dir if use_builtin else self._custom_dir
        target = self._safe_target(base, config.id)
        serialised = yaml.safe_dump(
            config.model_dump(mode="json"), sort_keys=False, allow_unicode=True
        )
        encoded = serialised.encode("utf-8")
        if len(encoded) > MAX_STRATEGY_PAYLOAD_BYTES:
            raise StrategyStudioError("serialised strategy YAML is too large", status_code=413)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{config.id}.", suffix=".yaml.tmp", dir=str(target.parent)
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            Path(temp_name).replace(target)
        except BaseException:
            Path(temp_name).unlink(missing_ok=True)
            raise
        return target

    # -- optimization apply ----------------------------------------------------

    def save_best_params(
        self,
        base_strategy_id: str,
        best_params: Mapping[str, object],
        *,
        strategy_id: str | None = None,
        suffix: str = "_opt",
        description: str | None = None,
    ) -> SavedStrategy:
        """Create + persist a new strategy from ``best_params`` over the base config.

        Returns the typed view/config together with the exact YAML path written and
        refreshes the in-memory strategy catalog immediately (so ``GET
        /api/strategies`` and ``POST /api/backtests`` see the new id without a
        restart). Existing ids are rejected with a clear 409 on repeat apply.
        """
        base = self.core_config(base_strategy_id)
        if base is None:
            raise StrategyStudioError(
                f"unknown base strategy {base_strategy_id!r}", status_code=404
            )
        candidate_id = self.validate_strategy_id(strategy_id or f"{base.id}{suffix}")
        if self._find(candidate_id) is not None:
            raise StrategyStudioError(f"strategy {candidate_id!r} already exists", status_code=409)
        tuned = apply_dotted_params(
            base, best_params, strategy_id=candidate_id, allow_new_params=True
        )
        tuned = tuned.model_copy(
            update={
                "description": description
                or f"{base.description or base.id} (Optuna-tuned from {base.id})".strip(),
                "tags": tuple(dict.fromkeys((*base.tags, "optuna-tuned"))),
            }
        )
        target = self._write_config(tuned, builtin=False)
        self._refresh_catalog(tuned)
        return SavedStrategy(view=self.view_for(tuned, target=target), config=tuned, path=target)

    def apply_best_params(
        self,
        base_strategy_id: str,
        best_params: Mapping[str, object],
        *,
        strategy_id: str | None = None,
        suffix: str = "_opt",
        description: str | None = None,
    ) -> tuple[StrategyConfigView, StrategyConfig]:
        """Backwards-compatible wrapper around :meth:`save_best_params`."""
        saved = self.save_best_params(
            base_strategy_id,
            best_params,
            strategy_id=strategy_id,
            suffix=suffix,
            description=description,
        )
        return saved.view, saved.config


def numeric_default(value: object, fallback: float) -> float:
    """Finite numeric fallback helper (used by the fixture optimizer)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return fallback
    result = float(value)
    return result if math.isfinite(result) else fallback


#: Immutable view of the fallback catalog for callers that only need metadata.
FALLBACK_PARAMETERS: Mapping[str, ParameterSpec] = MappingProxyType(
    {spec.key: spec for spec in fallback_parameter_catalog()}
)
