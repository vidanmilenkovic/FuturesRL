"""Typed parameter catalogue for the Phase-10 Strategy Studio and Optuna optimizer.

The catalogue is the single machine-readable description of *every* configurable
parameter of the FinRL-X strategy pipeline (universe, allocator, timing, risk and
contract translator) plus the five Strategy Studio hypothesis-template presets.
It is consumed by:

* ``GET /api/strategies/schema/catalog`` (FastAPI) — the Studio renders sliders,
  number inputs, toggles and multiselect pills from it;
* the walk-forward Optuna optimizer — it filters ``optuna_tunable`` entries and
  resolves each catalog key to a concrete dotted path inside a strategy config;
* the YAML writer — ``yaml_encoding`` tells the API how to encode a UI value into
  a ``configs.strategy.StageConfig.params`` scalar (strategy stage params are
  ``float | int | str | bool``; multiselect/object values are encoded as
  comma-separated strings or JSON object strings and parsed by the stage
  builders).

Stable identifiers
------------------
``ParameterSpec.key`` is stable and human-readable: ``<stage>.<stage_name>.<param>``
(e.g. ``"timing.tsmom.lookback_fast"``, ``"risk.llm_risk.confidence_threshold"``,
``"allocator.drl.total_timesteps"``). ``launch``/``stage_name`` bind the parameter
to a pipeline stage so :func:`resolve_param_path` can find the concrete dotted
path inside any strategy dict (timing/risk are ordered lists, so the index is
resolved dynamically from the stage name).

Dynamic instrument master / Studio dropdowns
-------------------------------------------
The catalog is **not** a static snapshot of the instrument master: root choices
(``universe.regime.risk_off_roots``) and
sector choices are rebuilt from :func:`configs.instruments.load_instruments`, and
:func:`parameter_catalog` caches one catalog per *instrument-master fingerprint*
(roots + sectors + per-root contract specs). Adding futures roots/sectors to
``configs/instruments.yaml`` therefore yields updated catalog choices without a
code change; :func:`refresh_parameter_catalog` forces a reload after the file
changes, and callers may pass an explicit ``instrument_master`` instead.

The payload also carries the Studio's universe explorer and stage dropdowns:
:func:`universe_metadata` (sector-grouped roots with multiplier/tick/micro data,
sector groups All/Equities/Rates/Energy/Metals/FX/Agri/Crypto) and per-stage
``choices`` (universe selectors, allocators, timing overlays, risk overlays).

Optuna selection & active-stage filtering
-----------------------------------------
Each parameter carries ``optuna_tunable`` (may it be sampled at all) and
``optuna_default_selected`` (pre-checked in the Studio optimizer). Exactly
:data:`DEFAULT_SELECTED_PARAMETER_KEYS` (six curated core hypotheses) are
pre-selected; structural/data-hygiene parameters in
:data:`NON_TUNABLE_PARAMETER_KEYS` (universe composition/history, regime
switches, macro-lockout plumbing, risk enable toggles, translator execution
details) are never tunable. :func:`active_parameter_keys`,
:func:`parameters_for_strategy` and :func:`default_selected_parameter_keys`
filter to the *active* stage choices/overlays of a concrete strategy (inactive
selectors/overlays are excluded), and :func:`validate_strategy_roots` /
:func:`strategy_root_symbols` guard root lists (regime/universe)
against roots that are not in the instrument master.

Helper API (also the contract reported to the API/optimizer owners)
-------------------------------------------------------------------
* :func:`parameter_catalog` — cached :class:`ParameterCatalog` singleton
  (``CATALOG`` is the same object as a module constant).
* :func:`catalog_dict` — JSON-ready payload for the REST endpoint.
* :func:`resolve_param_path` / :func:`get_parameter_value` — read bindings.
* :func:`validate_parameter_value` — partial-update validation (type, range,
  choices, cardinality); raises ``ValueError`` naming the key.
* :func:`encode_parameter_value` / :func:`decode_parameter_value` — YAML↔UI
  conversions (comma strings, JSON object strings).
* :func:`apply_parameter_overrides` — deep-copy a strategy dict and apply
  UI-native overrides as YAML-native values.

Only hyperparameters are catalogued; runtime artefacts such as
``DRLSpec.model_path`` (trained checkpoint path) are intentionally excluded —
they are run configuration, not tunable parameters.

Import note: this module and the cached ``parameter_catalog()`` callable share a
name, and ``src.strategies`` re-exports the callable. Use
``from src.strategies.parameter_catalog import catalog_dict, parameter_catalog``
(or ``importlib.import_module("src.strategies.parameter_catalog")``); the
``import src.strategies.parameter_catalog as x`` form binds the *function* from
the package namespace instead of the module.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from configs.instruments import InstrumentsConfig, InstrumentSpec, load_instruments
from configs.strategy import StrategyConfig

#: Catalogue schema version (bumped on incompatible JSON changes).
CATALOG_VERSION: int = 1

#: YAML encoding identifiers for UI → ``StageConfig.params`` serialisation.
ENCODING_SCALAR: str = "scalar"
ENCODING_COMMA_INTS: str = "comma_ints"
ENCODING_COMMA_STRINGS: str = "comma_strings"
ENCODING_JSON_OBJECT: str = "json_object"
ENCODINGS: frozenset[str] = frozenset(
    {ENCODING_SCALAR, ENCODING_COMMA_INTS, ENCODING_COMMA_STRINGS, ENCODING_JSON_OBJECT}
)

#: Where in a strategy dict the parameter lives.
LOCATION_STAGE_PARAMS: str = "stage_params"
LOCATION_TRANSLATOR: str = "translator"
LOCATION_DRL_BLOCK: str = "drl_block"
LOCATIONS: frozenset[str] = frozenset(
    {LOCATION_STAGE_PARAMS, LOCATION_TRANSLATOR, LOCATION_DRL_BLOCK}
)

#: Curated core parameters pre-selected when the Studio opens the Optuna optimizer.
#: Exactly these five are ``optuna_default_selected=True``; every other parameter
#: is ``False`` so a fresh tuning run explores only the high-signal hypothesis
#: space. v1.5.3: the carry floor is *not* pre-selected any more — it stays fully
#: tunable (searched whenever the carry_filter overlay is active) but a generic
#: trend-following run should not silently start trading the carry hypothesis.
DEFAULT_SELECTED_PARAMETER_KEYS: tuple[str, ...] = (
    "timing.tsmom.lookback_fast",
    "timing.tsmom.lookback_slow",
    "timing.kama.window",
    "allocator.volatility_target.target_annual_volatility",
    "risk.quant_risk.trailing_stop_pct",
)

#: One-line math/why explanations for each allocator family (Studio math tooltips).
#: Additive metadata; the engine math lives in ``src/strategies/allocators/``.
ALLOCATOR_FORMULAS: Mapping[str, str] = MappingProxyType(
    {
        "equal_weight": "w_i = L / N - every selected root gets the same signed notional weight; only gross "
        "leverage L matters.",
        "volatility_target": "w = inverse-vol weights * min(target_vol / sqrt(w^T Sigma w), L_max) - inverse "
        "volatility sizes each root, then the book is scaled so ex-ante annual volatility hits the target.",
        "risk_parity": "w_i * (Sigma w)_i = b_i * (w^T Sigma w) - each root contributes its budgeted share of "
        "total risk; the fixed point is solved iteratively.",
        "mean_variance": "w proportional to Sigma^-1 mu (max w^T mu - lambda/2 * w^T Sigma w under the "
        "weight/gross caps) - expected-return scores are scaled by covariance risk and a risk-aversion penalty.",
        "drl": "w = clip(policy(o_t), -w_max, w_max), scaled to L_max - the neural policy maps the observation "
        "to target weights; risk caps and the deterministic risk projection still apply.",
    }
)

#: Advanced/math-detail parameters (kept behind the Studio "Advanced Parameters" collapse).
ADVANCED_PARAMETER_KEYS: frozenset[str] = frozenset(
    {
        "allocator.risk_parity.max_iterations",
        "allocator.risk_parity.tolerance",
        "allocator.risk_parity.ridge",
        "allocator.risk_parity.shrinkage",
        "allocator.risk_parity.min_volatility",
        "allocator.volatility_target.ridge",
        "allocator.volatility_target.shrinkage",
        "allocator.volatility_target.min_volatility",
        "allocator.mean_variance.ridge",
        "allocator.mean_variance.shrinkage",
        "allocator.mean_variance.min_volatility",
        "allocator.drl.risk_penalty_lambda",
        "timing.kama.trend_band",
        "timing.kama.feature_band",
        "timing.tsmom.min_scale",
        "timing.tsmom.max_scale",
        "translator.min_micro_notional_ratio",
    }
)


#: Structural / data-hygiene parameters that must never enter an Optuna search
#: space: universe composition and history filters, regime enable/logic switches,
#: macro-lockout plumbing, risk enable toggles and translator execution details.
#: Tuning these would overfit data availability/execution plumbing instead of
#: strategy hypotheses (and they are not a source of alpha).
NON_TUNABLE_PARAMETER_KEYS: frozenset[str] = frozenset(
    {
        "universe.liquidity.min_avg_dollar_volume",
        "universe.liquidity.min_open_interest",
        "universe.liquidity.top_n",
        "universe.liquidity.min_history_days",
        "universe.regime.top_n",
        "universe.regime.min_universe",
        "universe.regime.filter_negative_trend",
        "universe.regime.require_both",
        "universe.regime.risk_off_roots",
        "universe.regime.risk_off_sectors",
        "universe.composite.mode",
        "universe.composite.components",
        "allocator.drl.normalize_observations",
        "risk.quant_risk.volatility_targeting_enabled",
        "risk.quant_risk.daily_loss_kill_switch_enabled",
        "risk.llm_risk.sample_count",
        "risk.llm_risk.cache_ttl_seconds",
        "risk.llm_risk.macro_lockout_minutes_before",
        "risk.llm_risk.macro_lockout_minutes_after",
        "risk.llm_risk.macro_lockout_action",
        "risk.llm_risk.macro_lockout_leverage_cap",
        "translator.rounding_mode",
        "translator.rebalance_buffer_contracts",
        "translator.enable_micro_routing",
        "translator.min_micro_notional_ratio",
    }
)

type CatalogScalar = int | float | bool | str | None
type CatalogDefault = CatalogScalar | tuple[CatalogScalar, ...]
type UIValue = object

#: Studio universe-sector groups, in display order: ``(id, label, description)``.
#: ``all`` is always first and contains every root; the seven asset-class groups
#: follow (empty groups are still exposed so the UI dropdown stays stable).
UNIVERSE_SECTOR_GROUPS: tuple[tuple[str, str, str], ...] = (
    ("all", "All", "Every futures root in the instrument master"),
    ("equities", "Equities", "Equity index futures (ES/NQ/YM/RTY families)"),
    ("rates", "Rates", "Interest-rate and bond futures"),
    ("energy", "Energy", "Crude oil, refined products and natural gas futures"),
    ("metals", "Metals", "Precious and industrial metals futures"),
    ("fx", "FX", "Currency futures"),
    ("agri", "Agri", "Grains, oilseeds, softs and livestock futures"),
    ("crypto", "Crypto", "Crypto-currency futures"),
)

#: ``configs/instruments.yaml`` sector id (or common alias) -> Studio group id.
SECTOR_GROUP_ALIASES: Mapping[str, str] = {
    "equity": "equities",
    "equities": "equities",
    "equity_index": "equities",
    "index": "equities",
    "indices": "equities",
    "rates": "rates",
    "rate": "rates",
    "fixed_income": "rates",
    "bonds": "rates",
    "energy": "energy",
    "metals": "metals",
    "metal": "metals",
    "precious_metals": "metals",
    "industrial_metals": "metals",
    "fx": "fx",
    "currency": "fx",
    "currencies": "fx",
    "agri": "agri",
    "agriculture": "agri",
    "agricultural": "agri",
    "grains": "agri",
    "grain": "agri",
    "oilseeds": "agri",
    "softs": "agri",
    "livestock": "agri",
    "crypto": "crypto",
    "cryptocurrencies": "crypto",
    "digital_assets": "crypto",
}

#: Fallback group per :class:`~src.core.types.AssetClass` value.
ASSET_CLASS_GROUPS: Mapping[str, str] = {
    "equity_index": "equities",
    "rates": "rates",
    "energy": "energy",
    "metals": "metals",
    "fx": "fx",
    "agriculture": "agri",
    "crypto": "crypto",
}

#: Display metadata for studio stage dropdown options.
STAGE_OPTION_LABELS: Mapping[str, Mapping[str, tuple[str, str]]] = {
    "universe": {
        "liquidity": (
            "Liquidity",
            "Rank roots by traded dollar volume, open interest and history.",
        ),
        "regime": (
            "Adaptive regime",
            "Slow trend/macro-vol regime plus a fast 3-day risk-off shock filter.",
        ),
        "composite": ("Composite", "Combine several selectors by intersection or union."),
    },
    "allocator": {
        "volatility_target": (
            "Volatility target",
            "Inverse-volatility weights scaled to an annualised volatility target.",
        ),
        "risk_parity": ("Risk parity", "Equal-risk-contribution allocation across the universe."),
        "equal_weight": (
            "Equal weight",
            "Equal signed notional per candidate (naive diversification baseline).",
        ),
        "mean_variance": (
            "Mean variance",
            "Regularised inverse-covariance mean-variance allocation.",
        ),
        "drl": ("DRL policy", "Trained PPO/CPPO policy allocator."),
    },
    "timing": {
        "tsmom": ("TSMOM", "Volatility-scaled time-series momentum."),
        "kama": ("KAMA", "Kaufman adaptive moving-average trend agreement."),
        "carry_filter": ("Carry filter", "Term-structure carry/roll-yield tilt."),
        "mean_reversion": (
            "Mean reversion",
            "Fade RSI/Bollinger extremes unless turbulence vetoes.",
        ),
        "composite": ("Composite", "Apply several timing overlays in order."),
    },
    "risk": {
        "quant_risk": (
            "Quant risk",
            "Volatility targeting, drawdown limits and correlated-cluster caps.",
        ),
        "llm_risk": ("LLM Risk-First", "Confidence-filtered LLM semantic risk circuit breaker."),
    },
    "translator": {
        "translator": ("Contract translator", "Approved weights → discrete futures contracts."),
    },
}

#: Preferred dropdown ordering per stage (registry entries not listed are appended).
_STAGE_CHOICE_ORDER: Mapping[str, tuple[str, ...]] = {
    "universe": ("liquidity", "regime", "composite"),
    "allocator": ("volatility_target", "risk_parity", "equal_weight", "mean_variance", "drl"),
    "timing": ("tsmom", "kama", "carry_filter", "mean_reversion", "composite"),
}


class ParameterType(StrEnum):
    """Value kind of one catalogued parameter."""

    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    SELECT = "select"
    MULTISELECT = "multiselect"


@dataclass(frozen=True, slots=True)
class StageOption:
    """One selectable implementation of a pipeline stage (Studio dropdown entry)."""

    name: str
    label: str
    description: str
    parameter_keys: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable representation."""
        return {
            "name": self.name,
            "label": self.label,
            "description": self.description,
            "parameter_keys": list(self.parameter_keys),
        }


@dataclass(frozen=True, slots=True)
class UniverseRoot:
    """Studio metadata for one futures root in the instrument master."""

    root: str
    name: str
    asset_class: str
    sector: str
    sector_group: str
    exchange: str
    multiplier: float
    tick_size: float
    micro_root: str | None
    standard_root: str | None
    currency: str

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable representation."""
        return {
            "root": self.root,
            "name": self.name,
            "asset_class": self.asset_class,
            "sector": self.sector,
            "sector_group": self.sector_group,
            "exchange": self.exchange,
            "multiplier": self.multiplier,
            "tick_size": self.tick_size,
            "micro_root": self.micro_root,
            "standard_root": self.standard_root,
            "currency": self.currency,
        }


@dataclass(frozen=True, slots=True)
class UniverseSector:
    """One Studio universe group (``All`` + asset-class sectors)."""

    id: str
    label: str
    description: str
    roots: tuple[UniverseRoot, ...]

    @property
    def count(self) -> int:
        """Number of roots in this sector group."""
        return len(self.roots)

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable representation (``count`` is derived)."""
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "count": self.count,
            "roots": [root.to_dict() for root in self.roots],
        }


@dataclass(frozen=True, slots=True)
class UniverseMetadata:
    """Sector-grouped instrument-master metadata for the Studio universe explorer."""

    sectors: tuple[UniverseSector, ...]
    roots: tuple[UniverseRoot, ...]

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable representation."""
        return {
            "sectors": [sector.to_dict() for sector in self.sectors],
            "roots": [root.to_dict() for root in self.roots],
        }


# ---------------------------------------------------------------------------
# Catalog data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """One catalogued pipeline parameter (the Studio/Optuna unit of config)."""

    key: str
    stage: str
    stage_name: str
    param: str
    label: str
    description: str
    type: ParameterType
    default: CatalogDefault
    min: float | int | None = None
    max: float | int | None = None
    step: float | int | None = None
    choices: tuple[CatalogScalar, ...] = ()
    unit: str | None = None
    optuna_tunable: bool = True
    optuna_default_selected: bool = False
    log_scale: bool = False
    yaml_encoding: str = ENCODING_SCALAR
    min_items: int | None = None
    max_items: int | None = None
    location: str = LOCATION_STAGE_PARAMS
    #: Optional one-line math formula/explanation for the Studio UI (allocator math tooltips).
    formula: str | None = None
    #: Advanced/math-detail parameter: hidden behind the Studio's "Advanced Parameters" section.
    advanced: bool = False

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable representation (the **stable** API contract)."""
        return {
            "key": self.key,
            "stage": self.stage,
            "stage_name": self.stage_name,
            "param": self.param,
            "label": self.label,
            "description": self.description,
            "type": self.type.value,
            "default": _json_default(self.default),
            "min": self.min,
            "max": self.max,
            "step": self.step,
            "choices": list(self.choices),
            "unit": self.unit,
            "optuna_tunable": self.optuna_tunable,
            "optuna_default_selected": self.optuna_default_selected,
            "log_scale": self.log_scale,
            "yaml_encoding": self.yaml_encoding,
            "min_items": self.min_items,
            "max_items": self.max_items,
            "location": self.location,
            "formula": self.formula,
            "advanced": self.advanced,
        }


@dataclass(frozen=True, slots=True)
class CatalogStage:
    """A top-level pipeline stage group in the Studio accordion."""

    id: str
    label: str
    description: str
    parameter_keys: tuple[str, ...]
    choices: tuple[StageOption, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable representation."""
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "parameter_keys": list(self.parameter_keys),
            "choices": [option.to_dict() for option in self.choices],
        }


@dataclass(frozen=True, slots=True)
class PresetTemplate:
    """A Strategy Studio hypothesis template (a complete declarative strategy)."""

    id: str
    name: str
    description: str
    hypothesis: str
    tags: tuple[str, ...]
    parameter_keys: tuple[str, ...]
    strategy: Mapping[str, object]

    def to_config(self) -> StrategyConfig:
        """Validate the preset as a :class:`~configs.strategy.StrategyConfig`."""
        return StrategyConfig.model_validate(copy.deepcopy(dict(self.strategy)))

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable representation (``strategy`` is POST-ready)."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "hypothesis": self.hypothesis,
            "tags": list(self.tags),
            "parameter_keys": list(self.parameter_keys),
            "strategy": _json_ready(self.strategy),
        }


@dataclass(frozen=True, slots=True)
class ParameterCatalog:
    """The complete typed parameter catalogue + preset templates."""

    version: int
    stages: tuple[CatalogStage, ...]
    parameters: tuple[ParameterSpec, ...]
    presets: tuple[PresetTemplate, ...]
    instrument_master: InstrumentsConfig

    # -- lookups --------------------------------------------------------------

    def get(self, key: str) -> ParameterSpec:
        """Return the spec for ``key`` (``KeyError`` with the known keys on miss)."""
        for spec in self.parameters:
            if spec.key == key:
                return spec
        raise KeyError(
            f"unknown parameter key {key!r}; known keys: {sorted(spec.key for spec in self.parameters)!r}"
        )

    def for_stage(self, stage: str) -> tuple[ParameterSpec, ...]:
        """All parameters of a top-level stage (``universe``, ``allocator``, ...)."""
        return tuple(spec for spec in self.parameters if spec.stage == stage)

    def for_stage_name(self, stage_name: str) -> tuple[ParameterSpec, ...]:
        """All parameters of one sub-stage (``liquidity``, ``tsmom``, ``drl``, ...)."""
        wanted = stage_name.strip().lower()
        return tuple(spec for spec in self.parameters if spec.stage_name == wanted)

    def tunable(self) -> tuple[ParameterSpec, ...]:
        """Parameters the Optuna optimizer may sample."""
        return tuple(spec for spec in self.parameters if spec.optuna_tunable)

    def stage_choices(self) -> Mapping[str, tuple[str, ...]]:
        """Stage id -> selectable implementation names (Studio dropdowns)."""
        return {stage.id: tuple(option.name for option in stage.choices) for stage in self.stages}

    def universe_metadata(self) -> UniverseMetadata:
        """Sector-grouped metadata for the instrument master this catalog was built from."""
        return universe_metadata(self.instrument_master)

    def universe_sectors(self) -> tuple[UniverseSector, ...]:
        """Sector groups (``All`` + asset-class sectors) for the Studio universe explorer."""
        return self.universe_metadata().sectors

    # -- validation / serialisation ------------------------------------------

    def validate(self) -> None:
        """Validate catalog invariants (called on construction and in tests)."""
        keys: set[str] = set()
        for spec in self.parameters:
            if spec.key in keys:
                raise ValueError(f"duplicate parameter key {spec.key!r}")
            keys.add(spec.key)
            if spec.stage not in {"universe", "allocator", "timing", "risk", "translator"}:
                raise ValueError(f"{spec.key}: unknown stage {spec.stage!r}")
            if spec.location not in LOCATIONS:
                raise ValueError(f"{spec.key}: unknown location {spec.location!r}")
            if spec.yaml_encoding not in ENCODINGS:
                raise ValueError(f"{spec.key}: unknown yaml_encoding {spec.yaml_encoding!r}")
            self._validate_spec(spec)
        stage_ids = {stage.id for stage in self.stages}
        for stage in self.stages:
            if not stage.parameter_keys:
                raise ValueError(f"stage {stage.id!r} has no parameters")
            for key in stage.parameter_keys:
                if key not in keys:
                    raise ValueError(f"stage {stage.id!r} references unknown key {key!r}")
        for spec in self.parameters:
            if spec.stage not in stage_ids:
                raise ValueError(f"{spec.key}: stage {spec.stage!r} is not declared in stages")
        for stage in self.stages:
            seen: set[str] = set()
            for option in stage.choices:
                if not option.name or option.name in seen:
                    raise ValueError(
                        f"stage {stage.id!r} has an invalid/duplicate choice {option.name!r}"
                    )
                seen.add(option.name)
                for key in option.parameter_keys:
                    if key not in keys:
                        raise ValueError(
                            f"stage {stage.id!r} choice {option.name!r} references unknown key {key!r}"
                        )
            if stage.id == "universe" and not stage.choices:
                raise ValueError("the universe stage must expose selector choices")
            if stage.id == "allocator" and not stage.choices:
                raise ValueError("the allocator stage must expose allocator choices")
        selected_keys = {spec.key for spec in self.parameters if spec.optuna_default_selected}
        expected_selected = set(DEFAULT_SELECTED_PARAMETER_KEYS)
        if selected_keys != expected_selected:
            raise ValueError(
                "optuna_default_selected must be exactly "
                f"{sorted(expected_selected)!r}, got {sorted(selected_keys)!r}"
            )
        known_keys = {spec.key for spec in self.parameters}
        missing_hygiene = sorted(NON_TUNABLE_PARAMETER_KEYS - known_keys)
        if missing_hygiene:
            raise ValueError(f"non-tunable hygiene keys are not catalogued: {missing_hygiene!r}")
        missing_advanced = sorted(ADVANCED_PARAMETER_KEYS - known_keys)
        if missing_advanced:
            raise ValueError(f"advanced UI metadata references unknown keys: {missing_advanced!r}")
        for spec in self.parameters:
            if spec.optuna_default_selected and not spec.optuna_tunable:
                raise ValueError(
                    f"{spec.key}: optuna_default_selected requires optuna_tunable=True"
                )
            if spec.key in NON_TUNABLE_PARAMETER_KEYS and spec.optuna_tunable:
                raise ValueError(
                    f"{spec.key}: structural/data-hygiene parameter must not be tunable"
                )
        metadata = self.universe_metadata()
        root_ids = {root.root for root in metadata.roots}
        if not root_ids:
            raise ValueError("universe metadata has no roots")
        covered: list[str] = [
            root.root for sector in metadata.sectors if sector.id != "all" for root in sector.roots
        ]
        if len(covered) != len(set(covered)):
            raise ValueError("a root is assigned to more than one non-All sector group")
        if set(covered) != root_ids:
            raise ValueError("sector groups do not partition the instrument master roots")
        preset_ids: set[str] = set()
        for preset in self.presets:
            if preset.id in preset_ids:
                raise ValueError(f"duplicate preset id {preset.id!r}")
            preset_ids.add(preset.id)
            for key in preset.parameter_keys:
                if key not in keys:
                    raise ValueError(f"preset {preset.id!r} references unknown key {key!r}")

    @staticmethod
    def _validate_spec(spec: ParameterSpec) -> None:
        if spec.type is ParameterType.BOOL:
            if spec.choices:
                raise ValueError(f"{spec.key}: bool parameters must not declare choices")
            if not isinstance(spec.default, bool):
                raise ValueError(f"{spec.key}: bool parameter default must be a bool")
            if spec.optuna_tunable:
                pass  # categorical {true,false} sampling is supported
            return
        if spec.type is ParameterType.SELECT:
            if len(spec.choices) < 2:
                raise ValueError(f"{spec.key}: select parameters need at least two choices")
            if spec.default not in spec.choices:
                raise ValueError(f"{spec.key}: select default {spec.default!r} is not a choice")
            return
        if spec.type is ParameterType.MULTISELECT:
            if not spec.choices:
                raise ValueError(f"{spec.key}: multiselect parameters need non-empty choices")
            if not isinstance(spec.default, tuple):
                raise ValueError(f"{spec.key}: multiselect default must be a tuple")
            if not set(spec.default) <= set(spec.choices):
                raise ValueError(f"{spec.key}: multiselect default is not a subset of choices")
            if spec.min_items is not None and len(spec.default) < spec.min_items:
                raise ValueError(f"{spec.key}: default has fewer than min_items entries")
            if spec.optuna_tunable:
                raise ValueError(f"{spec.key}: multiselect parameters cannot be optuna_tunable")
            if spec.yaml_encoding == ENCODING_SCALAR:
                raise ValueError(f"{spec.key}: multiselect parameters need a list yaml encoding")
            return
        # int / float
        if spec.choices:
            raise ValueError(f"{spec.key}: numeric parameters must not declare choices")
        if spec.min is not None and spec.max is not None and spec.min > spec.max:
            raise ValueError(f"{spec.key}: min must not exceed max")
        if spec.step is not None and spec.step <= 0:
            raise ValueError(f"{spec.key}: step must be strictly positive")
        if spec.log_scale and (spec.min is None or spec.min <= 0):
            raise ValueError(f"{spec.key}: log_scale requires a strictly positive finite min")
        numeric_default = spec.default
        if numeric_default is not None:
            if spec.type is ParameterType.INT:
                if isinstance(numeric_default, bool) or not isinstance(numeric_default, int):
                    raise ValueError(f"{spec.key}: int parameter default must be an int")
            elif isinstance(numeric_default, bool) or not isinstance(numeric_default, (int, float)):
                raise ValueError(f"{spec.key}: float parameter default must be a number")
            if spec.min is not None and numeric_default < spec.min:
                raise ValueError(f"{spec.key}: default is below min")
            if spec.max is not None and numeric_default > spec.max:
                raise ValueError(f"{spec.key}: default is above max")
        if spec.default is None and spec.optuna_tunable:
            raise ValueError(f"{spec.key}: optional (None-default) parameters cannot be tunable")
        if spec.optuna_tunable and (spec.min is None or spec.max is None or spec.step is None):
            raise ValueError(
                f"{spec.key}: optuna_tunable parameters require finite min/max/step bounds"
            )

    def to_dict(self) -> dict[str, object]:
        """JSON-ready payload for ``GET /api/strategies/schema/catalog``.

        Additive top-level keys beyond the Phase-10 contract: ``universe``
        (:meth:`universe_metadata`, sector-grouped instrument master) and
        ``stage_choices`` (stage id -> dropdown option names). Existing
        ``version``/``stages``/``parameters``/``presets`` shapes are unchanged
        except that every stage dict now also carries ``choices``.
        """
        return {
            "version": self.version,
            "stages": [stage.to_dict() for stage in self.stages],
            "parameters": [spec.to_dict() for spec in self.parameters],
            "presets": [preset.to_dict() for preset in self.presets],
            "universe": self.universe_metadata().to_dict(),
            "stage_choices": {
                stage_id: list(names) for stage_id, names in self.stage_choices().items()
            },
            # Additive UI metadata: one-line math per allocator family + the advanced
            # parameter keys (the UI can render tooltips/collapse sections without
            # hard-coding the mapping).
            "formulas": dict(ALLOCATOR_FORMULAS),
            "advanced_parameter_keys": sorted(ADVANCED_PARAMETER_KEYS),
        }


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def _json_default(value: CatalogDefault) -> object:
    return list(value) if isinstance(value, tuple) else value


def _json_ready(value: object) -> object:
    """Recursively convert tuples to lists so the payload is JSON-serialisable."""
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


# ---------------------------------------------------------------------------
# Spec construction helper
# ---------------------------------------------------------------------------


def _p(
    stage: str,
    stage_name: str,
    param: str,
    label: str,
    description: str,
    type_: ParameterType,
    default: CatalogDefault,
    *,
    min: float | int | None = None,
    max: float | int | None = None,
    step: float | int | None = None,
    choices: tuple[CatalogScalar, ...] = (),
    unit: str | None = None,
    tunable: bool = True,
    log_scale: bool = False,
    yaml_encoding: str = ENCODING_SCALAR,
    min_items: int | None = None,
    max_items: int | None = None,
    location: str = LOCATION_STAGE_PARAMS,
) -> ParameterSpec:
    key = f"translator.{param}" if stage == "translator" else f"{stage}.{stage_name}.{param}"
    return ParameterSpec(
        key=key,
        stage=stage,
        stage_name=stage_name,
        param=param,
        label=label,
        description=description,
        type=type_,
        default=default,
        min=min,
        max=max,
        step=step,
        choices=choices,
        unit=unit,
        optuna_tunable=tunable,
        log_scale=log_scale,
        yaml_encoding=yaml_encoding,
        min_items=min_items,
        max_items=max_items,
        location=location,
    )


# ---------------------------------------------------------------------------
# Catalogue rows: universe (S_t)
# ---------------------------------------------------------------------------


def _universe_specs(master: InstrumentsConfig) -> tuple[ParameterSpec, ...]:
    root_choices: tuple[CatalogScalar, ...] = tuple(master.roots())
    sector_choices: tuple[CatalogScalar, ...] = tuple(master.sectors())
    return (
        _p(
            "universe",
            "liquidity",
            "min_avg_dollar_volume",
            "Minimum dollar volume",
            "Liquidity floor: drop roots whose average daily traded notional is below this value. "
            "Higher values keep only the most tradable contracts, reducing slippage and roll cost.",
            ParameterType.FLOAT,
            50_000_000.0,
            min=0.0,
            max=1e11,
            step=1_000_000.0,
            unit="USD",
        ),
        _p(
            "universe",
            "liquidity",
            "min_open_interest",
            "Minimum open interest",
            "Open-interest floor: drop roots whose front contract has fewer than this many "
            "contracts outstanding, which protects against illiquid months and roll squeezes.",
            ParameterType.FLOAT,
            5_000.0,
            min=0.0,
            max=10_000_000.0,
            step=500.0,
            unit="contracts",
        ),
        _p(
            "universe",
            "liquidity",
            "top_n",
            "Universe size cap",
            "Hard cap on the number of roots passed to the allocator (None keeps every root that "
            "passes the filters). Smaller universes concentrate risk but are easier to hedge.",
            ParameterType.INT,
            10,
            min=1,
            max=50,
            step=1,
            tunable=True,
        ),
        _p(
            "universe",
            "liquidity",
            "min_history_days",
            "Minimum history (days)",
            "Reject roots with less than this many return observations so features and covariance "
            "estimates are warmed up before the root is traded.",
            ParameterType.INT,
            252,
            min=0,
            max=756,
            step=1,
            unit="days",
        ),
        _p(
            "universe",
            "regime",
            "trend_lookback_days",
            "Slow trend lookback",
            "Slow-regime trend window (26 weeks by default). Breadth of this trailing return is "
            "the first leg of the risk-off detector.",
            ParameterType.INT,
            126,
            min=21,
            max=252,
            step=1,
            unit="days",
        ),
        _p(
            "universe",
            "regime",
            "vol_lookback_days",
            "Macro volatility lookback",
            "Window for the cross-sectional annualised volatility gauge used to classify the "
            "macro regime.",
            ParameterType.INT,
            60,
            min=10,
            max=252,
            step=1,
            unit="days",
        ),
        _p(
            "universe",
            "regime",
            "shock_lookback_days",
            "Fast shock lookback",
            "Short window (3 days by default) for the fast risk-off shock detector.",
            ParameterType.INT,
            3,
            min=1,
            max=10,
            step=1,
            unit="days",
        ),
        _p(
            "universe",
            "regime",
            "shock_z_threshold",
            "Shock z-score threshold",
            "Standardised 3-day move at which the fast risk-off filter triggers; lower values "
            "de-risk sooner but produce more false alarms.",
            ParameterType.FLOAT,
            3.0,
            min=0.5,
            max=10.0,
            step=0.1,
            unit="sigma",
        ),
        _p(
            "universe",
            "regime",
            "shock_return_threshold",
            "Shock raw-return threshold",
            "Raw 3-day cross-sectional drawdown that triggers risk-off even if volatility is low.",
            ParameterType.FLOAT,
            0.02,
            min=0.0,
            max=0.20,
            step=0.005,
            unit="return",
        ),
        _p(
            "universe",
            "regime",
            "macro_vol_threshold",
            "Macro volatility threshold",
            "Annualised volatility above which the slow regime counts as stressed.",
            ParameterType.FLOAT,
            0.22,
            min=0.01,
            max=1.0,
            step=0.01,
            unit="annual vol",
        ),
        _p(
            "universe",
            "regime",
            "breadth_threshold",
            "Negative-trend breadth",
            "Fraction of roots that must be trending down for the slow risk-off regime to engage.",
            ParameterType.FLOAT,
            0.60,
            min=0.0,
            max=1.0,
            step=0.05,
        ),
        _p(
            "universe",
            "regime",
            "filter_negative_trend",
            "Filter falling roots",
            "When true, risk-off excludes roots with a negative slow trend instead of only the "
            "explicitly excluded roots.",
            ParameterType.BOOL,
            True,
        ),
        _p(
            "universe",
            "regime",
            "min_universe",
            "Minimum universe size",
            "The regime selector never returns fewer than this many roots (or the empty set when "
            "everything is explicitly excluded).",
            ParameterType.INT,
            1,
            min=1,
            max=20,
            step=1,
        ),
        _p(
            "universe",
            "regime",
            "top_n",
            "Regime universe cap",
            "Optional cap on the risk-regime universe size.",
            ParameterType.INT,
            10,
            min=1,
            max=50,
            step=1,
            tunable=True,
        ),
        _p(
            "universe",
            "regime",
            "require_both",
            "Require both regime filters",
            "When true the slow trend filter AND the fast shock filter must both fire, which is "
            "more conservative than the default OR.",
            ParameterType.BOOL,
            False,
        ),
        _p(
            "universe",
            "regime",
            "risk_off_roots",
            "Excluded risk-off roots",
            "Roots removed from the universe when the regime is risk-off.",
            ParameterType.MULTISELECT,
            (),
            choices=root_choices,
            tunable=False,
            yaml_encoding=ENCODING_COMMA_STRINGS,
            location=LOCATION_STAGE_PARAMS,
        ),
        _p(
            "universe",
            "regime",
            "risk_off_sectors",
            "Excluded risk-off sectors",
            "Instrument-master sectors removed from the universe when the regime is risk-off.",
            ParameterType.MULTISELECT,
            (),
            choices=sector_choices,
            tunable=False,
            yaml_encoding=ENCODING_COMMA_STRINGS,
        ),
        _p(
            "universe",
            "composite",
            "components",
            "Composite components",
            "Selectors combined by the composite universe selector, in evaluation order.",
            ParameterType.MULTISELECT,
            ("liquidity", "regime"),
            choices=("liquidity", "regime"),
            tunable=False,
            yaml_encoding=ENCODING_COMMA_STRINGS,
            min_items=1,
            max_items=2,
        ),
        _p(
            "universe",
            "composite",
            "mode",
            "Composite set operation",
            "How component universes are combined: intersection (AND, conservative) or union (OR).",
            ParameterType.SELECT,
            "intersection",
            choices=("intersection", "union"),
            tunable=True,
        ),
    )


# ---------------------------------------------------------------------------
# Catalogue rows: allocators (A_t)
# ---------------------------------------------------------------------------


def _allocator_specs() -> tuple[ParameterSpec, ...]:
    return (
        _p(
            "allocator",
            "equal_weight",
            "gross_leverage",
            "Gross leverage",
            "Total gross notional deployed by the equal-weight baseline, as a multiple of equity.",
            ParameterType.FLOAT,
            1.0,
            min=0.0,
            max=5.0,
            step=0.1,
            unit="x equity",
        ),
        _p(
            "allocator",
            "equal_weight",
            "max_weight",
            "Max single-instrument weight",
            "Optional cap on any single leg's notional weight for the equal-weight allocator.",
            ParameterType.FLOAT,
            0.5,
            min=0.01,
            max=1.0,
            step=0.01,
        ),
        _p(
            "allocator",
            "risk_parity",
            "target_annual_volatility",
            "Target annual volatility",
            "Scale the equal-risk-contribution book so its ex-ante annualised volatility matches "
            "this target; the classic risk-parity risk budget.",
            ParameterType.FLOAT,
            0.10,
            min=0.01,
            max=0.60,
            step=0.01,
            unit="annual vol",
        ),
        _p(
            "allocator",
            "risk_parity",
            "gross_leverage",
            "Gross leverage (no vol target)",
            "Gross leverage applied when no explicit volatility target is configured.",
            ParameterType.FLOAT,
            1.0,
            min=0.0,
            max=5.0,
            step=0.1,
            unit="x equity",
        ),
        _p(
            "allocator",
            "risk_parity",
            "max_gross_leverage",
            "Max gross leverage",
            "Hard cap on the gross leverage the volatility-targeted risk-parity book may use.",
            ParameterType.FLOAT,
            3.0,
            min=0.1,
            max=10.0,
            step=0.1,
            unit="x equity",
        ),
        _p(
            "allocator",
            "risk_parity",
            "max_weight",
            "Max single-instrument weight",
            "Optional cap on any single leg's weight after risk budgeting.",
            ParameterType.FLOAT,
            0.5,
            min=0.01,
            max=1.0,
            step=0.01,
        ),
        _p(
            "allocator",
            "risk_parity",
            "covariance_lookback_days",
            "Covariance lookback",
            "Trailing window used to rebuild the covariance matrix when the state does not carry "
            "one.",
            ParameterType.INT,
            60,
            min=10,
            max=252,
            step=1,
            unit="days",
        ),
        _p(
            "allocator",
            "risk_parity",
            "max_iterations",
            "ERC solver iterations",
            "Maximum iterations of the equal-risk-contribution fixed-point solver.",
            ParameterType.INT,
            500,
            min=10,
            max=5_000,
            step=10,
            tunable=False,
        ),
        _p(
            "allocator",
            "risk_parity",
            "tolerance",
            "ERC solver tolerance",
            "Convergence tolerance of the equal-risk-contribution iterations.",
            ParameterType.FLOAT,
            1e-10,
            min=1e-12,
            max=1e-3,
            step=1e-10,
            tunable=False,
        ),
        _p(
            "allocator",
            "risk_parity",
            "ridge",
            "Covariance ridge",
            "Diagonal ridge added to the covariance matrix so it stays invertible when roots "
            "outnumber observations.",
            ParameterType.FLOAT,
            1e-10,
            min=1e-12,
            max=1e-2,
            step=1e-10,
            tunable=False,
            log_scale=True,
        ),
        _p(
            "allocator",
            "risk_parity",
            "shrinkage",
            "Covariance shrinkage",
            "Shrink covariance towards its diagonal (0 = raw sample covariance, 1 = diagonal "
            "only) to stabilise risk budgeting on short histories.",
            ParameterType.FLOAT,
            0.0,
            min=0.0,
            max=0.99,
            step=0.05,
        ),
        _p(
            "allocator",
            "risk_parity",
            "min_volatility",
            "Minimum volatility floor",
            "Floor applied to per-instrument volatility estimates before risk budgeting.",
            ParameterType.FLOAT,
            0.002,
            min=0.0001,
            max=0.10,
            step=0.0005,
            log_scale=True,
        ),
        _p(
            "allocator",
            "mean_variance",
            "risk_aversion",
            "Risk aversion (lambda)",
            "Mean-variance risk-aversion coefficient: lower values take larger positions for the "
            "same expected-return estimate.",
            ParameterType.FLOAT,
            5.0,
            min=0.1,
            max=50.0,
            step=0.1,
        ),
        _p(
            "allocator",
            "mean_variance",
            "ridge",
            "Covariance ridge",
            "Diagonal ridge making the inverse-covariance mean-variance solve numerically stable.",
            ParameterType.FLOAT,
            1e-8,
            min=1e-12,
            max=1e-2,
            step=1e-8,
            log_scale=True,
        ),
        _p(
            "allocator",
            "mean_variance",
            "gross_leverage",
            "Gross leverage",
            "Gross leverage used for the equal-weight fallback when the mean-variance solution "
            "degenerates.",
            ParameterType.FLOAT,
            1.0,
            min=0.0,
            max=5.0,
            step=0.1,
            unit="x equity",
        ),
        _p(
            "allocator",
            "mean_variance",
            "max_gross_leverage",
            "Max gross leverage",
            "Hard cap on gross leverage of the mean-variance solution.",
            ParameterType.FLOAT,
            3.0,
            min=0.1,
            max=10.0,
            step=0.1,
            unit="x equity",
        ),
        _p(
            "allocator",
            "mean_variance",
            "max_weight",
            "Max single-instrument weight",
            "Optional cap on any single leg's weight.",
            ParameterType.FLOAT,
            0.5,
            min=0.01,
            max=1.0,
            step=0.01,
        ),
        _p(
            "allocator",
            "mean_variance",
            "long_only",
            "Long-only",
            "Clip negative expected-return bets to zero (true) or allow short legs (false).",
            ParameterType.BOOL,
            True,
        ),
        _p(
            "allocator",
            "mean_variance",
            "expected_return_window",
            "Expected-return window",
            "Trailing window of cumulative returns used as the expected-return estimate.",
            ParameterType.INT,
            126,
            min=21,
            max=504,
            step=1,
            unit="days",
        ),
        _p(
            "allocator",
            "mean_variance",
            "covariance_lookback_days",
            "Covariance lookback",
            "Trailing window used to rebuild the covariance matrix.",
            ParameterType.INT,
            60,
            min=10,
            max=252,
            step=1,
            unit="days",
        ),
        _p(
            "allocator",
            "mean_variance",
            "shrinkage",
            "Covariance shrinkage",
            "Shrink covariance towards its diagonal for stability.",
            ParameterType.FLOAT,
            0.0,
            min=0.0,
            max=0.99,
            step=0.05,
        ),
        _p(
            "allocator",
            "mean_variance",
            "min_volatility",
            "Minimum volatility floor",
            "Floor applied to per-instrument volatility estimates.",
            ParameterType.FLOAT,
            0.002,
            min=0.0001,
            max=0.10,
            step=0.0005,
            log_scale=True,
        ),
        _p(
            "allocator",
            "volatility_target",
            "target_annual_volatility",
            "Target annual volatility",
            "Scale inverse-volatility weights so the ex-ante portfolio volatility matches this "
            "annualised target (the core of volatility-targeted TSMOM).",
            ParameterType.FLOAT,
            0.12,
            min=0.01,
            max=0.60,
            step=0.01,
            unit="annual vol",
        ),
        _p(
            "allocator",
            "volatility_target",
            "covariance_lookback_days",
            "Covariance lookback",
            "Trailing window for the volatility/correlation estimate used in sizing.",
            ParameterType.INT,
            60,
            min=10,
            max=252,
            step=1,
            unit="days",
        ),
        _p(
            "allocator",
            "volatility_target",
            "max_gross_leverage",
            "Max gross leverage",
            "Cap on the leverage the volatility scaler may demand when ex-ante volatility "
            "collapses.",
            ParameterType.FLOAT,
            3.0,
            min=0.1,
            max=10.0,
            step=0.1,
            unit="x equity",
        ),
        _p(
            "allocator",
            "volatility_target",
            "min_gross_leverage",
            "Min gross leverage",
            "Floor on the leverage scaler (0 allows the book to shrink fully when volatility "
            "explodes).",
            ParameterType.FLOAT,
            0.0,
            min=0.0,
            max=5.0,
            step=0.1,
            unit="x equity",
        ),
        _p(
            "allocator",
            "volatility_target",
            "max_weight",
            "Max single-instrument weight",
            "Optional cap on any single leg's weight after volatility scaling.",
            ParameterType.FLOAT,
            0.5,
            min=0.01,
            max=1.0,
            step=0.01,
        ),
        _p(
            "allocator",
            "volatility_target",
            "ridge",
            "Covariance ridge",
            "Diagonal ridge added to the covariance matrix before scaling.",
            ParameterType.FLOAT,
            1e-10,
            min=1e-12,
            max=1e-2,
            step=1e-10,
            log_scale=True,
        ),
        _p(
            "allocator",
            "volatility_target",
            "shrinkage",
            "Covariance shrinkage",
            "Shrink covariance towards its diagonal for stability.",
            ParameterType.FLOAT,
            0.0,
            min=0.0,
            max=0.99,
            step=0.05,
        ),
        _p(
            "allocator",
            "volatility_target",
            "min_volatility",
            "Minimum volatility floor",
            "Floor applied to per-instrument volatility estimates.",
            ParameterType.FLOAT,
            0.002,
            min=0.0001,
            max=0.10,
            step=0.0005,
            log_scale=True,
        ),
        *_drl_allocator_specs(),
    )


def _drl_allocator_specs() -> tuple[ParameterSpec, ...]:
    return (
        _p(
            "allocator",
            "drl",
            "algorithm",
            "DRL algorithm",
            "Stable-Baselines3 algorithm trained by the DRL allocator. CPPO is constrained PPO and "
            "trains as PPO against the semantic-risk-shaped reward.",
            ParameterType.SELECT,
            "PPO",
            choices=("PPO", "CPPO"),
            tunable=False,
        ),
        _p(
            "allocator",
            "drl",
            "policy",
            "Policy network",
            "Stable-Baselines3 policy class for the futures weight action space.",
            ParameterType.SELECT,
            "MlpPolicy",
            choices=("MlpPolicy", "MlpLstmPolicy"),
            tunable=False,
        ),
        _p(
            "allocator",
            "drl",
            "total_timesteps",
            "Training timesteps",
            "Training budget passed to SB3 learn() when the allocator trains a fresh policy.",
            ParameterType.INT,
            100_000,
            min=1_024,
            max=2_000_000,
            step=1_024,
            unit="steps",
        ),
        _p(
            "allocator",
            "drl",
            "learning_rate",
            "Learning rate",
            "Optimiser step size for the DRL policy.",
            ParameterType.FLOAT,
            3e-4,
            min=1e-6,
            max=1e-1,
            step=1e-6,
            log_scale=True,
        ),
        _p(
            "allocator",
            "drl",
            "gamma",
            "Discount factor",
            "Reward discount factor; higher values give the policy a longer horizon.",
            ParameterType.FLOAT,
            0.99,
            min=0.80,
            max=0.9999,
            step=0.001,
        ),
        _p(
            "allocator",
            "drl",
            "seed",
            "Random seed",
            "Training seed for reproducible DRL runs.",
            ParameterType.INT,
            7,
            min=0,
            max=10_000,
            step=1,
            tunable=False,
        ),
        _p(
            "allocator",
            "drl",
            "risk_penalty_lambda",
            "Semantic risk penalty (lambda)",
            "FinRL Risk-First Lagrangian weight for the env reward penalty on semantic risk; 0 "
            "disables shaping.",
            ParameterType.FLOAT,
            0.5,
            min=0.0,
            max=10.0,
            step=0.05,
        ),
        _p(
            "allocator",
            "drl",
            "confidence_threshold",
            "LLM confidence floor (tau)",
            "Anti-hallucination confidence floor below which a risk assessment is ignored by the "
            "reward penalty.",
            ParameterType.FLOAT,
            0.70,
            min=0.0,
            max=1.0,
            step=0.01,
        ),
        _p(
            "allocator",
            "drl",
            "safe_risk_score",
            "Safe semantic risk score (theta)",
            "Semantic risk score at or below which the reward penalty is neutral.",
            ParameterType.FLOAT,
            1.5,
            min=1.0,
            max=4.0,
            step=0.1,
        ),
        _p(
            "allocator",
            "drl",
            "action_mode",
            "Action interpretation",
            "How the policy action becomes weights: raw signed weights (weights) or gross-"
            "normalised to the configured leverage (normalized).",
            ParameterType.SELECT,
            "weights",
            choices=("weights", "normalized"),
            tunable=False,
        ),
        _p(
            "allocator",
            "drl",
            "gross_leverage",
            "Gross leverage cap",
            "Maximum gross leverage (and gross normalisation target) applied to policy output.",
            ParameterType.FLOAT,
            1.0,
            min=0.1,
            max=5.0,
            step=0.1,
            unit="x equity",
        ),
        _p(
            "allocator",
            "drl",
            "max_instrument_weight",
            "Max single-instrument weight",
            "Per-instrument clip applied to the policy's raw action before gross scaling.",
            ParameterType.FLOAT,
            1.0,
            min=0.05,
            max=2.0,
            step=0.05,
        ),
        _p(
            "allocator",
            "drl",
            "device",
            "Training/inference device",
            "Stable-Baselines3 device: cpu (default, deterministic and fastest for the small MLP "
            "policies) or auto/cuda for GPU hosts.",
            ParameterType.SELECT,
            "cpu",
            choices=("cpu", "cuda", "auto"),
            tunable=False,
        ),
        _p(
            "allocator",
            "drl",
            "reward_risk_penalty_lambda",
            "DRL reward risk penalty (config)",
            "DRLSpec-level semantic risk penalty lambda used when training via "
            "python -m src.strategies.train_drl.",
            ParameterType.FLOAT,
            0.5,
            min=0.0,
            max=10.0,
            step=0.05,
            location=LOCATION_DRL_BLOCK,
        ),
        _p(
            "allocator",
            "drl",
            "normalize_observations",
            "Normalise observations",
            "Apply observation normalisation during DRL training (true by default).",
            ParameterType.BOOL,
            True,
            location=LOCATION_DRL_BLOCK,
        ),
    )


# ---------------------------------------------------------------------------
# Catalogue rows: timing overlays (T_t)
# ---------------------------------------------------------------------------


def _timing_specs() -> tuple[ParameterSpec, ...]:
    timing_names: tuple[CatalogScalar, ...] = (
        "tsmom",
        "kama",
        "carry_filter",
        "mean_reversion",
    )
    return (
        _p(
            "timing",
            "tsmom",
            "lookback_fast",
            "TSMOM fast lookback",
            "Fast time-series-momentum lookback; its sign is the primary directional signal.",
            ParameterType.INT,
            21,
            min=2,
            max=126,
            step=1,
            unit="days",
        ),
        _p(
            "timing",
            "tsmom",
            "lookback_slow",
            "TSMOM slow lookback",
            "Slow momentum lookback used to confirm the fast signal (disagreement flattens the "
            "leg).",
            ParameterType.INT,
            126,
            min=21,
            max=504,
            step=1,
            unit="days",
        ),
        _p(
            "timing",
            "tsmom",
            "vol_lookback",
            "Volatility lookback",
            "Lookback for the volatility used to scale the momentum direction.",
            ParameterType.INT,
            20,
            min=5,
            max=126,
            step=1,
            unit="days",
        ),
        _p(
            "timing",
            "tsmom",
            "target_volatility",
            "Volatility scaling target",
            "Optional per-instrument annualised volatility target; the momentum factor is scaled "
            "by target/realised volatility (None keeps the raw +/-1 factor).",
            ParameterType.FLOAT,
            0.12,
            min=0.01,
            max=1.0,
            step=0.01,
            unit="annual vol",
            tunable=True,
        ),
        _p(
            "timing",
            "tsmom",
            "min_scale",
            "Minimum vol scale",
            "Floor on the volatility scaling factor (0 allows the leg to go flat when volatility "
            "explodes).",
            ParameterType.FLOAT,
            0.0,
            min=0.0,
            max=5.0,
            step=0.05,
        ),
        _p(
            "timing",
            "tsmom",
            "max_scale",
            "Maximum vol scale",
            "Cap on the volatility scaling factor so a quiet market cannot demand absurd leverage.",
            ParameterType.FLOAT,
            3.0,
            min=0.5,
            max=10.0,
            step=0.1,
        ),
        _p(
            "timing",
            "tsmom",
            "require_slow_agreement",
            "Require fast/slow agreement",
            "Only trade when fast and slow momentum agree; disagreement flattens the leg.",
            ParameterType.BOOL,
            True,
        ),
        _p(
            "timing",
            "tsmom",
            "allow_flip",
            "Allow direction flips",
            "Allow the momentum sign to flip a leg's direction (false only scales existing legs).",
            ParameterType.BOOL,
            True,
        ),
        _p(
            "timing",
            "kama",
            "window",
            "KAMA efficiency window",
            "Kaufman efficiency-ratio lookback; shorter windows react faster to trend changes.",
            ParameterType.INT,
            10,
            min=2,
            max=50,
            step=1,
            unit="days",
        ),
        _p(
            "timing",
            "kama",
            "fast_period",
            "KAMA fast period",
            "Fast smoothing period of the adaptive moving average.",
            ParameterType.INT,
            2,
            min=1,
            max=20,
            step=1,
        ),
        _p(
            "timing",
            "kama",
            "slow_period",
            "KAMA slow period",
            "Slow smoothing period of the adaptive moving average.",
            ParameterType.INT,
            30,
            min=5,
            max=100,
            step=1,
        ),
        _p(
            "timing",
            "kama",
            "trend_band",
            "KAMA trend band",
            "Relative price/KAMA gap at which the trend agreement saturates at +/-1.",
            ParameterType.FLOAT,
            0.02,
            min=0.001,
            max=0.20,
            step=0.001,
            log_scale=True,
        ),
        _p(
            "timing",
            "kama",
            "feature_band",
            "KAMA feature fallback band",
            "Band used when only the ret_126d feature (no return path) is available.",
            ParameterType.FLOAT,
            0.05,
            min=0.001,
            max=0.50,
            step=0.001,
            log_scale=True,
        ),
        _p(
            "timing",
            "kama",
            "allow_flip",
            "Allow direction flips",
            "Allow a below-KAMA price to flip a long leg short (false flattens it instead).",
            ParameterType.BOOL,
            False,
        ),
        _p(
            "timing",
            "carry_filter",
            "min_carry_bps",
            "Minimum carry (bps)",
            "Annualised carry floor in basis points. The -50 bps default keeps mildly "
            "contangoed legs (small negative carry) in the book while filtering deep contango; "
            "legs below the floor are zeroed (zero mode) or scaled down (scale mode).",
            ParameterType.FLOAT,
            -50.0,
            min=-200.0,
            max=50.0,
            step=10.0,
            unit="bps",
        ),
        _p(
            "timing",
            "carry_filter",
            "carry_weight",
            "Carry tilt strength",
            "Strength of the continuous carry tilt applied above the minimum carry floor.",
            ParameterType.FLOAT,
            1.0,
            min=0.0,
            max=3.0,
            step=0.05,
        ),
        _p(
            "timing",
            "carry_filter",
            "carry_scale_bps",
            "Carry saturation scale",
            "Carry surplus (bps) at which the carry tilt approaches its maximum.",
            ParameterType.FLOAT,
            100.0,
            min=1.0,
            max=1_000.0,
            step=5.0,
            unit="bps",
        ),
        _p(
            "timing",
            "carry_filter",
            "max_scale",
            "Max carry tilt",
            "Cap on the carry multiplier applied to a leg.",
            ParameterType.FLOAT,
            1.5,
            min=1.0,
            max=5.0,
            step=0.05,
        ),
        _p(
            "timing",
            "carry_filter",
            "mode",
            "Carry filter mode",
            "zero: remove legs below the carry floor; scale: shrink them continuously.",
            ParameterType.SELECT,
            "zero",
            choices=("zero", "scale"),
            tunable=True,
        ),
        _p(
            "timing",
            "mean_reversion",
            "rsi_period",
            "RSI period",
            "Wilder RSI smoothing period for the mean-reversion entry filter.",
            ParameterType.INT,
            14,
            min=2,
            max=50,
            step=1,
            unit="days",
        ),
        _p(
            "timing",
            "mean_reversion",
            "oversold_threshold",
            "RSI oversold threshold",
            "RSI level below which the overlay boosts the reversion side of the book.",
            ParameterType.FLOAT,
            25.0,
            min=1.0,
            max=49.0,
            step=1.0,
            unit="RSI",
        ),
        _p(
            "timing",
            "mean_reversion",
            "overbought_threshold",
            "RSI overbought threshold",
            "RSI level above which the overlay fades the existing exposure.",
            ParameterType.FLOAT,
            75.0,
            min=51.0,
            max=99.0,
            step=1.0,
            unit="RSI",
        ),
        _p(
            "timing",
            "mean_reversion",
            "bollinger_window",
            "Bollinger window",
            "Lookback of the Bollinger mean/standard-deviation bands.",
            ParameterType.INT,
            20,
            min=5,
            max=100,
            step=1,
            unit="days",
        ),
        _p(
            "timing",
            "mean_reversion",
            "bollinger_std",
            "Bollinger width",
            "Band width in standard deviations; smaller values trigger reversion more often.",
            ParameterType.FLOAT,
            2.0,
            min=0.5,
            max=4.0,
            step=0.1,
            unit="sigma",
        ),
        _p(
            "timing",
            "mean_reversion",
            "turbulence_veto_threshold",
            "Turbulence veto",
            "Mahalanobis turbulence index at or above which the overlay freezes (never fade a "
            "structural break).",
            ParameterType.FLOAT,
            0.95,
            min=0.05,
            max=10.0,
            step=0.05,
        ),
        _p(
            "timing",
            "mean_reversion",
            "reversion_strength",
            "Reversion strength",
            "Multiplier applied to the reversion side (+50% by default) and to the stale side "
            "(-50%).",
            ParameterType.FLOAT,
            0.5,
            min=0.0,
            max=1.0,
            step=0.05,
        ),
        _p(
            "timing",
            "composite",
            "components",
            "Composite timing components",
            "Timing overlays combined by the composite timing stage, in application order.",
            ParameterType.MULTISELECT,
            ("tsmom", "kama"),
            choices=timing_names,
            tunable=False,
            yaml_encoding=ENCODING_COMMA_STRINGS,
            min_items=1,
            max_items=4,
        ),
    )


# ---------------------------------------------------------------------------
# Catalogue rows: risk overlays (R_t)
# ---------------------------------------------------------------------------


def _risk_specs() -> tuple[ParameterSpec, ...]:
    return (
        _p(
            "risk",
            "quant_risk",
            "vol_target",
            "Portfolio volatility target",
            "Ex-ante annualised volatility target for the quantitative overlay's gross-weight "
            "scaling.",
            ParameterType.FLOAT,
            0.12,
            min=0.01,
            max=0.60,
            step=0.01,
            unit="annual vol",
        ),
        _p(
            "risk",
            "quant_risk",
            "ewma_halflife_days",
            "Covariance EWMA half-life",
            "Half-life of the exponentially weighted covariance used for volatility targeting.",
            ParameterType.INT,
            20,
            min=2,
            max=252,
            step=1,
            unit="days",
        ),
        _p(
            "risk",
            "quant_risk",
            "max_gross_leverage",
            "Max gross leverage",
            "Hard cap on gross leverage after the volatility-target scaling.",
            ParameterType.FLOAT,
            3.0,
            min=0.1,
            max=10.0,
            step=0.1,
            unit="x equity",
        ),
        _p(
            "risk",
            "quant_risk",
            "max_single_instrument_weight",
            "Max single-instrument weight",
            "Concentration cap on any one futures root.",
            ParameterType.FLOAT,
            0.35,
            min=0.01,
            max=1.0,
            step=0.01,
        ),
        _p(
            "risk",
            "quant_risk",
            "max_sector_weight",
            "Max sector weight",
            "Correlation-cluster cap: aggregate notional allowed per instrument-master sector "
            "(e.g. all equity index roots).",
            ParameterType.FLOAT,
            0.40,
            min=0.01,
            max=1.0,
            step=0.01,
        ),
        _p(
            "risk",
            "quant_risk",
            "drawdown_limit_pct",
            "Drawdown limit",
            "Peak-to-trough drawdown that triggers the overlay's de-risking projection.",
            ParameterType.FLOAT,
            0.20,
            min=0.01,
            max=0.90,
            step=0.01,
            unit="drawdown",
        ),
        _p(
            "risk",
            "quant_risk",
            "daily_max_loss_pct",
            "Daily max-loss kill switch",
            "Intraday loss (fraction of starting daily NLV) that flattens the book and locks "
            "execution for the session.",
            ParameterType.FLOAT,
            0.03,
            min=0.001,
            max=0.30,
            step=0.001,
            unit="daily loss",
        ),
        _p(
            "risk",
            "quant_risk",
            "trailing_stop_pct",
            "Trailing stop",
            "High-water-mark trailing stop distance per position and at portfolio level.",
            ParameterType.FLOAT,
            0.10,
            min=0.01,
            max=0.90,
            step=0.01,
            unit="drawdown",
        ),
        _p(
            "risk",
            "quant_risk",
            "volatility_targeting_enabled",
            "Enable volatility targeting",
            "Toggle the EWMA covariance volatility-target scaling.",
            ParameterType.BOOL,
            True,
        ),
        _p(
            "risk",
            "quant_risk",
            "daily_loss_kill_switch_enabled",
            "Enable daily kill switch",
            "Toggle the daily max-loss flatten-and-lock behaviour.",
            ParameterType.BOOL,
            True,
        ),
        _p(
            "risk",
            "llm_risk",
            "confidence_threshold",
            "Confidence floor (tau)",
            "LLM confidence below which an assessment is neutralised as a likely hallucination.",
            ParameterType.FLOAT,
            0.70,
            min=0.0,
            max=1.0,
            step=0.01,
        ),
        _p(
            "risk",
            "llm_risk",
            "variance_max",
            "Multi-sample variance ceiling",
            "Maximum variance across LLM samples before the assessment is neutralised.",
            ParameterType.FLOAT,
            0.25,
            min=0.001,
            max=5.0,
            step=0.01,
        ),
        _p(
            "risk",
            "llm_risk",
            "sample_count",
            "LLM samples (K)",
            "Number of completions sampled for the empirical confidence/variance filter.",
            ParameterType.INT,
            3,
            min=1,
            max=10,
            step=1,
            tunable=False,
        ),
        _p(
            "risk",
            "llm_risk",
            "safe_score",
            "Safe semantic risk score (theta)",
            "Semantic risk score at or below which the circuit breaker treats the root as safe.",
            ParameterType.FLOAT,
            1.5,
            min=1.0,
            max=4.0,
            step=0.1,
        ),
        _p(
            "risk",
            "llm_risk",
            "critical_score",
            "Critical semantic risk score (kappa)",
            "Semantic risk score at or above which the deterministic circuit breaker forces the "
            "leg flat.",
            ParameterType.FLOAT,
            4.5,
            min=2.0,
            max=5.0,
            step=0.1,
        ),
        _p(
            "risk",
            "llm_risk",
            "risk_penalty_lambda",
            "Semantic risk penalty (lambda)",
            "Lagrangian weight applied to the semantic-risk reward penalty during DRL training.",
            ParameterType.FLOAT,
            0.5,
            min=0.0,
            max=10.0,
            step=0.05,
        ),
        _p(
            "risk",
            "llm_risk",
            "macro_lockout_minutes_before",
            "Macro lockout (before)",
            "Minutes before a Tier-1 macro release during which the lockout window caps/flattens "
            "exposure.",
            ParameterType.INT,
            30,
            min=0,
            max=240,
            step=5,
            unit="minutes",
        ),
        _p(
            "risk",
            "llm_risk",
            "macro_lockout_minutes_after",
            "Macro lockout (after)",
            "Minutes after a Tier-1 macro release during which the lockout window remains active.",
            ParameterType.INT,
            30,
            min=0,
            max=240,
            step=5,
            unit="minutes",
        ),
        _p(
            "risk",
            "llm_risk",
            "macro_lockout_action",
            "Macro lockout action",
            "Flatten the affected roots or merely cap portfolio leverage inside the lockout "
            "window.",
            ParameterType.SELECT,
            "flatten",
            choices=("flatten", "cap_leverage"),
            tunable=False,
        ),
        _p(
            "risk",
            "llm_risk",
            "macro_lockout_leverage_cap",
            "Macro lockout leverage cap",
            "Leverage cap applied when the macro lockout action is cap_leverage.",
            ParameterType.FLOAT,
            0.25,
            min=0.01,
            max=1.0,
            step=0.01,
        ),
        _p(
            "risk",
            "llm_risk",
            "sampling_temperature",
            "LLM sampling temperature",
            "Sampling temperature used when requesting risk assessments.",
            ParameterType.FLOAT,
            0.7,
            min=0.0,
            max=2.0,
            step=0.05,
        ),
        _p(
            "risk",
            "llm_risk",
            "cache_ttl_seconds",
            "LLM cache TTL",
            "Time-to-live of the cached assessment set; the synchronous pipeline refreshes it in "
            "the background.",
            ParameterType.FLOAT,
            60.0,
            min=0.0,
            max=3_600.0,
            step=5.0,
            unit="seconds",
        ),
    )


# ---------------------------------------------------------------------------
# Catalogue rows: contract translator
# ---------------------------------------------------------------------------


def _translator_specs() -> tuple[ParameterSpec, ...]:
    return (
        _p(
            "translator",
            "translator",
            "rounding_mode",
            "Contract rounding mode",
            "How fractional target contracts are rounded: floor toward zero (conservative), "
            "nearest (with rebalance buffer) or ceil away from zero.",
            ParameterType.SELECT,
            "nearest",
            choices=("floor", "nearest", "ceil"),
            tunable=False,
            location=LOCATION_TRANSLATOR,
        ),
        _p(
            "translator",
            "translator",
            "rebalance_buffer_contracts",
            "Rebalance buffer",
            "Dead-band in contracts applied to nearest rounding so marginal increments do not "
            "generate churn trades.",
            ParameterType.FLOAT,
            2.0,
            min=0.0,
            max=10.0,
            step=0.5,
            unit="contracts",
            location=LOCATION_TRANSLATOR,
        ),
        _p(
            "translator",
            "translator",
            "max_margin_utilization",
            "Max margin utilisation (eta)",
            "Hard initial-margin ceiling; targets are uniformly scaled down when projected "
            "utilisation would exceed it.",
            ParameterType.FLOAT,
            0.60,
            min=0.05,
            max=1.0,
            step=0.01,
            unit="margin/equity",
            location=LOCATION_TRANSLATOR,
        ),
        _p(
            "translator",
            "translator",
            "enable_micro_routing",
            "Enable micro routing",
            "Route residual fractional exposure into paired micro contracts (ES→MES, CL→MCL) to "
            "minimise tracking error.",
            ParameterType.BOOL,
            True,
            location=LOCATION_TRANSLATOR,
        ),
        _p(
            "translator",
            "translator",
            "min_micro_notional_ratio",
            "Micro routing dust floor",
            "Minimum residual notional (fraction of one micro contract) required before routing "
            "into micro contracts.",
            ParameterType.FLOAT,
            0.0,
            min=0.0,
            max=1.0,
            step=0.01,
            location=LOCATION_TRANSLATOR,
        ),
    )


# ---------------------------------------------------------------------------
# Shared default dictionaries (also used by the presets)
# ---------------------------------------------------------------------------

_TRANSLATOR_DEFAULTS: dict[str, object] = {
    "rounding_mode": "nearest",
    "rebalance_buffer_contracts": 2.0,
    "max_margin_utilization": 0.60,
    "enable_micro_routing": True,
    "min_micro_notional_ratio": 0.0,
}
_QUANT_RISK_DEFAULTS: dict[str, object] = {
    "vol_target": 0.12,
    "ewma_halflife_days": 20,
    "max_single_instrument_weight": 0.35,
    "max_sector_weight": 0.40,
    "drawdown_limit_pct": 0.20,
    "daily_max_loss_pct": 0.03,
    # v1.5.1: make the curated default-selected quant-risk parameter explicit in
    # generated preset strategies (it was catalogued but omitted from the stage
    # params, so the optimizer had to inject it as a new parameter).
    "trailing_stop_pct": 0.10,
}
_LLM_RISK_DEFAULTS: dict[str, object] = {
    "confidence_threshold": 0.70,
    "safe_score": 1.5,
    "critical_score": 4.5,
    "risk_penalty_lambda": 0.5,
}


def _risk_stages() -> list[dict[str, object]]:
    return [
        {"name": "quant_risk", "params": dict(_QUANT_RISK_DEFAULTS)},
        {"name": "llm_risk", "params": dict(_LLM_RISK_DEFAULTS)},
    ]


def _liquidity_stage(**params: object) -> dict[str, object]:
    defaults: dict[str, object] = {
        "min_avg_dollar_volume": 50_000_000.0,
        "min_open_interest": 5_000.0,
        "top_n": 10,
        "min_history_days": 252,
    }
    defaults.update(params)
    return {"name": "liquidity", "params": defaults}


# ---------------------------------------------------------------------------
# Preset templates (Strategy Studio hypothesis templates)
# ---------------------------------------------------------------------------


def _preset_specs() -> tuple[PresetTemplate, ...]:
    return (
        PresetTemplate(
            id="trend_following_tsmom_kama",
            name="Trend Following (TSMOM + KAMA)",
            description=(
                "Classical volatility-targeted time-series momentum with an adaptive KAMA trend "
                "confirmation and the two-tier risk overlay."
            ),
            hypothesis=(
                "Futures prices trend over multi-week horizons. Sizing each root to a common "
                "volatility target and confirming with an adaptive moving average captures the "
                "trend premium while cutting whipsaw exposure."
            ),
            tags=("trend", "classical", "tsmom", "vol-target"),
            parameter_keys=(
                "timing.tsmom.lookback_fast",
                "timing.tsmom.lookback_slow",
                "timing.kama.window",
                "allocator.volatility_target.target_annual_volatility",
                "universe.liquidity.top_n",
            ),
            strategy={
                "id": "preset_trend_following_tsmom_kama",
                "description": "Trend Following (TSMOM + KAMA) preset",
                "tags": ["trend", "classical", "tsmom", "vol-target"],
                "universe": _liquidity_stage(top_n=8),
                "allocator": {
                    "name": "volatility_target",
                    "params": {
                        "target_annual_volatility": 0.12,
                        "covariance_lookback_days": 60,
                        "max_gross_leverage": 3.0,
                    },
                },
                "timing": [
                    {
                        "name": "tsmom",
                        "params": {"lookback_fast": 21, "lookback_slow": 126, "vol_lookback": 20},
                    },
                    {"name": "kama", "params": {"window": 10, "fast_period": 2, "slow_period": 30}},
                ],
                "risk": _risk_stages(),
                "translator": dict(_TRANSLATOR_DEFAULTS),
            },
        ),
        PresetTemplate(
            id="commodity_carry_term_structure",
            name="Commodity Carry & Term Structure",
            description=(
                "Risk-parity commodity book tilted towards backwardated markets and filtered by "
                "annualised roll yield."
            ),
            hypothesis=(
                "Backwardated futures pay positive roll yield to long holders. Scaling a "
                "risk-parity book by term-structure carry harvests the roll premium while keeping "
                "risk balanced across sectors."
            ),
            tags=("carry", "term-structure", "commodity", "risk-parity"),
            parameter_keys=(
                "timing.carry_filter.min_carry_bps",
                "timing.carry_filter.carry_weight",
                "timing.tsmom.lookback_slow",
                "allocator.risk_parity.target_annual_volatility",
                "risk.quant_risk.max_sector_weight",
            ),
            strategy={
                "id": "preset_commodity_carry_term_structure",
                "description": "Commodity Carry & Term Structure preset",
                "tags": ["carry", "term-structure", "commodity", "risk-parity"],
                "universe": _liquidity_stage(top_n=10, min_open_interest=1_000.0),
                "allocator": {
                    "name": "risk_parity",
                    "params": {
                        "target_annual_volatility": 0.10,
                        "max_gross_leverage": 3.0,
                        "covariance_lookback_days": 60,
                    },
                },
                "timing": [
                    {
                        "name": "carry_filter",
                        "params": {"min_carry_bps": -50.0, "carry_weight": 1.0},
                    },
                    {
                        "name": "tsmom",
                        "params": {
                            "lookback_fast": 63,
                            "lookback_slow": 252,
                            "vol_lookback": 60,
                            "require_slow_agreement": False,
                        },
                    },
                ],
                "risk": _risk_stages(),
                "translator": dict(_TRANSLATOR_DEFAULTS),
            },
        ),
        PresetTemplate(
            id="panic_mean_reversion",
            name="Panic Mean-Reversion (RSI + Bollinger)",
            description=(
                "Contrarian reversion book that fades RSI/Bollinger extremes but freezes when the "
                "turbulence index signals a structural break."
            ),
            hypothesis=(
                "Sharp dislocations in liquid futures tend to partially revert, but only when the "
                "moves are noise rather than regime breaks. A turbulence veto keeps the overlay "
                "out of genuine crises."
            ),
            tags=("mean-reversion", "contrarian", "rsi", "bollinger", "turbulence"),
            parameter_keys=(
                "timing.mean_reversion.rsi_period",
                "timing.mean_reversion.oversold_threshold",
                "timing.mean_reversion.overbought_threshold",
                "timing.mean_reversion.turbulence_veto_threshold",
                "timing.mean_reversion.reversion_strength",
                "allocator.risk_parity.target_annual_volatility",
            ),
            strategy={
                "id": "preset_panic_mean_reversion",
                "description": "Panic Mean-Reversion (RSI + Bollinger) preset",
                "tags": ["mean-reversion", "contrarian", "rsi", "bollinger", "turbulence"],
                "universe": _liquidity_stage(top_n=10),
                "allocator": {
                    "name": "risk_parity",
                    "params": {
                        "target_annual_volatility": 0.10,
                        "max_gross_leverage": 3.0,
                        "covariance_lookback_days": 60,
                    },
                },
                "timing": [
                    {
                        "name": "mean_reversion",
                        "params": {
                            "rsi_period": 14,
                            "oversold_threshold": 25.0,
                            "overbought_threshold": 75.0,
                            "bollinger_window": 20,
                            "bollinger_std": 2.0,
                            "turbulence_veto_threshold": 0.95,
                            "reversion_strength": 0.5,
                        },
                    }
                ],
                "risk": _risk_stages(),
                "translator": dict(_TRANSLATOR_DEFAULTS),
            },
        ),
        PresetTemplate(
            id="drl_neural_policy_llm_risk",
            name="DRL Neural Policy + LLM Risk",
            description=(
                "PPO policy trained on the futures environment with the LLM Risk-First constraint "
                "engine and quantitative volatility targeting."
            ),
            hypothesis=(
                "A reinforcement-learning policy can learn non-linear cross-asset allocation "
                "dynamics that classical rules miss, while the deterministic LLM Risk-First "
                "circuit breaker keeps tail risk bounded."
            ),
            tags=("drl", "ppo", "llm-risk", "neural"),
            parameter_keys=(
                "allocator.drl.algorithm",
                "allocator.drl.total_timesteps",
                "allocator.drl.learning_rate",
                "allocator.drl.risk_penalty_lambda",
                "allocator.drl.reward_risk_penalty_lambda",
                "risk.llm_risk.critical_score",
                "timing.tsmom.lookback_fast",
            ),
            strategy={
                "id": "preset_drl_neural_policy_llm_risk",
                "description": "DRL Neural Policy + LLM Risk preset",
                "tags": ["drl", "ppo", "llm-risk", "neural"],
                "universe": _liquidity_stage(top_n=10),
                "allocator": {
                    "name": "drl",
                    "params": {
                        "algorithm": "PPO",
                        "policy": "MlpPolicy",
                        "total_timesteps": 100_000,
                        "learning_rate": 0.0003,
                        "gamma": 0.99,
                        "seed": 7,
                    },
                },
                "timing": [
                    {
                        "name": "tsmom",
                        "params": {"lookback_fast": 21, "lookback_slow": 126, "vol_lookback": 20},
                    }
                ],
                "risk": _risk_stages(),
                "translator": dict(_TRANSLATOR_DEFAULTS),
                "drl": {
                    "algorithm": "PPO",
                    "policy": "MlpPolicy",
                    "total_timesteps": 100_000,
                    "learning_rate": 0.0003,
                    "gamma": 0.99,
                    "seed": 7,
                    "reward_risk_penalty_lambda": 0.5,
                    "normalize_observations": True,
                    "model_path": None,
                },
            },
        ),
    )


# ---------------------------------------------------------------------------
# Instrument-master -> Studio universe metadata
# ---------------------------------------------------------------------------


def _sector_group_for(spec: InstrumentSpec) -> str:
    """Map an instrument spec to one of the Studio universe sector groups."""
    raw_sector = spec.sector.strip().lower()
    group = SECTOR_GROUP_ALIASES.get(raw_sector)
    if group is not None:
        return group
    group = ASSET_CLASS_GROUPS.get(spec.asset_class.value)
    return group if group is not None else "other"


def _universe_root(spec: InstrumentSpec) -> UniverseRoot:
    return UniverseRoot(
        root=spec.root,
        name=spec.name,
        asset_class=spec.asset_class.value,
        sector=spec.sector,
        sector_group=_sector_group_for(spec),
        exchange=spec.exchange,
        multiplier=spec.multiplier,
        tick_size=spec.tick_size,
        micro_root=spec.micro_root,
        standard_root=spec.standard_root,
        currency=spec.currency,
    )


def universe_roots(
    instrument_master: InstrumentsConfig | None = None,
) -> tuple[UniverseRoot, ...]:
    """Every futures root in the instrument master, in declaration order."""
    master = instrument_master if instrument_master is not None else load_instruments()
    return tuple(_universe_root(spec) for spec in master.instruments.values())


def universe_sectors(
    instrument_master: InstrumentsConfig | None = None,
) -> tuple[UniverseSector, ...]:
    """Sector-grouped universe metadata for the Studio universe explorer.

    Returns ``All`` first and then the seven canonical asset-class groups
    (Equities/Rates/Energy/Metals/FX/Agri/Crypto) — empty groups are included so
    the UI dropdown is stable — plus an ``Other`` group only when a root cannot be
    mapped to a canonical group. Every root appears exactly once outside ``All``.
    """
    roots = universe_roots(instrument_master)
    buckets: dict[str, list[UniverseRoot]] = {
        group_id: [] for group_id, _label, _description in UNIVERSE_SECTOR_GROUPS
    }
    for root in roots:
        buckets.setdefault(root.sector_group, []).append(root)
    sectors: list[UniverseSector] = []
    for group_id, label, description in UNIVERSE_SECTOR_GROUPS:
        group_roots = tuple(buckets.pop(group_id, ()))
        sectors.append(
            UniverseSector(
                id=group_id,
                label=label,
                description=description,
                roots=tuple(roots) if group_id == "all" else group_roots,
            )
        )
    leftover = tuple(root for remaining in buckets.values() for root in remaining)
    if leftover:
        sectors.append(
            UniverseSector(
                id="other",
                label="Other",
                description="Roots whose instrument-master sector/asset class has no Studio group",
                roots=leftover,
            )
        )
    return tuple(sectors)


def universe_metadata(
    instrument_master: InstrumentsConfig | None = None,
) -> UniverseMetadata:
    """JSON-friendly sector-grouped instrument-master metadata for the Studio."""
    return UniverseMetadata(
        sectors=universe_sectors(instrument_master),
        roots=universe_roots(instrument_master),
    )


# ---------------------------------------------------------------------------
# Stage dropdown choices (derived from the live registries)
# ---------------------------------------------------------------------------


def _ordered_choice_names(available: Sequence[str], stage_id: str) -> tuple[str, ...]:
    preferred = _STAGE_CHOICE_ORDER.get(stage_id, ())
    ordered = [name for name in preferred if name in available]
    ordered.extend(name for name in available if name not in ordered)
    return tuple(ordered)


def _registry_choice_names(stage_id: str) -> tuple[str, ...]:
    """Selectable implementation names for a stage, derived from its registry."""
    if stage_id == "universe":
        from src.strategies.selectors import SELECTOR_REGISTRY  # local import avoids cycles

        return _ordered_choice_names(tuple(SELECTOR_REGISTRY), stage_id)
    if stage_id == "allocator":
        from src.strategies.allocators import ALLOCATOR_REGISTRY  # local import avoids cycles

        return _ordered_choice_names(tuple(ALLOCATOR_REGISTRY), stage_id)
    if stage_id == "timing":
        from src.strategies.timing import TIMING_REGISTRY  # local import avoids cycles

        return _ordered_choice_names(tuple(TIMING_REGISTRY), stage_id)
    return ()


def _stage_choices(stage_id: str, parameters: tuple[ParameterSpec, ...]) -> tuple[StageOption, ...]:
    names = _registry_choice_names(stage_id)
    if not names:
        # risk / translator: derive from the catalogued sub-stage names (stable order).
        names = tuple(
            dict.fromkeys(spec.stage_name for spec in parameters if spec.stage == stage_id)
        )
    labels = STAGE_OPTION_LABELS.get(stage_id, {})
    options: list[StageOption] = []
    for name in names:
        parameter_keys = tuple(
            spec.key for spec in parameters if spec.stage == stage_id and spec.stage_name == name
        )
        label, description = labels.get(
            name, (name.replace("_", " ").title(), f"{stage_id} stage option {name}")
        )
        options.append(
            StageOption(
                name=name,
                label=label,
                description=description,
                parameter_keys=parameter_keys,
            )
        )
    return tuple(options)


# ---------------------------------------------------------------------------
# Catalog construction (cached per instrument-master fingerprint)
# ---------------------------------------------------------------------------

type MasterFingerprint = tuple[object, ...]

_CATALOG_CACHE: dict[MasterFingerprint, ParameterCatalog] = {}


def _master_fingerprint(master: InstrumentsConfig) -> MasterFingerprint:
    """Stable, hashable signature of the instrument master's content."""
    return (
        master.version,
        tuple(master.roots()),
        tuple(master.sectors()),
        tuple(
            (
                spec.root,
                spec.name,
                spec.asset_class.value,
                spec.sector,
                spec.exchange,
                spec.multiplier,
                spec.tick_size,
                spec.currency,
                spec.micro_root,
                spec.standard_root,
            )
            for spec in master.instruments.values()
        ),
    )


def _apply_optuna_selection(
    parameters: tuple[ParameterSpec, ...],
) -> tuple[ParameterSpec, ...]:
    """Apply the curated default-selection and structural non-tunable policy.

    Centralised here (instead of per-row flags) so the policy is one auditable
    list: exactly :data:`DEFAULT_SELECTED_PARAMETER_KEYS` are pre-selected and
    every :data:`NON_TUNABLE_PARAMETER_KEYS` entry is excluded from Optuna search
    spaces even if a row declared ``tunable=True``. Unknown/overlapping keys raise
    immediately (a typo can never silently widen the search space).
    """
    known = {spec.key for spec in parameters}
    curated = frozenset(DEFAULT_SELECTED_PARAMETER_KEYS)
    unknown = sorted((curated | NON_TUNABLE_PARAMETER_KEYS) - known)
    if unknown:
        raise ValueError(f"optuna selection policy references unknown parameter keys: {unknown!r}")
    conflicts = sorted(curated & NON_TUNABLE_PARAMETER_KEYS)
    if conflicts:
        raise ValueError(
            f"curated default parameters cannot be structural/non-tunable: {conflicts!r}"
        )
    adjusted: list[ParameterSpec] = []
    for spec in parameters:
        tunable = spec.optuna_tunable and spec.key not in NON_TUNABLE_PARAMETER_KEYS
        selected = spec.key in curated
        if selected and not tunable:  # pragma: no cover - guarded by the conflict check
            raise ValueError(f"{spec.key}: curated default selection requires tunable=True")
        adjusted.append(replace(spec, optuna_tunable=tunable, optuna_default_selected=selected))
    return tuple(adjusted)


def _apply_ui_metadata(
    parameters: tuple[ParameterSpec, ...],
) -> tuple[ParameterSpec, ...]:
    """Attach additive Studio metadata (math formula + advanced-parameter flag).

    Centralised so the frontend can render allocator math tooltips and an
    "Advanced Parameters" section from the catalog instead of hard-coding the
    mapping. Purely additive: existing keys/flags are untouched.
    """
    adjusted: list[ParameterSpec] = []
    for spec in parameters:
        formula = ALLOCATOR_FORMULAS.get(spec.stage_name) if spec.stage == "allocator" else None
        advanced = spec.key in ADVANCED_PARAMETER_KEYS
        adjusted.append(replace(spec, formula=formula, advanced=advanced))
    return tuple(adjusted)


def _build_catalog(master: InstrumentsConfig) -> ParameterCatalog:
    parameters: tuple[ParameterSpec, ...] = _apply_ui_metadata(
        _apply_optuna_selection(
            (
                *_universe_specs(master),
                *_allocator_specs(),
                *_timing_specs(),
                *_risk_specs(),
                *_translator_specs(),
            )
        )
    )
    stages = tuple(
        CatalogStage(
            id=stage_id,
            label=label,
            description=description,
            parameter_keys=tuple(spec.key for spec in parameters if spec.stage == stage_id),
            choices=_stage_choices(stage_id, parameters),
        )
        for stage_id, label, description in _STAGE_METADATA
    )
    catalog = ParameterCatalog(
        version=CATALOG_VERSION,
        stages=stages,
        parameters=parameters,
        presets=_preset_specs(),
        instrument_master=master,
    )
    catalog.validate()
    return catalog


_STAGE_METADATA: tuple[tuple[str, str, str], ...] = (
    (
        "universe",
        "Universe selection (S_t)",
        "Which futures roots are tradable: liquidity, open-interest and macro-regime filters.",
    ),
    (
        "allocator",
        "Portfolio allocation (A_t)",
        "How the target notional weights are sized: equal weight, risk parity, mean-variance, "
        "volatility targeting or a DRL policy.",
    ),
    (
        "timing",
        "Timing & signals (T_t)",
        "Directional and calendar overlays that scale the allocator's weights: TSMOM, KAMA, "
        "carry and mean-reversion.",
    ),
    (
        "risk",
        "Risk & LLM (R_t)",
        "Two-tier risk constraints: quantitative volatility/drawdown controls and the LLM "
        "Risk-First semantic circuit breaker.",
    ),
    (
        "translator",
        "Contract translator",
        "Deterministic mapping from approved weights to discrete futures contracts, including "
        "micro routing and the initial-margin ceiling.",
    ),
)


def parameter_catalog(
    instrument_master: InstrumentsConfig | None = None,
) -> ParameterCatalog:
    """Return the validated catalog for an instrument master (cached per master).

    ``instrument_master`` defaults to the live :func:`configs.instruments.load_instruments`
    master, so root/sector choices always mirror ``configs/instruments.yaml``. One
    catalog is cached per instrument-master *fingerprint* (roots + sectors + per-root
    contract specs); :func:`refresh_parameter_catalog` clears the instruments cache
    and rebuilds after the YAML changes, and callers with a freshly loaded master may
    pass it explicitly.
    """
    master = instrument_master if instrument_master is not None else load_instruments()
    fingerprint = _master_fingerprint(master)
    cached = _CATALOG_CACHE.get(fingerprint)
    if cached is None:
        cached = _build_catalog(master)
        _CATALOG_CACHE[fingerprint] = cached
    return cached


def refresh_parameter_catalog() -> ParameterCatalog:
    """Reload the instrument master and return the rebuilt catalog.

    Use after ``configs/instruments.yaml`` is edited/committed so a long-running
    API/Studio process picks up new roots/sectors immediately.
    """
    from configs.instruments import clear_instruments_cache  # local import: reload hook

    clear_instruments_cache()
    return parameter_catalog()


def clear_parameter_catalog_cache() -> None:
    """Drop every cached catalog (test/configuration-reload helper)."""
    _CATALOG_CACHE.clear()


#: Module-level constant mirror of the import-time catalog (``parameter_catalog()``).
#: Long-running processes should call :func:`parameter_catalog`/``refresh_parameter_catalog``
#: after the instrument master changes instead of relying on this snapshot.
CATALOG: ParameterCatalog = parameter_catalog()


def catalog_dict(
    instrument_master: InstrumentsConfig | None = None,
) -> dict[str, object]:
    """JSON-ready payload for ``GET /api/strategies/schema/catalog``.

    Includes the additive ``universe`` (sector-grouped instrument-master metadata)
    and ``stage_choices`` (universe selector / allocator / timing / risk dropdowns)
    keys plus the Phase-10 ``version``/``stages``/``parameters``/``presets`` shape.
    """
    return parameter_catalog(instrument_master).to_dict()


# ---------------------------------------------------------------------------
# Strategy <-> catalogue binding helpers
# ---------------------------------------------------------------------------


def _catalog_lookup(key_or_spec: str | ParameterSpec) -> ParameterSpec:
    if isinstance(key_or_spec, ParameterSpec):
        return key_or_spec
    return parameter_catalog().get(str(key_or_spec))


def _segments(path: str) -> list[str | int]:
    """Split a dotted strategy path into mapping keys and list indices."""
    segments: list[str | int] = []
    for part in path.split("."):
        if part.isdigit():
            segments.append(int(part))
        else:
            segments.append(part)
    if not segments:  # pragma: no cover - paths are always non-empty
        raise KeyError(f"invalid empty parameter path {path!r}")
    return segments


def _node_at(root: object, segments: Sequence[str | int]) -> object:
    node: Any = root
    for segment in segments:
        if isinstance(segment, int):
            if not isinstance(node, (list, tuple)):  # pragma: no cover - defensive
                raise KeyError(f"path segment {segment!r} expects a list")
            try:
                node = node[segment]
            except IndexError as exc:
                raise KeyError(f"list index {segment!r} is out of range") from exc
        else:
            if not isinstance(node, Mapping):
                raise KeyError(f"path segment {segment!r} expects a mapping")
            try:
                node = node[segment]
            except KeyError as exc:
                raise KeyError(f"strategy is missing key {segment!r}") from exc
    return node


def _set_at(root: object, segments: Sequence[str | int], value: object) -> None:
    if not segments:  # pragma: no cover - defensive
        raise KeyError("cannot set an empty parameter path")
    parent = root if len(segments) == 1 else _node_at(root, segments[:-1])
    last = segments[-1]
    if isinstance(last, int):
        if not isinstance(parent, list):  # pragma: no cover - defensive
            raise KeyError(f"path segment {last!r} expects a list")
        parent[last] = value
        return
    if not isinstance(parent, dict):
        raise KeyError(f"path segment {last!r} expects a mapping")
    parent[last] = value


def _stage_entries(strategy: Mapping[str, object], stage: str) -> Sequence[object]:
    entries = strategy.get(stage)
    if entries is None:
        raise KeyError(f"strategy has no {stage!r} stage")
    if isinstance(entries, (list, tuple)):
        return entries
    return ()


def _binding_segments(strategy: Mapping[str, object], spec: ParameterSpec) -> list[str | int]:
    """Resolve the concrete dotted path segments of ``spec`` in ``strategy``."""
    if spec.location == LOCATION_TRANSLATOR:
        if not isinstance(strategy.get("translator"), Mapping):
            raise KeyError("strategy has no 'translator' block")
        return ["translator", spec.param]
    if spec.location == LOCATION_DRL_BLOCK:
        if not isinstance(strategy.get("drl"), Mapping):
            raise KeyError(
                f"strategy has no 'drl' block; add a drl: section before binding {spec.key!r}"
            )
        return ["drl", spec.param]
    if spec.stage in {"universe", "allocator"}:
        stage = strategy.get(spec.stage)
        if not isinstance(stage, Mapping):
            raise KeyError(f"strategy has no {spec.stage!r} stage")
        name = str(stage.get("name", "")).strip().lower()
        if name != spec.stage_name:
            raise KeyError(
                f"strategy {spec.stage!r} stage is {name!r}, not {spec.stage_name!r} "
                f"(cannot bind {spec.key!r})"
            )
        return [spec.stage, "params", spec.param]
    entries = _stage_entries(strategy, spec.stage)
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            continue
        name = str(entry.get("name", "")).strip().lower()
        if name == spec.stage_name:
            return [spec.stage, index, "params", spec.param]
    raise KeyError(
        f"strategy has no {spec.stage_name!r} stage in the {spec.stage!r} list "
        f"(cannot bind {spec.key!r})"
    )


def resolve_param_path(strategy: Mapping[str, object], key: str) -> str:
    """Return the dotted path of catalog ``key`` inside a given strategy dict.

    The path uses mapping keys and *numeric list indices* for the ordered
    ``timing``/``risk`` stage lists, e.g. ``"timing.1.params.lookback_fast"`` for
    the *KAMA* stage of a strategy whose timing list is ``[tsmom, kama]``. This is
    the path the Optuna optimizer samples and the Studio writes.
    """
    spec = _catalog_lookup(key)
    segments = _binding_segments(strategy, spec)
    return ".".join(str(segment) for segment in segments)


def parameter_keys_for_strategy(strategy: Mapping[str, object]) -> tuple[str, ...]:
    """Catalog keys that are bindable to this particular strategy."""
    keys: list[str] = []
    for spec in parameter_catalog().parameters:
        try:
            _binding_segments(strategy, spec)
        except KeyError:
            continue
        keys.append(spec.key)
    return tuple(keys)


def get_parameter_value(strategy: Mapping[str, object], key: str) -> object:
    """Read ``key`` from a strategy dict and decode it into a UI-native value."""
    spec = _catalog_lookup(key)
    segments = _binding_segments(strategy, spec)
    raw = _node_at(strategy, segments)
    return decode_parameter_value(spec, raw)


# ---------------------------------------------------------------------------
# Active-strategy parameter filtering (Studio optimizer search space)
# ---------------------------------------------------------------------------


def _strategy_mapping(strategy: Mapping[str, object] | StrategyConfig) -> Mapping[str, object]:
    """Accept either a strategy dict (YAML/JSON form) or a validated config."""
    if isinstance(strategy, StrategyConfig):
        return strategy.model_dump(mode="json")
    return strategy


def _lower_choice(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip().lower()
    return None


def _choice_names(value: object) -> tuple[str, ...]:
    """Normalise a comma string / sequence param value into lowercase names."""
    if isinstance(value, str):
        return tuple(part.strip().lower() for part in value.split(",") if part.strip())
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return tuple(str(item).strip().lower() for item in value if str(item).strip())
    return ()


def _active_stage_choices(
    strategy: Mapping[str, object] | StrategyConfig,
) -> Mapping[str, tuple[str, ...]]:
    """Stage id -> active implementation names for one concrete strategy."""
    mapping = _strategy_mapping(strategy)
    active: dict[str, set[str]] = {
        "universe": set(),
        "allocator": set(),
        "timing": set(),
        "risk": set(),
        "translator": set(),
    }
    for stage_id in ("universe", "allocator"):
        block = mapping.get(stage_id)
        if isinstance(block, Mapping):
            name = _lower_choice(block.get("name"))
            if name is not None:
                active[stage_id].add(name)
    for stage_id in ("timing", "risk"):
        entries = mapping.get(stage_id)
        if not isinstance(entries, (list, tuple)):
            continue
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            name = _lower_choice(entry.get("name"))
            if name is None:
                continue
            active[stage_id].add(name)
            if stage_id == "timing" and name == "composite":
                params = entry.get("params")
                if isinstance(params, Mapping):
                    # A composite stage activates its configured components too.
                    active[stage_id].update(_choice_names(params.get("components")))
    if isinstance(mapping.get("translator"), Mapping):
        active["translator"].add("translator")
    return {stage_id: tuple(sorted(names)) for stage_id, names in active.items()}


def parameters_for_strategy(
    strategy: Mapping[str, object] | StrategyConfig,
) -> tuple[ParameterSpec, ...]:
    """Active, tunable parameter specs for one concrete strategy.

    Only parameters of the strategy's *selected* universe selector/allocator and
    the timing/risk stages actually present are returned (composite timing expands
    to its components); inactive selectors/overlays and every non-tunable
    structural parameter are excluded. This is the exact Optuna search-space
    candidate set for the strategy.
    """
    active = _active_stage_choices(strategy)
    return tuple(
        spec
        for spec in parameter_catalog().parameters
        if spec.optuna_tunable and spec.stage_name in active.get(spec.stage, ())
    )


def active_parameter_keys(
    strategy: Mapping[str, object] | StrategyConfig,
) -> tuple[str, ...]:
    """Catalog keys of :func:`parameters_for_strategy` (active + tunable)."""
    return tuple(spec.key for spec in parameters_for_strategy(strategy))


def default_selected_parameter_keys(
    strategy: Mapping[str, object] | StrategyConfig,
) -> tuple[str, ...]:
    """Curated default-selected keys that are active and tunable for this strategy."""
    catalog = parameter_catalog()
    return tuple(
        key for key in active_parameter_keys(strategy) if catalog.get(key).optuna_default_selected
    )


# ---------------------------------------------------------------------------
# Root-symbol validation (presets/configs vs the instrument master)
# ---------------------------------------------------------------------------

#: Stage-parameter keys that carry root-symbol lists in strategy configs.
ROOT_LIST_PARAMETER_KEYS: tuple[str, ...] = (
    "roots",
    "risk_off_roots",
)


def _roots_from_value(value: object) -> tuple[str, ...]:
    """Normalise a root-list param value (comma string, JSON object, sequence)."""
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ()
        if text.startswith("{"):
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                return ()
            return (
                tuple(str(root).strip().upper() for root in payload)
                if isinstance(payload, Mapping)
                else ()
            )
        return tuple(part.strip().upper() for part in text.split(",") if part.strip())
    if isinstance(value, Mapping):
        return tuple(str(root).strip().upper() for root in value)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return tuple(str(item).strip().upper() for item in value if str(item).strip())
    return ()


def strategy_root_symbols(
    strategy: Mapping[str, object] | StrategyConfig,
) -> tuple[str, ...]:
    """Every root symbol referenced by a strategy dict's root lists.

    Covers ``universe.params.{roots,risk_off_roots}`` and the
    same keys inside ``timing``/``risk`` stage params (e.g. a regime
    overlay's ``risk_off_roots``). Symbols are upper-cased, de-duplicated and
    returned in first-reference order.
    """
    mapping = _strategy_mapping(strategy)
    symbols: list[str] = []

    def add(values: Sequence[str]) -> None:
        for symbol in values:
            if symbol and symbol not in symbols:
                symbols.append(symbol)

    blocks: list[Mapping[str, object]] = []
    for stage_id in ("universe", "allocator"):
        block = mapping.get(stage_id)
        if isinstance(block, Mapping):
            blocks.append(block)
    for stage_id in ("timing", "risk"):
        entries = mapping.get(stage_id)
        if isinstance(entries, (list, tuple)):
            blocks.extend(entry for entry in entries if isinstance(entry, Mapping))
    for block in blocks:
        params = block.get("params")
        if not isinstance(params, Mapping):
            continue
        for key in ROOT_LIST_PARAMETER_KEYS:
            add(_roots_from_value(params.get(key)))
    return tuple(symbols)


def validate_strategy_roots(
    strategy: Mapping[str, object] | StrategyConfig,
    instrument_master: InstrumentsConfig | None = None,
) -> tuple[str, ...]:
    """Validate that every root a strategy references exists in the master.

    Returns the referenced symbols; raises ``ValueError`` naming any delisted /
    misspelt root. Presets, generated drafts and ``POST /api/strategies`` payloads
    can call this before persisting a strategy.
    """
    master = instrument_master if instrument_master is not None else load_instruments()
    symbols = strategy_root_symbols(strategy)
    unknown = [symbol for symbol in symbols if not master.has(symbol)]
    if unknown:
        raise ValueError(
            f"strategy references futures roots that are not in the instrument master: {unknown!r}"
        )
    return symbols


# ---------------------------------------------------------------------------
# Value validation / YAML <-> UI encoding
# ---------------------------------------------------------------------------


def _validate_json_object_value(
    spec: ParameterSpec, value: Mapping[object, object]
) -> dict[str, dict[int, float]]:
    allowed_roots = {str(choice) for choice in spec.choices}
    cleaned: dict[str, dict[int, float]] = {}
    for raw_root, curve in value.items():
        root = str(raw_root).strip().upper()
        if root not in allowed_roots:
            raise ValueError(
                f"{spec.key}: unknown root {raw_root!r} (choices: {sorted(allowed_roots)})"
            )
        if not isinstance(curve, Mapping):
            raise ValueError(f"{spec.key}: root {root!r} must map month -> bias")
        months: dict[int, float] = {}
        for raw_month, raw_bias in curve.items():
            try:
                month = int(raw_month)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{spec.key}: invalid month {raw_month!r}") from exc
            if not 1 <= month <= 12:
                raise ValueError(f"{spec.key}: month must be within 1..12, got {month}")
            try:
                bias = float(raw_bias)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{spec.key}: invalid bias {raw_bias!r}") from exc
            if not math.isfinite(bias) or bias < -1.0:
                raise ValueError(f"{spec.key}: bias must be finite and >= -1.0, got {bias!r}")
            months[month] = bias
        cleaned[root] = months
    return cleaned


def _coerce_choice(spec: ParameterSpec, item: object) -> CatalogScalar:
    if spec.choices and all(
        isinstance(choice, int) and not isinstance(choice, bool) for choice in spec.choices
    ):
        if isinstance(item, bool):
            raise ValueError(f"{spec.key}: expected an integer choice, got {item!r}")
        if isinstance(item, int):
            candidate: object = item
        elif isinstance(item, str) and item.strip().lstrip("-").isdigit():
            candidate = int(item.strip())
        else:
            raise ValueError(f"{spec.key}: expected an integer choice, got {item!r}")
        if candidate not in spec.choices:
            raise ValueError(f"{spec.key}: {candidate!r} is not one of {list(spec.choices)!r}")
        return int(candidate)  # type: ignore[arg-type]
    if not isinstance(item, str):
        raise ValueError(f"{spec.key}: expected a string choice, got {item!r}")
    text = item.strip()
    for choice in spec.choices:
        if isinstance(choice, str) and choice.lower() == text.lower():
            return choice
    raise ValueError(f"{spec.key}: {item!r} is not one of {list(spec.choices)!r}")


def _validate_multiselect(
    spec: ParameterSpec, value: object
) -> tuple[CatalogScalar, ...] | dict[str, dict[int, float]]:
    if spec.yaml_encoding == ENCODING_JSON_OBJECT:
        if isinstance(value, Mapping):
            return _validate_json_object_value(spec, value)
        if isinstance(value, str) and value.strip().startswith("{"):
            try:
                payload = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{spec.key}: invalid JSON object ({exc})") from exc
            if not isinstance(payload, Mapping):
                raise ValueError(f"{spec.key}: JSON form must be an object")
            return _validate_json_object_value(spec, payload)
    items: Sequence[object]
    if isinstance(value, str):
        items = [part for part in value.split(",") if part.strip()]
    elif isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
    else:
        raise ValueError(f"{spec.key}: expected a list of choices, got {value!r}")
    cleaned: list[CatalogScalar] = []
    for item in items:
        choice = _coerce_choice(spec, item)
        if choice not in cleaned:
            cleaned.append(choice)
    if spec.min_items is not None and len(cleaned) < spec.min_items:
        raise ValueError(f"{spec.key}: select at least {spec.min_items} value(s)")
    if spec.max_items is not None and len(cleaned) > spec.max_items:
        raise ValueError(f"{spec.key}: select at most {spec.max_items} value(s)")
    return tuple(cleaned)


def validate_parameter_value(key_or_spec: str | ParameterSpec, value: object) -> object:
    """Validate one UI-native parameter value (partial strategy update).

    Returns the normalised UI value. Raises ``ValueError`` naming the key when the
    type, numeric range, choices or cardinality are invalid. Optional parameters
    (``default is None``) accept ``None``.
    """
    spec = _catalog_lookup(key_or_spec)
    if value is None:
        if spec.default is None:
            return None
        raise ValueError(f"{spec.key}: value must not be None")
    kind = spec.type
    if kind is ParameterType.BOOL:
        if not isinstance(value, bool):
            raise ValueError(f"{spec.key}: expected a boolean, got {value!r}")
        return value
    if kind in {ParameterType.INT, ParameterType.FLOAT}:
        if isinstance(value, bool):
            raise ValueError(f"{spec.key}: expected a number, got {value!r}")
        if not isinstance(value, (int, float)):
            raise ValueError(f"{spec.key}: expected a number, got {value!r}")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{spec.key}: value must be finite, got {value!r}")
        if kind is ParameterType.INT and not number.is_integer():
            raise ValueError(f"{spec.key}: expected an integer, got {value!r}")
        normalised: float | int = int(number) if kind is ParameterType.INT else number
        if spec.min is not None and normalised < spec.min:
            raise ValueError(f"{spec.key}: value must be >= {spec.min}, got {value!r}")
        if spec.max is not None and normalised > spec.max:
            raise ValueError(f"{spec.key}: value must be <= {spec.max}, got {value!r}")
        return normalised
    if kind is ParameterType.SELECT:
        return _coerce_choice(spec, value)
    return _validate_multiselect(spec, value)


def decode_parameter_value(key_or_spec: str | ParameterSpec, raw: object) -> object:
    """Decode a YAML-native value into the UI-native representation."""
    spec = _catalog_lookup(key_or_spec)
    if raw is None:
        return None
    if spec.yaml_encoding == ENCODING_COMMA_INTS:
        if isinstance(raw, str):
            tokens = [token for token in raw.split(",") if token.strip()]
            return tuple(int(token.strip()) for token in tokens)
        if isinstance(raw, (list, tuple)):
            return tuple(int(item) for item in raw)
        raise ValueError(f"{spec.key}: expected a comma-separated string or list, got {raw!r}")
    if spec.yaml_encoding == ENCODING_COMMA_STRINGS:
        if isinstance(raw, str):
            return tuple(token.strip() for token in raw.split(",") if token.strip())
        if isinstance(raw, (list, tuple)):
            return tuple(str(item) for item in raw)
        raise ValueError(f"{spec.key}: expected a comma-separated string or list, got {raw!r}")
    if spec.yaml_encoding == ENCODING_JSON_OBJECT:
        if isinstance(raw, str):
            text = raw.strip()
            if text.startswith("{"):
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError as exc:  # pragma: no cover - validated upstream
                    raise ValueError(f"{spec.key}: invalid JSON object ({exc})") from exc
                if isinstance(payload, Mapping):
                    return _validate_json_object_value(spec, payload)
            return tuple(token.strip() for token in text.split(",") if token.strip())
        return _validate_json_object_value(spec, raw)  # type: ignore[arg-type]
    return raw


def encode_parameter_value(
    key_or_spec: str | ParameterSpec,
    value: object,
    *,
    instrument_master: InstrumentsConfig | None = None,
) -> CatalogScalar:
    """Validate a UI-native value and encode it for ``StageConfig.params``/YAML."""
    del (
        instrument_master
    )  # retained for backward-compatible call sites; no seasonal encoding remains
    spec = _catalog_lookup(key_or_spec)
    normalised = validate_parameter_value(spec, value)
    if spec.yaml_encoding == ENCODING_SCALAR:
        return normalised  # type: ignore[return-value]
    if spec.yaml_encoding == ENCODING_COMMA_INTS:
        assert isinstance(normalised, tuple)
        return ",".join(str(int(item)) for item in normalised)
    if spec.yaml_encoding == ENCODING_COMMA_STRINGS:
        assert isinstance(normalised, tuple)
        return ",".join(str(item) for item in normalised)
    # JSON object encoding (root -> {month: value}) for any remaining object params.
    if isinstance(normalised, Mapping):
        curves: dict[str, dict[str, float]] = {
            str(root): {str(int(month)): float(bias) for month, bias in curve.items()}
            for root, curve in normalised.items()
        }
        return json.dumps(curves, sort_keys=True)
    assert isinstance(normalised, tuple)
    return ",".join(str(item) for item in normalised)


def apply_parameter_overrides(
    strategy: Mapping[str, object],
    overrides: Mapping[str, object],
    *,
    instrument_master: InstrumentsConfig | None = None,
) -> dict[str, object]:
    """Deep-copy ``strategy`` and apply UI-native ``overrides`` as YAML-native values.

    Unknown keys, unbound stages, invalid values or non-serialisable choices raise
    ``ValueError``/``KeyError`` with the offending key named, so a partial Studio
    edit fails fast instead of producing an invalid YAML file.
    """
    updated: dict[str, object] = copy.deepcopy(dict(strategy))
    for key, value in overrides.items():
        spec = _catalog_lookup(key)
        encoded = encode_parameter_value(spec, value, instrument_master=instrument_master)
        segments = _segments(resolve_param_path(updated, spec.key))
        _set_at(updated, segments, encoded)
    return updated


__all__ = [
    "ASSET_CLASS_GROUPS",
    "CATALOG",
    "CATALOG_VERSION",
    "DEFAULT_SELECTED_PARAMETER_KEYS",
    "ENCODING_COMMA_INTS",
    "ENCODING_COMMA_STRINGS",
    "ENCODING_JSON_OBJECT",
    "ENCODING_SCALAR",
    "LOCATION_DRL_BLOCK",
    "LOCATION_STAGE_PARAMS",
    "LOCATION_TRANSLATOR",
    "NON_TUNABLE_PARAMETER_KEYS",
    "ROOT_LIST_PARAMETER_KEYS",
    "SECTOR_GROUP_ALIASES",
    "STAGE_OPTION_LABELS",
    "UNIVERSE_SECTOR_GROUPS",
    "CatalogDefault",
    "CatalogScalar",
    "CatalogStage",
    "ParameterCatalog",
    "ParameterSpec",
    "ParameterType",
    "PresetTemplate",
    "StageOption",
    "UniverseMetadata",
    "UniverseRoot",
    "UniverseSector",
    "active_parameter_keys",
    "apply_parameter_overrides",
    "catalog_dict",
    "clear_parameter_catalog_cache",
    "decode_parameter_value",
    "default_selected_parameter_keys",
    "encode_parameter_value",
    "get_parameter_value",
    "parameter_catalog",
    "parameter_keys_for_strategy",
    "parameters_for_strategy",
    "refresh_parameter_catalog",
    "resolve_param_path",
    "strategy_root_symbols",
    "universe_metadata",
    "universe_roots",
    "universe_sectors",
    "validate_parameter_value",
    "validate_strategy_roots",
]
