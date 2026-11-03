"""FinRL-X composable strategy pipeline (Module 2).

``w_t = R_t(T_t(A_t(S_t(X))))`` — universe selection, portfolio allocation,
timing/term-structure overlays and (injected) risk overlays, producing a fully
timed stage trace consumed by the backtester, the IBKR execution engine and the
dashboard's pipeline waterfall inspector.

Public surface
--------------
* :class:`ComposablePipeline`, :func:`build_pipeline`, :class:`IdentityRiskOverlay`
* selectors: :class:`LiquiditySelector`, :class:`RegimeSelector`,
  :class:`CompositeSelector` (+ :func:`build_selector`)
* allocators: :class:`EqualWeightAllocator`, :class:`RiskParityAllocator`,
  :class:`MeanVarianceAllocator`, :class:`VolatilityTargetAllocator`,
  :class:`DRLAllocator` (+ :func:`build_allocator`)
* timing: :class:`TSMOMTiming`, :class:`KAMATiming`, :class:`CarryFilter`,
  :class:`CompositeTiming` (+ :func:`build_timing`)

The Gymnasium DRL environment lives in :mod:`src.strategies.envs`
(``FuturesTradingEnv``) and is intentionally *not* imported here so that the
strategy package does not require ``gymnasium`` for classical strategies; import
it explicitly with ``from src.strategies.envs import FuturesTradingEnv``.
"""

from __future__ import annotations

from src.strategies.allocators import (
    ALLOCATOR_REGISTRY,
    DRLAllocator,
    EqualWeightAllocator,
    MeanVarianceAllocator,
    RiskParityAllocator,
    VolatilityTargetAllocator,
    build_allocator,
)
from src.strategies.parameter_catalog import (
    CATALOG,
    CatalogStage,
    ParameterCatalog,
    ParameterSpec,
    ParameterType,
    PresetTemplate,
    apply_parameter_overrides,
    catalog_dict,
    decode_parameter_value,
    encode_parameter_value,
    get_parameter_value,
    parameter_catalog,
    parameter_keys_for_strategy,
    resolve_param_path,
    validate_parameter_value,
)
from src.strategies.pipeline import (
    ComposablePipeline,
    IdentityRiskOverlay,
    build_pipeline,
)
from src.strategies.selectors import (
    SELECTOR_REGISTRY,
    CompositeSelector,
    LiquiditySelector,
    RegimeSelector,
    RegimeSnapshot,
    build_selector,
)
from src.strategies.timing import (
    ALL_MONTHS,
    DEFAULT_ROOT_MONTH_BIASES,
    TIMING_REGISTRY,
    CarryFilter,
    CompositeTiming,
    KAMATiming,
    MeanReversionTiming,
    SeasonalityTiming,
    TSMOMTiming,
    bollinger_bands,
    build_timing,
    kaufman_adaptive_moving_average,
    wilder_rsi,
)

__all__ = [
    "ALLOCATOR_REGISTRY",
    "ALL_MONTHS",
    "CATALOG",
    "DEFAULT_ROOT_MONTH_BIASES",
    "SELECTOR_REGISTRY",
    "TIMING_REGISTRY",
    "CarryFilter",
    "CatalogStage",
    "ComposablePipeline",
    "CompositeSelector",
    "CompositeTiming",
    "DRLAllocator",
    "EqualWeightAllocator",
    "IdentityRiskOverlay",
    "KAMATiming",
    "LiquiditySelector",
    "MeanReversionTiming",
    "MeanVarianceAllocator",
    "ParameterCatalog",
    "ParameterSpec",
    "ParameterType",
    "PresetTemplate",
    "RegimeSelector",
    "RegimeSnapshot",
    "RiskParityAllocator",
    "SeasonalityTiming",
    "TSMOMTiming",
    "VolatilityTargetAllocator",
    "apply_parameter_overrides",
    "bollinger_bands",
    "build_allocator",
    "build_pipeline",
    "build_selector",
    "build_timing",
    "catalog_dict",
    "decode_parameter_value",
    "encode_parameter_value",
    "get_parameter_value",
    "kaufman_adaptive_moving_average",
    "parameter_catalog",
    "parameter_keys_for_strategy",
    "resolve_param_path",
    "validate_parameter_value",
    "wilder_rsi",
]
