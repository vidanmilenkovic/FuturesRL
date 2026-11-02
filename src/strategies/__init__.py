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
    TIMING_REGISTRY,
    CarryFilter,
    CompositeTiming,
    KAMATiming,
    TSMOMTiming,
    build_timing,
    kaufman_adaptive_moving_average,
)

__all__ = [
    "ALLOCATOR_REGISTRY",
    "SELECTOR_REGISTRY",
    "TIMING_REGISTRY",
    "CarryFilter",
    "ComposablePipeline",
    "CompositeSelector",
    "CompositeTiming",
    "DRLAllocator",
    "EqualWeightAllocator",
    "IdentityRiskOverlay",
    "KAMATiming",
    "LiquiditySelector",
    "MeanVarianceAllocator",
    "RegimeSelector",
    "RegimeSnapshot",
    "RiskParityAllocator",
    "TSMOMTiming",
    "VolatilityTargetAllocator",
    "build_allocator",
    "build_pipeline",
    "build_selector",
    "build_timing",
    "kaufman_adaptive_moving_average",
]
