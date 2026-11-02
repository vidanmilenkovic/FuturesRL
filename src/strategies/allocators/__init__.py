"""Portfolio allocators (``A_t``) and their declarative build registry.

Registry keys match the ``allocator.name`` values used by the strategy YAMLs:
``equal_weight``, ``risk_parity``, ``mean_variance``, ``volatility_target`` and
``drl``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType

from configs.strategy import StageParam

from src.core.interfaces import PortfolioAllocator
from src.strategies.allocators.drl import (
    ACTION_MODES,
    SUPPORTED_ALGORITHMS,
    DRLAllocator,
    build_drl_allocator,
    import_stable_baselines3,
)
from src.strategies.allocators.equal_weight import (
    EqualWeightAllocator,
    build_equal_weight_allocator,
)
from src.strategies.allocators.mean_variance import (
    MeanVarianceAllocator,
    build_mean_variance_allocator,
)
from src.strategies.allocators.risk_parity import RiskParityAllocator, build_risk_parity_allocator
from src.strategies.allocators.volatility_target import (
    VolatilityTargetAllocator,
    build_volatility_target_allocator,
)

type AllocatorFactory = Callable[[Mapping[str, StageParam]], PortfolioAllocator]

#: ``stage name -> factory`` used by :func:`build_allocator`.
ALLOCATOR_REGISTRY: Mapping[str, AllocatorFactory] = MappingProxyType(
    {
        "equal_weight": build_equal_weight_allocator,
        "risk_parity": build_risk_parity_allocator,
        "mean_variance": build_mean_variance_allocator,
        "volatility_target": build_volatility_target_allocator,
        "drl": build_drl_allocator,
    }
)


def build_allocator(
    name: str,
    params: Mapping[str, StageParam] | None = None,
) -> PortfolioAllocator:
    """Build a portfolio allocator from its registry ``name`` and stage parameters."""
    key = name.strip().lower()
    factory = ALLOCATOR_REGISTRY.get(key)
    if factory is None:
        raise ValueError(
            f"unknown portfolio allocator {name!r}; known allocators: {sorted(ALLOCATOR_REGISTRY)}"
        )
    return factory(dict(params or {}))


__all__ = [
    "ACTION_MODES",
    "ALLOCATOR_REGISTRY",
    "SUPPORTED_ALGORITHMS",
    "AllocatorFactory",
    "DRLAllocator",
    "EqualWeightAllocator",
    "MeanVarianceAllocator",
    "RiskParityAllocator",
    "VolatilityTargetAllocator",
    "build_allocator",
    "build_drl_allocator",
    "build_equal_weight_allocator",
    "build_mean_variance_allocator",
    "build_risk_parity_allocator",
    "build_volatility_target_allocator",
    "import_stable_baselines3",
]
