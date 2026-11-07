"""Timing overlays (``T_t``) and their declarative build registry.

Registry keys match the ``timing[].name`` values used by the strategy YAMLs:
``tsmom``, ``kama``, ``carry_filter``, ``mean_reversion`` (plus ``composite``
to nest them). Every overlay is a pure
``WeightVector -> WeightVector`` transformation so the pipeline can chain them in
the configured YAML order.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType

from configs.strategy import StageParam

from src.core.interfaces import TimingOverlay
from src.strategies.timing.carry import CARRY_FILTER_MODES, CarryFilter, build_carry_filter
from src.strategies.timing.composite import CompositeTiming, build_composite_timing
from src.strategies.timing.kama import (
    KAMATiming,
    build_kama_timing,
    kaufman_adaptive_moving_average,
)
from src.strategies.timing.mean_reversion import (
    MeanReversionTiming,
    bollinger_bands,
    build_mean_reversion_timing,
    wilder_rsi,
)
from src.strategies.timing.tsmom import (
    FAST_RETURN_FEATURES,
    SLOW_RETURN_FEATURES,
    TSMOMTiming,
    build_tsmom_timing,
)

type TimingFactory = Callable[[Mapping[str, StageParam]], TimingOverlay]

#: ``stage name -> factory`` used by :func:`build_timing`.
TIMING_REGISTRY: Mapping[str, TimingFactory] = MappingProxyType(
    {
        "tsmom": build_tsmom_timing,
        "kama": build_kama_timing,
        "carry_filter": build_carry_filter,
        "mean_reversion": build_mean_reversion_timing,
        "composite": build_composite_timing,
    }
)


def build_timing(
    name: str,
    params: Mapping[str, StageParam] | None = None,
) -> TimingOverlay:
    """Build a timing overlay from its registry ``name`` and stage parameters."""
    key = name.strip().lower()
    factory = TIMING_REGISTRY.get(key)
    if factory is None:
        raise ValueError(
            f"unknown timing overlay {name!r}; known overlays: {sorted(TIMING_REGISTRY)}"
        )
    return factory(dict(params or {}))


__all__ = [
    "CARRY_FILTER_MODES",
    "FAST_RETURN_FEATURES",
    "SLOW_RETURN_FEATURES",
    "TIMING_REGISTRY",
    "CarryFilter",
    "CompositeTiming",
    "KAMATiming",
    "MeanReversionTiming",
    "TSMOMTiming",
    "TimingFactory",
    "bollinger_bands",
    "build_carry_filter",
    "build_composite_timing",
    "build_kama_timing",
    "build_mean_reversion_timing",
    "build_timing",
    "build_tsmom_timing",
    "kaufman_adaptive_moving_average",
    "wilder_rsi",
]
