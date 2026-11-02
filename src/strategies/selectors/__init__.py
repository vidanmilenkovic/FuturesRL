"""Universe selectors (``S_t``) and their declarative build registry.

The registry maps the stage names used by the strategy YAMLs to the concrete
implementations. Adding a selector means adding one factory here; nothing else
in the pipeline changes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType

from configs.instruments import InstrumentsConfig
from configs.strategy import StageParam

from src.core.interfaces import UniverseSelector
from src.strategies.selectors.composite import (
    COMPOSITE_MODES,
    CompositeSelector,
    build_composite_selector,
)
from src.strategies.selectors.liquidity import LiquiditySelector, build_liquidity_selector
from src.strategies.selectors.regime import RegimeSelector, RegimeSnapshot, build_regime_selector

type SelectorFactory = Callable[
    [Mapping[str, StageParam], InstrumentsConfig | None], UniverseSelector
]

#: ``stage name -> factory`` used by :func:`build_selector`.
SELECTOR_REGISTRY: Mapping[str, SelectorFactory] = MappingProxyType(
    {
        "liquidity": build_liquidity_selector,
        "regime": build_regime_selector,
        "composite": build_composite_selector,
    }
)


def build_selector(
    name: str,
    params: Mapping[str, StageParam] | None = None,
    instruments: InstrumentsConfig | None = None,
) -> UniverseSelector:
    """Build a universe selector from its registry ``name`` and stage parameters."""
    key = name.strip().lower()
    factory = SELECTOR_REGISTRY.get(key)
    if factory is None:
        raise ValueError(
            f"unknown universe selector {name!r}; known selectors: {sorted(SELECTOR_REGISTRY)}"
        )
    return factory(dict(params or {}), instruments)


__all__ = [
    "COMPOSITE_MODES",
    "SELECTOR_REGISTRY",
    "CompositeSelector",
    "LiquiditySelector",
    "RegimeSelector",
    "RegimeSnapshot",
    "SelectorFactory",
    "build_composite_selector",
    "build_liquidity_selector",
    "build_regime_selector",
    "build_selector",
]
