"""Composite timing overlay (``T_t``) — apply several overlays in order.

Strategy YAMLs declare timing overlays as an ordered list
(``tsmom -> kama -> carry_filter``); :class:`CompositeTiming` is the object that
actually threads a weight vector through that list. Order matters and is
preserved exactly as configured: TSMOM sets direction and vol scale, KAMA
confirms the trend, the carry filter removes negative-carry legs.

Because every overlay is a pure ``WeightVector -> WeightVector`` transformation,
composition is associative and each intermediate stage can be inspected by the
pipeline waterfall.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from configs.strategy import StageParam

from src.core.interfaces import TimingOverlay
from src.core.types import MarketState, WeightVector
from src.strategies._params import ParamReader


class CompositeTiming:
    """Sequentially apply a list of timing overlays."""

    __slots__ = ("_components",)

    def __init__(self, components: Sequence[TimingOverlay]) -> None:
        if not components:
            raise ValueError("CompositeTiming requires at least one overlay")
        for component in components:
            if not isinstance(component, TimingOverlay):
                raise TypeError(
                    "CompositeTiming components must satisfy the TimingOverlay protocol, "
                    f"got {type(component)!r}"
                )
        self._components = tuple(components)

    @property
    def name(self) -> str:
        """Registry name of this overlay."""
        return "composite"

    @property
    def components(self) -> tuple[TimingOverlay, ...]:
        """The composed overlays, in application order."""
        return self._components

    def adjust_timing(self, weights: WeightVector, state: MarketState) -> WeightVector:
        """Thread ``weights`` through every component in order."""
        adjusted = weights
        for component in self._components:
            adjusted = component.adjust_timing(adjusted, state)
        return adjusted


def build_composite_timing(params: Mapping[str, StageParam]) -> TimingOverlay:
    """Build a :class:`CompositeTiming` from ``components: "tsmom,kama"``.

    Components are built with their default parameters; callers needing
    per-component parameters should construct :class:`CompositeTiming` directly.
    """
    from src.strategies.timing import build_timing  # local import avoids a cycle

    reader = ParamReader("timing.composite", params)
    component_names = reader.as_sequence("components")
    reader.check()
    if not component_names:
        raise ValueError("composite timing requires a 'components' parameter")
    components: list[TimingOverlay] = []
    for component_name in component_names:
        if component_name.strip().lower() == "composite":
            raise ValueError("composite timing cannot contain a composite component")
        components.append(build_timing(component_name, {}))
    return CompositeTiming(components)


__all__ = ["CompositeTiming", "build_composite_timing"]
