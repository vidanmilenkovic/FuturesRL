"""Composite universe selector (``S_t``) — intersection / union of selectors.

Real desks rarely trust a single universe filter: a liquidity screen keeps the
book tradable, a regime screen keeps it on the right side of the macro cycle.
:class:`CompositeSelector` composes the two deterministically:

* ``intersection``: the roots every component accepts (a conservative AND);
* ``union``: every root at least one component accepts (a permissive OR).

Order is *always* explicit and deterministic: either the caller-supplied
``order`` (filtered to the selected set) or the concatenation of the component
outputs in component order. Nothing here iterates a Python ``set``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from configs.instruments import InstrumentsConfig
from configs.strategy import StageParam

from src.core.interfaces import UniverseSelector
from src.core.types import InstrumentId, MarketState
from src.strategies._params import ParamReader

#: Supported set operations.
COMPOSITE_MODES: frozenset[str] = frozenset({"intersection", "union"})


class CompositeSelector:
    """Combine several :class:`UniverseSelector` components."""

    __slots__ = ("_components", "_mode", "_order")

    def __init__(
        self,
        components: Sequence[UniverseSelector],
        *,
        mode: str = "intersection",
        order: Sequence[InstrumentId] | None = None,
    ) -> None:
        if not components:
            raise ValueError("CompositeSelector requires at least one component selector")
        normalised_mode = mode.strip().lower()
        if normalised_mode not in COMPOSITE_MODES:
            raise ValueError(
                f"composite mode must be one of {sorted(COMPOSITE_MODES)}, got {mode!r}"
            )
        for component in components:
            if not isinstance(component, UniverseSelector):
                raise TypeError(
                    "CompositeSelector components must satisfy the UniverseSelector protocol, "
                    f"got {type(component)!r}"
                )
        self._components = tuple(components)
        self._mode = normalised_mode
        self._order = tuple(order) if order is not None else None

    @property
    def name(self) -> str:
        """Registry name of this selector."""
        return "composite"

    @property
    def mode(self) -> str:
        """``intersection`` or ``union``."""
        return self._mode

    @property
    def components(self) -> tuple[UniverseSelector, ...]:
        """The composed selectors, in application order."""
        return self._components

    def select(self, state: MarketState) -> list[InstrumentId]:
        """Return the composed universe, preserving deterministic ordering."""
        outputs = [list(component.select(state)) for component in self._components]
        ordered_base: list[InstrumentId] = []
        seen: set[InstrumentId] = set()
        for output in outputs:
            for instrument_id in output:
                if instrument_id not in seen:
                    seen.add(instrument_id)
                    ordered_base.append(instrument_id)
        if self._mode == "union":
            selected = ordered_base
        else:
            members = set(outputs[0])
            for output in outputs[1:]:
                members &= set(output)
            selected = [instrument_id for instrument_id in ordered_base if instrument_id in members]
        if self._order is None:
            return selected
        explicit = set(selected)
        return [instrument_id for instrument_id in self._order if instrument_id in explicit]


def build_composite_selector(
    params: Mapping[str, StageParam],
    instruments: InstrumentsConfig | None = None,
) -> UniverseSelector:
    """Build a :class:`CompositeSelector` from declarative stage parameters.

    YAML cannot express nested stage objects, so components are named by their
    registry key (``components: "liquidity,regime"``) and built with their
    default parameters. Programmatic callers that need per-component parameters
    should construct :class:`CompositeSelector` directly.
    """
    from src.strategies.selectors import build_selector  # local import avoids a cycle

    reader = ParamReader("selector.composite", params)
    component_names = reader.as_sequence("components")
    mode = reader.as_str("mode", "intersection")
    order = reader.as_sequence("order")
    reader.check()
    if not component_names:
        raise ValueError("composite selector requires a 'components' parameter")
    components: list[UniverseSelector] = []
    for component_name in component_names:
        if component_name.strip().lower() == "composite":
            raise ValueError("composite selector cannot contain a composite component")
        components.append(build_selector(component_name, {}, instruments))
    return CompositeSelector(
        components,
        mode=mode,
        order=order or None,
    )


__all__ = ["COMPOSITE_MODES", "CompositeSelector", "build_composite_selector"]
