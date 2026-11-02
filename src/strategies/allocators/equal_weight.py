"""Equal-weight allocator (``A_t``) — the naive diversification baseline.

The point of this allocator is to be boring: it is the benchmark every smarter
allocator (risk parity, mean-variance, DRL) has to beat, and it must be exactly
reproducible. Weights are ``gross / n`` per candidate, optionally capped by a
per-instrument limit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from configs.strategy import StageParam

from src.core.interfaces import PortfolioAllocator
from src.core.types import InstrumentId, MarketState, WeightVector
from src.strategies._params import ParamReader


class EqualWeightAllocator:
    """Allocate an equal signed notional weight to every candidate."""

    __slots__ = ("_gross_leverage", "_max_weight")

    def __init__(
        self,
        *,
        gross_leverage: float = 1.0,
        max_weight: float | None = None,
    ) -> None:
        if not gross_leverage >= 0.0:
            raise ValueError("gross_leverage must be non-negative")
        if max_weight is not None and max_weight < 0.0:
            raise ValueError("max_weight must be non-negative")
        self._gross_leverage = float(gross_leverage)
        self._max_weight = max_weight

    @property
    def name(self) -> str:
        """Registry name of this allocator."""
        return "equal_weight"

    def allocate(self, candidates: Sequence[InstrumentId], state: MarketState) -> WeightVector:
        """Return ``gross / n`` for every candidate (empty for an empty universe)."""
        del state  # equal weighting is state independent by construction
        if not candidates:
            return WeightVector.empty()
        per_instrument = self._gross_leverage / len(candidates)
        if self._max_weight is not None:
            per_instrument = min(per_instrument, self._max_weight)
        return WeightVector(dict.fromkeys(candidates, per_instrument))


def build_equal_weight_allocator(params: Mapping[str, StageParam]) -> PortfolioAllocator:
    """Build an :class:`EqualWeightAllocator` from declarative stage parameters."""
    reader = ParamReader("allocator.equal_weight", params)
    allocator = EqualWeightAllocator(
        gross_leverage=reader.as_float("gross_leverage", 1.0),
        max_weight=reader.as_optional_float("max_weight"),
    )
    reader.check()
    return allocator


__all__ = ["EqualWeightAllocator", "build_equal_weight_allocator"]
