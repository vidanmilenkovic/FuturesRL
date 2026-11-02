"""Carry / term-structure timing filter (``T_t``).

Futures carry is the return earned by holding a contract as it converges to spot:
a backwardated market (``carry_annualized > 0``) pays the long, a contangoed
market (``carry_annualized < 0``) pays the short. The FinRL-X carry filter uses
the ``carry_annualized`` feature (falling back to the ``term_slope`` feature
scaled by the roll horizon) to

* **zero** legs whose annualised carry is below ``min_carry_bps`` (default
  ``0`` bps, i.e. never hold a negative-carry long), or
* **scale** legs continuously with the carry surplus when ``mode="scale"``.

A state with neither feature is left neutral (factor ``1.0``): missing carry data
means "no view", not "no position". Factors are clipped to
``[0, max_scale]`` so a single leg can never be levered arbitrarily by a data
glitch.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from configs.strategy import StageParam

from src.core.interfaces import TimingOverlay
from src.core.types import InstrumentId, MarketState, WeightVector
from src.data.schema import CARRY_ANNUALIZED, TERM_SLOPE
from src.strategies._numeric import clipped
from src.strategies._params import ParamReader

#: Supported filter modes.
CARRY_FILTER_MODES: frozenset[str] = frozenset({"zero", "scale"})

#: Annualisation multiplier converting a front/next term slope to annual carry.
TERM_SLOPE_ANNUALIZATION: float = 365.0 / 30.0

#: One basis point as a decimal fraction.
BASIS_POINT: float = 1e-4


class CarryFilter:
    """Scale or zero weights on the annualised futures carry signal."""

    __slots__ = (
        "_carry_scale_bps",
        "_carry_weight",
        "_max_scale",
        "_min_carry_bps",
        "_mode",
    )

    def __init__(
        self,
        *,
        min_carry_bps: float = 0.0,
        carry_weight: float = 1.0,
        carry_scale_bps: float = 100.0,
        max_scale: float = 1.5,
        mode: str = "zero",
    ) -> None:
        if carry_weight < 0.0:
            raise ValueError("carry_weight must be non-negative")
        if carry_scale_bps <= 0.0:
            raise ValueError("carry_scale_bps must be strictly positive")
        if max_scale < 1.0:
            raise ValueError("max_scale must be at least 1.0")
        normalised_mode = mode.strip().lower()
        if normalised_mode not in CARRY_FILTER_MODES:
            raise ValueError(
                f"carry filter mode must be one of {sorted(CARRY_FILTER_MODES)}, got {mode!r}"
            )
        self._min_carry_bps = float(min_carry_bps)
        self._carry_weight = float(carry_weight)
        self._carry_scale_bps = float(carry_scale_bps)
        self._max_scale = float(max_scale)
        self._mode = normalised_mode

    @property
    def name(self) -> str:
        """Registry name of this overlay."""
        return "carry_filter"

    @property
    def mode(self) -> str:
        """``zero`` or ``scale``."""
        return self._mode

    def carry_bps(self, state: MarketState, instrument_id: InstrumentId) -> float | None:
        """Annualised carry in basis points, or ``None`` when unavailable."""
        snapshot = state.instruments.get(instrument_id)
        if snapshot is None:
            return None
        annualised = snapshot.feature(CARRY_ANNUALIZED, math.nan)
        if math.isfinite(annualised):
            return annualised / BASIS_POINT
        slope = snapshot.feature(TERM_SLOPE, math.nan)
        if math.isfinite(slope):
            return slope * TERM_SLOPE_ANNUALIZATION / BASIS_POINT
        return None

    def scaling_factor(self, state: MarketState, instrument_id: InstrumentId) -> float:
        """Carry multiplier applied to the base weight (``1.0`` when unknown)."""
        carry = self.carry_bps(state, instrument_id)
        if carry is None:
            return 1.0
        if carry < self._min_carry_bps and self._mode == "zero":
            return 0.0
        surplus = (carry - self._min_carry_bps) / self._carry_scale_bps
        factor = 1.0 + self._carry_weight * math.tanh(surplus)
        return clipped(factor, 0.0, self._max_scale)

    def adjust_timing(self, weights: WeightVector, state: MarketState) -> WeightVector:
        """Apply the carry filter to every base weight."""
        adjusted: dict[InstrumentId, float] = {}
        for instrument_id, weight in weights.items():
            adjusted[instrument_id] = weight * self.scaling_factor(state, instrument_id)
        return WeightVector(adjusted)


def build_carry_filter(params: Mapping[str, StageParam]) -> TimingOverlay:
    """Build a :class:`CarryFilter` overlay from declarative stage parameters."""
    reader = ParamReader("timing.carry_filter", params)
    overlay = CarryFilter(
        min_carry_bps=reader.as_float("min_carry_bps", 0.0),
        carry_weight=reader.as_float("carry_weight", 1.0),
        carry_scale_bps=reader.as_float("carry_scale_bps", 100.0),
        max_scale=reader.as_float("max_scale", 1.5),
        mode=reader.as_str("mode", "zero"),
    )
    reader.check()
    return overlay


__all__ = ["CARRY_FILTER_MODES", "CarryFilter", "build_carry_filter"]
