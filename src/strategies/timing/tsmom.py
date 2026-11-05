"""Time-series momentum timing overlay (``T_t``) — vol-scaled sign scaling.

The FinRL-X timing stage is a *weight transformation*, not an allocator: it takes
the allocator's base notional weights and applies a directional, volatility
scaled TSMOM factor to each leg:

    ``w_i^timing = w_i^base * sign_i * clip(target_vol / vol_i, min_scale, max_scale)``

where ``sign_i`` is ``+1`` when the fast and slow lookback returns agree bullish,
``-1`` when they agree bearish and ``0`` when they disagree (a disagreing
fast/slow pair is a whipsaw, and the correct position size is zero). The
``target_vol / vol`` scaling is the classic Moskowitz-Ooi-Pedersen TSMOM sizing:
the same dollar risk budget is deployed in every root regardless of its vol.

The volatility used for the ``target_vol / vol`` sizing is annualised with
``bars_per_year(state.timeframe)`` (:func:`~src.strategies._numeric.annualized_volatility`),
so a 5m state deploys the same annual risk budget as a daily one; ``"1d"`` keeps
the exact historical ``sqrt(252)`` behaviour.

The overlay only ever *scales* existing legs — it never introduces an instrument
the allocator did not select — so ``gross`` can shrink to zero but the universe
is preserved.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from configs.strategy import StageParam

from src.core.interfaces import TimingOverlay
from src.core.types import InstrumentId, MarketState, WeightVector
from src.data.schema import RET_21D, RET_126D, TSMOM_21, TSMOM_126
from src.strategies._numeric import (
    annualized_volatility,
    clipped,
    finite_or,
    trailing_sum,
)
from src.strategies._params import ParamReader

#: Feature fallbacks for the fast lookback return, in preference order.
FAST_RETURN_FEATURES: tuple[str, ...] = (TSMOM_21, RET_21D)

#: Feature fallbacks for the slow lookback return, in preference order.
SLOW_RETURN_FEATURES: tuple[str, ...] = (TSMOM_126, RET_126D)


def _feature_return(
    state: MarketState, instrument_id: InstrumentId, names: Sequence[str]
) -> float | None:
    for feature_name in names:
        value = state.feature(instrument_id, feature_name, math.nan)
        if math.isfinite(value):
            return float(value)
    return None


def _sign(value: float) -> float:
    if value > 0.0:
        return 1.0
    if value < 0.0:
        return -1.0
    return 0.0


class TSMOMTiming:
    """Volatility-scaled time-series-momentum weight scaling."""

    __slots__ = (
        "_allow_flip",
        "_lookback_fast",
        "_lookback_slow",
        "_max_scale",
        "_min_scale",
        "_require_slow_agreement",
        "_target_volatility",
        "_vol_lookback",
    )

    def __init__(
        self,
        *,
        lookback_fast: int = 21,
        lookback_slow: int = 126,
        vol_lookback: int = 20,
        target_volatility: float | None = None,
        min_scale: float = 0.0,
        max_scale: float = 3.0,
        require_slow_agreement: bool = True,
        allow_flip: bool = True,
    ) -> None:
        if lookback_fast < 1 or lookback_slow < 1:
            raise ValueError("lookback windows must be positive")
        if lookback_fast >= lookback_slow:
            raise ValueError("lookback_fast must be smaller than lookback_slow")
        if vol_lookback < 2:
            raise ValueError("vol_lookback must be at least 2")
        if target_volatility is not None and target_volatility <= 0.0:
            raise ValueError("target_volatility must be strictly positive")
        if min_scale < 0.0:
            raise ValueError("min_scale must be non-negative")
        if max_scale < min_scale:
            raise ValueError("max_scale must not be smaller than min_scale")
        self._lookback_fast = int(lookback_fast)
        self._lookback_slow = int(lookback_slow)
        self._vol_lookback = int(vol_lookback)
        self._target_volatility = target_volatility
        self._min_scale = float(min_scale)
        self._max_scale = float(max_scale)
        self._require_slow_agreement = bool(require_slow_agreement)
        self._allow_flip = bool(allow_flip)

    @property
    def name(self) -> str:
        """Registry name of this overlay."""
        return "tsmom"

    def momentum(self, state: MarketState, instrument_id: InstrumentId) -> tuple[float, float]:
        """Fast and slow lookback returns for one instrument (never ``NaN``)."""
        fast = trailing_sum(state, instrument_id, self._lookback_fast)
        if fast is None:
            fast = _feature_return(state, instrument_id, FAST_RETURN_FEATURES)
        slow = trailing_sum(state, instrument_id, self._lookback_slow)
        if slow is None:
            slow = _feature_return(state, instrument_id, SLOW_RETURN_FEATURES)
        if fast is None and slow is None:
            return 0.0, 0.0
        if fast is None:
            fast = slow if slow is not None else 0.0
        if slow is None:
            slow = fast
        return finite_or(fast, 0.0), finite_or(slow, 0.0)

    def scaling_factor(self, state: MarketState, instrument_id: InstrumentId) -> float:
        """Directional, volatility-scaled multiplier for one instrument."""
        fast, slow = self.momentum(state, instrument_id)
        direction = _sign(fast)
        if self._require_slow_agreement and direction != 0.0 and _sign(slow) != direction:
            direction = 0.0
        if direction == 0.0:
            return 0.0
        scale = 1.0
        if self._target_volatility is not None:
            volatility = annualized_volatility(state, instrument_id, lookback=self._vol_lookback)
            if volatility is not None and volatility > 0.0:
                scale = clipped(
                    self._target_volatility / volatility, self._min_scale, self._max_scale
                )
        factor = direction * scale
        return factor if self._allow_flip else max(0.0, factor)

    def adjust_timing(self, weights: WeightVector, state: MarketState) -> WeightVector:
        """Scale every base weight by its TSMOM factor."""
        adjusted: dict[InstrumentId, float] = {}
        for instrument_id, weight in weights.items():
            adjusted[instrument_id] = weight * self.scaling_factor(state, instrument_id)
        return WeightVector(adjusted)


def build_tsmom_timing(params: Mapping[str, StageParam]) -> TimingOverlay:
    """Build a :class:`TSMOMTiming` overlay from declarative stage parameters."""
    reader = ParamReader("timing.tsmom", params)
    overlay = TSMOMTiming(
        lookback_fast=reader.as_int("lookback_fast", 21),
        lookback_slow=reader.as_int("lookback_slow", 126),
        vol_lookback=reader.as_int("vol_lookback", 20),
        target_volatility=reader.as_optional_float("target_volatility"),
        min_scale=reader.as_float("min_scale", 0.0),
        max_scale=reader.as_float("max_scale", 3.0),
        require_slow_agreement=reader.as_bool("require_slow_agreement", True),
        allow_flip=reader.as_bool("allow_flip", True),
    )
    reader.check()
    return overlay


__all__ = ["FAST_RETURN_FEATURES", "SLOW_RETURN_FEATURES", "TSMOMTiming", "build_tsmom_timing"]
