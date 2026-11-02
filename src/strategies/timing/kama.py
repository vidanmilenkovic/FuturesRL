"""Kaufman Adaptive Moving Average timing overlay (``T_t``).

KAMA is Perry Kaufman's adaptive trend filter: it speeds up when the price path
is efficient (a clean trend) and slows down in noise, which is exactly the
behaviour a futures trend overlay needs around choppy roll periods.

    ``ER_t  = |P_t - P_{t-n}| / sum_{i=t-n+1..t} |P_i - P_{i-1}|``   (efficiency ratio)
    ``SC_t  = (ER_t * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1))^2``    (smoothing constant)
    ``KAMA_t = KAMA_{t-1} + SC_t * (P_t - KAMA_{t-1})``

The price path is reconstructed from the point-in-time daily return series
(``cumprod(1 + r)``), so the overlay never needs an extra price feed and cannot
look ahead. The overlay then scales each base weight by its trend agreement:

    ``agreement = tanh((P_last - KAMA_last) / (KAMA_last * band))``

so a price persistently above its KAMA keeps the full weight, a price below it
flattens the leg (or flips it when ``allow_flip`` is set), and a price hovering
on the KAMA de-risks proportionally. States without a return series fall back to
the ``ret_126d`` feature sign, and a state with neither degrades to a neutral
factor of ``1.0`` rather than inventing a trend.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
from configs.strategy import StageParam
from numpy.typing import NDArray

from src.core.interfaces import TimingOverlay
from src.core.types import InstrumentId, MarketState, WeightVector
from src.data.schema import RET_126D
from src.strategies._numeric import clipped, finite_or, returns_series
from src.strategies._params import ParamReader

#: Default relative gap at which the trend agreement saturates to +/-1.
DEFAULT_TREND_BAND: float = 0.02

#: Fallback band used when only the ``ret_126d`` feature is available.
DEFAULT_FEATURE_BAND: float = 0.05


def kaufman_adaptive_moving_average(
    series: NDArray[np.float64],
    *,
    window: int = 10,
    fast_period: int = 2,
    slow_period: int = 30,
) -> NDArray[np.float64]:
    """Compute the KAMA of ``series`` (deterministic, O(n)).

    The first ``window`` points are seeded with the raw series (there is not
    enough history to estimate an efficiency ratio yet); from there the standard
    recursion is applied. A zero-length input returns a zero-length array.
    """
    if window < 1:
        raise ValueError("window must be positive")
    if fast_period < 1 or slow_period < 1:
        raise ValueError("fast_period and slow_period must be positive")
    if fast_period >= slow_period:
        raise ValueError("fast_period must be smaller than slow_period")
    size = int(series.size)
    if size == 0:
        return np.zeros(0, dtype=np.float64)
    fast_smoothing = 2.0 / (fast_period + 1.0)
    slow_smoothing = 2.0 / (slow_period + 1.0)
    kama = np.empty(size, dtype=np.float64)
    seed = min(window, size - 1)
    kama[: seed + 1] = series[: seed + 1]
    for index in range(seed + 1, size):
        change = abs(float(series[index] - series[index - window]))
        volatility = float(np.sum(np.abs(np.diff(series[index - window : index + 1]))))
        efficiency_ratio = change / volatility if volatility > 1e-15 else 0.0
        efficiency_ratio = min(max(efficiency_ratio, 0.0), 1.0)
        smoothing = (efficiency_ratio * (fast_smoothing - slow_smoothing) + slow_smoothing) ** 2
        kama[index] = kama[index - 1] + smoothing * (float(series[index]) - kama[index - 1])
    return kama


class KAMATiming:
    """Scale weights by agreement between the price and its KAMA."""

    __slots__ = (
        "_allow_flip",
        "_fast_period",
        "_feature_band",
        "_slow_period",
        "_trend_band",
        "_window",
    )

    def __init__(
        self,
        *,
        window: int = 10,
        fast_period: int = 2,
        slow_period: int = 30,
        trend_band: float = DEFAULT_TREND_BAND,
        feature_band: float = DEFAULT_FEATURE_BAND,
        allow_flip: bool = False,
    ) -> None:
        if window < 1:
            raise ValueError("window must be positive")
        if fast_period < 1 or slow_period < 1:
            raise ValueError("fast_period and slow_period must be positive")
        if fast_period >= slow_period:
            raise ValueError("fast_period must be smaller than slow_period")
        if trend_band <= 0.0:
            raise ValueError("trend_band must be strictly positive")
        if feature_band <= 0.0:
            raise ValueError("feature_band must be strictly positive")
        self._window = int(window)
        self._fast_period = int(fast_period)
        self._slow_period = int(slow_period)
        self._trend_band = float(trend_band)
        self._feature_band = float(feature_band)
        self._allow_flip = bool(allow_flip)

    @property
    def name(self) -> str:
        """Registry name of this overlay."""
        return "kama"

    def _agreement_from_history(
        self, state: MarketState, instrument_id: InstrumentId
    ) -> float | None:
        returns = returns_series(state, instrument_id)
        if returns is None or returns.size < self._window + 2:
            return None
        prices = np.cumprod(1.0 + np.clip(returns, -0.99, 10.0))
        kama = kaufman_adaptive_moving_average(
            prices,
            window=self._window,
            fast_period=self._fast_period,
            slow_period=self._slow_period,
        )
        reference = float(kama[-1])
        if not math.isfinite(reference) or abs(reference) <= 1e-12:
            return None
        gap = (float(prices[-1]) - reference) / abs(reference)
        return math.tanh(gap / self._trend_band)

    def _agreement_from_feature(
        self, state: MarketState, instrument_id: InstrumentId
    ) -> float | None:
        trailing = state.feature(instrument_id, RET_126D, math.nan)
        if not math.isfinite(trailing):
            return None
        return math.tanh(finite_or(trailing, 0.0) / self._feature_band)

    def agreement(self, state: MarketState, instrument_id: InstrumentId) -> float:
        """Trend agreement in ``[-1, 1]`` (0 when no history is available)."""
        from_history = self._agreement_from_history(state, instrument_id)
        if from_history is not None:
            return clipped(from_history, -1.0, 1.0)
        from_feature = self._agreement_from_feature(state, instrument_id)
        if from_feature is not None:
            return clipped(from_feature, -1.0, 1.0)
        return 0.0

    def scaling_factor(self, state: MarketState, instrument_id: InstrumentId) -> float:
        """Trend-agreement multiplier applied to the base weight."""
        agreement = self.agreement(state, instrument_id)
        return agreement if self._allow_flip else max(0.0, agreement)

    def adjust_timing(self, weights: WeightVector, state: MarketState) -> WeightVector:
        """Scale every base weight by its KAMA trend agreement."""
        adjusted: dict[InstrumentId, float] = {}
        for instrument_id, weight in weights.items():
            adjusted[instrument_id] = weight * self.scaling_factor(state, instrument_id)
        return WeightVector(adjusted)


def build_kama_timing(params: Mapping[str, StageParam]) -> TimingOverlay:
    """Build a :class:`KAMATiming` overlay from declarative stage parameters."""
    reader = ParamReader("timing.kama", params)
    overlay = KAMATiming(
        window=reader.as_int("window", 10),
        fast_period=reader.as_int("fast_period", 2),
        slow_period=reader.as_int("slow_period", 30),
        trend_band=reader.as_float("trend_band", DEFAULT_TREND_BAND),
        feature_band=reader.as_float("feature_band", DEFAULT_FEATURE_BAND),
        allow_flip=reader.as_bool("allow_flip", False),
    )
    reader.check()
    return overlay


__all__ = [
    "DEFAULT_FEATURE_BAND",
    "DEFAULT_TREND_BAND",
    "KAMATiming",
    "build_kama_timing",
    "kaufman_adaptive_moving_average",
]
