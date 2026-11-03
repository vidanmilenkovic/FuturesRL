"""Mean-reversion timing overlay (``T_t``) — fade extremes without look-ahead.

The FinRL-X timing stage usually follows trends (TSMOM/KAMA). This overlay is the
contrarian complement used by the *Panic Mean-Reversion* strategy template: it
fades statistically stretched moves, but only while the cross-sectional
turbulence index says the market is not in a structural break.

Signals (all computed from strictly point-in-time data)
-------------------------------------------------------
* **Oversold (buy-side reversion, ``signal = +1``)**: Wilder RSI below
  ``oversold_threshold`` **or** the mark below the lower Bollinger band
  (``SMA(window) - bollinger_std * stdev(window)``).
* **Overbought (sell-side reversion, ``signal = -1``)**: RSI above
  ``overbought_threshold`` **or** the mark above the upper Bollinger band.
* **Neutral (``signal = 0``)**: no reversion action.

The price path and the RSI/Bollinger inputs are reconstructed from the
instrument's trailing daily-return series (``cumprod(1 + r)``) when the data layer
provides one (it always does for MarketStates built by
:class:`~src.data.market_state_builder.MarketStateBuilder`); otherwise the
canonical ``rsi`` / ``bb_lower`` / ``bb_upper`` features and the snapshot mark are
used, and if neither is available the instrument is left untouched. Only the
current and past bars are used — the overlay cannot peek ahead.

Turbulence veto
---------------
``turbulence_veto_threshold`` (default ``0.95``) is the Mahalanobis turbulence
index above which the overlay freezes: fading a genuine structural break is how
mean-reversion books blow up, so when ``turbulence >= threshold`` the base
allocation passes through unchanged (no boost and no fade). A missing turbulence
feature is treated as benign (``0.0``) so a data-starved state degrades to normal
behaviour rather than a silent global veto.

Weight transformation
---------------------
Reversion only ever scales legs that the allocator already selected (the
``TimingOverlay`` contract cannot introduce an instrument):

``w_i^timing = w_i^base * (1 + reversion_strength)`` when the reversion signal
agrees with the leg direction (long leg + oversold, or short leg + overbought),

``w_i^timing = w_i^base * (1 - reversion_strength)`` when it fights the leg
direction (reduce the stale exposure),

and ``w_i^timing = w_i^base`` for neutral or turbulence-vetoed instruments.
``reversion_strength=0.5`` therefore means "+50% on the reversion side, -50% on
the stale side", and the factor is clamped into ``[0, 1 + reversion_strength]``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
from configs.strategy import StageParam
from numpy.typing import NDArray

from src.core.interfaces import TimingOverlay
from src.core.types import InstrumentId, MarketState, WeightVector
from src.data.schema import BB_LOWER, BB_UPPER, RSI, TURBULENCE
from src.strategies._numeric import clipped, finite_or, returns_series
from src.strategies._params import ParamReader

#: Default RSI lookback (Wilder's smoothing period).
DEFAULT_RSI_PERIOD: int = 14
#: Default RSI threshold below which a leg is considered oversold.
DEFAULT_OVERSOLD_THRESHOLD: float = 25.0
#: Default RSI threshold above which a leg is considered overbought.
DEFAULT_OVERBOUGHT_THRESHOLD: float = 75.0
#: Default Bollinger lookback window (bars).
DEFAULT_BOLLINGER_WINDOW: int = 20
#: Default Bollinger band width in standard deviations.
DEFAULT_BOLLINGER_STD: float = 2.0
#: Default Mahalanobis turbulence index above which reversion is vetoed.
DEFAULT_TURBULENCE_VETO_THRESHOLD: float = 0.95
#: Default reversion scaling strength (50% boost / 50% fade).
DEFAULT_REVERSION_STRENGTH: float = 0.5


def wilder_rsi(prices: NDArray[np.float64], period: int = DEFAULT_RSI_PERIOD) -> float | None:
    """Wilder-smoothed RSI of ``prices`` (``None`` when there is not enough data).

    Uses only the supplied (trailing) price path: average gains/losses are seeded
    with the first ``period`` moves and then smoothed with
    ``avg = (avg * (period - 1) + current) / period``. A flat/no-loss path returns
    ``100.0`` (all gains) or ``50.0`` (no moves at all); the function never returns
    ``NaN`` so callers can compare freely.
    """
    if period < 2:
        raise ValueError("period must be at least 2")
    values = np.asarray(prices, dtype=np.float64)
    if values.size < period + 1:
        return None
    deltas = np.diff(values)
    if deltas.size < period:
        return None
    gains = np.clip(deltas, 0.0, None)
    losses = np.clip(-deltas, 0.0, None)
    average_gain = float(np.mean(gains[:period]))
    average_loss = float(np.mean(losses[:period]))
    for index in range(period, deltas.size):
        average_gain = (average_gain * (period - 1) + float(gains[index])) / period
        average_loss = (average_loss * (period - 1) + float(losses[index])) / period
    if average_loss <= 1e-12:
        return 100.0 if average_gain > 1e-12 else 50.0
    relative_strength = average_gain / average_loss
    return 100.0 - 100.0 / (1.0 + relative_strength)


def bollinger_bands(
    prices: NDArray[np.float64], window: int, num_std: float
) -> tuple[float, float, float] | None:
    """``(last_price, lower_band, upper_band)`` from the trailing ``window`` bars."""
    if window < 2:
        raise ValueError("window must be at least 2")
    values = np.asarray(prices, dtype=np.float64)
    if values.size < window:
        return None
    sample = values[-window:]
    middle = float(np.mean(sample))
    deviation = float(np.std(sample, ddof=0))
    return float(sample[-1]), middle - num_std * deviation, middle + num_std * deviation


class MeanReversionTiming:
    """Fade RSI/Bollinger extremes, with a turbulence break-out veto."""

    __slots__ = (
        "_bollinger_std",
        "_bollinger_window",
        "_overbought_threshold",
        "_oversold_threshold",
        "_reversion_strength",
        "_rsi_period",
        "_turbulence_veto_threshold",
    )

    def __init__(
        self,
        *,
        rsi_period: int = DEFAULT_RSI_PERIOD,
        oversold_threshold: float = DEFAULT_OVERSOLD_THRESHOLD,
        overbought_threshold: float = DEFAULT_OVERBOUGHT_THRESHOLD,
        bollinger_window: int = DEFAULT_BOLLINGER_WINDOW,
        bollinger_std: float = DEFAULT_BOLLINGER_STD,
        turbulence_veto_threshold: float = DEFAULT_TURBULENCE_VETO_THRESHOLD,
        reversion_strength: float = DEFAULT_REVERSION_STRENGTH,
    ) -> None:
        if rsi_period < 2:
            raise ValueError("rsi_period must be at least 2")
        if bollinger_window < 2:
            raise ValueError("bollinger_window must be at least 2")
        if not 0.0 <= oversold_threshold < 50.0:
            raise ValueError("oversold_threshold must be within [0, 50)")
        if not 50.0 < overbought_threshold <= 100.0:
            raise ValueError("overbought_threshold must be within (50, 100]")
        if oversold_threshold >= overbought_threshold:
            raise ValueError("oversold_threshold must be smaller than overbought_threshold")
        if not math.isfinite(bollinger_std) or bollinger_std <= 0.0:
            raise ValueError("bollinger_std must be a finite strictly positive number")
        if not math.isfinite(turbulence_veto_threshold) or turbulence_veto_threshold <= 0.0:
            raise ValueError("turbulence_veto_threshold must be a finite strictly positive number")
        if not 0.0 <= reversion_strength <= 1.0:
            raise ValueError("reversion_strength must be within [0, 1]")
        self._rsi_period = int(rsi_period)
        self._oversold_threshold = float(oversold_threshold)
        self._overbought_threshold = float(overbought_threshold)
        self._bollinger_window = int(bollinger_window)
        self._bollinger_std = float(bollinger_std)
        self._turbulence_veto_threshold = float(turbulence_veto_threshold)
        self._reversion_strength = float(reversion_strength)

    @property
    def name(self) -> str:
        """Registry name of this overlay."""
        return "mean_reversion"

    @property
    def reversion_strength(self) -> float:
        """Configured reversion strength (``0.5`` → +/-50% scaling)."""
        return self._reversion_strength

    # -- signal inputs --------------------------------------------------------

    def rsi(self, state: MarketState, instrument_id: InstrumentId) -> float | None:
        """Point-in-time RSI for ``instrument_id`` (``None`` when unavailable)."""
        path = self._price_path(state, instrument_id)
        if path is not None:
            value = wilder_rsi(path, self._rsi_period)
            if value is not None:
                return value
        feature = state.feature(instrument_id, RSI, math.nan)
        if math.isfinite(feature):
            return float(feature)
        return None

    def bands(
        self, state: MarketState, instrument_id: InstrumentId
    ) -> tuple[float, float, float] | None:
        """``(price, lower, upper)`` Bollinger bands (``None`` when unavailable)."""
        path = self._price_path(state, instrument_id)
        if path is not None:
            computed = bollinger_bands(path, self._bollinger_window, self._bollinger_std)
            if computed is not None:
                return computed
        snapshot = state.instruments.get(instrument_id)
        if snapshot is None:
            return None
        lower = snapshot.feature(BB_LOWER, math.nan)
        upper = snapshot.feature(BB_UPPER, math.nan)
        if not (math.isfinite(lower) and math.isfinite(upper)):
            return None
        return snapshot.price, float(lower), float(upper)

    def turbulence(self, state: MarketState, instrument_id: InstrumentId) -> float:
        """Mahalanobis turbulence index (``0.0`` when the feature is missing)."""
        value = state.feature(instrument_id, TURBULENCE, math.nan)
        return finite_or(value, 0.0)

    def _price_path(
        self, state: MarketState, instrument_id: InstrumentId
    ) -> NDArray[np.float64] | None:
        series = returns_series(state, instrument_id)
        if series is None:
            return None
        minimum = max(self._rsi_period + 1, self._bollinger_window)
        if series.size < minimum:
            return None
        return np.cumprod(1.0 + np.clip(series, -0.99, 10.0))

    # -- signal / scaling -----------------------------------------------------

    def signal(self, state: MarketState, instrument_id: InstrumentId) -> int:
        """``+1`` oversold, ``-1`` overbought, ``0`` neutral or turbulence-vetoed."""
        if self.turbulence(state, instrument_id) >= self._turbulence_veto_threshold:
            return 0
        rsi = self.rsi(state, instrument_id)
        bands = self.bands(state, instrument_id)
        oversold = rsi is not None and rsi < self._oversold_threshold
        overbought = rsi is not None and rsi > self._overbought_threshold
        if bands is not None:
            price, lower, upper = bands
            oversold = oversold or price < lower
            overbought = overbought or price > upper
        if oversold and not overbought:
            return 1
        if overbought and not oversold:
            return -1
        return 0

    def scaling_factor(
        self, state: MarketState, instrument_id: InstrumentId, weight: float
    ) -> float:
        """Multiplier applied to ``weight`` for the current reversion signal."""
        signal = self.signal(state, instrument_id)
        if signal == 0 or weight == 0.0:
            return 1.0
        agrees = (signal > 0 and weight > 0.0) or (signal < 0 and weight < 0.0)
        factor = 1.0 + self._reversion_strength if agrees else 1.0 - self._reversion_strength
        upper = 1.0 + self._reversion_strength
        return clipped(factor, 0.0, upper)

    def adjust_timing(self, weights: WeightVector, state: MarketState) -> WeightVector:
        """Apply the mean-reversion scaling to every selected leg."""
        adjusted: dict[InstrumentId, float] = {}
        for instrument_id, weight in weights.items():
            adjusted[instrument_id] = weight * self.scaling_factor(state, instrument_id, weight)
        return WeightVector(adjusted)


def build_mean_reversion_timing(params: Mapping[str, StageParam]) -> TimingOverlay:
    """Build a :class:`MeanReversionTiming` overlay from declarative stage parameters."""
    reader = ParamReader("timing.mean_reversion", params)
    overlay = MeanReversionTiming(
        rsi_period=reader.as_int("rsi_period", DEFAULT_RSI_PERIOD),
        oversold_threshold=reader.as_float("oversold_threshold", DEFAULT_OVERSOLD_THRESHOLD),
        overbought_threshold=reader.as_float("overbought_threshold", DEFAULT_OVERBOUGHT_THRESHOLD),
        bollinger_window=reader.as_int("bollinger_window", DEFAULT_BOLLINGER_WINDOW),
        bollinger_std=reader.as_float("bollinger_std", DEFAULT_BOLLINGER_STD),
        turbulence_veto_threshold=reader.as_float(
            "turbulence_veto_threshold", DEFAULT_TURBULENCE_VETO_THRESHOLD
        ),
        reversion_strength=reader.as_float("reversion_strength", DEFAULT_REVERSION_STRENGTH),
    )
    reader.check()
    return overlay


__all__ = [
    "DEFAULT_BOLLINGER_STD",
    "DEFAULT_BOLLINGER_WINDOW",
    "DEFAULT_OVERBOUGHT_THRESHOLD",
    "DEFAULT_OVERSOLD_THRESHOLD",
    "DEFAULT_REVERSION_STRENGTH",
    "DEFAULT_RSI_PERIOD",
    "DEFAULT_TURBULENCE_VETO_THRESHOLD",
    "MeanReversionTiming",
    "bollinger_bands",
    "build_mean_reversion_timing",
    "wilder_rsi",
]
