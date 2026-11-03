"""Adaptive-regime universe selector (``S_t``) — slow trend + fast risk-off.

Implements the FinRL-X *Adaptive Regime Rotation* universe filter as two
independent detectors whose disjunction de-risks the book:

* **Slow regime** (26 weeks): cross-sectional breadth of the 126-day trend plus
  a macro-volatility gauge. When most roots trend down *and* macro volatility is
  elevated, the regime is risk-off.
* **Fast risk-off** (3 days): a short-horizon shock detector that standardises
  the cross-sectional 3-day move by the universe's daily volatility. A ``-z``
  shock (default ``3`` standard deviations) or a raw 3-day drawdown beyond the
  configured threshold triggers risk-off immediately.

In risk-off the selector excludes configured risk-off roots (by root symbol or
by instrument-master *sector*) and optionally filters out roots whose trend is
negative, but it never returns an empty universe: if every root is falling it
de-risks to the least-negative ``min_universe`` roots. The allocator then
decides how much (possibly zero) risk to take — an empty universe would silently
freeze the whole pipeline.

Everything is computed from point-in-time state only (return series, features,
covariance and the ``regime``/``metadata`` flags); no future data is touched.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from configs.instruments import InstrumentsConfig
from configs.strategy import StageParam

from src.core.interfaces import UniverseSelector
from src.core.types import InstrumentId, MarketState
from src.data.schema import RET_126D
from src.strategies._numeric import (
    ANNUALIZATION_FACTOR,
    DEFAULT_ANNUAL_VOLATILITY,
    annualized_volatility,
    finite_or,
    trailing_sum,
)
from src.strategies._params import ParamReader
from src.strategies.selectors._roots import (
    normalise_requested_roots,
    validate_requested_roots,
)

#: ``state.regime`` values that force a risk-off universe regardless of metrics.
RISK_OFF_REGIME_LABELS: frozenset[str] = frozenset(
    {"risk_off", "risk-off", "stress", "crisis", "panic", "high_vol"}
)

#: ``state.metadata`` flags that force a risk-off universe.
RISK_OFF_METADATA_FLAGS: tuple[str, ...] = ("macro_lockout", "risk_off", "circuit_breaker")


@dataclass(frozen=True, slots=True)
class RegimeSnapshot:
    """Point-in-time regime diagnosis (useful for telemetry / tests)."""

    risk_off: bool
    slow_risk_off: bool
    fast_risk_off: bool
    forced_risk_off: bool
    breadth_negative: float
    macro_volatility: float
    mean_trend: float
    mean_shock: float
    shock_zscore: float
    trends: Mapping[InstrumentId, float]
    shocks: Mapping[InstrumentId, float]


class RegimeSelector:
    """Filter the universe on a slow trend/macro-vol regime and a fast shock."""

    __slots__ = (
        "_breadth_threshold",
        "_filter_negative_trend",
        "_instrument_master",
        "_macro_vol_threshold",
        "_min_universe",
        "_require_both",
        "_risk_off_roots",
        "_risk_off_sectors",
        "_roots",
        "_shock_lookback_days",
        "_shock_return_threshold",
        "_shock_z_threshold",
        "_top_n",
        "_trend_lookback_days",
        "_vol_lookback_days",
    )

    def __init__(
        self,
        *,
        trend_lookback_days: int = 126,
        vol_lookback_days: int = 60,
        shock_lookback_days: int = 3,
        shock_z_threshold: float = 3.0,
        shock_return_threshold: float = 0.02,
        macro_vol_threshold: float = 0.22,
        breadth_threshold: float = 0.60,
        risk_off_roots: Sequence[InstrumentId] = (),
        risk_off_sectors: Sequence[str] = (),
        filter_negative_trend: bool = True,
        min_universe: int = 1,
        top_n: int | None = None,
        require_both: bool = False,
        roots: Sequence[str] | str = (),
        instrument_master: InstrumentsConfig | None = None,
    ) -> None:
        if trend_lookback_days < 2:
            raise ValueError("trend_lookback_days must be at least 2")
        if shock_lookback_days < 1:
            raise ValueError("shock_lookback_days must be positive")
        if vol_lookback_days < 3:
            raise ValueError("vol_lookback_days must be at least 3")
        if shock_z_threshold <= 0.0:
            raise ValueError("shock_z_threshold must be strictly positive")
        if shock_return_threshold < 0.0:
            raise ValueError("shock_return_threshold must be non-negative")
        if macro_vol_threshold <= 0.0:
            raise ValueError("macro_vol_threshold must be strictly positive")
        if not 0.0 <= breadth_threshold <= 1.0:
            raise ValueError("breadth_threshold must be within [0, 1]")
        if min_universe < 1:
            raise ValueError("min_universe must be at least 1")
        if top_n is not None and top_n < 1:
            raise ValueError("top_n must be a positive integer or None")
        self._trend_lookback_days = int(trend_lookback_days)
        self._vol_lookback_days = int(vol_lookback_days)
        self._shock_lookback_days = int(shock_lookback_days)
        self._shock_z_threshold = float(shock_z_threshold)
        self._shock_return_threshold = float(shock_return_threshold)
        self._macro_vol_threshold = float(macro_vol_threshold)
        self._breadth_threshold = float(breadth_threshold)
        self._risk_off_roots = tuple(str(root).strip().upper() for root in risk_off_roots)
        self._risk_off_sectors = tuple(str(sector).strip().lower() for sector in risk_off_sectors)
        self._filter_negative_trend = bool(filter_negative_trend)
        self._min_universe = int(min_universe)
        self._top_n = top_n
        self._require_both = bool(require_both)
        self._roots = normalise_requested_roots(roots)
        validate_requested_roots(self._roots, instrument_master, stage="regime")
        self._instrument_master = instrument_master

    @property
    def name(self) -> str:
        """Registry name of this selector."""
        return "regime"

    @property
    def roots(self) -> tuple[InstrumentId, ...]:
        """Explicit root allow-list from ``universe.params.roots`` (``()`` = all)."""
        return self._roots

    # -- diagnostics ----------------------------------------------------------

    def _trend(self, state: MarketState, instrument_id: InstrumentId) -> float:
        realised = trailing_sum(state, instrument_id, self._trend_lookback_days)
        if realised is not None:
            return realised
        return finite_or(state.feature(instrument_id, RET_126D, math.nan), 0.0)

    def _shock(self, state: MarketState, instrument_id: InstrumentId) -> float:
        realised = trailing_sum(state, instrument_id, self._shock_lookback_days)
        if realised is not None:
            return realised
        # No return series: a shock detector with no history must report "no
        # shock" rather than invent one from a single-day feature.
        return 0.0

    def _daily_volatility(self, state: MarketState, instrument_id: InstrumentId) -> float:
        annual = annualized_volatility(state, instrument_id, lookback=self._vol_lookback_days)
        if annual is None or annual <= 0.0:
            return DEFAULT_ANNUAL_VOLATILITY / ANNUALIZATION_FACTOR
        return annual / ANNUALIZATION_FACTOR

    def _forced_risk_off(self, state: MarketState) -> bool:
        if state.regime is not None and state.regime.strip().lower() in RISK_OFF_REGIME_LABELS:
            return True
        return any(bool(state.metadata.get(flag, False)) for flag in RISK_OFF_METADATA_FLAGS)

    def _excluded_roots(self) -> set[InstrumentId]:
        excluded = set(self._risk_off_roots)
        if self._instrument_master is not None and self._risk_off_sectors:
            for sector in self._risk_off_sectors:
                for spec in self._instrument_master.by_sector(sector):
                    excluded.add(spec.root)
        return excluded

    def diagnose(self, state: MarketState) -> RegimeSnapshot:
        """Compute the full regime diagnosis for ``state``."""
        instrument_ids = list(state.instruments)
        trends = {
            instrument_id: self._trend(state, instrument_id) for instrument_id in instrument_ids
        }
        shocks = {
            instrument_id: self._shock(state, instrument_id) for instrument_id in instrument_ids
        }
        annual_vols: list[float] = []
        for instrument_id in instrument_ids:
            annual = annualized_volatility(state, instrument_id, lookback=self._vol_lookback_days)
            if annual is not None and annual > 0.0:
                annual_vols.append(annual)
        macro_volatility = float(np.median(annual_vols)) if annual_vols else 0.0
        negative = sum(1 for trend in trends.values() if trend < 0.0)
        breadth_negative = negative / len(instrument_ids) if instrument_ids else 0.0
        mean_trend = float(np.mean(list(trends.values()))) if trends else 0.0
        shock_values = np.array([shocks[key] for key in instrument_ids], dtype=np.float64)
        mean_shock = float(np.mean(shock_values)) if shock_values.size else 0.0
        daily_vols = np.array(
            [self._daily_volatility(state, instrument_id) for instrument_id in instrument_ids],
            dtype=np.float64,
        )
        median_daily_vol = float(np.median(daily_vols)) if daily_vols.size else 0.0
        if median_daily_vol > 1e-12:
            dispersion = median_daily_vol * math.sqrt(float(self._shock_lookback_days))
            shock_zscore = mean_shock / dispersion
        else:
            shock_zscore = 0.0
        slow_risk_off = (
            breadth_negative >= self._breadth_threshold
            and macro_volatility >= self._macro_vol_threshold
        )
        fast_risk_off = (
            shock_zscore <= -self._shock_z_threshold or mean_shock <= -self._shock_return_threshold
        )
        forced = self._forced_risk_off(state)
        if self._require_both:
            detected = slow_risk_off and fast_risk_off
        else:
            detected = slow_risk_off or fast_risk_off
        return RegimeSnapshot(
            risk_off=bool(detected or forced),
            slow_risk_off=bool(slow_risk_off),
            fast_risk_off=bool(fast_risk_off),
            forced_risk_off=bool(forced),
            breadth_negative=float(breadth_negative),
            macro_volatility=float(macro_volatility),
            mean_trend=float(mean_trend),
            mean_shock=float(mean_shock),
            shock_zscore=float(shock_zscore),
            trends=trends,
            shocks=shocks,
        )

    # -- selection ------------------------------------------------------------

    def select(self, state: MarketState) -> list[InstrumentId]:
        """Return the regime-filtered universe, strongest trend first."""
        diagnosis = self.diagnose(state)
        # The regime signal is computed on the full state (market regime), but the
        # tradable candidate universe is restricted to the explicit allow-list.
        universe = list(diagnosis.trends)
        if self._roots:
            allowed = set(self._roots)
            universe = [instrument_id for instrument_id in universe if instrument_id in allowed]
            if not universe:
                return []
        ranked = sorted(
            universe,
            key=lambda instrument_id: (-diagnosis.trends[instrument_id], instrument_id),
        )
        if not diagnosis.risk_off:
            selected = ranked
        else:
            excluded = self._excluded_roots()
            eligible = [instrument_id for instrument_id in ranked if instrument_id not in excluded]
            if not eligible:
                # Every root was explicitly excluded: an empty universe is the
                # honest answer and the pipeline handles it (flat book).
                selected = []
            elif self._filter_negative_trend:
                positive = [
                    instrument_id
                    for instrument_id in eligible
                    if diagnosis.trends[instrument_id] > 0.0
                ]
                # Never freeze the pipeline on a market-wide sell-off: de-risk to
                # the least-negative eligible roots until the minimum is met.
                selected = (
                    positive
                    if len(positive) >= self._min_universe
                    else eligible[: self._min_universe]
                )
            else:
                selected = eligible
        return selected[: self._top_n] if self._top_n is not None else selected


def build_regime_selector(
    params: Mapping[str, StageParam],
    instruments: InstrumentsConfig | None = None,
) -> UniverseSelector:
    """Build a :class:`RegimeSelector` from declarative stage parameters."""
    reader = ParamReader("selector.regime", params)
    selector = RegimeSelector(
        trend_lookback_days=reader.as_int("trend_lookback_days", 126),
        vol_lookback_days=reader.as_int("vol_lookback_days", 60),
        shock_lookback_days=reader.as_int("shock_lookback_days", 3),
        shock_z_threshold=reader.as_float("shock_z_threshold", 3.0),
        shock_return_threshold=reader.as_float("shock_return_threshold", 0.02),
        macro_vol_threshold=reader.as_float("macro_vol_threshold", 0.22),
        breadth_threshold=reader.as_float("breadth_threshold", 0.60),
        risk_off_roots=reader.as_sequence("risk_off_roots"),
        risk_off_sectors=reader.as_sequence("risk_off_sectors"),
        filter_negative_trend=reader.as_bool("filter_negative_trend", True),
        min_universe=reader.as_int("min_universe", 1),
        top_n=reader.as_optional_int("top_n"),
        require_both=reader.as_bool("require_both", False),
        # Optional Studio-saved explicit root filter (intersected with state roots).
        roots=reader.as_sequence("roots"),
        instrument_master=instruments,
    )
    reader.check()
    return selector


__all__ = ["RegimeSelector", "RegimeSnapshot", "build_regime_selector"]
