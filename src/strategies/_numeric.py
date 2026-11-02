"""Numerical helpers shared by selectors, allocators and timing overlays.

Everything here is deterministic, pure and defensive: market states produced by
fixtures or by the data layer may be missing a covariance matrix, a return
series or a named feature, and a strategy stage must degrade gracefully instead
of emitting ``NaN`` weights (a ``NaN`` weight would either crash the contract
translator or, worse, silently flatten a live book).

Conventions
-----------
* Volatilities and covariances are **annualised** unless a function name says
  ``daily``; ``sqrt(252)`` is the annualisation factor.
* ``state.returns`` are **daily** simple returns.
* ``state.covariance`` (when present) is an annualised covariance matrix whose
  ordering is given by ``state.covariance_instruments``.
* ``vol_20d`` / ``vol_60d`` features are daily volatilities and are annualised
  on read.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from src.core.types import TRADING_DAYS_PER_YEAR, InstrumentId, MarketState, WeightVector
from src.data.schema import VOL_20D, VOL_60D

ANNUALIZATION_FACTOR: float = math.sqrt(float(TRADING_DAYS_PER_YEAR))

#: Conservative fallback annualised volatility when no estimate is available.
DEFAULT_ANNUAL_VOLATILITY: float = 0.15

#: Minimum annualised volatility used to floor covariance diagonals.
MIN_ANNUAL_VOLATILITY: float = 0.002

#: Volatility features that may substitute for a realised estimate (daily units).
VOLATILITY_FEATURES: tuple[str, ...] = (VOL_20D, VOL_60D)

type FloatArray = NDArray[np.float64]


def finite_or(value: float, default: float) -> float:
    """Return ``value`` when finite, otherwise ``default``."""
    return value if math.isfinite(value) else default


def clipped(value: float, lower: float, upper: float) -> float:
    """Clamp ``value`` into ``[lower, upper]`` (also sanitising ``NaN``)."""
    if not math.isfinite(value):
        return lower
    return max(lower, min(upper, value))


def returns_series(state: MarketState, instrument_id: InstrumentId) -> FloatArray | None:
    """Daily return series for ``instrument_id`` (``None`` when unavailable)."""
    series = state.returns.get(instrument_id)
    if series is None or series.size == 0:
        return None
    return np.asarray(series, dtype=np.float64)


def trailing_sum(state: MarketState, instrument_id: InstrumentId, window: int) -> float | None:
    """Cumulative return over the last ``window`` observations (``None`` if short)."""
    series = returns_series(state, instrument_id)
    if series is None or series.size < min(2, window):
        return None
    used = series[-min(window, series.size) :]
    total = float(np.sum(used))
    return total if math.isfinite(total) else None


def trailing_volatility(
    state: MarketState,
    instrument_id: InstrumentId,
    window: int,
    *,
    annualize: bool = True,
) -> float | None:
    """Realised volatility of the last ``window`` daily returns."""
    series = returns_series(state, instrument_id)
    if series is None or series.size < 3:
        return None
    used = series[-min(window, series.size) :]
    variance = float(np.var(used, ddof=1))
    if not math.isfinite(variance) or variance < 0.0:
        return None
    volatility = math.sqrt(variance)
    return volatility * ANNUALIZATION_FACTOR if annualize else volatility


def feature_volatility(state: MarketState, instrument_id: InstrumentId) -> float | None:
    """Annualised volatility implied by the ``vol_20d`` / ``vol_60d`` features."""
    for feature_name in VOLATILITY_FEATURES:
        value = state.feature(instrument_id, feature_name, math.nan)
        if math.isfinite(value) and value > 0.0:
            return value * ANNUALIZATION_FACTOR
    return None


def annualized_volatility(
    state: MarketState,
    instrument_id: InstrumentId,
    *,
    lookback: int = 60,
) -> float | None:
    """Best available annualised volatility estimate for one instrument.

    Priority: covariance diagonal (cross-sectional, forward-looking-free) ->
    realised returns -> feature volatility. ``None`` when nothing is known.
    """
    covariance_vol = state.annualized_vol(instrument_id)
    if math.isfinite(covariance_vol) and covariance_vol > 0.0:
        return covariance_vol
    realised = trailing_volatility(state, instrument_id, lookback)
    if realised is not None and realised > 0.0:
        return realised
    return feature_volatility(state, instrument_id)


def _correlation_from_returns(
    candidates: Sequence[InstrumentId],
    state: MarketState,
    lookback: int,
) -> FloatArray:
    """Correlation matrix estimated from aligned return series.

    Falls back to the identity matrix whenever fewer than two complete series are
    available: a risk model that assumes zero correlation is conservative next
    to a model that invents correlations from a handful of observations.
    """
    count = len(candidates)
    identity = np.eye(count, dtype=np.float64)
    usable: list[tuple[int, FloatArray]] = []
    for index, instrument_id in enumerate(candidates):
        series = returns_series(state, instrument_id)
        if series is None or series.size < 5:
            continue
        usable.append((index, series[-min(lookback, series.size) :]))
    if len(usable) < 2:
        return identity
    length = min(int(series.size) for _, series in usable)
    if length < 5:
        return identity
    matrix = np.zeros((count, length), dtype=np.float64)
    present = np.zeros(count, dtype=bool)
    for index, series in usable:
        matrix[index] = series[-length:]
        present[index] = True
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = np.corrcoef(matrix)
    correlation = np.asarray(correlation, dtype=np.float64)
    if correlation.shape != (count, count) or not np.all(np.isfinite(correlation)):
        return identity
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(correlation, 1.0)
    correlation = np.clip(correlation, -1.0, 1.0)
    return 0.5 * (correlation + correlation.T)


def _regularize_covariance(
    covariance: FloatArray,
    *,
    ridge: float,
    shrinkage: float,
    min_volatility: float,
    default_volatility: float,
) -> FloatArray:
    """Return a symmetric positive-definite covariance matrix.

    Steps: symmetrise, clip non-finite entries, shrink towards the diagonal,
    floor near-zero variances with a fallback volatility, add a ridge, then
    shift any residual negative eigenvalue away from zero. Singular or
    rank-deficient covariance matrices (fewer observations than assets, stale
    instrument masters) are therefore safe to invert and to sample from.
    """
    matrix = 0.5 * (covariance + covariance.T)
    matrix = np.where(np.isfinite(matrix), matrix, 0.0)
    count = matrix.shape[0]
    if count == 0:
        return matrix
    if shrinkage > 0.0:
        diagonal = np.diag(np.diag(matrix))
        matrix = (1.0 - shrinkage) * matrix + shrinkage * diagonal
    variances = np.diag(matrix).copy()
    floor = max(min_volatility, 0.0) ** 2
    weak = np.flatnonzero(variances < floor)
    if weak.size:
        replacement = max(default_volatility, min_volatility) ** 2
        matrix[weak, :] = 0.0
        matrix[:, weak] = 0.0
        matrix[weak, weak] = replacement
    if ridge > 0.0:
        matrix = matrix + ridge * np.eye(count, dtype=np.float64)
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues = np.linalg.eigvalsh(matrix)
    if eigenvalues.size and float(eigenvalues[0]) < 0.0:
        matrix = matrix + (-float(eigenvalues[0]) + 1e-12) * np.eye(count, dtype=np.float64)
    return matrix


def covariance_matrix(
    state: MarketState,
    candidates: Sequence[InstrumentId],
    *,
    lookback: int = 60,
    ridge: float = 1e-10,
    shrinkage: float = 0.0,
    min_volatility: float = MIN_ANNUAL_VOLATILITY,
    default_volatility: float = DEFAULT_ANNUAL_VOLATILITY,
) -> FloatArray:
    """Annualised covariance matrix aligned to ``candidates``.

    Uses the state's covariance sub-matrix when every candidate is covered;
    otherwise rebuilds one from realised volatilities and correlations. The
    result is always finite and positive definite.
    """
    count = len(candidates)
    if count == 0:
        return np.zeros((0, 0), dtype=np.float64)
    submatrix = state.covariance_submatrix(candidates) if state.covariance is not None else None
    if submatrix is not None:
        return _regularize_covariance(
            submatrix,
            ridge=ridge,
            shrinkage=shrinkage,
            min_volatility=min_volatility,
            default_volatility=default_volatility,
        )
    volatilities = np.empty(count, dtype=np.float64)
    for index, instrument_id in enumerate(candidates):
        estimate = annualized_volatility(state, instrument_id, lookback=lookback)
        volatilities[index] = (
            estimate if estimate is not None and estimate > 0.0 else default_volatility
        )
    correlation = _correlation_from_returns(candidates, state, lookback)
    covariance = correlation * np.outer(volatilities, volatilities)
    return _regularize_covariance(
        covariance,
        ridge=ridge,
        shrinkage=shrinkage,
        min_volatility=min_volatility,
        default_volatility=default_volatility,
    )


def annualized_volatility_vector(
    state: MarketState,
    candidates: Sequence[InstrumentId],
    *,
    lookback: int = 60,
) -> FloatArray:
    """Annualised volatility per candidate, derived from the regularised covariance."""
    covariance = covariance_matrix(state, candidates, lookback=lookback)
    variances = np.clip(np.diag(covariance), MIN_ANNUAL_VOLATILITY**2, None)
    return np.sqrt(variances)


def portfolio_volatility(weights: FloatArray, covariance: FloatArray) -> float:
    """Annualised portfolio volatility ``sqrt(w' S w)`` (finite, non-negative)."""
    if weights.size == 0:
        return 0.0
    variance = float(weights @ covariance @ weights)
    if not math.isfinite(variance) or variance <= 0.0:
        return 0.0
    return math.sqrt(variance)


def require_known_instruments(
    weights: Mapping[InstrumentId, float],
    state: MarketState,
    *,
    stage: str,
) -> None:
    """Raise ``ValueError`` when ``weights`` references an instrument not in ``state``."""
    unknown = sorted(key for key in weights if key not in state.instruments)
    if unknown:
        raise ValueError(
            f"{stage}: produced weights for instruments that are not part of the "
            f"market state: {unknown!r}"
        )


def sanitize_weights(
    weights: WeightVector,
    *,
    threshold: float = 1e-12,
    allowed: Sequence[InstrumentId] | None = None,
) -> WeightVector:
    """Drop non-finite and numerically negligible weights (waterfall hygiene).

    ``allowed`` optionally restricts the output to an explicit instrument list;
    the final pipeline passes its selector universe so that a misbehaving
    overlay can never leak an unintended instrument into the approved book.
    """
    permitted = set(allowed) if allowed is not None else None
    cleaned: dict[InstrumentId, float] = {}
    for instrument_id, value in weights.items():
        if permitted is not None and instrument_id not in permitted:
            continue
        if not math.isfinite(value) or abs(value) <= threshold:
            continue
        cleaned[instrument_id] = value
    return WeightVector(cleaned)
