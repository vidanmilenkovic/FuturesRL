"""Regularised mean-variance allocator (``A_t``).

Implements the FinRL-X classical mean-variance allocator without an external QP
solver: solve the first-order condition of

    ``max_w  mu' w - (lambda / 2) w' S w``   =>   ``w* = (1 / lambda) S^-1 mu``

using plain ``numpy.linalg``. The covariance matrix is regularised upstream
(variances floored, optional diagonal shrinkage, ridge), so ``S`` is invertible
even when the number of roots exceeds the history length; if the solve still
fails the code falls back to a pseudo-inverse and finally to equal weights, so it
can never emit ``NaN`` weights.

Expected returns are the trailing cumulative returns over ``expected_return_
window`` days (falling back to the ``ret_126d`` feature), i.e. a momentum-biased
estimate that is deliberately simple and auditable — nothing here peeks ahead.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
from configs.strategy import StageParam
from numpy.typing import NDArray

from src.core.interfaces import PortfolioAllocator
from src.core.types import InstrumentId, MarketState, WeightVector
from src.data.schema import RET_126D
from src.strategies._numeric import covariance_matrix, trailing_sum
from src.strategies._params import ParamReader


class MeanVarianceAllocator:
    """Risk-aversion scaled inverse-covariance allocator with a leverage cap."""

    __slots__ = (
        "_covariance_lookback_days",
        "_expected_return_window",
        "_gross_leverage",
        "_long_only",
        "_max_gross_leverage",
        "_max_weight",
        "_min_volatility",
        "_ridge",
        "_risk_aversion",
        "_shrinkage",
    )

    def __init__(
        self,
        *,
        risk_aversion: float = 5.0,
        ridge: float = 1e-8,
        gross_leverage: float = 1.0,
        max_gross_leverage: float = 3.0,
        max_weight: float | None = None,
        long_only: bool = True,
        expected_return_window: int = 126,
        covariance_lookback_days: int = 60,
        shrinkage: float = 0.0,
        min_volatility: float = 0.002,
    ) -> None:
        if risk_aversion <= 0.0:
            raise ValueError("risk_aversion must be strictly positive")
        if ridge < 0.0:
            raise ValueError("ridge must be non-negative")
        if gross_leverage < 0.0:
            raise ValueError("gross_leverage must be non-negative")
        if max_gross_leverage < 0.0:
            raise ValueError("max_gross_leverage must be non-negative")
        if max_weight is not None and max_weight <= 0.0:
            raise ValueError("max_weight must be strictly positive")
        if expected_return_window < 2:
            raise ValueError("expected_return_window must be at least 2")
        if covariance_lookback_days < 3:
            raise ValueError("covariance_lookback_days must be at least 3")
        if not 0.0 <= shrinkage < 1.0:
            raise ValueError("shrinkage must be within [0, 1)")
        if min_volatility <= 0.0:
            raise ValueError("min_volatility must be strictly positive")
        self._risk_aversion = float(risk_aversion)
        self._ridge = float(ridge)
        self._gross_leverage = float(gross_leverage)
        self._max_gross_leverage = float(max_gross_leverage)
        self._max_weight = max_weight
        self._long_only = bool(long_only)
        self._expected_return_window = int(expected_return_window)
        self._covariance_lookback_days = int(covariance_lookback_days)
        self._shrinkage = float(shrinkage)
        self._min_volatility = float(min_volatility)

    @property
    def name(self) -> str:
        """Registry name of this allocator."""
        return "mean_variance"

    # -- estimators -----------------------------------------------------------

    def expected_returns(
        self, candidates: Sequence[InstrumentId], state: MarketState
    ) -> NDArray[np.float64]:
        """Trailing expected-return estimate per candidate (``ret_126d`` fallback)."""
        estimates = np.zeros(len(candidates), dtype=np.float64)
        for index, instrument_id in enumerate(candidates):
            trailing = trailing_sum(state, instrument_id, self._expected_return_window)
            if trailing is not None and math.isfinite(trailing):
                estimates[index] = trailing
                continue
            feature = state.feature(instrument_id, RET_126D, math.nan)
            estimates[index] = feature if math.isfinite(feature) else 0.0
        return estimates

    def _inverse_covariance_weights(
        self,
        covariance: NDArray[np.float64],
        expected_returns: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        regularised = covariance + self._ridge * np.eye(covariance.shape[0], dtype=np.float64)
        try:
            solution = np.linalg.solve(regularised, expected_returns)
        except np.linalg.LinAlgError:
            solution = np.linalg.pinv(regularised) @ expected_returns
        solution = np.asarray(solution, dtype=np.float64)
        solution = np.nan_to_num(solution, nan=0.0, posinf=0.0, neginf=0.0)
        return solution / self._risk_aversion

    def _equal_fallback(self, count: int) -> NDArray[np.float64]:
        gross = self._gross_leverage if self._gross_leverage > 0.0 else 1.0
        return np.full(count, gross / count, dtype=np.float64)

    # -- allocation -----------------------------------------------------------

    def allocate(self, candidates: Sequence[InstrumentId], state: MarketState) -> WeightVector:
        """Return ``S^-1 mu / lambda`` with leverage and sign constraints applied."""
        count = len(candidates)
        if count == 0:
            return WeightVector.empty()
        covariance = covariance_matrix(
            state,
            candidates,
            lookback=self._covariance_lookback_days,
            shrinkage=self._shrinkage,
            min_volatility=self._min_volatility,
        )
        expected_returns = self.expected_returns(candidates, state)
        if not np.any(expected_returns):
            return WeightVector.from_array(candidates, self._equal_fallback(count))
        weights = self._inverse_covariance_weights(covariance, expected_returns)
        if self._long_only:
            weights = np.maximum(weights, 0.0)
        gross = float(np.sum(np.abs(weights)))
        if not np.isfinite(gross) or gross <= 1e-12:
            return WeightVector.from_array(candidates, self._equal_fallback(count))
        if gross > self._max_gross_leverage > 0.0:
            weights = weights * (self._max_gross_leverage / gross)
        if self._max_weight is not None:
            peak = float(np.max(np.abs(weights)))
            if peak > self._max_weight:
                weights = weights * (self._max_weight / peak)
        return WeightVector.from_array(candidates, weights)


def build_mean_variance_allocator(params: Mapping[str, StageParam]) -> PortfolioAllocator:
    """Build a :class:`MeanVarianceAllocator` from declarative stage parameters."""
    reader = ParamReader("allocator.mean_variance", params)
    allocator = MeanVarianceAllocator(
        risk_aversion=reader.as_float("risk_aversion", 5.0),
        ridge=reader.as_float("ridge", 1e-8),
        gross_leverage=reader.as_float("gross_leverage", 1.0),
        max_gross_leverage=reader.as_float("max_gross_leverage", 3.0),
        max_weight=reader.as_optional_float("max_weight"),
        long_only=reader.as_bool("long_only", True),
        expected_return_window=reader.as_int("expected_return_window", 126),
        covariance_lookback_days=reader.as_int("covariance_lookback_days", 60),
        shrinkage=reader.as_float("shrinkage", 0.0),
        min_volatility=reader.as_float("min_volatility", 0.002),
    )
    reader.check()
    return allocator


__all__ = ["MeanVarianceAllocator", "build_mean_variance_allocator"]
