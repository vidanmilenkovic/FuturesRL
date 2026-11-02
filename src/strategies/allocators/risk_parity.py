"""Equal-risk-contribution (risk parity) allocator (``A_t``).

Solves the classic risk-budgeting problem

    ``w_i * (S w)_i = b_i * w' S w``   for every ``i``

with ``b_i`` the risk budget (equal by default) and ``S`` the annualised
covariance matrix, using a deterministic multiplicative fixed-point iteration.
It is deliberately *not* an external QP solver: plain NumPy keeps the module
dependency-free and auditable.

Futures-specific robustness
---------------------------
* **Singular / rank-deficient covariance** (more roots than observations, stale
  covariance, duplicated micro/standard pairs) is handled upstream by
  :func:`~src.strategies._numeric.covariance_matrix`, which floors variances,
  shrinks towards the diagonal and shifts away any negative eigenvalue.
* **Missing covariance / missing returns**: the same helper rebuilds a
  covariance matrix from realised volatilities and correlations, so a state
  without a covariance matrix still produces a sensible book.
* Equal-risk-contribution weights are long-only by construction (a short leg has
  negative “risk contribution” and breaks the budget identity). The timing
  overlays apply the directional view; this allocator sizes risk.

The result is scaled either to a gross leverage target or, when
``target_annual_volatility`` is given, so the ex-ante portfolio volatility
matches the target (capped by ``max_gross_leverage``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from configs.strategy import StageParam
from numpy.typing import NDArray

from src.core.interfaces import PortfolioAllocator
from src.core.types import InstrumentId, MarketState, WeightVector
from src.strategies._numeric import covariance_matrix, portfolio_volatility
from src.strategies._params import ParamReader


class RiskParityAllocator:
    """Equal risk contribution across the candidate futures roots."""

    __slots__ = (
        "_covariance_lookback_days",
        "_gross_leverage",
        "_max_gross_leverage",
        "_max_iterations",
        "_max_weight",
        "_min_volatility",
        "_ridge",
        "_risk_budgets",
        "_shrinkage",
        "_target_annual_volatility",
        "_tolerance",
    )

    def __init__(
        self,
        *,
        target_annual_volatility: float | None = None,
        gross_leverage: float = 1.0,
        max_gross_leverage: float = 3.0,
        max_weight: float | None = None,
        risk_budgets: Mapping[InstrumentId, float] | None = None,
        covariance_lookback_days: int = 60,
        max_iterations: int = 500,
        tolerance: float = 1e-10,
        ridge: float = 1e-10,
        shrinkage: float = 0.0,
        min_volatility: float = 0.002,
    ) -> None:
        if target_annual_volatility is not None and target_annual_volatility <= 0.0:
            raise ValueError("target_annual_volatility must be strictly positive")
        if gross_leverage < 0.0:
            raise ValueError("gross_leverage must be non-negative")
        if max_gross_leverage < 0.0:
            raise ValueError("max_gross_leverage must be non-negative")
        if max_weight is not None and max_weight <= 0.0:
            raise ValueError("max_weight must be strictly positive")
        if covariance_lookback_days < 3:
            raise ValueError("covariance_lookback_days must be at least 3")
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if tolerance <= 0.0:
            raise ValueError("tolerance must be strictly positive")
        if ridge < 0.0:
            raise ValueError("ridge must be non-negative")
        if not 0.0 <= shrinkage < 1.0:
            raise ValueError("shrinkage must be within [0, 1)")
        if min_volatility <= 0.0:
            raise ValueError("min_volatility must be strictly positive")
        self._target_annual_volatility = target_annual_volatility
        self._gross_leverage = float(gross_leverage)
        self._max_gross_leverage = float(max_gross_leverage)
        self._max_weight = max_weight
        self._risk_budgets = dict(risk_budgets) if risk_budgets is not None else None
        self._covariance_lookback_days = int(covariance_lookback_days)
        self._max_iterations = int(max_iterations)
        self._tolerance = float(tolerance)
        self._ridge = float(ridge)
        self._shrinkage = float(shrinkage)
        self._min_volatility = float(min_volatility)

    @property
    def name(self) -> str:
        """Registry name of this allocator."""
        return "risk_parity"

    # -- risk budgets ---------------------------------------------------------

    def _budgets(self, candidates: Sequence[InstrumentId]) -> NDArray[np.float64]:
        count = len(candidates)
        if self._risk_budgets is None:
            return np.full(count, 1.0 / count, dtype=np.float64)
        raw = np.array(
            [max(float(self._risk_budgets.get(candidate, 0.0)), 0.0) for candidate in candidates],
            dtype=np.float64,
        )
        total = float(raw.sum())
        if total <= 1e-12:
            return np.full(count, 1.0 / count, dtype=np.float64)
        return raw / total

    # -- solver ---------------------------------------------------------------

    def _equal_risk_contribution(
        self,
        covariance: NDArray[np.float64],
        budgets: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Multiplicative fixed-point solve of the risk-budgeting equations."""
        count = covariance.shape[0]
        variances = np.clip(np.diag(covariance), self._min_volatility**2, None)
        inverse_volatility = 1.0 / np.sqrt(variances)
        weights = inverse_volatility / float(inverse_volatility.sum())
        for _ in range(self._max_iterations):
            marginal = covariance @ weights
            contributions = weights * marginal
            total = float(contributions.sum())
            if not np.isfinite(total) or total <= 1e-15:
                break
            targets = budgets * total
            ratio = np.where(
                contributions > 1e-18,
                targets / np.maximum(contributions, 1e-18),
                1.0,
            )
            updated = weights * np.sqrt(np.clip(ratio, 1e-8, 1e8))
            updated = np.maximum(updated, 0.0)
            normaliser = float(updated.sum())
            if not np.isfinite(normaliser) or normaliser <= 1e-15:
                break
            updated = updated / normaliser
            converged = bool(np.max(np.abs(updated - weights)) <= self._tolerance)
            weights = updated
            if converged:
                break
        total_weight = float(weights.sum())
        if total_weight <= 1e-15 or not np.isfinite(total_weight):
            return np.full(count, 1.0 / count, dtype=np.float64)
        return weights / total_weight

    # -- allocation -----------------------------------------------------------

    def allocate(self, candidates: Sequence[InstrumentId], state: MarketState) -> WeightVector:
        """Allocate by equal risk contribution and scale to the risk target."""
        if not candidates:
            return WeightVector.empty()
        covariance = covariance_matrix(
            state,
            candidates,
            lookback=self._covariance_lookback_days,
            ridge=self._ridge,
            shrinkage=self._shrinkage,
            min_volatility=self._min_volatility,
        )
        weights = self._equal_risk_contribution(covariance, self._budgets(candidates))
        if self._max_weight is not None:
            peak = float(weights.max())
            if peak > self._max_weight:
                weights = weights * (self._max_weight / peak)
        if self._target_annual_volatility is not None:
            volatility = portfolio_volatility(weights, covariance)
            if volatility > 1e-12:
                scale = self._target_annual_volatility / volatility
            else:
                scale = self._gross_leverage
            scale = min(scale, self._max_gross_leverage)
        else:
            scale = self._gross_leverage
        return WeightVector.from_array(candidates, weights * scale)


def build_risk_parity_allocator(params: Mapping[str, StageParam]) -> PortfolioAllocator:
    """Build a :class:`RiskParityAllocator` from declarative stage parameters."""
    reader = ParamReader("allocator.risk_parity", params)
    allocator = RiskParityAllocator(
        target_annual_volatility=reader.as_optional_float("target_annual_volatility"),
        gross_leverage=reader.as_float("gross_leverage", 1.0),
        max_gross_leverage=reader.as_float("max_gross_leverage", 3.0),
        max_weight=reader.as_optional_float("max_weight"),
        covariance_lookback_days=reader.as_int("covariance_lookback_days", 60),
        max_iterations=reader.as_int("max_iterations", 500),
        tolerance=reader.as_float("tolerance", 1e-10),
        ridge=reader.as_float("ridge", 1e-10),
        shrinkage=reader.as_float("shrinkage", 0.0),
        min_volatility=reader.as_float("min_volatility", 0.002),
    )
    reader.check()
    return allocator


__all__ = ["RiskParityAllocator", "build_risk_parity_allocator"]
