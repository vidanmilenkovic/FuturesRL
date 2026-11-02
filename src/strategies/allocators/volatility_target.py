"""Inverse-volatility allocator with portfolio volatility targeting (``A_t``).

Two ideas, in this order:

1. **Inverse-volatility sizing.** Each root receives a weight inversely
   proportional to its annualised volatility, correcting for correlation when a
   covariance matrix is available. High-volatility energy roots therefore get a
   smaller notional than low-volatility rates roots, which is the whole point of
   volatility-normalised futures allocation.
2. **Ex-ante volatility targeting.** The whole book is scaled so that
   ``sqrt(w' S w)`` equals ``target_annual_volatility``. The scale factor is
   capped by ``max_gross_leverage`` so a collapsed ex-ante volatility can never
   demand absurd leverage (a classic way to blow up a vol-targeted book).

This is the allocator behind ``configs/strategies/tsmom_vol_target.yaml`` and the
canonical "Volatility-Targeted TSMOM" building block of the FinRL-X architecture.
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


class VolatilityTargetAllocator:
    """Inverse-volatility weights scaled to an ex-ante portfolio volatility target."""

    __slots__ = (
        "_covariance_lookback_days",
        "_max_gross_leverage",
        "_max_weight",
        "_min_gross_leverage",
        "_min_volatility",
        "_ridge",
        "_shrinkage",
        "_target_annual_volatility",
    )

    def __init__(
        self,
        *,
        target_annual_volatility: float = 0.12,
        covariance_lookback_days: int = 60,
        max_gross_leverage: float = 3.0,
        min_gross_leverage: float = 0.0,
        max_weight: float | None = None,
        ridge: float = 1e-10,
        shrinkage: float = 0.0,
        min_volatility: float = 0.002,
    ) -> None:
        if target_annual_volatility <= 0.0:
            raise ValueError("target_annual_volatility must be strictly positive")
        if covariance_lookback_days < 3:
            raise ValueError("covariance_lookback_days must be at least 3")
        if max_gross_leverage <= 0.0:
            raise ValueError("max_gross_leverage must be strictly positive")
        if min_gross_leverage < 0.0:
            raise ValueError("min_gross_leverage must be non-negative")
        if min_gross_leverage > max_gross_leverage:
            raise ValueError("min_gross_leverage must not exceed max_gross_leverage")
        if max_weight is not None and max_weight <= 0.0:
            raise ValueError("max_weight must be strictly positive")
        if ridge < 0.0:
            raise ValueError("ridge must be non-negative")
        if not 0.0 <= shrinkage < 1.0:
            raise ValueError("shrinkage must be within [0, 1)")
        if min_volatility <= 0.0:
            raise ValueError("min_volatility must be strictly positive")
        self._target_annual_volatility = float(target_annual_volatility)
        self._covariance_lookback_days = int(covariance_lookback_days)
        self._max_gross_leverage = float(max_gross_leverage)
        self._min_gross_leverage = float(min_gross_leverage)
        self._max_weight = max_weight
        self._ridge = float(ridge)
        self._shrinkage = float(shrinkage)
        self._min_volatility = float(min_volatility)

    @property
    def name(self) -> str:
        """Registry name of this allocator."""
        return "volatility_target"

    def inverse_volatility_weights(self, covariance: NDArray[np.float64]) -> NDArray[np.float64]:
        """Unit-gross inverse-volatility weights from a covariance matrix."""
        count = int(covariance.shape[0])
        variances = np.clip(np.diag(covariance), self._min_volatility**2, None)
        inverse = 1.0 / np.sqrt(variances)
        total = float(inverse.sum())
        if not np.isfinite(total) or total <= 1e-15:
            return np.full(count, 1.0 / count, dtype=np.float64)
        return inverse / total

    def allocate(self, candidates: Sequence[InstrumentId], state: MarketState) -> WeightVector:
        """Allocate inverse-volatility weights and scale to the volatility target."""
        count = len(candidates)
        if count == 0:
            return WeightVector.empty()
        covariance = covariance_matrix(
            state,
            candidates,
            lookback=self._covariance_lookback_days,
            ridge=self._ridge,
            shrinkage=self._shrinkage,
            min_volatility=self._min_volatility,
        )
        weights = self.inverse_volatility_weights(covariance)
        ex_ante_volatility = portfolio_volatility(weights, covariance)
        if ex_ante_volatility > 1e-12:
            scale = self._target_annual_volatility / ex_ante_volatility
        else:
            scale = self._max_gross_leverage
        scale = max(self._min_gross_leverage, min(scale, self._max_gross_leverage))
        weights = weights * scale
        if self._max_weight is not None:
            peak = float(np.max(np.abs(weights)))
            if peak > self._max_weight:
                weights = weights * (self._max_weight / peak)
        return WeightVector.from_array(candidates, weights)


def build_volatility_target_allocator(params: Mapping[str, StageParam]) -> PortfolioAllocator:
    """Build a :class:`VolatilityTargetAllocator` from declarative stage parameters."""
    reader = ParamReader("allocator.volatility_target", params)
    allocator = VolatilityTargetAllocator(
        target_annual_volatility=reader.as_float("target_annual_volatility", 0.12),
        covariance_lookback_days=reader.as_int("covariance_lookback_days", 60),
        max_gross_leverage=reader.as_float("max_gross_leverage", 3.0),
        min_gross_leverage=reader.as_float("min_gross_leverage", 0.0),
        max_weight=reader.as_optional_float("max_weight"),
        ridge=reader.as_float("ridge", 1e-10),
        shrinkage=reader.as_float("shrinkage", 0.0),
        min_volatility=reader.as_float("min_volatility", 0.002),
    )
    reader.check()
    return allocator


__all__ = ["VolatilityTargetAllocator", "build_volatility_target_allocator"]
