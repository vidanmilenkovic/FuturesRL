"""Shared observation vector construction for DRL policies.

A Stable-Baselines3 policy trained on :class:`~src.strategies.envs.FuturesTradingEnv`
expects a flat ``float`` vector laid out exactly as the environment builds it:

    ``[ per-root features (n_assets * n_features), current weights (n_assets), equity ratio ]``

:func:`build_observation` is the single implementation of that layout so the
training environment and the live :class:`~src.strategies.allocators.DRLAllocator`
can never drift apart (an observation-layout mismatch silently destroys a
trained policy's behaviour, it does not raise).

The default feature set is the FinRL-X feature block required by
``src.data.schema.REQUIRED_FEATURES``-style market states.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from src.core.types import InstrumentId, MarketState
from src.data.schema import (
    CARRY_ANNUALIZED,
    RET_1D,
    RET_21D,
    RET_126D,
    TERM_SLOPE,
    VOL_20D,
    VOLUME_OI_RATIO,
)

#: Per-instrument features exposed to the policy in default configuration.
DEFAULT_OBSERVATION_FEATURES: tuple[str, ...] = (
    RET_1D,
    RET_21D,
    RET_126D,
    VOL_20D,
    TERM_SLOPE,
    CARRY_ANNUALIZED,
    VOLUME_OI_RATIO,
)


def feature_matrix(
    state: MarketState,
    instrument_ids: Sequence[InstrumentId],
    feature_names: Sequence[str] = DEFAULT_OBSERVATION_FEATURES,
) -> NDArray[np.float64]:
    """Dense ``(n_assets, n_features)`` matrix of point-in-time features.

    Missing or non-finite features are represented as ``0.0`` so that a policy
    never observes ``NaN`` (which would poison every downstream gradient).
    """
    matrix = np.zeros((len(instrument_ids), len(feature_names)), dtype=np.float64)
    for row, instrument_id in enumerate(instrument_ids):
        snapshot = state.instruments.get(instrument_id)
        if snapshot is None:
            continue
        for column, feature_name in enumerate(feature_names):
            value = snapshot.feature(feature_name, math.nan)
            if math.isfinite(value):
                matrix[row, column] = value
    return matrix


def observation_from_components(
    features: NDArray[np.float64],
    weights: NDArray[np.float64] | None = None,
    *,
    equity_ratio: float = 1.0,
) -> NDArray[np.float64]:
    """Assemble the flat observation vector from pre-computed components.

    ``features`` is ``(n_assets, n_features)``; ``weights`` is the currently held
    signed notional weight vector (zeros when omitted); ``equity_ratio`` is
    ``equity / initial_equity``.
    """
    n_assets = int(features.shape[0])
    held = (
        np.zeros(n_assets, dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    if held.shape != (n_assets,):
        raise ValueError(f"weights shape {held.shape} does not match {n_assets} assets")
    flat = np.concatenate(
        [
            np.asarray(features, dtype=np.float64).reshape(-1),
            held,
            np.array([float(equity_ratio)], dtype=np.float64),
        ]
    )
    sanitised = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
    return np.asarray(sanitised, dtype=np.float64)


def build_observation(
    state: MarketState,
    instrument_ids: Sequence[InstrumentId],
    *,
    feature_names: Sequence[str] = DEFAULT_OBSERVATION_FEATURES,
    current_weights: Mapping[InstrumentId, float] | None = None,
    equity_ratio: float = 1.0,
) -> NDArray[np.float64]:
    """Flat observation vector for ``instrument_ids`` from a market state."""
    features = feature_matrix(state, instrument_ids, feature_names)
    held = np.array(
        [
            float((current_weights or {}).get(instrument_id, 0.0))
            for instrument_id in instrument_ids
        ],
        dtype=np.float64,
    )
    return observation_from_components(features, held, equity_ratio=equity_ratio)


def observation_dimension(n_assets: int, n_features: int) -> int:
    """Length of the flat observation vector (features + weights + equity ratio)."""
    return n_assets * n_features + n_assets + 1
