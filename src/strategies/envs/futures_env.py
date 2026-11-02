"""Gymnasium futures trading environment for DRL allocation (Module 2).

``FuturesTradingEnv`` is the training ground for :class:`DRLAllocator` (PPO,
CPPO/CPRL, SAC, TD3, A2C) and is fully usable with plain ``gymnasium`` — the
Stable-Baselines3 stack is *not* imported here, so the environment can be unit
tested (and driven by any Gymnasium-compatible trainer) without torch.

Futures-specific mechanics
--------------------------
* **Continuous action = target weights.** The policy outputs a signed notional
  weight per root (``[-max_instrument_weight, +max_instrument_weight]``); gross
  leverage is capped by ``max_gross_leverage`` each step.
* **Transaction costs.** Commission is charged in basis points of traded
  notional; slippage is charged per traded contract at
  ``slippage_ticks * tick_size * multiplier`` — both legs of a rebalance pay.
* **Margin feasibility.** Initial margin is projected from the *actual* discrete
  contract count implied by the target weights. If projected margin utilisation
  exceeds ``max_margin_utilization`` the book is uniformly scaled down
  (a margin call, penalised). If equity falls below the maintenance margin the
  broker force-liquidates at a penalty and the episode terminates.
* **Episode termination.** Bankruptcy (equity below
  ``bankruptcy_threshold * initial_equity``) or a peak-to-trough drawdown beyond
  ``drawdown_limit`` ends the episode; reaching the end of the price matrix
  truncates it.
* **FinRL Risk-First Module 2 hook.** When :class:`~src.core.types.RiskAssessment`
  objects are supplied (via the constructor, ``reset(options=...)``,
  :meth:`set_risk_assessments` or the ``risk_assessments=`` keyword of
  :meth:`step`), the reward becomes

      ``net_pnl_return - lambda * sum_i |w_i| * I(c_i >= tau) * max(0, S_i - theta_safe)``

  i.e. the exact Lagrangian semantic-risk penalty from the architecture spec.
  Assessments below the confidence floor ``tau`` are ignored, so a hallucinating
  LLM cannot move the reward.

Reward units
------------
``reward = (net_pnl / equity_before) * reward_scaling - risk_penalty`` so that
the PnL term is a fractional return (scale-free across account sizes) and the
penalty is expressed in the same dimensionless units as the spec's
``sum_i |w_i|`` term. ``reward_scaling`` lets a trainer scale the PnL term
without touching the risk penalty.

Determinism
-----------
There is no stochasticity: ``reset(seed=...)`` restarts at the first bar with a
flat book, and the same action sequence always produces the same equity path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from src.core.types import InstrumentId, RiskAssessment
from src.strategies._observation import (
    DEFAULT_OBSERVATION_FEATURES,
    observation_dimension,
    observation_from_components,
)
from src.strategies._risk import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_SAFE_RISK_SCORE,
    assessments_by_instrument,
    semantic_risk_penalty,
)

#: Minimum equity below which the account is treated as bankrupt (fraction).
DEFAULT_BANKRUPTCY_THRESHOLD: float = 0.20


class FuturesTradingEnv(gymnasium.Env[NDArray[np.float64], NDArray[np.float32]]):
    """Margin-aware continuous-action futures portfolio environment."""

    def __init__(
        self,
        *,
        prices: NDArray[np.float64],
        instrument_ids: Sequence[InstrumentId],
        features: NDArray[np.float64] | None = None,
        feature_names: Sequence[str] | None = None,
        multipliers: Sequence[float] | None = None,
        initial_margins: Sequence[float] | None = None,
        maintenance_margins: Sequence[float] | None = None,
        tick_sizes: Sequence[float] | None = None,
        initial_equity: float = 1_000_000.0,
        commission_bps: float = 1.0,
        slippage_ticks: float = 1.0,
        max_gross_leverage: float = 1.0,
        max_instrument_weight: float = 1.0,
        max_margin_utilization: float = 0.60,
        maintenance_margin_fraction: float = 0.9,
        drawdown_limit: float = 0.20,
        bankruptcy_threshold: float = DEFAULT_BANKRUPTCY_THRESHOLD,
        margin_call_penalty_bps: float = 5.0,
        forced_liquidation_penalty_bps: float = 10.0,
        risk_penalty_lambda: float = 0.5,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        safe_risk_score: float = DEFAULT_SAFE_RISK_SCORE,
        reward_scaling: float = 1.0,
    ) -> None:
        super().__init__()
        self.metadata = {"render_modes": []}
        price_matrix = np.asarray(prices, dtype=np.float64)
        if price_matrix.ndim != 2:
            raise ValueError(
                f"prices must be a 2-D (n_steps, n_assets) array, got {price_matrix.ndim}-D"
            )
        if price_matrix.shape[0] < 2:
            raise ValueError("prices must contain at least two time steps")
        if price_matrix.shape[1] < 1:
            raise ValueError("prices must contain at least one asset")
        if not np.all(np.isfinite(price_matrix)) or np.any(price_matrix <= 0.0):
            raise ValueError("prices must be finite and strictly positive")
        asset_count = int(price_matrix.shape[1])
        ids = tuple(str(instrument_id) for instrument_id in instrument_ids)
        if len(ids) != asset_count:
            raise ValueError(f"instrument_ids ({len(ids)}) must match prices width ({asset_count})")
        if len(set(ids)) != len(ids):
            raise ValueError("instrument_ids must be unique")
        if initial_equity <= 0.0:
            raise ValueError("initial_equity must be strictly positive")
        if commission_bps < 0.0:
            raise ValueError("commission_bps must be non-negative")
        if slippage_ticks < 0.0:
            raise ValueError("slippage_ticks must be non-negative")
        if max_gross_leverage <= 0.0:
            raise ValueError("max_gross_leverage must be strictly positive")
        if max_instrument_weight <= 0.0:
            raise ValueError("max_instrument_weight must be strictly positive")
        if not 0.0 < max_margin_utilization <= 1.0:
            raise ValueError("max_margin_utilization must be within (0, 1]")
        if not 0.0 <= maintenance_margin_fraction <= 1.0:
            raise ValueError("maintenance_margin_fraction must be within [0, 1]")
        if not 0.0 < drawdown_limit <= 1.0:
            raise ValueError("drawdown_limit must be within (0, 1]")
        if not 0.0 <= bankruptcy_threshold < 1.0:
            raise ValueError("bankruptcy_threshold must be within [0, 1)")
        if margin_call_penalty_bps < 0.0 or forced_liquidation_penalty_bps < 0.0:
            raise ValueError("margin call / liquidation penalties must be non-negative")
        if risk_penalty_lambda < 0.0:
            raise ValueError("risk_penalty_lambda must be non-negative")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be within [0, 1]")
        if safe_risk_score < 1.0:
            raise ValueError("safe_risk_score must be at least 1.0")
        if reward_scaling <= 0.0:
            raise ValueError("reward_scaling must be strictly positive")
        self._prices = price_matrix
        self._instrument_ids = ids
        self._asset_count = asset_count
        self._step_count = int(price_matrix.shape[0])
        self._multipliers = self._asset_vector(multipliers, default=1.0, name="multipliers")
        self._tick_sizes = self._asset_vector(tick_sizes, default=1.0, name="tick_sizes")
        initial = self._asset_vector(initial_margins, default=0.0, name="initial_margins")
        maintenance = self._asset_vector(
            maintenance_margins, default=-1.0, name="maintenance_margins"
        )
        maintenance = np.where(
            maintenance < 0.0, initial * maintenance_margin_fraction, maintenance
        )
        if np.any(initial < 0.0) or np.any(maintenance < 0.0):
            raise ValueError("margins must be non-negative")
        self._initial_margins = initial
        self._maintenance_margins = maintenance
        if features is None:
            feature_matrix = np.zeros((self._step_count, asset_count, 0), dtype=np.float64)
            names: tuple[str, ...] = ()
        else:
            feature_matrix = np.asarray(features, dtype=np.float64)
            if feature_matrix.ndim != 3:
                raise ValueError("features must be a 3-D (n_steps, n_assets, n_features) array")
            expected = (self._step_count, asset_count)
            if feature_matrix.shape[:2] != expected:
                raise ValueError(
                    f"features shape {feature_matrix.shape[:2]} does not match prices {expected}"
                )
            if not np.all(np.isfinite(feature_matrix)):
                raise ValueError("features must be finite")
            if feature_names is None:
                names = tuple(f"feature_{index}" for index in range(feature_matrix.shape[2]))
            else:
                names = tuple(str(name) for name in feature_names)
                if len(names) != feature_matrix.shape[2]:
                    raise ValueError(
                        f"feature_names ({len(names)}) must match features depth "
                        f"({feature_matrix.shape[2]})"
                    )
        self._features = feature_matrix
        self._feature_names = names
        self._initial_equity = float(initial_equity)
        self._commission_rate = float(commission_bps) * 1e-4
        self._slippage_ticks = float(slippage_ticks)
        self._max_gross_leverage = float(max_gross_leverage)
        self._max_instrument_weight = float(max_instrument_weight)
        self._max_margin_utilization = float(max_margin_utilization)
        self._drawdown_limit = float(drawdown_limit)
        self._bankruptcy_threshold = float(bankruptcy_threshold)
        self._margin_call_penalty_rate = float(margin_call_penalty_bps) * 1e-4
        self._forced_liquidation_penalty_rate = float(forced_liquidation_penalty_bps) * 1e-4
        self._risk_penalty_lambda = float(risk_penalty_lambda)
        self._confidence_threshold = float(confidence_threshold)
        self._safe_risk_score = float(safe_risk_score)
        self._reward_scaling = float(reward_scaling)

        observation_size = observation_dimension(asset_count, self._features.shape[2])
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(observation_size,), dtype=np.float64
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(asset_count,), dtype=np.float32)

        # Episode state (initialised deterministically by reset()).
        self._index = 0
        self._equity = self._initial_equity
        self._peak_equity = self._initial_equity
        self._weights = np.zeros(asset_count, dtype=np.float64)
        self._risk_assessments: dict[InstrumentId, RiskAssessment] = {}
        self._finished = False

    # -- construction helpers -------------------------------------------------

    def _asset_vector(
        self, values: Sequence[float] | None, *, default: float, name: str
    ) -> NDArray[np.float64]:
        if values is None:
            return np.full(self._asset_count, default, dtype=np.float64)
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (self._asset_count,):
            raise ValueError(f"{name} must have {self._asset_count} entries, got {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite")
        return array

    # -- introspection --------------------------------------------------------

    @property
    def instrument_ids(self) -> tuple[InstrumentId, ...]:
        """Asset identifiers, aligned with the action/weight dimension."""
        return self._instrument_ids

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Per-asset feature names included in the observation."""
        return self._feature_names

    @property
    def n_assets(self) -> int:
        """Number of tradable futures roots."""
        return self._asset_count

    @property
    def equity(self) -> float:
        """Current account equity."""
        return self._equity

    @property
    def peak_equity(self) -> float:
        """High-water mark of the equity curve."""
        return self._peak_equity

    @property
    def current_weights(self) -> NDArray[np.float64]:
        """Currently held signed notional weights (read-only copy)."""
        copy = np.array(self._weights, copy=True)
        copy.setflags(write=False)
        return copy

    @property
    def index(self) -> int:
        """Current bar index within the price matrix."""
        return self._index

    # -- risk assessments -----------------------------------------------------

    def set_risk_assessments(
        self,
        assessments: Sequence[RiskAssessment] | Mapping[InstrumentId, RiskAssessment] | None,
    ) -> None:
        """Install the LLM risk assessments used by the reward-shaping hook."""
        self._risk_assessments = assessments_by_instrument(assessments)

    # -- observation ----------------------------------------------------------

    def _observation(self) -> NDArray[np.float64]:
        index = min(self._index, self._step_count - 1)
        features = self._features[index]
        return observation_from_components(
            features,
            self._weights,
            equity_ratio=self._equity / self._initial_equity,
        )

    def _info(self) -> dict[str, Any]:
        equity = self._equity
        return {
            "step": self._index,
            "equity": float(equity),
            "equity_ratio": float(equity / self._initial_equity),
            "weights": tuple(float(weight) for weight in self._weights),
            "gross_leverage": float(np.sum(np.abs(self._weights))),
            "drawdown": float(1.0 - equity / self._peak_equity) if self._peak_equity else 0.0,
        }

    # -- gymnasium API --------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float64], dict[str, Any]]:
        """Restart the episode at bar zero with a flat, fully funded book."""
        super().reset(seed=seed)
        self._index = 0
        self._equity = self._initial_equity
        self._peak_equity = self._initial_equity
        self._weights = np.zeros(self._asset_count, dtype=np.float64)
        self._finished = False
        self._risk_assessments = {}
        if options is not None and "risk_assessments" in options:
            self.set_risk_assessments(options["risk_assessments"])
        return self._observation(), self._info()

    def _target_weights(self, action: NDArray[np.float64]) -> NDArray[np.float64]:
        """Clip the policy action to the per-instrument and gross leverage caps."""
        clipped = np.clip(action, -self._max_instrument_weight, self._max_instrument_weight)
        gross = float(np.sum(np.abs(clipped)))
        if gross > self._max_gross_leverage > 0.0:
            clipped = clipped * (self._max_gross_leverage / gross)
        return clipped

    def _margin_requirement(
        self, weights: NDArray[np.float64], equity: float, prices: NDArray[np.float64]
    ) -> tuple[float, float]:
        """Projected ``(initial_margin, maintenance_margin)`` for target weights."""
        contracts = np.abs(weights) * equity / (prices * self._multipliers)
        initial = float(np.sum(contracts * self._initial_margins))
        maintenance = float(np.sum(contracts * self._maintenance_margins))
        return max(initial, 0.0), max(maintenance, 0.0)

    def step(
        self,
        action: NDArray[np.float32],
        *,
        risk_assessments: Sequence[RiskAssessment]
        | Mapping[InstrumentId, RiskAssessment]
        | None = None,
    ) -> tuple[NDArray[np.float64], float, bool, bool, dict[str, Any]]:
        """Rebalance to ``action``, then mark the book to the next bar."""
        if self._finished:
            raise RuntimeError("step() called after the episode ended; call reset() first")
        if risk_assessments is not None:
            self.set_risk_assessments(risk_assessments)

        prices_now = self._prices[self._index]
        prices_next = self._prices[self._index + 1]
        equity_before = self._equity
        previous_weights = self._weights
        target = self._target_weights(np.asarray(action, dtype=np.float64).reshape(-1))
        if target.size != self._asset_count:
            raise ValueError(
                f"action has {target.size} entries but the environment has {self._asset_count} assets"
            )

        # -- rebalance costs ------------------------------------------------
        traded = np.abs(target - previous_weights)
        turnover = float(np.sum(traded))
        commission = self._commission_rate * turnover * equity_before
        slippage = float(
            np.sum(traded * equity_before * self._slippage_ticks * self._tick_sizes / prices_now)
        )
        equity_after_costs = equity_before - commission - slippage

        # -- margin call: scale the target into the utilisation budget ------
        margin_required, maintenance_required = self._margin_requirement(
            target, equity_after_costs, prices_now
        )
        margin_utilisation = (
            margin_required / equity_after_costs if equity_after_costs > 0.0 else 0.0
        )
        margin_call = False
        margin_call_cost = 0.0
        if margin_utilisation > self._max_margin_utilization:
            scale = self._max_margin_utilization / margin_utilisation
            reduced = np.abs(target) * (1.0 - scale)
            target = target * scale
            margin_call_cost = (
                self._margin_call_penalty_rate * float(np.sum(reduced)) * equity_after_costs
            )
            equity_after_costs -= margin_call_cost
            margin_call = True
            margin_required, maintenance_required = self._margin_requirement(
                target, equity_after_costs, prices_now
            )
            margin_utilisation = (
                margin_required / equity_after_costs if equity_after_costs > 0.0 else 0.0
            )

        # -- mark to market --------------------------------------------------
        price_returns = prices_next / prices_now - 1.0
        gross_pnl = float(equity_after_costs * float(np.dot(target, price_returns)))
        equity_next = equity_after_costs + gross_pnl

        # -- forced liquidation on a maintenance-margin breach --------------
        forced_liquidation = False
        liquidation_cost = 0.0
        liquidated_notional = 0.0
        if maintenance_required > 0.0 and equity_next < maintenance_required:
            forced_liquidation = True
            liquidated_notional = float(np.sum(np.abs(target))) * equity_after_costs
            liquidation_cost = self._forced_liquidation_penalty_rate * liquidated_notional
            equity_next -= liquidation_cost
            target = np.zeros(self._asset_count, dtype=np.float64)

        total_costs = commission + slippage + margin_call_cost + liquidation_cost
        net_pnl = gross_pnl - total_costs
        equity_ratio_change = net_pnl / equity_before if equity_before > 0.0 else 0.0

        # -- FinRL Risk-First Module 2 reward shaping -----------------------
        risk_penalty = semantic_risk_penalty(
            self._instrument_ids,
            target,
            self._risk_assessments,
            risk_penalty_lambda=self._risk_penalty_lambda,
            confidence_threshold=self._confidence_threshold,
            safe_risk_score=self._safe_risk_score,
        )
        reward = equity_ratio_change * self._reward_scaling - risk_penalty

        # -- bookkeeping / termination --------------------------------------
        self._equity = float(equity_next)
        self._weights = target
        self._index += 1
        self._peak_equity = max(self._peak_equity, self._equity)
        drawdown = 1.0 - self._equity / self._peak_equity if self._peak_equity > 0.0 else 0.0
        bankrupt = self._equity <= self._initial_equity * self._bankruptcy_threshold
        terminated = bool(forced_liquidation or bankrupt or drawdown >= self._drawdown_limit)
        truncated = bool(self._index >= self._step_count - 1) and not terminated
        if terminated or truncated:
            self._finished = True

        info = self._info()
        info.update(
            {
                "gross_pnl": gross_pnl,
                "net_pnl": net_pnl,
                "transaction_costs": commission,
                "slippage_costs": slippage,
                "turnover": turnover,
                "margin_required": margin_required,
                "maintenance_margin": maintenance_required,
                "margin_utilization": float(margin_utilisation),
                "risk_penalty": float(risk_penalty),
                "margin_call": margin_call,
                "forced_liquidation": forced_liquidation,
                "bankrupt": bool(bankrupt),
                "liquidated_notional": float(liquidated_notional),
            }
        )
        return self._observation(), float(reward), terminated, truncated, info


__all__ = ["DEFAULT_BANKRUPTCY_THRESHOLD", "DEFAULT_OBSERVATION_FEATURES", "FuturesTradingEnv"]
