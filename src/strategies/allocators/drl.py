"""Deep-reinforcement-learning allocator (``A_t``) — PPO / CPPO / SAC / TD3 / A2C.

:class:`DRLAllocator` is the bridge between the FinRL-X pipeline and a
Stable-Baselines3 policy trained on
:class:`~src.strategies.envs.FuturesTradingEnv`:

* ``train(env, total_timesteps)`` builds the SB3 model for the configured
  algorithm and runs ``learn`` (plus the walk-forward driver exposed by the
  environment's price matrix);
* ``allocate(candidates, state)`` runs ``model.predict(observation,
  deterministic=True)`` against a live/backtest :class:`MarketState` and maps the
  continuous action (target weights) to a :class:`WeightVector`;
* :meth:`DRLAllocator.semantic_risk_penalty` exposes the FinRL Risk-First
  "Module 2" Lagrangian penalty so a constrained trainer (CPPO / CPRL) can shape
  rewards outside the environment.

Heavy dependencies (``stable-baselines3`` and therefore ``torch``) are imported
**lazily**: importing :mod:`src.strategies` never imports torch, and a missing
install raises a :class:`RuntimeError` with the exact install command instead of
an ``ImportError`` traceback from deep inside a training loop.

``CPPO`` (Constrained PPO) is not shipped by Stable-Baselines3 itself; the
FuturesRL-X implementation is PPO trained against the semantic-risk-shaped
reward implemented by the environment (Module 2 hook), which is mathematically
the Lagrangian relaxation of the constrained objective. That is stated here
rather than hidden behind an import alias.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from configs.strategy import StageParam
from numpy.typing import NDArray

from src.core.interfaces import PortfolioAllocator
from src.core.types import InstrumentId, MarketState, RiskAssessment, WeightVector
from src.strategies._observation import DEFAULT_OBSERVATION_FEATURES, build_observation
from src.strategies._params import ParamReader
from src.strategies._risk import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_SAFE_RISK_SCORE,
    semantic_risk_penalty_mapping,
)

#: Algorithms supported by the allocator (``CPPO`` is PPO + reward shaping).
SUPPORTED_ALGORITHMS: tuple[str, ...] = ("PPO", "CPPO", "SAC", "TD3", "A2C")

#: Action interpretation modes.
ACTION_MODES: frozenset[str] = frozenset({"weights", "normalized"})

_INSTALL_HINT = (
    "DRLAllocator requires the optional deep-RL stack. Install it with "
    "`pip install 'stable-baselines3>=2.3' 'gymnasium>=1.0' torch` "
    "(or `pip install -e '.[drl]'`)."
)


def import_stable_baselines3() -> Any:
    """Import Stable-Baselines3 lazily, raising an actionable ``RuntimeError``.

    Returning ``Any`` is deliberate: the SB3 package either is not installed (in
    which case this raises) or is dynamically typed for our purposes, and we do
    not want a hard import of torch anywhere in the strategy package.
    """
    # The lazy import is intentional: importing torch at package import time
    # would make every backtest and API process pay the DRL dependency cost.
    try:
        import stable_baselines3
    except ImportError as exc:
        raise RuntimeError(_INSTALL_HINT) from exc
    return stable_baselines3


class DRLAllocator:
    """Wrap a Stable-Baselines3 futures policy as a pipeline allocator.

    Training runs on ``device="cpu"`` by default: the FinRL-X futures policies
    are small MLPs whose per-step overhead on GPU exceeds the compute, and an
    explicit CPU default avoids Stable-Baselines3's "auto device / no CUDA"
    warning in headless backtest and API processes. Pass ``device="auto"`` or
    ``device="cuda"`` to opt into GPU training or auto-detection.
    """

    __slots__ = (
        "_action_mode",
        "_algorithm",
        "_confidence_threshold",
        "_device",
        "_feature_names",
        "_gamma",
        "_gross_leverage",
        "_learning_rate",
        "_max_instrument_weight",
        "_model",
        "_policy",
        "_risk_penalty_lambda",
        "_safe_risk_score",
        "_seed",
        "_total_timesteps",
        "_verbose",
    )

    def __init__(
        self,
        *,
        algorithm: str = "PPO",
        policy: str = "MlpPolicy",
        total_timesteps: int = 100_000,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        seed: int = 7,
        risk_penalty_lambda: float = 0.5,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        safe_risk_score: float = DEFAULT_SAFE_RISK_SCORE,
        action_mode: str = "weights",
        gross_leverage: float = 1.0,
        max_instrument_weight: float = 1.0,
        feature_names: Sequence[str] = DEFAULT_OBSERVATION_FEATURES,
        verbose: int = 0,
        device: str = "cpu",
        model: Any = None,
    ) -> None:
        normalised = algorithm.strip().upper()
        if normalised not in SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"unsupported DRL algorithm {algorithm!r}; "
                f"supported algorithms: {list(SUPPORTED_ALGORITHMS)}"
            )
        if not policy.strip():
            raise ValueError("policy must be a non-empty Stable-Baselines3 policy name")
        if total_timesteps < 1:
            raise ValueError("total_timesteps must be positive")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be strictly positive")
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must be within (0, 1]")
        if risk_penalty_lambda < 0.0:
            raise ValueError("risk_penalty_lambda must be non-negative")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be within [0, 1]")
        if safe_risk_score < 1.0:
            raise ValueError("safe_risk_score must be at least 1.0")
        mode = action_mode.strip().lower()
        if mode not in ACTION_MODES:
            raise ValueError(
                f"action_mode must be one of {sorted(ACTION_MODES)}, got {action_mode!r}"
            )
        if gross_leverage <= 0.0:
            raise ValueError("gross_leverage must be strictly positive")
        if max_instrument_weight <= 0.0:
            raise ValueError("max_instrument_weight must be strictly positive")
        if not feature_names:
            raise ValueError("feature_names must not be empty")
        if not device.strip():
            raise ValueError("device must be a non-empty Stable-Baselines3 device string")
        self._algorithm = normalised
        self._policy = policy.strip()
        self._total_timesteps = int(total_timesteps)
        self._learning_rate = float(learning_rate)
        self._gamma = float(gamma)
        self._seed = int(seed)
        self._risk_penalty_lambda = float(risk_penalty_lambda)
        self._confidence_threshold = float(confidence_threshold)
        self._safe_risk_score = float(safe_risk_score)
        self._action_mode = mode
        self._gross_leverage = float(gross_leverage)
        self._max_instrument_weight = float(max_instrument_weight)
        self._feature_names = tuple(feature_names)
        self._verbose = int(verbose)
        self._device = device.strip()
        self._model: Any = model

    # -- introspection --------------------------------------------------------

    @property
    def name(self) -> str:
        """Registry name of this allocator."""
        return "drl"

    @property
    def algorithm(self) -> str:
        """Configured SB3 algorithm identifier."""
        return self._algorithm

    @property
    def is_trained(self) -> bool:
        """Whether a model has been attached or trained."""
        return self._model is not None

    @property
    def model(self) -> Any:
        """The wrapped SB3 model, raising a clear error when none is attached."""
        return self._require_model()

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Per-instrument observation features expected by the policy."""
        return self._feature_names

    @property
    def total_timesteps(self) -> int:
        """Configured training budget."""
        return self._total_timesteps

    @property
    def device(self) -> str:
        """Stable-Baselines3 device used for training (``cpu`` by default)."""
        return self._device

    def _require_model(self) -> Any:
        if self._model is None:
            raise RuntimeError(
                "DRLAllocator has no model: call train(env, total_timesteps) first, or "
                "pass a trained Stable-Baselines3 model as `model=`."
            )
        return self._model

    def attach_model(self, model: Any) -> None:
        """Attach an externally trained SB3-compatible model."""
        if model is None:
            raise ValueError("model must not be None")
        self._model = model

    # -- training -------------------------------------------------------------

    def _resolve_algorithm_class(self, framework: Any) -> Any:
        if self._algorithm == "CPPO":
            # Constrained PPO: PPO against the semantic-risk-shaped reward.
            return framework.PPO
        return getattr(framework, self._algorithm)

    def train(self, env: Any, total_timesteps: int | None = None) -> Any:
        """Train the configured SB3 algorithm on ``env`` and return the model."""
        budget = self._total_timesteps if total_timesteps is None else int(total_timesteps)
        if budget < 1:
            raise ValueError("total_timesteps must be positive")
        framework = import_stable_baselines3()
        algorithm_class = self._resolve_algorithm_class(framework)
        model = algorithm_class(
            self._policy,
            env,
            learning_rate=self._learning_rate,
            gamma=self._gamma,
            seed=self._seed,
            verbose=self._verbose,
            device=self._device,
        )
        model.learn(total_timesteps=budget)
        self._model = model
        return model

    # -- inference ------------------------------------------------------------

    def build_observation(
        self,
        state: MarketState,
        candidates: Sequence[InstrumentId],
        *,
        current_weights: Mapping[InstrumentId, float] | None = None,
        equity_ratio: float = 1.0,
    ) -> NDArray[np.float64]:
        """Observation vector layout shared with ``FuturesTradingEnv``."""
        return build_observation(
            state,
            candidates,
            feature_names=self._feature_names,
            current_weights=current_weights,
            equity_ratio=equity_ratio,
        )

    def weights_from_action(
        self, candidates: Sequence[InstrumentId], action: object
    ) -> WeightVector:
        """Map a raw policy action to a validated target weight vector."""
        array = np.asarray(action, dtype=np.float64).reshape(-1)
        if array.size != len(candidates):
            raise ValueError(
                f"policy action has {array.size} entries but {len(candidates)} candidates "
                "were supplied; the policy was trained on a different universe"
            )
        if not np.all(np.isfinite(array)):
            raise RuntimeError("policy returned a non-finite action vector")
        weights = np.clip(array, -self._max_instrument_weight, self._max_instrument_weight)
        gross = float(np.sum(np.abs(weights)))
        if self._action_mode == "normalized":
            if gross <= 1e-12:
                return WeightVector(dict.fromkeys(candidates, 0.0))
            weights = weights * (self._gross_leverage / gross)
        elif gross > self._gross_leverage:
            weights = weights * (self._gross_leverage / gross)
        return WeightVector.from_array(candidates, weights)

    def allocate(self, candidates: Sequence[InstrumentId], state: MarketState) -> WeightVector:
        """Run the trained policy deterministically and return target weights."""
        if not candidates:
            return WeightVector.empty()
        model = self._require_model()
        observation = self.build_observation(state, candidates)
        prediction = model.predict(observation, deterministic=True)
        action = prediction[0] if isinstance(prediction, tuple) else prediction
        return self.weights_from_action(candidates, action)

    # -- risk hook ------------------------------------------------------------

    def semantic_risk_penalty(
        self,
        weights: Mapping[InstrumentId, float],
        assessments: Sequence[RiskAssessment] | Mapping[InstrumentId, RiskAssessment] | None,
    ) -> float:
        """FinRL "Module 2" Lagrangian penalty for a weight vector.

        Exposed so CPPO/CPRL trainers that own the reward loop (outside the
        Gymnasium environment) can apply the identical anti-hallucination
        confidence filter and semantic-risk penalty.
        """
        return semantic_risk_penalty_mapping(
            weights,
            assessments,
            risk_penalty_lambda=self._risk_penalty_lambda,
            confidence_threshold=self._confidence_threshold,
            safe_risk_score=self._safe_risk_score,
        )


def build_drl_allocator(params: Mapping[str, StageParam]) -> PortfolioAllocator:
    """Build a :class:`DRLAllocator` from declarative stage parameters."""
    reader = ParamReader("allocator.drl", params)
    allocator = DRLAllocator(
        algorithm=reader.as_str("algorithm", "PPO"),
        policy=reader.as_str("policy", "MlpPolicy"),
        total_timesteps=reader.as_int("total_timesteps", 100_000),
        learning_rate=reader.as_float("learning_rate", 3e-4),
        gamma=reader.as_float("gamma", 0.99),
        seed=reader.as_int("seed", 7),
        risk_penalty_lambda=reader.as_float("risk_penalty_lambda", 0.5),
        confidence_threshold=reader.as_float("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD),
        safe_risk_score=reader.as_float("safe_risk_score", DEFAULT_SAFE_RISK_SCORE),
        action_mode=reader.as_str("action_mode", "weights"),
        gross_leverage=reader.as_float("gross_leverage", 1.0),
        max_instrument_weight=reader.as_float("max_instrument_weight", 1.0),
        verbose=reader.as_int("verbose", 0),
        device=reader.as_str("device", "cpu"),
    )
    reader.check()
    return allocator


__all__ = [
    "ACTION_MODES",
    "SUPPORTED_ALGORITHMS",
    "DRLAllocator",
    "build_drl_allocator",
    "import_stable_baselines3",
]
