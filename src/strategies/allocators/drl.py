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

import json
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from configs.strategy import StageParam
from numpy.typing import NDArray

from src.core.interfaces import PortfolioAllocator
from src.core.types import InstrumentId, MarketState, RiskAssessment, WeightVector
from src.strategies._observation import (
    DEFAULT_OBSERVATION_FEATURES,
    build_observation,
    feature_matrix,
    observation_from_components,
)
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


#: Sidecar suffix written by the trainer next to the SB3 checkpoint (``.zip.meta.json``).
CHECKPOINT_META_SUFFIX: str = ".meta.json"


@dataclass(frozen=True, slots=True)
class DRLCheckpointMetadata:
    """Parsed ``<checkpoint>.zip.meta.json`` sidecar written by the DRL trainer.

    The trainer (``python -m src.strategies.train_drl`` / ``POST /api/drl/train``)
    writes this sidecar so inference can prove a checkpoint and the live pipeline
    agree on the observation layout. The two hard invariants are ``universe`` (the
    per-row root order) and ``feature_names`` (the per-row feature order); shapes
    are recorded for actionable mismatch errors and bind-preview reporting.
    """

    model_path: Path
    meta_path: Path
    universe: tuple[InstrumentId, ...]
    feature_names: tuple[str, ...]
    observation_shape: tuple[int, ...] = ()
    action_shape: tuple[int, ...] = ()
    algorithm: str | None = None
    total_timesteps: int | None = None
    strategy_id: str | None = None
    trained_at: str | None = None

    @property
    def observation_dimension(self) -> int:
        """Flat observation length implied by the metadata (features+weights+equity)."""
        if self.observation_shape:
            return self.observation_shape[0]
        return len(self.universe) * len(self.feature_names) + len(self.universe) + 1

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable representation (sidecar-compatible keys)."""
        return {
            "model_path": str(self.model_path),
            "meta_path": str(self.meta_path),
            "universe": list(self.universe),
            "feature_names": list(self.feature_names),
            "observation_shape": list(self.observation_shape),
            "action_shape": list(self.action_shape),
            "algorithm": self.algorithm,
            "total_timesteps": self.total_timesteps,
            "strategy_id": self.strategy_id,
            "trained_at": self.trained_at,
        }


def checkpoint_meta_path(model_path: str | Path) -> Path:
    """Path of the metadata sidecar for ``model_path`` (``<model>.zip.meta.json``)."""
    return Path(f"{Path(model_path)}{CHECKPOINT_META_SUFFIX}")


def _meta_str_list(
    payload: Mapping[str, object], key: str, meta_path: Path, *, upper: bool
) -> tuple[str, ...]:
    raw = payload.get(key)
    if not isinstance(raw, list) or not raw:
        raise ValueError(
            f"checkpoint metadata {meta_path}: {key!r} must be a non-empty list of strings"
        )
    normalised: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                f"checkpoint metadata {meta_path}: {key!r} entries must be non-empty strings, "
                f"got {item!r}"
            )
        token = item.strip()
        if upper:
            token = token.upper()
        if token not in normalised:
            normalised.append(token)
    return tuple(normalised)


def _meta_shape(payload: Mapping[str, object], key: str, meta_path: Path) -> tuple[int, ...]:
    raw = payload.get(key)
    if raw is None or raw == []:
        return ()
    if not isinstance(raw, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) and item > 0 for item in raw
    ):
        raise ValueError(
            f"checkpoint metadata {meta_path}: {key!r} must be a list of positive ints, got {raw!r}"
        )
    return tuple(raw)


def load_checkpoint_metadata(model_path: str | Path) -> DRLCheckpointMetadata:
    """Read and validate the ``<checkpoint>.zip.meta.json`` sidecar.

    Raises ``FileNotFoundError`` when the sidecar is absent (legacy checkpoint
    fallback is handled by :class:`DRLAllocator` with a warning) and ``ValueError``
    with the sidecar path when it is malformed or missing the universe/features
    needed for observation alignment. Importing/using this helper does not import
    torch or Stable-Baselines3.
    """
    path = Path(model_path)
    meta_path = checkpoint_meta_path(path)
    if not meta_path.is_file():
        raise FileNotFoundError(f"DRL checkpoint metadata sidecar not found: {meta_path}")
    try:
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"checkpoint metadata {meta_path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError(f"checkpoint metadata {meta_path} must be a JSON object")
    universe = _meta_str_list(raw, "universe", meta_path, upper=True)
    feature_names = _meta_str_list(raw, "feature_names", meta_path, upper=False)
    algorithm = raw.get("algorithm")
    total_timesteps = raw.get("total_timesteps")
    strategy_id = raw.get("strategy_id")
    trained_at = raw.get("trained_at")
    return DRLCheckpointMetadata(
        model_path=path,
        meta_path=meta_path,
        universe=universe,
        feature_names=feature_names,
        observation_shape=_meta_shape(raw, "observation_shape", meta_path),
        action_shape=_meta_shape(raw, "action_shape", meta_path),
        algorithm=algorithm if isinstance(algorithm, str) else None,
        total_timesteps=(
            int(total_timesteps)
            if isinstance(total_timesteps, int) and not isinstance(total_timesteps, bool)
            else None
        ),
        strategy_id=strategy_id if isinstance(strategy_id, str) else None,
        trained_at=trained_at if isinstance(trained_at, str) else None,
    )


def _missing_metadata_warning(model_path: Path, meta_path: Path) -> None:
    warnings.warn(
        f"DRL checkpoint {model_path} has no metadata sidecar {meta_path}; "
        "universe/feature observation alignment is disabled and the policy receives "
        "candidates in selector order (legacy behaviour). Retrain the checkpoint with "
        "`python -m src.strategies.train_drl` (or POST /api/drl/train) to write the sidecar.",
        UserWarning,
        stacklevel=3,
    )


def checkpoint_universe_bind_patch(
    metadata: DRLCheckpointMetadata | str | Path,
) -> dict[str, object]:
    """Universe patch for the Studio "Bind" flow: ``{"roots": [...], "top_n": N}``.

    ``roots`` follows the checkpoint universe order; callers that persist a
    strategy YAML/API stage must encode it as a comma-separated string because
    ``configs.strategy.StageParam`` is scalar-only (see
    :func:`checkpoint_strategy_params_patch`).
    """
    meta = (
        metadata
        if isinstance(metadata, DRLCheckpointMetadata)
        else load_checkpoint_metadata(metadata)
    )
    if not meta.universe:
        raise RuntimeError(
            f"DRL checkpoint {meta.model_path} metadata has an empty universe; "
            "cannot bind a universe patch"
        )
    return {"roots": list(meta.universe), "top_n": len(meta.universe)}


def checkpoint_strategy_params_patch(
    metadata: DRLCheckpointMetadata | str | Path,
) -> dict[str, object]:
    """PUT-ready ``universe.params`` scalar patch (``roots`` comma string, ``top_n``)."""
    patch = checkpoint_universe_bind_patch(metadata)
    roots = patch["roots"]
    assert isinstance(roots, list)
    return {"roots": ",".join(str(root) for root in roots), "top_n": patch["top_n"]}


def checkpoint_bind_preview(model_path: str | Path) -> dict[str, object]:
    """Non-raising Bind descriptor for a checkpoint (API/frontend friendly).

    Returns ``metadata_available=False`` plus a ``warning`` and ``metadata_error``
    (with ``roots_patch``/``params_patch`` set to ``None``) when the sidecar is
    missing or malformed, so the UI can show "Universe synced (N roots)" vs the
    metadata-unavailable warning without special-casing exceptions.
    """
    path = Path(model_path)
    meta_path = checkpoint_meta_path(path)
    checkpoint_name = path.name[: -len(".zip")] if path.name.endswith(".zip") else path.name
    preview: dict[str, object] = {
        "checkpoint_name": checkpoint_name,
        "model_path": str(path),
        "meta_path": str(meta_path),
        "metadata_available": False,
        "metadata_error": None,
        "universe": [],
        "feature_names": [],
        "observation_shape": [],
        "action_shape": [],
        "roots_patch": None,
        "params_patch": None,
        "warning": None,
    }
    try:
        meta = load_checkpoint_metadata(path)
    except FileNotFoundError as exc:
        preview["metadata_error"] = str(exc)
        preview["warning"] = (
            "Checkpoint metadata is unavailable; the universe cannot be synced on Bind. "
            "Retrain the checkpoint to write the sidecar."
        )
        return preview
    except ValueError as exc:
        preview["metadata_error"] = str(exc)
        preview["warning"] = f"Checkpoint metadata is invalid: {exc}"
        return preview
    preview.update(
        {
            "metadata_available": True,
            "universe": list(meta.universe),
            "feature_names": list(meta.feature_names),
            "observation_shape": list(meta.observation_shape),
            "action_shape": list(meta.action_shape),
            "roots_patch": checkpoint_universe_bind_patch(meta),
            "params_patch": checkpoint_strategy_params_patch(meta),
        }
    )
    return preview


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

    Checkpoints
    -----------
    ``model_path`` points at a Stable-Baselines3 ``.zip`` checkpoint produced by
    ``python -m src.strategies.train_drl`` (or any ``model.save()`` call). The
    checkpoint is loaded lazily on the first :meth:`allocate`/:attr:`model` access
    via ``algorithm_class.load(path, device=self._device)`` (a single attempt whose
    result - success or failure - is cached), so importing/building a pipeline never
    imports torch and a bad path fails with an actionable error at the point of use.
    The configured ``algorithm`` must match the checkpoint's algorithm (``CPPO``
    checkpoints are plain ``PPO`` and therefore load as ``PPO``).
    """

    __slots__ = (
        "_action_mode",
        "_algorithm",
        "_checkpoint_error",
        "_confidence_threshold",
        "_device",
        "_feature_names",
        "_gamma",
        "_gross_leverage",
        "_learning_rate",
        "_max_instrument_weight",
        "_metadata",
        "_metadata_loaded",
        "_model",
        "_model_path",
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
        model_path: str | Path | None = None,
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
        if model_path is not None and not isinstance(model_path, (str, Path)):
            raise TypeError("model_path must be a str, pathlib.Path or None")
        if isinstance(model_path, str) and not model_path.strip():
            raise ValueError("model_path must be a non-empty path when provided")
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
        self._model_path: Path | None = None if model_path is None else Path(model_path)
        self._checkpoint_error: Exception | None = None
        self._metadata: DRLCheckpointMetadata | None = None
        self._metadata_loaded = False
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
    def checkpoint_metadata(self) -> DRLCheckpointMetadata | None:
        """Sidecar metadata for the configured checkpoint (``None`` when absent/legacy).

        Missing metadata is a warning fallback (legacy candidate-order behaviour);
        malformed metadata raises an actionable ``RuntimeError`` naming the sidecar.
        """
        return self._load_metadata_once()

    @property
    def expected_roots(self) -> tuple[InstrumentId, ...]:
        """Root order the checkpoint was trained on (``()`` without metadata)."""
        meta = self._load_metadata_once()
        return () if meta is None else meta.universe

    @property
    def expected_feature_names(self) -> tuple[str, ...]:
        """Feature order the checkpoint expects (configured order without metadata)."""
        meta = self._load_metadata_once()
        return self._feature_names if meta is None else meta.feature_names

    def universe_bind_patch(self) -> dict[str, object]:
        """``{"roots": [...], "top_n": N}`` patch from the checkpoint metadata.

        Raises ``RuntimeError`` when the checkpoint has no usable metadata (the
        caller should surface the metadata-unavailable warning instead of binding).
        """
        meta = self._load_metadata_once()
        if meta is None:
            raise RuntimeError(
                f"DRL checkpoint {self.model_path!r} has no metadata sidecar; cannot derive "
                "a universe bind patch (retrain to write <checkpoint>.zip.meta.json)"
            )
        return checkpoint_universe_bind_patch(meta)

    @property
    def total_timesteps(self) -> int:
        """Configured training budget."""
        return self._total_timesteps

    @property
    def device(self) -> str:
        """Stable-Baselines3 device used for training (``cpu`` by default)."""
        return self._device

    @property
    def model_path(self) -> str | None:
        """Configured SB3 checkpoint path (``None`` when no checkpoint is configured)."""
        return None if self._model_path is None else str(self._model_path)

    def _require_model(self) -> Any:
        """Return the attached/trained/loaded model, loading a checkpoint lazily.

        A configured ``model_path`` is loaded at most once: the successful model (or
        the failure) is cached, so repeated inference never re-reads the checkpoint
        and never silently changes behaviour mid-run.
        """
        if self._model is not None:
            return self._model
        if self._model_path is not None:
            if self._checkpoint_error is not None:
                raise self._checkpoint_error
            try:
                self._model = self._load_checkpoint(self._model_path)
            except Exception as exc:
                self._checkpoint_error = exc
                raise
            return self._model
        raise RuntimeError(
            "DRLAllocator has no model: call train(env, total_timesteps) first, pass a "
            "trained Stable-Baselines3 model as `model=`, or point `model_path=` at a "
            "checkpoint produced by `python -m src.strategies.train_drl`."
        )

    def _load_checkpoint(self, path: Path) -> Any:
        """Load one SB3 checkpoint from ``path`` (fails with actionable errors)."""
        if not path.is_file():
            raise RuntimeError(
                f"DRL checkpoint not found: {path}; train one with "
                f"`python -m src.strategies.train_drl --output {path}` or attach an "
                "existing model via attach_model()."
            )
        framework = import_stable_baselines3()
        algorithm_class = self._resolve_algorithm_class(framework)
        try:
            return algorithm_class.load(str(path), device=self._device)
        except Exception as exc:
            raise RuntimeError(
                f"failed to load DRL checkpoint {path} as {self._algorithm}: {exc}. "
                "Ensure the .zip was saved by the same Stable-Baselines3 algorithm "
                "(CPPO checkpoints are plain PPO)."
            ) from exc

    def attach_model(self, model: Any) -> None:
        """Attach an externally trained SB3-compatible model."""
        if model is None:
            raise ValueError("model must not be None")
        self._model = model
        self._checkpoint_error = None

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
        self._checkpoint_error = None
        return model

    # -- inference ------------------------------------------------------------

    def _load_metadata_once(self) -> DRLCheckpointMetadata | None:
        """Load/cache the checkpoint sidecar; missing sidecar = warning fallback."""
        if self._metadata_loaded:
            return self._metadata
        self._metadata_loaded = True
        if self._model_path is None:
            self._metadata = None
            return None
        meta_path = checkpoint_meta_path(self._model_path)
        if not meta_path.is_file():
            _missing_metadata_warning(self._model_path, meta_path)
            self._metadata = None
            return None
        try:
            self._metadata = load_checkpoint_metadata(self._model_path)
        except ValueError as exc:
            raise RuntimeError(
                f"invalid DRL checkpoint metadata for {self._model_path}: {exc}"
            ) from exc
        return self._metadata

    def _build_aligned_observation(
        self,
        state: MarketState,
        candidates: Sequence[InstrumentId],
        *,
        metadata: DRLCheckpointMetadata,
        current_weights: Mapping[InstrumentId, float] | None,
        equity_ratio: float,
    ) -> NDArray[np.float64]:
        """Observation in the checkpoint's root/feature order (missing roots flat).

        Expected roots that the selector did not pass as candidates (or that are
        absent from the state) contribute a zero/flat row: zero features, zero held
        weight. Extra candidates outside the checkpoint universe are ignored. The
        observation width is therefore always the checkpoint's trained width.
        """
        candidate_set = set(candidates)
        expected = metadata.universe
        features = feature_matrix(state, expected, metadata.feature_names)
        held = np.zeros(len(expected), dtype=np.float64)
        for row, root in enumerate(expected):
            if root not in candidate_set or root not in state.instruments:
                features[row, :] = 0.0
                continue
            held[row] = 0.0 if current_weights is None else float(current_weights.get(root, 0.0))
        return observation_from_components(features, held, equity_ratio=equity_ratio)

    def _validate_observation_size(
        self,
        observation: NDArray[np.float64],
        metadata: DRLCheckpointMetadata | None,
    ) -> None:
        """Raise an actionable error when metadata/model/observation disagree."""
        size = int(observation.size)
        expected_from_meta = None if metadata is None else metadata.observation_dimension
        model = self._model
        space = None if model is None else getattr(model, "observation_space", None)
        shape = None if space is None else getattr(space, "shape", None)
        model_size = (
            int(shape[0])
            if isinstance(shape, tuple) and len(shape) == 1 and isinstance(shape[0], int)
            else None
        )
        if expected_from_meta is not None and expected_from_meta != size:
            assert metadata is not None
            raise RuntimeError(
                "DRL checkpoint observation mismatch: metadata "
                f"{self._model_path} declares observation length {expected_from_meta} "
                f"(universe={len(metadata.universe)} roots x {len(metadata.feature_names)} "
                f"features + weights + equity) but the aligned observation has {size} entries; "
                "retrain the checkpoint with the current feature set or inspect the sidecar."
            )
        if model_size is not None and model_size != size:
            detail = (
                f" (checkpoint metadata aligns to {expected_from_meta}, built {size})"
                if expected_from_meta is not None
                else f" (built {size})"
            )
            raise RuntimeError(
                "DRL policy observation-space mismatch: the loaded policy expects observation "
                f"length {model_size} but the observation has {size} entries{detail}; the "
                "checkpoint feature_names/universe do not match the current feature set."
            )

    def build_observation(
        self,
        state: MarketState,
        candidates: Sequence[InstrumentId],
        *,
        current_weights: Mapping[InstrumentId, float] | None = None,
        equity_ratio: float = 1.0,
    ) -> NDArray[np.float64]:
        """Observation vector layout shared with ``FuturesTradingEnv``.

        With checkpoint metadata the vector is built in the checkpoint's universe
        and feature order (missing/inactive roots flat, extra candidates ignored);
        without metadata the legacy candidate-order layout is preserved.
        """
        metadata = self._load_metadata_once()
        if metadata is None:
            return build_observation(
                state,
                candidates,
                feature_names=self._feature_names,
                current_weights=current_weights,
                equity_ratio=equity_ratio,
            )
        return self._build_aligned_observation(
            state,
            candidates,
            metadata=metadata,
            current_weights=current_weights,
            equity_ratio=equity_ratio,
        )

    def weights_from_action(
        self,
        candidates: Sequence[InstrumentId],
        action: object,
        *,
        action_candidates: Sequence[InstrumentId] | None = None,
    ) -> WeightVector:
        """Map a raw policy action to a validated target weight vector.

        ``action_candidates`` is the ordered universe the action vector refers to
        (the checkpoint universe). When it differs from ``candidates`` the action is
        remapped: expected roots feed their actual candidate weight, extra
        candidates stay flat (0), and expected roots the selector dropped are
        discarded. Gross-target semantics are unchanged (clip + leverage scaling
        apply to the mapped actual-candidate weights).
        """
        array = np.asarray(action, dtype=np.float64).reshape(-1)
        expected = tuple(candidates) if action_candidates is None else tuple(action_candidates)
        if array.size != len(expected):
            if action_candidates is None:
                raise ValueError(
                    f"policy action has {array.size} entries but {len(candidates)} candidates "
                    "were supplied; the policy was trained on a different universe"
                )
            raise ValueError(
                f"policy action has {array.size} entries but the checkpoint expects "
                f"{len(expected)} roots {list(expected)!r}; the checkpoint universe and the "
                "loaded policy/action shape disagree"
            )
        if not np.all(np.isfinite(array)):
            raise RuntimeError("policy returned a non-finite action vector")
        if action_candidates is not None and expected != tuple(candidates):
            mapped = np.zeros(len(candidates), dtype=np.float64)
            index = {root: position for position, root in enumerate(candidates)}
            for position, root in enumerate(expected):
                actual = index.get(root)
                if actual is not None:
                    mapped[actual] = array[position]
            array = mapped
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
        metadata = self._load_metadata_once()
        observation = self.build_observation(state, candidates)
        self._validate_observation_size(observation, metadata)
        prediction = model.predict(observation, deterministic=True)
        action = prediction[0] if isinstance(prediction, tuple) else prediction
        return self.weights_from_action(
            candidates,
            action,
            action_candidates=None if metadata is None else metadata.universe,
        )

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
        model_path=reader.as_optional_str("model_path"),
    )
    reader.check()
    return allocator


__all__ = [
    "ACTION_MODES",
    "CHECKPOINT_META_SUFFIX",
    "SUPPORTED_ALGORITHMS",
    "DRLAllocator",
    "DRLCheckpointMetadata",
    "build_drl_allocator",
    "checkpoint_bind_preview",
    "checkpoint_meta_path",
    "checkpoint_strategy_params_patch",
    "checkpoint_universe_bind_patch",
    "import_stable_baselines3",
    "load_checkpoint_metadata",
]
