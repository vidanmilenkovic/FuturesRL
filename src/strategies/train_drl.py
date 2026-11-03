"""Train a futures DRL allocator checkpoint from deterministic offline data.

Usage
-----
.. code-block:: console

    python -m src.strategies.train_drl \
        --strategy configs/strategies/drl_ppo_llm_risk.yaml \
        --offline --timesteps 100000 \
        --output artifacts/models/drl_ppo_llm_risk_ppo.zip --json

The CLI is deliberately *offline first*: the training environment is built from
the deterministic :class:`~src.data.databento_loader.OfflineBarSource` (seeded
synthetic daily OHLCV), normalised by
:class:`~src.data.feature_processor.FeatureProcessor` (canonical FinRL-X
features), and restricted to the strategy's configured universe with
:class:`~src.data.market_state_builder.MarketStateBuilder` + the configured
universe selector. The same command therefore always produces the same training
data and never touches Databento, IBKR or any other network dependency.
``--offline`` is accepted explicitly for parity with the runner/backtest CLIs and
is currently the only supported source (recorded as ``offline: true`` in the
model metadata).

Outputs
-------
* the Stable-Baselines3 checkpoint at ``--output`` (``.zip`` is appended when the
  path does not already end with it), and
* a sidecar ``<output>.meta.json`` describing exactly how the model was produced:
  strategy id, algorithm/policy, total timesteps, seed, device, feature names,
  observation shape and finite clipping bounds, universe, offline bar count, SB3
  version, optional git commit, timestamps and training duration.

Load the checkpoint back into a strategy pipeline with
``DRLAllocator(model_path=<output>)`` or, declaratively, ``drl.model_path: <output>``
in the strategy YAML.

Algorithms
----------
``PPO``/``SAC``/``TD3``/``A2C`` map to their Stable-Baselines3 classes.
``CPPO`` (constrained PPO / CPRL) maps to PPO trained against the environment's
semantic-risk-shaped reward (the existing :class:`DRLAllocator` semantics); the
metadata records both the configured ``algorithm`` and the effective
``sb3_algorithm``.

Exit codes
----------
``0`` success; ``2`` actionable error (invalid arguments, missing strategy/DRL
section, missing Stable-Baselines3, unwritable output path, unusable offline
data). Errors are printed as ``error: ...`` on stderr without a Python traceback.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import re
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import polars as pl
from configs.instruments import InstrumentsConfig, load_instruments
from configs.strategy import DRLSpec, StrategyConfig, load_strategy_config
from gymnasium import spaces
from numpy.typing import NDArray

from src.core.types import InstrumentId
from src.data import schema as s
from src.data.databento_loader import OfflineBarSource
from src.data.feature_processor import FeatureProcessor
from src.data.market_state_builder import MarketStateBuilder
from src.strategies._observation import DEFAULT_OBSERVATION_FEATURES
from src.strategies.allocators.drl import (
    SUPPORTED_ALGORITHMS,
    DRLAllocator,
    import_stable_baselines3,
)
from src.strategies.envs import FuturesTradingEnv
from src.strategies.selectors import build_selector

#: Default strategy config trained by the CLI.
DEFAULT_STRATEGY_PATH: Path = Path("configs/strategies/drl_ppo_llm_risk.yaml")
#: Default checkpoint directory (relative to the repo root / working directory).
DEFAULT_OUTPUT_DIR: Path = Path("artifacts/models")
#: Number of synthetic daily bars generated per root for training.
DEFAULT_PERIODS: int = 512
#: Maximum number of offline training-data builds kept in the in-process LRU cache.
#: Bounded so a long-running API cannot pin memory for many master/seed/period
#: combinations (one cached 45-root frame is tens of MB).
TRAINING_DATA_CACHE_MAXSIZE: int = 4
#: Observation-feature configuration hashed into the cache key (future-proofing:
#: changing the DRL feature set must invalidate cached training data).
TRAINING_DATA_FEATURE_CONFIG: tuple[str, ...] = DEFAULT_OBSERVATION_FEATURES
#: Default SB3 device (CPU keeps the tiny MLP policies deterministic and fast).
DEFAULT_DEVICE: str = "cpu"
#: Hard upper bound for a single training run (guards CLI/API from runaway jobs).
MAX_TIMESTEPS: int = 2_000_000
#: Number of reward-curve windows recorded per run (bounded summary size).
DEFAULT_REWARD_CURVE_WINDOWS: int = 20
#: Checkpoint stem contract: safe file name, no path separators or traversal.
CHECKPOINT_NAME_PATTERN: re.Pattern[str] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
#: Checkpoint suffixes recognised when resolving/listing model artifacts.
CHECKPOINT_SUFFIX: str = ".zip"
#: Sidecar suffix for the machine-readable reward curve (next to the .meta.json).
REWARD_CURVE_SUFFIX: str = ".reward_curve.json"
#: Maximum completed-episode returns embedded in the bounded curve summary.
MAX_RECORDED_EPISODES: int = 100
#: Filesystem-style repo root, only used for the optional git commit metadata.
_REPO_ROOT: Path = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class TrainingData:
    """Aligned price/feature matrices plus the contract metadata for one universe."""

    prices: NDArray[np.float64]
    features: NDArray[np.float64]
    timestamps: tuple[datetime, ...]
    instrument_ids: tuple[InstrumentId, ...]
    feature_names: tuple[str, ...]
    feature_fallbacks: tuple[str, ...]
    multipliers: tuple[float, ...]
    initial_margins: tuple[float, ...]
    maintenance_margins: tuple[float, ...]
    tick_sizes: tuple[float, ...]
    periods: int

    @property
    def usable_rows(self) -> int:
        """Number of fully-featured rows handed to the environment."""
        return int(self.prices.shape[0])


# ---------------------------------------------------------------------------
# Bounded training-data cache (45-root offline builds are expensive)
# ---------------------------------------------------------------------------

type TrainingDataCacheKey = tuple[object, ...]

#: LRU of immutable training-data snapshots, guarded by ``_TRAINING_DATA_CACHE_LOCK``.
_TRAINING_DATA_CACHE: OrderedDict[TrainingDataCacheKey, TrainingData] = OrderedDict()
_TRAINING_DATA_CACHE_LOCK = threading.Lock()
_TRAINING_DATA_CACHE_HITS = 0
_TRAINING_DATA_CACHE_MISSES = 0
_TRAINING_DATA_CACHE_EVICTIONS = 0


def _instrument_master_fingerprint(instruments: InstrumentsConfig) -> tuple[object, ...]:
    """Hashable content fingerprint of the instrument master (roots + full specs).

    Serialising every spec means any contract change (multiplier, tick size, margins,
    roll rules, micro/standard routing, currency, root order, ...) produces a new
    cache key, so a reloaded/expanded ``configs/instruments.yaml`` can never be
    served stale cached training data (no cross-master contamination).
    """
    return (
        instruments.version,
        tuple(instruments.roots()),
        tuple(instruments.sectors()),
        tuple(spec.model_dump_json() for spec in instruments.instruments.values()),
    )


def _training_data_cache_key(
    config: StrategyConfig,
    instruments: InstrumentsConfig,
    *,
    seed: int,
    periods: int,
) -> TrainingDataCacheKey:
    """Cache identity of one ``build_training_data`` call.

    Includes every input that can change the returned matrices: the instrument
    master content, ``seed``, ``periods``, the observation-feature configuration,
    the universe selector (name + params) and the backtest initial equity used to
    build the point-in-time state the selector ranks on.
    """
    initial_equity = config.backtest.initial_equity if config.backtest is not None else 1_000_000.0
    return (
        _instrument_master_fingerprint(instruments),
        int(seed),
        int(periods),
        TRAINING_DATA_FEATURE_CONFIG,
        config.universe.name.strip().lower(),
        tuple(sorted(config.universe.params.items())),
        float(initial_equity),
    )


def _copy_training_data(data: TrainingData) -> TrainingData:
    """Return a caller-owned copy (writable arrays, immutable metadata tuples)."""
    return replace(
        data,
        prices=np.array(data.prices, dtype=np.float64, copy=True),
        features=np.array(data.features, dtype=np.float64, copy=True),
    )


def _freeze_training_data(data: TrainingData) -> TrainingData:
    """Return a cached snapshot whose numpy arrays are read-only."""
    prices = np.array(data.prices, dtype=np.float64, copy=True)
    features = np.array(data.features, dtype=np.float64, copy=True)
    prices.setflags(write=False)
    features.setflags(write=False)
    return replace(data, prices=prices, features=features)


def _training_data_cache_lookup(key: TrainingDataCacheKey) -> TrainingData | None:
    global _TRAINING_DATA_CACHE_MISSES, _TRAINING_DATA_CACHE_HITS
    with _TRAINING_DATA_CACHE_LOCK:
        cached = _TRAINING_DATA_CACHE.get(key)
        if cached is None:
            _TRAINING_DATA_CACHE_MISSES += 1
            return None
        _TRAINING_DATA_CACHE.move_to_end(key)
        _TRAINING_DATA_CACHE_HITS += 1
        return cached


def _training_data_cache_store(key: TrainingDataCacheKey, data: TrainingData) -> None:
    global _TRAINING_DATA_CACHE_EVICTIONS
    frozen = _freeze_training_data(data)
    with _TRAINING_DATA_CACHE_LOCK:
        _TRAINING_DATA_CACHE[key] = frozen
        _TRAINING_DATA_CACHE.move_to_end(key)
        while len(_TRAINING_DATA_CACHE) > TRAINING_DATA_CACHE_MAXSIZE:
            _TRAINING_DATA_CACHE.popitem(last=False)
            _TRAINING_DATA_CACHE_EVICTIONS += 1


def training_data_cache_info() -> dict[str, int]:
    """Bounded-cache telemetry: ``size``/``maxsize``/``hits``/``misses``/``evictions``."""
    with _TRAINING_DATA_CACHE_LOCK:
        return {
            "size": len(_TRAINING_DATA_CACHE),
            "maxsize": TRAINING_DATA_CACHE_MAXSIZE,
            "hits": _TRAINING_DATA_CACHE_HITS,
            "misses": _TRAINING_DATA_CACHE_MISSES,
            "evictions": _TRAINING_DATA_CACHE_EVICTIONS,
        }


def clear_training_data_cache() -> None:
    """Drop every cached build and reset the hit/miss/eviction counters (tests/API)."""
    global _TRAINING_DATA_CACHE_EVICTIONS, _TRAINING_DATA_CACHE_HITS, _TRAINING_DATA_CACHE_MISSES
    with _TRAINING_DATA_CACHE_LOCK:
        _TRAINING_DATA_CACHE.clear()
        _TRAINING_DATA_CACHE_HITS = 0
        _TRAINING_DATA_CACHE_MISSES = 0
        _TRAINING_DATA_CACHE_EVICTIONS = 0


# ---------------------------------------------------------------------------
# Argument parsing helpers
# ---------------------------------------------------------------------------


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer (>= 1)")
    return parsed


def _non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer (>= 0)")
    return parsed


def _algorithm(value: str) -> str:
    normalised = value.strip().upper()
    if normalised not in SUPPORTED_ALGORITHMS:
        raise argparse.ArgumentTypeError(
            f"unsupported algorithm {value!r}; choose from {list(SUPPORTED_ALGORITHMS)}"
        )
    return normalised


def _bounded_timesteps(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from exc
    try:
        return validate_timesteps(parsed)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _learning_rate_arg(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a float, got {value!r}") from exc
    try:
        validated = validate_learning_rate(parsed)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    assert validated is not None
    return validated


def _gamma_arg(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a float, got {value!r}") from exc
    try:
        validated = validate_gamma(parsed)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    assert validated is not None
    return validated


def _penalty_arg(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a float, got {value!r}") from exc
    try:
        validated = validate_llm_reward_penalty_lambda(parsed)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    assert validated is not None
    return validated


def _checkpoint_name_arg(value: str) -> str:
    try:
        return validate_checkpoint_name(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser (invalid arguments exit with status 2)."""
    parser = argparse.ArgumentParser(
        prog="python -m src.strategies.train_drl",
        description=(
            "Train a FinRL-X futures DRL allocator checkpoint from deterministic offline "
            "data and write the .zip checkpoint plus <output>.meta.json metadata."
        ),
    )
    parser.add_argument(
        "--strategy",
        type=Path,
        default=DEFAULT_STRATEGY_PATH,
        help=f"strategy YAML with a drl: section (default: {DEFAULT_STRATEGY_PATH})",
    )
    parser.add_argument(
        "--timesteps",
        type=_positive_int,
        default=None,
        help="training budget override (default: config.drl.total_timesteps)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "checkpoint path; .zip is appended when missing "
            "(default: artifacts/models/<strategy_id>_<algorithm>.zip)"
        ),
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="use the deterministic offline synthetic bar source (currently the only source)",
    )
    parser.add_argument(
        "--seed",
        type=_non_negative_int,
        default=None,
        help="random seed override (default: config.drl.seed)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help=f"Stable-Baselines3 device (default: {DEFAULT_DEVICE})",
    )
    parser.add_argument(
        "--algorithm",
        type=_algorithm,
        default=None,
        help="override config.drl.algorithm (PPO/CPPO/SAC/TD3/A2C)",
    )
    parser.add_argument(
        "--periods",
        type=_positive_int,
        default=DEFAULT_PERIODS,
        help=f"number of offline daily bars per root (default: {DEFAULT_PERIODS})",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=_checkpoint_name_arg,
        default=None,
        help=(
            "checkpoint stem inside --output-dir (default: <strategy_id>_<algorithm>); "
            "ignored when --output is given"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "directory for the .zip/.meta.json/reward-curve artifacts "
            f"(default: {DEFAULT_OUTPUT_DIR}; ignored when --output is given)"
        ),
    )
    parser.add_argument(
        "--learning-rate",
        type=_learning_rate_arg,
        default=None,
        help="learning-rate override (default: config.drl.learning_rate; must be in (0, 1])",
    )
    parser.add_argument(
        "--gamma",
        type=_gamma_arg,
        default=None,
        help="discount-factor override (default: config.drl.gamma; must be within [0, 1])",
    )
    parser.add_argument(
        "--llm-reward-penalty-lambda",
        type=_penalty_arg,
        default=None,
        help=(
            "FinRL semantic-risk reward-shaping lambda override "
            "(default: config.drl.reward_risk_penalty_lambda; must be >= 0)"
        ),
    )
    parser.add_argument(
        "--reward-curve-windows",
        type=_positive_int,
        default=DEFAULT_REWARD_CURVE_WINDOWS,
        help=(
            f"number of reward-curve windows to record (default: {DEFAULT_REWARD_CURVE_WINDOWS})"
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the summary as a single JSON object on stdout (machine readable)",
    )
    return parser


# ---------------------------------------------------------------------------
# Configuration / filesystem helpers
# ---------------------------------------------------------------------------


def _resolve_drl_spec(config: StrategyConfig) -> DRLSpec:
    if config.drl is None:
        raise RuntimeError(
            f"strategy {config.id!r} has no 'drl:' section; add algorithm/policy/"
            "total_timesteps/learning_rate/gamma/seed and re-run"
        )
    return config.drl


def _resolve_checkpoint_path(
    output: Path | None,
    output_dir: Path | None,
    checkpoint_name: str | None,
    config: StrategyConfig,
    algorithm: str,
) -> Path:
    """Resolve the checkpoint path (explicit ``--output`` wins over name/dir)."""
    if output is not None:
        candidate = Path(output)
    else:
        directory = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
        stem = (
            checkpoint_name if checkpoint_name is not None else f"{config.id}_{algorithm.lower()}"
        )
        candidate = directory / stem
    return candidate if candidate.suffix == CHECKPOINT_SUFFIX else Path(f"{candidate}.zip")


def _prepare_output_path(checkpoint: Path) -> Path:
    """Create the output directory and prove it is writable (actionable errors)."""
    parent = checkpoint.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"output directory is not writable: {parent} ({exc})") from exc
    if checkpoint.is_dir():
        raise RuntimeError(f"output path is a directory, expected a .zip file: {checkpoint}")
    probe = parent / f".train_drl_probe.{id(checkpoint)}"
    try:
        probe.write_text("", encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"output directory is not writable: {parent} ({exc})") from exc
    finally:
        with contextlib.suppress(OSError):
            probe.unlink()
    return checkpoint


def _git_commit() -> str | None:
    """Best-effort git commit hash for checkpoint provenance (``None`` if unknown)."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    commit = completed.stdout.strip()
    return commit or None


def _sb3_version() -> str:
    framework = import_stable_baselines3()
    return str(getattr(framework, "__version__", "unknown"))


# ---------------------------------------------------------------------------
# Parameter validation (shared by the CLI, the callable and the API service)
# ---------------------------------------------------------------------------


def normalise_algorithm(algorithm: str) -> str:
    """Validate/normalise an algorithm id against the supported SB3 mapping."""
    candidate = algorithm.strip().upper()
    if candidate not in SUPPORTED_ALGORITHMS:
        raise ValueError(
            f"unsupported algorithm {algorithm!r}; choose from {list(SUPPORTED_ALGORITHMS)}"
        )
    return candidate


def validate_timesteps(timesteps: int) -> int:
    """Validate a training budget (bounded to ``MAX_TIMESTEPS``)."""
    if isinstance(timesteps, bool) or not isinstance(timesteps, int):
        raise ValueError(f"timesteps must be an integer, got {type(timesteps)!r}")
    if timesteps < 1 or timesteps > MAX_TIMESTEPS:
        raise ValueError(f"timesteps must be within [1, {MAX_TIMESTEPS}], got {timesteps}")
    return timesteps


def validate_checkpoint_name(checkpoint_name: str) -> str:
    """Validate a checkpoint file stem (no traversal, no path separators)."""
    if not isinstance(checkpoint_name, str):
        raise ValueError(f"checkpoint_name must be a string, got {type(checkpoint_name)!r}")
    cleaned = checkpoint_name.strip()
    if not CHECKPOINT_NAME_PATTERN.fullmatch(cleaned) or ".." in cleaned:
        raise ValueError(
            "checkpoint_name must be 1-64 characters of letters, digits, '.', '_' or '-', "
            f"start with a letter/digit and contain no '..' path traversal; got {checkpoint_name!r}"
        )
    return cleaned


def validate_learning_rate(learning_rate: float | None) -> float | None:
    """Validate an optional learning-rate override (strictly positive, <= 1)."""
    if learning_rate is None:
        return None
    value = float(learning_rate)
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError(f"learning_rate must be within (0, 1], got {value!r}")
    return value


def validate_gamma(gamma: float | None) -> float | None:
    """Validate an optional discount-factor override (within ``[0, 1]``)."""
    if gamma is None:
        return None
    value = float(gamma)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"gamma must be within [0, 1], got {value!r}")
    return value


def validate_llm_reward_penalty_lambda(value: float | None) -> float | None:
    """Validate an optional FinRL semantic-risk penalty lambda (``>= 0``)."""
    if value is None:
        return None
    penalty = float(value)
    if not math.isfinite(penalty) or penalty < 0.0:
        raise ValueError(f"llm_reward_penalty_lambda must be a finite >= 0 number, got {value!r}")
    return penalty


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
    return seed


def _validate_device(device: str) -> str:
    cleaned = device.strip()
    if not cleaned:
        raise ValueError("device must be a non-empty Stable-Baselines3 device string")
    return cleaned


def _validate_periods(periods: int) -> int:
    if isinstance(periods, bool) or not isinstance(periods, int) or periods < 2:
        raise ValueError(f"periods must be an integer >= 2, got {periods!r}")
    return periods


def _validate_reward_curve_windows(windows: int, timesteps: int) -> tuple[int, int]:
    """Return ``(windows, window_steps)`` for the bounded reward-curve summary."""
    if isinstance(windows, bool) or not isinstance(windows, int) or windows < 1:
        raise ValueError(f"reward_curve_windows must be a positive integer, got {windows!r}")
    effective_windows = min(windows, timesteps)
    window_steps = max(1, timesteps // effective_windows)
    return effective_windows, window_steps


# ---------------------------------------------------------------------------
# Reward-curve capture (training-time statistics without touching the env)
# ---------------------------------------------------------------------------


class _RewardCurveRecorder(gym.Wrapper[Any, Any, Any, Any]):
    """Transparent env wrapper recording a bounded reward curve while training.

    Records (a) the mean reward of at most ``windows`` contiguous step windows and
    (b) the return of every completed episode. Wrapping is transparent to SB3: the
    observation/action spaces and the step contract are forwarded unchanged, so the
    policy sees exactly the same environment as without the recorder.
    """

    def __init__(self, env: gym.Env[Any, Any], *, window_steps: int, max_points: int) -> None:
        if window_steps < 1:
            raise ValueError("window_steps must be >= 1")
        if max_points < 1:
            raise ValueError("max_points must be >= 1")
        super().__init__(env)
        self._window_size = int(window_steps)
        self._max_points = int(max_points)
        self._step_index = 0
        self._window_sum = 0.0
        self._window_count = 0
        self._episode_sum = 0.0
        self.points: list[dict[str, float | int]] = []
        self.window_mean_rewards: list[float] = []
        self.episode_rewards: list[float] = []

    @property
    def window_steps(self) -> int:
        """Number of environment steps aggregated into one curve point."""
        return self._window_size

    @property
    def final_mean_reward(self) -> float | None:
        """Mean reward of the last recorded window (``None`` when empty)."""
        return self.window_mean_rewards[-1] if self.window_mean_rewards else None

    @property
    def best_mean_reward(self) -> float | None:
        """Best window mean reward observed (``None`` when empty)."""
        return max(self.window_mean_rewards) if self.window_mean_rewards else None

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        value = float(reward)
        self._step_index += 1
        self._window_sum += value
        self._window_count += 1
        self._episode_sum += value
        if terminated or truncated:
            self.episode_rewards.append(self._episode_sum)
            self._episode_sum = 0.0
        if self._window_count >= self._window_size:
            self._flush_window()
        return observation, value, terminated, truncated, info

    def flush(self) -> None:
        """Emit the trailing partial window (call once after training stops)."""
        self._flush_window()

    def _flush_window(self) -> None:
        if self._window_count <= 0:
            return
        mean_reward = self._window_sum / self._window_count
        self.points.append({"timestep": self._step_index, "mean_reward": mean_reward})
        self.window_mean_rewards.append(mean_reward)
        self._window_sum = 0.0
        self._window_count = 0

    def curve_payload(self) -> dict[str, object]:
        """Bounded, JSON-serialisable reward-curve summary embedded in metadata.

        The raw window series is down-sampled to at most ``max_points`` contiguous
        buckets (SB3 often steps far more than ``total_timesteps`` in its first
        rollout, e.g. PPO's 2048-step minimum), so a checkpoint sidecar never grows
        with the training budget. Only the most recent ``MAX_RECORDED_EPISODES``
        completed-episode returns are embedded; the total count is preserved.
        """
        means = list(self.window_mean_rewards)
        points = [dict(point) for point in self.points]
        if len(means) > self._max_points:
            bucket_size = math.ceil(len(means) / self._max_points)
            downsampled_means: list[float] = []
            downsampled_points: list[dict[str, float | int]] = []
            for start in range(0, len(means), bucket_size):
                chunk = means[start : start + bucket_size]
                chunk_points = points[start : start + bucket_size]
                downsampled_means.append(math.fsum(chunk) / len(chunk))
                downsampled_points.append(
                    {
                        "timestep": chunk_points[-1]["timestep"],
                        "mean_reward": downsampled_means[-1],
                    }
                )
            means = downsampled_means
            points = downsampled_points
        episodes = self.episode_rewards
        return {
            "window_steps": self._window_size,
            "points": points,
            "window_mean_rewards": means,
            "episode_rewards": episodes[-MAX_RECORDED_EPISODES:],
            "episode_count": len(episodes),
            "env_steps": self._step_index,
            "samples": len(means),
            "final_mean_reward": means[-1] if means else None,
            "best_mean_reward": max(means) if means else None,
        }


# ---------------------------------------------------------------------------
# Offline training data
# ---------------------------------------------------------------------------


def _pivot(
    frame: pl.DataFrame, value_column: str, instrument_ids: Sequence[InstrumentId]
) -> pl.DataFrame:
    """Pivot ``(timestamp, root)`` rows into one column per root, sorted by time."""
    pivoted = frame.pivot(
        on=s.ROOT_SYMBOL_COLUMN,
        index=s.TIMESTAMP_COLUMN,
        values=value_column,
        aggregate_function="first",
    ).sort(s.TIMESTAMP_COLUMN)
    missing = [root for root in instrument_ids if root not in pivoted.columns]
    if missing:
        raise RuntimeError(
            f"offline feature frame is missing roots {missing} for column {value_column!r}"
        )
    return pivoted


def _frame_array(
    frame: pl.DataFrame, instrument_ids: Sequence[InstrumentId]
) -> NDArray[np.float64]:
    return np.asarray(frame.select(list(instrument_ids)).to_numpy(), dtype=np.float64)


def _resolve_universe(
    config: StrategyConfig,
    instruments: InstrumentsConfig,
    frame: pl.DataFrame,
    initial_equity: float,
) -> tuple[InstrumentId, ...]:
    """Resolve the training universe with the strategy's configured selector."""
    builder = MarketStateBuilder(instrument_master=instruments, initial_equity=initial_equity)
    timestamps = builder.timestamps(frame)
    if not timestamps:
        raise RuntimeError("offline feature frame is empty; cannot select a training universe")
    state = builder.build(frame, timestamps[-1], equity=initial_equity)
    selector = build_selector(config.universe.name, config.universe.params, instruments)
    selected = tuple(selector.select(state))
    if selected:
        return selected
    # A too-strict liquidity/regime filter must not make training impossible.
    fallback = tuple(
        root
        for root in sorted(frame[s.ROOT_SYMBOL_COLUMN].unique().to_list())
        if isinstance(root, str)
    )
    if not fallback:  # pragma: no cover - guarded by the empty-frame check above
        raise RuntimeError("offline feature frame contains no roots to train on")
    return fallback


def build_training_data(
    config: StrategyConfig,
    instruments: InstrumentsConfig,
    *,
    seed: int,
    periods: int = DEFAULT_PERIODS,
) -> TrainingData:
    """Build deterministic offline price/feature matrices for the configured universe.

    The 45-root offline build is expensive (synthetic F1/F2 bars + the full
    ``FeatureProcessor`` frame + universe selection), so results are cached in a
    small bounded in-process LRU — one entry per (instrument-master content,
    ``seed``, ``periods``, feature config, universe selector, initial equity) —
    with :data:`TRAINING_DATA_CACHE_MAXSIZE` entries. Repeated calls with the same
    inputs (API retries, multiple seeds sharing a build, tests) return the cached
    matrices, while the cached snapshot keeps read-only numpy arrays and each call
    receives its own writable defensive copy, so callers can never mutate cached
    state. Use :func:`training_data_cache_info` for hit/miss telemetry and
    :func:`clear_training_data_cache` to evict everything.
    """
    key = _training_data_cache_key(config, instruments, seed=seed, periods=periods)
    cached = _training_data_cache_lookup(key)
    if cached is not None:
        return _copy_training_data(cached)
    data = _build_training_data_uncached(config, instruments, seed=seed, periods=periods)
    _training_data_cache_store(key, data)
    return _copy_training_data(data)


def _build_training_data_uncached(
    config: StrategyConfig,
    instruments: InstrumentsConfig,
    *,
    seed: int,
    periods: int = DEFAULT_PERIODS,
) -> TrainingData:
    """Uncached implementation behind :func:`build_training_data`."""
    initial_equity = config.backtest.initial_equity if config.backtest is not None else 1_000_000.0
    # Training must see the expanded instrument master, not the legacy 5-root
    # synthetic default: pass the configured universe roots explicitly.
    source = OfflineBarSource(
        roots=tuple(instruments.roots()),
        seed=seed,
        periods=periods,
        instrument_master=instruments,
    )
    generated_roots = tuple(source.roots)
    bars = source.load(
        generated_roots,
        source.start,
        source.start + timedelta(days=periods),
    )
    if bars.is_empty():
        raise RuntimeError("OfflineBarSource produced no bars; increase --periods")
    feature_frame = FeatureProcessor(instrument_master=instruments).compute(bars)
    universe = _resolve_universe(config, instruments, feature_frame, initial_equity)

    price_frame = _pivot(feature_frame, s.CLOSE_COLUMN, universe)
    prices = _frame_array(price_frame, universe)
    feature_names = DEFAULT_OBSERVATION_FEATURES
    feature_stack = np.stack(
        [_frame_array(_pivot(feature_frame, name, universe), universe) for name in feature_names],
        axis=-1,
    )
    if prices.shape[0] != feature_stack.shape[0]:  # pragma: no cover - same frame source
        raise RuntimeError("price and feature matrices are misaligned")
    finite = np.isfinite(feature_stack)
    available = np.any(finite.reshape(-1, len(feature_names)), axis=0)
    feature_fallbacks = tuple(
        name for name, present in zip(feature_names, available, strict=True) if not present
    )
    required = np.flatnonzero(available)
    if required.size == 0:  # pragma: no cover - canonical features always exist
        raise RuntimeError("offline feature frame carries no usable features")
    rows_ok = np.all(finite[:, :, required], axis=(1, 2))
    bad_rows = np.flatnonzero(~rows_ok)
    start_index = 0 if bad_rows.size == 0 else int(bad_rows[-1]) + 1
    if prices.shape[0] - start_index < 2:
        raise RuntimeError(
            "offline data has fewer than two fully-featured rows after warm-up; "
            f"increase --periods (currently {periods})"
        )
    prices = prices[start_index:]
    feature_stack = feature_stack[start_index:]
    timestamps = MarketStateBuilder(instrument_master=instruments).timestamps(feature_frame)
    timestamps = timestamps[start_index:]
    if not np.all(np.isfinite(prices)):  # pragma: no cover - synthetic bars are finite
        raise RuntimeError("offline price matrix contains non-finite prices")
    # Features the offline source cannot produce (e.g. term structure / carry from a
    # single synthetic expiration) are zero-filled: that is exactly what the pipeline
    # observation builder does for a missing feature on a real MarketState, so the
    # policy never sees a different convention at inference time.
    feature_stack = np.nan_to_num(feature_stack, nan=0.0, posinf=0.0, neginf=0.0)

    specs = [instruments.get(root) for root in universe]
    return TrainingData(
        prices=np.asarray(prices, dtype=np.float64),
        features=np.asarray(feature_stack, dtype=np.float64),
        timestamps=timestamps,
        instrument_ids=universe,
        feature_names=feature_names,
        feature_fallbacks=feature_fallbacks,
        multipliers=tuple(spec.multiplier for spec in specs),
        initial_margins=tuple(spec.margins.initial for spec in specs),
        maintenance_margins=tuple(spec.margins.maintenance for spec in specs),
        tick_sizes=tuple(spec.tick_size for spec in specs),
        periods=periods,
    )


def build_env(
    config: StrategyConfig,
    data: TrainingData,
    *,
    drl: DRLSpec,
) -> FuturesTradingEnv:
    """Build the Gymnasium futures environment used for training."""
    backtest = config.backtest
    initial_equity = backtest.initial_equity if backtest is not None else 1_000_000.0
    commission_bps = backtest.commission_bps if backtest is not None else 1.0
    slippage_ticks = backtest.slippage_ticks if backtest is not None else 1.0
    return FuturesTradingEnv(
        prices=data.prices,
        instrument_ids=data.instrument_ids,
        features=data.features,
        feature_names=data.feature_names,
        multipliers=data.multipliers,
        initial_margins=data.initial_margins,
        maintenance_margins=data.maintenance_margins,
        tick_sizes=data.tick_sizes,
        initial_equity=initial_equity,
        commission_bps=commission_bps,
        slippage_ticks=slippage_ticks,
        max_gross_leverage=1.0,
        max_margin_utilization=config.translator.max_margin_utilization,
        risk_penalty_lambda=drl.reward_risk_penalty_lambda,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _box_shape(space: spaces.Space[object]) -> list[int]:
    shape = space.shape
    return [] if shape is None else [int(dimension) for dimension in shape]


def _observation_metadata(env: FuturesTradingEnv) -> tuple[list[int], dict[str, object]]:
    space = env.observation_space
    if not isinstance(space, spaces.Box):  # pragma: no cover - env always exposes a Box
        raise RuntimeError("FuturesTradingEnv must expose a Box observation space")
    bounds: dict[str, object] = {
        "low": float(np.min(space.low)),
        "high": float(np.max(space.high)),
        "dtype": str(space.dtype),
        "clipped": True,
    }
    return _box_shape(space), bounds


def train_drl(
    *,
    strategy: str | Path = DEFAULT_STRATEGY_PATH,
    algorithm: str | None = None,
    timesteps: int | None = None,
    learning_rate: float | None = None,
    gamma: float | None = None,
    llm_reward_penalty_lambda: float | None = None,
    checkpoint_name: str | None = None,
    output: str | Path | None = None,
    output_dir: str | Path | None = None,
    offline: bool = True,
    seed: int | None = None,
    device: str = DEFAULT_DEVICE,
    periods: int | None = None,
    reward_curve_windows: int = DEFAULT_REWARD_CURVE_WINDOWS,
) -> dict[str, object]:
    """Train one DRL checkpoint and return the JSON-serialisable run summary.

    This is the single callable shared by the CLI (:func:`main`) and the REST API
    (:class:`src.api.drl_service.DRLService`), so both entry points produce
    byte-identical checkpoints/metadata for the same arguments.

    Parameters
    ----------
    strategy:
        Strategy YAML with a ``drl:`` section (algorithm/policy/seed defaults).
    algorithm:
        Optional ``PPO``/``CPPO``/``SAC``/``TD3``/``A2C`` override (case-insensitive).
    timesteps:
        Optional training-budget override (bounded to ``MAX_TIMESTEPS``).
    learning_rate, gamma, llm_reward_penalty_lambda:
        Optional hyperparameter overrides (validated ranges; config values used
        when omitted).
    checkpoint_name:
        Optional checkpoint stem inside ``output_dir`` (validated; no traversal).
        Ignored when ``output`` is given.
    output:
        Explicit checkpoint path (``.zip`` appended when missing). Highest priority.
    output_dir:
        Artifact directory used with ``checkpoint_name``/default naming (default:
        ``artifacts/models``).
    offline:
        Currently the only supported data source is the deterministic offline one;
        it is recorded in the summary. ``offline=False`` is rejected.
    seed, device, periods:
        Optional seed/device/offline-bar-count overrides.
    reward_curve_windows:
        Number of windows in the bounded reward-curve summary (default 20).

    Returns
    -------
    dict[str, object]
        The training summary, identical to the ``<checkpoint>.meta.json`` sidecar:
        model/meta/reward-curve paths, hyperparameters, universe/features, timing
        and the ``reward_curve`` values (window means + completed-episode returns).

    Raises
    ------
    ValueError
        For invalid algorithm/timesteps/learning_rate/gamma/penalty/checkpoint_name/
        device/periods/offline arguments (actionable message).
    RuntimeError
        For missing strategy/DRL config, missing Stable-Baselines3, unwritable
        output paths or unusable offline training data.
    """
    if offline is not True:
        raise ValueError(
            "only deterministic offline training is supported; pass offline=True "
            "(the CLI --offline flag is accepted for parity)"
        )
    started = time.perf_counter()
    strategy_path = Path(strategy)
    try:
        config = load_strategy_config(strategy_path)
    except FileNotFoundError as exc:
        raise RuntimeError(f"strategy config not found: {strategy_path}") from exc
    drl = _resolve_drl_spec(config)
    resolved_algorithm = normalise_algorithm(algorithm if algorithm is not None else drl.algorithm)
    resolved_timesteps = validate_timesteps(drl.total_timesteps if timesteps is None else timesteps)
    resolved_learning_rate = validate_learning_rate(
        drl.learning_rate if learning_rate is None else learning_rate
    )
    resolved_gamma = validate_gamma(drl.gamma if gamma is None else gamma)
    resolved_penalty = validate_llm_reward_penalty_lambda(
        drl.reward_risk_penalty_lambda
        if llm_reward_penalty_lambda is None
        else llm_reward_penalty_lambda
    )
    resolved_seed = _validate_seed(drl.seed if seed is None else seed)
    resolved_device = _validate_device(device)
    resolved_periods = _validate_periods(DEFAULT_PERIODS if periods is None else periods)
    resolved_checkpoint_name = (
        validate_checkpoint_name(checkpoint_name) if checkpoint_name is not None else None
    )
    effective_windows, window_steps = _validate_reward_curve_windows(
        reward_curve_windows, resolved_timesteps
    )
    # Effective hyperparameters (config values were already validated by pydantic;
    # overrides were validated above). Kept as non-optional floats for the summary.
    effective_learning_rate = 3e-4 if resolved_learning_rate is None else resolved_learning_rate
    effective_gamma = 0.99 if resolved_gamma is None else resolved_gamma
    effective_penalty = 0.5 if resolved_penalty is None else resolved_penalty
    checkpoint = _prepare_output_path(
        _resolve_checkpoint_path(
            Path(output) if output is not None else None,
            Path(output_dir) if output_dir is not None else None,
            resolved_checkpoint_name,
            config,
            resolved_algorithm,
        )
    )
    meta_path = Path(f"{checkpoint}.meta.json")
    reward_curve_path = Path(f"{checkpoint}{REWARD_CURVE_SUFFIX}")
    # Fail fast (and cleanly) when the optional DRL stack is unavailable, before
    # the comparatively expensive offline data build.
    sb3_version = _sb3_version()

    instruments = load_instruments()
    data = build_training_data(config, instruments, seed=resolved_seed, periods=resolved_periods)
    env = build_env(config, data, drl=drl)
    recorder = _RewardCurveRecorder(env, window_steps=window_steps, max_points=effective_windows)
    allocator = DRLAllocator(
        algorithm=resolved_algorithm,
        policy=drl.policy,
        total_timesteps=resolved_timesteps,
        learning_rate=effective_learning_rate,
        gamma=effective_gamma,
        seed=resolved_seed,
        risk_penalty_lambda=effective_penalty,
        device=resolved_device,
        feature_names=data.feature_names,
        verbose=0,
    )
    training_started = time.perf_counter()
    model = allocator.train(recorder, total_timesteps=resolved_timesteps)
    recorder.flush()
    duration_seconds = time.perf_counter() - training_started
    model.save(str(checkpoint))
    reward_curve = recorder.curve_payload()
    reward_curve_path.write_text(
        json.dumps(reward_curve, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    observation_shape, observation_bounds = _observation_metadata(env)
    trained_at = datetime.now(UTC)
    sb3_algorithm = "PPO" if resolved_algorithm == "CPPO" else resolved_algorithm
    hyperparameters: dict[str, object] = {
        "algorithm": resolved_algorithm,
        "sb3_algorithm": sb3_algorithm,
        "policy": drl.policy,
        "total_timesteps": resolved_timesteps,
        "learning_rate": effective_learning_rate,
        "gamma": effective_gamma,
        "llm_reward_penalty_lambda": effective_penalty,
        "seed": resolved_seed,
        "device": resolved_device,
        "reward_curve_windows": effective_windows,
    }
    summary: dict[str, object] = {
        "status": "ok",
        "strategy_id": config.id,
        "strategy_path": str(strategy_path),
        "algorithm": resolved_algorithm,
        "sb3_algorithm": sb3_algorithm,
        "policy": drl.policy,
        "total_timesteps": resolved_timesteps,
        "learning_rate": effective_learning_rate,
        "gamma": effective_gamma,
        "llm_reward_penalty_lambda": effective_penalty,
        "seed": resolved_seed,
        "device": resolved_device,
        "checkpoint_name": checkpoint.name[: -len(CHECKPOINT_SUFFIX)]
        if checkpoint.name.endswith(CHECKPOINT_SUFFIX)
        else checkpoint.name,
        "model_path": str(checkpoint),
        "meta_path": str(meta_path),
        "reward_curve_path": str(reward_curve_path),
        "universe": list(data.instrument_ids),
        "feature_names": list(data.feature_names),
        "feature_fallbacks": list(data.feature_fallbacks),
        "observation_shape": observation_shape,
        "observation_bounds": observation_bounds,
        "action_shape": _box_shape(env.action_space),
        "offline": True,
        "periods": resolved_periods,
        "usable_rows": data.usable_rows,
        "initial_equity": float(config.backtest.initial_equity)
        if config.backtest is not None
        else 1_000_000.0,
        "commission_bps": float(config.backtest.commission_bps)
        if config.backtest is not None
        else 1.0,
        "slippage_ticks": float(config.backtest.slippage_ticks)
        if config.backtest is not None
        else 1.0,
        "trained_at": trained_at.isoformat(),
        "duration_seconds": round(duration_seconds, 6),
        "total_duration_seconds": round(time.perf_counter() - started, 6),
        "sb3_version": sb3_version,
        "git_commit": _git_commit(),
        "python_version": sys.version.split()[0],
        "hyperparameters": hyperparameters,
        "reward_curve": reward_curve,
    }
    meta_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def run_training(args: argparse.Namespace) -> dict[str, object]:
    """CLI adapter: map parsed arguments onto :func:`train_drl`."""
    return train_drl(
        strategy=Path(args.strategy),
        algorithm=args.algorithm,
        timesteps=args.timesteps,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        llm_reward_penalty_lambda=args.llm_reward_penalty_lambda,
        checkpoint_name=args.checkpoint_name,
        output=args.output,
        output_dir=args.output_dir,
        offline=True,
        seed=args.seed,
        device=args.device,
        periods=args.periods,
        reward_curve_windows=args.reward_curve_windows,
    )


def _print_human(summary: dict[str, object]) -> None:
    print(f"trained {summary['algorithm']} checkpoint: {summary['model_path']}")
    print(f"metadata: {summary['meta_path']}")
    print(
        "timesteps="
        f"{summary['total_timesteps']} seed={summary['seed']} device={summary['device']} "
        f"universe={summary['universe']} rows={summary['usable_rows']}"
    )
    print(
        f"duration={summary['duration_seconds']}s sb3={summary['sb3_version']} "
        f"observation_shape={summary['observation_shape']}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: ``0`` on success, ``2`` on an actionable error."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.json:
            # Keep stdout machine-readable even if an SB3/gymnasium helper prints.
            with contextlib.redirect_stdout(sys.stderr):
                summary = run_training(args)
        else:
            summary = run_training(args)
    except (RuntimeError, ValueError, OSError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(summary, sort_keys=True))
    else:
        _print_human(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
