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
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

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
#: Default SB3 device (CPU keeps the tiny MLP policies deterministic and fast).
DEFAULT_DEVICE: str = "cpu"
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


def _resolve_checkpoint_path(output: Path | None, config: StrategyConfig, algorithm: str) -> Path:
    candidate = (
        output if output is not None else DEFAULT_OUTPUT_DIR / f"{config.id}_{algorithm.lower()}"
    )
    candidate = Path(candidate)
    return candidate if candidate.suffix == ".zip" else Path(f"{candidate}.zip")


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
    """Build deterministic offline price/feature matrices for the configured universe."""
    initial_equity = config.backtest.initial_equity if config.backtest is not None else 1_000_000.0
    source = OfflineBarSource(seed=seed, periods=periods, instrument_master=instruments)
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


def run_training(args: argparse.Namespace) -> dict[str, object]:
    """Execute one training run; returns the JSON-serialisable summary."""
    started = time.perf_counter()
    strategy_path = Path(args.strategy)
    try:
        config = load_strategy_config(strategy_path)
    except FileNotFoundError as exc:
        raise RuntimeError(f"strategy config not found: {strategy_path}") from exc
    drl = _resolve_drl_spec(config)
    algorithm = str(args.algorithm) if args.algorithm is not None else drl.algorithm.upper()
    timesteps = int(drl.total_timesteps if args.timesteps is None else args.timesteps)
    seed = int(drl.seed if args.seed is None else args.seed)
    device = str(args.device)
    periods = int(args.periods)
    checkpoint = _prepare_output_path(_resolve_checkpoint_path(args.output, config, algorithm))
    meta_path = Path(f"{checkpoint}.meta.json")
    # Fail fast (and cleanly) when the optional DRL stack is unavailable, before
    # the comparatively expensive offline data build.
    sb3_version = _sb3_version()

    instruments = load_instruments()
    data = build_training_data(config, instruments, seed=seed, periods=periods)
    env = build_env(config, data, drl=drl)
    allocator = DRLAllocator(
        algorithm=algorithm,
        policy=drl.policy,
        total_timesteps=timesteps,
        learning_rate=drl.learning_rate,
        gamma=drl.gamma,
        seed=seed,
        risk_penalty_lambda=drl.reward_risk_penalty_lambda,
        device=device,
        feature_names=data.feature_names,
        verbose=0,
    )
    training_started = time.perf_counter()
    model = allocator.train(env, total_timesteps=timesteps)
    duration_seconds = time.perf_counter() - training_started
    model.save(str(checkpoint))
    observation_shape, observation_bounds = _observation_metadata(env)
    trained_at = datetime.now(UTC)
    summary: dict[str, object] = {
        "status": "ok",
        "strategy_id": config.id,
        "strategy_path": str(strategy_path),
        "algorithm": algorithm,
        "sb3_algorithm": "PPO" if algorithm == "CPPO" else algorithm,
        "policy": drl.policy,
        "total_timesteps": timesteps,
        "seed": seed,
        "device": device,
        "model_path": str(checkpoint),
        "meta_path": str(meta_path),
        "universe": list(data.instrument_ids),
        "feature_names": list(data.feature_names),
        "feature_fallbacks": list(data.feature_fallbacks),
        "observation_shape": observation_shape,
        "observation_bounds": observation_bounds,
        "action_shape": _box_shape(env.action_space),
        "offline": True,
        "periods": periods,
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
    }
    meta_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


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
