"""Phase-10 A2 integration test: non-zero offline term structure for DRL training.

``data-eng``'s :mod:`src.data.term_structure` supplies deterministic synthetic
front/second-month (F1/F2) curves for the offline bar source, so
:class:`src.data.feature_processor.FeatureProcessor` can now emit real
``term_slope`` and ``carry_annualized`` features. This test verifies the A2
acceptance contract end to end at the training-data and environment level:

* ``src.strategies.train_drl.build_training_data(...)`` (the public helper shared
  by the ``train_drl`` CLI/API path) reports ``feature_fallbacks == ()``;
* ``term_slope`` / ``carry_annualized`` are finite, never zero, sign-opposite
  (``sign(carry) == -sign(slope)``), evolve over time and contain both positive
  and negative values for *every* trained root (backwardation and contango);
* two identical calls are exactly deterministic;
* the resulting :class:`~src.strategies.envs.FuturesTradingEnv` has finite
  observation bounds and returns in-space, finite observations.

The optional tiny PPO training smoke is ``slow``-marked, uses a 64-timestep CPU
budget and is skipped when Stable-Baselines3 is unavailable; the fast data/env
assertions run in under two seconds.
"""

from __future__ import annotations

import inspect
import math
from pathlib import Path

import numpy as np
import pytest
from configs.instruments import InstrumentsConfig, load_instruments
from configs.strategy import StrategyConfig, load_strategy_config
from numpy.typing import NDArray
from src.strategies._observation import DEFAULT_OBSERVATION_FEATURES
from src.strategies.allocators import DRLAllocator
from src.strategies.envs import FuturesTradingEnv
from src.strategies.train_drl import TrainingData, build_env, build_training_data

from tests.fixtures.market import make_market_state

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
DRL_STRATEGY_PATH = REPO_ROOT / "configs" / "strategies" / "drl_ppo_llm_risk.yaml"

#: Canonical features that the synthetic F1/F2 curve must populate (A2).
TERM_STRUCTURE_FEATURES: tuple[str, ...] = ("term_slope", "carry_annualized")
#: Offline daily bars per root (512 -> ~386 usable rows after feature warm-up).
OFFLINE_PERIODS: int = 512
SEED: int = 7


def _build_offline_training_data(
    config: StrategyConfig,
    instruments: InstrumentsConfig,
    *,
    seed: int = SEED,
    periods: int = OFFLINE_PERIODS,
) -> TrainingData:
    """Call the trainer's data helper with ``offline=True`` when it supports it.

    The Phase-9/10 trainer is offline-only, so ``build_training_data`` may or may
    not expose an explicit ``offline`` keyword depending on the current
    ``src/strategies/train_drl.py`` refactor; the test asserts the offline path
    either way instead of pinning an incidental signature detail.
    """
    parameters = inspect.signature(build_training_data).parameters
    optional: dict[str, object] = {"seed": seed, "periods": periods}
    if "offline" in parameters:
        optional["offline"] = True
    return build_training_data(config, instruments, **optional)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def drl_config() -> StrategyConfig:
    """The committed DRL strategy config (algorithm/universe/features under test)."""
    return load_strategy_config(DRL_STRATEGY_PATH)


@pytest.fixture(scope="module")
def instruments() -> InstrumentsConfig:
    """The committed instrument master."""
    return load_instruments()


@pytest.fixture(scope="module")
def training_data(drl_config: StrategyConfig, instruments: InstrumentsConfig) -> TrainingData:
    """Deterministic offline training data for the DRL strategy (built once)."""
    assert drl_config.drl is not None
    return _build_offline_training_data(drl_config, instruments)


def _feature_slice(data: TrainingData, feature_name: str) -> NDArray[np.float64]:
    """Time-by-root matrix of one observation feature."""
    return data.features[:, :, data.feature_names.index(feature_name)]


# ---------------------------------------------------------------------------
# A2: non-zero synthetic term structure reaches the training data
# ---------------------------------------------------------------------------


def test_feature_fallbacks_are_empty(training_data: TrainingData) -> None:
    """A2 acceptance: no canonical observation feature is zero-filled any more."""
    assert training_data.feature_fallbacks == ()
    assert training_data.feature_names == DEFAULT_OBSERVATION_FEATURES
    assert set(TERM_STRUCTURE_FEATURES) <= set(training_data.feature_names)


def test_term_slope_and_carry_are_finite_non_zero_and_two_sided(
    training_data: TrainingData,
) -> None:
    rows, root_count = training_data.prices.shape
    assert rows >= 2
    assert len(training_data.instrument_ids) == root_count
    for feature_name in TERM_STRUCTURE_FEATURES:
        values = _feature_slice(training_data, feature_name)
        assert values.shape == (rows, root_count)
        assert np.all(np.isfinite(values))
        # The synthetic curve has a strict minimum |carry|: no zero-filled columns.
        assert np.all(values != 0.0)
        assert float(np.min(np.abs(values))) > 1e-9
        assert int(np.count_nonzero(values > 0.0)) > 0
        assert int(np.count_nonzero(values < 0.0)) > 0
        # The calendar-time regime cycle makes the curve evolve across the window.
        assert float(np.std(values)) > 0.0
        assert np.unique(np.round(values, 12)).size > 1
        # Every trained root sees both backwardation and contango in the sample.
        for column, root in enumerate(training_data.instrument_ids):
            per_root = values[:, column]
            assert np.any(per_root > 0.0), f"{root} never in backwardation for {feature_name}"
            assert np.any(per_root < 0.0), f"{root} never in contango for {feature_name}"
            assert np.unique(np.round(per_root, 12)).size > 1

    slope = _feature_slice(training_data, "term_slope")
    carry = _feature_slice(training_data, "carry_annualized")
    # Canonical sign convention: carry = -term_slope * 365 / days_between_expiries.
    assert np.array_equal(np.sign(carry), -np.sign(slope))
    # Sanity bounds for annualised futures carry (synthetic curves are conservative).
    assert float(np.max(np.abs(slope))) < 0.10
    assert float(np.max(np.abs(carry))) < 1.0
    assert float(np.max(np.abs(training_data.prices))) > 0.0


def test_offline_training_data_is_deterministic(
    drl_config: StrategyConfig,
    instruments: InstrumentsConfig,
    training_data: TrainingData,
) -> None:
    again = _build_offline_training_data(drl_config, instruments)
    assert again.feature_fallbacks == ()
    assert again.instrument_ids == training_data.instrument_ids
    assert again.feature_names == training_data.feature_names
    assert again.timestamps == training_data.timestamps
    assert again.usable_rows == training_data.usable_rows
    assert np.array_equal(again.prices, training_data.prices)
    assert np.array_equal(again.features, training_data.features)
    assert np.all(np.isfinite(again.features))
    assert np.all(again.prices > 0.0)


# ---------------------------------------------------------------------------
# Environment built from the A2 training data
# ---------------------------------------------------------------------------


def test_env_has_finite_observation_bounds_and_valid_observations(
    drl_config: StrategyConfig, training_data: TrainingData
) -> None:
    assert drl_config.drl is not None
    env = build_env(drl_config, training_data, drl=drl_config.drl)
    assert isinstance(env, FuturesTradingEnv)
    assert env.instrument_ids == training_data.instrument_ids
    assert env.feature_names == training_data.feature_names
    asset_count = len(training_data.instrument_ids)
    expected_dim = asset_count * len(DEFAULT_OBSERVATION_FEATURES) + asset_count + 1
    space = env.observation_space
    assert space.shape == (expected_dim,)
    assert space.dtype == np.float32
    assert space.low.dtype == np.float32 and space.high.dtype == np.float32
    assert np.all(np.isfinite(space.low)) and np.all(np.isfinite(space.high))
    assert np.all(space.low == -1_000_000.0)
    assert np.all(space.high == 1_000_000.0)

    observation, info = env.reset(seed=SEED)
    assert observation.dtype == np.float32
    assert np.all(np.isfinite(observation))
    assert space.contains(observation)
    assert math.isfinite(float(info["equity"]))
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    observation, reward, terminated, truncated, step_info = env.step(action)
    assert np.all(np.isfinite(observation))
    assert space.contains(observation)
    assert math.isfinite(float(reward))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert math.isfinite(float(step_info["equity"]))
    # The env feature block (including the A2 term-structure columns) is finite.
    assert np.all(np.isfinite(training_data.features))


def test_env_feature_block_matches_training_features(
    drl_config: StrategyConfig, training_data: TrainingData
) -> None:
    """The observation's per-root feature block is exactly the A2 feature matrix row."""
    env = build_env(drl_config, training_data, drl=drl_config.drl)
    observation, _ = env.reset(seed=SEED)
    feature_dim = len(DEFAULT_OBSERVATION_FEATURES)
    asset_count = len(training_data.instrument_ids)
    feature_block = observation[: asset_count * feature_dim].reshape(asset_count, feature_dim)
    expected = training_data.features[0]
    assert np.allclose(feature_block, expected.astype(np.float32))
    # The A2 columns are present and non-zero in the very first observation.
    for name in TERM_STRUCTURE_FEATURES:
        column = training_data.feature_names.index(name)
        assert np.all(feature_block[:, column] != 0.0)


# ---------------------------------------------------------------------------
# Optional tiny PPO training smoke (slow, skipped without Stable-Baselines3)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_tiny_ppo_training_smoke_on_offline_term_structure_data(
    drl_config: StrategyConfig, training_data: TrainingData
) -> None:
    """Train PPO for a tiny CPU budget and allocate finite, capped weights."""
    pytest.importorskip("stable_baselines3")
    assert drl_config.drl is not None
    env = build_env(drl_config, training_data, drl=drl_config.drl)
    allocator = DRLAllocator(
        algorithm="PPO",
        total_timesteps=64,
        device="cpu",
        seed=SEED,
        feature_names=training_data.feature_names,
        gross_leverage=1.0,
    )
    model = allocator.train(env, total_timesteps=64)
    assert model is not None
    assert allocator.is_trained is True
    state = make_market_state(roots=tuple(training_data.instrument_ids))
    weights = allocator.allocate(tuple(training_data.instrument_ids), state)
    assert set(weights.as_dict()) == set(training_data.instrument_ids)
    assert all(np.isfinite(value) for value in weights.as_dict().values())
    assert weights.gross <= 1.0 + 1e-9


def test_short_window_training_data_also_has_no_fallbacks(
    drl_config: StrategyConfig, instruments: InstrumentsConfig
) -> None:
    """A shorter offline window (256 bars) is still fully featured after warm-up.

    The ``train_drl`` summary/metadata derives ``feature_fallbacks`` from exactly
    this ``TrainingData`` field (tightened to ``[]`` post-A2 by risk-eng's suite);
    asserting it here at a second window length guards the data path the trainer
    serialises and keeps the A2 gap closed for API-triggered short runs.
    """
    data = _build_offline_training_data(drl_config, instruments, periods=256)
    assert data.feature_fallbacks == ()
    assert data.usable_rows >= 2
    assert data.feature_names == DEFAULT_OBSERVATION_FEATURES
    for feature_name in TERM_STRUCTURE_FEATURES:
        values = _feature_slice(data, feature_name)
        assert np.all(np.isfinite(values))
        assert np.all(values != 0.0)
        # A short window may cover only part of the calendar regime cycle, so it
        # need not contain both signs (the 512-bar test asserts two-sidedness);
        # it must still be a real, time-varying curve rather than a constant fill.
        assert np.unique(np.round(values, 12)).size > 1
        assert float(np.std(values)) > 0.0


@pytest.mark.parametrize("seed", [11, 21])
def test_a2_term_structure_holds_for_other_offline_seeds(
    drl_config: StrategyConfig, instruments: InstrumentsConfig, seed: int
) -> None:
    """The synthetic F1/F2 curve is calendar-driven, not seed luck.

    ``term_slope``/``carry_annualized`` come from data-eng's deterministic
    calendar-time regime model, while the price path is seeded; the A2 guarantees
    must therefore hold for every offline seed, not just the strategy default.
    """
    data = _build_offline_training_data(drl_config, instruments, seed=seed)
    assert data.feature_fallbacks == ()
    for feature_name in TERM_STRUCTURE_FEATURES:
        values = _feature_slice(data, feature_name)
        assert np.all(np.isfinite(values))
        assert np.all(values != 0.0)
        assert int(np.count_nonzero(values > 0.0)) > 0
        assert int(np.count_nonzero(values < 0.0)) > 0
    slope = _feature_slice(data, "term_slope")
    carry = _feature_slice(data, "carry_annualized")
    assert np.array_equal(np.sign(carry), -np.sign(slope))
