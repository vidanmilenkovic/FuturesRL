"""Unit tests for the portfolio allocators (``A_t``)."""

from __future__ import annotations

import dataclasses
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from configs.instruments import load_instruments
from configs.strategy import DRLSpec, StageConfig, StrategyConfig
from src.core.interfaces import PortfolioAllocator
from src.core.types import AssetClass, InstrumentSnapshot, MarketState, WeightVector
from src.strategies._numeric import covariance_matrix, portfolio_volatility
from src.strategies._observation import DEFAULT_OBSERVATION_FEATURES
from src.strategies._params import ParamError
from src.strategies.allocators import (
    ALLOCATOR_REGISTRY,
    DRLAllocator,
    EqualWeightAllocator,
    MeanVarianceAllocator,
    RiskParityAllocator,
    VolatilityTargetAllocator,
    build_allocator,
    import_stable_baselines3,
)
from src.strategies.envs import FuturesTradingEnv
from src.strategies.pipeline import build_pipeline

from tests.fixtures.market import FIXED_TIMESTAMP, make_market_state, make_risk_assessment


def _drift_state(roots: tuple[str, ...], drifts: dict[str, float]) -> MarketState:
    """Fixture state whose return series have deterministic per-root drifts."""
    state = make_market_state(roots=roots, seed=5)
    rng = np.random.default_rng(5)
    series = {root: rng.normal(drifts[root], 0.004, size=260) for root in state.instruments}
    return dataclasses.replace(state, returns=series)


class _FakeModel:
    """Duck-typed Stable-Baselines3 model returning a fixed action."""

    def __init__(self, action: object) -> None:
        self.action = action

    def predict(self, observation: object, deterministic: bool = True) -> tuple[object, None]:
        assert deterministic is True
        assert observation is not None
        return self.action, None


# ---------------------------------------------------------------------------
# Equal weight
# ---------------------------------------------------------------------------


def test_equal_weight_allocator() -> None:
    state = make_market_state()
    allocator = EqualWeightAllocator(gross_leverage=2.0)
    weights = allocator.allocate(state.ids, state)
    assert weights.gross == pytest.approx(2.0)
    assert weights.net == pytest.approx(2.0)
    assert len(weights) == len(state.ids)
    assert allocator.name == "equal_weight"
    assert allocator.allocate([], state) == WeightVector.empty()


def test_equal_weight_allocator_respects_max_weight() -> None:
    state = make_market_state()
    allocator = EqualWeightAllocator(gross_leverage=5.0, max_weight=0.25)
    weights = allocator.allocate(state.ids, state)
    assert weights.max_abs() == pytest.approx(0.25)
    with pytest.raises(ValueError, match="gross_leverage"):
        EqualWeightAllocator(gross_leverage=-1.0)
    with pytest.raises(ValueError, match="max_weight"):
        EqualWeightAllocator(max_weight=-0.1)


# ---------------------------------------------------------------------------
# Risk parity
# ---------------------------------------------------------------------------


def _risk_contributions(
    state: MarketState, candidates: tuple[str, ...], weights: WeightVector
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    covariance = covariance_matrix(state, candidates)
    vector = weights.to_array(candidates)
    contributions = vector * (covariance @ vector)
    return np.asarray(contributions, dtype=np.float64)


def test_risk_parity_equalizes_risk_contributions() -> None:
    state = make_market_state()
    candidates = state.ids
    allocator = RiskParityAllocator()
    weights = allocator.allocate(candidates, state)
    assert allocator.name == "risk_parity"
    assert weights.gross == pytest.approx(1.0)
    assert all(value > 0.0 for value in weights.as_dict().values())
    contributions = _risk_contributions(state, candidates, weights)
    total = float(contributions.sum())
    assert total > 0.0
    shares = contributions / total
    assert float(shares.max() - shares.min()) < 0.02


def test_risk_parity_volatility_target_and_gross_cap() -> None:
    state = make_market_state()
    candidates = state.ids
    targeted = RiskParityAllocator(target_annual_volatility=0.10)
    weights = targeted.allocate(candidates, state)
    realised = portfolio_volatility(
        weights.to_array(candidates), covariance_matrix(state, candidates)
    )
    assert realised == pytest.approx(0.10, rel=1e-6)
    capped = RiskParityAllocator(target_annual_volatility=10.0, max_gross_leverage=2.5)
    assert capped.allocate(candidates, state).gross == pytest.approx(2.5)


def test_risk_parity_survives_singular_and_missing_covariance() -> None:
    state = make_market_state()
    vector = np.linspace(1.0, 2.0, len(state.instruments))
    rank_one = np.outer(vector, vector)
    singular = dataclasses.replace(state, covariance=rank_one)
    weights = RiskParityAllocator().allocate(state.ids, singular)
    assert math.isfinite(weights.gross)
    assert weights.gross == pytest.approx(1.0)
    no_covariance = make_market_state(with_covariance=False)
    assert RiskParityAllocator().allocate(no_covariance.ids, no_covariance).gross > 0.0
    nothing = make_market_state(with_covariance=False, with_returns=False)
    fallback = RiskParityAllocator().allocate(nothing.ids, nothing)
    assert fallback.gross == pytest.approx(1.0)


def test_risk_parity_custom_budgets_and_caps() -> None:
    state = make_market_state()
    budgets = {root: (3.0 if root == "ZN" else 1.0) for root in state.ids}
    allocator = RiskParityAllocator(risk_budgets=budgets, max_weight=0.20)
    weights = allocator.allocate(state.ids, state)
    assert weights.max_abs() <= 0.20 + 1e-12
    assert allocator.allocate([], state) == WeightVector.empty()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"target_annual_volatility": 0.0}, "target_annual_volatility"),
        ({"gross_leverage": -0.5}, "gross_leverage"),
        ({"max_gross_leverage": -1.0}, "max_gross_leverage"),
        ({"max_weight": 0.0}, "max_weight"),
        ({"covariance_lookback_days": 2}, "covariance_lookback_days"),
        ({"max_iterations": 0}, "max_iterations"),
        ({"tolerance": 0.0}, "tolerance"),
        ({"ridge": -1e-9}, "ridge"),
        ({"shrinkage": 1.0}, "shrinkage"),
        ({"min_volatility": 0.0}, "min_volatility"),
    ],
)
def test_risk_parity_rejects_invalid_arguments(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        RiskParityAllocator(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Mean variance
# ---------------------------------------------------------------------------


def test_mean_variance_long_only_follows_expected_returns() -> None:
    state = _drift_state(("ES", "CL"), {"ES": 0.002, "CL": -0.001})
    allocator = MeanVarianceAllocator()
    weights = allocator.allocate(state.ids, state)
    assert allocator.name == "mean_variance"
    assert weights["ES"] > 0.0
    assert weights.get("CL", 0.0) == pytest.approx(0.0)
    assert weights.gross <= 3.0 + 1e-12


def test_mean_variance_allows_shorts_when_configured() -> None:
    state = _drift_state(("ES", "CL"), {"ES": 0.002, "CL": -0.001})
    allocator = MeanVarianceAllocator(long_only=False)
    weights = allocator.allocate(state.ids, state)
    assert weights["ES"] > 0.0
    assert weights["CL"] < 0.0


def test_mean_variance_respects_caps_and_fallbacks() -> None:
    state = _drift_state(("ES", "CL", "GC"), {"ES": 0.003, "CL": 0.0, "GC": 0.001})
    capped = MeanVarianceAllocator(max_gross_leverage=0.5, max_weight=0.1)
    weights = capped.allocate(state.ids, state)
    assert weights.gross <= 0.5 + 1e-12
    assert weights.max_abs() <= 0.1 + 1e-12
    # A flat expected-return estimate degenerates to equal fallback weights.
    flat = make_market_state(
        roots=("ES", "CL", "GC"),
        with_returns=False,
        features={root: {"ret_126d": 0.0} for root in ("ES", "CL", "GC")},
    )
    fallback = MeanVarianceAllocator().allocate(flat.ids, flat)
    assert fallback.gross == pytest.approx(1.0)
    assert len(set(fallback.as_dict().values())) == 1
    assert MeanVarianceAllocator().allocate([], flat) == WeightVector.empty()


def test_mean_variance_survives_singular_covariance() -> None:
    state = make_market_state()
    singular = dataclasses.replace(
        state, covariance=np.zeros((len(state.instruments), len(state.instruments)))
    )
    weights = MeanVarianceAllocator(ridge=1e-6).allocate(state.ids, singular)
    assert math.isfinite(weights.gross)
    assert weights.gross > 0.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"risk_aversion": 0.0}, "risk_aversion"),
        ({"ridge": -1.0}, "ridge"),
        ({"gross_leverage": -1.0}, "gross_leverage"),
        ({"max_gross_leverage": -1.0}, "max_gross_leverage"),
        ({"max_weight": -0.2}, "max_weight"),
        ({"expected_return_window": 1}, "expected_return_window"),
        ({"covariance_lookback_days": 2}, "covariance_lookback_days"),
        ({"shrinkage": 1.2}, "shrinkage"),
        ({"min_volatility": 0.0}, "min_volatility"),
    ],
)
def test_mean_variance_rejects_invalid_arguments(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        MeanVarianceAllocator(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Volatility target
# ---------------------------------------------------------------------------


def test_volatility_target_inverse_vol_and_vol_target() -> None:
    state = make_market_state()
    candidates = state.ids
    allocator = VolatilityTargetAllocator(target_annual_volatility=0.12)
    weights = allocator.allocate(candidates, state)
    assert allocator.name == "volatility_target"
    # Low-vol rates (ZN) get the largest weight, high-vol energy (CL) the smallest.
    assert weights["ZN"] > weights["ES"] > weights["CL"]
    realised = portfolio_volatility(
        weights.to_array(candidates), covariance_matrix(state, candidates)
    )
    assert realised == pytest.approx(0.12, rel=1e-6)


def test_volatility_target_gross_bounds() -> None:
    state = make_market_state()
    capped = VolatilityTargetAllocator(target_annual_volatility=10.0, max_gross_leverage=2.0)
    assert capped.allocate(state.ids, state).gross == pytest.approx(2.0)
    floored = VolatilityTargetAllocator(target_annual_volatility=0.0001, min_gross_leverage=0.5)
    assert floored.allocate(state.ids, state).gross == pytest.approx(0.5)
    strict = VolatilityTargetAllocator(max_weight=0.1)
    assert strict.allocate(state.ids, state).max_abs() <= 0.1 + 1e-12
    assert VolatilityTargetAllocator().allocate([], state) == WeightVector.empty()


def test_volatility_target_falls_back_without_any_volatility_data() -> None:
    state = make_market_state(with_covariance=False, with_returns=False)
    weights = VolatilityTargetAllocator().allocate(state.ids, state)
    assert weights.gross > 0.0
    assert all(value > 0.0 for value in weights.as_dict().values())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"target_annual_volatility": 0.0}, "target_annual_volatility"),
        ({"covariance_lookback_days": 1}, "covariance_lookback_days"),
        ({"max_gross_leverage": 0.0}, "max_gross_leverage"),
        ({"min_gross_leverage": -1.0}, "min_gross_leverage"),
        ({"min_gross_leverage": 5.0, "max_gross_leverage": 2.0}, "min_gross_leverage"),
        ({"max_weight": 0.0}, "max_weight"),
        ({"ridge": -1.0}, "ridge"),
        ({"shrinkage": 2.0}, "shrinkage"),
        ({"min_volatility": 0.0}, "min_volatility"),
    ],
)
def test_volatility_target_rejects_invalid_arguments(
    kwargs: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        VolatilityTargetAllocator(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# DRL allocator
# ---------------------------------------------------------------------------


def test_drl_allocator_requires_a_model_or_the_sb3_stack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allocator = DRLAllocator()
    with pytest.raises(RuntimeError, match="no model"):
        allocator.allocate(("ES",), make_market_state())
    assert allocator.is_trained is False
    assert allocator.name == "drl"
    assert allocator.algorithm == "PPO"
    assert allocator.device == "cpu"
    # The DRL stack is imported lazily: when unavailable the allocator raises a
    # clear, actionable RuntimeError instead of an ImportError from deep inside
    # torch. ``None`` in sys.modules simulates the package being absent.
    monkeypatch.setitem(sys.modules, "stable_baselines3", None)
    with pytest.raises(RuntimeError, match="stable-baselines3"):
        import_stable_baselines3()
    with pytest.raises(RuntimeError, match="stable-baselines3"):
        allocator.train(object())


def test_drl_allocator_maps_policy_action_to_weights() -> None:
    state = make_market_state(roots=("ES", "NQ", "CL"))
    model = _FakeModel(np.array([0.5, -0.25, 0.0]))
    allocator = DRLAllocator(model=model, gross_leverage=1.0)
    assert allocator.is_trained is True
    weights = allocator.allocate(state.ids, state)
    assert weights["ES"] == pytest.approx(0.5)
    assert weights["NQ"] == pytest.approx(-0.25)
    assert weights["CL"] == pytest.approx(0.0)
    assert allocator.allocate([], state) == WeightVector.empty()
    assert allocator.model is model
    with pytest.raises(ValueError, match="model must not be None"):
        allocator.attach_model(None)


def test_drl_allocator_average_action_modes_and_validation() -> None:
    state = make_market_state(roots=("ES", "NQ", "CL"))
    clipped = DRLAllocator(model=_FakeModel(np.array([3.0, 3.0, 3.0])), gross_leverage=1.0)
    weights = clipped.allocate(state.ids, state)
    assert weights.gross == pytest.approx(1.0)
    assert weights.max_abs() == pytest.approx(1.0 / 3.0)
    normalised = DRLAllocator(
        model=_FakeModel(np.array([2.0, 1.0, 1.0])),
        action_mode="normalized",
        gross_leverage=2.0,
    )
    assert normalised.allocate(state.ids, state).gross == pytest.approx(2.0)
    zero = DRLAllocator(model=_FakeModel(np.zeros(3)), action_mode="normalized")
    assert zero.allocate(state.ids, state).is_flat()
    with pytest.raises(ValueError, match="candidates"):
        clipped.allocate(("ES", "NQ"), state)
    with pytest.raises(RuntimeError, match="non-finite"):
        DRLAllocator(model=_FakeModel(np.array([np.nan, 0.0, 0.0]))).allocate(state.ids, state)
    with pytest.raises(ValueError, match="candidates"):
        clipped.weights_from_action(("ES", "NQ"), np.zeros(3))


def test_drl_allocator_semantic_risk_penalty_hook() -> None:
    allocator = DRLAllocator(risk_penalty_lambda=0.5)
    critical = [make_risk_assessment("ES", semantic_risk_score=4.0, confidence=0.9)]
    assert allocator.semantic_risk_penalty({"ES": 1.0}, critical) == pytest.approx(
        0.5 * (4.0 - 1.5)
    )
    low_confidence = [make_risk_assessment("ES", semantic_risk_score=5.0, confidence=0.1)]
    assert allocator.semantic_risk_penalty({"ES": 1.0}, low_confidence) == 0.0
    benign = [make_risk_assessment("ES", semantic_risk_score=1.5, confidence=0.99)]
    assert allocator.semantic_risk_penalty({"ES": 1.0}, benign) == 0.0
    assert allocator.semantic_risk_penalty({"ES": 1.0}, None) == 0.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"algorithm": "DQN"}, "unsupported DRL algorithm"),
        ({"policy": ""}, "policy"),
        ({"total_timesteps": 0}, "total_timesteps"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"gamma": 0.0}, "gamma"),
        ({"gamma": 1.5}, "gamma"),
        ({"risk_penalty_lambda": -1.0}, "risk_penalty_lambda"),
        ({"confidence_threshold": 1.5}, "confidence_threshold"),
        ({"safe_risk_score": 0.5}, "safe_risk_score"),
        ({"action_mode": "gross"}, "action_mode"),
        ({"gross_leverage": 0.0}, "gross_leverage"),
        ({"max_instrument_weight": 0.0}, "max_instrument_weight"),
        ({"feature_names": ()}, "feature_names"),
        ({"device": ""}, "device"),
    ],
)
def test_drl_allocator_rejects_invalid_arguments(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        DRLAllocator(**kwargs)


def test_drl_allocator_device_is_configurable_and_reported() -> None:
    assert DRLAllocator().device == "cpu"
    assert DRLAllocator(device="cpu").device == "cpu"
    assert DRLAllocator(device="auto").device == "auto"


def test_drl_allocator_builds_lazily_from_yaml_params() -> None:
    allocator = build_allocator(
        "drl",
        {
            "algorithm": "PPO",
            "policy": "MlpPolicy",
            "total_timesteps": 100_000,
            "learning_rate": 0.0003,
            "gamma": 0.99,
            "seed": 7,
        },
    )
    assert isinstance(allocator, DRLAllocator)
    assert allocator.total_timesteps == 100_000
    assert allocator.device == "cpu"
    assert allocator.feature_names
    with pytest.raises(ParamError, match="unknown parameter"):
        build_allocator("drl", {"algoritm": "PPO"})


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_allocator_registry_contents_and_builds() -> None:
    assert set(ALLOCATOR_REGISTRY) == {
        "equal_weight",
        "risk_parity",
        "mean_variance",
        "volatility_target",
        "drl",
    }
    state = make_market_state()
    for name in ("equal_weight", "risk_parity", "mean_variance", "volatility_target"):
        allocator = build_allocator(name)
        assert isinstance(allocator, PortfolioAllocator)
        weights = allocator.allocate(state.ids, state)
        assert isinstance(weights, WeightVector)
        assert math.isfinite(weights.gross)
    with pytest.raises(ValueError, match="unknown portfolio allocator"):
        build_allocator("nope")


def test_allocators_are_independent_of_instrument_master() -> None:
    snapshot = InstrumentSnapshot(
        instrument_id="ES",
        root_symbol="ES",
        price=4800.0,
        asset_class=AssetClass.EQUITY_INDEX,
    )
    state = MarketState.from_snapshots(FIXED_TIMESTAMP, [snapshot])
    weights = EqualWeightAllocator().allocate(("ES",), state)
    assert weights["ES"] == pytest.approx(1.0)


#: Per-instrument observation features used by the tiny DRL fixtures.
_DRL_FEATURE_NAMES: tuple[str, ...] = ("ret_1d", "vol_20d", "term_slope")


def _tiny_drl_env(feature_names: tuple[str, ...] = _DRL_FEATURE_NAMES) -> FuturesTradingEnv:
    """A tiny deterministic 3-asset futures env for DRL smoke tests."""
    rng = np.random.default_rng(7)
    steps = 32
    prices = 100.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.004, (steps, 3)), axis=0)
    features = rng.normal(0.0, 0.01, (steps, 3, len(feature_names)))
    return FuturesTradingEnv(
        prices=prices,
        instrument_ids=("ES", "NQ", "CL"),
        features=features,
        feature_names=feature_names,
        multipliers=(50.0, 20.0, 1000.0),
        tick_sizes=(0.25, 0.25, 0.01),
        initial_margins=(15_000.0, 20_000.0, 5_000.0),
        maintenance_margins=(13_600.0, 18_100.0, 4_500.0),
        commission_bps=2.0,
        max_gross_leverage=2.0,
        drawdown_limit=1.0,
        bankruptcy_threshold=0.0,
    )


def _train_tiny_ppo(
    tmp_path: Path,
    *,
    feature_names: tuple[str, ...] = _DRL_FEATURE_NAMES,
    gross_leverage: float = 2.0,
) -> tuple[DRLAllocator, Path]:
    """Train PPO for a tiny budget on the tiny env and save a checkpoint."""
    allocator = DRLAllocator(
        algorithm="PPO",
        total_timesteps=64,
        device="cpu",
        feature_names=feature_names,
        gross_leverage=gross_leverage,
        seed=7,
    )
    model = allocator.train(_tiny_drl_env(feature_names), total_timesteps=64)
    checkpoint = tmp_path / "drl_checkpoint.zip"
    model.save(str(checkpoint))
    return allocator, checkpoint


@pytest.mark.slow
def test_drl_allocator_trains_ppo_on_futures_env_end_to_end() -> None:
    """Train PPO for a tiny budget and allocate a finite, capped weight vector."""
    pytest.importorskip("stable_baselines3")
    allocator = DRLAllocator(
        algorithm="PPO",
        total_timesteps=128,
        device="cpu",
        feature_names=_DRL_FEATURE_NAMES,
        gross_leverage=2.0,
        seed=7,
    )
    model = allocator.train(_tiny_drl_env(), total_timesteps=128)
    assert model is not None
    assert allocator.is_trained is True
    state = make_market_state(roots=("ES", "NQ", "CL"))
    weights = allocator.allocate(state.ids, state)
    assert isinstance(weights, WeightVector)
    assert set(weights.as_dict()) == set(state.ids)
    assert all(np.isfinite(value) for value in weights.as_dict().values())
    assert weights.gross <= 2.0 + 1e-9


@pytest.mark.slow
def test_drl_allocator_loads_trained_checkpoint_from_model_path(tmp_path: Path) -> None:
    """A saved SB3 checkpoint loads lazily via ``model_path`` and allocates."""
    pytest.importorskip("stable_baselines3")
    _, checkpoint = _train_tiny_ppo(tmp_path)
    allocator = DRLAllocator(
        algorithm="PPO",
        model_path=checkpoint,
        device="cpu",
        feature_names=_DRL_FEATURE_NAMES,
        gross_leverage=2.0,
    )
    assert allocator.model_path == str(checkpoint)
    assert allocator.is_trained is False  # lazy: not loaded until first use
    state = make_market_state(roots=("ES", "NQ", "CL"))
    weights = allocator.allocate(state.ids, state)
    assert allocator.is_trained is True
    assert isinstance(weights, WeightVector)
    assert set(weights.as_dict()) == set(state.ids)
    assert all(np.isfinite(value) for value in weights.as_dict().values())
    assert weights.gross <= 2.0 + 1e-9
    # The loaded policy is deterministic and the checkpoint is only loaded once.
    assert allocator.allocate(state.ids, state) == weights
    with pytest.raises(ValueError, match="model must not be None"):
        allocator.attach_model(None)


def test_drl_allocator_missing_checkpoint_is_actionable(tmp_path: Path) -> None:
    missing = tmp_path / "missing.zip"
    allocator = DRLAllocator(model_path=missing)
    assert allocator.model_path == str(missing)
    with pytest.raises(RuntimeError, match="checkpoint not found") as excinfo:
        _ = allocator.model
    assert str(missing) in str(excinfo.value)
    with pytest.raises(RuntimeError, match="train_drl"):
        allocator.allocate(("ES",), make_market_state(roots=("ES",)))


def test_drl_allocator_checkpoint_requires_sb3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "model.zip"
    checkpoint.write_bytes(b"placeholder")
    monkeypatch.setitem(sys.modules, "stable_baselines3", None)
    allocator = DRLAllocator(model_path=checkpoint)
    with pytest.raises(RuntimeError, match="stable-baselines3"):
        allocator.allocate(("ES",), make_market_state(roots=("ES",)))


def test_drl_allocator_corrupt_checkpoint_is_actionable(tmp_path: Path) -> None:
    pytest.importorskip("stable_baselines3")
    checkpoint = tmp_path / "corrupt.zip"
    checkpoint.write_bytes(b"not a zip file")
    allocator = DRLAllocator(model_path=checkpoint)
    with pytest.raises(RuntimeError, match="failed to load"):
        _ = allocator.model


def test_drl_allocator_rejects_bad_model_path() -> None:
    with pytest.raises(ValueError, match="model_path"):
        DRLAllocator(model_path="  ")
    with pytest.raises(TypeError, match="model_path"):
        DRLAllocator(model_path=123)  # type: ignore[arg-type]


def test_build_allocator_passes_model_path(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.zip"
    checkpoint.write_bytes(b"placeholder")
    allocator = build_allocator("drl", {"algorithm": "PPO", "model_path": str(checkpoint)})
    assert isinstance(allocator, DRLAllocator)
    assert allocator.model_path == str(checkpoint)
    assert allocator.is_trained is False  # loading is lazy


@pytest.mark.slow
def test_pipeline_uses_config_drl_model_path_fallback(tmp_path: Path) -> None:
    """``drl.model_path`` in the strategy config backs a checkpoint-less allocator stage."""
    pytest.importorskip("stable_baselines3")
    _, checkpoint = _train_tiny_ppo(
        tmp_path, feature_names=DEFAULT_OBSERVATION_FEATURES, gross_leverage=1.0
    )
    config = StrategyConfig(
        id="drl_model_path_fallback",
        universe=StageConfig(name="liquidity"),
        allocator=StageConfig(name="drl", params={"algorithm": "PPO"}),
        drl=DRLSpec(algorithm="PPO", model_path=str(checkpoint), total_timesteps=64),
    )
    pipeline = build_pipeline(config, load_instruments())
    allocator = pipeline.allocator
    assert isinstance(allocator, DRLAllocator)
    assert allocator.model_path == str(checkpoint)
    state = make_market_state(roots=("ES", "NQ", "CL"))
    weights = allocator.allocate(state.ids, state)
    assert set(weights.as_dict()) == set(state.ids)
    assert all(np.isfinite(value) for value in weights.as_dict().values())
    assert weights.gross <= 1.0 + 1e-9


def test_pipeline_prefers_explicit_allocator_model_path(tmp_path: Path) -> None:
    """An explicit allocator-stage ``model_path`` wins over ``drl.model_path``."""
    explicit = tmp_path / "explicit.zip"
    explicit.write_bytes(b"placeholder")
    config = StrategyConfig(
        id="drl_model_path_explicit",
        universe=StageConfig(name="liquidity"),
        allocator=StageConfig(name="drl", params={"model_path": str(explicit)}),
        drl=DRLSpec(model_path=str(tmp_path / "fallback.zip")),
    )
    pipeline = build_pipeline(config, load_instruments())
    assert isinstance(pipeline.allocator, DRLAllocator)
    assert pipeline.allocator.model_path == str(explicit)
