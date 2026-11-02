"""Unit tests for the Gymnasium ``FuturesTradingEnv`` (Module 2 DRL environment)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from src.core.types import RiskAssessment
from src.strategies.envs import FuturesTradingEnv
from src.strategies.envs.futures_env import OBSERVATION_BOUND

from tests.fixtures.market import make_risk_assessment

PRICES = np.array(
    [
        [100.0, 50.0],
        [110.0, 45.0],
        [100.0, 55.0],
        [105.0, 60.0],
    ],
    dtype=np.float64,
)
IDS = ("ES", "CL")


def _base_env(**overrides: object) -> FuturesTradingEnv:
    params: dict[str, object] = {
        "prices": PRICES,
        "instrument_ids": IDS,
        "multipliers": (1.0, 1.0),
        "tick_sizes": (0.1, 0.1),
        "initial_equity": 1_000_000.0,
        "commission_bps": 10.0,
        "slippage_ticks": 1.0,
        "max_gross_leverage": 2.0,
        "max_instrument_weight": 1.0,
        "max_margin_utilization": 1.0,
        "drawdown_limit": 1.0,
        "bankruptcy_threshold": 0.0,
    }
    params.update(overrides)
    return FuturesTradingEnv(**params)  # type: ignore[arg-type]


def _action(*values: float) -> np.ndarray[tuple[int], np.dtype[np.float32]]:
    return np.array(values, dtype=np.float32)


# ---------------------------------------------------------------------------
# Spaces / reset
# ---------------------------------------------------------------------------


def test_spaces_and_properties() -> None:
    features = np.zeros((PRICES.shape[0], 2, 3), dtype=np.float64)
    env = _base_env(features=features, feature_names=("a", "b", "c"))
    assert env.instrument_ids == IDS
    assert env.n_assets == 2
    assert env.feature_names == ("a", "b", "c")
    assert env.observation_space.shape == (2 * 3 + 2 + 1,)
    assert env.action_space.shape == (2,)
    assert env.observation_space.dtype == np.float32
    assert env.action_space.dtype == np.float32
    assert env.action_space.low.min() == -1.0
    assert env.action_space.high.max() == 1.0
    assert env.metadata == {"render_modes": []}


def test_reset_is_deterministic_and_flat() -> None:
    env = _base_env()
    first, info = env.reset(seed=7)
    assert np.all(np.isfinite(first))
    assert info["equity"] == pytest.approx(1_000_000.0)
    assert info["gross_leverage"] == 0.0
    assert env.equity == pytest.approx(1_000_000.0)
    assert env.peak_equity == pytest.approx(1_000_000.0)
    assert np.array_equal(env.current_weights, np.zeros(2))
    second, _ = env.reset(seed=7)
    assert np.array_equal(first, second)
    third, _ = env.reset()
    assert np.array_equal(first, third)


def test_observation_includes_features_and_equity_ratio() -> None:
    steps = PRICES.shape[0]
    features = np.arange(steps * 2 * 2, dtype=np.float64).reshape(steps, 2, 2)
    env = _base_env(features=features)
    observation, _ = env.reset(seed=1)
    expected = np.concatenate([features[0].reshape(-1), [0.0, 0.0], [1.0]])
    assert np.allclose(observation, expected)
    observation, _, _, _, _ = env.step(_action(0.0, 0.0))
    expected = np.concatenate([features[1].reshape(-1), [0.0, 0.0], [1.0]])
    assert np.allclose(observation, expected)


def test_observation_dimension_without_features() -> None:
    env = _base_env()
    assert env.observation_space.shape == (2 + 1,)
    observation, _ = env.reset(seed=3)
    assert observation.shape == (3,)


# ---------------------------------------------------------------------------
# Finite observation space (no +/-inf bounds) and bound clipping
# ---------------------------------------------------------------------------


def test_observation_space_has_finite_float32_bounds() -> None:
    """The Box must be finite/float32 so gymnasium and sb3 need no inf-bound warnings."""
    env = _base_env()
    space = env.observation_space
    assert OBSERVATION_BOUND == 1_000_000.0
    assert space.dtype == np.float32
    assert space.low.dtype == np.float32
    assert space.high.dtype == np.float32
    assert np.all(np.isfinite(space.low))
    assert np.all(np.isfinite(space.high))
    assert np.all(space.low == -1_000_000.0)
    assert np.all(space.high == 1_000_000.0)
    observation, _ = env.reset(seed=0)
    assert observation.dtype == np.float32
    assert space.contains(observation)
    observation, _, _, _, _ = env.step(_action(1.0, -1.0))
    assert observation.dtype == np.float32
    assert space.contains(observation)


def test_gymnasium_check_env_emits_zero_warnings() -> None:
    """check_env must emit no warnings (the old infinite bounds produced two).

    Gymnasium warns "A Box observation space minimum value is -infinity ..." and
    "... maximum value is infinity ..." for unbounded Box spaces; both must be gone.
    """
    steps = PRICES.shape[0]
    features = np.arange(steps * 2 * 2, dtype=np.float64).reshape(steps, 2, 2)
    for env in (_base_env(features=features, feature_names=("ret_1d", "vol_20d")), _base_env()):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            check_env(env, skip_render_check=True)
        assert [str(warning.message) for warning in caught] == []


def test_extreme_feature_values_are_clipped_into_bounds() -> None:
    steps = PRICES.shape[0]
    extreme = np.empty((steps, 2, 2), dtype=np.float64)
    extreme[..., 0] = 1e30
    extreme[..., 1] = -1e30
    env = _base_env(features=extreme)
    observation, _ = env.reset(seed=0)
    assert env.observation_space.contains(observation)
    assert observation[:4].tolist() == [1e6, -1e6, 1e6, -1e6]
    observation, _, _, _, _ = env.step(_action(1.0, -1.0))
    assert env.observation_space.contains(observation)
    assert observation[:4].tolist() == [1e6, -1e6, 1e6, -1e6]


def test_extreme_equity_ratio_is_clipped_into_bounds() -> None:
    extreme_prices = np.array(
        [[100.0, 50.0], [100.0, 50.0], [1e12, 50.0], [1e12, 50.0]], dtype=np.float64
    )
    env = FuturesTradingEnv(
        prices=extreme_prices,
        instrument_ids=IDS,
        multipliers=(1.0, 1.0),
        tick_sizes=(0.01, 0.01),
        max_gross_leverage=1.0,
        drawdown_limit=1.0,
        bankruptcy_threshold=0.0,
    )
    env.reset(seed=0)
    # The first bar is flat; the +10,000,000,000% jump happens on the second step.
    env.step(_action(1.0, 0.0))
    observation, _, terminated, truncated, info = env.step(_action(1.0, 0.0))
    assert terminated is False
    assert truncated is False
    # The raw equity ratio is astronomically large; the observation is clipped to
    # the finite Box upper bound so it always lies inside the observation space.
    assert info["equity_ratio"] > OBSERVATION_BOUND
    assert observation[-1] == np.float32(OBSERVATION_BOUND)
    assert env.observation_space.contains(observation)


# ---------------------------------------------------------------------------
# Step mechanics
# ---------------------------------------------------------------------------


def test_zero_action_has_no_costs_and_no_pnl() -> None:
    env = _base_env()
    env.reset(seed=0)
    _, reward, terminated, truncated, info = env.step(_action(0.0, 0.0))
    assert reward == pytest.approx(0.0)
    assert info["turnover"] == 0.0
    assert info["transaction_costs"] == 0.0
    assert info["slippage_costs"] == 0.0
    assert info["net_pnl"] == 0.0
    assert info["equity"] == pytest.approx(1_000_000.0)
    assert terminated is False
    assert truncated is False


def test_rebalance_costs_and_pnl_accounting_identity() -> None:
    env = _base_env()
    env.reset(seed=0)
    _, reward, _, _, info = env.step(_action(1.0, -1.0))
    # turnover 2 -> 10 bps commission on 2.0 notional of 1m equity = 2_000
    assert info["transaction_costs"] == pytest.approx(2_000.0)
    # slippage: 1 tick * 0.1 * 1m / price, both legs
    expected_slippage = 1_000_000.0 * 0.1 / 100.0 + 1_000_000.0 * 0.1 / 50.0
    assert info["slippage_costs"] == pytest.approx(expected_slippage)
    equity_after = 1_000_000.0 - 2_000.0 - expected_slippage
    expected_pnl = equity_after * (0.10 + 0.10)
    expected_equity = equity_after + expected_pnl
    assert info["gross_pnl"] == pytest.approx(expected_pnl)
    assert info["net_pnl"] == pytest.approx(expected_pnl - 2_000.0 - expected_slippage)
    assert info["equity"] == pytest.approx(expected_equity)
    assert info["equity"] == pytest.approx(1_000_000.0 + info["net_pnl"])
    assert reward == pytest.approx(info["net_pnl"] / 1_000_000.0)


def test_action_is_clipped_to_instrument_and_gross_caps() -> None:
    env = _base_env(max_gross_leverage=1.0)
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.array([5.0, 5.0], dtype=np.float32))
    assert info["gross_leverage"] == pytest.approx(1.0)
    assert info["weights"] == pytest.approx((0.5, 0.5))


def test_repeated_weights_incur_no_turnover() -> None:
    env = _base_env()
    env.reset(seed=0)
    env.step(_action(0.5, 0.5))
    _, _, _, _, info = env.step(_action(0.5, 0.5))
    assert info["turnover"] == pytest.approx(0.0)
    assert info["transaction_costs"] == pytest.approx(0.0)


def test_episode_truncates_at_the_end_of_the_price_matrix() -> None:
    env = _base_env()
    env.reset(seed=0)
    truncated = False
    steps = 0
    while not truncated:
        _, _, terminated, truncated, info = env.step(_action(0.0, 0.0))
        steps += 1
        assert not terminated
    assert steps == PRICES.shape[0] - 1
    assert info["step"] == PRICES.shape[0] - 1
    with pytest.raises(RuntimeError, match="after the episode ended"):
        env.step(_action(0.0, 0.0))


def test_drawdown_limit_terminates_the_episode() -> None:
    crash_prices = np.array([[100.0], [80.0], [80.0]], dtype=np.float64)
    env = FuturesTradingEnv(
        prices=crash_prices,
        instrument_ids=("ES",),
        multipliers=(1.0,),
        tick_sizes=(0.01,),
        max_gross_leverage=1.0,
        drawdown_limit=0.05,
    )
    env.reset(seed=0)
    _, _, terminated, truncated, info = env.step(_action(1.0))
    assert terminated is True
    assert truncated is False
    assert info["drawdown"] >= 0.05


def test_bankruptcy_terminates_the_episode() -> None:
    crash_prices = np.array([[100.0], [15.0], [15.0]], dtype=np.float64)
    env = FuturesTradingEnv(
        prices=crash_prices,
        instrument_ids=("ES",),
        multipliers=(1.0,),
        tick_sizes=(0.01,),
        max_gross_leverage=1.0,
        drawdown_limit=1.0,
        bankruptcy_threshold=0.20,
    )
    env.reset(seed=0)
    _, _, terminated, _, info = env.step(_action(1.0))
    assert terminated is True
    assert info["bankrupt"] is True


# ---------------------------------------------------------------------------
# Margin dynamics
# ---------------------------------------------------------------------------


def test_margin_call_scales_the_book_into_the_utilisation_budget() -> None:
    env = FuturesTradingEnv(
        prices=np.array([[100.0], [100.0], [100.0]], dtype=np.float64),
        instrument_ids=("ES",),
        multipliers=(1.0,),
        tick_sizes=(0.01,),
        initial_margins=(1_000_000.0,),
        maintenance_margins=(900_000.0,),
        max_margin_utilization=0.60,
        max_gross_leverage=1.0,
    )
    env.reset(seed=0)
    _, _, terminated, _, info = env.step(_action(1.0))
    assert info["margin_call"] is True
    assert info["margin_utilization"] <= 0.60 + 1e-9
    assert info["margin_required"] == pytest.approx(
        0.60 * (info["equity"] - info["gross_pnl"]), rel=1e-6
    )
    assert terminated is False


def test_forced_liquidation_on_maintenance_margin_breach() -> None:
    env = FuturesTradingEnv(
        prices=np.array([[100.0], [100.0], [100.0]], dtype=np.float64),
        instrument_ids=("ES",),
        multipliers=(1.0,),
        tick_sizes=(0.01,),
        initial_margins=(10.0,),
        maintenance_margins=(100.0,),
        max_margin_utilization=1.0,
        max_instrument_weight=2.0,
        max_gross_leverage=2.0,
        forced_liquidation_penalty_bps=10.0,
        drawdown_limit=1.0,
        bankruptcy_threshold=0.0,
    )
    env.reset(seed=0)
    _, _, terminated, _, info = env.step(_action(2.0))
    assert info["forced_liquidation"] is True
    assert terminated is True
    assert info["weights"] == (0.0,)
    assert info["liquidated_notional"] > 0.0
    assert info["net_pnl"] < 0.0


def test_maintenance_margin_defaults_to_a_fraction_of_initial() -> None:
    env = FuturesTradingEnv(
        prices=np.array([[100.0], [100.0]], dtype=np.float64),
        instrument_ids=("ES",),
        multipliers=(1.0,),
        initial_margins=(1_000.0,),
        maintenance_margin_fraction=0.5,
    )
    assert env  # construction succeeded with maintenance margins derived
    env.reset(seed=0)
    _, _, _, _, info = env.step(_action(1.0))
    assert info["maintenance_margin"] == pytest.approx(0.5 * info["margin_required"])


# ---------------------------------------------------------------------------
# FinRL Risk-First Module 2 reward shaping
# ---------------------------------------------------------------------------


def test_risk_penalty_matches_the_lagrangian_formula() -> None:
    action = _action(0.5, 0.5)
    plain = _base_env(risk_penalty_lambda=0.5)
    plain.reset(seed=0)
    _, reward_plain, _, _, info_plain = plain.step(action)
    assessed = _base_env(risk_penalty_lambda=0.5)
    assessed.set_risk_assessments(
        [
            make_risk_assessment("ES", semantic_risk_score=4.0, confidence=0.9),
            make_risk_assessment("CL", semantic_risk_score=1.0, confidence=0.9),
        ]
    )
    assessed.reset(seed=0)
    assessed.set_risk_assessments(
        [
            make_risk_assessment("ES", semantic_risk_score=4.0, confidence=0.9),
            make_risk_assessment("CL", semantic_risk_score=1.0, confidence=0.9),
        ]
    )
    _, reward_assessed, _, _, info_assessed = assessed.step(action)
    expected_penalty = 0.5 * (0.5 * (4.0 - 1.5) + 0.5 * 0.0)
    assert info_assessed["risk_penalty"] == pytest.approx(expected_penalty)
    assert info_plain["risk_penalty"] == 0.0
    assert reward_plain - reward_assessed == pytest.approx(expected_penalty)


def test_risk_penalty_ignores_low_confidence_and_safe_scores() -> None:
    env = _base_env(risk_penalty_lambda=0.5, confidence_threshold=0.7, safe_risk_score=1.5)
    env.reset(seed=0)
    assessments: list[RiskAssessment] = [
        make_risk_assessment("ES", semantic_risk_score=5.0, confidence=0.2),
        make_risk_assessment("CL", semantic_risk_score=1.5, confidence=0.99),
    ]
    _, _, _, _, info = env.step(_action(0.5, 0.5), risk_assessments=assessments)
    assert info["risk_penalty"] == 0.0


def test_risk_assessments_can_be_injected_through_reset_options() -> None:
    env = _base_env(risk_penalty_lambda=1.0)
    assessments = [make_risk_assessment("ES", semantic_risk_score=3.0, confidence=0.9)]
    env.reset(seed=0, options={"risk_assessments": assessments})
    _, _, _, _, info = env.step(_action(1.0, 0.0))
    assert info["risk_penalty"] == pytest.approx(1.0 * 1.0 * (3.0 - 1.5))


def test_zero_risk_lambda_disables_shaping() -> None:
    env = _base_env(risk_penalty_lambda=0.0)
    env.reset(seed=0)
    assessments = [make_risk_assessment("ES", semantic_risk_score=5.0, confidence=1.0)]
    _, _, _, _, info = env.step(_action(1.0, 0.0), risk_assessments=assessments)
    assert info["risk_penalty"] == 0.0


def test_reward_scaling_multiplies_only_the_pnl_term() -> None:
    scaled = _base_env(reward_scaling=100.0)
    scaled.reset(seed=0)
    _, reward, _, _, info = scaled.step(_action(0.0, 0.0))
    assert reward == pytest.approx(0.0)
    env = _base_env(reward_scaling=2.0)
    env.reset(seed=0)
    _, doubled, _, _, info = env.step(_action(1.0, -1.0))
    assert doubled == pytest.approx(2.0 * info["net_pnl"] / 1_000_000.0)


# ---------------------------------------------------------------------------
# Determinism / compatibility
# ---------------------------------------------------------------------------


def test_episodes_are_deterministic_given_the_same_actions() -> None:
    actions = [_action(0.4, -0.2), _action(0.1, 0.3), _action(-0.5, 0.5)]
    rewards: list[list[float]] = []
    equities: list[list[float]] = []
    for _ in range(2):
        env = _base_env()
        env.reset(seed=11)
        run_rewards: list[float] = []
        run_equity: list[float] = []
        for action in actions:
            _, reward, terminated, truncated, info = env.step(action)
            run_rewards.append(reward)
            run_equity.append(info["equity"])
            if terminated or truncated:
                break
        rewards.append(run_rewards)
        equities.append(run_equity)
    assert rewards[0] == rewards[1]
    assert equities[0] == equities[1]


def test_current_weights_are_read_only() -> None:
    env = _base_env()
    env.reset(seed=0)
    weights = env.current_weights
    with pytest.raises(ValueError, match="read-only"):
        weights[0] = 1.0


def test_gymnasium_check_env_passes() -> None:
    features = np.zeros((PRICES.shape[0], 2, 2), dtype=np.float64)
    env = _base_env(features=features, feature_names=("ret_1d", "vol_20d"))
    check_env(env, skip_render_check=True)
    check_env(_base_env(), skip_render_check=True)


def test_stable_baselines3_check_env_passes() -> None:
    pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.env_checker import check_env as sb3_check_env

    sb3_check_env(_base_env())


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"prices": np.zeros((5, 2))}, "strictly positive"),
        ({"prices": np.array([[1.0, np.nan]])}, "at least two time steps"),
        ({"prices": np.array([1.0, 2.0])}, "2-D"),
        ({"prices": np.array([[1.0], [2.0]]), "instrument_ids": ("ES", "CL")}, "must match prices"),
        ({"instrument_ids": ("ES", "ES")}, "unique"),
        ({"initial_equity": 0.0}, "initial_equity"),
        ({"commission_bps": -1.0}, "commission_bps"),
        ({"slippage_ticks": -1.0}, "slippage_ticks"),
        ({"max_gross_leverage": 0.0}, "max_gross_leverage"),
        ({"max_instrument_weight": 0.0}, "max_instrument_weight"),
        ({"max_margin_utilization": 0.0}, "max_margin_utilization"),
        ({"maintenance_margin_fraction": 1.5}, "maintenance_margin_fraction"),
        ({"drawdown_limit": 0.0}, "drawdown_limit"),
        ({"bankruptcy_threshold": 1.0}, "bankruptcy_threshold"),
        ({"margin_call_penalty_bps": -1.0}, "penalties must be non-negative"),
        ({"risk_penalty_lambda": -1.0}, "risk_penalty_lambda"),
        ({"confidence_threshold": 2.0}, "confidence_threshold"),
        ({"safe_risk_score": 0.5}, "safe_risk_score"),
        ({"reward_scaling": 0.0}, "reward_scaling"),
        ({"multipliers": (1.0,)}, "multipliers"),
        ({"tick_sizes": (0.1,)}, "tick_sizes"),
        ({"initial_margins": (1.0,)}, "initial_margins"),
        ({"initial_margins": (-1.0, -1.0)}, "margins"),
        ({"features": np.zeros((2, 2, 1))}, "features shape"),
        ({"features": np.zeros((2, 2))}, "3-D"),
        (
            {"features": np.full((PRICES.shape[0], 2, 1), np.inf)},
            "features must be finite",
        ),
        (
            {"features": np.zeros((PRICES.shape[0], 2, 2)), "feature_names": ("only_one",)},
            "feature_names",
        ),
    ],
)
def test_constructor_validation(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _base_env(**overrides)


def test_step_rejects_wrong_action_size() -> None:
    env = _base_env()
    env.reset(seed=0)
    with pytest.raises(ValueError, match="action has"):
        env.step(np.zeros(3, dtype=np.float32))
