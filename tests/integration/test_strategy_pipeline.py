"""Integration test: declarative YAML -> ComposablePipeline -> PipelineResult.

This is the Phase 2 acceptance test for the FinRL-X weight-centric contract: the
committed ``configs/strategies/*.yaml`` files must load, build through the stage
registries, and run end-to-end against a deterministic fixture
:class:`~src.core.types.MarketState`, producing a valid :class:`WeightVector` (and
discrete contract targets when a translator is injected).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pytest
from configs.instruments import InstrumentsConfig
from configs.strategy import StrategyConfig, discover_strategy_configs, load_strategy_config
from src.core.interfaces import ContractTranslatorProtocol, RiskOverlay
from src.core.types import (
    ContractTargetSet,
    InstrumentId,
    MarketState,
    WeightVector,
)
from src.strategies import (
    ComposablePipeline,
    DRLAllocator,
    IdentityRiskOverlay,
    build_pipeline,
)
from src.strategies._observation import DEFAULT_OBSERVATION_FEATURES
from src.strategies.envs import FuturesTradingEnv

from tests.fixtures.market import make_contract_targets, make_market_state

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
TSMOM_CONFIG = REPO_ROOT / "configs" / "strategies" / "tsmom_vol_target.yaml"
DRL_CONFIG = REPO_ROOT / "configs" / "strategies" / "drl_ppo_llm_risk.yaml"


class _RecordingTranslator(ContractTranslatorProtocol):
    """Deterministic translator double (``src/risk`` owns the real one)."""

    def __init__(self) -> None:
        self.calls: list[tuple[dict[InstrumentId, float], float | None]] = []

    def translate(
        self,
        weights: WeightVector,
        state: MarketState,
        *,
        equity: float | None = None,
    ) -> ContractTargetSet:
        del state
        self.calls.append((weights.as_dict(), equity))
        quantities = {
            root: (1 if value > 0.0 else -1) for root, value in weights.items() if abs(value) > 1e-6
        }
        return make_contract_targets(quantities)


class _NamedRiskOverlay(RiskOverlay):
    """Named halving risk overlay used to prove injection wiring."""

    def __init__(self, name: str, factor: float) -> None:
        self._name = name
        self._factor = factor

    @property
    def name(self) -> str:
        return self._name

    def apply_risk(self, weights: WeightVector, state: MarketState) -> WeightVector:
        del state
        return weights.scaled(self._factor)


def _assert_valid_result(
    result: object,
    state: MarketState,
    *,
    expected_universe: Sequence[InstrumentId] | None = None,
) -> None:
    from src.core.types import PipelineResult

    assert isinstance(result, PipelineResult)
    assert result.timestamp == state.timestamp
    if expected_universe is not None:
        assert result.universe == tuple(expected_universe)
    assert set(result.universe) <= set(state.instruments)
    for stage in (
        result.base_weights,
        result.timing_weights,
        result.risk_weights,
        result.final_weights,
    ):
        assert isinstance(stage, WeightVector)
        assert all(np.isfinite(value) for value in stage.as_dict().values())
        assert set(stage.as_dict()) <= set(result.universe)
    assert result.elapsed_ms is not None
    assert result.elapsed_ms >= 0.0


# ---------------------------------------------------------------------------
# Classical (non-DRL) pipeline: the Phase 2 acceptance path
# ---------------------------------------------------------------------------


def test_tsmom_yaml_builds_and_runs_end_to_end(
    instrument_master: InstrumentsConfig, market_state: MarketState
) -> None:
    config = load_strategy_config(TSMOM_CONFIG)
    pipeline = build_pipeline(config, instrument_master)
    assert isinstance(pipeline, ComposablePipeline)
    result = pipeline.run(market_state)
    _assert_valid_result(result, market_state)
    assert result.universe == ("NQ", "ES", "GC")
    assert result.base_weights.gross > 0.0
    # The volatility-target allocator caps gross leverage at 3.0 in the YAML.
    assert result.base_weights.gross <= 3.0 + 1e-12
    # No risk overlays were injected: the risk stage is the identity of timing.
    assert result.risk_weights == result.timing_weights
    assert result.final_weights.gross <= result.risk_weights.gross + 1e-12
    assert result.diagnostics["skipped_risk_stages"] == ("quant_risk", "llm_risk")
    assert result.diagnostics["pipeline"] == "tsmom_vol_target"
    assert result.targets is None


def test_tsmom_pipeline_is_reproducible_across_instances(
    instrument_master: InstrumentsConfig,
) -> None:
    config = load_strategy_config(TSMOM_CONFIG)
    state = make_market_state()
    first = build_pipeline(config, instrument_master).run(state)
    second = build_pipeline(config, instrument_master).run(state)
    assert first.universe == second.universe
    for attribute in ("base_weights", "timing_weights", "risk_weights", "final_weights"):
        assert getattr(first, attribute) == getattr(second, attribute)


def test_tsmom_pipeline_attaches_contract_targets(
    instrument_master: InstrumentsConfig, market_state: MarketState
) -> None:
    translator = _RecordingTranslator()
    config = load_strategy_config(TSMOM_CONFIG)
    result = build_pipeline(config, instrument_master, translator=translator).run(market_state)
    _assert_valid_result(result, market_state)
    assert result.targets is not None
    assert isinstance(result.targets, ContractTargetSet)
    assert translator.calls, "translator was never invoked"
    weights_arg, equity_arg = translator.calls[0]
    assert weights_arg == result.final_weights.as_dict()
    assert equity_arg == market_state.equity
    assert result.diagnostics["targets_attached"] is True


def test_tsmom_pipeline_applies_named_risk_overlays_in_config_order(
    instrument_master: InstrumentsConfig, market_state: MarketState
) -> None:
    config = load_strategy_config(TSMOM_CONFIG)
    overlays = [_NamedRiskOverlay("llm_risk", 0.5), _NamedRiskOverlay("quant_risk", 0.5)]
    pipeline = build_pipeline(config, instrument_master, risk_overlays=overlays)
    assert tuple(overlay.name for overlay in pipeline.risk_overlays) == ("quant_risk", "llm_risk")
    result = pipeline.run(market_state)
    _assert_valid_result(result, market_state)
    assert result.diagnostics["risk_overlays"] == ("quant_risk", "llm_risk")
    assert result.diagnostics["skipped_risk_stages"] == ()
    assert result.risk_weights.gross == pytest.approx(0.25 * result.timing_weights.gross)


def test_identity_overlays_are_a_no_op_on_the_yaml_pipeline(
    instrument_master: InstrumentsConfig, market_state: MarketState
) -> None:
    config = load_strategy_config(TSMOM_CONFIG)
    overlays = [IdentityRiskOverlay(), IdentityRiskOverlay()]
    pipeline = build_pipeline(config, instrument_master, risk_overlays=overlays)
    result = pipeline.run(market_state)
    _assert_valid_result(result, market_state)
    assert result.risk_weights == result.timing_weights
    # Identities do not match the declared stage names, so they are reported skipped.
    assert result.diagnostics["risk_overlays"] == ("identity", "identity")
    assert result.diagnostics["skipped_risk_stages"] == ("quant_risk", "llm_risk")


def test_all_committed_strategy_configs_load_and_build(
    instrument_master: InstrumentsConfig,
) -> None:
    configs = discover_strategy_configs(REPO_ROOT / "configs" / "strategies")
    assert set(configs) == {"tsmom_vol_target", "drl_ppo_llm_risk"}
    for config in configs.values():
        pipeline = build_pipeline(config, instrument_master)
        assert pipeline.name == config.id
    # The DRL YAML is declarative and builds lazily: the optional torch stack is
    # only needed once a model is trained/allocated.
    drl_pipeline = build_pipeline(load_strategy_config(DRL_CONFIG), instrument_master)
    assert isinstance(drl_pipeline.allocator, DRLAllocator)
    with pytest.raises(RuntimeError, match="no model"):
        drl_pipeline.run(make_market_state())


def test_pipeline_tolerates_missing_covariance_and_returns(
    instrument_master: InstrumentsConfig,
) -> None:
    config = load_strategy_config(TSMOM_CONFIG)
    bare = make_market_state(with_covariance=False, with_returns=False)
    result = build_pipeline(config, instrument_master).run(bare)
    _assert_valid_result(result, bare)
    assert result.universe  # liquidity selector works from snapshots alone
    assert all(np.isfinite(value) for value in result.final_weights.as_dict().values())


# ---------------------------------------------------------------------------
# Cross-module: DRL observation contract shared by env and allocator
# ---------------------------------------------------------------------------


def test_env_and_drl_allocator_share_the_observation_layout(
    market_state: MarketState,
) -> None:
    roots = market_state.ids
    feature_names = DEFAULT_OBSERVATION_FEATURES
    steps = 8
    closes = np.array([[market_state.price(root) for root in roots]] * steps, dtype=np.float64)
    feature_row = np.array(
        [
            [
                market_state.instruments[root].feature(feature_name, 0.0)
                for feature_name in feature_names
            ]
            for root in roots
        ],
        dtype=np.float64,
    )
    features = np.repeat(feature_row[None, :, :], steps, axis=0)
    env = FuturesTradingEnv(
        prices=closes,
        instrument_ids=roots,
        features=features,
        feature_names=feature_names,
    )
    allocator = DRLAllocator(feature_names=feature_names)
    observation = allocator.build_observation(market_state, roots)
    env_observation, _ = env.reset(seed=0)
    assert observation.shape == env_observation.shape == env.observation_space.shape
    assert np.allclose(observation, env_observation)


def test_drl_config_params_are_consumed_by_the_registry(
    instrument_master: InstrumentsConfig,
) -> None:
    config: StrategyConfig = load_strategy_config(DRL_CONFIG)
    assert config.allocator.name == "drl"
    allocator = build_pipeline(config, instrument_master).allocator
    assert isinstance(allocator, DRLAllocator)
    declared: Mapping[str, object] = config.allocator.params
    assert allocator.algorithm == declared["algorithm"]
    assert allocator.total_timesteps == declared["total_timesteps"]
