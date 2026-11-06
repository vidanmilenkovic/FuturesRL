"""Serialization tests for modular composite strategy configurations (Workstation UX)."""

from __future__ import annotations

import pytest
from configs.instruments import load_instruments
from configs.strategy import (
    BacktestSpec,
    StageConfig,
    StrategyConfig,
    TranslatorConfig,
    parse_strategy_config,
)
from pydantic import ValidationError
from src.strategies.pipeline import build_pipeline

from tests.fixtures.market import make_market_state


def _modular_strategy() -> StrategyConfig:
    """A valid pipeline with independently parameterized timing overlays and a composite universe."""
    return StrategyConfig(
        id="studio_modular_composite",
        description="Manual Builder modular composite serialization test.",
        universe=StageConfig(
            name="composite",
            params={
                "components": "liquidity,regime",
                "mode": "intersection",
                "roots": "ES,NQ,CL,GC",
            },
        ),
        allocator=StageConfig(
            name="volatility_target",
            params={"target_annual_volatility": 0.12, "max_gross_leverage": 2.0},
        ),
        timing=(
            StageConfig(name="tsmom", params={"lookback_fast": 10, "lookback_slow": 80}),
            StageConfig(name="kama", params={"window": 12, "fast_period": 3, "slow_period": 40}),
            StageConfig(name="carry_filter", params={"min_carry_bps": -50.0, "carry_weight": 1.5}),
        ),
        risk=(StageConfig(name="quant_risk", params={"vol_target": 0.10}),),
        translator=TranslatorConfig(),
        backtest=BacktestSpec(start="2025-01-02", end="2025-12-31"),
    )


def test_composite_timing_stack_keeps_independent_params() -> None:
    strategy = _modular_strategy()
    timing = {stage.name: dict(stage.params) for stage in strategy.timing}
    assert timing["tsmom"]["lookback_fast"] == 10
    assert timing["kama"]["window"] == 12
    assert timing["carry_filter"]["carry_weight"] == 1.5
    # Each overlay keeps its own parameter dictionary: they are separate stages, not merged.
    assert set(timing["tsmom"]) != set(timing["kama"])
    assert len(strategy.model_dump()["timing"]) == 3


def test_strategy_round_trips_through_payload_and_yaml_parser() -> None:
    strategy = _modular_strategy()
    payload = strategy.model_dump()
    reparsed = parse_strategy_config(payload)
    assert reparsed.timing == strategy.timing
    assert reparsed.universe.name == "composite"
    assert reparsed.universe.params["components"] == "liquidity,regime"
    assert reparsed.universe.params["mode"] == "intersection"
    assert reparsed.model_dump() == strategy.model_dump()


def test_composite_universe_config_builds_a_valid_pipeline() -> None:
    strategy = _modular_strategy()
    pipeline = build_pipeline(strategy, load_instruments())
    assert pipeline.name == strategy.id
    assert pipeline.selector.name == "composite"
    result = pipeline.run(make_market_state(roots=("ES", "NQ", "CL", "GC")))
    assert result.universe
    assert set(result.final_weights.weights) <= set(result.universe)


def test_timing_stage_params_must_remain_scalar_for_yaml_serialization() -> None:
    strategy = _modular_strategy()
    with pytest.raises(ValidationError):
        # A nested parameter object is not a valid declarative stage scalar.
        StageConfig(name="tsmom", params={"lookback_fast": {"nested": 1}})  # type: ignore[dict-item]
    payload = strategy.model_dump()
    for stage in payload["timing"]:
        for value in stage["params"].values():
            assert isinstance(value, (str, int, float, bool))
