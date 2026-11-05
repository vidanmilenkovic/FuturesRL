"""Integration tests for the FuturesRL risk engine.

These tests exercise the full ``R_t -> N_i,t`` chain built from the Tier-A LLM
Risk-First overlay, the Tier-B quantitative overlay and the deterministic
contract translator, plus their wiring to the declarative strategy configs. All
tests are offline and deterministic.
"""

from __future__ import annotations

import json

import pytest
from configs.instruments import load_instruments
from configs.strategy import discover_strategy_configs
from src.core.interfaces import RiskOverlay
from src.core.types import RiskAssessment, RoundingMode, WeightVector
from src.risk import (
    ContractTranslator,
    LLMRiskConfig,
    LLMRiskOverlay,
    MacroEventCalendar,
    QuantRiskConfig,
    QuantRiskOverlay,
    semantic_risk_penalty,
)

from tests.fixtures.market import (
    FIXED_TIMESTAMP,
    make_market_state,
    make_risk_assessment,
    make_weight_vector,
)

pytestmark = pytest.mark.integration


class FakeLLMClient:
    """Deterministic AsyncLLMClient used by the sync overlay tests."""

    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)

    async def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        return self.responses[0]


def _json_batch(records: list[tuple[str, float, float]]) -> str:
    return json.dumps(
        {
            "assessments": [
                {
                    "instrument_id": root,
                    "sentiment_score": 3.0,
                    "semantic_risk_score": score,
                    "confidence": confidence,
                    "reasoning_summary": "integration fixture",
                }
                for root, score, confidence in records
            ]
        }
    )


def _neutral_assessments(roots: tuple[str, ...], score: float = 2.0) -> list[RiskAssessment]:
    return [make_risk_assessment(root, semantic_risk_score=score, confidence=0.9) for root in roots]


def _quant_overlay() -> QuantRiskOverlay:
    return QuantRiskOverlay(
        QuantRiskConfig(
            volatility_targeting_enabled=False,
            max_gross_leverage=2.5,
            max_single_instrument_weight=0.5,
            max_sector_weight=0.5,
            daily_max_loss_pct=0.5,
            trailing_stop_pct=0.5,
            drawdown_limit_pct=0.5,
        )
    )


def _llm_overlay(config: LLMRiskConfig | None = None) -> LLMRiskOverlay:
    client = FakeLLMClient(_json_batch([("ES", 2.0, 0.9)]))
    return LLMRiskOverlay(client, config or LLMRiskConfig(sample_count=1))


# ---------------------------------------------------------------------------
# Full overlay -> translator chain
# ---------------------------------------------------------------------------


def test_full_risk_pipeline_translates_to_feasible_contracts() -> None:
    state = make_market_state(equity=1_000_000.0)
    weights = make_weight_vector({"ES": 0.4, "NQ": 0.25, "CL": -0.15, "GC": 0.1})
    llm = _llm_overlay()
    llm.set_cached_assessments(_neutral_assessments(tuple(weights)))
    quant = _quant_overlay()
    translator = ContractTranslator(rounding_mode=RoundingMode.NEAREST)

    llm_weights = llm.apply_risk(weights, state)
    # theta_safe < 2.0 < kappa -> every weight scaled by (4.5 - 2.0) / (4.5 - 1.5).
    for root, value in weights.items():
        assert llm_weights[root] == pytest.approx(value * (2.5 / 3.0))
    risk_weights = quant.apply_risk(llm_weights, state)
    assert risk_weights.gross <= 2.5 + 1e-12
    targets = translator.translate(risk_weights, state)

    assert targets.gross_notional > 0.0
    assert translator.margin_utilization(targets, state.equity) <= 0.60 + 1e-12
    # Deterministic: replaying the same final weights yields identical contracts.
    assert translator.translate(risk_weights, state) == targets
    # The reward-shaping penalty exposed to the DRL environment is consistent.
    assert llm.penalty(weights) == pytest.approx(
        semantic_risk_penalty(weights, llm.latest_assessments, llm.config.penalty_lambda, 1.5, 0.70)
    )


def test_critical_llm_alert_flattens_translated_book() -> None:
    state = make_market_state(equity=1_000_000.0)
    weights = make_weight_vector({"ES": 0.5, "CL": 0.2})
    llm = _llm_overlay()
    llm.set_cached_assessments(
        [
            make_risk_assessment("ES", semantic_risk_score=5.0, confidence=0.95),
            make_risk_assessment("CL", semantic_risk_score=1.2, confidence=0.95),
        ]
    )
    risk_weights = _quant_overlay().apply_risk(llm.apply_risk(weights, state), state)
    assert risk_weights["ES"] == 0.0
    targets = ContractTranslator().translate(risk_weights, state)
    assert targets.get("ES") is None
    assert targets.get("MES") is None
    assert targets.get("CL") is not None


def test_macro_lockout_flattens_book_end_to_end() -> None:
    state = make_market_state(equity=1_000_000.0)
    event = (("FOMC", FIXED_TIMESTAMP, 1),)
    calendar = MacroEventCalendar.from_events(event)
    llm = LLMRiskOverlay(
        FakeLLMClient(_json_batch([("ES", 2.0, 0.9)])),
        LLMRiskConfig(
            sample_count=1,
            macro_lockout_action="flatten",
            macro_lockout_minutes_before=30,
            macro_lockout_minutes_after=30,
        ),
        calendar=calendar,
    )
    targets = ContractTranslator().translate(
        llm.apply_risk(make_weight_vector({"ES": 0.5}), state), state
    )
    assert len(targets) == 0


def test_quant_daily_kill_switch_flattens_book_end_to_end() -> None:
    quant = QuantRiskOverlay(
        QuantRiskConfig(
            volatility_targeting_enabled=False,
            max_gross_leverage=2.5,
            max_single_instrument_weight=0.5,
            max_sector_weight=0.5,
            daily_max_loss_pct=0.03,
            trailing_stop_pct=0.5,
            drawdown_limit_pct=0.5,
        )
    )
    quant.update_equity(1_000_000.0, timestamp=FIXED_TIMESTAMP)
    quant.update_equity(960_000.0, timestamp=FIXED_TIMESTAMP)
    state = make_market_state(equity=960_000.0)
    assert quant.kill_switch_active
    targets = ContractTranslator().translate(
        quant.apply_risk(make_weight_vector({"ES": 0.5, "NQ": 0.3}), state), state
    )
    assert len(targets) == 0


def test_small_equity_micro_routing_end_to_end() -> None:
    state = make_market_state(equity=200_000.0)
    llm = _llm_overlay()
    llm.set_cached_assessments([make_risk_assessment("ES", semantic_risk_score=1.2)])
    quant = QuantRiskOverlay(
        QuantRiskConfig(
            volatility_targeting_enabled=False,
            max_gross_leverage=3.0,
            max_single_instrument_weight=0.5,
            max_sector_weight=0.5,
            daily_max_loss_pct=0.5,
            trailing_stop_pct=0.5,
            drawdown_limit_pct=0.5,
        )
    )
    translator = ContractTranslator(rounding_mode=RoundingMode.NEAREST)
    final_weights = quant.apply_risk(llm.apply_risk(make_weight_vector({"ES": 0.30}), state), state)
    assert final_weights["ES"] == pytest.approx(0.30)
    targets = translator.translate(final_weights, state)
    # 60,000 USD target cannot hold an ES contract: routed into 3 MES contracts.
    assert targets.quantities == {"MES": 3}
    assert translator.margin_utilization(targets, state.equity) == pytest.approx(
        4_500.0 / 200_000.0
    )


def test_translator_margin_ceiling_holds_across_equity_shocks() -> None:
    translator = ContractTranslator(max_margin_utilization=0.60)
    weights = make_weight_vector({"ES": 1.5, "NQ": 1.0, "CL": -0.75, "GC": 0.5, "ZN": -0.25})
    for equity in (50_000.0, 250_000.0, 1_000_000.0, 5_000_000.0):
        state = make_market_state(equity=equity)
        targets = translator.translate(weights, state)
        assert translator.margin_utilization(targets, equity) <= 0.60 + 1e-12
        # A standard/micro family never hedges itself, even after scaling.
        for standard_root, micro_root in (
            ("ES", "MES"),
            ("NQ", "MNQ"),
            ("CL", "MCL"),
            ("GC", "MGC"),
        ):
            quantities = [
                target.quantity
                for target in targets
                if target.instrument_id in (standard_root, micro_root)
            ]
            assert not (
                any(quantity > 0 for quantity in quantities)
                and any(quantity < 0 for quantity in quantities)
            )


# ---------------------------------------------------------------------------
# Declarative strategy wiring (production YAMLs)
# ---------------------------------------------------------------------------


def test_production_yaml_risk_stage_names_match_overlay_names() -> None:
    configs = discover_strategy_configs()
    declared = {stage.name for strategy in configs.values() for stage in strategy.risk}
    assert declared == {"quant_risk", "llm_risk"}
    overlays: tuple[RiskOverlay, ...] = (_quant_overlay(), _llm_overlay())
    assert {overlay.name for overlay in overlays} == declared
    master = load_instruments()
    for strategy in configs.values():
        # The production translator config must build from the frozen config types.
        translator = ContractTranslator.from_config(strategy.translator, instruments=master)
        assert translator.max_margin_utilization == strategy.translator.max_margin_utilization


def test_composable_pipeline_applies_named_risk_overlays() -> None:
    from src.strategies.pipeline import build_pipeline

    configs = discover_strategy_configs()
    config = configs["tsmom_vol_target"]
    state = make_market_state(equity=1_000_000.0)
    overlays: tuple[RiskOverlay, ...] = (_quant_overlay(), _llm_overlay())
    pipeline = build_pipeline(
        config,
        load_instruments(),
        translator=ContractTranslator(),
        risk_overlays=overlays,
    )
    assert tuple(overlay.name for overlay in pipeline.risk_overlays) == (
        "quant_risk",
        "llm_risk",
    )
    result = pipeline.run(state)
    assert result.diagnostics["skipped_risk_stages"] == ()
    assert result.diagnostics["risk_overlays"] == ("quant_risk", "llm_risk")
    assert result.timestamp == state.timestamp
    assert result.targets is not None
    assert result.final_weights.gross <= 2.5 + 1e-12


def test_risk_pipeline_is_deterministic() -> None:
    state = make_market_state(equity=1_000_000.0)
    weights = make_weight_vector({"ES": 0.4, "NQ": -0.2, "CL": 0.1})

    def run_once() -> tuple[WeightVector, object]:
        llm = _llm_overlay()
        llm.set_cached_assessments(_neutral_assessments(tuple(weights)))
        quant = _quant_overlay()
        translator = ContractTranslator()
        final = quant.apply_risk(llm.apply_risk(weights, state), state)
        return final, translator.translate(final, state)

    first_weights, first_targets = run_once()
    second_weights, second_targets = run_once()
    assert first_weights == second_weights
    assert first_targets == second_targets
