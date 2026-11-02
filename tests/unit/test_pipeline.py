"""Unit tests for the FinRL-X ``ComposablePipeline`` (``w_t = R_t(T_t(A_t(S_t(X))))``)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest
from configs.instruments import InstrumentsConfig
from configs.strategy import StageConfig, StrategyConfig, load_strategy_config
from src.core.interfaces import StrategyPipeline
from src.core.types import (
    ContractTargetSet,
    InstrumentId,
    MarketState,
    PipelineResult,
    WeightVector,
)
from src.strategies._params import ParamError, ParamReader
from src.strategies.allocators import DRLAllocator
from src.strategies.pipeline import (
    ComposablePipeline,
    IdentityRiskOverlay,
    build_pipeline,
)

from tests.fixtures.market import make_contract_targets, make_market_state


class _StaticSelector:
    def __init__(self, instruments: Sequence[InstrumentId], name: str = "static") -> None:
        self._instruments = tuple(instruments)
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def select(self, state: MarketState) -> Sequence[InstrumentId]:
        del state
        return self._instruments


class _NonSequenceSelector:
    @property
    def name(self) -> str:
        return "non-sequence"

    def select(self, state: MarketState) -> Sequence[InstrumentId]:
        del state
        return 42  # type: ignore[return-value]


class _StaticAllocator:
    def __init__(self, weights: Mapping[InstrumentId, float], name: str = "static") -> None:
        self._weights = dict(weights)
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def allocate(self, candidates: Sequence[InstrumentId], state: MarketState) -> WeightVector:
        del candidates, state
        return WeightVector(self._weights)


class _ScalingTiming:
    def __init__(self, factor: float, name: str = "scale", calls: list[str] | None = None) -> None:
        self._factor = factor
        self._name = name
        self._calls = calls

    @property
    def name(self) -> str:
        return self._name

    def adjust_timing(self, weights: WeightVector, state: MarketState) -> WeightVector:
        del state
        if self._calls is not None:
            self._calls.append(self._name)
        return weights.scaled(self._factor)


class _ScalingRisk:
    def __init__(self, factor: float, name: str = "risk", calls: list[str] | None = None) -> None:
        self._factor = factor
        self._name = name
        self._calls = calls

    @property
    def name(self) -> str:
        return self._name

    def apply_risk(self, weights: WeightVector, state: MarketState) -> WeightVector:
        del state
        if self._calls is not None:
            self._calls.append(self._name)
        return weights.scaled(self._factor)


class _FakeTranslator:
    def __init__(self) -> None:
        self.calls: list[tuple[WeightVector, float | None]] = []

    def translate(
        self,
        weights: WeightVector,
        state: MarketState,
        *,
        equity: float | None = None,
    ) -> ContractTargetSet:
        del state
        self.calls.append((weights, equity))
        quantities = {root: 1 for root, value in weights.items() if value > 0.0}
        return make_contract_targets(quantities)


def _pipeline(**overrides: object) -> ComposablePipeline:
    params: dict[str, object] = {
        "name": "unit-pipeline",
        "selector": _StaticSelector(("ES", "NQ")),
        "allocator": _StaticAllocator({"ES": 0.5, "NQ": -0.2}),
    }
    params.update(overrides)
    return ComposablePipeline(**params)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_pipeline_stage_chain_and_trace() -> None:
    translator = _FakeTranslator()
    pipeline = _pipeline(
        timing=(_ScalingTiming(2.0, "double"),),
        risk=(_ScalingRisk(0.5, "half"),),
        translator=translator,
    )
    assert isinstance(pipeline, StrategyPipeline)
    assert pipeline.name == "unit-pipeline"
    result = pipeline.run(make_market_state())
    assert isinstance(result, PipelineResult)
    assert result.universe == ("ES", "NQ")
    assert result.base_weights == {"ES": 0.5, "NQ": -0.2}
    assert result.timing_weights == {"ES": 1.0, "NQ": -0.4}
    assert result.risk_weights == {"ES": 0.5, "NQ": -0.2}
    assert result.final_weights == {"ES": 0.5, "NQ": -0.2}
    assert result.targets is not None
    assert result.elapsed_ms is not None and result.elapsed_ms >= 0.0
    assert result.timestamp == make_market_state().timestamp
    assert translator.calls[0][0] == result.final_weights
    assert translator.calls[0][1] == make_market_state().equity


def test_pipeline_diagnostics_are_scalar_friendly() -> None:
    result = _pipeline().run(make_market_state())
    diagnostics = result.diagnostics
    assert diagnostics["pipeline"] == "unit-pipeline"
    assert diagnostics["selector"] == "static"
    assert diagnostics["allocator"] == "static"
    assert diagnostics["universe_size"] == 2
    assert diagnostics["empty_universe"] is False
    assert diagnostics["targets_attached"] is False
    notes = diagnostics["stage_notes"]
    timings = diagnostics["stage_timings_ms"]
    assert isinstance(notes, Mapping)
    assert isinstance(timings, Mapping)
    assert set(notes) == {"selector.static", "allocator.static"}
    assert all(isinstance(value, str) for value in notes.values())
    assert all(float(value) >= 0.0 for value in timings.values())
    assert diagnostics["final_gross"] == pytest.approx(0.7)
    assert diagnostics["final_net"] == pytest.approx(0.3)


def test_identity_risk_overlay_is_a_true_pass_through() -> None:
    overlay = IdentityRiskOverlay()
    weights = WeightVector({"ES": 0.3})
    state = make_market_state(roots=("ES",))
    assert overlay.name == "identity"
    assert overlay.apply_risk(weights, state) is weights
    result = _pipeline(risk=(overlay,)).run(make_market_state())
    assert result.risk_weights == result.timing_weights


def test_overlays_are_applied_in_configured_order() -> None:
    calls: list[str] = []
    pipeline = _pipeline(
        timing=(
            _ScalingTiming(1.0, "timing-a", calls),
            _ScalingTiming(1.0, "timing-b", calls),
        ),
        risk=(_ScalingRisk(1.0, "risk-a", calls), _ScalingRisk(1.0, "risk-b", calls)),
    )
    pipeline.run(make_market_state())
    assert calls == ["timing-a", "timing-b", "risk-a", "risk-b"]
    assert pipeline.stage_names == (
        "selector.static",
        "allocator.static",
        "timing[0].timing-a",
        "timing[1].timing-b",
        "risk[0].risk-a",
        "risk[1].risk-b",
    )


def test_pipeline_run_is_deterministic() -> None:
    pipeline = _pipeline(
        timing=(_ScalingTiming(1.5, "scale"),),
        risk=(_ScalingRisk(0.8, "risk"),),
    )
    state = make_market_state()
    first = pipeline.run(state)
    second = pipeline.run(state)
    assert first.universe == second.universe
    for attribute in ("base_weights", "timing_weights", "risk_weights", "final_weights"):
        assert getattr(first, attribute) == getattr(second, attribute)


def test_empty_universe_produces_a_flat_book_and_skips_overlays() -> None:
    translator = _FakeTranslator()
    pipeline = _pipeline(
        selector=_StaticSelector(()),
        timing=(_ScalingTiming(2.0),),
        risk=(_ScalingRisk(2.0),),
        translator=translator,
    )
    result = pipeline.run(make_market_state())
    assert result.universe == ()
    assert result.final_weights == WeightVector.empty()
    assert result.base_weights == WeightVector.empty()
    assert result.diagnostics["empty_universe"] is True
    assert result.diagnostics["overlays_skipped"] is True
    assert translator.calls[0][0] == WeightVector.empty()


def test_negligible_weights_are_sanitised_out() -> None:
    pipeline = _pipeline(
        allocator=_StaticAllocator({"ES": 0.5, "NQ": 1e-12, "CL": 0.25}),
        selector=_StaticSelector(("ES", "NQ", "CL")),
    )
    result = pipeline.run(make_market_state())
    assert result.final_weights == {"ES": 0.5, "CL": 0.25}
    assert "NQ" not in result.final_weights
    assert result.diagnostics["sanitised_weights"] == 1


def test_no_translator_means_no_targets() -> None:
    result = _pipeline().run(make_market_state())
    assert result.targets is None
    assert result.diagnostics["targets_attached"] is False


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_constructor_validation() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        _pipeline(name="  ")
    with pytest.raises(TypeError, match="UniverseSelector"):
        _pipeline(selector=object())
    with pytest.raises(TypeError, match="PortfolioAllocator"):
        _pipeline(allocator=object())
    with pytest.raises(TypeError, match="TimingOverlay"):
        _pipeline(timing=(object(),))
    with pytest.raises(TypeError, match="RiskOverlay"):
        _pipeline(risk=(object(),))
    with pytest.raises(ValueError, match="negligible_weight"):
        _pipeline(negligible_weight=-1.0)


def test_run_rejects_non_market_state() -> None:
    with pytest.raises(TypeError, match="MarketState"):
        _pipeline().run(object())  # type: ignore[arg-type]


def test_selector_validation() -> None:
    with pytest.raises(ValueError, match="unknown instrument"):
        _pipeline(selector=_StaticSelector(("ZZ",))).run(make_market_state())
    with pytest.raises(ValueError, match="duplicate instrument"):
        _pipeline(selector=_StaticSelector(("ES", "ES"))).run(make_market_state())
    with pytest.raises(TypeError, match="sequence of instrument ids"):
        _pipeline(selector=_NonSequenceSelector()).run(make_market_state())
    with pytest.raises(ValueError, match="invalid instrument id"):
        _pipeline(selector=_StaticSelector(("",))).run(make_market_state())


def test_allocator_output_validation() -> None:
    unknown = _pipeline(allocator=_StaticAllocator({"ZZ": 1.0}))
    with pytest.raises(ValueError, match="not part of the market state"):
        unknown.run(make_market_state())
    outside = _pipeline(allocator=_StaticAllocator({"CL": 1.0}))
    with pytest.raises(ValueError, match="outside the selected universe"):
        outside.run(make_market_state())
    with pytest.raises(TypeError, match="must return a WeightVector"):
        _pipeline(allocator=_MappingAllocator()).run(make_market_state())


class _MappingAllocator:
    @property
    def name(self) -> str:
        return "mapping"

    def allocate(self, candidates: Sequence[InstrumentId], state: MarketState) -> WeightVector:
        del candidates, state
        return {"ES": 0.5}  # type: ignore[return-value]


def test_overlay_output_validation() -> None:
    with pytest.raises(TypeError, match="must return a WeightVector"):
        _pipeline(timing=(_MappingTiming(),)).run(make_market_state())
    with pytest.raises(TypeError, match="must return a WeightVector"):
        _pipeline(risk=(_MappingRisk(),)).run(make_market_state())


class _MappingTiming:
    @property
    def name(self) -> str:
        return "mapping-timing"

    def adjust_timing(self, weights: WeightVector, state: MarketState) -> WeightVector:
        del weights, state
        return {"ES": 0.1}  # type: ignore[return-value]


class _MappingRisk:
    @property
    def name(self) -> str:
        return "mapping-risk"

    def apply_risk(self, weights: WeightVector, state: MarketState) -> WeightVector:
        del weights, state
        return {"ES": 0.1}  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Config-driven construction
# ---------------------------------------------------------------------------


def test_build_pipeline_wires_registries_and_skips_missing_risk(
    instrument_master: InstrumentsConfig,
) -> None:
    config = load_strategy_config("configs/strategies/tsmom_vol_target.yaml")
    pipeline = build_pipeline(config, instrument_master)
    assert pipeline.name == config.id
    assert pipeline.selector.name == "liquidity"
    assert pipeline.allocator.name == "volatility_target"
    assert tuple(overlay.name for overlay in pipeline.timing_overlays) == (
        "tsmom",
        "kama",
        "carry_filter",
    )
    assert pipeline.risk_overlays == ()
    assert pipeline.translator is None
    result = pipeline.run(make_market_state())
    assert result.diagnostics["skipped_risk_stages"] == ("quant_risk", "llm_risk")


def test_build_pipeline_matches_risk_overlays_by_name(
    instrument_master: InstrumentsConfig,
) -> None:
    config = load_strategy_config("configs/strategies/tsmom_vol_target.yaml")
    quant = _ScalingRisk(0.5, "quant_risk")
    llm = _ScalingRisk(0.5, "llm_risk")
    pipeline = build_pipeline(
        config, instrument_master, translator=None, risk_overlays=[llm, quant]
    )
    # Config order wins, not injection order.
    assert tuple(overlay.name for overlay in pipeline.risk_overlays) == ("quant_risk", "llm_risk")
    result = pipeline.run(make_market_state())
    assert result.diagnostics["skipped_risk_stages"] == ()
    assert result.diagnostics["risk_overlays"] == ("quant_risk", "llm_risk")


def test_build_pipeline_falls_back_to_injected_overlay_order(
    instrument_master: InstrumentsConfig,
) -> None:
    config = load_strategy_config("configs/strategies/tsmom_vol_target.yaml")
    first = _ScalingRisk(1.0, "custom-a")
    second = _ScalingRisk(1.0, "custom-b")
    pipeline = build_pipeline(config, instrument_master, risk_overlays=[first, second])
    assert tuple(overlay.name for overlay in pipeline.risk_overlays) == ("custom-a", "custom-b")
    result = pipeline.run(make_market_state())
    # No declared stage name matched, so the declared stages are reported skipped.
    assert result.diagnostics["skipped_risk_stages"] == ("quant_risk", "llm_risk")


def test_build_pipeline_attaches_the_translator(instrument_master: InstrumentsConfig) -> None:
    config = load_strategy_config("configs/strategies/tsmom_vol_target.yaml")
    translator = _FakeTranslator()
    pipeline = build_pipeline(config, instrument_master, translator=translator)
    assert pipeline.translator is translator
    result = pipeline.run(make_market_state())
    assert result.targets is not None
    assert result.diagnostics["targets_attached"] is True


def test_build_pipeline_rejects_unknown_stage_names(instrument_master: InstrumentsConfig) -> None:
    unknown_selector = StrategyConfig(
        id="bad-pipeline",
        universe=StageConfig(name="does_not_exist"),
        allocator=StageConfig(name="equal_weight"),
    )
    with pytest.raises(ValueError, match="unknown universe selector"):
        build_pipeline(unknown_selector, instrument_master)
    unknown_allocator = StrategyConfig(
        id="bad-pipeline",
        universe=StageConfig(name="liquidity"),
        allocator=StageConfig(name="does_not_exist"),
    )
    with pytest.raises(ValueError, match="unknown portfolio allocator"):
        build_pipeline(unknown_allocator, instrument_master)
    unknown_timing = StrategyConfig(
        id="bad-pipeline",
        universe=StageConfig(name="liquidity"),
        allocator=StageConfig(name="equal_weight"),
        timing=(StageConfig(name="does_not_exist"),),
    )
    with pytest.raises(ValueError, match="unknown timing overlay"):
        build_pipeline(unknown_timing, instrument_master)


def test_drl_strategy_builds_lazily_and_fails_clearly_at_allocation(
    instrument_master: InstrumentsConfig,
) -> None:
    config = load_strategy_config("configs/strategies/drl_ppo_llm_risk.yaml")
    pipeline = build_pipeline(config, instrument_master)
    assert isinstance(pipeline.allocator, DRLAllocator)
    with pytest.raises(RuntimeError, match="no model"):
        pipeline.run(make_market_state())


# ---------------------------------------------------------------------------
# Declarative parameter reader (shared by all stage registries)
# ---------------------------------------------------------------------------


def test_param_reader_conversions() -> None:
    reader = ParamReader(
        "stage",
        {
            "ratio": 0.25,
            "numeric_string": "2.5",
            "integer": 3,
            "integer_float": 4.0,
            "integer_string": "5",
            "flag": True,
            "flag_int": 0,
            "flag_string": "off",
            "text": "carry",
            "names": "tsmom, kama ,carry_filter",
            "empty_names": "  ",
            "boolean_number": True,
        },
    )
    assert reader.stage == "stage"
    assert reader.has("ratio") is True
    assert reader.has("absent") is False
    assert reader.as_float("ratio", 0.0) == pytest.approx(0.25)
    assert reader.as_float("numeric_string", 0.0) == pytest.approx(2.5)
    assert reader.as_float("boolean_number", 0.0) == pytest.approx(1.0)
    assert reader.as_float("absent", 0.75) == pytest.approx(0.75)
    assert reader.as_optional_float("absent") is None
    assert reader.as_optional_float("ratio") == pytest.approx(0.25)
    assert reader.as_int("integer", 0) == 3
    assert reader.as_int("integer_float", 0) == 4
    assert reader.as_int("integer_string", 0) == 5
    assert reader.as_optional_int("absent") is None
    assert reader.as_optional_int("absent", 9) == 9
    assert reader.as_bool("flag", False) is True
    assert reader.as_bool("flag_int", True) is False
    assert reader.as_bool("flag_string", True) is False
    assert reader.as_bool("absent", True) is True
    assert reader.as_str("text", "fallback") == "carry"
    assert reader.as_optional_str("absent") is None
    assert reader.as_optional_str("absent", "x") == "x"
    assert reader.as_sequence("names") == ("tsmom", "kama", "carry_filter")
    assert reader.as_sequence("empty_names") == ()
    assert reader.as_sequence("absent") == ()
    assert reader.as_sequence("absent", ("tsmom",)) == ("tsmom",)
    reader.check()
    with pytest.raises(ParamError, match="unknown parameter"):
        ParamReader("stage", {"unused": 1}).check()


@pytest.mark.parametrize(
    ("params", "reader", "message"),
    [
        ({"value": "not-a-number"}, "as_float", "expected a number"),
        ({"value": True}, "as_int", "boolean"),
        ({"value": 1.5}, "as_int", "expected an integer"),
        ({"value": "1.5"}, "as_int", "expected an integer"),
        ({"value": 2}, "as_bool", "expected a boolean"),
        ({"value": "maybe"}, "as_bool", "expected a boolean"),
        ({"value": 3}, "as_str", "expected a string"),
        ({"value": 3}, "as_sequence", "comma-separated string"),
        ({"value": 3}, "as_optional_str", "expected a string"),
    ],
)
def test_param_reader_rejects_bad_values(
    params: dict[str, object], reader: str, message: str
) -> None:
    param_reader = ParamReader("stage", params)  # type: ignore[arg-type]
    method = getattr(param_reader, reader)
    with pytest.raises(ParamError, match=message):
        if reader in {"as_optional_str", "as_sequence"}:
            method("value")
        else:
            method("value", 0.0)


def test_build_pipeline_reports_unused_injected_overlays(
    instrument_master: InstrumentsConfig,
) -> None:
    config = load_strategy_config("configs/strategies/tsmom_vol_target.yaml")
    quant = _ScalingRisk(0.5, "quant_risk")
    orphan = _ScalingRisk(0.5, "custom_orphan")
    pipeline = build_pipeline(config, instrument_master, risk_overlays=[quant, orphan])
    assert tuple(overlay.name for overlay in pipeline.risk_overlays) == ("quant_risk",)
    result = pipeline.run(make_market_state())
    assert result.diagnostics["risk_overlays"] == ("quant_risk",)
    assert result.diagnostics["skipped_risk_stages"] == ("llm_risk",)
    assert result.diagnostics["unused_risk_overlays"] == ("custom_orphan",)


def test_importing_the_strategy_package_does_not_pull_gymnasium() -> None:
    """Classical strategies must not require the optional DRL stack at import."""
    import subprocess
    import sys

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, src.strategies; "
            "assert 'gymnasium' not in sys.modules, 'gymnasium leaked into imports'; "
            "assert 'stable_baselines3' not in sys.modules, 'sb3 leaked into imports'; "
            "assert 'torch' not in sys.modules, 'torch leaked into imports'",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
