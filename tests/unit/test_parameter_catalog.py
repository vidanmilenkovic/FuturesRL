"""Unit tests for the Phase-10 typed parameter catalog + Strategy Studio presets."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from configs.instruments import load_instruments
from configs.strategy import (
    StrategyConfig,
    TranslatorConfig,
    discover_strategy_configs,
    load_strategy_config,
)
from src.strategies import (
    ALLOCATOR_REGISTRY,
    CATALOG,
    SELECTOR_REGISTRY,
    TIMING_REGISTRY,
    ParameterCatalog,
    ParameterType,
    apply_parameter_overrides,
    build_pipeline,
    catalog_dict,
    decode_parameter_value,
    encode_parameter_value,
    get_parameter_value,
    parameter_catalog,
    parameter_keys_for_strategy,
    resolve_param_path,
    validate_parameter_value,
)
from src.strategies.timing import ALL_MONTHS

from tests.fixtures.market import make_market_state

PRESET_IDS: tuple[str, ...] = (
    "trend_following_tsmom_kama",
    "commodity_carry_term_structure",
    "panic_mean_reversion",
    "seasonal_commodity_cycle",
    "drl_neural_policy_llm_risk",
)
STRUCTURED_KEYS: tuple[str, ...] = (
    "universe.regime.risk_off_roots",
    "timing.seasonality.active_months",
    "timing.seasonality.root_month_biases",
    "translator.rounding_mode",
    "timing.mean_reversion.rsi_period",
)


def _yaml_strategy() -> dict[str, object]:
    config = load_strategy_config("configs/strategies/tsmom_vol_target.yaml")
    return config.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Catalog shape / invariants
# ---------------------------------------------------------------------------


def test_catalog_is_a_cached_singleton() -> None:
    catalog = parameter_catalog()
    assert isinstance(catalog, ParameterCatalog)
    assert catalog is CATALOG
    assert parameter_catalog() is catalog
    assert catalog.version == 1
    assert [stage.id for stage in catalog.stages] == [
        "universe",
        "allocator",
        "timing",
        "risk",
        "translator",
    ]
    assert catalog.validate() is None


def test_catalog_json_is_serialisable_and_well_formed() -> None:
    payload = catalog_dict()
    encoded = json.dumps(payload)  # must not raise
    assert len(encoded) > 10_000
    assert set(payload) == {"version", "stages", "parameters", "presets"}
    parameters = payload["parameters"]
    assert isinstance(parameters, list) and parameters
    required = {
        "key",
        "stage",
        "stage_name",
        "param",
        "label",
        "description",
        "type",
        "default",
        "min",
        "max",
        "step",
        "choices",
        "unit",
        "optuna_tunable",
        "log_scale",
        "yaml_encoding",
        "min_items",
        "max_items",
        "location",
    }
    valid_types = {kind.value for kind in ParameterType}
    valid_encodings = {"scalar", "comma_ints", "comma_strings", "json_object"}
    valid_locations = {"stage_params", "translator", "drl_block"}
    for entry in parameters:
        assert set(entry) == required
        assert entry["type"] in valid_types
        assert entry["yaml_encoding"] in valid_encodings
        assert entry["location"] in valid_locations
        if entry["stage"] == "translator":
            assert entry["key"] == f"translator.{entry['param']}"
        else:
            assert entry["key"] == f"{entry['stage']}.{entry['stage_name']}.{entry['param']}"
    stage_ids = {stage["id"] for stage in payload["stages"]}
    assert {entry["stage"] for entry in parameters} <= stage_ids


def test_catalog_parameter_invariants() -> None:
    catalog = parameter_catalog()
    keys = [spec.key for spec in catalog.parameters]
    assert len(keys) == len(set(keys))
    for spec in catalog.parameters:
        assert spec.label and spec.description
        assert spec.description.strip()
        if spec.type in {ParameterType.INT, ParameterType.FLOAT}:
            assert not spec.choices
            if spec.default is not None:
                if spec.min is not None:
                    assert spec.min <= spec.default
                if spec.max is not None:
                    assert spec.max >= spec.default
            if spec.optuna_tunable:
                assert spec.min is not None and spec.max is not None and spec.step is not None
                assert spec.min <= spec.max
            if spec.step is not None:
                assert spec.step > 0
            if spec.log_scale:
                assert spec.min is not None and spec.min > 0
        elif spec.type is ParameterType.BOOL:
            assert isinstance(spec.default, bool)
            assert not spec.choices
        elif spec.type is ParameterType.SELECT:
            assert len(spec.choices) >= 2
            assert spec.default in spec.choices
        else:
            assert spec.choices
            assert isinstance(spec.default, tuple)
            assert set(spec.default) <= set(spec.choices)
            assert spec.optuna_tunable is False
            assert spec.yaml_encoding != "scalar"


def test_catalog_covers_every_registered_stage() -> None:
    catalog = parameter_catalog()
    for name in SELECTOR_REGISTRY:
        if name == "composite":
            continue
        assert catalog.for_stage_name(name), f"missing selector params for {name}"
    for name in ALLOCATOR_REGISTRY:
        assert catalog.for_stage_name(name), f"missing allocator params for {name}"
    for name in TIMING_REGISTRY:
        if name == "composite":
            continue
        assert catalog.for_stage_name(name), f"missing timing params for {name}"
    for risk_stage in ("quant_risk", "llm_risk"):
        assert catalog.for_stage_name(risk_stage)
    translator_fields = set(TranslatorConfig.model_fields)
    catalogued = {spec.param for spec in catalog.for_stage("translator")}
    assert catalogued == translator_fields


def test_catalog_covers_committed_yaml_configs() -> None:
    catalog = parameter_catalog()
    known = {spec.key for spec in catalog.parameters}
    for config in discover_strategy_configs("configs/strategies").values():
        stages = [
            ("universe", config.universe),
            ("allocator", config.allocator),
            *[("timing", stage) for stage in config.timing],
            *[("risk", stage) for stage in config.risk],
        ]
        for stage_name, stage in stages:
            for param in stage.params:
                key = f"{stage_name}.{stage.name}.{param}"
                assert key in known, f"{config.id}: parameter {key!r} is not catalogued"
        for translator_field in TranslatorConfig.model_fields:
            assert translator_field in {spec.param for spec in catalog.for_stage("translator")}
        if config.drl is not None:
            for field_name in type(config.drl).model_fields:
                if field_name == "model_path":
                    continue  # runtime artefact, intentionally not a hyperparameter
                assert f"allocator.drl.{field_name}" in known


def test_new_overlay_parameters_are_catalogued() -> None:
    catalog = parameter_catalog()
    mean_reversion = {spec.param for spec in catalog.for_stage_name("mean_reversion")}
    assert mean_reversion == {
        "rsi_period",
        "oversold_threshold",
        "overbought_threshold",
        "bollinger_window",
        "bollinger_std",
        "turbulence_veto_threshold",
        "reversion_strength",
    }
    seasonality = {spec.param for spec in catalog.for_stage_name("seasonality")}
    assert seasonality == {"active_months", "root_month_biases", "out_of_season_scale"}
    months = catalog.get("timing.seasonality.active_months")
    assert months.type is ParameterType.MULTISELECT
    assert set(months.choices) == set(ALL_MONTHS)
    roots = catalog.get("timing.seasonality.root_month_biases")
    assert "CL" in roots.choices and "GC" in roots.choices
    assert roots.yaml_encoding == "json_object"


def test_optuna_tunable_subset_is_numeric_or_categorical() -> None:
    catalog = parameter_catalog()
    tunable = catalog.tunable()
    assert tunable
    assert {spec.type for spec in tunable} <= {
        ParameterType.INT,
        ParameterType.FLOAT,
        ParameterType.SELECT,
        ParameterType.BOOL,
    }
    for spec in tunable:
        if spec.type in {ParameterType.INT, ParameterType.FLOAT}:
            assert spec.min is not None and spec.max is not None and spec.step is not None
        if spec.type is ParameterType.SELECT:
            assert spec.choices


# ---------------------------------------------------------------------------
# Strategy bindings (paths / read / partial updates)
# ---------------------------------------------------------------------------


def test_resolve_param_path_handles_ordered_stages() -> None:
    strategy = _yaml_strategy()
    assert resolve_param_path(strategy, "timing.kama.window") == "timing.1.params.window"
    assert resolve_param_path(strategy, "timing.carry_filter.min_carry_bps") == (
        "timing.2.params.min_carry_bps"
    )
    assert resolve_param_path(strategy, "universe.liquidity.top_n") == "universe.params.top_n"
    assert resolve_param_path(strategy, "allocator.volatility_target.max_gross_leverage") == (
        "allocator.params.max_gross_leverage"
    )
    assert resolve_param_path(strategy, "risk.llm_risk.confidence_threshold") == (
        "risk.1.params.confidence_threshold"
    )
    assert resolve_param_path(strategy, "translator.max_margin_utilization") == (
        "translator.max_margin_utilization"
    )


def test_resolve_param_path_errors_for_unbound_stages() -> None:
    strategy = _yaml_strategy()
    with pytest.raises(KeyError, match="no 'mean_reversion' stage"):
        resolve_param_path(strategy, "timing.mean_reversion.rsi_period")
    with pytest.raises(KeyError, match="no 'drl' block"):
        resolve_param_path(strategy, "allocator.drl.reward_risk_penalty_lambda")
    with pytest.raises(KeyError, match="not 'risk_parity'"):
        resolve_param_path(strategy, "allocator.risk_parity.gross_leverage")
    with pytest.raises(KeyError, match="unknown parameter key"):
        resolve_param_path(strategy, "timing.tsmom.not_a_param")


def test_get_parameter_value_decodes_yaml_values() -> None:
    strategy = _yaml_strategy()
    assert get_parameter_value(strategy, "timing.tsmom.lookback_fast") == 21
    assert get_parameter_value(strategy, "risk.llm_risk.confidence_threshold") == pytest.approx(0.7)
    droot = load_strategy_config("configs/strategies/drl_ppo_llm_risk.yaml").model_dump(mode="json")
    assert get_parameter_value(droot, "allocator.drl.reward_risk_penalty_lambda") == pytest.approx(
        0.5
    )
    assert get_parameter_value(droot, "allocator.drl.normalize_observations") is True


def test_validate_parameter_value_types_ranges_and_choices() -> None:
    assert validate_parameter_value("timing.tsmom.lookback_fast", 30) == 30
    assert validate_parameter_value("timing.tsmom.lookback_fast", 30.0) == 30
    assert validate_parameter_value(
        "allocator.volatility_target.target_annual_volatility", 0.1
    ) == (pytest.approx(0.1))
    assert validate_parameter_value("risk.quant_risk.volatility_targeting_enabled", False) is False
    assert validate_parameter_value("translator.rounding_mode", "FLOOR") == "floor"
    assert validate_parameter_value("timing.seasonality.active_months", [1, "2", 12]) == (1, 2, 12)
    invalid: tuple[tuple[str, object], ...] = (
        ("timing.tsmom.lookback_fast", True),
        ("timing.tsmom.lookback_fast", 1),
        ("timing.tsmom.lookback_fast", 10_000),
        ("timing.tsmom.lookback_fast", 12.5),
        ("timing.tsmom.lookback_fast", None),
        ("timing.tsmom.lookback_fast", "fast"),
        ("risk.quant_risk.volatility_targeting_enabled", "yes"),
        ("translator.rounding_mode", "bankers"),
        ("timing.seasonality.active_months", []),
        ("timing.seasonality.active_months", [0]),
        ("universe.regime.risk_off_roots", ["ZZZ"]),
        ("allocator.volatility_target.target_annual_volatility", float("nan")),
    )
    for key, value in invalid:
        with pytest.raises(ValueError):
            validate_parameter_value(key, value)
    # Validation errors name the offending key for actionable Studio feedback.
    with pytest.raises(ValueError, match="lookback_fast"):
        validate_parameter_value("timing.tsmom.lookback_fast", 10_000)
    # Duplicate multiselect entries are de-duplicated rather than rejected.
    assert validate_parameter_value("timing.seasonality.active_months", [1, 1]) == (1,)
    with pytest.raises(KeyError, match="unknown parameter key"):
        validate_parameter_value("nope.nope.nope", 1)


def test_encode_decode_yaml_round_trips() -> None:
    assert decode_parameter_value("timing.seasonality.active_months", "1,2,3") == (1, 2, 3)
    assert decode_parameter_value("timing.seasonality.active_months", [1, 2]) == (1, 2)
    assert encode_parameter_value("timing.seasonality.active_months", [1, 2, 3]) == "1,2,3"
    assert encode_parameter_value("universe.regime.risk_off_roots", ["CL", "GC"]) == "CL,GC"
    assert decode_parameter_value("universe.regime.risk_off_roots", "CL,GC") == ("CL", "GC")
    encoded = encode_parameter_value("timing.seasonality.root_month_biases", ["CL"])
    decoded = decode_parameter_value("timing.seasonality.root_month_biases", encoded)
    assert isinstance(decoded, dict)
    assert decoded["CL"][2] == pytest.approx(0.10)
    custom = '{"GC": {"1": 0.2, "9": -0.05}}'
    decoded_custom = decode_parameter_value("timing.seasonality.root_month_biases", custom)
    assert decoded_custom == {"GC": {1: 0.2, 9: -0.05}}
    assert decode_parameter_value("timing.seasonality.active_months", "-1") == (-1,)
    assert encode_parameter_value("timing.tsmom.lookback_fast", 30) == 30
    with pytest.raises(ValueError, match="is not one of"):
        encode_parameter_value("timing.seasonality.root_month_biases", ["ZZZ"])


def test_apply_parameter_overrides_is_deep_and_validated() -> None:
    strategy = _yaml_strategy()
    original = json.loads(json.dumps(strategy))
    updated = apply_parameter_overrides(
        strategy,
        {
            "timing.tsmom.lookback_fast": 30,
            "universe.liquidity.top_n": 7,
            "translator.max_margin_utilization": 0.5,
            "risk.llm_risk.critical_score": 4.8,
        },
    )
    assert strategy == original  # deep-copied, source untouched
    assert updated["timing"][0]["params"]["lookback_fast"] == 30
    assert updated["universe"]["params"]["top_n"] == 7
    assert updated["translator"]["max_margin_utilization"] == 0.5
    # The overridden strategy is still a valid StrategyConfig.
    assert StrategyConfig.model_validate(updated).id == "tsmom_vol_target"
    # Multiselect/object overrides are encoded into YAML-native scalars.
    seasonal = apply_parameter_overrides(
        parameter_catalog().presets[3].strategy,
        {
            "timing.seasonality.active_months": [1, 2, 11, 12],
            "timing.seasonality.root_month_biases": ["CL", "GC"],
        },
    )
    timing_params = seasonal["timing"][0]["params"]
    assert timing_params["active_months"] == "1,2,11,12"
    assert isinstance(timing_params["root_month_biases"], str)
    assert json.loads(timing_params["root_month_biases"])["CL"]["2"] == pytest.approx(0.1)
    with pytest.raises(KeyError, match="unknown parameter key"):
        apply_parameter_overrides(strategy, {"timing.nope.nope": 1})
    with pytest.raises(ValueError, match="lookback_fast"):
        apply_parameter_overrides(strategy, {"timing.tsmom.lookback_fast": 10_000})
    with pytest.raises(KeyError, match="no 'drl' block"):
        apply_parameter_overrides(strategy, {"allocator.drl.normalize_observations": False})


def test_parameter_keys_for_strategy_filters_by_stage_presence() -> None:
    strategy = _yaml_strategy()
    keys = parameter_keys_for_strategy(strategy)
    assert "timing.tsmom.lookback_fast" in keys
    assert "timing.mean_reversion.rsi_period" not in keys
    assert "allocator.drl.total_timesteps" not in keys
    assert "translator.max_margin_utilization" in keys
    drl_strategy = load_strategy_config("configs/strategies/drl_ppo_llm_risk.yaml").model_dump(
        mode="json"
    )
    drl_keys = parameter_keys_for_strategy(drl_strategy)
    assert "allocator.drl.total_timesteps" in drl_keys
    assert "allocator.drl.reward_risk_penalty_lambda" in drl_keys


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


def test_presets_are_complete_valid_strategies() -> None:
    catalog = parameter_catalog()
    assert tuple(preset.id for preset in catalog.presets) == PRESET_IDS
    catalog_keys = {spec.key for spec in catalog.parameters}
    for preset in catalog.presets:
        assert preset.name and preset.description and preset.hypothesis
        assert preset.tags
        assert set(preset.parameter_keys) <= catalog_keys
        config = preset.to_config()  # must validate as StrategyConfig
        assert config.id.startswith("preset_")
        assert config.universe.name and config.allocator.name
        assert preset.parameter_keys
    payload = catalog_dict()["presets"]
    assert isinstance(payload, list) and len(payload) == 5
    for entry in payload:
        assert set(entry) == {
            "id",
            "name",
            "description",
            "hypothesis",
            "tags",
            "parameter_keys",
            "strategy",
        }
        # The strategy payload is POST-ready: it validates straight into StrategyConfig.
        assert StrategyConfig.model_validate(entry["strategy"]).id == entry["strategy"]["id"]
    seasonal = next(item for item in payload if item["id"] == "seasonal_commodity_cycle")
    seasonality_params = seasonal["strategy"]["timing"][0]["params"]
    assert seasonality_params["active_months"] == "1,2,3,4,5,9,10,11,12"
    assert seasonality_params["root_month_biases"] == "CL,GC"
    drl_preset = next(item for item in payload if item["id"] == "drl_neural_policy_llm_risk")
    assert drl_preset["strategy"]["drl"]["algorithm"] == "PPO"
    assert drl_preset["strategy"]["allocator"]["name"] == "drl"


def test_presets_build_and_run_through_the_pipeline() -> None:
    master = load_instruments()
    state = make_market_state()
    catalog = parameter_catalog()
    for preset in catalog.presets:
        pipeline = build_pipeline(preset.to_config(), master)
        if preset.id == "drl_neural_policy_llm_risk":
            assert pipeline.allocator.name == "drl"
            with pytest.raises(RuntimeError, match="no model"):
                pipeline.run(state)  # checkpoint not bound in the preset
            continue
        result = pipeline.run(state)
        assert result.universe
        assert result.final_weights is not None
        for root in result.final_weights:
            assert math.isfinite(result.final_weights[root])


def test_catalog_preset_parameter_keys_are_bound_by_their_preset() -> None:
    for preset in parameter_catalog().presets:
        if preset.id == "drl_neural_policy_llm_risk":
            # The preset's drl block exists, so drl-block keys must bind.
            assert "allocator.drl.reward_risk_penalty_lambda" in preset.parameter_keys
        for key in preset.parameter_keys:
            assert resolve_param_path(preset.strategy, key)


def test_new_overlays_can_be_selected_from_a_yaml_config(tmp_path: Path) -> None:
    """A strategy YAML can select the new overlays by name with scalar params."""
    yaml_text = """
id: mean_reversion_seasonal_test
description: mean-reversion + seasonality wiring test
tags: [test]
universe:
  selector: liquidity
  params:
    min_avg_dollar_volume: 0.0
allocator:
  name: volatility_target
  params:
    target_annual_volatility: 0.12
    max_gross_leverage: 3.0
timing:
  - name: mean_reversion
    params:
      rsi_period: 14
      oversold_threshold: 30.0
      overbought_threshold: 70.0
      turbulence_veto_threshold: 0.95
      reversion_strength: 0.5
  - name: seasonality
    params:
      active_months: "1,2,3,9,10,11,12"
      root_month_biases: "CL,GC"
      out_of_season_scale: 0.5
risk: []
translator:
  rounding_mode: floor
  max_margin_utilization: 0.6
"""
    path = tmp_path / "mean_reversion_seasonal.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    config = load_strategy_config(path)
    assert [stage.name for stage in config.timing] == ["mean_reversion", "seasonality"]
    pipeline = build_pipeline(config, load_instruments())
    result = pipeline.run(make_market_state())
    assert result.universe
    assert result.timing_weights
    assert result.final_weights.gross <= result.timing_weights.gross + 1e-9
    assert result.diagnostics["timing_overlays"] == ("mean_reversion", "seasonality")
