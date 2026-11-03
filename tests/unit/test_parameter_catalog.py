"""Unit tests for the Phase-10 typed parameter catalog + Strategy Studio presets."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
from configs.instruments import InstrumentsConfig, load_instruments, parse_instruments_config
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
from src.strategies.parameter_catalog import (
    DEFAULT_SELECTED_PARAMETER_KEYS,
    NON_TUNABLE_PARAMETER_KEYS,
    UNIVERSE_SECTOR_GROUPS,
    active_parameter_keys,
    clear_parameter_catalog_cache,
    default_selected_parameter_keys,
    parameters_for_strategy,
    refresh_parameter_catalog,
    strategy_root_symbols,
    universe_metadata,
    universe_roots,
    universe_sectors,
    validate_strategy_roots,
)
from src.strategies.timing import ALL_MONTHS, DEFAULT_ROOT_MONTH_BIASES

from tests.fixtures.market import make_market_state

REPO_ROOT = Path(__file__).resolve().parents[2]

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
    assert catalog is CATALOG  # import-time catalog is the default-master cache entry
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
    assert set(payload) >= {"version", "stages", "parameters", "presets"}
    assert {"universe", "stage_choices"} <= set(payload)
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
        "optuna_default_selected",
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
        if spec.optuna_default_selected:
            assert spec.optuna_tunable
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
    # Root choices are derived from the live instrument master (no hardcoded list).
    assert roots.choices == tuple(load_instruments().roots())
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


# ---------------------------------------------------------------------------
# Dynamic instrument master: universe metadata + stage dropdown choices
# ---------------------------------------------------------------------------

CANONICAL_SECTOR_IDS: tuple[str, ...] = tuple(
    group_id for group_id, _label, _desc in UNIVERSE_SECTOR_GROUPS
)


def _custom_master_with_extra_root(
    root: str = "ZC9",
    *,
    sector: str = "agri",
    asset_class: str = "agriculture",
    name: str = "Corn (custom catalog test)",
) -> InstrumentsConfig:
    """Minimal single-root instrument master derived from the committed ES spec."""
    master = load_instruments()
    entry = master.get("ES").model_dump(mode="json")
    entry.pop("micro_root", None)
    entry.pop("standard_root", None)
    entry.update(
        {
            "root": root,
            "name": name,
            "asset_class": asset_class,
            "sector": sector,
            "exchange": "CBOT",
            "databento_symbol": f"{root}.FUT",
            "ibkr_exchange": "CBOT",
        }
    )
    return parse_instruments_config(
        {
            "version": master.version,
            "defaults": master.defaults.model_dump(mode="json"),
            "instruments": {root: entry},
        }
    )


def test_universe_sectors_partition_the_instrument_master() -> None:
    master = load_instruments()
    sectors = universe_sectors()
    # Canonical Studio groups come first, in display order (All first).
    assert [sector.id for sector in sectors] == list(CANONICAL_SECTOR_IDS)
    assert sectors[0].id == "all"
    assert sectors[0].count == len(master.roots())
    assert [root.root for root in sectors[0].roots] == list(master.roots())
    # Every root lands in exactly one non-All group and the groups partition the master.
    covered = [root.root for sector in sectors[1:] for root in sector.roots]
    assert len(covered) == len(set(covered))
    assert set(covered) == set(master.roots())
    assert sum(sector.count for sector in sectors[1:]) == len(master.roots())
    for sector in sectors:
        assert sector.count == len(sector.roots)
        assert sector.label and sector.description
    # The seven canonical asset-class groups all exist (empty ones stay exposed too).
    assert {sector.id for sector in sectors} == set(CANONICAL_SECTOR_IDS)


def test_universe_sector_groups_map_instrument_master_sectors() -> None:
    master = load_instruments()
    by_group: dict[str, set[str]] = {}
    for spec in master.instruments.values():
        group = next(root.sector_group for root in universe_roots() if root.root == spec.root)
        by_group.setdefault(group, set()).add(spec.sector)
    for group_id, raw_sectors in by_group.items():
        assert group_id in CANONICAL_SECTOR_IDS, (
            f"root sectors {raw_sectors} mapped to unknown group"
        )
    # The seven canonical groups account for every current instrument-master sector.
    assert set(by_group) == set(CANONICAL_SECTOR_IDS[1:])
    # Spot-check the UI grouping for the sectors currently in the master.
    sectors = {sector.id: sector for sector in universe_sectors()}
    for group_id, raw_sectors in by_group.items():
        assert {root.sector for root in sectors[group_id].roots} == raw_sectors


def test_universe_root_metadata_matches_instrument_master() -> None:
    master = load_instruments()
    roots = universe_roots()
    assert tuple(root.root for root in roots) == master.roots()
    for meta in roots:
        spec = master.get(meta.root)
        assert meta.name == spec.name
        assert meta.asset_class == spec.asset_class.value
        assert meta.sector == spec.sector
        assert meta.exchange == spec.exchange
        assert meta.multiplier == spec.multiplier
        assert meta.tick_size == spec.tick_size
        assert meta.micro_root == spec.micro_root
        assert meta.standard_root == spec.standard_root
        assert meta.currency == spec.currency
        assert meta.sector_group in CANONICAL_SECTOR_IDS
    payload = [root.to_dict() for root in roots]
    assert set(payload[0]) == {
        "root",
        "name",
        "asset_class",
        "sector",
        "sector_group",
        "exchange",
        "multiplier",
        "tick_size",
        "micro_root",
        "standard_root",
        "currency",
    }
    json.dumps(payload)  # JSON-serialisable


def test_catalog_root_and_sector_choices_track_instrument_master() -> None:
    master = load_instruments()
    catalog = parameter_catalog()
    assert catalog.get("universe.regime.risk_off_roots").choices == tuple(master.roots())
    assert catalog.get("universe.regime.risk_off_sectors").choices == tuple(master.sectors())
    assert catalog.get("timing.seasonality.root_month_biases").choices == tuple(master.roots())
    # The catalog is bound to the master it was built from.
    assert catalog.instrument_master is not None
    assert tuple(catalog.instrument_master.roots()) == tuple(master.roots())
    assert len(catalog.instrument_master.roots()) == len(master.roots())


def test_universe_metadata_is_additive_in_catalog_payload() -> None:
    payload = catalog_dict()
    assert isinstance(payload["universe"], dict)
    universe = payload["universe"]
    assert set(universe) == {"sectors", "roots"}
    json.dumps(universe)
    assert tuple(root["root"] for root in universe["roots"]) == tuple(load_instruments().roots())
    assert universe_metadata().to_dict() == universe
    first_sector = universe["sectors"][0]
    assert set(first_sector) >= {"id", "label", "description", "count", "roots"}
    assert first_sector["id"] == "all"
    assert first_sector["count"] == len(load_instruments().roots())
    # Catalog object exposes the same universe metadata.
    catalog = parameter_catalog()
    assert catalog.universe_metadata().to_dict() == universe
    assert tuple(sector.id for sector in catalog.universe_sectors()) == CANONICAL_SECTOR_IDS


def test_stage_choices_expose_registry_dropdowns() -> None:
    catalog = parameter_catalog()
    payload = catalog_dict()
    stage_choices = catalog.stage_choices()
    assert stage_choices["universe"] == ("liquidity", "regime", "composite")
    assert set(stage_choices["universe"]) == set(SELECTOR_REGISTRY)
    assert stage_choices["allocator"][:2] == ("volatility_target", "risk_parity")
    assert set(stage_choices["allocator"]) == set(ALLOCATOR_REGISTRY)
    assert set(stage_choices["timing"]) == set(TIMING_REGISTRY)
    assert set(stage_choices["risk"]) == {"quant_risk", "llm_risk"}
    assert set(stage_choices["translator"]) == {"translator"}
    # Payload mirrors the object API and each option carries bindable parameter keys.
    assert payload["stage_choices"] == {
        stage_id: list(names) for stage_id, names in stage_choices.items()
    }
    known_keys = {spec.key for spec in catalog.parameters}
    payload_stages = {stage["id"]: stage for stage in payload["stages"]}
    for stage_id, names in stage_choices.items():
        options = payload_stages[stage_id]["choices"]
        assert [option["name"] for option in options] == list(names)
        for option in options:
            assert option["label"] and option["description"]
            assert set(option["parameter_keys"]) <= known_keys
    # Universe/allocator dropdown options expose the expected stage parameters.
    assert {spec.param for spec in catalog.for_stage_name("liquidity")} <= {
        key.split(".")[-1]
        for option in payload_stages["universe"]["choices"]
        if option["name"] == "liquidity"
        for key in option["parameter_keys"]
    }
    assert {spec.param for spec in catalog.for_stage_name("volatility_target")} <= {
        key.split(".")[-1]
        for option in payload_stages["allocator"]["choices"]
        if option["name"] == "volatility_target"
        for key in option["parameter_keys"]
    }


def test_parameter_catalog_is_cached_per_instrument_master_and_refreshes() -> None:
    clear_parameter_catalog_cache()
    default_catalog = parameter_catalog()
    custom = _custom_master_with_extra_root()
    custom_catalog = parameter_catalog(custom)
    assert custom_catalog is not default_catalog
    assert custom_catalog.instrument_master is custom
    assert custom_catalog.get("universe.regime.risk_off_roots").choices == ("ZC9",)
    assert tuple(root.root for root in universe_roots(custom)) == ("ZC9",)
    custom_sectors = universe_sectors(custom)
    assert [sector.id for sector in custom_sectors] == list(CANONICAL_SECTOR_IDS)
    agri = next(sector for sector in custom_sectors if sector.id == "agri")
    assert agri.count == 1
    assert agri.roots[0].root == "ZC9"
    # Cached per instrument-master fingerprint.
    assert parameter_catalog(custom) is custom_catalog
    clear_parameter_catalog_cache()
    rebuilt = parameter_catalog(custom)
    assert rebuilt is not custom_catalog
    assert rebuilt.get("universe.regime.risk_off_roots").choices == ("ZC9",)
    assert rebuilt.get("timing.seasonality.root_month_biases").choices == ("ZC9",)
    # refresh_parameter_catalog reloads the committed default master and re-primes the cache.
    refreshed = refresh_parameter_catalog()
    assert refreshed is parameter_catalog()
    assert len(refreshed.get("universe.regime.risk_off_roots").choices) == len(
        load_instruments().roots()
    )
    custom_catalog.validate()
    rebuilt.validate()


def test_universe_sector_mapping_falls_back_for_unmapped_sectors() -> None:
    # Unknown sector string but a known asset class -> asset-class fallback group.
    fallback = _custom_master_with_extra_root(
        root="ZZ1", sector="exotic", asset_class="agriculture"
    )
    meta = universe_roots(fallback)[0]
    assert meta.sector_group == "agri"
    # Unknown sector *and* unknown asset class -> additive "Other" group only when needed.
    unknown = _custom_master_with_extra_root(root="ZZ2", sector="exotic", asset_class="unknown")
    sectors = universe_sectors(unknown)
    assert [sector.id for sector in sectors] == [*CANONICAL_SECTOR_IDS, "other"]
    assert sectors[-1].count == 1
    assert sectors[-1].roots[0].root == "ZZ2"


def test_seasonal_encoding_handles_expanded_instrument_master() -> None:
    master = load_instruments()
    extra_roots = [root for root in master.roots() if root not in DEFAULT_ROOT_MONTH_BIASES]
    assert extra_roots, "expected the expanded master to add roots beyond the bespoke curves"
    target = extra_roots[0]
    encoded = encode_parameter_value("timing.seasonality.root_month_biases", [target])
    payload = json.loads(encoded)
    assert target in payload
    assert isinstance(payload[target], dict)
    decoded = decode_parameter_value("timing.seasonality.root_month_biases", encoded)
    assert isinstance(decoded, dict) and target in decoded
    # Custom instrument masters can supply their own root choices via the typed spec.
    custom = _custom_master_with_extra_root()
    custom_spec = parameter_catalog(custom).get("timing.seasonality.root_month_biases")
    custom_encoded = encode_parameter_value(custom_spec, ["ZC9"], instrument_master=custom)
    assert "ZC9" in json.loads(custom_encoded)
    with pytest.raises(ValueError):
        encode_parameter_value("timing.seasonality.root_month_biases", ["NOT_A_ROOT"])


def test_presets_remain_valid_with_expanded_instrument_master() -> None:
    master = load_instruments()
    catalog = parameter_catalog()
    catalog_roots = set(master.roots())
    for preset in catalog.presets:
        config = preset.to_config()
        assert config.id.startswith("preset_")
        for stage in config.timing:
            if stage.name != "seasonality":
                continue
            raw = str(stage.params.get("root_month_biases", ""))
            if raw and not raw.startswith("{"):
                assert {root.strip() for root in raw.split(",")} <= catalog_roots
        pipeline = build_pipeline(config, master)
        assert pipeline.selector.name == config.universe.name


def test_catalog_imports_without_gymnasium_sb3_or_torch() -> None:
    script = (
        "import importlib, json, sys; "
        "catalog = importlib.import_module('src.strategies.parameter_catalog'); "
        "payload = catalog.catalog_dict(); json.dumps(payload); "
        "assert 'gymnasium' not in sys.modules, 'gymnasium imported'; "
        "assert 'stable_baselines3' not in sys.modules, 'stable_baselines3 imported'; "
        "assert 'torch' not in sys.modules, 'torch imported'; "
        "assert payload['stage_choices']['universe'] == ['liquidity', 'regime', 'composite']; "
        "assert set(payload['universe']) == {'sectors', 'roots'}"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


# ---------------------------------------------------------------------------
# v1.5.1 Fix 1: Optuna default selection + tunable hygiene + active filtering
# ---------------------------------------------------------------------------

#: Curated core hypotheses pre-selected in the Studio optimizer (v1.5.3 Fix 1:
#: five keys — the carry floor was removed from the global pre-selection).
CURATED_DEFAULT_KEYS: frozenset[str] = frozenset(
    {
        "timing.tsmom.lookback_fast",
        "timing.tsmom.lookback_slow",
        "timing.kama.window",
        "allocator.volatility_target.target_annual_volatility",
        "risk.quant_risk.trailing_stop_pct",
    }
)

#: Root tokens that must never appear as tradable roots (delisted/non-canonical).
DELISTED_ROOT_TOKENS: frozenset[str] = frozenset({"GE", "ZQ", "KE", "BTC"})


def _synthetic_strategy(**overrides: object) -> StrategyConfig:
    from configs.strategy import StageConfig

    payload: dict[str, object] = {
        "id": "active_filter_synthetic",
        "universe": StageConfig(name="liquidity", params={}),
        "allocator": StageConfig(name="volatility_target", params={}),
        "timing": (StageConfig(name="tsmom", params={}),),
        "risk": (),
        "translator": TranslatorConfig(),
    }
    payload.update(overrides)
    return StrategyConfig.model_validate(payload)


def test_optuna_default_selected_is_exactly_the_curated_core() -> None:
    catalog = parameter_catalog()
    selected = tuple(spec.key for spec in catalog.parameters if spec.optuna_default_selected)
    assert set(selected) == CURATED_DEFAULT_KEYS == set(DEFAULT_SELECTED_PARAMETER_KEYS)
    assert len(tuple(DEFAULT_SELECTED_PARAMETER_KEYS)) == 5
    # v1.5.3: the carry floor stays tunable but is no longer globally pre-selected.
    assert catalog.get("timing.carry_filter.min_carry_bps").optuna_default_selected is False
    for key in selected:
        assert catalog.get(key).optuna_tunable is True
    # Payload: the additive flag exists on every entry, is always a bool, and only
    # the five curated keys are True (False, not omitted/null, for all others).
    payload = catalog_dict()
    flags = [entry["optuna_default_selected"] for entry in payload["parameters"]]
    assert all(isinstance(flag, bool) for flag in flags)
    assert sum(flags) == 5
    true_keys = {
        entry["key"] for entry in payload["parameters"] if entry["optuna_default_selected"]
    }
    assert true_keys == CURATED_DEFAULT_KEYS
    assert payload["version"] == 1  # additive field only; contract version unchanged


def test_structural_and_hygiene_parameters_are_not_tunable() -> None:
    catalog = parameter_catalog()
    # Explicit Lead-listed hygiene keys must all be present and non-tunable.
    listed = {
        "universe.liquidity.min_avg_dollar_volume",
        "universe.liquidity.min_open_interest",
        "universe.liquidity.min_history_days",
        "universe.liquidity.top_n",
        "universe.regime.top_n",
        "risk.llm_risk.sample_count",
        "risk.llm_risk.cache_ttl_seconds",
        "risk.llm_risk.macro_lockout_minutes_before",
        "risk.llm_risk.macro_lockout_minutes_after",
        "risk.llm_risk.macro_lockout_action",
        "risk.llm_risk.macro_lockout_leverage_cap",
    }
    assert listed <= set(NON_TUNABLE_PARAMETER_KEYS)
    for key in listed:
        spec = catalog.get(key)
        assert spec.optuna_tunable is False, key
        assert spec.optuna_default_selected is False, key
    # The full structural/hygiene policy is enforced on every catalog entry.
    for key in NON_TUNABLE_PARAMETER_KEYS:
        spec = catalog.get(key)
        assert spec.optuna_tunable is False, key
        assert spec.optuna_default_selected is False, key
    # Structural non-tunables are excluded from the optimizer candidate pool.
    tunable_keys = {spec.key for spec in catalog.tunable()}
    assert NON_TUNABLE_PARAMETER_KEYS.isdisjoint(tunable_keys)
    assert tunable_keys >= CURATED_DEFAULT_KEYS


def test_active_filtering_for_tsmom_config() -> None:
    strategy = load_strategy_config("configs/strategies/tsmom_vol_target.yaml").model_dump(
        mode="json"
    )
    keys = active_parameter_keys(strategy)
    specs = parameters_for_strategy(strategy)
    assert tuple(spec.key for spec in specs) == keys
    assert all(spec.optuna_tunable for spec in specs)
    # Active overlays/stages are included.
    for key in (
        "timing.tsmom.lookback_fast",
        "timing.tsmom.lookback_slow",
        "timing.kama.window",
        "timing.carry_filter.min_carry_bps",
        "allocator.volatility_target.target_annual_volatility",
        "risk.quant_risk.trailing_stop_pct",
        "risk.llm_risk.confidence_threshold",
    ):
        assert key in keys, key
    # Inactive selector/overlays and non-tunable params are excluded.
    inactive_or_structural = (
        "universe.regime.shock_z_threshold",  # liquidity selector active, regime not
        "universe.liquidity.top_n",  # active selector but structural/non-tunable
        "allocator.risk_parity.target_annual_volatility",
        "allocator.equal_weight.gross_leverage",
        "allocator.drl.total_timesteps",
        "timing.mean_reversion.rsi_period",
        "timing.seasonality.out_of_season_scale",
        "translator.rebalance_buffer_contracts",
    )
    for key in inactive_or_structural:
        assert key not in keys, key
    # The TSMOM YAML activates all six curated defaults.
    assert set(default_selected_parameter_keys(strategy)) == CURATED_DEFAULT_KEYS


def test_active_filtering_for_panic_mean_reversion_preset() -> None:
    preset = next(p for p in parameter_catalog().presets if p.id == "panic_mean_reversion")
    keys = active_parameter_keys(preset.strategy)
    for key in (
        "timing.mean_reversion.rsi_period",
        "timing.mean_reversion.turbulence_veto_threshold",
        "allocator.risk_parity.target_annual_volatility",
        "risk.quant_risk.trailing_stop_pct",
    ):
        assert key in keys, key
    for key in (
        "timing.tsmom.lookback_fast",
        "timing.kama.window",
        "timing.carry_filter.min_carry_bps",
        "allocator.volatility_target.target_annual_volatility",
    ):
        assert key not in keys, key
    # Only the curated default that this preset's active stages expose is selected.
    assert set(default_selected_parameter_keys(preset.strategy)) == {
        "risk.quant_risk.trailing_stop_pct"
    }


def test_active_filtering_expands_composite_timing_components() -> None:
    from configs.strategy import StageConfig

    composite = _synthetic_strategy(
        timing=(StageConfig(name="composite", params={"components": "tsmom,kama"}),)
    )
    strategy = composite.model_dump(mode="json")
    keys = active_parameter_keys(strategy)
    assert "timing.tsmom.lookback_fast" in keys
    assert "timing.kama.window" in keys
    # Non-composited overlays and the composite plumbing itself are excluded.
    assert "timing.carry_filter.min_carry_bps" not in keys
    assert "timing.mean_reversion.rsi_period" not in keys
    assert all(not key.startswith("timing.composite.") for key in keys)
    assert set(default_selected_parameter_keys(composite)) == {
        "allocator.volatility_target.target_annual_volatility",
        "timing.tsmom.lookback_fast",
        "timing.tsmom.lookback_slow",
        "timing.kama.window",
    }


def test_active_filtering_for_drl_config_excludes_non_selected_allocators() -> None:
    strategy = load_strategy_config("configs/strategies/drl_ppo_llm_risk.yaml").model_dump(
        mode="json"
    )
    keys = active_parameter_keys(strategy)
    assert "allocator.drl.total_timesteps" in keys
    assert "allocator.drl.risk_penalty_lambda" in keys
    for key in (
        "allocator.volatility_target.target_annual_volatility",
        "allocator.risk_parity.gross_leverage",
        "allocator.equal_weight.gross_leverage",
        "allocator.mean_variance.risk_aversion",
        "allocator.drl.algorithm",  # active but structural/non-tunable
        "timing.kama.window",  # not in this strategy's timing list
        "timing.carry_filter.min_carry_bps",
    ):
        assert key not in keys, key
    assert set(default_selected_parameter_keys(strategy)) == {
        "timing.tsmom.lookback_fast",
        "timing.tsmom.lookback_slow",
        "risk.quant_risk.trailing_stop_pct",
    }


def test_active_filtering_accepts_validated_config_objects() -> None:
    config = load_strategy_config("configs/strategies/tsmom_vol_target.yaml")
    assert active_parameter_keys(config) == active_parameter_keys(config.model_dump(mode="json"))
    # Order follows catalog order (allocator before timing before risk).
    assert set(default_selected_parameter_keys(config)) == set(DEFAULT_SELECTED_PARAMETER_KEYS)


def test_presets_and_configs_reference_only_master_roots() -> None:
    master = load_instruments()
    canonical = set(master.roots())
    catalog = parameter_catalog()
    referenced: set[str] = set()
    for preset in catalog.presets:
        roots = validate_strategy_roots(preset.strategy, master)
        assert set(roots) <= canonical
        referenced.update(roots)
        serialized = json.dumps(preset.strategy).upper()
        for token in DELISTED_ROOT_TOKENS:
            assert f'"{token}"' not in serialized, f"preset {preset.id} references {token}"
    for config in discover_strategy_configs("configs/strategies").values():
        strategy = config.model_dump(mode="json")
        roots = validate_strategy_roots(strategy, master)
        assert set(roots) <= canonical
        referenced.update(roots)
        serialized = json.dumps(strategy).upper()
        for token in DELISTED_ROOT_TOKENS:
            assert f'"{token}"' not in serialized, f"{config.id} references {token}"
    assert DELISTED_ROOT_TOKENS.isdisjoint(referenced)
    # The one preset that configures seasons uses canonical roots only.
    seasonal = next(p for p in catalog.presets if p.id == "seasonal_commodity_cycle")
    assert set(strategy_root_symbols(seasonal.strategy)) == {"CL", "GC"}


def test_strategy_root_symbols_extracts_regime_and_seasonality_roots() -> None:
    from configs.strategy import StageConfig

    strategy = _synthetic_strategy(
        universe=StageConfig(
            name="regime", params={"risk_off_roots": "CL,GC", "risk_off_sectors": "energy,metals"}
        ),
        timing=(
            StageConfig(
                name="seasonality",
                params={"root_month_biases": '{"ZC": {"1": 0.2}, "ZS": {"9": -0.1}}'},
            ),
        ),
    )
    symbols = strategy_root_symbols(strategy)
    assert symbols == ("CL", "GC", "ZC", "ZS")
    assert validate_strategy_roots(strategy) == symbols
    # Non-root lists are ignored.
    assert "energy" not in symbols and "metals" not in symbols


def test_validate_strategy_roots_rejects_delisted_roots() -> None:
    from configs.strategy import StageConfig

    strategy = _synthetic_strategy(
        universe=StageConfig(name="regime", params={"risk_off_roots": "CL,GE,ZQ,KE,BTC"})
    )
    with pytest.raises(ValueError, match=r"GE|ZQ|KE|BTC"):
        validate_strategy_roots(strategy)
    symbols = strategy_root_symbols(strategy)
    assert DELISTED_ROOT_TOKENS & set(symbols) == {"GE", "ZQ", "KE", "BTC"}


def test_preset_strategies_define_their_active_default_selected_parameters() -> None:
    """Every curated default the optimizer may pre-select must exist in the preset.

    v1.5.1 consistency guard: the optimizer samples the active default-selected
    keys; each of those must already be present in the preset strategy's stage
    params (no silent ``allow_new_params`` injection for generated presets).
    """
    catalog = parameter_catalog()
    for preset in catalog.presets:
        active_defaults = default_selected_parameter_keys(preset.strategy)
        for key in active_defaults:
            # ``get_parameter_value`` raises KeyError when the stage param is absent.
            value = get_parameter_value(preset.strategy, key)
            assert value is not None, (preset.id, key)
    # The curated quant-risk default is explicit in every preset's quant_risk stage.
    for preset in catalog.presets:
        risk_stages = preset.strategy.get("risk")
        assert isinstance(risk_stages, list)
        quant_risk = next(
            stage
            for stage in risk_stages
            if isinstance(stage, dict) and stage["name"] == "quant_risk"
        )
        assert quant_risk["params"]["trailing_stop_pct"] == pytest.approx(0.10)


def test_carry_filter_bounds_default_and_not_default_selected() -> None:
    """v1.5.3 Fix 1: mild contango allowed by default; carry not pre-selected."""
    catalog = parameter_catalog()
    spec = catalog.get("timing.carry_filter.min_carry_bps")
    assert spec.type is ParameterType.FLOAT
    assert spec.default == pytest.approx(-50.0)
    assert spec.min == pytest.approx(-200.0)
    assert spec.max == pytest.approx(50.0)
    assert spec.step == pytest.approx(10.0)
    assert spec.unit == "bps"
    assert "contango" in spec.description.lower()
    # Still tunable (searched when the carry_filter overlay is active) ...
    assert spec.optuna_tunable is True
    assert spec.key in {item.key for item in catalog.tunable()}
    # ... but never part of the global curated default selection.
    assert spec.optuna_default_selected is False
    assert spec.key not in set(DEFAULT_SELECTED_PARAMETER_KEYS)
    payload_entry = next(
        entry for entry in catalog_dict()["parameters"] if entry["key"] == spec.key
    )
    assert payload_entry["optuna_tunable"] is True
    assert payload_entry["optuna_default_selected"] is False
    assert payload_entry["default"] == pytest.approx(-50.0)
    assert payload_entry["min"] == pytest.approx(-200.0)
    assert payload_entry["max"] == pytest.approx(50.0)


def test_default_selected_helper_never_returns_carry() -> None:
    """Carry is active+tunable for a TSMOM config but is not pre-selected anywhere."""
    catalog = parameter_catalog()
    carry_key = "timing.carry_filter.min_carry_bps"
    tsmom_strategy = load_strategy_config("configs/strategies/tsmom_vol_target.yaml").model_dump(
        mode="json"
    )
    # The carry stage is active, so the parameter remains in the optimizer's pool.
    assert carry_key in active_parameter_keys(tsmom_strategy)
    assert carry_key not in default_selected_parameter_keys(tsmom_strategy)
    assert set(default_selected_parameter_keys(tsmom_strategy)) == CURATED_DEFAULT_KEYS
    # A preset that configures the carry overlay still does not pre-select it.
    carry_preset = next(
        preset for preset in catalog.presets if preset.id == "commodity_carry_term_structure"
    )
    assert carry_key in active_parameter_keys(carry_preset.strategy)
    assert carry_key not in default_selected_parameter_keys(carry_preset.strategy)
    # v1.5.3: no preset globally re-adds carry to the default selection.
    for preset in catalog.presets:
        assert carry_key not in default_selected_parameter_keys(preset.strategy)
    # Catalog-wide invariant: the curated five contain no carry parameter at all.
    assert all("carry" not in key for key in DEFAULT_SELECTED_PARAMETER_KEYS)
