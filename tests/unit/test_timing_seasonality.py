"""Unit tests for the seasonality timing overlay (Phase-10 Part B1)."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime

import pytest
from configs.instruments import load_instruments
from src.core.interfaces import TimingOverlay
from src.core.types import MarketState, WeightVector
from src.strategies._params import ParamError
from src.strategies.parameter_catalog import parameter_catalog
from src.strategies.pipeline import build_pipeline
from src.strategies.timing import (
    ALL_MONTHS,
    DEFAULT_ROOT_MONTH_BIASES,
    SeasonalityTiming,
    build_seasonality_timing,
    build_timing,
)

from tests.fixtures.market import make_market_state


def _state_at(month: int, *, day: int = 15) -> MarketState:
    state = make_market_state(roots=("CL", "GC", "ZN"))
    return dataclasses.replace(state, timestamp=datetime(2024, month, day, 21, 0, tzinfo=UTC))


def test_defaults_are_neutral_all_year() -> None:
    overlay = SeasonalityTiming()
    assert isinstance(overlay, TimingOverlay)
    assert overlay.name == "seasonality"
    assert overlay.active_months == ALL_MONTHS
    assert dict(overlay.root_month_biases) == {}
    for month in ALL_MONTHS:
        state = _state_at(month)
        assert overlay.scaling_factor(state, "CL") == pytest.approx(1.0)
    weights = WeightVector({"CL": 0.4, "GC": -0.2})
    assert overlay.adjust_timing(weights, _state_at(7)) == weights


def test_utc_month_boundaries() -> None:
    overlay = SeasonalityTiming(root_month_biases={"CL": {1: 0.2, 2: 0.4}})
    january_end = dataclasses.replace(
        make_market_state(roots=("CL",)),
        timestamp=datetime(2024, 1, 31, 23, 59, 59, tzinfo=UTC),
    )
    february_start = dataclasses.replace(
        make_market_state(roots=("CL",)),
        timestamp=datetime(2024, 2, 1, 0, 0, 0, tzinfo=UTC),
    )
    assert overlay.month_for(january_end) == 1
    assert overlay.month_for(february_start) == 2
    assert overlay.scaling_factor(january_end, "CL") == pytest.approx(1.2)
    assert overlay.scaling_factor(february_start, "CL") == pytest.approx(1.4)


def test_bias_tilt_and_out_of_season_scaling() -> None:
    overlay = SeasonalityTiming(
        active_months=(1, 2, 3),
        root_month_biases={"CL": {1: 0.2, 7: 0.3}, "GC": {1: -0.1}},
        out_of_season_scale=0.5,
    )
    in_season = _state_at(1)
    out_of_season = _state_at(7)
    assert overlay.scaling_factor(in_season, "CL") == pytest.approx(1.2)
    assert overlay.scaling_factor(in_season, "GC") == pytest.approx(0.9)
    # July: bias 1.3 then out-of-season scale 0.5.
    assert overlay.scaling_factor(out_of_season, "CL") == pytest.approx(1.3 * 0.5)
    assert overlay.scaling_factor(out_of_season, "GC") == pytest.approx(0.5)
    weights = WeightVector({"CL": 1.0, "GC": -1.0})
    adjusted = overlay.adjust_timing(weights, out_of_season)
    assert adjusted["CL"] == pytest.approx(0.65)
    assert adjusted["GC"] == pytest.approx(-0.5)


def test_out_of_season_scale_zero_flattens_out_of_season_legs() -> None:
    overlay = SeasonalityTiming(
        active_months=(5, 6, 7), root_month_biases={"CL": {1: 0.2}}, out_of_season_scale=0.0
    )
    adjusted = overlay.adjust_timing(WeightVector({"CL": 1.0}), _state_at(1))
    assert adjusted["CL"] == pytest.approx(0.0)
    assert overlay.adjust_timing(WeightVector({"CL": 1.0}), _state_at(6))["CL"] == pytest.approx(
        1.0
    )


def test_biases_are_frozen_and_root_normalised() -> None:
    overlay = SeasonalityTiming(root_month_biases={" cl ": {1: 0.2}})
    assert overlay.bias_for("cl", 1) == pytest.approx(0.2)
    assert overlay.bias_for("CL", 2) == pytest.approx(0.0)
    with pytest.raises(TypeError):
        overlay.root_month_biases["CL"] = {1: 0.3}  # type: ignore[index]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"active_months": ()}, "active_months"),
        ({"active_months": (0,)}, "1..12"),
        ({"active_months": (13,)}, "1..12"),
        ({"active_months": (1, 1)}, "duplicate"),
        ({"out_of_season_scale": -0.1}, "out_of_season_scale"),
        ({"out_of_season_scale": float("nan")}, "out_of_season_scale"),
        ({"root_month_biases": {"CL": {0: 0.1}}}, "1..12"),
        ({"root_month_biases": {"CL": {1: float("nan")}}}, "finite"),
        ({"root_month_biases": {"CL": {1: -1.5}}}, ">= -1.0"),
        ({"root_month_biases": {"": {1: 0.1}}}, "non-empty"),
    ],
)
def test_seasonality_rejects_invalid_arguments(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SeasonalityTiming(**kwargs)  # type: ignore[arg-type]


def test_default_curves_are_conservative_and_frozen() -> None:
    assert "CL" in DEFAULT_ROOT_MONTH_BIASES and "GC" in DEFAULT_ROOT_MONTH_BIASES
    for curve in DEFAULT_ROOT_MONTH_BIASES.values():
        assert 1 <= len(curve) <= 12
        assert all(1 <= month <= 12 for month in curve)
        assert all(abs(bias) <= 0.5 for bias in curve.values())


def test_builder_parses_comma_and_json_forms() -> None:
    comma = build_timing(
        "seasonality",
        {"active_months": "1,2,3", "root_month_biases": "CL,GC", "out_of_season_scale": 0.25},
    )
    assert isinstance(comma, SeasonalityTiming)
    assert comma.active_months == (1, 2, 3)
    assert set(comma.root_month_biases) == {"CL", "GC"}
    # The built-in CL curve has a +10% February and a +10% July bias.
    assert comma.adjust_timing(WeightVector({"CL": 1.0}), _state_at(2))["CL"] == pytest.approx(1.10)
    assert comma.adjust_timing(WeightVector({"CL": 1.0}), _state_at(7))["CL"] == pytest.approx(
        1.10 * 0.25
    )

    json_form = build_timing("seasonality", {"root_month_biases": '{"CL": {"1": 0.2, "7": 0.3}}'})
    assert isinstance(json_form, SeasonalityTiming)
    assert json_form.scaling_factor(_state_at(1), "CL") == pytest.approx(1.2)
    assert json_form.scaling_factor(_state_at(7), "CL") == pytest.approx(1.3)
    with pytest.raises(ValueError):
        build_timing("seasonality", {"active_months": "1-3"})


def test_builder_rejects_unknown_roots_and_bad_json() -> None:
    with pytest.raises(ParamError, match="no built-in seasonal curve"):
        build_timing("seasonality", {"root_month_biases": "NOT_A_ROOT"})
    with pytest.raises(ParamError, match="invalid JSON"):
        build_timing("seasonality", {"root_month_biases": '{"CL": '})
    with pytest.raises(ParamError, match="unknown parameter"):
        build_timing("seasonality", {"months": "1,2"})
    with pytest.raises(ParamError, match="active_months"):
        build_timing("seasonality", {"active_months": ""})
    assert isinstance(build_seasonality_timing({}), SeasonalityTiming)


def test_pipeline_wiring_with_named_overlay_params() -> None:
    seasonal = parameter_catalog().presets[3]
    config = seasonal.to_config()
    pipeline = build_pipeline(
        config, __import__("configs.instruments", fromlist=["load_instruments"]).load_instruments()
    )
    result = pipeline.run(_state_at(1))
    assert result.timing_weights
    assert any(
        abs(result.timing_weights.get(root) - result.base_weights.get(root)) > 1e-12
        for root in result.timing_weights
    )
    # Out-of-season scaling reduces at least one leg in July.
    july = pipeline.run(_state_at(7))
    assert july.timing_weights.gross <= result.timing_weights.gross + 1e-12


def test_mean_reversion_preset_runs_through_the_pipeline() -> None:
    preset = parameter_catalog().presets[2]
    config = preset.to_config()
    pipeline = build_pipeline(config, load_instruments())
    result = pipeline.run(make_market_state())
    assert result.final_weights is not None
    assert result.universe
