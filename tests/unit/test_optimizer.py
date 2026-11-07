"""Unit tests for the deterministic/offline walk-forward optimizer.

Optuna is an optional dependency (absent in CI), so these tests exercise the
deterministic Latin-Hypercube fallback and use injected fake walk-forward
runners/spies to prove every trial is scored on OOS folds only.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest
import src.backtest.optimizer as optimizer_module
from configs.strategy import StrategyConfig, load_strategy_config
from pydantic import ValidationError
from src.backtest import optimizer as opt
from src.backtest.optimizer import (
    HARD_INVALID_OBJECTIVE,
    OBJECTIVE_METRICS,
    OptimizationRequest,
    ParameterRange,
    TrialState,
    TrialSummary,
    WalkForwardOptimizer,
    apply_parameters,
    default_search_space,
    deterministic_lhs_parameter_sets,
    deterministic_param_importances,
    filter_search_space_for_config,
    is_legacy_search_space,
    load_optimization,
    run_optimization,
    validate_persisted_optimization,
    write_optimization_artifact,
)
from src.strategies.parameter_catalog import parameter_catalog

REPO_ROOT = Path(__file__).resolve().parents[2]
STRATEGY_PATH = REPO_ROOT / "configs" / "strategies" / "tsmom_vol_target.yaml"


@pytest.fixture()
def config() -> StrategyConfig:
    return load_strategy_config(STRATEGY_PATH)


@pytest.fixture()
def small_space() -> tuple[ParameterRange, ...]:
    return (
        ParameterRange(
            param_path="allocator.params.target_annual_volatility",
            type="float",
            low=0.05,
            high=0.25,
            step=0.01,
        ),
        ParameterRange(
            param_path="timing.0.params.lookback_fast",
            type="int",
            low=10,
            high=40,
            step=10,
        ),
        ParameterRange(
            param_path="translator.max_margin_utilization",
            type="float",
            low=0.4,
            high=0.8,
            step=0.1,
        ),
    )


def _fake_report(
    *,
    sharpe: float = 0.5,
    sortino: float = 0.7,
    calmar: float = 0.3,
    cagr: float = 0.10,
    return_on_margin: float = 0.2,
    max_drawdown: float = 0.10,
    rachev: float = 1.5,
    fold_count: int = 4,
    n_trades: int = 6,
    avg_gross_exposure: float = 0.5,
) -> Any:
    aggregate: dict[str, Any] = {
        "sharpe": {"mean": sharpe, "std": 0.1},
        "sortino": {"mean": sortino, "std": 0.1},
        "calmar": {"mean": calmar, "std": 0.1},
        "cagr": {"mean": cagr, "std": 0.02},
        "return_on_max_initial_margin": {"mean": return_on_margin, "std": 0.01},
        "max_drawdown": {"mean": max_drawdown, "std": 0.01},
        "n_folds": fold_count,
    }
    per_fold_trades = n_trades // max(1, fold_count)
    folds = tuple(
        SimpleNamespace(metrics={"rachev_ratio_05": rachev}, n_trades=per_fold_trades)
        for _ in range(fold_count)
    )
    return SimpleNamespace(
        aggregate=aggregate,
        folds=folds,
        fold_count=fold_count,
        total_trades=n_trades,
        avg_gross_exposure=avg_gross_exposure,
    )


class _SpyRunner:
    """Records every OOS runner call and returns a deterministic fake report."""

    def __init__(self, report: Any | None = None) -> None:
        self.calls: list[tuple[StrategyConfig, dict[str, Any]]] = []
        self._report = report if report is not None else _fake_report()

    def __call__(self, config: StrategyConfig, **kwargs: Any) -> Any:
        self.calls.append((config, kwargs))
        return self._report


class _ExplodingRunner:
    def __call__(self, config: StrategyConfig, **kwargs: Any) -> Any:
        del config, kwargs
        raise RuntimeError("walk-forward exploded")


def _request(
    config: StrategyConfig,
    space: tuple[ParameterRange, ...],
    *,
    n_trials: int = 2,
    objective: str = "sharpe",
    walk_forward: bool = True,
    purge_bars: int = 3,
    embargo_bars: int = 2,
    wf_splits: int = 4,
    drawdown_penalty_threshold: float = 0.20,
    min_trades: int = 3,
    min_avg_gross_exposure: float = 0.01,
) -> OptimizationRequest:
    return OptimizationRequest(
        base_strategy_id=config.id,
        objective_metric=objective,  # type: ignore[arg-type]
        n_trials=n_trials,
        walk_forward=walk_forward,
        wf_splits=wf_splits,
        purge_bars=purge_bars,
        embargo_bars=embargo_bars,
        drawdown_penalty_threshold=drawdown_penalty_threshold,
        min_trades=min_trades,
        min_avg_gross_exposure=min_avg_gross_exposure,
        search_space=space,
    )


def _optimizer(
    config: StrategyConfig,
    space: tuple[ParameterRange, ...],
    tmp_path: Path,
    *,
    runner: Any | None = None,
    n_trials: int = 2,
    seed: int = 7,
    **request_kwargs: Any,
) -> WalkForwardOptimizer:
    return WalkForwardOptimizer(
        config,
        request=_request(config, space, n_trials=n_trials, **request_kwargs),
        backtest_runner=runner if runner is not None else _SpyRunner(),
        output_dir=tmp_path,
        use_optuna=False,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Request / range models
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"param_path": "", "type": "float", "low": 0.0, "high": 1.0},
        {"param_path": "a..b", "type": "float", "low": 0.0, "high": 1.0},
        {"param_path": "a", "type": "float"},
        {"param_path": "a", "type": "float", "low": 1.0, "high": 1.0},
        {"param_path": "a", "type": "int", "low": 1.5, "high": 5.0},
        {"param_path": "a", "type": "int", "low": 1, "high": 5, "step": 0.5},
        {"param_path": "a", "type": "float", "low": 0.0, "high": 1.0, "step": 0.0},
        {"param_path": "a", "type": "categorical", "choices": ()},
        {"param_path": "a", "type": "categorical", "choices": ("x", "x")},
        {"param_path": "a", "type": "categorical", "choices": ("x",), "low": 0.0},
        {"param_path": "a", "type": "float", "low": 0.0, "high": 1.0, "choices": ("x",)},
    ],
)
def test_parameter_range_validation(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ParameterRange(**kwargs)


def test_parameter_range_valid_helpers() -> None:
    parameter = ParameterRange(param_path="a", type="float", low=1, high=5, step=0.5)
    assert parameter.numeric_bounds() == (1.0, 5.0)
    categorical = ParameterRange(param_path="b", type="categorical", choices=("a", "b"))
    with pytest.raises(ValueError):
        categorical.numeric_bounds()
    with pytest.raises(ValidationError):
        ParameterRange(param_path="c", type="bogus", low=0, high=1)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_trials": 1},
        {"n_trials": 201},
        {"n_trials": "abc"},
        {"wf_splits": 1},
        {"wf_splits": 21},
        {"purge_bars": -1},
        {"embargo_bars": -1},
        {"drawdown_penalty_threshold": 1.5},
        {"objective_metric": "profit"},
    ],
)
def test_request_validation(config: StrategyConfig, kwargs: dict[str, Any]) -> None:
    base: dict[str, Any] = {
        "base_strategy_id": config.id,
        "search_space": (
            ParameterRange(
                param_path="allocator.params.target_annual_volatility",
                type="float",
                low=0.05,
                high=0.25,
            ),
        ),
    }
    base.update(kwargs)
    with pytest.raises(ValidationError):
        OptimizationRequest(**base)


def test_request_rejects_empty_and_duplicate_search_space(config: StrategyConfig) -> None:
    with pytest.raises(ValidationError):
        OptimizationRequest(base_strategy_id=config.id, search_space=())
    duplicated = ParameterRange(
        param_path="translator.max_margin_utilization", type="float", low=0.4, high=0.8
    )
    with pytest.raises(ValidationError):
        OptimizationRequest(base_strategy_id=config.id, search_space=(duplicated, duplicated))
    with pytest.raises(ValidationError):
        OptimizationRequest(base_strategy_id="  ", search_space=(duplicated,))


# ---------------------------------------------------------------------------
# Deterministic fallback sampler
# ---------------------------------------------------------------------------


def test_lhs_sampler_determinism(small_space: tuple[ParameterRange, ...]) -> None:
    first = deterministic_lhs_parameter_sets(small_space, 8, seed=7)
    second = deterministic_lhs_parameter_sets(small_space, 8, seed=7)
    other = deterministic_lhs_parameter_sets(small_space, 8, seed=99)
    assert first == second
    assert len(first) == 8
    assert first != other
    assert len({tuple(sorted(params.items())) for params in first}) > 1


def test_lhs_sampler_bounds_and_step_alignment() -> None:
    space = (
        ParameterRange(param_path="i", type="int", low=10, high=40, step=10),
        ParameterRange(param_path="f", type="float", low=0.05, high=0.25, step=0.05),
        ParameterRange(param_path="c", type="categorical", choices=("alpha", "beta", "gamma")),
    )
    parameter_sets = deterministic_lhs_parameter_sets(space, 20, seed=1)
    assert len(parameter_sets) == 20
    int_values: set[int] = set()
    float_values: set[float] = set()
    categorical_values: set[str] = set()
    for params in parameter_sets:
        int_value = params["i"]
        float_value = params["f"]
        choice = params["c"]
        assert isinstance(int_value, int)
        assert 10 <= int_value <= 40
        assert (int_value - 10) % 10 == 0
        assert isinstance(float_value, float)
        assert 0.05 - 1e-9 <= float_value <= 0.25 + 1e-9
        assert round((float_value - 0.05) / 0.05) == pytest.approx(
            (float_value - 0.05) / 0.05, abs=1e-6
        )
        assert choice in {"alpha", "beta", "gamma"}
        int_values.add(int_value)
        float_values.add(float_value)
        categorical_values.add(str(choice))
    assert len(int_values) == 4  # 10, 20, 30, 40
    assert len(float_values) == 5  # 0.05 .. 0.25
    assert categorical_values == {"alpha", "beta", "gamma"}


def test_lhs_sampler_validation(small_space: tuple[ParameterRange, ...]) -> None:
    with pytest.raises(ValueError):
        deterministic_lhs_parameter_sets(small_space, 0)
    with pytest.raises(TypeError):
        deterministic_lhs_parameter_sets(small_space, 4, seed="x")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        deterministic_lhs_parameter_sets((), 4)


def test_deterministic_param_importances() -> None:
    space = (
        ParameterRange(param_path="x", type="float", low=0.0, high=1.0),
        ParameterRange(param_path="y", type="float", low=0.0, high=1.0),
    )
    trials = (
        TrialSummary(trial_number=0, params={"x": 0.0, "y": 1.0}, objective_value=0.0),
        TrialSummary(trial_number=1, params={"x": 1.0, "y": 0.0}, objective_value=1.0),
        TrialSummary(trial_number=2, params={"x": 0.5, "y": 0.5}, objective_value=0.5),
    )
    importances = deterministic_param_importances(space, trials)
    assert set(importances) == {"x", "y"}
    assert sum(importances.values()) == pytest.approx(1.0)
    # Both parameters are perfectly correlated with the objective (x + y = 1),
    # so the normalised importance is split equally.
    assert importances["x"] == pytest.approx(0.5)
    assert importances["y"] == pytest.approx(0.5)
    assert deterministic_param_importances(space, trials[:1]) == {"x": 0.0, "y": 0.0}
    # Constant objective -> no importance signal.
    constant = (
        TrialSummary(trial_number=0, params={"x": 0.0, "y": 0.0}, objective_value=1.0),
        TrialSummary(trial_number=1, params={"x": 1.0, "y": 1.0}, objective_value=1.0),
    )
    assert deterministic_param_importances(space, constant) == {"x": 0.0, "y": 0.0}


# ---------------------------------------------------------------------------
# Parameter application
# ---------------------------------------------------------------------------


def test_apply_parameters_by_dotted_path(config: StrategyConfig) -> None:
    applied = apply_parameters(
        config,
        {
            "allocator.params.target_annual_volatility": 0.18,
            "timing.0.params.lookback_fast": 15,
            "timing.tsmom.params.lookback_slow": 90,
            "risk.0.params.max_single_instrument_weight": 0.25,
            "translator.max_margin_utilization": 0.55,
        },
    )
    assert applied.allocator.params["target_annual_volatility"] == pytest.approx(0.18)
    assert applied.timing[0].params["lookback_fast"] == 15
    assert applied.timing[0].params["lookback_slow"] == 90
    assert applied.risk[0].params["max_single_instrument_weight"] == pytest.approx(0.25)
    assert applied.translator.max_margin_utilization == pytest.approx(0.55)
    # The base config remains immutable / untouched.
    assert config.allocator.params["target_annual_volatility"] == pytest.approx(0.12)


def test_apply_parameters_validation(config: StrategyConfig) -> None:
    with pytest.raises(ValueError, match="unknown parameter path"):
        apply_parameters(config, {"allocator.params.does_not_exist": 1.0})
    with pytest.raises(ValueError, match="unknown parameter path"):
        apply_parameters(config, {"no_such_section.params.value": 1.0})
    with pytest.raises(ValueError, match="unknown parameter path"):
        apply_parameters(config, {"timing.9.params.lookback_fast": 10})
    with pytest.raises(ValueError, match="unknown parameter path"):
        apply_parameters(config, {"timing.not_a_stage.params.lookback_fast": 10})
    with pytest.raises(ValueError, match="invalid StrategyConfig"):
        apply_parameters(config, {"translator.max_margin_utilization": 5.0})
    # Catalogued stage params omitted from the base config stay strict by default
    # and become injectable only for optimizer-sampled defaults.
    with pytest.raises(ValueError, match="unknown parameter path"):
        apply_parameters(config, {"risk.0.params.trailing_stop_pct": 0.10})
    injected = apply_parameters(
        config,
        {"risk.0.params.trailing_stop_pct": 0.10},
        allow_new_params=True,
    )
    assert injected.risk[0].params["trailing_stop_pct"] == pytest.approx(0.10)


def test_optimizer_filters_inactive_search_space_paths(
    config: StrategyConfig, tmp_path: Path
) -> None:
    inactive_spaces = (
        (ParameterRange(param_path="allocator.name", type="float", low=0.0, high=1.0),),
        (ParameterRange(param_path="allocator.params.nope", type="float", low=0.0, high=1.0),),
        (
            ParameterRange(
                param_path="timing.mean_reversion.rsi_period",
                type="int",
                low=2,
                high=20,
            ),
        ),
        (
            ParameterRange(
                param_path="universe.params.top_n",
                type="int",
                low=3,
                high=20,
            ),
        ),
    )
    for space in inactive_spaces:
        with pytest.raises(ValueError, match="no active tunable parameters"):
            WalkForwardOptimizer(
                config,
                request=_request(config, space),
                backtest_runner=_SpyRunner(),
                output_dir=tmp_path,
                use_optuna=False,
            )


def test_optimizer_filters_explicit_ui_search_space(
    config: StrategyConfig,
    tmp_path: Path,
) -> None:
    explicit = (
        ParameterRange(param_path="timing.tsmom.lookback_fast", type="int", low=5, high=60, step=5),
        ParameterRange(param_path="timing.mean_reversion.rsi_period", type="int", low=5, high=30),
        ParameterRange(param_path="universe.params.top_n", type="int", low=3, high=20),
        ParameterRange(
            param_path="allocator.params.target_annual_volatility",
            type="float",
            low=0.05,
            high=0.30,
            step=0.01,
        ),
    )
    spy = _SpyRunner()
    optimizer = WalkForwardOptimizer(
        config,
        request=_request(config, explicit),
        backtest_runner=spy,
        output_dir=tmp_path,
        use_optuna=False,
    )
    filtered_paths = [parameter.param_path for parameter in optimizer.request.search_space]
    assert filtered_paths == [
        "timing.0.params.lookback_fast",
        "allocator.params.target_annual_volatility",
    ]
    result = optimizer.optimize()
    for trial in result.trials:
        assert set(trial.params) == {
            "timing.0.params.lookback_fast",
            "allocator.params.target_annual_volatility",
        }
    assert [parameter.param_path for parameter in result.request.search_space] == filtered_paths


def test_filter_search_space_for_config(config: StrategyConfig) -> None:
    explicit = (
        ParameterRange(
            param_path="allocator.params.target_annual_volatility",
            type="float",
            low=0.05,
            high=0.30,
            step=0.01,
        ),
        ParameterRange(param_path="timing.mean_reversion.rsi_period", type="int", low=2, high=20),
        ParameterRange(param_path="universe.params.top_n", type="int", low=3, high=20),
        ParameterRange(param_path="timing.tsmom.lookback_fast", type="int", low=5, high=60, step=5),
        ParameterRange(
            param_path="allocator.params.target_annual_volatility",
            type="float",
            low=0.10,
            high=0.20,
        ),
    )
    filtered = filter_search_space_for_config(explicit, config)
    assert [parameter.param_path for parameter in filtered] == [
        "allocator.params.target_annual_volatility",
        "timing.0.params.lookback_fast",
    ]
    # The first occurrence wins (order kept, duplicates removed).
    assert filtered[0].low == pytest.approx(0.05)
    assert filtered[0].high == pytest.approx(0.30)
    with pytest.raises(ValueError, match="search space is empty"):
        filter_search_space_for_config((), config)
    with pytest.raises(ValueError, match="no active tunable parameters"):
        filter_search_space_for_config(
            (
                ParameterRange(
                    param_path="timing.mean_reversion.rsi_period",
                    type="int",
                    low=2,
                    high=20,
                ),
                ParameterRange(param_path="universe.params.top_n", type="int", low=3, high=20),
            ),
            config,
        )


def test_filter_search_space_for_config_fallback_without_catalog(
    config: StrategyConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(optimizer_module, "_catalog_search_context", lambda config: None)
    space = (
        ParameterRange(
            param_path="allocator.params.target_annual_volatility",
            type="float",
            low=0.05,
            high=0.30,
        ),
        ParameterRange(param_path="universe.params.top_n", type="int", low=3, high=20),
        ParameterRange(param_path="timing.mean_reversion.rsi_period", type="int", low=2, high=20),
    )
    filtered = filter_search_space_for_config(space, config)
    assert [parameter.param_path for parameter in filtered] == [
        "allocator.params.target_annual_volatility"
    ]


def test_optimizer_rejects_mismatched_strategy_id(config: StrategyConfig, tmp_path: Path) -> None:
    space = (
        ParameterRange(
            param_path="allocator.params.target_annual_volatility",
            type="float",
            low=0.05,
            high=0.25,
        ),
    )
    request = OptimizationRequest(base_strategy_id="other_strategy", search_space=space)
    with pytest.raises(ValueError, match="does not match"):
        WalkForwardOptimizer(
            config,
            request=request,
            backtest_runner=_SpyRunner(),
            output_dir=tmp_path,
            use_optuna=False,
        )


def test_default_search_space_is_active_only(config: StrategyConfig) -> None:
    space = default_search_space(config)
    paths = [parameter.param_path for parameter in space]
    assert paths == [
        "allocator.params.target_annual_volatility",
        "timing.0.params.lookback_fast",
        "timing.0.params.lookback_slow",
        "timing.1.params.window",
        "risk.0.params.trailing_stop_pct",
    ]
    assert len(paths) == len(set(paths))
    # No data-hygiene/structural, inactive overlays, LLM plumbing or translator
    # execution parameters in the default space.
    forbidden_fragments = (
        "min_avg_dollar_volume",
        "min_open_interest",
        "min_history_days",
        "top_n",
        "mean_reversion",
        "llm_risk",
        "macro_lockout",
        "translator",
    )
    for path in paths:
        assert not any(fragment in path for fragment in forbidden_fragments), path
    # Bounds/steps come from the strategy catalog (not the old heuristics).
    catalog = parameter_catalog()
    fast = catalog.get("timing.tsmom.lookback_fast")
    slow = catalog.get("timing.tsmom.lookback_slow")
    fast_range = next(p for p in space if p.param_path == "timing.0.params.lookback_fast")
    slow_range = next(p for p in space if p.param_path == "timing.0.params.lookback_slow")
    assert fast_range.low == pytest.approx(float(fast.min))
    assert fast_range.high <= float(fast.max)
    assert fast_range.step == pytest.approx(float(fast.step))
    assert slow_range.low >= float(slow.min)
    assert slow_range.high <= float(slow.max)
    # Curated defaults clip the catalog bounds to classic signal horizons.
    assert fast_range.high == pytest.approx(63.0)
    assert slow_range.low == pytest.approx(64.0)
    assert slow_range.high == pytest.approx(252.0)
    # Fast/slow lookbacks are partitioned by the deterministic constraint so the
    # default space never samples an invalid TSMOM definition.
    assert fast_range.high < slow_range.low
    for parameter in space:
        assert parameter.numeric_bounds()[0] < parameter.numeric_bounds()[1]
        assert parameter.type in {"int", "float"}


def test_default_lookback_pair_is_constrained(config: StrategyConfig) -> None:
    space = default_search_space(config)
    parameter_sets = deterministic_lhs_parameter_sets(space, 40, seed=3)
    for params in parameter_sets:
        fast = params["timing.0.params.lookback_fast"]
        slow = params["timing.0.params.lookback_slow"]
        assert isinstance(fast, int) and isinstance(slow, int)
        assert fast < slow, params


def test_default_search_space_panic_mean_reversion() -> None:
    preset = next(
        template
        for template in parameter_catalog().presets
        if template.id == "panic_mean_reversion"
    )
    panic_config = preset.to_config()
    space = default_search_space(panic_config)
    paths = [parameter.param_path for parameter in space]
    assert paths == [
        "allocator.params.target_annual_volatility",
        "timing.0.params.rsi_period",
        "timing.0.params.oversold_threshold",
        "timing.0.params.overbought_threshold",
        "timing.0.params.turbulence_veto_threshold",
        "timing.0.params.reversion_strength",
        "risk.0.params.trailing_stop_pct",
    ]
    assert not any(
        ("tsmom" in path) or ("kama" in path) or ("carry_filter" in path) for path in paths
    )
    assert not any("llm_risk" in path or "translator" in path for path in paths)
    assert len(space) == 7


def test_default_search_space_fallback_without_catalog(
    config: StrategyConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(optimizer_module, "_catalog_search_context", lambda config: None)
    space = default_search_space(config)
    paths = [parameter.param_path for parameter in space]
    assert paths == [
        "allocator.params.target_annual_volatility",
        "timing.0.params.lookback_fast",
        "timing.0.params.lookback_slow",
        "timing.1.params.window",
    ]


# ---------------------------------------------------------------------------
# Zero-activity / flat-trial penalty + warm-up bars (v1.5.3)
# ---------------------------------------------------------------------------


def test_request_zero_activity_fields_and_validation(
    config: StrategyConfig,
    small_space: tuple[ParameterRange, ...],
) -> None:
    request = _request(config, small_space)
    assert request.min_trades == 3
    assert request.min_avg_gross_exposure == pytest.approx(0.01)
    base: dict[str, Any] = {
        "base_strategy_id": config.id,
        "search_space": small_space,
    }
    with pytest.raises(ValidationError):
        OptimizationRequest(**{**base, "min_trades": -1})
    with pytest.raises(ValidationError):
        OptimizationRequest(**{**base, "min_avg_gross_exposure": -0.1})
    with pytest.raises(ValidationError):
        OptimizationRequest(**{**base, "min_avg_gross_exposure": 1.5})


def test_trial_summary_activity_defaults() -> None:
    trial = TrialSummary(trial_number=0)
    assert trial.n_trades == 0
    assert trial.avg_gross_exposure == pytest.approx(0.0)
    assert trial.zero_activity is False
    payload = trial.to_dict()
    assert payload["n_trades"] == 0
    assert payload["avg_gross_exposure"] == pytest.approx(0.0)
    assert payload["zero_activity"] is False


def _flat_report(**overrides: Any) -> Any:
    return _fake_report(n_trades=0, avg_gross_exposure=0.0, **overrides)


def test_flat_trials_are_penalized_and_never_best(
    config: StrategyConfig,
    small_space: tuple[ParameterRange, ...],
    tmp_path: Path,
) -> None:
    optimizer = _optimizer(
        config,
        small_space,
        tmp_path,
        runner=_SpyRunner(_flat_report()),
        n_trials=3,
    )
    result = optimizer.optimize()
    assert result.trials
    for trial in result.trials:
        assert trial.state is TrialState.COMPLETE
        assert trial.zero_activity is True
        assert trial.objective_value == pytest.approx(HARD_INVALID_OBJECTIVE)
        assert trial.n_trades == 0
        assert trial.avg_gross_exposure == pytest.approx(0.0)
        # Raw metrics are still recorded for diagnosis.
        assert trial.raw_objective == pytest.approx(0.5)
    assert result.best_trial_number is None
    assert result.best_params == {}
    assert result.best_value is None
    assert result.best_raw_value is None
    assert result.best_strategy_config == {}
    assert result.fold_count == 0
    assert result.completed_trials == 3
    assert result.failed_trials == 0
    assert result.artifact_path is not None
    persisted = load_optimization(result.artifact_path)
    assert persisted.best_trial_number is None
    assert all(trial.zero_activity for trial in persisted.trials)


def test_low_exposure_trial_is_zero_activity(
    config: StrategyConfig,
    small_space: tuple[ParameterRange, ...],
    tmp_path: Path,
) -> None:
    report = _fake_report(n_trades=10, avg_gross_exposure=0.001)
    result = _optimizer(
        config, small_space, tmp_path, runner=_SpyRunner(report), n_trials=2
    ).optimize()
    for trial in result.trials:
        assert trial.n_trades == 10
        assert trial.avg_gross_exposure == pytest.approx(0.001)
        assert trial.zero_activity is True
        assert trial.objective_value == pytest.approx(HARD_INVALID_OBJECTIVE)
    assert result.best_trial_number is None


def test_activity_thresholds_are_inclusive(
    config: StrategyConfig,
    small_space: tuple[ParameterRange, ...],
    tmp_path: Path,
) -> None:
    report = _fake_report(n_trades=3, avg_gross_exposure=0.01)
    result = _optimizer(
        config,
        small_space,
        tmp_path,
        runner=_SpyRunner(report),
        n_trials=2,
        min_trades=3,
        min_avg_gross_exposure=0.01,
    ).optimize()
    for trial in result.trials:
        assert trial.zero_activity is False
        assert trial.objective_value == pytest.approx(0.5)
    assert result.best_trial_number is not None


class _SequenceRunner:
    """Runner returning a scripted report per trial call."""

    def __init__(self, reports: list[Any]) -> None:
        self._reports = reports
        self.calls: list[dict[str, Any]] = []

    def __call__(self, config: StrategyConfig, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self._reports[min(len(self.calls) - 1, len(self._reports) - 1)]


def test_active_trial_wins_over_higher_raw_flat_trial(
    config: StrategyConfig,
    small_space: tuple[ParameterRange, ...],
    tmp_path: Path,
) -> None:
    runner = _SequenceRunner(
        [
            _fake_report(sharpe=0.10, n_trades=6, avg_gross_exposure=0.5),
            _fake_report(sharpe=0.90, n_trades=0, avg_gross_exposure=0.0),
        ]
    )
    result = _optimizer(config, small_space, tmp_path, runner=runner, n_trials=2).optimize()
    assert [trial.zero_activity for trial in result.trials] == [False, True]
    assert result.trials[1].raw_objective == pytest.approx(0.90)
    assert result.trials[1].objective_value == pytest.approx(HARD_INVALID_OBJECTIVE)
    assert result.best_trial_number == 0
    assert result.best_value == pytest.approx(0.10)
    assert result.trials[0].n_trades == 6


def test_activity_falls_back_to_combined_result_equity_curve(
    config: StrategyConfig,
    small_space: tuple[ParameterRange, ...],
    tmp_path: Path,
) -> None:
    combined = SimpleNamespace(
        n_trades=8,
        equity_curve=pl.DataFrame(
            {
                "equity": [1_000_000.0, 1_000_000.0, 1_000_000.0],
                "gross_exposure": [500_000.0, 200_000.0, 500_000.0],
            }
        ),
    )
    report = SimpleNamespace(
        aggregate={"sharpe": {"mean": 0.5, "std": 0.1}},
        folds=(SimpleNamespace(metrics={"rachev_ratio_05": 1.5}, n_trades=4),) * 2,
        fold_count=2,
        combined_result=combined,
    )
    result = _optimizer(
        config, small_space, tmp_path, runner=_SpyRunner(report), n_trials=2
    ).optimize()
    for trial in result.trials:
        assert trial.n_trades == 8  # summed over folds when total_trades is absent
        assert trial.avg_gross_exposure == pytest.approx((0.5 + 0.2 + 0.5) / 3.0)
        assert trial.zero_activity is False


def test_default_search_space_trials_plumb_activity(
    config: StrategyConfig,
    tmp_path: Path,
) -> None:
    report = _fake_report(n_trades=7, avg_gross_exposure=0.30)
    result = _optimizer(
        config,
        default_search_space(config),
        tmp_path,
        runner=_SpyRunner(report),
        n_trials=2,
    ).optimize()
    for trial in result.trials:
        assert trial.n_trades == 7
        assert trial.avg_gross_exposure == pytest.approx(0.30)
        assert trial.zero_activity is False
    assert result.best_trial_number is not None
    payload = result.to_dict()
    assert payload["trials"][0]["n_trades"] == 7
    assert payload["trials"][0]["zero_activity"] is False


def test_optuna_backend_flat_trials_not_best(
    config: StrategyConfig,
    small_space: tuple[ParameterRange, ...],
    tmp_path: Path,
) -> None:
    if optimizer_module._load_optuna_module() is None:
        pytest.skip("optuna is not installed in this environment")
    optimizer = WalkForwardOptimizer(
        config,
        request=_request(config, small_space, n_trials=2),
        backtest_runner=_SpyRunner(_flat_report()),
        output_dir=tmp_path,
        use_optuna=True,
    )
    result = optimizer.optimize()
    assert result.backend.value == "optuna"
    assert result.best_trial_number is None
    assert result.best_params == {}
    assert all(trial.zero_activity for trial in result.trials)
    assert all(
        trial.objective_value == pytest.approx(HARD_INVALID_OBJECTIVE) for trial in result.trials
    )


def test_offline_window_is_at_least_750_with_warmup(
    config: StrategyConfig,
    tmp_path: Path,
) -> None:
    optimizer = _optimizer(
        config,
        default_search_space(config),
        tmp_path,
        n_trials=2,
    )
    request = optimizer.request
    periods = optimizer._offline_bar_periods()
    formula = 150 * request.wf_splits + 2 * (request.purge_bars + request.embargo_bars) + 300
    warmup_window = request.purge_bars + (request.wf_splits + 1) * 252
    assert periods >= 750
    assert periods >= formula
    assert periods >= warmup_window
    assert optimizer._warmup_periods() >= 252
    bars, states = optimizer._prepare_window()
    assert bars is not None and states is not None
    timestamps = int(bars["timestamp_utc"].n_unique())
    assert timestamps >= 750
    assert len(states) == timestamps
    # States are built over the full window (train->test warm-up), not truncated.
    assert len({state.timestamp for state in states}) == timestamps


def test_in_sample_offline_window_stays_compatible(
    config: StrategyConfig,
    small_space: tuple[ParameterRange, ...],
    tmp_path: Path,
) -> None:
    request = _request(config, small_space, n_trials=2, walk_forward=False)
    optimizer = WalkForwardOptimizer(
        config,
        request=request,
        backtest_runner=_SpyRunner(),
        output_dir=tmp_path,
        use_optuna=False,
    )
    assert optimizer._offline_bar_periods() == 260


# ---------------------------------------------------------------------------
# OOS-only evaluation + objective / penalty
# ---------------------------------------------------------------------------


def test_trials_use_walk_forward_oos_only(
    config: StrategyConfig,
    small_space: tuple[ParameterRange, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = _SpyRunner()
    monkeypatch.setattr(
        optimizer_module,
        "run_from_config",
        lambda *args, **kwargs: pytest.fail("in-sample backtest must not be used"),
    )
    optimizer = _optimizer(config, small_space, tmp_path, runner=spy, n_trials=3)
    result = optimizer.optimize()
    assert len(spy.calls) == 3
    assert len(result.trials) == 3
    for called_config, kwargs in spy.calls:
        assert isinstance(called_config, StrategyConfig)
        assert "bars" in kwargs and "states" in kwargs
        assert kwargs["splits"] == 4
        assert kwargs["purge_bars"] == 3
        assert kwargs["embargo_bars"] == 2
        assert kwargs["offline"] is False
        assert kwargs["instrument_master"] is not None
    # Objective is derived from the OOS aggregate fake (sharpe mean 0.5).
    assert all(trial.raw_objective == pytest.approx(0.5) for trial in result.trials)
    assert all(trial.oos_rachev == pytest.approx(1.5) for trial in result.trials)
    assert result.walk_forward is True
    assert result.fold_count == 4


def test_objective_metric_mapping(
    config: StrategyConfig,
    small_space: tuple[ParameterRange, ...],
    tmp_path: Path,
) -> None:
    expected = {
        "sharpe": 0.5,
        "sortino": 0.7,
        "calmar": 0.3,
        "cagr": 0.10,
        "rachev": 1.5,
        "return_on_margin": 0.2,
    }
    for metric in OBJECTIVE_METRICS:
        optimizer = _optimizer(
            config,
            small_space,
            tmp_path,
            n_trials=2,
            objective=metric,
            drawdown_penalty_threshold=0.9,  # no penalty for 0.10 OOS drawdown
        )
        result = optimizer.optimize()
        best = result.trials[result.best_trial_number or 0]
        assert best.raw_objective == pytest.approx(expected[metric])
        assert best.objective_value == pytest.approx(expected[metric])


def test_drawdown_penalty_applied_above_threshold(
    config: StrategyConfig,
    small_space: tuple[ParameterRange, ...],
    tmp_path: Path,
) -> None:
    report = _fake_report(sharpe=0.5, max_drawdown=0.30)
    optimizer = _optimizer(
        config,
        small_space,
        tmp_path,
        runner=_SpyRunner(report),
        n_trials=2,
        drawdown_penalty_threshold=0.20,
    )
    result = optimizer.optimize()
    for trial in result.trials:
        assert trial.raw_objective == pytest.approx(0.5)
        assert trial.max_dd == pytest.approx(0.30)
        assert trial.objective_value == pytest.approx(0.5 - (0.30 - 0.20))
    assert result.drawdown_penalty_threshold == pytest.approx(0.20)

    doubled = _optimizer(
        config,
        small_space,
        tmp_path,
        runner=_SpyRunner(report),
        n_trials=2,
    )
    doubled._penalty_coefficient = 2.0  # type: ignore[attr-defined]
    doubled_result = doubled.optimize()
    assert doubled_result.trials[0].objective_value == pytest.approx(0.5 - 2 * 0.10)

    below = _optimizer(
        config,
        small_space,
        tmp_path,
        runner=_SpyRunner(_fake_report(sharpe=0.5, max_drawdown=0.05)),
        n_trials=2,
    )
    below_result = below.optimize()
    assert below_result.trials[0].objective_value == pytest.approx(0.5)


def test_in_sample_opt_in_uses_single_run(
    config: StrategyConfig,
    small_space: tuple[ParameterRange, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = SimpleNamespace(
        n_trades=6,
        equity_curve=pl.DataFrame({"equity": [1_000_000.0], "gross_exposure": [500_000.0]}),
    )
    monkeypatch.setattr(optimizer_module, "run_from_config", lambda *args, **kwargs: sentinel)
    fake_metrics = SimpleNamespace(
        sharpe=1.0,
        sortino=1.1,
        calmar=0.9,
        cagr=0.2,
        rachev_ratio_05=1.3,
        return_on_max_initial_margin=0.3,
        max_drawdown=0.1,
    )
    monkeypatch.setattr(
        "src.analytics.metrics.compute_metrics_from_result",
        lambda result: fake_metrics,
    )
    spy = _SpyRunner()
    optimizer = _optimizer(
        config,
        small_space,
        tmp_path,
        runner=spy,
        n_trials=2,
        walk_forward=False,
    )
    result = optimizer.optimize()
    assert spy.calls == []  # no walk-forward evaluation for the opt-in in-sample path
    assert result.walk_forward is False
    assert result.trials[0].raw_objective == pytest.approx(1.0)
    assert result.fold_count == 1


# ---------------------------------------------------------------------------
# Backend selection / failures / determinism
# ---------------------------------------------------------------------------


def test_backend_selection_is_deterministic_without_optuna(
    config: StrategyConfig,
    small_space: tuple[ParameterRange, ...],
    tmp_path: Path,
) -> None:
    forced = _optimizer(config, small_space, tmp_path)
    assert forced.backend.value == "deterministic"
    if optimizer_module._load_optuna_module() is None:
        auto = WalkForwardOptimizer(
            config,
            request=_request(config, small_space),
            backtest_runner=_SpyRunner(),
            output_dir=tmp_path,
            use_optuna=None,
        )
        assert auto.backend.value == "deterministic"
        with pytest.raises(RuntimeError, match="Optuna is not installed"):
            WalkForwardOptimizer(
                config,
                request=_request(config, small_space),
                backtest_runner=_SpyRunner(),
                output_dir=tmp_path,
                use_optuna=True,
            )


def test_default_output_dir_uses_phase10_settings(
    config: StrategyConfig,
    small_space: tuple[ParameterRange, ...],
) -> None:
    expected = optimizer_module._default_output_dir()
    optimizer = WalkForwardOptimizer(
        config,
        request=_request(config, small_space),
        backtest_runner=_SpyRunner(),
        use_optuna=False,
    )
    assert optimizer._output_dir == expected
    assert optimizer._output_dir.name == "optimizations"


def test_optuna_backend_when_installed(
    config: StrategyConfig,
    tmp_path: Path,
) -> None:
    optuna_module = optimizer_module._load_optuna_module()
    if optuna_module is None:
        pytest.skip("optuna is not installed in this environment")
    spy = _SpyRunner()
    space = default_search_space(config)
    optimizer = WalkForwardOptimizer(
        config,
        request=_request(config, space, n_trials=2),
        backtest_runner=spy,
        output_dir=tmp_path,
        use_optuna=True,
    )
    result = optimizer.optimize()
    assert result.backend.value == "optuna"
    assert len(spy.calls) == 2
    assert result.completed_trials == 2
    assert result.failed_trials == 0
    for trial in result.trials:
        assert trial.state is TrialState.COMPLETE
        assert "timing.0.params.lookback_fast" in trial.params


def test_same_seed_is_reproducible(
    config: StrategyConfig,
    small_space: tuple[ParameterRange, ...],
    tmp_path: Path,
) -> None:
    first = _optimizer(config, small_space, tmp_path / "a", n_trials=5, seed=11).optimize()
    second = _optimizer(config, small_space, tmp_path / "b", n_trials=5, seed=11).optimize()
    other = _optimizer(config, small_space, tmp_path / "c", n_trials=5, seed=12).optimize()
    assert first.study_id == second.study_id
    assert [trial.params for trial in first.trials] == [trial.params for trial in second.trials]
    assert [trial.objective_value for trial in first.trials] == [
        trial.objective_value for trial in second.trials
    ]
    assert first.best_params == second.best_params
    assert first.param_importances == second.param_importances
    assert [trial.params for trial in first.trials] != [trial.params for trial in other.trials]


def test_optimize_accepts_runner_override(
    config: StrategyConfig,
    small_space: tuple[ParameterRange, ...],
    tmp_path: Path,
) -> None:
    spy = _SpyRunner()
    optimizer = _optimizer(config, small_space, tmp_path, runner=_ExplodingRunner(), n_trials=2)
    result = optimizer.optimize(backtest_runner=spy)
    assert len(spy.calls) == 2
    assert result.completed_trials == 2
    # The constructor-injected runner is restored after the study.
    assert optimizer._backtest_runner is not None
    with pytest.raises(RuntimeError):
        optimizer.optimize()
    with pytest.raises(TypeError):
        optimizer.optimize(backtest_runner="not-callable")  # type: ignore[arg-type]


def test_all_trials_failing_raises(
    config: StrategyConfig,
    small_space: tuple[ParameterRange, ...],
    tmp_path: Path,
) -> None:
    optimizer = _optimizer(config, small_space, tmp_path, runner=_ExplodingRunner(), n_trials=2)
    with pytest.raises(RuntimeError, match="all 2 optimization trials failed"):
        optimizer.optimize()


def test_failed_trial_summary_is_recorded_when_partial(
    config: StrategyConfig,
    small_space: tuple[ParameterRange, ...],
    tmp_path: Path,
) -> None:
    class _FlakyRunner:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, config: StrategyConfig, **kwargs: Any) -> Any:
            del config, kwargs
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("transient failure")
            return _fake_report()

    optimizer = _optimizer(config, small_space, tmp_path, runner=_FlakyRunner(), n_trials=2)
    result = optimizer.optimize()
    assert result.completed_trials == 1
    assert result.failed_trials == 1
    failed = [trial for trial in result.trials if trial.state is TrialState.FAIL]
    assert failed and failed[0].error is not None


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_persistence_round_trip(
    config: StrategyConfig,
    small_space: tuple[ParameterRange, ...],
    tmp_path: Path,
) -> None:
    result = _optimizer(config, small_space, tmp_path, n_trials=3).optimize()
    assert result.artifact_path is not None
    artifact = Path(result.artifact_path)
    assert artifact.is_file()
    assert artifact.parent == tmp_path
    assert list(tmp_path.glob("*.tmp")) == []
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["study_id"] == result.study_id
    assert payload["schema_version"] == opt.OPTIMIZATION_SCHEMA_VERSION
    assert payload["request"]["base_strategy_id"] == config.id
    loaded = load_optimization(artifact)
    assert loaded.to_dict() == result.to_dict()
    by_id = load_optimization(result.study_id, output_dir=tmp_path)
    assert by_id.to_dict() == result.to_dict()
    # A second explicit write path also round-trips.
    alternate = tmp_path / "copy.json"
    write_optimization_artifact(result, alternate)
    assert load_optimization(alternate).to_dict() == result.to_dict()
    with pytest.raises(FileNotFoundError):
        load_optimization("no-such-study", output_dir=tmp_path)
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid optimization artifact JSON"):
        load_optimization(malformed)
    invalid_payload = tmp_path / "invalid.json"
    invalid_payload.write_text(json.dumps({"study_id": "x"}), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid optimization artifact"):
        load_optimization(invalid_payload)


def test_result_json_contract(
    config: StrategyConfig,
    small_space: tuple[ParameterRange, ...],
    tmp_path: Path,
) -> None:
    result = _optimizer(config, small_space, tmp_path, n_trials=2).optimize()
    payload = result.to_dict()
    assert set(payload) >= {
        "schema_version",
        "study_id",
        "base_strategy_id",
        "objective_metric",
        "direction",
        "n_trials",
        "completed_trials",
        "failed_trials",
        "best_trial_number",
        "best_params",
        "best_value",
        "best_raw_value",
        "best_strategy_config",
        "trials",
        "param_importances",
        "drawdown_penalty_threshold",
        "walk_forward",
        "fold_count",
        "backend",
        "seed",
        "created_at",
        "duration_ms",
        "artifact_path",
        "request",
    }
    json.dumps(payload)
    assert payload["direction"] == "maximize"
    assert payload["best_strategy_config"]["id"] == config.id
    assert len(payload["trials"]) == 2
    assert payload["trials"][0]["state"] == "COMPLETE"
    assert {"n_trades", "avg_gross_exposure", "zero_activity"} <= set(payload["trials"][0])
    assert set(payload["param_importances"]) == {parameter.param_path for parameter in small_space}
    rebuilt = opt.OptimizationResult.from_dict(payload)
    assert rebuilt.to_dict() == payload
    assert run_optimization(config, _request(config, small_space), output_dir=tmp_path) is not None


# ---------------------------------------------------------------------------
# Persisted-study validation (legacy/inactive search spaces)
# ---------------------------------------------------------------------------


def _clean_persisted_study(
    config: StrategyConfig,
    tmp_path: Path,
) -> tuple[Any, Path]:
    optimizer = _optimizer(
        config,
        default_search_space(config),
        tmp_path,
        n_trials=2,
    )
    result = optimizer.optimize()
    assert result.artifact_path is not None
    artifact = Path(result.artifact_path)
    return load_optimization(artifact), artifact


def test_clean_v151_study_is_valid(config: StrategyConfig, tmp_path: Path) -> None:
    loaded, artifact = _clean_persisted_study(config, tmp_path)
    original_bytes = artifact.read_bytes()
    assert validate_persisted_optimization(loaded) == []
    assert validate_persisted_optimization(loaded, config) == []
    assert validate_persisted_optimization(loaded.to_dict()) == []
    assert is_legacy_search_space(loaded) is False
    # Validation is read-only: canonical artifact bytes are untouched.
    assert artifact.read_bytes() == original_bytes


def test_fabricated_old_tsmom_study_is_legacy() -> None:
    old_study = {
        "base_strategy_id": "tsmom_vol_target",
        "request": {
            "base_strategy_id": "tsmom_vol_target",
            "search_space": [
                {"param_path": "allocator.params.gross_leverage"},
                {"param_path": "allocator.params.risk_aversion"},
                {"param_path": "universe.params.top_n"},
                {"param_path": "timing.0.params.lookback_fast"},
            ],
        },
        "best_params": {"allocator.params.gross_leverage": 1.0},
        "trials": [
            {
                "params": {
                    "timing.0.params.lookback_fast": 21,
                    "allocator.params.risk_aversion": 2.0,
                }
            }
        ],
    }
    snapshot = json.loads(json.dumps(old_study))
    reasons = validate_persisted_optimization(old_study)
    assert reasons
    assert is_legacy_search_space(old_study) is True
    text = "\n".join(reasons)
    assert "non-tunable structural/hygiene parameter path 'universe.params.top_n'" in text
    assert "allocator.params.gross_leverage" in text
    assert "allocator.params.risk_aversion" in text
    assert "timing.0.params.lookback_fast" not in text
    assert reasons == sorted(set(reasons))
    # No mutation of the input payload.
    assert old_study == snapshot


def test_old_study_typed_result_is_legacy(config: StrategyConfig, tmp_path: Path) -> None:
    clean, _artifact = _clean_persisted_study(config, tmp_path)
    legacy_space = (
        ParameterRange(
            param_path="allocator.params.gross_leverage", type="float", low=0.5, high=2.0
        ),
        ParameterRange(param_path="universe.params.top_n", type="int", low=3, high=20),
        ParameterRange(
            param_path="allocator.params.risk_aversion", type="float", low=0.1, high=5.0
        ),
    )
    legacy_request = clean.request.model_copy(update={"search_space": legacy_space})
    legacy_result = clean.model_copy(
        update={
            "request": legacy_request,
            "best_params": {"allocator.params.gross_leverage": 1.0},
        }
    )
    reasons = validate_persisted_optimization(legacy_result)
    text = "\n".join(reasons)
    assert is_legacy_search_space(legacy_result) is True
    assert "allocator.params.gross_leverage" in text
    assert "allocator.params.risk_aversion" in text
    assert "universe.params.top_n" in text


def test_old_drl_study_with_equal_weight_paths_is_legacy() -> None:
    old_study = {
        "base_strategy_id": "drl_ppo_llm_risk",
        "request": {
            "base_strategy_id": "drl_ppo_llm_risk",
            "search_space": [
                {"param_path": "allocator.equal_weight.gross_leverage"},
                {"param_path": "allocator.params.risk_aversion"},
                {"param_path": "timing.0.params.lookback_fast"},
            ],
        },
        "best_params": {},
    }
    reasons = validate_persisted_optimization(old_study)
    text = "\n".join(reasons)
    assert "allocator.equal_weight.gross_leverage" in text
    assert "allocator.params.risk_aversion" in text
    assert "timing.0.params.lookback_fast" not in text
    assert is_legacy_search_space(old_study) is True


def test_unresolvable_base_strategy_is_conservative() -> None:
    old_study = {
        "base_strategy_id": "custom_undeployed_strategy_xyz",
        "request": {
            "search_space": [
                {"param_path": "timing.0.params.lookback_fast"},
                {"param_path": "universe.params.top_n"},
                {"param_path": "foo.params.bar"},
            ]
        },
    }
    reasons = validate_persisted_optimization(old_study)
    text = "\n".join(reasons)
    assert "universe.params.top_n" in text and "structural" in text
    assert "foo.params.bar" in text and "unknown stage" in text
    assert "timing.0.params.lookback_fast" not in text
    assert "could not be resolved" in text


def test_fixture_payload_is_skipped() -> None:
    fixture = {
        "provenance": "fixture",
        "backend": "deterministic-fixture",
        "base_strategy_id": "tsmom_vol_target",
        "message": "optimizer unavailable",
        "best_params": {"allocator.params.gross_leverage": 1.0},
    }
    assert validate_persisted_optimization(fixture) == []
    assert is_legacy_search_space(fixture) is False
    # Non-engine backend without provenance is skipped as well.
    assert (
        validate_persisted_optimization(
            {"backend": "deterministic-fixture", "base_strategy_id": "tsmom_vol_target"}
        )
        == []
    )


def test_validation_input_handling(config: StrategyConfig) -> None:
    with pytest.raises(TypeError):
        validate_persisted_optimization(object())  # type: ignore[arg-type]
    # Non-canonical payload without a base strategy id has nothing to validate.
    assert validate_persisted_optimization({"best_params": {"whatever": 1.0}}) == []
    clean = {
        "base_strategy_id": config.id,
        "request": {"search_space": []},
        "best_params": {},
    }
    assert validate_persisted_optimization(clean) == []
    # A config with a mismatched id is ignored; the helper resolves the base id.
    other = StrategyConfig.model_validate(
        load_strategy_config(
            REPO_ROOT / "configs" / "strategies" / "drl_ppo_llm_risk.yaml"
        ).model_dump(mode="python")
    )
    reasons = validate_persisted_optimization(
        {
            "base_strategy_id": "tsmom_vol_target",
            "request": {"search_space": [{"param_path": "universe.params.top_n"}]},
        },
        other,
    )
    assert reasons and "universe.params.top_n" in reasons[0]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli_env() -> dict[str, str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO_ROOT)
    environment.pop("PYTHONWARNINGS", None)
    return environment


def test_cli_help_is_warning_free() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-W",
            "error::RuntimeWarning",
            "-m",
            "src.backtest.optimizer",
            "--help",
        ],
        cwd=REPO_ROOT,
        env=_cli_env(),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "RuntimeWarning" not in completed.stderr
    assert "--trials" in completed.stdout
    assert "--objective" in completed.stdout


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--trials", "1"],
        ["--trials", "201"],
        ["--trials", "abc"],
        ["--objective", "profit"],
        ["--splits", "1"],
        ["--purge-bars", "-1"],
        ["--embargo-bars", "-1"],
    ],
)
def test_cli_rejects_invalid_arguments(extra_args: list[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        optimizer_module.main(["--strategy", str(STRATEGY_PATH), "--offline", *extra_args])
    assert excinfo.value.code == 2


def test_cli_rejects_missing_strategy_and_bars(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as missing_strategy:
        optimizer_module.main(["--strategy", str(tmp_path / "missing.yaml"), "--offline"])
    assert missing_strategy.value.code == 2
    with pytest.raises(SystemExit) as missing_bars:
        optimizer_module.main(
            [
                "--strategy",
                str(STRATEGY_PATH),
                "--bars",
                str(tmp_path / "missing.parquet"),
                "--offline",
            ]
        )
    assert missing_bars.value.code == 2


def test_cli_end_to_end_with_fake_oos_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spy = _SpyRunner(_fake_report())
    monkeypatch.setattr(optimizer_module, "run_walk_forward", spy)
    output = tmp_path / "opt.json"
    studies = tmp_path / "studies"
    exit_code = optimizer_module.main(
        [
            "--strategy",
            str(STRATEGY_PATH),
            "--trials",
            "3",
            "--objective",
            "sharpe",
            "--offline",
            "--output",
            str(output),
            "--output-dir",
            str(studies),
            "--splits",
            "3",
            "--purge-bars",
            "2",
            "--embargo-bars",
            "1",
        ]
    )
    assert exit_code == 0
    assert output.is_file()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["n_trials"] == 3
    assert len(payload["trials"]) == 3
    assert payload["backend"] in {"deterministic", "optuna"}
    assert payload["best_strategy_config"]["id"] == "tsmom_vol_target"
    assert Path(payload["artifact_path"]).is_file()
    assert Path(payload["artifact_path"]).parent == studies
    assert len(spy.calls) == 3
    assert spy.calls[0][1]["splits"] == 3
    assert spy.calls[0][1]["purge_bars"] == 2
    assert spy.calls[0][1]["embargo_bars"] == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["study_id"] == payload["study_id"]
    assert summary["output"] == str(output)
