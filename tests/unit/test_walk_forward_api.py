"""Unit tests for the public walk-forward API (``run_walk_forward``/report).

Covers the Backtest Lab backend contract: exact fold count/geometry with no
leakage, the typed report's JSON shape vs the CLI report, convenience properties,
bars/states/offline input modes, injected pipeline/translator handling and input
validation.
"""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import pytest
import src.backtest.engine as engine_module
from configs.instruments import InstrumentsConfig, load_instruments
from configs.strategy import StrategyConfig, load_strategy_config
from src.backtest import BacktestMode, WalkForwardReport, run_walk_forward
from src.backtest.engine import (
    SimpleContractTranslator,
    WalkForwardFoldReport,
    build_states_from_bars,
)
from src.core.types import MarketState, PipelineResult, WeightVector

from tests.fixtures.market import make_bars_frame

REPO_ROOT = Path(__file__).resolve().parents[2]
STRATEGY_PATH = REPO_ROOT / "configs" / "strategies" / "tsmom_vol_target.yaml"

REPORT_KEYS = {
    "report",
    "schema_version",
    "run",
    "walk_forward",
    "folds",
    "aggregate",
    "combined_metrics",
    "concatenated_equity_curve",
    "tearsheet_path",
}
FOLD_KEYS = {
    "fold",
    "train_start",
    "train_end",
    "test_start",
    "test_end",
    "train_start_time",
    "train_end_time",
    "test_start_time",
    "test_end_time",
    "purge_gap",
    "train_size",
    "test_size",
    "n_bars",
    "return",
    "ending_equity",
    "n_trades",
    "n_rebalances",
    "metrics",
}


class _ConstantPipeline:
    """Deterministic test pipeline approved weights (no cross-fold state)."""

    def __init__(self, weights: dict[str, float]) -> None:
        self._weights = WeightVector(weights)

    @property
    def name(self) -> str:
        return "walk-forward-api-pipeline"

    def run(self, state: MarketState) -> PipelineResult:
        return PipelineResult(
            timestamp=state.timestamp,
            universe=state.ids,
            base_weights=self._weights,
            timing_weights=self._weights,
            risk_weights=self._weights,
            final_weights=self._weights,
        )


@pytest.fixture()
def instrument_master() -> InstrumentsConfig:
    return load_instruments()


@pytest.fixture()
def strategy_config() -> StrategyConfig:
    return load_strategy_config(STRATEGY_PATH)


@pytest.fixture()
def pipeline() -> _ConstantPipeline:
    return _ConstantPipeline({"ES": 0.30, "NQ": 0.20, "CL": -0.10})


@pytest.fixture()
def translator(instrument_master: InstrumentsConfig) -> SimpleContractTranslator:
    return SimpleContractTranslator(instrument_master)


def _fixture_bars(periods: int = 96):
    return make_bars_frame(("ES", "NQ", "CL"), periods=periods, seed=7)


def test_run_walk_forward_bars_report_contract(
    strategy_config: StrategyConfig,
    instrument_master: InstrumentsConfig,
    pipeline: _ConstantPipeline,
    translator: SimpleContractTranslator,
) -> None:
    report = run_walk_forward(
        strategy_config,
        bars=_fixture_bars(96),
        splits=4,
        purge_bars=3,
        embargo_bars=2,
        initial_equity=1_000_000.0,
        mode=BacktestMode.EVENT,
        instrument_master=instrument_master,
        pipeline=pipeline,
        translator=translator,
    )
    assert isinstance(report, WalkForwardReport)
    assert report.fold_count == 4
    assert len(report.folds) == 4
    assert report.walk_forward.requested_splits == 4
    assert report.walk_forward.fold_count == 4
    assert report.walk_forward.purge_bars == 3
    assert report.walk_forward.embargo_bars == 2
    assert report.walk_forward.expanding is True
    assert report.run.n_bars == 96
    assert report.run.strategy == strategy_config.id
    assert report.run.mode is BacktestMode.EVENT

    for fold in report.folds:
        assert isinstance(fold, WalkForwardFoldReport)
        assert 0 < fold.train_size <= fold.train_end - fold.train_start
        assert fold.test_size == fold.test_end - fold.test_start
        assert fold.n_bars <= fold.test_size
        assert fold.purge_gap == 3
        # No leakage: training ends strictly before the test window and the
        # configured purge gap is fully excluded.
        assert fold.train_end < fold.test_start
        assert fold.train_end + 3 == fold.test_start
        assert fold.n_trades >= 0
        assert fold.n_rebalances >= 0
        assert fold.ending_equity > 0.0
        assert "sharpe" in fold.metrics
    for previous, current in pairwise(report.folds):
        assert current.test_start == previous.test_end
        assert current.test_start > previous.test_start
        assert current.fold == previous.fold + 1
    assert report.folds[0].train_start == 0
    assert report.folds[-1].test_end == report.run.n_bars

    # Convenience properties are consistent with the typed payload.
    assert report.mean_sharpe == pytest.approx(report.aggregate["sharpe"]["mean"])
    assert report.total_trades == sum(fold.n_trades for fold in report.folds)
    assert report.total_trades > 0
    assert report.total_rebalances == sum(fold.n_rebalances for fold in report.folds)
    assert report.total_rebalances > 0
    assert report.final_equity == pytest.approx(report.concatenated_equity_curve.equity[-1])
    assert report.combined_result is not None
    assert report.combined_result.n_trades == report.total_trades
    assert report.combined_result.n_rebalances == report.total_rebalances
    assert report.combined_result.ending_equity == pytest.approx(report.final_equity)

    payload = report.to_dict()
    assert set(payload) == REPORT_KEYS
    assert payload["report"] == "walk_forward"
    assert payload["schema_version"] == engine_module.WALK_FORWARD_REPORT_VERSION
    assert payload["tearsheet_path"] is None
    run_payload = payload["run"]
    assert isinstance(run_payload, dict)
    assert set(run_payload) == {
        "run_id",
        "strategy",
        "mode",
        "initial_equity",
        "generated_at",
        "window_start",
        "window_end",
        "n_bars",
    }
    wf_payload = payload["walk_forward"]
    assert isinstance(wf_payload, dict)
    assert set(wf_payload) == {
        "requested_splits",
        "fold_count",
        "purge_bars",
        "embargo_bars",
        "train_days",
        "test_days",
        "step_days",
        "expanding",
    }
    folds_payload = payload["folds"]
    assert isinstance(folds_payload, list) and len(folds_payload) == 4
    for fold_payload in folds_payload:
        assert set(fold_payload) == FOLD_KEYS
        assert set(fold_payload["metrics"]) >= {"cagr", "annual_volatility", "sharpe"}
    aggregate_payload = payload["aggregate"]
    assert isinstance(aggregate_payload, dict)
    assert aggregate_payload["n_folds"] == 4
    assert set(aggregate_payload["sharpe"]) == {"mean", "std"}
    curve = payload["concatenated_equity_curve"]
    assert isinstance(curve, dict)
    assert len(curve["equity"]) == len(curve["timestamps"]) == len(curve["drawdown"])
    assert len(curve["equity"]) == sum(fold.test_size for fold in report.folds)
    json.dumps(payload)  # fully JSON-serialisable


def test_run_walk_forward_cli_json_parity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    strategy_config: StrategyConfig,
    instrument_master: InstrumentsConfig,
) -> None:
    """The helper payload and the CLI JSON report carry the same data."""

    def fake_pipeline(
        config: StrategyConfig, master: InstrumentsConfig, *, offline: bool = False
    ) -> _ConstantPipeline:
        del config, master, offline
        return _ConstantPipeline({"ES": 0.30, "NQ": 0.20, "CL": -0.10})

    def fake_translator(
        master: InstrumentsConfig, *, offline: bool = False
    ) -> SimpleContractTranslator:
        del offline
        return SimpleContractTranslator(master)

    monkeypatch.setattr(engine_module, "_resolve_cli_pipeline", fake_pipeline)
    monkeypatch.setattr(engine_module, "_resolve_cli_translator", fake_translator)
    monkeypatch.setattr(engine_module, "_resolve_cli_state_builder", lambda master: None)

    bars = _fixture_bars(96)
    bars_path = tmp_path / "bars.parquet"
    bars.write_parquet(bars_path)
    output = tmp_path / "wf.json"
    exit_code = engine_module.main(
        [
            "--strategy",
            str(STRATEGY_PATH),
            "--bars",
            str(bars_path),
            "--walk-forward",
            "--wf-splits",
            "4",
            "--purge-bars",
            "3",
            "--embargo-bars",
            "2",
            "--mode",
            "event",
            "--initial-equity",
            "1000000",
            "--output",
            str(output),
        ]
    )
    assert exit_code == 0
    cli_payload = json.loads(output.read_text(encoding="utf-8"))
    tearsheet_path = cli_payload["tearsheet_path"]
    assert tearsheet_path is not None and Path(tearsheet_path).exists()
    # The helper itself never writes files, so its payload path is None; the CLI
    # injects the written tear-sheet path. Everything else must be identical.
    cli_payload["tearsheet_path"] = None

    report = run_walk_forward(
        strategy_config,
        bars=bars,
        splits=4,
        purge_bars=3,
        embargo_bars=2,
        initial_equity=1_000_000.0,
        mode=BacktestMode.EVENT,
        instrument_master=instrument_master,
    )
    helper_payload = report.to_dict()
    assert helper_payload == cli_payload
    assert list(helper_payload) == list(cli_payload)
    for helper_fold, cli_fold in zip(helper_payload["folds"], cli_payload["folds"], strict=True):
        assert list(helper_fold) == list(cli_fold)
        assert helper_fold == cli_fold


def test_run_walk_forward_with_states_only(
    strategy_config: StrategyConfig,
    instrument_master: InstrumentsConfig,
    pipeline: _ConstantPipeline,
    translator: SimpleContractTranslator,
) -> None:
    bars = _fixture_bars(90)
    states = build_states_from_bars(bars, instrument_master, initial_equity=1_000_000.0)
    report = run_walk_forward(
        strategy_config,
        states=states,
        splits=3,
        purge_bars=2,
        embargo_bars=1,
        initial_equity=1_000_000.0,
        mode=BacktestMode.VECTORIZED,
        instrument_master=instrument_master,
        pipeline=pipeline,
        translator=translator,
    )
    assert report.fold_count == 3
    assert report.run.n_bars == len(states)
    assert all(fold.n_bars > 0 for fold in report.folds)
    assert report.final_equity > 0.0
    assert report.combined_result is not None
    payload = report.to_dict()
    json.dumps(payload)
    assert len(payload["concatenated_equity_curve"]["equity"]) > 0


def test_run_walk_forward_offline_defaults(
    monkeypatch: pytest.MonkeyPatch,
    strategy_config: StrategyConfig,
    instrument_master: InstrumentsConfig,
    pipeline: _ConstantPipeline,
    translator: SimpleContractTranslator,
) -> None:
    # Use the engine's built-in deterministic state builder to keep the test fast
    # while still exercising the offline bar source and default window resolution.
    monkeypatch.setattr(engine_module, "_resolve_cli_state_builder", lambda master: None)
    report = run_walk_forward(
        strategy_config,
        splits=2,
        purge_bars=1,
        embargo_bars=1,
        instrument_master=instrument_master,
        pipeline=pipeline,
        translator=translator,
    )
    assert report.fold_count == 2
    assert report.run.n_bars == 260  # deterministic offline source window
    assert report.final_equity > 0.0


def test_run_walk_forward_insufficient_bars_raises(
    strategy_config: StrategyConfig,
    instrument_master: InstrumentsConfig,
    pipeline: _ConstantPipeline,
    translator: SimpleContractTranslator,
) -> None:
    with pytest.raises(ValueError, match="insufficient bars"):
        run_walk_forward(
            strategy_config,
            bars=make_bars_frame(("ES",), periods=12, seed=1),
            splits=5,
            purge_bars=6,
            embargo_bars=1,
            instrument_master=instrument_master,
            pipeline=pipeline,
            translator=translator,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"splits": 1},
        {"splits": 0},
        {"purge_bars": -1},
        {"embargo_bars": -1},
        {"initial_equity": 0.0},
        {"initial_equity": -5.0},
    ],
)
def test_run_walk_forward_validation(
    kwargs: dict[str, object],
    strategy_config: StrategyConfig,
    instrument_master: InstrumentsConfig,
    pipeline: _ConstantPipeline,
    translator: SimpleContractTranslator,
) -> None:
    with pytest.raises(ValueError):
        run_walk_forward(
            strategy_config,
            bars=_fixture_bars(60),
            instrument_master=instrument_master,
            pipeline=pipeline,
            translator=translator,
            **kwargs,  # type: ignore[arg-type]
        )


def test_run_walk_forward_rejects_bad_mode(
    strategy_config: StrategyConfig,
    instrument_master: InstrumentsConfig,
    pipeline: _ConstantPipeline,
    translator: SimpleContractTranslator,
) -> None:
    with pytest.raises(TypeError):
        run_walk_forward(
            strategy_config,
            bars=_fixture_bars(60),
            mode="event",  # type: ignore[arg-type]
            instrument_master=instrument_master,
            pipeline=pipeline,
            translator=translator,
        )


def test_injected_pipeline_and_translator_are_honoured(
    monkeypatch: pytest.MonkeyPatch,
    strategy_config: StrategyConfig,
    instrument_master: InstrumentsConfig,
    pipeline: _ConstantPipeline,
    translator: SimpleContractTranslator,
) -> None:
    def _must_not_resolve(*args: object, **kwargs: object) -> None:
        raise AssertionError("injected pipeline/translator should bypass resolution")

    monkeypatch.setattr(engine_module, "_resolve_cli_pipeline", _must_not_resolve)
    monkeypatch.setattr(engine_module, "_resolve_cli_translator", _must_not_resolve)
    report = run_walk_forward(
        strategy_config,
        bars=_fixture_bars(60),
        splits=3,
        purge_bars=2,
        embargo_bars=1,
        mode=BacktestMode.EVENT,
        instrument_master=instrument_master,
        pipeline=pipeline,
        translator=translator,
    )
    assert report.fold_count == 3
    assert report.total_trades > 0
    assert report.combined_result is not None


def test_state_duplicates_rejected(
    strategy_config: StrategyConfig,
    instrument_master: InstrumentsConfig,
    pipeline: _ConstantPipeline,
    translator: SimpleContractTranslator,
) -> None:
    bars = _fixture_bars(60)
    states = build_states_from_bars(bars, instrument_master, initial_equity=1_000_000.0)
    with pytest.raises(ValueError, match="duplicate"):
        run_walk_forward(
            strategy_config,
            states=[*states, states[0]],
            splits=2,
            purge_bars=1,
            embargo_bars=1,
            instrument_master=instrument_master,
            pipeline=pipeline,
            translator=translator,
        )


def test_walk_forward_exports() -> None:
    import src.backtest as package

    assert package.run_walk_forward is engine_module.run_walk_forward
    assert package.WalkForwardReport is engine_module.WalkForwardReport
    assert "run_walk_forward" in package.__all__
    assert "WalkForwardReport" in package.__all__
    assert "run_walk_forward" in dir(package)
    assert "WalkForwardReport" in dir(package)
