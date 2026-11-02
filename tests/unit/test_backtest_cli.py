"""Unit tests for the production walk-forward CLI and the lazy backtest package.

Covers:

* ``--walk-forward`` report structure, fold count/geometry and no-leakage gaps;
* argparse validation for ``--wf-splits`` / ``--purge-bars`` / ``--embargo-bars``
  (invalid values must exit 2);
* config-sourced purge/embargo defaults and the zero-gap boundary case;
* the PEP 562 lazy ``src.backtest.__getattr__`` (importing the package must not
  import ``src.backtest.engine``; attribute access must);
* ``python -m src.backtest.engine --help`` under
  ``-W error::RuntimeWarning`` (zero ``<frozen runpy>`` warnings).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from itertools import pairwise
from pathlib import Path

import pytest
import src.backtest.engine as engine_module
from src.backtest.engine import SimpleContractTranslator
from src.core.types import MarketState, PipelineResult, WeightVector

from tests.fixtures.market import make_bars_frame

REPO_ROOT = Path(__file__).resolve().parents[2]
STRATEGY_PATH = REPO_ROOT / "configs" / "strategies" / "tsmom_vol_target.yaml"
TIMESTAMP_COLUMN = "timestamp_utc"


class _ConstantPipeline:
    """Deterministic test pipeline: the same approved weights every rebalance."""

    def __init__(self, weights: dict[str, float]) -> None:
        self._weights = WeightVector(weights)

    @property
    def name(self) -> str:
        return "cli-test-pipeline"

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
def cli_components(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the CLI's lazy layer resolution with deterministic fakes."""

    def fake_pipeline(
        config: object, instrument_master: object, *, offline: bool = False
    ) -> _ConstantPipeline:
        del config, instrument_master, offline
        return _ConstantPipeline({"ES": 0.30, "NQ": 0.20, "CL": -0.10})

    def fake_translator(
        instrument_master: object, *, offline: bool = False
    ) -> SimpleContractTranslator:
        del offline
        return SimpleContractTranslator(instrument_master)  # type: ignore[arg-type]

    monkeypatch.setattr(engine_module, "_resolve_cli_pipeline", fake_pipeline)
    monkeypatch.setattr(engine_module, "_resolve_cli_translator", fake_translator)
    monkeypatch.setattr(engine_module, "_resolve_cli_state_builder", lambda master: None)


def _write_bars(
    tmp_path: Path, *, periods: int, roots: tuple[str, ...] = ("ES", "NQ", "CL")
) -> Path:
    bars = make_bars_frame(roots, periods=periods, seed=7)
    path = tmp_path / f"bars_{periods}.parquet"
    bars.write_parquet(path)
    return path


def _run_walk_forward(
    tmp_path: Path,
    *,
    bars_path: Path,
    splits: int,
    purge: int,
    embargo: int,
    mode: str = "event",
    html: bool = False,
) -> tuple[int, dict[str, object], Path]:
    output = tmp_path / "walk_forward.json"
    html_path = tmp_path / "walk_forward.html"
    argv = [
        "--strategy",
        str(STRATEGY_PATH),
        "--bars",
        str(bars_path),
        "--walk-forward",
        "--wf-splits",
        str(splits),
        "--purge-bars",
        str(purge),
        "--embargo-bars",
        str(embargo),
        "--mode",
        mode,
        "--output",
        str(output),
    ]
    if html:
        argv.extend(["--html", str(html_path)])
    exit_code = engine_module.main(argv)
    report = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {}
    return exit_code, report, html_path


def test_walk_forward_report_structure_and_no_leakage(tmp_path: Path, cli_components: None) -> None:
    bars_path = _write_bars(tmp_path, periods=120)
    exit_code, report, html_path = _run_walk_forward(
        tmp_path, bars_path=bars_path, splits=5, purge=3, embargo=2, html=True
    )
    assert exit_code == 0
    assert report["report"] == "walk_forward"
    assert report["schema_version"] == engine_module.WALK_FORWARD_REPORT_VERSION
    assert set(report) >= {
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
    run = report["run"]
    assert isinstance(run, dict)
    assert run["strategy"] == "tsmom_vol_target"
    assert run["mode"] == "event"
    assert run["n_bars"] == 120
    config = report["walk_forward"]
    assert isinstance(config, dict)
    assert config["requested_splits"] == 5
    assert config["fold_count"] == 5
    assert config["purge_bars"] == 3
    assert config["embargo_bars"] == 2

    folds = report["folds"]
    assert isinstance(folds, list)
    assert len(folds) == 5
    # Walk-forward geometry: strictly increasing, non-overlapping, purge gap exact.
    for fold in folds:
        assert isinstance(fold, dict)
        assert {"fold", "train_start", "train_end", "test_start", "test_end"} <= set(fold)
        assert {"n_bars", "return", "ending_equity", "n_trades"} <= set(fold)
        assert fold["train_end"] < fold["test_start"]
        assert fold["train_end"] + config["purge_bars"] == fold["test_start"]
        assert fold["test_size"] == fold["test_end"] - fold["test_start"]
        assert fold["n_bars"] == fold["test_size"] == 19
        assert 0 < fold["train_size"] <= fold["train_end"] - fold["train_start"]
        assert fold["purge_gap"] == config["purge_bars"]
        metrics = fold["metrics"]
        assert isinstance(metrics, dict)
        assert {"cagr", "annual_volatility", "sharpe", "sortino", "max_drawdown"} <= set(metrics)
        assert all(isinstance(value, (int, float)) for value in metrics.values())
    for previous, current in pairwise(folds):
        assert current["test_start"] == previous["test_end"]
        assert current["test_start"] > previous["test_start"]
        assert current["fold"] == previous["fold"] + 1
    assert folds[0]["train_start"] == 0  # expanding window
    assert folds[0]["train_size"] == folds[0]["train_end"]  # no prior embargo
    assert folds[-1]["test_end"] == run["n_bars"]
    assert sum(fold["n_trades"] for fold in folds) > 0

    aggregate = report["aggregate"]
    assert isinstance(aggregate, dict)
    assert aggregate["n_folds"] == 5
    for key in (
        "cagr",
        "annual_volatility",
        "sharpe",
        "sortino",
        "max_drawdown",
        "return",
        "ending_equity",
        "n_trades",
    ):
        stats = aggregate[key]
        assert isinstance(stats, dict)
        assert {"mean", "std"} <= set(stats)
        assert isinstance(stats["mean"], (int, float))
        assert stats["std"] >= 0.0

    combined_metrics = report["combined_metrics"]
    assert isinstance(combined_metrics, dict)
    assert combined_metrics["initial_equity"] > 0.0
    assert "sharpe" in combined_metrics
    curve = report["concatenated_equity_curve"]
    assert isinstance(curve, dict)
    assert len(curve["equity"]) == 5 * 19
    assert len(curve["timestamps"]) == len(curve["equity"])
    # A normal tearsheet is written alongside the report.
    tearsheet_path = Path(str(report["tearsheet_path"]))
    assert tearsheet_path.exists()
    tearsheet = json.loads(tearsheet_path.read_text(encoding="utf-8"))
    assert tearsheet["accounting"]["residual"] == pytest.approx(0.0, abs=1e-6)
    assert html_path.exists()
    assert "<!DOCTYPE html>" in html_path.read_text(encoding="utf-8")


def test_walk_forward_vectorized_mode(tmp_path: Path, cli_components: None) -> None:
    bars_path = _write_bars(tmp_path, periods=90)
    exit_code, report, _html = _run_walk_forward(
        tmp_path, bars_path=bars_path, splits=3, purge=2, embargo=1, mode="vectorized"
    )
    assert exit_code == 0
    folds = report["folds"]
    assert isinstance(folds, list) and len(folds) == 3
    assert report["run"]["mode"] == "vectorized"
    # vectorised runs still produce a JSON-serialisable combined curve/tearsheet
    assert report["tearsheet_path"] is not None


def test_walk_forward_uses_config_purge_and_embargo_defaults(
    tmp_path: Path, cli_components: None
) -> None:
    bars_path = _write_bars(tmp_path, periods=120)
    output = tmp_path / "defaults.json"
    exit_code = engine_module.main(
        [
            "--strategy",
            str(STRATEGY_PATH),
            "--bars",
            str(bars_path),
            "--walk-forward",
            "--wf-splits",
            "4",
            "--output",
            str(output),
        ]
    )
    assert exit_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    config = report["walk_forward"]
    # configs/strategies/tsmom_vol_target.yaml declares purge=5 and embargo=5.
    assert config["purge_bars"] == 5
    assert config["embargo_bars"] == 5
    assert len(report["folds"]) == 4


def test_walk_forward_zero_gap_boundary(tmp_path: Path, cli_components: None) -> None:
    bars_path = _write_bars(tmp_path, periods=120)
    exit_code, report, _html = _run_walk_forward(
        tmp_path, bars_path=bars_path, splits=5, purge=0, embargo=0
    )
    assert exit_code == 0
    folds = report["folds"]
    for fold in folds:
        assert fold["purge_gap"] == 0
        assert fold["train_end"] == fold["test_start"]
        assert fold["train_end"] <= fold["test_start"]
    assert len(folds) == 5


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--wf-splits", "1"],
        ["--wf-splits", "0"],
        ["--wf-splits", "-3"],
        ["--wf-splits", "abc"],
        ["--wf-splits", "2.5"],
        ["--purge-bars", "-1"],
        ["--purge-bars", "not-a-number"],
        ["--embargo-bars", "-2"],
        ["--embargo-bars", ""],
    ],
)
def test_walk_forward_flags_reject_invalid_values(extra_args: list[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        engine_module.main(["--strategy", str(STRATEGY_PATH), "--offline", *extra_args])
    assert excinfo.value.code == 2


def test_walk_forward_insufficient_window_returns_error(
    tmp_path: Path, cli_components: None, capsys: pytest.CaptureFixture[str]
) -> None:
    bars_path = _write_bars(tmp_path, periods=12, roots=("ES",))
    exit_code, _report, _html = _run_walk_forward(
        tmp_path, bars_path=bars_path, splits=5, purge=6, embargo=1
    )
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "error:" in captured.err


def test_walk_forward_helpers_validate_geometry(tmp_path: Path) -> None:
    splitter, folds = engine_module._build_walk_forward_folds(
        n_samples=120, splits=5, purge_bars=3, embargo_bars=2
    )
    assert len(folds) == 5
    assert splitter.train_days == 22
    assert splitter.test_days == 19
    assert splitter.purge_days == 3
    assert splitter.embargo_days == 2
    with pytest.raises(ValueError):
        engine_module._build_walk_forward_folds(
            n_samples=120, splits=1, purge_bars=3, embargo_bars=2
        )
    with pytest.raises(ValueError):
        engine_module._build_walk_forward_folds(
            n_samples=10, splits=5, purge_bars=6, embargo_bars=2
        )
    with pytest.raises(ValueError):
        engine_module._build_walk_forward_folds(n_samples=1, splits=2, purge_bars=0, embargo_bars=0)
    timestamps = engine_module._runnable_timestamps(make_bars_frame(("ES",), periods=5, seed=1))
    assert len(timestamps) == 5
    empty = make_bars_frame(("ES",), periods=2, seed=1).head(0)
    with pytest.raises(ValueError):
        engine_module._runnable_timestamps(empty)


def test_walk_forward_default_report_path() -> None:
    from configs.strategy import load_strategy_config

    config = load_strategy_config(STRATEGY_PATH)
    path = engine_module._default_walk_forward_path(config)
    assert path.name == f"{config.id}_walk_forward.json"
    assert path.parent.name == "artifacts"


def test_backtest_package_attribute_access() -> None:
    import src.backtest as package

    assert package.BacktestEngine is engine_module.BacktestEngine
    assert package.EventBacktester is engine_module.EventBacktester
    assert package.BacktestResult is engine_module.BacktestResult
    assert package.BacktestMode is engine_module.BacktestMode
    assert package.main is engine_module.main
    assert package.run_from_config is engine_module.run_from_config
    assert package.RebalanceSchedule is engine_module.RebalanceSchedule
    assert package.EQUITY_CURVE_SCHEMA is engine_module.EQUITY_CURVE_SCHEMA
    assert package.StateBuilder is engine_module.StateBuilder
    assert package.build_states_from_bars is engine_module.build_states_from_bars
    assert package.default_state_builder is engine_module.default_state_builder
    expected = {
        "BacktestEngine",
        "EventBacktester",
        "BacktestResult",
        "BacktestMode",
        "main",
        "run_from_config",
    }
    assert expected <= set(package.__all__)
    assert expected <= set(dir(package))
    with pytest.raises(AttributeError):
        _ = package.this_symbol_does_not_exist


def _subprocess_environment() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    env.pop("PYTHONWARNINGS", None)
    return env


def test_module_entrypoint_help_has_no_runpy_runtime_warning() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-W",
            "error::RuntimeWarning",
            "-m",
            "src.backtest.engine",
            "--help",
        ],
        cwd=REPO_ROOT,
        env=_subprocess_environment(),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "RuntimeWarning" not in completed.stderr
    assert "runpy" not in completed.stderr
    assert "--walk-forward" in completed.stdout
    assert "--wf-splits" in completed.stdout
    assert "--purge-bars" in completed.stdout
    assert "--embargo-bars" in completed.stdout


def test_backtest_package_imports_engine_lazily() -> None:
    script = (
        "import sys\n"
        "import src.backtest\n"
        "assert 'src.backtest.engine' not in sys.modules, 'engine imported eagerly'\n"
        "from src.backtest import BacktestEngine, main, run_from_config, RebalanceSchedule\n"
        "assert 'src.backtest.engine' in sys.modules, 'lazy getattr did not import engine'\n"
        "print(BacktestEngine.__name__, callable(main), callable(run_from_config), "
        "RebalanceSchedule.daily().every_n_bars)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=_subprocess_environment(),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "BacktestEngine True True 1" in completed.stdout
    assert "RuntimeWarning" not in completed.stderr
