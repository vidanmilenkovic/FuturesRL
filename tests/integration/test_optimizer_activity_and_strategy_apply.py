"""Independent v1.5.3 verification: optimizer no-flat-winner + apply visibility.

Verification-only suite for branch ``fix/v1.5.3-optuna-apply`` (tasks 47/48/49);
no production edits. It pins the v1.5.3 contracts:

1. **Optimizer zero-trade regression** (real ``WalkForwardOptimizer`` on the
   committed TSMOM config, offline, deterministic backend, 4 trials / 5 folds):
   every COMPLETE trial records ``n_trades``/``avg_gross_exposure``; flat or
   under-exposed trials carry ``zero_activity=True`` and the hard objective
   ``-10.0``; the best trial is always an active trial with
   ``n_trades >= min_trades`` (a wholly flat study never gets crowned); the
   offline window is >=750 bars so lookbacks/folds have warm-up; active trials
   have finite OOS metrics.
2. **Catalog**: ``timing.carry_filter.min_carry_bps`` = default -50 / min -200 /
   max +50 / step 10 bps and is NOT pre-selected; exactly 5 curated
   ``optuna_default_selected`` keys; the default helper never returns carry for a
   TSMOM config while it stays active/tunable; the API catalog JSON preserves the
   5 booleans.
3. **Apply visibility**: POST /api/optimize/{study_id}/apply on a clean study
   returns ``saved_strategy_id``/``strategy_path``/``backtest_run_id``, persists
   the YAML (paper-like hermetic configs dir), and the saved strategy + the
   verification backtest run are visible immediately through GET /api/strategies
   and GET /api/backtests (no restart); repeat apply is a documented 409.

All data is generated offline (deterministic bar source); no TWS, no network,
no real ``configs/`` writes (``configs_dir`` points at ``tmp_path``).
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

import pytest
from configs.instruments import InstrumentsConfig, load_instruments
from configs.settings import Settings
from configs.strategy import StrategyConfig, load_strategy_config
from fastapi.testclient import TestClient
from src.api.main import create_app
from src.api.optimize_service import OptimizationStudy
from src.backtest.optimizer import (
    HARD_INVALID_OBJECTIVE,
    MIN_OFFLINE_BARS,
    OptimizationRequest,
    ParameterRange,
    TrialState,
    WalkForwardOptimizer,
    default_search_space,
)
from src.strategies.parameter_catalog import (
    active_parameter_keys,
    default_selected_parameter_keys,
    parameter_catalog,
    parameters_for_strategy,
)

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"
STRATEGIES_DIR = CONFIGS_DIR / "strategies"
INSTRUMENTS_PATH = CONFIGS_DIR / "instruments.yaml"
TSMOM_PATH = STRATEGIES_DIR / "tsmom_vol_target.yaml"

FIXED_CLOCK = datetime(2026, 11, 3, 12, 0, tzinfo=UTC)

#: v1.5.3 frozen curated Optuna pre-selection (exactly five keys, carry removed).
EXPECTED_DEFAULT_SELECTED: frozenset[str] = frozenset(
    {
        "timing.tsmom.lookback_fast",
        "timing.tsmom.lookback_slow",
        "timing.kama.window",
        "allocator.volatility_target.target_annual_volatility",
        "risk.quant_risk.trailing_stop_pct",
    }
)
CARRY_KEY = "timing.carry_filter.min_carry_bps"
MIN_TRADES = 3
MIN_AVG_GROSS_EXPOSURE = 0.01


# ---------------------------------------------------------------------------
# Fixtures / shared helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def master() -> InstrumentsConfig:
    return load_instruments(INSTRUMENTS_PATH)


@pytest.fixture(scope="module")
def tsmom_config() -> StrategyConfig:
    return load_strategy_config(TSMOM_PATH)


def _settings(
    tmp_path: Path,
    *,
    environment: str = "backtest",
    configs_dir: Path | None = None,
    telemetry_interval: float = 0.02,
) -> Settings:
    """Hermetic settings (explicitly disables IBKR/TWS wiring)."""
    return Settings(
        environment=environment,
        data_dir=tmp_path / "data",
        parquet_dir=tmp_path / "data" / "parquet",
        duckdb_path=tmp_path / "data" / "v153.duckdb",
        report_dir=tmp_path / "artifacts",
        configs_dir=configs_dir or CONFIGS_DIR,
        instruments_path=INSTRUMENTS_PATH,
        telemetry_interval_seconds=telemetry_interval,
        api_enable_ibkr=False,
        ibkr_paper=True,
    )


def _curated_space(config: StrategyConfig) -> tuple[ParameterRange, ...]:
    space = default_search_space(config)
    assert len(space) == 5, (
        f"expected the five curated defaults, got {[p.param_path for p in space]}"
    )
    assert not any(p.param_path.endswith("min_carry_bps") for p in space)
    return space


def _request(
    config: StrategyConfig,
    *,
    n_trials: int = 4,
    wf_splits: int = 5,
    min_trades: int = MIN_TRADES,
    min_avg_gross_exposure: float = MIN_AVG_GROSS_EXPOSURE,
    search_space: tuple[ParameterRange, ...] | None = None,
) -> OptimizationRequest:
    return OptimizationRequest(
        base_strategy_id=config.id,
        objective_metric="sharpe",
        n_trials=n_trials,
        wf_splits=wf_splits,
        purge_bars=2,
        embargo_bars=2,
        min_trades=min_trades,
        min_avg_gross_exposure=min_avg_gross_exposure,
        search_space=search_space or default_search_space(config),
    )


def _offline_optimizer(
    tmp_path: Path,
    config: StrategyConfig,
    *,
    request: OptimizationRequest,
    seed: int = 7,
) -> WalkForwardOptimizer:
    return WalkForwardOptimizer(
        config,
        request=request,
        use_optuna=False,  # deterministic LHS backend: reproducible, no network
        output_dir=tmp_path / "optimizations",
        persist=False,
        offline=True,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# 1. Real optimizer: activity recording, flat-trial penalty, no flat winner
# ---------------------------------------------------------------------------


def test_real_optimizer_records_activity_and_window(
    tmp_path: Path, tsmom_config: StrategyConfig
) -> None:
    request = _request(tsmom_config, n_trials=4, wf_splits=5)
    optimizer = _offline_optimizer(tmp_path, tsmom_config, request=request)
    result = optimizer.optimize()

    # The study really ran the walk-forward engine (not a fixture).
    assert result.backend.value == "deterministic"
    assert result.walk_forward is True
    assert result.request.wf_splits == 5
    assert result.completed_trials == 4, result.trials

    # >=750 offline bars per root so lookbacks (up to 252) and folds have warm-up.
    bars = optimizer._window_bars
    states = optimizer._window_states
    assert bars is not None and bars.height > 0
    roots = bars["root_symbol"].n_unique()
    periods = bars.height // roots
    assert periods >= MIN_OFFLINE_BARS == 750, f"only {periods} offline bars (roots={roots})"
    assert states is not None and len(states) >= 750

    completed = [trial for trial in result.trials if trial.state is TrialState.COMPLETE]
    assert len(completed) == 4
    for trial in completed:
        assert isinstance(trial.n_trades, int) and trial.n_trades >= 0
        assert isinstance(trial.avg_gross_exposure, float) and trial.avg_gross_exposure >= 0.0
        assert isinstance(trial.zero_activity, bool)
        assert math.isfinite(trial.objective_value)
        is_flat = (
            trial.n_trades < request.min_trades
            or trial.avg_gross_exposure < request.min_avg_gross_exposure
        )
        if is_flat:
            assert trial.zero_activity is True, f"flat trial not flagged: {trial}"
            assert trial.objective_value <= HARD_INVALID_OBJECTIVE, (
                f"flat trial objective {trial.objective_value} not penalized"
            )
        else:
            assert trial.zero_activity is False, f"active trial wrongly flagged flat: {trial}"
            assert trial.n_trades >= request.min_trades
            for metric in (
                trial.oos_sharpe,
                trial.oos_sortino,
                trial.oos_calmar,
                trial.oos_cagr,
                trial.oos_rachev,
                trial.oos_return_on_margin,
                trial.max_dd,
                trial.raw_objective,
            ):
                assert math.isfinite(metric), f"non-finite OOS metric {metric} in {trial}"

    active = [trial for trial in completed if not trial.zero_activity]
    assert active, "the curated TSMOM study should contain active trials"
    best = next(trial for trial in completed if trial.trial_number == result.best_trial_number)
    assert best.zero_activity is False, "a flat trial must never be crowned best"
    assert best.n_trades >= request.min_trades
    assert best.objective_value == max(trial.objective_value for trial in active)
    assert result.best_value == best.objective_value
    assert result.best_params, "an active best trial must carry its parameters"
    assert result.best_strategy_config


def test_real_optimizer_never_crowns_flat_trials(
    tmp_path: Path, tsmom_config: StrategyConfig
) -> None:
    """A study whose completed trials are all flat persists with no winner."""
    # Force the flat classification deterministically via an impossible trade
    # floor (all COMPLETE trials become zero_activity, never FAIL).
    request = _request(tsmom_config, n_trials=2, wf_splits=3, min_trades=10_000)
    optimizer = _offline_optimizer(tmp_path, tsmom_config, request=request)
    result = optimizer.optimize()

    assert result.completed_trials == 2
    assert result.best_trial_number is None, "flat trials must not produce a best trial"
    assert result.best_params == {}
    assert result.best_value is None
    assert result.best_raw_value is None
    assert result.best_strategy_config in ({}, None)
    assert result.fold_count == 0

    for trial in result.trials:
        assert trial.state is TrialState.COMPLETE, "flat trials stay COMPLETE, not FAIL"
        assert trial.zero_activity is True
        assert trial.objective_value <= HARD_INVALID_OBJECTIVE
        assert math.isfinite(trial.raw_objective), "raw OOS metrics are still recorded"


# ---------------------------------------------------------------------------
# 2. Catalog: carry bounds/default + exactly five pre-selected keys
# ---------------------------------------------------------------------------


def test_catalog_carry_bounds_and_preselection() -> None:
    catalog = parameter_catalog()
    carry = catalog.get(CARRY_KEY)
    assert carry.default == pytest.approx(-50.0)
    assert carry.min == pytest.approx(-200.0)
    assert carry.max == pytest.approx(50.0)
    assert carry.step == pytest.approx(10.0)
    assert carry.unit == "bps"
    assert carry.optuna_tunable is True, "carry stays tunable"
    assert carry.optuna_default_selected is False, "carry is no longer pre-selected"

    selected = tuple(spec.key for spec in catalog.parameters if spec.optuna_default_selected)
    assert set(selected) == EXPECTED_DEFAULT_SELECTED
    assert len(selected) == 5, f"expected exactly 5 curated defaults, got {sorted(selected)!r}"


def test_catalog_helpers_omit_carry_for_tsmom(tsmom_config: StrategyConfig) -> None:
    defaults = set(default_selected_parameter_keys(tsmom_config))
    assert defaults == EXPECTED_DEFAULT_SELECTED
    assert CARRY_KEY not in defaults
    active = set(active_parameter_keys(tsmom_config))
    assert CARRY_KEY in active, "carry remains an active tunable for a carry-active strategy"
    specs = {spec.key: spec for spec in parameters_for_strategy(tsmom_config)}
    assert CARRY_KEY in specs
    assert specs[CARRY_KEY].optuna_default_selected is False


def test_api_catalog_exposes_five_defaults(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path), clock=lambda: FIXED_CLOCK)
    with TestClient(app) as client:
        response = client.get("/api/strategies/schema/catalog")
    assert response.status_code == 200, response.text
    raw = json.loads(response.text)
    parameters = raw["parameters"]
    assert len(parameters) > 100
    selected = {
        entry["key"] for entry in parameters if entry.get("optuna_default_selected") is True
    }
    assert selected == EXPECTED_DEFAULT_SELECTED
    assert len(selected) == 5
    assert all(isinstance(entry["optuna_default_selected"], bool) for entry in parameters)
    by_key = {entry["key"]: entry for entry in parameters}
    carry = by_key[CARRY_KEY]
    assert carry["optuna_tunable"] is True
    assert carry["optuna_default_selected"] is False
    assert carry["default"] == pytest.approx(-50.0)
    assert carry["min"] == pytest.approx(-200.0)
    assert carry["max"] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# 3. Apply visibility: saved strategy + verification run are visible immediately
# ---------------------------------------------------------------------------


def _paper_settings(tmp_path: Path) -> Settings:
    """Paper-like hermetic settings: dev/paper built-in writes go to a tmp tree."""
    configs_dir = tmp_path / "configs"
    strategies = configs_dir / "strategies"
    strategies.mkdir(parents=True, exist_ok=True)
    # The base strategy must exist in the hermetic built-in tree; never write the
    # committed configs/ in this verification.
    (strategies / "tsmom_vol_target.yaml").write_text(
        TSMOM_PATH.read_text(encoding="utf-8"), encoding="utf-8"
    )
    return _settings(tmp_path, environment="paper", configs_dir=configs_dir)


def _clean_study(
    tmp_path: Path,
    config: StrategyConfig,
    *,
    best_params: dict[str, object] | None = None,
) -> dict[str, object]:
    """A v1.5.3-clean canonical study payload (5-param curated space)."""
    space = [
        {
            "param_path": entry.param_path,
            "type": entry.type,
            "low": entry.low,
            "high": entry.high,
            "step": entry.step,
        }
        for entry in default_search_space(config)
    ]
    if best_params is None:
        # Paths that already exist in the committed TSMOM YAML (control case).
        best_params = {
            "timing.0.params.lookback_fast": 21,
            "timing.0.params.lookback_slow": 126,
            "timing.1.params.window": 10,
            "allocator.params.target_annual_volatility": 0.12,
        }
    study = OptimizationStudy.model_validate(
        {
            "study_id": "clean-v153-study",
            "base_strategy_id": config.id,
            "objective_metric": "sharpe",
            "n_trials": 5,
            "completed_trials": 5,
            "created_at": FIXED_CLOCK.isoformat(),
            "best_params": best_params,
            "best_value": 0.5,
        }
    )
    payload = study.model_dump(mode="json")
    payload.pop("provenance", None)  # engine-canonical artifacts leave it implicit
    payload["seed"] = 7
    payload["request"] = {"base_strategy_id": config.id, "search_space": space}
    directory = tmp_path / "artifacts" / "optimizations"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "clean-v153-study.json").write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_apply_persists_strategy_and_verification_run_immediately(
    tmp_path: Path, tsmom_config: StrategyConfig
) -> None:
    settings = _paper_settings(tmp_path)
    _clean_study(tmp_path, tsmom_config)
    saved_id = "tsmom_v153_opt"
    app = create_app(settings, clock=lambda: FIXED_CLOCK)
    with TestClient(app) as client:
        # The clean study is listed before apply.
        listed = client.get("/api/optimize")
        assert listed.status_code == 200, listed.text
        assert any(row["study_id"] == "clean-v153-study" for row in listed.json())
        strategies_before = {row["id"] for row in client.get("/api/strategies").json()}
        assert saved_id not in strategies_before

        response = client.post(
            "/api/optimize/clean-v153-study/apply",
            json={"new_strategy_id": saved_id, "verification_backtest": True},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["study_id"] == "clean-v153-study"
        assert body["saved_strategy_id"] == saved_id
        assert body["strategy_path"], "apply must report the persisted YAML path"
        yaml_path = Path(body["strategy_path"])
        assert yaml_path.is_file(), f"strategy YAML not persisted: {yaml_path}"
        assert yaml_path == tmp_path / "configs" / "strategies" / f"{saved_id}.yaml"
        assert body["backtest_run_id"], "verification backtest run id missing"
        verification = body["verification_backtest"]
        assert isinstance(verification, dict)
        assert verification["status"] == "completed", verification
        assert verification["run"]["kind"] == "walk_forward"
        assert verification["run"]["strategy_id"] == saved_id
        assert verification["run"]["id"] == body["backtest_run_id"]

        # Immediately visible (no restart) in the strategy catalog...
        strategies = client.get("/api/strategies")
        assert strategies.status_code == 200, strategies.text
        ids = {row["id"] for row in strategies.json()}
        assert saved_id in ids
        # ...and in the Backtest Lab run history + available strategies.
        backtests = client.get("/api/backtests")
        assert backtests.status_code == 200, backtests.text
        payload = backtests.json()
        runs = {row["id"]: row for row in payload["runs"]}
        assert body["backtest_run_id"] in runs
        verification_run = runs[body["backtest_run_id"]]
        assert verification_run["strategy_id"] == saved_id
        assert verification_run["kind"] == "walk_forward"
        assert saved_id in payload["available_strategies"]
        # The persisted run is retrievable by id immediately too.
        detail = client.get(f"/api/backtests/{body['backtest_run_id']}")
        assert detail.status_code == 200, detail.text
        assert detail.json()["strategy_id"] == saved_id

        # Repeat apply with the same target id is a clear, documented 409.
        repeat = client.post(
            "/api/optimize/clean-v153-study/apply",
            json={"new_strategy_id": saved_id},
        )
        assert repeat.status_code == 409, repeat.text
        assert "already exists" in repeat.json()["detail"]

        # A different target id still applies and becomes immediately visible.
        second = client.post(
            "/api/optimize/clean-v153-study/apply",
            json={"new_strategy_id": "tsmom_v153_opt_2"},
        )
        assert second.status_code == 200, second.text
        second_body = second.json()
        assert second_body["saved_strategy_id"] == "tsmom_v153_opt_2"
        assert Path(second_body["strategy_path"]).is_file()
        assert "tsmom_v153_opt_2" in {row["id"] for row in client.get("/api/strategies").json()}


def test_apply_default_request_runs_verification_backtest(
    tmp_path: Path, tsmom_config: StrategyConfig
) -> None:
    """``verification_backtest`` defaults on: an empty body still persists a run."""
    settings = _paper_settings(tmp_path)
    _clean_study(tmp_path, tsmom_config)
    app = create_app(settings, clock=lambda: FIXED_CLOCK)
    with TestClient(app) as client:
        response = client.post(
            "/api/optimize/clean-v153-study/apply",
            json={"new_strategy_id": "tsmom_default_apply"},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["saved_strategy_id"] == "tsmom_default_apply"
        assert body["backtest_run_id"], "default apply must trigger the verification run"
        assert body["verification_backtest"]["status"] == "completed"
        run_ids = {row["id"] for row in client.get("/api/backtests").json()["runs"]}
        assert body["backtest_run_id"] in run_ids


def test_apply_accepts_optimizer_injectable_parameters(
    tmp_path: Path, tsmom_config: StrategyConfig, master: InstrumentsConfig
) -> None:
    """A REAL optimizer study must apply even when it tunes a catalog-injectable
    parameter that the committed YAML omits (regression: 422 ``does not exist``)."""
    settings = _paper_settings(tmp_path)
    request = _request(tsmom_config, n_trials=2, wf_splits=3)
    optimizer = WalkForwardOptimizer(
        tsmom_config,
        request=request,
        instrument_master=master,
        use_optuna=False,
        output_dir=settings.optimization_artifacts_dir,
        persist=True,
        offline=True,
        seed=7,
    )
    result = optimizer.optimize()
    assert result.best_trial_number is not None
    injectable = {key for key in result.best_params if key.endswith("trailing_stop_pct")}
    assert injectable == {"risk.0.params.trailing_stop_pct"}, result.best_params
    assert "trailing_stop_pct" not in (tsmom_config.model_dump(mode="json")["risk"][0]["params"]), (
        "regression precondition: base YAML must omit trailing_stop_pct"
    )

    app = create_app(settings, clock=lambda: FIXED_CLOCK)
    with TestClient(app) as client:
        listed = {row["study_id"] for row in client.get("/api/optimize").json()}
        assert result.study_id in listed
        response = client.post(
            f"/api/optimize/{result.study_id}/apply",
            json={"new_strategy_id": "tsmom_injectable_opt"},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["saved_strategy_id"] == "tsmom_injectable_opt"
        yaml_path = Path(body["strategy_path"])
        assert yaml_path.is_file()
        saved_yaml = yaml_path.read_text(encoding="utf-8")
        assert "trailing_stop_pct" in saved_yaml, "injectable tuned param was dropped"
        assert body["backtest_run_id"]
        assert "tsmom_injectable_opt" in {row["id"] for row in client.get("/api/strategies").json()}
