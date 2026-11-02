"""Lead-owned full-stack integration test.

Wires the *real* modules together offline:
offline data source -> feature processor -> point-in-time MarketState builder ->
FinRL-X pipeline (with quant + LLM risk overlays) -> ContractTranslator ->
futures-native event backtester -> analytics metrics -> JSON/HTML tear sheet.

Only the external LLM transport and the (explicitly synthetic) offline bar source
are stand-ins; every other component is the production implementation.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from configs.instruments import InstrumentsConfig, load_instruments
from configs.settings import Settings
from configs.strategy import StageConfig, load_strategy_config
from src.analytics import build_tearsheet, compute_metrics_from_result, write_tearsheet
from src.backtest.engine import BacktestResult, EventBacktester
from src.data import FeatureProcessor, MarketStateBuilder, OfflineBarSource
from src.risk import ContractTranslator, LLMRiskOverlay, MacroEventCalendar, QuantRiskOverlay
from src.strategies import build_pipeline

from tests.fixtures.market import make_risk_assessment

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOTS = ("ES", "NQ", "CL", "GC", "ZN")
START = datetime(2024, 1, 2, 21, 0, tzinfo=UTC)
PERIODS = 340
INITIAL_EQUITY = 1_000_000.0


class _StaticLLMClient:
    """Deterministic local stand-in for the async LLM transport (offline tests)."""

    async def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        del prompt, system_prompt, temperature
        return json.dumps(
            {
                "assessments": [
                    {
                        "instrument_id": root,
                        "sentiment_score": 3.0,
                        "semantic_risk_score": 1.8,
                        "confidence": 0.9,
                        "reasoning_summary": "offline integration fixture",
                    }
                    for root in ROOTS
                ]
            }
        )


@pytest.fixture(scope="module")
def instrument_master() -> InstrumentsConfig:
    return load_instruments(REPO_ROOT / "configs" / "instruments.yaml")


@pytest.fixture(scope="module")
def settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    base = tmp_path_factory.mktemp("full-stack")
    return Settings(
        _env_file=None,
        environment="backtest",
        data_dir=base / "data",
        parquet_dir=base / "data" / "parquet",
        duckdb_path=base / "data" / "integration.duckdb",
        report_dir=base / "artifacts",
        instruments_path=REPO_ROOT / "configs" / "instruments.yaml",
    )


@pytest.fixture(scope="module")
def market_states(instrument_master: InstrumentsConfig) -> list:
    source = OfflineBarSource(ROOTS, seed=7, periods=PERIODS)
    bars = source.load(ROOTS, START, START + timedelta(days=PERIODS))
    assert bars.height > 250 * len(ROOTS)
    features = FeatureProcessor().compute(bars)
    assert isinstance(features, pl.DataFrame)
    builder = MarketStateBuilder(instrument_master=instrument_master, initial_equity=INITIAL_EQUITY)
    timestamps = builder.timestamps(features)
    assert len(timestamps) > 260
    states = builder.build_series(features, timestamps)
    # Most recent window keeps the integration test fast.
    return states[-80:]


def _strategy_config():
    config = load_strategy_config(REPO_ROOT / "configs" / "strategies" / "tsmom_vol_target.yaml")
    return config.model_copy(
        update={
            "universe": StageConfig(
                name="liquidity",
                params={
                    "min_avg_dollar_volume": 0.0,
                    "min_open_interest": 0.0,
                    "top_n": 5,
                    "min_history_days": 0,
                },
            )
        }
    )


def _risk_overlays(instrument_master: InstrumentsConfig):
    quant = QuantRiskOverlay(instruments=instrument_master)
    llm = LLMRiskOverlay(
        _StaticLLMClient(),
        calendar=MacroEventCalendar.from_json(REPO_ROOT / "configs" / "macro_calendar.yaml"),
        instruments=instrument_master,
    )
    llm.set_cached_assessments(
        [make_risk_assessment(root, semantic_risk_score=1.8, confidence=0.9) for root in ROOTS]
    )
    return [quant, llm]


def test_market_states_are_point_in_time(market_states, instrument_master) -> None:
    state = market_states[-1]
    assert set(state.ids) == set(ROOTS)
    assert state.covariance is not None
    assert state.covariance_instruments == state.ids
    for feature in ("ret_1d", "ret_21d", "ret_126d", "vol_20d", "volume_oi_ratio"):
        assert np.isfinite(state.feature("ES", feature)), feature
    assert state.equity == pytest.approx(INITIAL_EQUITY)
    # Documented data-layer behaviour: a single-contract synthetic frame has no
    # term structure; term_slope/carry_annualized are NaN until a continuous
    # (multi-expiry) frame or exogenous curve is supplied. Live ingestion always
    # provides the continuous frame, so production pipelines see finite values.
    assert np.isnan(state.feature("ES", "term_slope"))


def test_full_pipeline_runs_and_attaches_targets(
    market_states, instrument_master: InstrumentsConfig
) -> None:
    translator = ContractTranslator(instruments=instrument_master)
    pipeline = build_pipeline(
        _strategy_config(),
        instrument_master,
        translator=translator,
        risk_overlays=_risk_overlays(instrument_master),
    )
    non_empty_targets = 0
    for state in market_states:
        result = pipeline.run(state)
        assert result.universe, "universe selector must return at least one root"
        assert set(result.final_weights) <= set(instrument_master.roots())
        assert "quant_risk" in str(result.diagnostics.get("risk_overlays"))
        assert "llm_risk" in str(result.diagnostics.get("risk_overlays"))
        assert not result.diagnostics.get("skipped_risk_stages")
        if result.targets is not None and len(result.targets) > 0:
            non_empty_targets += 1
            assert set(result.targets.quantities) <= set(instrument_master.roots())
            util = translator.margin_utilization(result.targets, INITIAL_EQUITY)
            assert util <= 0.60 + 1e-9
    # The synthetic trending series must produce at least some live targets,
    # otherwise the wiring (translator attachment) would be unproven.
    assert non_empty_targets > 0


def _run_backtest(market_states, instrument_master: InstrumentsConfig) -> BacktestResult:
    translator = ContractTranslator(instruments=instrument_master)
    pipeline = build_pipeline(
        _strategy_config(),
        instrument_master,
        translator=translator,
        risk_overlays=_risk_overlays(instrument_master),
    )
    engine = EventBacktester(
        pipeline,
        translator,
        instrument_master=instrument_master,
        initial_equity=INITIAL_EQUITY,
    )
    return engine.run(states=list(market_states))


def test_backtest_accounting_identity_and_analytics(
    market_states, instrument_master: InstrumentsConfig, tmp_path: Path
) -> None:
    result = _run_backtest(market_states, instrument_master)

    # -- accounting identity ------------------------------------------------
    result.check_accounting_identity(tolerance=1e-6)
    assert result.accounting_residual == pytest.approx(0.0, abs=1e-6)
    assert result.ending_equity > 0.0
    assert result.n_trades > 0
    assert result.n_rebalances > 0
    assert result.equity_curve.height == len(market_states)
    assert not result.forced_liquidations, "sane risk settings must not force liquidation"

    # -- analytics ----------------------------------------------------------
    metrics = compute_metrics_from_result(result)
    assert np.isfinite(metrics.sharpe)
    assert np.isfinite(metrics.annualized_volatility)
    assert 0.0 <= metrics.max_drawdown <= 1.0
    assert np.isfinite(metrics.rachev_ratio_05)
    assert metrics.peak_margin_utilization <= 0.60 + 1e-9
    assert metrics.price_pnl + metrics.roll_pnl == pytest.approx(
        metrics.gross_pnl, rel=0.0, abs=1e-6
    )

    # -- tear sheet JSON + HTML --------------------------------------------
    tearsheet = build_tearsheet(result, metrics, title="Full-stack integration")
    payload = tearsheet.to_dict()
    assert payload["title"] == "Full-stack integration"
    assert payload["strategy_id"] == result.strategy_id
    metrics_payload = payload["metrics"]
    assert isinstance(metrics_payload, dict)
    for key in ("cagr", "sharpe", "sortino", "calmar", "max_drawdown", "var_95_historical"):
        assert key in metrics_payload, key

    json_path = write_tearsheet(tearsheet, tmp_path / "tearsheet.json")
    html_path = write_tearsheet(tearsheet, tmp_path / "tearsheet.html")
    assert json.loads(json_path.read_text(encoding="utf-8"))["title"] == "Full-stack integration"
    assert "<html" in html_path.read_text(encoding="utf-8").lower()


def test_settings_directories(settings: Settings) -> None:
    settings.ensure_directories()
    assert settings.parquet_dir.is_dir()
    assert settings.report_dir.is_dir()
