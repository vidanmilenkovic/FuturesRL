"""Phase-9 core integration: discretization diagnostics, LLM provider factory,
news ingestion and the public walk-forward API wired to real platform modules.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from configs.instruments import InstrumentsConfig, load_instruments
from configs.settings import Settings
from configs.strategy import StageConfig, load_strategy_config
from src.backtest.engine import run_walk_forward
from src.core.types import WeightVector
from src.data.news_ingestor import NewsIngestor
from src.risk import (
    ContractTranslator,
    DiscretizationReport,
    LLMRiskConfig,
    LLMRiskOverlay,
    MicroRoute,
    build_llm_client,
)
from src.strategies import build_pipeline

from tests.fixtures.market import FIXED_TIMESTAMP, make_market_state

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOTS = ("ES", "NQ", "CL", "GC", "ZN")


@pytest.fixture(scope="module")
def instrument_master() -> InstrumentsConfig:
    return load_instruments(REPO_ROOT / "configs" / "instruments.yaml")


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        environment="backtest",
        data_dir=tmp_path / "data",
        parquet_dir=tmp_path / "data" / "parquet",
        duckdb_path=tmp_path / "data" / "x.duckdb",
        report_dir=tmp_path / "artifacts",
        instruments_path=REPO_ROOT / "configs" / "instruments.yaml",
    )


def _pipeline(master: InstrumentsConfig, translator: ContractTranslator):
    config = load_strategy_config(REPO_ROOT / "configs" / "strategies" / "tsmom_vol_target.yaml")
    config = config.model_copy(
        update={
            "universe": StageConfig(
                name="liquidity",
                params={
                    "min_avg_dollar_volume": 0.0,
                    "min_open_interest": 0.0,
                    "top_n": 5,
                    "min_history_days": 0,
                },
            ),
            "allocator": StageConfig(name="equal_weight", params={}),
            "timing": (),
            "risk": (),
        }
    )
    return build_pipeline(config, master, translator=translator, risk_overlays=[])


# ---------------------------------------------------------------------------
# Translator discretization report (Part A2 backend)
# ---------------------------------------------------------------------------


def test_discretization_report_matches_manual_weights(instrument_master: InstrumentsConfig) -> None:
    translator = ContractTranslator(instruments=instrument_master)
    state = make_market_state(ROOTS, timestamp=FIXED_TIMESTAMP, equity=1_000_000.0)
    result = _pipeline(instrument_master, translator).run(state)
    assert result.targets is not None
    report = translator.discretization_report(
        result.final_weights, result.targets, state, equity=1_000_000.0
    )
    assert isinstance(report, DiscretizationReport)
    assert isinstance(report.micro_routes, tuple)
    assert all(isinstance(route, MicroRoute) for route in report.micro_routes)

    # tracking error == half the L1 distance between target and achieved weights
    union = set(report.target_weights) | set(report.achieved_weights)
    manual = 0.5 * sum(
        abs(report.target_weights.get(root, 0.0) - report.achieved_weights.get(root, 0.0))
        for root in union
    )
    assert report.tracking_error == pytest.approx(manual, abs=1e-12)
    assert report.tracking_error_pct == pytest.approx(report.tracking_error * 100.0)
    assert report.tracking_error >= 0.0
    assert report.margin_utilization >= 0.0

    # Achieved weights are consistent with the emitted contract quantities (USD).
    for target in result.targets:
        assert report.achieved_weights[target.instrument_id] == pytest.approx(
            target.signed_notional / 1_000_000.0, rel=1e-9
        )

    # Micro routing summary reflects the translator output families.
    routed_roots = {route.micro_root for route in report.micro_routes if route.micro_root}
    target_roots = set(result.targets.quantities)
    assert routed_roots <= target_roots


def test_discretization_empty_book_is_zero(instrument_master: InstrumentsConfig) -> None:
    translator = ContractTranslator(instruments=instrument_master)
    state = make_market_state(ROOTS, equity=1_000_000.0)
    empty = WeightVector.empty()
    report = translator.discretization_report(empty, translator.translate(empty, state), state)
    assert report.tracking_error == 0.0
    assert report.tracking_error_pct == 0.0


# ---------------------------------------------------------------------------
# LLM provider factory (Part B3 switch)
# ---------------------------------------------------------------------------


def test_offline_llm_factory_drives_the_risk_overlay(
    instrument_master: InstrumentsConfig, tmp_path: Path
) -> None:
    settings = _settings(tmp_path)
    assert settings.llm_provider == "offline"
    client = build_llm_client(settings)
    overlay = LLMRiskOverlay(
        client,
        LLMRiskConfig(cache_ttl_seconds=60.0, sample_count=1),
        instruments=instrument_master,
    )
    state = make_market_state(ROOTS, timestamp=FIXED_TIMESTAMP)

    async def refresh():
        return await overlay.refresh(state, ROOTS)

    assessments = __import__("asyncio").run(refresh())
    assert assessments, "offline client must produce deterministic assessments"
    ids = {assessment.instrument_id for assessment in assessments}
    assert ids <= set(ROOTS)
    for assessment in assessments:
        assert 1.0 <= assessment.semantic_risk_score <= 5.0
        assert 0.0 <= assessment.confidence <= 1.0
    assert overlay.is_cache_fresh()
    # Neutral offline scores (S <= theta_safe) must pass weights through.
    out = overlay.apply_risk(WeightVector({"ES": 0.4, "CL": -0.2}), state)
    assert out.get("ES") == pytest.approx(0.4)
    assert out.get("CL") == pytest.approx(-0.2)


# ---------------------------------------------------------------------------
# News ingestor (Part B3)
# ---------------------------------------------------------------------------


def test_news_ingestor_offline_fixture_is_deterministic(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    ingestor = NewsIngestor(settings, clock=lambda: FIXED_TIMESTAMP)
    first = ingestor.fetch(limit=5)
    second = NewsIngestor(settings, clock=lambda: FIXED_TIMESTAMP).fetch(limit=5)
    assert ingestor.source_kind == "fixture"
    assert first == second
    assert 1 <= len(first) <= 5
    payload = ingestor.to_dict(first)
    json.dumps(payload)  # strictly JSON-serialisable
    for row in payload:
        assert row["source_kind"] in {"rss", "ibkr", "fixture"}
        assert row["severity"] in {"info", "warn", "critical"}
        datetime.fromisoformat(str(row["published_at"]))


# ---------------------------------------------------------------------------
# Public walk-forward API (Part A3 backend)
# ---------------------------------------------------------------------------


def test_walk_forward_report_end_to_end(instrument_master: InstrumentsConfig) -> None:
    config = load_strategy_config(REPO_ROOT / "configs" / "strategies" / "tsmom_vol_target.yaml")
    config = config.model_copy(
        update={
            "universe": StageConfig(
                name="liquidity",
                params={
                    "min_avg_dollar_volume": 0.0,
                    "min_open_interest": 0.0,
                    "top_n": 3,
                    "min_history_days": 0,
                },
            ),
            "allocator": StageConfig(name="equal_weight", params={}),
            "timing": (),
            "risk": (),
        }
    )
    states = [
        make_market_state(
            ("ES", "NQ", "CL"),
            timestamp=datetime(2024, 1, 2, 21, 0, tzinfo=UTC) + timedelta(days=index),
            equity=1_000_000.0,
        )
        for index in range(80)
    ]
    report = run_walk_forward(
        config,
        states=states,
        splits=3,
        purge_bars=2,
        embargo_bars=2,
        initial_equity=1_000_000.0,
        instrument_master=instrument_master,
    )
    payload = report.to_dict()
    assert payload["report"] == "walk_forward"
    assert report.fold_count == 3
    assert len(payload["folds"]) == 3
    assert payload["walk_forward"]["purge_bars"] == 2
    assert payload["walk_forward"]["embargo_bars"] == 2
    # No leakage: strictly ordered, disjoint test windows with an exact purge gap.
    previous_end = -1
    for fold in payload["folds"]:
        assert fold["train_end"] < fold["test_start"]
        assert fold["test_start"] == fold["train_end"] + 2
        assert fold["test_start"] >= previous_end  # contiguous, non-overlapping tests
        previous_end = fold["test_end"]
        assert fold["n_trades"] >= 0
        assert "n_rebalances" in fold
    assert report.total_trades == sum(fold["n_trades"] for fold in payload["folds"])
    assert report.total_rebalances == sum(fold["n_rebalances"] for fold in payload["folds"])
    assert report.mean_sharpe == pytest.approx(payload["aggregate"]["sharpe"]["mean"])
    assert report.final_equity == pytest.approx(payload["concatenated_equity_curve"]["equity"][-1])
    json.dumps(payload)  # strictly JSON-serialisable
