"""Integration tests for the FuturesRL REST telemetry API.

Every test runs fully offline against the deterministic fixture clock: no
Databento key, no IBKR/TWS session and no network. The suite exercises the real
service adapters where possible (``SimulatedBroker`` for execution, the FinRL-X
pipeline + ``BacktestEngine`` for the pipeline/backtest routes) and the documented
fixture fallbacks otherwise.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from configs.settings import Settings
from fastapi.testclient import TestClient
from src.api import fixtures
from src.api.main import create_app

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Deterministic clock injected into the app (macro windows/timestamps depend on it).
FIXED_TIMESTAMP = datetime(2026, 11, 2, 17, 0, tzinfo=UTC)

#: Fast telemetry publisher interval used by the websocket tests.
TELEMETRY_INTERVAL = 0.02


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        environment="backtest",
        data_dir=tmp_path / "data",
        parquet_dir=tmp_path / "data" / "parquet",
        duckdb_path=tmp_path / "data" / "api.duckdb",
        report_dir=tmp_path / "artifacts",
        instruments_path=REPO_ROOT / "configs" / "instruments.yaml",
        telemetry_interval_seconds=TELEMETRY_INTERVAL,
        api_cors_origins=("http://localhost:3080", "http://127.0.0.1:3080"),
        # Hermetic: a developer .env enabling IBKR must never leak into these
        # offline API tests (module-scoped fixtures are built before the
        # function-scoped environment-isolation fixture runs).
        api_enable_ibkr=False,
    )


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TestClient]:
    """Application started through its lifespan with a fixed offline clock."""
    tmp_path = tmp_path_factory.mktemp("api-rest")
    app = create_app(_settings(tmp_path), clock=lambda: FIXED_TIMESTAMP)
    with TestClient(app) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------


def test_health_reports_offline_fixture_mode(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "degraded"}
    assert body["app_name"] == "FuturesRL"
    assert body["version"] == "1.9.0"
    assert body["environment"] == "backtest"
    assert body["offline_mode"] is True
    assert body["uptime_seconds"] >= 0.0
    assert body["server_time"].startswith("2026-11-02T17:00:00")


def test_cors_allows_configured_dashboard_origin(client: TestClient) -> None:
    response = client.get("/health", headers={"Origin": "http://localhost:3080"})
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3080"


def test_openapi_metadata_and_docs(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    assert schema["info"]["title"] == "FuturesRL Telemetry API"
    assert schema["info"]["version"] == "1.9.0"
    paths = schema["paths"]
    for route in (
        "/health",
        "/api/positions",
        "/api/pipeline/latest",
        "/api/risk/state",
        "/api/risk/thresholds",
        "/api/risk/flatten-all",
        "/api/term-structure",
        "/api/strategies",
        "/api/backtests",
        "/api/telemetry/status",
    ):
        assert route in paths, route
    assert client.get("/docs").status_code == 200


# ---------------------------------------------------------------------------
# Command Center: account / positions contract
# ---------------------------------------------------------------------------


def test_account_snapshot_contract(client: TestClient) -> None:
    response = client.get("/api/positions")
    assert response.status_code == 200
    body = response.json()

    assert body["net_liquidation"] > 0.0
    assert 0.0 <= body["margin_utilization"] < 1.0
    assert body["gross_leverage"] > 0.0
    assert len(body["equity_curve"]) >= 2
    curve_times = [
        datetime.fromisoformat(point["time"].replace("Z", "+00:00"))
        for point in body["equity_curve"]
    ]
    # Chart-safe time axis: strictly ascending, never in the future.
    assert curve_times == sorted(curve_times)
    assert len(set(curve_times)) == len(curve_times)
    assert curve_times[-1] <= FIXED_TIMESTAMP
    assert curve_times[-1] == FIXED_TIMESTAMP
    assert len(body["positions"]) >= 5
    # v1.5.1 attribution fields: offline/fixture semantics unchanged.
    assert body["equity_curve_source"] == "fixture"
    assert body["equity_curve_live_points"] == 0
    assert body["portfolio_volatility"] is None
    assert body["portfolio_volatility_source"] is None

    for position in body["positions"]:
        assert set(position) >= {
            "instrument_id",
            "root_symbol",
            "asset_class",
            "contract_symbol",
            "expiration",
            "days_to_roll",
            "quantity",
            "average_price",
            "market_price",
            "multiplier",
            "notional",
            "weight",
            "unrealized_pnl",
            "initial_margin",
            "maintenance_margin",
        }
        assert position["multiplier"] > 0
        assert math.isfinite(position["unrealized_pnl"])
        assert position["notional"] == pytest.approx(
            position["quantity"] * position["market_price"] * position["multiplier"], rel=1e-9
        )
        assert isinstance(position["days_to_roll"], int)

    # Weights are signed notional fractions of NLV and mirror the position book.
    total_weight = sum(position["weight"] for position in body["positions"])
    assert total_weight == pytest.approx(body["net_exposure"] / body["net_liquidation"], rel=1e-6)
    assert body["initial_margin_used"] == pytest.approx(
        body["margin_utilization"] * body["net_liquidation"], rel=1e-9
    )


# ---------------------------------------------------------------------------
# Pipeline Inspector
# ---------------------------------------------------------------------------


def test_pipeline_waterfall_contract(client: TestClient) -> None:
    response = client.get("/api/pipeline/latest")
    assert response.status_code == 200
    body = response.json()

    assert [stage["key"] for stage in body["stages"]] == [
        "universe",
        "base",
        "timing",
        "risk",
        "targets",
    ]
    assert body["cycle_id"].startswith("rc-20261102")
    assert body["universe"]
    assert body["elapsed_ms"] >= 0.0
    assert body["data_source"] in {"engine", "fixture"}

    gross_by_stage = {stage["key"]: stage["gross"] for stage in body["stages"]}
    assert gross_by_stage["base"] > 0.0
    assert gross_by_stage["targets"] == pytest.approx(
        sum(abs(row["weight_final"]) for row in body["instruments"]), rel=1e-6
    )

    for row in body["instruments"]:
        # Weights are published rounded to 8 decimals; deltas are computed from
        # the unrounded stage values before rounding.
        assert row["weight_final"] == pytest.approx(row["weight_risk"], abs=1e-8)
        assert row["delta_timing"] == pytest.approx(
            row["weight_timing"] - row["weight_base"], abs=1e-6
        )
        assert row["delta_risk"] == pytest.approx(
            row["weight_risk"] - row["weight_timing"], abs=1e-6
        )

    assert body["diagnostics"]
    for diagnostic in body["diagnostics"]:
        assert diagnostic["status"] in {"ok", "info", "warn", "critical"}

    # The real risk stack must be wired: quant + the LLM Risk-First overlay.
    risk_diagnostics = [item for item in body["diagnostics"] if item["stage"] == "Risk overlays"]
    assert risk_diagnostics, body["diagnostics"]
    applied = risk_diagnostics[0]
    assert applied["status"] == "ok"
    assert applied["value"] == "2 applied"
    assert "quant_risk" in applied["detail"] and "llm_risk" in applied["detail"]
    assert not [item for item in body["diagnostics"] if item["stage"] == "Risk stage skipped"]


# ---------------------------------------------------------------------------
# LLM Risk-First Monitor
# ---------------------------------------------------------------------------


def test_risk_state_contract(client: TestClient) -> None:
    response = client.get("/api/risk/state")
    assert response.status_code == 200
    body = response.json()

    thresholds = body["thresholds"]
    assert 1.0 <= thresholds["llm_safe_score"] < thresholds["llm_critical_score"] <= 5.0
    assert 0.0 <= thresholds["llm_confidence_threshold"] <= 1.0
    assert (
        0.0
        < thresholds["margin_warning_threshold"]
        < thresholds["margin_critical_threshold"]
        <= 1.0
    )

    assert body["assessments"]
    assert body["macro_events"]
    assert body["news"]
    for assessment in body["assessments"]:
        assert assessment["action"] in {"pass", "scale", "liquidate", "neutralized"}
        assert 0.0 <= assessment["confidence"] <= 1.0
        assert 1.0 <= assessment["semantic_risk_score"] <= 5.0
        assert 0.0 <= assessment["weight_multiplier"] <= 1.0
    assert body["neutralized_instruments"]
    assert "macro_lockout" in body
    assert body["circuit_breaker_active"] is False

    # Provenance: the wired LLM Risk-First overlay produced the assessments.
    assert {assessment["source"] for assessment in body["assessments"]} == {"llm-offline-client"}
    actions = {assessment["action"] for assessment in body["assessments"]}
    assert {"scale", "neutralized"} <= actions
    assert "GC" in body["neutralized_instruments"]


def test_risk_threshold_override_roundtrip_and_validation(client: TestClient) -> None:
    response = client.put(
        "/api/risk/thresholds",
        json={"llm_critical_score": 4.2, "llm_confidence_threshold": 0.75},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["llm_critical_score"] == pytest.approx(4.2)
    assert body["llm_confidence_threshold"] == pytest.approx(0.75)

    state = client.get("/api/risk/state").json()
    assert state["thresholds"]["llm_critical_score"] == pytest.approx(4.2)

    invalid = client.put("/api/risk/thresholds", json={"llm_critical_score": 1.2})
    assert invalid.status_code == 422
    # The invalid override must not have been applied.
    assert client.get("/api/risk/state").json()["thresholds"][
        "llm_critical_score"
    ] == pytest.approx(4.2)

    # Restore the committed risk contract thresholds.
    restored = client.put(
        "/api/risk/thresholds",
        json={"llm_critical_score": 4.5, "llm_confidence_threshold": 0.70},
    )
    assert restored.status_code == 200


# ---------------------------------------------------------------------------
# Term structure
# ---------------------------------------------------------------------------


def test_term_structure_contract(client: TestClient) -> None:
    response = client.get("/api/term-structure")
    assert response.status_code == 200
    body = response.json()

    assert body["curves"]
    assert body["roll_calendar"]
    for curve in body["curves"]:
        assert curve["regime"] in {"contango", "backwardation", "flat"}
        assert len(curve["contracts"]) >= 2
        contracts = curve["contracts"]
        assert contracts[0]["is_front"] is True
        assert [contract["days_to_expiry"] for contract in contracts] == sorted(
            contract["days_to_expiry"] for contract in contracts
        )
        for contract in contracts:
            assert contract["price"] > 0.0
            assert contract["days_to_expiry"] >= 0
    for roll in body["roll_calendar"]:
        assert roll["status"] in {"scheduled", "imminent", "completed"}
        assert roll["from_expiration"] <= roll["to_expiration"]
        assert isinstance(roll["days_to_roll"], int)


# ---------------------------------------------------------------------------
# Strategy config editor
# ---------------------------------------------------------------------------


def test_strategy_catalog_and_editor(client: TestClient) -> None:
    response = client.get("/api/strategies")
    assert response.status_code == 200
    strategies = response.json()
    assert {strategy["id"] for strategy in strategies} == {
        "drl_ppo_llm_risk",
        "tsmom_vol_target",
    }
    original = next(strategy for strategy in strategies if strategy["id"] == "tsmom_vol_target")
    assert original["universe"]["name"] == "liquidity"

    detail = client.get("/api/strategies/tsmom_vol_target")
    assert detail.status_code == 200
    assert detail.json()["allocator"]["name"] == "volatility_target"

    edited = client.put(
        "/api/strategies/tsmom_vol_target",
        json={"description": "edited by the API integration test"},
    )
    assert edited.status_code == 200
    assert edited.json()["description"] == "edited by the API integration test"

    persisted = client.get("/api/strategies/tsmom_vol_target")
    assert persisted.status_code == 200
    assert persisted.json()["description"] == "edited by the API integration test"

    invalid = client.put("/api/strategies/tsmom_vol_target", json={"universe": {"params": {}}})
    assert invalid.status_code == 422
    unknown = client.put("/api/strategies/does_not_exist", json={"description": "nope"})
    assert unknown.status_code == 404

    restored = client.put(
        "/api/strategies/tsmom_vol_target", json={"description": original["description"]}
    )
    assert restored.status_code == 200


# ---------------------------------------------------------------------------
# Backtest Lab
# ---------------------------------------------------------------------------


def test_backtest_list_contains_seeded_fixture_runs(client: TestClient) -> None:
    response = client.get("/api/backtests")
    assert response.status_code == 200
    body = response.json()
    assert body["available_strategies"] == ["drl_ppo_llm_risk", "tsmom_vol_target"]
    assert body["runs"]
    for run in body["runs"]:
        assert run["status"] == "completed"
        assert run["data_source"] == "fixture"
        assert run["metrics"] is not None


def test_backtest_run_uses_real_engine_and_returns_tearsheet(client: TestClient) -> None:
    request = {
        "strategy_id": "tsmom_vol_target",
        "start": "2025-01-02",
        "end": "2025-12-31",
        "initial_equity": 1_000_000.0,
        "rebalance_frequency": "W-FRI",
        "walk_forward": False,
        "random_seed": 7,
    }
    response = client.post("/api/backtests", json=request)
    assert response.status_code == 200
    run = response.json()
    assert run["status"] == "completed"
    assert run["strategy_id"] == "tsmom_vol_target"
    assert run["data_source"] == "engine", run["error"]
    assert run["error"] is None
    assert run["duration_seconds"] is not None and run["duration_seconds"] >= 0.0
    assert run["n_trades"] is not None and run["n_trades"] > 0
    assert run["n_rebalances"] is not None and run["n_rebalances"] > 0
    metrics = run["metrics"]
    assert metrics is not None
    assert math.isfinite(metrics["sharpe"])
    assert metrics["max_drawdown"] >= 0.0
    assert metrics["var_95"] >= 0.0

    detail = client.get(f"/api/backtests/{run['id']}")
    assert detail.status_code == 200
    assert detail.json()["id"] == run["id"]

    tearsheet = client.get(f"/api/backtests/{run['id']}/tearsheet")
    assert tearsheet.status_code == 200
    sheet = tearsheet.json()
    assert sheet["run_id"] == run["id"]
    assert len(sheet["equity_curve"]) > 100
    assert sheet["equity_curve"][0]["value"] == pytest.approx(request["initial_equity"], rel=1e-6)
    assert sheet["rolling_sharpe"]
    assert sheet["underwater"]
    assert sheet["tail_distribution"]
    assert {"gain", "loss"} == {point["tail"] for point in sheet["tail_distribution"]}
    assert sheet["roll_yield_attribution"]
    assert sheet["monthly_returns"]
    assert math.isfinite(sheet["metrics"]["rachev_ratio"])
    # task-86 additive cost-breakdown/margin-regime transparency for engine runs.
    breakdown = sheet["cost_breakdown"]
    assert breakdown is not None
    assert breakdown["total_costs"] == pytest.approx(
        breakdown["total_commissions"] + breakdown["slippage"] + breakdown["funding"]
    )
    assert breakdown["total_execution_costs"] == pytest.approx(
        breakdown["total_commissions"] + breakdown["slippage"]
    )
    assert breakdown["roll_cost_total"] >= 0.0
    margin_regime = sheet["margin_regime"]
    assert margin_regime is not None
    assert isinstance(margin_regime["stress_applied"], bool)
    assert margin_regime["stress_multiplier"] >= 1.0

    assert client.get("/api/backtests/bt-unknown").status_code == 404
    assert client.get("/api/backtests/bt-unknown/tearsheet").status_code == 404


def test_offline_llm_client_satisfies_production_overlay_contract(client: TestClient) -> None:
    """The offline client must speak the real strict JSON + overlay API (no network)."""
    from src.risk.llm_risk_overlay import LLMRiskAssessmentBatch, LLMRiskOverlay

    container = client.app.state.container
    state = container.market.latest_state()
    thresholds = container.risk.thresholds()
    offline = fixtures.OfflineLLMClient(state, thresholds, FIXED_TIMESTAMP)

    batch = LLMRiskAssessmentBatch.from_json_text(offline.payload)
    assert {record.instrument_id for record in batch.assessments} == set(state.instruments)

    overlay = LLMRiskOverlay(
        offline,
        container.settings.risk,
        calendar=None,
        instruments=container.instrument_master,
    )
    seeded = batch.to_risk_assessments(timestamp=FIXED_TIMESTAMP, source="llm-offline-client")
    overlay.set_cached_assessments(seeded)
    assert set(overlay.latest_assessments) == {record.instrument_id for record in seeded}
    # No HTTP adapter is constructed by the offline client: completion is in-process.
    assert not offline.payload.startswith("http")


def test_backtest_short_window_is_non_degenerate(client: TestClient) -> None:
    """A 6-month run must still trade (feature warm-up + fixture pipeline variant)."""
    response = client.post(
        "/api/backtests",
        json={
            "strategy_id": "tsmom_vol_target",
            "start": "2024-01-01",
            "end": "2024-06-30",
            "initial_equity": 1_000_000.0,
            "rebalance_frequency": "W-FRI",
            "walk_forward": False,
            "random_seed": 7,
        },
    )
    assert response.status_code == 200
    run = response.json()
    assert run["data_source"] == "engine", run["error"]
    assert run["error"] is None
    assert run["n_trades"] is not None and run["n_trades"] > 0
    assert run["n_rebalances"] is not None and run["n_rebalances"] > 0
    metrics = run["metrics"]
    assert metrics is not None
    assert abs(metrics["cagr"]) > 0.0
    assert metrics["turnover"] > 0.0
    assert math.isfinite(metrics["sharpe"])
    assert math.isfinite(metrics["rachev_ratio"])

    sheet = client.get(f"/api/backtests/{run['id']}/tearsheet").json()
    equity_values = [point["value"] for point in sheet["equity_curve"]]
    assert len(equity_values) > 100
    assert equity_values[0] == pytest.approx(1_000_000.0, rel=1e-6)
    assert len(set(equity_values)) > 10, "equity curve is flat: backtest produced no activity"


def test_backtest_rejects_unknown_strategy_and_invalid_window(client: TestClient) -> None:
    unknown = client.post(
        "/api/backtests",
        json={
            "strategy_id": "no_such_strategy",
            "start": "2025-01-02",
            "end": "2025-12-31",
            "initial_equity": 1_000_000.0,
        },
    )
    assert unknown.status_code == 404
    invalid = client.post(
        "/api/backtests",
        json={
            "strategy_id": "tsmom_vol_target",
            "start": "2025-12-31",
            "end": "2025-01-02",
            "initial_equity": 1_000_000.0,
        },
    )
    assert invalid.status_code == 422


# ---------------------------------------------------------------------------
# Telemetry status
# ---------------------------------------------------------------------------


def test_telemetry_status(client: TestClient) -> None:
    response = client.get("/api/telemetry/status")
    assert response.status_code == 200
    body = response.json()
    assert body["stream_active"] is True
    assert body["publisher_interval_seconds"] == pytest.approx(TELEMETRY_INTERVAL)
    assert "telemetry" in body["topics"]
    assert body["published_events"] >= 0


def test_unknown_route_returns_404(client: TestClient) -> None:
    assert client.get("/api/does-not-exist").status_code == 404


def test_ibkr_wiring_never_serves_simulated_book(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """IBKR enabled without a session: honest unavailable state, never fake positions."""
    monkeypatch.setenv("FUTURESRL_API_ENABLE_IBKR", "1")
    settings = _settings(tmp_path).model_copy(
        update={
            "ibkr_connect_timeout_seconds": 0.1,
            "ibkr_reconnect_max_attempts": 1,
            "ibkr_heartbeat_interval_seconds": 0.1,
        }
    )
    with TestClient(create_app(settings, clock=lambda: FIXED_TIMESTAMP)) as client:
        health = client.get("/health").json()
        # Dynamic health: IBKR is enabled, the socket is down, a reconnect probe runs.
        assert health["offline_mode"] is True
        assert health["ibkr_enabled"] is True
        assert health["ibkr_connected"] is False
        assert health["ibkr_last_error"]
        assert health["status"] == "degraded"
        # No IBKR book was ever fetched: a clear 503 instead of simulated positions.
        positions = client.get("/api/positions")
        assert positions.status_code == 503
        assert "no live account book" in positions.json()["detail"]
        status = client.get("/api/telemetry/status").json()
        assert status["stream_active"] is True


def test_health_reports_offline_ibkr_fields(client: TestClient) -> None:
    """Pure offline processes still expose the additive Phase-10 health fields."""
    health = client.get("/health").json()
    assert health["ibkr_enabled"] is False
    assert health["ibkr_reconnecting"] is False
    assert health["ibkr_last_error"] is None
    account = client.get("/api/positions").json()
    assert account["source"] in {"simulated", "fixture", "engine", "runner-bridge"}
    assert account["stale"] is False
    assert account["stale_reason"] is None


# ---------------------------------------------------------------------------
# Emergency flatten kill switch (kept last: it flattens the shared broker)
# ---------------------------------------------------------------------------


def test_emergency_flatten_all(client: TestClient) -> None:
    before = client.get("/api/positions").json()
    assert before["positions"], "expected an open demo book before flattening"

    response = client.post("/api/risk/flatten-all")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "flattened"
    assert body["flattened_positions"] == len(before["positions"])
    assert body["receipts"]
    assert body["execution_locked_until"] is not None
    for receipt in body["receipts"]:
        assert receipt["side"] in {"BUY", "SELL"}
        assert receipt["quantity"] > 0
        assert receipt["status"] in {"FILLED", "SUBMITTED", "REJECTED", "UNKNOWN"}

    after = client.get("/api/positions").json()
    assert after["positions"] == []
    assert after["net_liquidation"] > 0.0

    risk_state = client.get("/api/risk/state").json()
    assert risk_state["circuit_breaker_active"] is True
    assert "kill switch" in risk_state["circuit_breaker_reason"].lower()

    repeated = client.post("/api/risk/flatten-all")
    assert repeated.status_code == 200
    assert repeated.json()["status"] == "no_positions"
