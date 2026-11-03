"""Phase-9 API integration tests: live bridge, persisted backtests, walk-forward.

Every test runs fully offline against deterministic fixtures; the runner bridge is
exercised both through a synthetic snapshot file and through the real
``OfflineRunnerSession`` when the execution package ships it. No network, no IBKR.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from configs.settings import Settings
from fastapi.testclient import TestClient
from src.api.main import create_app

pytestmark = pytest.mark.integration

REPO_ROOT: Path = Path(__file__).resolve().parents[2]

#: Deterministic clock injected into every app instance.
FIXED_TIMESTAMP = datetime(2026, 11, 2, 17, 0, tzinfo=UTC)

#: Fast telemetry interval so the app starts quickly in tests.
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
    )


def _app(settings: Settings) -> TestClient:
    return TestClient(create_app(settings, clock=lambda: FIXED_TIMESTAMP))


def _write_bridge_snapshot(
    settings: Settings,
    *,
    sequence: int = 7,
    stale: bool = False,
    fresh_after_seconds: float = 3600.0,
) -> None:
    """Publish a valid synthetic runner-bridge snapshot (execution package contract)."""
    module = pytest.importorskip("src.execution.runner_bridge")
    store = module.RunnerBridgeStore(settings)
    updated_at = FIXED_TIMESTAMP - (timedelta(hours=2) if stale else timedelta(seconds=1))
    snapshot = module.RunnerBridgeSnapshot(
        schema_version=module.BRIDGE_SCHEMA_VERSION,
        runner_id="runner-phase9-test",
        mode="offline",
        sequence=sequence,
        updated_at=updated_at,
        stale_after_seconds=fresh_after_seconds,
        status={"hydrated": True, "locked": False, "halted": False, "states_processed": 12},
        account={
            "net_liquidation": 1_250_000.0,
            "cash_balance": 900_000.0,
            "initial_margin_used": 150_000.0,
            "maintenance_margin_used": 140_000.0,
            "margin_utilization": 0.12,
        },
        pipeline={
            "timestamp": FIXED_TIMESTAMP.isoformat(),
            "universe": "ES,NQ,CL",
            "universe_size": 3,
            "gross_leverage": 0.75,
            "net_leverage": 0.25,
            "targets": 4,
            "elapsed_ms": 3.5,
            "tracking_error_pct": 1.25,
        },
        positions=(
            module.BridgePosition(
                instrument_id="ES",
                contract_symbol="ES_202612",
                quantity=2,
                average_price=4800.0,
                market_price=4820.0,
                multiplier=50.0,
                unrealized_pnl=2000.0,
                initial_margin=15000.0,
            ),
        ),
        working_orders=(
            module.BridgeWorkingOrder(
                order_id="bridge-order-1",
                instrument_id="NQ",
                contract_symbol="NQ_202612",
                side="BUY",
                quantity=3,
                filled_quantity=1,
                remaining_quantity=2,
                order_type="LMT",
                limit_price=16900.0,
                status="SUBMITTED",
                updated_at=updated_at,
            ),
        ),
        fills=(
            module.BridgeFill(
                order_id="bridge-fill-1",
                instrument_id="ES",
                side="BUY",
                quantity=2,
                price=4800.0,
                commission=4.2,
                slippage_ticks=0.5,
                timestamp=updated_at - timedelta(seconds=30),
            ),
        ),
    )
    store.save(snapshot)


@pytest.fixture()
def client(tmp_path: Path) -> Iterator[TestClient]:
    with _app(_settings(tmp_path)) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# Health / system
# ---------------------------------------------------------------------------


def test_health_additive_fields(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["offline_mode"] is True
    for field in (
        "ibkr_connected",
        "runner_bridge_available",
        "runner_bridge_stale",
        "runner_mode",
    ):
        assert field in body, field
    assert body["ibkr_connected"] is False
    assert body["runner_bridge_available"] is False
    assert body["runner_bridge_stale"] is True
    assert body["runner_mode"] is None


def test_openapi_exposes_phase9_routes(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    for route in (
        "/api/orders/working",
        "/api/fills/recent",
        "/api/runner/status",
        "/api/runner/step",
        "/api/runner/reset",
        "/api/positions",
        "/api/pipeline/latest",
        "/api/backtests",
    ):
        assert route in paths, route


# ---------------------------------------------------------------------------
# Pipeline discretization
# ---------------------------------------------------------------------------


def test_pipeline_reports_discretization_fields(client: TestClient) -> None:
    body = client.get("/api/pipeline/latest").json()
    assert isinstance(body["tracking_error_pct"], float | int)
    assert body["tracking_error_pct"] >= 0.0
    assert isinstance(body["micro_routing"], list)
    assert body["tracking_error_source"] in {"translator", "fallback"}
    for row in body["instruments"]:
        assert "weight_actual" in row
        assert row["weight_actual"] is None or isinstance(row["weight_actual"], float | int)
    if body["tracking_error_source"] == "translator":
        # The real risk translator reports the discrete-book tracking error and
        # (for standard/micro families) routing rows.
        assert isinstance(body["tracking_error_pct"], float | int)
        for route in body["micro_routing"]:
            assert set(route) == {
                "family",
                "standard_root",
                "standard_quantity",
                "micro_root",
                "micro_quantity",
            }


def test_pipeline_prefers_fresh_runner_bridge(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_bridge_snapshot(settings)
    with _app(settings) as client:
        body = client.get("/api/pipeline/latest").json()
        assert body["data_source"] == "runner-bridge"
        assert body["universe"] == ["ES", "NQ", "CL"]
        assert body["tracking_error_pct"] >= 0.0
        assert [stage["key"] for stage in body["stages"]] == [
            "universe",
            "base",
            "timing",
            "risk",
            "targets",
        ]
        health = client.get("/health").json()
        assert health["runner_bridge_available"] is True
        assert health["runner_bridge_stale"] is False
        assert health["runner_mode"] == "offline"


# ---------------------------------------------------------------------------
# Orders / fills / positions (bridge-first)
# ---------------------------------------------------------------------------


def test_orders_and_fills_fallback_without_bridge(client: TestClient) -> None:
    orders = client.get("/api/orders/working").json()
    assert orders["source"] in {"simulated", "fixture"}
    assert isinstance(orders["orders"], list)
    fills = client.get("/api/fills/recent?limit=5").json()
    assert fills["source"] in {"simulated", "fixture"}
    assert fills["limit"] == 5
    assert len(fills["fills"]) <= 5
    for fill in fills["fills"]:
        assert fill["quantity"] > 0
        assert fill["price"] > 0.0
    account = client.get("/api/positions").json()
    assert account["source"] in {"simulated", "fixture", "engine"}
    assert account["net_liquidation"] > 0.0


def test_orders_fills_positions_prefer_fresh_bridge(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_bridge_snapshot(settings)
    with _app(settings) as client:
        orders = client.get("/api/orders/working").json()
        assert orders["source"] == "runner-bridge"
        assert len(orders["orders"]) == 1
        row = orders["orders"][0]
        assert row["order_id"] == "bridge-order-1"
        assert row["instrument_id"] == "NQ"
        assert row["side"] == "BUY"
        assert row["quantity"] == 3
        assert row["filled_quantity"] == 1
        assert row["remaining_quantity"] == 2
        assert row["status"] == "SUBMITTED"

        fills = client.get("/api/fills/recent?limit=10").json()
        assert fills["source"] == "runner-bridge"
        assert fills["fills"][0]["order_id"] == "bridge-fill-1"
        assert fills["fills"][0]["commission"] == pytest.approx(4.2)

        account = client.get("/api/positions").json()
        assert account["source"] == "runner-bridge"
        assert account["net_liquidation"] == pytest.approx(1_250_000.0)
        assert account["margin_utilization"] == pytest.approx(0.12)
        assert [position["instrument_id"] for position in account["positions"]] == ["ES"]
        assert account["positions"][0]["quantity"] == 2


def test_stale_bridge_falls_back(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_bridge_snapshot(settings, stale=True)
    with _app(settings) as client:
        status = client.get("/api/runner/status").json()
        assert status["bridge_available"] is True
        assert status["bridge_stale"] is True
        orders = client.get("/api/orders/working").json()
        assert orders["source"] in {"simulated", "fixture"}
        assert orders["bridge_stale"] is True
        health = client.get("/health").json()
        assert health["runner_bridge_available"] is True
        assert health["runner_bridge_stale"] is True


# ---------------------------------------------------------------------------
# Runner status / step / reset
# ---------------------------------------------------------------------------


def test_runner_status_without_bridge(client: TestClient) -> None:
    body = client.get("/api/runner/status").json()
    assert body["bridge_available"] is False
    assert body["bridge_stale"] is True
    assert body["source"] == "unavailable"
    assert body["runner_mode"] is None
    assert body["mode"] is None
    assert body["stale"] is True
    assert isinstance(body["session_available"], bool)
    assert body["session_active"] is False


def test_runner_step_invalid_body(client: TestClient) -> None:
    assert client.post("/api/runner/step", json={"states": 0}).status_code == 422
    assert client.post("/api/runner/step", json={"states": 1, "mode": "live"}).status_code == 422


def test_runner_step_with_offline_session(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    runner_module = pytest.importorskip("src.execution.runner")
    has_session = hasattr(runner_module, "OfflineRunnerSession")
    with _app(settings) as client:
        response = client.post("/api/runner/step", json={"states": 2})
        if not has_session:
            assert response.status_code == 503
            assert "offline runner session unavailable" in response.json()["detail"]
            return
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["status"] == "completed"
        assert body["mode"] == "offline"
        assert body["states_processed"] >= 2
        assert body["source"] in {"runner-bridge", "simulated", "fixture"}
        status = client.get("/api/runner/status").json()
        assert status["session_active"] is True
        assert status["session_available"] is True
        if body["source"] == "runner-bridge":
            assert status["bridge_available"] is True
            assert status["bridge_stale"] is False
            assert status["runner_mode"] == "offline"
            assert status["sequence"] is not None
            assert status["mode"] == "offline"
            assert status["stale"] is False
            assert body["bridge"] is not None
            assert body["bridge"]["mode"] == "offline"
            health = client.get("/health").json()
            assert health["runner_bridge_available"] is True
            assert health["runner_bridge_stale"] is False
            assert health["runner_mode"] == "offline"
        reset = client.post("/api/runner/reset")
        assert reset.status_code == 200
        assert reset.json()["session_active"] is False


def test_runner_step_returns_actionable_503_when_session_unavailable(
    client: TestClient,
) -> None:
    container = client.app.state.container
    controller = container.live._session
    controller._session_class = None
    controller._error = "OfflineRunnerSession is not available (fault injection)"
    response = client.post("/api/runner/step", json={"states": 1})
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "offline runner session unavailable" in detail
    assert "GET /api/runner/status" in detail


# ---------------------------------------------------------------------------
# Backtest persistence across app restarts
# ---------------------------------------------------------------------------


def _single_pass_request() -> dict[str, object]:
    return {
        "strategy_id": "tsmom_vol_target",
        "start": "2025-01-02",
        "end": "2025-12-31",
        "initial_equity": 1_000_000.0,
        "rebalance_frequency": "W-FRI",
        "walk_forward": False,
        "random_seed": 7,
    }


def test_custom_backtest_persists_across_app_instances(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with _app(settings) as first:
        created = first.post("/api/backtests", json=_single_pass_request()).json()
        assert created["status"] == "completed"
        assert created["data_source"] == "engine", created["error"]
        run_id = created["id"]
        artifact = settings.backtest_artifacts_dir / f"{run_id}.json"
        assert artifact.is_file()
        assert client_tearsheet(first, run_id)

    # A brand-new app/container instance (same Settings) must list the run.
    with _app(settings) as second:
        runs = second.get("/api/backtests").json()["runs"]
        ids = {run["id"] for run in runs}
        assert run_id in ids
        loaded = second.get(f"/api/backtests/{run_id}").json()
        assert loaded["strategy_id"] == "tsmom_vol_target"
        assert loaded["data_source"] == "engine"
        tearsheet = second.get(f"/api/backtests/{run_id}/tearsheet").json()
        assert tearsheet["run_id"] == run_id
        assert tearsheet["equity_curve"]
        # Seeded demo runs stay non-persisted (data_source=fixture).
        fixture_runs = [run for run in runs if run["data_source"] == "fixture"]
        assert fixture_runs
        for run in fixture_runs:
            assert not (settings.backtest_artifacts_dir / f"{run['id']}.json").exists()


def client_tearsheet(client: TestClient, run_id: str) -> bool:
    response = client.get(f"/api/backtests/{run_id}/tearsheet")
    return response.status_code == 200 and bool(response.json()["equity_curve"])


def test_corrupt_artifact_is_skipped_without_bricking_the_api(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with _app(settings) as first:
        created = first.post("/api/backtests", json=_single_pass_request()).json()
        run_id = created["id"]
    broken = settings.backtest_artifacts_dir / "bt-9999.json"
    broken.write_text("{not valid json", encoding="utf-8")
    with _app(settings) as second:
        runs = second.get("/api/backtests").json()["runs"]
        ids = {run["id"] for run in runs}
        assert run_id in ids
        assert "bt-9999" not in ids
        assert not broken.is_file()  # quarantined


# ---------------------------------------------------------------------------
# Walk-forward backtests
# ---------------------------------------------------------------------------


def _walk_forward_request(splits: int = 4) -> dict[str, object]:
    return {
        "strategy_id": "tsmom_vol_target",
        "start": "2025-01-02",
        "end": "2025-12-31",
        "initial_equity": 1_000_000.0,
        "rebalance_frequency": "W-FRI",
        "walk_forward": True,
        "wf_splits": splits,
        "purge_bars": 3,
        "embargo_bars": 2,
        "random_seed": 7,
    }


def test_walk_forward_backtest_returns_folds_and_metrics(client: TestClient) -> None:
    response = client.post("/api/backtests", json=_walk_forward_request())
    assert response.status_code == 200, response.text
    run = response.json()
    assert run["kind"] == "walk_forward"
    assert run["n_folds"] is not None and run["n_folds"] >= 2
    assert run["data_source"] in {"engine", "fixture"}
    if run["data_source"] == "fixture":
        assert "fallback" in (run["error"] or "")
    assert run["n_trades"] is not None and run["n_trades"] >= 0
    assert run["n_rebalances"] is not None and run["n_rebalances"] >= 0
    metrics = run["metrics"]
    for key in (
        "sharpe",
        "cagr",
        "annual_volatility",
        "max_drawdown",
        "sortino",
        "calmar",
        "omega",
        "var_95",
        "cvar_95",
        "rachev_ratio",
        "rom",
        "avg_margin_utilization",
        "turnover",
    ):
        assert key in metrics, key
        assert isinstance(metrics[key], float | int)

    tearsheet = client.get(f"/api/backtests/{run['id']}/tearsheet").json()
    folds = tearsheet["folds"]
    assert len(folds) == run["n_folds"]
    for index, fold in enumerate(folds):
        assert fold["fold_index"] == index + 1
        assert fold["equity_curve"], f"fold {index} has no equity series"
        assert fold["underwater"], f"fold {index} has no underwater series"
    assert tearsheet["equity_curve"]
    assert tearsheet["underwater"]
    assert tearsheet["metrics"] == metrics


def test_walk_forward_backtest_persists_folds(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with _app(settings) as first:
        created = first.post("/api/backtests", json=_walk_forward_request(3)).json()
        run_id = created["id"]
        assert created["kind"] == "walk_forward"
    with _app(settings) as second:
        loaded = second.get(f"/api/backtests/{run_id}").json()
        assert loaded["kind"] == "walk_forward"
        assert loaded["n_folds"] == created["n_folds"]
        tearsheet = second.get(f"/api/backtests/{run_id}/tearsheet").json()
        assert len(tearsheet["folds"]) == loaded["n_folds"]
        assert all(fold["equity_curve"] for fold in tearsheet["folds"])


def test_walk_forward_fixture_fallback_when_helper_unavailable(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing/failing walk-forward helper degrades to an explicit fixture run."""
    import src.backtest.engine as engine_module

    def _unavailable(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("walk-forward helper not available in this build")

    monkeypatch.setattr(engine_module, "run_walk_forward", _unavailable)
    response = client.post("/api/backtests", json=_walk_forward_request(3))
    assert response.status_code == 200, response.text
    run = response.json()
    assert run["kind"] == "walk_forward"
    assert run["data_source"] == "fixture"
    assert run["error"] is not None
    assert "fallback" in run["error"]
    assert run["n_folds"] == 3
    tearsheet = client.get(f"/api/backtests/{run['id']}/tearsheet").json()
    assert len(tearsheet["folds"]) == 3
    assert all(fold["equity_curve"] for fold in tearsheet["folds"])


# ---------------------------------------------------------------------------
# News provenance
# ---------------------------------------------------------------------------


def test_risk_state_reports_news_source(client: TestClient) -> None:
    body = client.get("/api/risk/state").json()
    assert "news_source" in body
    assert body["news_source"] in {"fixture", "news-ingestor"}
    assert body["news"]
    for item in body["news"]:
        assert item["headline"]
        assert item["severity"] in {"ok", "info", "warn", "critical"}
