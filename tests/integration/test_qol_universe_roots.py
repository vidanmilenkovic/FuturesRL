"""Integration repro: Studio-saved explicit universe roots must run engine backtests.

Strategy Studio persists ``universe.params.roots`` (a comma-separated root list)
next to the selector thresholds. Before the fix the liquidity selector rejected
that key via ``ParamReader`` ("unknown parameter(s) ['roots']"), so every
Studio-saved config fell back to a synthetic fixture backtest instead of the
engine.

The working tree currently carries such a Studio-saved ``tsmom_vol_target.yaml``
for the QoL repro; the Lead reverts it before the release commit, so these tests
skip cleanly when the committed YAML has no ``roots`` key.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from configs.instruments import load_instruments
from configs.settings import Settings
from configs.strategy import StrategyConfig, load_strategy_config
from fastapi.testclient import TestClient
from src.api.main import create_app
from src.strategies.selectors import build_selector

REPO_ROOT = Path(__file__).resolve().parents[2]
STUDIO_SAVED_TSMOM = REPO_ROOT / "configs" / "strategies" / "tsmom_vol_target.yaml"
FIXED_TIMESTAMP = datetime(2025, 3, 21, 12, 0, tzinfo=UTC)
BACKTEST_REQUEST: dict[str, Any] = {
    "strategy_id": "tsmom_vol_target",
    "start": "2025-01-02",
    "end": "2025-12-31",
    "initial_equity": 1_000_000.0,
    "rebalance_frequency": "W-FRI",
    "walk_forward": False,
    "random_seed": 7,
}


def _settings(tmp_path: Path) -> Settings:
    """Hermetic offline API settings (mirrors ``tests/integration/test_api.py``)."""
    return Settings(
        environment="backtest",
        data_dir=tmp_path / "data",
        parquet_dir=tmp_path / "data" / "parquet",
        duckdb_path=tmp_path / "data" / "api.duckdb",
        report_dir=tmp_path / "artifacts",
        instruments_path=REPO_ROOT / "configs" / "instruments.yaml",
        telemetry_interval_seconds=0.02,
        api_cors_origins=("http://localhost:3080",),
        api_enable_ibkr=False,
    )


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TestClient]:
    """Offline API application with a fixed clock."""
    tmp_path = tmp_path_factory.mktemp("qol-universe-roots")
    app = create_app(_settings(tmp_path), clock=lambda: FIXED_TIMESTAMP)
    with TestClient(app) as test_client:
        yield test_client


def _studio_saved_config() -> StrategyConfig:
    """Load the Working-tree Studio-saved TSMOM config (skip if reverted)."""
    config = load_strategy_config(STUDIO_SAVED_TSMOM)
    if "roots" not in config.universe.params:
        pytest.skip(
            "committed tsmom_vol_target.yaml has no Studio-saved universe roots "
            "(Lead reverted it to the release form)"
        )
    return config


def test_studio_saved_roots_build_selector_without_param_error() -> None:
    config = _studio_saved_config()
    master = load_instruments()
    # Previously raised ParamError("unknown parameter(s) ['roots']").
    selector = build_selector(config.universe.name, config.universe.params, master)
    assert selector.roots, "explicit Studio roots were not accepted by the selector"
    assert set(selector.roots) <= set(master.roots())
    assert "es" not in selector.roots and all(root.isupper() for root in selector.roots)


def test_engine_backtest_runs_with_studio_saved_roots(client: TestClient) -> None:
    _studio_saved_config()  # skip gate must run before the API call
    response = client.post("/api/backtests", json=BACKTEST_REQUEST)
    assert response.status_code == 200
    run = response.json()
    # The whole blocker: a real engine run, not the fixture fallback.
    assert run["data_source"] == "engine", run["error"]
    assert run["error"] is None
    assert run["status"] == "completed"
    assert run["strategy_id"] == "tsmom_vol_target"
    assert run["n_rebalances"] is not None and run["n_rebalances"] > 0
    assert run["n_trades"] is not None and run["n_trades"] > 0
