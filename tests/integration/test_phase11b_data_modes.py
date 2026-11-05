"""Phase-11B data-cache API + cached_real/synthetic backtest/optimize tests."""

from __future__ import annotations

import shutil
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml
from configs.settings import Settings
from fastapi.testclient import TestClient
from src.api import fixtures
from src.api.data_service import DataService, DataServiceError
from src.api.main import create_app
from src.api.phase11b_schemas import EstimateCostRequest, IngestRequest
from src.data.storage import ParquetRepository

pytestmark = pytest.mark.integration

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
FIXED_TIMESTAMP = datetime(2026, 11, 3, 22, 0, tzinfo=UTC)
TELEMETRY_INTERVAL = 0.02


def _settings(tmp_path: Path, *, roots: str = "ES,NQ") -> Settings:
    configs_dir = tmp_path / "configs"
    strategies_dir = configs_dir / "strategies"
    strategies_dir.mkdir(parents=True, exist_ok=True)
    for source in sorted((REPO_ROOT / "configs" / "strategies").glob("*.yaml")):
        shutil.copyfile(source, strategies_dir / source.name)
    base = strategies_dir / "tsmom_vol_target.yaml"
    payload = yaml.safe_load(base.read_text(encoding="utf-8"))
    payload["universe"]["params"]["roots"] = roots
    base.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return Settings(
        environment="backtest",
        configs_dir=configs_dir,
        data_dir=tmp_path / "data",
        parquet_dir=tmp_path / "data" / "parquet",
        duckdb_path=tmp_path / "data" / "api.duckdb",
        report_dir=tmp_path / "artifacts",
        instruments_path=REPO_ROOT / "configs" / "instruments.yaml",
        telemetry_interval_seconds=TELEMETRY_INTERVAL,
        api_cors_origins=("http://localhost:3080",),
    )


def _write_real_bars(settings: Settings, roots: tuple[str, ...]) -> int:
    master = settings.instruments
    timestamps = fixtures.weekdays_between(date(2023, 1, 2), date(2025, 6, 30))
    frame = fixtures._build_bars(master, roots, timestamps, seed=7)
    repository = ParquetRepository(settings)
    return repository.write_bars(frame, schema="ohlcv-1d")


class FakeFetcher:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def fetch_bars_sync(self, roots: list[str], start: datetime, end: datetime) -> Any:
        master = self._settings.instruments
        timestamps = fixtures.weekdays_between(date(2024, 1, 2), date(2024, 12, 31))
        return fixtures._build_bars(master, tuple(roots), timestamps, seed=11)


class FakeLoader:
    def __init__(self, settings: Settings, *, cost: float = 0.5) -> None:
        self._settings = settings
        self._cost = cost
        self.ingest_calls: list[dict[str, object]] = []

    def estimate_cost_usd(self, roots, start, end, schema="ohlcv-1d", **_: object) -> Any:
        return type(
            "_Estimate",
            (),
            {
                "roots": tuple(roots),
                "schema": schema,
                "start": start.date() if isinstance(start, datetime) else date(2024, 1, 2),
                "end": end.date() if isinstance(end, datetime) else date(2024, 12, 31),
                "estimated_cost_usd": self._cost,
                "billable_size_bytes": 2_000_000,
                "billable_size_megabytes": 2.0,
                "details": {"metadata_available": True},
            },
        )()

    def ingest(self, roots, start, end, schema="ohlcv-1d", *, allow_over_budget=False):
        self.ingest_calls.append(
            {
                "roots": list(roots),
                "start": start,
                "end": end,
                "schema": schema,
                "allow_over_budget": allow_over_budget,
            }
        )
        timestamps = fixtures.weekdays_between(date(2024, 1, 2), date(2024, 12, 31))
        return fixtures._build_bars(self._settings.instruments, tuple(roots), timestamps, seed=13)


def _app(settings: Settings, service: DataService | None = None) -> TestClient:
    app = create_app(settings, clock=lambda: FIXED_TIMESTAMP)
    if service is not None:
        app.state.data_service = service
    return TestClient(app)


# ---------------------------------------------------------------------------
# Data inventory / estimate / ingest
# ---------------------------------------------------------------------------


def test_data_inventory_includes_cached_and_empty_roots(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_real_bars(settings, ("ES", "NQ"))
    repository = ParquetRepository(settings)
    service = DataService(settings=settings, repository=repository, clock=lambda: FIXED_TIMESTAMP)
    response = service.inventory()
    assert response.totals.cached_roots == 2
    assert response.totals.total_roots >= 45
    assert response.totals.total_bars > 0
    cached = {row.root: row for row in response.roots}
    assert cached["ES"].cached_bars > 0
    assert cached["ES"].start_date is not None
    assert cached["CL"].cached_bars == 0
    assert cached["CL"].source == "empty"
    assert response.databento_configured is False


def test_data_inventory_route_injected(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_real_bars(settings, ("ES",))
    service = DataService(settings=settings, clock=lambda: FIXED_TIMESTAMP)
    with _app(settings, service) as client:
        response = client.get("/api/data/inventory")
        assert response.status_code == 200, response.text
        body = response.json()
        assert {"roots", "totals", "databento_configured", "ibkr_available", "schema"} <= set(body)
        assert body["totals"]["cached_roots"] >= 1


def test_estimate_cost_ibkr_and_databento(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    loader = FakeLoader(settings, cost=0.75)
    service = DataService(
        settings=settings,
        clock=lambda: FIXED_TIMESTAMP,
        databento_loader_provider=lambda: loader,
    )
    ibkr = service.estimate_cost(
        EstimateCostRequest(
            roots=["ES", "NQ"], start=date(2024, 1, 2), end=date(2024, 12, 31), source="ibkr"
        )
    )
    assert ibkr.estimated_cost_usd == 0.0
    assert ibkr.source == "ibkr"

    databento = service.estimate_cost(
        EstimateCostRequest(
            roots=["ES"], start=date(2024, 1, 2), end=date(2024, 12, 31), source="databento"
        )
    )
    assert databento.estimated_cost_usd == pytest.approx(0.75)
    assert databento.billable_size_bytes == 2_000_000
    assert databento.billable_size_mb == pytest.approx(2.0)
    assert databento.source == "databento"

    with pytest.raises(DataServiceError) as excinfo:
        service.estimate_cost(
            EstimateCostRequest(roots=["UNKNOWN"], start=date(2024, 1, 2), end=date(2024, 12, 31))
        )
    assert excinfo.value.status_code == 422


def test_ingest_ibkr_free_persists_and_records_source(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    service = DataService(
        settings=settings,
        clock=lambda: FIXED_TIMESTAMP,
        ibkr_available_provider=lambda: True,
        ibkr_fetcher_provider=lambda: FakeFetcher(settings),
    )
    response = service.ingest(
        IngestRequest(
            source="ibkr",
            roots=["ES", "NQ"],
            start=date(2024, 1, 2),
            end=date(2024, 12, 31),
        )
    )
    assert response.source == "ibkr"
    assert response.roots_ingested == ["ES", "NQ"]
    assert response.bars_written > 0
    assert response.cost_estimate is None
    assert [row.root for row in response.continuous_series] == ["ES", "NQ"]
    inventory = service.inventory()
    cached = {row.root: row for row in inventory.roots}
    assert cached["ES"].source == "ibkr"
    assert cached["ES"].cached_bars == response.bars_written // 2


def test_ingest_ibkr_unavailable_is_409(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    service = DataService(
        settings=settings,
        clock=lambda: FIXED_TIMESTAMP,
        ibkr_available_provider=lambda: False,
    )
    with pytest.raises(DataServiceError) as excinfo:
        service.ingest(
            IngestRequest(
                source="ibkr", roots=["ES"], start=date(2024, 1, 2), end=date(2024, 12, 31)
            )
        )
    assert excinfo.value.status_code == 503
    assert excinfo.value.detail is not None
    assert excinfo.value.detail["error"] == "ibkr_unreachable"
    assert f"{settings.ibkr_host}:{settings.ibkr_port}" in str(excinfo.value.detail["message"])


def test_ingest_databento_budget_guard_and_override(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    loader = FakeLoader(settings, cost=5.0)
    service = DataService(
        settings=settings,
        clock=lambda: FIXED_TIMESTAMP,
        databento_loader_provider=lambda: loader,
    )
    with pytest.raises(DataServiceError) as excinfo:
        service.ingest(
            IngestRequest(
                source="databento",
                roots=["ES"],
                start=date(2024, 1, 2),
                end=date(2024, 12, 31),
            )
        )
    assert excinfo.value.status_code == 402
    assert excinfo.value.detail is not None
    assert excinfo.value.detail["error"] == "databento_budget_exceeded"
    assert excinfo.value.detail["override_allowed"] is True
    assert "$5.00 > $2.00 cap" in str(excinfo.value.detail["message"])
    assert loader.ingest_calls == []  # never downloaded when the guard refuses

    response = service.ingest(
        IngestRequest(
            source="databento",
            roots=["ES"],
            start=date(2024, 1, 2),
            end=date(2024, 12, 31),
            allow_over_budget=True,
            confirm=True,
        )
    )
    assert loader.ingest_calls and loader.ingest_calls[0]["allow_over_budget"] is True
    assert response.bars_written > 0
    assert response.cost_estimate is not None
    assert response.cost_estimate.estimated_cost_usd == pytest.approx(5.0)
    assert response.warnings


# ---------------------------------------------------------------------------
# Backtest data modes
# ---------------------------------------------------------------------------


def test_backtest_cached_real_uses_parquet_bars(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_real_bars(settings, ("ES", "NQ"))
    with _app(settings) as client:
        response = client.post(
            "/api/backtests",
            json={
                "strategy_id": "tsmom_vol_target",
                "start": "2025-01-02",
                "end": "2025-03-31",
                "initial_equity": 1_000_000.0,
                "rebalance_frequency": "W-FRI",
                "walk_forward": False,
                "random_seed": 7,
                "data_mode": "cached_real",
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["data_mode"] == "cached_real"
        assert body["data_origin"] == "real_parquet"
        assert body["cached_bars"] is not None
        assert body["cached_bars"]["roots"] == ["ES", "NQ"]
        assert body["cached_bars"]["bar_count"] > 0
        assert body["timeframe"] == "1d"
        assert body["cached_bars"]["timeframe"] == "1d"
        assert body["data_source"] == "engine"


def test_backtest_auto_falls_back_synthetic_and_cached_real_missing_422(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with _app(settings) as client:
        auto = client.post(
            "/api/backtests",
            json={
                "strategy_id": "tsmom_vol_target",
                "start": "2025-01-02",
                "end": "2025-03-31",
                "walk_forward": False,
            },
        )
        assert auto.status_code == 200, auto.text
        assert auto.json()["data_origin"] == "synthetic_fixture"
        assert auto.json()["data_mode"] == "auto"
        assert auto.json()["cached_bars"] is None

        missing = client.post(
            "/api/backtests",
            json={
                "strategy_id": "tsmom_vol_target",
                "start": "2025-01-02",
                "end": "2025-03-31",
                "data_mode": "cached_real",
            },
        )
        assert missing.status_code == 422
        assert "cached real" in missing.json()["detail"]


def test_backtest_synthetic_offline_ignores_cache(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_real_bars(settings, ("ES", "NQ"))
    with _app(settings) as client:
        response = client.post(
            "/api/backtests",
            json={
                "strategy_id": "tsmom_vol_target",
                "start": "2025-01-02",
                "end": "2025-03-31",
                "data_mode": "synthetic_offline",
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["data_origin"] == "synthetic_fixture"
        assert response.json()["data_mode"] == "synthetic_offline"


def test_optimize_cached_real_metadata(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_real_bars(settings, ("ES", "NQ"))
    with _app(settings) as client:
        response = client.post(
            "/api/optimize",
            json={
                "base_strategy_id": "tsmom_vol_target",
                "n_trials": 2,
                "wf_splits": 2,
                "data_mode": "cached_real",
            },
        )
        assert response.status_code == 200, response.text
        study = response.json()
        assert study["data_mode"] == "cached_real"
        assert study["data_origin"] == "real_parquet"
        assert study["cached_bars"] is not None
        assert study["cached_bars"]["roots"] == ["ES", "NQ"]
        assert study["timeframe"] == "1d"
        assert study["cached_bars"]["timeframe"] == "1d"
        listed = {row["study_id"]: row for row in client.get("/api/optimize").json()}
        assert listed[study["study_id"]]["data_mode"] == "cached_real"
        assert listed[study["study_id"]]["data_origin"] == "real_parquet"


def test_optimize_cached_real_missing_is_422(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with _app(settings) as client:
        response = client.post(
            "/api/optimize",
            json={
                "base_strategy_id": "tsmom_vol_target",
                "n_trials": 2,
                "wf_splits": 2,
                "data_mode": "cached_real",
            },
        )
        assert response.status_code == 422, response.text
        assert "cached real" in response.json()["detail"]


def test_estimate_missing_databento_key_is_400_with_env_name(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    class MissingKeyLoader:
        def estimate_cost_usd(self, *args: object, **kwargs: object) -> object:
            raise RuntimeError(
                "Databento API key is not configured: set FUTURESRLX_DATABENTO_API_KEY"
            )

    service = DataService(
        settings=settings,
        clock=lambda: FIXED_TIMESTAMP,
        databento_loader_provider=lambda: MissingKeyLoader(),
    )
    with pytest.raises(DataServiceError) as excinfo:
        service.estimate_cost(
            EstimateCostRequest(roots=["ES"], start=date(2024, 1, 2), end=date(2024, 12, 31))
        )
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail is not None
    assert excinfo.value.detail["error"] == "databento_api_key_missing"
    assert "FUTURESRLX_DATABENTO_API_KEY" in str(excinfo.value.detail["message"])


def test_inventory_timeframe_fields(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_real_bars(settings, ("ES",))
    service = DataService(settings=settings, clock=lambda: FIXED_TIMESTAMP)
    response = service.inventory(timeframe="1d")
    assert response.timeframe == "1d"
    assert response.base_schema == "ohlcv-1d"
    assert response.schema == "ohlcv-1d"
    assert all(row.timeframe == "1d" for row in response.roots)


def test_ingest_databento_resamples_to_target_timeframe(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    class MinuteLoader:
        def __init__(self) -> None:
            self.ingest_calls: list[dict[str, object]] = []

        def estimate_cost_usd(self, roots, start, end, schema="ohlcv-1d", **_: object):
            return type(
                "_Estimate",
                (),
                {
                    "roots": tuple(roots),
                    "schema": schema,
                    "start": start,
                    "end": end,
                    "estimated_cost_usd": 0.1,
                    "billable_size_bytes": 1000,
                    "billable_size_megabytes": 0.001,
                    "details": {},
                },
            )()

        def ingest(self, roots, start, end, schema="ohlcv-1d", *, allow_over_budget=False):
            self.ingest_calls.append({"schema": schema})
            timestamps = tuple(
                datetime(2024, 1, 2, hour, 0, tzinfo=UTC) for hour in range(0, 24, 1)
            ) + tuple(datetime(2024, 1, 3, hour, 0, tzinfo=UTC) for hour in range(0, 24, 1))
            return fixtures._build_bars(settings.instruments, tuple(roots), timestamps, seed=21)

    loader = MinuteLoader()
    service = DataService(
        settings=settings,
        clock=lambda: FIXED_TIMESTAMP,
        databento_loader_provider=lambda: loader,
    )
    response = service.ingest(
        IngestRequest(
            source="databento",
            roots=["ES"],
            start=date(2024, 1, 2),
            end=date(2024, 1, 3),
            timeframe="5m",
        )
    )
    assert loader.ingest_calls and loader.ingest_calls[0]["schema"] == "ohlcv-1m"
    assert response.timeframe == "5m"
    assert response.base_schema == "ohlcv-1m"
    assert response.resampled is True
    assert response.bars_written > 0
    inventory = service.inventory(timeframe="5m")
    cached = {row.root: row for row in inventory.roots}
    assert cached["ES"].cached_bars == response.bars_written
    assert cached["ES"].schema == "ohlcv-5m"


def test_synthetic_backtest_propagates_timeframe(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with _app(settings) as client:
        response = client.post(
            "/api/backtests",
            json={
                "strategy_id": "tsmom_vol_target",
                "start": "2025-01-02",
                "end": "2025-03-31",
                "data_mode": "synthetic_offline",
                "timeframe": "1h",
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["timeframe"] == "1h"
        assert body["data_origin"] == "synthetic_fixture"


def test_data_routes_surface_structured_errors(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    loader = FakeLoader(settings, cost=5.0)
    service = DataService(
        settings=settings,
        clock=lambda: FIXED_TIMESTAMP,
        databento_loader_provider=lambda: loader,
        ibkr_available_provider=lambda: False,
    )
    with _app(settings, service) as client:
        budget = client.post(
            "/api/data/ingest",
            json={
                "source": "databento",
                "roots": ["ES"],
                "start": "2024-01-02",
                "end": "2024-12-31",
            },
        )
        assert budget.status_code == 402, budget.text
        detail = budget.json()["detail"]
        assert detail["error"] == "databento_budget_exceeded"
        assert "$5.00 > $2.00 cap" in detail["message"]
        assert detail["override_allowed"] is True

        unreachable = client.post(
            "/api/data/ingest",
            json={
                "source": "ibkr",
                "roots": ["ES"],
                "start": "2024-01-02",
                "end": "2024-12-31",
            },
        )
        assert unreachable.status_code == 503, unreachable.text
        ibkr_detail = unreachable.json()["detail"]
        assert ibkr_detail["error"] == "ibkr_unreachable"
        assert f"{settings.ibkr_host}:{settings.ibkr_port}" in ibkr_detail["message"]

    class MissingKeyLoader:
        def estimate_cost_usd(self, *args: object, **kwargs: object) -> object:
            raise RuntimeError(
                "Databento API key is not configured: set FUTURESRLX_DATABENTO_API_KEY"
            )

    missing_service = DataService(
        settings=settings,
        clock=lambda: FIXED_TIMESTAMP,
        databento_loader_provider=lambda: MissingKeyLoader(),
    )
    with _app(settings, missing_service) as client:
        missing = client.post(
            "/api/data/estimate-cost",
            json={"roots": ["ES"], "start": "2024-01-02", "end": "2024-12-31"},
        )
        assert missing.status_code == 400, missing.text
        key_detail = missing.json()["detail"]
        assert key_detail["error"] == "databento_api_key_missing"
        assert "FUTURESRLX_DATABENTO_API_KEY" in key_detail["message"]


def test_ingest_ibkr_resamples_to_target_timeframe(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    fetcher = FakeFetcher(settings)
    service = DataService(
        settings=settings,
        clock=lambda: FIXED_TIMESTAMP,
        ibkr_available_provider=lambda: True,
        ibkr_fetcher_provider=lambda: fetcher,
    )
    response = service.ingest(
        IngestRequest(
            source="ibkr",
            roots=["ES"],
            start=date(2024, 1, 2),
            end=date(2024, 12, 31),
            timeframe="5m",
        )
    )
    assert response.timeframe == "5m"
    assert response.base_schema == "ohlcv-1m"
    assert response.resampled is True
    assert response.bars_written > 0
    inventory = service.inventory(timeframe="5m")
    cached = {row.root: row for row in inventory.roots}
    assert cached["ES"].cached_bars == response.bars_written
    assert cached["ES"].schema == "ohlcv-5m"


def test_invalid_timeframe_validation_and_ibkr_estimate_echo(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    service = DataService(settings=settings, clock=lambda: FIXED_TIMESTAMP)
    with _app(settings, service) as client:
        invalid_inventory = client.get("/api/data/inventory?timeframe=2h")
        assert invalid_inventory.status_code == 422, invalid_inventory.text
        inventory_detail = invalid_inventory.json()["detail"]
        assert inventory_detail["error"] == "unsupported_timeframe"

        invalid_estimate = client.post(
            "/api/data/estimate-cost",
            json={
                "roots": ["ES"],
                "start": "2024-01-02",
                "end": "2024-12-31",
                "source": "ibkr",
                "timeframe": "2h",
            },
        )
        assert invalid_estimate.status_code == 422, invalid_estimate.text
        assert invalid_estimate.json()["detail"]["error"] == "unsupported_timeframe"

        estimate = client.post(
            "/api/data/estimate-cost",
            json={
                "roots": ["ES"],
                "start": "2024-01-02",
                "end": "2024-12-31",
                "source": "ibkr",
                "timeframe": "5m",
            },
        )
        assert estimate.status_code == 200, estimate.text
        body = estimate.json()
        assert body["timeframe"] == "5m"
        assert body["schema"] == "ohlcv-5m"
        assert body["base_schema"] == "ohlcv-1m"
        assert body["estimated_cost_usd"] == 0.0
        assert body["source"] == "ibkr"
