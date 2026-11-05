"""Independent Phase 11B verification: data cache inventory, cost guard, data API,
real-vs-synthetic backtests/Optuna and frontend contract sanity.

Verification-only suite for branch ``feat/phase-11b-data-cache-manager``
(tasks 63-65); no production edits unless a real bug is found (reported first).
Everything is hermetic: ``tmp_path`` Parquet caches, fake Databento/IBKR clients,
no network and no paid vendor calls.

1. **Inventory/provenance** (task 63): all instrument-master roots (empty ones
   included), sector/schema/counts/start/end/size, legacy ``databento`` default,
   explicit ``ibkr`` sidecar, corrupt-partition tolerance.
2. **Databento cost guard** (task 63): ``metadata.get_cost``/``get_billable_size``
   before ``get_range``, hard cap refusal (no download), explicit override,
   per-call/setting caps, missing key/metadata errors.
3. **Data API** (task 64): inventory, estimate-cost (databento vs free IBKR),
   ingest with fake IBKR/Databento providers, validation, 409 IBKR-unavailable,
   409 budget refusal/no download, override with confirmation, provenance.
4. **cached_real vs synthetic** (task 64): backtest data_mode ``cached_real`` uses
   real Parquet bars (``data_origin=real_parquet``, ``cached_bars>0``), ``auto``
   falls back to synthetic with the badge fields, ``cached_real`` with an empty
   cache is an actionable error; Optuna persists the same metadata.
5. **Frontend contract sanity**: the additive ``data_origin``/``data_mode``/
   ``cached_bars``/inventory/estimate/ingest fields are present with stable types.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from configs.instruments import InstrumentsConfig, load_instruments
from configs.settings import Settings
from fastapi.testclient import TestClient
from src.api.data_service import DataService
from src.api.main import create_app
from src.api.phase11b_schemas import CostEstimateResponse, DataInventoryResponse
from src.data.databento_loader import (
    CostEstimate,
    DatabentoBudgetExceededError,
    DatabentoCostEstimateError,
    DatabentoLoader,
)
from src.data.storage import ParquetRepository

from tests.fixtures.market import make_bars_frame

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"
INSTRUMENTS_PATH = CONFIGS_DIR / "instruments.yaml"
FIXED_CLOCK = datetime(2026, 11, 3, 12, 0, tzinfo=UTC)
CACHED_ROOTS = ("ES", "NQ", "CL")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _settings(
    tmp_path: Path,
    *,
    configs_dir: Path | None = None,
    databento_api_key: str | None = None,
    max_cost: float = 2.00,
) -> Settings:
    return Settings(
        environment="backtest",
        data_dir=tmp_path / "data",
        parquet_dir=tmp_path / "data" / "parquet",
        duckdb_path=tmp_path / "data" / "phase11b.duckdb",
        report_dir=tmp_path / "artifacts",
        configs_dir=configs_dir or CONFIGS_DIR,
        instruments_path=INSTRUMENTS_PATH,
        telemetry_interval_seconds=0.02,
        api_enable_ibkr=False,
        ibkr_paper=True,
        databento_api_key=databento_api_key,
        databento_max_cost_usd_per_job=max_cost,
    )


@pytest.fixture(scope="module")
def master() -> InstrumentsConfig:
    return load_instruments(INSTRUMENTS_PATH)


def _seed_cache(
    settings: Settings,
    *,
    roots: tuple[str, ...] = CACHED_ROOTS,
    periods: int = 400,
    start: datetime | None = None,
    sources: dict[str, str] | None = None,
) -> tuple[ParquetRepository, pl.DataFrame]:
    repository = ParquetRepository(settings)
    frame = make_bars_frame(
        roots=roots,
        periods=periods,
        start=start or (FIXED_CLOCK - timedelta(days=600)),
        instrument_master=settings.instruments,
    )
    repository.write_bars(frame)
    for root, source in (sources or {}).items():
        repository.record_source(root, source)  # type: ignore[arg-type]
    return repository, frame


def _strategy_with_roots(tmp_path: Path) -> Path:
    """Hermetic configs tree whose tsmom strategy declares explicit universe roots."""
    configs = tmp_path / "configs"
    strategies = configs / "strategies"
    strategies.mkdir(parents=True, exist_ok=True)
    text = (CONFIGS_DIR / "strategies" / "tsmom_vol_target.yaml").read_text(encoding="utf-8")
    text = text.replace(
        "    min_history_days: 252\n",
        "    min_history_days: 252\n    roots: ES,NQ,CL\n",
        1,
    )
    (strategies / "tsmom_vol_target.yaml").write_text(text, encoding="utf-8")
    return configs


class _FakeDatabentoClient:
    """Fake Databento Historical client (metadata + timeseries, no network)."""

    def __init__(self, *, cost: float = 1.25, billable: int = 123_456, rows: int = 3) -> None:
        self.cost = cost
        self.billable = billable
        self.rows = rows
        self.calls: list[str] = []
        self.metadata = self
        self.timeseries = self

    def get_cost(self, **kwargs: object) -> float:
        self.calls.append("get_cost")
        return self.cost

    def get_billable_size(self, **kwargs: object) -> int:
        self.calls.append("get_billable_size")
        return self.billable

    def get_range(self, **kwargs: object) -> pl.DataFrame:
        self.calls.append("get_range")
        start = kwargs.get("start")
        base = start if isinstance(start, datetime) else FIXED_CLOCK
        return pl.DataFrame(
            {
                "ts_event": [base + timedelta(days=index) for index in range(self.rows)],
                "raw_symbol": ["ESZ6"] * self.rows,
                "open": [4800.0 + index for index in range(self.rows)],
                "high": [4810.0] * self.rows,
                "low": [4790.0] * self.rows,
                "close": [4805.0 + index for index in range(self.rows)],
                "volume": [10.0 + index for index in range(self.rows)],
            }
        )


class _FakeLoader:
    """DataService loader provider double (estimate + ingest)."""

    def __init__(self, *, cost: float = 5.0, billable: int = 2_000_000, fail: bool = False) -> None:
        self.cost = cost
        self.billable = billable
        self.fail = fail
        self.estimate_calls = 0
        self.ingest_calls: list[dict[str, object]] = []

    def estimate_cost_usd(self, roots: Any, start: Any, end: Any, schema: str) -> CostEstimate:
        self.estimate_calls += 1
        return CostEstimate(
            roots=tuple(str(root) for root in roots),
            schema=schema,
            start=start,
            end=end,
            estimated_cost_usd=self.cost,
            billable_size_bytes=self.billable,
            billable_size_megabytes=self.billable / 1_000_000.0,
        )

    def ingest(
        self,
        roots: Any,
        start: Any,
        end: Any,
        schema: str,
        *,
        allow_over_budget: bool = False,
    ) -> pl.DataFrame:
        if self.fail:
            raise RuntimeError("vendor exploded")
        self.ingest_calls.append(
            {"roots": list(roots), "allow_over_budget": allow_over_budget, "schema": schema}
        )
        return make_bars_frame(
            roots=tuple(str(root) for root in roots),
            periods=5,
            start=FIXED_CLOCK - timedelta(days=30),
            instrument_master=load_instruments(INSTRUMENTS_PATH),
        )


class _FakeIBKRFetcher:
    def __init__(self) -> None:
        self.fetch_calls: list[tuple[list[str], datetime, datetime]] = []

    def fetch_bars_sync(self, roots: Any, start: datetime, end: datetime) -> pl.DataFrame:
        self.fetch_calls.append((list(roots), start, end))
        return make_bars_frame(
            roots=tuple(str(root) for root in roots),
            periods=4,
            start=start,
            instrument_master=load_instruments(INSTRUMENTS_PATH),
        )


# ---------------------------------------------------------------------------
# 1. Inventory / provenance (task 63)
# ---------------------------------------------------------------------------


def test_inventory_covers_every_master_root_and_empty_cache(
    tmp_path: Path, master: InstrumentsConfig
) -> None:
    settings = _settings(tmp_path)
    repository, _frame = _seed_cache(settings, periods=30, sources={"ES": "ibkr"})
    rows = repository.inventory("ohlcv-1d")
    assert [row.root for row in rows] == sorted(row.root for row in rows)
    master_roots = set(master.roots())
    by_root = {row.root: row for row in rows}
    assert master_roots <= set(by_root), "every instrument-master root must be inventoried"
    assert len(rows) >= len(master_roots)
    for root in master_roots - set(CACHED_ROOTS):
        empty = by_root[root]
        assert empty.cached_bars == 0
        assert empty.source == "empty"
        assert empty.start_date is None and empty.end_date is None
        assert empty.sector == master.get(root).sector
    es = by_root["ES"]
    assert es.cached_bars == 30
    assert es.source == "ibkr"
    assert es.start_date is not None and es.end_date is not None
    assert es.start_date < es.end_date
    assert es.size_bytes > 0
    nq = by_root["NQ"]
    assert nq.cached_bars == 30
    assert nq.source == "databento", "legacy partition without sidecar defaults to databento"
    assert es.sector == master.get("ES").sector
    assert es.schema == "ohlcv-1d"
    assert es.to_dict()["start_date"] == es.start_date.isoformat()


def test_inventory_corrupt_partition_is_tolerated(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    repository, _frame = _seed_cache(settings, roots=("ES", "NQ"), periods=10)
    corrupt = repository.partition_path("NQ")
    corrupt.write_bytes(b"not a parquet file")
    rows = {row.root: row for row in repository.inventory("ohlcv-1d")}
    assert rows["NQ"].cached_bars == 0
    assert rows["NQ"].source == "empty"
    assert rows["ES"].cached_bars == 10, "a corrupt partition must not hide healthy roots"
    with pytest.raises(ValueError):
        repository.record_source("ES", "nope")  # type: ignore[arg-type]
    sidecar = repository.record_source("ES", "databento")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["source"] == "databento"
    assert payload["root"] == "ES"
    assert payload["schema"] == "ohlcv-1d"


def test_inventory_dataset_override_is_empty(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    repository, _frame = _seed_cache(settings, roots=("ES",), periods=5)
    rows = repository.inventory("ohlcv-1d", dataset="OTHER.DATASET")
    assert all(row.cached_bars == 0 and row.source == "empty" for row in rows)


# ---------------------------------------------------------------------------
# 2. Databento cost estimate + hard budget guard (task 63)
# ---------------------------------------------------------------------------


def test_cost_estimate_calls_metadata_before_any_download(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    loader = DatabentoLoader(settings, repository=ParquetRepository(settings))
    client = _FakeDatabentoClient(cost=1.25, billable=123_456)
    estimate = loader.estimate_cost_usd(
        ("ES", "NQ"), FIXED_CLOCK - timedelta(days=30), FIXED_CLOCK, client=client
    )
    assert estimate.estimated_cost_usd == pytest.approx(1.25)
    assert estimate.billable_size_bytes == 123_456
    assert estimate.billable_size_megabytes == pytest.approx(0.123456)
    assert estimate.source == "databento" and estimate.currency == "USD"
    assert client.calls == ["get_cost", "get_billable_size"]
    assert "get_range" not in client.calls, "estimating must never download"


def test_ingest_refuses_over_budget_without_download(tmp_path: Path) -> None:
    settings = _settings(tmp_path, max_cost=0.50)
    loader = DatabentoLoader(settings, repository=ParquetRepository(settings))
    client = _FakeDatabentoClient(cost=2.00)
    with pytest.raises(DatabentoBudgetExceededError) as excinfo:
        loader.ingest(("ES",), FIXED_CLOCK - timedelta(days=30), FIXED_CLOCK, client=client)
    error = excinfo.value
    assert error.estimated_cost_usd == pytest.approx(2.00)
    assert error.max_cost_usd == pytest.approx(0.50)
    assert error.roots == ("ES",)
    assert client.calls == ["get_cost", "get_billable_size"]
    assert "get_range" not in client.calls, "refused ingest must not download"
    assert not loader.repository.available_roots("ohlcv-1d")


def test_ingest_override_and_per_call_cap(tmp_path: Path) -> None:
    settings = _settings(tmp_path, max_cost=0.50)
    repository = ParquetRepository(settings)
    loader = DatabentoLoader(settings, repository=repository)
    client = _FakeDatabentoClient(cost=2.00)
    frame = loader.ingest(
        ("ES",),
        FIXED_CLOCK - timedelta(days=30),
        FIXED_CLOCK,
        client=client,
        allow_over_budget=True,
    )
    assert frame.height == 3
    assert client.calls[-1] == "get_range"
    assert repository.available_roots("ohlcv-1d") == ("ES",)
    client2 = _FakeDatabentoClient(cost=2.00)
    loader.ingest(
        ("ES",), FIXED_CLOCK - timedelta(days=30), FIXED_CLOCK, client=client2, max_cost_usd=10.0
    )
    assert client2.calls[-1] == "get_range"


def test_ingest_missing_key_and_metadata_errors_are_actionable(tmp_path: Path) -> None:
    settings = _settings(tmp_path)  # no API key configured
    loader = DatabentoLoader(settings, repository=ParquetRepository(settings))
    with pytest.raises(RuntimeError) as excinfo:
        loader.estimate_cost_usd(("ES",), FIXED_CLOCK - timedelta(days=1), FIXED_CLOCK)
    assert "No Databento API key configured" in str(excinfo.value)
    with pytest.raises(ValueError):
        loader.estimate_cost_usd((), FIXED_CLOCK - timedelta(days=1), FIXED_CLOCK, client=object())

    class _BrokenMetadata:
        def get_cost(self, **kwargs: object) -> float:
            raise RuntimeError("metadata down")

    class _BrokenClient:
        metadata = _BrokenMetadata()
        timeseries = _FakeDatabentoClient()

    with pytest.raises(DatabentoCostEstimateError) as excinfo:
        loader.estimate_cost_usd(
            ("ES",), FIXED_CLOCK - timedelta(days=1), FIXED_CLOCK, client=_BrokenClient()
        )
    assert "no data was downloaded" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 3. Data API (task 64)
# ---------------------------------------------------------------------------


@contextmanager
def _data_app(
    tmp_path: Path,
    *,
    loader: _FakeLoader | None = None,
    fetcher: _FakeIBKRFetcher | None = None,
    ibkr_available: bool = True,
    seed: bool = True,
) -> Any:
    settings = _settings(tmp_path)
    if seed:
        _seed_cache(settings, periods=40, sources={"ES": "ibkr"})
    app = create_app(settings, clock=lambda: FIXED_CLOCK)
    with TestClient(app) as client:
        service = DataService(
            settings=settings,
            repository=ParquetRepository(settings),
            instrument_master=load_instruments(INSTRUMENTS_PATH),
            clock=lambda: FIXED_CLOCK,
            databento_loader_provider=(lambda: loader) if loader is not None else None,
            ibkr_fetcher_provider=(lambda: fetcher) if fetcher is not None else None,
            ibkr_available_provider=lambda: ibkr_available,
        )
        app.state.data_service = service
        yield client, service


def test_api_inventory_contract(tmp_path: Path) -> None:
    with _data_app(tmp_path) as (client, _service):
        response = client.get("/api/data/inventory")
        assert response.status_code == 200, response.text
        body = response.json()
        for field in (
            "roots",
            "totals",
            "databento_configured",
            "ibkr_available",
            "schema",
            "generated_at",
        ):
            assert field in body
        assert body["schema"] == "ohlcv-1d"
        assert body["databento_configured"] is False
        assert body["ibkr_available"] is True
        roots = {row["root"]: row for row in body["roots"]}
        assert len(roots) >= 45
        assert roots["ES"]["cached_bars"] == 40
        assert roots["ES"]["source"] == "ibkr"
        assert roots["NQ"]["source"] == "databento"
        assert roots["6A"]["cached_bars"] == 0 and roots["6A"]["source"] == "empty"
        totals = body["totals"]
        assert totals["cached_roots"] == 3
        assert totals["total_roots"] == len(roots)
        assert totals["total_bars"] == 120
        assert totals["total_size_bytes"] > 0
        assert totals["start_date"] and totals["end_date"]
        # Typed model accepts the payload (frontend contract shape).
        DataInventoryResponse.model_validate(body)


def test_api_estimate_cost_ibkr_free_and_databento(tmp_path: Path) -> None:
    loader = _FakeLoader(cost=3.5, billable=5_000_000)
    with _data_app(tmp_path, loader=loader) as (client, _service):
        ibkr = client.post(
            "/api/data/estimate-cost",
            json={
                "roots": ["ES", "NQ"],
                "start": "2026-01-01",
                "end": "2026-01-10",
                "source": "ibkr",
            },
        )
        assert ibkr.status_code == 200, ibkr.text
        body = ibkr.json()
        assert body["source"] == "ibkr"
        assert body["estimated_cost_usd"] == 0.0
        assert body["billable_size_bytes"] == 0
        assert body["currency"] == "USD"
        assert loader.estimate_calls == 0, "IBKR estimate must not touch Databento"

        databento = client.post(
            "/api/data/estimate-cost",
            json={
                "roots": ["ES", "NQ"],
                "start": "2026-01-01",
                "end": "2026-01-10",
                "source": "databento",
            },
        )
        assert databento.status_code == 200, databento.text
        estimate = databento.json()
        assert estimate["source"] == "databento"
        assert estimate["estimated_cost_usd"] == pytest.approx(3.5)
        assert estimate["billable_size_bytes"] == 5_000_000
        assert estimate["billable_size_mb"] == pytest.approx(5.0)
        assert estimate["roots"] == ["ES", "NQ"]
        CostEstimateResponse.model_validate(estimate)

        unknown = client.post(
            "/api/data/estimate-cost",
            json={
                "roots": ["NOPE"],
                "start": "2026-01-01",
                "end": "2026-01-10",
                "source": "ibkr",
            },
        )
        assert unknown.status_code == 422
        bad_window = client.post(
            "/api/data/estimate-cost",
            json={
                "roots": ["ES"],
                "start": "2026-01-10",
                "end": "2026-01-01",
                "source": "ibkr",
            },
        )
        assert bad_window.status_code == 422


def test_api_estimate_cost_without_databento_key_is_400(tmp_path: Path) -> None:
    # No provider injected and no key -> the real loader raises the actionable error.
    with _data_app(tmp_path) as (client, _service):
        response = client.post(
            "/api/data/estimate-cost",
            json={
                "roots": ["ES"],
                "start": "2026-01-01",
                "end": "2026-01-10",
                "source": "databento",
            },
        )
        assert response.status_code == 400, response.text
        detail = response.json()["detail"]
        assert detail["error"] == "databento_api_key_missing"
        assert "FUTURESRLX_DATABENTO_API_KEY" in detail["message"]


def test_api_ingest_databento_budget_refusal_and_override(tmp_path: Path) -> None:
    loader = _FakeLoader(cost=5.0)
    with _data_app(tmp_path / "app", loader=loader) as (client, service):
        refused = client.post(
            "/api/data/ingest",
            json={
                "source": "databento",
                "roots": ["ES", "NQ"],
                "start": "2026-01-01",
                "end": "2026-01-10",
            },
        )
        assert refused.status_code == 402, refused.text
        detail = refused.json()["detail"]
        assert detail["error"] == "databento_budget_exceeded"
        assert detail["max_cost_usd"] == pytest.approx(2.00)
        assert detail["estimated_cost_usd"] == pytest.approx(5.0)
        assert "$5.00 > $2.00 cap" in detail["message"]
        assert detail["override_allowed"] is True
        assert loader.ingest_calls == [], "a refused job must never download"

        missing_confirm = client.post(
            "/api/data/ingest",
            json={
                "source": "databento",
                "roots": ["ES", "NQ"],
                "start": "2026-01-01",
                "end": "2026-01-10",
                "allow_over_budget": True,
                "confirm": False,
            },
        )
        assert missing_confirm.status_code == 400
        assert loader.ingest_calls == []

        allowed = client.post(
            "/api/data/ingest",
            json={
                "source": "databento",
                "roots": ["ES", "NQ"],
                "start": "2026-01-01",
                "end": "2026-01-10",
                "allow_over_budget": True,
                "confirm": True,
            },
        )
        assert allowed.status_code == 200, allowed.text
        body = allowed.json()
        assert body["source"] == "databento"
        assert body["roots_ingested"] == ["ES", "NQ"]
        assert body["bars_written"] > 0
        assert body["cost_estimate"]["estimated_cost_usd"] == pytest.approx(5.0)
        assert loader.ingest_calls and loader.ingest_calls[0]["allow_over_budget"] is True
        assert any("override" in warning for warning in body["warnings"])
        # The override ingest recorded provenance for the written roots.
        inventory = service.repository.inventory("ohlcv-1d")
        by_root = {row.root: row for row in inventory}
        assert by_root["ES"].source == "databento"
        assert by_root["ES"].cached_bars > 0


def test_api_ingest_ibkr_unavailable_and_success(tmp_path: Path) -> None:
    fetcher = _FakeIBKRFetcher()
    with _data_app(tmp_path / "down", fetcher=fetcher, ibkr_available=False) as (client, _service):
        refused = client.post(
            "/api/data/ingest",
            json={
                "source": "ibkr",
                "roots": ["ES"],
                "start": "2026-01-01",
                "end": "2026-01-10",
            },
        )
        assert refused.status_code == 503, refused.text
        detail = refused.json()["detail"]
        assert detail["error"] == "ibkr_unreachable"
        assert (
            f"{_settings(tmp_path).ibkr_host}:{_settings(tmp_path).ibkr_port}" in detail["message"]
        )
        assert fetcher.fetch_calls == []

    with _data_app(tmp_path / "up", fetcher=fetcher, ibkr_available=True) as (client, service):
        response = client.post(
            "/api/data/ingest",
            json={
                "source": "ibkr",
                "roots": ["ES", "NQ"],
                "start": "2026-01-01",
                "end": "2026-01-10",
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["source"] == "ibkr"
        assert body["cost_estimate"] is None
        assert body["bars_written"] > 0
        assert body["continuous_series"]
        assert fetcher.fetch_calls
        by_root = {row.root: row for row in service.repository.inventory("ohlcv-1d")}
        assert by_root["ES"].source == "ibkr"
        assert by_root["NQ"].source == "ibkr"


def test_api_data_ingest_validation_errors(tmp_path: Path) -> None:
    with _data_app(tmp_path) as (client, _service):
        for payload in (
            {"source": "ibkr", "roots": [], "start": "2026-01-01", "end": "2026-01-10"},
            {"source": "ibkr", "roots": ["NOPE"], "start": "2026-01-01", "end": "2026-01-10"},
            {"source": "ibkr", "roots": ["ES"], "start": "2026-01-10", "end": "2026-01-01"},
        ):
            response = client.post("/api/data/ingest", json=payload)
            assert response.status_code == 422, (payload, response.text)


# ---------------------------------------------------------------------------
# 4. cached_real vs synthetic backtests + Optuna (task 64)
# ---------------------------------------------------------------------------


@contextmanager
def _strategy_app(tmp_path: Path, *, seed: bool = True) -> Any:
    configs = _strategy_with_roots(tmp_path)
    settings = _settings(tmp_path, configs_dir=configs)
    if seed:
        _seed_cache(settings, periods=400)
    app = create_app(settings, clock=lambda: FIXED_CLOCK)
    with TestClient(app) as client:
        yield client, settings


def _backtest_payload(data_mode: str) -> dict[str, object]:
    return {
        "strategy_id": "tsmom_vol_target",
        "start": (FIXED_CLOCK - timedelta(days=300)).date().isoformat(),
        "end": (FIXED_CLOCK - timedelta(days=30)).date().isoformat(),
        "initial_equity": 1_000_000.0,
        "walk_forward": False,
        "data_mode": data_mode,
    }


def test_backtest_cached_real_uses_real_parquet(tmp_path: Path) -> None:
    with _strategy_app(tmp_path, seed=True) as (client, _settings_obj):
        response = client.post("/api/backtests", json=_backtest_payload("cached_real"))
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["data_origin"] == "real_parquet"
        assert body["data_mode"] == "cached_real"
        assert body["cached_bars"] is not None
        assert body["cached_bars"]["roots"] == list(CACHED_ROOTS)
        assert body["cached_bars"]["bar_count"] > 0
        assert body["cached_bars"]["schema"] == "ohlcv-1d"  # JSON alias of schema_name
        assert body["status"] == "completed"
        assert body["metrics"] is not None
        # List and detail expose the same provenance badges immediately.
        listed = {row["id"]: row for row in client.get("/api/backtests").json()["runs"]}
        assert listed[body["id"]]["data_origin"] == "real_parquet"
        detail = client.get(f"/api/backtests/{body['id']}").json()
        assert detail["data_mode"] == "cached_real"
        assert detail["cached_bars"]["bar_count"] == body["cached_bars"]["bar_count"]


def test_backtest_auto_falls_back_to_synthetic_when_cache_empty(tmp_path: Path) -> None:
    with _strategy_app(tmp_path, seed=False) as (client, _settings_obj):
        response = client.post("/api/backtests", json=_backtest_payload("auto"))
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["data_origin"] == "synthetic_fixture"
        assert body["data_mode"] == "auto"
        assert body["cached_bars"] is None
        assert body["status"] in {"completed", "degraded"}


def test_backtest_cached_real_empty_cache_is_422(tmp_path: Path) -> None:
    with _strategy_app(tmp_path, seed=False) as (client, _settings_obj):
        response = client.post("/api/backtests", json=_backtest_payload("cached_real"))
        assert response.status_code == 422, response.text
        detail = response.json()["detail"]
        assert "no sufficient cached real" in detail
        assert "data_mode='auto'" in detail


def test_backtest_synthetic_offline_never_reads_cache(tmp_path: Path) -> None:
    with _strategy_app(tmp_path, seed=True) as (client, _settings_obj):
        response = client.post("/api/backtests", json=_backtest_payload("synthetic_offline"))
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["data_origin"] == "synthetic_fixture"
        assert body["data_mode"] == "synthetic_offline"
        assert body["cached_bars"] is None


def test_optimize_cached_real_metadata_and_empty_cache_errors(tmp_path: Path) -> None:
    with _strategy_app(tmp_path, seed=True) as (client, _settings_obj):
        response = client.post(
            "/api/optimize",
            json={
                "base_strategy_id": "tsmom_vol_target",
                "n_trials": 2,
                "wf_splits": 3,
                "data_mode": "cached_real",
            },
        )
        assert response.status_code == 200, response.text
        study = response.json()
        assert study["data_origin"] == "real_parquet"
        assert study["data_mode"] == "cached_real"
        assert study["cached_bars"] is not None
        assert study["cached_bars"]["bar_count"] > 0
        assert study["best_trial_number"] is not None or study["completed_trials"] > 0
        listed = {row["study_id"]: row for row in client.get("/api/optimize").json()}
        assert listed[study["study_id"]]["data_origin"] == "real_parquet"
        detail = client.get(f"/api/optimize/{study['study_id']}").json()
        assert detail["data_mode"] == "cached_real"

    with _strategy_app(tmp_path / "empty", seed=False) as (client, _settings_obj):
        auto = client.post(
            "/api/optimize",
            json={
                "base_strategy_id": "tsmom_vol_target",
                "n_trials": 2,
                "wf_splits": 3,
                "data_mode": "auto",
            },
        )
        assert auto.status_code == 200, auto.text
        assert auto.json()["data_origin"] == "synthetic_fixture"
        assert auto.json()["cached_bars"] is None

        cached_empty = client.post(
            "/api/optimize",
            json={
                "base_strategy_id": "tsmom_vol_target",
                "n_trials": 2,
                "wf_splits": 3,
                "data_mode": "cached_real",
            },
        )
        assert cached_empty.status_code == 422, cached_empty.text
        assert "no sufficient cached real" in cached_empty.json()["detail"]


# ---------------------------------------------------------------------------
# 5. Frontend contract sanity (task 65 consumes these additive fields)
# ---------------------------------------------------------------------------


def test_contract_fields_for_frontend_badges(tmp_path: Path) -> None:
    with _strategy_app(tmp_path, seed=True) as (client, _settings_obj):
        body = client.post("/api/backtests", json=_backtest_payload("cached_real")).json()
        # Backtest Lab badges/toggles read exactly these additive fields.
        assert body["data_origin"] in {"real_parquet", "synthetic_fixture"}
        assert body["data_mode"] in {"auto", "cached_real", "synthetic_offline"}
        assert set(body["cached_bars"]) == {
            "roots",
            "bar_count",
            "start_date",
            "end_date",
            "schema",
            "timeframe",
        }
        assert body["timeframe"] == "1d"

    with _data_app(tmp_path / "data") as (client, _service):
        inventory = client.get("/api/data/inventory").json()
        assert inventory["totals"]["cached_roots"] >= 0
        assert inventory["totals"]["total_roots"] >= 45
        assert isinstance(inventory["databento_configured"], bool)
        assert isinstance(inventory["ibkr_available"], bool)
        estimate = client.post(
            "/api/data/estimate-cost",
            json={
                "roots": ["ES"],
                "start": "2026-01-01",
                "end": "2026-01-10",
                "source": "ibkr",
            },
        ).json()
        assert set(estimate) >= {
            "estimated_cost_usd",
            "billable_size_bytes",
            "billable_size_mb",
            "source",
            "currency",
        }
