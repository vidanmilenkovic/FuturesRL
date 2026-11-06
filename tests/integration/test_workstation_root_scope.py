"""Workstation UX: 45-root scope filters for term-structure/pipeline endpoints."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from configs.settings import Settings
from fastapi.testclient import TestClient
from src.api import fixtures
from src.api.main import create_app
from src.data.storage import ParquetRepository

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXED = datetime(2026, 11, 6, 12, 0, tzinfo=UTC)
TELEMETRY_INTERVAL = 0.02


def _settings(tmp_path: Path) -> Settings:
    configs = tmp_path / "configs"
    strategies = configs / "strategies"
    strategies.mkdir(parents=True, exist_ok=True)
    for source in sorted((REPO_ROOT / "configs" / "strategies").glob("*.yaml")):
        shutil.copyfile(source, strategies / source.name)
    return Settings(
        environment="backtest",
        configs_dir=configs,
        data_dir=tmp_path / "data",
        parquet_dir=tmp_path / "data" / "parquet",
        duckdb_path=tmp_path / "data" / "api.duckdb",
        report_dir=tmp_path / "artifacts",
        instruments_path=REPO_ROOT / "configs" / "instruments.yaml",
        telemetry_interval_seconds=TELEMETRY_INTERVAL,
        api_cors_origins=("http://localhost:3080",),
    )


@pytest.fixture()
def client(tmp_path: Path) -> Iterator[TestClient]:
    with TestClient(create_app(_settings(tmp_path), clock=lambda: FIXED)) as test_client:
        yield test_client


def test_default_term_and_pipeline_payload_unchanged(client: TestClient) -> None:
    term = client.get("/api/term-structure")
    assert term.status_code == 200, term.text
    body = term.json()
    assert body["scope"] == "strategy"
    assert body["selected_roots"] == []
    assert body["sector"] is None
    assert body["curves"] and body["roll_calendar"] is not None

    pipeline = client.get("/api/pipeline/latest")
    assert pipeline.status_code == 200, pipeline.text
    pipe_body = pipeline.json()
    assert pipe_body["scope"] == "strategy"
    assert pipe_body["selected_roots"] == []
    assert pipe_body["stages"] and pipe_body["instruments"]


def test_term_scope_all_covers_instrument_master(client: TestClient) -> None:
    response = client.get("/api/term-structure?scope=all")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["scope"] == "all"
    assert len(body["selected_roots"]) >= 45
    assert body["data_origin"] == "deterministic_instrument_master"
    assert body["curves"]
    assert response.headers["x-data-origin"] == "deterministic_instrument_master"
    assert response.headers["x-scope-roots"]


def test_term_and_pipeline_explicit_roots_and_sector(client: TestClient) -> None:
    explicit = client.get("/api/term-structure?roots=ES,NQ,CL")
    assert explicit.status_code == 200, explicit.text
    explicit_body = explicit.json()
    assert explicit_body["selected_roots"] == ["ES", "NQ", "CL"]
    assert {curve["root"] for curve in explicit_body["curves"]} == {"ES", "NQ", "CL"}

    pipeline = client.get("/api/pipeline/latest?roots=ES,NQ,CL")
    assert pipeline.status_code == 200, pipeline.text
    pipe_body = pipeline.json()
    assert pipe_body["selected_roots"] == ["ES", "NQ", "CL"]
    assert {row["instrument_id"] for row in pipe_body["instruments"]} == {
        "ES",
        "NQ",
        "CL",
    }

    sector = client.get("/api/term-structure?sector=energy")
    assert sector.status_code == 200, sector.text
    sector_body = sector.json()
    assert sector_body["sector"] == "energy"
    assert sector_body["selected_roots"]
    assert all(
        client.app.state.container.instrument_master.get(root).sector in {"energy", "energies"}
        for root in sector_body["selected_roots"]
    )


def test_scope_validation_errors(client: TestClient) -> None:
    for url in (
        "/api/term-structure?roots=NOPE",
        "/api/term-structure?sector=nope",
        "/api/term-structure?scope=bogus",
        "/api/pipeline/latest?roots=NOPE",
        "/api/pipeline/latest?sector=nope",
    ):
        response = client.get(url)
        assert response.status_code == 422, (url, response.text)


def test_cached_scope_uses_real_parquet_roots(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    master = settings.instruments
    timestamps = fixtures.weekdays_between(
        (FIXED.date()).fromordinal(FIXED.date().toordinal() - 400), FIXED.date()
    )
    frame = fixtures._build_bars(master, ("ES", "NQ"), timestamps, seed=7)
    ParquetRepository(settings).write_bars(frame, schema="ohlcv-1d")
    with TestClient(create_app(settings, clock=lambda: FIXED)) as client:
        response = client.get("/api/term-structure?scope=cached")
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["scope"] == "cached"
        assert set(body["selected_roots"]) == {"ES", "NQ"}
        assert body["data_origin"] == "cached_parquet"

        pipeline = client.get("/api/pipeline/latest?scope=cached")
        assert pipeline.status_code == 200, pipeline.text
        pipe_body = pipeline.json()
        assert set(pipe_body["selected_roots"]) == {"ES", "NQ"}
