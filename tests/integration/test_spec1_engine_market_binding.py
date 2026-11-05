"""Spec-1 integration tests (task-71): EngineMarketService live NLV + cached Parquet binding.

Offline and deterministic: a fake IBKR connection manager provides a real-style
account NLV, and tmp Parquet repositories provide cached bars. No TWS, Databento
or network access anywhere in this module.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import src.execution.ibkr_client as ibkr_client_module
from configs.settings import Settings
from fastapi.testclient import TestClient
from src.api import fixtures
from src.api.main import create_app
from src.api.services import INITIAL_EQUITY, EngineMarketService, InMemoryStrategyCatalog
from src.core.types import AccountSnapshot, InstrumentId, Position
from src.data import schema as data_schema
from src.data.storage import ParquetRepository
from src.execution.ibkr_client import MarginImpact

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXED_TIMESTAMP = datetime(2026, 11, 5, 17, 0, tzinfo=UTC)
LIVE_NLV = 1_000_000.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _settings(tmp_path: Path, *, api_enable_ibkr: bool = False) -> Settings:
    return Settings(
        _env_file=None,
        environment="backtest",
        data_dir=tmp_path / "data",
        parquet_dir=tmp_path / "data" / "parquet",
        duckdb_path=tmp_path / "data" / "spec1.duckdb",
        report_dir=tmp_path / "artifacts",
        instruments_path=REPO_ROOT / "configs" / "instruments.yaml",
        telemetry_interval_seconds=3600.0,
        api_enable_ibkr=api_enable_ibkr,
        ibkr_connect_timeout_seconds=0.05,
        ibkr_reconnect_max_attempts=1,
        ibkr_heartbeat_interval_seconds=0.05,
    )


def _market(settings: Settings, *, days: int = 320) -> fixtures.OfflineMarket:
    return fixtures.build_offline_market(
        settings.instruments, clock=lambda: FIXED_TIMESTAMP, days=days
    )


def _catalog(settings: Settings) -> InMemoryStrategyCatalog:
    return InMemoryStrategyCatalog(settings=settings, clock=lambda: FIXED_TIMESTAMP)


def _service(
    settings: Settings,
    *,
    market: fixtures.OfflineMarket | None = None,
    equity_provider: Any = None,
    account_provider: Any = None,
) -> EngineMarketService:
    return EngineMarketService(
        market=market or _market(settings),
        settings=settings,
        catalog=_catalog(settings),
        clock=lambda: FIXED_TIMESTAMP,
        equity_provider=equity_provider,
        account_provider=account_provider,
    )


def _seed_cache(
    settings: Settings,
    *,
    roots: tuple[str, ...] = ("ES", "NQ", "CL"),
    days: int = 320,
    close_scale: float = 1.0,
) -> pl.DataFrame:
    """Write deterministic cached real bars for ``roots`` into the tmp Parquet repo."""
    market = _market(settings, days=days)
    frame = market.bars.filter(pl.col(data_schema.ROOT_SYMBOL_COLUMN).is_in(list(roots)))
    if close_scale != 1.0:
        price_columns = (
            data_schema.OPEN_COLUMN,
            data_schema.HIGH_COLUMN,
            data_schema.LOW_COLUMN,
            data_schema.CLOSE_COLUMN,
            data_schema.SETTLEMENT_COLUMN,
        )
        frame = frame.with_columns(
            *[(pl.col(column) * close_scale).alias(column) for column in price_columns]
        )
    repository = ParquetRepository(settings)
    repository.write_bars(frame, mode="overwrite")
    return frame


def _seed_multi_expiry_cache(settings: Settings, root: str = "ES") -> None:
    """Write a cached root with two distinct expiry closes (real forward curve)."""
    market = _market(settings, days=320)
    base = market.bars.filter(pl.col(data_schema.ROOT_SYMBOL_COLUMN) == root)

    def _contract_rows(expiration: date, price: float, instrument_id: str) -> pl.DataFrame:
        price_columns = (
            data_schema.OPEN_COLUMN,
            data_schema.HIGH_COLUMN,
            data_schema.LOW_COLUMN,
            data_schema.CLOSE_COLUMN,
            data_schema.SETTLEMENT_COLUMN,
        )
        return base.with_columns(
            pl.lit(expiration).alias(data_schema.EXPIRATION_COLUMN),
            *[pl.lit(price).alias(column) for column in price_columns],
            pl.lit(instrument_id).alias(data_schema.INSTRUMENT_ID_COLUMN),
        )

    first = _contract_rows(date(2027, 3, 19), 100.0, f"{root}_20270319")
    second = _contract_rows(date(2027, 6, 18), 101.0, f"{root}_20270618")
    repository = ParquetRepository(settings)
    repository.write_bars(pl.concat([first, second], how="vertical_relaxed"), mode="overwrite")


# ---------------------------------------------------------------------------
# A) live NLV split-brain
# ---------------------------------------------------------------------------


def test_latest_state_uses_injected_live_nlv(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    service = _service(settings, equity_provider=lambda: LIVE_NLV)

    state = service.latest_state()

    assert state.equity == pytest.approx(LIVE_NLV)
    assert service.equity_source == "live-account"
    assert service.data_backing["equity_source"] == "live-account"
    assert service.data_backing["degraded_reason"] is None


def test_pipeline_values_account_with_live_nlv(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    service = _service(settings, equity_provider=lambda: LIVE_NLV)

    response = service.pipeline()

    assert response.timestamp == _market(settings).as_of
    diagnostics = {item.stage: item for item in response.diagnostics}
    assert diagnostics["Equity source"].status == "ok"
    assert diagnostics["Equity source"].value == "1,000,000.00"
    assert response.instruments
    for row in response.instruments:
        assert row.notional == pytest.approx(row.weight_final * LIVE_NLV, abs=0.01)
    margin = diagnostics["Margin feasibility"]
    assert "1,000,000" in margin.detail
    assert response.data_source == "engine"


def test_no_provider_keeps_old_fixture_equity(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    service = _service(settings)

    state = service.latest_state()
    response = service.pipeline()

    assert state.equity == pytest.approx(INITIAL_EQUITY)
    assert service.equity_source == "fixture"
    diagnostics = {item.stage: item for item in response.diagnostics}
    assert diagnostics["Equity source"].status == "warn"
    assert diagnostics["Equity source"].value == f"{INITIAL_EQUITY:,.2f}"


def test_account_provider_object_nlv_and_invalid_values_fall_back(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    account = dataclasses.replace(
        AccountSnapshot(
            timestamp=FIXED_TIMESTAMP,
            net_liquidation=1_234_567.0,
            cash_balance=1_234_567.0,
            buying_power=2_000_000.0,
            initial_margin_used=0.0,
            maintenance_margin_used=0.0,
            excess_liquidity=1_234_567.0,
        )
    )
    from_account = _service(settings, account_provider=lambda: account)
    assert from_account.latest_state().equity == pytest.approx(1_234_567.0)

    for bad_provider in (
        lambda: None,
        lambda: -5.0,
        lambda: float("nan"),
        lambda: (_ for _ in ()).throw(RuntimeError("provider down")),
    ):
        service = _service(settings, equity_provider=bad_provider)
        assert service.latest_state().equity == pytest.approx(INITIAL_EQUITY)
        assert service.equity_source == "fixture"


def test_equity_change_updates_cached_state_without_rebuild(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    values = iter([1_000_000.0, 1_050_000.0])
    service = _service(settings, equity_provider=lambda: next(values, 1_050_000.0))

    first = service.latest_state()
    second = service.latest_state()

    assert first.equity == pytest.approx(1_000_000.0)
    assert second.equity == pytest.approx(1_050_000.0)
    # Same immutable state except for the equity field (fast replace, no rebuild).
    assert second.instruments == first.instruments


# ---------------------------------------------------------------------------
# B) cached Parquet first for state / pipeline / term structure
# ---------------------------------------------------------------------------


def test_cached_parquet_backing_used_for_state_and_pipeline(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    cached = _seed_cache(settings)
    service = _service(settings)

    assert service.data_origin == "cached_parquet"
    backing = service.data_backing
    assert backing["partial"] is True
    assert backing["cached_roots"] == ["CL", "ES", "NQ"]
    assert backing["fallback_roots"], "partial cache must fall back to fixture roots"
    state = service.latest_state()
    assert {"ES", "NQ", "CL"} <= set(state.instruments)
    assert len(backing["roots"]) > 3

    response = service.pipeline()
    diagnostics = {item.stage: item for item in response.diagnostics}
    assert diagnostics["Data backing"].value == "cached_parquet"
    assert diagnostics["Data backing"].status == "info"
    assert "fixture fallback" in diagnostics["Data backing"].detail
    cached_times = set(cached[data_schema.TIMESTAMP_COLUMN].to_list())
    assert state.timestamp in cached_times
    assert state.equity == pytest.approx(INITIAL_EQUITY)


def test_cached_term_structure_uses_cached_prices_and_real_curve(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _seed_cache(settings, roots=("ES",), close_scale=1.10)
    service = _service(settings)

    response = service.term_structure()

    curves = {curve.root: curve for curve in response.curves}
    assert "ES" in curves
    assert curves["ES"].contracts
    cached_front = curves["ES"].contracts[0]
    fixture_front = next(
        curve.contracts[0]
        for curve in fixtures.term_structure(_market(settings)).curves
        if curve.root == "ES"
    )
    # Cached close is ~10% above the synthetic fixture price: the curve is cache-backed.
    assert cached_front.price == pytest.approx(fixture_front.price * 1.10, rel=5e-3)
    assert response.as_of == service.latest_state().timestamp


def test_cached_multi_expiry_bars_build_real_forward_curve(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _seed_multi_expiry_cache(settings, "ES")
    service = _service(settings)

    response = service.term_structure()

    curve = next(curve for curve in response.curves if curve.root == "ES")
    assert [contract.price for contract in curve.contracts[:2]] == [
        pytest.approx(100.0),
        pytest.approx(101.0),
    ]
    assert [contract.expiration for contract in curve.contracts[:2]] == [
        date(2027, 3, 19),
        date(2027, 6, 18),
    ]
    assert curve.contracts[0].is_front is True
    assert curve.contracts[1].is_front is False
    assert curve.regime == "contango"
    assert curve.slope == pytest.approx((101.0 - 100.0) / 100.0, rel=1e-9)
    assert curve.roll_yield_annualized < 0.0
    assert service.data_origin == "cached_parquet"


def test_partial_or_short_cache_falls_back_without_crashing(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    # Too-short cached partition: below CACHED_MIN_BARS_PER_ROOT -> fixture fallback.
    market = _market(settings, days=320)
    short_frame = market.bars.filter(pl.col(data_schema.ROOT_SYMBOL_COLUMN) == "ES").head(10)
    ParquetRepository(settings).write_bars(short_frame, mode="overwrite")
    service = _service(settings)

    # A short cached history is tolerated: cached roots bind, other roots fall back
    # to fixtures and the state/term-structure views never crash.
    assert service.data_origin == "cached_parquet"
    assert service.data_backing["partial"] is True
    state = service.latest_state()
    assert state.equity == pytest.approx(INITIAL_EQUITY)
    assert "ES" in state.instruments
    diagnostics = {item.stage: item for item in service.pipeline().diagnostics}
    assert diagnostics["Data backing"].value == "cached_parquet"
    assert service.term_structure().curves


def test_corrupt_cache_partition_degrades_to_fixtures(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    repository = ParquetRepository(settings)
    partition = repository.partition_path("ES", data_schema.DEFAULT_BAR_SCHEMA)
    partition.parent.mkdir(parents=True, exist_ok=True)
    partition.write_bytes(b"this is not a parquet file")
    service = _service(settings)

    state = service.latest_state()

    assert state.equity == pytest.approx(INITIAL_EQUITY)
    assert service.data_origin == "fixture"
    assert service.term_structure().curves


def test_live_nlv_applies_to_cached_parquet_state(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _seed_cache(settings)
    service = _service(settings, equity_provider=lambda: LIVE_NLV)

    state = service.latest_state()
    response = service.pipeline()

    assert service.data_origin == "cached_parquet"
    assert state.equity == pytest.approx(LIVE_NLV)
    assert {item.stage: item for item in response.diagnostics}["Equity source"].value == (
        "1,000,000.00"
    )


def test_refresh_data_picks_up_new_cache(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    service = _service(settings)
    assert service.data_origin == "fixture"

    _seed_cache(settings)
    service.refresh_data()

    assert service.data_origin == "cached_parquet"


# ---------------------------------------------------------------------------
# A) Container / IBKR wiring: pipeline NLV matches the positions view
# ---------------------------------------------------------------------------


class _FakeIBKRManager:
    """Minimal deterministic IBKR manager stand-in for the Container wiring test."""

    def __init__(self, master: Any, *, nlv: float, connected: bool = True) -> None:
        self.instruments = master
        self.is_connected = connected
        self.is_reconnecting = False
        self.last_error: str | None = None
        self.heartbeat_running = False
        self.fail_connect = not connected
        self._fetch_calls = 0
        self._account = AccountSnapshot(
            timestamp=FIXED_TIMESTAMP,
            net_liquidation=nlv,
            cash_balance=nlv,
            buying_power=nlv * 2.0,
            initial_margin_used=0.0,
            maintenance_margin_used=0.0,
            excess_liquidity=nlv,
        )
        self._positions: dict[InstrumentId, Position] = {}

    @property
    def price_provider(self) -> Any:
        return lambda _root: None

    async def what_if_order(self, _order: object) -> MarginImpact:
        return MarginImpact(
            initial_margin_before=0.0,
            maintenance_margin_before=0.0,
            initial_margin_change=0.0,
            maintenance_margin_change=0.0,
            initial_margin_after=0.0,
            maintenance_margin_after=0.0,
        )

    async def connect(self) -> None:
        if self.fail_connect:
            self.last_error = "TimeoutError: TWS unavailable"
            raise ConnectionError(self.last_error)
        self.is_connected = True
        self.last_error = None

    async def ensure_connected(self, *, max_attempts: int = 1) -> bool:
        del max_attempts
        if self.fail_connect:
            self.last_error = "TimeoutError: TWS unavailable"
            return False
        self.is_connected = True
        self.last_error = None
        return True

    async def disconnect(self) -> None:
        self.is_connected = False

    def invalidate(self, reason: str) -> None:
        self.is_connected = False
        self.last_error = reason

    def ensure_heartbeat(self) -> bool:
        return False

    async def fetch_account(self) -> AccountSnapshot:
        if not self.is_connected:
            raise ConnectionError("IBKR connection is not established")
        self._fetch_calls += 1
        return self._account

    async def fetch_positions(self, prices: Any = None) -> dict[InstrumentId, Position]:
        del prices
        if not self.is_connected:
            raise ConnectionError("IBKR connection is not established")
        return dict(self._positions)


def _app_with_fake_ibkr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, nlv: float, connected: bool = True
) -> tuple[TestClient, _FakeIBKRManager]:
    settings = _settings(tmp_path, api_enable_ibkr=True)
    manager = _FakeIBKRManager(settings.instruments, nlv=nlv, connected=connected)
    monkeypatch.setattr(
        ibkr_client_module, "IBKRConnectionManager", lambda *args, **kwargs: manager
    )
    return TestClient(create_app(settings, clock=lambda: FIXED_TIMESTAMP)), manager


def test_container_pipeline_nlv_matches_positions_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _manager = _app_with_fake_ibkr(tmp_path, monkeypatch, nlv=LIVE_NLV)
    with client:
        positions = client.get("/api/positions")
        assert positions.status_code == 200, positions.text
        positions_nlv = positions.json()["net_liquidation"]
        assert positions_nlv == pytest.approx(LIVE_NLV)

        container = client.app.state.container
        state = container.market.latest_state()
        assert state.equity == pytest.approx(positions_nlv)

        pipeline = client.get("/api/pipeline/latest")
        assert pipeline.status_code == 200, pipeline.text
        body = pipeline.json()
        diagnostics = {item["stage"]: item for item in body["diagnostics"]}
        assert diagnostics["Equity source"]["value"] == f"{positions_nlv:,.2f}"
        assert diagnostics["Equity source"]["status"] == "ok"
        for row in body["instruments"]:
            assert row["notional"] == pytest.approx(row["weight_final"] * positions_nlv, abs=0.01)


def test_container_without_ibkr_keeps_fixture_equity(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with TestClient(create_app(settings, clock=lambda: FIXED_TIMESTAMP)) as client:
        container = client.app.state.container
        assert container.market.latest_state().equity == pytest.approx(INITIAL_EQUITY)
        body = client.get("/api/pipeline/latest").json()
        diagnostics = {item["stage"]: item for item in body["diagnostics"]}
        assert diagnostics["Equity source"]["value"] == f"{INITIAL_EQUITY:,.2f}"


def test_ibkr_connected_without_snapshot_does_not_fake_live_equity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A connected adapter with no fetched book must not be labelled as live NLV."""

    class _NoSnapshotManager(_FakeIBKRManager):
        async def fetch_account(self) -> AccountSnapshot:
            raise ConnectionError("account summary unavailable")

    manager = _NoSnapshotManager(_settings(tmp_path).instruments, nlv=1_000_000.0)
    settings = _settings(tmp_path, api_enable_ibkr=True)
    monkeypatch.setattr(
        ibkr_client_module, "IBKRConnectionManager", lambda *args, **kwargs: manager
    )
    with TestClient(create_app(settings, clock=lambda: FIXED_TIMESTAMP)) as client:
        container = client.app.state.container
        state = container.market.latest_state()
        assert state.equity == pytest.approx(INITIAL_EQUITY)
        diagnostics = {
            item["stage"]: item for item in client.get("/api/pipeline/latest").json()["diagnostics"]
        }
        assert diagnostics["Equity source"]["status"] == "warn"
        assert diagnostics["Equity source"]["value"] == f"{INITIAL_EQUITY:,.2f}"
        assert manager.last_error is not None


def test_term_structure_route_exposes_data_origin_header(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _seed_cache(settings)
    with TestClient(create_app(settings, clock=lambda: FIXED_TIMESTAMP)) as client:
        response = client.get("/api/term-structure")
        assert response.status_code == 200
        assert response.headers["X-Data-Origin"] == "cached_parquet"
        assert "X-Data-As-Of" in response.headers
        assert response.headers["X-Data-Cached-Roots"] == "3"
        assert response.json()["curves"]
