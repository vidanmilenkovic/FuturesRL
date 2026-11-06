"""Workstation UX tests: LLM provider status/ping + 45-root Risk & LLM scope.

Offline and deterministic: fake LLM clients for every transport outcome, a tmp
Parquet cache for ``scope=cached``, and no network/TWS/Databento access anywhere.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest
from configs.settings import Settings
from fastapi.testclient import TestClient
from src.api import fixtures
from src.api.llm_status import LLM_PING_PROMPT, LLMStatusService
from src.api.main import create_app
from src.data import schema as data_schema
from src.data.storage import ParquetRepository
from src.risk.llm_risk_overlay import LLMClientError

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXED_TIMESTAMP = datetime(2026, 11, 6, 17, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Fakes / helpers
# ---------------------------------------------------------------------------


class FakeClock:
    def __init__(self, now: datetime = FIXED_TIMESTAMP) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


class CountingMonotonic:
    def __init__(self) -> None:
        self.values = [0.0, 0.0, 0.0, 0.0]
        self.index = 0

    def __call__(self) -> float:
        value = self.values[min(self.index, len(self.values) - 1)]
        self.index += 1
        return value


class FakeLLMClient:
    """Deterministic AsyncLLMClient stand-in."""

    def __init__(
        self,
        *,
        response: str = '{"ok": true}',
        error: BaseException | None = None,
        delay: float = 0.0,
    ) -> None:
        self.response = response
        self.error = error
        self.delay = delay
        self.calls: list[dict[str, object]] = []

    async def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        self.calls.append(
            {"prompt": prompt, "system_prompt": system_prompt, "temperature": temperature}
        )
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return self.response


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    base: dict[str, object] = {
        "environment": "backtest",
        "data_dir": tmp_path / "data",
        "parquet_dir": tmp_path / "data" / "parquet",
        "duckdb_path": tmp_path / "data" / "llm.duckdb",
        "report_dir": tmp_path / "artifacts",
        "instruments_path": REPO_ROOT / "configs" / "instruments.yaml",
        "telemetry_interval_seconds": 3600.0,
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[arg-type]


def _status_service(
    settings: Settings,
    *,
    client: FakeLLMClient | None = None,
    factory_calls: list[int] | None = None,
    clock: FakeClock | None = None,
    monotonic: CountingMonotonic | None = None,
    timeout: float = 5.0,
    cache_entries: int = 0,
) -> LLMStatusService:
    calls = factory_calls if factory_calls is not None else []

    def factory() -> FakeLLMClient:
        calls.append(1)
        assert client is not None
        return client

    return LLMStatusService(
        settings=settings,
        clock=(clock or FakeClock()),
        monotonic=monotonic or CountingMonotonic(),
        client_factory=factory,
        ping_timeout_seconds=timeout,
        cache_entries_provider=lambda: cache_entries,
    )


@pytest.fixture()
def client(tmp_path: Path) -> Iterator[TestClient]:
    with TestClient(create_app(_settings(tmp_path), clock=lambda: FIXED_TIMESTAMP)) as test_client:
        yield test_client


def _seed_cache(settings: Settings, roots: tuple[str, ...] = ("ES", "NQ", "CL")) -> None:
    market = fixtures.build_offline_market(
        settings.instruments, clock=lambda: FIXED_TIMESTAMP, days=320
    )
    frame = market.bars.filter(pl.col(data_schema.ROOT_SYMBOL_COLUMN).is_in(list(roots)))
    ParquetRepository(settings).write_bars(frame, mode="overwrite")


# ---------------------------------------------------------------------------
# A) LLM status / ping
# ---------------------------------------------------------------------------


def test_llm_status_offline_stub(client: TestClient) -> None:
    response = client.get("/api/risk/llm-status")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["provider"] == "offline"
    assert body["model"]
    assert body["api_key_configured"] is False
    assert body["mode"] == "offline_stub"
    assert body["status"] == "offline_stub"
    assert body["last_latency_ms"] is None
    assert body["last_checked_at"] is None
    assert body["cache_entries"] == 0
    assert body["error_detail"] is None


def test_health_includes_llm_status_object(client: TestClient) -> None:
    health = client.get("/health").json()
    assert health["llm_status"]["provider"] == "offline"
    assert health["llm_status"]["mode"] == "offline_stub"
    assert health["llm_status"]["status"] == "offline_stub"
    assert set(health["llm_status"]) >= {
        "provider",
        "model",
        "api_key_configured",
        "mode",
        "status",
        "last_latency_ms",
        "last_checked_at",
        "cache_entries",
        "error_detail",
    }


async def test_offline_ping_performs_no_network_io(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    factory_calls: list[int] = []
    service = _status_service(settings, factory_calls=factory_calls)
    clock = service._clock
    checked = await service.ping()
    assert checked.status == "offline_stub"
    assert checked.mode == "offline_stub"
    assert checked.last_checked_at is not None
    assert checked.last_latency_ms is None
    assert factory_calls == [], "offline status must never build a transport"
    assert clock is not None


async def test_live_provider_missing_key_is_offline_stub(tmp_path: Path) -> None:
    settings = _settings(tmp_path).model_copy(
        update={"llm_provider": "openai", "llm_api_key": None}
    )
    factory_calls: list[int] = []
    service = _status_service(settings, factory_calls=factory_calls)
    status = service.status()
    assert status.mode == "offline_stub"
    assert status.status == "offline_stub"
    assert status.api_key_configured is False
    assert status.error_detail is not None and "API key" in status.error_detail
    pinged = await service.ping()
    assert pinged.status == "offline_stub"
    assert pinged.error_detail is not None
    assert "API key not configured" in pinged.error_detail
    assert factory_calls == []


async def test_live_ping_success_reports_online_and_latency(tmp_path: Path) -> None:
    settings = _settings(tmp_path).model_copy(update={"llm_provider": "vllm", "llm_api_key": None})
    fake = FakeLLMClient()
    monotonic = CountingMonotonic()
    monotonic.values = [0.0, 0.123]
    service = _status_service(settings, client=fake, monotonic=monotonic)
    status = service.status()
    assert status.mode == "live_api"
    assert status.status == "unreachable"
    assert status.error_detail is not None

    pinged = await service.ping()

    assert pinged.status == "online"
    assert pinged.mode == "live_api"
    assert pinged.last_latency_ms == pytest.approx(123.0, abs=1e-3)
    assert pinged.last_checked_at is not None
    assert pinged.error_detail is None
    assert len(fake.calls) == 1
    assert fake.calls[0]["prompt"] == LLM_PING_PROMPT
    assert fake.calls[0]["temperature"] == 0.0


async def test_auth_error_classification(tmp_path: Path) -> None:
    settings = _settings(tmp_path).model_copy(update={"llm_provider": "vllm", "llm_api_key": None})
    fake = FakeLLMClient(error=LLMClientError("provider returned 401 Unauthorized"))
    service = _status_service(settings, client=fake)
    pinged = await service.ping()
    assert pinged.status == "auth_error"
    assert pinged.last_latency_ms is None
    assert pinged.error_detail is not None
    assert "401" in pinged.error_detail


async def test_unreachable_on_timeout_is_bounded(tmp_path: Path) -> None:
    settings = _settings(tmp_path).model_copy(update={"llm_provider": "vllm", "llm_api_key": None})
    fake = FakeLLMClient(delay=0.5)
    service = _status_service(settings, client=fake, timeout=0.01)
    pinged = await asyncio.wait_for(service.ping(), timeout=1.0)
    assert pinged.status == "unreachable"
    assert pinged.error_detail is not None
    assert "timed out" in pinged.error_detail.lower()


async def test_unreachable_on_transport_error_and_empty_completion(tmp_path: Path) -> None:
    settings = _settings(tmp_path).model_copy(update={"llm_provider": "vllm", "llm_api_key": None})
    broken = _status_service(
        settings, client=FakeLLMClient(error=RuntimeError("connection refused"))
    )
    status = await broken.ping()
    assert status.status == "unreachable"
    assert status.error_detail is not None and "RuntimeError" in status.error_detail

    empty = _status_service(settings, client=FakeLLMClient(response="   "))
    status = await empty.ping()
    assert status.status == "unreachable"
    assert status.error_detail is not None and "empty completion" in status.error_detail


def test_cache_entries_provider_is_honest(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    def provider() -> int:
        return 7

    service = LLMStatusService(
        settings=settings,
        clock=lambda: FIXED_TIMESTAMP,
        cache_entries_provider=provider,
    )
    assert service.status().cache_entries == 7

    def failing_provider() -> int:
        raise RuntimeError("cache unavailable")

    broken = LLMStatusService(
        settings=settings,
        clock=lambda: FIXED_TIMESTAMP,
        cache_entries_provider=failing_provider,
    )
    assert broken.status().cache_entries == 0


async def test_ping_route_uses_container_status_service(client: TestClient, tmp_path: Path) -> None:
    fake = FakeLLMClient()
    settings = _settings(tmp_path).model_copy(update={"llm_provider": "vllm", "llm_api_key": None})
    service = _status_service(settings, client=fake, cache_entries=3)
    client.app.state.container.llm_status = service

    before = client.get("/api/risk/llm-status").json()
    assert before["cache_entries"] == 3
    assert before["mode"] == "live_api"

    pinged = client.post("/api/risk/llm-ping")
    assert pinged.status_code == 200, pinged.text
    body = pinged.json()
    assert body["status"] == "online"
    assert body["last_latency_ms"] is not None
    assert len(fake.calls) == 1
    assert client.get("/health").json()["llm_status"]["status"] == "online"


# ---------------------------------------------------------------------------
# B) 45-root Risk & LLM scope
# ---------------------------------------------------------------------------


def test_default_scope_strategy_is_backward_compatible(client: TestClient) -> None:
    container = client.app.state.container
    state = container.market.latest_state()
    body = client.get("/api/risk/state").json()
    legacy = client.get("/api/risk/state?scope=strategy").json()

    assert body["scope"] == "strategy"
    assert body["sector"] is None
    assert body["selected_roots"] == sorted(state.instruments)
    assert body["derived_roots"] == []
    assert body["assessment_source"] == "fixture"
    assert [row["instrument_id"] for row in body["assessments"]] == sorted(state.instruments)
    assert legacy["assessments"] == body["assessments"]
    assert legacy["macro_events"] == body["macro_events"]


def test_scope_all_covers_all_45_roots_with_derived_provenance(client: TestClient) -> None:
    container = client.app.state.container
    master_roots = list(container.instrument_master.roots())
    body = client.get("/api/risk/state?scope=all").json()

    assert len(master_roots) == 45
    assert body["scope"] == "all"
    assert body["selected_roots"] == sorted(master_roots)
    assert len(body["assessments"]) == 45
    assert {row["instrument_id"] for row in body["assessments"]} == set(master_roots)
    state_roots = set(container.market.latest_state().instruments)
    derived = set(body["derived_roots"])
    assert derived
    assert derived == set(master_roots) - state_roots
    assert body["assessment_source"] in {"mixed", "derived"}
    # Derived roots are explicitly flagged, never mislabelled as live/cache data.
    for row in body["assessments"]:
        if row["instrument_id"] in derived:
            assert row["action"] in {"pass", "scale", "liquidate", "neutralized"}


def test_sector_filter_and_scope(client: TestClient) -> None:
    body = client.get("/api/risk/state?scope=all&sector=energy").json()
    expected = sorted(
        root
        for root in client.app.state.container.instrument_master.roots()
        if client.app.state.container.instrument_master.get(root).sector == "energy"
    )
    assert body["sector"] == "energy"
    assert body["selected_roots"] == expected
    assert {row["instrument_id"] for row in body["assessments"]} == set(expected)


def test_explicit_roots_filter_overrides_scope(client: TestClient) -> None:
    body = client.get("/api/risk/state?scope=all&roots=es,CL,ZN").json()
    assert body["selected_roots"] == ["CL", "ES", "ZN"]
    assert {row["instrument_id"] for row in body["assessments"]} == {"CL", "ES", "ZN"}
    assert body["derived_roots"] == []


def test_cached_scope_uses_cached_parquet_roots(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _seed_cache(settings, roots=("ES", "NQ", "CL"))
    with TestClient(create_app(settings, clock=lambda: FIXED_TIMESTAMP)) as local:
        body = local.get("/api/risk/state?scope=cached").json()
        assert body["scope"] == "cached"
        assert body["selected_roots"] == ["CL", "ES", "NQ"]
        assert {row["instrument_id"] for row in body["assessments"]} == {"CL", "ES", "NQ"}


def test_cached_scope_without_cache_is_honestly_empty(
    client: TestClient,
) -> None:
    body = client.get("/api/risk/state?scope=cached").json()
    assert body["scope"] == "cached"
    assert body["selected_roots"] == []
    assert body["assessments"] == []


def test_invalid_scope_filters_return_422(client: TestClient) -> None:
    assert client.get("/api/risk/state?roots=ES,ZZ").status_code == 422
    unknown_sector = client.get("/api/risk/state?scope=all&sector=bogus")
    assert unknown_sector.status_code == 422
    # Invalid scope value is rejected by FastAPI's Literal validation.
    assert client.get("/api/risk/state?scope=nope").status_code == 422


def test_llm_status_cache_entries_after_risk_state(client: TestClient) -> None:
    state = client.get("/api/risk/state").json()
    assert state["assessments"]
    status = client.get("/api/risk/llm-status").json()
    assert status["cache_entries"] == len(state["assessments"])
