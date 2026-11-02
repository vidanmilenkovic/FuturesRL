"""Phase-7 unit tests: async cached LLM risk worker and non-blocking projection.

The synchronous ``LLMRiskOverlay.apply_risk`` must never await LLM I/O. These
tests use deterministic fake clients (configurable latency/failures), a fake wall
clock and a fake sleep, so everything is offline and reproducible.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta

import pytest
from configs.settings import RiskSettings
from pydantic import ValidationError
from src.core.types import MarketState
from src.risk.llm_risk_overlay import (
    AsyncLLMClient,
    AsyncLLMRiskCacheWorker,
    LLMClientError,
    LLMRiskCacheEntry,
    LLMRiskConfig,
    LLMRiskOverlay,
)

from tests.fixtures.market import (
    FIXED_TIMESTAMP,
    make_market_state,
    make_risk_assessment,
    make_weight_vector,
)

#: A benign score at/below the safe baseline: the projection is a pass-through.
BENIGN_SCORE = 1.2


# ---------------------------------------------------------------------------
# Deterministic fakes
# ---------------------------------------------------------------------------


def _record(
    root: str, *, score: float = BENIGN_SCORE, confidence: float = 0.9
) -> dict[str, object]:
    return {
        "instrument_id": root,
        "sentiment_score": 3.0,
        "semantic_risk_score": score,
        "confidence": confidence,
        "reasoning_summary": "deterministic async-cache fixture",
    }


def _json_batch(*records: dict[str, object]) -> str:
    return json.dumps({"assessments": list(records)})


class CountingClient:
    """AsyncLLMClient with optional latency, call counting and fault injection."""

    def __init__(self, *, delay: float = 0.0) -> None:
        self.delay = delay
        self.calls = 0
        self.fail = False
        self._root = "ES"

    async def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        self.calls += 1
        if self.delay > 0.0:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise LLMClientError("simulated provider outage")
        return _json_batch(_record(self._root))


class FakeClock:
    """Mutable deterministic wall clock."""

    def __init__(self, start: datetime = FIXED_TIMESTAMP) -> None:
        self.current = start

    def now(self) -> datetime:
        return self.current

    def advance(self, seconds: float) -> None:
        self.current = self.current + timedelta(seconds=seconds)


def _fake_sleep(clock: FakeClock) -> Callable[[float], Awaitable[None]]:
    async def _sleep(seconds: float) -> None:
        clock.advance(seconds)
        await asyncio.sleep(0)

    return _sleep


def _config(*, ttl: float = 60.0, **overrides: float | int) -> LLMRiskConfig:
    base: dict[str, float | int] = {"sample_count": 1, "cache_ttl_seconds": ttl}
    base.update(overrides)
    return LLMRiskConfig(**base)  # type: ignore[arg-type]


def _overlay(
    client: AsyncLLMClient,
    *,
    ttl: float = 60.0,
    clock: FakeClock | None = None,
) -> LLMRiskOverlay:
    return LLMRiskOverlay(
        client,
        _config(ttl=ttl),
        clock=clock.now if clock is not None else None,
    )


def _state(timestamp: datetime = FIXED_TIMESTAMP) -> MarketState:
    return make_market_state(timestamp=timestamp, equity=1_000_000.0)


async def _wait_until(predicate: Callable[[], bool], *, attempts: int = 4000) -> bool:
    for _ in range(attempts):
        if predicate():
            return True
        await asyncio.sleep(0)
    return False


# ---------------------------------------------------------------------------
# Non-blocking synchronous projection
# ---------------------------------------------------------------------------


async def test_apply_risk_returns_immediately_while_refresh_is_in_flight() -> None:
    client = CountingClient(delay=0.25)
    overlay = _overlay(client)
    state = _state()

    started = time.perf_counter()
    result = overlay.apply_risk(make_weight_vector({"ES": 0.4}), state)
    elapsed = time.perf_counter() - started

    assert elapsed < 0.05, f"apply_risk blocked for {elapsed:.3f}s on slow LLM I/O"
    # Nothing cached yet: existing pass-through behaviour is preserved.
    assert result["ES"] == pytest.approx(0.4)
    assert overlay.pending_refresh_count == 1
    assert overlay.valid_until_utc is None
    assert overlay.staleness_count == 1

    deadline = time.perf_counter() + 5.0
    while overlay.valid_until_utc is None and time.perf_counter() < deadline:
        await asyncio.sleep(0.01)
    assert overlay.valid_until_utc is not None
    assert overlay.refreshed_at_utc is not None
    assert overlay.valid_until_utc - overlay.refreshed_at_utc == timedelta(seconds=60.0)
    assert overlay.latest_assessments["ES"].semantic_risk_score == pytest.approx(BENIGN_SCORE)
    assert overlay.last_refresh == FIXED_TIMESTAMP

    await overlay.drain_pending_refresh(timeout=5.0)
    assert overlay.pending_refresh_count == 0


async def test_stale_last_known_good_cache_is_used_while_refreshing() -> None:
    clock = FakeClock()
    client = CountingClient(delay=0.2)
    overlay = _overlay(client, ttl=0.05, clock=clock)
    overlay.set_cached_assessments(
        [make_risk_assessment("ES", semantic_risk_score=5.0, confidence=0.95)]
    )
    clock.advance(1.0)
    assert not overlay.is_cache_fresh()

    started = time.perf_counter()
    projected = overlay.apply_risk(make_weight_vector({"ES": 0.3}), _state())
    elapsed = time.perf_counter() - started
    assert elapsed < 0.05
    # The stale-but-critical assessment still flattens ES immediately.
    assert projected["ES"] == 0.0
    assert overlay.pending_refresh_count == 1

    await overlay.drain_pending_refresh(timeout=5.0)
    # The fresh benign assessment replaces the stale critical one.
    assert overlay.latest_assessments["ES"].semantic_risk_score == pytest.approx(BENIGN_SCORE)
    assert overlay.apply_risk(make_weight_vector({"ES": 0.3}), _state())["ES"] == pytest.approx(0.3)


def test_no_running_event_loop_uses_cache_and_records_staleness() -> None:
    clock = FakeClock()
    client = CountingClient()
    overlay = _overlay(client, ttl=0.05, clock=clock)
    overlay.set_cached_assessments(
        [make_risk_assessment("ES", semantic_risk_score=BENIGN_SCORE, confidence=0.9)]
    )
    assert overlay.is_cache_fresh()
    assert overlay.staleness_count == 0
    clock.advance(1.0)

    # Synchronous backtest runner: no loop, no task, no exception.
    result = overlay.apply_risk(make_weight_vector({"ES": 0.3}), _state())
    assert result["ES"] == pytest.approx(0.3)
    assert overlay.pending_refresh_count == 0
    assert overlay.staleness_count == 1
    assert overlay.last_stale_at == clock.now()
    assert overlay.valid_until_utc is not None  # the cache is retained
    assert overlay.cache_age_seconds() == pytest.approx(1.0)
    assert client.calls == 0


async def test_drain_timeout_does_not_cancel_refresh_and_stop_cleans_up() -> None:
    client = CountingClient(delay=1.0)
    overlay = _overlay(client)
    overlay.apply_risk(make_weight_vector({"ES": 0.4}), _state())
    assert overlay.pending_refresh_count == 1
    await overlay.drain_pending_refresh(timeout=0.0)
    # A caller deadline does not cancel the in-flight cache refresh.
    assert overlay.pending_refresh_count == 1
    await overlay.stop_background_cache()
    assert overlay.pending_refresh_count == 0
    assert overlay.refresh_errors == 0  # cancellation is not a refresh failure
    with pytest.raises(ValueError, match="timeout"):
        await overlay.drain_pending_refresh(timeout=-1.0)


async def test_single_flight_fire_and_forget_refreshes() -> None:
    client = CountingClient(delay=0.02)
    overlay = _overlay(client)
    state = _state()
    overlay.apply_risk(make_weight_vector({"ES": 0.4}), state)
    overlay.apply_risk(make_weight_vector({"ES": 0.4}), state)
    assert overlay.pending_refresh_count == 1
    await overlay.drain_pending_refresh()  # no explicit deadline
    assert client.calls == 1
    assert overlay.pending_refresh_count == 0


async def test_direct_refresh_in_flight_blocks_duplicate_schedules() -> None:
    client = CountingClient(delay=0.05)
    overlay = _overlay(client)
    state = _state()
    direct = asyncio.create_task(overlay.refresh(state, ("ES",)))
    await asyncio.sleep(0)
    assert overlay.refresh_in_flight == 1
    # The stale-cache trigger must not stack a second LLM call batch.
    overlay.apply_risk(make_weight_vector({"ES": 0.4}), state)
    assert overlay.pending_refresh_count == 0
    worker = AsyncLLMRiskCacheWorker(
        overlay,
        state_provider=_state,
        candidates_provider=lambda snapshot: ("ES",),
    )
    assert await worker.run_once() is False
    await direct
    assert client.calls == 1
    assert overlay.is_cache_fresh()


# ---------------------------------------------------------------------------
# TTL lifecycle
# ---------------------------------------------------------------------------


async def test_ttl_expiry_and_cache_metadata() -> None:
    clock = FakeClock()
    client = CountingClient()
    overlay = _overlay(client, ttl=0.05, clock=clock)
    assert overlay.cache_entry is None
    assert overlay.valid_until_utc is None
    assert overlay.refreshed_at_utc is None
    assert overlay.is_cache_fresh() is False
    assert overlay.cache_age_seconds() == float("inf")

    worker = AsyncLLMRiskCacheWorker(
        overlay,
        state_provider=_state,
        candidates_provider=lambda state: ("ES",),
        sleep=_fake_sleep(clock),
        clock=clock.now,
    )
    assert await worker.run_once() is True
    assert overlay.is_cache_fresh()
    refreshed = overlay.refreshed_at_utc
    valid_until = overlay.valid_until_utc
    assert refreshed is not None and valid_until is not None
    assert valid_until - refreshed == timedelta(seconds=0.05)
    assert overlay.cache_age_seconds() == pytest.approx(0.0)

    clock.advance(0.05)
    assert not overlay.is_cache_fresh()
    assert await worker.run_once() is True
    assert worker.refreshes == 2
    assert client.calls == 2
    assert overlay.cache_age_seconds() == pytest.approx(0.0)


def test_worker_interval_is_min_of_ttl_and_configuration() -> None:
    overlay = _overlay(CountingClient(), ttl=60.0)
    default_worker = AsyncLLMRiskCacheWorker(
        overlay,
        state_provider=_state,
        candidates_provider=lambda state: ("ES",),
    )
    assert default_worker.interval_seconds == pytest.approx(60.0)

    shorter = AsyncLLMRiskCacheWorker(
        overlay,
        state_provider=_state,
        candidates_provider=lambda state: ("ES",),
        interval_seconds=10.0,
    )
    assert shorter.interval_seconds == pytest.approx(10.0)

    long_ttl_overlay = _overlay(CountingClient(), ttl=5.0)
    capped = AsyncLLMRiskCacheWorker(
        long_ttl_overlay,
        state_provider=_state,
        candidates_provider=lambda state: ("ES",),
        interval_seconds=30.0,
    )
    assert capped.interval_seconds == pytest.approx(5.0)

    with pytest.raises(ValueError, match="interval_seconds"):
        AsyncLLMRiskCacheWorker(
            overlay,
            state_provider=_state,
            candidates_provider=lambda state: ("ES",),
            interval_seconds=0.0,
        )
    with pytest.raises(ValueError, match="max_failures"):
        AsyncLLMRiskCacheWorker(
            overlay,
            state_provider=_state,
            candidates_provider=lambda state: ("ES",),
            max_failures=0,
        )
    with pytest.raises(TypeError, match="overlay"):
        AsyncLLMRiskCacheWorker(
            object(),  # type: ignore[arg-type]
            state_provider=_state,
            candidates_provider=lambda state: ("ES",),
        )


# ---------------------------------------------------------------------------
# Failure handling: last-known-good cache survives
# ---------------------------------------------------------------------------


async def test_refresh_failure_keeps_last_known_good_cache() -> None:
    clock = FakeClock()
    client = CountingClient()
    overlay = _overlay(client, ttl=0.05, clock=clock)
    worker = AsyncLLMRiskCacheWorker(
        overlay,
        state_provider=_state,
        candidates_provider=lambda state: ("ES",),
        sleep=_fake_sleep(clock),
        clock=clock.now,
    )
    assert await worker.run_once() is True
    cached_at = overlay.refreshed_at_utc
    cached_valid_until = overlay.valid_until_utc
    last_good = overlay.latest_assessments["ES"]

    client.fail = True
    clock.advance(1.0)
    assert await worker.run_once() is False
    assert worker.failures == 1
    assert worker.consecutive_failures == 1
    assert worker.last_error is not None and "LLMClientError" in worker.last_error
    assert overlay.refresh_errors == 1
    assert overlay.last_refresh_error is not None
    # Last-known-good cache is untouched (only the timestamp ages).
    assert overlay.latest_assessments["ES"] == last_good
    assert overlay.refreshed_at_utc == cached_at
    assert overlay.valid_until_utc == cached_valid_until
    result = overlay.apply_risk(make_weight_vector({"ES": 0.3}), _state())
    assert result["ES"] == pytest.approx(0.3)

    client.fail = False
    assert await worker.run_once() is True
    assert worker.refreshes == 2
    assert worker.consecutive_failures == 0


async def test_background_refresh_failure_is_contained() -> None:
    clock = FakeClock()
    client = CountingClient()
    client.fail = True
    overlay = _overlay(client, ttl=0.05, clock=clock)
    overlay.set_cached_assessments(
        [make_risk_assessment("ES", semantic_risk_score=BENIGN_SCORE, confidence=0.9)]
    )
    clock.advance(1.0)
    overlay.apply_risk(make_weight_vector({"ES": 0.3}), _state())
    assert overlay.pending_refresh_count == 1
    await overlay.drain_pending_refresh(timeout=5.0)
    assert overlay.pending_refresh_count == 0
    assert overlay.refresh_errors == 1
    assert overlay.last_refresh_error is not None
    assert "LLMClientError" in overlay.last_refresh_error
    # The last-known-good cache survives a failed fire-and-forget refresh.
    assert overlay.latest_assessments["ES"].semantic_risk_score == pytest.approx(BENIGN_SCORE)
    assert overlay.apply_risk(make_weight_vector({"ES": 0.3}), _state())["ES"] == pytest.approx(0.3)


async def test_worker_skips_without_state_or_candidates_and_counts_provider_errors() -> None:
    client = CountingClient()
    overlay = _overlay(client)
    no_state = AsyncLLMRiskCacheWorker(
        overlay,
        state_provider=lambda: None,
        candidates_provider=lambda state: ("ES",),
    )
    assert await no_state.run_once() is False
    empty_candidates = AsyncLLMRiskCacheWorker(
        overlay,
        state_provider=_state,
        candidates_provider=lambda state: (),
    )
    assert await empty_candidates.run_once() is False
    assert client.calls == 0
    assert no_state.failures == 0

    def _boom() -> MarketState:
        raise RuntimeError("state feed down")

    broken = AsyncLLMRiskCacheWorker(
        overlay,
        state_provider=_boom,
        candidates_provider=lambda state: ("ES",),
    )
    assert await broken.run_once() is False
    assert broken.failures == 1
    assert broken.last_error is not None and "RuntimeError" in broken.last_error

    def _bad_candidates(state: MarketState) -> tuple[str, ...]:
        raise RuntimeError("universe feed down")

    bad_universe = AsyncLLMRiskCacheWorker(
        overlay,
        state_provider=_state,
        candidates_provider=_bad_candidates,
    )
    assert await bad_universe.run_once() is False
    assert bad_universe.failures == 1

    with pytest.raises(TypeError, match="state_provider"):
        AsyncLLMRiskCacheWorker(
            overlay,
            state_provider=object(),  # type: ignore[arg-type]
            candidates_provider=lambda state: ("ES",),
        )
    with pytest.raises(TypeError, match="candidates_provider"):
        AsyncLLMRiskCacheWorker(
            overlay,
            state_provider=_state,
            candidates_provider=object(),  # type: ignore[arg-type]
        )


async def test_worker_backoff_is_exponential_and_capped() -> None:
    client = CountingClient()
    client.fail = True
    clock = FakeClock()
    overlay = _overlay(client, ttl=60.0, clock=clock)
    sleeps: list[float] = []

    async def recording_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        await asyncio.sleep(0)

    worker = AsyncLLMRiskCacheWorker(
        overlay,
        state_provider=_state,
        candidates_provider=lambda state: ("ES",),
        sleep=recording_sleep,
        clock=clock.now,
        max_failures=4,
    )
    worker.start()
    try:
        assert await _wait_until(lambda: not worker.is_running, attempts=4000)
    finally:
        await worker.stop()

    assert worker.failures == 4
    assert worker.consecutive_failures == 4
    # interval(60s) * 2**n, capped at 300s, and no sleep after the final failure.
    assert sleeps == pytest.approx([120.0, 240.0, 300.0])
    assert overlay.refresh_errors == 4


# ---------------------------------------------------------------------------
# Worker lifecycle / single-flight
# ---------------------------------------------------------------------------


async def test_worker_start_stop_is_clean() -> None:
    clock = FakeClock()
    client = CountingClient()
    overlay = _overlay(client, ttl=0.05, clock=clock)
    worker = AsyncLLMRiskCacheWorker(
        overlay,
        state_provider=_state,
        candidates_provider=lambda state: ("ES",),
        sleep=_fake_sleep(clock),
        clock=clock.now,
    )
    assert not worker.is_running
    worker.start()
    worker.start()  # idempotent
    assert worker.is_running
    assert await _wait_until(lambda: worker.refreshes >= 3)
    assert worker.last_run_at is not None
    assert not worker.stop_event.is_set()
    await worker.stop()
    assert worker.stop_event.is_set()
    assert not worker.is_running
    assert worker._task is None
    assert overlay.pending_refresh_count == 0
    # Stopping a never-started worker is a harmless no-op.
    await worker.stop()
    assert not worker.is_running


async def test_worker_run_once_is_single_flight() -> None:
    client = CountingClient(delay=0.01)
    overlay = _overlay(client)
    worker = AsyncLLMRiskCacheWorker(
        overlay,
        state_provider=_state,
        candidates_provider=lambda state: ("ES",),
    )
    first, second = await asyncio.gather(worker.run_once(), worker.run_once())
    assert sorted([first, second]) == [False, True]
    assert worker.refreshes == 1
    assert client.calls == 1


def test_worker_start_requires_running_loop() -> None:
    overlay = _overlay(CountingClient())
    worker = AsyncLLMRiskCacheWorker(
        overlay,
        state_provider=_state,
        candidates_provider=lambda state: ("ES",),
    )
    with pytest.raises(RuntimeError, match="no running event loop"):
        worker.start()


# ---------------------------------------------------------------------------
# TTL configuration via RiskSettings
# ---------------------------------------------------------------------------


def test_cache_ttl_is_sourced_from_risk_settings() -> None:
    settings = RiskSettings(llm_cache_ttl_seconds=12.5)
    config = LLMRiskConfig.from_risk_settings(settings)
    # New field...
    assert config.cache_ttl_seconds == pytest.approx(12.5)
    # ...and every previously mapped field stays backward compatible.
    assert config.confidence_threshold == settings.llm_confidence_threshold
    assert config.variance_max == settings.llm_semantic_variance_max
    assert config.sample_count == settings.llm_sample_count
    assert config.safe_score == settings.llm_safe_score
    assert config.critical_score == settings.llm_critical_score
    assert config.penalty_lambda == settings.llm_risk_penalty_lambda
    assert config.macro_lockout_minutes_before == settings.macro_lockout_minutes_before
    assert config.macro_lockout_minutes_after == settings.macro_lockout_minutes_after

    clock = FakeClock()
    overlay = LLMRiskOverlay(CountingClient(), settings, clock=clock.now)
    assert overlay.config.cache_ttl_seconds == pytest.approx(12.5)
    overlay.set_cached_assessments(
        [make_risk_assessment("ES", semantic_risk_score=BENIGN_SCORE, confidence=0.9)]
    )
    refreshed = overlay.refreshed_at_utc
    valid_until = overlay.valid_until_utc
    assert refreshed is not None and valid_until is not None
    assert valid_until - refreshed == timedelta(seconds=12.5)
    assert overlay.is_cache_fresh(now=clock.now() + timedelta(seconds=12.4))
    assert not overlay.is_cache_fresh(now=clock.now() + timedelta(seconds=12.5))

    with pytest.raises(ValidationError):
        LLMRiskConfig(cache_ttl_seconds=0.0)
    with pytest.raises(ValidationError):
        RiskSettings(llm_cache_ttl_seconds=0.0)


def test_cache_entry_validates_its_window() -> None:
    entry = LLMRiskCacheEntry(
        assessments={},
        refreshed_at_utc=FIXED_TIMESTAMP,
        valid_until_utc=FIXED_TIMESTAMP + timedelta(seconds=1.0),
    )
    assert entry.refreshed_at_utc == FIXED_TIMESTAMP
    with pytest.raises(ValueError, match="valid_until_utc"):
        LLMRiskCacheEntry(
            assessments={},
            refreshed_at_utc=FIXED_TIMESTAMP,
            valid_until_utc=FIXED_TIMESTAMP,
        )
    with pytest.raises(ValueError):
        LLMRiskCacheEntry(
            assessments={},
            refreshed_at_utc=datetime(2024, 1, 2, 21, 0),  # naive
            valid_until_utc=datetime(2024, 1, 2, 21, 1, tzinfo=UTC),
        )
