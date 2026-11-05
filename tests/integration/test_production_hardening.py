"""Phase-7 integration tests: production-hardening upgrades 1-4.

Wires the real components together offline:

1. cold-start hydration -> runner warm start -> first live bar (with broker NLV
   enrichment and a warm pipeline prime);
2. durable risk/kill-switch state across a simulated container restart, with
   field preservation (rate-limiter fields written by the OMS survive the risk
   overlay's writes);
3. open-order-aware reconciliation (in-flight working orders suppress drift
   inside the grace window; stale/absent orders halt);
4. non-blocking TTL-cached LLM risk worker (slow provider never blocks the
   synchronous projection or the event loop).
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from configs.instruments import InstrumentsConfig, load_instruments
from configs.settings import RiskSettings, Settings
from src.analytics import compute_metrics_from_result  # noqa: F401 - import smoke
from src.core.event_bus import EventBus, TelemetryEvent
from src.core.types import (
    AccountSnapshot,
    InstrumentId,
    MarketState,
    OrderReceipt,
    OrderStatus,
    Position,
    WeightVector,
)
from src.data import FeatureProcessor, OfflineBarSource, ParquetRepository
from src.execution.hydrator import ColdStartHydrator
from src.execution.reconciler import PositionReconciler, WorkingOrderSnapshot
from src.execution.runner import RunnerNotHydratedError, TradingRunner
from src.risk.llm_risk_overlay import LLMRiskConfig, LLMRiskOverlay
from src.risk.quant_risk_overlay import QuantRiskOverlay
from src.risk.state_store import (
    FileRiskStateStore,
    RateLimiterState,
)
from src.strategies import build_pipeline
from src.strategies.pipeline import ComposablePipeline

from tests.fixtures.market import FIXED_TIMESTAMP, make_market_state

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOTS = ("ES", "NQ", "CL")
INITIAL_EQUITY = 1_000_000.0
LIVE_TS = datetime(2025, 1, 6, 21, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    base = tmp_path_factory.mktemp("phase7-hardening")
    return Settings(
        _env_file=None,
        environment="backtest",
        data_dir=base / "data",
        parquet_dir=base / "data" / "parquet",
        duckdb_path=base / "data" / "state.duckdb",
        report_dir=base / "artifacts",
        instruments_path=REPO_ROOT / "configs" / "instruments.yaml",
        risk_state_store_uri=str(base / "risk_state.json"),
    )


@pytest.fixture(scope="module")
def instrument_master() -> InstrumentsConfig:
    return load_instruments(REPO_ROOT / "configs" / "instruments.yaml")


def _repository_with_history(settings: Settings) -> tuple[ParquetRepository, datetime]:
    """Seed a temporary Parquet cache with deterministic offline history."""
    source = OfflineBarSource(ROOTS, seed=7, periods=340)
    bars = source.load(ROOTS, source.start, source.start + timedelta(days=360))
    # The offline source is single-contract; supply the explicit continuous-frame
    # term-structure columns that a real Databento multi-expiry ingestion emits, so
    # the hydrator's warm-up validation exercises finite term_slope / carry.
    bars = bars.with_columns(
        pl.lit(0.0025).alias("term_slope"),
        pl.lit(-0.0304).alias("carry_annualized"),
    )
    repository = ParquetRepository(settings)
    repository.write_bars(bars)
    as_of = max(bars["timestamp_utc"].to_list())
    return repository, as_of


def _strategy_config():
    from configs.strategy import StageConfig, load_strategy_config

    config = load_strategy_config(REPO_ROOT / "configs" / "strategies" / "tsmom_vol_target.yaml")
    return config.model_copy(
        update={
            "universe": StageConfig(
                name="liquidity",
                params={
                    "min_avg_dollar_volume": 0.0,
                    "min_open_interest": 0.0,
                    "top_n": len(ROOTS),
                    "min_history_days": 0,
                },
            ),
            "allocator": StageConfig(name="equal_weight", params={}),
            "timing": (),
            "risk": (),
        }
    )


class _RecordingPipeline:
    """Records every state passed to the real pipeline (warm-up ordering proof)."""

    def __init__(self, inner: ComposablePipeline) -> None:
        self._inner = inner
        self.states: list[MarketState] = []

    @property
    def name(self) -> str:
        return self._inner.name

    def run(self, state: MarketState):
        self.states.append(state)
        return self._inner.run(state)


class _FakeStreamer:
    """Async streamer yielding two cold live states (equity=None like real feeds)."""

    def __init__(self, states: list[MarketState]) -> None:
        self._states = states
        self.connected = False

    @property
    def is_connected(self) -> bool:
        return self.connected

    async def connect(self) -> None:
        self.connected = True

    async def disconnect(self) -> None:
        self.connected = False

    async def stream(self):
        for state in self._states:
            yield state


class _FakeBroker:
    """Target broker with a live account NLV but no per-state equity."""

    def __init__(self, equity: float = INITIAL_EQUITY) -> None:
        self._equity = equity
        self.submitted: list[tuple[datetime, dict[str, int]]] = []

    @property
    def is_locked(self) -> bool:
        return False

    async def get_positions(self) -> dict[InstrumentId, Position]:
        return {}

    async def get_account(self) -> AccountSnapshot:
        return AccountSnapshot(
            timestamp=LIVE_TS,
            net_liquidation=self._equity,
            cash_balance=self._equity,
            initial_margin_used=0.0,
        )

    async def submit_targets(self, targets, current_positions, timestamp):
        self.submitted.append((timestamp, dict(targets.quantities)))
        return [
            OrderReceipt(order_id=f"o-{index}", status=OrderStatus.FILLED)
            for index in range(len(targets))
        ]


# ---------------------------------------------------------------------------
# Upgrade 1: hydrate before the first live bar; warm pipeline + NLV enrichment
# ---------------------------------------------------------------------------


def test_runner_hydrates_before_first_live_bar(
    settings: Settings, instrument_master: InstrumentsConfig
) -> None:
    repository, as_of = _repository_with_history(settings)
    hydrator = ColdStartHydrator(
        repository=repository,
        instrument_master=instrument_master,
        feature_processor=FeatureProcessor(),
        lookback_days=settings.hydration_lookback_days,
        backfill_enabled=False,
        clock=lambda: as_of,
    )
    translator = __import__(
        "src.risk.contract_translator", fromlist=["ContractTranslator"]
    ).ContractTranslator(instruments=instrument_master)
    pipeline = _RecordingPipeline(
        build_pipeline(
            _strategy_config(), instrument_master, translator=translator, risk_overlays=[]
        )
    )
    live_states = [
        make_market_state(ROOTS, timestamp=LIVE_TS + timedelta(days=offset), equity=None)
        for offset in range(2)
    ]
    streamer = _FakeStreamer(live_states)
    broker = _FakeBroker()
    bus: EventBus[TelemetryEvent] = EventBus()
    published: list[TelemetryEvent] = []

    async def collector(event: TelemetryEvent) -> None:
        published.append(event)

    bus.subscribe("telemetry", collector)
    runner = TradingRunner(
        settings=settings,
        pipeline=pipeline,
        streamer=streamer,
        broker=broker,
        hydrator=hydrator,
        instrument_master=instrument_master,
        translator=translator,
        event_bus=bus,
        clock=lambda: as_of,
        dry_run=True,
        halt_on_error=True,
        prime_pipeline=True,
    )

    # A cold runner must refuse to process a live bar.
    with pytest.raises(RunnerNotHydratedError):
        asyncio.run(runner.run_once(live_states[0]))

    status = asyncio.run(runner.run(roots=ROOTS, max_states=2))

    assert status.hydrated is True
    assert status.states_processed == 2
    assert not status.halted
    assert broker.submitted, "the warmed runner must submit targets on live bars"
    # Hydration + optional prime happen before the first streamed state.
    assert len(pipeline.states) == 3
    assert pipeline.states[0].timestamp == as_of  # warm pipeline prime
    assert [state.timestamp for state in pipeline.states[1:]] == [
        state.timestamp for state in live_states
    ]
    # Live states had equity=None; the runner enriched them from the broker NLV.
    assert all(state.equity == pytest.approx(INITIAL_EQUITY) for state in pipeline.states[1:])
    kinds = [event.kind for event in published]
    assert "hydration" in kinds
    assert "pipeline_warmup" in kinds
    assert "orders" in kinds
    assert "runner_stopped" in kinds


# ---------------------------------------------------------------------------
# Upgrade 2: durable state survives a simulated container restart
# ---------------------------------------------------------------------------


def test_risk_and_oms_state_survive_restart(
    settings: Settings, instrument_master: InstrumentsConfig, tmp_path: Path
) -> None:
    store = FileRiskStateStore(tmp_path / "risk_state.json")
    risk = RiskSettings(daily_max_loss_pct=0.03)
    overlay = QuantRiskOverlay(risk, instruments=instrument_master, state_store=store)
    overlay.update_equity(INITIAL_EQUITY, FIXED_TIMESTAMP)
    overlay.update_equity(INITIAL_EQUITY * 1.04, FIXED_TIMESTAMP)
    overlay.engage_kill_switch("operator emergency")

    # OMS-side fields written by another subsystem must be preserved.
    persisted = store.load()
    assert persisted is not None
    store.save(
        persisted.updated(
            rate_limiter=RateLimiterState(tokens=1.5, capacity=5.0, last_refill_epoch_seconds=100.0)
        )
    )

    # --- simulated container restart: a brand-new overlay on the same store ---
    restarted = QuantRiskOverlay(risk, instruments=instrument_master, state_store=store)
    assert restarted.persisted_state.portfolio_high_water_mark == pytest.approx(
        INITIAL_EQUITY * 1.04
    )
    assert restarted.kill_switch_active is True
    flat = restarted.apply_risk(WeightVector({"ES": 0.5, "NQ": -0.5}), make_market_state(ROOTS))
    assert flat.is_flat(), "a persisted kill switch must keep flattening after restart"
    # OMS rate-limiter fields survived the overlay's load-modify-save cycle.
    assert store.load().rate_limiter is not None
    assert store.load().rate_limiter.tokens == pytest.approx(1.5)  # type: ignore[union-attr]

    restarted.reset_kill_switch()
    after_reset = store.load()
    assert after_reset is not None and after_reset.kill_switch_locked is False
    assert after_reset.rate_limiter is not None  # preserved again

    # --- session roll: daily counters reset, HWM + kill-switch latch survive ---
    rolled = after_reset.rolled_to_day(after_reset.trading_day or FIXED_TIMESTAMP.date())
    assert rolled.starting_daily_nlv is None
    assert rolled.portfolio_high_water_mark == pytest.approx(INITIAL_EQUITY * 1.04)
    assert rolled.rate_limiter is not None


# ---------------------------------------------------------------------------
# Upgrade 3: open-order-aware reconciliation
# ---------------------------------------------------------------------------


def _reconciler(
    *,
    broker_qty: int,
    expected_qty: int,
    working: list[WorkingOrderSnapshot],
    now: datetime,
) -> PositionReconciler:
    async def broker_positions() -> dict[str, Position]:
        if broker_qty == 0:
            return {}
        return {
            "ES": Position(
                instrument_id="ES",
                quantity=broker_qty,
                average_price=4800.0,
                market_price=4800.0,
                multiplier=50.0,
            )
        }

    async def working_orders() -> list[WorkingOrderSnapshot]:
        return working

    return PositionReconciler(
        expected_positions={"ES": expected_qty},
        broker_positions=broker_positions,
        working_orders=working_orders,
        reconcile_grace_seconds=30.0,
        clock=lambda: now,
    )


def test_open_order_suppresses_drift_inside_grace_window() -> None:
    now = FIXED_TIMESTAMP
    # Only part of the gap is explained by the in-flight order (1 of 3 lots):
    # without the grace window this is drift, but the order is fresh, so it must
    # be suppressed until the order's grace window expires.
    snapshot = WorkingOrderSnapshot(
        order_id="o-1",
        instrument_id="ES",
        signed_remaining=1,
        updated_at=now - timedelta(seconds=5),
        status="SUBMITTED",
    )
    reconciler = _reconciler(broker_qty=0, expected_qty=3, working=[snapshot], now=now)
    assert asyncio.run(reconciler.reconcile()) is True
    assert reconciler.halted is False
    assert reconciler.last_grace_active is True
    assert reconciler.last_working_deltas == {"ES": 1}


def test_stale_open_order_halts_reconciliation() -> None:
    now = FIXED_TIMESTAMP
    stale = WorkingOrderSnapshot(
        order_id="o-1",
        instrument_id="ES",
        signed_remaining=1,
        updated_at=now - timedelta(seconds=300),
        status="SUBMITTED",
    )
    reconciler = _reconciler(broker_qty=0, expected_qty=3, working=[stale], now=now)
    assert asyncio.run(reconciler.reconcile()) is False
    assert reconciler.halted is True
    assert reconciler.last_grace_active is False


def test_partial_fill_with_remaining_open_order_is_not_drift() -> None:
    now = FIXED_TIMESTAMP
    remaining = WorkingOrderSnapshot(
        order_id="o-1",
        instrument_id="ES",
        signed_remaining=1,
        updated_at=now - timedelta(seconds=2),
        status="PARTIALLY_FILLED",
    )
    reconciler = _reconciler(broker_qty=1, expected_qty=2, working=[remaining], now=now)
    assert asyncio.run(reconciler.reconcile()) is True
    assert reconciler.halted is False
    # After the order completes with no fill, the same book IS a drift.
    completed = _reconciler(broker_qty=1, expected_qty=2, working=[], now=now)
    assert asyncio.run(completed.reconcile()) is False
    assert completed.halted is True


# ---------------------------------------------------------------------------
# Upgrade 4: async TTL-cached LLM risk worker never blocks the sync projection
# ---------------------------------------------------------------------------


class _SlowLLMClient:
    def __init__(self, latency: float = 0.2) -> None:
        self.latency = latency
        self.calls = 0

    async def complete(
        self, prompt: str, *, system_prompt: str | None = None, temperature: float = 0.0
    ) -> str:
        del prompt, system_prompt, temperature
        self.calls += 1
        await asyncio.sleep(self.latency)
        return json.dumps(
            {
                "assessments": [
                    {
                        "instrument_id": root,
                        "sentiment_score": 3.0,
                        "semantic_risk_score": 2.0,
                        "confidence": 0.9,
                        "reasoning_summary": "integration fixture",
                    }
                    for root in ROOTS
                ]
            }
        )


def test_slow_llm_refresh_is_non_blocking_and_ttl_cached(
    instrument_master: InstrumentsConfig,
) -> None:
    from src.risk.llm_risk_overlay import AsyncLLMRiskCacheWorker

    client = _SlowLLMClient(latency=0.2)
    overlay = LLMRiskOverlay(
        client,
        LLMRiskConfig(cache_ttl_seconds=0.05, sample_count=1),
        instruments=instrument_master,
    )
    state = make_market_state(ROOTS, timestamp=LIVE_TS)
    weights = WeightVector({"ES": 0.4, "NQ": -0.3})

    # Synchronous projection outside an event loop: no raise, no task leak.
    sync_output = overlay.apply_risk(weights, state)
    assert isinstance(sync_output, WeightVector)

    async def scenario() -> None:
        started = asyncio.get_running_loop().time()
        result = overlay.apply_risk(weights, state)
        elapsed = asyncio.get_running_loop().time() - started
        assert isinstance(result, WeightVector)
        assert elapsed < 0.05, f"apply_risk blocked on LLM I/O ({elapsed:.3f}s)"
        assert overlay.pending_refresh_count >= 1

        worker = AsyncLLMRiskCacheWorker(
            overlay,
            state_provider=lambda: state,
            candidates_provider=lambda current: current.ids,
            interval_seconds=0.05,
        )
        worker.start()
        try:
            deadline = asyncio.get_running_loop().time() + 3.0
            while worker.refreshes < 1 and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.02)
            assert overlay.valid_until_utc is not None
            assert overlay.is_cache_fresh()
            assert worker.refreshes >= 1
            assert overlay.latest_assessments
        finally:
            await worker.stop()
        assert not worker.is_running
        assert overlay.pending_refresh_count == 0

    asyncio.run(scenario())
    assert client.calls >= 1
