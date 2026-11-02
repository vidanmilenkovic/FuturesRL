"""Unit tests for the live/paper trading runner (offline, deterministic).

Every dependency is a deterministic fake: the streamer yields fixture
:class:`~src.core.types.MarketState`s, the broker records ``submit_targets``
calls, the hydrator records when hydration happened, and the clock/event bus are
injected. No test connects to IBKR, Databento or the network.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from configs.instruments import InstrumentsConfig
from configs.settings import Settings
from src.core.event_bus import EventBus, TelemetryEvent
from src.core.types import (
    ContractTargetSet,
    MarketState,
    OrderReceipt,
    OrderStatus,
    PipelineResult,
    Position,
    WeightVector,
)
from src.execution.hydrator import HydrationError, HydrationResult
from src.execution.runner import (
    RunnerConfigError,
    RunnerNotHydratedError,
    RunnerStatus,
    TradingRunner,
    _check_configuration,
    _parse_roots,
    main,
)
from src.risk.state_store import InMemoryRiskStateStore, PersistedRiskState

from tests.fixtures.market import (
    FIXED_TIMESTAMP,
    make_account_snapshot,
    make_bars_frame,
    make_contract_targets,
    make_market_state,
    make_weight_vector,
)
from tests.unit.test_ibkr_client import FakeClock

TELEMETRY_TOPIC = "telemetry"


# ---------------------------------------------------------------------------
# Deterministic fakes
# ---------------------------------------------------------------------------


class FakeHydrator:
    """Records hydration calls; returns a genuine :class:`HydrationResult`."""

    def __init__(
        self,
        events: list[str] | None = None,
        *,
        fail: BaseException | None = None,
    ) -> None:
        self.events: list[str] = events if events is not None else []
        self.fail = fail
        self.calls: list[tuple[tuple[str, ...], datetime, Any]] = []

    def hydrate(
        self,
        roots: tuple[str, ...],
        as_of: datetime,
        *,
        equity: float | None = None,
        allow_backfill: bool | None = None,
    ) -> HydrationResult:
        del allow_backfill
        self.events.append("hydrate")
        resolved = tuple(roots)
        self.calls.append((resolved, as_of, equity))
        if self.fail is not None:
            raise self.fail
        state = make_market_state(roots=resolved, timestamp=as_of)
        return HydrationResult(
            states=(state,),
            bars=make_bars_frame(roots=resolved, periods=2),
            roots=resolved,
            lookback_days=400,
            coverage=dict.fromkeys(resolved, 300),
            required_bars=260,
            backfilled=False,
            warnings=(),
            as_of=as_of,
        )


class FakeStreamer:
    """In-memory ``MarketDataStreamer`` yielding prepared market states."""

    def __init__(self, states: list[MarketState], events: list[str] | None = None) -> None:
        self._states = tuple(states)
        self.events: list[str] = events if events is not None else []
        self.connect_calls = 0
        self.disconnect_calls = 0
        self.stream_calls = 0
        self.is_connected = False

    async def connect(self) -> None:
        self.connect_calls += 1
        self.is_connected = True

    async def disconnect(self) -> None:
        self.disconnect_calls += 1
        self.is_connected = False

    async def stream(self) -> AsyncIterator[MarketState]:
        self.stream_calls += 1
        for index, state in enumerate(self._states):
            self.events.append(f"stream:{index}")
            yield state


class FakeBroker:
    """Records ``submit_targets`` calls and returns filled receipts."""

    def __init__(
        self,
        positions: dict[str, Position] | None = None,
        events: list[str] | None = None,
    ) -> None:
        self.positions = dict(positions or {})
        self.events: list[str] = events if events is not None else []
        self.submitted: list[tuple[ContractTargetSet, dict[str, int], datetime]] = []
        self.connect_calls = 0
        self.disconnect_calls = 0
        self.is_locked = False
        self.account_calls = 0

    async def connect(self) -> None:
        self.connect_calls += 1

    async def disconnect(self) -> None:
        self.disconnect_calls += 1

    async def get_positions(self) -> dict[str, Position]:
        return dict(self.positions)

    async def get_account(self) -> Any:
        self.account_calls += 1
        return make_account_snapshot()

    async def submit_targets(
        self,
        targets: ContractTargetSet,
        current_positions: dict[str, int],
        timestamp: datetime,
        *,
        flatten: bool = False,
    ) -> list[OrderReceipt]:
        del flatten
        self.events.append("submit")
        self.submitted.append((targets, dict(current_positions), timestamp))
        return [
            OrderReceipt(
                order_id=f"ord-{index}",
                status=OrderStatus.FILLED,
                filled_quantity=target.quantity,
                average_price=target.price,
                timestamp=timestamp,
            )
            for index, target in enumerate(targets)
        ]


class LegacyBroker(FakeBroker):
    """Broker exposing only ``orders_from_targets`` + ``submit_orders``."""

    async def orders_from_targets(
        self,
        targets: ContractTargetSet,
        current_positions: dict[str, int],
        timestamp: datetime,
    ) -> list[Any]:
        from src.execution.oms import build_orders_from_targets

        return build_orders_from_targets(targets, current_positions, timestamp)

    async def submit_orders(self, orders: list[Any]) -> list[OrderReceipt]:
        self.events.append("submit")
        return [
            OrderReceipt(
                order_id=order.order_id,
                status=OrderStatus.FILLED,
                filled_quantity=order.quantity,
                timestamp=order.timestamp,
            )
            for order in orders
        ]


class FakePipeline:
    """``StrategyPipeline`` double producing deterministic contract targets."""

    name = "fake-pipeline"

    def __init__(
        self,
        *,
        with_targets: bool = True,
        fail_on: set[datetime] | None = None,
        on_run: Callable[[MarketState], None] | None = None,
    ) -> None:
        self.with_targets = with_targets
        self.fail_on = fail_on or set()
        self.on_run = on_run
        self.calls: list[datetime] = []

    def run(self, state: MarketState) -> PipelineResult:
        self.calls.append(state.timestamp)
        if self.on_run is not None:
            self.on_run(state)
        if state.timestamp in self.fail_on:
            raise RuntimeError("pipeline exploded")
        weights = make_weight_vector({"ES": 0.5})
        targets = make_contract_targets({"ES": 1}) if self.with_targets else None
        return PipelineResult(
            timestamp=state.timestamp,
            universe=("ES",),
            base_weights=weights,
            timing_weights=weights,
            risk_weights=weights,
            final_weights=weights,
            targets=targets,
        )


class FakeTranslator:
    """Records ``translate`` calls and returns a fixed target set."""

    def __init__(self) -> None:
        self.calls: list[tuple[WeightVector, MarketState, float | None]] = []

    def translate(
        self,
        weights: WeightVector,
        state: MarketState,
        *,
        equity: float | None = None,
    ) -> ContractTargetSet:
        self.calls.append((weights, state, equity))
        return make_contract_targets({"NQ": 2})


def telemetry_bus() -> tuple[EventBus[TelemetryEvent], list[TelemetryEvent]]:
    bus: EventBus[TelemetryEvent] = EventBus()
    events: list[TelemetryEvent] = []
    bus.subscribe(TELEMETRY_TOPIC, events.append)
    return bus, events


def single_state_states(count: int = 1) -> list[MarketState]:
    return [
        make_market_state(roots=("ES",), timestamp=FIXED_TIMESTAMP + timedelta(days=index))
        for index in range(count)
    ]


def make_runner(
    settings: Settings,
    *,
    states: list[MarketState] | None = None,
    hydrator: FakeHydrator | None = None,
    streamer: FakeStreamer | None = None,
    broker: FakeBroker | None = None,
    pipeline: FakePipeline | None = None,
    translator: FakeTranslator | None = None,
    event_bus: EventBus[TelemetryEvent] | None = None,
    state_store: InMemoryRiskStateStore | object | None = None,
    clock: FakeClock | None = None,
    halt_on_error: bool = False,
    dry_run: bool = True,
    events: list[str] | None = None,
) -> tuple[TradingRunner, dict[str, Any]]:
    ordered_events: list[str] = events if events is not None else []
    market_states = states if states is not None else single_state_states()
    components: dict[str, Any] = {
        "events": ordered_events,
        "hydrator": hydrator if hydrator is not None else FakeHydrator(ordered_events),
        "streamer": streamer
        if streamer is not None
        else FakeStreamer(market_states, ordered_events),
        "broker": broker if broker is not None else FakeBroker(events=ordered_events),
        "pipeline": pipeline if pipeline is not None else FakePipeline(),
        "translator": translator,
    }
    runner = TradingRunner(
        settings=settings,
        pipeline=components["pipeline"],
        streamer=components["streamer"],
        broker=components["broker"],
        hydrator=components["hydrator"],
        translator=translator,
        event_bus=event_bus,
        state_store=state_store,
        clock=clock if clock is not None else FakeClock(FIXED_TIMESTAMP),
        halt_on_error=halt_on_error,
        dry_run=dry_run,
    )
    return runner, components


# ---------------------------------------------------------------------------
# Hydration ordering (upgrade 1: hydrate BEFORE the first IBKR bar)
# ---------------------------------------------------------------------------


async def test_warm_start_runs_before_the_first_streamed_state(settings: Settings) -> None:
    runner, fakes = make_runner(settings, states=single_state_states(2))
    broker: FakeBroker = fakes["broker"]
    status = await runner.run(roots=("ES",))
    events: list[str] = fakes["events"]
    assert events[0] == "hydrate"
    assert events[1].startswith("stream:")
    first_submit = events.index("submit")
    assert first_submit > events.index("stream:0")
    assert runner.hydrated
    assert status.states_processed == 2
    assert status.orders_submitted == 2
    assert broker.connect_calls == 0  # dry run never touches the venues
    assert status.halted is False
    assert status.started_at == FIXED_TIMESTAMP
    assert status.ended_at is not None
    # The first submitted targets are the pipeline's approved targets for the
    # first streamed state (which is only possible after hydration).
    targets, current_positions, timestamp = broker.submitted[0]
    assert targets == make_contract_targets({"ES": 1})
    assert current_positions == {}
    assert timestamp == FIXED_TIMESTAMP


async def test_hydrator_receives_the_requested_roots_and_as_of(settings: Settings) -> None:
    clock = FakeClock(FIXED_TIMESTAMP)
    runner, fakes = make_runner(settings, clock=clock)
    await runner.run(roots=("es", "nq"))
    hydrator: FakeHydrator = fakes["hydrator"]
    roots, as_of, equity = hydrator.calls[0]
    assert roots == ("ES", "NQ")
    assert as_of == FIXED_TIMESTAMP
    assert equity is None
    assert runner.hydration is not None


async def test_default_root_universe_excludes_micro_contracts(
    settings: Settings,
    instrument_master: InstrumentsConfig,
) -> None:
    runner, fakes = make_runner(settings)
    await runner.run()
    roots = fakes["hydrator"].calls[0][0]
    expected = {spec.root for spec in instrument_master.instruments.values() if not spec.is_micro}
    assert set(roots) == expected
    assert all(instrument_master.get(root).is_micro is False for root in roots)


async def test_hydration_failure_propagates_and_marks_runner_halted(
    settings: Settings,
) -> None:
    failure = HydrationError("cold pipeline")
    runner, _ = make_runner(
        settings, hydrator=FakeHydrator(fail=failure), states=single_state_states(1)
    )
    with pytest.raises(HydrationError):
        await runner.run(roots=("ES",))
    assert runner.hydrated is False
    assert runner.halted is True


# ---------------------------------------------------------------------------
# run_once / hydration guard
# ---------------------------------------------------------------------------


async def test_run_once_requires_warm_start(settings: Settings) -> None:
    runner, fakes = make_runner(settings, states=single_state_states(1))
    state = single_state_states(1)[0]
    with pytest.raises(RunnerNotHydratedError):
        await runner.run_once(state)
    assert fakes["broker"].submitted == []
    assert fakes["pipeline"].calls == []  # the pipeline never saw a cold state
    runner.warm_start(("ES",))
    receipts = await runner.run_once(state)
    assert len(receipts) == 1
    assert receipts[0].status is OrderStatus.FILLED
    assert runner.status.states_processed == 1
    assert len(fakes["broker"].submitted) == 1


# ---------------------------------------------------------------------------
# Persisted kill switch (upgrade 2: never trade through a latched halt)
# ---------------------------------------------------------------------------


async def test_persisted_kill_switch_prevents_orders(settings: Settings) -> None:
    store = InMemoryRiskStateStore(
        PersistedRiskState(kill_switch_locked=True, kill_switch_reason="daily max loss")
    )
    bus, events = telemetry_bus()
    runner, fakes = make_runner(
        settings, state_store=store, event_bus=bus, states=single_state_states(2)
    )
    status = await runner.run(roots=("ES",))
    assert status.halted is True
    assert status.locked is True
    assert status.states_processed == 0
    assert fakes["broker"].submitted == []
    assert fakes["streamer"].stream_calls == 0
    lock_events = [event for event in events if event.kind == "kill_switch"]
    assert lock_events and lock_events[0].values["reason"] == "daily max loss"
    assert lock_events[0].values["locked"] is True


async def test_kill_switch_latched_mid_run_stops_submitting(settings: Settings) -> None:
    store = InMemoryRiskStateStore()

    def lock_after_first_run(state: MarketState) -> None:
        del state
        store.save(PersistedRiskState(kill_switch_locked=True, kill_switch_reason="operator"))

    pipeline = FakePipeline(on_run=lock_after_first_run)
    runner, fakes = make_runner(
        settings,
        pipeline=pipeline,
        state_store=store,
        states=single_state_states(2),
    )
    status = await runner.run(roots=("ES",))
    assert status.halted is True
    assert status.locked is True
    assert fakes["broker"].submitted == []  # the lock is honoured before submission


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------


async def test_telemetry_events_are_published_in_order(settings: Settings) -> None:
    bus, events = telemetry_bus()
    runner, _ = make_runner(settings, event_bus=bus, states=single_state_states(1))
    await runner.run(roots=("ES",))
    kinds = [event.kind for event in events]
    assert kinds[0] == "runner_started"
    assert "hydration" in kinds
    assert "state" in kinds
    assert "pipeline" in kinds
    assert "orders" in kinds
    assert kinds[-1] == "runner_stopped"
    orders = next(event for event in events if event.kind == "orders")
    assert orders.values["orders"] == 1
    assert orders.values["accepted"] == 1
    assert orders.values["rejected"] == 0
    assert orders.values["net_liquidation"] == pytest.approx(1_000_000.0)
    # Telemetry payloads only carry JSON scalars.
    for event in events:
        for value in event.values.values():
            assert value is None or isinstance(value, (str, int, float, bool))


# ---------------------------------------------------------------------------
# Error isolation / halt_on_error
# ---------------------------------------------------------------------------


async def test_per_state_failures_are_isolated(settings: Settings) -> None:
    states = single_state_states(3)
    pipeline = FakePipeline(fail_on={states[1].timestamp})
    bus, events = telemetry_bus()
    runner, fakes = make_runner(settings, states=states, pipeline=pipeline, event_bus=bus)
    status = await runner.run(roots=("ES",))
    assert status.states_processed == 3
    assert status.orders_submitted == 2
    assert status.halted is False
    assert status.last_error is not None and "pipeline exploded" in status.last_error
    assert len(fakes["broker"].submitted) == 2
    error_events = [event for event in events if event.kind == "error"]
    assert error_events
    assert error_events[0].values["phase"] == "state"
    assert "pipeline exploded" in str(error_events[0].values["error"])


async def test_halt_on_error_stops_the_loop(settings: Settings) -> None:
    states = single_state_states(3)
    pipeline = FakePipeline(fail_on={states[0].timestamp})
    runner, fakes = make_runner(settings, states=states, pipeline=pipeline, halt_on_error=True)
    status = await runner.run(roots=("ES",))
    assert status.halted is True
    assert status.states_processed == 1
    assert status.orders_submitted == 0
    assert status.last_error is not None
    assert fakes["broker"].submitted == []


async def test_max_states_limits_the_loop(settings: Settings) -> None:
    runner, fakes = make_runner(settings, states=single_state_states(3))
    status = await runner.run(roots=("ES",), max_states=2)
    assert status.states_processed == 2
    assert len(fakes["broker"].submitted) == 2
    with pytest.raises(ValueError):
        await runner.run(roots=("ES",), max_states=0)


async def test_dry_run_never_connects_and_live_run_connects_gracefully(
    settings: Settings,
) -> None:
    dry_runner, dry_fakes = make_runner(settings, states=single_state_states(1), dry_run=True)
    await dry_runner.run(roots=("ES",))
    assert dry_fakes["streamer"].connect_calls == 0
    assert dry_fakes["streamer"].disconnect_calls == 0
    assert dry_fakes["broker"].connect_calls == 0

    live_runner, live_fakes = make_runner(settings, states=single_state_states(1), dry_run=False)
    status = await live_runner.run(roots=("ES",))
    assert status.halted is False
    assert live_fakes["streamer"].connect_calls == 1
    assert live_fakes["streamer"].disconnect_calls == 1
    assert live_fakes["broker"].connect_calls == 1
    assert live_fakes["broker"].disconnect_calls == 1


# ---------------------------------------------------------------------------
# Translator / legacy broker adapter
# ---------------------------------------------------------------------------


async def test_translator_is_used_when_the_pipeline_returns_no_targets(
    settings: Settings,
) -> None:
    translator = FakeTranslator()
    runner, fakes = make_runner(
        settings,
        pipeline=FakePipeline(with_targets=False),
        translator=translator,
        states=single_state_states(1),
    )
    await runner.run(roots=("ES",))
    assert len(translator.calls) == 1
    weights, state, equity = translator.calls[0]
    assert weights.get("ES") == pytest.approx(0.5)
    assert state.timestamp == FIXED_TIMESTAMP
    assert equity == state.equity
    targets, _, _ = fakes["broker"].submitted[0]
    assert targets == make_contract_targets({"NQ": 2})


async def test_broker_without_submit_targets_uses_the_order_fallback(
    settings: Settings,
) -> None:
    events: list[str] = []
    legacy = LegacyBroker(events=events)
    runner, fakes = make_runner(
        settings, broker=legacy, states=single_state_states(1), events=events
    )
    await runner.run(roots=("ES",))
    assert fakes["events"].count("submit") == 1
    assert runner.status.orders_submitted == 1


# ---------------------------------------------------------------------------
# Persisted runner-owned risk state
# ---------------------------------------------------------------------------


async def test_runner_persists_starting_daily_nlv_without_clobbering_other_fields(
    settings: Settings,
) -> None:
    initial = PersistedRiskState(
        portfolio_high_water_mark=1_200_000.0,
        instrument_high_water_marks={"ES": 4_900.0},
        extra={"risk_overlay": "hwm"},
    )
    store = InMemoryRiskStateStore(initial)
    states = single_state_states(2)  # both carry equity=1_000_000
    runner, _ = make_runner(settings, state_store=store, states=states)
    await runner.run(roots=("ES",))
    persisted = store.load()
    assert persisted is not None
    assert persisted.starting_daily_nlv == pytest.approx(1_000_000.0)
    assert persisted.trading_day == FIXED_TIMESTAMP.date()
    assert persisted.portfolio_high_water_mark == pytest.approx(1_200_000.0)
    assert dict(persisted.instrument_high_water_marks) == {"ES": 4_900.0}
    assert dict(persisted.extra) == {"risk_overlay": "hwm"}
    assert persisted.updated_at is not None
    assert runner.persisted_state is not None


async def test_state_store_errors_do_not_kill_the_trading_loop(settings: Settings) -> None:
    class FlakyStore:
        def load(self) -> PersistedRiskState | None:
            raise OSError("state store unavailable")

        def save(self, state: PersistedRiskState) -> None:
            raise OSError("state store unavailable")

        def clear(self) -> None:
            pass

    runner, fakes = make_runner(settings, state_store=FlakyStore(), states=single_state_states(1))
    status = await runner.run(roots=("ES",))
    assert status.states_processed == 1
    assert len(fakes["broker"].submitted) == 1
    assert status.last_error is not None and "risk state" in status.last_error


# ---------------------------------------------------------------------------
# Status snapshot
# ---------------------------------------------------------------------------


async def test_status_snapshot_is_typed_and_serialisable(settings: Settings) -> None:
    runner, _ = make_runner(settings, states=single_state_states(2))
    status = await runner.run(roots=("ES",))
    assert isinstance(status, RunnerStatus)
    payload = status.as_dict()
    assert payload["hydrated"] is True
    assert payload["states_processed"] == 2
    assert payload["orders_submitted"] == 2
    assert payload["roots"] == ["ES"]
    assert json.dumps(payload)  # JSON-serialisable
    with pytest.raises(ValueError):
        RunnerStatus(states_processed=-1)


# ---------------------------------------------------------------------------
# CLI configuration check (offline)
# ---------------------------------------------------------------------------


def test_parse_roots_handles_commas_and_blanks() -> None:
    assert _parse_roots(None) is None
    assert _parse_roots("  ") is None
    assert _parse_roots("es, nq") == ("ES", "NQ")
    with pytest.raises(RunnerConfigError):
        _parse_roots(",")


def test_check_configuration_is_offline_and_complete(
    settings: Settings, instrument_master: InstrumentsConfig
) -> None:
    strategy_path = settings.strategies_dir / "tsmom_vol_target.yaml"
    summary = _check_configuration(settings, str(strategy_path))
    assert summary["ok"] is True
    assert summary["instrument_roots"] == list(instrument_master.roots())
    assert summary["state_store"] == "InMemoryRiskStateStore"
    assert summary["kill_switch_locked"] is False
    strategy = summary["strategy"]
    assert isinstance(strategy, dict)
    assert strategy["id"]
    assert "rounding_mode" in strategy
    assert summary["hydration_required_lookback_days"] == 374
    assert summary["hydration_lookback_adequate"] is True
    assert summary["hydration"]["backfill_enabled"] is True


def test_check_configuration_without_a_strategy(settings: Settings) -> None:
    summary = _check_configuration(settings, None)
    assert summary["ok"] is True
    assert summary["strategy"] is None


def test_main_check_mode_prints_json_and_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--check"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["strategy"]["id"]


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_runner_exposes_the_expected_public_surface(settings: Settings) -> None:
    from src.core.interfaces import MarketDataStreamer, StrategyPipeline

    runner, fakes = make_runner(settings)
    assert isinstance(fakes["streamer"], MarketDataStreamer)
    assert isinstance(fakes["pipeline"], StrategyPipeline)
    assert runner.settings is settings
    assert runner.pipeline is fakes["pipeline"]
    assert runner.streamer is fakes["streamer"]
    assert runner.broker is fakes["broker"]
    assert runner.hydrator is fakes["hydrator"]
    assert runner.hydration is None
    assert runner.persisted_state is None
    assert runner.instrument_master.roots()
    with pytest.raises(TypeError):
        TradingRunner(  # type: ignore[call-arg]
            settings="not-settings",  # type: ignore[arg-type]
            pipeline=fakes["pipeline"],
            streamer=fakes["streamer"],
            broker=fakes["broker"],
            hydrator=fakes["hydrator"],
        )
    with pytest.raises(ValueError):
        TradingRunner(
            settings=settings,
            pipeline=fakes["pipeline"],
            streamer=fakes["streamer"],
            broker=fakes["broker"],
            hydrator=fakes["hydrator"],
            telemetry_topic="",
        )


def test_runner_module_clock_default_is_utc() -> None:
    from src.execution.runner import _utc_now

    now = _utc_now()
    assert now.tzinfo is not None
    assert now.astimezone(UTC).tzinfo is UTC
    assert isinstance(FIXED_TIMESTAMP, datetime)
