"""Live/paper trading runner: hydrate, restore risk state, stream, trade, report.

:class:`TradingRunner` is the top of the execution stack for Phase 7. It wires the
already-verified Phase 1-6 components into a single supervised loop and adds the
two production guarantees the live deployment was missing:

1. **Hydrate before the first IBKR bar.** A cold process has no trailing history,
   so the first live bar would produce ``NaN`` features and no covariance. The
   runner *always* calls :class:`~src.execution.hydrator.ColdStartHydrator` in
   :meth:`TradingRunner.warm_start` before it consumes a single streamed
   :class:`~src.core.types.MarketState`; any attempt to process a state earlier
   raises :class:`RunnerNotHydratedError`.
2. **Never trade through a persisted kill switch.** The durable
   :class:`~src.risk.state_store.RiskStateStore` may hold a latched kill switch
   from a previous process (daily max loss, drawdown stop, operator emergency).
   The runner loads it before the loop and refuses to submit orders while
   ``kill_switch_locked`` is set.

Per streamed state the runner executes the FinRL-X pipeline sequentially and
awaits every broker/streamer call (no fire-and-forget trading path):

``state -> pipeline.run(state) -> targets (or translator) -> OMS.submit_targets``
``      -> telemetry on EventBus -> persisted starting-daily-NLV anchor``.

Per-state failures are isolated: they are logged, published on the telemetry
topic and — when ``halt_on_error`` is set — stop the loop. Telemetry handler
failures can never abort a rebalance: the bus already isolates them and the
runner awaits :meth:`src.core.event_bus.EventBus.publish` for deterministic
ordering.

Cold-start history is sourced through a multi-vendor fallback chain (Phase 8):
``auto`` = Databento (API key) -> IBKR historical daily bars (paper/live) ->
synthetic offline history only when explicitly allowed; ``--backfill-source`` can
force one vendor. ``--offline`` runs the whole loop end-to-end against the
deterministic :class:`~src.data.databento_loader.OfflineBarSource` and an
:class:`OfflineTargetBroker` wrapping the simulated venue, so the runner can be
exercised with no IBKR/Databento connectivity at all.

Production wiring lives in :func:`main`: ``ib_async``/Databento are imported
*lazily* only on the trading path, and ``--check`` validates settings, the
instrument master, the strategy YAML and the hydration configuration without
opening a single socket. Every test injects deterministic fakes and runs fully
offline.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import sys
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

import polars as pl
from configs.instruments import InstrumentsConfig, load_instruments
from configs.settings import Settings, get_settings

from src.core.event_bus import EventBus, TelemetryEvent
from src.core.interfaces import ContractTranslatorProtocol, MarketDataStreamer, StrategyPipeline
from src.core.types import (
    AccountSnapshot,
    ContractTargetSet,
    InstrumentId,
    MarketState,
    OrderEvent,
    OrderReceipt,
    OrderStatus,
    Position,
    ScalarValue,
    ensure_utc,
)
from src.data import schema as s
from src.data.feature_processor import FeatureProcessor
from src.data.market_state_builder import MarketStateBuilder
from src.data.storage import ParquetRepository
from src.execution.hydrator import (
    CALENDAR_SAFETY_DAYS,
    HYDRATION_SOURCE_HINTS,
    HYDRATION_SOURCES,
    MIN_REQUIRED_BARS,
    BackfillFn,
    ColdStartHydrator,
    HydrationError,
    HydrationResult,
    MultiSourceBackfill,
    add_neutral_term_structure,
    build_backfill_chain,
    offline_backfill,
)
from src.risk.state_store import PersistedRiskState, RiskStateStore

if TYPE_CHECKING:
    from configs.strategy import StrategyConfig

__all__ = [
    "DEFAULT_TELEMETRY_TOPIC",
    "OFFLINE_DEFAULT_ROOTS",
    "OFFLINE_INITIAL_CASH",
    "OFFLINE_REPLAY_STATES",
    "OFFLINE_SIMULATION_END",
    "OfflineMarketStateStreamer",
    "OfflineTargetBroker",
    "RunnerConfigError",
    "RunnerNotHydratedError",
    "RunnerStatus",
    "TargetBroker",
    "TradingRunner",
    "build_hydration_chain",
    "describe_hydration_sources",
    "main",
]

logger = logging.getLogger(__name__)

#: Wall-clock provider (UTC, timezone-aware). Mirrors ``src.execution.ibkr_client.Clock``.
type Clock = Callable[[], datetime]

#: Function surface used for ``self.broker.submit_targets(...)``.
type SubmitTargetsFn = Callable[
    [ContractTargetSet, Mapping[InstrumentId, int], datetime],
    Awaitable[Sequence[OrderReceipt]],
]

#: Default telemetry topic consumed by the WebSocket broadcaster.
DEFAULT_TELEMETRY_TOPIC: str = "telemetry"

#: Default strategy config used by the CLI when ``--strategy`` is omitted.
DEFAULT_STRATEGY_FILE: str = "tsmom_vol_target.yaml"

#: Fixed decision timestamp of the deterministic ``--offline`` simulation. A
#: constant (not "now") keeps the offline CLI reproducible across machines/days.
OFFLINE_SIMULATION_END: datetime = datetime(2025, 6, 30, 21, 0, tzinfo=UTC)

#: Starting cash of the offline simulated account.
OFFLINE_INITIAL_CASH: float = 1_000_000.0

#: Number of replay states the offline streamer emits (live-loop window after the
#: hydrated warm-up).
OFFLINE_REPLAY_STATES: int = 30

#: Default futures roots replayed by ``--offline`` (kept small for a fast demo).
OFFLINE_DEFAULT_ROOTS: tuple[InstrumentId, ...] = ("ES", "NQ", "CL", "GC", "ZN")


class RunnerNotHydratedError(RuntimeError):
    """Raised when a state is processed before :meth:`TradingRunner.warm_start`."""


class RunnerConfigError(RuntimeError):
    """Raised when the runner is wired with a dependency it cannot drive."""


@runtime_checkable
class TargetBroker(Protocol):
    """Minimal broker surface the runner drives (satisfied by the Phase-5 OMS).

    ``submit_targets`` is the ``BrokerAdapter``-compatible target submission
    surface of :class:`src.execution.oms.OrderManagementSystem`; ``get_positions``
    is part of the unified :class:`src.core.interfaces.BrokerAdapter` protocol.
    """

    async def get_positions(self) -> Mapping[InstrumentId, Position]:
        """Current broker positions keyed by root instrument id."""
        ...

    async def submit_targets(
        self,
        targets: ContractTargetSet,
        current_positions: Mapping[InstrumentId, int],
        timestamp: datetime,
        *,
        flatten: bool = False,
    ) -> list[OrderReceipt]:
        """Translate targets into delta orders and submit them through the gate."""
        ...


@dataclass(frozen=True, slots=True)
class RunnerStatus:
    """Immutable snapshot of the runner's observable state.

    ``states_processed`` counts the streamed states the runner attempted to
    process (including a state whose pipeline/broker call failed); per-state
    failures are additionally reported through ``last_error`` and telemetry.
    """

    hydrated: bool = False
    states_processed: int = 0
    orders_submitted: int = 0
    orders_rejected: int = 0
    halted: bool = False
    locked: bool = False
    last_error: str | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    roots: tuple[InstrumentId, ...] = ()

    def __post_init__(self) -> None:
        for counter in ("states_processed", "orders_submitted", "orders_rejected"):
            value = getattr(self, counter)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{counter} must be a non-negative int, got {value!r}")
        for name in ("started_at", "ended_at"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, ensure_utc(value))
        object.__setattr__(self, "roots", tuple(self.roots))

    def as_dict(self) -> dict[str, object]:
        """JSON-serialisable snapshot (timestamps ISO-8601 UTC)."""
        return {
            "hydrated": self.hydrated,
            "states_processed": self.states_processed,
            "orders_submitted": self.orders_submitted,
            "orders_rejected": self.orders_rejected,
            "halted": self.halted,
            "locked": self.locked,
            "last_error": self.last_error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "roots": list(self.roots),
        }


class TradingRunner:
    """Supervised live/paper trading loop with mandatory cold-start hydration.

    Parameters
    ----------
    settings:
        Runtime settings (risk limits, hydration configuration, environment).
    pipeline:
        FinRL-X strategy pipeline (``run(state) -> PipelineResult``).
    streamer:
        Live market-data streamer emitting immutable market states.
    broker:
        Target broker (Phase-5 OMS): positions + ``submit_targets``.
    hydrator:
        Cold-start hydrator; always executed before the first streamed state.
    instrument_master:
        Instrument master used to resolve the default root universe.
    translator:
        Optional ``w -> N`` translator used when the pipeline does not produce
        contract targets itself.
    event_bus:
        Telemetry bus. ``None`` disables publishing (tests/backtests).
    state_store:
        Durable risk-state store; restores the kill switch and persists the
        runner-owned starting-daily-NLV anchor via load-modify-save.
    clock:
        Injectable UTC clock (deterministic tests).
    telemetry_topic:
        Topic the runner publishes :class:`~src.core.event_bus.TelemetryEvent`s on.
    halt_on_error:
        Stop the loop on the first per-state failure (production default is to
        keep the process alive and report; operators can raise this).
    dry_run:
        Never connect/disconnect the streamer or broker (injected fakes).
    """

    def __init__(
        self,
        *,
        settings: Settings,
        pipeline: StrategyPipeline,
        streamer: MarketDataStreamer,
        broker: TargetBroker,
        hydrator: ColdStartHydrator,
        instrument_master: InstrumentsConfig | None = None,
        translator: ContractTranslatorProtocol | None = None,
        event_bus: EventBus[TelemetryEvent] | None = None,
        state_store: RiskStateStore | None = None,
        clock: Clock | None = None,
        telemetry_topic: str = DEFAULT_TELEMETRY_TOPIC,
        halt_on_error: bool = False,
        dry_run: bool = False,
        prime_pipeline: bool = False,
    ) -> None:
        if not isinstance(settings, Settings):
            raise TypeError(f"settings must be a Settings, got {type(settings)!r}")
        if not telemetry_topic:
            raise ValueError("telemetry_topic must be a non-empty string")
        self._settings = settings
        self._pipeline = pipeline
        self._streamer = streamer
        self._broker = broker
        self._hydrator = hydrator
        self._master = instrument_master
        self._translator = translator
        self._event_bus = event_bus
        self._state_store = state_store
        self._clock: Clock = clock if clock is not None else _utc_now
        self._telemetry_topic = telemetry_topic
        self._halt_on_error = halt_on_error
        self._dry_run = dry_run
        self._prime_pipeline = prime_pipeline
        self._hydrated = False
        self._hydration: HydrationResult | None = None
        self._persisted: PersistedRiskState | None = None
        self._states_processed = 0
        self._orders_submitted = 0
        self._orders_rejected = 0
        self._halted = False
        self._stop_requested = False
        self._last_error: str | None = None
        self._started_at: datetime | None = None
        self._ended_at: datetime | None = None

    # -- introspection ---------------------------------------------------------

    @property
    def settings(self) -> Settings:
        """Runtime settings the runner was built with."""
        return self._settings

    @property
    def pipeline(self) -> StrategyPipeline:
        """Strategy pipeline driven once per streamed state."""
        return self._pipeline

    @property
    def streamer(self) -> MarketDataStreamer:
        """Market-data streamer the loop consumes."""
        return self._streamer

    @property
    def broker(self) -> TargetBroker:
        """Target broker (positions + target submission)."""
        return self._broker

    @property
    def hydrator(self) -> ColdStartHydrator:
        """Cold-start hydrator used by :meth:`warm_start`."""
        return self._hydrator

    @property
    def instrument_master(self) -> InstrumentsConfig:
        """Instrument master used to resolve the default root universe."""
        if self._master is None:
            self._master = load_instruments()
        return self._master

    @property
    def hydrated(self) -> bool:
        """Whether :meth:`warm_start` completed successfully."""
        return self._hydrated

    @property
    def hydration(self) -> HydrationResult | None:
        """Latest successful hydration result (``None`` before warm start)."""
        return self._hydration

    @property
    def persisted_state(self) -> PersistedRiskState | None:
        """Last durable risk state observed/persisted by the runner."""
        return self._persisted

    @property
    def halted(self) -> bool:
        """Whether the loop stopped because of a kill switch or an error."""
        return self._halted

    @property
    def status(self) -> RunnerStatus:
        """Immutable snapshot of the runner's counters and lifecycle."""
        return RunnerStatus(
            hydrated=self._hydrated,
            states_processed=self._states_processed,
            orders_submitted=self._orders_submitted,
            orders_rejected=self._orders_rejected,
            halted=self._halted,
            locked=self._cached_lock_reason() is not None,
            last_error=self._last_error,
            started_at=self._started_at,
            ended_at=self._ended_at,
            roots=self._hydration.roots if self._hydration is not None else (),
        )

    # -- operator controls -----------------------------------------------------

    def stop(self) -> None:
        """Ask the loop to exit after the state currently being processed."""
        self._stop_requested = True

    # -- hydration -------------------------------------------------------------

    def warm_start(
        self,
        roots: Sequence[InstrumentId] | None = None,
        *,
        equity: float | None = None,
    ) -> HydrationResult:
        """Hydrate the warm-up window before any live bar may be processed.

        Raises
        ------
        HydrationError
            Propagated from the hydrator: a cold pipeline must never trade.
        """
        resolved = self._resolve_roots(roots)
        hydration = self._hydrator.hydrate(resolved, self._clock(), equity=equity)
        self._hydration = hydration
        self._hydrated = True
        logger.info(
            "warm start complete: %d states, %d bars, roots=%s, backfilled=%s",
            len(hydration.states),
            hydration.bar_count,
            list(hydration.roots),
            hydration.backfilled,
        )
        return hydration

    def _resolve_roots(self, roots: Sequence[InstrumentId] | None) -> tuple[InstrumentId, ...]:
        """Normalise an explicit root list or fall back to standard roots."""
        if roots is None:
            master = self.instrument_master
            default = tuple(spec.root for spec in master.instruments.values() if not spec.is_micro)
            if not default:
                raise RunnerConfigError(
                    "cannot resolve a default root universe: the instrument master declares "
                    "no standard (non-micro) root"
                )
            return default
        normalised = tuple(sorted({root.strip().upper() for root in roots}))
        if not normalised or any(not root for root in normalised):
            raise RunnerConfigError("at least one non-empty futures root is required")
        return normalised

    # -- main loop -------------------------------------------------------------

    async def run(
        self,
        *,
        roots: Sequence[InstrumentId] | None = None,
        max_states: int | None = None,
    ) -> RunnerStatus:
        """Warm start, restore risk state, stream states and trade until stopped.

        Hydration failures deliberately propagate (:class:`HydrationError`): a
        runner that cannot warm its pipeline must be loud, not silently idle.
        Per-state trading failures are isolated and published on telemetry.
        """
        if max_states is not None and (isinstance(max_states, bool) or max_states < 1):
            raise ValueError("max_states must be a positive int or None")
        self._reset_counters()
        self._started_at = self._clock()
        self._ended_at = None
        try:
            await self._publish_telemetry(
                "runner_started",
                {
                    "environment": self._settings.environment,
                    "dry_run": self._dry_run,
                    "halt_on_error": self._halt_on_error,
                },
            )
            hydration = self.warm_start(roots)
            await self._publish_telemetry(
                "hydration",
                {
                    "roots": ",".join(hydration.roots),
                    "states": len(hydration.states),
                    "bars": hydration.bar_count,
                    "coverage": sum(hydration.coverage.values()),
                    "required_bars": hydration.required_bars,
                    "lookback_days": hydration.lookback_days,
                    "backfilled": hydration.backfilled,
                    "warnings": len(hydration.warnings),
                },
            )
            self._load_persisted()
            lock_reason = self._current_lock_reason()
            if lock_reason is not None:
                self._halted = True
                await self._publish_telemetry(
                    "kill_switch", {"locked": True, "reason": lock_reason}
                )
                logger.critical(
                    "runner refusing to trade: persisted kill switch is engaged (%s)", lock_reason
                )
            else:
                await self._maybe_connect(self._streamer)
                await self._maybe_connect(self._broker)
                if self._prime_pipeline:
                    await self._prime_pipeline_once(hydration)
                await self._stream_loop(max_states)
        except asyncio.CancelledError:
            self._halted = True
            raise
        except HydrationError:
            self._halted = True
            raise
        except Exception as exc:
            self._record_error("run", exc)
            await self._publish_telemetry(
                "error",
                {
                    "phase": "run",
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:512],
                },
            )
            self._halted = True
        finally:
            await self._shutdown()
            self._ended_at = self._clock()
            await self._publish_telemetry(
                "runner_stopped",
                {
                    "states_processed": self._states_processed,
                    "orders_submitted": self._orders_submitted,
                    "halted": self._halted,
                    "hydrated": self._hydrated,
                    "ended_at": self._ended_at.isoformat() if self._ended_at else None,
                },
            )
        return self.status

    async def _stream_loop(self, max_states: int | None) -> None:
        """Consume the streamer sequentially, isolating per-state failures."""
        processed = 0
        async for state in self._streamer.stream():
            if self._stop_requested or self._halted:
                break
            if max_states is not None and processed >= max_states:
                break
            processed += 1
            try:
                await self._process_state(state)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._record_error("state", exc)
                await self._publish_telemetry(
                    "error",
                    {
                        "phase": "state",
                        "timestamp": state.timestamp.isoformat(),
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:512],
                    },
                )
                if self._halt_on_error:
                    self._halted = True
                    break

    async def run_once(self, state: MarketState) -> list[OrderReceipt]:
        """Process exactly one state (deterministic tests / replay tooling).

        Hydration is mandatory — ``run_once`` raises
        :class:`RunnerNotHydratedError` until :meth:`warm_start` has succeeded.
        The persisted kill switch is refreshed first, so a lock engaged by
        another process is honoured immediately.
        """
        if not self._hydrated:
            raise RunnerNotHydratedError(
                "TradingRunner.run_once() called before warm_start(); the pipeline would "
                "observe cold features. Call warm_start(roots) first (it hydrates the "
                "historical lookback window before any live state is processed)."
            )
        if self._started_at is None:
            self._started_at = self._clock()
        self._load_persisted()
        receipts = await self._process_state(state)
        self._ended_at = self._clock()
        return receipts

    # -- account-equity enrichment / warm pipeline prime -----------------------

    async def _state_with_account_equity(self, state: MarketState) -> MarketState:
        """Fill a missing ``state.equity`` from the broker account (base currency).

        Live market-data states carry prices/features but not the account NLV;
        the translator needs ``E_t`` to size contracts. Only *missing* equity is
        filled (a state that already carries equity is never overridden), and the
        account base currency + FX table from settings are attached so the
        Phase-7 base-currency normalisation sees the real IBKR denomination.
        """
        if state.equity is not None and state.equity > 0.0:
            return state
        getter = getattr(self._broker, "get_account", None)
        if not callable(getter):
            return state
        try:
            account = await cast("Callable[[], Awaitable[AccountSnapshot]]", getter)()
        except Exception as exc:
            logger.warning("could not fetch the broker account for equity enrichment: %s", exc)
            return state
        equity = float(account.net_liquidation)
        if not math.isfinite(equity) or equity <= 0.0:
            return state
        return replace(
            state,
            equity=equity,
            account_currency=self._settings.account_currency,
            fx_rates=dict(self._settings.fx_rates),
        )

    async def _prime_pipeline_once(self, hydration: HydrationResult) -> None:
        """Run the pipeline once on the warm historical state, before any live bar.

        Optional production warm-up (``prime_pipeline=True``): it exercises the
        full ``S -> A -> T -> R -> N`` chain on the hydrated window, anchors the
        persisted ``starting_daily_nlv`` from the broker account and proves the
        pipeline is warm on the very first live bar. It never submits orders.
        """
        state = await self._state_with_account_equity(hydration.final_state)
        result = self._pipeline.run(state)
        self._ensure_daily_anchor(state)
        self._persisted = self._persisted  # keep property semantics explicit
        await self._publish_telemetry(
            "pipeline_warmup",
            {
                "timestamp": state.timestamp.isoformat(),
                "universe": ",".join(result.universe),
                "final_weights": len(result.final_weights),
                "gross_leverage": sum(
                    abs(weight) for weight in result.final_weights.weights.values()
                ),
                "account_currency": state.account_currency,
                "equity": state.equity,
            },
        )
        logger.info(
            "pipeline warm-up complete on %s: universe=%d gross=%.4f",
            state.timestamp.isoformat(),
            len(result.universe),
            sum(abs(weight) for weight in result.final_weights.weights.values()),
        )

    # -- per-state pipeline ----------------------------------------------------

    async def _process_state(self, state: MarketState) -> list[OrderReceipt]:
        """Run the pipeline for one state and submit the resulting targets."""
        if not self._hydrated:
            raise RunnerNotHydratedError(
                "refusing to process a MarketState before warm_start(); cold-start hydration "
                "is mandatory before the first live bar"
            )
        if not isinstance(state, MarketState):
            raise TypeError(f"state must be a MarketState, got {type(state)!r}")
        state = await self._state_with_account_equity(state)
        self._states_processed += 1
        await self._publish_telemetry(
            "state",
            {
                "timestamp": state.timestamp.isoformat(),
                "instruments": ",".join(state.ids),
                "equity": state.equity,
                "account_currency": state.account_currency,
            },
        )
        result = self._pipeline.run(state)
        await self._publish_telemetry(
            "pipeline",
            {
                "timestamp": result.timestamp.isoformat(),
                "universe": ",".join(result.universe),
                "final_weights": len(result.final_weights),
                "gross_leverage": sum(
                    abs(weight) for weight in result.final_weights.weights.values()
                ),
                "elapsed_ms": result.elapsed_ms,
                "targets": len(result.targets) if result.targets is not None else None,
            },
        )
        targets = result.targets
        if targets is None and self._translator is not None:
            # The translator resolves the base currency/FX table from ``state``
            # itself (Phase-7 base-currency normalisation), so only the explicit
            # equity override is threaded through here.
            targets = self._translator.translate(result.final_weights, state, equity=state.equity)
        if targets is None:
            self._ensure_daily_anchor(state)
            await self._publish_telemetry(
                "orders", {"timestamp": state.timestamp.isoformat(), "submitted": 0}
            )
            return []
        lock_reason = self._current_lock_reason()
        if lock_reason is not None:
            self._halted = True
            await self._publish_telemetry("kill_switch", {"locked": True, "reason": lock_reason})
            logger.critical("kill switch engaged (%s); not submitting targets", lock_reason)
            return []
        self._ensure_daily_anchor(state)
        positions = await self._broker.get_positions()
        current = {
            instrument_id: int(position.quantity)
            for instrument_id, position in positions.items()
            if int(position.quantity) != 0
        }
        receipts = await self._submit_targets(targets, current, state.timestamp)
        accepted = [receipt for receipt in receipts if receipt.status is not OrderStatus.REJECTED]
        rejected = len(receipts) - len(accepted)
        self._orders_submitted += len(accepted)
        self._orders_rejected += rejected
        summary = await self._account_summary()
        await self._publish_telemetry(
            "orders",
            {
                "timestamp": state.timestamp.isoformat(),
                "targets": len(targets),
                "orders": len(receipts),
                "accepted": len(accepted),
                "rejected": rejected,
                "order_ids": ",".join(receipt.order_id for receipt in receipts)[:512],
                **summary,
            },
        )
        logger.info(
            "state %s processed: %d targets, %d orders accepted, %d rejected",
            state.timestamp.isoformat(),
            len(targets),
            len(accepted),
            rejected,
        )
        return receipts

    async def _submit_targets(
        self,
        targets: ContractTargetSet,
        current_positions: Mapping[InstrumentId, int],
        timestamp: datetime,
    ) -> list[OrderReceipt]:
        """Submit ``targets`` through the broker's target or order surface."""
        submit = getattr(self._broker, "submit_targets", None)
        if callable(submit):
            caller = cast("SubmitTargetsFn", submit)
            return list(await caller(targets, current_positions, timestamp))
        orders_from_targets = getattr(self._broker, "orders_from_targets", None)
        submit_orders = getattr(self._broker, "submit_orders", None)
        if callable(orders_from_targets) and callable(submit_orders):
            builder = cast(
                "Callable[[ContractTargetSet, Mapping[InstrumentId, int], datetime], list[OrderEvent]]",
                orders_from_targets,
            )
            sender = cast(
                "Callable[[Sequence[OrderEvent]], Awaitable[list[OrderReceipt]]]", submit_orders
            )
            orders = builder(targets, current_positions, timestamp)
            return list(await sender(orders))
        raise RunnerConfigError(
            "broker exposes neither submit_targets(targets, current_positions, timestamp) nor "
            "orders_from_targets(...)+submit_orders(...); cannot route the approved targets"
        )

    async def _account_summary(self) -> dict[str, ScalarValue]:
        """Publish-able account/PNL/margin scalars from the broker (best effort)."""
        getter = getattr(self._broker, "get_account", None)
        if not callable(getter):
            return {}
        try:
            account = await cast("Callable[[], Awaitable[AccountSnapshot]]", getter)()
        except Exception as exc:
            logger.warning("could not fetch the broker account for telemetry: %s", exc)
            return {"account_error": f"{type(exc).__name__}: {exc}"}
        return {
            "net_liquidation": account.net_liquidation,
            "cash_balance": account.cash_balance,
            "initial_margin_used": account.initial_margin_used,
            "margin_utilization": account.margin_utilization,
        }

    # -- persisted risk state (runner-owned fields only) -----------------------

    def _load_persisted(self) -> PersistedRiskState | None:
        """Load the durable risk state (``None`` when unset/unavailable)."""
        store = self._state_store
        if store is None:
            return None
        try:
            self._persisted = store.load()
        except Exception as exc:
            logger.exception("could not load persisted risk state; continuing without it")
            self._last_error = f"risk state load failed: {type(exc).__name__}: {exc}"
            return None
        return self._persisted

    def _current_lock_reason(self) -> str | None:
        """Reason the kill switch is engaged, or ``None`` when trading is allowed.

        Checks both the durable store (a lock latched by a previous process or the
        risk overlay) and the broker/OMS gate (a lock latched in this process).
        """
        store = self._state_store
        if store is not None:
            try:
                current = store.load()
            except Exception as exc:
                logger.exception("could not refresh persisted risk state")
                self._last_error = f"risk state load failed: {type(exc).__name__}: {exc}"
            else:
                if current is not None:
                    self._persisted = current
                    if current.kill_switch_locked:
                        return current.kill_switch_reason or "persisted kill switch locked"
        if bool(getattr(self._broker, "is_locked", False)):
            gate = getattr(self._broker, "gate", None)
            reason = str(getattr(gate, "lock_reason", "") or "") if gate is not None else ""
            return reason or "broker kill switch engaged"
        return None

    def _cached_lock_reason(self) -> str | None:
        """Kill-switch reason from already-observed state (no store I/O)."""
        persisted = self._persisted
        if persisted is not None and persisted.kill_switch_locked:
            return persisted.kill_switch_reason or "persisted kill switch locked"
        if bool(getattr(self._broker, "is_locked", False)):
            gate = getattr(self._broker, "gate", None)
            reason = str(getattr(gate, "lock_reason", "") or "") if gate is not None else ""
            return reason or "broker kill switch engaged"
        return None

    def _ensure_daily_anchor(self, state: MarketState) -> None:
        """Persist ``starting_daily_nlv`` once per session (runner-owned field)."""
        if self._state_store is None:
            return
        persisted = self._persisted
        if persisted is not None and persisted.starting_daily_nlv is not None:
            return
        equity = state.equity
        if equity is None or equity <= 0.0:
            return

        def mutator(current: PersistedRiskState) -> PersistedRiskState:
            if current.starting_daily_nlv is not None:
                return current
            return current.updated(
                starting_daily_nlv=float(equity),
                trading_day=current.trading_day or state.timestamp.date(),
            )

        updated = self._update_persisted(mutator)
        if updated is not None:
            logger.info(
                "persisted starting_daily_nlv=%.2f for trading day %s",
                updated.starting_daily_nlv or 0.0,
                updated.trading_day,
            )

    def _update_persisted(
        self,
        mutator: Callable[[PersistedRiskState], PersistedRiskState],
    ) -> PersistedRiskState | None:
        """Load-modify-save the durable risk state, preserving unrelated fields.

        The risk overlay/API persist their own fields (high-water marks, daily
        loss latch) concurrently; re-reading before writing keeps those intact.
        """
        store = self._state_store
        if store is None:
            return None
        try:
            current = store.load()
            base = current if current is not None else PersistedRiskState()
            updated = mutator(base)
            stamped = updated.updated(updated_at=self._clock())
            store.save(stamped)
        except Exception as exc:
            logger.exception("could not persist runner-owned risk state")
            self._last_error = f"risk state persistence failed: {type(exc).__name__}: {exc}"
            return None
        self._persisted = stamped
        return stamped

    # -- lifecycle helpers -----------------------------------------------------

    async def _maybe_connect(self, target: object) -> None:
        """Connect ``target`` unless this is an injected dry run."""
        if self._dry_run:
            return
        connect = getattr(target, "connect", None)
        if callable(connect):
            await cast("Callable[[], Awaitable[None]]", connect)()

    async def _disconnect(self, target: object) -> None:
        """Disconnect ``target``, isolating shutdown failures."""
        disconnect = getattr(target, "disconnect", None)
        if not callable(disconnect):
            return
        try:
            await cast("Callable[[], Awaitable[None]]", disconnect)()
        except Exception:
            logger.exception("graceful shutdown failed for %s", type(target).__name__)

    async def _shutdown(self) -> None:
        """Gracefully release the streamer and broker sessions."""
        if self._dry_run:
            return
        await self._disconnect(self._streamer)
        await self._disconnect(self._broker)

    async def shutdown(self) -> None:
        """Public graceful-shutdown hook (disconnects streamer and broker)."""
        self.stop()
        await self._shutdown()

    def _record_error(self, phase: str, exc: BaseException) -> None:
        """Record and log a failure; telemetry is published by the caller."""
        message = f"{phase} failed: {type(exc).__name__}: {exc}"
        self._last_error = message
        logger.exception("trading runner %s failed", phase)

    def _reset_counters(self) -> None:
        self._states_processed = 0
        self._orders_submitted = 0
        self._orders_rejected = 0
        self._halted = False
        self._stop_requested = False
        self._last_error = None

    async def _publish_telemetry(
        self, kind: str, values: Mapping[str, ScalarValue] | None = None
    ) -> None:
        """Publish one telemetry event (handler failures are isolated by the bus)."""
        bus = self._event_bus
        if bus is None:
            return
        event = TelemetryEvent(
            kind=kind,
            timestamp_ns=int(self._clock().timestamp() * 1_000_000_000),
            values=dict(values or {}),
        )
        try:
            await bus.publish(self._telemetry_topic, event)
        except Exception:  # pragma: no cover - the bus already isolates handlers
            logger.exception("telemetry publish failed on topic %r", self._telemetry_topic)


# ---------------------------------------------------------------------------
# Offline execution stack (``--offline`` determinism, Phase 8)
# ---------------------------------------------------------------------------


class _SimulatedBrokerLike(Protocol):
    """Subset of :class:`src.backtest.simulated_broker.SimulatedBroker` used offline."""

    @property
    def is_connected(self) -> bool:
        """Whether the simulated session is connected."""
        ...

    async def connect(self) -> None:
        """Mark the simulated session connected."""
        ...

    async def disconnect(self) -> None:
        """Tear the simulated session down."""
        ...

    def get_positions_sync(self) -> Mapping[InstrumentId, Position]:
        """Current simulated positions keyed by futures root."""
        ...

    def get_account_sync(self, timestamp: datetime | None = None) -> AccountSnapshot:
        """Simulated account/valuation snapshot."""
        ...

    def set_market(
        self,
        timestamp: datetime,
        quotes: Any,
    ) -> None:
        """Install the bar quotes used by the matching engine."""
        ...

    def place_order(self, order: OrderEvent) -> OrderReceipt:
        """Match one order against the current bar."""
        ...

    async def cancel_all(self) -> list[OrderReceipt]:
        """Cancel every working order."""
        ...

    async def flatten_all(self) -> list[OrderReceipt]:
        """Flatten the book and cancel working orders."""
        ...


class OfflineMarketStateStreamer:
    """Deterministic offline ``MarketDataStreamer`` for ``--offline`` runs.

    Replays the seeded :class:`~src.data.databento_loader.OfflineBarSource`
    history through the canonical feature processor and the point-in-time state
    builder: the last ``replay_states`` timestamps of the history window are
    emitted as live-loop states (all earlier history is used to warm the
    features/covariance, exactly like a real session). Every state carries the
    canonical bar rows of its timestamp so an injected broker can mark the book.

    No network, no credentials, no wall clock: the simulation window is a fixed
    constant, so the whole offline loop is reproducible across machines and days.
    """

    def __init__(
        self,
        *,
        roots: Sequence[InstrumentId],
        start: datetime,
        end: datetime,
        equity: float,
        instrument_master: InstrumentsConfig,
        replay_states: int = OFFLINE_REPLAY_STATES,
        on_state: Callable[[MarketState, pl.DataFrame], None] | None = None,
    ) -> None:
        normalised = tuple(sorted({root.strip().upper() for root in roots}))
        if not normalised or any(not root for root in normalised):
            raise ValueError("offline streamer requires at least one non-empty futures root")
        if end <= start:
            raise ValueError("offline streamer requires end after start")
        if not math.isfinite(equity) or equity <= 0.0:
            raise ValueError("offline streamer equity must be finite and strictly positive")
        if isinstance(replay_states, bool) or replay_states < 1:
            raise ValueError("replay_states must be a positive int")
        self._roots = normalised
        self._start = ensure_utc(start)
        self._end = ensure_utc(end)
        self._equity = float(equity)
        self._master = instrument_master
        self._replay_states = replay_states
        self._on_state = on_state
        self._connected = False
        self._cache: list[tuple[MarketState, pl.DataFrame]] | None = None

    # -- introspection ---------------------------------------------------------

    @property
    def roots(self) -> tuple[InstrumentId, ...]:
        """Roots replayed by this streamer."""
        return self._roots

    @property
    def is_connected(self) -> bool:
        """Whether the offline session is "connected" (in-memory flag only)."""
        return self._connected

    @property
    def replay_states(self) -> int:
        """Number of states emitted by the live loop."""
        return self._replay_states

    # -- MarketDataStreamer ----------------------------------------------------

    async def connect(self) -> None:
        """Mark the offline session connected (no I/O)."""
        self._connected = True

    async def disconnect(self) -> None:
        """Mark the offline session disconnected (no I/O)."""
        self._connected = False

    async def stream(self) -> AsyncIterator[MarketState]:
        """Yield the deterministic replay states in chronological order."""
        for state, bars in self._states():
            if self._on_state is not None:
                self._on_state(state, bars)
            yield state

    # -- internals -------------------------------------------------------------

    def _states(self) -> list[tuple[MarketState, pl.DataFrame]]:
        """Build (and memoise) the replay states from the synthetic history."""
        if self._cache is not None:
            return self._cache
        from src.data.databento_loader import OfflineBarSource

        source = OfflineBarSource(self._roots, periods=2, start=self._start)
        bars = source.load(self._roots, self._start, self._end + timedelta(microseconds=1))
        bars = add_neutral_term_structure(bars, context="offline-replay")
        processor = FeatureProcessor(instrument_master=self._master)
        features = processor.compute(bars)
        timestamps = sorted(
            {
                value
                for value in features[s.TIMESTAMP_COLUMN].to_list()
                if isinstance(value, datetime)
            }
        )
        if not timestamps:
            raise RunnerConfigError(
                "offline replay produced no bars; check the offline simulation window"
            )
        decision_times = timestamps[-self._replay_states :]
        builder = MarketStateBuilder(
            instrument_master=self._master,
            feature_processor=processor,
            initial_equity=self._equity,
        )
        states = builder.build_series(features, decision_times)
        frames = {
            stamp: bars.filter(pl.col(s.TIMESTAMP_COLUMN) == pl.lit(stamp))
            for stamp in decision_times
        }
        cache: list[tuple[MarketState, pl.DataFrame]] = []
        for state in states:
            frame = frames.get(state.timestamp)
            if frame is None:  # pragma: no cover - timestamps are built from the frame
                raise RunnerConfigError(
                    f"offline replay has no bar rows for state {state.timestamp.isoformat()}"
                )
            cache.append((state, frame))
        self._cache = cache
        return cache


class OfflineTargetBroker:
    """``TargetBroker`` adapter over the simulated venue for ``--offline`` mode.

    Combines :class:`~src.backtest.simulated_broker.SimulatedBroker` (matching,
    accounting, margin, costs) with the Phase-5 deterministic
    :func:`~src.execution.oms.build_orders_from_targets` delta translation and
    exposes the async ``get_positions`` / ``get_account`` / ``submit_targets``
    surface :class:`TradingRunner` drives. It therefore exercises the *same*
    runner loop and order-delta logic as the live OMS, without any network.
    """

    #: Broker identifier reported by the runner telemetry.
    name = "offline-simulated"

    def __init__(
        self,
        *,
        simulated_broker: _SimulatedBrokerLike,
        instrument_master: InstrumentsConfig,
    ) -> None:
        self._broker = simulated_broker
        self._master = instrument_master
        self._last_state: MarketState | None = None

    # -- introspection ---------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        """Whether the underlying simulated session is connected."""
        return self._broker.is_connected

    @property
    def is_locked(self) -> bool:
        """The offline adapter never latches a kill switch (the runner still checks)."""
        return False

    @property
    def last_market_state(self) -> MarketState | None:
        """Most recent market state installed by the offline streamer."""
        return self._last_state

    # -- BrokerAdapter lifecycle -----------------------------------------------

    async def connect(self) -> None:
        """Connect the simulated session."""
        await self._broker.connect()

    async def disconnect(self) -> None:
        """Disconnect the simulated session."""
        await self._broker.disconnect()

    # -- account / positions ---------------------------------------------------

    async def get_positions(self) -> Mapping[InstrumentId, Position]:
        """Current simulated positions keyed by futures root."""
        return self._broker.get_positions_sync()

    async def get_account(self) -> AccountSnapshot:
        """Simulated account valuation/margin snapshot."""
        return self._broker.get_account_sync()

    # -- market data / execution ----------------------------------------------

    def update_market(self, state: MarketState, bars: pl.DataFrame) -> None:
        """Install the bar quotes of ``state`` into the simulated matching engine."""
        from src.backtest.simulated_broker import BarQuote

        self._last_state = state
        quotes: list[BarQuote] = []
        for root in state.ids:
            rows = bars.filter(
                (pl.col(s.ROOT_SYMBOL_COLUMN) == root)
                & (pl.col(s.TIMESTAMP_COLUMN) == pl.lit(state.timestamp))
            ).sort(s.EXPIRATION_COLUMN)
            if rows.is_empty():
                continue
            record = rows.row(0, named=True)
            expiration = (
                record.get(s.EXPIRATION_COLUMN)
                if isinstance(record.get(s.EXPIRATION_COLUMN), date)
                else None
            )
            snapshot = state.snapshot(root)
            settlement = snapshot.settlement
            for quote_root in self._quote_roots(root):
                quotes.append(
                    BarQuote(
                        root=quote_root,
                        open=float(record[s.OPEN_COLUMN]),
                        high=float(record[s.HIGH_COLUMN]),
                        low=float(record[s.LOW_COLUMN]),
                        close=float(record[s.CLOSE_COLUMN]),
                        volume=float(record[s.VOLUME_COLUMN]),
                        bid=snapshot.bid if quote_root == root else None,
                        ask=snapshot.ask if quote_root == root else None,
                        expiration=expiration,
                        settlement=(
                            settlement
                            if quote_root == root and settlement is not None and settlement > 0
                            else None
                        ),
                    )
                )
        if quotes:
            self._broker.set_market(state.timestamp, quotes)

    def _quote_roots(self, root: InstrumentId) -> tuple[InstrumentId, ...]:
        """Roots to mark for ``root``: itself plus its paired micro contract.

        The translator may auto-route a standard leg's residual into the micro
        root (``ES -> MES``) even when the market state only carries the standard
        snapshot (the pair shares the same underlying), so the simulated venue
        must accept order flow for the micro contract too.
        """
        if not self._master.has(root):
            return (root,)
        micro = self._master.get(root).micro_root
        if micro is None or micro == root:
            return (root,)
        return (root, micro)

    async def submit_targets(
        self,
        targets: ContractTargetSet,
        current_positions: Mapping[InstrumentId, int],
        timestamp: datetime,
        *,
        flatten: bool = False,
    ) -> list[OrderReceipt]:
        """Translate targets into delta orders and match them in the simulator."""
        from src.execution.oms import build_orders_from_targets

        orders = build_orders_from_targets(targets, current_positions, timestamp, flatten=flatten)
        return [self._broker.place_order(order) for order in orders]

    async def cancel_all(self) -> list[OrderReceipt]:
        """Cancel every working simulated order."""
        return await self._broker.cancel_all()

    async def flatten_all(self) -> list[OrderReceipt]:
        """Flatten the simulated book."""
        return await self._broker.flatten_all()


def _utc_now() -> datetime:
    return datetime.now(UTC)


# ---------------------------------------------------------------------------
# CLI: configuration check (offline) + production wiring (lazy heavy imports)
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="futuresrl-runner",
        description=(
            "Cold-start hydrated live/paper trading runner. Use --check to validate the "
            "configuration offline without opening any IBKR/Databento connection, or "
            "--offline to run the deterministic synthetic pipeline end-to-end."
        ),
    )
    parser.add_argument(
        "--strategy",
        default=None,
        help="declarative strategy YAML (default: configs/strategies/tsmom_vol_target.yaml)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate settings/instrument master/strategy/hydration config and exit",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help=(
            "run the deterministic synthetic pipeline end-to-end (no IBKR/Databento, "
            "fixed simulation window, simulated broker)"
        ),
    )
    parser.add_argument(
        "--backfill-source",
        choices=list(HYDRATION_SOURCES),
        default=None,
        help=(
            "historical backfill source: auto = Databento -> IBKR -> offline (only when "
            "allowed); databento/ibkr/offline force one source "
            "(default: Settings.hydration_source)"
        ),
    )
    parser.add_argument(
        "--roots",
        default=None,
        help="comma-separated futures roots to trade (default: every standard root)",
    )
    parser.add_argument(
        "--max-states",
        type=int,
        default=None,
        help="stop after N streamed states (default: run until stopped)",
    )
    parser.add_argument("--log-level", default=None, help="override the configured log level")
    return parser


def _parse_roots(raw: str | None) -> tuple[InstrumentId, ...] | None:
    """Parse ``--roots ES,NQ`` into a normalised tuple."""
    if raw is None or not raw.strip():
        return None
    roots = tuple(part.strip().upper() for part in raw.split(",") if part.strip())
    if not roots:
        raise RunnerConfigError("--roots was provided but contains no root symbols")
    return roots


def _strategy_path(settings: Settings, raw: str | None) -> str | None:
    """Resolve the strategy YAML path (explicit flag or the default template)."""
    if raw is not None:
        return raw
    candidate = settings.strategies_dir / DEFAULT_STRATEGY_FILE
    return str(candidate) if candidate.is_file() else None


# ---------------------------------------------------------------------------
# Hydration source resolution (Phase 8 multi-vendor fallback)
# ---------------------------------------------------------------------------


def _resolve_hydration_source(settings: Settings, source: str | None) -> str:
    """Resolve ``--backfill-source`` / ``Settings.hydration_source`` to a selector."""
    resolved = str(source or settings.hydration_source).strip().lower()
    if resolved not in HYDRATION_SOURCES:
        raise HydrationError(
            f"unknown hydration source {source!r}; expected one of {list(HYDRATION_SOURCES)}. "
            + HYDRATION_SOURCE_HINTS
        )
    return resolved


def _databento_configured(settings: Settings) -> bool:
    """Whether a Databento API key is configured (``FUTURESRLX_DATABENTO_API_KEY``)."""
    return settings.databento_api_key_value is not None


def _ibkr_configured(settings: Settings) -> bool:
    """Whether IBKR historical data can be used (paper/live environment configured)."""
    return settings.environment in {"paper", "live"}


def describe_hydration_sources(
    settings: Settings,
    *,
    source: str | None = None,
    offline_allowed: bool = False,
) -> dict[str, object]:
    """Describe the resolved multi-source hydration fallback (``--check`` output).

    Returns the requested selector, the ordered list of sources that would be
    tried, per-source availability, the reasons unavailable sources are skipped
    and the actionable operator hints. Pure configuration inspection: nothing is
    connected and no network call is made.
    """
    resolved = _resolve_hydration_source(settings, source)
    available: dict[str, bool] = {
        "databento": _databento_configured(settings),
        "ibkr": _ibkr_configured(settings),
        "offline": bool(offline_allowed or resolved == "offline"),
    }
    reasons: dict[str, str] = {}
    if not available["databento"]:
        reasons["databento"] = "FUTURESRLX_DATABENTO_API_KEY is not configured"
    if not available["ibkr"]:
        reasons["ibkr"] = (
            "IBKR historical data requires FUTURESRLX_ENVIRONMENT=paper|live with TWS or IB "
            "Gateway running and FUTURESRLX_IBKR_PORT set (paper 7497/4002, live 7496/4001)"
        )
    if not available["offline"]:
        reasons["offline"] = (
            "offline synthetic history is only used with --offline or --backfill-source offline"
        )
    if resolved == "auto":
        order = [name for name in ("databento", "ibkr", "offline") if available[name]]
    else:
        order = [resolved]
    return {
        "requested": resolved,
        "order": order,
        "available": available,
        "unavailable_reasons": reasons,
        "databento_api_key_configured": available["databento"],
        "ibkr_configured": available["ibkr"],
        "offline_allowed": available["offline"],
        "hints": HYDRATION_SOURCE_HINTS,
    }


def build_hydration_chain(
    settings: Settings,
    *,
    repository: ParquetRepository,
    instrument_master: InstrumentsConfig,
    source: str | None = None,
    offline_allowed: bool = False,
    databento_fetch: BackfillFn | None = None,
    ibkr_fetch: BackfillFn | None = None,
    offline_fetch: BackfillFn | None = None,
) -> MultiSourceBackfill:
    """Build the ordered multi-vendor history fallback chain for hydration.

    ``auto`` order: Databento (API key configured) -> IBKR historical (paper/live
    configured) -> offline synthetic history (only when ``offline_allowed``). A
    forced selector fills only that vendor; when it is unavailable the chain
    builder fails closed with :data:`HYDRATION_SOURCE_HINTS`. The vendor callables
    are injectable so tests never construct a real Databento/IBKR client.
    """
    resolved = _resolve_hydration_source(settings, source)
    description = describe_hydration_sources(
        settings, source=resolved, offline_allowed=offline_allowed
    )
    order = cast("list[str]", description["order"])
    available = cast("dict[str, bool]", description["available"])
    databento = databento_fetch
    if databento is None and "databento" in order and available["databento"]:
        from src.data.databento_loader import DatabentoLoader

        databento = DatabentoLoader(
            settings, repository=repository, instrument_master=instrument_master
        ).ingest
    ibkr = ibkr_fetch
    if ibkr is None and "ibkr" in order and available["ibkr"]:
        from src.data.ibkr_data import IBKRHistoricalBarFetcher

        ibkr = IBKRHistoricalBarFetcher(
            settings, instrument_master=instrument_master
        ).fetch_bars_sync
    offline = offline_fetch
    if offline is None and "offline" in order and available["offline"]:
        offline = offline_backfill
    chain = build_backfill_chain(
        databento=databento,
        ibkr=ibkr,
        offline=offline,
        source=resolved,
        offline_allowed=offline_allowed,
    )
    logger.info("hydration backfill chain resolved: %s", chain.sources or ("(none)",))
    return chain


def _check_configuration(
    settings: Settings,
    strategy_path: str | None,
    *,
    offline: bool = False,
    backfill_source: str | None = None,
) -> dict[str, object]:
    """Validate the runtime configuration without connecting to anything."""
    from configs.strategy import load_strategy_config

    from src.risk.contract_translator import ContractTranslator
    from src.risk.state_store import build_risk_state_store

    try:
        master = settings.instruments
    except (OSError, ValueError) as exc:
        raise RunnerConfigError(f"could not load the instrument master: {exc}") from exc
    repository = ParquetRepository(settings)
    # ``--offline`` implies the synthetic source unless the operator forced one.
    effective_source = (
        "offline" if offline and backfill_source in (None, "auto") else backfill_source
    )
    resolved_source = _resolve_hydration_source(settings, effective_source)
    offline_allowed = bool(offline or resolved_source == "offline")
    description = describe_hydration_sources(
        settings, source=resolved_source, offline_allowed=offline_allowed
    )
    # Building the chain only constructs vendor adapters (DatabentoLoader /
    # IBKRHistoricalBarFetcher); it never connects, so --check stays offline.
    chain = build_hydration_chain(
        settings,
        repository=repository,
        instrument_master=master,
        source=resolved_source,
        offline_allowed=offline_allowed,
    )
    summary: dict[str, object] = {
        "environment": settings.environment,
        "dry_run": True,
        "offline": bool(offline),
        "backfill_source": resolved_source,
        "instrument_roots": list(master.roots()),
        "account_currency": settings.account_currency,
        "fx_rates": dict(settings.fx_rates),
        "risk_state_store_uri": settings.risk_state_store_uri,
        "hydration": {
            "lookback_days": settings.hydration_lookback_days,
            "backfill_enabled": settings.hydration_backfill_enabled,
            "required_bars": MIN_REQUIRED_BARS,
        },
        "hydration_sources": description,
        "hydration_backfill_chain": list(chain.sources),
        "hydration_backfill_ready": bool(chain),
    }
    store = build_risk_state_store(settings.risk_state_store_uri, duckdb_path=settings.duckdb_path)
    summary["state_store"] = type(store).__name__
    persisted = store.load()
    summary["kill_switch_locked"] = bool(persisted.kill_switch_locked) if persisted else False
    if strategy_path is not None:
        try:
            config = load_strategy_config(strategy_path)
        except (OSError, ValueError) as exc:
            raise RunnerConfigError(
                f"could not load the strategy configuration ({strategy_path!r}): {exc}"
            ) from exc
        translator = ContractTranslator.from_config(config.translator, instruments=master)
        summary["strategy"] = {
            "id": config.id,
            "universe": config.universe.name,
            "allocator": config.allocator.name,
            "timing": [stage.name for stage in config.timing],
            "risk": [stage.name for stage in config.risk],
            "rounding_mode": translator.rounding_mode.value,
            "max_margin_utilization": translator.max_margin_utilization,
        }
        summary["strategy_path"] = strategy_path
    else:
        summary["strategy"] = None
    required_lookback = math.ceil(MIN_REQUIRED_BARS * 7.0 / 5.0) + CALENDAR_SAFETY_DAYS
    summary["hydration_required_lookback_days"] = required_lookback
    summary["hydration_lookback_adequate"] = settings.hydration_lookback_days >= required_lookback
    summary["ok"] = True
    return summary


# ---------------------------------------------------------------------------
# Production wiring (lazy heavy imports) and the deterministic offline loop
# ---------------------------------------------------------------------------


def _load_strategy(
    settings: Settings, strategy_path: str | None
) -> tuple[StrategyConfig, InstrumentsConfig]:
    """Load the strategy config + instrument master (shared by both run modes)."""
    from configs.strategy import load_strategy_config

    if strategy_path is None:
        raise RunnerConfigError(
            f"--strategy is required (no default strategy YAML found in {settings.strategies_dir})"
        )
    try:
        return load_strategy_config(strategy_path), settings.instruments
    except (OSError, ValueError) as exc:
        raise RunnerConfigError(
            f"could not load the strategy/instrument configuration ({strategy_path!r}): {exc}"
        ) from exc


def _run_offline(settings: Settings, args: argparse.Namespace) -> RunnerStatus:
    """Run the full runner loop against the deterministic offline stack.

    ``--offline`` must work with an empty data directory and no credentials:
    hydration backfills the synthetic history through the offline chain source,
    the streamer replays fixed warm states and the simulated broker fills the
    runner's target orders. No IBKR/Databento client is constructed.
    """
    from src.backtest.simulated_broker import SimulatedBroker
    from src.risk.contract_translator import ContractTranslator
    from src.risk.state_store import build_risk_state_store
    from src.strategies.pipeline import build_pipeline

    strategy_path = _strategy_path(settings, args.strategy)
    config, master = _load_strategy(settings, strategy_path)
    translator = ContractTranslator.from_config(
        config.translator,
        instruments=master,
        account_currency=settings.account_currency,
        fx_rates=settings.fx_rates,
    )
    pipeline = build_pipeline(config, master, translator=translator)
    repository = ParquetRepository(settings)
    state_store = build_risk_state_store(
        settings.risk_state_store_uri, duckdb_path=settings.duckdb_path
    )
    end = OFFLINE_SIMULATION_END
    history_start = end - timedelta(days=settings.hydration_lookback_days)
    roots = _parse_roots(args.roots) or OFFLINE_DEFAULT_ROOTS
    chain = build_hydration_chain(
        settings,
        repository=repository,
        instrument_master=master,
        source="offline",
        offline_allowed=True,
    )
    hydrator = ColdStartHydrator(
        repository=repository,
        instrument_master=master,
        lookback_days=settings.hydration_lookback_days,
        backfill=chain,
        backfill_enabled=True,
        clock=lambda: end,
    )
    simulated = SimulatedBroker(master, initial_cash=OFFLINE_INITIAL_CASH, timestamp=history_start)
    broker = OfflineTargetBroker(simulated_broker=simulated, instrument_master=master)
    streamer = OfflineMarketStateStreamer(
        roots=roots,
        start=history_start,
        end=end,
        equity=OFFLINE_INITIAL_CASH,
        instrument_master=master,
        on_state=broker.update_market,
    )
    runner = TradingRunner(
        settings=settings,
        pipeline=pipeline,
        streamer=streamer,
        broker=broker,
        hydrator=hydrator,
        instrument_master=master,
        translator=translator,
        event_bus=None,
        state_store=state_store,
        clock=lambda: end,
        dry_run=True,
        prime_pipeline=True,
    )
    return asyncio.run(runner.run(roots=roots, max_states=args.max_states))


def _run_live(settings: Settings, args: argparse.Namespace) -> RunnerStatus:  # pragma: no cover
    """Build the production IBKR/Databento wiring lazily and run until stopped."""
    from src.data.ibkr_data import IBKRMarketDataStreamer
    from src.execution.ibkr_client import IBKRConnectionManager
    from src.execution.oms import OrderManagementSystem
    from src.risk.contract_translator import ContractTranslator
    from src.risk.state_store import build_risk_state_store
    from src.strategies.pipeline import build_pipeline

    strategy_path = _strategy_path(settings, args.strategy)
    config, master = _load_strategy(settings, strategy_path)
    repository = ParquetRepository(settings)
    state_store = build_risk_state_store(
        settings.risk_state_store_uri, duckdb_path=settings.duckdb_path
    )
    offline_allowed = bool(
        args.backfill_source == "offline" or settings.hydration_source == "offline"
    )
    chain = build_hydration_chain(
        settings,
        repository=repository,
        instrument_master=master,
        source=args.backfill_source,
        offline_allowed=offline_allowed,
    )
    hydrator = ColdStartHydrator(
        repository=repository,
        instrument_master=master,
        lookback_days=settings.hydration_lookback_days,
        backfill=chain if settings.hydration_backfill_enabled else None,
        backfill_enabled=settings.hydration_backfill_enabled,
    )
    manager = IBKRConnectionManager(settings, instrument_master=master)
    oms = OrderManagementSystem(
        manager=manager,
        settings=settings,
        instrument_master=master,
        state_store=state_store,
    )
    translator = ContractTranslator.from_config(
        config.translator,
        instruments=master,
        account_currency=settings.account_currency,
        fx_rates=settings.fx_rates,
    )
    overlays = []
    try:
        from src.risk.quant_risk_overlay import QuantRiskOverlay

        overlays.append(
            QuantRiskOverlay(settings.risk, instruments=master, state_store=state_store)
        )
    except Exception as exc:  # pragma: no cover - defensive production wiring
        logger.error("quant risk overlay could not be wired into the live pipeline: %s", exc)
    pipeline = build_pipeline(
        config,
        master,
        translator=translator,
        risk_overlays=overlays,
    )
    streamer = IBKRMarketDataStreamer(settings, instrument_master=master)
    runner = TradingRunner(
        settings=settings,
        pipeline=pipeline,
        streamer=streamer,
        broker=oms,
        hydrator=hydrator,
        instrument_master=master,
        translator=translator,
        event_bus=None,
        state_store=state_store,
        prime_pipeline=True,
    )
    return asyncio.run(runner.run(roots=_parse_roots(args.roots), max_states=args.max_states))


def _print_cli_failure(exc: BaseException) -> None:
    """Print an actionable CLI error block (never a raw traceback) to stderr."""
    message = str(exc).strip() or type(exc).__name__
    lines = [f"futuresrl-runner: error: {message}"]
    # Hydration failures are the ones an operator fixes by choosing a history
    # source, so only they get the vendor-setup hints (avoid noisy duplicates when
    # the chain message already embeds them).
    if isinstance(exc, HydrationError) and HYDRATION_SOURCE_HINTS not in message:
        lines.append(HYDRATION_SOURCE_HINTS)
    sys.stderr.write("\n".join(lines) + "\n")
    sys.stderr.flush()


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: ``--check`` validates config, ``--offline`` runs a demo."""
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    settings = get_settings()
    log_level = (args.log_level or settings.log_level).upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        if args.max_states is not None and args.max_states < 1:
            raise RunnerConfigError(
                f"--max-states must be a positive integer, got {args.max_states}"
            )
        if args.check:
            summary = _check_configuration(
                settings,
                _strategy_path(settings, args.strategy),
                offline=args.offline,
                backfill_source=args.backfill_source,
            )
            json.dump(summary, sys.stdout, indent=2, sort_keys=True, default=str)
            sys.stdout.write("\n")
            return 0
        if args.offline and args.backfill_source not in (None, "auto", "offline"):
            raise RunnerConfigError(
                "--offline replays the deterministic synthetic stack and cannot be combined "
                f"with --backfill-source={args.backfill_source}; use --backfill-source offline "
                "or omit it"
            )
        status = _run_offline(settings, args) if args.offline else _run_live(settings, args)
    except (HydrationError, RunnerConfigError) as exc:
        _print_cli_failure(exc)
        return 2
    json.dump(status.as_dict(), sys.stdout, indent=2, sort_keys=True, default=str)
    sys.stdout.write("\n")
    return 0 if not status.halted else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
