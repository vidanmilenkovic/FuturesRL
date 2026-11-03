"""WebSocket telemetry broadcaster for ``/ws/telemetry``.

The broadcaster is the only writer on the telemetry side of
``src.core.event_bus.EventBus``: a background publisher task builds one typed
:class:`~src.api.schemas.TelemetryEnvelope` per interval (injectable through
``Settings.telemetry_interval_seconds``), publishes a flattened
:class:`~src.core.event_bus.TelemetryEvent` on the ``"telemetry"`` topic for
in-process consumers (and backtest/live parity), then fans the typed envelope out
to every connected WebSocket client. Slow clients get a bounded queue, so one
stalled browser can never delay the trading loop.

Frame cadence (deterministic, independent of wall-clock time):

``pnl → margin → positions → heartbeat → pnl → margin → risk_alert → heartbeat``
with the pipeline waterfall every tenth frame. Values are derived from the live
service adapters (offline simulated broker / risk engine), with a tiny
deterministic sinusoid so an offline demo session visibly ticks; production
wiring uses the same frame shapes and no jitter.

Phase-10 guarantee: once IBKR is enabled the broadcaster consumes only the IBKR
adapter's live or last-known-good (``stale``) book. A failed account refresh keeps
the previous book and an absent book degrades the frame to a heartbeat -- simulated
positions are never published into a live session.

v1.5.2 state-sync guarantee: for an IBKR-enabled process the account-driven frames
are built from the **same enriched account as ``GET /api/positions``** (injectable
``account_provider`` -> ``LiveStateService.account``): last-known-good/live IBKR
book, real curve/volatility attribution, and no demo PnL jitter. Offline frames keep
their deterministic jitter and previous behaviour.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
from collections.abc import Awaitable, Callable
from datetime import datetime

from configs.settings import Settings

from src.api import fixtures
from src.api.protocols import ExecutionService, MarketDataService, RiskService
from src.api.schemas import (
    AccountSnapshotResponse,
    EquityPoint,
    HeartbeatTelemetryPayload,
    MarginTelemetryPayload,
    PipelineTelemetryPayload,
    PnlTelemetryPayload,
    PositionsTelemetryPayload,
    RiskAlertTelemetryPayload,
    TelemetryEnvelope,
    TelemetryKind,
    TelemetryStatusResponse,
)
from src.core.event_bus import EventBus, TelemetryEvent

logger = logging.getLogger(__name__)

__all__ = ["TelemetryBroadcaster"]

#: Deterministic frame order; index modulo the length drives the frame kind.
_FRAME_ORDER: tuple[TelemetryKind, ...] = (
    "pnl",
    "margin",
    "positions",
    "heartbeat",
    "pnl",
    "margin",
    "risk_alert",
    "heartbeat",
    "pnl",
    "pipeline",
)

#: Maximum buffered frames per client before the oldest frame is dropped.
_CLIENT_QUEUE_SIZE: int = 64

#: Amplitude of the offline telemetry jitter (USD) for the pnl frame.
_DEMO_PNL_JITTER_USD: float = 180.0


class TelemetryBroadcaster:
    """Publishes typed telemetry frames to EventBus subscribers and WebSockets."""

    def __init__(
        self,
        *,
        settings: Settings,
        clock: fixtures.Clock,
        execution: ExecutionService,
        risk: RiskService,
        market: MarketDataService,
        account_provider: Callable[[], Awaitable[AccountSnapshotResponse]] | None = None,
    ) -> None:
        self._settings = settings
        self._clock = clock
        self._execution = execution
        #: Same enriched account as ``GET /api/positions`` (``LiveStateService.account``).
        #: Used for IBKR frames so WS attribution can never disagree with REST.
        self._account_provider = account_provider
        self._risk = risk
        self._market = market
        self._bus: EventBus[TelemetryEvent] = EventBus()
        self._clients: dict[int, asyncio.Queue[TelemetryEnvelope]] = {}
        self._next_client_id = 1
        self._sequence = 0
        self._task: asyncio.Task[None] | None = None
        self._account_cache: AccountSnapshotResponse | None = None
        self._account_refreshed_at: datetime | None = None
        self._interval = float(settings.telemetry_interval_seconds)

    # -- introspection ---------------------------------------------------------

    @property
    def bus(self) -> EventBus[TelemetryEvent]:
        """The underlying event bus (observable in tests / in-process consumers)."""
        return self._bus

    @property
    def interval_seconds(self) -> float:
        """Publisher interval from the settings."""
        return self._interval

    @property
    def sequence(self) -> int:
        """Sequence number of the most recently queued frame."""
        return self._sequence

    def status(self) -> TelemetryStatusResponse:
        """Broadcaster status for ``GET /api/telemetry/status``."""
        now = self._clock()
        return TelemetryStatusResponse(
            stream_active=self._task is not None and not self._task.done(),
            connected_clients=len(self._clients),
            published_events=self._bus.published_count,
            publisher_interval_seconds=self._interval,
            topics=sorted(set(self._bus.topics) | {"telemetry"}),
            server_time=now,
        )

    # -- lifecycle -------------------------------------------------------------

    async def start(self) -> None:
        """Start the background publisher task (idempotent)."""
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(self._run(), name="futuresrlx-telemetry-publisher")
        logger.info("telemetry broadcaster started (interval=%.3fs)", self._interval)

    async def stop(self) -> None:
        """Stop the publisher and disconnect every client queue."""
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._clients.clear()
        logger.info("telemetry broadcaster stopped")

    async def _run(self) -> None:
        while True:
            try:
                await self.publish_next()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("telemetry publisher frame failed")
            await asyncio.sleep(self._interval)

    # -- client fan-out --------------------------------------------------------

    def connect(self) -> tuple[int, asyncio.Queue[TelemetryEnvelope]]:
        """Register a client queue; returns ``(client_id, queue)``."""
        client_id = self._next_client_id
        self._next_client_id += 1
        queue: asyncio.Queue[TelemetryEnvelope] = asyncio.Queue(maxsize=_CLIENT_QUEUE_SIZE)
        self._clients[client_id] = queue
        return client_id, queue

    def disconnect(self, client_id: int) -> None:
        """Remove a client queue (idempotent)."""
        self._clients.pop(client_id, None)

    # -- frame construction ----------------------------------------------------

    def _ibkr_enabled(self) -> bool:
        """Whether the execution adapter is the real IBKR engine."""
        return bool(getattr(self._execution, "ibkr_enabled", False))

    async def _account(self) -> AccountSnapshotResponse | None:
        """Latest account snapshot, or ``None`` when no honest book exists yet.

        For a pure offline/simulated process the snapshot is fetched once (the
        deterministic broker never changes its own book). For an IBKR-enabled
        process every account-frame uses the injected enriched provider (the same
        ``LiveStateService.account`` as ``GET /api/positions``) at a bounded
        cadence; a fetch failure falls back to the **last-known-good IBKR book** --
        the telemetry stream must never replace a live book with simulated data.
        """
        now = self._clock()
        if not self._ibkr_enabled():
            if self._account_cache is None:
                self._account_cache = await self._execution.account()
                self._account_refreshed_at = now
            return self._account_cache
        refreshed_at = self._account_refreshed_at
        if (
            self._account_cache is not None
            and refreshed_at is not None
            and (now - refreshed_at).total_seconds() < max(self._interval, 0.5)
        ):
            return self._account_cache
        try:
            provider = self._account_provider
            if provider is not None:
                account = await provider()
            else:
                account = await self._execution.account()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "live account refresh failed; keeping last-known-good/stale book: %s", exc
            )
            return self._account_cache
        self._account_cache = account
        self._account_refreshed_at = now
        return account

    def _next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    def _heartbeat_envelope(self, now: datetime, sequence: int) -> TelemetryEnvelope:
        """Idle frame used when an account-driven frame has no honest book."""
        return TelemetryEnvelope(
            kind="heartbeat",
            timestamp=now,
            sequence=sequence,
            payload=HeartbeatTelemetryPayload(status="ok", sequence=sequence),
        )

    async def _build_envelope(self, kind: TelemetryKind) -> TelemetryEnvelope:
        now = self._clock()
        sequence = self._next_sequence()
        account: AccountSnapshotResponse | None = None
        if kind in {"pnl", "margin", "positions"}:
            account = await self._account()
            if account is None:
                # IBKR enabled but no live/last-good book yet: never fall back to
                # simulated positions (the dashboard keeps its last REST snapshot).
                return self._heartbeat_envelope(now, sequence)
        ibkr_enabled = self._ibkr_enabled()
        open_positions = len(account.positions) if account is not None else 0
        # IBKR flat book: margin utilisation and book volatility are exactly 0.0 by
        # definition, even if a stale/non-enriched adapter row carries leftover
        # margin figures (the bug this frame contract fixes).
        flat_ibkr_book = ibkr_enabled and open_positions == 0
        if kind == "pnl":
            assert account is not None
            initial_margin = 0.0 if flat_ibkr_book else account.initial_margin_used
            margin_utilization = 0.0 if flat_ibkr_book else account.margin_utilization
            book_volatility = account.portfolio_volatility
            if flat_ibkr_book and book_volatility is None:
                book_volatility = 0.0
            # Live IBKR frames carry the real book untouched (no demo jitter), so
            # the WS attribution can never disagree with GET /api/positions.
            jitter = 0.0 if ibkr_enabled else _DEMO_PNL_JITTER_USD * math.sin(sequence * 0.7)
            net_liquidation = account.net_liquidation + jitter
            day_pnl = account.day_pnl + jitter
            return TelemetryEnvelope(
                kind="pnl",
                timestamp=now,
                sequence=sequence,
                payload=PnlTelemetryPayload(
                    net_liquidation=round(net_liquidation, 2),
                    day_pnl=round(day_pnl, 2),
                    day_pnl_pct=round(day_pnl / net_liquidation, 8) if net_liquidation else 0.0,
                    point=EquityPoint(time=now, value=round(net_liquidation, 2)),
                    portfolio_volatility=book_volatility,
                    initial_margin_used=initial_margin,
                    margin_utilization=margin_utilization,
                    open_positions=open_positions,
                ),
            )
        if kind == "margin":
            assert account is not None
            margin_volatility = account.portfolio_volatility
            if flat_ibkr_book and margin_volatility is None:
                margin_volatility = 0.0
            return TelemetryEnvelope(
                kind="margin",
                timestamp=now,
                sequence=sequence,
                payload=MarginTelemetryPayload(
                    initial_margin_used=0.0 if flat_ibkr_book else account.initial_margin_used,
                    maintenance_margin_used=0.0
                    if flat_ibkr_book
                    else account.maintenance_margin_used,
                    margin_utilization=0.0 if flat_ibkr_book else account.margin_utilization,
                    warning_threshold=self._settings.risk.margin_warning_threshold,
                    critical_threshold=self._settings.risk.margin_critical_threshold,
                    portfolio_volatility=margin_volatility,
                    open_positions=open_positions,
                ),
            )
        if kind == "positions":
            assert account is not None
            return TelemetryEnvelope(
                kind="positions",
                timestamp=now,
                sequence=sequence,
                payload=PositionsTelemetryPayload(
                    positions=account.positions,
                    gross_leverage=account.gross_leverage,
                    net_exposure=account.net_exposure,
                ),
            )
        if kind == "risk_alert":
            state = self._risk.state()
            return TelemetryEnvelope(
                kind="risk_alert",
                timestamp=now,
                sequence=sequence,
                payload=RiskAlertTelemetryPayload(
                    assessments=state.assessments,
                    circuit_breaker_active=state.circuit_breaker_active,
                    circuit_breaker_reason=state.circuit_breaker_reason,
                    macro_lockout=state.macro_lockout,
                ),
            )
        if kind == "pipeline":
            pipeline = self._market.pipeline()
            return TelemetryEnvelope(
                kind="pipeline",
                timestamp=now,
                sequence=sequence,
                payload=PipelineTelemetryPayload(pipeline=pipeline),
            )
        return TelemetryEnvelope(
            kind="heartbeat",
            timestamp=now,
            sequence=sequence,
            payload=HeartbeatTelemetryPayload(status="ok", sequence=sequence),
        )

    def heartbeat(self) -> TelemetryEnvelope:
        """Build an idle heartbeat frame (used when a socket has no queued data)."""
        now = self._clock()
        sequence = self._next_sequence()
        return TelemetryEnvelope(
            kind="heartbeat",
            timestamp=now,
            sequence=sequence,
            payload=HeartbeatTelemetryPayload(status="ok", sequence=sequence),
        )

    def _flatten(self, envelope: TelemetryEnvelope) -> dict[str, str | int | float | bool | None]:
        payload = envelope.payload
        values: dict[str, str | int | float | bool | None] = {
            "sequence": envelope.sequence,
            "timestamp": envelope.timestamp.isoformat(),
        }
        if isinstance(payload, PnlTelemetryPayload):
            values.update(
                net_liquidation=payload.net_liquidation,
                day_pnl=payload.day_pnl,
                day_pnl_pct=payload.day_pnl_pct,
                portfolio_volatility=payload.portfolio_volatility,
                initial_margin_used=payload.initial_margin_used,
                margin_utilization=payload.margin_utilization,
                open_positions=payload.open_positions,
            )
        elif isinstance(payload, MarginTelemetryPayload):
            values.update(
                margin_utilization=payload.margin_utilization,
                initial_margin_used=payload.initial_margin_used,
                portfolio_volatility=payload.portfolio_volatility,
                open_positions=payload.open_positions,
            )
        elif isinstance(payload, PositionsTelemetryPayload):
            values.update(
                position_count=len(payload.positions),
                gross_leverage=payload.gross_leverage,
                net_exposure=payload.net_exposure,
            )
        elif isinstance(payload, RiskAlertTelemetryPayload):
            values.update(
                circuit_breaker_active=payload.circuit_breaker_active,
                assessment_count=len(payload.assessments),
                macro_lockout_active=payload.macro_lockout.active,
            )
        elif isinstance(payload, PipelineTelemetryPayload):
            values.update(
                cycle_id=payload.pipeline.cycle_id,
                gross=payload.pipeline.stages[-1].gross if payload.pipeline.stages else 0.0,
            )
        else:
            values.update(status=payload.status)
        return values

    async def publish_next(self) -> TelemetryEnvelope:
        """Build + publish one frame and fan it out to connected clients."""
        index = (self._sequence) % len(_FRAME_ORDER)
        kind = _FRAME_ORDER[index]
        envelope = await self._build_envelope(kind)
        await self._bus.publish(
            "telemetry",
            TelemetryEvent(
                kind=envelope.kind,
                timestamp_ns=int(envelope.timestamp.timestamp() * 1_000_000_000),
                values=self._flatten(envelope),
            ),
        )
        for queue in tuple(self._clients.values()):
            if queue.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(envelope)
        return envelope
