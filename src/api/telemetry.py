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
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math

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
    ) -> None:
        self._settings = settings
        self._clock = clock
        self._execution = execution
        self._risk = risk
        self._market = market
        self._bus: EventBus[TelemetryEvent] = EventBus()
        self._clients: dict[int, asyncio.Queue[TelemetryEnvelope]] = {}
        self._next_client_id = 1
        self._sequence = 0
        self._task: asyncio.Task[None] | None = None
        self._account_cache: AccountSnapshotResponse | None = None
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

    async def _account(self) -> AccountSnapshotResponse:
        if self._account_cache is None:
            self._account_cache = await self._execution.account()
        return self._account_cache

    def _next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    async def _build_envelope(self, kind: TelemetryKind) -> TelemetryEnvelope:
        now = self._clock()
        sequence = self._next_sequence()
        if kind == "pnl":
            account = await self._account()
            jitter = _DEMO_PNL_JITTER_USD * math.sin(sequence * 0.7)
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
                ),
            )
        if kind == "margin":
            account = await self._account()
            return TelemetryEnvelope(
                kind="margin",
                timestamp=now,
                sequence=sequence,
                payload=MarginTelemetryPayload(
                    initial_margin_used=account.initial_margin_used,
                    maintenance_margin_used=account.maintenance_margin_used,
                    margin_utilization=account.margin_utilization,
                    warning_threshold=self._settings.risk.margin_warning_threshold,
                    critical_threshold=self._settings.risk.margin_critical_threshold,
                ),
            )
        if kind == "positions":
            account = await self._account()
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
            )
        elif isinstance(payload, MarginTelemetryPayload):
            values.update(
                margin_utilization=payload.margin_utilization,
                initial_margin_used=payload.initial_margin_used,
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
