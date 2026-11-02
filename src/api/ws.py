"""WebSocket telemetry endpoint (``/ws/telemetry``).

The socket is a thin, back-pressure-safe consumer of
:class:`~src.api.telemetry.TelemetryBroadcaster`: register a bounded queue, forward
every typed :class:`~src.api.schemas.TelemetryEnvelope` as JSON, emit an idle
heartbeat when the publisher is quiet, and always unregister on disconnect (clean
or abrupt) so a closed browser tab cannot leak a queue.
"""

from __future__ import annotations

import asyncio
import logging
from typing import cast

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.api.schemas import TelemetryEnvelope
from src.api.services import Container
from src.api.telemetry import TelemetryBroadcaster

logger = logging.getLogger(__name__)

router = APIRouter(tags=["telemetry"])

#: Minimum idle timeout before an unsolicited heartbeat is sent.
_MIN_HEARTBEAT_TIMEOUT: float = 0.25


def _container(websocket: WebSocket) -> Container:
    container = getattr(websocket.app.state, "container", None)
    if container is None:  # pragma: no cover - only during misconfigured startup
        raise RuntimeError("service container is not initialised")
    return cast(Container, container)


async def _send(websocket: WebSocket, envelope: TelemetryEnvelope) -> None:
    await websocket.send_text(envelope.model_dump_json())


@router.websocket("/ws/telemetry")
async def telemetry_socket(websocket: WebSocket) -> None:
    """Stream typed telemetry frames to a browser client."""
    await websocket.accept()
    container = _container(websocket)
    broadcaster: TelemetryBroadcaster = container.telemetry
    client_id, queue = broadcaster.connect()
    idle_timeout = max(broadcaster.interval_seconds * 3.0, _MIN_HEARTBEAT_TIMEOUT)
    logger.debug("telemetry websocket connected (client=%s)", client_id)
    try:
        # Immediate heartbeat so the dashboard can mark the stream as live without
        # waiting a full publisher interval.
        await _send(websocket, broadcaster.heartbeat())
        while True:
            try:
                envelope = await asyncio.wait_for(queue.get(), timeout=idle_timeout)
            except TimeoutError:
                envelope = broadcaster.heartbeat()
            await _send(websocket, envelope)
    except WebSocketDisconnect:
        logger.debug("telemetry websocket disconnected (client=%s)", client_id)
    except asyncio.CancelledError:  # pragma: no cover - server shutdown
        raise
    except Exception:
        logger.debug("telemetry websocket transport error (client=%s)", client_id, exc_info=True)
    finally:
        broadcaster.disconnect(client_id)
