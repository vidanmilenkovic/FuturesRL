"""Telemetry broadcaster status endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from src.api.deps import ContainerDep
from src.api.schemas import TelemetryStatusResponse

router = APIRouter(tags=["telemetry"])


@router.get(
    "/telemetry/status",
    response_model=TelemetryStatusResponse,
    summary="WebSocket telemetry status",
    description=(
        "Publisher interval, connected client count, published frame count and "
        "subscribed EventBus topics for ``/ws/telemetry``."
    ),
)
def telemetry_status(container: ContainerDep) -> TelemetryStatusResponse:
    """Return the broadcaster status."""
    return container.telemetry_status()
