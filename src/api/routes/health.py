"""Health endpoint (unversioned, used by Docker/compose healthchecks)."""

from __future__ import annotations

from fastapi import APIRouter

from src.api.deps import ContainerDep
from src.api.schemas import HealthResponse

router = APIRouter(tags=["system"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness / readiness probe",
    description=(
        "Reports process health, environment and whether the API is serving "
        "deterministic offline fixtures instead of a live IBKR/Databento session."
    ),
)
def health(container: ContainerDep) -> HealthResponse:
    """Return the process health payload."""
    return container.health()
