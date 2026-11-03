"""Live-runner bridge endpoints: status, offline step and reset (Phase 9)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from src.api.deps import LiveDep
from src.api.live import RunnerSessionUnavailableError
from src.api.schemas import RunnerStatusResponse, RunnerStepRequest, RunnerStepResponse

router = APIRouter(tags=["runner"])


@router.get(
    "/runner/status",
    response_model=RunnerStatusResponse,
    summary="Live runner bridge status",
    description=(
        "Runner bridge availability/staleness (runner id, mode, sequence, updated_at, "
        "stale_after_seconds, status) plus the API's singleton offline-runner session "
        "state. Never 503s: ``bridge_available=false``/``source=unavailable`` when the "
        "runner has never published a snapshot."
    ),
)
def runner_status(live: LiveDep) -> RunnerStatusResponse:
    """Return bridge staleness and session state."""
    return live.runner_status()


@router.post(
    "/runner/step",
    response_model=RunnerStepResponse,
    summary="Advance the offline runner",
    description=(
        "Warm-start (once) and advance the singleton offline ``OfflineRunnerSession`` "
        "by ``states`` MarketStates, then return the refreshed bridge summary. "
        "Returns 503 with an actionable message when the execution package does not "
        "ship ``OfflineRunnerSession`` in this deployment."
    ),
    responses={503: {"description": "Offline runner session unavailable"}},
)
async def runner_step(payload: RunnerStepRequest, live: LiveDep) -> RunnerStepResponse:
    """Advance the offline runner and report the bridge snapshot."""
    try:
        return await live.runner_step(payload)
    except RunnerSessionUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"offline runner session unavailable: {exc}. Start the live runner "
                "(``python -m src.execution.runner``) to populate the bridge, or use "
                "GET /api/runner/status for bridge-only telemetry."
            ),
        ) from exc


@router.post(
    "/runner/reset",
    response_model=RunnerStatusResponse,
    summary="Reset the offline runner session",
    description="Closes the singleton offline runner session; the bridge snapshot is retained.",
)
async def runner_reset(live: LiveDep) -> RunnerStatusResponse:
    """Close the offline runner session and return the status."""
    return await live.runner_reset()
