"""Live-runner bridge endpoints: status, offline step and reset (Phase 9)."""

from __future__ import annotations

from typing import Annotated, cast

from fastapi import APIRouter, Depends, HTTPException, Request, status

from src.api.deps import LiveDep
from src.api.live import RunnerSessionUnavailableError
from src.api.phase11_schemas import (
    RebalanceExecuteRequest,
    RebalanceExecuteResponse,
    RebalancePreviewRequest,
    RebalancePreviewResponse,
)
from src.api.rebalance_service import RebalanceError, RebalanceService
from src.api.schemas import RunnerStatusResponse, RunnerStepRequest, RunnerStepResponse

router = APIRouter(tags=["runner"])


def get_rebalance_service(request: Request) -> RebalanceService:
    """Return (and cache) the Phase-11A rebalance preview/execute service.

    Tests can inject ``app.state.rebalance_service`` with fakes; production builds
    it from the live container (settings/market/catalog/execution/clock).
    """
    injected = getattr(request.app.state, "rebalance_service", None)
    if (
        injected is not None
        and callable(getattr(injected, "preview", None))
        and callable(getattr(injected, "execute", None))
    ):
        return cast(RebalanceService, injected)
    container = getattr(request.app.state, "container", None)
    if container is None:  # pragma: no cover - create_app always builds a container
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="service container is unavailable",
        )
    service = RebalanceService(
        settings=cast("object", container.settings),  # type: ignore[arg-type]
        market=container.market,
        catalog=container.strategies,
        execution=container.execution,
        clock=container.clock,
    )
    request.app.state.rebalance_service = service
    return service


RebalanceDep = Annotated[RebalanceService, Depends(get_rebalance_service)]


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


@router.post(
    "/runner/rebalance-preview",
    response_model=RebalancePreviewResponse,
    summary="Preview the next runner rebalance (no orders submitted)",
    description=(
        "Hydrates the latest state with the current broker NLV/currency/positions, runs "
        "the full S->A->T->R pipeline + contract translator, computes delta orders against "
        "the current book and reports the per-order pre-trade gate verdict, projected "
        "margin utilization (60% cap), tracking error and mode (``ibkr-paper`` when a "
        "connected IBKR paper adapter is active, else ``offline``). Never submits orders."
    ),
    responses={
        404: {"description": "Unknown strategy id"},
        422: {"description": "Invalid strategy/roots payload"},
        503: {"description": "Execution book unavailable"},
    },
)
async def rebalance_preview(
    payload: RebalancePreviewRequest, rebalance: RebalanceDep
) -> RebalancePreviewResponse:
    """Run the full pipeline and return proposed delta orders (no submission)."""
    try:
        return await rebalance.preview(payload)
    except RebalanceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.post(
    "/runner/rebalance-execute",
    response_model=RebalanceExecuteResponse,
    summary="Execute the next runner rebalance (paper/offline only)",
    description=(
        "Recomputes the preview plan and submits ONLY gate-approved delta orders through "
        "the active paper OMS (IBKR paper when connected) or the simulated broker, "
        "refreshes the book and publishes a RunnerBridgeStore snapshot. ``confirm=true`` "
        "is required. Returns 409 in the live environment, for unsafe/unconnected IBKR "
        "sessions, or when a fixture-only offline adapter cannot execute; repeated "
        "identical executions are idempotent (no duplicate submission)."
    ),
    responses={
        400: {"description": "Missing confirmation"},
        404: {"description": "Unknown strategy id"},
        409: {"description": "Live/unsafe execution or no executable adapter"},
        422: {"description": "Invalid strategy/roots payload"},
    },
)
async def rebalance_execute(
    payload: RebalanceExecuteRequest, rebalance: RebalanceDep
) -> RebalanceExecuteResponse:
    """Execute the approved rebalance delta through the active safe adapter."""
    try:
        return await rebalance.execute(payload)
    except RebalanceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
