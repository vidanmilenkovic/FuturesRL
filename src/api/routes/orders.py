"""Working orders and recent fills endpoints (Command Center live book)."""

from __future__ import annotations

from fastapi import APIRouter, Query

from src.api.deps import LiveDep
from src.api.schemas import RecentFillsResponse, WorkingOrdersResponse

router = APIRouter(tags=["orders"])


@router.get(
    "/orders/working",
    response_model=WorkingOrdersResponse,
    summary="Working orders",
    description=(
        "Orders that are still working (submitted / partially filled, not terminal). "
        "Prefers a fresh live-runner bridge snapshot (``source=runner-bridge``); "
        "falls back to the local simulated/fixture execution adapter and reports "
        "the provenance in ``source``."
    ),
)
async def working_orders(live: LiveDep) -> WorkingOrdersResponse:
    """Return the working order book."""
    return await live.working_orders()


@router.get(
    "/fills/recent",
    response_model=RecentFillsResponse,
    summary="Recent fills",
    description=(
        "Most recent executions, newest first. Prefers a fresh live-runner bridge "
        "snapshot (``source=runner-bridge``); otherwise reports the simulated/"
        "fixture adapter's fills."
    ),
)
async def recent_fills(
    live: LiveDep,
    limit: int = Query(default=50, ge=1, le=1000, description="Maximum rows to return."),
) -> RecentFillsResponse:
    """Return the most recent fills."""
    return await live.recent_fills(limit)
