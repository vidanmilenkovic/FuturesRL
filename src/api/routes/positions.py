"""Account snapshot / active positions endpoint (Command Center view)."""

from __future__ import annotations

from fastapi import APIRouter

from src.api.deps import ContainerDep
from src.api.schemas import AccountSnapshotResponse

router = APIRouter(tags=["positions"])


@router.get(
    "/positions",
    response_model=AccountSnapshotResponse,
    summary="Account snapshot, positions and equity curve",
    description=(
        "NLV, margin consumption, gross/net exposure, the active futures contract "
        "book (expiry, days-to-roll, signed quantity, notional weight w_i, "
        "unrealised PnL, margin) and a mark-to-market equity curve. Prefers a fresh "
        "live-runner bridge snapshot (``source=runner-bridge``)."
    ),
)
async def account_snapshot(container: ContainerDep) -> AccountSnapshotResponse:
    """Return the current account/position snapshot (bridge-first)."""
    return await container.live.account()
