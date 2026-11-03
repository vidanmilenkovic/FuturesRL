"""Operator maintenance endpoints (paper/backtest state reset, QoL).

These endpoints are destructive by design and therefore only available outside the
``live`` environment. Every clear action is explicitly confirmed by an exact body
phrase and reported back as per-action receipts; repeated calls are safe.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from src.api.deps import ContainerDep
from src.api.schemas import PaperStateResetRequest, PaperStateResetResponse
from src.api.services import (
    OperatorActionUnavailableError,
    OperatorConfirmationError,
    OperatorResetError,
)

router = APIRouter(tags=["maintenance"])


@router.post(
    "/maintenance/reset-paper-state",
    response_model=PaperStateResetResponse,
    summary="Reset operator paper/backtest state (bridge file, risk state, paper cache)",
    description=(
        "Non-live only (409 in ``live``). Requires body ``confirm`` exactly "
        '``"RESET PAPER STATE"``. Independently clears the runner bridge snapshot '
        "file (``runner_bridge_dir/runner_state.json``), the persisted risk state "
        "store entries (file/DuckDB/Redis) and the simulated/paper in-memory session "
        "cache according to the request flags. Returns receipts with the removed "
        "files, store reset and session reset; idempotent and never destructive "
        "without the confirmation phrase."
    ),
    responses={
        400: {"description": "Missing/incorrect confirmation phrase"},
        409: {"description": "Environment is live (reset unavailable)"},
    },
)
async def reset_paper_state(
    payload: PaperStateResetRequest,
    container: ContainerDep,
) -> PaperStateResetResponse:
    """Clear paper/backtest operator state and return the receipts."""
    try:
        return await container.reset_paper_state(payload)
    except OperatorConfirmationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except OperatorActionUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except OperatorResetError as exc:  # pragma: no cover - defensive catch-all
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
