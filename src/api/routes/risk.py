"""LLM Risk-First endpoints: state, threshold overrides and the kill switch."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from src.api.deps import ExecutionDep, RiskDep
from src.api.schemas import (
    FlattenAllResponse,
    RiskStateResponse,
    RiskThresholdOverride,
    RiskThresholds,
)

router = APIRouter(tags=["risk"])


@router.get(
    "/risk/state",
    response_model=RiskStateResponse,
    summary="LLM Risk-First state",
    description=(
        "Per-root sentiment, semantic risk S_i,t, confidence c_i,t, the "
        "deterministic breaker action, the macro lockout window and the headline "
        "feed. Also reports kill-switch status when execution is halted."
    ),
)
def risk_state(risk: RiskDep) -> RiskStateResponse:
    """Return the current LLM Risk-First state."""
    return risk.state()


@router.put(
    "/risk/thresholds",
    response_model=RiskThresholds,
    summary="Override risk thresholds",
    description=(
        "Validate and apply a partial override of the LLM risk / margin / "
        "volatility thresholds. Throws 422 when the merged thresholds violate the "
        "risk contract (e.g. κ must exceed θ_safe)."
    ),
)
def update_thresholds(risk: RiskDep, payload: RiskThresholdOverride) -> RiskThresholds:
    """Apply a partial threshold override and return the effective thresholds."""
    try:
        return risk.update_thresholds(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post(
    "/risk/flatten-all",
    response_model=FlattenAllResponse,
    summary="Emergency flatten-all kill switch",
    description=(
        "Cancel working orders, market-close every futures position and lock "
        "execution. Runs against the configured execution adapter (offline "
        "simulated broker by default, IBKR OMS when connected)."
    ),
    responses={409: {"description": "Execution service cannot accept a flatten right now"}},
)
async def flatten_all(execution: ExecutionDep) -> FlattenAllResponse:
    """Flatten every position through the execution adapter's kill switch."""
    try:
        return await execution.flatten_all()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"flatten-all failed: {type(exc).__name__}: {exc}",
        ) from exc
