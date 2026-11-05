"""LLM Risk-First endpoints: state, threshold overrides and the kill switch."""

from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException, status

from src.api.deps import ContainerDep, ExecutionDep, RiskDep
from src.api.schemas import (
    FlattenAllResponse,
    RiskLockResetRequest,
    RiskLockResetResponse,
    RiskStateResponse,
    RiskThresholdOverride,
    RiskThresholds,
)
from src.api.services import (
    OperatorActionUnavailableError,
    OperatorConfirmationError,
    OperatorResetError,
    OperatorTokenError,
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


@router.post(
    "/risk/reset-lock",
    response_model=RiskLockResetResponse,
    summary="Operator reset of the execution/risk kill switch",
    description=(
        "Guarded manual reset of the execution kill switch and the durable risk "
        "state (persisted kill switch/daily-loss latch, starting daily NLV and "
        "HWM/trailing-stop state). Requires body ``confirm`` exactly "
        '``"RESET RISK LOCK"``; when ``FUTURESRL_OPERATOR_RESET_TOKEN`` is set the '
        "matching token must be supplied in the body ``operator_token`` or the "
        "``X-Operator-Token`` header. Refused (409) while an emergency flatten is in "
        "progress or when the adapter cannot reset. Never runs unauthenticated when a "
        "token is configured."
    ),
    responses={
        400: {"description": "Missing/incorrect confirmation phrase"},
        403: {"description": "Missing/incorrect operator reset token"},
        409: {"description": "Reset not possible now (active flatten / unsupported)"},
    },
)
def reset_risk_lock(
    payload: RiskLockResetRequest,
    container: ContainerDep,
    x_operator_token: str | None = Header(default=None, alias="X-Operator-Token"),
) -> RiskLockResetResponse:
    """Clear the kill switch + persisted risk lock after double confirmation."""
    try:
        return container.reset_risk_lock(payload, header_token=x_operator_token)
    except OperatorConfirmationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except OperatorTokenError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except OperatorActionUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except OperatorResetError as exc:  # pragma: no cover - defensive catch-all
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
