"""LLM Risk-First endpoints: state, threshold overrides and the kill switch."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Header, HTTPException, Query, status

from src.api.deps import ContainerDep, ExecutionDep, RiskDep
from src.api.schemas import (
    FlattenAllResponse,
    LLMStatusResponse,
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
    summary="LLM Risk-First state (45-root scopes)",
    description=(
        "Per-root sentiment, semantic risk S_i,t, confidence c_i,t, the "
        "deterministic breaker action, the macro lockout window and the headline "
        "feed. Also reports kill-switch status when execution is halted. "
        "``scope=strategy`` (default) keeps the legacy strategy-root behaviour; "
        "``scope=cached`` covers roots with cached Parquet bars; ``scope=all`` "
        "covers every instrument-master root (45, sector-filterable). Roots without "
        "a live/cached snapshot are deterministically derived and listed in "
        "``derived_roots`` with honest ``assessment_source`` provenance."
    ),
    responses={422: {"description": "Unknown sector or roots filter"}},
)
def risk_state(
    container: ContainerDep,
    scope: Literal["strategy", "cached", "all"] = Query(
        default="strategy",
        description="Root scope: strategy (default), cached Parquet roots, or all 45 roots.",
    ),
    sector: str | None = Query(
        default=None,
        description="Optional instrument-master sector filter (e.g. equity, energy, rates).",
    ),
    roots: str | None = Query(
        default=None,
        description="Optional comma-separated explicit futures roots (overrides scope).",
    ),
) -> RiskStateResponse:
    """Return the LLM Risk-First state for the requested root scope."""
    try:
        return container.risk_state_scoped(scope=scope, sector=sector, roots=roots)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc


@router.get(
    "/risk/llm-status",
    response_model=LLMStatusResponse,
    summary="LLM provider health/status",
    description=(
        "Honest status of the configured LLM provider: provider/model, key "
        "configuration, live_api vs offline_stub mode, last ping status "
        "(online/offline_stub/auth_error/unreachable), latency, last check time, "
        "assessment-cache entry count and an error detail when degraded. Read-only: "
        "never performs network I/O and never invents health."
    ),
)
def llm_status(container: ContainerDep) -> LLMStatusResponse:
    """Return the current LLM provider status snapshot."""
    return container.llm_status_view()


@router.post(
    "/risk/llm-ping",
    response_model=LLMStatusResponse,
    summary="Bounded LLM provider ping",
    description=(
        "Issues one tiny completion to the configured LLM transport with a hard "
        "~5s timeout, measures latency and classifies the result as online, "
        "auth_error or unreachable (offline_stub configurations perform no network "
        "I/O). No order, risk-budget or trading side effects."
    ),
)
async def llm_ping(container: ContainerDep) -> LLMStatusResponse:
    """Probe the configured LLM provider once and return the updated status."""
    return await container.llm_ping()


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
