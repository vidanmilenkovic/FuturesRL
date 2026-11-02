"""Backtest Lab endpoints: run, list, inspect and tear sheet."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from starlette.concurrency import run_in_threadpool

from src.api.deps import BacktestDep
from src.api.schemas import (
    BacktestListResponse,
    BacktestRunConfig,
    BacktestRunSummary,
    TearsheetResponse,
)

router = APIRouter(tags=["backtests"])


@router.get(
    "/backtests",
    response_model=BacktestListResponse,
    summary="List backtest runs",
    description=(
        "Run history (newest first) plus the strategy ids available for a new "
        "run. The API seeds deterministic fixture runs (``data_source=fixture``) at "
        "boot so the tear-sheet comparison view is populated immediately."
    ),
)
def list_backtests(backtests: BacktestDep) -> BacktestListResponse:
    """Return the run history and available strategies."""
    return backtests.list_runs()


@router.post(
    "/backtests",
    response_model=BacktestRunSummary,
    summary="Run a deterministic offline backtest",
    description=(
        "Executes the production ``BacktestEngine`` (vectorised mode) over the "
        "deterministic offline market window — no Databento, no IBKR, no network. "
        "Returns the stored run summary including the full institutional metric "
        "set; fetch the series via ``GET /api/backtests/{run_id}/tearsheet``."
    ),
    responses={
        404: {"description": "Unknown strategy id"},
        500: {"description": "Backtest execution failed"},
    },
)
async def create_backtest(
    payload: BacktestRunConfig,
    backtests: BacktestDep,
) -> BacktestRunSummary:
    """Run a backtest through the engine adapter (thread-pooled, non-blocking)."""
    try:
        return await run_in_threadpool(backtests.run, payload)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get(
    "/backtests/{run_id}",
    response_model=BacktestRunSummary,
    summary="Get one backtest run",
    responses={404: {"description": "Unknown run id"}},
)
def get_backtest(run_id: str, backtests: BacktestDep) -> BacktestRunSummary:
    """Return one stored run summary."""
    summary = backtests.get_run(run_id)
    if summary is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"unknown backtest run {run_id!r}"
        )
    return summary


@router.get(
    "/backtests/{run_id}/tearsheet",
    response_model=TearsheetResponse,
    summary="Get a run tear sheet",
    description=(
        "Equity curve, rolling Sharpe, underwater drawdown, Rachev gain/loss tail "
        "distribution, roll-yield attribution, monthly returns and the full metric "
        "set for one run."
    ),
    responses={404: {"description": "Unknown run id"}},
)
def get_tearsheet(run_id: str, backtests: BacktestDep) -> TearsheetResponse:
    """Return the tear sheet for one stored run."""
    tearsheet = backtests.tearsheet(run_id)
    if tearsheet is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"unknown backtest run {run_id!r}"
        )
    return tearsheet
