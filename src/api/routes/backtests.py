"""Backtest Lab endpoints: run, list, inspect and tear sheet."""

from __future__ import annotations

import csv
import io

from fastapi import APIRouter, HTTPException, Response, status
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from src.api.backtests_service import CachedDataUnavailableError
from src.api.deps import BacktestDep
from src.api.schemas import (
    ArtifactDeleteResponse,
    BacktestListResponse,
    BacktestRunConfig,
    BacktestRunSummary,
    TearsheetResponse,
    TradeRow,
)

#: CSV header contract for ``GET /api/backtests/{run_id}/trades.csv``.
TRADE_CSV_HEADERS: tuple[str, ...] = (
    "timestamp",
    "order_id",
    "instrument_id",
    "side",
    "quantity",
    "price",
    "notional",
    "commissions",
    "slippage",
    "kind",
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
    except CachedDataUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
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


def _trades_csv(trades: list[TradeRow]) -> str:
    """Render the additive trade tape as CSV (header-only when the tape is empty)."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(TRADE_CSV_HEADERS)
    for trade in trades:
        writer.writerow(
            (
                trade.timestamp.isoformat(),
                trade.order_id,
                trade.instrument_id,
                trade.side,
                trade.quantity,
                trade.price,
                trade.notional,
                trade.commissions,
                trade.slippage,
                trade.kind,
            )
        )
    return buffer.getvalue()


@router.delete(
    "/backtests/{run_id}",
    response_model=ArtifactDeleteResponse,
    summary="Delete a backtest run and its artifact",
    description=(
        "Removes the run summary/tear sheet from the in-memory store and deletes the "
        "persisted ``backtests/<run_id>.json`` artifact. Seeded fixture runs (and "
        "engine failures) live in memory only; deleting them removes the row for the "
        "current process, while boot seeds regenerate after an API restart. Unknown "
        "ids return 404; the receipt reports ``artifact_deleted``."
    ),
    responses={404: {"description": "Unknown run id"}},
)
def delete_backtest(run_id: str, backtests: BacktestDep) -> ArtifactDeleteResponse:
    """Delete one stored backtest run."""
    try:
        artifact_deleted = backtests.delete_run(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return ArtifactDeleteResponse(
        id=run_id,
        deleted=True,
        artifact_deleted=artifact_deleted,
        path=str(backtests.artifact_path(run_id)) if artifact_deleted else None,
    )


@router.get(
    "/backtests/{run_id}/export.json",
    summary="Export a full backtest run as JSON",
    description=(
        "Attachment (``application/json``) containing the full run summary (including "
        "the additive ``strategy_config`` snapshot), the tear sheet (including the "
        "``trades`` tape) and a top-level ``strategy_config`` alias for convenience. "
        "Works for legacy artifacts too (snapshot may be null, trades empty)."
    ),
    response_class=JSONResponse,
    responses={
        200: {"content": {"application/json": {}}, "description": "Backtest JSON export"},
        404: {"description": "Unknown run id"},
    },
)
def export_backtest_json(run_id: str, backtests: BacktestDep) -> JSONResponse:
    """Download one run's summary + tear sheet + strategy config as JSON."""
    summary = backtests.get_run(run_id)
    tearsheet = backtests.tearsheet(run_id)
    if summary is None or tearsheet is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"unknown backtest run {run_id!r}"
        )
    payload = {
        "summary": summary.model_dump(mode="json"),
        "tearsheet": tearsheet.model_dump(mode="json"),
        "strategy_config": summary.strategy_config,
    }
    return JSONResponse(
        content=payload,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{run_id}-export.json"'},
    )


@router.get(
    "/backtests/{run_id}/trades.csv",
    summary="Export a backtest trade tape as CSV",
    description=(
        "Attachment (``text/csv``) with the header "
        "``timestamp,order_id,instrument_id,side,quantity,price,notional,commissions,"
        "slippage,kind`` and one row per executed trade. Fixture/legacy runs without a "
        "tape return the header-only CSV."
    ),
    response_class=Response,
    responses={
        200: {"content": {"text/csv": {}}, "description": "Backtest trades CSV attachment"},
        404: {"description": "Unknown run id"},
    },
)
def export_backtest_trades(run_id: str, backtests: BacktestDep) -> Response:
    """Download one run's trade tape as CSV."""
    tearsheet = backtests.tearsheet(run_id)
    if tearsheet is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"unknown backtest run {run_id!r}"
        )
    return Response(
        content=_trades_csv(tearsheet.trades),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{run_id}-trades.csv"'},
    )
