"""Phase-11B data-cache API: inventory, Databento cost estimate and ingest."""

from __future__ import annotations

from typing import Annotated, cast

from fastapi import APIRouter, Depends, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from src.api.data_service import DataService, DataServiceError
from src.api.phase11b_schemas import (
    CostEstimateResponse,
    DataInventoryResponse,
    EstimateCostRequest,
    IngestRequest,
    IngestResponse,
)

router = APIRouter(tags=["data"])


def get_data_service(request: Request) -> DataService:
    """Return (and cache) the data-cache service (tests may inject a fake)."""
    injected = getattr(request.app.state, "data_service", None)
    if callable(getattr(injected, "inventory", None)):
        return cast(DataService, injected)
    container = getattr(request.app.state, "container", None)
    if container is None:  # pragma: no cover - create_app always builds one
        raise HTTPException(status_code=503, detail="service container is unavailable")
    service = DataService(
        settings=cast("object", container.settings),  # type: ignore[arg-type]
        instrument_master=container.instrument_master,
        clock=container.clock,
    )
    request.app.state.data_service = service
    return service


DataDep = Annotated[DataService, Depends(get_data_service)]


@router.get(
    "/data/inventory",
    response_model=DataInventoryResponse,
    summary="Cached Parquet inventory across the instrument master",
    description=(
        "Per-root cached bars/date range/size/provenance (``databento``/``ibkr``/"
        "``empty``) for every instrument-master root, plus cache totals, whether a "
        "Databento API key is configured, and whether the IBKR paper/TWS ingest path "
        "is available."
    ),
)
def data_inventory(data: DataDep, schema: str = "ohlcv-1d") -> DataInventoryResponse:
    """Return the Phase-11B cache inventory."""
    return data.inventory(schema=schema)


@router.post(
    "/data/estimate-cost",
    response_model=CostEstimateResponse,
    summary="Estimate a Databento ingest cost (or free IBKR cost)",
    description=(
        "Databento: metadata cost/billable-size estimate before any download (400 when "
        "no API key/SDK is configured). IBKR: returns the free-ingest zero-cost shape. "
        "Invalid roots/windows return 422."
    ),
    responses={
        400: {"description": "Databento API key/SDK unavailable"},
        422: {"description": "Invalid roots/window/schema"},
    },
)
async def estimate_cost(payload: EstimateCostRequest, data: DataDep) -> CostEstimateResponse:
    """Estimate ingest cost without downloading anything."""
    try:
        return await run_in_threadpool(data.estimate_cost, payload)
    except DataServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail or str(exc)) from exc


@router.post(
    "/data/ingest",
    response_model=IngestResponse,
    summary="Ingest cached bars from IBKR (free) or Databento (guarded)",
    description=(
        "IBKR: downloads bars through a connected paper TWS/IB Gateway session and "
        "persists them into the Parquet cache (409 when unavailable/unsafe). Databento: "
        "ALWAYS estimates cost first and refuses the download when the per-job budget "
        "is exceeded unless the operator explicitly confirms both ``allow_over_budget`` "
        "and ``confirm`` (409 with the estimate/cap otherwise). Continuous F1/F2 series "
        "are built after a successful ingest."
    ),
    responses={
        400: {"description": "Missing confirmation / Databento unavailable"},
        409: {"description": "IBKR not connected/unsafe or budget exceeded"},
        422: {"description": "Invalid roots/window/schema"},
    },
)
async def ingest_data(payload: IngestRequest, data: DataDep) -> IngestResponse:
    """Ingest/cache bars for the requested roots and window."""
    try:
        return await run_in_threadpool(data.ingest, payload)
    except DataServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail or str(exc)) from exc
