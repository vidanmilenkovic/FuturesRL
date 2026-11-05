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


def _tcp_reachable(host: str, port: int, *, timeout: float = 0.5) -> bool:
    """Short TCP reachability probe (never opens an IBKR session itself)."""
    import socket

    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def _make_ibkr_fetcher(settings: object, instrument_master: object, timeframe: str) -> object:
    """Build a dedicated-session historical fetcher with a non-clashing client id."""
    from src.api.data_service import _ibkr_bar_size
    from src.data.ibkr_data import IBKRHistoricalBarFetcher

    copied = settings.model_copy(  # type: ignore[attr-defined]
        update={"ibkr_client_id": min(int(settings.ibkr_client_id) + 3, 999)}  # type: ignore[attr-defined]
    )
    return IBKRHistoricalBarFetcher(
        copied,
        instrument_master=instrument_master,  # type: ignore[arg-type]
        bar_size=_ibkr_bar_size(timeframe),
    )


def get_data_service(request: Request) -> DataService:
    """Return (and cache) the data-cache service (tests may inject a fake)."""
    injected = getattr(request.app.state, "data_service", None)
    if callable(getattr(injected, "inventory", None)):
        return cast(DataService, injected)
    container = getattr(request.app.state, "container", None)
    if container is None:  # pragma: no cover - create_app always builds one
        raise HTTPException(status_code=503, detail="service container is unavailable")
    settings = cast("object", container.settings)
    execution = getattr(container, "execution", None)
    master = container.instrument_master

    def ibkr_available() -> bool:
        if bool(getattr(execution, "connected", False)):
            return True
        return _tcp_reachable(settings.ibkr_host, settings.ibkr_port)  # type: ignore[attr-defined]

    def ibkr_fetcher(timeframe: str = "1d") -> object:
        return _make_ibkr_fetcher(settings, master, timeframe)

    service = DataService(
        settings=settings,  # type: ignore[arg-type]
        instrument_master=master,
        clock=container.clock,
        ibkr_available_provider=ibkr_available,
        ibkr_fetcher_provider=ibkr_fetcher,
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
def data_inventory(
    data: DataDep,
    timeframe: str = "1d",
    schema: str | None = None,
) -> DataInventoryResponse:
    """Return the Phase-11B cache inventory for one bar timeframe."""
    try:
        return data.inventory(timeframe=timeframe, schema=schema)
    except DataServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail or str(exc)) from exc


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
        400: {"description": "Databento API key/SDK unavailable (FUTURESRLX_DATABENTO_API_KEY)"},
        402: {"description": "Databento cost guard exceeded (structured detail + override flag)"},
        422: {"description": "Invalid roots/window/timeframe"},
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
        400: {"description": "Missing Databento API key or override confirmation"},
        402: {"description": "Databento cost guard exceeded (structured detail + override flag)"},
        409: {"description": "Unsafe/refused operation (e.g. live environment)"},
        422: {"description": "Invalid roots/window/timeframe"},
        503: {"description": "IBKR TWS/IB Gateway unreachable on host:port"},
    },
)
async def ingest_data(payload: IngestRequest, data: DataDep) -> IngestResponse:
    """Ingest/cache bars for the requested roots and window."""
    try:
        return await run_in_threadpool(data.ingest, payload)
    except DataServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail or str(exc)) from exc
