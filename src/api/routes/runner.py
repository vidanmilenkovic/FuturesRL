"""Live-runner bridge endpoints: status, offline step and reset (Phase 9)."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import time
from datetime import UTC, datetime
from typing import Annotated, cast

from fastapi import APIRouter, Depends, HTTPException, Request, status

from src.api.deps import ContainerDep, LiveDep
from src.api.live import RunnerSessionUnavailableError
from src.api.rebalance_service import RebalanceError, RebalanceService
from src.api.schemas import (
    IbkrActiveConnection,
    IbkrConnectionSwitchRequest,
    IbkrConnectionSwitchResponse,
    IbkrPortInfo,
    IbkrPortsResponse,
    IbkrRequestedConnection,
    RebalanceExecuteRequest,
    RebalanceExecuteResponse,
    RebalancePreviewRequest,
    RebalancePreviewResponse,
    RunnerStatusResponse,
    RunnerStepRequest,
    RunnerStepResponse,
)

router = APIRouter(tags=["runner"])


def get_rebalance_service(request: Request) -> RebalanceService:
    """Return (and cache) the Phase-11A rebalance preview/execute service.

    Tests can inject ``app.state.rebalance_service`` with fakes; production builds
    it from the live container (settings/market/catalog/execution/clock).
    """
    injected = getattr(request.app.state, "rebalance_service", None)
    if (
        injected is not None
        and callable(getattr(injected, "preview", None))
        and callable(getattr(injected, "execute", None))
    ):
        return cast(RebalanceService, injected)
    container = getattr(request.app.state, "container", None)
    if container is None:  # pragma: no cover - create_app always builds a container
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="service container is unavailable",
        )
    service = RebalanceService(
        settings=cast("object", container.settings),  # type: ignore[arg-type]
        market=container.market,
        catalog=container.strategies,
        execution=container.execution,
        clock=container.clock,
    )
    request.app.state.rebalance_service = service
    return service


RebalanceDep = Annotated[RebalanceService, Depends(get_rebalance_service)]


@router.get(
    "/runner/status",
    response_model=RunnerStatusResponse,
    summary="Live runner bridge status",
    description=(
        "Runner bridge availability/staleness (runner id, mode, sequence, updated_at, "
        "stale_after_seconds, status) plus the API's singleton offline-runner session "
        "state. Never 503s: ``bridge_available=false``/``source=unavailable`` when the "
        "runner has never published a snapshot."
    ),
)
def runner_status(live: LiveDep) -> RunnerStatusResponse:
    """Return bridge staleness and session state."""
    return live.runner_status()


@router.post(
    "/runner/step",
    response_model=RunnerStepResponse,
    summary="Advance the offline runner",
    description=(
        "Warm-start (once) and advance the singleton offline ``OfflineRunnerSession`` "
        "by ``states`` MarketStates, then return the refreshed bridge summary. "
        "Returns 503 with an actionable message when the execution package does not "
        "ship ``OfflineRunnerSession`` in this deployment."
    ),
    responses={503: {"description": "Offline runner session unavailable"}},
)
async def runner_step(payload: RunnerStepRequest, live: LiveDep) -> RunnerStepResponse:
    """Advance the offline runner and report the bridge snapshot."""
    try:
        return await live.runner_step(payload)
    except RunnerSessionUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"offline runner session unavailable: {exc}. Start the live runner "
                "(``python -m src.execution.runner``) to populate the bridge, or use "
                "GET /api/runner/status for bridge-only telemetry."
            ),
        ) from exc


@router.post(
    "/runner/reset",
    response_model=RunnerStatusResponse,
    summary="Reset the offline runner session",
    description="Closes the singleton offline runner session; the bridge snapshot is retained.",
)
async def runner_reset(live: LiveDep) -> RunnerStatusResponse:
    """Close the offline runner session and return the status."""
    return await live.runner_reset()


@router.post(
    "/runner/rebalance-preview",
    response_model=RebalancePreviewResponse,
    summary="Preview the next runner rebalance (no orders submitted)",
    description=(
        "Hydrates the latest state with the current broker NLV/currency/positions, runs "
        "the full S->A->T->R pipeline + contract translator, computes delta orders against "
        "the current book and reports the per-order pre-trade gate verdict, projected "
        "margin utilization (60% cap), tracking error and mode (``ibkr-paper`` when a "
        "connected IBKR paper adapter is active, else ``offline``). Never submits orders."
    ),
    responses={
        404: {"description": "Unknown strategy id"},
        422: {"description": "Invalid strategy/roots payload"},
        503: {"description": "Execution book unavailable"},
    },
)
async def rebalance_preview(
    payload: RebalancePreviewRequest, rebalance: RebalanceDep
) -> RebalancePreviewResponse:
    """Run the full pipeline and return proposed delta orders (no submission)."""
    try:
        return await rebalance.preview(payload)
    except RebalanceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.post(
    "/runner/rebalance-execute",
    response_model=RebalanceExecuteResponse,
    summary="Execute the next runner rebalance (paper/offline only)",
    description=(
        "Recomputes the preview plan and submits ONLY gate-approved delta orders through "
        "the active paper OMS (IBKR paper when connected) or the simulated broker, "
        "refreshes the book and publishes a RunnerBridgeStore snapshot. ``confirm=true`` "
        "is required. Returns 409 in the live environment, for unsafe/unconnected IBKR "
        "sessions, or when a fixture-only offline adapter cannot execute; repeated "
        "identical executions are idempotent (no duplicate submission)."
    ),
    responses={
        400: {"description": "Missing confirmation"},
        404: {"description": "Unknown strategy id"},
        409: {"description": "Live/unsafe execution or no executable adapter"},
        422: {"description": "Invalid strategy/roots payload"},
    },
)
async def rebalance_execute(
    payload: RebalanceExecuteRequest, rebalance: RebalanceDep
) -> RebalanceExecuteResponse:
    """Execute the approved rebalance delta through the active safe adapter."""
    try:
        return await rebalance.execute(payload)
    except RebalanceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


# -- IBKR port discovery + guarded connection switch --------------------------
#: Probe table: port -> (mode, gateway) derived from IBKR's documented ports.
IBKR_CONNECTION_PORTS: tuple[tuple[int, str, str], ...] = (
    (7497, "paper", "tws"),
    (4002, "paper", "ib_gateway"),
    (7496, "live", "tws"),
    (4001, "live", "ib_gateway"),
)

#: Non-destructive TCP probe timeout (seconds).
IBKR_PORT_PROBE_TIMEOUT_SECONDS: float = 0.25


def ibkr_port_metadata(port: int) -> tuple[str, str] | None:
    """``(mode, gateway)`` for a known IBKR port, or ``None`` when unknown."""
    for known_port, mode, gateway in IBKR_CONNECTION_PORTS:
        if known_port == port:
            return mode, gateway
    return None


async def probe_ibkr_port(
    host: str, port: int, *, timeout: float = IBKR_PORT_PROBE_TIMEOUT_SECONDS
) -> tuple[bool, float | None]:
    """Non-destructive TCP reachability probe; never opens an IBKR session."""
    started = time.perf_counter()
    try:
        _reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=timeout
        )
    except (OSError, TimeoutError):
        return False, None
    else:
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()
        return True, round((time.perf_counter() - started) * 1000.0, 1)


def _active_connection(settings: object) -> IbkrActiveConnection:
    port = int(getattr(settings, "ibkr_port", 7497))
    metadata = ibkr_port_metadata(port)
    return IbkrActiveConnection(
        port=port,
        client_id=int(getattr(settings, "ibkr_client_id", 0)),
        environment=str(getattr(settings, "environment", "backtest")),
        mode=cast("object", metadata[0] if metadata else None),  # type: ignore[arg-type]
        gateway=cast("object", metadata[1] if metadata else None),  # type: ignore[arg-type]
    )


@router.get(
    "/runner/ibkr-ports",
    response_model=IbkrPortsResponse,
    summary="Probe IBKR TWS/IB Gateway ports (non-destructive)",
    description=(
        "Runs short (250ms) TCP reachability probes against the documented IBKR ports "
        "(7497 TWS Paper, 4002 IB Gateway Paper, 7496 TWS Live, 4001 IB Gateway Live) "
        "and reports the active configured port/client id/environment. No IBKR session "
        "is opened, no orders are touched and no connection state changes."
    ),
)
async def runner_ibkr_ports(container: ContainerDep) -> IbkrPortsResponse:
    """Return IBKR port reachability plus the active configured connection."""
    settings = container.settings
    host = str(getattr(settings, "ibkr_host", "127.0.0.1"))
    probes = await asyncio.gather(
        *(probe_ibkr_port(host, port) for port, _mode, _gateway in IBKR_CONNECTION_PORTS)
    )
    ports = [
        IbkrPortInfo(
            port=port,
            mode=cast("object", mode),  # type: ignore[arg-type]
            gateway=cast("object", gateway),  # type: ignore[arg-type]
            reachable=reachable,
            latency_ms=latency,
        )
        for (port, mode, gateway), (reachable, latency) in zip(
            IBKR_CONNECTION_PORTS, probes, strict=True
        )
    ]
    warnings: list[str] = []
    if not any(info.reachable for info in ports):
        warnings.append(
            f"no reachable IBKR TWS/IB Gateway port on {host}; start TWS or IB Gateway "
            "(paper 7497/4002, live 7496/4001)"
        )
    active = _active_connection(settings)
    if active.mode is None:
        warnings.append(f"configured port {active.port} is not a documented IBKR TWS/Gateway port")
    return IbkrPortsResponse(
        host=host,
        ports=ports,
        active=active,
        warnings=warnings,
        probed_at=datetime.now(UTC),
    )


async def _attempt_runtime_switch(
    container: object, port: int, client_id: int
) -> tuple[bool, list[str]]:
    """Best-effort safe runtime reconnect; returns ``(applied, warnings)``.

    Only a *successful* adapter/manager reconnect is reported as applied. Anything
    else (no support, failed reconnect, errors) returns not-applied so the caller can
    report ``restart_required`` instead of falsely claiming a switch.
    """
    execution = getattr(container, "execution", None)
    target = execution
    for name in ("switch_connection", "reconnect_connection", "reconnect_to"):
        method = getattr(target, name, None)
        if callable(method):
            try:
                result = method(port, client_id)
                if inspect.isawaitable(result):
                    result = await result
            except Exception as exc:
                return False, [f"runtime reconnect failed: {type(exc).__name__}: {exc}"]
            if result is True or (isinstance(result, dict) and result.get("applied") is True):
                return True, []
            return False, ["execution adapter declined the runtime switch"]
    return False, ["active execution adapter does not support a safe runtime connection switch"]


@router.post(
    "/runner/ibkr-connection",
    response_model=IbkrConnectionSwitchResponse,
    summary="Guarded IBKR connection switch (paper/live)",
    description=(
        "Requests a connection target change. Paper ports (7497/4002) need no phrase; "
        'live ports (7496/4001) require ``confirmation == "LIVE"`` and are refused '
        "with 409 in a paper environment. The endpoint only reports ``applied=true`` "
        "when a safe runtime reconnect actually succeeded; otherwise it returns "
        '``status="restart_required"`` with an actionable message (it never claims an '
        "unsafe switch happened)."
    ),
    responses={
        400: {"description": "Missing/invalid live confirmation"},
        409: {"description": "Live requested in a paper environment / refused switch"},
        422: {"description": "Invalid port/client id"},
    },
)
async def runner_ibkr_connection(
    payload: IbkrConnectionSwitchRequest, container: ContainerDep
) -> IbkrConnectionSwitchResponse:
    """Validate and (when safe) apply an IBKR connection target change."""
    metadata = ibkr_port_metadata(payload.port)
    if metadata is None:
        raise HTTPException(
            status_code=422,
            detail=(
                f"unsupported IBKR port {payload.port}; expected one of "
                f"{[port for port, _mode, _gateway in IBKR_CONNECTION_PORTS]}"
            ),
        )
    mode, gateway = metadata
    settings = container.settings
    environment = str(getattr(settings, "environment", "backtest"))
    if mode == "live":
        if environment == "paper":
            raise HTTPException(
                status_code=409,
                detail=(
                    "refusing a live IBKR connection switch in the paper environment; "
                    "switch the deployment environment explicitly before connecting live"
                ),
            )
        if payload.confirmation != "LIVE":
            raise HTTPException(
                status_code=400,
                detail='live IBKR connection switch requires confirmation exactly "LIVE"',
            )
    requested_client_id = (
        int(payload.client_id)
        if payload.client_id is not None
        else int(getattr(settings, "ibkr_client_id", 0))
    )
    requested = IbkrRequestedConnection(
        port=payload.port,
        mode=cast("object", mode),  # type: ignore[arg-type]
        gateway=cast("object", gateway),  # type: ignore[arg-type]
        client_id=requested_client_id,
    )
    active = _active_connection(settings)
    applied, warnings = await _attempt_runtime_switch(container, payload.port, requested_client_id)
    if applied:
        active = IbkrActiveConnection(
            port=payload.port,
            client_id=requested_client_id,
            environment=environment,
            mode=requested.mode,
            gateway=requested.gateway,
        )
        return IbkrConnectionSwitchResponse(
            status="applied",
            applied=True,
            restart_required=False,
            requested=requested,
            active=active,
            environment=environment,
            warnings=warnings,
            message=(
                f"IBKR connection switched to {requested.gateway}/{requested.mode} port "
                f"{requested.port} (client id {requested_client_id})"
            ),
        )
    warnings.append(
        "restart the API with the requested IBKR port/client id to apply the switch "
        f"(FUTURESRLX_IBKR_PORT={requested.port}, "
        f"FUTURESRLX_IBKR_CLIENT_ID={requested_client_id})"
    )
    return IbkrConnectionSwitchResponse(
        status="restart_required",
        applied=False,
        restart_required=True,
        requested=requested,
        active=active,
        environment=environment,
        warnings=warnings,
        message=(
            "no safe runtime reconnect is available for this adapter; the connection "
            "was NOT switched. Restart the API with the requested IBKR port/client id."
        ),
    )
