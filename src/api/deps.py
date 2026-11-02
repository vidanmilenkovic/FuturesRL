"""FastAPI dependency providers.

The app factory stores one :class:`~src.api.services.Container` on
``app.state`` during startup; routes depend on the service *protocols* through the
annotated dependency aliases below. Tests can therefore override the container or
individual services with deterministic fakes without touching route code.
"""

from __future__ import annotations

from typing import Annotated, cast

from fastapi import Depends, Request

from src.api.protocols import (
    BacktestService,
    ExecutionService,
    MarketDataService,
    RiskService,
    StrategyCatalog,
)
from src.api.services import Container

__all__ = [
    "BacktestDep",
    "ContainerDep",
    "ExecutionDep",
    "MarketDataDep",
    "RiskDep",
    "StrategyCatalogDep",
    "get_backtests",
    "get_container",
    "get_execution",
    "get_market_data",
    "get_risk",
    "get_strategies",
    "require_container",
]


def get_container(request: Request) -> Container:
    """Return the process service container created by the lifespan."""
    container = getattr(request.app.state, "container", None)
    if container is None:  # pragma: no cover - only during misconfigured startup
        raise RuntimeError(
            "service container is not initialised; the app must be started through "
            "its lifespan (TestClient context manager or uvicorn)"
        )
    return cast(Container, container)


def require_container(request: Request) -> Container:
    """Alias of :func:`get_container` (used by the WebSocket endpoint)."""
    return get_container(request)


ContainerDep = Annotated[Container, Depends(get_container)]


def get_market_data(container: ContainerDep) -> MarketDataService:
    """Market/pipeline service protocol."""
    return container.market


def get_execution(container: ContainerDep) -> ExecutionService:
    """Execution/account service protocol."""
    return container.execution


def get_risk(container: ContainerDep) -> RiskService:
    """LLM risk service protocol."""
    return container.risk


def get_strategies(container: ContainerDep) -> StrategyCatalog:
    """Declarative strategy catalog protocol."""
    return container.strategies


def get_backtests(container: ContainerDep) -> BacktestService:
    """Backtest run-store/engine protocol."""
    return container.backtests


MarketDataDep = Annotated[MarketDataService, Depends(get_market_data)]
ExecutionDep = Annotated[ExecutionService, Depends(get_execution)]
RiskDep = Annotated[RiskService, Depends(get_risk)]
StrategyCatalogDep = Annotated[StrategyCatalog, Depends(get_strategies)]
BacktestDep = Annotated[BacktestService, Depends(get_backtests)]
