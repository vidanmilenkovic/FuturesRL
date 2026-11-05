"""FastAPI dependency providers.

The app factory stores one :class:`~src.api.services.Container` on
``app.state`` during startup; routes depend on the service *protocols* through the
annotated dependency aliases below. Tests can therefore override the container or
individual services with deterministic fakes without touching route code.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Annotated, cast

from configs.settings import Settings, get_settings
from fastapi import Depends, Request

from src.api import fixtures
from src.api.bridge_pipeline import BridgePipelineService
from src.api.optimize_service import (
    OptimizationApplyRequest,
    OptimizationService,
    OptimizationStudy,
)
from src.api.protocols import (
    BacktestService,
    ExecutionService,
    LiveStateService,
    MarketDataService,
    RiskService,
    StrategyCatalog,
)
from src.api.schemas import BacktestRunConfig, StrategyConfigView
from src.api.services import Container
from src.api.studio_service import StrategyStudioService

__all__ = [
    "BacktestDep",
    "BridgePipelineDep",
    "ContainerDep",
    "ExecutionDep",
    "LiveDep",
    "MarketDataDep",
    "OptimizeDep",
    "RiskDep",
    "StrategyCatalogDep",
    "StudioDep",
    "get_backtests",
    "get_bridge_pipeline_service",
    "get_container",
    "get_execution",
    "get_live",
    "get_market_data",
    "get_optimize_service",
    "get_risk",
    "get_strategies",
    "get_studio_service",
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


def get_live(container: ContainerDep) -> LiveStateService:
    """Runner bridge / live-state service protocol."""
    return container.live


MarketDataDep = Annotated[MarketDataService, Depends(get_market_data)]
ExecutionDep = Annotated[ExecutionService, Depends(get_execution)]
RiskDep = Annotated[RiskService, Depends(get_risk)]
StrategyCatalogDep = Annotated[StrategyCatalog, Depends(get_strategies)]
BacktestDep = Annotated[BacktestService, Depends(get_backtests)]
LiveDep = Annotated[LiveStateService, Depends(get_live)]


# -- Strategy Studio / optimize / bridge dependency providers -----------------

type VerificationRunner = Callable[
    [OptimizationApplyRequest, OptimizationStudy, StrategyConfigView], Mapping[str, object] | None
]


def _settings_and_clock(request: Request) -> tuple[Settings, fixtures.Clock]:
    container = getattr(request.app.state, "container", None)
    if container is not None:
        return cast(Settings, container.settings), cast(fixtures.Clock, container.clock)
    return get_settings(), fixtures.utc_now


def get_studio_service(request: Request) -> StrategyStudioService:
    """Return (and cache) the Strategy Studio service for this app."""
    cached = getattr(request.app.state, "studio_service", None)
    if isinstance(cached, StrategyStudioService):
        return cached
    settings, clock = _settings_and_clock(request)
    container = getattr(request.app.state, "container", None)
    catalog = getattr(container, "strategies", None) if container is not None else None
    service = StrategyStudioService(settings=settings, clock=clock, catalog=catalog)
    request.app.state.studio_service = service
    return service


def _build_verification_runner(
    container: object | None, studio: StrategyStudioService
) -> VerificationRunner | None:
    backtests = getattr(container, "backtests", None) if container is not None else None
    if backtests is None:
        return None

    def runner(
        apply_request: OptimizationApplyRequest,
        study: OptimizationStudy,
        strategy: StrategyConfigView,
    ) -> dict[str, object]:
        base = studio.core_config(study.base_strategy_id)
        spec = base.backtest if base is not None else None
        start = apply_request.start or (spec.start if spec is not None else None)
        end = apply_request.end or (spec.end if spec is not None else None)
        if start is None or end is None:
            return {
                "status": "skipped",
                "message": (
                    "no verification window: provide start/end or a base strategy backtest block"
                ),
            }
        request = BacktestRunConfig(
            strategy_id=strategy.id,
            start=start,
            end=end,
            initial_equity=apply_request.initial_equity
            or (spec.initial_equity if spec is not None else 1_000_000.0),
            rebalance_frequency=spec.rebalance_frequency if spec is not None else "W-FRI",
            walk_forward=True,
            random_seed=spec.random_seed if spec is not None else 7,
        )
        try:
            summary = backtests.run(request)
        except KeyError:
            return {
                "status": "skipped",
                "message": (
                    f"strategy {strategy.id!r} is not registered with the backtest service yet; "
                    "retry verification after the API reloads the strategy catalog"
                ),
            }
        return {"status": "completed", "run": summary.model_dump(mode="json")}

    return runner


def get_optimize_service(request: Request) -> OptimizationService:
    """Return (and cache) the optimization service for this app."""
    cached = getattr(request.app.state, "optimize_service", None)
    if isinstance(cached, OptimizationService):
        return cached
    studio = get_studio_service(request)
    settings, clock = _settings_and_clock(request)
    container = getattr(request.app.state, "container", None)
    service = OptimizationService(
        settings=settings,
        studio=studio,
        clock=clock,
        backtest_runner=_build_verification_runner(container, studio),
    )
    request.app.state.optimize_service = service
    return service


def get_bridge_pipeline_service(request: Request) -> BridgePipelineService:
    """Return (and cache) the runner-bridge pipeline mapper for this app."""
    cached = getattr(request.app.state, "bridge_pipeline_service", None)
    if isinstance(cached, BridgePipelineService):
        return cached
    settings, clock = _settings_and_clock(request)
    container = getattr(request.app.state, "container", None)
    master = getattr(container, "instrument_master", None) if container is not None else None
    service = BridgePipelineService(
        settings=settings,
        clock=clock,
        instrument_master=master if master is not None else settings.instruments,
    )
    request.app.state.bridge_pipeline_service = service
    return service


StudioDep = Annotated[StrategyStudioService, Depends(get_studio_service)]
OptimizeDep = Annotated[OptimizationService, Depends(get_optimize_service)]
BridgePipelineDep = Annotated[BridgePipelineService, Depends(get_bridge_pipeline_service)]
