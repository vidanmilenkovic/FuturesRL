"""FastAPI application factory for the FuturesRL-X telemetry backend.

Run locally::

    uvicorn src.api.main:app --host 0.0.0.0 --port 8000

The module-level :data:`app` is deliberately cheap to import: the service
container (offline market fixture, simulated broker, pipeline caches) is built by
the lifespan on startup, so ``import src.api.main`` never touches the data or
execution layers.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from configs.settings import Settings, get_settings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api import fixtures
from src.api.routes import ROUTERS
from src.api.routes.health import router as health_router
from src.api.services import Container, build_container
from src.api.version import API_VERSION
from src.api.ws import router as websocket_router

logger = logging.getLogger(__name__)


#: OpenAPI tag metadata (orders the generated docs).
OPENAPI_TAGS: list[dict[str, str]] = [
    {"name": "system", "description": "Liveness / readiness probes."},
    {
        "name": "positions",
        "description": "Account valuation and active futures positions (Command Center).",
    },
    {
        "name": "pipeline",
        "description": "FinRL-X weight pipeline waterfall and stage diagnostics.",
    },
    {
        "name": "risk",
        "description": "LLM Risk-First assessments, threshold overrides and the kill switch.",
    },
    {
        "name": "term-structure",
        "description": "Forward curves, carry and the futures roll calendar.",
    },
    {
        "name": "strategies",
        "description": "Declarative YAML strategy configs (validated in-memory editor).",
    },
    {
        "name": "backtests",
        "description": "Offline deterministic backtests and institutional tear sheets.",
    },
    {
        "name": "telemetry",
        "description": "WebSocket telemetry broadcaster status.",
    },
    {
        "name": "orders",
        "description": "Working orders and recent fills (Command Center live book).",
    },
    {
        "name": "runner",
        "description": "Live-runner bridge status and the offline runner step/reset controls.",
    },
]


def create_app(
    settings: Settings | None = None,
    *,
    clock: fixtures.Clock | None = None,
    container: Container | None = None,
) -> FastAPI:
    """Build the FastAPI application.

    Parameters
    ----------
    settings:
        Runtime settings (defaults to the process-wide :func:`get_settings`).
    clock:
        Injectable UTC clock for deterministic tests; the app uses it for macro
        countdowns, timestamps and the offline fixture window.
    container:
        Pre-built service container (tests / embedding). When omitted the app
        builds one in its lifespan, so importing this module stays cheap.
    """
    resolved_settings = settings if settings is not None else get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        active = (
            container if container is not None else build_container(resolved_settings, clock=clock)
        )
        for warning in active.warnings:
            logger.warning("service wiring: %s", warning)
        app.state.container = active
        await active.start()
        logger.info(
            "FuturesRL-X API ready (environment=%s, offline_mode=%s)",
            resolved_settings.environment,
            active.offline_mode,
        )
        try:
            yield
        finally:
            await active.stop()

    application = FastAPI(
        title="FuturesRL-X Telemetry API",
        version=API_VERSION,
        summary="Futures-native FinRL-X telemetry, risk and backtest API.",
        description=(
            "Real-time REST + WebSocket backend for the FuturesRL-X dashboard. "
            "Exposes the FinRL-X weight pipeline trace, the LLM Risk-First state "
            "and kill switch, term structure/roll calendar, declarative strategy "
            "configs and deterministic offline backtest tear sheets. Runs fully "
            "offline when no IBKR session or Databento key is configured."
        ),
        lifespan=lifespan,
        openapi_url="/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_tags=OPENAPI_TAGS,
        contact={"name": "FuturesRL-X Team"},
        license_info={"name": "Proprietary"},
    )
    # Development convenience: allow the documented dashboard ports
    # (next dev :3100, compose :3080) without mutating the frozen settings
    # contract. Paper/live environments use exactly the configured origins.
    cors_origins = list(resolved_settings.api_cors_origins)
    if resolved_settings.environment == "development":
        for extra in (
            "http://localhost:3100",
            "http://127.0.0.1:3100",
            "http://localhost:3080",
            "http://127.0.0.1:3080",
        ):
            if extra not in cors_origins:
                cors_origins.append(extra)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    application.include_router(health_router)
    application.include_router(websocket_router)
    for router in ROUTERS:
        application.include_router(router, prefix="/api")
    return application


#: Uvicorn entry point (``uvicorn src.api.main:app``).
app: FastAPI = create_app()


def main() -> None:
    """CLI entry point (``futuresrl-api``): run uvicorn with the configured settings."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower(),
        proxy_headers=True,
    )


if __name__ == "__main__":  # pragma: no cover - process entry point
    main()
