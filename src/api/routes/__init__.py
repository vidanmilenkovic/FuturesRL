"""REST router registry (mounted under ``/api`` by the app factory)."""

from __future__ import annotations

from fastapi import APIRouter

from src.api.routes import (
    backtests,
    pipeline,
    positions,
    risk,
    strategies,
    telemetry,
    term_structure,
)

#: Every REST router mounted at ``/api`` by :func:`src.api.main.create_app`.
ROUTERS: tuple[APIRouter, ...] = (
    positions.router,
    pipeline.router,
    risk.router,
    term_structure.router,
    strategies.router,
    backtests.router,
    telemetry.router,
)

__all__ = ["ROUTERS"]
