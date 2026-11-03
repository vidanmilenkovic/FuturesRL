"""Additive QoL REST models: artifact deletion receipts and strategy identity bodies.

Kept in a dedicated module so the shared ``src/api/schemas.py`` only gains the
backtest snapshot/trade fields (coordinated with the risk/maintenance stream).
"""

from __future__ import annotations

from pydantic import Field

from src.api.schemas import APIModel, StrictModel

__all__ = ["ArtifactDeleteResponse", "StrategyIdentityRequest"]


class ArtifactDeleteResponse(APIModel):
    """Uniform receipt for ``DELETE`` optimization/backtest/strategy endpoints."""

    id: str
    deleted: bool = True
    #: Whether an on-disk artifact/YAML file was removed (fixture/in-memory-only rows
    #: may be removed from the store without a persisted file).
    artifact_deleted: bool = False
    #: File path that was removed, when applicable (strategies only by default).
    path: str | None = None


class StrategyIdentityRequest(StrictModel):
    """Body for ``POST /api/strategies/{id}/clone`` and ``/rename``."""

    new_strategy_id: str = Field(min_length=1, max_length=64)
