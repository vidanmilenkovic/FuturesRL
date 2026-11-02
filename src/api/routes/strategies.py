"""Declarative strategy config endpoints (in-memory YAML editor)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from src.api.deps import StrategyCatalogDep
from src.api.schemas import StrategyConfigUpdate, StrategyConfigView

router = APIRouter(tags=["strategies"])


@router.get(
    "/strategies",
    response_model=list[StrategyConfigView],
    summary="List declarative strategy configs",
    description="Discoverable ``configs/strategies/*.yaml`` pipelines, validated against the frozen schema.",
)
def list_strategies(catalog: StrategyCatalogDep) -> list[StrategyConfigView]:
    """Return every strategy config, sorted by id."""
    return catalog.list_configs()


@router.get(
    "/strategies/{strategy_id}",
    response_model=StrategyConfigView,
    summary="Get one strategy config",
    responses={404: {"description": "Unknown strategy id"}},
)
def get_strategy(strategy_id: str, catalog: StrategyCatalogDep) -> StrategyConfigView:
    """Return one strategy config by id."""
    view = catalog.get(strategy_id)
    if view is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"unknown strategy {strategy_id!r}",
        )
    return view


@router.put(
    "/strategies/{strategy_id}",
    response_model=StrategyConfigView,
    summary="Validate and edit a strategy config (in memory)",
    description=(
        "Merge a partial edit into the in-memory config, re-validate the whole "
        "pipeline with the frozen ``configs.strategy`` models (unknown stages, "
        "bad params, invalid windows are rejected with 422) and return the stored "
        "config. The committed YAML is never written."
    ),
    responses={
        404: {"description": "Unknown strategy id"},
        422: {"description": "Edited config failed validation"},
    },
)
def update_strategy(
    strategy_id: str,
    payload: StrategyConfigUpdate,
    catalog: StrategyCatalogDep,
) -> StrategyConfigView:
    """Apply and validate a partial config edit."""
    try:
        view = catalog.update(strategy_id, payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"invalid strategy config: {exc}") from exc
    if view is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"unknown strategy {strategy_id!r}",
        )
    return view
