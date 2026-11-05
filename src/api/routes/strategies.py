"""Declarative strategy config endpoints: Strategy Studio catalog, CRUD and YAML persistence."""

from __future__ import annotations

import json
from collections.abc import Mapping

import yaml
from fastapi import APIRouter, HTTPException, Request, Response, status

from src.api.deps import StudioDep
from src.api.schemas import ArtifactDeleteResponse, StrategyConfigView, StrategyIdentityRequest
from src.api.studio_service import (
    MAX_STRATEGY_PAYLOAD_BYTES,
    StrategyCatalogResponse,
    StrategyStudioError,
)

router = APIRouter(tags=["strategies"])


async def _read_strategy_payload(request: Request) -> dict[str, object]:
    """Accept a JSON object or a YAML document as the strategy definition."""
    raw = await request.body()
    if not raw or not raw.strip():
        raise HTTPException(status_code=422, detail="strategy payload must not be empty")
    if len(raw) > MAX_STRATEGY_PAYLOAD_BYTES:
        raise HTTPException(status_code=413, detail="strategy payload is too large")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=422, detail=f"strategy payload must be UTF-8: {exc}"
        ) from exc
    content_type = request.headers.get("content-type", "").lower()
    payload: object
    if "json" in content_type:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=422, detail=f"invalid JSON payload: {exc}") from exc
    else:
        try:
            payload = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise HTTPException(status_code=422, detail=f"invalid YAML payload: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise HTTPException(
            status_code=422, detail="strategy payload must be a JSON/YAML mapping (object)"
        )
    return dict(payload)


def _studio_http_error(exc: StrategyStudioError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail=str(exc))


@router.get(
    "/strategies/schema/catalog",
    response_model=StrategyCatalogResponse,
    summary="Strategy Studio parameter catalog + hypothesis presets",
    description=(
        "Typed parameter registry for the five-stage FinRL-X pipeline (universe, "
        "allocator, timing, risk, translator) with types/defaults/ranges/choices and "
        "the five Phase-10 hypothesis presets (complete StrategyConfig payloads ready "
        "for POST /api/strategies). Consumed lazily from "
        "``src.strategies.parameter_catalog``; ``source``/``preset_source`` report "
        "whether the live registry or the deterministic fallback was used."
    ),
)
def strategy_catalog(studio: StudioDep) -> StrategyCatalogResponse:
    """Return the parameter catalog + presets for the Studio UI."""
    return studio.catalog()


@router.get(
    "/strategies",
    response_model=list[StrategyConfigView],
    summary="List declarative strategy configs",
    description=(
        "Every discoverable strategy: committed ``configs/strategies/*.yaml`` plus "
        "Studio-authored overlays under ``artifacts/strategies`` (overlays win)."
    ),
)
def list_strategies(studio: StudioDep) -> list[StrategyConfigView]:
    """Return every strategy config, sorted by id."""
    return studio.list_configs()


@router.get(
    "/strategies/{strategy_id}",
    response_model=StrategyConfigView,
    summary="Get one strategy config",
    responses={404: {"description": "Unknown strategy id"}},
)
def get_strategy(strategy_id: str, studio: StudioDep) -> StrategyConfigView:
    """Return one strategy config by id."""
    view = studio.get_config(strategy_id)
    if view is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"unknown strategy {strategy_id!r}",
        )
    return view


@router.post(
    "/strategies",
    response_model=StrategyConfigView,
    summary="Create and persist a strategy YAML",
    description=(
        "Validates a complete StrategyConfig payload (universe/allocator required; "
        "stage names and parameters validated against the frozen config models) and "
        "writes it atomically to ``configs/strategies/<id>.yaml`` in "
        "development/paper/live or to the ``artifacts/strategies`` overlay in "
        "test/backtest. The in-memory strategy catalog is refreshed immediately so the "
        "new id is visible to ``GET /api/strategies`` and runnable via "
        "``POST /api/backtests`` without a restart. Ids must be strict slugs; traversal "
        "attempts, duplicates and invalid configs are rejected (400/409/422) before any "
        "filesystem write."
    ),
    responses={
        400: {"description": "Unsafe/invalid strategy id"},
        409: {"description": "Strategy id already exists"},
        413: {"description": "Payload too large"},
        422: {"description": "Invalid strategy config"},
    },
)
async def create_strategy(request: Request, studio: StudioDep) -> StrategyConfigView:
    """Create a new custom strategy YAML."""
    payload = await _read_strategy_payload(request)
    try:
        return studio.create_strategy(payload)
    except StrategyStudioError as exc:
        raise _studio_http_error(exc) from exc


@router.delete(
    "/strategies/{strategy_id}",
    response_model=ArtifactDeleteResponse,
    summary="Delete a custom/tuned strategy YAML",
    description=(
        "Deletes a custom or optimized strategy YAML (``configs/strategies`` in "
        "development/paper/live, the ``artifacts/strategies`` overlay otherwise) and "
        "removes it from the runtime strategy catalog immediately. Committed built-in "
        "strategies are protected and return 409 (clone them to a custom id instead). "
        "Unknown ids return 404."
    ),
    responses={
        404: {"description": "Unknown strategy id"},
        409: {"description": "Committed built-in strategies cannot be deleted"},
    },
)
def delete_strategy(strategy_id: str, studio: StudioDep) -> ArtifactDeleteResponse:
    """Delete one custom/tuned strategy YAML."""
    try:
        path = studio.delete_strategy(strategy_id)
    except StrategyStudioError as exc:
        raise _studio_http_error(exc) from exc
    return ArtifactDeleteResponse(
        id=strategy_id, deleted=True, artifact_deleted=True, path=str(path)
    )


@router.post(
    "/strategies/{strategy_id}/clone",
    response_model=StrategyConfigView,
    summary="Clone a strategy to a new custom id",
    description=(
        "Copies the validated config to ``new_strategy_id`` (environment write rules "
        "apply), refreshes the runtime catalog immediately and returns the new view. "
        "Built-in strategies can be cloned; unknown source ids return 404 and existing "
        "target ids return 409."
    ),
    responses={
        400: {"description": "Unsafe/invalid new strategy id"},
        404: {"description": "Unknown source strategy id"},
        409: {"description": "Target strategy id already exists"},
    },
)
def clone_strategy(
    strategy_id: str, payload: StrategyIdentityRequest, studio: StudioDep
) -> StrategyConfigView:
    """Clone one strategy under a new custom id."""
    try:
        return studio.clone_strategy(strategy_id, payload.new_strategy_id)
    except StrategyStudioError as exc:
        raise _studio_http_error(exc) from exc


@router.post(
    "/strategies/{strategy_id}/rename",
    response_model=StrategyConfigView,
    summary="Rename a custom strategy",
    description=(
        "Persists the config under ``new_strategy_id``, deletes the old YAML and swaps "
        "the runtime catalog entry (old id disappears, new id is immediately runnable). "
        "Committed built-ins cannot be renamed (409 — clone instead); existing target "
        "ids return 409; unknown sources return 404."
    ),
    responses={
        400: {"description": "Unsafe/invalid new strategy id / unchanged id"},
        404: {"description": "Unknown source strategy id"},
        409: {"description": "Built-in source or target id already exists"},
    },
)
def rename_strategy(
    strategy_id: str, payload: StrategyIdentityRequest, studio: StudioDep
) -> StrategyConfigView:
    """Rename one custom strategy."""
    try:
        return studio.rename_strategy(strategy_id, payload.new_strategy_id)
    except StrategyStudioError as exc:
        raise _studio_http_error(exc) from exc


@router.get(
    "/strategies/{strategy_id}/yaml",
    summary="Download a strategy YAML",
    description=(
        "Returns the current strategy YAML as a ``text/yaml`` attachment named "
        "``<strategy_id>.yaml`` (the exact persisted file). Unknown ids return 404."
    ),
    response_class=Response,
    responses={
        200: {"content": {"text/yaml": {}}, "description": "Strategy YAML attachment"},
        404: {"description": "Unknown strategy id"},
    },
)
def download_strategy_yaml(strategy_id: str, studio: StudioDep) -> Response:
    """Download one strategy as a YAML attachment."""
    try:
        content = studio.strategy_yaml(strategy_id)
    except StrategyStudioError as exc:
        raise _studio_http_error(exc) from exc
    return Response(
        content=content,
        media_type="text/yaml",
        headers={
            "Content-Disposition": f'attachment; filename="{strategy_id}.yaml"',
        },
    )


@router.put(
    "/strategies/{strategy_id}",
    response_model=StrategyConfigView,
    summary="Validate, merge and persist a strategy YAML",
    description=(
        "Merges a partial (or full) edit into the existing strategy, re-validates the "
        "whole pipeline with the frozen ``configs.strategy`` models and persists the "
        "result atomically. Built-in strategies are edited in place only in "
        "development/paper/live; backtest/test environments keep the committed YAML "
        "immutable and persist an overlay under ``artifacts/strategies``."
    ),
    responses={
        400: {"description": "Unsafe/invalid strategy id"},
        404: {"description": "Unknown strategy id"},
        422: {"description": "Edited config failed validation"},
    },
)
async def update_strategy(
    strategy_id: str, request: Request, studio: StudioDep
) -> StrategyConfigView:
    """Apply, validate and persist a strategy edit."""
    payload = await _read_strategy_payload(request)
    try:
        return studio.update_strategy(strategy_id, payload)
    except StrategyStudioError as exc:
        raise _studio_http_error(exc) from exc
