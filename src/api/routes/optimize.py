"""Optuna walk-forward optimization endpoints for the Strategy Studio."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from starlette.concurrency import run_in_threadpool

from src.api.optimize_service import (
    OptimizationApplyRequest,
    OptimizationApplyResponse,
    OptimizationError,
    OptimizationStudy,
    OptimizationStudySummary,
    OptimizeRequest,
)
from src.api.phase10_deps import OptimizeDep
from src.api.qol_schemas import ArtifactDeleteResponse

router = APIRouter(tags=["optimize"])


@router.post(
    "/optimize",
    response_model=OptimizationStudy,
    summary="Run a walk-forward optimization study",
    description=(
        "Runs the synchronous WalkForwardOptimizer (purged/embargoed out-of-sample "
        "folds; every trial scored only on OOS metrics) over the base strategy, "
        "persists ``artifacts/optimizations/<study_id>.json`` and returns the full "
        "study with trials, parameter importances and the YAML-ready best config. "
        "Falls back to a deterministic offline fixture study (``provenance=fixture``) "
        "when the optimizer package is not deployed."
    ),
    responses={
        404: {"description": "Unknown base strategy"},
        422: {"description": "Invalid optimization request"},
        500: {"description": "Optimization store failure"},
    },
)
async def run_optimization(payload: OptimizeRequest, optimize: OptimizeDep) -> OptimizationStudy:
    """Run an optimization study in the worker thread (synchronous engine)."""
    try:
        return await run_in_threadpool(optimize.run, payload)
    except OptimizationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.get(
    "/optimize",
    response_model=list[OptimizationStudySummary],
    summary="List persisted optimization studies",
    description="Study history (newest first) reloaded from the optimization artifact store.",
)
def list_optimizations(optimize: OptimizeDep) -> list[OptimizationStudySummary]:
    """Return persisted study summaries."""
    return optimize.list_studies()


@router.get(
    "/optimize/{study_id}",
    response_model=OptimizationStudy,
    summary="Get one optimization study",
    responses={404: {"description": "Unknown study id"}},
)
def get_optimization(study_id: str, optimize: OptimizeDep) -> OptimizationStudy:
    """Return one persisted study including its trials."""
    try:
        return optimize.get_study(study_id)
    except OptimizationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.post(
    "/optimize/{study_id}/apply",
    response_model=OptimizationApplyResponse,
    summary="Apply a study's best parameters as a new strategy",
    description=(
        "Writes the tuned parameters over the base strategy into a new visible strategy "
        "YAML (``<base>_opt`` unless ``new_strategy_id`` is given; development/paper/live "
        "write ``configs/strategies/<id>.yaml``, test/backtest write the artifacts "
        "overlay), refreshes the in-memory catalog immediately, and always triggers + "
        "persists a walk-forward verification backtest through the existing backtest "
        "service (window defaults to the base strategy backtest block; override with "
        "``start``/``end``). The response additively returns ``saved_strategy_id``, "
        "``strategy_path`` and ``backtest_run_id``. Re-applying to an existing id returns "
        "409; pass a unique ``new_strategy_id``."
    ),
    responses={
        404: {"description": "Unknown study/base strategy"},
        409: {"description": "Target strategy id already exists / no best parameters"},
        422: {"description": "Tuned parameters are not a valid strategy config"},
    },
)
async def apply_optimization(
    study_id: str,
    payload: OptimizationApplyRequest,
    optimize: OptimizeDep,
) -> OptimizationApplyResponse:
    """Apply best parameters and persist the tuned strategy YAML."""
    try:
        return await run_in_threadpool(optimize.apply, study_id, payload)
    except OptimizationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.delete(
    "/optimize/{study_id}",
    response_model=ArtifactDeleteResponse,
    summary="Delete an optimization study and its artifact",
    description=(
        "Removes the study from the in-memory history and deletes the canonical "
        "``optimizations/<study_id>.json`` artifact (legacy quarantine files are left "
        "untouched). The study immediately disappears from list/detail/apply. Unknown "
        "ids return 404; the response receipt reports ``artifact_deleted``."
    ),
    responses={404: {"description": "Unknown study id"}},
)
def delete_optimization(study_id: str, optimize: OptimizeDep) -> ArtifactDeleteResponse:
    """Delete one persisted optimization study."""
    try:
        artifact_deleted = optimize.delete_study(study_id)
    except OptimizationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    return ArtifactDeleteResponse(
        id=study_id,
        deleted=True,
        artifact_deleted=artifact_deleted,
        path=str(optimize.output_dir / f"{study_id}.json") if artifact_deleted else None,
    )
