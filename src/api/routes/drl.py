"""DRL model registry and offline training endpoints.

Mounted by the app factory under ``/api`` (router registered by the Lead in
``src/api/routes/__init__.py``), so the public paths are:

* ``GET  /api/drl/models`` - registry of ``artifacts/models/*.zip`` checkpoints.
* ``POST /api/drl/train``  - validate + run one deterministic offline training
  run (training executes in a worker thread so the event loop stays responsive)
  and persist ``<name>.zip`` + ``<name>.zip.meta.json`` + reward curve.

Both handlers depend on :func:`src.api.drl_service.get_drl_service`, which resolves
the standalone :class:`~src.api.drl_service.DRLService` from ``app.state`` /
``Container.settings`` -- no edits to the shared service container are required.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from starlette.concurrency import run_in_threadpool

from src.api.drl_service import (
    DRLModelListResponse,
    DrlServiceDep,
    DRLTrainRequest,
    DRLTrainResponse,
)

router = APIRouter(prefix="/drl", tags=["drl"])


@router.get(
    "/models",
    response_model=DRLModelListResponse,
    summary="List trained DRL checkpoints",
    description=(
        "Registry view over the models directory (default ``artifacts/models``): one "
        "entry per ``*.zip`` checkpoint with its parsed ``*.zip.meta.json`` "
        "(algorithm, timesteps, universe roots, feature names, created timestamp, git "
        "commit, reward-curve summary). Entries with a missing/corrupt sidecar are "
        "still listed with ``metadata_available=false`` and a ``metadata_error``."
    ),
)
def list_drl_models(service: DrlServiceDep) -> DRLModelListResponse:
    """Return every discovered checkpoint, newest first."""
    return service.list_models()


@router.post(
    "/train",
    response_model=DRLTrainResponse,
    summary="Train a DRL checkpoint (deterministic offline)",
    description=(
        "Validates the algorithm (PPO/CPPO/SAC/TD3/A2C), the bounded timesteps and the "
        "checkpoint name, then trains the strategy's ``drl:`` section from the "
        "deterministic offline data source (no Databento/IBKR/network), persists the "
        "checkpoint + metadata + reward curve and returns the training summary. "
        "Training runs in a worker thread; keep ``timesteps`` <= 20000 for a synchronous "
        "API smoke run."
    ),
    responses={
        404: {"description": "Unknown strategy id"},
        422: {"description": "Invalid request or strategy without a drl: section"},
        500: {"description": "Training infrastructure failure (e.g. missing DRL stack)"},
    },
)
async def train_drl_checkpoint(
    payload: DRLTrainRequest, service: DrlServiceDep
) -> DRLTrainResponse:
    """Run one training request in the thread pool (never blocks the event loop)."""
    try:
        return await run_in_threadpool(service.train, payload)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"DRL training failed: {exc}",
        ) from exc
