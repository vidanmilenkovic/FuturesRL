"""FinRL-X pipeline waterfall endpoint (Pipeline Inspector view)."""

from __future__ import annotations

from fastapi import APIRouter

from src.api.deps import ContainerDep
from src.api.schemas import PipelineWaterfallResponse

router = APIRouter(tags=["pipeline"])


@router.get(
    "/pipeline/latest",
    response_model=PipelineWaterfallResponse,
    summary="Latest w_t = R_t(T_t(A_t(S_t(X)))) trace",
    description=(
        "Per-stage gross/net exposures (S_t, w_base, w_timing, w_risk, N_i,t), a "
        "per-instrument delta table (including the achieved ``weight_actual``), "
        "discretization tracking error + micro-routing, stage diagnostics and the "
        "target contract quantities produced by the deterministic contract "
        "translator. Prefers a fresh live-runner bridge pipeline snapshot "
        "(``data_source=runner-bridge``)."
    ),
)
def latest_pipeline(container: ContainerDep) -> PipelineWaterfallResponse:
    """Return the most recent pipeline trace (bridge-first)."""
    bridged = container.live.bridge_pipeline()
    if bridged is not None:
        return bridged
    return container.market.pipeline()
