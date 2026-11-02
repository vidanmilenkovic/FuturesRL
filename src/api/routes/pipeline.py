"""FinRL-X pipeline waterfall endpoint (Pipeline Inspector view)."""

from __future__ import annotations

from fastapi import APIRouter

from src.api.deps import MarketDataDep
from src.api.schemas import PipelineWaterfallResponse

router = APIRouter(tags=["pipeline"])


@router.get(
    "/pipeline/latest",
    response_model=PipelineWaterfallResponse,
    summary="Latest w_t = R_t(T_t(A_t(S_t(X)))) trace",
    description=(
        "Per-stage gross/net exposures (S_t, w_base, w_timing, w_risk, N_i,t), a "
        "per-instrument delta table, stage diagnostics and the target contract "
        "quantities produced by the deterministic contract translator."
    ),
)
def latest_pipeline(market: MarketDataDep) -> PipelineWaterfallResponse:
    """Return the most recent pipeline trace."""
    return market.pipeline()
