"""FinRL-X pipeline waterfall endpoint (Pipeline Inspector view)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.api import root_scope
from src.api.deps import BridgePipelineDep, ContainerDep
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
def latest_pipeline(
    container: ContainerDep,
    bridge: BridgePipelineDep,
    scope: str = "strategy",
    sector: str | None = None,
    roots: str | None = None,
) -> PipelineWaterfallResponse:
    """Return the most recent pipeline trace (bridge-first).

    Precedence: a fresh runner-bridge snapshot **with per-instrument rows** (full
    Weight matrix, identical fields to the engine waterfall) → the compact bridge
    scalar summary (Phase-9 snapshots without rows) → the engine waterfall.
    """
    if scope.strip().lower() != "strategy" or sector or roots:
        try:
            selected, canonical_scope, canonical_sector = root_scope.resolve_scoped_roots(
                settings=container.settings,
                master=container.instrument_master,
                market=container.market,
                scope=scope,
                sector=sector,
                roots=roots,
            )
            scoped_market, data_origin = root_scope.build_scoped_market(
                settings=container.settings,
                master=container.instrument_master,
                market=container.market,
                selected=selected,
            )
            state = root_scope.scoped_state(scoped_market)
            scoped = root_scope.scoped_pipeline(container.market, state).model_copy(
                update={
                    "scope": canonical_scope,
                    "sector": canonical_sector,
                    "selected_roots": selected,
                    "data_origin": data_origin,
                }
            )
        except root_scope.RootScopeError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
        return scoped
    matrix = bridge.pipeline()
    if matrix is not None:
        return matrix
    compact = container.live.bridge_pipeline()
    if compact is not None:
        return compact
    return container.market.pipeline()
