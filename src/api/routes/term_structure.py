"""Futures term-structure and roll-calendar endpoint."""

from __future__ import annotations

from collections.abc import Mapping

from fastapi import APIRouter, HTTPException, Response

from src.api import root_scope
from src.api.deps import ContainerDep, MarketDataDep
from src.api.schemas import TermStructureResponse

router = APIRouter(tags=["term-structure"])


@router.get(
    "/term-structure",
    response_model=TermStructureResponse,
    summary="Forward curves and roll calendar",
    description=(
        "Contango/backwardation curve per futures root (F1…Fn with expiry, price "
        "and days to expiry), annualised roll yield and the upcoming contract "
        "rolls derived from the instrument master's roll rules."
    ),
)
def term_structure(
    market: MarketDataDep,
    container: ContainerDep,
    response: Response,
    scope: str = "strategy",
    sector: str | None = None,
    roots: str | None = None,
) -> TermStructureResponse:
    """Return the forward curves and roll calendar with additive data provenance.

    ``TermStructureResponse`` has no source field (frozen schema), so the backing
    evidence is exposed additively via response headers: ``X-Data-Origin``
    (``cached_parquet`` / ``mixed_parquet_fixture`` / ``fixture``), ``X-Data-As-Of``
    and ``X-Data-Cached-Roots``.
    """
    if scope.strip().lower() != "strategy" or sector or roots:
        try:
            selected, canonical_scope, canonical_sector = root_scope.resolve_scoped_roots(
                settings=container.settings,
                master=container.instrument_master,
                market=market,
                scope=scope,
                sector=sector,
                roots=roots,
            )
            scoped_market, data_origin = root_scope.build_scoped_market(
                settings=container.settings,
                master=container.instrument_master,
                market=market,
                selected=selected,
            )
        except root_scope.RootScopeError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
        scoped = root_scope.scoped_term_structure(scoped_market).model_copy(
            update={
                "scope": canonical_scope,
                "sector": canonical_sector,
                "selected_roots": selected,
                "data_origin": data_origin,
            }
        )
        response.headers["X-Data-Origin"] = data_origin
        response.headers["X-Data-Roots"] = str(len(selected))
        response.headers["X-Scope-Roots"] = ",".join(selected)
        return scoped
    result = market.term_structure()
    origin = getattr(market, "data_origin", None)
    if isinstance(origin, str):
        response.headers["X-Data-Origin"] = origin
    backing = getattr(market, "data_backing", None)
    info: object = backing() if callable(backing) else backing
    if isinstance(info, Mapping):
        as_of = info.get("as_of")
        if isinstance(as_of, str):
            response.headers["X-Data-As-Of"] = as_of
        cached_roots = info.get("cached_roots")
        if isinstance(cached_roots, list | tuple):
            response.headers["X-Data-Cached-Roots"] = str(len(cached_roots))
        roots = info.get("roots")
        if isinstance(roots, list | tuple):
            response.headers["X-Data-Roots"] = str(len(roots))
    return result
