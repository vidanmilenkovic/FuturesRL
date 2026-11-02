"""Futures term-structure and roll-calendar endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from src.api.deps import MarketDataDep
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
def term_structure(market: MarketDataDep) -> TermStructureResponse:
    """Return the deterministic forward curves and roll calendar."""
    return market.term_structure()
