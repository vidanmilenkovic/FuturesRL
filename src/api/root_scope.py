"""Workstation UX root-scope helpers for term-structure/pipeline inspection.

``scope=strategy`` keeps the existing default response. Scoped requests
(``scope=cached|all`` or explicit ``sector``/``roots`` filters) resolve a root set,
build a point-in-time state (cached Parquet preferred, deterministic instrument-master
pricing for uncached roots) and recompute the pipeline/term structure for exactly that
set. Provenance is always honest: never a fabricated live attribution.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Literal, cast

import polars as pl
from configs.instruments import InstrumentsConfig
from configs.settings import Settings

from src.api import fixtures
from src.api.protocols import MarketDataService
from src.api.schemas import PipelineWaterfallResponse, TermStructureResponse
from src.core.types import MarketState

logger = logging.getLogger(__name__)

__all__ = [
    "RootScopeError",
    "build_scoped_market",
    "resolve_scoped_roots",
    "scoped_pipeline",
    "scoped_term_structure",
]

type Scope = Literal["strategy", "cached", "all"]

#: Canonical sector groups accepted by the workstation UI (+ plural aliases).
_SECTOR_ALIASES: dict[str, str] = {
    "equity": "equities",
    "equities": "equities",
    "equity_index": "equities",
    "index": "equities",
    "rate": "rates",
    "rates": "rates",
    "interest_rates": "rates",
    "energy": "energy",
    "energies": "energy",
    "metal": "metals",
    "metals": "metals",
    "fx": "fx",
    "forex": "fx",
    "currency": "fx",
    "agri": "agri",
    "agriculture": "agri",
    "grains": "agri",
    "crypto": "crypto",
    "cryptocurrency": "crypto",
}
VALID_SECTORS: tuple[str, ...] = (
    "equities",
    "rates",
    "energy",
    "metals",
    "fx",
    "agri",
    "crypto",
)

#: Minimum bars per root for cached data to be considered usable in a scoped view.
_MIN_CACHED_BARS = 60


class RootScopeError(RuntimeError):
    """API-mappable invalid scope/sector/roots selection (422)."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.status_code = 422


def _sector_group(master: InstrumentsConfig, root: str) -> str:
    spec = master.get(root)
    raw = str(getattr(spec, "sector", "") or "").strip().lower()
    group = _SECTOR_ALIASES.get(raw)
    if group is not None:
        return group
    asset_class = str(getattr(spec, "asset_class", "") or "").strip().lower()
    for alias, canonical in _SECTOR_ALIASES.items():
        if alias == asset_class:
            return canonical
    return "unknown"


def _normalise_scope(scope: str) -> Scope:
    normalized = (scope or "strategy").strip().lower()
    if normalized not in {"strategy", "cached", "all"}:
        raise RootScopeError(
            f"invalid scope {scope!r}; expected one of ['strategy', 'cached', 'all']"
        )
    return cast(Scope, normalized)


def resolve_scoped_roots(
    *,
    settings: Settings,
    master: InstrumentsConfig,
    market: MarketDataService,
    scope: str,
    sector: str | None,
    roots: str | None,
) -> tuple[list[str], Scope, str | None]:
    """Resolve the selected root list, canonical scope and sector (422 on invalid)."""
    normalized_scope = _normalise_scope(scope)
    canonical_sector: str | None = None
    if sector is not None and sector.strip():
        canonical_sector = _SECTOR_ALIASES.get(sector.strip().lower())
        if canonical_sector is None:
            raise RootScopeError(
                f"invalid sector {sector!r}; expected one of {list(VALID_SECTORS)}"
            )
    explicit: list[str] = []
    if roots is not None and roots.strip():
        explicit = list(
            dict.fromkeys(part.strip().upper() for part in roots.split(",") if part.strip())
        )
        unknown = [root for root in explicit if not master.has(root)]
        if unknown:
            raise RootScopeError(f"unknown futures root(s): {unknown!r}")
    if explicit:
        selected = list(explicit)
        if canonical_sector is not None:
            selected = [
                root for root in selected if _sector_group(master, root) == canonical_sector
            ]
        if not selected:
            raise RootScopeError(f"no requested roots belong to sector {canonical_sector!r}")
        return selected, normalized_scope, canonical_sector
    if normalized_scope == "all":
        selected = list(master.roots())
    elif normalized_scope == "cached":
        cached: list[str] = []
        backing = getattr(market, "data_backing", None)
        info = backing() if callable(backing) else backing
        if isinstance(info, dict):
            cached = [str(root) for root in info.get("cached_roots", [])]
        if not cached:
            try:
                from src.data.storage import ParquetRepository

                entries = ParquetRepository(settings).inventory("ohlcv-1d")
                cached = [row.root for row in entries if row.cached_bars > 0]
            except Exception as exc:  # pragma: no cover - cache introspection best effort
                logger.warning("cached-root inventory unavailable: %s", exc)
        selected = [root for root in cached if master.has(root)]
        if not selected:
            raise RootScopeError(
                "no cached Parquet roots are available; ingest data first or use "
                "scope='all' for deterministic instrument-master inspection"
            )
    else:  # strategy + no explicit roots: sector filter across the master
        selected = list(master.roots())
    if canonical_sector is not None:
        selected = [root for root in selected if _sector_group(master, root) == canonical_sector]
        if not selected:
            raise RootScopeError(
                f"no instrument-master roots belong to sector {canonical_sector!r}"
            )
    return sorted(dict.fromkeys(selected)), normalized_scope, canonical_sector


def _cached_frame(
    settings: Settings, roots: list[str], start: datetime, end: datetime
) -> tuple[pl.DataFrame | None, list[str]]:
    """Cached daily bars plus the roots with enough coverage (empty frame on miss)."""
    from src.data.storage import ParquetRepository

    try:
        frame = ParquetRepository(settings).read_bars(roots, start, end, schema="ohlcv-1d")
    except Exception as exc:  # pragma: no cover - corrupt cache is a data error
        logger.warning("scoped cached read failed: %s", exc)
        return None, []
    if frame is None or getattr(frame, "is_empty", lambda: True)():
        return frame, []
    cached_roots: list[str] = []
    for root in roots:
        try:
            rows = frame.filter(pl.col("root_symbol") == root).height
        except Exception:  # pragma: no cover - defensive
            rows = 0
        if rows >= _MIN_CACHED_BARS:
            cached_roots.append(root)
    return frame, cached_roots


def build_scoped_market(
    *,
    settings: Settings,
    master: InstrumentsConfig,
    market: MarketDataService,
    selected: list[str],
) -> tuple[fixtures.OfflineMarket, str]:
    """Build a point-in-time offline market for ``selected`` roots + provenance origin."""
    if not selected:
        raise RootScopeError("cannot build a scoped market with no selected roots")

    as_of: datetime
    equity = fixtures.DEFAULT_INITIAL_EQUITY
    try:
        as_of = market.latest_state().timestamp
        equity = market.latest_state().equity or equity
    except Exception:  # pragma: no cover - market service fallback
        as_of = datetime.now(UTC)
    window_start = (as_of - timedelta(days=520)).date()
    end_date = as_of.date()
    deterministic = fixtures.build_window_market(
        master, window_start, end_date, seed=7, roots=selected, warmup_bars=260
    )
    frame = deterministic.bars
    origin = "deterministic_instrument_master"
    try:
        cached, cached_roots = _cached_frame(settings, selected, as_of - timedelta(days=520), as_of)
        if cached_roots and cached is not None:
            if len(cached_roots) == len(selected):
                frame = cached
                origin = "cached_parquet"
            else:
                deterministic_roots = [root for root in selected if root not in cached_roots]
                missing_market = fixtures.build_window_market(
                    master,
                    window_start,
                    end_date,
                    seed=7,
                    roots=deterministic_roots,
                    warmup_bars=260,
                )
                frame = pl.concat(
                    [cached, missing_market.bars],
                    how="diagonal_relaxed",
                ).sort(["timestamp_utc", "root_symbol"])
                origin = "mixed_cached_and_deterministic"
    except Exception as exc:  # pragma: no cover - cache preference is best effort
        logger.warning("scoped cached preference unavailable: %s", exc)
    timestamps = tuple(
        sorted(
            {
                value
                for value in frame.get_column("timestamp_utc").to_list()
                if isinstance(value, datetime)
            }
        )
    )
    latest = (
        frame.sort("timestamp_utc")
        .group_by("root_symbol")
        .agg(pl.col("close").last().alias("close"))
    )
    prices = {
        str(record["root_symbol"]): float(record["close"])
        for record in latest.iter_rows(named=True)
    }
    return (
        fixtures.OfflineMarket(
            instrument_master=master,
            bars=frame,
            timestamps=timestamps,
            prices=prices,
            seed=7,
        ),
        origin,
    )


def scoped_state(market: fixtures.OfflineMarket, equity: float | None = None) -> MarketState:
    """Build the point-in-time MarketState for a scoped offline market."""
    from src.data.market_state_builder import MarketStateBuilder

    builder = MarketStateBuilder(
        instrument_master=market.instrument_master,
        initial_equity=equity or fixtures.DEFAULT_INITIAL_EQUITY,
    )
    return builder.build(market.bars, market.as_of, equity=equity)


def scoped_pipeline(
    market_service: MarketDataService, state: MarketState
) -> PipelineWaterfallResponse:
    """Run the real pipeline for a scoped state via the market service helper."""
    builder = getattr(market_service, "pipeline_for_state", None)
    if not callable(builder):
        raise RootScopeError(
            "the configured market service does not support scoped pipeline inspection"
        )
    return cast(PipelineWaterfallResponse, builder(state))


def scoped_term_structure(market: fixtures.OfflineMarket) -> TermStructureResponse:
    """Deterministic term structure for a scoped offline market."""
    return fixtures.term_structure(market)
