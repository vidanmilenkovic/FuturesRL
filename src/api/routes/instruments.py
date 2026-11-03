"""Sector-grouped futures instrument universe endpoint (Strategy Studio UX).

The Strategy Studio needs the *expanded* tradable universe (roots, sector grouping
and standard/micro pair links) to render root pills and micro-routing hints. This
endpoint derives everything dynamically from ``settings.instruments`` — the
instrument master is the single source of truth, so newly added roots/sectors
appear automatically with no code change or hardcoded root list.

Contract (all client-side ordering/labelling is derived from these fields):

``GET /api/instruments`` -> ``{generated_at, count, sectors, roots}``

* ``sectors`` are returned in the fixed Studio order
  ``equities, rates, energy, metals, fx, agri, crypto`` (empty groups included so
  the UI can render all seven lanes before every sector is populated).
* ``roots`` inside a sector and the flat ``roots`` list are alphabetical.
* ``micro_root`` / ``standard_root`` mirror the instrument master's pair links, so
  the UI can show both directions (standard -> micro and micro -> standard).
* Unknown sectors are preserved in a trailing ``other`` group rather than dropped;
  ``count`` always equals ``sum(sector.count) == len(roots)``.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

from configs.instruments import (
    InstrumentsConfig,
    InstrumentSpec,
    clear_instruments_cache,
    load_instruments,
)
from configs.settings import Settings
from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from src.api.deps import ContainerDep
from src.core.types import AssetClass

router = APIRouter(tags=["instruments"])


class _InstrumentsModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class InstrumentSummary(_InstrumentsModel):
    """One futures root in the Studio universe."""

    root: str
    name: str
    asset_class: AssetClass
    sector: str
    exchange: str
    multiplier: float = Field(gt=0.0)
    tick_size: float = Field(gt=0.0)
    currency: str
    micro_root: str | None = None
    standard_root: str | None = None


class SectorGroup(_InstrumentsModel):
    """One sector lane of the Studio universe."""

    id: str
    label: str
    count: int = Field(ge=0)
    roots: list[InstrumentSummary] = Field(default_factory=list)


class InstrumentsResponse(_InstrumentsModel):
    """``GET /api/instruments`` payload."""

    generated_at: datetime
    count: int = Field(ge=0)
    sectors: list[SectorGroup] = Field(default_factory=list)
    roots: list[InstrumentSummary] = Field(default_factory=list)


#: Studio sector order: (id, UI label, config `sector`/asset-class aliases).
_SECTOR_DEFINITIONS: tuple[tuple[str, str, frozenset[str]], ...] = (
    ("equities", "Equities", frozenset({"equity", "equities", "equity_index", "index"})),
    ("rates", "Rates", frozenset({"rates", "rate", "interest_rates", "fixed_income"})),
    ("energy", "Energy", frozenset({"energy", "energies", "crude"})),
    ("metals", "Metals", frozenset({"metals", "metal", "precious_metals"})),
    ("fx", "FX", frozenset({"fx", "forex", "currency", "currencies"})),
    ("agri", "Agriculture", frozenset({"agri", "agriculture", "ags", "grains"})),
    ("crypto", "Crypto", frozenset({"crypto", "cryptocurrency", "digital_assets"})),
)

#: Asset-class fallback when the instrument master's ``sector`` string is unknown.
_ASSET_CLASS_SECTORS: Mapping[AssetClass, str] = {
    AssetClass.EQUITY_INDEX: "equities",
    AssetClass.RATES: "rates",
    AssetClass.ENERGY: "energy",
    AssetClass.METALS: "metals",
    AssetClass.FX: "fx",
    AssetClass.AGRICULTURE: "agri",
    AssetClass.CRYPTO: "crypto",
}

_SECTOR_IDS: frozenset[str] = frozenset(entry[0] for entry in _SECTOR_DEFINITIONS)
_SECTOR_LABELS: Mapping[str, str] = {entry[0]: entry[1] for entry in _SECTOR_DEFINITIONS}
_SECTOR_ALIASES: Mapping[str, str] = {
    alias: sector_id for sector_id, _label, aliases in _SECTOR_DEFINITIONS for alias in aliases
}

#: Path -> (mtime, master) cache so a concurrently updated YAML is picked up live.
_MASTER_CACHE: dict[str, tuple[float, InstrumentsConfig]] = {}
_MASTER_LOCK = threading.Lock()


def resolve_sector(spec: InstrumentSpec) -> str:
    """Canonical Studio sector id for one instrument spec (``other`` when unknown)."""
    raw = str(getattr(spec, "sector", "") or "").strip().lower()
    resolved = _SECTOR_ALIASES.get(raw)
    if resolved is not None:
        return resolved
    return _ASSET_CLASS_SECTORS.get(spec.asset_class, "other")


def _master_path(settings: Settings) -> Path:
    return Path(settings.instruments_path or (settings.configs_dir / "instruments.yaml"))


def load_current_master(settings: Settings) -> InstrumentsConfig:
    """Instrument master, refreshed when the YAML changed on disk.

    ``configs.instruments.load_instruments`` is process-cached; the Studio endpoint
    re-checks the file mtime and clears the cache when data-eng adds roots, so the
    endpoint reflects the expanded universe without an API restart.
    """
    path = _master_path(settings)
    try:
        mtime = path.stat().st_mtime
    except OSError:  # pragma: no cover - committed master always exists
        return settings.instruments
    key = str(path.resolve())
    with _MASTER_LOCK:
        cached = _MASTER_CACHE.get(key)
        if cached is not None and cached[0] == mtime:
            return cached[1]
        clear_instruments_cache()
        master = load_instruments(path)
        _MASTER_CACHE[key] = (mtime, master)
        return master


def _summary(spec: InstrumentSpec, sector: str) -> InstrumentSummary:
    return InstrumentSummary(
        root=spec.root,
        name=spec.name,
        asset_class=spec.asset_class,
        sector=sector,
        exchange=spec.exchange,
        multiplier=spec.multiplier,
        tick_size=spec.tick_size,
        currency=spec.currency,
        micro_root=spec.micro_root,
        standard_root=spec.standard_root,
    )


def build_instruments_response(
    master: InstrumentsConfig, *, generated_at: datetime
) -> InstrumentsResponse:
    """Group an instrument master into the fixed Studio sector order."""
    groups: dict[str, list[InstrumentSummary]] = {sector_id: [] for sector_id in _SECTOR_IDS}
    for spec in master.instruments.values():
        sector_id = resolve_sector(spec)
        groups.setdefault(sector_id, []).append(_summary(spec, sector_id))
    sectors: list[SectorGroup] = []
    ordered_ids: list[str] = [entry[0] for entry in _SECTOR_DEFINITIONS]
    for extra in sorted(sector_id for sector_id in groups if sector_id not in _SECTOR_IDS):
        ordered_ids.append(extra)
    for sector_id in ordered_ids:
        roots = sorted(groups.get(sector_id, []), key=lambda row: row.root)
        sectors.append(
            SectorGroup(
                id=sector_id,
                label=_SECTOR_LABELS.get(sector_id, sector_id.replace("_", " ").title()),
                count=len(roots),
                roots=roots,
            )
        )
    roots = sorted((row for group in sectors for row in group.roots), key=lambda row: row.root)
    return InstrumentsResponse(
        generated_at=generated_at,
        count=len(roots),
        sectors=sectors,
        roots=roots,
    )


@router.get(
    "/instruments",
    response_model=InstrumentsResponse,
    summary="Sector-grouped futures instrument universe",
    description=(
        "The expanded tradable futures universe derived live from the instrument "
        "master: seven Studio sector lanes in fixed order (equities, rates, energy, "
        "metals, fx, agri, crypto), alphabetically sorted roots with standard/micro "
        "pair links, and a flat alphabetical roots list. ``count`` always equals "
        "``len(roots) == sum(sectors[].count)``."
    ),
)
def list_instruments(container: ContainerDep) -> InstrumentsResponse:
    """Return the sector-grouped instrument universe for the Studio."""
    master = load_current_master(container.settings)
    return build_instruments_response(master, generated_at=container.clock())
