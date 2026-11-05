"""Phase-11B data-cache API contracts (inventory / estimate / ingest)."""

from __future__ import annotations

import warnings
from datetime import date, datetime
from typing import Literal

from pydantic import Field

from src.api.schemas.core import APIModel

warnings.filterwarnings(
    "ignore",
    message='Field name "schema".*shadows an attribute',
    category=UserWarning,
)

__all__ = [
    "CacheTotals",
    "ContinuousSeriesSummary",
    "CostEstimateResponse",
    "DataInventoryResponse",
    "DataMode",
    "DataOrigin",
    "EstimateCostRequest",
    "IngestRequest",
    "IngestResponse",
    "RootCacheSummary",
]

type DataMode = Literal["auto", "cached_real", "synthetic_offline"]
type DataOrigin = Literal["real_parquet", "synthetic_fixture"]
type CacheSource = Literal["databento", "ibkr", "empty"]


class RootCacheSummary(APIModel):
    """Cached Parquet coverage for one futures root."""

    root: str
    sector: str = ""
    schema: str = Field(default="ohlcv-1d")  # type: ignore[assignment]
    cached_bars: int = Field(default=0, ge=0)
    start_date: date | None = None
    end_date: date | None = None
    size_bytes: int = Field(default=0, ge=0)
    source: CacheSource = "empty"
    timeframe: str = "1d"


class CacheTotals(APIModel):
    """Aggregate cache coverage across the instrument master."""

    cached_roots: int = Field(default=0, ge=0)
    total_roots: int = Field(default=0, ge=0)
    total_bars: int = Field(default=0, ge=0)
    total_size_bytes: int = Field(default=0, ge=0)
    start_date: date | None = None
    end_date: date | None = None


class DataInventoryResponse(APIModel):
    """``GET /api/data/inventory`` payload."""

    roots: list[RootCacheSummary] = Field(default_factory=list)
    totals: CacheTotals = Field(default_factory=CacheTotals)
    databento_configured: bool = False
    ibkr_available: bool = False
    schema: str = Field(default="ohlcv-1d")  # type: ignore[assignment]
    timeframe: str = "1d"
    base_schema: str = "ohlcv-1d"
    generated_at: datetime


class EstimateCostRequest(APIModel):
    """``POST /api/data/estimate-cost`` body."""

    roots: list[str] = Field(default_factory=list)
    start: date
    end: date
    schema: str = Field(default="ohlcv-1d")  # type: ignore[assignment]
    timeframe: str = "1d"
    source: Literal["databento", "ibkr"] = "databento"


class CostEstimateResponse(APIModel):
    """Cost/billable-size estimate for one ingest window."""

    roots: list[str] = Field(default_factory=list)
    schema: str = Field(default="ohlcv-1d")  # type: ignore[assignment]
    timeframe: str = "1d"
    base_schema: str = "ohlcv-1d"
    start: date
    end: date
    estimated_cost_usd: float = 0.0
    billable_size_bytes: int = 0
    billable_size_mb: float = 0.0
    source: Literal["databento", "ibkr"] = "databento"
    currency: str = "USD"
    details: dict[str, object] | None = None


class IngestRequest(APIModel):
    """``POST /api/data/ingest`` body."""

    source: Literal["ibkr", "databento"]
    roots: list[str] = Field(default_factory=list)
    start: date
    end: date
    schema: str = Field(default="ohlcv-1d")  # type: ignore[assignment]
    timeframe: str = "1d"
    allow_over_budget: bool = False
    confirm: bool = True


class ContinuousSeriesSummary(APIModel):
    """One continuous F1/F2 series built after an ingest."""

    root: str
    bars: int = Field(default=0, ge=0)
    start_date: date | None = None
    end_date: date | None = None


class IngestResponse(APIModel):
    """``POST /api/data/ingest`` payload."""

    source: Literal["ibkr", "databento"]
    roots_ingested: list[str] = Field(default_factory=list)
    bars_written: int = Field(default=0, ge=0)
    start: date
    end: date
    cost_estimate: CostEstimateResponse | None = None
    timeframe: str = "1d"
    base_schema: str = "ohlcv-1d"
    resampled: bool = False
    continuous_series: list[ContinuousSeriesSummary] = Field(default_factory=list)
    duration_seconds: float = 0.0
    warnings: list[str] = Field(default_factory=list)
