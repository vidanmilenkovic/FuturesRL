"""Phase-11B data-cache API service: inventory, cost estimate and ingestion.

The service is a thin, honest adapter over the data-eng cache primitives:

* ``ParquetRepository.inventory`` / ``record_source`` (task-63) for dashboard
  coverage + provenance, with a tolerant filesystem fallback so the API can boot
  before the helper lands;
* ``DatabentoLoader.estimate_cost_usd`` + the hard budget guard (never calls
  ``timeseries.get_range`` when the estimate exceeds the cap);
* ``IBKRHistoricalBarFetcher.fetch_bars_sync`` for the free paper-TWS ingest path.

Tests inject fake repository/fetcher/loader providers; no network is touched.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, date, datetime
from datetime import time as dt_time
from pathlib import Path
from typing import cast

import polars as pl
from configs.instruments import InstrumentsConfig
from configs.settings import Settings

from src.api import fixtures
from src.api.phase11b_schemas import (
    CacheSource,
    CacheTotals,
    ContinuousSeriesSummary,
    CostEstimateResponse,
    DataInventoryResponse,
    EstimateCostRequest,
    IngestRequest,
    IngestResponse,
    RootCacheSummary,
)
from src.data import schema as s
from src.data.storage import ParquetRepository

logger = logging.getLogger(__name__)

__all__ = ["DataService", "DataServiceError"]

_DEFAULT_SCHEMA = "ohlcv-1d"
_SOURCE_SIDECAR_NAME = "part.parquet.meta.json"


#: Fallback timeframe registry (data-eng task-67 helpers are imported lazily when present).
_TIMEFRAME_FALLBACK: dict[str, tuple[str, str, str]] = {
    # timeframe -> (target schema, databento base schema, ibkr bar size)
    "1m": ("ohlcv-1m", "ohlcv-1m", "1 min"),
    "5m": ("ohlcv-5m", "ohlcv-1m", "5 mins"),
    "15m": ("ohlcv-15m", "ohlcv-1m", "15 mins"),
    "30m": ("ohlcv-30m", "ohlcv-1m", "30 mins"),
    "1h": ("ohlcv-1h", "ohlcv-1m", "1 hour"),
    "4h": ("ohlcv-4h", "ohlcv-1h", "4 hours"),
    "1d": ("ohlcv-1d", "ohlcv-1d", "1 day"),
    "1w": ("ohlcv-1w", "ohlcv-1d", "1 week"),
}


def _tf_entry(timeframe: str) -> tuple[str, str, str]:
    normalized = (timeframe or "1d").strip().lower()
    entry = _TIMEFRAME_FALLBACK.get(normalized)
    if entry is None:
        raise DataServiceError(
            f"unsupported timeframe {timeframe!r}; supported: {sorted(_TIMEFRAME_FALLBACK)}",
            status_code=422,
            detail={
                "error": "unsupported_timeframe",
                "message": f"unsupported timeframe {timeframe!r}",
                "supported_timeframes": sorted(_TIMEFRAME_FALLBACK),
            },
        )
    try:
        import importlib

        tf_module = importlib.import_module("src.data.timeframes")
        spec = tf_module.get_timeframe(normalized)
        schema = getattr(spec, "schema", None) or tf_module.schema_for_timeframe(normalized)
        base = tf_module.databento_base_schema(normalized)
        bar_size = tf_module.ibkr_bar_size(normalized)
        return (str(schema), str(base), str(bar_size))
    except Exception:
        return entry


def _target_schema(timeframe: str) -> str:
    return _tf_entry(timeframe)[0]


def _canonical_timeframe(timeframe: str) -> str:
    normalized = (timeframe or "1d").strip().lower()
    try:
        import importlib

        tf_module = importlib.import_module("src.data.timeframes")
        return str(tf_module.get_timeframe(normalized).name)
    except Exception:
        return normalized


def _base_schema(timeframe: str) -> str:
    return _tf_entry(timeframe)[1]


_SCHEMA_TIMEFRAMES: dict[str, str] = {
    "ohlcv-1m": "1m",
    "ohlcv-5m": "5m",
    "ohlcv-15m": "15m",
    "ohlcv-30m": "30m",
    "ohlcv-1h": "1h",
    "ohlcv-4h": "4h",
    "ohlcv-1d": "1d",
    "ohlcv-1w": "1w",
}


def _timeframe_for_schema(schema: str) -> str:
    return _SCHEMA_TIMEFRAMES.get(schema, "1d")


def _ibkr_base_timeframe(timeframe: str) -> str | None:
    """Finer timeframe IBKR must fetch before resampling to ``timeframe``."""
    normalized = (timeframe or "1d").strip().lower()
    try:
        import importlib

        tf_module = importlib.import_module("src.data.timeframes")
        base = tf_module.base_timeframe(normalized)
        return str(base) if base else None
    except Exception:
        return {
            "5m": "1m",
            "15m": "1m",
            "30m": "1m",
            "1h": "1m",
            "4h": "1h",
            "1w": "1d",
        }.get(normalized)


def _ibkr_bar_size(timeframe: str) -> str:
    normalized = (timeframe or "1d").strip().lower()
    base_tf = _ibkr_base_timeframe(normalized) or normalized
    try:
        import importlib

        tf_module = importlib.import_module("src.data.timeframes")
        return str(tf_module.ibkr_bar_size(base_tf))
    except Exception:
        return _TIMEFRAME_FALLBACK.get(base_tf, _TIMEFRAME_FALLBACK["1d"])[2]


def _ibkr_unreachable_message(settings: Settings) -> str:
    return (
        f"IBKR TWS/Gateway unreachable on {settings.ibkr_host}:{settings.ibkr_port}; "
        "start TWS or IB Gateway (paper port) and retry"
    )


def _ibkr_unreachable_detail(settings: Settings) -> dict[str, object]:
    return {
        "error": "ibkr_unreachable",
        "message": _ibkr_unreachable_message(settings),
        "host": settings.ibkr_host,
        "port": settings.ibkr_port,
    }


def resample_frame(
    frame: pl.DataFrame, target_timeframe: str, *, base_timeframe: str | None = None
) -> pl.DataFrame:
    """Resample canonical bars to ``target_timeframe`` (look-ahead-safe).

    Uses the data-eng ``src.data.resampler`` helper when deployed; otherwise a
    deterministic Polars ``group_by_dynamic`` fallback with right-closed/right-labeled
    UTC windows. Returns the input unchanged when the target is already the source.
    """
    normalized = (target_timeframe or "1d").strip().lower()
    source = (base_timeframe or normalized).strip().lower()
    if normalized == source:
        return frame
    try:
        import importlib

        resample_module = importlib.import_module("src.data.resampler")
        resampled = resample_module.resample_bars(frame, normalized)
        if isinstance(resampled, pl.DataFrame):
            return resampled
    except Exception:
        pass
    every = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "4h": "4h",
        "1d": "1d",
        "1w": "1w",
    }.get(normalized)
    if every is None:
        return frame
    from src.data import schema as bar_schema

    time_col = bar_schema.TIMESTAMP_COLUMN
    try:
        grouped = (
            frame.sort(time_col)
            .group_by_dynamic(
                time_col,
                every=every,
                period=every,
                label="right",
                closed="right",
                group_by=["instrument_id", "root_symbol", "expiration_date"],
            )
            .agg(
                pl.col("open").first().alias("open"),
                pl.col("high").max().alias("high"),
                pl.col("low").min().alias("low"),
                pl.col("close").last().alias("close"),
                pl.col("volume").sum().alias("volume"),
                pl.col("open_interest").last().alias("open_interest"),
                pl.col("settlement").last().alias("settlement"),
            )
        )
    except Exception:
        return frame
    return grouped.sort([time_col, "root_symbol"])


class DataServiceError(RuntimeError):
    """API-mappable data-cache error (``status_code`` + optional detail payload)."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 422,
        detail: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.detail = dict(detail) if detail is not None else None


def _as_aware(value: date, *, end_of_day: bool = False) -> datetime:
    moment = dt_time(23, 59, 59) if end_of_day else dt_time.min
    return datetime.combine(value, moment, tzinfo=UTC)


def _as_date(value: object) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


def _field(obj: object, name: str, default: object = None) -> object:
    if isinstance(obj, Mapping):
        return cast(Mapping[str, object], obj).get(name, default)
    return getattr(obj, name, default)


def _series_date(column: pl.Series, *, latest: bool) -> date | None:
    value = column.max() if latest else column.min()
    return value.date() if isinstance(value, datetime) else None


class DataService:
    """Inventory/estimate/ingest over the Phase-11B Parquet cache."""

    def __init__(
        self,
        *,
        settings: Settings,
        repository: ParquetRepository | None = None,
        instrument_master: InstrumentsConfig | None = None,
        clock: fixtures.Clock = fixtures.utc_now,
        databento_loader_provider: Callable[[], object] | None = None,
        ibkr_fetcher_provider: Callable[..., object] | None = None,
        ibkr_available_provider: Callable[[], bool] | None = None,
    ) -> None:
        self._settings = settings
        self._repository = repository if repository is not None else ParquetRepository(settings)
        self._master = instrument_master if instrument_master is not None else settings.instruments
        self._clock = clock
        self._databento_provider = databento_loader_provider
        self._ibkr_provider = ibkr_fetcher_provider
        self._ibkr_available = ibkr_available_provider

    # -- helpers ---------------------------------------------------------------

    @property
    def repository(self) -> ParquetRepository:
        return self._repository

    @property
    def instrument_master(self) -> InstrumentsConfig:
        return self._master

    def _sector_for(self, root: str) -> str:
        try:
            spec = self._master.get(root)
        except KeyError:
            return "unknown"
        sector = str(getattr(spec, "sector", "") or "").strip()
        return sector or "unknown"

    def _databento_configured(self) -> bool:
        getter = getattr(self._settings, "databento_api_key_value", None)
        value = getter if not callable(getter) else getter()
        return value is not None and bool(str(value))

    def _ibkr_is_available(self) -> bool:
        if self._ibkr_available is not None:
            try:
                return bool(self._ibkr_available())
            except Exception:
                return False
        return self._settings.environment == "paper"

    def _normalise_roots(self, roots: Sequence[str]) -> list[str]:
        requested = [root.strip().upper() for root in roots if root.strip()]
        if not requested:
            raise DataServiceError("roots must contain at least one futures root", status_code=422)
        missing = sorted({root for root in requested if not self._master.has(root)})
        if missing:
            raise DataServiceError(f"unknown futures root(s): {missing!r}", status_code=422)
        return list(dict.fromkeys(requested))

    def _validate_window(self, start: date, end: date) -> None:
        if end <= start:
            raise DataServiceError("end must be after start", status_code=422)

    def _loader(self) -> object:
        if self._databento_provider is not None:
            return self._databento_provider()
        from src.data.databento_loader import DatabentoLoader

        return DatabentoLoader(
            self._settings,
            repository=self._repository,
            instrument_master=self._master,
        )

    def _fetcher(self, timeframe: str = "1d") -> object:
        provider = self._ibkr_provider
        if provider is not None:
            try:
                return provider(timeframe)
            except TypeError:
                return provider()
        from src.data.ibkr_data import IBKRHistoricalBarFetcher

        return IBKRHistoricalBarFetcher(
            self._settings,
            instrument_master=self._master,
            bar_size=_ibkr_bar_size(timeframe),
        )

    @staticmethod
    def _databento_error(exc: BaseException) -> DataServiceError:
        text = f"{type(exc).__name__}: {exc}"
        lowered = text.lower()
        if "databento_api_key" in lowered or "api key" in lowered or "not configured" in lowered:
            message = (
                "Databento API key is required: set FUTURESRLX_DATABENTO_API_KEY "
                f"(loader error: {text})"
            )
            return DataServiceError(
                message,
                status_code=400,
                detail={"error": "databento_api_key_missing", "message": message},
            )
        return DataServiceError(
            f"Databento cost estimate failed: {text}",
            status_code=422,
            detail={"error": "databento_estimate_failed", "message": text},
        )

    # -- inventory -------------------------------------------------------------

    def inventory(
        self, *, timeframe: str = "1d", schema: str | None = None
    ) -> DataInventoryResponse:
        target_schema = schema or _target_schema(timeframe)
        rows: list[RootCacheSummary] = []
        helper = getattr(self._repository, "inventory", None)
        if callable(helper):
            try:
                entries = helper(target_schema)
            except TypeError:
                entries = helper()
            except Exception as exc:  # pragma: no cover - repository helper is tolerant
                logger.warning("repository.inventory failed (%s); filesystem fallback", exc)
                entries = None
            if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
                entries = None
            if entries is not None:
                for entry in entries:
                    rows.append(self._summary_from_entry(entry, target_schema, timeframe))
        if not rows:
            rows = self._filesystem_inventory(target_schema, timeframe)
        rows.sort(key=lambda row: row.root)
        totals = self._totals(rows)
        return DataInventoryResponse(
            roots=rows,
            totals=totals,
            databento_configured=self._databento_configured(),
            ibkr_available=self._ibkr_is_available(),
            schema=target_schema,
            timeframe=(timeframe or "1d").strip().lower(),
            base_schema=_base_schema(timeframe),
            generated_at=self._clock(),
        )

    def _summary_from_entry(
        self, entry: object, schema: str, timeframe: str = "1d"
    ) -> RootCacheSummary:
        root = str(_field(entry, "root", "") or "").strip().upper()
        return RootCacheSummary(
            root=root,
            sector=str(_field(entry, "sector", "") or self._sector_for(root)),
            schema=str(_field(entry, "schema", schema) or schema),
            cached_bars=int(cast(int, _field(entry, "cached_bars", 0) or 0)),
            start_date=_as_date(_field(entry, "start_date")),
            end_date=_as_date(_field(entry, "end_date")),
            size_bytes=int(cast(int, _field(entry, "size_bytes", 0) or 0)),
            source=cast(CacheSource, str(_field(entry, "source", "empty") or "empty")),
            timeframe=(timeframe or "1d").strip().lower(),
        )

    def _filesystem_inventory(self, schema: str, timeframe: str = "1d") -> list[RootCacheSummary]:
        roots = list(self._master.roots())
        schema_dir = self._repository.schema_dir(schema)
        if schema_dir.is_dir():
            known = set(roots)
            for child in schema_dir.iterdir():
                if child.is_dir() and child.name.upper() not in known:
                    roots.append(child.name.upper())
        rows: list[RootCacheSummary] = []
        for root in roots:
            path = self._repository.partition_path(root, schema)
            if not path.is_file():
                rows.append(
                    RootCacheSummary(
                        root=root,
                        sector=self._sector_for(root),
                        schema=schema,
                        timeframe=timeframe,
                    )
                )
                continue
            try:
                stats = pl.read_parquet(path, columns=[s.TIMESTAMP_COLUMN])
                timestamps = stats.get_column(s.TIMESTAMP_COLUMN)
                bars = timestamps.len()
                start_date = _series_date(timestamps, latest=False)
                end_date = _series_date(timestamps, latest=True)
            except Exception as exc:  # pragma: no cover - corrupt partition must not crash
                logger.warning("corrupt cache partition %s: %s", path, exc)
                rows.append(
                    RootCacheSummary(
                        root=root,
                        sector=self._sector_for(root),
                        schema=schema,
                        timeframe=timeframe,
                    )
                )
                continue
            sidecar = self._sidecar_path(path)
            rows.append(
                RootCacheSummary(
                    root=root,
                    sector=self._sector_for(root),
                    schema=schema,
                    cached_bars=bars,
                    start_date=start_date,
                    end_date=end_date,
                    size_bytes=path.stat().st_size,
                    source=self._sidecar_source(sidecar),
                    timeframe=timeframe,
                )
            )
        return rows

    @staticmethod
    def _sidecar_path(partition: Path) -> Path:
        return partition.parent / _SOURCE_SIDECAR_NAME

    @staticmethod
    def _sidecar_source(path: Path) -> CacheSource:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return "databento"
        source = payload.get("source") if isinstance(payload, Mapping) else None
        return cast(CacheSource, source if source in {"databento", "ibkr"} else "databento")

    @staticmethod
    def _totals(rows: Sequence[RootCacheSummary]) -> CacheTotals:
        cached = [row for row in rows if row.cached_bars > 0]
        starts = [row.start_date for row in cached if row.start_date is not None]
        ends = [row.end_date for row in cached if row.end_date is not None]
        return CacheTotals(
            cached_roots=len(cached),
            total_roots=len(rows),
            total_bars=sum(row.cached_bars for row in rows),
            total_size_bytes=sum(row.size_bytes for row in rows),
            start_date=min(starts) if starts else None,
            end_date=max(ends) if ends else None,
        )

    # -- estimate --------------------------------------------------------------

    def estimate_cost(self, request: EstimateCostRequest) -> CostEstimateResponse:
        roots = self._normalise_roots(request.roots)
        self._validate_window(request.start, request.end)
        start = _as_aware(request.start)
        end = _as_aware(request.end, end_of_day=True)
        # Validate/canonicalise the timeframe before any source-specific shortcut.
        _tf_entry(request.timeframe)
        timeframe = _canonical_timeframe(request.timeframe)
        target_schema = _target_schema(request.timeframe)
        base_schema = _base_schema(request.timeframe)
        if request.source == "ibkr":
            return CostEstimateResponse(
                roots=roots,
                schema=target_schema,
                timeframe=timeframe,
                base_schema=base_schema,
                start=request.start,
                end=request.end,
                estimated_cost_usd=0.0,
                billable_size_bytes=0,
                billable_size_mb=0.0,
                source="ibkr",
                currency="USD",
                details={"note": "IBKR paper/TWS historical ingest is free"},
            )
        loader = self._loader()
        estimator = getattr(loader, "estimate_cost_usd", None)
        if not callable(estimator):
            raise DataServiceError(
                "Databento cost estimation is unavailable in this deployment "
                "(deploy the task-63 data layer)",
                status_code=400,
            )
        try:
            estimate = estimator(roots, start, end, base_schema)
        except Exception as exc:
            raise self._databento_error(exc) from exc
        return self._estimate_response(
            estimate,
            roots,
            request,
            target_schema=target_schema,
            base_schema=base_schema,
        )

    def _estimate_response(
        self,
        estimate: object,
        roots: Sequence[str],
        request: EstimateCostRequest,
        *,
        target_schema: str | None = None,
        base_schema: str | None = None,
    ) -> CostEstimateResponse:
        estimate_roots = cast(Sequence[object], _field(estimate, "roots", ()) or ())
        billable = int(cast(int, _field(estimate, "billable_size_bytes", 0) or 0))
        megabytes = float(
            cast(float, _field(estimate, "billable_size_megabytes", billable / 1_000_000) or 0.0)
        )
        resolved_target = target_schema or request.schema
        resolved_base = base_schema or str(
            _field(estimate, "schema", request.schema) or request.schema
        )
        return CostEstimateResponse(
            roots=[str(root) for root in estimate_roots] or list(roots),
            schema=resolved_target,
            timeframe=(request.timeframe or "1d").strip().lower(),
            base_schema=resolved_base,
            start=_as_date(_field(estimate, "start")) or request.start,
            end=_as_date(_field(estimate, "end")) or request.end,
            estimated_cost_usd=float(
                cast(float, _field(estimate, "estimated_cost_usd", 0.0) or 0.0)
            ),
            billable_size_bytes=billable,
            billable_size_mb=round(max(0.0, megabytes), 6),
            source="databento",
            currency="USD",
            details=cast("dict[str, object] | None", _field(estimate, "details")),
        )

    # -- ingest ----------------------------------------------------------------

    def ingest(self, request: IngestRequest) -> IngestResponse:
        roots = self._normalise_roots(request.roots)
        self._validate_window(request.start, request.end)
        _tf_entry(request.timeframe)
        target_timeframe = _canonical_timeframe(request.timeframe)
        target_schema = _target_schema(target_timeframe)
        base_schema = _base_schema(target_timeframe)
        started = time.perf_counter()
        warnings: list[str] = []
        cost: CostEstimateResponse | None = None
        resampled = False
        if request.source == "ibkr":
            frame = self._ingest_ibkr(roots, request, warnings)
            ibkr_base_tf = _ibkr_base_timeframe(target_timeframe)
            if (
                ibkr_base_tf is not None
                and ibkr_base_tf != target_timeframe
                and not frame.is_empty()
            ):
                frame = resample_frame(frame, target_timeframe, base_timeframe=ibkr_base_tf)
                resampled = True
            if not frame.is_empty():
                try:
                    self._repository.write_bars(frame, schema=target_schema)
                except Exception as exc:  # pragma: no cover - cache write failure
                    warnings.append(f"target-timeframe cache write failed: {exc}")
                self._record_sources(roots, "ibkr", target_schema, warnings)
        else:
            frame, cost, resampled = self._ingest_databento(
                roots, request, warnings, target_schema=target_schema, base_schema=base_schema
            )
        duration = round(time.perf_counter() - started, 3)
        written = frame.height if frame is not None else 0
        series = self._continuous_series(frame, roots, warnings)
        return IngestResponse(
            source=request.source,
            roots_ingested=roots,
            bars_written=written,
            start=request.start,
            end=request.end,
            cost_estimate=cost,
            timeframe=target_timeframe,
            base_schema=base_schema,
            resampled=resampled,
            continuous_series=series,
            duration_seconds=duration,
            warnings=warnings,
        )

    def _ingest_ibkr(
        self, roots: Sequence[str], request: IngestRequest, warnings: list[str]
    ) -> pl.DataFrame:
        if self._settings.environment == "live":
            raise DataServiceError(
                "refusing IBKR ingest in the live environment; use a paper TWS session",
                status_code=409,
            )
        if not self._ibkr_is_available():
            raise DataServiceError(
                _ibkr_unreachable_message(self._settings),
                status_code=503,
                detail=_ibkr_unreachable_detail(self._settings),
            )
        fetcher = self._fetcher(request.timeframe)
        fetch = getattr(fetcher, "fetch_bars_sync", None)
        if not callable(fetch):
            raise DataServiceError(
                "IBKR historical fetcher is unavailable in this deployment", status_code=409
            )
        start = _as_aware(request.start)
        end = _as_aware(request.end, end_of_day=True)
        try:
            frame = fetch(roots, start, end)
        except RuntimeError as exc:
            raise DataServiceError(
                _ibkr_unreachable_message(self._settings),
                status_code=503,
                detail={
                    **_ibkr_unreachable_detail(self._settings),
                    "reason": f"{type(exc).__name__}: {exc}",
                },
            ) from exc
        if not isinstance(frame, pl.DataFrame):
            raise DataServiceError("IBKR fetcher returned an invalid frame", status_code=502)
        if frame.is_empty():
            warnings.append("IBKR returned no bars for the requested window")
        return frame

    def _ingest_databento(
        self,
        roots: Sequence[str],
        request: IngestRequest,
        warnings: list[str],
        *,
        target_schema: str | None = None,
        base_schema: str | None = None,
    ) -> tuple[pl.DataFrame, CostEstimateResponse, bool]:
        resolved_target = target_schema or _target_schema(request.timeframe)
        resolved_base = base_schema or _base_schema(request.timeframe)
        loader = self._loader()
        estimator = getattr(loader, "estimate_cost_usd", None)
        ingest = getattr(loader, "ingest", None)
        if not callable(estimator) or not callable(ingest):
            raise DataServiceError(
                "Databento ingest is unavailable in this deployment (deploy task-63 helpers)",
                status_code=400,
            )
        start = _as_aware(request.start)
        end = _as_aware(request.end, end_of_day=True)
        try:
            estimate = estimator(roots, start, end, resolved_base)
        except Exception as exc:
            raise self._databento_error(exc) from exc
        estimate_response = self._estimate_response(
            estimate,
            roots,
            EstimateCostRequest.model_validate(
                {
                    "roots": list(roots),
                    "start": request.start,
                    "end": request.end,
                    "schema": resolved_target,
                    "timeframe": request.timeframe,
                    "source": "databento",
                }
            ),
            target_schema=resolved_target,
            base_schema=resolved_base,
        )
        cap = float(self._settings.databento_max_cost_usd_per_job)
        over_budget = estimate_response.estimated_cost_usd > cap
        if over_budget and not request.allow_over_budget:
            estimated = float(estimate_response.estimated_cost_usd)
            message = (
                f"Databento cost guard exceeded (${estimated:.2f} > ${cap:.2f} cap); "
                "set allow_over_budget=true and confirm=true to override, or raise "
                "FUTURESRLX_DATABENTO_MAX_COST_USD_PER_JOB"
            )
            raise DataServiceError(
                message,
                status_code=402,
                detail={
                    "error": "databento_budget_exceeded",
                    "message": message,
                    "estimated_cost_usd": estimated,
                    "max_cost_usd": cap,
                    "cap": cap,
                    "override_allowed": True,
                },
            )
        if over_budget and not request.confirm:
            raise DataServiceError(
                "operator confirmation required to exceed the Databento per-job budget",
                status_code=400,
                detail={"estimate": estimate_response.model_dump(mode="json"), "max_cost_usd": cap},
            )
        try:
            frame = ingest(
                roots,
                start,
                end,
                resolved_base,
                allow_over_budget=bool(
                    over_budget and request.allow_over_budget and request.confirm
                ),
            )
        except Exception as exc:
            name = type(exc).__name__
            if "Budget" in name:
                raise DataServiceError(
                    f"Databento cost guard exceeded: {exc}",
                    status_code=402,
                    detail={
                        "error": "databento_budget_exceeded",
                        "message": str(exc),
                        "override_allowed": True,
                    },
                ) from exc
            raise DataServiceError(
                f"Databento ingest failed: {name}: {exc}", status_code=502
            ) from exc
        if over_budget:
            warnings.append(
                f"operator budget override used: estimate ${estimate_response.estimated_cost_usd:.2f} "
                f"> cap ${cap:.2f}"
            )
        result_frame = frame if isinstance(frame, pl.DataFrame) else pl.DataFrame()
        resampled = resolved_target != resolved_base and not result_frame.is_empty()
        if resampled:
            result_frame = resample_frame(
                result_frame,
                (request.timeframe or "1d").strip().lower(),
                base_timeframe=_timeframe_for_schema(resolved_base),
            )
            self._persist_target(result_frame, request, resolved_target, warnings)
        if isinstance(frame, pl.DataFrame) and not frame.is_empty():
            self._record_sources(roots, "databento", resolved_target, warnings)
        return (result_frame, estimate_response, resampled)

    def _persist_target(
        self,
        resampled: pl.DataFrame,
        request: IngestRequest,
        target_schema: str,
        warnings: list[str],
    ) -> None:
        """Persist the resampled target timeframe (helper-first, tolerant fallback)."""
        writer = getattr(self._repository, "write_timeframe", None)
        if callable(writer):
            try:
                writer(request.timeframe, resampled)
                return
            except TypeError:
                try:
                    writer(resampled, request.timeframe)
                    return
                except Exception as exc:  # pragma: no cover - helper API variant
                    warnings.append(f"target-timeframe cache write failed: {exc}")
            except Exception as exc:  # pragma: no cover - best effort
                warnings.append(f"target-timeframe cache write failed: {exc}")
        try:
            self._repository.write_bars(resampled, schema=target_schema)
        except Exception as exc:
            warnings.append(f"target-timeframe cache write failed: {exc}")

    def _record_sources(
        self, roots: Sequence[str], source: str, schema: str, warnings: list[str]
    ) -> None:
        recorder = getattr(self._repository, "record_source", None)
        if callable(recorder):
            for root in roots:
                try:
                    recorder(root, source, schema)
                except Exception as exc:  # pragma: no cover - provenance best effort
                    warnings.append(f"could not record source for {root}: {exc}")
            return
        # Fallback sidecar (same on-disk contract task-63 will write).
        for root in roots:
            partition = self._repository.partition_path(root, schema)
            try:
                partition.parent.mkdir(parents=True, exist_ok=True)
                sidecar = self._sidecar_path(partition)
                sidecar.write_text(
                    json.dumps(
                        {
                            "root": root,
                            "schema": schema,
                            "source": source,
                            "updated_at": self._clock().isoformat(),
                        },
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
            except OSError as exc:  # pragma: no cover - best effort
                warnings.append(f"could not record source for {root}: {exc}")

    def _continuous_series(
        self, frame: pl.DataFrame | None, roots: Sequence[str], warnings: list[str]
    ) -> list[ContinuousSeriesSummary]:
        if frame is None or frame.is_empty():
            return []
        try:
            from src.data.continuous_roller import ContinuousContractBuilder

            builder = ContinuousContractBuilder(instrument_master=self._master)
            continuous = builder.build_frame(frame)
        except Exception as exc:
            warnings.append(f"continuous F1/F2 series unavailable: {type(exc).__name__}: {exc}")
            continuous = None
        source_frame = continuous if continuous is not None and not continuous.is_empty() else frame
        root_column = s.ROOT_SYMBOL_COLUMN if s.ROOT_SYMBOL_COLUMN in source_frame.columns else None
        if root_column is None:
            return [
                ContinuousSeriesSummary(
                    root=root,
                    bars=0,
                    start_date=None,
                    end_date=None,
                )
                for root in roots
            ]
        rows: list[ContinuousSeriesSummary] = []
        for root in roots:
            subset = source_frame.filter(pl.col(root_column) == root)
            timestamps = (
                subset.get_column(s.TIMESTAMP_COLUMN)
                if s.TIMESTAMP_COLUMN in subset.columns
                else None
            )
            rows.append(
                ContinuousSeriesSummary(
                    root=root,
                    bars=subset.height,
                    start_date=(
                        _series_date(timestamps, latest=False)
                        if timestamps is not None and subset.height
                        else None
                    ),
                    end_date=(
                        _series_date(timestamps, latest=True)
                        if timestamps is not None and subset.height
                        else None
                    ),
                )
            )
        return rows
