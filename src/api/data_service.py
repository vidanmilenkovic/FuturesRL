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
        ibkr_fetcher_provider: Callable[[], object] | None = None,
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

    def _fetcher(self) -> object:
        if self._ibkr_provider is not None:
            return self._ibkr_provider()
        from src.data.ibkr_data import IBKRHistoricalBarFetcher

        return IBKRHistoricalBarFetcher(self._settings, instrument_master=self._master)

    # -- inventory -------------------------------------------------------------

    def inventory(self, *, schema: str = s.DEFAULT_BAR_SCHEMA) -> DataInventoryResponse:
        rows: list[RootCacheSummary] = []
        helper = getattr(self._repository, "inventory", None)
        if callable(helper):
            try:
                entries = helper(schema)
            except TypeError:
                entries = helper()
            except Exception as exc:  # pragma: no cover - repository helper is tolerant
                logger.warning("repository.inventory failed (%s); filesystem fallback", exc)
                entries = None
            if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
                entries = None
            if entries is not None:
                for entry in entries:
                    rows.append(self._summary_from_entry(entry, schema))
        if not rows:
            rows = self._filesystem_inventory(schema)
        rows.sort(key=lambda row: row.root)
        totals = self._totals(rows)
        return DataInventoryResponse(
            roots=rows,
            totals=totals,
            databento_configured=self._databento_configured(),
            ibkr_available=self._ibkr_is_available(),
            schema=schema,
            generated_at=self._clock(),
        )

    def _summary_from_entry(self, entry: object, schema: str) -> RootCacheSummary:
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
        )

    def _filesystem_inventory(self, schema: str) -> list[RootCacheSummary]:
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
                    RootCacheSummary(root=root, sector=self._sector_for(root), schema=schema)
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
                    RootCacheSummary(root=root, sector=self._sector_for(root), schema=schema)
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
        if request.source == "ibkr":
            return CostEstimateResponse(
                roots=roots,
                schema=request.schema,
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
            estimate = estimator(roots, start, end, request.schema)
        except Exception as exc:
            status_code = (
                400 if "databento" in str(exc).lower() or "key" in str(exc).lower() else 422
            )
            raise DataServiceError(
                f"Databento cost estimate failed: {type(exc).__name__}: {exc}",
                status_code=status_code,
            ) from exc
        return self._estimate_response(estimate, roots, request)

    def _estimate_response(
        self,
        estimate: object,
        roots: Sequence[str],
        request: EstimateCostRequest,
    ) -> CostEstimateResponse:
        estimate_roots = cast(Sequence[object], _field(estimate, "roots", ()) or ())
        billable = int(cast(int, _field(estimate, "billable_size_bytes", 0) or 0))
        megabytes = float(
            cast(float, _field(estimate, "billable_size_megabytes", billable / 1_000_000) or 0.0)
        )
        return CostEstimateResponse(
            roots=[str(root) for root in estimate_roots] or list(roots),
            schema=str(_field(estimate, "schema", request.schema) or request.schema),
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
        started = time.perf_counter()
        warnings: list[str] = []
        cost: CostEstimateResponse | None = None
        if request.source == "ibkr":
            frame = self._ingest_ibkr(roots, request, warnings)
        else:
            frame, cost = self._ingest_databento(roots, request, warnings)
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
                "IBKR paper/TWS session is not available; start IB Gateway/TWS "
                "(paper port) or use source='databento'",
                status_code=409,
            )
        fetcher = self._fetcher()
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
                f"IBKR historical ingest failed: {exc}", status_code=409
            ) from exc
        if not isinstance(frame, pl.DataFrame):
            raise DataServiceError("IBKR fetcher returned an invalid frame", status_code=502)
        if frame.is_empty():
            warnings.append("IBKR returned no bars for the requested window")
            return frame
        written = self._repository.write_bars(frame, schema=request.schema)
        del written
        self._record_sources(roots, "ibkr", request.schema, warnings)
        return frame

    def _ingest_databento(
        self, roots: Sequence[str], request: IngestRequest, warnings: list[str]
    ) -> tuple[pl.DataFrame, CostEstimateResponse]:
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
            estimate = estimator(roots, start, end, request.schema)
        except Exception as exc:
            status_code = (
                400 if "databento" in str(exc).lower() or "key" in str(exc).lower() else 422
            )
            raise DataServiceError(
                f"Databento cost estimate failed: {type(exc).__name__}: {exc}",
                status_code=status_code,
            ) from exc
        estimate_response = self._estimate_response(
            estimate,
            roots,
            EstimateCostRequest.model_validate(
                {
                    "roots": list(roots),
                    "start": request.start,
                    "end": request.end,
                    "schema": request.schema,
                    "source": "databento",
                }
            ),
        )
        cap = float(self._settings.databento_max_cost_usd_per_job)
        over_budget = estimate_response.estimated_cost_usd > cap
        if over_budget and not request.allow_over_budget:
            raise DataServiceError(
                f"Databento budget exceeded: estimate ${estimate_response.estimated_cost_usd:.2f} "
                f"> cap ${cap:.2f}; confirm the operator override (allow_over_budget+confirm) "
                "or raise FUTURESRLX_DATABENTO_MAX_COST_USD_PER_JOB",
                status_code=409,
                detail={
                    "estimate": estimate_response.model_dump(mode="json"),
                    "max_cost_usd": cap,
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
                request.schema,
                allow_over_budget=bool(
                    over_budget and request.allow_over_budget and request.confirm
                ),
            )
        except Exception as exc:
            name = type(exc).__name__
            if "Budget" in name:
                raise DataServiceError(
                    f"Databento budget guard refused the download: {exc}", status_code=409
                ) from exc
            raise DataServiceError(
                f"Databento ingest failed: {name}: {exc}", status_code=502
            ) from exc
        if over_budget:
            warnings.append(
                f"operator budget override used: estimate ${estimate_response.estimated_cost_usd:.2f} "
                f"> cap ${cap:.2f}"
            )
        if isinstance(frame, pl.DataFrame) and not frame.is_empty():
            self._record_sources(roots, "databento", request.schema, warnings)
        return (
            frame if isinstance(frame, pl.DataFrame) else pl.DataFrame(),
            estimate_response,
        )

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
