"""Partitioned Parquet cache with a DuckDB analytics facade.

The repository is the persistence boundary of the data layer. Every adapter
(Databento historical, IBKR live, offline synthetic) writes through
:class:`ParquetRepository`, and every consumer (roller, feature processor,
market-state builder, backtester) reads from it.

Storage layout
--------------
::

    {parquet_dir}/{dataset}/{schema}/{root}/part.parquet

* ``dataset`` is the Databento dataset id (``GLBX.MDP3`` by default).
* ``schema`` is the canonical schema name (``ohlcv-1d``, ``ohlcv-1m``, ``tbbo``,
  ``definition``, ...).
* ``root`` is the futures root symbol (``ES``, ``CL``, ...).

Guarantees
----------
* **Canonical schema enforcement** — every write goes through
  :func:`src.data.schema.validator_for_schema`; a frame with a missing column,
  a non-UTC timestamp or a bid > ask quote is rejected before it can reach disk.
* **Atomic writes** — a write builds the complete file inside a temporary
  directory and then ``os.replace``-s it onto the target. Readers therefore
  either observe the previous complete file or the new complete file, never a
  half-written Parquet.
* **Idempotent dedupe** — rows are deduplicated on the canonical primary key
  (``timestamp_utc, instrument_id, root_symbol, expiration_date``) with the
  incoming row winning (``keep="last"``), so re-ingesting an overlapping
  window is safe.
* **Read path** — :meth:`read_bars` / :meth:`read_frame` concatenate the
  per-root partitions, filter the half-open UTC window ``[start, end)`` and
  re-validate the result.
* **SQL path** — :meth:`query` / :meth:`query_bars` register read-only DuckDB
  views over the Parquet globs (``bars``, ``definitions``, ``ohlcv_1m``, ...)
  and return Polars frames.

Concurrency
-----------
Writes are serialised per target file with re-entrant locks and are atomic on
disk, so multiple threads (or multiple repository instances in one process) can
write disjoint roots concurrently and readers never need a lock. This is the
"thread/process-safe enough for tests and a single ingestion process" contract;
it is not a distributed lock — run one ingestion worker per dataset in
production.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import shutil
import tempfile
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal, cast

import duckdb
import polars as pl
from configs.instruments import load_instruments
from configs.settings import Settings, get_settings

from src.core.types import InstrumentId
from src.data import schema as s
from src.data import timeframes as tf
from src.data.resampler import resample_bars

__all__ = [
    "PARTITION_FILE_NAME",
    "SOURCE_SIDECAR_NAME",
    "ParquetRepository",
    "RootCacheSummary",
    "StorageError",
    "WriteMode",
]

#: How :meth:`ParquetRepository.write_frame` merges with existing data.
type WriteMode = Literal["append", "overwrite"]

logger = logging.getLogger(__name__)

#: File name of the single Parquet partition per (dataset, schema, root).
PARTITION_FILE_NAME = "part.parquet"

#: File name of the provenance sidecar written next to each partition.
SOURCE_SIDECAR_NAME = f"{PARTITION_FILE_NAME}.meta.json"

#: Valid provenance sources for a cached partition.
type CacheSource = Literal["databento", "ibkr", "empty"]
_VALID_RECORDED_SOURCES: frozenset[str] = frozenset({"databento", "ibkr"})

#: Characters allowed in a root symbol (also protects against path traversal).
_ROOT_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9._-]*$")

#: In-process lock registry: one re-entrant lock per absolute target path.
_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


class StorageError(RuntimeError):
    """Raised when the on-disk cache cannot satisfy a storage operation."""


@dataclass(frozen=True, slots=True)
class RootCacheSummary:
    """Inventory row for one futures root in the Parquet cache.

    Returned by :meth:`ParquetRepository.inventory` for every instrument-master
    root (including roots without a partition) and for any cached non-master root.
    """

    root: InstrumentId
    sector: str
    schema: str
    cached_bars: int
    start_date: date | None
    end_date: date | None
    size_bytes: int
    source: CacheSource

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly row (dates rendered as ISO ``YYYY-MM-DD`` strings)."""
        return {
            "root": self.root,
            "sector": self.sector,
            "schema": self.schema,
            "cached_bars": self.cached_bars,
            "start_date": self.start_date.isoformat() if self.start_date is not None else None,
            "end_date": self.end_date.isoformat() if self.end_date is not None else None,
            "size_bytes": self.size_bytes,
            "source": self.source,
        }


def _atomic_write_json(payload: Mapping[str, object], target: Path) -> None:
    """Atomically write ``payload`` as JSON to ``target`` (temp file + ``os.replace``)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(dict(payload), stream, indent=2, sort_keys=True)
            stream.write("\n")
        Path(temp_name).replace(target)
    except Exception:
        with contextlib.suppress(OSError):
            Path(temp_name).unlink()
        raise


def _lock_for(path: Path) -> threading.RLock:
    """Return the process-wide write lock for ``path``."""
    key = str(path)
    with _LOCKS_GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _LOCKS[key] = lock
        return lock


def _safe_root(root: InstrumentId) -> str:
    """Validate and normalise a futures root for use as a directory name."""
    if not isinstance(root, str) or not root.strip():
        raise ValueError(f"root symbol must be a non-empty string, got {root!r}")
    normalised = root.strip().upper()
    if not _ROOT_PATTERN.match(normalised):
        raise ValueError(f"root symbol {root!r} contains unsafe characters")
    return normalised


def _atomic_write_parquet(frame: pl.DataFrame, target: Path) -> None:
    """Write ``frame`` to ``target`` atomically (temp directory + ``os.replace``)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_directory = Path(
        tempfile.mkdtemp(prefix=f".{target.stem}.", suffix=".tmp", dir=target.parent)
    )
    try:
        temporary_file = temporary_directory / target.name
        frame.write_parquet(temporary_file)
        temporary_file.replace(target)
    finally:
        shutil.rmtree(temporary_directory, ignore_errors=True)


def _assert_within(base: Path, target: Path) -> Path:
    """Resolve ``target`` and assert that it lives inside ``base``."""
    resolved_base = base.resolve()
    resolved_target = target.resolve()
    if resolved_target != resolved_base and resolved_base not in resolved_target.parents:
        raise StorageError(
            f"refusing to operate on path outside {resolved_base}: {resolved_target}"
        )
    return resolved_target


def _view_name(schema: str) -> str:
    """DuckDB-safe view name for a schema directory (``ohlcv-1m`` -> ``ohlcv_1m``)."""
    return re.sub(r"[^a-z0-9_]", "_", schema.lower())


def _key_columns_for(schema_name: str) -> tuple[str, ...]:
    """Canonical primary-key columns for a repository ``schema`` name."""
    if schema_name == s.DATABENTO_DEFINITION_SCHEMA:
        return s.DEFINITION_KEY_COLUMNS
    if schema_name == "continuous":
        return s.CONTINUOUS_KEY_COLUMNS
    if schema_name in s.DATABENTO_QUOTE_SCHEMAS:
        return s.QUOTE_KEY_COLUMNS
    if schema_name in s.DATABENTO_BOOK_SCHEMAS:
        return s.BOOK_KEY_COLUMNS
    return s.BAR_KEY_COLUMNS


class ParquetRepository:
    """Partitioned Parquet cache + DuckDB query facade.

    Parameters
    ----------
    settings:
        Runtime settings; every path defaults to this object. When ``None`` the
        process-wide :func:`configs.settings.get_settings` singleton is used.
    parquet_dir / duckdb_path / dataset:
        Explicit overrides, mainly for ``tmp_path`` tests.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        parquet_dir: Path | str | None = None,
        duckdb_path: Path | str | None = None,
        dataset: str | None = None,
    ) -> None:
        resolved = settings if settings is not None else get_settings()
        self._settings = resolved
        self._parquet_dir = Path(parquet_dir or resolved.parquet_dir)
        self._duckdb_path = Path(duckdb_path or resolved.duckdb_path)
        self._default_dataset = dataset or resolved.databento_dataset

    # -- introspection ---------------------------------------------------------

    @property
    def settings(self) -> Settings:
        """Settings object this repository was built from."""
        return self._settings

    @property
    def parquet_dir(self) -> Path:
        """Root directory of the partitioned Parquet cache."""
        return self._parquet_dir

    @property
    def duckdb_path(self) -> Path:
        """DuckDB database path reserved for persistent analytics (not required)."""
        return self._duckdb_path

    @property
    def dataset(self) -> str:
        """Default dataset id (``GLBX.MDP3`` unless configured otherwise)."""
        return self._default_dataset

    def __repr__(self) -> str:
        return (
            f"ParquetRepository(parquet_dir={self._parquet_dir!s}, "
            f"dataset={self._default_dataset!r})"
        )

    # -- path helpers ----------------------------------------------------------

    def dataset_dir(self, dataset: str | None = None) -> Path:
        """Directory holding every schema partition of ``dataset``."""
        return self._parquet_dir / (dataset or self._default_dataset)

    def schema_dir(self, schema: str, dataset: str | None = None) -> Path:
        """Directory holding every root partition of ``schema``."""
        if not schema:
            raise ValueError("schema must be a non-empty string")
        return self.dataset_dir(dataset) / schema

    def root_dir(
        self, root: InstrumentId, schema: str = s.DEFAULT_BAR_SCHEMA, dataset: str | None = None
    ) -> Path:
        """Directory holding the Parquet partition of one root."""
        return self.schema_dir(schema, dataset) / _safe_root(root)

    def partition_path(
        self, root: InstrumentId, schema: str = s.DEFAULT_BAR_SCHEMA, dataset: str | None = None
    ) -> Path:
        """Absolute path of the single Parquet partition of ``root``."""
        return self.root_dir(root, schema, dataset) / PARTITION_FILE_NAME

    def source_sidecar_path(
        self, root: InstrumentId, schema: str = s.DEFAULT_BAR_SCHEMA, dataset: str | None = None
    ) -> Path:
        """Absolute path of the provenance sidecar written next to the partition."""
        return self.root_dir(root, schema, dataset) / SOURCE_SIDECAR_NAME

    # -- lifecycle -------------------------------------------------------------

    def list_datasets(self) -> tuple[str, ...]:
        """Dataset ids present in the cache, sorted."""
        if not self._parquet_dir.is_dir():
            return ()
        return tuple(sorted(entry.name for entry in self._parquet_dir.iterdir() if entry.is_dir()))

    def list_schemas(self, dataset: str | None = None) -> tuple[str, ...]:
        """Schemas present for ``dataset``, sorted."""
        base = self.dataset_dir(dataset)
        if not base.is_dir():
            return ()
        return tuple(sorted(entry.name for entry in base.iterdir() if entry.is_dir()))

    def available_roots(
        self, schema: str = s.DEFAULT_BAR_SCHEMA, dataset: str | None = None
    ) -> tuple[InstrumentId, ...]:
        """Roots that have at least one Parquet partition for ``schema``."""
        base = self.schema_dir(schema, dataset)
        if not base.is_dir():
            return ()
        roots = [
            entry.name
            for entry in base.iterdir()
            if entry.is_dir() and (entry / PARTITION_FILE_NAME).is_file()
        ]
        return tuple(sorted(roots))

    def has_data(self, schema: str = s.DEFAULT_BAR_SCHEMA, dataset: str | None = None) -> bool:
        """Whether ``schema`` contains any data at all."""
        return bool(self.available_roots(schema, dataset))

    def inventory(
        self, schema: str = s.DEFAULT_BAR_SCHEMA, dataset: str | None = None
    ) -> list[RootCacheSummary]:
        """Per-root cache inventory across every instrument-master root.

        Returns one :class:`RootCacheSummary` per root: all instrument-master
        roots (including roots with no cached partition, reported with
        ``cached_bars=0``/``source="empty"``) plus any cached non-master roots
        (sector ``"unknown"``), sorted by root.

        Partition stats (row count, ``timestamp_utc`` min/max, file size) are read
        from the Parquet metadata; a missing, unreadable or corrupt partition is
        degraded to an empty row with a warning instead of raising. Provenance is
        read from the :func:`record_source` sidecar; legacy partitions without a
        sidecar report ``source="databento"``.
        """
        resolved_dataset = dataset or self._default_dataset
        master = load_instruments(self._settings.instruments_path)
        cached_roots = set(self.available_roots(schema, resolved_dataset))
        roots = sorted(set(master.roots()) | cached_roots)
        summaries: list[RootCacheSummary] = []
        for root in roots:
            spec = master.get(root) if master.has(root) else None
            sector = spec.sector if spec is not None else "unknown"
            path = self.partition_path(root, schema, resolved_dataset)
            cached_bars = 0
            start_date: date | None = None
            end_date: date | None = None
            size_bytes = 0
            source: CacheSource = "empty"
            if path.is_file():
                with contextlib.suppress(OSError):
                    size_bytes = path.stat().st_size
                cached_bars, start_date, end_date = self._partition_bounds(path)
                if cached_bars > 0:
                    source = self._read_partition_source(root, schema, resolved_dataset)
            summaries.append(
                RootCacheSummary(
                    root=root,
                    sector=sector,
                    schema=schema,
                    cached_bars=cached_bars,
                    start_date=start_date,
                    end_date=end_date,
                    size_bytes=size_bytes,
                    source=source,
                )
            )
        return summaries

    def record_source(
        self,
        root: InstrumentId,
        source: CacheSource,
        schema: str = s.DEFAULT_BAR_SCHEMA,
        dataset: str | None = None,
        *,
        updated_at: datetime | None = None,
    ) -> Path:
        """Record the provenance (``databento``/``ibkr``) of a cached root.

        Writes an atomic ``part.parquet.meta.json`` sidecar next to the partition
        (creating the root directory when needed) and returns its path. ``inventory``
        reads this sidecar for :attr:`RootCacheSummary.source`; partitions written
        before this mechanism (no sidecar) default to ``"databento"``.
        """
        if source not in _VALID_RECORDED_SOURCES:
            raise ValueError(
                f"source must be one of {sorted(_VALID_RECORDED_SOURCES)}, got {source!r}"
            )
        resolved_dataset = dataset or self._default_dataset
        timestamp = updated_at if updated_at is not None else datetime.now(UTC)
        if timestamp.tzinfo is None:
            raise ValueError("updated_at must be timezone-aware (UTC expected)")
        target = self.source_sidecar_path(root, schema, resolved_dataset)
        payload: dict[str, object] = {
            "root": _safe_root(root),
            "schema": schema,
            "dataset": resolved_dataset,
            "source": source,
            "updated_at": timestamp.isoformat(),
        }
        _atomic_write_json(payload, target)
        return target

    def _read_partition_source(self, root: InstrumentId, schema: str, dataset: str) -> CacheSource:
        """Read a partition's provenance sidecar (legacy default ``databento``)."""
        sidecar = self.source_sidecar_path(root, schema, dataset)
        if not sidecar.is_file():
            return "databento"
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning(
                "ignoring unreadable source sidecar for root %r (%s): %s",
                root,
                sidecar,
                exc,
            )
            return "databento"
        recorded = payload.get("source") if isinstance(payload, dict) else None
        return cast(CacheSource, recorded) if recorded in _VALID_RECORDED_SOURCES else "databento"

    @staticmethod
    def _partition_bounds(path: Path) -> tuple[int, date | None, date | None]:
        """Read ``(row_count, min_date, max_date)`` from a partition (empty on corruption)."""
        try:
            lazy = pl.scan_parquet(path)
            column_names = lazy.collect_schema().names()
            row_count = int(lazy.select(pl.len()).collect().item())
            if s.TIMESTAMP_COLUMN not in column_names:
                return (row_count, None, None)
            bounds = lazy.select(
                pl.col(s.TIMESTAMP_COLUMN).min().alias("start"),
                pl.col(s.TIMESTAMP_COLUMN).max().alias("end"),
            ).collect()
            start_value = bounds["start"][0]
            end_value = bounds["end"][0]
            start_date = start_value.date() if isinstance(start_value, datetime) else None
            end_date = end_value.date() if isinstance(end_value, datetime) else None
            return (row_count, start_date, end_date)
        except Exception as exc:
            logger.warning("could not read Parquet partition %s: %s", path, exc)
            return (0, None, None)

    def delete_root(
        self, root: InstrumentId, schema: str = s.DEFAULT_BAR_SCHEMA, dataset: str | None = None
    ) -> None:
        """Delete the partition of one root (no-op when absent)."""
        target = _assert_within(self._parquet_dir, self.root_dir(root, schema, dataset))
        shutil.rmtree(target, ignore_errors=True)

    def delete_schema(self, schema: str, dataset: str | None = None) -> None:
        """Delete every partition of ``schema`` (no-op when absent)."""
        target = _assert_within(self._parquet_dir, self.schema_dir(schema, dataset))
        shutil.rmtree(target, ignore_errors=True)

    def delete_dataset(self, dataset: str | None = None) -> None:
        """Delete every schema of ``dataset`` (no-op when absent)."""
        target = _assert_within(self._parquet_dir, self.dataset_dir(dataset))
        shutil.rmtree(target, ignore_errors=True)

    # -- writing ---------------------------------------------------------------

    def write_bars(
        self,
        frame: pl.DataFrame,
        *,
        dataset: str | None = None,
        schema: str = s.DEFAULT_BAR_SCHEMA,
        mode: WriteMode = "append",
    ) -> int:
        """Validate ``frame`` as canonical bars and persist it partitioned by root."""
        if not s.is_bar_schema(schema):
            raise ValueError(
                f"write_bars requires a bar schema {list(s.ALL_BAR_SCHEMAS)}, got {schema!r}"
            )
        return self.write_frame(frame, schema=schema, dataset=dataset, mode=mode)

    def write_frame(
        self,
        frame: pl.DataFrame,
        *,
        schema: str,
        dataset: str | None = None,
        mode: WriteMode = "append",
        partition_column: str = s.ROOT_SYMBOL_COLUMN,
    ) -> int:
        """Validate and persist ``frame`` under ``schema``, one file per root.

        Returns the number of input rows written (duplicates removed by the
        merge are not counted separately).
        """
        if mode not in ("append", "overwrite"):
            raise ValueError(f"mode must be 'append' or 'overwrite', got {mode!r}")
        if not isinstance(frame, pl.DataFrame):
            raise TypeError(f"frame must be a polars DataFrame, got {type(frame)!r}")
        if partition_column not in frame.columns:
            raise StorageError(
                f"cannot partition schema {schema!r}: missing {partition_column!r} column"
            )
        validated = s.validator_for_schema(schema)(frame)
        key_columns = _key_columns_for(schema)
        rows_written = 0
        for key, partition in validated.group_by(partition_column, maintain_order=True):
            root = _safe_root(str(key[0]))
            partition = s.dedupe_frame(partition, key_columns, keep="last")
            target = self.partition_path(root, schema, dataset)
            with _lock_for(target):
                merged = partition
                if mode == "append" and target.is_file():
                    existing = pl.read_parquet(target)
                    merged = pl.concat([existing, partition], how="vertical_relaxed")
                    merged = s.dedupe_frame(merged, key_columns, keep="last")
                _atomic_write_parquet(merged, target)
            rows_written += partition.height
        return rows_written

    def store_definitions(
        self,
        frame: pl.DataFrame,
        *,
        dataset: str | None = None,
        mode: WriteMode = "append",
    ) -> int:
        """Persist a canonical contract-definition frame."""
        return self.write_frame(
            frame, schema=s.DATABENTO_DEFINITION_SCHEMA, dataset=dataset, mode=mode
        )

    # -- multi-timeframe cache ---------------------------------------------------

    def schema_for_timeframe(self, timeframe: str) -> str:
        """Repository schema for ``timeframe`` (e.g. ``5m`` -> ``ohlcv-5m``)."""
        return tf.schema_for_timeframe(timeframe)

    def write_timeframe(
        self,
        timeframe: str,
        frame: pl.DataFrame,
        *,
        dataset: str | None = None,
        mode: WriteMode = "append",
    ) -> int:
        """Validate canonical bars as ``timeframe`` and persist them in its partition.

        Returns the number of input rows written. ``frame`` must already be at the
        target timeframe (use :func:`src.data.resampler.resample_bars` first).
        """
        spec = tf.get_timeframe(timeframe)
        validated = s.ensure_bars_schema(frame, context=f"write_timeframe[{spec.name}]")
        return self.write_bars(
            validated, dataset=dataset, schema=self.schema_for_timeframe(spec.name), mode=mode
        )

    def read_timeframe(
        self,
        timeframe: str,
        roots: Sequence[InstrumentId] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        dataset: str | None = None,
        *,
        persist: bool = False,
    ) -> pl.DataFrame:
        """Read cached ``timeframe`` bars, auto-resampling a finer base cache.

        If a requested root has no target-timeframe partition but its finer base
        timeframe is cached (``5m/15m/30m/1h`` from ``1m``, ``4h`` from ``1h``,
        ``1w`` from ``1d``), the base bars are resampled look-ahead-safely via
        :func:`src.data.resampler.resample_bars`. ``persist=True`` additionally
        writes the resampled rows into the target schema cache.

        Returns a canonical bar frame for the target timeframe (empty canonical
        frame when neither the target nor a base partition exists).
        """
        spec = tf.get_timeframe(timeframe)
        if start is not None and start.tzinfo is None:
            raise ValueError("start must be timezone-aware (UTC expected)")
        if end is not None and end.tzinfo is None:
            raise ValueError("end must be timezone-aware (UTC expected)")
        if start is not None and end is not None and start > end:
            raise ValueError(
                f"start ({start.isoformat()}) must not be after end ({end.isoformat()})"
            )
        target_schema = self.schema_for_timeframe(spec.name)
        resolved_roots = tuple(s.validate_roots(roots)) if roots is not None else None
        cached = self.read_bars(resolved_roots, start, end, schema=target_schema, dataset=dataset)
        target_roots = set(self.available_roots(target_schema, dataset))
        resampled_frames: list[pl.DataFrame] = []
        base_source = spec.resample_from
        if base_source is not None:
            base_schema = self.schema_for_timeframe(base_source)
            base_roots = set(self.available_roots(base_schema, dataset))
            candidates = (
                base_roots - target_roots
                if resolved_roots is None
                else set(resolved_roots) - target_roots
            )
            missing_roots: list[InstrumentId] = sorted(
                root for root in candidates if root in base_roots
            )
            if missing_roots:
                base = self.read_bars(
                    missing_roots,
                    start,
                    end,
                    schema=base_schema,
                    dataset=dataset,
                )
                if not base.is_empty():
                    resampled = resample_bars(base, spec.name, source_timeframe=base_source)
                    resampled_frames.append(resampled)
                    if persist:
                        self.write_bars(
                            resampled,
                            dataset=dataset,
                            schema=target_schema,
                            mode="append",
                        )
        frames = [frame for frame in [cached, *resampled_frames] if not frame.is_empty()]
        if not frames:
            return self._empty_frame(target_schema)
        merged = pl.concat(frames, how="vertical_relaxed")
        merged = s.validator_for_schema(target_schema)(merged)
        if start is not None or end is not None:
            merged = s.filter_time_window(merged, start, end)
        return s.dedupe_frame(merged, s.BAR_KEY_COLUMNS, keep="last")

    # -- reading ---------------------------------------------------------------

    def read_frame(
        self,
        schema: str = s.DEFAULT_BAR_SCHEMA,
        *,
        roots: Sequence[InstrumentId] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        dataset: str | None = None,
    ) -> pl.DataFrame:
        """Read and re-validate every partition of ``schema`` in ``[start, end)``.

        ``roots=None`` reads every available root. The result is sorted by the
        canonical primary key. Reading an empty schema returns an empty frame
        with the canonical schema (never ``None``).
        """
        if start is not None and start.tzinfo is None:
            raise ValueError("start must be timezone-aware (UTC expected)")
        if end is not None and end.tzinfo is None:
            raise ValueError("end must be timezone-aware (UTC expected)")
        if start is not None and end is not None and start > end:
            raise ValueError(
                f"start ({start.isoformat()}) must not be after end ({end.isoformat()})"
            )
        available = self.available_roots(schema, dataset)
        if roots is not None:
            requested = {_safe_root(root) for root in roots}
            selected = tuple(root for root in available if root in requested)
        else:
            selected = available
        frames: list[pl.DataFrame] = []
        for root in selected:
            path = self.partition_path(root, schema, dataset)
            if path.is_file():
                frames.append(pl.read_parquet(path))
        if not frames:
            return self._empty_frame(schema)
        merged = pl.concat(frames, how="vertical_relaxed")
        merged = s.validator_for_schema(schema)(merged)
        key_columns = _key_columns_for(schema)
        if start is not None or end is not None:
            merged = s.filter_time_window(merged, start, end)
        return s.dedupe_frame(merged, key_columns, keep="last")

    def read_bars(
        self,
        roots: Sequence[InstrumentId] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        schema: str = s.DEFAULT_BAR_SCHEMA,
        *,
        dataset: str | None = None,
    ) -> pl.DataFrame:
        """Read canonical bars for ``roots`` in the half-open window ``[start, end)``."""
        if not s.is_bar_schema(schema):
            raise ValueError(
                f"read_bars requires a bar schema {list(s.ALL_BAR_SCHEMAS)}, got {schema!r}"
            )
        frame = self.read_frame(schema, roots=roots, start=start, end=end, dataset=dataset)
        return s.ensure_bars_schema(frame, context=f"read_bars[{schema}]")

    def read_root(
        self, root: InstrumentId, schema: str = s.DEFAULT_BAR_SCHEMA, dataset: str | None = None
    ) -> pl.DataFrame:
        """Read a single root partition (no time filter)."""
        return self.read_frame(schema, roots=[root], dataset=dataset)

    def load_definitions(
        self,
        roots: Sequence[InstrumentId] | None = None,
        *,
        as_of: datetime | None = None,
        dataset: str | None = None,
        active_only: bool = False,
    ) -> pl.DataFrame:
        """Read point-in-time contract definitions.

        When ``as_of`` is given only definitions whose ``activation`` is at or
        before ``as_of`` are returned (``active_only`` additionally drops
        contracts that already expired by ``as_of``).
        """
        frame = self.read_frame(s.DATABENTO_DEFINITION_SCHEMA, roots=roots, dataset=dataset)
        if frame.is_empty():
            return frame
        if as_of is not None:
            if as_of.tzinfo is None:
                raise ValueError("as_of must be timezone-aware (UTC expected)")
            frame = frame.filter(
                pl.col("activation").is_null() | (pl.col("activation") <= pl.lit(as_of))
            )
        if active_only and as_of is not None:
            frame = frame.filter(pl.col(s.EXPIRATION_COLUMN) >= pl.lit(as_of.date()))
        return frame

    def time_bounds(
        self,
        schema: str = s.DEFAULT_BAR_SCHEMA,
        *,
        root: InstrumentId | None = None,
        dataset: str | None = None,
    ) -> tuple[datetime | None, datetime | None]:
        """``(first, last)`` timestamp present for ``schema`` (``None`` when empty)."""
        roots = [root] if root is not None else None
        frame = self.read_frame(schema, roots=roots, dataset=dataset)
        if frame.is_empty():
            return (None, None)
        column = frame[s.TIMESTAMP_COLUMN]
        return (column.min(), column.max())  # type: ignore[return-value]

    def _empty_frame(self, schema: str) -> pl.DataFrame:
        """Empty frame carrying the canonical schema for ``schema``."""
        if s.is_bar_schema(schema) or schema in {"bars", "ohlcv"}:
            return s.empty_bars_frame()
        if schema == s.DATABENTO_DEFINITION_SCHEMA:
            return s.empty_definitions_frame()
        if schema == "continuous":
            return s.empty_continuous_frame()
        if schema in s.DATABENTO_QUOTE_SCHEMAS:
            return pl.DataFrame(schema=s.QUOTE_COLUMNS)
        if schema in s.DATABENTO_BOOK_SCHEMAS:
            levels = 1 if schema == "mbp-1" else 10
            return pl.DataFrame(schema=s.book_level_columns(levels))
        return s.empty_bars_frame()

    # -- DuckDB analytics ------------------------------------------------------

    def query(self, sql: str, *, dataset: str | None = None) -> pl.DataFrame:
        """Run read-only SQL against the cache and return a Polars frame.

        Every schema that currently holds data is registered as a view named
        after the schema (``ohlcv_1d``, ``ohlcv_1m``, ``tbbo``,
        ``definition``, ``continuous``); the default daily bar schema is also
        exposed as ``bars``. The connection is in-memory and read-only: the
        Parquet files stay the single source of truth.
        """
        if not sql.strip():
            raise ValueError("sql must be a non-empty statement")
        connection = duckdb.connect()
        try:
            self._register_views(connection, dataset=dataset)
            table = connection.sql(sql).to_arrow_table()
        finally:
            connection.close()
        if table is None:  # pragma: no cover - DuckDB never returns None for SELECT
            return pl.DataFrame()
        return cast(pl.DataFrame, pl.from_arrow(table))

    def query_bars(
        self,
        sql: str,
        *,
        schema: str = s.DEFAULT_BAR_SCHEMA,
        dataset: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pl.DataFrame:
        """Run SQL with a ``bars`` view pinned to ``schema``.

        ``bars`` is materialised from a time-filtered subquery when ``start`` /
        ``end`` are given, otherwise it points straight at the Parquet glob.
        """
        if not s.is_bar_schema(schema):
            raise ValueError(
                f"query_bars requires a bar schema {list(s.DATABENTO_BAR_SCHEMAS)}, got {schema!r}"
            )
        pattern = self._schema_glob(schema, dataset)
        if pattern is None:
            return s.empty_bars_frame()
        relation = f"read_parquet('{pattern}', union_by_name = true)"
        if start is not None or end is not None:
            clauses = ["TRUE"]
            if start is not None:
                clauses.append(f"\"{s.TIMESTAMP_COLUMN}\" >= TIMESTAMPTZ '{start.isoformat()}'")
            if end is not None:
                clauses.append(f"\"{s.TIMESTAMP_COLUMN}\" < TIMESTAMPTZ '{end.isoformat()}'")
            relation = f"(SELECT * FROM {relation} WHERE {' AND '.join(clauses)})"
        connection = duckdb.connect()
        try:
            self._register_views(connection, dataset=dataset)
            connection.execute(f"CREATE OR REPLACE VIEW bars AS SELECT * FROM {relation}")
            table = connection.sql(sql).to_arrow_table()
        finally:
            connection.close()
        if table is None:  # pragma: no cover - DuckDB never returns None for SELECT
            return pl.DataFrame()
        return cast(pl.DataFrame, pl.from_arrow(table))

    def _schema_glob(self, schema: str, dataset: str | None) -> str | None:
        """Parquet glob for a schema, or ``None`` when it holds no data."""
        base = self.schema_dir(schema, dataset)
        if not base.is_dir() or not any(base.glob(f"*/{PARTITION_FILE_NAME}")):
            return None
        pattern = base / "*" / PARTITION_FILE_NAME
        return str(pattern).replace("'", "''")

    def _register_views(
        self, connection: duckdb.DuckDBPyConnection, *, dataset: str | None
    ) -> None:
        """Register one DuckDB view per non-empty schema directory."""
        base = self.dataset_dir(dataset)
        if not base.is_dir():
            return
        bar_views: list[str] = []
        for schema_dir in sorted(base.iterdir()):
            if not schema_dir.is_dir():
                continue
            schema_name = schema_dir.name
            pattern = self._schema_glob(schema_name, dataset)
            if pattern is None:
                continue
            view = _view_name(schema_name)
            statement = (
                f'CREATE OR REPLACE VIEW "{view}" AS '
                f"SELECT * FROM read_parquet('{pattern}', union_by_name = true)"
            )
            connection.execute(statement)
            if s.is_bar_schema(schema_name):
                bar_views.append(view)
            if schema_name == s.DATABENTO_DEFINITION_SCHEMA:
                connection.execute(
                    'CREATE OR REPLACE VIEW "definitions" AS '
                    f"SELECT * FROM read_parquet('{pattern}', union_by_name = true)"
                )
        aliased = (
            _view_name(s.DEFAULT_BAR_SCHEMA)
            if _view_name(s.DEFAULT_BAR_SCHEMA) in bar_views
            else (bar_views[0] if bar_views else None)
        )
        if aliased is not None:
            connection.execute(f'CREATE OR REPLACE VIEW "bars" AS SELECT * FROM "{aliased}"')

    # -- misc ------------------------------------------------------------------

    def summary(self, dataset: str | None = None) -> Mapping[str, Mapping[str, tuple[str, ...]]]:
        """``dataset -> schema -> roots`` inventory of the cache (read-only)."""
        resolved = dataset or self._default_dataset
        return {
            resolved: {
                schema: self.available_roots(schema, resolved)
                for schema in self.list_schemas(resolved)
            }
        }
