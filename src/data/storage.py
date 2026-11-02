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

import re
import shutil
import tempfile
import threading
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

import duckdb
import polars as pl
from configs.settings import Settings, get_settings

from src.core.types import InstrumentId
from src.data import schema as s

__all__ = ["ParquetRepository", "StorageError", "WriteMode"]

#: How :meth:`ParquetRepository.write_frame` merges with existing data.
type WriteMode = Literal["append", "overwrite"]

#: File name of the single Parquet partition per (dataset, schema, root).
PARTITION_FILE_NAME = "part.parquet"

#: Characters allowed in a root symbol (also protects against path traversal).
_ROOT_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9._-]*$")

#: In-process lock registry: one re-entrant lock per absolute target path.
_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


class StorageError(RuntimeError):
    """Raised when the on-disk cache cannot satisfy a storage operation."""


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
                f"write_bars requires a bar schema {list(s.DATABENTO_BAR_SCHEMAS)}, got {schema!r}"
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
                f"read_bars requires a bar schema {list(s.DATABENTO_BAR_SCHEMAS)}, got {schema!r}"
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
        if schema in s.DATABENTO_BAR_SCHEMAS or schema in {"bars", "ohlcv"}:
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
