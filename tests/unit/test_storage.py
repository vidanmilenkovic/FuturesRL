"""Unit tests for the partitioned Parquet repository.

Everything runs against ``tmp_path`` (the ``settings`` fixture) and the
deterministic market-data factories: no network, no Databento, no IBKR.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from configs.settings import Settings
from src.data import schema as s
from src.data.storage import (
    PARTITION_FILE_NAME,
    ParquetRepository,
    StorageError,
)

from tests.fixtures.market import DEFAULT_EXPIRATION, FIXED_TIMESTAMP, make_bars_frame


def bar_frame(
    roots: tuple[str, ...] = ("ES", "NQ"), periods: int = 6, seed: int = 7
) -> pl.DataFrame:
    """Deterministic canonical daily bars for ``roots``."""
    return make_bars_frame(roots, periods=periods, start=FIXED_TIMESTAMP, seed=seed)


def definitions_frame(root: str = "ES") -> pl.DataFrame:
    """Deterministic canonical definition frame for ``root``."""
    return s.ensure_definitions_schema(
        pl.DataFrame(
            {
                s.INSTRUMENT_ID_COLUMN: [f"{root}_TEST"],
                s.ROOT_SYMBOL_COLUMN: [root],
                s.EXPIRATION_COLUMN: [DEFAULT_EXPIRATION],
                "multiplier": [50.0 if root == "ES" else 20.0],
                "tick_size": [0.25],
                "exchange": ["CME"],
                "currency": ["USD"],
                "activation": [datetime(2023, 1, 1, tzinfo=UTC)],
            }
        )
    )


def quote_frame(root: str = "ES", periods: int = 3) -> pl.DataFrame:
    """Deterministic canonical top-of-book frame for ``root``."""
    stamps = [FIXED_TIMESTAMP + timedelta(minutes=5 * index) for index in range(periods)]
    return s.ensure_quote_schema(
        pl.DataFrame(
            {
                s.TIMESTAMP_COLUMN: stamps,
                s.INSTRUMENT_ID_COLUMN: [f"{root}_TEST"] * periods,
                s.ROOT_SYMBOL_COLUMN: [root] * periods,
                s.EXPIRATION_COLUMN: [DEFAULT_EXPIRATION] * periods,
                s.BID_PRICE_COLUMN: [100.0 + index for index in range(periods)],
                s.ASK_PRICE_COLUMN: [100.25 + index for index in range(periods)],
                s.BID_SIZE_COLUMN: [10.0] * periods,
                s.ASK_SIZE_COLUMN: [12.0] * periods,
                s.BID_COUNT_COLUMN: [1.0] * periods,
                s.ASK_COUNT_COLUMN: [1.0] * periods,
            }
        )
    )


# ---------------------------------------------------------------------------
# Layout & lifecycle
# ---------------------------------------------------------------------------


def test_layout_and_paths(settings: Settings) -> None:
    repo = ParquetRepository(settings)
    expected_root = settings.parquet_dir / settings.databento_dataset / s.DEFAULT_BAR_SCHEMA / "ES"
    assert repo.dataset == settings.databento_dataset
    assert repo.parquet_dir == settings.parquet_dir
    assert repo.duckdb_path == settings.duckdb_path
    assert repo.dataset_dir() == settings.parquet_dir / settings.databento_dataset
    assert repo.schema_dir(s.DEFAULT_BAR_SCHEMA) == expected_root.parent
    assert repo.root_dir("es", s.DEFAULT_BAR_SCHEMA) == expected_root
    assert repo.partition_path("ES") == expected_root / PARTITION_FILE_NAME
    assert "ParquetRepository" in repr(repo)


def test_lifecycle_helpers(settings: Settings) -> None:
    repo = ParquetRepository(settings)
    assert repo.list_datasets() == ()
    assert repo.list_schemas() == ()
    assert repo.available_roots() == ()
    assert not repo.has_data()

    repo.write_bars(bar_frame(("ES", "NQ")))
    assert repo.list_datasets() == (settings.databento_dataset,)
    assert repo.list_schemas() == (s.DEFAULT_BAR_SCHEMA,)
    assert repo.available_roots() == ("ES", "NQ")
    assert repo.has_data()
    assert repo.summary()[settings.databento_dataset][s.DEFAULT_BAR_SCHEMA] == ("ES", "NQ")

    repo.delete_root("ES")
    assert repo.available_roots() == ("NQ",)
    repo.delete_schema(s.DEFAULT_BAR_SCHEMA)
    assert repo.available_roots() == ()
    repo.delete_dataset()
    assert repo.list_datasets() == ()


def test_unsafe_root_and_partition_column_rejected(settings: Settings) -> None:
    repo = ParquetRepository(settings)
    with pytest.raises(ValueError):
        repo.root_dir("../evil")
    frame = bar_frame(("ES",)).drop(s.ROOT_SYMBOL_COLUMN)
    with pytest.raises(StorageError):
        repo.write_frame(frame, schema=s.DEFAULT_BAR_SCHEMA)


# ---------------------------------------------------------------------------
# Write / read
# ---------------------------------------------------------------------------


def test_write_read_roundtrip(settings: Settings) -> None:
    repo = ParquetRepository(settings)
    frame = bar_frame(("ES", "NQ"), periods=5)
    assert repo.write_bars(frame) == frame.height
    assert repo.read_bars().height == frame.height
    read = repo.read_bars(["ES"])
    assert read.height == 5
    assert read["root_symbol"].unique().to_list() == ["ES"]
    assert read.columns == list(s.BAR_COLUMNS)
    assert s.ensure_bars_schema(read).equals(read)
    assert read[s.TIMESTAMP_COLUMN].is_sorted()
    assert [path.name for path in repo.root_dir("ES").iterdir()] == [PARTITION_FILE_NAME]


def test_append_dedupes_on_primary_key(settings: Settings) -> None:
    repo = ParquetRepository(settings)
    frame = bar_frame(("ES",), periods=4)
    repo.write_bars(frame)
    assert repo.read_bars(["ES"]).height == 4

    # Re-ingesting the same window must not duplicate rows.
    repo.write_bars(frame)
    assert repo.read_bars(["ES"]).height == 4

    # The newer row for an existing key wins (overlapping re-ingest).
    updated = frame.with_columns((pl.col(s.CLOSE_COLUMN) + 1.0).alias(s.CLOSE_COLUMN))
    repo.write_bars(updated)
    read = repo.read_bars(["ES"])
    assert read.height == 4
    assert read[s.CLOSE_COLUMN].to_list() == updated[s.CLOSE_COLUMN].to_list()


def test_overwrite_mode_replaces_partition(settings: Settings) -> None:
    repo = ParquetRepository(settings)
    full = bar_frame(("ES", "NQ"), periods=6)
    repo.write_bars(full)
    replacement = bar_frame(("ES",), periods=2, seed=11)
    repo.write_bars(replacement, mode="overwrite")
    assert repo.read_bars(["ES"]).height == 2
    assert repo.read_bars(["NQ"]).height == 6
    with pytest.raises(ValueError):
        repo.write_bars(replacement, mode="upsert")  # type: ignore[arg-type]


def test_time_window_is_half_open(settings: Settings) -> None:
    repo = ParquetRepository(settings)
    frame = bar_frame(("ES",), periods=6)
    repo.write_bars(frame)
    stamps = frame[s.TIMESTAMP_COLUMN].to_list()
    window = repo.read_bars(["ES"], stamps[1], stamps[4])
    assert window[s.TIMESTAMP_COLUMN].to_list() == stamps[1:4]
    assert repo.read_bars(["ES"], start=stamps[5]).height == 1
    assert repo.read_bars(["ES"], end=stamps[0]).height == 0
    with pytest.raises(ValueError):
        repo.read_bars(["ES"], stamps[4], stamps[1])
    with pytest.raises(ValueError):
        repo.read_bars(["ES"], start=datetime(2024, 1, 1))  # naive
    with pytest.raises(ValueError):
        repo.read_bars(["ES"], end=datetime(2024, 1, 1))  # naive


def test_schema_enforcement(settings: Settings) -> None:
    repo = ParquetRepository(settings)
    frame = bar_frame(("ES",), periods=3)
    with pytest.raises(s.SchemaError):
        repo.write_bars(frame.drop(s.CLOSE_COLUMN))
    invalid = frame.with_columns(pl.col(s.HIGH_COLUMN) - 1000.0)
    with pytest.raises(s.SchemaError):
        repo.write_bars(invalid)
    with pytest.raises(ValueError):
        repo.write_bars(frame, schema="tbbo")
    with pytest.raises(ValueError):
        repo.read_bars(["ES"], schema="tbbo")


def test_empty_reads_return_canonical_empty_frames(settings: Settings) -> None:
    repo = ParquetRepository(settings)
    empty = repo.read_bars(["ES"])
    assert empty.is_empty()
    assert empty.columns == list(s.BAR_COLUMNS)
    assert repo.read_bars(["ES"], schema="ohlcv-1m").columns == list(s.BAR_COLUMNS)
    assert repo.read_frame("definition").columns == list(s.DEFINITION_COLUMNS)
    assert repo.read_frame("tbbo").columns == list(s.QUOTE_COLUMNS)
    assert repo.time_bounds() == (None, None)


def test_generic_frames_and_non_bar_schemas(settings: Settings) -> None:
    repo = ParquetRepository(settings)
    quotes = quote_frame("ES", periods=4)
    assert repo.write_frame(quotes, schema="tbbo") == 4
    assert repo.available_roots("tbbo") == ("ES",)
    back = repo.read_frame("tbbo", roots=["ES"])
    assert back.height == 4
    assert back.columns[: len(s.QUOTE_COLUMNS)] == list(s.QUOTE_COLUMNS)
    with pytest.raises(ValueError):
        repo.write_frame(quotes, schema=s.DEFAULT_BAR_SCHEMA)


def test_definitions_store_and_point_in_time_load(settings: Settings) -> None:
    repo = ParquetRepository(settings)
    definitions = pl.concat([definitions_frame("ES"), definitions_frame("NQ")])
    assert repo.store_definitions(definitions) == 2
    assert repo.available_roots("definition") == ("ES", "NQ")
    assert repo.load_definitions().height == 2
    assert repo.load_definitions(["ES"]).height == 1
    before = repo.load_definitions(as_of=datetime(2022, 6, 1, tzinfo=UTC))
    assert before.is_empty()  # activation is 2023-01-01
    after = repo.load_definitions(as_of=datetime(2024, 1, 1, tzinfo=UTC))
    assert after.height == 2
    with pytest.raises(ValueError):
        repo.load_definitions(as_of=datetime(2024, 1, 1))  # naive


def test_time_bounds(settings: Settings) -> None:
    repo = ParquetRepository(settings)
    frame = bar_frame(("ES",), periods=4)
    repo.write_bars(frame)
    stamps = frame[s.TIMESTAMP_COLUMN].to_list()
    assert repo.time_bounds() == (stamps[0], stamps[-1])
    assert repo.time_bounds(root="ES") == (stamps[0], stamps[-1])
    assert repo.time_bounds(root="NQ") == (None, None)


# ---------------------------------------------------------------------------
# DuckDB facade
# ---------------------------------------------------------------------------


def test_duckdb_query_and_query_bars(settings: Settings) -> None:
    repo = ParquetRepository(settings)
    frame = bar_frame(("ES", "NQ"), periods=5)
    repo.write_bars(frame)
    repo.store_definitions(definitions_frame("ES"))

    grouped = repo.query(
        "select root_symbol, count(*) as n from bars group by root_symbol order by root_symbol"
    )
    assert grouped["root_symbol"].to_list() == ["ES", "NQ"]
    assert grouped["n"].to_list() == [5, 5]
    assert repo.query("select count(*) as n from definitions")["n"].to_list() == [1]

    # query_bars pins the `bars` view to a schema and optional window.
    stamps = frame[s.TIMESTAMP_COLUMN].to_list()
    windowed = repo.query_bars("select count(*) as n from bars", start=stamps[1], end=stamps[3])
    assert windowed["n"].to_list() == [4]
    with pytest.raises(ValueError):
        repo.query("   ")
    with pytest.raises(ValueError):
        repo.query_bars("select 1", schema="tbbo")


def test_duckdb_views_expose_every_schema(settings: Settings) -> None:
    repo = ParquetRepository(settings)
    repo.write_bars(bar_frame(("ES",), periods=3))
    repo.write_frame(quote_frame("ES"), schema="tbbo")
    repo.write_bars(bar_frame(("ES",), periods=2), schema="ohlcv-1m")
    tables = repo.query("select table_name from information_schema.tables")
    names = set(tables["table_name"].to_list())
    assert {"bars", "ohlcv_1d", "ohlcv_1m", "tbbo"} <= names


# ---------------------------------------------------------------------------
# Atomicity / concurrency
# ---------------------------------------------------------------------------


def test_atomic_writes_leave_no_temporary_directories(settings: Settings) -> None:
    repo = ParquetRepository(settings)
    repo.write_bars(bar_frame(("ES", "NQ"), periods=3))
    schema_dir = repo.schema_dir(s.DEFAULT_BAR_SCHEMA)
    leftovers = [path for path in schema_dir.rglob("*") if path.name.endswith(".tmp")]
    assert leftovers == []
    for root in ("ES", "NQ"):
        target = repo.partition_path(root)
        assert target.is_file()
        assert pl.read_parquet(target).height == 3


def test_concurrent_writes_of_disjoint_roots(settings: Settings) -> None:
    repo = ParquetRepository(settings)
    roots = ("ES", "NQ", "CL", "GC")
    errors: list[BaseException] = []

    def write(root: str) -> None:
        try:
            repo.write_bars(bar_frame((root,), periods=4, seed=3))
        except BaseException as exc:  # pragma: no cover - failure surface
            errors.append(exc)

    threads = [threading.Thread(target=write, args=(root,)) for root in roots]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert repo.available_roots() == tuple(sorted(roots))
    assert repo.read_bars().height == 4 * len(roots)


def test_partition_files_are_independent_per_root(settings: Settings) -> None:
    repo = ParquetRepository(settings)
    repo.write_bars(bar_frame(("ES", "NQ"), periods=3))
    es_path = repo.partition_path("ES")
    nq_path = repo.partition_path("NQ")
    assert es_path != nq_path
    assert es_path.parent.parent.parent == repo.dataset_dir()
    assert repo.dataset_dir().parent == Path(settings.parquet_dir)


# ---------------------------------------------------------------------------
# Canonical schema helper coverage (extension of the frozen schema contract)
# ---------------------------------------------------------------------------


def test_normalise_timestamp_column_accepts_common_dtypes() -> None:
    naive = pl.DataFrame({"ts": [datetime(2024, 1, 2, 21, 0)]})
    assert s.normalise_timestamp_column(naive, "ts").schema["ts"] == pl.Datetime("us", "UTC")

    dates = pl.DataFrame({"ts": [DEFAULT_EXPIRATION]})
    normalised = s.normalise_timestamp_column(dates, "ts")
    assert normalised.schema["ts"] == pl.Datetime("us", "UTC")
    assert normalised["ts"][0].date() == DEFAULT_EXPIRATION

    epochs = pl.DataFrame({"ts": [1_704_229_200_000_000_000]})
    assert s.normalise_timestamp_column(epochs, "ts")["ts"][0] == datetime(
        2024, 1, 2, 21, 0, tzinfo=UTC
    )
    with pytest.raises(s.SchemaError):
        s.normalise_timestamp_column(pl.DataFrame({"other": [1]}), "ts")


def test_schema_validators_reject_invalid_frames() -> None:
    with pytest.raises(s.SchemaError):
        s.ensure_quote_schema(pl.DataFrame({"timestamp_utc": [], "instrument_id": []}))
    crossed = quote_frame("ES", periods=1).with_columns(pl.lit(200.0).alias(s.BID_PRICE_COLUMN))
    with pytest.raises(s.SchemaError, match="bid_px > ask_px"):
        s.ensure_quote_schema(crossed)
    null_bid = quote_frame("ES", periods=1).with_columns(
        pl.lit(None, dtype=pl.Float64).alias(s.BID_PRICE_COLUMN)
    )
    with pytest.raises(s.SchemaError, match="bid_px must not be null"):
        s.ensure_quote_schema(null_bid)

    with pytest.raises(s.SchemaError):
        s.ensure_definitions_schema(pl.DataFrame({"root_symbol": ["ES"]}))
    null_key = pl.DataFrame(
        {
            s.INSTRUMENT_ID_COLUMN: [None],
            s.ROOT_SYMBOL_COLUMN: ["ES"],
            s.EXPIRATION_COLUMN: [DEFAULT_EXPIRATION],
        }
    )
    with pytest.raises(s.SchemaError, match="must not be null"):
        s.ensure_definitions_schema(null_key)

    with pytest.raises(s.SchemaError):
        s.ensure_continuous_schema(pl.DataFrame({"timestamp_utc": []}))
    with pytest.raises(s.SchemaError):
        s.ensure_key_schema(pl.DataFrame({"timestamp_utc": []}))
    with pytest.raises(s.SchemaError):
        s.dedupe_frame(pl.DataFrame({"timestamp_utc": []}))
    with pytest.raises(s.SchemaError):
        s.filter_time_window(pl.DataFrame({"other": []}))
    with pytest.raises(ValueError):
        s.filter_time_window(quote_frame("ES"), datetime(2024, 1, 1))
    with pytest.raises(ValueError):
        s.filter_time_window(quote_frame("ES"), end=datetime(2024, 1, 1))


def test_schema_catalogue_helpers() -> None:
    assert s.is_bar_schema("ohlcv-1d") is True
    assert s.is_bar_schema("tbbo") is False
    assert s.bar_interval("ohlcv-1m") == timedelta(minutes=1)
    assert s.bar_interval("ohlcv-1h") == timedelta(hours=1)
    assert s.bar_interval("ohlcv-1d") == timedelta(days=1)
    with pytest.raises(ValueError):
        s.bar_interval("tbbo")
    assert s.validator_for_schema("ohlcv-1m") is s.ensure_bars_schema
    assert s.validator_for_schema("definition") is s.ensure_definitions_schema
    assert s.validator_for_schema("tbbo") is s.ensure_quote_schema
    assert s.validator_for_schema("continuous") is s.ensure_continuous_schema
    assert s.validator_for_schema("mbp-10") is s.ensure_key_schema
    assert set(s.BOOK_KEY_COLUMNS) == set(s.BAR_KEY_COLUMNS)
    assert set(s.QUOTE_KEY_COLUMNS) == set(s.BAR_KEY_COLUMNS)
    assert s.DEFINITION_KEY_COLUMNS == (s.INSTRUMENT_ID_COLUMN, s.EXPIRATION_COLUMN)
    assert s.CONTINUOUS_KEY_COLUMNS == (s.TIMESTAMP_COLUMN, s.ROOT_SYMBOL_COLUMN)
    with pytest.raises(ValueError):
        s.book_level_columns(0)
    assert "bid_px_09" in s.book_level_columns(10)
    assert set(s.MBP_1_COLUMNS) < set(s.MBP_10_COLUMNS)
    assert set(s.REQUIRED_FEATURES) < set(s.ALL_FEATURES)


def test_dedupe_frame_keeps_requested_row() -> None:
    frame = pl.DataFrame({"a": [1, 1, 2], "b": [1.0, 2.0, 3.0]})
    last = s.dedupe_frame(frame, ("a",), keep="last")
    assert last.sort("a")["b"].to_list() == [2.0, 3.0]
    first = s.dedupe_frame(frame, ("a",), keep="first")
    assert first.sort("a")["b"].to_list() == [1.0, 3.0]
