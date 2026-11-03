"""Unit tests for Databento historical ingestion and the offline synthetic source.

All Databento interactions use an in-memory fake ``Historical`` client; the SDK
and the network are never touched. Missing SDK / missing API key must raise
clear, actionable ``RuntimeError``s instead of silently faking vendor data.
"""

from __future__ import annotations

import sys
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd
import polars as pl
import pytest
from configs.settings import Settings
from src.core.interfaces import HistoricalDataSource
from src.data import schema as s
from src.data.databento_loader import (
    DatabentoIngestError,
    DatabentoLoader,
    OfflineBarSource,
    synthetic_reference_price,
)
from src.data.storage import ParquetRepository

from tests.fixtures.market import DEFAULT_EXPIRATION, FIXED_TIMESTAMP

START = FIXED_TIMESTAMP
END = START + timedelta(days=4)


def ns(moment: datetime) -> int:
    """Convert a datetime to Databento nanoseconds since the epoch."""
    return int(moment.timestamp() * 1e9)


class FakeStore:
    """Minimal ``databento.DBNStore`` stand-in backed by a Polars frame."""

    def __init__(self, frame: pl.DataFrame | pd.DataFrame) -> None:
        self._frame = frame

    def to_arrow(self) -> Any:
        return self._frame.to_arrow()

    def to_df(self) -> pd.DataFrame:
        if isinstance(self._frame, pd.DataFrame):
            return self._frame
        return self._frame.to_pandas()


class FakeTimeseries:
    """Records ``get_range`` calls and returns a canned store."""

    def __init__(self, store: FakeStore | None = None, error: Exception | None = None) -> None:
        self._store = store
        self._error = error
        self.calls: list[dict[str, Any]] = []

    def get_range(self, **kwargs: Any) -> FakeStore:
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        assert self._store is not None
        return self._store


class FakeClient:
    """Minimal ``databento.Historical`` stand-in."""

    def __init__(self, store: FakeStore | None = None, error: Exception | None = None) -> None:
        self.timeseries = FakeTimeseries(store, error)


class RawTimeseries:
    """``get_range`` returning an arbitrary object (unsupported response test)."""

    def get_range(self, **kwargs: Any) -> object:
        del kwargs
        return object()


class RawClient:
    """Client whose store object cannot be converted to a frame."""

    def __init__(self) -> None:
        self.timeseries = RawTimeseries()


def ohlcv_raw(symbol: str = "ESM4", periods: int = 3) -> pl.DataFrame:
    """Raw Databento ``ohlcv-1d`` frame (nanosecond event timestamps)."""
    stamps = [START + timedelta(days=index) for index in range(periods)]
    closes = [4800.0 + index for index in range(periods)]
    return pl.DataFrame(
        {
            "ts_event": [ns(moment) for moment in stamps],
            "symbol": [symbol] * periods,
            "open": closes,
            "high": [value + 5.0 for value in closes],
            "low": [value - 5.0 for value in closes],
            "close": closes,
            "volume": [1_000.0] * periods,
        }
    )


def tbbo_raw(symbol: str = "ESM4") -> pl.DataFrame:
    """Raw Databento ``tbbo`` frame."""
    stamps = [START + timedelta(minutes=index) for index in range(3)]
    return pl.DataFrame(
        {
            "ts_recv": [ns(moment) for moment in stamps],
            "symbol": [symbol] * 3,
            "bid_px": [4800.0, 4800.25, 4800.5],
            "ask_px": [4800.25, 4800.5, 4800.75],
            "bid_sz": [10.0, 11.0, 12.0],
            "ask_sz": [9.0, 10.0, 11.0],
            "bid_ct": [1.0, 1.0, 1.0],
            "ask_ct": [1.0, 1.0, 1.0],
        }
    )


def mbp_raw(symbol: str = "ESM4") -> pl.DataFrame:
    """Raw Databento ``mbp-1`` frame."""
    stamps = [START + timedelta(seconds=index) for index in range(2)]
    return pl.DataFrame(
        {
            "ts_event": [ns(moment) for moment in stamps],
            "symbol": [symbol] * 2,
            "bid_px_00": [4800.0, 4800.25],
            "bid_sz_00": [10.0, 11.0],
            "ask_px_00": [4800.25, 4800.5],
            "ask_sz_00": [9.0, 10.0],
        }
    )


def make_loader(settings: Settings, **kwargs: Any) -> DatabentoLoader:
    """Loader bound to a tmp_path repository."""
    return DatabentoLoader(settings, repository=ParquetRepository(settings), **kwargs)


# ---------------------------------------------------------------------------
# Missing SDK / credentials
# ---------------------------------------------------------------------------


def test_ingest_without_api_key_raises_actionable_error(settings: Settings) -> None:
    loader = make_loader(settings)
    with pytest.raises(RuntimeError, match="FUTURESRLX_DATABENTO_API_KEY"):
        loader.ingest(["ES"], START, END)


def test_ingest_without_sdk_raises_actionable_error(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "databento", None)
    loader = make_loader(settings, api_key="fake")
    with pytest.raises(RuntimeError, match="Databento SDK is not installed"):
        loader.ingest(["ES"], START, END)


def test_loader_properties(settings: Settings) -> None:
    loader = make_loader(settings, api_key="secret", dataset="XNAS.ITCH")
    assert loader.dataset == "XNAS.ITCH"
    assert loader.repository.parquet_dir == settings.parquet_dir
    assert loader.instrument_master.has("ES")
    assert loader.available_roots() == ()


# ---------------------------------------------------------------------------
# Bar ingestion / normalisation
# ---------------------------------------------------------------------------


def test_ingest_bars_normalises_and_caches(settings: Settings) -> None:
    loader = make_loader(settings)
    timeseries = FakeTimeseries(FakeStore(ohlcv_raw(periods=3)))
    client = FakeClient()
    client.timeseries = timeseries

    frame = loader.ingest(["ES"], START, END, client=client)

    assert frame.columns == list(s.BAR_COLUMNS)
    assert frame.height == 3
    assert frame[s.ROOT_SYMBOL_COLUMN].unique().to_list() == ["ES"]
    assert frame[s.INSTRUMENT_ID_COLUMN].unique().to_list() == ["ESM4"]
    assert frame[s.EXPIRATION_COLUMN].unique().to_list() == [date(2024, 6, 21)]
    assert frame[s.OPEN_INTEREST_COLUMN].to_list() == [0.0, 0.0, 0.0]
    assert s.ensure_bars_schema(frame).equals(frame)

    call = timeseries.calls[0]
    assert call["dataset"] == settings.databento_dataset
    assert call["schema"] == s.DEFAULT_BAR_SCHEMA
    assert call["symbols"] == ["ES.FUT"]
    assert call["stype_in"] == "parent"
    assert call["start"] == START
    assert call["end"] == END
    assert loader.repository.available_roots(s.DEFAULT_BAR_SCHEMA) == ("ES",)


def test_ingest_accepts_pandas_store(settings: Settings) -> None:
    loader = make_loader(settings)
    pandas_frame = pd.DataFrame(
        {
            "ts_event": [ns(START)],
            "symbol": ["ESM4"],
            "open": [4800.0],
            "high": [4805.0],
            "low": [4795.0],
            "close": [4801.0],
            "volume": [10.0],
        }
    )
    frame = loader.ingest(["ES"], START, END, client=FakeClient(FakeStore(pandas_frame)))
    assert frame.height == 1


def test_ingest_empty_response_raises(settings: Settings) -> None:
    loader = make_loader(settings)
    empty = FakeStore(pl.DataFrame({"ts_event": [], "symbol": []}))
    with pytest.raises(DatabentoIngestError, match="no 'ohlcv-1d' rows"):
        loader.ingest(["ES"], START, END, client=FakeClient(empty))


def test_ingest_vendor_failure_raises(settings: Settings) -> None:
    loader = make_loader(settings)
    client = FakeClient(error=ConnectionError("vendor down"))
    with pytest.raises(DatabentoIngestError, match="get_range failed"):
        loader.ingest(["ES"], START, END, client=client)


def test_ingest_validation_errors(settings: Settings) -> None:
    loader = make_loader(settings)
    with pytest.raises(ValueError, match="unsupported Databento schema"):
        loader.ingest(["ES"], START, END, schema="not-a-schema", client=FakeClient())
    with pytest.raises(ValueError):
        loader.ingest([], START, END, client=FakeClient())
    with pytest.raises(ValueError):
        loader.ingest(["ES"], datetime(2024, 1, 1), END, client=FakeClient())  # naive
    with pytest.raises(ValueError):
        loader.ingest(["ES"], END, START, client=FakeClient())
    with pytest.raises(DatabentoIngestError, match="unsupported Databento response"):
        loader.ingest(["ES"], START, END, client=RawClient())


def test_ingest_continuous_symbol_requires_definitions(settings: Settings) -> None:
    loader = make_loader(settings)
    raw = ohlcv_raw(symbol="ES.v.0")
    with pytest.raises(DatabentoIngestError, match="definition"):
        loader.ingest(["ES"], START, END, client=FakeClient(FakeStore(raw)), stype_in="continuous")


def test_ingest_continuous_symbol_with_definition_join(settings: Settings) -> None:
    loader = make_loader(settings)
    # Store a definition whose instrument_id is the native continuous symbol.
    definitions = s.ensure_definitions_schema(
        pl.DataFrame(
            {
                s.INSTRUMENT_ID_COLUMN: ["ES.v.0"],
                s.ROOT_SYMBOL_COLUMN: ["ES"],
                s.EXPIRATION_COLUMN: [DEFAULT_EXPIRATION],
                "multiplier": [50.0],
                "tick_size": [0.25],
                "exchange": ["CME"],
                "currency": ["USD"],
                "activation": [datetime(2023, 1, 1, tzinfo=UTC)],
            }
        )
    )
    loader.repository.store_definitions(definitions)
    client = FakeClient(FakeStore(ohlcv_raw(symbol="ES.v.0", periods=2)))
    frame = loader.ingest(["ES"], START, END, client=client, stype_in="continuous")
    assert frame[s.INSTRUMENT_ID_COLUMN].to_list() == ["ES.v.0", "ES.v.0"]
    assert client.timeseries.calls[0]["symbols"] == ["ES.v.0"]


def test_ingest_quotes_schema(settings: Settings) -> None:
    loader = make_loader(settings)
    frame = loader.ingest(["ES"], START, END, "tbbo", client=FakeClient(FakeStore(tbbo_raw())))
    assert frame.height == 3
    assert frame.columns[: len(s.QUOTE_COLUMNS)] == list(s.QUOTE_COLUMNS)
    assert frame[s.BID_PRICE_COLUMN].to_list() == [4800.0, 4800.25, 4800.5]
    assert loader.repository.available_roots("tbbo") == ("ES",)


def test_ingest_book_schema(settings: Settings) -> None:
    loader = make_loader(settings)
    frame = loader.ingest(["ES"], START, END, "mbp-1", client=FakeClient(FakeStore(mbp_raw())))
    assert frame.height == 2
    assert "bid_px_00" in frame.columns
    assert frame["bid_px_00"].dtype == pl.Float64
    assert loader.repository.available_roots("mbp-1") == ("ES",)


def test_ingest_definitions_and_load(settings: Settings) -> None:
    loader = make_loader(settings)
    padded = pl.DataFrame(
        {
            "ts_event": [ns(START), ns(START)],
            "symbol": ["ESM4", "ESU4"],
            "expiration": [
                ns(datetime(2024, 6, 21, tzinfo=UTC)),
                ns(datetime(2024, 9, 20, tzinfo=UTC)),
            ],
            "unit_of_measure_qty": [50.0, 50.0],
            "min_price_increment": [0.25, 0.25],
            "exchange": ["CME", "CME"],
            "currency": ["USD", "USD"],
            "activation": [ns(datetime(2023, 1, 1, tzinfo=UTC))] * 2,
        }
    )
    frame = loader.ingest(["ES"], START, END, "definition", client=FakeClient(FakeStore(padded)))
    assert frame.columns == list(s.DEFINITION_COLUMNS)
    assert frame.height == 2
    assert frame[s.ROOT_SYMBOL_COLUMN].to_list() == ["ES", "ES"]
    assert frame["multiplier"].to_list() == [50.0, 50.0]
    assert loader.repository.available_roots("definition") == ("ES",)

    cached = loader.load_definitions(["ES"], as_of=datetime(2024, 1, 1, tzinfo=UTC))
    assert cached.height == 2
    with pytest.raises(ValueError):
        loader.load_definitions(["ES"], as_of=datetime(2024, 1, 1))  # naive


def test_load_definitions_cache_miss_requires_credentials(settings: Settings) -> None:
    loader = make_loader(settings)
    with pytest.raises(RuntimeError, match="FUTURESRLX_DATABENTO_API_KEY"):
        loader.load_definitions(["ES"], as_of=datetime(2024, 1, 1, tzinfo=UTC))


# ---------------------------------------------------------------------------
# HistoricalDataSource protocol: cache-first load
# ---------------------------------------------------------------------------


def test_load_returns_complete_cache_without_network(settings: Settings) -> None:
    loader = make_loader(settings)
    loader.ingest(["ES"], START, END, client=FakeClient(FakeStore(ohlcv_raw(periods=4))))
    cached = loader.load(["ES"], START, END)
    assert cached.height == 4
    assert cached.equals(loader.repository.read_bars(["ES"], START, END))
    with pytest.raises(ValueError):
        loader.load(["ES"], START, END, schema="tbbo")
    assert isinstance(loader, HistoricalDataSource)


def test_load_incomplete_cache_falls_back_to_ingest(settings: Settings) -> None:
    loader = make_loader(settings)
    loader.ingest(
        ["ES"], START, START + timedelta(days=2), client=FakeClient(FakeStore(ohlcv_raw(periods=2)))
    )
    with pytest.raises(RuntimeError, match="FUTURESRLX_DATABENTO_API_KEY"):
        loader.load(["ES"], START, END)


def test_available_roots_unions_bar_schemas(settings: Settings) -> None:
    loader = make_loader(settings)
    loader.ingest(["ES"], START, END, client=FakeClient(FakeStore(ohlcv_raw())))
    loader.ingest(
        ["CL"],
        START,
        END,
        "ohlcv-1m",
        client=FakeClient(FakeStore(ohlcv_raw(symbol="CLF4"))),
    )
    assert loader.available_roots() == ("CL", "ES")


# ---------------------------------------------------------------------------
# Offline deterministic source
# ---------------------------------------------------------------------------


def test_offline_source_is_deterministic_and_synthetic() -> None:
    source = OfflineBarSource(("ES", "NQ"), seed=11, periods=64)
    window_start = source.start
    window_end = window_start + timedelta(days=10)
    first = source.load(["ES", "NQ"], window_start, window_end)
    second = source.load(["ES", "NQ"], window_start, window_end)
    assert source.is_synthetic is True
    assert source.term_structure_enabled is True
    assert source.available_roots() == ("ES", "NQ")
    assert first.equals(second)
    assert s.ensure_bars_schema(first).equals(first)
    # Two roots x ten daily bars x (front F1 + synthetic second month F2).
    assert first.height == 40
    assert set(first[s.ROOT_SYMBOL_COLUMN].to_list()) == {"ES", "NQ"}
    assert all("_SYN" in instrument for instrument in first[s.INSTRUMENT_ID_COLUMN].to_list())
    per_moment = first.group_by([s.ROOT_SYMBOL_COLUMN, s.TIMESTAMP_COLUMN]).agg(
        pl.col(s.EXPIRATION_COLUMN).n_unique().alias("expirations")
    )
    assert per_moment["expirations"].unique().to_list() == [2]
    assert isinstance(source, HistoricalDataSource)


def test_offline_source_intraday_intervals() -> None:
    source = OfflineBarSource(("ES",), periods=8)
    minute_bars = source.load(
        ["ES"], source.start, source.start + timedelta(hours=2), schema="ohlcv-1m"
    )
    hour_bars = source.load(
        ["ES"], source.start, source.start + timedelta(hours=6), schema="ohlcv-1h"
    )
    # Front + synthetic second month per root/timestamp.
    assert minute_bars.height == 240
    assert hour_bars.height == 12
    with pytest.raises(ValueError, match="only produces OHLCV bars"):
        source.load(["ES"], source.start, source.start + timedelta(days=1), schema="tbbo")


def test_offline_source_definitions_match_bars() -> None:
    source = OfflineBarSource(("ES", "CL"), seed=3, periods=16)
    bars = source.load(["ES", "CL"], source.start, source.start + timedelta(days=5))
    definitions = source.load_definitions(["ES", "CL"], as_of=source.start)
    assert definitions.columns == list(s.DEFINITION_COLUMNS)
    # Every emitted (front or synthetic second-month) contract has a definition.
    assert set(bars[s.INSTRUMENT_ID_COLUMN].unique().to_list()) <= set(
        definitions[s.INSTRUMENT_ID_COLUMN].to_list()
    )
    assert set(bars[s.EXPIRATION_COLUMN].unique().to_list()) <= set(
        definitions[s.EXPIRATION_COLUMN].to_list()
    )
    # Per-root multipliers come from the real instrument master.
    multipliers = {
        record[s.ROOT_SYMBOL_COLUMN]: record["multiplier"]
        for record in definitions.unique(s.ROOT_SYMBOL_COLUMN).iter_rows(named=True)
    }
    assert multipliers == {"ES": 50.0, "CL": 1_000.0}
    assert set(definitions[s.ROOT_SYMBOL_COLUMN].to_list()) == {"ES", "CL"}


def test_offline_source_validation() -> None:
    with pytest.raises(ValueError):
        OfflineBarSource(("ES",), periods=1)
    with pytest.raises(ValueError):
        OfflineBarSource(("ES",), start=datetime(2024, 1, 1))  # naive
    source = OfflineBarSource(("ES",), periods=4)
    assert source.roots == ("ES",)
    assert source.instrument_master.has("ES")
    assert synthetic_reference_price("ES") == 4_800.0
    assert synthetic_reference_price("UNKNOWN") == 1_000.0
    assert source.load_definitions(["ES"], as_of=source.start).height > 1
    with pytest.raises(TypeError):
        OfflineBarSource(("ES",), term_structure="yes")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        OfflineBarSource(("ES",), term_structure_model=object())  # type: ignore[arg-type]


def test_offline_source_legacy_single_series_mode() -> None:
    legacy = OfflineBarSource(("ES", "CL"), seed=7, periods=8, term_structure=False)
    bars = legacy.load(["ES", "CL"], legacy.start, legacy.start + timedelta(days=6))
    assert legacy.term_structure_enabled is False
    assert bars.height == 12  # two roots x six daily bars, no second month
    per_moment = bars.group_by([s.ROOT_SYMBOL_COLUMN, s.TIMESTAMP_COLUMN]).agg(
        pl.col(s.EXPIRATION_COLUMN).n_unique().alias("expirations")
    )
    assert per_moment["expirations"].unique().to_list() == [1]
    single_expiry = bars[s.EXPIRATION_COLUMN].n_unique()
    assert single_expiry == 2  # one front contract per root
