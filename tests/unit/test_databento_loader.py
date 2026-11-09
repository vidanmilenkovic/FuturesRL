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
from pydantic import ValidationError
from src.core.interfaces import HistoricalDataSource
from src.data import schema as s
from src.data.databento_loader import (
    CostEstimate,
    DatabentoBudgetExceededError,
    DatabentoCostEstimateError,
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
    with pytest.raises(RuntimeError, match="FUTURESRL_DATABENTO_API_KEY"):
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
    with pytest.raises(RuntimeError, match="FUTURESRL_DATABENTO_API_KEY"):
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
    with pytest.raises(RuntimeError, match="FUTURESRL_DATABENTO_API_KEY"):
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


# ---------------------------------------------------------------------------
# Phase-11B: cost estimation + hard per-job budget guard
# ---------------------------------------------------------------------------


class FakeMetadata:
    """Records ``get_cost``/``get_billable_size`` calls without any vendor network."""

    def __init__(
        self,
        cost: float = 0.42,
        billable_size: int = 123_456,
        *,
        cost_error: Exception | None = None,
        size_error: Exception | None = None,
    ) -> None:
        self.cost = cost
        self.billable_size = billable_size
        self.cost_error = cost_error
        self.size_error = size_error
        self.cost_calls: list[dict[str, Any]] = []
        self.size_calls: list[dict[str, Any]] = []

    def get_cost(self, **kwargs: Any) -> float:
        self.cost_calls.append(kwargs)
        if self.cost_error is not None:
            raise self.cost_error
        return self.cost

    def get_billable_size(self, **kwargs: Any) -> int:
        self.size_calls.append(kwargs)
        if self.size_error is not None:
            raise self.size_error
        return self.billable_size


def metadata_client(metadata: FakeMetadata | None = None, *, periods: int = 3) -> Any:
    client = FakeClient(FakeStore(ohlcv_raw(periods=periods)))
    client.metadata = metadata if metadata is not None else FakeMetadata()  # type: ignore[attr-defined]
    return client


def test_historical_client_supports_new_key_and_legacy_api_key(settings: Settings) -> None:
    loader = make_loader(settings, api_key="fake")
    calls: list[dict[str, object]] = []

    class KeyOnly:
        def __init__(self, *, key: str) -> None:
            calls.append({"key": key})

    class LegacyOnly:
        def __init__(self, *, api_key: str) -> None:
            calls.append({"api_key": api_key})

    loader._import_sdk = lambda: type("SDK", (), {"Historical": KeyOnly})()  # type: ignore[method-assign]
    loader._client()
    loader._import_sdk = lambda: type("SDK", (), {"Historical": LegacyOnly})()  # type: ignore[method-assign]
    loader._client()
    assert calls == [{"key": "fake"}, {"api_key": "fake"}]


def test_estimate_cost_supports_metadata_without_unit_keyword(settings: Settings) -> None:
    loader = make_loader(settings, api_key="fake")

    class ModernMetadata(FakeMetadata):
        def get_cost(self, **kwargs: Any) -> float:
            if "unit" in kwargs:
                raise TypeError("Historical.metadata.get_cost() got an unexpected keyword 'unit'")
            return super().get_cost(**kwargs)

    metadata = ModernMetadata(cost=0.31)
    client = metadata_client(metadata)
    estimate = loader.estimate_cost_usd(["ES"], START, END, client=client)
    assert estimate.estimated_cost_usd == pytest.approx(0.31)
    assert "unit" not in metadata.cost_calls[0]


def test_estimate_cost_uses_metadata_before_any_download(settings: Settings) -> None:
    loader = make_loader(settings, api_key="fake")
    metadata = FakeMetadata(cost=0.72, billable_size=2_500_000)
    client = FakeClient(FakeStore(ohlcv_raw()))
    client.metadata = metadata  # type: ignore[attr-defined]
    estimate = loader.estimate_cost_usd(["es"], START, END, client=client)
    assert isinstance(estimate, CostEstimate)
    assert estimate.roots == ("ES",)
    assert estimate.schema == s.DEFAULT_BAR_SCHEMA
    assert estimate.estimated_cost_usd == pytest.approx(0.72)
    assert estimate.billable_size_bytes == 2_500_000
    assert estimate.billable_size_megabytes == pytest.approx(2.5)
    assert estimate.source == "databento"
    assert estimate.currency == "USD"
    assert estimate.dataset == settings.databento_dataset
    assert estimate.details is not None and estimate.details["metadata_available"] is True
    cost_call = metadata.cost_calls[0]
    assert cost_call["dataset"] == settings.databento_dataset
    assert cost_call["schema"] == s.DEFAULT_BAR_SCHEMA
    assert cost_call["symbols"] == ["ES.FUT"]
    assert cost_call["start"] == START and cost_call["end"] == END
    assert cost_call["unit"] == "usd"
    assert metadata.size_calls and metadata.size_calls[0]["symbols"] == ["ES.FUT"]
    assert client.timeseries.calls == []  # estimation never downloads
    payload = estimate.to_dict()
    assert payload["roots"] == ["ES"]
    assert payload["start"] == START.isoformat()


def test_estimate_cost_empty_roots_rejected(settings: Settings) -> None:
    loader = make_loader(settings, api_key="fake")
    with pytest.raises(ValueError, match="roots must not be empty"):
        loader.estimate_cost_usd([], START, END, client=metadata_client())


def test_estimate_cost_rejects_bad_window_and_schema(settings: Settings) -> None:
    loader = make_loader(settings, api_key="fake")
    with pytest.raises(ValueError, match="start"):
        loader.estimate_cost_usd(["ES"], END, START, client=metadata_client())
    with pytest.raises(ValueError, match="unsupported Databento schema"):
        loader.estimate_cost_usd(["ES"], START, END, "bogus", client=metadata_client())


def test_estimate_cost_without_api_key_is_actionable(settings: Settings) -> None:
    loader = make_loader(settings)
    with pytest.raises(RuntimeError, match="FUTURESRL_DATABENTO_API_KEY"):
        loader.estimate_cost_usd(["ES"], START, END)


def test_estimate_cost_missing_metadata_methods_falls_back_to_zero(
    settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    loader = make_loader(settings, api_key="fake")
    client = FakeClient(FakeStore(ohlcv_raw()))
    with caplog.at_level(logging.WARNING, logger="src.data.databento_loader"):
        estimate = loader.estimate_cost_usd(["ES"], START, END, client=client)
    assert estimate.estimated_cost_usd == 0.0
    assert estimate.billable_size_bytes == 0
    assert estimate.details is not None and estimate.details["metadata_available"] is False
    assert client.timeseries.calls == []
    assert any("no metadata.get_cost" in record.message for record in caplog.records)


def test_estimate_cost_metadata_failure_is_actionable(settings: Settings) -> None:
    loader = make_loader(settings, api_key="fake")
    client = metadata_client(FakeMetadata(cost_error=RuntimeError("vendor 503")))
    with pytest.raises(DatabentoCostEstimateError, match=r"metadata\.get_cost failed"):
        loader.estimate_cost_usd(["ES"], START, END, client=client)
    assert client.timeseries.calls == []


def test_ingest_refuses_over_budget_without_download(
    settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    loader = make_loader(settings, api_key="fake")
    metadata = FakeMetadata(cost=5.0, billable_size=10_000)
    client = metadata_client(metadata)
    with (
        caplog.at_level(logging.WARNING, logger="src.data.databento_loader"),
        pytest.raises(DatabentoBudgetExceededError) as excinfo,
    ):
        loader.ingest(["ES"], START, END, client=client)
    error = excinfo.value
    assert error.estimated_cost_usd == pytest.approx(5.0)
    assert error.max_cost_usd == pytest.approx(2.0)
    assert error.roots == ("ES",)
    assert error.schema == s.DEFAULT_BAR_SCHEMA
    assert "FUTURESRL_DATABENTO_MAX_COST_USD_PER_JOB" in str(error)
    assert "allow_over_budget=True" in str(error)
    assert metadata.cost_calls  # estimated first
    assert client.timeseries.calls == []  # never downloaded on refusal


def test_ingest_operator_override_downloads_and_logs(
    settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    loader = make_loader(settings, api_key="fake")
    metadata = FakeMetadata(cost=5.0)
    client = metadata_client(metadata)
    with caplog.at_level(logging.WARNING, logger="src.data.databento_loader"):
        frame = loader.ingest(["ES"], START, END, client=client, allow_over_budget=True)
    assert frame.height == 3
    assert len(client.timeseries.calls) == 1
    assert any("budget override" in record.message for record in caplog.records)
    assert metadata.cost_calls  # estimate still ran before download


def test_ingest_explicit_max_cost_override(settings: Settings) -> None:
    loader = make_loader(settings, api_key="fake")
    client = metadata_client(FakeMetadata(cost=5.0))
    frame = loader.ingest(["ES"], START, END, client=client, max_cost_usd=10.0)
    assert frame.height == 3
    assert len(client.timeseries.calls) == 1


def test_ingest_under_budget_proceeds_normally(settings: Settings) -> None:
    loader = make_loader(settings, api_key="fake")
    metadata = FakeMetadata(cost=0.5)
    client = metadata_client(metadata)
    frame = loader.ingest(["ES"], START, END, client=client)
    assert frame.height == 3
    assert len(metadata.cost_calls) == 1
    assert len(client.timeseries.calls) == 1


def test_ingest_negative_max_cost_rejected(settings: Settings) -> None:
    loader = make_loader(settings, api_key="fake")
    with pytest.raises(ValueError, match="max_cost_usd must be >= 0"):
        loader.ingest(["ES"], START, END, client=metadata_client(), max_cost_usd=-1.0)


def test_ingest_uses_configured_cap(settings: Settings) -> None:
    configured = settings.model_copy(update={"databento_max_cost_usd_per_job": 0.25})
    loader = DatabentoLoader(
        configured,
        repository=ParquetRepository(configured),
        api_key="fake",
    )
    client = metadata_client(FakeMetadata(cost=1.0))
    with pytest.raises(DatabentoBudgetExceededError) as excinfo:
        loader.ingest(["ES"], START, END, client=client)
    assert excinfo.value.max_cost_usd == pytest.approx(0.25)
    assert client.timeseries.calls == []


def test_budget_setting_default_and_validation(settings: Settings) -> None:
    assert Settings(_env_file=None).databento_max_cost_usd_per_job == 2.0
    with pytest.raises(ValidationError):
        Settings(_env_file=None, databento_max_cost_usd_per_job=-0.01)
    configured = settings.model_copy(update={"databento_max_cost_usd_per_job": 0.0})
    loader = DatabentoLoader(
        configured,
        repository=ParquetRepository(configured),
        api_key="fake",
    )
    with pytest.raises(DatabentoBudgetExceededError):
        loader.ingest(["ES"], START, END, client=metadata_client(FakeMetadata(cost=0.01)))


def test_ingest_estimates_before_download_in_order(settings: Settings) -> None:
    order: list[str] = []

    class OrderedMetadata(FakeMetadata):
        def get_cost(self, **kwargs: Any) -> float:
            order.append("metadata.get_cost")
            return super().get_cost(**kwargs)

        def get_billable_size(self, **kwargs: Any) -> int:
            order.append("metadata.get_billable_size")
            return super().get_billable_size(**kwargs)

    class OrderedTimeseries(FakeTimeseries):
        def get_range(self, **kwargs: Any) -> FakeStore:
            order.append("timeseries.get_range")
            return super().get_range(**kwargs)

    loader = make_loader(settings, api_key="fake")
    client = FakeClient()
    client.timeseries = OrderedTimeseries(FakeStore(ohlcv_raw(periods=2)))
    client.metadata = OrderedMetadata(cost=0.5)  # type: ignore[attr-defined]
    loader.ingest(["ES"], START, END, client=client)
    assert order == ["metadata.get_cost", "metadata.get_billable_size", "timeseries.get_range"]


def test_ingest_records_databento_provenance_sidecar(settings: Settings) -> None:
    import json as _json

    loader = make_loader(settings, api_key="fake")
    loader.ingest(["ES"], START, END, client=metadata_client(FakeMetadata(cost=0.5)))
    sidecar = loader.repository.source_sidecar_path("ES", s.DEFAULT_BAR_SCHEMA)
    assert sidecar.is_file()
    payload = _json.loads(sidecar.read_text())
    assert payload["root"] == "ES"
    assert payload["source"] == "databento"
    assert payload["schema"] == s.DEFAULT_BAR_SCHEMA
    inventory = {row.root: row for row in loader.repository.inventory()}
    assert inventory["ES"].source == "databento"
    assert inventory["ES"].cached_bars == 3


def test_ingest_provenance_failure_does_not_fail_ingest(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    loader = make_loader(settings, api_key="fake")

    def boom(*args: object, **kwargs: object) -> None:
        raise OSError("sidecar disk full")

    monkeypatch.setattr(loader.repository, "record_source", boom)
    frame = loader.ingest(["ES"], START, END, client=metadata_client(FakeMetadata(cost=0.5)))
    assert frame.height == 3  # persisted data still returned; provenance is best effort


def test_dbnstore_dataframe_index_is_restored_as_timestamp() -> None:
    """Databento 0.89 returns ts_event as the pandas index, not a column."""
    import pandas as pd
    from src.data.databento_loader import _as_polars

    indexed = pd.DataFrame(
        {
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [10],
            "symbol": ["ES"],
        },
        index=pd.Index([pd.Timestamp(START)], name="ts_event"),
    )

    class IndexedStore:
        def to_df(self) -> Any:
            return indexed

    frame = _as_polars(IndexedStore())
    assert "ts_event" in frame.columns
    assert frame["ts_event"].to_list() == [START]


def test_large_ingest_is_split_into_persisted_windows(settings: Settings) -> None:
    loader = make_loader(settings, api_key="fake")
    metadata = FakeMetadata(cost=0.5, billable_size=200_000_000)
    client = metadata_client(metadata)
    end = START + timedelta(days=400)
    frame = loader.ingest(["ES"], START, end, client=client)
    assert frame.height > 0
    assert len(client.timeseries.calls) == 3
    assert [call["start"].date() for call in client.timeseries.calls] == [
        START.date(),
        (START + timedelta(days=180)).date(),
        (START + timedelta(days=360)).date(),
    ]


def test_parent_futures_request_drops_spread_and_option_rows(settings: Settings) -> None:
    loader = make_loader(settings, api_key="fake")
    raw = pl.DataFrame(
        {
            "ts_event": [START, START],
            "raw_symbol": ["ESZ6", "ESZ6-ESH7"],
            "open": [5_000.0, 5_000.0],
            "high": [5_001.0, 5_001.0],
            "low": [4_999.0, 4_999.0],
            "close": [5_000.5, 5_000.5],
            "volume": [10, 10],
        }
    )
    frame = loader._normalise_bars(raw, roots=("ES",), reference=START)
    assert frame.height == 1
    assert frame["instrument_id"].to_list() == ["ESZ6"]
    assert frame["root_symbol"].to_list() == ["ES"]
