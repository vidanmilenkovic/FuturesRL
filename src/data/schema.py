"""Canonical market-data schema shared by every data adapter.

Every historical/live adapter normalises to this immutable Polars schema keyed by
``(timestamp_utc, instrument_id, root_symbol, expiration_date)``. The backtester,
feature processor and continuous-roll engine rely on these exact column names,
so they live here as a frozen contract: adapters may *add* columns, never rename
or drop these.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta
from typing import Literal

import polars as pl
from polars.datatypes import DataTypeClass

from src.core.types import InstrumentId

# -- bar schema -----------------------------------------------------------------

#: A Polars dtype is either an instantiated dtype (``pl.Float64()``) or the
#: dtype singleton class (``pl.Float64``); the canonical schemas use both forms.
type PolarsDType = pl.DataType | DataTypeClass


TIMESTAMP_COLUMN = "timestamp_utc"
INSTRUMENT_ID_COLUMN = "instrument_id"
ROOT_SYMBOL_COLUMN = "root_symbol"
EXPIRATION_COLUMN = "expiration_date"
OPEN_COLUMN = "open"
HIGH_COLUMN = "high"
LOW_COLUMN = "low"
CLOSE_COLUMN = "close"
VOLUME_COLUMN = "volume"
OPEN_INTEREST_COLUMN = "open_interest"
SETTLEMENT_COLUMN = "settlement"

#: Ordered mandatory bar columns and their canonical Polars dtypes.
BAR_COLUMNS: Mapping[str, PolarsDType] = {
    TIMESTAMP_COLUMN: pl.Datetime("us", "UTC"),
    INSTRUMENT_ID_COLUMN: pl.Utf8,
    ROOT_SYMBOL_COLUMN: pl.Utf8,
    EXPIRATION_COLUMN: pl.Date,
    OPEN_COLUMN: pl.Float64,
    HIGH_COLUMN: pl.Float64,
    LOW_COLUMN: pl.Float64,
    CLOSE_COLUMN: pl.Float64,
    VOLUME_COLUMN: pl.Float64,
    OPEN_INTEREST_COLUMN: pl.Float64,
    SETTLEMENT_COLUMN: pl.Float64,
}

#: Primary-key columns that uniquely identify a bar.
BAR_KEY_COLUMNS: tuple[str, ...] = (
    TIMESTAMP_COLUMN,
    INSTRUMENT_ID_COLUMN,
    ROOT_SYMBOL_COLUMN,
    EXPIRATION_COLUMN,
)

# -- contract definition schema -------------------------------------------------

DEFINITION_COLUMNS: Mapping[str, PolarsDType] = {
    INSTRUMENT_ID_COLUMN: pl.Utf8,
    ROOT_SYMBOL_COLUMN: pl.Utf8,
    EXPIRATION_COLUMN: pl.Date,
    "multiplier": pl.Float64,
    "tick_size": pl.Float64,
    "exchange": pl.Utf8,
    "currency": pl.Utf8,
    "activation": pl.Datetime("us", "UTC"),
}

# -- continuous-series schema ---------------------------------------------------

CONTINUOUS_COLUMNS: Mapping[str, PolarsDType] = {
    TIMESTAMP_COLUMN: pl.Datetime("us", "UTC"),
    ROOT_SYMBOL_COLUMN: pl.Utf8,
    "front_symbol": pl.Utf8,
    "front_expiration": pl.Date,
    "next_symbol": pl.Utf8,
    "next_expiration": pl.Date,
    OPEN_COLUMN: pl.Float64,
    HIGH_COLUMN: pl.Float64,
    LOW_COLUMN: pl.Float64,
    CLOSE_COLUMN: pl.Float64,
    "adjusted_close": pl.Float64,
    "panama_adjustment": pl.Float64,
    VOLUME_COLUMN: pl.Float64,
    OPEN_INTEREST_COLUMN: pl.Float64,
    "roll_flag": pl.Boolean,
    "roll_gap": pl.Float64,
}

# -- canonical feature names ----------------------------------------------------
#
# The feature processor must emit these names (it may emit additional columns
# prefixed with ``feat_``). Strategy selectors/timing overlays and the DRL
# environment are coded against these names.

RET_1D = "ret_1d"
RET_5D = "ret_5d"
RET_21D = "ret_21d"
RET_63D = "ret_63d"
RET_126D = "ret_126d"
RET_252D = "ret_252d"
VOL_20D = "vol_20d"
VOL_60D = "vol_60d"
ATR = "atr"
GARMAN_KLASS_VOL = "garman_klass_vol"
MACD = "macd"
MACD_SIGNAL = "macd_signal"
RSI = "rsi"
CCI = "cci"
ADX = "adx"
BB_UPPER = "bb_upper"
BB_LOWER = "bb_lower"
BB_WIDTH = "bb_width"
TERM_SLOPE = "term_slope"
CARRY_ANNUALIZED = "carry_annualized"
VOLUME_OI_RATIO = "volume_oi_ratio"
OFI = "ofi"
TURBULENCE = "turbulence"
KAMA = "kama"
TSMOM_21 = "tsmom_21"
TSMOM_126 = "tsmom_126"

#: Features the pipeline assumes exist on every :class:`~src.core.types.MarketState`.
REQUIRED_FEATURES: tuple[str, ...] = (
    RET_1D,
    RET_21D,
    RET_126D,
    VOL_20D,
    TERM_SLOPE,
    CARRY_ANNUALIZED,
    VOLUME_OI_RATIO,
)

#: Optional but standard FinRL-X technical features.
TECHNICAL_FEATURES: tuple[str, ...] = (
    MACD,
    MACD_SIGNAL,
    RSI,
    CCI,
    ADX,
    BB_UPPER,
    BB_LOWER,
    BB_WIDTH,
    ATR,
    GARMAN_KLASS_VOL,
    OFI,
    TURBULENCE,
    KAMA,
    TSMOM_21,
    TSMOM_126,
)


class SchemaError(ValueError):
    """Raised when a frame violates the canonical market-data schema."""


def empty_bars_frame() -> pl.DataFrame:
    """An empty frame with the canonical bar schema."""
    return pl.DataFrame(schema=BAR_COLUMNS)


def empty_definitions_frame() -> pl.DataFrame:
    """An empty frame with the canonical contract-definition schema."""
    return pl.DataFrame(schema=DEFINITION_COLUMNS)


def empty_continuous_frame() -> pl.DataFrame:
    """An empty frame with the canonical continuous-series schema."""
    return pl.DataFrame(schema=CONTINUOUS_COLUMNS)


def ensure_bars_schema(frame: pl.DataFrame, *, context: str = "bars") -> pl.DataFrame:
    """Validate and normalise a bar frame to the canonical schema.

    Returns the frame with canonical dtypes and column order; raises
    :class:`SchemaError` on missing columns or invalid values. Extra columns are
    preserved (appended after the canonical ones).
    """
    missing = [column for column in BAR_COLUMNS if column not in frame.columns]
    if missing:
        raise SchemaError(f"{context}: missing required columns {missing}")
    try:
        normalised = frame.with_columns(
            pl.col(TIMESTAMP_COLUMN).cast(pl.Datetime("us", "UTC")),
            pl.col(EXPIRATION_COLUMN).cast(pl.Date),
            pl.col(ROOT_SYMBOL_COLUMN).cast(pl.Utf8),
            pl.col(INSTRUMENT_ID_COLUMN).cast(pl.Utf8),
            *[
                pl.col(column).cast(dtype)
                for column, dtype in BAR_COLUMNS.items()
                if column
                not in (
                    TIMESTAMP_COLUMN,
                    EXPIRATION_COLUMN,
                    ROOT_SYMBOL_COLUMN,
                    INSTRUMENT_ID_COLUMN,
                )
            ],
        )
    except pl.exceptions.PolarsError as exc:  # pragma: no cover - defensive
        raise SchemaError(f"{context}: dtype normalisation failed: {exc}") from exc
    extra = [column for column in normalised.columns if column not in BAR_COLUMNS]
    normalised = normalised.select([*BAR_COLUMNS, *extra])
    if normalised.height:
        _validate_bar_values(normalised, context=context)
    return normalised


def _validate_bar_values(frame: pl.DataFrame, *, context: str) -> None:
    """Check OHLC/volume invariants on a non-empty normalised frame."""
    bad_ohlc = frame.filter(
        (pl.col(HIGH_COLUMN) < pl.col(LOW_COLUMN))
        | (pl.col(HIGH_COLUMN) < pl.col(OPEN_COLUMN))
        | (pl.col(HIGH_COLUMN) < pl.col(CLOSE_COLUMN))
        | (pl.col(LOW_COLUMN) > pl.col(OPEN_COLUMN))
        | (pl.col(LOW_COLUMN) > pl.col(CLOSE_COLUMN))
    )
    if bad_ohlc.height:
        raise SchemaError(f"{context}: {bad_ohlc.height} rows violate OHLC ordering")
    negative = frame.filter((pl.col(VOLUME_COLUMN) < 0) | (pl.col(OPEN_INTEREST_COLUMN) < 0))
    if negative.height:
        raise SchemaError(f"{context}: volume/open_interest must be non-negative")
    nulls = frame.filter(
        pl.col(TIMESTAMP_COLUMN).is_null()
        | pl.col(INSTRUMENT_ID_COLUMN).is_null()
        | pl.col(ROOT_SYMBOL_COLUMN).is_null()
    )
    if nulls.height:
        raise SchemaError(f"{context}: key columns must not be null")


def validate_roots(roots: Sequence[InstrumentId]) -> tuple[InstrumentId, ...]:
    """Normalise and validate a root list (upper-case, non-empty, unique)."""
    if not roots:
        raise ValueError("at least one futures root is required")
    normalised = tuple(sorted({root.strip().upper() for root in roots}))
    if any(not root for root in normalised):
        raise ValueError("futures roots must be non-empty strings")
    if len(normalised) != len(roots):
        raise ValueError("duplicate futures roots are not allowed")
    return normalised


# ===========================================================================
# Extended contracts added by the Phase-1 data layer.
#
# Everything below is *additive*: the canonical bar/definition/continuous
# schemas above are frozen, but the adapters also need a canonical top-of-book
# (``tbbo``) quote schema, an order-book (``mbp``) schema, the cross-asset macro
# feature names and generic schema validators used by the Parquet repository and
# the Databento/IBKR adapters.
# ===========================================================================

#: Databento schema used for daily bars unless overridden.
DEFAULT_BAR_SCHEMA = "ohlcv-1d"

# ---------------------------------------------------------------------------
# Top-of-book quote schema (Databento ``tbbo``)
# ---------------------------------------------------------------------------

BID_PRICE_COLUMN = "bid_px"
ASK_PRICE_COLUMN = "ask_px"
BID_SIZE_COLUMN = "bid_sz"
ASK_SIZE_COLUMN = "ask_sz"
BID_COUNT_COLUMN = "bid_ct"
ASK_COUNT_COLUMN = "ask_ct"

#: Ordered mandatory top-of-book columns and their canonical dtypes.
QUOTE_COLUMNS: Mapping[str, PolarsDType] = {
    TIMESTAMP_COLUMN: pl.Datetime("us", "UTC"),
    INSTRUMENT_ID_COLUMN: pl.Utf8,
    ROOT_SYMBOL_COLUMN: pl.Utf8,
    EXPIRATION_COLUMN: pl.Date,
    BID_PRICE_COLUMN: pl.Float64,
    ASK_PRICE_COLUMN: pl.Float64,
    BID_SIZE_COLUMN: pl.Float64,
    ASK_SIZE_COLUMN: pl.Float64,
    BID_COUNT_COLUMN: pl.Float64,
    ASK_COUNT_COLUMN: pl.Float64,
}

#: Primary-key columns that uniquely identify a top-of-book quote.
QUOTE_KEY_COLUMNS: tuple[str, ...] = BAR_KEY_COLUMNS

# ---------------------------------------------------------------------------
# Market-by-price order book schema (Databento ``mbp-1`` / ``mbp-10``)
# ---------------------------------------------------------------------------


def book_level_columns(levels: int) -> dict[str, PolarsDType]:
    """Column -> dtype mapping for a ``levels``-deep market-by-price frame."""
    if levels < 1:
        raise ValueError("levels must be >= 1")
    columns: dict[str, PolarsDType] = {
        TIMESTAMP_COLUMN: pl.Datetime("us", "UTC"),
        INSTRUMENT_ID_COLUMN: pl.Utf8,
        ROOT_SYMBOL_COLUMN: pl.Utf8,
        EXPIRATION_COLUMN: pl.Date,
    }
    for level in range(levels):
        columns[f"bid_px_{level:02d}"] = pl.Float64
        columns[f"bid_sz_{level:02d}"] = pl.Float64
        columns[f"ask_px_{level:02d}"] = pl.Float64
        columns[f"ask_sz_{level:02d}"] = pl.Float64
    return columns


#: Canonical ``mbp-1`` frame (timestamp + best bid/ask only).
MBP_1_COLUMNS: Mapping[str, PolarsDType] = book_level_columns(1)

#: Canonical ``mbp-10`` frame (timestamp + ten book levels per side).
MBP_10_COLUMNS: Mapping[str, PolarsDType] = book_level_columns(10)

#: Primary-key columns that uniquely identify a depth snapshot.
BOOK_KEY_COLUMNS: tuple[str, ...] = BAR_KEY_COLUMNS

# ---------------------------------------------------------------------------
# Contract-definition / continuous-series key columns
# ---------------------------------------------------------------------------

#: Primary-key columns that uniquely identify a contract definition row.
DEFINITION_KEY_COLUMNS: tuple[str, ...] = (INSTRUMENT_ID_COLUMN, EXPIRATION_COLUMN)

#: Primary-key columns that uniquely identify a continuous-series row.
CONTINUOUS_KEY_COLUMNS: tuple[str, ...] = (TIMESTAMP_COLUMN, ROOT_SYMBOL_COLUMN)

# ---------------------------------------------------------------------------
# Cross-asset macro regime features (fed by an exogenous macro frame)
# ---------------------------------------------------------------------------

MACRO_VIX_TERM_SLOPE = "macro_vix_term_slope"
MACRO_YIELD_CURVE_SLOPE = "macro_yield_curve_slope"
MACRO_DXY_RETURN_5D = "macro_dxy_return_5d"
MACRO_DXY_TREND_21D = "macro_dxy_trend_21d"

#: Cross-asset macro-regime feature names emitted by the feature processor.
MACRO_FEATURES: tuple[str, ...] = (
    MACRO_VIX_TERM_SLOPE,
    MACRO_YIELD_CURVE_SLOPE,
    MACRO_DXY_RETURN_5D,
    MACRO_DXY_TREND_21D,
)

#: Every canonical feature name the feature processor may emit.
ALL_FEATURES: tuple[str, ...] = (*REQUIRED_FEATURES, *TECHNICAL_FEATURES, *MACRO_FEATURES)

#: Canonical per-instrument features usable inside ``MarketState`` snapshots.
SNAPSHOT_FEATURES: tuple[str, ...] = (*REQUIRED_FEATURES, *TECHNICAL_FEATURES)

# ---------------------------------------------------------------------------
# Databento schema catalogue
# ---------------------------------------------------------------------------

#: Bar schemas consumed by :class:`src.data.storage.ParquetRepository`.
DATABENTO_BAR_SCHEMAS: tuple[str, ...] = ("ohlcv-1m", "ohlcv-1h", "ohlcv-1d")

#: Derived multi-timeframe bar schemas cached by the repository (resampled from
#: the Databento base schemas). They share the canonical bar schema/validator.
TIMEFRAME_BAR_SCHEMAS: tuple[str, ...] = (
    "ohlcv-5m",
    "ohlcv-15m",
    "ohlcv-30m",
    "ohlcv-4h",
    "ohlcv-1w",
)

#: Every supported canonical bar schema (vendor base + derived timeframes).
ALL_BAR_SCHEMAS: tuple[str, ...] = (*DATABENTO_BAR_SCHEMAS, *TIMEFRAME_BAR_SCHEMAS)

#: Tick/quote schemas normalised to :data:`QUOTE_COLUMNS`.
DATABENTO_QUOTE_SCHEMAS: tuple[str, ...] = ("tbbo",)

#: Market-by-price schemas normalised to :func:`book_level_columns`.
DATABENTO_BOOK_SCHEMAS: tuple[str, ...] = ("mbp-1", "mbp-10")

#: Contract-definition schema (point-in-time contract specifications).
DATABENTO_DEFINITION_SCHEMA = "definition"

#: Every schema this data layer understands.
DATABENTO_SCHEMAS: tuple[str, ...] = (
    *DATABENTO_BAR_SCHEMAS,
    *DATABENTO_QUOTE_SCHEMAS,
    *DATABENTO_BOOK_SCHEMAS,
    DATABENTO_DEFINITION_SCHEMA,
)

#: Bar interval in seconds for every supported bar schema.
BAR_SCHEMA_INTERVALS: Mapping[str, int] = {
    "ohlcv-1m": 60,
    "ohlcv-5m": 300,
    "ohlcv-15m": 900,
    "ohlcv-30m": 1_800,
    "ohlcv-1h": 3_600,
    "ohlcv-4h": 14_400,
    "ohlcv-1d": 86_400,
    "ohlcv-1w": 604_800,
}

#: Databento native continuous symbology suffixes (volume / calendar).
NATIVE_CONTINUOUS_SUFFIXES: Mapping[str, str] = {"volume": ".v.0", "calendar": ".c.0"}


def is_bar_schema(schema: str) -> bool:
    """Whether ``schema`` is one of the supported OHLCV bar schemas.

    Covers the Databento base schemas (``ohlcv-1m/1h/1d``) plus the derived
    multi-timeframe cache schemas (``ohlcv-5m/15m/30m/4h/1w``).
    """
    return schema in ALL_BAR_SCHEMAS


def bar_interval(schema: str) -> timedelta:
    """Bar duration for a bar ``schema`` (``ValueError`` for non-bar schemas)."""
    try:
        return timedelta(seconds=BAR_SCHEMA_INTERVALS[schema])
    except KeyError as exc:
        raise ValueError(
            f"{schema!r} is not a bar schema; expected one of {list(ALL_BAR_SCHEMAS)}"
        ) from exc


# ---------------------------------------------------------------------------
# Generic schema helpers
# ---------------------------------------------------------------------------

FrameValidator = Callable[[pl.DataFrame], pl.DataFrame]


def normalise_timestamp_column(
    frame: pl.DataFrame, column: str = TIMESTAMP_COLUMN, *, time_zone: str = "UTC"
) -> pl.DataFrame:
    """Return ``frame`` with ``column`` coerced to ``Datetime("us", time_zone)``.

    Accepts tz-aware/tz-naive datetime, date and integer columns (integers are
    interpreted as nanoseconds since the Unix epoch, matching Databento).
    """
    if column not in frame.columns:
        raise SchemaError(f"missing timestamp column {column!r}")
    dtype = frame.schema[column]
    if isinstance(dtype, pl.Datetime):
        expression = pl.col(column)
        if dtype.time_zone is None:
            expression = expression.dt.replace_time_zone(time_zone)
        elif dtype.time_zone != time_zone:
            expression = expression.dt.convert_time_zone(time_zone)
        expression = expression.dt.cast_time_unit("us")
    elif dtype == pl.Date:
        expression = (
            pl.col(column)
            .cast(pl.Datetime("us"))
            .dt.replace_time_zone(time_zone)
            .dt.cast_time_unit("us")
        )
    elif dtype.is_integer():
        expression = (
            pl.col(column)
            .cast(pl.Int64)
            .cast(pl.Datetime("ns"))
            .dt.replace_time_zone(time_zone)
            .dt.cast_time_unit("us")
        )
    else:  # pragma: no cover - exotic dtype, best-effort cast
        expression = pl.col(column).cast(pl.Datetime("us", time_zone))
    return frame.with_columns(expression.alias(column))


def ensure_key_schema(
    frame: pl.DataFrame,
    *,
    context: str = "frame",
    key_columns: Sequence[str] = BAR_KEY_COLUMNS,
) -> pl.DataFrame:
    """Minimal validator used for opaque schemas (``mbp-10`` & friends).

    Requires the key columns, coerces the timestamp to UTC microseconds and
    leaves every other column untouched (Databento depth frames are sparse and
    their column set is version-dependent).
    """
    missing = [column for column in key_columns if column not in frame.columns]
    if missing:
        raise SchemaError(f"{context}: missing required columns {missing}")
    normalised = normalise_timestamp_column(frame, TIMESTAMP_COLUMN)
    if EXPIRATION_COLUMN in normalised.columns:
        normalised = normalised.with_columns(pl.col(EXPIRATION_COLUMN).cast(pl.Date))
    for column in (INSTRUMENT_ID_COLUMN, ROOT_SYMBOL_COLUMN):
        if column in normalised.columns:
            normalised = normalised.with_columns(pl.col(column).cast(pl.Utf8))
    return normalised


def ensure_quote_schema(frame: pl.DataFrame, *, context: str = "quotes") -> pl.DataFrame:
    """Validate and normalise a top-of-book quote frame to :data:`QUOTE_COLUMNS`."""
    missing = [
        column
        for column in (TIMESTAMP_COLUMN, INSTRUMENT_ID_COLUMN, ROOT_SYMBOL_COLUMN)
        if column not in frame.columns
    ]
    if missing:
        raise SchemaError(f"{context}: missing required columns {missing}")
    normalised = normalise_timestamp_column(frame, TIMESTAMP_COLUMN)
    if EXPIRATION_COLUMN not in normalised.columns:
        normalised = normalised.with_columns(pl.lit(None, dtype=pl.Date).alias(EXPIRATION_COLUMN))
    normalised = normalised.with_columns(
        pl.col(INSTRUMENT_ID_COLUMN).cast(pl.Utf8),
        pl.col(ROOT_SYMBOL_COLUMN).cast(pl.Utf8),
        pl.col(EXPIRATION_COLUMN).cast(pl.Date),
    )
    for column in (
        BID_PRICE_COLUMN,
        ASK_PRICE_COLUMN,
        BID_SIZE_COLUMN,
        ASK_SIZE_COLUMN,
        BID_COUNT_COLUMN,
        ASK_COUNT_COLUMN,
    ):
        if column not in normalised.columns:
            normalised = normalised.with_columns(pl.lit(None, dtype=pl.Float64).alias(column))
        else:
            normalised = normalised.with_columns(pl.col(column).cast(pl.Float64))
    extra = [column for column in normalised.columns if column not in QUOTE_COLUMNS]
    normalised = normalised.select([*QUOTE_COLUMNS, *extra])
    if normalised.height:
        if normalised.filter(pl.col(BID_PRICE_COLUMN).is_null()).height:
            raise SchemaError(f"{context}: bid_px must not be null")
        if normalised.filter(pl.col(ASK_PRICE_COLUMN).is_null()).height:
            raise SchemaError(f"{context}: ask_px must not be null")
        crossed = normalised.filter(pl.col(BID_PRICE_COLUMN) > pl.col(ASK_PRICE_COLUMN))
        if crossed.height:
            raise SchemaError(f"{context}: {crossed.height} quotes have bid_px > ask_px")
    return normalised


def ensure_definitions_schema(frame: pl.DataFrame, *, context: str = "definitions") -> pl.DataFrame:
    """Validate and normalise a contract-definition frame."""
    missing = [
        column
        for column in (INSTRUMENT_ID_COLUMN, ROOT_SYMBOL_COLUMN, EXPIRATION_COLUMN)
        if column not in frame.columns
    ]
    if missing:
        raise SchemaError(f"{context}: missing required columns {missing}")
    normalised = frame.with_columns(
        pl.col(INSTRUMENT_ID_COLUMN).cast(pl.Utf8),
        pl.col(ROOT_SYMBOL_COLUMN).cast(pl.Utf8),
        pl.col(EXPIRATION_COLUMN).cast(pl.Date),
    )
    if "activation" in normalised.columns:
        # Activation timestamps may arrive as Databento nanosecond integers; the
        # generic dtype cast below would mis-read them as microseconds.
        normalised = normalise_timestamp_column(normalised, "activation")
    for column, dtype in DEFINITION_COLUMNS.items():
        if column in (INSTRUMENT_ID_COLUMN, ROOT_SYMBOL_COLUMN, EXPIRATION_COLUMN):
            continue
        if column not in normalised.columns:
            normalised = normalised.with_columns(pl.lit(None, dtype=dtype).alias(column))
        else:
            normalised = normalised.with_columns(pl.col(column).cast(dtype))
    extra = [column for column in normalised.columns if column not in DEFINITION_COLUMNS]
    normalised = normalised.select([*DEFINITION_COLUMNS, *extra])
    if normalised.height:
        nulls = normalised.filter(
            pl.col(INSTRUMENT_ID_COLUMN).is_null() | pl.col(EXPIRATION_COLUMN).is_null()
        )
        if nulls.height:
            raise SchemaError(f"{context}: definition key columns must not be null")
    return normalised


def ensure_continuous_schema(frame: pl.DataFrame, *, context: str = "continuous") -> pl.DataFrame:
    """Validate and normalise a continuous-series frame returned by the roller."""
    required = (
        TIMESTAMP_COLUMN,
        ROOT_SYMBOL_COLUMN,
        CLOSE_COLUMN,
        "adjusted_close",
    )
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise SchemaError(f"{context}: missing required columns {missing}")
    normalised = normalise_timestamp_column(frame, TIMESTAMP_COLUMN)
    normalised = normalised.with_columns(pl.col(ROOT_SYMBOL_COLUMN).cast(pl.Utf8))
    for column, dtype in CONTINUOUS_COLUMNS.items():
        if column == TIMESTAMP_COLUMN:
            continue
        if column in normalised.columns:
            normalised = normalised.with_columns(pl.col(column).cast(dtype))
    extra = [column for column in normalised.columns if column not in CONTINUOUS_COLUMNS]
    return normalised.select([*CONTINUOUS_COLUMNS, *extra])


def dedupe_frame(
    frame: pl.DataFrame,
    key_columns: Sequence[str] = BAR_KEY_COLUMNS,
    *,
    keep: Literal["first", "last"] = "last",
    context: str = "frame",
) -> pl.DataFrame:
    """Sort by ``key_columns`` and drop duplicate keys (``keep`` row wins)."""
    missing = [column for column in key_columns if column not in frame.columns]
    if missing:
        raise SchemaError(f"{context}: cannot dedupe, missing key columns {missing}")
    ordered = frame.sort(list(key_columns))
    return ordered.unique(subset=list(key_columns), keep=keep, maintain_order=True)


def validator_for_schema(schema: str) -> FrameValidator:
    """Return the canonical frame validator for a repository ``schema`` name."""
    if schema in ALL_BAR_SCHEMAS or schema in {"bars", "ohlcv"}:
        return ensure_bars_schema
    if schema == DATABENTO_DEFINITION_SCHEMA:
        return ensure_definitions_schema
    if schema in DATABENTO_QUOTE_SCHEMAS or schema == "quotes":
        return ensure_quote_schema
    if schema == "continuous":
        return ensure_continuous_schema
    return ensure_key_schema


def filter_time_window(
    frame: pl.DataFrame,
    start: datetime | None = None,
    end: datetime | None = None,
    *,
    timestamp_column: str = TIMESTAMP_COLUMN,
) -> pl.DataFrame:
    """Filter ``frame`` to the half-open window ``[start, end)`` (UTC)."""
    if timestamp_column not in frame.columns:
        raise SchemaError(f"missing timestamp column {timestamp_column!r}")
    expression = pl.col(timestamp_column)
    filtered = frame
    if start is not None:
        if start.tzinfo is None:
            raise ValueError("start must be timezone-aware (UTC expected)")
        filtered = filtered.filter(expression >= pl.lit(start))
        expression = pl.col(timestamp_column)
    if end is not None:
        if end.tzinfo is None:
            raise ValueError("end must be timezone-aware (UTC expected)")
        filtered = filtered.filter(expression < pl.lit(end))
    return filtered
