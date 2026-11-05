"""Look-ahead-safe bar resampling for the canonical futures schema.

The resampler turns finer canonical bar frames (e.g. ``ohlcv-1m``) into coarser
timeframes from :mod:`src.data.timeframes` (5m/15m/30m/1h/4h/1d/1w). It is used by
:meth:`src.data.storage.ParquetRepository.read_timeframe` to auto-populate a
missing target cache from the finest available base partition.

Look-ahead safety
-----------------
Windows use Polars ``group_by_dynamic`` with ``label="right"`` and
``closed="right"`` in UTC: a resampled bar is stamped at the **right boundary**
(close time) of its window, and contains only source bars whose timestamps are
``<= label``. Consumers therefore never see a bar before it would be available
in real time. Equivalently, every source bar timestamp is at or before its
resampled bar label.

Aggregation contract
--------------------
Grouped by ``(instrument_id, root_symbol, expiration_date)``:

* ``open`` = first, ``high`` = max, ``low`` = min, ``close`` = last
* ``volume`` = sum, ``open_interest`` = last, ``settlement`` = last
* ``vwap`` = volume-weighted price (input ``vwap`` column when present, else
  ``close``); falls back to the window's last ``close`` when total volume is 0

The output keeps the canonical bar schema/column order (plus the ``vwap`` extra
column) and is sorted by the canonical bar primary key.
"""

from __future__ import annotations

import logging
from typing import cast

import polars as pl

from src.data import schema as s
from src.data.timeframes import TimeframeSpec, get_timeframe, timeframe_seconds

__all__ = ["VWAP_COLUMN", "resample_bars"]

logger = logging.getLogger(__name__)

#: Extra output column carrying the volume-weighted average price.
VWAP_COLUMN = "vwap"

#: Columns that identify an independent contract series.
GROUP_COLUMNS: tuple[str, ...] = (
    s.INSTRUMENT_ID_COLUMN,
    s.ROOT_SYMBOL_COLUMN,
    s.EXPIRATION_COLUMN,
)


def _empty_resampled_frame() -> pl.DataFrame:
    """Empty frame with the canonical bar schema plus the ``vwap`` column."""
    return pl.DataFrame(schema={**s.BAR_COLUMNS, VWAP_COLUMN: pl.Float64})


def _infer_source_seconds(frame: pl.DataFrame) -> int | None:
    """Infer the source bar duration (seconds) from unique timestamp spacing."""
    stamps = frame.select(pl.col(s.TIMESTAMP_COLUMN).unique().sort()).to_series()
    if stamps.len() < 2:
        return None
    differences = stamps.diff().drop_nulls().dt.total_seconds()
    positive = differences.filter(differences > 0)
    if positive.is_empty():
        return None
    median_value = positive.median()
    median = float(cast(float, median_value)) if median_value is not None else 0.0
    if median <= 0.0:
        return None
    return round(median)


def _resolve_source_seconds(
    frame: pl.DataFrame,
    source_timeframe: str | TimeframeSpec | None,
) -> int | None:
    """Source bar duration in seconds (explicit timeframe or inferred)."""
    if source_timeframe is not None:
        return timeframe_seconds(source_timeframe)
    return _infer_source_seconds(frame)


def _validate_target_coarser(target: TimeframeSpec, source_seconds: int | None) -> None:
    """Reject targets finer than the source bars (resampling can only coarsen)."""
    if source_seconds is None:
        return
    if target.seconds < source_seconds:
        raise ValueError(
            f"cannot resample to {target.name!r}: target timeframe is finer than the "
            f"source ({source_seconds}s); resampling can only coarsen bars"
        )


def resample_bars(
    df: pl.DataFrame,
    target_timeframe: str | TimeframeSpec,
    *,
    source_timeframe: str | TimeframeSpec | None = None,
) -> pl.DataFrame:
    """Resample canonical bars to ``target_timeframe`` without look-ahead.

    Parameters
    ----------
    df:
        Canonical bar frame (``src.data.schema.BAR_COLUMNS``; extra columns are
        tolerated, an input ``vwap`` column is used as the vwap weight source).
    target_timeframe:
        Canonical timeframe name (``"5m"`` ... ``"1w"``) or :class:`TimeframeSpec`.
    source_timeframe:
        Optional explicit source timeframe; when omitted it is inferred from the
        median timestamp spacing (used only to reject finer targets).

    Raises
    ------
    ValueError
        Unknown target timeframe or a target finer than the source bars.
    SchemaError
        The input is not a canonical bar frame.
    """
    if not isinstance(df, pl.DataFrame):
        raise TypeError(f"df must be a polars DataFrame, got {type(df)!r}")
    target = get_timeframe(target_timeframe)
    if df.is_empty():
        _validate_target_coarser(
            target,
            timeframe_seconds(source_timeframe) if source_timeframe is not None else None,
        )
        return _empty_resampled_frame()
    bars = s.ensure_bars_schema(df, context=f"resample_bars[{target.name}]")
    source_seconds = _resolve_source_seconds(bars, source_timeframe)
    _validate_target_coarser(target, source_seconds)

    ordered = bars.sort([*GROUP_COLUMNS, s.TIMESTAMP_COLUMN])
    vwap_source = VWAP_COLUMN if VWAP_COLUMN in ordered.columns else s.CLOSE_COLUMN
    vwap_expression = (
        pl.when(pl.col(s.VOLUME_COLUMN).sum() > 0)
        .then((pl.col(vwap_source) * pl.col(s.VOLUME_COLUMN)).sum() / pl.col(s.VOLUME_COLUMN).sum())
        .otherwise(pl.col(s.CLOSE_COLUMN).last())
        .alias(VWAP_COLUMN)
    )
    grouped = ordered.group_by_dynamic(
        index_column=s.TIMESTAMP_COLUMN,
        every=target.every,
        period=target.every,
        label="right",
        closed="right",
        group_by=list(GROUP_COLUMNS),
    ).agg(
        pl.col(s.OPEN_COLUMN).first().alias(s.OPEN_COLUMN),
        pl.col(s.HIGH_COLUMN).max().alias(s.HIGH_COLUMN),
        pl.col(s.LOW_COLUMN).min().alias(s.LOW_COLUMN),
        pl.col(s.CLOSE_COLUMN).last().alias(s.CLOSE_COLUMN),
        pl.col(s.VOLUME_COLUMN).sum().alias(s.VOLUME_COLUMN),
        pl.col(s.OPEN_INTEREST_COLUMN).last().alias(s.OPEN_INTEREST_COLUMN),
        pl.col(s.SETTLEMENT_COLUMN).last().alias(s.SETTLEMENT_COLUMN),
        vwap_expression,
    )
    selected = grouped.select([*s.BAR_COLUMNS, VWAP_COLUMN]).with_columns(
        pl.col(s.TIMESTAMP_COLUMN).cast(pl.Datetime("us", "UTC")),
        pl.col(VWAP_COLUMN).cast(pl.Float64),
    )
    validated = s.ensure_bars_schema(selected, context=f"resample_bars[{target.name}]")
    return validated.sort(list(s.BAR_KEY_COLUMNS))
