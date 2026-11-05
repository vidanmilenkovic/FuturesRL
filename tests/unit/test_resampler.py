"""Unit tests for the look-ahead-safe canonical bar resampler (fake frames only)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import polars as pl
import pytest
from src.data import schema as s
from src.data.resampler import VWAP_COLUMN, resample_bars
from src.data.timeframes import supported_timeframes

#: A fixed UTC minute-bar anchor (09:31) used by the deterministic fixtures.
ANCHOR = datetime(2024, 1, 2, 9, 31, tzinfo=UTC)
EXPIRATION = date(2024, 3, 15)


def minute_bars(
    periods: int = 12,
    *,
    root: str = "ES",
    instrument: str = "ES_FUT",
    expiration: date = EXPIRATION,
) -> pl.DataFrame:
    """Deterministic 1-minute canonical bars starting at ``ANCHOR``."""
    return pl.DataFrame(
        [
            {
                s.TIMESTAMP_COLUMN: ANCHOR + timedelta(minutes=index),
                s.INSTRUMENT_ID_COLUMN: instrument,
                s.ROOT_SYMBOL_COLUMN: root,
                s.EXPIRATION_COLUMN: expiration,
                s.OPEN_COLUMN: 100.0 + index,
                s.HIGH_COLUMN: 101.0 + index,
                s.LOW_COLUMN: 99.0 + index,
                s.CLOSE_COLUMN: 100.5 + index,
                s.VOLUME_COLUMN: float(index + 1),
                s.OPEN_INTEREST_COLUMN: 1_000.0 + index,
                s.SETTLEMENT_COLUMN: 100.0 + index,
            }
            for index in range(periods)
        ]
    )


def expected_window(indices: list[int], *, root: str = "ES", instrument: str = "ES_FUT") -> dict:
    """Expected canonical resampled row for the source rows at ``indices``."""
    rows = minute_bars(periods=max(indices) + 1, root=root, instrument=instrument)
    selected = rows[indices]
    volume_weights = selected[s.VOLUME_COLUMN].to_numpy()
    closes = selected[s.CLOSE_COLUMN].to_numpy()
    total_volume = float(volume_weights.sum())
    vwap = (
        float((closes * volume_weights).sum() / total_volume)
        if total_volume > 0.0
        else float(selected[s.CLOSE_COLUMN][-1])
    )
    return {
        s.OPEN_COLUMN: float(selected[s.OPEN_COLUMN][0]),
        s.HIGH_COLUMN: float(selected[s.HIGH_COLUMN].max()),
        s.LOW_COLUMN: float(selected[s.LOW_COLUMN].min()),
        s.CLOSE_COLUMN: float(selected[s.CLOSE_COLUMN][-1]),
        s.VOLUME_COLUMN: total_volume,
        s.OPEN_INTEREST_COLUMN: float(selected[s.OPEN_INTEREST_COLUMN][-1]),
        s.SETTLEMENT_COLUMN: float(selected[s.SETTLEMENT_COLUMN][-1]),
        VWAP_COLUMN: vwap,
    }


def assert_row(row: dict, expected: dict) -> None:
    assert set(expected) <= set(row)
    for column, value in expected.items():
        assert row[column] == pytest.approx(value), column


@pytest.mark.parametrize(
    ("target", "window_indices", "labels"),
    [
        (
            "5m",
            [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11]],
            [
                datetime(2024, 1, 2, 9, 35, tzinfo=UTC),
                datetime(2024, 1, 2, 9, 40, tzinfo=UTC),
                datetime(2024, 1, 2, 9, 45, tzinfo=UTC),
            ],
        ),
        ("15m", [list(range(12))], [datetime(2024, 1, 2, 9, 45, tzinfo=UTC)]),
        ("30m", [list(range(12))], [datetime(2024, 1, 2, 10, 0, tzinfo=UTC)]),
        ("1h", [list(range(12))], [datetime(2024, 1, 2, 10, 0, tzinfo=UTC)]),
        ("1d", [list(range(12))], [datetime(2024, 1, 3, tzinfo=UTC)]),
        ("1w", [list(range(12))], [datetime(2024, 1, 8, tzinfo=UTC)]),
    ],
)
def test_resample_ohlcv_volume_vwap_and_right_labeled_windows(
    target: str, window_indices: list[list[int]], labels: list[datetime]
) -> None:
    source = minute_bars()
    result = resample_bars(source, target)
    assert result.height == len(window_indices)
    assert result[s.TIMESTAMP_COLUMN].to_list() == labels
    for index, indices in enumerate(window_indices):
        row = result.row(index, named=True)
        assert_row(row, expected_window(list(indices)))
    assert result.columns == [*s.BAR_COLUMNS, VWAP_COLUMN]


def test_resample_labels_are_right_boundaries_with_no_lookahead() -> None:
    source = minute_bars()
    result = resample_bars(source, "5m")
    labels = result[s.TIMESTAMP_COLUMN].to_list()
    assert labels == [
        datetime(2024, 1, 2, 9, 35, tzinfo=UTC),
        datetime(2024, 1, 2, 9, 40, tzinfo=UTC),
        datetime(2024, 1, 2, 9, 45, tzinfo=UTC),
    ]
    # Every source bar belongs to exactly one window and is never later than its
    # resampled bar label (label == window close time -> no look-ahead).
    for timestamp in source[s.TIMESTAMP_COLUMN].to_list():
        matching = [label for label in labels if label >= timestamp]
        assert matching
        label = min(matching)
        assert (label - timestamp).total_seconds() < 300
        assert (label - timestamp).total_seconds() >= 0


def test_resample_groups_by_contract_keys() -> None:
    es = minute_bars(periods=6, root="ES", instrument="ES_FUT", expiration=EXPIRATION)
    nq = minute_bars(
        periods=6,
        root="NQ",
        instrument="NQ_FUT",
        expiration=date(2024, 3, 21),
    )
    # A third series: same root/instrument but the next expiration month.
    es_next = minute_bars(
        periods=6,
        root="ES",
        instrument="ES_FUT",
        expiration=date(2024, 6, 21),
    )
    combined = pl.concat([es, nq, es_next], how="vertical_relaxed")
    result = resample_bars(combined, "5m")
    assert result.height == 6  # 3 contracts x 2 windows
    keys = result.select(
        [s.INSTRUMENT_ID_COLUMN, s.ROOT_SYMBOL_COLUMN, s.EXPIRATION_COLUMN]
    ).unique()
    assert keys.height == 3
    assert result.select(s.TIMESTAMP_COLUMN).unique().height == 2


def test_resample_is_sorted_by_canonical_key() -> None:
    unsorted = minute_bars(periods=6).reverse()
    result = resample_bars(unsorted, "5m")
    assert result.equals(result.sort(list(s.BAR_KEY_COLUMNS)))


def test_resample_vwap_falls_back_to_close_when_volume_is_zero() -> None:
    source = minute_bars(periods=3).with_columns(pl.lit(0.0).alias(s.VOLUME_COLUMN))
    row = resample_bars(source, "5m").row(0, named=True)
    assert row[VWAP_COLUMN] == pytest.approx(102.5)  # last close
    assert row["volume"] == 0.0


def test_resample_uses_existing_vwap_column_when_available() -> None:
    source = minute_bars(periods=3).with_columns(
        pl.Series(VWAP_COLUMN, [10.0, 20.0, 30.0]),
        pl.Series(s.VOLUME_COLUMN, [1.0, 3.0, 0.0]),
    )
    row = resample_bars(source, "5m").row(0, named=True)
    assert row[VWAP_COLUMN] == pytest.approx((10.0 * 1 + 20.0 * 3) / 4.0)


def test_resample_empty_frame_returns_canonical_empty_frame() -> None:
    result = resample_bars(s.empty_bars_frame(), "15m")
    assert result.height == 0
    assert result.columns == [*s.BAR_COLUMNS, VWAP_COLUMN]
    assert result.schema[s.TIMESTAMP_COLUMN] == pl.Datetime("us", "UTC")


def test_resample_single_bar_minimal_frame() -> None:
    result = resample_bars(minute_bars(periods=1), "1h")
    assert result.height == 1
    row = result.row(0, named=True)
    assert row[s.TIMESTAMP_COLUMN] == datetime(2024, 1, 2, 10, 0, tzinfo=UTC)
    assert row[s.OPEN_COLUMN] == 100.0
    assert row[s.CLOSE_COLUMN] == 100.5
    assert row["volume"] == 1.0


@pytest.mark.parametrize("target", ["1m", "5m", "15m", "30m"])
def test_resample_rejects_target_finer_than_source(target: str) -> None:
    coarse = minute_bars(periods=120)
    hourly = resample_bars(coarse, "1h")
    with pytest.raises(ValueError, match="finer than the source"):
        resample_bars(hourly, target, source_timeframe="1h")


def test_resample_rejects_finer_target_with_explicit_source() -> None:
    with pytest.raises(ValueError, match="finer than the source"):
        resample_bars(minute_bars(periods=3), "5m", source_timeframe="1h")


def test_resample_rejects_unknown_timeframe() -> None:
    with pytest.raises(ValueError, match="unsupported timeframe"):
        resample_bars(minute_bars(), "7m")


def test_resample_rejects_non_polars_input() -> None:
    with pytest.raises(TypeError, match="polars DataFrame"):
        resample_bars(object(), "5m")  # type: ignore[arg-type]


def test_resample_rejects_non_canonical_frame() -> None:
    with pytest.raises(s.SchemaError):
        resample_bars(pl.DataFrame({"timestamp_utc": [ANCHOR]}), "5m")


@pytest.mark.parametrize("target", supported_timeframes())
def test_resample_supported_timeframes_from_minute_bars(target: str) -> None:
    result = resample_bars(minute_bars(periods=16), target)
    assert not result.is_empty()
    assert result.columns == [*s.BAR_COLUMNS, VWAP_COLUMN]
    assert result[s.TIMESTAMP_COLUMN].is_sorted()
    source_max = minute_bars(periods=16)[s.TIMESTAMP_COLUMN].max()
    assert result[s.TIMESTAMP_COLUMN].max() >= source_max


def test_resample_daily_to_weekly_counts_and_volume() -> None:
    daily = pl.DataFrame(
        [
            {
                s.TIMESTAMP_COLUMN: datetime(2024, 1, 2, tzinfo=UTC) + timedelta(days=index),
                s.INSTRUMENT_ID_COLUMN: "ES_FUT",
                s.ROOT_SYMBOL_COLUMN: "ES",
                s.EXPIRATION_COLUMN: EXPIRATION,
                s.OPEN_COLUMN: 100.0,
                s.HIGH_COLUMN: 101.0,
                s.LOW_COLUMN: 99.0,
                s.CLOSE_COLUMN: 100.5,
                s.VOLUME_COLUMN: 1.0,
                s.OPEN_INTEREST_COLUMN: 10.0,
                s.SETTLEMENT_COLUMN: 100.5,
            }
            for index in range(14)
        ]
    )
    weekly = resample_bars(daily, "1w", source_timeframe="1d")
    assert weekly.height == 2
    assert weekly[s.VOLUME_COLUMN].to_list() == [7.0, 7.0]
    assert weekly[s.TIMESTAMP_COLUMN].to_list() == [
        datetime(2024, 1, 8, tzinfo=UTC),
        datetime(2024, 1, 15, tzinfo=UTC),
    ]
