"""Unit tests for the canonical multi-timeframe registry (no data/network)."""

from __future__ import annotations

import dataclasses

import pytest
from src.data.timeframes import (
    TIMEFRAMES,
    TimeframeSpec,
    bars_per_year,
    base_timeframe,
    databento_base_schema,
    get_timeframe,
    ibkr_bar_size,
    resample_source,
    schema_for_timeframe,
    supported_timeframes,
    timeframe_seconds,
)

#: (name, every, ibkr bar size, databento base schema, bars/year, resample from)
EXPECTED: tuple[tuple[str, str, str, str, int, str | None], ...] = (
    ("1m", "1m", "1 min", "ohlcv-1m", 347_760, None),
    ("5m", "5m", "5 mins", "ohlcv-1m", 69_552, "1m"),
    ("15m", "15m", "15 mins", "ohlcv-1m", 23_184, "1m"),
    ("30m", "30m", "30 mins", "ohlcv-1m", 11_592, "1m"),
    ("1h", "1h", "1 hour", "ohlcv-1h", 5_796, "1m"),
    ("4h", "4h", "4 hours", "ohlcv-1h", 1_449, "1h"),
    ("1d", "1d", "1 day", "ohlcv-1d", 252, None),
    ("1w", "1w", "1 week", "ohlcv-1d", 52, "1d"),
)


def test_registry_exact_values_and_order() -> None:
    assert supported_timeframes() == tuple(row[0] for row in EXPECTED)
    assert len(TIMEFRAMES) == 8
    for spec, expected in zip(TIMEFRAMES, EXPECTED, strict=True):
        name, every, ibkr, databento, annual, resample = expected
        assert spec == TimeframeSpec(
            name=name,
            every=every,
            ibkr_bar_size=ibkr,
            databento_schema=databento,
            bars_per_year=annual,
            resample_from=resample,
        )
        assert spec.name == name
        assert spec.every == every
        assert spec.ibkr_bar_size == ibkr
        assert spec.databento_schema == databento
        assert spec.bars_per_year == annual
        assert spec.resample_from == resample


def test_timeframe_spec_is_frozen_and_helpers_are_consistent() -> None:
    spec = get_timeframe("5m")
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.name = "10m"  # type: ignore[misc]
    assert spec.databento_base_schema == "ohlcv-1m"
    assert spec.schema == "ohlcv-5m"
    assert spec.seconds == 300
    assert bars_per_year(spec) == 69_552
    assert ibkr_bar_size(spec) == "5 mins"


@pytest.mark.parametrize(
    ("timeframe", "expected"),
    [
        ("1m", 347_760),
        ("5m", 69_552),
        ("15m", 23_184),
        ("30m", 11_592),
        ("1h", 5_796),
        ("4h", 1_449),
        ("1d", 252),
        ("1w", 52),
    ],
)
def test_bars_per_year_values(timeframe: str, expected: int) -> None:
    assert bars_per_year(timeframe) == expected
    assert get_timeframe(timeframe).bars_per_year == expected


@pytest.mark.parametrize(
    ("timeframe", "seconds"),
    [
        ("1m", 60),
        ("5m", 300),
        ("15m", 900),
        ("30m", 1_800),
        ("1h", 3_600),
        ("4h", 14_400),
        ("1d", 86_400),
        ("1w", 604_800),
    ],
)
def test_timeframe_seconds(timeframe: str, seconds: int) -> None:
    assert timeframe_seconds(timeframe) == seconds


@pytest.mark.parametrize(
    ("timeframe", "expected"),
    [
        ("1m", "ohlcv-1m"),
        ("5m", "ohlcv-1m"),
        ("15m", "ohlcv-1m"),
        ("30m", "ohlcv-1m"),
        ("1h", "ohlcv-1h"),
        ("4h", "ohlcv-1h"),
        ("1d", "ohlcv-1d"),
        ("1w", "ohlcv-1d"),
    ],
)
def test_databento_base_schema_mapping(timeframe: str, expected: str) -> None:
    assert databento_base_schema(timeframe) == expected
    assert get_timeframe(timeframe).databento_base_schema == expected


@pytest.mark.parametrize(
    ("timeframe", "source"),
    [
        ("1m", None),
        ("5m", "1m"),
        ("15m", "1m"),
        ("30m", "1m"),
        ("1h", "1m"),
        ("4h", "1h"),
        ("1d", None),
        ("1w", "1d"),
    ],
)
def test_base_timeframe_mapping(timeframe: str, source: str | None) -> None:
    assert base_timeframe(timeframe) == source
    assert resample_source(timeframe) == source


def test_schema_for_timeframe_values() -> None:
    assert [schema_for_timeframe(tf) for tf in supported_timeframes()] == [
        "ohlcv-1m",
        "ohlcv-5m",
        "ohlcv-15m",
        "ohlcv-30m",
        "ohlcv-1h",
        "ohlcv-4h",
        "ohlcv-1d",
        "ohlcv-1w",
    ]
    assert schema_for_timeframe("5m") == "ohlcv-5m"
    assert schema_for_timeframe("1w") == "ohlcv-1w"


@pytest.mark.parametrize("name", ["1m", " 5M ", "15m", "30m", "1h", "4h", "1d", "1w", "1H"])
def test_get_timeframe_is_case_and_whitespace_tolerant(name: str) -> None:
    assert get_timeframe(name).name == name.strip().lower()


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("5min", "5m"),
        ("15mins", "15m"),
        ("30min", "30m"),
        ("1hr", "1h"),
        ("4hours", "4h"),
        ("daily", "1d"),
        ("weekly", "1w"),
        ("1wk", "1w"),
    ],
)
def test_get_timeframe_aliases(alias: str, canonical: str) -> None:
    assert get_timeframe(alias).name == canonical


def test_unknown_timeframe_raises_actionable_value_error() -> None:
    with pytest.raises(ValueError, match="unsupported timeframe"):
        get_timeframe("7m")
    with pytest.raises(ValueError, match="unsupported timeframe"):
        bars_per_year("bogus")


def test_get_timeframe_rejects_non_strings() -> None:
    with pytest.raises(TypeError, match="timeframe must be a string"):
        get_timeframe(5)  # type: ignore[arg-type]


def test_ibkr_bar_size_values() -> None:
    assert [ibkr_bar_size(tf.name) for tf in TIMEFRAMES] == [
        "1 min",
        "5 mins",
        "15 mins",
        "30 mins",
        "1 hour",
        "4 hours",
        "1 day",
        "1 week",
    ]
