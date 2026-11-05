"""Canonical multi-timeframe registry for FuturesRL.

Single source of truth for the supported bar timeframes: the Polars window
duration, the IBKR historical bar-size string, the Databento schema used to
fetch the **finest** base data for that timeframe family, the annualization
factor (``bars_per_year``) used by strategy/risk/metrics code, and the canonical
base timeframe used for auto-resampling.

Timeframes (exact values)
-------------------------
======  =======  =========  ===========  =============  ================
name    every    ibkr       databento    bars_per_year   resample from
======  =======  =========  ===========  =============  ================
1m      1m       1 min      ohlcv-1m     347760         (native)
5m      5m       5 mins     ohlcv-1m     69552          1m
15m     15m      15 mins    ohlcv-1m     23184          1m
30m     30m      30 mins    ohlcv-1m     11592          1m
1h      1h       1 hour     ohlcv-1h     5796           1m
4h      4h       4 hours    ohlcv-1h     1449           1h
1d      1d       1 day      ohlcv-1d     252            (native)
1w      1w       1 week     ohlcv-1d     52             1d
======  =======  =========  ===========  =============  ================

The module is dependency-free (stdlib + ``dataclasses``) so strategy, risk,
backtest and API layers can import it without pulling in Polars or the data
storage stack.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

__all__ = [
    "TIMEFRAMES",
    "TimeframeSpec",
    "bars_per_year",
    "base_timeframe",
    "databento_base_schema",
    "get_timeframe",
    "ibkr_bar_size",
    "resample_source",
    "schema_for_timeframe",
    "supported_timeframes",
    "timeframe_seconds",
]

#: Canonical target/repository schema prefix for derived timeframe partitions.
_TIMEFRAME_SCHEMA_PREFIX = "ohlcv-"


@dataclass(frozen=True, slots=True)
class TimeframeSpec:
    """Immutable specification of one supported bar timeframe."""

    #: Canonical timeframe name (``"1m"``, ``"5m"``, ..., ``"1w"``).
    name: str
    #: Polars ``every``/``period`` duration string for ``group_by_dynamic``.
    every: str
    #: IBKR historical ``barSizeSetting`` string (``"1 min"``, ``"1 hour"``, ...).
    ibkr_bar_size: str
    #: Databento schema that provides the finest base data for this timeframe.
    databento_schema: str
    #: Trading bars per year (annualization factor for strategies/risk/metrics).
    bars_per_year: int
    #: Finer canonical timeframe auto-resampled into this one (``None`` = native).
    resample_from: str | None = None

    @property
    def databento_base_schema(self) -> str:
        """Alias for :attr:`databento_schema` (finest Databento base schema)."""
        return self.databento_schema

    @property
    def schema(self) -> str:
        """Repository schema for this timeframe (``ohlcv-1m`` ... ``ohlcv-1w``)."""
        return f"{_TIMEFRAME_SCHEMA_PREFIX}{self.name}"

    @property
    def seconds(self) -> int:
        """Wall-clock duration of one bar in seconds (UTC)."""
        return _TIMEFRAME_SECONDS[self.name]


#: Exact per-timeframe wall-clock durations in seconds (UTC).
_TIMEFRAME_SECONDS: Mapping[str, int] = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1_800,
    "1h": 3_600,
    "4h": 14_400,
    "1d": 86_400,
    "1w": 604_800,
}

#: Supported timeframes in canonical order, fastest first.
TIMEFRAMES: tuple[TimeframeSpec, ...] = (
    TimeframeSpec(
        name="1m",
        every="1m",
        ibkr_bar_size="1 min",
        databento_schema="ohlcv-1m",
        bars_per_year=347_760,
        resample_from=None,
    ),
    TimeframeSpec(
        name="5m",
        every="5m",
        ibkr_bar_size="5 mins",
        databento_schema="ohlcv-1m",
        bars_per_year=69_552,
        resample_from="1m",
    ),
    TimeframeSpec(
        name="15m",
        every="15m",
        ibkr_bar_size="15 mins",
        databento_schema="ohlcv-1m",
        bars_per_year=23_184,
        resample_from="1m",
    ),
    TimeframeSpec(
        name="30m",
        every="30m",
        ibkr_bar_size="30 mins",
        databento_schema="ohlcv-1m",
        bars_per_year=11_592,
        resample_from="1m",
    ),
    TimeframeSpec(
        name="1h",
        every="1h",
        ibkr_bar_size="1 hour",
        databento_schema="ohlcv-1h",
        bars_per_year=5_796,
        resample_from="1m",
    ),
    TimeframeSpec(
        name="4h",
        every="4h",
        ibkr_bar_size="4 hours",
        databento_schema="ohlcv-1h",
        bars_per_year=1_449,
        resample_from="1h",
    ),
    TimeframeSpec(
        name="1d",
        every="1d",
        ibkr_bar_size="1 day",
        databento_schema="ohlcv-1d",
        bars_per_year=252,
        resample_from=None,
    ),
    TimeframeSpec(
        name="1w",
        every="1w",
        ibkr_bar_size="1 week",
        databento_schema="ohlcv-1d",
        bars_per_year=52,
        resample_from="1d",
    ),
)

#: Canonical timeframe name -> spec (exact, case-sensitive canonical keys).
TIMEFRAME_BY_NAME: Mapping[str, TimeframeSpec] = {spec.name: spec for spec in TIMEFRAMES}

#: Common operator/alias spellings accepted by :func:`get_timeframe`.
_ALIASES: Mapping[str, str] = {
    "1min": "1m",
    "1mins": "1m",
    "1minute": "1m",
    "5min": "5m",
    "5mins": "5m",
    "15min": "15m",
    "15mins": "15m",
    "30min": "30m",
    "30mins": "30m",
    "1hr": "1h",
    "1hour": "1h",
    "1hours": "1h",
    "4hr": "4h",
    "4hour": "4h",
    "4hours": "4h",
    "1day": "1d",
    "daily": "1d",
    "1week": "1w",
    "1wk": "1w",
    "weekly": "1w",
}


def _normalise(name: str) -> str:
    """Normalise a timeframe string (trim, lowercase, common aliases)."""
    if not isinstance(name, str):
        raise TypeError(f"timeframe must be a string, got {type(name)!r}")
    cleaned = name.strip().lower()
    return _ALIASES.get(cleaned, cleaned)


def _as_spec(timeframe: str | TimeframeSpec) -> TimeframeSpec:
    """Resolve a timeframe string/spec to a :class:`TimeframeSpec`."""
    if isinstance(timeframe, TimeframeSpec):
        return timeframe
    return get_timeframe(timeframe)


def get_timeframe(timeframe: str | TimeframeSpec) -> TimeframeSpec:
    """Return the :class:`TimeframeSpec` for ``timeframe``.

    Canonical names (``"1m"`` ... ``"1w"``) are case/whitespace tolerant and a
    few common aliases (``"5min"``, ``"1hour"``, ``"daily"``, ...) are accepted.
    Unknown names raise an actionable :class:`ValueError` listing every
    supported timeframe.
    """
    if isinstance(timeframe, TimeframeSpec):
        return timeframe
    key = _normalise(timeframe)
    spec = TIMEFRAME_BY_NAME.get(key)
    if spec is None:
        raise ValueError(
            f"unsupported timeframe {timeframe!r}; expected one of "
            f"{supported_timeframes()} (aliases like '5min'/'1hour' are also accepted)"
        )
    return spec


def supported_timeframes() -> tuple[str, ...]:
    """Canonical supported timeframe names in canonical order."""
    return tuple(spec.name for spec in TIMEFRAMES)


def timeframe_seconds(timeframe: str | TimeframeSpec) -> int:
    """Wall-clock duration of one ``timeframe`` bar in seconds (UTC)."""
    return _as_spec(timeframe).seconds


def bars_per_year(timeframe: str | TimeframeSpec) -> int:
    """Trading bars per year for ``timeframe`` (annualization factor)."""
    return _as_spec(timeframe).bars_per_year


def ibkr_bar_size(timeframe: str | TimeframeSpec) -> str:
    """IBKR ``barSizeSetting`` string for ``timeframe`` (e.g. ``"1 min"``)."""
    return _as_spec(timeframe).ibkr_bar_size


def databento_base_schema(timeframe: str | TimeframeSpec) -> str:
    """Finest Databento schema used to fetch base data for ``timeframe``.

    E.g. ``5m -> ohlcv-1m``, ``4h -> ohlcv-1h``, ``1w -> ohlcv-1d``. Estimators and
    downloads should use this schema and then resample to the target timeframe.
    """
    return _as_spec(timeframe).databento_schema


def schema_for_timeframe(timeframe: str | TimeframeSpec) -> str:
    """Repository schema for a timeframe's cached bars (``ohlcv-5m``, ``ohlcv-1w``...)."""
    return _as_spec(timeframe).schema


def base_timeframe(timeframe: str | TimeframeSpec) -> str | None:
    """Canonical finer timeframe auto-resampled into ``timeframe`` (or ``None``).

    ``5m/15m/30m/1h -> 1m``, ``4h -> 1h``, ``1w -> 1d``; ``1m``/``1d`` are native
    (``None``). IBKR/Databento ingestion can fetch ``base_timeframe(tf) or tf``
    and then resample to the requested target timeframe.
    """
    return _as_spec(timeframe).resample_from


def resample_source(timeframe: str | TimeframeSpec) -> str | None:
    """Alias for :func:`base_timeframe` (source timeframe used for resampling)."""
    return base_timeframe(timeframe)
