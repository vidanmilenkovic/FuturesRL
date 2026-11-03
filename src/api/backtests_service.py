"""Backtest run store and real ``BacktestEngine`` adapter.

``POST /api/backtests`` executes the production
:class:`~src.backtest.engine.BacktestEngine` in vectorised mode over the
deterministic offline market (no Databento, no IBKR, no network), computes the
institutional metrics with :mod:`src.analytics.metrics` and stores the mapped
tear sheet in the in-memory run store. ``GET /api/backtests`` additionally serves
three seeded *fixture* runs (clearly marked ``data_source="fixture"``) so the
Backtest Lab has comparison material immediately after boot.

Failure policy
--------------
If the engine (or the strategy/data layer it needs) cannot be imported or raises,
the POST still returns a completed deterministic fixture run whose ``error``
field explains why, and ``data_source="fixture"`` — the API never pretends a
fallback is a real simulation.
"""

from __future__ import annotations

import inspect
import json
import logging
import math
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from datetime import time as dt_time
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import polars as pl
from configs.instruments import InstrumentsConfig
from configs.settings import Settings
from configs.strategy import StrategyConfig
from pydantic import ValidationError

from src.api import fixtures
from src.api.protocols import StrategyCatalog
from src.api.schemas import (
    AttributionRow,
    BacktestListResponse,
    BacktestRunConfig,
    BacktestRunSummary,
    CachedBarsSummary,
    MetricsSet,
    MonthlyReturn,
    SeriesPoint,
    TailPoint,
    TearsheetResponse,
    TradeRow,
    WalkForwardFold,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.analytics.metrics import PerformanceMetrics
    from src.backtest.engine import BacktestResult

logger = logging.getLogger(__name__)

__all__ = ["EngineBacktestService"]

#: Rolling window (bars) used by the tear sheet rolling-Sharpe panel.
ROLLING_WINDOW: int = 63

#: Number of tail observations per side on the Rachev plot.
TAIL_OBSERVATIONS: int = 30

#: Maximum retained runs in the in-memory store.
MAX_RUNS: int = 40

#: Seed runs created at boot (one per strategy, capped).
SEED_RUNS: int = 3

#: Version of the persisted backtest-artifact JSON payload.
ARTIFACT_SCHEMA_VERSION: int = 1

#: Default metric aliases when mapping a walk-forward report onto ``MetricsSet``.
_METRIC_ALIASES: Mapping[str, tuple[str, ...]] = {
    "cagr": ("cagr", "annualized_return", "annual_return"),
    "annual_volatility": ("annual_volatility", "annualized_volatility", "volatility"),
    "sharpe": ("sharpe", "sharpe_ratio", "mean_sharpe"),
    "sortino": ("sortino", "sortino_ratio"),
    "calmar": ("calmar", "calmar_ratio"),
    "omega": ("omega", "omega_ratio"),
    "max_drawdown": ("max_drawdown", "max_drawdown_pct"),
    "max_drawdown_duration_days": ("max_drawdown_duration_days", "max_drawdown_duration"),
    "var_95": ("var_95", "var_95_historical"),
    "var_99": ("var_99", "var_99_historical"),
    "cvar_95": ("cvar_95",),
    "cvar_99": ("cvar_99",),
    "rachev_ratio": ("rachev_ratio", "rachev_ratio_05"),
    "rom": ("rom", "return_on_max_initial_margin"),
    "avg_margin_utilization": ("avg_margin_utilization", "average_margin_utilization"),
    "peak_margin_utilization": ("peak_margin_utilization",),
    "turnover": ("turnover",),
    "implementation_shortfall_ticks": (
        "implementation_shortfall_ticks",
        "implementation_shortfall",
    ),
    "commission_to_gross_pnl": ("commission_to_gross_pnl", "commission_to_gross_pnl_ratio"),
}


def _series_points(timestamps: Sequence[datetime], values: Sequence[float]) -> list[SeriesPoint]:
    return [
        SeriesPoint(time=timestamp, value=round(float(value), 6))
        for timestamp, value in zip(timestamps, values, strict=False)
        if math.isfinite(float(value))
    ]


def _monthly_returns(
    timestamps: Sequence[datetime], equity: Sequence[float]
) -> list[MonthlyReturn]:
    last_by_month: dict[str, float] = {}
    for timestamp, value in zip(timestamps, equity, strict=False):
        last_by_month[timestamp.strftime("%Y-%m")] = float(value)
    output: list[MonthlyReturn] = []
    previous: float | None = None
    for month, value in last_by_month.items():
        if previous is None or previous <= 0.0:
            output.append(MonthlyReturn(month=month, value=0.0))
        else:
            output.append(MonthlyReturn(month=month, value=round(value / previous - 1.0, 8)))
        previous = value
    return output


def _tail_distribution(
    returns: np.ndarray, *, observations: int = TAIL_OBSERVATIONS
) -> list[TailPoint]:
    if returns.size < 4:
        return []
    usable = returns[np.isfinite(returns)]
    if usable.size < 4:
        return []
    sorted_returns = np.sort(usable)
    count = max(1, min(observations, sorted_returns.size // 8 or 1))
    losses = sorted_returns[:count]
    gains = sorted_returns[-count:][::-1]
    points: list[TailPoint] = []
    for rank, value in enumerate(losses, start=1):
        points.append(TailPoint(rank=rank, return_pct=round(float(value), 8), tail="loss"))
    for rank, value in enumerate(gains, start=1):
        points.append(TailPoint(rank=rank, return_pct=round(float(value), 8), tail="gain"))
    return points


def _performance_metrics(metrics: PerformanceMetrics) -> MetricsSet:
    return MetricsSet(
        cagr=round(metrics.cagr, 8),
        annual_volatility=round(metrics.annualized_volatility, 8),
        sharpe=round(metrics.sharpe, 6),
        sortino=round(metrics.sortino, 6),
        calmar=round(metrics.calmar, 6),
        omega=round(metrics.omega, 6),
        max_drawdown=round(metrics.max_drawdown, 8),
        max_drawdown_duration_days=round(metrics.max_drawdown_duration_days, 4),
        var_95=round(metrics.var_95_historical, 8),
        var_99=round(metrics.var_99_historical, 8),
        cvar_95=round(metrics.cvar_95, 8),
        cvar_99=round(metrics.cvar_99, 8),
        rachev_ratio=round(metrics.rachev_ratio_05, 6),
        rom=round(metrics.return_on_max_initial_margin, 8),
        avg_margin_utilization=round(metrics.average_margin_utilization, 8),
        peak_margin_utilization=round(metrics.peak_margin_utilization, 8),
        turnover=round(metrics.turnover, 6),
        implementation_shortfall_ticks=round(metrics.implementation_shortfall_ticks, 6),
        commission_to_gross_pnl=round(metrics.commission_to_gross_pnl_ratio, 8),
    )


def _simple_metrics(values: Sequence[float], count_per_year: int = 252) -> MetricsSet:
    """Arithmetic fallback when :mod:`src.analytics` is unavailable."""
    equity = np.asarray([float(value) for value in values], dtype=np.float64)
    if equity.size < 2:
        raise ValueError("an equity curve needs at least two observations")
    returns = np.diff(equity) / equity[:-1]
    years = max(returns.size / count_per_year, 1.0 / count_per_year)
    cagr = float((equity[-1] / equity[0]) ** (1.0 / years) - 1.0)
    volatility = float(np.std(returns, ddof=1)) * math.sqrt(count_per_year)
    sharpe = cagr / volatility if volatility > 0 else 0.0
    peaks = np.maximum.accumulate(equity)
    drawdowns = equity / peaks - 1.0
    max_drawdown = float(-drawdowns.min()) if drawdowns.size else 0.0
    return MetricsSet(
        cagr=round(cagr, 8),
        annual_volatility=round(volatility, 8),
        sharpe=round(sharpe, 6),
        sortino=round(sharpe, 6),
        calmar=round(cagr / max_drawdown if max_drawdown > 0 else 0.0, 6),
        omega=1.0,
        max_drawdown=round(max_drawdown, 8),
        max_drawdown_duration_days=0.0,
        var_95=round(float(-np.quantile(returns, 0.05)), 8),
        var_99=round(float(-np.quantile(returns, 0.01)), 8),
        cvar_95=round(float(-returns[returns <= np.quantile(returns, 0.05)].mean()), 8),
        cvar_99=round(float(-returns[returns <= np.quantile(returns, 0.01)].mean()), 8),
        rachev_ratio=1.0,
        rom=round(abs(cagr) / 0.4, 8),
        avg_margin_utilization=0.4,
        peak_margin_utilization=0.55,
        turnover=1.0,
        implementation_shortfall_ticks=0.0,
        commission_to_gross_pnl=0.0,
    )


def _object_mapping(value: object) -> Mapping[str, object]:
    """Best-effort mapping view of a report/metrics object."""
    if isinstance(value, Mapping):
        return cast(Mapping[str, object], value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            result = to_dict()
        except Exception:  # pragma: no cover - defensive report adapter
            return {}
        if isinstance(result, Mapping):
            return cast(Mapping[str, object], result)
    as_dict = getattr(value, "model_dump", None) if not isinstance(value, type) else None
    if callable(as_dict):
        try:
            result = as_dict()
        except Exception:  # pragma: no cover - defensive report adapter
            return {}
        if isinstance(result, Mapping):
            return cast(Mapping[str, object], result)
    return {}


def _object_sequence(value: object) -> list[object]:
    """Sequence view of a report field (empty when absent/wrong-typed)."""
    if isinstance(value, Mapping):
        return []
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return list(cast(Sequence[object], value))
    return []


def _mappings(value: object) -> list[Mapping[str, object]]:
    """Mappings for a sequence of report records (mapping or dataclass-like)."""
    records: list[Mapping[str, object]] = []
    for item in _object_sequence(value):
        mapping = _object_mapping(item)
        if mapping:
            records.append(mapping)
    return records


def _first_float(mappings: Sequence[Mapping[str, object]], aliases: Sequence[str]) -> float | None:
    for mapping in mappings:
        for alias in aliases:
            value = mapping.get(alias)
            if isinstance(value, bool):
                continue
            if isinstance(value, int | float) and math.isfinite(float(value)):
                return float(value)
    return None


def _first_int(mappings: Sequence[Mapping[str, object]], aliases: Sequence[str]) -> int | None:
    number = _first_float(mappings, aliases)
    return None if number is None else round(number)


def _first_datetime(mapping: Mapping[str, object], aliases: Sequence[str]) -> datetime | None:
    for alias in aliases:
        value = mapping.get(alias)
        if isinstance(value, datetime):
            return value.astimezone(UTC) if value.tzinfo is not None else None
        if isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                continue
            return parsed.astimezone(UTC) if parsed.tzinfo is not None else None
        if value is not None and hasattr(value, "year") and hasattr(value, "month"):
            try:
                return datetime.combine(cast("date", value), dt_time.min, tzinfo=UTC)
            except TypeError:  # pragma: no cover - exotic date-like value
                continue
    return None


def _metric_mappings(
    report: object, report_dict: Mapping[str, object]
) -> list[Mapping[str, object]]:
    """Every mapping that may carry walk-forward aggregate metrics."""
    found: list[Mapping[str, object]] = []
    sources: list[object] = [report_dict]
    for key in (
        "combined_metrics",
        "aggregate_metrics",
        "aggregated_metrics",
        "mean_metrics",
        "aggregate",
        "metrics",
        "summary",
    ):
        if key in report_dict:
            sources.append(report_dict[key])
    for name in ("combined_metrics", "aggregate_metrics", "aggregate", "metrics", "summary"):
        value = getattr(report, name, None)
        if value is not None:
            sources.append(value)
    for source in sources:
        mapping = _object_mapping(source)
        if not mapping:
            continue
        found.append(mapping)
        for nested_key in ("metrics", "combined_metrics", "aggregate", "mean", "mean_metrics"):
            nested = mapping.get(nested_key)
            nested_mapping = _object_mapping(nested) if nested is not None else {}
            if nested_mapping:
                found.append(nested_mapping)
        # Aggregate stats are often ``{"sharpe": {"mean": 1.2, ...}, ...}``.
        flattened: dict[str, object] = {}
        for metric_key, stats in mapping.items():
            stats_mapping = _object_mapping(stats) if isinstance(stats, Mapping) else {}
            if not stats_mapping:
                continue
            for stat_key in ("mean", "combined", "pooled", "median", "value"):
                number = _first_float([stats_mapping], (stat_key,))
                if number is not None:
                    flattened.setdefault(metric_key, number)
                    flattened[f"{metric_key}_{stat_key}"] = number
                    break
        if flattened:
            found.append(flattened)
    return found


def _metrics_with_defaults(
    mappings: Sequence[Mapping[str, object]], fallback: MetricsSet
) -> MetricsSet:
    """Map a (versioned) walk-forward metric payload onto ``MetricsSet``."""
    values: dict[str, float] = {
        field: float(getattr(fallback, field)) for field in MetricsSet.model_fields
    }
    for field, aliases in _METRIC_ALIASES.items():
        resolved = _first_float(mappings, aliases)
        if resolved is not None:
            values[field] = resolved
    return MetricsSet(**values)


def _series_points_from_payload(
    payload: Mapping[str, object], keys: Sequence[str]
) -> list[SeriesPoint]:
    """Parse a fold's series payload (list of points, values+timestamps, or mapping)."""
    for key in keys:
        raw = payload.get(key)
        if raw is None:
            continue
        entries = _object_sequence(raw)
        points: list[SeriesPoint] = []
        for entry in entries:
            mapping = _object_mapping(entry)
            if mapping:
                timestamp = _first_datetime(mapping, ("time", "timestamp", "date"))
                value = _first_float([mapping], ("value", "equity", "drawdown"))
                if timestamp is not None and value is not None:
                    points.append(SeriesPoint(time=timestamp, value=round(value, 6)))
        if points:
            return points
        mapping = _object_mapping(raw)
        if mapping:
            timestamps = _object_sequence(mapping.get("timestamps") or mapping.get("times"))
            values = _object_sequence(mapping.get("values") or mapping.get("equity"))
            times = [
                parsed
                for parsed in (_fold_datetime(item) for item in timestamps)
                if parsed is not None
            ]
            numbers = [
                float(item)
                for item in values
                if isinstance(item, int | float) and not isinstance(item, bool)
            ]
            return _series_points(times, numbers)
        numbers = [
            float(item)
            for item in entries
            if isinstance(item, int | float) and not isinstance(item, bool)
        ]
        times = [
            parsed
            for parsed in (
                _fold_datetime(item) for item in _object_sequence(payload.get("timestamps"))
            )
            if parsed is not None
        ]
        if numbers and len(times) == len(numbers):
            return _series_points(times, numbers)
    return []


def _fold_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo is not None else None
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed.astimezone(UTC) if parsed.tzinfo is not None else None
    return None


def _walks(
    payload: Mapping[str, object],
) -> tuple[list[datetime], list[float]]:
    timestamps: list[datetime] = []
    values: list[float] = []
    for entry in _object_sequence(payload.get("timestamps") or payload.get("times")):
        parsed = _fold_datetime(entry)
        if parsed is not None:
            timestamps.append(parsed)
    for entry in _object_sequence(payload.get("values") or payload.get("equity")):
        if isinstance(entry, int | float) and not isinstance(entry, bool):
            values.append(float(entry))
    return timestamps, values


def _fold_from_payload(index: int, payload: Mapping[str, object]) -> WalkForwardFold:
    equity = _series_points_from_payload(
        payload, ("equity_curve", "equity", "test_equity", "equity_values")
    )
    underwater = _series_points_from_payload(
        payload, ("underwater", "underwater_curve", "drawdown")
    )
    stored_fold = _first_int([payload], ("fold", "fold_index", "index"))
    fold_index = stored_fold + 1 if stored_fold is not None and stored_fold >= 0 else index
    metrics = _object_mapping(payload.get("metrics"))
    return WalkForwardFold(
        fold_index=fold_index,
        train_start=_first_datetime(
            payload, ("train_start_time", "train_start", "train_start_date", "train_from")
        ),
        train_end=_first_datetime(
            payload, ("train_end_time", "train_end", "train_end_date", "train_to")
        ),
        test_start=_first_datetime(
            payload,
            ("test_start_time", "test_start", "test_start_date", "test_from", "start"),
        ),
        test_end=_first_datetime(
            payload, ("test_end_time", "test_end", "test_end_date", "test_to", "end")
        ),
        sharpe=_first_float([payload, metrics], ("sharpe", "sharpe_ratio", "mean_sharpe")),
        return_pct=_first_float([payload, metrics], ("return_pct", "total_return", "return")),
        n_trades=_first_int([payload, metrics], ("n_trades", "total_trades", "trades")),
        equity_curve=equity,
        underwater=underwater,
    )


def _fold_payloads(report: object, report_dict: Mapping[str, object]) -> list[Mapping[str, object]]:
    for key in ("folds", "fold_reports", "results", "splits", "fold_metrics"):
        if key in report_dict:
            records = _mappings(report_dict[key])
            if records:
                return records
    for name in ("folds", "fold_reports", "results", "splits"):
        value = getattr(report, name, None)
        if value is not None:
            records = _mappings(value)
            if records:
                return records
    return []


def _series_from_curve_mapping(payload: Mapping[str, object], value_key: str) -> list[SeriesPoint]:
    raw = payload.get("concatenated_equity_curve") or payload.get("equity_curve")
    mapping = _object_mapping(raw)
    if not mapping:
        return []
    timestamps = [
        parsed
        for parsed in (_fold_datetime(item) for item in _object_sequence(mapping.get("timestamps")))
        if parsed is not None
    ]
    numbers = [
        float(item)
        for item in _object_sequence(mapping.get(value_key))
        if isinstance(item, int | float) and not isinstance(item, bool)
    ]
    if len(timestamps) == len(numbers) and numbers:
        return _series_points(timestamps, numbers)
    return []


def _merge_combined_folds(
    folds: Sequence[WalkForwardFold],
    combined: Sequence[SeriesPoint],
    drawdown: Sequence[SeriesPoint],
    fold_count: int,
) -> list[WalkForwardFold]:
    """Attach per-fold equity/underwater slices from the compounded WF curve."""
    if not combined or fold_count < 1:
        return list(folds)
    timestamps = [point.time for point in combined]
    values = [point.value for point in combined]
    underwater_values = [point.value for point in drawdown] if drawdown else None
    synthesized = _synthesize_walk_forward_folds(
        timestamps, values, fold_count, underwater=underwater_values
    )
    merged: list[WalkForwardFold] = []
    for index, synthesized_fold in enumerate(synthesized):
        if index < len(folds):
            merged.append(
                folds[index].model_copy(
                    update={
                        "equity_curve": synthesized_fold.equity_curve,
                        "underwater": synthesized_fold.underwater,
                        "test_start": folds[index].test_start or synthesized_fold.test_start,
                        "test_end": folds[index].test_end or synthesized_fold.test_end,
                        "sharpe": folds[index].sharpe
                        if folds[index].sharpe is not None
                        else synthesized_fold.sharpe,
                        "return_pct": folds[index].return_pct
                        if folds[index].return_pct is not None
                        else synthesized_fold.return_pct,
                    }
                )
            )
        else:
            merged.append(synthesized_fold)
    return merged


def _run_id_number(run_id: str) -> int | None:
    """Numeric suffix of a ``bt-0007`` run id (``None`` for unknown formats)."""
    _, _, suffix = run_id.partition("-")
    return int(suffix) if suffix.isdigit() else None


def _report_attributes(report: object) -> Mapping[str, object]:
    """Scalar/list attributes of a walk-forward report object (mapping view)."""
    names = (
        "fold_count",
        "n_folds",
        "splits",
        "mean_sharpe",
        "sharpe",
        "total_trades",
        "n_trades",
        "total_rebalances",
        "n_rebalances",
        "rebalances",
        "final_equity",
        "combined_metrics",
        "aggregate_metrics",
        "aggregate",
        "folds",
        "fold_reports",
    )
    mapping: dict[str, object] = {}
    for name in names:
        value = getattr(report, name, None)
        if value is not None:
            mapping[name] = value
    return mapping


def _synthesize_walk_forward_folds(
    timestamps: Sequence[datetime],
    values: Sequence[float],
    count: int,
    *,
    underwater: Sequence[float] | None = None,
) -> list[WalkForwardFold]:
    """Deterministic per-fold series for the fixture fallback (and missing folds)."""
    size = min(len(timestamps), len(values))
    if size < 2 or count < 1:
        return []
    numeric = np.asarray([float(value) for value in values[:size]], dtype=np.float64)
    moments = list(timestamps[:size])
    peaks = np.maximum.accumulate(numeric)
    chunks = np.array_split(np.arange(size, dtype=np.int64), max(1, min(count, size)))
    folds: list[WalkForwardFold] = []
    for fold_index, indices in enumerate(chunks, start=1):
        if indices.size == 0:
            continue
        slice_values = [float(numeric[index]) for index in indices]
        slice_times = [moments[int(index)] for index in indices]
        returns = np.diff(np.asarray(slice_values, dtype=np.float64)) / np.asarray(
            slice_values[:-1], dtype=np.float64
        )
        deviation = float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0
        sharpe = float(np.mean(returns) / deviation * math.sqrt(252.0)) if deviation > 0.0 else 0.0
        return_pct = (
            float(slice_values[-1] / slice_values[0] - 1.0) if slice_values[0] != 0.0 else 0.0
        )
        if underwater is not None and len(underwater) >= size:
            drawdown = [float(underwater[int(index)]) for index in indices]
        else:
            drawdown = [
                float(numeric[index] / peaks[index] - 1.0) if peaks[index] > 0 else 0.0
                for index in indices
            ]
        folds.append(
            WalkForwardFold(
                fold_index=fold_index,
                test_start=slice_times[0],
                test_end=slice_times[-1],
                sharpe=round(sharpe, 6),
                return_pct=round(return_pct, 8),
                equity_curve=_series_points(slice_times, slice_values),
                underwater=_series_points(slice_times, drawdown),
            )
        )
    return folds


def fixture_backtest_config(config: StrategyConfig) -> StrategyConfig:
    """Fixture-simulation variant of a production strategy config.

    The committed YAML stays the single source of truth for display/editing
    (``/api/strategies``), but the deterministic offline fixture market has no
    real term structure and its synthetic series makes KAMA and TSMOM disagree,
    which zeroes the ``carry_filter`` + ``kama`` chain and produces a flat tear
    sheet. For API backtests we therefore inject a variant that keeps the real
    allocator/risk/translator stages but reduces the timing overlay to TSMOM (the
    momentum signal that *is* present in the fixture features). This is a
    data-fixture adaptation, never a change to the production config.
    """
    tsmom = next((stage for stage in config.timing if stage.name == "tsmom"), None)
    if tsmom is None or len(config.timing) <= 1:
        return config
    return config.model_copy(update={"timing": (tsmom,)})


class CachedDataUnavailableError(ValueError):
    """Raised when ``data_mode="cached_real"`` has no sufficient real cache."""


def _strategy_roots(config: StrategyConfig) -> tuple[str, ...] | None:
    """Explicit roots configured on the strategy universe, or ``None`` for default."""
    params = getattr(config.universe, "params", None) or {}
    raw = params.get("roots") if isinstance(params, Mapping) else None
    if isinstance(raw, str):
        roots = tuple(part.strip().upper() for part in raw.split(",") if part.strip())
        return roots or None
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        roots = tuple(str(item).strip().upper() for item in raw if str(item).strip())
        return roots or None
    return None


def _strategy_config_snapshot(config: StrategyConfig) -> dict[str, object]:
    """JSON-safe snapshot of the exact validated config used for a run."""
    return config.model_dump(mode="json")


def _trade_rows(trades: Sequence[object]) -> list[TradeRow]:
    """Map engine ``TradeRecord``s onto the additive API ``TradeRow`` contract."""
    rows: list[TradeRow] = []
    for trade in trades:
        timestamp = getattr(trade, "timestamp", None)
        if not isinstance(timestamp, datetime):
            continue
        rows.append(
            TradeRow(
                timestamp=timestamp,
                order_id=str(getattr(trade, "order_id", "")),
                instrument_id=str(getattr(trade, "instrument_id", "")),
                side=str(getattr(trade, "side", "")),
                quantity=int(getattr(trade, "quantity", 0)),
                price=float(getattr(trade, "price", 0.0)),
                notional=float(getattr(trade, "notional", 0.0)),
                commissions=float(getattr(trade, "commissions", 0.0)),
                slippage=float(getattr(trade, "slippage", 0.0)),
                kind=str(getattr(trade, "kind", "")),
            )
        )
    return rows


class EngineBacktestService:
    """In-memory run store backed by the production backtest engine."""

    def __init__(
        self,
        *,
        settings: Settings,
        market: fixtures.OfflineMarket,
        catalog: StrategyCatalog,
        clock: fixtures.Clock = fixtures.utc_now,
    ) -> None:
        self._settings = settings
        self._market = market
        self._catalog = catalog
        self._clock = clock
        self._master: InstrumentsConfig = market.instrument_master
        self._runs: dict[str, BacktestRunSummary] = {}
        self._tearsheets: dict[str, TearsheetResponse] = {}
        self._next_id = 1
        self._artifacts_dir: Path = settings.backtest_artifacts_dir
        self._persisted_ids: set[str] = set()
        self._warnings: list[str] = []
        self._load_persisted_runs()
        self._seed_fixture_runs()

    @property
    def warnings(self) -> tuple[str, ...]:
        """Non-fatal artifact-store warnings (corrupt files, write failures)."""
        return tuple(self._warnings)

    # -- protocol API ----------------------------------------------------------

    def list_runs(self) -> BacktestListResponse:
        """Run history, newest first."""
        runs = sorted(self._runs.values(), key=lambda run: run.created_at, reverse=True)
        return BacktestListResponse(runs=runs, available_strategies=self._catalog.available_ids())

    def get_run(self, run_id: str) -> BacktestRunSummary | None:
        """Stored run summary or ``None``."""
        return self._runs.get(run_id)

    def tearsheet(self, run_id: str) -> TearsheetResponse | None:
        """Stored tear sheet for ``run_id``."""
        return self._tearsheets.get(run_id)

    def run(self, request: BacktestRunConfig) -> BacktestRunSummary:
        """Execute one deterministic backtest/walk-forward with the requested data mode."""
        config = self._catalog.core_config(request.strategy_id)
        if config is None:
            raise KeyError(f"unknown strategy {request.strategy_id!r}")
        cached_market: fixtures.OfflineMarket | None = None
        cached_summary: CachedBarsSummary | None = None
        if request.data_mode in ("auto", "cached_real"):
            cached = self._cached_real_market(config, request)
            if cached is None and request.data_mode == "cached_real":
                raise CachedDataUnavailableError(
                    f"no sufficient cached real ohlcv-1d bars for strategy "
                    f"{request.strategy_id!r} roots in the requested window; ingest data "
                    "via POST /api/data/ingest or use data_mode='auto'/'synthetic_offline'"
                )
            if cached is not None:
                cached_market, cached_summary = cached
        if request.walk_forward:
            return self._run_walk_forward(
                config, request, market=cached_market, cached_bars=cached_summary
            )
        started = time.perf_counter()
        try:
            summary, tearsheet = self._execute_engine(
                config, request, market=cached_market, cached_bars=cached_summary
            )
            run_id = self._allocate_run_id()
            duration = time.perf_counter() - started
            stored = summary.model_copy(
                update={
                    "id": run_id,
                    "created_at": self._clock(),
                    "duration_seconds": round(duration, 3),
                    "kind": "backtest",
                }
            )
            tearsheet = tearsheet.model_copy(update={"run_id": run_id})
            self._store_run(stored, tearsheet)
            return stored
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            return self._fixture_run(request, reason=reason, started=started)

    # -- engine execution ------------------------------------------------------

    def _cached_real_market(
        self, config: StrategyConfig, request: BacktestRunConfig
    ) -> tuple[fixtures.OfflineMarket, CachedBarsSummary] | None:
        """Load real cached ``ohlcv-1d`` bars and build an offline market, if sufficient."""
        from src.data.storage import ParquetRepository

        roots = _strategy_roots(config)
        if roots is None:
            return None
        master = self._master
        selected = tuple(root for root in roots if master.has(root))
        if not selected or len(selected) != len(roots):
            return None
        repository = ParquetRepository(self._settings)
        start = datetime.combine(request.start - timedelta(days=520), dt_time.min, tzinfo=UTC)
        end = datetime.combine(request.end, dt_time(23, 59, 59), tzinfo=UTC)
        try:
            frame = repository.read_bars(selected, start, end, schema="ohlcv-1d")
        except Exception as exc:  # pragma: no cover - corrupt cache is a data error
            logger.warning("cached_real read failed: %s", exc)
            return None
        if frame.is_empty():
            return None
        minimum = 200
        for root in selected:
            rows = frame.filter(pl.col("root_symbol") == root).height
            if rows < minimum:
                return None
        timestamps = tuple(
            sorted(
                {
                    value
                    for value in frame.get_column("timestamp_utc").to_list()
                    if isinstance(value, datetime)
                }
            )
        )
        if not timestamps:
            return None
        latest = (
            frame.sort("timestamp_utc")
            .group_by("root_symbol")
            .agg(pl.col("close").last().alias("close"))
        )
        prices = {
            str(record["root_symbol"]): float(record["close"])
            for record in latest.iter_rows(named=True)
        }
        summary = CachedBarsSummary.model_validate(
            {
                "roots": list(selected),
                "bar_count": frame.height,
                "start_date": timestamps[0].date(),
                "end_date": timestamps[-1].date(),
                "schema": "ohlcv-1d",
            }
        )
        market = fixtures.OfflineMarket(
            instrument_master=master,
            bars=frame,
            timestamps=timestamps,
            prices=prices,
            seed=request.random_seed,
        )
        return market, summary

    def _execute_engine(
        self,
        config: StrategyConfig,
        request: BacktestRunConfig,
        *,
        market: fixtures.OfflineMarket | None = None,
        cached_bars: CachedBarsSummary | None = None,
    ) -> tuple[BacktestRunSummary, TearsheetResponse]:
        from src.api.services import build_pipeline_parts
        from src.backtest.engine import (
            BacktestEngine,
            BacktestMode,
            RebalanceSchedule,
        )

        window = market or fixtures.build_window_market(
            self._master,
            request.start,
            request.end,
            seed=request.random_seed,
            warmup_bars=260,
        )
        states = fixtures.build_rebalance_states(
            window,
            rebalance_frequency=request.rebalance_frequency,
            initial_equity=request.initial_equity,
        )
        if not states:
            raise ValueError("no market states available for the requested window")
        variable = fixture_backtest_config(config)
        pipeline, translator = build_pipeline_parts(
            variable,
            self._master,
            self._settings,
            clock=lambda: self._market.as_of,
            state=states[-1],
        )
        engine = BacktestEngine(
            pipeline,
            translator,
            mode=BacktestMode.VECTORIZED,
            instrument_master=self._master,
            initial_equity=request.initial_equity,
            rebalance_schedule=RebalanceSchedule.from_frequency(request.rebalance_frequency),
        )
        # Warm-up bars are only for feature history: the result curve covers the
        # exact requested window.
        start_moment = datetime.combine(request.start, dt_time.min, tzinfo=UTC)
        end_moment = datetime.combine(request.end, dt_time(23, 59, 59), tzinfo=UTC)
        result = engine.run(states=states, start=start_moment, end=end_moment)
        from src.analytics.metrics import compute_metrics_from_result

        performance = compute_metrics_from_result(result)
        tearsheet = self._engine_tearsheet(result, performance)
        metrics = _performance_metrics(performance)
        summary = BacktestRunSummary(
            id="pending",
            strategy_id=config.id,
            status="completed",
            created_at=self._clock(),
            duration_seconds=None,
            config=request,
            metrics=metrics,
            error=None,
            n_trades=result.n_trades,
            n_rebalances=result.n_rebalances,
            data_source="engine",
            strategy_config=_strategy_config_snapshot(config),
            data_origin="real_parquet" if cached_bars is not None else "synthetic_fixture",
            data_mode=request.data_mode,
            cached_bars=cached_bars,
        )
        return summary, tearsheet

    def _engine_tearsheet(
        self, result: BacktestResult, performance: PerformanceMetrics
    ) -> TearsheetResponse:
        from src.analytics.metrics import rolling_sharpe, underwater_curve

        frame = result.equity_curve
        timestamps = [
            value.astimezone(UTC)
            for value in frame.get_column("timestamp_utc").to_list()
            if isinstance(value, datetime)
        ]
        equity = [float(value) for value in frame.get_column("equity").to_list()]
        if len(timestamps) != len(equity) or not equity:
            raise ValueError("backtest equity curve is empty or inconsistent")
        returns = np.diff(np.asarray(equity, dtype=np.float64)) / np.asarray(
            equity[:-1], dtype=np.float64
        )
        window = min(ROLLING_WINDOW, max(2, returns.size))
        rolling = rolling_sharpe(returns, window) if returns.size >= 2 else np.asarray([])
        rolling_points = [
            SeriesPoint(time=timestamps[index + 1], value=round(float(value), 6))
            for index, value in enumerate(rolling)
            if math.isfinite(float(value)) and index + 1 < len(timestamps)
        ]
        underwater = underwater_curve(equity)
        underwater_points = _series_points(timestamps, [float(value) for value in underwater])
        return TearsheetResponse(
            run_id="pending",
            equity_curve=_series_points(timestamps, equity),
            rolling_sharpe=rolling_points,
            underwater=underwater_points,
            tail_distribution=_tail_distribution(returns),
            roll_yield_attribution=self._attribution(result),
            monthly_returns=_monthly_returns(timestamps, equity),
            metrics=_performance_metrics(performance),
            trades=_trade_rows(result.trades),
        )

    def _attribution(self, result: BacktestResult) -> list[AttributionRow]:
        """Allocate aggregate spot/roll PnL across roots by traded notional.

        The engine reports portfolio-level ``price_pnl`` / ``roll_pnl``; per-root
        attribution is derived from the trade tape (share of traded notional),
        which is the same decomposition used by a roll-yield attribution report.
        """
        traded: dict[str, float] = {}
        for trade in result.trades:
            traded[trade.instrument_id] = traded.get(trade.instrument_id, 0.0) + abs(trade.notional)
        total = sum(traded.values())
        if total <= 0.0 or not traded:
            return []
        rows: list[AttributionRow] = []
        for root in sorted(traded):
            share = traded[root] / total
            rows.append(
                AttributionRow(
                    root=root,
                    spot_pnl=round(result.price_pnl * share, 2),
                    roll_pnl=round(result.roll_pnl * share, 2),
                )
            )
        return rows

    # -- walk-forward execution ------------------------------------------------

    def _run_walk_forward(
        self,
        config: StrategyConfig,
        request: BacktestRunConfig,
        *,
        market: fixtures.OfflineMarket | None = None,
        cached_bars: CachedBarsSummary | None = None,
    ) -> BacktestRunSummary:
        """Execute a purged/embargoed walk-forward run (fixture fallback on error)."""
        started = time.perf_counter()
        try:
            summary, tearsheet = self._execute_walk_forward(
                config, request, market=market, cached_bars=cached_bars
            )
            run_id = self._allocate_run_id()
            duration = time.perf_counter() - started
            stored = summary.model_copy(
                update={
                    "id": run_id,
                    "created_at": self._clock(),
                    "duration_seconds": round(duration, 3),
                }
            )
            tearsheet = tearsheet.model_copy(update={"run_id": run_id})
            self._store_run(stored, tearsheet)
            return stored
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "walk-forward backtest unavailable; deterministic fixture fallback: %s", reason
            )
            return self._fixture_walk_forward(config, request, reason=reason, started=started)

    def _execute_walk_forward(
        self,
        config: StrategyConfig,
        request: BacktestRunConfig,
        *,
        market: fixtures.OfflineMarket | None = None,
        cached_bars: CachedBarsSummary | None = None,
    ) -> tuple[BacktestRunSummary, TearsheetResponse]:
        from src.backtest.engine import run_walk_forward

        window = market or fixtures.build_window_market(
            self._master,
            request.start,
            request.end,
            seed=request.random_seed,
            warmup_bars=260,
        )
        states = fixtures.build_rebalance_states(
            window,
            rebalance_frequency=request.rebalance_frequency,
            initial_equity=request.initial_equity,
        )
        if not states:
            raise ValueError("no market states available for the requested walk-forward window")
        variable = fixture_backtest_config(config)
        report = self._call_walk_forward(run_walk_forward, variable, request, window, states)
        report_dict = _object_mapping(report)
        attributes = _report_attributes(report)
        lookup: list[Mapping[str, object]] = (
            [attributes, report_dict] if attributes else [report_dict]
        )
        fold_payloads = _fold_payloads(report, report_dict)
        fold_count = (
            _first_int(lookup, ("fold_count", "n_folds", "splits"))
            or len(fold_payloads)
            or request.wf_splits
        )
        n_trades = _first_int(lookup, ("total_trades", "n_trades", "trades"))
        n_rebalances = _first_int(lookup, ("total_rebalances", "n_rebalances", "rebalances"))
        curve, timestamps = self._fixture_curve(
            request.strategy_id,
            seed=request.random_seed,
            initial_equity=request.initial_equity,
        )
        values = [value for _, value in curve]
        fallback_metrics = self._metrics_for(values, timestamps)
        metrics = self._walk_forward_metrics(report, report_dict, fallback_metrics)
        folds = [
            _fold_from_payload(index + 1, payload) for index, payload in enumerate(fold_payloads)
        ]
        if not folds:
            folds = _synthesize_walk_forward_folds(timestamps, values, fold_count)
        fold_count = max(fold_count, len(folds))
        combined = self._combined_equity(report, report_dict)
        drawdown = _series_from_curve_mapping(report_dict, "drawdown")
        folds = _merge_combined_folds(folds, combined, drawdown, fold_count)
        base = self._fixture_tearsheet("pending", timestamps, values, metrics)
        combined_result = getattr(report, "combined_result", None)
        walk_forward_trades = _trade_rows(
            cast(Sequence[object], getattr(combined_result, "trades", ()) or ())
        )
        tearsheet = base.model_copy(
            update={
                "equity_curve": combined or base.equity_curve,
                "underwater": drawdown or base.underwater,
                "folds": folds,
                "trades": walk_forward_trades,
            }
        )
        summary = BacktestRunSummary(
            id="pending",
            strategy_id=config.id,
            status="completed",
            created_at=self._clock(),
            duration_seconds=None,
            config=request,
            metrics=metrics,
            error=None,
            n_trades=n_trades if n_trades is not None else 0,
            n_rebalances=n_rebalances if n_rebalances is not None else fold_count,
            data_source="engine",
            kind="walk_forward",
            n_folds=fold_count,
            strategy_config=_strategy_config_snapshot(config),
            data_origin="real_parquet" if cached_bars is not None else "synthetic_fixture",
            data_mode=request.data_mode,
            cached_bars=cached_bars,
        )
        return summary, tearsheet

    def _call_walk_forward(
        self,
        runner: object,
        config: StrategyConfig,
        request: BacktestRunConfig,
        window: fixtures.OfflineMarket,
        states: Sequence[object],
    ) -> object:
        """Call the (versioned) ``run_walk_forward`` helper, adapting keyword names."""
        function = runner
        if not callable(function):  # pragma: no cover - defensive
            raise TypeError("run_walk_forward is not callable")
        try:
            signature = inspect.signature(function)
        except (TypeError, ValueError):  # pragma: no cover - builtins only
            return function(config)
        candidates: dict[str, object] = {
            "config": config,
            "strategy_config": config,
            "request": request,
            "market": window,
            "window": window,
            "offline_market": window,
            "states": states,
            "rebalance_states": states,
            "bars": window.bars,
            "frame": window.bars,
            "instrument_master": self._master,
            "instruments": self._master,
            "initial_equity": request.initial_equity,
            "equity": request.initial_equity,
            "rebalance_frequency": request.rebalance_frequency,
            "splits": request.wf_splits,
            "n_splits": request.wf_splits,
            "wf_splits": request.wf_splits,
            "purge_bars": request.purge_bars,
            "purge": request.purge_bars,
            "embargo_bars": request.embargo_bars,
            "embargo": request.embargo_bars,
            "seed": request.random_seed,
            "random_seed": request.random_seed,
            "start": datetime.combine(request.start, dt_time.min, tzinfo=UTC),
            "end": datetime.combine(request.end, dt_time(23, 59, 59), tzinfo=UTC),
        }
        kwargs: dict[str, object] = {}
        takes_config_keyword = False
        for name, parameter in signature.parameters.items():
            if name == "self":
                continue
            if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                continue
            if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
                continue
            if name in ("config", "strategy_config"):
                takes_config_keyword = True
            if name in candidates:
                kwargs[name] = candidates[name]
            elif parameter.default is inspect.Parameter.empty:
                raise TypeError(
                    f"run_walk_forward requires unsupported parameter {name!r}; "
                    "cannot adapt to this helper version"
                )
        if takes_config_keyword:
            return function(**kwargs)
        return function(config, **kwargs)

    def _walk_forward_metrics(
        self,
        report: object,
        report_dict: Mapping[str, object],
        fallback: MetricsSet,
    ) -> MetricsSet:
        mappings = _metric_mappings(report, report_dict)
        attributes = _report_attributes(report)
        if attributes:
            mappings.append(attributes)
        return _metrics_with_defaults(mappings, fallback)

    def _combined_equity(
        self, report: object, report_dict: Mapping[str, object]
    ) -> list[SeriesPoint]:
        points = _series_from_curve_mapping(report_dict, "equity")
        if points:
            return points
        keys = (
            "equity_curve",
            "combined_equity_curve",
            "portfolio_equity_curve",
            "equity",
            "combined_equity",
        )
        attributes = _object_mapping(report)
        return _series_points_from_payload(attributes, keys)

    def _fixture_walk_forward(
        self,
        config: StrategyConfig,
        request: BacktestRunConfig,
        *,
        reason: str,
        started: float | None = None,
    ) -> BacktestRunSummary:
        """Deterministic walk-forward fixture fallback with explicit error text."""
        run_id = self._allocate_run_id()
        duration = time.perf_counter() - started if started is not None else 0.05
        curve, timestamps = self._fixture_curve(
            request.strategy_id,
            seed=request.random_seed,
            initial_equity=request.initial_equity,
        )
        values = [value for _, value in curve]
        metrics = self._metrics_for(values, timestamps)
        folds = _synthesize_walk_forward_folds(timestamps, values, request.wf_splits)
        tearsheet = self._fixture_tearsheet(run_id, timestamps, values, metrics)
        tearsheet = tearsheet.model_copy(update={"folds": folds})
        summary = BacktestRunSummary(
            id=run_id,
            strategy_id=config.id,
            status="completed",
            created_at=self._clock(),
            duration_seconds=round(duration, 3),
            config=request,
            metrics=metrics,
            error=f"deterministic fixture walk-forward fallback: {reason}",
            n_trades=request.wf_splits * 3,
            n_rebalances=request.wf_splits,
            data_source="fixture",
            kind="walk_forward",
            n_folds=request.wf_splits,
            strategy_config=self._config_snapshot_for(request.strategy_id),
            data_mode=request.data_mode,
            data_origin="synthetic_fixture",
        )
        self._store_run(summary, tearsheet)
        return summary

    # -- fixture fallback ------------------------------------------------------

    def _fixture_run(
        self,
        request: BacktestRunConfig,
        *,
        reason: str,
        started: float | None = None,
    ) -> BacktestRunSummary:
        run_id = self._allocate_run_id()
        duration = time.perf_counter() - started if started is not None else 0.05
        curve, timestamps = self._fixture_curve(
            request.strategy_id, seed=request.random_seed, initial_equity=request.initial_equity
        )
        values = [value for _, value in curve]
        metrics = self._metrics_for(values, timestamps)
        tearsheet = self._fixture_tearsheet(run_id, timestamps, values, metrics)
        summary = BacktestRunSummary(
            id=run_id,
            strategy_id=request.strategy_id,
            status="completed",
            created_at=self._clock(),
            duration_seconds=round(duration, 3),
            config=request,
            metrics=metrics,
            error=f"deterministic fixture fallback: {reason}",
            n_trades=None,
            n_rebalances=None,
            data_source="fixture",
            kind="backtest",
            strategy_config=self._config_snapshot_for(request.strategy_id),
            data_mode=request.data_mode,
            data_origin="synthetic_fixture",
        )
        self._store_run(summary, tearsheet)
        return summary

    def _seed_fixture_runs(self) -> None:
        default_requests = self._default_requests()
        for offset, request in enumerate(default_requests):
            run_id = self._allocate_run_id()
            curve, timestamps = self._fixture_curve(
                request.strategy_id, seed=request.random_seed, initial_equity=request.initial_equity
            )
            values = [value for _, value in curve]
            metrics = self._metrics_for(values, timestamps)
            tearsheet = self._fixture_tearsheet(run_id, timestamps, values, metrics)
            created = self._clock() - timedelta(days=offset + 1)
            summary = BacktestRunSummary(
                id=run_id,
                strategy_id=request.strategy_id,
                status="completed",
                created_at=created,
                duration_seconds=round(84.0 + 37.0 * offset, 1),
                config=request,
                metrics=metrics,
                error=None,
                n_trades=None,
                n_rebalances=None,
                data_source="fixture",
                strategy_config=self._config_snapshot_for(request.strategy_id),
            )
            self._runs[run_id] = summary
            self._tearsheets[run_id] = tearsheet

    def _default_requests(self) -> list[BacktestRunConfig]:
        requests: list[BacktestRunConfig] = []
        for strategy_id in self._catalog.available_ids()[:SEED_RUNS]:
            config = self._catalog.core_config(strategy_id)
            if config is None:
                continue
            spec = config.backtest
            requests.append(
                BacktestRunConfig(
                    strategy_id=strategy_id,
                    start=spec.start if spec is not None else self._market.timestamps[0].date(),
                    end=spec.end if spec is not None else self._market.as_of.date(),
                    initial_equity=spec.initial_equity
                    if spec is not None
                    else fixtures.DEFAULT_INITIAL_EQUITY,
                    rebalance_frequency=spec.rebalance_frequency if spec is not None else "W-FRI",
                    walk_forward=spec.walk_forward is not None if spec is not None else True,
                    random_seed=spec.random_seed if spec is not None else 7,
                )
            )
        return requests

    def _fixture_curve(
        self,
        strategy_id: str,
        *,
        seed: int,
        initial_equity: float,
    ) -> tuple[list[tuple[datetime, float]], list[datetime]]:
        timestamps = list(self._market.timestamps[-504:])
        rng = np.random.default_rng(seed + sum(ord(char) for char in strategy_id))
        returns = rng.normal(0.00045, 0.0068, size=len(timestamps))
        values = initial_equity * np.cumprod(1.0 + returns)
        return (
            [
                (timestamp, float(value))
                for timestamp, value in zip(timestamps, values, strict=True)
            ],
            timestamps,
        )

    def _metrics_for(self, values: Sequence[float], timestamps: Sequence[datetime]) -> MetricsSet:
        try:
            from src.analytics.metrics import compute_metrics

            performance = compute_metrics(
                [float(value) for value in values],
                timestamps=list(timestamps),
                initial_equity=float(values[0]) if values else None,
            )
        except Exception:
            return _simple_metrics(values)
        return _performance_metrics(performance)

    def _fixture_tearsheet(
        self,
        run_id: str,
        timestamps: Sequence[datetime],
        values: Sequence[float],
        metrics: MetricsSet,
    ) -> TearsheetResponse:
        equity = np.asarray([float(value) for value in values], dtype=np.float64)
        returns = (
            np.diff(equity) / equity[:-1] if equity.size > 1 else np.asarray([], dtype=np.float64)
        )
        rolling: list[SeriesPoint] = []
        underwater_points: list[SeriesPoint] = []
        try:
            from src.analytics.metrics import rolling_sharpe, underwater_curve

            window = min(ROLLING_WINDOW, max(2, returns.size))
            if returns.size >= 2:
                series = rolling_sharpe(returns, window)
                rolling = [
                    SeriesPoint(time=timestamps[index + 1], value=round(float(value), 6))
                    for index, value in enumerate(series)
                    if math.isfinite(float(value)) and index + 1 < len(timestamps)
                ]
            underwater_points = _series_points(
                list(timestamps), [float(value) for value in underwater_curve(equity)]
            )
        except Exception:
            peaks = np.maximum.accumulate(equity) if equity.size else equity
            underwater_points = _series_points(
                list(timestamps),
                [float(value) for value in (equity / peaks - 1.0 if equity.size else [])],
            )
        attribution = self._fixture_attribution(timestamps, returns)
        return TearsheetResponse(
            run_id=run_id,
            equity_curve=_series_points(list(timestamps), list(equity)),
            rolling_sharpe=rolling,
            underwater=underwater_points,
            tail_distribution=_tail_distribution(returns),
            roll_yield_attribution=attribution,
            monthly_returns=_monthly_returns(list(timestamps), list(equity)),
            metrics=metrics,
        )

    def _fixture_attribution(
        self, timestamps: Sequence[datetime], returns: np.ndarray
    ) -> list[AttributionRow]:
        del timestamps
        gross_return = float(np.sum(returns)) if returns.size else 0.0
        anchor = fixtures.DEFAULT_INITIAL_EQUITY * 0.22 * gross_return
        rows: list[AttributionRow] = []
        for index, root in enumerate(("ES", "NQ", "CL", "GC", "ZN")):
            share = 1.0 / (index + 1)
            rows.append(
                AttributionRow(
                    root=root,
                    spot_pnl=round(anchor * share * 1.6, 2),
                    roll_pnl=round(anchor * share * (0.35 if index % 2 else -0.22), 2),
                )
            )
        return rows

    # -- deletion --------------------------------------------------------------

    def artifact_path(self, run_id: str) -> Path:
        """Persisted artifact path for one run id (may not exist for seed runs)."""
        return self._artifact_path(run_id)

    def _config_snapshot_for(self, strategy_id: str) -> dict[str, object] | None:
        """Best-effort strategy-config snapshot for fixture/seeded runs."""
        config = self._catalog.core_config(strategy_id)
        return _strategy_config_snapshot(config) if config is not None else None

    def delete_run(self, run_id: str) -> bool:
        """Delete one stored run + its persisted artifact.

        Returns whether a persisted artifact file was removed. Raises ``KeyError``
        for unknown ids. Seeded fixture runs live in memory only (their artifact is
        not persisted) and reappear after an API restart; this is documented on the
        DELETE route.
        """
        summary = self._runs.pop(run_id, None)
        if summary is None:
            raise KeyError(f"unknown backtest run {run_id!r}")
        self._tearsheets.pop(run_id, None)
        self._persisted_ids.discard(run_id)
        path = self._artifact_path(run_id)
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return False
        except OSError:  # pragma: no cover - best effort artifact cleanup
            logger.warning("could not delete backtest artifact %s", path)
            return False

    # -- persistence -----------------------------------------------------------

    def _artifact_path(self, run_id: str) -> Path:
        return self._artifacts_dir / f"{run_id}.json"

    def _load_persisted_runs(self) -> None:
        """Reload custom runs written by a previous API process (newest-first list)."""
        directory = self._artifacts_dir
        if not directory.is_dir():
            return
        for path in sorted(directory.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                self._quarantine_artifact(path, exc)
                continue
            if not isinstance(payload, Mapping):
                self._quarantine_artifact(path, ValueError("artifact payload is not a JSON object"))
                continue
            mapping = cast(Mapping[str, object], payload)
            try:
                summary = BacktestRunSummary.model_validate(mapping["summary"])
                tearsheet = TearsheetResponse.model_validate(mapping["tearsheet"])
            except (KeyError, TypeError, ValidationError) as exc:
                self._quarantine_artifact(path, exc)
                continue
            if summary.id in self._runs:
                continue
            self._runs[summary.id] = summary
            self._tearsheets[summary.id] = tearsheet
            self._persisted_ids.add(summary.id)
            number = _run_id_number(summary.id)
            if number is not None:
                self._next_id = max(self._next_id, number)
        if self._persisted_ids:
            logger.info(
                "loaded %d persisted backtest run(s) from %s",
                len(self._persisted_ids),
                directory,
            )

    def _quarantine_artifact(self, path: Path, exc: BaseException) -> None:
        warning = f"corrupt backtest artifact {path.name}: {type(exc).__name__}: {exc}; skipped"
        logger.warning(warning)
        self._warnings.append(warning)
        try:
            path.replace(path.with_name(f"{path.name}.corrupt-{int(time.time())}"))
        except OSError:  # pragma: no cover - best-effort quarantine
            logger.exception("failed to quarantine corrupt backtest artifact %s", path)

    def _persist_run(self, summary: BacktestRunSummary, tearsheet: TearsheetResponse) -> None:
        """Atomically persist one custom run (temp file + ``os.replace``)."""
        try:
            directory = self._artifacts_dir
            directory.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "summary": summary.model_dump(mode="json"),
                "tearsheet": tearsheet.model_dump(mode="json"),
                "request": summary.config.model_dump(mode="json"),
            }
            fd, temp_name = tempfile.mkstemp(
                prefix=f".{summary.id}.", suffix=".json.tmp", dir=str(directory)
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, sort_keys=True)
                    handle.flush()
                    os.fsync(handle.fileno())
                Path(temp_name).replace(self._artifact_path(summary.id))
            except BaseException:
                Path(temp_name).unlink(missing_ok=True)
                raise
            self._persisted_ids.add(summary.id)
        except Exception as exc:  # the artifact store must never fail the API run
            warning = f"could not persist backtest {summary.id}: {type(exc).__name__}: {exc}"
            logger.warning(warning)
            self._warnings.append(warning)

    def _store_run(
        self,
        summary: BacktestRunSummary,
        tearsheet: TearsheetResponse,
        *,
        persist: bool = True,
    ) -> None:
        self._runs[summary.id] = summary
        self._tearsheets[summary.id] = tearsheet
        self._trim_store()
        if persist:
            self._persist_run(summary, tearsheet)

    # -- store -----------------------------------------------------------------

    def _allocate_run_id(self) -> str:
        while True:
            self._next_id += 1
            candidate = f"bt-{self._next_id:04d}"
            if candidate not in self._runs:
                return candidate

    def _trim_store(self) -> None:
        if len(self._runs) <= MAX_RUNS:
            return
        ordered = sorted(self._runs.values(), key=lambda run: run.created_at)
        for stale in ordered[: len(self._runs) - MAX_RUNS]:
            self._runs.pop(stale.id, None)
            self._tearsheets.pop(stale.id, None)
            if stale.id in self._persisted_ids:
                self._persisted_ids.discard(stale.id)
                try:
                    self._artifact_path(stale.id).unlink(missing_ok=True)
                except OSError:  # pragma: no cover - best-effort cleanup
                    logger.warning("could not delete trimmed backtest artifact %s", stale.id)
