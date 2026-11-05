"""Institutional futures performance metrics.

``compute_metrics`` turns an equity curve (plus optional margin, execution and
attribution inputs) into an immutable :class:`PerformanceMetrics` object that the
tear-sheet generator renders and the API serves. Every ratio has a *documented,
finite* convention for degenerate inputs -- the analytics layer never emits
``NaN`` or ``inf`` into JSON:

* zero return variance -> ``sharpe = 0.0`` (zero-risk sentinel);
* no downside observations -> ``sortino = 0.0``, ``omega = 1.0`` (undefined
  ratios are neutral, never infinite);
* no drawdown -> ``calmar = 0.0``;
* no loss tail -> ``var = cvar = 0.0`` and ``rachev_ratio = 0.0``;
* no margin deployed -> ``return_on_max_initial_margin = 0.0``;
* non-positive gross PnL -> ``commission_to_gross_pnl_ratio = 0.0``;
* fewer than two equity observations -> ``ValueError`` (a curve needs at least one
  return to define any statistic).

Formulas
--------
* CAGR ``(ending / initial) ** (1 / years) - 1`` with ``years = n_returns / ppy``.
* Sharpe ``(mean(r) - rf_daily) / std(r) * sqrt(ppy)`` (sample stdev).
* Sortino uses downside deviation ``sqrt(mean(min(r - rf_daily, 0)^2))``.
* Calmar ``CAGR / |max drawdown|``; max drawdown is reported as a *positive*
  fraction (``0.25`` means -25%).
* VaR/CVaR are positive loss fractions at the 95% / 99% confidence levels;
  parametric values use a Gaussian fit (``statistics.NormalDist``).
* Rachev ratio at ``alpha = 0.05`` is ``mean(upper 5% returns) / |mean(lower 5%
  returns)|`` -- the tail-gain-to-tail-loss ratio that validates the LLM
  Risk-First circuit breaker.
* Roll yield attribution: ``roll_yield_annualized = (roll_pnl / initial) / years``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from datetime import datetime
from statistics import NormalDist
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from numpy.typing import NDArray

from src.core.types import TRADING_DAYS_PER_YEAR, ensure_utc
from src.data.schema import TIMESTAMP_COLUMN

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from src.backtest.engine import BacktestResult

#: Alpha used by the Rachev tail ratio (5% upper / lower tails).
RACHEV_ALPHA: float = 0.05

#: Fraction of periods used for the "upper/lower tail" conventions.
DEFAULT_ANNUALIZATION: int = TRADING_DAYS_PER_YEAR

#: Canonical timeframe -> bars/year, used only while ``src.data.timeframes`` (task-67)
#: is unavailable; the registry is authoritative when importable.
_TIMEFRAME_BARS_PER_YEAR_FALLBACK: dict[str, int] = {
    "1m": 347_760,
    "5m": 69_552,
    "15m": 23_184,
    "30m": 11_592,
    "1h": 5_796,
    "4h": 1_449,
    "1d": 252,
    "1w": 52,
}


def canonical_timeframe(timeframe: str) -> str:
    """Canonical timeframe name (``"1d"``) validated via ``src.data.timeframes``.

    Falls back to the local canonical table while the data registry is missing.
    """
    if not isinstance(timeframe, str) or not timeframe.strip():
        raise ValueError("timeframe must be a non-empty string")
    key = timeframe.strip().lower()
    try:
        from src.data.timeframes import get_timeframe

        return get_timeframe(key).name
    except ImportError:
        pass
    if key in _TIMEFRAME_BARS_PER_YEAR_FALLBACK:
        return key
    supported = ", ".join(sorted(_TIMEFRAME_BARS_PER_YEAR_FALLBACK))
    raise ValueError(f"unknown timeframe {timeframe!r}; supported: {supported}")


def resolve_periods_per_year(
    timeframe: str = "1d",
    periods_per_year: int | None = None,
) -> int:
    """Resolve bars/year from a timeframe, honouring an explicit override.

    ``periods_per_year`` wins when supplied (legacy callers/tests keep exact
    control); otherwise :func:`src.data.timeframes.bars_per_year` is used, with a
    local canonical fallback if the data registry is not importable. ``1d``
    resolves to the historical 252 exactly.
    """
    if periods_per_year is not None:
        if not isinstance(periods_per_year, int) or isinstance(periods_per_year, bool):
            raise ValueError("periods_per_year must be a positive int")
        if periods_per_year < 1:
            raise ValueError("periods_per_year must be a positive int")
        return periods_per_year
    key = canonical_timeframe(timeframe)
    try:
        from src.data.timeframes import bars_per_year

        resolved = bars_per_year(key)
        if isinstance(resolved, int) and resolved >= 1:
            return resolved
    except ImportError:
        pass
    return _TIMEFRAME_BARS_PER_YEAR_FALLBACK[key]


_NORMAL = NormalDist()

#: Any dispersion below this is numerically zero (constant series convention).
_ZERO_VARIANCE_TOLERANCE: float = 1e-12


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    """Immutable, JSON-serialisable performance/risk/attribution metric set."""

    n_periods: int
    periods_per_year: int
    start: datetime | None
    end: datetime | None
    initial_equity: float
    ending_equity: float
    net_pnl: float
    total_return: float
    cagr: float
    annualized_volatility: float
    sharpe: float
    sortino: float
    calmar: float
    omega: float
    max_drawdown: float
    max_drawdown_duration: int
    max_drawdown_duration_days: float
    var_95_historical: float
    var_99_historical: float
    var_95_parametric: float
    var_99_parametric: float
    cvar_95: float
    cvar_99: float
    cvar_95_parametric: float
    cvar_99_parametric: float
    rachev_ratio_05: float
    return_on_max_initial_margin: float
    average_margin_utilization: float
    peak_margin_utilization: float
    max_margin_used: float
    implementation_shortfall_ticks: float
    turnover: float
    annualized_turnover: float
    commission_to_gross_pnl_ratio: float
    price_pnl: float
    roll_pnl: float
    funding: float
    gross_pnl: float
    spot_pnl_share: float
    roll_pnl_share: float
    roll_yield_annualized: float
    risk_free_rate: float = 0.0
    information_ratio: float | None = None
    beta: float | None = None
    alpha: float | None = None
    benchmark_total_return: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    timeframe: str = "1d"

    def __post_init__(self) -> None:
        for metric in fields(self):
            value = getattr(self, metric.name)
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"metric {metric.name!r} must be finite, got {value!r}")
        if self.n_periods < 1:
            raise ValueError("n_periods must be at least 1")
        if self.periods_per_year < 1:
            raise ValueError("periods_per_year must be at least 1")

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable representation of every metric."""
        result: dict[str, object] = {}
        for metric in fields(self):
            value = getattr(self, metric.name)
            if isinstance(value, datetime):
                result[metric.name] = value.isoformat()
            else:
                result[metric.name] = value
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_equity(
    equity_curve: pl.DataFrame | pl.Series | Sequence[float] | NDArray[np.float64],
) -> tuple[NDArray[np.float64], tuple[datetime, ...] | None]:
    if isinstance(equity_curve, pl.DataFrame):
        if "equity" not in equity_curve.columns:
            raise ValueError("equity_curve DataFrame must contain an 'equity' column")
        values = np.asarray(equity_curve.get_column("equity").to_list(), dtype=np.float64)
        timestamps: tuple[datetime, ...] | None = None
        if TIMESTAMP_COLUMN in equity_curve.columns:
            timestamps = tuple(
                ensure_utc(value) for value in equity_curve.get_column(TIMESTAMP_COLUMN).to_list()
            )
        return values, timestamps
    if isinstance(equity_curve, pl.Series):
        return np.asarray(equity_curve.to_list(), dtype=np.float64), None
    return np.asarray(list(equity_curve), dtype=np.float64), None


def _series_from_frame(frame: pl.DataFrame, column: str) -> NDArray[np.float64] | None:
    if not isinstance(frame, pl.DataFrame) or column not in frame.columns:
        return None
    return np.asarray(frame.get_column(column).to_list(), dtype=np.float64)


def _validate_equity(equity: NDArray[np.float64]) -> None:
    if equity.ndim != 1:
        raise ValueError("equity curve must be one-dimensional")
    if equity.size < 2:
        raise ValueError("equity curve must contain at least two observations to compute metrics")
    if not np.all(np.isfinite(equity)):
        raise ValueError("equity curve contains non-finite values")
    if np.any(equity <= 0.0):
        raise ValueError("equity curve must be strictly positive")


def _max_drawdown_duration(equity: NDArray[np.float64]) -> tuple[float, int]:
    """Return ``(max_drawdown_positive_fraction, max_duration_in_periods)``."""
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1.0
    max_drawdown = float(max(0.0, -float(np.min(drawdown))))
    longest = 0
    current = 0
    for index in range(1, equity.size):
        if equity[index] < peak[index]:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return max_drawdown, int(longest)


def _historical_var(returns: NDArray[np.float64], confidence: float) -> float:
    if returns.size < 2:
        return 0.0
    quantile = float(np.quantile(returns, 1.0 - confidence))
    return max(0.0, -quantile)


def _historical_cvar(returns: NDArray[np.float64], confidence: float) -> float:
    if returns.size < 2:
        return 0.0
    quantile = float(np.quantile(returns, 1.0 - confidence))
    tail = returns[returns <= quantile]
    if tail.size == 0:
        return _historical_var(returns, confidence)
    return max(0.0, -float(np.mean(tail)))


def _parametric_var(mean: float, std: float, confidence: float) -> float:
    if std <= 0.0:
        return 0.0
    z_score = _NORMAL.inv_cdf(1.0 - confidence)
    return max(0.0, -(mean + z_score * std))


def _parametric_cvar(mean: float, std: float, confidence: float) -> float:
    if std <= 0.0:
        return 0.0
    tail_probability = 1.0 - confidence
    z_score = _NORMAL.inv_cdf(tail_probability)
    expected_shortfall = -(mean - std * _NORMAL.pdf(z_score) / tail_probability)
    return max(0.0, float(expected_shortfall))


def _rachev_ratio(returns: NDArray[np.float64], alpha: float = RACHEV_ALPHA) -> float:
    if returns.size < 2:
        return 0.0
    lower_threshold = float(np.quantile(returns, alpha))
    upper_threshold = float(np.quantile(returns, 1.0 - alpha))
    lower_tail = returns[returns <= lower_threshold]
    upper_tail = returns[returns >= upper_threshold]
    if lower_tail.size == 0 or upper_tail.size == 0:
        return 0.0
    lower_mean = float(np.mean(lower_tail))
    upper_mean = float(np.mean(upper_tail))
    if lower_mean >= 0.0:
        # No loss tail: the ratio is undefined, returned as the neutral sentinel.
        return 0.0
    return max(0.0, upper_mean / abs(lower_mean))


def _safe_ratio(numerator: float, denominator: float, *, neutral: float = 0.0) -> float:
    if denominator == 0.0 or not math.isfinite(denominator):
        return neutral
    value = numerator / denominator
    return float(value) if math.isfinite(value) else neutral


def _annualized_return(initial: float, ending: float, years: float) -> float:
    if years <= 0.0 or initial <= 0.0:
        return 0.0
    growth = ending / initial
    if growth <= 0.0:
        return 0.0
    try:
        return float(growth ** (1.0 / years) - 1.0)
    except OverflowError:
        # Ultra-short intraday samples can imply absurd annualised growth; return
        # the largest finite float so the frozen metric contract stays finite/JSON-safe.
        return float(np.finfo(np.float64).max)


def compute_metrics(
    equity_curve: pl.DataFrame | pl.Series | Sequence[float] | NDArray[np.float64],
    *,
    timestamps: Sequence[datetime] | None = None,
    initial_equity: float | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int | None = None,
    timeframe: str = "1d",
    margin_used: Sequence[float] | pl.Series | NDArray[np.float64] | None = None,
    margin_utilization: Sequence[float] | pl.Series | NDArray[np.float64] | None = None,
    trade_shortfall_ticks: Sequence[float] | None = None,
    traded_notional: float | Sequence[float] | None = None,
    commissions: float = 0.0,
    price_pnl: float = 0.0,
    roll_pnl: float = 0.0,
    funding: float = 0.0,
    benchmark_returns: Sequence[float] | None = None,
) -> PerformanceMetrics:
    """Compute the full institutional metric set for an equity curve.

    ``equity_curve`` may be a canonical equity-curve ``DataFrame`` (with
    ``equity``/``timestamp_utc`` and optional ``margin_used`` /
    ``margin_utilization`` / ``turnover`` columns), a ``pl.Series``, a NumPy
    array or a plain sequence of equity values.
    """
    if not math.isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be finite")
    if periods_per_year is None:
        resolved_timeframe = canonical_timeframe(timeframe)
    else:
        try:
            resolved_timeframe = canonical_timeframe(timeframe)
        except ValueError:
            # Explicit ppy override wins; keep the raw label for metadata only.
            resolved_timeframe = (
                timeframe.strip().lower() if isinstance(timeframe, str) else timeframe
            )
    periods_per_year = resolve_periods_per_year(timeframe, periods_per_year)

    frame: pl.DataFrame | None = equity_curve if isinstance(equity_curve, pl.DataFrame) else None
    equity, frame_timestamps = _extract_equity(equity_curve)
    _validate_equity(equity)
    resolved_timestamps: tuple[datetime, ...] | None = None
    if timestamps is not None:
        resolved_timestamps = tuple(ensure_utc(value) for value in timestamps)
        if len(resolved_timestamps) != equity.size:
            raise ValueError("timestamps length must match the equity curve length")
    elif frame_timestamps is not None:
        resolved_timestamps = frame_timestamps

    if margin_used is None and frame is not None:
        margin_used = _series_from_frame(frame, "margin_used")
    if margin_utilization is None and frame is not None:
        margin_utilization = _series_from_frame(frame, "margin_utilization")
    if traded_notional is None and frame is not None:
        turnover_column = _series_from_frame(frame, "turnover")
        if turnover_column is not None:
            traded_notional = turnover_column.tolist()

    starting = float(initial_equity) if initial_equity is not None else float(equity[0])
    if not math.isfinite(starting) or starting <= 0.0:
        raise ValueError("initial_equity must be finite and strictly positive")
    ending = float(equity[-1])
    returns = np.diff(equity) / equity[:-1]
    n_returns = int(returns.size)
    years = n_returns / periods_per_year
    rf_daily = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0

    mean_return = float(np.mean(returns))
    std_return = float(np.std(returns, ddof=1)) if n_returns > 1 else 0.0
    excess = returns - rf_daily
    excess_mean = float(np.mean(excess))
    annualized_volatility = std_return * math.sqrt(periods_per_year)
    sharpe = (
        excess_mean / std_return * math.sqrt(periods_per_year)
        if std_return > _ZERO_VARIANCE_TOLERANCE
        else 0.0
    )
    downside = np.minimum(excess, 0.0)
    downside_deviation = float(np.sqrt(np.mean(downside**2))) if downside.size else 0.0
    sortino = (
        excess_mean / downside_deviation * math.sqrt(periods_per_year)
        if downside_deviation > _ZERO_VARIANCE_TOLERANCE
        else 0.0
    )
    max_drawdown, max_drawdown_duration = _max_drawdown_duration(equity)
    cagr = _annualized_return(starting, ending, years)
    calmar = _safe_ratio(cagr, max_drawdown)
    gains = float(np.sum(np.maximum(returns - rf_daily, 0.0)))
    losses = float(np.sum(np.maximum(rf_daily - returns, 0.0)))
    omega = gains / losses if losses > 0.0 else 1.0
    if resolved_timestamps is not None and equity.size > 1:
        span_days = (resolved_timestamps[-1] - resolved_timestamps[0]).total_seconds() / 86_400.0
        period_days = span_days / (equity.size - 1)
    else:
        period_days = 365.25 / periods_per_year
    max_drawdown_days = float(max_drawdown_duration) * period_days

    mean_for_params = float(np.mean(returns))
    std_for_params = float(np.std(returns, ddof=0))
    var_95_historical = _historical_var(returns, 0.95)
    var_99_historical = _historical_var(returns, 0.99)
    var_95_parametric = _parametric_var(mean_for_params, std_for_params, 0.95)
    var_99_parametric = _parametric_var(mean_for_params, std_for_params, 0.99)
    cvar_95 = _historical_cvar(returns, 0.95)
    cvar_99 = _historical_cvar(returns, 0.99)
    cvar_95_parametric = _parametric_cvar(mean_for_params, std_for_params, 0.95)
    cvar_99_parametric = _parametric_cvar(mean_for_params, std_for_params, 0.99)

    # -- margin efficiency -----------------------------------------------------
    margin_values: NDArray[np.float64] | None = None
    if margin_used is not None:
        margin_values = np.asarray(list(margin_used), dtype=np.float64)
        if margin_values.size not in (0, equity.size):
            raise ValueError("margin_used length must match the equity curve length")
    utilization_values: NDArray[np.float64] | None = None
    if margin_utilization is not None:
        utilization_values = np.asarray(list(margin_utilization), dtype=np.float64)
        if utilization_values.size not in (0, equity.size):
            raise ValueError("margin_utilization length must match the equity curve length")
    elif margin_values is not None and margin_values.size == equity.size:
        utilization_values = margin_values / equity
    max_margin = (
        float(np.max(margin_values)) if margin_values is not None and margin_values.size else 0.0
    )
    average_utilization = (
        float(np.mean(utilization_values))
        if utilization_values is not None and utilization_values.size
        else 0.0
    )
    peak_utilization = (
        float(np.max(utilization_values))
        if utilization_values is not None and utilization_values.size
        else 0.0
    )
    net_pnl = ending - starting
    return_on_margin = _safe_ratio(net_pnl, max_margin)

    # -- execution quality -----------------------------------------------------
    shortfall_values = (
        np.asarray(list(trade_shortfall_ticks), dtype=np.float64)
        if trade_shortfall_ticks is not None
        else np.array([], dtype=np.float64)
    )
    implementation_shortfall = float(np.mean(shortfall_values)) if shortfall_values.size else 0.0
    if traded_notional is None:
        total_traded = 0.0
    elif isinstance(traded_notional, (int, float)):
        total_traded = float(traded_notional)
    else:
        total_traded = float(np.sum(np.asarray(list(traded_notional), dtype=np.float64)))
    mean_equity = float(np.mean(equity))
    turnover = _safe_ratio(total_traded, mean_equity)
    annualized_turnover = turnover * periods_per_year / n_returns if n_returns else 0.0
    gross_pnl = float(price_pnl) + float(roll_pnl)
    # A non-positive gross PnL makes the ratio meaningless; use the neutral sentinel.
    commission_ratio = _safe_ratio(float(commissions), gross_pnl) if gross_pnl > 0.0 else 0.0
    attribution_scale = abs(float(price_pnl)) + abs(float(roll_pnl))
    spot_share = float(price_pnl) / attribution_scale if attribution_scale > 0.0 else 0.0
    roll_share = float(roll_pnl) / attribution_scale if attribution_scale > 0.0 else 0.0
    roll_yield = float(roll_pnl) / starting / years if years > 0.0 and starting > 0.0 else 0.0

    # -- optional benchmark ----------------------------------------------------
    information_ratio: float | None = None
    beta: float | None = None
    alpha: float | None = None
    benchmark_total_return: float | None = None
    if benchmark_returns is not None:
        benchmark = np.asarray(list(benchmark_returns), dtype=np.float64)
        if benchmark.size != n_returns:
            raise ValueError("benchmark_returns length must match the number of returns")
        active = returns - benchmark
        benchmark_total_return = float(np.prod(1.0 + benchmark) - 1.0)
        active_std = float(np.std(active, ddof=1)) if active.size > 1 else 0.0
        information_ratio = (
            float(np.mean(active)) / active_std * math.sqrt(periods_per_year)
            if active_std > _ZERO_VARIANCE_TOLERANCE
            else 0.0
        )
        benchmark_variance = float(np.var(benchmark, ddof=1)) if benchmark.size > 1 else 0.0
        if benchmark_variance > 0.0:
            covariance = float(np.cov(returns, benchmark, ddof=1)[0, 1])
            beta = covariance / benchmark_variance
            alpha = (mean_return - float(beta) * float(np.mean(benchmark))) * periods_per_year
        else:
            beta = 0.0
            alpha = mean_return * periods_per_year

    metadata: dict[str, object] = {
        "conventions": "zero-variance ratios -> 0; no downside -> sortino 0, omega 1; "
        "zero drawdown -> calmar 0; VaR/CVaR are positive loss fractions",
        "rachev_alpha": RACHEV_ALPHA,
        "timeframe": resolved_timeframe,
        "periods_per_year": periods_per_year,
    }
    return PerformanceMetrics(
        n_periods=n_returns,
        periods_per_year=periods_per_year,
        start=resolved_timestamps[0] if resolved_timestamps else None,
        end=resolved_timestamps[-1] if resolved_timestamps else None,
        initial_equity=starting,
        ending_equity=ending,
        net_pnl=net_pnl,
        total_return=net_pnl / starting,
        cagr=cagr,
        annualized_volatility=annualized_volatility,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        omega=omega,
        max_drawdown=max_drawdown,
        max_drawdown_duration=max_drawdown_duration,
        max_drawdown_duration_days=max_drawdown_days,
        var_95_historical=var_95_historical,
        var_99_historical=var_99_historical,
        var_95_parametric=var_95_parametric,
        var_99_parametric=var_99_parametric,
        cvar_95=cvar_95,
        cvar_99=cvar_99,
        cvar_95_parametric=cvar_95_parametric,
        cvar_99_parametric=cvar_99_parametric,
        rachev_ratio_05=_rachev_ratio(returns),
        return_on_max_initial_margin=return_on_margin,
        average_margin_utilization=average_utilization,
        peak_margin_utilization=peak_utilization,
        max_margin_used=max_margin,
        implementation_shortfall_ticks=implementation_shortfall,
        turnover=turnover,
        annualized_turnover=annualized_turnover,
        commission_to_gross_pnl_ratio=commission_ratio,
        price_pnl=float(price_pnl),
        roll_pnl=float(roll_pnl),
        funding=float(funding),
        gross_pnl=gross_pnl,
        spot_pnl_share=spot_share,
        roll_pnl_share=roll_share,
        roll_yield_annualized=roll_yield,
        risk_free_rate=risk_free_rate,
        information_ratio=information_ratio,
        beta=beta,
        alpha=alpha,
        benchmark_total_return=benchmark_total_return,
        metadata=metadata,
        timeframe=str(resolved_timeframe),
    )


def compute_metrics_from_result(
    result: BacktestResult,
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: int | None = None,
    timeframe: str = "1d",
    benchmark_returns: Sequence[float] | None = None,
) -> PerformanceMetrics:
    """Compute metrics directly from a :class:`~src.backtest.engine.BacktestResult`."""
    curve = result.equity_curve
    shortfall_ticks = [fill.execution_cost_ticks for fill in result.fills]
    if not shortfall_ticks:
        shortfall_ticks = [trade.execution_cost_ticks for trade in result.trades]
    return compute_metrics(
        curve,
        initial_equity=result.initial_equity,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
        timeframe=timeframe,
        trade_shortfall_ticks=shortfall_ticks,
        traded_notional=result.turnover,
        commissions=result.total_commissions,
        price_pnl=result.price_pnl,
        roll_pnl=result.roll_pnl,
        funding=result.funding,
        benchmark_returns=benchmark_returns,
    )


def rolling_sharpe(
    returns: Sequence[float] | NDArray[np.float64],
    window: int,
    *,
    periods_per_year: int | None = None,
    timeframe: str = "1d",
    risk_free_rate: float = 0.0,
) -> NDArray[np.float64]:
    """Rolling annualised Sharpe ratio (``nan`` until the window is full)."""
    values = np.asarray(list(returns), dtype=np.float64)
    if window < 2:
        raise ValueError("window must be >= 2")
    periods_per_year = resolve_periods_per_year(timeframe, periods_per_year)
    output = np.full(values.shape, np.nan, dtype=np.float64)
    rf_daily = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    for index in range(window - 1, values.size):
        chunk = values[index - window + 1 : index + 1] - rf_daily
        std = float(np.std(chunk, ddof=1))
        if std > 0.0:
            output[index] = float(np.mean(chunk)) / std * math.sqrt(periods_per_year)
        else:
            output[index] = 0.0
    return output


def underwater_curve(equity: Sequence[float] | NDArray[np.float64]) -> NDArray[np.float64]:
    """Drawdown series ``equity / running_peak - 1`` (non-positive fractions)."""
    values = np.asarray(list(equity), dtype=np.float64)
    if values.size == 0:
        return np.array([], dtype=np.float64)
    peaks = np.maximum.accumulate(values)
    return values / peaks - 1.0
