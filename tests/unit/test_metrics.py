"""Unit tests for the institutional performance metrics (closed-form checks)."""

from __future__ import annotations

import json
import math
import statistics
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from src.analytics.metrics import (
    PerformanceMetrics,
    compute_metrics,
    compute_metrics_from_result,
    rolling_sharpe,
    underwater_curve,
)
from src.backtest.engine import EQUITY_CURVE_SCHEMA, BacktestMode, BacktestResult

PPY = 252


def equity_from_returns(start: float, returns: list[float]) -> list[float]:
    values = [start]
    for period_return in returns:
        values.append(values[-1] * (1.0 + period_return))
    return values


def test_constant_returns_degenerate_conventions() -> None:
    returns = [0.01] * 99
    equity = equity_from_returns(100.0, returns)
    metrics = compute_metrics(equity, periods_per_year=PPY)
    assert metrics.n_periods == 99
    assert metrics.annualized_volatility == pytest.approx(0.0)
    assert metrics.sharpe == pytest.approx(0.0)
    assert metrics.sortino == pytest.approx(0.0)
    assert metrics.calmar == pytest.approx(0.0)
    assert metrics.omega == pytest.approx(1.0)
    assert metrics.max_drawdown == pytest.approx(0.0)
    assert metrics.max_drawdown_duration == 0
    assert metrics.var_95_historical == pytest.approx(0.0)
    assert metrics.cvar_95 == pytest.approx(0.0)
    assert metrics.rachev_ratio_05 == pytest.approx(0.0)
    expected_cagr = (equity[-1] / equity[0]) ** (PPY / 99) - 1.0
    assert metrics.cagr == pytest.approx(expected_cagr)
    assert metrics.total_return == pytest.approx(equity[-1] / equity[0] - 1.0)
    assert metrics.net_pnl == pytest.approx(equity[-1] - equity[0])
    assert metrics.initial_equity == pytest.approx(100.0)
    assert metrics.ending_equity == pytest.approx(equity[-1])


def test_known_drawdown_and_duration() -> None:
    equity = [100.0, 110.0, 99.0, 108.0]
    timestamps = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=index) for index in range(4)]
    metrics = compute_metrics(equity, timestamps=timestamps, periods_per_year=PPY)
    assert metrics.max_drawdown == pytest.approx(0.1)
    assert metrics.max_drawdown_duration == 2
    assert metrics.max_drawdown_duration_days == pytest.approx(2.0)
    returns = np.diff(equity) / np.array(equity[:-1])
    years = len(returns) / PPY
    expected_cagr = (equity[-1] / equity[0]) ** (1.0 / years) - 1.0
    assert metrics.cagr == pytest.approx(expected_cagr)
    assert metrics.calmar == pytest.approx(expected_cagr / 0.1)
    assert metrics.start == timestamps[0]
    assert metrics.end == timestamps[-1]


def _metrics_with_returns(returns: list[float]) -> PerformanceMetrics:
    equity = equity_from_returns(100.0, returns)
    return compute_metrics(equity, periods_per_year=PPY)


def test_var_and_cvar_closed_form() -> None:
    # 21 returns: quantile(0.05) is exactly the second-smallest observation.
    returns = [-0.10, -0.05] + [0.0] * 17 + [0.14, 0.15]
    metrics = _metrics_with_returns(returns)
    assert metrics.var_95_historical == pytest.approx(0.05)
    assert metrics.cvar_95 == pytest.approx((0.10 + 0.05) / 2.0)
    assert metrics.var_99_historical == pytest.approx(0.09)
    assert metrics.cvar_99 == pytest.approx(0.10)
    # Parametric VaR follows a Gaussian fit.
    mean = statistics.fmean(returns)
    std = statistics.pstdev(returns)
    z95 = statistics.NormalDist().inv_cdf(0.05)
    expected_parametric = max(0.0, -(mean + z95 * std))
    assert metrics.var_95_parametric == pytest.approx(expected_parametric)
    assert metrics.var_99_parametric >= 0.0
    assert metrics.cvar_95_parametric >= metrics.var_95_parametric


def test_rachev_ratio_closed_form() -> None:
    returns = [-0.10, -0.05] + [0.0] * 17 + [0.14, 0.15]
    metrics = _metrics_with_returns(returns)
    assert metrics.rachev_ratio_05 == pytest.approx(0.145 / 0.075, rel=1e-9)


def test_sharpe_sortino_omega_closed_form() -> None:
    returns = [0.01, 0.02, -0.01, 0.03]
    metrics = _metrics_with_returns(returns)
    mean = statistics.fmean(returns)
    variance = sum((value - mean) ** 2 for value in returns) / (len(returns) - 1)
    std = math.sqrt(variance)
    assert metrics.annualized_volatility == pytest.approx(std * math.sqrt(PPY))
    assert metrics.sharpe == pytest.approx(mean / std * math.sqrt(PPY))
    downside = [min(value, 0.0) for value in returns]
    downside_deviation = math.sqrt(sum(value * value for value in downside) / len(downside))
    assert metrics.sortino == pytest.approx(mean / downside_deviation * math.sqrt(PPY))
    assert metrics.omega == pytest.approx(
        sum(value for value in returns if value > 0.0)
        / abs(sum(value for value in returns if value < 0.0))
    )


def test_risk_free_rate_changes_sharpe() -> None:
    returns = [0.01, 0.02, -0.01, 0.03]
    equity = equity_from_returns(100.0, returns)
    without = compute_metrics(equity, periods_per_year=PPY, risk_free_rate=0.0)
    with_rf = compute_metrics(equity, periods_per_year=PPY, risk_free_rate=0.05)
    assert with_rf.sharpe < without.sharpe
    assert with_rf.risk_free_rate == pytest.approx(0.05)


def test_margin_efficiency_and_rom() -> None:
    equity = [1_000_000.0, 1_050_000.0, 1_100_000.0]
    margin_used = [100_000.0, 200_000.0, 150_000.0]
    utilization = [0.10, 0.19, 0.14]
    metrics = compute_metrics(
        equity,
        margin_used=margin_used,
        margin_utilization=utilization,
        periods_per_year=PPY,
    )
    assert metrics.max_margin_used == pytest.approx(200_000.0)
    assert metrics.average_margin_utilization == pytest.approx(statistics.fmean(utilization))
    assert metrics.peak_margin_utilization == pytest.approx(0.19)
    assert metrics.return_on_max_initial_margin == pytest.approx(100_000.0 / 200_000.0)
    # Utilisation can be derived from margin used when not supplied.
    derived = compute_metrics(equity, margin_used=margin_used, periods_per_year=PPY)
    assert derived.average_margin_utilization == pytest.approx(
        statistics.fmean(
            [
                margin / equity_value
                for margin, equity_value in zip(margin_used, equity, strict=True)
            ]
        )
    )
    assert derived.peak_margin_utilization == pytest.approx(200_000.0 / 1_050_000.0)


def test_execution_quality_and_attribution() -> None:
    equity = [1_000_000.0, 1_010_000.0, 1_020_000.0]
    metrics = compute_metrics(
        equity,
        trade_shortfall_ticks=[0.5, 1.5, 1.0],
        traded_notional=4_000_000.0,
        commissions=5_000.0,
        price_pnl=70_000.0,
        roll_pnl=-30_000.0,
        funding=0.0,
        periods_per_year=PPY,
    )
    assert metrics.implementation_shortfall_ticks == pytest.approx(1.0)
    expected_turnover = 4_000_000.0 / statistics.fmean(equity)
    assert metrics.turnover == pytest.approx(expected_turnover)
    assert metrics.annualized_turnover == pytest.approx(expected_turnover * PPY / 2)
    assert metrics.gross_pnl == pytest.approx(40_000.0)
    assert metrics.commission_to_gross_pnl_ratio == pytest.approx(5_000.0 / 40_000.0)
    assert metrics.spot_pnl_share == pytest.approx(70_000.0 / 100_000.0)
    assert metrics.roll_pnl_share == pytest.approx(-30_000.0 / 100_000.0)
    years = 2 / PPY
    assert metrics.roll_yield_annualized == pytest.approx(-30_000.0 / 1_000_000.0 / years)
    # A per-bar turnover series is summed.
    series = compute_metrics(
        equity, traded_notional=[1_000_000.0, 3_000_000.0], periods_per_year=PPY
    )
    assert series.turnover == pytest.approx(expected_turnover)
    # Non-positive gross PnL uses the neutral sentinel.
    losing = compute_metrics(equity, commissions=100.0, price_pnl=-50.0, roll_pnl=-50.0)
    assert losing.commission_to_gross_pnl_ratio == pytest.approx(0.0)
    zero_attribution = compute_metrics(equity)
    assert zero_attribution.spot_pnl_share == pytest.approx(0.0)


def test_benchmark_metrics() -> None:
    returns = [0.01, 0.02, -0.01, 0.03, 0.0]
    benchmark = [0.005, 0.01, -0.005, 0.02, 0.001]
    equity = equity_from_returns(100.0, returns)
    metrics = compute_metrics(equity, benchmark_returns=benchmark, periods_per_year=PPY)
    assert metrics.information_ratio is not None
    assert metrics.beta is not None
    assert metrics.alpha is not None
    assert metrics.benchmark_total_return == pytest.approx(math.prod(1 + b for b in benchmark) - 1)
    active = np.array(returns) - np.array(benchmark)
    expected_ir = float(np.mean(active)) / float(np.std(active, ddof=1)) * math.sqrt(PPY)
    assert metrics.information_ratio == pytest.approx(expected_ir)
    # Degenerate benchmark (zero variance).
    flat = compute_metrics(equity, benchmark_returns=[0.0] * 5, periods_per_year=PPY)
    assert flat.beta == pytest.approx(0.0)
    assert flat.information_ratio is not None
    assert math.isfinite(flat.information_ratio)


def test_dataframe_and_series_inputs() -> None:
    equity = [1_000_000.0, 1_001_000.0, 1_002_000.0]
    timestamps = [datetime(2024, 1, 1 + index, tzinfo=UTC) for index in range(3)]
    frame = pl.DataFrame(
        {
            "timestamp_utc": timestamps,
            "equity": equity,
            "margin_used": [100.0, 200.0, 300.0],
            "margin_utilization": [0.0001, 0.0002, 0.0003],
            "turnover": [1_000.0, 2_000.0, 3_000.0],
        }
    )
    from_frame = compute_metrics(frame, periods_per_year=PPY)
    assert from_frame.start == timestamps[0]
    assert from_frame.max_margin_used == pytest.approx(300.0)
    assert from_frame.turnover > 0.0
    from_series = compute_metrics(pl.Series(equity), periods_per_year=PPY)
    assert from_series.ending_equity == pytest.approx(1_002_000.0)
    from_array = compute_metrics(np.array(equity), periods_per_year=PPY)
    assert from_array.total_return == pytest.approx(equity[-1] / equity[0] - 1.0)
    with_timestamps = compute_metrics(equity, timestamps=timestamps, periods_per_year=PPY)
    assert with_timestamps.start == timestamps[0]


def test_metric_validation() -> None:
    with pytest.raises(ValueError):
        compute_metrics([100.0])
    with pytest.raises(ValueError):
        compute_metrics([100.0, float("nan")])
    with pytest.raises(ValueError):
        compute_metrics([100.0, 0.0])
    with pytest.raises(ValueError):
        compute_metrics([100.0, 101.0], periods_per_year=0)
    with pytest.raises(ValueError):
        compute_metrics([100.0, 101.0], risk_free_rate=float("nan"))
    with pytest.raises(ValueError):
        compute_metrics(
            [100.0, 101.0],
            timestamps=[datetime(2024, 1, 1, tzinfo=UTC)],
        )
    with pytest.raises(ValueError):
        compute_metrics([100.0, 101.0], margin_used=[1.0])
    with pytest.raises(ValueError):
        compute_metrics([100.0, 101.0], margin_utilization=[1.0])
    with pytest.raises(ValueError):
        compute_metrics([100.0, 101.0], benchmark_returns=[0.0, 0.1])
    with pytest.raises(ValueError):
        compute_metrics(pl.DataFrame({"other": [1.0, 2.0]}))
    with pytest.raises(ValueError):
        compute_metrics(pl.DataFrame({"equity": [1.0, 2.0]}), periods_per_year=0)


def test_metrics_are_finite_and_json_serialisable() -> None:
    equity = [1_000_000.0, 990_000.0, 1_010_000.0, 1_005_000.0]
    metrics = compute_metrics(
        equity,
        timestamps=[datetime(2024, 1, 1 + index, tzinfo=UTC) for index in range(4)],
        margin_used=[100_000.0] * 4,
    )
    payload = metrics.to_dict()
    encoded = json.dumps(payload)
    assert '"sharpe"' in encoded
    assert payload["start"] == "2024-01-01T00:00:00+00:00"
    assert all(isinstance(value, (int, float, str, type(None), dict)) for value in payload.values())
    with pytest.raises(ValueError):
        PerformanceMetrics(
            n_periods=1,
            periods_per_year=PPY,
            start=None,
            end=None,
            initial_equity=1.0,
            ending_equity=1.0,
            net_pnl=0.0,
            total_return=0.0,
            cagr=float("inf"),
            annualized_volatility=0.0,
            sharpe=0.0,
            sortino=0.0,
            calmar=0.0,
            omega=1.0,
            max_drawdown=0.0,
            max_drawdown_duration=0,
            max_drawdown_duration_days=0.0,
            var_95_historical=0.0,
            var_99_historical=0.0,
            var_95_parametric=0.0,
            var_99_parametric=0.0,
            cvar_95=0.0,
            cvar_99=0.0,
            cvar_95_parametric=0.0,
            cvar_99_parametric=0.0,
            rachev_ratio_05=0.0,
            return_on_max_initial_margin=0.0,
            average_margin_utilization=0.0,
            peak_margin_utilization=0.0,
            max_margin_used=0.0,
            implementation_shortfall_ticks=0.0,
            turnover=0.0,
            annualized_turnover=0.0,
            commission_to_gross_pnl_ratio=0.0,
            price_pnl=0.0,
            roll_pnl=0.0,
            funding=0.0,
            gross_pnl=0.0,
            spot_pnl_share=0.0,
            roll_pnl_share=0.0,
            roll_yield_annualized=0.0,
        )


def test_rolling_sharpe_and_underwater_curve() -> None:
    returns = np.array([0.01, 0.02, -0.01, 0.03, 0.0, -0.02, 0.01])
    series = rolling_sharpe(returns, 3, periods_per_year=PPY)
    assert np.isnan(series[0]) and np.isnan(series[1])
    first_window = returns[:3]
    expected = float(np.mean(first_window)) / float(np.std(first_window, ddof=1)) * math.sqrt(PPY)
    assert series[2] == pytest.approx(expected)
    flat = rolling_sharpe(np.zeros(5), 2)
    assert flat[-1] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        rolling_sharpe(returns, 1)
    with pytest.raises(ValueError):
        rolling_sharpe(returns, 3, periods_per_year=0)
    curve = underwater_curve([100.0, 110.0, 99.0, 121.0])
    assert curve[0] == pytest.approx(0.0)
    assert curve[2] == pytest.approx(99.0 / 110.0 - 1.0)
    assert curve[3] == pytest.approx(0.0)
    assert underwater_curve([]).size == 0


def make_result(equity_curve: pl.DataFrame, initial_equity: float = 1_000_000.0) -> BacktestResult:
    return BacktestResult(
        run_id="unit-run",
        strategy_id="unit-strategy",
        mode=BacktestMode.EVENT,
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 1, 4, tzinfo=UTC),
        initial_equity=initial_equity,
        ending_equity=float(equity_curve["equity"][-1]),
        equity_curve=equity_curve,
        price_pnl=5_000.0,
        roll_pnl=-1_000.0,
        commissions=100.0,
        fees=20.0,
        slippage=30.0,
        turnover=2_000_000.0,
    )


def make_curve() -> pl.DataFrame:
    timestamps = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=index) for index in range(4)]
    rows = [
        {
            "timestamp_utc": timestamp,
            "equity": equity,
            "cash": equity,
            "unrealized_pnl": 0.0,
            "price_pnl": 1_000.0,
            "roll_pnl": -200.0,
            "commissions": 10.0,
            "fees": 2.0,
            "slippage": 3.0,
            "funding": 0.0,
            "margin_used": 100_000.0,
            "margin_utilization": 0.1,
            "gross_exposure": 500_000.0,
            "net_exposure": 100_000.0,
            "drawdown": 0.0,
            "turnover": 50_000.0,
            "n_positions": 2,
        }
        for timestamp, equity in zip(
            timestamps, [1_000_000.0, 1_010_000.0, 1_005_000.0, 1_020_000.0], strict=True
        )
    ]
    return pl.DataFrame(rows).select(
        [pl.col(column).cast(dtype) for column, dtype in EQUITY_CURVE_SCHEMA.items()]
    )


def test_compute_metrics_from_result() -> None:
    result = make_result(make_curve())
    metrics = compute_metrics_from_result(result)
    assert metrics.initial_equity == pytest.approx(1_000_000.0)
    assert metrics.ending_equity == pytest.approx(1_020_000.0)
    assert metrics.price_pnl == pytest.approx(5_000.0)
    assert metrics.roll_pnl == pytest.approx(-1_000.0)
    assert metrics.max_margin_used == pytest.approx(100_000.0)
    assert metrics.turnover == pytest.approx(
        2_000_000.0 / statistics.fmean([1_000_000.0, 1_010_000.0, 1_005_000.0, 1_020_000.0])
    )
