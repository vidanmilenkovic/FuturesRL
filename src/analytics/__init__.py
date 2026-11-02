"""Institutional futures analytics: performance metrics and tear sheets.

Public API
----------
* :func:`compute_metrics` / :class:`PerformanceMetrics` -- CAGR, vol, Sharpe,
  Sortino, Calmar, Omega, drawdown, VaR/CVaR, Rachev ratio, RoM, margin
  efficiency, execution quality and spot-vs-roll attribution.
* :func:`compute_metrics_from_result` -- metrics straight from a ``BacktestResult``.
* :func:`rolling_sharpe`, :func:`underwater_curve` -- chart-ready series.
* :class:`Tearsheet`, :func:`build_tearsheet`, :func:`write_tearsheet`,
  :func:`compare_tearsheets` -- JSON + standalone HTML reports and the Backtest
  Lab comparison payload.
"""

from src.analytics.metrics import (
    RACHEV_ALPHA,
    PerformanceMetrics,
    compute_metrics,
    compute_metrics_from_result,
    rolling_sharpe,
    underwater_curve,
)
from src.analytics.tearsheet import (
    Tearsheet,
    build_tearsheet,
    compare_tearsheets,
    write_tearsheet,
)

__all__ = [
    "RACHEV_ALPHA",
    "PerformanceMetrics",
    "Tearsheet",
    "build_tearsheet",
    "compare_tearsheets",
    "compute_metrics",
    "compute_metrics_from_result",
    "rolling_sharpe",
    "underwater_curve",
    "write_tearsheet",
]
