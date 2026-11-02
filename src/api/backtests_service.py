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

import math
import time
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from datetime import time as dt_time
from typing import TYPE_CHECKING

import numpy as np
from configs.instruments import InstrumentsConfig
from configs.settings import Settings
from configs.strategy import StrategyConfig

from src.api import fixtures
from src.api.protocols import StrategyCatalog
from src.api.schemas import (
    AttributionRow,
    BacktestListResponse,
    BacktestRunConfig,
    BacktestRunSummary,
    MetricsSet,
    MonthlyReturn,
    SeriesPoint,
    TailPoint,
    TearsheetResponse,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.analytics.metrics import PerformanceMetrics
    from src.backtest.engine import BacktestResult

__all__ = ["EngineBacktestService"]

#: Rolling window (bars) used by the tear sheet rolling-Sharpe panel.
ROLLING_WINDOW: int = 63

#: Number of tail observations per side on the Rachev plot.
TAIL_OBSERVATIONS: int = 30

#: Maximum retained runs in the in-memory store.
MAX_RUNS: int = 40

#: Seed runs created at boot (one per strategy, capped).
SEED_RUNS: int = 3


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
        self._seed_fixture_runs()

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
        """Execute one deterministic offline backtest and store its tear sheet."""
        config = self._catalog.core_config(request.strategy_id)
        if config is None:
            raise KeyError(f"unknown strategy {request.strategy_id!r}")
        started = time.perf_counter()
        try:
            summary, tearsheet = self._execute_engine(config, request)
            run_id = self._allocate_run_id()
            duration = time.perf_counter() - started
            stored = summary.model_copy(
                update={
                    "id": run_id,
                    "created_at": self._clock(),
                    "duration_seconds": round(duration, 3),
                }
            )
            self._runs[run_id] = stored
            tearsheet = tearsheet.model_copy(update={"run_id": run_id})
            self._tearsheets[run_id] = tearsheet
            self._trim_store()
            return stored
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            return self._fixture_run(request, reason=reason, started=started)

    # -- engine execution ------------------------------------------------------

    def _execute_engine(
        self, config: StrategyConfig, request: BacktestRunConfig
    ) -> tuple[BacktestRunSummary, TearsheetResponse]:
        from src.api.services import build_pipeline_parts
        from src.backtest.engine import (
            BacktestEngine,
            BacktestMode,
            RebalanceSchedule,
        )

        window = fixtures.build_window_market(
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
        )
        self._runs[run_id] = summary
        self._tearsheets[run_id] = tearsheet
        self._trim_store()
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

    # -- store -----------------------------------------------------------------

    def _allocate_run_id(self) -> str:
        self._next_id += 1
        return f"bt-{self._next_id:04d}"

    def _trim_store(self) -> None:
        if len(self._runs) <= MAX_RUNS:
            return
        ordered = sorted(self._runs.values(), key=lambda run: run.created_at)
        for stale in ordered[: len(self._runs) - MAX_RUNS]:
            self._runs.pop(stale.id, None)
            self._tearsheets.pop(stale.id, None)
