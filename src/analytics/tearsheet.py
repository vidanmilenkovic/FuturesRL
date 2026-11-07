"""JSON + standalone-HTML tear-sheet generator for futures backtests.

The tear sheet is what the Backtest Lab renders and what the API serves: a
self-contained, dependency-free report (inline CSS, hand-built SVG sparklines and
a PnL-attribution waterfall -- no matplotlib/plotly required) plus a
JSON-serialisable payload for the web frontend.

``compare_tearsheets`` builds the multi-run comparison the Backtest Lab uses:
benchmark versus strategies, rolling Sharpe, underwater drawdown, Rachev tail
data and roll-yield attribution series.
"""

from __future__ import annotations

import html
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from src.analytics.metrics import (
    RACHEV_ALPHA,
    PerformanceMetrics,
    compute_metrics_from_result,
    rolling_sharpe,
    underwater_curve,
)
from src.data.schema import TIMESTAMP_COLUMN

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.backtest.engine import BacktestResult

#: Maximum number of curve points embedded in a tear sheet (deterministic stride).
_MAX_CURVE_POINTS: int = 1_000

_CSS = """
:root { color-scheme: dark; --bg:#0b1220; --panel:#111c30; --ink:#e5edf9; --muted:#8fa3bf;
        --pos:#22c55e; --neg:#ef4444; --accent:#38bdf8; --grid:#1e293b; }
* { box-sizing: border-box; }
body { margin:0; padding:32px; background:var(--bg); color:var(--ink);
       font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
h1 { margin:0 0 4px; font-size:26px; letter-spacing:-0.02em; }
h2 { margin:28px 0 10px; font-size:16px; color:var(--accent); text-transform:uppercase;
     letter-spacing:0.08em; }
.sub { color:var(--muted); margin:0 0 20px; font-size:13px; }
.cards { display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); gap:12px; }
.card { background:var(--panel); border:1px solid var(--grid); border-radius:10px; padding:12px 14px; }
.card .label { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:0.06em; }
.card .value { font-size:20px; font-weight:600; margin-top:4px; }
.pos { color:var(--pos); } .neg { color:var(--neg); }
.panel { background:var(--panel); border:1px solid var(--grid); border-radius:10px;
         padding:14px; overflow:hidden; }
svg { display:block; width:100%; height:auto; }
table { width:100%; border-collapse:collapse; font-size:13px; }
th, td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--grid); }
th { color:var(--muted); font-weight:500; }
td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
footer { margin-top:32px; color:var(--muted); font-size:12px; }
"""


# ---------------------------------------------------------------------------
# SVG helpers (no plotting dependency)
# ---------------------------------------------------------------------------


def _downsample(values: Sequence[float], max_points: int = _MAX_CURVE_POINTS) -> list[float]:
    values_list = [float(value) for value in values]
    if len(values_list) <= max_points:
        return values_list
    stride = math.ceil(len(values_list) / max_points)
    sampled = values_list[::stride]
    if sampled[-1] != values_list[-1]:
        sampled.append(values_list[-1])
    return sampled


def _svg_line(
    values: Sequence[float],
    *,
    width: int = 640,
    height: int = 160,
    stroke: str = "#38bdf8",
    fill: str = "rgba(56,189,248,0.12)",
    zero_line: bool = False,
) -> str:
    """Render a deterministic SVG line chart for a numeric series."""
    numeric = [float(value) for value in values if value is not None and math.isfinite(value)]
    if len(numeric) < 2:
        return (
            f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="insufficient data">'
            f'<rect width="{width}" height="{height}" fill="none"/>'
            f'<text x="{width / 2}" y="{height / 2}" fill="#8fa3bf" font-size="12" '
            f'text-anchor="middle">insufficient data</text></svg>'
        )
    minimum = min(numeric)
    maximum = max(numeric)
    span = maximum - minimum
    if span <= 0.0:
        span = 1.0
    padding = 12.0
    inner_width = width - 2 * padding
    inner_height = height - 2 * padding
    step = inner_width / (len(numeric) - 1)

    def to_y(value: float) -> float:
        scaled = (value - minimum) / span
        return padding + (1.0 - scaled) * inner_height

    points = " ".join(
        f"{padding + index * step:.2f},{to_y(value):.2f}" for index, value in enumerate(numeric)
    )
    area = (
        f"{padding:.2f},{height - padding:.2f} "
        + points
        + f" {width - padding:.2f},{height - padding:.2f}"
    )
    zero_markup = ""
    if zero_line and minimum < 0.0 < maximum:
        zero_y = to_y(0.0)
        zero_markup = (
            f'<line x1="{padding}" y1="{zero_y:.2f}" x2="{width - padding}" '
            f'y2="{zero_y:.2f}" stroke="#475569" stroke-width="1" stroke-dasharray="4 4"/>'
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="time series">'
        f'<polygon points="{area}" fill="{fill}" stroke="none"/>'
        f"{zero_markup}"
        f'<polyline points="{points}" fill="none" stroke="{stroke}" stroke-width="2" '
        f'stroke-linejoin="round" stroke-linecap="round"/>'
        f'<text x="{padding}" y="{padding - 2}" fill="#8fa3bf" font-size="10">'
        f"{maximum:.4g}</text>"
        f'<text x="{padding}" y="{height - 2}" fill="#8fa3bf" font-size="10">'
        f"{minimum:.4g}</text>"
        f"</svg>"
    )


def _svg_waterfall(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    width: int = 640,
    height: int = 180,
) -> str:
    """Render a simple attribution waterfall (bars from a running baseline)."""
    pairs = [
        (str(label), float(value))
        for label, value in zip(labels, values, strict=True)
        if math.isfinite(float(value))
    ]
    if not pairs:
        return (
            f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="attribution">'
            f'<text x="{width / 2}" y="{height / 2}" fill="#8fa3bf" font-size="12" '
            f'text-anchor="middle">no attribution data</text></svg>'
        )
    running = 0.0
    levels: list[tuple[str, float, float, bool]] = []
    for label, value in pairs:
        start = running
        running += value
        levels.append((label, start, running, value >= 0.0))
    baseline = min(0.0, min(min(start, end) for _, start, end, _ in levels))
    ceiling = max(0.0, max(max(start, end) for _, start, end, _ in levels))
    span = ceiling - baseline or 1.0
    padding = 12.0
    inner_width = width - 2 * padding
    inner_height = height - 2 * padding - 14.0
    bar_width = inner_width / len(levels)

    def to_y(value: float) -> float:
        return padding + (ceiling - value) / span * inner_height

    bars: list[str] = []
    for index, (label, start, end, positive) in enumerate(levels):
        x = padding + index * bar_width + bar_width * 0.15
        bar_w = bar_width * 0.7
        top = min(to_y(start), to_y(end))
        bar_h = max(1.0, abs(to_y(start) - to_y(end)))
        colour = "#22c55e" if positive else "#ef4444"
        bars.append(
            f'<rect x="{x:.2f}" y="{top:.2f}" width="{bar_w:.2f}" height="{bar_h:.2f}" '
            f'fill="{colour}" rx="2"/>'
        )
        bars.append(
            f'<text x="{x + bar_w / 2:.2f}" y="{height - 3}" fill="#8fa3bf" '
            f'font-size="9" text-anchor="middle">{html.escape(label)}</text>'
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="pnl attribution">'
        + "".join(bars)
        + "</svg>"
    )


# ---------------------------------------------------------------------------
# Tearsheet
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Tearsheet:
    """A complete backtest report: typed metrics, JSON payload and HTML document."""

    title: str
    strategy_id: str
    run_id: str
    generated_at: datetime
    metrics: PerformanceMetrics
    payload: dict[str, object] = field(default_factory=dict)
    html: str = ""

    def __post_init__(self) -> None:
        if not self.title:
            raise ValueError("tearsheet title must be non-empty")
        if not isinstance(self.payload, dict):
            raise ValueError("tearsheet payload must be a dict")

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable representation of the whole report."""
        return {
            "title": self.title,
            "strategy_id": self.strategy_id,
            "run_id": self.run_id,
            "generated_at": self.generated_at.isoformat(),
            "metrics": self.metrics.to_dict(),
            **self.payload,
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialise the tear sheet to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, default=str)

    def write(self, path: Path | str, *, fmt: str | None = None) -> Path:
        """Write the tear sheet to ``path`` (JSON or HTML based on suffix/``fmt``)."""
        return write_tearsheet(self, path, fmt=fmt)


def _float_list(values: Sequence[float] | NDArray[np.float64]) -> list[float]:
    return [float(value) for value in values]


def _json_series(values: Sequence[float] | NDArray[np.float64]) -> list[float | None]:
    return [float(value) if math.isfinite(float(value)) else None for value in values]


def _monthly_returns(
    timestamps: Sequence[datetime],
    equity: Sequence[float] | NDArray[np.float64],
    initial_equity: float,
) -> list[dict[str, object]]:
    if not timestamps or len(timestamps) != len(equity):
        return []
    month_last: dict[str, float] = {}
    for timestamp, value in zip(timestamps, equity, strict=True):
        month_last[ensure_month_key(timestamp)] = float(value)
    previous = float(initial_equity)
    output: list[dict[str, object]] = []
    for month, value in month_last.items():
        period_return = value / previous - 1.0 if previous > 0.0 else 0.0
        output.append({"month": month, "return": period_return})
        previous = value
    return output


def ensure_month_key(timestamp: datetime) -> str:
    """Return the ``YYYY-MM`` key of a timestamp (UTC)."""
    moment = (
        timestamp.astimezone(UTC) if timestamp.tzinfo is not None else timestamp.replace(tzinfo=UTC)
    )
    return f"{moment.year:04d}-{moment.month:02d}"


def _headline(metrics: PerformanceMetrics) -> dict[str, float]:
    return {
        "cagr": metrics.cagr,
        "annualized_volatility": metrics.annualized_volatility,
        "sharpe": metrics.sharpe,
        "sortino": metrics.sortino,
        "calmar": metrics.calmar,
        "omega": metrics.omega,
        "max_drawdown": metrics.max_drawdown,
        "max_drawdown_duration_days": metrics.max_drawdown_duration_days,
        "var_95": metrics.var_95_historical,
        "cvar_95": metrics.cvar_95,
        "rachev_ratio_05": metrics.rachev_ratio_05,
        "return_on_max_initial_margin": metrics.return_on_max_initial_margin,
        "average_margin_utilization": metrics.average_margin_utilization,
        "peak_margin_utilization": metrics.peak_margin_utilization,
        "implementation_shortfall_ticks": metrics.implementation_shortfall_ticks,
        "turnover": metrics.turnover,
        "commission_to_gross_pnl_ratio": metrics.commission_to_gross_pnl_ratio,
        "roll_yield_annualized": metrics.roll_yield_annualized,
        "total_return": metrics.total_return,
    }


def _format_percent(value: float) -> str:
    """Format a fraction as a percentage string."""
    return f"{value * 100:.2f}%"


def _format_ratio(value: float) -> str:
    """Format a ratio with two decimals."""
    return f"{value:.2f}"


def _card(label: str, value: str, *, positive: bool | None = None) -> str:
    css_class = "value"
    if positive is True:
        css_class += " pos"
    elif positive is False:
        css_class += " neg"
    return (
        f'<div class="card"><div class="label">{html.escape(label)}</div>'
        f'<div class="{css_class}">{html.escape(value)}</div></div>'
    )


def _metric_table(rows: Sequence[tuple[str, str]]) -> str:
    body = "".join(
        f'<tr><th>{html.escape(label)}</th><td class="num">{html.escape(value)}</td></tr>'
        for label, value in rows
    )
    return f"<table><tbody>{body}</tbody></table>"


def _render_html(
    *,
    title: str,
    strategy_id: str,
    run_id: str,
    mode: str,
    start: str,
    end: str,
    metrics: PerformanceMetrics,
    equity_svg: str,
    drawdown_svg: str,
    rolling_svg: str,
    attribution_svg: str,
    result: BacktestResult,
) -> str:
    pct = _format_percent
    ratio = _format_ratio
    cards = "".join(
        [
            _card("CAGR", pct(metrics.cagr), positive=metrics.cagr >= 0.0),
            _card("Ann. Vol", pct(metrics.annualized_volatility)),
            _card("Sharpe", ratio(metrics.sharpe), positive=metrics.sharpe >= 0.0),
            _card("Sortino", ratio(metrics.sortino), positive=metrics.sortino >= 0.0),
            _card("Calmar", ratio(metrics.calmar)),
            _card("Omega", ratio(metrics.omega)),
            _card("Max Drawdown", pct(-metrics.max_drawdown)),
            _card("VaR 95%", pct(metrics.var_95_historical)),
            _card("CVaR 95%", pct(metrics.cvar_95)),
            _card("Rachev (5%)", ratio(metrics.rachev_ratio_05)),
            _card("RoM", pct(metrics.return_on_max_initial_margin)),
            _card("Peak Margin", pct(metrics.peak_margin_utilization)),
        ]
    )
    risk_rows = [
        ("Annualised volatility", pct(metrics.annualized_volatility)),
        ("Sharpe ratio", ratio(metrics.sharpe)),
        ("Sortino ratio", ratio(metrics.sortino)),
        ("Calmar ratio", ratio(metrics.calmar)),
        ("Omega ratio", ratio(metrics.omega)),
        ("Max drawdown", pct(metrics.max_drawdown)),
        ("Max drawdown duration (days)", f"{metrics.max_drawdown_duration_days:.1f}"),
        ("VaR 95% historical", pct(metrics.var_95_historical)),
        ("VaR 99% historical", pct(metrics.var_99_historical)),
        ("VaR 95% parametric", pct(metrics.var_95_parametric)),
        ("VaR 99% parametric", pct(metrics.var_99_parametric)),
        ("CVaR 95% historical", pct(metrics.cvar_95)),
        ("CVaR 99% historical", pct(metrics.cvar_99)),
        ("CVaR 95% parametric", pct(metrics.cvar_95_parametric)),
        ("CVaR 99% parametric", pct(metrics.cvar_99_parametric)),
        ("Rachev ratio (alpha=0.05)", ratio(metrics.rachev_ratio_05)),
    ]
    execution_rows = [
        ("Implementation shortfall (ticks)", f"{metrics.implementation_shortfall_ticks:.3f}"),
        ("Turnover (x mean equity)", f"{metrics.turnover:.3f}"),
        ("Annualised turnover", f"{metrics.annualized_turnover:.3f}"),
        ("Commission / gross PnL", f"{metrics.commission_to_gross_pnl_ratio:.4f}"),
        ("Return on max initial margin", pct(metrics.return_on_max_initial_margin)),
        ("Average margin utilisation", pct(metrics.average_margin_utilization)),
        ("Peak margin utilisation", pct(metrics.peak_margin_utilization)),
        ("Max margin used", f"{metrics.max_margin_used:,.2f}"),
    ]
    attribution_rows = [
        ("Price (spot) PnL", f"{result.price_pnl:,.2f}"),
        ("Roll (carry) PnL", f"{result.roll_pnl:,.2f}"),
        ("Commissions", f"{result.commissions:,.2f}"),
        ("Exchange / regulatory fees", f"{result.fees:,.2f}"),
        ("Slippage (spread + impact)", f"{result.slippage:,.2f}"),
        ("Funding", f"{result.funding:,.2f}"),
        ("Roll / BAG execution costs", f"{result.roll_cost_total:,.2f}"),
        ("  of which commissions", f"{result.roll_commissions:,.2f}"),
        ("  of which exchange / reg fees", f"{result.roll_fees:,.2f}"),
        ("  of which spread + slippage", f"{result.roll_slippage:,.2f}"),
        ("Forced-liquidation costs", f"{result.forced_liquidation_cost:,.2f}"),
        ("Total execution costs", f"{result.cost_breakdown['total_costs']:,.2f}"),
        ("Roll yield (annualised)", pct(metrics.roll_yield_annualized)),
    ]
    roll_rows = "".join(
        f"<tr><td>{html.escape(event.timestamp.isoformat())}</td>"
        f"<td>{html.escape(event.instrument_id)}</td>"
        f'<td class="num">{event.from_price:.4f}</td>'
        f'<td class="num">{event.to_price:.4f}</td>'
        f'<td class="num">{event.roll_gap:.4f}</td>'
        f'<td class="num">{event.roll_pnl:,.2f}</td>'
        f'<td class="num">{event.total_cost:,.2f}</td></tr>'
        for event in result.roll_events
    )
    warning_text = (
        "".join(f"<li>{html.escape(warning)}</li>" for warning in result.warnings)
        if result.warnings
        else "<li>none</li>"
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{html.escape(title)}</title>
<style>{_CSS}</style>
</head>
<body>
<header>
  <h1>{html.escape(title)}</h1>
  <p class="sub">Strategy {html.escape(strategy_id)} &middot; run {html.escape(run_id)} &middot;
  mode {html.escape(mode)} &middot; {html.escape(start)} &rarr; {html.escape(end)}</p>
</header>
<section class="cards">{cards}</section>
<section><h2>Equity Curve</h2><div class="panel">{equity_svg}</div></section>
<section><h2>Underwater Drawdown</h2><div class="panel">{drawdown_svg}</div></section>
<section><h2>Rolling Sharpe</h2><div class="panel">{rolling_svg}</div></section>
<section><h2>PnL Attribution</h2><div class="panel">{attribution_svg}</div></section>
<section><h2>Risk &amp; Tail Metrics</h2><div class="panel">{_metric_table(risk_rows)}</div></section>
<section><h2>Execution Quality</h2><div class="panel">{_metric_table(execution_rows)}</div></section>
<section><h2>Spot vs Roll Yield Attribution</h2><div class="panel">{_metric_table(attribution_rows)}</div></section>
<section><h2>Roll Log</h2><div class="panel">
<table><thead><tr><th>Timestamp</th><th>Root</th><th class="num">From</th><th class="num">To</th>
<th class="num">Gap</th><th class="num">Roll PnL</th><th class="num">Cost</th></tr></thead>
<tbody>{roll_rows or '<tr><td colspan="7">no rolls executed</td></tr>'}</tbody></table></div></section>
<section><h2>Warnings</h2><div class="panel"><ul>{warning_text}</ul></div></section>
<footer>Generated by FuturesRL at {html.escape(result.end.isoformat())}.
Accounting identity residual: {result.accounting_residual:.6f}.</footer>
</body>
</html>
"""


def build_tearsheet(
    result: BacktestResult,
    metrics: PerformanceMetrics | None = None,
    *,
    title: str | None = None,
    benchmark: BacktestResult | None = None,
    rolling_window: int = 63,
    generated_at: datetime | None = None,
) -> Tearsheet:
    """Build a JSON payload + standalone HTML report from a backtest result."""
    del benchmark  # comparison is handled by ``compare_tearsheets``
    if rolling_window < 2:
        raise ValueError("rolling_window must be >= 2")
    frame = result.equity_curve
    if frame.height == 0:
        raise ValueError("cannot build a tear sheet for an empty equity curve")
    metrics = metrics or compute_metrics_from_result(result)
    generated = ensure_utc_datetime(generated_at) if generated_at is not None else datetime.now(UTC)
    timestamps = [
        ensure_utc_datetime(value) for value in frame.get_column(TIMESTAMP_COLUMN).to_list()
    ]
    equity = np.asarray(frame.get_column("equity").to_list(), dtype=np.float64)
    equity_sampled = _downsample([float(value) for value in equity])
    drawdown = underwater_curve(equity)
    drawdown_sampled = _downsample([float(value) for value in drawdown])
    returns = np.diff(equity) / equity[:-1] if equity.size > 1 else np.array([], dtype=np.float64)
    rolling = rolling_sharpe(
        returns,
        min(rolling_window, max(2, returns.size)),
        periods_per_year=metrics.periods_per_year,
        risk_free_rate=metrics.risk_free_rate,
    )
    rolling_values: list[float | None] = []
    if rolling.size:
        padded = np.concatenate([np.full(equity.size - rolling.size, np.nan), rolling])
        rolling_values = _json_series(padded)
    monthly = _monthly_returns(timestamps, equity, result.initial_equity)

    # -- attribution series ----------------------------------------------------
    price_series = np.asarray(frame.get_column("price_pnl").to_list(), dtype=np.float64)
    roll_series = np.asarray(frame.get_column("roll_pnl").to_list(), dtype=np.float64)
    cumulative_price = np.cumsum(price_series)
    cumulative_roll = np.cumsum(roll_series)
    attribution_labels = ["Price", "Roll", "Comm.", "Fees", "Slip."]
    attribution_values = [
        result.price_pnl,
        result.roll_pnl,
        -result.commissions,
        -result.fees,
        -result.slippage,
    ]

    # -- Rachev tail data ------------------------------------------------------
    if returns.size >= 2:
        lower_threshold = float(np.quantile(returns, RACHEV_ALPHA))
        upper_threshold = float(np.quantile(returns, 1.0 - RACHEV_ALPHA))
        lower_tail = [float(value) for value in returns[returns <= lower_threshold]]
        upper_tail = [float(value) for value in returns[returns >= upper_threshold]]
        lower_mean = float(np.mean(lower_tail)) if lower_tail else 0.0
        upper_mean = float(np.mean(upper_tail)) if upper_tail else 0.0
    else:
        lower_threshold = upper_threshold = 0.0
        lower_tail, upper_tail = [], []
        lower_mean = upper_mean = 0.0

    margin_used = np.asarray(frame.get_column("margin_used").to_list(), dtype=np.float64)
    margin_utilization = np.asarray(
        frame.get_column("margin_utilization").to_list(), dtype=np.float64
    )
    trade_kinds: dict[str, int] = {}
    for trade in result.trades:
        trade_kinds[trade.kind] = trade_kinds.get(trade.kind, 0) + 1
    payload: dict[str, object] = {
        "meta": {
            "mode": result.mode.value,
            "start": result.start.isoformat(),
            "end": result.end.isoformat(),
            "generated_at": generated.isoformat(),
            "n_bars": result.n_bars,
            "n_rebalances": result.n_rebalances,
            "warnings": list(result.warnings),
        },
        "equity_curve": {
            "timestamps": [timestamp.isoformat() for timestamp in timestamps],
            "equity": _json_series(equity_sampled),
            "drawdown": _json_series(drawdown_sampled),
            "margin_used": _json_series(_downsample([float(v) for v in margin_used])),
            "margin_utilization": _json_series(_downsample([float(v) for v in margin_utilization])),
        },
        "rolling_sharpe": {"window": rolling_window, "values": rolling_values},
        "monthly_returns": monthly,
        "attribution": {
            "cumulative_price_pnl": _json_series(_downsample([float(v) for v in cumulative_price])),
            "cumulative_roll_pnl": _json_series(_downsample([float(v) for v in cumulative_roll])),
            "price_pnl": result.price_pnl,
            "roll_pnl": result.roll_pnl,
            "commissions": result.commissions,
            "fees": result.fees,
            "slippage": result.slippage,
            "funding": result.funding,
            "roll_cost_total": result.roll_cost_total,
            "roll_commissions": result.roll_commissions,
            "roll_fees": result.roll_fees,
            "roll_slippage": result.roll_slippage,
            "roll_yield_annualized": metrics.roll_yield_annualized,
        },
        "cost_breakdown": result.cost_breakdown,
        "margin_regime": {
            "stress_applied": result.margin_stress_applied,
            "stress_multiplier": result.margin_stress_multiplier,
            "stress_roots": list(result.margin_stress_roots),
            "calibrated": result.margin_calibrated,
            "calibrated_roots": list(result.margin_calibrated_roots),
        },
        "rachev": {
            "alpha": RACHEV_ALPHA,
            "lower_threshold": lower_threshold,
            "upper_threshold": upper_threshold,
            "lower_tail": lower_tail,
            "upper_tail": upper_tail,
            "lower_mean": lower_mean,
            "upper_mean": upper_mean,
            "ratio": metrics.rachev_ratio_05,
        },
        "trades": {
            "count": len(result.trades),
            "fills": len(result.fills),
            "notional_turnover": result.turnover,
            "commissions": result.commissions,
            "fees": result.fees,
            "slippage": result.slippage,
            "by_kind": trade_kinds,
        },
        "rolls": [event.to_dict() for event in result.roll_events],
        "forced_liquidations": [event.to_dict() for event in result.forced_liquidations],
        "accounting": {
            "initial_equity": result.initial_equity,
            "ending_equity": result.ending_equity,
            "price_pnl": result.price_pnl,
            "roll_pnl": result.roll_pnl,
            "commissions": result.total_commissions,
            "slippage": result.slippage,
            "funding": result.funding,
            "residual": result.accounting_residual,
        },
    }
    resolved_title = title or f"FuturesRL tear sheet — {result.strategy_id}"
    equity_svg = _svg_line(equity_sampled, stroke="#38bdf8")
    drawdown_svg = _svg_line(
        drawdown_sampled,
        stroke="#ef4444",
        fill="rgba(239,68,68,0.12)",
        zero_line=True,
    )
    rolling_clean = [value for value in rolling_values if value is not None]
    rolling_svg = _svg_line(rolling_clean, stroke="#a78bfa", zero_line=True)
    attribution_svg = _svg_waterfall(attribution_labels, attribution_values)
    html_document = _render_html(
        title=resolved_title,
        strategy_id=result.strategy_id,
        run_id=result.run_id,
        mode=result.mode.value,
        start=result.start.isoformat(),
        end=result.end.isoformat(),
        metrics=metrics,
        equity_svg=equity_svg,
        drawdown_svg=drawdown_svg,
        rolling_svg=rolling_svg,
        attribution_svg=attribution_svg,
        result=result,
    )
    return Tearsheet(
        title=resolved_title,
        strategy_id=result.strategy_id,
        run_id=result.run_id,
        generated_at=generated,
        metrics=metrics,
        payload=payload,
        html=html_document,
    )


def ensure_utc_datetime(value: datetime) -> datetime:
    """Normalise a timestamp to timezone-aware UTC."""
    return value.astimezone(UTC) if value.tzinfo is not None else value.replace(tzinfo=UTC)


def write_tearsheet(
    tearsheet: Tearsheet | BacktestResult,
    path: Path | str,
    *,
    fmt: str | None = None,
) -> Path:
    """Write a tear sheet to ``path`` as JSON (default) or standalone HTML."""
    resolved = Path(path)
    if not isinstance(tearsheet, Tearsheet):
        tearsheet = build_tearsheet(tearsheet)
    resolved_format = (fmt or resolved.suffix.lstrip(".") or "json").lower()
    if resolved_format in {"htm", "html"}:
        content = tearsheet.html
    elif resolved_format == "json":
        content = tearsheet.to_json()
    else:
        raise ValueError(f"unsupported tear-sheet format {resolved_format!r}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content, encoding="utf-8")
    return resolved


def compare_tearsheets(
    tearsheets: Mapping[str, Tearsheet],
    *,
    benchmark: Tearsheet | None = None,
) -> dict[str, object]:
    """Build the Backtest Lab comparison payload for several runs.

    The output contains a headline comparison table plus per-run aligned series
    (normalised equity, underwater drawdown, rolling Sharpe, Rachev tails and
    cumulative roll-yield attribution) so the frontend can overlay runs without
    additional computation.
    """
    if not tearsheets:
        raise ValueError("compare_tearsheets requires at least one tearsheet")
    comparison_rows: list[dict[str, object]] = []
    runs: dict[str, object] = {}
    for name in tearsheets:
        tearsheet = tearsheets[name]
        metrics = tearsheet.metrics
        comparison_rows.append(
            {"name": name, "strategy_id": tearsheet.strategy_id, **_headline(metrics)}
        )
        equity_payload = tearsheet.payload.get("equity_curve", {})
        equity_values = equity_payload.get("equity", []) if isinstance(equity_payload, dict) else []
        initial = tearsheet.payload.get("accounting", {})
        initial_equity = (
            float(initial.get("initial_equity", 1.0)) if isinstance(initial, dict) else 1.0
        )
        normalised_equity = (
            [float(value) / initial_equity for value in equity_values if value is not None]
            if initial_equity > 0.0
            else []
        )
        attribution = tearsheet.payload.get("attribution", {})
        rachev = tearsheet.payload.get("rachev", {})
        rolling = tearsheet.payload.get("rolling_sharpe", {})
        runs[name] = {
            "strategy_id": tearsheet.strategy_id,
            "run_id": tearsheet.run_id,
            "generated_at": tearsheet.generated_at.isoformat(),
            "headline": _headline(metrics),
            "normalised_equity_curve": normalised_equity,
            "drawdown": equity_payload.get("drawdown", [])
            if isinstance(equity_payload, dict)
            else [],
            "rolling_sharpe": rolling.get("values", []) if isinstance(rolling, dict) else [],
            "cumulative_roll_pnl": attribution.get("cumulative_roll_pnl", [])
            if isinstance(attribution, dict)
            else [],
            "cumulative_price_pnl": attribution.get("cumulative_price_pnl", [])
            if isinstance(attribution, dict)
            else [],
            "rachev": rachev if isinstance(rachev, dict) else {},
        }
    benchmark_payload: dict[str, object] | None = None
    if benchmark is not None:
        benchmark_payload = {
            "strategy_id": benchmark.strategy_id,
            "run_id": benchmark.run_id,
            "headline": _headline(benchmark.metrics),
        }
        comparison_rows.append(
            {
                "name": "benchmark",
                "strategy_id": benchmark.strategy_id,
                **_headline(benchmark.metrics),
            }
        )
    return {
        "comparison_table": comparison_rows,
        "runs": runs,
        "benchmark": benchmark_payload,
        "summary": {
            "n_runs": len(tearsheets),
            "best_sharpe": max(
                (tearsheet.metrics.sharpe for tearsheet in tearsheets.values()), default=0.0
            ),
            "best_cagr": max(
                (tearsheet.metrics.cagr for tearsheet in tearsheets.values()), default=0.0
            ),
            "worst_max_drawdown": max(
                (tearsheet.metrics.max_drawdown for tearsheet in tearsheets.values()),
                default=0.0,
            ),
        },
    }
