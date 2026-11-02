"""Unit tests for the JSON + standalone-HTML tear sheet generator."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from src.analytics.metrics import compute_metrics_from_result
from src.analytics.tearsheet import (
    Tearsheet,
    _downsample,
    _svg_line,
    _svg_waterfall,
    build_tearsheet,
    compare_tearsheets,
    ensure_month_key,
    ensure_utc_datetime,
    write_tearsheet,
)
from src.backtest.engine import (
    EQUITY_CURVE_SCHEMA,
    BacktestMode,
    BacktestResult,
    ForcedLiquidationEvent,
    RollExecution,
    TradeRecord,
)
from src.core.types import OrderSide


def make_equity_curve() -> pl.DataFrame:
    timestamps = [
        datetime(2024, 1, 2, 21, 0, tzinfo=UTC) + timedelta(days=7 * index) for index in range(6)
    ]
    equities = [1_000_000.0, 1_010_000.0, 1_005_000.0, 1_025_000.0, 1_015_000.0, 1_040_000.0]
    rows = []
    for index, (timestamp, equity) in enumerate(zip(timestamps, equities, strict=True)):
        rows.append(
            {
                "timestamp_utc": timestamp,
                "equity": equity,
                "cash": equity,
                "unrealized_pnl": 0.0,
                "price_pnl": 4_000.0 if index % 2 else -1_500.0,
                "roll_pnl": -500.0 if index == 2 else 0.0,
                "commissions": 100.0,
                "fees": 20.0,
                "slippage": 30.0,
                "funding": 0.0,
                "margin_used": 120_000.0 + index * 1_000.0,
                "margin_utilization": (120_000.0 + index * 1_000.0) / equity,
                "gross_exposure": 600_000.0,
                "net_exposure": 100_000.0,
                "drawdown": 0.0,
                "turnover": 250_000.0,
                "n_positions": 3,
            }
        )
    return pl.DataFrame(rows).select(
        [pl.col(column).cast(dtype) for column, dtype in EQUITY_CURVE_SCHEMA.items()]
    )


def make_trades() -> tuple[TradeRecord, ...]:
    timestamp = datetime(2024, 1, 2, 21, 0, tzinfo=UTC)
    return (
        TradeRecord(
            timestamp=timestamp,
            order_id="trade-1",
            instrument_id="ES",
            side=OrderSide.BUY,
            quantity=2,
            price=4800.125,
            reference_price=4800.0,
            kind="entry",
            notional=480_012.5,
            commissions=5.1,
            slippage=12.5,
            execution_cost_ticks=0.5,
        ),
        TradeRecord(
            timestamp=timestamp + timedelta(days=7),
            order_id="trade-2",
            instrument_id="CL",
            side=OrderSide.SELL,
            quantity=3,
            price=72.49,
            reference_price=72.5,
            kind="exit",
            notional=217_470.0,
            commissions=7.65,
            slippage=15.0,
            execution_cost_ticks=1.5,
        ),
    )


def make_result(
    *,
    strategy_id: str = "unit-strategy",
    equity_curve: pl.DataFrame | None = None,
    warnings: tuple[str, ...] = (),
) -> BacktestResult:
    curve = equity_curve if equity_curve is not None else make_equity_curve()
    trades = make_trades()
    timestamp = datetime(2024, 1, 2, 21, 0, tzinfo=UTC)
    roll_events = (
        RollExecution(
            timestamp=timestamp + timedelta(days=14),
            instrument_id="ES",
            quantity=2,
            from_expiration=datetime(2024, 3, 15, tzinfo=UTC).date(),
            to_expiration=datetime(2024, 6, 21, tzinfo=UTC).date(),
            from_price=4810.0,
            to_price=4812.5,
            roll_pnl=-250.0,
            total_cost=36.0,
            bag_used=False,
        ),
    )
    liquidations = (
        ForcedLiquidationEvent(
            timestamp=timestamp + timedelta(days=21),
            equity_before=20_000.0,
            maintenance_margin=27_200.0,
            equity_after=19_500.0,
            positions_closed=("ES",),
            penalty_cost=49.0,
            reason="equity below maintenance margin",
        ),
    )
    return BacktestResult(
        run_id="run-2024-01",
        strategy_id=strategy_id,
        mode=BacktestMode.EVENT,
        start=curve["timestamp_utc"][0],
        end=curve["timestamp_utc"][-1],
        initial_equity=1_000_000.0,
        ending_equity=float(curve["equity"][-1]),
        equity_curve=curve,
        trades=trades,
        roll_events=roll_events,
        forced_liquidations=liquidations,
        price_pnl=27_500.0,
        roll_pnl=-500.0,
        commissions=600.0,
        fees=120.0,
        slippage=180.0,
        funding=0.0,
        turnover=1_500_000.0,
        margin_used_max=125_000.0,
        margin_utilization_avg=0.121,
        margin_utilization_peak=0.125,
        settlement_pnl=0.0,
        metadata={"strategy": strategy_id},
        warnings=warnings,
    )


def test_build_tearsheet_structure() -> None:
    result = make_result()
    metrics = compute_metrics_from_result(result)
    generated = datetime(2024, 2, 1, 12, 0, tzinfo=UTC)
    tearsheet = build_tearsheet(result, metrics, generated_at=generated)
    assert isinstance(tearsheet, Tearsheet)
    assert tearsheet.generated_at == generated
    assert tearsheet.title
    assert tearsheet.metrics is metrics
    payload = tearsheet.payload
    for key in (
        "meta",
        "equity_curve",
        "rolling_sharpe",
        "monthly_returns",
        "attribution",
        "rachev",
        "trades",
        "rolls",
        "forced_liquidations",
        "accounting",
    ):
        assert key in payload, key
    meta = payload["meta"]
    assert isinstance(meta, dict)
    assert meta["mode"] == "event"
    assert meta["n_bars"] == 6
    equity_payload = payload["equity_curve"]
    assert isinstance(equity_payload, dict)
    assert len(equity_payload["equity"]) == 6
    assert len(equity_payload["timestamps"]) == 6
    monthly = payload["monthly_returns"]
    assert isinstance(monthly, list) and monthly
    assert {"month", "return"} <= set(monthly[0])
    rachev = payload["rachev"]
    assert isinstance(rachev, dict)
    assert rachev["alpha"] == pytest.approx(0.05)
    assert "lower_tail" in rachev and "upper_tail" in rachev
    trades_payload = payload["trades"]
    assert isinstance(trades_payload, dict)
    assert trades_payload["count"] == 2
    assert trades_payload["by_kind"] == {"entry": 1, "exit": 1}
    accounting = payload["accounting"]
    assert isinstance(accounting, dict)
    assert accounting["residual"] == pytest.approx(result.accounting_residual)
    # The whole payload must be JSON-serialisable.
    encoded = tearsheet.to_json()
    decoded = json.loads(encoded)
    assert decoded["run_id"] == "run-2024-01"
    assert decoded["metrics"]["sharpe"] == pytest.approx(metrics.sharpe)
    assert isinstance(tearsheet.to_dict(), dict)


def test_tearsheet_html_contents_and_escaping() -> None:
    result = make_result(strategy_id="<script>alert('x')</script>", warnings=("<b>warn</b>",))
    tearsheet = build_tearsheet(result, title="<b>Unsafe & title</b>")
    html = tearsheet.html
    assert "<!DOCTYPE html>" in html
    assert "<style>" in html
    assert "<svg" in html
    assert "Equity Curve" in html
    assert "Underwater Drawdown" in html
    assert "Rolling Sharpe" in html
    assert "PnL Attribution" in html
    assert "Spot vs Roll Yield Attribution" in html
    assert "Roll Log" in html
    # Untrusted strings must be escaped, never injected as markup.
    assert "<script>alert('x')</script>" not in html
    assert "&lt;script&gt;" in html
    assert "<b>Unsafe &amp; title</b>" not in html
    assert "&lt;b&gt;Unsafe &amp; title&lt;/b&gt;" in html
    assert "&lt;b&gt;warn&lt;/b&gt;" in html
    assert "Accounting identity residual" in html


def test_tearsheet_html_without_rolls_or_warnings() -> None:
    curve = make_equity_curve()
    result = BacktestResult(
        run_id="flat-run",
        strategy_id="flat",
        mode=BacktestMode.VECTORIZED,
        start=curve["timestamp_utc"][0],
        end=curve["timestamp_utc"][-1],
        initial_equity=1_000_000.0,
        ending_equity=float(curve["equity"][-1]),
        equity_curve=curve,
    )
    tearsheet = build_tearsheet(result)
    assert "no rolls executed" in tearsheet.html
    assert "<li>none</li>" in tearsheet.html


def test_write_tearsheet_json_and_html(tmp_path: Path) -> None:
    tearsheet = build_tearsheet(make_result())
    json_path = write_tearsheet(tearsheet, tmp_path / "report.json")
    assert json_path.exists()
    decoded = json.loads(json_path.read_text(encoding="utf-8"))
    assert decoded["title"] == tearsheet.title
    html_path = write_tearsheet(tearsheet, tmp_path / "report.html")
    assert html_path.exists()
    assert "<!DOCTYPE html>" in html_path.read_text(encoding="utf-8")
    # Explicit format overrides an unusual suffix.
    custom = write_tearsheet(tearsheet, tmp_path / "report.txt", fmt="json")
    assert json.loads(custom.read_text(encoding="utf-8"))["run_id"] == tearsheet.run_id
    # A BacktestResult is accepted and converted.
    from_result = write_tearsheet(make_result(), tmp_path / "from-result.json")
    assert json.loads(from_result.read_text(encoding="utf-8"))["strategy_id"]
    with pytest.raises(ValueError):
        write_tearsheet(tearsheet, tmp_path / "bad.yaml", fmt="yaml")
    # Tearsheet.write delegates to write_tearsheet.
    delegated = tearsheet.write(tmp_path / "delegated.json")
    assert delegated.exists()


def test_compare_tearsheets() -> None:
    first = build_tearsheet(make_result(strategy_id="tsmom"))
    second = build_tearsheet(make_result(strategy_id="drl"))
    benchmark = build_tearsheet(make_result(strategy_id="benchmark"))
    comparison = compare_tearsheets({"tsmom": first, "drl": second}, benchmark=benchmark)
    assert comparison["summary"]["n_runs"] == 2
    table = comparison["comparison_table"]
    assert isinstance(table, list) and len(table) == 3
    assert table[-1]["name"] == "benchmark"
    runs = comparison["runs"]
    assert isinstance(runs, dict)
    assert set(runs) == {"tsmom", "drl"}
    run_payload = runs["tsmom"]
    assert {"headline", "normalised_equity_curve", "drawdown", "rolling_sharpe"} <= set(run_payload)
    assert run_payload["normalised_equity_curve"][0] == pytest.approx(1.0)
    assert comparison["benchmark"] is not None
    # JSON-serialisable.
    json.dumps(comparison)
    with pytest.raises(ValueError):
        compare_tearsheets({})
    without_benchmark = compare_tearsheets({"only": first})
    assert without_benchmark["benchmark"] is None


def test_build_tearsheet_validation() -> None:
    result = make_result()
    with pytest.raises(ValueError):
        build_tearsheet(result, rolling_window=1)
    empty = pl.DataFrame(schema=EQUITY_CURVE_SCHEMA)
    empty_result = BacktestResult(
        run_id="empty",
        strategy_id="empty",
        mode=BacktestMode.EVENT,
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 1, 2, tzinfo=UTC),
        initial_equity=1_000_000.0,
        ending_equity=1_000_000.0,
        equity_curve=empty,
    )
    with pytest.raises(ValueError):
        build_tearsheet(empty_result)


def test_svg_and_serialisation_helpers() -> None:
    line = _svg_line([1.0, 2.0, 1.5])
    assert "<polyline" in line and "<polygon" in line
    assert "insufficient data" in _svg_line([1.0])
    assert "insufficient data" in _svg_line([])
    zero_crossing = _svg_line([-1.0, 1.0], zero_line=True)
    assert "stroke-dasharray" in zero_crossing
    waterfall = _svg_waterfall(["a", "b"], [1.0, -0.5])
    assert waterfall.count("<rect") == 2
    assert "no attribution data" in _svg_waterfall([], [])
    assert _downsample([1.0, 2.0, 3.0], 5) == [1.0, 2.0, 3.0]
    assert len(_downsample([float(value) for value in range(100)], 10)) <= 11
    assert ensure_month_key(datetime(2024, 3, 15, tzinfo=UTC)) == "2024-03"
    naive = datetime(2024, 3, 15, 12, 0)
    assert ensure_month_key(naive) == "2024-03"
    assert ensure_utc_datetime(naive).tzinfo is not None
    aware = datetime(2024, 3, 15, 12, 0, tzinfo=UTC)
    assert ensure_utc_datetime(aware) == aware


def test_tearsheet_validation() -> None:
    with pytest.raises(ValueError):
        Tearsheet(
            title="",
            strategy_id="x",
            run_id="x",
            generated_at=datetime(2024, 1, 1, tzinfo=UTC),
            metrics=compute_metrics_from_result(make_result()),
        )
