"""Local CSV trade/equity journal for paper/live benchmark runs."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from src.core.types import OrderReceipt, OrderStatus
from src.execution.trade_journal import EQUITY_COLUMNS, TRADE_COLUMNS, TradeJournal

T0 = datetime(2026, 11, 9, 14, 0, tzinfo=UTC)


def _account(equity: float = 100_000.0) -> dict[str, float]:
    return {
        "net_liquidation": equity,
        "cash_balance": 80_000.0,
        "buying_power": 200_000.0,
        "initial_margin_used": 12_000.0,
        "maintenance_margin_used": 10_000.0,
        "excess_liquidity": 90_000.0,
        "margin_utilization": 0.12,
    }


def _receipt(status: OrderStatus = OrderStatus.FILLED) -> OrderReceipt:
    return OrderReceipt(
        order_id="ord-1",
        status=status,
        filled_quantity=2,
        average_price=5_000.0,
        commission=2.5,
        timestamp=T0,
        instrument_id="ES",
        side="BUY",
        requested_quantity=2,
        contract_symbol="ESZ6",
    )


def test_trade_journal_writes_trades_and_equity_csv(tmp_path: Path) -> None:
    journal = TradeJournal(tmp_path / "live_trades", mode="paper", account_id="DU123")
    journal.record_account(_account(), timestamp=T0)
    journal.record_trade(_receipt(), account=_account(), timestamp=T0)
    journal.record_account(_account(101_000.0), timestamp=T0 + timedelta(days=1))
    journal.record_trade(_receipt(OrderStatus.REJECTED), account=_account(101_000.0), timestamp=T0)

    assert journal.trades_path.read_text(encoding="utf-8").splitlines()[0].split(",") == list(
        TRADE_COLUMNS
    )
    assert journal.equity_path.read_text(encoding="utf-8").splitlines()[0].split(",") == list(
        EQUITY_COLUMNS
    )
    snapshot = journal.snapshot()
    assert snapshot["summary"] == {
        "starting_equity": 100_000.0,
        "latest_equity": 101_000.0,
        "total_pnl": 1_000.0,
        "total_return": pytest.approx(0.01),
        "max_drawdown": 0.0,
        "trade_rows": 2,
        "filled_trades": 1,
        "equity_points": 2,
        "last_trade_at": T0.isoformat(),
    }
    trade = snapshot["trades"][0]
    assert trade["instrument_id"] == "ES"
    assert trade["side"] == "BUY"
    assert trade["filled_quantity"] == 2.0
    assert trade["commission"] == 2.5
    assert trade["net_liquidation"] == 100_000.0
    assert snapshot["equity"][0]["daily_pnl"] == 0.0
    assert snapshot["equity"][1]["cumulative_pnl"] == 1_000.0


def test_trade_journal_reports_empty_snapshot_and_drawdown(tmp_path: Path) -> None:
    journal = TradeJournal(tmp_path / "live_trades", mode="paper")
    empty = journal.snapshot()
    assert empty["trades"] == []
    assert empty["equity"] == []
    assert empty["summary"]["trade_rows"] == 0
    assert empty["summary"]["max_drawdown"] is None

    journal.record_account(_account(100_000.0), timestamp=T0)
    journal.record_account(_account(90_000.0), timestamp=T0 + timedelta(hours=1))
    snapshot = journal.snapshot()
    assert snapshot["summary"]["max_drawdown"] == pytest.approx(0.10)
    assert snapshot["summary"]["total_pnl"] == pytest.approx(-10_000.0)
    assert snapshot["equity"][1]["daily_pnl"] == pytest.approx(-10_000.0)


def test_trade_journal_survives_unwritable_directory(tmp_path: Path) -> None:
    blocked = tmp_path / "file"
    blocked.write_text("not a directory", encoding="utf-8")
    journal = TradeJournal(blocked, mode="paper")
    journal.record_account(_account())
    journal.record_trade(_receipt())
    assert journal.snapshot()["trades"] == []
