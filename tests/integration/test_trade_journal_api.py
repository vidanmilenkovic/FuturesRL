"""Read-only API surface for the local paper/live trade journal."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from configs.settings import Settings
from fastapi.testclient import TestClient
from src.api.main import create_app
from src.core.types import OrderReceipt, OrderStatus
from src.execution.trade_journal import TradeJournal

pytestmark = pytest.mark.integration

T0 = datetime(2026, 11, 9, 14, 0, tzinfo=UTC)


def _client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings, clock=lambda: T0)) as client:
        yield client


def test_trade_journal_api_and_csv_downloads(settings: Settings) -> None:
    settings.environment = "paper"
    journal = TradeJournal.from_settings(settings)
    journal.record_account(
        {"net_liquidation": 100_000.0, "cash_balance": 80_000.0, "margin_utilization": 0.12},
        timestamp=T0,
    )
    journal.record_trade(
        OrderReceipt(
            order_id="ord-1",
            status=OrderStatus.FILLED,
            filled_quantity=2,
            average_price=5_000.0,
            commission=2.5,
            timestamp=T0,
            instrument_id="ES",
            side="BUY",
            requested_quantity=2,
        ),
        account={"net_liquidation": 100_000.0},
        timestamp=T0,
    )

    for client in _client(settings):
        response = client.get("/api/runner/trade-journal")
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["enabled"] is True
        assert body["summary"]["filled_trades"] == 1
        assert body["trades"][0]["instrument_id"] == "ES"
        assert body["trades_path"].endswith("trades.csv")

        trades = client.get("/api/runner/trade-journal/trades.csv")
        assert trades.status_code == 200
        assert trades.headers["content-type"].startswith("text/csv")
        assert "instrument_id" in trades.text
        assert "ES" in trades.text

        equity = client.get("/api/runner/trade-journal/equity.csv")
        assert equity.status_code == 200
        assert "net_liquidation" in equity.text


def test_trade_journal_api_empty_and_missing_downloads(settings: Settings) -> None:
    settings.environment = "paper"
    for client in _client(settings):
        response = client.get("/api/runner/trade-journal")
        assert response.status_code == 200
        body = response.json()
        assert body["trades"] == []
        assert body["summary"]["trade_rows"] == 0
        assert client.get("/api/runner/trade-journal/trades.csv").status_code == 404
        assert client.get("/api/runner/trade-journal/equity.csv").status_code == 404
