"""Unit tests for the deterministic in-memory simulated broker."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from configs.instruments import InstrumentsConfig
from src.backtest.cost_model import CostModel
from src.backtest.margin_simulator import MarginSimulator
from src.backtest.simulated_broker import (
    BarQuote,
    FillPriceMode,
    FillRecord,
    SimulatedBroker,
    TradeKind,
)
from src.core.interfaces import BrokerAdapter
from src.core.types import OrderEvent, OrderSide, OrderStatus, OrderType

TIMESTAMP = datetime(2024, 1, 2, 21, 0, tzinfo=UTC)
EXPIRATION = date(2024, 3, 15)


@pytest.fixture()
def cost_model(instrument_master: InstrumentsConfig) -> CostModel:
    return CostModel(
        instrument_master=instrument_master,
        impact_coefficient=0.0,
        base_slippage_ticks=0.0,
    )


@pytest.fixture()
def broker(instrument_master: InstrumentsConfig, cost_model: CostModel) -> SimulatedBroker:
    instance = SimulatedBroker(
        instrument_master,
        cost_model,
        margin_simulator=MarginSimulator(instrument_master),
        initial_cash=1_000_000.0,
        timestamp=TIMESTAMP,
    )
    instance.set_market(
        TIMESTAMP,
        {
            "ES": BarQuote(
                "ES",
                open=4800.0,
                high=4810.0,
                low=4790.0,
                close=4800.0,
                volume=250_000.0,
                bid=4799.875,
                ask=4800.125,
                expiration=EXPIRATION,
            )
        },
    )
    return instance


def _order(
    broker: SimulatedBroker,
    root: str,
    quantity: int,
    side: str = "BUY",
    order_id: str = "test-order",
) -> OrderEvent:
    return OrderEvent.market(
        order_id=order_id,
        timestamp=broker.timestamp,
        instrument_id=root,
        side=OrderSide.BUY if side == "BUY" else OrderSide.SELL,
        quantity=quantity,
    )


def test_market_buy_fill_and_accounting(broker: SimulatedBroker) -> None:
    receipt = broker.place_order(_order(broker, "ES", 2))
    assert receipt.status is OrderStatus.FILLED
    assert receipt.filled_quantity == 2
    assert receipt.average_price is not None
    assert receipt.average_price > 4800.0  # paid the half-spread
    assert receipt.commission == pytest.approx(2 * (0.85 + 1.40))
    position = broker.get_positions_sync()["ES"]
    assert position.quantity == 2
    assert position.average_price == pytest.approx(receipt.average_price)
    assert position.market_price == pytest.approx(4800.0)
    assert position.multiplier == 50.0
    assert position.initial_margin == pytest.approx(15_000.0)
    assert broker.cash == pytest.approx(1_000_000.0 - receipt.commission)
    # The entry execution cost is carried as unrealised PnL until settlement.
    assert broker.net_liquidation < 1_000_000.0
    fill = broker.fills[-1]
    assert isinstance(fill, FillRecord)
    assert fill.kind is TradeKind.REBALANCE
    assert fill.signed_quantity == 2
    core_fill = fill.to_core_fill()
    assert core_fill.quantity == 2
    assert core_fill.commission == pytest.approx(fill.commissions)


def test_market_sell_opens_short(broker: SimulatedBroker) -> None:
    receipt = broker.place_order(_order(broker, "ES", 3, side="SELL"))
    assert receipt.status is OrderStatus.FILLED
    position = broker.get_positions_sync()["ES"]
    assert position.quantity == -3
    assert position.average_price < 4800.0  # received the bid side
    assert broker.fills[-1].signed_quantity == -3


def test_reducing_and_flipping_positions(broker: SimulatedBroker) -> None:
    broker.place_order(_order(broker, "ES", 2, order_id="open"))
    cash_after_open = broker.cash
    close = broker.place_order(_order(broker, "ES", 2, side="SELL", order_id="close"))
    assert close.status is OrderStatus.FILLED
    position = broker.get_positions_sync()["ES"]
    assert position.quantity == 0
    # Closing at the bid realises the entry spread plus the exit spread.
    assert broker.cash < cash_after_open + 2 * (4800.0 - 4800.0) * 50.0
    flip = broker.place_order(_order(broker, "ES", 1, side="SELL", order_id="flip"))
    assert flip.status is OrderStatus.FILLED
    assert broker.get_positions_sync()["ES"].quantity == -1


def test_limit_order_fill_and_rejection(broker: SimulatedBroker) -> None:
    buy_limit = OrderEvent(
        order_id="limit-buy",
        timestamp=broker.timestamp,
        instrument_id="ES",
        side=OrderSide.BUY,
        quantity=1,
        order_type=OrderType.LIMIT,
        limit_price=4795.0,
    )
    receipt = broker.place_order(buy_limit)
    assert receipt.status is OrderStatus.FILLED
    assert receipt.average_price == pytest.approx(4795.0)
    assert broker.fills[-1].slippage == pytest.approx(0.0)
    stale = OrderEvent(
        order_id="limit-stale",
        timestamp=broker.timestamp,
        instrument_id="ES",
        side=OrderSide.BUY,
        quantity=1,
        order_type=OrderType.LIMIT,
        limit_price=4700.0,
    )
    rejected = broker.place_order(stale)
    assert rejected.status is OrderStatus.REJECTED
    assert "not triggered" in rejected.message
    sell_limit = OrderEvent(
        order_id="limit-sell",
        timestamp=broker.timestamp,
        instrument_id="ES",
        side=OrderSide.SELL,
        quantity=1,
        order_type=OrderType.LIMIT,
        limit_price=4805.0,
    )
    assert broker.place_order(sell_limit).status is OrderStatus.FILLED


def test_stop_orders(broker: SimulatedBroker) -> None:
    buy_stop = OrderEvent(
        order_id="stop-buy",
        timestamp=broker.timestamp,
        instrument_id="ES",
        side=OrderSide.BUY,
        quantity=1,
        order_type=OrderType.STOP,
        stop_price=4805.0,
    )
    receipt = broker.place_order(buy_stop)
    assert receipt.status is OrderStatus.FILLED
    assert receipt.average_price is not None and receipt.average_price >= 4805.0
    sell_stop = OrderEvent(
        order_id="stop-sell",
        timestamp=broker.timestamp,
        instrument_id="ES",
        side=OrderSide.SELL,
        quantity=1,
        order_type=OrderType.STOP,
        stop_price=4795.0,
    )
    assert broker.place_order(sell_stop).status is OrderStatus.FILLED
    untriggered = OrderEvent(
        order_id="stop-untriggered",
        timestamp=broker.timestamp,
        instrument_id="ES",
        side=OrderSide.BUY,
        quantity=1,
        order_type=OrderType.STOP,
        stop_price=4900.0,
    )
    assert broker.place_order(untriggered).status is OrderStatus.REJECTED
    stop_limit = OrderEvent(
        order_id="stop-limit",
        timestamp=broker.timestamp,
        instrument_id="ES",
        side=OrderSide.BUY,
        quantity=1,
        order_type=OrderType.STOP_LIMIT,
        stop_price=4805.0,
        limit_price=4808.0,
    )
    assert broker.place_order(stop_limit).status is OrderStatus.FILLED


def test_partial_fills_and_working_orders(
    instrument_master: InstrumentsConfig, cost_model: CostModel
) -> None:
    broker = SimulatedBroker(
        instrument_master,
        cost_model,
        initial_cash=1_000_000.0,
        max_participation=0.5,
        timestamp=TIMESTAMP,
    )
    broker.set_market(
        TIMESTAMP,
        {"ES": BarQuote("ES", 4800.0, 4810.0, 4790.0, 4800.0, volume=100.0)},
    )
    first = broker.place_order(_order(broker, "ES", 100, order_id="partial"))
    assert first.status is OrderStatus.PARTIALLY_FILLED
    assert first.filled_quantity == 50
    assert broker.get_positions_sync()["ES"].quantity == 50
    assert broker.working_orders and broker.working_orders[0].order_id == "partial"
    # New bar with deeper liquidity completes the order.
    broker.set_market(
        datetime(2024, 1, 3, 21, 0, tzinfo=UTC),
        {"ES": BarQuote("ES", 4800.0, 4810.0, 4790.0, 4800.0, volume=10_000.0)},
    )
    receipts = broker.process_working_orders()
    assert receipts and receipts[0].status is OrderStatus.FILLED
    assert broker.get_positions_sync()["ES"].quantity == 100
    assert not broker.working_orders

    broker.set_market(
        datetime(2024, 1, 4, 21, 0, tzinfo=UTC),
        {"ES": BarQuote("ES", 4800.0, 4810.0, 4790.0, 4800.0, volume=100.0)},
    )
    second = broker.place_order(_order(broker, "ES", 100, order_id="cancel-me"))
    assert second.status is OrderStatus.PARTIALLY_FILLED
    cancelled = broker.cancel_order_sync("cancel-me")
    assert cancelled.status is OrderStatus.CANCELLED
    missing = broker.cancel_order_sync("does-not-exist")
    assert missing.status is OrderStatus.UNKNOWN


def test_reject_without_market_data(
    instrument_master: InstrumentsConfig, cost_model: CostModel
) -> None:
    broker = SimulatedBroker(instrument_master, cost_model, initial_cash=1_000_000.0)
    receipt = broker.place_order(_order(broker, "NQ", 1))
    assert receipt.status is OrderStatus.REJECTED


def test_no_liquidity_keeps_order_working(
    instrument_master: InstrumentsConfig, cost_model: CostModel
) -> None:
    broker = SimulatedBroker(
        instrument_master,
        cost_model,
        initial_cash=1_000_000.0,
        max_participation=0.5,
        timestamp=TIMESTAMP,
    )
    broker.set_market(
        TIMESTAMP,
        {"ES": BarQuote("ES", 4800.0, 4800.0, 4800.0, 4800.0, volume=0.0)},
    )
    receipt = broker.place_order(_order(broker, "ES", 1))
    assert receipt.status is OrderStatus.SUBMITTED
    assert broker.working_orders
    assert broker.process_working_orders()[0].status is OrderStatus.SUBMITTED


def test_flatten_all_and_cancel_all(broker: SimulatedBroker) -> None:
    broker.place_order(_order(broker, "ES", 3, order_id="long"))
    broker.place_order(_order(broker, "ES", 1, side="SELL", order_id="offset"))
    broker.place_order(_order(broker, "ES", 1, side="SELL", order_id="short"))
    receipts = broker.flatten_all_sync(kind=TradeKind.FORCED_LIQUIDATION)
    assert len(receipts) == 1
    assert receipts[0].status is OrderStatus.FILLED
    assert broker.get_positions_sync()["ES"].quantity == 0
    assert broker.fills[-1].kind is TradeKind.FORCED_LIQUIDATION
    assert broker.cancel_all_sync() == []  # nothing working after flatten


def test_settlement(broker: SimulatedBroker) -> None:
    broker.place_order(_order(broker, "ES", 2))
    broker.set_market(
        datetime(2024, 1, 3, 21, 0, tzinfo=UTC),
        {
            "ES": BarQuote(
                "ES",
                open=4805.0,
                high=4820.0,
                low=4800.0,
                close=4810.0,
                volume=250_000.0,
                settlement=4810.0,
            )
        },
    )
    cash_before = broker.cash
    result = broker.settle()
    assert result.settlement_pnl != 0.0
    assert broker.cash == pytest.approx(cash_before + result.settlement_pnl)
    position = broker.get_positions_sync()["ES"]
    assert position.average_price == pytest.approx(4810.0)
    assert position.market_price == pytest.approx(4810.0)
    assert position.unrealized_pnl == pytest.approx(0.0)
    explicit = broker.settle({"ES": 4820.0})
    assert explicit.settlement_pnl == pytest.approx(2 * 10.0 * 50.0)


def test_account_snapshot_and_margins(broker: SimulatedBroker) -> None:
    broker.place_order(_order(broker, "ES", 2))
    snapshot = broker.get_account_sync()
    assert snapshot.initial_margin_used == pytest.approx(30_000.0)
    assert snapshot.maintenance_margin_used == pytest.approx(27_200.0)
    initial, maintenance = broker.margin_requirements()
    assert (initial, maintenance) == pytest.approx((30_000.0, 27_200.0))
    assert snapshot.net_liquidation == pytest.approx(broker.net_liquidation)


def test_account_snapshot_raises_on_depleted_account(
    instrument_master: InstrumentsConfig, cost_model: CostModel
) -> None:
    broker = SimulatedBroker(
        instrument_master, cost_model, initial_cash=1_000.0, timestamp=TIMESTAMP
    )
    broker.set_market(
        TIMESTAMP,
        {"ES": BarQuote("ES", 4800.0, 4800.0, 4800.0, 4800.0, volume=1.0)},
    )
    broker.apply_fill("ES", OrderSide.BUY, 1, 4800.0)
    broker.set_market(
        datetime(2024, 1, 3, 21, 0, tzinfo=UTC),
        {"ES": BarQuote("ES", 4700.0, 4700.0, 4700.0, 4700.0, volume=1.0)},
    )
    with pytest.raises(ValueError, match="non-positive"):
        broker.get_account_sync()


def test_external_fill_and_roll(broker: SimulatedBroker) -> None:
    broker.apply_fill("ES", OrderSide.BUY, 2, 4800.0, commission=1.0, fees=2.0)
    position = broker.get_positions_sync()["ES"]
    assert position.quantity == 2
    assert broker.cash == pytest.approx(1_000_000.0 - 3.0)
    close_receipt, open_receipt = broker.roll_position(
        "ES",
        from_price=4800.0,
        to_price=4805.0,
        to_expiration=date(2024, 6, 21),
        to_contract_symbol="ES_20240621",
        use_bag=False,
    )
    assert close_receipt.status is OrderStatus.FILLED
    assert open_receipt.status is OrderStatus.FILLED
    position = broker.get_positions_sync()["ES"]
    assert position.quantity == 2
    assert position.expiration == date(2024, 6, 21)
    assert position.contract_symbol == "ES_20240621"
    roll_fills = [
        fill for fill in broker.fills if fill.kind in {TradeKind.ROLL_CLOSE, TradeKind.ROLL_OPEN}
    ]
    assert len(roll_fills) == 2
    assert broker.cash < 1_000_000.0 - 3.0


def test_roll_validation(broker: SimulatedBroker) -> None:
    with pytest.raises(ValueError, match="no open position"):
        broker.roll_position("ES", from_price=4800.0, to_price=4801.0)
    broker.place_order(_order(broker, "ES", 1))
    with pytest.raises(ValueError):
        broker.roll_position("ES", from_price=-1.0, to_price=4801.0)
    with pytest.raises(ValueError):
        broker.roll_position("ES", from_price=4800.0, to_price=0.0)


def test_bar_quote_validation() -> None:
    with pytest.raises(ValueError):
        BarQuote("", 1.0, 1.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        BarQuote("ES", 1.0, 1.0, 1.0, 0.0)
    with pytest.raises(ValueError):
        BarQuote("ES", 2.0, 1.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        BarQuote("ES", 1.0, 1.0, 1.0, 1.0, volume=-1.0)
    with pytest.raises(ValueError):
        BarQuote("ES", 1.0, 1.0, 1.0, 1.0, bid=1.0)
    with pytest.raises(ValueError):
        BarQuote("ES", 1.0, 1.0, 1.0, 1.0, bid=1.1, ask=1.0)
    with pytest.raises(ValueError):
        BarQuote("ES", 1.0, 1.0, 1.0, 1.0, settlement=-5.0)
    quote = BarQuote("es", 1.0, 1.0, 1.0, 1.0, bid=0.9, ask=1.1)
    assert quote.root == "ES"
    assert quote.mid == pytest.approx(1.0)
    assert quote.spread_price == pytest.approx(0.2)


def test_broker_construction_validation(
    instrument_master: InstrumentsConfig, cost_model: CostModel
) -> None:
    with pytest.raises(ValueError):
        SimulatedBroker(instrument_master, cost_model, initial_cash=0.0)
    with pytest.raises(TypeError):
        SimulatedBroker(
            instrument_master,
            cost_model,
            fill_price_mode="close",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError):
        SimulatedBroker(instrument_master, cost_model, max_participation=0.0)
    with pytest.raises(ValueError):
        SimulatedBroker(instrument_master, cost_model, max_participation=1.5)
    with pytest.raises(ValueError):
        SimulatedBroker(instrument_master, cost_model, name="")


def test_set_market_validation(broker: SimulatedBroker) -> None:
    with pytest.raises(ValueError):
        broker.set_market(TIMESTAMP, {"NQ": BarQuote("ES", 1.0, 1.0, 1.0, 1.0)})
    broker.update_quote(BarQuote("NQ", 16_800.0, 16_800.0, 16_800.0, 16_800.0, volume=1.0))
    broker.reset_monthly_volume()
    assert broker.working_orders == ()


def test_slippage_model_used_by_broker(
    instrument_master: InstrumentsConfig,
) -> None:
    stiff = CostModel(instrument_master=instrument_master, impact_coefficient=50.0)
    broker = SimulatedBroker(
        instrument_master, stiff, initial_cash=1_000_000.0, timestamp=TIMESTAMP
    )
    broker.set_market(
        TIMESTAMP,
        {"ES": BarQuote("ES", 4800.0, 4810.0, 4790.0, 4800.0, volume=10.0)},
    )
    receipt = broker.place_order(_order(broker, "ES", 5))
    assert receipt.status is OrderStatus.FILLED
    assert receipt.average_price is not None and receipt.average_price > 4801.0


def test_fill_price_modes(instrument_master: InstrumentsConfig, cost_model: CostModel) -> None:
    for mode in (FillPriceMode.OPEN, FillPriceMode.MID, FillPriceMode.CLOSE):
        broker = SimulatedBroker(
            instrument_master,
            cost_model,
            initial_cash=1_000_000.0,
            fill_price_mode=mode,
            timestamp=TIMESTAMP,
        )
        broker.set_market(
            TIMESTAMP,
            {
                "ES": BarQuote(
                    "ES", 4800.0, 4810.0, 4790.0, 4805.0, volume=1.0, bid=4804.0, ask=4806.0
                )
            },
        )
        receipt = broker.place_order(_order(broker, "ES", 1))
        assert receipt.status is OrderStatus.FILLED
        expected = {
            FillPriceMode.OPEN: 4800.0,
            FillPriceMode.MID: 4805.0,
            FillPriceMode.CLOSE: 4805.0,
        }[mode]
        assert receipt.average_price is not None and receipt.average_price >= expected


async def test_async_protocol_surface(broker: SimulatedBroker) -> None:
    assert isinstance(broker, BrokerAdapter)
    assert broker.is_connected is False
    await broker.connect()
    assert broker.is_connected is True
    order = _order(broker, "ES", 1, order_id="async-order")
    receipt = await broker.submit_order(order)
    assert receipt.status is OrderStatus.FILLED
    receipt = await broker.submit_order(_order(broker, "ES", 1, order_id="async-second"))
    assert receipt.status is OrderStatus.FILLED
    receipts = await broker.submit_orders(
        [_order(broker, "ES", 1, side="SELL", order_id="async-2")]
    )
    assert receipts[0].status is OrderStatus.FILLED
    snapshot = await broker.get_account()
    assert snapshot.net_liquidation > 0.0
    positions = await broker.get_positions()
    assert "ES" in positions
    flat = await broker.flatten_all()
    assert flat
    assert (await broker.get_positions())["ES"].quantity == 0
    assert await broker.cancel_all() == []
    unknown = await broker.cancel_order("missing")
    assert unknown.status is OrderStatus.UNKNOWN
    await broker.disconnect()
    assert broker.is_connected is False
