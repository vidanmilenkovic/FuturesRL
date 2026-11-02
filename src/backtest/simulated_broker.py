"""In-memory simulated broker implementing the live ``BrokerAdapter`` contract.

The simulated broker is the event-driven backtester's execution venue and the
reference implementation of the unified broker protocol used by IBKR paper and
live trading. It owns cash and positions, matches market/limit/stop orders
against injected bar prices through the :class:`~src.backtest.cost_model.CostModel`
and delegates settlement/margin arithmetic to a
:class:`~src.backtest.margin_simulator.MarginSimulator`, so the exact same
accounting code runs in the vectorised and event-driven backtesters.

Design conventions
------------------
* **Fills are deterministic.** Market orders fill at the bar close (configurable
  to open or mid) plus the model's spread/impact cost. Limit orders fill only if
  the bar trades through the limit price and capture their limit (or the bar
  open when it is better). Stop orders trigger off the bar high/low.
* **Partial fills are opt-in.** With ``max_participation`` set, at most that
  fraction of the bar volume fills in one bar and the remainder stays working.
* **Futures accounting.** Opening a position moves no notional cash; cash changes
  only by realised PnL, commissions, fees and daily settlement. A ``Position``
  is always reported with a mark price so it stays compatible with IBKR
  telemetry.
* **The matching API is synchronous** (the backtester is a tight loop) while the
  public ``async`` methods implement the ``BrokerAdapter`` protocol; the sync
  equivalents are suffixed with ``_sync``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from enum import StrEnum
from types import MappingProxyType

from configs.instruments import InstrumentsConfig

from src.backtest.cost_model import CostBreakdown, CostModel
from src.backtest.margin_simulator import MarginSimulator, SettlementResult
from src.core.types import (
    AccountSnapshot,
    Fill,
    InstrumentId,
    OrderEvent,
    OrderReceipt,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    ensure_utc,
)

# ---------------------------------------------------------------------------
# Enumerations / records
# ---------------------------------------------------------------------------


class FillPriceMode(StrEnum):
    """Reference price used for market-order fills inside a bar."""

    OPEN = "open"
    CLOSE = "close"
    MID = "mid"


class TradeKind(StrEnum):
    """Semantic classification of a fill (used by PnL attribution/tear sheets)."""

    REBALANCE = "rebalance"
    ENTRY = "entry"
    EXIT = "exit"
    ROLL_CLOSE = "roll_close"
    ROLL_OPEN = "roll_open"
    FLATTEN = "flatten"
    FORCED_LIQUIDATION = "forced_liquidation"


@dataclass(frozen=True, slots=True)
class BarQuote:
    """One bar of market data for the active contract of a root."""

    root: str
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    bid: float | None = None
    ask: float | None = None
    expiration: date | None = None
    contract_symbol: str | None = None
    settlement: float | None = None

    def __post_init__(self) -> None:
        if not self.root:
            raise ValueError("root must be a non-empty string")
        object.__setattr__(self, "root", self.root.strip().upper())
        for name in ("open", "high", "low", "close"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and strictly positive")
            object.__setattr__(self, name, value)
        if not self.low <= min(self.open, self.close) <= max(self.open, self.close) <= self.high:
            raise ValueError(f"inconsistent OHLC quote for {self.root}")
        if not math.isfinite(self.volume) or self.volume < 0.0:
            raise ValueError("volume must be finite and non-negative")
        if (self.bid is None) != (self.ask is None):
            raise ValueError("bid and ask must be provided together")
        if (
            self.bid is not None
            and self.ask is not None
            and (self.bid <= 0.0 or self.ask <= 0.0 or self.bid > self.ask)
        ):
            raise ValueError("invalid bid/ask quote")
        if self.settlement is not None and (
            not math.isfinite(self.settlement) or self.settlement <= 0.0
        ):
            raise ValueError("settlement must be finite and strictly positive")

    @property
    def mid(self) -> float:
        """Mid price when a two-sided quote exists, otherwise the close."""
        if self.bid is not None and self.ask is not None:
            return 0.5 * (self.bid + self.ask)
        return self.close

    @property
    def spread_price(self) -> float | None:
        """Quoted bid/ask spread in price units."""
        if self.bid is None or self.ask is None:
            return None
        return self.ask - self.bid


@dataclass(frozen=True, slots=True)
class FillRecord:
    """A fill together with the full cost-model breakdown that produced it."""

    order_id: str
    instrument_id: InstrumentId
    side: OrderSide
    quantity: int
    price: float
    reference_price: float
    timestamp: datetime
    cost: CostBreakdown
    kind: TradeKind = TradeKind.REBALANCE
    message: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", ensure_utc(self.timestamp))
        if self.quantity <= 0:
            raise ValueError("fill quantity must be strictly positive")
        if self.price <= 0.0 or self.reference_price <= 0.0:
            raise ValueError("fill and reference prices must be strictly positive")

    @property
    def signed_quantity(self) -> int:
        """Signed fill quantity (+ for buys, - for sells)."""
        return self.side.multiplier * self.quantity

    @property
    def notional(self) -> float:
        """Traded notional in account currency."""
        return self.quantity * self.price * self.cost.multiplier

    @property
    def commissions(self) -> float:
        """Commission plus exchange/regulatory fees charged for the fill."""
        return self.cost.commission + self.cost.fees

    @property
    def slippage(self) -> float:
        """Price-impact cost (spread + volume impact) charged for the fill."""
        return self.cost.spread_cost + self.cost.slippage_cost

    @property
    def execution_cost_ticks(self) -> float:
        """Per-contract execution price impact in ticks."""
        return self.cost.execution_cost_ticks

    def to_core_fill(self) -> Fill:
        """Convert to the core :class:`~src.core.types.Fill` contract."""
        return Fill(
            order_id=self.order_id,
            instrument_id=self.instrument_id,
            side=self.side,
            quantity=self.quantity,
            price=self.price,
            timestamp=self.timestamp,
            commission=self.commissions,
            slippage_ticks=self.execution_cost_ticks,
        )

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable representation."""
        return {
            "order_id": self.order_id,
            "instrument_id": self.instrument_id,
            "side": self.side.value,
            "quantity": self.quantity,
            "price": self.price,
            "reference_price": self.reference_price,
            "timestamp": self.timestamp.isoformat(),
            "commissions": self.commissions,
            "slippage": self.slippage,
            "execution_cost_ticks": self.execution_cost_ticks,
            "kind": self.kind.value,
            "message": self.message,
        }


@dataclass(slots=True)
class _WorkingOrder:
    """Internal state of a partially filled / not-yet-triggered order."""

    order: OrderEvent
    remaining: int
    kind: TradeKind
    extra_slippage_ticks: float = 0.0
    extra_fee_per_contract: float = 0.0


# ---------------------------------------------------------------------------
# Simulated broker
# ---------------------------------------------------------------------------


class SimulatedBroker:
    """Deterministic in-memory broker implementing the ``BrokerAdapter`` protocol."""

    def __init__(
        self,
        instrument_master: InstrumentsConfig | None = None,
        cost_model: CostModel | None = None,
        *,
        margin_simulator: MarginSimulator | None = None,
        initial_cash: float = 1_000_000.0,
        fill_price_mode: FillPriceMode = FillPriceMode.CLOSE,
        max_participation: float | None = None,
        timestamp: datetime | None = None,
        name: str = "simulated",
    ) -> None:
        if not math.isfinite(initial_cash) or initial_cash <= 0.0:
            raise ValueError("initial_cash must be finite and strictly positive")
        if not isinstance(fill_price_mode, FillPriceMode):
            raise TypeError("fill_price_mode must be a FillPriceMode")
        if max_participation is not None and (
            not math.isfinite(max_participation) or not 0.0 < max_participation <= 1.0
        ):
            raise ValueError("max_participation must be within (0, 1]")
        if not name:
            raise ValueError("name must be a non-empty string")
        self._name = name
        self._instrument_master = instrument_master
        self._cost_model = cost_model or CostModel(instrument_master=instrument_master)
        self._margin_simulator = margin_simulator or MarginSimulator(instrument_master)
        self._cash = float(initial_cash)
        self._fill_price_mode = fill_price_mode
        self._max_participation = max_participation
        self._timestamp: datetime = (
            ensure_utc(timestamp) if timestamp is not None else datetime(1970, 1, 1, tzinfo=UTC)
        )
        self._connected = False
        self._quotes: dict[InstrumentId, BarQuote] = {}
        self._marks: dict[InstrumentId, float] = {}
        self._positions: dict[InstrumentId, Position] = {}
        self._working: dict[str, _WorkingOrder] = {}
        self._fills: list[FillRecord] = []
        self._order_counter = 0
        self._monthly_volume: dict[InstrumentId, int] = {}

    # -- protocol / state properties ------------------------------------------

    @property
    def name(self) -> str:
        """Broker identifier (``simulated``)."""
        return self._name

    @property
    def is_connected(self) -> bool:
        """Whether the simulated session is connected."""
        return self._connected

    @property
    def timestamp(self) -> datetime:
        """Timestamp of the most recent market snapshot."""
        return self._timestamp

    @property
    def cash(self) -> float:
        """Cash balance (futures cash changes by realised PnL and costs only)."""
        return self._cash

    @property
    def cost_model(self) -> CostModel:
        """The injected cost model."""
        return self._cost_model

    @property
    def margin_simulator(self) -> MarginSimulator:
        """The injected margin/settlement simulator."""
        return self._margin_simulator

    @property
    def fills(self) -> tuple[FillRecord, ...]:
        """Every fill executed so far, in execution order."""
        return tuple(self._fills)

    @property
    def working_orders(self) -> tuple[OrderEvent, ...]:
        """Orders that are still working (partial fills / untriggered stops)."""
        return tuple(working.order for working in self._working.values())

    @property
    def net_liquidation(self) -> float:
        """Cash plus the sum of unrealised PnL at current marks."""
        return self._margin_simulator.equity(self._cash, self.get_positions_sync())

    # -- market data injection -------------------------------------------------

    def set_market(
        self,
        timestamp: datetime,
        quotes: Mapping[InstrumentId, BarQuote] | Sequence[BarQuote],
    ) -> None:
        """Install the current bar quotes and mark every position."""
        self._timestamp = ensure_utc(timestamp)
        if isinstance(quotes, Mapping):
            items: tuple[tuple[InstrumentId, BarQuote], ...] = tuple(quotes.items())
        else:
            items = tuple((quote.root, quote) for quote in quotes)
        updated_marks: dict[InstrumentId, float] = dict(self._marks)
        for key, quote in items:
            root = key.strip().upper()
            if root != quote.root:
                raise ValueError(f"quote key {key!r} does not match quote root {quote.root!r}")
            updated_marks[root] = quote.close
            self._quotes[root] = quote
        self._marks = updated_marks
        self._revalue_positions()

    def update_quote(self, quote: BarQuote) -> None:
        """Install/update a single bar quote."""
        self.set_market(self._timestamp, {quote.root: quote})

    def reset_monthly_volume(self) -> None:
        """Reset the volume-tiered commission counter (month rollover)."""
        self._monthly_volume.clear()

    def _revalue_positions(self) -> None:
        for root, position in tuple(self._positions.items()):
            mark = self._marks.get(root)
            if mark is not None and mark != position.market_price:
                self._positions[root] = replace(position, market_price=mark)

    def _quote(self, root: str) -> BarQuote | None:
        return self._quotes.get(root.strip().upper())

    def _next_order_id(self, prefix: str) -> str:
        self._order_counter += 1
        return f"{prefix}-{self._order_counter}"

    # -- connection lifecycle (protocol) --------------------------------------

    async def connect(self) -> None:
        """Mark the simulated session as connected."""
        self._connected = True

    async def disconnect(self) -> None:
        """Mark the simulated session as disconnected and cancel working orders."""
        self._connected = False
        self._working.clear()

    # -- order matching -------------------------------------------------------

    def place_order(
        self,
        order: OrderEvent,
        *,
        kind: TradeKind = TradeKind.REBALANCE,
        extra_slippage_ticks: float = 0.0,
        extra_fee_per_contract: float = 0.0,
    ) -> OrderReceipt:
        """Match one order against the current bar and return a receipt."""
        if not math.isfinite(extra_slippage_ticks) or extra_slippage_ticks < 0.0:
            raise ValueError("extra_slippage_ticks must be finite and non-negative")
        if not math.isfinite(extra_fee_per_contract) or extra_fee_per_contract < 0.0:
            raise ValueError("extra_fee_per_contract must be finite and non-negative")
        working = self._working.get(order.order_id)
        if working is not None:
            return self._try_fill(working)
        attempt = self._match(order)
        if attempt is None:
            return self._reject(
                order, "order does not match the current bar (limit/stop not triggered)"
            )
        fill_qty = self._fill_quantity(order.instrument_id, order.quantity)
        if fill_qty <= 0:
            self._working[order.order_id] = _WorkingOrder(
                order=order,
                remaining=order.quantity,
                kind=kind,
                extra_slippage_ticks=extra_slippage_ticks,
                extra_fee_per_contract=extra_fee_per_contract,
            )
            return self._pending_receipt(order, order.quantity)
        return self._execute(
            order=order,
            working=_WorkingOrder(
                order=order,
                remaining=order.quantity,
                kind=kind,
                extra_slippage_ticks=extra_slippage_ticks,
                extra_fee_per_contract=extra_fee_per_contract,
            ),
            attempt=attempt,
            fill_qty=fill_qty,
        )

    def _try_fill(self, working: _WorkingOrder) -> OrderReceipt:
        """Attempt to fill the remainder of a working order on the current bar."""
        attempt = self._match(working.order)
        if attempt is None:
            return self._pending_receipt(working.order, working.remaining)
        fill_qty = self._fill_quantity(working.order.instrument_id, working.remaining)
        if fill_qty <= 0:
            return self._pending_receipt(working.order, working.remaining)
        return self._execute(
            order=working.order,
            working=working,
            attempt=attempt,
            fill_qty=fill_qty,
        )

    def _execute(
        self,
        *,
        order: OrderEvent,
        working: _WorkingOrder,
        attempt: tuple[str, float],
        fill_qty: int,
    ) -> OrderReceipt:
        match_kind, reference_price = attempt
        is_limit = match_kind == "limit"
        cost = self._transaction_cost(
            order,
            fill_qty,
            reference_price,
            is_limit=is_limit,
            extra_slippage_ticks=working.extra_slippage_ticks,
            extra_fee_per_contract=working.extra_fee_per_contract,
        )
        fill_price = self._execution_price(order.side, reference_price, cost, is_limit=is_limit)
        self._apply_fill(
            order=order,
            fill_quantity=fill_qty,
            fill_price=fill_price,
            reference_price=reference_price,
            cost=cost,
            kind=working.kind,
        )
        working.remaining -= fill_qty
        if working.remaining <= 0:
            self._working.pop(order.order_id, None)
            return OrderReceipt(
                order_id=order.order_id,
                status=OrderStatus.FILLED,
                filled_quantity=fill_qty,
                average_price=fill_price,
                commission=cost.commission + cost.fees,
                message="filled",
                timestamp=self._timestamp,
            )
        self._working[order.order_id] = working
        return OrderReceipt(
            order_id=order.order_id,
            status=OrderStatus.PARTIALLY_FILLED,
            filled_quantity=fill_qty,
            average_price=fill_price,
            commission=cost.commission + cost.fees,
            message=f"partially filled, {working.remaining} remaining",
            timestamp=self._timestamp,
        )

    def process_working_orders(self) -> list[OrderReceipt]:
        """Try to fill every working order against the latest quotes."""
        receipts: list[OrderReceipt] = []
        for order_id in list(self._working):
            working = self._working.get(order_id)
            if working is None:
                continue
            receipts.append(self._try_fill(working))
        return receipts

    def _match(self, order: OrderEvent) -> tuple[str, float] | None:
        """Return ``(match_kind, reference_price)`` when the current bar qualifies."""
        quote = self._quote(order.instrument_id)
        if quote is None:
            mark = self._marks.get(order.instrument_id)
            if mark is None:
                return None
            if order.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT):
                limit = order.limit_price
                if limit is None:
                    return None
                return ("limit", self._limit_reference(order.side, mark, limit, quote=None))
            return ("market", mark)
        if order.order_type in (
            OrderType.MARKET,
            OrderType.ADAPTIVE,
            OrderType.TWAP,
            OrderType.VWAP,
        ):
            return ("market", self._market_reference(quote))
        if order.order_type is OrderType.LIMIT:
            limit = order.limit_price
            if limit is None:
                return None
            if order.side is OrderSide.BUY:
                if quote.low <= limit:
                    return (
                        "limit",
                        self._limit_reference(order.side, quote.open, limit, quote=quote),
                    )
                return None
            if quote.high >= limit:
                return ("limit", self._limit_reference(order.side, quote.open, limit, quote=quote))
            return None
        if order.order_type is OrderType.STOP:
            stop = order.stop_price
            if stop is None:
                return None
            if order.side is OrderSide.BUY:
                if quote.high >= stop:
                    return ("market", max(stop, quote.open))
                return None
            if quote.low <= stop:
                return ("market", min(stop, quote.open))
            return None
        if order.order_type is OrderType.STOP_LIMIT:
            stop = order.stop_price
            limit = order.limit_price
            if stop is None or limit is None:
                return None
            if order.side is OrderSide.BUY:
                if quote.high >= stop:
                    return (
                        "limit",
                        self._limit_reference(
                            order.side, max(stop, quote.open), limit, quote=quote
                        ),
                    )
                return None
            if quote.low <= stop:
                return (
                    "limit",
                    self._limit_reference(order.side, min(stop, quote.open), limit, quote=quote),
                )
            return None
        return None

    def _market_reference(self, quote: BarQuote) -> float:
        if self._fill_price_mode is FillPriceMode.OPEN:
            return quote.open
        if self._fill_price_mode is FillPriceMode.MID:
            return quote.mid
        return quote.close

    @staticmethod
    def _limit_reference(
        side: OrderSide, candidate: float, limit: float, *, quote: BarQuote | None
    ) -> float:
        reference = min(candidate, limit) if side is OrderSide.BUY else max(candidate, limit)
        if quote is not None:
            reference = min(max(reference, quote.low), quote.high)
        return reference

    @staticmethod
    def _execution_price(
        side: OrderSide,
        reference_price: float,
        cost: CostBreakdown,
        *,
        is_limit: bool,
    ) -> float:
        if is_limit:
            return reference_price
        return reference_price + side.multiplier * cost.execution_cost_ticks * cost.tick_size

    def _fill_quantity(self, root: str, remaining: int) -> int:
        if self._max_participation is None:
            return remaining
        quote = self._quote(root)
        volume = quote.volume if quote is not None else 0.0
        capacity: int = math.floor(volume * self._max_participation)
        if capacity <= 0:
            return 0
        return min(remaining, capacity)

    def _instrument_spec(self, root: str) -> tuple[float, float, float, float]:
        """Return ``(multiplier, tick_size, initial_margin, maintenance_margin)``."""
        if self._instrument_master is not None and self._instrument_master.has(root):
            spec = self._instrument_master.get(root)
            return (
                float(spec.multiplier),
                float(spec.tick_size),
                float(spec.margins.initial),
                float(spec.margins.maintenance),
            )
        position = self._positions.get(root)
        if position is not None:
            return (
                position.multiplier,
                0.01,
                position.initial_margin,
                position.maintenance_margin,
            )
        raise KeyError(f"cannot resolve instrument specification for {root!r}")

    def _transaction_cost(
        self,
        order: OrderEvent,
        quantity: int,
        reference_price: float,
        *,
        is_limit: bool,
        extra_slippage_ticks: float = 0.0,
        extra_fee_per_contract: float = 0.0,
    ) -> CostBreakdown:
        root = order.instrument_id.strip().upper()
        multiplier, tick_size, _, _ = self._instrument_spec(root)
        quote = self._quote(root)
        volume = quote.volume if quote is not None else 0.0
        spread = quote.spread_price if quote is not None else None
        quoted_spread_ticks = spread / tick_size if spread is not None else None
        return self._cost_model.transaction_cost(
            root,
            quantity,
            reference_price,
            tick_size,
            multiplier,
            volume,
            monthly_volume=self._monthly_volume.get(root, 0),
            quoted_spread_ticks=quoted_spread_ticks,
            limit_order=is_limit,
            extra_slippage_ticks=extra_slippage_ticks,
            extra_fee_per_contract=extra_fee_per_contract,
        )

    def _reject(self, order: OrderEvent, message: str) -> OrderReceipt:
        return OrderReceipt(
            order_id=order.order_id,
            status=OrderStatus.REJECTED,
            message=message,
            timestamp=self._timestamp,
        )

    def _pending_receipt(self, order: OrderEvent, remaining: int) -> OrderReceipt:
        return OrderReceipt(
            order_id=order.order_id,
            status=OrderStatus.SUBMITTED,
            message=f"working, {remaining} remaining",
            timestamp=self._timestamp,
        )

    # -- position / cash bookkeeping -----------------------------------------

    def _apply_fill(
        self,
        *,
        order: OrderEvent,
        fill_quantity: int,
        fill_price: float,
        reference_price: float,
        cost: CostBreakdown,
        kind: TradeKind,
    ) -> None:
        root = order.instrument_id.strip().upper()
        signed_delta = order.side.multiplier * fill_quantity
        existing = self._positions.get(root)
        if existing is None:
            multiplier, _, initial_margin, maintenance_margin = self._instrument_spec(root)
            self._positions[root] = Position(
                instrument_id=root,
                quantity=signed_delta,
                average_price=fill_price,
                market_price=self._marks.get(root, reference_price),
                multiplier=multiplier,
                expiration=order.expiration,
                initial_margin=initial_margin,
                maintenance_margin=maintenance_margin,
                contract_symbol=order.contract_symbol,
            )
        else:
            old_quantity = existing.quantity
            new_quantity = old_quantity + signed_delta
            realized = 0.0
            if old_quantity != 0 and signed_delta * old_quantity < 0:
                closed = min(abs(old_quantity), abs(signed_delta))
                direction = 1 if old_quantity > 0 else -1
                realized = (
                    (fill_price - existing.average_price) * direction * closed * existing.multiplier
                )
            if new_quantity == 0 or old_quantity == 0 or (new_quantity > 0) != (old_quantity > 0):
                average_price = fill_price
            elif abs(new_quantity) > abs(old_quantity):
                added = abs(signed_delta)
                total = abs(old_quantity) + added
                average_price = (
                    existing.average_price * abs(old_quantity) + fill_price * added
                ) / total
            else:
                average_price = existing.average_price
            if realized != 0.0:
                self._cash += realized
            self._positions[root] = replace(
                existing,
                quantity=new_quantity,
                average_price=average_price,
                market_price=self._marks.get(root, existing.market_price),
                expiration=order.expiration or existing.expiration,
                contract_symbol=order.contract_symbol or existing.contract_symbol,
            )
        self._cash -= cost.commission + cost.fees
        self._monthly_volume[root] = self._monthly_volume.get(root, 0) + fill_quantity
        self._fills.append(
            FillRecord(
                order_id=order.order_id,
                instrument_id=root,
                side=order.side,
                quantity=fill_quantity,
                price=fill_price,
                reference_price=reference_price,
                timestamp=self._timestamp,
                cost=cost,
                kind=kind,
                message=order.tag,
            )
        )
        self._revalue_positions()

    def apply_fill(
        self,
        instrument_id: InstrumentId,
        side: OrderSide,
        quantity: int,
        price: float,
        *,
        commission: float = 0.0,
        fees: float = 0.0,
        kind: TradeKind = TradeKind.REBALANCE,
        order_id: str | None = None,
        expiration: date | None = None,
    ) -> OrderReceipt:
        """Book an externally matched fill (used by tests and roll simulation)."""
        root = instrument_id.strip().upper()
        multiplier, tick_size, _, _ = self._instrument_spec(root)
        cost = CostBreakdown(
            root=root,
            quantity=quantity,
            price=price,
            tick_size=tick_size,
            multiplier=multiplier,
            commission=commission,
            fees=fees,
            commission_per_contract=commission / quantity if quantity else 0.0,
            fee_per_contract=fees / quantity if quantity else 0.0,
        )
        order = OrderEvent.market(
            order_id=order_id or self._next_order_id("external"),
            timestamp=self._timestamp,
            instrument_id=root,
            side=side,
            quantity=quantity,
            expiration=expiration,
        )
        self._apply_fill(
            order=order,
            fill_quantity=quantity,
            fill_price=price,
            reference_price=price,
            cost=cost,
            kind=kind,
        )
        return OrderReceipt(
            order_id=order.order_id,
            status=OrderStatus.FILLED,
            filled_quantity=quantity,
            average_price=price,
            commission=commission + fees,
            message="external fill",
            timestamp=self._timestamp,
        )

    # -- settlement / margin ---------------------------------------------------

    def settle(
        self, settlement_prices: Mapping[InstrumentId, float] | None = None
    ) -> SettlementResult:
        """Run daily mark-to-market settlement at the provided/official prices."""
        prices: dict[InstrumentId, float] = {}
        for root in sorted(self._positions):
            price: float | None = None
            if settlement_prices is not None:
                price = settlement_prices.get(root)
            if price is None:
                quote = self._quote(root)
                if quote is not None and quote.settlement is not None:
                    price = quote.settlement
            if price is None:
                price = self._marks.get(root)
            if price is None:
                raise ValueError(f"no settlement price available for {root!r}")
            prices[root] = float(price)
        result = self._margin_simulator.settle(
            self._timestamp,
            cash=self._cash,
            positions=self.get_positions_sync(),
            settlement_prices=prices,
        )
        self._cash = result.cash_after
        for position in result.positions:
            self._marks[position.instrument_id] = position.market_price
            self._positions[position.instrument_id] = position
        return result

    def get_account_sync(self, timestamp: datetime | None = None) -> AccountSnapshot:
        """Synchronous account snapshot (raises when the account is depleted)."""
        return self._margin_simulator.account_snapshot(
            timestamp or self._timestamp,
            cash=self._cash,
            positions=self.get_positions_sync(),
        )

    def get_positions_sync(self) -> Mapping[InstrumentId, Position]:
        """Current positions keyed by futures root."""
        return MappingProxyType(dict(self._positions))

    def margin_requirements(self) -> tuple[float, float]:
        """``(initial_margin, maintenance_margin)`` required by the position book."""
        requirements = self._margin_simulator.margin_requirements(self.get_positions_sync())
        return requirements.initial_margin, requirements.maintenance_margin

    # -- cancellation / flattening --------------------------------------------

    def cancel_order_sync(self, order_id: str) -> OrderReceipt:
        """Cancel one working order (or report ``UNKNOWN`` when not outstanding)."""
        working = self._working.pop(order_id, None)
        if working is None:
            return OrderReceipt(
                order_id=order_id,
                status=OrderStatus.UNKNOWN,
                message="order is not working",
                timestamp=self._timestamp,
            )
        return OrderReceipt(
            order_id=order_id,
            status=OrderStatus.CANCELLED,
            message="cancelled",
            timestamp=self._timestamp,
        )

    def cancel_all_sync(self) -> list[OrderReceipt]:
        """Cancel every working order."""
        return [self.cancel_order_sync(order_id) for order_id in list(self._working)]

    def flatten_all_sync(
        self,
        *,
        kind: TradeKind = TradeKind.FLATTEN,
        extra_slippage_ticks: float = 0.0,
        extra_fee_per_contract: float = 0.0,
        tag: str = "",
    ) -> list[OrderReceipt]:
        """Flatten every open position with market orders (emergency kill switch)."""
        self.cancel_all_sync()
        receipts: list[OrderReceipt] = []
        for root in sorted(self._positions):
            position = self._positions[root]
            if position.quantity == 0:
                continue
            side = OrderSide.SELL if position.quantity > 0 else OrderSide.BUY
            order = OrderEvent.market(
                order_id=self._next_order_id("flatten"),
                timestamp=self._timestamp,
                instrument_id=root,
                side=side,
                quantity=abs(position.quantity),
                expiration=position.expiration,
                contract_symbol=position.contract_symbol,
                tag=tag,
            )
            receipts.append(
                self.place_order(
                    order,
                    kind=kind,
                    extra_slippage_ticks=extra_slippage_ticks,
                    extra_fee_per_contract=extra_fee_per_contract,
                )
            )
        return receipts

    def roll_position(
        self,
        instrument_id: InstrumentId,
        *,
        from_price: float,
        to_price: float,
        to_expiration: date | None = None,
        to_contract_symbol: str | None = None,
        use_bag: bool = False,
        order_prefix: str = "roll",
    ) -> tuple[OrderReceipt, OrderReceipt]:
        """Close a position in the front contract and reopen it in the next one.

        Both legs are executed through the cost model (two-legged costs); with
        ``use_bag`` the model applies the native calendar-spread combo discount.
        Returns ``(close_receipt, open_receipt)``.
        """
        root = instrument_id.strip().upper()
        position = self._positions.get(root)
        if position is None or position.quantity == 0:
            raise ValueError(f"cannot roll {root!r}: no open position")
        if not math.isfinite(from_price) or from_price <= 0.0:
            raise ValueError("from_price must be finite and strictly positive")
        if not math.isfinite(to_price) or to_price <= 0.0:
            raise ValueError("to_price must be finite and strictly positive")
        multiplier, tick_size, _, _ = self._instrument_spec(root)
        quote = self._quote(root)
        volume = quote.volume if quote is not None else 0.0
        roll_cost = self._cost_model.roll_cost(
            root,
            abs(position.quantity),
            from_price,
            to_price,
            tick_size,
            multiplier,
            volume,
            monthly_volume=self._monthly_volume.get(root, 0),
            use_bag=use_bag,
        )
        closing_side = OrderSide.SELL if position.quantity > 0 else OrderSide.BUY
        close_fill_price = (
            from_price
            + closing_side.multiplier * roll_cost.close_leg.execution_cost_ticks * tick_size
        )
        close_record = FillRecord(
            order_id=self._next_order_id(f"{order_prefix}-close"),
            instrument_id=root,
            side=closing_side,
            quantity=abs(position.quantity),
            price=max(close_fill_price, tick_size),
            reference_price=from_price,
            timestamp=self._timestamp,
            cost=roll_cost.close_leg,
            kind=TradeKind.ROLL_CLOSE,
            message="roll close leg",
        )
        self._apply_recorded_fill(close_record)
        reopened = self._positions.get(root)
        if reopened is not None and reopened.quantity != 0:
            raise AssertionError("roll close leg did not flatten the position")
        opening_side = OrderSide.BUY if position.quantity > 0 else OrderSide.SELL
        open_fill_price = (
            to_price + opening_side.multiplier * roll_cost.open_leg.execution_cost_ticks * tick_size
        )
        open_record = FillRecord(
            order_id=self._next_order_id(f"{order_prefix}-open"),
            instrument_id=root,
            side=opening_side,
            quantity=abs(position.quantity),
            price=max(open_fill_price, tick_size),
            reference_price=to_price,
            timestamp=self._timestamp,
            cost=roll_cost.open_leg,
            kind=TradeKind.ROLL_OPEN,
            message="roll open leg",
        )
        self._apply_recorded_fill(open_record)
        current = self._positions.get(root)
        if current is not None:
            self._positions[root] = replace(
                current,
                expiration=to_expiration or current.expiration,
                contract_symbol=to_contract_symbol or current.contract_symbol,
            )
        self._marks[root] = to_price
        self._revalue_positions()
        close_receipt = OrderReceipt(
            order_id=close_record.order_id,
            status=OrderStatus.FILLED,
            filled_quantity=close_record.quantity,
            average_price=close_record.price,
            commission=close_record.commissions,
            message="roll close leg",
            timestamp=self._timestamp,
        )
        open_receipt = OrderReceipt(
            order_id=open_record.order_id,
            status=OrderStatus.FILLED,
            filled_quantity=open_record.quantity,
            average_price=open_record.price,
            commission=open_record.commissions,
            message="roll open leg",
            timestamp=self._timestamp,
        )
        return close_receipt, open_receipt

    def _apply_recorded_fill(self, record: FillRecord) -> None:
        """Book a pre-costed fill (roll legs) without re-running the cost model."""
        root = record.instrument_id
        signed_delta = record.signed_quantity
        current = self._positions.get(root)
        multiplier, _, initial_margin, maintenance_margin = self._instrument_spec(root)
        realized = 0.0
        if current is None:
            self._positions[root] = Position(
                instrument_id=root,
                quantity=signed_delta,
                average_price=record.price,
                market_price=self._marks.get(root, record.price),
                multiplier=multiplier,
                expiration=None,
                initial_margin=initial_margin,
                maintenance_margin=maintenance_margin,
            )
        else:
            old_quantity = current.quantity
            new_quantity = old_quantity + signed_delta
            if old_quantity != 0 and signed_delta * old_quantity < 0:
                closed = min(abs(old_quantity), abs(signed_delta))
                direction = 1 if old_quantity > 0 else -1
                realized = (record.price - current.average_price) * direction * closed * multiplier
            if new_quantity == 0 or old_quantity == 0 or (new_quantity > 0) != (old_quantity > 0):
                average_price = record.price
            elif abs(new_quantity) > abs(old_quantity):
                added = abs(signed_delta)
                total = abs(old_quantity) + added
                average_price = (
                    current.average_price * abs(old_quantity) + record.price * added
                ) / total
            else:
                average_price = current.average_price
            self._positions[root] = replace(
                current,
                quantity=new_quantity,
                average_price=average_price,
                market_price=self._marks.get(root, current.market_price),
            )
        if realized != 0.0:
            self._cash += realized
        self._cash -= record.commissions
        self._monthly_volume[root] = self._monthly_volume.get(root, 0) + record.quantity
        self._fills.append(record)
        self._revalue_positions()

    # -- protocol async API ---------------------------------------------------

    async def get_account(self) -> AccountSnapshot:
        """Async account snapshot (protocol)."""
        return self.get_account_sync()

    async def get_positions(self) -> Mapping[InstrumentId, Position]:
        """Async positions snapshot (protocol)."""
        return self.get_positions_sync()

    async def submit_order(self, order: OrderEvent) -> OrderReceipt:
        """Submit one order through the simulated matcher (protocol)."""
        return self.place_order(order)

    async def submit_orders(self, orders: Sequence[OrderEvent]) -> list[OrderReceipt]:
        """Submit a batch of orders in sequence (protocol)."""
        return [self.place_order(order) for order in orders]

    async def cancel_order(self, order_id: str) -> OrderReceipt:
        """Cancel one outstanding order (protocol)."""
        return self.cancel_order_sync(order_id)

    async def cancel_all(self) -> list[OrderReceipt]:
        """Cancel every outstanding order (protocol)."""
        return self.cancel_all_sync()

    async def flatten_all(self) -> list[OrderReceipt]:
        """Emergency kill-switch: flatten every position and cancel all orders."""
        return self.flatten_all_sync()
