"""Tick-aware, deterministic futures transaction-cost model.

The cost model is the single source of truth for *how expensive* it is to trade
a futures contract in the backtester. It decomposes every transaction into the
five components the tear sheet attributes separately:

``spread``
    Crossing the bid/ask. The spread is tick-aware: a quote is never assumed to
    be tighter than ``min_spread_ticks`` and the minimum realistic spread is one
    tick, so a round trip always costs at least one tick of price.
``slippage``
    Market-impact of the order relative to the bar volume, modelled either as a
    square-root (default) or a linear participation function. Deterministic: the
    same order always produces exactly the same impact.
``commission``
    Tiered IBKR per-contract commission (lower rates at higher monthly volume)
    plus an optional basis-point commission used by declarative configs.
``fees``
    CME / CBOT / NYMEX / COMEX exchange fees, clearing fees and the NFA
    regulatory assessment, all charged per contract per side.
``funding``
    Futures do not pay funding (it is embedded in the roll basis); the field
    exists so the accounting identity and the tear sheet can represent it.

Every public method returns an immutable typed breakdown. Nothing in this module
is random or depends on wall-clock time, which keeps backtests reproducible.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from configs.instruments import InstrumentsConfig

from src.core.types import AssetClass

# ---------------------------------------------------------------------------
# Fee schedules
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FeeSchedule:
    """Per-contract commission and regulatory-fee schedule for one venue.

    ``commission_tiers`` is a non-empty ascending sequence of
    ``(monthly_volume_threshold, per_contract_rate)`` pairs. The applicable
    rate is the last tier whose threshold is ``<=`` the (cumulative) monthly
    contract volume, mirroring IBKR's volume-tiered futures commission ladder.
    """

    commission_tiers: tuple[tuple[int, float], ...] = (
        (0, 0.85),
        (1_000, 0.65),
        (10_000, 0.45),
    )
    exchange_fee: float = 1.38
    clearing_fee: float = 0.30
    nfa_fee: float = 0.02
    minimum_commission: float = 0.0

    def __post_init__(self) -> None:
        if not self.commission_tiers:
            raise ValueError("commission_tiers must not be empty")
        previous = -1
        for threshold, rate in self.commission_tiers:
            if threshold <= previous:
                raise ValueError("commission_tiers thresholds must be strictly ascending")
            if threshold < 0:
                raise ValueError("commission tier thresholds must be non-negative")
            if not math.isfinite(rate) or rate < 0.0:
                raise ValueError("commission rates must be finite and non-negative")
            previous = threshold
        if self.commission_tiers[0][0] != 0:
            raise ValueError("commission_tiers must start at a 0-contract threshold")
        for name in ("exchange_fee", "clearing_fee", "nfa_fee", "minimum_commission"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

    @property
    def fee_per_contract(self) -> float:
        """Exchange + clearing + NFA fee charged per contract per side."""
        return self.exchange_fee + self.clearing_fee + self.nfa_fee

    def commission_rate(self, monthly_volume: int = 0) -> float:
        """Commission rate per contract for ``monthly_volume`` traded contracts."""
        if isinstance(monthly_volume, bool) or not isinstance(monthly_volume, int):
            raise TypeError("monthly_volume must be an int")
        if monthly_volume < 0:
            raise ValueError("monthly_volume must be non-negative")
        rate = self.commission_tiers[0][1]
        for threshold, tier_rate in self.commission_tiers:
            if monthly_volume >= threshold:
                rate = tier_rate
        return max(rate, self.minimum_commission)

    def commission_for(self, quantity: int, monthly_volume: int = 0) -> float:
        """Commission charged for ``quantity`` contracts."""
        if quantity < 0:
            raise ValueError("quantity must be non-negative")
        return self.commission_rate(monthly_volume) * quantity

    def fees_for(self, quantity: int) -> float:
        """Exchange + clearing + NFA fees charged for ``quantity`` contracts."""
        if quantity < 0:
            raise ValueError("quantity must be non-negative")
        return self.fee_per_contract * quantity

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable representation (used in run metadata)."""
        return {
            "commission_tiers": [[threshold, rate] for threshold, rate in self.commission_tiers],
            "exchange_fee": self.exchange_fee,
            "clearing_fee": self.clearing_fee,
            "nfa_fee": self.nfa_fee,
            "fee_per_contract": self.fee_per_contract,
        }


#: Exchange-specific default fee schedules (CME group venues).
EXCHANGE_FEE_SCHEDULES: Mapping[str, FeeSchedule] = MappingProxyType(
    {
        "CME": FeeSchedule(exchange_fee=1.38, clearing_fee=0.30, nfa_fee=0.02),
        "CBOT": FeeSchedule(exchange_fee=1.18, clearing_fee=0.28, nfa_fee=0.02),
        "NYMEX": FeeSchedule(exchange_fee=1.50, clearing_fee=0.32, nfa_fee=0.02),
        "COMEX": FeeSchedule(exchange_fee=1.42, clearing_fee=0.30, nfa_fee=0.02),
    }
)

#: Instrument-class default fee schedules (used when the exchange is unknown).
ASSET_CLASS_FEE_SCHEDULES: Mapping[AssetClass, FeeSchedule] = MappingProxyType(
    {
        AssetClass.EQUITY_INDEX: FeeSchedule(exchange_fee=1.38, clearing_fee=0.30, nfa_fee=0.02),
        AssetClass.RATES: FeeSchedule(exchange_fee=1.18, clearing_fee=0.28, nfa_fee=0.02),
        AssetClass.ENERGY: FeeSchedule(exchange_fee=1.50, clearing_fee=0.32, nfa_fee=0.02),
        AssetClass.METALS: FeeSchedule(exchange_fee=1.42, clearing_fee=0.30, nfa_fee=0.02),
        AssetClass.FX: FeeSchedule(exchange_fee=1.20, clearing_fee=0.28, nfa_fee=0.02),
        AssetClass.AGRICULTURE: FeeSchedule(exchange_fee=1.25, clearing_fee=0.28, nfa_fee=0.02),
        AssetClass.CRYPTO: FeeSchedule(exchange_fee=2.00, clearing_fee=0.40, nfa_fee=0.02),
        AssetClass.UNKNOWN: FeeSchedule(),
    }
)

#: Generic fallback schedule.
DEFAULT_FEE_SCHEDULE: FeeSchedule = FeeSchedule()


def default_fee_schedule(asset_class: AssetClass) -> FeeSchedule:
    """Return the default fee schedule for one instrument class."""
    return ASSET_CLASS_FEE_SCHEDULES.get(asset_class, DEFAULT_FEE_SCHEDULE)


# ---------------------------------------------------------------------------
# Slippage model
# ---------------------------------------------------------------------------


class SlippageModel(StrEnum):
    """Functional form of the volume-participation market-impact model."""

    SQUARE_ROOT = "sqrt"
    LINEAR = "linear"


# ---------------------------------------------------------------------------
# Typed cost breakdowns
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    """Immutable decomposition of one transaction's execution cost.

    Monetary fields are in account currency. ``*_ticks`` fields are *per
    contract* price ticks so they can be compared across instruments.
    """

    root: str
    quantity: int
    price: float
    tick_size: float
    multiplier: float
    spread_cost: float = 0.0
    slippage_cost: float = 0.0
    commission: float = 0.0
    fees: float = 0.0
    funding: float = 0.0
    spread_ticks: float = 0.0
    slippage_ticks: float = 0.0
    commission_per_contract: float = 0.0
    fee_per_contract: float = 0.0

    def __post_init__(self) -> None:
        if not self.root:
            raise ValueError("root must be a non-empty string")
        if isinstance(self.quantity, bool) or not isinstance(self.quantity, int):
            raise TypeError("quantity must be an int")
        if self.quantity < 0:
            raise ValueError("quantity must be non-negative")
        for name in (
            "price",
            "tick_size",
            "multiplier",
            "spread_cost",
            "slippage_cost",
            "commission",
            "fees",
            "funding",
            "spread_ticks",
            "slippage_ticks",
            "commission_per_contract",
            "fee_per_contract",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.tick_size <= 0.0 or self.multiplier <= 0.0 or self.price <= 0.0:
            raise ValueError("price, tick_size and multiplier must be strictly positive")

    @property
    def total(self) -> float:
        """Total transaction cost in account currency."""
        return self.spread_cost + self.slippage_cost + self.commission + self.fees + self.funding

    @property
    def execution_cost(self) -> float:
        """Price-impact cost only (spread + slippage), excluding commission/fees."""
        return self.spread_cost + self.slippage_cost

    @property
    def execution_cost_ticks(self) -> float:
        """Execution price impact in per-contract ticks (spread + slippage)."""
        return self.spread_ticks + self.slippage_ticks

    @property
    def notional(self) -> float:
        """Traded notional ``quantity * price * multiplier``."""
        return self.quantity * self.price * self.multiplier

    @property
    def total_ticks_per_contract(self) -> float:
        """Total cost expressed in per-contract ticks (fees converted to ticks)."""
        tick_value = self.tick_size * self.multiplier
        if tick_value <= 0.0 or self.quantity == 0:
            return 0.0
        fee_ticks = (self.commission + self.fees) / (tick_value * self.quantity)
        return self.spread_ticks + self.slippage_ticks + fee_ticks

    def scaled(self, factor: float) -> CostBreakdown:
        """Return the breakdown scaled by ``factor`` (leg aggregation helper)."""
        if not math.isfinite(factor) or factor < 0.0:
            raise ValueError("factor must be finite and non-negative")
        return CostBreakdown(
            root=self.root,
            quantity=self.quantity,
            price=self.price,
            tick_size=self.tick_size,
            multiplier=self.multiplier,
            spread_cost=self.spread_cost * factor,
            slippage_cost=self.slippage_cost * factor,
            commission=self.commission * factor,
            fees=self.fees * factor,
            funding=self.funding * factor,
            spread_ticks=self.spread_ticks,
            slippage_ticks=self.slippage_ticks,
            commission_per_contract=self.commission_per_contract * factor,
            fee_per_contract=self.fee_per_contract * factor,
        )

    def combined(self, other: CostBreakdown) -> CostBreakdown:
        """Add two breakdowns for the same root into one (round-trip helper)."""
        if other.root != self.root:
            raise ValueError("cannot combine cost breakdowns for different roots")
        quantity = self.quantity + other.quantity
        notional = self.notional + other.notional
        price = notional / (quantity * self.multiplier) if quantity else self.price
        return CostBreakdown(
            root=self.root,
            quantity=quantity,
            price=price,
            tick_size=self.tick_size,
            multiplier=self.multiplier,
            spread_cost=self.spread_cost + other.spread_cost,
            slippage_cost=self.slippage_cost + other.slippage_cost,
            commission=self.commission + other.commission,
            fees=self.fees + other.fees,
            funding=self.funding + other.funding,
            spread_ticks=self.spread_ticks + other.spread_ticks,
            slippage_ticks=self.slippage_ticks + other.slippage_ticks,
            commission_per_contract=self.commission_per_contract + other.commission_per_contract,
            fee_per_contract=self.fee_per_contract + other.fee_per_contract,
        )

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable representation."""
        return {
            "root": self.root,
            "quantity": self.quantity,
            "price": self.price,
            "tick_size": self.tick_size,
            "multiplier": self.multiplier,
            "spread_cost": self.spread_cost,
            "slippage_cost": self.slippage_cost,
            "commission": self.commission,
            "fees": self.fees,
            "funding": self.funding,
            "total": self.total,
            "spread_ticks": self.spread_ticks,
            "slippage_ticks": self.slippage_ticks,
            "total_ticks_per_contract": self.total_ticks_per_contract,
        }

    @classmethod
    def zero(cls, root: str, *, price: float, tick_size: float, multiplier: float) -> CostBreakdown:
        """A zero-cost breakdown (no order / no position change)."""
        return cls(
            root=root,
            quantity=0,
            price=price,
            tick_size=tick_size,
            multiplier=multiplier,
        )


@dataclass(frozen=True, slots=True)
class RollCostBreakdown:
    """Two-legged roll cost: closing the old contract and opening the new one.

    A native IBKR ``BAG`` calendar-spread order trades both legs as one combo:
    the exchange charges per leg but IBKR applies a combo commission discount and
    the execution price is a single calendar-spread quote (modelled as a reduced
    spread). ``bag_discount`` reports the saving versus two independent orders.
    """

    root: str
    quantity: int
    from_price: float
    to_price: float
    tick_size: float
    multiplier: float
    close_leg: CostBreakdown
    open_leg: CostBreakdown
    bag_discount: float = 0.0

    def __post_init__(self) -> None:
        if self.close_leg.root != self.root or self.open_leg.root != self.root:
            raise ValueError("roll leg root mismatch")
        if self.bag_discount < 0.0 or not math.isfinite(self.bag_discount):
            raise ValueError("bag_discount must be finite and non-negative")

    @property
    def spread_cost(self) -> float:
        """Total spread cost across both legs."""
        return self.close_leg.spread_cost + self.open_leg.spread_cost

    @property
    def slippage_cost(self) -> float:
        """Total market-impact cost across both legs."""
        return self.close_leg.slippage_cost + self.open_leg.slippage_cost

    @property
    def commission(self) -> float:
        """Total commission across both legs (after any BAG discount)."""
        return self.close_leg.commission + self.open_leg.commission

    @property
    def fees(self) -> float:
        """Total exchange/clearing/NFA fees across both legs."""
        return self.close_leg.fees + self.open_leg.fees

    @property
    def funding(self) -> float:
        """Funding component (always zero for futures rolls)."""
        return 0.0

    @property
    def total(self) -> float:
        """Total roll execution cost in account currency."""
        return self.spread_cost + self.slippage_cost + self.commission + self.fees + self.funding

    @property
    def total_before_bag_discount(self) -> float:
        """Roll cost if the legs were executed as two independent orders."""
        return self.total + self.bag_discount

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable representation."""
        return {
            "root": self.root,
            "quantity": self.quantity,
            "from_price": self.from_price,
            "to_price": self.to_price,
            "spread_cost": self.spread_cost,
            "slippage_cost": self.slippage_cost,
            "commission": self.commission,
            "fees": self.fees,
            "bag_discount": self.bag_discount,
            "total": self.total,
            "close_leg": self.close_leg.to_dict(),
            "open_leg": self.open_leg.to_dict(),
        }


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------


class CostModel:
    """Deterministic, injectable futures transaction-cost model.

    The defaults are conservative IBKR/CME Group retail estimates; every knob
    can be overridden by the caller or by a declarative strategy config.
    """

    def __init__(
        self,
        *,
        instrument_master: InstrumentsConfig | None = None,
        fee_schedules: Mapping[str, FeeSchedule] | None = None,
        slippage_model: SlippageModel = SlippageModel.SQUARE_ROOT,
        impact_coefficient: float = 8.0,
        max_impact_ticks: float = 5.0,
        min_spread_ticks: float = 1.0,
        base_slippage_ticks: float = 0.0,
        commission_bps: float = 0.0,
        bag_commission_discount: float = 0.20,
        bag_spread_factor: float = 0.5,
        penalty_fee_per_contract: float = 15.0,
        monthly_volume: int = 0,
    ) -> None:
        if not isinstance(slippage_model, SlippageModel):
            raise TypeError("slippage_model must be a SlippageModel")
        for name, value in (
            ("impact_coefficient", impact_coefficient),
            ("max_impact_ticks", max_impact_ticks),
            ("min_spread_ticks", min_spread_ticks),
            ("base_slippage_ticks", base_slippage_ticks),
            ("commission_bps", commission_bps),
            ("bag_commission_discount", bag_commission_discount),
            ("penalty_fee_per_contract", penalty_fee_per_contract),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not 0.0 <= bag_spread_factor <= 1.0:
            raise ValueError("bag_spread_factor must be within [0, 1]")
        if isinstance(monthly_volume, bool) or not isinstance(monthly_volume, int):
            raise TypeError("monthly_volume must be an int")
        if monthly_volume < 0:
            raise ValueError("monthly_volume must be non-negative")
        self._instrument_master = instrument_master
        self._fee_schedules: dict[str, FeeSchedule] = dict(fee_schedules or {})
        self.slippage_model = slippage_model
        self.impact_coefficient = float(impact_coefficient)
        self.max_impact_ticks = float(max_impact_ticks)
        self.min_spread_ticks = float(min_spread_ticks)
        self.base_slippage_ticks = float(base_slippage_ticks)
        self.commission_bps = float(commission_bps)
        self.bag_commission_discount = float(bag_commission_discount)
        self.bag_spread_factor = float(bag_spread_factor)
        self.penalty_fee_per_contract = float(penalty_fee_per_contract)
        self.monthly_volume = monthly_volume

    # -- instrument metadata helpers -----------------------------------------

    @property
    def instrument_master(self) -> InstrumentsConfig | None:
        """The injected instrument master, if any."""
        return self._instrument_master

    def _asset_class(self, root: str) -> AssetClass:
        if self._instrument_master is None or not self._instrument_master.has(root):
            return AssetClass.UNKNOWN
        return self._instrument_master.get(root).asset_class

    def _exchange(self, root: str) -> str | None:
        if self._instrument_master is None or not self._instrument_master.has(root):
            return None
        return self._instrument_master.get(root).exchange

    def fee_schedule(self, root: str) -> FeeSchedule:
        """Resolve the fee schedule for ``root``.

        Resolution order: explicit root override, explicit exchange override,
        explicit asset-class override, the venue default from the instrument
        master, then the instrument-class default.
        """
        key = root.strip().upper()
        if key in self._fee_schedules:
            return self._fee_schedules[key]
        exchange = self._exchange(key)
        if exchange is not None and exchange.upper() in self._fee_schedules:
            return self._fee_schedules[exchange.upper()]
        asset_class = self._asset_class(key)
        if asset_class.value in self._fee_schedules:
            return self._fee_schedules[asset_class.value]
        if exchange is not None:
            venue = EXCHANGE_FEE_SCHEDULES.get(exchange.upper())
            if venue is not None:
                return venue
        return default_fee_schedule(asset_class)

    def commission_per_contract(self, root: str, monthly_volume: int | None = None) -> float:
        """Tiered IBKR commission per contract for ``root``."""
        volume = self.monthly_volume if monthly_volume is None else monthly_volume
        return self.fee_schedule(root).commission_rate(volume)

    def fee_per_contract(self, root: str) -> float:
        """Exchange + clearing + NFA fees per contract for ``root``."""
        return self.fee_schedule(root).fee_per_contract

    # -- impact / spread estimation ------------------------------------------

    def participation_rate(self, quantity: int, bar_volume: float) -> float:
        """Order size as a fraction of bar volume (``1.0`` when volume is absent)."""
        if quantity < 0:
            raise ValueError("quantity must be non-negative")
        if not math.isfinite(bar_volume) or bar_volume < 0.0:
            raise ValueError("bar_volume must be finite and non-negative")
        if bar_volume <= 0.0:
            # No liquidity information: treat the order as fully participative.
            return 1.0
        return quantity / bar_volume

    def estimated_impact_ticks(
        self,
        quantity: int,
        bar_volume: float,
        *,
        extra_slippage_ticks: float = 0.0,
    ) -> float:
        """Volume-participation market impact in per-contract ticks."""
        if not math.isfinite(extra_slippage_ticks) or extra_slippage_ticks < 0.0:
            raise ValueError("extra_slippage_ticks must be finite and non-negative")
        participation = self.participation_rate(quantity, bar_volume)
        if self.slippage_model is SlippageModel.SQUARE_ROOT:
            raw = self.impact_coefficient * math.sqrt(participation)
        else:
            raw = self.impact_coefficient * participation
        impact = min(raw, self.max_impact_ticks) + self.base_slippage_ticks
        return impact + extra_slippage_ticks

    def estimated_spread_ticks(
        self,
        *,
        quoted_spread_ticks: float | None = None,
        is_bag: bool = False,
        limit_order: bool = False,
    ) -> float:
        """Per-contract half-spread cost in ticks.

        The full bid/ask spread is never tighter than ``min_spread_ticks`` and a
        crossing order pays half of it. A limit order is assumed to earn its
        price (no spread paid); a BAG combo trades at a calendar-spread quote so
        the per-leg spread is scaled by ``bag_spread_factor``.
        """
        if limit_order:
            return 0.0
        quoted = self.min_spread_ticks if quoted_spread_ticks is None else quoted_spread_ticks
        if not math.isfinite(quoted) or quoted < 0.0:
            raise ValueError("quoted_spread_ticks must be finite and non-negative")
        full_spread = max(self.min_spread_ticks, quoted)
        half = 0.5 * full_spread
        if is_bag:
            half *= self.bag_spread_factor
        return half

    # -- transaction costs ----------------------------------------------------

    def transaction_cost(
        self,
        root: str,
        quantity: int,
        price: float,
        tick_size: float,
        multiplier: float,
        bar_volume: float,
        *,
        monthly_volume: int | None = None,
        quoted_spread_ticks: float | None = None,
        is_bag: bool = False,
        limit_order: bool = False,
        extra_slippage_ticks: float = 0.0,
        extra_fee_per_contract: float = 0.0,
    ) -> CostBreakdown:
        """Full transaction cost for one order side.

        ``quantity`` is an absolute contract count (direction is applied by the
        caller through the fill price). A zero quantity returns a zero
        breakdown, which keeps rebalance arithmetic branch-free.
        """
        self._validate_order_inputs(root, quantity, price, tick_size, multiplier, bar_volume)
        if quantity == 0:
            return CostBreakdown.zero(root, price=price, tick_size=tick_size, multiplier=multiplier)
        if not math.isfinite(extra_fee_per_contract) or extra_fee_per_contract < 0.0:
            raise ValueError("extra_fee_per_contract must be finite and non-negative")
        schedule = self.fee_schedule(root)
        volume = self.monthly_volume if monthly_volume is None else monthly_volume
        if isinstance(volume, bool) or not isinstance(volume, int) or volume < 0:
            raise ValueError("monthly_volume must be a non-negative int")

        commission_rate = schedule.commission_rate(volume)
        if is_bag:
            commission_rate = max(0.0, commission_rate - self.bag_commission_discount)
        commission = commission_rate * quantity
        notional = quantity * price * multiplier
        if self.commission_bps > 0.0:
            commission += notional * self.commission_bps / 10_000.0
        fees = (schedule.fee_per_contract + extra_fee_per_contract) * quantity

        spread_ticks = self.estimated_spread_ticks(
            quoted_spread_ticks=quoted_spread_ticks,
            is_bag=is_bag,
            limit_order=limit_order,
        )
        slippage_ticks = (
            0.0
            if limit_order
            else self.estimated_impact_ticks(
                quantity, bar_volume, extra_slippage_ticks=extra_slippage_ticks
            )
        )
        tick_value = tick_size * multiplier
        return CostBreakdown(
            root=root.strip().upper(),
            quantity=quantity,
            price=price,
            tick_size=tick_size,
            multiplier=multiplier,
            spread_cost=spread_ticks * tick_value * quantity,
            slippage_cost=slippage_ticks * tick_value * quantity,
            commission=commission,
            fees=fees,
            funding=0.0,
            spread_ticks=spread_ticks,
            slippage_ticks=slippage_ticks,
            commission_per_contract=commission_rate,
            fee_per_contract=schedule.fee_per_contract + extra_fee_per_contract,
        )

    def round_trip_cost(
        self,
        root: str,
        quantity: int,
        price: float,
        tick_size: float,
        multiplier: float,
        bar_volume: float,
        *,
        monthly_volume: int | None = None,
        quoted_spread_ticks: float | None = None,
    ) -> CostBreakdown:
        """Cost of opening and closing ``quantity`` contracts at ``price``."""
        opening = self.transaction_cost(
            root,
            quantity,
            price,
            tick_size,
            multiplier,
            bar_volume,
            monthly_volume=monthly_volume,
            quoted_spread_ticks=quoted_spread_ticks,
        )
        closing = self.transaction_cost(
            root,
            quantity,
            price,
            tick_size,
            multiplier,
            bar_volume,
            monthly_volume=monthly_volume,
            quoted_spread_ticks=quoted_spread_ticks,
        )
        return opening.combined(closing)

    def roll_cost(
        self,
        root: str,
        quantity: int,
        from_price: float,
        to_price: float,
        tick_size: float,
        multiplier: float,
        bar_volume: float,
        *,
        monthly_volume: int | None = None,
        use_bag: bool = False,
        quoted_spread_ticks: float | None = None,
    ) -> RollCostBreakdown:
        """Two-legged roll cost, optionally with the native BAG combo discount."""
        self._validate_order_inputs(root, quantity, from_price, tick_size, multiplier, bar_volume)
        if not math.isfinite(to_price) or to_price <= 0.0:
            raise ValueError("to_price must be finite and strictly positive")
        volume = self.monthly_volume if monthly_volume is None else monthly_volume
        close_leg = self.transaction_cost(
            root,
            quantity,
            from_price,
            tick_size,
            multiplier,
            bar_volume,
            monthly_volume=volume,
            quoted_spread_ticks=quoted_spread_ticks,
            is_bag=use_bag,
        )
        open_leg = self.transaction_cost(
            root,
            quantity,
            to_price,
            tick_size,
            multiplier,
            bar_volume,
            monthly_volume=volume,
            quoted_spread_ticks=quoted_spread_ticks,
            is_bag=use_bag,
        )
        bag_discount = 0.0
        if use_bag:
            independent_close = self.transaction_cost(
                root,
                quantity,
                from_price,
                tick_size,
                multiplier,
                bar_volume,
                monthly_volume=volume,
                quoted_spread_ticks=quoted_spread_ticks,
                is_bag=False,
            )
            independent_open = self.transaction_cost(
                root,
                quantity,
                to_price,
                tick_size,
                multiplier,
                bar_volume,
                monthly_volume=volume,
                quoted_spread_ticks=quoted_spread_ticks,
                is_bag=False,
            )
            bag_discount = max(
                0.0,
                independent_close.total + independent_open.total - close_leg.total - open_leg.total,
            )
        return RollCostBreakdown(
            root=root.strip().upper(),
            quantity=quantity,
            from_price=from_price,
            to_price=to_price,
            tick_size=tick_size,
            multiplier=multiplier,
            close_leg=close_leg,
            open_leg=open_leg,
            bag_discount=bag_discount,
        )

    def forced_liquidation_cost(
        self,
        root: str,
        quantity: int,
        price: float,
        tick_size: float,
        multiplier: float,
        bar_volume: float,
        *,
        penalty_ticks: float | None = None,
        penalty_fee_per_contract: float | None = None,
        quoted_spread_ticks: float | None = None,
    ) -> CostBreakdown:
        """Cost of a forced broker liquidation (penalty slippage + penalty fee)."""
        # ``penalty_ticks`` is an extra price concession; the fee is per contract.
        extra_ticks = 2.0 if penalty_ticks is None else penalty_ticks
        extra_fee = (
            self.penalty_fee_per_contract
            if penalty_fee_per_contract is None
            else penalty_fee_per_contract
        )
        return self.transaction_cost(
            root,
            quantity,
            price,
            tick_size,
            multiplier,
            bar_volume,
            quoted_spread_ticks=quoted_spread_ticks,
            extra_slippage_ticks=max(0.0, extra_ticks),
            extra_fee_per_contract=max(0.0, extra_fee),
        )

    # -- metadata -------------------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable description of the cost model (run metadata)."""
        return {
            "slippage_model": self.slippage_model.value,
            "impact_coefficient": self.impact_coefficient,
            "max_impact_ticks": self.max_impact_ticks,
            "min_spread_ticks": self.min_spread_ticks,
            "base_slippage_ticks": self.base_slippage_ticks,
            "commission_bps": self.commission_bps,
            "bag_commission_discount": self.bag_commission_discount,
            "bag_spread_factor": self.bag_spread_factor,
            "penalty_fee_per_contract": self.penalty_fee_per_contract,
            "monthly_volume": self.monthly_volume,
        }

    @staticmethod
    def _validate_order_inputs(
        root: str,
        quantity: int,
        price: float,
        tick_size: float,
        multiplier: float,
        bar_volume: float,
    ) -> None:
        if not isinstance(root, str) or not root.strip():
            raise ValueError("root must be a non-empty string")
        if isinstance(quantity, bool) or not isinstance(quantity, int):
            raise TypeError("quantity must be an int")
        if quantity < 0:
            raise ValueError("quantity must be non-negative")
        for name, value, strictly_positive in (
            ("price", price, True),
            ("tick_size", tick_size, True),
            ("multiplier", multiplier, True),
            ("bar_volume", bar_volume, False),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if strictly_positive and value <= 0.0:
                raise ValueError(f"{name} must be strictly positive")
            if not strictly_positive and value < 0.0:
                raise ValueError(f"{name} must be non-negative")
