"""Futures-native backtest engine: vectorised and event-driven execution.

``BacktestEngine`` is the single entry point the API, CLI and research notebooks
use to run a FinRL-X strategy against historical futures bars. It consumes only
the frozen contracts (:class:`~src.core.interfaces.StrategyPipeline`,
:class:`~src.core.interfaces.ContractTranslatorProtocol`, ``MarketState``) and
injected callables, so the data layer, strategy layer and risk layer can evolve
independently of the backtester.

Two execution modes return the *same* typed :class:`BacktestResult`:

``vectorized``
    Weights -> notional returns -> equity, with Polars-backed equity curves. Fast
    enough for parameter sweeps; margin utilisation is estimated from fractional
    implied contracts.

``event``
    A bar-by-bar loop through :class:`~src.backtest.simulated_broker.SimulatedBroker`
    with daily mark-to-market settlement, margin-call forced liquidation,
    two-legged roll execution and discrete contract accounting. This is the mode
    that must reconcile with live/paper execution.

Accounting identity (enforced on every run, tolerance ``1e-6`` relative)::

    Ending Equity = Starting Equity + PricePnL + RollPnL
                    - Commissions - Fees - Slippage - Funding

``Commissions`` are IBKR commissions (the ``commissions`` field) and ``Fees`` are
CME/CBOT/NYMEX/COMEX exchange + clearing + NFA fees; together they are the
"commissions" term of the specification. ``Slippage`` is the spread plus
volume-impact execution drag. Roll PnL is the basis gap correction
``-quantity * (F_new - F_old) * multiplier`` at each roll, which decomposes total
PnL into directional price return and roll yield (carry) exactly.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import statistics
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from itertools import pairwise
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

import numpy as np
import polars as pl
from configs.instruments import InstrumentsConfig, load_instruments
from configs.strategy import StrategyConfig, load_strategy_config
from numpy.typing import NDArray

from src.backtest.cost_model import CostModel
from src.backtest.margin_simulator import MarginRequirements, MarginSimulator, utc_calendar_day
from src.backtest.simulated_broker import (
    BarQuote,
    FillRecord,
    SimulatedBroker,
    TradeKind,
)
from src.backtest.walk_forward import PurgedWalkForwardSplit, WalkForwardFold
from src.core.interfaces import ContractTranslatorProtocol, StrategyPipeline
from src.core.types import (
    TRADING_DAYS_PER_YEAR,
    AssetClass,
    ContractTarget,
    ContractTargetSet,
    InstrumentId,
    InstrumentSnapshot,
    MarketState,
    OrderEvent,
    OrderSide,
    PipelineResult,
    Position,
    ScalarValue,
    WeightVector,
    ensure_utc,
)
from src.data.schema import (
    CLOSE_COLUMN,
    EXPIRATION_COLUMN,
    HIGH_COLUMN,
    LOW_COLUMN,
    OPEN_COLUMN,
    OPEN_INTEREST_COLUMN,
    ROOT_SYMBOL_COLUMN,
    SETTLEMENT_COLUMN,
    TIMESTAMP_COLUMN,
    VOLUME_COLUMN,
    ensure_bars_schema,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids importing analytics eagerly
    from src.analytics.metrics import PerformanceMetrics

#: Callable contract for building a ``MarketState`` from a bar frame.
type StateBuilder = Callable[[pl.DataFrame, datetime, float], MarketState]

#: Equity-curve schema shared by both backtest modes.
EQUITY_CURVE_SCHEMA: Mapping[str, pl.DataType | type[pl.DataType]] = MappingProxyType(
    {
        TIMESTAMP_COLUMN: pl.Datetime("us", "UTC"),
        "equity": pl.Float64,
        "cash": pl.Float64,
        "unrealized_pnl": pl.Float64,
        "price_pnl": pl.Float64,
        "roll_pnl": pl.Float64,
        "commissions": pl.Float64,
        "fees": pl.Float64,
        "slippage": pl.Float64,
        "funding": pl.Float64,
        "margin_used": pl.Float64,
        "margin_utilization": pl.Float64,
        "gross_exposure": pl.Float64,
        "net_exposure": pl.Float64,
        "drawdown": pl.Float64,
        "turnover": pl.Float64,
        "n_positions": pl.Int64,
    }
)

#: Weight-history schema (FinRL-X pipeline waterfall inspector contract).
WEIGHT_HISTORY_SCHEMA: Mapping[str, pl.DataType | type[pl.DataType]] = MappingProxyType(
    {
        TIMESTAMP_COLUMN: pl.Datetime("us", "UTC"),
        "stage": pl.Utf8,
        "instrument_id": pl.Utf8,
        "weight": pl.Float64,
    }
)


class BacktestMode(StrEnum):
    """Execution mode of the backtest engine."""

    EVENT = "event"
    VECTORIZED = "vectorized"


class AccountingIdentityError(AssertionError):
    """Raised when a backtest violates the accounting identity."""


# ---------------------------------------------------------------------------
# Rebalance schedule
# ---------------------------------------------------------------------------

_WEEKDAYS: Mapping[str, int] = MappingProxyType(
    {"MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4, "SAT": 5, "SUN": 6}
)


@dataclass(frozen=True, slots=True)
class RebalanceSchedule:
    """When the pipeline is re-run and the portfolio re-targeted.

    Exactly one of ``every_n_bars``, ``weekday`` or ``monthly`` is used; the
    first bar of a run always rebalances so a position can be established.
    """

    every_n_bars: int | None = None
    weekday: int | None = None
    monthly: bool = False

    def __post_init__(self) -> None:
        active = sum(
            value is not None and value is not False
            for value in (self.every_n_bars, self.weekday, self.monthly)
        )
        if active != 1:
            raise ValueError("exactly one rebalance rule must be configured")
        if self.every_n_bars is not None:
            if isinstance(self.every_n_bars, bool) or not isinstance(self.every_n_bars, int):
                raise TypeError("every_n_bars must be an int")
            if self.every_n_bars < 1:
                raise ValueError("every_n_bars must be >= 1")
        if self.weekday is not None:
            if isinstance(self.weekday, bool) or not isinstance(self.weekday, int):
                raise TypeError("weekday must be an int")
            if not 0 <= self.weekday <= 6:
                raise ValueError("weekday must be within 0..6 (Monday=0)")

    def should_rebalance(
        self,
        index: int,
        timestamp: datetime,
        previous_timestamp: datetime | None = None,
    ) -> bool:
        """Whether the portfolio should be re-targeted at this bar."""
        if index == 0:
            return True
        if self.every_n_bars is not None:
            return index % self.every_n_bars == 0
        if self.weekday is not None:
            return ensure_utc(timestamp).weekday() == self.weekday
        return (
            previous_timestamp is None
            or ensure_utc(timestamp).month != ensure_utc(previous_timestamp).month
        )

    @classmethod
    def daily(cls) -> RebalanceSchedule:
        """Rebalance on every bar."""
        return cls(every_n_bars=1)

    @classmethod
    def every_n(cls, bars: int) -> RebalanceSchedule:
        """Rebalance every ``bars`` bars."""
        return cls(every_n_bars=bars)

    @classmethod
    def weekly(cls, weekday: int = 4) -> RebalanceSchedule:
        """Rebalance on one weekday (Friday by default)."""
        return cls(weekday=weekday)

    @classmethod
    def from_frequency(cls, value: str) -> RebalanceSchedule:
        """Parse a declarative frequency string (``D``, ``W-FRI``, ``M``, ``5``)."""
        text = value.strip().upper()
        if not text:
            raise ValueError("rebalance frequency must be a non-empty string")
        if text in {"D", "DAILY", "1D"}:
            return cls.daily()
        if text in {"M", "MONTHLY"}:
            return cls(monthly=True)
        if text.startswith("W-") and len(text) == 5 and text[2:] in _WEEKDAYS:
            return cls.weekly(_WEEKDAYS[text[2:]])
        if text in {"W", "WEEKLY"}:
            return cls.weekly()
        numeric = text[:-1] if text.endswith("B") else text
        try:
            bars = int(numeric)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(f"unsupported rebalance frequency {value!r}") from exc
        if bars < 1:
            raise ValueError("rebalance frequency must be >= 1 bar")
        return cls.every_n(bars)

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable representation."""
        return {
            "every_n_bars": self.every_n_bars,
            "weekday": self.weekday,
            "monthly": self.monthly,
        }


# ---------------------------------------------------------------------------
# Backtest result records
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TradeRecord:
    """One executed trade in the backtest (normalised across both modes)."""

    timestamp: datetime
    order_id: str
    instrument_id: InstrumentId
    side: OrderSide
    quantity: int
    price: float
    reference_price: float
    kind: str
    notional: float = 0.0
    commissions: float = 0.0
    slippage: float = 0.0
    execution_cost_ticks: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", ensure_utc(self.timestamp))
        if self.quantity <= 0:
            raise ValueError("trade quantity must be strictly positive")
        if self.price <= 0.0 or self.reference_price <= 0.0:
            raise ValueError("trade prices must be strictly positive")
        for name in (
            "notional",
            "commissions",
            "slippage",
            "execution_cost_ticks",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)

    @property
    def signed_quantity(self) -> int:
        """Signed trade quantity (+ buys, - sells)."""
        return self.side.multiplier * self.quantity

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable representation."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "order_id": self.order_id,
            "instrument_id": self.instrument_id,
            "side": self.side.value,
            "quantity": self.quantity,
            "price": self.price,
            "reference_price": self.reference_price,
            "kind": self.kind,
            "notional": self.notional,
            "commissions": self.commissions,
            "slippage": self.slippage,
            "execution_cost_ticks": self.execution_cost_ticks,
        }


@dataclass(frozen=True, slots=True)
class RollExecution:
    """One simulated contract roll (two-legged execution + roll-yield attribution)."""

    timestamp: datetime
    instrument_id: InstrumentId
    quantity: int
    from_expiration: date | None
    to_expiration: date | None
    from_price: float
    to_price: float
    roll_pnl: float
    total_cost: float
    bag_used: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", ensure_utc(self.timestamp))
        if self.quantity <= 0:
            raise ValueError("roll quantity must be strictly positive")
        for name in ("from_price", "to_price", "roll_pnl", "total_cost"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)

    @property
    def roll_gap(self) -> float:
        """Price gap ``F_new - F_old`` at the roll instant."""
        return self.to_price - self.from_price

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable representation."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "instrument_id": self.instrument_id,
            "quantity": self.quantity,
            "from_expiration": self.from_expiration.isoformat() if self.from_expiration else None,
            "to_expiration": self.to_expiration.isoformat() if self.to_expiration else None,
            "from_price": self.from_price,
            "to_price": self.to_price,
            "roll_gap": self.roll_gap,
            "roll_pnl": self.roll_pnl,
            "total_cost": self.total_cost,
            "bag_used": self.bag_used,
        }


@dataclass(frozen=True, slots=True)
class ForcedLiquidationEvent:
    """One simulated maintenance-margin forced liquidation."""

    timestamp: datetime
    equity_before: float
    maintenance_margin: float
    equity_after: float
    positions_closed: tuple[InstrumentId, ...]
    penalty_cost: float
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", ensure_utc(self.timestamp))
        for name in ("equity_before", "maintenance_margin", "equity_after", "penalty_cost"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable representation."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "equity_before": self.equity_before,
            "maintenance_margin": self.maintenance_margin,
            "equity_after": self.equity_after,
            "positions_closed": list(self.positions_closed),
            "penalty_cost": self.penalty_cost,
            "reason": self.reason,
        }


@dataclass(slots=True)
class _Attribution:
    """Mutable accumulator for the backtest attribution components."""

    price_pnl: float = 0.0
    roll_pnl: float = 0.0
    commissions: float = 0.0
    fees: float = 0.0
    slippage: float = 0.0
    funding: float = 0.0
    turnover: float = 0.0

    def add_fill(self, fill: FillRecord) -> None:
        """Accumulate one fill's costs and notional turnover."""
        self.commissions += fill.cost.commission
        self.fees += fill.cost.fees
        self.slippage += fill.cost.spread_cost + fill.cost.slippage_cost
        self.funding += fill.cost.funding
        self.turnover += fill.notional

    @property
    def total_commissions(self) -> float:
        """IBKR commission plus exchange/regulatory fees."""
        return self.commissions + self.fees

    @property
    def total_costs(self) -> float:
        """Every execution cost component (commissions + fees + slippage + funding)."""
        return self.total_commissions + self.slippage + self.funding


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Immutable, JSON-serialisable result of one backtest run.

    The equity curve is a :class:`polars.DataFrame` with the
    :data:`EQUITY_CURVE_SCHEMA` columns; per-stage pipeline weights are in
    ``weight_history`` (long format) so the API can render the FinRL-X waterfall.
    """

    run_id: str
    strategy_id: str
    mode: BacktestMode
    start: datetime
    end: datetime
    initial_equity: float
    ending_equity: float
    equity_curve: pl.DataFrame
    weight_history: pl.DataFrame = field(default_factory=pl.DataFrame)
    trades: tuple[TradeRecord, ...] = ()
    fills: tuple[FillRecord, ...] = ()
    roll_events: tuple[RollExecution, ...] = ()
    forced_liquidations: tuple[ForcedLiquidationEvent, ...] = ()
    price_pnl: float = 0.0
    roll_pnl: float = 0.0
    commissions: float = 0.0
    fees: float = 0.0
    slippage: float = 0.0
    funding: float = 0.0
    turnover: float = 0.0
    margin_used_max: float = 0.0
    margin_utilization_avg: float = 0.0
    margin_utilization_peak: float = 0.0
    settlement_pnl: float = 0.0
    #: Cost-breakdown transparency (task-86): roll/BAG execution cost components
    #: are subsets of ``commissions``/``fees``/``slippage`` (never double-counted).
    roll_cost_total: float = 0.0
    roll_commissions: float = 0.0
    roll_fees: float = 0.0
    roll_slippage: float = 0.0
    forced_liquidation_cost: float = 0.0
    #: Stress-regime/calibrated margin provenance (task-86).
    margin_stress_applied: bool = False
    margin_stress_multiplier: float = 1.0
    margin_stress_roots: tuple[str, ...] = ()
    margin_calibrated: bool = False
    margin_calibrated_roots: tuple[str, ...] = ()
    metadata: Mapping[str, ScalarValue] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", ensure_utc(self.start))
        object.__setattr__(self, "end", ensure_utc(self.end))
        for name in (
            "initial_equity",
            "ending_equity",
            "price_pnl",
            "roll_pnl",
            "commissions",
            "fees",
            "slippage",
            "funding",
            "turnover",
            "margin_used_max",
            "margin_utilization_avg",
            "margin_utilization_peak",
            "settlement_pnl",
            "roll_cost_total",
            "roll_commissions",
            "roll_fees",
            "roll_slippage",
            "forced_liquidation_cost",
            "margin_stress_multiplier",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "margin_stress_roots", tuple(sorted(self.margin_stress_roots)))
        object.__setattr__(
            self, "margin_calibrated_roots", tuple(sorted(self.margin_calibrated_roots))
        )
        object.__setattr__(self, "margin_stress_applied", bool(self.margin_stress_applied))
        object.__setattr__(self, "margin_calibrated", bool(self.margin_calibrated))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    # -- headline statistics ---------------------------------------------------

    @property
    def net_pnl(self) -> float:
        """Ending equity minus starting equity."""
        return self.ending_equity - self.initial_equity

    @property
    def total_return(self) -> float:
        """Simple return over the whole run."""
        if self.initial_equity <= 0.0:
            return 0.0
        return self.net_pnl / self.initial_equity

    @property
    def total_commissions(self) -> float:
        """IBKR commission plus exchange/regulatory fees."""
        return self.commissions + self.fees

    @property
    def total_costs(self) -> float:
        """Every execution cost component."""
        return self.total_commissions + self.slippage + self.funding

    @property
    def cost_breakdown(self) -> dict[str, float]:
        """Exact execution-cost breakdown for result/tearsheet/API consumers.

        ``*_total`` fields are the headline components summed by
        :attr:`total_costs`; the ``roll_*`` entries are subsets of those headline
        components attributable to roll_close/roll_open executions (never added
        again), and ``non_roll_*`` are the complement. ``forced_liquidation_cost``
        is a subset of commissions/fees/slippage from liquidation fills.
        """
        non_roll_commissions = max(0.0, self.commissions - self.roll_commissions)
        non_roll_fees = max(0.0, self.fees - self.roll_fees)
        non_roll_slippage = max(0.0, self.slippage - self.roll_slippage)
        return {
            "broker_commissions": self.commissions,
            "exchange_and_regulatory_fees": self.fees,
            "slippage": self.slippage,
            "funding": self.funding,
            "roll_cost_total": self.roll_cost_total,
            "roll_commissions": self.roll_commissions,
            "roll_exchange_and_regulatory_fees": self.roll_fees,
            "roll_slippage": self.roll_slippage,
            "non_roll_commissions": non_roll_commissions,
            "non_roll_exchange_and_regulatory_fees": non_roll_fees,
            "non_roll_slippage": non_roll_slippage,
            "forced_liquidation_cost": self.forced_liquidation_cost,
            "total_commissions": self.total_commissions,
            "total_execution_costs": self.total_commissions + self.slippage,
            "total_costs": self.total_costs,
        }

    @property
    def n_bars(self) -> int:
        """Number of bars in the equity curve."""
        return self.equity_curve.height

    @property
    def n_trades(self) -> int:
        """Number of executed trades."""
        return len(self.trades)

    @property
    def n_rebalances(self) -> int:
        """Number of distinct rebalance timestamps in the weight history."""
        if self.weight_history.height == 0:
            return 0
        return int(self.weight_history.select(pl.col(TIMESTAMP_COLUMN).n_unique()).item())

    @property
    def accounting_residual(self) -> float:
        """Residual of the accounting identity (should be ~0)."""
        return self.ending_equity - (
            self.initial_equity
            + self.price_pnl
            + self.roll_pnl
            - self.total_commissions
            - self.slippage
            - self.funding
        )

    def check_accounting_identity(self, tolerance: float = 1e-6) -> None:
        """Assert ``Ending = Starting + PricePnL + RollPnL - Costs`` within tolerance."""
        residual = self.accounting_residual
        scale = max(1.0, abs(self.initial_equity), abs(self.ending_equity))
        if abs(residual) > tolerance * scale:
            raise AccountingIdentityError(
                "accounting identity violated: residual "
                f"{residual:.6f} > tolerance {tolerance * scale:.6f} "
                f"(ending={self.ending_equity:.2f}, starting={self.initial_equity:.2f}, "
                f"price_pnl={self.price_pnl:.2f}, roll_pnl={self.roll_pnl:.2f}, "
                f"commissions={self.total_commissions:.2f}, slippage={self.slippage:.2f})"
            )

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable summary (frames are expanded to plain lists)."""
        return {
            "run_id": self.run_id,
            "strategy_id": self.strategy_id,
            "mode": self.mode.value,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "initial_equity": self.initial_equity,
            "ending_equity": self.ending_equity,
            "net_pnl": self.net_pnl,
            "total_return": self.total_return,
            "price_pnl": self.price_pnl,
            "roll_pnl": self.roll_pnl,
            "commissions": self.commissions,
            "fees": self.fees,
            "slippage": self.slippage,
            "funding": self.funding,
            "total_costs": self.total_costs,
            "turnover": self.turnover,
            "margin_used_max": self.margin_used_max,
            "margin_utilization_avg": self.margin_utilization_avg,
            "margin_utilization_peak": self.margin_utilization_peak,
            "settlement_pnl": self.settlement_pnl,
            "roll_cost_total": self.roll_cost_total,
            "roll_commissions": self.roll_commissions,
            "roll_fees": self.roll_fees,
            "roll_slippage": self.roll_slippage,
            "forced_liquidation_cost": self.forced_liquidation_cost,
            "cost_breakdown": self.cost_breakdown,
            "margin_stress_applied": self.margin_stress_applied,
            "margin_stress_multiplier": self.margin_stress_multiplier,
            "margin_stress_roots": list(self.margin_stress_roots),
            "margin_calibrated": self.margin_calibrated,
            "margin_calibrated_roots": list(self.margin_calibrated_roots),
            "n_bars": self.n_bars,
            "n_trades": self.n_trades,
            "n_rolls": len(self.roll_events),
            "n_forced_liquidations": len(self.forced_liquidations),
            "accounting_residual": self.accounting_residual,
            "metadata": dict(self.metadata.items()),
            "warnings": list(self.warnings),
            "equity_curve": self._curve_records(),
            "roll_events": [event.to_dict() for event in self.roll_events],
            "forced_liquidations": [event.to_dict() for event in self.forced_liquidations],
            "trades": [trade.to_dict() for trade in self.trades],
        }

    def _curve_records(self) -> list[dict[str, object]]:
        if self.equity_curve.height == 0:
            return []
        columns = (
            "equity",
            "cash",
            "unrealized_pnl",
            "price_pnl",
            "roll_pnl",
            "commissions",
            "fees",
            "slippage",
            "margin_used",
            "margin_utilization",
            "gross_exposure",
            "net_exposure",
            "drawdown",
            "turnover",
        )
        timestamps: list[datetime] = [
            ensure_utc(value) for value in self.equity_curve.get_column(TIMESTAMP_COLUMN).to_list()
        ]
        data = {column: self.equity_curve.get_column(column).to_list() for column in columns}
        records: list[dict[str, object]] = []
        for index, timestamp in enumerate(timestamps):
            record: dict[str, object] = {"timestamp": timestamp.isoformat()}
            for column in columns:
                record[column] = data[column][index]
            records.append(record)
        return records


# ---------------------------------------------------------------------------
# Bar/context helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _BarRow:
    """One canonical bar row used by the engine's internal lookups."""

    timestamp: datetime
    root: str
    expiration: date | None
    open: float
    high: float
    low: float
    close: float
    volume: float
    open_interest: float
    settlement: float | None


@dataclass(frozen=True, slots=True)
class _BarContext:
    """Pre-processed bar frame with timestamp/contract lookups."""

    frame: pl.DataFrame
    timestamps: tuple[datetime, ...]
    rows_by_timestamp: Mapping[datetime, tuple[_BarRow, ...]]
    active: Mapping[datetime, Mapping[str, _BarRow]]

    def contract_price(
        self, timestamp: datetime, root: str, expiration: date | None
    ) -> float | None:
        """Close price of a specific contract at ``timestamp`` (``None`` if absent)."""
        for row in self.rows_by_timestamp.get(timestamp, ()):
            if row.root == root and row.expiration == expiration:
                return row.close
        return None

    def contract_volume(self, timestamp: datetime, root: str, expiration: date | None) -> float:
        """Bar volume of a specific contract at ``timestamp`` (``0`` if absent)."""
        for row in self.rows_by_timestamp.get(timestamp, ()):
            if row.root == root and row.expiration == expiration:
                return row.volume
        return 0.0


def _prepare_bars(bars: pl.DataFrame) -> _BarContext:
    """Validate, sort and index a canonical bar frame."""
    normalised = ensure_bars_schema(bars)
    if normalised.height == 0:
        raise ValueError("backtest bars frame is empty")
    ordered = normalised.sort([TIMESTAMP_COLUMN, ROOT_SYMBOL_COLUMN, EXPIRATION_COLUMN])
    rows_by_timestamp: dict[datetime, list[_BarRow]] = {}
    timestamps: dict[datetime, None] = {}
    for raw in ordered.iter_rows(named=True):
        timestamp = raw[TIMESTAMP_COLUMN]
        if not isinstance(timestamp, datetime):
            raise ValueError("timestamp_utc column must contain datetimes")
        timestamp = ensure_utc(timestamp)
        root = raw[ROOT_SYMBOL_COLUMN]
        if not isinstance(root, str) or not root:
            raise ValueError("root_symbol column must contain non-empty strings")
        expiration = raw[EXPIRATION_COLUMN]
        if expiration is not None and not isinstance(expiration, date):
            raise ValueError("expiration_date column must contain dates")
        settlement = raw[SETTLEMENT_COLUMN]
        row = _BarRow(
            timestamp=timestamp,
            root=root.strip().upper(),
            expiration=expiration,
            open=float(raw[OPEN_COLUMN]),
            high=float(raw[HIGH_COLUMN]),
            low=float(raw[LOW_COLUMN]),
            close=float(raw[CLOSE_COLUMN]),
            volume=float(raw[VOLUME_COLUMN]),
            open_interest=float(raw[OPEN_INTEREST_COLUMN]),
            settlement=float(settlement) if settlement is not None else None,
        )
        rows_by_timestamp.setdefault(timestamp, []).append(row)
        timestamps.setdefault(timestamp, None)
    active: dict[datetime, Mapping[str, _BarRow]] = {}
    for timestamp, rows in rows_by_timestamp.items():
        # Rows are sorted by expiration, so the first row per root is the front
        # contract: the deterministic front-month rule used by the state builder.
        front: dict[str, _BarRow] = {}
        for row in rows:
            front.setdefault(row.root, row)
        active[timestamp] = MappingProxyType(front)
    return _BarContext(
        frame=normalised,
        timestamps=tuple(sorted(timestamps)),
        rows_by_timestamp=MappingProxyType(
            {timestamp: tuple(rows) for timestamp, rows in rows_by_timestamp.items()}
        ),
        active=MappingProxyType(active),
    )


def _rolling_stat(values: NDArray[np.float64], window: int, *, std: bool) -> NDArray[np.float64]:
    """Rolling sum (or sample std) with ``nan`` for the first ``window - 1`` values."""
    result = np.full(values.shape, np.nan, dtype=np.float64)
    if window < 1 or values.size < window:
        return result
    for index in range(window - 1, values.size):
        chunk = values[index - window + 1 : index + 1]
        if std:
            result[index] = float(np.std(chunk, ddof=1)) if chunk.size > 1 else 0.0
        else:
            result[index] = float(np.sum(chunk))
    return result


def _bar_features(
    closes: NDArray[np.float64], volumes: NDArray[np.float64], oi: NDArray[np.float64], index: int
) -> dict[str, float]:
    """Point-in-time features for one root at ``index`` (no future data used)."""
    rets = np.zeros(max(closes.size - 1, 1), dtype=np.float64)
    if closes.size > 1:
        rets = closes[1:] / closes[:-1] - 1.0
    # ``rets[i - 1]`` is the return realised *into* bar ``i``: using ``rets[i]``
    # would leak the next bar's return into the current state.
    ret_1d = float(rets[index - 1]) if 0 < index <= rets.size else 0.0
    rets_history = rets[: max(index, 0)]
    features: dict[str, float] = {
        "ret_1d": ret_1d,
        "ret_5d": float(np.sum(rets_history[-5:])) if rets_history.size else 0.0,
        "ret_21d": float(np.sum(rets_history[-21:])) if rets_history.size else 0.0,
        "ret_63d": float(np.sum(rets_history[-63:])) if rets_history.size else 0.0,
        "ret_126d": float(np.sum(rets_history[-126:])) if rets_history.size else 0.0,
        "ret_252d": float(np.sum(rets_history[-252:])) if rets_history.size else 0.0,
        "vol_20d": float(np.std(rets_history[-20:], ddof=1)) if rets_history.size > 2 else 0.0,
        "vol_60d": float(np.std(rets_history[-60:], ddof=1)) if rets_history.size > 2 else 0.0,
        "tsmom_21": float(np.sum(rets_history[-21:])) if rets_history.size else 0.0,
        "tsmom_126": float(np.sum(rets_history[-126:])) if rets_history.size else 0.0,
        "term_slope": 0.0,
        "carry_annualized": 0.0,
        "volume_oi_ratio": float(volumes[index] / oi[index]) if oi[index] > 0.0 else 0.0,
    }
    return features


def _contract_symbol(root: str, expiration: date | None) -> str:
    if expiration is None:
        return root
    return f"{root}_{expiration.strftime('%Y%m%d')}"


def build_states_from_bars(
    bars: pl.DataFrame,
    instrument_master: InstrumentsConfig | None = None,
    *,
    initial_equity: float | None = None,
    timeframe: str = "1d",
) -> list[MarketState]:
    """Build deterministic point-in-time market states from canonical bars.

    This is the engine's built-in offline state builder (the production one lives
    in :mod:`src.data.market_state_builder` and can be injected instead). It
    selects the front contract per root, derives trailing return/volatility
    features only from bars at or before the timestamp (zero look-ahead) and
    attaches a synthetic one-tick bid/ask quote around the close.
    """
    context = _prepare_bars(bars)
    root_series: dict[str, list[_BarRow]] = {}
    for timestamp in context.timestamps:
        for root, row in context.active[timestamp].items():
            root_series.setdefault(root, []).append(row)
    specs: dict[str, tuple[float, AssetClass]] = {}
    if instrument_master is not None:
        for root in root_series:
            if instrument_master.has(root):
                spec = instrument_master.get(root)
                specs[root] = (float(spec.tick_size), spec.asset_class)
    closes: dict[str, NDArray[np.float64]] = {
        root: np.array([row.close for row in rows], dtype=np.float64)
        for root, rows in root_series.items()
    }
    volumes: dict[str, NDArray[np.float64]] = {
        root: np.array([row.volume for row in rows], dtype=np.float64)
        for root, rows in root_series.items()
    }
    open_interests: dict[str, NDArray[np.float64]] = {
        root: np.array([row.open_interest for row in rows], dtype=np.float64)
        for root, rows in root_series.items()
    }
    positions: dict[str, dict[datetime, int]] = {
        root: {row.timestamp: index for index, row in enumerate(rows)}
        for root, rows in root_series.items()
    }
    equity = initial_equity if initial_equity is not None and initial_equity > 0.0 else None
    states: list[MarketState] = []
    for timestamp in context.timestamps:
        snapshots: list[InstrumentSnapshot] = []
        for root in sorted(context.active[timestamp]):
            row = context.active[timestamp][root]
            tick_size, asset_class = specs.get(root, (0.01, AssetClass.UNKNOWN))
            half_spread = min(0.5 * tick_size, row.close * 0.0001)
            index = positions[root][timestamp]
            features = _bar_features(closes[root], volumes[root], open_interests[root], index)
            snapshots.append(
                InstrumentSnapshot(
                    instrument_id=root,
                    root_symbol=root,
                    price=row.close,
                    asset_class=asset_class,
                    expiration=row.expiration,
                    bid=max(row.close - half_spread, tick_size * 0.5),
                    ask=row.close + half_spread,
                    volume=row.volume,
                    open_interest=row.open_interest,
                    settlement=row.settlement or row.close,
                    features=features,
                )
            )
        states.append(
            MarketState(
                timestamp=timestamp,
                instruments={snapshot.instrument_id: snapshot for snapshot in snapshots},
                equity=equity,
                timeframe=timeframe,
            )
        )
    return states


def default_state_builder(
    bars: pl.DataFrame,
    timestamp: datetime,
    equity: float,
    *,
    instrument_master: InstrumentsConfig | None = None,
    timeframe: str = "1d",
) -> MarketState:
    """Example injected-builder signature: state for one timestamp.

    Production deployments inject :mod:`src.data.market_state_builder`; this
    function exists so callers can rely on the documented callable contract
    without importing the data layer.
    """
    states = build_states_from_bars(
        bars, instrument_master, initial_equity=equity, timeframe=timeframe
    )
    target = ensure_utc(timestamp)
    for state in states:
        if state.timestamp == target:
            return state
    raise KeyError(f"no bar available at {target.isoformat()}")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def _empty_weight_history() -> pl.DataFrame:
    return pl.DataFrame(schema=WEIGHT_HISTORY_SCHEMA)


def _curve_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=EQUITY_CURVE_SCHEMA)
    frame = pl.DataFrame(rows)
    return frame.select(
        [pl.col(column).cast(dtype) for column, dtype in EQUITY_CURVE_SCHEMA.items()]
    )


def _weight_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    if not rows:
        return _empty_weight_history()
    frame = pl.DataFrame(rows)
    return frame.select(
        [pl.col(column).cast(dtype) for column, dtype in WEIGHT_HISTORY_SCHEMA.items()]
    )


class EventBacktester:
    """Bar-by-bar event-driven backtester reusing the live ``BrokerAdapter``."""

    def __init__(
        self,
        strategy: StrategyPipeline,
        translator: ContractTranslatorProtocol,
        *,
        instrument_master: InstrumentsConfig | None = None,
        cost_model: CostModel | None = None,
        margin_simulator: MarginSimulator | None = None,
        broker: SimulatedBroker | None = None,
        initial_equity: float = 1_000_000.0,
        rebalance_schedule: RebalanceSchedule | None = None,
        rebalance_every: int = 1,
        state_builder: StateBuilder | None = None,
        penalty_ticks: float = 2.0,
        penalty_fee_per_contract: float = 15.0,
        use_bag_rolls: bool = False,
        annualization_factor: int = TRADING_DAYS_PER_YEAR,
        timeframe: str = "1d",
    ) -> None:
        if not math.isfinite(initial_equity) or initial_equity <= 0.0:
            raise ValueError("initial_equity must be finite and strictly positive")
        if not isinstance(annualization_factor, int) or annualization_factor < 1:
            raise ValueError("annualization_factor must be a positive int")
        self._strategy = strategy
        self._translator = translator
        self._instrument_master = instrument_master
        self._cost_model = cost_model or CostModel(instrument_master=instrument_master)
        self._margin_simulator = margin_simulator or MarginSimulator(
            instrument_master,
            penalty_ticks=penalty_ticks,
            penalty_fee_per_contract=penalty_fee_per_contract,
        )
        self._broker = broker
        self._initial_equity = float(initial_equity)
        self._schedule = rebalance_schedule or (
            RebalanceSchedule.every_n(rebalance_every)
            if rebalance_every != 1
            else RebalanceSchedule.daily()
        )
        self._state_builder = state_builder
        self._use_bag_rolls = use_bag_rolls
        self._annualization_factor = annualization_factor
        self._timeframe = _canonical_engine_timeframe(timeframe)

    # -- public runner ---------------------------------------------------------

    def run(
        self,
        *,
        bars: pl.DataFrame | None = None,
        states: Sequence[MarketState] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        run_id: str | None = None,
    ) -> BacktestResult:
        """Run the event-driven simulation and return an attributed result."""
        context = _prepare_bars(bars) if bars is not None else None
        state_list = _resolve_states(
            bars,
            states,
            self._state_builder,
            self._initial_equity,
            start=start,
            end=end,
            timeframe=self._timeframe,
        )
        if not state_list:
            raise ValueError("no market states available for the requested window")
        # Intraday bars settle once per UTC calendar day; daily/coarser bars keep
        # the historical per-bar settlement behaviour (1 bar == 1 session).
        intraday_bars = _states_are_intraday(state_list)
        last_settlement_day = None
        broker = self._broker or SimulatedBroker(
            self._instrument_master,
            self._cost_model,
            margin_simulator=self._margin_simulator,
            initial_cash=self._initial_equity,
        )
        broker.set_market(
            state_list[0].timestamp,
            _quotes_for_state(state_list[0], context),
        )
        attribution = _Attribution()
        fill_cursor = len(broker.fills)
        equity_rows: list[dict[str, object]] = []
        weight_rows: list[dict[str, object]] = []
        trades: list[TradeRecord] = []
        roll_events: list[RollExecution] = []
        liquidation_events: list[ForcedLiquidationEvent] = []
        warnings: list[str] = []
        settlement_pnl = 0.0
        last_contract_marks: dict[tuple[str, date | None], float] = {}
        previous_marks: dict[str, float] = {}
        previous_equity = float(self._initial_equity)
        peak_equity = float(self._initial_equity)
        previous_timestamp: datetime | None = None
        roll_pnl_total = 0.0
        equity = previous_equity
        order_counter = 0
        depleted = False
        for index, state in enumerate(state_list):
            timestamp = state.timestamp
            self._margin_simulator.set_volatilities(_state_volatilities(state))
            quotes = _extend_quotes(
                _quotes_for_state(state, context),
                state,
                self._instrument_master,
                extra_roots=tuple(broker.get_positions_sync()),
            )
            broker.set_market(timestamp, quotes)
            close_marks = {root: quote.close for root, quote in quotes.items()}
            costs_before_bar = attribution.total_costs
            commissions_before_bar = attribution.commissions
            fees_before_bar = attribution.fees
            slippage_before_bar = attribution.slippage
            funding_before_bar = attribution.funding
            turnover_before_bar = attribution.turnover
            roll_before_bar = roll_pnl_total
            equity_before_bar = previous_equity
            # 1) Daily mark-to-market settlement at the prices of the contracts
            #    that were actually held (the roll has not happened yet). For
            #    intraday bars this is gated to the first bar of each UTC day.
            settle_due = True
            if intraday_bars:
                current_day = utc_calendar_day(timestamp)
                settle_due = last_settlement_day is None or current_day > last_settlement_day
                if settle_due:
                    last_settlement_day = current_day
            positions = broker.get_positions_sync()
            if positions and settle_due:
                pre_marks = {
                    root: _contract_mark(
                        context,
                        timestamp,
                        root,
                        position.expiration,
                        last_contract_marks,
                        fallback=close_marks.get(root, position.market_price),
                    )
                    for root, position in positions.items()
                }
                settlement = broker.settle(pre_marks)
                settlement_pnl += settlement.settlement_pnl
            # 2) Roll every contract whose active expiration changed.
            for root, position in sorted(broker.get_positions_sync().items()):
                if position.quantity == 0:
                    continue
                snapshot = state.instruments.get(root)
                active_expiration = (
                    snapshot.expiration
                    if snapshot is not None
                    else _sibling_expiration(state, root, self._instrument_master)
                )
                if active_expiration is None:
                    active = context.active[timestamp].get(root) if context is not None else None
                    active_expiration = active.expiration if active is not None else None
                if active_expiration is None or active_expiration == position.expiration:
                    continue
                new_price = close_marks.get(root, position.market_price)
                old_price = _contract_mark(
                    context,
                    timestamp,
                    root,
                    position.expiration,
                    last_contract_marks,
                    fallback=previous_marks.get(root, position.market_price),
                )
                roll_pnl = -position.quantity * (new_price - old_price) * position.multiplier
                roll_pnl_total += roll_pnl
                broker.roll_position(
                    root,
                    from_price=old_price,
                    to_price=new_price,
                    to_expiration=active_expiration,
                    to_contract_symbol=_contract_symbol(root, active_expiration),
                    use_bag=self._use_bag_rolls,
                )
                new_fills = broker.fills[fill_cursor:]
                fill_cursor = len(broker.fills)
                for fill in new_fills:
                    attribution.add_fill(fill)
                roll_events.append(
                    RollExecution(
                        timestamp=timestamp,
                        instrument_id=root,
                        quantity=abs(position.quantity),
                        from_expiration=position.expiration,
                        to_expiration=active_expiration,
                        from_price=old_price,
                        to_price=new_price,
                        roll_pnl=roll_pnl,
                        total_cost=sum(
                            fill.cost.spread_cost
                            + fill.cost.slippage_cost
                            + fill.cost.commission
                            + fill.cost.fees
                            for fill in new_fills
                        ),
                        bag_used=self._use_bag_rolls,
                    )
                )
            # 3) Maintenance-margin check on the settled/rolled book.
            liquidation = self._force_liquidation_if_needed(broker, timestamp)
            if liquidation is not None:
                liquidation_events.append(liquidation)
                new_fills = broker.fills[fill_cursor:]
                for fill in new_fills:
                    attribution.add_fill(fill)
                trades.extend(_trade_from_fill(fill) for fill in new_fills)
                fill_cursor = len(broker.fills)
            # 4) Rebalance the portfolio through the broker.
            equity_before_rebalance = broker.net_liquidation
            if (
                liquidation is None
                and self._schedule.should_rebalance(index, timestamp, previous_timestamp)
                and equity_before_rebalance > 0.0
            ):
                targeted_state = replace(state, equity=equity_before_rebalance)
                pipeline_result = self._strategy.run(targeted_state)
                _record_weights(weight_rows, pipeline_result)
                try:
                    if targeted_state.account_currency == "USD" and not targeted_state.fx_rates:
                        targets = self._translator.translate(
                            pipeline_result.final_weights,
                            targeted_state,
                            equity=equity_before_rebalance,
                        )
                    else:
                        targets = self._translator.translate(
                            pipeline_result.final_weights,
                            targeted_state,
                            equity=equity_before_rebalance,
                            account_currency=targeted_state.account_currency,
                            fx_rates=targeted_state.fx_rates,
                        )
                except Exception as exc:  # pragma: no cover - defensive translator guard
                    warnings.append(f"translator failed at {timestamp.isoformat()}: {exc}")
                    targets = ContractTargetSet.empty()
                # The translator may route residual exposure into micro roots that
                # the MarketState does not carry (ES -> MES, ...). Synthesize their
                # quotes from the same-underlying standard contract so the exact
                # live target set is executable in the backtest too.
                if len(targets):
                    quotes = _extend_quotes(
                        quotes,
                        targeted_state,
                        self._instrument_master,
                        extra_roots=tuple(targets.quantities),
                    )
                    broker.set_market(timestamp, quotes)
                    close_marks = {root: quote.close for root, quote in quotes.items()}
                order_counter, new_trades = self._execute_targets(
                    broker,
                    targets,
                    timestamp,
                    close_marks,
                    order_counter,
                    warnings,
                )
                trades.extend(new_trades)
                for fill in broker.fills[fill_cursor:]:
                    attribution.add_fill(fill)
                fill_cursor = len(broker.fills)
            # 5) End-of-bar settlement at the mark: realises embedded execution
            #    costs and any intraday move into cash. Intraday bars must not
            #    settle again within the same UTC day (one settlement per day).
            positions = broker.get_positions_sync()
            if positions and not intraday_bars:
                settlement = broker.settle(close_marks)
                settlement_pnl += settlement.settlement_pnl
            # 6) Post-trade margin call (e.g. after a large rebalance).
            liquidation = self._force_liquidation_if_needed(broker, timestamp)
            if liquidation is not None:
                liquidation_events.append(liquidation)
                new_fills = broker.fills[fill_cursor:]
                for fill in new_fills:
                    attribution.add_fill(fill)
                trades.extend(_trade_from_fill(fill) for fill in new_fills)
                fill_cursor = len(broker.fills)
                positions = broker.get_positions_sync()
                if positions and not intraday_bars:
                    settlement = broker.settle(close_marks)
                    settlement_pnl += settlement.settlement_pnl
            equity = broker.net_liquidation
            if equity <= 0.0:
                warnings.append(
                    f"account depleted at {timestamp.isoformat()} (equity {equity:.2f}); "
                    "backtest truncated"
                )
                depleted = True
                equity = 0.0
            costs_this_bar = attribution.total_costs - costs_before_bar
            roll_this_bar = roll_pnl_total - roll_before_bar
            price_pnl_bar = equity - equity_before_bar + costs_this_bar - roll_this_bar
            if depleted:
                equity_rows.append(
                    {
                        TIMESTAMP_COLUMN: timestamp,
                        "equity": 0.0,
                        "cash": broker.cash,
                        "unrealized_pnl": 0.0,
                        "price_pnl": price_pnl_bar,
                        "roll_pnl": roll_this_bar,
                        "commissions": attribution.commissions - commissions_before_bar,
                        "fees": attribution.fees - fees_before_bar,
                        "slippage": attribution.slippage - slippage_before_bar,
                        "funding": attribution.funding - funding_before_bar,
                        "margin_used": 0.0,
                        "margin_utilization": 0.0,
                        "gross_exposure": 0.0,
                        "net_exposure": 0.0,
                        "drawdown": -1.0,
                        "turnover": attribution.turnover - turnover_before_bar,
                        "n_positions": 0,
                    }
                )
                break
            margin_used, margin_utilization = _margin_utilisation(equity, broker)
            gross, net = _exposures(broker.get_positions_sync())
            peak_equity = max(peak_equity, equity)
            equity_rows.append(
                {
                    TIMESTAMP_COLUMN: timestamp,
                    "equity": equity,
                    "cash": broker.cash,
                    "unrealized_pnl": equity - broker.cash,
                    "price_pnl": price_pnl_bar,
                    "roll_pnl": roll_this_bar,
                    "commissions": attribution.commissions - commissions_before_bar,
                    "fees": attribution.fees - fees_before_bar,
                    "slippage": attribution.slippage - slippage_before_bar,
                    "funding": attribution.funding - funding_before_bar,
                    "margin_used": margin_used,
                    "margin_utilization": margin_utilization,
                    "gross_exposure": gross,
                    "net_exposure": net,
                    "drawdown": equity / peak_equity - 1.0,
                    "turnover": attribution.turnover - turnover_before_bar,
                    "n_positions": sum(
                        1
                        for position in broker.get_positions_sync().values()
                        if position.quantity != 0
                    ),
                }
            )
            previous_equity = equity
            previous_timestamp = timestamp
            previous_marks = dict(close_marks)
            if context is not None:
                for row in context.rows_by_timestamp.get(timestamp, ()):
                    last_contract_marks[(row.root, row.expiration)] = row.close
        ending_equity = 0.0 if depleted else broker.net_liquidation
        price_pnl_total = (
            ending_equity - self._initial_equity + attribution.total_costs - roll_pnl_total
        )
        attribution.price_pnl = price_pnl_total
        attribution.roll_pnl = roll_pnl_total
        result = _build_result(
            run_id=run_id or _default_run_id(self._strategy.name, BacktestMode.EVENT, state_list),
            strategy=self._strategy,
            mode=BacktestMode.EVENT,
            state_list=state_list,
            initial_equity=self._initial_equity,
            ending_equity=ending_equity,
            equity_rows=equity_rows,
            weight_rows=weight_rows,
            trades=trades,
            fills=broker.fills,
            roll_events=roll_events,
            liquidations=liquidation_events,
            attribution=attribution,
            settlement_pnl=settlement_pnl,
            margin_requirements=self._margin_simulator.last_requirements,
            metadata={
                "broker": broker.name,
                "cost_model": json.dumps(self._cost_model.to_dict(), sort_keys=True),
                "margin_simulator": json.dumps(self._margin_simulator.to_dict(), sort_keys=True),
                "rebalance_schedule": json.dumps(self._schedule.to_dict(), sort_keys=True),
                "use_bag_rolls": self._use_bag_rolls,
                "annualization_factor": self._annualization_factor,
                "timeframe": self._timeframe,
                "margin_regime": json.dumps(
                    self._margin_simulator.last_requirements.to_dict(), sort_keys=True
                ),
                "mode": BacktestMode.EVENT.value,
            },
            warnings=warnings,
        )
        result.check_accounting_identity()
        return result

    def _force_liquidation_if_needed(
        self,
        broker: SimulatedBroker,
        timestamp: datetime,
    ) -> ForcedLiquidationEvent | None:
        """Flatten the book when equity falls below the maintenance margin."""
        positions = broker.get_positions_sync()
        if not any(position.quantity != 0 for position in positions.values()):
            return None
        equity = broker.net_liquidation
        if equity <= 0.0:
            return None
        margin_call = self._margin_simulator.check_margin_call(
            timestamp, equity=equity, positions=positions
        )
        if not margin_call.triggered:
            return None
        plan = self._margin_simulator.forced_liquidation_plan(margin_call)
        closed = tuple(position.instrument_id for position in plan.positions)
        cursor = len(broker.fills)
        broker.flatten_all_sync(
            kind=TradeKind.FORCED_LIQUIDATION,
            extra_slippage_ticks=self._margin_simulator.penalty_ticks,
            extra_fee_per_contract=self._margin_simulator.penalty_fee_per_contract,
            tag="margin_call",
        )
        penalty_cost = sum(
            fill.cost.commission + fill.cost.fees + fill.cost.spread_cost + fill.cost.slippage_cost
            for fill in broker.fills[cursor:]
        )
        return ForcedLiquidationEvent(
            timestamp=timestamp,
            equity_before=equity,
            maintenance_margin=margin_call.maintenance_margin_used,
            equity_after=broker.net_liquidation,
            positions_closed=closed,
            penalty_cost=penalty_cost,
            reason=margin_call.reason,
        )

    def _execute_targets(
        self,
        broker: SimulatedBroker,
        targets: ContractTargetSet,
        timestamp: datetime,
        close_marks: Mapping[str, float],
        order_counter: int,
        warnings: list[str],
    ) -> tuple[int, list[TradeRecord]]:
        target_quantities: dict[str, ContractTarget] = {
            target.instrument_id: target for target in targets
        }
        positions = broker.get_positions_sync()
        roots = sorted(set(target_quantities) | set(positions))
        trades: list[TradeRecord] = []
        for root in roots:
            current = positions[root].quantity if root in positions else 0
            target_quantity = target_quantities[root].quantity if root in target_quantities else 0
            delta = target_quantity - current
            if delta == 0:
                continue
            order_counter += 1
            reference = close_marks.get(root)
            if reference is None:
                position = positions.get(root)
                reference = position.market_price if position is not None else None
            if reference is None:
                warnings.append(f"no price for {root}; skipping rebalance order")
                continue
            target = target_quantities.get(root)
            order = OrderEvent.market(
                order_id=f"bt-{order_counter:06d}",
                timestamp=timestamp,
                instrument_id=root,
                side=OrderSide.BUY if delta > 0 else OrderSide.SELL,
                quantity=abs(delta),
                expiration=target.expiration if target is not None else None,
                contract_symbol=target.contract_symbol if target is not None else None,
                tag="rebalance",
            )
            if current == 0:
                kind = TradeKind.ENTRY
            elif target_quantity == 0:
                kind = TradeKind.EXIT
            else:
                kind = TradeKind.REBALANCE
            receipt = broker.place_order(order, kind=kind)
            if receipt.status.value in {"REJECTED", "SUBMITTED"}:
                warnings.append(
                    f"order for {root} {delta:+d} at {timestamp.isoformat()} "
                    f"not filled: {receipt.message}"
                )
                continue
            fill = broker.fills[-1]
            trades.append(_trade_from_fill(fill))
        return order_counter, trades


class BacktestEngine:
    """Unified backtest runner exposing vectorised and event-driven modes."""

    def __init__(
        self,
        strategy: StrategyPipeline,
        translator: ContractTranslatorProtocol,
        *,
        mode: BacktestMode = BacktestMode.EVENT,
        instrument_master: InstrumentsConfig | None = None,
        cost_model: CostModel | None = None,
        margin_simulator: MarginSimulator | None = None,
        broker: SimulatedBroker | None = None,
        initial_equity: float = 1_000_000.0,
        rebalance_schedule: RebalanceSchedule | None = None,
        rebalance_every: int = 1,
        state_builder: StateBuilder | None = None,
        penalty_ticks: float = 2.0,
        penalty_fee_per_contract: float = 15.0,
        use_bag_rolls: bool = False,
        annualization_factor: int = TRADING_DAYS_PER_YEAR,
        timeframe: str = "1d",
    ) -> None:
        if not isinstance(mode, BacktestMode):
            raise TypeError("mode must be a BacktestMode")
        self._mode = mode
        self._strategy = strategy
        self._translator = translator
        self._instrument_master = instrument_master
        self._cost_model = cost_model or CostModel(instrument_master=instrument_master)
        self._margin_simulator = margin_simulator or MarginSimulator(
            instrument_master,
            penalty_ticks=penalty_ticks,
            penalty_fee_per_contract=penalty_fee_per_contract,
        )
        self._initial_equity = float(initial_equity)
        self._rebalance_schedule = rebalance_schedule
        self._rebalance_every = rebalance_every
        self._state_builder = state_builder
        self._use_bag_rolls = use_bag_rolls
        self._annualization_factor = annualization_factor
        self._timeframe = _canonical_engine_timeframe(timeframe)
        self._event = EventBacktester(
            strategy,
            translator,
            instrument_master=instrument_master,
            cost_model=self._cost_model,
            margin_simulator=self._margin_simulator,
            broker=broker,
            initial_equity=initial_equity,
            rebalance_schedule=rebalance_schedule,
            rebalance_every=rebalance_every,
            state_builder=state_builder,
            penalty_ticks=penalty_ticks,
            penalty_fee_per_contract=penalty_fee_per_contract,
            use_bag_rolls=use_bag_rolls,
            annualization_factor=annualization_factor,
            timeframe=self._timeframe,
        )

    @property
    def mode(self) -> BacktestMode:
        """Configured execution mode."""
        return self._mode

    def run(
        self,
        *,
        bars: pl.DataFrame | None = None,
        states: Sequence[MarketState] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        run_id: str | None = None,
    ) -> BacktestResult:
        """Run the configured backtest mode and return a typed result."""
        if self._mode is BacktestMode.EVENT:
            return self._event.run(bars=bars, states=states, start=start, end=end, run_id=run_id)
        return self.run_vectorized(bars=bars, states=states, start=start, end=end, run_id=run_id)

    def run_vectorized(
        self,
        *,
        bars: pl.DataFrame | None = None,
        states: Sequence[MarketState] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        run_id: str | None = None,
    ) -> BacktestResult:
        """Run the fast vectorised weights -> returns -> equity mode."""
        context = _prepare_bars(bars) if bars is not None else None
        state_list = _resolve_states(
            bars,
            states,
            self._state_builder,
            self._initial_equity,
            start=start,
            end=end,
            timeframe=self._timeframe,
        )
        if not state_list:
            raise ValueError("no market states available for the requested window")
        schedule = self._rebalance_schedule or (
            RebalanceSchedule.every_n(self._rebalance_every)
            if self._rebalance_every != 1
            else RebalanceSchedule.daily()
        )
        equity = float(self._initial_equity)
        previous_equity = equity
        weights: WeightVector = WeightVector.empty()
        contracts: dict[str, float] = {}
        previous_marks: dict[str, float] = {}
        previous_expirations: dict[str, date | None] = {}
        last_contract_marks: dict[tuple[str, date | None], float] = {}
        attribution = _Attribution()
        equity_rows: list[dict[str, object]] = []
        weight_rows: list[dict[str, object]] = []
        trades: list[TradeRecord] = []
        roll_events: list[RollExecution] = []
        warnings: list[str] = []
        roll_pnl_total = 0.0
        price_pnl_total = 0.0
        peak_equity = equity
        previous_timestamp: datetime | None = None
        order_counter = 0
        for index, state in enumerate(state_list):
            timestamp = state.timestamp
            self._margin_simulator.set_volatilities(_state_volatilities(state))
            marks = _marks_for_state(state)
            multipliers = _multipliers_for_roots(tuple(sorted(marks)), self._instrument_master)
            # Mark the existing book to market on the continuous front prices.
            price_pnl_bar = 0.0
            for root, weight in weights.active.items():
                previous_mark = previous_marks.get(root)
                if previous_mark is None or previous_mark <= 0.0:
                    continue
                price_pnl_bar += previous_equity * weight * (marks[root] / previous_mark - 1.0)
            # Roll attribution for the implied contract book.
            roll_pnl_bar = 0.0
            for root in sorted(contracts):
                active_expiration = _active_expiration(
                    state, context, root, timestamp, self._instrument_master
                )
                old_expiration = previous_expirations.get(root)
                if active_expiration is None or active_expiration == old_expiration:
                    continue
                new_price = marks[root]
                old_price = _contract_mark(
                    context,
                    timestamp,
                    root,
                    old_expiration,
                    last_contract_marks,
                    fallback=previous_marks.get(root, new_price),
                )
                quantity = contracts[root]
                roll_pnl = -quantity * (new_price - old_price) * multipliers.get(root, 1.0)
                roll_pnl_bar += roll_pnl
                roll_pnl_total += roll_pnl
                roll_events.append(
                    RollExecution(
                        timestamp=timestamp,
                        instrument_id=root,
                        quantity=max(1, round(abs(quantity))),
                        from_expiration=old_expiration,
                        to_expiration=active_expiration,
                        from_price=old_price,
                        to_price=new_price,
                        roll_pnl=roll_pnl,
                        total_cost=0.0,
                        bag_used=False,
                    )
                )
            equity_after_price = equity + price_pnl_bar + roll_pnl_bar
            # Rebalance on schedule.
            costs_bar = 0.0
            commissions_bar = 0.0
            fees_bar = 0.0
            slippage_bar = 0.0
            turnover_bar = 0.0
            if equity_after_price > 0.0 and schedule.should_rebalance(
                index, timestamp, previous_timestamp
            ):
                targeted_state = replace(state, equity=equity_after_price)
                pipeline_result = self._strategy.run(targeted_state)
                _record_weights(weight_rows, pipeline_result)
                new_weights = pipeline_result.final_weights
                roots = sorted(set(new_weights.weights) | set(weights.weights) | set(marks))
                for root in roots:
                    if root not in marks:
                        warnings.append(f"no price for {root}; weight ignored")
                        continue
                    previous_notional = weights.get(root) * equity_after_price
                    new_notional = new_weights.get(root) * equity_after_price
                    delta_notional = new_notional - previous_notional
                    if abs(delta_notional) < 1e-9:
                        continue
                    multiplier = multipliers.get(root, 1.0)
                    delta_contracts = delta_notional / (marks[root] * multiplier)
                    quantity = math.floor(abs(delta_contracts) + 0.5)
                    if quantity <= 0:
                        continue
                    order_counter += 1
                    tick_size = _tick_size(root, self._instrument_master)
                    cost = self._cost_model.transaction_cost(
                        root,
                        quantity,
                        marks[root],
                        tick_size,
                        multiplier,
                        _state_volume(state, root),
                    )
                    costs_bar += cost.total
                    commissions_bar += cost.commission
                    fees_bar += cost.fees
                    slippage_bar += cost.spread_cost + cost.slippage_cost
                    turnover_bar += abs(delta_notional)
                    side = OrderSide.BUY if delta_contracts > 0 else OrderSide.SELL
                    fill_price = (
                        marks[root] + side.multiplier * cost.execution_cost_ticks * tick_size
                    )
                    trades.append(
                        TradeRecord(
                            timestamp=timestamp,
                            order_id=f"vec-{order_counter:06d}",
                            instrument_id=root,
                            side=side,
                            quantity=quantity,
                            price=fill_price,
                            reference_price=marks[root],
                            kind=(
                                TradeKind.ENTRY.value
                                if weights.get(root) == 0.0
                                else TradeKind.REBALANCE.value
                            ),
                            notional=quantity * marks[root] * multiplier,
                            commissions=cost.commission + cost.fees,
                            slippage=cost.spread_cost + cost.slippage_cost,
                            execution_cost_ticks=cost.execution_cost_ticks,
                        )
                    )
                weights = new_weights
                contracts = {
                    root: (
                        weights.get(root)
                        * equity_after_price
                        / (marks[root] * multipliers.get(root, 1.0))
                    )
                    for root in weights.weights
                    if root in marks
                }
            equity = equity_after_price - costs_bar
            attribution.price_pnl = price_pnl_total + price_pnl_bar
            price_pnl_total += price_pnl_bar
            attribution.commissions += commissions_bar
            attribution.fees += fees_bar
            attribution.slippage += slippage_bar
            attribution.turnover += turnover_bar
            attribution.roll_pnl = roll_pnl_total
            peak_equity = max(peak_equity, equity) if equity > 0.0 else peak_equity
            margin_used = _vectorized_margin(
                contracts,
                marks,
                multipliers,
                self._instrument_master,
                self._margin_simulator,
            )
            gross = sum(abs(weight) * equity for weight in weights.weights.values())
            net = sum(weight * equity for weight in weights.weights.values())
            equity_rows.append(
                {
                    TIMESTAMP_COLUMN: timestamp,
                    "equity": equity,
                    "cash": equity,
                    "unrealized_pnl": 0.0,
                    "price_pnl": price_pnl_bar,
                    "roll_pnl": roll_pnl_bar,
                    "commissions": commissions_bar,
                    "fees": fees_bar,
                    "slippage": slippage_bar,
                    "funding": 0.0,
                    "margin_used": margin_used,
                    "margin_utilization": margin_used / equity if equity > 0.0 else 0.0,
                    "gross_exposure": gross,
                    "net_exposure": net,
                    "drawdown": equity / peak_equity - 1.0 if peak_equity > 0.0 else 0.0,
                    "turnover": turnover_bar,
                    "n_positions": sum(
                        1 for contract in contracts.values() if abs(contract) > 1e-9
                    ),
                }
            )
            previous_equity = equity
            previous_marks = dict(marks)
            previous_expirations = {
                root: _active_expiration(state, context, root, timestamp, self._instrument_master)
                for root in marks
            }
            previous_timestamp = timestamp
            if context is not None:
                for row in context.rows_by_timestamp.get(timestamp, ()):
                    last_contract_marks[(row.root, row.expiration)] = row.close
            if equity <= 0.0:
                warnings.append(f"account depleted at {timestamp.isoformat()}")
                break
        ending_equity = max(equity, 0.0)
        attribution.price_pnl = (
            ending_equity - self._initial_equity + attribution.total_costs - roll_pnl_total
        )
        attribution.roll_pnl = roll_pnl_total
        result = _build_result(
            run_id=run_id
            or _default_run_id(self._strategy.name, BacktestMode.VECTORIZED, state_list),
            strategy=self._strategy,
            mode=BacktestMode.VECTORIZED,
            state_list=state_list,
            initial_equity=self._initial_equity,
            ending_equity=ending_equity,
            equity_rows=equity_rows,
            weight_rows=weight_rows,
            trades=trades,
            fills=(),
            roll_events=roll_events,
            liquidations=[],
            attribution=attribution,
            settlement_pnl=0.0,
            margin_requirements=self._margin_simulator.last_requirements,
            metadata={
                "cost_model": json.dumps(self._cost_model.to_dict(), sort_keys=True),
                "rebalance_schedule": json.dumps(schedule.to_dict(), sort_keys=True),
                "annualization_factor": self._annualization_factor,
                "timeframe": self._timeframe,
                "margin_regime": json.dumps(
                    self._margin_simulator.last_requirements.to_dict(), sort_keys=True
                ),
                "mode": BacktestMode.VECTORIZED.value,
            },
            warnings=warnings,
        )
        result.check_accounting_identity()
        return result


# ---------------------------------------------------------------------------
# Shared result construction / helpers
# ---------------------------------------------------------------------------


def _record_weights(rows: list[dict[str, object]], result: PipelineResult) -> None:
    for stage, vector in (
        ("base", result.base_weights),
        ("timing", result.timing_weights),
        ("risk", result.risk_weights),
        ("final", result.final_weights),
    ):
        for instrument_id, weight in vector.weights.items():
            rows.append(
                {
                    TIMESTAMP_COLUMN: result.timestamp,
                    "stage": stage,
                    "instrument_id": instrument_id,
                    "weight": float(weight),
                }
            )


def _trade_from_fill(fill: FillRecord) -> TradeRecord:
    return TradeRecord(
        timestamp=fill.timestamp,
        order_id=fill.order_id,
        instrument_id=fill.instrument_id,
        side=fill.side,
        quantity=fill.quantity,
        price=fill.price,
        reference_price=fill.reference_price,
        kind=fill.kind.value,
        notional=fill.notional,
        commissions=fill.commissions,
        slippage=fill.slippage,
        execution_cost_ticks=fill.execution_cost_ticks,
    )


def _margin_utilisation(equity: float, broker: SimulatedBroker) -> tuple[float, float]:
    if equity <= 0.0:
        return 0.0, 0.0
    initial_margin, _ = broker.margin_requirements()
    return initial_margin, initial_margin / equity


def _exposures(positions: Mapping[InstrumentId, Position]) -> tuple[float, float]:
    gross = 0.0
    net = 0.0
    for position in positions.values():
        notional = position.quantity * position.market_price * position.multiplier
        gross += abs(notional)
        net += notional
    return gross, net


def _bar_for_state(context: _BarContext | None, state: MarketState, root: str) -> _BarRow | None:
    """Bar matching the state's active contract, falling back to the front bar."""
    if context is None:
        return None
    rows = context.rows_by_timestamp.get(state.timestamp, ())
    snapshot = state.instruments.get(root)
    expiration = snapshot.expiration if snapshot is not None else None
    if expiration is not None:
        for row in rows:
            if row.root == root and row.expiration == expiration:
                return row
    return context.active.get(state.timestamp, {}).get(root)


def _quotes_for_state(
    state: MarketState,
    context: _BarContext | None,
) -> dict[str, BarQuote]:
    quotes: dict[str, BarQuote] = {}
    for root, snapshot in state.instruments.items():
        bar = _bar_for_state(context, state, root)
        if bar is not None:
            quote = BarQuote(
                root=root,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
                bid=snapshot.bid,
                ask=snapshot.ask,
                expiration=bar.expiration,
                contract_symbol=_contract_symbol(root, bar.expiration),
                settlement=bar.settlement or bar.close,
            )
        else:
            quote = BarQuote(
                root=root,
                open=snapshot.price,
                high=snapshot.price,
                low=snapshot.price,
                close=snapshot.price,
                volume=snapshot.volume,
                bid=snapshot.bid,
                ask=snapshot.ask,
                expiration=snapshot.expiration,
                contract_symbol=_contract_symbol(root, snapshot.expiration),
                settlement=snapshot.settlement or snapshot.price,
            )
        quotes[root] = quote
    return quotes


def _sibling_root(root: str, instrument_master: InstrumentsConfig | None) -> str | None:
    """Return the standard/micro sibling root of ``root`` from the instrument master."""
    if instrument_master is None or not instrument_master.has(root):
        return None
    spec = instrument_master.get(root)
    if spec.standard_root:
        return spec.standard_root
    if spec.micro_root:
        return spec.micro_root
    return None


def _sibling_expiration(
    state: MarketState, root: str, instrument_master: InstrumentsConfig | None
) -> date | None:
    """Expiration of ``root`` inherited from its standard/micro sibling snapshot."""
    sibling = _sibling_root(root, instrument_master)
    if sibling is None or sibling not in state.instruments:
        return None
    return state.instruments[sibling].expiration


def _synthesized_quote(
    root: str,
    instrument_master: InstrumentsConfig | None,
    quotes: Mapping[str, BarQuote],
    *,
    expiration_hint: date | None = None,
) -> BarQuote | None:
    """Derive a micro/standard sibling quote from the same-underlying contract.

    The ContractTranslator deliberately routes fractional standard exposure into
    the paired micro contract, and it prices the micro leg from the standard
    snapshot. The backtester therefore synthesizes the missing leg's bar quote
    from its sibling (price is the same underlying, tick/multiplier/margins come
    from the instrument master) so the exact live target set is executable. The
    function is deterministic and never fabricates a price without a sibling.
    """
    if instrument_master is None or not instrument_master.has(root):
        return None
    sibling = _sibling_root(root, instrument_master)
    if sibling is None:
        return None
    sibling_quote = quotes.get(sibling)
    if sibling_quote is None:
        return None
    spec = instrument_master.get(root)
    half_spread = 0.5 * spec.tick_size
    expiration = expiration_hint if expiration_hint is not None else sibling_quote.expiration
    price = sibling_quote.close
    return BarQuote(
        root=root,
        open=price,
        high=price,
        low=price,
        close=price,
        volume=sibling_quote.volume,
        bid=max(price - half_spread, spec.tick_size * 0.5),
        ask=price + half_spread,
        expiration=expiration,
        contract_symbol=_contract_symbol(root, expiration),
        settlement=price,
    )


def _extend_quotes(
    quotes: Mapping[str, BarQuote],
    state: MarketState,
    instrument_master: InstrumentsConfig | None,
    *,
    extra_roots: Sequence[str] = (),
) -> dict[str, BarQuote]:
    """Add deterministic micro/standard sibling quotes for ``extra_roots``."""
    extended = dict(quotes)
    for root in extra_roots:
        normalized = root.strip().upper()
        if normalized in extended:
            continue
        sibling_expiration = _sibling_expiration(state, normalized, instrument_master)
        quote = _synthesized_quote(
            normalized,
            instrument_master,
            extended,
            expiration_hint=sibling_expiration,
        )
        if quote is not None:
            extended[normalized] = quote
    return extended


def _marks_for_state(state: MarketState) -> dict[str, float]:
    return {root: snapshot.price for root, snapshot in state.instruments.items()}


def _state_volume(state: MarketState, root: str) -> float:
    snapshot = state.instruments.get(root)
    return snapshot.volume if snapshot is not None else 0.0


def _active_expiration(
    state: MarketState,
    context: _BarContext | None,
    root: str,
    timestamp: datetime,
    instrument_master: InstrumentsConfig | None = None,
) -> date | None:
    snapshot = state.instruments.get(root)
    if snapshot is not None and snapshot.expiration is not None:
        return snapshot.expiration
    sibling_expiration = _sibling_expiration(state, root, instrument_master)
    if sibling_expiration is not None:
        return sibling_expiration
    if context is not None:
        bar = context.active.get(timestamp, {}).get(root)
        if bar is not None:
            return bar.expiration
    return snapshot.expiration if snapshot is not None else None


def _contract_mark(
    context: _BarContext | None,
    timestamp: datetime,
    root: str,
    expiration: date | None,
    marks: Mapping[tuple[str, date | None], float],
    *,
    fallback: float,
) -> float:
    if context is not None:
        price = context.contract_price(timestamp, root, expiration)
        if price is not None and price > 0.0:
            return price
    return marks.get((root, expiration), fallback)


def _tick_size(root: str, instrument_master: InstrumentsConfig | None) -> float:
    if instrument_master is not None and instrument_master.has(root):
        return float(instrument_master.get(root).tick_size)
    return 0.01


def _multipliers_for_roots(
    roots: Sequence[str], instrument_master: InstrumentsConfig | None
) -> dict[str, float]:
    multipliers: dict[str, float] = {}
    for root in roots:
        if instrument_master is not None and instrument_master.has(root):
            multipliers[root] = float(instrument_master.get(root).multiplier)
        else:
            multipliers[root] = 1.0
    return multipliers


def _vectorized_margin(
    contracts: Mapping[str, float],
    marks: Mapping[str, float],
    multipliers: Mapping[str, float],
    instrument_master: InstrumentsConfig | None,
    margin_simulator: MarginSimulator | None = None,
) -> float:
    margin = 0.0
    for root, quantity in contracts.items():
        if root not in marks or root not in multipliers:
            continue
        if margin_simulator is not None and (
            margin_simulator.calibrated_margins or instrument_master is not None
        ):
            initial, _maintenance = margin_simulator.effective_margin(root, 1)
            if initial > 0.0:
                margin += abs(quantity) * initial
                continue
        if instrument_master is not None and instrument_master.has(root):
            margin += abs(quantity) * float(instrument_master.get(root).margins.initial)
        else:
            margin += abs(quantity) * marks[root] * multipliers[root] * 0.10
    return margin


def _resolve_states(
    bars: pl.DataFrame | None,
    states: Sequence[MarketState] | None,
    state_builder: StateBuilder | None,
    initial_equity: float,
    *,
    start: datetime | None,
    end: datetime | None,
    timeframe: str = "1d",
) -> list[MarketState]:
    start_utc = ensure_utc(start) if start is not None else None
    end_utc = ensure_utc(end) if end is not None else None
    if states is not None:
        selected = list(states)
        for state in selected:
            if not isinstance(state, MarketState):
                raise TypeError("states must contain MarketState instances")
        timestamps = [state.timestamp for state in selected]
        if any(later <= earlier for earlier, later in pairwise(timestamps)):
            raise ValueError("states must be in strictly increasing timestamp order")
        selected = [
            state if state.timeframe == timeframe else replace(state, timeframe=timeframe)
            for state in selected
        ]
        return [
            state
            for state in selected
            if (start_utc is None or state.timestamp >= start_utc)
            and (end_utc is None or state.timestamp <= end_utc)
        ]
    if bars is None:
        raise ValueError("either bars or states must be provided")
    frame = _prepare_bars(bars).frame
    if state_builder is not None:
        context = _prepare_bars(frame)
        built: list[MarketState] = []
        for timestamp in context.timestamps:
            state = state_builder(frame, timestamp, initial_equity)
            if start_utc is not None and state.timestamp < start_utc:
                continue
            if end_utc is not None and state.timestamp > end_utc:
                continue
            built.append(state)
        return built
    built = build_states_from_bars(frame, None, initial_equity=initial_equity, timeframe=timeframe)
    return [
        state
        for state in built
        if (start_utc is None or state.timestamp >= start_utc)
        and (end_utc is None or state.timestamp <= end_utc)
    ]


def _state_volatilities(state: MarketState) -> dict[str, float]:
    """Point-in-time annualised per-root volatilities from a MarketState."""
    return {root: state.annualized_vol(root) for root in state.ids}


def _canonical_engine_timeframe(timeframe: str) -> str:
    """Validate/canonicalise a bar timeframe via the data registry."""
    from src.data.timeframes import get_timeframe

    return get_timeframe(timeframe).name


def _states_are_intraday(states: Sequence[MarketState]) -> bool:
    """Whether the state sequence has a sub-daily median bar spacing.

    Used to decide whether daily mark-to-market settlement must be gated to UTC
    calendar-day boundaries (intraday bars) or may keep the legacy every-bar
    settlement of daily backtests. Purely timestamp based (no data look-ahead).
    """
    timestamps = sorted({state.timestamp for state in states})
    if len(timestamps) < 2:
        return False
    deltas = sorted(
        (later - earlier).total_seconds()
        for earlier, later in pairwise(timestamps)
        if later > earlier
    )
    if not deltas:
        return False
    median = deltas[len(deltas) // 2]
    return 0.0 < median < 86_400.0


def _default_run_id(
    strategy_name: str, mode: BacktestMode, state_list: Sequence[MarketState]
) -> str:
    first = state_list[0].timestamp.strftime("%Y%m%d")
    last = state_list[-1].timestamp.strftime("%Y%m%d")
    return f"{strategy_name}-{mode.value}-{first}-{last}"


def _build_result(
    *,
    run_id: str,
    strategy: StrategyPipeline,
    mode: BacktestMode,
    state_list: Sequence[MarketState],
    initial_equity: float,
    ending_equity: float,
    equity_rows: list[dict[str, object]],
    weight_rows: list[dict[str, object]],
    trades: list[TradeRecord],
    fills: Sequence[FillRecord],
    roll_events: list[RollExecution],
    liquidations: list[ForcedLiquidationEvent],
    attribution: _Attribution,
    settlement_pnl: float,
    metadata: Mapping[str, ScalarValue],
    warnings: list[str],
    margin_requirements: MarginRequirements | None = None,
) -> BacktestResult:
    equity_curve = _curve_frame(equity_rows)
    if equity_curve.height == 0:
        start = state_list[0].timestamp
        end = state_list[-1].timestamp
    else:
        start = ensure_utc(equity_curve.get_column(TIMESTAMP_COLUMN)[0])
        end = ensure_utc(equity_curve.get_column(TIMESTAMP_COLUMN)[-1])
    margin_values = (
        equity_curve.get_column("margin_used").to_numpy()
        if equity_curve.height
        else np.array([], dtype=np.float64)
    )
    margin_used = float(np.max(margin_values)) if margin_values.size else 0.0
    utilizations = (
        equity_curve.get_column("margin_utilization").to_numpy()
        if equity_curve.height
        else np.array([], dtype=np.float64)
    )
    utilization_avg = float(np.mean(utilizations)) if utilizations.size else 0.0
    utilization_peak = float(np.max(utilizations)) if utilizations.size else 0.0
    metadata_full: dict[str, ScalarValue] = {
        "strategy": strategy.name,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "n_states": len(state_list),
    }
    metadata_full.update(metadata)
    roll_commissions = 0.0
    roll_fees = 0.0
    roll_slippage = 0.0
    for fill in fills:
        if fill.kind.value in {"roll_close", "roll_open"}:
            roll_commissions += fill.cost.commission
            roll_fees += fill.cost.fees
            roll_slippage += fill.cost.spread_cost + fill.cost.slippage_cost
    roll_cost_total = (
        roll_commissions + roll_fees + roll_slippage
        if fills
        else sum(float(event.total_cost) for event in roll_events)
    )
    forced_liquidation_cost = sum(float(event.penalty_cost) for event in liquidations)
    stress_roots: tuple[str, ...] = ()
    calibrated_roots: tuple[str, ...] = ()
    stress_applied = False
    stress_multiplier = 1.0
    calibrated = False
    if margin_requirements is not None:
        stress_roots = margin_requirements.stress_roots
        calibrated_roots = margin_requirements.calibrated_roots
        stress_applied = margin_requirements.stress_applied
        stress_multiplier = margin_requirements.stress_multiplier
        calibrated = margin_requirements.calibrated
    return BacktestResult(
        run_id=run_id,
        strategy_id=strategy.name,
        mode=mode,
        start=start,
        end=end,
        initial_equity=initial_equity,
        ending_equity=ending_equity,
        equity_curve=equity_curve,
        weight_history=_weight_frame(weight_rows),
        trades=tuple(trades),
        fills=tuple(fills),
        roll_events=tuple(roll_events),
        forced_liquidations=tuple(liquidations),
        price_pnl=attribution.price_pnl,
        roll_pnl=attribution.roll_pnl,
        commissions=attribution.commissions,
        fees=attribution.fees,
        slippage=attribution.slippage,
        funding=attribution.funding,
        turnover=attribution.turnover,
        margin_used_max=margin_used,
        margin_utilization_avg=utilization_avg,
        margin_utilization_peak=utilization_peak,
        settlement_pnl=settlement_pnl,
        roll_cost_total=roll_cost_total,
        roll_commissions=roll_commissions,
        roll_fees=roll_fees,
        roll_slippage=roll_slippage,
        forced_liquidation_cost=forced_liquidation_cost,
        margin_stress_applied=stress_applied,
        margin_stress_multiplier=stress_multiplier,
        margin_stress_roots=stress_roots,
        margin_calibrated=calibrated,
        margin_calibrated_roots=calibrated_roots,
        metadata=metadata_full,
        warnings=tuple(warnings),
    )


# ---------------------------------------------------------------------------
# Declarative runner + CLI
# ---------------------------------------------------------------------------


def _resolve_pipeline(
    config: StrategyConfig,
    pipeline: StrategyPipeline | None,
    instrument_master: InstrumentsConfig,
) -> StrategyPipeline:
    """Resolve a pipeline from ``src.strategies.pipeline`` unless one is injected."""
    if pipeline is not None:
        return pipeline
    try:
        from src.strategies import pipeline as pipeline_module
    except ImportError as exc:
        raise RuntimeError(
            "no StrategyPipeline supplied and src.strategies.pipeline is unavailable; "
            "pass pipeline=... explicitly or finish the strategy layer"
        ) from exc
    for factory_name in ("build_pipeline", "build_pipeline_from_config", "pipeline_from_config"):
        factory = getattr(pipeline_module, factory_name, None)
        if not callable(factory):
            continue
        try:
            parameters = list(inspect.signature(factory).parameters)
        except (TypeError, ValueError):  # pragma: no cover - exotic callables
            parameters = []
        if "instruments" in parameters:
            built = factory(config, instrument_master)
        else:
            built = factory(config)
        if not isinstance(built, StrategyPipeline):
            raise RuntimeError(f"{factory_name} did not return a StrategyPipeline")
        return built
    raise RuntimeError("src.strategies.pipeline does not expose a build_pipeline(config) factory")


def _resolve_translator(
    instrument_master: InstrumentsConfig,
    translator: ContractTranslatorProtocol | None,
) -> ContractTranslatorProtocol:
    if translator is not None:
        return translator
    try:
        from src.risk import contract_translator as translator_module
    except ImportError as exc:
        raise RuntimeError(
            "no ContractTranslatorProtocol supplied and src.risk.contract_translator is "
            "unavailable; pass translator=... explicitly or finish the risk layer"
        ) from exc
    translator_cls = getattr(translator_module, "ContractTranslator", None)
    if translator_cls is None:
        raise RuntimeError("src.risk.contract_translator.ContractTranslator is not available")
    return cast(ContractTranslatorProtocol, translator_cls(instrument_master))


def run_from_config(
    config: StrategyConfig,
    *,
    bars: pl.DataFrame | None = None,
    states: Sequence[MarketState] | None = None,
    pipeline: StrategyPipeline | None = None,
    translator: ContractTranslatorProtocol | None = None,
    instrument_master: InstrumentsConfig | None = None,
    mode: BacktestMode = BacktestMode.EVENT,
    state_builder: StateBuilder | None = None,
    initial_equity: float | None = None,
    cost_model: CostModel | None = None,
) -> BacktestResult:
    """Run a declarative :class:`StrategyConfig` end to end.

    Data and runtime dependencies are injected (``bars``/``states``); the
    strategy and translator are resolved lazily from their packages unless
    supplied explicitly, so a missing layer produces a clear error rather than an
    import-time failure.
    """
    master = instrument_master or load_instruments()
    spec = config.backtest
    equity = (
        initial_equity
        if initial_equity is not None
        else (spec.initial_equity if spec is not None else 1_000_000.0)
    )
    resolved_pipeline = _resolve_pipeline(config, pipeline, master)
    resolved_translator = _resolve_translator(master, translator)
    schedule = (
        RebalanceSchedule.from_frequency(spec.rebalance_frequency)
        if spec is not None
        else RebalanceSchedule.daily()
    )
    effective_cost_model = cost_model or CostModel(
        instrument_master=master,
        commission_bps=spec.commission_bps if spec is not None else 0.0,
        base_slippage_ticks=spec.slippage_ticks if spec is not None else 0.0,
    )
    timeframe = spec.timeframe if spec is not None else "1d"
    from src.data.timeframes import bars_per_year

    engine = BacktestEngine(
        resolved_pipeline,
        resolved_translator,
        mode=mode,
        instrument_master=master,
        cost_model=effective_cost_model,
        initial_equity=equity,
        rebalance_schedule=schedule,
        state_builder=state_builder,
        annualization_factor=bars_per_year(timeframe),
        timeframe=timeframe,
    )
    start = (
        datetime.combine(spec.start, datetime.min.time(), tzinfo=UTC) if spec is not None else None
    )
    end = datetime.combine(spec.end, datetime.min.time(), tzinfo=UTC) if spec is not None else None
    return engine.run(bars=bars, states=states, start=start, end=end)


# -- offline CLI fallbacks ---------------------------------------------------

_OFFLINE_ROOTS: tuple[str, ...] = ("ES", "NQ", "CL", "GC", "ZN")
_SECTOR_DAILY_VOL: Mapping[str, float] = MappingProxyType(
    {"equity": 0.18, "rates": 0.06, "energy": 0.35, "metals": 0.15, "fx": 0.08}
)


class OfflineMomentumPipeline:
    """Deterministic offline fallback pipeline used by ``--offline`` mode.

    It is a real sign-of-trailing-momentum strategy over the state's front
    contracts, deliberately simple so the CLI can exercise the whole backtest and
    tear-sheet stack without the strategy layer. Production runs use the real
    FinRL-X pipeline from ``src.strategies.pipeline``.
    """

    def __init__(self, name: str = "offline_momentum", gross: float = 1.0) -> None:
        self._name = name
        self._gross = gross

    @property
    def name(self) -> str:
        """Pipeline name used in telemetry and run metadata."""
        return self._name

    def run(self, state: MarketState) -> PipelineResult:
        """Return a momentum-signed equal-weight vector for the state universe."""
        signals: dict[str, float] = {}
        for root in state.ids:
            momentum = state.snapshot(root).feature("ret_21d", 0.0)
            if math.isnan(momentum) or abs(momentum) < 1e-12:
                signals[root] = 0.0
            else:
                signals[root] = 1.0 if momentum > 0.0 else -1.0
        active = {root: value for root, value in signals.items() if value != 0.0}
        if active:
            per_instrument = self._gross / len(active)
            weights = WeightVector(
                {root: value * per_instrument for root, value in signals.items()}
            )
        else:
            weights = WeightVector.empty()
        return PipelineResult(
            timestamp=state.timestamp,
            universe=state.ids,
            base_weights=weights,
            timing_weights=weights,
            risk_weights=weights,
            final_weights=weights,
            diagnostics={"pipeline": "offline_fallback"},
        )


class SimpleContractTranslator:
    """Deterministic offline fallback weight -> contracts translator."""

    def __init__(self, instrument_master: InstrumentsConfig) -> None:
        self._master = instrument_master

    def translate(
        self,
        weights: WeightVector,
        state: MarketState,
        *,
        equity: float | None = None,
        account_currency: str | None = None,
        fx_rates: Mapping[str, float] | None = None,
    ) -> ContractTargetSet:
        """Round ``w * equity`` notional to the nearest whole contract."""
        capital = equity if equity is not None else state.equity
        if capital is None:
            raise ValueError("equity must be available for contract translation")
        currency = (account_currency or state.account_currency).strip().upper()
        if currency != "USD":
            merged = dict(state.fx_rates)
            if fx_rates:
                merged.update(fx_rates)
            if currency not in merged:
                raise KeyError(
                    f"no FX rate for currency {currency!r} (fx_rates[{currency!r}] needed)"
                )
            capital *= merged[currency]
        if capital <= 0.0:
            raise ValueError("equity must be available for contract translation")
        targets: list[ContractTarget] = []
        for root, weight in weights.weights.items():
            if abs(weight) < 1e-12 or root not in state.instruments:
                continue
            snapshot = state.snapshot(root)
            spec = self._master.get(root)
            raw = weight * capital / (snapshot.price * spec.multiplier)
            quantity = math.floor(abs(raw) + 0.5)
            if quantity == 0:
                continue
            targets.append(
                ContractTarget(
                    instrument_id=root,
                    quantity=quantity if raw >= 0 else -quantity,
                    price=snapshot.price,
                    multiplier=spec.multiplier,
                    expiration=snapshot.expiration,
                    initial_margin=spec.margins.initial,
                    maintenance_margin=spec.margins.maintenance,
                    contract_symbol=_contract_symbol(root, snapshot.expiration),
                    asset_class=spec.asset_class,
                )
            )
        return ContractTargetSet(tuple(targets))


def _offline_bars(
    config: StrategyConfig,
    instrument_master: InstrumentsConfig,
    *,
    periods: int = 260,
    seed: int | None = None,
) -> pl.DataFrame:
    """Generate a deterministic offline daily bar frame for the CLI."""
    spec = config.backtest
    start_date = spec.start if spec is not None else date(2024, 1, 2)
    resolved_seed = spec.random_seed if seed is None and spec is not None else (seed or 7)
    rng = np.random.default_rng(resolved_seed)
    roots = [root for root in _OFFLINE_ROOTS if instrument_master.has(root)]
    if not roots:
        roots = list(instrument_master.roots())[:5]
    timestamps: list[datetime] = []
    cursor = datetime.combine(start_date, datetime.min.time(), tzinfo=UTC) + timedelta(hours=21)
    while len(timestamps) < periods:
        if cursor.weekday() < 5:
            timestamps.append(cursor)
        cursor += timedelta(days=1)
    rows: dict[str, list[object]] = {
        TIMESTAMP_COLUMN: [],
        "instrument_id": [],
        ROOT_SYMBOL_COLUMN: [],
        EXPIRATION_COLUMN: [],
        OPEN_COLUMN: [],
        HIGH_COLUMN: [],
        LOW_COLUMN: [],
        CLOSE_COLUMN: [],
        VOLUME_COLUMN: [],
        OPEN_INTEREST_COLUMN: [],
        SETTLEMENT_COLUMN: [],
    }
    expiration = date(start_date.year + 1, 3, 15)
    for root_index, root in enumerate(roots):
        instrument = instrument_master.get(root)
        price = 1_000.0 if not instrument.multiplier else 100.0 * instrument.multiplier**0.25
        daily_vol = _SECTOR_DAILY_VOL.get(instrument.sector, 0.15) / math.sqrt(252.0)
        for step, timestamp in enumerate(timestamps):
            returns = rng.normal(0.0002, daily_vol)
            open_price = price
            close_price = max(open_price * (1.0 + returns), instrument.tick_size)
            high = max(open_price, close_price) * (1.0 + abs(rng.normal(0.0, daily_vol * 0.5)))
            low = min(open_price, close_price) * (1.0 - abs(rng.normal(0.0, daily_vol * 0.5)))
            volume = float(abs(rng.normal(250_000.0, 25_000.0)))
            rows[TIMESTAMP_COLUMN].append(timestamp)
            rows["instrument_id"].append(_contract_symbol(root, expiration))
            rows[ROOT_SYMBOL_COLUMN].append(root)
            rows[EXPIRATION_COLUMN].append(expiration)
            rows[OPEN_COLUMN].append(open_price)
            rows[HIGH_COLUMN].append(high)
            rows[LOW_COLUMN].append(low)
            rows[CLOSE_COLUMN].append(close_price)
            rows[VOLUME_COLUMN].append(volume)
            rows[OPEN_INTEREST_COLUMN].append(float(1_000_000.0 + 1_000.0 * root_index + step))
            rows[SETTLEMENT_COLUMN].append(close_price)
            price = close_price
    return ensure_bars_schema(pl.DataFrame(rows), context="offline_bars")


def _resolve_cli_state_builder(
    instrument_master: InstrumentsConfig,
) -> StateBuilder | None:
    """Lazily use the data layer's state builder when it exists; else default.

    The data layer owns the canonical point-in-time ``MarketState`` builder. When
    it is importable we adapt its ``MarketStateBuilder.build`` method to the
    engine's injected ``StateBuilder`` callable; otherwise the engine's built-in
    deterministic builder is used, so ``--offline`` never depends on a layer that
    is not ready yet.
    """
    try:
        from src.data import market_state_builder as module
    except ImportError:
        return None
    for candidate in ("build_market_state", "build_state", "market_state_builder"):
        builder = getattr(module, candidate, None)
        if callable(builder):
            return cast(StateBuilder, builder)
    builder_cls = getattr(module, "MarketStateBuilder", None)
    if not isinstance(builder_cls, type):
        return None
    instance = builder_cls(instrument_master=instrument_master)

    def _build_state(frame: pl.DataFrame, timestamp: datetime, equity: float) -> MarketState:
        return cast(MarketState, instance.build(frame, timestamp, equity=equity))

    return _build_state


def _resolve_cli_pipeline(
    config: StrategyConfig,
    instrument_master: InstrumentsConfig,
    *,
    offline: bool,
) -> StrategyPipeline:
    try:
        return _resolve_pipeline(config, None, instrument_master)
    except RuntimeError:
        if not offline:
            raise
        print(
            "warning: strategy layer unavailable; using the deterministic offline "
            "momentum pipeline (offline mode)",
            file=sys.stderr,
        )
        return OfflineMomentumPipeline(name=config.id)


def _resolve_cli_translator(
    instrument_master: InstrumentsConfig, *, offline: bool
) -> ContractTranslatorProtocol:
    try:
        return _resolve_translator(instrument_master, None)
    except RuntimeError:
        if not offline:
            raise
        print(
            "warning: risk translator unavailable; using the deterministic offline "
            "translator (offline mode)",
            file=sys.stderr,
        )
        return SimpleContractTranslator(instrument_master)


def _default_report_path(config: StrategyConfig, mode: BacktestMode) -> Path:
    return Path("reports") / f"{config.id}_{mode.value}_tearsheet.json"


# ---------------------------------------------------------------------------
# Walk-forward CLI
# ---------------------------------------------------------------------------

#: Version of the walk-forward JSON report schema emitted by the CLI.
WALK_FORWARD_REPORT_VERSION: int = 1

#: Aggregate statistics reported for every fold metric.
WALK_FORWARD_AGGREGATE_FIELDS: tuple[str, ...] = (
    "cagr",
    "annual_volatility",
    "sharpe",
    "sortino",
    "calmar",
    "max_drawdown",
    "return_on_max_initial_margin",
    "roll_yield_annualized",
)


def _arg_at_least_two(value: str) -> int:
    """Argparse type: integer >= 2 (walk-forward fold count)."""
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be an integer >= 2") from exc
    if parsed < 2:
        raise argparse.ArgumentTypeError("must be >= 2 walk-forward folds")
    return parsed


def _arg_non_negative_int(value: str) -> int:
    """Argparse type: integer >= 0 (purge / embargo bar counts)."""
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def _default_walk_forward_path(config: StrategyConfig) -> Path:
    """Default JSON report path for a walk-forward CLI run."""
    return Path("artifacts") / f"{config.id}_walk_forward.json"


def _runnable_timestamps(bars: pl.DataFrame) -> tuple[datetime, ...]:
    """Sorted, unique decision timestamps of the runnable window."""
    if TIMESTAMP_COLUMN not in bars.columns:
        raise ValueError(f"walk-forward bars must contain a {TIMESTAMP_COLUMN!r} column")
    raw = bars.get_column(TIMESTAMP_COLUMN).unique().sort().to_list()
    timestamps = tuple(ensure_utc(cast(datetime, value)) for value in raw)
    if len(timestamps) < 2:
        raise ValueError(
            f"walk-forward requires at least two bars in the runnable window, got {len(timestamps)}"
        )
    return timestamps


def _build_walk_forward_folds(
    *,
    n_samples: int,
    splits: int,
    purge_bars: int,
    embargo_bars: int,
) -> tuple[PurgedWalkForwardSplit, list[WalkForwardFold]]:
    """Derive exactly ``splits`` purged/embargoed folds from the runnable window.

    The walk-forward geometry is derived from the window length so that the CLI's
    ``--wf-splits`` always yields that many folds: ``test_days`` is sized so the
    first training block plus ``splits`` back-to-back test blocks fill the window,
    ``train_days`` is the remainder, and the splitter enforces the purge gap and
    post-test embargo. Test windows are strictly increasing and non-overlapping.
    """
    if splits < 2:
        raise ValueError("walk-forward requires at least 2 folds")
    if purge_bars < 0 or embargo_bars < 0:
        raise ValueError("purge and embargo bar counts must be non-negative")
    if n_samples < 2:
        raise ValueError("walk-forward requires at least two observations")
    test_days = max(embargo_bars + 1, (n_samples - purge_bars) // (splits + 1))
    if test_days < 1:
        raise ValueError("unable to derive a positive walk-forward test window")
    train_days = n_samples - purge_bars - splits * test_days
    if train_days < 1 or train_days <= purge_bars:
        raise ValueError(
            f"insufficient bars for {splits} walk-forward folds with "
            f"purge={purge_bars} and embargo={embargo_bars}: "
            f"resolved train_days={train_days}, test_days={test_days} from "
            f"{n_samples} observations"
        )
    splitter = PurgedWalkForwardSplit(
        train_days=train_days,
        test_days=test_days,
        purge_days=purge_bars,
        embargo_days=embargo_bars,
        step_days=test_days,
        expanding=True,
    )
    folds = splitter.split(n_samples)
    if len(folds) != splits:
        raise ValueError(f"walk-forward splitter produced {len(folds)} folds, expected {splits}")
    return splitter, folds


def _walk_forward_metric_payload(metrics: PerformanceMetrics) -> dict[str, float]:
    """The per-fold metric subset embedded in the walk-forward JSON report."""
    return {
        "cagr": metrics.cagr,
        "annual_volatility": metrics.annualized_volatility,
        "annualized_volatility": metrics.annualized_volatility,
        "sharpe": metrics.sharpe,
        "sortino": metrics.sortino,
        "calmar": metrics.calmar,
        "omega": metrics.omega,
        "max_drawdown": metrics.max_drawdown,
        "max_drawdown_duration_days": metrics.max_drawdown_duration_days,
        "var_95_historical": metrics.var_95_historical,
        "var_99_historical": metrics.var_99_historical,
        "cvar_95": metrics.cvar_95,
        "cvar_99": metrics.cvar_99,
        "rachev_ratio_05": metrics.rachev_ratio_05,
        "return_on_max_initial_margin": metrics.return_on_max_initial_margin,
        "average_margin_utilization": metrics.average_margin_utilization,
        "peak_margin_utilization": metrics.peak_margin_utilization,
        "implementation_shortfall_ticks": metrics.implementation_shortfall_ticks,
        "turnover": metrics.turnover,
        "commission_to_gross_pnl_ratio": metrics.commission_to_gross_pnl_ratio,
        "roll_yield_annualized": metrics.roll_yield_annualized,
    }


def _aggregate_stats(values: Sequence[float]) -> dict[str, float]:
    """Mean and population standard deviation of a fold-metric series."""
    if not values:
        return {"mean": 0.0, "std": 0.0}
    mean = float(statistics.fmean(values))
    std = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
    return {"mean": mean, "std": std}


def _walk_forward_aggregate(
    *,
    fold_metrics: Sequence[PerformanceMetrics],
    fold_results: Sequence[BacktestResult],
) -> dict[str, object]:
    """Aggregate ``{mean, std}`` statistics across folds."""
    getters: dict[str, Callable[[PerformanceMetrics], float]] = {
        "cagr": lambda metric: metric.cagr,
        "annual_volatility": lambda metric: metric.annualized_volatility,
        "sharpe": lambda metric: metric.sharpe,
        "sortino": lambda metric: metric.sortino,
        "calmar": lambda metric: metric.calmar,
        "max_drawdown": lambda metric: metric.max_drawdown,
        "return_on_max_initial_margin": lambda metric: metric.return_on_max_initial_margin,
        "roll_yield_annualized": lambda metric: metric.roll_yield_annualized,
    }
    aggregate: dict[str, object] = {"n_folds": len(fold_metrics)}
    for field_name in WALK_FORWARD_AGGREGATE_FIELDS:
        aggregate[field_name] = _aggregate_stats(
            [getters[field_name](metric) for metric in fold_metrics]
        )
    aggregate["return"] = _aggregate_stats([result.total_return for result in fold_results])
    aggregate["ending_equity"] = _aggregate_stats([result.ending_equity for result in fold_results])
    aggregate["n_trades"] = _aggregate_stats([float(result.n_trades) for result in fold_results])
    return aggregate


#: Equity-curve columns that are denominated in account currency (scaled when
#: chaining independent fold curves into one walk-forward equity path).
_WALK_FORWARD_SCALED_COLUMNS: tuple[str, ...] = (
    "equity",
    "cash",
    "unrealized_pnl",
    "price_pnl",
    "roll_pnl",
    "commissions",
    "fees",
    "slippage",
    "funding",
    "margin_used",
    "gross_exposure",
    "net_exposure",
    "turnover",
)


def _combined_walk_forward_result(
    *,
    run_id: str,
    strategy_id: str,
    mode: BacktestMode,
    initial_equity: float,
    fold_results: Sequence[tuple[WalkForwardFold, BacktestResult]],
    metadata: Mapping[str, ScalarValue],
) -> BacktestResult:
    """Chain the independent fold curves into one compounding walk-forward result.

    Each fold is backtested independently (fresh capital, flat start) so no test
    outcome can leak between folds. For the reported path the fold returns are
    compounded: fold ``k`` starts at fold ``k-1``'s ending equity. The combined
    result is made accounting-consistent before being returned, so the regular
    tearsheet / identity checks apply to it as well.
    """
    rows: list[dict[str, object]] = []
    running_equity = initial_equity
    for _fold, result in fold_results:
        frame = result.equity_curve
        if frame.height == 0:
            continue
        factor = running_equity / result.initial_equity if result.initial_equity > 0.0 else 1.0
        timestamps = [
            ensure_utc(cast(datetime, value))
            for value in frame.get_column(TIMESTAMP_COLUMN).to_list()
        ]
        scaled = {
            column: frame.get_column(column).to_numpy().astype(np.float64) * factor
            for column in _WALK_FORWARD_SCALED_COLUMNS
        }
        utilizations = frame.get_column("margin_utilization").to_numpy().astype(np.float64)
        position_counts = frame.get_column("n_positions").to_list()
        for index, timestamp in enumerate(timestamps):
            row: dict[str, object] = {TIMESTAMP_COLUMN: timestamp}
            for column in _WALK_FORWARD_SCALED_COLUMNS:
                row[column] = float(scaled[column][index])
            row["margin_utilization"] = float(utilizations[index])
            row["n_positions"] = int(position_counts[index])
            row["drawdown"] = 0.0
            rows.append(row)
        running_equity = float(frame.get_column("equity")[-1]) * factor
    if not rows:
        raise ValueError("walk-forward produced no test observations")
    frame = _curve_frame(rows)
    equities = frame.get_column("equity").to_numpy().astype(np.float64)
    peaks = np.maximum.accumulate(equities)
    frame = frame.with_columns(
        pl.Series("drawdown", (equities / peaks - 1.0).tolist(), dtype=pl.Float64)
    )
    trades = tuple(trade for _fold, result in fold_results for trade in result.trades)
    roll_events = tuple(event for _fold, result in fold_results for event in result.roll_events)
    liquidations = tuple(
        event for _fold, result in fold_results for event in result.forced_liquidations
    )
    weight_frames = [
        result.weight_history for _fold, result in fold_results if result.weight_history.height
    ]
    weight_history = (
        pl.concat(weight_frames, how="vertical_relaxed")
        if weight_frames
        else _empty_weight_history()
    )
    commissions = float(frame.get_column("commissions").sum())
    fees = float(frame.get_column("fees").sum())
    slippage = float(frame.get_column("slippage").sum())
    funding = float(frame.get_column("funding").sum())
    turnover = float(frame.get_column("turnover").sum())
    roll_pnl = float(frame.get_column("roll_pnl").sum())
    ending_equity = float(equities[-1])
    # Keep the accounting identity exact for the compounded curve.
    price_pnl = (
        ending_equity - initial_equity + (commissions + fees + slippage + funding) - roll_pnl
    )
    utilizations = frame.get_column("margin_utilization").to_numpy().astype(np.float64)
    result = BacktestResult(
        run_id=run_id,
        strategy_id=strategy_id,
        mode=mode,
        start=ensure_utc(cast(datetime, frame.get_column(TIMESTAMP_COLUMN)[0])),
        end=ensure_utc(cast(datetime, frame.get_column(TIMESTAMP_COLUMN)[-1])),
        initial_equity=initial_equity,
        ending_equity=ending_equity,
        equity_curve=frame,
        weight_history=weight_history,
        trades=trades,
        fills=(),
        roll_events=roll_events,
        forced_liquidations=liquidations,
        price_pnl=price_pnl,
        roll_pnl=roll_pnl,
        commissions=commissions,
        fees=fees,
        slippage=slippage,
        funding=funding,
        turnover=turnover,
        margin_used_max=float(np.max(frame.get_column("margin_used").to_numpy())),
        margin_utilization_avg=float(np.mean(utilizations)) if utilizations.size else 0.0,
        margin_utilization_peak=float(np.max(utilizations)) if utilizations.size else 0.0,
        metadata=metadata,
    )
    result.check_accounting_identity()
    return result


@dataclass(frozen=True, slots=True)
class WalkForwardEquityCurve:
    """Compounded walk-forward equity path (each fold's return chains to the next)."""

    timestamps: tuple[str, ...]
    equity: tuple[float, ...]
    drawdown: tuple[float, ...]

    def __post_init__(self) -> None:
        if not (len(self.timestamps) == len(self.equity) == len(self.drawdown)):
            raise ValueError("walk-forward equity curve series must have equal lengths")
        if not self.equity:
            raise ValueError("walk-forward equity curve must not be empty")
        for value in self.equity:
            if not math.isfinite(value):
                raise ValueError("walk-forward equity values must be finite")
        for value in self.drawdown:
            if not math.isfinite(value):
                raise ValueError("walk-forward drawdown values must be finite")

    @property
    def final_equity(self) -> float:
        """Equity at the end of the compounded walk-forward path."""
        return float(self.equity[-1])

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable curve payload."""
        return {
            "timestamps": list(self.timestamps),
            "equity": list(self.equity),
            "drawdown": list(self.drawdown),
        }


@dataclass(frozen=True, slots=True)
class WalkForwardRunMetadata:
    """Run-level metadata of one walk-forward validation."""

    run_id: str
    strategy: str
    mode: BacktestMode
    initial_equity: float
    generated_at: datetime
    window_start: datetime
    window_end: datetime
    n_bars: int

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if not self.strategy:
            raise ValueError("strategy must be non-empty")
        if not isinstance(self.mode, BacktestMode):
            raise TypeError("mode must be a BacktestMode")
        for name in ("initial_equity",):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and strictly positive")
        for name in ("generated_at", "window_start", "window_end"):
            object.__setattr__(self, name, ensure_utc(getattr(self, name)))
        if isinstance(self.n_bars, bool) or not isinstance(self.n_bars, int) or self.n_bars < 1:
            raise ValueError("n_bars must be a positive int")

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable run metadata (matches the CLI report)."""
        return {
            "run_id": self.run_id,
            "strategy": self.strategy,
            "mode": self.mode.value,
            "initial_equity": self.initial_equity,
            "generated_at": self.generated_at.isoformat(),
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "n_bars": self.n_bars,
        }


@dataclass(frozen=True, slots=True)
class WalkForwardConfigSummary:
    """Geometry of the walk-forward split used for the run."""

    requested_splits: int
    fold_count: int
    purge_bars: int
    embargo_bars: int
    train_days: int
    test_days: int
    step_days: int
    expanding: bool

    def __post_init__(self) -> None:
        for name in (
            "requested_splits",
            "fold_count",
            "purge_bars",
            "embargo_bars",
            "train_days",
            "test_days",
            "step_days",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int")
        if self.requested_splits < 2:
            raise ValueError("requested_splits must be >= 2")
        if self.fold_count < 1:
            raise ValueError("fold_count must be >= 1")
        if self.purge_bars < 0 or self.embargo_bars < 0:
            raise ValueError("purge_bars and embargo_bars must be non-negative")
        if self.train_days < 1 or self.test_days < 1 or self.step_days < 1:
            raise ValueError("train/test/step days must be positive")
        if not isinstance(self.expanding, bool):
            raise TypeError("expanding must be a bool")

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable split geometry (matches the CLI report)."""
        return {
            "requested_splits": self.requested_splits,
            "fold_count": self.fold_count,
            "purge_bars": self.purge_bars,
            "embargo_bars": self.embargo_bars,
            "train_days": self.train_days,
            "test_days": self.test_days,
            "step_days": self.step_days,
            "expanding": self.expanding,
        }


@dataclass(frozen=True, slots=True)
class WalkForwardFoldReport:
    """One out-of-sample walk-forward fold with its metrics and bookkeeping."""

    fold: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_start_time: datetime
    train_end_time: datetime | None
    test_start_time: datetime
    test_end_time: datetime
    purge_gap: int
    train_size: int
    test_size: int
    n_bars: int
    return_pct: float
    ending_equity: float
    n_trades: int
    n_rebalances: int
    metrics: Mapping[str, float]

    def __post_init__(self) -> None:
        for name in (
            "fold",
            "train_start",
            "train_end",
            "test_start",
            "test_end",
            "purge_gap",
            "train_size",
            "test_size",
            "n_bars",
            "n_trades",
            "n_rebalances",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int")
        if self.fold < 0:
            raise ValueError("fold index must be non-negative")
        if self.train_start > self.train_end:
            raise ValueError("train_start must not exceed train_end")
        if self.test_start > self.test_end:
            raise ValueError("test_start must not exceed test_end")
        if self.train_end > self.test_start:
            raise ValueError("training data must end at or before the test window starts")
        if self.train_size < 1 or self.test_size < 1:
            raise ValueError("train_size and test_size must be positive")
        if self.n_bars < 0 or self.n_trades < 0 or self.n_rebalances < 0:
            raise ValueError("n_bars, n_trades and n_rebalances must be non-negative")
        if not math.isfinite(self.return_pct) or not math.isfinite(self.ending_equity):
            raise ValueError("fold return and ending equity must be finite")
        for name in ("train_start_time", "test_start_time", "test_end_time"):
            object.__setattr__(self, name, ensure_utc(getattr(self, name)))
        if self.train_end_time is not None:
            object.__setattr__(self, "train_end_time", ensure_utc(self.train_end_time))
        frozen_metrics: dict[str, float] = {}
        for key, value in self.metrics.items():
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(f"metric {key!r} must be finite")
            frozen_metrics[key] = numeric
        object.__setattr__(self, "metrics", MappingProxyType(frozen_metrics))

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable fold payload (matches the CLI report shape)."""
        return {
            "fold": self.fold,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "train_start_time": self.train_start_time.isoformat(),
            "train_end_time": (
                self.train_end_time.isoformat() if self.train_end_time is not None else None
            ),
            "test_start_time": self.test_start_time.isoformat(),
            "test_end_time": self.test_end_time.isoformat(),
            "purge_gap": self.purge_gap,
            "train_size": self.train_size,
            "test_size": self.test_size,
            "n_bars": self.n_bars,
            "return": self.return_pct,
            "ending_equity": self.ending_equity,
            "n_trades": self.n_trades,
            "n_rebalances": self.n_rebalances,
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True, slots=True)
class WalkForwardReport:
    """Frozen walk-forward validation report for the Backtest Lab API.

    ``to_dict()`` produces exactly the JSON document the CLI writes, while the
    convenience properties (``mean_sharpe``, ``total_trades``,
    ``total_rebalances``, ``fold_count``, ``final_equity``) give the API cheap
    headline access without re-parsing the payload. ``combined_result`` keeps the
    full compounded :class:`BacktestResult` for tear-sheet generation and is
    excluded from the JSON payload / equality / repr.
    """

    run: WalkForwardRunMetadata
    walk_forward: WalkForwardConfigSummary
    folds: tuple[WalkForwardFoldReport, ...]
    aggregate: Mapping[str, object]
    combined_metrics: Mapping[str, object]
    concatenated_equity_curve: WalkForwardEquityCurve
    tearsheet_path: str | None = None
    combined_result: BacktestResult | None = field(default=None, repr=False, compare=False)
    report: str = "walk_forward"
    schema_version: int = WALK_FORWARD_REPORT_VERSION

    def __post_init__(self) -> None:
        if self.report != "walk_forward":
            raise ValueError("report identifier must be 'walk_forward'")
        if self.schema_version < 1:
            raise ValueError("schema_version must be >= 1")
        if len(self.folds) != self.walk_forward.fold_count:
            raise ValueError("walk_forward.fold_count must equal the number of fold reports")
        if self.folds and not self.folds[0].fold == 0:
            raise ValueError("fold reports must start at index 0")
        frozen_aggregate: dict[str, object] = {}
        for key, value in self.aggregate.items():
            if isinstance(value, Mapping):
                frozen_aggregate[key] = MappingProxyType(dict(value))
            else:
                frozen_aggregate[key] = value
        object.__setattr__(self, "aggregate", MappingProxyType(frozen_aggregate))
        object.__setattr__(self, "combined_metrics", MappingProxyType(dict(self.combined_metrics)))
        if self.tearsheet_path is not None and not isinstance(self.tearsheet_path, str):
            raise TypeError("tearsheet_path must be a string or None")

    # -- convenience properties (Backtest Lab headline contract) --------------

    @property
    def mean_sharpe(self) -> float:
        """Mean walk-forward Sharpe ratio (``aggregate['sharpe']['mean']``)."""
        stats = self.aggregate.get("sharpe")
        if not isinstance(stats, Mapping):
            return 0.0
        mean = stats.get("mean")
        return float(mean) if isinstance(mean, (int, float)) else 0.0

    @property
    def total_trades(self) -> int:
        """Sum of executed trades across every out-of-sample fold."""
        return sum(fold.n_trades for fold in self.folds)

    @property
    def total_rebalances(self) -> int:
        """Sum of rebalances across every out-of-sample fold."""
        return sum(fold.n_rebalances for fold in self.folds)

    @property
    def fold_count(self) -> int:
        """Number of walk-forward folds in the report."""
        return len(self.folds)

    @property
    def final_equity(self) -> float:
        """Ending equity of the compounded walk-forward equity path."""
        return self.concatenated_equity_curve.final_equity

    # -- JSON payload ---------------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable report with the exact CLI report shape."""
        aggregate_payload: dict[str, object] = {}
        for key, value in self.aggregate.items():
            aggregate_payload[key] = dict(value) if isinstance(value, Mapping) else value
        return {
            "report": self.report,
            "schema_version": self.schema_version,
            "run": self.run.to_dict(),
            "walk_forward": self.walk_forward.to_dict(),
            "folds": [fold.to_dict() for fold in self.folds],
            "aggregate": aggregate_payload,
            "combined_metrics": dict(self.combined_metrics),
            "concatenated_equity_curve": self.concatenated_equity_curve.to_dict(),
            "tearsheet_path": self.tearsheet_path,
        }


def _state_timestamps(states: Sequence[MarketState]) -> tuple[datetime, ...]:
    """Sorted, unique decision timestamps implied by a sequence of market states."""
    timestamps = tuple(sorted({ensure_utc(state.timestamp) for state in states}))
    if len(timestamps) < 2:
        raise ValueError(
            "walk-forward requires at least two states in the runnable window, "
            f"got {len(timestamps)}"
        )
    return timestamps


def _states_by_timestamp(states: Sequence[MarketState]) -> dict[datetime, MarketState]:
    """Map market states by timestamp, rejecting duplicates."""
    mapping: dict[datetime, MarketState] = {}
    for state in states:
        if not isinstance(state, MarketState):
            raise TypeError("states must contain MarketState instances")
        timestamp = ensure_utc(state.timestamp)
        if timestamp in mapping:
            raise ValueError(f"duplicate MarketState timestamp {timestamp.isoformat()}")
        mapping[timestamp] = state
    return mapping


def run_walk_forward(
    config: StrategyConfig,
    *,
    bars: pl.DataFrame | None = None,
    states: Sequence[MarketState] | None = None,
    splits: int = 5,
    purge_bars: int = 5,
    embargo_bars: int = 5,
    initial_equity: float | None = None,
    mode: BacktestMode | None = None,
    instrument_master: InstrumentsConfig | None = None,
    pipeline: StrategyPipeline | None = None,
    translator: ContractTranslatorProtocol | None = None,
    offline: bool = False,
) -> WalkForwardReport:
    """Run purged/embargoed walk-forward validation and return a typed report.

    Parameters
    ----------
    config:
        Declarative strategy config (its ``backtest`` block supplies the default
        initial equity, cost model and rebalance frequency).
    bars:
        Canonical Polars OHLCV frame for the runnable window. When omitted (and
        ``states`` is also omitted) deterministic offline bars are generated.
    states:
        Optional prebuilt point-in-time ``MarketState`` sequence. When supplied
        it is used instead of building states from ``bars``; if both are given,
        ``bars`` supplies execution/roll context while ``states`` supplies the
        signal path.
    splits, purge_bars, embargo_bars:
        Walk-forward fold count and leakage guards (purge before / embargo after
        each test window, in bars).
    initial_equity, mode, instrument_master:
        Optional overrides; otherwise the config/engine defaults are used.
    pipeline, translator:
        Optional injected FinRL-X pipeline / contract translator. When omitted
        the helper resolves them exactly like the CLI does (fresh instance per
        fold so no overlay state can leak between folds).
    offline:
        Allow the CLI's deterministic offline fallbacks for pipeline/translator
        resolution. Implied when neither ``bars`` nor ``states`` is supplied.

    Each fold is backtested independently over its own test segment (flat start,
    configured initial equity) with the training segment reserved for feature
    warm-up only; the combined curve compounds the fold returns. The helper does
    not write files — the CLI adds the tearsheet path and persists the report.
    """
    from src.analytics.metrics import compute_metrics_from_result

    if isinstance(splits, bool) or not isinstance(splits, int) or splits < 2:
        raise ValueError("splits must be an integer >= 2")
    for name, value in (("purge_bars", purge_bars), ("embargo_bars", embargo_bars)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if mode is not None and not isinstance(mode, BacktestMode):
        raise TypeError("mode must be a BacktestMode or None")
    if not isinstance(offline, bool):
        raise TypeError("offline must be a bool")
    resolved_mode = mode if mode is not None else BacktestMode.VECTORIZED
    master = instrument_master if instrument_master is not None else load_instruments()
    spec = config.backtest
    wf_timeframe = spec.timeframe if spec is not None else "1d"
    from src.data.timeframes import bars_per_year

    wf_annualization = bars_per_year(wf_timeframe)
    resolved_equity = (
        initial_equity
        if initial_equity is not None
        else (spec.initial_equity if spec is not None else 1_000_000.0)
    )
    if not math.isfinite(resolved_equity) or resolved_equity <= 0.0:
        raise ValueError("initial_equity must be finite and strictly positive")
    cost_model = CostModel(
        instrument_master=master,
        commission_bps=spec.commission_bps if spec is not None else 0.0,
        base_slippage_ticks=spec.slippage_ticks if spec is not None else 0.0,
    )
    schedule = (
        RebalanceSchedule.from_frequency(spec.rebalance_frequency)
        if spec is not None
        else RebalanceSchedule.daily()
    )
    window_bars = bars
    if window_bars is None and states is None:
        # No input window: use the same deterministic offline source as the CLI.
        window_bars = _offline_bars(config, master)
        offline = True
    if window_bars is not None:
        timestamps = _runnable_timestamps(window_bars)
    else:
        if states is None:
            raise ValueError("either bars or states must be supplied")
        timestamps = _state_timestamps(states)
    if states is not None:
        state_by_timestamp = _states_by_timestamp(
            [
                state if state.timeframe == wf_timeframe else replace(state, timeframe=wf_timeframe)
                for state in states
            ]
        )
    else:
        if window_bars is None:
            raise ValueError("either bars or states must be supplied")
        state_builder = _resolve_cli_state_builder(master)
        all_states = _resolve_states(
            window_bars,
            None,
            state_builder,
            resolved_equity,
            start=None,
            end=None,
            timeframe=wf_timeframe,
        )
        state_by_timestamp = _states_by_timestamp(all_states)
    splitter, folds = _build_walk_forward_folds(
        n_samples=len(timestamps),
        splits=splits,
        purge_bars=purge_bars,
        embargo_bars=embargo_bars,
    )
    run_id = f"{config.id}-walk-forward-{timestamps[0]:%Y%m%d}-{timestamps[-1]:%Y%m%d}"
    fold_results: list[tuple[WalkForwardFold, BacktestResult]] = []
    fold_metrics: list[PerformanceMetrics] = []
    fold_reports: list[WalkForwardFoldReport] = []
    for fold in folds:
        test_start = timestamps[fold.test_start]
        test_end = timestamps[fold.test_end - 1]
        fold_states = [
            state
            for timestamp in timestamps[fold.test_start : fold.test_end]
            if (state := state_by_timestamp.get(timestamp)) is not None
        ]
        if not fold_states:
            raise ValueError(
                f"walk-forward fold {fold.fold_index} has no market states in its test window"
            )
        fold_pipeline = (
            pipeline
            if pipeline is not None
            else _resolve_cli_pipeline(config, master, offline=offline)
        )
        fold_translator = (
            translator
            if translator is not None
            else _resolve_cli_translator(master, offline=offline)
        )
        fold_engine = BacktestEngine(
            fold_pipeline,
            fold_translator,
            mode=resolved_mode,
            instrument_master=master,
            cost_model=cost_model,
            initial_equity=resolved_equity,
            rebalance_schedule=schedule,
            state_builder=None,
            annualization_factor=wf_annualization,
            timeframe=wf_timeframe,
        )
        fold_run_id = f"{run_id}-fold-{fold.fold_index}"
        if window_bars is not None:
            fold_result = fold_engine.run(
                bars=window_bars,
                states=fold_states,
                run_id=fold_run_id,
            )
        else:
            fold_result = fold_engine.run(states=fold_states, run_id=fold_run_id)
        fold_metric = compute_metrics_from_result(fold_result, timeframe=wf_timeframe)
        fold_results.append((fold, fold_result))
        fold_metrics.append(fold_metric)
        fold_reports.append(
            WalkForwardFoldReport(
                fold=fold.fold_index,
                train_start=fold.train_start,
                train_end=fold.train_end,
                test_start=fold.test_start,
                test_end=fold.test_end,
                train_start_time=timestamps[fold.train_start],
                train_end_time=(
                    timestamps[fold.train_end - 1] if fold.train_end > fold.train_start else None
                ),
                test_start_time=test_start,
                test_end_time=test_end,
                purge_gap=fold.purge_gap,
                train_size=fold.train_size,
                test_size=fold.test_size,
                n_bars=fold_result.n_bars,
                return_pct=fold_result.total_return,
                ending_equity=fold_result.ending_equity,
                n_trades=fold_result.n_trades,
                n_rebalances=fold_result.n_rebalances,
                metrics=_walk_forward_metric_payload(fold_metric),
            )
        )
    combined = _combined_walk_forward_result(
        run_id=run_id,
        strategy_id=config.id,
        mode=resolved_mode,
        initial_equity=resolved_equity,
        fold_results=fold_results,
        metadata={
            "strategy": config.id,
            "walk_forward": True,
            "folds": len(folds),
            "purge_bars": purge_bars,
            "embargo_bars": embargo_bars,
            "timeframe": wf_timeframe,
        },
    )
    combined_metrics = compute_metrics_from_result(combined, timeframe=wf_timeframe)
    combined_curve = combined.equity_curve
    concatenated_curve = WalkForwardEquityCurve(
        timestamps=tuple(
            ensure_utc(cast(datetime, value)).isoformat()
            for value in combined_curve.get_column(TIMESTAMP_COLUMN).to_list()
        ),
        equity=tuple(float(value) for value in combined_curve.get_column("equity").to_list()),
        drawdown=tuple(float(value) for value in combined_curve.get_column("drawdown").to_list()),
    )
    aggregate = _walk_forward_aggregate(
        fold_metrics=fold_metrics,
        fold_results=[result for _fold, result in fold_results],
    )
    return WalkForwardReport(
        run=WalkForwardRunMetadata(
            run_id=run_id,
            strategy=config.id,
            mode=resolved_mode,
            initial_equity=resolved_equity,
            generated_at=combined.end,
            window_start=timestamps[0],
            window_end=timestamps[-1],
            n_bars=len(timestamps),
        ),
        walk_forward=WalkForwardConfigSummary(
            requested_splits=splits,
            fold_count=len(folds),
            purge_bars=purge_bars,
            embargo_bars=embargo_bars,
            train_days=splitter.train_days,
            test_days=splitter.test_days,
            step_days=splitter.step_days,
            expanding=splitter.expanding,
        ),
        folds=tuple(fold_reports),
        aggregate=aggregate,
        combined_metrics=combined_metrics.to_dict(),
        concatenated_equity_curve=concatenated_curve,
        combined_result=combined,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: run a declarative strategy offline and write a tear sheet."""
    parser = argparse.ArgumentParser(
        prog="futuresrl-backtest",
        description="Run a FuturesRL declarative strategy backtest (offline, no network).",
    )
    parser.add_argument(
        "--strategy",
        required=True,
        help="Path to a declarative strategy YAML (e.g. configs/strategies/tsmom_vol_target.yaml)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use deterministic synthetic bars and offline fallbacks (never hits the network)",
    )
    parser.add_argument(
        "--bars",
        type=Path,
        default=None,
        help="Optional Parquet file with canonical ohlcv bars (instead of --offline data)",
    )
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in BacktestMode],
        default=BacktestMode.VECTORIZED.value,
        help="Execution mode (default: vectorized)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON tear-sheet path (default: reports/<strategy>_<mode>_tearsheet.json)",
    )
    parser.add_argument(
        "--html",
        type=Path,
        default=None,
        help="Optional output HTML tear-sheet path",
    )
    parser.add_argument(
        "--initial-equity",
        type=float,
        default=None,
        help="Override the backtest initial equity",
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Run purged + embargoed walk-forward validation over the runnable window",
    )
    parser.add_argument(
        "--wf-splits",
        type=_arg_at_least_two,
        default=5,
        metavar="N",
        help="Number of walk-forward folds (default: 5)",
    )
    parser.add_argument(
        "--purge-bars",
        type=_arg_non_negative_int,
        default=None,
        metavar="N",
        help="Bars purged before each test window (default: config walk_forward or 5)",
    )
    parser.add_argument(
        "--embargo-bars",
        type=_arg_non_negative_int,
        default=None,
        metavar="N",
        help="Bars embargoed after each test window (default: config walk_forward or 5)",
    )
    args = parser.parse_args(argv)
    try:
        config = load_strategy_config(args.strategy)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    instrument_master = load_instruments()
    mode = BacktestMode(args.mode)
    try:
        if args.bars is not None:
            bars = ensure_bars_schema(pl.read_parquet(args.bars), context="cli_bars")
        else:
            bars = _offline_bars(config, instrument_master)
        pipeline = _resolve_cli_pipeline(config, instrument_master, offline=args.offline)
        translator = _resolve_cli_translator(instrument_master, offline=args.offline)
        spec = config.backtest
        cost_model = CostModel(
            instrument_master=instrument_master,
            commission_bps=spec.commission_bps if spec is not None else 0.0,
            base_slippage_ticks=spec.slippage_ticks if spec is not None else 0.0,
        )
        schedule = (
            RebalanceSchedule.from_frequency(spec.rebalance_frequency)
            if spec is not None
            else RebalanceSchedule.daily()
        )
        state_builder = _resolve_cli_state_builder(instrument_master)
        initial_equity = (
            args.initial_equity
            if args.initial_equity is not None
            else (spec.initial_equity if spec is not None else 1_000_000.0)
        )
        if args.walk_forward:
            spec_wf = spec.walk_forward if spec is not None else None
            resolved_purge = (
                args.purge_bars
                if args.purge_bars is not None
                else (spec_wf.purge_days if spec_wf is not None else 5)
            )
            resolved_embargo = (
                args.embargo_bars
                if args.embargo_bars is not None
                else (spec_wf.embargo_days if spec_wf is not None else 5)
            )
            walk_forward_output = args.output or _default_walk_forward_path(config)
            report = run_walk_forward(
                config,
                bars=bars,
                splits=args.wf_splits,
                purge_bars=resolved_purge,
                embargo_bars=resolved_embargo,
                initial_equity=initial_equity,
                mode=mode,
                instrument_master=instrument_master,
                offline=args.offline,
            )
            tearsheet_path: str | None = None
            if report.combined_result is not None:
                from src.analytics.tearsheet import build_tearsheet, write_tearsheet

                tearsheet = build_tearsheet(
                    report.combined_result,
                    title=f"Walk-forward — {config.id}",
                    generated_at=report.combined_result.end,
                )
                tearsheet_file = walk_forward_output.with_name(
                    f"{walk_forward_output.stem}_tearsheet.json"
                )
                write_tearsheet(tearsheet, tearsheet_file, fmt="json")
                tearsheet_path = str(tearsheet_file)
                if args.html is not None:
                    write_tearsheet(tearsheet, args.html, fmt="html")
            report = replace(report, tearsheet_path=tearsheet_path)
            payload = report.to_dict()
            walk_forward_output.parent.mkdir(parents=True, exist_ok=True)
            walk_forward_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(
                json.dumps(
                    {
                        "strategy": config.id,
                        "mode": mode.value,
                        "report": report.report,
                        "folds": report.fold_count,
                        "run_id": report.run.run_id,
                        "output": str(walk_forward_output),
                        "tearsheet_path": payload["tearsheet_path"],
                        "aggregate": payload["aggregate"],
                    },
                    indent=2,
                )
            )
            return 0
        engine = BacktestEngine(
            pipeline,
            translator,
            mode=mode,
            instrument_master=instrument_master,
            cost_model=cost_model,
            initial_equity=initial_equity,
            rebalance_schedule=schedule,
            state_builder=state_builder,
        )
        result = engine.run(bars=bars)
    except Exception as exc:  # pragma: no cover - CLI guard
        print(f"error: backtest failed: {exc}", file=sys.stderr)
        return 1
    from src.analytics.metrics import compute_metrics_from_result
    from src.analytics.tearsheet import build_tearsheet, write_tearsheet

    metrics = compute_metrics_from_result(result)
    tearsheet = build_tearsheet(result, metrics, title=f"Backtest — {config.id}")
    output = args.output or _default_report_path(config, mode)
    write_tearsheet(tearsheet, output, fmt="json")
    if args.html is not None:
        write_tearsheet(tearsheet, args.html, fmt="html")
    print(
        json.dumps(
            {
                "strategy": config.id,
                "mode": mode.value,
                "run_id": result.run_id,
                "ending_equity": result.ending_equity,
                "total_return": result.total_return,
                "sharpe": metrics.sharpe,
                "max_drawdown": metrics.max_drawdown,
                "output": str(output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
