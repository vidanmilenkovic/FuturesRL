"""Futures margin, daily mark-to-market settlement and forced-liquidation simulator.

Futures accounts differ from cash-equity accounts in two ways that matter for a
faithful backtest:

* **Daily mark-to-market settlement.** At the end of every session the exchange
  settles the account at the official settlement price: the day's unrealised PnL
  becomes realised cash and every position's carrying price resets to the
  settlement price. Settlement is a *conservation* operation -- net liquidation
  value is unchanged -- but it changes the cash balance and therefore the excess
  liquidity available to post new margin.
* **Margin calls.** If net liquidation value falls below the maintenance margin
  requirement the broker force-liquidates positions at unfavourable execution
  prices plus a liquidation penalty fee.

This module is a deterministic calculator: it never holds mutable state. The
event-driven broker owns cash/positions and delegates the settlement and margin
arithmetic to an injected :class:`MarginSimulator`; the vectorised backtester
uses the same class to size margin utilisation from fractional contract
exposure. It reuses the immutable ``Position`` and ``AccountSnapshot`` core
types so the simulated account is wire-compatible with IBKR telemetry.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

from configs.instruments import InstrumentsConfig

from src.core.types import (
    AccountSnapshot,
    InstrumentId,
    Position,
    ensure_utc,
)

# ---------------------------------------------------------------------------
# Typed result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MarginRequirements:
    """Aggregate initial and maintenance margin required by a position book."""

    initial_margin: float = 0.0
    maintenance_margin: float = 0.0

    def __post_init__(self) -> None:
        for name in ("initial_margin", "maintenance_margin"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

    @property
    def utilisation(self) -> float:
        """Maintenance-to-initial ratio (``0`` when no margin is required)."""
        if self.initial_margin <= 0.0:
            return 0.0
        return self.maintenance_margin / self.initial_margin


@dataclass(frozen=True, slots=True)
class SettlementResult:
    """Outcome of one daily mark-to-market settlement pass.

    The settlement identity is invariant by construction:
    ``equity_after == equity_before`` (the PnL is moved from unrealised to
    realised, never created or destroyed) and
    ``cash_after == cash_before + settlement_pnl``.
    """

    timestamp: datetime
    cash_before: float
    cash_after: float
    settlement_pnl: float
    unrealized_pnl_before: float
    equity_before: float
    equity_after: float
    positions: tuple[Position, ...]
    initial_margin_used: float
    maintenance_margin_used: float
    excess_liquidity: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", ensure_utc(self.timestamp))
        for name in (
            "cash_before",
            "cash_after",
            "settlement_pnl",
            "unrealized_pnl_before",
            "equity_before",
            "equity_after",
            "initial_margin_used",
            "maintenance_margin_used",
            "excess_liquidity",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)

    @property
    def margin_utilization(self) -> float:
        """Initial margin utilisation after settlement (``0`` when flat)."""
        if self.equity_after <= 0.0:
            return 0.0
        return self.initial_margin_used / self.equity_after

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable representation."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "cash_before": self.cash_before,
            "cash_after": self.cash_after,
            "settlement_pnl": self.settlement_pnl,
            "unrealized_pnl_before": self.unrealized_pnl_before,
            "equity_before": self.equity_before,
            "equity_after": self.equity_after,
            "initial_margin_used": self.initial_margin_used,
            "maintenance_margin_used": self.maintenance_margin_used,
            "margin_utilization": self.margin_utilization,
            "excess_liquidity": self.excess_liquidity,
        }


@dataclass(frozen=True, slots=True)
class MarginCall:
    """A maintenance-margin breach evaluated at one point in time."""

    timestamp: datetime
    triggered: bool
    equity: float
    initial_margin_used: float
    maintenance_margin_used: float
    deficit: float
    utilisation: float
    positions: tuple[Position, ...]
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", ensure_utc(self.timestamp))
        for name in (
            "equity",
            "initial_margin_used",
            "maintenance_margin_used",
            "deficit",
            "utilisation",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable representation."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "triggered": self.triggered,
            "equity": self.equity,
            "initial_margin_used": self.initial_margin_used,
            "maintenance_margin_used": self.maintenance_margin_used,
            "deficit": self.deficit,
            "utilisation": self.utilisation,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class ForcedLiquidationPlan:
    """Deterministic plan for liquidating a margin-call position book."""

    timestamp: datetime
    positions: tuple[Position, ...]
    estimated_slippage_cost: float
    estimated_penalty_fee: float
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", ensure_utc(self.timestamp))
        for name in ("estimated_slippage_cost", "estimated_penalty_fee"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

    @property
    def total_estimated_cost(self) -> float:
        """Estimated total liquidation penalty in account currency."""
        return self.estimated_slippage_cost + self.estimated_penalty_fee

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable representation."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "positions": [
                {"instrument_id": p.instrument_id, "quantity": p.quantity} for p in self.positions
            ],
            "estimated_slippage_cost": self.estimated_slippage_cost,
            "estimated_penalty_fee": self.estimated_penalty_fee,
            "total_estimated_cost": self.total_estimated_cost,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


class MarginSimulator:
    """Position-level IM/MM tracking, MTM settlement and forced-liquidation logic.

    Margin levels come from the ``Position`` objects (which the broker populates
    from the instrument master) with the instrument master as a fallback. When
    an instrument declares only an initial margin, the maintenance margin falls
    back to ``initial * maintenance_ratio`` (default ``0.90``, a conservative
    CME-like ratio).
    """

    def __init__(
        self,
        instrument_master: InstrumentsConfig | None = None,
        *,
        maintenance_ratio: float = 0.90,
        penalty_ticks: float = 2.0,
        penalty_fee_per_contract: float = 15.0,
        margin_call_buffer: float = 0.0,
    ) -> None:
        if not math.isfinite(maintenance_ratio) or not 0.0 <= maintenance_ratio <= 1.0:
            raise ValueError("maintenance_ratio must be finite and within [0, 1]")
        for name, value in (
            ("penalty_ticks", penalty_ticks),
            ("penalty_fee_per_contract", penalty_fee_per_contract),
            ("margin_call_buffer", margin_call_buffer),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        self._instrument_master = instrument_master
        self.maintenance_ratio = float(maintenance_ratio)
        self.penalty_ticks = float(penalty_ticks)
        self.penalty_fee_per_contract = float(penalty_fee_per_contract)
        self.margin_call_buffer = float(margin_call_buffer)

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _validate_positions(positions: Mapping[InstrumentId, Position]) -> None:
        for instrument_id, position in positions.items():
            if instrument_id != position.instrument_id:
                raise ValueError(
                    f"position key {instrument_id!r} does not match "
                    f"instrument_id {position.instrument_id!r}"
                )

    def _margin_per_contract(self, position: Position) -> tuple[float, float]:
        """Resolve ``(initial, maintenance)`` margin per contract for a position."""
        initial = float(position.initial_margin)
        maintenance = float(position.maintenance_margin)
        master = self._instrument_master
        if (
            (initial <= 0.0 or maintenance <= 0.0)
            and master is not None
            and master.has(position.instrument_id)
        ):
            spec = master.get(position.instrument_id)
            if initial <= 0.0:
                initial = float(spec.margins.initial)
            if maintenance <= 0.0:
                maintenance = float(spec.margins.maintenance)
        if initial < 0.0 or maintenance < 0.0:
            raise ValueError(f"negative margin for {position.instrument_id!r}")
        if maintenance <= 0.0 and initial > 0.0:
            maintenance = initial * self.maintenance_ratio
        if maintenance > initial and initial > 0.0:
            maintenance = initial
        return initial, maintenance

    # -- margin arithmetic ----------------------------------------------------

    def margin_requirements(self, positions: Mapping[InstrumentId, Position]) -> MarginRequirements:
        """Total initial and maintenance margin for ``positions``."""
        self._validate_positions(positions)
        initial = 0.0
        maintenance = 0.0
        for position in positions.values():
            per_contract_initial, per_contract_maintenance = self._margin_per_contract(position)
            quantity = abs(position.quantity)
            initial += per_contract_initial * quantity
            maintenance += per_contract_maintenance * quantity
        return MarginRequirements(initial_margin=initial, maintenance_margin=maintenance)

    def unrealized_pnl(self, positions: Mapping[InstrumentId, Position]) -> float:
        """Sum of position-level unrealised PnL at current market prices."""
        self._validate_positions(positions)
        return sum(position.unrealized_pnl for position in positions.values())

    def equity(self, cash: float, positions: Mapping[InstrumentId, Position]) -> float:
        """Net liquidation value ``cash + unrealised PnL``."""
        if not math.isfinite(cash):
            raise ValueError("cash must be finite")
        return cash + self.unrealized_pnl(positions)

    def account_snapshot(
        self,
        timestamp: datetime,
        *,
        cash: float,
        positions: Mapping[InstrumentId, Position],
    ) -> AccountSnapshot:
        """Build an :class:`AccountSnapshot` from cash and marked positions.

        Raises ``ValueError`` when net liquidation is non-positive: the core
        ``AccountSnapshot`` contract requires a strictly positive NLV, and a
        depleted account must trigger the forced-liquidation path instead of
        being silently reported.
        """
        requirements = self.margin_requirements(positions)
        net_liquidation = self.equity(cash, positions)
        if net_liquidation <= 0.0:
            raise ValueError(
                f"net liquidation value is non-positive ({net_liquidation:.2f}); "
                "the account is depleted and must be liquidated"
            )
        return AccountSnapshot(
            timestamp=timestamp,
            net_liquidation=net_liquidation,
            cash_balance=cash,
            buying_power=max(0.0, net_liquidation - requirements.initial_margin),
            initial_margin_used=requirements.initial_margin,
            maintenance_margin_used=requirements.maintenance_margin,
            excess_liquidity=net_liquidation - requirements.initial_margin,
        )

    # -- daily mark-to-market settlement -------------------------------------

    def settle(
        self,
        timestamp: datetime,
        *,
        cash: float,
        positions: Mapping[InstrumentId, Position],
        settlement_prices: Mapping[InstrumentId, float] | None = None,
    ) -> SettlementResult:
        """Perform one daily mark-to-market settlement pass.

        Every position is re-marked to its settlement price; the price change
        since the position's carrying (average) price is credited to cash and the
        carrying price resets to the settlement price. Missing settlement prices
        fall back to the position's current ``market_price`` (no PnL that day).
        """
        if not math.isfinite(cash):
            raise ValueError("cash must be finite")
        self._validate_positions(positions)
        prices = settlement_prices or {}
        settlement_pnl = 0.0
        unrealized_before = 0.0
        settled: list[Position] = []
        for instrument_id in sorted(positions):
            position = positions[instrument_id]
            price = prices.get(instrument_id, position.market_price)
            if not math.isfinite(price) or price <= 0.0:
                raise ValueError(
                    f"settlement price for {instrument_id!r} must be finite and positive"
                )
            position_pnl = (
                position.quantity * (price - position.average_price) * position.multiplier
            )
            # Equity is conserved *at the settlement price*: the PnL that was
            # unrealised at that price becomes realised cash.
            unrealized_before += position_pnl
            settlement_pnl += position_pnl
            settled.append(
                Position(
                    instrument_id=position.instrument_id,
                    quantity=position.quantity,
                    average_price=price,
                    market_price=price,
                    multiplier=position.multiplier,
                    expiration=position.expiration,
                    initial_margin=position.initial_margin,
                    maintenance_margin=position.maintenance_margin,
                    contract_symbol=position.contract_symbol,
                )
            )
        cash_after = cash + settlement_pnl
        equity_before = cash + unrealized_before
        equity_after = cash_after
        requirements = self.margin_requirements(
            {position.instrument_id: position for position in settled}
        )
        if abs(equity_after - equity_before) > 1e-6 * max(1.0, abs(equity_before)):
            # Defensive: settlement must never create or destroy value.
            raise AssertionError(
                f"daily settlement violated value conservation: {equity_before} -> {equity_after}"
            )
        return SettlementResult(
            timestamp=timestamp,
            cash_before=cash,
            cash_after=cash_after,
            settlement_pnl=settlement_pnl,
            unrealized_pnl_before=unrealized_before,
            equity_before=equity_before,
            equity_after=equity_after,
            positions=tuple(settled),
            initial_margin_used=requirements.initial_margin,
            maintenance_margin_used=requirements.maintenance_margin,
            excess_liquidity=equity_after - requirements.initial_margin,
        )

    # -- margin calls / forced liquidation ------------------------------------

    def check_margin_call(
        self,
        timestamp: datetime,
        *,
        equity: float,
        positions: Mapping[InstrumentId, Position],
    ) -> MarginCall:
        """Evaluate the maintenance-margin call condition for the account.

        A call is triggered when ``equity < maintenance_margin * (1 + buffer)``.
        An empty position book can never trigger a call (there is nothing to
        liquidate).
        """
        if not math.isfinite(equity):
            raise ValueError("equity must be finite")
        self._validate_positions(positions)
        requirements = self.margin_requirements(positions)
        threshold = requirements.maintenance_margin * (1.0 + self.margin_call_buffer)
        triggered = (
            any(position.quantity != 0 for position in positions.values()) and equity < threshold
        )
        deficit = max(0.0, threshold - equity) if triggered else 0.0
        utilisation = requirements.initial_margin / equity if equity > 0.0 else 0.0
        reason = ""
        if triggered:
            reason = (
                f"equity {equity:.2f} below maintenance margin {threshold:.2f} "
                f"(deficit {deficit:.2f})"
            )
        active_positions = tuple(
            positions[instrument_id]
            for instrument_id in sorted(positions)
            if positions[instrument_id].quantity != 0
        )
        return MarginCall(
            timestamp=timestamp,
            triggered=triggered,
            equity=equity,
            initial_margin_used=requirements.initial_margin,
            maintenance_margin_used=requirements.maintenance_margin,
            deficit=deficit,
            utilisation=utilisation,
            positions=active_positions,
            reason=reason,
        )

    def forced_liquidation_plan(self, margin_call: MarginCall) -> ForcedLiquidationPlan:
        """Build the forced-liquidation plan for a triggered margin call."""
        if not margin_call.triggered:
            raise ValueError("cannot build a liquidation plan for a non-triggered margin call")
        slippage_cost = 0.0
        penalty_fee = 0.0
        for position in margin_call.positions:
            tick_size = 0.0
            if self._instrument_master is not None and self._instrument_master.has(
                position.instrument_id
            ):
                tick_size = self._instrument_master.get(position.instrument_id).tick_size
            slippage_cost += (
                self.penalty_ticks * tick_size * position.multiplier * abs(position.quantity)
            )
            penalty_fee += self.penalty_fee_per_contract * abs(position.quantity)
        return ForcedLiquidationPlan(
            timestamp=margin_call.timestamp,
            positions=margin_call.positions,
            estimated_slippage_cost=abs(slippage_cost),
            estimated_penalty_fee=abs(penalty_fee),
            reason=margin_call.reason,
        )

    def utilisation(self, snapshot: AccountSnapshot) -> float:
        """Initial-margin utilisation of an account snapshot."""
        return snapshot.margin_utilization

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable configuration description (run metadata)."""
        return {
            "maintenance_ratio": self.maintenance_ratio,
            "penalty_ticks": self.penalty_ticks,
            "penalty_fee_per_contract": self.penalty_fee_per_contract,
            "margin_call_buffer": self.margin_call_buffer,
        }
