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
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from types import MappingProxyType

from configs.instruments import InstrumentsConfig

from src.core.types import (
    AccountSnapshot,
    InstrumentId,
    Position,
    ensure_utc,
)


def utc_calendar_day(timestamp: datetime) -> date:
    """UTC calendar day of ``timestamp`` (daily settlement boundary)."""
    return ensure_utc(timestamp).date()


def is_new_utc_day(previous_timestamp: datetime | None, timestamp: datetime) -> bool:
    """Whether ``timestamp`` crosses into a later UTC calendar day than ``previous``.

    Intraday bars within the same UTC day return ``False``; the first bar of a new
    UTC day (or any bar when no previous timestamp is supplied) returns ``True``.
    """
    current = utc_calendar_day(timestamp)
    if previous_timestamp is None:
        return True
    return current > utc_calendar_day(previous_timestamp)


@dataclass(slots=True)
class DailySettlementClock:
    """UTC calendar-day gate enforcing at most one daily settlement per day.

    The event-driven backtester uses this for intraday bars so mark-to-market
    settlement fires on UTC-day boundaries instead of once per bar. Daily (or
    coarser) bars keep their legacy per-bar settlement behaviour.
    """

    last_settled_day: date | None = None
    settlements: int = 0

    def due(self, timestamp: datetime) -> bool:
        """Whether a daily settlement is due at ``timestamp`` (not yet today)."""
        day = utc_calendar_day(timestamp)
        return self.last_settled_day is None or day > self.last_settled_day

    def mark(self, timestamp: datetime) -> None:
        """Record that the UTC day of ``timestamp`` has been settled."""
        self.last_settled_day = utc_calendar_day(timestamp)
        self.settlements += 1

    def reset(self) -> None:
        """Clear the gate (new backtest run)."""
        self.last_settled_day = None
        self.settlements = 0


#: Relative persisted IBKR calibration document (task-84/85 contract).
CALIBRATED_COSTS_RELATIVE_PATH: Path = Path("cache") / "ibkr_calibrated_costs.json"

#: Default stress-regime margin multiplier (task-86).
DEFAULT_STRESS_MARGIN_MULTIPLIER: float = 1.35

#: Default annualised volatility threshold above which stress margins apply.
DEFAULT_STRESS_VOL_THRESHOLD: float = 0.30


@dataclass(frozen=True, slots=True)
class CalibratedMargin:
    """IBKR whatIf-calibrated initial/maintenance margin for one root (USD)."""

    initial_margin: float
    maintenance_margin: float
    calibrated_at: str | None = None
    margin_source: str = "ibkr_whatif"

    def __post_init__(self) -> None:
        for name in ("initial_margin", "maintenance_margin"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and strictly positive")
            object.__setattr__(self, name, value)


def default_calibrated_costs_path() -> Path:
    """Resolve the persisted calibration path (shared data-eng task-84 contract).

    Delegates to :func:`src.backtest.cost_model.default_calibrated_costs_path`
    (env override ``FUTURESRL_IBKR_CALIBRATED_COSTS_PATH`` then
    ``settings.data_dir/cache/ibkr_calibrated_costs.json``) with a pure-path
    fallback so margin code can never fail on settings/import issues.
    """
    try:
        from src.backtest.cost_model import (
            default_calibrated_costs_path as cost_model_default_path,
        )

        return cost_model_default_path()
    except Exception:  # pragma: no cover - settings/import fallback
        override = os.environ.get("FUTURESRL_IBKR_CALIBRATED_COSTS_PATH") or os.environ.get(
            "FUTURESRL_CALIBRATED_COSTS_PATH"
        )
        if override:
            return Path(override).expanduser()
        return Path(__file__).resolve().parents[2] / "data" / CALIBRATED_COSTS_RELATIVE_PATH


def load_calibrated_margins(
    path: Path | str | None = None,
    *,
    max_age_days: float | None = None,
) -> dict[str, CalibratedMargin]:
    """Load calibrated initial/maintenance margins from the shared IBKR JSON.

    Reuses data-eng task-84's validated loader (``schema_version``/staleness/
    provenance contract) and converts per-root overrides to :class:`CalibratedMargin`.
    Absent, unreadable, corrupt, stale or empty documents fail cleanly to ``{}`` so
    the simulator/translator fall back to YAML/position margins; roots without
    usable positive initial+maintenance margins are skipped individually.
    """
    try:
        from src.backtest.cost_model import (
            CalibrationError,
            load_calibrated_costs,
        )
        from src.backtest.cost_model import (
            default_calibrated_costs_path as cost_model_default_path,
        )

        resolved = Path(path) if path is not None else cost_model_default_path()
        parsed = load_calibrated_costs(
            resolved,
            max_age_days=max_age_days,
        )
    except (CalibrationError, ImportError, OSError, ValueError):
        return {}
    result: dict[str, CalibratedMargin] = {}
    calibrated_at = parsed.calibrated_at.isoformat()
    for raw_root, override in parsed.roots.items():
        initial = override.initial_margin
        maintenance = override.maintenance_margin
        if initial is None or maintenance is None:
            continue
        if initial <= 0.0 or maintenance <= 0.0:
            continue
        result[str(raw_root).strip().upper()] = CalibratedMargin(
            initial_margin=float(initial),
            maintenance_margin=float(maintenance),
            calibrated_at=calibrated_at,
            margin_source=override.margin_source,
        )
    return result


@dataclass(frozen=True, slots=True)
class MarginRequirements:
    """Aggregate initial and maintenance margin required by a position book."""

    initial_margin: float = 0.0
    maintenance_margin: float = 0.0
    #: Whether the stress-regime multiplier was applied to any root.
    stress_applied: bool = False
    #: Effective stress multiplier (``1.0`` when no root is stressed).
    stress_multiplier: float = 1.0
    #: Roots whose margin was stress-multiplied (sorted).
    stress_roots: tuple[InstrumentId, ...] = ()
    #: Whether any root used an IBKR-calibrated margin override.
    calibrated: bool = False
    #: Roots using calibrated margins (sorted).
    calibrated_roots: tuple[InstrumentId, ...] = ()

    def __post_init__(self) -> None:
        for name in ("initial_margin", "maintenance_margin"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        multiplier = float(self.stress_multiplier)
        if not math.isfinite(multiplier) or multiplier < 1.0:
            raise ValueError("stress_multiplier must be finite and >= 1.0")
        object.__setattr__(self, "stress_multiplier", multiplier)
        object.__setattr__(self, "stress_roots", tuple(sorted(self.stress_roots)))
        object.__setattr__(self, "calibrated_roots", tuple(sorted(self.calibrated_roots)))
        object.__setattr__(self, "stress_applied", bool(self.stress_applied))
        object.__setattr__(self, "calibrated", bool(self.calibrated))

    @property
    def utilisation(self) -> float:
        """Maintenance-to-initial ratio (``0`` when no margin is required)."""
        if self.initial_margin <= 0.0:
            return 0.0
        return self.maintenance_margin / self.initial_margin

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable representation (stress/calibration provenance)."""
        return {
            "initial_margin": self.initial_margin,
            "maintenance_margin": self.maintenance_margin,
            "utilisation": self.utilisation,
            "stress_applied": self.stress_applied,
            "stress_multiplier": self.stress_multiplier,
            "stress_roots": list(self.stress_roots),
            "calibrated": self.calibrated,
            "calibrated_roots": list(self.calibrated_roots),
        }


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
        stress_margin_multiplier: float = DEFAULT_STRESS_MARGIN_MULTIPLIER,
        stress_vol_threshold: float = DEFAULT_STRESS_VOL_THRESHOLD,
        calibrated_margins: Mapping[str, CalibratedMargin] | None = None,
        calibrated_costs_path: Path | str | None = None,
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
        if not math.isfinite(stress_margin_multiplier) or stress_margin_multiplier < 1.0:
            raise ValueError("stress_margin_multiplier must be finite and >= 1.0")
        if not math.isfinite(stress_vol_threshold) or stress_vol_threshold <= 0.0:
            raise ValueError("stress_vol_threshold must be finite and strictly positive")
        self._instrument_master = instrument_master
        self.maintenance_ratio = float(maintenance_ratio)
        self.penalty_ticks = float(penalty_ticks)
        self.penalty_fee_per_contract = float(penalty_fee_per_contract)
        self.margin_call_buffer = float(margin_call_buffer)
        self.stress_margin_multiplier = float(stress_margin_multiplier)
        self.stress_vol_threshold = float(stress_vol_threshold)
        self.calibrated_costs_path = (
            Path(calibrated_costs_path) if calibrated_costs_path is not None else None
        )
        if calibrated_margins is None:
            calibrated_margins = load_calibrated_margins(self.calibrated_costs_path)
        self.calibrated_margins: Mapping[str, CalibratedMargin] = MappingProxyType(
            {str(root).strip().upper(): margin for root, margin in calibrated_margins.items()}
        )
        self._volatilities: Mapping[InstrumentId, float] = MappingProxyType({})
        self._last_requirements: MarginRequirements = MarginRequirements()
        self._last_stress_roots: tuple[str, ...] = ()
        self._last_calibrated_roots: tuple[str, ...] = ()

    # -- stress/calibration context ------------------------------------------

    def set_volatilities(self, volatilities: Mapping[InstrumentId, float]) -> None:
        """Point-in-time annualised per-root volatilities used by stress margins.

        The event/vectorised backtester sets this once per bar from the
        ``MarketState`` (point-in-time, no look-ahead). Roots missing/with
        non-finite volatility are treated as normal regime.
        """
        cleaned: dict[InstrumentId, float] = {}
        for root, value in volatilities.items():
            number = float(value)
            if math.isfinite(number) and number >= 0.0:
                cleaned[str(root).strip().upper()] = number
        self._volatilities = MappingProxyType(cleaned)

    @property
    def calibrated_margin_count(self) -> int:
        """Number of roots with an applied IBKR-calibrated margin override."""
        return len(self.calibrated_margins)

    def reload_calibrated_margins(self) -> int:
        """Reload the calibration file (e.g. after task-85 whatIf calibration).

        Returns the number of roots with usable margin overrides; absent/corrupt/
        stale files cleanly fall back to an empty mapping (YAML margins).
        """
        self.calibrated_margins = MappingProxyType(
            load_calibrated_margins(self.calibrated_costs_path)
        )
        return len(self.calibrated_margins)

    @property
    def last_requirements(self) -> MarginRequirements:
        """Most recent margin requirements (stress/calibration provenance)."""
        return self._last_requirements

    def _apply_margin_regime(
        self,
        instrument_id: InstrumentId,
        initial: float,
        maintenance: float,
    ) -> tuple[float, float, bool, bool]:
        """Apply calibrated override first, then the stress multiplier.

        Returns ``(initial, maintenance, calibrated, stress)``; calibrated JSON
        values are authoritative for the root and the stress multiplier is applied
        on top of them when the root's point-in-time annualised volatility exceeds
        ``stress_vol_threshold``. Normal-regime, uncalibrated roots are unchanged.
        """
        calibrated = False
        override = self.calibrated_margins.get(instrument_id)
        if override is not None:
            initial = override.initial_margin
            maintenance = override.maintenance_margin
            calibrated = True
        stressed = False
        volatility = self._volatilities.get(instrument_id)
        if (
            volatility is not None
            and volatility >= self.stress_vol_threshold
            and self.stress_margin_multiplier > 1.0
        ):
            initial *= self.stress_margin_multiplier
            maintenance *= self.stress_margin_multiplier
            stressed = True
        return initial, maintenance, calibrated, stressed

    def effective_margin(
        self, instrument_id: InstrumentId, quantity: int = 1
    ) -> tuple[float, float]:
        """Effective ``(initial, maintenance)`` margin for one root/quantity.

        Uses the calibrated override when available, the instrument-master margin
        otherwise, then applies the stress multiplier from the current volatility
        context. Used by the vectorised margin estimate.
        """
        initial = 0.0
        maintenance = 0.0
        master = self._instrument_master
        if master is not None and master.has(instrument_id):
            spec = master.get(instrument_id)
            initial = float(spec.margins.initial)
            maintenance = float(spec.margins.maintenance)
        initial, maintenance, _calibrated, _stressed = self._apply_margin_regime(
            instrument_id, initial, maintenance
        )
        quantity_abs = abs(int(quantity))
        return initial * quantity_abs, maintenance * quantity_abs

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
        initial, maintenance, _calibrated, _stressed = self._apply_margin_regime(
            position.instrument_id, initial, maintenance
        )
        return initial, maintenance

    # -- margin arithmetic ----------------------------------------------------

    def margin_requirements(self, positions: Mapping[InstrumentId, Position]) -> MarginRequirements:
        """Total initial and maintenance margin for ``positions``."""
        self._validate_positions(positions)
        initial = 0.0
        maintenance = 0.0
        stress_roots: list[str] = []
        calibrated_roots: list[str] = []
        for position in positions.values():
            if position.quantity == 0:
                continue
            raw_initial = float(position.initial_margin)
            raw_maintenance = float(position.maintenance_margin)
            master = self._instrument_master
            if (
                (raw_initial <= 0.0 or raw_maintenance <= 0.0)
                and master is not None
                and master.has(position.instrument_id)
            ):
                spec = master.get(position.instrument_id)
                if raw_initial <= 0.0:
                    raw_initial = float(spec.margins.initial)
                if raw_maintenance <= 0.0:
                    raw_maintenance = float(spec.margins.maintenance)
            if (
                position.instrument_id in self.calibrated_margins
                or position.instrument_id in self._volatilities
            ):
                _initial, _maintenance, calibrated, stressed = self._apply_margin_regime(
                    position.instrument_id, raw_initial, raw_maintenance
                )
                if calibrated:
                    calibrated_roots.append(position.instrument_id)
                if stressed:
                    stress_roots.append(position.instrument_id)
            per_contract_initial, per_contract_maintenance = self._margin_per_contract(position)
            quantity = abs(position.quantity)
            initial += per_contract_initial * quantity
            maintenance += per_contract_maintenance * quantity
        requirements = MarginRequirements(
            initial_margin=initial,
            maintenance_margin=maintenance,
            stress_applied=bool(stress_roots),
            stress_multiplier=self.stress_margin_multiplier if stress_roots else 1.0,
            stress_roots=tuple(stress_roots),
            calibrated=bool(calibrated_roots),
            calibrated_roots=tuple(calibrated_roots),
        )
        self._last_requirements = requirements
        self._last_stress_roots = tuple(sorted(set(stress_roots)))
        self._last_calibrated_roots = tuple(sorted(set(calibrated_roots)))
        return requirements

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
            "stress_margin_multiplier": self.stress_margin_multiplier,
            "stress_vol_threshold": self.stress_vol_threshold,
            "calibrated_margin_roots": self.calibrated_margin_count,
            "calibrated_costs_path": (
                str(self.calibrated_costs_path) if self.calibrated_costs_path is not None else None
            ),
            "last_requirements": self._last_requirements.to_dict(),
        }
