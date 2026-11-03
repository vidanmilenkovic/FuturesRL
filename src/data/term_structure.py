"""Deterministic synthetic futures term structure for offline research.

The offline/DRL training path has no vendor second-month (F2) quotes, so
``FeatureProcessor`` cannot derive ``term_slope = (F2 - F1) / F1`` and
``carry_annualized = -term_slope * 365 / days`` from a single synthetic series.
Historically that gap was zero-filled (``train_drl`` reported non-empty
``feature_fallbacks``). This module supplies a *deterministic*, documented
synthetic F1/F2 curve so the offline pipeline produces realistic, non-zero —
and clearly synthetic — term structure and carry:

* every root has a base **annualised roll yield for a long** (positive =
  backwardation, negative = contango) taken from its asset class (equity/rates/
  metals in contango, energy in backwardation, FX near flat), overridable per
  root;
* a slow deterministic *regime* cycle (calendar-time based, no randomness and
  no future data) modulates that carry, so slopes are root-specific and evolve
  across regimes instead of being a constant;
* the expiry pair comes from the instrument master's delivery cycle
  (``configs/instruments.yaml``) with the configured calendar roll buffer
  (``days_before_fnd`` plus a physical-delivery safety buffer), so the synthetic
  front/second pair rolls before expiry exactly like the local roll engine and a
  roll decision can never depend on future prices.

Sign convention (identical to :mod:`src.data.continuous_roller` and
:mod:`src.data.feature_processor`)::

    term_slope       = (F2 - F1) / F1
    carry_annualized = -term_slope * 365 / days_between_expiries

Contango therefore means ``F2 > F1`` (positive ``term_slope``, negative
``carry_annualized``); backwardation means ``F2 < F1`` (negative
``term_slope``, positive ``carry_annualized``).

Everything here is **synthetic research data** (contract ids stay ``_SYN``):
never use these curves as a vendor-data substitute in paper/live reporting.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Literal

from configs.instruments import InstrumentsConfig, InstrumentSpec, load_instruments

from src.core.types import InstrumentId, ensure_utc
from src.data import schema as s
from src.data.contracts import next_contract_expiration

__all__ = [
    "CarryProfile",
    "SyntheticTermStructure",
    "TermRegime",
    "TermStructureCurve",
    "offline_term_structure_metadata",
]

#: Curve regime implied by the sign of the annualised roll yield.
type TermRegime = Literal["contango", "backwardation", "neutral"]

#: Annualised roll yield for a long position per root (positive = backwardation).
BASE_ANNUALIZED_CARRY: Mapping[str, float] = {
    # equity index (dividend/financing basis -> mild contango)
    "ES": -0.0040,
    "MES": -0.0040,
    "NQ": -0.0030,
    "MNQ": -0.0030,
    "YM": -0.0040,
    "MYM": -0.0040,
    "RTY": -0.0030,
    "M2K": -0.0030,
    "NKD": -0.0040,
    # rates (coupon carry -> contango, SOFR near flat)
    "ZT": -0.0040,
    "ZF": -0.0040,
    "ZN": -0.0070,
    "TN": -0.0080,
    "ZB": -0.0090,
    "UB": -0.0100,
    "SR3": -0.0020,
    # energy (crude/nat-gas backwardated, refined products contango)
    "CL": 0.0300,
    "MCL": 0.0300,
    "NG": 0.0400,
    "MNG": 0.0400,
    "RB": -0.0100,
    "HO": -0.0080,
    # metals (gold/silver/platinum contango, copper backwardated)
    "GC": -0.0100,
    "MGC": -0.0100,
    "SI": -0.0120,
    "SIL": -0.0120,
    "HG": 0.0100,
    "MHG": 0.0100,
    "PL": -0.0080,
    "PA": -0.0060,
    # fx (interest-rate differential basis; mixed signs)
    "6E": 0.0020,
    "6J": -0.0200,
    "6B": 0.0050,
    "6A": -0.0100,
    "6C": 0.0050,
    "6S": -0.0050,
    # agriculture (storage carry -> contango; livestock live-curve contango)
    "ZC": -0.0300,
    "ZS": -0.0250,
    "ZW": -0.0280,
    "ZM": -0.0200,
    "ZL": -0.0180,
    "LE": -0.0120,
    "HE": -0.0200,
    # crypto (spot/futures basis -> contango)
    "MBT": -0.0500,
    "MET": -0.0500,
}

#: Asset-class fallback when a root is not in :data:`BASE_ANNUALIZED_CARRY`.
ASSET_CLASS_ANNUALIZED_CARRY: Mapping[str, float] = {
    "equity_index": -0.0040,
    "energy": 0.0300,
    "metals": -0.0100,
    "rates": -0.0070,
    "fx": 0.0020,
    "agriculture": -0.0050,
    "crypto": 0.0000,
    "unknown": -0.0040,
}

#: Deterministic regime multipliers applied to the base carry (calendar-cycled).
REGIME_CARRY_SCALES: tuple[float, ...] = (1.25, 0.55, -0.45, 0.85)

#: Default length of one synthetic carry regime, in calendar days.
DEFAULT_REGIME_SPAN_DAYS: int = 90
#: Default length of the smooth carry modulation cycle, in calendar days.
DEFAULT_CYCLE_DAYS: float = 180.0
#: Default amplitude of the smooth carry modulation (fraction of base carry).
DEFAULT_CARRY_AMPLITUDE: float = 0.35
#: Safety floor so a synthetic slope is never exactly zero (non-degenerate features).
MIN_ABS_ANNUALIZED_CARRY: float = 1e-4
#: Absolute carry below which the implied regime is reported as ``neutral``.
_NEUTRAL_CARRY_BAND: float = 1e-4
#: Extra physical-delivery roll buffer (calendar days) matching the roll engine.
PHYSICAL_DELIVERY_ROLL_BUFFER_DAYS: int = 7
#: Default roll buffer for roots missing from the instrument master.
DEFAULT_ROLL_DAYS: int = 5
#: Default delivery cycle for roots missing from the instrument master.
DEFAULT_CONTRACT_MONTHS: tuple[int, ...] = (3, 6, 9, 12)
#: Deterministic seconds-per-day used for the calendar arithmetic.
_SECONDS_PER_DAY = 86_400.0


@dataclass(frozen=True, slots=True)
class CarryProfile:
    """Deterministic synthetic carry configuration of one futures root."""

    root: str
    base_annualized_carry: float
    regime_amplitude: float
    cycle_phase: float
    roll_days: int
    contract_months: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TermStructureCurve:
    """One point of the synthetic F1/F2 curve at a decision timestamp."""

    root: str
    timestamp: datetime
    front_expiration: date
    second_expiration: date
    front_price: float
    second_price: float
    term_slope: float
    carry_annualized: float
    regime: TermRegime

    @property
    def days_between_expiries(self) -> int:
        """Calendar days between the front and second expiration (>= 1)."""
        return max((self.second_expiration - self.front_expiration).days, 1)

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable representation."""
        return {
            "root": self.root,
            "timestamp": self.timestamp.isoformat(),
            "front_expiration": self.front_expiration.isoformat(),
            "second_expiration": self.second_expiration.isoformat(),
            "front_price": self.front_price,
            "second_price": self.second_price,
            "term_slope": self.term_slope,
            "carry_annualized": self.carry_annualized,
            "regime": self.regime,
        }


class SyntheticTermStructure:
    """Deterministic synthetic F1/F2 builder shared by offline data consumers.

    Parameters
    ----------
    instrument_master:
        Instrument master supplying the delivery cycle / roll buffer. Loaded from
        ``configs/instruments.yaml`` lazily when omitted.
    regime_span_days, cycle_days, amplitude:
        Calendar-time regime controls. They are *deterministic functions of the
        timestamp* (no random draws, no state, no future bars), so replaying the
        same window always yields the same curves.
    """

    def __init__(
        self,
        *,
        instrument_master: InstrumentsConfig | None = None,
        regime_span_days: int = DEFAULT_REGIME_SPAN_DAYS,
        cycle_days: float = DEFAULT_CYCLE_DAYS,
        amplitude: float = DEFAULT_CARRY_AMPLITUDE,
    ) -> None:
        if regime_span_days < 1:
            raise ValueError("regime_span_days must be >= 1")
        if cycle_days <= 0.0:
            raise ValueError("cycle_days must be strictly positive")
        if not 0.0 <= amplitude <= 1.0:
            raise ValueError("amplitude must be within [0, 1]")
        self._master = instrument_master
        self._regime_span_days = regime_span_days
        self._cycle_days = cycle_days
        self._amplitude = amplitude

    # -- introspection ---------------------------------------------------------

    @property
    def instrument_master(self) -> InstrumentsConfig:
        """Instrument master used for delivery cycles and roll buffers."""
        if self._master is None:
            self._master = load_instruments()
        return self._master

    @property
    def regime_span_days(self) -> int:
        """Calendar length of one deterministic carry regime."""
        return self._regime_span_days

    @property
    def cycle_days(self) -> float:
        """Length of the smooth carry modulation cycle."""
        return self._cycle_days

    # -- per-root configuration ------------------------------------------------

    def _spec_or_none(self, root: InstrumentId) -> InstrumentSpec | None:
        master = self.instrument_master
        normalised = root.strip().upper()
        return master.get(normalised) if master.has(normalised) else None

    def _base_carry(self, root: InstrumentId, spec: InstrumentSpec | None) -> float:
        normalised = root.strip().upper()
        if normalised in BASE_ANNUALIZED_CARRY:
            return BASE_ANNUALIZED_CARRY[normalised]
        asset_class = str(spec.asset_class) if spec is not None else "unknown"
        return ASSET_CLASS_ANNUALIZED_CARRY.get(
            asset_class, ASSET_CLASS_ANNUALIZED_CARRY["unknown"]
        )

    @staticmethod
    def _phase(root: InstrumentId) -> float:
        """Stable per-root phase offset in radians (no RNG, order-independent)."""
        checksum = sum((index + 1) * ord(char) for index, char in enumerate(root.strip().upper()))
        return math.radians(checksum % 360)

    def carry_profile(self, root: InstrumentId) -> CarryProfile:
        """Deterministic carry configuration for ``root``.

        A paired micro contract shares its standard contract's underlying curve
        (base carry + cycle phase), so ``P_micro == P_standard`` also holds for
        the synthetic second-month leg. The micro keeps its own delivery cycle /
        roll-buffer configuration (aligned by the instrument master).
        """
        normalised = root.strip().upper()
        if not normalised:
            raise ValueError("root must be a non-empty string")
        spec = self._spec_or_none(normalised)
        family_root = normalised
        family_spec = spec
        if spec is not None and spec.standard_root is not None:
            family_root = spec.standard_root
            family_spec = self._spec_or_none(family_root)
        roll_days = (
            spec.roll.days_before_fnd + PHYSICAL_DELIVERY_ROLL_BUFFER_DAYS
            if spec is not None and spec.roll.physical_delivery
            else (spec.roll.days_before_fnd if spec is not None else DEFAULT_ROLL_DAYS)
        )
        months = tuple(spec.roll.contract_months) if spec is not None else DEFAULT_CONTRACT_MONTHS
        return CarryProfile(
            root=normalised,
            base_annualized_carry=self._base_carry(family_root, family_spec),
            regime_amplitude=self._amplitude,
            cycle_phase=self._phase(family_root),
            roll_days=roll_days,
            contract_months=months,
        )

    def roll_days(self, root: InstrumentId) -> int:
        """Calendar roll buffer (days before expiry) for ``root``."""
        return self.carry_profile(root).roll_days

    def contract_months(self, root: InstrumentId) -> tuple[int, ...]:
        """Declared delivery cycle months for ``root``."""
        return self.carry_profile(root).contract_months

    # -- curve math ------------------------------------------------------------

    @staticmethod
    def _elapsed_days(timestamp: datetime) -> float:
        """Absolute epoch day count of ``timestamp`` (deterministic calendar input)."""
        return ensure_utc(timestamp).timestamp() / _SECONDS_PER_DAY

    def annualized_carry_at(self, root: InstrumentId, timestamp: datetime) -> float:
        """Effective annualised roll yield for a long at ``timestamp``.

        Positive = backwardation, negative = contango. Deterministic: a discrete
        calendar regime multiplier times a smooth per-root cycle; the result is
        floored away from exact zero so the derived features are never degenerate.
        """
        profile = self.carry_profile(root)
        elapsed = self._elapsed_days(timestamp)
        regime_index = int(elapsed // self._regime_span_days) % len(REGIME_CARRY_SCALES)
        regime_scale = REGIME_CARRY_SCALES[regime_index]
        modulation = 1.0 + profile.regime_amplitude * math.sin(
            profile.cycle_phase + 2.0 * math.pi * elapsed / self._cycle_days
        )
        carry = profile.base_annualized_carry * regime_scale * modulation
        if abs(carry) < MIN_ABS_ANNUALIZED_CARRY:
            carry = math.copysign(MIN_ABS_ANNUALIZED_CARRY, profile.base_annualized_carry or 1.0)
        return carry

    def regime_at(self, root: InstrumentId, timestamp: datetime) -> TermRegime:
        """Regime implied by the effective annualised carry at ``timestamp``."""
        carry = self.annualized_carry_at(root, timestamp)
        if carry > _NEUTRAL_CARRY_BAND:
            return "backwardation"
        if carry < -_NEUTRAL_CARRY_BAND:
            return "contango"
        return "neutral"

    def curve(
        self,
        root: InstrumentId,
        timestamp: datetime,
        *,
        front_price: float,
        front_expiration: date,
        second_expiration: date,
    ) -> TermStructureCurve:
        """Build the synthetic F1/F2 curve point for one front price."""
        if second_expiration <= front_expiration:
            raise ValueError(
                "second_expiration must be after front_expiration, got "
                f"{front_expiration.isoformat()} -> {second_expiration.isoformat()}"
            )
        if not math.isfinite(front_price) or front_price <= 0.0:
            raise ValueError(f"front_price must be a finite positive number, got {front_price!r}")
        moment = ensure_utc(timestamp)
        days = max((second_expiration - front_expiration).days, 1)
        carry = self.annualized_carry_at(root, moment)
        term_slope = -carry * (float(days) / 365.0)
        return TermStructureCurve(
            root=root.strip().upper(),
            timestamp=moment,
            front_expiration=front_expiration,
            second_expiration=second_expiration,
            front_price=float(front_price),
            second_price=float(front_price) * (1.0 + term_slope),
            term_slope=term_slope,
            carry_annualized=carry,
            regime=self.regime_at(root, moment),
        )

    # -- expiry chain / roll-aware pair selection ------------------------------

    def expiration_chain(
        self,
        root: InstrumentId,
        *,
        reference: datetime,
        horizon_days: int,
    ) -> tuple[date, ...]:
        """Sorted expiry chain covering ``horizon_days`` from ``reference``.

        Uses only the delivery cycle and the reference date — never prices — so
        the synthetic roll schedule cannot look ahead.
        """
        if horizon_days < 0:
            raise ValueError("horizon_days must be non-negative")
        moment = ensure_utc(reference)
        reference_date = moment.date()
        months = self.contract_months(root)
        limit = reference_date + timedelta(days=horizon_days)
        chain: list[date] = []
        expiration = next_contract_expiration(months, reference_date)
        while len(chain) < 2 or expiration <= limit:
            chain.append(expiration)
            expiration = next_contract_expiration(months, expiration)
            if len(chain) >= 128:  # pragma: no cover - defensive, cycles are monthly+
                break
        return tuple(chain)

    def pair_at(
        self, root: InstrumentId, timestamp: datetime, chain: Sequence[date]
    ) -> tuple[date, date] | None:
        """Front/second expiry pair at ``timestamp`` (calendar roll aware)."""
        if len(chain) < 2:
            return None
        moment = ensure_utc(timestamp)
        roll_days = self.roll_days(root)
        for index in range(len(chain) - 1):
            expiration = chain[index]
            roll_threshold = datetime.combine(
                expiration - timedelta(days=roll_days), time.min, tzinfo=UTC
            )
            if moment < roll_threshold:
                return (expiration, chain[index + 1])
        return (chain[-2], chain[-1])


# ---------------------------------------------------------------------------
# Public helper for training/API metadata
# ---------------------------------------------------------------------------


def offline_term_structure_metadata(
    roots: Sequence[InstrumentId],
    *,
    start: datetime | None = None,
    periods: int = 256,
    instrument_master: InstrumentsConfig | None = None,
    model: SyntheticTermStructure | None = None,
) -> dict[str, dict[str, object]]:
    """Per-root deterministic offline term-structure metadata (JSON-friendly).

    This is the data-side helper consumers (e.g. the DRL trainer model metadata
    or the Strategy Studio) can import to describe the synthetic curve the
    offline pipeline used: front/second expirations, configured roll buffer,
    base/effective annualised carry and sampled min/mean/max ``term_slope`` /
    ``carry_annualized`` plus the set of regimes seen over the sample window.

    The values are derived purely from the calendar and the instrument master, so
    repeated calls for the same arguments are identical.
    """
    if not roots:
        raise ValueError("at least one futures root is required")
    moment = ensure_utc(start) if start is not None else datetime(2024, 1, 2, 21, 0, tzinfo=UTC)
    if periods < 2:
        raise ValueError("periods must be >= 2")
    structure = (
        model if model is not None else SyntheticTermStructure(instrument_master=instrument_master)
    )
    sampled_steps = max(1, min(periods, 64))
    step_size = max(1, periods // sampled_steps)
    summary: dict[str, dict[str, object]] = {}
    for root in s.validate_roots(roots):
        profile = structure.carry_profile(root)
        chain = structure.expiration_chain(root, reference=moment, horizon_days=periods + 400)
        slopes: list[float] = []
        carries: list[float] = []
        regimes: set[str] = set()
        for step in range(0, periods, step_size):
            timestamp = moment + timedelta(days=step)
            pair = structure.pair_at(root, timestamp, chain)
            if pair is None:  # pragma: no cover - chain always has >= 2 expiries
                continue
            curve = structure.curve(
                root,
                timestamp,
                front_price=100.0,
                front_expiration=pair[0],
                second_expiration=pair[1],
            )
            slopes.append(curve.term_slope)
            carries.append(curve.carry_annualized)
            regimes.add(curve.regime)
        summary[profile.root] = {
            "synthetic": True,
            "front_expiration": chain[0].isoformat(),
            "second_expiration": chain[1].isoformat(),
            "roll_days": profile.roll_days,
            "contract_months": list(profile.contract_months),
            "base_annualized_carry": profile.base_annualized_carry,
            "mean_term_slope": float(sum(slopes) / len(slopes)) if slopes else 0.0,
            "min_term_slope": min(slopes) if slopes else 0.0,
            "max_term_slope": max(slopes) if slopes else 0.0,
            "mean_carry_annualized": float(sum(carries) / len(carries)) if carries else 0.0,
            "regimes": tuple(sorted(regimes)),
            "regime_span_days": structure.regime_span_days,
        }
    return summary
