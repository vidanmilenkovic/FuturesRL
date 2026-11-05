"""Continuous futures roll engine: roll schedules + price adjustment.

Training DRL/ML models on raw expiring contracts injects artificial price jumps
at every roll. This module implements the three roll-trigger methodologies and
the three price-adjustment methodologies required by FuturesRL.

Roll triggers
-------------
``VOLUME_OI_CROSSOVER`` (:class:`VolumeOICrossoverRollSchedule`)
    Roll at the first bar where the next contract's volume (and optionally OI)
    exceeds the front contract's by a configurable ratio.

``CALENDAR_DAYS_BEFORE_FND`` (:class:`CalendarRollSchedule`)
    Roll ``N`` calendar days before the front contract's last trade date (LTD),
    which is the only expiry information available in the canonical bar schema.
    Physical-delivery roots (``CL``, ``GC``, ``ZC``, ...) roll *earlier* by an
    extra safety buffer so the strategy can never be forced into delivery.

``DATABENTO_NATIVE`` (:class:`DatabentoNativeRollSchedule`)
    The vendor decides the roll; the local calendar becomes a *cross-check*.
    A native continuous frame (``ES.v.0`` volume-based / ``ES.c.0`` calendar
    based) is scanned for contract-identity changes and compared with the local
    schedule (:class:`NativeRollCrossCheck`).

Price adjustments
-----------------
Let roll ``i`` switch from front contract ``F1`` to next contract ``F2`` at
roll timestamp ``t_i`` with *simultaneous* prices ``f1_i = F1(t_i)``,
``f2_i = F2(t_i)`` and roll gap ``g_i = f2_i - f1_i`` (the calendar spread).

* **PANAMA** (back-adjusted additive), anchored to the current front contract::

      adjustment(t) = sum_{i: t_i > t} g_i
      adjusted_close(t) = close(t) + adjustment(t)

  Positive ``g_i`` means contango (next contract above front). Absolute point
  moves and dollar PnL per contract are preserved; the *most recent* prices are
  unadjusted.
* **PROPORTIONAL** (ratio-adjusted multiplicative), same anchoring::

      ratio_factor(t) = prod_{i: t_i > t} (f1_i / f2_i)
      adjusted_close(t) = close(t) * ratio_factor(t)

  Percentage/log-return features are preserved (the residual is the intra-roll
  calendar-spread drift, which is real information rather than a synthetic jump).
* **UNADJUSTED**: raw front price in ``close``/``adjusted_close`` plus explicit
  ``next_close``, ``term_slope = (F2 - F1) / F1`` and
  ``carry_annualized = -term_slope * 365 / days_between_expiries``. This is the
  series to use for point-in-time margin/notional accounting.

Zero look-ahead
---------------
Roll decisions depend only on volume/open-interest/expiry information and data
available at or before the decision timestamp ``t`` (no price look-ahead): a
change to *future* prices can never move an earlier roll. Back-adjustment
necessarily uses the *later* roll gaps by construction (that is what anchoring
means); point-in-time consumers must therefore use the unadjusted series.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Protocol, runtime_checkable

import polars as pl
from configs.instruments import InstrumentsConfig, InstrumentSpec, load_instruments

from src.core.types import InstrumentId, PriceAdjustment, RollEvent, RollTrigger
from src.data import schema as s
from src.data.contracts import parse_contract_symbol

__all__ = [
    "CalendarRollSchedule",
    "ContinuousContractBuilder",
    "DatabentoNativeRollSchedule",
    "NativeRollCrossCheck",
    "RollScheduleCalculator",
    "VolumeOICrossoverRollSchedule",
]


# ---------------------------------------------------------------------------
# Internal panel: one root's dated contracts indexed by timestamp
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _PanelRow:
    """One raw bar of one specific contract."""

    timestamp: datetime
    expiration: date
    instrument_id: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    open_interest: float


@dataclass(slots=True)
class _RootPanel:
    """Per-root lookup tables shared by every roll schedule and the builder."""

    root: str
    expirations: tuple[date, ...]
    timestamps: tuple[datetime, ...]
    rows: dict[date, dict[datetime, _PanelRow]]
    available: dict[datetime, tuple[date, ...]]

    def row(self, expiration: date, timestamp: datetime) -> _PanelRow | None:
        """Bar of ``expiration`` at ``timestamp`` (``None`` when absent)."""
        return self.rows.get(expiration, {}).get(timestamp)

    def close(self, expiration: date, timestamp: datetime) -> float | None:
        """Close of ``expiration`` at ``timestamp`` (``None`` when absent)."""
        row = self.row(expiration, timestamp)
        return row.close if row is not None else None

    def common_timestamps(self, left: date, right: date) -> tuple[datetime, ...]:
        """Timestamps where both contracts have a bar, ascending."""
        left_rows = self.rows.get(left, {})
        right_rows = self.rows.get(right, {})
        return tuple(sorted(set(left_rows) & set(right_rows)))

    def front_at(self, timestamp: datetime) -> date | None:
        """Nearest-expiration contract trading at ``timestamp``."""
        available = self.available.get(timestamp, ())
        return available[0] if available else None

    def next_after(self, timestamp: datetime, expiration: date) -> date | None:
        """Smallest expiration strictly after ``expiration`` trading at ``timestamp``."""
        for candidate in self.available.get(timestamp, ()):
            if candidate > expiration:
                return candidate
        return None


def _build_panel(root: str, bars: pl.DataFrame) -> _RootPanel:
    """Build the :class:`_RootPanel` for ``root`` from a canonical bar frame."""
    normalised_root = root.strip().upper()
    if not normalised_root:
        raise ValueError("root must be a non-empty string")
    required = (
        s.TIMESTAMP_COLUMN,
        s.ROOT_SYMBOL_COLUMN,
        s.EXPIRATION_COLUMN,
        s.CLOSE_COLUMN,
        s.VOLUME_COLUMN,
        s.OPEN_INTEREST_COLUMN,
    )
    missing = [column for column in required if column not in bars.columns]
    if missing:
        raise s.SchemaError(
            f"continuous roller needs raw multi-expiration bars; missing columns {missing}"
        )
    frame = bars.filter(pl.col(s.ROOT_SYMBOL_COLUMN).cast(pl.Utf8) == normalised_root)
    if frame.is_empty():
        raise ValueError(f"no bars found for root {normalised_root!r}")
    frame = frame.sort([s.TIMESTAMP_COLUMN, s.EXPIRATION_COLUMN])
    rows: dict[date, dict[datetime, _PanelRow]] = {}
    available: dict[datetime, list[date]] = {}
    for record in frame.iter_rows(named=True):
        timestamp = record[s.TIMESTAMP_COLUMN]
        expiration = record[s.EXPIRATION_COLUMN]
        if not isinstance(timestamp, datetime) or not isinstance(expiration, date):
            raise s.SchemaError("bar frame timestamp/expiration columns have invalid dtypes")
        panel_row = _PanelRow(
            timestamp=timestamp,
            expiration=expiration,
            instrument_id=str(record[s.INSTRUMENT_ID_COLUMN]),
            open=float(record[s.OPEN_COLUMN]),
            high=float(record[s.HIGH_COLUMN]),
            low=float(record[s.LOW_COLUMN]),
            close=float(record[s.CLOSE_COLUMN]),
            volume=float(record[s.VOLUME_COLUMN]),
            open_interest=float(record[s.OPEN_INTEREST_COLUMN]),
        )
        rows.setdefault(expiration, {})[timestamp] = panel_row
        available.setdefault(timestamp, []).append(expiration)
    expirations = tuple(sorted(rows))
    timestamps = tuple(sorted(available))
    frozen_available = {timestamp: tuple(sorted(values)) for timestamp, values in available.items()}
    return _RootPanel(
        root=normalised_root,
        expirations=expirations,
        timestamps=timestamps,
        rows=rows,
        available=frozen_available,
    )


# ---------------------------------------------------------------------------
# Roll schedule protocol + calculators
# ---------------------------------------------------------------------------


@runtime_checkable
class RollScheduleCalculator(Protocol):
    """A roll-trigger methodology producing explicit :class:`RollEvent`s."""

    @property
    def trigger(self) -> RollTrigger:
        """RollTrigger methodology implemented by this calculator."""
        ...

    def compute(self, root: InstrumentId, bars: pl.DataFrame) -> tuple[RollEvent, ...]:
        """Return the roll events for ``root`` ordered by roll timestamp."""
        ...


class _BaseRollSchedule(ABC):
    """Shared roll-event assembly for the bar-driven schedules."""

    _trigger: RollTrigger = RollTrigger.CALENDAR_DAYS_BEFORE_FND

    def __init__(self, instrument_master: InstrumentsConfig | None = None) -> None:
        self._master = instrument_master

    @property
    def trigger(self) -> RollTrigger:
        """RollTrigger methodology implemented by this calculator."""
        return self._trigger

    @property
    def instrument_master(self) -> InstrumentsConfig:
        """Instrument master used for per-root roll configuration."""
        if self._master is None:
            self._master = load_instruments()
        return self._master

    def _spec_or_none(self, root: InstrumentId) -> InstrumentSpec | None:
        master = self.instrument_master
        return master.get(root) if master.has(root) else None

    @abstractmethod
    def _select_roll_datetime(
        self, panel: _RootPanel, front: date, next_contract: date
    ) -> datetime | None:
        """Return the roll timestamp for a front -> next contract pair."""

    def compute(self, root: InstrumentId, bars: pl.DataFrame) -> tuple[RollEvent, ...]:
        """Return the ordered roll events for ``root`` (zero price look-ahead)."""
        panel = _build_panel(root, bars)
        if len(panel.expirations) < 2:
            return ()
        events: list[RollEvent] = []
        active = panel.expirations[0]
        last_roll = panel.timestamps[0] - timedelta(seconds=1)
        for candidate in panel.expirations[1:]:
            roll_datetime = self._select_roll_datetime(panel, active, candidate)
            if roll_datetime is None:
                # The candidate has no overlapping bars (e.g. a missing/halted
                # contract): skip it and keep looking for the next available one.
                continue
            if roll_datetime < last_roll:
                roll_datetime = last_roll
            from_price = panel.close(active, roll_datetime)
            to_price = panel.close(candidate, roll_datetime)
            if from_price is None or to_price is None:  # pragma: no cover - defensive
                continue
            events.append(
                RollEvent(
                    root_symbol=panel.root,
                    roll_datetime=roll_datetime,
                    from_expiration=active,
                    to_expiration=candidate,
                    trigger=self._trigger,
                    from_price=from_price,
                    to_price=to_price,
                )
            )
            last_roll = roll_datetime
            active = candidate
        return tuple(events)


class VolumeOICrossoverRollSchedule(_BaseRollSchedule):
    """Roll when the next contract's volume (and optionally OI) overtakes the front.

    The decision at timestamp ``t`` uses only the bars at ``t`` (already known
    and published at ``t``), so no future price or volume can move an earlier
    roll. When the two contracts never cross before the front contract stops
    trading, the last common bar is used as a continuity fallback.
    """

    _trigger = RollTrigger.VOLUME_OI_CROSSOVER

    def __init__(
        self,
        *,
        min_volume_ratio: float | None = None,
        min_oi_ratio: float = 1.0,
        require_both: bool = False,
        instrument_master: InstrumentsConfig | None = None,
    ) -> None:
        super().__init__(instrument_master)
        if min_volume_ratio is not None and min_volume_ratio <= 0:
            raise ValueError("min_volume_ratio must be strictly positive")
        if min_oi_ratio <= 0:
            raise ValueError("min_oi_ratio must be strictly positive")
        self._min_volume_ratio = min_volume_ratio
        self._min_oi_ratio = min_oi_ratio
        self._require_both = require_both

    def _volume_ratio(self, root: InstrumentId) -> float:
        if self._min_volume_ratio is not None:
            return self._min_volume_ratio
        spec = self._spec_or_none(root)
        return spec.roll.min_volume_ratio if spec is not None else 1.0

    def _select_roll_datetime(
        self, panel: _RootPanel, front: date, next_contract: date
    ) -> datetime | None:
        common = panel.common_timestamps(front, next_contract)
        if not common:
            return None
        volume_ratio = self._volume_ratio(panel.root)
        for timestamp in common:
            front_row = panel.rows[front][timestamp]
            next_row = panel.rows[next_contract][timestamp]
            volume_cross = next_row.volume > front_row.volume * volume_ratio
            oi_cross = next_row.open_interest > front_row.open_interest * self._min_oi_ratio
            if self._require_both:
                if volume_cross and oi_cross:
                    return timestamp
            elif volume_cross:
                return timestamp
        return common[-1]


class CalendarRollSchedule(_BaseRollSchedule):
    """Roll ``N`` calendar days before the front contract's expiration (LTD proxy).

    Physical-delivery roots roll earlier: the per-root ``days_before_fnd`` is
    extended by ``physical_buffer_days`` so the strategy can never be forced
    into physical delivery.
    """

    _trigger = RollTrigger.CALENDAR_DAYS_BEFORE_FND

    def __init__(
        self,
        *,
        days_before_expiration: int | None = None,
        physical_buffer_days: int = 7,
        instrument_master: InstrumentsConfig | None = None,
        physical_delivery: bool | None = None,
    ) -> None:
        super().__init__(instrument_master)
        if days_before_expiration is not None and days_before_expiration < 0:
            raise ValueError("days_before_expiration must be non-negative")
        if physical_buffer_days < 0:
            raise ValueError("physical_buffer_days must be non-negative")
        self._days_before = days_before_expiration
        self._physical_buffer_days = physical_buffer_days
        self._physical_override = physical_delivery

    def _days_before_for(self, spec: InstrumentSpec | None) -> int:
        if self._days_before is not None:
            base = self._days_before
        elif spec is not None:
            base = spec.roll.days_before_fnd
        else:
            base = 5
        physical = (
            self._physical_override
            if self._physical_override is not None
            else (spec.roll.physical_delivery if spec is not None else False)
        )
        return base + (self._physical_buffer_days if physical else 0)

    def _select_roll_datetime(
        self, panel: _RootPanel, front: date, next_contract: date
    ) -> datetime | None:
        common = panel.common_timestamps(front, next_contract)
        if not common:
            return None
        days_before = self._days_before_for(self._spec_or_none(panel.root))
        threshold = datetime.combine(front - timedelta(days=days_before), time.min, tzinfo=UTC)
        for timestamp in common:
            if timestamp >= threshold:
                return timestamp
        return common[-1]


class DatabentoNativeRollSchedule:
    """Roll timing dictated by Databento native continuous symbology.

    Databento publishes ``[ROOT].v.0`` (volume-based) and ``[ROOT].c.0``
    (calendar-based) continuous series whose underlying contract identity
    changes at the vendor's roll. This schedule scans such a frame and emits the
    observed roll events, which :meth:`cross_verify` compares against the local
    schedule (there is no local trigger — the vendor is the source of truth).
    """

    _trigger = RollTrigger.DATABENTO_NATIVE

    def __init__(
        self,
        native_frame: pl.DataFrame | None = None,
        *,
        instrument_master: InstrumentsConfig | None = None,
        tolerance_days: int = 2,
    ) -> None:
        if tolerance_days < 0:
            raise ValueError("tolerance_days must be non-negative")
        self._native_frame = native_frame
        self._master = instrument_master
        self._tolerance_days = tolerance_days

    @property
    def trigger(self) -> RollTrigger:
        """Native continuous symbology trigger."""
        return self._trigger

    @property
    def native_frame(self) -> pl.DataFrame | None:
        """Vendor continuous frame backing this schedule (``None`` when unset)."""
        return self._native_frame

    def compute(self, root: InstrumentId, bars: pl.DataFrame) -> tuple[RollEvent, ...]:
        """Return the vendor roll events stored on this schedule."""
        del bars  # the vendor frame already encodes the roll; bars are unused
        if self._native_frame is None:
            raise ValueError(
                "DatabentoNativeRollSchedule requires a native continuous frame "
                "(constructor argument `native_frame=`); load it with "
                "DatabentoLoader.ingest(..., stype_in='continuous') and pass it in."
            )
        return self.detect_rolls(root, self._native_frame)

    def detect_rolls(self, root: InstrumentId, native_frame: pl.DataFrame) -> tuple[RollEvent, ...]:
        """Detect contract switches in a native ``[ROOT].v.0``/``[ROOT].c.0`` frame."""
        normalised_root = root.strip().upper()
        frame = native_frame
        timestamp_column = s.TIMESTAMP_COLUMN if s.TIMESTAMP_COLUMN in frame.columns else None
        if timestamp_column is None:
            for candidate in ("ts_event", "ts_recv"):
                if candidate in frame.columns:
                    timestamp_column = candidate
                    break
        if timestamp_column is None:
            raise s.SchemaError("native continuous frame has no timestamp column")
        frame = s.normalise_timestamp_column(frame, timestamp_column)
        if s.ROOT_SYMBOL_COLUMN in frame.columns:
            frame = frame.filter(pl.col(s.ROOT_SYMBOL_COLUMN).cast(pl.Utf8) == normalised_root)
        identity_column = None
        for candidate in ("raw_symbol", "instrument_id", "contract_symbol", "symbol"):
            if candidate in frame.columns:
                identity_column = candidate
                break
        if identity_column is None:
            raise s.SchemaError(
                "native continuous frame needs a contract identity column "
                "(raw_symbol/instrument_id/contract_symbol/symbol)"
            )
        if s.CLOSE_COLUMN not in frame.columns:
            raise s.SchemaError(f"native continuous frame needs a {s.CLOSE_COLUMN!r} column")
        if frame.is_empty():
            return ()
        reference = frame[s.TIMESTAMP_COLUMN].max()
        if not isinstance(reference, datetime):  # pragma: no cover - schema guarantees it
            reference = datetime.now(UTC)
        ordered = frame.sort(s.TIMESTAMP_COLUMN)
        events: list[RollEvent] = []
        previous = ordered.row(0, named=True)
        for record in ordered.iter_rows(named=True):
            previous_identity = str(previous[identity_column])
            identity = str(record[identity_column])
            if identity == previous_identity:
                previous = record
                continue
            timestamp = record[s.TIMESTAMP_COLUMN]
            previous_parsed = parse_contract_symbol(
                previous_identity, roots=[normalised_root], reference=reference
            )
            current_parsed = parse_contract_symbol(
                identity, roots=[normalised_root], reference=reference
            )
            if previous_parsed is None or current_parsed is None:
                raise s.SchemaError(
                    f"cannot parse native continuous contract symbols "
                    f"{previous_identity!r} -> {identity!r} for root {normalised_root!r}"
                )
            from_expiration = previous_parsed[1]
            to_expiration = current_parsed[1]
            if to_expiration > from_expiration:
                events.append(
                    RollEvent(
                        root_symbol=normalised_root,
                        roll_datetime=timestamp,
                        from_expiration=from_expiration,
                        to_expiration=to_expiration,
                        trigger=self._trigger,
                        from_price=float(previous[s.CLOSE_COLUMN]),
                        to_price=float(record[s.CLOSE_COLUMN]),
                    )
                )
            previous = record
        return tuple(events)

    def cross_check(
        self,
        root: InstrumentId,
        local_rolls: tuple[RollEvent, ...],
        native_frame: pl.DataFrame | None = None,
    ) -> NativeRollCrossCheck:
        """Cross-verify the local schedule against the vendor continuous series."""
        frame = native_frame if native_frame is not None else self._native_frame
        if frame is None:
            raise ValueError("cross_check requires a native continuous frame")
        native_rolls = self.detect_rolls(root, frame)
        return NativeRollCrossCheck.cross_verify(
            root=root,
            local_rolls=local_rolls,
            native_rolls=native_rolls,
            tolerance_days=self._tolerance_days,
        )


@dataclass(frozen=True, slots=True)
class NativeRollCrossCheck:
    """Result of comparing a local roll schedule with Databento native rolls."""

    root: str
    local_rolls: tuple[RollEvent, ...]
    native_rolls: tuple[RollEvent, ...]
    matched: tuple[tuple[RollEvent, RollEvent], ...] = ()
    mismatches: tuple[str, ...] = ()
    tolerance_days: int = 2

    @property
    def agrees(self) -> bool:
        """Whether every local and vendor roll matched within tolerance."""
        return not self.mismatches

    @classmethod
    def cross_verify(
        cls,
        *,
        root: InstrumentId,
        local_rolls: tuple[RollEvent, ...],
        native_rolls: tuple[RollEvent, ...],
        tolerance_days: int = 2,
    ) -> NativeRollCrossCheck:
        """Match local and native rolls by expiration pair and roll date."""
        local_by_pair = {
            (event.from_expiration, event.to_expiration): event for event in local_rolls
        }
        matched: list[tuple[RollEvent, RollEvent]] = []
        mismatches: list[str] = []
        tolerance = timedelta(days=tolerance_days)
        used: set[tuple[date, date]] = set()
        for native in native_rolls:
            key = (native.from_expiration, native.to_expiration)
            local = local_by_pair.get(key)
            if local is None:
                mismatches.append(
                    f"native roll {native.from_expiration}->{native.to_expiration} at "
                    f"{native.roll_datetime.isoformat()} has no local counterpart"
                )
                continue
            used.add(key)
            drift = abs(native.roll_datetime - local.roll_datetime)
            if drift > tolerance:
                mismatches.append(
                    f"roll {native.from_expiration}->{native.to_expiration} drifts "
                    f"{drift.days} day(s) (native {native.roll_datetime.date()}, "
                    f"local {local.roll_datetime.date()}, tolerance {tolerance_days})"
                )
                continue
            matched.append((local, native))
        for key, local in local_by_pair.items():
            if key not in used:
                mismatches.append(
                    f"local roll {local.from_expiration}->{local.to_expiration} at "
                    f"{local.roll_datetime.isoformat()} has no native counterpart"
                )
        return cls(
            root=root.strip().upper(),
            local_rolls=local_rolls,
            native_rolls=native_rolls,
            matched=tuple(matched),
            mismatches=tuple(mismatches),
            tolerance_days=tolerance_days,
        )


# ---------------------------------------------------------------------------
# Continuous contract builder
# ---------------------------------------------------------------------------


class ContinuousContractBuilder:
    """Stitch raw expiring contracts into one continuous series per root."""

    def __init__(
        self,
        *,
        instrument_master: InstrumentsConfig | None = None,
        roll_schedule: RollScheduleCalculator | None = None,
        trigger: RollTrigger = RollTrigger.CALENDAR_DAYS_BEFORE_FND,
        adjustment: PriceAdjustment = PriceAdjustment.PANAMA,
        days_before_expiration: int | None = None,
        physical_buffer_days: int = 7,
        native_frame: pl.DataFrame | None = None,
        tolerance_days: int = 2,
    ) -> None:
        self._master = instrument_master
        self._roll_schedule: RollScheduleCalculator
        if roll_schedule is not None:
            self._roll_schedule = roll_schedule
        elif trigger is RollTrigger.VOLUME_OI_CROSSOVER:
            self._roll_schedule = VolumeOICrossoverRollSchedule(instrument_master=instrument_master)
        elif trigger is RollTrigger.DATABENTO_NATIVE:
            self._roll_schedule = DatabentoNativeRollSchedule(
                native_frame=native_frame,
                instrument_master=instrument_master,
                tolerance_days=tolerance_days,
            )
        else:
            self._roll_schedule = CalendarRollSchedule(
                days_before_expiration=days_before_expiration,
                physical_buffer_days=physical_buffer_days,
                instrument_master=instrument_master,
                physical_delivery=None,
            )
        self._adjustment = adjustment

    @property
    def roll_schedule(self) -> RollScheduleCalculator:
        """Active roll-trigger methodology."""
        return self._roll_schedule

    @property
    def adjustment(self) -> PriceAdjustment:
        """Default price-adjustment methodology."""
        return self._adjustment

    @property
    def instrument_master(self) -> InstrumentsConfig:
        """Instrument master used for per-root roll configuration."""
        if self._master is None:
            self._master = load_instruments()
        return self._master

    # -- roll events -----------------------------------------------------------

    def roll_events(self, bars: pl.DataFrame, root: InstrumentId) -> tuple[RollEvent, ...]:
        """Roll events for ``root`` from the configured schedule."""
        return self._roll_schedule.compute(root, bars)

    def cross_verify(
        self,
        bars: pl.DataFrame,
        native_frame: pl.DataFrame,
        root: InstrumentId,
        *,
        tolerance_days: int = 2,
    ) -> NativeRollCrossCheck:
        """Compare this builder's local rolls with Databento native rolls."""
        local = self.roll_events(bars, root)
        schedule = DatabentoNativeRollSchedule(
            native_frame=native_frame,
            instrument_master=self._master,
            tolerance_days=tolerance_days,
        )
        native = schedule.detect_rolls(root, native_frame)
        return NativeRollCrossCheck.cross_verify(
            root=root,
            local_rolls=local,
            native_rolls=native,
            tolerance_days=tolerance_days,
        )

    # -- series ----------------------------------------------------------------

    def build(
        self,
        bars: pl.DataFrame,
        root: InstrumentId,
        *,
        adjustment: PriceAdjustment | None = None,
    ) -> pl.DataFrame:
        """Build the continuous series for ``root`` (canonical continuous schema)."""
        panel = _build_panel(root, bars)
        events = tuple(self._roll_schedule.compute(root, bars))
        method = adjustment if adjustment is not None else self._adjustment
        gaps = [event.roll_gap for event in events]
        additive_suffix: list[float] = [0.0] * (len(events) + 1)
        for index in range(len(events) - 1, -1, -1):
            additive_suffix[index] = additive_suffix[index + 1] + gaps[index]
        ratio_suffix: list[float] = [1.0] * (len(events) + 1)
        for index in range(len(events) - 1, -1, -1):
            event = events[index]
            ratio_suffix[index] = (
                ratio_suffix[index + 1] * (event.from_price / event.to_price)
                if event.to_price != 0.0
                else ratio_suffix[index + 1]
            )
        roll_by_timestamp = {event.roll_datetime: event for event in events}
        records: list[dict[str, object]] = []
        roll_index = 0
        active = panel.expirations[0]
        for timestamp in panel.timestamps:
            while roll_index < len(events) and events[roll_index].roll_datetime <= timestamp:
                active = events[roll_index].to_expiration
                roll_index += 1
            available = panel.available.get(timestamp, ())
            front = active if active in available else self._fallback_front(available, active)
            if front is None:  # pragma: no cover - available is non-empty by construction
                continue
            row = panel.rows[front][timestamp]
            next_expiration = panel.next_after(timestamp, front)
            next_row = (
                panel.rows[next_expiration][timestamp] if next_expiration is not None else None
            )
            additive = additive_suffix[roll_index]
            ratio = ratio_suffix[roll_index]
            if method is PriceAdjustment.PANAMA:
                adjusted_close = row.close + additive
                panama_adjustment = additive
            elif method is PriceAdjustment.PROPORTIONAL:
                adjusted_close = row.close * ratio
                panama_adjustment = 0.0
            else:
                adjusted_close = row.close
                panama_adjustment = 0.0
            roll_event = roll_by_timestamp.get(timestamp)
            next_close = next_row.close if next_row is not None else None
            term_slope: float | None = None
            carry_annualized: float | None = None
            if next_row is not None and row.close != 0.0:
                term_slope = (next_row.close - row.close) / row.close
                days = max((next_row.expiration - row.expiration).days, 1)
                carry_annualized = -term_slope * (365.0 / days)
            records.append(
                {
                    s.TIMESTAMP_COLUMN: timestamp,
                    s.ROOT_SYMBOL_COLUMN: panel.root,
                    "front_symbol": row.instrument_id,
                    "front_expiration": row.expiration,
                    "next_symbol": next_row.instrument_id if next_row is not None else None,
                    "next_expiration": next_row.expiration if next_row is not None else None,
                    s.OPEN_COLUMN: row.open,
                    s.HIGH_COLUMN: row.high,
                    s.LOW_COLUMN: row.low,
                    s.CLOSE_COLUMN: row.close,
                    "adjusted_close": adjusted_close,
                    "panama_adjustment": panama_adjustment,
                    "ratio_factor": ratio,
                    s.VOLUME_COLUMN: row.volume,
                    s.OPEN_INTEREST_COLUMN: row.open_interest,
                    "roll_flag": roll_event is not None,
                    "roll_gap": roll_event.roll_gap if roll_event is not None else None,
                    "front_close": row.close,
                    "next_close": next_close,
                    s.TERM_SLOPE: term_slope,
                    s.CARRY_ANNUALIZED: carry_annualized,
                }
            )
        frame = pl.DataFrame(records, infer_schema_length=None)
        return s.ensure_continuous_schema(
            frame.sort(s.TIMESTAMP_COLUMN), context=f"continuous[{panel.root}]"
        )

    def build_frame(
        self, bars: pl.DataFrame, *, adjustment: PriceAdjustment | None = None
    ) -> pl.DataFrame:
        """Build continuous series for every root present in ``bars``."""
        roots = tuple(
            sorted(
                {
                    str(value)
                    for value in bars[s.ROOT_SYMBOL_COLUMN].cast(pl.Utf8).unique().to_list()
                }
            )
        )
        frames = [self.build(bars, root, adjustment=adjustment) for root in roots]
        return pl.concat(frames, how="vertical_relaxed")

    def panama(self, bars: pl.DataFrame, root: InstrumentId) -> pl.DataFrame:
        """Panama (back-adjusted additive) series for ``root``."""
        return self.build(bars, root, adjustment=PriceAdjustment.PANAMA)

    def proportional(self, bars: pl.DataFrame, root: InstrumentId) -> pl.DataFrame:
        """Proportional (ratio-adjusted multiplicative) series for ``root``."""
        return self.build(bars, root, adjustment=PriceAdjustment.PROPORTIONAL)

    def unadjusted(self, bars: pl.DataFrame, root: InstrumentId) -> pl.DataFrame:
        """Unadjusted front/second-month series with explicit carry and term slope."""
        return self.build(bars, root, adjustment=PriceAdjustment.UNADJUSTED)

    def calendar_spread(self, bars: pl.DataFrame, root: InstrumentId) -> pl.DataFrame:
        """Front/second-month calendar-spread series (raw prices, no adjustment)."""
        panel = _build_panel(root, bars)
        records: list[dict[str, object]] = []
        for timestamp in panel.timestamps:
            front = panel.front_at(timestamp)
            if front is None:  # pragma: no cover - timestamps derive from available bars
                continue
            next_expiration = panel.next_after(timestamp, front)
            if next_expiration is None:
                continue
            front_row = panel.rows[front][timestamp]
            next_row = panel.rows[next_expiration][timestamp]
            records.append(
                {
                    s.TIMESTAMP_COLUMN: timestamp,
                    s.ROOT_SYMBOL_COLUMN: panel.root,
                    "front_symbol": front_row.instrument_id,
                    "next_symbol": next_row.instrument_id,
                    "front_expiration": front,
                    "next_expiration": next_expiration,
                    "front_close": front_row.close,
                    "next_close": next_row.close,
                    "roll_gap": next_row.close - front_row.close,
                }
            )
        return pl.DataFrame(records, infer_schema_length=None)

    @staticmethod
    def _fallback_front(available: tuple[date, ...], active: date) -> date | None:
        """Front contract at a timestamp when ``active`` has no bar (missing data)."""
        if not available:
            return None
        later = [expiration for expiration in available if expiration >= active]
        return later[0] if later else available[-1]
