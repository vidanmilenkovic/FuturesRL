"""Deterministic offline market data and derived fixtures for the API layer.

The telemetry API must be fully explorable with **no Databento key, no IBKR
session and no network**. This module generates the deterministic inputs used by
the offline services:

* a seeded daily OHLCV frame for the configured futures roots (weekdays ending
  at the injected clock's date),
* the seed position book and mark-to-market equity history,
* the deterministic Tier-1 macro calendar and lockout window,
* deterministic LLM Risk-First assessments and headline feed (the real
  ``src.risk.llm_risk_overlay`` provider is used when importable; otherwise this
  deterministic seed is explicitly reported as such),
* forward curves + roll calendar derived from the real instrument master and
  the execution layer's CME expiry helpers.

Everything is a pure function of ``(instrument_master, clock, seed)`` so tests
inject a fixed clock and get identical output on every run. None of this data is
presented as live: the API reports ``offline_mode=true`` and every service that
falls back records why.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta

import numpy as np
import polars as pl
from configs.instruments import InstrumentsConfig, InstrumentSpec, load_instruments

from src.api.schemas import (
    EquityPoint,
    MacroEvent,
    MacroLockout,
    NewsEvent,
    PositionRow,
    RiskAssessmentRow,
    RiskThresholds,
    RollCalendarEntry,
    TermContract,
    TermCurve,
    TermStructureResponse,
)
from src.core.types import (
    TRADING_DAYS_PER_YEAR,
    AssetClass,
    InstrumentSnapshot,
    MarketState,
    Position,
    RiskAssessment,
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

__all__ = [
    "Clock",
    "OfflineLLMClient",
    "OfflineMarket",
    "build_offline_market",
    "build_rebalance_states",
    "build_window_market",
    "equity_history",
    "fixture_book",
    "macro_calendar",
    "macro_lockout",
    "news_feed",
    "risk_assessment_core",
    "risk_assessment_rows",
    "risk_assessments",
    "term_structure",
    "utc_now",
]

#: Injectable clock returning a timezone-aware UTC datetime.
type Clock = Callable[[], datetime]

#: Default seed position book: ``(root, signed_quantity)``.
_DEFAULT_BOOK: tuple[tuple[str, int], ...] = (
    ("ES", 14),
    ("NQ", 6),
    ("CL", -8),
    ("GC", 3),
    ("ZN", -12),
    ("MES", 18),
)

#: Roots the offline simulated broker trades (standard + one micro leg).
TRADABLE_ROOTS: tuple[str, ...] = ("ES", "MES", "NQ", "CL", "GC", "ZN")

#: Daily vol per sector for the deterministic GBM fixture.
_SECTOR_DAILY_VOL: Mapping[str, float] = {
    "equity": 0.18,
    "rates": 0.06,
    "energy": 0.35,
    "metals": 0.15,
    "fx": 0.08,
}

#: Annualised term-structure slope per sector (positive = contango).
_SECTOR_TERM_SLOPE: Mapping[str, float] = {
    "equity": 0.0020,
    "rates": 0.0011,
    "energy": -0.0048,
    "metals": 0.0013,
    "fx": -0.0004,
}

#: Deterministic initial equity used by the offline account/backtest fixtures.
DEFAULT_INITIAL_EQUITY: float = 4_000_000.0

#: Maximum number of daily bars materialised per offline backtest request.
MAX_BACKTEST_BARS: int = 520

#: Backtest fixture bar count used when a request spans a longer window.
_MAX_HISTORY_DAYS: int = 760


def utc_now() -> datetime:
    """Default wall clock (timezone-aware UTC)."""
    return datetime.now(UTC)


# ---------------------------------------------------------------------------
# Deterministic bar generation
# ---------------------------------------------------------------------------


def _weekday_timestamps(end: date, periods: int) -> tuple[datetime, ...]:
    """``periods`` weekday timestamps ending on ``end`` at the US close (21:00Z)."""
    stamps: list[datetime] = []
    cursor = end
    while len(stamps) < periods:
        if cursor.weekday() < 5:
            stamps.append(datetime.combine(cursor, time(21, 0), tzinfo=UTC))
        cursor -= timedelta(days=1)
    return tuple(reversed(stamps))


def _sector_vol(spec: InstrumentSpec) -> float:
    return _SECTOR_DAILY_VOL.get(spec.sector, 0.15) / math.sqrt(TRADING_DAYS_PER_YEAR)


def _round_to_tick(price: float, tick_size: float) -> float:
    ticks = round(price / tick_size)
    return max(round(ticks * tick_size, 10), tick_size)


def _reference_price(spec: InstrumentSpec) -> float:
    """Deterministic reference mark for ``spec`` (realistic CME price level).

    Uses the data layer's synthetic reference price table when importable so the
    fixture notionals/margins match the instrument master's multipliers; falls
    back to a multiplier-derived level if the data layer is unavailable.
    """
    try:
        from src.data.databento_loader import synthetic_reference_price
    except ImportError:  # pragma: no cover - data layer is always present today
        return float(max(50.0, 100.0 * float(spec.multiplier) ** 0.25))
    return float(synthetic_reference_price(spec.root))


def _record_float(record: Mapping[str, object], key: str, default: float = 0.0) -> float:
    """Coerce a polars row value to ``float`` (``default`` when absent)."""
    value = record.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return default
    return float(value)


def _build_bars(
    master: InstrumentsConfig,
    roots: Sequence[str],
    timestamps: Sequence[datetime],
    *,
    seed: int,
) -> pl.DataFrame:
    """Seeded OHLCV frame for ``roots`` aligned to ``timestamps``."""
    rng = np.random.default_rng(seed)
    expiration = date(timestamps[-1].year + 1, 3, 15)
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
    for root_index, root in enumerate(roots):
        spec = master.get(root)
        price = _reference_price(spec)
        daily_vol = _sector_vol(spec)
        drift = 0.00018 + 0.00004 * root_index
        for step, timestamp in enumerate(timestamps):
            shock = float(rng.normal(drift, daily_vol))
            open_price = price
            close_price = max(open_price * (1.0 + shock), spec.tick_size)
            high = max(open_price, close_price) * (
                1.0 + abs(float(rng.normal(0.0, daily_vol * 0.5)))
            )
            low = min(open_price, close_price) * (
                1.0 - abs(float(rng.normal(0.0, daily_vol * 0.5)))
            )
            volume = float(abs(rng.normal(3_000_000.0, 250_000.0)))
            rows[TIMESTAMP_COLUMN].append(timestamp)
            rows["instrument_id"].append(f"{root}_FIXTURE")
            rows[ROOT_SYMBOL_COLUMN].append(root)
            rows[EXPIRATION_COLUMN].append(expiration)
            rows[OPEN_COLUMN].append(open_price)
            rows[HIGH_COLUMN].append(high)
            rows[LOW_COLUMN].append(low)
            rows[CLOSE_COLUMN].append(close_price)
            rows[VOLUME_COLUMN].append(volume)
            rows[OPEN_INTEREST_COLUMN].append(float(2_000_000 + 50_000 * root_index + step))
            rows[SETTLEMENT_COLUMN].append(close_price)
            price = close_price
    return ensure_bars_schema(pl.DataFrame(rows), context="api_offline_bars")


def weekdays_between(start: date, end: date) -> tuple[datetime, ...]:
    """Daily US-close timestamps (21:00 UTC) for weekdays in ``(start, end]``."""
    stamps: list[datetime] = []
    cursor = start + timedelta(days=1)
    while cursor <= end:
        if cursor.weekday() < 5:
            stamps.append(datetime.combine(cursor, time(21, 0), tzinfo=UTC))
        cursor += timedelta(days=1)
    return tuple(stamps)


@dataclass(frozen=True, slots=True)
class OfflineMarket:
    """Deterministic offline market snapshot shared by all API services."""

    instrument_master: InstrumentsConfig
    bars: pl.DataFrame
    timestamps: tuple[datetime, ...]
    prices: Mapping[str, float]
    seed: int

    @property
    def as_of(self) -> datetime:
        """Timestamp of the latest generated bar (the offline "now")."""
        return self.timestamps[-1]

    @property
    def roots(self) -> tuple[str, ...]:
        """Roots present in the fixture bars, in declaration order."""
        return tuple(sorted(self.prices))

    def price(self, root: str) -> float:
        """Latest fixture mark for ``root`` (raises ``KeyError`` if unknown)."""
        return self.prices[root]

    def timestamps_of(self, root: str) -> tuple[datetime, ...]:
        """Bar timestamps for ``root`` (exposed for rolling analytics fixtures)."""
        del root
        return self.timestamps

    def snapshot(self, market_state: MarketState) -> Mapping[str, InstrumentSnapshot]:
        """Instrument snapshots of ``market_state`` (convenience proxy)."""
        return market_state.instruments


def build_offline_market(
    instrument_master: InstrumentsConfig | None = None,
    *,
    clock: Clock = utc_now,
    days: int = _MAX_HISTORY_DAYS,
    seed: int = 7,
) -> OfflineMarket:
    """Generate the deterministic offline market dataset.

    ``days`` daily bars are generated for the tradable roots, ending at the
    injected clock's date (the API therefore behaves sensibly regardless of when
    it is started; tests inject a fixed clock for byte-identical output).
    """
    master = instrument_master if instrument_master is not None else load_instruments()
    roots = tuple(root for root in TRADABLE_ROOTS if master.has(root))
    if not roots:
        roots = master.roots()[:5]
    now = clock().astimezone(UTC)
    capped_days = max(1, min(days, _MAX_HISTORY_DAYS))
    timestamps = _weekday_timestamps(now.date(), capped_days)
    frame = _build_bars(master, roots, timestamps, seed=seed)
    latest = (
        frame.sort(TIMESTAMP_COLUMN)
        .group_by(ROOT_SYMBOL_COLUMN)
        .agg(pl.col(CLOSE_COLUMN).last().alias("close"))
    )
    prices = {
        str(record[ROOT_SYMBOL_COLUMN]): float(record["close"])
        for record in latest.iter_rows(named=True)
    }
    return OfflineMarket(
        instrument_master=master,
        bars=frame,
        timestamps=timestamps,
        prices=prices,
        seed=seed,
    )


def build_window_market(
    instrument_master: InstrumentsConfig,
    start: date,
    end: date,
    *,
    seed: int = 7,
    max_bars: int = MAX_BACKTEST_BARS,
    roots: Sequence[str] | None = None,
    warmup_bars: int = 0,
) -> OfflineMarket:
    """Deterministic market window for ``[start, end]`` used by API backtests.

    ``warmup_bars`` prepends extra trading days before ``start`` when the
    requested window is short: the FinRL-X features (TSMOM 126d lookback, KAMA,
    covariance) need history before the first rebalance, so a 6-month request is
    simulated with a ~1-year history warm-up. The engine is still filtered to
    ``[start, end]`` by the caller, so warm-up bars never enter the result curve.

    The total window is capped to ``max_bars`` daily bars (keeping the most
    recent), so an arbitrary multi-year request still returns quickly.
    """
    selected = tuple(root for root in (roots or TRADABLE_ROOTS) if instrument_master.has(root))
    if not selected:
        selected = instrument_master.roots()[:5]
    timestamps = weekdays_between(start, end)
    if not timestamps:
        raise ValueError("requested backtest window contains no weekdays")
    if warmup_bars > 0 and len(timestamps) < warmup_bars + 60:
        missing = warmup_bars + 60 - len(timestamps)
        warmup: list[datetime] = []
        cursor = start
        while len(warmup) < missing:
            cursor -= timedelta(days=1)
            if cursor.weekday() < 5:
                warmup.append(datetime.combine(cursor, time(21, 0), tzinfo=UTC))
        timestamps = tuple(sorted(warmup)) + timestamps
    if len(timestamps) > max_bars:
        timestamps = timestamps[-max_bars:]
    frame = _build_bars(instrument_master, selected, timestamps, seed=seed)
    latest = (
        frame.sort(TIMESTAMP_COLUMN)
        .group_by(ROOT_SYMBOL_COLUMN)
        .agg(pl.col(CLOSE_COLUMN).last().alias("close"))
    )
    prices = {
        str(record[ROOT_SYMBOL_COLUMN]): float(record["close"])
        for record in latest.iter_rows(named=True)
    }
    return OfflineMarket(
        instrument_master=instrument_master,
        bars=frame,
        timestamps=timestamps,
        prices=prices,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Account / position fixtures
# ---------------------------------------------------------------------------


def fixture_book() -> tuple[tuple[str, int], ...]:
    """Seed ``(root, signed_contracts)`` position book for the demo account."""
    return _DEFAULT_BOOK


def equity_history(
    market: OfflineMarket,
    net_liquidation: float,
    *,
    points: int = TRADING_DAYS_PER_YEAR,
    seed: int = 11,
    now: datetime | None = None,
) -> list[EquityPoint]:
    """Seeded equity curve ending exactly at ``net_liquidation``.

    When ``now`` is before the last daily close (e.g. intraday), the final point
    is stamped at ``now`` so an appended live telemetry point stays strictly
    after the historical curve (chart libraries require ascending time).
    """
    stamps = list(market.timestamps[-min(points, len(market.timestamps)) :])
    if now is not None and stamps:
        moment = now.astimezone(UTC)
        if moment < stamps[-1] and (len(stamps) < 2 or moment > stamps[-2]):
            stamps[-1] = moment
    rng = np.random.default_rng(seed)
    values: list[float] = []
    level = 1.0
    for _ in stamps:
        level *= 1.0 + 0.00045 + float(rng.normal(0.0, 0.0082))
        values.append(level)
    scale = net_liquidation / values[-1] if values and values[-1] > 0 else 1.0
    return [
        EquityPoint(time=timestamp, value=round(value * scale, 2))
        for timestamp, value in zip(stamps, values, strict=True)
    ]


def position_rows(
    master: InstrumentsConfig,
    positions: Mapping[str, Position],
    net_liquidation: float,
    *,
    as_of: date,
) -> list[PositionRow]:
    """Map core :class:`Position` objects into dashboard rows."""
    rows: list[PositionRow] = []
    for root in sorted(positions):
        position = positions[root]
        if position.quantity == 0:
            continue
        spec = master.get(root)
        roll_date, _ = roll_window(spec, as_of)
        rows.append(
            PositionRow(
                instrument_id=position.instrument_id,
                root_symbol=position.instrument_id,
                asset_class=spec.asset_class,
                contract_symbol=position.contract_symbol,
                expiration=position.expiration,
                days_to_roll=(roll_date - as_of).days,
                quantity=position.quantity,
                average_price=position.average_price,
                market_price=position.market_price,
                multiplier=position.multiplier,
                notional=position.signed_notional,
                weight=position.signed_notional / net_liquidation,
                unrealized_pnl=position.unrealized_pnl,
                initial_margin=position.initial_margin,
                maintenance_margin=position.maintenance_margin,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Roll arithmetic
# ---------------------------------------------------------------------------


def roll_window(spec: InstrumentSpec, as_of: date) -> tuple[date, date]:
    """``(roll_date, next_expiration)`` for ``spec`` as of ``as_of``.

    Uses the execution layer's CME third-Friday expiry helpers when importable;
    falls back to the instrument master's contract months otherwise (the API
    must import cleanly even when ``src.execution`` is absent).
    """
    try:
        from src.execution.ibkr_client import (
            active_contract_expiration,
            upcoming_contract_expirations,
        )
    except ImportError:  # pragma: no cover - execution layer always present today
        active = _fallback_expiry(spec, as_of, offset=0)
        following = _fallback_expiry(spec, as_of, offset=1)
    else:
        active = active_contract_expiration(spec, as_of)
        candidates = upcoming_contract_expirations(spec, as_of, count=3)
        following = next((expiry for expiry in candidates if expiry > active), active)
    roll_date = active - timedelta(days=spec.roll.days_before_fnd)
    return roll_date, following


def _fallback_expiry(spec: InstrumentSpec, as_of: date, *, offset: int) -> date:
    """Third Friday of the next ``offset`` contract month for ``spec``."""
    months = spec.roll.contract_months
    year, month = as_of.year, as_of.month
    found = 0
    for _ in range(36):
        if month in months:
            candidate = _third_friday(year, month)
            if candidate > as_of and found >= offset:
                return candidate
            if candidate > as_of:
                found += 1
        month += 1
        if month > 12:
            year += 1
            month = 1
    return as_of + timedelta(days=30 * (offset + 1))  # pragma: no cover - defensive


def _third_friday(year: int, month: int) -> date:
    first = date(year, month, 1)
    friday = 4
    offset = (friday - first.weekday()) % 7
    return first + timedelta(days=offset + 14)


# ---------------------------------------------------------------------------
# Pipeline / backtest state construction
# ---------------------------------------------------------------------------


def schedule_decisions(
    timestamps: Sequence[datetime],
    *,
    rebalance_frequency: str,
) -> tuple[datetime, ...]:
    """Timestamps at which the pipeline is re-run for ``rebalance_frequency``."""
    try:
        from src.backtest.engine import RebalanceSchedule
    except ImportError as exc:  # pragma: no cover - backtest module always present
        raise RuntimeError(f"backtest engine unavailable: {exc}") from exc
    schedule = RebalanceSchedule.from_frequency(rebalance_frequency)
    decisions: list[datetime] = []
    previous: datetime | None = None
    for index, timestamp in enumerate(timestamps):
        if schedule.should_rebalance(index, timestamp, previous):
            decisions.append(timestamp)
        previous = timestamp
    return tuple(decisions)


def build_rebalance_states(
    market: OfflineMarket,
    *,
    rebalance_frequency: str,
    initial_equity: float,
    timeframe: str = "1d",
) -> list[MarketState]:
    """Build one market state per bar, full-featured at rebalance timestamps.

    :class:`~src.data.market_state_builder.MarketStateBuilder` is used for the
    rebalance dates (it is the production point-in-time builder); the
    intermediate mark-to-market bars use a lightweight price/volume state built
    from the same deterministic frame. The pipeline never runs on the
    lightweight bars, so features/covariance are only required where they are
    actually consumed — this keeps a full-year offline backtest around one
    second instead of tens of seconds.
    """
    try:
        from src.data.market_state_builder import MarketStateBuilder
    except ImportError as exc:  # pragma: no cover - data layer always present
        raise RuntimeError(f"market state builder unavailable: {exc}") from exc

    decisions = schedule_decisions(market.timestamps, rebalance_frequency=rebalance_frequency)
    builder = MarketStateBuilder(instrument_master=market.instrument_master, timeframe=timeframe)
    full_states: dict[datetime, MarketState] = {}
    if decisions:
        for timestamp, state in zip(
            decisions,
            builder.build_series(market.bars, decisions),
            strict=True,
        ):
            full_states[timestamp] = state
    lightweight = _lightweight_states(market, initial_equity)
    return [
        full_states.get(state.timestamp, state)
        for state in lightweight
        if state.timestamp in full_states or state.timestamp not in decisions
    ]


def _lightweight_states(market: OfflineMarket, initial_equity: float) -> list[MarketState]:
    """Cheap per-bar states carrying marks/volumes (no features/covariance)."""
    master = market.instrument_master
    frame = market.bars.sort([TIMESTAMP_COLUMN, ROOT_SYMBOL_COLUMN])
    by_timestamp: dict[datetime, list[dict[str, object]]] = {}
    for record in frame.iter_rows(named=True):
        timestamp = record[TIMESTAMP_COLUMN]
        if not isinstance(timestamp, datetime):
            continue
        moment = timestamp.astimezone(UTC)
        by_timestamp.setdefault(moment, []).append(record)
    states: list[MarketState] = []
    for timestamp in market.timestamps:
        records = by_timestamp.get(timestamp, [])
        snapshots: dict[str, InstrumentSnapshot] = {}
        for record in records:
            root = str(record[ROOT_SYMBOL_COLUMN])
            spec = master.get(root)
            close = _record_float(record, CLOSE_COLUMN)
            expiration = record.get(EXPIRATION_COLUMN)
            snapshots[root] = InstrumentSnapshot(
                instrument_id=root,
                root_symbol=root,
                price=close,
                asset_class=spec.asset_class,
                expiration=expiration if isinstance(expiration, date) else None,
                volume=_record_float(record, VOLUME_COLUMN),
                open_interest=_record_float(record, OPEN_INTEREST_COLUMN),
                settlement=_record_float(record, SETTLEMENT_COLUMN),
            )
        if snapshots:
            states.append(
                MarketState(
                    timestamp=timestamp,
                    instruments=snapshots,
                    equity=initial_equity,
                )
            )
    return states


# ---------------------------------------------------------------------------
# Macro calendar / lockout
# ---------------------------------------------------------------------------

_TIER1_ROOTS: tuple[str, ...] = ("ES", "NQ", "ZN", "GC", "CL", "6E")


def _next_weekday(now: datetime, weekday: int, *, hour: int, minute: int) -> datetime:
    candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    days_ahead = (weekday - candidate.weekday()) % 7
    candidate += timedelta(days=days_ahead)
    if candidate <= now:
        candidate += timedelta(days=7)
    return candidate


def _next_monthly(now: datetime, *, day: int, hour: int, minute: int) -> datetime:
    candidate = now.replace(day=1, hour=hour, minute=minute, second=0, microsecond=0)
    if candidate.month == 12:
        candidate = candidate.replace(year=candidate.year + 1, month=1, day=day)
    else:
        candidate = candidate.replace(month=candidate.month + 1, day=day)
    return candidate


def _next_fomc(now: datetime) -> datetime:
    """Next FOMC statement (18:00Z / 14:00 ET) from a deterministic schedule."""
    for year in (now.year, now.year + 1):
        for month in (1, 3, 5, 6, 9, 11, 12):
            candidate = _third_friday(year, month) - timedelta(days=2)
            moment = datetime.combine(candidate, time(18, 0), tzinfo=UTC)
            if moment > now:
                return moment
    return now + timedelta(days=30)  # pragma: no cover - defensive


def macro_calendar(now: datetime, roots: Sequence[str] | None = None) -> list[MacroEvent]:
    """Deterministic upcoming macro releases used for lockout windows."""
    del roots
    moment = now.astimezone(UTC)
    events: list[tuple[str, str, datetime, int, tuple[str, ...]]] = [
        (
            "macro-cpi",
            "US CPI (YoY)",
            _next_monthly(moment, day=15, hour=13, minute=30),
            1,
            ("ES", "NQ", "ZN", "GC"),
        ),
        (
            "macro-eia",
            "EIA Crude Inventory",
            _next_weekday(moment, weekday=2, hour=15, minute=30),
            1,
            ("CL",),
        ),
        (
            "macro-nfp",
            "Non-Farm Payrolls",
            _next_weekday(moment, weekday=4, hour=13, minute=30),
            1,
            ("ES", "NQ", "ZN"),
        ),
        ("macro-fomc", "FOMC Statement + Presser", _next_fomc(moment), 1, _TIER1_ROOTS),
        (
            "macro-wasde",
            "USDA WASDE",
            _next_monthly(moment, day=12, hour=17, minute=0),
            2,
            ("ZC", "ZW"),
        ),
    ]
    if moment.weekday() == 4:
        # NFP is only "next" if the Friday release is still ahead of us.
        nfp = events[2][2]
        if nfp <= moment:
            events[2] = (events[2][0], events[2][1], nfp + timedelta(days=7), 1, events[2][4])
    output = [
        MacroEvent(
            id=event_id,
            event=name,
            event_time=event_time,
            tier=tier,
            minutes_until=max(0, round((event_time - moment).total_seconds() / 60.0)),
            roots=list(affected),
        )
        for event_id, name, event_time, tier, affected in events
    ]
    return sorted(output, key=lambda event: event.event_time)


def macro_lockout(
    events: Sequence[MacroEvent],
    thresholds: RiskThresholds,
    now: datetime,
) -> MacroLockout:
    """Active lockout window for the next Tier-1 release, if any."""
    moment = now.astimezone(UTC)
    before = timedelta(minutes=thresholds.macro_lockout_minutes_before)
    after = timedelta(minutes=thresholds.macro_lockout_minutes_after)
    for event in events:
        if event.tier != 1:
            continue
        start = event.event_time - before
        end = event.event_time + after
        if start <= moment <= end:
            return MacroLockout(
                active=True,
                event=event.event,
                starts_at=start,
                ends_at=end,
                minutes_remaining=max(0, round((end - moment).total_seconds() / 60.0)),
            )
    return MacroLockout(active=False, minutes_remaining=0)


# ---------------------------------------------------------------------------
# Deterministic LLM Risk-First assessments / news
# ---------------------------------------------------------------------------


def _stable_unit(*parts: str) -> float:
    """Stable pseudo-uniform value in ``[0, 1)`` derived from string parts."""
    digest = 0
    for part in parts:
        for char in part:
            digest = (digest * 131 + ord(char)) % 1_000_003
    return digest / 1_000_003.0


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def risk_assessment_core(
    state: MarketState,
    thresholds: RiskThresholds,
    now: datetime,
) -> tuple[RiskAssessment, ...]:
    """Deterministic core Risk-First assessments for the state's universe.

    The scoring uses the real canonical features (21d momentum, realised vol,
    turbulence, carry) and is then pushed through the same deterministic Module-3
    projection the LLM overlay implements: ``c < tau`` neutralises to
    ``theta_safe``; ``S >= kappa`` forces weight 0; ``theta_safe < S < kappa``
    scales linearly. These objects are the exact seed for
    :class:`OfflineLLMClient` and for ``LLMRiskOverlay.set_cached_assessments``.
    """
    moment = now.astimezone(UTC)
    assessments: list[RiskAssessment] = []
    for root in sorted(state.instruments):
        snapshot = state.instruments[root]
        momentum = _finite(snapshot.feature("ret_21d"), 0.0)
        # vol_20d / turbulence are annualised canonical features; carry may be
        # unavailable in the offline feature frame (0.0) — all inputs are finite.
        volatility = _finite(snapshot.feature("vol_20d"), 0.15)
        turbulence = _finite(snapshot.feature("turbulence"), 1.0)
        carry = _finite(snapshot.feature("carry_annualized"), 0.0)
        sentiment = _clamp(3.0 + 22.0 * math.tanh(momentum * 18.0), 1.0, 5.0)
        # Calibrated so a typical equity/metal root lands near theta_safe while a
        # high-vol energy root crosses into the deterministic scale-down band.
        raw_risk = (
            1.10
            + 6.5 * max(0.0, volatility - 0.06)
            + 2.0 * abs(momentum)
            + 0.02 * turbulence
            + 0.35 * abs(carry)
        )
        confidence = _clamp(
            0.86 + 0.12 * _stable_unit(root, "c") - 0.22 * _stable_unit(root, "x"), 0.0, 1.0
        )
        # The offline fixture intentionally exercises the Module-1 anti-hallucination
        # path on GC: a sub-threshold confidence must neutralise the signal to
        # theta_safe regardless of the raw semantic risk estimate.
        if root == "GC":
            confidence = min(confidence, 0.62)
        neutralised = confidence < thresholds.llm_confidence_threshold
        semantic = _clamp(thresholds.llm_safe_score if neutralised else raw_risk, 1.0, 5.0)
        if neutralised:
            summary = (
                "Neutralised by the Module-1 confidence/variance filter "
                f"(c={confidence:.2f} < tau={thresholds.llm_confidence_threshold:.2f}) -> theta_safe."
            )
        elif semantic >= thresholds.llm_critical_score:
            summary = (
                f"Semantic risk S={semantic:.2f} >= kappa={thresholds.llm_critical_score:.2f} with "
                f"c={confidence:.2f} >= tau: deterministic circuit breaker forces w=0."
            )
        elif semantic > thresholds.llm_safe_score:
            multiplier = (thresholds.llm_critical_score - semantic) / (
                thresholds.llm_critical_score - thresholds.llm_safe_score
            )
            summary = (
                f"S={semantic:.2f} inside (theta_safe, kappa): weight scaled x{multiplier:.3f} by the "
                "deterministic post-hoc projection."
            )
        else:
            summary = (
                f"Benign macro/news backdrop (S={semantic:.2f} <= "
                f"theta_safe={thresholds.llm_safe_score:.2f})."
            )
        assessments.append(
            RiskAssessment(
                instrument_id=root,
                timestamp=moment,
                sentiment_score=round(sentiment, 4),
                semantic_risk_score=round(semantic, 4),
                confidence=round(confidence, 4),
                reasoning_summary=summary,
                source="llm-offline-client",
            )
        )
    return tuple(assessments)


def risk_assessment_rows(
    assessments: Sequence[RiskAssessment],
    thresholds: RiskThresholds,
    now: datetime,
) -> list[RiskAssessmentRow]:
    """Map core assessments to the frozen dashboard contract (breaker action included)."""
    del now
    rows: list[RiskAssessmentRow] = []
    for assessment in assessments:
        confidence = assessment.confidence
        semantic = assessment.semantic_risk_score
        if confidence < thresholds.llm_confidence_threshold:
            action = "neutralized"
            multiplier = 1.0
        elif semantic >= thresholds.llm_critical_score:
            action = "liquidate"
            multiplier = 0.0
        elif semantic > thresholds.llm_safe_score:
            action = "scale"
            multiplier = (thresholds.llm_critical_score - semantic) / (
                thresholds.llm_critical_score - thresholds.llm_safe_score
            )
        else:
            action = "pass"
            multiplier = 1.0
        rows.append(
            RiskAssessmentRow(
                instrument_id=assessment.instrument_id,
                timestamp=assessment.timestamp,
                sentiment_score=assessment.sentiment_score,
                semantic_risk_score=assessment.semantic_risk_score,
                confidence=confidence,
                reasoning_summary=assessment.reasoning_summary,
                source=assessment.source,
                action=action,  # type: ignore[arg-type]
                weight_multiplier=round(_clamp(multiplier, 0.0, 1.0), 4),
            )
        )
    return rows


def risk_assessments(
    state: MarketState,
    thresholds: RiskThresholds,
    now: datetime,
) -> list[RiskAssessmentRow]:
    """Convenience wrapper: deterministic assessments in dashboard form."""
    core = risk_assessment_core(state, thresholds, now)
    return risk_assessment_rows(core, thresholds, now)


def llm_risk_payload_json(assessments: Sequence[RiskAssessment]) -> str:
    """Serialise assessments as the strict JSON contract the LLM overlay parses.

    Matches the schema enforced by ``LLMRiskAssessmentBatch`` (no extra keys, one
    record per root) so the deterministic offline client exercises the exact same
    validation path as a real OpenAI/Anthropic/vLLM completion.
    """
    payload = {
        "assessments": [
            {
                "instrument_id": assessment.instrument_id,
                "sentiment_score": float(assessment.sentiment_score),
                "semantic_risk_score": float(assessment.semantic_risk_score),
                "confidence": float(assessment.confidence),
                "reasoning_summary": assessment.reasoning_summary,
            }
            for assessment in assessments
        ]
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


class OfflineLLMClient:
    """Deterministic in-process ``AsyncLLMClient`` (never touches the network).

    Implements the production ``AsyncLLMClient`` protocol used by
    :class:`src.risk.llm_risk_overlay.LLMRiskOverlay`: every completion returns the
    same strict JSON payload derived from the offline market state, so the API can
    wire the **real** LLM risk overlay (Module 1/3 validation, macro lockout
    calendar, deterministic breaker) without credentials or HTTP.
    """

    __slots__ = ("_assessments", "_payload")

    def __init__(
        self,
        state: MarketState,
        thresholds: RiskThresholds,
        now: datetime,
    ) -> None:
        core = risk_assessment_core(state, thresholds, now)
        self._assessments = core
        self._payload = llm_risk_payload_json(core)

    @property
    def payload(self) -> str:
        """The strict JSON completion returned for every prompt."""
        return self._payload

    @property
    def assessments(self) -> tuple[RiskAssessment, ...]:
        """Core assessments encoded in the payload."""
        return self._assessments

    async def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        """Return the deterministic strict JSON completion for ``prompt``."""
        del prompt, system_prompt, temperature
        return self._payload


def _finite(value: float, default: float) -> float:
    return value if isinstance(value, int | float) and math.isfinite(value) else default


_NEWS_TEMPLATES: tuple[tuple[str, str, tuple[str, ...], float], ...] = (
    (
        "news-cpi",
        "Core CPI consensus 3.1% — options market pricing a 1.4sigma move in ES",
        ("ES", "NQ", "ZN"),
        2.8,
    ),
    (
        "news-opec",
        "OPEC+ delegates signal no change to production quotas",
        ("CL",),
        1.9,
    ),
    (
        "news-auction",
        "10Y auction tails 2.1bp; duration complex under pressure",
        ("ZN", "NQ"),
        2.4,
    ),
    (
        "news-gold",
        "Gold ETF inflows accelerate for a third consecutive session",
        ("GC",),
        3.9,
    ),
    (
        "news-fed",
        "Fed speaker: policy remains restrictive and data dependent",
        ("ES", "ZN"),
        3.1,
    ),
    (
        "news-eia",
        "EIA crude build 4.2M bbl versus -1.1M expected",
        ("CL",),
        1.6,
    ),
)


def news_feed(
    assessments: Sequence[RiskAssessmentRow],
    now: datetime,
) -> list[NewsEvent]:
    """Deterministic headline feed with severity derived from the risk state."""
    severity_by_root: dict[str, str] = {
        assessment.instrument_id: (
            "critical"
            if assessment.action == "liquidate"
            else "warn"
            if assessment.action == "scale"
            else "info"
        )
        for assessment in assessments
    }
    events: list[NewsEvent] = []
    for index, (event_id, headline, roots, sentiment) in enumerate(_NEWS_TEMPLATES):
        severity = "info"
        for root in roots:
            candidate = severity_by_root.get(root, "info")
            if candidate == "critical" or (candidate == "warn" and severity == "info"):
                severity = candidate
        summary = {
            "critical": "Structural risk escalation: deterministic breaker may force a flat weight.",
            "warn": "Elevated semantic risk: the post-hoc projection scales exposure down.",
            "info": "No semantic-risk escalation; signal flows through the standard overlays.",
        }[severity]
        events.append(
            NewsEvent(
                id=event_id,
                timestamp=now - timedelta(minutes=4 + index * 23),
                headline=headline,
                source="deterministic-fixture",
                roots=list(roots),
                sentiment_score=sentiment,
                severity=severity,  # type: ignore[arg-type]
                summary=summary,
            )
        )
    return events


# ---------------------------------------------------------------------------
# Term structure & roll calendar
# ---------------------------------------------------------------------------

_CURVE_POINTS: int = 6


def _expiries(spec: InstrumentSpec, as_of: date) -> tuple[date, ...]:
    try:
        from src.execution.ibkr_client import (
            upcoming_contract_expirations,
        )
    except ImportError:  # pragma: no cover - execution layer always present today
        return tuple(
            _fallback_expiry(spec, as_of, offset=offset) for offset in range(_CURVE_POINTS)
        )
    expiries = upcoming_contract_expirations(spec, as_of, count=_CURVE_POINTS)
    return tuple(sorted(set(expiries)))[:_CURVE_POINTS]


def term_structure(market: OfflineMarket) -> TermStructureResponse:
    """Forward curves + roll calendar derived from the real instrument master."""
    as_of_date = market.as_of.date()
    curves: list[TermCurve] = []
    calendar: list[RollCalendarEntry] = []
    for root in market.roots:
        spec = market.instrument_master.get(root)
        expiries = _expiries(spec, as_of_date)
        if not expiries:  # pragma: no cover - defensive
            continue
        front_price = market.price(root)
        slope = _SECTOR_TERM_SLOPE.get(spec.sector, 0.0) + 0.0003 * (
            _stable_unit(root, "slope") - 0.5
        )
        contracts: list[TermContract] = []
        for index, expiration in enumerate(expiries):
            days = max((expiration - as_of_date).days, 0)
            jitter = 0.0004 * (_stable_unit(root, str(index)) - 0.5)
            price = _round_to_tick(
                front_price * (1.0 + slope * days / 365.0 + jitter),
                spec.tick_size,
            )
            contracts.append(
                TermContract(
                    expiration=expiration,
                    price=price,
                    days_to_expiry=days,
                    is_front=index == 0,
                    open_interest=2_000_000 - index * 210_000,
                )
            )
        first, second = contracts[0], contracts[1] if len(contracts) > 1 else contracts[0]
        observed_slope = (second.price - first.price) / first.price if first.price > 0 else 0.0
        days_between = max(second.days_to_expiry - first.days_to_expiry, 1)
        roll_yield = -(observed_slope * 365.0) / days_between
        regime = (
            "flat"
            if abs(observed_slope) < 0.0004
            else "contango"
            if observed_slope > 0
            else "backwardation"
        )
        curves.append(
            TermCurve(
                root=root,
                asset_class=spec.asset_class,
                regime=regime,  # type: ignore[arg-type]
                slope=round(observed_slope, 8),
                roll_yield_annualized=round(roll_yield, 8),
                contracts=contracts,
            )
        )
        roll_date, following = roll_window(spec, as_of_date)
        days_to_roll = (roll_date - as_of_date).days
        calendar.append(
            RollCalendarEntry(
                root=root,
                from_expiration=expiries[0],
                to_expiration=following if following > expiries[0] else expiries[0],
                roll_date=roll_date,
                days_to_roll=days_to_roll,
                trigger=spec.roll.trigger.value,
                method="two-legged MKT" if spec.roll.physical_delivery else "BAG calendar spread",
                status="completed"
                if days_to_roll < 0
                else "imminent"
                if days_to_roll <= 7
                else "scheduled",
            )
        )
    calendar.sort(key=lambda entry: entry.days_to_roll)
    return TermStructureResponse(as_of=market.as_of, curves=curves, roll_calendar=calendar)


def asset_class_or_unknown(master: InstrumentsConfig, root: str) -> AssetClass:
    """Asset class for ``root`` (``UNKNOWN`` when absent from the master)."""
    return master.get(root).asset_class if master.has(root) else AssetClass.UNKNOWN
