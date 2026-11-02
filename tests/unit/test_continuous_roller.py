"""Unit tests for the continuous futures roll engine.

Covers the three roll triggers (volume/OI crossover, calendar before LTD,
Databento native cross-verification), the three price adjustments (exact Panama
and proportional math, unadjusted + explicit carry/term slope), zero look-ahead
behaviour and missing-contract handling.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from configs.instruments import load_instruments
from src.core.types import PriceAdjustment, RollTrigger
from src.data import schema as s
from src.data.continuous_roller import (
    CalendarRollSchedule,
    ContinuousContractBuilder,
    DatabentoNativeRollSchedule,
    NativeRollCrossCheck,
    RollScheduleCalculator,
    VolumeOICrossoverRollSchedule,
)
from src.data.contracts import (
    contract_symbol,
    month_code,
    native_continuous_symbol,
    next_contract_expiration,
    parse_contract_symbol,
    parse_native_continuous_symbol,
    third_friday,
)

T0 = datetime(2024, 3, 1, tzinfo=UTC)
TIMES = tuple(T0 + timedelta(days=index) for index in range(7))
A_EXP = date(2024, 3, 15)
B_EXP = date(2024, 6, 14)

#: (root, expiration, timestamp, close, volume, open_interest)
BarRow = tuple[str, date, datetime, float, float, float]


def bars_from(rows: list[BarRow]) -> pl.DataFrame:
    """Build a canonical multi-expiration bar frame from explicit rows."""
    data: dict[str, list[object]] = {column: [] for column in s.BAR_COLUMNS}
    for root, expiration, timestamp, close, volume, open_interest in rows:
        data[s.TIMESTAMP_COLUMN].append(timestamp)
        data[s.INSTRUMENT_ID_COLUMN].append(f"{root}{expiration:%y%m}")
        data[s.ROOT_SYMBOL_COLUMN].append(root)
        data[s.EXPIRATION_COLUMN].append(expiration)
        data[s.OPEN_COLUMN].append(close)
        data[s.HIGH_COLUMN].append(close * 1.01)
        data[s.LOW_COLUMN].append(close * 0.99)
        data[s.CLOSE_COLUMN].append(close)
        data[s.VOLUME_COLUMN].append(volume)
        data[s.OPEN_INTEREST_COLUMN].append(open_interest)
        data[s.SETTLEMENT_COLUMN].append(close)
    return s.ensure_bars_schema(pl.DataFrame(data))


def two_contract_frame() -> pl.DataFrame:
    """Front contract A (t0..t3) and next contract B (t2..t6) with a t3 crossover."""
    rows: list[BarRow] = [
        ("ES", A_EXP, TIMES[0], 100.0, 100.0, 1_000.0),
        ("ES", A_EXP, TIMES[1], 101.0, 100.0, 1_000.0),
        ("ES", A_EXP, TIMES[2], 102.0, 100.0, 1_000.0),
        ("ES", A_EXP, TIMES[3], 103.0, 100.0, 1_000.0),
        ("ES", B_EXP, TIMES[2], 105.0, 10.0, 500.0),
        ("ES", B_EXP, TIMES[3], 106.0, 500.0, 600.0),
        ("ES", B_EXP, TIMES[4], 107.0, 700.0, 800.0),
        ("ES", B_EXP, TIMES[5], 108.0, 800.0, 900.0),
        ("ES", B_EXP, TIMES[6], 109.0, 900.0, 1_000.0),
    ]
    return bars_from(rows)


def rolling_contract_frame(
    roots: tuple[str, ...],
    *,
    front_expiration: date,
    next_expiration: date,
    start: datetime,
    days: int = 51,
    next_starts_day: int = 19,
) -> pl.DataFrame:
    """Two-expiration daily frame for ``roots`` used by calendar tests."""
    rows: list[BarRow] = []
    for root in roots:
        for index in range(days):
            timestamp = start + timedelta(days=index)
            rows.append((root, front_expiration, timestamp, 100.0 + index, 100.0, 1_000.0))
            if index >= next_starts_day:
                rows.append((root, next_expiration, timestamp, 102.0 + index, 100.0, 1_000.0))
    return bars_from(rows)


# ---------------------------------------------------------------------------
# Roll triggers
# ---------------------------------------------------------------------------


def test_volume_crossover_roll_event() -> None:
    frame = two_contract_frame()
    schedule = VolumeOICrossoverRollSchedule()
    events = schedule.compute("ES", frame)
    assert schedule.trigger is RollTrigger.VOLUME_OI_CROSSOVER
    assert isinstance(schedule, RollScheduleCalculator)
    assert len(events) == 1
    event = events[0]
    assert event.root_symbol == "ES"
    assert event.roll_datetime == TIMES[3]
    assert event.from_expiration == A_EXP
    assert event.to_expiration == B_EXP
    assert event.trigger is RollTrigger.VOLUME_OI_CROSSOVER
    assert event.from_price == 103.0
    assert event.to_price == 106.0
    assert event.roll_gap == 3.0
    expected_yield = -(3.0 / 103.0) * (365.0 / (B_EXP - A_EXP).days)
    assert event.roll_yield_annualized == pytest.approx(expected_yield)


def test_volume_crossover_falls_back_to_last_common_bar() -> None:
    frame = two_contract_frame().with_columns(
        pl.when(pl.col(s.EXPIRATION_COLUMN) == B_EXP)
        .then(pl.lit(1.0))
        .otherwise(pl.col(s.VOLUME_COLUMN))
        .alias(s.VOLUME_COLUMN)
    )
    events = VolumeOICrossoverRollSchedule().compute("ES", frame)
    assert events[0].roll_datetime == TIMES[3]  # last common timestamp


def test_volume_schedule_require_both_volume_and_oi() -> None:
    rows: list[BarRow] = [
        ("ES", A_EXP, TIMES[0], 100.0, 100.0, 1_000.0),
        ("ES", A_EXP, TIMES[1], 101.0, 100.0, 1_000.0),
        ("ES", A_EXP, TIMES[2], 102.0, 100.0, 1_000.0),
        ("ES", A_EXP, TIMES[3], 103.0, 100.0, 1_000.0),
        ("ES", A_EXP, TIMES[4], 104.0, 100.0, 1_000.0),
        ("ES", B_EXP, TIMES[2], 105.0, 10.0, 500.0),
        ("ES", B_EXP, TIMES[3], 106.0, 500.0, 600.0),
        ("ES", B_EXP, TIMES[4], 107.0, 700.0, 1_200.0),
        ("ES", B_EXP, TIMES[5], 108.0, 800.0, 1_300.0),
    ]
    frame = bars_from(rows)
    # Volume crosses at t3 but open interest only at t4 -> require_both rolls at t4.
    assert VolumeOICrossoverRollSchedule().compute("ES", frame)[0].roll_datetime == TIMES[3]
    strict = VolumeOICrossoverRollSchedule(require_both=True).compute("ES", frame)
    assert strict[0].roll_datetime == TIMES[4]


def test_volume_schedule_ratio_and_validation() -> None:
    with pytest.raises(ValueError):
        VolumeOICrossoverRollSchedule(min_volume_ratio=0.0)
    with pytest.raises(ValueError):
        VolumeOICrossoverRollSchedule(min_oi_ratio=0.0)
    schedule = VolumeOICrossoverRollSchedule(min_volume_ratio=10.0)
    # B never exceeds 10x A volume -> continuity fallback.
    assert schedule.compute("ES", two_contract_frame())[0].roll_datetime == TIMES[3]


def test_calendar_schedule_rolls_before_ltd() -> None:
    frame = two_contract_frame()
    schedule = CalendarRollSchedule(days_before_expiration=12)
    assert schedule.trigger is RollTrigger.CALENDAR_DAYS_BEFORE_FND
    events = schedule.compute("ES", frame)
    assert events[0].roll_datetime == TIMES[2]  # first shared bar on/after A_EXP - 12d
    assert events[0].from_price == 102.0
    assert events[0].to_price == 105.0

    late = CalendarRollSchedule(days_before_expiration=5).compute("ES", frame)
    assert late[0].roll_datetime == TIMES[3]  # no bar after the threshold -> fallback


def test_calendar_schedule_physical_delivery_rolls_earlier() -> None:
    master = load_instruments()
    start = datetime(2024, 5, 1, tzinfo=UTC)
    frame = rolling_contract_frame(
        ("ES", "CL"),
        front_expiration=date(2024, 6, 21),
        next_expiration=date(2024, 9, 20),
        start=start,
    )
    schedule = CalendarRollSchedule(instrument_master=master)
    es_roll = schedule.compute("ES", frame)[0]
    cl_roll = schedule.compute("CL", frame)[0]
    assert cl_roll.roll_datetime < es_roll.roll_datetime
    assert cl_roll.roll_datetime.date() == date(2024, 6, 9)  # 12 days before LTD
    assert es_roll.roll_datetime.date() == date(2024, 6, 16)  # 5 days before LTD
    # Explicit override: no physical buffer when physical_delivery=False.
    override = CalendarRollSchedule(instrument_master=master, physical_delivery=False).compute(
        "CL", frame
    )[0]
    assert override.roll_datetime.date() == date(2024, 6, 16)


def test_calendar_schedule_validation() -> None:
    with pytest.raises(ValueError):
        CalendarRollSchedule(days_before_expiration=-1)
    with pytest.raises(ValueError):
        CalendarRollSchedule(physical_buffer_days=-1)


# ---------------------------------------------------------------------------
# Databento native symbology cross-verification
# ---------------------------------------------------------------------------


def native_frame(swap_index: int = 1) -> pl.DataFrame:
    """Native continuous frame whose raw symbol changes at ``swap_index``."""
    stamps = list(TIMES[1:5])
    symbols = ["ESH4"] * swap_index + ["ESM4"] * (len(stamps) - swap_index)
    return pl.DataFrame(
        {
            s.TIMESTAMP_COLUMN: stamps,
            s.ROOT_SYMBOL_COLUMN: ["ES"] * len(stamps),
            "raw_symbol": symbols,
            s.CLOSE_COLUMN: [106.0, 106.5, 107.0, 108.0],
        }
    )


def native_local_frame() -> pl.DataFrame:
    """Local frame whose volume crossover matches the native roll at TIMES[2]."""
    rows: list[BarRow] = [
        ("ES", date(2024, 3, 15), TIMES[0], 100.0, 100.0, 1_000.0),
        ("ES", date(2024, 3, 15), TIMES[1], 101.0, 100.0, 1_000.0),
        ("ES", date(2024, 3, 15), TIMES[2], 102.0, 100.0, 1_000.0),
        ("ES", date(2024, 6, 21), TIMES[1], 106.0, 10.0, 500.0),
        ("ES", date(2024, 6, 21), TIMES[2], 107.0, 500.0, 600.0),
        ("ES", date(2024, 6, 21), TIMES[3], 108.0, 600.0, 700.0),
        ("ES", date(2024, 6, 21), TIMES[4], 109.0, 700.0, 800.0),
    ]
    return bars_from(rows)


def test_native_schedule_detects_rolls() -> None:
    schedule = DatabentoNativeRollSchedule(native_frame=native_frame())
    assert schedule.trigger is RollTrigger.DATABENTO_NATIVE
    assert schedule.native_frame is not None
    assert isinstance(schedule, RollScheduleCalculator)
    events = schedule.detect_rolls("ES", native_frame())
    assert len(events) == 1
    event = events[0]
    assert event.trigger is RollTrigger.DATABENTO_NATIVE
    assert event.roll_datetime == TIMES[2]
    assert event.from_expiration == date(2024, 3, 15)
    assert event.to_expiration == date(2024, 6, 21)
    assert (event.from_price, event.to_price) == (106.0, 106.5)
    assert schedule.compute("ES", native_local_frame()) == events


def test_native_cross_check_agrees_and_flags_drift() -> None:
    local = VolumeOICrossoverRollSchedule().compute("ES", native_local_frame())
    schedule = DatabentoNativeRollSchedule(native_frame=native_frame(), tolerance_days=2)
    check = schedule.cross_check("ES", local)
    assert isinstance(check, NativeRollCrossCheck)
    assert check.agrees
    assert len(check.matched) == 1
    assert check.mismatches == ()

    drifted = DatabentoNativeRollSchedule(
        native_frame=native_frame(swap_index=2), tolerance_days=0
    ).cross_check("ES", local)
    assert not drifted.agrees
    assert any("drifts" in message for message in drifted.mismatches)


def test_native_cross_check_reports_unmatched_rolls() -> None:
    native_rolls = DatabentoNativeRollSchedule().detect_rolls("ES", native_frame())
    check = NativeRollCrossCheck.cross_verify(
        root="ES", local_rolls=(), native_rolls=native_rolls, tolerance_days=2
    )
    assert not check.agrees
    assert "no local counterpart" in check.mismatches[0]
    local = VolumeOICrossoverRollSchedule().compute("ES", native_local_frame())
    other = NativeRollCrossCheck.cross_verify(
        root="ES", local_rolls=local, native_rolls=(), tolerance_days=2
    )
    assert not other.agrees
    # A longer native/expiry pair also shows up as a local-only mismatch.
    mismatch = NativeRollCrossCheck.cross_verify(
        root="ES",
        local_rolls=local,
        native_rolls=native_rolls,
        tolerance_days=2,
    )
    assert mismatch.agrees


def test_native_schedule_errors() -> None:
    with pytest.raises(ValueError, match="native continuous frame"):
        DatabentoNativeRollSchedule().compute("ES", two_contract_frame())
    with pytest.raises(ValueError, match="cross_check requires"):
        DatabentoNativeRollSchedule().cross_check("ES", ())
    broken = native_frame().drop("raw_symbol")
    with pytest.raises(s.SchemaError, match="identity column"):
        DatabentoNativeRollSchedule().detect_rolls("ES", broken)
    no_timestamp = native_frame().drop(s.TIMESTAMP_COLUMN)
    with pytest.raises(s.SchemaError, match="timestamp"):
        DatabentoNativeRollSchedule().detect_rolls("ES", no_timestamp)
    unparseable = native_frame().with_columns(
        pl.Series("raw_symbol", ["???", "ESM4", "ESM4", "ESM4"], dtype=pl.Utf8)
    )
    with pytest.raises(s.SchemaError, match="cannot parse"):
        DatabentoNativeRollSchedule().detect_rolls("ES", unparseable)
    empty = pl.DataFrame(
        {
            s.TIMESTAMP_COLUMN: [],
            "raw_symbol": [],
            s.CLOSE_COLUMN: [],
        },
        schema={
            s.TIMESTAMP_COLUMN: pl.Datetime("us", "UTC"),
            "raw_symbol": pl.Utf8,
            s.CLOSE_COLUMN: pl.Float64,
        },
    )
    assert DatabentoNativeRollSchedule().detect_rolls("ES", empty) == ()


# ---------------------------------------------------------------------------
# Price adjustments: exact mathematics
# ---------------------------------------------------------------------------


def test_panama_additive_back_adjustment_exact() -> None:
    frame = two_contract_frame()
    builder = ContinuousContractBuilder(roll_schedule=VolumeOICrossoverRollSchedule())
    series = builder.build(frame, "ES", adjustment=PriceAdjustment.PANAMA)
    assert series.columns[: len(s.CONTINUOUS_COLUMNS)] == list(s.CONTINUOUS_COLUMNS)
    assert s.ensure_continuous_schema(series).equals(series)
    assert series[s.CLOSE_COLUMN].to_list() == [100.0, 101.0, 102.0, 106.0, 107.0, 108.0, 109.0]
    assert series["adjusted_close"].to_list() == [
        103.0,
        104.0,
        105.0,
        106.0,
        107.0,
        108.0,
        109.0,
    ]
    assert series["panama_adjustment"].to_list() == [3.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0]
    # Absolute point changes are preserved (no synthetic jump at the roll).
    adjusted_returns = np.diff(series["adjusted_close"].to_numpy())
    assert adjusted_returns.tolist() == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


def test_panama_negative_gap_sign_convention() -> None:
    rows: list[BarRow] = [
        ("CL", A_EXP, TIMES[0], 100.0, 100.0, 1_000.0),
        ("CL", A_EXP, TIMES[1], 100.0, 100.0, 1_000.0),
        ("CL", B_EXP, TIMES[1], 95.0, 500.0, 1_000.0),
        ("CL", B_EXP, TIMES[2], 96.0, 600.0, 1_000.0),
    ]
    frame = bars_from(rows)
    event = VolumeOICrossoverRollSchedule().compute("CL", frame)[0]
    assert event.roll_gap == -5.0  # backwardation: next below front
    series = ContinuousContractBuilder(roll_schedule=VolumeOICrossoverRollSchedule()).build(
        frame, "CL", adjustment=PriceAdjustment.PANAMA
    )
    # Pre-roll history is shifted down by the (negative) gap.
    assert series["adjusted_close"].to_list() == [95.0, 95.0, 96.0]
    assert series["panama_adjustment"].to_list() == [-5.0, 0.0, 0.0]


def test_proportional_ratio_adjustment_exact() -> None:
    frame = two_contract_frame()
    series = ContinuousContractBuilder(roll_schedule=VolumeOICrossoverRollSchedule()).build(
        frame, "ES", adjustment=PriceAdjustment.PROPORTIONAL
    )
    ratio = 103.0 / 106.0
    expected = [100.0 * ratio, 101.0 * ratio, 102.0 * ratio, 106.0, 107.0, 108.0, 109.0]
    assert series["adjusted_close"].to_list() == pytest.approx(expected)
    assert series["ratio_factor"].to_list()[:3] == pytest.approx([ratio] * 3)
    assert series["ratio_factor"].to_list()[3:] == pytest.approx([1.0] * 4)
    assert series["panama_adjustment"].to_list() == [0.0] * 7


def test_unadjusted_series_exposes_carry_and_term_slope() -> None:
    frame = two_contract_frame()
    series = ContinuousContractBuilder(roll_schedule=VolumeOICrossoverRollSchedule()).unadjusted(
        frame, "ES"
    )
    assert series[s.CLOSE_COLUMN].to_list() == [100.0, 101.0, 102.0, 106.0, 107.0, 108.0, 109.0]
    assert series["adjusted_close"].to_list() == series[s.CLOSE_COLUMN].to_list()
    assert series["roll_flag"].to_list() == [False, False, False, True, False, False, False]
    roll_gaps = series["roll_gap"].to_list()
    assert roll_gaps[3] == 3.0
    assert roll_gaps[:3] == [None, None, None]
    assert series["front_symbol"].to_list()[:3] == ["ES2403"] * 3
    assert series["front_symbol"].to_list()[3:] == ["ES2406"] * 4
    assert series["next_symbol"].to_list()[2] == "ES2406"
    # Term structure is only observable while the next contract trades.
    term = series[s.TERM_SLOPE].to_list()
    days = (B_EXP - A_EXP).days
    assert term[2] == pytest.approx((105.0 - 102.0) / 102.0)
    assert series[s.CARRY_ANNUALIZED].to_list()[2] == pytest.approx(
        -((105.0 - 102.0) / 102.0) * (365.0 / days)
    )
    assert all(value is None for value in term[3:])


# ---------------------------------------------------------------------------
# Zero look-ahead
# ---------------------------------------------------------------------------


def test_future_price_change_never_moves_earlier_roll() -> None:
    frame = two_contract_frame()
    schedule = VolumeOICrossoverRollSchedule()
    before = schedule.compute("ES", frame)
    mutated = frame.with_columns(
        pl.when(pl.col(s.TIMESTAMP_COLUMN) >= TIMES[4])
        .then(pl.col(s.CLOSE_COLUMN) * 10.0)
        .otherwise(pl.col(s.CLOSE_COLUMN))
        .alias(s.CLOSE_COLUMN)
    )
    after = schedule.compute("ES", mutated)
    assert [event.roll_datetime for event in before] == [event.roll_datetime for event in after]
    assert [event.roll_gap for event in before] == [event.roll_gap for event in after]


def test_future_volume_change_never_moves_earlier_roll() -> None:
    frame = two_contract_frame()
    schedule = VolumeOICrossoverRollSchedule()
    before = schedule.compute("ES", frame)
    mutated = frame.with_columns(
        pl.when(pl.col(s.TIMESTAMP_COLUMN) >= TIMES[4])
        .then(pl.lit(1_000_000.0))
        .otherwise(pl.col(s.VOLUME_COLUMN))
        .alias(s.VOLUME_COLUMN)
    )
    after = schedule.compute("ES", mutated)
    assert before[0].roll_datetime == TIMES[3]
    assert after[0].roll_datetime == TIMES[3]


def test_calendar_roll_depends_only_on_expiry_and_availability() -> None:
    frame = two_contract_frame()
    schedule = CalendarRollSchedule(days_before_expiration=12)
    before = schedule.compute("ES", frame)
    mutated = frame.with_columns(pl.col(s.CLOSE_COLUMN) * 3.0)
    after = schedule.compute("ES", mutated)
    assert [event.roll_datetime for event in before] == [event.roll_datetime for event in after]


# ---------------------------------------------------------------------------
# Missing contracts, single contract, multi-root
# ---------------------------------------------------------------------------


def test_missing_next_contract_is_skipped() -> None:
    third_exp = date(2024, 9, 20)
    rows: list[BarRow] = [
        ("ES", A_EXP, TIMES[0], 100.0, 100.0, 1_000.0),
        ("ES", A_EXP, TIMES[1], 101.0, 100.0, 1_000.0),
        ("ES", A_EXP, TIMES[2], 102.0, 100.0, 1_000.0),
        ("ES", third_exp, TIMES[1], 110.0, 10.0, 500.0),
        ("ES", third_exp, TIMES[2], 111.0, 500.0, 600.0),
        ("ES", third_exp, TIMES[3], 112.0, 600.0, 700.0),
    ]
    frame = bars_from(rows)
    events = VolumeOICrossoverRollSchedule().compute("ES", frame)
    assert len(events) == 1
    assert events[0].from_expiration == A_EXP
    assert events[0].to_expiration == third_exp
    series = ContinuousContractBuilder(roll_schedule=VolumeOICrossoverRollSchedule()).build(
        frame, "ES"
    )
    assert series.height == 4
    assert series["roll_flag"].to_list() == [False, False, True, False]


def test_single_contract_series_has_no_rolls() -> None:
    rows: list[BarRow] = [
        ("ZN", A_EXP, TIMES[index], 110.0 + index, 1_000.0, 10_000.0) for index in range(5)
    ]
    frame = bars_from(rows)
    assert VolumeOICrossoverRollSchedule().compute("ZN", frame) == ()
    assert CalendarRollSchedule().compute("ZN", frame) == ()
    series = ContinuousContractBuilder().build(frame, "ZN")
    assert series.height == 5
    assert set(series["roll_flag"].to_list()) == {False}
    assert set(series["next_symbol"].to_list()) == {None}
    assert set(series[s.TERM_SLOPE].to_list()) == {None}
    assert series["adjusted_close"].to_list() == series[s.CLOSE_COLUMN].to_list()


def test_build_frame_handles_every_root() -> None:
    frame = pl.concat(
        [
            two_contract_frame(),
            bars_from(
                [
                    ("NQ", A_EXP, TIMES[index], 16_000.0 + index, 1_000.0, 10_000.0)
                    for index in range(4)
                ]
            ),
        ],
        how="vertical_relaxed",
    )
    builder = ContinuousContractBuilder(roll_schedule=VolumeOICrossoverRollSchedule())
    combined = builder.build_frame(frame)
    assert set(combined[s.ROOT_SYMBOL_COLUMN].to_list()) == {"ES", "NQ"}
    assert combined.height == 11
    assert builder.panama(frame, "ES").height == 7
    assert builder.proportional(frame, "ES").height == 7
    assert builder.unadjusted(frame, "ES").height == 7


def test_calendar_spread_series() -> None:
    spread = ContinuousContractBuilder().calendar_spread(two_contract_frame(), "ES")
    assert spread.columns[:3] == [
        s.TIMESTAMP_COLUMN,
        s.ROOT_SYMBOL_COLUMN,
        "front_symbol",
    ]
    first = spread.row(0, named=True)
    assert first["front_close"] == 102.0
    assert first["next_close"] == 105.0
    assert first["roll_gap"] == 3.0


# ---------------------------------------------------------------------------
# Builder wiring, properties and errors
# ---------------------------------------------------------------------------


def test_builder_trigger_variants_and_properties() -> None:
    calendar_builder = ContinuousContractBuilder(trigger=RollTrigger.CALENDAR_DAYS_BEFORE_FND)
    assert isinstance(calendar_builder.roll_schedule, CalendarRollSchedule)
    assert calendar_builder.adjustment is PriceAdjustment.PANAMA

    volume_builder = ContinuousContractBuilder(trigger=RollTrigger.VOLUME_OI_CROSSOVER)
    assert isinstance(volume_builder.roll_schedule, VolumeOICrossoverRollSchedule)

    native_builder = ContinuousContractBuilder(
        trigger=RollTrigger.DATABENTO_NATIVE, native_frame=native_frame()
    )
    native_events = native_builder.roll_events(native_local_frame(), "ES")
    assert native_events[0].trigger is RollTrigger.DATABENTO_NATIVE

    injected = VolumeOICrossoverRollSchedule()
    custom = ContinuousContractBuilder(roll_schedule=injected)
    assert custom.roll_schedule is injected
    assert custom.instrument_master.has("ES")


def test_builder_cross_verify() -> None:
    builder = ContinuousContractBuilder(roll_schedule=VolumeOICrossoverRollSchedule())
    check = builder.cross_verify(native_local_frame(), native_frame(), "ES")
    assert check.agrees
    assert len(check.matched) == 1


def test_builder_errors() -> None:
    frame = two_contract_frame()
    builder = ContinuousContractBuilder()
    with pytest.raises(ValueError, match="no bars found"):
        builder.build(frame, "GC")
    with pytest.raises(s.SchemaError):
        builder.build(frame.drop(s.VOLUME_COLUMN), "ES")
    with pytest.raises(ValueError):
        builder.build(pl.DataFrame(), "ES")


# ---------------------------------------------------------------------------
# Contract calendar helpers
# ---------------------------------------------------------------------------


def test_contract_symbol_helpers() -> None:
    assert month_code(3) == "H"
    with pytest.raises(ValueError):
        month_code(13)
    assert contract_symbol("es", date(2024, 3, 15)) == "ESH4"
    assert native_continuous_symbol("es", "calendar") == "ES.c.0"
    with pytest.raises(ValueError):
        native_continuous_symbol("ES", "weekly")
    assert parse_native_continuous_symbol("ES.v.0") == ("ES", "volume")
    assert parse_native_continuous_symbol("ES.c.0") == ("ES", "calendar")
    assert parse_native_continuous_symbol("ESH4") is None


def test_parse_contract_symbol_variants() -> None:
    reference = T0
    assert parse_contract_symbol("ESM4", roots=["ES", "MES"], reference=reference) == (
        "ES",
        date(2024, 6, 21),
    )
    assert parse_contract_symbol("6EM4", roots=["6E"], reference=reference) == (
        "6E",
        date(2024, 6, 21),
    )
    assert parse_contract_symbol("CLF4", roots=["CL"], reference=reference) == (
        "CL",
        date(2024, 1, 19),
    )
    assert parse_contract_symbol("ESH24", roots=["ES"], reference=reference) == (
        "ES",
        date(2024, 3, 15),
    )
    assert parse_contract_symbol("XYZQ24", reference=reference) == ("XYZ", date(2024, 8, 16))
    assert parse_contract_symbol("ES.FUT", reference=reference) is None
    assert parse_contract_symbol("ES.v.0", reference=reference) is None
    assert parse_contract_symbol("not-a-symbol", reference=reference) is None
    assert third_friday(2024, 6) == date(2024, 6, 21)


def test_next_contract_expiration() -> None:
    assert next_contract_expiration((3, 6, 9, 12), T0) == date(2024, 3, 15)
    assert next_contract_expiration((3, 6, 9, 12), date(2024, 4, 1)) == date(2024, 6, 21)
    assert next_contract_expiration(tuple(range(1, 13)), date(2024, 1, 2)) == date(2024, 1, 19)
    with pytest.raises(ValueError):
        next_contract_expiration((), date(2024, 1, 2))
