"""Unit tests for the FinRL-X feature processor.

Verifies the canonical feature names, causal (no look-ahead) computation,
technical-indicator invariants, futures term structure/carry, OFI aggregation,
cross-asset macro inputs and the Mahalanobis turbulence index.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from src.data import schema as s
from src.data.continuous_roller import ContinuousContractBuilder, VolumeOICrossoverRollSchedule
from src.data.feature_processor import FeatureProcessor

from tests.fixtures.market import FIXED_TIMESTAMP, make_bars_frame

START = FIXED_TIMESTAMP
ROOTS = ("ES", "NQ", "CL")


def fixture_bars(periods: int = 300, roots: tuple[str, ...] = ROOTS) -> pl.DataFrame:
    """Deterministic canonical bars for the requested roots."""
    return make_bars_frame(roots, periods=periods, start=START)


def multi_expiration_bars(periods: int = 120) -> pl.DataFrame:
    """Two-expiration ES frame with a constant contango (next = front * 1.02)."""
    front = fixture_bars(periods, roots=("ES",))
    nxt = front.with_columns(
        (pl.col(s.CLOSE_COLUMN) * 1.02).alias(s.CLOSE_COLUMN),
        (pl.col(s.OPEN_COLUMN) * 1.02).alias(s.OPEN_COLUMN),
        (pl.col(s.HIGH_COLUMN) * 1.02).alias(s.HIGH_COLUMN),
        (pl.col(s.LOW_COLUMN) * 1.02).alias(s.LOW_COLUMN),
        pl.lit(date(2024, 6, 21)).alias(s.EXPIRATION_COLUMN),
        pl.lit("ES_NEXT").alias(s.INSTRUMENT_ID_COLUMN),
    )
    return s.ensure_bars_schema(pl.concat([front, nxt], how="vertical_relaxed"))


def finite_values(frame: pl.DataFrame, column: str) -> np.ndarray:
    """Finite values of ``column`` as a numpy array."""
    values = frame[column].cast(pl.Float64).to_numpy()
    return values[np.isfinite(values)]


def nan_aware_equal(left: pl.DataFrame, right: pl.DataFrame) -> bool:
    """Element-wise equality treating NaN as equal to NaN."""
    for column in left.columns:
        lhs = np.nan_to_num(left[column].cast(pl.Float64).to_numpy(), nan=-1.0e30)
        rhs = np.nan_to_num(right[column].cast(pl.Float64).to_numpy(), nan=-1.0e30)
        if not np.array_equal(lhs, rhs):
            return False
    return True


# ---------------------------------------------------------------------------
# Schema / canonical feature names
# ---------------------------------------------------------------------------


def test_compute_emits_all_canonical_features() -> None:
    frame = fixture_bars(periods=64)
    features = FeatureProcessor().compute(frame)
    missing = [name for name in s.ALL_FEATURES if name not in features.columns]
    assert missing == []
    assert features.columns[: len(s.BAR_COLUMNS)] == list(s.BAR_COLUMNS)
    assert s.ensure_bars_schema(features).equals(features)
    assert features.height == frame.height
    assert features[s.ROOT_SYMBOL_COLUMN].unique().sort().to_list() == sorted(ROOTS)


def test_feature_names_property_matches_schema() -> None:
    processor = FeatureProcessor()
    assert processor.feature_names == s.ALL_FEATURES


def test_compute_is_deterministic() -> None:
    frame = fixture_bars(periods=64)
    processor = FeatureProcessor()
    assert processor.compute(frame).equals(processor.compute(frame))


# ---------------------------------------------------------------------------
# Returns / volatility / technicals
# ---------------------------------------------------------------------------


def test_return_and_momentum_features_match_manual_calculation() -> None:
    bars = fixture_bars(periods=80, roots=("ES",))
    features = FeatureProcessor().compute(bars).sort(s.TIMESTAMP_COLUMN)
    close = bars.sort(s.TIMESTAMP_COLUMN)[s.CLOSE_COLUMN].to_numpy()
    returns = close[1:] / close[:-1] - 1.0
    np.testing.assert_allclose(features[s.RET_1D].to_numpy()[1:], returns, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        features[s.RET_21D].to_numpy()[21:],
        close[21:] / close[:-21] - 1.0,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        features[s.TSMOM_21].to_numpy(), features[s.RET_21D].to_numpy(), equal_nan=True
    )
    np.testing.assert_allclose(
        features[s.TSMOM_126].to_numpy(), features[s.RET_126D].to_numpy(), equal_nan=True
    )


def test_technical_bounds_and_finiteness() -> None:
    features = FeatureProcessor().compute(fixture_bars(periods=200))
    rsi = finite_values(features, s.RSI)
    assert rsi.size > 0
    assert ((rsi >= 0.0) & (rsi <= 100.0)).all()
    assert (finite_values(features, s.ADX) >= 0.0).all()
    assert (finite_values(features, s.ATR) > 0.0).all()
    assert (finite_values(features, s.VOL_20D) >= 0.0).all()
    assert (finite_values(features, s.VOL_60D) >= 0.0).all()
    assert (finite_values(features, s.GARMAN_KLASS_VOL) >= 0.0).all()
    assert (finite_values(features, s.BB_WIDTH) >= 0.0).all()
    bands = features.filter(pl.col(s.BB_UPPER).is_finite() & pl.col(s.BB_LOWER).is_finite())
    assert (bands[s.BB_UPPER].to_numpy() > bands[s.BB_LOWER].to_numpy()).all()
    assert (finite_values(features, s.KAMA) > 0.0).all()


def test_volume_oi_ratio_matches_volume_over_open_interest() -> None:
    bars = fixture_bars(periods=40, roots=("CL",))
    features = FeatureProcessor().compute(bars).sort(s.TIMESTAMP_COLUMN)
    expected = (
        bars.sort(s.TIMESTAMP_COLUMN)[s.VOLUME_COLUMN].to_numpy()
        / bars.sort(s.TIMESTAMP_COLUMN)[s.OPEN_INTEREST_COLUMN].to_numpy()
    )
    np.testing.assert_allclose(features[s.VOLUME_OI_RATIO].to_numpy(), expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# Futures term structure / carry
# ---------------------------------------------------------------------------


def test_multi_expiration_term_structure() -> None:
    frame = multi_expiration_bars(periods=30)
    features = FeatureProcessor().compute(frame)
    slope = features.filter(pl.col(s.TERM_SLOPE).is_not_null())[s.TERM_SLOPE].to_numpy()
    carry = features.filter(pl.col(s.CARRY_ANNUALIZED).is_not_null())[s.CARRY_ANNUALIZED].to_numpy()
    assert slope.size == 30
    assert carry.size == 30
    np.testing.assert_allclose(slope, 0.02, rtol=0.0, atol=1e-12)
    days = (date(2024, 6, 21) - date(2024, 3, 15)).days
    np.testing.assert_allclose(carry, -0.02 * 365.0 / days, rtol=0.0, atol=1e-12)


def test_single_contract_has_unknown_term_structure() -> None:
    features = FeatureProcessor().compute(fixture_bars(periods=10, roots=("ES",)))
    assert finite_values(features, s.TERM_SLOPE).size == 0
    assert finite_values(features, s.CARRY_ANNUALIZED).size == 0


def test_continuous_frame_is_accepted() -> None:
    front = date(2024, 3, 15)
    second = date(2024, 6, 14)
    frame = fixture_bars(periods=40, roots=("ES",)).with_columns(
        pl.lit(front).alias(s.EXPIRATION_COLUMN)
    )
    next_frame = frame.with_columns(
        (pl.col(s.CLOSE_COLUMN) * 1.03).alias(s.CLOSE_COLUMN),
        (pl.col(s.OPEN_COLUMN) * 1.03).alias(s.OPEN_COLUMN),
        (pl.col(s.HIGH_COLUMN) * 1.03).alias(s.HIGH_COLUMN),
        (pl.col(s.LOW_COLUMN) * 1.03).alias(s.LOW_COLUMN),
        pl.lit(second).alias(s.EXPIRATION_COLUMN),
        pl.lit("ES_NEXT").alias(s.INSTRUMENT_ID_COLUMN),
    )
    combined = s.ensure_bars_schema(pl.concat([frame, next_frame], how="vertical_relaxed"))
    continuous = ContinuousContractBuilder(
        roll_schedule=VolumeOICrossoverRollSchedule()
    ).unadjusted(combined, "ES")
    features = FeatureProcessor().compute(continuous)
    assert features.height == continuous.height
    slope = finite_values(features, s.TERM_SLOPE)
    assert slope.size > 0
    np.testing.assert_allclose(slope, 0.03, rtol=0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Turbulence
# ---------------------------------------------------------------------------


def test_turbulence_is_cross_sectional_and_deterministic() -> None:
    bars = fixture_bars(periods=300)
    processor = FeatureProcessor()
    features = processor.compute(bars)
    last_timestamp = features[s.TIMESTAMP_COLUMN].max()
    last_rows = features.filter(pl.col(s.TIMESTAMP_COLUMN) == last_timestamp)
    assert last_rows.height == len(ROOTS)
    assert last_rows[s.TURBULENCE].n_unique() == 1
    assert np.isfinite(last_rows[s.TURBULENCE].to_numpy()).all()
    assert features.equals(processor.compute(bars))


def test_turbulence_needs_history() -> None:
    features = FeatureProcessor().compute(fixture_bars(periods=6))
    assert finite_values(features, s.TURBULENCE).size == 0


# ---------------------------------------------------------------------------
# Order flow imbalance
# ---------------------------------------------------------------------------


def quotes_frame() -> pl.DataFrame:
    """Small deterministic top-of-book frame with known OFI values."""
    stamps = [START + timedelta(minutes=index) for index in range(3)]
    return s.ensure_quote_schema(
        pl.DataFrame(
            {
                s.TIMESTAMP_COLUMN: stamps,
                s.INSTRUMENT_ID_COLUMN: ["ES_TEST"] * 3,
                s.ROOT_SYMBOL_COLUMN: ["ES"] * 3,
                s.EXPIRATION_COLUMN: [date(2024, 3, 15)] * 3,
                s.BID_PRICE_COLUMN: [100.0, 100.5, 100.5],
                s.ASK_PRICE_COLUMN: [101.0, 101.5, 101.5],
                s.BID_SIZE_COLUMN: [10.0, 12.0, 8.0],
                s.ASK_SIZE_COLUMN: [10.0, 9.0, 9.0],
                s.BID_COUNT_COLUMN: [1.0, 1.0, 1.0],
                s.ASK_COUNT_COLUMN: [1.0, 1.0, 1.0],
            }
        )
    )


def test_order_flow_imbalance_formula() -> None:
    ofi = FeatureProcessor().compute_order_flow_imbalance(quotes_frame())
    values = ofi[s.OFI].to_numpy()
    assert np.isnan(values[0])
    assert values[1] == pytest.approx(22.0)
    assert values[2] == pytest.approx(-4.0)


def test_order_flow_imbalance_empty_quotes() -> None:
    empty = s.ensure_quote_schema(
        pl.DataFrame(
            {
                s.TIMESTAMP_COLUMN: [],
                s.INSTRUMENT_ID_COLUMN: [],
                s.ROOT_SYMBOL_COLUMN: [],
                s.EXPIRATION_COLUMN: [],
                s.BID_PRICE_COLUMN: [],
                s.ASK_PRICE_COLUMN: [],
            },
            schema={
                s.TIMESTAMP_COLUMN: pl.Datetime("us", "UTC"),
                s.INSTRUMENT_ID_COLUMN: pl.Utf8,
                s.ROOT_SYMBOL_COLUMN: pl.Utf8,
                s.EXPIRATION_COLUMN: pl.Date,
                s.BID_PRICE_COLUMN: pl.Float64,
                s.ASK_PRICE_COLUMN: pl.Float64,
            },
        )
    )
    ofi = FeatureProcessor().compute_order_flow_imbalance(empty)
    assert ofi.is_empty()


def test_ofi_is_aggregated_into_bar_windows() -> None:
    bars = fixture_bars(periods=4, roots=("ES",))
    quotes = s.ensure_quote_schema(
        pl.DataFrame(
            {
                s.TIMESTAMP_COLUMN: [
                    START + timedelta(hours=1),
                    START + timedelta(hours=2),
                ],
                s.INSTRUMENT_ID_COLUMN: ["ES_TEST"] * 2,
                s.ROOT_SYMBOL_COLUMN: ["ES"] * 2,
                s.EXPIRATION_COLUMN: [date(2024, 3, 15)] * 2,
                s.BID_PRICE_COLUMN: [100.0, 100.5],
                s.ASK_PRICE_COLUMN: [101.0, 101.5],
                s.BID_SIZE_COLUMN: [10.0, 12.0],
                s.ASK_SIZE_COLUMN: [10.0, 9.0],
                s.BID_COUNT_COLUMN: [1.0, 1.0],
                s.ASK_COUNT_COLUMN: [1.0, 1.0],
            }
        )
    )
    features = FeatureProcessor().compute(bars, quotes=quotes).sort(s.TIMESTAMP_COLUMN)
    near = features.filter(pl.col(s.TIMESTAMP_COLUMN) == bars[s.TIMESTAMP_COLUMN].min())
    assert np.isnan(near[s.OFI].to_numpy()[0])
    # Second bar of the day absorbs the (22.0) OFI of the second quote.
    second_bar = features.filter(
        pl.col(s.TIMESTAMP_COLUMN) == bars[s.TIMESTAMP_COLUMN].min() + timedelta(days=1)
    )
    assert second_bar[s.OFI].to_numpy()[0] == pytest.approx(22.0)
    # No quote feed at all => OFI unknown.
    no_quotes = FeatureProcessor().compute(bars)
    assert set(no_quotes[s.OFI].to_list()) == {None}


# ---------------------------------------------------------------------------
# Macro regime inputs
# ---------------------------------------------------------------------------


def exogenous_frame(rows: int = 30) -> pl.DataFrame:
    """Exogenous macro frame with VIX, yields and DXY."""
    stamps = [START + timedelta(days=index) for index in range(rows)]
    return pl.DataFrame(
        {
            s.TIMESTAMP_COLUMN: stamps,
            "vix_front": [15.0] * rows,
            "vix_back": [18.0] * rows,
            "yield_2y": [4.0] * rows,
            "yield_10y": [4.5] * rows,
            "dxy": [100.0 + 0.1 * index for index in range(rows)],
        }
    )


def test_macro_features_from_raw_inputs() -> None:
    bars = fixture_bars(periods=30, roots=("ES",))
    features = FeatureProcessor().compute(bars, exogenous=exogenous_frame())
    assert features[s.MACRO_VIX_TERM_SLOPE].to_numpy()[-1] == pytest.approx(
        np.log(18.0) / np.log(15.0)
    )
    assert features[s.MACRO_YIELD_CURVE_SLOPE].to_numpy()[-1] == pytest.approx(0.5)
    dxy = np.array([100.0 + 0.1 * index for index in range(30)])
    assert features[s.MACRO_DXY_RETURN_5D].to_numpy()[-1] == pytest.approx(dxy[-1] / dxy[-6] - 1.0)
    assert features[s.MACRO_DXY_TREND_21D].to_numpy()[-1] == pytest.approx(dxy[-1] / dxy[-22] - 1.0)


def test_macro_features_from_explicit_columns_and_asof_alignment() -> None:
    bars = fixture_bars(periods=10, roots=("ES",))
    exogenous = pl.DataFrame(
        {
            s.TIMESTAMP_COLUMN: [START - timedelta(hours=1), START + timedelta(days=5)],
            s.MACRO_VIX_TERM_SLOPE: [0.10, 0.20],
            s.MACRO_YIELD_CURVE_SLOPE: [0.30, 0.40],
            s.MACRO_DXY_RETURN_5D: [-0.01, -0.02],
            s.MACRO_DXY_TREND_21D: [-0.03, -0.04],
        }
    )
    features = FeatureProcessor().compute(bars, exogenous=exogenous)
    first_rows = features.filter(pl.col(s.TIMESTAMP_COLUMN) < START + timedelta(days=5))
    assert set(first_rows[s.MACRO_VIX_TERM_SLOPE].to_list()) == {0.10}
    later = features.filter(pl.col(s.TIMESTAMP_COLUMN) >= START + timedelta(days=5))
    assert set(later[s.MACRO_YIELD_CURVE_SLOPE].to_list()) == {0.40}


def test_macro_columns_are_null_without_exogenous_frame() -> None:
    features = FeatureProcessor().compute(fixture_bars(periods=5, roots=("ES",)))
    for name in s.MACRO_FEATURES:
        assert set(features[name].to_list()) == {None}


# ---------------------------------------------------------------------------
# Causality
# ---------------------------------------------------------------------------


def test_features_are_strictly_causal() -> None:
    frame = fixture_bars(periods=260)
    processor = FeatureProcessor()
    baseline = processor.compute(frame)
    cutoff = START + timedelta(days=200)
    mutated = frame.with_columns(
        pl.when(pl.col(s.TIMESTAMP_COLUMN) >= cutoff)
        .then(pl.col(s.CLOSE_COLUMN) * 5.0)
        .otherwise(pl.col(s.CLOSE_COLUMN))
        .alias(s.CLOSE_COLUMN)
    )
    mutated = mutated.with_columns(
        pl.max_horizontal(s.HIGH_COLUMN, s.CLOSE_COLUMN).alias(s.HIGH_COLUMN),
        pl.min_horizontal(s.LOW_COLUMN, s.CLOSE_COLUMN).alias(s.LOW_COLUMN),
    )
    perturbed = processor.compute(mutated)
    feature_columns = list(s.ALL_FEATURES)
    past_baseline = baseline.filter(pl.col(s.TIMESTAMP_COLUMN) < cutoff).select(feature_columns)
    past_perturbed = perturbed.filter(pl.col(s.TIMESTAMP_COLUMN) < cutoff).select(feature_columns)
    assert nan_aware_equal(past_baseline, past_perturbed)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validation_errors() -> None:
    processor = FeatureProcessor()
    with pytest.raises(ValueError, match="empty frame"):
        processor.compute(pl.DataFrame())
    with pytest.raises(TypeError):
        processor.compute("not-a-frame")  # type: ignore[arg-type]
    bars = fixture_bars(periods=5, roots=("ES",))
    with pytest.raises(s.SchemaError, match="root_symbol"):
        processor.compute(bars.drop(s.ROOT_SYMBOL_COLUMN))
    with pytest.raises(s.SchemaError, match="expiration_date"):
        processor.compute(bars.drop(s.EXPIRATION_COLUMN))
    with pytest.raises(ValueError):
        FeatureProcessor(volatility_window=0)
    with pytest.raises(ValueError):
        FeatureProcessor(turbulence_window=0)


def test_custom_windows_are_honoured() -> None:
    bars = fixture_bars(periods=80, roots=("ES",))
    fast = FeatureProcessor(volatility_window=5, long_volatility_window=10).compute(bars)
    slow = FeatureProcessor().compute(bars)
    assert finite_values(fast, s.VOL_20D).size > finite_values(slow, s.VOL_20D).size
    assert finite_values(fast, s.RET_21D).size >= 1
