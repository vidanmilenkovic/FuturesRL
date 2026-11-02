"""Unit tests for the point-in-time MarketStateBuilder."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from configs.settings import Settings
from src.core.types import AssetClass, MarketState
from src.data import schema as s
from src.data.feature_processor import FeatureProcessor
from src.data.market_state_builder import MarketStateBuilder

from tests.fixtures.market import FIXED_TIMESTAMP, make_bars_frame

START = FIXED_TIMESTAMP
ROOTS = ("ES", "NQ", "CL")


def fixture_bars(periods: int = 260, roots: tuple[str, ...] = ROOTS) -> pl.DataFrame:
    """Deterministic canonical daily bars."""
    return make_bars_frame(roots, periods=periods, start=START)


def multi_expiration_bars(periods: int = 120) -> pl.DataFrame:
    """Two-expiration ES frame so the snapshot carries term structure features."""
    front = fixture_bars(periods, roots=("ES",)).with_columns(
        pl.lit(date(2024, 3, 15)).alias(s.EXPIRATION_COLUMN)
    )
    nxt = front.with_columns(
        (pl.col(s.CLOSE_COLUMN) * 1.02).alias(s.CLOSE_COLUMN),
        (pl.col(s.OPEN_COLUMN) * 1.02).alias(s.OPEN_COLUMN),
        (pl.col(s.HIGH_COLUMN) * 1.02).alias(s.HIGH_COLUMN),
        (pl.col(s.LOW_COLUMN) * 1.02).alias(s.LOW_COLUMN),
        pl.lit(date(2024, 6, 21)).alias(s.EXPIRATION_COLUMN),
        pl.lit("ES_NEXT").alias(s.INSTRUMENT_ID_COLUMN),
    )
    return s.ensure_bars_schema(pl.concat([front, nxt], how="vertical_relaxed"))


def moment(offset_days: int) -> datetime:
    """Decision timestamp inside the generated window."""
    return START + timedelta(days=offset_days)


# ---------------------------------------------------------------------------
# Core construction
# ---------------------------------------------------------------------------


def test_build_from_raw_bars() -> None:
    bars = fixture_bars()
    builder = MarketStateBuilder(initial_equity=1_000_000.0)
    state = builder.build(bars, moment(200))
    assert isinstance(state, MarketState)
    assert state.timestamp == moment(200)
    assert state.ids == tuple(sorted(ROOTS))
    assert state.equity == 1_000_000.0

    last_rows = bars.filter(pl.col(s.TIMESTAMP_COLUMN) == moment(200))
    for root in ROOTS:
        expected = last_rows.filter(pl.col(s.ROOT_SYMBOL_COLUMN) == root)[s.CLOSE_COLUMN][0]
        snapshot = state.snapshot(root)
        assert snapshot.price == pytest.approx(expected)
        assert snapshot.root_symbol == root
        assert snapshot.asset_class is AssetClass.EQUITY_INDEX or snapshot.asset_class in {
            AssetClass.EQUITY_INDEX,
            AssetClass.ENERGY,
        }
        assert snapshot.expiration is not None
        assert snapshot.volume > 0
        assert snapshot.open_interest > 0
        assert s.RET_1D in snapshot.features
        assert s.VOL_20D in snapshot.features
        assert s.ATR in snapshot.features
        assert set(snapshot.features) <= set(s.SNAPSHOT_FEATURES)

    assert state.covariance is not None
    assert state.covariance.shape == (len(ROOTS), len(ROOTS))
    assert state.covariance_instruments == tuple(sorted(ROOTS))
    assert np.allclose(state.covariance, state.covariance.T)
    assert (np.diag(state.covariance) > 0.0).all()
    for root in ROOTS:
        assert 0.02 < state.annualized_vol(root) < 0.6
        assert state.returns[root].size > 0
        assert np.isfinite(state.returns[root]).all()

    assert state.metadata["universe"] == tuple(sorted(ROOTS))
    assert state.metadata["point_in_time_rows"] == 3 * 201
    assert isinstance(state.metadata["point_in_time_rows"], int)


def test_term_structure_features_reach_the_snapshot() -> None:
    bars = multi_expiration_bars()
    state = MarketStateBuilder().build(bars, moment(100))
    snapshot = state.snapshot("ES")
    assert snapshot.feature(s.TERM_SLOPE) == pytest.approx(0.02, rel=0.0, abs=1e-12)
    assert snapshot.feature(s.CARRY_ANNUALIZED) < 0.0


def test_equity_override_and_default() -> None:
    bars = fixture_bars(periods=80)
    builder = MarketStateBuilder()
    assert builder.build(bars, moment(60)).equity is None
    assert builder.build(bars, moment(60), equity=500_000.0).equity == 500_000.0
    with pytest.raises(ValueError):
        builder.build(bars, moment(60), equity=-1.0)


def test_build_from_feature_frame_matches_raw_bars() -> None:
    bars = fixture_bars(periods=120)
    features = FeatureProcessor().compute(bars)
    builder = MarketStateBuilder(initial_equity=100_000.0)
    from_features = builder.build(features, moment(100))
    from_bars = builder.build(bars, moment(100))
    assert from_features.price("ES") == pytest.approx(from_bars.price("ES"))
    assert np.allclose(from_features.covariance, from_bars.covariance)  # type: ignore[arg-type]
    assert from_features.metadata["point_in_time_rows"] == from_bars.metadata["point_in_time_rows"]


def test_regime_and_metadata_passthrough() -> None:
    bars = fixture_bars(periods=80)
    state = MarketStateBuilder().build(
        bars,
        moment(60),
        regime="risk_off",
        metadata={"macro_lockout": True, "custom_score": 1.5},
    )
    assert state.regime == "risk_off"
    assert state.metadata["macro_lockout"] is True
    assert state.metadata["custom_score"] == 1.5


def test_macro_metadata_from_feature_frame() -> None:
    bars = fixture_bars(periods=60)
    exogenous = pl.DataFrame(
        {
            s.TIMESTAMP_COLUMN: [START + timedelta(days=index) for index in range(60)],
            s.MACRO_VIX_TERM_SLOPE: [0.05] * 60,
            s.MACRO_YIELD_CURVE_SLOPE: [0.25] * 60,
            s.MACRO_DXY_RETURN_5D: [-0.01] * 60,
            s.MACRO_DXY_TREND_21D: [-0.02] * 60,
        }
    )
    features = FeatureProcessor().compute(bars, exogenous=exogenous)
    state = MarketStateBuilder().build(features, moment(50))
    assert state.metadata[s.MACRO_VIX_TERM_SLOPE] == pytest.approx(0.05)
    assert state.metadata[s.MACRO_YIELD_CURVE_SLOPE] == pytest.approx(0.25)
    assert state.metadata[s.MACRO_DXY_RETURN_5D] == pytest.approx(-0.01)
    assert state.metadata[s.MACRO_DXY_TREND_21D] == pytest.approx(-0.02)


# ---------------------------------------------------------------------------
# Point-in-time guarantee
# ---------------------------------------------------------------------------


def test_future_rows_cannot_change_the_state() -> None:
    bars = fixture_bars(periods=260)
    decision = moment(180)
    builder = MarketStateBuilder(initial_equity=1_000_000.0)
    baseline = builder.build(bars, decision)

    future = bars.filter(pl.col(s.TIMESTAMP_COLUMN) > decision).with_columns(
        pl.col(s.CLOSE_COLUMN) * 10.0
    )
    future = future.with_columns(
        pl.max_horizontal(s.HIGH_COLUMN, s.CLOSE_COLUMN).alias(s.HIGH_COLUMN),
        pl.min_horizontal(s.LOW_COLUMN, s.CLOSE_COLUMN).alias(s.LOW_COLUMN),
    )
    combined = pl.concat(
        [bars.filter(pl.col(s.TIMESTAMP_COLUMN) <= decision), future],
        how="vertical_relaxed",
    )
    perturbed = builder.build(combined, decision)

    assert perturbed.ids == baseline.ids
    for root in ROOTS:
        assert perturbed.price(root) == pytest.approx(baseline.price(root))
        for feature_name in (s.RET_1D, s.VOL_20D, s.ATR):
            assert perturbed.feature(root, feature_name) == pytest.approx(
                baseline.feature(root, feature_name)
            )
        assert np.allclose(perturbed.returns[root], baseline.returns[root])
    assert np.allclose(perturbed.covariance, baseline.covariance)  # type: ignore[arg-type]
    assert perturbed.metadata["point_in_time_rows"] == baseline.metadata["point_in_time_rows"]


# ---------------------------------------------------------------------------
# Series / timestamps
# ---------------------------------------------------------------------------


def test_build_series_and_timestamps() -> None:
    bars = fixture_bars(periods=100)
    builder = MarketStateBuilder(initial_equity=1_000_000.0)
    stamps = builder.timestamps(bars)
    assert stamps[0] == START
    assert stamps[-1] == START + timedelta(days=99)
    selected = stamps[60:63]
    series = builder.build_series(bars, selected)
    assert [state.timestamp for state in series] == list(selected)
    assert series[-1].price("ES") == pytest.approx(builder.build(bars, selected[-1]).price("ES"))
    assert builder.latest_timestamp(bars) == stamps[-1]
    assert (
        builder.timestamps(pl.DataFrame(schema={s.TIMESTAMP_COLUMN: pl.Datetime("us", "UTC")}))
        == ()
    )
    with pytest.raises(ValueError):
        builder.latest_timestamp(
            pl.DataFrame(schema={s.TIMESTAMP_COLUMN: pl.Datetime("us", "UTC")})
        )


def test_build_series_from_feature_frame() -> None:
    bars = fixture_bars(periods=60)
    features = FeatureProcessor().compute(bars)
    builder = MarketStateBuilder()
    stamps = builder.timestamps(features)[40:42]
    series = builder.build_series(features, stamps)
    assert len(series) == 2


# ---------------------------------------------------------------------------
# Edge cases / validation
# ---------------------------------------------------------------------------


def test_unknown_root_degrades_to_unknown_asset_class() -> None:
    bars = fixture_bars(periods=40, roots=("ES",)).with_columns(
        pl.lit("ZZZ").alias(s.ROOT_SYMBOL_COLUMN),
        pl.lit("ZZZ_TEST").alias(s.INSTRUMENT_ID_COLUMN),
    )
    state = MarketStateBuilder().build(bars, moment(30))
    assert state.ids == ("ZZZ",)
    assert state.snapshot("ZZZ").asset_class is AssetClass.UNKNOWN


def test_short_history_has_no_covariance() -> None:
    bars = fixture_bars(periods=5, roots=("ES", "NQ"))
    state = MarketStateBuilder().build(bars, moment(4))
    assert state.covariance is None
    assert state.covariance_instruments == ()
    assert state.returns


def test_constructor_validation() -> None:
    with pytest.raises(ValueError):
        MarketStateBuilder(covariance_halflife=0.0)
    with pytest.raises(ValueError):
        MarketStateBuilder(covariance_window=1)
    with pytest.raises(ValueError):
        MarketStateBuilder(covariance_min_periods=1)
    with pytest.raises(ValueError):
        MarketStateBuilder(returns_window=0)
    with pytest.raises(ValueError):
        MarketStateBuilder(initial_equity=0.0)


def test_build_validation_errors() -> None:
    builder = MarketStateBuilder()
    with pytest.raises(TypeError):
        builder.build("not-a-frame", moment(1))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="empty frame"):
        builder.build(pl.DataFrame(), moment(1))
    bars = fixture_bars(periods=10, roots=("ES",))
    with pytest.raises(ValueError, match="no rows at or before"):
        builder.build(bars, START - timedelta(days=1))
    with pytest.raises(ValueError):
        builder.build(bars, datetime(2024, 1, 1))  # naive timestamp


def test_instrument_master_is_lazy(settings: Settings) -> None:
    builder = MarketStateBuilder()
    assert builder.instrument_master.has("ES")
    assert builder.feature_processor.feature_names == s.ALL_FEATURES
    builder_with_master = MarketStateBuilder(feature_processor=FeatureProcessor())
    assert builder_with_master.feature_processor is not None
