"""Integration tests for the Phase-1 data pipeline.

Raw data -> partitioned Parquet cache -> continuous roll engine -> feature
processor -> immutable MarketState. Everything is offline and deterministic:
the synthetic ``OfflineBarSource`` and the tmp_path repository stand in for
Databento, and the instrument master comes from the committed config.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from configs.settings import Settings
from src.core.types import AssetClass, MarketState, PriceAdjustment
from src.data import schema as s
from src.data.continuous_roller import ContinuousContractBuilder, VolumeOICrossoverRollSchedule
from src.data.databento_loader import DatabentoLoader, OfflineBarSource
from src.data.feature_processor import FeatureProcessor
from src.data.market_state_builder import MarketStateBuilder
from src.data.storage import ParquetRepository

from tests.fixtures.market import FIXED_TIMESTAMP, make_bars_frame

pytestmark = pytest.mark.integration

ROOTS = ("ES", "NQ", "CL")
START = FIXED_TIMESTAMP


def multi_expiration_frame(periods: int = 260) -> pl.DataFrame:
    """Front/second-month ES bars with a deterministic volume crossover."""
    front_expiration = date(2024, 3, 15)
    next_expiration = date(2024, 6, 21)
    front = make_bars_frame(("ES",), periods=periods, start=START).with_columns(
        pl.lit(front_expiration).alias(s.EXPIRATION_COLUMN)
    )
    next_month = front.with_columns(
        (pl.col(s.CLOSE_COLUMN) * 1.015).alias(s.CLOSE_COLUMN),
        (pl.col(s.OPEN_COLUMN) * 1.015).alias(s.OPEN_COLUMN),
        (pl.col(s.HIGH_COLUMN) * 1.015).alias(s.HIGH_COLUMN),
        (pl.col(s.LOW_COLUMN) * 1.015).alias(s.LOW_COLUMN),
        pl.lit(next_expiration).alias(s.EXPIRATION_COLUMN),
        pl.lit("ES_NEXT").alias(s.INSTRUMENT_ID_COLUMN),
        (pl.col(s.VOLUME_COLUMN) * 0.2).alias(s.VOLUME_COLUMN),
    )
    # Back the front contract off late in the window so volume crosses over.
    cutoff = START + timedelta(days=periods - 80)
    frames = [
        front.with_columns(
            pl.when(pl.col(s.TIMESTAMP_COLUMN) >= cutoff)
            .then(pl.col(s.VOLUME_COLUMN) * 0.01)
            .otherwise(pl.col(s.VOLUME_COLUMN))
            .alias(s.VOLUME_COLUMN)
        ),
        next_month,
    ]
    return s.ensure_bars_schema(pl.concat(frames, how="vertical_relaxed"))


def test_offline_bars_through_repository_features_and_state(settings: Settings) -> None:
    repository = ParquetRepository(settings)
    source = OfflineBarSource(ROOTS, seed=5, periods=320)
    window_end = source.start + timedelta(days=260)
    bars = source.load(list(ROOTS), source.start, window_end)
    assert bars.height == 260 * len(ROOTS)

    repository.write_bars(bars)
    cached = repository.read_bars(list(ROOTS), source.start, window_end)
    assert cached.height == bars.height
    assert repository.available_roots() == tuple(sorted(ROOTS))

    features = FeatureProcessor().compute(cached)
    assert features.height == cached.height
    assert set(s.ALL_FEATURES) <= set(features.columns)

    builder = MarketStateBuilder(initial_equity=2_000_000.0)
    decision = source.start + timedelta(days=250)
    state = builder.build(features, decision)
    assert isinstance(state, MarketState)
    assert state.ids == tuple(sorted(ROOTS))
    assert state.equity == 2_000_000.0
    assert state.covariance is not None
    assert state.covariance.shape == (len(ROOTS), len(ROOTS))
    assert state.covariance_instruments == state.ids
    assert all(state.returns[root].size > 0 for root in state.ids)
    for root in state.ids:
        assert state.snapshot(root).asset_class in {
            AssetClass.EQUITY_INDEX,
            AssetClass.ENERGY,
        }
        assert s.ATR in state.snapshot(root).features
        assert state.price(root) > 0

    # A later decision timestamp must include strictly more history.
    earlier = builder.build(features, decision - timedelta(days=30))
    assert earlier.metadata["point_in_time_rows"] < state.metadata["point_in_time_rows"]


def test_loader_serves_cached_synthetic_data_without_credentials(settings: Settings) -> None:
    repository = ParquetRepository(settings)
    loader = DatabentoLoader(settings, repository=repository)
    source = OfflineBarSource(("ES", "CL"), seed=9, periods=128)
    bars = source.load(["ES", "CL"], source.start, source.start + timedelta(days=60))
    repository.write_bars(bars)

    loaded = loader.load(["ES", "CL"], source.start, source.start + timedelta(days=60))
    assert loaded.equals(repository.read_bars(["ES", "CL"]))
    assert loader.available_roots() == ("CL", "ES")

    definitions = source.load_definitions(["ES", "CL"], as_of=source.start)
    repository.store_definitions(definitions)
    cached_definitions = loader.load_definitions(
        ["ES", "CL"], as_of=source.start + timedelta(days=1)
    )
    assert cached_definitions.height == 2
    assert set(cached_definitions["root_symbol"].to_list()) == {"ES", "CL"}


def test_loader_never_fakes_vendor_data(settings: Settings) -> None:
    loader = DatabentoLoader(settings, repository=ParquetRepository(settings))
    source = OfflineBarSource(("ES",), periods=8)
    with pytest.raises(RuntimeError, match="FUTURESRLX_DATABENTO_API_KEY"):
        loader.ingest(["ES"], source.start, source.start + timedelta(days=2))
    assert source.is_synthetic is True


def test_roller_features_and_point_in_time_states(settings: Settings) -> None:
    repository = ParquetRepository(settings)
    bars = multi_expiration_frame(periods=260)
    repository.write_bars(bars)

    cached = repository.read_bars(["ES"])
    builder = ContinuousContractBuilder(roll_schedule=VolumeOICrossoverRollSchedule())
    events = builder.roll_events(cached, "ES")
    assert len(events) == 1
    roll = events[0]
    assert roll.trigger is not None
    assert roll.roll_datetime > START

    panama = builder.build(cached, "ES", adjustment=PriceAdjustment.PANAMA)
    unadjusted = builder.build(cached, "ES", adjustment=PriceAdjustment.UNADJUSTED)
    assert panama.height == unadjusted.height == 260
    assert unadjusted["roll_flag"].sum() == 1
    assert panama["adjusted_close"].to_numpy()[-1] == pytest.approx(
        unadjusted["adjusted_close"].to_numpy()[-1]
    )
    finite_gaps = [value for value in panama["panama_adjustment"].to_numpy() if np.isfinite(value)]
    assert finite_gaps  # non-empty adjustment series
    assert panama["panama_adjustment"].to_numpy()[0] != pytest.approx(
        panama["panama_adjustment"].to_numpy()[-1]
    )

    features = FeatureProcessor().compute(unadjusted)
    assert features["term_slope"].drop_nulls().len() > 0

    state_builder = MarketStateBuilder(initial_equity=1_000_000.0)
    series = state_builder.build_series(
        features,
        [
            roll.roll_datetime - timedelta(days=2),
            roll.roll_datetime,
            roll.roll_datetime + timedelta(days=2),
        ],
    )
    assert len(series) == 3
    assert [state.timestamp for state in series] == sorted(state.timestamp for state in series)
    for state in series:
        assert state.ids == ("ES",)
        assert state.covariance is not None


def test_duckdb_and_roller_agree_on_cached_bars(settings: Settings) -> None:
    repository = ParquetRepository(settings)
    bars = make_bars_frame(ROOTS, periods=90, start=START)
    repository.write_bars(bars)
    count = repository.query_bars("select count(*) as n from bars")["n"].to_list()
    assert count == [90 * len(ROOTS)]

    cached = repository.read_bars(ROOTS)
    builder = ContinuousContractBuilder()
    from_cache = builder.unadjusted(cached, "ES")
    from_memory = builder.unadjusted(bars, "ES")
    assert from_cache.equals(from_memory)


def test_multiple_roots_produce_ordered_state_series(settings: Settings) -> None:
    bars = make_bars_frame(ROOTS, periods=300, start=START)
    features = FeatureProcessor().compute(bars)
    timestamps = [START + timedelta(days=offset) for offset in (150, 200, 250)]
    states = MarketStateBuilder(initial_equity=1_500_000.0).build_series(features, timestamps)
    assert [state.timestamp for state in states] == timestamps
    prices = [state.price("ES") for state in states]
    assert len(set(prices)) == len(prices)
    for state in states:
        assert state.metadata["point_in_time_rows"] > 0
        assert state.covariance is not None
        for root in state.ids:
            assert np.isfinite(state.returns[root]).all()


def test_repository_cache_is_deduplicated_across_repeated_pipeline_runs(
    settings: Settings,
) -> None:
    repository = ParquetRepository(settings)
    bars = make_bars_frame(("ES",), periods=40, start=START)
    for _ in range(3):
        repository.write_bars(bars)
    cached = repository.read_bars(("ES",))
    assert cached.height == 40
    feature_a = FeatureProcessor().compute(cached)
    feature_b = FeatureProcessor().compute(repository.read_bars(("ES",)))
    # Turbulence is deterministic for the same canonical input.
    assert np.allclose(
        np.nan_to_num(feature_a["turbulence"].to_numpy(), nan=-1.0),
        np.nan_to_num(feature_b["turbulence"].to_numpy(), nan=-1.0),
    )
