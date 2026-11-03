"""Phase-10 universe tests: deterministic offline F1/F2 data for every root.

The expanded 45-root instrument master must be fully usable offline: every root
gets deterministic synthetic front (F1) + second-month (F2) bars with a real
term structure/carry, so feature engineering, universe selection and backtests
work across all seven sectors without network access or look-ahead.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from configs.instruments import InstrumentsConfig, load_instruments
from src.data import schema as s
from src.data.databento_loader import OfflineBarSource, synthetic_reference_price
from src.data.feature_processor import FeatureProcessor
from src.data.market_state_builder import MarketStateBuilder
from src.data.term_structure import offline_term_structure_metadata
from src.strategies.selectors.liquidity import LiquiditySelector

START = datetime(2024, 1, 2, 21, 0, tzinfo=UTC)
PERIODS = 180  # > 126 so every REQUIRED_FEATURES return window is populated
SECTORS = ("equity", "rates", "energy", "metals", "fx", "agri", "crypto")


@pytest.fixture(scope="module")
def master() -> InstrumentsConfig:
    return load_instruments()


@pytest.fixture(scope="module")
def all_roots(master: InstrumentsConfig) -> tuple[str, ...]:
    return master.roots()


@pytest.fixture(scope="module")
def bars(all_roots: tuple[str, ...]) -> pl.DataFrame:
    source = OfflineBarSource(all_roots, seed=7, periods=PERIODS)
    return source.load(list(all_roots), START, START + timedelta(days=PERIODS))


@pytest.fixture(scope="module")
def features(bars: pl.DataFrame) -> pl.DataFrame:
    return FeatureProcessor().compute(bars)


@pytest.fixture(scope="module")
def final_rows(features: pl.DataFrame) -> pl.DataFrame:
    return (
        features.sort(s.TIMESTAMP_COLUMN)
        .group_by(s.ROOT_SYMBOL_COLUMN, maintain_order=True)
        .last()
        .sort(s.ROOT_SYMBOL_COLUMN)
    )


def test_offline_source_covers_every_declared_root(
    bars: pl.DataFrame, all_roots: tuple[str, ...]
) -> None:
    assert len(all_roots) == 45
    assert bars.height == 2 * PERIODS * len(all_roots)
    assert set(bars[s.ROOT_SYMBOL_COLUMN].unique().to_list()) == set(all_roots)
    per_moment = bars.group_by([s.ROOT_SYMBOL_COLUMN, s.TIMESTAMP_COLUMN]).agg(
        pl.col(s.EXPIRATION_COLUMN).n_unique().alias("legs")
    )
    assert per_moment["legs"].unique().to_list() == [2]
    for record in (
        bars.group_by(s.ROOT_SYMBOL_COLUMN)
        .agg(pl.col(s.EXPIRATION_COLUMN).n_unique().alias("expiries"))
        .iter_rows(named=True)
    ):
        assert record["expiries"] >= 2, record[s.ROOT_SYMBOL_COLUMN]
    # Every bar row is canonical and includes a positive volume / open interest.
    assert s.ensure_bars_schema(bars).equals(bars)
    assert (bars[s.VOLUME_COLUMN] > 0.0).all()
    assert (bars[s.OPEN_INTEREST_COLUMN] > 0.0).all()


def test_offline_source_is_deterministic_for_all_roots(all_roots: tuple[str, ...]) -> None:
    first_source = OfflineBarSource(all_roots, seed=7, periods=PERIODS)
    second_source = OfflineBarSource(all_roots, seed=7, periods=PERIODS)
    first = first_source.load(list(all_roots), START, START + timedelta(days=PERIODS))
    second = second_source.load(list(all_roots), START, START + timedelta(days=PERIODS))
    assert first.equals(second)
    assert FeatureProcessor().compute(first).equals(FeatureProcessor().compute(second))


def test_features_are_finite_and_structured_for_every_root(
    final_rows: pl.DataFrame, all_roots: tuple[str, ...]
) -> None:
    assert final_rows.height == len(all_roots)
    assert final_rows[s.ROOT_SYMBOL_COLUMN].to_list() == sorted(all_roots)
    for feature in s.REQUIRED_FEATURES:
        values = final_rows[feature].to_numpy()
        assert np.isfinite(values).all(), feature
    slopes = final_rows[s.TERM_SLOPE].to_numpy()
    carries = final_rows[s.CARRY_ANNUALIZED].to_numpy()
    assert np.isfinite(slopes).all() and np.isfinite(carries).all()
    assert (np.abs(slopes) > 0.0).all()
    assert (np.abs(carries) > 0.0).all()
    assert np.allclose(np.sign(carries), -np.sign(slopes))
    assert (final_rows[s.VOLUME_OI_RATIO].to_numpy() > 0.0).all()


def test_term_structure_features_have_no_nans_anywhere(features: pl.DataFrame) -> None:
    for name in (s.TERM_SLOPE, s.CARRY_ANNUALIZED):
        assert features[name].null_count() == 0, name
        assert features[name].is_nan().sum() == 0, name


def test_sector_carry_profiles_are_consistent(master: InstrumentsConfig) -> None:
    """Per-root mean slope signs follow the documented sector carry profiles."""
    metadata = offline_term_structure_metadata(list(master.roots()), start=START, periods=PERIODS)
    assert len(metadata) == 45
    json.dumps(metadata)
    # Backwardated sectors/contracts: positive long carry, negative slope.
    for root in ("CL", "MCL", "NG", "MNG", "HG", "MHG", "6B", "6C"):
        assert float(metadata[root]["mean_carry_annualized"]) > 0.0, root
        assert float(metadata[root]["mean_term_slope"]) < 0.0, root
    # Contango sectors: negative long carry, positive slope.
    for root in ("ES", "NQ", "ZN", "GC", "ZC", "MBT", "MET", "6J"):
        assert float(metadata[root]["mean_carry_annualized"]) < 0.0, root
        assert float(metadata[root]["mean_term_slope"]) > 0.0, root
    for root, info in metadata.items():
        spec = master.get(root)
        assert info["roll_days"] == (
            spec.roll.days_before_fnd + (7 if spec.roll.physical_delivery else 0)
        )
        assert info["contract_months"] == list(spec.roll.contract_months)
        assert info["regimes"]


def test_synthetic_reference_prices_cover_every_root(master: InstrumentsConfig) -> None:
    for root in master.roots():
        assert synthetic_reference_price(root) > 0.0, root
        assert synthetic_reference_price(root) < 1_000_000.0, root
    # Micro and standard contracts share the same underlying price level.
    for standard_root in ("ES", "NQ", "YM", "RTY", "CL", "NG", "GC", "SI", "HG"):
        micro_root = master.micro_for(standard_root).root
        assert synthetic_reference_price(standard_root) == pytest.approx(
            synthetic_reference_price(micro_root)
        )


def test_paired_micro_standard_share_the_same_underlying_mark(
    bars: pl.DataFrame, master: InstrumentsConfig
) -> None:
    """``P_micro == P_standard`` for F1 *and* F2 (ContractTranslator invariant)."""
    pairs = {
        spec.root: spec.micro_root
        for spec in master.instruments.values()
        if spec.micro_root is not None
    }
    assert len(pairs) == 9
    price_columns = (
        s.OPEN_COLUMN,
        s.HIGH_COLUMN,
        s.LOW_COLUMN,
        s.CLOSE_COLUMN,
        s.SETTLEMENT_COLUMN,
        s.EXPIRATION_COLUMN,
    )
    for standard_root, micro_root in pairs.items():
        standard = bars.filter(pl.col(s.ROOT_SYMBOL_COLUMN) == standard_root).sort(
            [s.TIMESTAMP_COLUMN, s.EXPIRATION_COLUMN]
        )
        micro = bars.filter(pl.col(s.ROOT_SYMBOL_COLUMN) == micro_root).sort(
            [s.TIMESTAMP_COLUMN, s.EXPIRATION_COLUMN]
        )
        assert standard.height == micro.height == 2 * PERIODS, standard_root
        for column in price_columns:
            assert standard[column].to_list() == micro[column].to_list(), (standard_root, column)
        # The micro keeps its own deterministic volume/open-interest and ids.
        assert (micro[s.VOLUME_COLUMN] > 0.0).all()
        assert (micro[s.OPEN_INTEREST_COLUMN] > 0.0).all()


def test_offline_metadata_is_deterministic_and_serialisable(master: InstrumentsConfig) -> None:
    first = offline_term_structure_metadata(list(master.roots()), start=START, periods=PERIODS)
    second = offline_term_structure_metadata(list(master.roots()), start=START, periods=PERIODS)
    assert first == second
    for root, info in first.items():
        assert info["synthetic"] is True
        assert abs(float(info["mean_term_slope"])) > 0.0, root
        assert abs(float(info["mean_carry_annualized"])) > 0.0, root
        assert info["front_expiration"] < info["second_expiration"]


def test_liquidity_selection_works_across_all_sectors(
    final_rows: pl.DataFrame, master: InstrumentsConfig
) -> None:
    features = final_rows
    state = MarketStateBuilder(instrument_master=master).build(
        features, START + timedelta(days=PERIODS - 1)
    )
    assert len(state) == 45
    unrestricted = LiquiditySelector(
        min_avg_dollar_volume=0.0, min_open_interest=0.0, top_n=None
    ).select(state)
    assert len(unrestricted) == 45
    assert set(unrestricted) == set(master.roots())
    limited = LiquiditySelector(min_avg_dollar_volume=0.0, min_open_interest=0.0, top_n=10).select(
        state
    )
    assert len(limited) == 10
    assert set(limited) <= set(master.roots())
    sectors = {master.get(root).sector for root in limited}
    assert len(sectors) >= 2


def test_every_root_has_matching_definitions(all_roots: tuple[str, ...]) -> None:
    source = OfflineBarSource(all_roots, seed=7, periods=32)
    bars = source.load(list(all_roots), START, START + timedelta(days=32))
    definitions = source.load_definitions(list(all_roots), as_of=START)
    bar_ids = set(bars[s.INSTRUMENT_ID_COLUMN].unique().to_list())
    definition_ids = set(definitions[s.INSTRUMENT_ID_COLUMN].unique().to_list())
    assert bar_ids <= definition_ids
    assert set(definitions[s.ROOT_SYMBOL_COLUMN].unique().to_list()) == set(all_roots)
    assert definitions.height >= len(all_roots)


def test_legacy_default_universe_is_unchanged() -> None:
    """Backward compatibility: no-argument offline source keeps the 5-root default."""
    source = OfflineBarSource(seed=7, periods=32)
    assert source.roots == tuple(sorted(("ES", "NQ", "CL", "GC", "ZN")))
    frame = source.load(list(source.roots), START, START + timedelta(days=32))
    assert frame.height == 2 * 32 * len(source.roots)
    assert set(frame[s.ROOT_SYMBOL_COLUMN].unique().to_list()) == set(source.roots)


def test_each_sector_has_offline_curve_features(bars: pl.DataFrame) -> None:
    """Every sector id is represented in the offline feature frame with carry."""
    features = FeatureProcessor().compute(
        bars.filter(pl.col(s.ROOT_SYMBOL_COLUMN).is_in(["ES", "ZN", "CL", "GC", "6E", "ZC", "MBT"]))
    )
    by_root = {
        record[s.ROOT_SYMBOL_COLUMN]: record
        for record in features.sort(s.TIMESTAMP_COLUMN)
        .group_by(s.ROOT_SYMBOL_COLUMN, maintain_order=True)
        .last()
        .iter_rows(named=True)
    }
    assert set(by_root) == {"ES", "ZN", "CL", "GC", "6E", "ZC", "MBT"}
    assert len({record[s.CARRY_ANNUALIZED] for record in by_root.values()}) == len(by_root)
