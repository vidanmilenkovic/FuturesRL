"""Unit tests for the deterministic synthetic offline F1/F2 term structure.

Verifies that the offline data source used by DRL training / offline backtests
produces a roll-aware synthetic front/second-month pair with finite, non-zero and
root/regime-consistent ``term_slope`` / ``carry_annualized`` (instead of the old
NaN -> zero-filled fallbacks), that the curve is deterministic and look-ahead
free, and that the public metadata helper is JSON-friendly.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from configs.instruments import load_instruments
from configs.strategy import load_strategy_config
from src.data import schema as s
from src.data.databento_loader import OfflineBarSource
from src.data.feature_processor import FeatureProcessor
from src.data.term_structure import (
    BASE_ANNUALIZED_CARRY,
    CarryProfile,
    SyntheticTermStructure,
    TermStructureCurve,
    offline_term_structure_metadata,
)

START = datetime(2024, 1, 2, 21, 0, tzinfo=UTC)
ROOTS: tuple[str, ...] = ("ES", "NQ", "CL", "GC", "ZN")
PERIODS = 512
EXPECTED_ROWS = 2 * PERIODS * len(ROOTS)  # front + synthetic second month


def make_bars(periods: int = PERIODS, roots: tuple[str, ...] = ROOTS) -> pl.DataFrame:
    """Deterministic canonical offline bars with the synthetic F1/F2 pair."""
    source = OfflineBarSource(roots, seed=7, periods=periods)
    return source.load(list(roots), source.start, source.start + timedelta(days=periods))


def final_feature_rows(frame: pl.DataFrame) -> pl.DataFrame:
    """Last fully-featured row per root (sorted by root)."""
    features = FeatureProcessor().compute(frame)
    return (
        features.sort(s.TIMESTAMP_COLUMN)
        .group_by(s.ROOT_SYMBOL_COLUMN, maintain_order=True)
        .last()
        .sort(s.ROOT_SYMBOL_COLUMN)
    )


# ---------------------------------------------------------------------------
# Synthetic curve model
# ---------------------------------------------------------------------------


def test_curve_math_and_sign_convention() -> None:
    model = SyntheticTermStructure()
    moment = START + timedelta(days=10)
    curve = model.curve(
        "ES",
        moment,
        front_price=4_800.0,
        front_expiration=datetime(2024, 3, 15, tzinfo=UTC).date(),
        second_expiration=datetime(2024, 6, 21, tzinfo=UTC).date(),
    )
    assert isinstance(curve, TermStructureCurve)
    assert isinstance(curve.regime, str)
    days = curve.days_between_expiries
    assert days == (curve.second_expiration - curve.front_expiration).days
    # term_slope = -carry * days/365 and second = front * (1 + slope).
    assert curve.term_slope == pytest.approx(-curve.carry_annualized * days / 365.0)
    assert curve.second_price == pytest.approx(curve.front_price * (1.0 + curve.term_slope))
    # Equity index is in contango: F2 > F1, positive slope, negative long carry.
    assert curve.carry_annualized < 0.0
    assert curve.term_slope > 0.0
    assert curve.second_price > curve.front_price
    assert curve.regime == "contango"
    payload = curve.to_dict()
    assert payload["front_expiration"] == "2024-03-15"
    assert payload["second_expiration"] == "2024-06-21"
    assert json.dumps(payload)


def test_curve_energy_backwardation() -> None:
    model = SyntheticTermStructure()
    moment = START + timedelta(days=10)
    curve = model.curve(
        "CL",
        moment,
        front_price=72.0,
        front_expiration=datetime(2024, 1, 19, tzinfo=UTC).date(),
        second_expiration=datetime(2024, 2, 16, tzinfo=UTC).date(),
    )
    # WTI is in backwardation: F2 < F1, negative slope, positive long carry.
    assert curve.carry_annualized > 0.0
    assert curve.term_slope < 0.0
    assert curve.second_price < curve.front_price
    assert curve.regime == "backwardation"


def test_carry_profiles_follow_instrument_master() -> None:
    master = load_instruments()
    model = SyntheticTermStructure(instrument_master=master)
    es = model.carry_profile("ES")
    cl = model.carry_profile("CL")
    zn = model.carry_profile("ZN")
    assert isinstance(es, CarryProfile)
    assert es.base_annualized_carry == pytest.approx(BASE_ANNUALIZED_CARRY["ES"])
    assert es.base_annualized_carry < 0.0  # contango
    assert cl.base_annualized_carry > 0.0  # backwardation
    assert es.roll_days == master.get("ES").roll.days_before_fnd
    assert zn.roll_days == master.get("ZN").roll.days_before_fnd
    # Physical-delivery energy rolls earlier than its plain calendar buffer.
    assert cl.roll_days > master.get("CL").roll.days_before_fnd
    assert cl.contract_months == tuple(master.get("CL").roll.contract_months)
    assert model.roll_days("ES") == es.roll_days
    assert model.contract_months("ES") == tuple(master.get("ES").roll.contract_months)


def test_carry_is_deterministic_and_varies_across_regimes() -> None:
    model = SyntheticTermStructure()
    moment = START + timedelta(days=40)
    assert model.annualized_carry_at("ES", moment) == model.annualized_carry_at("ES", moment)
    carries = {
        model.annualized_carry_at("CL", START + timedelta(days=offset))
        for offset in range(0, 720, 10)
    }
    assert len(carries) > 10  # evolves with the deterministic regime cycle
    regimes = {
        model.regime_at("CL", START + timedelta(days=offset)) for offset in range(0, 720, 10)
    }
    assert regimes == {"contango", "backwardation"}
    assert model.regime_at("ES", moment) in {"contango", "backwardation", "neutral"}


def test_expiration_chain_and_roll_pair_are_calendar_only() -> None:
    model = SyntheticTermStructure()
    chain = model.expiration_chain("ES", reference=START, horizon_days=800)
    assert len(chain) >= 3
    assert list(chain) == sorted(chain)
    assert len(set(chain)) == len(chain)
    pair = model.pair_at("ES", START, chain)
    assert pair is not None
    front, second = pair
    assert front == chain[0] and second == chain[1]
    # The front switches to the second month only inside the configured roll buffer.
    roll_days = model.roll_days("ES")
    just_before = datetime.combine(
        front - timedelta(days=roll_days), datetime.min.time(), tzinfo=UTC
    ) - timedelta(seconds=1)
    at_roll = datetime.combine(front - timedelta(days=roll_days), datetime.min.time(), tzinfo=UTC)
    assert model.pair_at("ES", just_before, chain) == (front, second)
    assert model.pair_at("ES", at_roll, chain) == (second, chain[2])
    assert model.pair_at("ES", START, ()) is None


def test_model_validation() -> None:
    with pytest.raises(ValueError):
        SyntheticTermStructure(regime_span_days=0)
    with pytest.raises(ValueError):
        SyntheticTermStructure(cycle_days=0.0)
    with pytest.raises(ValueError):
        SyntheticTermStructure(amplitude=1.5)
    model = SyntheticTermStructure()
    with pytest.raises(ValueError):
        model.carry_profile(" ")
    with pytest.raises(ValueError):
        model.expiration_chain("ES", reference=START, horizon_days=-1)
    with pytest.raises(ValueError):
        model.curve(
            "ES",
            START,
            front_price=100.0,
            front_expiration=datetime(2024, 6, 21, tzinfo=UTC).date(),
            second_expiration=datetime(2024, 3, 15, tzinfo=UTC).date(),
        )
    with pytest.raises(ValueError):
        model.curve(
            "ES",
            START,
            front_price=float("nan"),
            front_expiration=datetime(2024, 3, 15, tzinfo=UTC).date(),
            second_expiration=datetime(2024, 6, 21, tzinfo=UTC).date(),
        )


# ---------------------------------------------------------------------------
# OfflineBarSource F1/F2 integration
# ---------------------------------------------------------------------------


def test_offline_source_emits_front_and_second_month_pairs() -> None:
    bars = make_bars(periods=64, roots=("ES", "CL"))
    assert bars.height == 2 * 64 * 2
    per_moment = bars.group_by([s.ROOT_SYMBOL_COLUMN, s.TIMESTAMP_COLUMN]).agg(
        pl.col(s.EXPIRATION_COLUMN).n_unique().alias("expirations")
    )
    assert per_moment["expirations"].unique().to_list() == [2]

    sample = (
        bars.filter((pl.col(s.ROOT_SYMBOL_COLUMN) == "ES") & (pl.col(s.TIMESTAMP_COLUMN) == START))
        .sort(s.EXPIRATION_COLUMN)
        .select([s.EXPIRATION_COLUMN, s.CLOSE_COLUMN, s.INSTRUMENT_ID_COLUMN])
    )
    assert sample.height == 2
    front, second = sample.row(0, named=True), sample.row(1, named=True)
    assert front[s.EXPIRATION_COLUMN] < second[s.EXPIRATION_COLUMN]
    assert front[s.INSTRUMENT_ID_COLUMN] != second[s.INSTRUMENT_ID_COLUMN]
    assert front[s.CLOSE_COLUMN] != second[s.CLOSE_COLUMN]
    model = SyntheticTermStructure()
    expected = model.curve(
        "ES",
        START,
        front_price=float(front[s.CLOSE_COLUMN]),
        front_expiration=front[s.EXPIRATION_COLUMN],
        second_expiration=second[s.EXPIRATION_COLUMN],
    )
    assert float(second[s.CLOSE_COLUMN]) == pytest.approx(expected.second_price)


def test_features_have_finite_nonzero_root_specific_term_structure() -> None:
    frame = make_bars()
    assert frame.height == EXPECTED_ROWS
    last = final_feature_rows(frame)
    slopes = last[s.TERM_SLOPE].to_numpy()
    carries = last[s.CARRY_ANNUALIZED].to_numpy()
    assert np.isfinite(slopes).all()
    assert np.isfinite(carries).all()
    assert (np.abs(slopes) > 0.0).all()
    assert (np.abs(carries) > 0.0).all()
    # Root-species (contango/backwardation) and values are distinct across roots.
    assert last[s.ROOT_SYMBOL_COLUMN].to_list() == sorted(ROOTS)
    assert last[s.TERM_SLOPE].n_unique() == last.height
    by_root = {record[s.ROOT_SYMBOL_COLUMN]: record for record in last.iter_rows(named=True)}
    assert by_root["ES"][s.TERM_SLOPE] > 0.0 and by_root["ES"][s.CARRY_ANNUALIZED] < 0.0
    assert by_root["NQ"][s.CARRY_ANNUALIZED] < 0.0
    assert by_root["CL"][s.TERM_SLOPE] < 0.0 and by_root["CL"][s.CARRY_ANNUALIZED] > 0.0
    assert by_root["GC"][s.CARRY_ANNUALIZED] < 0.0
    assert by_root["ZN"][s.CARRY_ANNUALIZED] < 0.0
    assert len({record[s.CARRY_ANNUALIZED] for record in last.iter_rows(named=True)}) == len(ROOTS)


def test_every_offline_bar_row_has_term_structure() -> None:
    features = FeatureProcessor().compute(make_bars(periods=120, roots=("ES", "CL")))
    assert features[s.TERM_SLOPE].null_count() == 0
    assert features[s.CARRY_ANNUALIZED].null_count() == 0
    assert features[s.TERM_SLOPE].is_nan().sum() == 0
    assert features[s.CARRY_ANNUALIZED].is_nan().sum() == 0


def test_offline_source_term_structure_is_deterministic() -> None:
    first = make_bars(periods=120, roots=("ES", "CL"))
    second = make_bars(periods=120, roots=("ES", "CL"))
    assert first.equals(second)
    features_first = FeatureProcessor().compute(first)
    features_second = FeatureProcessor().compute(second)
    assert features_first.select(list(s.ALL_FEATURES)).equals(
        features_second.select(list(s.ALL_FEATURES))
    )
    assert offline_term_structure_metadata(("ES", "CL"), start=START, periods=120) == (
        offline_term_structure_metadata(("ES", "CL"), start=START, periods=120)
    )


def test_legacy_single_series_mode_has_no_term_structure() -> None:
    source = OfflineBarSource(("ES",), seed=7, periods=32, term_structure=False)
    frame = source.load(["ES"], source.start, source.start + timedelta(days=32))
    assert frame.height == 32
    features = FeatureProcessor().compute(frame)
    assert features[s.TERM_SLOPE].is_nan().all()
    assert features[s.CARRY_ANNUALIZED].is_nan().all()


# ---------------------------------------------------------------------------
# Roll boundaries / no look-ahead
# ---------------------------------------------------------------------------


def test_offline_roll_boundaries_switch_calendar_front_month() -> None:
    source = OfflineBarSource(("CL",), seed=7, periods=PERIODS)
    frame = source.load(["CL"], source.start, source.start + timedelta(days=PERIODS))
    per_moment = (
        frame.sort(s.TIMESTAMP_COLUMN)
        .group_by(s.TIMESTAMP_COLUMN, maintain_order=True)
        .agg(
            pl.col(s.EXPIRATION_COLUMN).min().alias("front"),
            pl.col(s.EXPIRATION_COLUMN).max().alias("second"),
        )
        .sort(s.TIMESTAMP_COLUMN)
    )
    roll_days = SyntheticTermStructure().roll_days("CL")
    changes = per_moment.filter(pl.col("front") != pl.col("front").shift(1)).filter(
        pl.col("front").shift(1).is_not_null()
    )
    assert changes.height >= 3  # monthly WTI cycles over the window
    for record in changes.iter_rows(named=True):
        timestamp = record[s.TIMESTAMP_COLUMN]
        front_before = per_moment.filter(pl.col(s.TIMESTAMP_COLUMN) < timestamp).sort(
            s.TIMESTAMP_COLUMN
        )["front"][-1]
        # The new front was the prior second month; the roll happens exactly at
        # the configured calendar buffer before the prior expiry (no price input).
        assert record["front"] > front_before
        roll_start = datetime.combine(
            front_before - timedelta(days=roll_days), datetime.min.time(), tzinfo=UTC
        )
        assert timestamp >= roll_start
        assert timestamp < roll_start + timedelta(days=2)


def test_no_lookahead_truncated_history_matches_full_history() -> None:
    decision = START + timedelta(days=300)
    source = OfflineBarSource(ROOTS, seed=7, periods=PERIODS)
    full = source.load(list(ROOTS), source.start, source.start + timedelta(days=PERIODS))
    # Strict point-in-time: dropping every future row must not change the
    # decision-time features (roll calendar is calendar-only; windows are causal).
    truncated = full.filter(pl.col(s.TIMESTAMP_COLUMN) <= decision)
    assert truncated.height < full.height
    processor = FeatureProcessor()
    full_features = processor.compute(full)
    truncated_features = processor.compute(truncated)
    full_at_decision = full_features.filter(pl.col(s.TIMESTAMP_COLUMN) == decision)
    truncated_at_decision = truncated_features.filter(pl.col(s.TIMESTAMP_COLUMN) == decision)
    assert full_at_decision.height == truncated_at_decision.height == len(ROOTS)
    columns = list(s.ALL_FEATURES)
    assert full_at_decision.select(columns).equals(truncated_at_decision.select(columns))


# ---------------------------------------------------------------------------
# Metadata helper (consumed by trainer / API checkpoints)
# ---------------------------------------------------------------------------


def test_offline_term_structure_metadata_contract() -> None:
    metadata = offline_term_structure_metadata(ROOTS, start=START, periods=PERIODS)
    assert set(metadata) == set(ROOTS)
    rendered = json.dumps(metadata, sort_keys=True)
    decoded = json.loads(rendered)
    for root in ROOTS:
        info = decoded[root]
        assert set(info) >= {
            "synthetic",
            "front_expiration",
            "second_expiration",
            "roll_days",
            "contract_months",
            "base_annualized_carry",
            "mean_term_slope",
            "min_term_slope",
            "max_term_slope",
            "mean_carry_annualized",
            "regimes",
            "regime_span_days",
        }
        assert info["synthetic"] is True
        assert info["front_expiration"] < info["second_expiration"]
        assert info["roll_days"] >= 0
        assert isinstance(info["contract_months"], list) and info["contract_months"]
        assert abs(float(info["mean_term_slope"])) > 0.0
        assert abs(float(info["mean_carry_annualized"])) > 0.0
        assert "contango" in info["regimes"] or "backwardation" in info["regimes"]
    # Root-specific: energy slope has the opposite sign of equity index.
    assert float(decoded["CL"]["mean_term_slope"]) < 0.0
    assert float(decoded["ES"]["mean_term_slope"]) > 0.0
    assert metadata == offline_term_structure_metadata(ROOTS, start=START, periods=PERIODS)


def test_offline_term_structure_metadata_validation() -> None:
    with pytest.raises(ValueError):
        offline_term_structure_metadata(())
    with pytest.raises(ValueError):
        offline_term_structure_metadata(("ES",), periods=1)
    with pytest.raises(ValueError):
        offline_term_structure_metadata(("ES",), start=datetime(2024, 1, 2))  # naive


# ---------------------------------------------------------------------------
# DRL training data contract (train_drl consumes the offline source unchanged)
# ---------------------------------------------------------------------------


def test_build_training_data_reports_no_feature_fallbacks() -> None:
    from src.strategies.train_drl import build_training_data

    config = load_strategy_config("configs/strategies/drl_ppo_llm_risk.yaml")
    instruments = load_instruments()
    data = build_training_data(config, instruments, seed=7, periods=PERIODS)
    assert data.feature_fallbacks == ()
    names = list(data.feature_names)
    for feature in (s.TERM_SLOPE, s.CARRY_ANNUALIZED, s.VOLUME_OI_RATIO):
        assert feature in names
        values = data.features[:, :, names.index(feature)]
        assert np.isfinite(values).all(), feature
        assert np.abs(values).max() > 0.0, feature
    # Distinct carry/slope across the trained universe (not a single filler value).
    # The trainer now sources every declared root (including paired micros), and a
    # micro shares its standard contract's underlying curve, so distinctness is
    # asserted per family rather than per root.
    slope_index = names.index(s.TERM_SLOPE)
    carry_index = names.index(s.CARRY_ANNUALIZED)
    slopes = np.nanmean(data.features[:, :, slope_index], axis=0)
    carries = np.nanmean(data.features[:, :, carry_index], axis=0)
    assert np.abs(carries).min() > 0.0
    families = {root: (instruments.get(root).standard_root or root) for root in data.instrument_ids}
    assert len(families) == len(data.instrument_ids)  # one entry per trained root
    assert len(set(families.values())) >= 5  # multiple sectors represented
    family_slopes: dict[str, float] = {}
    family_carries: dict[str, float] = {}
    for index, root in enumerate(data.instrument_ids):
        family = families[root]
        family_slopes.setdefault(family, float(slopes[index]))
        family_carries.setdefault(family, float(carries[index]))
        # Paired standard/micro roots quote one underlying => identical curve.
        assert slopes[index] == pytest.approx(family_slopes[family])
        assert carries[index] == pytest.approx(family_carries[family])
    assert len(set(np.round(list(family_slopes.values()), 12))) == len(family_slopes)
    assert len(set(np.round(list(family_carries.values()), 12))) == len(family_carries)


def test_offline_source_signature_is_trainer_compatible() -> None:
    """The DRL trainer builds ``OfflineBarSource(seed=..., periods=..., master=...)``."""
    source = OfflineBarSource(seed=7, periods=64, instrument_master=load_instruments())
    assert source.roots  # default synthetic universe
    assert isinstance(source.start, datetime)
    frame = source.load(source.roots, source.start, source.start + timedelta(days=32))
    assert frame.height > 0
    assert s.ensure_bars_schema(frame).equals(frame)
