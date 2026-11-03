"""Phase-10 universe tests: 45-root futures instrument master + sector contract.

These tests are the canonical guard for the Studio universe expansion: seven
sector ids (``equity | rates | energy | metals | fx | agri | crypto``), both
directions of the standard <-> micro links, tradable contract specs and
backward compatibility for the original 13 roots.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from configs.instruments import (
    InstrumentsConfig,
    clear_instruments_cache,
    load_instruments,
)
from src.core.types import AssetClass

REPO_ROOT = Path(__file__).resolve().parents[2]
INSTRUMENTS_PATH = REPO_ROOT / "configs" / "instruments.yaml"

#: Canonical Phase-10 universe: sector id -> declared roots (standard + micro).
EXPECTED_UNIVERSE: dict[str, tuple[str, ...]] = {
    "equity": ("ES", "MES", "NQ", "MNQ", "YM", "MYM", "RTY", "M2K", "NKD"),
    "rates": ("ZT", "ZF", "ZN", "TN", "ZB", "UB", "SR3"),
    "energy": ("CL", "MCL", "NG", "MNG", "RB", "HO"),
    "metals": ("GC", "MGC", "SI", "SIL", "HG", "MHG", "PL", "PA"),
    "fx": ("6E", "6J", "6B", "6A", "6C", "6S"),
    "agri": ("ZC", "ZS", "ZW", "ZM", "ZL", "LE", "HE"),
    "crypto": ("MBT", "MET"),
}

EXPECTED_SECTORS: set[str] = set(EXPECTED_UNIVERSE)

SECTOR_ASSET_CLASS: dict[str, AssetClass] = {
    "equity": AssetClass.EQUITY_INDEX,
    "rates": AssetClass.RATES,
    "energy": AssetClass.ENERGY,
    "metals": AssetClass.METALS,
    "fx": AssetClass.FX,
    "agri": AssetClass.AGRICULTURE,
    "crypto": AssetClass.CRYPTO,
}

#: Standard root -> micro root routing pairs (both directions are validated).
EXPECTED_MICRO_PAIRS: dict[str, str] = {
    "ES": "MES",
    "NQ": "MNQ",
    "YM": "MYM",
    "RTY": "M2K",
    "CL": "MCL",
    "NG": "MNG",
    "GC": "MGC",
    "SI": "SIL",
    "HG": "MHG",
}

#: Original 13-root specs that must stay byte-compatible for existing callers.
LEGACY_ROOTS: tuple[str, ...] = (
    "ES",
    "MES",
    "NQ",
    "MNQ",
    "YM",
    "RTY",
    "CL",
    "MCL",
    "GC",
    "MGC",
    "ZN",
    "ZT",
    "6E",
)


@pytest.fixture(scope="module")
def master() -> InstrumentsConfig:
    """The committed instrument master."""
    clear_instruments_cache()
    return load_instruments(INSTRUMENTS_PATH)


def test_universe_has_45_unique_roots(master: InstrumentsConfig) -> None:
    roots = master.roots()
    assert len(roots) == 45
    assert len(set(roots)) == 45
    assert all(isinstance(root, str) and root for root in roots)


def test_every_expected_root_and_sector_is_declared(master: InstrumentsConfig) -> None:
    expected_roots = {root for roots in EXPECTED_UNIVERSE.values() for root in roots}
    assert set(master.roots()) == expected_roots
    for sector, roots in EXPECTED_UNIVERSE.items():
        declared = {spec.root for spec in master.by_sector(sector)}
        assert declared == set(roots), sector
        for root in roots:
            assert master.get(root).sector == sector


def test_seven_sectors_partition_the_universe(master: InstrumentsConfig) -> None:
    assert set(master.sectors()) == EXPECTED_SECTORS
    assert len(master.sectors()) == 7
    seen: set[str] = set()
    for sector in master.sectors():
        roots = {spec.root for spec in master.by_sector(sector)}
        assert roots, sector
        assert not (roots & seen), f"sector {sector} overlaps another sector"
        seen |= roots
    assert seen == set(master.roots())


def test_sector_asset_class_mapping(master: InstrumentsConfig) -> None:
    for sector, expected in SECTOR_ASSET_CLASS.items():
        specs = master.by_sector(sector)
        assert specs
        for spec in specs:
            assert spec.asset_class is expected, (spec.root, spec.asset_class)
    for asset_class in SECTOR_ASSET_CLASS.values():
        assert master.by_asset_class(asset_class), asset_class


def test_micro_standard_links_are_consistent_both_directions(
    master: InstrumentsConfig,
) -> None:
    declared_pairs = {
        spec.root: spec.micro_root for spec in master.instruments.values() if spec.micro_root
    }
    assert declared_pairs == EXPECTED_MICRO_PAIRS
    for standard_root, micro_root in EXPECTED_MICRO_PAIRS.items():
        standard = master.get(standard_root)
        micro = master.get(micro_root)
        assert master.micro_for(standard_root) is micro
        assert master.standard_for(micro_root) is standard
        assert micro.standard_root == standard_root
        assert standard.micro_root == micro_root
        assert micro.is_micro is True
        assert standard.is_micro is False
        assert micro.multiplier < standard.multiplier
        assert micro.asset_class is standard.asset_class
        assert micro.sector == standard.sector
    # No chains (micro of a micro / standard of a standard) and no cycles.
    for spec in master.instruments.values():
        if spec.is_micro:
            assert spec.micro_root is None
        else:
            assert spec.standard_root is None
    for standard_root in EXPECTED_MICRO_PAIRS:
        seen: set[str] = set()
        current = master.get(standard_root)
        while current.micro_root is not None:
            assert current.root not in seen
            seen.add(current.root)
            current = master.get(current.micro_root)
        assert current.root not in seen


def test_every_root_is_tradable_and_well_formed(master: InstrumentsConfig) -> None:
    for spec in master.instruments.values():
        assert spec.multiplier > 0.0, spec.root
        assert spec.tick_size > 0.0, spec.root
        assert spec.margins.initial > 0.0, spec.root
        assert spec.margins.maintenance > 0.0, spec.root
        assert spec.margins.maintenance <= spec.margins.initial, spec.root
        assert spec.databento_symbol == f"{spec.root}.FUT"
        assert spec.ibkr_exchange
        assert spec.exchange
        assert spec.name
        assert spec.currency == "USD"
        assert 0 <= spec.price_precision <= 8
        assert spec.tick_value == pytest.approx(spec.multiplier * spec.tick_size)
        assert tuple(sorted(set(spec.roll.contract_months))) == spec.roll.contract_months
        assert all(1 <= month <= 12 for month in spec.roll.contract_months)
        assert spec.roll.days_before_fnd >= 0
        assert spec.roll.min_volume_ratio > 0


def test_physical_delivery_roots_are_flagged(master: InstrumentsConfig) -> None:
    physical = {spec.root for spec in master.instruments.values() if spec.roll.physical_delivery}
    assert {
        "CL",
        "MCL",
        "NG",
        "MNG",
        "RB",
        "HO",
        "GC",
        "MGC",
        "SI",
        "SIL",
        "HG",
        "MHG",
        "PL",
        "PA",
        "ZC",
        "ZS",
        "ZW",
        "ZM",
        "ZL",
        "LE",
        "HE",
    } <= physical
    # Financial/crypto contracts are cash-settled.
    for root in ("ES", "NQ", "ZN", "SR3", "6E", "MBT", "MET"):
        assert master.get(root).roll.physical_delivery is False


def test_paired_roots_share_underlying_roll_config(master: InstrumentsConfig) -> None:
    """Micro and standard contracts quote one underlying -> same delivery/roll."""
    for standard_root, micro_root in EXPECTED_MICRO_PAIRS.items():
        standard = master.get(standard_root)
        micro = master.get(micro_root)
        assert micro.roll.contract_months == standard.roll.contract_months
        assert micro.roll.days_before_fnd == standard.roll.days_before_fnd
        assert micro.roll.physical_delivery == standard.roll.physical_delivery
        assert micro.exchange == standard.exchange
        assert micro.currency == standard.currency


def test_key_new_contract_specs(master: InstrumentsConfig) -> None:
    # CME 3M SOFR: $25 per bp point, quarter-bp tick.
    sr3 = master.get("SR3")
    assert sr3.multiplier == pytest.approx(25.0)
    assert sr3.tick_size == pytest.approx(0.0025)
    # Nikkei 225 (USD): 5 USD index multiplier, 5-point tick.
    nkd = master.get("NKD")
    assert nkd.multiplier == pytest.approx(5.0)
    assert nkd.tick_size == pytest.approx(5.0)
    # CME Micro Bitcoin / Micro Ether: 0.1 coin multiplier.
    mbt = master.get("MBT")
    met = master.get("MET")
    assert (mbt.multiplier, mbt.tick_size) == (pytest.approx(0.1), pytest.approx(5.0))
    assert (met.multiplier, met.tick_size) == (pytest.approx(0.1), pytest.approx(0.5))
    # Energy / metals contract units.
    assert master.get("NG").multiplier == pytest.approx(10_000.0)
    assert master.get("SI").multiplier == pytest.approx(5_000.0)
    assert master.get("HG").multiplier == pytest.approx(25_000.0)
    assert master.get("ZC").multiplier == pytest.approx(50.0)
    assert master.get("ZL").multiplier == pytest.approx(600.0)
    # New standard <-> micro pairs route correctly.
    assert master.micro_for("YM").root == "MYM"
    assert master.micro_for("RTY").root == "M2K"
    assert master.micro_for("NG").root == "MNG"
    assert master.micro_for("SI").root == "SIL"
    assert master.micro_for("HG").root == "MHG"
    assert master.standard_for("MYM").root == "YM"
    assert master.standard_for("M2K").root == "RTY"


def test_legacy_roots_keep_their_specs(master: InstrumentsConfig) -> None:
    for root in LEGACY_ROOTS:
        assert master.has(root)
    expected = {
        "ES": (50.0, 0.25, "CME"),
        "NQ": (20.0, 0.25, "CME"),
        "YM": (5.0, 1.0, "CBOT"),
        "RTY": (50.0, 0.10, "CME"),
        "CL": (1000.0, 0.01, "NYMEX"),
        "GC": (100.0, 0.10, "COMEX"),
        "ZN": (1000.0, 0.015625, "CBOT"),
        "ZT": (2000.0, 0.0078125, "CBOT"),
        "6E": (125_000.0, 0.00005, "CME"),
    }
    for root, (multiplier, tick, exchange) in expected.items():
        spec = master.get(root)
        assert spec.multiplier == pytest.approx(multiplier)
        assert spec.tick_size == pytest.approx(tick)
        assert spec.exchange == exchange
    # Existing margin estimates are preserved.
    assert master.get("ES").margins.initial == pytest.approx(15_000.0)
    assert master.get("ES").margins.maintenance == pytest.approx(13_600.0)
    # First five declared roots are unchanged so [:5] fixtures/backtests are stable.
    assert master.roots()[:5] == ("ES", "MES", "NQ", "MNQ", "YM")


def test_defaults_are_unchanged(master: InstrumentsConfig) -> None:
    defaults = master.defaults
    assert defaults.currency == "USD"
    assert defaults.rounding_mode.value == "nearest"
    assert defaults.rebalance_buffer_contracts == pytest.approx(2.0)
    assert defaults.max_margin_utilization == pytest.approx(0.60)
    assert defaults.max_order_contracts == 200


def test_instrument_master_reload_is_deterministic(master: InstrumentsConfig) -> None:
    again = load_instruments(INSTRUMENTS_PATH)
    assert again is master  # lru-cached
    clear_instruments_cache()
    reloaded = load_instruments(INSTRUMENTS_PATH)
    assert reloaded.roots() == master.roots()
    assert len(reloaded.instruments) == len(master.instruments)
