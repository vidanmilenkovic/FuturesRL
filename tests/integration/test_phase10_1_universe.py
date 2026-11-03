"""End-to-end verification of the Phase-10.1 expanded futures universe (45+ roots).

This is a **verification-only** suite: it edits no production code. It proves that
the expanded instrument master is internally consistent and that the whole stack
works with it:

1. ``configs/instruments.yaml`` integrity (>= 40 roots, 7 sectors, bidirectional
   standard <-> micro links, no cycles, positive contract specs).
2. ``ContractTranslator`` for *every* root (and micro) with sufficient equity,
   insufficient equity (micro-routing path), shorts and a mixed all-root book -
   margin utilisation never exceeds the 60% ceiling and nothing divides by zero.
3. Deterministic offline states covering all roots + the real ``ComposablePipeline``
   (finite weights, no NaNs, bounded runtime) and every universe selector.
4. Backtest smoke (``run_from_config`` + tiny ``run_walk_forward``) and
   ``train_drl.build_training_data`` all-root coverage with empty feature fallbacks.
5. API exposure (``/api/instruments`` sector-grouped universe, Studio parameter
   catalog universe choices) with ``pytest.skip`` while the concurrent endpoints
   are not registered yet (task-31/task-32).

The expectations are dynamic (derived from the loaded instrument master), so the
suite keeps working as long as the expansion contract holds - no hardcoded root
list. Tests are offline and deterministic.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
from configs.instruments import InstrumentsConfig, load_instruments
from configs.settings import Settings
from configs.strategy import StageConfig, StrategyConfig, TranslatorConfig
from fastapi.testclient import TestClient
from src.api.main import create_app
from src.backtest.engine import (
    BacktestMode,
    BacktestResult,
    run_from_config,
    run_walk_forward,
)
from src.core.types import ContractTargetSet, MarketState, RoundingMode, WeightVector
from src.data import schema as s
from src.data.databento_loader import OfflineBarSource
from src.data.feature_processor import FeatureProcessor
from src.data.market_state_builder import MarketStateBuilder
from src.risk.contract_translator import ContractTranslator
from src.strategies._observation import DEFAULT_OBSERVATION_FEATURES
from src.strategies.allocators import build_allocator
from src.strategies.pipeline import ComposablePipeline, build_pipeline
from src.strategies.selectors import build_selector
from src.strategies.train_drl import build_training_data

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
#: Minimum number of futures roots the expanded universe must carry.
MIN_UNIVERSE_ROOTS: int = 40
#: The seven sectors required by the Studio UX universe picker (config sector ids).
EXPECTED_SECTORS: frozenset[str] = frozenset(
    {"equity", "rates", "energy", "metals", "fx", "agri", "crypto"}
)
#: `/api/instruments` Studio sector ids (UI order) and config-sector -> API-id mapping.
EXPECTED_API_SECTOR_IDS: tuple[str, ...] = (
    "equities",
    "rates",
    "energy",
    "metals",
    "fx",
    "agri",
    "crypto",
)
CONFIG_SECTOR_TO_API_ID: Mapping[str, str] = {
    "equity": "equities",
    "rates": "rates",
    "energy": "energy",
    "metals": "metals",
    "fx": "fx",
    "agri": "agri",
    "crypto": "crypto",
}
#: Sector -> AssetClass mapping required by the expanded config.
SECTOR_ASSET_CLASSES: Mapping[str, str] = {
    "equity": "equity_index",
    "rates": "rates",
    "energy": "energy",
    "metals": "metals",
    "fx": "fx",
    "agri": "agriculture",
    "crypto": "crypto",
}
#: Deterministic offline window used by the integration fixtures.
FRAME_PERIODS: int = 160
#: Number of trailing point-in-time states exercised by pipeline/backtest tests.
STATE_COUNT: int = 36
#: Account equity used by pipeline/backtest/translator fixtures.
EQUITY: float = 1_000_000.0
#: Margin util ceiling from the architecture spec (and translator default).
MARGIN_CEILING: float = 0.60


# ---------------------------------------------------------------------------
# Shared deterministic fixtures (built once per module; no network)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def master() -> InstrumentsConfig:
    """The expanded committed instrument master."""
    return load_instruments(REPO_ROOT / "configs" / "instruments.yaml")


@pytest.fixture(scope="module")
def bars(master: InstrumentsConfig) -> pl.DataFrame:
    """Deterministic offline F1/F2 bars covering every declared root."""
    roots = tuple(master.roots())
    source = OfflineBarSource(roots=roots, seed=7, periods=FRAME_PERIODS, instrument_master=master)
    return source.load(roots, source.start, source.start + timedelta(days=FRAME_PERIODS))


@pytest.fixture(scope="module")
def feature_frame(master: InstrumentsConfig, bars: pl.DataFrame) -> pl.DataFrame:
    """Canonical FinRL-X feature frame for the expanded universe."""
    return FeatureProcessor(instrument_master=master).compute(bars)


@pytest.fixture(scope="module")
def all_states(master: InstrumentsConfig, feature_frame: pl.DataFrame) -> tuple[MarketState, ...]:
    """Trailing point-in-time states covering every root (deterministic)."""
    builder = MarketStateBuilder(instrument_master=master, initial_equity=EQUITY)
    timestamps = builder.timestamps(feature_frame)[-STATE_COUNT:]
    assert timestamps, "offline feature frame carries no timestamps"
    return tuple(builder.build_series(feature_frame, timestamps))


@pytest.fixture(scope="module")
def final_state(all_states: tuple[MarketState, ...]) -> MarketState:
    """Latest state of the offline window."""
    return all_states[-1]


@pytest.fixture(scope="module")
def translator(master: InstrumentsConfig) -> ContractTranslator:
    """Deterministic contract translator bound to the expanded master."""
    return ContractTranslator(instruments=master, rounding_mode=RoundingMode.NEAREST)


@pytest.fixture(scope="module")
def universal_config(master: InstrumentsConfig) -> StrategyConfig:
    """Synthetic all-root strategy config: equal weight over the whole universe."""
    return StrategyConfig(
        id="phase10_1_universe_verification",
        description="Equal-weight all-root verification pipeline (no YAML on disk).",
        universe=StageConfig(
            name="liquidity",
            params={
                "min_avg_dollar_volume": 0.0,
                "min_open_interest": 0.0,
                "min_history_days": 0,
            },
        ),
        allocator=StageConfig(name="equal_weight", params={"gross_leverage": 1.0}),
        timing=(),
        risk=(),
        translator=TranslatorConfig(),
    )


@pytest.fixture(scope="module")
def pipeline(
    master: InstrumentsConfig,
    universal_config: StrategyConfig,
    translator: ContractTranslator,
) -> ComposablePipeline:
    """Production FinRL-X pipeline over the expanded universe."""
    return build_pipeline(universal_config, master, translator=translator)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _all_roots(master: InstrumentsConfig) -> tuple[str, ...]:
    return tuple(master.roots())


def _standard_micro_pairs(master: InstrumentsConfig) -> dict[str, str]:
    return {
        spec.root: spec.micro_root
        for spec in master.instruments.values()
        if spec.micro_root is not None
    }


def _contract_notional_usd(master: InstrumentsConfig, state: MarketState, root: str) -> float:
    spec = master.get(root)
    return state.price(root) * spec.multiplier


def _sufficient_equity(
    master: InstrumentsConfig, state: MarketState, *, weight: float = 0.02
) -> float:
    """Equity that yields ``weight * equity / contract_notional >= 1.5`` for every root."""
    largest = max(_contract_notional_usd(master, state, root) for root in _all_roots(master))
    return largest / weight * 1.5


def _assert_finite_targets(targets: ContractTargetSet, master: InstrumentsConfig) -> None:
    assert isinstance(targets, ContractTargetSet)
    for target in targets:
        assert master.has(target.instrument_id), f"unknown target root {target.instrument_id!r}"
        assert isinstance(target.quantity, int) and not isinstance(target.quantity, bool)
        assert math.isfinite(target.price) and target.price > 0.0
        assert math.isfinite(target.multiplier) and target.multiplier > 0.0
        assert math.isfinite(target.initial_margin) and target.initial_margin >= 0.0
        assert math.isfinite(target.maintenance_margin) and target.maintenance_margin >= 0.0
    assert math.isfinite(targets.gross_notional)
    assert math.isfinite(targets.net_notional)
    assert math.isfinite(targets.total_initial_margin)
    assert targets.total_initial_margin >= 0.0


def _assert_finite_weight_vector(weights: WeightVector, label: str) -> None:
    for root, value in weights.items():
        assert isinstance(root, str) and root
        assert math.isfinite(value), f"{label} weight for {root!r} is not finite: {value!r}"


def _assert_finite_numbers(value: object, *, path: str = "metrics") -> None:
    """Recursively assert every numeric leaf is finite (no NaN/inf)."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite_numbers(item, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _assert_finite_numbers(item, path=f"{path}[{index}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        assert math.isfinite(float(value)), f"{path} is not finite: {value!r}"


# ---------------------------------------------------------------------------
# 1. Instrument master integrity
# ---------------------------------------------------------------------------


def test_instrument_master_is_expanded_and_consistent(master: InstrumentsConfig) -> None:
    roots = _all_roots(master)
    assert len(roots) >= MIN_UNIVERSE_ROOTS, (
        f"expanded universe must carry >= {MIN_UNIVERSE_ROOTS} roots, got {len(roots)}: {roots}"
    )
    assert len(set(roots)) == len(roots), "root list contains duplicates"
    assert set(roots) == set(master.instruments), "root order/keys disagree with the mapping"
    assert set(master.sectors()) == EXPECTED_SECTORS, (
        f"sectors must be exactly {sorted(EXPECTED_SECTORS)}, got {sorted(master.sectors())}"
    )


def test_instrument_specs_are_valid(master: InstrumentsConfig) -> None:
    pairs = _standard_micro_pairs(master)
    for root in _all_roots(master):
        spec = master.get(root)
        assert spec.root == root
        assert spec.sector in EXPECTED_SECTORS
        assert spec.asset_class.value == SECTOR_ASSET_CLASSES[spec.sector], (
            f"{root}: asset_class {spec.asset_class.value!r} does not match sector "
            f"{spec.sector!r} (expected {SECTOR_ASSET_CLASSES[spec.sector]!r})"
        )
        assert spec.multiplier > 0.0 and math.isfinite(spec.multiplier)
        assert spec.tick_size > 0.0 and math.isfinite(spec.tick_size)
        assert math.isfinite(spec.margins.initial) and spec.margins.initial > 0.0
        assert math.isfinite(spec.margins.maintenance) and spec.margins.maintenance > 0.0
        assert spec.margins.maintenance <= spec.margins.initial
        assert spec.name.strip()
        assert spec.exchange.strip()
        assert spec.databento_symbol.strip()
        assert spec.ibkr_exchange.strip()
        assert spec.currency.isalpha() and len(spec.currency) == 3
        assert spec.roll.contract_months, f"{root}: empty roll contract_months"
        assert spec.roll.days_before_fnd >= 0
        assert isinstance(spec.roll.physical_delivery, bool)

    # Standard <-> micro links must be bidirectional, single-level and cycle-free.
    for root, micro_root in pairs.items():
        standard = master.get(root)
        micro = master.get(micro_root)
        assert standard.standard_root is None, (
            f"{root}: standard root must not declare standard_root"
        )
        assert standard.micro_root == micro_root
        assert micro.standard_root == root, (
            f"{micro_root}: standard_root {micro.standard_root!r} must be {root!r}"
        )
        assert micro.micro_root is None, f"{micro_root}: a micro root must not declare micro_root"
        assert micro.sector == standard.sector, f"{micro_root}: sector mismatch with {root}"
        assert micro.asset_class is standard.asset_class
        assert micro.multiplier < standard.multiplier, (
            f"{micro_root}: micro multiplier must be smaller than {root}"
        )
        assert micro.tick_size <= standard.tick_size * 1.0 + 1e-12

    # Every micro root is reachable from exactly one standard root (no orphans).
    all_micro_roots = set(pairs.values())
    declared_micro = {spec.root for spec in master.instruments.values() if spec.standard_root}
    assert all_micro_roots == declared_micro


# ---------------------------------------------------------------------------
# 2. ContractTranslator across every root
# ---------------------------------------------------------------------------


def test_translator_handles_every_root_with_sufficient_equity(
    master: InstrumentsConfig, final_state: MarketState, translator: ContractTranslator
) -> None:
    equity = _sufficient_equity(master, final_state)
    assert math.isfinite(equity) and equity > 0.0
    for root in _all_roots(master):
        weight = 0.02
        targets = translator.translate(WeightVector({root: weight}), final_state, equity=equity)
        _assert_finite_targets(targets, master)
        assert targets.get(root) is not None, (
            f"{root}: sufficient equity ({equity:,.0f}) must produce a standard/micro leg"
        )
        assert targets.get(root).quantity > 0  # type: ignore[union-attr]
        assert translator.margin_utilization(targets, equity) <= MARGIN_CEILING + 1e-9

        short = translator.translate(WeightVector({root: -weight}), final_state, equity=equity)
        _assert_finite_targets(short, master)
        short_target = short.get(root)
        assert short_target is not None, f"{root}: short with sufficient equity must emit a leg"
        assert short_target.quantity < 0
        assert short.net_notional < 0.0
        assert translator.margin_utilization(short, equity) <= MARGIN_CEILING + 1e-9


def test_translator_micro_routing_for_standard_roots(
    master: InstrumentsConfig, final_state: MarketState, translator: ContractTranslator
) -> None:
    """Insufficient equity routes the standard root's residual into its micro pair."""
    for root, micro_root in _standard_micro_pairs(master).items():
        standard_notional = _contract_notional_usd(master, final_state, root)
        # w=0.5, equity=0.5*notional -> raw 0.25 standard => micro residual ~0.25*ratio.
        equity = 0.5 * standard_notional
        targets = translator.translate(WeightVector({root: 0.5}), final_state, equity=equity)
        _assert_finite_targets(targets, master)
        assert targets.get(root) is None, (
            f"{root}: 0.25 raw standard contracts must round to zero standard contracts"
        )
        micro_target = targets.get(micro_root)
        assert micro_target is not None, (
            f"{root}: insufficient equity must route the residual into {micro_root}, got "
            f"{dict(targets.quantities)!r}"
        )
        assert micro_target.quantity > 0
        assert translator.margin_utilization(targets, equity) <= MARGIN_CEILING + 1e-9

        short = translator.translate(WeightVector({root: -0.5}), final_state, equity=equity)
        _assert_finite_targets(short, master)
        assert short.get(root) is None
        short_micro = short.get(micro_root)
        assert short_micro is not None and short_micro.quantity < 0
        assert translator.margin_utilization(short, equity) <= MARGIN_CEILING + 1e-9


def test_micro_roots_are_never_routed_upwards(
    master: InstrumentsConfig, final_state: MarketState, translator: ContractTranslator
) -> None:
    """Explicit micro-root targets stay micro (never upsampled, never cross-routed)."""
    equity = _sufficient_equity(master, final_state)
    micro_roots = sorted(spec.root for spec in master.instruments.values() if spec.standard_root)
    assert micro_roots, "expanded universe must contain paired micro contracts"
    for micro_root in micro_roots:
        targets = translator.translate(WeightVector({micro_root: 0.02}), final_state, equity=equity)
        _assert_finite_targets(targets, master)
        ids = {target.instrument_id for target in targets}
        assert ids <= {micro_root}, (
            f"{micro_root}: micro root routed into {sorted(ids - {micro_root})}"
        )
        target = targets.get(micro_root)
        assert target is not None and target.quantity > 0


def test_translator_all_roots_mixed_book_respects_margin_ceiling(
    master: InstrumentsConfig, final_state: MarketState, translator: ContractTranslator
) -> None:
    roots = _all_roots(master)
    notionals = sorted((_contract_notional_usd(master, final_state, root), root) for root in roots)
    cheapest = [root for _, root in notionals[:4]]
    # Explicit long/short cheap roots (guaranteed to clear the discrete threshold)
    # plus a tiny alternating weight for every remaining root: an all-root mixed book.
    weights: dict[str, float] = {
        cheapest[0]: -0.5,
        cheapest[1]: 0.5,
        cheapest[2]: -0.3,
        cheapest[3]: 0.3,
    }
    remainder = [root for root in roots if root not in weights]
    tiny_weight = 0.5 / len(remainder)
    for index, root in enumerate(remainder):
        weights[root] = tiny_weight if index % 2 == 0 else -tiny_weight
    targets = translator.translate(WeightVector(weights), final_state, equity=EQUITY)
    _assert_finite_targets(targets, master)
    utilization = translator.margin_utilization(targets, EQUITY)
    assert utilization <= MARGIN_CEILING + 1e-9, (
        f"all-root book margin utilisation {utilization:.4f} exceeds {MARGIN_CEILING}"
    )
    assert len(targets) > 0, "all-root mixed book at 1M equity must emit some contracts"
    assert any(target.quantity > 0 for target in targets), "mixed book must contain longs"
    assert any(target.quantity < 0 for target in targets), "mixed book must contain shorts"

    # Tiny equity must not divide by zero or violate the ceiling (empty/flat is fine).
    tiny = translator.translate(WeightVector(weights), final_state, equity=1.0)
    _assert_finite_targets(tiny, master)
    assert translator.margin_utilization(tiny, 1.0) <= MARGIN_CEILING + 1e-9


def test_offline_paired_micro_roots_share_the_same_underlying_mark(
    master: InstrumentsConfig, bars: pl.DataFrame
) -> None:
    """A standard root and its micro pair are the same underlying: identical marks.

    `ContractTranslator` sizes and routes standard/micro residuals assuming
    ``P_micro == P_standard`` (the documented same-underlying invariant); an offline
    synthetic source that gives each paired root an independent price path breaks
    residual sizing and micro routing.
    """
    invalid: list[str] = []
    for standard_root, micro_root in _standard_micro_pairs(master).items():
        standard_closes = (
            bars.filter(pl.col(s.ROOT_SYMBOL_COLUMN) == standard_root)
            .sort(s.TIMESTAMP_COLUMN)
            .get_column(s.CLOSE_COLUMN)
            .tail(10)
            .to_numpy()
        )
        micro_closes = (
            bars.filter(pl.col(s.ROOT_SYMBOL_COLUMN) == micro_root)
            .sort(s.TIMESTAMP_COLUMN)
            .get_column(s.CLOSE_COLUMN)
            .tail(10)
            .to_numpy()
        )
        if standard_closes.size != micro_closes.size or standard_closes.size == 0:
            invalid.append(f"{standard_root}/{micro_root}: missing/uneven bars")
            continue
        ratios = micro_closes / standard_closes
        if not np.allclose(ratios, 1.0, rtol=1e-9, atol=1e-12):
            invalid.append(
                f"{standard_root}/{micro_root}: price ratio range "
                f"[{float(np.min(ratios)):.6f}, {float(np.max(ratios)):.6f}] != 1"
            )
    assert not invalid, (
        "offline paired standard/micro roots must share the same underlying price "
        "(same mark) for translator micro routing/tracking to be correct: " + "; ".join(invalid)
    )


# ---------------------------------------------------------------------------
# 3. Offline states, pipeline and universe selectors over the expanded universe
# ---------------------------------------------------------------------------


def test_offline_source_and_features_cover_every_root(
    master: InstrumentsConfig, bars: pl.DataFrame, feature_frame: pl.DataFrame
) -> None:
    roots = _all_roots(master)
    bar_roots = set(bars[s.ROOT_SYMBOL_COLUMN].unique().to_list())
    assert bar_roots == set(roots), (
        f"offline bars missing roots: {sorted(set(roots) - bar_roots)!r}"
    )
    # Every (root, timestamp) carries the deterministic F1/F2 pair.
    pair_counts = (
        bars.group_by([s.ROOT_SYMBOL_COLUMN, s.TIMESTAMP_COLUMN])
        .len()
        .get_column("len")
        .unique()
        .to_list()
    )
    assert pair_counts == [2], (
        f"offline source must emit exactly 2 expirations per bar, got {pair_counts}"
    )

    feature_roots = set(feature_frame[s.ROOT_SYMBOL_COLUMN].unique().to_list())
    assert feature_roots == set(roots), (
        f"feature frame missing roots: {sorted(set(roots) - feature_roots)!r}"
    )
    feature_names = tuple(DEFAULT_OBSERVATION_FEATURES)
    term_series: dict[str, np.ndarray] = {}
    for root in roots:
        rows = (
            feature_frame.filter(pl.col(s.ROOT_SYMBOL_COLUMN) == root)
            .sort(s.TIMESTAMP_COLUMN)
            .tail(10)
        )
        assert rows.height == 10, f"{root}: missing feature rows"
        for name in feature_names:
            if name not in rows.columns:
                continue
            values = np.asarray(rows.get_column(name).to_numpy(), dtype=np.float64)
            assert np.all(np.isfinite(values)), f"{root}: feature {name!r} has non-finite values"
        if s.TERM_SLOPE in rows.columns:
            term_series[root] = np.asarray(
                rows.get_column(s.TERM_SLOPE).to_numpy(), dtype=np.float64
            )
        if s.CARRY_ANNUALIZED in rows.columns:
            carry = np.asarray(rows.get_column(s.CARRY_ANNUALIZED).to_numpy(), dtype=np.float64)
            assert np.all(np.isfinite(carry)), f"{root}: carry has non-finite values"
    assert term_series, "feature frame is missing the term_slope feature"
    flat_roots = sorted(
        root for root, values in term_series.items() if float(np.max(np.abs(values))) == 0.0
    )
    assert not flat_roots, f"term_slope is identically zero for roots {flat_roots!r} (task-30 A2)"


def test_pipeline_over_expanded_universe_is_finite_and_bounded(
    master: InstrumentsConfig,
    universal_config: StrategyConfig,
    pipeline: ComposablePipeline,
    final_state: MarketState,
) -> None:
    started = time.perf_counter()
    result = pipeline.run(final_state)
    elapsed = time.perf_counter() - started
    assert elapsed < 30.0, f"pipeline over {len(master.roots())} roots took {elapsed:.2f}s (>30s)"

    universe = tuple(result.universe)
    assert len(universe) >= MIN_UNIVERSE_ROOTS, (
        f"universal selector must keep all expanded roots, got {len(universe)}"
    )
    assert set(universe) <= set(master.roots())
    assert len(set(universe)) == len(universe)
    for stage, weights in (
        ("base", result.base_weights),
        ("timing", result.timing_weights),
        ("risk", result.risk_weights),
        ("final", result.final_weights),
    ):
        _assert_finite_weight_vector(weights, stage)
    assert result.final_weights.gross > 0.0
    assert result.final_weights.gross <= 1.0 + 1e-9
    assert set(result.final_weights.weights) == set(universe)
    assert result.targets is not None, "pipeline with translator must attach contract targets"
    _assert_finite_targets(result.targets, master)
    assert result.targets.total_initial_margin / EQUITY <= MARGIN_CEILING + 1e-9

    # Replays stay deterministic and fast.
    replay = pipeline.run(final_state)
    assert replay.final_weights == result.final_weights
    assert (time.perf_counter() - started) < 30.0


@pytest.mark.parametrize(
    ("selector_name", "params"),
    [
        ("liquidity", {"min_avg_dollar_volume": 0.0, "min_open_interest": 0.0}),
        ("regime", {}),
        ("composite", {"components": "liquidity,regime", "mode": "intersection"}),
    ],
)
def test_universe_selectors_return_valid_members(
    selector_name: str,
    params: dict[str, float | str],
    master: InstrumentsConfig,
    final_state: MarketState,
) -> None:
    selector = build_selector(selector_name, params, master)
    selected = selector.select(final_state)
    members = tuple(selected)
    assert len(set(members)) == len(members), f"{selector_name}: duplicate roots {members!r}"
    assert set(members) <= set(master.roots()), (
        f"{selector_name}: selected roots outside the instrument master: "
        f"{sorted(set(members) - set(master.roots()))!r}"
    )
    for root in members:
        assert master.has(root) and master.get(root).root == root
    assert selector.select(final_state) == list(members), (
        f"{selector_name}: non-deterministic selection"
    )
    if selector_name == "liquidity":
        assert set(members) == set(final_state.ids), (
            "zero-threshold liquidity selector must keep the whole expanded universe"
        )
    assert members, f"{selector_name}: expanded universe must select at least one root"


def test_equal_weight_allocator_builds_over_expanded_universe(
    master: InstrumentsConfig, final_state: MarketState
) -> None:
    allocator = build_allocator("equal_weight", {"gross_leverage": 1.0})
    weights = allocator.allocate(final_state.ids, final_state)
    _assert_finite_weight_vector(weights, "equal_weight")
    assert len(weights) == len(final_state.ids)
    assert weights.gross == pytest.approx(1.0)
    assert all(root in master.instruments for root in weights)


# ---------------------------------------------------------------------------
# 4. Backtest / DRL training-data smoke on the expanded universe
# ---------------------------------------------------------------------------


def test_backtest_smoke_over_expanded_universe(
    master: InstrumentsConfig,
    universal_config: StrategyConfig,
    pipeline: ComposablePipeline,
    translator: ContractTranslator,
    all_states: tuple[MarketState, ...],
) -> None:
    started = time.perf_counter()
    result = run_from_config(
        universal_config,
        states=all_states,
        pipeline=pipeline,
        translator=translator,
        instrument_master=master,
        initial_equity=EQUITY,
        mode=BacktestMode.EVENT,
    )
    elapsed = time.perf_counter() - started
    assert isinstance(result, BacktestResult)
    assert elapsed < 60.0, f"expanded-universe backtest took {elapsed:.2f}s (>60s)"
    assert math.isfinite(result.ending_equity) and result.ending_equity > 0.0
    assert math.isfinite(result.total_return)
    assert math.isfinite(result.price_pnl) and math.isfinite(result.roll_pnl)
    assert math.isfinite(result.commissions) and math.isfinite(result.slippage)
    assert math.isfinite(result.turnover)
    assert math.isfinite(result.margin_utilization_peak)
    assert result.margin_utilization_peak <= MARGIN_CEILING + 1e-6
    assert result.n_trades >= 0 and result.n_rebalances > 0
    assert result.equity_curve.height > 1
    equity = np.asarray(result.equity_curve.get_column("equity").to_numpy(), dtype=np.float64)
    assert np.all(np.isfinite(equity))
    assert equity[-1] == pytest.approx(result.ending_equity, rel=1e-9)
    for trade in result.trades[:5]:
        assert math.isfinite(trade.price)
        assert isinstance(trade.quantity, int)


def test_walk_forward_smoke_over_expanded_universe(
    master: InstrumentsConfig,
    universal_config: StrategyConfig,
    pipeline: ComposablePipeline,
    translator: ContractTranslator,
    all_states: tuple[MarketState, ...],
) -> None:
    started = time.perf_counter()
    report = run_walk_forward(
        universal_config,
        states=all_states,
        splits=2,
        purge_bars=1,
        embargo_bars=1,
        initial_equity=EQUITY,
        instrument_master=master,
        pipeline=pipeline,
        translator=translator,
    )
    elapsed = time.perf_counter() - started
    assert elapsed < 60.0, f"expanded-universe walk-forward took {elapsed:.2f}s (>60s)"
    assert report.walk_forward.fold_count == 2
    assert len(report.folds) == 2
    assert [fold.fold for fold in report.folds] == [0, 1]
    for fold in report.folds:
        assert fold.test_start < fold.test_end
        assert fold.n_bars > 0
        assert fold.n_rebalances > 0
        assert math.isfinite(fold.return_pct)
        assert math.isfinite(fold.ending_equity) and fold.ending_equity > 0.0
        _assert_finite_numbers(fold.metrics, path=f"fold[{fold.fold}].metrics")
    _assert_finite_numbers(dict(report.aggregate), path="aggregate")
    _assert_finite_numbers(dict(report.combined_metrics), path="combined_metrics")


def test_drl_training_data_covers_expanded_universe(
    master: InstrumentsConfig, universal_config: StrategyConfig
) -> None:
    started = time.perf_counter()
    data = build_training_data(universal_config, master, seed=7, periods=FRAME_PERIODS)
    elapsed = time.perf_counter() - started
    assert elapsed < 60.0, f"DRL training-data build took {elapsed:.2f}s (>60s)"
    assert len(data.instrument_ids) >= MIN_UNIVERSE_ROOTS, (
        "OfflineBarSource/train_drl must cover the expanded universe when the configured "
        f"selector keeps all roots; got {len(data.instrument_ids)}: {data.instrument_ids!r}"
    )
    assert set(data.instrument_ids) == set(master.roots()), (
        "DRL training data must match the configured all-root universe; missing: "
        f"{sorted(set(master.roots()) - set(data.instrument_ids))!r}"
    )
    assert len(set(data.instrument_ids)) == len(data.instrument_ids)
    assert data.feature_fallbacks == (), (
        f"canonical features must be complete for every root, missing: {data.feature_fallbacks!r}"
    )
    assert data.prices.ndim == 2 and data.prices.shape[1] == len(data.instrument_ids)
    assert data.features.ndim == 3 and data.features.shape[1] == len(data.instrument_ids)
    assert np.all(np.isfinite(data.prices)) and np.all(data.prices > 0.0)
    assert np.all(np.isfinite(data.features))
    assert data.usable_rows >= 2
    assert len(data.multipliers) == len(data.instrument_ids)
    assert all(multiplier > 0.0 for multiplier in data.multipliers)


# ---------------------------------------------------------------------------
# 5. API exposure (skipped while concurrent Studio endpoints are not registered)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def api_client(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TestClient]:
    """Offline API app (real container, deterministic fixture clock)."""
    tmp_path = tmp_path_factory.mktemp("phase10_1-universe-api")
    settings = Settings(
        environment="backtest",
        data_dir=tmp_path / "data",
        parquet_dir=tmp_path / "data" / "parquet",
        duckdb_path=tmp_path / "data" / "universe.duckdb",
        report_dir=tmp_path / "artifacts",
        instruments_path=REPO_ROOT / "configs" / "instruments.yaml",
        telemetry_interval_seconds=0.05,
    )
    app = create_app(settings, clock=lambda: datetime(2026, 11, 3, 12, 0, tzinfo=UTC))
    with TestClient(app) as client:
        yield client


def test_api_instruments_endpoint_lists_expanded_universe(
    api_client: TestClient, master: InstrumentsConfig
) -> None:
    response = api_client.get("/api/instruments")
    if response.status_code == 404:
        pytest.skip("GET /api/instruments is not registered yet (task-32 pending)")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["count"] >= MIN_UNIVERSE_ROOTS, (
        f"/api/instruments count {body['count']} < {MIN_UNIVERSE_ROOTS}"
    )
    sectors = body.get("sectors")
    assert isinstance(sectors, list) and len(sectors) == len(EXPECTED_API_SECTOR_IDS), (
        f"/api/instruments must expose {len(EXPECTED_API_SECTOR_IDS)} sectors, got "
        f"{[sector.get('id') for sector in sectors or []]!r}"
    )
    sector_ids = tuple(str(sector["id"]) for sector in sectors)
    assert sector_ids == EXPECTED_API_SECTOR_IDS, (
        f"/api/instruments sector ids/order must be {EXPECTED_API_SECTOR_IDS}, got {sector_ids}"
    )
    flat_roots: dict[str, dict[str, Any]] = {}
    for sector in sectors:
        assert int(sector["count"]) == len(sector.get("roots", []))
        for entry in sector.get("roots", []):
            root = str(entry["root"])
            assert root not in flat_roots, f"duplicate root {root!r} across sectors"
            flat_roots[root] = entry
            spec = master.get(root)
            assert str(sector["id"]) == CONFIG_SECTOR_TO_API_ID[spec.sector]
            assert entry["asset_class"] == spec.asset_class.value
    assert set(flat_roots) == set(master.roots())
    assert int(body["count"]) == len(flat_roots) == len(body["roots"])
    for root, entry in flat_roots.items():
        spec = master.get(root)
        assert entry["micro_root"] == spec.micro_root
        assert entry["standard_root"] == spec.standard_root
        assert float(entry["multiplier"]) == pytest.approx(spec.multiplier)
        assert float(entry["tick_size"]) == pytest.approx(spec.tick_size)
        if spec.micro_root is not None:
            assert flat_roots[spec.micro_root]["standard_root"] == root
    assert json.dumps(body)  # strictly JSON-serialisable payload


def test_api_strategy_catalog_exposes_universe_choices(
    api_client: TestClient, master: InstrumentsConfig
) -> None:
    response = api_client.get("/api/strategies/schema/catalog")
    assert response.status_code == 200, response.text
    body = response.json()
    assert json.dumps(body)  # JSON-serialisable (Studio contract)
    assert body.get("parameters"), "parameter catalog must not be empty"

    universe = body.get("universe")
    if not isinstance(universe, dict) or len(universe.get("roots") or []) < MIN_UNIVERSE_ROOTS:
        pytest.skip(
            "Studio catalog universe metadata is not passed through yet (task-31 API passthrough); "
            f"found universe={type(universe).__name__}"
        )
    sector_groups = universe.get("sectors") or []
    roots = _catalog_universe_roots(body, master)
    assert set(master.roots()) == roots, (
        "catalog universe choices must include every expanded root; missing: "
        f"{sorted(set(master.roots()) - roots)!r}, extra: {sorted(roots - set(master.roots()))!r}"
    )
    # Catalog universe exposes an "all" group followed by the 7 Studio sector lanes.
    expected_ids = ("all", *EXPECTED_API_SECTOR_IDS)
    assert (
        isinstance(sector_groups, list)
        and tuple(str(group["id"]) for group in sector_groups) == expected_ids
    ), (
        f"catalog universe must expose {expected_ids!r} sector groups, got "
        f"{[group.get('id') for group in sector_groups]!r}"
    )
    all_group = sector_groups[0]
    assert int(all_group["count"]) == len(master.roots())
    for group in sector_groups[1:]:
        assert int(group["count"]) == len(group.get("roots", []))
    stage_choices = body.get("stage_choices")
    assert isinstance(stage_choices, dict) and stage_choices, (
        "catalog stage_choices must expose dynamic selector/allocator dropdown options"
    )
    assert json.dumps(body)  # additive Studio metadata stays JSON-serialisable


def _catalog_universe_roots(body: Mapping[str, Any], master: InstrumentsConfig) -> set[str]:
    """Collect root-like identifiers from the catalog's universe metadata/choices."""
    known = frozenset(master.roots())
    candidates: set[str] = set()

    def _collect(value: object) -> None:
        if isinstance(value, str):
            if value.strip().upper() in known:
                candidates.add(value.strip().upper())
        elif isinstance(value, Mapping):
            for item in value.values():
                _collect(item)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for item in value:
                _collect(item)

    universe = body.get("universe")
    if universe is not None:
        _collect(universe)
    # Also scan universe-stage parameter choices (task-31 makes them dynamic).
    parameters = body.get("parameters", [])
    if isinstance(parameters, Mapping):
        parameters = list(parameters.values())
    for parameter in parameters:
        if isinstance(parameter, Mapping) and "universe" in str(parameter.get("stage", "")):
            _collect(parameter.get("choices"))
    return candidates
