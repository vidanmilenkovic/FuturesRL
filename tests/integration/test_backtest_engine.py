"""Integration tests for the futures backtest engine.

These exercise the full event-driven and vectorised pipelines with injected
deterministic strategy/translator fakes, the real :class:`ContractTranslator`
(micro-routing parity regression), roll execution, margin-call liquidation,
walk-forward leakage safety, the declarative runner and the offline CLI.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from configs.instruments import InstrumentsConfig
from configs.strategy import BacktestSpec, StageConfig, StrategyConfig
from src.backtest.engine import (
    EQUITY_CURVE_SCHEMA,
    AccountingIdentityError,
    BacktestEngine,
    BacktestMode,
    BacktestResult,
    EventBacktester,
    ForcedLiquidationEvent,
    OfflineMomentumPipeline,
    RebalanceSchedule,
    RollExecution,
    SimpleContractTranslator,
    TradeRecord,
    build_states_from_bars,
    default_state_builder,
    main,
    run_from_config,
)
from src.backtest.walk_forward import PurgedWalkForwardSplit
from src.core.types import (
    AssetClass,
    ContractTargetSet,
    InstrumentSnapshot,
    MarketState,
    OrderSide,
    PipelineResult,
    Position,
    WeightVector,
)
from src.data.schema import ensure_bars_schema
from src.risk.contract_translator import ContractTranslator

from tests.fixtures.market import (
    DEFAULT_EXPIRATION,
    FIXED_TIMESTAMP,
    make_bars_frame,
    make_market_state,
    make_snapshot,
)

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
INITIAL_EQUITY = 1_000_000.0


class ConstantPipeline:
    """Deterministic pipeline returning the same approved weights every cycle."""

    def __init__(self, weights: Mapping[str, float], name: str = "constant-pipeline") -> None:
        self._weights = WeightVector(dict(weights))
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def run(self, state: MarketState) -> PipelineResult:
        return PipelineResult(
            timestamp=state.timestamp,
            universe=state.ids,
            base_weights=self._weights,
            timing_weights=self._weights,
            risk_weights=self._weights,
            final_weights=self._weights,
        )


class RaisingTranslator:
    """Translator that always raises, exercising the defensive warning path."""

    def translate(
        self, weights: WeightVector, state: MarketState, *, equity: float | None = None
    ) -> ContractTargetSet:
        del weights, state, equity
        raise RuntimeError("translator unavailable")


def make_states_from_bars(
    bars: pl.DataFrame, instrument_master: InstrumentsConfig, *, equity: float = INITIAL_EQUITY
) -> list[MarketState]:
    return build_states_from_bars(bars, instrument_master, initial_equity=equity)


# ---------------------------------------------------------------------------
# Accounting identity / modes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", [BacktestMode.EVENT, BacktestMode.VECTORIZED])
def test_accounting_identity_both_modes(
    instrument_master: InstrumentsConfig, mode: BacktestMode
) -> None:
    bars = make_bars_frame(("ES", "NQ", "CL"), periods=80, seed=11)
    pipeline = ConstantPipeline({"ES": 0.2, "NQ": 0.1, "CL": -0.1})
    translator = ContractTranslator(instruments=instrument_master)
    engine = BacktestEngine(
        pipeline,
        translator,
        mode=mode,
        instrument_master=instrument_master,
        initial_equity=INITIAL_EQUITY,
        rebalance_schedule=RebalanceSchedule.every_n(5),
    )
    result = engine.run(bars=bars)
    assert isinstance(result, BacktestResult)
    assert result.mode is mode
    assert result.ending_equity > 0.0
    assert result.n_bars == 80
    assert result.n_trades > 0
    assert result.price_pnl != 0.0
    result.check_accounting_identity()
    assert result.accounting_residual == pytest.approx(0.0, abs=1e-6)
    # Equity curve schema is identical in both modes.
    assert tuple(result.equity_curve.columns) == tuple(EQUITY_CURVE_SCHEMA)
    assert result.margin_utilization_avg >= 0.0
    assert result.margin_utilization_peak >= result.margin_utilization_avg
    assert result.margin_used_max >= 0.0
    # Per-stage weights history is present for the waterfall inspector.
    stages = set(result.weight_history.get_column("stage").unique().to_list())
    assert {"base", "timing", "risk", "final"} <= stages
    assert result.n_rebalances > 1
    payload = result.to_dict()
    json.dumps(payload)
    assert payload["mode"] == mode.value
    assert payload["n_bars"] == 80
    assert len(payload["trades"]) == result.n_trades
    if mode is BacktestMode.EVENT:
        assert result.fills
        assert result.commissions > 0.0
        assert result.slippage > 0.0
        assert result.fees > 0.0
    else:
        assert not result.fills
        assert result.commissions > 0.0


def test_event_and_vectorized_agree_on_flat_strategy(
    instrument_master: InstrumentsConfig,
) -> None:
    bars = make_bars_frame(("ES",), periods=40, seed=3)
    pipeline = ConstantPipeline({})
    translator = ContractTranslator(instruments=instrument_master)
    results = {}
    for mode in (BacktestMode.EVENT, BacktestMode.VECTORIZED):
        engine = BacktestEngine(
            pipeline,
            translator,
            mode=mode,
            instrument_master=instrument_master,
            initial_equity=INITIAL_EQUITY,
        )
        results[mode] = engine.run(bars=bars)
    assert results[BacktestMode.EVENT].ending_equity == pytest.approx(INITIAL_EQUITY)
    assert results[BacktestMode.VECTORIZED].ending_equity == pytest.approx(INITIAL_EQUITY)
    assert results[BacktestMode.EVENT].n_trades == 0
    assert results[BacktestMode.VECTORIZED].n_trades == 0
    for result in results.values():
        result.check_accounting_identity()


# ---------------------------------------------------------------------------
# Micro-routing regression (deployment parity)
# ---------------------------------------------------------------------------


def test_micro_routed_targets_execute_in_event_engine(
    instrument_master: InstrumentsConfig,
) -> None:
    """A standard-only MarketState must still execute the translator's micro leg."""
    timestamps = [FIXED_TIMESTAMP + timedelta(days=offset) for offset in range(3)]
    states = [
        make_market_state(("ES",), timestamp=timestamp, equity=200_000.0)
        for timestamp in timestamps
    ]
    pipeline = ConstantPipeline({"ES": 0.30})
    translator = ContractTranslator(instruments=instrument_master)
    engine = EventBacktester(
        pipeline,
        translator,
        instrument_master=instrument_master,
        initial_equity=200_000.0,
    )
    result = engine.run(states=states)
    targets = translator.translate(WeightVector({"ES": 0.30}), states[0])
    assert set(targets.quantities) == {"MES"}
    first_bar_trades = [trade for trade in result.trades if trade.timestamp == states[0].timestamp]
    assert first_bar_trades
    assert {trade.instrument_id for trade in first_bar_trades} == {"MES"}
    assert not any("no price" in warning for warning in result.warnings)
    traded: dict[str, int] = {}
    for trade in first_bar_trades:
        traded[trade.instrument_id] = traded.get(trade.instrument_id, 0) + trade.signed_quantity
    assert traded == dict(targets.quantities)
    # Micro cost model used MES multiplier/tick (not ES): commission per contract
    # is the same tier but execution notional is 1/10th of the standard contract.
    fill = next(fill for fill in result.fills if fill.instrument_id == "MES")
    assert fill.cost.multiplier == pytest.approx(5.0)
    result.check_accounting_identity()


def test_standard_legs_unchanged_by_micro_synthesis(
    instrument_master: InstrumentsConfig,
) -> None:
    """A standard-only target book produces no synthetic quotes or warnings."""
    bars = make_bars_frame(("ES", "NQ"), periods=30, seed=5)
    pipeline = ConstantPipeline({"ES": 0.2, "NQ": 0.1})
    translator = SimpleContractTranslator(instrument_master)
    engine = EventBacktester(
        pipeline,
        translator,
        instrument_master=instrument_master,
        initial_equity=INITIAL_EQUITY,
        rebalance_schedule=RebalanceSchedule.every_n(10),
    )
    result = engine.run(bars=bars)
    assert result.warnings == ()
    assert {trade.instrument_id for trade in result.trades} <= {"ES", "NQ"}
    result.check_accounting_identity()


# ---------------------------------------------------------------------------
# Roll execution
# ---------------------------------------------------------------------------


def _roll_bars_and_states(
    *, periods: int = 60, roll_index: int = 30
) -> tuple[pl.DataFrame, list[MarketState]]:
    start = datetime(2024, 1, 2, 21, 0, tzinfo=UTC)
    timestamps = [start + timedelta(days=offset) for offset in range(periods)]
    front_expiration = date(2024, 3, 15)
    next_expiration = date(2024, 6, 21)
    rows: dict[str, list[object]] = {
        "timestamp_utc": [],
        "instrument_id": [],
        "root_symbol": [],
        "expiration_date": [],
        "open": [],
        "high": [],
        "low": [],
        "close": [],
        "volume": [],
        "open_interest": [],
        "settlement": [],
    }
    states: list[MarketState] = []
    price = 4800.0
    for index, timestamp in enumerate(timestamps):
        price = price * (1.0 + 0.001 * math.sin(index / 3.0))
        next_price = price + 5.0
        for tag, expiration, contract_price in (
            ("ES_FRONT", front_expiration, price),
            ("ES_NEXT", next_expiration, next_price),
        ):
            if tag == "ES_NEXT" and index < 20:
                continue
            rows["timestamp_utc"].append(timestamp)
            rows["instrument_id"].append(tag)
            rows["root_symbol"].append("ES")
            rows["expiration_date"].append(expiration)
            rows["open"].append(contract_price)
            rows["high"].append(contract_price * 1.001)
            rows["low"].append(contract_price * 0.999)
            rows["close"].append(contract_price)
            rows["volume"].append(250_000.0)
            rows["open_interest"].append(1_000_000.0)
            rows["settlement"].append(contract_price)
        active_expiration = front_expiration if index < roll_index else next_expiration
        active_price = price if index < roll_index else next_price
        snapshot = InstrumentSnapshot(
            instrument_id="ES",
            root_symbol="ES",
            price=active_price,
            asset_class=AssetClass.EQUITY_INDEX,
            expiration=active_expiration,
            bid=active_price - 0.125,
            ask=active_price + 0.125,
            volume=250_000.0,
            open_interest=1_000_000.0,
            settlement=active_price,
            features={"ret_21d": 0.01},
        )
        states.append(
            MarketState(
                timestamp=timestamp,
                instruments={"ES": snapshot},
                equity=INITIAL_EQUITY,
            )
        )
    return ensure_bars_schema(pl.DataFrame(rows)), states


def test_roll_execution_and_roll_yield_attribution(
    instrument_master: InstrumentsConfig,
) -> None:
    bars, states = _roll_bars_and_states()
    pipeline = ConstantPipeline({"ES": 0.5})
    translator = SimpleContractTranslator(instrument_master)
    engine = EventBacktester(
        pipeline,
        translator,
        instrument_master=instrument_master,
        initial_equity=INITIAL_EQUITY,
        rebalance_schedule=RebalanceSchedule.every_n(10),
    )
    result = engine.run(bars=bars, states=states)
    assert len(result.roll_events) == 1
    roll = result.roll_events[0]
    assert isinstance(roll, RollExecution)
    assert roll.from_expiration == date(2024, 3, 15)
    assert roll.to_expiration == date(2024, 6, 21)
    assert roll.quantity == 2
    assert roll.roll_gap == pytest.approx(5.0)
    # Long position rolling up the curve pays the gap: negative roll yield.
    assert roll.roll_pnl == pytest.approx(-result.roll_events[0].quantity * 5.0 * 50.0)
    assert roll.total_cost > 0.0
    assert result.roll_pnl == pytest.approx(roll.roll_pnl)
    assert result.trades
    result.check_accounting_identity()
    positions = result.equity_curve.select(["timestamp_utc", "n_positions", "margin_used"])
    assert positions["n_positions"].max() == 1


def test_bag_roll_costs_less_than_two_legged_roll(
    instrument_master: InstrumentsConfig,
) -> None:
    bars, states = _roll_bars_and_states()
    pipeline = ConstantPipeline({"ES": 0.5})
    translator = SimpleContractTranslator(instrument_master)
    costs: dict[bool, float] = {}
    for use_bag in (False, True):
        engine = EventBacktester(
            pipeline,
            translator,
            instrument_master=instrument_master,
            initial_equity=INITIAL_EQUITY,
            rebalance_schedule=RebalanceSchedule.every_n(10),
            use_bag_rolls=use_bag,
        )
        result = engine.run(bars=bars, states=states)
        assert result.roll_events
        costs[use_bag] = result.roll_events[0].total_cost
        assert result.roll_events[0].bag_used is use_bag
        result.check_accounting_identity()
    assert costs[True] < costs[False]


# ---------------------------------------------------------------------------
# Forced liquidation / account depletion
# ---------------------------------------------------------------------------


def _crash_bars_and_states(
    prices: Sequence[float], *, initial_equity: float
) -> tuple[pl.DataFrame, list[MarketState]]:
    start = datetime(2024, 1, 2, 21, 0, tzinfo=UTC)
    timestamps = [start + timedelta(days=offset) for offset in range(len(prices))]
    rows: dict[str, list[object]] = {
        "timestamp_utc": [],
        "instrument_id": [],
        "root_symbol": [],
        "expiration_date": [],
        "open": [],
        "high": [],
        "low": [],
        "close": [],
        "volume": [],
        "open_interest": [],
        "settlement": [],
    }
    states: list[MarketState] = []
    for timestamp, price in zip(timestamps, prices, strict=True):
        rows["timestamp_utc"].append(timestamp)
        rows["instrument_id"].append("ES_FRONT")
        rows["root_symbol"].append("ES")
        rows["expiration_date"].append(DEFAULT_EXPIRATION)
        rows["open"].append(price)
        rows["high"].append(price * 1.01)
        rows["low"].append(price * 0.99)
        rows["close"].append(price)
        rows["volume"].append(250_000.0)
        rows["open_interest"].append(1_000_000.0)
        rows["settlement"].append(price)
        states.append(
            MarketState.from_snapshots(
                timestamp,
                [
                    make_snapshot(
                        "ES",
                        price=price,
                        expiration=DEFAULT_EXPIRATION,
                        features={"ret_21d": 0.0},
                    )
                ],
                equity=initial_equity,
            )
        )
    return ensure_bars_schema(pl.DataFrame(rows)), states


def test_forced_liquidation_on_maintenance_margin_breach(
    instrument_master: InstrumentsConfig,
) -> None:
    bars, states = _crash_bars_and_states(
        [4800.0, 4790.0, 3000.0, 2990.0], initial_equity=100_000.0
    )
    pipeline = ConstantPipeline({"ES": 2.5})
    translator = SimpleContractTranslator(instrument_master)
    engine = EventBacktester(
        pipeline,
        translator,
        instrument_master=instrument_master,
        initial_equity=100_000.0,
        rebalance_schedule=RebalanceSchedule.every_n(100),
    )
    result = engine.run(bars=bars, states=states)
    assert len(result.forced_liquidations) == 1
    liquidation = result.forced_liquidations[0]
    assert isinstance(liquidation, ForcedLiquidationEvent)
    assert liquidation.positions_closed == ("ES",)
    assert liquidation.maintenance_margin == pytest.approx(13_600.0)
    assert liquidation.equity_before < liquidation.maintenance_margin
    assert liquidation.equity_after > 0.0
    assert liquidation.penalty_cost > 0.0
    assert "maintenance margin" in liquidation.reason
    assert any(trade.kind == "forced_liquidation" for trade in result.trades)
    # No positions survive the liquidation.
    assert result.equity_curve["n_positions"][-1] == 0
    result.check_accounting_identity()


def test_depleted_account_truncates_run(instrument_master: InstrumentsConfig) -> None:
    bars, states = _crash_bars_and_states([4800.0, 500.0], initial_equity=100_000.0)
    pipeline = ConstantPipeline({"ES": 2.5})
    translator = SimpleContractTranslator(instrument_master)
    engine = EventBacktester(
        pipeline,
        translator,
        instrument_master=instrument_master,
        initial_equity=100_000.0,
        rebalance_schedule=RebalanceSchedule.every_n(100),
    )
    result = engine.run(bars=bars, states=states)
    assert result.warnings
    assert any("depleted" in warning for warning in result.warnings)
    assert result.ending_equity == pytest.approx(0.0)
    assert result.equity_curve["equity"][-1] == pytest.approx(0.0)
    assert result.n_bars == 2
    result.check_accounting_identity()


# ---------------------------------------------------------------------------
# Walk-forward leakage safety through the engine
# ---------------------------------------------------------------------------


def test_walk_forward_window_isolated_from_future_data(
    instrument_master: InstrumentsConfig,
) -> None:
    start = datetime(2024, 1, 2, 21, 0, tzinfo=UTC)
    timestamps = [start + timedelta(days=offset) for offset in range(80)]
    rng = np.random.default_rng(17)
    prices = 4800.0 * np.cumprod(1.0 + rng.normal(0.0002, 0.008, size=len(timestamps)))
    base_states = [
        MarketState.from_snapshots(
            timestamp,
            [
                make_snapshot(
                    "ES",
                    price=float(price),
                    expiration=DEFAULT_EXPIRATION,
                    features={"ret_21d": 0.01},
                )
            ],
            equity=INITIAL_EQUITY,
        )
        for timestamp, price in zip(timestamps, prices, strict=True)
    ]
    splitter = PurgedWalkForwardSplit(train_days=30, test_days=15, purge_days=3, embargo_days=2)
    fold = splitter.split(len(base_states))[0]
    pipeline = ConstantPipeline({"ES": 0.3})
    translator = SimpleContractTranslator(instrument_master)
    window_start = base_states[fold.test_start].timestamp
    window_end = base_states[fold.test_end - 1].timestamp

    def run(states: Sequence[MarketState]) -> BacktestResult:
        engine = EventBacktester(
            pipeline,
            translator,
            instrument_master=instrument_master,
            initial_equity=INITIAL_EQUITY,
            rebalance_schedule=RebalanceSchedule.every_n(5),
        )
        return engine.run(states=list(states), start=window_start, end=window_end)

    baseline = run(base_states)
    # Mutate every observation after the first fold's test window: the earlier
    # window result must be bit-for-bit identical (no future leakage).
    mutated = list(base_states)
    for index in range(fold.test_end, len(mutated)):
        original = mutated[index]
        snapshot = make_snapshot(
            "ES", price=original.price("ES") * 3.0, expiration=DEFAULT_EXPIRATION
        )
        mutated[index] = MarketState(
            timestamp=original.timestamp,
            instruments={"ES": snapshot},
            equity=original.equity,
        )
    leaked = run(mutated)
    assert leaked.ending_equity == pytest.approx(baseline.ending_equity)
    assert leaked.price_pnl == pytest.approx(baseline.price_pnl)
    assert leaked.n_bars == baseline.n_bars
    assert fold.train_end <= fold.test_start
    assert max(fold.train_indices) < min(fold.test_indices)


# ---------------------------------------------------------------------------
# Declarative runner
# ---------------------------------------------------------------------------


def test_run_from_config_with_injected_components(
    instrument_master: InstrumentsConfig,
) -> None:
    bars = make_bars_frame(("ES", "NQ"), periods=40, seed=2)
    config = StrategyConfig(
        id="integration-runner",
        universe=StageConfig(name="liquidity"),
        allocator=StageConfig(name="equal_weight"),
        backtest=BacktestSpec(
            start=date(2024, 1, 2),
            end=date(2024, 2, 1),
            initial_equity=500_000.0,
            rebalance_frequency="W-FRI",
            commission_bps=1.0,
            slippage_ticks=0.5,
        ),
    )
    pipeline = ConstantPipeline({"ES": 0.2, "NQ": 0.1})
    translator = SimpleContractTranslator(instrument_master)
    result = run_from_config(
        config,
        bars=bars,
        pipeline=pipeline,
        translator=translator,
        instrument_master=instrument_master,
        mode=BacktestMode.VECTORIZED,
    )
    assert result.strategy_id == pipeline.name
    assert result.initial_equity == pytest.approx(500_000.0)
    assert result.n_bars < bars.height  # the config end date truncates the window
    result.check_accounting_identity()


def test_translator_failure_is_warned_and_flat(
    instrument_master: InstrumentsConfig,
) -> None:
    bars = make_bars_frame(("ES",), periods=20, seed=4)
    pipeline = ConstantPipeline({"ES": 0.2})
    engine = EventBacktester(
        pipeline,
        RaisingTranslator(),  # type: ignore[arg-type]
        instrument_master=instrument_master,
        initial_equity=INITIAL_EQUITY,
    )
    result = engine.run(bars=bars)
    assert any("translator failed" in warning for warning in result.warnings)
    assert result.n_trades == 0
    assert result.ending_equity == pytest.approx(INITIAL_EQUITY)
    result.check_accounting_identity()


# ---------------------------------------------------------------------------
# Offline fallbacks (used by the CLI when a layer is unavailable)
# ---------------------------------------------------------------------------


def test_offline_pipeline_and_translator(instrument_master: InstrumentsConfig) -> None:
    state = make_market_state(("ES", "NQ"), equity=INITIAL_EQUITY)
    state = MarketState(
        timestamp=state.timestamp,
        instruments=state.instruments,
        equity=INITIAL_EQUITY,
    )
    features = {
        "ES": {"ret_21d": 0.05},
        "NQ": {"ret_21d": -0.05},
    }
    state = make_market_state(("ES", "NQ"), features=features, equity=INITIAL_EQUITY)
    pipeline = OfflineMomentumPipeline(name="offline")
    result = pipeline.run(state)
    assert result.timestamp == state.timestamp
    assert result.final_weights.get("ES") > 0.0
    assert result.final_weights.get("NQ") < 0.0
    assert result.diagnostics["pipeline"] == "offline_fallback"
    flat_state = make_market_state(
        ("ES",), features={"ES": {"ret_21d": 0.0}}, equity=INITIAL_EQUITY
    )
    assert pipeline.run(flat_state).final_weights.is_flat()

    translator = SimpleContractTranslator(instrument_master)
    targets = translator.translate(
        WeightVector({"ES": 0.30, "NQ": -0.20}), state, equity=INITIAL_EQUITY
    )
    quantities = targets.quantities
    assert quantities.get("ES") == 1
    assert quantities.get("NQ", 0) < 0
    # Zero weights and unknown roots never produce targets.
    empty = translator.translate(
        WeightVector({"ES": 0.0, "ZZZ": 0.1}), state, equity=INITIAL_EQUITY
    )
    assert len(empty) == 0
    with pytest.raises(ValueError):
        translator.translate(
            WeightVector({"ES": 0.3}),
            MarketState.from_snapshots(state.timestamp, [make_snapshot("ES")]),
        )


def test_build_states_from_bars_has_no_lookahead(instrument_master: InstrumentsConfig) -> None:
    bars = make_bars_frame(("ES", "CL"), periods=60, seed=8)
    states = build_states_from_bars(bars, instrument_master, initial_equity=INITIAL_EQUITY)
    assert len(states) == 60
    first = states[0]
    snapshot = first.snapshot("ES")
    assert snapshot.bid is not None and snapshot.ask is not None
    assert snapshot.bid < snapshot.price < snapshot.ask
    assert "ret_21d" in snapshot.features
    assert math.isfinite(snapshot.feature("vol_20d"))
    assert first.equity == pytest.approx(INITIAL_EQUITY)
    # Mutating future bars cannot change the state built for an earlier bar.
    ordered = ensure_bars_schema(bars).sort(["timestamp_utc", "root_symbol"])
    later = states[29].timestamp
    mutated = ordered.with_columns(
        [
            pl.when(pl.col("timestamp_utc") > later)
            .then(pl.col(column) * 10.0)
            .otherwise(pl.col(column))
            .alias(column)
            for column in ("open", "high", "low", "close")
        ]
    )
    mutated_states = build_states_from_bars(
        mutated, instrument_master, initial_equity=INITIAL_EQUITY
    )
    for index in range(30):
        assert mutated_states[index].timestamp == states[index].timestamp
        for root in ("ES", "CL"):
            assert mutated_states[index].price(root) == pytest.approx(states[index].price(root))
            assert mutated_states[index].feature(root, "ret_21d") == pytest.approx(
                states[index].feature(root, "ret_21d")
            )
    # The injected-callable builder contract is honoured.
    state = default_state_builder(
        bars, states[10].timestamp, INITIAL_EQUITY, instrument_master=instrument_master
    )
    assert state.timestamp == states[10].timestamp
    with pytest.raises(KeyError):
        default_state_builder(bars, datetime(2030, 1, 1, tzinfo=UTC), INITIAL_EQUITY)


def test_offline_bars_are_deterministic(instrument_master: InstrumentsConfig) -> None:
    from src.backtest.engine import _offline_bars

    config = StrategyConfig(
        id="offline-bars",
        universe=StageConfig(name="liquidity"),
        allocator=StageConfig(name="equal_weight"),
        backtest=BacktestSpec(start=date(2024, 1, 2), end=date(2024, 12, 31)),
    )
    first = _offline_bars(config, instrument_master, periods=30)
    second = _offline_bars(config, instrument_master, periods=30)
    assert first.equals(second)
    assert first.height == 30 * min(5, len(instrument_master.roots()))
    assert set(first["root_symbol"].unique().to_list()) <= set(instrument_master.roots())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_offline_writes_tearsheet(tmp_path: Path) -> None:
    output = tmp_path / "tearsheet.json"
    html_output = tmp_path / "tearsheet.html"
    strategy_path = REPO_ROOT / "configs" / "strategies" / "tsmom_vol_target.yaml"
    exit_code = main(
        [
            "--strategy",
            str(strategy_path),
            "--offline",
            "--output",
            str(output),
            "--html",
            str(html_output),
            "--mode",
            "vectorized",
            "--initial-equity",
            "500000",
        ]
    )
    assert exit_code == 0
    assert output.exists() and html_output.exists()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["title"]
    assert payload["metrics"]["sharpe"] is not None
    assert payload["equity_curve"]["equity"]
    assert payload["accounting"]["residual"] == pytest.approx(0.0, abs=1e-6)
    html_text = html_output.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in html_text and "<svg" in html_text
    assert main(["--strategy", str(tmp_path / "missing.yaml"), "--offline"]) == 2


# ---------------------------------------------------------------------------
# Input validation / schedule / result records
# ---------------------------------------------------------------------------


def test_engine_input_validation(instrument_master: InstrumentsConfig) -> None:
    pipeline = ConstantPipeline({"ES": 0.2})
    translator = SimpleContractTranslator(instrument_master)
    with pytest.raises(ValueError):
        EventBacktester(pipeline, translator, initial_equity=0.0)
    with pytest.raises(ValueError):
        BacktestEngine(pipeline, translator, annualization_factor=0)
    with pytest.raises(TypeError):
        BacktestEngine(pipeline, translator, mode="event")  # type: ignore[arg-type]
    engine = BacktestEngine(pipeline, translator, instrument_master=instrument_master)
    with pytest.raises(ValueError):
        engine.run()
    empty = make_bars_frame(("ES",), periods=1).head(0)
    with pytest.raises(ValueError):
        engine.run(bars=empty)
    states = make_states_from_bars(make_bars_frame(("ES",), periods=5, seed=9), instrument_master)
    reversed_states = list(reversed(states))
    with pytest.raises(ValueError):
        engine.run(states=reversed_states)
    with pytest.raises(TypeError):
        engine.run(states=[object()])  # type: ignore[list-item]
    # A state-builder callable is honoured by the engine.
    build = EventBacktester(
        pipeline,
        translator,
        instrument_master=instrument_master,
        state_builder=lambda frame, moment, equity: default_state_builder(
            frame, moment, equity, instrument_master=instrument_master
        ),
    )
    built = build.run(bars=make_bars_frame(("ES",), periods=10, seed=2))
    assert built.n_bars == 10
    built.check_accounting_identity()


def test_rebalance_schedule() -> None:
    assert RebalanceSchedule.daily().should_rebalance(1, FIXED_TIMESTAMP)
    assert RebalanceSchedule.every_n(5).should_rebalance(10, FIXED_TIMESTAMP)
    assert not RebalanceSchedule.every_n(5).should_rebalance(7, FIXED_TIMESTAMP)
    assert RebalanceSchedule.every_n(5).should_rebalance(0, FIXED_TIMESTAMP)
    friday = datetime(2024, 1, 5, 21, 0, tzinfo=UTC)
    monday = datetime(2024, 1, 8, 21, 0, tzinfo=UTC)
    assert RebalanceSchedule.weekly().should_rebalance(1, friday)
    assert not RebalanceSchedule.weekly().should_rebalance(1, monday)
    monthly = RebalanceSchedule(monthly=True)
    assert monthly.should_rebalance(1, monday, datetime(2023, 12, 29, tzinfo=UTC))
    assert not monthly.should_rebalance(1, monday, friday)
    assert RebalanceSchedule.from_frequency("D").every_n_bars == 1
    assert RebalanceSchedule.from_frequency("daily").every_n_bars == 1
    assert RebalanceSchedule.from_frequency("W-FRI").weekday == 4
    assert RebalanceSchedule.from_frequency("W").weekday == 4
    assert RebalanceSchedule.from_frequency("M").monthly is True
    assert RebalanceSchedule.from_frequency("5").every_n_bars == 5
    assert RebalanceSchedule.from_frequency("5B").every_n_bars == 5
    with pytest.raises(ValueError):
        RebalanceSchedule()
    with pytest.raises(ValueError):
        RebalanceSchedule.from_frequency("")
    with pytest.raises(ValueError):
        RebalanceSchedule.from_frequency("fortnightly")
    with pytest.raises(ValueError):
        RebalanceSchedule.from_frequency("0")
    with pytest.raises(ValueError):
        RebalanceSchedule(weekday=7)
    with pytest.raises(TypeError):
        RebalanceSchedule(every_n_bars=1.5)  # type: ignore[arg-type]
    payload = RebalanceSchedule.daily().to_dict()
    assert payload["every_n_bars"] == 1


def test_result_records_and_identity_guard(
    instrument_master: InstrumentsConfig,
) -> None:
    bars = make_bars_frame(("ES",), periods=20, seed=6)
    result = BacktestEngine(
        ConstantPipeline({"ES": 0.2}),
        SimpleContractTranslator(instrument_master),
        mode=BacktestMode.VECTORIZED,
        instrument_master=instrument_master,
        initial_equity=INITIAL_EQUITY,
    ).run(bars=bars)
    assert result.n_rebalances >= 1
    tampered = replace(result, ending_equity=result.ending_equity + 10_000.0)
    with pytest.raises(AccountingIdentityError):
        tampered.check_accounting_identity()

    timestamp = FIXED_TIMESTAMP
    trade = TradeRecord(
        timestamp=timestamp,
        order_id="t1",
        instrument_id="ES",
        side=OrderSide.BUY,
        quantity=1,
        price=4800.0,
        reference_price=4800.0,
        kind="entry",
    )
    assert trade.signed_quantity == 1
    assert trade.to_dict()["kind"] == "entry"
    with pytest.raises(ValueError):
        TradeRecord(
            timestamp=timestamp,
            order_id="bad",
            instrument_id="ES",
            side=OrderSide.BUY,
            quantity=0,
            price=4800.0,
            reference_price=4800.0,
            kind="entry",
        )
    roll = RollExecution(
        timestamp=timestamp,
        instrument_id="ES",
        quantity=1,
        from_expiration=date(2024, 3, 15),
        to_expiration=date(2024, 6, 21),
        from_price=4800.0,
        to_price=4805.0,
        roll_pnl=-250.0,
        total_cost=30.0,
        bag_used=False,
    )
    assert roll.roll_gap == pytest.approx(5.0)
    assert roll.to_dict()["to_expiration"] == "2024-06-21"
    with pytest.raises(ValueError):
        RollExecution(
            timestamp=timestamp,
            instrument_id="ES",
            quantity=0,
            from_expiration=None,
            to_expiration=None,
            from_price=4800.0,
            to_price=4805.0,
            roll_pnl=-250.0,
            total_cost=30.0,
            bag_used=False,
        )
    liquidation = ForcedLiquidationEvent(
        timestamp=timestamp,
        equity_before=10_000.0,
        maintenance_margin=13_600.0,
        equity_after=9_900.0,
        positions_closed=("ES",),
        penalty_cost=100.0,
    )
    assert liquidation.to_dict()["positions_closed"] == ["ES"]
    with pytest.raises(ValueError):
        ForcedLiquidationEvent(
            timestamp=timestamp,
            equity_before=float("nan"),
            maintenance_margin=0.0,
            equity_after=0.0,
            positions_closed=(),
            penalty_cost=0.0,
        )


def test_vectorized_margin_fallback_without_instrument_master() -> None:
    states = make_market_state(("ES",), equity=INITIAL_EQUITY)
    pipeline = ConstantPipeline({"ES": 0.2})
    translator = _FallbackTranslator()
    engine = BacktestEngine(
        pipeline,
        translator,
        mode=BacktestMode.VECTORIZED,
        initial_equity=INITIAL_EQUITY,
        rebalance_schedule=RebalanceSchedule.daily(),
    )
    result = engine.run(
        states=[states, replace(states, timestamp=states.timestamp + timedelta(days=1))]
    )
    assert result.margin_used_max >= 0.0
    result.check_accounting_identity()


class _FallbackTranslator:
    """Minimal translator for the master-less margin fallback path."""

    def translate(
        self, weights: WeightVector, state: MarketState, *, equity: float | None = None
    ) -> ContractTargetSet:
        del weights, state, equity
        return ContractTargetSet.empty()


def test_position_type_is_available_for_engine_outputs() -> None:
    position = Position(
        instrument_id="ES",
        quantity=1,
        average_price=4800.0,
        market_price=4800.0,
        multiplier=50.0,
    )
    assert position.unrealized_pnl == pytest.approx(0.0)
