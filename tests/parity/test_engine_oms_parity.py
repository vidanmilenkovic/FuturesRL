"""Deployment-parity tests: backtest engine vs live IBKR OMS.

The spec's core deployment-consistency claim is that, for an identical
``MarketState`` and identical approved weights ``w_t``, the *backtesting engine*
and the *live IBKR execution path* must act on exactly the same contract
targets and generate exactly the same order deltas. These tests prove that
claim end-to-end:

* both paths use the same deterministic :class:`ContractTranslator`; a second,
  independently constructed translator is byte-for-byte equal in output;
* the backtester's event-driven engine executes trades that are identical to
  the orders the live OMS would generate from the same target set
  (``build_orders_from_targets`` is the live delta logic);
* applying either set of deltas to the same starting book produces the same
  positions, including shorts and standard -> micro routing;
* the second rebalance (after costs and a new target) remains identical.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from configs.instruments import InstrumentsConfig
from src.backtest.engine import BacktestResult, EventBacktester
from src.core.types import (
    ContractTargetSet,
    InstrumentId,
    MarketState,
    PipelineResult,
    WeightVector,
)
from src.execution.oms import build_orders_from_targets
from src.risk.contract_translator import ContractTranslator
from tests.fixtures.market import make_market_state

pytestmark = pytest.mark.parity

INITIAL_EQUITY = 1_000_000.0
TIMESTAMP = datetime(2024, 1, 2, 21, 0, tzinfo=UTC)


class ScriptedPipeline:
    """Deterministic ``StrategyPipeline`` returning scripted weight vectors.

    Used instead of a real strategy so the parity assertion isolates the
    contract pipeline: whatever the strategy approved is what both engines must
    trade.
    """

    def __init__(self, weight_sequence: Sequence[Mapping[InstrumentId, float]]) -> None:
        if not weight_sequence:
            raise ValueError("weight_sequence must not be empty")
        self._sequence = [WeightVector.from_mapping(dict(weights)) for weights in weight_sequence]
        self._calls = 0

    @property
    def name(self) -> str:
        return "scripted-parity"

    def run(self, state: MarketState) -> PipelineResult:
        index = min(self._calls, len(self._sequence) - 1)
        self._calls += 1
        weights = self._sequence[index]
        return PipelineResult(
            timestamp=state.timestamp,
            universe=tuple(weights),
            base_weights=weights,
            timing_weights=weights,
            risk_weights=weights,
            final_weights=weights,
        )


def _states(
    *, equity: float = INITIAL_EQUITY, roots: tuple[str, ...] = ("ES", "NQ", "CL")
) -> list[MarketState]:
    """Two daily states with identical prices (no mark-to-market drift)."""
    return [
        make_market_state(roots, timestamp=TIMESTAMP + timedelta(days=offset), equity=equity)
        for offset in range(2)
    ]


def _run_engine(
    weights: Sequence[Mapping[str, float]],
    states: Sequence[MarketState],
    *,
    equity: float = INITIAL_EQUITY,
    instrument_master: InstrumentsConfig,
) -> BacktestResult:
    pipeline = ScriptedPipeline(weights)
    translator = ContractTranslator(instruments=instrument_master)
    engine = EventBacktester(
        pipeline,
        translator,
        instrument_master=instrument_master,
        initial_equity=equity,
    )
    return engine.run(states=list(states))


def _oms_deltas(
    targets: ContractTargetSet,
    positions: Mapping[InstrumentId, int],
    timestamp: datetime,
) -> set[tuple[str, str, int]]:
    orders = build_orders_from_targets(targets, positions, timestamp)
    return {(order.instrument_id, order.side.value, order.quantity) for order in orders}


def test_independent_translators_are_identical(
    instrument_master: InstrumentsConfig,
) -> None:
    """Two independently constructed translators must agree exactly."""
    states = _states()
    weights = WeightVector.from_mapping({"ES": 0.30, "NQ": 0.20, "CL": -0.15})
    first = ContractTranslator(instruments=instrument_master).translate(
        weights, replace(states[0], equity=INITIAL_EQUITY)
    )
    second = ContractTranslator(instruments=instrument_master).translate(
        weights, replace(states[0], equity=INITIAL_EQUITY)
    )
    assert [(t.instrument_id, t.quantity, t.price, t.multiplier) for t in first] == [
        (t.instrument_id, t.quantity, t.price, t.multiplier) for t in second
    ]
    assert first.quantities == second.quantities


def test_first_rebalance_engine_equals_live_oms(
    instrument_master: InstrumentsConfig,
) -> None:
    """The engine's first rebalance trades must equal the orders OMS would send."""
    states = _states()
    weights = {"ES": 0.30, "NQ": 0.20, "CL": -0.15}
    result = _run_engine([weights], states, instrument_master=instrument_master)

    targets = ContractTranslator(instruments=instrument_master).translate(
        WeightVector.from_mapping(weights), replace(states[0], equity=INITIAL_EQUITY)
    )
    engine_deltas = {
        (trade.instrument_id, trade.side.value, trade.quantity)
        for trade in result.trades
        if trade.timestamp == states[0].timestamp
    }
    live_deltas = _oms_deltas(targets, {}, states[0].timestamp)
    assert engine_deltas == live_deltas
    assert engine_deltas, "engine must trade on the first bar"
    # Applying either path's deltas to a flat book reproduces the same targets.
    backtest_positions: dict[InstrumentId, int] = {}
    for trade in result.trades:
        if trade.timestamp == states[0].timestamp:
            backtest_positions[trade.instrument_id] = (
                backtest_positions.get(trade.instrument_id, 0) + trade.signed_quantity
            )
    assert backtest_positions == dict(targets.quantities)


def test_second_rebalance_engine_equals_live_oms(
    instrument_master: InstrumentsConfig,
) -> None:
    """Repeated rebalances stay identical after costs and a new target."""
    states = _states()
    first_weights = {"ES": 0.30, "CL": -0.15}
    second_weights = {"ES": 0.10, "NQ": 0.25, "CL": 0.05}
    result = _run_engine(
        [first_weights, second_weights], states, instrument_master=instrument_master
    )

    translator = ContractTranslator(instruments=instrument_master)
    first_targets = translator.translate(
        WeightVector.from_mapping(first_weights), replace(states[0], equity=INITIAL_EQUITY)
    )
    positions = dict(first_targets.quantities)
    # Equity after the first rebalance row (identical prices => no MTM drift).
    equity_after_first = float(
        result.equity_curve.filter(result.equity_curve["timestamp_utc"] == states[0].timestamp)[
            "equity"
        ].item()
    )
    second_targets = translator.translate(
        WeightVector.from_mapping(second_weights), replace(states[1], equity=equity_after_first)
    )
    engine_deltas = {
        (trade.instrument_id, trade.side.value, trade.quantity)
        for trade in result.trades
        if trade.timestamp == states[1].timestamp
    }
    live_deltas = _oms_deltas(second_targets, positions, states[1].timestamp)
    assert engine_deltas == live_deltas
    assert engine_deltas, "second bar must produce a material rebalance"


def test_micro_routing_parity_small_account(
    instrument_master: InstrumentsConfig,
) -> None:
    """With 200k equity, both engines route ES exposure to MES identically."""
    states = _states(equity=200_000.0, roots=("ES",))
    weights = {"ES": 0.30}
    result = _run_engine([weights], states, equity=200_000.0, instrument_master=instrument_master)
    translator = ContractTranslator(instruments=instrument_master)
    targets = translator.translate(
        WeightVector.from_mapping(weights), replace(states[0], equity=200_000.0)
    )
    assert set(targets.quantities) == {"MES"}
    engine_instruments = {
        trade.instrument_id for trade in result.trades if trade.timestamp == states[0].timestamp
    }
    assert engine_instruments == {"MES"}
    engine_deltas = {
        (trade.instrument_id, trade.side.value, trade.quantity)
        for trade in result.trades
        if trade.timestamp == states[0].timestamp
    }
    assert engine_deltas == _oms_deltas(targets, {}, states[0].timestamp)


def test_short_positions_parity(instrument_master: InstrumentsConfig) -> None:
    """Shorts are translated and traded identically on both paths."""
    states = _states(roots=("CL", "GC", "ZN"))
    weights = {"CL": -0.20, "GC": 0.10, "ZN": -0.05}
    result = _run_engine([weights], states, instrument_master=instrument_master)
    translator = ContractTranslator(instruments=instrument_master)
    targets = translator.translate(
        WeightVector.from_mapping(weights), replace(states[0], equity=INITIAL_EQUITY)
    )
    assert any(target.quantity < 0 for target in targets), "fixture must exercise shorts"
    engine_positions: dict[InstrumentId, int] = {}
    for trade in result.trades:
        if trade.timestamp == states[0].timestamp:
            engine_positions[trade.instrument_id] = (
                engine_positions.get(trade.instrument_id, 0) + trade.signed_quantity
            )
    assert engine_positions == dict(targets.quantities)
    live_deltas = _oms_deltas(targets, {}, states[0].timestamp)
    engine_deltas = {
        (trade.instrument_id, trade.side.value, trade.quantity)
        for trade in result.trades
        if trade.timestamp == states[0].timestamp
    }
    assert engine_deltas == live_deltas
