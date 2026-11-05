"""Intraday vs daily mark-to-market settlement behaviour (Spec 1, task-70).

Daily/coarser bars keep the historical per-bar settlement (1 bar == 1 session);
intraday bars settle at most once per UTC calendar day, even when several bars
share the same UTC day. Margin-call checks remain per bar.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta

from configs.instruments import InstrumentsConfig
from src.backtest.engine import (
    EventBacktester,
    SimpleContractTranslator,
    build_states_from_bars,
    run_from_config,
)
from src.backtest.margin_simulator import MarginSimulator
from src.core.types import MarketState, PipelineResult, WeightVector
from src.data.timeframes import bars_per_year

from tests.fixtures.market import make_market_state

BASE = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)


class _ConstantPipeline:
    def __init__(self, weights: Mapping[str, float]) -> None:
        self._weights = WeightVector(dict(weights))

    @property
    def name(self) -> str:
        return "constant"

    def run(self, state: MarketState) -> PipelineResult:
        return PipelineResult(
            timestamp=state.timestamp,
            universe=state.ids,
            base_weights=self._weights,
            timing_weights=self._weights,
            risk_weights=self._weights,
            final_weights=self._weights,
        )


class _CountingMarginSimulator(MarginSimulator):
    def __init__(self, instruments: InstrumentsConfig) -> None:
        super().__init__(instruments)
        self.settle_timestamps: list[datetime] = []

    def settle(
        self,
        timestamp: datetime,
        *,
        cash: float,
        positions: Mapping[str, object],
        settlement_prices: Mapping[str, float] | None = None,
    ) -> object:
        self.settle_timestamps.append(timestamp)
        return super().settle(
            timestamp,
            cash=cash,
            positions=positions,  # type: ignore[arg-type]
            settlement_prices=settlement_prices,
        )


def _run(
    instrument_master: InstrumentsConfig,
    *,
    timestamps: list[datetime],
) -> tuple[object, list[datetime]]:
    states = [
        make_market_state(("ES",), timestamp=timestamp, equity=1_000_000.0)
        for timestamp in timestamps
    ]
    spy = _CountingMarginSimulator(instrument_master)
    engine = EventBacktester(
        _ConstantPipeline({"ES": 0.30}),
        SimpleContractTranslator(instrument_master),
        instrument_master=instrument_master,
        initial_equity=1_000_000.0,
        margin_simulator=spy,
    )
    result = engine.run(states=states)
    return result, spy.settle_timestamps


def test_intraday_settlement_once_per_utc_day(
    instrument_master: InstrumentsConfig,
) -> None:
    timestamps = [
        BASE,
        BASE + timedelta(hours=1),
        BASE + timedelta(hours=2),
        # Next UTC calendar day: settlement boundary.
        BASE + timedelta(days=1),
        BASE + timedelta(days=1, hours=1),
    ]
    result, settlement_timestamps = _run(instrument_master, timestamps=timestamps)
    # Same-day bars after the first position was opened must not settle again.
    assert len(settlement_timestamps) == 1
    assert settlement_timestamps[0].date() == (BASE + timedelta(days=1)).date()
    assert len({stamp.date() for stamp in settlement_timestamps}) == len(settlement_timestamps)
    result.check_accounting_identity()  # type: ignore[attr-defined]


def test_daily_bars_keep_legacy_per_bar_settlement(
    instrument_master: InstrumentsConfig,
) -> None:
    timestamps = [BASE, BASE + timedelta(days=1), BASE + timedelta(days=2)]
    _result, settlement_timestamps = _run(instrument_master, timestamps=timestamps)
    # Daily bars settle on every bar (legacy behaviour): pre-roll + end-of-bar.
    assert len(settlement_timestamps) > 1
    by_day = Counter(stamp.date() for stamp in settlement_timestamps)
    assert any(count >= 2 for count in by_day.values())
    # The intraday run settles strictly less often than the equivalent daily run.
    intraday_result, intraday_stamps = _run(
        instrument_master,
        timestamps=[BASE, BASE + timedelta(hours=1), BASE + timedelta(hours=2)],
    )
    intraday_result.check_accounting_identity()  # type: ignore[attr-defined]
    assert len(intraday_stamps) == 0
    assert len(settlement_timestamps) > len(intraday_stamps)


def test_states_are_intraday_detection() -> None:
    from src.backtest.engine import _states_are_intraday

    daily = [
        make_market_state(("ES",), timestamp=BASE + timedelta(days=offset)) for offset in range(3)
    ]
    intraday = [
        make_market_state(("ES",), timestamp=BASE + timedelta(hours=offset)) for offset in range(3)
    ]
    assert _states_are_intraday(daily) is False
    assert _states_are_intraday(intraday) is True
    assert _states_are_intraday(intraday[:1]) is False


def test_build_states_from_bars_propagates_timeframe(
    instrument_master: InstrumentsConfig,
) -> None:
    from tests.fixtures.market import make_bars_frame

    bars = make_bars_frame(("ES",), periods=4, seed=3)
    states = build_states_from_bars(bars, instrument_master, timeframe="5m")
    assert states
    assert all(state.timeframe == "5m" for state in states)


def test_run_from_config_propagates_backtest_timeframe(
    instrument_master: InstrumentsConfig,
) -> None:
    from configs.strategy import load_strategy_config

    from tests.fixtures.market import make_market_state

    base = load_strategy_config(
        __import__("pathlib").Path("configs/strategies/tsmom_vol_target.yaml")
    )
    assert base.backtest is not None
    config = base.model_copy(
        update={"backtest": base.backtest.model_copy(update={"timeframe": "1h"})}
    )
    timestamps = [BASE, BASE + timedelta(hours=1), BASE + timedelta(hours=2)]
    states = [
        make_market_state(("ES",), timestamp=timestamp, equity=1_000_000.0)
        for timestamp in timestamps
    ]
    result = run_from_config(
        config,
        states=states,
        pipeline=_ConstantPipeline({"ES": 0.20}),
        translator=SimpleContractTranslator(instrument_master),
        instrument_master=instrument_master,
    )
    assert result.metadata["timeframe"] == "1h"
    assert result.metadata["annualization_factor"] == bars_per_year("1h")
