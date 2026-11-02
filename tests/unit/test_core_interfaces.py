"""Unit tests for :mod:`src.core.interfaces` structural protocol conformance."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import UTC, datetime

import polars as pl
from src.core.interfaces import (
    AsyncRiskAssessmentProvider,
    Backtestable,
    BrokerAdapter,
    ContractTranslatorProtocol,
    HistoricalDataSource,
    MarketDataStreamer,
    PortfolioAllocator,
    RiskAssessmentProvider,
    RiskOverlay,
    StrategyPipeline,
    TimingOverlay,
    UniverseSelector,
)
from src.core.types import (
    AccountSnapshot,
    ContractTargetSet,
    InstrumentId,
    MarketState,
    OrderEvent,
    OrderReceipt,
    OrderStatus,
    PipelineResult,
    Position,
    RiskAssessment,
    WeightVector,
)

from tests.fixtures.market import FIXED_TIMESTAMP, make_market_state


class _Selector:
    @property
    def name(self) -> str:
        return "fake-selector"

    def select(self, state: MarketState) -> Sequence[InstrumentId]:
        return state.ids


class _Allocator:
    @property
    def name(self) -> str:
        return "fake-allocator"

    def allocate(self, candidates: Sequence[InstrumentId], state: MarketState) -> WeightVector:
        return WeightVector.equal_weight(candidates, gross=1.0)


class _Timing:
    @property
    def name(self) -> str:
        return "fake-timing"

    def adjust_timing(self, weights: WeightVector, state: MarketState) -> WeightVector:
        return weights.scaled(0.5)


class _Risk:
    @property
    def name(self) -> str:
        return "fake-risk"

    def apply_risk(self, weights: WeightVector, state: MarketState) -> WeightVector:
        return weights


class _Translator:
    def translate(
        self, weights: WeightVector, state: MarketState, *, equity: float | None = None
    ) -> ContractTargetSet:
        return ContractTargetSet.empty()


class _Pipeline:
    @property
    def name(self) -> str:
        return "fake-pipeline"

    def run(self, state: MarketState) -> PipelineResult:
        return PipelineResult(
            timestamp=state.timestamp,
            universe=state.ids,
            base_weights=WeightVector.empty(),
            timing_weights=WeightVector.empty(),
            risk_weights=WeightVector.empty(),
            final_weights=WeightVector.empty(),
        )


class _RiskProvider:
    def assess(
        self, state: MarketState, candidates: Sequence[InstrumentId]
    ) -> Sequence[RiskAssessment]:
        return []


class _AsyncRiskProvider:
    async def assess_async(
        self, state: MarketState, candidates: Sequence[InstrumentId]
    ) -> Sequence[RiskAssessment]:
        return []


class _DataSource:
    def load(
        self,
        roots: Sequence[InstrumentId],
        start: datetime,
        end: datetime,
        *,
        schema: str = "ohlcv-1d",
    ) -> pl.DataFrame:
        return pl.DataFrame()

    def load_definitions(self, roots: Sequence[InstrumentId], *, as_of: datetime) -> pl.DataFrame:
        return pl.DataFrame()

    def available_roots(self) -> Sequence[InstrumentId]:
        return ()


class _Streamer:
    @property
    def is_connected(self) -> bool:
        return False

    async def connect(self) -> None:
        return None

    async def disconnect(self) -> None:
        return None

    def stream(self) -> AsyncIterator[MarketState]:
        async def _empty() -> AsyncIterator[MarketState]:
            return
            yield  # pragma: no cover - makes this an async generator

        return _empty()


class _Broker:
    @property
    def name(self) -> str:
        return "fake-broker"

    @property
    def is_connected(self) -> bool:
        return False

    async def connect(self) -> None:
        return None

    async def disconnect(self) -> None:
        return None

    async def get_account(self) -> AccountSnapshot:
        return AccountSnapshot(timestamp=FIXED_TIMESTAMP, net_liquidation=1_000_000.0)

    async def get_positions(self) -> Mapping[InstrumentId, Position]:
        return {}

    async def submit_order(self, order: OrderEvent) -> OrderReceipt:
        return OrderReceipt(order_id=order.order_id, status=OrderStatus.FILLED)

    async def submit_orders(self, orders: Sequence[OrderEvent]) -> list[OrderReceipt]:
        return []

    async def cancel_order(self, order_id: str) -> OrderReceipt:
        return OrderReceipt(order_id=order_id, status=OrderStatus.CANCELLED)

    async def cancel_all(self) -> list[OrderReceipt]:
        return []

    async def flatten_all(self) -> list[OrderReceipt]:
        return []


def test_protocols_are_runtime_checkable_and_structural() -> None:
    assert isinstance(_Selector(), UniverseSelector)
    assert isinstance(_Allocator(), PortfolioAllocator)
    assert isinstance(_Timing(), TimingOverlay)
    assert isinstance(_Risk(), RiskOverlay)
    assert isinstance(_Translator(), ContractTranslatorProtocol)
    assert isinstance(_Pipeline(), StrategyPipeline)
    assert isinstance(_Pipeline(), Backtestable)
    assert isinstance(_RiskProvider(), RiskAssessmentProvider)
    assert isinstance(_AsyncRiskProvider(), AsyncRiskAssessmentProvider)
    assert isinstance(_DataSource(), HistoricalDataSource)
    assert isinstance(_Streamer(), MarketDataStreamer)
    assert isinstance(_Broker(), BrokerAdapter)


def test_non_conforming_objects_are_rejected() -> None:
    class Incomplete:
        pass

    assert not isinstance(Incomplete(), UniverseSelector)
    assert not isinstance(Incomplete(), StrategyPipeline)
    assert not isinstance(Incomplete(), BrokerAdapter)


def test_pipeline_fake_runs_end_to_end() -> None:
    state = make_market_state()
    allocator = _Allocator()
    timing = _Timing()
    risk = _Risk()
    base = allocator.allocate(state.ids[:3], state)
    timing_weights = timing.adjust_timing(base, state)
    final = risk.apply_risk(timing_weights, state)
    result = _Pipeline().run(state)
    assert final.gross == base.gross * 0.5
    assert result.universe == state.ids
    assert result.timestamp == datetime(2024, 1, 2, 21, 0, tzinfo=UTC)
