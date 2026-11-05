"""Protocol interfaces for the FuturesRL architecture.

These ``typing.Protocol`` definitions are the deployment-parity contract: a
strategy, backtester and live IBKR engine are interchangeable as long as they
honour the same protocols. Implementations live in the module packages
(``src.data``, ``src.strategies``, ``src.risk``, ``src.execution``, ...) and are
never imported here, which keeps the core free of runtime dependencies and
prevents circular imports.

All protocols are structural: tests can inject deterministic fakes without
inheritance, and production code is never coupled to a concrete class.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import datetime
from typing import Protocol, runtime_checkable

import polars as pl

from src.core.types import (
    AccountSnapshot,
    ContractTargetSet,
    InstrumentId,
    MarketState,
    OrderEvent,
    OrderReceipt,
    PipelineResult,
    Position,
    RiskAssessment,
    WeightVector,
)

# ---------------------------------------------------------------------------
# FinRL-X composable pipeline stages: w_t = R_t(T_t(A_t(S_t(X))))
# ---------------------------------------------------------------------------


@runtime_checkable
class UniverseSelector(Protocol):
    """``S_t``: filter the tradable futures roots for the current state."""

    @property
    def name(self) -> str:
        """Human-readable selector name used in telemetry / waterfall views."""
        ...

    def select(self, state: MarketState) -> Sequence[InstrumentId]:
        """Return the ordered tradable universe, given point-in-time state."""
        ...


@runtime_checkable
class PortfolioAllocator(Protocol):
    """``A_t``: map the candidate universe to raw signed notional weights."""

    @property
    def name(self) -> str:
        """Human-readable allocator name."""
        ...

    def allocate(self, candidates: Sequence[InstrumentId], state: MarketState) -> WeightVector:
        """Return raw base weights ``w_t^base`` for ``candidates``."""
        ...


@runtime_checkable
class TimingOverlay(Protocol):
    """``T_t``: scale base weights using momentum / term-structure signals."""

    @property
    def name(self) -> str:
        """Human-readable timing overlay name."""
        ...

    def adjust_timing(self, weights: WeightVector, state: MarketState) -> WeightVector:
        """Return timing-adjusted weights ``w_t^timing``."""
        ...


@runtime_checkable
class RiskOverlay(Protocol):
    """``R_t``: apply risk constraints producing the approved weights ``w_t``."""

    @property
    def name(self) -> str:
        """Human-readable risk overlay name."""
        ...

    def apply_risk(self, weights: WeightVector, state: MarketState) -> WeightVector:
        """Return approved weights ``w_t``."""
        ...


@runtime_checkable
class StrategyPipeline(Protocol):
    """The composed FinRL-X pipeline consumed by backtest and execution."""

    @property
    def name(self) -> str:
        """Pipeline name (matches the declarative YAML config id)."""
        ...

    def run(self, state: MarketState) -> PipelineResult:
        """Execute one full rebalance cycle and return the staged trace."""
        ...


@runtime_checkable
class ContractTranslatorProtocol(Protocol):
    """Deterministic ``w_t -> N_i,t`` translator contract."""

    def translate(
        self,
        weights: WeightVector,
        state: MarketState,
        *,
        equity: float | None = None,
        account_currency: str | None = None,
        fx_rates: Mapping[str, float] | None = None,
    ) -> ContractTargetSet:
        """Convert approved weights into discrete contract targets.

        ``account_currency`` / ``fx_rates`` let a USD-denominated contract
        translator consume a non-USD account's ``equity`` (Phase 7).
        """
        ...


# ---------------------------------------------------------------------------
# Risk assessment providers (Tier A: LLM Risk-First engine)
# ---------------------------------------------------------------------------


@runtime_checkable
class RiskAssessmentProvider(Protocol):
    """Synchronous source of the latest LLM risk assessments."""

    def assess(
        self, state: MarketState, candidates: Sequence[InstrumentId]
    ) -> Sequence[RiskAssessment]:
        """Return one assessment per candidate root."""
        ...


@runtime_checkable
class AsyncRiskAssessmentProvider(Protocol):
    """Asynchronous LLM client adapter used by the live risk loop."""

    async def assess_async(
        self, state: MarketState, candidates: Sequence[InstrumentId]
    ) -> Sequence[RiskAssessment]:
        """Return one assessment per candidate root."""
        ...


@runtime_checkable
class MacroCalendar(Protocol):
    """Scheduled Tier-1 macro event calendar used for lockout windows."""

    def upcoming_events(
        self, start: datetime, end: datetime
    ) -> Sequence[tuple[str, datetime, int]]:
        """Return ``(event_name, event_time, tier)`` triples in the window."""
        ...


# ---------------------------------------------------------------------------
# Data layer contracts
# ---------------------------------------------------------------------------


@runtime_checkable
class HistoricalDataSource(Protocol):
    """Historical market data source normalised to the unified Polars schema."""

    def load(
        self,
        roots: Sequence[InstrumentId],
        start: datetime,
        end: datetime,
        *,
        schema: str = "ohlcv-1d",
    ) -> pl.DataFrame:
        """Load normalised OHLCV bars for ``roots`` in ``[start, end)``."""
        ...

    def load_definitions(self, roots: Sequence[InstrumentId], *, as_of: datetime) -> pl.DataFrame:
        """Load point-in-time contract definitions (multipliers, expiries)."""
        ...

    def available_roots(self) -> Sequence[InstrumentId]:
        """Roots present in the cache."""
        ...


@runtime_checkable
class MarketDataStreamer(Protocol):
    """Live/paper market data streamer emitting immutable market states."""

    @property
    def is_connected(self) -> bool:
        """Whether the stream is currently connected."""
        ...

    async def connect(self) -> None:
        """Establish the streaming connection."""
        ...

    async def disconnect(self) -> None:
        """Tear down the streaming connection."""
        ...

    def stream(self) -> AsyncIterator[MarketState]:
        """Yield market states as they arrive."""
        ...


@runtime_checkable
class FeatureProcessorProtocol(Protocol):
    """Feature engineering contract: raw bars in, feature frame out."""

    def compute(self, bars: pl.DataFrame) -> pl.DataFrame:
        """Return the feature frame with the unified schema preserved."""
        ...


# ---------------------------------------------------------------------------
# Execution contracts
# ---------------------------------------------------------------------------


@runtime_checkable
class BrokerAdapter(Protocol):
    """Unified broker protocol implemented by simulated, paper and live brokers."""

    @property
    def name(self) -> str:
        """Broker identifier (``simulated``, ``ibkr-paper``, ``ibkr-live``)."""
        ...

    @property
    def is_connected(self) -> bool:
        """Whether the broker session is connected."""
        ...

    async def connect(self) -> None:
        """Connect / authenticate, recovering the broker session."""
        ...

    async def disconnect(self) -> None:
        """Disconnect, cancelling outstanding subscriptions."""
        ...

    async def get_account(self) -> AccountSnapshot:
        """Snapshot of account valuation and margin consumption."""
        ...

    async def get_positions(self) -> Mapping[InstrumentId, Position]:
        """Current broker positions keyed by root instrument id."""
        ...

    async def submit_order(self, order: OrderEvent) -> OrderReceipt:
        """Submit a single order through the pre-trade risk gate."""
        ...

    async def submit_orders(self, orders: Sequence[OrderEvent]) -> list[OrderReceipt]:
        """Submit a batch of orders atomically ordered (rolls are two-legged)."""
        ...

    async def cancel_order(self, order_id: str) -> OrderReceipt:
        """Cancel one outstanding order."""
        ...

    async def cancel_all(self) -> list[OrderReceipt]:
        """Cancel every outstanding order."""
        ...

    async def flatten_all(self) -> list[OrderReceipt]:
        """Emergency kill-switch: flatten every position and cancel all orders."""
        ...


@runtime_checkable
class PreTradeRiskGate(Protocol):
    """Independent pre-trade hard gate (fat finger, margin, rate limit)."""

    async def check(self, order: OrderEvent) -> OrderReceipt | None:
        """Return ``None`` when the order passes, otherwise a rejection receipt."""
        ...


@runtime_checkable
class PositionReconciler(Protocol):
    """Periodic internal-versus-broker state reconciliation contract."""

    async def reconcile(self) -> bool:
        """Reconcile state; return ``True`` when internal and broker agree."""
        ...


# ---------------------------------------------------------------------------
# Backtesting contracts
# ---------------------------------------------------------------------------


@runtime_checkable
class Backtestable(Protocol):
    """Anything that can be driven through an event loop by the backtester."""

    def run(self, state: MarketState) -> PipelineResult:
        """Produce a target portfolio for ``state``."""
        ...
