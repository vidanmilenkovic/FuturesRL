"""Dependency-injection protocols for the FuturesRL-X telemetry API.

Every route depends on a *protocol*, never on a concrete adapter, so the API is
testable with deterministic fakes and can run offline with no IBKR session:

* :class:`BacktestService` — run/list backtests and produce tear sheets,
* :class:`ExecutionService` — account snapshot + emergency flatten kill switch,
* :class:`RiskService` — LLM risk state and threshold overrides,
* :class:`MarketDataService` — pipeline waterfall and term structure,
* :class:`StrategyCatalog` — declarative strategy configs (in-memory editor).

The concrete implementations live in :mod:`src.api.services` and
:mod:`src.api.backtests_service`; wiring happens once in
:meth:`src.api.services.build_container`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from configs.strategy import StrategyConfig

from src.api.schemas import (
    AccountSnapshotResponse,
    BacktestListResponse,
    BacktestRunConfig,
    BacktestRunSummary,
    FlattenAllResponse,
    PipelineWaterfallResponse,
    RiskStateResponse,
    RiskThresholdOverride,
    RiskThresholds,
    StrategyConfigUpdate,
    StrategyConfigView,
    TearsheetResponse,
    TermStructureResponse,
)
from src.core.types import InstrumentId, MarketState, Position

__all__ = [
    "BacktestService",
    "ExecutionService",
    "MarketDataService",
    "RiskService",
    "StrategyCatalog",
]


@runtime_checkable
class StrategyCatalog(Protocol):
    """Read/validate/edit the declarative strategy configs (in memory)."""

    def list_configs(self) -> list[StrategyConfigView]:
        """Every discoverable strategy, sorted by id."""
        ...

    def get(self, strategy_id: str) -> StrategyConfigView | None:
        """One strategy view, or ``None`` when unknown."""
        ...

    def update(self, strategy_id: str, update: StrategyConfigUpdate) -> StrategyConfigView | None:
        """Merge + validate an edit; raise ``ValueError`` when invalid."""
        ...

    def core_config(self, strategy_id: str) -> StrategyConfig | None:
        """Validated core config consumed by the backtest engine."""
        ...

    def available_ids(self) -> list[str]:
        """All available strategy ids, sorted."""
        ...


@runtime_checkable
class ExecutionService(Protocol):
    """Account/position snapshot plus the emergency flatten kill switch."""

    async def account(self) -> AccountSnapshotResponse:
        """Current account valuation, positions and equity curve."""
        ...

    async def positions(self) -> Mapping[InstrumentId, Position]:
        """Broker positions keyed by root."""
        ...

    @property
    def is_locked(self) -> bool:
        """Whether execution is halted by the kill switch."""
        ...

    @property
    def lock_reason(self) -> str:
        """Human-readable kill-switch reason (empty when unlocked)."""
        ...

    async def flatten_all(self) -> FlattenAllResponse:
        """Market-close every position, cancel working orders and lock execution."""
        ...


@runtime_checkable
class RiskService(Protocol):
    """LLM Risk-First state plus the editable threshold overrides."""

    def state(self) -> RiskStateResponse:
        """Current assessments, circuit-breaker state and macro lockout."""
        ...

    def thresholds(self) -> RiskThresholds:
        """Effective thresholds (settings + API overrides)."""
        ...

    def update_thresholds(self, override: RiskThresholdOverride) -> RiskThresholds:
        """Validate + persist an override; raise ``ValueError`` when inconsistent."""
        ...


@runtime_checkable
class MarketDataService(Protocol):
    """Pipeline waterfall and term-structure/roll data."""

    def latest_state(self) -> MarketState:
        """Latest point-in-time market state used by the pipeline/risk views."""
        ...

    def pipeline(self) -> PipelineWaterfallResponse:
        """Most recent ``w_t = R_t(T_t(A_t(S_t(X))))`` trace."""
        ...

    def term_structure(self) -> TermStructureResponse:
        """Forward curves and upcoming roll calendar."""
        ...


@runtime_checkable
class BacktestService(Protocol):
    """Run store + offline backtest engine adapter."""

    def list_runs(self) -> BacktestListResponse:
        """Run history (newest first) plus available strategy ids."""
        ...

    def get_run(self, run_id: str) -> BacktestRunSummary | None:
        """A stored run summary, or ``None`` when unknown."""
        ...

    def run(self, request: BacktestRunConfig) -> BacktestRunSummary:
        """Execute a deterministic offline backtest and store the result."""
        ...

    def tearsheet(self, run_id: str) -> TearsheetResponse | None:
        """Tear sheet series + metrics for a stored run."""
        ...
