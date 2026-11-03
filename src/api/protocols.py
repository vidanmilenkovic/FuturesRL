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
    RecentFillsResponse,
    RiskStateResponse,
    RiskThresholdOverride,
    RiskThresholds,
    RunnerStatusResponse,
    RunnerStepRequest,
    RunnerStepResponse,
    StrategyConfigUpdate,
    StrategyConfigView,
    TearsheetResponse,
    TermStructureResponse,
    WorkingOrdersResponse,
)
from src.core.types import InstrumentId, MarketState, Position

__all__ = [
    "BacktestService",
    "ExecutionService",
    "LiveStateService",
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


@runtime_checkable
class LiveStateService(Protocol):
    """Bridge-first live state (orders, fills, account, runner session)."""

    def bridge_available(self) -> bool:
        """Whether a runner bridge snapshot is currently readable."""
        ...

    def bridge_stale(self) -> bool:
        """Whether the latest bridge snapshot is stale (or absent)."""
        ...

    def runner_mode(self) -> str | None:
        """Runner mode reported by the bridge (``None`` when absent)."""
        ...

    def bridge_pipeline(self) -> PipelineWaterfallResponse | None:
        """Fresh runner pipeline snapshot, or ``None`` when unavailable/stale."""
        ...

    async def account(self) -> AccountSnapshotResponse:
        """Bridge-backed account snapshot when fresh, else the execution adapter."""
        ...

    async def working_orders(self) -> WorkingOrdersResponse:
        """Working orders (fresh bridge preferred)."""
        ...

    async def recent_fills(self, limit: int = 50) -> RecentFillsResponse:
        """Recent fills, newest first (fresh bridge preferred)."""
        ...

    def runner_status(self) -> RunnerStatusResponse:
        """Bridge staleness plus offline-runner session state."""
        ...

    async def runner_step(self, request: RunnerStepRequest) -> RunnerStepResponse:
        """Advance the singleton offline runner session."""
        ...

    async def runner_reset(self) -> RunnerStatusResponse:
        """Close the singleton offline runner session and report status."""
        ...

    async def start(self) -> None:
        """Lifecycle hook (no-op unless a session is eagerly required)."""
        ...

    async def close(self) -> None:
        """Release the offline runner session on shutdown."""
        ...
