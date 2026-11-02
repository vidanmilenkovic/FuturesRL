"""Typed Pydantic request/response models for the FuturesRL-X telemetry API.

This module is the **frozen JSON contract**: every field name below is consumed
by ``frontend/lib/types.ts`` one-to-one. Changing a field is a breaking change
for the dashboard and must be mirrored there.

Conventions
-----------
* JSON is ``snake_case``; timestamps are timezone-aware UTC datetimes serialised
  as ISO-8601 (``2026-11-02T17:00:00Z``-equivalent with ``+00:00``).
* Money is USD, weights are signed notional fractions of NLV, percentages are
  fractions (``0.37`` = 37%).
* ``None`` always means "not available", never zero.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.core.types import AssetClass

# ---------------------------------------------------------------------------
# Shared / primitive aliases
# ---------------------------------------------------------------------------

type Environment = Literal["development", "paper", "live", "backtest"]
type Severity = Literal["ok", "info", "warn", "critical"]
type TermRegime = Literal["contango", "backwardation", "flat"]
type RiskAction = Literal["pass", "scale", "liquidate", "neutralized"]
type BacktestStatus = Literal["queued", "running", "completed", "failed"]
type DataSource = Literal["engine", "fixture"]
type PipelineStageKey = Literal["universe", "base", "timing", "risk", "targets"]
type TelemetryKind = Literal["pnl", "margin", "positions", "risk_alert", "pipeline", "heartbeat"]
#: A scalar allowed in a declarative pipeline stage ``params`` mapping.
type StageParamValue = bool | int | float | str


class APIModel(BaseModel):
    """Base response model (tolerant of additive backend fields)."""

    model_config = ConfigDict(extra="ignore")


class StrictModel(BaseModel):
    """Base request model (rejects unknown fields)."""

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# System / health
# ---------------------------------------------------------------------------


class HealthResponse(APIModel):
    """Liveness/readiness payload consumed by the dashboard shell."""

    status: Literal["ok", "degraded"]
    app_name: str
    version: str
    environment: Environment
    offline_mode: bool = Field(
        description="True when the API serves deterministic offline fixtures (no IBKR/Databento)."
    )
    uptime_seconds: float = Field(ge=0.0)
    server_time: datetime


class TelemetryStatusResponse(APIModel):
    """State of the WebSocket telemetry broadcaster."""

    stream_active: bool
    connected_clients: int = Field(ge=0)
    published_events: int = Field(ge=0)
    publisher_interval_seconds: float = Field(gt=0.0)
    topics: list[str]
    server_time: datetime


# ---------------------------------------------------------------------------
# Account / positions
# ---------------------------------------------------------------------------


class EquityPoint(APIModel):
    """One point of the equity curve (daily close, mark-to-market)."""

    time: datetime
    value: float


class PositionRow(APIModel):
    """One active futures contract position with derived analytics fields."""

    instrument_id: str
    root_symbol: str
    asset_class: AssetClass
    contract_symbol: str | None = None
    expiration: date | None = None
    days_to_roll: int | None = None
    quantity: int
    average_price: float
    market_price: float
    multiplier: float = Field(gt=0.0)
    notional: float
    weight: float
    unrealized_pnl: float
    initial_margin: float = Field(ge=0.0)
    maintenance_margin: float = Field(ge=0.0)


class AccountSnapshotResponse(APIModel):
    """Broker account valuation at one instant, plus the equity curve."""

    timestamp: datetime
    net_liquidation: float = Field(gt=0.0)
    cash_balance: float
    buying_power: float
    initial_margin_used: float = Field(ge=0.0)
    maintenance_margin_used: float = Field(ge=0.0)
    excess_liquidity: float
    margin_utilization: float = Field(ge=0.0)
    day_pnl: float
    day_pnl_pct: float
    gross_leverage: float = Field(ge=0.0)
    net_exposure: float
    equity_curve: list[EquityPoint] = Field(default_factory=list)
    positions: list[PositionRow] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# FinRL-X pipeline waterfall
# ---------------------------------------------------------------------------


class PipelineStage(APIModel):
    """One stage of ``S_t -> w_base -> w_timing -> w_risk -> N_i,t``."""

    key: PipelineStageKey
    label: str
    formula: str
    gross: float = Field(ge=0.0)
    net: float
    instrument_count: int = Field(ge=0)
    note: str = ""


class PipelineInstrumentRow(APIModel):
    """Per-instrument weight trace across the pipeline stages."""

    instrument_id: str
    asset_class: AssetClass
    selected: bool
    weight_base: float
    weight_timing: float
    weight_risk: float
    weight_final: float
    delta_timing: float
    delta_risk: float
    target_contracts: int | None = None
    notional: float | None = None
    reason: str = ""


class PipelineDiagnostic(APIModel):
    """One stage diagnostic / guardrail result for the waterfall inspector."""

    stage: str
    detail: str
    value: str
    status: Severity


class PipelineWaterfallResponse(APIModel):
    """A complete rebalance cycle trace for the Pipeline Inspector view."""

    timestamp: datetime
    cycle_id: str
    universe: list[str]
    stages: list[PipelineStage]
    instruments: list[PipelineInstrumentRow]
    diagnostics: list[PipelineDiagnostic]
    elapsed_ms: float = Field(ge=0.0)
    data_source: DataSource = "fixture"


# ---------------------------------------------------------------------------
# LLM Risk-First engine
# ---------------------------------------------------------------------------


class RiskThresholds(APIModel):
    """Effective risk parameters (the API override layer above ``RiskSettings``)."""

    llm_confidence_threshold: float = Field(ge=0.0, le=1.0)
    llm_safe_score: float = Field(ge=1.0, le=5.0)
    llm_critical_score: float = Field(ge=1.0, le=5.0)
    max_margin_utilization: float = Field(gt=0.0, le=1.0)
    margin_warning_threshold: float = Field(gt=0.0, le=1.0)
    margin_critical_threshold: float = Field(gt=0.0, le=1.0)
    daily_max_loss_pct: float = Field(gt=0.0, le=1.0)
    target_annual_volatility: float = Field(gt=0.0)
    macro_lockout_minutes_before: int = Field(ge=0)
    macro_lockout_minutes_after: int = Field(ge=0)


class RiskThresholdOverride(StrictModel):
    """Partial risk-threshold update (``PUT /api/risk/thresholds``)."""

    llm_confidence_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    llm_safe_score: float | None = Field(default=None, ge=1.0, le=5.0)
    llm_critical_score: float | None = Field(default=None, ge=1.0, le=5.0)
    max_margin_utilization: float | None = Field(default=None, gt=0.0, le=1.0)
    margin_warning_threshold: float | None = Field(default=None, gt=0.0, le=1.0)
    margin_critical_threshold: float | None = Field(default=None, gt=0.0, le=1.0)
    daily_max_loss_pct: float | None = Field(default=None, gt=0.0, le=1.0)
    target_annual_volatility: float | None = Field(default=None, gt=0.0)
    macro_lockout_minutes_before: int | None = Field(default=None, ge=0)
    macro_lockout_minutes_after: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _partial_ordering(self) -> RiskThresholdOverride:
        if (
            self.llm_safe_score is not None
            and self.llm_critical_score is not None
            and self.llm_critical_score <= self.llm_safe_score
        ):
            raise ValueError("llm_critical_score must exceed llm_safe_score")
        if (
            self.margin_warning_threshold is not None
            and self.margin_critical_threshold is not None
            and self.margin_critical_threshold <= self.margin_warning_threshold
        ):
            raise ValueError("margin_critical_threshold must exceed margin_warning_threshold")
        return self


class RiskAssessmentRow(APIModel):
    """Structured LLM assessment plus the deterministic breaker projection."""

    instrument_id: str
    timestamp: datetime
    sentiment_score: float = Field(ge=1.0, le=5.0)
    semantic_risk_score: float = Field(ge=1.0, le=5.0)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning_summary: str
    source: str
    action: RiskAction
    weight_multiplier: float = Field(ge=0.0, le=1.0)


class MacroEvent(APIModel):
    """One scheduled Tier-1/Tier-2 macro release used for lockout windows."""

    id: str
    event: str
    event_time: datetime
    tier: int = Field(ge=1)
    minutes_until: int
    roots: list[str] = Field(default_factory=list)


class MacroLockout(APIModel):
    """Whether a Tier-1 macro lockout window is currently active."""

    active: bool
    event: str | None = None
    starts_at: datetime | None = None
    ends_at: datetime | None = None
    minutes_remaining: int = Field(ge=0)


class NewsEvent(APIModel):
    """One LLM-ingested headline for the Risk Monitor live feed."""

    id: str
    timestamp: datetime
    headline: str
    source: str
    roots: list[str] = Field(default_factory=list)
    sentiment_score: float = Field(ge=1.0, le=5.0)
    severity: Severity
    summary: str


class RiskStateResponse(APIModel):
    """Complete LLM Risk-First state for the Risk Monitor view."""

    timestamp: datetime
    thresholds: RiskThresholds
    circuit_breaker_active: bool
    circuit_breaker_reason: str
    macro_lockout: MacroLockout
    assessments: list[RiskAssessmentRow] = Field(default_factory=list)
    macro_events: list[MacroEvent] = Field(default_factory=list)
    news: list[NewsEvent] = Field(default_factory=list)
    neutralized_instruments: list[str] = Field(default_factory=list)


class FlattenReceipt(APIModel):
    """Per-leg acknowledgement of the emergency flatten."""

    order_id: str
    instrument_id: str
    side: Literal["BUY", "SELL"]
    quantity: int = Field(ge=0)
    status: Literal["FILLED", "SUBMITTED", "REJECTED", "UNKNOWN"]
    average_price: float | None = None
    message: str = ""


class FlattenAllResponse(APIModel):
    """Result of the ``POST /api/risk/flatten-all`` kill switch."""

    status: Literal["flattened", "halted", "no_positions"]
    timestamp: datetime
    flattened_positions: int = Field(ge=0)
    cancelled_orders: int = Field(ge=0)
    receipts: list[FlattenReceipt] = Field(default_factory=list)
    execution_locked_until: datetime | None = None
    message: str = ""


# ---------------------------------------------------------------------------
# Term structure & roll calendar
# ---------------------------------------------------------------------------


class TermContract(APIModel):
    """One listed expiry on a root's forward curve."""

    expiration: date
    price: float = Field(gt=0.0)
    days_to_expiry: int = Field(ge=0)
    is_front: bool
    open_interest: int = Field(ge=0)


class TermCurve(APIModel):
    """The forward curve of one futures root."""

    root: str
    asset_class: AssetClass
    regime: TermRegime
    slope: float
    roll_yield_annualized: float
    contracts: list[TermContract] = Field(default_factory=list)


class RollCalendarEntry(APIModel):
    """One scheduled roll from the front to the next contract."""

    root: str
    from_expiration: date
    to_expiration: date
    roll_date: date
    days_to_roll: int
    trigger: str
    method: str
    status: Literal["scheduled", "imminent", "completed"]


class TermStructureResponse(APIModel):
    """Curves plus roll calendar for the Term Structure view."""

    as_of: datetime
    curves: list[TermCurve] = Field(default_factory=list)
    roll_calendar: list[RollCalendarEntry] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


class StageConfig(APIModel):
    """One declarative pipeline stage (registry name + params)."""

    name: str
    params: dict[str, StageParamValue] = Field(default_factory=dict)


class StrategyConfigView(APIModel):
    """Dashboard view of a validated declarative strategy pipeline."""

    id: str
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    universe: StageConfig
    allocator: StageConfig
    timing: list[StageConfig] = Field(default_factory=list)
    risk: list[StageConfig] = Field(default_factory=list)
    updated_at: datetime


class StrategyConfigUpdate(BaseModel):
    """Editable subset of a strategy config (``PUT /api/strategies/{id}``).

    Unknown fields are ignored so a client can PUT back a full
    ``StrategyConfigView`` (including the read-only ``id``/``updated_at``) without
    a validation error.
    """

    model_config = ConfigDict(extra="ignore")

    description: str | None = None
    tags: list[str] | None = None
    universe: StageConfig | None = None
    allocator: StageConfig | None = None
    timing: list[StageConfig] | None = None
    risk: list[StageConfig] | None = None


# ---------------------------------------------------------------------------
# Backtests & tear sheets
# ---------------------------------------------------------------------------


class BacktestRunConfig(StrictModel):
    """Request body for ``POST /api/backtests``."""

    strategy_id: str
    start: date
    end: date
    initial_equity: float = Field(default=1_000_000.0, gt=0.0)
    rebalance_frequency: str = "W-FRI"
    walk_forward: bool = True
    random_seed: int = 7

    @model_validator(mode="after")
    def _ordered_dates(self) -> BacktestRunConfig:
        if self.end <= self.start:
            raise ValueError("backtest end must be after start")
        return self


class MetricsSet(APIModel):
    """Institutional metrics rendered by the backtest tear-sheet lab."""

    cagr: float
    annual_volatility: float
    sharpe: float
    sortino: float
    calmar: float
    omega: float
    max_drawdown: float
    max_drawdown_duration_days: float
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    rachev_ratio: float
    rom: float
    avg_margin_utilization: float
    peak_margin_utilization: float
    turnover: float
    implementation_shortfall_ticks: float
    commission_to_gross_pnl: float


class SeriesPoint(APIModel):
    """A named time-series point (equity, rolling Sharpe, drawdown)."""

    time: datetime
    value: float


class TailPoint(APIModel):
    """One observation of the Rachev gain/loss tail plot."""

    rank: int = Field(ge=1)
    return_pct: float
    tail: Literal["gain", "loss"]


class AttributionRow(APIModel):
    """Spot/price versus roll-yield PnL attribution for one root."""

    root: str
    spot_pnl: float
    roll_pnl: float


class MonthlyReturn(APIModel):
    """Calendar-month return."""

    month: str
    value: float


class BacktestRunSummary(APIModel):
    """Stored backtest run metadata (list/get responses)."""

    id: str
    strategy_id: str
    status: BacktestStatus
    created_at: datetime
    duration_seconds: float | None = None
    config: BacktestRunConfig
    metrics: MetricsSet | None = None
    error: str | None = None
    #: Activity counts (additive): populated for real-engine runs, ``None`` for seeds.
    n_trades: int | None = Field(default=None, ge=0)
    n_rebalances: int | None = Field(default=None, ge=0)
    #: ``engine`` = real BacktestEngine; ``fixture`` = deterministic seeded run.
    data_source: DataSource = "engine"


class BacktestListResponse(APIModel):
    """Run history for the Backtest Lab landing view."""

    runs: list[BacktestRunSummary] = Field(default_factory=list)
    available_strategies: list[str] = Field(default_factory=list)


class TearsheetResponse(APIModel):
    """Tear sheet series + metrics for one run."""

    run_id: str
    equity_curve: list[SeriesPoint] = Field(default_factory=list)
    rolling_sharpe: list[SeriesPoint] = Field(default_factory=list)
    underwater: list[SeriesPoint] = Field(default_factory=list)
    tail_distribution: list[TailPoint] = Field(default_factory=list)
    roll_yield_attribution: list[AttributionRow] = Field(default_factory=list)
    monthly_returns: list[MonthlyReturn] = Field(default_factory=list)
    metrics: MetricsSet


# ---------------------------------------------------------------------------
# Telemetry envelopes (/ws/telemetry)
# ---------------------------------------------------------------------------


class PnlTelemetryPayload(APIModel):
    """Tick PnL frame."""

    net_liquidation: float
    day_pnl: float
    day_pnl_pct: float
    point: EquityPoint


class MarginTelemetryPayload(APIModel):
    """Margin-utilisation frame."""

    initial_margin_used: float
    maintenance_margin_used: float
    margin_utilization: float
    warning_threshold: float
    critical_threshold: float


class PositionsTelemetryPayload(APIModel):
    """Position-book frame."""

    positions: list[PositionRow] = Field(default_factory=list)
    gross_leverage: float
    net_exposure: float


class RiskAlertTelemetryPayload(APIModel):
    """LLM risk / circuit-breaker frame."""

    assessments: list[RiskAssessmentRow] = Field(default_factory=list)
    circuit_breaker_active: bool
    circuit_breaker_reason: str
    macro_lockout: MacroLockout


class PipelineTelemetryPayload(APIModel):
    """Pipeline waterfall frame."""

    pipeline: PipelineWaterfallResponse


class HeartbeatTelemetryPayload(APIModel):
    """Keep-alive frame emitted when the stream is otherwise quiet."""

    status: Literal["ok"]
    sequence: int


type TelemetryPayload = (
    PnlTelemetryPayload
    | MarginTelemetryPayload
    | PositionsTelemetryPayload
    | RiskAlertTelemetryPayload
    | PipelineTelemetryPayload
    | HeartbeatTelemetryPayload
)


class TelemetryEnvelope(APIModel):
    """One frame pushed over ``/ws/telemetry``."""

    type: Literal["telemetry"] = "telemetry"
    kind: TelemetryKind
    timestamp: datetime
    sequence: int = Field(ge=1)
    payload: TelemetryPayload


__all__ = [
    "AccountSnapshotResponse",
    "AttributionRow",
    "BacktestListResponse",
    "BacktestRunConfig",
    "BacktestRunSummary",
    "BacktestStatus",
    "DataSource",
    "Environment",
    "EquityPoint",
    "FlattenAllResponse",
    "FlattenReceipt",
    "HealthResponse",
    "HeartbeatTelemetryPayload",
    "MacroEvent",
    "MacroLockout",
    "MarginTelemetryPayload",
    "MetricsSet",
    "MonthlyReturn",
    "NewsEvent",
    "PipelineDiagnostic",
    "PipelineInstrumentRow",
    "PipelineStage",
    "PipelineTelemetryPayload",
    "PipelineWaterfallResponse",
    "PnlTelemetryPayload",
    "PositionRow",
    "PositionsTelemetryPayload",
    "RiskAction",
    "RiskAlertTelemetryPayload",
    "RiskAssessmentRow",
    "RiskStateResponse",
    "RiskThresholdOverride",
    "RiskThresholds",
    "RollCalendarEntry",
    "SeriesPoint",
    "Severity",
    "StageConfig",
    "StageParamValue",
    "StrategyConfigUpdate",
    "StrategyConfigView",
    "TailPoint",
    "TearsheetResponse",
    "TelemetryEnvelope",
    "TelemetryKind",
    "TelemetryPayload",
    "TelemetryStatusResponse",
    "TermContract",
    "TermCurve",
    "TermRegime",
    "TermStructureResponse",
]
