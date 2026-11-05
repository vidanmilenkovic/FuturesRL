"""Typed Pydantic request/response models for the FuturesRL telemetry API.

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
type DataSource = Literal["engine", "fixture", "runner-bridge"]
#: Provenance of a live/snapshot view: fresh runner bridge vs local fallback.
type SnapshotSource = Literal["runner-bridge", "engine", "simulated", "fixture"]
#: Provenance of the account equity curve (live IBKR vs synthetic vs offline).
type EquityCurveSource = Literal["ibkr-live", "simulated-backfill", "fixture"]
#: Provenance of the account/position snapshot.
type AccountSource = Literal["engine", "simulated", "fixture", "runner-bridge"]
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
    #: Additive Phase-9 connectivity/runner fields (frontend derives the mode badge).
    ibkr_connected: bool = False
    runner_bridge_available: bool = False
    runner_bridge_stale: bool = True
    runner_mode: str | None = None
    #: Additive Phase-10 IBKR resilience fields (dynamic, recomputed per request).
    ibkr_enabled: bool = False
    ibkr_reconnecting: bool = False
    ibkr_last_error: str | None = None


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
    #: Provenance: ``runner-bridge`` wins over ``engine``/``simulated``/``fixture``.
    source: AccountSource = "fixture"
    #: Additive Phase-10 provenance: ``True`` when this is a last-known-good book
    #: served while the live IBKR session is unavailable (never simulated data).
    stale: bool = False
    #: Human-readable reason the book is stale (``None`` when fresh).
    stale_reason: str | None = None
    #: Additive v1.5.1 attribution: provenance of ``equity_curve``.
    #: ``ibkr-live`` = real persisted IBKR NLV points; ``simulated-backfill`` =
    #: IBKR enabled but <5 live points so the synthetic backfill curve is shown;
    #: ``fixture`` = offline/simulated/bridge adapter (existing behaviour).
    equity_curve_source: EquityCurveSource = "fixture"
    #: Number of persisted real IBKR NLV points available for this book.
    equity_curve_live_points: int = Field(default=0, ge=0)
    #: Number of DISTINCT UTC calendar days covered by the live IBKR history.
    #: ``ibkr-live`` requires at least 5 distinct days (not merely 5 intraday points).
    equity_curve_live_days: int = Field(default=0, ge=0)
    #: Live recorded IBKR points for 1D plotting (latest recorded UTC day), even
    #: while the multi-day curve is still badged ``simulated-backfill``.
    equity_curve_intraday: list[EquityPoint] = Field(default_factory=list)
    #: Annualised ex-ante portfolio volatility (fraction, e.g. 0.119) from the
    #: live book weights x market-state covariance. ``0.0`` for a flat IBKR book,
    #: ``None`` when unavailable (never a synthetic period vol).
    portfolio_volatility: float | None = Field(default=None, ge=0.0)
    #: ``ex-ante-book`` | ``unavailable`` for IBKR; ``None`` for offline/bridge.
    portfolio_volatility_source: str | None = None


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
    #: Weight actually achieved by the discrete contract book (signed notional / equity).
    weight_actual: float | None = None
    delta_timing: float
    delta_risk: float
    target_contracts: int | None = None
    notional: float | None = None
    #: Why the weight changed across stages (same vocabulary as the engine waterfall).
    driver: str = ""
    #: Per-cycle market regime (bridge rows expose it when the runner publishes one).
    regime: str | None = None
    #: Human-readable adjustment reason (engine waterfall uses this; equals ``driver``).
    reason: str = ""


class PipelineDiagnostic(APIModel):
    """One stage diagnostic / guardrail result for the waterfall inspector."""

    stage: str
    detail: str
    value: str
    status: Severity


class MicroRoutingEntry(APIModel):
    """One standard -> micro contract routing pair from the discretization step."""

    family: str
    standard_root: str
    standard_quantity: int
    micro_root: str
    micro_quantity: int


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
    #: Rounding/discretization quality of ``w_t -> N_i,t`` (percentage points).
    tracking_error_pct: float = Field(default=0.0, ge=0.0)
    micro_routing: list[MicroRoutingEntry] = Field(default_factory=list)
    #: ``translator`` when the real discretization report was used, else ``fallback``.
    tracking_error_source: Literal["translator", "fallback"] = "fallback"


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
    #: ``news-ingestor`` when live/ingested headlines are wired, else ``fixture``.
    news_source: str = "fixture"
    #: Additive QoL kill-switch status, read from the execution adapter and the
    #: persisted risk state store; never synthesised when neither reports a latch.
    kill_switch_active: bool = False
    kill_switch_reason: str = ""
    daily_loss_breaker_active: bool = False
    lock_until: datetime | None = None
    #: Whether the execution adapter can accept an operator ``POST /api/risk/reset-lock``.
    reset_available: bool = False
    #: Whether ``FUTURESRL_OPERATOR_RESET_TOKEN`` is configured (token required).
    operator_reset_token_required: bool = False


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


#: Exact body confirmation phrase for ``POST /api/risk/reset-lock``.
RISK_LOCK_RESET_CONFIRM: str = "RESET RISK LOCK"

#: Exact body confirmation phrase for ``POST /api/maintenance/reset-paper-state``.
PAPER_STATE_RESET_CONFIRM: str = "RESET PAPER STATE"


class RiskLockResetRequest(StrictModel):
    """Body of ``POST /api/risk/reset-lock`` (operator kill-switch reset)."""

    confirm: str = Field(
        description=f'Must be exactly "{RISK_LOCK_RESET_CONFIRM}".',
    )
    #: Operator token; required when ``FUTURESRL_OPERATOR_RESET_TOKEN`` is set.
    operator_token: str | None = Field(default=None, repr=False)
    #: Alias of ``operator_token`` (accepted for script/frontend ergonomics).
    token: str | None = Field(default=None, repr=False)
    #: Also unlock/clear the durable risk state (persisted kill switch/daily latch,
    #: HWM/trailing-stop state). Defaults to true.
    reset_persisted_state: bool = True


class RiskLockResetResponse(APIModel):
    """Result of ``POST /api/risk/reset-lock`` with the updated lock status."""

    status: Literal["reset", "already_unlocked"]
    timestamp: datetime
    execution_reset: bool = False
    persisted_state_reset: bool = False
    kill_switch_active: bool = False
    kill_switch_reason: str = ""
    daily_loss_breaker_active: bool = False
    lock_until: datetime | None = None
    warnings: list[str] = Field(default_factory=list)
    message: str = ""


class PaperStateResetRequest(StrictModel):
    """Body of ``POST /api/maintenance/reset-paper-state``."""

    confirm: str = Field(
        description=f'Must be exactly "{PAPER_STATE_RESET_CONFIRM}".',
    )
    clear_risk_state: bool = True
    clear_runner_bridge: bool = True
    clear_paper_cache: bool = True


class PaperStateResetReceipts(APIModel):
    """Operator receipts for a paper-state reset (files/store/session actions)."""

    files_removed: list[str] = Field(default_factory=list)
    risk_state_store_cleared: bool = False
    risk_state_store_uri: str | None = None
    paper_session_reset: bool = False


class PaperStateResetResponse(APIModel):
    """Result of ``POST /api/maintenance/reset-paper-state``."""

    status: Literal["reset", "noop", "partial"]
    timestamp: datetime
    environment: Environment
    confirm_ok: bool
    receipts: PaperStateResetReceipts = Field(default_factory=PaperStateResetReceipts)
    warnings: list[str] = Field(default_factory=list)
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
    #: Additive QoL metadata: committed built-ins are protected from delete/rename.
    builtin: bool = False
    #: Persisted YAML path backing this view (overlay in test/backtest envs).
    path: str | None = None


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


class CachedBarsSummary(APIModel):
    """Phase-11B metadata describing the real Parquet bars used by a run/study."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    roots: list[str] = Field(default_factory=list)
    bar_count: int = Field(default=0, ge=0)
    start_date: date | None = None
    end_date: date | None = None
    schema_name: str = Field(default="ohlcv-1d", alias="schema")
    timeframe: str = "1d"


class BacktestRunConfig(StrictModel):
    """Request body for ``POST /api/backtests``."""

    strategy_id: str
    start: date
    end: date
    initial_equity: float = Field(default=1_000_000.0, gt=0.0)
    rebalance_frequency: str = "W-FRI"
    walk_forward: bool = False
    wf_splits: int = Field(default=5, ge=2)
    purge_bars: int = Field(default=5, ge=0)
    embargo_bars: int = Field(default=5, ge=0)
    random_seed: int = 7
    #: Phase-11B data source selection: auto falls back to synthetic fixtures,
    #: cached_real requires real cached Parquet bars, synthetic_offline never reads
    #: the cache.
    data_mode: Literal["auto", "cached_real", "synthetic_offline"] = "auto"
    #: Bar timeframe selector (Spec-1); synthetic runs propagate it for annualization.
    timeframe: str = "1d"

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
    #: ``backtest`` = single-pass run; ``walk_forward`` = purged/embargoed WF run.
    kind: Literal["backtest", "walk_forward"] = "backtest"
    #: Number of walk-forward folds (``None`` for a single-pass backtest).
    n_folds: int | None = Field(default=None, ge=0)
    #: Additive QoL snapshot of the exact validated strategy config used for the run.
    #: ``None`` for legacy artifacts / old fixture seeds (clients tolerate null).
    strategy_config: dict[str, object] | None = None
    #: Phase-11B data provenance (synthetic is the historical default/legacy value).
    data_origin: Literal["real_parquet", "synthetic_fixture"] = "synthetic_fixture"
    data_mode: Literal["auto", "cached_real", "synthetic_offline"] = "auto"
    cached_bars: CachedBarsSummary | None = None
    timeframe: str = "1d"


class BacktestListResponse(APIModel):
    """Run history for the Backtest Lab landing view."""

    runs: list[BacktestRunSummary] = Field(default_factory=list)
    available_strategies: list[str] = Field(default_factory=list)


class WalkForwardFold(APIModel):
    """One purged/embargoed walk-forward fold in the tear sheet."""

    fold_index: int = Field(ge=1)
    train_start: datetime | None = None
    train_end: datetime | None = None
    test_start: datetime | None = None
    test_end: datetime | None = None
    sharpe: float | None = None
    return_pct: float | None = None
    n_trades: int | None = Field(default=None, ge=0)
    equity_curve: list[SeriesPoint] = Field(default_factory=list)
    underwater: list[SeriesPoint] = Field(default_factory=list)


class TradeRow(APIModel):
    """One executed trade exposed by the tear sheet / CSV export (additive QoL)."""

    timestamp: datetime
    order_id: str
    instrument_id: str
    side: str
    quantity: int = Field(ge=0)
    price: float
    notional: float = 0.0
    commissions: float = 0.0
    slippage: float = 0.0
    kind: str = ""


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
    #: Per-fold series for ``kind="walk_forward"`` runs (empty for single-pass runs).
    folds: list[WalkForwardFold] = Field(default_factory=list)
    #: Additive QoL trade tape (empty for legacy artifacts / fixture runs).
    trades: list[TradeRow] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Live runner bridge / working orders / recent fills (Phase 9)
# ---------------------------------------------------------------------------


class WorkingOrderRow(APIModel):
    """One working (not fully filled, not terminal) order."""

    order_id: str
    instrument_id: str
    side: Literal["BUY", "SELL"]
    quantity: int = Field(ge=0)
    filled_quantity: int = Field(default=0, ge=0)
    #: Contracts still working at the venue (``quantity - filled_quantity`` unless reported).
    remaining_quantity: int = Field(default=0, ge=0)
    status: str = "SUBMITTED"
    order_type: str | None = None
    limit_price: float | None = None
    updated_at: datetime | None = None


class WorkingOrdersResponse(APIModel):
    """``GET /api/orders/working`` payload."""

    timestamp: datetime
    source: SnapshotSource
    orders: list[WorkingOrderRow] = Field(default_factory=list)
    bridge_stale: bool | None = None


class FillRow(APIModel):
    """One recent execution fill."""

    order_id: str
    instrument_id: str
    side: Literal["BUY", "SELL"]
    quantity: int = Field(gt=0)
    price: float = Field(gt=0.0)
    commission: float = 0.0
    timestamp: datetime


class RecentFillsResponse(APIModel):
    """``GET /api/fills/recent`` payload."""

    timestamp: datetime
    source: SnapshotSource
    fills: list[FillRow] = Field(default_factory=list)
    limit: int = Field(default=50, ge=1)
    bridge_stale: bool | None = None


class RunnerBridgeSummary(APIModel):
    """Summary of the runner bridge snapshot the API is currently reading."""

    runner_id: str | None = None
    mode: str | None = None
    sequence: int | None = None
    updated_at: datetime | None = None
    stale_after_seconds: float | None = None
    stale: bool = True
    status: str | None = None


class RunnerStatusResponse(APIModel):
    """``GET /api/runner/status`` payload (bridge staleness + session state)."""

    timestamp: datetime
    bridge_available: bool
    bridge_stale: bool
    runner_mode: str | None = None
    #: Alias of ``runner_mode`` (the bridge contract calls it ``mode``).
    mode: str | None = None
    #: Alias of ``bridge_stale`` (the bridge contract calls it ``stale``).
    stale: bool = True
    session_active: bool = False
    session_available: bool = False
    runner_id: str | None = None
    sequence: int | None = None
    updated_at: datetime | None = None
    stale_after_seconds: float | None = None
    status: str | None = None
    source: Literal["runner-bridge", "unavailable"] = "unavailable"
    message: str = ""


class RunnerStepRequest(StrictModel):
    """``POST /api/runner/step`` body."""

    states: int = Field(default=1, ge=1, le=10_000)
    mode: Literal["offline"] = "offline"


class RunnerStepResponse(APIModel):
    """``POST /api/runner/step`` payload."""

    status: Literal["completed", "unavailable", "error"]
    states_processed: int = Field(default=0, ge=0)
    mode: str = "offline"
    source: SnapshotSource | Literal["unavailable"] = "runner-bridge"
    bridge: RunnerBridgeSummary | None = None
    message: str = ""
    timestamp: datetime


# ---------------------------------------------------------------------------
# Telemetry envelopes (/ws/telemetry)
# ---------------------------------------------------------------------------


class PnlTelemetryPayload(APIModel):
    """Tick PnL frame.

    The additive attribution fields let the dashboard render the risk budget from
    the same enriched IBKR book as ``GET /api/positions`` instead of re-deriving it
    from a synthetic chart. For a flat IBKR book they are exactly ``0.0``/``0``.
    """

    net_liquidation: float
    day_pnl: float
    day_pnl_pct: float
    point: EquityPoint
    #: Ex-ante book volatility (annualised fraction); ``0.0`` for a flat IBKR book,
    #: ``None`` when unavailable/offline.
    portfolio_volatility: float | None = None
    initial_margin_used: float | None = None
    margin_utilization: float | None = None
    #: Number of open positions in the book driving this frame.
    open_positions: int = 0


class MarginTelemetryPayload(APIModel):
    """Margin-utilisation frame with additive live-book attribution."""

    initial_margin_used: float
    maintenance_margin_used: float
    margin_utilization: float
    warning_threshold: float
    critical_threshold: float
    #: Ex-ante book volatility (annualised fraction); ``0.0`` for a flat IBKR book.
    portfolio_volatility: float | None = None
    #: Number of open positions in the book driving this frame.
    open_positions: int = 0


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
    "PAPER_STATE_RESET_CONFIRM",
    "RISK_LOCK_RESET_CONFIRM",
    "AccountSnapshotResponse",
    "AttributionRow",
    "BacktestListResponse",
    "BacktestRunConfig",
    "BacktestRunSummary",
    "BacktestStatus",
    "DataSource",
    "Environment",
    "EquityCurveSource",
    "EquityPoint",
    "FillRow",
    "FlattenAllResponse",
    "FlattenReceipt",
    "HealthResponse",
    "HeartbeatTelemetryPayload",
    "MacroEvent",
    "MacroLockout",
    "MarginTelemetryPayload",
    "MetricsSet",
    "MicroRoutingEntry",
    "MonthlyReturn",
    "NewsEvent",
    "PaperStateResetReceipts",
    "PaperStateResetRequest",
    "PaperStateResetResponse",
    "PipelineDiagnostic",
    "PipelineInstrumentRow",
    "PipelineStage",
    "PipelineTelemetryPayload",
    "PipelineWaterfallResponse",
    "PnlTelemetryPayload",
    "PositionRow",
    "PositionsTelemetryPayload",
    "RecentFillsResponse",
    "RiskAction",
    "RiskAlertTelemetryPayload",
    "RiskAssessmentRow",
    "RiskLockResetRequest",
    "RiskLockResetResponse",
    "RiskStateResponse",
    "RiskThresholdOverride",
    "RiskThresholds",
    "RollCalendarEntry",
    "RunnerBridgeSummary",
    "RunnerStatusResponse",
    "RunnerStepRequest",
    "RunnerStepResponse",
    "SeriesPoint",
    "Severity",
    "SnapshotSource",
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
    "WalkForwardFold",
    "WorkingOrderRow",
    "WorkingOrdersResponse",
]
