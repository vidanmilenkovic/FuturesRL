/**
 * Typed contracts shared by the FuturesRL-X dashboard.
 *
 * These mirror `src/api/schemas.py` (Pydantic) one-to-one: the backend is the
 * source of truth for field names, the frontend never renames or reshapes a
 * payload. All timestamps are ISO-8601 UTC strings; all monetary values are USD
 * unless the field says otherwise. `null` is always "not available", never 0.
 */

// ---------------------------------------------------------------------------
// Primitives
// ---------------------------------------------------------------------------

export type AssetClass =
  | "equity_index"
  | "rates"
  | "energy"
  | "metals"
  | "fx"
  | "agriculture"
  | "crypto"
  | "unknown";

export type RiskAction = "pass" | "scale" | "liquidate" | "neutralized";

export type Severity = "ok" | "info" | "warn" | "critical";

export type TermRegime = "contango" | "backwardation" | "flat";

export type BacktestStatus = "queued" | "running" | "completed" | "failed";

export type ConnectionState = "connecting" | "live" | "demo" | "offline";

// ---------------------------------------------------------------------------
// System / health
// ---------------------------------------------------------------------------

export interface HealthResponse {
  status: "ok" | "degraded";
  app_name: string;
  version: string;
  environment: "development" | "paper" | "live" | "backtest";
  offline_mode: boolean;
  uptime_seconds: number;
  server_time: string;
}

export interface TelemetryStatusResponse {
  stream_active: boolean;
  connected_clients: number;
  published_events: number;
  publisher_interval_seconds: number;
  topics: string[];
  server_time: string;
}

// ---------------------------------------------------------------------------
// Account / positions (Command Center)
// ---------------------------------------------------------------------------

export interface PositionRow {
  instrument_id: string;
  root_symbol: string;
  asset_class: AssetClass;
  contract_symbol: string | null;
  expiration: string | null;
  days_to_roll: number | null;
  quantity: number;
  average_price: number;
  market_price: number;
  multiplier: number;
  notional: number;
  weight: number;
  unrealized_pnl: number;
  initial_margin: number;
  maintenance_margin: number;
}

export interface AccountSnapshotResponse {
  timestamp: string;
  net_liquidation: number;
  cash_balance: number;
  buying_power: number;
  initial_margin_used: number;
  maintenance_margin_used: number;
  excess_liquidity: number;
  margin_utilization: number;
  day_pnl: number;
  day_pnl_pct: number;
  gross_leverage: number;
  net_exposure: number;
  /** Point-in-time equity curve used by the Command Center chart. */
  equity_curve: EquityPoint[];
  positions: PositionRow[];
}

export interface EquityPoint {
  time: string;
  value: number;
}

// ---------------------------------------------------------------------------
// FinRL-X pipeline waterfall
// ---------------------------------------------------------------------------

export type PipelineStageKey = "universe" | "base" | "timing" | "risk" | "targets";

export interface PipelineStage {
  key: PipelineStageKey;
  label: string;
  formula: string;
  gross: number;
  net: number;
  instrument_count: number;
  note: string;
}

export interface PipelineInstrumentRow {
  instrument_id: string;
  asset_class: AssetClass;
  selected: boolean;
  weight_base: number;
  weight_timing: number;
  weight_risk: number;
  weight_final: number;
  delta_timing: number;
  delta_risk: number;
  target_contracts: number | null;
  notional: number | null;
  reason: string;
}

export interface PipelineDiagnostic {
  stage: string;
  detail: string;
  value: string;
  status: Severity;
}

export interface PipelineWaterfallResponse {
  timestamp: string;
  cycle_id: string;
  universe: string[];
  stages: PipelineStage[];
  instruments: PipelineInstrumentRow[];
  diagnostics: PipelineDiagnostic[];
  elapsed_ms: number;
  /** `engine` = real FinRL-X pipeline, `fixture` = documented offline fallback. */
  data_source?: "engine" | "fixture";
}

// ---------------------------------------------------------------------------
// LLM Risk-First engine
// ---------------------------------------------------------------------------

export interface RiskThresholds {
  llm_confidence_threshold: number;
  llm_safe_score: number;
  llm_critical_score: number;
  max_margin_utilization: number;
  margin_warning_threshold: number;
  margin_critical_threshold: number;
  daily_max_loss_pct: number;
  target_annual_volatility: number;
  macro_lockout_minutes_before: number;
  macro_lockout_minutes_after: number;
}

export type RiskThresholdOverride = Partial<RiskThresholds>;

export interface RiskAssessmentRow {
  instrument_id: string;
  timestamp: string;
  sentiment_score: number;
  semantic_risk_score: number;
  confidence: number;
  reasoning_summary: string;
  source: string;
  action: RiskAction;
  weight_multiplier: number;
}

export interface MacroEvent {
  id: string;
  event: string;
  event_time: string;
  tier: number;
  minutes_until: number;
  roots: string[];
}

export interface MacroLockout {
  active: boolean;
  event: string | null;
  starts_at: string | null;
  ends_at: string | null;
  minutes_remaining: number;
}

export interface NewsEvent {
  id: string;
  timestamp: string;
  headline: string;
  source: string;
  roots: string[];
  sentiment_score: number;
  severity: Severity;
  summary: string;
}

export interface RiskStateResponse {
  timestamp: string;
  thresholds: RiskThresholds;
  circuit_breaker_active: boolean;
  circuit_breaker_reason: string;
  macro_lockout: MacroLockout;
  assessments: RiskAssessmentRow[];
  macro_events: MacroEvent[];
  news: NewsEvent[];
  neutralized_instruments: string[];
}

export interface FlattenReceipt {
  order_id: string;
  instrument_id: string;
  side: "BUY" | "SELL";
  quantity: number;
  status: "FILLED" | "SUBMITTED" | "REJECTED" | "UNKNOWN";
  average_price: number | null;
  message: string;
}

export interface FlattenAllResponse {
  status: "flattened" | "halted" | "no_positions";
  timestamp: string;
  flattened_positions: number;
  cancelled_orders: number;
  receipts: FlattenReceipt[];
  execution_locked_until: string | null;
  message: string;
}

// ---------------------------------------------------------------------------
// Term structure & roll calendar
// ---------------------------------------------------------------------------

export interface TermContract {
  expiration: string;
  price: number;
  days_to_expiry: number;
  is_front: boolean;
  open_interest: number;
}

export interface TermCurve {
  root: string;
  asset_class: AssetClass;
  regime: TermRegime;
  slope: number;
  roll_yield_annualized: number;
  contracts: TermContract[];
}

export interface RollCalendarEntry {
  root: string;
  from_expiration: string;
  to_expiration: string;
  roll_date: string;
  days_to_roll: number;
  trigger: string;
  method: string;
  status: "scheduled" | "imminent" | "completed";
}

export interface TermStructureResponse {
  as_of: string;
  curves: TermCurve[];
  roll_calendar: RollCalendarEntry[];
}

// ---------------------------------------------------------------------------
// Strategies / backtests
// ---------------------------------------------------------------------------

export interface StageConfig {
  name: string;
  params: Record<string, number | string | boolean>;
}

export interface StrategyConfig {
  id: string;
  description: string;
  tags: string[];
  universe: StageConfig;
  allocator: StageConfig;
  timing: StageConfig[];
  risk: StageConfig[];
  updated_at: string;
}

export interface MetricsSet {
  cagr: number;
  annual_volatility: number;
  sharpe: number;
  sortino: number;
  calmar: number;
  omega: number;
  max_drawdown: number;
  max_drawdown_duration_days: number;
  var_95: number;
  var_99: number;
  cvar_95: number;
  cvar_99: number;
  rachev_ratio: number;
  rom: number;
  avg_margin_utilization: number;
  peak_margin_utilization: number;
  turnover: number;
  implementation_shortfall_ticks: number;
  commission_to_gross_pnl: number;
}

export interface SeriesPoint {
  time: string;
  value: number;
}

export interface TailPoint {
  rank: number;
  return_pct: number;
  tail: "gain" | "loss";
}

export interface AttributionRow {
  root: string;
  spot_pnl: number;
  roll_pnl: number;
}

export interface TearsheetResponse {
  run_id: string;
  equity_curve: SeriesPoint[];
  rolling_sharpe: SeriesPoint[];
  underwater: SeriesPoint[];
  tail_distribution: TailPoint[];
  roll_yield_attribution: AttributionRow[];
  monthly_returns: { month: string; value: number }[];
  metrics: MetricsSet;
}

export interface BacktestRunConfig {
  strategy_id: string;
  start: string;
  end: string;
  initial_equity: number;
  rebalance_frequency: string;
  walk_forward: boolean;
  random_seed: number;
}

export interface BacktestRunSummary {
  id: string;
  strategy_id: string;
  status: BacktestStatus;
  created_at: string;
  duration_seconds: number | null;
  config: BacktestRunConfig;
  metrics: MetricsSet | null;
  error: string | null;
  /** Activity counts: populated for real-engine runs, null for seeded fixtures. */
  n_trades?: number | null;
  n_rebalances?: number | null;
  /** Additive: `engine` = real BacktestEngine, `fixture` = deterministic seed. */
  data_source?: "engine" | "fixture";
}

export interface BacktestRunDetail extends BacktestRunSummary {
  tearsheet: TearsheetResponse | null;
}

export interface BacktestListResponse {
  runs: BacktestRunSummary[];
  available_strategies: string[];
}

// ---------------------------------------------------------------------------
// Telemetry envelopes (/ws/telemetry)
// ---------------------------------------------------------------------------

export type TelemetryKind =
  | "pnl"
  | "margin"
  | "positions"
  | "risk_alert"
  | "pipeline"
  | "heartbeat";

export interface PnlTelemetryPayload {
  net_liquidation: number;
  day_pnl: number;
  day_pnl_pct: number;
  point: EquityPoint;
}

export interface MarginTelemetryPayload {
  initial_margin_used: number;
  maintenance_margin_used: number;
  margin_utilization: number;
  warning_threshold: number;
  critical_threshold: number;
}

export interface PositionsTelemetryPayload {
  positions: PositionRow[];
  gross_leverage: number;
  net_exposure: number;
}

export interface RiskAlertTelemetryPayload {
  assessments: RiskAssessmentRow[];
  circuit_breaker_active: boolean;
  circuit_breaker_reason: string;
  macro_lockout: MacroLockout;
}

export interface PipelineTelemetryPayload {
  pipeline: PipelineWaterfallResponse;
}

export interface HeartbeatTelemetryPayload {
  status: "ok";
  sequence: number;
}

export type TelemetryPayload =
  | PnlTelemetryPayload
  | MarginTelemetryPayload
  | PositionsTelemetryPayload
  | RiskAlertTelemetryPayload
  | PipelineTelemetryPayload
  | HeartbeatTelemetryPayload;

export interface TelemetryEnvelope {
  type: "telemetry";
  kind: TelemetryKind;
  timestamp: string;
  sequence: number;
  payload: TelemetryPayload;
}

// ---------------------------------------------------------------------------
// API client envelope
// ---------------------------------------------------------------------------

export interface ApiResult<T> {
  data: T;
  /** `live` = served by the FastAPI backend, `demo` = deterministic fixture. */
  source: "live" | "demo";
  error: string | null;
}
