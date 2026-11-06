/**
 * Typed contracts shared by the FuturesRL dashboard.
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
  /** Phase 9 additive bridge state (absent on older backends). */
  ibkr_connected?: boolean;
  runner_bridge_available?: boolean;
  runner_bridge_stale?: boolean;
  runner_mode?: string | null;
  /** Phase 10 additive reconnect state (absent on older backends). */
  ibkr_enabled?: boolean;
  ibkr_reconnecting?: boolean;
  ibkr_last_error?: string | null;
  /** Workstation UX additive LLM status. */
  llm_status?: LLMStatusResponse | null;
}

export type LlmMode = "live_api" | "offline_stub";
export type LlmStatus =
  | "online"
  | "offline_stub"
  | "auth_error"
  | "unreachable";

export interface LLMStatusResponse {
  provider: string;
  model: string;
  api_key_configured: boolean;
  mode: LlmMode | string;
  status: LlmStatus | string;
  last_latency_ms: number | null;
  last_checked_at: string | null;
  cache_entries: number;
  error_detail: string | null;
}

/** Data provenance shown by the ribbon badge. */
export type DataModeKey =
  | "connecting"
  | "ui_fallback"
  | "offline_engine"
  | "reconnecting"
  | "paper_ibkr"
  | "live_ibkr";

export interface DataMode {
  key: DataModeKey;
  label: string;
  description: string;
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
  /** Additive: last-known-good book served while IBKR reconnects. */
  stale?: boolean;
  stale_reason?: string | null;
  source?: string;
  /** v1.5.1 live-book attribution (task-38). */
  equity_curve_source?: "ibkr-live" | "simulated-backfill" | "fixture" | string;
  equity_curve_live_points?: number;
  /** Distinct UTC days covered by the persisted live NLV history. */
  equity_curve_live_days?: number;
  /** Real intraday live NLV points for the 1D timeframe. */
  equity_curve_intraday?: EquityPoint[];
  portfolio_volatility?: number | null;
  portfolio_volatility_source?: string | null;
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

export interface MicroRoutingRow {
  family: string;
  standard_root: string;
  standard_quantity: number;
  micro_root: string;
  micro_quantity: number;
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
  /** Phase 9 additive: broker-confirmed weight after fills (if bridged). */
  weight_actual?: number | null;
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
  /** Phase 9 additive: realised tracking error of the executed book (percent). */
  tracking_error_pct?: number | null;
  /** Phase 9 additive: standard → micro contract routing applied by the translator. */
  micro_routing?: MicroRoutingRow[];
  /** Workstation UX additive 45-root scope provenance. */
  scope?: RootScope | string | null;
  sector?: string | null;
  selected_roots?: string[];
  data_origin?: string | null;
}

// ---------------------------------------------------------------------------
// Workstation UX — shared root scope selection
// ---------------------------------------------------------------------------

export type RootScope = "strategy" | "cached" | "all";

export interface RootScopeQuery {
  scope?: RootScope;
  sector?: string | null;
  roots?: string[];
}

export interface RootScopeState extends RootScopeQuery {
  scope: RootScope;
  sector: string;
  roots: string[];
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
  /** Workstation UX additive tooltip metadata (absent on older payloads). */
  regime?: string | null;
  confidence_dispersion?: number | null;
  dispersion?: number | null;
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
  /** QoL additive kill-switch/execution-lock state (absent on older backends). */
  kill_switch_active?: boolean;
  kill_switch_reason?: string | null;
  daily_loss_breaker_active?: boolean;
  lock_until?: string | null;
  execution_locked_until?: string | null;
  reset_available?: boolean;
  operator_reset_token_required?: boolean;
  /** Workstation UX additive 45-root scope / per-root tooltip metadata. */
  scope?: RootScope | string | null;
  sector?: string | null;
  selected_roots?: string[];
  derived_roots?: string[];
  assessment_source?: string | null;
  data_origin?: string | null;
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
  /** Optional backend-provided annualised roll yield for this leg (percent). */
  roll_yield_annualized?: number;
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
  /** Workstation UX additive 45-root scope provenance. */
  scope?: RootScope | string | null;
  sector?: string | null;
  selected_roots?: string[];
  data_origin?: string | null;
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
  /** Present on studio/catalog payloads; not exposed by GET /api/strategies today. */
  drl?: StudioDrlConfig;
  /** QoL: committed built-in strategy flag (delete/rename protection). */
  builtin?: boolean;
  source?: string | null;
  path?: string | null;
  strategy_path?: string | null;
}

// ---------------------------------------------------------------------------
// Strategy Studio (Phase 10)
// ---------------------------------------------------------------------------

export type StudioStageKey = "universe" | "allocator" | "timing" | "risk" | "translator";

export type StudioParamType =
  | "int"
  | "float"
  | "bool"
  | "select"
  | "multiselect"
  | "root_multiselect"
  | "text";

export interface StudioParamOption {
  value: string | number | boolean;
  label: string;
  description?: string;
}

/** One editable strategy parameter served by GET /api/strategies/schema/catalog. */
export type StudioParamValue = string | number | boolean | string[];

export interface StudioStageConfig {
  name: string;
  params: Record<string, StudioParamValue>;
}

export interface StudioParameter {
  /** Dotted draft path understood by the builder, e.g. `timing.tsmom.params.lookback_fast`. */
  path: string;
  stage: StudioStageKey;
  /** Overlay / registry stage name (tsmom, quant_risk, ...), when applicable. */
  group?: string | null;
  label: string;
  description: string;
  type: StudioParamType;
  default: StudioParamValue;
  min?: number | null;
  max?: number | null;
  step?: number | null;
  options?: StudioParamOption[];
  unit?: string | null;
  optuna_tunable: boolean;
  /** Curated Optuna default selection (v1.5.1, optuna_default_selected). */
  optuna_default_selected?: boolean;
  /** Structural / data-hygiene flag from the v1.5.1 catalog. */
  hygiene?: boolean;
  structural?: boolean;
  /** Catalog hints (Phase 10): log-scaled sliders and YAML encodings. */
  log_scale?: boolean;
  yaml_encoding?: string | null;
  /** Additive Workstation UX metadata: one-line math/explanation (allocator tooltips). */
  formula?: string | null;
  /** Additive Workstation UX metadata: advanced/math-detail parameter for the collapse section. */
  advanced?: boolean;
  min_items?: number | null;
  max_items?: number | null;
  /** Backend stage group (e.g. timing overlay name). */
  stage_name?: string | null;
  /** Stable catalog key used in optimization search spaces. */
  key?: string;
  /** `stage_params` | `translator` | `drl_block`. */
  location?: string | null;
}

export interface StudioTranslatorConfig {
  rounding_mode: "nearest" | "floor" | "ceil" | string;
  rebalance_buffer_contracts: number;
  max_margin_utilization: number;
  enable_micro_routing: boolean;
  min_micro_notional_ratio: number;
}

/** Hypothesis template returned by the studio catalog (5 presets). */
export interface StudioDrlConfig {
  algorithm: string;
  policy: string;
  total_timesteps: number;
  learning_rate: number;
  gamma: number;
  seed: number;
  reward_risk_penalty_lambda: number;
  normalize_observations: boolean;
  model_path: string | null;
}

export interface StudioPreset {
  id: string;
  name: string;
  description: string;
  hypothesis: string;
  tags: string[];
  universe: StudioStageConfig;
  allocator: StudioStageConfig;
  timing: StudioStageConfig[];
  risk: StudioStageConfig[];
  translator: StudioTranslatorConfig;
  /** Complete POST-ready strategy dict returned by the catalog. */
  strategy?: StudioDraft | null;
  drl?: StudioDrlConfig;
}

export interface InstrumentRoot {
  root: string;
  name: string;
  asset_class: string;
  sector: string;
  exchange: string;
  multiplier: number;
  tick_size: number;
  currency: string;
  /** Paired micro root for a standard contract (None for micros). */
  micro_root: string | null;
  /** Standard root a micro contract rolls up to (None for standards). */
  standard_root: string | null;
}

export interface InstrumentSector {
  id: string;
  label: string;
  count: number;
  roots: InstrumentRoot[];
}

export interface InstrumentsResponse {
  generated_at: string;
  count: number;
  sectors: InstrumentSector[];
  roots: InstrumentRoot[];
}

export interface StudioUniverseMetadata {
  roots: InstrumentRoot[];
  sectors: InstrumentSector[];
}

export interface StudioCatalogResponse {
  version?: string | null;
  roots: string[];
  parameters: StudioParameter[];
  presets: StudioPreset[];
  /** Optional list of saved strategy views served with the catalog. */
  strategies?: StrategyConfig[];
  source?: string | null;
  preset_source?: string | null;
  generated_at?: string | null;
  /** Additive selector/allocator type choices (task-31). */
  selector_choices?: StudioParamOption[];
  allocator_choices?: StudioParamOption[];
  /** Raw `stage_choices` map from the catalog (universe/allocator/timing/risk/translator). */
  stage_choices?: Record<string, string[]>;
  /** Additive sector-grouped universe metadata (task-31). */
  universe?: StudioUniverseMetadata | null;
  /** Additive Workstation UX metadata: allocator family -> one-line math/why explanation. */
  formulas?: Record<string, string>;
  /** Additive Workstation UX metadata: keys of advanced/math-detail parameters. */
  advanced_parameter_keys?: string[];
}

/** Editable draft the Manual Builder serializes to YAML + POST/PUT payload. */
export interface StudioDraft {
  id: string;
  description: string;
  tags: string[];
  universe: StudioStageConfig;
  allocator: StudioStageConfig;
  timing: StudioStageConfig[];
  risk: StudioStageConfig[];
  translator: StudioTranslatorConfig;
  /** Optional DRL policy section (present in the DRL preset). */
  drl?: StudioDrlConfig;
}

export interface StudioSaveResult {
  strategy: StrategyConfig;
  /** `live` = FastAPI persisted the YAML, `demo` = local fixture (UI fallback). */
  persisted: boolean;
  message: string;
}

export interface StudioValidation {
  valid: boolean;
  errors: string[];
  warnings: string[];
}

// ---------------------------------------------------------------------------
// Optuna auto-optimizer (Phase 10)
// ---------------------------------------------------------------------------

export type OptimizationObjective =
  | "sharpe"
  | "sortino"
  | "calmar"
  | "cagr"
  | "rachev"
  | "return_on_margin";

export interface OptimizationParamRange {
  param_path: string;
  type: "int" | "float" | "categorical";
  low?: number;
  high?: number;
  step?: number;
  choices?: string[];
}

export interface OptimizationRequest {
  base_strategy_id: string;
  objective_metric: OptimizationObjective;
  n_trials: number;
  walk_forward: boolean;
  wf_splits: number;
  purge_bars: number;
  embargo_bars: number;
  drawdown_penalty_threshold: number;
  search_space: OptimizationParamRange[];
  /** Optional deterministic seed (defaults to 7 on the backend). */
  seed?: number;
  /** Phase 11B: auto = real cached bars when sufficient, else synthetic. */
  data_mode?: MarketDataMode;
  /** Phase 11C target bar timeframe. */
  timeframe?: DataTimeframe;
}

export interface OptimizationTrial {
  trial_number: number;
  params: Record<string, number | string | boolean>;
  state?: string | null;
  oos_sharpe: number;
  cagr: number;
  max_dd: number;
  calmar: number;
  rachev: number;
  duration_ms: number;
  /** Objective value after the drawdown penalty (what Optuna maximised). */
  value: number;
  best: boolean;
  /** v1.5.3 activity metrics (flat/low-exposure trials are marked inactive). */
  n_trades?: number | null;
  avg_gross_exposure?: number | null;
  zero_activity?: boolean;
}

export interface OptimizationImportance {
  param_path: string;
  importance: number;
}

export interface OptimizationStudy {
  study_id: string;
  base_strategy_id: string;
  objective_metric: OptimizationObjective | string;
  n_trials: number;
  walk_forward: boolean;
  wf_splits: number;
  status: "completed" | "running" | "failed" | string;
  created_at: string;
  duration_ms: number;
  best_value: number | null;
  best_trial_number: number | null;
  best_params: Record<string, number | string | boolean>;
  param_importances: OptimizationImportance[];
  trials: OptimizationTrial[];
  /** Full strategy config ready to save/apply (optional in the contract). */
  best_strategy_config?: StudioDraft | null;
  direction?: string | null;
  completed_trials?: number | null;
  fold_count?: number | null;
  backend?: string | null;
  provenance?: "engine" | "fixture" | string | null;
  artifact_path?: string | null;
  message?: string | null;
  best_raw_value?: number | null;
  /** Phase 11B additive real/synthetic provenance. */
  data_origin?: DataOrigin | null;
  data_mode?: MarketDataMode | string | null;
  /** Phase 11C target bar timeframe. */
  timeframe?: DataTimeframe | string | null;
  cached_bars?: number | CachedBarsMeta | null;
}

export interface OptimizationListResponse {
  studies: OptimizationStudy[];
}

export interface OptimizationVerificationBacktest {
  status: "completed" | "skipped" | string;
  message?: string;
  run?: { id?: string; [key: string]: unknown } | null;
}

export interface OptimizationApplyResult {
  study_id: string;
  strategy_id: string;
  /** v1.5.3 additive apply fields (task-49). */
  saved_strategy_id?: string | null;
  strategy_path?: string | null;
  strategy?: StrategyConfig | null;
  backtest_run_id?: string | null;
  applied: boolean;
  message: string;
  best_params?: Record<string, number | string | boolean>;
  verification_backtest?: OptimizationVerificationBacktest | null;
}

// ---------------------------------------------------------------------------
// DRL neural trainer (Phase 10)
// ---------------------------------------------------------------------------

export type DrlAlgorithm = "PPO" | "CPPO" | "SAC" | "TD3" | "A2C";

export interface DrlTrainRequest {
  strategy_id: string;
  algorithm: DrlAlgorithm | string;
  timesteps: number;
  learning_rate: number | null;
  gamma: number | null;
  llm_reward_penalty_lambda: number | null;
  checkpoint_name: string | null;
  seed: number | null;
  device: string;
  periods: number | null;
  offline: true;
}

export interface DrlRewardPoint {
  step: number;
  reward: number;
}

export interface DrlRewardSummary {
  window_steps: number | null;
  samples: number;
  episode_count: number;
  env_steps: number | null;
  final_mean_reward: number | null;
  best_mean_reward: number | null;
  path: string | null;
}

export interface DrlModel {
  name: string;
  algorithm: string;
  timesteps: number;
  universe: string[];
  features: string[];
  created_at: string;
  git_commit: string | null;
  size_bytes: number | null;
  reward_curve: DrlRewardPoint[];
  reward_summary: DrlRewardSummary | null;
  hyperparameters: Record<string, number | string | boolean>;
  /** Metadata fields for the `.meta.json` inspector (listing or training payload). */
  meta: Record<string, unknown>;
  path: string | null;
  meta_path?: string | null;
  metadata_available?: boolean;
  metadata_error?: string | null;
  sb3_algorithm?: string | null;
  strategy_id?: string | null;
  policy?: string | null;
  learning_rate?: number | null;
  gamma?: number | null;
  llm_reward_penalty_lambda?: number | null;
  seed?: number | null;
  device?: string | null;
  offline?: boolean | null;
}

export interface DrlModelsResponse {
  models: DrlModel[];
  count?: number | null;
  models_dir?: string | null;
  generated_at?: string | null;
}

export interface DrlTrainResult {
  model: DrlModel;
  reward_curve: DrlRewardPoint[];
  summary: Record<string, unknown>;
}

export interface DrlBindResult {
  model: string;
  strategy_id: string;
  bound: boolean;
  message: string;
  /** Phase 11A add-on: universe sync applied with the checkpoint metadata. */
  universe_synced?: boolean;
  synced_roots?: string[];
  top_n?: number | null;
  metadata_available?: boolean;
  warning?: string | null;
  strategy?: StrategyConfig | null;
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

export interface BacktestTrade {
  timestamp?: string;
  order_id?: string;
  instrument_id?: string;
  side?: "BUY" | "SELL" | string;
  quantity?: number;
  price?: number;
  notional?: number;
  commissions?: number;
  commission?: number;
  slippage?: number;
  kind?: string;
}

export interface TearsheetResponse {
  run_id: string;
  /** QoL: persisted trade ledger (additive; may be missing on legacy runs). */
  trades?: BacktestTrade[];
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
  /** Phase 9 additive walk-forward controls. */
  wf_splits?: number;
  purge_bars?: number;
  embargo_bars?: number;
  /** Phase 11B: auto = real cached bars when sufficient, else synthetic. */
  data_mode?: MarketDataMode;
  /** Phase 11C target bar timeframe. */
  timeframe?: DataTimeframe;
}

export type MarketDataMode = "auto" | "cached_real" | "synthetic_offline";
export type DataOrigin = "real_parquet" | "synthetic_fixture" | string;

export interface CachedBarsMeta {
  bars?: number | null;
  bar_count?: number | null;
  start_date?: string | null;
  end_date?: string | null;
  roots?: string[] | null;
  timeframe?: string | null;
  [key: string]: unknown;
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
  /** Phase 9/10 additive: `walk_forward` runs carry fold metadata. */
  kind?: "walk_forward" | "backtest" | "standard" | string | null;
  wf_splits?: number | null;
  /** Phase 10 additive: actual number of walk-forward folds reported by the API. */
  n_folds?: number | null;
  /** QoL: strategy config snapshot captured when the run was created. */
  strategy_config?: Record<string, unknown> | null;
  strategy_yaml?: string | null;
  /** Phase 11B additive real/synthetic provenance. */
  data_origin?: DataOrigin | null;
  data_mode?: MarketDataMode | string | null;
  /** Phase 11C target bar timeframe. */
  timeframe?: DataTimeframe | string | null;
  cached_bars?: number | CachedBarsMeta | null;
}

export interface BacktestRunDetail extends BacktestRunSummary {
  tearsheet: TearsheetResponse | null;
}

export interface BacktestListResponse {
  runs: BacktestRunSummary[];
  available_strategies: string[];
}

// ---------------------------------------------------------------------------
// Execution bridge: working orders, fills, runner status
// ---------------------------------------------------------------------------

export interface WorkingOrder {
  order_id: string;
  instrument_id: string;
  contract_symbol: string | null;
  side: "BUY" | "SELL";
  quantity: number;
  filled_quantity: number;
  remaining_quantity: number;
  order_type: string;
  limit_price: number | null;
  status: string;
  updated_at: string;
}

export interface WorkingOrdersResponse {
  as_of: string;
  source: string;
  mode?: string | null;
  stale?: boolean;
  sequence?: number;
  orders: WorkingOrder[];
}

export interface Fill {
  order_id: string;
  instrument_id: string;
  side: "BUY" | "SELL";
  quantity: number;
  price: number;
  commission: number;
  slippage_ticks: number;
  timestamp: string;
}

export interface FillsResponse {
  as_of: string;
  source: string;
  fills: Fill[];
}

export interface RunnerStatus {
  mode: string;
  available: boolean;
  stale: boolean;
  sequence: number;
  updated_at: string;
  runner_id: string;
  session_running: boolean;
  states_processed: number;
}

export interface IbkrPortInfo {
  port: number;
  mode: "paper" | "live" | string;
  gateway: "tws" | "ib_gateway" | string;
  reachable: boolean;
  latency_ms: number | null;
}

export interface IbkrActiveConnection {
  port: number | null;
  client_id: number | null;
  environment: string | null;
  mode: string | null;
  gateway: string | null;
}

export interface IbkrPortsResponse {
  host: string;
  ports: IbkrPortInfo[];
  active: IbkrActiveConnection | null;
  warnings: string[];
  probed_at: string | null;
}

export interface IbkrConnectionRequest {
  port: number;
  client_id: number | null;
  confirmation: string | null;
}

export interface IbkrConnectionResponse {
  status: "applied" | "restart_required" | "refused" | string;
  applied: boolean;
  restart_required: boolean;
  requested?: IbkrPortInfo & { client_id?: number | null };
  active?: IbkrActiveConnection | null;
  environment?: string | null;
  warnings: string[];
  message: string;
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
  /** v1.5.2 additive IBKR-sourced fields (null on offline/backtest frames). */
  portfolio_volatility?: number | null;
  initial_margin_used?: number | null;
  margin_utilization?: number | null;
  open_positions?: number;
}

export interface MarginTelemetryPayload {
  initial_margin_used: number;
  maintenance_margin_used: number;
  margin_utilization: number;
  warning_threshold: number;
  critical_threshold: number;
  /** v1.5.2 additive IBKR-sourced fields (absent on simulated frames). */
  portfolio_volatility?: number | null;
  open_positions?: number;
}

export interface PositionsTelemetryPayload {
  positions: PositionRow[];
  gross_leverage: number;
  net_exposure: number;
  /** v1.5.2 additive IBKR-sourced fields (optional; may be absent offline). */
  portfolio_volatility?: number | null;
  open_positions?: number;
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
  /** Phase 11C: raw HTTP status / FastAPI `detail` for exact error rendering. */
  status?: number | null;
  detail?: unknown;
  detailCode?: string | null;
}


// ---------------------------------------------------------------------------
// QoL artifact management / operation contracts
// ---------------------------------------------------------------------------

export interface ArtifactMutationResult {
  ok: boolean;
  message: string;
  status?: string | null;
  receipts?: Record<string, unknown> | unknown[] | null;
}

export interface StrategyMutationResult {
  ok: boolean;
  strategy: StrategyConfig | null;
  message: string;
}

export interface RiskLockResetResult {
  ok: boolean;
  message: string;
  kill_switch_active?: boolean | null;
  state?: RiskStateResponse | null;
}

export interface PaperStateResetResult {
  ok: boolean;
  message: string;
  receipts?: Record<string, unknown> | unknown[] | null;
}

export interface BacktestExportFormat {
  kind: "json" | "trades-csv";
  filename: string;
}


// ---------------------------------------------------------------------------
// Phase 11A — Rebalance Book preview / execute
// ---------------------------------------------------------------------------

export interface RebalancePosition {
  symbol: string;
  quantity: number;
}

export type RebalanceOrderSide = "BUY" | "SELL" | string;

export interface RebalanceOrderPreview {
  symbol: string;
  side: RebalanceOrderSide;
  quantity: number;
  order_type?: string | null;
  est_price?: number | null;
  est_margin_impact?: number | null;
  pre_trade_gate_passed?: boolean | null;
  rejection_reason?: string | null;
}

export interface RebalanceGateSummary {
  passed: boolean;
  reasons: string[];
}

export interface RebalancePreviewResponse {
  strategy_id: string;
  account_nlv: number;
  account_currency: string;
  current_positions: RebalancePosition[];
  target_contracts: RebalancePosition[];
  proposed_orders: RebalanceOrderPreview[];
  projected_margin_utilization: number;
  margin_utilization_cap?: number | null;
  tracking_error_pct: number | null;
  mode: "ibkr-paper" | "offline" | string;
  message?: string | null;
  warnings?: string[];
  strategy_roots?: string[];
  generated_at?: string | null;
  gate?: RebalanceGateSummary | null;
}

export interface RebalanceFill {
  order_id?: string | null;
  symbol?: string | null;
  side?: string | null;
  quantity?: number | null;
  price?: number | null;
  status?: string | null;
  message?: string | null;
}

export interface RebalanceExecuteResponse {
  strategy_id: string;
  applied: boolean;
  mode: "ibkr-paper" | "offline" | string;
  status?: string | null;
  submitted_order_ids: string[];
  submitted_orders: RebalanceOrderPreview[];
  rejected_orders?: RebalanceOrderPreview[];
  fills: RebalanceFill[];
  errors: string[];
  account_nlv?: number | null;
  account_currency?: string | null;
  projected_margin_utilization?: number | null;
  margin_utilization_cap?: number | null;
  tracking_error_pct?: number | null;
  updated_positions?: RebalancePosition[];
  bridge_updated?: boolean | null;
  message?: string | null;
}


// ---------------------------------------------------------------------------
// Phase 11B — market data cache inventory / cost / ingest
// ---------------------------------------------------------------------------

export interface DataCacheRoot {
  root: string;
  sector: string;
  schema: string;
  cached_bars: number;
  start_date: string | null;
  end_date: string | null;
  size_bytes: number;
  source: "databento" | "ibkr" | "empty" | string;
}

export interface DataInventoryTotals {
  cached_roots: number;
  total_roots: number;
  total_bars: number;
  total_size_bytes: number;
  start_date: string | null;
  end_date: string | null;
}

export interface DataInventoryResponse {
  roots: DataCacheRoot[];
  timeframe?: DataTimeframe | string | null;
  base_schema?: string | null;
  totals: DataInventoryTotals;
  databento_configured: boolean;
  ibkr_available: boolean;
  schema: string;
  generated_at: string | null;
  warnings?: string[];
}

export type DataSource = "ibkr" | "databento";

export type DataTimeframe = "1m" | "5m" | "15m" | "30m" | "1h" | "4h" | "1d" | "1w";

export interface DataCostRequest {
  source: DataSource;
  roots: string[];
  start: string;
  end: string;
  schema?: string;
  timeframe?: DataTimeframe;
}

export interface DataCostEstimate {
  timeframe?: DataTimeframe | string | null;
  base_schema?: string | null;
  estimated_cost_usd: number;
  billable_size_bytes: number;
  billable_size_mb: number;
  source: DataSource | "unavailable" | string;
  currency: string;
  details?: Record<string, unknown> | null;
}

export interface DataIngestRequest extends DataCostRequest {
  allow_over_budget?: boolean;
  confirm?: boolean;
}

export interface DataIngestResult {
  ok: boolean;
  timeframe?: DataTimeframe | string | null;
  base_schema?: string | null;
  resampled?: boolean | null;
  source: DataSource | string;
  roots_ingested: string[];
  bars_written: number;
  start_date: string | null;
  end_date: string | null;
  duration_seconds: number | null;
  cost_estimate: DataCostEstimate | null;
  continuous_series: unknown[] | null;
  warnings: string[];
  message: string;
}
