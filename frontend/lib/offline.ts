/**
 * Honest, empty payloads used only when the FastAPI backend is unreachable.
 *
 * These contain no synthetic prices, PnL or positions: arrays are empty and
 * nullable metrics are null. Every API call still returns `source: "demo"` and
 * an error so the shell can show the offline/error banner.
 */

import type {
  AccountSnapshotResponse,
  BacktestListResponse,
  DrlBindResult,
  DrlModel,
  DrlModelsResponse,
  DrlTrainRequest,
  DrlTrainResult,
  FillsResponse,
  HealthResponse,
  InstrumentsResponse,
  OptimizationApplyResult,
  OptimizationListResponse,
  OptimizationRequest,
  OptimizationStudy,
  PipelineWaterfallResponse,
  PositionRow,
  RiskStateResponse,
  RiskThresholds,
  RunnerStatus,
  StrategyConfig,
  StudioCatalogResponse,
  TearsheetResponse,
  TelemetryStatusResponse,
  TermStructureResponse,
  WorkingOrdersResponse,
} from "./types";

const now = () => new Date().toISOString();

export function emptyPositions(_netLiquidation?: number): PositionRow[] {
  return [];
}

export function emptyAccountSnapshot(): AccountSnapshotResponse {
  return {
    timestamp: now(),
    net_liquidation: 0,
    cash_balance: 0,
    buying_power: 0,
    initial_margin_used: 0,
    maintenance_margin_used: 0,
    excess_liquidity: 0,
    margin_utilization: 0,
    day_pnl: 0,
    day_pnl_pct: 0,
    gross_leverage: 0,
    net_exposure: 0,
    equity_curve: [],
    positions: [],
    source: "engine",
    stale: true,
    stale_reason: "API unavailable",
    equity_curve_source: "fixture",
    equity_curve_live_points: 0,
    equity_curve_live_days: 0,
    equity_curve_intraday: [],
    portfolio_volatility: null,
    portfolio_volatility_source: "unavailable",
  };
}

export function emptyPipeline(): PipelineWaterfallResponse {
  return {
    timestamp: now(),
    cycle_id: "unavailable",
    universe: [],
    stages: [],
    instruments: [],
    diagnostics: [],
    elapsed_ms: 0,
    data_source: "fixture",
    data_origin: "fixture",
    selected_roots: [],
  };
}

export function emptyThresholds(): RiskThresholds {
  return {
    llm_confidence_threshold: 0.7,
    llm_safe_score: 1.5,
    llm_critical_score: 4.5,
    max_margin_utilization: 0.6,
    margin_warning_threshold: 0.6,
    margin_critical_threshold: 0.8,
    daily_max_loss_pct: 0.03,
    target_annual_volatility: 0.12,
    macro_lockout_minutes_before: 30,
    macro_lockout_minutes_after: 30,
  };
}

export function emptyRiskState(): RiskStateResponse {
  return {
    timestamp: now(),
    thresholds: emptyThresholds(),
    circuit_breaker_active: false,
    circuit_breaker_reason: "",
    macro_lockout: { active: false, event: null, starts_at: null, ends_at: null, minutes_remaining: 0 },
    assessments: [],
    macro_events: [],
    news: [],
    neutralized_instruments: [],
    scope: "strategy",
    sector: null,
    selected_roots: [],
    derived_roots: [],
    assessment_source: "fixture",
    data_origin: "fixture",
  };
}

export function emptyTermStructure(): TermStructureResponse {
  return {
    as_of: now(),
    curves: [],
    roll_calendar: [],
    scope: "strategy",
    sector: null,
    selected_roots: [],
    data_origin: "fixture",
  };
}

export function emptyStrategies(): StrategyConfig[] {
  return [];
}

export function emptyBacktestList(): BacktestListResponse {
  return { runs: [], available_strategies: [] };
}

export function emptyTearsheet(_runId?: string): TearsheetResponse | null {
  return null;
}

export function emptyHealth(): HealthResponse {
  return {
    status: "degraded",
    app_name: "FuturesRL",
    version: "unavailable",
    environment: "backtest",
    offline_mode: true,
    uptime_seconds: 0,
    server_time: now(),
  };
}

export function emptyWorkingOrders(): WorkingOrdersResponse {
  return { as_of: now(), source: "unavailable", stale: true, sequence: 0, orders: [] };
}

export function emptyRecentFills(): FillsResponse {
  return { as_of: now(), source: "unavailable", fills: [] };
}

export function emptyRunnerStatus(_advance = 0): RunnerStatus {
  return {
    mode: "unavailable",
    available: false,
    stale: true,
    sequence: 0,
    updated_at: now(),
    runner_id: "",
    session_running: false,
    states_processed: 0,
  };
}

export function emptyTelemetryStatus(): TelemetryStatusResponse {
  return {
    stream_active: false,
    connected_clients: 0,
    published_events: 0,
    publisher_interval_seconds: 0,
    topics: [],
    server_time: now(),
  };
}

export function emptyStudioCatalog(): StudioCatalogResponse {
  return {
    version: null,
    roots: [],
    parameters: [],
    presets: [],
    source: "unavailable",
    generated_at: now(),
  };
}

export function emptySavedStrategy(payload: Record<string, unknown>): StrategyConfig {
  return {
    ...(payload as unknown as StrategyConfig),
    updated_at: now(),
  };
}

export function emptyOptimizationList(): OptimizationListResponse {
  return { studies: [] };
}

export function emptyOptimizationStudy(request?: Partial<OptimizationRequest>): OptimizationStudy {
  return {
    study_id: "unavailable",
    base_strategy_id: request?.base_strategy_id ?? "",
    objective_metric: request?.objective_metric ?? "sharpe",
    n_trials: 0,
    walk_forward: request?.walk_forward ?? true,
    wf_splits: request?.wf_splits ?? 5,
    status: "failed",
    created_at: now(),
    duration_ms: 0,
    best_value: null,
    best_trial_number: null,
    best_params: {},
    param_importances: [],
    trials: [],
    message: "API unavailable",
  };
}

export function emptyOptimizationApply(studyId: string): OptimizationApplyResult {
  return {
    study_id: studyId,
    strategy_id: "",
    saved_strategy_id: null,
    strategy_path: null,
    strategy: null,
    backtest_run_id: null,
    applied: false,
    message: "API unavailable — strategy was not saved.",
    verification_backtest: { status: "skipped", message: "API unavailable" },
  };
}

export function emptyDrlModels(): DrlModelsResponse {
  return { models: [] };
}

export function emptyDrlTrain(request: DrlTrainRequest): DrlTrainResult {
  const model: DrlModel = {
    name: request.checkpoint_name ?? "unavailable",
    algorithm: request.algorithm,
    timesteps: 0,
    universe: [],
    features: [],
    created_at: now(),
    git_commit: null,
    size_bytes: null,
    reward_curve: [],
    reward_summary: null,
    hyperparameters: {},
    meta: {},
    path: null,
  };
  return { model, reward_curve: [], summary: {} };
}

export function emptyDrlBind(model: DrlModel, strategyId: string): DrlBindResult {
  return {
    model: model.name,
    strategy_id: strategyId,
    bound: false,
    message: "API unavailable — model was not bound.",
  };
}

export function emptyInstruments(): InstrumentsResponse {
  return { generated_at: now(), count: 0, sectors: [], roots: [] };
}
