/**
 * Deterministic demo dataset.
 *
 * The dashboard must be fully explorable with the FastAPI backend switched off
 * (`npm run dev` standalone, static preview, screenshots, CI). When a REST call
 * fails the API client falls back to the factories below and the shell lights
 * the `UI FALLBACK` badge. Everything here is seeded (mulberry32) so two
 * users see identical numbers, and every value is internally consistent
 * (positions sum to the reported NLV, metrics are computed from the generated
 * equity curve, pipeline targets are derived from the final weights).
 */

import type {
  AccountSnapshotResponse,
  AttributionRow,
  BacktestListResponse,
  BacktestRunConfig,
  BacktestRunSummary,
  ConnectionState,
  EquityPoint,
  HealthResponse,
  MacroEvent,
  MacroLockout,
  MetricsSet,
  NewsEvent,
  PipelineDiagnostic,
  PipelineInstrumentRow,
  PipelineStage,
  PipelineWaterfallResponse,
  PositionRow,
  RiskAssessmentRow,
  RiskStateResponse,
  RiskThresholds,
  RollCalendarEntry,
  SeriesPoint,
  StrategyConfig,
  TailPoint,
  TearsheetResponse,
  TelemetryEnvelope,
  TermContract,
  TermCurve,
  TermStructureResponse,
  FillsResponse,
  RunnerStatus,
  WorkingOrdersResponse,
  DrlModel,
  DrlModelsResponse,
  DrlRewardPoint,
  DrlTrainRequest,
  DrlTrainResult,
  OptimizationApplyResult,
  OptimizationListResponse,
  OptimizationRequest,
  OptimizationStudy,
  OptimizationTrial,
  StageConfig,
  StudioCatalogResponse,
  StudioParameter,
  StudioPreset,
  StudioTranslatorConfig,
  StudioParamValue,
  InstrumentRoot,
  InstrumentSector,
  InstrumentsResponse,
} from "./types";

// ---------------------------------------------------------------------------
// Deterministic RNG + time helpers
// ---------------------------------------------------------------------------

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4_294_967_296;
  };
}

/** Standard normal via Box-Muller (deterministic given `rng`). */
function gaussian(rng: () => number): number {
  const u = Math.max(rng(), 1e-9);
  const v = Math.max(rng(), 1e-9);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

const MS_PER_DAY = 86_400_000;

function nowMs(): number {
  return Date.now();
}

function isoFromMs(ms: number): string {
  return new Date(ms).toISOString();
}

function daysAheadIso(days: number, reference = nowMs()): string {
  return isoFromMs(reference + days * MS_PER_DAY);
}

function daysAgoIso(days: number, reference = nowMs()): string {
  return isoFromMs(reference - days * MS_PER_DAY);
}

function dateOnly(iso: string): string {
  return iso.slice(0, 10);
}

// ---------------------------------------------------------------------------
// Reference portfolio
// ---------------------------------------------------------------------------

export const ACCOUNT_STARTING_NLV = 1_250_000;

interface RawPosition {
  instrument_id: string;
  asset_class: PositionRow["asset_class"];
  contract_symbol: string;
  quantity: number;
  average_price: number;
  market_price: number;
  multiplier: number;
  initial_margin: number;
  maintenance_margin: number;
  days_to_roll: number;
  expiration: string;
}

const RAW_POSITIONS: RawPosition[] = [
  {
    instrument_id: "ES",
    asset_class: "equity_index",
    contract_symbol: "ESZ5",
    quantity: 14,
    average_price: 5241.25,
    market_price: 5280.5,
    multiplier: 50,
    initial_margin: 15_000,
    maintenance_margin: 13_600,
    days_to_roll: 12,
    expiration: daysAheadIso(24),
  },
  {
    instrument_id: "NQ",
    asset_class: "equity_index",
    contract_symbol: "NQZ5",
    quantity: 6,
    average_price: 18_312,
    market_price: 18_455.75,
    multiplier: 20,
    initial_margin: 21_000,
    maintenance_margin: 19_100,
    days_to_roll: 12,
    expiration: daysAheadIso(24),
  },
  {
    instrument_id: "CL",
    asset_class: "energy",
    contract_symbol: "CLF6",
    quantity: -8,
    average_price: 78.42,
    market_price: 77.95,
    multiplier: 1_000,
    initial_margin: 6_200,
    maintenance_margin: 5_600,
    days_to_roll: 5,
    expiration: daysAheadIso(17),
  },
  {
    instrument_id: "GC",
    asset_class: "metals",
    contract_symbol: "GCG6",
    quantity: 3,
    average_price: 2_351.4,
    market_price: 2_378.2,
    multiplier: 100,
    initial_margin: 11_000,
    maintenance_margin: 10_000,
    days_to_roll: 21,
    expiration: daysAheadIso(53),
  },
  {
    instrument_id: "ZN",
    asset_class: "rates",
    contract_symbol: "ZNH6",
    quantity: -12,
    average_price: 110.84,
    market_price: 110.92,
    multiplier: 1_000,
    initial_margin: 1_800,
    maintenance_margin: 1_620,
    days_to_roll: 38,
    expiration: daysAheadIso(70),
  },
  {
    instrument_id: "MES",
    asset_class: "equity_index",
    contract_symbol: "MESZ5",
    quantity: 18,
    average_price: 5_266.25,
    market_price: 5_280.5,
    multiplier: 5,
    initial_margin: 1_500,
    maintenance_margin: 1_360,
    days_to_roll: 12,
    expiration: daysAheadIso(24),
  },
];

export function buildPositions(netLiquidation: number): PositionRow[] {
  return RAW_POSITIONS.map((p) => {
    const notional = p.quantity * p.market_price * p.multiplier;
    return {
      instrument_id: p.instrument_id,
      root_symbol: p.instrument_id,
      asset_class: p.asset_class,
      contract_symbol: p.contract_symbol,
      expiration: p.expiration,
      days_to_roll: p.days_to_roll,
      quantity: p.quantity,
      average_price: p.average_price,
      market_price: p.market_price,
      multiplier: p.multiplier,
      notional,
      weight: notional / netLiquidation,
      unrealized_pnl: p.quantity * (p.market_price - p.average_price) * p.multiplier,
      initial_margin: p.initial_margin,
      maintenance_margin: p.maintenance_margin,
    };
  });
}

const RAW_IM_TOTAL = RAW_POSITIONS.reduce(
  (sum, p) => sum + Math.abs(p.quantity) * p.initial_margin,
  0,
);

export function buildAccount(netLiquidation = ACCOUNT_STARTING_NLV): AccountSnapshotResponse {
  const positions = buildPositions(netLiquidation);
  const initialMargin = RAW_IM_TOTAL;
  const unrealized = positions.reduce((sum, p) => sum + p.unrealized_pnl, 0);
  const dayPnl = Math.round(unrealized * 0.28 + 4_180);
  const grossNotional = positions.reduce((sum, p) => sum + Math.abs(p.notional), 0);
  const netExposure = positions.reduce((sum, p) => sum + p.notional, 0);
  return {
    timestamp: isoFromMs(nowMs()),
    net_liquidation: netLiquidation,
    cash_balance: netLiquidation - unrealized,
    buying_power: netLiquidation * 2.1,
    initial_margin_used: initialMargin,
    maintenance_margin_used: initialMargin * 0.905,
    excess_liquidity: netLiquidation - initialMargin,
    margin_utilization: initialMargin / netLiquidation,
    day_pnl: dayPnl,
    day_pnl_pct: dayPnl / netLiquidation,
    gross_leverage: grossNotional / netLiquidation,
    net_exposure: netExposure,
    equity_curve: generateEquityCurve(20260101, 0.0082, 0.00055, netLiquidation).points,
    positions,
    equity_curve_source: "fixture",
    equity_curve_live_points: 0,
    portfolio_volatility: null,
    portfolio_volatility_source: "unavailable",
  };
}

// ---------------------------------------------------------------------------
// Equity curve + derived performance series
// ---------------------------------------------------------------------------

const TRADING_DAYS = 252;

/** Last `total` weekdays ending today (UTC), ascending — unique, gap-free x-axis. */
function tradingCalendar(total: number, reference = nowMs()): string[] {
  const days: number[] = [];
  const cursor = new Date(reference);
  while (days.length < total) {
    const weekday = cursor.getUTCDay();
    if (weekday !== 0 && weekday !== 6) days.push(cursor.getTime());
    cursor.setUTCDate(cursor.getUTCDate() - 1);
  }
  return days.reverse().map((ms) => isoFromMs(ms));
}

export interface GeneratedCurve {
  points: EquityPoint[];
  returns: number[];
}

export function generateEquityCurve(
  seed = 20260101,
  dailyVol = 0.0082,
  drift = 0.00055,
  finalValue = ACCOUNT_STARTING_NLV,
  length = TRADING_DAYS,
): GeneratedCurve {
  const rng = mulberry32(seed);
  const calendar = tradingCalendar(length);
  const raw: number[] = [];
  let level = 1;
  for (let i = 0; i < length; i += 1) {
    const shock = gaussian(rng) * dailyVol;
    // Occasional risk-off cluster to make the underwater chart meaningful.
    const regime = i > length * 0.42 && i < length * 0.55 ? -0.0012 : 0;
    level *= 1 + drift + shock + regime;
    raw.push(level);
  }
  const scale = finalValue / (raw[raw.length - 1] ?? 1);
  const returns: number[] = [];
  for (let i = 0; i < length; i += 1) {
    const previous = i === 0 ? raw[0]! / (1 + drift) : raw[i - 1]!;
    returns.push(raw[i]! / previous - 1);
  }
  return {
    points: raw.map((value, index) => ({
      time: calendar[index] ?? isoFromMs(nowMs()),
      value: Math.round(value * scale * 100) / 100,
    })),
    returns,
  };
}

function mean(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function stdev(values: number[]): number {
  if (values.length < 2) return 0;
  const m = mean(values);
  return Math.sqrt(values.reduce((sum, value) => sum + (value - m) ** 2, 0) / (values.length - 1));
}

function quantile(sorted: number[], q: number): number {
  if (sorted.length === 0) return 0;
  const position = (sorted.length - 1) * q;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return sorted[lower] ?? 0;
  const weight = position - lower;
  return (sorted[lower] ?? 0) * (1 - weight) + (sorted[upper] ?? 0) * weight;
}

export function computeMetrics(points: EquityPoint[], returns: number[]): MetricsSet {
  const safeReturns = returns.filter((value) => Number.isFinite(value));
  const first = points[0]?.value ?? 1;
  const last = points[points.length - 1]?.value ?? 1;
  const years = Math.max(safeReturns.length / TRADING_DAYS, 1 / TRADING_DAYS);
  const cagr = Math.pow(last / first, 1 / years) - 1;
  const annualVol = stdev(safeReturns) * Math.sqrt(TRADING_DAYS);
  const sharpe = annualVol > 0 ? (cagr - 0.042) / annualVol : 0;
  const downside = safeReturns.filter((value) => value < 0);
  const downsideVol = stdev(downside) * Math.sqrt(TRADING_DAYS);
  const sortino = downsideVol > 0 ? (cagr - 0.042) / downsideVol : 0;

  let peak = points[0]?.value ?? 1;
  let peakIndex = 0;
  let maxDrawdown = 0;
  let maxDrawdownDuration = 0;
  let currentDuration = 0;
  points.forEach((point, index) => {
    if (point.value >= peak) {
      peak = point.value;
      peakIndex = index;
      currentDuration = 0;
    } else {
      currentDuration = index - peakIndex;
      maxDrawdownDuration = Math.max(maxDrawdownDuration, currentDuration);
    }
    const drawdown = peak > 0 ? (peak - point.value) / peak : 0;
    maxDrawdown = Math.max(maxDrawdown, drawdown);
  });

  const sorted = [...safeReturns].sort((a, b) => a - b);
  const var95 = -quantile(sorted, 0.05);
  const var99 = -quantile(sorted, 0.01);
  const tail95 = sorted.slice(0, Math.max(1, Math.floor(sorted.length * 0.05)));
  const cvar95 = -mean(tail95);
  const tail99 = sorted.slice(0, Math.max(1, Math.floor(sorted.length * 0.01)));
  const cvar99 = -mean(tail99);
  const gains = safeReturns.filter((value) => value > 0).sort((a, b) => b - a);
  const gainTail = mean(gains.slice(0, Math.max(1, Math.floor(gains.length * 0.05))));
  const rachev = cvar95 > 0 ? gainTail / cvar95 : 0;
  const omega = cvar95 > 0 ? gainTail / cvar95 : 0;
  const totalReturn = last / first - 1;

  return {
    cagr,
    annual_volatility: annualVol,
    sharpe,
    sortino,
    calmar: maxDrawdown > 0 ? cagr / maxDrawdown : 0,
    omega,
    max_drawdown: maxDrawdown,
    max_drawdown_duration_days: Math.round((maxDrawdownDuration / TRADING_DAYS) * 365),
    var_95: var95,
    var_99: var99,
    cvar_95: cvar95,
    cvar_99: cvar99,
    rachev_ratio: rachev,
    rom: Math.max(0.18, totalReturn / 0.42),
    avg_margin_utilization: 0.38 + Math.min(0.1, Math.abs(annualVol) * 0.2),
    peak_margin_utilization: 0.54 + Math.min(0.12, Math.abs(annualVol) * 0.25),
    turnover: 1.42 + annualVol * 4,
    implementation_shortfall_ticks: 0.38,
    commission_to_gross_pnl: 0.061,
  };
}

function buildRollingSharpe(returns: number[], window = 63): SeriesPoint[] {
  const calendar = tradingCalendar(returns.length);
  const output: SeriesPoint[] = [];
  for (let i = window; i < returns.length; i += 1) {
    const slice = returns.slice(i - window, i);
    const vol = stdev(slice) * Math.sqrt(TRADING_DAYS);
    const annualReturn = mean(slice) * TRADING_DAYS;
    output.push({
      time: calendar[i] ?? isoFromMs(nowMs()),
      value: vol > 0 ? (annualReturn - 0.042) / vol : 0,
    });
  }
  return output;
}

function buildUnderwater(points: EquityPoint[]): SeriesPoint[] {
  let peak = points[0]?.value ?? 1;
  return points.map((point) => {
    peak = Math.max(peak, point.value);
    return { time: point.time, value: peak > 0 ? point.value / peak - 1 : 0 };
  });
}

function buildTailDistribution(returns: number[]): TailPoint[] {
  const sorted = [...returns].sort((a, b) => a - b);
  const take = Math.min(30, Math.floor(sorted.length * 0.12));
  const losses = sorted.slice(0, take).map((value, index) => ({
    rank: index + 1,
    return_pct: value,
    tail: "loss" as const,
  }));
  const gains = sorted
    .slice(-take)
    .reverse()
    .map((value, index) => ({ rank: index + 1, return_pct: value, tail: "gain" as const }));
  return [...losses, ...gains];
}

function buildMonthlyReturns(points: EquityPoint[]): { month: string; value: number }[] {
  const byMonth = new Map<string, { first: number; last: number }>();
  points.forEach((point) => {
    const month = point.time.slice(0, 7);
    const bucket = byMonth.get(month);
    if (bucket) bucket.last = point.value;
    else byMonth.set(month, { first: point.value, last: point.value });
  });
  return [...byMonth.entries()].map(([month, bucket]) => ({
    month,
    value: bucket.first > 0 ? bucket.last / bucket.first - 1 : 0,
  }));
}

const ROLL_ATTRIBUTION_ROOTS = ["ES", "NQ", "CL", "GC", "ZN"] as const;

function buildAttribution(seed: number): AttributionRow[] {
  const rng = mulberry32(seed);
  return ROLL_ATTRIBUTION_ROOTS.map((root) => {
    const spot = 40_000 + rng() * 120_000;
    const roll = (rng() - 0.45) * 30_000;
    return {
      root,
      spot_pnl: Math.round(spot),
      roll_pnl: Math.round(roll),
    };
  });
}

// ---------------------------------------------------------------------------
// Pipeline waterfall
// ---------------------------------------------------------------------------

interface PipelineInput {
  root: string;
  assetClass: PipelineInstrumentRow["asset_class"];
  base: number;
  timingMultiplier: number;
  riskMultiplier: number;
  price: number;
  multiplier: number;
  selected: boolean;
  reason: string;
}

const PIPELINE_INPUTS: PipelineInput[] = [
  {
    root: "ES",
    assetClass: "equity_index",
    base: 0.26,
    timingMultiplier: 1.08,
    riskMultiplier: 0.94,
    price: 5_280.5,
    multiplier: 50,
    selected: true,
    reason: "TSMOM 126d positive; liquid front month",
  },
  {
    root: "NQ",
    assetClass: "equity_index",
    base: 0.18,
    timingMultiplier: 0.96,
    riskMultiplier: 0.9,
    price: 18_455.75,
    multiplier: 20,
    selected: true,
    reason: "Equity cluster cap applied after risk overlay",
  },
  {
    root: "CL",
    assetClass: "energy",
    base: -0.12,
    timingMultiplier: 0.78,
    riskMultiplier: 0.62,
    price: 77.95,
    multiplier: 1_000,
    selected: true,
    reason: "LLM semantic risk 3.4 → linear circuit-breaker scale",
  },
  {
    root: "GC",
    assetClass: "metals",
    base: 0.09,
    timingMultiplier: 1.14,
    riskMultiplier: 1.0,
    price: 2_378.2,
    multiplier: 100,
    selected: true,
    reason: "Safe-haven carry positive; LLM signal neutralised (c < τ)",
  },
  {
    root: "ZN",
    assetClass: "rates",
    base: -0.06,
    timingMultiplier: 1.02,
    riskMultiplier: 0.88,
    price: 110.92,
    multiplier: 1_000,
    selected: true,
    reason: "Rates vol targeting trims duration beta",
  },
  {
    root: "6E",
    assetClass: "fx",
    base: 0.0,
    timingMultiplier: 0.0,
    riskMultiplier: 0.0,
    price: 1.095,
    multiplier: 125_000,
    selected: false,
    reason: "Filtered by liquidity/OI universe selector",
  },
];

export function mockPipeline(netLiquidation = ACCOUNT_STARTING_NLV): PipelineWaterfallResponse {
  const instruments: PipelineInstrumentRow[] = PIPELINE_INPUTS.map((input) => {
    const timing = input.selected ? input.base * input.timingMultiplier : 0;
    const risk = timing * input.riskMultiplier;
    const targetContracts = input.selected
      ? Math.round((risk * netLiquidation) / (input.price * input.multiplier))
      : null;
    return {
      instrument_id: input.root,
      asset_class: input.assetClass,
      selected: input.selected,
      weight_base: input.base,
      weight_timing: timing,
      weight_risk: risk,
      weight_final: risk,
      delta_timing: timing - input.base,
      delta_risk: risk - timing,
      target_contracts: targetContracts,
      notional: risk * netLiquidation,
      // Fixture preview: assume the executed book matches the approved target
      // to 3 decimal places (real backends report the fill-confirmed weight).
      weight_actual: input.selected ? Number(risk.toFixed(3)) : null,
      reason: input.reason,
    };
  });

  const selected = instruments.filter((row) => row.selected);
  const timestamp = isoFromMs(nowMs());
  const sumAbs = (key: keyof PipelineInstrumentRow) =>
    selected.reduce((sum, row) => sum + Math.abs(Number(row[key] ?? 0)), 0);
  const sumNet = (key: keyof PipelineInstrumentRow) =>
    selected.reduce((sum, row) => sum + Number(row[key] ?? 0), 0);
  const contractCount = selected.reduce((sum, row) => sum + Math.abs(row.target_contracts ?? 0), 0);

  const stages: PipelineStage[] = [
    {
      key: "universe",
      label: "S_t · Universe",
      formula: "S_t(X≤t) → roots",
      gross: 0,
      net: 0,
      instrument_count: selected.length,
      note: "Liquidity + open-interest + term-structure regime filter",
    },
    {
      key: "base",
      label: "w_base · Allocation",
      formula: "A_t(candidates)",
      gross: sumAbs("weight_base"),
      net: sumNet("weight_base"),
      instrument_count: selected.length,
      note: "Risk-parity seed portfolio, gross leverage target 1.0×",
    },
    {
      key: "timing",
      label: "w_timing · Overlay",
      formula: "T_t(w_base)",
      gross: sumAbs("weight_timing"),
      net: sumNet("weight_timing"),
      instrument_count: selected.length,
      note: "TSMOM 21/126d + KAMA + annualised carry filter",
    },
    {
      key: "risk",
      label: "w_risk · Approved",
      formula: "R_t(w_timing) → w_t",
      gross: sumAbs("weight_risk"),
      net: sumNet("weight_risk"),
      instrument_count: selected.length,
      note: "12% vol target, drawdown guard, LLM Risk-First projection",
    },
    {
      key: "targets",
      label: "N_i,t · Contracts",
      formula: "N = w_t·E_t / (P·M)",
      gross: sumAbs("weight_final"),
      net: sumNet("weight_final"),
      instrument_count: contractCount,
      note: `${contractCount} contracts after floor rounding + micro routing`,
    },
  ];

  const diagnostics: PipelineDiagnostic[] = [
    {
      stage: "Universe selector",
      detail: "6 candidate roots → 5 eligible (6E filtered: OI below 20d median)",
      value: "5 / 6",
      status: "ok",
    },
    {
      stage: "Volatility targeting",
      detail: "EWMA 20d covariance · realised 13.4% vs 12.0% target → scale 0.895",
      value: "0.895×",
      status: "warn",
    },
    {
      stage: "Risk overlays",
      detail: "applied: quant_risk, llm_risk",
      value: "2 applied",
      status: "ok",
    },
    {
      stage: "LLM confidence filter",
      detail: "GC signal neutralised (c = 0.58 < τ = 0.70) · variance 0.31 > 0.25",
      value: "1 neutralised",
      status: "info",
    },
    {
      stage: "Circuit breaker",
      detail: "CL semantic risk 3.4 ∈ (θ_safe, κ) → weight × 0.62 linear projection",
      value: "CL −38%",
      status: "warn",
    },
    {
      stage: "Margin feasibility",
      detail: "Projected IM 462k / NLV 1.25M = 37.0% ≤ η_max 60% → no scale-down",
      value: "37.0%",
      status: "ok",
    },
    {
      stage: "Contract translator",
      detail: "Nearest rounding, buffer 2 contracts, micro routing idle (capital sufficient)",
      value: `${contractCount} lots`,
      status: "ok",
    },
  ];

  return {
    timestamp,
    cycle_id: `rc-${new Date(timestamp).toISOString().slice(0, 10).replace(/-/g, "")}-041`,
    universe: selected.map((row) => row.instrument_id),
    stages,
    instruments,
    diagnostics,
    elapsed_ms: 42.7,
    tracking_error_pct: 4.21,
    micro_routing: [
      {
        family: "equity",
        standard_root: "ES",
        standard_quantity: 14,
        micro_root: "MES",
        micro_quantity: 18,
      },
    ],
  };
}

// ---------------------------------------------------------------------------
// Risk state
// ---------------------------------------------------------------------------

export function mockThresholds(): RiskThresholds {
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

interface AssessmentSeed {
  root: string;
  sentiment: number;
  semantic: number;
  confidence: number;
  summary: string;
  neutralized: boolean;
}

const ASSESSMENT_SEEDS: AssessmentSeed[] = [
  {
    root: "ES",
    sentiment: 3.7,
    semantic: 1.4,
    confidence: 0.91,
    summary: "Broad equity momentum intact; no Tier-1 event within lockout window.",
    neutralized: false,
  },
  {
    root: "NQ",
    sentiment: 3.2,
    semantic: 1.9,
    confidence: 0.84,
    summary: "Long-duration tech sensitive to 10Y yields; earnings dispersion elevated.",
    neutralized: false,
  },
  {
    root: "CL",
    sentiment: 1.9,
    semantic: 3.4,
    confidence: 0.88,
    summary: "EIA inventory surprise plus OPEC+ headline risk; cautious supply shock.",
    neutralized: false,
  },
  {
    root: "GC",
    sentiment: 3.4,
    semantic: 2.1,
    confidence: 0.58,
    summary: "Safe-haven bid unclear across samples; sample variance above threshold.",
    neutralized: true,
  },
  {
    root: "ZN",
    sentiment: 2.6,
    semantic: 1.7,
    confidence: 0.86,
    summary: "Duration carry stable; CPI print in lockout horizon caps leverage.",
    neutralized: false,
  },
];

export function mockRiskState(): RiskStateResponse {
  const thresholds = mockThresholds();
  const reference = nowMs();
  const assessments: RiskAssessmentRow[] = ASSESSMENT_SEEDS.map((seed) => {
    const neutralized = seed.neutralized;
    const score = neutralized ? thresholds.llm_safe_score : seed.semantic;
    let action: RiskAssessmentRow["action"] = "pass";
    let multiplier = 1;
    if (neutralized) {
      action = "neutralized";
      multiplier = 1;
    } else if (score >= thresholds.llm_critical_score && seed.confidence >= thresholds.llm_confidence_threshold) {
      action = "liquidate";
      multiplier = 0;
    } else if (score > thresholds.llm_safe_score) {
      action = "scale";
      multiplier = (thresholds.llm_critical_score - score) / (thresholds.llm_critical_score - thresholds.llm_safe_score);
    }
    return {
      instrument_id: seed.root,
      timestamp: isoFromMs(reference - 45_000),
      sentiment_score: seed.sentiment,
      semantic_risk_score: neutralized ? thresholds.llm_safe_score : seed.semantic,
      confidence: seed.confidence,
      reasoning_summary: neutralized
        ? "Neutralised by Module 1 confidence/variance filter → baseline θ_safe."
        : seed.summary,
      source: "llm-risk-first",
      action,
      weight_multiplier: Number(multiplier.toFixed(4)),
    };
  });

  const macroEvents: MacroEvent[] = [
    {
      id: "macro-cpi",
      event: "US CPI (YoY)",
      event_time: isoFromMs(reference + 42 * 60_000),
      tier: 1,
      minutes_until: 42,
      roots: ["ES", "NQ", "ZN", "GC"],
    },
    {
      id: "macro-eia",
      event: "EIA Crude Inventory",
      event_time: isoFromMs(reference + 5 * 3_600_000 + 18 * 60_000),
      tier: 1,
      minutes_until: 318,
      roots: ["CL"],
    },
    {
      id: "macro-nfp",
      event: "Non-Farm Payrolls",
      event_time: isoFromMs(reference + 2 * 86_400_000 + 6 * 3_600_000),
      tier: 1,
      minutes_until: 2 * 1_440 + 360,
      roots: ["ES", "NQ", "ZN"],
    },
    {
      id: "macro-fomc",
      event: "FOMC Statement + Presser",
      event_time: isoFromMs(reference + 6 * 86_400_000 + 19 * 3_600_000),
      tier: 1,
      minutes_until: 6 * 1_440 + 1_140,
      roots: ["ES", "NQ", "ZN", "GC", "CL", "6E"],
    },
    {
      id: "macro-wasde",
      event: "USDA WASDE",
      event_time: isoFromMs(reference + 9 * 86_400_000 + 17 * 3_600_000),
      tier: 2,
      minutes_until: 9 * 1_440 + 1_020,
      roots: ["ZC", "ZW"],
    },
  ];

  const macroLockout: MacroLockout = {
    active: false,
    event: null,
    starts_at: null,
    ends_at: null,
    minutes_remaining: 0,
  };

  const news: NewsEvent[] = [
    {
      id: "news-1",
      timestamp: isoFromMs(reference - 4 * 60_000),
      headline: "Core CPI consensus 3.1% — options market pricing 1.4σ move in ES",
      source: "Reuters / macro-calendar",
      roots: ["ES", "NQ", "ZN"],
      sentiment_score: 2.8,
      severity: "warn",
      summary: "Tier-1 print inside lockout horizon; leverage caps may engage 30m prior.",
    },
    {
      id: "news-2",
      timestamp: isoFromMs(reference - 17 * 60_000),
      headline: "OPEC+ delegates signal no change to production quotas",
      source: "Bloomberg",
      roots: ["CL"],
      sentiment_score: 1.9,
      severity: "critical",
      summary: "Supply headline risk; LLM semantic risk 3.4 → deterministic weight scale.",
    },
    {
      id: "news-3",
      timestamp: isoFromMs(reference - 31 * 60_000),
      headline: "10Y auction tails 2.1bp; duration complex under pressure",
      source: "Dow Jones",
      roots: ["ZN", "NQ"],
      sentiment_score: 2.4,
      severity: "info",
      summary: "Rates vol target trims exposure; carry filter unchanged.",
    },
    {
      id: "news-4",
      timestamp: isoFromMs(reference - 58 * 60_000),
      headline: "Gold ETF inflows accelerate for third consecutive session",
      source: "FT",
      roots: ["GC"],
      sentiment_score: 3.9,
      severity: "info",
      summary: "Safe-haven demand confirmed; LLM confidence below τ so signal neutralised.",
    },
    {
      id: "news-5",
      timestamp: isoFromMs(reference - 92 * 60_000),
      headline: "Fed speaker Waller: policy remains restrictive, data dependent",
      source: "CNBC",
      roots: ["ES", "ZN"],
      sentiment_score: 3.1,
      severity: "info",
      summary: "In line with base case; no semantic risk escalation.",
    },
    {
      id: "news-6",
      timestamp: isoFromMs(reference - 133 * 60_000),
      headline: "EIA crude build 4.2M bbl vs −1.1M expected",
      source: "EIA",
      roots: ["CL"],
      sentiment_score: 1.6,
      severity: "warn",
      summary: "Bearish inventory surprise; short leg sized via risk overlay.",
    },
  ];

  return {
    timestamp: isoFromMs(reference),
    thresholds,
    circuit_breaker_active: false,
    circuit_breaker_reason: "No root above κ = 4.5 with c ≥ τ = 0.70",
    macro_lockout: macroLockout,
    assessments,
    macro_events: macroEvents,
    news,
    neutralized_instruments: assessments
      .filter((row) => row.action === "neutralized")
      .map((row) => row.instrument_id),
  };
}

// ---------------------------------------------------------------------------
// Term structure
// ---------------------------------------------------------------------------

interface CurveSeed {
  root: string;
  assetClass: TermCurve["asset_class"];
  frontPrice: number;
  monthlySlope: number;
  tick: number;
  months: number[];
}

const CURVE_SEEDS: CurveSeed[] = [
  { root: "ES", assetClass: "equity_index", frontPrice: 5_280.5, monthlySlope: 0.0018, tick: 0.25, months: [1, 2, 3, 6, 9, 12] },
  { root: "NQ", assetClass: "equity_index", frontPrice: 18_455.75, monthlySlope: 0.0021, tick: 0.25, months: [1, 2, 3, 6, 9, 12] },
  { root: "CL", assetClass: "energy", frontPrice: 77.95, monthlySlope: -0.0045, tick: 0.01, months: [1, 2, 3, 6, 9, 12] },
  { root: "GC", assetClass: "metals", frontPrice: 2_378.2, monthlySlope: 0.0011, tick: 0.1, months: [1, 2, 4, 6, 8, 12] },
  { root: "ZN", assetClass: "rates", frontPrice: 110.92, monthlySlope: 0.0009, tick: 0.015625, months: [1, 2, 3, 6, 9, 12] },
];

function monthsToExpiryIso(months: number, reference: number): string {
  const date = new Date(reference);
  date.setUTCDate(date.getUTCDate() + Math.round(months * 30.44));
  return isoFromMs(date.getTime());
}

export function mockTermStructure(): TermStructureResponse {
  const reference = nowMs();
  const curves: TermCurve[] = CURVE_SEEDS.map((seed) => {
    const contracts: TermContract[] = seed.months.map((months, index) => {
      const drift = seed.frontPrice * seed.monthlySlope * months;
      const jitter = (((index * 37) % 7) - 3) * seed.tick * 0.5;
      const price = Number((seed.frontPrice + drift + jitter).toFixed(4));
      return {
        expiration: dateOnly(monthsToExpiryIso(months, reference)),
        price,
        days_to_expiry: Math.round(months * 30.44),
        is_front: index === 0,
        open_interest: Math.round(1_500_000 - index * 180_000),
      };
    });
    const front = contracts[0]!;
    const second = contracts[1]!;
    const slope = (second.price - front.price) / front.price;
    const days = Math.max(second.days_to_expiry - front.days_to_expiry, 1);
    const regime: TermCurve["regime"] = Math.abs(slope) < 0.0004 ? "flat" : slope > 0 ? "contango" : "backwardation";
    return {
      root: seed.root,
      asset_class: seed.assetClass,
      regime,
      slope,
      roll_yield_annualized: -(slope * 365) / days,
      contracts,
    };
  });

  const rollCalendarRows: RollCalendarEntry[] = [
    {
      root: "CL",
      from_expiration: dateOnly(daysAheadIso(17)),
      to_expiration: dateOnly(monthsToExpiryIso(2, reference)),
      roll_date: isoFromMs(reference + 5 * MS_PER_DAY),
      days_to_roll: 5,
      trigger: "calendar_days_before_fnd",
      method: "BAG calendar spread",
      status: "imminent",
    },
    {
      root: "ES",
      from_expiration: dateOnly(daysAheadIso(24)),
      to_expiration: dateOnly(monthsToExpiryIso(3, reference)),
      roll_date: isoFromMs(reference + 12 * MS_PER_DAY + 6 * 3_600_000),
      days_to_roll: 12,
      trigger: "calendar_days_before_fnd",
      method: "BAG calendar spread",
      status: "scheduled",
    },
    {
      root: "NQ",
      from_expiration: dateOnly(daysAheadIso(24)),
      to_expiration: dateOnly(monthsToExpiryIso(3, reference)),
      roll_date: isoFromMs(reference + 12 * MS_PER_DAY + 6 * 3_600_000),
      days_to_roll: 12,
      trigger: "volume_oi_crossover",
      method: "two-legged MKT",
      status: "scheduled",
    },
    {
      root: "GC",
      from_expiration: dateOnly(daysAheadIso(53)),
      to_expiration: dateOnly(monthsToExpiryIso(4, reference)),
      roll_date: isoFromMs(reference + 21 * MS_PER_DAY + 6 * 3_600_000),
      days_to_roll: 21,
      trigger: "calendar_days_before_fnd",
      method: "BAG calendar spread",
      status: "scheduled",
    },
    {
      root: "ZN",
      from_expiration: dateOnly(daysAheadIso(70)),
      to_expiration: dateOnly(monthsToExpiryIso(3, reference)),
      roll_date: isoFromMs(reference + 38 * MS_PER_DAY + 6 * 3_600_000),
      days_to_roll: 38,
      trigger: "volume_oi_crossover",
      method: "two-legged MKT",
      status: "scheduled",
    },
  ];
  const rollCalendar = [...rollCalendarRows].sort((a, b) => a.days_to_roll - b.days_to_roll);

  return { as_of: isoFromMs(reference), curves, roll_calendar: rollCalendar };
}

// ---------------------------------------------------------------------------
// Strategies & backtests
// ---------------------------------------------------------------------------

export function mockStrategies(): StrategyConfig[] {
  return [
    {
      id: "tsmom_vol_target",
      description: "Volatility-targeted TSMOM across equity index, rates, energy, metals and FX futures.",
      tags: ["classical", "tsmom", "vol-target"],
      universe: { name: "liquidity_oi", params: { min_open_interest: 250_000, min_adv: 50_000 } },
      allocator: { name: "risk_parity", params: { covariance_lookback_days: 60 } },
      timing: [
        { name: "tsmom", params: { fast_window: 21, slow_window: 126 } },
        { name: "kama", params: { er_window: 10, fast: 2, slow: 30 } },
        { name: "carry_filter", params: { min_carry: -0.12, max_carry: 0.35 } },
      ],
      risk: [
        { name: "vol_target", params: { target_annual_volatility: 0.12 } },
        { name: "drawdown_guard", params: { drawdown_limit_pct: 0.2 } },
        { name: "correlation_guard", params: { max_sector_weight: 0.4 } },
      ],
      updated_at: daysAgoIso(6),
    },
    {
      id: "drl_ppo_llm_risk",
      description: "PPO allocation with LLM Risk-First reward shaping and deterministic circuit breaker.",
      tags: ["drl", "ppo", "llm-risk"],
      universe: { name: "regime_selector", params: { slow_trend_weeks: 26, fast_shock_days: 3 } },
      allocator: { name: "drl_allocator", params: { algorithm: "PPO", training_steps: 200_000 } },
      timing: [{ name: "tsmom", params: { fast_window: 21, slow_window: 126 } }],
      risk: [
        { name: "llm_risk_first", params: { confidence_threshold: 0.7, critical_score: 4.5 } },
        { name: "vol_target", params: { target_annual_volatility: 0.12 } },
      ],
      updated_at: daysAgoIso(2),
    },
    {
      id: "carry_riskparity",
      description: "Roll-yield carry harvest with risk-parity sizing and macro lockout flat.",
      tags: ["carry", "risk-parity", "macro"],
      universe: { name: "liquidity_oi", params: { min_open_interest: 500_000, min_adv: 75_000 } },
      allocator: { name: "mean_variance", params: { risk_aversion: 4.0 } },
      timing: [{ name: "carry_filter", params: { min_carry: 0.0, max_carry: 0.5 } }],
      risk: [
        { name: "vol_target", params: { target_annual_volatility: 0.1 } },
        { name: "macro_lockout", params: { flatten_minutes_before: 30 } },
      ],
      updated_at: daysAgoIso(11),
    },
  ];
}

function runConfig(strategyId: string, offsetDays: number): BacktestRunConfig {
  return {
    strategy_id: strategyId,
    start: dateOnly(daysAgoIso(offsetDays + 760)),
    end: dateOnly(daysAgoIso(offsetDays)),
    initial_equity: 1_000_000,
    rebalance_frequency: "W-FRI",
    walk_forward: true,
    random_seed: 7,
  };
}

const RUN_SEEDS: {
  id: string;
  strategyId: string;
  seed: number;
  vol: number;
  createdDaysAgo: number;
  duration: number;
  walkForward: boolean;
}[] = [
  { id: "bt-0031", strategyId: "tsmom_vol_target", seed: 20260101, vol: 0.0082, createdDaysAgo: 1, duration: 184.2, walkForward: true },
  { id: "bt-0030", strategyId: "drl_ppo_llm_risk", seed: 20260102, vol: 0.0068, createdDaysAgo: 3, duration: 512.7, walkForward: true },
  { id: "bt-0029", strategyId: "carry_riskparity", seed: 20260103, vol: 0.0096, createdDaysAgo: 7, duration: 121.5, walkForward: false },
];

export function mockBacktestList(): BacktestListResponse {
  const runs: BacktestRunSummary[] = RUN_SEEDS.map((seed) => {
    const curve = generateEquityCurve(seed.seed, seed.vol);
    const metrics = computeMetrics(curve.points, curve.returns);
    return {
      id: seed.id,
      strategy_id: seed.strategyId,
      status: "completed",
      created_at: daysAgoIso(seed.createdDaysAgo),
      duration_seconds: seed.duration,
      config: runConfig(seed.strategyId, seed.createdDaysAgo),
      metrics,
      error: null,
      n_trades: null,
      n_rebalances: null,
      data_source: "fixture",
      kind: seed.walkForward ? "walk_forward" : "standard",
      wf_splits: seed.walkForward ? 5 : null,
      n_folds: seed.walkForward ? 5 : null,
    };
  });
  return {
    runs,
    available_strategies: ["tsmom_vol_target", "drl_ppo_llm_risk", "carry_riskparity"],
  };
}

export function mockTearsheet(runId: string): TearsheetResponse | null {
  const seed = RUN_SEEDS.find((entry) => entry.id === runId) ?? RUN_SEEDS[0]!;
  const curve = generateEquityCurve(seed.seed, seed.vol);
  return {
    run_id: seed.id,
    equity_curve: curve.points,
    rolling_sharpe: buildRollingSharpe(curve.returns),
    underwater: buildUnderwater(curve.points),
    tail_distribution: buildTailDistribution(curve.returns),
    roll_yield_attribution: buildAttribution(seed.seed + 11),
    monthly_returns: buildMonthlyReturns(curve.points),
    metrics: computeMetrics(curve.points, curve.returns),
  };
}

// ---------------------------------------------------------------------------
// Health / telemetry fallbacks
// ---------------------------------------------------------------------------

export function mockHealth(): HealthResponse {
  return {
    status: "ok",
    app_name: "FuturesRL-X",
    version: "1.0.0",
    environment: "paper",
    offline_mode: true,
    uptime_seconds: 0,
    server_time: isoFromMs(nowMs()),
    ibkr_connected: false,
    runner_bridge_available: false,
    runner_bridge_stale: false,
    runner_mode: "fixture",
  };
}

export function mockWorkingOrders(): WorkingOrdersResponse {
  return {
    as_of: isoFromMs(nowMs()),
    source: "fixture",
    mode: "fixture",
    stale: false,
    sequence: 0,
    orders: [],
  };
}

export function mockRecentFills(): FillsResponse {
  return {
    as_of: isoFromMs(nowMs()),
    source: "fixture",
    fills: [],
  };
}

export function mockRunnerStatus(advance = 0): RunnerStatus {
  return {
    mode: "fixture",
    available: false,
    stale: false,
    sequence: advance,
    updated_at: isoFromMs(nowMs()),
    runner_id: "fixture-runner",
    session_running: false,
    states_processed: advance,
  };
}

export function mockTelemetryStatus() {
  return {
    stream_active: false,
    connected_clients: 0,
    published_events: 0,
    publisher_interval_seconds: 1,
    topics: ["telemetry"],
    server_time: isoFromMs(nowMs()),
  };
}

// ---------------------------------------------------------------------------
// Client-side telemetry simulator (used when /ws/telemetry is unreachable)
// ---------------------------------------------------------------------------

export interface TelemetrySimulatorSeed {
  netLiquidation: number;
  initialMargin: number;
  marginUtilization: number;
}

export function createTelemetrySimulator(seed: TelemetrySimulatorSeed) {
  const rng = mulberry32(0xc0ffee);
  let tick = 0;
  let nlv = seed.netLiquidation;
  let equityCurve = generateEquityCurve(20260101, 0.0082, 0.00055, seed.netLiquidation).points;

  return function nextEnvelope(): TelemetryEnvelope {
    tick += 1;
    const timestamp = isoFromMs(nowMs());
    const pnlDelta = gaussian(rng) * 640;
    nlv = Math.max(seed.netLiquidation * 0.8, nlv + pnlDelta);
    const point: EquityPoint = { time: timestamp, value: Math.round(nlv * 100) / 100 };
    equityCurve = [...equityCurve.slice(-251), point];
    const utilization = Math.min(0.95, Math.max(0.05, seed.marginUtilization + gaussian(rng) * 0.004));
    const kindRoll = tick % 4;
    if (kindRoll === 0) {
      return {
        type: "telemetry",
        kind: "pnl",
        timestamp,
        sequence: tick,
        payload: {
          net_liquidation: point.value,
          day_pnl: Math.round((nlv - seed.netLiquidation * 0.996) * 100) / 100,
          day_pnl_pct: (nlv - seed.netLiquidation * 0.996) / seed.netLiquidation,
          point,
        },
      };
    }
    if (kindRoll === 1) {
      return {
        type: "telemetry",
        kind: "margin",
        timestamp,
        sequence: tick,
        payload: {
          initial_margin_used: Math.round(utilization * nlv),
          maintenance_margin_used: Math.round(utilization * nlv * 0.905),
          margin_utilization: utilization,
          warning_threshold: 0.6,
          critical_threshold: 0.8,
        },
      };
    }
    if (kindRoll === 2) {
      const positions = buildPositions(nlv);
      const grossNotional = positions.reduce((sum, position) => sum + Math.abs(position.notional), 0);
      const netExposure = positions.reduce((sum, position) => sum + position.notional, 0);
      return {
        type: "telemetry",
        kind: "positions",
        timestamp,
        sequence: tick,
        payload: {
          positions,
          gross_leverage: grossNotional / nlv,
          net_exposure: netExposure,
        },
      };
    }
    return {
      type: "telemetry",
      kind: "heartbeat",
      timestamp,
      sequence: tick,
      payload: { status: "ok", sequence: tick },
    };
  };
}

/** Simulated connection state label used by the demo indicator. */
export function demoConnectionLabel(state: ConnectionState): string {
  switch (state) {
    case "live":
      return "LIVE";
    case "demo":
      return "UI FALLBACK";
    case "connecting":
      return "CONNECTING";
    default:
      return "OFFLINE";
  }
}


// ---------------------------------------------------------------------------
// Strategy Studio (Phase 10)
// ---------------------------------------------------------------------------

const STUDIO_ROOTS: string[] = [
  "ES", "NQ", "RTY", "YM", "ZN", "ZB", "ZF", "CL", "GC", "SI", "HG", "6E", "6J", "ZC", "ZS", "LE",
];

function studioTranslator(overrides: Partial<StudioTranslatorConfig> = {}): StudioTranslatorConfig {
  return {
    rounding_mode: "nearest",
    rebalance_buffer_contracts: 2,
    max_margin_utilization: 0.6,
    enable_micro_routing: true,
    min_micro_notional_ratio: 0,
    ...overrides,
  };
}

function param(
  path: string,
  stage: StudioParameter["stage"],
  label: string,
  description: string,
  type: StudioParameter["type"],
  defaultValue: StudioParamValue,
  extra: Partial<StudioParameter> = {},
): StudioParameter {
  return {
    path,
    key: extra.key ?? path,
    stage,
    label,
    description,
    type,
    default: defaultValue,
    min: extra.min ?? null,
    max: extra.max ?? null,
    step: extra.step ?? null,
    options: extra.options,
    unit: extra.unit ?? null,
    optuna_tunable: extra.optuna_tunable ?? false,
    optuna_default_selected: extra.optuna_default_selected ?? false,
    hygiene: extra.hygiene ?? false,
    structural: extra.structural ?? false,
    log_scale: extra.log_scale ?? false,
    yaml_encoding: extra.yaml_encoding ?? null,
    location: extra.location ?? null,
    min_items: extra.min_items ?? null,
    max_items: extra.max_items ?? null,
    group: extra.group ?? null,
    stage_name: extra.stage_name ?? null,
  };
}

/** Deterministic fallback catalog consumed when the live catalog is absent. */
export function mockStudioCatalog(): StudioCatalogResponse {
  const parameters: StudioParameter[] = [
    param("universe.selector", "universe", "Universe selector", "Filters the tradable roots by liquidity, open interest and term-structure regime.", "select", "liquidity", {
      options: [
        { value: "liquidity", label: "Liquidity + OI" },
        { value: "regime", label: "Macro regime" },
        { value: "composite", label: "Composite" },
      ],
    }),
    param("universe.roots", "universe", "Universe roots", "Futures roots eligible for this strategy. Selected roots still pass the point-in-time liquidity filter.", "root_multiselect", "ES,NQ,RTY,YM,ZN,CL,GC,6E", {
      min: 2, max: 16, step: 1, yaml_encoding: "comma_strings",
    }),
    param("universe.params.min_avg_dollar_volume", "universe", "Min ADV (USD)", "Roots below this average daily dollar volume are excluded before allocation.", "float", 50_000_000, {
      min: 10_000_000, max: 200_000_000, step: 5_000_000, unit: "$", optuna_tunable: false, hygiene: true,
    }),
    param("universe.params.min_open_interest", "universe", "Min open interest", "Reject thin contracts that would be costly to roll or exit.", "int", 5_000, {
      min: 1_000, max: 20_000, step: 500, optuna_tunable: false, hygiene: true,
    }),
    param("universe.params.top_n", "universe", "Max roots", "Hard cap on the number of roots admitted after ranking.", "int", 10, {
      min: 3, max: 15, step: 1, optuna_tunable: false, hygiene: true,
    }),
    param("universe.params.min_history_days", "universe", "Min history (days)", "Warm-up requirement so trend and covariance features are available.", "int", 252, {
      min: 60, max: 504, step: 21, optuna_tunable: false, hygiene: true,
    }),

    param("allocator.name", "allocator", "Allocator", "Base portfolio construction before timing and risk overlays.", "select", "risk_parity", {
      options: [
        { value: "risk_parity", label: "Risk parity (ERC)" },
        { value: "volatility_target", label: "Volatility target" },
        { value: "mean_variance", label: "Mean variance" },
        { value: "equal_weight", label: "Equal weight" },
        { value: "drl", label: "DRL policy" },
      ],
    }),
    param("allocator.params.target_annual_volatility", "allocator", "Target annual vol", "Portfolio volatility the allocator aims for before overlays.", "float", 0.12, {
      min: 0.05, max: 0.3, step: 0.005, unit: "%", optuna_tunable: true, optuna_default_selected: true,
    }),
    param("allocator.params.covariance_lookback_days", "allocator", "Covariance lookback (days)", "EWMA window used to estimate the risk model.", "int", 60, {
      min: 20, max: 120, step: 5, optuna_tunable: true,
    }),
    param("allocator.params.max_gross_leverage", "allocator", "Max gross leverage", "Ceiling on the sum of absolute weights before risk scaling.", "float", 3.0, {
      min: 1, max: 5, step: 0.1, unit: "×", optuna_tunable: true,
    }),
    param("allocator.params.risk_aversion", "allocator", "Risk aversion", "Mean-variance utility penalty on portfolio variance.", "float", 4.0, {
      min: 1, max: 10, step: 0.5, optuna_tunable: true,
    }),

    param("timing.tsmom.enabled", "timing", "Time-series momentum", "Trade with the 21/126-day trend, sized by volatility.", "bool", true, { group: "TSMOM" }),
    param("timing.tsmom.params.lookback_fast", "timing", "Fast lookback (days)", "Short trend window; shorter reacts faster but churns more.", "int", 21, {
      min: 5, max: 63, step: 1, group: "TSMOM", optuna_tunable: true, optuna_default_selected: true,
    }),
    param("timing.tsmom.params.lookback_slow", "timing", "Slow lookback (days)", "Long trend window; defines the structural trend regime.", "int", 126, {
      min: 63, max: 252, step: 7, group: "TSMOM", optuna_tunable: true, optuna_default_selected: true,
    }),
    param("timing.tsmom.params.vol_lookback", "timing", "Volatility lookback", "Realised volatility window used to size the momentum signal.", "int", 20, {
      min: 10, max: 60, step: 5, group: "TSMOM",
    }),
    param("timing.kama.enabled", "timing", "KAMA trend filter", "Adaptive moving average confirmation to avoid whipsaw entries.", "bool", true, { group: "KAMA" }),
    param("timing.kama.params.window", "timing", "KAMA efficiency window", "Window for the Kaufman efficiency ratio.", "int", 10, {
      min: 5, max: 30, step: 1, group: "KAMA", optuna_tunable: true, optuna_default_selected: true,
    }),
    param("timing.carry_filter.enabled", "timing", "Carry / term-structure filter", "Tilt towards backwardated roots and penalise expensive contango.", "bool", true, { group: "Carry" }),
    param("timing.carry_filter.params.min_carry_bps", "timing", "Min carry (bps)", "Minimum annualised roll yield required to keep a long position.", "float", 0, {
      min: -50, max: 50, step: 5, group: "Carry", optuna_tunable: true, optuna_default_selected: true,
    }),
    param("timing.mean_reversion.enabled", "timing", "Mean reversion (RSI + Bollinger)", "Fade panic sell-offs when RSI or the lower Bollinger band is breached and turbulence is calm.", "bool", false, { group: "Mean reversion" }),
    param("timing.mean_reversion.params.rsi_period", "timing", "RSI period", "RSI window for oversold / overbought detection.", "int", 14, {
      min: 5, max: 30, step: 1, group: "Mean reversion", optuna_tunable: true,
    }),
    param("timing.mean_reversion.params.oversold_threshold", "timing", "Oversold threshold", "RSI level that triggers a reversion long.", "float", 25, {
      min: 10, max: 40, step: 0.5, group: "Mean reversion",
    }),
    param("timing.mean_reversion.params.overbought_threshold", "timing", "Overbought threshold", "RSI level that reduces or shorts the position.", "float", 75, {
      min: 60, max: 90, step: 0.5, group: "Mean reversion",
    }),
    param("timing.mean_reversion.params.reversion_strength", "timing", "Reversion strength", "How aggressively an oversold signal tilts the base weight.", "float", 0.5, {
      min: 0, max: 1, step: 0.05, group: "Mean reversion", optuna_tunable: true,
    }),
    param("timing.mean_reversion.params.bollinger_window", "timing", "Bollinger window", "Window for the Bollinger band envelope.", "int", 20, {
      min: 10, max: 50, step: 1, group: "Mean reversion", optuna_tunable: true,
    }),
    param("timing.mean_reversion.params.bollinger_std", "timing", "Bollinger std", "Band width in standard deviations.", "float", 2.0, {
      min: 1, max: 3, step: 0.1, group: "Mean reversion",
    }),
    param("timing.mean_reversion.params.turbulence_veto_threshold", "timing", "Turbulence veto", "Skip reversion when the Mahalanobis turbulence index exceeds this level.", "float", 0.95, {
      min: 0.5, max: 1, step: 0.05, group: "Mean reversion",
    }),
    param("timing.seasonality.enabled", "timing", "Seasonality tilt", "Tilt weights by root-specific calendar seasonality; scale down out of season.", "bool", false, { group: "Seasonality" }),
    param("timing.seasonality.params.active_months", "timing", "Active months", "Calendar months in which the seasonal tilt is allowed to apply (all by default).", "multiselect", [
      "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
    ], {
      options: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"].map(
        (label, index) => ({ value: String(index + 1), label }),
      ),
      group: "Seasonality", min_items: 1, max_items: 12, min: 1, max: 12, step: 1,
      yaml_encoding: "comma_ints",
    }),
    param("timing.seasonality.params.root_month_biases", "timing", "Seasonal roots", "Roots with built-in commodity seasonality curves (backend encodes the month biases).", "multiselect", [
      "CL", "GC", "ZC", "ZS", "LE", "HG",
    ], {
      options: ["CL", "GC", "ZC", "ZS", "LE", "HG", "SI", "6E"].map((root) => ({ value: root, label: root })),
      group: "Seasonality", min_items: 1, max_items: 8, min: 1, max: 8, step: 1,
      yaml_encoding: "json_object",
    }),
    param("timing.seasonality.params.out_of_season_scale", "timing", "Out-of-season scale", "Multiplier applied to weights outside the active month window.", "float", 1.0, {
      min: 0, max: 2, step: 0.05, group: "Seasonality", optuna_tunable: true,
    }),

    param("risk.quant_risk.enabled", "risk", "Quant risk overlay", "Volatility targeting, covariance scaling, drawdown stops and daily kill-switch.", "bool", true, { group: "Quant risk" }),
    param("risk.quant_risk.params.vol_target", "risk", "Vol target", "Annualised portfolio volatility enforced by the quant overlay.", "float", 0.12, {
      min: 0.05, max: 0.25, step: 0.005, group: "Quant risk", optuna_tunable: true,
    }),
    param("risk.quant_risk.params.ewma_halflife_days", "risk", "EWMA half-life (days)", "Covariance decay half-life for risk scaling.", "int", 20, {
      min: 5, max: 60, step: 1, group: "Quant risk", optuna_tunable: true,
    }),
    param("risk.quant_risk.params.max_single_instrument_weight", "risk", "Max single weight", "Concentration cap per root after risk projection.", "float", 0.35, {
      min: 0.05, max: 0.6, step: 0.01, group: "Quant risk", optuna_tunable: true,
    }),
    param("risk.quant_risk.params.max_sector_weight", "risk", "Max sector weight", "Correlation-cluster cap (e.g. equity index beta bucket).", "float", 0.4, {
      min: 0.1, max: 0.8, step: 0.01, group: "Quant risk",
    }),
    param("risk.quant_risk.params.drawdown_limit_pct", "risk", "Drawdown limit", "Trailing drawdown that forces de-risking.", "float", 0.2, {
      min: 0.05, max: 0.5, step: 0.01, unit: "%", group: "Quant risk", optuna_tunable: true,
    }),
    param("risk.quant_risk.params.trailing_stop_pct", "risk", "Trailing stop", "High-water-mark trailing stop distance per position.", "float", 0.1, {
      min: 0.01, max: 0.5, step: 0.01, unit: "%", group: "Quant risk", optuna_tunable: true, optuna_default_selected: true,
    }),
    param("risk.quant_risk.params.daily_max_loss_pct", "risk", "Daily max loss", "Intraday loss that trips the kill-switch and flattens the book.", "float", 0.03, {
      min: 0.005, max: 0.1, step: 0.001, unit: "%", group: "Quant risk", optuna_tunable: true,
    }),
    param("risk.llm_risk.enabled", "risk", "LLM Risk-First overlay", "Semantic risk scores with the anti-hallucination filter and deterministic circuit breaker.", "bool", true, { group: "LLM risk" }),
    param("risk.llm_risk.params.confidence_threshold", "risk", "Confidence τ", "Below this confidence the LLM signal is neutralised to θ_safe.", "float", 0.7, {
      min: 0.4, max: 0.95, step: 0.01, group: "LLM risk", optuna_tunable: true,
    }),
    param("risk.llm_risk.params.safe_score", "risk", "Safe score θ_safe", "Semantic risk score treated as benign.", "float", 1.5, {
      min: 1, max: 2.5, step: 0.1, group: "LLM risk",
    }),
    param("risk.llm_risk.params.critical_score", "risk", "Critical score κ", "At or above this score the breaker forces the weight to zero.", "float", 4.5, {
      min: 3, max: 5, step: 0.1, group: "LLM risk",
    }),
    param("risk.llm_risk.params.risk_penalty_lambda", "risk", "LLM penalty λ", "Reward-shaping weight for semantic risk during DRL training.", "float", 0.5, {
      min: 0, max: 2, step: 0.05, group: "LLM risk", optuna_tunable: true,
    }),

    param("translator.rounding_mode", "translator", "Rounding mode", "Contract conversion rounding. Conservative strategies should use floor.", "select", "nearest", {
      options: [
        { value: "nearest", label: "Nearest" },
        { value: "floor", label: "Floor (conservative)" },
        { value: "ceil", label: "Ceil" },
      ],
    }),
    param("translator.rebalance_buffer_contracts", "translator", "Rebalance buffer", "Minimum contract delta before an order is sent, to prevent fee churn.", "int", 2, {
      min: 0, max: 10, step: 1, optuna_tunable: true,
    }),
    param("translator.max_margin_utilization", "translator", "Margin ceiling η_max", "Hard projected initial-margin ceiling enforced before order generation (policy max 60%).", "float", 0.6, {
      min: 0.3, max: 0.6, step: 0.01, unit: "%", optuna_tunable: true,
    }),
    param("translator.enable_micro_routing", "translator", "Micro routing", "Route residual exposure to MES/MNQ/MCL/MGC when standard capital is too large.", "bool", true),
    param("translator.min_micro_notional_ratio", "translator", "Min micro ratio", "Minimum fraction of a standard contract before micro routing engages.", "float", 0, {
      min: 0, max: 0.5, step: 0.01, optuna_tunable: true,
    }),
  ];

  const presets: StudioPreset[] = [
    {
      id: "trend_following_tsmom_kama",
      name: "Trend Following (TSMOM + KAMA)",
      description: "Volatility-targeted time-series momentum across the liquid futures universe with KAMA confirmation and a carry filter.",
      hypothesis: "Markets trend; sizing by inverse volatility harvests persistent multi-month moves while capping risk.",
      tags: ["classical", "tsmom", "vol-target"],
      universe: { name: "liquidity", params: { roots: "ES,NQ,RTY,YM,ZN,CL,GC,6E", min_avg_dollar_volume: 50_000_000, min_open_interest: 5_000, top_n: 10, min_history_days: 252 } },
      allocator: { name: "volatility_target", params: { target_annual_volatility: 0.12, covariance_lookback_days: 60, max_gross_leverage: 3.0 } },
      timing: [
        { name: "tsmom", params: { lookback_fast: 21, lookback_slow: 126, vol_lookback: 20 } },
        { name: "kama", params: { window: 10 } },
        { name: "carry_filter", params: { min_carry_bps: 0 } },
      ],
      risk: [
        { name: "quant_risk", params: { vol_target: 0.12, ewma_halflife_days: 20, max_single_instrument_weight: 0.35, max_sector_weight: 0.4, drawdown_limit_pct: 0.2, daily_max_loss_pct: 0.03 } },
        { name: "llm_risk", params: { confidence_threshold: 0.7, safe_score: 1.5, critical_score: 4.5, risk_penalty_lambda: 0.5 } },
      ],
      translator: studioTranslator(),
    },
    {
      id: "commodity_carry_term_structure",
      name: "Commodity Carry & Term Structure",
      description: "Roll-yield carry harvest on backwardated commodities with risk-parity sizing and a macro lockout.",
      hypothesis: "Backwardated curves pay a roll yield; risk-parity sizing diversifies the carry factor across energy, metals and agriculture.",
      tags: ["carry", "risk-parity", "macro"],
      universe: { name: "liquidity", params: { roots: "CL,GC,SI,HG,ZC,ZS,LE", min_avg_dollar_volume: 75_000_000, min_open_interest: 5_000, top_n: 8, min_history_days: 252 } },
      allocator: { name: "risk_parity", params: { covariance_lookback_days: 60 } },
      timing: [{ name: "carry_filter", params: { min_carry_bps: 10 } }],
      risk: [
        { name: "quant_risk", params: { vol_target: 0.1, ewma_halflife_days: 20, max_single_instrument_weight: 0.35, max_sector_weight: 0.4, drawdown_limit_pct: 0.2, daily_max_loss_pct: 0.03 } },
      ],
      translator: studioTranslator({ max_margin_utilization: 0.55, rebalance_buffer_contracts: 2 }),
    },
    {
      id: "panic_mean_reversion",
      name: "Panic Mean-Reversion (RSI + Bollinger)",
      description: "Fade panic sell-offs with RSI and Bollinger bands; suppress reversion entries during high-turbulence regime breaks.",
      hypothesis: "Fade RSI/Bollinger extremes only when the turbulence index says the move is noise, not a regime break.",
      tags: ["mean-reversion", "rsi", "bollinger"],
      universe: { name: "liquidity", params: { roots: "ES,NQ,RTY,YM,CL,GC", min_avg_dollar_volume: 50_000_000, min_open_interest: 5_000, top_n: 8, min_history_days: 252 } },
      allocator: { name: "equal_weight", params: {} },
      timing: [
        { name: "mean_reversion", params: { rsi_period: 14, oversold_threshold: 25, overbought_threshold: 75, bollinger_window: 20, bollinger_std: 2.0, turbulence_veto_threshold: 0.95, reversion_strength: 0.5 } },
        { name: "tsmom", params: { lookback_fast: 21, lookback_slow: 126, vol_lookback: 20 } },
      ],
      risk: [
        { name: "quant_risk", params: { vol_target: 0.1, drawdown_limit_pct: 0.15, daily_max_loss_pct: 0.025 } },
        { name: "llm_risk", params: { confidence_threshold: 0.7 } },
      ],
      translator: studioTranslator({ rounding_mode: "floor", rebalance_buffer_contracts: 3 }),
    },
    {
      id: "seasonal_commodity_cycle",
      name: "Seasonal Commodity Cycle",
      description: "Calendar-seasonality tilt across energy, metals and grains with carry confirmation and vol targeting.",
      hypothesis: "Commodity demand/inventory cycles create repeatable calendar tilts; lean into seasonal strength and de-risk out of season.",
      tags: ["seasonality", "commodities", "carry"],
      universe: { name: "regime", params: { roots: "CL,GC,SI,HG,ZC,ZS,LE", min_avg_dollar_volume: 60_000_000, min_open_interest: 5_000, top_n: 8, min_history_days: 252 } },
      allocator: { name: "risk_parity", params: { covariance_lookback_days: 60 } },
      timing: [
        {
          name: "seasonality",
          params: {
            active_months: ["4", "5", "6", "7", "8", "9", "10"],
            root_month_biases: ["CL", "GC", "ZC", "ZS"],
            out_of_season_scale: 0.6,
          },
        },
        { name: "carry_filter", params: { min_carry_bps: 0 } },
      ],
      risk: [
        { name: "quant_risk", params: { vol_target: 0.12, drawdown_limit_pct: 0.2 } },
      ],
      translator: studioTranslator({ max_margin_utilization: 0.5 }),
    },
    {
      id: "drl_neural_policy_llm_risk",
      name: "DRL Neural Policy + LLM Risk",
      description: "PPO allocator over the liquid universe with TSMOM timing, the LLM Risk-First overlay and deterministic circuit breaker.",
      hypothesis: "A learned policy can capture non-linear cross-asset structure while the deterministic risk layer bounds tail behaviour.",
      tags: ["drl", "ppo", "llm-risk"],
      universe: { name: "liquidity", params: { roots: "ES,NQ,RTY,YM,ZN,CL,GC,6E", min_avg_dollar_volume: 50_000_000, min_open_interest: 5_000, top_n: 10, min_history_days: 252 } },
      allocator: { name: "drl", params: { algorithm: "PPO", total_timesteps: 100_000, learning_rate: 0.0003, gamma: 0.99 } },
      timing: [{ name: "tsmom", params: { lookback_fast: 21, lookback_slow: 126, vol_lookback: 20 } }],
      risk: [
        { name: "quant_risk", params: { vol_target: 0.12, drawdown_limit_pct: 0.2 } },
        { name: "llm_risk", params: { confidence_threshold: 0.7, risk_penalty_lambda: 0.5 } },
      ],
      translator: studioTranslator(),
      drl: {
        algorithm: "PPO",
        policy: "MlpPolicy",
        total_timesteps: 100_000,
        learning_rate: 0.0003,
        gamma: 0.99,
        seed: 7,
        reward_risk_penalty_lambda: 0.5,
        normalize_observations: true,
        model_path: null,
      },
    },
  ];

  const instruments = mockInstruments();
  return {
    version: "1.0",
    roots: STUDIO_ROOTS,
    parameters,
    presets,
    strategies: mockStrategies(),
    stage_choices: {
      universe: ["liquidity", "regime", "composite"],
      allocator: ["volatility_target", "risk_parity", "equal_weight", "mean_variance", "drl"],
      timing: ["tsmom", "kama", "carry_filter", "mean_reversion", "seasonality", "composite"],
      risk: ["quant_risk", "llm_risk"],
      translator: ["translator"],
    },
    selector_choices: [
      { value: "liquidity", label: "Liquidity + OI" },
      { value: "regime", label: "Macro regime" },
      { value: "composite", label: "Composite" },
    ],
    allocator_choices: [
      { value: "volatility_target", label: "Volatility target" },
      { value: "risk_parity", label: "Risk parity" },
      { value: "equal_weight", label: "Equal weight" },
      { value: "mean_variance", label: "Mean variance" },
      { value: "drl", label: "DRL policy" },
    ],
    universe: { roots: instruments.roots, sectors: instruments.sectors },
  };
}

/** Deterministic fallback for POST /api/strategies / PUT /api/strategies/{id}. */
export function mockSavedStrategy(payload: Record<string, unknown>): StrategyConfig {
  const universeRaw = (payload.universe ?? {}) as Record<string, unknown>;
  const allocatorRaw = (payload.allocator ?? {}) as Record<string, unknown>;
  const asStage = (raw: unknown): StageConfig => {
    const record = (raw ?? {}) as Record<string, unknown>;
    const params: Record<string, number | string | boolean> = {};
    for (const [key, value] of Object.entries((record.params ?? {}) as Record<string, unknown>)) {
      if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
        params[key] = value;
      }
    }
    return { name: String(record.name ?? record.selector ?? "unknown"), params };
  };
  const timing = Array.isArray(payload.timing) ? payload.timing.map(asStage) : [];
  const risk = Array.isArray(payload.risk) ? payload.risk.map(asStage) : [];
  return {
    id: String(payload.id ?? "studio_strategy").trim().toLowerCase().replace(/\s+/g, "_"),
    description: String(payload.description ?? ""),
    tags: Array.isArray(payload.tags) ? payload.tags.map(String) : [],
    universe: asStage(universeRaw),
    allocator: asStage(allocatorRaw),
    timing,
    risk,
    updated_at: isoFromMs(nowMs()),
  };
}

// ---------------------------------------------------------------------------
// Optuna optimizer fixtures (Phase 10)
// ---------------------------------------------------------------------------

const OPTIMIZER_PARAM_BOUNDS: Record<string, [number, number, number]> = {
  "universe.params.min_avg_dollar_volume": [20_000_000, 120_000_000, 5_000_000],
  "universe.params.top_n": [3, 12, 1],
  "allocator.params.target_annual_volatility": [0.07, 0.18, 0.005],
  "allocator.params.covariance_lookback_days": [20, 120, 5],
  "timing.tsmom.params.lookback_fast": [5, 63, 1],
  "timing.tsmom.params.lookback_slow": [84, 252, 7],
  "risk.quant_risk.params.vol_target": [0.07, 0.18, 0.005],
  "risk.quant_risk.params.drawdown_limit_pct": [0.08, 0.3, 0.01],
  "risk.llm_risk.params.confidence_threshold": [0.55, 0.9, 0.01],
  "translator.rebalance_buffer_contracts": [0, 8, 1],
  "translator.max_margin_utilization": [0.4, 0.6, 0.01],
};

/** Deterministic fallback for POST /api/optimize. */
export function mockOptimizationStudy(request?: Partial<OptimizationRequest>): OptimizationStudy {
  const seed = 0x5eed + (request?.n_trials ?? 25) * 13 + String(request?.objective_metric ?? "sharpe").length;
  const rng = mulberry32(seed);
  const nTrials = Math.max(2, Math.min(200, request?.n_trials ?? 25));
  const objective = request?.objective_metric ?? "sharpe";
  const threshold = request?.drawdown_penalty_threshold ?? 0.2;
  const spaces = (request?.search_space ?? []).filter((entry) => entry.low !== undefined && entry.high !== undefined);
  const trials: OptimizationTrial[] = [];
  let bestValue = Number.NEGATIVE_INFINITY;
  let bestTrial = 1;
  for (let index = 0; index < nTrials; index += 1) {
    const trialNumber = index + 1;
    const params: Record<string, number | string | boolean> = {};
    for (const entry of spaces.length > 0 ? spaces : Object.keys(OPTIMIZER_PARAM_BOUNDS).slice(0, 6).map((path) => {
      const bounds = OPTIMIZER_PARAM_BOUNDS[path] as [number, number, number];
      return { param_path: path, low: bounds[0], high: bounds[1], type: "float" as const, step: bounds[2] };
    })) {
      if (entry.low === undefined || entry.high === undefined) continue;
      const value = entry.low + rng() * (entry.high - entry.low);
      const step = entry.step ?? (entry.type === "int" ? 1 : 0.01);
      params[entry.param_path] = entry.type === "int" ? Math.round(value) : Math.round(value / step) * step;
    }
    const base = 0.72 + rng() * 0.55;
    const maxDd = 0.08 + rng() * 0.18;
    const penalty = objective === "sharpe" && maxDd > threshold ? (maxDd - threshold) * 1.8 : 0;
    const value = Math.max(-0.5, base - penalty);
    const trial: OptimizationTrial = {
      trial_number: trialNumber,
      params,
      oos_sharpe: Math.max(-0.4, Number((base + (rng() - 0.5) * 0.18).toFixed(3))),
      cagr: Number((0.06 + rng() * 0.16).toFixed(4)),
      max_dd: Number(maxDd.toFixed(4)),
      calmar: Number((0.4 + rng() * 1.1).toFixed(3)),
      rachev: Number((0.9 + rng() * 0.6).toFixed(3)),
      duration_ms: Math.round(180 + rng() * 520),
      value: Number(value.toFixed(4)),
      best: false,
    };
    if (value > bestValue) {
      bestValue = value;
      bestTrial = trialNumber;
    }
    trials.push(trial);
  }
  const winner = trials[bestTrial - 1] as OptimizationTrial;
  winner.best = true;
  const importances = Object.keys(OPTIMIZER_PARAM_BOUNDS)
    .slice(0, 8)
    .map((path, index) => ({
      param_path: path,
      importance: Number((0.42 / (index + 1) + rng() * 0.06).toFixed(4)),
    }))
    .sort((a, b) => b.importance - a.importance);
  const studyId = `opt-${request?.base_strategy_id ?? "tsmom_vol_target"}-${nTrials}-${objective}`;
  return {
    study_id: studyId,
    base_strategy_id: request?.base_strategy_id ?? "tsmom_vol_target",
    objective_metric: objective,
    n_trials: nTrials,
    walk_forward: true,
    wf_splits: request?.wf_splits ?? 5,
    status: "completed",
    created_at: isoFromMs(nowMs() - 86_400_000),
    duration_ms: 48_200 + Math.round(rng() * 30_000),
    best_value: Number(bestValue.toFixed(4)),
    best_trial_number: bestTrial,
    best_params: { ...winner.params },
    param_importances: importances,
    trials,
    best_strategy_config: null,
    direction: "maximize",
    completed_trials: nTrials,
    fold_count: request?.wf_splits ?? 5,
    backend: "deterministic",
    provenance: "fixture",
    artifact_path: `artifacts/optimizations/${studyId}.json`,
    message: "UI FALLBACK: deterministic offline optimizer fixture (no Optuna study run).",
  };
}

export function mockOptimizationList(): OptimizationListResponse {
  return {
    studies: [
      mockOptimizationStudy({ n_trials: 25, objective_metric: "sharpe", wf_splits: 5 }),
      mockOptimizationStudy({ n_trials: 18, objective_metric: "calmar", wf_splits: 4 }),
    ],
  };
}

export function mockOptimizationApply(studyId: string): OptimizationApplyResult {
  const study = mockOptimizationList().studies.find((entry) => entry.study_id === studyId) ?? mockOptimizationStudy();
  return {
    study_id: studyId,
    strategy_id: `${study.base_strategy_id}_opt`,
    strategy: null,
    backtest_run_id: null,
    applied: true,
    verification_backtest: {
      status: "skipped",
      message: "UI FALLBACK: verification backtest skipped in offline preview.",
    },
    message: "UI FALLBACK: best parameters applied to an in-memory strategy config.",
  };
}

// ---------------------------------------------------------------------------
// DRL trainer fixtures (Phase 10)
// ---------------------------------------------------------------------------

function drlRewardCurve(seed: number, length = 96): DrlRewardPoint[] {
  const rng = mulberry32(seed);
  const points: DrlRewardPoint[] = [];
  let level = 0.1;
  for (let index = 0; index < length; index += 1) {
    level += (rng() - 0.42) * 0.06 + 0.004;
    points.push({
      step: Math.round(((index + 1) / length) * 20_000),
      reward: Number(level.toFixed(4)),
    });
  }
  return points;
}

function drlModel(
  name: string,
  algorithm: string,
  timesteps: number,
  universe: string[],
  seed: number,
  createdDaysAgo: number,
): DrlModel {
  const curve = drlRewardCurve(seed);
  const finals = curve.map((point) => point.reward);
  const finalMean = finals[finals.length - 1] ?? 0;
  const bestMean = Math.max(...finals, 0);
  const hyperparameters = {
    algorithm,
    sb3_algorithm: algorithm === "CPPO" ? "PPO" : algorithm,
    policy: "MlpPolicy",
    total_timesteps: timesteps,
    learning_rate: 0.0003,
    gamma: 0.99,
    llm_reward_penalty_lambda: 0.5,
    seed,
    device: "cpu",
  };
  const meta = {
    checkpoint_name: name,
    model_path: `artifacts/models/${name}.zip`,
    meta_path: `artifacts/models/${name}.zip.meta.json`,
    metadata_available: true,
    metadata_error: null,
    algorithm,
    sb3_algorithm: hyperparameters.sb3_algorithm,
    strategy_id: "tsmom_vol_target",
    policy: "MlpPolicy",
    total_timesteps: timesteps,
    learning_rate: 0.0003,
    gamma: 0.99,
    llm_reward_penalty_lambda: 0.5,
    seed,
    device: "cpu",
    offline: true,
    universe,
    feature_names: ["tsmom", "carry", "term_slope", "turbulence", "llm_risk_score"],
    reward_curve: {
      window_steps: 500,
      samples: curve.length,
      episode_count: Math.round(timesteps / 500),
      env_steps: timesteps,
      final_mean_reward: finalMean,
      best_mean_reward: bestMean,
      path: `artifacts/models/${name}.zip.reward_curve.json`,
    },
    created_at: daysAgoIso(createdDaysAgo),
    git_commit: "a3f9c21",
    format: "sb3-checkpoint-meta-v1",
  };
  return {
    name,
    algorithm,
    timesteps,
    universe,
    features: meta.feature_names,
    created_at: meta.created_at,
    git_commit: meta.git_commit,
    size_bytes: 3_200_000 + seed * 997,
    reward_curve: curve,
    reward_summary: {
      window_steps: 500,
      samples: curve.length,
      episode_count: Math.round(timesteps / 500),
      env_steps: timesteps,
      final_mean_reward: finalMean,
      best_mean_reward: bestMean,
      path: meta.reward_curve.path,
    },
    hyperparameters,
    meta,
    path: meta.model_path,
    meta_path: meta.meta_path,
    metadata_available: true,
    metadata_error: null,
    sb3_algorithm: hyperparameters.sb3_algorithm,
    strategy_id: "tsmom_vol_target",
    policy: "MlpPolicy",
    learning_rate: 0.0003,
    gamma: 0.99,
    llm_reward_penalty_lambda: 0.5,
    seed,
    device: "cpu",
    offline: true,
  };
}

export function mockDrlModels(): DrlModelsResponse {
  const models = [
    drlModel("ppo_tsmom_100k", "PPO", 20_000, ["ES", "NQ", "ZN", "CL", "GC"], 7, 2),
    drlModel("cppo_llm_risk_250k", "CPPO", 20_000, ["ES", "NQ", "RTY", "CL", "GC", "6E"], 11, 4),
    drlModel("sac_carry_150k", "SAC", 20_000, ["CL", "GC", "ZC", "HG", "6E"], 13, 9),
  ];
  return {
    models,
    count: models.length,
    models_dir: "artifacts/models",
    generated_at: isoFromMs(nowMs()),
  };
}

export function mockDrlTrain(request: DrlTrainRequest): DrlTrainResult {
  const algorithm = request.algorithm || "PPO";
  const requested = (request.checkpoint_name ?? "").trim();
  const baseName = requested.length > 0
    ? requested.toLowerCase().replace(/[^a-z0-9._-]+/g, "_")
    : `drl_${algorithm.toLowerCase()}_${request.timesteps}`;
  const seed = 17 + algorithm.length + (request.timesteps % 997);
  const model = drlModel(baseName, algorithm, request.timesteps, ["ES", "NQ", "ZN", "CL", "GC"], seed, 0);
  model.created_at = isoFromMs(nowMs());
  model.hyperparameters = {
    ...model.hyperparameters,
    learning_rate: request.learning_rate ?? 0.0003,
    gamma: request.gamma ?? 0.99,
    llm_reward_penalty_lambda: request.llm_reward_penalty_lambda ?? 0.5,
    strategy_id: request.strategy_id,
    device: request.device,
  };
  model.reward_curve = drlRewardCurve(seed + 3, 64).map((point) => ({
    step: point.step,
    reward: point.reward,
  }));
  const finals = model.reward_curve.map((point) => point.reward);
  model.reward_summary = {
    window_steps: 500,
    samples: model.reward_curve.length,
    episode_count: 40,
    env_steps: request.timesteps,
    final_mean_reward: finals[finals.length - 1] ?? 0,
    best_mean_reward: Math.max(...finals, 0),
    path: `artifacts/models/${model.name}.zip.reward_curve.json`,
  };
  model.meta = {
    ...model.meta,
    checkpoint_name: model.name,
    strategy_id: request.strategy_id,
    algorithm,
    total_timesteps: request.timesteps,
    duration_seconds: 4.2,
    status: "ok",
    offline: request.offline,
  };
  return {
    model,
    reward_curve: model.reward_curve,
    summary: {
      status: "ok",
      strategy_id: request.strategy_id,
      algorithm,
      checkpoint_name: model.name,
      total_timesteps: request.timesteps,
      offline: request.offline,
      note: "UI FALLBACK: deterministic offline training summary (no SB3 run).",
    },
  };
}

export function mockDrlBind(model: DrlModel, strategyId: string): {
  model: string;
  strategy_id: string;
  bound: boolean;
  message: string;
} {
  return {
    model: model.name,
    strategy_id: strategyId,
    bound: true,
    message: `UI FALLBACK: ${model.name} bound to ${strategyId} in memory.`,
  };
}


// ---------------------------------------------------------------------------
// Instrument universe fallback (GET /api/instruments, task-32)
// ---------------------------------------------------------------------------

interface RootSeed {
  root: string;
  name: string;
  sector: string;
  asset_class: string;
  exchange: string;
  multiplier: number;
  tick_size: number;
  micro_root?: string;
}

const ROOT_SEEDS: RootSeed[] = [
  // Equities — 9
  { root: "ES", name: "E-mini S&P 500", sector: "equities", asset_class: "equity_index", exchange: "CME", multiplier: 50, tick_size: 0.25, micro_root: "MES" },
  { root: "MES", name: "Micro E-mini S&P 500", sector: "equities", asset_class: "equity_index", exchange: "CME", multiplier: 5, tick_size: 0.25 },
  { root: "NQ", name: "E-mini Nasdaq-100", sector: "equities", asset_class: "equity_index", exchange: "CME", multiplier: 20, tick_size: 0.25, micro_root: "MNQ" },
  { root: "MNQ", name: "Micro E-mini Nasdaq-100", sector: "equities", asset_class: "equity_index", exchange: "CME", multiplier: 2, tick_size: 0.25 },
  { root: "YM", name: "E-mini Dow Jones", sector: "equities", asset_class: "equity_index", exchange: "CBOT", multiplier: 5, tick_size: 1 },
  { root: "MYM", name: "Micro E-mini Dow Jones", sector: "equities", asset_class: "equity_index", exchange: "CBOT", multiplier: 0.5, tick_size: 1 },
  { root: "RTY", name: "E-mini Russell 2000", sector: "equities", asset_class: "equity_index", exchange: "CME", multiplier: 50, tick_size: 0.1, micro_root: "M2K" },
  { root: "M2K", name: "Micro E-mini Russell 2000", sector: "equities", asset_class: "equity_index", exchange: "CME", multiplier: 5, tick_size: 0.1 },
  { root: "NKD", name: "Nikkei 225 (USD)", sector: "equities", asset_class: "equity_index", exchange: "CME", multiplier: 5, tick_size: 5 },
  // Rates — 7
  { root: "ZN", name: "10-Year Treasury Note", sector: "rates", asset_class: "rates", exchange: "CBOT", multiplier: 1000, tick_size: 0.015625 },
  { root: "ZB", name: "30-Year Treasury Bond", sector: "rates", asset_class: "rates", exchange: "CBOT", multiplier: 1000, tick_size: 0.03125 },
  { root: "ZF", name: "5-Year Treasury Note", sector: "rates", asset_class: "rates", exchange: "CBOT", multiplier: 1000, tick_size: 0.0078125 },
  { root: "ZT", name: "2-Year Treasury Note", sector: "rates", asset_class: "rates", exchange: "CBOT", multiplier: 2000, tick_size: 0.00390625 },
  { root: "ZQ", name: "30-Day Fed Funds", sector: "rates", asset_class: "rates", exchange: "CBOT", multiplier: 4167, tick_size: 0.0025 },
  { root: "GE", name: "Eurodollar", sector: "rates", asset_class: "rates", exchange: "CME", multiplier: 2500, tick_size: 0.0025 },
  { root: "SR3", name: "3-Month SOFR", sector: "rates", asset_class: "rates", exchange: "CME", multiplier: 2500, tick_size: 0.0025 },
  // Energy — 6
  { root: "CL", name: "WTI Crude Oil", sector: "energy", asset_class: "energy", exchange: "NYMEX", multiplier: 1000, tick_size: 0.01, micro_root: "MCL" },
  { root: "MCL", name: "Micro WTI Crude Oil", sector: "energy", asset_class: "energy", exchange: "NYMEX", multiplier: 100, tick_size: 0.01 },
  { root: "RB", name: "RBOB Gasoline", sector: "energy", asset_class: "energy", exchange: "NYMEX", multiplier: 42000, tick_size: 0.0001 },
  { root: "HO", name: "NY Harbor ULSD", sector: "energy", asset_class: "energy", exchange: "NYMEX", multiplier: 42000, tick_size: 0.0001 },
  { root: "NG", name: "Henry Hub Natural Gas", sector: "energy", asset_class: "energy", exchange: "NYMEX", multiplier: 10000, tick_size: 0.001, micro_root: "MNG" },
  { root: "MNG", name: "Micro Henry Hub Natural Gas", sector: "energy", asset_class: "energy", exchange: "NYMEX", multiplier: 2500, tick_size: 0.001 },
  // Metals — 8
  { root: "GC", name: "Gold", sector: "metals", asset_class: "metals", exchange: "COMEX", multiplier: 100, tick_size: 0.1, micro_root: "MGC" },
  { root: "MGC", name: "Micro Gold", sector: "metals", asset_class: "metals", exchange: "COMEX", multiplier: 10, tick_size: 0.1 },
  { root: "SI", name: "Silver", sector: "metals", asset_class: "metals", exchange: "COMEX", multiplier: 5000, tick_size: 0.005, micro_root: "SIL" },
  { root: "SIL", name: "Micro Silver", sector: "metals", asset_class: "metals", exchange: "COMEX", multiplier: 1000, tick_size: 0.005 },
  { root: "HG", name: "Copper", sector: "metals", asset_class: "metals", exchange: "COMEX", multiplier: 25000, tick_size: 0.0005, micro_root: "MHG" },
  { root: "MHG", name: "Micro Copper", sector: "metals", asset_class: "metals", exchange: "COMEX", multiplier: 2500, tick_size: 0.0005 },
  { root: "PL", name: "Platinum", sector: "metals", asset_class: "metals", exchange: "NYMEX", multiplier: 50, tick_size: 0.1 },
  { root: "PA", name: "Palladium", sector: "metals", asset_class: "metals", exchange: "NYMEX", multiplier: 100, tick_size: 0.5 },
  // FX — 6
  { root: "6E", name: "Euro FX", sector: "fx", asset_class: "fx", exchange: "CME", multiplier: 125000, tick_size: 0.00005 },
  { root: "6J", name: "Japanese Yen", sector: "fx", asset_class: "fx", exchange: "CME", multiplier: 12500000, tick_size: 0.0000005 },
  { root: "6B", name: "British Pound", sector: "fx", asset_class: "fx", exchange: "CME", multiplier: 62500, tick_size: 0.0001 },
  { root: "6A", name: "Australian Dollar", sector: "fx", asset_class: "fx", exchange: "CME", multiplier: 100000, tick_size: 0.00005 },
  { root: "6C", name: "Canadian Dollar", sector: "fx", asset_class: "fx", exchange: "CME", multiplier: 100000, tick_size: 0.00005 },
  { root: "6S", name: "Swiss Franc", sector: "fx", asset_class: "fx", exchange: "CME", multiplier: 125000, tick_size: 0.00005 },
  // Agri — 7
  { root: "ZC", name: "Corn", sector: "agri", asset_class: "agriculture", exchange: "CBOT", multiplier: 50, tick_size: 0.25 },
  { root: "ZS", name: "Soybeans", sector: "agri", asset_class: "agriculture", exchange: "CBOT", multiplier: 50, tick_size: 0.25 },
  { root: "ZW", name: "Chicago Wheat", sector: "agri", asset_class: "agriculture", exchange: "CBOT", multiplier: 50, tick_size: 0.25 },
  { root: "LE", name: "Live Cattle", sector: "agri", asset_class: "agriculture", exchange: "CME", multiplier: 400, tick_size: 0.025 },
  { root: "HE", name: "Lean Hogs", sector: "agri", asset_class: "agriculture", exchange: "CME", multiplier: 400, tick_size: 0.025 },
  { root: "ZL", name: "Soybean Oil", sector: "agri", asset_class: "agriculture", exchange: "CBOT", multiplier: 600, tick_size: 0.01 },
  { root: "KE", name: "KC Wheat", sector: "agri", asset_class: "agriculture", exchange: "CBOT", multiplier: 50, tick_size: 0.25 },
  // Crypto — 2
  { root: "BTC", name: "Bitcoin Futures", sector: "crypto", asset_class: "crypto", exchange: "CME", multiplier: 5, tick_size: 5, micro_root: "MBT" },
  { root: "MBT", name: "Micro Bitcoin", sector: "crypto", asset_class: "crypto", exchange: "CME", multiplier: 0.1, tick_size: 5 },
];

const SECTOR_LABELS: Record<string, string> = {
  equity: "Equities",
  equities: "Equities",
  rates: "Rates",
  energy: "Energy",
  metals: "Metals",
  fx: "FX",
  agri: "Agri",
  crypto: "Crypto",
};

const SECTOR_ORDER = ["equities", "rates", "energy", "metals", "fx", "agri", "crypto"];

export function mockInstruments(): InstrumentsResponse {
  const roots: InstrumentRoot[] = ROOT_SEEDS.map((seed) => {
    const standardRoot = ROOT_SEEDS.find((entry) => entry.micro_root === seed.root)?.root ?? null;
    return {
      root: seed.root,
      name: seed.name,
      asset_class: seed.asset_class,
      sector: seed.sector,
      exchange: seed.exchange,
      multiplier: seed.multiplier,
      tick_size: seed.tick_size,
      currency: "USD",
      micro_root: seed.micro_root ?? null,
      standard_root: standardRoot,
    };
  });
  const sectors: InstrumentSector[] = SECTOR_ORDER.map((id) => {
    const sectorRoots = roots
      .filter((entry) => entry.sector === id)
      .sort((a, b) => a.root.localeCompare(b.root));
    return {
      id,
      label: SECTOR_LABELS[id] ?? id,
      count: sectorRoots.length,
      roots: sectorRoots,
    };
  }).filter((sector) => sector.count > 0);
  return {
    generated_at: isoFromMs(nowMs()),
    count: roots.length,
    sectors,
    roots,
  };
}
