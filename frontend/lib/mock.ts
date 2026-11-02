/**
 * Deterministic demo dataset.
 *
 * The dashboard must be fully explorable with the FastAPI backend switched off
 * (`npm run dev` standalone, static preview, screenshots, CI). When a REST call
 * fails the API client falls back to the factories below and the shell lights
 * the `DEMO DATA` indicator. Everything here is seeded (mulberry32) so two
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

const RUN_SEEDS: { id: string; strategyId: string; seed: number; vol: number; createdDaysAgo: number; duration: number }[] = [
  { id: "bt-0031", strategyId: "tsmom_vol_target", seed: 20260101, vol: 0.0082, createdDaysAgo: 1, duration: 184.2 },
  { id: "bt-0030", strategyId: "drl_ppo_llm_risk", seed: 20260102, vol: 0.0068, createdDaysAgo: 3, duration: 512.7 },
  { id: "bt-0029", strategyId: "carry_riskparity", seed: 20260103, vol: 0.0096, createdDaysAgo: 7, duration: 121.5 },
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
      return {
        type: "telemetry",
        kind: "positions",
        timestamp,
        sequence: tick,
        payload: {
          positions: buildPositions(nlv),
          gross_leverage: 0.78,
          net_exposure: nlv * 0.21,
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
      return "DEMO DATA";
    case "connecting":
      return "CONNECTING";
    default:
      return "OFFLINE";
  }
}
