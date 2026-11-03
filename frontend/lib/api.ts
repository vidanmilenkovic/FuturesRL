/**
 * Typed API client for the FuturesRL-X FastAPI backend.
 *
 * Contract: every call site receives `ApiResult<T>` and therefore always has a
 * renderable value — `source: "live"` when FastAPI answered, `source: "demo"`
 * with a deterministic fixture when the backend is unreachable. This is what
 * makes `npm run build` + `next start` and offline previews fully functional
 * while keeping the same typed surface as the real API (`src/api/schemas.py`).
 */

import {
  buildAccount,
  buildPositions,
  createTelemetrySimulator,
  mockBacktestList,
  mockHealth,
  mockPipeline,
  mockRiskState,
  mockStrategies,
  mockDrlBind,
  mockDrlModels,
  mockDrlTrain,
  mockInstruments,
  mockOptimizationApply,
  mockOptimizationList,
  mockOptimizationStudy,
  mockRecentFills,
  mockRunnerStatus,
  mockSavedStrategy,
  mockStudioCatalog,
  mockTelemetryStatus,
  mockTearsheet,
  mockTermStructure,
  mockWorkingOrders,
} from "./mock";
import {
  draftToPayload,
  normalizeDrlModelsResponse,
  normalizeDrlTrainResult,
  normalizeOptimizationApply,
  normalizeOptimizationList,
  normalizeInstrumentsResponse,
  normalizeOptimizationStudy,
  normalizeStrategyConfig,
  normalizeStudioCatalog,
  type YamlEncodingMap,
} from "./studio";
import type {
  AccountSnapshotResponse,
  ApiResult,
  BacktestListResponse,
  ConnectionState,
  ArtifactMutationResult,
  BacktestRunConfig,
  BacktestRunSummary,
  DataCostEstimate,
  DataCostRequest,
  DataIngestRequest,
  DataIngestResult,
  DataInventoryResponse,
  DataCacheRoot,
  PaperStateResetResult,
  RiskLockResetResult,
  FlattenAllResponse,
  HealthResponse,
  PipelineWaterfallResponse,
  RiskStateResponse,
  RiskThresholdOverride,
  RiskThresholds,
  StrategyConfig,
  StrategyMutationResult,
  TearsheetResponse,
  DrlBindResult,
  DrlModel,
  DrlModelsResponse,
  DrlTrainRequest,
  DrlTrainResult,
  FillsResponse,
  InstrumentsResponse,
  OptimizationApplyResult,
  OptimizationListResponse,
  OptimizationRequest,
  OptimizationStudy,
  RebalanceExecuteResponse,
  RebalancePreviewResponse,
  RebalanceOrderPreview,
  RebalancePosition,
  RunnerStatus,
  StudioCatalogResponse,
  StudioDraft,
  StudioSaveResult,
  TelemetryEnvelope,
  TelemetryStatusResponse,
  TermStructureResponse,
  WorkingOrdersResponse,
} from "./types";

const DEFAULT_TIMEOUT_MS = 2_500;

function trimTrailingSlash(url: string): string {
  return url.replace(/\/+$/, "");
}

export function apiBaseUrl(): string {
  const configured = process.env.NEXT_PUBLIC_API_BASE_URL;
  if (configured && configured.length > 0) return trimTrailingSlash(configured);
  if (typeof window !== "undefined") {
    // Same-origin deployments (docker-compose frontend proxy) work by default.
    const { protocol, hostname } = window.location;
    return `${protocol}//${hostname}:8000`;
  }
  return "http://127.0.0.1:8000";
}

export function wsTelemetryUrl(): string {
  const configured = process.env.NEXT_PUBLIC_WS_URL;
  if (configured && configured.length > 0) return configured;
  const base = apiBaseUrl().replace(/^http/, "ws");
  return `${base}/ws/telemetry`;
}

export class ApiError extends Error {
  readonly status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit, timeoutMs = DEFAULT_TIMEOUT_MS): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`${apiBaseUrl()}${path}`, {
      ...init,
      signal: controller.signal,
      headers: {
        Accept: "application/json",
        ...(init?.body ? { "Content-Type": "application/json" } : {}),
        ...init?.headers,
      },
      cache: "no-store",
    });
    if (!response.ok) {
      throw new ApiError(`${response.status} ${response.statusText} · ${path}`, response.status);
    }
    return (await response.json()) as T;
  } finally {
    clearTimeout(timer);
  }
}

async function withFallback<T>(
  path: string,
  fallback: () => T,
  init?: RequestInit,
  timeoutMs?: number,
): Promise<ApiResult<T>> {
  try {
    const data = await request<T>(path, init, timeoutMs);
    return { data, source: "live", error: null };
  } catch (error) {
    return {
      data: fallback(),
      source: "demo",
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

// ---------------------------------------------------------------------------
// System
// ---------------------------------------------------------------------------

export function getHealth(): Promise<ApiResult<HealthResponse>> {
  return withFallback("/health", mockHealth, undefined, 1_500);
}

export function getTelemetryStatus(): Promise<ApiResult<TelemetryStatusResponse>> {
  return withFallback("/api/telemetry/status", mockTelemetryStatus);
}

// ---------------------------------------------------------------------------
// Account / positions
// ---------------------------------------------------------------------------

export function getAccount(): Promise<ApiResult<AccountSnapshotResponse>> {
  // The first live snapshot can block while IBKR streams the ledger; allow more
  // than the default 2.5 s so the dashboard does not silently fall back to mock data.
  return withFallback("/api/positions", () => buildAccount(), undefined, 15_000);
}

// ---------------------------------------------------------------------------
// Pipeline / risk / term structure
// ---------------------------------------------------------------------------

export function getPipeline(): Promise<ApiResult<PipelineWaterfallResponse>> {
  return withFallback("/api/pipeline/latest", () => mockPipeline());
}

export function getRiskState(): Promise<ApiResult<RiskStateResponse>> {
  return withFallback("/api/risk/state", mockRiskState);
}

export function updateRiskThresholds(
  overrides: RiskThresholdOverride,
): Promise<ApiResult<RiskThresholds>> {
  return withFallback(
    "/api/risk/thresholds",
    () => ({ ...mockRiskState().thresholds, ...overrides }),
    { method: "PUT", body: JSON.stringify(overrides) },
  );
}

export function flattenAll(): Promise<ApiResult<FlattenAllResponse>> {
  return withFallback("/api/risk/flatten-all", () => {
    const positions = buildPositions(buildAccount().net_liquidation);
    return {
      status: "flattened" as const,
      timestamp: new Date().toISOString(),
      flattened_positions: positions.length,
      cancelled_orders: 2,
      receipts: positions.map((position, index) => ({
        order_id: `demo-flat-${index + 1}`,
        instrument_id: position.instrument_id,
        side: position.quantity > 0 ? ("SELL" as const) : ("BUY" as const),
        quantity: Math.abs(position.quantity),
        status: "FILLED" as const,
        average_price: position.market_price,
        message: "UI FALLBACK: simulated fill (no broker connected)",
      })),
      execution_locked_until: null,
      message: "UI FALLBACK: flatten simulated locally",
    };
  }, { method: "POST" }, 4_000);
}

// ---------------------------------------------------------------------------
// Execution bridge: working orders / fills / runner
// ---------------------------------------------------------------------------

export function getWorkingOrders(): Promise<ApiResult<WorkingOrdersResponse>> {
  return withFallback("/api/orders/working", mockWorkingOrders);
}

export function getRecentFills(limit = 50): Promise<ApiResult<FillsResponse>> {
  return withFallback(`/api/fills/recent?limit=${limit}`, mockRecentFills);
}

export function getRunnerStatus(): Promise<ApiResult<RunnerStatus>> {
  return withFallback("/api/runner/status", () => mockRunnerStatus());
}

export function stepRunner(states = 1): Promise<ApiResult<RunnerStatus>> {
  return withFallback(
    "/api/runner/step",
    () => mockRunnerStatus(states),
    { method: "POST", body: JSON.stringify({ states }) },
  );
}

// ---------------------------------------------------------------------------
// Term structure
// ---------------------------------------------------------------------------

export function getTermStructure(): Promise<ApiResult<TermStructureResponse>> {
  return withFallback("/api/term-structure", mockTermStructure);
}

// ---------------------------------------------------------------------------
// Strategies
// ---------------------------------------------------------------------------

export function listStrategies(): Promise<ApiResult<StrategyConfig[]>> {
  return withFallback("/api/strategies", mockStrategies);
}

export function updateStrategy(id: string, config: StrategyConfig): Promise<ApiResult<StrategyConfig>> {
  return withFallback(`/api/strategies/${encodeURIComponent(id)}`, () => config, {
    method: "PUT",
    body: JSON.stringify(config),
  });
}

// ---------------------------------------------------------------------------
// Backtests
// ---------------------------------------------------------------------------

export function listBacktests(): Promise<ApiResult<BacktestListResponse>> {
  return withFallback("/api/backtests", mockBacktestList);
}

export function getBacktest(id: string): Promise<ApiResult<BacktestRunSummary | null>> {
  return withFallback(`/api/backtests/${encodeURIComponent(id)}`, () => {
    return mockBacktestList().runs.find((run) => run.id === id) ?? null;
  });
}

export function getTearsheet(id: string): Promise<ApiResult<TearsheetResponse | null>> {
  return withFallback(`/api/backtests/${encodeURIComponent(id)}/tearsheet`, () =>
    mockTearsheet(id),
  );
}

export function runBacktest(config: BacktestRunConfig): Promise<ApiResult<BacktestRunSummary>> {
  return withFallback(
    "/api/backtests",
    () => {
      const list = mockBacktestList();
      const base = list.runs.find((run) => run.strategy_id === config.strategy_id) ?? list.runs[0]!;
      return {
        ...base,
        id: `demo-${Date.now().toString(36)}`,
        created_at: new Date().toISOString(),
        config,
        duration_seconds: 96.4,
      };
    },
    { method: "POST", body: JSON.stringify(config) },
    20_000,
  );
}

// ---------------------------------------------------------------------------
// Phase 11B — market data cache inventory / cost / ingest
// ---------------------------------------------------------------------------

function fallbackDataInventory(): DataInventoryResponse {
  return {
    roots: [],
    totals: {
      cached_roots: 0,
      total_roots: 45,
      total_bars: 0,
      total_size_bytes: 0,
      start_date: null,
      end_date: null,
    },
    databento_configured: false,
    ibkr_available: false,
    schema: "ohlcv-1d",
    generated_at: null,
  };
}

function normalizeCacheRoot(raw: unknown): DataCacheRoot | null {
  if (!isRecord(raw)) return null;
  const root = asMessage(raw.root ?? raw.symbol ?? "").toUpperCase();
  if (!root) return null;
  return {
    root,
    sector: asMessage(raw.sector ?? "") || "unknown",
    schema: asMessage(raw.schema ?? "ohlcv-1d") || "ohlcv-1d",
    cached_bars:
      typeof raw.cached_bars === "number"
        ? raw.cached_bars
        : typeof raw.bars === "number"
          ? raw.bars
          : 0,
    start_date: asMessage(raw.start_date ?? raw.start ?? "") || null,
    end_date: asMessage(raw.end_date ?? raw.end ?? "") || null,
    size_bytes:
      typeof raw.size_bytes === "number"
        ? raw.size_bytes
        : typeof raw.bytes === "number"
          ? raw.bytes
          : 0,
    source: asMessage(raw.source ?? (Number(raw.cached_bars ?? 0) > 0 ? "databento" : "empty")) || "empty",
  };
}

function normalizeDataInventory(raw: unknown): DataInventoryResponse {
  const root = isRecord(raw) ? raw : {};
  const roots = asRebalanceArray(root.roots)
    .map(normalizeCacheRoot)
    .filter((entry): entry is DataCacheRoot => entry !== null)
    .sort((a, b) => a.root.localeCompare(b.root));
  const totalsRaw = isRecord(root.totals) ? root.totals : {};
  const cached = roots.filter((entry) => entry.cached_bars > 0);
  return {
    roots,
    totals: {
      cached_roots:
        typeof totalsRaw.cached_roots === "number" ? totalsRaw.cached_roots : cached.length,
      total_roots:
        typeof totalsRaw.total_roots === "number" ? totalsRaw.total_roots : roots.length,
      total_bars:
        typeof totalsRaw.total_bars === "number"
          ? totalsRaw.total_bars
          : roots.reduce((sum, entry) => sum + entry.cached_bars, 0),
      total_size_bytes:
        typeof totalsRaw.total_size_bytes === "number"
          ? totalsRaw.total_size_bytes
          : roots.reduce((sum, entry) => sum + entry.size_bytes, 0),
      start_date: asMessage(totalsRaw.start_date ?? "") || null,
      end_date: asMessage(totalsRaw.end_date ?? "") || null,
    },
    databento_configured: root.databento_configured === true,
    ibkr_available: root.ibkr_available === true,
    schema: asMessage(root.schema ?? "ohlcv-1d") || "ohlcv-1d",
    generated_at: asMessage(root.generated_at ?? "") || null,
    warnings: asRebalanceArray(root.warnings).map((warning) => asMessage(warning)).filter(Boolean),
  };
}

export async function getDataInventory(): Promise<ApiResult<DataInventoryResponse>> {
  return withFallback<unknown>("/api/data/inventory", fallbackDataInventory, undefined, 15_000).then(
    (result) => ({
      ...result,
      data: result.source === "live" ? normalizeDataInventory(result.data) : fallbackDataInventory(),
    }),
  );
}

function normalizeDataCost(raw: unknown, source: DataCostRequest["source"]): DataCostEstimate {
  const root = isRecord(raw) ? raw : {};
  const bytes =
    typeof root.billable_size_bytes === "number"
      ? root.billable_size_bytes
      : typeof root.billable_size === "number"
        ? root.billable_size
        : 0;
  return {
    estimated_cost_usd:
      typeof root.estimated_cost_usd === "number" ? root.estimated_cost_usd : 0,
    billable_size_bytes: bytes,
    billable_size_mb:
      typeof root.billable_size_mb === "number"
        ? root.billable_size_mb
        : typeof root.billable_size_megabytes === "number"
          ? root.billable_size_megabytes
          : bytes / (1024 * 1024),
    source: asMessage(root.source ?? source) || source,
    currency: asMessage(root.currency ?? "USD") || "USD",
    details: isRecord(root.details) ? root.details : null,
  };
}

export function estimateDataCost(request: DataCostRequest): Promise<ApiResult<DataCostEstimate>> {
  return withFallback<unknown>(
    "/api/data/estimate-cost",
    () => normalizeDataCost({ estimated_cost_usd: 0, source: "unavailable" }, request.source),
    {
      method: "POST",
      body: JSON.stringify({
        roots: request.roots,
        start: request.start,
        end: request.end,
        schema: request.schema ?? "ohlcv-1d",
        source: request.source,
      }),
    },
    60_000,
  ).then((result) => ({
    ...result,
    data: normalizeDataCost(result.data, request.source),
  }));
}

function fallbackDataIngest(request: DataIngestRequest): DataIngestResult {
  return {
    ok: false,
    source: request.source,
    roots_ingested: [],
    bars_written: 0,
    start_date: request.start,
    end_date: request.end,
    duration_seconds: null,
    cost_estimate: null,
    continuous_series: null,
    warnings: [],
    message: "Market data ingest requires the FastAPI data API.",
  };
}

export function ingestDataCache(request: DataIngestRequest): Promise<ApiResult<DataIngestResult>> {
  return withFallback<unknown>(
    "/api/data/ingest",
    () => fallbackDataIngest(request),
    {
      method: "POST",
      body: JSON.stringify({
        source: request.source,
        roots: request.roots,
        start: request.start,
        end: request.end,
        schema: request.schema ?? "ohlcv-1d",
        allow_over_budget: request.allow_over_budget === true,
        confirm: request.confirm !== false,
      }),
    },
    300_000,
  ).then((result) => {
    if (result.source !== "live") {
      return { ...result, data: fallbackDataIngest(request) };
    }
    const raw = isRecord(result.data) ? result.data : {};
    const cost = isRecord(raw.cost_estimate)
      ? normalizeDataCost(raw.cost_estimate, request.source)
      : null;
    return {
      ...result,
      data: {
        ok: true,
        source: asMessage(raw.source ?? request.source) || request.source,
        roots_ingested: asRebalanceArray(raw.roots_ingested).map((root) => asMessage(root)).filter(Boolean),
        bars_written: typeof raw.bars_written === "number" ? raw.bars_written : 0,
        start_date: asMessage(raw.start_date ?? raw.start ?? "") || null,
        end_date: asMessage(raw.end_date ?? raw.end ?? "") || null,
        duration_seconds:
          typeof raw.duration_seconds === "number" ? raw.duration_seconds : null,
        cost_estimate: cost,
        continuous_series: asRebalanceArray(raw.continuous_series),
        warnings: asRebalanceArray(raw.warnings).map((warning) => asMessage(warning)).filter(Boolean),
        message: asMessage(raw.message ?? "") || "Cache backfill completed.",
      },
    };
  });
}

// ---------------------------------------------------------------------------
// Phase 11A — Rebalance Book (runner preview / execute)
// ---------------------------------------------------------------------------

function asRebalanceNumber(value: unknown, fallback = 0): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function asRebalanceArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function normalizeRebalancePositions(value: unknown): RebalancePosition[] {
  if (Array.isArray(value)) {
    return value.map((entry) => {
      const row = isRecord(entry) ? entry : {};
      return {
        symbol: asMessage(row.symbol ?? row.instrument_id ?? row.root ?? row.contract ?? ""),
        quantity: asRebalanceNumber(
          row.quantity ?? row.qty ?? row.current_quantity ?? row.target_quantity ?? row.net_quantity,
          0,
        ),
      };
    });
  }
  if (isRecord(value)) {
    return Object.entries(value).map(([symbol, quantity]) => ({
      symbol,
      quantity: asRebalanceNumber(quantity, 0),
    }));
  }
  return [];
}

function normalizeRebalanceOrders(value: unknown): RebalanceOrderPreview[] {
  return asRebalanceArray(value).map((entry) => {
    const row = isRecord(entry) ? entry : {};
    const rawSide = asMessage(row.side ?? row.action ?? "").toUpperCase();
    return {
      // Prefer the root/instrument id for the compact `BUY +2 ES` receipt;
      // the backend's `symbol` may be the full dated contract symbol.
      symbol: asMessage(row.instrument_id ?? row.root ?? row.symbol ?? ""),
      side: rawSide,
      quantity: asRebalanceNumber(row.quantity ?? row.qty ?? row.contracts, 0),
      order_type: asMessage(row.order_type ?? row.type ?? "") || null,
      est_price: typeof row.est_price === "number" ? row.est_price : null,
      est_margin_impact:
        typeof row.est_margin_impact === "number" ? row.est_margin_impact : null,
      pre_trade_gate_passed:
        typeof row.pre_trade_gate_passed === "boolean"
          ? row.pre_trade_gate_passed
          : typeof row.gate_passed === "boolean"
            ? row.gate_passed
            : typeof row.approved === "boolean"
              ? row.approved
              : null,
      rejection_reason:
        asMessage(row.rejection_reason ?? row.reason ?? row.gate_reason ?? "") || null,
    };
  });
}

function normalizeRebalancePreview(raw: unknown, strategyId: string): RebalancePreviewResponse {
  const root = isRecord(raw) ? raw : {};
  const proposedOrders = normalizeRebalanceOrders(root.proposed_orders ?? root.orders);
  const rejected = proposedOrders.filter((order) => order.pre_trade_gate_passed === false);
  const gateRoot = isRecord(root.pre_trade_gate)
    ? root.pre_trade_gate
    : isRecord(root.gate)
      ? root.gate
      : null;
  const gateReasons = gateRoot
    ? asRebalanceArray(gateRoot.reasons ?? gateRoot.rejection_reasons)
        .map((reason) => asMessage(reason))
        .filter(Boolean)
    : rejected.map((order) => order.rejection_reason).filter(Boolean) as string[];
  const gatePassed =
    gateRoot && typeof gateRoot.passed === "boolean"
      ? gateRoot.passed
      : rejected.length === 0 && proposedOrders.every((order) => order.pre_trade_gate_passed !== false);
  const trackingRaw = root.tracking_error_pct ?? root.tracking_error;
  return {
    strategy_id: asMessage(root.strategy_id) || strategyId,
    account_nlv: asRebalanceNumber(root.account_nlv ?? root.nlv, 0),
    account_currency: asMessage(root.account_currency ?? root.currency ?? "USD") || "USD",
    current_positions: normalizeRebalancePositions(root.current_positions ?? root.positions),
    target_contracts: normalizeRebalancePositions(root.target_contracts ?? root.targets),
    proposed_orders: proposedOrders,
    projected_margin_utilization: asRebalanceNumber(
      root.projected_margin_utilization ?? root.margin_utilization,
      0,
    ),
    margin_utilization_cap:
      typeof root.margin_utilization_cap === "number" ? root.margin_utilization_cap : null,
    tracking_error_pct:
      typeof trackingRaw === "number" && Number.isFinite(trackingRaw) ? trackingRaw : null,
    mode: asMessage(root.mode ?? root.execution_mode ?? "offline") || "offline",
    message: asMessage(root.message ?? "") || null,
    warnings: asRebalanceArray(root.warnings).map((warning) => asMessage(warning)).filter(Boolean),
    strategy_roots: asRebalanceArray(root.strategy_roots).map((root) => asMessage(root)).filter(Boolean),
    generated_at: asMessage(root.generated_at ?? "") || null,
    gate: gateRoot || rejected.length > 0
      ? {
          passed: gatePassed,
          reasons: gateReasons as string[],
        }
      : null,
  };
}

function fallbackRebalancePreview(strategyId: string): RebalancePreviewResponse {
  return {
    strategy_id: strategyId,
    account_nlv: 0,
    account_currency: "USD",
    current_positions: [],
    target_contracts: [],
    proposed_orders: [],
    projected_margin_utilization: 0,
    tracking_error_pct: null,
    mode: "offline",
    message: "Runner bridge unavailable — no rebalance preview could be computed.",
    gate: null,
  };
}

export function previewRebalance(
  strategyId: string,
  roots?: string[],
): Promise<ApiResult<RebalancePreviewResponse>> {
  return withFallback<unknown>(
    "/api/runner/rebalance-preview",
    () => fallbackRebalancePreview(strategyId),
    {
      method: "POST",
      body: JSON.stringify({
        strategy_id: strategyId,
        ...(roots && roots.length > 0 ? { roots } : {}),
        dry_run: true,
      }),
    },
    60_000,
  ).then((result) => ({
    ...result,
    data: normalizeRebalancePreview(result.data, strategyId),
  }));
}

function fallbackRebalanceExecute(strategyId: string): RebalanceExecuteResponse {
  return {
    strategy_id: strategyId,
    applied: false,
    mode: "offline",
    submitted_order_ids: [],
    submitted_orders: [],
    fills: [],
    errors: ["Execute requires the runner API — offline preview cannot submit orders."],
    message: "Execute requires the runner API.",
  };
}

export function executeRebalance(
  strategyId: string,
  roots?: string[],
): Promise<ApiResult<RebalanceExecuteResponse>> {
  return withFallback<unknown>(
    "/api/runner/rebalance-execute",
    () => fallbackRebalanceExecute(strategyId),
    {
      method: "POST",
      body: JSON.stringify({
        strategy_id: strategyId,
        ...(roots && roots.length > 0 ? { roots } : {}),
        confirm: true,
      }),
    },
    120_000,
  ).then((result) => {
    if (result.source !== "live") {
      return { ...result, data: fallbackRebalanceExecute(strategyId) };
    }
    const root = isRecord(result.data) ? result.data : {};
    const submittedOrders = normalizeRebalanceOrders(root.submitted_orders ?? root.orders);
    const ids = asRebalanceArray(root.submitted_order_ids ?? root.order_ids)
      .map((id) => asMessage(id))
      .filter(Boolean);
    if (ids.length === 0) {
      ids.push(
        ...submittedOrders
          .map((order) => asMessage((order as unknown as Record<string, unknown>).order_id))
          .filter(Boolean),
      );
    }
    const fills = asRebalanceArray(root.fills).map((entry) => {
      const row = isRecord(entry) ? entry : {};
      return {
        order_id: asMessage(row.order_id ?? row.id ?? "") || null,
        symbol: asMessage(row.symbol ?? row.instrument_id ?? "") || null,
        side: asMessage(row.side ?? "") || null,
        quantity: typeof row.quantity === "number" ? row.quantity : null,
        price: typeof row.price === "number" ? row.price : null,
        status: asMessage(row.status ?? "") || null,
        message: asMessage(row.message ?? "") || null,
      };
    });
    return {
      ...result,
      data: {
        strategy_id: asMessage(root.strategy_id) || strategyId,
        applied:
          typeof root.applied === "boolean"
            ? root.applied
            : Boolean(root.executed ?? root.success ?? ids.length > 0),
        mode: asMessage(root.mode ?? root.execution_mode ?? "offline") || "offline",
        submitted_order_ids: ids,
        submitted_orders: submittedOrders,
        rejected_orders: normalizeRebalanceOrders(root.rejected_orders),
        fills,
        errors: asRebalanceArray(root.errors)
          .map((error) => asMessage(error))
          .filter(Boolean),
        account_nlv: typeof root.account_nlv === "number" ? root.account_nlv : null,
        account_currency: asMessage(root.account_currency ?? "") || null,
        status: asMessage(root.status ?? "") || null,
        projected_margin_utilization:
          typeof root.projected_margin_utilization === "number"
            ? root.projected_margin_utilization
            : null,
        margin_utilization_cap:
          typeof root.margin_utilization_cap === "number" ? root.margin_utilization_cap : null,
        tracking_error_pct:
          typeof root.tracking_error_pct === "number" ? root.tracking_error_pct : null,
        updated_positions: normalizeRebalancePositions(root.updated_positions),
        bridge_updated:
          typeof root.bridge_updated === "boolean" ? root.bridge_updated : null,
        message: asMessage(root.message ?? "") || null,
      },
    };
  });
}

// ---------------------------------------------------------------------------
// QoL artifact management / operator operations
// ---------------------------------------------------------------------------

async function requestRaw(
  path: string,
  init?: RequestInit,
  timeoutMs = DEFAULT_TIMEOUT_MS,
): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`${apiBaseUrl()}${path}`, {
      ...init,
      signal: controller.signal,
      headers: {
        Accept: "*/*",
        ...(init?.body ? { "Content-Type": "application/json" } : {}),
        ...init?.headers,
      },
      cache: "no-store",
    });
    if (!response.ok) {
      throw new ApiError(`${response.status} ${response.statusText} · ${path}`, response.status);
    }
    return response;
  } finally {
    clearTimeout(timer);
  }
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export async function deleteOptimization(studyId: string): Promise<ApiResult<ArtifactMutationResult>> {
  return mutationRequest(`/api/optimize/${encodeURIComponent(studyId)}`, "DELETE", undefined, () =>
    artifactRequired("Optimization studies cannot be deleted without the FastAPI backend."),
  );
}

export async function deleteBacktest(runId: string): Promise<ApiResult<ArtifactMutationResult>> {
  return mutationRequest(`/api/backtests/${encodeURIComponent(runId)}`, "DELETE", undefined, () =>
    artifactRequired("Backtest runs cannot be deleted without the FastAPI backend."),
  );
}

export async function deleteStrategy(strategyId: string): Promise<ApiResult<ArtifactMutationResult>> {
  return mutationRequest(`/api/strategies/${encodeURIComponent(strategyId)}`, "DELETE", undefined, () =>
    artifactRequired("Strategies cannot be deleted without the FastAPI backend."),
  );
}

async function mutationRequest(
  path: string,
  method: string,
  body: unknown,
  fallback: () => ArtifactMutationResult,
  timeoutMs = 30_000,
): Promise<ApiResult<ArtifactMutationResult>> {
  try {
    const response = await requestRaw(
      path,
      { method, ...(body !== undefined ? { body: JSON.stringify(body) } : {}) },
      timeoutMs,
    );
    const text = await response.text();
    let parsed: unknown = null;
    try {
      parsed = text.length > 0 ? JSON.parse(text) : null;
    } catch {
      parsed = { message: text };
    }
    const message =
      parsed && typeof parsed === "object" && "message" in parsed
        ? asMessage((parsed as Record<string, unknown>).message)
        : `${method === "DELETE" ? "Deleted" : "Updated"} successfully.`;
    return {
      data: {
        ok: true,
        message,
        status:
          parsed && typeof parsed === "object" && "status" in parsed
            ? asMessage((parsed as Record<string, unknown>).status)
            : "ok",
      },
      source: "live",
      error: null,
    };
  } catch (error) {
    return { data: fallback(), source: "demo", error: errorMessage(error) };
  }
}

function artifactRequired(message: string): ArtifactMutationResult {
  return { ok: false, message };
}

function asMessage(value: unknown): string {
  return typeof value === "string" ? value : value === null || value === undefined ? "" : String(value);
}

export async function cloneStrategy(
  strategyId: string,
  newStrategyId: string,
): Promise<ApiResult<StrategyMutationResult>> {
  return strategyMutation(`/api/strategies/${encodeURIComponent(strategyId)}/clone`, newStrategyId, "clone");
}

export async function renameStrategy(
  strategyId: string,
  newStrategyId: string,
): Promise<ApiResult<StrategyMutationResult>> {
  return strategyMutation(`/api/strategies/${encodeURIComponent(strategyId)}/rename`, newStrategyId, "rename");
}

async function strategyMutation(
  path: string,
  newStrategyId: string,
  operation: "clone" | "rename",
): Promise<ApiResult<StrategyMutationResult>> {
  try {
    const raw = await requestRaw(
      path,
      { method: "POST", body: JSON.stringify({ new_strategy_id: newStrategyId }) },
      30_000,
    );
    const parsed = (await raw.json()) as unknown;
    const strategy = normalizeStrategyConfig(parsed);
    if (!strategy) {
      return {
        data: { ok: false, strategy: null, message: "The API returned an unrecognized strategy payload." },
        source: "demo",
        error: "unrecognized strategy payload",
      };
    }
    return {
      data: { ok: true, strategy, message: `${operation === "clone" ? "Cloned" : "Renamed"} to ${strategy.id}.` },
      source: "live",
      error: null,
    };
  } catch (error) {
    return {
      data: {
        ok: false,
        strategy: null,
        message:
          operation === "clone"
            ? "Cloning requires the FastAPI backend."
            : "Renaming requires the FastAPI backend.",
      },
      source: "demo",
      error: errorMessage(error),
    };
  }
}

/** Download a raw artifact (YAML/CSV/JSON) through the browser. */
export async function downloadArtifact(
  path: string,
  fallbackFilename: string,
): Promise<{ ok: boolean; error: string | null }> {
  try {
    const response = await requestRaw(path, { method: "GET" }, 30_000);
    const blob = await response.blob();
    const disposition = response.headers.get("content-disposition") ?? "";
    const match = /filename\*?=(?:UTF-8''|")?([^";]+)/i.exec(disposition);
    const filename = (match?.[1] ?? fallbackFilename).replace(/"/g, "").trim() || fallbackFilename;
    if (typeof window === "undefined") throw new Error("downloads require a browser");
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    anchor.rel = "noopener";
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2_000);
    return { ok: true, error: null };
  } catch (error) {
    return { ok: false, error: errorMessage(error) };
  }
}

export function downloadStrategyYaml(strategyId: string): Promise<{ ok: boolean; error: string | null }> {
  return downloadArtifact(
    `/api/strategies/${encodeURIComponent(strategyId)}/yaml`,
    `${strategyId}.yaml`,
  );
}

export function exportTearSheet(runId: string): Promise<{ ok: boolean; error: string | null }> {
  return downloadArtifact(
    `/api/backtests/${encodeURIComponent(runId)}/export.json`,
    `${runId}-tearsheet.json`,
  );
}

export function exportTradesCsv(runId: string): Promise<{ ok: boolean; error: string | null }> {
  return downloadArtifact(
    `/api/backtests/${encodeURIComponent(runId)}/trades.csv`,
    `${runId}-trades.csv`,
  );
}

export async function resetRiskLock(
  options: { operatorToken?: string } = {},
): Promise<ApiResult<RiskLockResetResult>> {
  try {
    const response = await requestRaw(
      "/api/risk/reset-lock",
      {
        method: "POST",
        body: JSON.stringify({
          confirm: "RESET RISK LOCK",
          operator_token: options.operatorToken?.trim() || null,
          reset_persisted_state: true,
        }),
      },
      30_000,
    );
    const parsed = (await response.json()) as Record<string, unknown>;
    return {
      data: {
        ok: true,
        message: asMessage(parsed.message) || "Risk lock reset accepted.",
        kill_switch_active:
          typeof parsed.kill_switch_active === "boolean" ? parsed.kill_switch_active : null,
        state: null,
      },
      source: "live",
      error: null,
    };
  } catch (error) {
    return {
      data: { ok: false, message: "Risk-lock reset requires the FastAPI backend." },
      source: "demo",
      error: errorMessage(error),
    };
  }
}

export async function resetPaperState(
  options: { confirm: string; clearRiskState: boolean; clearRunnerBridge: boolean; clearPaperCache: boolean },
): Promise<ApiResult<PaperStateResetResult>> {
  try {
    const response = await requestRaw(
      "/api/maintenance/reset-paper-state",
      {
        method: "POST",
        body: JSON.stringify({
          confirm: options.confirm,
          clear_risk_state: options.clearRiskState,
          clear_runner_bridge: options.clearRunnerBridge,
          clear_paper_cache: options.clearPaperCache,
        }),
      },
      60_000,
    );
    const parsed = (await response.json()) as Record<string, unknown>;
    return {
      data: {
        ok: true,
        message: asMessage(parsed.message) || "Paper state reset.",
        receipts:
          parsed.receipts && typeof parsed.receipts === "object"
            ? (parsed.receipts as Record<string, unknown> | unknown[])
            : null,
      },
      source: "live",
      error: null,
    };
  } catch (error) {
    return {
      data: { ok: false, message: "Clearing paper state requires the FastAPI backend." },
      source: "demo",
      error: errorMessage(error),
    };
  }
}

export async function resetRunnerBridge(): Promise<ApiResult<ArtifactMutationResult>> {
  try {
    const response = await requestRaw(
      "/api/runner/reset",
      { method: "POST", body: JSON.stringify({}) },
      30_000,
    );
    const parsed = (await response.json()) as Record<string, unknown>;
    const stopped = parsed.session_running === false;
    return {
      data: {
        ok: true,
        message: stopped
          ? "Runner bridge reset — offline session closed."
          : "Runner bridge reset requested.",
        status: asMessage(parsed.mode) || "ok",
      },
      source: "live",
      error: null,
    };
  } catch (error) {
    return {
      data: artifactRequired("Runner bridge reset requires the FastAPI backend."),
      source: "demo",
      error: errorMessage(error),
    };
  }
}

// ---------------------------------------------------------------------------
// Telemetry WebSocket
// ---------------------------------------------------------------------------

export interface TelemetryConnection {
  close: () => void;
}

/**
 * Subscribe to `/ws/telemetry`.
 *
 * On the first socket error/failure the hook starts the deterministic local
 * simulator so the Command Center and Risk views still animate in demo mode.
 * The caller gets `onStatus("live" | "demo")` transitions so the shell can show
 * the `UI FALLBACK` data-mode badge.
 */
export function connectTelemetry(options: {
  onMessage: (message: TelemetryEnvelope) => void;
  onStatus: (status: ConnectionState) => void;
}): TelemetryConnection {
  const simulator = createTelemetrySimulator({
    netLiquidation: 1_250_000,
    initialMargin: 463_200,
    marginUtilization: 0.371,
  });

  let socket: WebSocket | null = null;
  let closed = false;
  let retries = 0;
  let simulatorTimer: ReturnType<typeof setInterval> | null = null;

  const startSimulator = () => {
    if (simulatorTimer !== null || closed) return;
    options.onStatus("demo");
    simulatorTimer = setInterval(() => {
      options.onMessage(simulator());
    }, 1_400);
  };

  const scheduleRetry = () => {
    if (closed) return;
    startSimulator();
    if (retries >= 5) return;
    retries += 1;
    window.setTimeout(() => {
      if (closed) return;
      try {
        open();
      } catch {
        /* stay on the deterministic simulator */
      }
    }, retries * 2_000);
  };

  function open(): void {
    if (closed) return;
    options.onStatus("connecting");
    try {
      socket = new WebSocket(wsTelemetryUrl());
    } catch {
      scheduleRetry();
      return;
    }
    socket.onopen = () => {
      if (closed) {
        socket?.close();
        return;
      }
      retries = 0;
      options.onStatus("live");
      if (simulatorTimer !== null) {
        clearInterval(simulatorTimer);
        simulatorTimer = null;
      }
    };
    socket.onmessage = (event: MessageEvent<string>) => {
      try {
        const parsed = JSON.parse(event.data) as TelemetryEnvelope;
        if (parsed && parsed.type === "telemetry") options.onMessage(parsed);
      } catch {
        /* ignore malformed frames, keep the stream alive */
      }
    };
    socket.onerror = () => {
      options.onStatus("demo");
    };
    socket.onclose = () => {
      socket = null;
      if (closed) return;
      scheduleRetry();
    };
  }

  open();

  return {
    close: () => {
      closed = true;
      if (simulatorTimer !== null) clearInterval(simulatorTimer);
      socket?.close();
      socket = null;
      options.onStatus("offline");
    },
  };
}


// ---------------------------------------------------------------------------
// Instrument universe (Strategy Studio, task-32)
// ---------------------------------------------------------------------------

export function getInstruments(): Promise<ApiResult<InstrumentsResponse>> {
  const fallback = mockInstruments();
  return withFallback<unknown>("/api/instruments", () => fallback).then((result) => ({
    ...result,
    data: normalizeInstrumentsResponse(result.data, fallback),
  }));
}

// ---------------------------------------------------------------------------
// Strategy Studio (Phase 10)
// ---------------------------------------------------------------------------

export function getStudioCatalog(): Promise<ApiResult<StudioCatalogResponse>> {
  const fallback = mockStudioCatalog();
  return withFallback<unknown>("/api/strategies/schema/catalog", () => fallback).then((result) => ({
    ...result,
    data: normalizeStudioCatalog(result.data, fallback),
  }));
}

/**
 * Create (POST) or update (PUT) a strategy YAML. The Manual Builder uses this
 * for "Save Strategy YAML"; the API persists to configs/strategies or
 * artifacts/strategies and re-validates the full FinRL-X pipeline contract.
 */
export function saveStudioStrategy(
  draft: StudioDraft,
  options: { update?: boolean; encodings?: YamlEncodingMap } = {},
): Promise<ApiResult<StudioSaveResult>> {
  const payload = draftToPayload(draft, options.encodings);
  const path = options.update ? `/api/strategies/${encodeURIComponent(draft.id)}` : "/api/strategies";
  return withFallback<unknown>(
    path,
    () => ({ strategy: mockSavedStrategy(payload), persisted: false, message: "UI FALLBACK: strategy saved in memory only." }),
    { method: options.update ? "PUT" : "POST", body: JSON.stringify(payload) },
    15_000,
  ).then((result) => {
    if (result.source === "demo") return result as ApiResult<unknown> as ApiResult<StudioSaveResult>;
    const normalized = normalizeStrategyConfig(isRecord(result.data) ? (result.data.strategy ?? result.data) : result.data);
    if (!normalized) {
      return {
        data: {
          strategy: mockSavedStrategy(payload),
          persisted: false,
          message: "API returned an unrecognized strategy payload; kept the local draft.",
        },
        source: "demo" as const,
        error: "unrecognized strategy payload",
      };
    }
    return {
      data: {
        strategy: normalized,
        persisted: true,
        message: `Saved ${normalized.id} to the strategy catalog.`,
      },
      source: "live" as const,
      error: null,
    };
  });
}

// ---------------------------------------------------------------------------
// Optuna auto-optimizer (Phase 10)
// ---------------------------------------------------------------------------

export function listOptimizations(): Promise<ApiResult<OptimizationListResponse>> {
  return withFallback<unknown>("/api/optimize", mockOptimizationList).then((result) => ({
    ...result,
    data:
      result.source === "live"
        ? normalizeOptimizationList(result.data)
        : (result.data as OptimizationListResponse),
  }));
}

export function getOptimization(
  studyId: string,
): Promise<ApiResult<OptimizationStudy | null>> {
  return withFallback<unknown>(`/api/optimize/${encodeURIComponent(studyId)}`, () =>
    mockOptimizationStudy(),
  ).then((result) => {
    // v1.5.2: the API quarantines legacy studies; a 404 means "archived", not
    // "render a fake fixture study".
    if (result.source === "demo" && /\b404\b/.test(result.error ?? "")) {
      return {
        data: null,
        source: "live" as const,
        error: "archived",
      };
    }
    const normalized = normalizeOptimizationStudy(result.data);
    if (normalized) return { ...result, data: normalized };
    const fallback = mockOptimizationStudy();
    return {
      data: { ...fallback, study_id: studyId },
      source: "demo" as const,
      error: "unrecognized optimization payload",
    };
  });
}

export function runOptimization(
  request: OptimizationRequest,
): Promise<ApiResult<OptimizationStudy>> {
  return withFallback<unknown>(
    "/api/optimize",
    () => mockOptimizationStudy(request),
    { method: "POST", body: JSON.stringify(request) },
    120_000,
  ).then((result) => {
    if (result.source === "demo") return result as ApiResult<unknown> as ApiResult<OptimizationStudy>;
    const normalized = normalizeOptimizationStudy(result.data);
    if (normalized) return { ...result, data: normalized };
    const fallback = mockOptimizationStudy(request);
    return { data: fallback, source: "demo" as const, error: "unrecognized optimization payload" };
  });
}

export function applyOptimization(
  studyId: string,
  options: { newStrategyId?: string; verificationBacktest?: boolean } = {},
): Promise<ApiResult<OptimizationApplyResult>> {
  const body = {
    new_strategy_id: options.newStrategyId,
    verification_backtest: options.verificationBacktest !== false,
  };
  return withFallback<unknown>(
    `/api/optimize/${encodeURIComponent(studyId)}/apply`,
    () => mockOptimizationApply(studyId),
    { method: "POST", body: JSON.stringify(body) },
    300_000,
  ).then((result) => {
    if (result.source === "demo" && /\b404\b/.test(result.error ?? "")) {
      return {
        data: {
          study_id: studyId,
          strategy_id: "",
          saved_strategy_id: null,
          strategy_path: null,
          strategy: null,
          backtest_run_id: null,
          applied: false,
          message:
            "Legacy optimization run archived — pick a current study (the API no longer serves it).",
          verification_backtest: { status: "skipped", message: "archived" },
        },
        source: "live" as const,
        error: "archived",
      };
    }
    return {
      ...result,
      data: result.source === "live"
        ? normalizeOptimizationApply(result.data, studyId)
        : (result.data as OptimizationApplyResult),
    };
  });
}

// ---------------------------------------------------------------------------
// DRL neural trainer (Phase 10)
// ---------------------------------------------------------------------------

export function listDrlModels(): Promise<ApiResult<DrlModelsResponse>> {
  return withFallback<unknown>("/api/drl/models", mockDrlModels).then((result) => ({
    ...result,
    data:
      result.source === "live"
        ? normalizeDrlModelsResponse(result.data)
        : (result.data as DrlModelsResponse),
  }));
}

export function trainDrlModel(request: DrlTrainRequest): Promise<ApiResult<DrlTrainResult>> {
  return withFallback<unknown>(
    "/api/drl/train",
    () => mockDrlTrain(request),
    { method: "POST", body: JSON.stringify(request) },
    180_000,
  ).then((result) => {
    const normalized = normalizeDrlTrainResult(result.data);
    if (normalized) return { ...result, data: normalized };
    const fallback = mockDrlTrain(request);
    return { data: fallback, source: "demo" as const, error: "unrecognized DRL training payload" };
  });
}

/**
 * Bind a trained checkpoint to a strategy by writing the DRL allocator config
 * through the existing PUT /api/strategies/{id} editor.
 */
export function bindDrlModel(
  strategy: StrategyConfig,
  model: DrlModel,
): Promise<ApiResult<DrlBindResult>> {
  // Phase 11A: the checkpoint's .meta.json universe is the source of truth for
  // the strategy universe on bind. Preserve every other universe parameter and
  // only apply roots/top_n when the metadata is actually available.
  const modelRoots = (Array.isArray(model.universe) ? model.universe : [])
    .map((root) => String(root).trim())
    .filter(Boolean);
  const metadataAvailable = model.metadata_available !== false && modelRoots.length > 0;
  const universe = metadataAvailable
    ? {
        ...strategy.universe,
        params: {
          ...strategy.universe.params,
          roots: modelRoots.join(","),
          top_n: modelRoots.length,
        },
      }
    : strategy.universe;
  // Strip UI-only metadata returned by GET /api/strategies; the PUT contract
  // accepts the strategy stages/params only.
  const baseStrategy: StrategyConfig = { ...strategy };
  delete baseStrategy.builtin;
  delete baseStrategy.path;
  delete baseStrategy.strategy_path;
  delete baseStrategy.source;
  const payload: StrategyConfig = {
    ...baseStrategy,
    universe,
    allocator: {
      name: "drl",
      params: {
        algorithm: model.algorithm,
        total_timesteps: model.timesteps,
        model_path: model.path ?? `artifacts/models/${model.name}.zip`,
        policy: "MlpPolicy",
      },
    },
  };
  return withFallback<unknown>(
    `/api/strategies/${encodeURIComponent(strategy.id)}`,
    () => mockDrlBind(model, strategy.id),
    { method: "PUT", body: JSON.stringify(payload) },
  ).then((result) => {
    const universeSynced = result.source === "live" && metadataAvailable;
    return {
      ...result,
      data:
        result.source === "live"
          ? {
              model: model.name,
              strategy_id: strategy.id,
              bound: true,
              message: universeSynced
                ? `Bound ${model.name} to ${strategy.id} · universe synced (${modelRoots.length} roots).`
                : `Bound ${model.name} to ${strategy.id}${metadataAvailable ? "" : " · checkpoint metadata unavailable, universe unchanged."}`,
              universe_synced: universeSynced,
              synced_roots: metadataAvailable ? modelRoots : [],
              top_n: metadataAvailable ? modelRoots.length : null,
              metadata_available: metadataAvailable,
              warning: metadataAvailable
                ? null
                : "Checkpoint metadata unavailable — universe sync was skipped.",
              strategy: payload,
            }
          : {
              ...(result.data as DrlBindResult),
              universe_synced: false,
              synced_roots: metadataAvailable ? modelRoots : [],
              top_n: metadataAvailable ? modelRoots.length : null,
              metadata_available: metadataAvailable,
              warning: metadataAvailable
                ? "UI fallback — bind is in-memory only."
                : "Checkpoint metadata unavailable — universe sync was skipped.",
            },
    };
  });
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
