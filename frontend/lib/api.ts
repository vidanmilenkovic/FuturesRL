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
  mockRecentFills,
  mockRunnerStatus,
  mockTelemetryStatus,
  mockTearsheet,
  mockTermStructure,
  mockWorkingOrders,
} from "./mock";
import type {
  AccountSnapshotResponse,
  ApiResult,
  BacktestListResponse,
  ConnectionState,
  BacktestRunConfig,
  BacktestRunSummary,
  FlattenAllResponse,
  HealthResponse,
  PipelineWaterfallResponse,
  RiskStateResponse,
  RiskThresholdOverride,
  RiskThresholds,
  StrategyConfig,
  TearsheetResponse,
  FillsResponse,
  RunnerStatus,
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
  return withFallback("/api/positions", () => buildAccount());
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
