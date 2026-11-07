"use client";

import { useEffect, useMemo, useState } from "react";

import {
  Badge,
  Button,
  EmptyState,
  Select,
} from "@/components/ui/primitives";
import { Modal } from "@/components/ui/dialog";
import { useAccount } from "@/components/providers/account-provider";
import {
  executeRebalance,
  getAccount,
  getPipeline,
  getRecentFills,
  getRunnerStatus,
  getWorkingOrders,
  listStrategies,
  previewRebalance,
} from "@/lib/api";
import { cn } from "@/lib/cn";
import { formatCurrency, formatInteger, formatPctPlain } from "@/lib/format";
import type {
  RebalanceExecuteResponse,
  RebalanceOrderPreview,
  RebalancePosition,
  RebalancePreviewResponse,
} from "@/lib/types";
import { useAsyncData } from "@/lib/use-async-data";

const MARGIN_CAP = 0.6;

function marginFraction(value: number | null | undefined): number | null {
  if (typeof value !== "number" || !Number.isFinite(value)) return null;
  return value > 1.5 ? value / 100 : value;
}

function formatPercentValue(value: number | null | undefined): string {
  const fraction = marginFraction(value);
  return fraction === null ? "—" : `${(fraction * 100).toFixed(1)}%`;
}

function formatOrder(order: RebalanceOrderPreview): string {
  const side = String(order.side ?? "").toUpperCase() === "SELL" ? "SELL" : "BUY";
  const sign = side === "SELL" ? "-" : "+";
  return `${side} ${sign}${Math.abs(order.quantity)} ${order.symbol || "?"}`;
}

function positionMap(rows: RebalancePosition[]): Map<string, number> {
  const map = new Map<string, number>();
  for (const row of rows) {
    if (!row.symbol) continue;
    map.set(row.symbol, (map.get(row.symbol) ?? 0) + row.quantity);
  }
  return map;
}

/**
 * Phase 11A Rebalance Book control bar: whatIf preview of the runner's target
 * book, PreTradeRiskGate diagnostics, then a guarded paper/offline execute.
 * Shared by Overview and Weights.
 */
export function RebalanceBar({
  className,
  onExecuted,
}: {
  className?: string;
  onExecuted?: () => void;
}) {
  const strategiesSource = useAsyncData(listStrategies);
  const { reload: reloadAccount } = useAccount();
  const strategies = useMemo(() => strategiesSource.data ?? [], [strategiesSource.data]);
  const [strategyId, setStrategyId] = useState("");
  const [preview, setPreview] = useState<RebalancePreviewResponse | null>(null);
  const [previewOpen, setPreviewOpen] = useState(false);
  const [previewing, setPreviewing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [confirmChecked, setConfirmChecked] = useState(false);
  const [executing, setExecuting] = useState(false);
  const [execResult, setExecResult] = useState<RebalanceExecuteResponse | null>(null);
  const [execError, setExecError] = useState<string | null>(null);

  useEffect(() => {
    if (strategyId || strategies.length === 0) return;
    const preferred =
      strategies.find((strategy) => strategy.id === "tsmom_vol_target") ?? strategies[0];
    setStrategyId(preferred?.id ?? "");
  }, [strategies, strategyId]);

  const strategy = strategies.find((entry) => entry.id === strategyId) ?? null;
  const selectedRoots = strategy ? rootsFromStrategy(strategy) : [];

  const runPreview = async () => {
    if (!strategyId) return;
    setPreviewing(true);
    setError(null);
    setExecResult(null);
    setExecError(null);
    setConfirmChecked(false);
    const result = await previewRebalance(strategyId, selectedRoots);
    setPreviewing(false);
    if (result.source !== "live") {
      setError(describeRunError(result.error, result.data.message || "Rebalance preview failed."));
      return;
    }
    setPreview(result.data);
    setPreviewOpen(true);
  };

  const runExecute = async () => {
    if (!preview || !strategyId) return;
    setExecuting(true);
    setExecError(null);
    const result = await executeRebalance(strategyId, selectedRoots);
    setExecuting(false);
    if (result.source !== "live") {
      setExecError(describeRunError(result.error, result.data.message || "Execute failed."));
      return;
    }
    setExecResult(result.data);
    if (!result.data.applied) {
      setExecError(result.data.message || result.data.errors[0] || "Rebalance execute did not apply.");
    }
    // Refresh the broker/runner surfaces immediately: positions, working orders,
    // fills, runner bridge status and the pipeline trace.
    await Promise.allSettled([
      reloadAccount(),
      getAccount(),
      getWorkingOrders(),
      getRecentFills(),
      getRunnerStatus(),
      getPipeline(),
    ]);
    if (typeof window !== "undefined") {
      window.dispatchEvent(new Event("futuresrl:refresh-execution"));
    }
    onExecuted?.();
  };

  const current = useMemo(() => positionMap(preview?.current_positions ?? []), [preview]);
  const target = useMemo(() => positionMap(preview?.target_contracts ?? []), [preview]);
  const bookRows = useMemo(() => {
    const symbols = [...new Set([...current.keys(), ...target.keys()])].sort();
    return symbols.map((symbol) => ({
      symbol,
      current: current.get(symbol) ?? 0,
      target: target.get(symbol) ?? 0,
      delta: (target.get(symbol) ?? 0) - (current.get(symbol) ?? 0),
    }));
  }, [current, target]);
  const orders = preview?.proposed_orders ?? [];
  const rejectedOrders = orders.filter((order) => order.pre_trade_gate_passed === false);
  const approvedOrders = orders.filter((order) => order.pre_trade_gate_passed !== false);
  const gatePassed = preview?.gate?.passed ?? (rejectedOrders.length === 0 && orders.length > 0);
  const margin = marginFraction(preview?.projected_margin_utilization);
  const marginCap = marginFraction(preview?.margin_utilization_cap) ?? MARGIN_CAP;
  const canExecute = Boolean(
    preview && orders.length > 0 && approvedOrders.length > 0 && !executing && !execResult,
  );

  return (
    <section
      className={cn("panel shrink-0 px-3 py-2", className)}
      data-testid="rebalance-bar"
      aria-label="Rebalance Book"
    >
      <div className="flex flex-wrap items-center gap-2">
        <div className="flex items-center gap-2">
          <span className="text-[12px] font-medium text-zinc-200">Rebalance Book</span>
          <Badge tone={preview?.mode === "ibkr-paper" ? "gain" : "warn"}>
            {preview?.mode === "ibkr-paper" ? "IBKR paper" : preview ? "offline" : "whatIf"}
          </Badge>
        </div>
        <Select
          value={strategyId}
          onChange={setStrategyId}
          options={strategies.map((entry) => ({ value: entry.id, label: entry.id }))}
          className="!h-7 !w-[190px] shrink-0 text-[11px]"
          id="rebalance-strategy"
        />
        <Button
          variant="secondary"
          size="xs"
          testId="preview-rebalance"
          onClick={() => void runPreview()}
          disabled={!strategyId || previewing || executing}
          title="Build the target book and run the PreTradeRiskGate without submitting orders"
        >
          {previewing ? (
            <>
              <span aria-hidden className="h-3 w-3 animate-spin rounded-full border-2 border-white/30 border-t-white" />
              Previewing…
            </>
          ) : (
            "Preview Rebalance (whatIf)"
          )}
        </Button>
        {preview ? (
          <Button
            variant="primary"
            size="xs"
            testId="open-execute-rebalance"
            onClick={() => {
              setConfirmChecked(false);
              setExecError(null);
              setPreviewOpen(true);
            }}
            disabled={!canExecute || executing}
            title="Execute the approved paper/offline rebalance"
          >
            Execute Paper Rebalance
          </Button>
        ) : null}
        <span className="min-w-0 truncate text-[10px] text-zinc-600">
          {strategy ? `${strategy.universe.name} · ${selectedRoots.length} roots` : "select a strategy"}
        </span>
        {preview ? (
          <span className="ml-auto flex flex-wrap items-center gap-3 text-[10.5px] text-zinc-500">
            <span>NLV <span className="font-mono text-zinc-300">{formatCurrency(preview.account_nlv)}</span></span>
            <span>Margin <span className="font-mono text-zinc-300">{formatPercentValue(preview.projected_margin_utilization)}</span> / {(marginCap * 100).toFixed(0)}%</span>
            <span>Tracking <span className="font-mono text-zinc-300">{preview.tracking_error_pct === null ? "n/a" : `${preview.tracking_error_pct.toFixed(2)}%`}</span></span>
            <Badge tone={gatePassed ? "gain" : "warn"}>{gatePassed ? "Gate passed" : `${rejectedOrders.length} rejected`}</Badge>
          </span>
        ) : null}
      </div>
      {error ? (
        <p role="alert" data-testid="rebalance-error" className="mt-1.5 text-[10.5px] text-amber-300">
          {error}
        </p>
      ) : null}

      <Modal
        open={previewOpen}
        title={`Rebalance preview · ${preview?.strategy_id ?? strategyId}`}
        description={
          <span>
            whatIf target book for the runner bridge. Submit only after reviewing margin, gate status and delta
            orders.
          </span>
        }
        onClose={() => {
          if (executing) return;
          setPreviewOpen(false);
        }}
        busy={executing}
        width="max-w-4xl"
        footer={
          <>
            <Button variant="ghost" onClick={() => setPreviewOpen(false)} disabled={executing}>
              Close
            </Button>
            {execResult ? null : (
              <Button
                variant="danger"
                testId="execute-rebalance"
                onClick={() => void runExecute()}
                disabled={!canExecute || !confirmChecked}
                title={
                  !confirmChecked
                    ? "Confirm the paper/offline execution first"
                    : !canExecute
                      ? "No approved delta orders to execute"
                      : "Submit approved delta orders"
                }
              >
                {executing ? (
                  <>
                    <span aria-hidden className="h-3 w-3 animate-spin rounded-full border-2 border-white/30 border-t-white" />
                    Executing…
                  </>
                ) : (
                  "Execute Paper Rebalance"
                )}
              </Button>
            )}
          </>
        }
      >
        {preview ? (
          <div className="max-h-[66vh] space-y-2.5 overflow-y-auto pr-1" data-testid="rebalance-preview-modal">
            <div className="flex flex-wrap items-center gap-3 rounded-md border border-edge bg-canvas/60 px-2.5 py-2 text-[11px]">
              <span data-testid="rebalance-mode">
                <Badge tone={preview.mode === "ibkr-paper" ? "gain" : "warn"}>
                  {preview.mode === "ibkr-paper" ? "IBKR Paper" : "Offline simulated"}
                </Badge>
              </span>
              <span data-testid="rebalance-account">NLV <span className="font-mono text-zinc-200">{formatCurrency(preview.account_nlv)} {preview.account_currency}</span></span>
              <span data-testid="rebalance-margin">Margin <span className="font-mono text-zinc-200">{formatPercentValue(preview.projected_margin_utilization)}</span> <span className="text-zinc-600">vs {(marginCap * 100).toFixed(0)}% cap</span></span>
              <span data-testid="rebalance-tracking-error">Tracking error <span className="font-mono text-zinc-200">{preview.tracking_error_pct === null ? "n/a" : `${preview.tracking_error_pct.toFixed(2)}%`}</span></span>
              <span data-testid="rebalance-gate-status">
                <Badge tone={gatePassed ? "gain" : "warn"}>
                  {gatePassed ? "PreTradeRiskGate passed" : "PreTradeRiskGate rejected orders"}
                </Badge>
              </span>
              <span className="ml-auto flex items-center gap-2" data-testid="rebalance-margin-vs-cap">
                <span className="h-1.5 w-24 overflow-hidden rounded-full bg-zinc-800">
                  <span
                    className={cn("block h-full rounded-full", (margin ?? 0) > marginCap ? "bg-loss" : "bg-gain")}
                    style={{ width: `${Math.min(100, (margin ?? 0) * 100).toFixed(1)}%` }}
                  />
                </span>
                <span className="font-mono text-zinc-500">{formatPctPlain(marginCap)} cap</span>
              </span>
            </div>
            {preview.mode !== "ibkr-paper" ? (
              <p className="rounded border border-warn/30 bg-warn-soft/30 px-2 py-1.5 text-[10.5px] text-amber-300" data-testid="rebalance-offline-note">
                Offline simulated broker — orders will not reach IBKR. Connect the IBKR paper bridge for a real paper book.
              </p>
            ) : null}
            {preview.message ? (
              <p className="text-[10.5px] text-zinc-500">{preview.message}</p>
            ) : null}
            {preview.warnings && preview.warnings.length > 0 ? (
              <ul className="space-y-0.5 rounded border border-warn/30 bg-warn-soft/20 px-2 py-1.5 text-[10px] text-amber-300" data-testid="rebalance-warnings">
                {preview.warnings.map((warning, index) => (
                  <li key={`${warning}-${index}`}>· {warning}</li>
                ))}
              </ul>
            ) : null}

            <div className="grid grid-cols-2 gap-2.5">
              <div className="rounded border border-edge bg-canvas/40">
                <div className="border-b border-hairline px-2 py-1.5 text-[10.5px] font-medium text-zinc-300">
                  Current vs target quantity
                </div>
                <div className="max-h-48 overflow-auto">
                  <table className="data-table" data-testid="rebalance-book-table">
                    <thead><tr><th>Symbol</th><th className="text-right">Current</th><th className="text-right">Target</th><th className="text-right">Delta</th></tr></thead>
                    <tbody>
                      {bookRows.length === 0 ? (
                        <tr><td colSpan={4} className="py-3 text-center text-[10.5px] text-zinc-600">Flat book</td></tr>
                      ) : bookRows.map((row) => (
                        <tr key={row.symbol}>
                          <td className="font-mono text-[10.5px] text-zinc-300">{row.symbol}</td>
                          <td className="num text-right text-zinc-400">{formatInteger(row.current)}</td>
                          <td className="num text-right text-zinc-200">{formatInteger(row.target)}</td>
                          <td className={cn("num text-right", row.delta > 0 ? "text-gain" : row.delta < 0 ? "text-loss" : "text-zinc-500")}>
                            {row.delta > 0 ? "+" : ""}{formatInteger(row.delta)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
              <div className="rounded border border-edge bg-canvas/40">
                <div className="border-b border-hairline px-2 py-1.5 text-[10.5px] font-medium text-zinc-300">
                  PreTradeRiskGate
                </div>
                <div className="max-h-48 overflow-auto" data-testid="rebalance-gate-reasons">
                  {orders.length === 0 ? (
                    <p className="px-2 py-3 text-[10.5px] text-zinc-600">No delta orders after the gate.</p>
                  ) : (
                    <ul className="space-y-1 px-2 py-1.5">
                      {orders.map((order, index) => (
                        <li key={`${order.symbol}-${index}`} className="flex items-start gap-1.5 text-[10.5px]">
                          <Badge tone={order.pre_trade_gate_passed === false ? "loss" : "gain"}>
                            {order.pre_trade_gate_passed === false ? "REJECTED" : "PASS"}
                          </Badge>
                          <span className="min-w-0 flex-1 text-zinc-400">
                            {formatOrder(order)}
                            {order.rejection_reason ? <span className="text-rose-300"> · {order.rejection_reason}</span> : null}
                          </span>
                        </li>
                      ))}
                    </ul>
                  )}
                  {preview.gate?.reasons.length ? (
                    <p className="border-t border-hairline px-2 py-1 text-[10px] text-rose-300">
                      {preview.gate.reasons.join(" · ")}
                    </p>
                  ) : null}
                </div>
              </div>
            </div>

            <div className="rounded border border-edge bg-canvas/40">
              <div className="border-b border-hairline px-2 py-1.5 text-[10.5px] font-medium text-zinc-300">
                Delta orders · {orders.length}
              </div>
              <div className="max-h-44 overflow-auto">
                <table className="data-table" data-testid="rebalance-orders-table">
                  <thead><tr><th>Order</th><th>Type</th><th className="text-right">Est price</th><th className="text-right">Est margin</th><th>Gate</th></tr></thead>
                  <tbody>
                    {orders.length === 0 ? (
                      <tr><td colSpan={5} className="py-3 text-center text-[10.5px] text-zinc-600">No orders proposed.</td></tr>
                    ) : orders.map((order, index) => (
                      <tr key={`${order.symbol}-${order.side}-${index}`}>
                        <td className="font-mono text-[10.5px] text-zinc-200">{formatOrder(order)}</td>
                        <td className="text-[10.5px] text-zinc-500">{order.order_type ?? "MKT"}</td>
                        <td className="num text-right text-zinc-400">{order.est_price === null || order.est_price === undefined ? "—" : formatNumber(order.est_price)}</td>
                        <td className="num text-right text-zinc-400">{order.est_margin_impact === null || order.est_margin_impact === undefined ? "—" : formatCurrency(order.est_margin_impact)}</td>
                        <td>
                          <Badge tone={order.pre_trade_gate_passed === false ? "loss" : "gain"}>
                            {order.pre_trade_gate_passed === false ? "REJECTED" : "PASS"}
                          </Badge>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {!execResult ? (
              <label className="flex items-start gap-2 rounded border border-warn/30 bg-warn-soft/20 px-2.5 py-2 text-[10.5px] leading-4 text-zinc-300">
                <input
                  type="checkbox"
                  checked={confirmChecked}
                  onChange={(event) => setConfirmChecked(event.target.checked)}
                  className="mt-0.5 h-3 w-3 rounded border-zinc-600 bg-canvas accent-red-500"
                  data-testid="rebalance-confirm-checkbox"
                />
                I confirm submitting the approved delta orders to the {preview.mode === "ibkr-paper" ? "IBKR paper" : "offline simulated"} broker.
              </label>
            ) : null}

            {execError ? (
              <p role="alert" data-testid="rebalance-exec-error" className="rounded border border-loss/30 bg-loss-soft/40 px-2 py-1.5 text-[10.5px] text-rose-300">
                {execError}
              </p>
            ) : null}
            {execResult ? (
              <div
                data-testid="rebalance-exec-result"
                data-applied={execResult.applied ? "true" : "false"}
                className="rounded border border-gain/30 bg-gain-soft/20 px-2.5 py-2 text-[10.5px] text-zinc-300"
              >
                <p className="font-medium text-emerald-300">
                  {execResult.applied ? "Execute submitted" : "Execute did not apply"} · {execResult.mode}
                </p>
                <p className="mt-0.5">
                  Status: {execResult.status ?? "submitted"} · Orders:{" "}
                  {execResult.submitted_order_ids.length > 0 ? execResult.submitted_order_ids.join(", ") : "none"}
                </p>
                {execResult.fills.length > 0 ? (
                  <p className="mt-0.5">Fills: {execResult.fills.map((fill) => `${fill.symbol ?? "?"} ${fill.status ?? ""}`).join(", ")}</p>
                ) : null}
                {execResult.rejected_orders && execResult.rejected_orders.length > 0 ? (
                  <p className="mt-0.5 text-amber-300">
                    Rejected:{" "}
                    {execResult.rejected_orders
                      .map((order) => `${formatOrder(order)}${order.rejection_reason ? ` (${order.rejection_reason})` : ""}`)
                      .join(" · ")}
                  </p>
                ) : null}
                {execResult.bridge_updated !== null && execResult.bridge_updated !== undefined ? (
                  <p className="mt-0.5 text-zinc-500">
                    Bridge book {execResult.bridge_updated ? "updated" : "not updated"}
                  </p>
                ) : null}
                {execResult.errors.length > 0 ? (
                  <p className="mt-0.5 text-rose-300">Errors: {execResult.errors.join(" · ")}</p>
                ) : null}
              </div>
            ) : null}
          </div>
        ) : (
          <EmptyState title="No preview" description="Run Preview Rebalance first." />
        )}
      </Modal>
    </section>
  );
}

function describeRunError(error: string | null | undefined, fallback: string): string {
  const raw = error ?? "";
  if (/\b409\b/.test(raw)) {
    return `Execution rejected (409): live/unsafe environment or no executable book. No orders were submitted. ${fallback}`;
  }
  if (/\b422\b/.test(raw)) {
    return `Invalid strategy or roots (422). ${fallback}`;
  }
  if (/\b404\b/.test(raw)) {
    return `Unknown strategy (404). ${fallback}`;
  }
  if (/\b503\b/.test(raw)) {
    return `Execution book unavailable (503). ${fallback}`;
  }
  return raw ? `${fallback} (${raw})` : fallback;
}

function formatNumber(value: number): string {
  return Number.isFinite(value) ? value.toLocaleString("en-US", { maximumFractionDigits: 2 }) : "—";
}

function rootsFromStrategy(strategy: { universe: { params: Record<string, unknown> } }): string[] {
  const roots = strategy.universe.params.roots;
  if (Array.isArray(roots)) return roots.map(String);
  if (typeof roots === "string") return roots.split(",").map((root) => root.trim()).filter(Boolean);
  return [];
}
