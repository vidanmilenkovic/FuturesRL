"use client";

import { useEffect, useMemo, useState } from "react";

import { Modal } from "@/components/ui/dialog";
import {
  Badge,
  Button,
  Field,
  Select,
  TextInput,
} from "@/components/ui/primitives";
import { estimateDataCost, getDataInventory, ingestDataCache } from "@/lib/api";
import { formatBytes } from "@/lib/data-cache";
import { cn } from "@/lib/cn";
import type {
  DataCacheRoot,
  DataCostEstimate,
  DataIngestResult,
  DataInventoryResponse,
  DataSource,
} from "@/lib/types";
import { useAsyncData } from "@/lib/use-async-data";

interface InventoryState {
  data: DataInventoryResponse | null;
  source: "live" | "demo" | "loading";
  error: string | null;
  loading: boolean;
  reload: () => void;
}

/**
 * Phase 11B Market Data Cache entry point. Button exposes cached/total counts
 * and opens the inventory/cost/ingest modal used by the Operator drawer and
 * Backtest Lab.
 */
export function MarketDataCacheButton({
  label = "Market Data Cache",
  className,
  testId = "market-data-cache-button",
}: {
  label?: string;
  className?: string;
  testId?: string;
}) {
  const inventory = useAsyncData(getDataInventory);
  const [open, setOpen] = useState(false);
  const cached = inventory.data?.totals.cached_roots ?? 0;
  const total = inventory.data?.totals.total_roots ?? 45;

  return (
    <>
      <Button
        variant="ghost"
        size="xs"
        testId={testId}
        onClick={() => {
          inventory.reload();
          setOpen(true);
        }}
        className={cn("shrink-0", className)}
        title="Open the Market Data Cache manager"
      >
        <span className="text-zinc-400">{label}</span>
        <span className="ml-1 rounded bg-surface px-1 py-0.5 font-mono text-[9.5px] text-zinc-500" data-testid="market-data-cache-count">
          {inventory.source === "live" ? `${cached}/${total}` : "n/a"}
        </span>
      </Button>
      <MarketDataCacheModal open={open} onClose={() => setOpen(false)} inventory={inventory} />
    </>
  );
}

function MarketDataCacheModal({
  open,
  onClose,
  inventory,
}: {
  open: boolean;
  onClose: () => void;
  inventory: InventoryState;
}) {
  const data = inventory.data;
  const roots = useMemo(() => data?.roots ?? [], [data]);
  const sectors = useMemo(() => [...new Set(roots.map((root) => root.sector))].sort(), [roots]);
  const [sector, setSector] = useState("all");
  const [selectedRoots, setSelectedRoots] = useState<Set<string>>(() => new Set());
  const [start, setStart] = useState("");
  const [end, setEnd] = useState("");
  const [source, setSource] = useState<DataSource>("ibkr");
  const [overBudget, setOverBudget] = useState(false);
  const [cost, setCost] = useState<DataCostEstimate | null>(null);
  const [costError, setCostError] = useState<string | null>(null);
  const [costLoading, setCostLoading] = useState(false);
  const [ingesting, setIngesting] = useState(false);
  const [ingestResult, setIngestResult] = useState<DataIngestResult | null>(null);
  const [ingestError, setIngestError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    inventory.reload();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  useEffect(() => {
    if (!data) return;
    setSelectedRoots((current) =>
      current.size > 0 ? current : new Set(data.roots.map((root) => root.root)),
    );
    setStart((current) => current || data.totals.start_date || "2020-01-01");
    setEnd((current) => current || data.totals.end_date || new Date().toISOString().slice(0, 10));
  }, [data]);

  const visibleRoots = sector === "all" ? roots : roots.filter((root) => root.sector === sector);
  const selected = roots.filter((root) => selectedRoots.has(root.root));
  const totalBytes = data?.totals.total_size_bytes ?? 0;
  const summaryRange =
    data?.totals.start_date && data.totals.end_date
      ? `${data.totals.start_date} → ${data.totals.end_date}`
      : "no cached date range";
  const sourceAvailable =
    source === "ibkr" ? Boolean(data?.ibkr_available) : Boolean(data?.databento_configured);
  const inventoryAvailable = inventory.source === "live" && Boolean(data);
  const canRun = inventoryAvailable && selected.length > 0 && start.length > 0 && end.length > 0 && start <= end;

  const toggleRoot = (root: string) => {
    setSelectedRoots((current) => {
      const next = new Set(current);
      if (next.has(root)) next.delete(root);
      else next.add(root);
      return next;
    });
    setCost(null);
    setCostError(null);
    setIngestResult(null);
    setIngestError(null);
  };

  const setVisibleSelection = (select: boolean) => {
    setSelectedRoots((current) => {
      const next = new Set(current);
      for (const root of visibleRoots) {
        if (select) next.add(root.root);
        else next.delete(root.root);
      }
      return next;
    });
    setCost(null);
    setCostError(null);
  };

  const checkCost = async () => {
    if (!canRun) return;
    setCostLoading(true);
    setCostError(null);
    setCost(null);
    const result = await estimateDataCost({
      source,
      roots: [...selectedRoots],
      start,
      end,
      schema: data?.schema ?? "ohlcv-1d",
    });
    setCostLoading(false);
    if (result.source !== "live") {
      setCostError(
        describeDataError(result.error, result.data?.estimated_cost_usd, "Cost estimate failed."),
      );
      return;
    }
    setCost(result.data);
  };

  const runBackfill = async () => {
    if (!canRun) return;
    setIngesting(true);
    setIngestError(null);
    setIngestResult(null);
    const result = await ingestDataCache({
      source,
      roots: [...selectedRoots],
      start,
      end,
      schema: data?.schema ?? "ohlcv-1d",
      allow_over_budget: overBudget,
      confirm: true,
    });
    setIngesting(false);
    if (result.source !== "live" || !result.data.ok) {
      setIngestError(describeDataError(result.error, null, result.data.message || "Backfill failed."));
      return;
    }
    setIngestResult(result.data);
    inventory.reload();
  };

  return (
    <Modal
      open={open}
      title="Market Data Cache"
      description="Inspect the Parquet daily-bar cache, estimate Databento cost, and backfill missing roots from IBKR or Databento."
      onClose={onClose}
      width="max-w-5xl"
      footer={<Button variant="ghost" onClick={onClose}>Close</Button>}
    >
      <div className="max-h-[72vh] space-y-2.5 overflow-y-auto pr-1" data-testid="market-data-cache-modal">
        {inventory.source !== "live" ? (
          <div role="alert" data-testid="market-data-inventory-error" className="rounded border border-loss/40 bg-loss-soft/40 px-2.5 py-2 text-[11px] leading-4 text-rose-300">
            Data inventory unavailable{inventory.error ? `: ${inventory.error}` : "."} The FastAPI data API is required for cache
            management.
          </div>
        ) : null}
        <div className="rounded-md border border-edge bg-canvas/60 px-2.5 py-2" data-testid="market-data-inventory-summary">
          <div className="flex flex-wrap items-center gap-3 text-[11px] text-zinc-400">
            <span className="text-[12px] font-medium text-zinc-200">
              {data?.totals.cached_roots ?? 0} / {data?.totals.total_roots ?? 45} roots cached
            </span>
            <span className="font-mono text-zinc-500">{summaryRange}</span>
            <span>Size <span className="font-mono text-zinc-300">{formatBytes(totalBytes)}</span></span>
            <span>Bars <span className="font-mono text-zinc-300">{(data?.totals.total_bars ?? 0).toLocaleString("en-US")}</span></span>
            <Badge tone={data?.databento_configured ? "gain" : "warn"}>
              Databento {data?.databento_configured ? "configured" : "not configured"}
            </Badge>
            <Badge tone={data?.ibkr_available ? "gain" : "warn"}>
              IBKR {data?.ibkr_available ? "available" : "disconnected"}
            </Badge>
            <span className="ml-auto font-mono text-[9.5px] text-zinc-600" data-testid="market-data-schema">{data?.schema ?? "ohlcv-1d"}</span>
          </div>
          {data?.warnings && data.warnings.length > 0 ? (
            <p className="mt-1 text-[10px] text-amber-300">{data.warnings.join(" · ")}</p>
          ) : null}
        </div>

        <div className="grid grid-cols-12 gap-2.5">
          <div className="col-span-12 space-y-2 lg:col-span-5">
            <div className="grid grid-cols-2 gap-2">
              <Field label="Sector" htmlFor="data-sector">
                <Select
                  id="data-sector"
                  value={sector}
                  onChange={setSector}
                  options={[{ value: "all", label: `All sectors (${roots.length})` }, ...sectors.map((entry) => ({ value: entry, label: entry }))]}
                />
              </Field>
              <Field label="Source" htmlFor="data-source">
                <Select
                  id="data-source"
                  value={source}
                  onChange={(value) => {
                    setSource(value as DataSource);
                    setCost(null);
                    setCostError(null);
                    setIngestError(null);
                  }}
                  options={[
                    { value: "ibkr", label: "IBKR Historical (Free)" },
                    { value: "databento", label: "Databento Pay-As-You-Go" },
                  ]}
                />
              </Field>
              <Field label="Start" htmlFor="data-start">
                <TextInput id="data-start" type="date" value={start} onChange={setStart} />
              </Field>
              <Field label="End" htmlFor="data-end">
                <TextInput id="data-end" type="date" value={end} onChange={setEnd} />
              </Field>
            </div>
            <div className="flex flex-wrap items-center gap-1.5">
              <Button variant="secondary" size="xs" testId="data-check-cost" onClick={() => void checkCost()} disabled={!canRun || costLoading}>
                {costLoading ? (
                  <>
                    <span aria-hidden className="h-3 w-3 animate-spin rounded-full border-2 border-white/30 border-t-white" />
                    Checking…
                  </>
                ) : (
                  "Check Cost"
                )}
              </Button>
              <Button variant="primary" size="xs" testId="data-backfill" onClick={() => void runBackfill()} disabled={!canRun || ingesting || !sourceAvailable}>
                {ingesting ? (
                  <>
                    <span aria-hidden className="h-3 w-3 animate-spin rounded-full border-2 border-white/30 border-t-white" />
                    Backfilling…
                  </>
                ) : (
                  "Backfill Parquet Cache"
                )}
              </Button>
              <span className="text-[10px] text-zinc-600">{selected.length} root{selected.length === 1 ? "" : "s"} selected</span>
            </div>
            {!sourceAvailable ? (
              <p className="rounded border border-warn/30 bg-warn-soft/30 px-2 py-1 text-[10px] text-amber-300" data-testid="data-source-unavailable">
                {source === "ibkr"
                  ? "IBKR Historical is unavailable (no connected TWS/paper session); backfill is disabled."
                  : "Databento is not configured (missing API key); backfill is disabled."}
              </p>
            ) : null}
            {source === "databento" ? (
              <label className="flex items-start gap-2 rounded border border-warn/30 bg-warn-soft/20 px-2 py-1.5 text-[10px] leading-4 text-zinc-300">
                <input
                  type="checkbox"
                  checked={overBudget}
                  onChange={(event) => setOverBudget(event.target.checked)}
                  className="mt-0.5 h-3 w-3 rounded border-zinc-600 bg-canvas accent-red-500"
                  data-testid="data-allow-over-budget"
                />
                I explicitly allow this Databento backfill to exceed the cost guard (operator confirmation).
              </label>
            ) : null}
            {costError ? (
              <div role="alert" data-testid="data-cost-error" className="rounded border border-loss/40 bg-loss-soft/40 px-2 py-1.5 text-[10.5px] text-rose-300">
                {costError}
              </div>
            ) : null}
            {cost ? (
              <div data-testid="data-cost-result" className="rounded border border-gain/30 bg-gain-soft/20 px-2.5 py-2 text-[11px] text-zinc-300">
                <p className={cn("font-medium", cost.details?.metadata_available === false ? "text-amber-300" : "text-emerald-300")}>
                  {cost.source === "ibkr"
                    ? "$0.00 · IBKR historical (free)"
                    : cost.details?.metadata_available === false
                      ? "Estimate unavailable — $0.00 fallback"
                      : `$${cost.estimated_cost_usd.toFixed(4)} USD`}
                </p>
                <p className="mt-0.5 text-[10px] text-zinc-500">
                  Billable {cost.billable_size_mb.toFixed(2)} MB ({cost.billable_size_bytes.toLocaleString("en-US")} bytes) ·
                  source {cost.source}
                </p>
              </div>
            ) : null}
            {ingestError ? (
              <div role="alert" data-testid="data-ingest-error" className="rounded border border-loss/40 bg-loss-soft/40 px-2 py-1.5 text-[10.5px] text-rose-300">
                {ingestError}
              </div>
            ) : null}
            {ingestResult ? (
              <div data-testid="data-ingest-result" className="rounded border border-gain/30 bg-gain-soft/20 px-2.5 py-2 text-[10.5px] text-zinc-300">
                <p className="font-medium text-emerald-300">Backfill complete · {ingestResult.source}</p>
                <p className="mt-0.5">
                  {ingestResult.roots_ingested.length} roots · {ingestResult.bars_written.toLocaleString("en-US")} bars written
                  {ingestResult.duration_seconds !== null ? ` · ${ingestResult.duration_seconds.toFixed(2)}s` : ""}
                </p>
                <p className="mt-0.5 font-mono text-[9.5px] text-zinc-500">{ingestResult.roots_ingested.join(", ") || "no roots"}</p>
                {ingestResult.warnings.length > 0 ? (
                  <p className="mt-0.5 text-amber-300">{ingestResult.warnings.join(" · ")}</p>
                ) : null}
              </div>
            ) : null}
          </div>

          <div className="col-span-12 rounded border border-edge bg-canvas/40 lg:col-span-7">
            <div className="flex items-center justify-between border-b border-hairline px-2 py-1.5 text-[10.5px] text-zinc-400">
              <span>Roots · cached bars / source</span>
              <span className="flex items-center gap-1">
                <button type="button" className="focus-ring rounded px-1 py-0.5 text-[9.5px] text-zinc-500 hover:bg-surface hover:text-zinc-300" onClick={() => setVisibleSelection(true)}>Select all</button>
                <button type="button" className="focus-ring rounded px-1 py-0.5 text-[9.5px] text-zinc-500 hover:bg-surface hover:text-zinc-300" onClick={() => setVisibleSelection(false)}>Clear</button>
              </span>
            </div>
            <div className="grid max-h-[280px] grid-cols-1 gap-0.5 overflow-y-auto p-1 text-[10.5px] sm:grid-cols-2" data-testid="data-root-list">
              {visibleRoots.map((root) => (
                <RootRow
                  key={root.root}
                  root={root}
                  checked={selectedRoots.has(root.root)}
                  onToggle={() => toggleRoot(root.root)}
                />
              ))}
              {visibleRoots.length === 0 ? (
                <p className="px-2 py-3 text-[10.5px] text-zinc-600">No roots in this sector.</p>
              ) : null}
            </div>
          </div>
        </div>
      </div>
    </Modal>
  );
}

function RootRow({ root, checked, onToggle }: { root: DataCacheRoot; checked: boolean; onToggle: () => void }) {
  return (
    <label className="flex cursor-pointer items-center gap-1.5 rounded px-1 py-0.5 hover:bg-surface">
      <input
        type="checkbox"
        checked={checked}
        onChange={onToggle}
        className="h-3 w-3 rounded border-zinc-600 bg-canvas accent-blue-500"
        data-testid={`data-root-${root.root}`}
      />
      <span className="w-10 font-mono text-zinc-300">{root.root}</span>
      <span className={cn("min-w-0 flex-1 truncate font-mono text-[9.5px]", root.cached_bars > 0 ? "text-emerald-400" : "text-zinc-600")}>
        {root.cached_bars > 0 ? `${root.cached_bars.toLocaleString("en-US")} · ${root.source}` : "empty"}
      </span>
      <span className="text-[9px] text-zinc-700">{root.start_date && root.end_date ? `${root.start_date.slice(0, 7)}→${root.end_date.slice(0, 7)}` : ""}</span>
    </label>
  );
}

function describeDataError(error: string | null | undefined, cost: number | null | undefined, fallback: string): string {
  const raw = error ?? "";
  if (/\b402\b/.test(raw)) {
    return `Cost guard rejected the Databento job (402)${cost !== null && cost !== undefined ? ` · estimate $${cost.toFixed(4)}` : ""}. Enable the over-budget confirmation and retry.`;
  }
  if (/\b409\b/.test(raw)) {
    return `Backfill blocked (409): source unavailable OR Databento cost guard exceeded. For Databento, enable the over-budget confirmation and retry. ${fallback}`;
  }
  if (/\b400\b/.test(raw)) {
    return `Missing Databento API key or invalid request (400): ${fallback}`;
  }
  if (/\b422\b/.test(raw)) {
    return `Invalid roots / date window (422): ${fallback}`;
  }
  return raw ? `${fallback} (${raw})` : fallback;
}
