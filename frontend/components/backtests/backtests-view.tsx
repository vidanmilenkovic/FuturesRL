"use client";

import { useEffect, useMemo, useState } from "react";

import { AttributionBars } from "@/components/charts/plots";
import { MultiSeriesChart, TimeSeriesChart, type OverlaySeries } from "@/components/charts/time-series";
import {
  Badge,
  Button,
  EmptyState,
  Field,
  Panel,
  Segmented,
  Select,
  Skeleton,
  TextInput,
  type Tone,
} from "@/components/ui/primitives";
import { Modal } from "@/components/ui/dialog";
import { DownloadIcon, TrashIcon } from "@/components/shell/icons";
import { MarketDataCacheButton } from "@/components/data/market-data-cache";
import {
  deleteBacktest,
  exportTearSheet,
  exportTradesCsv,
  getBacktest,
  getTearsheet,
  listBacktests,
  listStrategies,
  runBacktest,
} from "@/lib/api";
import { diffStrategyConfigs, formatDiffValue } from "@/lib/artifacts";
import { cn } from "@/lib/cn";
import {
  DATA_MODE_OPTIONS,
  DATA_TIMEFRAME_OPTIONS,
  DATA_TIMEFRAMES,
  cachedBarsDetail,
  dataOriginBadge,
} from "@/lib/data-cache";
import {
  formatCurrency,
  formatDateTime,
  formatDuration,
  formatInteger,
  formatNumber,
  formatPct,
  formatPctPlain,
  formatSignedCurrency,
} from "@/lib/format";
import type {
  BacktestRunConfig,
  BacktestRunSummary,
  CachedBarsMeta,
  DataTimeframe,
  MarketDataMode,
  StrategyConfig,
  TearsheetResponse,
} from "@/lib/types";
import { useAsyncData } from "@/lib/use-async-data";

type ChartMode = "sharpe" | "attribution";
type Better = "high" | "low";

interface MetricRow {
  label: string;
  value: (run: BacktestRunSummary) => number | null;
  format: (value: number) => string;
  better: Better;
}

interface MetricGroup {
  title: string;
  rows: MetricRow[];
}

const STATUS_TONE: Record<string, Tone> = {
  completed: "gain",
  running: "accent",
  queued: "neutral",
  failed: "loss",
};

const METRIC_GROUPS: MetricGroup[] = [
  {
    title: "Return & risk",
    rows: [
      { label: "CAGR", value: (run) => run.metrics?.cagr ?? null, format: (value) => formatPct(value), better: "high" },
      { label: "Annualised vol", value: (run) => run.metrics?.annual_volatility ?? null, format: (value) => formatPctPlain(value), better: "low" },
      { label: "Sharpe", value: (run) => run.metrics?.sharpe ?? null, format: (value) => value.toFixed(2), better: "high" },
      { label: "Sortino", value: (run) => run.metrics?.sortino ?? null, format: (value) => value.toFixed(2), better: "high" },
      { label: "Calmar", value: (run) => run.metrics?.calmar ?? null, format: (value) => value.toFixed(2), better: "high" },
      { label: "Omega", value: (run) => run.metrics?.omega ?? null, format: (value) => value.toFixed(2), better: "high" },
      { label: "Max drawdown", value: (run) => run.metrics?.max_drawdown ?? null, format: (value) => formatPctPlain(value), better: "low" },
      { label: "Max DD duration", value: (run) => run.metrics?.max_drawdown_duration_days ?? null, format: (value) => `${formatNumber(value, 0)}d`, better: "low" },
    ],
  },
  {
    title: "Tail risk",
    rows: [
      { label: "VaR 95 (1d)", value: (run) => run.metrics?.var_95 ?? null, format: (value) => formatPctPlain(value), better: "low" },
      { label: "VaR 99 (1d)", value: (run) => run.metrics?.var_99 ?? null, format: (value) => formatPctPlain(value), better: "low" },
      { label: "CVaR 95", value: (run) => run.metrics?.cvar_95 ?? null, format: (value) => formatPctPlain(value), better: "low" },
      { label: "CVaR 99", value: (run) => run.metrics?.cvar_99 ?? null, format: (value) => formatPctPlain(value), better: "low" },
      { label: "Rachev ratio", value: (run) => run.metrics?.rachev_ratio ?? null, format: (value) => value.toFixed(3), better: "high" },
    ],
  },
  {
    title: "Futures & execution",
    rows: [
      { label: "Return on margin", value: (run) => run.metrics?.rom ?? null, format: (value) => formatPct(value), better: "high" },
      { label: "Avg margin util", value: (run) => run.metrics?.avg_margin_utilization ?? null, format: (value) => formatPctPlain(value), better: "low" },
      { label: "Peak margin util", value: (run) => run.metrics?.peak_margin_utilization ?? null, format: (value) => formatPctPlain(value), better: "low" },
      { label: "Turnover", value: (run) => run.metrics?.turnover ?? null, format: (value) => `${value.toFixed(2)}×`, better: "low" },
      { label: "Impl. shortfall", value: (run) => run.metrics?.implementation_shortfall_ticks ?? null, format: (value) => `${value.toFixed(2)} ticks`, better: "low" },
      { label: "Comm / gross PnL", value: (run) => run.metrics?.commission_to_gross_pnl ?? null, format: (value) => formatPctPlain(value), better: "low" },
      { label: "Trades", value: (run) => run.n_trades ?? null, format: (value) => formatInteger(value), better: "high" },
      { label: "Rebalances", value: (run) => run.n_rebalances ?? null, format: (value) => formatInteger(value), better: "high" },
    ],
  },
];

const REBALANCE_OPTIONS = [
  { value: "W-FRI", label: "Weekly (Fri)" },
  { value: "M", label: "Monthly" },
  { value: "D", label: "Daily" },
];

/** View 5 — backtest lab & tear-sheet comparison. */
export function BacktestsView() {
  const backtests = useAsyncData(listBacktests);
  const strategies = useAsyncData(listStrategies);
  const [runs, setRuns] = useState<BacktestRunSummary[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [compareIds, setCompareIds] = useState<string[]>([]);

  const [strategyId, setStrategyId] = useState("");
  const [start, setStart] = useState("2025-01-02");
  const [end, setEnd] = useState("2025-12-31");
  const [initialEquity, setInitialEquity] = useState(1_000_000);
  const [rebalance, setRebalance] = useState("W-FRI");
  const [walkForward, setWalkForward] = useState(true);
  const [dataMode, setDataMode] = useState<MarketDataMode>("auto");
  const [timeframe, setTimeframe] = useState<DataTimeframe>("1d");
  const [wfSplits, setWfSplits] = useState(5);
  const [purgeBars, setPurgeBars] = useState("");
  const [embargoBars, setEmbargoBars] = useState("");
  const [running, setRunning] = useState(false);
  const [runnerMessage, setRunnerMessage] = useState<string | null>(null);
  const [runnerDemo, setRunnerDemo] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<BacktestRunSummary | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [exportOpen, setExportOpen] = useState(false);
  const [exportMessage, setExportMessage] = useState<string | null>(null);
  const [exportBusy, setExportBusy] = useState<string | null>(null);
  const [runDetails, setRunDetails] = useState<Record<string, BacktestRunSummary | null>>({});

  useEffect(() => {
    if (backtests.data) {
      const runsNow = backtests.data.runs;
      const requestedRun =
        typeof window !== "undefined" ? new URLSearchParams(window.location.search).get("run") : null;
      const requestedTimeframe =
        typeof window !== "undefined" ? new URLSearchParams(window.location.search).get("timeframe") : null;
      if (requestedTimeframe && DATA_TIMEFRAMES.includes(requestedTimeframe as DataTimeframe)) {
        setTimeframe(requestedTimeframe as DataTimeframe);
      }
      setRuns(runsNow);
      setSelectedRunId((current) => {
        if (current && runsNow.some((run) => run.id === current)) return current;
        if (requestedRun && runsNow.some((run) => run.id === requestedRun)) return requestedRun;
        return runsNow[0]?.id ?? null;
      });
      setCompareIds((current) => {
        if (current.length > 0) return current;
        const requestedValid =
          requestedRun && runsNow.some((run) => run.id === requestedRun) ? requestedRun : null;
        const defaults = runsNow.slice(0, 2).map((run) => run.id);
        return requestedValid ? [requestedValid, ...defaults.filter((id) => id !== requestedValid)].slice(0, 2) : defaults;
      });
    }
  }, [backtests.data]);

  // QoL parameter diff: fetch strategy_config snapshots for the compared runs.
  const detailKey = columnsSignature(compareIds, selectedRunId);
  useEffect(() => {
    let cancelled = false;
    const ids = [...new Set([...compareIds, ...(selectedRunId ? [selectedRunId] : [])])];
    if (ids.length === 0) return;
    void Promise.all(
      ids.map(async (id) => {
        const result = await getBacktest(id);
        return [id, result.data] as const;
      }),
    ).then((entries) => {
      if (!cancelled) setRunDetails(Object.fromEntries(entries));
    });
    return () => {
      cancelled = true;
    };
  }, [detailKey, compareIds, selectedRunId]);

  useEffect(() => {
    if (!strategyId && strategies.data && strategies.data.length > 0) {
      const requested =
        typeof window !== "undefined" ? new URLSearchParams(window.location.search).get("strategy") : null;
      const match = requested
        ? strategies.data.find((strategy) => strategy.id === requested)
        : undefined;
      setStrategyId(match?.id ?? strategies.data[0]?.id ?? "");
    }
  }, [strategies.data, strategyId]);

  const selectedRun = useMemo(
    () => runs.find((run) => run.id === selectedRunId) ?? null,
    [runs, selectedRunId],
  );
  const compareRuns = useMemo(() => {
    const matched = compareIds
      .map((id) => runs.find((run) => run.id === id))
      .filter((run): run is BacktestRunSummary => Boolean(run));
    return matched.length > 0 ? matched : selectedRun ? [selectedRun] : [];
  }, [compareIds, runs, selectedRun]);

  const toggleCompare = (id: string) => {
    setCompareIds((current) =>
      current.includes(id)
        ? current.filter((entry) => entry !== id)
        : [...current, id].slice(-3),
    );
  };

  const submitRun = async () => {
    if (!strategyId) return;
    setRunning(true);
    setRunnerMessage(null);
    const config: BacktestRunConfig = {
      strategy_id: strategyId,
      start,
      end,
      initial_equity: initialEquity,
      rebalance_frequency: rebalance,
      walk_forward: walkForward,
      random_seed: 7,
      data_mode: dataMode,
      timeframe,
      ...(walkForward ? { wf_splits: wfSplits } : {}),
      ...(purgeBars.trim() !== "" ? { purge_bars: Number(purgeBars) } : {}),
      ...(embargoBars.trim() !== "" ? { embargo_bars: Number(embargoBars) } : {}),
    };
    const result = await runBacktest(config);
    const run = result.data;
    setRuns((current) => [run, ...current]);
    setSelectedRunId(run.id);
    setCompareIds((current) => [run.id, ...current.filter((id) => id !== run.id)].slice(0, 3));
    setRunnerDemo(result.source === "demo");
    setRunnerMessage(
      result.source === "live"
        ? `Run ${run.id} completed · ${formatInteger(run.n_trades ?? 0)} trades · ${formatDuration(run.duration_seconds)} · ${dataOriginBadge(run.data_origin).label} · TF ${run.timeframe ?? timeframe}`

        : `Backend unreachable — deterministic UI-fallback tear sheet ${run.id}.`,
    );
    setRunning(false);
  };

  const columns = compareRuns.length > 0 ? compareRuns : selectedRun ? [selectedRun] : [];
  const strategiesOptions: StrategyConfig[] = strategies.data ?? [];
  const splitsValid = !walkForward || (Number.isFinite(wfSplits) && wfSplits >= 2 && wfSplits <= 20);
  const invalid = !strategyId || start >= end || initialEquity <= 0 || !splitsValid;

  if (backtests.loading && runs.length === 0) {
    return (
      <div className="grid grid-cols-12 gap-3">
        <Skeleton className="col-span-12 h-16" />
        <Skeleton className="col-span-12 h-44" />
        <Skeleton className="col-span-7 h-96" />
        <Skeleton className="col-span-5 h-96" />
      </div>
    );
  }

  return (
    <div className="grid grid-cols-12 gap-3">
      {/* --------------------------------------------------------- runner bar */}
      <Panel
        className="col-span-12"
        title="Backtests"
        flush
        actions={<MarketDataCacheButton />}
      >
        <div className="flex flex-wrap items-end gap-3 px-3 py-2.5">
          <Field label="Strategy" htmlFor="bt-strategy" className="w-[220px]">
            <Select
              id="bt-strategy"
              value={strategyId}
              onChange={setStrategyId}
              disabled={running}
              options={strategiesOptions.map((strategy) => ({ value: strategy.id, label: strategy.id }))}
            />
          </Field>
          <Field label="Start" htmlFor="bt-start" className="w-[132px]">
            <TextInput id="bt-start" type="date" value={start} onChange={setStart} disabled={running} />
          </Field>
          <Field label="End" htmlFor="bt-end" className="w-[132px]">
            <TextInput id="bt-end" type="date" value={end} onChange={setEnd} disabled={running} />
          </Field>
          <Field label="Initial equity" htmlFor="bt-equity" className="w-[132px]">
            <TextInput
              id="bt-equity"
              type="number"
              step={100_000}
              value={initialEquity}
              onChange={(value) => setInitialEquity(Number(value))}
              disabled={running}
            />
          </Field>
          <Field label="Rebalance" htmlFor="bt-rebalance" className="w-[148px]">
            <Select
              id="bt-rebalance"
              value={rebalance}
              onChange={setRebalance}
              disabled={running}
              options={REBALANCE_OPTIONS}
            />
          </Field>
          <Field label="Data mode" htmlFor="bt-data-mode" className="w-[240px]">
            <Select
              id="bt-data-mode"
              value={dataMode}
              onChange={(value) => setDataMode(value as MarketDataMode)}
              disabled={running}
              options={DATA_MODE_OPTIONS.map((option) => ({ value: option.value, label: option.label }))}
            />
          </Field>
          <Field label="Timeframe" htmlFor="bt-timeframe" className="w-[112px]">
            <Select
              id="bt-timeframe"
              value={timeframe}
              onChange={(value) => setTimeframe(value as DataTimeframe)}
              disabled={running}
              options={DATA_TIMEFRAME_OPTIONS.map((option) => ({ value: option.value, label: option.label }))}
            />
          </Field>
          <div className="flex items-end gap-2">
            <label
              htmlFor="bt-walkforward"
              className={cn(
                "flex h-7 items-center gap-1.5 text-[11px] font-medium",
                walkForward ? "text-zinc-200" : "text-zinc-500",
              )}
              title="Purged & embargoed walk-forward validation"
            >
              <input
                id="bt-walkforward"
                type="checkbox"
                checked={walkForward}
                disabled={running}
                onChange={(event) => setWalkForward(event.target.checked)}
                className="h-3.5 w-3.5 rounded border border-zinc-600 bg-canvas accent-blue-500"
              />
              Walk-forward
            </label>
            <Field label="Splits" htmlFor="bt-splits" className="w-[62px]">
              <TextInput
                id="bt-splits"
                type="number"
                min={2}
                value={wfSplits}
                onChange={(value) => setWfSplits(Number(value))}
                disabled={!walkForward || running}
                className="text-center"
              />
            </Field>
            <Field label="Purge" htmlFor="bt-purge" className="w-[58px]">
              <TextInput
                id="bt-purge"
                type="number"
                min={0}
                value={purgeBars}
                placeholder="auto"
                onChange={setPurgeBars}
                disabled={running}
                className="text-center"
              />
            </Field>
            <Field label="Embargo" htmlFor="bt-embargo" className="w-[62px]">
              <TextInput
                id="bt-embargo"
                type="number"
                min={0}
                value={embargoBars}
                placeholder="auto"
                onChange={setEmbargoBars}
                disabled={running}
                className="text-center"
              />
            </Field>
          </div>
          <div className="flex items-center gap-2 pb-0.5">
            <Button variant="primary" onClick={() => void submitRun()} disabled={running || invalid}>
              {running ? "Running…" : "Run backtest"}
            </Button>
            <Button variant="ghost" onClick={() => backtests.reload()} disabled={running}>
              Refresh
            </Button>
          </div>
          {!splitsValid ? (
            <span className="pb-1 text-[11px] text-loss">Walk-forward splits must be 2–20</span>
          ) : null}
          <div className="ml-auto flex min-w-0 items-center gap-2 pb-0.5 text-[11px]">
            {runnerMessage ? (
              <span className={cn("truncate", runnerDemo ? "text-amber-400" : "text-emerald-400")} title={runnerMessage}>
                {runnerMessage}
              </span>
            ) : (
              <span className="text-zinc-600">
                Offline deterministic engine · {formatInteger(columns.length)} run{columns.length === 1 ? "" : "s"} compared
              </span>
            )}
          </div>
        </div>
      </Panel>

      {/* -------------------------------------------------------- run history */}
      <Panel
        className="col-span-12"
        title="Run history"
        flush
        actions={<span className="text-[11px] text-zinc-600">{runs.length} runs · select up to 3 to compare</span>}
      >
        <div className="max-h-[196px] overflow-auto">
          <table className="data-table">
            <thead>
              <tr>
                <th className="w-10" />
                <th className="w-28">Run</th>
                <th>Strategy</th>
                <th className="w-24">Status</th>
                <th className="w-24">Source</th>
                <th className="w-40">Data</th>
                <th className="w-32">Kind</th>
                <th className="text-right">CAGR</th>
                <th className="text-right">Sharpe</th>
                <th className="text-right">Max DD</th>
                <th className="text-right">Rachev</th>
                <th className="text-right">Trades</th>
                <th className="text-right">Rebal.</th>
                <th className="text-right">Duration</th>
                <th className="w-32">Created</th>
                <th className="w-10" />
              </tr>
            </thead>
            <tbody>
              {runs.map((run) => {
                const inCompare = compareIds.includes(run.id);
                return (
                  <tr
                    key={run.id}
                    className={cn("cursor-pointer", selectedRunId === run.id && "bg-surface")}
                    onClick={() => setSelectedRunId(run.id)}
                  >
                    <td onClick={(event) => event.stopPropagation()}>
                      <input
                        type="checkbox"
                        aria-label={`Compare run ${run.id}`}
                        checked={inCompare}
                        onChange={() => toggleCompare(run.id)}
                        onClick={(event) => event.stopPropagation()}
                        className="h-3.5 w-3.5 rounded border border-zinc-600 bg-canvas accent-blue-500"
                      />
                    </td>
                    <td className="font-mono text-[11.5px] text-zinc-200">{run.id}</td>
                    <td className="text-zinc-400">{run.strategy_id}</td>
                    <td>
                      <Badge tone={STATUS_TONE[run.status] ?? "neutral"}>{run.status}</Badge>
                    </td>
                    <td>
                      <Badge tone={run.data_source === "fixture" ? "warn" : "gain"}>
                        {run.data_source === "fixture" ? "Fixture" : "Engine"}
                      </Badge>
                    </td>
                    <td>
                      <DataOriginBadge
                        origin={run.data_origin}
                        cachedBars={run.cached_bars}
                        mode={run.data_mode}
                        timeframe={run.timeframe}
                      />
                    </td>
                    <td>
                      {(() => {
                        const folds = run.n_folds ?? run.wf_splits ?? run.config?.wf_splits ?? null;
                        if (run.kind !== "walk_forward") {
                          return <span className="text-[11px] text-zinc-600">—</span>;
                        }
                        return (
                          <Badge
                            tone="accent"
                            title={
                              folds !== null && folds !== undefined
                                ? `Purged & embargoed walk-forward validation · ${folds} folds`
                                : "Purged & embargoed walk-forward validation · fold count not reported"
                            }
                          >
                            {folds !== null && folds !== undefined ? `WF · ${folds} folds` : "WF"}
                          </Badge>
                        );
                      })()}
                    </td>
                    <td className={cn("num text-right", (run.metrics?.cagr ?? 0) >= 0 ? "text-gain" : "text-loss")}>
                      {run.metrics ? formatPct(run.metrics.cagr) : "—"}
                    </td>
                    <td className="num text-right text-zinc-200">{run.metrics ? run.metrics.sharpe.toFixed(2) : "—"}</td>
                    <td className="num text-right text-loss/90">
                      {run.metrics ? formatPctPlain(run.metrics.max_drawdown) : "—"}
                    </td>
                    <td className="num text-right text-zinc-400">{run.metrics ? run.metrics.rachev_ratio.toFixed(2) : "—"}</td>
                    <td className="num text-right text-zinc-400">
                      {run.n_trades === null || run.n_trades === undefined ? "—" : formatInteger(run.n_trades)}
                    </td>
                    <td className="num text-right text-zinc-400">
                      {run.n_rebalances === null || run.n_rebalances === undefined ? "—" : formatInteger(run.n_rebalances)}
                    </td>
                    <td className="num text-right text-zinc-500">
                      {run.duration_seconds === null ? "—" : formatDuration(run.duration_seconds)}
                    </td>
                    <td className="text-[11px] text-zinc-500">{formatDateTime(run.created_at)}</td>
                    <td className="text-right">
                      <button
                        type="button"
                        aria-label={`Delete backtest ${run.id}`}
                        title="Delete this backtest artifact"
                        data-testid={`delete-backtest-${run.id}`}
                        onClick={(event) => {
                          event.stopPropagation();
                          setDeleteError(null);
                          setDeleteTarget(run);
                        }}
                        className="focus-ring grid h-6 w-6 place-items-center rounded text-zinc-600 hover:bg-loss-soft hover:text-rose-300"
                      >
                        <TrashIcon className="h-3.5 w-3.5" />
                      </button>
                    </td>
                  </tr>
                );
              })}
              {runs.length === 0 ? (
                <tr>
                  <td colSpan={14} className="py-8 text-center text-zinc-600">
                    No backtest runs yet — start one from the experiment runner.
                  </td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
      </Panel>

      {/* ------------------------------------------------------ chart stack */}
      <Panel
        className="col-span-12 xl:col-span-7"
        title={
          selectedRun
            ? `Tear sheet · ${selectedRun.id}${compareRuns.length > 1 ? ` · ${compareRuns.length}-run overlay` : ""}`
            : "Tear sheet"
        }
        actions={
          <div className="flex items-center gap-3">
            {selectedRun ? (
              <DataOriginBadge
                origin={selectedRun.data_origin}
                cachedBars={selectedRun.cached_bars}
                mode={selectedRun.data_mode}
                timeframe={selectedRun.timeframe}
              />
            ) : null}
            {selectedRun?.metrics ? (
              <div className="flex items-center gap-3 font-mono text-[11px] tabular-nums">
                <span className="text-zinc-500">
                  CAGR <span className={selectedRun.metrics.cagr >= 0 ? "text-gain" : "text-loss"}>{formatPct(selectedRun.metrics.cagr)}</span>
                </span>
                <span className="text-zinc-500">
                  Sharpe <span className="text-zinc-200">{selectedRun.metrics.sharpe.toFixed(2)}</span>
                </span>
                <span className="hidden text-zinc-500 sm:inline">
                  Max DD <span className="text-loss">{formatPctPlain(selectedRun.metrics.max_drawdown)}</span>
                </span>
              </div>
            ) : null}
            <div className="relative">
              <Button
                variant="secondary"
                size="xs"
                onClick={() => {
                  setExportMessage(null);
                  setExportOpen((open) => !open);
                }}
                disabled={!selectedRun}
                testId="export-menu-button"
                title="Download run artifacts"
              >
                <DownloadIcon className="h-3 w-3" />
                Export
              </Button>
              {exportOpen && selectedRun ? (
                <div className="absolute right-0 top-7 z-30 w-[220px] rounded-md border border-edge bg-panel p-1 shadow-popover">
                  {(() => {
                    const fixture = selectedRun.data_source === "fixture";
                    const unavailable = fixture || selectedRun.status !== "completed";
                    return (
                      <>
                        <button
                          type="button"
                          data-testid="export-trades-csv"
                          disabled={unavailable || exportBusy !== null}
                          onClick={() => {
                            setExportOpen(false);
                            setExportBusy("trades");
                            void exportTradesCsv(selectedRun.id).then((result) => {
                              setExportBusy(null);
                              if (!result.ok) setExportMessage(`Trades CSV export failed${result.error ? `: ${result.error}` : "."}`);
                            });
                          }}
                          className="flex w-full items-center justify-between rounded px-2 py-1.5 text-left text-[11px] text-zinc-300 hover:bg-surface disabled:cursor-not-allowed disabled:opacity-40"
                          title={fixture ? "Fixture runs have no persisted trade ledger" : "Download the trade ledger CSV"}
                        >
                          Export Trades CSV
                          {exportBusy === "trades" ? <span className="text-[10px] text-zinc-500">…</span> : null}
                        </button>
                        <button
                          type="button"
                          data-testid="export-tearsheet-json"
                          disabled={unavailable || exportBusy !== null}
                          onClick={() => {
                            setExportOpen(false);
                            setExportBusy("json");
                            void exportTearSheet(selectedRun.id).then((result) => {
                              setExportBusy(null);
                              if (!result.ok) setExportMessage(`Tear-sheet JSON export failed${result.error ? `: ${result.error}` : "."}`);
                            });
                          }}
                          className="flex w-full items-center justify-between rounded px-2 py-1.5 text-left text-[11px] text-zinc-300 hover:bg-surface disabled:cursor-not-allowed disabled:opacity-40"
                          title={fixture ? "Fixture runs have no persisted export artifact" : "Download the full tear sheet + strategy config JSON"}
                        >
                          Export Full Tear Sheet JSON
                          {exportBusy === "json" ? <span className="text-[10px] text-zinc-500">…</span> : null}
                        </button>
                        {unavailable ? (
                          <p className="px-2 py-1 text-[9.5px] leading-4 text-amber-400/80">
                            {fixture
                              ? "Fixture runs have no persisted artifacts."
                              : "Exports are available after the run completes."}
                          </p>
                        ) : null}
                      </>
                    );
                  })()}
                </div>
              ) : null}
            </div>
          </div>
        }
      >
        {exportMessage ? (
          <div role="alert" className="mb-2 rounded border border-loss/30 bg-loss-soft/40 px-2 py-1 text-[10.5px] text-rose-300">
            {exportMessage}
          </div>
        ) : null}
        {selectedRun ? <CostBreakdown run={selectedRun} /> : null}
        {selectedRunId ? (
          <TearsheetOverlay
            key={`${compareIds.join("|")}|${selectedRunId}`}
            runs={compareRuns}
            selectedRun={selectedRun}
          />
        ) : (
          <EmptyState title="No run selected" description="Select a run from the history table to inspect its tear sheet." />
        )}
      </Panel>

      {/* ------------------------------------------------------- metrics table */}
      <Panel
        className="col-span-12 xl:col-span-5"
        title="Metrics comparison"
        flush
        actions={<span className="text-[11px] text-zinc-600">best value highlighted</span>}
      >
        {columns.length === 0 ? (
          <EmptyState title="No metrics" description="Run a backtest to populate the institutional metric table." />
        ) : (
          <div className="max-h-[calc(100vh-468px)] min-h-[320px] overflow-auto">
            <table className="data-table">
              <thead>
                <tr>
                  <th className="sticky left-0 z-20 bg-panel">Metric</th>
                  {columns.map((run) => (
                    <th key={run.id} className="text-right">
                      <div className={cn("font-mono text-[11px] font-medium", run.id === selectedRunId ? "text-zinc-100" : "text-zinc-500")}>
                        {run.id}
                      </div>
                      <div className="text-[10px] font-normal normal-case tracking-normal text-zinc-600">{run.strategy_id}</div>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {METRIC_GROUPS.map((group) => (
                  <MetricGroupRows key={group.title} group={group} runs={columns} selectedRunId={selectedRunId} />
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>

      <ParameterDiffAccordion runs={compareRuns} details={runDetails} />
      <Modal
        open={deleteTarget !== null}
        title="Delete backtest"
        description={
          <span>
            Remove <span className="font-mono text-zinc-300">{deleteTarget?.id}</span> from history and delete its
            persisted artifact. This cannot be undone.
          </span>
        }
        onClose={() => {
          if (deleting) return;
          setDeleteTarget(null);
          setDeleteError(null);
        }}
        busy={deleting}
        footer={
          <>
            <Button
              variant="ghost"
              onClick={() => {
                setDeleteTarget(null);
                setDeleteError(null);
              }}
              disabled={deleting}
            >
              Cancel
            </Button>
            <Button
              variant="danger"
              testId="confirm-delete-backtest"
              disabled={deleting}
              onClick={() => {
                if (!deleteTarget) return;
                const target = deleteTarget;
                setDeleting(true);
                setDeleteError(null);
                void deleteBacktest(target.id).then((result) => {
                  setDeleting(false);
                  if (result.source === "live" && result.data.ok) {
                    setRuns((current) => current.filter((run) => run.id !== target.id));
                    setCompareIds((current) => current.filter((id) => id !== target.id));
                    if (selectedRunId === target.id) setSelectedRunId(null);
                    setDeleteTarget(null);
                    backtests.reload();
                  } else {
                    setDeleteError(
                      result.data.message || `Delete failed${result.error ? `: ${result.error}` : "."}`,
                    );
                  }
                });
              }}
            >
              {deleting ? "Deleting…" : "Delete backtest"}
            </Button>
          </>
        }
      >
        {deleteError ? (
          <div
            role="alert"
            data-testid="delete-backtest-error"
            className="rounded-md border border-loss/40 bg-loss-soft/40 px-2.5 py-2 text-[11px] leading-4 text-rose-300"
          >
            {deleteError}
          </div>
        ) : (
          <p className="text-[10.5px] leading-4 text-zinc-600">
            Seeded fixture runs may be regenerated by the API on restart; persisted engine runs are deleted from disk.
          </p>
        )}
      </Modal>
    </div>
  );
}

function columnsSignature(compareIds: string[], selectedRunId: string | null): string {
  return `${compareIds.join("|")}::${selectedRunId ?? ""}`;
}

function ParameterDiffAccordion({
  runs,
  details,
}: {
  runs: BacktestRunSummary[];
  details: Record<string, BacktestRunSummary | null>;
}) {
  const pair = runs.slice(0, 2);
  const [a, b] = pair;
  const snapshotA = a ? details[a.id]?.strategy_config ?? null : null;
  const snapshotB = b ? details[b.id]?.strategy_config ?? null : null;
  const rows = useMemo(() => diffStrategyConfigs(snapshotA, snapshotB), [snapshotA, snapshotB]);
  return (
    <Panel className="col-span-12 xl:col-span-5" title="Parameter diff" flush>
      <details className="group" data-testid="parameter-diff">
        <summary className="flex cursor-pointer select-none items-center gap-2 px-3 py-2 text-[11px] text-zinc-400 hover:text-zinc-200">
          <span className="text-zinc-600 transition-transform group-open:rotate-90">›</span>
          {pair.length < 2
            ? "Select at least two runs to compare parameters"
            : `${a?.id ?? "Run A"} vs ${b?.id ?? "Run B"}`}
          {pair.length === 2 && (snapshotA === null || snapshotB === null) ? (
            <Badge tone="warn" className="ml-auto">snapshot unavailable</Badge>
          ) : pair.length === 2 ? (
            <Badge tone={rows.length > 0 ? "accent" : "neutral"} className="ml-auto">
              {rows.length} changed
            </Badge>
          ) : null}
        </summary>
        <div className="max-h-[280px] overflow-auto border-t border-hairline">
          {pair.length < 2 ? (
            <p className="px-3 py-3 text-[11px] text-zinc-600">Check two or three runs in the history table.</p>
          ) : snapshotA === null || snapshotB === null ? (
            <p
              data-testid="parameter-diff-unavailable"
              className="px-3 py-3 text-[11px] leading-4 text-amber-300/90"
            >
              Parameter snapshot unavailable for this legacy run. Re-run the backtest with the current engine to
              capture a strategy_config snapshot.
            </p>
          ) : rows.length === 0 ? (
            <p className="px-3 py-3 text-[11px] text-zinc-600">No parameter differences between these runs.</p>
          ) : (
            <table className="data-table" data-testid="parameter-diff-table">
              <thead>
                <tr>
                  <th>Parameter</th>
                  <th className="w-36">{a?.id}</th>
                  <th className="w-36">{b?.id}</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr key={row.path}>
                    <td className="font-mono text-[10.5px] text-zinc-400">{row.path}</td>
                    <td className="font-mono text-[10.5px] text-rose-300">{formatDiffValue(row.a)}</td>
                    <td className="font-mono text-[10.5px] text-emerald-300">{formatDiffValue(row.b)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </details>
    </Panel>
  );
}

function CostBreakdown({ run }: { run: BacktestRunSummary }) {
  const breakdown = run.cost_breakdown ?? run.costs ?? null;
  if (!breakdown) {
    return (
      <div data-testid="cost-breakdown" className="mb-2 rounded border border-hairline bg-canvas/40 px-2 py-1.5 text-[10px] text-zinc-600">
        Cost breakdown not available for fixture/legacy runs.
      </div>
    );
  }
  const rows = [
    ["Broker commissions", breakdown.broker_commissions ?? breakdown.total_commissions],
    ["Exchange & regulatory fees", breakdown.exchange_and_regulatory_fees ?? breakdown.exchange_reg_fees ?? breakdown.exchange_and_reg_fees],
    ["Slippage", breakdown.slippage ?? breakdown.slippage_cost],
    ["Funding", breakdown.funding],
    ["Roll & BAG execution", breakdown.roll_cost_total ?? breakdown.roll_costs ?? breakdown.roll_cost],
    ["Forced liquidation", breakdown.forced_liquidation_cost],
    ["Total execution costs", breakdown.total_execution_costs],
    ["Total costs", breakdown.total_costs ?? breakdown.total_drag ?? breakdown.total_cost],
  ] as const;
  return (
    <div data-testid="cost-breakdown" className="mb-2 rounded border border-edge bg-canvas/50 px-2.5 py-2">
      <div className="mb-1 text-[10px] font-medium uppercase tracking-[0.04em] text-zinc-500">Cost breakdown</div>
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        {rows.map(([label, value]) => (
          <div key={label}>
            <div className="text-[9px] text-zinc-600">{label}</div>
            <div className="font-mono text-[11px] tabular-nums text-zinc-300">
              {typeof value === "number" ? formatCurrency(Math.abs(value)) : "—"}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function DataOriginBadge({
  origin,
  cachedBars,
  mode,
  timeframe,
}: {
  origin: string | null | undefined;
  cachedBars: number | CachedBarsMeta | null | undefined;
  mode: string | null | undefined;
  timeframe?: string | null;
}) {
  const meta = dataOriginBadge(origin);
  const detail = cachedBarsDetail(cachedBars);
  return (
    <span
      data-testid={meta.real ? "data-origin-real" : "data-origin-synthetic"}
      data-origin={meta.real ? "real_parquet" : "synthetic_fixture"}
      title={[meta.title, detail, mode ? `mode ${mode}` : null, timeframe ? `timeframe ${timeframe}` : null].filter(Boolean).join(" · ")}
      className="inline-flex flex-col gap-0.5"
    >
      <Badge tone={meta.tone}>{meta.label}</Badge>
      <span className="font-mono text-[8.5px] leading-3 text-zinc-600">
        {timeframe ? `TF ${timeframe}` : ""}{detail ? ` · ${detail}` : ""}
      </span>
    </span>
  );
}

function MetricGroupRows({
  group,
  runs,
  selectedRunId,
}: {
  group: MetricGroup;
  runs: BacktestRunSummary[];
  selectedRunId: string | null;
}) {
  return (
    <>
      <tr>
        <td colSpan={runs.length + 1} className="h-7 bg-surface/60 px-3 text-[10.5px] font-semibold uppercase tracking-[0.05em] text-zinc-500">
          {group.title}
        </td>
      </tr>
      {group.rows.map((row) => {
        const values = runs.map((run) => row.value(run));
        const finite = values.filter((value): value is number => value !== null && Number.isFinite(value));
        const best =
          finite.length > 0
            ? row.better === "high"
              ? Math.max(...finite)
              : Math.min(...finite)
            : null;
        return (
          <tr key={row.label}>
            <td className="sticky left-0 z-10 bg-panel text-[12px] text-zinc-400">{row.label}</td>
            {values.map((value, index) => {
              const isBest = runs.length > 1 && value !== null && best !== null && value === best;
              const run = runs[index];
              return (
                <td
                  key={`${row.label}-${run?.id ?? index}`}
                  className={cn(
                    "num text-right text-[12px]",
                    isBest ? "font-semibold text-emerald-400" : run?.id === selectedRunId ? "text-zinc-100" : "text-zinc-300",
                  )}
                >
                  {value === null ? "—" : row.format(value)}
                </td>
              );
            })}
          </tr>
        );
      })}
    </>
  );
}

const OVERLAY_COLORS = ["#3B82F6", "#F59E0B", "#8B5CF6"];

interface FetchedSheet {
  run: BacktestRunSummary;
  sheet: TearsheetResponse | null;
}

/**
 * Multi-run tear sheet: overlays ALL checked runs on the equity and underwater
 * charts (rebased equity, aligned by timestamp, distinct colours + legend) and
 * keeps the rolling-Sharpe / attribution panel for the selected run.
 */
function TearsheetOverlay({
  runs,
  selectedRun,
}: {
  runs: BacktestRunSummary[];
  selectedRun: BacktestRunSummary | null;
}) {
  const [fetched, setFetched] = useState<FetchedSheet[]>([]);
  const [loading, setLoading] = useState(true);
  const [chartMode, setChartMode] = useState<ChartMode>("sharpe");

  const fetchRunIds = useMemo(() => {
    const ids = runs.map((run) => run.id);
    if (selectedRun && !ids.includes(selectedRun.id)) ids.push(selectedRun.id);
    return ids;
  }, [runs, selectedRun]);
  const fetchKey = fetchRunIds.join("|");

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    const allRuns = [...runs];
    if (selectedRun && !allRuns.some((run) => run.id === selectedRun.id)) allRuns.push(selectedRun);
    void Promise.all(
      allRuns.map(async (run) => {
        const result = await getTearsheet(run.id);
        return { run, sheet: result.data } satisfies FetchedSheet;
      }),
    ).then((results) => {
      if (cancelled) return;
      setFetched(results);
      setLoading(false);
    });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fetchKey]);

  const sheetFor = useMemo(() => {
    const map = new Map<string, TearsheetResponse | null>();
    for (const entry of fetched) map.set(entry.run.id, entry.sheet);
    return map;
  }, [fetched]);

  const overlayRuns = runs.length > 0 ? runs : selectedRun ? [selectedRun] : [];
  const equitySeries: OverlaySeries[] = [];
  const underwaterSeries: OverlaySeries[] = [];
  overlayRuns.forEach((run, index) => {
    const sheet = sheetFor.get(run.id);
    if (!sheet) return;
    const firstEquity = sheet.equity_curve[0]?.value ?? 0;
    if (firstEquity > 0) {
      equitySeries.push({
        id: run.id,
        label: `${run.id} · ${run.strategy_id}`,
        color: OVERLAY_COLORS[index % OVERLAY_COLORS.length] ?? "#3B82F6",
        data: sheet.equity_curve.map((point) => ({
          time: point.time,
          value: (point.value / firstEquity) * 100,
        })),
      });
    }
    if (sheet.underwater.length > 0) {
      underwaterSeries.push({
        id: run.id,
        label: `${run.id} · ${run.strategy_id}`,
        color: OVERLAY_COLORS[index % OVERLAY_COLORS.length] ?? "#3B82F6",
        data: sheet.underwater,
      });
    }
  });

  const selectedSheet = selectedRun ? sheetFor.get(selectedRun.id) ?? null : null;
  const selectedMetrics = selectedRun?.metrics ?? selectedSheet?.metrics ?? null;

  if (loading && fetched.length === 0) {
    return (
      <div className="flex flex-col gap-2">
        <Skeleton className="h-[150px]" />
        <Skeleton className="h-[92px]" />
        <Skeleton className="h-[118px]" />
      </div>
    );
  }

  if (overlayRuns.length === 0) {
    return (
      <EmptyState
        title="No tear sheet selected"
        description="Check runs in the history table to overlay their equity and underwater curves."
      />
    );
  }

  return (
    <div className="flex flex-col">
      <div
        className="mb-1 flex flex-wrap items-center justify-between gap-2"
        data-testid="overlay-legend"
        data-run-count={equitySeries.length}
      >
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
          {overlayRuns.map((run, index) => {
            const color = OVERLAY_COLORS[index % OVERLAY_COLORS.length] ?? "#3B82F6";
            const available = Boolean(sheetFor.get(run.id));
            return (
              <span
                key={run.id}
                className="flex items-center gap-1.5 text-[11px]"
                data-run-id={run.id}
                title={`${run.id} · ${run.strategy_id}${available ? "" : " · tear sheet unavailable"}`}
              >
                <span className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: color }} />
                <span className="font-mono text-zinc-300">{run.id}</span>
                <span className="text-zinc-600">{run.strategy_id}</span>
                {!available ? <Badge tone="warn">unavailable</Badge> : null}
              </span>
            );
          })}
        </div>
        <span className="text-[10.5px] text-zinc-600">equity rebased to 100</span>
      </div>

      <div data-testid="equity-overlay" data-series-count={equitySeries.length}>
        {equitySeries.length > 0 ? (
          <MultiSeriesChart
            series={equitySeries}
            height={152}
            formatValue={(value) => value.toFixed(1)}
            ariaLabel="Equity curve overlay (rebased to 100) for checked backtest runs"
          />
        ) : (
          <EmptyState title="Equity series unavailable" description="Tear sheets could not be loaded for the checked runs." className="py-6" />
        )}
      </div>

      <div className="mb-1 mt-2 flex items-baseline justify-between border-t border-hairline pt-2">
        <span className="metric-label">Underwater drawdown</span>
        {selectedMetrics ? (
          <span className="font-mono text-[11px] tabular-nums text-loss">
            selected max {formatPctPlain(selectedMetrics.max_drawdown)} ·{" "}
            {formatNumber(selectedMetrics.max_drawdown_duration_days, 0)}d
          </span>
        ) : null}
      </div>
      <div data-testid="underwater-overlay" data-series-count={underwaterSeries.length}>
        {underwaterSeries.length > 0 ? (
          <MultiSeriesChart
            series={underwaterSeries}
            height={92}
            showTimeAxis={false}
            formatValue={(value) => formatPctPlain(value)}
            ariaLabel="Underwater drawdown overlay for checked backtest runs"
          />
        ) : (
          <EmptyState title="Underwater series unavailable" description="Tear sheets could not be loaded for the checked runs." className="py-5" />
        )}
      </div>

      <div className="mb-1 mt-2 flex items-center justify-between border-t border-hairline pt-2">
        <span className="metric-label">
          {selectedRun ? `${selectedRun.id} · ` : ""}
          {chartMode === "sharpe" ? "Rolling Sharpe (63d)" : "Spot vs roll attribution"}
        </span>
        <Segmented
          ariaLabel="Third chart"
          value={chartMode}
          onChange={setChartMode}
          options={[
            { value: "sharpe", label: "Sharpe" },
            { value: "attribution", label: "Attribution" },
          ]}
        />
      </div>
      {chartMode === "sharpe" ? (
        selectedSheet ? (
          <TimeSeriesChart
            data={selectedSheet.rolling_sharpe}
            kind="line"
            height={118}
            color="#A1A1AA"
            formatValue={(value) => value.toFixed(2)}
            ariaLabel="Rolling Sharpe ratio for the selected run"
          />
        ) : (
          <EmptyState title="Rolling Sharpe unavailable" description="The selected run has no rolling-Sharpe series." className="py-5" />
        )
      ) : selectedSheet && selectedSheet.roll_yield_attribution.length > 0 ? (
        <div className="pt-1">
          <AttributionBars
            rows={selectedSheet.roll_yield_attribution}
            format={(value) => formatSignedCurrency(value, true)}
          />
        </div>
      ) : (
        <EmptyState
          title="No attribution rows"
          description="The engine reported no per-root spot/roll attribution for the selected run."
          className="py-5"
        />
      )}
    </div>
  );
}
