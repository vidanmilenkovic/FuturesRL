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
import { getTearsheet, listBacktests, listStrategies, runBacktest } from "@/lib/api";
import { cn } from "@/lib/cn";
import {
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
  const [wfSplits, setWfSplits] = useState(5);
  const [purgeBars, setPurgeBars] = useState("");
  const [embargoBars, setEmbargoBars] = useState("");
  const [running, setRunning] = useState(false);
  const [runnerMessage, setRunnerMessage] = useState<string | null>(null);
  const [runnerDemo, setRunnerDemo] = useState(false);

  useEffect(() => {
    if (backtests.data) {
      setRuns(backtests.data.runs);
      setSelectedRunId((current) =>
        current && backtests.data?.runs.some((run) => run.id === current)
          ? current
          : (backtests.data?.runs[0]?.id ?? null),
      );
      setCompareIds((current) => {
        if (current.length > 0) return current;
        return backtests.data?.runs.slice(0, 2).map((run) => run.id) ?? [];
      });
    }
  }, [backtests.data]);

  useEffect(() => {
    if (!strategyId && strategies.data && strategies.data.length > 0) {
      setStrategyId(strategies.data[0]?.id ?? "");
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
        ? `Run ${run.id} completed · ${formatInteger(run.n_trades ?? 0)} trades · ${formatDuration(run.duration_seconds)}`
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
      <Panel className="col-span-12" title="Experiment runner" flush>
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
                <th className="w-32">Kind</th>
                <th className="text-right">CAGR</th>
                <th className="text-right">Sharpe</th>
                <th className="text-right">Max DD</th>
                <th className="text-right">Rachev</th>
                <th className="text-right">Trades</th>
                <th className="text-right">Rebal.</th>
                <th className="text-right">Duration</th>
                <th className="w-32">Created</th>
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
                      {run.kind === "walk_forward" ? (
                        <Badge tone="accent" title={`Purged & embargoed walk-forward validation · ${run.wf_splits ?? "?"} folds`}>
                          {`WF · ${run.wf_splits ?? "—"} folds`}
                        </Badge>
                      ) : (
                        <span className="text-[11px] text-zinc-600">—</span>
                      )}
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
          selectedRun?.metrics ? (
            <div className="flex items-center gap-3 font-mono text-[11px] tabular-nums">
              <span className="text-zinc-500">
                CAGR <span className={selectedRun.metrics.cagr >= 0 ? "text-gain" : "text-loss"}>{formatPct(selectedRun.metrics.cagr)}</span>
              </span>
              <span className="text-zinc-500">
                Sharpe <span className="text-zinc-200">{selectedRun.metrics.sharpe.toFixed(2)}</span>
              </span>
              <span className="text-zinc-500">
                Max DD <span className="text-loss">{formatPctPlain(selectedRun.metrics.max_drawdown)}</span>
              </span>
            </div>
          ) : null
        }
      >
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
    </div>
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
