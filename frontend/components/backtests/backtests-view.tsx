"use client";

import { useEffect, useMemo, useState } from "react";

import { AttributionBars } from "@/components/charts/plots";
import { TimeSeriesChart } from "@/components/charts/time-series";
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
  formatCompactUsd,
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
      walk_forward: true,
      random_seed: 7,
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
        : `Backend unreachable — deterministic demo tear sheet ${run.id}.`,
    );
    setRunning(false);
  };

  const columns = compareRuns.length > 0 ? compareRuns : selectedRun ? [selectedRun] : [];
  const strategiesOptions: StrategyConfig[] = strategies.data ?? [];
  const invalid = !strategyId || start >= end || initialEquity <= 0;

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
          <div className="flex items-center gap-2 pb-0.5">
            <Button variant="primary" onClick={() => void submitRun()} disabled={running || invalid}>
              {running ? "Running…" : "Run backtest"}
            </Button>
            <Button variant="ghost" onClick={() => backtests.reload()} disabled={running}>
              Refresh
            </Button>
          </div>
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
                      <Badge tone={run.data_source === "engine" ? "gain" : "warn"}>
                        {run.data_source === "engine" ? "Engine" : "Fixture"}
                      </Badge>
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
                  <td colSpan={13} className="py-8 text-center text-zinc-600">
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
        title={selectedRun ? `Tear sheet · ${selectedRun.id}` : "Tear sheet"}
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
          <TearsheetStack key={selectedRunId} runId={selectedRunId} />
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

function TearsheetStack({ runId }: { runId: string }) {
  const tearsheet = useAsyncData(() => getTearsheet(runId));
  const [chartMode, setChartMode] = useState<ChartMode>("sharpe");

  if (tearsheet.loading && !tearsheet.data) {
    return (
      <div className="flex flex-col gap-2">
        <Skeleton className="h-[168px]" />
        <Skeleton className="h-[104px]" />
        <Skeleton className="h-[150px]" />
      </div>
    );
  }

  const sheet: TearsheetResponse | null = tearsheet.data;
  if (!sheet) {
    return (
      <EmptyState
        title="Tear sheet unavailable"
        description={`The API did not return a tear sheet for run ${runId}.`}
      />
    );
  }

  const metrics = sheet.metrics;

  return (
    <div className="flex flex-col">
      <div className="mb-1 flex items-baseline justify-between">
        <span className="metric-label">Equity curve</span>
        <span className="font-mono text-[11px] tabular-nums text-zinc-500">
          {formatCurrency(sheet.equity_curve[0]?.value ?? 0, true)} →{" "}
          <span className={cn(metrics.cagr >= 0 ? "text-gain" : "text-loss")}>
            {formatCurrency(sheet.equity_curve[sheet.equity_curve.length - 1]?.value ?? 0, true)}
          </span>
        </span>
      </div>
      <TimeSeriesChart
        data={sheet.equity_curve}
        kind="area"
        height={160}
        color="#3B82F6"
        fillColor="rgba(59,130,246,0.14)"
        formatValue={(value) => formatCompactUsd(value)}
        ariaLabel="Backtest equity curve"
      />

      <div className="mb-1 mt-2 flex items-baseline justify-between border-t border-hairline pt-2">
        <span className="metric-label">Underwater drawdown</span>
        <span className="font-mono text-[11px] tabular-nums text-loss">
          max {formatPctPlain(metrics.max_drawdown)} · {formatNumber(metrics.max_drawdown_duration_days, 0)}d
        </span>
      </div>
      <TimeSeriesChart
        data={sheet.underwater}
        kind="baseline"
        height={100}
        showTimeAxis={false}
        formatValue={(value) => formatPctPlain(value)}
        ariaLabel="Backtest underwater drawdown"
      />

      <div className="mb-1 mt-2 flex items-center justify-between border-t border-hairline pt-2">
        <span className="metric-label">
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
        <TimeSeriesChart
          data={sheet.rolling_sharpe}
          kind="line"
          height={132}
          color="#A1A1AA"
          formatValue={(value) => value.toFixed(2)}
          ariaLabel="Rolling Sharpe ratio"
        />
      ) : sheet.roll_yield_attribution.length > 0 ? (
        <div className="pt-1">
          <AttributionBars
            rows={sheet.roll_yield_attribution}
            format={(value) => formatSignedCurrency(value, true)}
          />
        </div>
      ) : (
        <EmptyState
          title="No attribution rows"
          description="The engine reported no per-root spot/roll attribution for this run."
          className="py-6"
        />
      )}
    </div>
  );
}
