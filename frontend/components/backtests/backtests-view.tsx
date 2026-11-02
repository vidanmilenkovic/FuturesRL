"use client";

import { useEffect, useMemo, useState } from "react";

import { LightweightChart } from "@/components/charts/lightweight-chart";
import { AttributionBars, TailScatter, UnderwaterChart } from "@/components/charts/svg-charts";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Field, Segmented, Select, Skeleton, TextInput } from "@/components/ui/controls";
import { Panel } from "@/components/ui/panel";
import { getTearsheet, listBacktests, listStrategies, runBacktest } from "@/lib/api";
import { cn } from "@/lib/cn";
import {
  formatCurrency,
  formatDateTime,
  formatInteger,
  formatNumber,
  formatPct,
  formatPctPlain,
} from "@/lib/format";
import { useAsyncData } from "@/lib/use-async-data";
import type {
  BacktestRunConfig,
  BacktestRunSummary,
  MetricsSet,
  StrategyConfig,
  TearsheetResponse,
} from "@/lib/types";

const METRIC_ROWS: { key: keyof MetricsSet; label: string; format: (value: number) => string; higherIsBetter: boolean }[] = [
  { key: "cagr", label: "CAGR", format: (value) => formatPct(value), higherIsBetter: true },
  { key: "annual_volatility", label: "annual vol", format: (value) => formatPctPlain(value), higherIsBetter: false },
  { key: "sharpe", label: "sharpe", format: (value) => value.toFixed(2), higherIsBetter: true },
  { key: "sortino", label: "sortino", format: (value) => value.toFixed(2), higherIsBetter: true },
  { key: "calmar", label: "calmar", format: (value) => value.toFixed(2), higherIsBetter: true },
  { key: "omega", label: "omega", format: (value) => value.toFixed(2), higherIsBetter: true },
  { key: "max_drawdown", label: "max drawdown", format: (value) => formatPctPlain(value), higherIsBetter: false },
  { key: "max_drawdown_duration_days", label: "max dd duration", format: (value) => `${Math.round(value)}d`, higherIsBetter: false },
  { key: "var_95", label: "VaR 95 (1d)", format: (value) => formatPctPlain(value), higherIsBetter: false },
  { key: "var_99", label: "VaR 99 (1d)", format: (value) => formatPctPlain(value), higherIsBetter: false },
  { key: "cvar_95", label: "CVaR 95", format: (value) => formatPctPlain(value), higherIsBetter: false },
  { key: "cvar_99", label: "CVaR 99", format: (value) => formatPctPlain(value), higherIsBetter: false },
  { key: "rachev_ratio", label: "rachev ratio", format: (value) => value.toFixed(2), higherIsBetter: true },
  { key: "rom", label: "return on margin", format: (value) => formatPct(value), higherIsBetter: true },
  { key: "avg_margin_utilization", label: "avg margin util", format: (value) => formatPctPlain(value), higherIsBetter: false },
  { key: "turnover", label: "turnover", format: (value) => `${value.toFixed(2)}×`, higherIsBetter: false },
  { key: "implementation_shortfall_ticks", label: "impl. shortfall", format: (value) => `${value.toFixed(2)} ticks`, higherIsBetter: false },
];

/**
 * View 5 — Backtest Lab & Tear Sheet Compare.
 *
 * Runs offline deterministic backtests through POST /api/backtests, compares up
 * to three runs against each other and renders the institutional tear sheet:
 * rolling Sharpe, underwater drawdown, Rachev tail plot and roll-yield
 * attribution.
 */
export function BacktestsView() {
  const backtests = useAsyncData(listBacktests);
  const strategies = useAsyncData(listStrategies);
  const [runs, setRuns] = useState<BacktestRunSummary[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [compareIds, setCompareIds] = useState<string[]>([]);

  useEffect(() => {
    if (backtests.data) {
      setRuns(backtests.data.runs);
      setSelectedRunId((current) =>
        current && backtests.data!.runs.some((run) => run.id === current)
          ? current
          : (backtests.data!.runs[0]?.id ?? null),
      );
      setCompareIds((current) => {
        if (current.length > 0) return current;
        return backtests.data!.runs.slice(0, 2).map((run) => run.id);
      });
    }
  }, [backtests.data]);

  const selectedRun = useMemo(
    () => runs.find((run) => run.id === selectedRunId) ?? null,
    [runs, selectedRunId],
  );
  const compareRuns = useMemo(
    () => compareIds.map((id) => runs.find((run) => run.id === id)).filter(Boolean) as BacktestRunSummary[],
    [compareIds, runs],
  );

  const toggleCompare = (id: string) => {
    setCompareIds((current) =>
      current.includes(id) ? current.filter((entry) => entry !== id) : [...current, id].slice(-3),
    );
  };

  if (backtests.loading && runs.length === 0) {
    return (
      <div className="space-y-3">
        <Skeleton className="h-24" />
        <Skeleton className="h-[260px]" />
        <div className="grid gap-3 xl:grid-cols-2">
          <Skeleton className="h-[280px]" />
          <Skeleton className="h-[280px]" />
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-1 gap-3 xl:grid-cols-3">
        <RunPanel
          strategies={strategies.data ?? []}
          onCreated={(run) => {
            setRuns((current) => [run, ...current]);
            setSelectedRunId(run.id);
            setCompareIds((current) => [...current, run.id].slice(-3));
          }}
          onRefresh={() => backtests.reload()}
        />

        <Panel
          title="run history"
          subtitle="offline deterministic engine · walk-forward"
          className="xl:col-span-2"
          actions={<Badge tone="info">{runs.length} runs</Badge>}
          contentClassName="px-3 pt-2 pb-0"
        >
          <div className="-mx-3 max-h-[308px] overflow-auto">
            <table className="data-table min-w-[880px]">
              <thead>
                <tr>
                  <th>run</th>
                  <th>strategy</th>
                  <th>status</th>
                  <th className="num">cagr</th>
                  <th className="num">sharpe</th>
                  <th className="num">max dd</th>
                  <th className="num">calmar</th>
                  <th className="num">rachev</th>
                  <th className="num">trades</th>
                  <th className="num">rebal.</th>
                  <th className="num">duration</th>
                  <th>created</th>
                  <th>compare</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((run) => {
                  const metrics = run.metrics;
                  const inCompare = compareIds.includes(run.id);
                  return (
                    <tr
                      key={run.id}
                      className={cn("cursor-pointer", selectedRunId === run.id ? "bg-signal-info/[0.06]" : "")}
                      onClick={() => setSelectedRunId(run.id)}
                    >
                      <td className="font-mono text-terminal-100">{run.id}</td>
                      <td className="text-terminal-300">{run.strategy_id}</td>
                      <td>
                        <Badge tone={run.status === "completed" ? "up" : run.status === "failed" ? "critical" : "info"} dot>
                          {run.status}
                        </Badge>
                      </td>
                      <td className={cn("num", (metrics?.cagr ?? 0) >= 0 ? "text-signal-up" : "text-signal-down")}>
                        {metrics ? formatPct(metrics.cagr) : "—"}
                      </td>
                      <td className="num text-terminal-100">{metrics ? metrics.sharpe.toFixed(2) : "—"}</td>
                      <td className="num text-signal-down">{metrics ? formatPctPlain(metrics.max_drawdown) : "—"}</td>
                      <td className="num text-terminal-100">{metrics ? metrics.calmar.toFixed(2) : "—"}</td>
                      <td className="num text-signal-violet">{metrics ? metrics.rachev_ratio.toFixed(2) : "—"}</td>
                      <td className="num text-terminal-300">
                        {run.n_trades === null || run.n_trades === undefined ? "—" : formatInteger(run.n_trades)}
                      </td>
                      <td className="num text-terminal-300">
                        {run.n_rebalances === null || run.n_rebalances === undefined ? "—" : formatInteger(run.n_rebalances)}
                      </td>
                      <td className="num text-terminal-400">
                        {run.duration_seconds === null ? "—" : `${formatNumber(run.duration_seconds, 1)}s`}
                      </td>
                      <td className="text-terminal-400">{formatDateTime(run.created_at)}</td>
                      <td>
                        <input
                          type="checkbox"
                          aria-label={`Compare run ${run.id}`}
                          checked={inCompare}
                          onChange={() => toggleCompare(run.id)}
                          onClick={(event) => event.stopPropagation()}
                          className="h-3.5 w-3.5 accent-[#22d3ee]"
                        />
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Panel>
      </div>

      {/* Comparison */}
      <Panel
        title="tear sheet compare"
        subtitle={`${compareRuns.length} run(s) selected · metrics highlighted where best`}
        actions={<Badge tone="violet">α = 5% tail</Badge>}
      >
        {compareRuns.length === 0 ? (
          <p className="py-6 text-center font-mono text-[10px] uppercase tracking-[0.12em] text-terminal-400">
            select runs from the history table to compare
          </p>
        ) : (
          <>
            <NormalizedEquityChart runs={compareRuns} />
            <div className="mt-2 -mx-3 overflow-x-auto">
              <table className="data-table min-w-[720px]">
                <thead>
                  <tr>
                    <th>metric</th>
                    {compareRuns.map((run) => (
                      <th key={run.id} className="num">
                        {run.id} · {run.strategy_id}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {METRIC_ROWS.map((row) => {
                    const values = compareRuns.map((run) => run.metrics?.[row.key] ?? 0);
                    const best = row.higherIsBetter ? Math.max(...values) : Math.min(...values);
                    return (
                      <tr key={row.key}>
                        <td className="text-terminal-300">{row.label}</td>
                        {values.map((value, index) => (
                          <td
                            key={`${row.key}-${index}`}
                            className={cn(
                              "num",
                              value === best && values.length > 1
                                ? "font-semibold text-signal-up"
                                : "text-terminal-200",
                            )}
                          >
                            {row.format(value)}
                          </td>
                        ))}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </>
        )}
      </Panel>

      {selectedRunId ? (
        <TearsheetLoader key={selectedRunId} runId={selectedRunId} run={selectedRun} />
      ) : null}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Run form
// ---------------------------------------------------------------------------

function RunPanel({
  strategies,
  onCreated,
  onRefresh,
}: {
  strategies: StrategyConfig[];
  onCreated: (run: BacktestRunSummary) => void;
  onRefresh: () => void;
}) {
  const today = useMemo(() => new Date(), []);
  const isoDay = (offsetDays: number) =>
    new Date(today.getTime() + offsetDays * 86_400_000).toISOString().slice(0, 10);
  const [strategyId, setStrategyId] = useState("");
  const [start, setStart] = useState(isoDay(-760));
  const [end, setEnd] = useState(isoDay(0));
  const [equity, setEquity] = useState(5_000_000);
  const [walkForward, setWalkForward] = useState(true);
  const [running, setRunning] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [demo, setDemo] = useState(false);

  useEffect(() => {
    if (strategies.length > 0 && strategyId === "") {
      setStrategyId(strategies[0]!.id);
    }
  }, [strategies, strategyId]);

  const invalid = start >= end || equity <= 0 || strategyId === "";

  const submit = async () => {
    if (invalid) return;
    setRunning(true);
    setMessage(null);
    const config: BacktestRunConfig = {
      strategy_id: strategyId,
      start,
      end,
      initial_equity: equity,
      rebalance_frequency: "W-FRI",
      walk_forward: walkForward,
      random_seed: 7,
    };
    const result = await runBacktest(config);
    const run = result.data;
    onCreated(run);
    setDemo(result.source === "demo");
    setMessage(
      result.source === "live"
        ? `run ${run.id} queued on the offline engine`
        : `backend offline — run ${run.id} is a deterministic demo tear sheet`,
    );
    setRunning(false);
    onRefresh();
  };

  return (
    <Panel title="run backtest" subtitle="POST /api/backtests · offline engine" actions={<Badge tone={demo ? "warn" : "info"}>{demo ? "demo" : "engine"}</Badge>}>
      <div className="grid grid-cols-2 gap-2">
        <Field label="strategy" htmlFor="bt-strategy" className="col-span-2">
          <Select id="bt-strategy" value={strategyId} onChange={(event) => setStrategyId(event.target.value)} disabled={running}>
            <option value="">— select strategy —</option>
            {strategies.map((strategy) => (
              <option key={strategy.id} value={strategy.id}>
                {strategy.id}
              </option>
            ))}
          </Select>
        </Field>
        <Field label="start" htmlFor="bt-start">
          <TextInput id="bt-start" type="date" value={start} onChange={(event) => setStart(event.target.value)} disabled={running} />
        </Field>
        <Field label="end" htmlFor="bt-end">
          <TextInput id="bt-end" type="date" value={end} onChange={(event) => setEnd(event.target.value)} disabled={running} />
        </Field>
        <Field label="initial equity" htmlFor="bt-equity">
          <TextInput
            id="bt-equity"
            type="number"
            step={1000}
            value={equity}
            onChange={(event) => setEquity(Number(event.target.value))}
            disabled={running}
          />
        </Field>
        <Field label="rebalance" htmlFor="bt-freq">
          <Select id="bt-freq" defaultValue="W-FRI" disabled>
            <option value="W-FRI">W-FRI</option>
          </Select>
        </Field>
      </div>

      <label className="mt-2 flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.1em] text-terminal-300">
        <input
          type="checkbox"
          checked={walkForward}
          onChange={(event) => setWalkForward(event.target.checked)}
          disabled={running}
          className="h-3.5 w-3.5 accent-[#22d3ee]"
        />
        purged &amp; embargoed walk-forward
      </label>

      {invalid ? (
        <p className="mt-2 font-mono text-[10px] uppercase tracking-[0.08em] text-signal-down">
          {start >= end ? "end must be after start" : "select a strategy and positive equity"}
        </p>
      ) : null}
      {message ? (
        <p className={cn("mt-2 font-mono text-[10px] uppercase tracking-[0.08em]", demo ? "text-signal-warn" : "text-signal-up")}>
          {message}
        </p>
      ) : null}

      <div className="mt-3 flex items-center gap-2">
        <Button variant="primary" size="sm" onClick={() => void submit()} disabled={invalid || running}>
          {running ? "running…" : "run backtest"}
        </Button>
        <Button variant="ghost" size="sm" onClick={onRefresh} disabled={running}>
          refresh
        </Button>
      </div>
      <p className="mt-2 border-t border-terminal-600/40 pt-2 font-mono text-[9px] leading-relaxed text-terminal-500">
        Runs use the real BacktestEngine with deterministic fixture market data; no Databento or IBKR calls are
        made. Metrics include Rachev ratio (α=5%) to validate the LLM Risk-First engine.
      </p>
    </Panel>
  );
}

// ---------------------------------------------------------------------------
// Normalized multi-run equity chart
// ---------------------------------------------------------------------------

const RUN_COLORS = ["#22d3ee", "#34d399", "#a78bfa"];

function NormalizedEquityChart({ runs }: { runs: BacktestRunSummary[] }) {
  // Each run's curve is fetched lazily below; for the comparison header we use
  // the selected run's own tear sheet chart instead. Curves are normalized to
  // 100 at inception so strategies with different AUM are comparable.
  const [series, setSeries] = useState<{ id: string; points: { time: string; value: number }[] }[]>([]);

  useEffect(() => {
    let cancelled = false;
    Promise.all(
      runs.map(async (run) => {
        const result = await getTearsheet(run.id);
        const points = result.data?.equity_curve ?? [];
        const base = points[0]?.value ?? 1;
        if (base <= 0) return { id: run.id, points: [] };
        return {
          id: run.id,
          points: points.map((point) => ({ time: point.time, value: (point.value / base) * 100 })),
        };
      }),
    ).then((all) => {
      if (!cancelled) setSeries(all);
    });
    return () => {
      cancelled = true;
    };
  }, [runs]);

  const aligned = useMemo(() => {
    // Resample all curves onto the union of dates (ffill) for an honest overlay.
    const dates = [...new Set(series.flatMap((entry) => entry.points.map((point) => point.time)))].sort();
    const output: { time: string; value: number }[] = [];
    const cursors = series.map(() => -1);
    const lastValues = series.map(() => 100);
    const maps = series.map(
      (entry) => new Map(entry.points.map((point) => [point.time, point.value])),
    );
    for (const date of dates) {
      maps.forEach((map, index) => {
        const value = map.get(date);
        if (value !== undefined) {
          lastValues[index] = value;
          cursors[index] = 1;
        }
      });
      // Plot the first run (selected), and the average of the rest as comparison.
      if (lastValues[0] !== undefined) {
        output.push({ time: date, value: Number(lastValues[0]!.toFixed(3)) });
      }
    }
    return output;
  }, [series]);

  const comparison = useMemo(() => {
    const output: { time: string; value: number }[] = [];
    const maps = series.slice(1).map((entry) => new Map(entry.points.map((point) => [point.time, point.value])));
    if (maps.length === 0) return output;
    const dates = [...new Set(series.slice(1).flatMap((entry) => entry.points.map((point) => point.time)))].sort();
    const last = maps.map(() => 100);
    for (const date of dates) {
      maps.forEach((map, index) => {
        const value = map.get(date);
        if (value !== undefined) last[index] = value;
      });
      const average = last.reduce((sum, value) => sum + value, 0) / last.length;
      output.push({ time: date, value: Number(average.toFixed(3)) });
    }
    return output;
  }, [series]);

  if (runs.length === 1) {
    return (
      <div className="mb-2">
        <LightweightChart
          data={selectedSeries(series) ?? []}
          kind="line"
          height={200}
          color={RUN_COLORS[0]}
          ariaLabel="Normalized equity curve for the selected backtest run"
        />
      </div>
    );
  }

  return (
    <div className="mb-2">
      <LightweightChart
        data={aligned}
        kind="line"
        comparison={comparison}
        color={RUN_COLORS[0]}
        comparisonColor={RUN_COLORS[1]}
        height={220}
        percent={false}
        ariaLabel="Normalized equity curves (base 100) for compared backtest runs"
      />
      <div className="mt-1 flex flex-wrap gap-3 font-mono text-[9px] uppercase tracking-[0.1em] text-terminal-400">
        <span className="flex items-center gap-1">
          <span className="h-0.5 w-4 bg-signal-info" /> {runs[0]?.id}
        </span>
        <span className="flex items-center gap-1">
          <span className="h-0.5 w-4 bg-signal-violet" /> mean of {Math.max(0, runs.length - 1)} compared run(s)
        </span>
        <span className="ml-auto text-terminal-500">rebased to 100 at inception</span>
      </div>
    </div>
  );
}

function selectedSeries(series: { id: string; points: { time: string; value: number }[] }[]) {
  return series[0]?.points ?? [];
}

// ---------------------------------------------------------------------------
// Tear sheet loader / renderer
// ---------------------------------------------------------------------------

function TearsheetLoader({ runId, run }: { runId: string; run: BacktestRunSummary | null }) {
  const tearsheet = useAsyncData(() => getTearsheet(runId));

  if (tearsheet.loading && !tearsheet.data) {
    return (
      <div className="grid gap-3 xl:grid-cols-2">
        <Skeleton className="h-[240px]" />
        <Skeleton className="h-[240px]" />
        <Skeleton className="h-[280px]" />
        <Skeleton className="h-[280px]" />
      </div>
    );
  }

  const sheet = tearsheet.data;
  if (!sheet) {
    return (
      <Panel title="tear sheet">
        <p className="py-8 text-center font-mono text-xs uppercase tracking-[0.12em] text-signal-down">
          tear sheet unavailable for {runId}
        </p>
      </Panel>
    );
  }

  return <Tearsheet run={run} sheet={sheet} />;
}

function Tearsheet({ run, sheet }: { run: BacktestRunSummary | null; sheet: TearsheetResponse }) {
  const [chart, setChart] = useState<"equity" | "sharpe">("sharpe");
  const metrics = sheet.metrics;

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2 rounded-md border border-terminal-600/60 bg-terminal-850/70 px-3 py-2">
        <Badge tone="violet">{sheet.run_id}</Badge>
        <span className="font-mono text-[10px] uppercase tracking-[0.1em] text-terminal-300">
          {run?.strategy_id ?? "—"} · {run ? `${run.config.start} → ${run.config.end}` : ""}
        </span>
        <span className="hidden font-mono text-[10px] uppercase tracking-[0.1em] text-terminal-500 sm:inline">
          {run ? formatCurrency(run.config.initial_equity) : ""} initial · walk-forward{" "}
          {run?.config.walk_forward ? "on" : "off"}
        </span>
        <div className="ml-auto">
          <Segmented
            size="sm"
            ariaLabel="Tear sheet chart"
            value={chart}
            onChange={setChart}
            options={[
              { value: "sharpe", label: "rolling sharpe" },
              { value: "equity", label: "equity" },
            ]}
          />
        </div>
      </div>

      <div className="grid grid-cols-1 gap-3 xl:grid-cols-3">
        <Panel
          title={chart === "sharpe" ? "rolling sharpe (63d)" : "equity curve"}
          subtitle="annualised · risk-free 4.2%"
          className="xl:col-span-2"
          actions={<Badge tone={metrics.sharpe >= 1 ? "up" : "warn"}>{metrics.sharpe.toFixed(2)} sharpe</Badge>}
        >
          {chart === "sharpe" ? (
            <LightweightChart
              data={sheet.rolling_sharpe}
              kind="line"
              height={220}
              color="#a78bfa"
              ariaLabel="Rolling 63-day Sharpe ratio"
            />
          ) : (
            <LightweightChart
              data={sheet.equity_curve}
              kind="area"
              height={220}
              color="#34d399"
              secondaryColor="rgba(52,211,153,0.2)"
              ariaLabel="Equity curve of the selected backtest run"
            />
          )}
        </Panel>

        <Panel title="metrics summary" subtitle="institutional futures metrics">
          <dl className="grid grid-cols-2 gap-x-3 gap-y-1.5">
            <Metric label="CAGR" value={formatPct(metrics.cagr)} tone={metrics.cagr >= 0 ? "up" : "down"} />
            <Metric label="ann. vol" value={formatPctPlain(metrics.annual_volatility)} />
            <Metric label="sharpe" value={metrics.sharpe.toFixed(2)} tone="info" />
            <Metric label="sortino" value={metrics.sortino.toFixed(2)} />
            <Metric label="calmar" value={metrics.calmar.toFixed(2)} />
            <Metric label="max drawdown" value={formatPctPlain(metrics.max_drawdown)} tone="down" />
            <Metric label="CVaR 95" value={formatPctPlain(metrics.cvar_95)} tone="down" />
            <Metric label="rachev ratio" value={metrics.rachev_ratio.toFixed(2)} tone="violet" />
            <Metric label="RoM" value={formatPct(metrics.rom)} />
            <Metric label="avg margin" value={formatPctPlain(metrics.avg_margin_utilization)} />
          </dl>
          <p className="mt-2 border-t border-terminal-600/40 pt-2 font-mono text-[9px] leading-relaxed text-terminal-500">
            Rachev ratio = mean(α-tail gains) / mean(α-tail losses) with α = 5%. A ratio &gt; 1 is evidence the
            LLM Risk-First overlay truncates the loss tail without giving up the gain tail.
          </p>
        </Panel>
      </div>

      <div className="grid grid-cols-1 gap-3 xl:grid-cols-3">
        <Panel title="underwater drawdown" subtitle="peak-to-trough" className="xl:col-span-2">
          <UnderwaterChart
            points={sheet.underwater}
            ariaLabel="Underwater drawdown curve"
          />
          <div className="mt-1 flex flex-wrap gap-4 border-t border-terminal-600/40 pt-2 font-mono text-[9px] uppercase tracking-[0.1em] text-terminal-400">
            <span>max dd {formatPctPlain(metrics.max_drawdown)}</span>
            <span>duration {metrics.max_drawdown_duration_days}d</span>
            <span>VaR95 {formatPctPlain(metrics.var_95)}</span>
            <span>VaR99 {formatPctPlain(metrics.var_99)}</span>
          </div>
        </Panel>

        <Panel title="rachev tail plot" subtitle="α = 5% gain vs loss tails">
          <TailScatter
            points={sheet.tail_distribution}
            ariaLabel="Tail return distribution used for the Rachev ratio"
          />
          <div className="mt-1 flex items-center justify-between border-t border-terminal-600/40 pt-2 font-mono text-[10px] uppercase tracking-[0.1em]">
            <span className="text-terminal-400">rachev</span>
            <span className={cn("font-semibold", metrics.rachev_ratio >= 1 ? "text-signal-up" : "text-signal-down")}>
              {metrics.rachev_ratio.toFixed(3)}
            </span>
          </div>
        </Panel>
      </div>

      <div className="grid grid-cols-1 gap-3 xl:grid-cols-3">
        <Panel title="roll yield attribution" subtitle="spot PnL vs carry PnL by root" className="xl:col-span-2">
          <AttributionBars
            rows={sheet.roll_yield_attribution}
            ariaLabel="Roll yield attribution by futures root"
          />
          <div className="mt-1 flex items-center gap-4 border-t border-terminal-600/40 pt-2 font-mono text-[9px] uppercase tracking-[0.1em] text-terminal-400">
            <span className="flex items-center gap-1"><span className="h-1.5 w-4 bg-signal-info" /> spot/price PnL</span>
            <span className="flex items-center gap-1"><span className="h-1.5 w-4 bg-signal-up" /> roll yield</span>
            <span className="ml-auto text-terminal-500">implementation shortfall {metrics.implementation_shortfall_ticks.toFixed(2)} ticks</span>
          </div>
        </Panel>

        <Panel title="monthly returns" subtitle="calendar months">
          <div className="grid grid-cols-3 gap-1 sm:grid-cols-4">
            {sheet.monthly_returns.map((month) => {
              const intensity = Math.min(1, Math.abs(month.value) / 0.05);
              return (
                <div
                  key={month.month}
                  className={cn(
                    "rounded-sm border px-1.5 py-1 font-mono text-[9px] tabular-nums",
                    month.value >= 0
                      ? "border-signal-up/25 text-signal-up"
                      : "border-signal-down/25 text-signal-down",
                  )}
                  style={{
                    backgroundColor:
                      month.value >= 0
                        ? `rgba(52,211,153,${0.06 + intensity * 0.22})`
                        : `rgba(251,113,133,${0.06 + intensity * 0.22})`,
                  }}
                >
                  <div className="text-terminal-400">{month.month.slice(2)}</div>
                  <div>{formatPct(month.value, 1)}</div>
                </div>
              );
            })}
          </div>
          <div className="mt-2 flex items-center justify-between border-t border-terminal-600/40 pt-2 font-mono text-[9px] uppercase tracking-[0.1em] text-terminal-400">
            <span>turnover {metrics.turnover.toFixed(2)}×</span>
            <span>comm/gross {formatPctPlain(metrics.commission_to_gross_pnl)}</span>
          </div>
        </Panel>
      </div>

      <Panel title="execution quality & margin efficiency" subtitle="futures-specific attribution">
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Metric label="return on margin" value={formatPct(metrics.rom)} tone="up" />
          <Metric label="avg margin util" value={formatPctPlain(metrics.avg_margin_utilization)} />
          <Metric label="peak margin util" value={formatPctPlain(metrics.peak_margin_utilization)} tone="warn" />
          <Metric label="impl. shortfall" value={`${metrics.implementation_shortfall_ticks.toFixed(2)} ticks`} />
        </div>
      </Panel>
    </div>
  );
}

function Metric({
  label,
  value,
  tone = "neutral",
}: {
  label: string;
  value: string;
  tone?: "neutral" | "up" | "down" | "warn" | "info" | "violet";
}) {
  return (
    <div>
      <dt className="font-mono text-[9px] uppercase tracking-[0.1em] text-terminal-500">{label}</dt>
      <dd
        className={cn(
          "font-mono text-sm tabular-nums",
          tone === "up"
            ? "text-signal-up"
            : tone === "down"
              ? "text-signal-down"
              : tone === "warn"
                ? "text-signal-warn"
                : tone === "info"
                  ? "text-signal-info"
                  : tone === "violet"
                    ? "text-signal-violet"
                    : "text-terminal-100",
        )}
      >
        {value}
      </dd>
    </div>
  );
}
