"use client";

import { useEffect, useMemo, useState } from "react";

import {
  Badge,
  Button,
  Checkbox,
  EmptyState,
  Field,
  Panel,
  Select,
  TextInput,
  type Tone,
} from "@/components/ui/primitives";
import { applyOptimization, getOptimization, listOptimizations, runOptimization } from "@/lib/api";
import { cn } from "@/lib/cn";
import { formatInteger, formatNumber, formatPct } from "@/lib/format";
import {
  activeTunableParameters,
  buildSearchSpace,
  defaultOptunaBounds,
  defaultOptunaSelection,
  formatObjectiveMetric,
  formatParamPath,
  importancePercent,
  structuralParametersForStrategy,
} from "@/lib/studio";
import type {
  OptimizationObjective,
  OptimizationRequest,
  OptimizationStudy,
  OptimizationTrial,
  StrategyConfig,
  StudioCatalogResponse,
  StudioParameter,
} from "@/lib/types";
import { useAsyncData } from "@/lib/use-async-data";

import { ImportanceBars, MiniStat, OrdinalLineChart } from "./controls";

const OBJECTIVES: { value: OptimizationObjective; label: string }[] = [
  { value: "sharpe", label: "Sharpe ratio" },
  { value: "sortino", label: "Sortino ratio" },
  { value: "calmar", label: "Calmar ratio" },
  { value: "cagr", label: "CAGR" },
  { value: "rachev", label: "Rachev ratio" },
  { value: "return_on_margin", label: "Return on margin" },
];

type SortKey = "trial" | "value" | "sharpe" | "cagr" | "dd";

export function OptimizerPanel({
  catalog,
  strategies,
  onApplied,
  onNotify,
}: {
  catalog: StudioCatalogResponse;
  strategies: StrategyConfig[];
  onApplied: (strategy: StrategyConfig) => void;
  onNotify: (message: string, tone: Tone) => void;
}) {
  const studies = useAsyncData(listOptimizations);
  const [baseId, setBaseId] = useState("");
  const [objective, setObjective] = useState<OptimizationObjective>("sharpe");
  const [nTrials, setNTrials] = useState(25);
  const [wfSplits, setWfSplits] = useState(5);
  const [drawdownPenalty, setDrawdownPenalty] = useState(0.2);
  const [bounds, setBounds] = useState<Record<string, { enabled: boolean; low: number; high: number }>>({});
  const [study, setStudy] = useState<OptimizationStudy | null>(null);
  const [running, setRunning] = useState(false);
  const [applying, setApplying] = useState(false);
  const [message, setMessage] = useState<{ text: string; tone: Tone } | null>(null);
  const [sort, setSort] = useState<{ key: SortKey; dir: "asc" | "desc" }>({ key: "value", dir: "desc" });

  const baseStrategy = useMemo(
    () => strategies.find((strategy) => strategy.id === baseId) ?? null,
    [strategies, baseId],
  );
  // v1.5.1: only parameters active in the selected base strategy are tunable.
  const activeParams = useMemo(
    () => (baseStrategy ? activeTunableParameters(baseStrategy, catalog) : []),
    [baseStrategy, catalog],
  );
  const structuralParams = useMemo(
    () => (baseStrategy ? structuralParametersForStrategy(baseStrategy, catalog) : []),
    [baseStrategy, catalog],
  );
  const activeKey = activeParams.map((parameter) => parameter.path).join("|");
  const tunables = activeParams;

  useEffect(() => {
    if (!baseId && strategies.length > 0) {
      // Prefer the canonical TSMOM strategy as the initial study base.
      const preferred =
        strategies.find((strategy) => strategy.id === "tsmom_vol_target") ?? strategies[0];
      setBaseId(preferred?.id ?? "");
    }
  }, [baseId, strategies]);

  // Recompute the active list and curated default checks whenever the base
  // strategy changes (or the catalog's active parameter set changes).
  useEffect(() => {
    if (!baseStrategy) {
      setBounds({});
      return;
    }
    const defaults = defaultOptunaSelection(activeParams);
    setBounds(defaultOptunaBounds(activeParams, defaults));
    setMessage(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [baseId, activeKey, baseStrategy]);

  const checkedCount = useMemo(
    () => activeParams.filter((parameter) => bounds[parameter.path]?.enabled).length,
    [activeParams, bounds],
  );

  useEffect(() => {
    if (study === null && studies.data && studies.data.studies.length > 0) {
      setStudy(studies.data.studies[0] ?? null);
    }
  }, [studies.data, study]);

  // GET /api/optimize returns summaries (no trials); load the full study on demand.
  useEffect(() => {
    if (!study || study.trials.length > 0) return;
    let cancelled = false;
    void getOptimization(study.study_id).then((result) => {
      if (!cancelled && result.data.trials.length > 0) setStudy(result.data);
    });
    return () => {
      cancelled = true;
    };
  }, [study]);

  const run = async () => {
    if (!baseId) return;
    setRunning(true);
    setMessage(null);
    const request: OptimizationRequest = {
      base_strategy_id: baseId,
      objective_metric: objective,
      n_trials: Math.max(2, Math.min(200, Math.round(nTrials))),
      walk_forward: true,
      wf_splits: Math.max(2, Math.min(10, Math.round(wfSplits))),
      purge_bars: 5,
      embargo_bars: 5,
      drawdown_penalty_threshold: drawdownPenalty,
      search_space: buildSearchSpace(activeParams, bounds),
      seed: 7,
    };
    const result = await runOptimization(request);
    setRunning(false);
    setStudy(result.data);
    setMessage({
      text:
        result.source === "live"
          ? `Study ${result.data.study_id} completed · best ${objective} ${formatObjectiveValue(result.data.best_value)}`
          : `UI fallback study ${result.data.study_id} · best ${objective} ${formatObjectiveValue(result.data.best_value)}`,
      tone: result.source === "live" ? "gain" : "warn",
    });
    studies.reload();
    onNotify(`Optimization complete · ${result.data.n_trials} trials`, result.source === "live" ? "gain" : "warn");
  };

  const apply = async () => {
    if (!study) return;
    setApplying(true);
    const result = await applyOptimization(study.study_id);
    setApplying(false);
    const verification = result.data.verification_backtest;
    const verificationNote =
      verification && verification.status !== "completed"
        ? ` · verification ${verification.status}${verification.message ? `: ${verification.message}` : ""}`
        : "";
    setMessage({
      text: (result.data.message || `Applied best parameters as ${result.data.strategy_id}.`) + verificationNote,
      tone: result.data.applied ? (verification && verification.status === "skipped" ? "warn" : "gain") : "warn",
    });
    onNotify(`Applied ${study.study_id} → ${result.data.strategy_id}`, result.source === "live" ? "gain" : "warn");
    if (result.data.strategy) onApplied(result.data.strategy);
    const appliedId = result.data.strategy_id;
    if (appliedId) {
      studies.reload();
    }
  };

  const sortedTrials = useMemo(() => {
    if (!study) return [];
    const value = (trial: OptimizationTrial): number => {
      switch (sort.key) {
        case "trial":
          return trial.trial_number;
        case "sharpe":
          return trial.oos_sharpe;
        case "cagr":
          return trial.cagr;
        case "dd":
          return trial.max_dd;
        default:
          return trial.value;
      }
    };
    return [...study.trials].sort((a, b) =>
      sort.dir === "asc" ? value(a) - value(b) : value(b) - value(a),
    );
  }, [study, sort]);

  const toggleSort = (key: SortKey) => {
    setSort((current) =>
      current.key === key ? { key, dir: current.dir === "asc" ? "desc" : "asc" } : { key, dir: "desc" },
    );
  };

  const history = study
    ? (() => {
        let best = Number.NEGATIVE_INFINITY;
        const bestValues: number[] = [];
        const trialValues: number[] = [];
        for (const trial of [...study.trials].sort((a, b) => a.trial_number - b.trial_number)) {
          best = Math.max(best, trial.value);
          trialValues.push(trial.value);
          bestValues.push(best);
        }
        return { trialValues, bestValues };
      })()
    : null;

  const importances = (study?.param_importances ?? []).map((entry) => ({
    label: formatParamPath(entry.param_path),
    value: entry.importance,
    percent: importancePercent(entry.importance, (study?.param_importances ?? []).map((item) => item.importance)),
  }));

  return (
    <div className="grid min-h-full grid-cols-12 gap-3">
      {/* ------------------------------------------------------------ config */}
      <div className="col-span-12 flex min-h-0 flex-col gap-3 xl:col-span-5">
        <Panel title="Study configuration" className="shrink-0">
          <div className="grid grid-cols-2 gap-x-3 gap-y-2.5">
            <Field label="Base strategy" htmlFor="opt-base" className="col-span-2">
              <Select
                id="opt-base"
                value={baseId}
                onChange={setBaseId}
                options={[
                  { value: "", label: "— select base strategy —" },
                  ...strategies.map((strategy) => ({ value: strategy.id, label: strategy.id })),
                ]}
              />
            </Field>
            <Field label="Objective" htmlFor="opt-objective">
              <Select
                id="opt-objective"
                value={objective}
                onChange={(value) => setObjective(value as OptimizationObjective)}
                options={OBJECTIVES}
              />
            </Field>
            <Field label="Trials" htmlFor="opt-trials">
              <TextInput
                id="opt-trials"
                type="number"
                min={2}
                value={nTrials}
                onChange={(value) => setNTrials(Number(value))}
              />
            </Field>
            <div className="col-span-2 flex items-end justify-between gap-3">
              <Checkbox
                id="opt-wf"
                checked
                onChange={() => undefined}
                disabled
                label={
                  <span>
                    <span className="text-[12px] font-medium text-zinc-300">Walk-forward validation</span>
                    <span className="block text-[10.5px] text-zinc-600">
                      Locked ON — every trial is scored only on purged/embargoed OOS folds.
                    </span>
                  </span>
                }
              />
              <Field label="Folds" htmlFor="opt-folds" className="w-[58px] shrink-0">
                <TextInput
                  id="opt-folds"
                  type="number"
                  min={2}
                  value={wfSplits}
                  onChange={(value) => setWfSplits(Number(value))}
                  className="text-center"
                />
              </Field>
              <Field label="Max DD penalty" htmlFor="opt-ddpen" className="w-[74px] shrink-0">
                <TextInput
                  id="opt-ddpen"
                  type="number"
                  min={0}
                  step={0.01}
                  value={drawdownPenalty}
                  onChange={(value) => setDrawdownPenalty(Number(value))}
                  className="text-center"
                />
              </Field>
            </div>
            <div className="col-span-2 rounded-md border border-warn/30 bg-warn-soft/40 px-2.5 py-2 text-[10.5px] leading-4 text-amber-300">
              Data-snooping guard: in-sample metrics are never optimised. Trials with OOS max drawdown above{" "}
              {(drawdownPenalty * 100).toFixed(0)}% receive a penalty.
            </div>
          </div>
        </Panel>

        <Panel
          title="Tunable bounds"
          className="max-h-[356px] shrink-0"
          actions={
            <span className="flex items-center gap-2">
              {checkedCount > 8 ? (
                <Badge tone="warn" title="Recommended: tune 3–6 parameters per study to prevent overfitting">
                  overfitting risk
                </Badge>
              ) : null}
              <span className="text-[10.5px] text-zinc-600" data-testid="optuna-selected-count">
                {checkedCount} checked
              </span>
            </span>
          }
        >
          {checkedCount > 8 ? (
            <div
              className="mb-2 shrink-0 rounded border border-warn/30 bg-warn-soft/40 px-2 py-1.5 text-[10px] leading-4 text-amber-300"
              data-testid="optuna-overfit-warning"
            >
              Recommended: tune 3–6 parameters per study to prevent overfitting.
            </div>
          ) : null}
          <div className="scroll-area -mx-1 min-h-0 flex-1 overflow-y-auto px-1">
            {tunables.length === 0 ? (
              <EmptyState
                title="No active tunable parameters"
                description={`No numeric Optuna parameters are active for ${baseId || "this strategy"}. Inactive selector/overlay parameters and structural hygiene params are excluded.`}
              />
            ) : (
              <div className="space-y-1">
                {tunables.map((parameter) => {
                  const entry = bounds[parameter.path] ?? {
                    enabled: false,
                    low: typeof parameter.default === "number" ? parameter.default : 0,
                    high: typeof parameter.default === "number" ? parameter.default : 1,
                  };
                  return (
                    <TunableRow
                      key={parameter.path}
                      parameter={parameter}
                      entry={entry}
                      onChange={(next) => setBounds((current) => ({ ...current, [parameter.path]: next }))}
                    />
                  );
                })}
              </div>
            )}
          </div>
          {structuralParams.length > 0 ? (
            <details className="mt-1 shrink-0 border-t border-hairline pt-1.5" data-testid="optuna-structural-params">
              <summary className="cursor-pointer select-none text-[10px] font-medium text-zinc-500 hover:text-zinc-300" title="Structural / data-hygiene parameters are read-only and never tuned">
                Structural parameters excluded ({structuralParams.length}) — not optimized
              </summary>
              <div className="mt-1 max-h-24 space-y-0.5 overflow-y-auto">
                {structuralParams.map((parameter) => (
                  <div
                    key={parameter.path}
                    className="flex items-center gap-1.5 rounded px-1 py-0.5 text-[10px] text-zinc-600"
                    data-testid="optuna-structural-row"
                    title="Structural / data-hygiene parameter — excluded from Optuna to prevent data-snooping"
                  >
                    <span className="h-1 w-1 rounded-full bg-zinc-600" />
                    <span className="truncate">{parameter.label}</span>
                    <span className="ml-auto shrink-0 font-mono text-[9px] text-zinc-700">
                      {parameter.path.split(".").pop()}
                    </span>
                  </div>
                ))}
              </div>
            </details>
          ) : null}
          <div className="mt-2 flex items-center gap-2 border-t border-hairline pt-2">
            <Button variant="primary" onClick={() => void run()} disabled={running || !baseId}>
              {running ? "Optimizing…" : "Run optimization"}
            </Button>
            <span className="font-mono text-[10.5px] text-zinc-600">
              {study?.study_id ?? "no active study"}
            </span>
            {message ? (
              <span
                className={cn(
                  "ml-auto min-w-0 truncate text-[10.5px]",
                  message.tone === "gain" ? "text-emerald-400" : message.tone === "warn" ? "text-amber-400" : "text-zinc-500",
                )}
                title={message.text}
              >
                {message.text}
              </span>
            ) : null}
          </div>
        </Panel>

        <Panel
          title="Persisted studies"
          className="max-h-[210px] shrink-0"
          actions={
            <Button variant="ghost" size="xs" onClick={() => studies.reload()}>
              Refresh
            </Button>
          }
        >
          <div className="scroll-area -mx-1 min-h-0 flex-1 overflow-y-auto px-1">
            {(studies.data?.studies ?? []).length === 0 ? (
              <p className="py-2 text-[11px] text-zinc-600">No persisted optimization studies yet.</p>
            ) : (
              <div className="space-y-1">
                {(studies.data?.studies ?? []).map((entry) => (
                  <button
                    key={entry.study_id}
                    type="button"
                    onClick={() => setStudy(entry)}
                    className={cn(
                      "focus-ring flex w-full items-center justify-between gap-2 rounded border px-2 py-1.5 text-left text-[10.5px] transition-colors",
                      study?.study_id === entry.study_id
                        ? "border-accent/40 bg-accent-soft/40"
                        : "border-transparent hover:bg-surface",
                    )}
                  >
                    <span className="min-w-0">
                      <span className="block truncate font-mono text-zinc-300">{entry.study_id}</span>
                      <span className="block truncate text-zinc-600">
                        {entry.base_strategy_id} · {entry.objective_metric} · {entry.n_trials} trials
                      </span>
                    </span>
                    <span className="shrink-0 font-mono tabular-nums text-emerald-400">
                      {formatObjectiveValue(entry.best_value)}
                    </span>
                  </button>
                ))}
              </div>
            )}
          </div>
        </Panel>
      </div>

      {/* ----------------------------------------------------------- results */}
      <div className="col-span-12 flex min-h-0 flex-col gap-3 xl:col-span-7">
        <Panel
          title="Objective history"
          className="shrink-0"
          actions={
            <div className="flex items-center gap-2">
              {study ? (
                <span className="flex items-center gap-1.5">
                  {study.provenance ? (
                    <Badge tone={study.provenance === "engine" ? "gain" : "warn"} title={study.message ?? "Optimization provenance"}>
                      {study.backend ?? study.provenance}
                    </Badge>
                  ) : null}
                  <Badge tone={study.walk_forward ? "accent" : "warn"}>
                    WF · {study.fold_count ?? study.wf_splits} folds
                  </Badge>
                </span>
              ) : null}
              <Button
                variant="primary"
                size="xs"
                onClick={() => void apply()}
                disabled={!study || applying || study.status !== "completed"}
                title="Persist the best parameters as a new strategy config"
              >
                {applying ? "Applying…" : "Apply Best Parameters"}
              </Button>
            </div>
          }
        >
          {study && history ? (
            <>
              <div className="mb-1 flex items-center justify-between gap-2">
                <div className="flex items-center gap-3 text-[10.5px] text-zinc-500">
                  <span className="flex items-center gap-1.5">
                    <span className="h-1.5 w-1.5 rounded-full bg-accent" />
                    trial {formatObjectiveMetric(study.objective_metric)}
                  </span>
                  <span className="flex items-center gap-1.5">
                    <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
                    best so far
                  </span>
                </div>
                <span className="font-mono text-[10.5px] text-zinc-600">
                  {study.n_trials} trials · {formatInteger(study.duration_ms)} ms
                </span>
              </div>
              <OrdinalLineChart
                series={[
                  { label: "trial", color: "#3B82F6", values: history.trialValues },
                  { label: "best so far", color: "#10B981", values: history.bestValues },
                ]}
                height={208}
                formatValue={(value) => formatObjectiveValue(value)}
                xLabels={["1", String(Math.round((study.n_trials + 1) / 2)), String(study.n_trials)]}
                ariaLabel="Optimization trial objective and best-so-far"
              />
              <div className="mt-2 grid grid-cols-5 gap-3 border-t border-hairline pt-2">
                <MiniStat label="Best value" value={<span className="text-emerald-400">{formatObjectiveValue(study.best_value)}</span>} />
                <MiniStat label="Best trial" value={study.best_trial_number ?? "—"} />
                <MiniStat label="Objective" value={formatObjectiveMetric(study.objective_metric)} />
                <MiniStat label="OOS folds" value={study.wf_splits} />
                <MiniStat label="Base" value={study.base_strategy_id} />
              </div>
            </>
          ) : (
            <EmptyState
              title="No optimization study selected"
              description="Choose a persisted study or run the optimizer. Every trial is evaluated on walk-forward OOS folds only."
            />
          )}
        </Panel>

        <Panel title="Parameter importance" className="shrink-0" actions={<span className="text-[10.5px] text-zinc-600">share of OOS objective variance</span>}>
          <ImportanceBars rows={importances} maxRows={7} />
        </Panel>

        <Panel
          title="Trials"
          flush
          className="min-h-0 flex-1"
          actions={
            study ? (
              <span className="text-[10.5px] text-zinc-600">
                best #{study.best_trial_number ?? "—"} · {study.trials.length} rows
              </span>
            ) : null
          }
        >
          <div className="min-h-0 flex-1 overflow-auto">
            {study ? (
              <table className="data-table">
                <thead>
                  <tr>
                    <th className="w-16">
                      <SortButton label="Trial" active={sort.key === "trial"} dir={sort.dir} onClick={() => toggleSort("trial")} />
                    </th>
                    <th className="w-20 text-right">
                      <SortButton label="Objective" active={sort.key === "value"} dir={sort.dir} onClick={() => toggleSort("value")} align="right" />
                    </th>
                    <th className="w-20 text-right">
                      <SortButton label="Sharpe" active={sort.key === "sharpe"} dir={sort.dir} onClick={() => toggleSort("sharpe")} align="right" />
                    </th>
                    <th className="w-20 text-right">
                      <SortButton label="CAGR" active={sort.key === "cagr"} dir={sort.dir} onClick={() => toggleSort("cagr")} align="right" />
                    </th>
                    <th className="w-20 text-right">
                      <SortButton label="Max DD" active={sort.key === "dd"} dir={sort.dir} onClick={() => toggleSort("dd")} align="right" />
                    </th>
                    <th className="w-20 text-right">Calmar</th>
                    <th className="w-20 text-right">Rachev</th>
                    <th className="w-24 text-right">Duration</th>
                    <th>Parameters</th>
                  </tr>
                </thead>
                <tbody>
                  {sortedTrials.map((trial) => (
                    <tr key={trial.trial_number} className={trial.best ? "bg-gain-soft/25" : undefined}>
                      <td className="font-mono text-[11px] text-zinc-300">
                        {trial.best ? <Badge tone="gain">best</Badge> : `#${trial.trial_number}`}
                      </td>
                      <td className="num text-right text-zinc-200">{formatObjectiveValue(trial.value)}</td>
                      <td className="num text-right text-zinc-300">{formatNumber(trial.oos_sharpe, 2)}</td>
                      <td className={cn("num text-right", trial.cagr >= 0 ? "text-gain/90" : "text-loss/90")}>{formatPct(trial.cagr)}</td>
                      <td className="num text-right text-loss/90">{formatPct(trial.max_dd, 1)}</td>
                      <td className="num text-right text-zinc-400">{formatNumber(trial.calmar, 2)}</td>
                      <td className="num text-right text-zinc-400">{formatNumber(trial.rachev, 2)}</td>
                      <td className="num text-right text-zinc-500">{formatInteger(trial.duration_ms)} ms</td>
                      <td className="max-w-[260px] truncate font-mono text-[10px] text-zinc-600" title={JSON.stringify(trial.params)}>
                        {Object.entries(trial.params)
                          .slice(0, 3)
                          .map(([key, value]) => `${shortParam(key)}=${String(value)}`)
                          .join(" · ")}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <EmptyState title="No trials" description="Run an optimization study to populate the trial ledger." />
            )}
          </div>
        </Panel>
      </div>
    </div>
  );
}

function formatObjectiveValue(value: number): string {
  return Number.isFinite(value) ? value.toFixed(3) : "—";
}

function shortParam(path: string): string {
  const parts = path.split(".");
  return parts[parts.length - 1] ?? path;
}

function TunableRow({
  parameter,
  entry,
  onChange,
}: {
  parameter: StudioParameter;
  entry: { enabled: boolean; low: number; high: number };
  onChange: (entry: { enabled: boolean; low: number; high: number }) => void;
}) {
  return (
    <div
      className="flex items-center gap-2 rounded border border-transparent px-1 py-1 text-[10.5px] hover:border-hairline"
      data-testid="optuna-tunable-row"
      data-optuna-path={parameter.path}
    >
      <input
        type="checkbox"
        checked={entry.enabled}
        onChange={(event) => onChange({ ...entry, enabled: event.target.checked })}
        aria-label={`Tune ${parameter.label}`}
        data-testid={`optuna-check-${parameter.path.split(".").pop() ?? parameter.path}`}
        className="h-3 w-3 rounded border border-zinc-600 bg-canvas accent-blue-500"
      />
      <span className="min-w-0 flex-1 truncate text-zinc-400" title={parameter.path}>
        {parameter.label}
      </span>
      <span className="w-[64px] shrink-0">
        <TextInput
          type="number"
          value={entry.low}
          onChange={(value) => onChange({ ...entry, low: Number(value) })}
          className="h-6 px-1 text-right text-[10px]"
          disabled={!entry.enabled}
        />
      </span>
      <span className="text-zinc-600">–</span>
      <span className="w-[64px] shrink-0">
        <TextInput
          type="number"
          value={entry.high}
          onChange={(value) => onChange({ ...entry, high: Number(value) })}
          className="h-6 px-1 text-right text-[10px]"
          disabled={!entry.enabled}
        />
      </span>
    </div>
  );
}

function SortButton({
  label,
  active,
  dir,
  onClick,
  align = "left",
}: {
  label: string;
  active: boolean;
  dir: "asc" | "desc";
  onClick: () => void;
  align?: "left" | "right";
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "focus-ring inline-flex items-center gap-1 text-[11px] font-semibold uppercase tracking-[0.04em] transition-colors",
        active ? "text-zinc-200" : "text-zinc-500 hover:text-zinc-300",
        align === "right" && "justify-end",
      )}
    >
      {label}
      {active ? <span className="font-mono text-[9px]">{dir === "asc" ? "▲" : "▼"}</span> : null}
    </button>
  );
}
