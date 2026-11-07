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
import { Modal } from "@/components/ui/dialog";
import { TrashIcon } from "@/components/shell/icons";
import {
  applyOptimization,
  deleteOptimization,
  getOptimization,
  listOptimizations,
  runOptimization,
} from "@/lib/api";
import { cn } from "@/lib/cn";
import {
  DATA_MODE_OPTIONS,
  DATA_TIMEFRAME_OPTIONS,
  cachedBarsDetail,
  dataOriginBadge,
} from "@/lib/data-cache";
import { formatInteger, formatNumber, formatPct } from "@/lib/format";
import {
  activeTunableParameters,
  buildSearchSpace,
  defaultOptunaBounds,
  defaultOptunaSelection,
  formatObjectiveMetric,
  formatParamPath,
  ID_PATTERN,
  importancePercent,
  isLegacyOptimizationStudy,
  structuralParametersForStrategy,
} from "@/lib/studio";
import type {
  CachedBarsMeta,
  DataTimeframe,
  MarketDataMode,
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
  onStrategiesChanged,
  onNotify,
  timeframe,
  onTimeframeChange,
}: {
  catalog: StudioCatalogResponse;
  strategies: StrategyConfig[];
  onStrategiesChanged: () => void;
  onNotify: (message: string, tone: Tone) => void;
  timeframe: DataTimeframe;
  onTimeframeChange: (timeframe: DataTimeframe) => void;
}) {
  const studies = useAsyncData(listOptimizations);
  const reloadStudies = studies.reload;
  const [baseId, setBaseId] = useState("");
  const [objective, setObjective] = useState<OptimizationObjective>("sharpe");
  const [nTrials, setNTrials] = useState(25);
  const [dataMode, setDataMode] = useState<MarketDataMode>("auto");
  const [wfSplits, setWfSplits] = useState(5);
  const [drawdownPenalty, setDrawdownPenalty] = useState(0.2);
  const [bounds, setBounds] = useState<Record<string, { enabled: boolean; low: number; high: number }>>({});
  const [study, setStudy] = useState<OptimizationStudy | null>(null);
  const [running, setRunning] = useState(false);
  const [applying, setApplying] = useState(false);
  const [message, setMessage] = useState<{ text: string; tone: Tone } | null>(null);
  const [archivedIds, setArchivedIds] = useState<Set<string>>(() => new Set());
  const [applyOpen, setApplyOpen] = useState(false);
  const [applyTargetId, setApplyTargetId] = useState("");
  const [applyError, setApplyError] = useState<string | null>(null);
  const [applyBanner, setApplyBanner] = useState<{
    id: string;
    strategyPath: string | null;
    backtestRunId: string | null;
  } | null>(null);
  const [applyToast, setApplyToast] = useState<{
    strategyId: string;
    backtestRunId: string | null;
  } | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<OptimizationStudy | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);
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

  // v1.5.2: never auto-select a legacy study. The API quarantines legacy
  // artifacts (task-44); this client-side guard drops any stale legacy id still
  // in local state and prefers the newest clean study from the list.
  const cleanStudies = useMemo(
    () =>
      (studies.data?.studies ?? []).filter(
        (entry) =>
          !archivedIds.has(entry.study_id) &&
          !isLegacyOptimizationStudy(entry, strategies, catalog),
      ),
    [studies.data, strategies, catalog, archivedIds],
  );
  const legacyStudyCount = (studies.data?.studies.length ?? 0) - cleanStudies.length;

  useEffect(() => {
    if (!studies.data) return;
    if (study && isLegacyOptimizationStudy(study, strategies, catalog)) {
      setStudy(null);
      return;
    }
    if (study === null && cleanStudies.length > 0) {
      setStudy(cleanStudies[0] ?? null);
    }
  }, [studies.data, study, cleanStudies, strategies, catalog]);

  // GET /api/optimize returns summaries (no trials); load the full study on demand.
  useEffect(() => {
    if (!study || study.trials.length > 0) return;
    if (isLegacyOptimizationStudy(study, strategies, catalog)) return;
    let cancelled = false;
    void getOptimization(study.study_id).then((result) => {
      if (cancelled) return;
      if (!result.data) {
        // API quarantined/archived a legacy study (404) — refresh the clean list.
        setMessage({
          text: "Legacy optimization run archived — pick a current study.",
          tone: "warn",
        });
        setArchivedIds((current) => {
          const next = new Set(current);
          next.add(study.study_id);
          return next;
        });
        setStudy(null);
        reloadStudies();
        return;
      }
      if (result.data.trials.length > 0) setStudy(result.data);
    });
    return () => {
      cancelled = true;
    };
  }, [study, strategies, catalog, reloadStudies]);

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
      data_mode: dataMode,
      timeframe,
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

  useEffect(() => {
    if (!applyToast) return;
    const timer = setTimeout(() => setApplyToast(null), 12_000);
    return () => clearTimeout(timer);
  }, [applyToast]);

  const openApplyDialog = () => {
    if (!study) return;
    const base = baseStrategy?.id ?? study.base_strategy_id ?? "strategy";
    setApplyTargetId(`${base}_opt`);
    setApplyError(null);
    setApplyOpen(true);
  };

  const apply = async () => {
    if (!study) return;
    const targetId = applyTargetId.trim();
    if (!ID_PATTERN.test(targetId)) {
      setApplyError("Strategy ID must match ^[a-z][a-z0-9_]{0,63}$ (lowercase slug, no hyphens).");
      return;
    }
    setApplying(true);
    setApplyError(null);
    const result = await applyOptimization(study.study_id, {
      newStrategyId: targetId,
      verificationBacktest: true,
    });
    setApplying(false);

    if (result.source !== "live") {
      const status = result.error ?? "";
      if (/\b409\b/.test(status)) {
        setApplyError(`Strategy ID "${targetId}" already exists — choose a different ID.`);
      } else if (/\b422\b/.test(status)) {
        setApplyError(
          "The backend rejected the applied strategy (422 invalid configuration). Adjust the study or target ID and retry.",
        );
      } else if (/\b404\b/.test(status)) {
        setApplyError("Legacy optimization run archived — pick a current study.");
      } else {
        setApplyError(
          result.data.message ||
            "Apply requires the FastAPI backend — offline preview cannot persist the strategy or run the verification backtest.",
        );
      }
      return;
    }
    if (!result.data.applied) {
      setApplyError(
        result.data.message ||
          "The backend did not persist the strategy — no verification backtest was created.",
      );
      return;
    }

    const savedId =
      result.data.saved_strategy_id ?? result.data.strategy_id ?? targetId;
    const backtestRunId =
      result.data.backtest_run_id ??
      result.data.verification_backtest?.run?.id ??
      null;
    setApplyBanner({
      id: savedId,
      strategyPath: result.data.strategy_path ?? null,
      backtestRunId,
    });
    setApplyToast({ strategyId: savedId, backtestRunId });
    setApplyOpen(false);
    setMessage({
      text: `Saved strategy "${savedId}"${
        backtestRunId ? ` and created verification backtest ${backtestRunId}` : ""
      }.`,
      tone: "gain",
    });
    onNotify(
      `Applied ${study.study_id} → ${savedId}${backtestRunId ? ` · ${backtestRunId}` : ""}`,
      "gain",
    );
    // Refresh both the strategy catalog (Studio Load-existing / Backtests
    // navigation refreshes on mount) and the optimization study list.
    onStrategiesChanged();
    studies.reload();
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

  const hasActiveTrial = Boolean(study?.trials.some((trial) => !trial.zero_activity));
  const canApplyStudy =
    Boolean(study) &&
    study?.status === "completed" &&
    study?.best_trial_number !== null &&
    Object.keys(study?.best_params ?? {}).length > 0;

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
            <Field label="Data mode" htmlFor="opt-data-mode">
              <Select
                id="opt-data-mode"
                value={dataMode}
                onChange={(value) => setDataMode(value as MarketDataMode)}
                options={DATA_MODE_OPTIONS.map((option) => ({ value: option.value, label: option.label }))}
              />
            </Field>
            <Field label="Bar timeframe" htmlFor="opt-timeframe">
              <Select
                id="opt-timeframe"
                value={timeframe}
                onChange={(value) => onTimeframeChange(value as DataTimeframe)}
                options={DATA_TIMEFRAME_OPTIONS.map((option) => ({ value: option.value, label: option.label }))}
              />
            </Field>
            <div className="col-span-2 rounded border border-hairline bg-canvas/50 px-2 py-1 text-[10px] leading-4 text-zinc-500">
              Auto uses real cached Parquet bars when the strategy roots have enough history; otherwise the study falls back to the
              synthetic fixture. Cached-real fails with an actionable error if the cache is insufficient.
            </div>
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
            {cleanStudies.length === 0 ? (
              <p className="py-2 text-[11px] text-zinc-600">
                No persisted optimization studies yet.
              </p>
            ) : (
              <div className="space-y-1">
                {cleanStudies.map((entry) => (
                  <div
                    key={entry.study_id}
                    data-testid="optimization-study-row"
                    data-study-id={entry.study_id}
                    data-selected={study?.study_id === entry.study_id ? "true" : "false"}
                    className={cn(
                      "flex w-full items-center gap-1 rounded border pr-1 transition-colors",
                      study?.study_id === entry.study_id
                        ? "border-accent/40 bg-accent-soft/40"
                        : "border-transparent hover:bg-surface",
                    )}
                  >
                    <button
                      type="button"
                      onClick={() => setStudy(entry)}
                      className="focus-ring flex min-w-0 flex-1 items-center justify-between gap-2 px-2 py-1.5 text-left text-[10.5px]"
                    >
                      <span className="min-w-0">
                        <span className="flex min-w-0 items-center gap-1.5">
                          <span className="truncate font-mono text-zinc-300">{entry.study_id}</span>
                          <StudyDataBadge study={entry} />
                        </span>
                        <span className="block truncate text-zinc-600">
                          {entry.base_strategy_id} · {entry.objective_metric} · {entry.n_trials} trials
                        </span>
                      </span>
                      <span className="shrink-0 font-mono tabular-nums text-emerald-400">
                        {formatObjectiveValue(entry.best_value)}
                      </span>
                    </button>
                    <button
                      type="button"
                      aria-label={`Delete study ${entry.study_id}`}
                      title="Delete this optimization study artifact"
                      data-testid={`delete-study-${entry.study_id}`}
                      onClick={() => {
                        setDeleteError(null);
                        setDeleteTarget(entry);
                      }}
                      className="focus-ring grid h-6 w-6 shrink-0 place-items-center rounded text-zinc-600 hover:bg-loss-soft hover:text-rose-300"
                    >
                      <TrashIcon className="h-3.5 w-3.5" />
                    </button>
                  </div>
                ))}
              </div>
            )}
            {legacyStudyCount > 0 ? (
              <p className="mt-1 px-1 text-[9.5px] text-amber-400/80" data-testid="legacy-studies-note">
                {legacyStudyCount} legacy study{legacyStudyCount === 1 ? "" : "s"} hidden (quarantined by API / client guard)
              </p>
            ) : null}
          </div>
        </Panel>
      </div>

      {/* ----------------------------------------------------------- results */}
      <div className="col-span-12 flex min-h-0 flex-col gap-3 xl:col-span-7">
        {applyBanner ? (
          <div
            role="status"
            data-testid="apply-success-banner"
            data-strategy-id={applyBanner.id}
            data-backtest-id={applyBanner.backtestRunId ?? ""}
            className="shrink-0 rounded-md border border-gain/40 bg-gain-soft/40 px-3 py-2"
          >
            <div className="flex items-start justify-between gap-2">
              <p className="text-[12px] leading-5 text-emerald-300">
                Saved strategy <span className="font-mono text-emerald-200">&quot;{applyBanner.id}&quot;</span>
                {applyBanner.backtestRunId ? (
                  <>
                    {" "}&amp; created verification backtest{" "}
                    <span className="font-mono text-emerald-200">{applyBanner.backtestRunId}</span>
                  </>
                ) : (
                  <> &amp; started the verification backtest</>
                )}
                .
              </p>
              <button
                type="button"
                aria-label="Dismiss apply confirmation"
                onClick={() => setApplyBanner(null)}
                className="focus-ring -mr-1 grid h-5 w-5 shrink-0 place-items-center rounded text-emerald-400/70 hover:bg-gain-soft hover:text-emerald-300"
              >
                ×
              </button>
            </div>
            <div className="mt-1 flex flex-wrap items-center gap-3 text-[11px]">
              <a
                href={`/backtests?strategy=${encodeURIComponent(applyBanner.id)}`}
                data-testid="apply-link-backtests"
                className="font-medium text-emerald-300 underline decoration-emerald-500/40 underline-offset-2 hover:text-emerald-200"
              >
                View in Backtests
              </a>
              <a
                href={`/studio?strategy=${encodeURIComponent(applyBanner.id)}`}
                data-testid="apply-link-builder"
                className="font-medium text-emerald-300 underline decoration-emerald-500/40 underline-offset-2 hover:text-emerald-200"
              >
                Open in Configuration
              </a>
              {applyBanner.strategyPath ? (
                <span className="truncate font-mono text-[10px] text-emerald-500/70" title={applyBanner.strategyPath}>
                  {applyBanner.strategyPath}
                </span>
              ) : null}
            </div>
          </div>
        ) : null}
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
                  <StudyDataBadge study={study} />
                </span>
              ) : null}
              <Button
                variant="primary"
                size="xs"
                onClick={openApplyDialog}
                disabled={!canApplyStudy || applying}
                title={
                  canApplyStudy
                    ? "Save the best parameters as a strategy and run a verification backtest"
                    : "No active trial / best parameter set to apply"
                }
                testId="open-apply-dialog"
              >
                Apply Best Parameters
              </Button>
            </div>
          }
        >
          {study && history && !hasActiveTrial ? (
            <EmptyState
              title="No active trial — flat study"
              description="Every trial in this study was flat or below the activity thresholds (objective −10, zero activity). There is no best parameter set to apply; widen the walk-forward window or adjust the strategy and run a new study."
            />
          ) : study && history ? (
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
                    <th className="w-20 text-right">Trades</th>
                    <th className="w-24 text-right">Avg exp</th>
                    <th>Parameters</th>
                  </tr>
                </thead>
                <tbody>
                  {sortedTrials.map((trial) => (
                    <tr
                      key={trial.trial_number}
                      className={cn(
                        trial.best ? "bg-gain-soft/25" : undefined,
                        trial.zero_activity && "opacity-60",
                      )}
                    >
                      <td className="font-mono text-[11px] text-zinc-300">
                        {trial.zero_activity ? (
                          <Badge tone="warn">flat</Badge>
                        ) : trial.best ? (
                          <Badge tone="gain">best</Badge>
                        ) : (
                          `#${trial.trial_number}`
                        )}
                      </td>
                      <td className="num text-right text-zinc-200">{formatObjectiveValue(trial.value)}</td>
                      <td className="num text-right text-zinc-300">{formatNumber(trial.oos_sharpe, 2)}</td>
                      <td className={cn("num text-right", trial.cagr >= 0 ? "text-gain/90" : "text-loss/90")}>{formatPct(trial.cagr)}</td>
                      <td className="num text-right text-loss/90">{formatPct(trial.max_dd, 1)}</td>
                      <td className="num text-right text-zinc-400">{formatNumber(trial.calmar, 2)}</td>
                      <td className="num text-right text-zinc-400">{formatNumber(trial.rachev, 2)}</td>
                      <td className="num text-right text-zinc-500">{formatInteger(trial.duration_ms)} ms</td>
                      <td className="num text-right text-zinc-400">
                        {typeof trial.n_trades === "number" ? formatInteger(trial.n_trades) : "—"}
                      </td>
                      <td className="num text-right text-zinc-400">
                        {typeof trial.avg_gross_exposure === "number"
                          ? `${(trial.avg_gross_exposure * 100).toFixed(1)}%`
                          : "—"}
                      </td>
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
      <Modal
        open={applyOpen}
        title="Apply best parameters"
        description={
          <span>
            Saves the best parameters from <span className="font-mono text-zinc-300">{study?.study_id ?? "study"}</span> as a
            new strategy and runs a persisted walk-forward verification backtest.
          </span>
        }
        onClose={() => {
          if (applying) return;
          setApplyOpen(false);
          setApplyError(null);
        }}
        busy={applying}
        footer={
          <>
            <Button
              variant="ghost"
              onClick={() => {
                setApplyOpen(false);
                setApplyError(null);
              }}
              disabled={applying}
            >
              Cancel
            </Button>
            <Button
              variant="primary"
              onClick={() => void apply()}
              disabled={applying || !ID_PATTERN.test(applyTargetId.trim())}
              testId="confirm-apply"
            >
              {applying ? (
                <span className="flex items-center gap-1.5">
                  <span
                    aria-hidden
                    className="h-3 w-3 animate-spin rounded-full border-2 border-white/30 border-t-white"
                  />
                  Saving &amp; Running Backtest...
                </span>
              ) : (
                "Save & Run Backtest"
              )}
            </Button>
          </>
        }
      >
        <div className="space-y-2">
          <Field
            label="Target strategy ID"
            htmlFor="apply-strategy-id"
            hint="A new YAML strategy is created from the study's best parameters."
          >
            <TextInput
              id="apply-strategy-id"
              value={applyTargetId}
              onChange={(value) => setApplyTargetId(value.toLowerCase().replace(/\s+/g, "_"))}
              disabled={applying}
            />
          </Field>
          {applyError ? (
            <div
              role="alert"
              data-testid="apply-error"
              className="rounded-md border border-loss/40 bg-loss-soft/40 px-2.5 py-2 text-[11px] leading-4 text-rose-300"
            >
              {applyError}
            </div>
          ) : null}
          <p className="text-[10.5px] leading-4 text-zinc-600">
            The backend writes the strategy visible to the Studio and Backtests, then auto-runs a verification
            walk-forward backtest. You will get deep links to both.
          </p>
        </div>
      </Modal>
      <Modal
        open={deleteTarget !== null}
        title="Delete optimization study"
        description={
          <span>
            Permanently remove <span className="font-mono text-zinc-300">{deleteTarget?.study_id}</span> and its
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
              testId="confirm-delete-study"
              disabled={deleting}
              onClick={() => {
                if (!deleteTarget) return;
                setDeleting(true);
                setDeleteError(null);
                void deleteOptimization(deleteTarget.study_id).then((result) => {
                  setDeleting(false);
                  if (result.source === "live" && result.data.ok) {
                    const removed = deleteTarget.study_id;
                    setDeleteTarget(null);
                    if (study?.study_id === removed) setStudy(null);
                    studies.reload();
                    setMessage({ text: `Deleted study ${removed}.`, tone: "gain" });
                    onNotify(`Deleted study ${removed}`, "gain");
                  } else {
                    setDeleteError(
                      result.data.message ||
                        `Delete failed${result.error ? `: ${result.error}` : "."}`,
                    );
                  }
                });
              }}
            >
              {deleting ? "Deleting…" : "Delete study"}
            </Button>
          </>
        }
      >
        {deleteError ? (
          <div
            role="alert"
            data-testid="delete-study-error"
            className="rounded-md border border-loss/40 bg-loss-soft/40 px-2.5 py-2 text-[11px] leading-4 text-rose-300"
          >
            {deleteError}
          </div>
        ) : (
          <p className="text-[10.5px] leading-4 text-zinc-600">
            The study disappears from the persisted list immediately. Legacy quarantine files are untouched.
          </p>
        )}
      </Modal>

      {applyToast ? (
        <div
          data-testid="apply-toast"
          data-strategy-id={applyToast.strategyId}
          data-run-id={applyToast.backtestRunId ?? ""}
          role="status"
          className="fixed bottom-12 right-4 z-50 w-[320px] rounded-md border border-gain/40 bg-panel px-3 py-2.5 shadow-popover"
        >
          <div className="flex items-start justify-between gap-2">
            <div>
              <p className="text-[11.5px] font-medium text-emerald-300">Verification backtest ready</p>
              <p className="mt-0.5 truncate font-mono text-[10px] text-zinc-500">
                {applyToast.backtestRunId ?? "backtest pending"}
              </p>
            </div>
            <button
              type="button"
              aria-label="Dismiss toast"
              onClick={() => setApplyToast(null)}
              className="focus-ring -mr-1 grid h-5 w-5 place-items-center rounded text-zinc-500 hover:bg-surface hover:text-zinc-300"
            >
              ×
            </button>
          </div>
          <div className="mt-1.5 flex items-center gap-3 text-[10.5px]">
            {applyToast.backtestRunId ? (
              <a
                href={`/backtests?run=${encodeURIComponent(applyToast.backtestRunId)}`}
                data-testid="toast-view-tearsheet"
                className="font-medium text-emerald-300 underline decoration-emerald-500/40 underline-offset-2 hover:text-emerald-200"
              >
                View Tear Sheet
              </a>
            ) : null}
            <a
              href={`/backtests?strategy=${encodeURIComponent(applyToast.strategyId)}`}
              className="text-zinc-400 underline decoration-zinc-600 underline-offset-2 hover:text-zinc-200"
            >
              Backtests
            </a>
          </div>
        </div>
      ) : null}
    </div>
  );
}

function StudyDataBadge({
  study,
}: {
  study: {
    data_origin?: string | null;
    data_mode?: string | null;
    timeframe?: string | null;
    cached_bars?: number | CachedBarsMeta | null;
  };
}) {
  const meta = dataOriginBadge(study.data_origin);
  const detail = cachedBarsDetail(study.cached_bars);
  return (
    <span
      data-testid={meta.real ? "study-data-origin-real" : "study-data-origin-synthetic"}
      data-origin={meta.real ? "real_parquet" : "synthetic_fixture"}
      title={[meta.title, detail, study.data_mode ? `mode ${study.data_mode}` : null].filter(Boolean).join(" · ")}
      className="inline-flex min-w-0 items-center gap-1"
    >
      <Badge tone={meta.tone}>{meta.label}</Badge>
      {detail ? <span className="truncate font-mono text-[8.5px] text-zinc-600">{detail}</span> : null}
      {study.timeframe ? <span className="font-mono text-[8.5px] text-zinc-600">TF {study.timeframe}</span> : null}
    </span>
  );
}

function formatObjectiveValue(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(3) : "—";
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
