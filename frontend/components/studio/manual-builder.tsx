"use client";

import { useRouter } from "next/navigation";
import { useCallback, useEffect, useMemo, useState } from "react";

import {
  Badge,
  Button,
  Field,
  Select,
  TextInput,
  type Tone,
} from "@/components/ui/primitives";
import { Modal } from "@/components/ui/dialog";
import { ChevronDownIcon, ChevronUpIcon, DownloadIcon, TrashIcon } from "@/components/shell/icons";
import {
  cloneStrategy,
  deleteStrategy,
  downloadStrategyYaml,
  getInstruments,
  listDrlModels,
  renameStrategy,
  saveStudioStrategy,
} from "@/lib/api";
import { isBuiltinStrategy } from "@/lib/artifacts";
import { DATA_TIMEFRAME_OPTIONS } from "@/lib/data-cache";
import { cn } from "@/lib/cn";
import {
  SECTOR_TABS,
  allocatorChoices,
  componentDefaults,
  defaultUniverseRoots,
  draftFromPreset,
  draftFromStrategy,
  draftToYaml,
  buildEncodingMap,
  getDraftValue,
  ID_PATTERN,
  isAdvancedParameter,
  numericBadge,
  orderUniverseRoots,
  overlayDefaults,
  parameterLabel,
  rootsOf,
  selectorChoices,
  setDraftValue,
  stageMeta,
  stageSummary,
  syncQuantCaps,
  universeRoots,
  validateDraft,
} from "@/lib/studio";
import type {
  DrlModel,
  InstrumentRoot,
  InstrumentSector,
  DataTimeframe,
  StrategyConfig,
  StudioCatalogResponse,
  StudioDraft,
  StudioParameter,
  StudioParamValue,
} from "@/lib/types";
import { useAsyncData } from "@/lib/use-async-data";

import { RangeField, ToggleRow } from "./controls";

const STAGE_STEPS = [
  { key: "universe", label: "1. Universe S_t", title: "Universe selection", code: "S_t" },
  { key: "allocator", label: "2. Allocator A_t", title: "Portfolio allocator", code: "A_t" },
  { key: "timing", label: "3. Timing & Signals T_t", title: "Timing & signals", code: "T_t" },
  { key: "risk", label: "4. Risk & LLM R_t", title: "Risk & LLM", code: "R_t" },
  { key: "translator", label: "5. Translator N_i,t", title: "Contract translator", code: "N_i,t" },
] as const;

type StageKey = (typeof STAGE_STEPS)[number]["key"];

const TIMING_OVERLAYS = [
  { name: "tsmom", title: "TSMOM", subtitle: "Time-series momentum with volatility sizing." },
  { name: "kama", title: "KAMA", subtitle: "Adaptive moving-average trend confirmation." },
  { name: "carry_filter", title: "Carry / term structure", subtitle: "Tilt towards backwardated roots; penalise expensive contango." },
  { name: "mean_reversion", title: "Mean reversion (RSI + Bollinger)", subtitle: "Fade oversold extremes only when turbulence says the move is noise." },
  { name: "seasonality", title: "Seasonality", subtitle: "Commodity calendar tilts with a root-month bias and out-of-season scale." },
] as const;

const RISK_OVERLAYS = [
  { name: "quant_risk", title: "Quant risk overlay", subtitle: "Hard vol ceiling, concentration caps, drawdown guard and daily kill-switch." },
  { name: "llm_risk", title: "LLM Risk-First overlay", subtitle: "Semantic risk scoring, anti-hallucination neutralisation and circuit breaker." },
] as const;

const TIMING_ORDER = TIMING_OVERLAYS.map((entry) => entry.name);
const RISK_ORDER = RISK_OVERLAYS.map((entry) => entry.name);

function withRoots(draft: StudioDraft, roots: string[]): StudioDraft {
  return {
    ...draft,
    universe: {
      ...draft.universe,
      params: { ...draft.universe.params, roots: roots.join(",") },
    },
  };
}

export function ManualBuilder({
  draft,
  catalog,
  strategies,
  onDraftChange,
  onStrategiesChanged,
  onNotify,
  timeframe,
  onTimeframeChange,
}: {
  draft: StudioDraft;
  catalog: StudioCatalogResponse;
  strategies: StrategyConfig[];
  onDraftChange: (draft: StudioDraft) => void;
  onStrategiesChanged: () => void;
  onNotify: (message: string, tone: Tone) => void;
  timeframe: DataTimeframe;
  onTimeframeChange: (timeframe: DataTimeframe) => void;
}) {
  const router = useRouter();
  const instrumentsSource = useAsyncData(getInstruments);
  const drlModelsSource = useAsyncData(listDrlModels);
  const [activeStage, setActiveStage] = useState<StageKey>("universe");
  const [loadId, setLoadId] = useState("");
  const [presetId, setPresetId] = useState("");
  const [saving, setSaving] = useState<null | "save" | "run">(null);
  const [saveNote, setSaveNote] = useState<{ text: string; tone: Tone } | null>(null);
  const [syncWithAllocator, setSyncWithAllocator] = useState(false);
  const [actionDialog, setActionDialog] = useState<null | "clone" | "rename" | "delete">(null);
  const [actionTargetId, setActionTargetId] = useState("");
  const [actionError, setActionError] = useState<string | null>(null);
  const [actionBusy, setActionBusy] = useState(false);
  const [saveToast, setSaveToast] = useState<{ strategyId: string } | null>(null);

  const sectors = useMemo(
    () => instrumentsSource.data?.sectors ?? catalog.universe?.sectors ?? [],
    [instrumentsSource.data, catalog.universe],
  );
  const rootIndex = useMemo(() => {
    const map = new Map<string, InstrumentRoot>();
    for (const root of universeRoots(sectors)) map.set(root.root, root);
    return map;
  }, [sectors]);

  const validation = useMemo(() => validateDraft(draft), [draft]);
  const encodings = useMemo(() => buildEncodingMap(catalog.parameters), [catalog.parameters]);
  const yaml = useMemo(() => draftToYaml(draft, encodings), [draft, encodings]);
  const editingExisting = strategies.some((strategy) => strategy.id === draft.id);
  const sourceStrategy = useMemo(
    () => strategies.find((strategy) => strategy.id === draft.id) ?? strategies.find((strategy) => strategy.id === loadId) ?? null,
    [strategies, draft.id, loadId],
  );
  const sourceBuiltin = isBuiltinStrategy(sourceStrategy);
  const selectedRoots = useMemo(() => rootsOf(draft), [draft]);
  const drlModels = drlModelsSource.data?.models ?? [];

  const applyDraft = useCallback(
    (next: StudioDraft) => {
      onDraftChange(syncWithAllocator ? syncQuantCaps(next) : next);
    },
    [onDraftChange, syncWithAllocator],
  );

  // Seed the master-universe selection from GET /api/instruments when the
  // preset/catalog did not carry explicit roots.
  useEffect(() => {
    if (sectors.length === 0) return;
    if (rootsOf(draft).length > 0) return;
    const defaults = defaultUniverseRoots(sectors);
    if (defaults.length > 0) onDraftChange(withRoots(draft, defaults));
  }, [sectors, draft, onDraftChange]);

  useEffect(() => {
    if (!saveToast) return;
    const timer = setTimeout(() => setSaveToast(null), 8_000);
    return () => clearTimeout(timer);
  }, [saveToast]);

  const loadExisting = (id: string) => {
    setLoadId(id);
    const strategy = strategies.find((entry) => entry.id === id);
    if (!strategy) return;
    onDraftChange(draftFromStrategy(strategy));
    setSaveNote({ text: `Loaded ${strategy.id} from the strategy catalog.`, tone: "neutral" });
  };

  const applyPreset = (id: string) => {
    setPresetId(id);
    const preset = catalog.presets.find((entry) => entry.id === id);
    if (!preset) return;
    onDraftChange(draftFromPreset(preset));
    setSaveNote({
      text: `${preset.name} — ${preset.hypothesis || preset.description}`,
      tone: "accent",
    });
  };

  const save = async (mode: "save" | "run") => {
    setSaving(mode);
    setSaveNote(null);
    const result = await saveStudioStrategy(draft, { update: editingExisting, encodings });
    setSaving(null);
    if (result.source === "live") {
      setSaveNote({
        text: result.data.persisted
          ? `${editingExisting ? "Updated" : "Created"} ${result.data.strategy.id} in the strategy catalog.`
          : result.data.message,
        tone: result.data.persisted ? "gain" : "warn",
      });
      onNotify(`Strategy ${result.data.strategy.id} saved`, result.data.persisted ? "gain" : "warn");
    } else {
      setSaveNote({
        text: "UI fallback — strategy valid but the API did not persist it. Saved in memory only.",
        tone: "warn",
      });
      onNotify("Strategy YAML validated (UI fallback — not persisted)", "warn");
    }
    onStrategiesChanged();
    if (mode === "run") {
      const id = result.data.strategy?.id ?? draft.id;
      setSaveToast({ strategyId: id });
      // Keep the toast/tear-sheet quick action visible briefly before handing
      // over to the Backtest Lab deep link.
      setTimeout(
        () => router.push(`/backtests?strategy=${encodeURIComponent(id)}&timeframe=${encodeURIComponent(timeframe)}`),
        1_200,
      );
    }
  };

  const openActionDialog = (kind: "clone" | "rename" | "delete") => {
    if (!sourceStrategy) return;
    setActionError(null);
    setActionTargetId(kind === "clone" ? `${sourceStrategy.id}_copy` : sourceStrategy.id);
    setActionDialog(kind);
  };

  const runStrategyAction = async () => {
    if (!sourceStrategy || !actionDialog) return;
    setActionError(null);
    if (actionDialog === "delete") {
      setActionBusy(true);
      const result = await deleteStrategy(sourceStrategy.id);
      setActionBusy(false);
      if (result.source === "live" && result.data.ok) {
        const removed = sourceStrategy.id;
        setActionDialog(null);
        setLoadId("");
        onStrategiesChanged();
        setSaveNote({ text: `Deleted strategy ${removed}.`, tone: "gain" });
        onNotify(`Deleted strategy ${removed}`, "gain");
      } else {
        setActionError(result.data.message || `Delete failed${result.error ? `: ${result.error}` : "."}`);
      }
      return;
    }
    if (!ID_PATTERN.test(actionTargetId.trim())) {
      setActionError("Strategy ID must match ^[a-z][a-z0-9_]{0,63}$ (lowercase slug).");
      return;
    }
    setActionBusy(true);
    const result =
      actionDialog === "clone"
        ? await cloneStrategy(sourceStrategy.id, actionTargetId.trim())
        : await renameStrategy(sourceStrategy.id, actionTargetId.trim());
    setActionBusy(false);
    if (result.source === "live" && result.data.ok && result.data.strategy) {
      const strategy = result.data.strategy;
      setActionDialog(null);
      setLoadId(strategy.id);
      onDraftChange(draftFromStrategy(strategy));
      onStrategiesChanged();
      if (typeof window !== "undefined") {
        window.history.replaceState({}, "", `/studio?strategy=${encodeURIComponent(strategy.id)}`);
      }
      setSaveNote({ text: `${actionDialog === "clone" ? "Cloned" : "Renamed"} to ${strategy.id}.`, tone: "gain" });
      onNotify(`Strategy ${strategy.id} ${actionDialog === "clone" ? "cloned" : "renamed"}`, "gain");
    } else {
      setActionError(
        result.data.message || `${actionDialog === "clone" ? "Clone" : "Rename"} failed${result.error ? `: ${result.error}` : "."}`,
      );
    }
  };

  const downloadYaml = async () => {
    if (!sourceStrategy) return;
    const result = await downloadStrategyYaml(sourceStrategy.id);
    if (result.ok) {
      onNotify(`Downloaded ${sourceStrategy.id}.yaml`, "gain");
    } else {
      setSaveNote({ text: `YAML download failed${result.error ? `: ${result.error}` : "."}`, tone: "warn" });
    }
  };

  const copyYaml = async () => {
    try {
      await navigator.clipboard.writeText(yaml);
      onNotify("Strategy YAML copied to the clipboard", "gain");
    } catch {
      onNotify("Clipboard unavailable in this browser context", "warn");
    }
  };

  const updateParameter = (parameter: StudioParameter, value: StudioParamValue) => {
    applyDraft(setDraftValue(draft, parameter, value));
  };

  const changeSelector = (name: string) => {
    const roots = rootsOf(draft);
    const params = componentDefaults(catalog.parameters, "universe", name);
    if (roots.length > 0) params.roots = roots.join(",");
    applyDraft({ ...draft, universe: { name, params } });
  };

  const changeAllocator = (name: string) => {
    const params = componentDefaults(catalog.parameters, "allocator", name);
    if (name === "drl") {
      params.algorithm = params.algorithm ?? "PPO";
      params.policy = params.policy ?? "MlpPolicy";
      params.total_timesteps = params.total_timesteps ?? 100_000;
      params.learning_rate = params.learning_rate ?? 0.0003;
      params.gamma = params.gamma ?? 0.99;
      params.seed = params.seed ?? 7;
    }
    applyDraft({ ...draft, allocator: { name, params } });
  };

  const toggleOverlay = (
    stage: "timing" | "risk",
    name: string,
    enabled: boolean,
    parameters: StudioParameter[],
  ) => {
    const order: readonly string[] = stage === "timing" ? TIMING_ORDER : RISK_ORDER;
    const current = stage === "timing" ? draft.timing : draft.risk;
    const nextList = enabled
      ? current.filter((entry) => entry.name !== name)
      : [...current, { name, params: overlayDefaults(parameters, name) }].sort(
          (a, b) => order.indexOf(a.name) - order.indexOf(b.name),
        );
    applyDraft(stage === "timing" ? { ...draft, timing: nextList } : { ...draft, risk: nextList });
  };

  const toggleSync = () => {
    const next = !syncWithAllocator;
    setSyncWithAllocator(next);
    if (next) onDraftChange(syncQuantCaps(draft));
  };

  const stageParametersFor = (stage: string, name: string) =>
    catalog.parameters.filter(
      (parameter) =>
        parameter.stage === stage &&
        parameter.stage_name === name &&
        !parameter.path.endsWith(".enabled") &&
        !parameter.path.endsWith(".selector") &&
        !parameter.path.endsWith(".name"),
    );

  const universeParams = stageParametersFor("universe", draft.universe.name).filter(
    (parameter) => parameter.type !== "root_multiselect",
  );
  const allocatorParams = stageParametersFor("allocator", draft.allocator.name);
  const translatorParams = catalog.parameters.filter(
    (parameter) => parameter.stage === "translator" && !parameter.path.endsWith(".enabled"),
  );

  const stageIndex = STAGE_STEPS.findIndex((entry) => entry.key === activeStage);
  const activeMeta = stageMeta(activeStage);

  return (
    <div className="flex min-h-0 flex-col gap-3 xl:h-[calc(100vh-160px)]" data-testid="manual-builder">
      {/* ------------------------------------------------------------- top bar */}
      <section className="panel flex shrink-0 flex-wrap items-end gap-3 px-3 py-2.5">
        <Field label="Load existing" htmlFor="studio-load" className="w-[188px]">
          <Select
            id="studio-load"
            value={loadId}
            onChange={loadExisting}
            options={[
              { value: "", label: "— select saved strategy —" },
              ...strategies.map((strategy) => ({ value: strategy.id, label: strategy.id })),
            ]}
          />
        </Field>
        <Field label="New from hypothesis template" htmlFor="studio-template" className="w-[226px]">
          <Select
            id="studio-template"
            value={presetId}
            onChange={applyPreset}
            options={[
              { value: "", label: "— 5 built-in presets —" },
              ...catalog.presets.map((preset) => ({ value: preset.id, label: preset.name })),
            ]}
          />
        </Field>
        <Field label="Strategy ID" htmlFor="studio-id" className="w-[190px]">
          <TextInput
            id="studio-id"
            value={draft.id}
            onChange={(value) => onDraftChange({ ...draft, id: value.toLowerCase().replace(/\s+/g, "_") })}
          />
        </Field>
        <Field label="Backtest timeframe" htmlFor="studio-timeframe" className="w-[132px]">
          <Select
            id="studio-timeframe"
            value={timeframe}
            onChange={(value) => onTimeframeChange(value as DataTimeframe)}
            options={DATA_TIMEFRAME_OPTIONS.map((option) => ({ value: option.value, label: option.value }))}
          />
        </Field>
        <div className="flex items-center gap-2 pb-0.5">
          <Button
            variant="primary"
            onClick={() => void save("save")}
            disabled={!validation.valid || saving !== null}
            title={validation.valid ? "Validate and persist the strategy YAML" : "Fix contract errors before saving"}
          >
            {saving === "save" ? "Saving…" : "Save Strategy YAML"}
          </Button>
          <Button
            variant="secondary"
            onClick={() => void save("run")}
            disabled={!validation.valid || saving !== null}
            title="Save, then open the Backtest Lab with this strategy selected"
          >
            {saving === "run" ? "Saving…" : "Save & Run in Backtest Lab"}
          </Button>
        </div>
        <div className="flex items-center gap-1 pb-0.5">
          <Button
            variant="ghost"
            size="xs"
            onClick={() => openActionDialog("clone")}
            disabled={!sourceStrategy || actionBusy}
            testId="studio-clone"
            title="Clone the loaded strategy under a new id"
          >
            Clone
          </Button>
          <Button
            variant="ghost"
            size="xs"
            onClick={() => openActionDialog("rename")}
            disabled={!sourceStrategy || sourceBuiltin || actionBusy}
            testId="studio-rename"
            title={sourceBuiltin ? "Committed built-in strategies cannot be renamed" : "Rename the loaded custom strategy"}
          >
            Rename
          </Button>
          <Button
            variant="ghost"
            size="xs"
            onClick={() => void downloadYaml()}
            disabled={!sourceStrategy}
            testId="studio-download-yaml"
            title="Download the current strategy YAML"
          >
            <DownloadIcon className="h-3 w-3" />
            YAML
          </Button>
          {sourceStrategy && !sourceBuiltin ? (
            <Button
              variant="ghost"
              size="xs"
              onClick={() => openActionDialog("delete")}
              testId="studio-delete"
              title="Delete this custom strategy"
              ariaLabel="Delete loaded strategy"
            >
              <TrashIcon className="h-3.5 w-3.5 text-rose-300" />
            </Button>
          ) : null}
        </div>
        <div className="ml-auto flex min-w-0 items-center gap-2 pb-0.5 text-[11px]">
          {saveNote ? (
            <span
              className={cn(
                "truncate",
                saveNote.tone === "gain"
                  ? "text-emerald-400"
                  : saveNote.tone === "warn"
                    ? "text-amber-400"
                    : "text-zinc-500",
              )}
              title={saveNote.text}
            >
              {saveNote.text}
            </span>
          ) : (
            <span className="text-zinc-600">
              {draft.timing.length + draft.risk.length} overlays · {selectedRoots.length}/{rootIndex.size} roots ·{" "}
              {catalog.presets.length} templates
            </span>
          )}
        </div>
      </section>

      <div className="grid min-h-0 flex-1 grid-cols-12 gap-3">
        {/* ------------------------------------------------------- stage switcher */}
        <div className="col-span-12 flex min-h-0 flex-col gap-2.5 xl:col-span-8">
          <div className="panel flex shrink-0 items-stretch gap-0.5 overflow-x-auto p-1" role="tablist" aria-label="Pipeline stages">
            {STAGE_STEPS.map((step) => {
              const active = step.key === activeStage;
              return (
                <button
                  key={step.key}
                  type="button"
                  role="tab"
                  aria-selected={active}
                  data-testid={`stage-tab-${step.key}`}
                  onClick={() => setActiveStage(step.key)}
                  className={cn(
                    "focus-ring flex min-w-0 flex-1 items-center justify-center rounded px-2 py-1.5 text-[11px] font-medium transition-colors",
                    active
                      ? "bg-accent-soft text-blue-300"
                      : "text-zinc-500 hover:bg-surface hover:text-zinc-300",
                  )}
                >
                  <span className="truncate">{step.label}</span>
                </button>
              );
            })}
          </div>

          <section className="panel flex min-h-0 flex-1 flex-col" data-testid="stage-card">
            <header className="panel-header shrink-0">
              <div className="min-w-0">
                <h2 className="panel-title truncate">
                  {activeMeta.title} <span className="font-mono text-[10px] text-zinc-600">{activeMeta.code}</span>
                </h2>
                <p className="truncate text-[10.5px] text-zinc-600" title={activeMeta.subtitle}>
                  {activeMeta.subtitle}
                </p>
              </div>
              <div className="flex shrink-0 items-center gap-2">
                <Badge tone="neutral" className="hidden sm:inline-flex">
                  {stageSummary(draft, activeStage)}
                </Badge>
                <span className="font-mono text-[10px] text-zinc-600">
                  stage {stageIndex + 1} / {STAGE_STEPS.length}
                </span>
              </div>
            </header>
            <div className="min-h-0 flex-1 overflow-y-auto px-3 py-3">
              {activeStage === "universe" ? (
                <div className="flex flex-col gap-3">
                  <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                    <Field label="Selector Type" htmlFor="studio-selector">
                      <Select
                        id="studio-selector"
                        value={draft.universe.name}
                        onChange={changeSelector}
                        options={selectorChoices(catalog).map((choice) => ({
                          value: String(choice.value),
                          label: choice.label,
                        }))}
                      />
                    </Field>
                    <p className="self-end pb-1 text-[10.5px] leading-4 text-zinc-600">
                      The selector filters eligible roots every bar; the master universe below defines which sectors and
                      micro pairs the strategy may trade.
                    </p>
                  </div>
                  <ParameterBlock
                    parameters={universeParams}
                    draft={draft}
                    onUpdate={updateParameter}
                    emptyText="This selector has no extra parameters."
                  />
                  <UniverseRootPicker
                    sectors={sectors}
                    selected={selectedRoots}
                    onChange={(roots) => applyDraft(withRoots(draft, roots))}
                  />
                </div>
              ) : activeStage === "allocator" ? (
                <div className="flex flex-col gap-3">
                  <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                    <Field label="Allocator Type" htmlFor="studio-allocator">
                      <Select
                        id="studio-allocator"
                        value={draft.allocator.name}
                        onChange={changeAllocator}
                        options={allocatorChoices(catalog).map((choice) => ({
                          value: String(choice.value),
                          label: choice.label,
                        }))}
                      />
                    </Field>
                    <p className="self-end pb-1 text-[10.5px] leading-4 text-zinc-600">
                      Base weights before timing. Sizing labels here are the allocator ceilings, not the hard post-timing
                      risk caps in Stage 4.
                    </p>
                  </div>
                  <ParameterBlock
                    parameters={allocatorParams}
                    draft={draft}
                    onUpdate={updateParameter}
                    emptyText="This allocator has no extra parameters."
                  />
                  {draft.allocator.name === "drl" ? (
                    <DrlCheckpointSelect
                      models={drlModels}
                      draft={draft}
                      onUpdate={updateParameter}
                      onReplace={(next) => applyDraft(next)}
                    />
                  ) : null}
                </div>
              ) : activeStage === "timing" ? (
                <div className="flex flex-col gap-2.5">
                  {TIMING_OVERLAYS.map((overlay) => {
                    const parameters = stageParametersFor("timing", overlay.name);
                    const enabled = draft.timing.some((entry) => entry.name === overlay.name);
                    return (
                      <OverlayCard
                        key={overlay.name}
                        name={overlay.name}
                        title={overlay.title}
                        subtitle={overlay.subtitle}
                        enabled={enabled}
                        parameters={parameters}
                        draft={draft}
                        onUpdate={updateParameter}
                        onToggle={() => toggleOverlay("timing", overlay.name, enabled, parameters)}
                      />
                    );
                  })}
                </div>
              ) : activeStage === "risk" ? (
                <div className="flex flex-col gap-2.5">
                  <div className="rounded-md border border-accent/25 bg-accent-soft/30 px-2.5 py-2">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <button
                        type="button"
                        aria-pressed={syncWithAllocator}
                        onClick={toggleSync}
                        data-testid="sync-with-allocator"
                        className={cn(
                          "focus-ring inline-flex h-6 items-center gap-2 rounded border px-2 text-[11px] font-medium transition-colors",
                          syncWithAllocator
                            ? "border-accent/40 bg-accent text-white"
                            : "border-edge bg-surface text-zinc-300 hover:border-zinc-600",
                        )}
                      >
                        <span className="relative h-3 w-5 rounded-full bg-zinc-700">
                          <span
                            className={cn(
                              "absolute top-0.5 h-2 w-2 rounded-full bg-white transition-all",
                              syncWithAllocator ? "left-2.5" : "left-0.5",
                            )}
                          />
                        </span>
                        Sync with Allocator
                      </button>
                      <span className="font-mono text-[10px] text-zinc-500">
                        {formatSyncPreview(draft)}
                      </span>
                    </div>
                    <p className="mt-1 text-[10px] leading-4 text-zinc-500">
                      Precedence: Stage 2 Base Allocator sizing → these Stage 4 hard caps clamp post-timing weights → the
                      Translator enforces the η_max margin ceiling. Sync mirrors the Stage 2 target vol / max weight into the
                      quant-risk hard caps; disable to keep independent ceilings.
                    </p>
                  </div>
                  {RISK_OVERLAYS.map((overlay) => {
                    const parameters = stageParametersFor("risk", overlay.name);
                    const enabled = draft.risk.some((entry) => entry.name === overlay.name);
                    return (
                      <OverlayCard
                        key={overlay.name}
                        name={overlay.name}
                        title={overlay.title}
                        subtitle={overlay.subtitle}
                        enabled={enabled}
                        parameters={parameters}
                        draft={draft}
                        onUpdate={updateParameter}
                        onToggle={() => toggleOverlay("risk", overlay.name, enabled, parameters)}
                      />
                    );
                  })}
                </div>
              ) : (
                <div className="flex flex-col gap-3">
                  <ParameterBlock
                    parameters={translatorParams}
                    draft={draft}
                    onUpdate={updateParameter}
                    emptyText="No translator parameters published by the catalog."
                  />
                  <div className="rounded-md border border-edge bg-canvas px-2.5 py-2 text-[10.5px] leading-4 text-zinc-500">
                    Deterministic post-risk translation: rounding mode, rebalance buffer, micro routing and the hard{" "}
                    {(draft.translator.max_margin_utilization * 100).toFixed(0)}% η_max margin ceiling.
                  </div>
                </div>
              )}
            </div>
          </section>
        </div>

        {/* ------------------------------------------------------- sticky YAML */}
        <div className="col-span-12 min-h-0 xl:col-span-4">
          <div
            className="flex min-h-0 flex-col xl:sticky xl:top-3 xl:h-full xl:max-h-[calc(100vh-160px)]"
            data-testid="yaml-sticky"
          >
            <section className="panel flex h-full min-h-0 flex-col" data-testid="yaml-panel">
              <header className="panel-header shrink-0">
                <div className="flex min-w-0 items-center gap-2">
                  <h2 className="panel-title">Live YAML preview</h2>
                  <Badge tone={validation.valid ? "gain" : "loss"}>
                    {validation.valid
                      ? "Valid FinRL-X Pipeline Contract"
                      : `Invalid · ${validation.errors.length} issue${validation.errors.length === 1 ? "" : "s"}`}
                  </Badge>
                </div>
                <Button variant="ghost" size="xs" onClick={() => void copyYaml()}>
                  Copy
                </Button>
              </header>
              <div className="flex min-h-0 flex-1 flex-col gap-2 p-2.5">
                <div className="flex shrink-0 items-center justify-between gap-2 text-[10px] text-zinc-600">
                  <span className="truncate font-mono">configs/strategies/{draft.id || "strategy"}.yaml</span>
                  <span className="shrink-0 font-mono">{yaml.split("\n").length} lines</span>
                </div>
                <pre
                  className="scroll-area min-h-0 flex-1 overflow-y-auto whitespace-pre-wrap break-words rounded-md border border-hairline bg-canvas p-2.5 font-mono text-[10.5px] leading-[1.55] text-zinc-400"
                  data-testid="yaml-scroll"
                >
                  {yaml}
                </pre>
                {validation.errors.length > 0 ? (
                  <div className="max-h-20 shrink-0 overflow-y-auto rounded border border-loss/30 bg-loss-soft/40 px-2 py-1.5">
                    {validation.errors.slice(0, 4).map((error) => (
                      <p key={error} className="text-[10px] leading-4 text-rose-300">
                        · {error}
                      </p>
                    ))}
                  </div>
                ) : null}
                {validation.errors.length === 0 && validation.warnings.length > 0 ? (
                  <div className="max-h-20 shrink-0 overflow-y-auto rounded border border-warn/30 bg-warn-soft/40 px-2 py-1.5">
                    {validation.warnings.slice(0, 3).map((warning) => (
                      <p key={warning} className="text-[10px] leading-4 text-amber-300">
                        · {warning}
                      </p>
                    ))}
                  </div>
                ) : null}
                <div
                  className="flex shrink-0 items-center justify-between gap-2 rounded-md border border-edge bg-canvas px-2.5 py-2"
                  data-testid="eta-footer"
                >
                  <span className="text-[11px] text-zinc-500">Margin ceiling η_max</span>
                  <span className="flex items-center gap-2">
                    <span className="font-mono text-[11px] tabular-nums text-zinc-200">
                      {(draft.translator.max_margin_utilization * 100).toFixed(0)}%
                    </span>
                    <Badge tone={draft.translator.max_margin_utilization <= 0.6 ? "gain" : "loss"}>
                      {draft.translator.max_margin_utilization <= 0.6 ? "≤ 60% policy cap" : "breach"}
                    </Badge>
                  </span>
                </div>
              </div>
            </section>
          </div>
        </div>
      </div>
      {actionDialog ? (
        <Modal
          open
          title={
            actionDialog === "clone"
              ? "Clone strategy"
              : actionDialog === "rename"
                ? "Rename strategy"
                : "Delete strategy"
          }
          description={
            <span>
              {actionDialog === "delete" ? (
                <>
                  Permanently delete <span className="font-mono text-zinc-300">{sourceStrategy?.id}</span> and its
                  YAML artifact.
                </>
              ) : (
                <>
                  {actionDialog === "clone" ? "Create a copy of" : "Rename"}{" "}
                  <span className="font-mono text-zinc-300">{sourceStrategy?.id}</span> to a new strategy id.
                </>
              )}
            </span>
          }
          onClose={() => {
            if (actionBusy) return;
            setActionDialog(null);
            setActionError(null);
          }}
          busy={actionBusy}
          footer={
            <>
              <Button
                variant="ghost"
                onClick={() => {
                  setActionDialog(null);
                  setActionError(null);
                }}
                disabled={actionBusy}
              >
                Cancel
              </Button>
              <Button
                variant={actionDialog === "delete" ? "danger" : "primary"}
                onClick={() => void runStrategyAction()}
                disabled={
                  actionBusy ||
                  (actionDialog !== "delete" && !ID_PATTERN.test(actionTargetId.trim()))
                }
                testId={actionDialog === "delete" ? "confirm-delete-strategy" : "confirm-strategy-action"}
              >
                {actionBusy
                  ? actionDialog === "delete"
                    ? "Deleting…"
                    : "Saving…"
                  : actionDialog === "delete"
                    ? "Delete strategy"
                    : actionDialog === "clone"
                      ? "Clone strategy"
                      : "Rename strategy"}
              </Button>
            </>
          }
        >
          <div className="space-y-2">
            {actionDialog !== "delete" ? (
              <Field label="New strategy ID" htmlFor="studio-action-id">
                <TextInput
                  id="studio-action-id"
                  value={actionTargetId}
                  onChange={(value) => setActionTargetId(value.toLowerCase().replace(/\s+/g, "_"))}
                  disabled={actionBusy}
                />
              </Field>
            ) : null}
            {actionError ? (
              <div
                role="alert"
                data-testid="strategy-action-error"
                className="rounded-md border border-loss/40 bg-loss-soft/40 px-2.5 py-2 text-[11px] leading-4 text-rose-300"
              >
                {actionError}
              </div>
            ) : null}
          </div>
        </Modal>
      ) : null}

      {saveToast ? (
        <div
          data-testid="save-toast"
          role="status"
          className="fixed bottom-12 right-4 z-50 w-[300px] rounded-md border border-accent/40 bg-panel px-3 py-2.5 shadow-popover"
        >
          <div className="flex items-start justify-between gap-2">
            <div>
              <p className="text-[11.5px] font-medium text-blue-300">Strategy saved & opening Backtest Lab</p>
              <p className="mt-0.5 truncate font-mono text-[10px] text-zinc-500">{saveToast.strategyId}</p>
            </div>
            <button
              type="button"
              aria-label="Dismiss toast"
              onClick={() => setSaveToast(null)}
              className="focus-ring -mr-1 grid h-5 w-5 place-items-center rounded text-zinc-500 hover:bg-surface hover:text-zinc-300"
            >
              ×
            </button>
          </div>
          <a
            href={`/backtests?strategy=${encodeURIComponent(saveToast.strategyId)}`}
            className="mt-1.5 inline-block text-[10.5px] font-medium text-blue-300 underline decoration-blue-500/40 underline-offset-2 hover:text-blue-200"
          >
            Open Backtest Lab now
          </a>
        </div>
      ) : null}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Stage content building blocks
// ---------------------------------------------------------------------------

function ParameterBlock({
  parameters,
  draft,
  onUpdate,
  emptyText,
}: {
  parameters: StudioParameter[];
  draft: StudioDraft;
  onUpdate: (parameter: StudioParameter, value: StudioParamValue) => void;
  emptyText: string;
}) {
  const [advancedOpen, setAdvancedOpen] = useState(false);
  if (parameters.length === 0) return <p className="text-[11px] text-zinc-600">{emptyText}</p>;
  const primary = parameters.filter((parameter) => !isAdvancedParameter(parameter));
  const advanced = parameters.filter((parameter) => isAdvancedParameter(parameter));
  return (
    <div>
      <div className="grid grid-cols-1 gap-x-4 gap-y-3 sm:grid-cols-2">
        {primary.map((parameter) => (
          <ParameterControl key={parameter.path} parameter={parameter} draft={draft} onUpdate={onUpdate} />
        ))}
      </div>
      {advanced.length > 0 ? (
        <div className="mt-3 border-t border-hairline pt-2">
          <button
            type="button"
            onClick={() => setAdvancedOpen((open) => !open)}
            aria-expanded={advancedOpen}
            data-testid="advanced-toggle"
            className="focus-ring flex items-center gap-1.5 rounded px-1 py-0.5 text-[10.5px] font-medium text-zinc-500 hover:text-zinc-300"
          >
            {advancedOpen ? <ChevronUpIcon className="h-3 w-3" /> : <ChevronDownIcon className="h-3 w-3" />}
            Advanced Parameters ({advanced.length})
          </button>
          {advancedOpen ? (
            <div className="mt-2 grid grid-cols-1 gap-x-4 gap-y-3 sm:grid-cols-2">
              {advanced.map((parameter) => (
                <ParameterControl key={parameter.path} parameter={parameter} draft={draft} onUpdate={onUpdate} />
              ))}
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

function ParameterControl({
  parameter,
  draft,
  onUpdate,
}: {
  parameter: StudioParameter;
  draft: StudioDraft;
  onUpdate: (parameter: StudioParameter, value: StudioParamValue) => void;
}) {
  const display = { ...parameter, label: parameterLabel(parameter.stage, parameter) };
  const current = getDraftValue(draft, parameter) ?? parameter.default;
  const update = (value: StudioParamValue) => onUpdate(parameter, value);

  if (parameter.type === "multiselect" && parameter.options && parameter.options.length > 0) {
    const rawSelected = Array.isArray(current)
      ? current
      : typeof current === "string" && current.trim().startsWith("{")
        ? Object.keys(JSON.parse(current) as Record<string, unknown>)
        : typeof current === "string" && current.length > 0
          ? current.split(",")
          : [];
    const selected = rawSelected.map((entry) => String(entry));
    return (
      <div className="sm:col-span-2">
        <div className="mb-1 flex items-baseline justify-between gap-2">
          <span className="text-[11px] font-medium text-zinc-400">{display.label}</span>
          <span className="font-mono text-[10px] text-zinc-500">{selected.length} selected</span>
        </div>
        <div className="flex flex-wrap gap-1">
          {parameter.options.map((option) => {
            const value = String(option.value);
            const active = selected.includes(value);
            return (
              <button
                key={value}
                type="button"
                aria-pressed={active}
                onClick={() =>
                  update(active ? selected.filter((entry) => entry !== value) : [...selected, value])
                }
                className={cn(
                  "focus-ring h-6 rounded border px-1.5 font-mono text-[10.5px] font-medium transition-colors",
                  active
                    ? "border-accent/40 bg-accent-soft text-blue-300"
                    : "border-edge bg-canvas text-zinc-500 hover:border-zinc-600 hover:text-zinc-300",
                )}
              >
                {option.label}
              </button>
            );
          })}
        </div>
      </div>
    );
  }
  if (parameter.type === "select") {
    return (
      <div>
        <div className="mb-1 text-[11px] font-medium text-zinc-400">{display.label}</div>
        <Select
          value={String(current)}
          onChange={(value) => update(value)}
          options={(parameter.options ?? []).map((option) => ({
            value: String(option.value),
            label: option.label,
          }))}
        />
      </div>
    );
  }
  if (parameter.type === "bool") {
    return (
      <div className="sm:col-span-2">
        <ToggleRow
          label={display.label}
          description={parameter.description}
          checked={Boolean(current)}
          onChange={(checked) => update(checked)}
        />
      </div>
    );
  }
  return (
    <RangeField
      parameter={display}
      value={Number(current)}
      onChange={(value) => update(value)}
      badge={numericBadge(parameter, Number(current))}
    />
  );
}

function OverlayCard({
  name,
  title,
  subtitle,
  enabled,
  parameters,
  draft,
  onUpdate,
  onToggle,
}: {
  name: string;
  title: string;
  subtitle: string;
  enabled: boolean;
  parameters: StudioParameter[];
  draft: StudioDraft;
  onUpdate: (parameter: StudioParameter, value: StudioParamValue) => void;
  onToggle: () => void;
}) {
  return (
    <section
      className={cn(
        "rounded-md border bg-panel",
        enabled ? "border-edge" : "border-hairline bg-panel/60",
      )}
      data-testid={`overlay-card-${name}`}
    >
      <div className="flex items-center justify-between gap-2 px-2.5 py-2">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <span className={cn("text-[12px] font-medium", enabled ? "text-zinc-100" : "text-zinc-400")}>
              {title}
            </span>
            <span className="font-mono text-[10px] text-zinc-600">{name}</span>
            {!enabled ? <Badge tone="neutral">disabled</Badge> : null}
          </div>
          <p className="mt-0.5 truncate text-[10px] text-zinc-600" title={subtitle}>
            {subtitle}
          </p>
        </div>
        <button
          type="button"
          aria-pressed={enabled}
          onClick={onToggle}
          data-testid={`overlay-toggle-${name}`}
          className={cn(
            "focus-ring inline-flex h-6 w-[62px] shrink-0 items-center justify-between rounded border px-1 font-mono text-[10px] font-medium transition-colors",
            enabled
              ? "border-accent/40 bg-accent-soft text-blue-300"
              : "border-edge bg-surface text-zinc-500 hover:text-zinc-300",
          )}
        >
          <span className="relative h-3 w-5 rounded-full bg-zinc-700">
            <span
              className={cn(
                "absolute top-0.5 h-2 w-2 rounded-full bg-white transition-all",
                enabled ? "left-2.5" : "left-0.5",
              )}
            />
          </span>
          {enabled ? "ON" : "OFF"}
        </button>
      </div>
      {enabled ? (
        <div className="border-t border-hairline px-2.5 py-2.5">
          <ParameterBlock parameters={parameters} draft={draft} onUpdate={onUpdate} emptyText="No parameters published for this overlay." />
        </div>
      ) : null}
    </section>
  );
}

function UniverseRootPicker({
  sectors,
  selected,
  onChange,
}: {
  sectors: InstrumentSector[];
  selected: string[];
  onChange: (roots: string[]) => void;
}) {
  const [sectorFilter, setSectorFilter] = useState("all");
  const [query, setQuery] = useState("");
  const selectedSet = useMemo(() => new Set(selected), [selected]);
  const allRoots = useMemo(() => universeRoots(sectors), [sectors]);
  const sectorById = useMemo(() => new Map(sectors.map((sector) => [sector.id, sector])), [sectors]);
  const totalSelected = useMemo(
    () => allRoots.filter((root) => selectedSet.has(root.root)).length,
    [allRoots, selectedSet],
  );

  const commit = (roots: string[]) => onChange(orderUniverseRoots([...new Set(roots)], sectors));

  const toggleRoot = (root: string) => {
    const next = new Set(selectedSet);
    if (next.has(root)) next.delete(root);
    else next.add(root);
    commit([...next]);
  };

  const selectSector = (sector: InstrumentSector | null) => {
    const roots = sector ? sector.roots.map((root) => root.root) : allRoots.map((root) => root.root);
    commit([...selected, ...roots]);
  };

  const clearSector = (sector: InstrumentSector | null) => {
    const drop = new Set((sector ? sector.roots : allRoots).map((root) => root.root));
    commit(selected.filter((root) => !drop.has(root)));
  };

  const searching = query.trim().length > 0;
  const matches = (root: InstrumentRoot) => {
    if (searching) {
      // Global search: match every root in the universe regardless of the
      // currently active sector tab.
      const q = query.trim().toLowerCase();
      return root.root.toLowerCase().includes(q) || root.name.toLowerCase().includes(q);
    }
    return sectorFilter === "all" || root.sector === sectorFilter;
  };

  const matchCount = allRoots.filter(matches).length;
  const visibleSectors = searching
    ? sectors
    : sectorFilter === "all"
      ? sectors
      : sectors.filter((sector) => sector.id === sectorFilter);
  const activeSector = sectorFilter === "all" ? null : sectorById.get(sectorFilter) ?? null;
  const activeLabel = searching ? "all roots" : activeSector?.label ?? "all roots";

  return (
    <div className="rounded-md border border-edge bg-canvas/50" data-testid="universe-root-picker">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-hairline px-2.5 py-2">
        <div className="flex items-center gap-2">
          <span className="text-[12px] font-medium text-zinc-200">Master universe roots</span>
          <Badge tone="accent">{totalSelected} / {allRoots.length} selected</Badge>
          <span className="hidden text-[10px] text-zinc-600 sm:inline">
            {sectors.length} sectors · paired micro roots included
          </span>
        </div>
        <div className="flex w-full flex-wrap items-center justify-end gap-2 2xl:w-auto">
          <TextInput
            id="universe-root-search"
            value={query}
            onChange={setQuery}
            placeholder="Search all 45 roots / names"
            className="h-6 w-[178px] text-[11px]"
          />
          {searching ? (
            <span className="whitespace-nowrap text-[9.5px] text-accent" data-testid="universe-search-count">
              {matchCount} match{matchCount === 1 ? "" : "es"} · all sectors
            </span>
          ) : null}
          <Button variant="secondary" size="xs" onClick={() => selectSector(activeSector)} testId="select-all-sector">
            Select all {activeSector ? activeSector.label : "roots"}
          </Button>
          <Button variant="ghost" size="xs" onClick={() => clearSector(activeSector)} testId="clear-sector">
            Clear {activeSector ? activeSector.label : "all"}
          </Button>
        </div>
      </div>
      <div className="flex flex-wrap gap-1 border-b border-hairline px-2 py-1.5" role="tablist" aria-label="Universe sectors">
        {SECTOR_TABS.map((tab) => {
          const sector = sectors.find((entry) => entry.id === tab.id);
          if (tab.id !== "all" && !sector) return null;
          const active = sectorFilter === tab.id;
          const count = tab.id === "all" ? allRoots.length : sector?.count ?? 0;
          return (
            <button
              key={tab.id}
              type="button"
              role="tab"
              aria-selected={active}
              data-testid={`sector-tab-${tab.id}`}
              onClick={() => setSectorFilter(tab.id)}
              className={cn(
                "focus-ring h-6 rounded border px-1.5 text-[10.5px] font-medium transition-colors",
                active
                  ? "border-accent/40 bg-accent-soft text-blue-300"
                  : "border-transparent text-zinc-500 hover:bg-surface hover:text-zinc-300",
              )}
            >
              {tab.label}
              <span className="ml-1 font-mono text-[9.5px] text-zinc-600">{count}</span>
            </button>
          );
        })}
      </div>
      <div className="px-2.5 py-2">
        {visibleSectors.length === 0 ? (
          <p className="text-[10.5px] text-zinc-600">No instruments returned by GET /api/instruments.</p>
        ) : searching && matchCount === 0 ? (
          <p className="text-[10.5px] text-zinc-500" data-testid="universe-search-empty">
            No roots match “{query.trim()}”. Clear the search to restore the active sector view.
          </p>
        ) : (
          <div className="flex flex-col gap-2">
            {visibleSectors.map((sector) => {
              const roots = sector.roots.filter(matches);
              if (roots.length === 0) return null;
              const sectorSelected = sector.roots.filter((root) => selectedSet.has(root.root)).length;
              return (
                <div key={sector.id}>
                  <div className="mb-1 flex items-center justify-between gap-2">
                    <span className="text-[10px] font-medium uppercase tracking-[0.05em] text-zinc-500">
                      {sector.label}
                      <span className="ml-1.5 font-mono normal-case tracking-normal text-zinc-600">
                        {sectorSelected}/{sector.count}
                      </span>
                    </span>
                    <span className="flex items-center gap-1">
                      <button
                        type="button"
                        onClick={() => selectSector(sector)}
                        className="focus-ring rounded px-1 py-0.5 text-[9.5px] text-zinc-500 hover:bg-surface hover:text-zinc-300"
                      >
                        Select all
                      </button>
                      <button
                        type="button"
                        onClick={() => clearSector(sector)}
                        className="focus-ring rounded px-1 py-0.5 text-[9.5px] text-zinc-500 hover:bg-surface hover:text-zinc-300"
                      >
                        Clear
                      </button>
                    </span>
                  </div>
                  <div className="flex flex-wrap gap-1">
                    {roots.map((root) => {
                      const active = selectedSet.has(root.root);
                      return (
                        <button
                          key={root.root}
                          type="button"
                          aria-pressed={active}
                          data-testid={`root-pill-${root.root}`}
                          title={`${root.name} · ${root.exchange} · ${root.multiplier}× tick ${root.tick_size}${
                            root.micro_root ? ` · micro ${root.micro_root}` : root.standard_root ? ` · rolls up to ${root.standard_root}` : ""
                          }`}
                          onClick={() => toggleRoot(root.root)}
                          className={cn(
                            "focus-ring inline-flex h-6 items-center gap-1 rounded border px-1.5 font-mono text-[10.5px] font-medium transition-colors",
                            active
                              ? "border-accent/40 bg-accent-soft text-blue-300"
                              : "border-edge bg-canvas text-zinc-500 hover:border-zinc-600 hover:text-zinc-300",
                          )}
                        >
                          <span>{root.root}</span>
                          {root.micro_root ? (
                            <span className="text-[8.5px] text-zinc-500">↔{root.micro_root}</span>
                          ) : root.standard_root ? (
                            <span className="text-[8.5px] text-zinc-500">→{root.standard_root}</span>
                          ) : null}
                        </button>
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
      <div className="border-t border-hairline px-2.5 py-1.5 text-[10px] text-zinc-600">
        Showing {activeLabel} · selected roots are written to <span className="font-mono">universe.params.roots</span> and the
        YAML draft; micro pairs remain available for the Contract Translator.
      </div>
    </div>
  );
}

function DrlCheckpointSelect({
  models,
  draft,
  onUpdate,
  onReplace,
}: {
  models: DrlModel[];
  draft: StudioDraft;
  onUpdate: (parameter: StudioParameter, value: StudioParamValue) => void;
  onReplace: (draft: StudioDraft) => void;
}) {
  const currentPath = typeof draft.allocator.params.model_path === "string" ? draft.allocator.params.model_path : "";
  const selected =
    models.find((model) => model.path === currentPath || model.name === currentPath) ?? null;
  const checkpointParameter: StudioParameter = {
    path: "allocator.params.model_path",
    key: "allocator.drl.model_path",
    stage: "allocator",
    group: "drl",
    label: "DRL checkpoint",
    description: "Trained SB3 checkpoint bound to the DRL allocator (from GET /api/drl/models).",
    type: "select",
    default: "",
    min: null,
    max: null,
    step: null,
    options: [],
    unit: null,
    optuna_tunable: false,
  };

  const select = (path: string) => {
    const model = models.find((entry) => (entry.path ?? entry.name) === path);
    if (!model) {
      onUpdate(checkpointParameter, "");
      return;
    }
    const next: StudioDraft = {
      ...draft,
      allocator: {
        ...draft.allocator,
        params: {
          ...draft.allocator.params,
          model_path: model.path ?? `artifacts/models/${model.name}.zip`,
          algorithm: model.algorithm || "PPO",
          total_timesteps: model.timesteps > 0 ? model.timesteps : draft.allocator.params.total_timesteps ?? 100_000,
        },
      },
    };
    onReplace(next);
  };

  return (
    <div className="rounded-md border border-edge bg-canvas/50 px-2.5 py-2" data-testid="drl-checkpoint-panel">
      <div className="mb-1 flex items-center justify-between gap-2">
        <span className="text-[11px] font-medium text-zinc-300">DRL checkpoint / model</span>
        <Badge tone="neutral">GET /api/drl/models</Badge>
      </div>
      {models.length === 0 ? (
        <p className="text-[10.5px] leading-4 text-zinc-600">
          No trained checkpoints returned. Train one in the DRL Neural Trainer sub-tab, then pick it here to bind the
          allocator to its <span className="font-mono">model_path</span>.
        </p>
      ) : (
        <>
          <Select
            value={selected ? (selected.path ?? selected.name) : ""}
            onChange={select}
            options={[
              { value: "", label: "— select trained checkpoint —" },
              ...models.map((model) => ({
                value: model.path ?? model.name,
                label: `${model.name} · ${model.algorithm || "?"} · ${model.timesteps.toLocaleString("en-US")} steps`,
              })),
            ]}
          />
          <div className="mt-1.5 flex flex-wrap items-center gap-2 text-[10px] text-zinc-600">
            {selected ? (
              <>
                <span className="font-mono">{selected.path ?? selected.name}</span>
                <span>{selected.universe.length > 0 ? selected.universe.join(" · ") : "universe n/a"}</span>
                {selected.reward_summary?.best_mean_reward !== null && selected.reward_summary?.best_mean_reward !== undefined ? (
                  <span className="font-mono text-emerald-400">
                    best {selected.reward_summary.best_mean_reward.toFixed(3)}
                  </span>
                ) : null}
              </>
            ) : (
              <span>Selecting a checkpoint writes its model_path, algorithm and timesteps into the allocator params.</span>
            )}
          </div>
        </>
      )}
    </div>
  );
}

function formatSyncPreview(draft: StudioDraft): string {
  const allocVol = draft.allocator.params.target_annual_volatility;
  const allocWeight =
    draft.allocator.params.max_weight ?? draft.allocator.params.max_single_instrument_weight;
  const quant = draft.risk.find((stage) => stage.name === "quant_risk");
  const vol = typeof allocVol === "number" ? `${(allocVol * 100).toFixed(1)}%` : "—";
  const weight = typeof allocWeight === "number" ? `${(allocWeight * 100).toFixed(1)}%` : "—";
  const hardVol = typeof quant?.params.vol_target === "number" ? `${(quant.params.vol_target * 100).toFixed(1)}%` : "—";
  const hardWeight =
    typeof quant?.params.max_single_instrument_weight === "number"
      ? `${(quant.params.max_single_instrument_weight * 100).toFixed(1)}%`
      : "—";
  return `Stage2 ${vol} / ${weight} → Stage4 ${hardVol} / ${hardWeight}`;
}
