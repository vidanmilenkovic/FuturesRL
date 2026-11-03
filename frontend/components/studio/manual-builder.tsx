"use client";

import { useRouter } from "next/navigation";
import { useMemo, useState } from "react";

import {
  Badge,
  Button,
  Field,
  Panel,
  Select,
  TextInput,
  type Tone,
} from "@/components/ui/primitives";
import { saveStudioStrategy } from "@/lib/api";
import { cn } from "@/lib/cn";
import {
  buildEncodingMap,
  draftFromPreset,
  draftFromStrategy,
  draftToYaml,
  getDraftValue,
  rootsOf,
  setDraftValue,
  stageMeta,
  stageSummary,
  validateDraft,
  withDefaultRoots,
} from "@/lib/studio";
import type {
  StrategyConfig,
  StudioCatalogResponse,
  StudioDraft,
  StudioParameter,
} from "@/lib/types";
import { ChevronDownIcon, ChevronUpIcon } from "@/components/shell/icons";

import { RangeField, RootPills, ToggleRow } from "./controls";

const STAGE_ORDER = ["universe", "allocator", "timing", "risk", "translator"] as const;

type StageKey = (typeof STAGE_ORDER)[number];

export function ManualBuilder({
  draft,
  catalog,
  strategies,
  onDraftChange,
  onStrategiesChanged,
  onNotify,
}: {
  draft: StudioDraft;
  catalog: StudioCatalogResponse;
  strategies: StrategyConfig[];
  onDraftChange: (draft: StudioDraft) => void;
  onStrategiesChanged: () => void;
  onNotify: (message: string, tone: Tone) => void;
}) {
  const router = useRouter();
  const [expanded, setExpanded] = useState<Set<StageKey>>(
    () => new Set<StageKey>(["universe", "allocator", "translator"]),
  );
  const [loadId, setLoadId] = useState("");
  const [presetId, setPresetId] = useState("");
  const [saving, setSaving] = useState<null | "save" | "run">(null);
  const [saveNote, setSaveNote] = useState<{ text: string; tone: Tone } | null>(null);

  const validation = useMemo(() => validateDraft(draft), [draft]);
  const encodings = useMemo(() => buildEncodingMap(catalog.parameters), [catalog.parameters]);
  const yaml = useMemo(() => draftToYaml(draft, encodings), [draft, encodings]);
  const editingExisting = strategies.some((strategy) => strategy.id === draft.id);
  const selectedRoots = useMemo(() => rootsOf(draft), [draft]);

  const toggleStage = (stage: StageKey) => {
    setExpanded((current) => {
      const next = new Set(current);
      if (next.has(stage)) next.delete(stage);
      else next.add(stage);
      return next;
    });
  };

  const loadExisting = (id: string) => {
    setLoadId(id);
    const strategy = strategies.find((entry) => entry.id === id);
    if (!strategy) return;
    onDraftChange(withDefaultRoots(draftFromStrategy(strategy), catalog.roots));
    setSaveNote({ text: `Loaded ${strategy.id} from the strategy catalog.`, tone: "neutral" });
  };

  const applyPreset = (id: string) => {
    setPresetId(id);
    const preset = catalog.presets.find((entry) => entry.id === id);
    if (!preset) return;
    onDraftChange(withDefaultRoots(draftFromPreset(preset), catalog.roots));
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
      router.push(`/backtests?strategy=${encodeURIComponent(id)}`);
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

  return (
    <div className="grid min-h-full grid-cols-12 gap-3">
      {/* ------------------------------------------------------------- top bar */}
      <section className="panel col-span-12 flex flex-wrap items-end gap-3 px-3 py-2.5">
        <Field label="Load existing" htmlFor="studio-load" className="w-[196px]">
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
        <Field label="New from hypothesis template" htmlFor="studio-template" className="w-[238px]">
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
        <Field label="Strategy ID" htmlFor="studio-id" className="w-[176px]">
          <TextInput
            id="studio-id"
            value={draft.id}
            onChange={(value) => onDraftChange({ ...draft, id: value.toLowerCase().replace(/\s+/g, "_") })}
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
        <div className="ml-auto flex min-w-0 items-center gap-2 pb-0.5 text-[11px]">
          {saveNote ? (
            <span className={cn("truncate", saveNote.tone === "gain" ? "text-emerald-400" : saveNote.tone === "warn" ? "text-amber-400" : "text-zinc-500")} title={saveNote.text}>
              {saveNote.text}
            </span>
          ) : (
            <span className="text-zinc-600">
              {draft.timing.length + draft.risk.length} overlays · {selectedRoots.length} roots ·{" "}
              {catalog.presets.length} templates
            </span>
          )}
        </div>
      </section>

      {/* ------------------------------------------------------ 5-stage builder */}
      <div className="col-span-12 flex flex-col gap-2.5 xl:col-span-8">
        {STAGE_ORDER.map((stage, index) => {
          const meta = stageMeta(stage);
          const isOpen = expanded.has(stage);
          // The live catalog describes every registry component; only show the
          // parameters of the component currently selected in the draft.
          const activeComponent =
            stage === "universe" ? draft.universe.name : stage === "allocator" ? draft.allocator.name : "";
          const stageParameters = catalog.parameters.filter(
            (parameter) =>
              parameter.stage === stage &&
              (activeComponent === "" ||
                !parameter.stage_name ||
                parameter.stage_name === activeComponent ||
                stage === "translator"),
          );
          const groups = new Map<string, StudioParameter[]>();
          for (const parameter of stageParameters) {
            const key = parameter.group ?? "";
            const list = groups.get(key) ?? [];
            list.push(parameter);
            groups.set(key, list);
          }
          return (
            <section key={stage} className="rounded-md border border-edge bg-panel shadow-panel">
              <button
                type="button"
                onClick={() => toggleStage(stage)}
                aria-expanded={isOpen}
                className="focus-ring flex w-full items-center gap-3 px-3 py-2.5 text-left"
              >
                <span className="grid h-6 w-6 shrink-0 place-items-center rounded border border-edge bg-canvas font-mono text-[10px] text-zinc-500">
                  {index + 1}
                </span>
                <span className="min-w-0">
                  <span className="flex items-center gap-2">
                    <span className="text-[13px] font-medium text-zinc-100">{meta.title}</span>
                    <span className="font-mono text-[10.5px] text-zinc-500">{meta.code}</span>
                    <Badge tone="neutral" className="hidden sm:inline-flex">{stageSummary(draft, stage)}</Badge>
                  </span>
                  <span className="mt-0.5 block truncate text-[11px] text-zinc-600" title={meta.subtitle}>
                    {meta.subtitle}
                  </span>
                </span>
                <span className="ml-auto shrink-0 text-zinc-600">
                  {isOpen ? <ChevronUpIcon className="h-4 w-4" /> : <ChevronDownIcon className="h-4 w-4" />}
                </span>
              </button>
              {isOpen ? (
                <div className="border-t border-hairline px-3 py-3">
                  {stage === "timing" || stage === "risk" ? (
                    <div className="flex flex-col gap-2">
                      {Array.from(groups.entries()).map(([groupName, groupParameters]) => {
                        const enabledParameter = groupParameters.find(
                          (parameter) => parameter.type === "bool" && parameter.path.endsWith(".enabled"),
                        );
                        const controls = groupParameters.filter((parameter) => parameter !== enabledParameter);
                        const overlays = stage === "timing" ? draft.timing : draft.risk;
                        const enabled = enabledParameter
                          ? Boolean(getDraftValue(draft, enabledParameter) ?? enabledParameter.default)
                          : groupName === "" || overlays.some((entry) => entry.name === groupName);
                        const heading = groupName || meta.title;
                        const toggleOverlay = () => {
                          if (enabledParameter) {
                            onDraftChange(setDraftValue(draft, enabledParameter, !enabled));
                            return;
                          }
                          if (groupName === "") return;
                          const next = { ...draft };
                          if (enabled) {
                            if (stage === "timing") {
                              next.timing = draft.timing.filter((entry) => entry.name !== groupName);
                            } else {
                              next.risk = draft.risk.filter((entry) => entry.name !== groupName);
                            }
                          } else {
                            const defaults: Record<string, string | number | boolean | string[]> = {};
                            for (const parameter of controls) {
                              const value = parameter.default;
                              defaults[parameter.path.split(".").pop() ?? parameter.path] = value;
                            }
                            const added = { name: groupName, params: defaults };
                            if (stage === "timing") next.timing = [...draft.timing, added];
                            else next.risk = [...draft.risk, added];
                          }
                          onDraftChange(next);
                        };
                        return (
                          <div key={`${stage}-${groupName}`} className="rounded-md border border-hairline bg-canvas/40 px-2.5 py-2">
                            <div className="mb-2 flex items-center justify-between gap-2">
                              <span className="text-[11px] font-medium text-zinc-400">{heading}</span>
                              {enabledParameter ? (
                                <button
                                  type="button"
                                  aria-pressed={enabled}
                                  onClick={toggleOverlay}
                                  className={cn(
                                    "focus-ring rounded border px-1.5 py-0.5 font-mono text-[10px] font-medium transition-colors",
                                    enabled
                                      ? "border-accent/40 bg-accent-soft text-blue-300"
                                      : "border-edge bg-surface text-zinc-500 hover:text-zinc-300",
                                  )}
                                >
                                  {enabled ? "ON" : "OFF"}
                                </button>
                              ) : null}
                            </div>
                            {enabled ? (
                              <ParameterGrid
                                parameters={controls}
                                draft={draft}
                                roots={catalog.roots}
                                onDraftChange={onDraftChange}
                              />
                            ) : (
                              <p className="text-[10.5px] text-zinc-600">
                                Overlay disabled — its parameters are omitted from the pipeline contract.
                              </p>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  ) : (
                    <ParameterGrid
                      parameters={stageParameters}
                      draft={draft}
                      roots={catalog.roots}
                      onDraftChange={onDraftChange}
                    />
                  )}
                </div>
              ) : null}
            </section>
          );
        })}
      </div>

      {/* ------------------------------------------------------- YAML preview */}
      <div className="col-span-12 xl:col-span-4">
        <Panel
          className="h-full"
          title="Live YAML preview"
          actions={
            <Badge tone={validation.valid ? "gain" : "loss"}>
              {validation.valid
                ? "Valid FinRL-X Pipeline Contract"
                : `Invalid · ${validation.errors.length} issue${validation.errors.length === 1 ? "" : "s"}`}
            </Badge>
          }
        >
          <div className="flex min-h-0 flex-1 flex-col gap-2">
            <div className="flex items-center justify-between gap-2 text-[10.5px] text-zinc-600">
              <span className="truncate font-mono">configs/strategies/{draft.id || "strategy"}.yaml</span>
              <Button variant="ghost" size="xs" onClick={() => void copyYaml()}>
                Copy YAML
              </Button>
            </div>
            <pre className="scroll-area min-h-[320px] flex-1 overflow-auto rounded-md border border-hairline bg-canvas p-2.5 font-mono text-[10.5px] leading-[1.55] text-zinc-400">
              {yaml}
            </pre>
            {validation.errors.length > 0 ? (
              <div className="rounded-md border border-loss/30 bg-loss-soft/40 px-2.5 py-2">
                {validation.errors.slice(0, 4).map((error) => (
                  <p key={error} className="text-[10.5px] leading-4 text-rose-300">
                    · {error}
                  </p>
                ))}
              </div>
            ) : null}
            {validation.warnings.length > 0 ? (
              <div className="rounded-md border border-warn/30 bg-warn-soft/40 px-2.5 py-2">
                {validation.warnings.slice(0, 3).map((warning) => (
                  <p key={warning} className="text-[10.5px] leading-4 text-amber-300">
                    · {warning}
                  </p>
                ))}
              </div>
            ) : null}
            <div className="flex items-center justify-between gap-2 rounded-md border border-edge bg-canvas px-2.5 py-2">
              <span className="text-[11px] text-zinc-500">Margin ceiling η_max</span>
              <span className="flex items-center gap-2">
                <span className="font-mono text-[11px] tabular-nums text-zinc-300">
                  {(draft.translator.max_margin_utilization * 100).toFixed(0)}%
                </span>
                <Badge tone={draft.translator.max_margin_utilization <= 0.6 ? "gain" : "loss"}>
                  {draft.translator.max_margin_utilization <= 0.6 ? "≤ 60% policy cap" : "breach"}
                </Badge>
              </span>
            </div>
          </div>
        </Panel>
      </div>
    </div>
  );
}

function ParameterGrid({
  parameters,
  draft,
  roots,
  onDraftChange,
}: {
  parameters: StudioParameter[];
  draft: StudioDraft;
  roots: string[];
  onDraftChange: (draft: StudioDraft) => void;
}) {
  if (parameters.length === 0) {
    return <p className="text-[11px] text-zinc-600">No configurable parameters in this stage.</p>;
  }
  return (
    <div className="grid grid-cols-1 gap-x-4 gap-y-2.5 sm:grid-cols-2">
      {parameters.map((parameter) => {
        const current = getDraftValue(draft, parameter) ?? parameter.default;
        const update = (value: string | number | boolean | string[]) =>
          onDraftChange(setDraftValue(draft, parameter, value));

        if (parameter.type === "root_multiselect") {
          const selected = Array.isArray(current) ? current : rootsOf(draft);
          return (
            <div key={parameter.path} className="sm:col-span-2">
              <div className="mb-1 flex items-baseline justify-between gap-2">
                <span className="text-[11px] font-medium text-zinc-400">{parameter.label}</span>
                <span className="font-mono text-[10.5px] text-zinc-500">{selected.length} selected</span>
              </div>
              <RootPills roots={roots} selected={selected} onChange={update} />
              <p className="mt-1 text-[10px] leading-4 text-zinc-600">{parameter.description}</p>
            </div>
          );
        }
        if (parameter.type === "multiselect" && parameter.options && parameter.options.length > 0) {
          const rawSelected = Array.isArray(current)
            ? current
            : typeof current === "string" && current.trim().startsWith("{")
              ? Object.keys(JSON.parse(current) as Record<string, unknown>)
              : typeof current === "string" && current.length > 0
                ? current.split(",")
                : [];
          const selected = rawSelected.map((entry) => String(entry));
          const choices = parameter.options.map((option) => ({
            value: String(option.value),
            label: option.label,
          }));
          return (
            <div key={parameter.path} className="sm:col-span-2">
              <div className="mb-1 flex items-baseline justify-between gap-2">
                <span className="text-[11px] font-medium text-zinc-400">{parameter.label}</span>
                <span className="font-mono text-[10.5px] text-zinc-500">
                  {selected.length} selected
                  {parameter.min_items ? ` · min ${parameter.min_items}` : ""}
                  {parameter.max_items ? ` · max ${parameter.max_items}` : ""}
                </span>
              </div>
              <div className="flex flex-wrap gap-1">
                {choices.map((choice) => {
                  const active = selected.includes(choice.value);
                  return (
                    <button
                      key={choice.value}
                      type="button"
                      aria-pressed={active}
                      onClick={() =>
                        update(
                          active
                            ? selected.filter((entry) => entry !== choice.value)
                            : [...selected, choice.value],
                        )
                      }
                      className={cn(
                        "focus-ring h-6 rounded-md border px-2 font-mono text-[11px] font-medium transition-colors",
                        active
                          ? "border-accent/40 bg-accent-soft text-blue-300"
                          : "border-edge bg-canvas text-zinc-500 hover:border-zinc-600 hover:text-zinc-300",
                      )}
                    >
                      {choice.label}
                    </button>
                  );
                })}
              </div>
              <p className="mt-1 line-clamp-2 text-[10px] leading-4 text-zinc-600">{parameter.description}</p>
            </div>
          );
        }
        if (parameter.type === "select") {
          const options = parameter.options ?? [];
          return (
            <div key={parameter.path}>
              <div className="mb-1 text-[11px] font-medium text-zinc-400">{parameter.label}</div>
              <Select
                value={String(current)}
                onChange={(value) => update(value)}
                options={options.map((option) => ({ value: String(option.value), label: option.label }))}
              />
              <p className="mt-1 line-clamp-2 text-[10px] leading-4 text-zinc-600">{parameter.description}</p>
            </div>
          );
        }
        if (parameter.type === "bool") {
          return (
            <div key={parameter.path} className="sm:col-span-2">
              <ToggleRow
                label={parameter.label}
                description={parameter.description}
                checked={Boolean(current)}
                onChange={update}
              />
            </div>
          );
        }
        return (
          <div key={parameter.path}>
            <RangeField parameter={parameter} value={Number(current)} onChange={update} />
          </div>
        );
      })}
    </div>
  );
}
