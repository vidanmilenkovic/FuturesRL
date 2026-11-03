"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import { Tabs } from "@/components/ui/primitives";
import { getStudioCatalog, listStrategies } from "@/lib/api";
import { draftFromPreset, draftFromStrategy } from "@/lib/studio";
import type { StudioDraft } from "@/lib/types";
import { useAsyncData } from "@/lib/use-async-data";

import { DrlPanel } from "./drl-panel";
import { ManualBuilder } from "./manual-builder";
import { OptimizerPanel } from "./optimizer-panel";

type StudioMode = "builder" | "optimizer" | "drl";

const MODES: { value: StudioMode; label: string }[] = [
  { value: "builder", label: "Manual Builder" },
  { value: "optimizer", label: "Optuna Auto-Optimizer" },
  { value: "drl", label: "DRL Neural Trainer" },
];

export function StudioView() {
  const catalog = useAsyncData(getStudioCatalog);
  const strategies = useAsyncData(listStrategies);
  const [mode, setMode] = useState<StudioMode>("builder");
  const [draft, setDraft] = useState<StudioDraft | null>(null);
  const [notice, setNotice] = useState<{ text: string; tone: "accent" | "gain" | "warn" | "loss" | "neutral" } | null>(null);

  useEffect(() => {
    if (draft !== null) return;
    const presets = catalog.data?.presets ?? [];
    const first = presets[0];
    if (first) setDraft(draftFromPreset(first));
  }, [catalog.data, draft]);

  useEffect(() => {
    if (!notice) return;
    const timer = setTimeout(() => setNotice(null), 7_000);
    return () => clearTimeout(timer);
  }, [notice]);

  const notify = (text: string, tone: "accent" | "gain" | "warn" | "loss" | "neutral") =>
    setNotice({ text, tone });

  // Deep link: /studio?strategy=<id> loads a saved strategy into the Manual
  // Builder (used by the Apply Best Parameters success banner and backtests).
  const handledQueryId = useRef<string | null>(null);
  useEffect(() => {
    if (typeof window === "undefined" || !strategies.data) return;
    const requested = new URLSearchParams(window.location.search).get("strategy");
    if (!requested || handledQueryId.current === requested) return;
    const match = strategies.data.find((strategy) => strategy.id === requested);
    if (!match) return;
    handledQueryId.current = requested;
    setDraft(draftFromStrategy(match));
    setMode("builder");
    notify(`Loaded ${match.id} from the strategy link`, "accent");
  }, [strategies.data]);

  const catalogBadge = useMemo((): { label: string; tone: "gain" | "warn" } => {
    if (catalog.source === "live") return { label: "Catalog live", tone: "gain" };
    return { label: "Catalog fallback", tone: "warn" };
  }, [catalog.source]);

  if (!catalog.data || !draft) {
    return (
      <div className="grid min-h-full grid-cols-12 gap-3">
        <div className="panel col-span-12 flex h-11 items-center gap-2 px-3">
          <div className="h-2.5 w-40 animate-pulse rounded bg-surface" />
        </div>
        <div className="panel col-span-12 xl:col-span-8 h-72 animate-pulse" />
        <div className="panel col-span-12 xl:col-span-4 h-72 animate-pulse" />
      </div>
    );
  }

  const strategiesList = strategies.data ?? [];
  const activeView = mode === "builder" ? "Manual strategy builder" : mode === "optimizer" ? "Optuna auto-optimizer" : "DRL neural trainer";

  return (
    <div className="flex min-h-full flex-col gap-3">
      <section className="panel col-span-12 flex flex-wrap items-center justify-between gap-3 px-3 py-1">
        <div className="flex min-w-0 items-center gap-4">
          <Tabs<StudioMode>
            ariaLabel="Strategy Studio modes"
            options={MODES}
            value={mode}
            onChange={setMode}
            className="border-b-0"
          />
          <span className="hidden truncate text-[11px] text-zinc-600 md:inline">
            {mode === "builder"
              ? "Compose the FinRL-X pipeline and validate the contract before saving."
              : mode === "optimizer"
                ? "Walk-forward Optuna search over the strategy's tunable parameters."
                : "Train and register SB3 policies, then bind a checkpoint to a strategy."}
          </span>
        </div>
        <div className="flex items-center gap-2 text-[10.5px] text-zinc-600">
          {notice ? (
            <span
              className={
                notice.tone === "gain"
                  ? "truncate text-emerald-400"
                  : notice.tone === "warn"
                    ? "truncate text-amber-400"
                    : notice.tone === "loss"
                      ? "truncate text-rose-400"
                      : "truncate text-zinc-400"
              }
              title={notice.text}
            >
              {notice.text}
            </span>
          ) : (
            <span className="hidden truncate sm:inline">{activeView}</span>
          )}
          <span className={`inline-flex items-center gap-1 rounded px-1.5 py-0.5 ${catalogBadge.tone === "gain" ? "bg-gain-soft text-emerald-300" : "bg-warn-soft text-amber-300"}`}>
            {catalogBadge.label}
          </span>
          <span className="hidden font-mono sm:inline">
            {catalog.data.presets.length} presets · {catalog.data.parameters.length} params
          </span>
        </div>
      </section>

      {mode === "builder" ? (
        <ManualBuilder
          draft={draft}
          catalog={catalog.data}
          strategies={strategiesList}
          onDraftChange={setDraft}
          onStrategiesChanged={strategies.reload}
          onNotify={notify}
        />
      ) : mode === "optimizer" ? (
        <OptimizerPanel
          catalog={catalog.data}
          strategies={strategiesList}
          onStrategiesChanged={strategies.reload}
          onNotify={notify}
        />
      ) : (
        <DrlPanel
          strategies={strategiesList}
          onStrategiesChanged={strategies.reload}
          onNotify={notify}
        />
      )}
    </div>
  );
}
