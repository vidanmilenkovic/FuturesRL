"use client";

import { useEffect, useMemo, useState } from "react";

import { WaterfallChart } from "@/components/charts/svg-charts";
import { useTelemetry } from "@/components/providers/telemetry-provider";
import { Badge } from "@/components/ui/badge";
import { Segmented, Skeleton } from "@/components/ui/controls";
import { Panel } from "@/components/ui/panel";
import { getPipeline } from "@/lib/api";
import { cn } from "@/lib/cn";
import {
  formatDateTime,
  formatNumber,
  formatSignedNumber,
  formatWeight,
  formatSignedCurrency,
} from "@/lib/format";
import { useAsyncData } from "@/lib/use-async-data";
import type { PipelineInstrumentRow, PipelineWaterfallResponse } from "@/lib/types";

type ArtifactStage = "base" | "timing" | "risk";

const STAGE_TONE = {
  universe: "neutral",
  base: "info",
  timing: "violet",
  risk: "up",
  targets: "warn",
} as const;

/**
 * View 2 — FinRL-X Pipeline Inspector.
 *
 * Renders the staged weight transformation
 * `S_t → w_base → w_timing → w_risk → N_i,t` as a waterfall, a per-instrument
 * weight-evolution table and the diagnostics emitted by every pipeline stage.
 */
export function PipelineView() {
  const pipeline = useAsyncData(getPipeline);
  const { latest } = useTelemetry();
  const [data, setData] = useState<PipelineWaterfallResponse | null>(null);
  const [focus, setFocus] = useState<ArtifactStage>("risk");

  useEffect(() => {
    if (pipeline.data) setData(pipeline.data);
  }, [pipeline.data]);

  useEffect(() => {
    const frame = latest.pipeline;
    if (frame && "pipeline" in frame.payload) {
      setData(frame.payload.pipeline);
    }
  }, [latest.pipeline]);

  const selected = useMemo(
    () => (data ? data.instruments.filter((row) => row.selected) : []),
    [data],
  );

  if (pipeline.loading && !data) {
    return (
      <div className="space-y-3">
        <Skeleton className="h-24" />
        <div className="grid gap-3 xl:grid-cols-3">
          <Skeleton className="h-[300px] xl:col-span-2" />
          <Skeleton className="h-[300px]" />
        </div>
        <Skeleton className="h-[360px]" />
      </div>
    );
  }

  if (!data) {
    return (
      <Panel title="pipeline inspector">
        <p className="py-10 text-center font-mono text-xs uppercase tracking-[0.12em] text-signal-down">
          pipeline trace unavailable
        </p>
      </Panel>
    );
  }

  const artifactValue = (row: PipelineInstrumentRow, stage: ArtifactStage): number => {
    if (stage === "base") return row.weight_base;
    if (stage === "timing") return row.weight_timing;
    return row.weight_risk;
  };

  const maxAbsWeight = Math.max(
    ...selected.map((row) => Math.max(Math.abs(row.weight_base), Math.abs(row.weight_timing), Math.abs(row.weight_risk))),
    0.0001,
  );

  return (
    <div className="space-y-3">
      {/* Header strip */}
      <div className="flex flex-wrap items-center gap-2 rounded-md border border-terminal-600/60 bg-terminal-850/70 px-3 py-2">
        <Badge tone="info">{data.cycle_id}</Badge>
        <span className="font-mono text-[10px] uppercase tracking-[0.1em] text-terminal-300">
          w_t = R_t(T_t(A_t(S_t(X≤t))))
        </span>
        <span className="hidden font-mono text-[10px] uppercase tracking-[0.1em] text-terminal-500 sm:inline">
          {formatDateTime(data.timestamp)} · {data.elapsed_ms.toFixed(1)} ms · {data.universe.length} roots
        </span>
        <div className="ml-auto flex items-center gap-2">
          <span className="font-mono text-[10px] uppercase tracking-[0.1em] text-terminal-400">instrument artifact</span>
          <Segmented
            size="sm"
            ariaLabel="Weight artifact to inspect"
            value={focus}
            onChange={setFocus}
            options={[
              { value: "base", label: "w_base", hint: "Allocator output before overlays" },
              { value: "timing", label: "w_timing", hint: "After TSMOM/KAMA/carry overlay" },
              { value: "risk", label: "w_risk", hint: "After vol target + LLM risk projection" },
            ]}
          />
        </div>
      </div>

      {/* Stage cards */}
      <div className="grid grid-cols-2 gap-2 md:grid-cols-3 xl:grid-cols-5">
        {data.stages.map((stage) => (
          <div
            key={stage.key}
            className="relative overflow-hidden rounded-md border border-terminal-600/60 bg-terminal-850/70 px-2.5 py-2"
          >
            <div className="flex items-center justify-between gap-2">
              <p className="truncate font-mono text-[10px] font-semibold uppercase tracking-[0.12em] text-terminal-200">
                {stage.label}
              </p>
              <Badge tone={STAGE_TONE[stage.key]}>{stage.key === "targets" ? "N" : stage.key.slice(0, 3)}</Badge>
            </div>
            <p className="mt-1 font-mono text-lg font-semibold tabular-nums text-terminal-100">
              {stage.gross.toFixed(3)}
              <span className="ml-1 text-[10px] font-normal text-terminal-400">gross</span>
            </p>
            <div className="flex items-center justify-between font-mono text-[9px] uppercase tracking-[0.08em] text-terminal-500">
              <span>net {formatSignedNumber(stage.net, 3)}</span>
              <span>{stage.instrument_count} {stage.key === "targets" ? "lots" : "roots"}</span>
            </div>
            <p className="mt-1 truncate font-mono text-[9px] text-terminal-500" title={stage.note}>
              {stage.formula}
            </p>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 gap-3 xl:grid-cols-3">
        <Panel
          title="weight transformation waterfall"
          subtitle="gross exposure by stage · dot = net exposure"
          className="xl:col-span-2"
          actions={<Badge tone="violet">R(T(A(S)))</Badge>}
        >
          <WaterfallChart
            stages={data.stages.map((stage) => ({
              key: stage.key,
              label: stage.label,
              gross: stage.gross,
              net: stage.net,
              note: stage.formula,
            }))}
          />
          <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1 border-t border-terminal-600/40 pt-2 font-mono text-[9px] uppercase tracking-[0.1em] text-terminal-400">
            <span className="flex items-center gap-1"><span className="h-2 w-2 rounded-sm bg-signal-info/40 ring-1 ring-signal-info" /> gross Σ|w|</span>
            <span className="flex items-center gap-1"><span className="h-2 w-2 rounded-full bg-signal-up" /> net long exposure</span>
            <span className="flex items-center gap-1"><span className="h-2 w-2 rounded-full bg-signal-down" /> net short exposure</span>
            <span className="ml-auto text-terminal-500">Δ vs previous stage shown per row</span>
          </div>
        </Panel>

        <Panel title="stage diagnostics" subtitle="guardrails & filters" actions={<Badge tone="neutral">{data.diagnostics.length}</Badge>}>
          <div className="divide-y divide-terminal-600/30">
            {data.diagnostics.map((diagnostic) => (
              <div key={diagnostic.stage} className="py-2">
                <div className="flex items-center justify-between gap-2">
                  <p className="font-mono text-[10px] uppercase tracking-[0.1em] text-terminal-200">{diagnostic.stage}</p>
                  <Badge tone={diagnostic.status === "ok" ? "up" : diagnostic.status}>{diagnostic.value}</Badge>
                </div>
                <p className="mt-0.5 font-mono text-[10px] leading-relaxed text-terminal-400">{diagnostic.detail}</p>
              </div>
            ))}
          </div>
        </Panel>
      </div>

      {/* Instrument table */}
      <Panel
        title="per-instrument transformation"
        subtitle={`S_t → w_base → w_timing → w_risk → N_i,t · highlighting ${focus}`}
        contentClassName="px-3 pt-2 pb-0"
      >
        <div className="-mx-3 overflow-x-auto">
          <table className="data-table min-w-[1080px]">
            <thead>
              <tr>
                <th>root</th>
                <th>class</th>
                <th className="num">S_t</th>
                <th className="num">w_base</th>
                <th className="num">w_timing</th>
                <th className="num">w_risk</th>
                <th className="num">Δ timing</th>
                <th className="num">Δ risk</th>
                <th>weight evolution</th>
                <th className="num">target N</th>
                <th className="num">notional</th>
                <th>adjustment driver</th>
              </tr>
            </thead>
            <tbody>
              {data.instruments.map((row) => (
                <tr key={row.instrument_id} className={row.selected ? "" : "opacity-45"}>
                  <td className="font-semibold text-terminal-100">{row.instrument_id}</td>
                  <td>
                    <Badge tone="neutral">{row.asset_class.replace("_index", "")}</Badge>
                  </td>
                  <td className="num">
                    {row.selected ? <Badge tone="up" dot>in</Badge> : <Badge tone="neutral">filtered</Badge>}
                  </td>
                  {(["base", "timing", "risk"] as const).map((stage) => (
                    <td
                      key={stage}
                      className={cn(
                        "num",
                        focus === stage ? "bg-signal-info/[0.06] font-semibold text-terminal-100" : "text-terminal-300",
                      )}
                    >
                      {formatWeight(artifactValue(row, stage))}
                    </td>
                  ))}
                  <td className={cn("num", row.delta_timing >= 0 ? "text-signal-up" : "text-signal-down")}>
                    {formatSignedNumber(row.delta_timing, 4)}
                  </td>
                  <td className={cn("num", row.delta_risk >= 0 ? "text-signal-up" : "text-signal-down")}>
                    {formatSignedNumber(row.delta_risk, 4)}
                  </td>
                  <td>
                    <div className="flex h-4 w-40 items-center gap-0.5" aria-hidden>
                      {(["base", "timing", "risk"] as const).map((stage) => {
                        const value = artifactValue(row, stage);
                        const width = Math.max(2, (Math.abs(value) / maxAbsWeight) * 100);
                        return (
                          <span key={stage} className="flex h-3 flex-1 items-center rounded-sm bg-terminal-700/60">
                            <span
                              className={cn(
                                "h-2 rounded-sm",
                                value >= 0 ? "bg-signal-up/80" : "bg-signal-down/80",
                                focus === stage ? "opacity-100" : "opacity-45",
                              )}
                              style={{ width: `${width}%` }}
                            />
                          </span>
                        );
                      })}
                    </div>
                  </td>
                  <td className="num font-semibold text-terminal-100">
                    {row.target_contracts === null ? "—" : row.target_contracts > 0 ? `+${row.target_contracts}` : row.target_contracts}
                  </td>
                  <td className="num text-terminal-300">
                    {row.notional === null ? "—" : formatSignedCurrency(row.notional, true)}
                  </td>
                  <td className="max-w-[280px] truncate text-[10px] text-terminal-400" title={row.reason}>
                    {row.reason}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      {/* Universe + legend */}
      <div className="grid grid-cols-1 gap-3 lg:grid-cols-3">
        <Panel title="universe selector S_t" subtitle="liquidity · open interest · regime">
          <div className="flex flex-wrap gap-1.5">
            {data.instruments.map((row) => (
              <span
                key={row.instrument_id}
                className={cn(
                  "inline-flex items-center gap-1.5 rounded-sm border px-2 py-1 font-mono text-[10px] uppercase tracking-[0.08em]",
                  row.selected
                    ? "border-signal-up/40 bg-signal-up/10 text-signal-up"
                    : "border-terminal-600/60 bg-terminal-800/50 text-terminal-500 line-through",
                )}
                title={row.reason}
              >
                {row.instrument_id}
                <span className={row.selected ? "text-terminal-200" : "text-terminal-600"}>
                  {row.selected ? formatWeight(row.weight_base) : "0.000"}
                </span>
              </span>
            ))}
          </div>
          <p className="mt-3 border-t border-terminal-600/40 pt-2 font-mono text-[10px] leading-relaxed text-terminal-400">
            Candidates are filtered on 20d ADV, open interest and term-structure regime before the allocator
            runs. Filtered roots are kept in the waterfall for auditability but are forced to zero notional.
          </p>
        </Panel>

        <Panel title="stage contract" subtitle="every stage is weight-preserving">
          <ol className="space-y-2">
            {[
              ["S_t", "UniverseSelection.select(state) → instruments"],
              ["A_t", "PortfolioAllocator.allocate(candidates, state) → w_base"],
              ["T_t", "TimingOverlay.adjust_timing(w_base, state) → w_timing"],
              ["R_t", "RiskOverlay.apply_risk(w_timing, state) → w_risk"],
              ["Translator", "N_i,t = w_t·E_t / (P_i,t·M_i) → discrete contracts"],
            ].map(([name, detail]) => (
              <li key={name} className="flex gap-2">
                <span className="shrink-0 font-mono text-[10px] font-semibold uppercase tracking-[0.1em] text-signal-info">
                  {name}
                </span>
                <span className="font-mono text-[10px] leading-relaxed text-terminal-400">{detail}</span>
              </li>
            ))}
          </ol>
        </Panel>

        <Panel title="gross leverage budget" subtitle="Σ|w_i| vs L_max">
          <div className="space-y-2">
            {data.stages.slice(1).map((stage) => (
              <div key={stage.key}>
                <div className="mb-1 flex justify-between font-mono text-[10px] uppercase tracking-[0.08em]">
                  <span className="text-terminal-300">{stage.key}</span>
                  <span className="tabular-nums text-terminal-200">{stage.gross.toFixed(3)}×</span>
                </div>
                <div className="h-1.5 w-full overflow-hidden rounded-full bg-terminal-700">
                  <div
                    className="h-full rounded-full bg-signal-info/70"
                    style={{ width: `${Math.min(100, (stage.gross / 3) * 100)}%` }}
                  />
                </div>
              </div>
            ))}
            <p className="border-t border-terminal-600/40 pt-2 font-mono text-[10px] text-terminal-400">
              L_max = 3.0× · margin ceiling η_max = 60% · translator rounds to{" "}
              {formatNumber(data.stages[data.stages.length - 1]?.instrument_count ?? 0, 0)} contracts.
            </p>
          </div>
        </Panel>
      </div>
    </div>
  );
}
