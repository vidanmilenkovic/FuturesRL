"use client";

import { useEffect, useMemo, type CSSProperties } from "react";

import { StageGroupedBars, type StageBar } from "@/components/charts/plots";
import { RebalanceBar } from "@/components/rebalance/rebalance-bar";
import { RootScopeFilterBar, useRootScopeFilter } from "@/components/filters/root-scope-filter";
import {
  Badge,
  EmptyState,
  Panel,
  Skeleton,
  type Tone,
} from "@/components/ui/primitives";
import { getPipeline, getTermStructure } from "@/lib/api";
import { cn } from "@/lib/cn";
import {
  formatCurrency,
  formatNumber,
  formatWeight,
} from "@/lib/format";
import type { PipelineInstrumentRow, PipelineWaterfallResponse, TermRegime } from "@/lib/types";
import { useAsyncData } from "@/lib/use-async-data";

const MICRO_ROOTS = new Set(["MES", "MNQ", "MCL", "MGC"]);

const STAGE_LABELS: Record<string, { label: string; formula: string }> = {
  universe: { label: "S_t · Universe", formula: "selector(X≤t)" },
  base: { label: "A_t · Allocation", formula: "allocate(candidates)" },
  timing: { label: "T_t · Timing", formula: "adjust_timing(w_base)" },
  risk: { label: "R_t · Risk", formula: "apply_risk(w_timing)" },
  targets: { label: "N_i,t · Contracts", formula: "w_t·E_t / (P·M)" },
};

const REGIME_TONE: Record<TermRegime, Tone> = {
  contango: "warn",
  backwardation: "gain",
  flat: "neutral",
};

const REGIME_LABEL: Record<TermRegime, string> = {
  contango: "Contango",
  backwardation: "Backwardation",
  flat: "Flat",
};

/** Compact regime tags sized for the dense weight matrix. */
const MATRIX_REGIME_LABEL: Record<TermRegime, string> = {
  contango: "Contango",
  backwardation: "Backward.",
  flat: "Flat",
};

function signedQuantity(value: number): string {
  return `${value >= 0 ? "+" : "−"}${Math.abs(value)}`;
}

/**
 * Compact standard→micro routing summary, e.g. `ES → +14 ES, +18 MES`.
 * Older backends do not report `micro_routing`; derive target micro legs from
 * the instrument rows instead of inventing pairs.
 */
function formatMicroRouting(pipeline: PipelineWaterfallResponse): string {
  if (Array.isArray(pipeline.micro_routing)) {
    if (pipeline.micro_routing.length === 0) return "Idle · no standard → micro routing";
    return pipeline.micro_routing
      .map(
        (route) =>
          `${route.standard_root} → ${signedQuantity(route.standard_quantity)} ${route.standard_root}, ${signedQuantity(route.micro_quantity)} ${route.micro_root}`,
      )
      .join(" · ");
  }
  const microLegs = pipeline.instruments
    .filter((row) => row.target_contracts !== null && MICRO_ROOTS.has(row.instrument_id))
    .map((row) => `${signedQuantity(row.target_contracts ?? 0)} ${row.instrument_id}`);
  return microLegs.length > 0 ? `Active · ${microLegs.join(", ")}` : "n/a · not reported by API";
}

function formatTrackingError(pipeline: PipelineWaterfallResponse): {
  value: string;
  tone: string;
  title: string;
} {
  const reported = pipeline.tracking_error_pct;
  if (typeof reported === "number" && Number.isFinite(reported)) {
    return {
      value: `${reported.toFixed(2)}%`,
      tone: reported > 5 ? "text-warn" : "text-zinc-200",
      title: "Realised tracking error of the executed book versus the approved target weights",
    };
  }
  return {
    value: "n/a · not reported by API",
    tone: "text-zinc-500",
    title: "This backend build does not report realised tracking error for the pipeline",
  };
}

function heatStyle(value: number, maxAbs: number, muted = false): CSSProperties {
  if (Math.abs(value) < 1e-9) return {};
  const alpha = Math.max(0.06, Math.min(0.32, (Math.abs(value) / Math.max(maxAbs, 1e-9)) * 0.3));
  if (muted) return {};
  return {
    backgroundColor:
      value >= 0 ? `rgba(16, 185, 129, ${alpha.toFixed(3)})` : `rgba(244, 63, 94, ${alpha.toFixed(3)})`,
  };
}

function parseOverlays(pipeline: PipelineWaterfallResponse): string[] {
  const diagnostic = pipeline.diagnostics.find((item) => item.stage === "Risk overlays");
  if (!diagnostic) return [];
  const match = /applied:\s*(.+)$/i.exec(diagnostic.detail);
  if (!match || !match[1]) return [];
  return match[1]
    .split(",")
    .map((name) => name.trim())
    .filter(Boolean);
}

function parseTranslator(pipeline: PipelineWaterfallResponse): string {
  const diagnostic = pipeline.diagnostics.find((item) => item.stage === "Contract translator");
  return diagnostic?.detail ?? "not reported";
}

function parseMargin(pipeline: PipelineWaterfallResponse): string {
  const diagnostic = pipeline.diagnostics.find((item) => item.stage === "Margin feasibility");
  return diagnostic?.value ?? "—";
}

/** View 2 — FinRL-X pipeline & weights inspector. */
export function PipelineView() {
  const filter = useRootScopeFilter();
  const pipeline = useAsyncData(() => getPipeline(filter.query));
  const terms = useAsyncData(() => getTermStructure(filter.query));

  useEffect(() => {
    if (!filter.hydrated) return;
    pipeline.reload();
    terms.reload();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filter.key, filter.hydrated, pipeline.reload, terms.reload]);

  const data = pipeline.data;
  const regimes = useMemo(() => {
    const map = new Map<string, TermRegime>();
    for (const curve of terms.data?.curves ?? []) map.set(curve.root, curve.regime);
    return map;
  }, [terms.data]);

  const stages: StageBar[] = useMemo(() => {
    if (!data) return [];
    return data.stages.map((stage) => {
      const meta = STAGE_LABELS[stage.key];
      return {
        key: stage.key,
        label: meta?.label ?? stage.label,
        formula: stage.formula,
        gross: stage.gross,
        net: stage.net,
        count: stage.instrument_count,
        countLabel: stage.key === "targets" ? "lots" : "roots",
      };
    });
  }, [data]);

  const maxAbs = useMemo(() => {
    const rows = data?.instruments ?? [];
    return Math.max(
      ...rows.map((row) =>
        Math.max(
          Math.abs(row.weight_base),
          Math.abs(row.weight_timing),
          Math.abs(row.weight_risk),
          Math.abs(row.weight_final),
        ),
      ),
      0.0001,
    );
  }, [data]);

  if (pipeline.loading && !data) {
    return (
      <div className="grid min-h-full grid-rows-[minmax(0,1fr)] grid-cols-12 gap-3">
        <Skeleton className="col-span-7 h-72" />
        <Skeleton className="col-span-5 h-72" />
        <Skeleton className="col-span-7 h-56" />
        <Skeleton className="col-span-5 h-56" />
      </div>
    );
  }

  if (!data) {
    return (
      <Panel title="Weights">
        <EmptyState
          title="Pipeline trace unavailable"
          description="The API did not return a FinRL-X rebalance trace and the deterministic fixture fallback could not be built."
        />
      </Panel>
    );
  }

  const overlays = parseOverlays(data);
  const translator = parseTranslator(data);
  const tracking = formatTrackingError(data);
  const selectedCount = data.instruments.filter((row) => row.selected).length;

  return (
    <div className="grid min-h-full grid-rows-[auto_auto_minmax(0,1fr)] grid-cols-12 gap-3">
      <RebalanceBar
        className="col-span-12"
        onExecuted={() => {
          pipeline.reload();
          terms.reload();
        }}
      />
      <RootScopeFilterBar
        scope={filter.state.scope}
        sector={filter.state.sector}
        roots={filter.state.roots}
        onScopeChange={filter.setScope}
        onSectorChange={filter.setSector}
        onToggleRoot={filter.toggleRoot}
        onClearRoots={filter.clearRoots}
        onRefresh={() => {
          pipeline.reload();
          terms.reload();
        }}
        refreshing={pipeline.loading}
        selectedCount={data.instruments.length}
        dataOrigin={data.data_origin ?? null}
      />
      {/* ------------------------------------------------------- left column */}
      <div className="col-span-12 flex min-h-0 flex-col gap-3 xl:col-span-7">
        <Panel
          title="Stage exposure · long / short / net"
          actions={
            <div className="flex items-center gap-2">
              <Badge tone={data.data_source === "engine" ? "gain" : "warn"}>
                {data.data_source === "engine" ? "Engine" : "Fixture"}
              </Badge>
              <span className="font-mono text-[11px] text-zinc-600">{data.cycle_id}</span>
            </div>
          }
        >
          <StageGroupedBars stages={stages} />
          <div className="mt-3 border-t border-hairline pt-2">
            <div className="grid grid-cols-4 gap-3">
              <TraceStat label="Gross (targets)" value={`${(data.stages.at(-1)?.gross ?? 0).toFixed(3)}×`} />
              <TraceStat label="Net (targets)" value={`${(data.stages.at(-1)?.net ?? 0).toFixed(3)}×`} />
              <TraceStat label="Universe" value={`${selectedCount} / ${data.instruments.length}`} />
              <TraceStat label="Elapsed" value={`${formatNumber(data.elapsed_ms, 2)} ms`} />
            </div>
          </div>
        </Panel>

        <Panel title="Execution trace" className="min-h-0 flex-1">
          <div className="grid grid-cols-2 gap-x-6 gap-y-2.5">
            <TraceRow label="Latency" value={`${formatNumber(data.elapsed_ms, 2)} ms`} />
            <TraceRow label="Modules" value={`5 stages · ${overlays.length} risk overlays`} />
            <TraceRow
              label="Risk overlays"
              value={overlays.length > 0 ? overlays.join(", ") : "none applied"}
              tone={overlays.length > 0 ? "text-zinc-200" : "text-warn"}
            />
            <TraceRow
              label="Tracking error"
              value={tracking.value}
              tone={tracking.tone}
              title={tracking.title}
            />
            <TraceRow
              label="Micro routing"
              value={formatMicroRouting(data)}
              tone={
                data.micro_routing && data.micro_routing.length > 0
                  ? "text-zinc-200"
                  : "text-zinc-500"
              }
              title="Standard → micro contract routing applied by the deterministic contract translator"
            />
            <TraceRow label="Margin feasibility" value={parseMargin(data)} />
            <TraceRow label="Translator" value={translator} />
          </div>

          <div className="mt-3 border-t border-hairline pt-2">
            <div className="metric-label mb-1.5">Diagnostics</div>
            <div className="space-y-1.5">
              {data.diagnostics.map((diagnostic) => (
                <div key={`${diagnostic.stage}-${diagnostic.value}`} className="flex items-start gap-2">
                  <Badge
                    tone={
                      diagnostic.status === "critical"
                        ? "loss"
                        : diagnostic.status === "warn"
                          ? "warn"
                          : diagnostic.status === "ok"
                            ? "gain"
                            : "neutral"
                    }
                  >
                    {diagnostic.stage}
                  </Badge>
                  <span className="min-w-0 flex-1 text-[11.5px] leading-5 text-zinc-500" title={diagnostic.detail}>
                    {diagnostic.value}
                    <span className="text-zinc-700"> · </span>
                    {diagnostic.detail}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </Panel>
      </div>

      {/* ------------------------------------------------------ right column */}
      <Panel
        className="col-span-12 min-h-0 xl:col-span-5"
        title="Weight matrix"
        actions={
          <div className="flex items-center gap-2 text-[11px] text-zinc-600">
            <span className="flex items-center gap-1">
              <span className="h-2 w-2 rounded-sm bg-gain/50" /> long
            </span>
            <span className="flex items-center gap-1">
              <span className="h-2 w-2 rounded-sm bg-loss/50" /> short
            </span>
          </div>
        }
        bodyClassName="min-h-0 p-0"
        flush
      >
        <div
          className="min-h-0 flex-1 overflow-y-auto overflow-x-hidden"
          data-testid="weight-matrix-scroll"
        >
          <table
            className="data-table w-full table-fixed text-[10.5px] [&_th]:px-1.5 [&_th]:text-[8.5px] [&_th]:tracking-[0.01em] [&_td]:h-9 [&_td]:px-1.5"
            data-testid="weight-matrix"
          >
            <colgroup>
              <col style={{ width: "7%" }} />
              <col style={{ width: "12%" }} />
              <col style={{ width: "9%" }} />
              <col style={{ width: "9%" }} />
              <col style={{ width: "9%" }} />
              <col style={{ width: "9%" }} />
              <col style={{ width: "9%" }} />
              <col style={{ width: "7%" }} />
              <col style={{ width: "10%" }} />
              <col style={{ width: "19%" }} />
            </colgroup>
            <thead>
              <tr>
                <th>Root</th>
                <th>Regime</th>
                <th className="text-right">w_base</th>
                <th className="text-right">w_tim</th>
                <th className="text-right">Tim Δ</th>
                <th className="text-right">w_risk</th>
                <th className="text-right">Risk Δ</th>
                <th className="text-right">N_i,t</th>
                <th className="text-right">Notional</th>
                <th>Driver</th>
              </tr>
            </thead>
            <tbody>
              {data.instruments.map((row) => {
                const regime = regimes.get(row.instrument_id);
                return (
                  <MatrixRow
                    key={row.instrument_id}
                    row={row}
                    regime={regime}
                    maxAbs={maxAbs}
                  />
                );
              })}
            </tbody>
          </table>
        </div>
        <div className="flex items-center justify-between border-t border-hairline px-3 py-2 text-[11px] text-zinc-600">
          <span>
            {selectedCount} selected · {data.instruments.length - selectedCount} filtered by S_t
          </span>
          <span className="font-mono">{data.cycle_id}</span>
        </div>
      </Panel>
    </div>
  );
}

function TraceStat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="metric-label">{label}</div>
      <div className="font-mono text-[12px] tabular-nums text-zinc-200">{value}</div>
    </div>
  );
}

function TraceRow({
  label,
  value,
  tone = "text-zinc-200",
  title,
}: {
  label: string;
  value: string;
  tone?: string;
  title?: string;
}) {
  return (
    <div className="min-w-0">
      <div className="metric-label">{label}</div>
      <div className={`truncate text-[12px] ${tone}`} title={title ?? value}>
        {value}
      </div>
    </div>
  );
}

function MatrixRow({
  row,
  regime,
  maxAbs,
}: {
  row: PipelineInstrumentRow;
  regime: TermRegime | undefined;
  maxAbs: number;
}) {
  const quantity = row.target_contracts;
  return (
    <tr className={row.selected ? "" : "opacity-50"}>
      <td className="font-medium text-zinc-100">
        <span className="flex items-center gap-1.5">
          {row.selected ? <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-accent" /> : null}
          <span className="truncate">{row.instrument_id}</span>
        </span>
      </td>
      <td>
        {regime ? (
          <Badge tone={REGIME_TONE[regime]} className="px-1 text-[9.5px]" title={REGIME_LABEL[regime]}>
            {MATRIX_REGIME_LABEL[regime]}
          </Badge>
        ) : (
          <span className="text-[10px] text-zinc-600">—</span>
        )}
      </td>
      <td className="num text-right" style={heatStyle(row.weight_base, maxAbs)}>
        {formatWeight(row.weight_base)}
      </td>
      <td className="num text-right" style={heatStyle(row.weight_timing, maxAbs)}>
        {formatWeight(row.weight_timing)}
      </td>
      <td
        className={cn("num text-right", row.delta_timing >= 0 ? "text-gain/90" : "text-loss/90")}
      >
        {row.delta_timing >= 0 ? "+" : "−"}
        {Math.abs(row.delta_timing).toFixed(3)}
      </td>
      <td className="num text-right" style={heatStyle(row.weight_risk, maxAbs)}>
        <span className="block">{formatWeight(row.weight_risk)}</span>
        {row.weight_actual !== null && row.weight_actual !== undefined ? (
          <span
            className="block text-[9px] leading-3 text-zinc-600"
            title={`Executed weight (broker-confirmed): ${row.weight_actual >= 0 ? "+" : "−"}${Math.abs(row.weight_actual).toFixed(4)}`}
            data-testid="weight-actual"
          >
            {row.weight_actual >= 0 ? "+" : "−"}
            {Math.abs(row.weight_actual).toFixed(3)}
          </span>
        ) : null}
      </td>
      <td className={cn("num text-right", row.delta_risk >= 0 ? "text-gain/90" : "text-loss/90")}>
        {row.delta_risk >= 0 ? "+" : "−"}
        {Math.abs(row.delta_risk).toFixed(3)}
      </td>
      <td className={cn("num text-right font-medium", quantity === null ? "text-zinc-600" : "text-zinc-100")}>
        {quantity === null ? "—" : quantity > 0 ? `+${quantity}` : quantity}
      </td>
      <td className="num text-right text-zinc-400">
        {row.notional === null ? "—" : formatCurrency(row.notional, true)}
      </td>
      <td className="text-[10.5px] text-zinc-500" data-testid="matrix-driver">
        <span className="block truncate" title={row.reason}>
          {row.reason}
        </span>
      </td>
    </tr>
  );
}
