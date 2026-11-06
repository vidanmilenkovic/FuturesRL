"use client";

import { useEffect, useMemo, useState } from "react";

import {
  ForwardCurveChart,
  type CurvePoint,
  type ForwardCurveMode,
} from "@/components/charts/plots";
import { RootScopeFilterBar, useRootScopeFilter } from "@/components/filters/root-scope-filter";
import {
  Badge,
  EmptyState,
  Panel,
  Segmented,
  Skeleton,
  Tabs,
  type Tone,
} from "@/components/ui/primitives";
import { getTermStructure } from "@/lib/api";
import { cn } from "@/lib/cn";
import { formatBps, formatDate, formatDateTime, formatPct } from "@/lib/format";
import type { RollCalendarEntry, TermCurve, TermRegime, TermStructureResponse } from "@/lib/types";
import { useAsyncData } from "@/lib/use-async-data";

type RollTab = "schedule" | "bag";

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

function frontPrice(curve: TermCurve): number {
  return curve.contracts[0]?.price ?? 0;
}

function secondPrice(curve: TermCurve): number {
  return curve.contracts[1]?.price ?? 0;
}

function lastPrice(curve: TermCurve): number {
  return curve.contracts[curve.contracts.length - 1]?.price ?? 0;
}

function lastLabel(curve: TermCurve): string {
  return `F${Math.max(curve.contracts.length, 1)}`;
}

function curveSpread(curve: TermCurve): { absolute: number; pct: number } {
  const front = frontPrice(curve);
  const last = lastPrice(curve);
  const absolute = last - front;
  return { absolute, pct: front > 0 ? (absolute / front) * 100 : 0 };
}

/** View 4 — futures term structure & roll calendar. */
export function TermStructureView() {
  const filter = useRootScopeFilter();
  const terms = useAsyncData(() => getTermStructure(filter.query));
  const [data, setData] = useState<TermStructureResponse | null>(null);
  const [root, setRoot] = useState<string | null>(null);
  const [mode, setMode] = useState<ForwardCurveMode>("normalized_basis");
  const [rollTab, setRollTab] = useState<RollTab>("schedule");

  useEffect(() => {
    const loaded = terms.data;
    if (!loaded) return;
    setData(loaded);
    setRoot((current) =>
      current && loaded.curves.some((curve) => curve.root === current)
        ? current
        : (loaded.curves[0]?.root ?? null),
    );
  }, [terms.data]);

  const curves = useMemo(() => data?.curves ?? [], [data]);
  const selected = useMemo(
    () => curves.find((curve) => curve.root === root) ?? curves[0] ?? null,
    [curves, root],
  );

  const selectedRoll = useMemo(
    () => (data?.roll_calendar ?? []).find((roll) => roll.root === selected?.root) ?? null,
    [data, selected],
  );

  useEffect(() => {
    if (!filter.hydrated) return;
    terms.reload();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filter.key, filter.hydrated, terms.reload]);

  if (terms.loading && !data) {
    return (
      <div className="grid grid-cols-12 gap-3">
        <Skeleton className="col-span-12 h-24" />
        <Skeleton className="col-span-8 h-80" />
        <Skeleton className="col-span-4 h-80" />
        <Skeleton className="col-span-12 h-64" />
      </div>
    );
  }

  if (!data || !selected) {
    return (
      <Panel title="Term structure">
        <EmptyState
          title="Term-structure data unavailable"
          description="The API did not return forward curves and the deterministic fixture fallback could not be built."
        />
      </Panel>
    );
  }

  const curvePoints: CurvePoint[] = selected.contracts.map((contract, index) => ({
    label: `F${index + 1}`,
    price: contract.price,
    days: contract.days_to_expiry,
    isFront: contract.is_front,
    rollYieldAnnualized: contract.roll_yield_annualized,
  }));
  const spread = curveSpread(selected);
  const spreadLabel = `${lastLabel(selected)}−F1`;
  const basis = secondPrice(selected) - frontPrice(selected);
  const basisPct = frontPrice(selected) > 0 ? basis / frontPrice(selected) : 0;
  const bagRolls = (data.roll_calendar ?? []).filter((roll) => roll.method.toLowerCase().includes("bag"));

  return (
    <div className="grid grid-cols-12 gap-3">
      <RootScopeFilterBar
        scope={filter.state.scope}
        sector={filter.state.sector}
        roots={filter.state.roots}
        onScopeChange={filter.setScope}
        onSectorChange={filter.setSector}
        onToggleRoot={filter.toggleRoot}
        onClearRoots={filter.clearRoots}
        onRefresh={() => terms.reload()}
        refreshing={terms.loading}
        selectedCount={data.curves.length}
        dataOrigin={data.data_origin ?? null}
      />
      {/* ------------------------------------------------------ root selector */}
      <Panel className="col-span-12" flush>
        <div className="no-scrollbar flex items-stretch gap-1 overflow-x-auto px-2 py-2">
          {curves.map((curve) => {
            const isSelected = curve.root === selected.root;
            const curveBasis =
              secondPrice(curve) > 0 && frontPrice(curve) > 0
                ? (secondPrice(curve) - frontPrice(curve)) / frontPrice(curve)
                : 0;
            return (
              <button
                key={curve.root}
                type="button"
                onClick={() => setRoot(curve.root)}
                aria-pressed={isSelected}
                className={cn(
                  "focus-ring flex min-w-[168px] flex-col gap-1 rounded-md border px-2.5 py-2 text-left transition-colors",
                  isSelected
                    ? "border-accent/50 bg-accent-soft"
                    : "border-transparent hover:border-edge hover:bg-surface",
                )}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="font-mono text-[12.5px] font-semibold text-zinc-100">{curve.root}</span>
                  <Badge tone={REGIME_TONE[curve.regime]}>{REGIME_LABEL[curve.regime]}</Badge>
                </div>
                <div className="flex items-baseline justify-between gap-2">
                  <span className="font-mono text-[11.5px] tabular-nums text-zinc-300">
                    {frontPrice(curve).toLocaleString("en-US", { maximumFractionDigits: 2 })}
                  </span>
                  <span
                    className={cn(
                      "font-mono text-[10.5px] tabular-nums",
                      curveBasis >= 0 ? "text-loss/90" : "text-gain/90",
                    )}
                  >
                    {formatBps(curveBasis, 1)}
                  </span>
                </div>
                <div className="flex items-baseline justify-between gap-2">
                  <span className="text-[10px] text-zinc-600">roll yield</span>
                  <span
                    className={cn(
                      "font-mono text-[10.5px] tabular-nums",
                      curve.roll_yield_annualized >= 0 ? "text-gain/90" : "text-loss/90",
                    )}
                  >
                    {formatPct(curve.roll_yield_annualized)}
                  </span>
                </div>
              </button>
            );
          })}
        </div>
      </Panel>

      {/* -------------------------------------------------------- forward curve */}
      <Panel
        className="col-span-12 xl:col-span-8"
        title={`${selected.root} · forward curve`}
        actions={
          <div className="flex flex-wrap items-center justify-end gap-2">
            <span
              data-testid="curve-spread-header"
              data-spread-abs={spread.absolute.toFixed(4)}
              data-spread-pct={spread.pct.toFixed(4)}
              data-spread-label={spreadLabel}
              title={`${spreadLabel} absolute ${spread.absolute >= 0 ? "+" : "−"}${Math.abs(
                spread.absolute,
              ).toFixed(2)} · ${spread.pct >= 0 ? "+" : "−"}${Math.abs(spread.pct).toFixed(2)}%`}
              className={cn(
                "font-mono text-[10.5px] tabular-nums",
                spread.absolute >= 0 ? "text-loss/90" : "text-gain/90",
              )}
            >
              {spreadLabel} {spread.absolute >= 0 ? "+" : "−"}
              {Math.abs(spread.absolute).toFixed(2)} ({spread.pct >= 0 ? "+" : "−"}
              {Math.abs(spread.pct).toFixed(2)}%)
            </span>
            <Badge tone={REGIME_TONE[selected.regime]}>{REGIME_LABEL[selected.regime]}</Badge>
            <Segmented
              ariaLabel="Curve mode"
              value={mode}
              onChange={setMode}
              options={[
                {
                  value: "normalized_basis",
                  label: "Normalized Basis %",
                  title: "Normalized Basis % vs. F1",
                },
                {
                  value: "roll_yield",
                  label: "Roll Yield %",
                  title: "Annualized Roll Yield %",
                },
                {
                  value: "raw_price",
                  label: "Raw Price $",
                  title: "Raw Contract Price ($)",
                },
              ]}
            />
          </div>
        }
      >
        <ForwardCurveChart points={curvePoints} mode={mode} ariaLabel={`${selected.root} forward curve`} />
        <div className="mt-2 grid grid-cols-2 gap-3 border-t border-hairline pt-2 sm:grid-cols-3 xl:grid-cols-6">
          <CurveStat label="Front (F1)" value={frontPrice(selected).toFixed(2)} />
          <CurveStat
            label="F2 − F1 basis"
            value={`${basis >= 0 ? "+" : "−"}${Math.abs(basis).toFixed(2)}`}
            tone={basis >= 0 ? "text-loss" : "text-gain"}
            sub={formatBps(basisPct, 1)}
          />
          <CurveStat
            label="Roll yield ann."
            value={formatPct(selected.roll_yield_annualized)}
            tone={selected.roll_yield_annualized >= 0 ? "text-gain" : "text-loss"}
          />
          <CurveStat
            label={`${spreadLabel} spread`}
            value={`${spread.absolute >= 0 ? "+" : "−"}${Math.abs(spread.absolute).toFixed(2)}`}
            sub={`${spread.pct >= 0 ? "+" : "−"}${Math.abs(spread.pct).toFixed(2)}%`}
            tone={spread.absolute >= 0 ? "text-loss" : "text-gain"}
          />
          <CurveStat label="Term slope" value={formatPct(selected.slope, 3)} />
          <CurveStat
            label="Next roll"
            value={selectedRoll ? `${selectedRoll.days_to_roll}d` : "—"}
            sub={selectedRoll ? formatDate(selectedRoll.roll_date) : "not scheduled"}
            tone={selectedRoll && selectedRoll.days_to_roll <= 7 ? "text-warn" : undefined}
          />
        </div>
        <p className="mt-1 text-[10.5px] text-zinc-600">
          {mode === "normalized_basis"
            ? "Normalized basis % vs F1: (Fn − F1) / F1 × 100; F1 = 0.00% with a dashed zero reference and a minimum ±1.5% axis span (expands for steeper curves). Positive = contango, negative = backwardation."
            : mode === "roll_yield"
              ? "Annualized roll yield % per deferred leg (F1 = 0%); positive = backwardation carry, negative = contango drag."
              : "Raw contract settlement prices ($); the axis keeps at least ±1.0% around F1 so flat curves still render with visible slope."}
        </p>
      </Panel>

      {/* ------------------------------------------------------ roll calendar */}
      <Panel
        className="col-span-12 xl:col-span-4"
        title="Roll schedule"
        flush
        actions={<span className="text-[11px] text-zinc-600">{data.roll_calendar.length} rolls</span>}
      >
        <Tabs
          ariaLabel="Roll views"
          value={rollTab}
          onChange={setRollTab}
          options={[
            { value: "schedule", label: "Schedule" },
            { value: "bag", label: "BAG queue", count: bagRolls.length },
          ]}
          className="px-3"
        />
        <div className="max-h-[318px] overflow-y-auto">
          <table className="data-table">
            <thead>
              <tr>
                <th>Root</th>
                <th>Roll date</th>
                <th className="text-right">DTR</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {(rollTab === "schedule" ? data.roll_calendar : bagRolls).map((roll) => (
                <RollRow
                  key={`${roll.root}-${roll.from_expiration}`}
                  roll={roll}
                  active={roll.root === selected.root}
                  onSelect={() => setRoot(roll.root)}
                />
              ))}
              {(rollTab === "bag" ? bagRolls : data.roll_calendar).length === 0 ? (
                <tr>
                  <td colSpan={4} className="py-6 text-center text-zinc-600">
                    {rollTab === "bag" ? "No BAG calendar-spread rolls scheduled." : "No rolls scheduled."}
                  </td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
        <div className="border-t border-hairline px-3 py-2 text-[10.5px] text-zinc-600">
          {selectedRoll ? (
            <span>
              {selected.root}: roll {formatDate(selectedRoll.roll_date)} · {selectedRoll.trigger.replace(/_/g, " ")} ·{" "}
              {selectedRoll.method}
            </span>
          ) : (
            "No roll scheduled for the selected root."
          )}
        </div>
      </Panel>

      {/* ----------------------------------------------------- multi-root matrix */}
      <Panel
        className="col-span-12"
        title="Curve matrix · all roots"
        actions={
          <span className="text-[11px] text-zinc-600">
            as of {formatDateTime(data.as_of)} · settlement
          </span>
        }
        flush
      >
        <div className="max-h-[300px] overflow-auto">
          <table className="data-table">
            <thead>
              <tr>
                <th className="sticky left-0 z-20 bg-panel">Root</th>
                <th>Regime</th>
                <th className="text-right">F2 − F1</th>
                <th className="text-right">Roll yield</th>
                {Array.from({ length: Math.max(...curves.map((curve) => curve.contracts.length)) }).map((_, index) => (
                  <th key={index} className="text-right">
                    F{index + 1}
                  </th>
                ))}
                <th className="text-right">Next roll</th>
              </tr>
            </thead>
            <tbody>
              {curves.map((curve) => {
                const roll = data.roll_calendar.find((entry) => entry.root === curve.root);
                const curveBasis =
                  secondPrice(curve) > 0 && frontPrice(curve) > 0
                    ? (secondPrice(curve) - frontPrice(curve)) / frontPrice(curve)
                    : 0;
                return (
                  <tr
                    key={curve.root}
                    className={cn("cursor-pointer", curve.root === selected.root && "bg-surface")}
                    onClick={() => setRoot(curve.root)}
                  >
                    <td className="sticky left-0 z-10 bg-panel font-medium text-zinc-100">
                      {curve.root}
                    </td>
                    <td>
                      <Badge tone={REGIME_TONE[curve.regime]}>{REGIME_LABEL[curve.regime]}</Badge>
                    </td>
                    <td className={cn("num text-right", curveBasis >= 0 ? "text-loss/90" : "text-gain/90")}>
                      {formatBps(curveBasis, 1)}
                    </td>
                    <td
                      className={cn(
                        "num text-right",
                        curve.roll_yield_annualized >= 0 ? "text-gain/90" : "text-loss/90",
                      )}
                    >
                      {formatPct(curve.roll_yield_annualized)}
                    </td>
                    {Array.from({ length: Math.max(...curves.map((item) => item.contracts.length)) }).map((_, index) => {
                      const contract = curve.contracts[index];
                      return (
                        <td key={index} className="num text-right text-zinc-400">
                          {contract ? contract.price.toFixed(2) : "—"}
                        </td>
                      );
                    })}
                    <td className="text-right">
                      {roll ? (
                        <span
                          className={cn(
                            "font-mono text-[11.5px] tabular-nums",
                            roll.days_to_roll <= 7 ? "text-warn" : "text-zinc-400",
                          )}
                        >
                          {roll.days_to_roll}d
                        </span>
                      ) : (
                        <span className="text-[11px] text-zinc-600">—</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div className="flex flex-wrap items-center justify-between gap-2 border-t border-hairline px-3 py-2 text-[10.5px] text-zinc-600">
          <span>
            {bagRolls.length} rolls queued as native BAG calendar spreads ·{" "}
            {data.roll_calendar.filter((roll) => roll.status === "imminent").length} imminent (≤7d)
          </span>
          <span>term slope = (F2 − F1) / F1 · roll yield annualised on expiry gap</span>
        </div>
      </Panel>
    </div>
  );
}

function CurveStat({
  label,
  value,
  sub,
  tone,
}: {
  label: string;
  value: string;
  sub?: string;
  tone?: string;
}) {
  return (
    <div>
      <div className="metric-label">{label}</div>
      <div className={cn("font-mono text-[12.5px] tabular-nums", tone ?? "text-zinc-200")}>{value}</div>
      {sub ? <div className="font-mono text-[10px] text-zinc-600">{sub}</div> : null}
    </div>
  );
}

function RollRow({
  roll,
  active,
  onSelect,
}: {
  roll: RollCalendarEntry;
  active: boolean;
  onSelect: () => void;
}) {
  const imminent = roll.days_to_roll <= 7 && roll.days_to_roll >= 0;
  return (
    <tr
      className={cn("cursor-pointer", active && "bg-surface")}
      onClick={onSelect}
      title={`${roll.from_expiration} → ${roll.to_expiration} · ${roll.trigger} · ${roll.method}`}
    >
      <td className="font-medium text-zinc-100">{roll.root}</td>
      <td className="num text-[11.5px] text-zinc-400">{formatDate(roll.roll_date)}</td>
      <td className={cn("num text-right", imminent ? "text-warn" : "text-zinc-400")}>
        {roll.days_to_roll}d
      </td>
      <td>
        <Badge tone={roll.status === "imminent" ? "warn" : roll.status === "completed" ? "neutral" : "accent"}>
          {roll.status === "imminent" ? "Imminent" : roll.status === "completed" ? "Done" : "Scheduled"}
        </Badge>
      </td>
    </tr>
  );
}
