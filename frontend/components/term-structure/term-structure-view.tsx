"use client";

import { useEffect, useMemo, useState } from "react";

import { CurveChart } from "@/components/charts/svg-charts";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/controls";
import { Panel } from "@/components/ui/panel";
import { getTermStructure } from "@/lib/api";
import { cn } from "@/lib/cn";
import { formatDate, formatPct, formatPctPlain } from "@/lib/format";
import { useAsyncData } from "@/lib/use-async-data";
import { useNow } from "@/lib/use-now";
import type { TermCurve, TermStructureResponse } from "@/lib/types";

const REGIME_TONE = {
  contango: "down",
  backwardation: "up",
  flat: "info",
} as const;

/**
 * View 4 — Futures Term Structure & Roll Calendar.
 *
 * Contango/backwardation curves (F1…Fn) per root plus the mandatory roll
 * schedule produced by the continuous-roll engine (calendar days before FND /
 * volume-OI crossover, executed as BAG calendar spreads).
 */
export function TermStructureView() {
  const term = useAsyncData(getTermStructure);
  const [data, setData] = useState<TermStructureResponse | null>(null);
  const [selectedRoot, setSelectedRoot] = useState<string>("ES");
  const now = useNow();

  useEffect(() => {
    const loaded = term.data;
    if (!loaded) return;
    setData(loaded);
    setSelectedRoot((current) =>
      loaded.curves.some((curve) => curve.root === current) ? current : (loaded.curves[0]?.root ?? "ES"),
    );
  }, [term.data]);

  const curves = useMemo(() => data?.curves ?? [], [data]);
  const selected = useMemo(() => curves.find((curve) => curve.root === selectedRoot) ?? curves[0], [curves, selectedRoot]);

  if (term.loading && !data) {
    return (
      <div className="space-y-3">
        <Skeleton className="h-16" />
        <div className="grid gap-3 xl:grid-cols-3">
          <Skeleton className="h-[340px] xl:col-span-2" />
          <Skeleton className="h-[340px]" />
        </div>
        <Skeleton className="h-[280px]" />
      </div>
    );
  }

  if (!data || !selected) {
    return (
      <Panel title="term structure">
        <p className="py-10 text-center font-mono text-xs uppercase tracking-[0.12em] text-signal-down">
          term-structure data unavailable
        </p>
      </Panel>
    );
  }

  const frontCurve = selected.contracts[0];
  const secondCurve = selected.contracts[1];

  return (
    <div className="space-y-3">
      {/* Regime strip */}
      <div className="grid grid-cols-2 gap-2 md:grid-cols-3 xl:grid-cols-5">
        {curves.map((curve) => (
          <button
            key={curve.root}
            type="button"
            onClick={() => setSelectedRoot(curve.root)}
            aria-pressed={curve.root === selected.root}
            className={cn(
              "rounded-md border px-2.5 py-2 text-left transition-colors focus-ring",
              curve.root === selected.root
                ? "border-signal-info/50 bg-signal-info/[0.07]"
                : "border-terminal-600/60 bg-terminal-850/70 hover:border-terminal-500",
            )}
          >
            <div className="flex items-center justify-between gap-2">
              <p className="font-mono text-sm font-semibold text-terminal-100">{curve.root}</p>
              <Badge tone={REGIME_TONE[curve.regime]} dot>
                {curve.regime}
              </Badge>
            </div>
            <p className="mt-1 font-mono text-[10px] uppercase tracking-[0.08em] text-terminal-400">
              F1 {frontPrice(curve).toLocaleString("en-US", { maximumFractionDigits: 2 })}
            </p>
            <p
              className={cn(
                "font-mono text-[10px] tabular-nums",
                curve.roll_yield_annualized >= 0 ? "text-signal-up" : "text-signal-down",
              )}
            >
              roll {formatPct(curve.roll_yield_annualized)} ann.
            </p>
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 gap-3 xl:grid-cols-3">
        <Panel
          title={`${selected.root} futures curve`}
          subtitle={`${selected.asset_class.replace("_", " ")} · F1…F${selected.contracts.length} · as of ${formatDate(data.as_of)}`}
          className="xl:col-span-2"
          actions={<Badge tone={REGIME_TONE[selected.regime]}>{selected.regime.toUpperCase()}</Badge>}
        >
          <CurveChart
            points={selected.contracts.map((contract, index) => ({
              label: `F${index + 1}`,
              price: contract.price,
              days: contract.days_to_expiry,
              isFront: contract.is_front,
            }))}
            regime={selected.regime}
            ariaLabel={`${selected.root} term structure curve`}
          />
          <dl className="mt-1 grid grid-cols-2 gap-2 border-t border-terminal-600/40 pt-2 sm:grid-cols-4">
            <CurveStat label="front (F1)" value={frontCurve ? frontCurve.price.toFixed(2) : "—"} />
            <CurveStat label="second (F2)" value={secondCurve ? secondCurve.price.toFixed(2) : "—"} />
            <CurveStat
              label="term slope"
              value={formatPct(selected.slope, 3)}
              tone={selected.slope >= 0 ? "down" : "up"}
            />
            <CurveStat
              label="roll yield ann."
              value={formatPct(selected.roll_yield_annualized)}
              tone={selected.roll_yield_annualized >= 0 ? "up" : "down"}
            />
          </dl>
        </Panel>

        <Panel title="roll calendar" subtitle="next rolls · BAG calendar spread execution" actions={<Badge tone="warn">{data.roll_calendar.filter((roll) => roll.days_to_roll <= 7).length} imminent</Badge>}>
          <div className="divide-y divide-terminal-600/30">
            {data.roll_calendar.map((roll) => {
              const imminent = roll.days_to_roll <= 7;
              const rollTime = new Date(roll.roll_date);
              const hoursRemaining = now === null ? null : Math.max(0, (rollTime.getTime() - now.getTime()) / 3_600_000);
              return (
                <div key={`${roll.root}-${roll.from_expiration}`} className="py-2">
                  <div className="flex items-center justify-between gap-2">
                    <p className="flex items-center gap-1.5 font-mono text-[12px] font-semibold text-terminal-100">
                      {roll.root}
                      <Badge tone={imminent ? "warn" : "neutral"} dot>
                        {roll.status}
                      </Badge>
                    </p>
                    <p className={cn("font-mono text-[11px] tabular-nums", imminent ? "text-signal-warn" : "text-terminal-200")}>
                      {roll.days_to_roll}d
                      {hoursRemaining !== null ? (
                        <span className="ml-1 text-[9px] text-terminal-500">
                          {hoursRemaining < 48 ? `${Math.floor(hoursRemaining)}h` : ""}
                        </span>
                      ) : null}
                    </p>
                  </div>
                  <p className="font-mono text-[10px] text-terminal-400">
                    {roll.from_expiration} → {roll.to_expiration} · {formatDate(roll.roll_date)}
                  </p>
                  <p className="font-mono text-[9px] uppercase tracking-[0.08em] text-terminal-500">
                    {roll.trigger.replace(/_/g, " ")} · {roll.method}
                  </p>
                </div>
              );
            })}
          </div>
        </Panel>
      </div>

      {/* Basis table */}
      <Panel
        title="curve table"
        subtitle="forward prices, basis and annualised roll yield per root"
        contentClassName="px-3 pt-2 pb-0"
      >
        <div className="-mx-3 overflow-x-auto">
          <table className="data-table min-w-[980px]">
            <thead>
              <tr>
                <th>root</th>
                <th>regime</th>
                {Array.from({ length: Math.max(...curves.map((curve) => curve.contracts.length)) }).map((_, index) => (
                  <th key={index} className="num">
                    F{index + 1}
                  </th>
                ))}
                <th className="num">F2−F1 basis</th>
                <th className="num">roll yield ann.</th>
                <th>next roll</th>
              </tr>
            </thead>
            <tbody>
              {curves.map((curve) => {
                const roll = data.roll_calendar.find((entry) => entry.root === curve.root);
                const basis = (curve.contracts[1]?.price ?? 0) - (curve.contracts[0]?.price ?? 0);
                return (
                  <tr key={curve.root}>
                    <td className="font-semibold text-terminal-100">{curve.root}</td>
                    <td>
                      <Badge tone={REGIME_TONE[curve.regime]} dot>
                        {curve.regime}
                      </Badge>
                    </td>
                    {Array.from({ length: Math.max(...curves.map((entry) => entry.contracts.length)) }).map((_, index) => (
                      <td key={index} className="num text-terminal-300">
                        {curve.contracts[index] ? curve.contracts[index]!.price.toFixed(2) : "—"}
                      </td>
                    ))}
                    <td className={cn("num", basis >= 0 ? "text-signal-down" : "text-signal-up")}>
                      {formatPct(curve.slope, 3)}
                    </td>
                    <td className={cn("num font-semibold", curve.roll_yield_annualized >= 0 ? "text-signal-up" : "text-signal-down")}>
                      {formatPct(curve.roll_yield_annualized)}
                    </td>
                    <td className="text-terminal-300">
                      {roll ? `${roll.days_to_roll}d · ${roll.method}` : "—"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Panel>

      <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
        <Panel title="carry interpretation" subtitle="annualised roll yield by regime">
          <ul className="space-y-1.5 font-mono text-[10px] leading-relaxed text-terminal-400">
            <li>
              <span className="text-signal-down">Contango</span> → negative roll yield: long exposure pays the
              curve, short harvests carry.
            </li>
            <li>
              <span className="text-signal-up">Backwardation</span> → positive roll yield: longs are paid to hold,
              physical tightness premium.
            </li>
            <li>
              <span className="text-signal-info">Flat</span> → carry ≈ 0; timing overlay dominates.
            </li>
          </ul>
        </Panel>
        <Panel title="roll mechanics" subtitle="no-physical-delivery guarantee">
          <p className="font-mono text-[10px] leading-relaxed text-terminal-400">
            Physical commodities (CL, GC) roll a configured number of calendar days before first-notice day.
            Execution uses a native BAG calendar spread when the venue supports it, falling back to two-legged
            market orders with the close-leg sent first. The simulator applies both legs&apos; fees and slippage.
          </p>
        </Panel>
        <Panel title="curve snapshot" subtitle="term-structure feature inputs">
          <div className="space-y-1 font-mono text-[10px] text-terminal-300">
            <p>term_slope = (F2 − F1) / F1</p>
            <p>carry_annualized = −slope × 365 / Δdays</p>
            <p>volume_oi_ratio · ofi · turbulence</p>
            <p className="text-terminal-500">
              Features are exported by the data layer and consumed by T_t / R_t; the dashboard only displays them.
            </p>
          </div>
          <p className="mt-2 border-t border-terminal-600/40 pt-2 font-mono text-[10px] text-terminal-400">
            {selected.root}: F2−F1 {formatPctPlain(selected.slope, 3)} ·{" "}
            {selected.contracts.length} listed expiries
          </p>
        </Panel>
      </div>
    </div>
  );
}

function frontPrice(curve: TermCurve): number {
  return curve.contracts[0]?.price ?? 0;
}

function CurveStat({ label, value, tone = "neutral" }: { label: string; value: string; tone?: "up" | "down" | "neutral" }) {
  return (
    <div>
      <dt className="font-mono text-[9px] uppercase tracking-[0.12em] text-terminal-500">{label}</dt>
      <dd
        className={cn(
          "font-mono text-sm tabular-nums",
          tone === "up" ? "text-signal-up" : tone === "down" ? "text-signal-down" : "text-terminal-100",
        )}
      >
        {value}
      </dd>
    </div>
  );
}
