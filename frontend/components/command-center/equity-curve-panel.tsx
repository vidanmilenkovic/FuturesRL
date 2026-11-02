"use client";

import { useMemo, useState } from "react";

import { LightweightChart } from "@/components/charts/lightweight-chart";
import { Segmented } from "@/components/ui/controls";
import { cn } from "@/lib/cn";
import { formatCurrency, formatPct } from "@/lib/format";
import type { EquityPoint } from "@/lib/types";

type RangeKey = "1M" | "3M" | "6M" | "1Y";

const RANGE_POINTS: Record<RangeKey, number> = {
  "1M": 21,
  "3M": 63,
  "6M": 126,
  "1Y": 252,
};

/**
 * Equity / PnL curve for the Command Center.
 *
 * Uses TradingView Lightweight Charts for the interactive crosshair and price
 * scale; the window selector and summary statistics are computed client-side
 * from the account snapshot's equity curve.
 */
export function EquityCurvePanel({ points }: { points: EquityPoint[] }) {
  const [range, setRange] = useState<RangeKey>("6M");

  const windowed = useMemo(() => {
    const count = RANGE_POINTS[range];
    return points.slice(Math.max(0, points.length - count));
  }, [points, range]);

  const first = windowed[0]?.value ?? 0;
  const last = windowed[windowed.length - 1]?.value ?? 0;
  const change = first > 0 ? last / first - 1 : 0;
  const high = windowed.length > 0 ? Math.max(...windowed.map((point) => point.value)) : 0;
  const low = windowed.length > 0 ? Math.min(...windowed.map((point) => point.value)) : 0;
  const positive = change >= 0;

  return (
    <div className="flex h-full flex-col">
      <div className="mb-2 flex flex-wrap items-end justify-between gap-2">
        <div className="flex items-baseline gap-3">
          <div>
            <p className="mono-label">net liquidation value</p>
            <p className="font-mono text-2xl font-semibold tabular-nums tracking-tight text-terminal-100">
              {formatCurrency(last)}
            </p>
          </div>
          <span
            className={cn(
              "font-mono text-sm tabular-nums",
              positive ? "text-signal-up" : "text-signal-down",
            )}
          >
            {formatPct(change)}
          </span>
          <span className="hidden font-mono text-[10px] uppercase tracking-[0.1em] text-terminal-500 sm:inline">
            {range}
          </span>
        </div>
        <Segmented
          size="sm"
          ariaLabel="Equity curve range"
          value={range}
          onChange={setRange}
          options={(Object.keys(RANGE_POINTS) as RangeKey[]).map((key) => ({
            value: key,
            label: key,
          }))}
        />
      </div>

      <div className="min-h-0 flex-1">
        <LightweightChart
          data={windowed}
          kind="area"
          height={264}
          color={positive ? "#34d399" : "#fb7185"}
          secondaryColor={positive ? "rgba(52,211,153,0.22)" : "rgba(251,113,133,0.22)"}
          ariaLabel={`Equity curve over the last ${range}`}
        />
      </div>

      <dl className="mt-2 grid grid-cols-3 gap-2 border-t border-terminal-600/40 pt-2 font-mono text-[10px] uppercase tracking-[0.1em]">
        <div>
          <dt className="text-terminal-400">period high</dt>
          <dd className="mt-0.5 tabular-nums text-terminal-100">{formatCurrency(high)}</dd>
        </div>
        <div>
          <dt className="text-terminal-400">period low</dt>
          <dd className="mt-0.5 tabular-nums text-terminal-100">{formatCurrency(low)}</dd>
        </div>
        <div>
          <dt className="text-terminal-400">samples</dt>
          <dd className="mt-0.5 tabular-nums text-terminal-100">{windowed.length} d</dd>
        </div>
      </dl>
    </div>
  );
}
