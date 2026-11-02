import type { ReactNode } from "react";

import { cn } from "@/lib/cn";
import { formatSignedNumber } from "@/lib/format";

import { toneBorder, toneText, type Tone } from "./badge";

export interface StatCardProps {
  label: string;
  value: ReactNode;
  delta?: number | null;
  deltaLabel?: string;
  hint?: ReactNode;
  tone?: Tone;
  sparkline?: ReactNode;
  className?: string;
  emphasis?: boolean;
  badge?: ReactNode;
}

/**
 * Compact KPI tile. `delta` is rendered with the platform sign convention
 * (green = positive) and an explicit arrow so colour is never the only signal.
 */
export function StatCard({
  label,
  value,
  delta,
  deltaLabel,
  hint,
  tone = "neutral",
  sparkline,
  className,
  emphasis = false,
  badge,
}: StatCardProps) {
  return (
    <div
      className={cn(
        "relative flex min-w-0 flex-col justify-between overflow-hidden rounded-md border bg-terminal-850/80 px-3 py-2.5 shadow-panel",
        toneBorder[tone],
        emphasis ? "ring-1 ring-inset ring-signal-info/20" : "",
        className,
      )}
    >
      <div className="flex items-start justify-between gap-2">
        <p className="font-mono text-[10px] uppercase tracking-[0.14em] text-terminal-300">{label}</p>
        {badge}
      </div>
      <div className="mt-1.5 flex flex-wrap items-baseline gap-x-2 gap-y-1">
        <p
          className={cn(
            "font-mono text-xl font-semibold tabular-nums tracking-tight",
            emphasis ? "text-terminal-100" : toneText[tone],
          )}
        >
          {value}
        </p>
        {delta !== undefined && delta !== null ? (
          <span
            className={cn(
              "inline-flex items-center gap-0.5 font-mono text-[11px] tabular-nums",
              delta > 0 ? "text-signal-up" : delta < 0 ? "text-signal-down" : "text-terminal-300",
            )}
            title={deltaLabel}
          >
            <span aria-hidden>{delta > 0 ? "▲" : delta < 0 ? "▼" : "•"}</span>
            {formatSignedNumber(delta * 100, 2)}%
          </span>
        ) : null}
      </div>
      {hint ? <div className="mt-1 text-[10px] text-terminal-400">{hint}</div> : null}
      {sparkline ? <div className="pointer-events-none absolute inset-x-0 bottom-0 opacity-60">{sparkline}</div> : null}
    </div>
  );
}
