"use client";

import { memo, useEffect } from "react";

import { cn } from "@/lib/cn";
import { formatClock } from "@/lib/format";
import { summarizeTelemetry } from "@/lib/telemetry";
import type { TelemetryKind } from "@/lib/types";

import { useTelemetry } from "../providers/telemetry-provider";
import { Badge, type Tone } from "../ui/primitives";

const KIND_TONE: Record<TelemetryKind, Tone> = {
  pnl: "gain",
  margin: "warn",
  positions: "accent",
  risk_alert: "loss",
  pipeline: "accent",
  heartbeat: "neutral",
};

/**
 * Collapsible raw telemetry console docked above the status bar. Shows the last
 * frames exactly as received (kind + payload summary + sequence), which is the
 * honest source for working-order/fill information the REST API does not expose.
 */
export const TelemetryDrawer = memo(function TelemetryDrawer({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const { events, status, sequence } = useTelemetry();

  useEffect(() => {
    if (!open) return;
    window.dispatchEvent(new Event("futuresrlx:pause-telemetry"));
    return () => {
      window.dispatchEvent(new Event("futuresrlx:resume-telemetry"));
    };
  }, [open]);

  return (
    <div
      className={cn(
        "absolute inset-x-0 bottom-7 z-40 flex flex-col border-t border-edge bg-panel transition-transform duration-150",
        open ? "translate-y-0" : "pointer-events-none translate-y-2 opacity-0",
      )}
      style={{ height: open ? 224 : 0, overflow: "hidden" }}
      aria-hidden={!open}
    >
      <div className="flex h-9 shrink-0 items-center justify-between border-b border-hairline px-3">
        <div className="flex items-center gap-2">
          <span className="text-[12px] font-medium text-zinc-200">Telemetry stream</span>
          <span className="font-mono text-[11px] text-zinc-600">
            {status} · seq {sequence.toString().padStart(5, "0")} · {events.length} buffered
          </span>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="focus-ring rounded px-1.5 py-0.5 text-[11px] text-zinc-500 hover:bg-surface hover:text-zinc-200"
        >
          Close
        </button>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto">
        <table className="data-table">
          <thead>
            <tr>
              <th className="w-24">Time</th>
              <th className="w-24">Kind</th>
              <th>Payload</th>
              <th className="w-16 text-right">Seq</th>
            </tr>
          </thead>
          <tbody>
            {events.length === 0 ? (
              <tr>
                <td colSpan={4} className="py-8 text-center text-zinc-600">
                  Waiting for telemetry frames…
                </td>
              </tr>
            ) : (
              events.slice(0, 80).map((frame, index) => (
                <tr key={`${frame.sequence}-${frame.kind}-${index}`}>
                  <td className="num text-zinc-500">{formatClock(frame.timestamp)}</td>
                  <td>
                    <Badge tone={KIND_TONE[frame.kind]}>{frame.kind}</Badge>
                  </td>
                  <td className="max-w-0 truncate text-zinc-400" title={summarizeTelemetry(frame)}>
                    {summarizeTelemetry(frame)}
                  </td>
                  <td className="num text-right text-zinc-600">{frame.sequence}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
});
