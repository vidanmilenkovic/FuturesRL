"use client";

import { Badge } from "@/components/ui/badge";
import { LiveDot } from "@/components/ui/controls";
import { Panel } from "@/components/ui/panel";
import { useTelemetry } from "@/components/providers/telemetry-provider";
import { formatClock, formatPctPlain } from "@/lib/format";
import type { TelemetryEnvelope, TelemetryKind } from "@/lib/types";

const KIND_TONE: Record<TelemetryKind, "up" | "warn" | "critical" | "info" | "violet" | "neutral"> = {
  pnl: "up",
  margin: "warn",
  positions: "info",
  risk_alert: "critical",
  pipeline: "violet",
  heartbeat: "neutral",
};

function summarize(message: TelemetryEnvelope): string {
  const payload = message.payload;
  switch (message.kind) {
    case "pnl":
      if ("net_liquidation" in payload) {
        return `NLV ${payload.net_liquidation.toLocaleString("en-US", { maximumFractionDigits: 0 })} · day ${formatPctPlain(payload.day_pnl_pct)}`;
      }
      break;
    case "margin":
      if ("margin_utilization" in payload) {
        return `IM util ${formatPctPlain(payload.margin_utilization)} · ${payload.initial_margin_used.toLocaleString("en-US", { maximumFractionDigits: 0 })} used`;
      }
      break;
    case "positions":
      if ("positions" in payload) {
        return `${payload.positions.length} legs · gross ${payload.gross_leverage.toFixed(2)}×`;
      }
      break;
    case "risk_alert":
      if ("circuit_breaker_active" in payload) {
        return payload.circuit_breaker_active
          ? `circuit breaker · ${payload.circuit_breaker_reason}`
          : "no circuit-breaker overrides";
      }
      break;
    case "pipeline":
      if ("pipeline" in payload) {
        return `cycle ${payload.pipeline.cycle_id} · ${payload.pipeline.instruments.filter((row) => row.selected).length} roots`;
      }
      break;
    case "heartbeat":
      return "stream alive";
  }
  return message.kind;
}

export function TelemetryTape() {
  const { events, status, sequence } = useTelemetry();

  return (
    <Panel
      title="telemetry tape"
      subtitle="/ws/telemetry · newest first"
      actions={<LiveDot state={status === "live" ? "live" : status === "demo" ? "demo" : "offline"} label={status} />}
      contentClassName="px-0 py-0"
    >
      <div className="terminal-scroll divide-y divide-terminal-600/30">
        {events.length === 0 ? (
          <p className="px-3 py-6 text-center font-mono text-[10px] uppercase tracking-[0.12em] text-terminal-400">
            awaiting telemetry frames…
          </p>
        ) : (
          events.slice(0, 24).map((event) => (
            <div key={`${event.sequence}-${event.kind}`} className="flex items-center gap-2 px-3 py-1.5 font-mono text-[10px]">
              <span className="w-14 shrink-0 tabular-nums text-terminal-500">{formatClock(event.timestamp)}</span>
              <Badge tone={KIND_TONE[event.kind]}>{event.kind}</Badge>
              <span className="truncate text-terminal-300">{summarize(event)}</span>
              <span className="ml-auto shrink-0 tabular-nums text-terminal-600">#{event.sequence}</span>
            </div>
          ))
        )}
      </div>
      <div className="border-t border-terminal-600/40 px-3 py-1.5 font-mono text-[9px] uppercase tracking-[0.12em] text-terminal-500">
        sequence {sequence.toString().padStart(5, "0")} · {events.length} frames buffered
      </div>
    </Panel>
  );
}
