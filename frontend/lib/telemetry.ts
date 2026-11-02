import type { TelemetryEnvelope } from "./types";

/** Human summary of one telemetry frame (used by the drawer and book table). */
export function summarizeTelemetry(frame: TelemetryEnvelope): string {
  const payload = frame.payload;
  if (frame.kind === "pnl" && "net_liquidation" in payload) {
    return `NLV ${payload.net_liquidation.toLocaleString("en-US", { maximumFractionDigits: 0 })} · day ${(payload.day_pnl_pct * 100).toFixed(2)}%`;
  }
  if (frame.kind === "margin" && "margin_utilization" in payload) {
    return `Initial margin ${(payload.margin_utilization * 100).toFixed(2)}% · ${payload.initial_margin_used.toLocaleString("en-US", { maximumFractionDigits: 0 })} used`;
  }
  if (frame.kind === "positions" && "positions" in payload) {
    return `${payload.positions.length} legs · gross ${payload.gross_leverage.toFixed(2)}×`;
  }
  if (frame.kind === "risk_alert" && "circuit_breaker_active" in payload) {
    return payload.circuit_breaker_active
      ? `Circuit breaker active · ${payload.circuit_breaker_reason}`
      : `No circuit-breaker overrides · ${payload.assessments.length} assessments`;
  }
  if (frame.kind === "pipeline" && "pipeline" in payload) {
    return `Cycle ${payload.pipeline.cycle_id} · ${payload.pipeline.instruments.filter((row) => row.selected).length} roots`;
  }
  if (frame.kind === "heartbeat") return "Stream heartbeat";
  return frame.kind;
}
