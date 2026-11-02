"use client";

import { useEffect, useMemo, useState } from "react";

import { useTelemetry } from "@/components/providers/telemetry-provider";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Field, ProgressBar, Skeleton, TextInput } from "@/components/ui/controls";
import { Panel } from "@/components/ui/panel";
import { getRiskState, updateRiskThresholds } from "@/lib/api";
import { cn } from "@/lib/cn";
import { formatDateTime, formatPctPlain, formatSignedNumber } from "@/lib/format";
import { useAsyncData } from "@/lib/use-async-data";
import { useNow } from "@/lib/use-now";
import type {
  MacroLockout,
  NewsEvent,
  RiskAlertTelemetryPayload,
  RiskAssessmentRow,
  RiskStateResponse,
  RiskThresholds,
  TelemetryPayload,
} from "@/lib/types";

function isRiskAlertPayload(payload: TelemetryPayload): payload is RiskAlertTelemetryPayload {
  return "circuit_breaker_active" in payload;
}

const ACTION_TONE = {
  pass: "up",
  scale: "warn",
  liquidate: "critical",
  neutralized: "neutral",
} as const;

/**
 * View 3 — LLM Risk-First Monitor.
 *
 * Surfaces the Tier-A structural constraint engine: per-root sentiment /
 * semantic risk / confidence, the deterministic breaker action, the macro
 * lockout countdown, the news feed and the editable risk thresholds.
 */
export function RiskView() {
  const risk = useAsyncData(getRiskState);
  const { latest } = useTelemetry();
  const [state, setState] = useState<RiskStateResponse | null>(null);
  const now = useNow();

  useEffect(() => {
    if (risk.data) setState(risk.data);
  }, [risk.data]);

  useEffect(() => {
    const frame = latest.risk_alert;
    if (!frame) return;
    const payload = frame.payload;
    if (!isRiskAlertPayload(payload)) return;
    setState((previous) =>
      previous
        ? {
            ...previous,
            timestamp: frame.timestamp,
            assessments: payload.assessments,
            circuit_breaker_active: payload.circuit_breaker_active,
            circuit_breaker_reason: payload.circuit_breaker_reason,
            macro_lockout: payload.macro_lockout,
          }
        : previous,
    );
  }, [latest.risk_alert]);

  if (risk.loading && !state) {
    return (
      <div className="space-y-3">
        <Skeleton className="h-20" />
        <div className="grid gap-3 xl:grid-cols-3">
          <Skeleton className="h-[280px] xl:col-span-2" />
          <Skeleton className="h-[280px]" />
        </div>
        <Skeleton className="h-[300px]" />
      </div>
    );
  }

  if (!state) {
    return (
      <Panel title="llm risk-first monitor">
        <p className="py-10 text-center font-mono text-xs uppercase tracking-[0.12em] text-signal-down">
          risk state unavailable
        </p>
      </Panel>
    );
  }

  const lockout = state.macro_lockout;
  const nextMacro = [...state.macro_events].sort((a, b) => a.minutes_until - b.minutes_until)[0];

  return (
    <div className="space-y-3">
      {/* Breaker / lockout banner */}
      <div className="grid grid-cols-1 gap-2 md:grid-cols-3">
        <div
          className={cn(
            "flex items-center gap-3 rounded-md border px-3 py-2.5",
            state.circuit_breaker_active
              ? "border-signal-critical/60 bg-signal-critical/10 animate-pulse-critical"
              : "border-signal-up/30 bg-signal-up/[0.04]",
          )}
        >
          <span
            aria-hidden
            className={cn(
              "grid h-9 w-9 shrink-0 place-items-center rounded-full border font-mono text-lg",
              state.circuit_breaker_active
                ? "border-signal-critical/60 text-signal-critical"
                : "border-signal-up/50 text-signal-up",
            )}
          >
            {state.circuit_breaker_active ? "!" : "✓"}
          </span>
          <div className="min-w-0">
            <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-terminal-400">circuit breaker</p>
            <p className={cn("truncate font-mono text-sm font-semibold uppercase tracking-[0.1em]", state.circuit_breaker_active ? "text-signal-critical" : "text-signal-up")}>
              {state.circuit_breaker_active ? "OVERRIDE ACTIVE" : "NO OVERRIDES"}
            </p>
            <p className="truncate font-mono text-[10px] text-terminal-400" title={state.circuit_breaker_reason}>
              {state.circuit_breaker_reason}
            </p>
          </div>
        </div>

        <MacroLockoutCard lockout={lockout} now={now} />

        <div className="flex items-center gap-3 rounded-md border border-terminal-600/60 bg-terminal-850/70 px-3 py-2.5">
          <span aria-hidden className="grid h-9 w-9 shrink-0 place-items-center rounded-full border border-signal-info/40 font-mono text-sm text-signal-info">
            c
          </span>
          <div className="min-w-0">
            <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-terminal-400">confidence filter</p>
            <p className="font-mono text-sm font-semibold uppercase tracking-[0.1em] text-terminal-100">
              τ = {state.thresholds.llm_confidence_threshold.toFixed(2)}
            </p>
            <p className="truncate font-mono text-[10px] text-terminal-400">
              {state.neutralized_instruments.length} signal(s) neutralised:{" "}
              {state.neutralized_instruments.join(", ") || "none"}
            </p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-3 xl:grid-cols-3">
        {/* Root risk cards */}
        <Panel
          title="per-root llm assessment"
          subtitle={`sentiment · c_i,t · S_i,t · breaker action · ${formatDateTime(state.timestamp)}`}
          className="xl:col-span-2"
          actions={
            <Badge tone="violet" dot>
              risk-first
            </Badge>
          }
        >
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
            {state.assessments.map((assessment) => (
              <AssessmentCard key={assessment.instrument_id} assessment={assessment} thresholds={state.thresholds} />
            ))}
          </div>
        </Panel>

        <div className="space-y-3">
          <Panel
            title="macro event lockout"
            subtitle="Tier-1 releases force flatten / cap leverage"
            actions={<Badge tone={nextMacro && nextMacro.minutes_until < 60 ? "warn" : "neutral"}>{state.macro_events.length} events</Badge>}
          >
            <div className="divide-y divide-terminal-600/30">
              {state.macro_events.map((event) => {
                const eventTime = new Date(event.event_time);
                const minutes =
                  now === null
                    ? event.minutes_until
                    : Math.max(0, Math.round((eventTime.getTime() - now.getTime()) / 60_000));
                const imminent = minutes <= 60;
                return (
                  <div key={event.id} className="flex items-center justify-between gap-2 py-2">
                    <div className="min-w-0">
                      <p className="flex items-center gap-1.5 truncate font-mono text-[11px] text-terminal-100">
                        {event.event}
                        <span className="font-mono text-[9px] uppercase tracking-[0.1em] text-terminal-500">T{event.tier}</span>
                      </p>
                      <p className="font-mono text-[9px] uppercase tracking-[0.08em] text-terminal-500">
                        {formatDateTime(event.event_time)} · {event.roots.join(" ")}
                      </p>
                    </div>
                    <span
                      className={cn(
                        "shrink-0 font-mono text-[10px] tabular-nums",
                        imminent ? "text-signal-warn" : "text-terminal-300",
                      )}
                    >
                      {minutes < 60 ? `${minutes}m` : `${Math.floor(minutes / 60)}h ${minutes % 60}m`}
                    </span>
                  </div>
                );
              })}
            </div>
            <p className="border-t border-terminal-600/40 pt-2 font-mono text-[10px] leading-relaxed text-terminal-400">
              Lockout window: −{state.thresholds.macro_lockout_minutes_before}m / +
              {state.thresholds.macro_lockout_minutes_after}m around each Tier-1 release. New entries are blocked;
              existing leverage is capped.
            </p>
          </Panel>

          <ThresholdEditor thresholds={state.thresholds} onUpdated={(thresholds) => setState((previous) => (previous ? { ...previous, thresholds } : previous))} />
        </div>
      </div>

      {/* News feed */}
      <div className="grid grid-cols-1 gap-3 lg:grid-cols-3">
        <Panel
          title="live macro / news feed"
          subtitle="LLM-ingested headlines · deterministic circuit breaker downstream"
          className="lg:col-span-2"
          actions={<Badge tone="info" dot>streaming</Badge>}
        >
          <div className="space-y-2">
            {state.news.map((event) => (
              <NewsRow key={event.id} event={event} />
            ))}
          </div>
        </Panel>

        <Panel title="breaker projection" subtitle="w_i ← w_i · max(0, (κ − S)/(κ − θ_safe))">
          <div className="space-y-2.5">
            {state.assessments.map((assessment) => {
              const { llm_safe_score: safe, llm_critical_score: critical } = state.thresholds;
              const ratio =
                assessment.semantic_risk_score >= critical
                  ? 1
                  : assessment.semantic_risk_score <= safe
                    ? 0
                    : (assessment.semantic_risk_score - safe) / (critical - safe);
              return (
                <div key={assessment.instrument_id}>
                  <div className="mb-1 flex items-center justify-between font-mono text-[10px] uppercase tracking-[0.08em]">
                    <span className="text-terminal-300">{assessment.instrument_id}</span>
                    <span className="tabular-nums text-terminal-200">
                      S {assessment.semantic_risk_score.toFixed(2)} · ×{assessment.weight_multiplier.toFixed(3)}
                    </span>
                  </div>
                  <ProgressBar
                    value={ratio}
                    tone={ratio >= 0.99 ? "critical" : ratio > 0 ? "warn" : "up"}
                    label={`${assessment.instrument_id} semantic risk projection`}
                  />
                </div>
              );
            })}
          </div>
          <div className="mt-3 border-t border-terminal-600/40 pt-2 font-mono text-[10px] leading-relaxed text-terminal-400">
            θ_safe = {state.thresholds.llm_safe_score.toFixed(1)} · κ ={" "}
            {state.thresholds.llm_critical_score.toFixed(1)}. Signals with c &lt; τ are neutralised to θ_safe
            before projection, so hallucinated events can never liquidate a book.
          </div>
        </Panel>
      </div>
    </div>
  );
}

function AssessmentCard({
  assessment,
  thresholds,
}: {
  assessment: RiskAssessmentRow;
  thresholds: RiskThresholds;
}) {
  const riskRatio = assessment.semantic_risk_score / 5;
  const confidenceBelowThreshold = assessment.confidence < thresholds.llm_confidence_threshold;
  return (
    <div className="rounded-sm border border-terminal-600/60 bg-terminal-900/50 px-2.5 py-2">
      <div className="flex items-center justify-between gap-2">
        <p className="font-mono text-sm font-semibold text-terminal-100">{assessment.instrument_id}</p>
        <Badge tone={ACTION_TONE[assessment.action]} dot>
          {assessment.action === "neutralized" ? "neutralised" : assessment.action}
        </Badge>
      </div>

      <div className="mt-2 grid grid-cols-3 gap-2">
        <Metric label="sentiment" value={formatSignedNumber(assessment.sentiment_score - 3, 1)} hint="1 – 5" />
        <Metric
          label="S_i,t"
          value={assessment.semantic_risk_score.toFixed(2)}
          hint={assessment.semantic_risk_score >= thresholds.llm_critical_score ? "≥ κ" : `θ ${thresholds.llm_safe_score}`}
          tone={assessment.semantic_risk_score >= thresholds.llm_critical_score ? "critical" : assessment.semantic_risk_score > thresholds.llm_safe_score ? "warn" : "up"}
        />
        <Metric
          label="c_i,t"
          value={assessment.confidence.toFixed(2)}
          hint={confidenceBelowThreshold ? "< τ" : "≥ τ"}
          tone={confidenceBelowThreshold ? "warn" : "up"}
        />
      </div>

      <div className="mt-2">
        <div className="mb-0.5 flex items-center justify-between font-mono text-[9px] uppercase tracking-[0.1em] text-terminal-500">
          <span>semantic risk</span>
          <span>
            θ_safe {thresholds.llm_safe_score} · κ {thresholds.llm_critical_score}
          </span>
        </div>
        <div className="relative h-1.5 w-full overflow-hidden rounded-full bg-terminal-700">
          <div
            className={cn(
              "h-full rounded-full",
              assessment.semantic_risk_score >= thresholds.llm_critical_score
                ? "bg-signal-critical"
                : assessment.semantic_risk_score > thresholds.llm_safe_score
                  ? "bg-signal-warn"
                  : "bg-signal-up",
            )}
            style={{ width: `${riskRatio * 100}%` }}
          />
          <span
            aria-hidden
            className="absolute top-0 h-full w-px bg-terminal-100/70"
            style={{ left: `${(thresholds.llm_critical_score / 5) * 100}%` }}
          />
        </div>
      </div>

      <p className="mt-2 line-clamp-2 font-mono text-[10px] leading-relaxed text-terminal-400" title={assessment.reasoning_summary}>
        {assessment.reasoning_summary}
      </p>
      <div className="mt-1 flex items-center justify-between font-mono text-[9px] uppercase tracking-[0.08em] text-terminal-500">
        <span>weight ×{assessment.weight_multiplier.toFixed(3)}</span>
        <span>{assessment.source}</span>
      </div>
    </div>
  );
}

function Metric({
  label,
  value,
  hint,
  tone = "neutral",
}: {
  label: string;
  value: string;
  hint?: string;
  tone?: "neutral" | "up" | "warn" | "critical";
}) {
  return (
    <div>
      <p className="font-mono text-[9px] uppercase tracking-[0.12em] text-terminal-500">{label}</p>
      <p
        className={cn(
          "font-mono text-sm font-semibold tabular-nums",
          tone === "up"
            ? "text-signal-up"
            : tone === "warn"
              ? "text-signal-warn"
              : tone === "critical"
                ? "text-signal-critical"
                : "text-terminal-100",
        )}
      >
        {value}
      </p>
      {hint ? <p className="font-mono text-[9px] text-terminal-500">{hint}</p> : null}
    </div>
  );
}

function MacroLockoutCard({ lockout, now }: { lockout: MacroLockout; now: Date | null }) {
  const remaining = lockout.minutes_remaining;
  const endsAt = lockout.ends_at;
  return (
    <div
      className={cn(
        "flex items-center gap-3 rounded-md border px-3 py-2.5",
        lockout.active ? "border-signal-lock/60 bg-signal-lock/10" : "border-terminal-600/60 bg-terminal-850/70",
      )}
    >
      <span
        aria-hidden
        className={cn(
          "grid h-9 w-9 shrink-0 place-items-center rounded-full border font-mono text-sm",
          lockout.active ? "border-signal-lock/60 text-signal-lock" : "border-terminal-500 text-terminal-300",
        )}
      >
        ◷
      </span>
      <div className="min-w-0">
        <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-terminal-400">macro lockout</p>
        <p
          className={cn(
            "font-mono text-sm font-semibold uppercase tracking-[0.1em]",
            lockout.active ? "text-signal-lock" : "text-terminal-100",
          )}
        >
          {lockout.active ? `${remaining}m remaining` : "clear"}
        </p>
        <p className="truncate font-mono text-[10px] text-terminal-400">
          {lockout.active && lockout.event
            ? `${lockout.event} · until ${formatDateTime(endsAt)}`
            : `no Tier-1 release inside window${now ? ` · ${formatClockSafe(now)}` : ""}`}
        </p>
      </div>
    </div>
  );
}

function formatClockSafe(date: Date): string {
  return `${date.toISOString().slice(11, 19)}Z`;
}

function NewsRow({ event }: { event: NewsEvent }) {
  const tone = event.severity === "critical" ? "critical" : event.severity === "warn" ? "warn" : "info";
  return (
    <article className="flex gap-2.5 border-b border-terminal-600/30 pb-2 last:border-0">
      <span aria-hidden className={cn("mt-1 h-2 w-2 shrink-0 rounded-full", tone === "critical" ? "bg-signal-critical" : tone === "warn" ? "bg-signal-warn" : "bg-signal-info")} />
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-1.5">
          <Badge tone={tone}>{event.severity}</Badge>
          <span className="font-mono text-[9px] uppercase tracking-[0.1em] text-terminal-500">{formatDateTime(event.timestamp)}</span>
          {event.roots.map((root) => (
            <span key={root} className="rounded-sm border border-terminal-500/50 px-1 font-mono text-[9px] uppercase tracking-[0.08em] text-terminal-300">
              {root}
            </span>
          ))}
        </div>
        <p className="mt-0.5 font-mono text-[11px] leading-snug text-terminal-100">{event.headline}</p>
        <p className="font-mono text-[10px] leading-relaxed text-terminal-400">{event.summary}</p>
        <p className="font-mono text-[9px] uppercase tracking-[0.1em] text-terminal-500">
          {event.source} · sentiment {event.sentiment_score.toFixed(1)}/5
        </p>
      </div>
    </article>
  );
}

function ThresholdEditor({
  thresholds,
  onUpdated,
}: {
  thresholds: RiskThresholds;
  onUpdated: (thresholds: RiskThresholds) => void;
}) {
  const [draft, setDraft] = useState(thresholds);
  const [status, setStatus] = useState<"idle" | "saving" | "saved" | "demo" | "error">("idle");
  const [message, setMessage] = useState<string | null>(null);

  useEffect(() => {
    setDraft(thresholds);
  }, [thresholds]);

  const validation = useMemo(() => {
    if (draft.llm_critical_score <= draft.llm_safe_score) return "κ must exceed θ_safe";
    if (draft.margin_critical_threshold <= draft.margin_warning_threshold) return "critical margin must exceed warning";
    if (draft.llm_confidence_threshold < 0 || draft.llm_confidence_threshold > 1) return "τ must be within [0, 1]";
    return null;
  }, [draft]);

  const fields: { key: keyof RiskThresholds; label: string; step: number; hint: string }[] = [
    { key: "llm_confidence_threshold", label: "confidence τ", step: 0.01, hint: "c < τ neutralises signal" },
    { key: "llm_safe_score", label: "θ_safe", step: 0.1, hint: "benign semantic risk" },
    { key: "llm_critical_score", label: "κ critical", step: 0.1, hint: "force flat at this S" },
    { key: "daily_max_loss_pct", label: "daily loss", step: 0.005, hint: "kill-switch fraction of NLV" },
  ];

  const save = async () => {
    if (validation) return;
    setStatus("saving");
    const result = await updateRiskThresholds(draft);
    onUpdated(result.data);
    setStatus(result.source === "live" ? "saved" : "demo");
    setMessage(
      result.source === "live"
        ? "thresholds persisted to the risk overlay"
        : "backend offline — overrides retained in this session only",
    );
  };

  return (
    <Panel title="risk thresholds" subtitle="PUT /api/risk/thresholds">
      <div className="grid grid-cols-2 gap-2">
        {fields.map((field) => (
          <Field key={field.key} label={field.label} htmlFor={`risk-${field.key}`} hint={field.hint}>
            <TextInput
              id={`risk-${field.key}`}
              type="number"
              step={field.step}
              value={draft[field.key]}
              onChange={(event) =>
                setDraft((previous) => ({ ...previous, [field.key]: Number(event.target.value) }))
              }
              aria-invalid={validation !== null}
            />
          </Field>
        ))}
      </div>

      <div className="mt-2 space-y-1 font-mono text-[10px] text-terminal-400">
        <p>
          margin warn {formatPctPlain(thresholds.margin_warning_threshold)} · critical{" "}
          {formatPctPlain(thresholds.margin_critical_threshold)}
        </p>
        <p>vol target {formatPctPlain(thresholds.target_annual_volatility)} annualised</p>
      </div>

      {validation ? (
        <p className="mt-2 font-mono text-[10px] uppercase tracking-[0.08em] text-signal-down">{validation}</p>
      ) : null}
      {message ? (
        <p
          className={cn(
            "mt-2 font-mono text-[10px] uppercase tracking-[0.08em]",
            status === "saved" ? "text-signal-up" : "text-signal-warn",
          )}
        >
          {message}
        </p>
      ) : null}

      <div className="mt-3 flex items-center gap-2">
        <Button variant="primary" size="sm" onClick={() => void save()} disabled={validation !== null || status === "saving"}>
          {status === "saving" ? "applying…" : "apply overrides"}
        </Button>
        <Button variant="ghost" size="sm" onClick={() => setDraft(thresholds)}>
          reset
        </Button>
      </div>
    </Panel>
  );
}
