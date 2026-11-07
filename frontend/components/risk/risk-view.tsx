"use client";

import { memo, useEffect, useMemo, useState } from "react";

import { RiskMeter } from "@/components/charts/plots";
import { RootScopeFilterBar, useRootScopeFilter } from "@/components/filters/root-scope-filter";
import { useAccount } from "@/components/providers/account-provider";
import { useTelemetry } from "@/components/providers/telemetry-provider";
import {
  Badge,
  Button,
  EmptyState,
  Field,
  KpiCard,
  Panel,
  Skeleton,
  TextInput,
  type Tone,
} from "@/components/ui/primitives";
import { getRiskState, updateRiskThresholds } from "@/lib/api";
import { cn } from "@/lib/cn";
import { formatDateTime, formatPctPlain } from "@/lib/format";
import type {
  NewsEvent,
  RiskAction,
  RiskAlertTelemetryPayload,
  RiskAssessmentRow,
  RiskStateResponse,
  RiskThresholds,
  TelemetryPayload,
} from "@/lib/types";

function isRiskAlertPayload(payload: TelemetryPayload): payload is RiskAlertTelemetryPayload {
  return "circuit_breaker_active" in payload;
}
import { useAsyncData } from "@/lib/use-async-data";
import { useNow } from "@/lib/use-now";

const ACTION_TONE: Record<RiskAction, Tone> = {
  pass: "gain",
  scale: "warn",
  liquidate: "loss",
  neutralized: "neutral",
};

const ACTION_LABEL: Record<RiskAction, string> = {
  pass: "Pass",
  scale: "Scaled",
  liquidate: "Liquidate",
  neutralized: "Neutralised",
};

const SEVERITY_TONE: Record<string, Tone> = {
  critical: "loss",
  warn: "warn",
  info: "accent",
  ok: "neutral",
};

/** View 3 — LLM Risk-First monitor. */
export function RiskView() {
  const filter = useRootScopeFilter();
  const { account, positions } = useAccount();
  const risk = useAsyncData(() => getRiskState(filter.query));
  const { latest } = useTelemetry();
  const now = useNow(1000);
  const [state, setState] = useState<RiskStateResponse | null>(null);

  useEffect(() => {
    if (risk.data) setState(risk.data);
  }, [risk.data]);

  useEffect(() => {
    if (!filter.hydrated) return;
    risk.reload();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filter.key, filter.hydrated, risk.reload]);

  useEffect(() => {
    if (filter.state.scope !== "strategy" || filter.state.sector || filter.state.roots.length > 0) {
      return;
    }
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
  }, [latest.risk_alert, filter.state.scope, filter.state.sector, filter.state.roots.length]);

  const nextTierOne = useMemo(
    () =>
      (state?.macro_events ?? [])
        .filter((event) => event.tier === 1)
        .sort((a, b) => new Date(a.event_time).getTime() - new Date(b.event_time).getTime())[0],
    [state],
  );

  if (risk.loading && !state) {
    return (
      <div className="grid grid-cols-12 gap-3">
        <Skeleton className="col-span-4 h-20" />
        <Skeleton className="col-span-4 h-20" />
        <Skeleton className="col-span-4 h-20" />
        <Skeleton className="col-span-8 h-96" />
        <Skeleton className="col-span-4 h-96" />
      </div>
    );
  }

  if (!state) {
    return (
      <Panel title="Risk">
        <EmptyState
          title="Risk state unavailable"
          description="The Risk-First API did not return a state payload and the deterministic fixture fallback could not be built."
        />
      </Panel>
    );
  }

  const lockout = state.macro_lockout;
  const neutralized = state.neutralized_instruments;
  const liquidations = state.assessments.filter((row) => row.action === "liquidate").length;

  const lockoutValue = lockout.active
    ? `T−${lockout.minutes_remaining}m`
    : nextTierOne && now
      ? compactCountdown(nextTierOne.event_time, now)
      : "—";

  const equityCurve = account?.equity_curve ?? [];
  const peakEquity = equityCurve.reduce((peak, point) => Math.max(peak, point.value), 0);
  const lastEquity = equityCurve[equityCurve.length - 1]?.value ?? 0;
  const trailingDrawdown = peakEquity > 0 ? Math.max(0, (peakEquity - lastEquity) / peakEquity) : null;
  const grossLeverage = account?.gross_leverage ?? 0;
  const grossCap = 3;
  const singleCap = 0.35;
  const largestWeight =
    account && account.net_liquidation > 0
      ? positions.reduce((largest, position) => Math.max(largest, Math.abs(position.notional) / account.net_liquidation), 0)
      : 0;
  const targetVol = state.thresholds.target_annual_volatility;
  const portfolioVol = account?.portfolio_volatility;
  const drawdownCap = 0.2;

  return (
    <div className="grid min-h-full grid-cols-12 grid-rows-[auto_auto_auto_minmax(0,1fr)] gap-3">
      <RootScopeFilterBar
        scope={filter.state.scope}
        sector={filter.state.sector}
        roots={filter.state.roots}
        onScopeChange={filter.setScope}
        onSectorChange={filter.setSector}
        onToggleRoot={filter.toggleRoot}
        onClearRoots={filter.clearRoots}
        onRefresh={() => risk.reload()}
        refreshing={risk.loading}
        selectedCount={state.assessments.length}
        dataOrigin={state.data_origin ?? null}
      />
      <Panel className="col-span-12" title="Quant Risk Thresholds" actions={<span className="text-[10.5px] text-zinc-600">deterministic · current vs limit</span>}>
        <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2 xl:grid-cols-5" data-testid="risk-hero-meters">
          <UtilizationMeter
            testId="risk-meter-margin"
            label="Initial margin"
            current={account?.margin_utilization ?? 0}
            limit={state.thresholds.max_margin_utilization}
            format={(value) => formatPctPlain(value)}
            plain="Portfolio IM must stay under the 60% hard cap; the translator scales orders down before breaching it."
          />
          <UtilizationMeter
            testId="risk-meter-daily-loss"
            label="Daily loss"
            current={Math.max(0, -(account?.day_pnl_pct ?? 0))}
            limit={state.thresholds.daily_max_loss_pct}
            format={(value) => formatPctPlain(value)}
            plain="Realised + unrealised drop from the day's starting NLV; the kill switch flattens at the limit."
          />
          <UtilizationMeter
            testId="risk-meter-vol"
            label="Volatility target"
            current={portfolioVol ?? null}
            limit={targetVol}
            format={(value) => formatPctPlain(value)}
            plain="Ex-ante book volatility vs the annualised target. The quant overlay scales gross weights to this budget."
          />
          <UtilizationMeter
            testId="risk-meter-gross"
            label="Gross / single-instrument caps"
            current={grossLeverage}
            limit={grossCap}
            format={(value) => `${value.toFixed(2)}×`}
            plain={`Gross leverage cap ${grossCap.toFixed(1)}×; largest single weight ${(largestWeight * 100).toFixed(1)}% vs ${(singleCap * 100).toFixed(0)}% cap.`}
            extra={{ label: "Largest single", util: largestWeight / singleCap, text: `${(largestWeight * 100).toFixed(1)}% / ${(singleCap * 100).toFixed(0)}%` }}
          />
          <UtilizationMeter
            testId="risk-meter-drawdown"
            label="Trailing DD + turbulence"
            current={trailingDrawdown}
            limit={drawdownCap}
            format={(value) => formatPctPlain(value)}
            plain="Peak-to-current equity drawdown vs the 20% trailing guard. Turbulence is not reported by this API payload."
            extra={{ label: "Turbulence", util: null, text: "n/a" }}
          />
        </div>
      </Panel>

      <div className="col-span-12 rounded-md border border-warn/30 bg-warn-soft/20 px-3 py-2 text-[11px] leading-5 text-amber-200" data-testid="llm-one-way-brake">
        <span className="font-medium text-amber-100">LLM Risk-First overlay · one-way brake.</span>{" "}
        The LLM overlay can only neutralise or scale exposure down — it never adds leverage or overrides the
        deterministic quant risk caps below. Confidence/dispersion filtering neutralises low-confidence signals.
      </div>

      {/* ------------------------------------------------------ status cards */}
      <KpiCard
        className="col-span-12 sm:col-span-4"
        label="Circuit breaker"
        value={`κ ${state.thresholds.llm_critical_score.toFixed(1)}`}
        footer={<Badge tone={state.circuit_breaker_active ? "loss" : "gain"}>{state.circuit_breaker_active ? "Active" : "Inactive"}</Badge>}
        hintClassName="truncate"
        hint={
          <span title={state.circuit_breaker_reason}>
            {state.circuit_breaker_active
              ? state.circuit_breaker_reason
              : `No root at or above κ with c ≥ ${state.thresholds.llm_confidence_threshold.toFixed(2)} · ${liquidations} forced flat`}
          </span>
        }
      />
      <KpiCard
        className="col-span-12 sm:col-span-4"
        label="Anti-hallucination"
        value={`τ ${state.thresholds.llm_confidence_threshold.toFixed(2)}`}
        footer={<Badge tone={neutralized.length > 0 ? "warn" : "gain"}>{neutralized.length} neutralised</Badge>}
        hintClassName="truncate"
        hint={
          neutralized.length > 0
            ? neutralized.join(", ")
            : "All LLM signals cleared the confidence and variance filter"
        }
      />
      <KpiCard
        className="col-span-12 sm:col-span-4"
        label={lockout.active ? "Macro lockout active" : "Next macro lockout"}
        value={lockoutValue}
        valueClassName={lockout.active ? "text-warn" : "text-zinc-50"}
        footer={<Badge tone={lockout.active ? "warn" : "neutral"}>{nextTierOne?.event ?? "No event"}</Badge>}
        hintClassName="truncate"
        hint={
          lockout.active && lockout.ends_at
            ? `Until ${formatDateTime(lockout.ends_at)} · entries flattened`
            : nextTierOne
              ? `${formatDateTime(nextTierOne.event_time)} · Tier ${nextTierOne.tier}`
              : "No Tier-1 releases scheduled"
        }
      />

      {/* ------------------------------------------------------- root cards */}
      <div className="col-span-12 flex min-h-0 flex-col gap-3 xl:col-span-8">
        <Panel
          title="Per-root assessments"
          actions={
            <div className="flex items-center gap-2 text-[11px] text-zinc-600">
              <span>{state.assessments.length} roots</span>
              {state.derived_roots && state.derived_roots.length > 0 ? (
                <Badge tone="warn" title="Assessed deterministically from neutral features (no live/cached snapshot)">
                  {state.derived_roots.length} derived
                </Badge>
              ) : null}
              {state.assessment_source ? (
                <span className="font-mono text-[9.5px] text-zinc-600">{state.assessment_source}</span>
              ) : null}
              <span className="text-zinc-700">·</span>
              <span className="flex items-center gap-1">
                <span className="h-1.5 w-1.5 rounded-full bg-gain" /> θ {state.thresholds.llm_safe_score.toFixed(1)}
              </span>
              <span className="flex items-center gap-1">
                <span className="h-1.5 w-1.5 rounded-full bg-loss" /> κ {state.thresholds.llm_critical_score.toFixed(1)}
              </span>
            </div>
          }
        >
          <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2 2xl:grid-cols-3">
            {state.assessments.map((assessment) => (
              <RootAssessmentCard key={assessment.instrument_id} assessment={assessment} thresholds={state.thresholds} />
            ))}
          </div>
        </Panel>

        <Panel title="Macro & news feed" className="min-h-0 flex-1" actions={<span className="text-[11px] text-zinc-600">{state.news.length} items</span>}>
          <div className="min-h-0 flex-1 space-y-0 overflow-y-auto">
            {state.news.map((event) => (
              <FeedRow key={event.id} event={event} />
            ))}
          </div>
        </Panel>
      </div>

      {/* --------------------------------------------------- right column */}
      <div className="col-span-12 flex min-h-0 flex-col gap-3 xl:col-span-4">
        <Panel title="Macro calendar" actions={<span className="text-[11px] text-zinc-600">Tier-1 lockout</span>}>
          <div className="divide-y divide-hairline/60">
            {state.macro_events.map((event) => {
              const minutes = now
                ? Math.max(0, Math.round((new Date(event.event_time).getTime() - now.getTime()) / 60_000))
                : event.minutes_until;
              const imminent = minutes <= 60;
              return (
                <div key={event.id} className="flex items-center justify-between gap-3 py-2">
                  <div className="min-w-0">
                    <div className="flex items-center gap-1.5">
                      <span className="truncate text-[12px] text-zinc-200">{event.event}</span>
                      <Badge tone={event.tier === 1 ? "warn" : "neutral"}>T{event.tier}</Badge>
                    </div>
                    <div className="truncate font-mono text-[10.5px] text-zinc-600">
                      {formatDateTime(event.event_time)}
                      {event.roots.length > 0 ? ` · ${event.roots.join(" ")}` : ""}
                    </div>
                  </div>
                  <span className={cn("shrink-0 font-mono text-[11px] tabular-nums", imminent ? "text-warn" : "text-zinc-500")}>
                    {minutes < 60 ? `${minutes}m` : `${Math.floor(minutes / 60)}h ${minutes % 60}m`}
                  </span>
                </div>
              );
            })}
          </div>
        </Panel>

        <div className="min-h-0 flex-1">
          <ThresholdForm
            thresholds={state.thresholds}
            onApplied={(next) => setState((previous) => (previous ? { ...previous, thresholds: next } : previous))}
          />
        </div>
      </div>
    </div>
  );
}

/** Short horizon countdown for the status cards: `14d 02h`, `2h 05m`, `12m`. */
function compactCountdown(iso: string, now: Date): string {
  const target = new Date(iso).getTime();
  if (Number.isNaN(target)) return "—";
  const delta = Math.max(0, target - now.getTime());
  const days = Math.floor(delta / 86_400_000);
  const hours = Math.floor((delta % 86_400_000) / 3_600_000);
  const minutes = Math.floor((delta % 3_600_000) / 60_000);
  if (days > 0) return `${days}d ${hours.toString().padStart(2, "0")}h`;
  if (hours > 0) return `${hours}h ${minutes.toString().padStart(2, "0")}m`;
  return `${minutes}m`;
}

const UtilizationMeter = memo(function UtilizationMeter({
  testId,
  label,
  current,
  limit,
  format,
  plain,
  extra,
}: {
  testId: string;
  label: string;
  current: number | null;
  limit: number;
  format: (value: number) => string;
  plain: string;
  extra?: { label: string; util: number | null; text: string };
}) {
  const util = current !== null && limit > 0 ? current / limit : null;
  const tone = util === null ? "neutral" : util >= 0.9 ? "loss" : util >= 0.7 ? "warn" : "gain";
  const color = tone === "loss" ? "bg-loss" : tone === "warn" ? "bg-warn" : "bg-gain";
  return (
    <div className="rounded-md border border-edge bg-canvas p-2.5" data-testid={testId} data-util={util === null ? "" : util.toFixed(4)}>
      <div className="flex items-start justify-between gap-2">
        <span className="text-[10.5px] font-medium uppercase tracking-[0.04em] text-zinc-500">{label}</span>
        <Badge tone={tone}>{util === null ? "n/a" : `${(util * 100).toFixed(0)}%`}</Badge>
      </div>
      <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-zinc-800">
        <div className={cn("h-full rounded-full", color)} style={{ width: `${Math.min(100, (util ?? 0) * 100).toFixed(1)}%` }} />
      </div>
      <div className="mt-1.5 flex items-baseline justify-between font-mono text-[11px] tabular-nums">
        <span className="text-zinc-200">{current === null ? "—" : format(current)}</span>
        <span className="text-zinc-600">limit {format(limit)}</span>
      </div>
      <p className="mt-1 text-[10px] leading-4 text-zinc-600">{plain}</p>
      {extra ? (
        <p className="mt-0.5 font-mono text-[9.5px] text-zinc-600">
          {extra.label}: {extra.text}
        </p>
      ) : null}
    </div>
  );
});

function RootAssessmentCard({
  assessment,
  thresholds,
}: {
  assessment: RiskAssessmentRow;
  thresholds: RiskThresholds;
}) {
  const confidenceBelow = assessment.confidence < thresholds.llm_confidence_threshold;
  return (
    <div className="rounded-md border border-edge bg-canvas p-2.5">
      <div className="mb-2 flex items-center justify-between gap-2">
        <span className="flex items-center gap-1.5">
          <span className="font-mono text-[12.5px] font-semibold text-zinc-100">{assessment.instrument_id}</span>
          <span
            className="rounded border border-edge px-1 text-[9px] text-zinc-500"
            title="Regime context inferred by the LLM overlay for this root (e.g. risk-on, risk-off, neutral)."
          >
            Regime: {assessment.regime ?? "n/a"}
          </span>
        </span>
        <Badge tone={ACTION_TONE[assessment.action]}>{ACTION_LABEL[assessment.action]}</Badge>
      </div>

      <div className="mb-2.5">
        <div className="mb-1 flex items-baseline justify-between">
          <span className="text-[10.5px] text-zinc-500">Semantic risk S_i,t</span>
          <span
            className={cn(
              "font-mono text-[11px] tabular-nums",
              assessment.semantic_risk_score >= thresholds.llm_critical_score
                ? "text-loss"
                : assessment.semantic_risk_score > thresholds.llm_safe_score
                  ? "text-warn"
                  : "text-gain",
            )}
          >
            {assessment.semantic_risk_score.toFixed(2)}
          </span>
        </div>
        <RiskMeter
          value={assessment.semantic_risk_score}
          safe={thresholds.llm_safe_score}
          critical={thresholds.llm_critical_score}
        />
      </div>

      <div className="grid grid-cols-3 items-end gap-2">
        <div>
          <div className="text-[10.5px] text-zinc-500">Confidence</div>
          <div className={cn("font-mono text-[12px] tabular-nums", confidenceBelow ? "text-warn" : "text-zinc-200")}>
            {assessment.confidence.toFixed(2)}
            {confidenceBelow ? <span className="ml-1 text-[9.5px] text-warn">{"< τ"}</span> : null}
          </div>
          <div
            className="mt-0.5 font-mono text-[9px] text-zinc-600"
            title="Dispersion across sampled LLM completions: high dispersion lowers confidence and can neutralise this signal."
          >
            disp {assessment.confidence_dispersion ?? assessment.dispersion ?? "n/a"}
          </div>
          <div className="relative mt-1 h-1 w-full overflow-hidden rounded-full bg-zinc-800">
            <div
              className={`h-full rounded-full ${confidenceBelow ? "bg-warn/80" : "bg-gain/80"}`}
              style={{ width: `${Math.round(assessment.confidence * 100)}%` }}
            />
            <span
              className="absolute inset-y-0 w-px bg-zinc-300/70"
              style={{ left: `${Math.round(thresholds.llm_confidence_threshold * 100)}%` }}
              title={`τ ${thresholds.llm_confidence_threshold.toFixed(2)}`}
            />
          </div>
        </div>
        <div>
          <div className="text-[10.5px] text-zinc-500">Weight ×</div>
          <div
            className="font-mono text-[12px] tabular-nums text-zinc-200"
            title="Deterministic weight multiplier applied to this root: 1.000 = unchanged; below 1.000 = scaled down; 0 = neutralised/liquidated by the LLM brake."
          >
            {assessment.weight_multiplier.toFixed(3)}
          </div>
        </div>
        <div className="text-right">
          <div className="text-[10.5px] text-zinc-500">Sentiment</div>
          <div className="font-mono text-[12px] tabular-nums text-zinc-300">
            {assessment.sentiment_score.toFixed(1)}
          </div>
        </div>
      </div>

      <p className="mt-2 line-clamp-2 text-[11px] leading-4 text-zinc-500" title={assessment.reasoning_summary}>
        {assessment.reasoning_summary}
      </p>
    </div>
  );
}

function FeedRow({ event }: { event: NewsEvent }) {
  return (
    <article className="flex gap-3 border-b border-hairline/60 py-2.5 last:border-b-0">
      <div className="w-14 shrink-0 pt-0.5">
        <div className="font-mono text-[10.5px] tabular-nums text-zinc-600">
          {new Date(event.timestamp).toLocaleTimeString("en-GB", {
            hour: "2-digit",
            minute: "2-digit",
            timeZone: "UTC",
          })}
        </div>
        <Badge tone={SEVERITY_TONE[event.severity] ?? "neutral"} className="mt-1">
          {event.severity}
        </Badge>
      </div>
      <div className="min-w-0 flex-1">
        <p className="text-[12.5px] leading-5 text-zinc-200">{event.headline}</p>
        <p className="mt-0.5 text-[11.5px] leading-5 text-zinc-500">{event.summary}</p>
        <div className="mt-1 flex items-center gap-2 text-[10.5px] text-zinc-600">
          <span className="font-mono">{event.source}</span>
          {event.roots.map((root) => (
            <span key={root} className="rounded border border-edge px-1 font-mono text-[10px]">
              {root}
            </span>
          ))}
          <span className="ml-auto font-mono">sentiment {event.sentiment_score.toFixed(1)}</span>
        </div>
      </div>
    </article>
  );
}

function ThresholdForm({
  thresholds,
  onApplied,
}: {
  thresholds: RiskThresholds;
  onApplied: (thresholds: RiskThresholds) => void;
}) {
  const [draft, setDraft] = useState<RiskThresholds>(thresholds);
  const [dirty, setDirty] = useState(false);
  const [status, setStatus] = useState<"idle" | "saving" | "saved" | "demo" | "error">("idle");
  const [message, setMessage] = useState<string | null>(null);

  useEffect(() => {
    if (!dirty) setDraft(thresholds);
  }, [thresholds, dirty]);

  const validation = useMemo(() => {
    if (draft.llm_critical_score <= draft.llm_safe_score) return "κ must exceed θ_safe";
    if (draft.llm_confidence_threshold < 0 || draft.llm_confidence_threshold > 1)
      return "τ must be within 0–1";
    return null;
  }, [draft]);

  const update = (key: keyof RiskThresholds, value: string) => {
    const parsed = Number(value);
    if (Number.isNaN(parsed)) return;
    setDraft((previous) => ({ ...previous, [key]: parsed }));
    setDirty(true);
    setStatus("idle");
    setMessage(null);
  };

  const apply = async () => {
    if (validation) return;
    setStatus("saving");
    const result = await updateRiskThresholds({
      llm_confidence_threshold: draft.llm_confidence_threshold,
      llm_safe_score: draft.llm_safe_score,
      llm_critical_score: draft.llm_critical_score,
      daily_max_loss_pct: draft.daily_max_loss_pct,
      target_annual_volatility: draft.target_annual_volatility,
      max_margin_utilization: draft.max_margin_utilization,
    });
    if (result.source === "live") {
      onApplied(result.data);
      setDraft(result.data);
      setDirty(false);
      setStatus("saved");
      setMessage("Thresholds applied to the risk overlay.");
    } else {
      setStatus("demo");
      setMessage("Backend unreachable — overrides kept in this session only.");
    }
  };

  const reset = () => {
    setDraft(thresholds);
    setDirty(false);
    setStatus("idle");
    setMessage(null);
  };

  return (
    <Panel
      title="Risk thresholds"
      className="h-full"
      actions={
        <span className="text-[11px] text-zinc-600">
          {status === "saved" ? <span className="text-emerald-400">Applied</span> : "PUT /api/risk/thresholds"}
        </span>
      }
    >
      <div className="grid grid-cols-2 gap-2.5">
        <Field label="Confidence τ" htmlFor="risk-tau" hint="c < τ neutralises signal">
          <TextInput
            id="risk-tau"
            type="number"
            step={0.01}
            value={draft.llm_confidence_threshold}
            onChange={(value) => update("llm_confidence_threshold", value)}
          />
        </Field>
        <Field label="Safe score θ" htmlFor="risk-theta" hint="benign semantic risk">
          <TextInput
            id="risk-theta"
            type="number"
            step={0.1}
            value={draft.llm_safe_score}
            onChange={(value) => update("llm_safe_score", value)}
          />
        </Field>
        <Field label="Critical κ" htmlFor="risk-kappa" hint="force flat at or above">
          <TextInput
            id="risk-kappa"
            type="number"
            step={0.1}
            value={draft.llm_critical_score}
            onChange={(value) => update("llm_critical_score", value)}
          />
        </Field>
        <Field label="Daily loss cap" htmlFor="risk-daily" hint="fraction of NLV">
          <TextInput
            id="risk-daily"
            type="number"
            step={0.005}
            value={draft.daily_max_loss_pct}
            onChange={(value) => update("daily_max_loss_pct", value)}
          />
        </Field>
        <Field label="Volatility target" htmlFor="risk-vol" hint="annualised">
          <TextInput
            id="risk-vol"
            type="number"
            step={0.01}
            value={draft.target_annual_volatility}
            onChange={(value) => update("target_annual_volatility", value)}
          />
        </Field>
        <Field label="Margin cap η_max" htmlFor="risk-margin" hint="fraction of NLV">
          <TextInput
            id="risk-margin"
            type="number"
            step={0.01}
            value={draft.max_margin_utilization}
            onChange={(value) => update("max_margin_utilization", value)}
          />
        </Field>
      </div>

      <div className="mt-2 flex items-center justify-between gap-2 text-[11px]">
        <span className={cn(validation ? "text-loss" : "text-zinc-600")}>
          {validation ?? `Margin warning ${formatPctPlain(draft.margin_warning_threshold)} · critical ${formatPctPlain(draft.margin_critical_threshold)}`}
        </span>
        {message ? (
          <span className={status === "saved" ? "text-emerald-400" : "text-amber-400"}>{message}</span>
        ) : null}
      </div>

      <div className="mt-2.5 flex items-center justify-end gap-2 border-t border-hairline pt-2.5">
        <Button variant="ghost" onClick={reset} disabled={status === "saving" || !dirty}>
          Reset
        </Button>
        <Button variant="primary" onClick={() => void apply()} disabled={validation !== null || status === "saving" || !dirty}>
          {status === "saving" ? "Applying…" : "Apply"}
        </Button>
      </div>
    </Panel>
  );
}
