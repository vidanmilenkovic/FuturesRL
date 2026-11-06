"use client";

import { memo, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";

import { useTelemetry } from "@/components/providers/telemetry-provider";
import { Badge, Button } from "@/components/ui/primitives";
import { cn } from "@/lib/cn";
import { getLLMStatus, pingLLM } from "@/lib/api";
import type { LLMStatusResponse } from "@/lib/types";
import { useAsyncData } from "@/lib/use-async-data";

const FALLBACK: LLMStatusResponse = {
  provider: "unknown",
  model: "unknown",
  api_key_configured: false,
  mode: "offline_stub",
  status: "unreachable",
  last_latency_ms: null,
  last_checked_at: null,
  cache_entries: 0,
  error_detail: "LLM status unavailable.",
};

const STATUS_META: Record<
  string,
  { label: string; tone: "gain" | "warn" | "loss" | "neutral"; className: string }
> = {
  online: {
    label: "LLM ONLINE",
    tone: "gain",
    className: "border-gain/40 bg-gain-soft/40 text-emerald-300 hover:bg-gain-soft/70",
  },
  offline_stub: {
    label: "LLM OFFLINE STUB",
    tone: "warn",
    className: "border-warn/30 bg-warn-soft/30 text-amber-300 hover:bg-warn-soft/60",
  },
  auth_error: {
    label: "LLM AUTH ERROR",
    tone: "loss",
    className: "border-loss/40 bg-loss-soft/40 text-rose-300 hover:bg-loss-soft/70",
  },
  unreachable: {
    label: "LLM UNREACHABLE",
    tone: "loss",
    className: "border-loss/40 bg-loss-soft/40 text-rose-300 hover:bg-loss-soft/70",
  },
};

/**
 * Global LLM health pill. State comes from /health.llm_status plus the
 * dedicated /api/risk/llm-status endpoint; ping is explicit and never inferred.
 */
export const LLMStatusPill = memo(function LLMStatusPill() {
  const { health } = useTelemetry();
  const statusSource = useAsyncData(getLLMStatus);
  const [override, setOverride] = useState<LLMStatusResponse | null>(null);
  const [open, setOpen] = useState(false);
  const [pinging, setPinging] = useState(false);
  const [pingError, setPingError] = useState<string | null>(null);
  const buttonRef = useRef<HTMLButtonElement | null>(null);
  const [anchor, setAnchor] = useState<{ top: number; right: number } | null>(null);

  const status: LLMStatusResponse = override ?? statusSource.data ?? health?.llm_status ?? FALLBACK;
  const meta = useMemo(
    () => STATUS_META[status.status] ?? STATUS_META.unreachable ?? { label: status.status, tone: "loss" as const, className: "border-loss/40 bg-loss-soft/40 text-rose-300" },
    [status.status],
  );

  const runPing = async () => {
    setPinging(true);
    setPingError(null);
    const result = await pingLLM();
    setPinging(false);
    if (result.source === "live" || result.data) {
      setOverride(result.data);
    }
    if (result.source !== "live") setPingError(result.error ?? result.data.error_detail ?? "LLM ping failed.");
    statusSource.reload();
  };

  return (
    <div className="relative" data-testid="llm-pill-wrapper">
      <button
        type="button"
        data-testid="llm-status-pill"
        data-status={status.status}
        ref={buttonRef}
        onClick={() => {
          const rect = buttonRef.current?.getBoundingClientRect();
          if (rect) {
            setAnchor({
              top: rect.bottom + 6,
              right: Math.max(8, window.innerWidth - rect.right),
            });
          }
          setOpen((previous) => !previous);
        }}
        aria-expanded={open}
        title={`${status.provider} / ${status.model} · ${status.status}${status.error_detail ? ` · ${status.error_detail}` : ""}`}
        className={cn(
          "focus-ring inline-flex h-6 items-center gap-1.5 rounded-md border px-2 text-[10.5px] font-medium transition-colors",
          meta.className,
        )}
      >
        <span
          className={cn(
            "h-1.5 w-1.5 rounded-full",
            meta.tone === "gain" ? "bg-emerald-400" : meta.tone === "warn" ? "bg-amber-400" : "bg-rose-400",
          )}
        />
        {meta.label}
      </button>
      {open && anchor && typeof document !== "undefined"
        ? createPortal(
        <div
          data-testid="llm-pill-popover"
          style={{ position: "fixed", top: anchor.top, right: anchor.right }}
          className="z-[60] w-[300px] rounded-md border border-edge bg-panel p-3 text-[11px] shadow-popover"
        >
          <div className="flex items-start justify-between gap-2">
            <div>
              <p className="text-[12px] font-medium text-zinc-200">LLM Risk-First connection</p>
              <p className="mt-0.5 font-mono text-[10px] text-zinc-500">
                {status.provider} / {status.model}
              </p>
            </div>
            <button
              type="button"
              aria-label="Close LLM status"
              onClick={() => setOpen(false)}
              className="focus-ring grid h-5 w-5 place-items-center rounded text-zinc-500 hover:bg-surface hover:text-zinc-300"
            >
              ×
            </button>
          </div>
          <div className="mt-2 space-y-1 border-t border-hairline pt-2 text-[10.5px] text-zinc-400">
            <div className="flex justify-between"><span className="text-zinc-600">Mode</span><span className="font-mono">{status.mode}</span></div>
            <div className="flex justify-between"><span className="text-zinc-600">API key</span><span>{status.api_key_configured ? "configured" : "missing"}</span></div>
            <div className="flex justify-between"><span className="text-zinc-600">Latency</span><span className="font-mono">{status.last_latency_ms === null ? "—" : `${status.last_latency_ms.toFixed(1)} ms`}</span></div>
            <div className="flex justify-between"><span className="text-zinc-600">Cache entries</span><span className="font-mono">{status.cache_entries}</span></div>
            <div className="flex justify-between"><span className="text-zinc-600">Last checked</span><span className="font-mono">{status.last_checked_at ? new Date(status.last_checked_at).toLocaleTimeString() : "never"}</span></div>
          </div>
          {status.error_detail ? (
            <p className="mt-2 rounded border border-warn/30 bg-warn-soft/20 px-2 py-1 text-[10px] leading-4 text-amber-300">
              {status.error_detail}
            </p>
          ) : null}
          {pingError ? (
            <p role="alert" data-testid="llm-ping-error" className="mt-2 rounded border border-loss/30 bg-loss-soft/30 px-2 py-1 text-[10px] text-rose-300">
              {pingError}
            </p>
          ) : null}
          <div className="mt-2 flex items-center justify-between gap-2">
            <Button variant="secondary" size="xs" testId="llm-ping-button" onClick={() => void runPing()} disabled={pinging}>
              {pinging ? "Testing…" : "Test LLM Connection"}
            </Button>
            <Badge tone={meta.tone}>{status.status}</Badge>
          </div>
        </div>,
        document.body,
      )
        : null}
    </div>
  );
});
