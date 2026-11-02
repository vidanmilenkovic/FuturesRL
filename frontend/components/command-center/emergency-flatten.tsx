"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ConfirmDialog } from "@/components/ui/confirm-dialog";
import { flattenAll } from "@/lib/api";
import { cn } from "@/lib/cn";
import { formatDateTime, formatInteger } from "@/lib/format";
import type { FlattenAllResponse, PositionRow } from "@/lib/types";

const ARM_DELAY_MS = 1_400;
const REQUIRED_PHRASE = "FLATTEN";

/**
 * Hardware-style EMERGENCY FLATTEN ALL kill switch.
 *
 * Three independent guards before an order can be sent:
 *   1. explicit modal confirmation,
 *   2. typed safety phrase (`FLATTEN`),
 *   3. a short arming delay so the button cannot be triggered by a stray
 *      double-enter / key repeat.
 * While the flatten is in flight the control is optimistically disabled and the
 * UI reports each execution stage; receipts are displayed afterwards for audit.
 */
export function EmergencyFlatten({ positions }: { positions: PositionRow[] }) {
  const [open, setOpen] = useState(false);
  const [phrase, setPhrase] = useState("");
  const [armed, setArmed] = useState(false);
  const [busy, setBusy] = useState(false);
  const [stage, setStage] = useState(0);
  const [result, setResult] = useState<FlattenAllResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const stageTimer = useRef<ReturnType<typeof setInterval> | null>(null);

  const totalLegs = positions.length;
  const totalContracts = positions.reduce((sum, position) => sum + Math.abs(position.quantity), 0);
  const phraseReady = phrase.trim().toUpperCase() === REQUIRED_PHRASE;

  useEffect(() => {
    if (!phraseReady || busy) {
      setArmed(false);
      return;
    }
    const timer = setTimeout(() => setArmed(true), ARM_DELAY_MS);
    return () => {
      clearTimeout(timer);
      setArmed(false);
    };
  }, [phraseReady, busy]);

  useEffect(() => {
    return () => {
      if (stageTimer.current !== null) clearInterval(stageTimer.current);
    };
  }, []);

  const stages = useMemo(
    () => [
      "LOCKING OMS · CANCELLING WORKING ORDERS",
      "TRANSMITTING MKT FLATTEN LEGS",
      "VERIFYING BROKER POSITIONS",
    ],
    [],
  );

  const openDialog = () => {
    setPhrase("");
    setArmed(false);
    setError(null);
    setResult(null);
    setStage(0);
    setOpen(true);
  };

  const closeDialog = () => {
    if (!busy) setOpen(false);
  };

  const execute = async () => {
    setBusy(true);
    setError(null);
    setStage(0);
    stageTimer.current = setInterval(() => {
      setStage((current) => Math.min(current + 1, stages.length - 1));
    }, 900);
    try {
      const response = await flattenAll();
      setResult(response.data);
      if (response.source === "demo") {
        setError("Backend unreachable — flatten was simulated against demo fixtures.");
      }
    } catch (flattenError) {
      setError(flattenError instanceof Error ? flattenError.message : String(flattenError));
    } finally {
      if (stageTimer.current !== null) {
        clearInterval(stageTimer.current);
        stageTimer.current = null;
      }
      setBusy(false);
      setOpen(false);
      setPhrase("");
    }
  };

  const flattened = result !== null && result.status === "flattened";

  return (
    <div className="relative overflow-hidden rounded-md border border-signal-critical/40 bg-gradient-to-b from-signal-critical/[0.07] to-terminal-850/60 p-3">
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 opacity-[0.07]"
        style={{
          backgroundImage:
            "repeating-linear-gradient(135deg, #f43f5e 0px, #f43f5e 10px, transparent 10px, transparent 20px)",
        }}
      />

      <div className="relative flex flex-wrap items-start justify-between gap-2">
        <div>
          <p className="flex items-center gap-2 font-mono text-[10px] font-semibold uppercase tracking-[0.2em] text-signal-critical">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-signal-critical" />
            kill switch · trading halt
          </p>
          <p className="mt-0.5 font-mono text-[10px] uppercase tracking-[0.08em] text-terminal-400">
            market orders · all roots · cancels working orders first
          </p>
        </div>
        <Badge tone={flattened ? "up" : "critical"} dot>
          {flattened ? "FLATTEN EXECUTED" : "ARMED"}
        </Badge>
      </div>

      <button
        type="button"
        onClick={openDialog}
        disabled={busy || flattened}
        aria-label="Emergency flatten all positions"
        className={cn(
          "group relative mt-3 flex w-full flex-col items-center justify-center gap-1 overflow-hidden rounded-sm border-2 px-4 py-4 transition-all duration-150 focus-ring",
          "border-signal-critical/70 bg-gradient-to-b from-signal-critical/25 to-signal-critical/[0.08]",
          "hover:from-signal-critical/35 hover:to-signal-critical/10 hover:shadow-glow-critical active:translate-y-px",
          "disabled:cursor-not-allowed disabled:opacity-60",
          busy ? "animate-pulse-critical" : "",
        )}
      >
        <span
          aria-hidden
          className="absolute inset-x-0 top-0 h-1 bg-[repeating-linear-gradient(45deg,#f43f5e_0,#f43f5e_8px,#0d1520_8px,#0d1520_16px)] opacity-70"
        />
        <span
          aria-hidden
          className="absolute inset-x-0 bottom-0 h-1 bg-[repeating-linear-gradient(45deg,#f43f5e_0,#f43f5e_8px,#0d1520_8px,#0d1520_16px)] opacity-70"
        />
        <span className="flex items-center gap-2 font-mono text-base font-bold uppercase tracking-[0.22em] text-signal-critical sm:text-lg">
          <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth={1.8} aria-hidden>
            <path d="M12 3 3 20h18L12 3Z" strokeLinejoin="round" />
            <path d="M12 9v5M12 17h.01" strokeLinecap="round" />
          </svg>
          {busy ? "FLATTENING…" : "EMERGENCY FLATTEN ALL"}
        </span>
        <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-signal-critical/80">
          {busy
            ? stages[stage]
            : `${formatInteger(totalLegs)} legs · ${formatInteger(totalContracts)} contracts · two-step confirm`}
        </span>
      </button>

      {busy ? (
        <div className="mt-2" aria-live="polite">
          <div className="h-0.5 w-full overflow-hidden rounded bg-terminal-700">
            <div
              className="h-full bg-signal-critical transition-all duration-700"
              style={{ width: `${((stage + 1) / stages.length) * 100}%` }}
            />
          </div>
        </div>
      ) : null}

      {error ? (
        <p className="mt-2 rounded-sm border border-signal-warn/40 bg-signal-warn/10 px-2 py-1.5 font-mono text-[10px] uppercase tracking-[0.06em] text-signal-warn">
          {error}
        </p>
      ) : null}

      {result ? (
        <div className="mt-2 space-y-2">
          <div className="flex flex-wrap items-center gap-2 font-mono text-[10px] uppercase tracking-[0.08em] text-terminal-300">
            <Badge tone={flattened ? "up" : "warn"}>{result.status}</Badge>
            <span>{result.flattened_positions} positions flattened</span>
            <span className="text-terminal-500">·</span>
            <span>{result.cancelled_orders} orders cancelled</span>
            <span className="ml-auto text-terminal-500">{formatDateTime(result.timestamp)}</span>
          </div>
          <div className="max-h-40 overflow-y-auto rounded-sm border border-terminal-600/50">
            <table className="data-table">
              <thead>
                <tr>
                  <th>order</th>
                  <th>root</th>
                  <th className="num">qty</th>
                  <th>side</th>
                  <th className="num">avg px</th>
                  <th>status</th>
                </tr>
              </thead>
              <tbody>
                {result.receipts.map((receipt) => (
                  <tr key={receipt.order_id}>
                    <td className="text-terminal-400">{receipt.order_id}</td>
                    <td className="text-terminal-100">{receipt.instrument_id}</td>
                    <td className="num">{receipt.quantity}</td>
                    <td className={receipt.side === "SELL" ? "text-signal-down" : "text-signal-up"}>
                      {receipt.side}
                    </td>
                    <td className="num">{receipt.average_price?.toFixed(2) ?? "—"}</td>
                    <td>
                      <Badge tone={receipt.status === "FILLED" ? "up" : receipt.status === "SUBMITTED" ? "info" : "critical"}>
                        {receipt.status}
                      </Badge>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <Button variant="outline" size="sm" onClick={() => setResult(null)}>
            Reset kill switch
          </Button>
        </div>
      ) : null}

      <ConfirmDialog
        open={open}
        title="Emergency flatten all futures positions"
        description={
          <span className="space-y-2">
            <span className="block">
              This transmits <strong className="text-signal-critical">market orders</strong> to close{" "}
              <strong className="text-terminal-100">{formatInteger(totalLegs)} legs</strong> (
              {formatInteger(totalContracts)} contracts), cancelling all working orders first. It cannot be
              undone and is not subject to the pre-trade margin gate.
            </span>
            <span className="block text-terminal-400">
              Normal risk overlays are bypassed for the kill switch; execution lock engages until a manual
              reset from the Risk Monitor.
            </span>
          </span>
        }
        requirePhrase={REQUIRED_PHRASE}
        phrase={phrase}
        onPhraseChange={setPhrase}
        confirmLabel={armed ? "Execute flatten" : "Arming…"}
        busy={busy}
        confirmDisabled={!armed}
        onConfirm={() => {
          void execute();
        }}
        onClose={closeDialog}
      >
        <div className="rounded-sm border border-terminal-600/60 bg-terminal-900/70 px-3 py-2">
          <p className="mono-label mb-1.5">target legs</p>
          <div className="flex flex-wrap gap-1.5">
            {positions.map((position) => (
              <span
                key={position.instrument_id}
                className="inline-flex items-center gap-1 rounded-sm border border-terminal-500/60 bg-terminal-800 px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-[0.08em] text-terminal-200"
              >
                {position.instrument_id}
                <span className={position.quantity > 0 ? "text-signal-up" : "text-signal-down"}>
                  {position.quantity > 0 ? "+" : ""}
                  {position.quantity}
                </span>
              </span>
            ))}
          </div>
          <p className="mt-2 font-mono text-[10px] uppercase tracking-[0.1em] text-signal-warn" aria-live="polite">
            {busy
              ? stages[stage]
              : armed
                ? "ARMED — confirm within this dialog to transmit"
                : phraseReady
                  ? "arming delay…"
                  : `type ${REQUIRED_PHRASE} to arm`}
          </p>
        </div>
      </ConfirmDialog>
    </div>
  );
}
