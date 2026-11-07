"use client";

import { memo, useEffect, useRef, useState } from "react";

import { cn } from "@/lib/cn";
import { resetPaperState, resetRunnerBridge } from "@/lib/api";
import type { PaperStateResetResult } from "@/lib/types";

import { MarketDataCacheButton } from "@/components/data/market-data-cache";
import { CostCalibrationPanel } from "./cost-calibration-panel";
import { IbkrConnectionManager } from "./ibkr-connection-manager";

import { useTelemetry } from "../providers/telemetry-provider";
import { Modal } from "../ui/dialog";
import { Badge, Button, Field, TextInput } from "../ui/primitives";

type OperatorAction = "paper" | "runner";

/**
 * Developer/operator settings drawer. Destructive resets are guarded by an
 * exact phrase + checkbox; paper-state clearing is unavailable in live.
 */
export const OperatorDrawer = memo(function OperatorDrawer({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const { health } = useTelemetry();
  const [action, setAction] = useState<OperatorAction | null>(null);
  const [confirm, setConfirm] = useState("");
  const [ack, setAck] = useState(false);
  const [clearRiskState, setClearRiskState] = useState(true);
  const [clearRunnerBridge, setClearRunnerBridge] = useState(true);
  const [clearPaperCache, setClearPaperCache] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [receipt, setReceipt] = useState<string | null>(null);

  const wasOpenRef = useRef(false);
  useEffect(() => {
    if (open) {
      window.dispatchEvent(new Event("futuresrl:pause-telemetry"));
      if (!wasOpenRef.current) {
        setAction(null);
        setConfirm("");
        setAck(false);
        setClearRiskState(true);
        setClearRunnerBridge(true);
        setClearPaperCache(true);
        setBusy(false);
        setError(null);
        setReceipt(null);
      }
    }
    wasOpenRef.current = open;
    return () => {
      if (open) window.dispatchEvent(new Event("futuresrl:resume-telemetry"));
    };
  }, [open]);

  const environment = health?.environment ?? "unknown";
  const live = environment === "live";
  const actionTitle = action === "paper" ? "Clear paper state" : "Reset runner bridge";
  const phrase = action === "paper" ? "RESET PAPER STATE" : "RESET RUNNER BRIDGE";

  const start = (next: OperatorAction) => {
    setAction(next);
    setConfirm("");
    setAck(false);
    setError(null);
    setReceipt(null);
  };

  const closeDialog = () => {
    if (busy) return;
    setAction(null);
    setError(null);
  };

  const run = async () => {
    if (!action) return;
    setBusy(true);
    setError(null);
    setReceipt(null);
    if (action === "paper") {
      const result = await resetPaperState({
        confirm,
        clearRiskState,
        clearRunnerBridge,
        clearPaperCache,
      });
      setBusy(false);
      if (result.source === "live" && result.data.ok) {
        setReceipt(formatPaperReceipt(result.data));
        setAction(null);
      } else {
        setError(result.data.message || `Clear paper state failed${result.error ? `: ${result.error}` : "."}`);
      }
      return;
    }
    const result = await resetRunnerBridge();
    setBusy(false);
    if (result.source === "live" && result.data.ok) {
      setReceipt(result.data.message || "Runner bridge reset.");
      setAction(null);
    } else {
      setError(result.data.message || `Runner reset failed${result.error ? `: ${result.error}` : "."}`);
    }
  };

  return (
    <>
      <div
        className={cn(
          "absolute bottom-7 right-0 z-40 flex w-[360px] max-w-[92vw] flex-col rounded-tl-md border border-edge bg-panel shadow-popover transition-transform duration-150",
          open ? "translate-y-0" : "pointer-events-none translate-y-2 opacity-0",
        )}
        style={{ maxHeight: open ? 540 : 0, overflow: "hidden" }}
        aria-hidden={!open}
        data-testid="operator-drawer"
      >
        <div className="flex h-9 shrink-0 items-center justify-between border-b border-hairline px-3">
          <div className="flex items-center gap-2">
            <span className="text-[12px] font-medium text-zinc-200">System</span>
            <Badge tone={live ? "loss" : "warn"}>{environment}</Badge>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="focus-ring rounded px-1.5 py-0.5 text-[11px] text-zinc-500 hover:bg-surface hover:text-zinc-200"
          >
            Close
          </button>
        </div>
        <div className="min-h-0 flex-1 space-y-2 overflow-y-auto p-3">
          <p className="text-[10.5px] leading-4 text-zinc-600">
            Developer/operator maintenance actions. All resets require an exact confirmation phrase and are logged
            by the API.
          </p>
          <div className="rounded-md border border-edge bg-canvas/60 p-2">
            <div className="flex items-start justify-between gap-2">
              <div>
                <p className="text-[11.5px] font-medium text-zinc-200">Clear paper state</p>
                <p className="mt-0.5 text-[10px] leading-4 text-zinc-600">
                  Clears the runner bridge snapshot, persisted risk-state store and paper cache. Non-live
                  environments only.
                </p>
              </div>
              <Button
                variant="secondary"
                size="xs"
                testId="operator-clear-paper-state"
                disabled={live}
                onClick={() => start("paper")}
                title={live ? "Paper state reset is unavailable in live" : "Clear paper state"}
              >
                Clear
              </Button>
            </div>
            {live ? (
              <p className="mt-1 text-[10px] text-rose-300">Unavailable in live — the API refuses paper resets.</p>
            ) : null}
          </div>
          <div className="rounded-md border border-edge bg-canvas/60 p-2">
            <div className="flex items-start justify-between gap-2">
              <div>
                <p className="text-[11.5px] font-medium text-zinc-200">Reset runner bridge</p>
                <p className="mt-0.5 text-[10px] leading-4 text-zinc-600">
                  Closes the singleton offline runner session. The bridge snapshot is retained.
                </p>
              </div>
              <Button
                variant="secondary"
                size="xs"
                testId="operator-reset-runner"
                onClick={() => start("runner")}
              >
                Reset
              </Button>
            </div>
          </div>
          <IbkrConnectionManager open={open} />
          <CostCalibrationPanel open={open} />
          <div className="rounded-md border border-edge bg-canvas/60 p-2">
            <div className="flex items-start justify-between gap-2">
              <div>
                <p className="text-[11.5px] font-medium text-zinc-200">Market Data Cache</p>
                <p className="mt-0.5 text-[10px] leading-4 text-zinc-600">
                  Inspect cached Parquet roots, estimate Databento cost and backfill IBKR/Databento daily bars.
                </p>
              </div>
              <MarketDataCacheButton label="Open" className="shrink-0" testId="operator-market-data-cache" />
            </div>
          </div>
          {receipt ? (
            <div
              role="status"
              data-testid="operator-receipt"
              className="rounded border border-gain/30 bg-gain-soft/30 px-2 py-1.5 text-[10.5px] leading-4 text-emerald-300"
            >
              {receipt}
            </div>
          ) : null}
        </div>
      </div>

      <Modal
        open={action !== null}
        title={actionTitle}
        description={
          action === "paper"
            ? "This clears persisted paper/risk state. It is idempotent but destructive for local session data."
            : "This closes the offline runner session. Reconnect/restart it to resume stepping."
        }
        onClose={closeDialog}
        busy={busy}
        footer={
          <>
            <Button variant="ghost" onClick={closeDialog} disabled={busy}>
              Cancel
            </Button>
            <Button
              variant="danger"
              testId={action === "paper" ? "confirm-clear-paper-state" : "confirm-reset-runner"}
              disabled={busy || confirm.trim() !== phrase || !ack}
              onClick={() => void run()}
            >
              {busy ? "Resetting…" : action === "paper" ? "Clear paper state" : "Reset runner bridge"}
            </Button>
          </>
        }
      >
        <div className="space-y-2.5">
          <label className="flex items-start gap-2 text-[11px] leading-4 text-zinc-400">
            <input
              type="checkbox"
              checked={ack}
              onChange={(event) => setAck(event.target.checked)}
              className="mt-0.5 h-3 w-3 rounded border-zinc-600 bg-canvas accent-red-500"
              data-testid="operator-ack"
            />
            I understand this reset is destructive for the current paper/offline session.
          </label>
          {action === "paper" ? (
            <div className="space-y-1.5 rounded border border-edge bg-canvas/50 p-2">
              {[
                ["clear_risk_state", "Clear persisted risk state", clearRiskState, setClearRiskState],
                ["clear_runner_bridge", "Clear runner bridge snapshot", clearRunnerBridge, setClearRunnerBridge],
                ["clear_paper_cache", "Clear paper cache", clearPaperCache, setClearPaperCache],
              ].map(([key, label, value, setter]) => (
                <label key={String(key)} className="flex items-center gap-2 text-[10.5px] text-zinc-400">
                  <input
                    type="checkbox"
                    checked={Boolean(value)}
                    onChange={(event) =>
                      (setter as (checked: boolean) => void)(event.target.checked)
                    }
                    className="h-3 w-3 rounded border-zinc-600 bg-canvas accent-blue-500"
                    data-testid={`operator-flag-${key}`}
                  />
                  {String(label)}
                </label>
              ))}
            </div>
          ) : null}
          <Field label={`Type ${phrase} to confirm`} htmlFor="operator-confirm">
            <TextInput
              id="operator-confirm"
              value={confirm}
              onChange={setConfirm}
              placeholder={phrase}
              disabled={busy}
            />
          </Field>
          {error ? (
            <div
              role="alert"
              data-testid="operator-error"
              className="rounded border border-loss/40 bg-loss-soft/40 px-2 py-1.5 text-[11px] text-rose-300"
            >
              {error}
            </div>
          ) : null}
        </div>
      </Modal>
    </>
  );
});

function formatPaperReceipt(result: PaperStateResetResult): string {
  const receipts = result.receipts;
  if (!receipts || Array.isArray(receipts)) {
    return result.message || "Paper state reset.";
  }
  const files = Array.isArray(receipts.files_removed) ? receipts.files_removed.length : 0;
  const store = receipts.risk_state_store_cleared === true ? "store cleared" : "store unchanged";
  const session = receipts.paper_session_reset === true ? "session reset" : "session unchanged";
  return `${result.message} · ${files} file(s) removed · ${store} · ${session}`;
}
