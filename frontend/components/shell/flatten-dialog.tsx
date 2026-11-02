"use client";

import { useState } from "react";

import { flattenAll } from "@/lib/api";
import { formatCurrency, formatDateTime, formatInteger } from "@/lib/format";

import { Button, Checkbox } from "../ui/primitives";
import { Modal } from "../ui/dialog";
import type { FlattenAllResponse, PositionRow } from "@/lib/types";

/**
 * Two-step flatten confirmation: the catalogue of positions is listed, the
 * operator must acknowledge the market-order impact, and only then does the
 * rose confirm button enable. The result (receipts + execution lock) is shown
 * in-place; the account snapshot is refreshed on completion.
 */
export function FlattenDialog({
  open,
  positions,
  onClose,
  onFlattened,
}: {
  open: boolean;
  positions: PositionRow[];
  onClose: () => void;
  onFlattened: () => void;
}) {
  const [acknowledged, setAcknowledged] = useState(false);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<FlattenAllResponse | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const reset = () => {
    setAcknowledged(false);
    setBusy(false);
    setResult(null);
    setNote(null);
    setError(null);
  };

  const close = () => {
    if (busy) return;
    reset();
    onClose();
  };

  const execute = async () => {
    setBusy(true);
    setError(null);
    try {
      const response = await flattenAll();
      setResult(response.data);
      if (response.source === "demo") {
        setNote("Backend unreachable — flatten was simulated against deterministic demo state.");
      }
      onFlattened();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Modal
      open={open}
      title="Flatten all positions"
      description="Market orders close every open futures leg and cancel working orders. This action cannot be undone."
      onClose={close}
      busy={busy}
      width="max-w-lg"
      footer={
        result ? (
          <Button variant="secondary" onClick={close}>
            Done
          </Button>
        ) : (
          <>
            <Button variant="ghost" onClick={close} disabled={busy}>
              Cancel
            </Button>
            <Button
              variant="danger"
              onClick={() => void execute()}
              disabled={busy || positions.length === 0 || !acknowledged}
            >
              {busy ? "Flattening…" : `Flatten ${formatInteger(positions.length)} position${positions.length === 1 ? "" : "s"}`}
            </Button>
          </>
        )
      }
    >
      {result ? (
        <div className="space-y-3">
          <div className="flex items-center justify-between rounded-md border border-hairline bg-canvas px-3 py-2">
            <span className="text-[12px] text-zinc-400">Status</span>
            <span className="text-[12px] font-medium text-emerald-300">
              {result.status === "flattened" ? "Flattened" : result.status === "no_positions" ? "No positions" : "Halted"}
            </span>
          </div>
          <div className="grid grid-cols-3 gap-2 text-center">
            <div className="rounded-md border border-hairline bg-canvas px-2 py-2">
              <div className="metric-label">Positions</div>
              <div className="font-mono text-[16px] font-semibold text-zinc-100">{result.flattened_positions}</div>
            </div>
            <div className="rounded-md border border-hairline bg-canvas px-2 py-2">
              <div className="metric-label">Cancelled</div>
              <div className="font-mono text-[16px] font-semibold text-zinc-100">{result.cancelled_orders}</div>
            </div>
            <div className="rounded-md border border-hairline bg-canvas px-2 py-2">
              <div className="metric-label">Receipts</div>
              <div className="font-mono text-[16px] font-semibold text-zinc-100">{result.receipts.length}</div>
            </div>
          </div>
          {result.receipts.length > 0 ? (
            <div className="max-h-44 overflow-y-auto rounded-md border border-hairline">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Root</th>
                    <th className="text-right">Qty</th>
                    <th>Side</th>
                    <th className="text-right">Fill</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {result.receipts.map((receipt) => (
                    <tr key={receipt.order_id}>
                      <td className="font-medium text-zinc-200">{receipt.instrument_id}</td>
                      <td className="num text-right">{receipt.quantity}</td>
                      <td className={receipt.side === "SELL" ? "text-loss" : "text-gain"}>{receipt.side}</td>
                      <td className="num text-right text-zinc-400">
                        {receipt.average_price === null ? "—" : receipt.average_price.toFixed(2)}
                      </td>
                      <td className="text-[11.5px] text-zinc-400">{receipt.status}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : null}
          <p className="text-[11.5px] leading-5 text-zinc-500">
            {result.message}
            {result.execution_locked_until ? ` · execution locked until ${formatDateTime(result.execution_locked_until)}` : ""}
          </p>
          {note ? <p className="text-[11.5px] text-amber-300">{note}</p> : null}
        </div>
      ) : (
        <div className="space-y-3">
          <div className="max-h-48 overflow-y-auto rounded-md border border-hairline">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Root</th>
                  <th>Contract</th>
                  <th className="text-right">Qty</th>
                  <th className="text-right">Notional</th>
                  <th>Side</th>
                </tr>
              </thead>
              <tbody>
                {positions.map((position) => (
                  <tr key={position.instrument_id}>
                    <td className="font-medium text-zinc-200">{position.instrument_id}</td>
                    <td className="text-zinc-500">{position.contract_symbol ?? "—"}</td>
                    <td className={`num text-right ${position.quantity > 0 ? "text-gain" : "text-loss"}`}>
                      {position.quantity > 0 ? `+${position.quantity}` : position.quantity}
                    </td>
                    <td className="num text-right text-zinc-400">{formatCurrency(position.notional, true)}</td>
                    <td className="text-[11.5px] text-zinc-400">{position.quantity > 0 ? "Buy to close" : "Sell to close"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {positions.length === 0 ? (
            <p className="text-[12px] text-zinc-500">No open positions to flatten.</p>
          ) : null}
          <Checkbox
            id="flatten-ack"
            checked={acknowledged}
            onChange={setAcknowledged}
            disabled={busy || positions.length === 0}
            label="I understand this sends market orders to close every position and locks execution."
          />
          {error ? <p className="text-[11.5px] text-rose-400">{error}</p> : null}
          {note ? <p className="text-[11.5px] text-amber-300">{note}</p> : null}
        </div>
      )}
    </Modal>
  );
}
