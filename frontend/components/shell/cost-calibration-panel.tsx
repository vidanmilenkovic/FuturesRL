"use client";

import { memo, useEffect, useState } from "react";

import { Badge, Button } from "@/components/ui/primitives";
import { calibrateCosts, getCalibratedCosts } from "@/lib/api";
import type { CalibratedCostsResponse } from "@/lib/types";

export const CostCalibrationPanel = memo(function CostCalibrationPanel({ open }: { open: boolean }) {
  const [costs, setCosts] = useState<CalibratedCostsResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  const load = async () => {
    setLoading(true);
    const result = await getCalibratedCosts();
    setLoading(false);
    setCosts(result.data);
    if (result.source !== "live") setError(result.error ?? "Calibrated costs API unavailable.");
    else setError(null);
  };

  useEffect(() => {
    if (!open) return;
    setError(null);
    setMessage(null);
    void load();
  }, [open]);

  const calibrate = async () => {
    setBusy(true);
    setError(null);
    setMessage(null);
    const result = await calibrateCosts();
    setBusy(false);
    if (result.source !== "live") {
      setError(result.error ?? result.data.message ?? "IBKR calibration failed.");
      return;
    }
    setMessage(`${result.data.status} · ${Object.keys(result.data.roots).length} roots calibrated`);
    await load();
  };

  const rootCount = costs ? Object.keys(costs.roots).length : 0;
  return (
    <div className="rounded-md border border-edge bg-canvas/60 p-2" data-testid="cost-calibration-panel">
      <div className="flex items-start justify-between gap-2">
        <div>
          <p className="text-[11.5px] font-medium text-zinc-200">Fees & margins calibration</p>
          <p className="mt-0.5 text-[10px] leading-4 text-zinc-600">
            Read-only IBKR whatIfOrder calibration for per-contract commissions and margins. No order is submitted.
          </p>
        </div>
        <span data-testid="calibration-badge">
          <Badge tone={costs?.available ? "gain" : "warn"}>
            {costs?.available && costs.calibrated_at
              ? `IBKR Calibrated (${new Date(costs.calibrated_at).toLocaleString()})`
              : "YAML Default Estimates"}
          </Badge>
        </span>
      </div>
      <div className="mt-1.5 flex flex-wrap items-center gap-2 text-[9.5px] text-zinc-600">
        <span>{rootCount} roots calibrated</span>
        {costs?.account_id ? <span>account {costs.account_id}</span> : null}
        {costs?.age_seconds !== null && costs?.age_seconds !== undefined ? (
          <span>{Math.round(costs.age_seconds / 60)}m old</span>
        ) : null}
        {costs?.source ? <span>source {costs.source}</span> : null}
      </div>
      <div className="mt-1.5 flex items-center gap-2">
        <Button variant="secondary" size="xs" testId="calibrate-costs-button" onClick={() => void calibrate()} disabled={busy || loading}>
          {busy ? "Calibrating…" : "Calibrate Fees & Margins from IBKR"}
        </Button>
        <Button variant="ghost" size="xs" testId="calibration-refresh" onClick={() => void load()} disabled={loading || busy}>
          Refresh
        </Button>
      </div>
      {costs?.warnings && costs.warnings.length > 0 ? (
        <p className="mt-1.5 text-[9.5px] leading-4 text-amber-300" data-testid="calibration-warnings">
          {costs.warnings.join(" · ")}
        </p>
      ) : null}
      {message ? (
        <p role="status" data-testid="calibration-message" className="mt-1.5 rounded border border-gain/30 bg-gain-soft/30 px-1.5 py-1 text-[9.5px] text-emerald-300">
          {message}
        </p>
      ) : null}
      {error ? (
        <p role="alert" data-testid="calibration-error" className="mt-1.5 rounded border border-loss/30 bg-loss-soft/30 px-1.5 py-1 text-[9.5px] text-rose-300">
          {error}
        </p>
      ) : null}
    </div>
  );
});
