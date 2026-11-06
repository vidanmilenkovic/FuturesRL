"use client";

import { memo, useEffect, useMemo, useState } from "react";

import { useAccount } from "@/components/providers/account-provider";
import { Badge, Button, Field, TextInput } from "@/components/ui/primitives";
import { getAccount, getHealth, getIbkrPorts, switchIbkrConnection } from "@/lib/api";
import { cn } from "@/lib/cn";
import type { IbkrPortInfo, IbkrPortsResponse } from "@/lib/types";

const LIVE_PORTS = new Set([7496, 4001]);

export const IbkrConnectionManager = memo(function IbkrConnectionManager({
  open,
}: {
  open: boolean;
}) {
  const { reload: reloadAccount } = useAccount();
  const [ports, setPorts] = useState<IbkrPortsResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [selectedPort, setSelectedPort] = useState<number | null>(null);
  const [clientId, setClientId] = useState("");
  const [liveConfirm, setLiveConfirm] = useState("");
  const [switching, setSwitching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<string | null>(null);

  const load = async () => {
    setLoading(true);
    const response = await getIbkrPorts();
    setLoading(false);
    if (response.source === "live") {
      setPorts(response.data);
      setSelectedPort((current) => current ?? response.data.active?.port ?? null);
      if (response.data.active?.client_id !== null && response.data.active?.client_id !== undefined) {
        setClientId(String(response.data.active.client_id));
      }
    } else {
      setPorts(response.data);
      setError(response.error ?? "IBKR port probe unavailable.");
    }
  };

  useEffect(() => {
    if (!open) return;
    setError(null);
    setResult(null);
    setLiveConfirm("");
    setSelectedPort(null);
    void load();
  }, [open]);

  const selected = useMemo<IbkrPortInfo | null>(
    () => ports?.ports.find((port) => port.port === selectedPort) ?? null,
    [ports, selectedPort],
  );
  const liveSelected = Boolean(selected && LIVE_PORTS.has(selected.port));
  const needsConfirmation = liveSelected;

  const apply = async () => {
    if (!selected) return;
    if (needsConfirmation && liveConfirm !== "LIVE") {
      setError('Live ports require the exact confirmation string "LIVE".');
      return;
    }
    setSwitching(true);
    setError(null);
    setResult(null);
    const parsedClient = clientId.trim() === "" ? null : Number(clientId);
    if (parsedClient !== null && (!Number.isInteger(parsedClient) || parsedClient < 0)) {
      setSwitching(false);
      setError("Client ID must be a non-negative integer.");
      return;
    }
    const response = await switchIbkrConnection({
      port: selected.port,
      client_id: parsedClient,
      confirmation: needsConfirmation ? liveConfirm : null,
    });
    setSwitching(false);
    if (response.source !== "live") {
      setError(response.error ?? response.data.message ?? "IBKR connection switch failed.");
      return;
    }
    const data = response.data;
    setResult(`${data.status} · ${data.message}`);
    setLiveConfirm("");
    await Promise.allSettled([load(), getHealth(), getAccount(), reloadAccount()]);
  };

  return (
    <div className="rounded-md border border-edge bg-canvas/60 p-2" data-testid="ibkr-connection-manager">
      <div className="flex items-start justify-between gap-2">
        <div>
          <p className="text-[11.5px] font-medium text-zinc-200">IBKR connection</p>
          <p className="mt-0.5 text-[10px] leading-4 text-zinc-600">
            Probe TWS / IB Gateway ports and apply a guarded target switch. Runtime switches may require an API restart.
          </p>
        </div>
        <Button variant="ghost" size="xs" testId="ibkr-ports-refresh" onClick={() => void load()} disabled={loading}>
          {loading ? "Probing…" : "Probe"}
        </Button>
      </div>
      {ports ? (
        <>
          <div className="mt-1.5 flex flex-wrap items-center gap-2 text-[9.5px] text-zinc-600">
            <span className="font-mono">{ports.host}</span>
            {ports.active ? (
              <Badge tone="accent">
                active {ports.active.port ?? "—"} · client {ports.active.client_id ?? "—"} · {ports.active.environment ?? "?"}
              </Badge>
            ) : (
              <Badge tone="warn">no active binding</Badge>
            )}
            {ports.probed_at ? <span>probed {new Date(ports.probed_at).toLocaleTimeString()}</span> : null}
          </div>
          <div className="mt-1.5 grid grid-cols-2 gap-1" data-testid="ibkr-port-list">
            {ports.ports.map((port) => (
              <button
                key={port.port}
                type="button"
                data-testid={`ibkr-port-${port.port}`}
                data-reachable={port.reachable ? "true" : "false"}
                onClick={() => setSelectedPort(port.port)}
                className={cn(
                  "focus-ring flex items-center justify-between gap-1 rounded border px-1.5 py-1 text-left text-[10px] transition-colors",
                  selectedPort === port.port
                    ? "border-accent/40 bg-accent-soft/40 text-blue-200"
                    : "border-edge bg-canvas text-zinc-400 hover:bg-surface",
                )}
              >
                <span className="font-mono">{port.port}</span>
                <span className="text-[9px] uppercase text-zinc-600">{port.mode} · {port.gateway.replace("ib_gateway", "gateway")}</span>
                <Badge tone={port.reachable ? "gain" : "neutral"}>
                  {port.reachable ? `${port.latency_ms?.toFixed(0) ?? "?"}ms` : "closed"}
                </Badge>
              </button>
            ))}
          </div>
          <div className="mt-1.5 grid grid-cols-2 gap-1.5">
            <Field label="Client ID" htmlFor="ibkr-client-id">
              <TextInput id="ibkr-client-id" type="number" min={0} value={clientId} onChange={setClientId} placeholder="keep current" />
            </Field>
            {needsConfirmation ? (
              <Field label='Live confirmation ("LIVE")' htmlFor="ibkr-live-confirm">
                <TextInput id="ibkr-live-confirm" value={liveConfirm} onChange={setLiveConfirm} placeholder="LIVE" />
              </Field>
            ) : (
              <div className="flex items-end pb-1 text-[9.5px] text-zinc-600">
                Paper ports need no confirmation.
              </div>
            )}
          </div>
          <div className="mt-1.5 flex items-center justify-between gap-2">
            <span className="text-[9.5px] text-zinc-600">
              {selected ? `Target ${selected.port} · ${selected.mode} · ${selected.gateway}` : "Select a port"}
            </span>
            <Button
              variant={liveSelected ? "danger" : "secondary"}
              size="xs"
              testId="ibkr-switch-button"
              onClick={() => void apply()}
              disabled={!selected || switching || (liveSelected && liveConfirm !== "LIVE")}
              title={
                liveSelected && ports.active?.environment === "paper"
                  ? "Live ports require LIVE confirmation; the API refuses them in a paper environment (409)"
                  : "Apply the guarded IBKR connection target"
              }
            >
              {switching ? "Switching…" : liveSelected ? "Switch to LIVE" : "Apply target"}
            </Button>
          </div>
          {liveSelected && ports.active?.environment === "paper" ? (
            <p className="mt-1 text-[9.5px] text-amber-300">
              API environment is paper — a LIVE switch will be refused with 409; the result is shown below.
            </p>
          ) : null}
          {ports.warnings.length > 0 ? <p className="mt-1 text-[9.5px] text-amber-300">{ports.warnings.join(" · ")}</p> : null}
        </>
      ) : null}
      {error ? (
        <p role="alert" data-testid="ibkr-switch-error" className="mt-1.5 rounded border border-loss/30 bg-loss-soft/30 px-1.5 py-1 text-[9.5px] text-rose-300">
          {error}
        </p>
      ) : null}
      {result ? (
        <p role="status" data-testid="ibkr-switch-result" className="mt-1.5 rounded border border-gain/30 bg-gain-soft/30 px-1.5 py-1 text-[9.5px] text-emerald-300">
          {result}
        </p>
      ) : null}
    </div>
  );
});
