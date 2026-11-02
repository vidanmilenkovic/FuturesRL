"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";

import { connectTelemetry, getHealth, type TelemetryConnection } from "@/lib/api";
import type { ConnectionState, TelemetryEnvelope, TelemetryKind } from "@/lib/types";

interface TelemetryContextValue {
  /** Socket state (`demo` = deterministic local simulator driving the UI). */
  status: ConnectionState;
  /** REST reachability of the FastAPI backend. */
  backend: "checking" | "online" | "offline";
  /** Latest envelope per telemetry kind. */
  latest: Partial<Record<TelemetryKind, TelemetryEnvelope>>;
  /** Rolling event log (newest first, capped at 60). */
  events: TelemetryEnvelope[];
  lastUpdate: string | null;
  sequence: number;
  isDemo: boolean;
  retry: () => void;
}

const TelemetryContext = createContext<TelemetryContextValue | null>(null);

const MAX_EVENTS = 60;

export function TelemetryProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<ConnectionState>("connecting");
  const [backend, setBackend] = useState<"checking" | "online" | "offline">("checking");
  const [latest, setLatest] = useState<Partial<Record<TelemetryKind, TelemetryEnvelope>>>({});
  const [events, setEvents] = useState<TelemetryEnvelope[]>([]);
  const [lastUpdate, setLastUpdate] = useState<string | null>(null);
  const connectionRef = useRef<TelemetryConnection | null>(null);
  const [reconnectNonce, setReconnectNonce] = useState(0);

  useEffect(() => {
    let cancelled = false;
    getHealth()
      .then((result) => {
        if (cancelled) return;
        setBackend(result.source === "live" ? "online" : "offline");
      })
      .catch(() => {
        if (!cancelled) setBackend("offline");
      });
    return () => {
      cancelled = true;
    };
  }, [reconnectNonce]);

  useEffect(() => {
    connectionRef.current = connectTelemetry({
      onMessage: (message) => {
        setLatest((previous) => ({ ...previous, [message.kind]: message }));
        setEvents((previous) => [message, ...previous].slice(0, MAX_EVENTS));
        setLastUpdate(message.timestamp);
      },
      onStatus: (next) => {
        setStatus(next);
        if (next === "offline") setBackend("offline");
      },
    });
    return () => {
      connectionRef.current?.close();
      connectionRef.current = null;
    };
  }, [reconnectNonce]);

  const retry = useCallback(() => {
    connectionRef.current?.close();
    setStatus("connecting");
    setBackend("checking");
    setReconnectNonce((nonce) => nonce + 1);
  }, []);

  const value = useMemo<TelemetryContextValue>(() => {
    const sequence =
      events.length > 0
        ? Math.max(...events.map((event) => event.sequence ?? 0))
        : 0;
    return {
      status,
      backend,
      latest,
      events,
      lastUpdate,
      sequence,
      isDemo: status === "demo" || backend === "offline",
      retry,
    };
  }, [status, backend, latest, events, lastUpdate, retry]);

  return <TelemetryContext.Provider value={value}>{children}</TelemetryContext.Provider>;
}

export function useTelemetry(): TelemetryContextValue {
  const context = useContext(TelemetryContext);
  if (!context) {
    throw new Error("useTelemetry must be used inside <TelemetryProvider>");
  }
  return context;
}
