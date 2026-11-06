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
import type {
  ConnectionState,
  HealthResponse,
  TelemetryEnvelope,
  TelemetryKind,
} from "@/lib/types";

interface TelemetryContextValue {
  /** Socket state (`demo` = deterministic local simulator driving the UI). */
  status: ConnectionState;
  /** REST reachability of the FastAPI backend. */
  backend: "checking" | "online" | "offline";
  /** Last `/health` payload (environment + offline_mode drive the mode pill). */
  health: HealthResponse | null;
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

function isEditableElement(target: EventTarget | null): boolean {
  return (
    target instanceof HTMLElement &&
    (target.matches("input, select, textarea") || target.isContentEditable)
  );
}

export function TelemetryProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<ConnectionState>("connecting");
  const [backend, setBackend] = useState<"checking" | "online" | "offline">("checking");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [latest, setLatest] = useState<Partial<Record<TelemetryKind, TelemetryEnvelope>>>({});
  const [events, setEvents] = useState<TelemetryEnvelope[]>([]);
  const [lastUpdate, setLastUpdate] = useState<string | null>(null);
  const connectionRef = useRef<TelemetryConnection | null>(null);
  const [reconnectNonce, setReconnectNonce] = useState(0);
  // Workstation UX: never replace drawer/modal/form state mid-interaction. Any
  // open modal/drawer emits pause events; focused form controls pause globally.
  const pauseCountRef = useRef(0);
  const inputFocusedRef = useRef(false);
  const updatesPaused = () => pauseCountRef.current > 0 || inputFocusedRef.current;

  useEffect(() => {
    const pause = () => {
      pauseCountRef.current += 1;
    };
    const resume = () => {
      pauseCountRef.current = Math.max(0, pauseCountRef.current - 1);
    };
    const focusIn = (event: FocusEvent) => {
      if (isEditableElement(event.target)) inputFocusedRef.current = true;
    };
    const focusOut = () => {
      window.setTimeout(() => {
        inputFocusedRef.current = isEditableElement(document.activeElement);
      }, 0);
    };
    window.addEventListener("futuresrlx:pause-telemetry", pause);
    window.addEventListener("futuresrlx:resume-telemetry", resume);
    document.addEventListener("focusin", focusIn);
    document.addEventListener("focusout", focusOut);
    return () => {
      window.removeEventListener("futuresrlx:pause-telemetry", pause);
      window.removeEventListener("futuresrlx:resume-telemetry", resume);
      document.removeEventListener("focusin", focusIn);
      document.removeEventListener("focusout", focusOut);
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    const probe = () => {
      getHealth()
        .then((result) => {
          if (cancelled || updatesPaused()) return;
          setBackend(result.source === "live" ? "online" : "offline");
          setHealth(result.data);
        })
        .catch(() => {
          if (!cancelled) setBackend("offline");
        });
    };
    probe();
    // Re-probe periodically so the data-mode badge flips when the backend or an
    // IBKR session comes up/down without a page reload.
    const timer = setInterval(probe, 15_000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [reconnectNonce]);

  useEffect(() => {
    connectionRef.current = connectTelemetry({
      onMessage: (message) => {
        if (updatesPaused()) return;
        setLatest((previous) => ({ ...previous, [message.kind]: message }));
        setEvents((previous) => [message, ...previous].slice(0, MAX_EVENTS));
        setLastUpdate(message.timestamp);
      },
      onStatus: (next) => {
        if (updatesPaused()) return;
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
    setHealth(null);
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
      health,
      latest,
      events,
      lastUpdate,
      sequence,
      isDemo: status === "demo" || backend === "offline",
      retry,
    };
  }, [status, backend, health, latest, events, lastUpdate, retry]);

  return <TelemetryContext.Provider value={value}>{children}</TelemetryContext.Provider>;
}

export function useTelemetry(): TelemetryContextValue {
  const context = useContext(TelemetryContext);
  if (!context) {
    throw new Error("useTelemetry must be used inside <TelemetryProvider>");
  }
  return context;
}
