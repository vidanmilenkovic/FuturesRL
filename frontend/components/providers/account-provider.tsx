"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

import { getAccount } from "@/lib/api";
import { mergeTelemetryIntoSeries } from "@/lib/equity-series";
import type { AccountSnapshotResponse, DataMode, PositionRow } from "@/lib/types";

import { useTelemetry } from "./telemetry-provider";

interface AccountContextValue {
  account: AccountSnapshotResponse | null;
  positions: PositionRow[];
  loading: boolean;
  /** `live` = FastAPI answered, `demo` = deterministic fixture fallback. */
  source: "live" | "demo" | "loading";
  error: string | null;
  /** Ribbon provenance state (UI FALLBACK / OFFLINE ENGINE / IBKR · RECONNECTING / PAPER / LIVE). */
  dataMode: DataMode;
  /** True when the account book is a last-known-good snapshot while IBKR reconnects. */
  stale: boolean;
  staleReason: string | null;
  reload: () => void;
}

const AccountContext = createContext<AccountContextValue | null>(null);

/**
 * Shared account snapshot used by the workspace ribbon ticker and the Command
 * Center. One REST fetch at mount, then live telemetry frames (`pnl`, `margin`,
 * `positions`) are merged in so every surface shows the same numbers.
 */
export function AccountProvider({ children }: { children: ReactNode }) {
  const { latest, backend, health, status } = useTelemetry();
  const [account, setAccount] = useState<AccountSnapshotResponse | null>(null);
  const [source, setSource] = useState<"live" | "demo" | "loading">("loading");
  const [error, setError] = useState<string | null>(null);
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    let cancelled = false;
    getAccount()
      .then((result) => {
        if (cancelled) return;
        setAccount(result.data);
        setSource(result.source);
        setError(result.error);
      })
      .catch((cause: unknown) => {
        if (!cancelled) {
          setError(cause instanceof Error ? cause.message : String(cause));
          setSource("demo");
        }
      });
    return () => {
      cancelled = true;
    };
  }, [nonce]);

  // Poll the account snapshot so a slow/initially-unavailable IBKR ledger, a
  // backend restart or a runner bridge transition cannot leave the dashboard
  // stuck on fixture values.
  useEffect(() => {
    const timer = setInterval(() => setNonce((value) => value + 1), 15_000);
    return () => clearInterval(timer);
  }, []);

  // Merge streaming frames into the account snapshot.
  useEffect(() => {
    // If the REST snapshot fell back to fixtures but the WS is connected to a
    // *different* backend, mixing the two sources would fabricate an equity jump.
    // Keep the deterministic fixture account coherent in that edge case.
    if (source === "demo" && status === "live") return;
    const pnl = latest.pnl;
    const margin = latest.margin;
    const positions = latest.positions;
    if (!pnl && !margin && !positions) return;
    // IBKR-owned books are REST-authoritative: stream frames may refresh PnL and
    // margin, but must never replace the broker position book (stale last-good).
    const ibkrManaged = Boolean(health?.ibkr_enabled);
    setAccount((previous) => {
      if (!previous) return previous;
      const next: AccountSnapshotResponse = { ...previous };
      if (pnl && "net_liquidation" in pnl.payload) {
        next.net_liquidation = pnl.payload.net_liquidation;
        next.day_pnl = pnl.payload.day_pnl;
        next.day_pnl_pct = pnl.payload.day_pnl_pct;
        // The provider keeps a *daily* series: live ticks update the last daily
        // bar in place and never append intraday timestamps (chart-safe).
        next.equity_curve = mergeTelemetryIntoSeries(
          previous.equity_curve,
          pnl.payload.point,
          "1M",
        );
      }
      if (margin && "margin_utilization" in margin.payload) {
        next.initial_margin_used = margin.payload.initial_margin_used;
        next.maintenance_margin_used = margin.payload.maintenance_margin_used;
        next.margin_utilization = margin.payload.margin_utilization;
        next.excess_liquidity = next.net_liquidation - margin.payload.initial_margin_used;
      }
      if (positions && "positions" in positions.payload && !ibkrManaged) {
        next.positions = positions.payload.positions;
        next.gross_leverage = positions.payload.gross_leverage;
        next.net_exposure = positions.payload.net_exposure;
      }
      return next;
    });
  }, [latest.pnl, latest.margin, latest.positions, source, status, health?.ibkr_enabled]);

  const reload = useCallback(() => setNonce((value) => value + 1), []);

  // When IBKR is enabled, a fixture book must never be shown. If the REST snapshot
  // failed (e.g. 503 while the session is being established), surface an honest
  // empty state and let the polling loop fetch the real book.
  useEffect(() => {
    if (!health?.ibkr_enabled) return;
    if (source === "demo" && account !== null) {
      setAccount(null);
    }
  }, [health?.ibkr_enabled, source, account]);

  // REST-authoritative refresh cadence for IBKR-owned accounts.
  useEffect(() => {
    if (!health?.ibkr_enabled) return;
    const period = account?.stale ? 5_000 : 15_000;
    const timer = setInterval(reload, period);
    return () => clearInterval(timer);
  }, [health?.ibkr_enabled, account?.stale, reload]);

  const dataMode = useMemo<DataMode>(() => {
    if (backend === "checking") {
      return {
        key: "connecting",
        label: "CONNECTING",
        description: "Probing the telemetry API and WebSocket stream",
      };
    }
    if (backend !== "online" || !health) {
      return {
        key: "ui_fallback",
        label: "UI FALLBACK",
        description:
          "FastAPI unreachable — deterministic fixtures and a local WebSocket simulator drive the UI",
      };
    }
    if (health.ibkr_enabled && !health.ibkr_connected) {
      return {
        key: "reconnecting",
        label: "IBKR · RECONNECTING",
        description:
          health.ibkr_last_error
            ? `IBKR session lost — reconnecting (${health.ibkr_last_error}); book is last-known-good`
            : "IBKR session lost — reconnect probe is running; book is last-known-good",
      };
    }
    if (health.ibkr_connected && health.environment === "live") {
      return {
        key: "live_ibkr",
        label: "LIVE · IBKR",
        description: "Live IBKR execution session connected (API-owned session)",
      };
    }
    if (health.ibkr_connected && health.environment === "paper") {
      return {
        key: "paper_ibkr",
        label: "PAPER · IBKR",
        description: "IBKR paper execution session connected (API-owned session)",
      };
    }
    // A separate TradingRunner publishes a runner bridge; a fresh bridge in
    // paper/live mode means the app IS connected to IBKR even when the API's own
    // execution service stays in offline mode.
    if (health.runner_bridge_available && !health.runner_bridge_stale) {
      if (health.runner_mode === "live") {
        return {
          key: "live_ibkr",
          label: "LIVE · IBKR",
          description: "Live IBKR runner bridge publishing fresh state",
        };
      }
      if (health.runner_mode === "paper") {
        return {
          key: "paper_ibkr",
          label: "PAPER · IBKR",
          description: "IBKR paper runner bridge publishing fresh state",
        };
      }
    }
    if (health.ibkr_enabled && !health.ibkr_connected) {
      return {
        key: "offline_engine",
        label: "OFFLINE ENGINE",
        description:
          health.ibkr_last_error
            ? `IBKR enabled but disconnected (${health.ibkr_last_error}); serving the last-known-good book`
            : "IBKR enabled but disconnected; serving the last-known-good book",
      };
    }
    if (health.offline_mode) {
      return {
        key: "offline_engine",
        label: "OFFLINE ENGINE",
        description:
          "API online — deterministic offline engine (no live IBKR session attached)",
      };
    }
    return {
      key: "offline_engine",
      label: "OFFLINE ENGINE",
      description:
        "API online — no IBKR session connected; the runner bridge may still be active",
    };
  }, [backend, health]);

  const value = useMemo<AccountContextValue>(
    () => ({
      account,
      positions: account?.positions ?? [],
      loading: source === "loading" && account === null,
      source,
      error,
      dataMode,
      stale: Boolean(account?.stale),
      staleReason: account?.stale_reason ?? null,
      reload,
    }),
    [account, source, error, dataMode, reload],
  );

  return <AccountContext.Provider value={value}>{children}</AccountContext.Provider>;
}

export function useAccount(): AccountContextValue {
  const context = useContext(AccountContext);
  if (!context) {
    throw new Error("useAccount must be used inside <AccountProvider>");
  }
  return context;
}
