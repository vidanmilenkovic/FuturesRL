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
import type { AccountSnapshotResponse, PositionRow } from "@/lib/types";

import { useTelemetry } from "./telemetry-provider";

interface AccountContextValue {
  account: AccountSnapshotResponse | null;
  positions: PositionRow[];
  loading: boolean;
  /** `live` = FastAPI answered, `demo` = deterministic fixture fallback. */
  source: "live" | "demo" | "loading";
  error: string | null;
  reload: () => void;
}

const AccountContext = createContext<AccountContextValue | null>(null);

/**
 * Shared account snapshot used by the workspace ribbon ticker and the Command
 * Center. One REST fetch at mount, then live telemetry frames (`pnl`, `margin`,
 * `positions`) are merged in so every surface shows the same numbers.
 */
export function AccountProvider({ children }: { children: ReactNode }) {
  const { latest } = useTelemetry();
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

  // Merge streaming frames into the account snapshot.
  useEffect(() => {
    const pnl = latest.pnl;
    const margin = latest.margin;
    const positions = latest.positions;
    if (!pnl && !margin && !positions) return;
    setAccount((previous) => {
      if (!previous) return previous;
      const next: AccountSnapshotResponse = { ...previous };
      if (pnl && "net_liquidation" in pnl.payload) {
        next.net_liquidation = pnl.payload.net_liquidation;
        next.day_pnl = pnl.payload.day_pnl;
        next.day_pnl_pct = pnl.payload.day_pnl_pct;
        const cutoff = new Date(pnl.payload.point.time).getTime();
        next.equity_curve = [
          ...previous.equity_curve
            .filter((point) => new Date(point.time).getTime() < cutoff)
            .slice(-359),
          pnl.payload.point,
        ];
      }
      if (margin && "margin_utilization" in margin.payload) {
        next.initial_margin_used = margin.payload.initial_margin_used;
        next.maintenance_margin_used = margin.payload.maintenance_margin_used;
        next.margin_utilization = margin.payload.margin_utilization;
        next.excess_liquidity = next.net_liquidation - margin.payload.initial_margin_used;
      }
      if (positions && "positions" in positions.payload) {
        next.positions = positions.payload.positions;
        next.gross_leverage = positions.payload.gross_leverage;
        next.net_exposure = positions.payload.net_exposure;
      }
      return next;
    });
  }, [latest.pnl, latest.margin, latest.positions]);

  const reload = useCallback(() => setNonce((value) => value + 1), []);

  const value = useMemo<AccountContextValue>(
    () => ({
      account,
      positions: account?.positions ?? [],
      loading: source === "loading" && account === null,
      source,
      error,
      reload,
    }),
    [account, source, error, reload],
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
