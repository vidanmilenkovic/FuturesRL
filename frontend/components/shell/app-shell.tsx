"use client";

import { usePathname } from "next/navigation";
import { useCallback, useState, type ReactNode } from "react";

import { getPipeline, listStrategies } from "@/lib/api";
import { cn } from "@/lib/cn";
import {
  formatCompactUsd,
  formatDateTime,
  formatInteger,
  formatPct,
  formatPctPlain,
  formatSignedCurrency,
} from "@/lib/format";
import { useAsyncData } from "@/lib/use-async-data";
import type { ConnectionState, DataModeKey } from "@/lib/types";

import { useAccount } from "../providers/account-provider";
import { useTelemetry } from "../providers/telemetry-provider";
import { Button, ModePill, NavTab, StatusDot, type Tone } from "../ui/primitives";
import { FlattenDialog } from "./flatten-dialog";
import { PowerIcon, ChevronDownIcon, ChevronUpIcon } from "./icons";
import { LLMStatusPill } from "./llm-status-pill";
import { OperatorDrawer } from "./operator-drawer";
import { TelemetryDrawer } from "./telemetry-drawer";

const NAV_ITEMS: { href: string; label: string }[] = [
  { href: "/", label: "Overview" },
  { href: "/pipeline", label: "Weights" },
  { href: "/risk", label: "Risk" },
  { href: "/term-structure", label: "Term Structure" },
  { href: "/studio", label: "Strategies" },
  { href: "/backtests", label: "Backtests" },
  { href: "/journal", label: "Journal" },
];

const MARGIN_CAP = 0.6;

function isActive(pathname: string, href: string): boolean {
  return href === "/" ? pathname === "/" : pathname.startsWith(href);
}

const MODE_TONE: Record<DataModeKey, Tone> = {
  connecting: "neutral",
  ui_fallback: "warn",
  offline_engine: "neutral",
  reconnecting: "warn",
  paper_ibkr: "accent",
  live_ibkr: "gain",
};

function connectionState(status: ConnectionState): { label: string; tone: Tone } {
  switch (status) {
    case "live":
      return { label: "Streaming", tone: "gain" };
    case "demo":
      return { label: "Simulated stream", tone: "warn" };
    case "connecting":
      return { label: "Connecting", tone: "neutral" };
    default:
      return { label: "Offline", tone: "loss" };
  }
}

/**
 * Full-viewport docked workspace shell: ribbon (brand, view tabs, always-visible
 * account ticker, data-mode pill, flatten-all), scrollable main surface and a
 * compact status bar with a collapsible telemetry console.
 */
export function AppShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const { status, backend, events, latest } = useTelemetry();
  const { account, positions, dataMode, reload } = useAccount();
  const strategies = useAsyncData(listStrategies);
  const pipeline = useAsyncData(getPipeline);
  const [flattenOpen, setFlattenOpen] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [operatorOpen, setOperatorOpen] = useState(false);
  const closeTelemetry = useCallback(() => setDrawerOpen(false), []);
  const toggleTelemetry = useCallback(() => setDrawerOpen((open) => !open), []);
  const closeOperator = useCallback(() => setOperatorOpen(false), []);
  const toggleOperator = useCallback(() => setOperatorOpen((open) => !open), []);

  const connection = connectionState(status);
  const modeTone = MODE_TONE[dataMode.key];
  const modeTooltip = `${dataMode.description} · API ${backend} · WS ${status}`;

  const marginUtil = account?.margin_utilization ?? 0;
  const dayPnl = account?.day_pnl ?? 0;
  const dayPnlPct = account?.day_pnl_pct ?? 0;
  const dayTone: Tone = dayPnl > 0 ? "gain" : dayPnl < 0 ? "loss" : "neutral";
  const marginTone: Tone = marginUtil >= 0.8 ? "loss" : marginUtil >= MARGIN_CAP ? "warn" : "gain";

  const availableStrategies = strategies.data ?? [];
  const strategyId =
    availableStrategies.find((strategy) => strategy.id === "tsmom_vol_target")?.id ??
    availableStrategies[0]?.id ??
    "—";
  const pipelineFrame = latest.pipeline;
  const lastRebalance =
    pipelineFrame && "pipeline" in pipelineFrame.payload
      ? pipelineFrame.payload.pipeline.timestamp
      : (pipeline.data?.timestamp ?? null);

  return (
    <div className="relative flex h-screen w-screen flex-col overflow-hidden bg-canvas">
      {/* ---------------------------------------------------------------- ribbon */}
      <header className="no-scrollbar flex h-12 shrink-0 items-center gap-2 overflow-x-auto border-b border-hairline bg-panel pl-3 pr-2">
        <div className="flex shrink-0 items-center gap-2">
          <span className="grid h-[22px] w-[22px] place-items-center rounded-[5px] border border-edge bg-canvas">
            <svg viewBox="0 0 20 20" className="h-3.5 w-3.5" fill="none" stroke="#3B82F6" strokeWidth={1.8} aria-hidden="true">
              <path d="M3.5 14.5v-9l3 5 3-5 3 5 3-5v9" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </span>
          <span className="text-[13px] font-semibold tracking-[-0.01em] text-zinc-100">FuturesRL</span>
        </div>

        <nav aria-label="Views" className="ml-1 flex h-12 shrink-0 items-center">
          {NAV_ITEMS.map((item) => (
            <NavTab
              key={item.href}
              href={item.href}
              label={item.label}
              active={isActive(pathname, item.href)}
            />
          ))}
        </nav>

        <div className="ml-auto flex shrink-0 items-center gap-4 pl-3">
          {/* inline account ticker */}
          <div className="hidden items-baseline gap-1.5 sm:flex">
            <span className="text-[10px] font-medium uppercase tracking-[0.05em] text-zinc-500">NLV</span>
            <span className="font-mono text-[12px] font-medium tabular-nums text-zinc-100">
              {account ? formatCompactUsd(account.net_liquidation) : "—"}
            </span>
          </div>

          <div className="hidden items-baseline gap-1.5 md:flex">
            <span className="text-[10px] font-medium uppercase tracking-[0.05em] text-zinc-500">Day PnL</span>
            <span
              className={cn(
                "font-mono text-[12px] font-medium tabular-nums",
                dayPnl > 0 ? "text-gain" : dayPnl < 0 ? "text-loss" : "text-zinc-300",
              )}
            >
              {formatSignedCurrency(dayPnl, true)}
            </span>
            <span
              className={cn(
                "font-mono text-[11px] tabular-nums",
                dayTone === "gain" ? "text-gain/80" : dayTone === "loss" ? "text-loss/80" : "text-zinc-500",
              )}
            >
              {formatPct(dayPnlPct)}
            </span>
          </div>

          <div className="hidden items-center gap-1.5 lg:flex" title={`Initial margin utilisation vs ${formatPctPlain(MARGIN_CAP)} policy cap`}>
            <span className="text-[10px] font-medium uppercase tracking-[0.05em] text-zinc-500">Margin</span>
            <span
              className={cn(
                "font-mono text-[12px] font-medium tabular-nums",
                marginTone === "loss" ? "text-loss" : marginTone === "warn" ? "text-warn" : "text-zinc-100",
              )}
            >
              {formatPctPlain(marginUtil)}
            </span>
            <span className="relative hidden h-1 w-[60px] overflow-hidden rounded-full bg-zinc-800 2xl:inline-block">
              <span
                className={cn(
                  "block h-full rounded-full",
                  marginTone === "loss" ? "bg-loss" : marginTone === "warn" ? "bg-warn" : "bg-gain",
                )}
                style={{ width: `${Math.min(100, (marginUtil / MARGIN_CAP) * 100)}%` }}
              />
            </span>
            <span className="hidden font-mono text-[10px] tabular-nums text-zinc-600 2xl:inline">60%</span>
          </div>

          <div className="hidden items-baseline gap-1.5 2xl:flex">
            <span className="text-[10px] font-medium uppercase tracking-[0.05em] text-zinc-500">Gross</span>
            <span className="font-mono text-[12px] font-medium tabular-nums text-zinc-100">
              {account ? `${account.gross_leverage.toFixed(2)}×` : "—"}
            </span>
          </div>

          <span data-testid="data-mode-badge" data-state={dataMode.key} data-mode={dataMode.label} title={modeTooltip}>
            <ModePill label={dataMode.label} tone={modeTone} title={modeTooltip} />
          </span>

          <LLMStatusPill />

          <Button
            variant="danger"
            size="xs"
            onClick={() => setFlattenOpen(true)}
            title="Flatten all positions (market orders)"
            ariaLabel="Flatten all positions"
            className="gap-1"
          >
            <PowerIcon className="h-3 w-3" />
            Flatten All
          </Button>
        </div>
      </header>

      {/* ------------------------------------------------------------------ main */}
      <main className="min-h-0 flex-1 overflow-y-auto overflow-x-hidden bg-canvas p-3">
        {backend === "offline" ? (
          <div
            role="alert"
            data-testid="api-offline-banner"
            className="mb-3 rounded-md border border-warn/30 bg-warn-soft/30 px-3 py-2 text-[11px] leading-5 text-amber-200"
          >
            <span className="font-medium text-amber-100">API unavailable.</span>{" "}
            Showing honest empty/error states — no synthetic mock data is substituted. Start the FastAPI backend to
            populate live payloads.
          </div>
        ) : null}
        {children}
      </main>

      {/* ------------------------------------------------------------ status bar */}
      <footer className="relative z-40 flex h-7 shrink-0 items-center gap-3 border-t border-hairline bg-panel px-3 text-[11px] text-zinc-500">
        <span className="flex items-center gap-1.5">
          <StatusDot tone={connection.tone} pulse={status === "live"} />
          <span className={cn(connection.tone === "gain" ? "text-emerald-300" : connection.tone === "warn" ? "text-amber-300" : "text-zinc-400")}>
            {connection.label}
          </span>
        </span>
        <span className="text-zinc-700">·</span>
        <span className="hidden items-center gap-1 sm:flex">
          <span className="text-zinc-600">Strategy</span>
          <span className="font-mono text-[11px] text-zinc-400">{strategyId}</span>
        </span>
        <span className="hidden text-zinc-700 sm:inline">·</span>
        <span className="hidden items-center gap-1 md:flex">
          <span className="text-zinc-600">Last rebalance</span>
          <span className="font-mono text-[11px] text-zinc-400">
            {lastRebalance ? formatDateTime(lastRebalance) : "—"}
          </span>
        </span>
        <span className="ml-auto flex items-center gap-2">
          <span
            className={cn(
              "hidden text-[10.5px] lg:inline",
              dataMode.key === "ui_fallback"
                ? "text-amber-400/80"
                : dataMode.key === "live_ibkr"
                  ? "text-emerald-400/80"
                  : dataMode.key === "paper_ibkr"
                    ? "text-blue-300/80"
                    : "text-zinc-600",
            )}
            title={modeTooltip}
          >
            {dataMode.key === "ui_fallback"
              ? "UI FALLBACK · deterministic fixtures"
              : dataMode.key === "offline_engine"
                ? `OFFLINE ENGINE · ${formatInteger(events.length)} frames`
                : `${dataMode.label} · ${formatInteger(events.length)} frames`}
          </span>
          <button
            type="button"
            onClick={toggleTelemetry}
            aria-expanded={drawerOpen}
            className="focus-ring flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] text-zinc-400 transition-colors hover:bg-surface hover:text-zinc-200"
          >
            Telemetry
            {drawerOpen ? <ChevronDownIcon className="h-3 w-3" /> : <ChevronUpIcon className="h-3 w-3" />}
          </button>
          <button
            type="button"
            data-testid="operator-button"
            onClick={toggleOperator}
            aria-expanded={operatorOpen}
            className="focus-ring flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] text-zinc-400 transition-colors hover:bg-surface hover:text-zinc-200"
          >
            System
          </button>
        </span>
      </footer>

      <TelemetryDrawer open={drawerOpen} onClose={closeTelemetry} />
      <OperatorDrawer open={operatorOpen} onClose={closeOperator} />
      <FlattenDialog
        open={flattenOpen}
        positions={positions}
        onClose={() => setFlattenOpen(false)}
        onFlattened={reload}
      />
    </div>
  );
}
