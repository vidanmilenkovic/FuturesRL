"use client";

import { usePathname } from "next/navigation";
import { useMemo, useState, type ReactNode } from "react";

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
import type { ConnectionState, HealthResponse } from "@/lib/types";

import { useAccount } from "../providers/account-provider";
import { useTelemetry } from "../providers/telemetry-provider";
import { Button, ModePill, NavTab, StatusDot, type Tone } from "../ui/primitives";
import { FlattenDialog } from "./flatten-dialog";
import { PowerIcon, ChevronDownIcon, ChevronUpIcon } from "./icons";
import { TelemetryDrawer } from "./telemetry-drawer";

const NAV_ITEMS: { href: string; label: string }[] = [
  { href: "/", label: "Command Center" },
  { href: "/pipeline", label: "Pipeline & Weights" },
  { href: "/risk", label: "Risk & LLM" },
  { href: "/term-structure", label: "Term Structure" },
  { href: "/backtests", label: "Backtest Lab" },
];

const MARGIN_CAP = 0.6;

function isActive(pathname: string, href: string): boolean {
  return href === "/" ? pathname === "/" : pathname.startsWith(href);
}

function modeState(
  backend: "checking" | "online" | "offline",
  health: HealthResponse | null,
  isDemo: boolean,
): { label: string; tone: Tone; title: string } {
  if (backend === "checking") {
    return { label: "CONNECTING", tone: "neutral", title: "Connecting to the telemetry API" };
  }
  if (isDemo || backend === "offline" || !health) {
    return {
      label: "DEMO DATA",
      tone: "warn",
      title: "FastAPI backend unreachable — deterministic fixtures + local telemetry simulator",
    };
  }
  if (health.offline_mode) {
    return {
      label: "DEMO DATA",
      tone: "warn",
      title: "API is serving deterministic offline fixtures (no IBKR/Databento session)",
    };
  }
  if (health.status === "degraded") {
    return {
      label: "DEGRADED",
      tone: "warn",
      title: "API reports degraded service — some adapters are running on fallbacks",
    };
  }
  if (health.environment === "live") {
    return { label: "LIVE · IBKR", tone: "gain", title: "Live IBKR execution session" };
  }
  return { label: "PAPER · IBKR", tone: "neutral", title: "IBKR paper trading session" };
}

function connectionState(status: ConnectionState): { label: string; tone: Tone } {
  switch (status) {
    case "live":
      return { label: "Streaming", tone: "gain" };
    case "demo":
      return { label: "Demo stream", tone: "warn" };
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
  const { status, backend, health, events, isDemo, latest } = useTelemetry();
  const { account, positions, reload } = useAccount();
  const strategies = useAsyncData(listStrategies);
  const pipeline = useAsyncData(getPipeline);
  const [flattenOpen, setFlattenOpen] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);

  const mode = useMemo(() => modeState(backend, health, isDemo), [backend, health, isDemo]);
  const connection = connectionState(status);

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
          <span className="text-[13px] font-semibold tracking-[-0.01em] text-zinc-100">FuturesRL-X</span>
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
            <span className="relative h-1 w-[60px] overflow-hidden rounded-full bg-zinc-800">
              <span
                className={cn(
                  "block h-full rounded-full",
                  marginTone === "loss" ? "bg-loss" : marginTone === "warn" ? "bg-warn" : "bg-gain",
                )}
                style={{ width: `${Math.min(100, (marginUtil / MARGIN_CAP) * 100)}%` }}
              />
            </span>
            <span className="font-mono text-[10px] tabular-nums text-zinc-600">60%</span>
          </div>

          <div className="hidden items-baseline gap-1.5 xl:flex">
            <span className="text-[10px] font-medium uppercase tracking-[0.05em] text-zinc-500">Gross</span>
            <span className="font-mono text-[12px] font-medium tabular-nums text-zinc-100">
              {account ? `${account.gross_leverage.toFixed(2)}×` : "—"}
            </span>
          </div>

          <ModePill label={mode.label} tone={mode.tone} title={mode.title} />

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
      <main className="min-h-0 flex-1 overflow-y-auto overflow-x-hidden bg-canvas p-3">{children}</main>

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
          <span className={cn("hidden text-[10.5px] lg:inline", isDemo ? "text-amber-400/80" : "text-zinc-600")}>
            {isDemo
              ? "DEMO DATA · deterministic fixtures"
              : `API ${health?.environment ?? "connected"} · ${formatInteger(events.length)} frames`}
          </span>
          <button
            type="button"
            onClick={() => setDrawerOpen((open) => !open)}
            aria-expanded={drawerOpen}
            className="focus-ring flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] text-zinc-400 transition-colors hover:bg-surface hover:text-zinc-200"
          >
            Telemetry
            {drawerOpen ? <ChevronDownIcon className="h-3 w-3" /> : <ChevronUpIcon className="h-3 w-3" />}
          </button>
        </span>
      </footer>

      <TelemetryDrawer open={drawerOpen} onClose={() => setDrawerOpen(false)} />
      <FlattenDialog
        open={flattenOpen}
        positions={positions}
        onClose={() => setFlattenOpen(false)}
        onFlattened={reload}
      />
    </div>
  );
}
