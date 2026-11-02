"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useMemo, useState, type ReactNode } from "react";

import { Badge } from "@/components/ui/badge";
import { LiveDot } from "@/components/ui/controls";
import { cn } from "@/lib/cn";
import { formatClock } from "@/lib/format";

import { useTelemetry } from "../providers/telemetry-provider";
import {
  BacktestIcon,
  CloseIcon,
  CommandCenterIcon,
  DatabaseIcon,
  MenuIcon,
  PipelineIcon,
  RefreshIcon,
  RiskIcon,
  TermStructureIcon,
} from "./icons";

interface NavItem {
  href: string;
  label: string;
  short: string;
  description: string;
  Icon: (props: { className?: string }) => ReactNode;
}

const NAV_ITEMS: NavItem[] = [
  {
    href: "/",
    label: "Command Center",
    short: "CMD",
    description: "Live account, margin, positions & kill switch",
    Icon: CommandCenterIcon,
  },
  {
    href: "/pipeline",
    label: "Pipeline Inspector",
    short: "PIPE",
    description: "FinRL-X waterfall S_t → w_base → w_t → N_i,t",
    Icon: PipelineIcon,
  },
  {
    href: "/risk",
    label: "LLM Risk Monitor",
    short: "RISK",
    description: "Sentiment, confidence, semantic risk & lockouts",
    Icon: RiskIcon,
  },
  {
    href: "/term-structure",
    label: "Term Structure",
    short: "TERM",
    description: "Curves, carry and roll calendar",
    Icon: TermStructureIcon,
  },
  {
    href: "/backtests",
    label: "Backtest Lab",
    short: "BTLAB",
    description: "Runs, tear sheets and attribution",
    Icon: BacktestIcon,
  },
];

function resolveTitle(pathname: string): { title: string; description: string } {
  const item = NAV_ITEMS.find((nav) =>
    nav.href === "/" ? pathname === "/" : pathname.startsWith(nav.href),
  );
  return item
    ? { title: item.label, description: item.description }
    : { title: "FuturesRL-X", description: "Quantitative futures command center" };
}

export function AppShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const { status, backend, isDemo, lastUpdate, sequence, retry } = useTelemetry();
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [clock, setClock] = useState<string>("--:--:--");
  const active = useMemo(() => resolveTitle(pathname), [pathname]);

  useEffect(() => {
    setDrawerOpen(false);
  }, [pathname]);

  useEffect(() => {
    setClock(formatClock(new Date().toISOString()));
    const timer = setInterval(() => setClock(formatClock(new Date().toISOString())), 1_000);
    return () => clearInterval(timer);
  }, []);

  const statusLabel =
    status === "live" ? "BACKEND LIVE" : status === "demo" ? "SIMULATED STREAM" : status === "connecting" ? "CONNECTING" : "OFFLINE";

  return (
    <div className="flex min-h-screen w-full">
      {/* Desktop sidebar */}
      <aside className="sticky top-0 hidden h-screen w-[232px] shrink-0 flex-col border-r border-terminal-600/60 bg-terminal-900/80 lg:flex">
        <Brand />
        <nav aria-label="Primary" className="flex-1 space-y-0.5 px-2 py-3">
          {NAV_ITEMS.map((item) => (
            <NavLink key={item.href} item={item} pathname={pathname} />
          ))}
        </nav>
        <SidebarFooter
          environment={backend === "online" ? "PAPER · IBGW" : "DEMO FIXTURES"}
          sequence={sequence}
        />
      </aside>

      {/* Mobile drawer */}
      {drawerOpen ? (
        <div className="fixed inset-0 z-40 lg:hidden">
          <button
            type="button"
            aria-label="Close navigation"
            className="absolute inset-0 bg-terminal-950/80 backdrop-blur-sm"
            onClick={() => setDrawerOpen(false)}
          />
          <aside className="relative flex h-full w-72 flex-col border-r border-terminal-600/70 bg-terminal-900">
            <div className="flex items-center justify-between px-3 py-2">
              <Brand compact />
              <button
                type="button"
                onClick={() => setDrawerOpen(false)}
                aria-label="Close navigation"
                className="rounded-sm p-1.5 text-terminal-300 hover:bg-terminal-700 hover:text-terminal-100 focus-ring"
              >
                <CloseIcon />
              </button>
            </div>
            <nav aria-label="Primary" className="flex-1 space-y-0.5 px-2 py-2">
              {NAV_ITEMS.map((item) => (
                <NavLink key={item.href} item={item} pathname={pathname} />
              ))}
            </nav>
            <SidebarFooter environment="MOBILE" sequence={sequence} />
          </aside>
        </div>
      ) : null}

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-30 border-b border-terminal-600/60 bg-terminal-950/85 backdrop-blur">
          <div className="flex items-center gap-3 px-3 py-2.5 sm:px-4">
            <button
              type="button"
              className="rounded-sm p-1.5 text-terminal-200 hover:bg-terminal-700 focus-ring lg:hidden"
              onClick={() => setDrawerOpen(true)}
              aria-label="Open navigation"
            >
              <MenuIcon />
            </button>
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2">
                <h1 className="truncate font-mono text-sm font-semibold uppercase tracking-[0.16em] text-terminal-100">
                  {active.title}
                </h1>
                {isDemo ? (
                  <Badge tone="warn" dot className="hidden sm:inline-flex">
                    DEMO DATA
                  </Badge>
                ) : null}
              </div>
              <p className="hidden truncate text-[11px] text-terminal-400 sm:block">{active.description}</p>
            </div>

            <div className="flex items-center gap-2 sm:gap-3">
              <span className="hidden items-center gap-1.5 font-mono text-[10px] uppercase tracking-[0.14em] text-terminal-400 md:flex">
                <DatabaseIcon className="h-3.5 w-3.5" />
                CME · NYMEX · COMEX · CBOT
              </span>
              <span className="hidden font-mono text-[11px] tabular-nums text-terminal-200 sm:inline">
                {clock}
                <span className="ml-1 text-[9px] text-terminal-400">UTC</span>
              </span>
              <Badge tone={status === "live" ? "up" : status === "demo" ? "warn" : "neutral"} dot>
                {statusLabel}
              </Badge>
              <button
                type="button"
                onClick={retry}
                title="Reconnect telemetry stream"
                aria-label="Reconnect telemetry stream"
                className="rounded-sm border border-terminal-500/60 p-1.5 text-terminal-300 transition-colors hover:border-signal-info/50 hover:text-signal-info focus-ring"
              >
                <RefreshIcon className="h-3.5 w-3.5" />
              </button>
            </div>
          </div>

          {isDemo ? (
            <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 border-t border-signal-warn/25 bg-signal-warn/[0.06] px-3 py-1 font-mono text-[10px] uppercase tracking-[0.1em] text-signal-warn sm:px-4">
              <span className="font-semibold">DEMO DATA</span>
              <span className="text-signal-warn/80">
                FastAPI backend unreachable — deterministic fixtures + local telemetry simulator.
              </span>
              <span className="ml-auto hidden text-signal-warn/60 md:inline">
                start: uvicorn src.api.main:app --port 8000
              </span>
            </div>
          ) : null}
        </header>

        <main className="flex-1 px-3 py-3 sm:px-4 sm:py-4">
          <div className="mx-auto w-full max-w-[1700px]">{children}</div>
        </main>

        <footer className="mt-auto flex flex-wrap items-center justify-between gap-2 border-t border-terminal-600/50 px-3 py-2 font-mono text-[10px] uppercase tracking-[0.1em] text-terminal-400 sm:px-4">
          <span>FuturesRL-X · FinRL-X weight-centric pipeline · v1.0.0</span>
          <span className="flex items-center gap-2">
            <LiveDot state={status === "live" ? "live" : status === "demo" ? "demo" : "offline"} />
            <span>seq {sequence.toString().padStart(5, "0")}</span>
            <span className="hidden sm:inline">last {lastUpdate ? formatClock(lastUpdate) : "--:--:--"}Z</span>
          </span>
        </footer>
      </div>
    </div>
  );
}

function Brand({ compact = false }: { compact?: boolean }) {
  return (
    <div className={cn("flex items-center gap-2.5 border-b border-terminal-600/60 px-3 py-3", compact && "border-b-0 py-1")}>
      <span className="relative grid h-8 w-8 shrink-0 place-items-center rounded-sm border border-signal-info/40 bg-signal-info/10">
        <span className="absolute inset-0 animate-pulse bg-signal-info/5" aria-hidden />
        <svg viewBox="0 0 24 24" className="h-4 w-4 text-signal-info" fill="none" stroke="currentColor" strokeWidth={1.8} aria-hidden>
          <path d="M4 18 9 8l4 6 3-4 4 8" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </span>
      <div className="min-w-0">
        <p className="truncate font-mono text-[12px] font-bold tracking-[0.18em] text-terminal-100">
          FUTURESRL-X
        </p>
        <p className="truncate font-mono text-[9px] uppercase tracking-[0.22em] text-terminal-400">
          quant terminal
        </p>
      </div>
    </div>
  );
}

function NavLink({ item, pathname }: { item: NavItem; pathname: string }) {
  const selected = item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);
  return (
    <Link
      href={item.href}
      aria-current={selected ? "page" : undefined}
      className={cn(
        "group relative flex items-start gap-2.5 rounded-sm px-2 py-2 transition-colors focus-ring",
        selected
          ? "bg-terminal-700/70 text-terminal-100"
          : "text-terminal-300 hover:bg-terminal-800 hover:text-terminal-100",
      )}
    >
      {selected ? (
        <span aria-hidden className="absolute inset-y-1 left-0 w-0.5 rounded-full bg-signal-info" />
      ) : null}
      <item.Icon className={cn("mt-0.5 h-4 w-4 shrink-0", selected ? "text-signal-info" : "text-terminal-400 group-hover:text-terminal-200")} />
      <span className="min-w-0">
        <span className="block truncate text-[12px] font-medium">{item.label}</span>
        <span className="block truncate font-mono text-[9px] uppercase tracking-[0.12em] text-terminal-500 group-hover:text-terminal-400">
          {item.short}
        </span>
      </span>
    </Link>
  );
}

function SidebarFooter({ environment, sequence }: { environment: string; sequence: number }) {
  return (
    <div className="space-y-2 border-t border-terminal-600/60 px-3 py-3">
      <div className="flex items-center justify-between font-mono text-[9px] uppercase tracking-[0.14em] text-terminal-400">
        <span>mode</span>
        <span className="text-terminal-200">{environment}</span>
      </div>
      <div className="flex items-center justify-between font-mono text-[9px] uppercase tracking-[0.14em] text-terminal-400">
        <span>telemetry</span>
        <span className="tabular-nums text-terminal-200">#{sequence.toString().padStart(5, "0")}</span>
      </div>
      <p className="font-mono text-[9px] leading-relaxed text-terminal-500">
        Risk-First engine · κ 4.5 · c ≥ 0.70 · η_max 60%
      </p>
    </div>
  );
}
