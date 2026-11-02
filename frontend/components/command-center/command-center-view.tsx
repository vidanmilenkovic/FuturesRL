"use client";

import { useEffect, useMemo, useState } from "react";

import { Sparkline } from "@/components/charts/svg-charts";
import { EquityCurvePanel } from "@/components/command-center/equity-curve-panel";
import { EmergencyFlatten } from "@/components/command-center/emergency-flatten";
import { PositionsTable } from "@/components/command-center/positions-table";
import { RiskLadder } from "@/components/command-center/risk-ladder";
import { TelemetryTape } from "@/components/command-center/telemetry-tape";
import { useTelemetry } from "@/components/providers/telemetry-provider";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/controls";
import { MarginGauge } from "@/components/ui/margin-gauge";
import { Panel } from "@/components/ui/panel";
import { StatCard } from "@/components/ui/stat-card";
import { getAccount, getRiskState } from "@/lib/api";
import { cn } from "@/lib/cn";
import { formatCurrency, formatNumber, formatPct, formatPctPlain, formatSignedCurrency } from "@/lib/format";
import { useAsyncData } from "@/lib/use-async-data";
import { mockThresholds } from "@/lib/mock";
import type { AccountSnapshotResponse, RiskThresholds } from "@/lib/types";

/**
 * View 1 — Command Center: real-time equity curve, NLV/KPI strip, margin gauge
 * with 60/80 zones, active futures contracts and the emergency kill switch.
 */
export function CommandCenterView() {
  const account = useAsyncData(getAccount);
  const risk = useAsyncData(getRiskState);
  const { latest, isDemo } = useTelemetry();
  const [snapshot, setSnapshot] = useState<AccountSnapshotResponse | null>(null);

  useEffect(() => {
    if (account.data) setSnapshot(account.data);
  }, [account.data]);

  // Merge live telemetry frames into the snapshot (pnl / margin / positions).
  useEffect(() => {
    const pnl = latest.pnl;
    const margin = latest.margin;
    const positions = latest.positions;
    if (!pnl && !margin && !positions) return;
    setSnapshot((previous) => {
      if (!previous) return previous;
      const next = { ...previous };
      if (pnl && "net_liquidation" in pnl.payload) {
        next.net_liquidation = pnl.payload.net_liquidation;
        next.day_pnl = pnl.payload.day_pnl;
        next.day_pnl_pct = pnl.payload.day_pnl_pct;
        // Lightweight Charts requires strictly ascending time: drop any curve
        // point at/after the live tick before appending it.
        const cutoff = new Date(pnl.payload.point.time).getTime();
        next.equity_curve = [
          ...previous.equity_curve
            .filter((point) => new Date(point.time).getTime() < cutoff)
            .slice(-251),
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

  const thresholds: RiskThresholds = risk.data?.thresholds ?? mockThresholds();
  const equitySeries = useMemo(
    () => (snapshot?.equity_curve ?? []).slice(-90).map((point) => point.value),
    [snapshot?.equity_curve],
  );

  if (account.loading && !snapshot) {
    return <CommandCenterSkeleton />;
  }

  if (!snapshot) {
    return (
      <Panel title="command center">
        <p className="py-10 text-center font-mono text-xs uppercase tracking-[0.12em] text-signal-down">
          account snapshot unavailable — {account.error ?? "unknown error"}
        </p>
      </Panel>
    );
  }

  const dayTone = snapshot.day_pnl >= 0 ? "up" : "down";

  return (
    <div className="grid grid-cols-1 gap-3 xl:grid-cols-3">
      <div className="space-y-3 xl:col-span-2">
        {/* KPI strip */}
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-6">
          <StatCard
            label="NLV"
            value={formatCurrency(snapshot.net_liquidation)}
            hint="net liquidation value"
            emphasis
          />
          <StatCard
            label="day pnl"
            value={formatSignedCurrency(snapshot.day_pnl)}
            delta={snapshot.day_pnl_pct}
            tone={dayTone}
            hint={`${formatPct(snapshot.day_pnl_pct)} on start-of-day NLV`}
            sparkline={<Sparkline values={equitySeries} width={140} height={26} />}
          />
          <StatCard
            label="margin util"
            value={formatPctPlain(snapshot.margin_utilization)}
            tone={
              snapshot.margin_utilization >= thresholds.margin_critical_threshold
                ? "critical"
                : snapshot.margin_utilization >= thresholds.margin_warning_threshold
                  ? "warn"
                  : "up"
            }
            hint={`IM ${formatCurrency(snapshot.initial_margin_used, true)}`}
          />
          <StatCard
            label="gross leverage"
            value={`${snapshot.gross_leverage.toFixed(2)}×`}
            tone="info"
            hint={`net ${formatCurrency(snapshot.net_exposure, true)}`}
          />
          <StatCard
            label="excess liquidity"
            value={formatCurrency(snapshot.excess_liquidity, true)}
            tone="up"
            hint="available above IM"
          />
          <StatCard
            label="open legs"
            value={formatNumber(snapshot.positions.length, 0)}
            tone="neutral"
            hint={`${snapshot.positions.reduce((sum, row) => sum + Math.abs(row.quantity), 0)} contracts`}
            badge={isDemo ? <Badge tone="warn">demo</Badge> : <Badge tone="up" dot>live</Badge>}
          />
        </div>

        <Panel
          title="equity curve"
          subtitle="NLV · mark-to-market"
          actions={<Badge tone="info">daily</Badge>}
        >
          <EquityCurvePanel points={snapshot.equity_curve} />
        </Panel>

        <Panel
          title="active futures contracts"
          subtitle="root · expiry · days to roll · qty · w_i · unrealised pnl · margin"
          actions={
            <span className={cn("font-mono text-[10px] uppercase tracking-[0.1em]", isDemo ? "text-signal-warn" : "text-signal-up")}>
              {isDemo ? "demo fixtures" : "broker reconciled"}
            </span>
          }
          contentClassName="px-3 pt-2 pb-0"
        >
          <PositionsTable positions={snapshot.positions} />
        </Panel>
      </div>

      <div className="space-y-3">
        <Panel title="margin utilisation" subtitle="η_max 60% · kill 80%">
          <MarginGauge
            utilization={snapshot.margin_utilization}
            warning={thresholds.margin_warning_threshold}
            critical={thresholds.margin_critical_threshold}
            label="Initial margin utilisation"
          />
          <p className="mt-1 border-t border-terminal-600/40 pt-2 text-center font-mono text-[10px] uppercase tracking-[0.1em] text-terminal-400">
            {snapshot.margin_utilization >= thresholds.margin_critical_threshold
              ? "critical zone — kill switch armed"
              : snapshot.margin_utilization >= thresholds.margin_warning_threshold
                ? "warning zone — no new leverage"
                : "nominal — full rebalance headroom"}
          </p>
        </Panel>

        <EmergencyFlatten positions={snapshot.positions} />

        <RiskLadder account={snapshot} thresholds={thresholds} />

        <TelemetryTape />
      </div>
    </div>
  );
}

function CommandCenterSkeleton() {
  return (
    <div className="grid grid-cols-1 gap-3 xl:grid-cols-3">
      <div className="space-y-3 xl:col-span-2">
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-6">
          {Array.from({ length: 6 }).map((_, index) => (
            <Skeleton key={index} className="h-[76px]" />
          ))}
        </div>
        <Skeleton className="h-[360px]" />
        <Skeleton className="h-[320px]" />
      </div>
      <div className="space-y-3">
        <Skeleton className="h-[280px]" />
        <Skeleton className="h-[180px]" />
        <Skeleton className="h-[260px]" />
      </div>
    </div>
  );
}
