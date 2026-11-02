"use client";

import { Badge } from "@/components/ui/badge";
import { Panel, PanelRow } from "@/components/ui/panel";
import { ProgressBar } from "@/components/ui/controls";
import { cn } from "@/lib/cn";
import {
  formatCurrency,
  formatPct,
  formatPctPlain,
  marginTone,
} from "@/lib/format";
import type { AccountSnapshotResponse, RiskThresholds } from "@/lib/types";

/** Margin / loss-budget ladder driven by the account snapshot + risk settings. */
export function RiskLadder({
  account,
  thresholds,
}: {
  account: AccountSnapshotResponse;
  thresholds: RiskThresholds;
}) {
  const tone = marginTone(
    account.margin_utilization,
    thresholds.margin_warning_threshold,
    thresholds.margin_critical_threshold,
  );
  const dailyLossBudget = account.net_liquidation * thresholds.daily_max_loss_pct;
  const lossUsed = account.day_pnl < 0 ? Math.abs(account.day_pnl) / dailyLossBudget : 0;

  return (
    <Panel title="risk ladder" subtitle="hard constraints · live">
      <div className="space-y-3">
        <ProgressRow
          label="margin utilisation"
          value={`${formatPctPlain(account.margin_utilization)} / ${formatPctPlain(thresholds.max_margin_utilization)}`}
          ratio={account.margin_utilization / thresholds.max_margin_utilization}
          tone={tone === "critical" ? "critical" : tone === "warning" ? "warn" : "up"}
        />
        <ProgressRow
          label="daily loss budget"
          value={`${formatCurrency(account.day_pnl < 0 ? Math.abs(account.day_pnl) : 0)} / ${formatCurrency(dailyLossBudget)}`}
          ratio={lossUsed}
          tone={lossUsed > 0.7 ? "critical" : lossUsed > 0.4 ? "warn" : "up"}
        />
        <ProgressRow
          label="excess liquidity"
          value={formatCurrency(account.excess_liquidity, true)}
          ratio={account.excess_liquidity / account.net_liquidation}
          tone="up"
        />
        <ProgressRow
          label="gross leverage"
          value={`${account.gross_leverage.toFixed(2)}× / 3.00×`}
          ratio={account.gross_leverage / 3}
          tone={account.gross_leverage > 2.4 ? "warn" : "info"}
        />
      </div>

      <dl className="mt-3 border-t border-terminal-600/40 pt-2">
        <PanelRow>
          <dt className="text-[11px] uppercase tracking-[0.08em] text-terminal-300">initial margin used</dt>
          <dd className="font-mono text-sm tabular-nums text-terminal-100">
            {formatCurrency(account.initial_margin_used)}
          </dd>
        </PanelRow>
        <PanelRow>
          <dt className="text-[11px] uppercase tracking-[0.08em] text-terminal-300">maintenance margin</dt>
          <dd className="font-mono text-sm tabular-nums text-terminal-300">
            {formatCurrency(account.maintenance_margin_used)}
          </dd>
        </PanelRow>
        <PanelRow>
          <dt className="text-[11px] uppercase tracking-[0.08em] text-terminal-300">net exposure</dt>
          <dd
            className={cn(
              "font-mono text-sm tabular-nums",
              account.net_exposure >= 0 ? "text-signal-up" : "text-signal-down",
            )}
          >
            {formatCurrency(account.net_exposure, true)}
          </dd>
        </PanelRow>
        <PanelRow>
          <dt className="text-[11px] uppercase tracking-[0.08em] text-terminal-300">day pnl</dt>
          <dd className="flex items-center gap-2 font-mono text-sm tabular-nums">
            <span className={account.day_pnl >= 0 ? "text-signal-up" : "text-signal-down"}>
              {formatCurrency(account.day_pnl)}
            </span>
            <span className={account.day_pnl >= 0 ? "text-signal-up/70" : "text-signal-down/70"}>
              {formatPct(account.day_pnl_pct)}
            </span>
          </dd>
        </PanelRow>
        <PanelRow>
          <dt className="text-[11px] uppercase tracking-[0.08em] text-terminal-300">kill switch</dt>
          <dd>
            <Badge tone="up" dot>
              standby
            </Badge>
          </dd>
        </PanelRow>
      </dl>
    </Panel>
  );
}

function ProgressRow({
  label,
  value,
  ratio,
  tone,
}: {
  label: string;
  value: string;
  ratio: number;
  tone: "info" | "up" | "down" | "warn" | "critical" | "violet";
}) {
  return (
    <div>
      <div className="mb-1 flex items-center justify-between gap-2">
        <span className="font-mono text-[10px] uppercase tracking-[0.12em] text-terminal-300">{label}</span>
        <span className="font-mono text-[10px] tabular-nums text-terminal-200">{value}</span>
      </div>
      <ProgressBar value={Math.min(ratio, 1)} tone={tone} label={label} />
    </div>
  );
}
