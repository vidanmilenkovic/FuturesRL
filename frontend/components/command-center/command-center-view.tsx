"use client";

import { useMemo, useState } from "react";

import { TimeSeriesChart } from "@/components/charts/time-series";
import { BulletBars, ExposureBars, WeightBar, type BulletRow } from "@/components/charts/plots";
import { useAccount } from "@/components/providers/account-provider";
import { useTelemetry } from "@/components/providers/telemetry-provider";
import {
  Badge,
  KpiCard,
  Panel,
  Segmented,
  SortHeader,
  Tabs,
  EmptyState,
  Skeleton,
  signedColor,
} from "@/components/ui/primitives";
import { getRiskState } from "@/lib/api";
import { cn } from "@/lib/cn";
import {
  formatClock,
  formatCompactUsd,
  formatCurrency,
  formatInteger,
  formatNumber,
  formatPct,
  formatPctPlain,
  formatSignedCurrency,
} from "@/lib/format";
import { summarizeTelemetry } from "@/lib/telemetry";
import type { PositionRow, RiskThresholds } from "@/lib/types";
import { useAsyncData } from "@/lib/use-async-data";

type Timeframe = "1D" | "1W" | "1M" | "3M" | "6M" | "1Y" | "ALL";
type BookTab = "positions" | "orders" | "fills" | "telemetry";
type SortKey = "root" | "class" | "qty" | "weight" | "upnl" | "notional" | "dtr";

const TIMEFRAMES: { value: Timeframe; label: string; bars: number }[] = [
  { value: "1D", label: "1D", bars: 2 },
  { value: "1W", label: "1W", bars: 5 },
  { value: "1M", label: "1M", bars: 21 },
  { value: "3M", label: "3M", bars: 63 },
  { value: "6M", label: "6M", bars: 126 },
  { value: "1Y", label: "1Y", bars: 252 },
  { value: "ALL", label: "ALL", bars: Number.POSITIVE_INFINITY },
];

const FALLBACK_THRESHOLDS: RiskThresholds = {
  llm_confidence_threshold: 0.7,
  llm_safe_score: 1.5,
  llm_critical_score: 4.5,
  max_margin_utilization: 0.6,
  margin_warning_threshold: 0.6,
  margin_critical_threshold: 0.8,
  daily_max_loss_pct: 0.03,
  target_annual_volatility: 0.12,
  macro_lockout_minutes_before: 30,
  macro_lockout_minutes_after: 30,
};

const ASSET_LABEL: Record<string, string> = {
  equity_index: "Equity index",
  rates: "Rates",
  energy: "Energy",
  metals: "Metals",
  fx: "FX",
  agriculture: "Agriculture",
  crypto: "Crypto",
  unknown: "Other",
};

const BOOK_TABS: { value: BookTab; label: string }[] = [
  { value: "positions", label: "Positions" },
  { value: "orders", label: "Working Orders" },
  { value: "fills", label: "Recent Fills" },
  { value: "telemetry", label: "Telemetry" },
];

export function CommandCenterView() {
  const { account, positions, loading, source } = useAccount();
  const { events } = useTelemetry();
  const risk = useAsyncData(getRiskState);
  const thresholds = risk.data?.thresholds ?? FALLBACK_THRESHOLDS;

  const [timeframe, setTimeframe] = useState<Timeframe>("6M");
  const [showDrawdown, setShowDrawdown] = useState(true);
  const [bookTab, setBookTab] = useState<BookTab>("positions");
  const [sort, setSort] = useState<{ key: SortKey; dir: "asc" | "desc" }>({
    key: "notional",
    dir: "desc",
  });

  const curve = useMemo(() => {
    if (!account) return [];
    return [...account.equity_curve]
      .filter((point) => Number.isFinite(new Date(point.time).getTime()))
      .sort((a, b) => new Date(a.time).getTime() - new Date(b.time).getTime());
  }, [account]);

  const windowPoints = useMemo(() => {
    const spec =
      TIMEFRAMES.find((item) => item.value === timeframe) ?? { value: "6M" as const, label: "6M", bars: 126 };
    if (!Number.isFinite(spec.bars)) return curve;
    return curve.slice(-Math.max(2, spec.bars));
  }, [curve, timeframe]);

  const drawdownSeries = useMemo(() => {
    let peak = Number.NEGATIVE_INFINITY;
    return windowPoints.map((point) => {
      peak = Math.max(peak, point.value);
      return { time: point.time, value: peak > 0 ? point.value / peak - 1 : 0 };
    });
  }, [windowPoints]);

  const period = useMemo(() => {
    const first = windowPoints[0]?.value ?? 0;
    const last = windowPoints[windowPoints.length - 1]?.value ?? 0;
    const change = first > 0 ? last / first - 1 : 0;
    const high = windowPoints.length ? Math.max(...windowPoints.map((point) => point.value)) : 0;
    const low = windowPoints.length ? Math.min(...windowPoints.map((point) => point.value)) : 0;
    const returns: number[] = [];
    for (let index = 1; index < windowPoints.length; index += 1) {
      const previous = windowPoints[index - 1]?.value ?? 0;
      const current = windowPoints[index]?.value ?? 0;
      if (previous > 0) returns.push(current / previous - 1);
    }
    const mean = returns.length ? returns.reduce((sum, value) => sum + value, 0) / returns.length : 0;
    const variance = returns.length
      ? returns.reduce((sum, value) => sum + (value - mean) ** 2, 0) / Math.max(1, returns.length - 1)
      : 0;
    const annualizedVol = Math.sqrt(variance) * Math.sqrt(252);
    const maxDrawdown = drawdownSeries.length
      ? Math.abs(Math.min(...drawdownSeries.map((point) => point.value)))
      : 0;
    return { first, last, change, high, low, annualizedVol, maxDrawdown };
  }, [windowPoints, drawdownSeries]);

  const totals = useMemo(() => {
    const unrealized = positions.reduce((sum, position) => sum + position.unrealized_pnl, 0);
    const margin = positions.reduce(
      (sum, position) => sum + Math.abs(position.quantity) * position.initial_margin,
      0,
    );
    const notional = positions.reduce((sum, position) => sum + Math.abs(position.notional), 0);
    return { unrealized, margin, notional };
  }, [positions]);

  const bullets: BulletRow[] = useMemo(() => {
    const marginCap = thresholds.max_margin_utilization;
    const dailyLossCap = (account?.net_liquidation ?? 0) * thresholds.daily_max_loss_pct;
    const dailyLoss = Math.max(0, -(account?.day_pnl ?? 0));
    return [
      {
        label: "Margin utilisation",
        value: (account?.margin_utilization ?? 0) * 100,
        cap: marginCap * 100,
        format: (value) => `${value.toFixed(1)}%`,
        hint: `Initial margin vs ${(marginCap * 100).toFixed(0)}% policy cap`,
      },
      {
        label: "Daily loss budget",
        value: dailyLoss,
        cap: Math.max(dailyLossCap, 1),
        format: (value) => formatCompactUsd(value),
        hint: `Realised + unrealised drawdown vs ${(thresholds.daily_max_loss_pct * 100).toFixed(1)}% kill-switch`,
      },
      {
        label: "Portfolio volatility",
        value: period.annualizedVol * 100,
        cap: thresholds.target_annual_volatility * 100,
        format: (value) => `${value.toFixed(1)}%`,
        hint: `Annualised ${timeframe} return volatility vs ${(thresholds.target_annual_volatility * 100).toFixed(0)}% target`,
      },
    ];
  }, [account, thresholds, period.annualizedVol, timeframe]);

  const exposureRows = useMemo(() => {
    const byClass = new Map<string, { long: number; short: number }>();
    for (const position of positions) {
      const key = position.asset_class;
      const entry = byClass.get(key) ?? { long: 0, short: 0 };
      if (position.notional >= 0) entry.long += position.notional;
      else entry.short += Math.abs(position.notional);
      byClass.set(key, entry);
    }
    return [...byClass.entries()]
      .map(([assetClass, value]) => ({
        label: ASSET_LABEL[assetClass] ?? assetClass,
        long: value.long,
        short: value.short,
      }))
      .sort((a, b) => b.long + b.short - (a.long + a.short));
  }, [positions]);

  const sortedPositions = useMemo(() => {
    const rows = [...positions];
    const direction = sort.dir === "asc" ? 1 : -1;
    rows.sort((a, b) => {
      const pick = (row: PositionRow): string | number => {
        switch (sort.key) {
          case "root":
            return row.instrument_id;
          case "class":
            return row.asset_class;
          case "qty":
            return row.quantity;
          case "weight":
            return row.weight;
          case "upnl":
            return row.unrealized_pnl;
          case "dtr":
            return row.days_to_roll ?? Number.POSITIVE_INFINITY;
          default:
            return Math.abs(row.notional);
        }
      };
      const left = pick(a);
      const right = pick(b);
      if (typeof left === "string" || typeof right === "string") {
        return String(left).localeCompare(String(right)) * direction;
      }
      return (left - right) * direction;
    });
    return rows;
  }, [positions, sort]);

  const maxWeight = Math.max(...positions.map((position) => Math.abs(position.weight)), 0.0001);

  const toggleSort = (key: SortKey) => {
    setSort((previous) =>
      previous.key === key
        ? { key, dir: previous.dir === "desc" ? "asc" : "desc" }
        : { key, dir: key === "root" || key === "class" ? "asc" : "desc" },
    );
  };

  if (loading && !account) {
    return (
      <div className="grid grid-cols-12 gap-3">
        {Array.from({ length: 6 }).map((_, index) => (
          <Skeleton key={index} className="col-span-2 h-20" />
        ))}
        <Skeleton className="col-span-8 h-72" />
        <Skeleton className="col-span-4 h-72" />
        <Skeleton className="col-span-12 h-72" />
      </div>
    );
  }

  if (!account) {
    return (
      <Panel title="Command Center">
        <EmptyState
          title="Account snapshot unavailable"
          description="The telemetry API did not return an account snapshot and the deterministic fixture could not be built."
        />
      </Panel>
    );
  }

  return (
    <div className="grid min-h-full grid-cols-12 grid-rows-[auto_auto_1fr] gap-3">
      {/* ---------------------------------------------------------------- KPI strip */}
      <div className="col-span-12 grid grid-cols-12 gap-3">
      <KpiCard
        className="col-span-6 sm:col-span-4 lg:col-span-2"
        label="Net liquidation"
        value={formatCurrency(account.net_liquidation)}
        hint={`Cash ${formatCompactUsd(account.cash_balance)} · IM ${formatCompactUsd(account.initial_margin_used)}`}
      />
      <KpiCard
        className="col-span-6 sm:col-span-4 lg:col-span-2"
        label="Day PnL"
        value={formatSignedCurrency(account.day_pnl)}
        valueClassName={signedColor(account.day_pnl)}
        hint={
          <span className="flex items-baseline gap-1.5">
            <span className={signedColor(account.day_pnl_pct)}>{formatPct(account.day_pnl_pct)}</span>
            <span>today</span>
          </span>
        }
      />
      <KpiCard
        className="col-span-6 sm:col-span-4 lg:col-span-2"
        label="Margin utilisation"
        value={formatPctPlain(account.margin_utilization)}
        valueClassName={
          account.margin_utilization >= thresholds.margin_critical_threshold
            ? "text-loss"
            : account.margin_utilization >= thresholds.margin_warning_threshold
              ? "text-warn"
              : "text-zinc-50"
        }
        hint={
          <span className="block">
            <span className="relative mb-1 block h-1.5 w-full overflow-hidden rounded-full bg-zinc-800">
              <span className="absolute inset-y-0 left-0 w-[60%] bg-zinc-600/70" />
              <span className="absolute inset-y-0 left-[60%] w-[20%] bg-warn/50" />
              <span className="absolute inset-y-0 left-[80%] w-[20%] bg-loss/50" />
              <span
                className="absolute inset-y-0 w-[2px] rounded-full bg-zinc-100"
                style={{ left: `${Math.min(100, account.margin_utilization * 100)}%` }}
              />
            </span>
            <span>
              IM {formatCompactUsd(account.initial_margin_used)} · cap{" "}
              {formatPctPlain(thresholds.max_margin_utilization)}
            </span>
          </span>
        }
      />
      <KpiCard
        className="col-span-6 sm:col-span-4 lg:col-span-2"
        label="Gross leverage"
        value={`${account.gross_leverage.toFixed(2)}×`}
        hint={`Net exposure ${formatCompactUsd(account.net_exposure)}`}
      />
      <KpiCard
        className="col-span-6 sm:col-span-4 lg:col-span-2"
        label="Excess liquidity"
        value={formatCompactUsd(account.excess_liquidity)}
        hint={`Buying power ${formatCompactUsd(account.buying_power)}`}
      />
      <KpiCard
        className="col-span-6 sm:col-span-4 lg:col-span-2"
        label="Open positions"
        value={formatInteger(positions.length)}
        hint={
          <span className={signedColor(totals.unrealized)}>
            {formatSignedCurrency(totals.unrealized, true)} unrealised
          </span>
        }
      />

      </div>

      <div className="col-span-12 grid grid-cols-12 gap-3">
      {/* ------------------------------------------------------------ equity panel */}
      <Panel
        className="col-span-12 xl:col-span-8"
        title="Equity & drawdown"
        actions={
          <div className="flex items-center gap-2">
            <Segmented
              ariaLabel="Equity timeframe"
              options={TIMEFRAMES.map((item) => ({ value: item.value, label: item.label }))}
              value={timeframe}
              onChange={setTimeframe}
            />
            <button
              type="button"
              onClick={() => setShowDrawdown((value) => !value)}
              aria-pressed={showDrawdown}
              className={cn(
                "focus-ring rounded-md border px-2 py-1 text-[11px] font-medium transition-colors",
                showDrawdown
                  ? "border-edge bg-surface text-zinc-200"
                  : "border-transparent text-zinc-500 hover:text-zinc-300",
              )}
            >
              Drawdown
            </button>
          </div>
        }
      >
        <div className="flex flex-wrap items-baseline justify-between gap-2 pb-2">
          <div className="flex items-baseline gap-3">
            <span className="font-mono text-[18px] font-semibold tabular-nums text-zinc-50">
              {formatCurrency(period.last)}
            </span>
            <span className={cn("font-mono text-[12px] tabular-nums", signedColor(period.change))}>
              {formatPct(period.change)}
            </span>
            <span className="text-[11px] text-zinc-600">
              {timeframe} · {source === "demo" ? "fixture curve" : "mark-to-market"}
            </span>
          </div>
          <div className="flex items-center gap-3 text-[11px] text-zinc-500">
            <span>
              High <span className="font-mono tabular-nums text-zinc-400">{formatCompactUsd(period.high)}</span>
            </span>
            <span>
              Low <span className="font-mono tabular-nums text-zinc-400">{formatCompactUsd(period.low)}</span>
            </span>
          </div>
        </div>

        <TimeSeriesChart
          data={windowPoints}
          kind="area"
          height={showDrawdown ? 190 : 250}
          color="#3B82F6"
          fillColor="rgba(59,130,246,0.14)"
          formatValue={(value) => formatCompactUsd(value)}
          ariaLabel={`Account equity curve over ${timeframe}`}
        />

        {showDrawdown ? (
          <div className="mt-1 border-t border-hairline pt-1">
            <div className="flex items-center justify-between px-0.5 pt-1">
              <span className="metric-label">Underwater</span>
              <span className="font-mono text-[11px] tabular-nums text-loss">
                max {formatPctPlain(period.maxDrawdown)}
              </span>
            </div>
            <TimeSeriesChart
              data={drawdownSeries}
              kind="baseline"
              height={84}
              showTimeAxis={false}
              formatValue={(value) => formatPctPlain(value)}
              ariaLabel="Underwater drawdown curve"
            />
          </div>
        ) : null}

        <div className="mt-2 grid grid-cols-2 gap-3 border-t border-hairline pt-2 sm:grid-cols-4">
          {[
            { label: "Period change", value: formatPct(period.change), tone: signedColor(period.change) },
            { label: "Max drawdown", value: formatPctPlain(period.maxDrawdown), tone: "text-loss" },
            { label: "Annualised vol", value: formatPctPlain(period.annualizedVol), tone: "text-zinc-200" },
            { label: "Gross notional", value: formatCompactUsd(totals.notional), tone: "text-zinc-200" },
          ].map((item) => (
            <div key={item.label}>
              <div className="metric-label">{item.label}</div>
              <div className={cn("font-mono text-[12.5px] tabular-nums", item.tone)}>{item.value}</div>
            </div>
          ))}
        </div>
      </Panel>

      {/* ------------------------------------------------------- right column */}
      <div className="col-span-12 flex flex-col gap-3 xl:col-span-4">
        <Panel
          title="Risk budget"
          actions={<span className="text-[11px] text-zinc-600">thresholds · API</span>}
        >
          <BulletBars rows={bullets} />
        </Panel>
        <Panel className="flex-1" title="Exposure by asset class">
          {exposureRows.length > 0 ? (
            <ExposureBars rows={exposureRows} format={(value) => formatCompactUsd(value)} />
          ) : (
            <EmptyState title="No open exposure" description="All positions are flat." />
          )}
          <div className="mt-3 grid grid-cols-2 gap-3 border-t border-hairline pt-2">
            <div>
              <div className="metric-label">Long notional</div>
              <div className="font-mono text-[12.5px] tabular-nums text-gain">
                {formatCompactUsd(
                  positions.filter((position) => position.notional > 0).reduce((sum, position) => sum + position.notional, 0),
                )}
              </div>
            </div>
            <div>
              <div className="metric-label">Short notional</div>
              <div className="font-mono text-[12.5px] tabular-nums text-loss">
                {formatCompactUsd(
                  Math.abs(
                    positions
                      .filter((position) => position.notional < 0)
                      .reduce((sum, position) => sum + position.notional, 0),
                  ),
                )}
              </div>
            </div>
          </div>
        </Panel>
      </div>

      </div>

      {/* --------------------------------------------------------------- book table */}
      <Panel className="col-span-12 min-h-0" title="Book" flush>
        <div className="flex items-center justify-between gap-3 px-3">
          <Tabs
            ariaLabel="Book views"
            options={BOOK_TABS.map((tab) => ({
              value: tab.value,
              label: tab.label,
              count:
                tab.value === "positions"
                  ? positions.length
                  : tab.value === "telemetry"
                    ? events.length
                    : undefined,
            }))}
            value={bookTab}
            onChange={setBookTab}
            className="flex-1"
          />
          <div className="shrink-0 text-[11px] text-zinc-600">
            {source === "demo" ? "DEMO DATA" : "Broker reconciled"} ·{" "}
            {formatInteger(positions.reduce((sum, position) => sum + Math.abs(position.quantity), 0))} contracts
          </div>
        </div>

        {bookTab === "positions" ? (
          <div className="min-h-0 flex-1 overflow-auto">
            <table className="data-table">
              <thead>
                <tr>
                  <th className="w-20">
                    <SortHeader label="Root" active={sort.key === "root"} direction={sort.dir} onClick={() => toggleSort("root")} />
                  </th>
                  <th className="w-24">
                    <SortHeader label="Class" active={sort.key === "class"} direction={sort.dir} onClick={() => toggleSort("class")} />
                  </th>
                  <th className="w-28">Contract</th>
                  <th className="w-20 text-right">
                    <SortHeader label="Qty" active={sort.key === "qty"} direction={sort.dir} onClick={() => toggleSort("qty")} align="right" />
                  </th>
                  <th className="w-24 text-right">Avg price</th>
                  <th className="w-24 text-right">Mark</th>
                  <th className="w-36">
                    <SortHeader label="w_i" active={sort.key === "weight"} direction={sort.dir} onClick={() => toggleSort("weight")} />
                  </th>
                  <th className="w-28 text-right">
                    <SortHeader label="Unrealised" active={sort.key === "upnl"} direction={sort.dir} onClick={() => toggleSort("upnl")} align="right" />
                  </th>
                  <th className="w-28 text-right">
                    <SortHeader label="Notional" active={sort.key === "notional"} direction={sort.dir} onClick={() => toggleSort("notional")} align="right" />
                  </th>
                  <th className="w-24 text-right">Margin</th>
                  <th className="w-24 text-right">
                    <SortHeader label="Days to roll" active={sort.key === "dtr"} direction={sort.dir} onClick={() => toggleSort("dtr")} align="right" />
                  </th>
                </tr>
              </thead>
              <tbody>
                {sortedPositions.map((position) => {
                  const rollWarning = position.days_to_roll !== null && position.days_to_roll <= 5;
                  return (
                    <tr key={position.instrument_id}>
                      <td className="font-medium text-zinc-100">{position.instrument_id}</td>
                      <td className="text-zinc-500">{ASSET_LABEL[position.asset_class] ?? position.asset_class}</td>
                      <td className="num text-[11px] text-zinc-500">{position.contract_symbol ?? "—"}</td>
                      <td className={cn("num text-right font-medium", position.quantity > 0 ? "text-gain" : "text-loss")}>
                        {position.quantity > 0 ? `+${position.quantity}` : position.quantity}
                      </td>
                      <td className="num text-right text-zinc-400">{formatNumber(position.average_price, 2)}</td>
                      <td className="num text-right text-zinc-200">{formatNumber(position.market_price, 2)}</td>
                      <td>
                        <div className="flex items-center gap-2">
                          <WeightBar weight={position.weight} maxAbs={maxWeight} />
                          <span className={cn("font-mono text-[11.5px] tabular-nums", signedColor(position.weight))}>
                            {position.weight >= 0 ? "+" : "−"}
                            {Math.abs(position.weight).toFixed(3)}
                          </span>
                        </div>
                      </td>
                      <td className={cn("num text-right font-medium", signedColor(position.unrealized_pnl))}>
                        {formatSignedCurrency(position.unrealized_pnl)}
                      </td>
                      <td className="num text-right text-zinc-400">{formatCurrency(position.notional, true)}</td>
                      <td className="num text-right text-zinc-500">
                        {formatCompactUsd(Math.abs(position.quantity) * position.initial_margin)}
                      </td>
                      <td className="num text-right">
                        {rollWarning ? (
                          <Badge tone="warn">≤5d</Badge>
                        ) : (
                          <span className="text-zinc-400">{position.days_to_roll ?? "—"}</span>
                        )}
                      </td>
                    </tr>
                  );
                })}
                {sortedPositions.length === 0 ? (
                  <tr>
                    <td colSpan={11} className="py-8 text-center text-zinc-600">
                      No open positions.
                    </td>
                  </tr>
                ) : null}
              </tbody>
            </table>
          </div>
        ) : null}

        {bookTab === "orders" ? (
          <EmptyState
            title="No working orders"
            description="The telemetry API exposes the position book, not the OMS working-order queue. Working orders remain inside the execution engine and are surfaced through the telemetry console when live."
          />
        ) : null}

        {bookTab === "fills" ? (
          <EmptyState
            title="No fills endpoint"
            description="Fill history is not exposed by the telemetry API. Emergency flatten receipts are listed in the Flatten All dialog, and raw execution frames are available in the telemetry console."
          />
        ) : null}

        {bookTab === "telemetry" ? (
          <div className="min-h-0 flex-1 overflow-auto">
            <table className="data-table">
              <thead>
                <tr>
                  <th className="w-24">Time</th>
                  <th className="w-28">Kind</th>
                  <th>Payload</th>
                  <th className="w-16 text-right">Seq</th>
                </tr>
              </thead>
              <tbody>
                {events.length === 0 ? (
                  <tr>
                    <td colSpan={4} className="py-8 text-center text-zinc-600">
                      Waiting for telemetry frames…
                    </td>
                  </tr>
                ) : (
                  events.slice(0, 60).map((frame, index) => (
                    <tr key={`${frame.sequence}-${frame.kind}-${index}`}>
                      <td className="num text-zinc-500">{formatClock(frame.timestamp)}</td>
                      <td>
                        <Badge tone={frame.kind === "risk_alert" ? "loss" : frame.kind === "pnl" ? "gain" : "accent"}>
                          {frame.kind}
                        </Badge>
                      </td>
                      <td className="max-w-0 truncate text-zinc-400" title={summarizeTelemetry(frame)}>
                        {summarizeTelemetry(frame)}
                      </td>
                      <td className="num text-right text-zinc-600">{frame.sequence}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        ) : null}
      </Panel>
    </div>
  );
}
