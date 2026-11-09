"use client";

import { useMemo } from "react";

import { TimeSeriesChart } from "@/components/charts/time-series";
import { Badge, EmptyState, KpiCard, Panel, Skeleton, signedColor } from "@/components/ui/primitives";
import { getTradeJournal } from "@/lib/api";
import { formatCurrency, formatDateTime, formatInteger, formatPct, formatSignedCurrency } from "@/lib/format";
import type { TradeJournalResponse, TradeJournalTrade } from "@/lib/types";
import { useAsyncData } from "@/lib/use-async-data";

function equitySeries(data: TradeJournalResponse) {
  return (data.equity ?? [])
    .filter((row) => row.net_liquidation !== null && row.timestamp_utc)
    .map((row) => ({ time: row.timestamp_utc as string, value: row.net_liquidation as number }));
}

function TradeRow({ row }: { row: TradeJournalTrade }) {
  return (
    <tr className="border-t border-edge/60 text-[11px]">
      <td className="px-2 py-1.5 text-zinc-400">{formatDateTime(row.timestamp_utc)}</td>
      <td className="px-2 py-1.5 font-medium text-zinc-200">{row.instrument_id || "—"}</td>
      <td className="px-2 py-1.5">{row.side || "—"}</td>
      <td className="px-2 py-1.5 font-mono tabular-nums">{row.filled_quantity ?? "—"}</td>
      <td className="px-2 py-1.5 font-mono tabular-nums">{formatCurrency(row.average_price)}</td>
      <td className="px-2 py-1.5 font-mono tabular-nums">{formatCurrency(row.commission)}</td>
      <td className="px-2 py-1.5">
        <Badge tone={row.status === "FILLED" ? "gain" : row.status === "REJECTED" ? "loss" : "neutral"}>
          {row.status || "UNKNOWN"}
        </Badge>
      </td>
      <td className="max-w-[260px] truncate px-2 py-1.5 text-zinc-500" title={row.message ?? undefined}>
        {row.message || "—"}
      </td>
    </tr>
  );
}

export function JournalView() {
  const journal = useAsyncData(() => getTradeJournal(500));
  const data = journal.data;
  const series = useMemo(() => (data ? equitySeries(data) : []), [data]);
  const summary = data?.summary;

  if (!data) {
    return (
      <div className="grid min-h-full grid-cols-12 gap-3">
        <Skeleton className="col-span-12 h-11" />
        <Skeleton className="col-span-12 h-72" />
      </div>
    );
  }

  const hasRows = series.length > 0 || (data.trades?.length ?? 0) > 0;
  const empty = !hasRows;

  return (
    <div className="grid min-h-full grid-cols-12 gap-3">
      <Panel
        className="col-span-12"
        title="Paper / live trade journal"
        actions={
          <div className="flex items-center gap-2">
            <Badge tone={data.enabled ? "gain" : "warn"}>
              {data.enabled ? "LIVE JOURNAL ENABLED" : "JOURNAL INACTIVE"}
            </Badge>
            <a className="text-[11px] text-accent hover:underline" href="/api/runner/trade-journal/trades.csv">
              trades.csv
            </a>
            <a className="text-[11px] text-accent hover:underline" href="/api/runner/trade-journal/equity.csv">
              equity.csv
            </a>
          </div>
        }
      >
        <p className="text-[11.5px] text-zinc-500">
          {empty
            ? "No journal rows yet. Start the paper/live runner; every order and account snapshot is written locally."
            : `${summary?.trade_rows ?? 0} trade rows · ${summary?.equity_points ?? 0} equity points · files under ${data.directory}`}
        </p>
        {journal.error ? (
          <p className="mt-1 text-[11px] text-warn">Journal API unavailable: {journal.error}</p>
        ) : null}
        {!data.enabled ? (
          <p className="mt-1 text-[11px] text-warn">
            Trading environment is not paper/live; the runner only writes the journal in those modes.
          </p>
        ) : null}
      </Panel>

      <KpiCard
        label="Starting equity"
        value={formatCurrency(summary?.starting_equity)}
        hint="First logged account snapshot"
      />
      <KpiCard
        label="Current equity"
        value={formatCurrency(summary?.latest_equity)}
        hint="Latest logged account snapshot"
      />
      <KpiCard
        label="Total PnL"
        value={formatSignedCurrency(summary?.total_pnl)}
        valueClassName={signedColor(summary?.total_pnl)}
        hint="Current − starting equity"
      />
      <KpiCard
        label="Return"
        value={formatPct((summary?.total_return ?? 0) * 100)}
        valueClassName={signedColor(summary?.total_return)}
        hint="Benchmark return"
      />
      <KpiCard
        label="Max drawdown"
        value={formatPct((summary?.max_drawdown ?? 0) * 100)}
        valueClassName={(summary?.max_drawdown ?? 0) > 0 ? "text-loss" : undefined}
        hint="Peak-to-trough on logged equity"
      />
      <KpiCard
        label="Filled / rows"
        value={`${formatInteger(summary?.filled_trades)} / ${formatInteger(summary?.trade_rows)}`}
        hint={`Last trade ${formatDateTime(summary?.last_trade_at)}`}
      />

      <Panel className="col-span-12" title="Account equity benchmark" flush>
        {series.length > 1 ? (
          <TimeSeriesChart
            data={series}
            kind="area"
            height={320}
            color="#22C55E"
            fillColor="rgba(34,197,94,0.14)"
            formatValue={(value) => formatCurrency(value, true)}
            ariaLabel="Paper/live account equity curve"
          />
        ) : (
          <div className="p-6">
            <EmptyState
              title={empty ? "Journal is empty" : "Not enough equity points yet"}
              description={
                empty
                  ? "Run the paper/live runner for at least one state. The runner writes trades.csv and equity.csv locally; no synthetic curve is shown."
                  : "At least two account snapshots are required to plot the benchmark curve."
              }
            />
          </div>
        )}
      </Panel>

      <Panel className="col-span-12" title="Recent fills" flush>
        {data.trades?.length ? (
          <div className="max-h-[420px] overflow-auto">
            <table className="w-full border-collapse text-left">
              <thead className="sticky top-0 bg-panel text-[10.5px] uppercase tracking-wide text-zinc-500">
                <tr>
                  <th className="px-2 py-1.5">Time (UTC)</th>
                  <th className="px-2 py-1.5">Root</th>
                  <th className="px-2 py-1.5">Side</th>
                  <th className="px-2 py-1.5">Qty</th>
                  <th className="px-2 py-1.5">Avg price</th>
                  <th className="px-2 py-1.5">Comm.</th>
                  <th className="px-2 py-1.5">Status</th>
                  <th className="px-2 py-1.5">Message</th>
                </tr>
              </thead>
              <tbody>
                {data.trades.map((row, index) => (
                  <TradeRow key={`${row.order_id ?? "row"}-${index}`} row={row} />
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="p-6">
            <EmptyState
              title="No trades logged yet"
              description="The runner appends one row per order acknowledgement once paper/live execution starts."
            />
          </div>
        )}
      </Panel>
    </div>
  );
}
