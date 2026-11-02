"use client";

import { useMemo, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Segmented } from "@/components/ui/controls";
import { cn } from "@/lib/cn";
import {
  formatCurrency,
  formatDate,
  formatInteger,
  formatNumber,
  formatSignedCurrency,
  formatWeight,
  pnlTone,
} from "@/lib/format";
import type { AssetClass, PositionRow } from "@/lib/types";

type Filter = "ALL" | "EQUITY_INDEX" | "RATES" | "ENERGY" | "METALS";

const FILTERS: { value: Filter; label: string }[] = [
  { value: "ALL", label: "All" },
  { value: "EQUITY_INDEX", label: "Equity" },
  { value: "RATES", label: "Rates" },
  { value: "ENERGY", label: "Energy" },
  { value: "METALS", label: "Metals" },
];

const ASSET_LABEL: Record<AssetClass, string> = {
  equity_index: "EQ",
  rates: "RATES",
  energy: "ENRGY",
  metals: "METAL",
  fx: "FX",
  agriculture: "AG",
  crypto: "CRYPTO",
  unknown: "—",
};

export function PositionsTable({ positions }: { positions: PositionRow[] }) {
  const [filter, setFilter] = useState<Filter>("ALL");

  const rows = useMemo(() => {
    const filtered =
      filter === "ALL"
        ? positions
        : positions.filter((position) => position.asset_class === filter.toLowerCase());
    return [...filtered].sort((left, right) => Math.abs(right.notional) - Math.abs(left.notional));
  }, [positions, filter]);

  const totals = useMemo(
    () =>
      rows.reduce(
        (accumulator, position) => ({
          notional: accumulator.notional + position.notional,
          unrealized: accumulator.unrealized + position.unrealized_pnl,
          margin: accumulator.margin + Math.abs(position.quantity) * position.initial_margin,
        }),
        { notional: 0, unrealized: 0, margin: 0 },
      ),
    [rows],
  );

  return (
    <div>
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <p className="font-mono text-[10px] uppercase tracking-[0.12em] text-terminal-400">
          {rows.length} active contracts · {formatInteger(rows.reduce((sum, row) => sum + Math.abs(row.quantity), 0))} lots
        </p>
        <Segmented size="sm" ariaLabel="Filter positions by asset class" value={filter} onChange={setFilter} options={FILTERS} />
      </div>

      <div className="-mx-3 overflow-x-auto">
        <table className="data-table min-w-[1040px]">
          <thead>
            <tr>
              <th>root</th>
              <th>contract</th>
              <th>class</th>
              <th>expiry</th>
              <th className="num">days to roll</th>
              <th className="num">qty</th>
              <th className="num">avg px</th>
              <th className="num">mark</th>
              <th className="num">w_i</th>
              <th className="num">notional</th>
              <th className="num">unreal. pnl</th>
              <th className="num">margin</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((position) => {
              const tone = pnlTone(position.unrealized_pnl);
              const long = position.quantity > 0;
              const rollImminent = position.days_to_roll !== null && position.days_to_roll <= 7;
              return (
                <tr key={position.instrument_id}>
                  <td>
                    <span className="flex items-center gap-2">
                      <span
                        aria-hidden
                        className={cn("h-3.5 w-0.5 rounded-full", long ? "bg-signal-up" : "bg-signal-down")}
                      />
                      <span className="font-semibold text-terminal-100">{position.root_symbol}</span>
                    </span>
                  </td>
                  <td className="text-terminal-300">{position.contract_symbol ?? "—"}</td>
                  <td>
                    <Badge tone="neutral">{ASSET_LABEL[position.asset_class]}</Badge>
                  </td>
                  <td className="text-terminal-300">{formatDate(position.expiration)}</td>
                  <td className="num">
                    <span className="inline-flex items-center gap-1.5">
                      {rollImminent ? <Badge tone="warn" dot>roll</Badge> : null}
                      <span className={rollImminent ? "text-signal-warn" : "text-terminal-200"}>
                        {position.days_to_roll ?? "—"}
                      </span>
                    </span>
                  </td>
                  <td className={cn("num font-semibold", long ? "text-signal-up" : "text-signal-down")}>
                    {position.quantity > 0 ? `+${position.quantity}` : position.quantity}
                  </td>
                  <td className="num text-terminal-300">{formatNumber(position.average_price, position.multiplier > 100 ? 3 : 2)}</td>
                  <td className="num text-terminal-100">{formatNumber(position.market_price, position.multiplier > 100 ? 3 : 2)}</td>
                  <td className="num">
                    <span className={cn("inline-flex items-center gap-2", long ? "text-signal-up" : "text-signal-down")}>
                      <span aria-hidden className="hidden h-1 w-10 overflow-hidden rounded-full bg-terminal-700 sm:inline-block">
                        <span
                          className={cn("block h-full rounded-full", long ? "bg-signal-up" : "bg-signal-down")}
                          style={{ width: `${Math.min(100, Math.abs(position.weight) * 320)}%` }}
                        />
                      </span>
                      {formatWeight(position.weight)}
                    </span>
                  </td>
                  <td className="num text-terminal-200">{formatCurrency(position.notional, true)}</td>
                  <td
                    className={cn(
                      "num font-semibold",
                      tone === "up" ? "text-signal-up" : tone === "down" ? "text-signal-down" : "text-terminal-300",
                    )}
                  >
                    {formatSignedCurrency(position.unrealized_pnl)}
                  </td>
                  <td className="num text-terminal-300">
                    {formatCurrency(Math.abs(position.quantity) * position.initial_margin, true)}
                  </td>
                </tr>
              );
            })}
            {rows.length === 0 ? (
              <tr>
                <td colSpan={12} className="py-8 text-center text-terminal-400">
                  No positions in this asset class.
                </td>
              </tr>
            ) : null}
          </tbody>
          <tfoot>
            <tr className="border-t border-terminal-500/60 bg-terminal-800/60">
              <td colSpan={4} className="px-2.5 py-2 font-mono text-[10px] uppercase tracking-[0.12em] text-terminal-300">
                portfolio total
              </td>
              <td />
              <td className="num font-semibold text-terminal-100">
                {rows.reduce((sum, row) => sum + Math.abs(row.quantity), 0)}
              </td>
              <td colSpan={2} />
              <td className="num text-terminal-300">
                {formatWeight(rows.reduce((sum, row) => sum + row.weight, 0))}
              </td>
              <td className="num font-semibold text-terminal-100">{formatCurrency(totals.notional, true)}</td>
              <td
                className={cn(
                  "num font-semibold",
                  totals.unrealized >= 0 ? "text-signal-up" : "text-signal-down",
                )}
              >
                {formatSignedCurrency(totals.unrealized)}
              </td>
              <td className="num text-terminal-100">{formatCurrency(totals.margin, true)}</td>
            </tr>
          </tfoot>
        </table>
      </div>
    </div>
  );
}
