/**
 * Pure equity-series helpers shared by the account provider and the Command
 * Center chart.
 *
 * The telemetry stream can deliver intraday ticks while the historical equity
 * curve is daily. Appending raw tick timestamps to a daily series produced a
 * flat intraday tail on every timeframe and made the `3M`/`6M`/… charts drift
 * away from the real mark-to-market close.
 *
 * Contract:
 * - `1D` is intraday: append the tick at its own timestamp, or update the last
 *   point in place when the tick falls within the same second.
 * - every daily timeframe (`1W/1M/3M/6M/1Y/ALL`) is daily: NEVER append an
 *   intraday timestamp — update the value of the last daily bar in place.
 * - stale ticks (older than the last point) are ignored.
 *
 * The helpers are deterministic and side-effect free so they can be unit-tested
 * without a DOM (see the headless verification in the Phase 9 report).
 */

import type { EquityPoint } from "./types";

export type EquityTimeframe = "1D" | "1W" | "1M" | "3M" | "6M" | "1Y" | "ALL";

/** Trading-day window rendered per timeframe (1D is intraday). */
export const TIMEFRAME_BARS: Record<EquityTimeframe, number> = {
  "1D": 2,
  "1W": 5,
  "1M": 21,
  "3M": 63,
  "6M": 126,
  "1Y": 252,
  ALL: Number.POSITIVE_INFINITY,
};

export function isIntradayTimeframe(timeframe: EquityTimeframe): boolean {
  return timeframe === "1D";
}

function parseMs(iso: string): number {
  const value = new Date(iso).getTime();
  return Number.isFinite(value) ? value : Number.NaN;
}

function sanitize(series: readonly EquityPoint[]): EquityPoint[] {
  return series
    .filter((point) => Number.isFinite(parseMs(point.time)) && Number.isFinite(point.value))
    .sort((a, b) => parseMs(a.time) - parseMs(b.time));
}

/**
 * Merge one live telemetry equity point into a chart series.
 *
 * Returns a NEW array (never mutates the input) so React state updates stay
 * referentially correct.
 */
export function mergeTelemetryIntoSeries(
  series: readonly EquityPoint[],
  tick: EquityPoint,
  timeframe: EquityTimeframe,
): EquityPoint[] {
  const base = sanitize(series);
  const tickMs = parseMs(tick.time);
  if (!Number.isFinite(tickMs) || !Number.isFinite(tick.value)) return base;

  if (isIntradayTimeframe(timeframe)) {
    if (base.length === 0) return [{ ...tick }];
    const last = base[base.length - 1];
    if (!last) return [{ ...tick }];
    const lastMs = parseMs(last.time);
    if (tickMs < lastMs) return base; // stale tick: never move the curve backwards
    if (tickMs - lastMs < 1_000) {
      // Same second — merge in place instead of appending a duplicate timestamp.
      return [...base.slice(0, -1), { ...tick }];
    }
    return [...base, { ...tick }];
  }

  // Daily timeframes: update the last daily bar in place, keep its timestamp.
  if (base.length === 0) return [{ ...tick }];
  const last = base[base.length - 1];
  if (!last) return [{ ...tick }];
  const lastMs = parseMs(last.time);
  if (tickMs < lastMs) return base;
  return [...base.slice(0, -1), { time: last.time, value: tick.value }];
}

/** Underwater drawdown series computed from the exact displayed equity window. */
export function deriveUnderwaterSeries(series: readonly EquityPoint[]): EquityPoint[] {
  let peak = Number.NEGATIVE_INFINITY;
  return sanitize(series).map((point) => {
    peak = Math.max(peak, point.value);
    return {
      time: point.time,
      value: peak > 0 ? point.value / peak - 1 : 0,
    };
  });
}

/** Slice a daily series to the visible trading-day window of a timeframe. */
export function sliceTimeframe(
  series: readonly EquityPoint[],
  timeframe: EquityTimeframe,
): EquityPoint[] {
  const clean = sanitize(series);
  const bars = TIMEFRAME_BARS[timeframe];
  if (!Number.isFinite(bars)) return clean;
  return clean.slice(-Math.max(2, bars));
}
