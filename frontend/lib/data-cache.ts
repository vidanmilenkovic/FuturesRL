import type { CachedBarsMeta, DataOrigin, DataTimeframe, MarketDataMode } from "./types";

export const DATA_TIMEFRAME_OPTIONS: { value: DataTimeframe; label: string }[] = [
  { value: "1m", label: "1 minute" },
  { value: "5m", label: "5 minutes" },
  { value: "15m", label: "15 minutes" },
  { value: "30m", label: "30 minutes" },
  { value: "1h", label: "1 hour" },
  { value: "4h", label: "4 hours" },
  { value: "1d", label: "1 day" },
  { value: "1w", label: "1 week" },
];

export const DATA_TIMEFRAMES: DataTimeframe[] = DATA_TIMEFRAME_OPTIONS.map((item) => item.value);

export const DATA_MODE_OPTIONS: { value: MarketDataMode; label: string }[] = [
  { value: "auto", label: "Auto — real cached bars if available, else synthetic" },
  { value: "cached_real", label: "Cached real Parquet data only" },
  { value: "synthetic_offline", label: "Synthetic offline fixture" },
];

export function dataOriginBadge(origin: DataOrigin | null | undefined): {
  label: string;
  real: boolean;
  tone: "gain" | "neutral";
  title: string;
} {
  if (origin === "real_parquet") {
    return {
      label: "REAL PARQUET DATA",
      real: true,
      tone: "gain",
      title: "Run/study used real cached Parquet bars",
    };
  }
  return {
    label: "SYNTHETIC FIXTURE",
    real: false,
    tone: "neutral",
    title: "Run/study used the synthetic offline fixture (no real cached bars)",
  };
}

export function cachedBarsCount(value: number | CachedBarsMeta | null | undefined): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (value && typeof value === "object") {
    if (typeof value.bars === "number") return value.bars;
    if (typeof value.bar_count === "number") return value.bar_count;
  }
  return null;
}

export function cachedBarsDetail(value: number | CachedBarsMeta | null | undefined): string | null {
  const count = cachedBarsCount(value);
  if (value && typeof value === "object") {
    const parts: string[] = [];
    if (count !== null) parts.push(`${count.toLocaleString("en-US")} bars`);
    if (value.start_date) parts.push(value.start_date);
    if (value.end_date) parts.push(value.end_date);
    if (Array.isArray(value.roots) && value.roots.length > 0) parts.push(`${value.roots.length} roots`);
    if (parts.length > 0) return parts.join(" · ");
  }
  return count === null ? null : `${count.toLocaleString("en-US")} cached bars`;
}

export function formatBytes(bytes: number | null | undefined): string {
  if (typeof bytes !== "number" || !Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let index = 0;
  while (value >= 1024 && index < units.length - 1) {
    value /= 1024;
    index += 1;
  }
  return `${value.toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}
