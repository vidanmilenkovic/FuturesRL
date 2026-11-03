"use client";

import { useEffect, useRef, useState } from "react";

export interface SeriesPoint {
  time: string;
  value: number;
}

export interface TimeSeriesChartProps {
  data: SeriesPoint[];
  kind?: "area" | "line" | "baseline";
  height: number;
  color?: string;
  fillColor?: string;
  /** Optional dashed comparison series rendered on the same scale. */
  comparison?: SeriesPoint[];
  comparisonColor?: string;
  formatValue?: (value: number) => string;
  formatTime?: (iso: string) => string;
  showTimeAxis?: boolean;
  showPriceAxis?: boolean;
  precision?: number;
  ariaLabel: string;
  className?: string;
}

const DEFAULT_FORMAT = (value: number): string =>
  value.toLocaleString("en-US", { maximumFractionDigits: 2 });

function defaultFormatTime(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleString("en-US", {
    month: "short",
    day: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
    timeZone: "UTC",
  });
}

/**
 * TradingView Lightweight Charts wrapper: area/line/baseline series with a
 * custom crosshair tooltip rendered in the DOM. The library is imported lazily
 * inside the effect so it never executes during SSR/pre-render.
 */
export function TimeSeriesChart({
  data,
  kind = "area",
  height,
  color = "#3B82F6",
  fillColor = "rgba(59,130,246,0.16)",
  comparison,
  comparisonColor = "#A1A1AA",
  formatValue = DEFAULT_FORMAT,
  formatTime = defaultFormatTime,
  showTimeAxis = true,
  showPriceAxis = true,
  precision = 2,
  ariaLabel,
  className,
}: TimeSeriesChartProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<import("lightweight-charts").IChartApi | null>(null);
  const seriesRef = useRef<import("lightweight-charts").ISeriesApi<
    "Area" | "Line" | "Baseline"
  > | null>(null);
  const comparisonRef = useRef<import("lightweight-charts").ISeriesApi<"Line"> | null>(null);
  const [tooltip, setTooltip] = useState<{
    left: number;
    top: number;
    time: number;
    value: number;
    comparison?: number;
  } | null>(null);
  const dataRef = useRef(data);
  dataRef.current = data;
  const comparisonDataRef = useRef(comparison);
  comparisonDataRef.current = comparison;

  // Create the chart once per series kind; data updates use setData.
  useEffect(() => {
    let disposed = false;
    const container = containerRef.current;

    setTooltip(null);
    void (async () => {
      const lw = await import("lightweight-charts");
      if (!container || disposed) return;
      const monoFamily =
        getComputedStyle(document.body).getPropertyValue("--font-jetbrains").trim() ||
        "ui-monospace, monospace";
      const chart = lw.createChart(container, {
        height,
        autoSize: true,
        layout: {
          background: { color: "transparent" },
          textColor: "#71717A",
          fontSize: 10,
          fontFamily: `"${monoFamily.replace(/"/g, "")}", ui-monospace, monospace`,
          attributionLogo: false,
        },
        grid: {
          vertLines: { color: "#1E1E22" },
          horzLines: { color: "#1E1E22" },
        },
        rightPriceScale: {
          visible: showPriceAxis,
          borderVisible: false,
          scaleMargins: { top: 0.12, bottom: 0.08 },
        },
        timeScale: {
          visible: showTimeAxis,
          borderVisible: false,
          timeVisible: true,
          secondsVisible: false,
          rightOffset: 2,
          fixLeftEdge: true,
        },
        crosshair: {
          mode: lw.CrosshairMode.Normal,
          vertLine: { color: "#3F3F46", width: 1, labelBackgroundColor: "#27272A" },
          horzLine: { color: "#3F3F46", width: 1, labelBackgroundColor: "#27272A" },
        },
        localization: {
          locale: "en-US",
          priceFormatter: (price: number) => formatValue(price),
        },
        handleScale: { axisPressedMouseMove: { time: true, price: false } },
      });
      chartRef.current = chart;

      if (kind === "line") {
        seriesRef.current = chart.addSeries(lw.LineSeries, {
          color,
          lineWidth: 2,
          priceLineVisible: false,
          lastValueVisible: false,
          crosshairMarkerRadius: 3,
        });
      } else if (kind === "baseline") {
        seriesRef.current = chart.addSeries(lw.BaselineSeries, {
          baseValue: { type: "price", price: 0 },
          topLineColor: "rgba(244,63,94,0)",
          topFillColor1: "rgba(244,63,94,0)",
          topFillColor2: "rgba(244,63,94,0)",
          bottomLineColor: "#F43F5E",
          bottomFillColor1: "rgba(244,63,94,0.02)",
          bottomFillColor2: "rgba(244,63,94,0.18)",
          priceLineVisible: false,
          lastValueVisible: false,
          crosshairMarkerRadius: 3,
        });
      } else {
        seriesRef.current = chart.addSeries(lw.AreaSeries, {
          lineColor: color,
          topColor: fillColor,
          bottomColor: "rgba(9,9,11,0)",
          lineWidth: 2,
          priceLineVisible: false,
          lastValueVisible: false,
          crosshairMarkerRadius: 3,
        });
      }

      if (comparison && comparison.length > 0) {
        comparisonRef.current = chart.addSeries(lw.LineSeries, {
          color: comparisonColor,
          lineWidth: 1,
          lineStyle: lw.LineStyle.Dashed,
          priceLineVisible: false,
          lastValueVisible: false,
          crosshairMarkerVisible: false,
        });
      }

      const formatTooltipValue = (value: unknown): string =>
        typeof value === "number" ? formatValue(value) : "—";

      chart.subscribeCrosshairMove((param) => {
        const series = seriesRef.current;
        if (!series || !param.point || param.time === undefined) {
          setTooltip(null);
          return;
        }
        const datum = param.seriesData.get(series) as { value?: number } | undefined;
        const value = datum?.value;
        if (typeof value !== "number") {
          setTooltip(null);
          return;
        }
        const comparisonSerie = comparisonRef.current;
        const comparisonValue = comparisonSerie
          ? (param.seriesData.get(comparisonSerie) as { value?: number } | undefined)?.value
          : undefined;
        setTooltip({
          left: param.point.x,
          top: param.point.y,
          time: typeof param.time === "number" ? param.time : Number(param.time),
          value,
          comparison: typeof comparisonValue === "number" ? comparisonValue : undefined,
        });
        void formatTooltipValue;
      });

      // Seed the freshly created chart with the current series (a chart can be
      // re-created on height/layout changes while the data prop is unchanged).
      const seedData = dataRef.current;
      if (seriesRef.current && seedData.length > 0) {
        const seen = new Set<number>();
        seriesRef.current.setData(
          seedData
            .map((point) => ({
              time: Math.floor(new Date(point.time).getTime() / 1000) as import("lightweight-charts").UTCTimestamp,
              value: point.value,
            }))
            .filter((row) => Number.isFinite(row.time) && !seen.has(row.time) && seen.add(row.time))
            .sort((a, b) => a.time - b.time),
        );
        chart.timeScale().fitContent();
      }
      const seedComparison = comparisonDataRef.current;
      if (comparisonRef.current && seedComparison && seedComparison.length > 0) {
        const seen = new Set<number>();
        comparisonRef.current.setData(
          seedComparison
            .map((point) => ({
              time: Math.floor(new Date(point.time).getTime() / 1000) as import("lightweight-charts").UTCTimestamp,
              value: point.value,
            }))
            .filter((row) => Number.isFinite(row.time) && !seen.has(row.time) && seen.add(row.time))
            .sort((a, b) => a.time - b.time),
        );
      }

      if (disposed) chart.remove();
    })();

    return () => {
      disposed = true;
      seriesRef.current = null;
      comparisonRef.current = null;
      chartRef.current?.remove();
      chartRef.current = null;
    };
    // Chart is re-created only when the series shape/axis layout changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [kind, height, showTimeAxis, showPriceAxis, precision]);

  useEffect(() => {
    const series = seriesRef.current;
    if (!series || data.length === 0) return;
    const seen = new Set<number>();
    const rows = data
      .map((point) => ({
        time: Math.floor(new Date(point.time).getTime() / 1000) as import("lightweight-charts").UTCTimestamp,
        value: point.value,
      }))
      .filter((row) => Number.isFinite(row.time) && !seen.has(row.time) && seen.add(row.time))
      .sort((a, b) => a.time - b.time);
    series.setData(rows);
    chartRef.current?.timeScale().fitContent();
  }, [data]);

  useEffect(() => {
    const series = comparisonRef.current;
    if (!series || !comparison || comparison.length === 0) return;
    const seen = new Set<number>();
    series.setData(
      comparison
        .map((point) => ({
          time: Math.floor(new Date(point.time).getTime() / 1000) as import("lightweight-charts").UTCTimestamp,
          value: point.value,
        }))
        .filter((row) => Number.isFinite(row.time) && !seen.has(row.time) && seen.add(row.time))
        .sort((a, b) => a.time - b.time),
    );
  }, [comparison]);

  return (
    <div className={`relative ${className ?? ""}`} style={{ height }}>
      <div ref={containerRef} role="img" aria-label={ariaLabel} className="h-full w-full" />
      {tooltip && tooltip.time ? (
        <div
          className="pointer-events-none absolute z-10 min-w-[104px] rounded-md border border-edge bg-surface px-2 py-1.5 shadow-popover"
          style={{
            left: Math.min(Math.max(tooltip.left + 12, 8), 520),
            top: Math.max(tooltip.top - 44, 4),
          }}
        >
          <div className="text-[10.5px] leading-4 text-zinc-500">
            {formatTime(new Date(tooltip.time * 1000).toISOString())}
          </div>
          <div className="font-mono text-[12px] font-medium tabular-nums text-zinc-100">
            {formatValue(tooltip.value)}
          </div>
          {tooltip.comparison !== undefined ? (
            <div className="font-mono text-[11px] tabular-nums text-zinc-400">
              {formatValue(tooltip.comparison)}
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Multi-series overlay (Backtest Lab: 2–3 runs on one chart)
// ---------------------------------------------------------------------------

export interface OverlaySeries {
  id: string;
  label: string;
  color: string;
  data: SeriesPoint[];
}

/**
 * Lightweight Charts overlay for 2–3 aligned series (equity / underwater).
 * Renders one line per run with a shared crosshair tooltip listing every series
 * value at the hovered timestamp.
 */
export function MultiSeriesChart({
  series,
  height,
  formatValue = DEFAULT_FORMAT,
  formatTime = defaultFormatTime,
  showTimeAxis = true,
  ariaLabel,
  className,
}: {
  series: OverlaySeries[];
  height: number;
  formatValue?: (value: number) => string;
  formatTime?: (iso: string) => string;
  showTimeAxis?: boolean;
  ariaLabel: string;
  className?: string;
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<import("lightweight-charts").IChartApi | null>(null);
  const seriesRefs = useRef<Map<string, import("lightweight-charts").ISeriesApi<"Line">>>(new Map());
  const seriesDataRef = useRef(series);
  seriesDataRef.current = series;
  const [tooltip, setTooltip] = useState<{
    left: number;
    top: number;
    time: number;
    values: { id: string; label: string; color: string; value: number }[];
  } | null>(null);
  const idsKey = series.map((entry) => entry.id).join("|");

  useEffect(() => {
    let disposed = false;
    const container = containerRef.current;
    const seriesMap = seriesRefs.current;
    setTooltip(null);

    void (async () => {
      const lw = await import("lightweight-charts");
      if (!container || disposed) return;
      const monoFamily =
        getComputedStyle(document.body).getPropertyValue("--font-jetbrains").trim() ||
        "ui-monospace, monospace";
      const chart = lw.createChart(container, {
        height,
        autoSize: true,
        layout: {
          background: { color: "transparent" },
          textColor: "#71717A",
          fontSize: 10,
          fontFamily: `"${monoFamily.replace(/"/g, "")}", ui-monospace, monospace`,
          attributionLogo: false,
        },
        grid: {
          vertLines: { color: "#1E1E22" },
          horzLines: { color: "#1E1E22" },
        },
        rightPriceScale: { borderVisible: false, scaleMargins: { top: 0.12, bottom: 0.08 } },
        timeScale: {
          visible: showTimeAxis,
          borderVisible: false,
          timeVisible: true,
          secondsVisible: false,
          rightOffset: 2,
          fixLeftEdge: true,
        },
        crosshair: {
          mode: lw.CrosshairMode.Normal,
          vertLine: { color: "#3F3F46", width: 1, labelBackgroundColor: "#27272A" },
          horzLine: { color: "#3F3F46", width: 1, labelBackgroundColor: "#27272A" },
        },
        localization: { locale: "en-US", priceFormatter: (price: number) => formatValue(price) },
        handleScale: { axisPressedMouseMove: { time: true, price: false } },
      });
      chartRef.current = chart;
      seriesMap.clear();

      for (const entry of seriesDataRef.current) {
        const line = chart.addSeries(lw.LineSeries, {
          color: entry.color,
          lineWidth: 2,
          priceLineVisible: false,
          lastValueVisible: false,
          crosshairMarkerRadius: 3,
        });
        seriesMap.set(entry.id, line);
        line.setData(toAscending(entry.data));
      }
      chart.timeScale().fitContent();

      chart.subscribeCrosshairMove((param) => {
        if (!param.point || param.time === undefined) {
          setTooltip(null);
          return;
        }
        const values: { id: string; label: string; color: string; value: number }[] = [];
        for (const entry of seriesDataRef.current) {
          const api = seriesMap.get(entry.id);
          if (!api) continue;
          const datum = param.seriesData.get(api) as { value?: number } | undefined;
          if (typeof datum?.value === "number") {
            values.push({ id: entry.id, label: entry.label, color: entry.color, value: datum.value });
          }
        }
        if (values.length === 0) {
          setTooltip(null);
          return;
        }
        setTooltip({
          left: param.point.x,
          top: param.point.y,
          time: typeof param.time === "number" ? param.time : Number(param.time),
          values,
        });
      });

      if (disposed) chart.remove();
    })();

    return () => {
      disposed = true;
      seriesMap.clear();
      chartRef.current?.remove();
      chartRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [idsKey, height, showTimeAxis]);

  // Refresh every series when the data changes (series ids are stable here; the
  // chart is re-created when the overlay set changes).
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    for (const entry of series) {
      const api = seriesRefs.current.get(entry.id);
      api?.setData(toAscending(entry.data));
    }
    chart.timeScale().fitContent();
  }, [series]);

  return (
    <div className={`relative ${className ?? ""}`} style={{ height }} data-series-count={series.length}>
      <div ref={containerRef} role="img" aria-label={ariaLabel} className="h-full w-full" />
      {tooltip ? (
        <div
          className="pointer-events-none absolute z-10 min-w-[168px] rounded-md border border-edge bg-surface px-2 py-1.5 shadow-popover"
          style={{
            left: Math.min(Math.max(tooltip.left + 12, 8), 720),
            top: Math.max(tooltip.top - 14 - tooltip.values.length * 15, 4),
          }}
        >
          <div className="text-[10.5px] leading-4 text-zinc-500">
            {formatTime(new Date(tooltip.time * 1000).toISOString())}
          </div>
          {tooltip.values.map((entry) => (
            <div key={entry.id} className="flex items-center justify-between gap-3">
              <span className="flex min-w-0 items-center gap-1.5">
                <span className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: entry.color }} />
                <span className="truncate text-[10.5px] text-zinc-400">{entry.label}</span>
              </span>
              <span className="font-mono text-[11px] tabular-nums text-zinc-100">
                {formatValue(entry.value)}
              </span>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}


function toAscending(data: SeriesPoint[]): { time: import("lightweight-charts").UTCTimestamp; value: number }[] {
  const seen = new Set<number>();
  return data
    .map((point) => ({
      time: Math.floor(new Date(point.time).getTime() / 1000) as import("lightweight-charts").UTCTimestamp,
      value: point.value,
    }))
    .filter((row) => Number.isFinite(row.time) && Number.isFinite(row.value) && !seen.has(row.time) && seen.add(row.time))
    .sort((a, b) => a.time - b.time);
}
