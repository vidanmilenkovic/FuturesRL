"use client";

import { useEffect, useRef } from "react";

import type { SeriesPoint } from "@/lib/types";

export interface LightweightChartProps {
  data: SeriesPoint[];
  /** Visual series kind; `area` is used for equity/PnL curves. */
  kind?: "area" | "line" | "histogram" | "baseline";
  height?: number;
  color?: string;
  secondaryColor?: string;
  /** Optional second line (e.g. benchmark) drawn on the same price scale. */
  comparison?: SeriesPoint[];
  comparisonColor?: string;
  /** Hide the time axis for sparkline-style mini charts. */
  hideTimeAxis?: boolean;
  /** Hide the price axis labels. */
  hidePriceAxis?: boolean;
  /** Percentage formatting for the price scale. */
  percent?: boolean;
  /** Baseline value for `kind: "baseline"` (e.g. 0 for PnL). */
  baselineValue?: number;
  className?: string;
  /** Accessible description of the chart. */
  ariaLabel: string;
}

/**
 * TradingView Lightweight Charts wrapper.
 *
 * Imported dynamically inside the effect so the library never executes during
 * SSR/prerender (it needs a DOM canvas) and so the initial page bundle stays
 * small — the equity curve is the only heavy client-side dependency.
 */
export function LightweightChart({
  data,
  kind = "area",
  height = 280,
  color = "#22d3ee",
  secondaryColor = "rgba(34,211,238,0.12)",
  comparison,
  comparisonColor = "#a78bfa",
  hideTimeAxis = false,
  hidePriceAxis = false,
  percent = false,
  baselineValue,
  className,
  ariaLabel,
}: LightweightChartProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<import("lightweight-charts").IChartApi | null>(null);
  const seriesRef = useRef<import("lightweight-charts").ISeriesApi<
    "Area" | "Line" | "Histogram" | "Baseline"
  > | null>(null);
  const comparisonRef = useRef<import("lightweight-charts").ISeriesApi<"Line"> | null>(null);

  // Create the chart once per mount; subsequent data updates only setData.
  useEffect(() => {
    let disposed = false;
    let resizeObserver: ResizeObserver | null = null;

    void (async () => {
      const lw = await import("lightweight-charts");
      const container = containerRef.current;
      if (!container || disposed) return;
      const chart = lw.createChart(container, {
        height,
        autoSize: true,
        layout: {
          background: { color: "transparent" },
          textColor: "#61798f",
          fontSize: 11,
          fontFamily:
            "ui-monospace, SFMono-Regular, Menlo, Monaco, Cascadia Mono, Consolas, monospace",
          attributionLogo: false,
        },
        grid: {
          vertLines: { color: "rgba(148,163,184,0.06)" },
          horzLines: { color: "rgba(148,163,184,0.06)" },
        },
        rightPriceScale: {
          visible: !hidePriceAxis,
          borderColor: "rgba(148,163,184,0.12)",
          scaleMargins: { top: 0.12, bottom: 0.12 },
        },
        timeScale: {
          visible: !hideTimeAxis,
          borderColor: "rgba(148,163,184,0.12)",
          timeVisible: true,
          secondsVisible: false,
          rightOffset: 4,
        },
        crosshair: {
          mode: lw.CrosshairMode.Normal,
          vertLine: { color: "rgba(34,211,238,0.35)", width: 1, labelBackgroundColor: "#0d1520" },
          horzLine: { color: "rgba(34,211,238,0.35)", width: 1, labelBackgroundColor: "#0d1520" },
        },
        localization: {
          priceFormatter: (price: number) =>
            percent ? `${(price * 100).toFixed(1)}%` : price.toLocaleString("en-US", { maximumFractionDigits: 2 }),
        },
        handleScale: { axisPressedMouseMove: { time: true, price: false } },
      });
      chartRef.current = chart;

      if (kind === "histogram") {
        const series = chart.addSeries(lw.HistogramSeries, {
          color,
          priceFormat: { type: "volume" },
          priceLineVisible: false,
          lastValueVisible: false,
        });
        seriesRef.current = series;
      } else if (kind === "line") {
        const series = chart.addSeries(lw.LineSeries, {
          color,
          lineWidth: 2,
          priceLineVisible: false,
          lastValueVisible: true,
        });
        seriesRef.current = series;
      } else if (kind === "baseline") {
        const series = chart.addSeries(lw.BaselineSeries, {
          baseValue: { type: "price", price: baselineValue ?? 0 },
          topLineColor: "#34d399",
          topFillColor1: "rgba(52,211,153,0.28)",
          topFillColor2: "rgba(52,211,153,0.02)",
          bottomLineColor: "#fb7185",
          bottomFillColor1: "rgba(251,113,133,0.02)",
          bottomFillColor2: "rgba(251,113,133,0.28)",
          priceLineVisible: false,
          lastValueVisible: true,
        });
        seriesRef.current = series;
      } else {
        const series = chart.addSeries(lw.AreaSeries, {
          lineColor: color,
          topColor: secondaryColor,
          bottomColor: "rgba(4,7,11,0)",
          lineWidth: 2,
          priceLineVisible: false,
          lastValueVisible: true,
        });
        seriesRef.current = series;
      }

      if (comparison && comparison.length > 0) {
        const line = chart.addSeries(lw.LineSeries, {
          color: comparisonColor,
          lineWidth: 1,
          lineStyle: lw.LineStyle.Dashed,
          priceLineVisible: false,
          lastValueVisible: false,
        });
        comparisonRef.current = line;
      }

      resizeObserver = new ResizeObserver(() => {
        // `autoSize` covers width; height stays fixed by options.
        chart.applyOptions({ height });
      });
      resizeObserver.observe(container);
      if (disposed) {
        chart.remove();
      }
    })();

    return () => {
      disposed = true;
      resizeObserver?.disconnect();
      seriesRef.current = null;
      comparisonRef.current = null;
      chartRef.current?.remove();
      chartRef.current = null;
    };
    // The chart is created once; data/labels update through the effects below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [kind, percent, hideTimeAxis, hidePriceAxis, baselineValue]);

  useEffect(() => {
    const series = seriesRef.current;
    if (!series || data.length === 0) return;
    series.setData(
      data.map((point) => ({
        time: (new Date(point.time).getTime() / 1000) as import("lightweight-charts").UTCTimestamp,
        value: point.value,
      })),
    );
    chartRef.current?.timeScale().fitContent();
  }, [data]);

  useEffect(() => {
    const series = comparisonRef.current;
    if (!series || !comparison || comparison.length === 0) return;
    series.setData(
      comparison.map((point) => ({
        time: (new Date(point.time).getTime() / 1000) as import("lightweight-charts").UTCTimestamp,
        value: point.value,
      })),
    );
  }, [comparison]);

  return (
    <div
      ref={containerRef}
      role="img"
      aria-label={ariaLabel}
      className={className}
      style={{ height }}
    />
  );
}
