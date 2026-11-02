"use client";

import { useId, useMemo } from "react";

import { cn } from "@/lib/cn";

// ---------------------------------------------------------------------------
// Sparkline
// ---------------------------------------------------------------------------

export function Sparkline({
  values,
  width = 120,
  height = 28,
  stroke,
  className,
  ariaLabel,
}: {
  values: number[];
  width?: number;
  height?: number;
  stroke?: string;
  className?: string;
  ariaLabel?: string;
}) {
  const { line, area } = useMemo(() => {
    if (values.length < 2) return { line: "", area: "" };
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || 1;
    const step = width / (values.length - 1);
    const points = values.map((value, index) => {
      const x = index * step;
      const y = height - ((value - min) / range) * (height - 2) - 1;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    });
    return {
      line: `M ${points.join(" L ")}`,
      area: `M 0,${height} L ${points.join(" L ")} L ${width},${height} Z`,
    };
  }, [values, width, height]);

  const positive = values.length > 1 ? values[values.length - 1]! >= values[0]! : true;
  const color = stroke ?? (positive ? "#34d399" : "#fb7185");
  const gradientId = useId().replace(/[:]/g, "");

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      width={width}
      height={height}
      className={cn("overflow-visible", className)}
      role="img"
      aria-label={ariaLabel ?? "sparkline"}
    >
      <defs>
        <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity={0.32} />
          <stop offset="100%" stopColor={color} stopOpacity={0} />
        </linearGradient>
      </defs>
      {area ? <path d={area} fill={`url(#${gradientId})`} /> : null}
      {line ? <path d={line} fill="none" stroke={color} strokeWidth={1.5} strokeLinejoin="round" /> : null}
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Generic helpers
// ---------------------------------------------------------------------------

function niceTicks(min: number, max: number, count = 4): number[] {
  if (!Number.isFinite(min) || !Number.isFinite(max) || min === max) return [min];
  const span = max - min;
  const step = Math.pow(10, Math.floor(Math.log10(span / count)));
  const candidates = [1, 2, 2.5, 5, 10].map((multiplier) => step * multiplier);
  const chosen = candidates.find((candidate) => span / candidate <= count * 1.6) ?? step;
  const start = Math.ceil(min / chosen) * chosen;
  const ticks: number[] = [];
  for (let value = start; value <= max + chosen * 0.001; value += chosen) {
    ticks.push(Number(value.toFixed(10)));
  }
  return ticks;
}

// ---------------------------------------------------------------------------
// Pipeline waterfall (stage gross exposure + step deltas)
// ---------------------------------------------------------------------------

export interface WaterfallStage {
  key: string;
  label: string;
  gross: number;
  net: number;
  note?: string;
}

const STAGE_COLORS = ["#64748b", "#22d3ee", "#a78bfa", "#34d399", "#fbbf24"];

export function WaterfallChart({
  stages,
  className,
}: {
  stages: WaterfallStage[];
  className?: string;
}) {
  const width = 760;
  const rowHeight = 44;
  const labelWidth = 168;
  const valueWidth = 84;
  const chartWidth = width - labelWidth - valueWidth;
  const height = stages.length * rowHeight + 18;
  const maxGross = Math.max(...stages.map((stage) => stage.gross), 0.0001);

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className={cn("h-auto w-full", className)}
      role="img"
      aria-label="FinRL-X pipeline waterfall: gross exposure by stage"
    >
      {[0, 0.25, 0.5, 0.75, 1].map((fraction) => (
        <line
          key={fraction}
          x1={labelWidth + chartWidth * fraction}
          x2={labelWidth + chartWidth * fraction}
          y1={4}
          y2={height - 14}
          stroke="rgba(148,163,184,0.12)"
          strokeDasharray="2 4"
        />
      ))}

      {stages.map((stage, index) => {
        const y = index * rowHeight + 14;
        const barWidth = Math.max(2, (stage.gross / maxGross) * chartWidth);
        const previous = index > 0 ? stages[index - 1]! : null;
        const delta = previous ? stage.gross - previous.gross : null;
        const color = STAGE_COLORS[index % STAGE_COLORS.length]!;
        return (
          <g key={stage.key}>
            <text
              x={0}
              y={y + 15}
              className="fill-terminal-200 font-mono"
              style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: "0.06em" }}
            >
              {stage.label}
            </text>
            <text
              x={0}
              y={y + 28}
              className="fill-terminal-500 font-mono"
              style={{ fontSize: 9, letterSpacing: "0.08em" }}
            >
              {stage.note ?? ""}
            </text>

            <rect
              x={labelWidth}
              y={y + 3}
              width={chartWidth}
              height={20}
              fill="rgba(148,163,184,0.06)"
              rx={2}
            />
            <rect
              x={labelWidth}
              y={y + 3}
              width={barWidth}
              height={20}
              rx={2}
              fill={color}
              fillOpacity={0.24}
              stroke={color}
              strokeOpacity={0.7}
            />
            {/* Signed net exposure marker relative to gross bar */}
            <line
              x1={labelWidth}
              x2={labelWidth + chartWidth}
              y1={y + 23}
              y2={y + 23}
              stroke="rgba(148,163,184,0.14)"
            />
            <circle
              cx={labelWidth + Math.min(1, Math.abs(stage.net) / maxGross) * chartWidth}
              cy={y + 13}
              r={3}
              fill={stage.net >= 0 ? "#34d399" : "#fb7185"}
            />

            <text
              x={width - 4}
              y={y + 15}
              textAnchor="end"
              className="fill-terminal-100 font-mono tabular-nums"
              style={{ fontSize: 11 }}
            >
              {stage.gross.toFixed(3)}×
            </text>
            <text
              x={width - 4}
              y={y + 28}
              textAnchor="end"
              className={cn("font-mono tabular-nums", delta !== null && delta < 0 ? "fill-signal-down" : "fill-signal-up")}
              style={{ fontSize: 9 }}
            >
              {delta === null ? "seed" : `${delta >= 0 ? "+" : ""}${delta.toFixed(3)}×`}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Term structure curve
// ---------------------------------------------------------------------------

export interface CurvePoint {
  label: string;
  price: number;
  days: number;
  isFront?: boolean;
}

export function CurveChart({
  points,
  regime,
  className,
  ariaLabel,
}: {
  points: CurvePoint[];
  regime: "contango" | "backwardation" | "flat";
  className?: string;
  ariaLabel: string;
}) {
  const width = 640;
  const height = 260;
  const padding = { top: 24, right: 22, bottom: 34, left: 58 };
  const innerW = width - padding.left - padding.right;
  const innerH = height - padding.top - padding.bottom;
  const prices = points.map((point) => point.price);
  const min = Math.min(...prices);
  const max = Math.max(...prices);
  const span = max - min || Math.max(1, max * 0.005);
  const yMin = min - span * 0.25;
  const yMax = max + span * 0.25;
  const x = (index: number) => padding.left + (index / Math.max(points.length - 1, 1)) * innerW;
  const y = (price: number) => padding.top + innerH - ((price - yMin) / (yMax - yMin)) * innerH;
  const ticks = niceTicks(yMin, yMax, 4);
  const tone = regime === "contango" ? "#fb7185" : regime === "backwardation" ? "#34d399" : "#22d3ee";
  const gradientId = useId().replace(/[:]/g, "");

  const path = points.map((point, index) => `${index === 0 ? "M" : "L"} ${x(index)} ${y(point.price)}`).join(" ");
  const area = `${path} L ${x(points.length - 1)} ${padding.top + innerH} L ${x(0)} ${padding.top + innerH} Z`;

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className={cn("h-auto w-full", className)}
      role="img"
      aria-label={ariaLabel}
    >
      <defs>
        <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={tone} stopOpacity={0.24} />
          <stop offset="100%" stopColor={tone} stopOpacity={0} />
        </linearGradient>
      </defs>

      {ticks.map((tick) => (
        <g key={tick}>
          <line
            x1={padding.left}
            x2={width - padding.right}
            y1={y(tick)}
            y2={y(tick)}
            stroke="rgba(148,163,184,0.1)"
          />
          <text
            x={padding.left - 8}
            y={y(tick) + 3}
            textAnchor="end"
            className="fill-terminal-400 font-mono"
            style={{ fontSize: 9 }}
          >
            {tick.toLocaleString("en-US", { maximumFractionDigits: 2 })}
          </text>
        </g>
      ))}

      <path d={area} fill={`url(#${gradientId})`} />
      <path d={path} fill="none" stroke={tone} strokeWidth={1.8} />

      {points.map((point, index) => (
        <g key={point.label}>
          <circle
            cx={x(index)}
            cy={y(point.price)}
            r={point.isFront ? 4.5 : 3}
            fill={point.isFront ? tone : "#0d1520"}
            stroke={tone}
            strokeWidth={1.5}
          />
          <text
            x={x(index)}
            y={height - 16}
            textAnchor="middle"
            className={point.isFront ? "fill-terminal-100 font-mono" : "fill-terminal-400 font-mono"}
            style={{ fontSize: 9, letterSpacing: "0.06em" }}
          >
            {point.label}
          </text>
          <text
            x={x(index)}
            y={y(point.price) - 9}
            textAnchor="middle"
            className="fill-terminal-300 font-mono"
            style={{ fontSize: 8 }}
          >
            {point.price.toLocaleString("en-US", { maximumFractionDigits: 2 })}
          </text>
        </g>
      ))}

      <text
        x={width - padding.right}
        y={padding.top - 6}
        textAnchor="end"
        className="font-mono"
        style={{ fontSize: 9, fill: tone, textTransform: "uppercase", letterSpacing: "0.14em" }}
      >
        {regime}
      </text>
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Horizontal / stacked attribution bars
// ---------------------------------------------------------------------------

export function AttributionBars({
  rows,
  className,
  ariaLabel,
}: {
  rows: { root: string; spot_pnl: number; roll_pnl: number }[];
  className?: string;
  ariaLabel: string;
}) {
  const width = 640;
  const rowHeight = 34;
  const labelWidth = 52;
  const height = rows.length * rowHeight + 8;
  const maxAbs = Math.max(...rows.flatMap((row) => [Math.abs(row.spot_pnl), Math.abs(row.roll_pnl)]), 1);
  const zeroX = labelWidth + (width - labelWidth - 80) / 2;
  const barScale = (width - labelWidth - 80) / 2 / maxAbs;

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className={cn("h-auto w-full", className)} role="img" aria-label={ariaLabel}>
      <line x1={zeroX} x2={zeroX} y1={0} y2={height} stroke="rgba(148,163,184,0.25)" strokeDasharray="3 3" />
      {rows.map((row, index) => {
        const y = index * rowHeight + 8;
        const spotWidth = row.spot_pnl * barScale;
        const rollWidth = row.roll_pnl * barScale;
        return (
          <g key={row.root}>
            <text
              x={0}
              y={y + 14}
              className="fill-terminal-200 font-mono"
              style={{ fontSize: 11 }}
            >
              {row.root}
            </text>
            <rect
              x={Math.min(zeroX, zeroX + spotWidth)}
              y={y + 2}
              width={Math.abs(spotWidth)}
              height={9}
              fill={row.spot_pnl >= 0 ? "#22d3ee" : "#64748b"}
              rx={1}
            />
            <rect
              x={Math.min(zeroX, zeroX + rollWidth)}
              y={y + 13}
              width={Math.abs(rollWidth)}
              height={9}
              fill={row.roll_pnl >= 0 ? "#34d399" : "#fb7185"}
              rx={1}
            />
            <text
              x={width - 2}
              y={y + 14}
              textAnchor="end"
              className="fill-terminal-300 font-mono tabular-nums"
              style={{ fontSize: 9 }}
            >
              {`${row.spot_pnl >= 0 ? "+" : "−"}${Math.abs(Math.round(row.spot_pnl / 1000))}k / ${row.roll_pnl >= 0 ? "+" : "−"}${Math.abs(Math.round(row.roll_pnl / 1000))}k`}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Tail-risk scatter (Rachev ratio input)
// ---------------------------------------------------------------------------

export function TailScatter({
  points,
  className,
  ariaLabel,
}: {
  points: { rank: number; return_pct: number; tail: "gain" | "loss" }[];
  className?: string;
  ariaLabel: string;
}) {
  const width = 520;
  const height = 240;
  const padding = { top: 18, right: 16, bottom: 26, left: 48 };
  const innerW = width - padding.left - padding.right;
  const innerH = height - padding.top - padding.bottom;
  const values = points.map((point) => point.return_pct);
  const maxAbs = Math.max(...values.map((value) => Math.abs(value)), 0.001) * 1.15;
  const y = (value: number) => padding.top + innerH / 2 - (value / maxAbs) * (innerH / 2);
  const maxRank = Math.max(...points.map((point) => point.rank), 1);
  const x = (rank: number) => padding.left + ((rank - 0.5) / maxRank) * innerW;

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className={cn("h-auto w-full", className)} role="img" aria-label={ariaLabel}>
      <line x1={padding.left} x2={width - padding.right} y1={y(0)} y2={y(0)} stroke="rgba(148,163,184,0.3)" />
      {[maxAbs, maxAbs / 2, -maxAbs / 2, -maxAbs].map((tick) => (
        <g key={tick}>
          <line
            x1={padding.left}
            x2={width - padding.right}
            y1={y(tick)}
            y2={y(tick)}
            stroke="rgba(148,163,184,0.08)"
          />
          <text
            x={padding.left - 6}
            y={y(tick) + 3}
            textAnchor="end"
            className="fill-terminal-400 font-mono"
            style={{ fontSize: 8 }}
          >
            {(tick * 100).toFixed(1)}%
          </text>
        </g>
      ))}
      {points.map((point) => (
        <circle
          key={`${point.tail}-${point.rank}`}
          cx={x(point.rank)}
          cy={y(point.return_pct)}
          r={point.tail === "gain" ? 2.6 : 2.2}
          fill={point.tail === "gain" ? "#34d399" : "#fb7185"}
          fillOpacity={point.tail === "gain" ? 0.85 : 0.75}
        />
      ))}
      <text
        x={width - padding.right}
        y={padding.top + 4}
        textAnchor="end"
        className="fill-signal-up font-mono"
        style={{ fontSize: 8, letterSpacing: "0.1em" }}
      >
        GAIN TAIL α=5%
      </text>
      <text
        x={width - padding.right}
        y={height - 10}
        textAnchor="end"
        className="fill-signal-down font-mono"
        style={{ fontSize: 8, letterSpacing: "0.1em" }}
      >
        LOSS TAIL α=5%
      </text>
      <text x={padding.left} y={height - 10} className="fill-terminal-500 font-mono" style={{ fontSize: 8 }}>
        rank
      </text>
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Underwater drawdown chart
// ---------------------------------------------------------------------------

export function UnderwaterChart({
  points,
  className,
  ariaLabel,
}: {
  points: { time: string; value: number }[];
  className?: string;
  ariaLabel: string;
}) {
  const width = 760;
  const height = 200;
  const padding = { top: 12, right: 12, bottom: 20, left: 46 };
  const innerW = width - padding.left - padding.right;
  const innerH = height - padding.top - padding.bottom;
  const values = points.map((point) => point.value);
  const min = Math.min(...values, -0.01);
  const x = (index: number) => padding.left + (index / Math.max(points.length - 1, 1)) * innerW;
  const y = (value: number) => padding.top + (value / (min || -1)) * innerH;
  const path = points.map((point, index) => `${index === 0 ? "M" : "L"} ${x(index)} ${y(point.value)}`).join(" ");
  const area = `${path} L ${x(points.length - 1)} ${padding.top} L ${padding.left} ${padding.top} Z`;

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className={cn("h-auto w-full", className)} role="img" aria-label={ariaLabel}>
      {[0, 0.25, 0.5, 0.75, 1].map((fraction) => {
        const percent = min * fraction;
        return (
          <g key={fraction}>
            <line
              x1={padding.left}
              x2={width - padding.right}
              y1={y(percent)}
              y2={y(percent)}
              stroke="rgba(148,163,184,0.08)"
            />
            <text
              x={padding.left - 6}
              y={y(percent) + 3}
              textAnchor="end"
              className="fill-terminal-400 font-mono"
              style={{ fontSize: 8 }}
            >
              {(percent * 100).toFixed(1)}%
            </text>
          </g>
        );
      })}
      <path d={area} fill="rgba(251,113,133,0.16)" />
      <path d={path} fill="none" stroke="#fb7185" strokeWidth={1.6} />
    </svg>
  );
}
