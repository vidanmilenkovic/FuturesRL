"use client";

import { useId } from "react";

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function niceTicks(min: number, max: number, count = 4): number[] {
  if (!Number.isFinite(min) || !Number.isFinite(max) || min === max) return [min];
  const span = max - min;
  const step = 10 ** Math.floor(Math.log10(span / count));
  const candidate = [1, 2, 2.5, 5, 10].map((m) => step * m).find((s) => span / s <= count * 1.6) ?? step;
  const start = Math.ceil(min / candidate) * candidate;
  const ticks: number[] = [];
  for (let value = start; value <= max + candidate * 0.001; value += candidate) {
    ticks.push(Number(value.toFixed(10)));
  }
  return ticks;
}

// ---------------------------------------------------------------------------
// Bidirectional weight cell bar (used by positions table + weight matrix)
// ---------------------------------------------------------------------------

export function WeightBar({
  weight,
  maxAbs,
  width = 64,
  marker = true,
}: {
  weight: number;
  maxAbs: number;
  width?: number;
  marker?: boolean;
}) {
  const scale = maxAbs > 0 ? clamp(Math.abs(weight) / maxAbs, 0, 1) : 0;
  const half = width / 2;
  const barWidth = scale * (half - 2);
  return (
    <span
      className="relative inline-block h-3 align-middle"
      style={{ width }}
      aria-hidden="true"
    >
      <span className="absolute inset-y-0 left-1/2 w-px -translate-x-1/2 bg-zinc-700" />
      {marker ? <span className="absolute inset-y-0 left-1/2 w-px -translate-x-1/2" /> : null}
      <span
        className={`absolute top-1/2 h-[5px] -translate-y-1/2 rounded-[2px] ${
          weight >= 0 ? "bg-gain/80" : "bg-loss/80"
        }`}
        style={{
          left: weight >= 0 ? half : half - barWidth,
          width: Math.max(barWidth, weight === 0 ? 0 : 1.5),
        }}
      />
    </span>
  );
}

// ---------------------------------------------------------------------------
// Pipeline: grouped horizontal long / short / net bars per stage
// ---------------------------------------------------------------------------

export interface StageBar {
  key: string;
  label: string;
  formula: string;
  gross: number;
  net: number;
  count: number;
  countLabel: string;
}

export function StageGroupedBars({ stages }: { stages: StageBar[] }) {
  const maxAbs = Math.max(
    ...stages.map((stage) => Math.max(Math.abs((stage.gross + stage.net) / 2), Math.abs((stage.gross - stage.net) / 2))),
    0.0001,
  );
  return (
    <div className="flex flex-col">
      <div className="mb-1 flex items-center gap-3">
        <div className="w-[160px] shrink-0" />
        <div className="flex-1" />
        <div className="flex w-[84px] shrink-0 items-center justify-end gap-3 text-[10px] text-zinc-600">
          <span className="flex items-center gap-1">
            <span className="h-1.5 w-3 rounded-sm bg-gain/80" /> long
          </span>
          <span className="flex items-center gap-1">
            <span className="h-1.5 w-3 rounded-sm bg-loss/80" /> short
          </span>
          <span className="flex items-center gap-1">
            <span className="h-1.5 w-1.5 rounded-full bg-accent" /> net
          </span>
        </div>
      </div>

      {stages.map((stage) => {
        const long = (stage.gross + stage.net) / 2;
        const short = (stage.gross - stage.net) / 2;
        const netLeft = 50 + (stage.net / maxAbs) * 50;
        const longWidth = (long / maxAbs) * 50;
        const shortWidth = (short / maxAbs) * 50;
        return (
          <div key={stage.key} className="flex h-11 items-center gap-3">
            <div className="w-[160px] shrink-0">
              <div className="truncate text-[12.5px] font-medium text-zinc-200">{stage.label}</div>
              <div className="truncate font-mono text-[10px] text-zinc-600">{stage.formula}</div>
            </div>
            <div className="relative h-8 flex-1">
              <div className="absolute inset-y-2 left-1/2 w-px -translate-x-1/2 bg-zinc-700" />
              {long > 0 ? (
                <div
                  className="absolute top-1/2 h-[6px] -translate-y-1/2 rounded-[2px] bg-gain/75"
                  style={{ left: "50%", width: `${longWidth}%` }}
                  title={`long ${long.toFixed(4)}`}
                />
              ) : null}
              {short > 0 ? (
                <div
                  className="absolute top-1/2 h-[6px] -translate-y-1/2 rounded-[2px] bg-loss/70"
                  style={{ left: `${50 - shortWidth}%`, width: `${shortWidth}%` }}
                  title={`short ${short.toFixed(4)}`}
                />
              ) : null}
              <div
                className="absolute top-1/2 h-2.5 w-[3px] -translate-y-1/2 rounded-full bg-accent"
                style={{ left: `${clamp(netLeft, 0.4, 99.6)}%` }}
                title={`net ${stage.net.toFixed(4)}`}
              />
            </div>
            <div className="w-[84px] shrink-0 text-right">
              <div className="font-mono text-[12px] tabular-nums text-zinc-200">
                {stage.gross.toFixed(3)}
              </div>
              <div className="font-mono text-[10px] tabular-nums text-zinc-600">
                {stage.count} {stage.countLabel}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Risk-budget bullet bars
// ---------------------------------------------------------------------------

export interface BulletRow {
  label: string;
  value: number;
  cap: number;
  format: (value: number) => string;
  hint?: string;
  /** Optional explicit tone; otherwise derived from value/cap. */
  tone?: "gain" | "warn" | "loss" | "accent";
}

export function BulletBars({ rows }: { rows: BulletRow[] }) {
  return (
    <div className="space-y-3.5">
      {rows.map((row) => {
        const ratio = row.cap > 0 ? row.value / row.cap : 0;
        const tone = row.tone ?? (ratio >= 1 ? "loss" : ratio >= 0.8 ? "warn" : "gain");
        const fill = tone === "loss" ? "bg-loss" : tone === "warn" ? "bg-warn" : tone === "accent" ? "bg-accent" : "bg-gain";
        return (
          <div key={row.label}>
            <div className="mb-1.5 flex items-baseline justify-between gap-2">
              <span className="text-[12px] text-zinc-400">{row.label}</span>
              <span className="font-mono text-[11.5px] tabular-nums text-zinc-300">
                {row.format(row.value)}
                <span className="text-zinc-600"> / {row.format(row.cap)}</span>
              </span>
            </div>
            <div className="relative h-1.5 w-full overflow-hidden rounded-full bg-zinc-800">
              <div
                className={`h-full rounded-full ${fill}`}
                style={{ width: `${clamp(ratio, 0, 1) * 100}%` }}
              />
            </div>
            {row.hint ? <p className="mt-1 text-[10.5px] text-zinc-600">{row.hint}</p> : null}
          </div>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Asset-class long / short exposure
// ---------------------------------------------------------------------------

export function ExposureBars({
  rows,
  format,
}: {
  rows: { label: string; long: number; short: number }[];
  format: (value: number) => string;
}) {
  const maxAbs = Math.max(...rows.flatMap((row) => [row.long, row.short]), 1);
  return (
    <div className="space-y-2.5">
      {rows.map((row) => (
        <div key={row.label}>
          <div className="mb-1 flex items-baseline justify-between">
            <span className="text-[12px] text-zinc-400">{row.label}</span>
            <span className="font-mono text-[11px] tabular-nums">
              <span className="text-gain">{format(row.long)}</span>
              <span className="text-zinc-700"> / </span>
              <span className="text-loss">{format(row.short)}</span>
            </span>
          </div>
          <div className="relative h-2">
            <div className="absolute inset-y-0 left-1/2 w-px -translate-x-1/2 bg-zinc-700" />
            {row.long > 0 ? (
              <div
                className="absolute inset-y-0 left-1/2 rounded-r-[2px] bg-gain/70"
                style={{ width: `${(row.long / maxAbs) * 50}%` }}
              />
            ) : null}
            {row.short > 0 ? (
              <div
                className="absolute inset-y-0 right-1/2 rounded-l-[2px] bg-loss/70"
                style={{ width: `${(row.short / maxAbs) * 50}%` }}
              />
            ) : null}
          </div>
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// LLM semantic-risk meter (segmented theta_safe / kappa)
// ---------------------------------------------------------------------------

export function RiskMeter({
  value,
  safe,
  critical,
  min = 1,
  max = 5,
}: {
  value: number;
  safe: number;
  critical: number;
  min?: number;
  max?: number;
}) {
  const span = max - min;
  const safePct = ((safe - min) / span) * 100;
  const criticalPct = ((critical - min) / span) * 100;
  const valuePct = ((clamp(value, min, max) - min) / span) * 100;
  const tone = value >= critical ? "bg-loss" : value > safe ? "bg-warn" : "bg-gain";
  return (
    <div>
      <div className="relative h-2 w-full overflow-hidden rounded-full">
        <div className="absolute inset-y-0 left-0 bg-gain/25" style={{ width: `${safePct}%` }} />
        <div
          className="absolute inset-y-0 bg-warn/25"
          style={{ left: `${safePct}%`, width: `${criticalPct - safePct}%` }}
        />
        <div
          className="absolute inset-y-0 right-0 bg-loss/25"
          style={{ left: `${criticalPct}%` }}
        />
      </div>
      <div className="relative mt-1 h-2 w-full">
        <div
          className={`absolute top-0 h-2 w-[3px] -translate-x-1/2 rounded-full ${tone}`}
          style={{ left: `${valuePct}%` }}
        />
        <span
          className="absolute top-1 -translate-x-1/2 font-mono text-[9px] text-zinc-600"
          style={{ left: `${safePct}%` }}
        >
          θ
        </span>
        <span
          className="absolute top-1 -translate-x-1/2 font-mono text-[9px] text-zinc-600"
          style={{ left: `${criticalPct}%` }}
        >
          κ
        </span>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Forward curve (price / normalized / basis)
// ---------------------------------------------------------------------------

export interface CurvePoint {
  label: string;
  price: number;
  days: number;
  isFront: boolean;
}

export function ForwardCurveChart({
  points,
  mode,
  ariaLabel,
}: {
  points: CurvePoint[];
  mode: "price" | "normalized" | "basis";
  ariaLabel: string;
}) {
  const width = 720;
  const height = 260;
  const padding = { top: 18, right: 22, bottom: 30, left: 74 };
  const innerW = width - padding.left - padding.right;
  const innerH = height - padding.top - padding.bottom;
  const frontPrice = points[0]?.price ?? 1;
  const values =
    mode === "normalized"
      ? points.map((point) => (point.price / frontPrice) * 100)
      : mode === "basis"
        ? points.map((point) => ((point.price - frontPrice) / frontPrice) * 10_000)
        : points.map((point) => point.price);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || Math.max(1, Math.abs(max) * 0.005);
  const yMin = mode === "basis" ? Math.min(min, 0) - span * 0.2 : min - span * 0.2;
  const yMax = mode === "basis" ? Math.max(max, 0) + span * 0.2 : max + span * 0.2;
  const x = (index: number) => padding.left + (index / Math.max(points.length - 1, 1)) * innerW;
  const y = (value: number) => padding.top + innerH - ((value - yMin) / (yMax - yMin)) * innerH;
  const ticks = niceTicks(yMin, yMax, 4);
  const gradientId = useId().replace(/[:]/g, "");
  const path = values.map((value, index) => `${index === 0 ? "M" : "L"} ${x(index)} ${y(value)}`).join(" ");
  const area = `${path} L ${x(points.length - 1)} ${padding.top + innerH} L ${x(0)} ${padding.top + innerH} Z`;

  const formatTick = (value: number) =>
    mode === "normalized"
      ? value.toFixed(1)
      : mode === "basis"
        ? `${value.toFixed(1)}bp`
        : value.toLocaleString("en-US", { maximumFractionDigits: 2 });

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="h-auto w-full" role="img" aria-label={ariaLabel}>
      <defs>
        <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#3B82F6" stopOpacity={0.16} />
          <stop offset="100%" stopColor="#3B82F6" stopOpacity={0} />
        </linearGradient>
      </defs>

      {ticks.map((tick) => (
        <g key={tick}>
          <line x1={padding.left} x2={width - padding.right} y1={y(tick)} y2={y(tick)} stroke="#1E1E22" />
          <text x={padding.left - 8} y={y(tick) + 3} textAnchor="end" fill="#71717A" fontSize={9} fontFamily="JetBrains Mono, monospace">
            {formatTick(tick)}
          </text>
        </g>
      ))}
      {mode === "basis" ? (
        <line x1={padding.left} x2={width - padding.right} y1={y(0)} y2={y(0)} stroke="#3F3F46" />
      ) : null}

      {mode === "basis" ? (
        values.map((value, index) => (
          <rect
            key={`basis-${points[index]?.label}`}
            x={x(index) - 10}
            y={Math.min(y(value), y(0))}
            width={20}
            height={Math.max(2, Math.abs(y(0) - y(value)))}
            fill={value >= 0 ? "rgba(244,63,94,0.65)" : "rgba(16,185,129,0.65)"}
            rx={2}
          />
        ))
      ) : (
        <>
          <path d={area} fill={`url(#${gradientId})`} />
          <path d={path} fill="none" stroke="#3B82F6" strokeWidth={1.75} />
        </>
      )}

      {points.map((point, index) => (
        <g key={`${point.label}-${point.days}`}>
          {mode !== "basis" ? (
            <circle
              cx={x(index)}
              cy={y((values[index] ?? 0) ?? 0)}
              r={point.isFront ? 3.5 : 2.5}
              fill={point.isFront ? "#3B82F6" : "#121215"}
              stroke="#3B82F6"
              strokeWidth={1.4}
            />
          ) : null}
          <text
            x={x(index)}
            y={height - 12}
            textAnchor="middle"
            fill={point.isFront ? "#E4E4E7" : "#71717A"}
            fontSize={9}
            fontFamily="JetBrains Mono, monospace"
          >
            {point.label}
          </text>
          {mode !== "basis" ? (
            <text
              x={index === 0 ? x(0) + 4 : x(index)}
              y={y((values[index] ?? 0)) - 9}
              textAnchor={index === 0 ? "start" : "middle"}
              fill="#71717A"
              fontSize={8.5}
              fontFamily="JetBrains Mono, monospace"
            >
              {mode === "normalized" ? (values[index] ?? 0).toFixed(2) : point.price.toFixed(2)}
            </text>
          ) : null}
        </g>
      ))}
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Backtest: spot vs roll attribution grouped bars
// ---------------------------------------------------------------------------

export function AttributionBars({
  rows,
  format,
}: {
  rows: { root: string; spot_pnl: number; roll_pnl: number }[];
  format: (value: number) => string;
}) {
  const maxAbs = Math.max(...rows.flatMap((row) => [Math.abs(row.spot_pnl), Math.abs(row.roll_pnl)]), 1);
  return (
    <div className="space-y-2">
      {rows.map((row) => (
        <div key={row.root} className="flex items-center gap-3">
          <span className="w-10 shrink-0 font-mono text-[11px] text-zinc-300">{row.root}</span>
          <div className="relative h-5 flex-1">
            <div className="absolute inset-y-0 left-1/2 w-px -translate-x-1/2 bg-zinc-700" />
            <div
              className={`absolute top-0.5 h-[6px] rounded-[2px] ${row.spot_pnl >= 0 ? "bg-accent/80" : "bg-zinc-600"}`}
              style={{
                left: row.spot_pnl >= 0 ? "50%" : `${50 - (Math.abs(row.spot_pnl) / maxAbs) * 50}%`,
                width: `${(Math.abs(row.spot_pnl) / maxAbs) * 50}%`,
              }}
              title={`spot ${format(row.spot_pnl)}`}
            />
            <div
              className={`absolute bottom-0.5 h-[6px] rounded-[2px] ${row.roll_pnl >= 0 ? "bg-gain/80" : "bg-loss/80"}`}
              style={{
                left: row.roll_pnl >= 0 ? "50%" : `${50 - (Math.abs(row.roll_pnl) / maxAbs) * 50}%`,
                width: `${(Math.abs(row.roll_pnl) / maxAbs) * 50}%`,
              }}
              title={`roll ${format(row.roll_pnl)}`}
            />
          </div>
          <span className="w-28 shrink-0 text-right font-mono text-[10.5px] tabular-nums">
            <span className="text-zinc-400">{format(row.spot_pnl)}</span>
            <span className="text-zinc-700"> / </span>
            <span className={row.roll_pnl >= 0 ? "text-gain" : "text-loss"}>{format(row.roll_pnl)}</span>
          </span>
        </div>
      ))}
    </div>
  );
}
