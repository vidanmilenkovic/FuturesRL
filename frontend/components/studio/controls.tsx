"use client";

import { useId, useMemo, useState, type ReactNode } from "react";

import { cn } from "@/lib/cn";
import { formatCompactUsd } from "@/lib/format";
import type { StudioParameter } from "@/lib/types";

// ---------------------------------------------------------------------------
// Value formatting
// ---------------------------------------------------------------------------

function decimalsFor(step: number | null | undefined): number {
  if (!step || step >= 1) return 0;
  const text = String(step);
  const dot = text.indexOf(".");
  return dot === -1 ? 0 : Math.min(4, text.length - dot - 1);
}

export function formatParameterValue(
  parameter: StudioParameter,
  value: string | number | boolean,
): string {
  if (typeof value === "boolean") return value ? "On" : "Off";
  if (typeof value === "string") return value;
  if (parameter.unit === "%") return `${(value * 100).toFixed(1)}%`;
  if (parameter.unit === "$") return formatCompactUsd(value);
  if (parameter.unit === "×") return `${value.toFixed(2)}×`;
  if (parameter.type === "int") return Math.round(value).toLocaleString("en-US");
  return value.toFixed(decimalsFor(parameter.step) || 2);
}

// ---------------------------------------------------------------------------
// Range + number input
// ---------------------------------------------------------------------------

export function RangeField({
  parameter,
  value,
  onChange,
  disabled = false,
}: {
  parameter: StudioParameter;
  value: number;
  onChange: (value: number) => void;
  disabled?: boolean;
}) {
  const id = useId();
  const min = parameter.min ?? 0;
  const max = parameter.max ?? Math.max(value * 2, 1);
  const step = parameter.step ?? (parameter.type === "int" ? 1 : 0.01);
  return (
    <div className={cn("min-w-0", disabled && "opacity-45")}>
      <div className="mb-1 flex items-baseline justify-between gap-2">
        <label htmlFor={id} className="truncate text-[11px] font-medium text-zinc-400" title={parameter.label}>
          {parameter.label}
        </label>
        <span className="shrink-0 font-mono text-[11px] tabular-nums text-zinc-200">
          {formatParameterValue(parameter, value)}
        </span>
      </div>
      <input
        id={id}
        type="range"
        className="studio-range w-full"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(Number(event.target.value))}
        aria-label={parameter.label}
      />
      <p className="mt-1 line-clamp-2 text-[10px] leading-4 text-zinc-600" title={parameter.description}>
        {parameter.description}
      </p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Root multiselect pills
// ---------------------------------------------------------------------------

export function RootPills({
  roots,
  selected,
  onChange,
  disabled = false,
}: {
  roots: string[];
  selected: string[];
  onChange: (roots: string[]) => void;
  disabled?: boolean;
}) {
  const toggle = (root: string) => {
    if (disabled) return;
    onChange(
      selected.includes(root)
        ? selected.filter((entry) => entry !== root)
        : [...selected, root],
    );
  };
  return (
    <div className="flex flex-wrap gap-1">
      {roots.map((root) => {
        const active = selected.includes(root);
        return (
          <button
            key={root}
            type="button"
            disabled={disabled}
            onClick={() => toggle(root)}
            aria-pressed={active}
            className={cn(
              "focus-ring h-6 rounded-md border px-2 font-mono text-[11px] font-medium transition-colors",
              active
                ? "border-accent/40 bg-accent-soft text-blue-300"
                : "border-edge bg-canvas text-zinc-500 hover:border-zinc-600 hover:text-zinc-300",
              disabled && "cursor-not-allowed opacity-40",
            )}
          >
            {root}
          </button>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Toggle / checkbox row
// ---------------------------------------------------------------------------

export function ToggleRow({
  label,
  description,
  checked,
  onChange,
  disabled = false,
}: {
  label: string;
  description?: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
}) {
  const id = useId();
  return (
    <label
      htmlFor={id}
      className={cn(
        "flex cursor-pointer items-start gap-2.5 rounded-md border border-edge bg-canvas/60 px-2.5 py-2 transition-colors hover:border-zinc-600",
        disabled && "cursor-not-allowed opacity-50",
      )}
    >
      <input
        id={id}
        type="checkbox"
        checked={checked}
        disabled={disabled}
        onChange={(event) => onChange(event.target.checked)}
        className="mt-0.5 h-3.5 w-3.5 shrink-0 rounded border border-zinc-600 bg-canvas accent-blue-500"
      />
      <span className="min-w-0">
        <span className="block text-[12px] font-medium text-zinc-300">{label}</span>
        {description ? <span className="mt-0.5 block text-[10.5px] leading-4 text-zinc-600">{description}</span> : null}
      </span>
    </label>
  );
}

// ---------------------------------------------------------------------------
// Compact labelled select/number input shell
// ---------------------------------------------------------------------------

export function InlineField({
  label,
  children,
  className,
}: {
  label: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("min-w-0", className)}>
      <div className="mb-1 text-[10.5px] font-medium uppercase tracking-[0.05em] text-zinc-500">{label}</div>
      {children}
    </div>
  );
}

// ---------------------------------------------------------------------------
// JSON inspector
// ---------------------------------------------------------------------------

export function JsonPre({ value, className }: { value: string; className?: string }) {
  return (
    <pre
      className={cn(
        "scroll-area min-h-0 overflow-auto rounded-md border border-hairline bg-canvas p-2 font-mono text-[10px] leading-[1.55] text-zinc-500",
        className,
      )}
    >
      {value}
    </pre>
  );
}

// ---------------------------------------------------------------------------
// Parameter importance bars
// ---------------------------------------------------------------------------

export function ImportanceBars({
  rows,
  maxRows = 8,
}: {
  rows: { label: string; value: number; percent: number }[];
  maxRows?: number;
}) {
  const visible = rows.slice(0, maxRows);
  if (visible.length === 0) {
    return <p className="text-[11px] text-zinc-600">No parameter importances reported for this study.</p>;
  }
  return (
    <div className="space-y-1.5">
      {visible.map((row) => (
        <div key={row.label} className="grid grid-cols-[minmax(0,1fr)_44px] items-center gap-2">
          <div className="min-w-0">
            <div className="mb-0.5 truncate font-mono text-[10.5px] text-zinc-400" title={row.label}>
              {row.label}
            </div>
            <div className="h-1.5 overflow-hidden rounded-full bg-surface">
              <div
                className="h-full rounded-full bg-accent/80"
                style={{ width: `${Math.max(2, row.percent)}%` }}
              />
            </div>
          </div>
          <span className="text-right font-mono text-[10.5px] tabular-nums text-zinc-500">
            {row.value.toFixed(3)}
          </span>
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Ordinal line chart (trial history / reward curve — x-axis is an index, not time)
// ---------------------------------------------------------------------------

export interface OrdinalSeries {
  label: string;
  color: string;
  values: number[];
}

export function OrdinalLineChart({
  series,
  height = 200,
  formatValue = (value: number) => value.toFixed(2),
  xLabels,
  yTicks = 4,
  ariaLabel,
  className,
}: {
  series: OrdinalSeries[];
  height?: number;
  formatValue?: (value: number) => string;
  /** Optional labels for first / middle / last x positions. */
  xLabels?: [string, string, string];
  yTicks?: number;
  ariaLabel: string;
  className?: string;
}) {
  const [hover, setHover] = useState<{ index: number; x: number; y: number } | null>(null);
  const width = 640;
  const padLeft = 50;
  const padRight = 10;
  const padTop = 10;
  const padBottom = 20;
  const plotW = width - padLeft - padRight;
  const plotH = height - padTop - padBottom;
  const length = Math.max(
    ...series.map((entry) => entry.values.length),
    2,
  );
  const allValues = series.flatMap((entry) => entry.values);
  const rawMin = Math.min(...allValues);
  const rawMax = Math.max(...allValues);
  const span = rawMax - rawMin || 1;
  const min = rawMin - span * 0.08;
  const max = rawMax + span * 0.08;
  const plotMin = Number.isFinite(min) ? min : 0;
  const plotMax = Number.isFinite(max) ? max : 1;
  const xFor = (index: number) => padLeft + (index / Math.max(1, length - 1)) * plotW;
  const yFor = (value: number) =>
    padTop + (1 - (value - plotMin) / Math.max(1e-9, plotMax - plotMin)) * plotH;

  const ticks = useMemo(
    () =>
      Array.from({ length: yTicks + 1 }, (_, index) => {
        const ratio = index / yTicks;
        return {
          value: plotMax - (plotMax - plotMin) * ratio,
          y: yFor(plotMax - (plotMax - plotMin) * ratio),
        };
      }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [plotMin, plotMax, height],
  );
  const xTickLabels = xLabels ?? ["0", String(Math.max(1, Math.round(length / 2))), String(length - 1)];

  return (
    <div className={cn("relative", className)} data-testid="ordinal-chart" data-series-count={series.length}>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="none"
        className="h-full w-full"
        role="img"
        aria-label={ariaLabel}
        style={{ height }}
        onMouseLeave={() => setHover(null)}
        onMouseMove={(event) => {
          const rect = event.currentTarget.getBoundingClientRect();
          const ratio = rect.width > 0 ? (event.clientX - rect.left) / rect.width : 0;
          const index = Math.round(ratio * (length - 1));
          setHover({
            index: Math.max(0, Math.min(length - 1, index)),
            x: xFor(index),
            y: padTop,
          });
        }}
      >
        {ticks.map((tick) => (
          <g key={tick.y}>
            <line x1={padLeft} x2={width - padRight} y1={tick.y} y2={tick.y} stroke="#1E1E22" strokeWidth={1} />
            <text x={padLeft - 6} y={tick.y + 3} textAnchor="end" fill="#71717A" fontSize={9}>
              {formatValue(tick.value)}
            </text>
          </g>
        ))}
        {[0, Math.round((length - 1) / 2), Math.max(0, length - 1)].map((index, position) => (
          <text
            key={`x-${position}`}
            x={xFor(index)}
            y={height - 6}
            textAnchor={position === 0 ? "start" : position === 2 ? "end" : "middle"}
            fill="#52525B"
            fontSize={9}
          >
            {xTickLabels[position]}
          </text>
        ))}
        {series.map((entry) => {
          const points = entry.values
            .map((value, index) => `${xFor(index).toFixed(2)},${yFor(value).toFixed(2)}`)
            .join(" ");
          return (
            <g key={entry.label}>
              <polyline
                points={points}
                fill="none"
                stroke={entry.color}
                strokeWidth={1.8}
                strokeLinejoin="round"
                strokeLinecap="round"
                vectorEffect="non-scaling-stroke"
              />
              {entry.values.length <= 60
                ? entry.values.map((value, index) => (
                    <circle
                      key={`${entry.label}-${index}`}
                      cx={xFor(index)}
                      cy={yFor(value)}
                      r={2}
                      fill="#09090B"
                      stroke={entry.color}
                      strokeWidth={1.2}
                    />
                  ))
                : null}
            </g>
          );
        })}
        {hover ? (
          <line
            x1={hover.x}
            x2={hover.x}
            y1={padTop}
            y2={padTop + plotH}
            stroke="#3F3F46"
            strokeWidth={1}
            strokeDasharray="3 3"
          />
        ) : null}
        {hover
          ? series.map((entry) => {
              const value = entry.values[hover.index];
              if (value === undefined) return null;
              return (
                <circle
                  key={`hover-${entry.label}`}
                  cx={hover.x}
                  cy={yFor(value)}
                  r={3}
                  fill={entry.color}
                  stroke="#09090B"
                  strokeWidth={1.5}
                />
              );
            })
          : null}
      </svg>
      {hover ? (
        <div className="pointer-events-none absolute left-12 top-1 rounded border border-edge bg-surface px-2 py-1 text-[10px] shadow-panel">
          <div className="mb-0.5 text-zinc-500">#{hover.index + 1}</div>
          {series.map((entry) => {
            const value = entry.values[hover.index];
            if (value === undefined) return null;
            return (
              <div key={entry.label} className="flex items-center justify-between gap-3">
                <span className="flex items-center gap-1.5 text-zinc-400">
                  <span className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: entry.color }} />
                  {entry.label}
                </span>
                <span className="font-mono tabular-nums text-zinc-100">{formatValue(value)}</span>
              </div>
            );
          })}
        </div>
      ) : null}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Small labelled stat used by the optimizer / DRL headers
// ---------------------------------------------------------------------------

export function MiniStat({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="min-w-0">
      <div className="truncate text-[10px] font-medium uppercase tracking-[0.05em] text-zinc-500">{label}</div>
      <div className="mt-0.5 truncate font-mono text-[12px] tabular-nums text-zinc-200">{value}</div>
    </div>
  );
}
