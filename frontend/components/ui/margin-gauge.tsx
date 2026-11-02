"use client";

import { cn } from "@/lib/cn";
import { marginTone } from "@/lib/format";

export interface MarginGaugeProps {
  utilization: number;
  warning?: number;
  critical?: number;
  size?: number;
  label?: string;
  className?: string;
}

const TONE_COLOR = {
  normal: "#34d399",
  warning: "#fbbf24",
  critical: "#f43f5e",
} as const;

/**
 * Radial margin-utilisation gauge.
 *
 * Zones are hard-coded to the platform risk contract: < 60% nominal, 60–80%
 * warning (initial margin cushion eroded), >= 80% critical (daily kill-switch
 * territory). Uses a 240° sweep so the needle reads like a physical instrument.
 */
export function MarginGauge({
  utilization,
  warning = 0.6,
  critical = 0.8,
  size = 208,
  label = "Initial margin utilisation",
  className,
}: MarginGaugeProps) {
  const tone = marginTone(utilization, warning, critical);
  const color = TONE_COLOR[tone];
  const clamped = Math.min(Math.max(utilization, 0), 1);
  const startAngle = 150;
  const sweep = 240;
  const radius = size / 2 - 18;
  const center = size / 2;
  const valueAngle = startAngle + sweep * clamped;

  const polar = (angleDeg: number, distance: number) => {
    const radians = (angleDeg * Math.PI) / 180;
    return {
      x: center + Math.cos(radians) * distance,
      y: center + Math.sin(radians) * distance,
    };
  };

  const arc = (from: number, to: number, distance: number) => {
    const start = polar(from, distance);
    const end = polar(to, distance);
    const largeArc = to - from > 180 ? 1 : 0;
    return `M ${start.x} ${start.y} A ${distance} ${distance} 0 ${largeArc} 1 ${end.x} ${end.y}`;
  };

  const needle = polar(valueAngle, radius - 6);
  const needleBase = polar(valueAngle, 14);
  const warningAngle = startAngle + sweep * warning;
  const criticalAngle = startAngle + sweep * critical;

  return (
    <div className={cn("flex flex-col items-center", className)}>
      <svg
        width={size}
        height={size * 0.82}
        viewBox={`0 0 ${size} ${size * 0.82}`}
        role="img"
        aria-label={`${label}: ${(utilization * 100).toFixed(1)} percent (${tone})`}
      >
        <defs>
          <linearGradient id="gauge-track" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="rgba(148,163,184,0.18)" />
            <stop offset="100%" stopColor="rgba(148,163,184,0.08)" />
          </linearGradient>
        </defs>

        {/* Track */}
        <path
          d={arc(startAngle, startAngle + sweep, radius)}
          fill="none"
          stroke="url(#gauge-track)"
          strokeWidth={16}
          strokeLinecap="round"
        />
        {/* Warning + critical zones */}
        <path
          d={arc(warningAngle, criticalAngle, radius)}
          fill="none"
          stroke="rgba(251,191,36,0.32)"
          strokeWidth={16}
          strokeLinecap="butt"
        />
        <path
          d={arc(criticalAngle, startAngle + sweep, radius)}
          fill="none"
          stroke="rgba(244,63,94,0.45)"
          strokeWidth={16}
          strokeLinecap="round"
        />
        {/* Value */}
        <path
          d={arc(startAngle, Math.max(startAngle + 0.4, valueAngle), radius)}
          fill="none"
          stroke={color}
          strokeWidth={16}
          strokeLinecap="round"
          style={{ filter: `drop-shadow(0 0 6px ${color}66)` }}
        />

        {/* Tick marks */}
        {[0, 0.2, 0.4, 0.6, 0.8, 1].map((tick) => {
          const angle = startAngle + sweep * tick;
          const outer = polar(angle, radius + 13);
          const inner = polar(angle, radius + 7);
          return (
            <line
              key={tick}
              x1={inner.x}
              y1={inner.y}
              x2={outer.x}
              y2={outer.y}
              stroke="rgba(148,163,184,0.5)"
              strokeWidth={1}
            />
          );
        })}

        {/* Needle */}
        <line
          x1={needleBase.x}
          y1={needleBase.y}
          x2={needle.x}
          y2={needle.y}
          stroke={color}
          strokeWidth={2.5}
          strokeLinecap="round"
        />
        <circle cx={center} cy={center} r={4} fill={color} />

        {/* Numeric readout */}
        <text
          x={center}
          y={center + radius * 0.62}
          textAnchor="middle"
          className="fill-terminal-100 font-mono"
          style={{ fontSize: size * 0.15, fontWeight: 600, letterSpacing: "-0.02em" }}
        >
          {(utilization * 100).toFixed(1)}%
        </text>
        <text
          x={center}
          y={center + radius * 0.86}
          textAnchor="middle"
          className="fill-terminal-300 font-mono"
          style={{ fontSize: size * 0.045, textTransform: "uppercase", letterSpacing: "0.18em" }}
        >
          {tone === "critical" ? "BREACH" : tone === "warning" ? "WARNING" : "NOMINAL"}
        </text>
      </svg>
      <div className="-mt-1 flex items-center gap-3 font-mono text-[10px] uppercase tracking-[0.12em] text-terminal-300">
        <span className="flex items-center gap-1">
          <span className="h-1.5 w-1.5 rounded-full bg-signal-up" /> &lt;{(warning * 100).toFixed(0)}%
        </span>
        <span className="flex items-center gap-1">
          <span className="h-1.5 w-1.5 rounded-full bg-signal-warn" /> {(warning * 100).toFixed(0)}–
          {(critical * 100).toFixed(0)}%
        </span>
        <span className="flex items-center gap-1">
          <span className="h-1.5 w-1.5 rounded-full bg-signal-critical" /> &gt;
          {(critical * 100).toFixed(0)}%
        </span>
      </div>
    </div>
  );
}
