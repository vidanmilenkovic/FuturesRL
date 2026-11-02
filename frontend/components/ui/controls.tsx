"use client";

import type { InputHTMLAttributes, ReactNode, SelectHTMLAttributes } from "react";

import { cn } from "@/lib/cn";

export function Segmented<T extends string>({
  options,
  value,
  onChange,
  size = "md",
  ariaLabel,
  className,
}: {
  options: { value: T; label: ReactNode; hint?: string }[];
  value: T;
  onChange: (value: T) => void;
  size?: "sm" | "md";
  ariaLabel: string;
  className?: string;
}) {
  return (
    <div
      role="tablist"
      aria-label={ariaLabel}
      className={cn(
        "inline-flex flex-wrap items-center gap-0.5 rounded-sm border border-terminal-600/70 bg-terminal-900/80 p-0.5",
        className,
      )}
    >
      {options.map((option) => {
        const selected = option.value === value;
        return (
          <button
            key={option.value}
            type="button"
            role="tab"
            aria-selected={selected}
            title={option.hint}
            onClick={() => onChange(option.value)}
            className={cn(
              "rounded-[3px] font-mono uppercase tracking-[0.1em] transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-signal-info/50",
              size === "sm" ? "px-2 py-1 text-[10px]" : "px-2.5 py-1.5 text-[11px]",
              selected
                ? "bg-signal-info/15 text-signal-info shadow-[inset_0_0_0_1px_rgba(34,211,238,0.3)]"
                : "text-terminal-300 hover:bg-terminal-700/60 hover:text-terminal-100",
            )}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}

export function Field({
  label,
  hint,
  htmlFor,
  children,
  className,
}: {
  label: ReactNode;
  hint?: ReactNode;
  htmlFor: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("min-w-0", className)}>
      <label
        htmlFor={htmlFor}
        className="mb-1 block font-mono text-[10px] uppercase tracking-[0.14em] text-terminal-300"
      >
        {label}
      </label>
      {children}
      {hint ? <p className="mt-1 text-[10px] text-terminal-400">{hint}</p> : null}
    </div>
  );
}

const controlClass =
  "h-9 w-full rounded-sm border border-terminal-500/70 bg-terminal-900/90 px-2.5 font-mono text-xs text-terminal-100 placeholder:text-terminal-400 transition-colors focus:border-signal-info/60 focus:outline-none focus:ring-1 focus:ring-signal-info/40 disabled:cursor-not-allowed disabled:opacity-50";

export function TextInput({ className, ...rest }: InputHTMLAttributes<HTMLInputElement>) {
  return <input className={cn(controlClass, className)} {...rest} />;
}

export function Select({
  className,
  children,
  ...rest
}: SelectHTMLAttributes<HTMLSelectElement> & { children: ReactNode }) {
  return (
    <select className={cn(controlClass, "appearance-none pr-7", className)} {...rest}>
      {children}
    </select>
  );
}

export function ProgressBar({
  value,
  max = 1,
  tone = "info",
  className,
  label,
}: {
  value: number;
  max?: number;
  tone?: "info" | "up" | "down" | "warn" | "critical" | "violet";
  className?: string;
  label?: string;
}) {
  const pct = Math.min(100, Math.max(0, (value / (max || 1)) * 100));
  const toneClass = {
    info: "bg-signal-info",
    up: "bg-signal-up",
    down: "bg-signal-down",
    warn: "bg-signal-warn",
    critical: "bg-signal-critical",
    violet: "bg-signal-violet",
  }[tone];
  return (
    <div
      className={cn("relative h-1.5 w-full overflow-hidden rounded-full bg-terminal-700", className)}
      role="progressbar"
      aria-valuenow={Math.round(pct)}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-label={label}
    >
      <div className={cn("h-full rounded-full transition-[width] duration-500", toneClass)} style={{ width: `${pct}%` }} />
    </div>
  );
}

export function Skeleton({ className }: { className?: string }) {
  return (
    <div
      aria-hidden
      className={cn("animate-pulse rounded-sm bg-terminal-700/60", className)}
    />
  );
}

export function EmptyState({
  title,
  description,
  action,
}: {
  title: string;
  description?: string;
  action?: ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 py-10 text-center">
      <p className="font-mono text-[11px] uppercase tracking-[0.14em] text-terminal-300">{title}</p>
      {description ? <p className="max-w-sm text-xs text-terminal-400">{description}</p> : null}
      {action}
    </div>
  );
}

export function LiveDot({
  state = "live",
  label,
}: {
  state?: "live" | "demo" | "offline" | "connecting";
  label?: string;
}) {
  const tone =
    state === "live"
      ? "bg-signal-up"
      : state === "demo"
        ? "bg-signal-warn"
        : state === "connecting"
          ? "bg-signal-info"
          : "bg-signal-down";
  return (
    <span className="inline-flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-[0.14em] text-terminal-200">
      <span className="relative flex h-1.5 w-1.5">
        <span className={cn("absolute inline-flex h-full w-full animate-ping rounded-full opacity-60", tone)} />
        <span className={cn("relative inline-flex h-1.5 w-1.5 rounded-full", tone)} />
      </span>
      {label ?? state}
    </span>
  );
}
