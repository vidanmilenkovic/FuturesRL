"use client";

import Link from "next/link";
import type { ReactNode } from "react";

/** Numeric text styling shared by every table, KPI and chart label. */
export const num = "font-mono tabular-nums";

export function signedColor(value: number | null | undefined): string {
  if (value === null || value === undefined || Math.abs(value) < 1e-9) return "text-zinc-400";
  return value > 0 ? "text-gain" : "text-loss";
}

// ---------------------------------------------------------------------------
// Panel
// ---------------------------------------------------------------------------

export function Panel({
  title,
  actions,
  children,
  className,
  bodyClassName,
  flush = false,
}: {
  title?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
  /** `flush` removes body padding (tables/charts fill the panel edge-to-edge). */
  flush?: boolean;
}) {
  return (
    <section className={`panel flex min-h-0 flex-col ${className ?? ""}`}>
      {title !== undefined ? (
        <header className="panel-header">
          <h2 className="panel-title truncate">{title}</h2>
          {actions ? <div className="flex shrink-0 items-center gap-2">{actions}</div> : null}
        </header>
      ) : null}
      <div className={`flex min-h-0 flex-1 flex-col ${flush ? "" : "p-3"} ${bodyClassName ?? ""}`}>
        {children}
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Buttons / tabs / segmented controls
// ---------------------------------------------------------------------------

type ButtonVariant = "primary" | "secondary" | "ghost" | "danger" | "outline";
type ButtonSize = "xs" | "sm" | "md";

const buttonVariants: Record<ButtonVariant, string> = {
  primary: "border border-accent bg-accent text-white hover:bg-accent-hover",
  secondary:
    "border border-edge bg-surface text-zinc-200 hover:border-zinc-600 hover:bg-zinc-800",
  ghost: "border border-transparent text-zinc-400 hover:bg-surface hover:text-zinc-200",
  danger: "border border-rose-600 bg-rose-600 text-white hover:bg-rose-500",
  outline: "border border-edge bg-transparent text-zinc-300 hover:bg-surface",
};

const buttonSizes: Record<ButtonSize, string> = {
  xs: "h-6 px-2 text-[11px]",
  sm: "h-7 px-2.5 text-[12px]",
  md: "h-8 px-3 text-[12px]",
};

export function Button({
  children,
  variant = "secondary",
  size = "sm",
  onClick,
  disabled,
  type = "button",
  className,
  title,
  ariaLabel,
}: {
  children: ReactNode;
  variant?: ButtonVariant;
  size?: ButtonSize;
  onClick?: () => void;
  disabled?: boolean;
  type?: "button" | "submit";
  className?: string;
  title?: string;
  ariaLabel?: string;
}) {
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      title={title}
      aria-label={ariaLabel}
      className={`focus-ring inline-flex select-none items-center justify-center gap-1.5 rounded-md font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${buttonVariants[variant]} ${buttonSizes[size]} ${className ?? ""}`}
    >
      {children}
    </button>
  );
}

export function Segmented<T extends string>({
  options,
  value,
  onChange,
  ariaLabel,
  className,
}: {
  options: { value: T; label: string; title?: string }[];
  value: T;
  onChange: (value: T) => void;
  ariaLabel: string;
  className?: string;
}) {
  return (
    <div
      role="tablist"
      aria-label={ariaLabel}
      className={`inline-flex items-center gap-0.5 rounded-md border border-edge bg-canvas p-0.5 ${className ?? ""}`}
    >
      {options.map((option) => {
        const selected = option.value === value;
        return (
          <button
            key={option.value}
            type="button"
            role="tab"
            aria-selected={selected}
            title={option.title}
            onClick={() => onChange(option.value)}
            className={`focus-ring h-6 rounded-[4px] px-2 text-[11px] font-medium transition-colors ${
              selected
                ? "bg-zinc-700/70 text-zinc-100"
                : "text-zinc-500 hover:text-zinc-300"
            }`}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}

export function Tabs<T extends string>({
  options,
  value,
  onChange,
  ariaLabel,
  className,
}: {
  options: { value: T; label: string; count?: number }[];
  value: T;
  onChange: (value: T) => void;
  ariaLabel: string;
  className?: string;
}) {
  return (
    <div
      role="tablist"
      aria-label={ariaLabel}
      className={`flex items-center gap-5 border-b border-hairline ${className ?? ""}`}
    >
      {options.map((option) => {
        const selected = option.value === value;
        return (
          <button
            key={option.value}
            type="button"
            role="tab"
            aria-selected={selected}
            onClick={() => onChange(option.value)}
            className={`focus-ring relative -mb-px flex h-9 items-center gap-1.5 text-[12.5px] font-medium transition-colors ${
              selected ? "text-zinc-100" : "text-zinc-500 hover:text-zinc-300"
            }`}
          >
            {option.label}
            {option.count !== undefined ? (
              <span className="font-mono text-[11px] text-zinc-600">{option.count}</span>
            ) : null}
            {selected ? (
              <span className="absolute inset-x-0 bottom-0 h-[2px] rounded-full bg-accent" />
            ) : null}
          </button>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Badges / status
// ---------------------------------------------------------------------------

export type Tone = "accent" | "gain" | "loss" | "warn" | "neutral";

const badgeTones: Record<Tone, string> = {
  accent: "bg-accent-soft text-blue-300",
  gain: "bg-gain-soft text-emerald-300",
  loss: "bg-loss-soft text-rose-300",
  warn: "bg-warn-soft text-amber-300",
  neutral: "bg-zinc-500/10 text-zinc-400",
};

export function Badge({
  children,
  tone = "neutral",
  className,
  title,
}: {
  children: ReactNode;
  tone?: Tone;
  className?: string;
  title?: string;
}) {
  return (
    <span
      title={title}
      className={`inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] font-medium leading-4 ${badgeTones[tone]} ${className ?? ""}`}
    >
      {children}
    </span>
  );
}

const dotTones: Record<Tone, string> = {
  accent: "bg-accent",
  gain: "bg-gain",
  loss: "bg-loss",
  warn: "bg-warn",
  neutral: "bg-zinc-500",
};

export function StatusDot({ tone = "neutral", pulse = false }: { tone?: Tone; pulse?: boolean }) {
  return (
    <span className="relative inline-flex h-1.5 w-1.5">
      {pulse ? (
        <span className={`absolute inline-flex h-full w-full animate-ping rounded-full opacity-50 ${dotTones[tone]}`} />
      ) : null}
      <span className={`relative inline-flex h-1.5 w-1.5 rounded-full ${dotTones[tone]}`} />
    </span>
  );
}

export function ModePill({
  label,
  tone = "neutral",
  title,
}: {
  label: string;
  tone?: Tone;
  title?: string;
}) {
  return (
    <span
      title={title}
      className={`inline-flex h-6 items-center gap-1.5 rounded-md border px-2 text-[11px] font-medium ${
        tone === "warn"
          ? "border-warn/30 bg-warn-soft text-amber-300"
          : tone === "gain"
            ? "border-gain/30 bg-gain-soft text-emerald-300"
            : tone === "accent"
              ? "border-accent/30 bg-accent-soft text-blue-300"
              : "border-edge bg-surface text-zinc-400"
      }`}
    >
      <StatusDot tone={tone} />
      {label}
    </span>
  );
}

// ---------------------------------------------------------------------------
// KPI
// ---------------------------------------------------------------------------

export function KpiCard({
  label,
  value,
  hint,
  valueClassName,
  className,
  footer,
  hintClassName,
}: {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
  valueClassName?: string;
  className?: string;
  footer?: ReactNode;
  hintClassName?: string;
}) {
  return (
    <div
      className={`flex h-20 flex-col justify-between rounded-md border border-edge bg-panel px-3 py-2.5 shadow-panel ${className ?? ""}`}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="truncate text-[11px] font-medium uppercase tracking-[0.04em] text-zinc-500">
          {label}
        </span>
        {footer}
      </div>
      <div className={`font-mono text-[22px] font-semibold leading-none tabular-nums ${valueClassName ?? "text-zinc-50"}`}>
        {value}
      </div>
      {hint ? (
        <div className={`min-h-[16px] text-[11px] leading-4 text-zinc-500 ${hintClassName ?? ""}`}>{hint}</div>
      ) : null}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Progress / bullets / empty states
// ---------------------------------------------------------------------------

export function EmptyState({
  title,
  description,
  action,
  className,
}: {
  title: string;
  description?: string;
  action?: ReactNode;
  className?: string;
}) {
  return (
    <div className={`flex flex-col items-center justify-center gap-1 px-4 py-10 text-center ${className ?? ""}`}>
      <p className="text-[12.5px] font-medium text-zinc-400">{title}</p>
      {description ? <p className="max-w-md text-[12px] leading-5 text-zinc-600">{description}</p> : null}
      {action ? <div className="mt-2">{action}</div> : null}
    </div>
  );
}

export function Skeleton({ className }: { className?: string }) {
  return <div aria-hidden className={`animate-pulse rounded-md bg-surface ${className ?? ""}`} />;
}

// ---------------------------------------------------------------------------
// Form controls
// ---------------------------------------------------------------------------

export function Field({
  label,
  htmlFor,
  hint,
  children,
  className,
}: {
  label: string;
  htmlFor: string;
  hint?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={`min-w-0 ${className ?? ""}`}>
      <label htmlFor={htmlFor} className="mb-1 block text-[11px] font-medium text-zinc-500">
        {label}
      </label>
      {children}
      {hint ? <p className="mt-1 text-[11px] leading-4 text-zinc-600">{hint}</p> : null}
    </div>
  );
}

const inputClass =
  "focus-ring h-7 w-full rounded-md border border-edge bg-canvas px-2 text-[12px] text-zinc-200 placeholder:text-zinc-600 transition-colors focus:border-accent/70 disabled:opacity-50";

export function TextInput({
  value,
  onChange,
  id,
  type = "text",
  step,
  min,
  max,
  placeholder,
  disabled,
  className,
}: {
  value: string | number;
  onChange: (value: string) => void;
  id?: string;
  type?: "text" | "number" | "date";
  step?: number;
  min?: number;
  max?: number;
  placeholder?: string;
  disabled?: boolean;
  className?: string;
}) {
  return (
    <input
      id={id}
      type={type}
      value={value}
      step={step}
      min={min}
      max={max}
      placeholder={placeholder}
      disabled={disabled}
      onChange={(event) => onChange(event.target.value)}
      className={`${inputClass} ${className ?? ""}`}
    />
  );
}

export function Select({
  value,
  onChange,
  id,
  options,
  disabled,
  className,
}: {
  value: string;
  onChange: (value: string) => void;
  id?: string;
  options: { value: string; label: string }[];
  disabled?: boolean;
  className?: string;
}) {
  return (
    <select
      id={id}
      value={value}
      disabled={disabled}
      onChange={(event) => onChange(event.target.value)}
      className={`${inputClass} appearance-none pr-6 ${className ?? ""}`}
    >
      {options.map((option) => (
        <option key={option.value} value={option.value} className="bg-panel">
          {option.label}
        </option>
      ))}
    </select>
  );
}

export function Checkbox({
  checked,
  onChange,
  label,
  id,
  disabled,
}: {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label: ReactNode;
  id?: string;
  disabled?: boolean;
}) {
  return (
    <label htmlFor={id} className="flex cursor-pointer items-start gap-2 text-[12px] text-zinc-400">
      <input
        id={id}
        type="checkbox"
        checked={checked}
        disabled={disabled}
        onChange={(event) => onChange(event.target.checked)}
        className="mt-0.5 h-3.5 w-3.5 shrink-0 rounded border border-zinc-600 bg-canvas accent-blue-500"
      />
      <span className="leading-5">{label}</span>
    </label>
  );
}

// ---------------------------------------------------------------------------
// Navigation link (used by shell)
// ---------------------------------------------------------------------------

export function NavTab({
  href,
  label,
  active,
}: {
  href: string;
  label: string;
  active: boolean;
}) {
  return (
    <Link
      href={href}
      aria-current={active ? "page" : undefined}
      className={`focus-ring relative flex h-12 items-center px-2 text-[13px] font-medium transition-colors ${
        active ? "text-zinc-100" : "text-zinc-500 hover:text-zinc-300"
      }`}
    >
      {label}
      {active ? <span className="absolute inset-x-2 bottom-0 h-[2px] rounded-full bg-accent" /> : null}
    </Link>
  );
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

export function SortHeader({
  label,
  active,
  direction,
  onClick,
  align = "left",
}: {
  label: string;
  active: boolean;
  direction: "asc" | "desc";
  onClick: () => void;
  align?: "left" | "right";
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`focus-ring flex w-full items-center gap-1 text-[11px] font-semibold uppercase tracking-[0.04em] transition-colors ${
        active ? "text-zinc-300" : "text-zinc-500 hover:text-zinc-300"
      } ${align === "right" ? "justify-end" : "justify-start"}`}
    >
      {label}
      <span className={`text-[9px] ${active ? "text-zinc-400" : "text-transparent"}`}>
        {direction === "asc" ? "▲" : "▼"}
      </span>
    </button>
  );
}
