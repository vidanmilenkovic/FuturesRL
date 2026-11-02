import type { ReactNode } from "react";

import { cn } from "@/lib/cn";

export type Tone = "neutral" | "up" | "down" | "warn" | "critical" | "info" | "violet";

export const toneText: Record<Tone, string> = {
  neutral: "text-terminal-100",
  up: "text-signal-up",
  down: "text-signal-down",
  warn: "text-signal-warn",
  critical: "text-signal-critical",
  info: "text-signal-info",
  violet: "text-signal-violet",
};

export const toneBg: Record<Tone, string> = {
  neutral: "bg-terminal-600/40",
  up: "bg-signal-up/10",
  down: "bg-signal-down/10",
  warn: "bg-signal-warn/10",
  critical: "bg-signal-critical/10",
  info: "bg-signal-info/10",
  violet: "bg-signal-violet/10",
};

export const toneBorder: Record<Tone, string> = {
  neutral: "border-terminal-500/60",
  up: "border-signal-up/30",
  down: "border-signal-down/30",
  warn: "border-signal-warn/40",
  critical: "border-signal-critical/50",
  info: "border-signal-info/40",
  violet: "border-signal-violet/40",
};

export const toneDot: Record<Tone, string> = {
  neutral: "bg-terminal-200",
  up: "bg-signal-up",
  down: "bg-signal-down",
  warn: "bg-signal-warn",
  critical: "bg-signal-critical",
  info: "bg-signal-info",
  violet: "bg-signal-violet",
};

export type BadgeTone = Tone;

export function Badge({
  children,
  tone = "neutral",
  dot = false,
  className,
  title,
}: {
  children: ReactNode;
  tone?: BadgeTone;
  dot?: boolean;
  className?: string;
  title?: string;
}) {
  return (
    <span
      title={title}
      className={cn(
        "inline-flex items-center gap-1.5 rounded-sm border px-1.5 py-0.5 font-mono text-[10px] font-medium uppercase tracking-[0.12em]",
        toneBg[tone],
        toneBorder[tone],
        toneText[tone],
        className,
      )}
    >
      {dot ? <span className={cn("h-1.5 w-1.5 rounded-full", toneDot[tone])} /> : null}
      {children}
    </span>
  );
}
