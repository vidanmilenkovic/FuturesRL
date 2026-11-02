import type { ReactNode } from "react";

import { cn } from "@/lib/cn";

export function Panel({
  title,
  subtitle,
  actions,
  children,
  className,
  contentClassName,
  id,
}: {
  title?: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  contentClassName?: string;
  id?: string;
}) {
  return (
    <section
      id={id}
      className={cn(
        "relative flex flex-col overflow-hidden rounded-md border border-terminal-600/70 bg-terminal-850/80 shadow-panel",
        className,
      )}
    >
      <span
        aria-hidden
        className="pointer-events-none absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-signal-info/25 to-transparent"
      />
      {title ? (
        <header className="flex flex-wrap items-start justify-between gap-3 border-b border-terminal-600/60 px-3 py-2.5">
          <div className="min-w-0">
            <h2 className="truncate font-mono text-[11px] font-semibold uppercase tracking-[0.16em] text-terminal-100">
              {title}
            </h2>
            {subtitle ? (
              <p className="mt-0.5 truncate font-mono text-[10px] uppercase tracking-[0.08em] text-terminal-300">
                {subtitle}
              </p>
            ) : null}
          </div>
          {actions ? <div className="flex shrink-0 items-center gap-2">{actions}</div> : null}
        </header>
      ) : null}
      <div className={cn("flex-1 px-3 py-3", contentClassName)}>{children}</div>
    </section>
  );
}

export function PanelRow({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div
      className={cn(
        "flex items-center justify-between gap-3 border-b border-terminal-600/40 px-1 py-2 last:border-0",
        className,
      )}
    >
      {children}
    </div>
  );
}

export function KeyValue({
  label,
  value,
  tone,
  mono = true,
}: {
  label: ReactNode;
  value: ReactNode;
  tone?: string;
  mono?: boolean;
}) {
  return (
    <div className="flex items-baseline justify-between gap-3 py-1">
      <dt className="text-[11px] uppercase tracking-[0.08em] text-terminal-300">{label}</dt>
      <dd className={cn("text-right text-sm", mono ? "font-mono tabular-nums" : "", tone)}>
        {value}
      </dd>
    </div>
  );
}
