import { forwardRef, type ButtonHTMLAttributes, type ReactNode } from "react";

import { cn } from "@/lib/cn";

type Variant = "primary" | "secondary" | "ghost" | "danger" | "outline" | "success";
type Size = "sm" | "md" | "lg";

const variants: Record<Variant, string> = {
  primary:
    "border border-signal-info/40 bg-signal-info/15 text-signal-info hover:bg-signal-info/25 hover:border-signal-info/70 focus-visible:ring-signal-info/60",
  secondary:
    "border border-terminal-500/70 bg-terminal-700/60 text-terminal-100 hover:bg-terminal-600 hover:border-terminal-400 focus-visible:ring-terminal-300/50",
  ghost:
    "border border-transparent text-terminal-200 hover:bg-terminal-700/70 hover:text-terminal-100 focus-visible:ring-terminal-300/40",
  outline:
    "border border-terminal-500/70 bg-transparent text-terminal-100 hover:border-signal-info/50 hover:text-signal-info focus-visible:ring-signal-info/40",
  danger:
    "border border-signal-critical/50 bg-signal-critical/15 text-signal-critical hover:bg-signal-critical/25 hover:border-signal-critical/80 focus-visible:ring-signal-critical/60",
  success:
    "border border-signal-up/40 bg-signal-up/15 text-signal-up hover:bg-signal-up/25 hover:border-signal-up/70 focus-visible:ring-signal-up/60",
};

const sizes: Record<Size, string> = {
  sm: "h-7 px-2.5 text-[11px]",
  md: "h-9 px-3.5 text-xs",
  lg: "h-11 px-5 text-sm",
};

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  leading?: ReactNode;
  trailing?: ReactNode;
  block?: boolean;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  {
    variant = "secondary",
    size = "md",
    leading,
    trailing,
    block = false,
    className,
    children,
    type = "button",
    ...rest
  },
  ref,
) {
  return (
    <button
      ref={ref}
      type={type}
      className={cn(
        "inline-flex select-none items-center justify-center gap-2 rounded-sm font-mono font-medium uppercase tracking-[0.1em] transition-colors duration-150",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-offset-terminal-950",
        "disabled:cursor-not-allowed disabled:opacity-45",
        variants[variant],
        sizes[size],
        block ? "w-full" : "",
        className,
      )}
      {...rest}
    >
      {leading}
      {children}
      {trailing}
    </button>
  );
});
