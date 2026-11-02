"use client";

import { useCallback, useEffect, useId, useRef, type ReactNode } from "react";

import { cn } from "@/lib/cn";
import { Button } from "./button";

export interface ConfirmDialogProps {
  open: boolean;
  title: string;
  description: ReactNode;
  confirmLabel: string;
  cancelLabel?: string;
  tone?: "danger" | "primary";
  busy?: boolean;
  /** Extra gate (e.g. a hardware-style arming delay) on top of the phrase. */
  confirmDisabled?: boolean;
  /** When set, the operator must type this exact phrase to arm the action. */
  requirePhrase?: string;
  phrase?: string;
  onPhraseChange?: (value: string) => void;
  onConfirm: () => void;
  onClose: () => void;
  children?: ReactNode;
}

/**
 * Accessible confirmation modal used for irreversible actions (the emergency
 * flatten-all kill switch). Supports an optional typed safety phrase and locks
 * both dismissal and the confirm button while the action is in flight.
 */
export function ConfirmDialog({
  open,
  title,
  description,
  confirmLabel,
  cancelLabel = "Cancel",
  tone = "danger",
  busy = false,
  confirmDisabled = false,
  requirePhrase,
  phrase = "",
  onPhraseChange,
  onConfirm,
  onClose,
  children,
}: ConfirmDialogProps) {
  const titleId = useId();
  const descriptionId = useId();
  const dialogRef = useRef<HTMLDivElement | null>(null);
  const confirmRef = useRef<HTMLButtonElement | null>(null);

  const phraseReady = !requirePhrase || phrase.trim().toUpperCase() === requirePhrase.toUpperCase();
  const dismiss = useCallback(() => {
    if (!busy) onClose();
  }, [busy, onClose]);

  useEffect(() => {
    if (!open) return;
    const previouslyFocused = document.activeElement as HTMLElement | null;
    confirmRef.current?.focus();
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        dismiss();
      }
      if (event.key === "Tab" && dialogRef.current) {
        const focusables = dialogRef.current.querySelectorAll<HTMLElement>(
          'button:not([disabled]), [href], input, select, textarea, [tabindex]:not([tabindex="-1"])',
        );
        if (focusables.length === 0) return;
        const first = focusables[0]!;
        const last = focusables[focusables.length - 1]!;
        if (event.shiftKey && document.activeElement === first) {
          event.preventDefault();
          last.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault();
          first.focus();
        }
      }
    };
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      previouslyFocused?.focus();
    };
  }, [open, dismiss]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      role="presentation"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) dismiss();
      }}
    >
      <div
        aria-hidden
        className="absolute inset-0 bg-terminal-950/85 backdrop-blur-sm"
      />
      <div
        ref={dialogRef}
        role="alertdialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={descriptionId}
        className={cn(
          "relative w-full max-w-lg overflow-hidden rounded-md border bg-terminal-850 shadow-2xl",
          tone === "danger" ? "border-signal-critical/50" : "border-signal-info/40",
        )}
      >
        <div
          className={cn(
            "h-1 w-full",
            tone === "danger"
              ? "bg-gradient-to-r from-signal-critical via-signal-critical/60 to-signal-critical"
              : "bg-gradient-to-r from-signal-info via-signal-info/60 to-signal-info",
          )}
        />
        <div className="px-5 py-4">
          <div className="flex items-start gap-3">
            <span
              aria-hidden
              className={cn(
                "mt-0.5 grid h-8 w-8 shrink-0 place-items-center rounded-full border text-base",
                tone === "danger"
                  ? "border-signal-critical/50 bg-signal-critical/10 text-signal-critical"
                  : "border-signal-info/50 bg-signal-info/10 text-signal-info",
              )}
            >
              {tone === "danger" ? "⚠" : "?"}
            </span>
            <div className="min-w-0">
              <h2
                id={titleId}
                className="font-mono text-sm font-semibold uppercase tracking-[0.14em] text-terminal-100"
              >
                {title}
              </h2>
              <div
                id={descriptionId}
                className="mt-1.5 text-sm leading-relaxed text-terminal-200"
              >
                {description}
              </div>
            </div>
          </div>

          {children ? <div className="mt-4">{children}</div> : null}

          {requirePhrase ? (
            <div className="mt-4">
              <label
                htmlFor={`${titleId}-phrase`}
                className="mb-1 block font-mono text-[10px] uppercase tracking-[0.14em] text-terminal-300"
              >
                Type <span className="text-signal-critical">{requirePhrase}</span> to arm
              </label>
              <input
                id={`${titleId}-phrase`}
                value={phrase}
                onChange={(event) => onPhraseChange?.(event.target.value)}
                autoComplete="off"
                spellCheck={false}
                disabled={busy}
                className="h-9 w-full rounded-sm border border-terminal-500/70 bg-terminal-900 px-3 font-mono text-sm uppercase tracking-[0.2em] text-terminal-100 placeholder:text-terminal-400 focus:border-signal-critical/60 focus:outline-none focus:ring-1 focus:ring-signal-critical/40 disabled:opacity-50"
                placeholder={requirePhrase}
              />
            </div>
          ) : null}

          <div className="mt-5 flex flex-wrap items-center justify-end gap-2">
            <Button variant="ghost" size="md" onClick={dismiss} disabled={busy}>
              {cancelLabel}
            </Button>
            <Button
              ref={confirmRef}
              variant={tone === "danger" ? "danger" : "primary"}
              size="md"
              onClick={onConfirm}
              disabled={busy || !phraseReady || confirmDisabled}
              aria-busy={busy}
            >
              {busy ? (
                <>
                  <span
                    aria-hidden
                    className="h-3 w-3 animate-spin rounded-full border border-current border-t-transparent"
                  />
                  Executing…
                </>
              ) : (
                confirmLabel
              )}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
