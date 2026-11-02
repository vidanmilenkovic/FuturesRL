"use client";

import { useCallback, useEffect, useId, useRef, type ReactNode } from "react";

/**
 * Accessible modal used by the flatten-all confirmation and other destructive
 * flows. Traps Tab focus, closes on Escape and backdrop click, and restores focus
 * to the previously active element when it unmounts.
 */
export function Modal({
  open,
  title,
  description,
  children,
  footer,
  onClose,
  busy = false,
  width = "max-w-md",
}: {
  open: boolean;
  title: string;
  description?: ReactNode;
  children?: ReactNode;
  footer?: ReactNode;
  onClose: () => void;
  busy?: boolean;
  width?: string;
}) {
  const titleId = useId();
  const descriptionId = useId();
  const dialogRef = useRef<HTMLDivElement | null>(null);

  const requestClose = useCallback(() => {
    if (!busy) onClose();
  }, [busy, onClose]);

  useEffect(() => {
    if (!open) return;
    const previous = document.activeElement as HTMLElement | null;
    const dialog = dialogRef.current;
    const focusable = dialog?.querySelectorAll<HTMLElement>(
      'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
    );
    focusable?.[0]?.focus();

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        requestClose();
        return;
      }
      if (event.key !== "Tab" || !focusable || focusable.length === 0) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (!first || !last) return;
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      previous?.focus();
    };
  }, [open, requestClose]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      role="presentation"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) requestClose();
      }}
    >
      <div aria-hidden className="absolute inset-0 bg-black/60" />
      <div
        ref={dialogRef}
        role="alertdialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={description ? descriptionId : undefined}
        className={`relative w-full ${width} overflow-hidden rounded-md border border-edge bg-panel shadow-popover`}
      >
        <header className="flex h-10 items-center justify-between border-b border-hairline px-4">
          <h2 id={titleId} className="text-[13px] font-medium text-zinc-100">
            {title}
          </h2>
          <button
            type="button"
            onClick={requestClose}
            disabled={busy}
            aria-label="Close dialog"
            className="focus-ring -mr-1 grid h-6 w-6 place-items-center rounded text-zinc-500 transition-colors hover:bg-surface hover:text-zinc-200 disabled:opacity-40"
          >
            <svg viewBox="0 0 16 16" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="1.6">
              <path d="M4 4l8 8M12 4l-8 8" strokeLinecap="round" />
            </svg>
          </button>
        </header>
        {description ? (
          <div id={descriptionId} className="px-4 pt-3 text-[12px] leading-5 text-zinc-400">
            {description}
          </div>
        ) : null}
        {children ? <div className="px-4 pb-1 pt-3">{children}</div> : null}
        {footer ? (
          <footer className="mt-3 flex h-12 items-center justify-end gap-2 border-t border-hairline px-4">
            {footer}
          </footer>
        ) : null}
      </div>
    </div>
  );
}
