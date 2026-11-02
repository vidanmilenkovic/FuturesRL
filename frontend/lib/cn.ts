/** Tiny class-name joiner (no dependency on clsx for a 6-line utility). */
export function cn(...values: Array<string | false | null | undefined>): string {
  return values.filter(Boolean).join(" ");
}
