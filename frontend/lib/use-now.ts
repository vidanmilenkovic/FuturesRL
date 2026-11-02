"use client";

import { useEffect, useState } from "react";

/**
 * Ticking wall clock for countdowns.
 *
 * Starts as `null` so server render and first client render agree, then updates
 * once mounted. One interval per subscribed component; the tick is coarse (1s)
 * to keep the dashboard cheap during long sessions.
 */
export function useNow(intervalMs = 1_000): Date | null {
  const [now, setNow] = useState<Date | null>(null);

  useEffect(() => {
    setNow(new Date());
    const timer = setInterval(() => setNow(new Date()), intervalMs);
    return () => clearInterval(timer);
  }, [intervalMs]);

  return now;
}
