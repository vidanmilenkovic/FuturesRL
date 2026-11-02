"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import type { ApiResult } from "./types";

interface AsyncState<T> {
  data: T | null;
  source: "live" | "demo" | "loading";
  error: string | null;
  loading: boolean;
}

/**
 * Load an `ApiResult<T>` once on mount (and on explicit `reload`).
 *
 * The initial state is `loading`, never prefilled: data is only computed in an
 * effect so server prerender and client hydration render identical skeletons
 * and the demo fixtures (which depend on `Date.now()`) cannot mismatch.
 */
export function useAsyncData<T>(loader: () => Promise<ApiResult<T>>): AsyncState<T> & {
  reload: () => void;
} {
  const [state, setState] = useState<AsyncState<T>>({
    data: null,
    source: "loading",
    error: null,
    loading: true,
  });
  const loaderRef = useRef(loader);
  loaderRef.current = loader;
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setState((previous) => ({ ...previous, loading: true }));
    loaderRef
      .current()
      .then((result) => {
        if (cancelled) return;
        setState({
          data: result.data,
          source: result.source,
          error: result.error,
          loading: false,
        });
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        setState({
          data: null,
          source: "demo",
          error: error instanceof Error ? error.message : String(error),
          loading: false,
        });
      });
    return () => {
      cancelled = true;
    };
  }, [nonce]);

  const reload = useCallback(() => setNonce((value) => value + 1), []);

  return { ...state, reload };
}
