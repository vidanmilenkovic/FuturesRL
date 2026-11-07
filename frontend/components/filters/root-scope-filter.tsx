"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import { Badge, Button, Panel, Select, TextInput } from "@/components/ui/primitives";
import { getInstruments } from "@/lib/api";
import { cn } from "@/lib/cn";
import type { InstrumentRoot, RootScope, RootScopeQuery, RootScopeState } from "@/lib/types";
import { useAsyncData } from "@/lib/use-async-data";

const STORAGE_KEY = "futuresrl:root-scope-filter";

const SCOPE_OPTIONS: { value: RootScope; label: string; hint: string }[] = [
  { value: "strategy", label: "Active Strategy", hint: "Roots used by the active strategy (default)" },
  { value: "cached", label: "Cached Parquet Roots", hint: "Only instrument-master roots with cached Parquet bars" },
  { value: "all", label: "All 45 CME Roots", hint: "Every instrument-master root, cached or deterministic" },
];

function normalizeScope(value: string | null): RootScope {
  return value === "all" || value === "cached" ? value : "strategy";
}

function normalizeRoots(value: string | null): string[] {
  if (!value) return [];
  return value
    .split(",")
    .map((root) => root.trim().toUpperCase())
    .filter(Boolean);
}

/** Shared scope/sector/root selection with URL + localStorage persistence. */
export function useRootScopeFilter() {
  const [state, setState] = useState<RootScopeState>({
    scope: "strategy",
    sector: "",
    roots: [],
  });
  const [hydrated, setHydrated] = useState(false);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const params = new URLSearchParams(window.location.search);
    const stored = window.localStorage.getItem(STORAGE_KEY);
    let persisted: Partial<RootScopeState> = {};
    if (stored) {
      try {
        const parsed = JSON.parse(stored) as Partial<RootScopeState>;
        persisted = parsed;
      } catch {
        persisted = {};
      }
    }
    setState({
      scope: normalizeScope(params.get("scope") ?? persisted.scope ?? null),
      sector: params.get("sector") ?? persisted.sector ?? "",
      roots: normalizeRoots(params.get("roots")) || persisted.roots || [],
    });
    setHydrated(true);
  }, []);

  useEffect(() => {
    if (!hydrated || typeof window === "undefined") return;
    const params = new URLSearchParams(window.location.search);
    if (state.scope !== "strategy") params.set("scope", state.scope);
    else params.delete("scope");
    if (state.sector) params.set("sector", state.sector);
    else params.delete("sector");
    if (state.roots.length > 0) params.set("roots", state.roots.join(","));
    else params.delete("roots");
    const query = params.toString();
    window.history.replaceState(null, "", `${window.location.pathname}${query ? `?${query}` : ""}`);
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  }, [hydrated, state]);

  const setScope = useCallback((scope: RootScope) => {
    setState((previous) => ({ ...previous, scope, roots: [] }));
  }, []);
  const setSector = useCallback((sector: string) => {
    setState((previous) => ({ ...previous, sector, roots: [] }));
  }, []);
  const toggleRoot = useCallback((root: string) => {
    setState((previous) => ({
      ...previous,
      roots: previous.roots.includes(root)
        ? previous.roots.filter((entry) => entry !== root)
        : [...previous.roots, root],
    }));
  }, []);
  const setRoots = useCallback((roots: string[]) => {
    setState((previous) => ({ ...previous, roots: [...new Set(roots.map((root) => root.toUpperCase()))] }));
  }, []);
  const clearRoots = useCallback(() => setState((previous) => ({ ...previous, roots: [] })), []);

  const query = useMemo<RootScopeQuery>(
    () => ({
      scope: state.scope,
      sector: state.sector || undefined,
      roots: state.roots,
    }),
    [state],
  );
  const key = useMemo(
    () => `${state.scope}|${state.sector}|${state.roots.join(",")}`,
    [state],
  );

  return { state, query, key, hydrated, setScope, setSector, toggleRoot, setRoots, clearRoots };
}

export function RootScopeFilterBar({
  scope,
  sector,
  roots,
  onScopeChange,
  onSectorChange,
  onToggleRoot,
  onClearRoots,
  onRefresh,
  refreshing,
  selectedCount,
  dataOrigin,
}: {
  scope: RootScope;
  sector: string;
  roots: string[];
  onScopeChange: (scope: RootScope) => void;
  onSectorChange: (sector: string) => void;
  onToggleRoot: (root: string) => void;
  onClearRoots: () => void;
  onRefresh?: () => void;
  refreshing?: boolean;
  selectedCount?: number | null;
  dataOrigin?: string | null;
}) {
  const instruments = useAsyncData(getInstruments);
  const [rootOpen, setRootOpen] = useState(false);
  const [query, setQuery] = useState("");

  const sectors = useMemo(() => instruments.data?.sectors ?? [], [instruments.data]);
  const allRoots = useMemo<InstrumentRoot[]>(
    () => instruments.data?.roots ?? sectors.flatMap((entry) => entry.roots),
    [instruments.data, sectors],
  );
  const visibleRoots = useMemo(() => {
    const q = query.trim().toLowerCase();
    const matches = q
      ? allRoots.filter(
          (root) =>
            root.root.toLowerCase().includes(q) ||
            root.name.toLowerCase().includes(q),
        )
      : allRoots;
    return (sector ? matches.filter((root) => root.sector === sector) : matches).slice(0, 45);
  }, [allRoots, query, sector]);

  return (
    <Panel className="col-span-12" flush>
      <div className="relative flex flex-wrap items-center gap-2 px-2.5 py-2" data-testid="root-scope-filter">
        <span className="text-[11.5px] font-medium text-zinc-200">Market scope</span>
        <div className="flex items-center gap-1" role="tablist" aria-label="Root scope">
          {SCOPE_OPTIONS.map((option) => (
            <button
              key={option.value}
              type="button"
              role="tab"
              aria-selected={scope === option.value}
              data-testid={`scope-${option.value}`}
              title={option.hint}
              onClick={() => onScopeChange(option.value)}
              className={cn(
                "focus-ring h-6 rounded border px-1.5 text-[10.5px] font-medium transition-colors",
                scope === option.value
                  ? "border-accent/40 bg-accent-soft text-blue-300"
                  : "border-transparent text-zinc-500 hover:bg-surface hover:text-zinc-300",
              )}
            >
              {option.label}
            </button>
          ))}
        </div>
        <Select
          id="root-filter-sector"
          value={sector}
          onChange={onSectorChange}
          options={[
            { value: "", label: "All sectors" },
            ...sectors.map((entry) => ({ value: entry.id, label: `${entry.label} (${entry.count})` })),
          ]}
          className="!h-6 !w-[148px] text-[10.5px]"
        />
        <Button
          variant="ghost"
          size="xs"
          testId="root-filter-roots"
          onClick={() => setRootOpen((open) => !open)}
          aria-expanded={rootOpen}
          title="Select specific roots"
        >
          {roots.length > 0 ? `${roots.length} roots` : "Specific roots"}
        </Button>
        {roots.length > 0 ? (
          <div className="flex max-w-[420px] flex-wrap items-center gap-1" data-testid="root-filter-selected">
            {roots.slice(0, 8).map((root) => (
              <button
                key={root}
                type="button"
                onClick={() => onToggleRoot(root)}
                title={`Remove ${root}`}
                className="rounded border border-accent/30 bg-accent-soft/40 px-1 font-mono text-[9.5px] text-blue-300"
              >
                {root} ×
              </button>
            ))}
            {roots.length > 8 ? <span className="text-[9.5px] text-zinc-600">+{roots.length - 8}</span> : null}
            <button
              type="button"
              onClick={onClearRoots}
              className="rounded px-1 text-[9.5px] text-zinc-500 hover:bg-surface hover:text-zinc-300"
            >
              clear
            </button>
          </div>
        ) : null}
        <span className="ml-auto flex items-center gap-2 text-[9.5px] text-zinc-600">
          {selectedCount !== null && selectedCount !== undefined ? (
            <Badge tone="neutral">{selectedCount} symbols</Badge>
          ) : null}
          {dataOrigin ? (
            <span data-testid="root-filter-origin" className="font-mono">
              {dataOrigin}
            </span>
          ) : null}
          {onRefresh ? (
            <Button
              variant="ghost"
              size="xs"
              testId="root-filter-refresh"
              onClick={onRefresh}
              disabled={refreshing}
              title="Refresh scoped data"
            >
              {refreshing ? "Refreshing…" : "Refresh"}
            </Button>
          ) : null}
        </span>
        {rootOpen ? (
          <div className="absolute left-2.5 top-[36px] z-30 w-[360px] rounded-md border border-edge bg-panel p-2 shadow-popover">
            <TextInput
              id="root-filter-search"
              value={query}
              onChange={setQuery}
              placeholder="Search roots / names"
              className="mb-1.5 h-6 text-[11px]"
            />
            <div className="max-h-52 overflow-y-auto">
              {visibleRoots.length === 0 ? (
                <p className="px-1 py-2 text-[10.5px] text-zinc-600">No roots found.</p>
              ) : (
                visibleRoots.map((root) => (
                  <label
                    key={root.root}
                    className="flex cursor-pointer items-center gap-1.5 rounded px-1 py-0.5 text-[10.5px] hover:bg-surface"
                  >
                    <input
                      type="checkbox"
                      checked={roots.includes(root.root)}
                      onChange={() => onToggleRoot(root.root)}
                      className="h-3 w-3 rounded border-zinc-600 bg-canvas accent-blue-500"
                      data-testid={`root-filter-option-${root.root}`}
                    />
                    <span className="w-10 font-mono text-zinc-300">{root.root}</span>
                    <span className="min-w-0 flex-1 truncate text-zinc-600">{root.name}</span>
                    <span className="text-[9px] text-zinc-700">{root.sector}</span>
                  </label>
                ))
              )}
            </div>
          </div>
        ) : null}
      </div>
    </Panel>
  );
}
