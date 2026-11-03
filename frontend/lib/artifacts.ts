/**
 * Artifact / strategy diff helpers for the QoL surfaces.
 */

import type { StrategyConfig } from "./types";

/** Committed strategy ids that must never be deleted/renamed via the UI. */
export const COMMITTED_STRATEGY_IDS = new Set([
  "tsmom_vol_target",
  "drl_ppo_llm_risk",
  "carry_riskparity",
]);

export function isBuiltinStrategy(strategy: StrategyConfig | null | undefined): boolean {
  if (!strategy) return false;
  return strategy.builtin === true || COMMITTED_STRATEGY_IDS.has(strategy.id);
}

/** Flatten a nested strategy/config object into dotted parameter paths. */
export function flattenConfig(
  value: unknown,
  prefix = "",
  output: Record<string, unknown> = {},
): Record<string, unknown> {
  if (Array.isArray(value)) {
    value.forEach((entry, index) => {
      const path = prefix.length > 0 ? `${prefix}.${index}` : String(index);
      flattenConfig(entry, path, output);
    });
    return output;
  }
  if (value !== null && typeof value === "object") {
    for (const [key, entry] of Object.entries(value as Record<string, unknown>)) {
      const path = prefix.length > 0 ? `${prefix}.${key}` : key;
      flattenConfig(entry, path, output);
    }
    return output;
  }
  if (prefix.length > 0) output[prefix] = value;
  return output;
}

const IGNORED_DIFF_PREFIXES = [
  "id",
  "description",
  "tags",
  "updated_at",
  "created_at",
  "strategy_path",
  "updatedAt",
];

export interface ConfigDiffRow {
  path: string;
  a: unknown;
  b: unknown;
}

function formatDiffValue(value: unknown): string {
  if (value === undefined) return "—";
  if (value === null) return "null";
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

export function diffStrategyConfigs(
  a: Record<string, unknown> | null | undefined,
  b: Record<string, unknown> | null | undefined,
): ConfigDiffRow[] {
  if (!a || !b) return [];
  const flatA = flattenConfig(a);
  const flatB = flattenConfig(b);
  const paths = new Set([...Object.keys(flatA), ...Object.keys(flatB)]);
  const rows: ConfigDiffRow[] = [];
  for (const path of [...paths].sort()) {
    if (IGNORED_DIFF_PREFIXES.some((prefix) => path === prefix)) continue;
    const valueA = flatA[path];
    const valueB = flatB[path];
    if (formatDiffValue(valueA) === formatDiffValue(valueB)) continue;
    rows.push({ path, a: valueA, b: valueB });
  }
  return rows;
}

export { formatDiffValue };
