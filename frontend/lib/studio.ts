/**
 * Strategy Studio pure helpers (Phase 10).
 *
 * The Studio edits a declarative FinRL-X pipeline draft
 * (``w_t = R_t(T_t(A_t(S_t(X))))`` plus the deterministic contract translator)
 * and serializes it to the same YAML shape as ``configs/strategies/*.yaml``.
 * Everything here is deterministic and DOM-free so it can be unit-tested and
 * reused by the Manual Builder, the Optuna panel and the DRL panel.
 */

import type {
  DrlModel,
  DrlModelsResponse,
  DrlRewardPoint,
  DrlTrainResult,
  OptimizationApplyResult,
  OptimizationImportance,
  OptimizationListResponse,
  OptimizationParamRange,
  OptimizationStudy,
  OptimizationTrial,
  SeriesPoint,
  StageConfig,
  StrategyConfig,
  InstrumentRoot,
  InstrumentSector,
  InstrumentsResponse,
  StudioUniverseMetadata,
  StudioCatalogResponse,
  StudioDraft,
  StudioDrlConfig,
  StudioParameter,
  StudioParamOption,
  StudioPreset,
  StudioStageConfig,
  StudioStageKey,
  StudioTranslatorConfig,
  StudioValidation,
  StudioParamValue,
} from "./types";
import { formatCompactUsd } from "./format";

// ---------------------------------------------------------------------------
// Defaults / cloning
// ---------------------------------------------------------------------------

export const DEFAULT_TRANSLATOR: StudioTranslatorConfig = {
  rounding_mode: "nearest",
  rebalance_buffer_contracts: 2,
  max_margin_utilization: 0.6,
  enable_micro_routing: true,
  min_micro_notional_ratio: 0,
};

export const STUDIO_ROOTS = [
  "ES",
  "NQ",
  "RTY",
  "YM",
  "ZN",
  "ZB",
  "ZF",
  "CL",
  "GC",
  "SI",
  "HG",
  "6E",
  "6J",
  "ZC",
  "ZS",
  "LE",
] as const;

const STAGE_LABELS: Record<string, { title: string; code: string; subtitle: string }> = {
  universe: {
    title: "Universe",
    code: "S_t",
    subtitle:
      "Which futures roots are liquid and tradable today — filtered on open interest, ADV and term-structure regime.",
  },
  allocator: {
    title: "Allocator",
    code: "A_t",
    subtitle:
      "How capital is seeded across the selected roots before any timing or risk overlay touches the weights.",
  },
  timing: {
    title: "Timing & signals",
    code: "T_t",
    subtitle:
      "Trend, carry and reversion overlays that tilt the base portfolio in proportion to signal conviction.",
  },
  risk: {
    title: "Risk & LLM",
    code: "R_t",
    subtitle:
      "Quant volatility targeting plus the LLM Risk-First constraint engine that can cap or block a root.",
  },
  translator: {
    title: "Contract translator",
    code: "N_i,t",
    subtitle:
      "Deterministic mapping of approved weights to integer contracts, with micro routing and the margin ceiling.",
  },
};

export function stageMeta(stage: string): { title: string; code: string; subtitle: string } {
  return STAGE_LABELS[stage] ?? { title: stage, code: stage, subtitle: "" };
}

function cloneParams(params: Record<string, StudioParamValue>): Record<string, StudioParamValue> {
  const copy: Record<string, StudioParamValue> = {};
  for (const [key, value] of Object.entries(params)) {
    copy[key] = Array.isArray(value) ? [...value] : value;
  }
  return copy;
}

export function cloneDraft(draft: StudioDraft): StudioDraft {
  return {
    ...draft,
    tags: [...draft.tags],
    universe: { name: draft.universe.name, params: cloneParams(draft.universe.params) },
    allocator: { name: draft.allocator.name, params: cloneParams(draft.allocator.params) },
    timing: draft.timing.map((stage) => ({ name: stage.name, params: cloneParams(stage.params) })),
    risk: draft.risk.map((stage) => ({ name: stage.name, params: cloneParams(stage.params) })),
    translator: { ...draft.translator },
    ...(draft.drl ? { drl: { ...draft.drl } } : {}),
  };
}

function normaliseTranslator(raw: unknown): StudioTranslatorConfig {
  const value = isRecord(raw) ? raw : {};
  return {
    rounding_mode: asString(value.rounding_mode, "nearest"),
    rebalance_buffer_contracts: asNumber(value.rebalance_buffer_contracts, 2),
    max_margin_utilization: asNumber(value.max_margin_utilization, 0.6),
    enable_micro_routing: value.enable_micro_routing === false ? false : true,
    min_micro_notional_ratio: asNumber(value.min_micro_notional_ratio, 0),
  };
}

function normalizeInstrumentRoot(raw: unknown): InstrumentRoot | null {
  if (!isRecord(raw)) return null;
  const root = asString(raw.root, "");
  if (root.length === 0) return null;
  return {
    root,
    name: asString(raw.name, root),
    asset_class: asString(raw.asset_class, "unknown"),
    // Catalog roots expose both `sector` (component/singular) and `sector_group`.
    sector: asString(raw.sector_group ?? raw.sector, "unknown"),
    exchange: asString(raw.exchange, ""),
    multiplier: asNumber(raw.multiplier, 0),
    tick_size: asNumber(raw.tick_size, 0),
    currency: asString(raw.currency, "USD"),
    micro_root: asString(raw.micro_root, "") || null,
    standard_root: asString(raw.standard_root, "") || null,
  };
}

function normalizeInstrumentSectors(raw: unknown, roots?: InstrumentRoot[]): InstrumentSector[] {
  const sectors = asArray(raw);
  if (sectors.length === 0) return [];
  return sectors
    .map((entry) => {
      if (!isRecord(entry)) return null;
      const id = asString(entry.id, "");
      if (id.length === 0 || id === "all") return null;
      const sectorRoots = asArray(entry.roots)
        .map(normalizeInstrumentRoot)
        .filter((root): root is InstrumentRoot => root !== null);
      const fallbackRoots = roots?.filter((root) => root.sector === id) ?? [];
      const resolved = sectorRoots.length > 0 ? sectorRoots : fallbackRoots;
      return {
        id,
        label: asString(entry.label, SECTOR_LABELS[id] ?? id),
        count: typeof entry.count === "number" ? entry.count : resolved.length,
        roots: resolved,
      } satisfies InstrumentSector;
    })
    .filter((sector): sector is InstrumentSector => sector !== null);
}

/** Normalize `GET /api/instruments` (task-32) with a deterministic fallback. */
export function normalizeInstrumentsResponse(
  raw: unknown,
  fallback: InstrumentsResponse,
): InstrumentsResponse {
  if (!isRecord(raw)) return fallback;
  const roots = asArray(raw.roots)
    .map(normalizeInstrumentRoot)
    .filter((root): root is InstrumentRoot => root !== null);
  const sectors = normalizeInstrumentSectors(raw.sectors, roots);
  if (roots.length === 0 && sectors.length === 0) return fallback;
  const resolvedRoots = roots.length > 0 ? roots : universeRoots(sectors);
  return {
    generated_at: asString(raw.generated_at, fallback.generated_at),
    count: typeof raw.count === "number" ? raw.count : resolvedRoots.length,
    sectors: sectors.length > 0 ? sectors : fallback.sectors,
    roots: resolvedRoots,
  };
}

export function normalizeStrategyConfig(raw: unknown): StrategyConfig | null {
  if (!isRecord(raw)) return null;
  const id = asString(raw.id ?? raw.strategy_id, "");
  if (id.length === 0) return null;
  return {
    id,
    description: asString(raw.description, ""),
    tags: asArray(raw.tags).map(String),
    universe: encodeStage(normalizeStage(raw.universe, "liquidity")),
    allocator: encodeStage(normalizeStage(raw.allocator, "risk_parity")),
    timing: asArray(raw.timing).map((stage) => encodeStage(normalizeStage(stage))),
    risk: asArray(raw.risk).map((stage) => encodeStage(normalizeStage(stage))),
    updated_at: asString(raw.updated_at, new Date().toISOString()),
    ...(isRecord(raw.drl) ? { drl: normalizeDrlConfig(raw.drl) } : {}),
  };
}

function normalizeDrlConfig(raw: unknown): StudioDrlConfig | undefined {
  if (!isRecord(raw)) return undefined;
  return {
    algorithm: asString(raw.algorithm, "PPO"),
    policy: asString(raw.policy, "MlpPolicy"),
    total_timesteps: asNumber(raw.total_timesteps, 100_000),
    learning_rate: asNumber(raw.learning_rate, 0.0003),
    gamma: asNumber(raw.gamma, 0.99),
    seed: asNumber(raw.seed, 7),
    reward_risk_penalty_lambda: asNumber(raw.reward_risk_penalty_lambda, 0.5),
    normalize_observations: asBoolean(raw.normalize_observations, true),
    model_path: asString(raw.model_path ?? "", "") || null,
  };
}

export function normalizeStage(raw: unknown, fallbackName = ""): StudioStageConfig {
  const record = isRecord(raw) ? raw : {};
  const params: Record<string, StudioParamValue> = {};
  for (const [key, value] of Object.entries(isRecord(record.params) ? record.params : {})) {
    if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
      params[key] = value;
    } else if (Array.isArray(value)) {
      params[key] = value.map((entry) => String(entry));
    }
  }
  return {
    name: asString(record.name ?? record.selector ?? fallbackName, fallbackName),
    params,
  };
}

/** Encode studio multiselect lists into StrategyConfig-compatible scalars. */
function encodeStage(stage: StudioStageConfig): StageConfig {
  const params: Record<string, string | number | boolean> = {};
  for (const [key, value] of Object.entries(stage.params)) {
    params[key] = Array.isArray(value) ? value.join(",") : value;
  }
  return { name: stage.name, params };
}

function clonePresetStage(stage: StudioStageConfig): StudioStageConfig {
  return { name: stage.name, params: cloneParams(stage.params) };
}

export function draftFromPreset(preset: StudioPreset, id?: string): StudioDraft {
  // The catalog returns a complete POST-ready strategy dict; prefer it verbatim.
  if (preset.strategy) {
    const draft = cloneDraft(preset.strategy);
    if (id) draft.id = id;
    return draft;
  }
  return {
    id: id ?? `preset_${preset.id}`,
    description: preset.description,
    tags: [...preset.tags],
    universe: clonePresetStage(preset.universe),
    allocator: clonePresetStage(preset.allocator),
    timing: preset.timing.map(clonePresetStage),
    risk: preset.risk.map(clonePresetStage),
    translator: { ...DEFAULT_TRANSLATOR, ...preset.translator },
    ...(preset.drl ? { drl: { ...preset.drl } } : {}),
  };
}

/** Map a `GET /api/strategies` view to an editable draft (translator defaults). */
export function draftFromStrategy(strategy: StrategyConfig): StudioDraft {
  return {
    id: strategy.id,
    description: strategy.description,
    tags: [...strategy.tags],
    universe: normalizeStage(strategy.universe, "liquidity"),
    allocator: normalizeStage(strategy.allocator, "risk_parity"),
    timing: strategy.timing.map((stage) => normalizeStage(stage)),
    risk: strategy.risk.map((stage) => normalizeStage(stage)),
    translator: { ...DEFAULT_TRANSLATOR },
    ...(strategy.drl ? { drl: { ...strategy.drl } } : {}),
  };
}

// ---------------------------------------------------------------------------
// Dotted-path parameter access
// ---------------------------------------------------------------------------

export function rootsOf(draft: StudioDraft): string[] {
  const raw = draft.universe.params.roots;
  if (Array.isArray(raw)) return raw.map(String);
  if (typeof raw === "string" && raw.trim().length > 0) {
    return raw
      .split(",")
      .map((value) => value.trim())
      .filter(Boolean);
  }
  return [];
}

export function rootsToParam(roots: string[]): string {
  return roots.join(",");
}

/** Live catalog presets do not carry an explicit root list; seed it for the UI. */
export function withDefaultRoots(draft: StudioDraft, roots: string[]): StudioDraft {
  if (rootsOf(draft).length > 0 || roots.length === 0) return draft;
  const next = cloneDraft(draft);
  next.universe.params.roots = rootsToParam(roots);
  return next;
}

function findStage(stages: StudioStageConfig[], name: string): StudioStageConfig | undefined {
  return stages.find((stage) => stage.name === name);
}

/** Read the value behind a catalog parameter path. */
export function getDraftValue(
  draft: StudioDraft,
  parameter: StudioParameter,
): StudioParamValue | undefined {
  const parts = parameter.path.split(".").filter(Boolean);
  const [stage, second, third, fourth] = parts;

  if (stage === "universe" || stage === "allocator") {
    const target = stage === "universe" ? draft.universe : draft.allocator;
    if (second === "selector" || second === "name") return target.name;
    if (second === "roots") return rootsOf(draft);
    if (second === "params" && third) return target.params[third];
    if (second) return target.params[second];
    return undefined;
  }

  if (stage === "timing" || stage === "risk") {
    const stages = stage === "timing" ? draft.timing : draft.risk;
    if (!second) return undefined;
    const overlay = findStage(stages, second);
    if (third === "enabled") return Boolean(overlay);
    if (stage === "risk" && second === "llm_risk" && third === "enabled") return Boolean(overlay);
    if (third === "params" && fourth) return overlay?.params[fourth];
    if (third) return overlay?.params[third];
    return undefined;
  }

  if (stage === "translator" && second) {
    return draft.translator[second as keyof StudioTranslatorConfig] as string | number | boolean;
  }
  return undefined;
}

/** Return a new draft with the catalog parameter set to `value`. */
export function setDraftValue(
  draft: StudioDraft,
  parameter: StudioParameter,
  value: StudioParamValue,
): StudioDraft {
  const next = cloneDraft(draft);
  const parts = parameter.path.split(".").filter(Boolean);
  const [stage, second, third, fourth] = parts;

  if (stage === "universe" || stage === "allocator") {
    const target = stage === "universe" ? next.universe : next.allocator;
    if (second === "selector" || second === "name") target.name = String(value);
    else if (second === "roots") target.params.roots = Array.isArray(value) ? rootsToParam(value) : String(value);
    else if (second === "params" && third) target.params[third] = value;
    else if (second) target.params[second] = value;
    return next;
  }

  if (stage === "timing" || stage === "risk") {
    const stages = stage === "timing" ? next.timing : next.risk;
    if (!second) return next;
    const overlayName = second;
    const index = stages.findIndex((entry) => entry.name === overlayName);
    if (third === "enabled") {
      const enabled = Boolean(value);
      if (enabled && index === -1) stages.push({ name: overlayName, params: {} });
      if (!enabled && index !== -1) stages.splice(index, 1);
      return next;
    }
    const target = index === -1 ? { name: overlayName, params: {} } : stages[index];
    if (!target) return next;
    if (third === "params" && fourth) target.params[fourth] = value;
    else if (third) target.params[third] = value;
    if (index === -1) stages.push(target);
    return next;
  }

  if (stage === "translator" && second) {
    const key = second as keyof StudioTranslatorConfig;
    if (key === "max_margin_utilization" || key === "rebalance_buffer_contracts" || key === "min_micro_notional_ratio") {
      next.translator[key] = Number(value);
    } else if (key === "enable_micro_routing") {
      next.translator.enable_micro_routing = Boolean(value);
    } else if (key === "rounding_mode") {
      next.translator.rounding_mode = String(value);
    }
    return next;
  }

  return next;
}

// ---------------------------------------------------------------------------
// JSON payload + YAML serialization (multiselect -> YAML-native scalar)
// ---------------------------------------------------------------------------

export type YamlEncodingMap = Record<string, string>;

function defaultYamlEncoding(parameter: StudioParameter): string {
  if (parameter.yaml_encoding) return parameter.yaml_encoding;
  if (parameter.path.endsWith("root_month_biases")) return "json_object";
  if (parameter.type === "multiselect") {
    const values = parameter.options?.map((option) => option.value) ?? [];
    const numeric = values.length > 0 && values.every((value) => typeof value === "number");
    return numeric ? "comma_ints" : "comma_strings";
  }
  return "scalar";
}

/** Map catalog parameter paths/keys to the backend `yaml_encoding`. */
export function buildEncodingMap(parameters: StudioParameter[]): YamlEncodingMap {
  const map: YamlEncodingMap = {};
  for (const parameter of parameters) {
    const encoding = defaultYamlEncoding(parameter);
    map[parameter.path] = encoding;
    if (parameter.key) map[parameter.key] = encoding;
  }
  return map;
}

function encodeParamValue(value: StudioParamValue, encoding: string | undefined): string | number | boolean {
  if (!Array.isArray(value)) return value;
  const list = value.map((entry) => String(entry));
  if (encoding === "json_object") {
    return JSON.stringify(Object.fromEntries(list.map((entry) => [entry, {}])));
  }
  // comma_ints / comma_strings / default multiselect encoding
  return list.join(",");
}

function encodeParams(
  prefix: string,
  params: Record<string, StudioParamValue>,
  encodings?: YamlEncodingMap,
): Record<string, string | number | boolean> {
  const encoded: Record<string, string | number | boolean> = {};
  for (const [key, value] of Object.entries(params)) {
    const encoding = encodings?.[`${prefix}.${key}`] ?? encodings?.[key];
    encoded[key] = encodeParamValue(value, encoding);
  }
  return encoded;
}

function stagePayload(
  stage: StudioStageConfig,
  prefix: string,
  encodings?: YamlEncodingMap,
): { name: string; params: Record<string, string | number | boolean> } {
  return { name: stage.name, params: encodeParams(prefix, stage.params, encodings) };
}

/** Payload accepted by POST /api/strategies / PUT /api/strategies/{id}. */
export function draftToPayload(
  draft: StudioDraft,
  encodings?: YamlEncodingMap,
): Record<string, unknown> {
  return {
    id: draft.id,
    description: draft.description,
    tags: draft.tags,
    universe: stagePayload(draft.universe, "universe.params", encodings),
    allocator: stagePayload(draft.allocator, "allocator.params", encodings),
    timing: draft.timing.map((stage) =>
      stagePayload(stage, `timing.${stage.name}.params`, encodings),
    ),
    risk: draft.risk.map((stage) =>
      stagePayload(stage, `risk.${stage.name}.params`, encodings),
    ),
    translator: { ...draft.translator },
    ...(draft.drl ? { drl: { ...draft.drl } } : {}),
  };
}

function yamlScalar(value: StudioParamValue): string {
  if (Array.isArray(value)) return `[${value.map((entry) => yamlScalar(entry)).join(", ")}]`;
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return "0";
    if (Number.isInteger(value)) return String(value);
    return value.toFixed(6).replace(/0+$/, "").replace(/\.$/, ".0");
  }
  const text = String(value);
  if (text.length === 0) return '""';
  if (/^[A-Za-z0-9_./+-]+$/.test(text) && !/^(true|false|null|yes|no|on|off)$/i.test(text)) {
    return text;
  }
  return JSON.stringify(text);
}

function yamlParams(params: Record<string, StudioParamValue>, indent: string): string[] {
  const entries = Object.entries(params);
  if (entries.length === 0) return [`${indent}params: {}`];
  const lines = [`${indent}params:`];
  for (const [key, value] of entries) {
    lines.push(`${indent}  ${key}: ${yamlScalar(value)}`);
  }
  return lines;
}

/**
 * Serialize the draft to the committed strategy YAML shape
 * (``universe.selector`` is the domain wording accepted by ``parse_strategy_config``).
 */
export function draftToYaml(draft: StudioDraft, encodings?: YamlEncodingMap): string {
  const lines: string[] = [];
  lines.push(`# FuturesRL-X declarative strategy pipeline — generated by Strategy Studio`);
  lines.push(`id: ${yamlScalar(draft.id)}`);
  const description = draft.description.trim().length > 0 ? draft.description.trim() : "Strategy Studio pipeline.";
  lines.push("description: >-");
  for (const line of description.split("\n")) lines.push(`  ${line}`);
  lines.push(`tags: [${draft.tags.map((tag) => yamlScalar(tag)).join(", ")}]`);
  lines.push("");
  lines.push("universe:");
  lines.push(`  selector: ${yamlScalar(draft.universe.name)}`);
  lines.push(...yamlParams(encodeParams("universe.params", draft.universe.params, encodings), "  "));
  lines.push("");
  lines.push("allocator:");
  lines.push(`  name: ${yamlScalar(draft.allocator.name)}`);
  lines.push(...yamlParams(encodeParams("allocator.params", draft.allocator.params, encodings), "  "));
  lines.push("");
  lines.push("timing:");
  if (draft.timing.length === 0) lines.push("  []");
  for (const stage of draft.timing) {
    lines.push(`  - name: ${yamlScalar(stage.name)}`);
    lines.push(...yamlParams(encodeParams(`timing.${stage.name}.params`, stage.params, encodings), "    "));
  }
  lines.push("");
  lines.push("risk:");
  if (draft.risk.length === 0) lines.push("  []");
  for (const stage of draft.risk) {
    lines.push(`  - name: ${yamlScalar(stage.name)}`);
    lines.push(...yamlParams(encodeParams(`risk.${stage.name}.params`, stage.params, encodings), "    "));
  }
  lines.push("");
  lines.push("translator:");
  lines.push(`  rounding_mode: ${yamlScalar(draft.translator.rounding_mode)}`);
  lines.push(`  rebalance_buffer_contracts: ${yamlScalar(draft.translator.rebalance_buffer_contracts)}`);
  lines.push(`  max_margin_utilization: ${yamlScalar(draft.translator.max_margin_utilization)}`);
  lines.push(`  enable_micro_routing: ${yamlScalar(draft.translator.enable_micro_routing)}`);
  lines.push(`  min_micro_notional_ratio: ${yamlScalar(draft.translator.min_micro_notional_ratio)}`);
  if (draft.drl) {
    lines.push("");
    lines.push("drl:");
    lines.push(`  algorithm: ${yamlScalar(draft.drl.algorithm)}`);
    lines.push(`  policy: ${yamlScalar(draft.drl.policy)}`);
    lines.push(`  total_timesteps: ${yamlScalar(draft.drl.total_timesteps)}`);
    lines.push(`  learning_rate: ${yamlScalar(draft.drl.learning_rate)}`);
    lines.push(`  gamma: ${yamlScalar(draft.drl.gamma)}`);
    lines.push(`  seed: ${yamlScalar(draft.drl.seed)}`);
    lines.push(`  reward_risk_penalty_lambda: ${yamlScalar(draft.drl.reward_risk_penalty_lambda)}`);
    lines.push(`  normalize_observations: ${yamlScalar(draft.drl.normalize_observations)}`);
    lines.push(`  model_path: ${draft.drl.model_path ? yamlScalar(draft.drl.model_path) : "null"}`);
  }
  lines.push("");
  return lines.join("\n");
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

export const ID_PATTERN = /^[a-z][a-z0-9_]{0,63}$/;

export function validateDraft(draft: StudioDraft): StudioValidation {
  const errors: string[] = [];
  const warnings: string[] = [];

  if (!ID_PATTERN.test(draft.id)) {
    errors.push("Strategy ID must match ^[a-z][a-z0-9_]{0,63}$ (lowercase slug, no hyphens).");
  }
  if (draft.universe.name.trim().length === 0) errors.push("Universe selector is required.");
  if (draft.allocator.name.trim().length === 0) errors.push("Allocator is required.");
  if (rootsOf(draft).length < 2) errors.push("Select at least 2 universe roots.");
  if (draft.timing.length === 0) warnings.push("No timing overlay selected — raw allocator weights will trade as-is.");
  if (draft.risk.length === 0) warnings.push("No risk overlay selected — the deterministic Circuit Breaker will not run.");
  if (draft.risk.length > 0 && !draft.risk.some((stage) => stage.name === "llm_risk")) {
    warnings.push("LLM Risk-First overlay is off; semantic risk scores will not constrain weights.");
  }
  if (!["nearest", "floor", "ceil"].includes(String(draft.translator.rounding_mode))) {
    errors.push("Translator rounding mode must be nearest, floor or ceil.");
  }
  if (!Number.isFinite(draft.translator.max_margin_utilization) || draft.translator.max_margin_utilization <= 0) {
    errors.push("Margin ceiling η_max must be positive.");
  } else if (draft.translator.max_margin_utilization > 0.6 + 1e-9) {
    errors.push("Margin ceiling η_max cannot exceed the hard 60% policy limit.");
  }
  if (draft.translator.rebalance_buffer_contracts < 0) errors.push("Rebalance buffer cannot be negative.");
  if (draft.translator.min_micro_notional_ratio < 0 || draft.translator.min_micro_notional_ratio > 1) {
    errors.push("Micro-notional threshold must be between 0 and 1.");
  }
  return { valid: errors.length === 0, errors, warnings };
}

/** Human one-line summary of a builder stage, shown in the accordion header. */
export function stageSummary(draft: StudioDraft, stage: string): string {
  if (stage === "universe") {
    const topN = draft.universe.params.top_n;
    const roots = rootsOf(draft);
    return `${draft.universe.name} · ${roots.length} roots${typeof topN === "number" ? ` · top ${topN}` : ""}`;
  }
  if (stage === "allocator") {
    return `${draft.allocator.name}`;
  }
  if (stage === "timing") {
    return draft.timing.length > 0 ? draft.timing.map((entry) => entry.name).join(" + ") : "no overlays";
  }
  if (stage === "risk") {
    return draft.risk.length > 0 ? draft.risk.map((entry) => entry.name).join(" + ") : "no overlays";
  }
  return `${draft.translator.rounding_mode} · η_max ${(draft.translator.max_margin_utilization * 100).toFixed(0)}%`;
}

// ---------------------------------------------------------------------------
// Universe / sector helpers (task-31 / task-32)
// ---------------------------------------------------------------------------

export const SECTOR_TABS: { id: string; label: string }[] = [
  { id: "all", label: "All" },
  { id: "equities", label: "Equities" },
  { id: "rates", label: "Rates" },
  { id: "energy", label: "Energy" },
  { id: "metals", label: "Metals" },
  { id: "fx", label: "FX" },
  { id: "agri", label: "Agri" },
  { id: "crypto", label: "Crypto" },
];

const SECTOR_LABELS: Record<string, string> = {
  equity: "Equities",
  equities: "Equities",
  rates: "Rates",
  energy: "Energy",
  metals: "Metals",
  fx: "FX",
  agri: "Agri",
  crypto: "Crypto",
};

export function universeRoots(forSectors: InstrumentSector[]): InstrumentRoot[] {
  return forSectors.flatMap((sector) => sector.roots);
}

/** Default selection: every standard contract (micros remain available pairs). */
export function defaultUniverseRoots(forSectors: InstrumentSector[]): string[] {
  const roots = universeRoots(forSectors);
  const standards = roots.filter((root) => root.standard_root === null).map((root) => root.root);
  return standards.length > 0 ? standards : roots.map((root) => root.root);
}

/** Root list ordered the way the universe endpoint groups it (stable YAML output). */
export function orderUniverseRoots(roots: string[], forSectors: InstrumentSector[]): string[] {
  const order = new Map(universeRoots(forSectors).map((root, index) => [root.root, index]));
  return [...roots].sort((a, b) => (order.get(a) ?? 999) - (order.get(b) ?? 999));
}

/** Mirror Stage-2 allocator sizing into Stage-4 hard risk caps. */
export function syncQuantCaps(draft: StudioDraft): StudioDraft {
  const allocator = draft.allocator;
  const targetVol = allocator.params.target_annual_volatility;
  // `max_weight` is the allocator's per-root weight cap; gross leverage is a
  // different concept and must not be mirrored into a single-root risk cap.
  const maxWeight =
    allocator.params.max_weight ?? allocator.params.max_single_instrument_weight;
  const quantIndex = draft.risk.findIndex((stage) => stage.name === "quant_risk");
  if (quantIndex === -1) return draft;
  if (targetVol === undefined && maxWeight === undefined) return draft;
  const next = cloneDraft(draft);
  const quant = next.risk[quantIndex];
  if (!quant) return draft;
  quant.params = { ...quant.params };
  if (typeof targetVol === "number") quant.params.vol_target = targetVol;
  if (typeof maxWeight === "number" && quant.params.max_single_instrument_weight !== undefined) {
    quant.params.max_single_instrument_weight = maxWeight;
  } else if (typeof maxWeight === "number") {
    quant.params.max_single_instrument_weight = maxWeight;
  }
  return next;
}

/** Build the default params for a timing/risk overlay from the catalog. */
export function overlayDefaults(
  parameters: StudioParameter[],
  groupName: string,
): Record<string, StudioParamValue> {
  const defaults: Record<string, StudioParamValue> = {};
  for (const parameter of parameters) {
    if (parameter.stage !== "timing" && parameter.stage !== "risk") continue;
    if (parameter.stage_name !== groupName) continue;
    const key = parameter.path.endsWith(".enabled")
      ? null
      : (parameter.path.split(".").pop() ?? parameter.path);
    if (!key || key === "enabled") continue;
    defaults[key] = Array.isArray(parameter.default) ? [...parameter.default] : parameter.default;
  }
  return defaults;
}

/** Parameters that should live under the collapsible Advanced section. */
export function isAdvancedParameter(parameter: StudioParameter): boolean {
  const haystack = `${parameter.key ?? ""} ${parameter.path} ${parameter.label}`.toLowerCase();
  return /(covariance ridge|covariance_ridge|covariance shrinkage|covariance_shrinkage|min_volatility|minimum volatility|feature_fallback|feature fallback|dust|min_micro_notional|micro.*floor|ridge|shrinkage|fallback_band|band_fallback|epsilon|lambda_l2)/.test(
    haystack,
  );
}

const LABEL_OVERRIDES: Record<string, string> = {
  "allocator:target_annual_volatility": "Base Allocator Target Vol",
  "allocator:max_gross_leverage": "Base Allocator Max Gross Leverage",
  "allocator:max_single_instrument_weight": "Base Allocator Max Weight",
  "allocator:max_weight": "Base Allocator Max Weight",
  "risk:vol_target": "Hard Portfolio Vol Ceiling",
  "risk:max_single_instrument_weight": "Hard Single-Root Cap",
  "risk:max_sector_weight": "Hard Sector Cap",
};

/** Plain-English labels that disambiguate Stage-2 sizing from Stage-4 caps. */
export function parameterLabel(stage: string, parameter: StudioParameter): string {
  const paramKey = parameter.path.split(".").pop() ?? "";
  return LABEL_OVERRIDES[`${stage}:${paramKey}`] ?? parameter.label;
}

/** Formatting badge for large/currency/percent numeric parameters. */
export function numericBadge(parameter: StudioParameter, value: number): string | null {
  if (!Number.isFinite(value)) return null;
  const haystack = `${parameter.key ?? ""} ${parameter.path} ${parameter.label} ${parameter.unit ?? ""}`.toLowerCase();
  if (
    parameter.unit === "$" ||
    parameter.unit?.toUpperCase() === "USD" ||
    haystack.includes("dollar volume") ||
    haystack.includes("notional")
  ) {
    return formatCompactUsd(value);
  }
  if (parameter.unit === "%") return `${(value * 100).toFixed(1)}%`;
  if (haystack.includes("annual_volatility") || haystack.includes("vol_target") || haystack.includes("drawdown") || haystack.includes("max_loss")) {
    return `${(value * 100).toFixed(1)}%`;
  }
  if (Math.abs(value) >= 1_000_000) return value.toLocaleString("en-US");
  return null;
}

/** Build the params for a newly selected universe selector/allocator component. */
export function componentDefaults(
  parameters: StudioParameter[],
  stage: "universe" | "allocator",
  componentName: string,
): Record<string, StudioParamValue> {
  const params: Record<string, StudioParamValue> = {};
  for (const parameter of parameters) {
    if (parameter.stage !== stage) continue;
    if (parameter.stage_name !== componentName) continue;
    const key = parameter.path.split(".").pop() ?? parameter.path;
    if (key === "selector" || key === "name" || key === "enabled") continue;
    if (parameter.location === "translator" || parameter.location === "drl_block") continue;
    params[key] = Array.isArray(parameter.default) ? [...parameter.default] : parameter.default;
  }
  return params;
}

export function selectorChoices(catalog: StudioCatalogResponse): StudioParamOption[] {
  if (catalog.selector_choices && catalog.selector_choices.length > 0) return catalog.selector_choices;
  return [
    { value: "liquidity", label: "Liquidity + OI" },
    { value: "regime", label: "Macro regime" },
    { value: "composite", label: "Composite" },
  ];
}

export function allocatorChoices(catalog: StudioCatalogResponse): StudioParamOption[] {
  if (catalog.allocator_choices && catalog.allocator_choices.length > 0) return catalog.allocator_choices;
  return [
    { value: "volatility_target", label: "Volatility target" },
    { value: "risk_parity", label: "Risk parity" },
    { value: "equal_weight", label: "Equal weight" },
    { value: "mean_variance", label: "Mean variance" },
    { value: "drl", label: "DRL policy" },
  ];
}

// ---------------------------------------------------------------------------
// Tolerant contract normalization (backend shapes may evolve during Phase 10)
// ---------------------------------------------------------------------------

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function asArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function asString(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}

function asNumber(value: unknown, fallback = 0): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function asBoolean(value: unknown, fallback = false): boolean {
  return typeof value === "boolean" ? value : fallback;
}

function normaliseOptions(raw: unknown): StudioParamOption[] | undefined {
  const values = asArray(raw);
  if (values.length === 0) return undefined;
  const options: StudioParamOption[] = [];
  for (const entry of values) {
    if (isRecord(entry)) {
      const value = entry.value ?? entry.key ?? entry.id;
      if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
        options.push({ value, label: asString(entry.label ?? entry.name, String(value)) });
      }
    } else if (typeof entry === "string" || typeof entry === "number" || typeof entry === "boolean") {
      options.push({ value: entry, label: String(entry) });
    }
  }
  return options.length > 0 ? options : undefined;
}

function normaliseParameter(raw: unknown): StudioParameter | null {
  if (!isRecord(raw)) return null;
  const stage = asString(raw.stage ?? raw.stage_key, "");
  if (!["universe", "allocator", "timing", "risk", "translator"].includes(stage)) return null;
  const stageName = asString(raw.stage_name ?? raw.group ?? raw.overlay, "");
  const paramName = asString(raw.param ?? raw.key ?? raw.name ?? raw.path ?? raw.param_path, "");
  if (paramName.length === 0) return null;

  // Authoritative path scheme: `timing.<overlay>.params.<param>`,
  // `universe.params.<param>`, `translator.<param>`.
  let path: string;
  if (stage === "timing" || stage === "risk") {
    path = stageName.length > 0 ? `${stage}.${stageName}.params.${paramName}` : `${stage}.params.${paramName}`;
  } else if (stage === "translator") {
    path = `translator.${paramName}`;
  } else {
    path = `${stage}.params.${paramName}`;
  }

  const typeRaw = asString(raw.type, "");
  const type = (
    ["int", "float", "bool", "select", "multiselect", "root_multiselect", "text"].includes(typeRaw)
      ? typeRaw
      : typeRaw === "integer"
        ? "int"
        : typeRaw === "boolean"
          ? "bool"
          : typeof raw.default === "boolean"
            ? "bool"
            : typeof raw.default === "number"
              ? Number.isInteger(raw.default)
                ? "int"
                : "float"
              : "text"
  ) as StudioParameter["type"];
  const defaultValue = raw.default ?? raw.value ?? "";
  if (
    !(
      typeof defaultValue === "string" ||
      typeof defaultValue === "number" ||
      typeof defaultValue === "boolean" ||
      Array.isArray(defaultValue)
    )
  ) {
    return null;
  }
  return {
    path,
    key: asString(raw.key ?? raw.path ?? path, path),
    stage: stage as StudioStageKey,
    group: stageName || null,
    label: asString(raw.label ?? raw.title, paramName),
    description: asString(raw.description ?? raw.help, ""),
    type,
    default: Array.isArray(defaultValue) ? defaultValue.map(String) : defaultValue,
    min: typeof raw.min === "number" ? raw.min : null,
    max: typeof raw.max === "number" ? raw.max : null,
    step: typeof raw.step === "number" ? raw.step : null,
    options: normaliseOptions(raw.choices ?? raw.options),
    unit: asString(raw.unit ?? "", "") || null,
    optuna_tunable: asBoolean(raw.optuna_tunable ?? raw.tunable, false),
    optuna_default_selected: asBoolean(raw.optuna_default_selected, false),
    hygiene: asBoolean(raw.hygiene ?? raw.data_hygiene, false),
    structural: asBoolean(raw.structural ?? raw.structural_parameter, false),
    log_scale: asBoolean(raw.log_scale, false),
    yaml_encoding: asString(raw.yaml_encoding ?? "", "") || null,
    location: asString(raw.location ?? "", "") || null,
    min_items: typeof raw.min_items === "number" ? raw.min_items : null,
    max_items: typeof raw.max_items === "number" ? raw.max_items : null,
    stage_name: stageName || null,
  };
}


function normaliseDraft(raw: Record<string, unknown>, id: string): StudioDraft | null {
  if (!isRecord(raw.universe) || !isRecord(raw.allocator)) return null;
  return {
    id: asString(raw.id, id.length > 0 ? `preset_${id}` : "preset_strategy"),
    description: asString(raw.description, ""),
    tags: asArray(raw.tags).map(String),
    universe: normalizeStage(raw.universe, "liquidity"),
    allocator: normalizeStage(raw.allocator, "risk_parity"),
    timing: asArray(raw.timing).map((stage) => normalizeStage(stage)),
    risk: asArray(raw.risk).map((stage) => normalizeStage(stage)),
    translator: normaliseTranslator(raw.translator),
    ...(isRecord(raw.drl) ? { drl: normalizeDrlConfig(raw.drl) } : {}),
  };
}

function normalisePreset(raw: unknown): StudioPreset | null {
  if (!isRecord(raw)) return null;
  const id = asString(raw.id ?? raw.key ?? raw.preset_id, "");
  if (id.length === 0) return null;
  const strategySource = isRecord(raw.strategy) ? raw.strategy : isRecord(raw.config) ? raw.config : null;
  if (strategySource) {
    const strategy = normaliseDraft(strategySource, id);
    if (strategy) {
      return {
        id,
        name: asString(raw.name ?? raw.title, id),
        description: asString(raw.description ?? strategy.description, ""),
        hypothesis: asString(raw.hypothesis ?? raw.description, ""),
        tags: asArray(raw.tags).map(String),
        universe: strategy.universe,
        allocator: strategy.allocator,
        timing: strategy.timing,
        risk: strategy.risk,
        translator: strategy.translator,
        strategy,
      };
    }
  }
  return {
    id,
    name: asString(raw.name ?? raw.title, id),
    description: asString(raw.description ?? raw.hypothesis, ""),
    hypothesis: asString(raw.hypothesis ?? raw.description, ""),
    tags: asArray(raw.tags).map(String),
    universe: normalizeStage(raw.universe, "liquidity"),
    allocator: normalizeStage(raw.allocator, "risk_parity"),
    timing: asArray(raw.timing).map((stage) => normalizeStage(stage)),
    risk: asArray(raw.risk).map((stage) => normalizeStage(stage)),
    translator: normaliseTranslator(raw.translator),
    strategy: null,
  };
}

/**
 * Normalize `GET /api/strategies/schema/catalog`. Falls back to the deterministic
 * fixture whenever the live payload is not yet shape-compatible.
 */
export function normalizeStudioCatalog(
  raw: unknown,
  fallback: StudioCatalogResponse,
): StudioCatalogResponse {
  const root = isRecord(raw) ? raw : {};
  const candidate = isRecord(root.catalog) ? root.catalog : root;
  const parameterRaw = asArray(candidate.parameters ?? candidate.parameter_catalog ?? candidate.params);
  const presetRaw = asArray(
    candidate.presets ?? candidate.templates ?? candidate.hypotheses ?? candidate.strategy_templates,
  );
  const strategiesRaw = asArray(candidate.strategies ?? candidate.saved_strategies);
  if (parameterRaw.length === 0 && presetRaw.length === 0) return fallback;
  const parameters = parameterRaw
    .map(normaliseParameter)
    .filter((entry): entry is StudioParameter => entry !== null);
  const presets = presetRaw
    .map(normalisePreset)
    .filter((entry): entry is StudioPreset => entry !== null);
  const rootsRaw = asArray(candidate.roots ?? candidate.universe_roots ?? candidate.instruments).map(
    String,
  );
  const presetRoots = new Set<string>();
  for (const preset of presets) {
    for (const source of [preset.universe, preset.allocator, ...preset.timing, ...preset.risk]) {
      const rawRoots = source.params.roots;
      const values = Array.isArray(rawRoots)
        ? rawRoots
        : typeof rawRoots === "string"
          ? rawRoots.split(",")
          : [];
      for (const value of values) {
        const cleaned = String(value).trim();
        if (cleaned.length > 0) presetRoots.add(cleaned);
      }
    }
  }
  const derivedRoots =
    rootsRaw.length > 0 ? rootsRaw : presetRoots.size > 2 ? [...presetRoots] : fallback.roots;
  const resolvedParameters = parameters.length > 0 ? parameters : fallback.parameters;
  // The universe root multiselect is a Studio affordance even when the backend
  // catalog does not publish it as a parameter (root selection lives in presets).
  const hasRootParam = resolvedParameters.some(
    (parameter) => parameter.stage === "universe" && parameter.path === "universe.roots",
  );
  if (!hasRootParam) {
    resolvedParameters.unshift({
      path: "universe.roots",
      stage: "universe",
      group: null,
      label: "Universe roots",
      description:
        "Futures roots eligible for this strategy; the selector still applies the point-in-time liquidity filter.",
      type: "root_multiselect",
      default: derivedRoots,
      min: 2,
      max: derivedRoots.length,
      step: 1,
      options: undefined,
      unit: null,
      optuna_tunable: false,
      log_scale: false,
      yaml_encoding: "json_list",
      min_items: 2,
      max_items: derivedRoots.length,
      stage_name: null,
    });
  }
  const stageChoices = isRecord(candidate.stage_choices) ? candidate.stage_choices : null;
  const stageChoiceOptions = new Map<string, StudioParamOption[]>();
  for (const stage of asArray(candidate.stages)) {
    if (!isRecord(stage)) continue;
    const id = asString(stage.id, "");
    const choices: StudioParamOption[] = [];
    for (const entry of asArray(stage.choices)) {
      if (!isRecord(entry)) continue;
      const value = asString(entry.name, "");
      if (value.length === 0) continue;
      choices.push({
        value,
        label: asString(entry.label, value),
        description: asString(entry.description, ""),
      });
    }
    if (id && choices.length > 0) stageChoiceOptions.set(id, choices);
  }
  const selectorRaw =
    stageChoiceOptions.get("universe") ??
    stageChoices?.universe ??
    candidate.selector_choices ??
    (isRecord(candidate.universe) ? candidate.universe.selector_choices : undefined) ??
    candidate.selector_types;
  const allocatorRaw =
    stageChoiceOptions.get("allocator") ??
    stageChoices?.allocator ??
    candidate.allocator_choices ??
    (isRecord(candidate.universe) ? candidate.universe.allocator_choices : undefined) ??
    candidate.allocators;
  const selectorParsed = normaliseOptions(selectorRaw);
  const allocatorParsed = normaliseOptions(allocatorRaw);
  const selectorLabelMap: Record<string, string> = {
    liquidity: "Liquidity + OI",
    regime: "Macro regime",
    composite: "Composite",
  };
  const allocatorLabelMap: Record<string, string> = {
    volatility_target: "Volatility target",
    risk_parity: "Risk parity",
    equal_weight: "Equal weight",
    mean_variance: "Mean variance",
    drl: "DRL policy",
  };
  const derivedSelectors = Array.from(
    new Set(
      resolvedParameters
        .filter((parameter) => parameter.stage === "universe" && parameter.stage_name)
        .map((parameter) => parameter.stage_name as string),
    ),
  ).sort(
    (a, b) =>
      (Object.keys(selectorLabelMap).indexOf(a) + 1 || 99) -
      (Object.keys(selectorLabelMap).indexOf(b) + 1 || 99),
  );
  const derivedAllocators = Array.from(
    new Set(
      resolvedParameters
        .filter((parameter) => parameter.stage === "allocator" && parameter.stage_name)
        .map((parameter) => parameter.stage_name as string),
    ),
  ).sort(
    (a, b) =>
      (Object.keys(allocatorLabelMap).indexOf(a) + 1 || 99) -
      (Object.keys(allocatorLabelMap).indexOf(b) + 1 || 99),
  );
  const universeMetaRaw = isRecord(candidate.universe) ? candidate.universe : null;
  const universeSectors = normalizeInstrumentSectors(
    universeMetaRaw?.sectors ?? candidate.sectors,
  );
  const universeRootsRaw = asArray(
    universeMetaRaw?.roots ?? candidate.universe_roots ?? candidate.instruments,
  )
    .map(normalizeInstrumentRoot)
    .filter((root): root is InstrumentRoot => root !== null);
  const universeMetadata: StudioUniverseMetadata | null =
    universeSectors.length > 0 || universeRootsRaw.length > 0
      ? {
          sectors:
            universeSectors.length > 0
              ? universeSectors
              : fallback.universe?.sectors ?? [],
          roots:
            universeRootsRaw.length > 0
              ? universeRootsRaw
              : universeSectors.flatMap((sector) => sector.roots),
        }
      : fallback.universe ?? null;
  return {
    version: asString(candidate.version ?? candidate.schema_version, fallback.version ?? "1"),
    roots: derivedRoots,
    parameters: resolvedParameters,
    source: asString(candidate.source, "") || fallback.source || null,
    preset_source: asString(candidate.preset_source, "") || fallback.preset_source || null,
    generated_at: asString(candidate.generated_at, "") || fallback.generated_at || null,
    stage_choices: stageChoices
      ? Object.fromEntries(
          Object.entries(stageChoices).map(([key, value]) => [key, asArray(value).map(String)]),
        )
      : undefined,
    selector_choices:
      selectorParsed ??
      (derivedSelectors.length > 0
        ? derivedSelectors.map((name) => ({
            value: name,
            label: selectorLabelMap[name] ?? name.replace(/_/g, " "),
          }))
        : fallback.selector_choices),
    allocator_choices:
      allocatorParsed ??
      (derivedAllocators.length > 0
        ? derivedAllocators.map((name) => ({
            value: name,
            label: allocatorLabelMap[name] ?? name.replace(/_/g, " "),
          }))
        : fallback.allocator_choices),
    universe: universeMetadata,
    presets: presets.length > 0 ? presets : fallback.presets,
    strategies:
      strategiesRaw.length > 0
        ? (strategiesRaw.map((entry) => normalizeStrategyConfig(entry)).filter(Boolean) as StrategyConfig[])
        : fallback.strategies,
  };
}

function normalizeTrial(raw: unknown, index: number): OptimizationTrial | null {
  if (!isRecord(raw)) return null;
  const trialNumber = asNumber(raw.trial_number ?? raw.number ?? raw.trial ?? index + 1, index + 1);
  const params = isRecord(raw.params ?? raw.sampled_params) ? (raw.params ?? raw.sampled_params) : {};
  const value = asNumber(
    raw.value ?? raw.objective ?? raw.objective_value ?? raw.raw_objective ?? raw.oos_sharpe ?? raw.sharpe,
    Number.NEGATIVE_INFINITY,
  );
  return {
    trial_number: trialNumber,
    params: params as Record<string, number | string | boolean>,
    oos_sharpe: asNumber(raw.oos_sharpe ?? raw.sharpe, 0),
    cagr: asNumber(raw.cagr, 0),
    max_dd: asNumber(raw.max_dd ?? raw.max_drawdown, 0),
    calmar: asNumber(raw.calmar, 0),
    rachev: asNumber(raw.rachev ?? raw.rachev_ratio, 0),
    duration_ms: asNumber(raw.duration_ms ?? raw.duration, 0),
    value: Number.isFinite(value) ? value : 0,
    best: asBoolean(raw.best, false),
  };
}

function normaliseImportances(raw: unknown): OptimizationImportance[] {
  if (Array.isArray(raw)) {
    return raw
      .map((entry) => {
        if (!isRecord(entry)) return null;
        const path = asString(entry.param_path ?? entry.param ?? entry.name ?? entry.key, "");
        if (path.length === 0) return null;
        return { param_path: path, importance: asNumber(entry.importance ?? entry.value, 0) };
      })
      .filter((entry): entry is OptimizationImportance => entry !== null);
  }
  if (isRecord(raw)) {
    return Object.entries(raw)
      .filter(([, value]) => typeof value === "number")
      .map(([key, value]) => ({ param_path: key, importance: value as number }));
  }
  return [];
}

function normaliseBestConfig(raw: unknown): StudioDraft | null {
  if (!isRecord(raw)) return null;
  if (!isRecord(raw.universe) || !isRecord(raw.allocator)) return null;
  return {
    id: asString(raw.id, "optimized_strategy"),
    description: asString(raw.description, ""),
    tags: asArray(raw.tags).map(String),
    universe: normalizeStage(raw.universe, "liquidity"),
    allocator: normalizeStage(raw.allocator, "risk_parity"),
    timing: asArray(raw.timing).map((stage) => normalizeStage(stage)),
    risk: asArray(raw.risk).map((stage) => normalizeStage(stage)),
    translator: normaliseTranslator(raw.translator),
  };
}

export function normalizeOptimizationStudy(raw: unknown): OptimizationStudy | null {
  if (!isRecord(raw)) return null;
  const source = isRecord(raw.study) ? raw.study : raw;
  const trials = asArray(source.trials ?? source.results)
    .map((trial, index) => normalizeTrial(trial, index))
    .filter((trial): trial is OptimizationTrial => trial !== null);
  // GET /api/optimize returns summaries without trials; keep them selectable and
  // fetch the full study on demand.
  const bestTrial =
    trials.length > 0
      ? (trials.find((trial) => trial.best) ??
        trials.reduce((best, trial) => (trial.value > best.value ? trial : best), trials[0] as OptimizationTrial))
      : null;
  const bestTrialNumberRaw = source.best_trial_number ?? source.best_trial;
  return {
    study_id: asString(source.study_id ?? source.id ?? source.name, `study_${Date.now()}`),
    base_strategy_id: asString(source.base_strategy_id ?? source.strategy_id, ""),
    objective_metric: asString(source.objective_metric ?? source.objective, "sharpe"),
    n_trials: asNumber(source.n_trials ?? source.trials_count, trials.length),
    completed_trials: asNumber(source.completed_trials ?? source.n_trials, trials.length),
    walk_forward: asBoolean(source.walk_forward, true),
    wf_splits: asNumber(source.wf_splits ?? source.fold_count ?? source.n_folds, 4),
    fold_count: asNumber(source.fold_count ?? source.wf_splits, 4),
    status: asString(source.status, "completed"),
    created_at: asString(source.created_at ?? source.created, new Date().toISOString()),
    duration_ms: asNumber(source.duration_ms ?? source.duration, 0),
    best_value: asNumber(source.best_value ?? source.best_objective, bestTrial?.value ?? 0),
    best_raw_value: asNumber(source.best_raw_value ?? source.best_value, bestTrial?.value ?? 0),
    best_trial_number:
      typeof bestTrialNumberRaw === "number"
        ? bestTrialNumberRaw
        : (bestTrial?.trial_number ?? null),
    best_params: isRecord(source.best_params)
      ? (source.best_params as Record<string, number | string | boolean>)
      : bestTrial
        ? { ...bestTrial.params }
        : {},
    param_importances: normaliseImportances(source.param_importances ?? source.importances),
    trials,
    best_strategy_config:
      normaliseBestConfig(source.best_strategy_config ?? source.best_config ?? source.best_strategy) ??
      null,
    direction: asString(source.direction, "") || null,
    backend: asString(source.backend, "") || null,
    provenance: asString(source.provenance, "") || null,
    artifact_path: asString(source.artifact_path, "") || null,
    message: asString(source.message, "") || null,
  };
}

export function normalizeOptimizationList(raw: unknown): OptimizationListResponse {
  const root = isRecord(raw) ? raw : {};
  const values = Array.isArray(raw)
    ? raw
    : asArray(root.studies ?? root.optimizations ?? root.results);
  const studies = values
    .map(normalizeOptimizationStudy)
    .filter((study): study is OptimizationStudy => study !== null);
  return { studies };
}

function normaliseRewardCurveFromPayload(raw: unknown): DrlRewardPoint[] {
  const source = isRecord(raw) && Array.isArray(raw.points) ? raw.points : raw;
  if (Array.isArray(source)) {
    return source
      .map((entry, index) => {
        if (isRecord(entry)) {
          return {
            step: asNumber(entry.timestep ?? entry.step ?? entry.episode, index + 1),
            reward: asNumber(entry.mean_reward ?? entry.reward ?? entry.value, 0),
          };
        }
        if (typeof entry === "number") return { step: index + 1, reward: entry };
        return null;
      })
      .filter((entry): entry is DrlRewardPoint => entry !== null);
  }
  return [];
}

function normaliseRewardSummary(raw: unknown): DrlModel["reward_summary"] {
  if (!isRecord(raw)) return null;
  return {
    window_steps: typeof raw.window_steps === "number" ? raw.window_steps : null,
    samples: asNumber(raw.samples, 0),
    episode_count: asNumber(raw.episode_count, 0),
    env_steps: typeof raw.env_steps === "number" ? raw.env_steps : null,
    final_mean_reward: typeof raw.final_mean_reward === "number" ? raw.final_mean_reward : null,
    best_mean_reward: typeof raw.best_mean_reward === "number" ? raw.best_mean_reward : null,
    path: asString(raw.path, "") || null,
  };
}

/**
 * Normalize one checkpoint from GET /api/drl/models or the POST /api/drl/train
 * metadata summary. Metadata-missing checkpoints are still listed (risk-eng
 * contract) and flagged with `metadata_available=false`.
 */
export function normalizeDrlModel(raw: unknown): DrlModel | null {
  if (!isRecord(raw)) return null;
  const name = asString(
    raw.checkpoint_name ?? raw.name ?? raw.model_name ?? raw.checkpoint ?? raw.id,
    "",
  );
  if (name.length === 0) return null;
  const meta = isRecord(raw.meta) ? raw.meta : raw;
  const curve = normaliseRewardCurveFromPayload(
    raw.reward_curve ?? meta.reward_curve ?? raw.reward_history,
  );
  const summary = normaliseRewardSummary(raw.reward_curve ?? meta.reward_curve);
  const features = asArray(raw.features ?? raw.feature_names ?? meta.features ?? meta.feature_names).map(String);
  const universe = asArray(raw.universe ?? raw.roots ?? meta.universe ?? meta.roots).map(String);
  return {
    name,
    algorithm: asString(raw.algorithm ?? meta.algorithm, ""),
    timesteps: asNumber(raw.total_timesteps ?? raw.timesteps ?? meta.total_timesteps ?? meta.timesteps, 0),
    universe,
    features,
    created_at: asString(raw.created_at ?? raw.trained_at ?? raw.created ?? meta.created_at, ""),
    git_commit: asString(raw.git_commit ?? meta.git_commit, "") || null,
    size_bytes:
      typeof raw.file_size_bytes === "number"
        ? raw.file_size_bytes
        : typeof raw.size_bytes === "number"
          ? raw.size_bytes
          : null,
    reward_curve: curve,
    reward_summary: summary,
    hyperparameters: isRecord(raw.hyperparameters)
      ? (raw.hyperparameters as Record<string, number | string | boolean>)
      : isRecord(meta.hyperparameters)
        ? (meta.hyperparameters as Record<string, number | string | boolean>)
        : {},
    meta,
    path: asString(raw.model_path ?? raw.path ?? raw.checkpoint_path, "") || null,
    meta_path: asString(raw.meta_path, "") || null,
    metadata_available: raw.metadata_available === false ? false : true,
    metadata_error: asString(raw.metadata_error, "") || null,
    sb3_algorithm: asString(raw.sb3_algorithm, "") || null,
    strategy_id: asString(raw.strategy_id, "") || null,
    policy: asString(raw.policy, "") || null,
    learning_rate: typeof raw.learning_rate === "number" ? raw.learning_rate : null,
    gamma: typeof raw.gamma === "number" ? raw.gamma : null,
    llm_reward_penalty_lambda:
      typeof raw.llm_reward_penalty_lambda === "number" ? raw.llm_reward_penalty_lambda : null,
    seed: typeof raw.seed === "number" ? raw.seed : null,
    device: asString(raw.device, "") || null,
    offline: typeof raw.offline === "boolean" ? raw.offline : null,
  };
}

export function normalizeDrlModelsResponse(raw: unknown): DrlModelsResponse {
  const root = isRecord(raw) ? raw : {};
  const values = Array.isArray(raw) ? raw : asArray(root.models);
  return {
    models: values.map(normalizeDrlModel).filter((model): model is DrlModel => model !== null),
    count: typeof root.count === "number" ? root.count : values.length,
    models_dir: asString(root.models_dir, "") || null,
    generated_at: asString(root.generated_at, "") || null,
  };
}

export function normalizeDrlTrainResult(raw: unknown): DrlTrainResult | null {
  const root = isRecord(raw) ? raw : {};
  const source = isRecord(root.model) ? root.model : root;
  const model = normalizeDrlModel(source);
  if (!model) return null;
  const curve = normaliseRewardCurveFromPayload(root.reward_curve ?? source.reward_curve);
  return {
    model: { ...model, reward_curve: curve.length > 0 ? curve : model.reward_curve },
    reward_curve: curve.length > 0 ? curve : model.reward_curve,
    summary: root,
  };
}

export function normalizeOptimizationApply(raw: unknown, fallbackStudyId: string): OptimizationApplyResult {
  const root = isRecord(raw) ? raw : {};
  const strategy = isRecord(root.strategy) ? normalizeStrategyConfig(root.strategy) : null;
  const verification = isRecord(root.verification_backtest)
    ? {
        status: asString(root.verification_backtest.status, "skipped"),
        message: asString(root.verification_backtest.message, "") || undefined,
      }
    : null;
  return {
    study_id: asString(root.study_id, fallbackStudyId),
    strategy_id: asString(strategy?.id ?? root.strategy_id, "") || `${fallbackStudyId}_opt`,
    strategy,
    backtest_run_id: isRecord(root.verification_backtest) && isRecord(root.verification_backtest.run)
      ? asString(root.verification_backtest.run.id, "") || null
      : null,
    applied: true,
    best_params: isRecord(root.best_params)
      ? (root.best_params as Record<string, number | string | boolean>)
      : undefined,
    verification_backtest: verification,
    message: asString(root.message, ""),
  };
}

// ---------------------------------------------------------------------------
// Chart series + formatting helpers
// ---------------------------------------------------------------------------

/** Objective history: trial N is projected onto a synthetic timeline. */
export function trialObjectiveSeries(
  study: OptimizationStudy,
  baseMs = Date.now(),
): { trial: SeriesPoint[]; best: SeriesPoint[] } {
  const sorted = [...study.trials].sort((a, b) => a.trial_number - b.trial_number);
  let best = Number.NEGATIVE_INFINITY;
  const trial: SeriesPoint[] = [];
  const bestLine: SeriesPoint[] = [];
  sorted.forEach((entry, index) => {
    const time = new Date(baseMs + index * 60_000).toISOString();
    trial.push({ time, value: entry.value });
    best = Math.max(best, entry.value);
    bestLine.push({ time, value: best });
  });
  return { trial, best: bestLine };
}

export function trialTimeFormatter(baseMs = Date.now()): (iso: string) => string {
  return (iso: string) => {
    const delta = new Date(iso).getTime() - baseMs;
    const index = Math.round(delta / 60_000) + 1;
    return Number.isFinite(index) ? `Trial ${index}` : "Trial";
  };
}

export function rewardSeries(model: DrlModel, baseMs = Date.now()): SeriesPoint[] {
  return model.reward_curve.map((point, index) => ({
    time: new Date(baseMs + index * 1_000).toISOString(),
    value: point.reward,
  }));
}

export function rewardTimeFormatter(model: DrlModel, baseMs = Date.now()): (iso: string) => string {
  const steps = model.reward_curve.map((point) => point.step);
  return (iso: string) => {
    const index = Math.round((new Date(iso).getTime() - baseMs) / 1_000);
    const step = steps[index];
    return step !== undefined ? `step ${step}` : "step";
  };
}

export function importancePercent(importance: number, values: number[]): number {
  const max = Math.max(...values.map((value) => Math.abs(value)), 1e-9);
  return Math.max(0, Math.min(100, (Math.abs(importance) / max) * 100));
}

export function formatObjectiveMetric(metric: string): string {
  switch (metric) {
    case "sharpe":
    case "sortino":
    case "calmar":
    case "rachev":
      return metric.charAt(0).toUpperCase() + metric.slice(1);
    case "cagr":
      return "CAGR";
    case "return_on_margin":
      return "Return on margin";
    default:
      return metric;
  }
}

export function formatParamPath(path: string): string {
  return path
    .replace(/\.params\./g, " · ")
    .replace(/\./g, " · ")
    .replace(/_/g, " ");
}

/** Build the Optuna search-space from enabled tunable bounds. */
export function buildSearchSpace(
  parameters: StudioParameter[],
  bounds: Record<string, { enabled: boolean; low: number; high: number }>,
): OptimizationParamRange[] {
  const space: OptimizationParamRange[] = [];
  for (const parameter of parameters) {
    const entry = bounds[parameter.path];
    if (!entry?.enabled) continue;
    if (parameter.type === "int" || parameter.type === "float") {
      const low = Math.min(entry.low, entry.high);
      const high = Math.max(entry.low, entry.high);
      space.push({
        param_path: parameter.key ?? parameter.path,
        type: parameter.type,
        low,
        high,
        ...(parameter.type === "int" ? { step: parameter.step ?? 1 } : {}),
      });
    } else if (parameter.type === "select" && parameter.options && parameter.options.length > 0) {
      space.push({
        param_path: parameter.key ?? parameter.path,
        type: "categorical",
        choices: parameter.options.map((option) => String(option.value)),
      });
    }
  }
  return space;
}

/** Derive the default bounds row for each tunable parameter. */
export function defaultTunableBounds(
  parameters: StudioParameter[],
): Record<string, { enabled: boolean; low: number; high: number }> {
  const bounds: Record<string, { enabled: boolean; low: number; high: number }> = {};
  for (const parameter of parameters) {
    if (!parameter.optuna_tunable) continue;
    if (parameter.type !== "int" && parameter.type !== "float") continue;
    const defaultNumber = typeof parameter.default === "number" ? parameter.default : 0;
    bounds[parameter.path] = {
      enabled: true,
      low: parameter.min ?? Math.min(defaultNumber * 0.5, 0),
      high: parameter.max ?? Math.max(defaultNumber * 1.5, 1),
    };
  }
  return bounds;
}

// ---------------------------------------------------------------------------
// v1.5.1 Optuna active-strategy filtering + curated default selection
// ---------------------------------------------------------------------------

/** Curated Optuna default keys from strategy-eng task-36. */
export const OPTUNA_DEFAULT_KEYS = new Set<string>([
  "timing.tsmom.lookback_fast",
  "timing.tsmom.lookback_slow",
  "timing.kama.window",
  "timing.carry_filter.min_carry_bps",
  "allocator.volatility_target.target_annual_volatility",
  "risk.quant_risk.trailing_stop_pct",
]);

/** Structural / data-hygiene keys that must never be tuned. */
const OPTUNA_HYGIENE_KEYS = new Set<string>([
  "universe.liquidity.min_avg_dollar_volume",
  "universe.liquidity.min_open_interest",
  "universe.liquidity.min_history_days",
  "universe.liquidity.top_n",
  "universe.regime.top_n",
  "risk.llm_risk.sample_count",
  "risk.llm_risk.cache_ttl_seconds",
  "risk.llm_risk.macro_lockout_minutes_before",
  "risk.llm_risk.macro_lockout_minutes_after",
  "risk.llm_risk.macro_lockout_action",
  "risk.llm_risk.macro_lockout_leverage_cap",
]);

const HYGIENE_PATTERN =
  /(min_avg_dollar_volume|min_open_interest|min_history_days|top_n|sample_count|cache_ttl|macro_lockout|risk_off_roots|risk_off_sectors)/i;

function parameterKeyCandidates(parameter: StudioParameter): string[] {
  return [parameter.key ?? "", parameter.path].filter((value) => value.length > 0);
}

/** True for data-hygiene / structural params (not selectable in Optuna). */
export function isStructuralParameter(parameter: StudioParameter): boolean {
  if (parameter.hygiene || parameter.structural) return true;
  if (!parameter.optuna_tunable) return true;
  const candidates = parameterKeyCandidates(parameter);
  if (candidates.some((key) => OPTUNA_HYGIENE_KEYS.has(key))) return true;
  return candidates.some((key) => HYGIENE_PATTERN.test(key));
}

function stageComponentNames(stages: { name: string; params: Record<string, StudioParamValue> }[]): Set<string> {
  const names = new Set<string>();
  for (const stage of stages) {
    names.add(stage.name);
    if (stage.name === "composite") {
      const components = stage.params.components;
      const values = Array.isArray(components)
        ? components.map(String)
        : typeof components === "string" && components.length > 0
          ? components.split(",").map((entry) => entry.trim())
          : [];
      for (const value of values) names.add(value);
    }
  }
  return names;
}

/** Stage choices active in the selected base strategy. */
export function activeStageChoices(strategy: StrategyConfig): {
  universe: Set<string>;
  allocator: Set<string>;
  timing: Set<string>;
  risk: Set<string>;
} {
  return {
    universe: stageComponentNames([strategy.universe]),
    allocator: stageComponentNames([strategy.allocator]),
    timing: stageComponentNames(strategy.timing),
    risk: stageComponentNames(strategy.risk),
  };
}

/** A parameter is active when its stage component/overlay is present in the strategy. */
export function isParameterActiveForStrategy(
  parameter: StudioParameter,
  strategy: StrategyConfig,
  choices?: ReturnType<typeof activeStageChoices>,
): boolean {
  const active = choices ?? activeStageChoices(strategy);
  const stageName = parameter.stage_name ?? "";
  if (parameter.stage === "translator") return true;
  if (stageName.length === 0) return true;
  if (parameter.stage === "universe") return active.universe.has(stageName);
  if (parameter.stage === "allocator") return active.allocator.has(stageName);
  if (parameter.stage === "timing") return active.timing.has(stageName);
  if (parameter.stage === "risk") return active.risk.has(stageName);
  return true;
}

/** Numeric tunable parameters active in the selected base strategy (no hygiene). */
export function activeTunableParameters(
  strategy: StrategyConfig,
  catalog: StudioCatalogResponse,
): StudioParameter[] {
  const choices = activeStageChoices(strategy);
  return catalog.parameters.filter(
    (parameter) =>
      (parameter.type === "int" || parameter.type === "float") &&
      parameter.optuna_tunable &&
      isParameterActiveForStrategy(parameter, strategy, choices) &&
      !isStructuralParameter(parameter),
  );
}

/** Active structural/hygiene params for the "excluded" explainer. */
export function structuralParametersForStrategy(
  strategy: StrategyConfig,
  catalog: StudioCatalogResponse,
): StudioParameter[] {
  const choices = activeStageChoices(strategy);
  return catalog.parameters.filter(
    (parameter) =>
      isParameterActiveForStrategy(parameter, strategy, choices) &&
      isStructuralParameter(parameter),
  );
}

/**
 * Pre-check only curated/default-selected active params: catalog
 * `optuna_default_selected` when available, otherwise the frozen curated keys.
 */
export function defaultOptunaSelection(parameters: StudioParameter[]): string[] {
  const flagged = parameters
    .filter((parameter) => parameter.optuna_default_selected)
    .map((parameter) => parameter.path);
  if (flagged.length > 0) return flagged;
  return parameters
    .filter((parameter) =>
      parameterKeyCandidates(parameter).some((key) => OPTUNA_DEFAULT_KEYS.has(key)),
    )
    .map((parameter) => parameter.path);
}

export function defaultOptunaBounds(
  parameters: StudioParameter[],
  selectedPaths: string[],
): Record<string, { enabled: boolean; low: number; high: number }> {
  const selected = new Set(selectedPaths);
  const bounds: Record<string, { enabled: boolean; low: number; high: number }> = {};
  for (const parameter of parameters) {
    if (parameter.type !== "int" && parameter.type !== "float") continue;
    const defaultNumber = typeof parameter.default === "number" ? parameter.default : 0;
    bounds[parameter.path] = {
      enabled: selected.has(parameter.path),
      low: parameter.min ?? Math.min(defaultNumber * 0.5, 0),
      high: parameter.max ?? Math.max(defaultNumber * 1.5, 1),
    };
  }
  return bounds;
}

export function drlModelMetaJson(model: DrlModel): string {
  return JSON.stringify(model.meta ?? {}, null, 2);
}

export function asStageParamValue(value: string | number | boolean | string[]): string | number | boolean {
  return Array.isArray(value) ? rootsToParam(value) : value;
}
