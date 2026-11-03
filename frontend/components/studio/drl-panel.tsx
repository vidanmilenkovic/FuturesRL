"use client";

import { useEffect, useMemo, useState } from "react";

import {
  Badge,
  Button,
  EmptyState,
  Field,
  Panel,
  Select,
  TextInput,
  type Tone,
} from "@/components/ui/primitives";
import { bindDrlModel, listDrlModels, trainDrlModel } from "@/lib/api";
import { cn } from "@/lib/cn";
import { formatDateTime, formatInteger } from "@/lib/format";
import { drlModelMetaJson } from "@/lib/studio";
import type {
  DrlAlgorithm,
  DrlModel,
  DrlTrainRequest,
  StrategyConfig,
} from "@/lib/types";
import { useAsyncData } from "@/lib/use-async-data";

import { InlineField, JsonPre, OrdinalLineChart } from "./controls";

const ALGORITHMS: { value: DrlAlgorithm; label: string }[] = [
  { value: "PPO", label: "PPO" },
  { value: "CPPO", label: "CPPO (constrained)" },
  { value: "SAC", label: "SAC" },
  { value: "TD3", label: "TD3" },
  { value: "A2C", label: "A2C" },
];

const CHECKPOINT_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/;

export function DrlPanel({
  strategies,
  onStrategiesChanged,
  onNotify,
}: {
  strategies: StrategyConfig[];
  onStrategiesChanged: () => void;
  onNotify: (message: string, tone: Tone) => void;
}) {
  const modelsSource = useAsyncData(listDrlModels);
  const [extraModels, setExtraModels] = useState<DrlModel[]>([]);
  const [algorithm, setAlgorithm] = useState<DrlAlgorithm>("PPO");
  const [baseId, setBaseId] = useState("");
  const [timesteps, setTimesteps] = useState(20_000);
  const [learningRate, setLearningRate] = useState(0.0003);
  const [gamma, setGamma] = useState(0.99);
  const [penaltyLambda, setPenaltyLambda] = useState(0.5);
  const [seed, setSeed] = useState(7);
  const [device, setDevice] = useState("cpu");
  const [checkpoint, setCheckpoint] = useState("ppo_studio_20k");
  const [selectedName, setSelectedName] = useState("");
  const [training, setTraining] = useState(false);
  const [binding, setBinding] = useState(false);
  const [message, setMessage] = useState<{ text: string; tone: Tone } | null>(null);

  useEffect(() => {
    if (!baseId && strategies.length > 0) {
      const match = strategies.find((strategy) => strategy.allocator.name === "drl") ?? strategies[0];
      setBaseId(match?.id ?? "");
    }
  }, [baseId, strategies]);

  const models = useMemo(() => {
    const merged = [...extraModels, ...(modelsSource.data?.models ?? [])];
    return merged.filter((model, index) => merged.findIndex((entry) => entry.name === model.name) === index);
  }, [extraModels, modelsSource.data]);

  useEffect(() => {
    if (!selectedName && models.length > 0) setSelectedName(models[0]?.name ?? "");
  }, [models, selectedName]);

  const selected = models.find((model) => model.name === selectedName) ?? models[0] ?? null;
  const baseStrategy = strategies.find((strategy) => strategy.id === baseId) ?? null;
  const checkpointValid = checkpoint.trim() === "" || (CHECKPOINT_PATTERN.test(checkpoint) && !checkpoint.includes(".."));
  const timestepsValid = Number.isFinite(timesteps) && timesteps >= 1 && timesteps <= 20_000;
  const canTrain = Boolean(baseStrategy) && checkpointValid && timestepsValid && !training;

  const train = async () => {
    if (!baseStrategy) return;
    setTraining(true);
    setMessage(null);
    const request: DrlTrainRequest = {
      strategy_id: baseStrategy.id,
      algorithm,
      timesteps: Math.max(1, Math.min(20_000, Math.round(timesteps))),
      learning_rate: Number.isFinite(learningRate) ? learningRate : null,
      gamma: Number.isFinite(gamma) ? gamma : null,
      llm_reward_penalty_lambda: Number.isFinite(penaltyLambda) ? penaltyLambda : null,
      checkpoint_name: checkpoint.trim().length > 0 ? checkpoint.trim() : null,
      seed: Number.isFinite(seed) ? Math.max(0, Math.round(seed)) : null,
      device: device.trim() || "cpu",
      periods: null,
      offline: true,
    };
    const result = await trainDrlModel(request);
    setTraining(false);
    setExtraModels((current) => [result.data.model, ...current.filter((entry) => entry.name !== result.data.model.name)]);
    setSelectedName(result.data.model.name);
    setMessage({
      text:
        result.source === "live"
          ? `Trained ${result.data.model.name} · ${formatInteger(result.data.model.timesteps)} steps · ${result.data.model.algorithm}`
          : `UI fallback training summary · ${result.data.model.name}`,
      tone: result.source === "live" ? "gain" : "warn",
    });
    onNotify(`Checkpoint ${result.data.model.name} registered`, result.source === "live" ? "gain" : "warn");
    modelsSource.reload();
  };

  const bind = async (model: DrlModel | null) => {
    if (!model || !baseStrategy) return;
    setBinding(true);
    const result = await bindDrlModel(baseStrategy, model);
    setBinding(false);
    setMessage({
      text:
        result.source === "live"
          ? `Bound ${model.name} → ${baseStrategy.id} (DRL allocator + model_path updated).`
          : `UI fallback: ${model.name} bound to ${baseStrategy.id} in memory.`,
      tone: result.source === "live" ? "gain" : "warn",
    });
    onNotify(`Bound ${model.name} → ${baseStrategy.id}`, result.source === "live" ? "gain" : "warn");
    onStrategiesChanged();
  };

  const paramsLabel = baseStrategy
    ? `${baseStrategy.universe.name}${typeof baseStrategy.universe.params.top_n === "number" ? ` · top ${baseStrategy.universe.params.top_n}` : ""}`
    : "—";

  return (
    <div className="grid min-h-full grid-cols-12 gap-3">
      {/* ------------------------------------------------------------ config */}
      <div className="col-span-12 flex min-h-0 flex-col gap-3 xl:col-span-5">
        <Panel
          title="Neural trainer configuration"
          className="shrink-0"
          actions={<Badge tone="neutral">offline · bounded 20k steps</Badge>}
        >
          <div className="grid grid-cols-2 gap-x-3 gap-y-2.5">
            <Field label="Algorithm" htmlFor="drl-algo">
              <Select
                id="drl-algo"
                value={algorithm}
                onChange={(value) => setAlgorithm(value as DrlAlgorithm)}
                options={ALGORITHMS}
              />
            </Field>
            <Field label="Base strategy" htmlFor="drl-base">
              <Select
                id="drl-base"
                value={baseId}
                onChange={setBaseId}
                options={strategies.map((strategy) => ({ value: strategy.id, label: strategy.id }))}
              />
            </Field>
            <Field label="Timesteps" htmlFor="drl-steps">
              <TextInput
                id="drl-steps"
                type="number"
                min={1}
                max={20_000}
                step={100}
                value={timesteps}
                onChange={(value) => setTimesteps(Number(value))}
              />
            </Field>
            <Field label="Learning rate" htmlFor="drl-lr">
              <TextInput
                id="drl-lr"
                type="number"
                step={0.0001}
                min={0}
                value={learningRate}
                onChange={(value) => setLearningRate(Number(value))}
              />
            </Field>
            <Field label="Gamma" htmlFor="drl-gamma">
              <TextInput
                id="drl-gamma"
                type="number"
                step={0.01}
                min={0}
                value={gamma}
                onChange={(value) => setGamma(Number(value))}
              />
            </Field>
            <Field label="LLM penalty λ" htmlFor="drl-lambda">
              <TextInput
                id="drl-lambda"
                type="number"
                step={0.05}
                min={0}
                value={penaltyLambda}
                onChange={(value) => setPenaltyLambda(Number(value))}
              />
            </Field>
            <Field label="Seed" htmlFor="drl-seed">
              <TextInput
                id="drl-seed"
                type="number"
                min={0}
                step={1}
                value={seed}
                onChange={(value) => setSeed(Number(value))}
              />
            </Field>
            <Field label="Device" htmlFor="drl-device">
              <Select
                id="drl-device"
                value={device}
                onChange={setDevice}
                options={[
                  { value: "cpu", label: "cpu" },
                  { value: "cuda", label: "cuda (if available)" },
                ]}
              />
            </Field>
            <div className="col-span-2">
              <div className="mb-1 flex items-baseline justify-between">
                <span className="text-[11px] font-medium text-zinc-500">Universe</span>
                <span className="text-[10px] text-zinc-600">inherited from the base strategy</span>
              </div>
              <div className="rounded-md border border-edge bg-canvas px-2.5 py-1.5 text-[11px] text-zinc-300">
                {paramsLabel}
                {baseStrategy ? (
                  <span className="ml-2 text-[10px] text-zinc-600">
                    rocks are selected by the strategy&apos;s universe selector
                  </span>
                ) : null}
              </div>
            </div>
            <Field
              label="Checkpoint name"
              htmlFor="drl-checkpoint"
              className="col-span-2"
              hint={
                checkpointValid
                  ? "Saved as artifacts/models/<name>.zip + .meta.json + reward_curve.json."
                  : "1–64 chars, letters/digits/._-, must start alphanumeric, no '..'."
              }
            >
              <TextInput
                id="drl-checkpoint"
                value={checkpoint}
                placeholder="auto-generated when empty"
                onChange={(value) => setCheckpoint(value)}
              />
            </Field>
          </div>
          <div className="mt-3 flex flex-wrap items-center gap-2 border-t border-hairline pt-2">
            <Button variant="primary" onClick={() => void train()} disabled={!canTrain}>
              {training ? "Training…" : "Train checkpoint"}
            </Button>
            <span className="font-mono text-[10.5px] text-zinc-600">
              {formatInteger(timesteps)} steps · {algorithm}
              {!timestepsValid ? " · out of bounds" : ""}
            </span>
            {message ? (
              <span
                className={cn(
                  "ml-auto min-w-0 truncate text-[10.5px]",
                  message.tone === "gain" ? "text-emerald-400" : message.tone === "warn" ? "text-amber-400" : "text-zinc-500",
                )}
                title={message.text}
              >
                {message.text}
              </span>
            ) : null}
          </div>
        </Panel>

        <Panel
          title="Reward summary"
          className="min-h-0 flex-1"
          actions={
            <Badge tone={modelsSource.source === "live" ? "gain" : "warn"}>
              {modelsSource.source === "live" ? "Registry live" : "UI fallback"}
            </Badge>
          }
        >
          <div className="grid grid-cols-3 gap-2">
            <InlineField label="Models">
              <span className="font-mono text-[13px] text-zinc-200">{models.length}</span>
            </InlineField>
            <InlineField label="Algorithms">
              <span className="font-mono text-[13px] text-zinc-200">
                {Array.from(new Set(models.map((model) => model.algorithm))).join(" · ") || "—"}
              </span>
            </InlineField>
            <InlineField label="Total steps">
              <span className="font-mono text-[13px] text-zinc-200">
                {formatInteger(models.reduce((sum, model) => sum + model.timesteps, 0))}
              </span>
            </InlineField>
          </div>
          <div className="mt-3 min-h-0 flex-1 overflow-auto">
            {models.length === 0 ? (
              <EmptyState title="No checkpoints trained yet" description="Train a DRL checkpoint to populate the model registry." />
            ) : (
              <div className="space-y-1.5">
                {models.slice(0, 5).map((model) => {
                  const last = model.reward_summary?.final_mean_reward ?? model.reward_curve[model.reward_curve.length - 1]?.reward ?? null;
                  return (
                    <div key={model.name} className="flex items-center justify-between gap-2 rounded border border-hairline bg-canvas/50 px-2 py-1.5">
                      <span className="min-w-0">
                        <span className="block truncate font-mono text-[11px] text-zinc-300">{model.name}</span>
                        <span className="block truncate text-[10px] text-zinc-600">
                          {model.algorithm} · {formatInteger(model.timesteps)} steps
                        </span>
                      </span>
                      <span className={cn("font-mono text-[11px] tabular-nums", last === null ? "text-zinc-600" : last >= 0 ? "text-emerald-400" : "text-loss")}>
                        {last === null ? "—" : last.toFixed(3)}
                      </span>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </Panel>
      </div>

      {/* ----------------------------------------------------------- registry */}
      <div className="col-span-12 flex min-h-0 flex-col gap-3 xl:col-span-7">
        <Panel
          title="Model registry"
          flush
          className="max-h-[330px] shrink-0"
          actions={
            <div className="flex items-center gap-2">
              <span className="font-mono text-[10.5px] text-zinc-600">
                {modelsSource.data?.models_dir ?? "artifacts/models"} · {modelsSource.data?.count ?? models.length}
              </span>
              <Button variant="ghost" size="xs" onClick={() => modelsSource.reload()}>
                Refresh
              </Button>
            </div>
          }
        >
          <div className="min-h-0 flex-1 overflow-auto">
            {models.length === 0 ? (
              <EmptyState
                title="Checkpoint registry is empty"
                description="Train a checkpoint or start the FastAPI backend to list artifacts/models/*.zip with their .meta.json sidecars."
              />
            ) : (
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Checkpoint</th>
                    <th className="w-20">Algorithm</th>
                    <th className="w-24 text-right">Timesteps</th>
                    <th className="w-44">Universe</th>
                    <th className="w-40">Created</th>
                    <th className="w-24 text-right">Action</th>
                  </tr>
                </thead>
                <tbody>
                  {models.map((model) => (
                    <tr
                      key={model.name}
                      onClick={() => setSelectedName(model.name)}
                      className={cn(
                        "cursor-pointer",
                        selected?.name === model.name ? "bg-accent-soft/30" : "hover:bg-surface",
                      )}
                    >
                      <td>
                        <span className="flex min-w-0 items-center gap-1.5">
                          <span className="truncate font-mono text-[11px] text-zinc-200">{model.name}</span>
                          {model.metadata_available === false ? (
                            <Badge tone="warn" title={model.metadata_error ?? "metadata unavailable"}>
                              metadata unavailable
                            </Badge>
                          ) : null}
                        </span>
                      </td>
                      <td>
                        <Badge tone="accent">{model.algorithm || "—"}</Badge>
                      </td>
                      <td className="num text-right text-zinc-300">{formatInteger(model.timesteps)}</td>
                      <td className="num text-[10.5px] text-zinc-500">{model.universe.join(" · ") || "—"}</td>
                      <td className="text-[10.5px] text-zinc-500">{model.created_at ? formatDateTime(model.created_at) : "—"}</td>
                      <td className="text-right">
                        <Button
                          variant="secondary"
                          size="xs"
                          disabled={!baseStrategy || binding}
                          onClick={() => {
                            setSelectedName(model.name);
                            void bind(model);
                          }}
                          title={baseStrategy ? `Write a DRL allocator using ${model.name} into ${baseStrategy.id}` : "Select a base strategy"}
                        >
                          Bind
                        </Button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </Panel>

        <Panel
          title={selected ? `Inspector · ${selected.name}` : "Model inspector"}
          className="min-h-0 flex-1"
          actions={
            selected ? (
              <div className="flex items-center gap-2">
                <Badge tone={selected.metadata_available === false ? "warn" : "neutral"}>
                  {selected.metadata_available === false ? "metadata unavailable" : ".meta.json"}
                </Badge>
                <Button
                  variant="primary"
                  size="xs"
                  disabled={!baseStrategy || binding}
                  onClick={() => void bind(selected)}
                  title={baseStrategy ? `Bind ${selected.name} to ${baseStrategy.id}` : "Select a base strategy"}
                >
                  {binding ? "Binding…" : `Bind to ${baseStrategy?.id ?? "strategy"}`}
                </Button>
              </div>
            ) : null
          }
        >
          {selected ? (
            <div className="grid min-h-0 flex-1 grid-rows-[auto_minmax(0,1fr)] gap-2">
              <div>
                <div className="mb-1 flex items-center justify-between text-[10.5px] text-zinc-600">
                  <span>
                    {selected.reward_curve.length > 0
                      ? `reward curve · ${selected.reward_curve.length} points`
                      : "reward curve unavailable in registry listing"}
                  </span>
                  <span className="font-mono">
                    {selected.size_bytes ? `${(selected.size_bytes / 1_000_000).toFixed(2)} MB` : "size n/a"}
                    {selected.git_commit ? ` · ${selected.git_commit}` : ""}
                  </span>
                </div>
                {selected.reward_curve.length > 0 ? (
                  <OrdinalLineChart
                    series={[
                      {
                        label: "mean reward",
                        color: "#8B5CF6",
                        values: selected.reward_curve.map((point) => point.reward),
                      },
                    ]}
                    height={136}
                    formatValue={(value) => value.toFixed(3)}
                    xLabels={[
                      String(selected.reward_curve[0]?.step ?? 0),
                      String(selected.reward_curve[Math.floor(selected.reward_curve.length / 2)]?.step ?? 0),
                      String(selected.reward_curve[selected.reward_curve.length - 1]?.step ?? 0),
                    ]}
                    ariaLabel={`Reward curve for ${selected.name}`}
                  />
                ) : (
                  <div className="grid grid-cols-4 gap-2 rounded-md border border-hairline bg-canvas/60 px-2.5 py-2">
                    <InlineField label="Final mean">
                      <span className="font-mono text-[12px] text-zinc-200">
                        {selected.reward_summary?.final_mean_reward?.toFixed(3) ?? "—"}
                      </span>
                    </InlineField>
                    <InlineField label="Best mean">
                      <span className="font-mono text-[12px] text-emerald-400">
                        {selected.reward_summary?.best_mean_reward?.toFixed(3) ?? "—"}
                      </span>
                    </InlineField>
                    <InlineField label="Samples">
                      <span className="font-mono text-[12px] text-zinc-300">{selected.reward_summary?.samples ?? "—"}</span>
                    </InlineField>
                    <InlineField label="Window steps">
                      <span className="font-mono text-[12px] text-zinc-300">{selected.reward_summary?.window_steps ?? "—"}</span>
                    </InlineField>
                  </div>
                )}
              </div>
              <div className="flex min-h-0 flex-col">
                <div className="mb-1 flex items-center justify-between text-[10.5px] text-zinc-600">
                  <span>checkpoint metadata (.meta.json)</span>
                  {selected.meta_path ? <span className="truncate font-mono">{selected.meta_path}</span> : null}
                </div>
                {selected.metadata_available === false ? (
                  <div className="rounded-md border border-warn/30 bg-warn-soft/40 px-2.5 py-2 text-[10.5px] leading-4 text-amber-300">
                    Metadata could not be read for this checkpoint: {selected.metadata_error ?? "unknown error"}. The file is still
                    listed for honesty; binding is disabled until metadata is available.
                  </div>
                ) : null}
                <JsonPre value={drlModelMetaJson(selected)} className="h-full min-h-[120px]" />
              </div>
            </div>
          ) : (
            <EmptyState title="No model selected" description="Select a checkpoint from the registry to inspect its reward curve and metadata." />
          )}
        </Panel>
      </div>
    </div>
  );
}
