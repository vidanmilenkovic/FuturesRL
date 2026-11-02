import type { Metadata } from "next";

import { PipelineView } from "@/components/pipeline/pipeline-view";

export const metadata: Metadata = {
  title: "Pipeline Inspector",
  description:
    "FinRL-X weight waterfall: S_t → w_base → w_timing → w_risk → target futures contracts N_i,t with per-instrument deltas and stage diagnostics.",
};

export default function PipelinePage() {
  return <PipelineView />;
}
