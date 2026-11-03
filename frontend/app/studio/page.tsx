import type { Metadata } from "next";

import { StudioView } from "@/components/studio/studio-view";

export const metadata: Metadata = {
  title: "Strategy Studio",
  description:
    "Compose FinRL-X futures strategies, validate the YAML pipeline contract, run walk-forward Optuna optimization and train DRL policies with the LLM risk overlay.",
};

export default function StudioPage() {
  return <StudioView />;
}
