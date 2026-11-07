import type { Metadata } from "next";

import { RiskView } from "@/components/risk/risk-view";

export const metadata: Metadata = {
  title: "Risk",
  description:
    "LLM Risk-First structural constraint engine monitor: sentiment, confidence c_i,t, semantic risk S_i,t, circuit-breaker overrides and macro lockout countdowns.",
};

export default function RiskPage() {
  return <RiskView />;
}
