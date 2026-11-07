import type { Metadata } from "next";

import { BacktestsView } from "@/components/backtests/backtests-view";

export const metadata: Metadata = {
  title: "Backtests",
  description:
    "Walk-forward futures backtests and tear sheet comparison: rolling Sharpe, underwater drawdown, Rachev tail plots and roll-yield attribution.",
};

export default function BacktestsPage() {
  return <BacktestsView />;
}
