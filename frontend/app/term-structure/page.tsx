import type { Metadata } from "next";

import { TermStructureView } from "@/components/term-structure/term-structure-view";

export const metadata: Metadata = {
  title: "Term Structure",
  description:
    "Futures term structure and roll calendar: contango/backwardation curves per root, annualised carry and roll countdowns.",
};

export default function TermStructurePage() {
  return <TermStructureView />;
}
