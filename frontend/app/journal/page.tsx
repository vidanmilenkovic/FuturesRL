import type { Metadata } from "next";

import { JournalView } from "@/components/journal/journal-view";

export const metadata: Metadata = {
  title: "Journal",
  description:
    "Local paper/live trade journal: every order, fill, commission and account snapshot written to CSV, with the benchmark equity curve.",
};

export default function JournalPage() {
  return <JournalView />;
}
