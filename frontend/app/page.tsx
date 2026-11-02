import type { Metadata } from "next";

import { CommandCenterView } from "@/components/command-center/command-center-view";

export const metadata: Metadata = {
  title: "Command Center",
  description:
    "Live/paper futures account monitor: equity curve, NLV, margin utilisation gauge and the emergency flatten-all kill switch.",
};

export default function CommandCenterPage() {
  return <CommandCenterView />;
}
