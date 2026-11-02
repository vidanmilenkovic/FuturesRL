import type { Metadata, Viewport } from "next";

import { TelemetryProvider } from "@/components/providers/telemetry-provider";
import { AppShell } from "@/components/shell/app-shell";

import "./globals.css";

export const metadata: Metadata = {
  title: {
    default: "FuturesRL-X · Quant Command Center",
    template: "%s · FuturesRL-X",
  },
  description:
    "Institutional futures command center: FinRL-X weight pipeline, LLM Risk-First monitor, term structure and walk-forward backtest tear sheets.",
  applicationName: "FuturesRL-X",
  robots: { index: false, follow: false },
  icons: {
    icon: "/favicon.svg",
  },
};

export const viewport: Viewport = {
  themeColor: "#04070b",
  colorScheme: "dark",
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-terminal-950 bg-scanlines text-terminal-100">
        <TelemetryProvider>
          <AppShell>{children}</AppShell>
        </TelemetryProvider>
      </body>
    </html>
  );
}
