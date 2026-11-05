import type { Metadata, Viewport } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";

import { AccountProvider } from "@/components/providers/account-provider";
import { TelemetryProvider } from "@/components/providers/telemetry-provider";
import { AppShell } from "@/components/shell/app-shell";

import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains",
  display: "swap",
});

export const metadata: Metadata = {
  title: {
    default: "FuturesRL · Institutional Futures Workspace",
    template: "%s · FuturesRL",
  },
  description:
    "Institutional futures workstation: FinRL-X weight pipeline, LLM Risk-First monitor, term structure, roll calendar and walk-forward backtest tear sheets.",
  applicationName: "FuturesRL",
  robots: { index: false, follow: false },
  icons: { icon: "/favicon.svg" },
};

export const viewport: Viewport = {
  themeColor: "#09090B",
  colorScheme: "dark",
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.variable} ${jetbrainsMono.variable} h-full bg-canvas font-sans`}>
        <TelemetryProvider>
          <AccountProvider>
            <AppShell>{children}</AppShell>
          </AccountProvider>
        </TelemetryProvider>
      </body>
    </html>
  );
}
