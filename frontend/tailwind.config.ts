import type { Config } from "tailwindcss";

/**
 * FuturesRL workspace design tokens.
 *
 * The palette is deliberately narrow: one canvas, one panel surface, one hover
 * surface, two hairline weights and a handful of signal colours. Accent blue is
 * used for interaction/selection only; green/red are reserved for PnL and
 * directional exposure; amber for warnings and degraded state.
 *
 * Typography contract: Inter (`font-sans`) for every piece of UI prose, labels,
 * buttons and table headers. JetBrains Mono (`font-mono`, tabular numerals) is
 * reserved for numbers: prices, PnL, percentages, contract counts, weights and
 * timestamps.
 */
const config: Config = {
  darkMode: "class",
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        canvas: "#09090B",
        panel: "#121215",
        surface: "#18181C",
        hairline: "#1E1E22",
        edge: "#222226",
        muted: "#A1A1AA",
        accent: {
          DEFAULT: "#3B82F6",
          hover: "#2563EB",
          soft: "rgba(59, 130, 246, 0.12)",
        },
        gain: {
          DEFAULT: "#10B981",
          soft: "rgba(16, 185, 129, 0.12)",
        },
        loss: {
          DEFAULT: "#F43F5E",
          soft: "rgba(244, 63, 94, 0.12)",
        },
        warn: {
          DEFAULT: "#F59E0B",
          soft: "rgba(245, 158, 11, 0.12)",
        },
      },
      fontFamily: {
        sans: [
          "var(--font-inter)",
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "sans-serif",
        ],
        mono: [
          "var(--font-jetbrains)",
          "JetBrains Mono",
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "monospace",
        ],
      },
      borderRadius: {
        md: "6px",
      },
      boxShadow: {
        panel: "0 1px 2px rgba(0, 0, 0, 0.35)",
        popover: "0 8px 24px rgba(0, 0, 0, 0.55), 0 0 0 1px rgba(255, 255, 255, 0.04)",
      },
      transitionDuration: {
        DEFAULT: "120ms",
      },
    },
  },
  plugins: [],
};

export default config;
