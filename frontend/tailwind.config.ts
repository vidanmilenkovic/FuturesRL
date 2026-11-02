import type { Config } from "tailwindcss";

/**
 * Dark "quant terminal" design tokens for FuturesRL-X.
 *
 * Colours are intentionally low-chroma surfaces with a small set of signal
 * accents (long/up = emerald, short/down = rose, warning = amber, risk = red,
 * system = cyan). Every view uses the same palette so cross-view comparison
 * (e.g. pipeline stage deltas vs. PnL) stays legible.
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
        terminal: {
          950: "#04070b",
          900: "#070c12",
          850: "#0a1018",
          800: "#0d1520",
          750: "#111b28",
          700: "#162231",
          600: "#1d2c3e",
          500: "#2a3d52",
          400: "#425a72",
          300: "#61798f",
          200: "#93a7b8",
          100: "#c3d1dc",
        },
        signal: {
          up: "#34d399",
          "up-dim": "#0f3d2f",
          down: "#fb7185",
          "down-dim": "#431b25",
          warn: "#fbbf24",
          "warn-dim": "#463414",
          critical: "#f43f5e",
          lock: "#f97316",
          info: "#22d3ee",
          "info-dim": "#0b3b45",
          violet: "#a78bfa",
        },
      },
      fontFamily: {
        sans: [
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "Roboto",
          "Helvetica Neue",
          "Arial",
          "sans-serif",
        ],
        mono: [
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "Monaco",
          "Cascadia Mono",
          "Consolas",
          "Liberation Mono",
          "monospace",
        ],
      },
      fontVariantNumeric: {
        tabular: "tabular-nums",
      },
      boxShadow: {
        panel: "0 1px 0 0 rgba(255,255,255,0.03) inset, 0 8px 24px -12px rgba(0,0,0,0.9)",
        glow: "0 0 0 1px rgba(34,211,238,0.35), 0 0 24px -4px rgba(34,211,238,0.35)",
        "glow-critical":
          "0 0 0 1px rgba(244,63,94,0.6), 0 0 32px -2px rgba(244,63,94,0.55)",
      },
      backgroundImage: {
        "grid-faint":
          "linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px)",
        "scanlines":
          "repeating-linear-gradient(0deg, rgba(255,255,255,0.018) 0px, rgba(255,255,255,0.018) 1px, transparent 1px, transparent 4px)",
      },
      backgroundSize: {
        grid: "28px 28px",
      },
      animation: {
        "pulse-critical": "pulse-critical 1.6s cubic-bezier(0.4,0,0.6,1) infinite",
        "marquee": "marquee 32s linear infinite",
        "flash": "flash 900ms ease-out 1",
      },
      keyframes: {
        "pulse-critical": {
          "0%, 100%": { opacity: "1", boxShadow: "0 0 0 0 rgba(244,63,94,0.45)" },
          "50%": { opacity: "0.86", boxShadow: "0 0 0 10px rgba(244,63,94,0)" },
        },
        marquee: {
          "0%": { transform: "translateX(0)" },
          "100%": { transform: "translateX(-50%)" },
        },
        flash: {
          "0%": { backgroundColor: "rgba(34,211,238,0.18)" },
          "100%": { backgroundColor: "transparent" },
        },
      },
    },
  },
  plugins: [],
};

export default config;
