# FuturesRL · Frontend (Next.js 15)

Dark "quant terminal" dashboard for the FuturesRL platform. Five views consume
the FastAPI backend over REST and `/ws/telemetry`; when the backend is not
reachable the app automatically falls back to deterministic demo fixtures and
shows a persistent **DEMO DATA** indicator, so it can be built and previewed
standalone.

## Stack

- Next.js 15 (App Router) + React 19 + TypeScript (strict)
- TailwindCSS 3 (custom `terminal` / `signal` design tokens)
- TradingView Lightweight Charts 5 (equity curve, rolling Sharpe)
- Hand-rolled accessible UI primitives (dialogs, segmented controls, gauges) —
  no shadcn CLI dependency
- ESLint 9 flat config (`next/core-web-vitals` + `next/typescript`)

## Views

| Route             | View                          | Highlights                                                                                                   |
| ----------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `/`               | Command Center                | Equity curve, NLV/KPI strip, margin gauge (60% warn / 80% critical), contracts table, **EMERGENCY FLATTEN ALL** |
| `/pipeline`       | FinRL-X Pipeline Inspector    | Waterfall `S_t → w_base → w_timing → w_risk → N_i,t`, per-instrument deltas, stage diagnostics                |
| `/risk`           | LLM Risk-First Monitor        | Sentiment / confidence `c_i,t` / semantic risk `S_i,t`, breaker actions, macro lockout countdown, thresholds  |
| `/term-structure` | Term Structure & Roll Calendar| Contango/backwardation curves per root, basis/roll-yield table, roll countdowns                              |
| `/backtests`      | Backtest Lab & Tear Sheets    | Run configs, metric comparison, rolling Sharpe, underwater drawdown, Rachev tail plot, roll-yield attribution |

## Local development

```bash
cd frontend
npm install
npm run dev            # http://localhost:3100
```

The dev server defaults to port **3100** so it never collides with the
production compose mapping (`3100`) or the API (`8000`).

The API origin is read from `NEXT_PUBLIC_API_BASE_URL` (default
`http://127.0.0.1:8000`). WebSocket telemetry is derived from it
(`ws://…/ws/telemetry`) unless `NEXT_PUBLIC_WS_URL` is set:

```bash
cp .env.local.example .env.local   # optional
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000 npm run dev
```

## Quality gates

```bash
npx tsc --noEmit       # strict type check
npm run lint           # ESLint (zero warnings)
npm run build          # production build (static prerender of all 5 routes)
npm run start          # serve the production build on :3100
```

## Demo mode behaviour

- Every REST helper in `lib/api.ts` returns `{ data, source, error }`. On a
  network error or timeout it resolves `source: "demo"` with the seeded
  factories in `lib/mock.ts` (identical numbers on every render).
- `TelemetryProvider` opens `/ws/telemetry`; on failure it starts a deterministic
  local telemetry simulator so PnL/margin/positions still animate.
- The shell header shows a **DEMO DATA** badge and a warning strip whenever the
  backend is offline. The emergency flatten dialog is fully exercisable in demo
  mode (it clearly reports that the flatten was simulated).

## Containerisation

Built by `docker/Dockerfile.frontend` (multi-stage `node:22-alpine`, Next.js
`output: "standalone"`). The API origin is baked in at build time:

```bash
docker build \
  --build-arg NEXT_PUBLIC_API_BASE_URL=http://localhost:8000 \
  -f ../docker/Dockerfile.frontend \
  -t futuresrl/frontend:1.0.0 .
```

Or run the whole stack with `docker compose -f ../docker/docker-compose.yml up --build`
→ dashboard on <http://localhost:3100>.
