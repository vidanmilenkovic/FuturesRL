# FuturesRL-X

A modular, production-grade algorithmic **futures** trading platform. FuturesRL-X adapts the
FinRL-X weight-centric composable pipeline to leveraged, expiring futures contracts and ships
three execution modes behind one interface: **historical backtesting**, **IBKR paper trading**
and **IBKR live execution**.

```
w_t = R_t( T_t( A_t( S_t( X_{≤t} ) ) ) )        # the only interface contract
        │       │       │       └── S_t  universe selection  (liquidity, OI, regime)
        │       │       └────────── A_t  portfolio allocation (risk parity, vol target, DRL)
        │       └────────────────── T_t  timing & term structure (TSMOM, KAMA, carry)
        └────────────────────────── R_t  risk overlays (LLM Risk-First, quant vol targeting)
```

The same approved target weight vector `w_t` flows into **both** the backtest engine and the
IBKR execution engine through the deterministic `ContractTranslator`, which converts signed
notional weights into discrete futures contracts (with automatic standard→micro routing and a
hard initial-margin utilisation ceiling `η_max = 0.60`). That property is enforced by the parity
test suite in `tests/parity/`.

---

## Table of contents

1. [Feature highlights](#feature-highlights)
2. [Repository layout](#repository-layout)
3. [Quickstart](#quickstart)
4. [Configuration](#configuration)
5. [Running the platform](#running-the-platform)
6. [Quality gates](#quality-gates)
7. [Architecture overview](#architecture-overview)
8. [Testing philosophy](#testing-philosophy)
9. [Development workflow & release history](#development-workflow--release-history)
10. [Risk & safety notes](#risk--safety-notes)

---

## Feature highlights

**Data layer** (`src/data/`)
- Databento historical ingestion (`ohlcv-1m/1h/1d`, `tbbo`, `mbp-1`, `mbp-10`, `definition`) with
  lazy SDK import and an offline deterministic source for reproducible research.
- Immutable canonical Polars/Parquet schema keyed by
  `(timestamp_utc, instrument_id, root_symbol, expiration_date)`.
- Partitioned Parquet cache + DuckDB SQL query layer.
- Continuous futures roll engine: volume/OI crossover, calendar N-days before FND/LTD, and
  Databento native symbology cross-checks, with **Panama additive**, **proportional ratio** and
  **unadjusted + explicit carry/term-slope** adjustment modes; strictly zero look-ahead.
- Feature processor: FinRL-X technicals (MACD, RSI, CCI, ADX, Bollinger, ATR, Garman-Klass),
  term-structure slope, annualised roll yield, volume/OI ratio, order-flow imbalance, and the
  Mahalanobis turbulence index.

**Strategy pipeline** (`src/strategies/`)
- Strict Protocols for `UniverseSelector`, `PortfolioAllocator`, `TimingOverlay`, `RiskOverlay`.
- Selectors: liquidity, slow/fast adaptive regime, composite.
- Allocators: equal weight, equal-risk-contribution risk parity, mean-variance, volatility
  target, and DRL (PPO / CPPO / SAC / TD3 / A2C via Stable-Baselines3).
- Timing overlays: TSMOM, KAMA, carry/term-structure filter, composite.
- Gymnasium `FuturesTradingEnv` with futures margin dynamics, transaction costs and the
  FinRL Risk-First semantic-risk reward penalty.

**Risk engine** (`src/risk/`)
- Tier A — **LLM Risk-First engine**: structured Pydantic assessments, multi-sample confidence &
  variance anti-hallucination filter, reward shaping, and a **deterministic post-hoc circuit
  breaker** (`S ≥ κ ∧ c ≥ τ ⇒ w = 0`, linear de-risking between `θ_safe` and `κ`) plus Tier-1
  macro event lockout windows.
- Tier B — quantitative overlay: EWMA covariance volatility targeting, gross/single/sector caps,
  high-water-mark trailing stops, daily max-loss kill switch, drawdown limits.
- `ContractTranslator`: notional → discrete contracts, micro routing, margin feasibility
  projection with uniform scale-down.

**Backtesting & analytics** (`src/backtest/`, `src/analytics/`)
- Vectorized Polars mode for parameter sweeps and an event-driven bar mode reusing the exact
  `BrokerAdapter` protocol used live.
- Daily mark-to-market settlement, initial/maintenance margin tracking, forced-liquidation
  simulation, two-legged roll execution costs, tick-aware slippage and tiered IBKR/CME fee
  schedules, and purged + embargoed walk-forward validation.
- Institutional metrics: CAGR, volatility, Sharpe, Sortino, Calmar, Omega, max drawdown and
  duration, historical & parametric VaR 95/99, CVaR, Rachev ratio, RoM, margin efficiency,
  implementation shortfall, and spot-vs-roll-yield attribution.

**Execution** (`src/execution/`)
- Async IBKR connection manager on `ib_async` (TWS/Gateway, paper & live ports), heartbeats,
  exponential-backoff reconnects, IBKR nightly reset-window handling, `whatIfOrder` margin checks.
- OMS with a mandatory pre-trade hard gate (fat-finger limits, duplicate/rapid-fire rate limiter,
  real-time margin impact) and an emergency `flatten_all()` kill switch.
- Automated contract rolls through native IBKR `BAG` calendar-spread orders, with a two-legged
  fallback.
- 30-second internal-vs-broker state reconciliation that halts automated rebalancing on drift.

**Telemetry & dashboard** (`src/api/`, `frontend/`)
- FastAPI REST + WebSocket (`/ws/telemetry`) backend with an in-memory backtest store and
  dependency-injected services (offline-capable, no IBKR required to run the UI).
- Next.js 15 + TypeScript + TailwindCSS + TradingView Lightweight Charts dashboard with five
  views: Command Center (margin gauge + EMERGENCY FLATTEN ALL), FinRL-X Pipeline Inspector,
  LLM Risk-First Monitor, Term Structure & Roll Calendar, Backtest Lab & Tear Sheet Compare.
- Docker Compose stack: TimescaleDB, Redis, headless IB Gateway, backend, frontend.

---

## Repository layout

```text
futures-app/
├── AGENTS.md / Instructions.md      # architecture specification & phased roadmap (source of truth)
├── configs/
│   ├── settings.py                  # Pydantic BaseSettings + .env loader
│   ├── instruments.py(.yaml)        # futures master: multipliers, ticks, margins, roll/micro rules
│   ├── strategy.py                  # typed declarative strategy config models
│   └── strategies/                  # YAML pipelines (tsmom_vol_target, drl_ppo_llm_risk)
├── src/
│   ├── core/                        # immutable contracts, Protocols, async event bus
│   ├── data/                        # Databento, storage, continuous roller, features, IBKR feed
│   ├── strategies/                  # pipeline composer, selectors, allocators, timing, envs
│   ├── risk/                        # LLM Risk-First engine, quant overlay, contract translator
│   ├── backtest/                    # engine, margin/cost simulators, walk-forward, simulated broker
│   ├── execution/                   # IBKR client, OMS pre-trade gate, BAG roller, reconciler
│   ├── analytics/                   # metrics, tearsheets
│   └── api/                         # FastAPI REST routes + WebSocket telemetry
├── frontend/                        # Next.js 15 command center UI
├── docker/                          # Dockerfiles + docker-compose stack
├── tests/
│   ├── unit/                        # offline deterministic unit tests
│   ├── integration/                 # cross-module end-to-end tests
│   └── parity/                      # backtest vs live ContractTarget parity proofs
├── scripts/quality_gate.sh          # lint + format + strict mypy + pytest (+ coverage gate)
├── pyproject.toml
└── README.md
```

---

## Quickstart

### 1. Python environment

```bash
git clone <repo-url> futures-app && cd futures-app
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e ".[dev]"            # core + test/lint/type tooling
# optional extras:
.venv/bin/pip install -e ".[live]"           # ib_async (IBKR paper/live)
.venv/bin/pip install -e ".[databento]"      # Databento historical ingestion
.venv/bin/pip install -e ".[drl]"            # gymnasium + stable-baselines3 (torch)
cp .env.example .env                         # then edit credentials
```

### 2. Verify the environment

```bash
./scripts/quality_gate.sh          # ruff + ruff format --check + mypy strict + pytest
./scripts/quality_gate.sh --cov    # additionally enforce ≥85% coverage on core/risk/translator
```

### 3. Explore offline (no credentials required)

```bash
# deterministic offline fixture source + declarative strategy config
.venv/bin/python -m src.backtest.engine \
  --strategy configs/strategies/tsmom_vol_target.yaml \
  --offline --output artifacts/tsmom_tearsheet.json
```

---

## Configuration

Configuration is layered: environment variables (`FUTURESRLX_*`), an optional `.env` file, then
the committed `configs/*.yaml` masters. Secrets use `pydantic.SecretStr` and are redacted from
logs and telemetry (`Settings.model_dump_redacted()`).

| Variable | Default | Meaning |
| --- | --- | --- |
| `FUTURESRLX_ENVIRONMENT` | `development` | `development` / `paper` / `live` / `backtest` |
| `FUTURESRLX_DATABENTO_API_KEY` | _(unset)_ | Databento historical ingestion key |
| `FUTURESRLX_DATA_DIR` / `PARQUET_DIR` / `DUCKDB_PATH` | `./data/...` | market-data cache |
| `FUTURESRLX_IBKR_HOST/PORT/CLIENT_ID` | `127.0.0.1:7497:42` | TWS/Gateway (paper 7497/4002, live 7496/4001) |
| `FUTURESRLX_IBKR_PAPER` | `true` | must be `false` with a live port |
| `FUTURESRLX_RISK__MAX_MARGIN_UTILIZATION` | `0.60` | hard initial-margin ceiling `η_max` |
| `FUTURESRLX_RISK__TARGET_ANNUAL_VOLATILITY` | `0.12` | ex-ante volatility target |
| `FUTURESRLX_RISK__LLM_CONFIDENCE_THRESHOLD` | `0.70` | `τ` for the anti-hallucination filter |
| `FUTURESRLX_RISK__LLM_CRITICAL_SCORE` | `4.5` | `κ` circuit-breaker threshold |
| `FUTURESRLX_RISK__DAILY_MAX_LOSS_PCT` | `0.03` | intraday kill switch |
| `FUTURESRLX_API_HOST` / `PORT` | `0.0.0.0:8000` | FastAPI telemetry backend |

The futures instrument master (`configs/instruments.yaml`) is the single source of truth for
multipliers, tick sizes, margin estimates, roll rules and standard↔micro pairs (ES↔MES, NQ↔MNQ,
CL↔MCL, GC↔MGC) for equity index, rates, energy, metals and FX futures. Margins are conservative
estimates; live orders are always confirmed against IBKR `whatIfOrder` before transmission.

Declarative strategy pipelines live in `configs/strategies/*.yaml` and describe the pipeline
stages by name + parameters — no executable code, so runs are reproducible and sweepable.

---

## Running the platform

### Backtests

```bash
# declarative config, offline deterministic data
.venv/bin/python -m src.backtest.engine --strategy configs/strategies/tsmom_vol_target.yaml --offline

# walk-forward validation
.venv/bin/python -m src.backtest.engine --strategy configs/strategies/drl_ppo_llm_risk.yaml \
  --walk-forward --output artifacts/wf_tearsheet.json
```

### Paper / live trading (IBKR)

```bash
# TWS or headless IB Gateway must be running and API-enabled
FUTURESRLX_ENVIRONMENT=paper FUTURESRLX_IBKR_PORT=7497 \
  .venv/bin/python -m src.execution.runner   # or wire via the API/dashboard
```

The OMS always runs the pre-trade gate (fat finger → `whatIfOrder` margin → rate limiter) before
transmitting; the dashboard's **EMERGENCY FLATTEN ALL** calls `flatten_all()` (cancel all, close
all, lock).

### API + dashboard

```bash
# backend (offline-capable; demo telemetry when no live engine is attached)
.venv/bin/python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# frontend
cd frontend && npm install && npm run dev     # http://localhost:3000
```

### Docker stack

```bash
docker compose -f docker/docker-compose.yml up --build
# TimescaleDB + Redis + IB Gateway (paper) + FastAPI backend + Next.js frontend
```

---

## Quality gates

Every commit must pass (see `scripts/quality_gate.sh`):

```bash
.venv/bin/ruff check .                 # zero lint errors
.venv/bin/ruff format --check .        # formatting
.venv/bin/python -m mypy src configs   # strict static typing, no untyped Any in core contracts
.venv/bin/python -m pytest tests/      # ≥85% coverage on core, risk and translator modules
```

All unit tests are **offline and deterministic**: they never call paid Databento endpoints or a
live IBKR session. Networked/adapter code is exercised through injected fakes.

---

## Architecture overview

### The immutable contract layer

`src/core/types.py` is the narrow waist of the platform. Every module exchanges only frozen
dataclasses (`MarketState`, `WeightVector`, `ContractTarget`, `ContractTargetSet`, `OrderEvent`,
`Fill`, `RiskAssessment`, `PipelineResult`, ...) with validated numeric invariants (naive
timestamps rejected, NaN weights rejected, crossed quotes rejected, negative margins rejected).
Where mappings/arrays are embedded they are defensively copied and made read-only.

`src/core/interfaces.py` defines structural `typing.Protocol`s for the pipeline stages, data
sources, broker adapters and risk providers. Backtest and live execution implement the *same*
protocol, which is what makes deployment parity testable.

`src/core/event_bus.py` provides the async pub/sub bus used identically by the live engine and
the event-driven backtester (topic fan-out, ordered delivery, handler-failure isolation).

### Futures contract translation

For every root `i`:

```
TargetNotional_i = w_i · E_t                     (E_t = current NLV, w_i signed)
N_i^raw          = TargetNotional_i / (P_i · M_i)  (P_i mark, M_i multiplier)
N_i              = round(N_i^raw)                 (floor / nearest + buffer / ceil)
micro routing:   fractional standard exposure → MES/MNQ/MCL/MGC
IM_proj          = Σ |N_i| · IM_i
if IM_proj / E_t > η_max: uniformly scale down and re-round
```

The result is a `ContractTargetSet`. The OMS diffs it against current positions to produce
`OrderEvent`s; the backtester's `SimulatedBroker` consumes the same set. `tests/parity/` asserts
both engines produce identical targets and deltas for identical market states.

### LLM Risk-First engine (Tier A)

1. **Structured output** — per root: `sentiment_score ∈ [1,5]`, `semantic_risk_score ∈ [1,5]`,
   `confidence ∈ [0,1]`, `reasoning_summary`.
2. **Module 1 — anti-hallucination** — multi-sample mean/confidence; if `c < τ` or sample variance
   exceeds `σ²_max`, the signal is neutralised to `S = θ_safe`.
3. **Module 2 — reward shaping** — `R_shaped = R_netPnL − λ Σ |w_i| · 1[c_i ≥ τ] · max(0, S_i − θ_safe)`.
4. **Module 3 — deterministic projection** — `S_i ≥ κ ∧ c_i ≥ τ ⇒ w_i = 0`;
   `θ_safe < S_i < κ ⇒ w_i ← w_i · (κ − S_i)/(κ − θ_safe)`, plus macro-event lockout windows.
   A neural policy can never override the hard projection: risk is enforced post-hoc.

---

## Testing philosophy

- **Offline & deterministic** — seeded synthetic fixtures in `tests/fixtures/market.py`; adapters
  are injected. No network, no paid APIs, no wall-clock races.
- **Property-based** — `hypothesis` proves the translator never exceeds the margin ceiling across
  arbitrary equity/price/weight shocks.
- **Accounting identity** — the backtest integration test asserts
  `Ending Equity = Starting Equity + Price PnL + Roll PnL − Commissions − Slippage`.
- **Anti-look-ahead** — roll/feature/walk-forward tests mutate future data and assert nothing
  changes in the past.
- **Parity** — identical `MarketState` ⇒ identical `ContractTarget`s (and OMS deltas) from the
  backtest and IBKR paths.
- **Ablation** — LLM hallucination fixtures (low confidence) are ignored, while high-confidence
  critical alerts deterministically zero exposure.

---

## Development workflow & release history

Conventional Commits, atomic commits, one feature branch per phase, a release tag at each phase
boundary:

| Phase | Branch | Release | Scope |
| --- | --- | --- | --- |
| 1 | `feat/phase-1-core-data` | `v0.1.0-data` | core contracts, config, data layer, continuous roller, features |
| 2 | `feat/phase-2-finrlx-pipeline` | `v0.2.0-pipeline` | pipeline, selectors/allocators/timing, DRL env, contract translator |
| 3 | `feat/phase-3-risk-engine` | `v0.3.0-risk` | LLM Risk-First engine, quant risk overlay |
| 4 | `feat/phase-4-backtest-analytics` | `v0.4.0-backtest` | backtester, margin/cost simulators, walk-forward, metrics, tearsheets |
| 5 | `feat/phase-5-execution` | `v0.5.0-execution` | IBKR async client, OMS pre-trade gate, BAG roller, reconciler |
| 6 | `feat/phase-6-api-dashboard` | `v1.0.0` | FastAPI REST/WS telemetry, Next.js dashboard, Docker stack |

---

## Risk & safety notes

- This software is research infrastructure, **not investment advice**. Futures are leveraged;
  losses can exceed the initial margin. Test on paper for a meaningful period before live use.
- Live trading requires the explicit `FUTURESRLX_ENVIRONMENT=live` opt-in, a live IBKR port and
  `ibkr_paper=false`; the settings validator refuses inconsistent configurations.
- The pre-trade gate is mandatory and cannot be bypassed by strategy code. The emergency kill
  switch (`flatten_all`) cancels all orders, market-closes positions and locks automated
  rebalancing until an explicit reset.
- LLM outputs are untrusted data: they are schema-validated, confidence-filtered, and only ever
  applied through bounded deterministic projections.
