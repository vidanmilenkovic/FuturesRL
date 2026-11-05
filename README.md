# FuturesRL-X

FuturesRL-X is a modular, production-oriented algorithmic **futures** trading platform. It adapts
the FinRL-X weight-centric composable pipeline to leveraged, expiring futures contracts and ships
one interface for **historical backtesting**, **IBKR paper trading** and **IBKR live execution**.

The pipeline is the contract:

```
w_t = R_t( T_t( A_t( S_t( X_{≤t} ) ) ) )
        │       │       │       └── S_t  universe selection   (liquidity, open interest, regime)
        │       │       └────────── A_t  portfolio allocation (risk parity, vol target, DRL)
        │       └────────────────── T_t  timing & term structure (TSMOM, KAMA, carry, seasonality)
        └────────────────────────── R_t  risk overlays        (LLM Risk-First, quant risk)
```

The same approved target weight vector `w_t` flows into **both** the backtest engine and the IBKR
execution engine through the deterministic `ContractTranslator`, which converts signed notional
weights into discrete futures contracts with automatic standard→micro routing and a hard initial-
margin utilisation ceiling (`η_max = 0.60` by default). That property is exercised by the parity
suite in `tests/parity/`.

## Documentation

| Document | Contents |
| --- | --- |
| [docs/USER_GUIDE.md](docs/USER_GUIDE.md) | Installation, configuration reference, data and IBKR setup, dashboard, backtests, live/paper runner, risk operations, troubleshooting |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design, module responsibilities, invariants, extension points |
| [CHANGELOG.md](CHANGELOG.md) | Versioned release notes (SemVer) |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development setup, quality gate, pull-request expectations |
| [LICENSE](LICENSE) | MIT license |

## Feature highlights

**Data layer** (`src/data/`)
- Databento historical ingestion (`ohlcv-1m/1h/1d`, `tbbo`, `mbp-1`, `mbp-10`, `definition`) with a
  lazy SDK import and a deterministic offline source for reproducible research.
- Immutable canonical Polars/Parquet schema keyed by
  `(timestamp_utc, instrument_id, root_symbol, expiration_date)` with a partitioned Parquet cache
  and a DuckDB query layer.
- Continuous futures roll engine: volume/OI crossover, calendar N-days before FND/LTD and Databento
  native symbology cross-checks, with Panama additive, proportional ratio and unadjusted + explicit
  carry/term-slope adjustment modes; strictly zero look-ahead.
- Feature processor: MACD, RSI, CCI, ADX, Bollinger, ATR, Garman-Klass, term-structure slope,
  annualised roll yield, volume/OI ratio, order-flow imbalance and the Mahalanobis turbulence index.
- Eight canonical timeframes (`1m`, `5m`, `15m`, `30m`, `1h`, `4h`, `1d`, `1w`) with a look-ahead-safe
  Polars resampler (right-boundary labels, UTC windows, volume-weighted VWAP) and automatic cache
  resampling from the finest available base timeframe.
- Cache inventory and budgeted backfills: IBKR historical data first, Databento pay-as-you-go with
  an explicit cost estimate and hard per-job budget guard; provenance recorded per root.

**Strategy pipeline** (`src/strategies/`)
- Strict Protocols for `UniverseSelector`, `PortfolioAllocator`, `TimingOverlay` and `RiskOverlay`.
- Selectors: liquidity, slow/fast adaptive regime, composite, with optional explicit root allow-lists.
- Allocators: equal weight, equal-risk-contribution risk parity, mean-variance, volatility target and
  DRL (PPO / CPPO / SAC / TD3 / A2C via Stable-Baselines3).
- Timing overlays: TSMOM, KAMA, carry/term-structure filter, mean reversion, seasonality, composite.
- Gymnasium `FuturesTradingEnv` with futures margin dynamics, transaction costs and the FinRL
  Risk-First semantic-risk reward penalty.
- Timeframe-aware quant math: `MarketState.timeframe` / `BacktestSpec.timeframe` drive
  `bars_per_year` annualisation (daily remains exactly 252 trading bars per year).

**Risk engine** (`src/risk/`)
- LLM Risk-First engine: structured Pydantic assessments, multi-sample confidence and variance
  anti-hallucination filter, reward shaping, a deterministic post-hoc circuit breaker
  (`S ≥ κ ∧ c ≥ τ ⇒ w = 0`, linear de-risking between `θ_safe` and `κ`) and macro-event lockouts.
- Quantitative overlay: EWMA covariance volatility targeting, gross/single-instrument/sector caps,
  high-water-mark trailing stops, UTC daily max-loss kill switch and drawdown limits.
- `ContractTranslator`: notional → discrete contracts, micro routing, margin feasibility projection
  and FX-normalised sizing for non-USD accounts.

**Backtesting & analytics** (`src/backtest/`, `src/analytics/`)
- Vectorized Polars mode for parameter sweeps and an event-driven bar mode that reuses the exact
  `BrokerAdapter` protocol used live.
- Daily mark-to-market settlement (UTC calendar days for intraday bars), initial/maintenance margin
  tracking, forced-liquidation simulation, roll execution costs, tick-aware slippage and tiered
  IBKR/CME fee schedules, plus purged and embargoed walk-forward validation.
- Institutional metrics: CAGR, volatility, Sharpe, Sortino, Calmar, Omega, max drawdown and
  duration, historical/parametric VaR 95/99, CVaR, Rachev ratio, return-on-margin, margin
  efficiency, implementation shortfall and spot-vs-roll attribution.

**Execution** (`src/execution/`)
- Async IBKR connection manager on `ib_async` (TWS/Gateway, paper and live ports), heartbeat,
  exponential-backoff reconnects, IBKR nightly reset-window handling and `whatIfOrder` margin checks.
- OMS with a mandatory pre-trade hard gate (fat-finger limits, duplicate/rapid-fire rate limiter,
  real-time margin impact) and an emergency `flatten_all()` kill switch.
- Automated contract rolls through native IBKR `BAG` calendar spreads with a two-legged fallback.
- State reconciliation that includes in-flight working-order deltas and halts automated rebalancing
  on unexplained drift.
- Cold-start hydration from the Parquet cache before the first live bar, durable risk/kill-switch
  state across restarts, and one-shot paper rebalances that never open tick subscriptions.

**Telemetry, API & dashboard** (`src/api/`, `frontend/`)
- FastAPI REST + WebSocket (`/ws/telemetry`) backend with dependency-injected, offline-capable
  services: positions, pipeline waterfall, risk state/thresholds, term structure, strategies,
  backtests, optimizer, DRL trainer and market-data cache endpoints.
- Next.js 15 + TypeScript + TailwindCSS + TradingView Lightweight Charts dashboard: Command Center,
  Pipeline & Weights, Risk & LLM Monitor, Term Structure & Roll Calendar, Backtest Lab and a
  Strategy Studio (manual builder, walk-forward Optuna optimizer, DRL trainer).
- Docker Compose stack: TimescaleDB, Redis, headless IB Gateway, FastAPI backend and Next.js
  frontend.

## Repository layout

```text
futuresrl-x/
├── configs/
│   ├── settings.py                  # Pydantic BaseSettings + .env loader
│   ├── instruments.py(.yaml)        # futures master: multipliers, ticks, margins, roll/micro rules
│   ├── strategy.py                  # typed declarative strategy config models
│   ├── macro_calendar.yaml          # Tier-1 macro event lockout calendar (editable template)
│   └── strategies/                  # YAML pipelines (tsmom_vol_target, drl_ppo_llm_risk)
├── src/
│   ├── core/                        # immutable contracts, Protocols, async event bus
│   ├── data/                        # vendor adapters, cache, roll engine, features, resampling
│   ├── strategies/                  # pipeline composer, selectors, allocators, timing, envs
│   ├── risk/                        # LLM Risk-First engine, quant overlay, contract translator
│   ├── backtest/                    # engine, margin/cost simulators, walk-forward, simulated broker
│   ├── execution/                   # IBKR client, OMS pre-trade gate, BAG roller, reconciler
│   ├── analytics/                   # metrics, tearsheets
│   └── api/                         # FastAPI REST routes + WebSocket telemetry
├── frontend/                        # Next.js 15 dashboard
├── docker/                          # Dockerfiles + compose stack
├── docs/                            # user guide, architecture, screenshots
├── tests/
│   ├── unit/                        # offline deterministic unit tests
│   ├── integration/                 # cross-module end-to-end tests
│   └── parity/                      # backtest vs live ContractTarget parity proofs
├── scripts/quality_gate.sh          # lint + format + strict mypy + pytest (+ coverage gate)
├── CHANGELOG.md                     # versioned release notes
├── CONTRIBUTING.md
├── LICENSE                          # MIT
├── Makefile                         # developer entry points
├── pyproject.toml
└── README.md
```

## Quickstart

### 1. Python environment

```bash
git clone <repo-url> futuresrl-x && cd futuresrl-x
python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e ".[all]"             # every optional extra
# or choose extras individually:
#   .venv/bin/pip install -e ".[dev]"          core + test/lint/type tooling
#   .venv/bin/pip install -e ".[live]"         ib_async (IBKR paper/live)
#   .venv/bin/pip install -e ".[databento]"    Databento historical ingestion
#   .venv/bin/pip install -e ".[drl]"          gymnasium + stable-baselines3 (torch)
#   .venv/bin/pip install -e ".[optimize]"     Optuna
cp .env.example .env                           # then edit credentials
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

## Configuration

Configuration is layered: environment variables (`FUTURESRLX_*`), an optional `.env` file, then the
committed `configs/*.yaml` masters. Secrets use `pydantic.SecretStr` and are redacted from logs and
telemetry (`Settings.model_dump_redacted()`). The full reference is in
[docs/USER_GUIDE.md](docs/USER_GUIDE.md#5-configuration); the most common variables are:

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
| `FUTURESRLX_RISK__DAILY_MAX_LOSS_PCT` | `0.03` | UTC daily kill switch |
| `FUTURESRLX_API_HOST` / `PORT` | `0.0.0.0:8000` | FastAPI telemetry backend |
| `FUTURESRLX_ACCOUNT_CURRENCY` | `USD` | IBKR account base currency (NLV denomination) |
| `FUTURESRLX_FX_RATES` | `{}` | JSON map `{"EUR": 1.10}` = USD per 1 unit |
| `FUTURESRLX_RISK_STATE_STORE_URI` | `""` | durable risk state: `""`/`memory:`/file path/`duckdb:`/`redis:` |
| `FUTURESRLX_HYDRATION_LOOKBACK_DAYS` | `400` | cold-start history window before live trading |

The futures instrument master (`configs/instruments.yaml`) is the single source of truth for the
45-root universe: multipliers, tick sizes, margin estimates, roll rules, sectors and standard↔micro
pairs (ES↔MES, NQ↔MNQ, CL↔MCL, GC↔MGC, …). Margins are conservative estimates; live orders are
always confirmed against IBKR `whatIfOrder` before transmission. Declarative strategy pipelines live
in `configs/strategies/*.yaml` and describe stages by name and parameters — no executable code, so
runs are reproducible and sweepable.

`configs/macro_calendar.yaml` is the editable Tier-1 macro release calendar used by the LLM
Risk-First lockout windows (FOMC/CPI/NFP/EIA/WASDE/OPEC, UTC timestamps). LLM adapters support
OpenAI, Anthropic, DeepSeek, Gemini and local vLLM endpoints; they are only used when explicitly
configured, and every signal passes through the confidence/variance filter and the deterministic
circuit-breaker projection before it can affect weights.

## Running the platform

### Backtests

```bash
# declarative config, offline deterministic data
.venv/bin/python -m src.backtest.engine --strategy configs/strategies/tsmom_vol_target.yaml --offline

# purged + embargoed walk-forward validation (per-fold reports + aggregate means)
.venv/bin/python -m src.backtest.engine --strategy configs/strategies/tsmom_vol_target.yaml \
  --offline --walk-forward --wf-splits 5 --purge-bars 5 --embargo-bars 5 \
  --output artifacts/wf_tsmom.json --html artifacts/wf_tsmom.html
```

The walk-forward report contains per-fold train/test ranges and metrics, aggregate mean/std
statistics and a compounded out-of-sample equity curve. See the
[user guide](docs/USER_GUIDE.md#8-running-backtests) for all flags.

### Paper / live trading (IBKR)

```bash
# deterministic end-to-end dry run: no IBKR, no Databento key, no network
.venv/bin/python -m src.execution.runner --offline --max-states 30 \
  --strategy configs/strategies/tsmom_vol_target.yaml

# TWS or headless IB Gateway must be running and API-enabled
FUTURESRLX_ENVIRONMENT=paper FUTURESRLX_IBKR_PORT=7497 \
  .venv/bin/python -m src.execution.runner --backfill-source auto
```

Hydration source resolution: `auto` = Databento (API key) → IBKR historical (paper/live) → offline
(only when explicitly allowed). If no source is available the CLI exits with an actionable message
(never a raw traceback) explaining `--offline`, IBKR TWS setup and `FUTURESRLX_DATABENTO_API_KEY`.
See the [user guide](docs/USER_GUIDE.md#9-running-live--paper-trading).

The OMS always runs the pre-trade gate (fat finger → `whatIfOrder` margin → rate limiter) before
transmitting; the dashboard's **EMERGENCY FLATTEN ALL** calls `flatten_all()` (cancel all, close
all, lock).

### API + dashboard

```bash
# backend (offline-capable; deterministic fixtures when no live engine is attached)
.venv/bin/python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# frontend
cd frontend && npm install && npm run dev     # http://localhost:3100
```

REST surface: `/health`, `/api/positions`, `/api/pipeline/latest`, `/api/risk/state`,
`PUT /api/risk/thresholds`, `POST /api/risk/flatten-all`, `/api/term-structure`,
`/api/strategies[/{id}]`, `/api/backtests[ POST, /{id}, /{id}/tearsheet]`, `/api/telemetry/status`,
`/api/orders/working`, `/api/fills/recent`, `/api/runner/status`, `POST /api/runner/step`,
`/api/strategies/schema/catalog`, `POST/PUT /api/strategies[ /{id}]`,
`/api/optimize[ POST, /{id}, /{id}/apply]`, `/api/drl/models`, `POST /api/drl/train`,
`/api/data/inventory`, `POST /api/data/estimate-cost`, `POST /api/data/ingest`, plus
`WS /ws/telemetry`. The backend runs **offline by default** (`/health.offline_mode=true`,
deterministic fixtures, `data_source` recorded on every pipeline/backtest payload); set
`FUTURESRLX_API_ENABLE_IBKR=1` to attach a real IBKR paper/live session. The dashboard shows a
visible data-mode badge whenever it is not talking to a live backend.

### Docker stack

```bash
docker compose -f docker/docker-compose.yml up --build
# TimescaleDB + Redis + IB Gateway (paper) + FastAPI backend + Next.js frontend
```

## Strategy Studio, optimizer & DRL trainer

The `/studio` view provides three workflows on top of the same typed strategy configuration:

- **Manual builder.** A five-stage pipeline builder (Universe → Allocator → Timing → Risk/LLM →
  Translator) with live YAML preview, 45-root sector picker, overlay enable/disable toggles and
  slider + numeric inputs. Presets cover trend following, commodity carry/term structure, panic
  mean-reversion, seasonal commodity cycle and DRL + LLM risk.
- **Walk-forward optimizer.** `src/backtest/optimizer.py::WalkForwardOptimizer` supports Optuna plus
  a deterministic Latin-Hypercube fallback. Every trial is scored on purged/embargoed
  **out-of-sample** folds, a drawdown penalty suppresses overfit trials, flat/zero-activity trials
  receive a hard penalty and can never be selected as best, and studies persist to
  `artifacts/optimizations/<study_id>.json`. Apply writes the tuned strategy, reloads the catalog and
  persists a verification walk-forward run.
- **DRL trainer.** `POST /api/drl/train` and `python -m src.strategies.train_drl` train PPO / CPPO /
  SAC / TD3 / A2C on `FuturesTradingEnv`, writing `<checkpoint>.zip`, `<checkpoint>.zip.meta.json`
  (universe, features, shapes, hyperparameters) and a bounded reward curve. `DRLAllocator` loads
  checkpoints lazily and aligns observations/actions to the checkpoint's trained universe.

`GET /api/strategies/schema/catalog` exposes every configurable parameter with a plain-English
description, bounds, defaults, YAML encoding and preset templates. The current catalog has 121
parameters across universe/allocator/timing/risk/translator; structural/data-hygiene parameters are
not tunable, 85 parameters are Optuna-tunable, and five curated core parameters are pre-selected by
default. The catalog module is importable without Gymnasium/Stable-Baselines3.

## Quality gates

Every pull request must pass (see `scripts/quality_gate.sh`):

```bash
.venv/bin/ruff check .                 # zero lint errors
.venv/bin/ruff format --check .        # formatting
.venv/bin/python -m mypy src configs   # strict static typing
.venv/bin/python -m pytest tests/      # hermetic suite + coverage gate
```

All unit tests are **offline and deterministic**: they never call paid Databento endpoints or a live
IBKR session. Networked/adapter code is exercised through injected fakes.

## Architecture

The complete design document is [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). In brief:

- `src/core/types.py` is the narrow waist of the platform: immutable, validated contracts
  (`MarketState`, `WeightVector`, `ContractTarget(Set)`, `OrderEvent`, `Fill`, `RiskAssessment`,
  `PipelineResult`, …) exchanged by every module.
- `src/core/interfaces.py` defines the structural Protocols shared by backtest and live execution;
  the pipeline composer in `src/strategies/pipeline.py` wires `S_t → A_t → T_t → R_t`.
- `src/risk/contract_translator.py` is the only place that turns weights into discrete contracts;
  parity tests assert identical targets and deltas from the backtest and IBKR paths.
- `src/api/` exposes the same state to the dashboard through typed REST/WebSocket contracts and
  labels every payload with its data origin.

## Testing philosophy

- **Offline & deterministic** — seeded synthetic fixtures in `tests/fixtures/market.py`; adapters
  are injected. No network, no paid APIs, no wall-clock races.
- **Property-based** — `hypothesis` proves the translator never exceeds the margin ceiling across
  arbitrary equity/price/weight shocks.
- **Accounting identity** — the backtest integration test asserts
  `Ending Equity = Starting Equity + Price PnL + Roll PnL − Commissions − Slippage`.
- **Anti-look-ahead** — roll/feature/resampling/walk-forward tests mutate future data and assert
  nothing changes in the past.
- **Parity** — identical `MarketState` ⇒ identical `ContractTarget`s (and OMS deltas) from the
  backtest and IBKR paths.
- **Hardening** — hydrate-before-first-bar ordering, restart-safe risk latches, open-order
  reconciliation grace windows, non-blocking TTL LLM cache and EUR↔USD sizing parity are covered in
  `tests/integration/test_production_hardening.py` and `tests/integration/test_fx_normalization.py`.
- **Ablation** — LLM hallucination fixtures (low confidence) are ignored, while high-confidence
  critical alerts deterministically zero exposure.

## Release history

Versions follow [Semantic Versioning](https://semver.org/) and release notes follow
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The complete history — from the first
foundation release through the current multi-timeframe engine — is maintained in
[CHANGELOG.md](CHANGELOG.md).

## Disclaimer

This software is research infrastructure, **not investment advice**. Futures are leveraged; losses
can exceed the initial margin. Test on paper for a meaningful period before live use.

Live trading requires the explicit `FUTURESRLX_ENVIRONMENT=live` opt-in, a live IBKR port and
`ibkr_paper=false`; the settings validator refuses inconsistent configurations. The pre-trade gate is
mandatory and cannot be bypassed by strategy code. LLM outputs are untrusted data: they are
schema-validated, confidence-filtered and only ever applied through bounded deterministic
projections. The Databento SDK and live IBKR socket paths are lazy, opt-in adapters; unit/CI tests
exercise them through injected fakes and never contact vendors. Real-feed smoke tests belong in a
credentialed paper environment.

## License

MIT — see [LICENSE](LICENSE).
