# FuturesRL-X — User Guide

**How to install, run, configure, and operate the platform** — including Databento historical data, IBKR paper/live trading, the dashboard, and the backtest/risk CLIs.

> FuturesRL-X is research and execution infrastructure, **not investment advice**. Futures are leveraged; you can lose more than your initial margin. Paper-trade and run a full backtest before risking live capital.

---

## Table of contents

1. [What you get](#1-what-you-get)
2. [Prerequisites](#2-prerequisites)
3. [Install](#3-install)
4. [Quick start: run the app](#4-quick-start-run-the-app)
5. [Configuration (`.env` / `FUTURESRLX_*`)](#5-configuration)
6. [Data sources: Databento, IBKR, offline](#6-data-sources)
7. [IBKR broker setup (paper & live)](#7-ibkr-broker-setup)
8. [Running backtests](#8-running-backtests)
9. [Running live / paper trading](#9-running-live--paper-trading)
10. [Operating the risk system](#10-operating-the-risk-system)
11. [Docker deployment](#11-docker-deployment)
12. [Quality gate, tests & CI](#12-quality-gate-tests--ci)
13. [Troubleshooting](#13-troubleshooting)
14. [Security & safety notes](#14-security--safety-notes)

---

## 1. What you get

| Component | Entry point | Purpose |
|---|---|---|
| FastAPI telemetry backend | `python -m uvicorn src.api.main:app` | REST + `/ws/telemetry` for positions, pipeline, risk, term structure, backtests |
| Next.js dashboard | `cd frontend && npm run dev` | 5-view quant workstation (Command Center, Pipeline & Weights, Risk & LLM, Term Structure, Backtest Lab) on port **3100** |
| Backtest CLI | `python -m src.backtest.engine` | Declarative strategy backtests + walk-forward validation + tear sheets |
| Live runner | `python -m src.execution.runner` | Cold-start hydration → pipeline → OMS → IBKR paper/live (or `--offline` dry run) |
| Quality gate | `./scripts/quality_gate.sh --cov` | ruff + format + strict mypy + pytest + coverage |

Everything is **offline-capable**: with no credentials the API/dashboard/backtests run on deterministic fixture data, and the runner can be exercised end-to-end with `--offline`.

---

## 2. Prerequisites

- **Python ≥ 3.12** (the reference environment is 3.14)
- **Node.js ≥ 20** (dashboard; reference: Node 22)
- Optional: **IBKR TWS or IB Gateway** for paper/live trading
- Optional: **Databento account/API key** for real historical data
- Optional: **Docker + Compose** for the containerised stack

---

## 3. Install

### Fresh clone

```bash
git clone <your-repo-url> futures-app && cd futures-app

python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e ".[dev]"          # core + test/lint/type tooling

# optional extras
.venv/bin/pip install -e ".[live]"         # ib_async (IBKR paper/live)
.venv/bin/pip install -e ".[databento]"    # Databento SDK
.venv/bin/pip install -e ".[drl]"          # gymnasium + stable-baselines3/torch

cd frontend && npm install && cd ..
cp .env.example .env                       # then edit
```

### This workspace (already installed)

`/home/vidan/futures-app` already has `.venv/` and `frontend/node_modules/`; you can skip the install step and go straight to [Quick start](#4-quick-start-run-the-app).

Verify everything:

```bash
./scripts/quality_gate.sh --cov
```

---

## 4. Quick start: run the app

Two terminals:

```bash
# Terminal 1 — backend on :8000 (offline/deterministic by default)
cd /home/vidan/futures-app
.venv/bin/python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000

# Terminal 2 — dashboard on :3100
cd /home/vidan/futures-app/frontend
npm run dev            # dev server
# or: npm run build && npm run start   (production build)
```

Open **http://localhost:3100**.

| Route | View |
|---|---|
| `/` | Command Center — NLV, intraday PnL, margin utilisation, equity curve, positions, flatten-all |
| `/pipeline` | Pipeline & Weights — `S_t → A_t → T_t → R_t → N_i,t` waterfall + per-root transformation matrix |
| `/risk` | Risk & LLM Monitor — semantic risk, confidence, circuit breaker, macro lockouts, thresholds |
| `/term-structure` | Term Structure & Roll Calendar — forward curves, basis, roll schedule |
| `/backtests` | Backtest Lab — run/compare backtests, tear sheets, rolling Sharpe, Rachev tails |

**Data mode.** The dashboard reads `/health.offline_mode`:
- `DEMO DATA` badge → backend not running (pure deterministic mock fixtures).
- backend running with `offline_mode: true` → real pipeline/broker simulation on fixture market data (no paid vendors).
- `PAPER · IBKR` / `LIVE · IBKR` → backend attached to a real IBKR session (`FUTURESRLX_API_ENABLE_IBKR=1`).

Frontend/API base URL is `NEXT_PUBLIC_API_BASE_URL` (default `http://127.0.0.1:8000`); WebSocket `/ws/telemetry` is derived from it.

Stop servers with `Ctrl-C` in each terminal.

---

## 5. Configuration

### How configuration loads

`configs/settings.py` (pydantic-settings) reads, in order of precedence:

1. **Environment variables** `FUTURESRLX_*`
2. **`.env`** in the repo root (gitignored)
3. Defaults in `Settings` / `RiskSettings`

Secrets (`FUTURESRLX_DATABENTO_API_KEY`, IBKR account) are `SecretStr` values, redacted from logs/telemetry. **Never commit `.env`** and never put real keys in `.env.example`.

Nested risk settings use a double underscore: `FUTURESRLX_RISK__MAX_MARGIN_UTILIZATION=0.5`.

### Settings reference

#### Application & data

| Variable | Default | Meaning |
|---|---|---|
| `FUTURESRLX_ENVIRONMENT` | `development` | `development` / `paper` / `live` / `backtest` (paper/live enforce matching IBKR ports) |
| `FUTURESRLX_LOG_LEVEL` | `INFO` | Python log level |
| `FUTURESRLX_DATA_DIR` | `./data` | writable data root |
| `FUTURESRLX_PARQUET_DIR` | `./data/parquet` | partitioned Parquet cache |
| `FUTURESRLX_DUCKDB_PATH` | `./data/futuresrl_x.duckdb` | DuckDB file (analytics + optional risk state) |
| `FUTURESRLX_REPORT_DIR` | `./artifacts` | backtest/tear-sheet output |
| `FUTURESRLX_INSTRUMENTS_PATH` | `configs/instruments.yaml` | instrument master override |
| `FUTURESRLX_DATABENTO_API_KEY` | _(unset)_ | Databento key (secret) |
| `FUTURESRLX_DATABENTO_DATASET` | `GLBX.MDP3` | Databento dataset (CME/CBOT/NYMEX/COMEX) |
| `FUTURESRLX_DATABENTO_DEFAULT_SCHEMA` | `ohlcv-1d` | default bar schema |
| `FUTURESRLX_DATABENTO_LIVE_ENABLED` | `false` | Databento live feed (off by default) |

#### IBKR

| Variable | Default | Meaning |
|---|---|---|
| `FUTURESRLX_IBKR_HOST` | `127.0.0.1` | TWS/IB Gateway host |
| `FUTURESRLX_IBKR_PORT` | `7497` | paper TWS `7497`, paper Gateway `4002`, live TWS `7496`, live Gateway `4001` |
| `FUTURESRLX_IBKR_CLIENT_ID` | `42` | API client id (unique per session) |
| `FUTURESRLX_IBKR_PAPER` | `true` | must be `false` with a live port |
| `FUTURESRLX_IBKR_ACCOUNT` | _(unset)_ | optional account filter |
| `FUTURESRLX_IBKR_CONNECT_TIMEOUT_SECONDS` | `10` | connect timeout |
| `FUTURESRLX_IBKR_RECONNECT_MAX_ATTEMPTS` | `10` | reconnect attempts with exponential backoff |
| `FUTURESRLX_IBKR_NIGHTLY_RESET_START_HOUR_ET` / `END_HOUR_ET` | `23` / `0` | IBKR nightly reset window (pins `:45` boundaries) |

#### Risk engine

| Variable | Default | Meaning |
|---|---|---|
| `FUTURESRLX_RISK__MAX_MARGIN_UTILIZATION` | `0.60` | hard initial-margin ceiling `η_max` |
| `FUTURESRLX_RISK__MARGIN_WARNING_THRESHOLD` / `MARGIN_CRITICAL_THRESHOLD` | `0.60` / `0.80` | dashboard gauge zones |
| `FUTURESRLX_RISK__TARGET_ANNUAL_VOLATILITY` | `0.12` | ex-ante vol target |
| `FUTURESRLX_RISK__MAX_GROSS_LEVERAGE` | `3.0` | gross leverage cap |
| `FUTURESRLX_RISK__MAX_SINGLE_INSTRUMENT_WEIGHT` / `MAX_SECTOR_WEIGHT` | `0.50` / `0.40` | concentration caps |
| `FUTURESRLX_RISK__DAILY_MAX_LOSS_PCT` | `0.03` | intraday kill switch |
| `FUTURESRLX_RISK__TRAILING_STOP_PCT` / `DRAWDOWN_LIMIT_PCT` | `0.10` / `0.20` | HWM stops |
| `FUTURESRLX_RISK__LLM_CONFIDENCE_THRESHOLD` | `0.70` | `τ` anti-hallucination threshold |
| `FUTURESRLX_RISK__LLM_SAFE_SCORE` / `LLM_CRITICAL_SCORE` | `1.5` / `4.5` | `θ_safe` / `κ` |
| `FUTURESRLX_RISK__LLM_SAMPLE_COUNT` | `3` | K samples for confidence/variance filter |
| `FUTURESRLX_RISK__LLM_CACHE_TTL_SECONDS` | `60` | async LLM assessment cache TTL |
| `FUTURESRLX_RISK__MACRO_LOCKOUT_MINUTES_BEFORE` / `AFTER` | `30` / `30` | Tier-1 event lockout window |
| `FUTURESRLX_RISK__MAX_ORDER_CONTRACTS` | `200` | fat-finger limit per order |
| `FUTURESRLX_RISK__ORDER_RATE_LIMIT_PER_SECOND` | `5` | token-bucket rate |
| `FUTURESRLX_RISK__DUPLICATE_ORDER_WINDOW_SECONDS` | `1.0` | duplicate suppression window |
| `FUTURESRLX_RISK__RECONCILE_GRACE_SECONDS` | `30` | open-order grace before drift halt |
| `FUTURESRLX_RISK__RECONCILIATION_INTERVAL_SECONDS` | `30` | reconciler period |

#### Account currency / FX

| Variable | Default | Meaning |
|---|---|---|
| `FUTURESRLX_ACCOUNT_CURRENCY` | `USD` | IBKR base currency; `MarketState.equity` is denominated in it |
| `FUTURESRLX_FX_RATES` | `{}` | JSON `{"EUR": 1.10}` = USD per 1 unit of currency (USD implicit 1.0) |

For a non-USD IBKR account, set both, e.g. `FUTURESRLX_ACCOUNT_CURRENCY=EUR` and `FUTURESRLX_FX_RATES='{"EUR": 1.10}'`. The translator converts `E_t` to USD before sizing; a missing rate fails closed (no orders) rather than mis-sizing.

#### Production hardening (Phase 7)

| Variable | Default | Meaning |
|---|---|---|
| `FUTURESRLX_RISK_STATE_STORE_URI` | `""` | durable risk state: `""`/`memory:` = in-memory, path = JSON file, `duckdb:` = DuckDB (uses `DUCKDB_PATH`), `redis:<key>` = injected Redis client |
| `FUTURESRLX_HYDRATION_LOOKBACK_DAYS` | `400` | history window warmed before the first live bar |
| `FUTURESRLX_HYDRATION_BACKFILL_ENABLED` | `true` | allow the hydrator to backfill missing history |
| `FUTURESRLX_HYDRATION_SOURCE` | `auto` | backfill source: `auto` / `databento` / `ibkr` / `offline` |

#### API / dashboard

| Variable | Default | Meaning |
|---|---|---|
| `FUTURESRLX_API_HOST` / `PORT` | `0.0.0.0` / `8000` | FastAPI bind |
| `FUTURESRLX_API_CORS_ORIGINS` | `localhost:3000`, `localhost:3100` | dashboard origins |
| `FUTURESRLX_TELEMETRY_INTERVAL_SECONDS` | `1.0` | WebSocket telemetry cadence |
| `FUTURESRLX_API_ENABLE_IBKR` | `0` (compose) | opt-in real IBKR session in the API container |

### Instrument master

`configs/instruments.yaml` is the single source of truth for multipliers, tick sizes, margin estimates, roll rules, and standard↔micro pairs (`ES↔MES`, `NQ↔MNQ`, `CL↔MCL`, `GC↔MGC`). Add a root by adding an entry (see existing structure: exchange, `margins.initial/maintenance`, `roll.contract_months`, `databento_symbol`, `ibkr_exchange`). Margins are conservative estimates; live orders are always confirmed via IBKR `whatIfOrder`.

### Strategy configs

`configs/strategies/*.yaml` are declarative pipelines, e.g.:

```yaml
id: tsmom_vol_target
universe: {selector: liquidity, params: {...}}
allocator: {name: volatility_target, params: {...}}
timing: [{name: tsmom, params: {...}}, ...]
risk: [{name: quant_risk, params: {...}}, {name: llm_risk, params: {...}}]
translator: {rounding_mode: nearest, max_margin_utilization: 0.60, enable_micro_routing: true}
backtest: {start: 2018-01-01, end: 2024-12-31, initial_equity: 1000000, rebalance_frequency: W-FRI}
```

Edit these YAMLs (or the API's in-memory strategy editor) to change selectors/overlays; no code change required.

---

## 6. Data sources

### Parquet cache layout

```
data/parquet/<dataset>/<schema>/<root>/part.parquet
```

(`<dataset>` = `GLBX.MDP3`, `<schema>` = `ohlcv-1d`/`ohlcv-1m`/`tbbo`/`definition`, …). Writes are atomic and deduplicated. Query with DuckDB via `ParquetRepository.query(...)`.

### Databento (recommended for research)

1. Create an API key at Databento → Dashboard → API keys.
2. Put it in `.env` (never in `.env.example` / git):

   ```bash
   FUTURESRLX_DATABENTO_API_KEY=db-xxxxxxxxxxxxxxxx
   FUTURESRLX_DATABENTO_DATASET=GLBX.MDP3
   ```

3. Ingest history (cache-first; re-runs are cheap because of dedupe):

   ```python
   from src.data import DatabentoLoader, ParquetRepository
   from src.data.continuous_roller import ContinuousContractBuilder

   repo = ParquetRepository()
   loader = DatabentoLoader(repository=repo)
   loader.ingest(["ES", "NQ", "CL", "GC", "ZN"],
                 start="2023-01-01", end="2024-12-31", schema="ohlcv-1d")
   loader.ingest_definitions(...)                     # contract specs / expiries
   continuous = ContinuousContractBuilder().build_frame(bars, root="ES")
   ```

   Or simply let the runner backfill: set `FUTURESRLX_HYDRATION_SOURCE=databento` (or leave `auto` with a key configured) and start it — the cold-start hydrator fetches the lookback window it needs and persists it.

4. Supported schemas: `ohlcv-1m`, `ohlcv-1h`, `ohlcv-1d`, `tbbo`, `mbp-1`, `mbp-10`, `definition`. If the SDK or key is missing the loader raises an actionable error (never silently fabricates vendor data).

### IBKR historical / live market data

- **Live/paper streaming**: `src/data/ibkr_data.IBKRMarketDataStreamer` streams 5-second bars, ticks and L1/L2 book updates via `ib_async` (requires a running TWS/Gateway).
- **Historical backfill**: the hydrator can fetch daily bars through IBKR's `reqHistoricalDataAsync` when `FUTURESRLX_HYDRATION_SOURCE=ibkr` (or `auto` with no Databento key and IBKR configured). Fetched bars are validated against the canonical schema and persisted to Parquet, so the next start is offline-cached.

### Offline (no credentials)

`OfflineBarSource` generates a deterministic seeded synthetic series (clearly marked `_SYN` instrument ids). Use it for tests, demos and runner dry-runs:

```bash
.venv/bin/python -m src.execution.runner --offline --max-states 3 \
  --strategy configs/strategies/tsmom_vol_target.yaml
```

Hydration source resolution in `auto` mode:

1. Databento (if `FUTURESRLX_DATABENTO_API_KEY` is set) →
2. IBKR historical (if paper/live is configured and reachable) →
3. Offline source — only when `--offline` or `FUTURESRLX_HYDRATION_SOURCE=offline` →
4. Otherwise: clean CLI error explaining the three ways to provide data (no stack trace).

---

## 7. IBKR broker setup

### TWS / IB Gateway prerequisites

1. Install **TWS** or **IB Gateway**, log into the paper or live account.
2. Enable the API: *File → Global Configuration → API → Settings*
   - Enable *ActiveX and Socket Clients*
   - Uncheck *Read-Only API*
   - Add `127.0.0.1` to *Trusted IPs*
   - Note the socket port: TWS paper **7497**, TWS live **7496**, Gateway paper **4002**, Gateway live **4001**
3. Create an API client id (any unique int; default `42`).
4. (Live only) enable futures trading permissions on the account, and set `FUTURESRLX_IBKR_PAPER=false` with a live port. The settings validator refuses inconsistent paper/live combinations.

### `.env` example (paper)

```bash
FUTURESRLX_ENVIRONMENT=paper
FUTURESRLX_IBKR_HOST=127.0.0.1
FUTURESRLX_IBKR_PORT=7497          # 4002 for headless IB Gateway
FUTURESRLX_IBKR_CLIENT_ID=42
FUTURESRLX_IBKR_PAPER=true
FUTURESRLX_IBKR_ACCOUNT=           # optional
FUTURESRLX_RISK_STATE_STORE_URI=duckdb:   # durable intraday risk state
```

### Validate before trading

```bash
.venv/bin/python -m src.execution.runner --check \
  --strategy configs/strategies/tsmom_vol_target.yaml
```

This opens **no socket**: it validates settings, instrument master, strategy YAML, FX/base currency, state-store type and hydration configuration, and prints a JSON summary. Exit code `0` = ready.

### IBKR nightly reset

IBKR resets its servers nightly around **23:45–00:45 ET**. The connection manager detects this window and waits; the durable risk-state store ensures the daily P&L baseline, high-water marks and kill-switch latch survive the reset/reconnect (see §10).

---

## 8. Running backtests

### Basic run

```bash
.venv/bin/python -m src.backtest.engine \
  --strategy configs/strategies/tsmom_vol_target.yaml \
  --offline \
  --output artifacts/tsmom_tearsheet.json \
  --html artifacts/tsmom_tearsheet.html
```

| Flag | Meaning |
|---|---|
| `--strategy` | declarative YAML (required) |
| `--offline` | deterministic synthetic bars, no network |
| `--bars <parquet>` | use a cached canonical Parquet bar file instead |
| `--mode {event,vectorized}` | event-driven accounting-parity mode (default) vs fast research mode |
| `--initial-equity` | override starting NLV |
| `--output` / `--html` | JSON / standalone HTML tear sheet paths |
| `--walk-forward` | run purged & embargoed walk-forward validation |
| `--wf-splits N` | number of walk-forward folds (default 5) |
| `--purge-bars N` / `--embargo-bars N` | leakage guards between train/test windows |

Walk-forward example:

```bash
.venv/bin/python -m src.backtest.engine \
  --strategy configs/strategies/tsmom_vol_target.yaml \
  --offline --walk-forward --wf-splits 5 --purge-bars 5 --embargo-bars 5 \
  --output artifacts/wf_tsmom.json --html artifacts/wf_tsmom.html
```

Output JSON includes run/fold metadata (`train_start/train_end/test_start/test_end`, fold metrics), per-fold metrics and aggregate mean/std statistics — useful for the Backtest Lab comparison.

### Programmatic API

```python
from src.backtest.engine import run_from_config
from configs.strategy import load_strategy_config

result = run_from_config(load_strategy_config("configs/strategies/tsmom_vol_target.yaml"))
result.check_accounting_identity()          # Ending = Starting + PricePnL + RollPnL - costs
print(result.to_dict()["metrics"])
```

### Tear sheets & comparison

`src.analytics.tearsheet.build_tearsheet(result)` → JSON + HTML; `compare_tearsheets({name: sheet})` → Backtest Lab comparison payload (normalised equity, underwater curve, rolling Sharpe, Rachev tails, roll-yield attribution).

---

## 9. Running live / paper trading

### Dry run (no broker, deterministic)

```bash
.venv/bin/python -m src.execution.runner --offline \
  --strategy configs/strategies/tsmom_vol_target.yaml \
  --max-states 3 --roots ES,NQ,CL
```

This hydrates from the offline source, builds a simulated broker/streamer, runs the full `pipeline → translator → OMS` chain for a few states and prints a JSON status. Use it to validate wiring without keys or TWS. Tip: the first couple of replayed states can legitimately be flat (TSMOM/KAMA gates), so use `--max-states 30` if you want to see simulated orders (`orders_submitted > 0`).

`--backfill-source` accepts `auto` (Databento → IBKR → offline-if-allowed), `databento`, `ibkr`, or `offline`; `--offline` implies the offline source and cannot be combined with a forced `databento`/`ibkr` source (clean exit 2). With `--check` the resolved source order is printed in the JSON summary.

### Paper / live

```bash
export FUTURESRLX_ENVIRONMENT=paper
export FUTURESRLX_IBKR_PORT=7497          # or 4002 for Gateway
export FUTURESRLX_RISK_STATE_STORE_URI=duckdb:

.venv/bin/python -m src.execution.runner \
  --strategy configs/strategies/tsmom_vol_target.yaml \
  --roots ES,NQ,CL,GC,ZN \
  --backfill-source auto                 # auto|databento|ibkr|offline
```

Startup sequence (in order):

1. validate settings/instrument master/strategy,
2. **hydrate** the lookback window (Databento → IBKR → offline fallback policy) and validate warm features/covariance,
3. load durable risk state; refuse to automate if the persisted kill switch is latched,
4. connect market-data streamer + OMS, prime the pipeline on the warm state,
5. for each live bar: `pipeline.run → ContractTargetSet → OMS pre-trade gate → IBKR`,
6. reconcile every 30 s and halt automated rebalancing on unapproved drift,
7. on SIGINT: cancel orders, disconnect, persist state.

`--check` performs only step 1 and 3 read-only and exits.

### What the live runner needs

- TWS/IB Gateway running and API-enabled (§7)
- `FUTURESRLX_ENVIRONMENT=paper|live` with matching port
- a writable Parquet cache (`FUTURESRLX_PARQUET_DIR`) for hydration
- optionally `FUTURESRLX_RISK_STATE_STORE_URI` so restarts/nightly resets keep risk state

### Monitoring

- API `GET /api/positions`, `/api/pipeline/latest`, `/api/risk/state`
- Dashboard Command Center (margin gauge, EMERGENCY FLATTEN ALL)
- WebSocket `ws://127.0.0.1:8000/ws/telemetry` (pnl/margin/positions/risk_alert/pipeline frames)
- stderr logs + persisted risk state file

---

## 10. Operating the risk system

### Kill switch / emergency flatten

- **Dashboard**: top-ribbon *Flatten All* → two-step confirm → `POST /api/risk/flatten-all` (with simulated broker) or OMS `flatten_all()` in live mode: cancel all orders, market-close all positions, engage the lock; the durable state store records the lock so a restart stays halted.
- **Programmatic**: `oms.flatten_all()` then `oms.reset_kill_switch()` only after an operator review.
- **Runner**: if the store says `kill_switch_locked`, the runner will *not* submit orders; reset via `oms.reset_kill_switch()` or the API, then restart.

### Thresholds

Live via dashboard Risk view → *Apply Changes* → `PUT /api/risk/thresholds`, or edit `.env` `FUTURESRLX_RISK__*` and restart. Overrides are validated (e.g. `θ_safe < κ`, warning < critical margin).

### Macro lockouts

`configs/macro_calendar.yaml` lists FOMC/CPI/NFP/EIA/WASDE/OPEC events (UTC). Within the configured `±30m` (default) window the LLM overlay flattens or caps leverage for affected roots. Edit the YAML and restart to change.

### Durable state

With `FUTURESRLX_RISK_STATE_STORE_URI` set, the following survive restarts and IBKR nightly resets:

- `starting_daily_nlv` (daily loss baseline)
- portfolio & per-instrument high-water marks
- `kill_switch_locked` + reason, daily-loss latch
- OMS token-bucket rate-limiter state

Day roll (UTC date change) resets the daily counter but deliberately keeps high-water marks and latched kill switches. Use `duckdb:` or Redis for higher order rates; a JSON file fsyncs on each persisted change.

### Reconciliation

Every 30 s (`FUTURESRLX_RISK__RECONCILIATION_INTERVAL_SECONDS`) the reconciler compares internal book vs broker. In-flight working orders (including partial fills) are added to the broker position; drift is only flagged/halted when it is unexplained for longer than `RECONCILE_GRACE_SECONDS`. On drift: automated rebalancing halts, an alert is emitted; investigate and `resume()` after approval.

---

## 11. Docker deployment

```bash
cp docker/.env.example docker/.env      # edit credentials/ports
docker compose -f docker/docker-compose.yml up --build
```

Services: TimescaleDB, Redis, headless IB Gateway (paper), FastAPI backend (:8000), Next.js frontend (:3080). The backend runs offline/demo by default; set `FUTURESRLX_API_ENABLE_IBKR=1` in `docker/.env` to attach a real IBKR session. Healthchecks gate startup order.

Without a Docker daemon (e.g. this workspace host) use the local runs in §4/§8/§9 instead; CI (`.github/workflows/ci.yml`) builds both images on push.

---

## 12. Quality gate, tests & CI

```bash
./scripts/quality_gate.sh          # ruff check + format check + mypy strict + pytest
./scripts/quality_gate.sh --cov    # + ≥85% coverage on core/risk/translator modules
```

The gate is hermetic: it unsets any inherited `FUTURESRLX_*` variables, and `tests/conftest.py` has an autouse fixture that strips them and ignores `.env` per test — so exporting `FUTURESRLX_ENVIRONMENT=paper` in your shell can never change test results. All tests are offline and deterministic (no paid/vendor calls; adapters are injected fakes).

Useful subsets:

```bash
.venv/bin/python -m pytest tests/unit -q
.venv/bin/python -m pytest tests/integration -q
.venv/bin/python -m pytest tests/parity -q
.venv/bin/python -m pytest -k walk_forward -q
```

---

## 13. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `HydrationError: ... No Databento API key configured` | Empty cache and no data source. Set `FUTURESRLX_DATABENTO_API_KEY`, start TWS/IB Gateway and use `--backfill-source ibkr`, or pass `--offline` for a synthetic dry run. |
| Runner prints a long traceback at startup | Fixed: CLI now prints an actionable message and exits `2`. If you see a traceback, check you are on the current branch/build. |
| `could not connect to IBKR ...` | TWS/Gateway not running, API not enabled, wrong port, or client-id already in use. Check *Global Configuration → API*, ports 7497/4002 (paper) or 7496/4001 (live). |
| `live environment requires an IBKR live port` | `FUTURESRLX_ENVIRONMENT=live` with paper port (or `ibkr_paper=true`). Fix the env pair. |
| Dashboard shows `DEMO DATA` | Backend not running/reachable on `127.0.0.1:8000`. Start terminal 1; refresh. |
| Port already in use | API: `--port 8001` (+ `NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8001`); dashboard is **3100** (`next dev -p 3100`). Port 3080 belongs to a different local UI. |
| `RuntimeWarning: 'src.backtest.engine' found in sys.modules` | Fixed by lazy package `__init__`; rebuild/use current commit. |
| Gym `Box observation space minimum is -infinity` warning | Fixed with finite bounds; use current commit. |
| Tests fail only on your machine | A host `FUTURESRLX_*` export or `.env` used to leak in — fixed by hermetic fixtures; if you see it, run `git status` / verify conftest on the current commit. |
| `margin utilization` -> `whatIf` rejection | Order would exceed `η_max` after margin impact; reduce gross/target or raise `RISK__MAX_MARGIN_UTILIZATION` deliberately. |
| `kill switch engaged: ...` on startup | Persisted lock (daily loss/drawdown/operator). Review `/api/risk/state`, fix the cause, then `oms.reset_kill_switch()` / API reset. |
| `reconciliation drift` | OMS/broker mismatch, or open orders older than the grace window. Check `/api/positions`, TWS positions, working orders; `approve_drift`/`resume` only after manual verification. |
| Databento `403/authentication` | Wrong/expired key, dataset entitlement missing (e.g. `GLBX.MDP3`), or IP restriction. |

Logs: `data/` holds cache; `artifacts/` holds tear sheets; API logs to stdout; runner logs to stderr with `run_id`/state telemetry.

---

## 14. Security & safety notes

- **Never commit `.env`** or real API keys. `.env.example` must stay placeholder-only (the quality gate/tests do not require secrets).
- Use IBKR **paper** first; set `FUTURESRLX_ENVIRONMENT=live` deliberately and verify ports/account before enabling.
- The pre-trade gate is mandatory: fat-finger size, duplicate/rapid-fire rate limiting, and real-time `whatIfOrder` margin impact. Live trading refuses to start when the persisted kill switch is latched.
- LLM outputs are schema-validated, confidence-filtered and only applied through bounded deterministic projections. Treat news/macro text as untrusted data.
- Reconcile and monitor; never disable the pre-trade gate or reconciler on a live account without a documented operator override.
- Futures margin is leveraged — losses can exceed capital. This software is provided for research/engineering use; validate thoroughly before any live deployment.
