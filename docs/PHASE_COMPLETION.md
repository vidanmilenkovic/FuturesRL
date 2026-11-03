# FuturesRL-X — Phase Completion Summaries

Each phase below was developed on a dedicated feature branch, gated by
`ruff check` + `ruff format --check` + strict `mypy` + `pytest`, and released
with an annotated git tag. All tests are offline and deterministic.

---

## Phase 1 — Foundation, core contracts & data layer (`v0.1.0-data`)

**Branch:** `feat/phase-1-core-data`

**Deliverables**
- Build/config foundation: `pyproject.toml` (ruff, mypy strict, pytest, coverage), `.gitignore`,
  `.env.example`, `scripts/quality_gate.sh`, `Makefile`, GitHub Actions CI, root `.dockerignore`.
- `src/core/types.py` — immutable contracts: `MarketState`, `WeightVector`, `ContractTarget(Set)`,
  `OrderEvent`, `Fill`, `Position`, `AccountSnapshot`, `RiskAssessment`, `PipelineResult`, `Bar`,
  `RollEvent`, enums; validated invariants and read-only embedded mappings/arrays.
- `src/core/interfaces.py` — structural Protocols for `S_t`/`A_t`/`T_t`/`R_t`, pipeline,
  translator, data sources, broker adapter, risk providers, reconciler.
- `src/core/event_bus.py` — async pub/sub bus shared by live and event-driven backtest paths.
- `configs/` — typed `Settings` (pydantic-settings, secret redaction, paper/live port guards),
  `InstrumentsConfig` master (ES/MES, NQ/MNQ, YM, RTY, CL/MCL, GC/MGC, ZN, ZT, 6E with
  multipliers, ticks, margins, roll rules, micro mappings), declarative strategy config models
  and YAML pipelines, macro event calendar.
- `src/data/` — canonical Polars/Parquet schema, partitioned `ParquetRepository` + DuckDB query
  facade, `DatabentoLoader` (lazy SDK; ohlcv 1m/1h/1d, tbbo, mbp-1/10, definition) and
  deterministic `OfflineBarSource`, continuous futures roll engine (volume/OI crossover,
  calendar N-days before FND/LTD, Databento native cross-check; Panama/proportional/unadjusted),
  FinRL-X feature processor (MACD/RSI/CCI/ADX/Bollinger/ATR/Garman-Klass, term structure, carry,
  volume/OI, OFI, turbulence, macro regime), point-in-time `MarketStateBuilder`, IBKR market-data
  streamer (`ib_async`, 5s bars/ticks/L1-L2, reconnect).
- Tests: deterministic fixtures shared by the whole suite, core-contract suite, config suite,
  event-bus suite, data-layer unit + integration suites.

**Verification (as merged)**
- `ruff check .` / `ruff format --check .` clean; `mypy src configs` strict clean.
- Data layer: 128 tests, 92% coverage (every module ≥87%), zero network/SDK calls.
- Commits: `feat(core)`, `feat(config)`, `test(core)`, `feat(data)` x4, `test(data)`.
- Tag: `v0.1.0-data`.

---

## Phase 2 — FinRL-X pipeline & futures contract translator (`v0.2.0-pipeline`)

**Branch:** `feat/phase-2-finrlx-pipeline`

**Deliverables**
- `src/strategies/pipeline.py` — `ComposablePipeline` implementing `w_t = R_t(T_t(A_t(S_t(X))))`
  with per-stage timing/diagnostics, strict stage-output validation, declarative YAML factory
  (`build_pipeline`), stage registries, `IdentityRiskOverlay`.
- Selectors: `LiquiditySelector`, `RegimeSelector` (slow 26-week trend/breadth/macro-vol + fast
  3-day shock detector), `CompositeSelector`.
- Allocators: `EqualWeightAllocator`, `RiskParityAllocator` (equal-risk-contribution fixed point),
  `MeanVarianceAllocator`, `VolatilityTargetAllocator`, `DRLAllocator` (PPO/SAC/TD3/A2C and CPPO
  via risk-shaped reward; lazy Stable-Baselines3 import, CPU default).
- Timing overlays: `TSMOMTiming`, `KAMATiming`, `CarryFilter`, `CompositeTiming`.
- `src/strategies/envs/futures_env.py` — Gymnasium `FuturesTradingEnv` with margin dynamics,
  forced liquidation, transaction costs and the Tier-A semantic-risk reward penalty.
- `src/risk/contract_translator.py` — deterministic notional→contracts translation: rounding
  modes (floor/nearest+buffer/ceil), same-direction standard↔micro routing (no self-hedging),
  margin feasibility projection with `η_max = 0.60` guarantee (hypothesis-tested).
- Tests: 220 tests including a ~3s PPO end-to-end training test; `src/strategies` coverage 96%,
  translator coverage 100%.

**Verification:** ruff/mypy strict clean; full suite offline deterministic.

---

## Phase 3 — LLM Risk-First engine & quantitative risk overlays (`v0.3.0-risk`)

**Branch:** `feat/phase-3-risk-engine`

**Deliverables**
- `src/risk/llm_risk_overlay.py`:
  - strict Pydantic LLM output schemas + async adapters for OpenAI / Anthropic / DeepSeek / local
    vLLM (lazy `httpx`, retry/backoff, no network in tests);
  - Module 1 anti-hallucination confidence & multi-sample variance filter (neutralise to
    `S = θ_safe` when `c < τ` or variance too high);
  - Module 2 exact Lagrangian reward penalty `λ·Σ|w_i|·1[c_i≥τ]·max(0, S_i−θ_safe)`;
  - Module 3 deterministic circuit breaker (`S ≥ κ ∧ c ≥ τ ⇒ w = 0`; linear de-risking between
    `θ_safe` and `κ`) + inclusive Tier-1 macro lockout windows via `MacroEventCalendar`.
- `src/risk/quant_risk_overlay.py`: EWMA covariance volatility targeting, gross/single/sector
  caps, high-water-mark trailing stops, daily max-loss kill switch, drawdown limit.
- `configs/macro_calendar.yaml`: editable FOMC/CPI/NFP/EIA/WASDE/OPEC/geopolitical template.
- Tests: 206 tests, 99% coverage, hypothesis margin-ceiling properties, hallucination ablation,
  deterministic-breaker assertions.

---

## Phase 4 — Futures-native backtester & analytics suite (`v0.4.0-backtest`)

**Branch:** `feat/phase-4-backtest-analytics`

**Deliverables**
- `src/backtest/cost_model.py`: tick-aware spread (≥1 tick), volume-impact slippage, tiered IBKR
  commissions + CME/CBOT/NYMEX/COMEX clearing/NFA fees, two-legged roll costs.
- `src/backtest/margin_simulator.py`: IM/MM tracking, daily mark-to-market settlement,
  margin-call detection, forced liquidation with penalty.
- `src/backtest/walk_forward.py`: purged + embargoed `PurgedWalkForwardSplit` (zero leakage).
- `src/backtest/simulated_broker.py`: full async `BrokerAdapter` (market/limit/stop matching,
  partial fills, settlement, rolls, flatten/cancel).
- `src/backtest/engine.py`: vectorized Polars mode + `EventBacktester`; `BacktestResult` with
  equity curve, weight history, trades, roll events, forced liquidations, margin series,
  attribution and the accounting identity
  `Ending = Starting + PricePnL + RollPnL − Commissions − Slippage − Funding`;
  `run_from_config` + offline CLI (`python -m src.backtest.engine ...`).
- `src/analytics/metrics.py`: CAGR, vol, Sharpe, Sortino, Calmar, Omega, drawdown/duration,
  historical+parametric VaR 95/99, CVaR, Rachev α=0.05, RoM, margin efficiency, implementation
  shortfall, turnover, commission ratio, spot-vs-roll attribution.
- `src/analytics/tearsheet.py`: JSON payload + standalone HTML report (inline CSS/SVG),
  `compare_tearsheets` for the Backtest Lab.
- Integration: full-stack offline test (data → features → states → pipeline → risk → translator →
  backtest → metrics → tear sheet) and Lead-owned deployment-parity suite.

**Verification**
- 114 backtest/analytics tests, 91% coverage; 5 parity tests proving the backtest engine and
  live OMS generate identical `ContractTarget` deltas (including micro routing and shorts).
- Accounting identity asserted to 1e-6 relative tolerance.

---

## Phase 5 — IBKR paper/live execution engine (`v0.5.0-execution`)

**Branch:** `feat/phase-5-execution`

**Deliverables**
- `src/execution/ibkr_client.py`: async `IBKRConnectionManager` on lazily imported `ib_async`
  (lock-protected connect, heartbeat, exponential backoff + jitter, IBKR nightly reset window
  23:45–00:45 ET, account/positions → core types, conId resolution, `whatIfOrder` margin).
- `src/execution/oms.py`: deterministic delta orders (idempotent UUID5 ids), `PreTradeRiskGate`
  (kill switch → fat finger → duplicate/rate limiter → mandatory whatIf margin), MKT/LMT/STP/
  STP-LMT/IBALGO/TWAP/VWAP, atomic bracket orders, `submit_targets`, emergency
  `flatten_all()` (cancel-all + market-close + lock), status/fill aggregation; structurally
  satisfies `BrokerAdapter`.
- `src/execution/ibkr_roller.py`: front/next selection from instrument master + expiry rules,
  native 1:1 `BAG` calendar-spread rolls, fill verification, two-legged fallback.
- `src/execution/reconciler.py`: internal-vs-IBKR position/cash reconciliation, drift reports,
  sticky halt, injectable-clock async loop.
- Tests: 127 execution tests (fake IB transport, offline), 95% coverage; parity suite remains green.

---

## Phase 6 — FastAPI telemetry, Next.js dashboard & Docker stack (`v1.0.0`)

**Branch:** `feat/phase-6-api-dashboard`

**Deliverables**
- `src/api/`: FastAPI app factory + lifespan, 10 REST routers (health, positions, pipeline
  waterfall, risk thresholds/state/flatten-all, term structure & roll calendar, strategies
  catalog/editor, backtests + tearsheets, telemetry status), typed Pydantic schemas, DI container
  with real wiring (EngineMarketService → MarketStateBuilder/build_pipeline/ContractTranslator/
  QuantRiskOverlay/LLM overlay; real `BacktestEngine` for POST runs; real `SimulatedBroker` for
  positions/kill switch; opt-in IBKR session), `/ws/telemetry` EventBus-backed broadcaster,
  documented offline/demo mode and fixture fallbacks.
- `frontend/`: Next.js 15 + TypeScript strict + Tailwind + TradingView Lightweight Charts; five
  views (Command Center with margin gauge + EMERGENCY FLATTEN ALL, Pipeline Inspector, LLM
  Risk-First Monitor, Term Structure & Roll Calendar, Backtest Lab & Tear Sheet Compare), typed
  API client with graceful DEMO DATA fallback and live WebSocket telemetry.
- `docker/`: multi-stage backend/frontend Dockerfiles + compose stack (TimescaleDB, Redis,
  headless IB Gateway, backend, frontend) with healthchecks, volumes and documented env vars.
- Tests: FastAPI route + WebSocket integration suites, offline deterministic.

**Verification**
- `ruff`/`mypy` clean on `src/api`; ~19 API tests pass in <4s.
- `next build` succeeds (8/8 pages), `tsc --noEmit` + ESLint clean.
- Docker binaries are not available on this build host; compose YAML, services, ports and env
  contracts were validated statically (documented limitation).

---

## Phase 7 — Production hardening (`v1.1.0`)

**Branch:** `feat/phase-7-production-hardening`

**Deliverables**
1. **Cold-start state hydration** — `src/execution/hydrator.py` (`ColdStartHydrator`,
   `HydrationResult`, `HydrationError`): reads the lookback window from the partitioned Parquet
   cache, optionally backfills missing history through an injected provider (Databento
   `ingest` / IBKR historical data), computes FinRL-X features and PIT `MarketState`s, and
   validates a fully warm state (required features finite, aligned EWMA covariance, non-empty
   return windows) before trading. `src/execution/runner.py` (`TradingRunner`) hydrates *before*
   the first streamed bar (`RunnerNotHydratedError` otherwise), enriches live states with the
   broker account NLV/base currency, primes the pipeline on the warm state, honours the durable
   kill switch, runs `pipeline -> translator -> OMS.submit_targets`, publishes telemetry and shuts
   down cleanly; `--check` validates the whole live configuration offline.
2. **Persistent risk & kill-switch state** — `src/risk/state_store.py` (`PersistedRiskState`,
   `RateLimiterState`, `RiskStateStore`, atomic JSON / DuckDB / Redis / in-memory backends,
   `build_risk_state_store`). `QuantRiskOverlay(state_store=...)` restores/re-persists daily NLV,
   portfolio & instrument high-water marks, kill-switch/daily-loss latches and trading day;
   day rolls keep HWM + stops while resetting daily counters. `OrderManagementSystem(state_store=...)`
   restores the kill-switch latch and OMS token bucket across restarts and persists lock/unlock/
   flatten via load-modify-save (preserving other writers' fields).
3. **Open-order-aware reconciliation** — `PositionReconciler` now consumes an injectable
   `working_orders` provider (`WorkingOrderSnapshot` with signed remaining quantity + `updated_at`)
   and a configurable `reconcile_grace_seconds`: drift is only flagged/halted when
   `broker_position + Σ working_deltas != expected` and the newest in-flight order is older than
   the grace window. Partial fills are explicitly handled; provider failures remain fail-closed;
   the no-provider path is byte-identical to Phase 5.
4. **Async cached LLM risk worker** — `LLMRiskOverlay` now has a TTL cache
   (`cache_ttl_seconds`, `valid_until_utc`, `is_cache_fresh`), a non-blocking fire-and-forget
   refresh from the synchronous `apply_risk` projection, and `AsyncLLMRiskCacheWorker` (single
   flight, TTL interval, capped exponential backoff, last-known-good assessments, cancellation-safe
   start/stop). Slow K-sample multi-provider calls can never stall the rebalance pipeline or the
   WebSocket event loop.
5. **Base-currency FX normalisation** — `MarketState` gained `account_currency` + `fx_rates`
   (USD-per-unit convention, `fx_rate_to_usd`, `equity_in_usd`, `converted_to_currency`);
   `ContractTranslator` converts `E_t` into USD for notional sizing and margin feasibility,
   converts non-USD contract prices for sizing only (target marks stay native), resolves overrides
   per-call > constructor > state, fails closed on missing rates, and leaves the USD path
   numerically identical. Pipeline/backtest/API/live-runner call sites pass the state's account
   FX so non-USD IBKR base accounts are sized correctly end-to-end.

**Verification (as merged)**
- `ruff check .` / `ruff format --check .` clean; `mypy src configs` strict clean.
- Full gate `./scripts/quality_gate.sh --cov`: **1082 passed**, coverage on `src/core` + `src/risk`
  = **93.97%** (translator 100%, quant overlay 100%, LLM overlay 99%, state store 82%).
- New tests: 19 hydrator, 23 runner, 12 reconciler, 7 OMS persistence; 18 durable-state store;
  18 quant persistence + 17 async LLM cache; 51 FX translator + 5 FX end-to-end integration;
  6 hardening integration tests (hydration ordering, restart persistence, open-order reconciliation,
  non-blocking LLM worker).
- Release: tag `v1.1.0` on `main` after a fast-forward merge of the phase branch.

---

## Phase 8 — Production polish & institutional UI (`v1.2.0`)

**Branch:** `feat/phase-8-production-polish`

**Backend, CLI & test-isolation fixes**
1. **Hermetic tests** — `tests/conftest.py` gained an autouse fixture that deletes every inherited
   `FUTURESRLX_*` variable, disables the repo `.env` file for `Settings()` and clears the settings
   cache around each test; `scripts/quality_gate.sh` unsets all `FUTURESRLX_*` variables before
   lint/type/test. The gate is now identical whether or not your shell exports
   `FUTURESRLX_ENVIRONMENT=paper` / `FUTURESRLX_RISK_STATE_STORE_URI=duckdb:`.
2. **Multi-source hydration fallback** — `src/execution/hydrator.py` composes a backfill chain:
   `auto` = Databento (API key) → IBKR historical (`reqHistoricalDataAsync` via
   `src/data/ibkr_data.IBKRHistoricalBarFetcher`) → `OfflineBarSource` (only when `--offline` or
   `FUTURESRLX_HYDRATION_SOURCE=offline`); forced sources fail closed with operator hints.
   `src/execution/runner.py` gained `--offline` and `--backfill-source {auto,databento,ibkr,offline}`
   plus an offline streamer/`OfflineTargetBroker` dry-run stack and clean exit-2 CLI errors (no raw
   traceback) that explain `--offline`, IBKR TWS/Gateway setup and the Databento key.
3. **Walk-forward CLI** — `src/backtest/engine.py` CLI supports `--walk-forward`,
   `--wf-splits`, `--purge-bars`, `--embargo-bars` with a real purged/embargoed fold runner,
   per-fold metrics, aggregate mean/std and a compounded out-of-sample curve/tear sheet.
4. **Zero runpy warnings** — PEP 562 lazy `__getattr__`/`__dir__` package inits for the
   `src/backtest.engine` and `src.execution.runner` export sets (`TYPE_CHECKING` imports for mypy);
   `python -W error::RuntimeWarning -m src.backtest.engine --help` and
   `... -m src.execution.runner --check` run with empty stderr.
5. **Finite Gym bounds** — `FuturesTradingEnv` observation space is
   `Box(-1e6, 1e6, shape, float32)` and every `reset()`/`step()` observation is clipped into it;
   `gymnasium.utils.env_checker.check_env` emits **zero** warnings (test asserts via
   `warnings.catch_warnings(record=True)`).
6. **Zero pytest warnings** — `StarletteDeprecationWarning` (upstream TestClient/httpx) is filtered
   in `[tool.pytest.ini_options].filterwarnings`; the suite reports `1144 passed` with no warnings.

**Frontend redesign** — `frontend/` was rebuilt from scratch as an institutional quant workstation
(Linear / Databento Portal / TradingView Pro / Bloomberg BQuant inspiration):
- Full-viewport docked shell (`w-screen h-screen overflow-hidden`): full-width h-12 ribbon with brand,
  horizontal view tabs and an always-visible account ticker (NLV, Day PnL, margin vs 60% cap, gross
  leverage, data-mode pill), compact two-step `Flatten All`, 12-col full-width main grid and an h-7
  status bar with collapsible telemetry drawer.
- Typography: Inter for all UI text, JetBrains Mono (tabular) strictly for numbers/tickers; exact
  zinc/blue/emerald/rose/amber token palette; 13px Title-Case panel titles, 11px uppercase table
  headers, 22px mono KPI values; no CRT scanlines, neon glows, hazard stripes, centered `max-w`
  containers or static "textbook" filler cards.
- Five rebuilt views: Command Center (dense KPIs, equity + underwater chart with timeframes and
  crosshair, risk budget bullets, asset-class exposure, sortable positions/orders/fills/telemetry
  tabs with `w_i` bars and roll countdowns); Pipeline & Weights (grouped long/short/net stage chart,
  execution trace, per-root transformation heatmap matrix); Risk & LLM Monitor (`κ`/`τ`/lockout
  cards, per-root `S_i,t` meters, news feed, macro calendar, interactive `PUT /api/risk/thresholds`
  form); Term Structure (root strip, F1–F6 curve with price/normalized/basis, roll schedule + BAG
  queue, multi-root matrix); Backtest Lab (compact experiment runner, run history/compare, synced
  equity/underwater/Sharpe-attribution charts and grouped institutional metrics with winners).
- All REST/WebSocket bindings preserved; deterministic DEMO DATA fallback with a visible mode pill;
  working-orders/fills render honest empty states where the API does not expose them.

**User guide** — new `docs/USER_GUIDE.md` (installation, full settings reference, Databento and IBKR
setup, dashboard, backtests/walk-forward, live runner, risk operations, Docker, troubleshooting).

**Verification (as merged)**
- `FUTURESRLX_ENVIRONMENT=paper FUTURESRLX_RISK_STATE_STORE_URI=duckdb: ./scripts/quality_gate.sh --cov`
  → ruff/format/mypy clean, **1144 passed, 0 warnings**, coverage `src/core`+`src/risk` **93.97%**.
- `python -W error::RuntimeWarning -m src.backtest.engine --help` and
  `... -m src.execution.runner --check` → exit 0, empty stderr.
- CLI proofs: empty-cache/no-source runner → exit 2 with actionable hints (no traceback);
  `runner --offline --max-states 2` → exit 0 and hydrated JSON status.
- Frontend: `npx tsc --noEmit` clean, `npm run lint` clean, `npm run build` success (8/8 pages),
  production `npm run start` → all five routes HTTP 200, anti-slop/filler/legacy-palette greps empty.

---

## Phase 9 — Live bridge, DRL training & UI telemetry polish (`v1.3.0`)

**Branch:** `feat/phase-9-live-bridge-and-drl`

**Part A — five UI / telemetry data-binding fixes**
1. **Equity chart timeframe vs live ticks** — new pure helper `frontend/lib/equity-series.ts`
   (`mergeTelemetryIntoSeries`): `1D` appends/merges intraday `/ws/telemetry` points; daily timeframes
   (`1W/1M/3M/6M/1Y/ALL`) update the last daily bar in place and never append intraday timestamps
   (underwater/drawdown derived from the same window). Verified headless: 3M keeps 63 daily points
   with a stable last timestamp while the value updates in place.
2. **Discretization tracking error** — `ContractTranslator.discretization_report()` computes
   `tracking_error = Σ|w_target − w_actual|/2` with `w_actual = N·P^USD·M / E^USD` (FX-aware) plus
   per-family `MicroRoute` summaries; `/api/pipeline/latest` exposes `tracking_error_pct`,
   `micro_routing[]`, `tracking_error_source` and per-instrument `weight_actual`, and the Pipeline
   view renders the real percentage/micro-routing (no more `n/a`). Weight matrix re-laid out with
   fixed fractional columns: no horizontal scrollbar, DRIVER fully visible at 1440/1920.
3. **Backtest Lab** — all checked runs (2–3) overlay on equity (rebased to 100) and underwater
   charts with distinct colours + legend; TRADES/REBAL populated from real `n_trades`/`n_rebalances`;
   Walk-Forward checkbox + splits (2–20) + purge/embargo inputs added and forwarded to
   `POST /api/backtests`; walk-forward runs show `kind="walk_forward"` + `n_folds` badges.
4. **Persisted backtest runs** — API-triggered runs (single-pass + walk-forward) are written
   atomically to `artifacts/backtests/<id>.json` (summary + tearsheet + request), reloaded on API
   startup (verified across an app restart), corrupt files quarantined, seeded demo runs kept in
   memory only.
5. **Three-state data-mode badge** — `UI FALLBACK` (amber, backend unreachable), `OFFLINE ENGINE`
   (zinc/blue, `offline_mode=true`), `PAPER · IBKR` (blue) / `LIVE · IBKR` (emerald) from
   `ibkr_connected` + environment; no `DEMO DATA` remains.

**Part B — core platform capabilities**
1. **Live runner ↔ API bridge** — `src/execution/runner_bridge.py` (versioned, atomic
   `data/bridge/runner_state.json` snapshot: mode/sequence/status/account/positions/working
   orders/fills/pipeline, 200-entry caps, 15 s staleness) written by `TradingRunner` after every
   processed state; `OfflineRunnerSession` exposes deterministic `warm_start/step/run/snapshot/close`.
   API adds `GET /api/orders/working`, `GET /api/fills/recent`, `GET /api/runner/status`,
   `POST /api/runner/step`, `POST /api/runner/reset`, and bridge-first `/api/positions` +
   `/api/pipeline/latest` (`source="runner-bridge"`), with honest simulated/fixture fallbacks.
2. **DRL training CLI** — `python -m src.strategies.train_drl --strategy ... --timesteps N
   --output artifacts/models/<name>.zip --offline` trains PPO/CPPO/SAC/TD3/A2C on
   `FuturesTradingEnv` from deterministic offline data, saves model + `<checkpoint>.meta.json`
   (algorithm/policy/timesteps/seed/device/features/obs bounds/universe/SB3 version/git commit),
   with actionable exit-2 errors. `DRLAllocator(model_path=...)` lazily loads checkpoints and
   `build_pipeline` honours `drl.model_path` (explicit allocator params win).
3. **Pluggable live news + LLM provider switch** — `src/data/news_ingestor.py` resolves
   RSS/Atom feeds → IBKR news → deterministic fixtures (root/severity/sentiment classification,
   dedupe, UTC timestamps, injectable fetcher/IBKR client/clock). New
   `FUTURESRLX_LLM_PROVIDER` (`offline|openai|anthropic|deepseek|vllm`), `LLM_MODEL`,
   `LLM_API_KEY`, optional `LLM_BASE_URL` drive `src.risk.llm_clients.build_llm_client`; the API's
   LLM overlay uses the real client when configured, and the async TTL cache keeps slow providers
   off the request path. News feeds/interval configured via `FUTURESRLX_NEWS_*`.
4. **E2E bug fixed during integration** — default API CORS origins now include the dashboard
   `:3100` and compose `:3080` origins (previously a browser at `127.0.0.1:3100` was blocked and
   showed `UI FALLBACK` even with the API up).
5. **Post-release follow-ups** — Google **Gemini** is now a first-class LLM provider
   (`GeminiClient` on Gemini's OpenAI-compatible endpoint, `FUTURESRLX_LLM_PROVIDER=gemini`,
   key required, verified live against the Generative Language API); unknown provider ids now fail
   with an actionable supported-provider message; keyless local `vllm` is supported. Subprocess CLI
   tests now run from a temp cwd so a developer's real repo-root `.env` can never leak into the
   hermetic test suite.
6. **API ↔ IBKR attachment and live ledger** — new `FUTURESRLX_API_ENABLE_IBKR` setting (honored from
   `.env`, the legacy env flag still works) makes the FastAPI backend own a real paper/live session,
   so `/health` reports `ibkr_connected=true` and the ribbon shows `PAPER · IBKR` (the badge also
   recognises a fresh paper/live runner bridge). The `ib_async` adapter now subscribes to account
   updates with an `accountSummary` fallback, is currency-aware for non-USD base accounts
   (`FUTURESRLX_ACCOUNT_CURRENCY` / `FX_RATES`), and `/api/positions` degrades to a 503 with
   actionable text instead of a raw 500. The dashboard allows a longer first account fetch and
   refreshes every 15 s so a slow ledger never leaves fixture values on screen. WSL2 NAT/mirrored
   setup is documented (and either Windows-host-IP or mirrored `127.0.0.1` works; verified live
   against TWS paper with EUR base account).

**Verification (as merged)**
- `FUTURESRLX_ENVIRONMENT=paper FUTURESRLX_RISK_STATE_STORE_URI=duckdb: ./scripts/quality_gate.sh --cov`
  → ruff/format/mypy clean, **1316 passed, 0 warnings**, coverage `src/core`+`src/risk` **94.33%**
  (translator 100%, llm_clients 100%, quant overlay 100%).
- Headless Chromium E2E against the real Phase-9 API: `UI FALLBACK`/`OFFLINE ENGINE`/`PAPER·IBKR`/
  `LIVE·IBKR` badges, real `tracking_error 9.78%` + micro-routing, 3-run overlays, WF badge,
  no matrix horizontal scroll at 1440/1920, 3 working-order rows + 4 fill rows rendered from the
  bridge payload, **zero page errors**; contract-render pass also green.
- Live backend smoke: `POST /api/runner/step` → bridge written (`sequence=3`, `mode=offline`),
  `source="runner-bridge"` on orders/status; `POST /api/backtests` with `walk_forward=true` →
  `kind="walk_forward"`, `n_folds=3`, `n_trades=35`; persisted `bt-0004.json` survived a uvicorn
  restart and its tearsheet exposed 3 folds.
- Frontend: `npx tsc --noEmit` clean, `npm run lint` zero warnings/errors, `npm run build` success
  (8/8 pages), production routes `/ /pipeline /risk /term-structure /backtests` all HTTP 200.


---

## Phase 10 — Strategy Studio, DRL trainer & Optuna optimizer (`v1.4.0`)

**Branch:** `feat/phase-10-strategy-studio-optuna`

### Part A — closed Phase-9 gaps
1. **Full per-instrument runner-bridge rows** — `runner_bridge.py` publishes `instruments[]`
   (`instrument_id/root/asset_class/selected/weight_base/weight_timing/weight_risk/weight_final/
   weight_actual/delta_timing/delta_risk/target_contracts/notional/driver/reason/regime`),
   `micro_routing[]`, `tracking_error_pct`, `tracking_error_source` (schema version stays 1, additive
   and backwards compatible). `/api/pipeline/latest` renders the Weight matrix identically when
   `source="runner-bridge"` via `src/api/bridge_pipeline.py`: parity-tested field-by-field against the
   engine waterfall; old snapshots without rows degrade to a compact bridge summary (no fabrication).
2. **Offline term structure/carry** — new `src/data/term_structure.py` + `OfflineBarSource` emits a
   deterministic roll-aware F1+F2 pair per root/timestamp (contango/backwardation carry profiles +
   calendar regime cycle; `term_structure=False` keeps legacy behaviour). `train_drl.build_training_data`
   now reports `feature_fallbacks == ()`, finite non-zero `term_slope`/`carry_annualized`
   (sign-opposite, two-sided per trained root at 512 bars; seed-independent). Hydrator coverage now
   counts distinct decision timestamps (multi-expiration history fix); offline orders end-to-end is
   tested via a temp strategy without the carry filter (the default carry filter correctly flattens an
   all-contango universe).
3. **Walk-forward fold badge** — Backtest Lab Run History binds `WF · N folds` from the actual
   `n_folds`/`wf_splits` returned by `/api/backtests` (no more `—`).

### Part B — Strategy Studio, optimizer, DRL API and UI
1. **Timing & selector expansion** — `MeanReversionTiming` (Wilder RSI + Bollinger fade, turbulence
   veto, reversion strength; RSI/Bollinger helpers) and `SeasonalityTiming` (UTC-month root bias
   curves, active months, out-of-season scaling) registered in `TIMING_REGISTRY` and YAML-selectable;
   5 hypothesis presets. New `src/strategies/parameter_catalog.py`: 121 parameters across
   universe/allocator/timing/risk/translator, 104 Optuna-tunable, full metadata (plain-English
   description, type, default/bounds/step, yaml_encoding, location) with validation/encoding/binding
   helpers and `GET /api/strategies/schema/catalog`.
2. **Walk-forward Optuna optimizer** — `src/backtest/optimizer.py::WalkForwardOptimizer` with optuna
   (+ deterministic Latin-Hypercube fallback), strict OOS-only trial evaluation through the public
   purged/embargoed `run_walk_forward` (default `walk_forward=True`), drawdown penalty, trial
   summaries/importances/best config, atomic `artifacts/optimizations/<study_id>.json` persistence,
   reload on API startup, CLI. API: `POST/GET /api/optimize`, detail, `POST /{id}/apply` (saves tuned
   YAML and runs a verification backtest via the refreshed in-memory catalog).
3. **FastAPI endpoints** — Strategy Studio CRUD (`GET /api/strategies/schema/catalog`,
   `POST /api/strategies`, `PUT /api/strategies/{id}` with safe path validation/immutable built-ins in
   test envs), optimization endpoints, DRL `GET /api/drl/models` + `POST /api/drl/train` (bounded,
   thread-pooled, algorithm/hyperparameter validation, reward curve + metadata), bridge pipeline
   parity. Routers registered by the Lead; new services are lazy (`phase10_deps.py`), no Container
   rewrites.
4. **Strategy Studio UI** — new 6th ribbon tab `/studio` (between Term Structure and Backtest Lab)
   with Manual Builder (5-stage accordion, presets, live YAML + validation), Optuna Auto-Optimizer
   (history chart/best-so-far, parameter-importance bars, sortable trials, apply), DRL Neural Trainer
   (training form, checkpoint registry, reward curve/meta inspector, Bind to Strategy), plus the
   `WF · N folds` fix and the new `IBKR · RECONNECTING` badge state.
5. **IBKR resilience hotfix (Phase-9 bug reports)** — dynamic reconnect after TWS restarts
   (`ensure_connected`, heartbeat one bound attempt/tick, 5 s liveness probe, request-path
   invalidate), dynamic health (`ibkr_enabled/ibkr_reconnecting/ibkr_last_error`, offline_mode
   derived), never-simulated book (last-known-good account/positions with `stale/stale_reason`,
   explicit 503 before first live book, telemetry never publishes simulated positions, flatten returns
   409 when unreachable), and additive frontend `IBKR · RECONNECTING` state. Hermetic conftest now
   strips env and disables `.env` at import time (module-scoped fixtures included).

### Verification (as merged)
- Full quality gate with hostile env (`FUTURESRLX_ENVIRONMENT=paper
  FUTURESRLX_RISK_STATE_STORE_URI=duckdb: ./scripts/quality_gate.sh --cov`):
  ruff/format/mypy strict clean, **1553 passed, 0 warnings**, coverage `src/core`+`src/risk`
  **94.24%** (translator 100%, quant risk 100%, llm risk 99%).
- API: 49 tests across `test_api/test_api_ws/test_phase9_api/test_phase10_api`, 20 DRL API tests,
  2 IBKR resilience end-to-end tests (startup-down 503 → TWS returns → live MES qty; flap → stale
  last-good → heal); catalog source `src.strategies.parameter_catalog` (121 params/5 presets);
  real optimizer + apply + verification backtest; bridge matrix parity.
- A2: `tests/integration/test_phase10_offline_drl.py` 9 tests — empty fallbacks, non-zero two-sided
  term/carry per root, deterministic, finite env bounds, tiny PPO smoke.
- Frontend: `npx tsc --noEmit` clean, `npm run lint` zero warnings, `npm run build` success, routes
  `/studio` + existing pages HTTP 200, headless `/studio` render with no page errors and live
  studio API calls (catalog/optimizer/DRL).
