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


### Phase 10 post-release hotfix — `v1.4.1`

- **IBKR ledger after TWS reconnects** (`fix/ibkr-ledger-reconnect`): TWS can keep a stale
  "accepted" API session for a client id, so reconnects either fail with "client id already in
  use" or stream only static tags without `NetLiquidation`. `IBKRConnectionManager` now rotates the
  client id within a bounded 8-id window on every connect attempt, the `ib_async` adapter serialises
  ledger reads and uses `accountSummary` as the primary request/response source (streaming account
  updates remain the fallback), and `IBKRExecutionService` serialises book refreshes so telemetry,
  routes and the liveness probe cannot race the same client.
- **Ledger currency auto-detection**: `fetch_account` no longer drops rows whose currency differs
  from `FUTURESRLX_ACCOUNT_CURRENCY`; it uses the broker's actual ledger currency (with a warning)
  so a EUR-base paper account still works when the config says USD. The operator's `.env` was
  corrected to `FUTURESRLX_ACCOUNT_CURRENCY=EUR` + `FX_RATES={"EUR":1.1216}`.
- Verified live against TWS paper (MES 1 position): `/health` connected, `/api/positions` 200 with
  `NetLiquidation ≈ €1,003,691` and MES @ 7568.62, dashboard `PAPER · IBKR` with the position
  visible and no "account snapshot unavailable"; full gate 1555 passed / 0 warnings.


---

## Studio UX overhaul & 45-root futures universe (`v1.5.0`)

**Branch:** `feat/studio-ux-universe`

- **Universe expansion** — `configs/instruments.yaml` now declares 45+ liquid CME Globex roots across
  seven sectors (Equity incl. NKD, Rates incl. SR3/TN/UB, Energy RB/HO, Metals PL/PA, six FX pairs,
  seven agri roots, Crypto MBT/MET) with realistic multipliers, tick sizes, margin estimates, roll
  calendars and standard↔micro links. `OfflineBarSource` synthesises roll-aware F1/F2 bars and
  sector-specific carry/volumes for every root, deterministic and vendor-free. New
  `GET /api/instruments` exposes the sector-grouped master (root/name/asset class/exchange/
  multiplier/tick/micro pair) for the Studio and external tooling.
- **Parameter catalog** — universe multiselects and sector groups are derived from the instrument
  master (no hardcoded 13-root lists); the catalog exposes sector-grouped universe metadata plus
  selector (liquidity/regime/composite) and allocator (volatility_target/risk_parity/equal_weight/
  mean_variance/drl) choices for the new Studio dropdowns.
- **Manual Builder redesign** — five-step stage switcher (Universe → Allocator → Timing → Risk →
  Translator) replaces the stacked wall; the live-YAML pane is sticky with internal scrolling and an
  anchored η_max footer. Stage 1 groups 45+ roots by sector tabs with Select All/Clear; Stage 1/2
  expose selector/allocator dropdowns; every timing overlay (TSMOM, KAMA, carry, mean reversion,
  seasonality) and risk overlay (quant risk, LLM risk) has a real enable/disable toggle that adds or
  removes the YAML stage; numeric parameters use a slider **plus** direct numeric input with formatted
  values; advanced math parameters (covariance ridge/shrinkage, vol floors, KAMA fallback band, micro
  dust floor) are collapsed under *Advanced Parameters*. Stage-2 sizing labels are separated from
  Stage-4 hard caps with a **Sync with Allocator** control.
- **Verification** — full hostile-env quality gate: **1622 passed, 0 failures, 0 warnings**, mypy
  strict clean on 103 source files, coverage `src/core`+`src/risk` **94.24%**. New
  `tests/integration/test_phase10_1_universe.py` (18 tests, 0 skips) covers all 45 roots end-to-end:
  config integrity, ContractTranslator long/short/micro-routing/margin<=60% for every root, paired
  standard/micro mark equality (all 9 families, F1+F2), offline F1/F2 + finite non-zero term/carry
  for every root, ComposablePipeline/selectors/backtest/walk-forward with finite metrics, DRL
  training data covering the full universe with `feature_fallbacks == ()`, `/api/instruments`
  (45 roots, 7 Studio sector lanes, bidirectional micro links) and catalog `universe`/`stage_choices`.
  The DRL offline data cache makes repeated 45-root builds near-free (cold 15.9 s → 0.0005 s hit;
  same outputs/fallbacks). Frontend: `tsc --noEmit` clean, `eslint` 0 warnings, `npm run build`
  success (9/9 pages, `/studio` 18.1 kB), routes 200; headless live checks at 1920×1080 and 1440×900
  show one stage card at a time, sticky YAML + anchored η_max footer, 8 sector tabs/45 root pills,
  working overlay toggles, slider+numeric inputs, advanced collapse, no page-level scroll,
  `PAPER · IBKR` badge and zero page errors. Live API: `/api/instruments` count 45 with sector counts
  9/7/6/8/6/7/2, catalog universe 8 groups, `/api/positions` 200 with the live MES book.

---

## v1.5.1 — surgical Studio/Optuna/IBKR attribution fixes

**Branch:** `fix/studio-optuna-ibkr-attribution`

1. **Optuna search-space hygiene.** `ParameterSpec.optuna_default_selected` (exactly 6 curated core
   parameters: TSMOM fast/slow, KAMA window, carry min-carry-bps, volatility-target target vol,
   quant-risk trailing stop) plus a 25-key structural/data-hygiene non-tunable policy. New catalog
   helpers (`active_parameter_keys`, `parameters_for_strategy`, `default_selected_parameter_keys`)
   filter to the selected universe selector/allocator, enabled timing overlays and enabled risk
   overlays. The Studio lists only active tunables for the selected base strategy, pre-checks the
   curated defaults (4–6), renders hygiene parameters as read-only structural rows, and warns above
   eight checked parameters; the optimizer's `default_search_space` is active-only (6 paths for
   TSMOM, mean-reversion-only for the panic preset) and `filter_search_space_for_config` drops
   inactive/hygiene paths from explicit UI requests.
2. **Live IBKR risk/equity attribution.** `LiveEquityHistoryStore` persists real observed NLV points
   to `data/ibkr_equity_history.json` (atomic, deduped, bounded 5000 points/30 days, corrupt
   quarantine). `/api/positions` additively reports `equity_curve_source`
   (`ibkr-live|simulated-backfill|fixture`), `equity_curve_live_points`, `portfolio_volatility` and
   `portfolio_volatility_source`. When IBKR is connected the risk budget uses ex-ante sqrt(w'Σw) from
   the live book (0.0 for a flat book, `—` when covariance is unavailable) and never the fixture
   curve; the Command Center badges simulated backfill vs live NLV points.
3. **Preset/config root validation.** Unit tests validate every preset template and committed strategy
   YAML root reference against `configs/instruments.yaml`; no delisted/non-canonical tokens (GE, ZQ,
   KE, BTC) are present or accepted.

**Verification** — hostile-env `./scripts/quality_gate.sh --cov` green (**1679 passed, 0 failures, 0 warnings**, coverage 94.24%),
frontend `tsc --noEmit`/`lint`/`build` clean, independent `tests/integration/test_phase10_2_v151.py`
(24 tests) plus catalog/optimizer/API acceptance suites green; live API restarted with v1.5.1 and
Command Center/Optuna verified against the real endpoints.


---

## v1.5.2 — state-sync, equity provenance & legacy optimization cleanup

1. **WS/REST IBKR parity.** `TelemetryBroadcaster` uses an injectable account provider wired to
   `LiveStateService.account`, so `pnl`/`margin`/`positions` frames carry the same enriched IBKR
   book as `GET /api/positions`; IBKR frames carry no demo jitter and publish actual
   margin/utilization/`portfolio_volatility`/`open_positions` (flat book = 0.0). Payloads gained
   additive attribution fields; no honest book still emits heartbeat only. Offline behaviour is
   unchanged.
2. **Distinct-day equity provenance.** `LiveEquityHistoryStore` now exposes distinct UTC days and
   intraday points. `equity_curve_source="ibkr-live"` requires **≥5 distinct UTC days** (not merely
   5 points); <5 days keeps the synthetic backfill badged as such, while `equity_curve_live_days` and
   `equity_curve_intraday` let the frontend plot real intraday points on `1D`. Live mode plots only
   real recorded points (no synthetic prefix).
3. **Legacy optimization purge.** New `validate_persisted_optimization` in the optimizer flags old
   studies whose search space/params include inactive overlays or structural/hygiene paths. On
   startup the API validates every persisted artifact, moves legacy/invalid studies to
   `artifacts/optimizations/legacy/`, excludes them from list/detail/apply/default selection, and
   remains idempotent/corrupt-safe; clean studies stay byte-canonical. The Studio also hides legacy
   ids client-side, so the Optuna pane auto-selects a clean active-space study.
4. **Frontend merge/equity guards.** While IBKR is connected the AccountProvider accepts only
   IBKR-sourced telemetry frames (ignoring synthetic-looking margin/vol/NLV values), keeps REST
   `portfolio_volatility`, and renders timeframe-aware "Live IBKR NLV"/"Simulated Backfill" badges
   with intraday support on `1D`.

**Verification** — hostile-env `./scripts/quality_gate.sh --cov`: **1710 passed, 0 failures,
0 warnings**, coverage 94.24%, mypy strict 103 files; independent v1.5.2 verification suite
(16 tests) and updated v1.5.1 acceptance suite (24 tests) green; frontend `tsc`/`lint`/`build`
clean; live API smoke + headless checks confirm flat IBKR REST and WS parity, distinct-day source
switching, and legacy-study quarantine.


---

## v1.5.3 — Optuna integrity (no flat winner) and visible apply

1. **Zero-trade/flat-trial protection.** `OptimizationRequest` gained `min_trades=3` and
   `min_avg_gross_exposure=0.01`; `TrialSummary` records `n_trades`, `avg_gross_exposure` and
   `zero_activity`. Trials below either threshold stay COMPLETE but receive objective `-10.0` and
   are excluded from best-trial selection; an all-flat study persists with no best trial
   (`best_trial_number=None`, empty params) instead of crowning a 0-trade run. Walk-forward offline
   windows are extended to ≥750 bars (default TSMOM ~1,265–1,514) with full warm-up across the
   train→test boundary.
2. **Carry tuning hygiene.** `carry_filter.min_carry_bps` default −50 bps, bounds [−200, +50],
   step 10 bps; removed from the curated `optuna_default_selected` set (now exactly five: TSMOM
   fast/slow, KAMA window, volatility-target target vol, quant trailing stop), while remaining
   tunable for explicit search spaces.
3. **Apply/save visibility.** In development/paper/live, `POST /api/strategies` and optimizer apply
   write `configs/strategies/<id>.yaml` (test/backtest keep artifact overlays); `GET /api/strategies`
   merges both trees and the in-memory catalog is upserted immediately. Optimizer apply now always
   triggers and persists a verification walk-forward backtest and additively returns
   `saved_strategy_id`, `strategy_path`, `backtest_run_id`. The Studio adds a target-ID apply modal,
   “Saving & Running Backtest...” disabled spinner state, emerald confirmation banner with Backtest
   Lab/Manual Builder deep links, immediate strategy-list refresh, inline 409/422/404 handling, and
   flat-study “No active trial” UI.
4. **Injectable-default apply fix.** Real optimizer studies can tune catalog-default parameters that
   the committed YAML omits (e.g. `risk.quant_risk.trailing_stop_pct`); `apply_dotted_params` now
   supports `allow_new_params=True` for missing final leaves inside stage `params` blocks, so applying
   a real TSMOM study no longer 422s while invalid paths still fail validation.

**Verification** — hostile-env `./scripts/quality_gate.sh --cov`: **1733 passed, 0 failures,
0 warnings**, coverage 94.24%, mypy strict 103 files; independent v1.5.3 verification (8 tests) plus
focused 210-test regression batch green; real optimizer run over 1,514 bars/root produced active
best trial (293 trades, 0.83 avg gross exposure) and forced-flat studies never selected a winner;
apply visibility confirmed end-to-end (strategy + verification run immediately in API). Frontend
`tsc`/`lint`/`build` clean; headless 1440/1920 apply modal/spinner/banner/list-refresh/5-default/
flat-study checks pass with zero page errors.


---

## v1.6.0 — QoL, artifact management, exports and operator controls

- **IBKR conId fix.** `IBKRMarketDataStreamer` qualifies every contract via
  `qualifyContractsAsync` before real-time bars/tick-by-tick/L1/L2 subscriptions and uses the
  conId-bearing contract for subscriptions/cancellation/events; unresolved roots raise actionable
  errors naming root/exchange (and are skipped in multi-root loops) instead of the `Contract ... can't
  be hashed` ValueError. Fake/legacy clients without qualification keep working; reconnect
  re-qualifies on the new client.
- **Artifact management.** DELETE endpoints for persisted optimization studies, backtests and
  custom/tuned strategies (built-ins protected, 404/409 semantics); strategy Clone/Rename/YAML
  download; durable `artifacts/strategies/.custom_ids.json` registry so custom strategies remain
  deletable/renamable across API restarts.
- **Backtest sharing & diffing.** Per-run `strategy_config` snapshots and `trades` persisted in
  summaries/tearsheets/artifacts; `GET /api/backtests/{id}/export.json` and `/trades.csv`
  attachments; comparison Parameter Diff consumes snapshots (legacy runs show an honest
  unavailable note).
- **Operator safety.** Additive risk-state kill-switch fields; guarded `POST /api/risk/reset-lock`
  (exact confirmation phrase + optional operator token/header) clears execution and persisted
  kill-switch/daily/HWM state while preserving OMS rate-limits; paper/backtest-only
  `POST /api/maintenance/reset-paper-state` clears the runner bridge, persisted risk store and paper
  session safely with receipts; live environments are refused.
- **Studio/Backtest/Command Center UX.** Delete/confirm actions and list refreshes; inline Clone/
  Rename/Download; Apply/Save success toast with direct `?run=` tear-sheet deep link; global Universe
  search across all sectors; Backtest export dropdown; parameter-diff accordion; guarded risk-lock
  reset modal; operator drawer (clear paper state / reset bridge). Studio-saved explicit
  `universe.params.roots` are honoured by liquidity/regime/composite selectors (unknown roots raise,
  state-absent roots intersect).
- **Verification** — hostile-env quality gate green with 0 warnings/94 %+ coverage; independent QoL
  verification (artifact delete/clone/rename/exports/diff/reset + conId qualification), focused API
  suites and frontend `tsc`/`lint`/`build` + headless 1440/1920 interaction checks.


---

## v1.7.0 — Phase 11A: IBKR paper execution without tick subscriptions

- **Front-month qualification.** `src/data/ibkr_contracts.py` resolves the active contract month from
  the instrument master roll rules (`contract_months`, `days_before_fnd`), builds an explicit
  `Future(..., lastTradeDateOrContractMonth=YYYYMM, includeExpired=False)` and qualifies it via
  `reqContractDetailsAsync`/`qualifyContractsAsync`, enforcing `conId > 0` and caching per
  `(root, month)`. The market-data streamer and historical fetcher never pass bare futures to IBKR.
- **Zero-subscription data fallback.** `FUTURESRLX_IBKR_MARKET_DATA_TYPE` (default 3) is requested
  with `reqMarketDataType` on connect. Paid tick/L1/L2 permission failures (10189/10090/354, sync or
  async) cancel the affected root's paid subscriptions, mark it degraded, and poll
  historical/delayed bars into the existing ingest path; diagnostics expose fallback roots/reasons and
  qualified conIds. All-roots qualification failure remains an actionable aggregate error.
- **One-shot rebalance.** `TradingRunner.run_rebalance_once()` + CLI `--rebalance-once` hydrate the
  lookback, connect the broker/OMS only (no market-data stream), enrich the latest state with live
  NLV/currency/FX and broker positions, run `S→A→T→R→N`, gate delta orders through
  `PreTradeRiskGate`/`whatIfOrder`, submit approved paper orders, publish the runner bridge snapshot
  and exit cleanly. Dry-run reports without submitting.
- **Dashboard rebalance.** `POST /api/runner/rebalance-preview` returns account/positions/targets,
  delta orders with per-order gate status/reasons, projected margin vs the 60% cap, tracking error and
  `mode=ibkr-paper|offline` without submitting; `POST /api/runner/rebalance-execute` submits approved
  deltas via the connected paper OMS or simulated broker, is idempotent on repeat, updates the bridge
  and refuses live/unsafe environments with 409. Command Center and Pipeline & Weights add a
  Rebalance Book bar + preview/execute modal with refresh of positions/orders/fills.
- **DRL universe guard.** Checkpoint `.meta.json` metadata is loaded (universe, feature order,
  observation/action shape); observations/actions align to the trained root order (missing roots flat,
  extra roots ignored) with actionable dimension errors; Bind to Strategy syncs the target
  `universe.params.roots`/`top_n` and surfaces a sync notice.
- **Verification** — front-month qualification/fallback, runner-once, API preview/execute and DRL
  alignment covered by unit + integration tests; hostile-env quality gate green with 0 warnings and
  94%+ coverage; frontend `tsc`/`lint`/`build` and headless 1440/1920 rebalance/DRL flows verified.


---

## v1.8.0 — Phase 11B: real-data cache manager, Databento cost guard and repo hygiene

- **Cache inventory.** `RootCacheSummary` + `ParquetRepository.inventory()` enumerate all 45
  instrument-master roots (empty included) with sector/schema/cached_bars/start_date/end_date/
  size_bytes/source; provenance sidecars record `databento`/`ibkr` (legacy partitions default
  databento; corrupt partitions degrade without hiding healthy roots).
- **Databento cost estimator and hard guard.** `CostEstimate` + `DatabentoLoader.estimate_cost_usd`
  call `metadata.get_cost(unit="usd")` and `metadata.get_billable_size` before any download.
  `FUTURESRLX_DATABENTO_MAX_COST_USD_PER_JOB` (default 2.00) is enforced inside `ingest`; an
  over-budget job raises an actionable budget error and never calls `get_range` unless the operator
  explicitly overrides (`allow_over_budget=True`, API requires confirm). Successful ingests write
  source provenance.
- **Data API and dashboard manager.** `GET /api/data/inventory` (totals, availability),
  `POST /api/data/estimate-cost` (IBKR $0 or Databento USD/MB), and `POST /api/data/ingest`
  (IBKR free historical/connected paper or guarded Databento, continuous F1/F2 summaries, provenance;
  409 budget/availability, 422 validation). Operator drawer/Backtest Lab expose a Market Data Cache
  modal with root/sector/date/source selection, cost check and guarded backfill.
- **Real vs synthetic data modes.** `data_mode=auto|cached_real|synthetic_offline` on backtests and
  optimization requests; cached real `ohlcv-1d` bars feed the real engine/walk-forward/
  `WalkForwardOptimizer(bars=...)`. Auto falls back synthetic; cached_real without sufficient data is
  an actionable error; synthetic_offline never reads the cache. Backtest/study responses and UI carry
  additive `data_origin` (`real_parquet`/`synthetic_fixture`), `data_mode` and `cached_bars`
  provenance, badged emerald vs zinc.
- **Repository hygiene.** `docs/screenshots/` now tracks only the four canonical screenshots linked by
  `README.md`/`docs/USER_GUIDE.md`; 21 transient verification PNGs were removed and
  `artifacts/screenshots/` is ignored for future headless runs.
- **Verification** — independent Phase 11B suite (inventory/provenance, cost guard/no-download,
  data API, cached-real backtest/Optuna); hostile-env quality gate green with 0 warnings/94%+
  coverage; frontend `tsc`/`lint`/`build` and headless 1440/1920 cache/badge/toggle flows verified.
