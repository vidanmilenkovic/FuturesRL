# Changelog

All notable changes to FuturesRL are documented in this file.

The format follows [Keep a Changelog 1.1](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Calendar dates are omitted:
the versions below are development milestones that were tagged in the project's private history;
the public release history starts with the current line.

## [Unreleased]

### Added

- Global LLM provider status pill and `GET /api/risk/llm-status` / `POST /api/risk/llm-ping`,
  with additive `/health.llm_status`, honest offline-stub/live/error states and latency.
- Inspectable 45-root scope for Term Structure, Pipeline & Weights and Risk & LLM
  (`scope=strategy|cached|all`, sector and explicit root filters) with cached-Parquet or
  deterministic instrument-master provenance.
- Non-destructive IBKR port discovery (`GET /api/runner/ibkr-ports`) for TWS/IB Gateway paper/live
  ports and a guarded connection-switch API that never claims an unapplied switch.
- Modular Strategy Studio composite stacks for timing/universe signals, allocator math tooltips,
  advanced/custom stage parameters and local custom presets.
- Term-structure forward-curve modes: normalized basis vs F1 (default), annualized roll yield and
  raw price, with minimum domains, a zero reference line and F6-F1 spread headers.

### Changed

- DRL support is focused on PPO and CPPO only; the experimental seasonality overlay was removed.
- Docker Compose now runs only the FastAPI backend, Next.js workstation and optional IB Gateway profile;
  TimescaleDB and Redis sidecars were removed in favour of local Parquet/Polars/DuckDB storage.

### Fixed

- Operator drawer and Market Data Cache no longer lose input focus or close native select dropdowns
  on telemetry/health polling ticks.
- Deterministic quantitative risk thresholds are now the primary Risk & LLM workspace, with the LLM
  overlay presented as an explicit one-way brake.
- API catalog responses now forward allocator formulas and advanced-parameter metadata to the Studio.

## [1.9.0]

### Added

- Eight canonical bar timeframes (`1m`, `5m`, `15m`, `30m`, `1h`, `4h`, `1d`, `1w`) with a
  dependency-free registry that defines bar duration, IBKR bar size, Databento base schema,
  trading bars per year and the finer base timeframe used for auto-resampling.
- Look-ahead-safe Polars resampler for canonical futures bars: UTC windows labelled at the right
  (close) boundary, OHLCV aggregation, last open interest and volume-weighted VWAP.
- Automatic cache resampling in `ParquetRepository.read_timeframe()`: 5m/15m/30m/1h are derived from
  cached 1m partitions, 4h from 1h and 1w from 1d, with optional persistence of the target schema.
- `BacktestSpec.timeframe` and `MarketState.timeframe`, plus timeframe-aware annualisation across
  volatility targeting, risk parity, TSMOM, the quant risk overlay, analytics and the backtest
  engine.
- Market-data cache API: `GET /api/data/inventory`, `POST /api/data/estimate-cost` and
  `POST /api/data/ingest`, with per-root provenance (`databento`, `ibkr`, `empty`), response
  details and timeframe selection.
- Real-data cache manager for backtests and optimizer studies: `data_mode="auto"`,
  `"cached_real"` and `"synthetic_offline"`, inclusive of timeframe-aware reads and explicit
  REAL/SYNTHETIC provenance in every run and study.
- Databento cost estimator and hard per-job budget guard (`$2.00` default) before any download, with
  an explicit operator override for intentional over-budget jobs.
- IBKR historical bar ingestion for the cache, using qualified front-month contracts and a dedicated
  client id so it cannot collide with the execution session.
- Timeframe selectors in the dashboard's Market Data Cache, Backtest Lab and Strategy Studio, plus a
  live-account NLV binding for the pipeline and term-structure state.

### Changed

- `1d` remains exactly 252 trading bars per year; intraday annualisation uses the registry value for
  the selected timeframe.
- Margin settlement and daily loss/kill-switch counters use UTC calendar-day boundaries for intraday
  bars instead of treating every bar as one day.
- Pipeline and term-structure state now prefer the real live account NLV and cached Parquet history
  when available, with explicit origin attribution instead of a fixture-implied equity.

### Fixed

- IBKR cache backfill returned a generic conflict error when the API was not attached to a TWS
  session; it now wires the active session and returns specific, actionable errors for budget
  refusal, a missing Databento key and an unreachable `host:port`.
- Dashboard ingest errors display the backend's exact detail rather than a hard-coded message.
- Repeated identical offline DRL training-data builds no longer pay the full 45-root feature cost
  (bounded, per-process cache with defensive copies).

## [1.8.0]

### Added

- Data-cache inventory and provenance APIs, plus scheduled backfill flows for IBKR historical and
  Databento pay-as-you-go data.
- Continuous F1/F2 roll series built from the cache, and data-mode selectors in the dashboard.
- Databento cost estimate and budget enforcement before ingestion.

### Changed

- Backtest and optimizer responses carry `data_origin` / cached-bar metadata so real and synthetic
  runs are distinguishable in both API and UI.
- Cache writes record source provenance in sidecar metadata, which survives restarts.

## [1.7.0]

### Added

- Qualified front-month contract resolution for IBKR market data, historical bars and order previews
  (`qualifyContractsAsync` / `reqContractDetailsAsync`, `conId > 0`, per-root/month caching).
- Zero-subscription paper execution: `python -m src.execution.runner --rebalance-once` hydrates,
  connects the broker/OMS only, enriches the latest state with live NLV/FX, runs the full pipeline,
  gates delta orders with `whatIfOrder`, submits approved orders and publishes the runner bridge.
- Rebalance Book controls in the dashboard with `Preview Rebalance (whatIf)` and
  `Execute Paper Rebalance`, showing current vs target contracts, deltas, margin versus the 60% cap
  and per-order pre-trade gate status.
- Delayed/paid market-data fallback for accounts without tick bundles: the streamer degrades the
  affected roots and polls historical/delayed bars instead of crashing on async permission errors.

### Changed

- IBKR market-data type is configurable (`FUTURESRL_IBKR_MARKET_DATA_TYPE`, default delayed).

### Fixed

- DRL checkpoint binding now reads the checkpoint metadata universe, syncs the target strategy's
  explicit roots/top-n and aligns observations/actions to the trained root order (padding missing
  roots flat and ignoring extras) rather than failing on shape mismatch.

## [1.6.0]

### Added

- Artifact management for persisted optimizer studies, backtest runs and custom strategies, with
  protected built-ins.
- Strategy Clone, Rename and Download-YAML actions; a durable custom-strategy registry.
- Backtest exports: trades CSV and full tear-sheet JSON, plus a persisted strategy-config snapshot
  used by the comparison parameter-diff view.
- Guarded operator resets: Reset Risk Lock / Resume Execution, Clear Paper State and Reset Runner
  Bridge, all double-confirmed and restricted to safe environments.
- Global 45-root Universe search in the Strategy Studio.

### Fixed

- IBKR contracts are qualified before tick/realtime/market-depth subscriptions; unresolved roots
  produce an actionable root/exchange message instead of a hash/`conId` traceback.
- Strategy selector supports Studio-saved explicit `roots` filters end-to-end.
- Apply/Save & Run shows a direct tear-sheet deep link, and the Backtest Lab consumes it.

## [1.5.3]

### Changed

- Walk-forward optimizer trials record out-of-sample trade count and average gross exposure;
  flat/zero-activity trials receive a hard objective penalty and can never be selected as best.
- Optimizer default windows generate enough offline history for long lookbacks and indicator
  warm-up across the train→test boundary.
- Carry floor defaults to −50 bps (bounds −200..+50) and is no longer pre-selected; curated Optuna
  defaults are the five core sizing/trend/risk parameters.
- Applying optimizer parameters persists the tuned strategy, reloads the catalog immediately and
  creates a verification walk-forward run with a persisted run id.

### Fixed

- Flat trials no longer win merely because their drawdown is small.
- Applied strategies and verification runs are visible in the Strategy Studio and Backtest Lab
  without a restart.

## [1.5.2]

### Added

- Distinct UTC-day equity provenance for IBKR: the multi-day chart switches to live NLV only after
  five distinct calendar days; the first-day view uses recorded intraday points.
- Startup validation and quarantine of legacy optimizer studies whose search spaces reference
  inactive overlays or structural parameters.

### Fixed

- Telemetry for positions, margin and PnL now sources the same enriched IBKR account as the
  REST positions endpoint, with no synthetic jitter and exact flat-book zeros.
- The dashboard ignores simulated-looking telemetry frames while IBKR is connected.
- Legacy studies are no longer auto-selected or applied.

## [1.5.1]

### Added

- Parameter catalog flags for Optuna tuning: structural/data-hygiene parameters are non-tunable,
  curated core parameters are pre-selected, and helpers expose active-stage-only parameter sets.
- Preset and strategy-configuration root lists are validated against the instrument master.
- Catalog payload exposes stage choices and dynamic universe metadata for the Studio.

### Fixed

- Live IBKR risk attribution uses ex-ante portfolio volatility from the actual book (0.0% when
  flat) instead of the synthetic fixture curve, with honest live/backfill provenance on charts.

## [1.5.0]

### Added

- Expanded instrument master to 45 roots across seven sectors with realistic multipliers, tick
  sizes, margins, roll calendars and standard↔micro links.
- Roll-aware offline synthetic bars for every root, enabling full-universe research without vendor
  credentials.
- `GET /api/instruments` sector-grouped universe metadata for the Studio and external tooling.
- Manual Strategy Builder redesign: five-step stage switcher, sector/root picker, overlay toggles,
  slider-plus-numeric inputs, advanced-parameter collapse, sticky YAML and anchored margin display.

### Changed

- Universe multiselects and sector groups are derived from the instrument master, so adding a root
  requires no catalog code changes.

## [1.4.1]

### Fixed

- IBKR ledger reconnection: client-id rotation within a bounded window, serialized ledger reads and
  an `accountSummary` primary path prevent stale "accepted" sessions from breaking reconnects.
- Ledger currency is detected from the broker instead of being silently dropped when it differs
  from the configured account currency.

## [1.4.0]

### Added

- Strategy Studio with Manual Builder, walk-forward optimizer and DRL trainer workflows.
- Parameter catalog (121 parameters across universe/allocator/timing/risk/translator) with typed
  metadata, validation, encoding helpers and preset templates.
- Purged/embargoed walk-forward optimizer with Optuna and a deterministic Latin-Hypercube fallback,
  out-of-sample scoring, drawdown penalty, trial history, parameter importances, apply flow and
  persisted studies.
- DRL trainer API and checkpoint registry: training, metadata sidecars, reward-curve inspection and
  strategy binding.
- Mean-reversion and seasonality timing overlays.
- Full per-instrument runner-bridge weight matrix and parity with the engine pipeline waterfall.
- Deterministic offline F1/F2 term structures so offline training and hydration see non-zero
  term slope and carry.

### Fixed

- Walk-forward fold badges bind the actual fold count returned by the API.
- IBKR reconnection resilience: reconnects after TWS restarts, reports reconnection state and never
  replaces a live book with simulated positions.

## [1.3.0]

### Added

- Live runner ↔ API bridge: atomic runner snapshots power positions, working orders, fills, runner
  status and a bridge-sourced pipeline waterfall.
- Persisted Backtest Lab runs and walk-forward runs across API restarts.
- DRL training CLI with checkpoint metadata and lazy model loading through strategy configuration.
- Pluggable news ingestion and LLM providers (OpenAI, Anthropic, DeepSeek, Gemini, vLLM/offline).
- Five-view dashboard telemetry polish: equity timeframe merging, discretization tracking error,
  multi-run overlays, walk-forward controls and three-state data-mode badges.

### Fixed

- Default API CORS origins include the local dashboard origin, preventing spurious UI-fallback
  states when the backend is healthy.
- API ↔ IBKR attachment exposes a real paper/live session and degrades positions to an actionable
  503 instead of a raw error when the ledger is unavailable.

## [1.2.0]

### Added

- Institutional dashboard redesign: full-viewport docked shell, dense KPI ribbon, trading-style
  typography and rebuilt Command Center, Pipeline & Weights, Risk & LLM Monitor, Term Structure and
  Backtest Lab views.
- Multi-source cold-start hydration with Databento, IBKR historical and offline fallback sources.
- Offline end-to-end runner and deterministic offline target broker for paper/dry-run workflows.
- Walk-forward backtest CLI flags (`--walk-forward`, splits, purge, embargo) with per-fold and
  aggregate reports.
- Lazy package initialization removes runtime warnings from module execution.
- Finite Gymnasium observation bounds, eliminating environment-checker warnings.

### Changed

- Test suite is hermetic: inherited environment variables and local `.env` files cannot leak into
  lint, type-check or test runs.

### Fixed

- CLI errors for unavailable data sources are actionable exit-code failures instead of tracebacks.
- Zero-warning test suite and zero-warning environment validation.

## [1.1.0]

### Added

- Cold-start state hydration that warms features and covariance before the first live bar.
- Durable risk and kill-switch state for JSON, DuckDB and Redis backends, surviving restarts and
  IBKR nightly resets.
- Open-order-aware reconciliation with configurable grace windows.
- Asynchronous cached LLM risk worker with single-flight refresh, cached fallback and capped backoff.
- Base-currency FX normalisation in the contract translator and market state.

### Fixed

- Non-USD accounts are sized correctly instead of being mis-sized by an implicit 1.0 FX rate.
- Slow LLM providers can no longer block the synchronous rebalance path or the WebSocket loop.
- Hydration ordering is enforced: the runner cannot process a live bar before a warm state exists.

## [1.0.0]

### Added

- FastAPI telemetry backend: positions, pipeline waterfall, risk state/thresholds/flatten-all, term
  structure and roll calendar, strategy catalog/editor, backtests/tearsheets and telemetry status,
  plus a WebSocket telemetry broadcaster.
- Next.js dashboard with five views (Command Center, Pipeline Inspector, Risk & LLM Monitor, Term
  Structure, Backtest Lab) and a typed API client with honest demo-data fallback.
- Docker Compose stack with backend, frontend, TimescaleDB, Redis and headless IB Gateway services.

### Security

- Offline/demo mode is explicit; the dashboard labels fixture data instead of presenting it as live.
- Kill-switch flatten-all requires no strategy cooperation and cannot be bypassed by pipeline code.

## [0.5.0]

### Added

- Async IBKR connection manager with heartbeat, exponential backoff, nightly reset handling and
  `whatIfOrder` margin checks.
- OMS with deterministic delta orders, mandatory pre-trade gate (fat finger, duplicate/rate limit,
  margin impact), bracket orders and emergency flatten-all.
- Native `BAG` calendar-spread roll execution with two-legged fallback.
- Internal-vs-broker reconciliation loop that halts automated rebalancing on drift.

### Security

- Pre-trade risk checks are mandatory and sit between strategy output and broker transmission.
- Live execution requires explicit environment opt-in and a live-port/paper-flag consistency check.

## [0.4.0]

### Added

- Vectorized and event-driven backtest engines sharing the broker interface used live.
- Margin simulator with mark-to-market settlement, maintenance-margin monitoring and forced
  liquidation with penalty.
- Tick-aware slippage, tiered IBKR/CME fee schedules and two-legged roll costs.
- Purged and embargoed walk-forward splits.
- Institutional analytics: CAGR, volatility, Sharpe, Sortino, Calmar, Omega, drawdown/duration,
  VaR/CVaR, Rachev ratio, return-on-margin, margin efficiency, implementation shortfall, turnover
  and spot-vs-roll attribution.
- JSON and standalone HTML tear sheets with run comparison.

### Fixed

- Accounting identity is asserted end-to-end:
  `Ending Equity = Starting Equity + Price PnL + Roll PnL − Commissions − Slippage`.

## [0.3.0]

### Added

- LLM Risk-First engine with strict structured outputs, confidence/variance anti-hallucination
  filtering, reward-shaping penalty and deterministic circuit breaker projection.
- Macro event calendar and Tier-1 lockout windows.
- Quantitative risk overlay: EWMA volatility targeting, gross/single-instrument/sector caps,
  high-water-mark trailing stops, UTC daily max-loss kill switch and drawdown limit.
- Editable macro calendar template.

### Security

- LLM output is treated as untrusted data and is only applied through deterministic, bounded risk
  projections.
- Risk overlay fails closed when required inputs are unavailable.

## [0.2.0]

### Added

- Composable strategy pipeline (`S_t → A_t → T_t → R_t`) with declarative YAML configuration and
  stage registries.
- Universe selectors: liquidity, adaptive slow/fast regime and composite.
- Allocators: equal weight, equal-risk-contribution risk parity, mean-variance, volatility target
  and DRL (PPO / CPPO / SAC / TD3 / A2C with lazy Stable-Baselines3 import).
- Timing overlays: TSMOM, KAMA, carry filter and composite.
- Gymnasium futures environment with margin dynamics, transaction costs and semantic-risk reward
  shaping.
- Deterministic `ContractTranslator` with rounding modes, micro routing and a hard margin ceiling;
  margin feasibility is property-tested.

### Security

- Contract translation is the only path from weights to contracts; no module sizes contracts ad hoc.

## [0.1.0]

### Added

- Project foundation: packaging, lint/format/type configuration, hermetic test setup, CI and
  developer scripts.
- Immutable domain contracts (`MarketState`, `WeightVector`, `ContractTarget(Set)`, `OrderEvent`,
  `Fill`, `Position`, `AccountSnapshot`, `RiskAssessment`, `PipelineResult`, `Bar`, `RollEvent`) with
  validated numeric invariants and read-only embedded data.
- Structural Protocols for selectors, allocators, timing overlays, risk overlays, data sources,
  broker adapters and reconcilers; shared asynchronous event bus.
- Typed settings with secret redaction and environment/paper/live consistency checks, plus the
  futures instrument master and declarative strategy configuration models.
- Canonical Polars/Parquet bar schema, partitioned repository with DuckDB queries, Databento loader,
  deterministic offline source, continuous roll engine, feature processor, point-in-time state
  builder and IBKR market-data streamer.
- Deterministic test fixtures and the first data-layer unit/integration suites.

### Security

- Secrets are wrapped so they cannot be printed accidentally, and settings redact them from
  serialized output.
- Naive timestamps, NaN weights, crossed quotes and negative margins are rejected at construction
  time.
