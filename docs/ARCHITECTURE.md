# Architecture

This document describes how FuturesRL is put together, which contracts every module relies on,
and where to extend it. It is the reference for engineers working on the codebase; operational
instructions live in [USER_GUIDE.md](USER_GUIDE.md).

## 1. Design goals

FuturesRL is research infrastructure for systematic futures strategies. The design favours:

1. **One interface between research and execution.** Every strategy stage produces or consumes the
   same signed target-weight vector; contract conversion happens exactly once, in a deterministic
   translator shared by backtest and live paths.
2. **Deterministic, offline-first verification.** The whole suite runs against seeded synthetic
   fixtures with injected adapters. Vendor SDKs and broker sockets are lazy, opt-in adapters.
3. **Honest data provenance.** Every payload states whether it is live, cached, delayed or synthetic;
   fixture data is never presented as a live book.
4. **Fail-closed risk.** The pre-trade gate, margin ceiling, kill switch and circuit breakers are
   deterministic and cannot be bypassed by strategy code.
5. **Point-in-time discipline.** Features, bars, resampling, folds and universe selection are
   computed only from data available at the decision timestamp; no look-ahead is permitted.
6. **Typed, declarative configuration.** Instruments and strategies are data (YAML/typed models),
   not executable code, so runs are reproducible and sweepable.

## 2. System map

```
                ┌──────────────────────────────────────────────────────────┐
                │                        Dashboard (Next.js)               │
                │  Overview · Pipeline · Risk · Term Structure · Lab │
                └──────────────────────────────┬───────────────────────────┘
                                               │ REST / WebSocket
                ┌──────────────────────────────▼───────────────────────────┐
                │              FastAPI telemetry and control API           │
                │  positions · pipeline · risk · strategies · backtests ·  │
                │  optimizer · DRL trainer · market-data cache             │
                └───────┬───────────────────────────────┬──────────────────┘
                        │                               │
        ┌───────────────▼──────────────┐   ┌────────────▼─────────────────┐
        │  Strategy pipeline (research)│   │  Live/paper execution engine │
        │  S_t → A_t → T_t → R_t       │   │  hydration · broker · OMS    │
        │  ContractTranslator          │   │  ContractTranslator          │
        └───────────────┬──────────────┘   └────────────┬─────────────────┘
                        │                               │
        ┌───────────────▼───────────────────────────────▼─────────────────┐
        │        Backtest engine (vectorized + event-driven) · analytics   │
        │        simulated broker · margin/cost simulators · walk-forward │
        └───────────────┬─────────────────────────────────────────────────┘
                        │
        ┌───────────────▼─────────────────────────────────────────────────┐
        │  Core contracts (`src/core`) · domain types · Protocols · bus    │
        └───────────────┬─────────────────────────────────────────────────┘
                        │
        ┌───────────────▼─────────────────────────────────────────────────┐
        │  Data layer: vendors, Parquet cache, roll engine, features,     │
        │  timeframes/resampling, point-in-time state builder             │
        └─────────────────────────────────────────────────────────────────┘
```

| Package | Responsibility |
| --- | --- |
| `src/core` | Immutable domain contracts, stage/broker Protocols, event bus, currency and FX helpers |
| `src/data` | Vendor adapters, cache, roll engine, features, timeframe registry/resampler, state builder, IBKR feed |
| `src/strategies` | Pipeline composer, selectors, allocators, timing overlays, Gymnasium environment, parameter catalog |
| `src/risk` | LLM Risk-First engine, quantitative overlay, contract translator, durable risk state |
| `src/backtest` | Vectorized/event-driven engines, margin and cost simulators, walk-forward, optimizer |
| `src/execution` | IBKR client, hydration, runner, OMS/pre-trade gate, roll execution, reconciliation |
| `src/analytics` | Metrics and tear sheets |
| `src/api` | FastAPI services, REST routes, WebSocket telemetry |
| `frontend` | Next.js dashboard |

## 3. Core contracts and invariants

`src/core/types.py` is the narrow waist of the system. Modules exchange only validated immutable
objects:

- `MarketState` — point-in-time market snapshot: timestamp, instrument snapshots (price, bid/ask,
  volume, open interest, features, margins), account equity and currency, FX rates, covariance
  matrix, per-instrument return series, regime label, metadata and the bar timeframe.
- `WeightVector` — signed target notional weights, keyed by root; validates finiteness and rejects
  unknown/duplicate instruments.
- `ContractTarget` / `ContractTargetSet` — concrete contract quantities with mark price, multiplier,
  margins and aggregate notional/margin totals.
- `OrderEvent`, `Fill`, `Position`, `AccountSnapshot`, `RiskAssessment`, `PipelineResult`, `Bar`,
  `RollEvent` — execution, risk and research records.

Invariants are enforced at construction time, not by convention:

- timestamps are timezone-aware and normalised to UTC; naive datetimes are rejected;
- prices, margins and multipliers are finite and positive where required; crossed quotes are
  rejected;
- weights are finite and cannot be silently mutated (embedded mappings/arrays are copied and made
  read-only);
- covariance matrices are symmetric positive-definite after regularisation, and their instrument
  order is explicit;
- a position/account record always identifies its instrument and currency.

`src/core/interfaces.py` defines structural `typing.Protocol`s for the pipeline stages, data sources,
broker adapters, risk providers and reconcilers. The backtest and live paths implement the same
Protocols; this is what makes parity testable.

`src/core/event_bus.py` provides the asynchronous pub/sub bus used identically by the live runner and
the event-driven backtester.

## 4. The strategy pipeline

The pipeline is a functional composition:

```
w_t = R_t( T_t( A_t( S_t( X_{≤t} ) ) ) )
```

`src/strategies/pipeline.py` builds the stages from a declarative strategy configuration and records
a per-stage trace (`PipelineResult`) with intermediate weights and diagnostics. Each stage has a
single responsibility:

### 4.1 Universe selection (`S_t`)

`src/strategies/selectors/` provides:

- **Liquidity selector** — filters roots by traded dollar volume, open interest and available return
  history, ranks deterministically by `(-dollar_volume, -open_interest, root)` and keeps `top_n`.
- **Regime selector** — combines a slow regime (cross-sectional trend breadth plus macro volatility)
  with a fast risk-off shock detector; in risk-off it excludes configured roots/sectors and can
  filter negative-trend roots without ever freezing the pipeline.
- **Composite selector** — deterministic intersection or union of components with an explicit order.

All selectors accept an optional explicit `roots` allow-list (the Strategies's saved universe
filter), validated against the instrument master and intersected with the point-in-time state.

### 4.2 Portfolio allocation (`A_t`)

`src/strategies/allocators/` provides equal weight, equal-risk-contribution risk parity,
mean-variance, volatility-targeted inverse-volatility and a DRL allocator (PPO/CPPO via
Stable-Baselines3, imported lazily). The DRL allocator loads checkpoints lazily, binds to the
checkpoint's metadata universe and aligns observations/actions to the trained root order.

### 4.3 Timing and term structure (`T_t`)

`src/strategies/timing/` provides TSMOM, KAMA, a carry/term-structure filter, mean reversion with a
turbulence veto, mean-reversion and a composite. Overlays are pure transforms of the weight vector:
they can scale or flatten existing legs but cannot introduce an instrument the allocator did not
select.

### 4.4 Risk overlays (`R_t`)

`src/risk/` applies the hard risk projection described in §6. The final weight vector is then
sanitised (finite, non-negligible, restricted to the selector universe) before translation.

Annualisation is timeframe-aware: `src/strategies/_numeric.py` derives the annualisation factor from
`MarketState.timeframe` via the canonical registry (`bars_per_year`). Daily states remain exactly
252 trading bars per year.

## 5. Futures contract translation

Futures are leveraged, expiring and discrete; cash-equity weight math does not transfer directly.
`src/risk/contract_translator.py` is the only module allowed to convert weights into contracts:

```
TargetNotional_i = w_i · E_t                          # E_t in USD (FX-normalised)
N_i^raw          = TargetNotional_i / (P_i · M_i)     # P_i mark, M_i multiplier
N_i              = round(N_i^raw)                     # floor / nearest + buffer / ceil
micro routing    = fractional standard exposure → MES / MNQ / MCL / MGC / ...
IM_proj          = Σ |N_i| · IM_i
if IM_proj / E_t > η_max: uniformly scale down and re-round
```

Details that matter:

- **Micro routing** maps fractional standard-contract exposure to the same-direction micro contract,
  never creating a self-hedge.
- **Rounding modes** are explicit (`floor` for conservative risk, `nearest` with a rebalance buffer,
  `ceil`); the buffer suppresses fee churn.
- **Margin feasibility** enforces the initial-margin ceiling (`η_max = 0.60` by default) by
  uniformly scaling the book and re-rounding; a hypothesis test proves the ceiling cannot be
  exceeded over arbitrary equity/price/weight inputs.
- **FX normalisation**: for non-USD accounts, equity is converted to USD for sizing and margin
  checks, contract prices are converted for sizing while target marks stay native, and a missing FX
  rate fails closed.
- **Discretization reporting** exposes tracking error (`Σ|w_target − w_actual|/2`) and micro-routing
  summaries for telemetry.

The translator returns a `ContractTargetSet`. The OMS diffs it against current positions; the
backtester's simulated broker consumes the same set. `tests/parity/` asserts identical targets and
deltas from both paths for identical market states.

## 6. Risk architecture

### 6.1 LLM Risk-First engine (Tier A)

`src/risk/llm_risk_overlay.py`:

1. **Structured output** per root: sentiment score, semantic risk score (`S ∈ [1,5]`), confidence
   (`c ∈ [0,1]`) and a reasoning summary, validated against strict schemas.
2. **Anti-hallucination filter**: multi-sample mean/confidence; when confidence is below `τ` or
   sample variance exceeds the configured maximum, the signal is neutralised to `θ_safe`.
3. **Reward shaping** for DRL training:
   `R_shaped = R_netPnL − λ Σ |w_i| · 1[c_i ≥ τ] · max(0, S_i − θ_safe)`.
4. **Deterministic projection** for backtest and live: `S_i ≥ κ ∧ c_i ≥ τ ⇒ w_i = 0`; between
   `θ_safe` and `κ` the weight scales linearly downward; Tier-1 macro events trigger lockout
   windows. A neural policy can never override this projection.

LLM calls are asynchronous and cached with a TTL; the synchronous rebalance path never awaits LLM
I/O. A background worker keeps the cache warm with last-known-good fallback.

### 6.2 Quantitative overlay (Tier B)

`src/risk/quant_risk_overlay.py` applies EWMA covariance volatility targeting, gross leverage,
single-instrument and sector caps, high-water-mark trailing stops, a UTC daily max-loss kill switch
and a portfolio drawdown limit. Durable state can be persisted across restarts, including daily
starting NLV, high-water marks, stopped roots and the operator/OMS kill-switch latch.

### 6.3 Safety gates

- The pre-trade gate is mandatory: kill switch → fat-finger limit → duplicate/rate limiter →
  `whatIfOrder` margin check → transmit.
- The emergency flatten path cancels all orders, closes positions and locks automated rebalancing
  until an explicit operator reset.
- Risk state is persisted atomically so a restart or IBKR nightly reset cannot re-arm a consumed
  risk budget.

## 7. Data layer

### 7.1 Canonical bars and storage

All historical data is normalised into an immutable canonical Polars schema keyed by
`(timestamp_utc, instrument_id, root_symbol, expiration_date)` with OHLCV, open interest, VWAP and
provenance metadata. `ParquetRepository` writes partitioned files atomically and exposes DuckDB SQL
queries, cache inventory and per-root provenance.

### 7.2 Vendor adapters

- **Databento** (`src/data/databento_loader.py`): lazy SDK import; `ohlcv-1m/1h/1d`, `tbbo`,
  `mbp-1/10` and `definition` schemas; cost estimation and a hard per-job budget guard before any
  download; deterministic `OfflineBarSource` for credential-free reproduction.
- **IBKR** (`src/data/ibkr_data.py`): lazy `ib_async` market-data streamer for live/paper and
  historical bar fetching for cache backfills. Contracts are qualified (front month resolved from
  roll rules, `conId` verified) before subscriptions.

### 7.3 Continuous futures and roll engine

`src/data/continuous_roller.py` builds continuous series using volume/OI crossover or calendar
N-days-before-FND/LTD triggers, with Databento native symbology cross-checks. Three adjustment modes
are supported: Panama additive, proportional ratio and unadjusted with explicit carry/term-slope.
Roll logic is point-in-time and never consumes future contract data.

### 7.4 Feature processor

`src/data/feature_processor.py` computes FinRL-X technicals (MACD, RSI, CCI, ADX, Bollinger, ATR,
Garman-Klass), term-structure slope, annualised roll yield, volume/OI ratio, order-flow imbalance and
the Mahalanobis turbulence index. The point-in-time state builder exposes only rows at or before the
decision timestamp.

### 7.5 Timeframes and resampling

`src/data/timeframes.py` is the single registry for the eight supported timeframes (`1m`, `5m`,
`15m`, `30m`, `1h`, `4h`, `1d`, `1w`): canonical name, bar duration, IBKR bar size, Databento base
schema, trading bars per year and the finer base timeframe used for resampling.
`src/data/resampler.py` aggregates canonical bars with right-boundary labels and UTC windows (no
look-ahead), and `ParquetRepository.read_timeframe()` auto-resamples from the finest cached base and
optionally persists the target schema. `MarketState.timeframe` and `BacktestSpec.timeframe` carry the
selection through strategy, risk, metrics and margin settlement.

### 7.6 Cold-start hydration

`src/execution/hydrator.py` loads the lookback window, backfills missing history through a
configured source chain (Databento → IBKR → offline when allowed), computes features and covariance,
validates a fully warm state and only then lets the runner process a live bar.

## 8. Backtest and execution parity

- **Backtest engines** (`src/backtest/`): a vectorized Polars mode for fast parameter sweeps and an
  event-driven bar mode for path-dependent mechanics such as rolls, partial fills, margin calls and
  forced liquidation. Both consume the same `MarketState`/pipeline contracts.
- **Margin and costs**: daily mark-to-market settlement (UTC calendar-day boundaries for intraday
  bars), initial/maintenance margin tracking, forced liquidation with penalty, tick-aware spread plus
  volume-impact slippage, tiered IBKR/CME fee schedules and two-legged roll costs.
- **Walk-forward**: purged and embargoed splits with explicit train/test boundaries; the optimizer
  scores trials only on out-of-sample folds.
- **Live execution** (`src/execution/`): hydration, IBKR connection manager, OMS pre-trade gate,
  native BAG roll execution and reconciliation sharing the same translator and risk contracts as the
  backtest.
- **Parity**: identical market states must produce identical `ContractTarget`s and OMS deltas from
  both paths; discrepancies are considered bugs in the shared contract, not expected drift.

## 9. API and dashboard

`src/api/main.py` creates the FastAPI application with dependency-injected services. The API is
offline-capable by default: when no live engine or IBKR session is attached it serves deterministic
fixtures and marks the payloads accordingly. Services own the wiring:

- `EngineMarketService` builds point-in-time states from cached Parquet first, falling back to
  synthetic fixtures, and binds the real live NLV when a broker account is available.
- `BacktestService` runs real engine backtests, persists runs and tear sheets, and returns explicit
  `data_source` / data-origin fields.
- `StrategyStudioService` validates and persists strategy YAML and serves the parameter catalog.
- `OptimizerService` runs/persists walk-forward studies and applies best parameters.
- `DRLService` trains checkpoints, writes metadata sidecars and lists models.
- `DataService` serves cache inventory, cost estimates and ingestion.

Every response that could contain synthetic, cached or stale data carries provenance/metadata so the
dashboard can label it. The frontend is a Next.js 15 dashboard with six views: Overview,
Weights, Risk, Term Structure & Roll Calendar, Backtests and Strategies. Live telemetry is delivered over `/ws/telemetry`.

## 10. Configuration

- `configs/instruments.yaml` — the 45-root futures master: multipliers, tick sizes, margin
  estimates, sectors, exchange symbols, roll rules and standard↔micro mappings.
- `configs/strategies/*.yaml` — declarative strategy pipelines: stage names plus scalar parameters.
  `configs/strategy.py` defines the typed models and rejects unknown fields.
- `configs/settings.py` — layered environment configuration (`FUTURESRL_*` env vars → `.env` →
  defaults) with secret redaction and environment/port consistency validation.
- `configs/macro_calendar.yaml` — Tier-1 macro event calendar for LLM lockouts.

Configuration is validated on load; invalid environment/port combinations fail startup rather than
silently running in the wrong mode.

## 11. Extension points

- **Add a futures root**: add an entry to `configs/instruments.yaml` (root, sector, asset class,
  exchange, multiplier, tick, margins, roll rules, micro mapping). Universe metadata, the catalog
  and offline synthetic data derive from the master automatically.
- **Add a selector, allocator or timing overlay**: implement the relevant Protocol in
  `src/strategies/{selectors,allocators,timing}`, register a factory in the corresponding registry,
  add catalog metadata if it should be Studio-configurable, and add tests.
- **Add a risk overlay**: implement the `RiskOverlay` Protocol, raise it in the pipeline registry and
  document its configuration. Overlays must preserve the universe contract and fail closed.
- **Add a data source or cache backend**: implement the storage/source Protocol, keep the canonical
  schema, and add provenance metadata for inventory.
- **Add a strategy preset**: extend the parameter catalog's preset templates; presets are validated
  against the instrument master and the active-stage filtering rules.
- **Add a DRL algorithm**: extend the allocator's supported algorithm registry and checkpoint
  metadata handling; inference must consume the checkpoint's recorded observation layout.

## 12. Testing and quality

- The suite is hermetic and deterministic: seeded fixtures, injected adapters, no network access and
  no wall-clock races.
- Property-based tests (Hypothesis) assert the margin ceiling for arbitrary equity/price/weight
  inputs.
- The accounting identity is asserted end-to-end in the backtest integration tests.
- Anti-look-ahead tests mutate future data and assert past results are unchanged, including
  resampling and walk-forward folds.
- Parity tests assert backtest/live contract-target identity.
- Risk tests cover hallucination ablation, deterministic circuit breakers, durable-state restarts and
  non-blocking LLM caches.
- `scripts/quality_gate.sh` runs `ruff check`, `ruff format --check`, strict `mypy` and the hermetic
  pytest suite (optionally enforcing coverage on core/risk/translator modules).

## 13. Operational notes

- **Environments**: `development`, `paper`, `live`, `backtest`. Live requires an explicit opt-in, a
  live IBKR port and `ibkr_paper=false`; inconsistent settings are rejected.
- **Data provenance**: API payloads and dashboard badges distinguish live, cached-real, delayed and
  synthetic data. Never present fixture values as a live account.
- **Risk state**: configure `FUTURESRL_RISK_STATE_STORE_URI` for durable JSON/DuckDB/Redis state in
  paper/live; without it, risk state is in-memory and resets with the process.
- **Broker/data adapters**: Databento and IBKR integrations are lazy and opt-in; CI exercises them
  through fakes. Real-feed smoke tests belong in a credentialed paper environment.
