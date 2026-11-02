# ROLE & OBJECTIVE
You are a Principal Quantitative Software Architect and Systems Engineer. Your objective is to build **FuturesRL-X**, a modular, production-grade algorithmic futures trading platform (supporting Historical Backtesting, IBKR Paper Trading, and IBKR Live Execution). 

You will build this repository incrementally across 6 sequential development phases, committing clean, tested, type-safe code to GitHub via atomic commits and feature branches.

---

# 1. CORE ARCHITECTURAL PARADIGM: ADAPTING FINRL-X TO FUTURES

## 1.1 The FinRL-X Weight-Centric Composable Pipeline
In the standard FinRL-X (`AI4Finance-Foundation/FinRL-Trading`) architecture, the sole interface contract between strategy research, backtesting, and broker execution is the target portfolio weight vector:
$$w_t = R_t(T_t(A_t(S_t(X_{\le t}))))$$

Where every stage is a contract-preserving transformation:
1. $S_t$ (**Universe Selection**): Filters active futures roots based on liquidity, open interest, and term-structure regime.
2. $A_t$ (**Portfolio Allocation**): Classical (Risk Parity, Mean-Variance), Rule-Based, or Deep Reinforcement Learning (PPO/CPPO/SAC) generating raw signed target notional weights $w_t^{\text{base}} \in [-L_{\max}, +L_{\max}]^N$.
3. $T_t$ (**Timing & Term-Structure Overlay**): Scales weights via Time-Series Momentum (TSMOM), Kaufman Adaptive Moving Average (KAMA), and carry/roll-yield filters.
4. $R_t$ (**LLM & Statistical Risk Overlay**): Applies volatility targeting, drawdown limits, and the **LLM Risk-First Structural Constraint Engine** to output the final approved target weight vector $w_t$.

## 1.2 Translating Equity Weights ($w_t$) into Futures Contracts ($N_{i,t}$)
Unlike cash equities (which trade fractional shares, have no expiration, and require 100% cash outlay), futures are leveraged, expiring derivatives with discrete contract multipliers ($M_i$), tick sizes, and CME SPAN / IBKR Initial & Maintenance Margin requirements.

To preserve FinRL-X's deployment consistency (identical $w_t$ flowing into both the Backtest Engine and the IBKR Execution Engine), implement a deterministic **Futures Contract Translator (`ContractTranslator`)** immediately downstream of $w_t$:

1. **Notional Exposure Calculation**:
   $$\text{Target Notional}_{i,t} = w_{i,t} \times E_t$$
   Where $E_t$ is current Net Liquidation Value (NLV) and $w_{i,t}$ is the signed notional weight (can be negative for short positions; $\sum \vert{}w_{i,t}\vert{}$ represents gross leverage).
2. **Discrete Contract Conversion**:
   $$N_{i,t}^{\text{raw}} = \frac{w_{i,t} \times E_t}{P_{i,t} \times M_i}$$
   Where $P_{i,t}$ is the current mark price of the active front/second-month contract and $M_i$ is the contract multiplier (e.g., $50 for ES, $5 for MES, $1,000 for CL).
3. **Rounding & Micro-Routing**:
   - Apply configurable rounding (`floor` toward zero for conservative risk, or `nearest` with a minimum rebalance buffer $\Delta N$ to prevent fee churn).
   - **Auto-Micro Routing**: If capital $E_t$ is too small for standard contracts (e.g., `ES`, `NQ`, `CL`, `GC`), automatically map leftover fractional notional exposure to Micro contracts (`MES`, `MNQ`, `MCL`, `MGC`) to minimize tracking error against $w_t$.
4. **Margin Feasibility Projection**:
   - Compute projected total Initial Margin: $IM_{\text{proj}} = \sum_i \vert{}N_{i,t}\vert{} \times IM_{i,t}$.
   - Enforce the hard Margin Cushion Constraint:
     $$\text{Margin Utilization} = \frac{IM_{\text{proj}}}{E_t} \le \eta_{\max} \quad (\text{default } \eta_{\max} = 0.60)$$
   - If violated, uniformly scale down $N_{i,t}$ across all legs before order generation.

---

# 2. SYSTEM MODULES & FEATURE SPECIFICATIONS

## Module 1: Data Layer (`src/data/`) — Databento + IBKR + FinRL-X Processor
- **Unified Schema**: All historical and live market data must normalize into an immutable Polars/Parquet schema indexed by `(timestamp_utc, instrument_id, root_symbol, expiration_date)`.
- **Databento Adapter (`databento_loader.py`)**:
  - Historical ingestion via `databento` Python SDK (`GLBX.MDP3` for CME/CBOT/NYMEX/COMEX).
  - Support schemas: `ohlcv-1m`, `ohlcv-1h`, `ohlcv-1d`, `tbbo` (Top of Book), `mbp-1` / `mbp-10` (Market by Price for order book imbalance features), and `definition` (point-in-time contract specs, multipliers, tick sizes, expiration dates).
  - Local caching layer using partitioned Parquet files (`data/parquet/{dataset}/{schema}/{root}/`) + DuckDB/TimescaleDB for fast SQL querying.
- **Continuous Futures Roll Engine (`continuous_roller.py`)**:
  - Training DRL/ML models on raw expiring contracts creates artificial price jumps at roll dates.
  - Implement 3 roll-trigger methodologies:
    1. **Volume/Open-Interest Crossover** (roll when next contract volume > front contract volume).
    2. **Calendar N-Days Prior to First Notice Date (FND) or Last Trade Date (LTD)** (mandatory earlier roll for physical commodities like `CL`, `GC`, `ZC` to strictly prevent physical delivery).
    3. **Databento Native Continuous Symbology** (`[ROOT].v.0` volume-based, `[ROOT].c.0` calendar-based) cross-verified locally.
  - Implement 3 price-adjustment methods for continuous backtest/training series:
    1. **Panama (Back-Adjusted Additive)**: $P_{\text{adj}} = P_{\text{raw}} + \sum \Delta_{\text{roll}}$. Preserves absolute point moves and dollar PnL per contract.
    2. **Proportional (Ratio-Adjusted Multiplicative)**: Preserves percentage returns for log-return features.
    3. **Unadjusted + Explicit Roll Yield**: Keeps raw prices for margin/notional accuracy while exporting `carry_annualized` and `term_slope` as explicit features.
- **IBKR Live/Paper Market Data Streamer (`ibkr_data.py`)**:
  - Built on `ib_async` (the modern async successor to `ib_insync`).
  - Streams real-time 5-second bars, tick-by-tick data, and L1/L2 order book updates when Databento live streaming is disabled, or acts as a secondary cross-check feed.
- **Feature Engineering Pipeline (`feature_processor.py`)**:
  - FinRL-X technical indicators: MACD, RSI, CCI, ADX, Bollinger Bands, ATR, Garman-Klass Volatility.
  - Futures-specific features: Term-structure slope ($\frac{F_2 - F_1}{F_1}$), Annualized Roll Yield, Volume/OI ratio, Order Flow Imbalance (OFI from `tbbo`), and Cross-Asset Macro Regime indicators (VIX term structure, 2Y/10Y Treasury yield curve slope, DXY).
  - Mahalanobis Turbulence Index (FinRL standard for structural market breaks).

## Module 2: Strategy & Composable Pipeline (`src/strategies/`)
- **Strict Protocol Interfaces (`interfaces.py`)**:
  - `UniverseSelector`: `select(state: MarketState) -> list[InstrumentId]`
  - `PortfolioAllocator`: `allocate(candidates: list[InstrumentId], state: MarketState) -> WeightVector`
  - `TimingOverlay`: `adjust_timing(weights: WeightVector, state: MarketState) -> WeightVector`
  - `RiskOverlay`: `apply_risk(weights: WeightVector, state: MarketState) -> WeightVector`
- **Out-of-the-Box Allocators & Overlays**:
  - *Classical*: Equal-Risk Contribution (Risk Parity across Equity Index, Rates, Energy, Metals, FX futures), Volatility-Targeted TSMOM.
  - *Adaptive Regime Rotation*: FinRL-X style Slow Regime (26-week trend + macro vol) + Fast Risk-Off (3-day shock detector).
  - *DRL Agents (`drl_agents.py`)*: Gymnasium-compatible futures environment (`FuturesTradingEnv`) supporting Stable-Baselines3 (`PPO`, `SAC`, `TD3`, `A2C`) and Constrained PPO (`CPPO`).

## Module 3: LLM + FinRL-X Risk-First Engine (`src/risk/`)
Implement a two-tier hybrid risk system combining quantitative portfolio risk controls with the **FinRL Risk-First LLM Architecture**:

### Tier A: LLM Risk-First Structural Constraint Engine (`llm_risk_overlay.py`)
Ingests real-time macro calendars (FOMC, CPI, NFP, EIA Crude Inventory, WASDE reports), financial news headlines, and central bank speeches via structured Pydantic outputs from an LLM provider (OpenAI / Anthropic / DeepSeek / local vLLM).
1. **Structured Schema Output**: For each futures root $i$, the LLM outputs:
   - `sentiment_score` $\in [1, 5]$
   - `semantic_risk_score` $S_{i,t} \in [1, 5]$ (where 5 = imminent macro shock, geopolitical tail event, or extreme illiquidity)
   - `confidence` $c_{i,t} \in [0.0, 1.0]$
   - `reasoning_summary`: Concise audit string.
2. **Module 1 — Empirical Confidence & Variance Filter (Anti-Hallucination)**:
   - Sample $K$ completions (or evaluate token log-probabilities). If confidence $c_{i,t} < \tau$ (default $\tau = 0.70$) or variance across samples exceeds threshold $\sigma^2_{\max}$, neutralize the LLM signal to baseline ($S_{i,t} = \theta_{\text{safe}}$) so hallucinations never trigger false trades.
3. **Module 2 — DRL Reward Shaping (Training Mode)**:
   - During RL training in `FuturesTradingEnv`, apply a Lagrangian semantic risk penalty to the step reward:
     $$R_t^{\text{shaped}} = R_t^{\text{net\_pnl}} - \lambda \sum_{i} |w_{i,t}| \cdot \mathbb{I}(c_{i,t} \ge \tau) \cdot \max(0, S_{i,t} - \theta_{\text{safe}})$$
4. **Module 3 — Deterministic Post-Hoc Circuit Breaker & Projection (Live & Backtest)**:
   - Never rely solely on a neural network policy to obey risk alerts. Enforce a hard deterministic projection on the weight vector $w_t$:
     - If $S_{i,t} \ge \kappa$ (Critical Threshold, e.g., $4.5$) AND $c_{i,t} \ge \tau$: Force $w_{i,t} = 0$ (immediate liquidation / block new entries).
     - If $\theta_{\text{safe}} < S_{i,t} < \kappa$: Scale weight linearly: $w_{i,t} \leftarrow w_{i,t} \times \frac{\kappa - S_{i,t}}{\kappa - \theta_{\text{safe}}}$.
   - **Macro Event Lockout Window**: Automatically flatten or cap leverage $M$ minutes before and after Tier-1 scheduled economic releases (e.g., FOMC, CPI).

### Tier B: Quantitative Futures Risk Overlay (`quant_risk_overlay.py`)
- **Ex-Ante Portfolio Volatility Targeting**: Scale gross weights so annualized portfolio volatility matches $\sigma_{\text{target}}$ (e.g., 12% annualized) using an EWMA covariance matrix across futures roots.
- **Intraday Drawdown & Trailing Stop-Loss**:
  - High-water mark trailing stop per contract and at the portfolio level.
  - **Daily Max Loss Kill-Switch**: If intraday realized + unrealized PnL drops below $-X\%$ of starting daily NLV, flatten all positions, cancel open orders, and lock execution until manual reset or next trading session.
- **Correlation Spike Guard**: Caps aggregate notional allocation to any single asset class cluster (e.g., `ES` + `NQ` + `YM` + `RTY` treated as one correlated equity beta bucket).

## Module 4: Backtesting & Verification Engine (`src/backtest/`)
- **Event-Driven & Vectorized Modes**:
  - Fast vectorized mode (Polars-backed) for rapid parameter sweeps and weight pipeline validation.
  - Tick/Bar event-driven simulation (`EventBacktester`) replicating the exact `ExecutionBroker` interface used in live IBKR trading.
- **Futures-Specific Realism**:
  - **Daily Mark-to-Market Settlement**: Cash balance adjusts daily based on settlement price changes, affecting available excess liquidity.
  - **Margin Call & Forced Liquidation Simulation**: Tracks Initial Margin ($IM$) and Maintenance Margin ($MM$) per bar. If $E_t < MM_t$, simulates forced broker liquidation at unfavorable slippage + penalty fee.
  - **Roll Execution Simulation**: When the active contract changes from $F_1$ to $F_2$, executes either two legged orders or a native calendar spread order (`BAG`), applying realistic bid-ask spread and exchange fees to both legs.
  - **Tick-Aware Slippage & Commission Model**:
    - Slippage modeled as a function of order size relative to bar volume + minimum 1-tick spread cost ($M_i \times \text{TickSize}_i$).
    - Tiered IBKR commissions + CME/CBOT/NYMEX exchange, clearing, and NFA regulatory fees per contract.
- **Anti-Lookahead Verification**:
  - Purged and Embargoed Walk-Forward Cross-Validation (`PurgedWalkForwardSplit`).

## Module 5: IBKR Execution Engine (`src/execution/`)
- **Unified Broker Protocol (`BrokerAdapter`)**: Identical method signatures for `SimulatedBroker`, `IBKRPaperBroker` (Port `7497` / `4002`), and `IBKRLiveBroker` (Port `7496` / `4001`).
- **IBKR Connection & State Manager (`ibkr_client.py`)**:
  - Asynchronous event loop via `ib_async` connecting to TWS or headless **IB Gateway** (containerized via Docker).
  - Automatic heartbeat monitoring, exponential backoff reconnection, and handling of IBKR nightly server reset windows (approx. 23:45–00:45 ET).
- **Contract Resolution & Roll Execution (`ibkr_roller.py`)**:
  - Resolves abstract root symbols (e.g., `"ES"`) to exact IBKR `Future(symbol='ES', lastTradeDateOrContractMonth='202612', exchange='CME')` `conId` objects.
  - Executes contract rolls using native IBKR **Combo/Calendar Spread Orders (`Bag`)** to eliminate leg-to-leg slippage risk during roll windows.
- **Order Management System (OMS) & Pre-Trade Risk Gate (`oms.py`)**:
  - Supports `MKT`, `LMT`, Adaptive Algo (`IBALGO`), `TWAP`, `VWAP`, and Bracket Orders (entry + native exchange stop-loss).
  - **Pre-Trade Hard Gate**: Independent validation module checking:
    1. Max order size per fat-finger limit.
    2. IBKR `whatIfOrder` API call to verify real-time post-trade Initial and Maintenance Margin impact before transmitting the live order.
    3. Duplicate order / rapid fire rate-limiter.
- **State Reconciliation Loop**: Every 30 seconds, compares internal expected contract positions against `ib.positions()` and `ib.accountValues()`. Triggers an alert and halts automated rebalancing if an unapproved drift is detected.

## Module 6: Metrics, Analytics & Visualization (`src/analytics/` & `frontend/`)
- **Institutional Futures Metrics (`metrics.py`)**:
  - **Return & Risk-Adjusted**: CAGR, Annualized Volatility, Sharpe Ratio, Sortino Ratio, Calmar Ratio, Omega Ratio.
  - **Tail Risk**: Maximum Drawdown (%), Max Drawdown Duration, 95%/99% Historical & Parametric Value-at-Risk (VaR), Expected Shortfall (CVaR), and **Rachev Ratio** (tail gain vs. tail loss at $\alpha = 0.05$, directly validating the LLM Risk-First engine).
  - **Futures-Specific Attribution**:
    - *Spot/Price Return vs. Roll Yield (Carry) Attribution*: Decomposes total PnL into directional price movement vs. contango/backwardation roll cost/gain.
    - *Margin Efficiency*: Return on Maximum Initial Margin Deployed ($RoM$), Average & Peak Margin Utilization (%).
    - *Execution Quality*: Implementation Shortfall (arrival price vs. fill price in ticks), Turnover, Commission-to-Gross-PnL Ratio.
- **Interactive Web Dashboard (`frontend/` + `src/api/`)**:
  - **Backend API**: FastAPI with REST endpoints for historical runs/configs and WebSockets (`/ws/telemetry`) pushing live tick PnL, margin utilization, active positions, and LLM risk alerts.
  - **Frontend UI**: Next.js (React + TypeScript + TailwindCSS + shadcn/ui) + **TradingView Lightweight Charts**.
  - **5 Core Views**:
    1. **Command Center (Live/Paper Monitor)**: Real-time equity curve, NLV, Margin Utilization gauge (with 60% warning / 80% critical zones), active futures contracts table (Root, Expiry, Days to Roll, Qty, Notional Weight $w_i$, Unrealized PnL), and a giant hardware-style **EMERGENCY FLATTEN ALL** button.
    2. **FinRL-X Pipeline Inspector**: Step-by-step visual waterfall of the current rebalance cycle showing how weights transform across stages: $S_t \rightarrow w_t^{\text{base}} \rightarrow w_t^{\text{timing}} \rightarrow w_t^{\text{risk}} \rightarrow \text{Target Contracts } N_{i,t}$.
    3. **LLM Risk-First Monitor**: Live feed of macro/news events, LLM confidence scores $c_{i,t}$, semantic risk scores $S_{i,t}$, and active Circuit Breaker overrides.
    4. **Futures Term Structure & Roll Calendar**: Interactive contango/backwardation curve per commodity root and upcoming roll countdowns.
    5. **Backtest Lab & Tear Sheet Compare**: Run walk-forward backtests, compare multiple strategy configs against benchmarks, and inspect rolling Sharpe, Underwater Drawdown, Rachev tail plots, and Roll Yield attribution.

---

# 3. REPOSITORY DIRECTORY STRUCTURE

Enforce the following clean, modular directory tree:

```text
futures-rl-x/
├── docker/
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   └── docker-compose.yml          # TimescaleDB, Redis, IB Gateway, Backend, Frontend
├── configs/
│   ├── settings.py                 # Pydantic BaseSettings + .env loader
│   ├── instruments.yaml            # Futures root definitions, multipliers, roll rules, micro mappings
│   └── strategies/                 # Declarative YAML strategy pipelines (FinRL-X style)
│       ├── tsmom_vol_target.yaml
│       └── drl_ppo_llm_risk.yaml
├── src/
│   ├── core/
│   │   ├── types.py                # Immutable dataclasses: MarketState, WeightVector, ContractTarget, OrderEvent
│   │   ├── interfaces.py           # Python Protocols for S_t, A_t, T_t, R_t, BrokerAdapter
│   │   └── event_bus.py            # Async pub/sub message bus for live & event-driven backtest parity
│   ├── data/
│   │   ├── databento_loader.py     # Historical & live Databento ingestion (ohlcv, tbbo, definition)
│   │   ├── ibkr_data.py            # Live/Paper IBKR market data streamer via ib_async
│   │   ├── continuous_roller.py    # Roll schedule calculator & Panama/Ratio price adjuster
│   │   ├── storage.py              # Partitioned Parquet + DuckDB/TimescaleDB repository
│   │   └── feature_processor.py    # Technicals, term structure, carry, OFI, turbulence index
│   ├── strategies/
│   │   ├── pipeline.py             # FinRL-X Composer: w_t = R_t(T_t(A_t(S_t(X))))
│   │   ├── selectors/              # Liquidity, OI, and regime universe selectors
│   │   ├── allocators/             # EqualWeight, RiskParity, MeanVariance, DRLAllocator (PPO/CPPO/SAC)
│   │   ├── timing/                 # TSMOM, KAMA, Carry/Term-Structure filters
│   │   └── envs/                   # Gymnasium FuturesTradingEnv with reward shaping & margin dynamics
│   ├── risk/
│   │   ├── llm_risk_overlay.py     # Module 1 (Confidence Filter), Module 2 (Reward Penalty), Module 3 (Circuit Breaker)
│   │   ├── quant_risk_overlay.py   # Vol targeting, covariance scaling, drawdown stops, correlation guards
│   │   └── contract_translator.py  # Maps approved weights w_t -> discrete contracts N_i,t + Micro routing + Margin check
│   ├── backtest/
│   │   ├── engine.py               # Unified backtest runner consuming StrategyPipeline
│   │   ├── margin_simulator.py     # SPAN/Initial/Maintenance margin & daily MTM settlement simulator
│   │   ├── cost_model.py           # Tick-aware slippage, spread, and IBKR/CME fee schedules
│   │   └── walk_forward.py         # Purged & embargoed walk-forward validator
│   ├── execution/
│   │   ├── ibkr_client.py          # Async IBKR connection manager, reconnect logic, whatIfOrder margin check
│   │   ├── oms.py                  # Order Management System, Pre-Trade Risk Gate, emergency kill-switch
│   │   ├── ibkr_roller.py          # Automated calendar spread (BAG) contract roll execution
│   │   └── reconciler.py           # Periodic internal-vs-IBKR position & cash state reconciliation
│   ├── analytics/
│   │   ├── metrics.py              # Sharpe, Sortino, Calmar, CVaR, Rachev Ratio, RoM, Carry Attribution
│   │   └── tearsheet.py            # JSON/HTML report generator
│   └── api/
│       ├── main.py                 # FastAPI entrypoint
│       ├── routes/                 # REST endpoints for backtests, strategies, positions, risk overrides
│       └── ws.py                   # WebSocket broadcaster for real-time telemetry
├── frontend/                       # Next.js 15 + TypeScript + Tailwind + TradingView Lightweight Charts
├── tests/
│   ├── unit/
│   ├── integration/
│   └── parity/                     # Tests proving Backtest and Live Execution produce identical ContractTargets
├── pyproject.toml
└── README.md