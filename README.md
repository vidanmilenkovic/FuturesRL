<p align="center">
  <img src="docs/Frame.png" alt="FuturesRL Command Center connected to an IBKR paper account" width="250" />
</p>



<p align="center"><b>An AI-Native Modular Infrastructure for Quantitative Futures Trading</b></p>

<p align="center">
An algorithmic futures platform that unifies multi-timeframe market data,
LLM risk sentiment, composable strategy generation, walk-forward optimization, and
broker execution through a weight-centric interface.  
</p>

<p align="center">
  <img alt="Python 3.12+" src="https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white" />
  <img alt="Broker: IBKR ib_async" src="https://img.shields.io/badge/Broker-IBKR%20-D8122A" />
  <img alt="Data: Databento" src="https://img.shields.io/badge/Data-Databento%20-4B8BBE" />
  <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow" />
</p>

<p align="center">
  <a href="#about">About</a> |
  <a href="#architecture">Architecture</a> |
  <a href="#quick-start">Quick Start</a> |
  <a href="#project-layout--documentation">Project Layout</a> |
  <a href="#acknowledgements--license">Acknowledgements</a>
</p>

---


## About

Inspired by [FinRL](https://github.com/AI4Finance-Foundation/FinRL-Trading)'s weight-centric architecture, FuturesRL is a modular research-to-execution platform for CME futures: a weight-centric strategy
pipeline, futures-native risk and contract translation, and one shared backtest/live path.

### Key Highlights

- **45+ CME futures across 7 sectors**, with standard↔micro routing.
- **Weight-centric composability**: swap universe selectors, allocators, DRL policies and overlays
  without touching risk or execution code.
- **Futures-native realism**: continuous rolls, F1–F6 term structure, margin/cost simulation and
  purged walk-forward validation.
- **Eight timeframes**, cache-aware data management and budgeted Databento/IBKR backfills.
- **Backtest-to-live parity**: the same Contract Translator feeds the event-driven engine and the
  IBKR OMS, with LLM and quantitative risk overlays in both paths.

---

FuturesRL leverages FinRL's weight-centric architecture and translates it into futures contracts:

$$
w_t = \mathcal{R}_t \left( \mathcal{T}_t \left( \mathcal{A}_t \left( \mathcal{S}_t(X_{\le t}) \right) \right) \right) \implies N_{i,t} = \mathcal{C}_t(w_t, \dots)
$$

Each transformation is contract-preserving — you can swap any module (e.g., risk-parity $\rightarrow$ DRL policy) without touching the rest of the pipeline, and the same weights flow identically through backtesting and live execution.

|Stage | Contract | What it does |
| --- | --- | --- |
| $\mathcal{S}$ | Universe selection | Filters liquid roots by dollar volume, open interest and regime. |
| $\mathcal{A}$ | Portfolio allocation | Produces signed target weights via risk parity, vol targeting, mean-variance or DRL. |
| $\mathcal{T}$ | Timing & term structure | Scales weights with TSMOM, KAMA, carry/roll-yield, mean-reversion and seasonality signals. |
| $\mathcal{R}$ | Risk overlays | Applies LLM Risk-First circuit breakers and deterministic quantitative limits. |
| $\mathcal{C}$ | Contract translation | Converts $\mathcal{w}$ into discrete CME contracts with micro routing, FX normalization and margin checks. |

## Architecture

WIP

#

## Quick Start

###  Clone & Launch

```bash
git clone https://github.com/vidanmilenkovic/FuturesRL-X
```
```
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
./scripts/dev.sh        
```

Default `localhost:3100`



## Project Layout & Documentation

```text
futuresrl-x/
├── ⚙️ configs/                     # typed settings, 45-root instrument  master, declarative strategies
│   ├── settings.py
│   ├── instruments.yaml
│   └── strategies/                 # tsmom_vol_target.yaml, drl_ppo_llm_risk.yaml
├──  src/                     
│   ├── src/data/                    # Databento/IBKR adapters, Parquet cache, rolls, 8 timeframes, features
│   ├── src/strategies/              # S_t selectors, A_t allocators, T_t overlays, DRL env/training
│   ├── src/risk/                    # LLM Risk-First, quant overlay, ContractTranslator, state store
│   ├── src/backtest/                # event/vectorized engine, margin/cost simulators, walk-forward
│   ├── src/analytics/               # institutional metrics, tearsheets, attribution
│   ├── src/execution/                # IBKR client/OMS, pre-trade gate, BAG rolls, reconciler, runner
│   ├── src/api/                     # FastAPI app, domain routes, schemas, telemetry, services
├── 🖥️ frontend/                    # Next.js 15 workstation: Command Center, Pipeline, Risk, Studio
├── 🧪 tests/                       # unit/, integration/, parity/ — hermetic, deterministic suites
├── 🐳 docker/                      # backend/frontend images, TimescaleDB/Redis/IB Gateway compose
├── 📚 docs/                        # USER_GUIDE.md, ARCHITECTURE.md, screenshots/
├── scripts/                        # quality_gate.sh, dev.sh, WSL/Windows host helper
├── CHANGELOG.md                    # Keep a Changelog + Semantic Versioning
├── CONTRIBUTING.md                 # setup, quality gate, architectural invariants and PR rules
├── LICENSE                         # MIT
└── pyproject.toml                  # package metadata, extras and CLI entry points
```



## Acknowledgements & License

The weight-centric abstraction is inspired by
**FinRL-X / FinRL-Trading** from the AI4Finance Foundation (Yang et al.,
*FinRL-X: An AI-Native Modular Infrastructure for Quantitative Trading*, [arXiv:2603.21330](https://arxiv.org/abs/2603.21330)).


Released under the **[MIT License](LICENSE)**.

**Futures trading involves substantial leverage, margin calls and risk of loss. FuturesRL is for
educational and research purposes only and is not investment advice; backtest and paper results do
not guarantee live performance. Use at your own risk.**
