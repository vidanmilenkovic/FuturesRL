<p align="center">
  <img src="docs/screenshots/command-center-ibkr-paper.png" alt="FuturesRL-X Command Center connected to an IBKR paper account" width="920" />
</p>

<h1 align="center">FuturesRL-X</h1>

<p align="center"><b>An AI-Native Modular Infrastructure for Quantitative Futures Trading</b></p>

<p align="center">
A deployment-consistent algorithmic futures platform that unifies multi-timeframe market data,
continuous contract rolling, composable strategy generation, purged walk-forward optimization, and
IBKR paper/live execution through a single weight-centric contract.
</p>

<p align="center">
  <img alt="Python 3.12+" src="https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white" />
  <img alt="Next.js 15" src="https://img.shields.io/badge/Next.js-15-000000?logo=nextdotjs&logoColor=white" />
  <img alt="Type Checked: mypy strict" src="https://img.shields.io/badge/Type%20Checked-mypy%20strict-2A6DB2" />
  <img alt="Code Style: ruff" src="https://img.shields.io/badge/Code%20Style-ruff-261230" />
  <img alt="Broker: IBKR ib_async" src="https://img.shields.io/badge/Broker-IBKR%20ib__async-D8122A" />
  <img alt="Data: Databento / Parquet" src="https://img.shields.io/badge/Data-Databento%20%2F%20Parquet-4B8BBE" />
  <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow" />
</p>

<p align="center">
  <a href="#overview">Overview</a> ·
  <a href="#the-pipeline">Pipeline</a> ·
  <a href="#workstation">Workstation</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#project-layout--documentation">Project Layout</a> ·
  <a href="#acknowledgements--license">Acknowledgements</a>
</p>

---

## Overview

FuturesRL-X is a modular research-to-execution platform for CME futures: a weight-centric strategy
pipeline, futures-native risk and contract translation, and one shared backtest/live path with a
FastAPI + Next.js operator workstation.

### Key Highlights

- **45+ CME futures across 7 sectors**, with standard↔micro routing and a 60% initial-margin ceiling.
- **Weight-centric composability**: swap universe selectors, allocators, DRL policies and overlays
  without touching risk or execution code.
- **Futures-native realism**: continuous rolls, F1–F6 term structure, margin/cost simulation and
  purged walk-forward validation.
- **Eight timeframes**, cache-aware data management and budgeted Databento/IBKR backfills.
- **Backtest-to-live parity**: the same `ContractTranslator` feeds the event-driven engine and the
  IBKR OMS, with LLM and quantitative risk overlays in both paths.

## The Pipeline

```text
X_{≤t}  ->  S_t  ->  A_t  ->  T_t  ->  R_t  ->  w_t  ->  C_t  ->  N_{i,t}
```

| Stage | Contract | What it does |
| --- | --- | --- |
| `S_t` | Universe selection | Filters liquid roots by dollar volume, open interest and regime. |
| `A_t` | Portfolio allocation | Produces signed target weights via risk parity, vol targeting, mean-variance or DRL. |
| `T_t` | Timing & term structure | Scales weights with TSMOM, KAMA, carry/roll-yield, mean-reversion and seasonality signals. |
| `R_t` | Risk overlays | Applies LLM Risk-First circuit breakers and deterministic quantitative limits. |
| `C_t` | Contract translation | Converts `w_t` into discrete CME contracts with micro routing, FX normalization and margin checks. |

## Workstation

<table>
  <tr>
    <td width="50%" align="center"><img src="docs/screenshots/command-center-ibkr-paper.png" alt="Command Center with IBKR paper telemetry" /></td>
    <td width="50%" align="center"><img src="docs/screenshots/strategy-studio-universe.png" alt="Strategy Studio universe and sector picker" /></td>
  </tr>
  <tr>
    <td align="center"><sub><b>Command Center</b> — live NLV, margin, positions and risk controls.</sub></td>
    <td align="center"><sub><b>Strategy Studio</b> — 45-root universe and sector-aware stage editing.</sub></td>
  </tr>
  <tr>
    <td width="50%" align="center"><img src="docs/screenshots/strategy-studio-builder.png" alt="Strategy Studio manual builder" /></td>
    <td width="50%" align="center"><img src="docs/screenshots/optuna-walk-forward-optimizer.png" alt="Optuna walk-forward optimizer" /></td>
  </tr>
  <tr>
    <td align="center"><sub><b>Manual Builder</b> — declarative pipeline composition and YAML preview.</sub></td>
    <td align="center"><sub><b>Optuna Optimizer</b> — purged OOS trials with real/synthetic provenance.</sub></td>
  </tr>
</table>

## Quick Start

### 1. Clone & Launch

```bash
git clone https://github.com/<your-org>/futuresrl-x.git
cd futuresrl-x
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
./scripts/dev.sh        # API :8000 · workstation :3100
```

`dev.sh` creates `.env` from `.env.example` when needed and starts both services against the
deterministic offline stack — no API keys or broker connection required.

### 2. CLI Workflow

```bash
futuresrlx-backtest --strategy configs/strategies/tsmom_vol_target.yaml --offline
futuresrlx-optimize --strategy configs/strategies/tsmom_vol_target.yaml --trials 25 --objective sharpe --offline
futuresrlx-train-drl --strategy configs/strategies/drl_ppo_llm_risk.yaml --timesteps 20000 --offline
futuresrlx-runner --check
```

## Project Layout & Documentation

```text
futuresrl-x/
├── configs/                  # typed settings, instrument master, declarative strategies
├── src/core/                 # immutable types, protocols, event bus
├── src/data/                 # Databento/IBKR adapters, Parquet cache, rolls, timeframes, features
├── src/strategies/           # S_t / A_t / T_t components, DRL environment and training
├── src/risk/                 # LLM Risk-First, quant overlay, ContractTranslator
├── src/backtest/             # event/vectorized engine, margin/cost models, walk-forward
├── src/execution/            # IBKR OMS, pre-trade gate, BAG rolls, reconciler, runner
├── src/api/                  # FastAPI services, domain routes, telemetry
├── frontend/                 # Next.js workstation
├── tests/ + docker/          # test suites; compose stack with DB, Redis and IB Gateway
└── docs/ + scripts/          # USER_GUIDE/ARCHITECTURE; quality gate and dev launcher
```

Detailed setup, configuration and operations live in the **[User Guide](docs/USER_GUIDE.md)**.
System design, invariants and extension points live in **[Architecture](docs/ARCHITECTURE.md)**.
Development workflow and PR expectations are in **[Contributing](CONTRIBUTING.md)**.

## Acknowledgements & License

The weight-centric `S_t → A_t → T_t → R_t` abstraction is inspired by
**FinRL-X / FinRL-Trading** from the AI4Finance Foundation (Yang et al.,
*FinRL-X: An AI-Native Modular Infrastructure for Quantitative Trading*, arXiv:2603.21330).
Portfolio-construction and validation ideas also draw on Grinold & Kahn, *Active Portfolio
Management* (1999), and López de Prado, *Advances in Financial Machine Learning* (2018).

Released under the **[MIT License](LICENSE)**.

Futures trading involves substantial leverage, margin calls and risk of loss. FuturesRL-X is for
educational and research purposes only and is not investment advice; backtest and paper results do
not guarantee live performance. Use at your own risk.
