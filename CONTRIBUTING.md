# Contributing to FuturesRL-X

Thanks for helping improve FuturesRL-X. This project values deterministic,
auditable research code: a change should be understandable, testable offline,
and safe to run against a paper account.

## Development setup

```bash
git clone https://github.com/<your-org>/futuresrl-x.git
cd futuresrl-x
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
cp .env.example .env          # never commit .env
./scripts/dev.sh               # API :8000 + dashboard :3100
```

Minimum setup for backend-only work: `pip install -e ".[dev]"`. Optional extras:
`live` (IBKR), `databento` (historical data), `drl` (PyTorch/Stable-Baselines3),
`optimize` (Optuna). `all` installs every extra.

## Quality gate

Every pull request must pass:

```bash
./scripts/quality_gate.sh --cov
```

This runs `ruff check`, `ruff format --check`, strict `mypy`, and the hermetic
pytest suite with the coverage gate. Tests must not require network access,
TWS/IB Gateway, paid market-data credentials, or a populated local cache.

Frontend changes additionally require:

```bash
cd frontend
npm run build
npm run lint
```

## Architectural invariants

- The pipeline contract is the target weight vector:
  `w_t = R_t(T_t(A_t(S_t(X_{<=t}))))`. Do not bypass a stage from research or
  execution code.
- Futures are discrete, expiring, leveraged instruments. Orders flow through the
  deterministic `ContractTranslator` (multiplier, tick size, margin, micro
  routing); never translate weights to contracts ad hoc.
- Backtest and live paths must share the same translator/risk contracts. Any
  parity change needs a parity test.
- Point-in-time discipline: no look-ahead in features, bars, resampling, folds,
  or universe selection. Resampled bars close at their right window boundary.
- Every synthetic, cached, delayed, or fixture value must be labelled honestly in
  API responses and UI (data origin/provenance); never present fixture data as
  live.
- Risk overlays are deterministic and fail closed. A kill switch or data
  unavailability must not silently fall back to simulated execution in a live
  context.
- Configuration is declarative (`configs/`, YAML strategies); code reads typed
  models rather than hard-coded instrument metadata.

## Pull requests

1. Open an issue or describe the problem/behaviour in the PR.
2. Keep commits atomic and scoped; include tests for new behaviour.
3. Run `./scripts/quality_gate.sh --cov` and the frontend build/lint locally.
4. Update `CHANGELOG.md` for user-visible changes.
5. Do not commit secrets, `.env`, `data/`, `artifacts/`, generated Parquet,
   checkpoints, or machine-specific paths.
6. By contributing you agree your work is licensed under the MIT License.
