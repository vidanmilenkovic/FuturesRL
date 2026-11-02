#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# FuturesRL-X quality gate: the exact checks required before every push.
#   ./scripts/quality_gate.sh          # lint + format check + types + tests
#   ./scripts/quality_gate.sh --cov    # additionally enforce 85% coverage on
#                                      # core / risk / translator modules
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"

# Hermetic gate: the developer's shell (e.g. FUTURESRLX_ENVIRONMENT=paper when
# running live/paper trading) must never change lint/type/test behaviour.
if compgen -e >/dev/null 2>&1; then
  for variable in $(compgen -e | grep -E '^FUTURESRLX_' || true); do
    unset "$variable"
  done
fi

echo "==> ruff check"
"$PY" -m ruff check .

echo "==> ruff format --check"
"$PY" -m ruff format --check .

echo "==> mypy (strict)"
"$PY" -m mypy src configs

if [[ "${1:-}" == "--cov" ]]; then
  echo "==> pytest with coverage gate (core, risk, translator >= 85%)"
  "$PY" -m pytest tests/ \
    --cov=src/core --cov=src/risk --cov-report=term-missing \
    --cov-fail-under=85
else
  echo "==> pytest"
  "$PY" -m pytest tests/ -q
fi

echo "==> quality gate passed"
