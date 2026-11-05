#!/usr/bin/env bash
# Start the FuturesRL API (:8000) and dashboard (:3100) with clean shutdown.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -f .env ]]; then
  if [[ -f .env.example ]]; then
    cp .env.example .env
    echo "Created .env from .env.example — edit credentials before live/paper use."
  else
    echo "error: .env.example is missing" >&2
    exit 1
  fi
fi

if [[ ! -x .venv/bin/python ]]; then
  echo "error: .venv not found. Create it first:" >&2
  echo "  python3.12 -m venv .venv && source .venv/bin/activate && pip install -e \".[all]\"" >&2
  exit 1
fi

if [[ ! -d frontend/node_modules ]]; then
  echo "error: frontend dependencies are missing. Run: (cd frontend && npm install)" >&2
  exit 1
fi

api_pid=""
web_pid=""
cleanup() {
  trap - INT TERM EXIT
  [[ -n "$web_pid" ]] && kill "$web_pid" 2>/dev/null || true
  [[ -n "$api_pid" ]] && kill "$api_pid" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

echo "Starting FuturesRL"
echo "  API:       http://127.0.0.1:8000/docs"
echo "  Dashboard: http://127.0.0.1:3100"
echo "Press Ctrl+C to stop both."

.venv/bin/python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload &
api_pid=$!
(cd frontend && npm run dev) &
web_pid=$!

wait "$api_pid" "$web_pid"
