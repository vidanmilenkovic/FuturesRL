#!/usr/bin/env sh
# ---------------------------------------------------------------------------
# FuturesRL backend entrypoint
#
# Waits for the optional TimescaleDB/Redis sidecars (best effort — the API is
# designed to run fully offline and must never crash because a dependency is
# still starting), then execs the CMD (uvicorn by default).
# ---------------------------------------------------------------------------
set -eu

host="${FUTURESRL_DB_HOST:-timescaledb}"
port="${FUTURESRL_DB_PORT:-5432}"
timeout="${FUTURESRL_DB_WAIT_SECONDS:-60}"

echo "[entrypoint] FuturesRL backend starting (environment=${FUTURESRL_ENVIRONMENT:-development})"

if [ "${FUTURESRL_DB_WAIT:-1}" = "1" ]; then
  elapsed=0
  while [ "$elapsed" -lt "$timeout" ]; do
    if (echo > "/dev/tcp/${host}/${port}") >/dev/null 2>&1; then
      echo "[entrypoint] database reachable at ${host}:${port}"
      break
    fi
    elapsed=$((elapsed + 2))
    echo "[entrypoint] waiting for ${host}:${port} (${elapsed}s/${timeout}s)"
    sleep 2
  done
  if [ "$elapsed" -ge "$timeout" ]; then
    echo "[entrypoint] WARNING: ${host}:${port} not reachable after ${timeout}s; starting API in offline mode"
  fi
fi

mkdir -p "${FUTURESRL_DATA_DIR:-/app/data}/parquet" "${FUTURESRL_REPORT_DIR:-/app/artifacts}"

exec "$@"
