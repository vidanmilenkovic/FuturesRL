#!/usr/bin/env sh
# FuturesRL backend entrypoint: bootstrap local-first storage and run the API.
set -eu

echo "[entrypoint] FuturesRL backend starting (environment=${FUTURESRL_ENVIRONMENT:-development})"
mkdir -p "${FUTURESRL_DATA_DIR:-/app/data}/parquet" "${FUTURESRL_REPORT_DIR:-/app/artifacts}"
exec "$@"
