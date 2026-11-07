# FuturesRL · Docker deployment

Containerised deployment of the FuturesRL workstation: FastAPI backend,
Next.js dashboard and an optional headless IB Gateway for broker connectivity.
Storage is **local-first** (partitioned Parquet, Polars, DuckDB) inside the
backend volume — there is no external database or cache service.

```
┌──────────────────┐   REST + WS    ┌──────────────────┐
│ frontend :3100   │ ─────────────► │ backend :8000    │
│ Next.js standalone│               │ FastAPI + /ws    │
└──────────────────┘                └────────┬─────────┘
                                             │
                                  ┌──────────▼──────────┐
                                  │ backend_data volume │
                                  │ Parquet + DuckDB    │
                                  └─────────────────────┘
                                             ▲
                                             │ 4003 (paper API, optional)
                                  ┌──────────┴──────────┐
                                  │ ib-gateway (profile)│
                                  │ 5900 noVNC          │
                                  └─────────────────────┘
```

## Quick start

```bash
cp docker/.env.example docker/.env      # optional: IBKR paper credentials
docker compose -f docker/docker-compose.yml up --build

# With the optional headless IB Gateway:
docker compose -f docker/docker-compose.yml --profile ibkr up --build
```

| Service | URL / port | Notes |
| --- | --- | --- |
| frontend | <http://localhost:3100> | Next.js workstation; honest error banner if API is down |
| backend | <http://localhost:8000> | REST `/api/*`, `/health`, `/ws/telemetry`, docs `/docs` |
| ib-gateway | `127.0.0.1:4003` (paper API) | Optional profile; noVNC on `127.0.0.1:5900` |
| storage | backend volume | Parquet cache + DuckDB; no external database/Redis |

The backend starts in **offline mode** when IBKR/Databento are unavailable: it
serves deterministic fixture data, so the stack remains usable end-to-end.

## Files

| File | Purpose |
| --- | --- |
| `Dockerfile.backend` | Multi-stage `python:3.12-slim`; non-root uvicorn `src.api.main:app` |
| `Dockerfile.frontend` | Multi-stage `node:22-alpine`; Next.js standalone runner |
| `docker-compose.yml` | Backend + frontend + optional IB Gateway profile; local volumes |
| `entrypoint.backend.sh` | Creates local data/artifact directories, then execs uvicorn |
| `.env.example` | Documented environment template (copy to `.env`, git-ignored) |
| `.dockerignore` | Canonical backend build-context ignore list |

> **Build context note.** The backend image needs the repository root as build
> context (`pyproject.toml` + `src/` + `configs/`). Pass the docker ignore file
> explicitly when the repository root has no `.dockerignore`:
> `docker buildx build --ignorefile docker/.dockerignore -f docker/Dockerfile.backend -t futuresrl/backend:1.0.0 .`

## Environment variables

| Variable | Default | Description |
| --- | --- | --- |
| `FUTURESRL_ENVIRONMENT` | `paper` | `development` / `paper` / `live` / `backtest` |
| `TWS_USERID` / `TWS_PASSWORD` | empty | Optional IBKR paper credentials for the gateway profile |
| `IB_GATEWAY_TRADING_MODE` | `paper` | `paper` or `live` (live uses internal/published port 4004) |
| `IB_GATEWAY_PORT` / `IB_GATEWAY_INTERNAL_PORT` / `IB_GATEWAY_VNC_PORT` | `4003` / `4003` / `5900` | Optional Gateway API / noVNC ports |
| `FUTURESRL_IBKR_HOST` / `_PORT` | `ib-gateway` / `4003` | Gateway endpoint inside the compose network |
| `FUTURESRL_API_ENABLE_IBKR` | `0` | `1` attempts a real IBKR session; offline fallback remains |
| `DATABENTO_API_KEY` | empty | Optional; blank keeps deterministic offline data |
| `API_PORT` / `FRONTEND_PORT` | `8000` / `3100` | Published ports |
| `NEXT_PUBLIC_API_BASE_URL` | `http://localhost:8000` | Browser-visible API origin (build-time) |
| `NEXT_PUBLIC_WS_URL` | `ws://localhost:8000/ws/telemetry` | Browser-visible telemetry URL |

## Healthchecks

| Service | Check |
| --- | --- |
| backend | `GET /health` via curl |
| frontend | `GET /` via wget |
| ib-gateway | Optional profile; no healthcheck is required for offline mode |

## Operational notes

- No secrets are committed: `.env` is git-ignored; the compose file contains
  only safe defaults and `${VAR}` interpolation.
- The optional IB Gateway API and noVNC ports bind to `127.0.0.1` only.
- IB Gateway may require an interactive 2FA login on first start; open
  `http://127.0.0.1:5900` to complete it.
- `backend_data` (Parquet + DuckDB) and `backend_artifacts` (tear sheets,
  studies, models) persist across restarts.
- No external database, Redis or message broker is required.
