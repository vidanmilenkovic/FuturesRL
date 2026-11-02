# FuturesRL-X · Docker deployment

Containerised deployment of the full platform: FastAPI telemetry backend,
Next.js dashboard, TimescaleDB, Redis and a headless IB Gateway (paper) for
broker connectivity.

```
┌──────────────────┐   REST + WS    ┌──────────────────┐
│ frontend :3080   │ ─────────────► │ backend :8000    │
│ Next.js standalone│               │ FastAPI + /ws    │
└──────────────────┘                └────┬──────┬──────┘
                                          │      │
                        ┌─────────────────┘      └──────────────┐
                        ▼                                        ▼
                ┌───────────────┐                        ┌───────────────┐
                │ timescaledb   │                        │ redis :6379   │
                │ (internal)    │                        │ (internal)    │
                └───────────────┘                        └───────────────┘
                        ▲
                        │ 4003 (paper API)
                ┌───────────────┐
                │ ib-gateway    │  ghcr.io/gnzsnz/ib-gateway
                │ 5900 noVNC    │
                └───────────────┘
```

## Quick start

```bash
cp docker/.env.example docker/.env      # edit secrets (IBKR paper credentials)
docker compose -f docker/docker-compose.yml up --build
```

| Service      | URL / port                    | Notes                                                       |
| ------------ | ----------------------------- | ----------------------------------------------------------- |
| frontend     | <http://localhost:3080>       | Next.js dashboard; shows DEMO DATA if the API is down       |
| backend      | <http://localhost:8000>       | REST `/api/*`, `/health`, `/ws/telemetry`, OpenAPI `/docs`  |
| ib-gateway   | `127.0.0.1:4003` (paper API)  | VNC/noVNC on `127.0.0.1:5900` for the Gateway login screen   |
| timescaledb  | `127.0.0.1:5432` (loopback)   | internal `timescaledb:5432` on `futuresrlx_net`             |
| redis        | internal `redis:6379`         | not published to the host                                   |

The backend starts in **offline mode** when IBKR/Databento are unavailable: it
serves deterministic fixture data, so the stack remains demo-able end-to-end.

## Files

| File                          | Purpose                                                            |
| ----------------------------- | ------------------------------------------------------------------ |
| `Dockerfile.backend`          | Multi-stage `python:3.12-slim`; runtime deps extracted from `pyproject.toml`; non-root; uvicorn `src.api.main:app` |
| `Dockerfile.frontend`         | Multi-stage `node:22-alpine`; `next build` → `output: "standalone"` runner |
| `docker-compose.yml`          | 5 services, named volumes, healthchecks, bridge network            |
| `entrypoint.backend.sh`       | Best-effort DB wait, data dir bootstrap, then `exec "$@"`          |
| `.env.example`                | Documented environment template (copy to `.env`, git-ignored)      |
| `.dockerignore`               | Canonical backend build-context ignore list (see note below)       |

> **Build context note.** The backend image needs the repository root as build
> context (`pyproject.toml` + `src/` + `configs/`). Docker only reads a
> `.dockerignore` from the context root; if the repository has no root
> `.dockerignore`, pass this one explicitly:
> `docker buildx build --ignorefile docker/.dockerignore -f docker/Dockerfile.backend -t futuresrlx/backend:1.0.0 .`
> The frontend build uses the small `frontend/` context and therefore honours
> `frontend/.dockerignore`.

## Environment variables

Compose interpolates `docker/.env` (never committed). Full list with defaults
lives in `.env.example`; the important ones:

| Variable                       | Default                | Description                                                        |
| ------------------------------ | ---------------------- | ------------------------------------------------------------------ |
| `FUTURESRLX_ENVIRONMENT`       | `paper`                | `development` / `paper` / `live` / `backtest`                      |
| `POSTGRES_USER/PASSWORD/DB`    | `futuresrlx`           | TimescaleDB credentials (change the password)                      |
| `REDIS_PASSWORD`               | `change-me-redis`      | Redis AUTH password                                                |
| `TWS_USERID` / `TWS_PASSWORD`  | *(empty)*              | IBKR **paper** credentials for the headless gateway                |
| `IB_GATEWAY_PORT`              | `4003`                 | Published paper Gateway API port (loopback only)                    |
| `FUTURESRLX_IBKR_HOST`         | `ib-gateway`           | Hostname the backend uses to reach the Gateway                     |
| `FUTURESRLX_IBKR_PORT`         | `4003`                 | Paper Gateway API port inside the compose network                  |
| `FUTURESRLX_API_ENABLE_IBKR`   | `0`                    | `1` attempts a real IBKR session (offline simulated fallback stays) |
| `DATABENTO_API_KEY`            | *(empty)*              | Optional; blank → offline deterministic fixtures                   |
| `API_PORT`                     | `8000`                 | Published backend port                                             |
| `FRONTEND_PORT`                | `3080`                 | Published dashboard port                                           |
| `NEXT_PUBLIC_API_BASE_URL`     | `http://localhost:8000`| Browser-visible API origin, baked into the Next.js bundle at build |
| `NEXT_PUBLIC_WS_URL`           | `ws://localhost:8000/ws/telemetry` | Browser-visible telemetry WebSocket URL                 |

## Healthchecks

| Service     | Check                                                    |
| ----------- | -------------------------------------------------------- |
| timescaledb | `pg_isready`                                              |
| redis       | `redis-cli ping` (authenticated)                          |
| backend     | `GET /health` via curl                                    |
| frontend    | `GET /` via wget                                          |

## Operational notes

- No secrets are committed: `.env` (if copied) is git-ignored; the compose file
  only contains safe defaults and `${VAR}` interpolation.
- Database and IB Gateway API ports bind to `127.0.0.1` only. The dashboard and
  REST API are the intended public surface.
- IB Gateway requires an interactive 2FA login on first start; `TWOFA_TIMEOUT_ACTION=restart`
  and `RELOGIN_AFTER_TWOFA_TIMEOUT=yes` keep the session recoverable. Open
  `http://127.0.0.1:5900` to complete the login when prompted.
- `backend_data` (Parquet cache + DuckDB) and `backend_artifacts` (backtest tear
  sheets) persist across restarts.
- The API tolerates a missing IB Gateway: `depends_on: service_started` plus
  lazy broker imports means telemetry continues from the offline engine.
