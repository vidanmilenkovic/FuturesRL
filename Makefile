# FuturesRL developer entry points
PY ?= .venv/bin/python

.PHONY: install lint format format-check type test cov gate dev up backtest api frontend docker

install:
	$(PY) -m pip install -e ".[dev]"

lint:
	$(PY) -m ruff check .

format:
	$(PY) -m ruff format .

format-check:
	$(PY) -m ruff format --check .

type:
	$(PY) -m mypy src configs

test:
	$(PY) -m pytest tests/ -q

cov:
	$(PY) -m pytest tests/ --cov=src/core --cov=src/risk --cov-report=term-missing --cov-fail-under=85

gate: lint format-check type cov

# One-command local stack: backend :8000 + dashboard :3100
dev:
	./scripts/dev.sh

up: dev

backtest:
	$(PY) -m src.backtest.engine --strategy configs/strategies/tsmom_vol_target.yaml --offline

api:
	$(PY) -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000

frontend:
	cd frontend && npm run dev

docker:
	docker compose -f docker/docker-compose.yml up --build
