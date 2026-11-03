"""QoL (task-54) integration tests: risk-lock reset + operator paper-state reset.

Everything is offline and deterministic: tmp settings/state-store paths, the
deterministic simulated execution adapter and an injected fixed clock. No IBKR,
no real network and no live environment mutations.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from configs.settings import Settings
from fastapi.testclient import TestClient
from src.api.main import create_app
from src.api.schemas import (
    PAPER_STATE_RESET_CONFIRM,
    RISK_LOCK_RESET_CONFIRM,
)
from src.risk.state_store import (
    PersistedRiskState,
    RateLimiterState,
    build_risk_state_store,
)

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
FIXED_TIMESTAMP = datetime(2026, 11, 3, 17, 0, tzinfo=UTC)


def _settings(
    tmp_path: Path,
    *,
    environment: str = "backtest",
    risk_state_store_uri: str = "",
) -> Settings:
    overrides: dict[str, object] = {
        "environment": environment,
        "data_dir": tmp_path / "data",
        "parquet_dir": tmp_path / "data" / "parquet",
        "duckdb_path": tmp_path / "data" / "qol.duckdb",
        "report_dir": tmp_path / "artifacts",
        "instruments_path": REPO_ROOT / "configs" / "instruments.yaml",
        "telemetry_interval_seconds": 0.2,
        "risk_state_store_uri": risk_state_store_uri,
    }
    if environment == "live":
        overrides.update({"ibkr_port": 7496, "ibkr_paper": False})
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


@pytest.fixture()
def client(tmp_path: Path) -> Iterator[TestClient]:
    with TestClient(create_app(_settings(tmp_path), clock=lambda: FIXED_TIMESTAMP)) as test_client:
        yield test_client


def _persisted_locked_state(
    store_path: Path,
) -> list[PersistedRiskState]:
    """Seed a durable locked risk state (kill switch + daily breaker + HWM)."""
    store = build_risk_state_store(str(store_path), duckdb_path=store_path.with_suffix(".duckdb"))
    state = PersistedRiskState(
        starting_daily_nlv=1_000_000.0,
        portfolio_high_water_mark=1_100_000.0,
        instrument_high_water_marks={"ES": 4_900.0},
        kill_switch_locked=True,
        kill_switch_reason="daily max loss breached",
        daily_loss_locked=True,
        trading_day=date(2026, 11, 3),
        rate_limiter=RateLimiterState(tokens=2.5, capacity=5.0, last_refill_epoch_seconds=123.0),
        extra={"quant_overlay.stopped_roots": "ES", "oms_keep": "kept"},
    )
    store.save(state)
    return [state]


def _write_bridge_snapshot(settings: Settings) -> Path:
    bridge_dir = settings.runner_bridge_dir
    bridge_dir.mkdir(parents=True, exist_ok=True)
    path = bridge_dir / "runner_state.json"
    path.write_text(json.dumps({"runner_id": "qol-test", "sequence": 1}), encoding="utf-8")
    return path


def _lock_execution(client: TestClient) -> None:
    response = client.post("/api/risk/flatten-all")
    assert response.status_code == 200, response.text
    assert response.json()["status"] in {"flattened", "no_positions"}


# ---------------------------------------------------------------------------
# A) risk state fields
# ---------------------------------------------------------------------------


def test_risk_state_exposes_additive_kill_switch_fields(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FUTURESRLX_OPERATOR_RESET_TOKEN", raising=False)
    body = client.get("/api/risk/state").json()
    assert body["kill_switch_active"] is False
    assert body["kill_switch_reason"] == ""
    assert body["daily_loss_breaker_active"] is False
    assert body["lock_until"] is None
    assert body["reset_available"] is True  # simulated adapter supports reset
    assert body["operator_reset_token_required"] is False


def test_risk_state_reports_lock_after_flatten(client: TestClient) -> None:
    _lock_execution(client)
    body = client.get("/api/risk/state").json()
    assert body["kill_switch_active"] is True
    assert body["kill_switch_reason"]
    assert body["lock_until"] is not None
    assert body["reset_available"] is True


# ---------------------------------------------------------------------------
# A) POST /api/risk/reset-lock
# ---------------------------------------------------------------------------


def test_reset_lock_requires_exact_confirmation_phrase(client: TestClient) -> None:
    _lock_execution(client)
    invalid = client.post("/api/risk/reset-lock", json={"confirm": "reset risk lock"})
    assert invalid.status_code == 400
    assert "RESET RISK LOCK" in invalid.json()["detail"]
    # No destructive action without the confirmation phrase.
    assert client.get("/api/risk/state").json()["kill_switch_active"] is True

    reset = client.post("/api/risk/reset-lock", json={"confirm": RISK_LOCK_RESET_CONFIRM})
    assert reset.status_code == 200, reset.text
    body = reset.json()
    assert body["status"] == "reset"
    assert body["execution_reset"] is True
    assert body["kill_switch_active"] is False
    assert body["lock_until"] is None
    after = client.get("/api/risk/state").json()
    assert after["kill_switch_active"] is False
    assert after["lock_until"] is None


def test_reset_lock_requires_operator_token_when_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FUTURESRLX_OPERATOR_RESET_TOKEN", "s3cret")
    with TestClient(create_app(_settings(tmp_path), clock=lambda: FIXED_TIMESTAMP)) as client:
        state = client.get("/api/risk/state").json()
        assert state["operator_reset_token_required"] is True
        _lock_execution(client)

        missing = client.post("/api/risk/reset-lock", json={"confirm": RISK_LOCK_RESET_CONFIRM})
        assert missing.status_code == 403
        wrong = client.post(
            "/api/risk/reset-lock",
            json={"confirm": RISK_LOCK_RESET_CONFIRM, "operator_token": "nope"},
        )
        assert wrong.status_code == 403
        # Lock stays engaged after failed authentication.
        assert client.get("/api/risk/state").json()["kill_switch_active"] is True

        via_header = client.post(
            "/api/risk/reset-lock",
            headers={"X-Operator-Token": "s3cret"},
            json={"confirm": RISK_LOCK_RESET_CONFIRM},
        )
        assert via_header.status_code == 200, via_header.text
        assert via_header.json()["kill_switch_active"] is False

        # Body token also works.
        _lock_execution(client)
        via_body = client.post(
            "/api/risk/reset-lock",
            json={"confirm": RISK_LOCK_RESET_CONFIRM, "operator_token": "s3cret"},
        )
        assert via_body.status_code == 200, via_body.text


def test_reset_lock_clears_persisted_risk_state(tmp_path: Path) -> None:
    store_path = tmp_path / "risk_state.json"
    _persisted_locked_state(store_path)
    with TestClient(
        create_app(
            _settings(tmp_path, risk_state_store_uri=str(store_path)),
            clock=lambda: FIXED_TIMESTAMP,
        )
    ) as client:
        before = client.get("/api/risk/state").json()
        assert before["kill_switch_active"] is True
        assert before["daily_loss_breaker_active"] is True
        assert before["kill_switch_reason"] == "daily max loss breached"

        response = client.post("/api/risk/reset-lock", json={"confirm": RISK_LOCK_RESET_CONFIRM})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["persisted_state_reset"] is True
        assert body["daily_loss_breaker_active"] is False

        store = build_risk_state_store(
            str(store_path), duckdb_path=store_path.with_suffix(".duckdb")
        )
        persisted = store.load()
        assert persisted is not None
        # Kill switch + daily latch + HWM/trailing-stop state cleared...
        assert persisted.kill_switch_locked is False
        assert persisted.kill_switch_reason == ""
        assert persisted.daily_loss_locked is False
        assert persisted.starting_daily_nlv is None
        assert persisted.portfolio_high_water_mark is None
        assert dict(persisted.instrument_high_water_marks) == {}
        assert "quant_overlay.stopped_roots" not in persisted.extra
        # ...while OMS-owned fields survive.
        assert persisted.rate_limiter == RateLimiterState(
            tokens=2.5, capacity=5.0, last_refill_epoch_seconds=123.0
        )
        assert persisted.extra["oms_keep"] == "kept"
        after = client.get("/api/risk/state").json()
        assert after["kill_switch_active"] is False
        assert after["daily_loss_breaker_active"] is False


def test_reset_lock_already_unlocked_is_idempotent(client: TestClient) -> None:
    first = client.post("/api/risk/reset-lock", json={"confirm": RISK_LOCK_RESET_CONFIRM})
    second = client.post("/api/risk/reset-lock", json={"confirm": RISK_LOCK_RESET_CONFIRM})
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["status"] == "already_unlocked"
    assert second.json()["status"] == "already_unlocked"


def test_reset_lock_refused_while_flatten_is_in_progress(client: TestClient) -> None:
    _lock_execution(client)
    container = client.app.state.container
    container.execution._flatten_in_progress = True
    try:
        response = client.post("/api/risk/reset-lock", json={"confirm": RISK_LOCK_RESET_CONFIRM})
    finally:
        container.execution._flatten_in_progress = False
    assert response.status_code == 409
    assert "flatten" in response.json()["detail"]
    assert client.get("/api/risk/state").json()["kill_switch_active"] is True


# ---------------------------------------------------------------------------
# B) POST /api/maintenance/reset-paper-state
# ---------------------------------------------------------------------------


def test_paper_state_reset_clears_bridge_store_and_session(tmp_path: Path) -> None:
    store_path = tmp_path / "risk_state.json"
    settings = _settings(tmp_path, environment="paper", risk_state_store_uri=str(store_path))
    bridge_path = _write_bridge_snapshot(settings)
    _persisted_locked_state(store_path)

    with TestClient(create_app(settings, clock=lambda: FIXED_TIMESTAMP)) as client:
        wrong = client.post("/api/maintenance/reset-paper-state", json={"confirm": "nope"})
        assert wrong.status_code == 400
        assert bridge_path.is_file()  # no destructive action without confirmation
        assert build_risk_state_store(str(store_path)).load() is not None

        response = client.post(
            "/api/maintenance/reset-paper-state",
            json={"confirm": PAPER_STATE_RESET_CONFIRM},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["status"] == "reset"
        assert body["environment"] == "paper"
        assert body["confirm_ok"] is True
        receipts = body["receipts"]
        assert receipts["risk_state_store_cleared"] is True
        assert receipts["risk_state_store_uri"] == str(store_path)
        assert receipts["paper_session_reset"] is True
        assert str(bridge_path) in receipts["files_removed"]
        assert not bridge_path.exists()
        assert build_risk_state_store(str(store_path)).load() is None
        assert body["warnings"] == []

        # Idempotent: a second reset is safe and reports nothing left to remove.
        again = client.post(
            "/api/maintenance/reset-paper-state",
            json={"confirm": PAPER_STATE_RESET_CONFIRM},
        )
        assert again.status_code == 200, again.text
        assert again.json()["receipts"]["files_removed"] == []
        assert again.json()["receipts"]["risk_state_store_cleared"] is False
        assert not bridge_path.exists()


def test_paper_state_reset_refuses_live_environment(tmp_path: Path) -> None:
    settings = _settings(tmp_path, environment="live")
    bridge_path = _write_bridge_snapshot(settings)
    with TestClient(create_app(settings, clock=lambda: FIXED_TIMESTAMP)) as client:
        response = client.post(
            "/api/maintenance/reset-paper-state",
            json={"confirm": PAPER_STATE_RESET_CONFIRM},
        )
        assert response.status_code == 409
        assert "live" in response.json()["detail"]
        # Refusal is non-destructive.
        assert bridge_path.is_file()


def test_paper_state_reset_respects_flags(tmp_path: Path) -> None:
    store_path = tmp_path / "risk_state.json"
    settings = _settings(tmp_path, environment="backtest", risk_state_store_uri=str(store_path))
    bridge_path = _write_bridge_snapshot(settings)
    _persisted_locked_state(store_path)
    with TestClient(create_app(settings, clock=lambda: FIXED_TIMESTAMP)) as client:
        response = client.post(
            "/api/maintenance/reset-paper-state",
            json={
                "confirm": PAPER_STATE_RESET_CONFIRM,
                "clear_risk_state": False,
                "clear_runner_bridge": False,
                "clear_paper_cache": False,
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "noop"
        assert body["receipts"]["files_removed"] == []
        assert body["receipts"]["risk_state_store_cleared"] is False
        assert bridge_path.is_file()
        assert build_risk_state_store(str(store_path)).load() is not None


def test_unknown_maintenance_route_is_404(client: TestClient) -> None:
    assert client.post("/api/maintenance/does-not-exist", json={}).status_code == 404
