"""Unit tests for the durable risk/execution state store (Phase 7)."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from src.risk.state_store import (
    DuckDBRiskStateStore,
    FileRiskStateStore,
    InMemoryRiskStateStore,
    PersistedRiskState,
    RateLimiterState,
    RedisRiskStateStore,
    StateStoreError,
    build_risk_state_store,
)

NOW = datetime(2024, 6, 12, 18, 30, tzinfo=UTC)


def _state(**overrides: object) -> PersistedRiskState:
    base: dict[str, object] = {
        "starting_daily_nlv": 1_000_000.0,
        "portfolio_high_water_mark": 1_050_000.0,
        "instrument_high_water_marks": {"ES": 4_850.0, "CL": 78.2},
        "kill_switch_locked": True,
        "kill_switch_reason": "daily max loss breached",
        "daily_loss_locked": True,
        "trading_day": date(2024, 6, 12),
        "rate_limiter": RateLimiterState(tokens=2.5, capacity=5.0, last_refill_epoch_seconds=1.0),
        "updated_at": NOW,
    }
    base.update(overrides)
    return PersistedRiskState(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# RateLimiterState
# ---------------------------------------------------------------------------


def test_rate_limiter_validates_and_refills() -> None:
    full = RateLimiterState.full(5.0, now_epoch_seconds=100.0)
    assert full.tokens == pytest.approx(5.0)
    refilled = full.refilled(now_epoch_seconds=101.5, rate_per_second=2.0)
    assert refilled.tokens == pytest.approx(5.0)  # clamped at capacity
    drained = RateLimiterState(tokens=0.0, capacity=5.0, last_refill_epoch_seconds=100.0)
    assert drained.refilled(now_epoch_seconds=101.5, rate_per_second=2.0).tokens == pytest.approx(
        3.0
    )
    # Clocks going backwards never refund tokens.
    assert drained.refilled(now_epoch_seconds=90.0, rate_per_second=2.0).tokens == pytest.approx(
        0.0
    )
    with pytest.raises(StateStoreError, match="capacity"):
        RateLimiterState(tokens=1.0, capacity=0.0, last_refill_epoch_seconds=1.0)
    with pytest.raises(StateStoreError, match="tokens"):
        RateLimiterState(tokens=9.0, capacity=5.0, last_refill_epoch_seconds=1.0)
    with pytest.raises(StateStoreError, match="anchor"):
        RateLimiterState(tokens=1.0, capacity=5.0, last_refill_epoch_seconds=float("nan"))


def test_rate_limiter_round_trip() -> None:
    state = RateLimiterState(tokens=2.5, capacity=5.0, last_refill_epoch_seconds=123.0)
    assert RateLimiterState.from_dict(state.to_dict()) == state
    with pytest.raises(StateStoreError, match="invalid rate_limiter"):
        RateLimiterState.from_dict({"tokens": 1.0})


# ---------------------------------------------------------------------------
# PersistedRiskState
# ---------------------------------------------------------------------------


def test_persisted_state_validation() -> None:
    state = _state()
    assert state.starting_daily_nlv == pytest.approx(1_000_000.0)
    with pytest.raises(StateStoreError, match="starting_daily_nlv"):
        _state(starting_daily_nlv=-5.0)
    with pytest.raises(StateStoreError, match="portfolio_high_water_mark"):
        _state(portfolio_high_water_mark=0.0)
    with pytest.raises(StateStoreError, match="reason"):
        _state(kill_switch_reason="")
    with pytest.raises(StateStoreError, match="instrument_high_water_marks"):
        _state(instrument_high_water_marks={"ES": -1.0})
    with pytest.raises(StateStoreError, match=r"naive|timezone"):
        _state(updated_at=datetime(2024, 6, 12, 18, 30))
    with pytest.raises(StateStoreError, match="version"):
        _state(version=0)


def test_persisted_state_mutations_keep_risk_latches() -> None:
    state = _state()
    rolled = state.rolled_to_day(date(2024, 6, 13))
    assert rolled.starting_daily_nlv is None
    assert rolled.daily_loss_locked is False
    assert rolled.trading_day == date(2024, 6, 13)
    assert rolled.kill_switch_locked is True  # trailing-stop halt survives midnight
    assert rolled.portfolio_high_water_mark == pytest.approx(1_050_000.0)
    unlocked = state.unlocked()
    assert unlocked.kill_switch_locked is False
    assert unlocked.kill_switch_reason == ""
    locked = PersistedRiskState().locked("operator emergency")
    assert locked.kill_switch_locked is True


def test_persisted_state_round_trip() -> None:
    state = _state()
    payload = state.to_dict()
    restored = PersistedRiskState.from_dict(payload)
    assert restored.starting_daily_nlv == pytest.approx(1_000_000.0)
    assert restored.portfolio_high_water_mark == pytest.approx(1_050_000.0)
    assert dict(restored.instrument_high_water_marks) == {"ES": 4_850.0, "CL": 78.2}
    assert restored.kill_switch_locked is True
    assert restored.daily_loss_locked is True
    assert restored.trading_day == date(2024, 6, 12)
    assert restored.rate_limiter == state.rate_limiter
    assert restored.updated_at == NOW


def test_persisted_state_rejects_future_version() -> None:
    payload = _state().to_dict()
    payload["version"] = 999
    with pytest.raises(StateStoreError, match="newer than supported"):
        PersistedRiskState.from_dict(payload)


# ---------------------------------------------------------------------------
# In-memory backend
# ---------------------------------------------------------------------------


def test_in_memory_store_round_trip() -> None:
    store = InMemoryRiskStateStore()
    assert store.load() is None
    store.save(_state())
    assert store.load() is not None
    store.clear()
    assert store.load() is None
    with pytest.raises(TypeError, match="PersistedRiskState"):
        store.save("nope")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# File backend
# ---------------------------------------------------------------------------


def test_file_store_round_trip_is_atomic(tmp_path: Path) -> None:
    path = tmp_path / "risk_state.json"
    store = FileRiskStateStore(path)
    store.save(_state())
    assert path.is_file()
    assert list(tmp_path.glob("*.tmp")) == []
    restored = store.load()
    assert restored is not None
    assert restored.starting_daily_nlv == pytest.approx(1_000_000.0)
    assert store.path == path.resolve()
    store.clear()
    assert store.load() is None
    store.clear()  # idempotent


def test_file_store_directory_path_uses_default_filename(tmp_path: Path) -> None:
    store = FileRiskStateStore(tmp_path)
    store.save(PersistedRiskState(starting_daily_nlv=500_000.0))
    assert (tmp_path / "risk_state.json").is_file()


def test_file_store_quarantines_corrupt_payload(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "risk_state.json"
    path.write_text("{not-json", encoding="utf-8")
    store = FileRiskStateStore(path)
    with caplog.at_level("ERROR"):
        assert store.load() is None
    quarantined = list(tmp_path.glob("risk_state.json.corrupt-*"))
    assert len(quarantined) == 1
    assert "corrupt" in caplog.text


def test_file_store_rejects_unknown_shape(tmp_path: Path) -> None:
    path = tmp_path / "risk_state.json"
    path.write_text(json.dumps(["not", "a", "mapping"]), encoding="utf-8")
    assert FileRiskStateStore(path).load() is None
    assert list(tmp_path.glob("risk_state.json.corrupt-*"))


def test_file_store_namespace_isolation(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    first = FileRiskStateStore(path, namespace_key="first")
    second = FileRiskStateStore(path, namespace_key="second")
    first.save(PersistedRiskState(starting_daily_nlv=100.0))
    assert second.load() is None
    assert first.load() is not None


# ---------------------------------------------------------------------------
# DuckDB backend
# ---------------------------------------------------------------------------


def test_duckdb_store_round_trip_and_reload(tmp_path: Path) -> None:
    db_path = tmp_path / "state.duckdb"
    store = DuckDBRiskStateStore(db_path)
    store.save(_state())
    restored = store.load()
    assert restored is not None
    assert restored.portfolio_high_water_mark == pytest.approx(1_050_000.0)
    # A new store instance (simulated container restart) sees the same state.
    reopened = DuckDBRiskStateStore(db_path)
    assert reopened.load() is not None
    reopened.clear()
    assert reopened.load() is None


def test_duckdb_store_ignores_corrupt_row(tmp_path: Path) -> None:
    db_path = tmp_path / "state.duckdb"
    store = DuckDBRiskStateStore(db_path)
    store.save(_state())
    connection = store._connect()  # type: ignore[attr-defined]
    connection.execute("UPDATE risk_state SET payload = ?", ["{bad-json"])
    assert store.load() is None


# ---------------------------------------------------------------------------
# Redis backend
# ---------------------------------------------------------------------------


class _FakeRedis:
    def __init__(self) -> None:
        self.data: dict[str, object] = {}

    def get(self, key: str) -> object | None:
        return self.data.get(key)

    def set(self, key: str, value: str) -> None:
        self.data[key] = value

    def delete(self, key: str) -> None:
        self.data.pop(key, None)


def test_redis_store_round_trip_bytes_and_clear() -> None:
    client = _FakeRedis()
    store = RedisRiskStateStore(client, key="test:risk")
    assert store.load() is None
    store.save(_state())
    assert store.key == "test:risk"
    client.data["test:risk"] = str(client.data["test:risk"]).encode()
    restored = store.load()
    assert restored is not None
    assert restored.kill_switch_locked is True
    store.clear()
    assert store.load() is None


def test_redis_store_rejects_corrupt_payload() -> None:
    client = _FakeRedis()
    store = RedisRiskStateStore(client)
    client.data["futuresrlx:risk_state"] = "not-json"
    assert store.load() is None
    client.data["futuresrlx:risk_state"] = json.dumps([1, 2, 3])
    assert store.load() is None


def test_redis_store_validates_client() -> None:
    with pytest.raises(TypeError, match="get"):
        RedisRiskStateStore(object())


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def test_build_risk_state_store_variants(tmp_path: Path) -> None:
    assert isinstance(build_risk_state_store(None), InMemoryRiskStateStore)
    assert isinstance(build_risk_state_store("memory:"), InMemoryRiskStateStore)
    assert isinstance(build_risk_state_store(tmp_path / "s.json"), FileRiskStateStore)
    assert isinstance(build_risk_state_store(f"file:{tmp_path / 'f.json'}"), FileRiskStateStore)
    db_path = tmp_path / "state.duckdb"
    assert isinstance(build_risk_state_store("duckdb:", duckdb_path=db_path), DuckDBRiskStateStore)
    assert isinstance(
        build_risk_state_store("redis:k", redis_client=_FakeRedis()), RedisRiskStateStore
    )
    with pytest.raises(ValueError, match="duckdb_path"):
        build_risk_state_store("duckdb:")
    with pytest.raises(ValueError, match="redis client"):
        build_risk_state_store("redis:")
