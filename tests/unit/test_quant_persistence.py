"""Phase-7 unit tests: durable risk / kill-switch state for the quant overlay.

Every test is offline and deterministic. A restart is simulated by constructing a
*new* :class:`QuantRiskOverlay` over the same :class:`RiskStateStore`, which is
exactly what happens on a container redeploy or an IBKR nightly reset.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from configs.settings import RiskSettings
from src.core.types import MarketState
from src.risk.quant_risk_overlay import QuantRiskConfig, QuantRiskOverlay
from src.risk.state_store import (
    FileRiskStateStore,
    InMemoryRiskStateStore,
    PersistedRiskState,
    RateLimiterState,
)

from tests.fixtures.market import FIXED_TIMESTAMP, make_market_state, make_weight_vector

NEXT_DAY = datetime(2024, 1, 3, 21, 0, tzinfo=UTC)
LATER_INTRADAY = datetime(2024, 1, 2, 22, 0, tzinfo=UTC)

#: OMS-owned fields that must survive the risk overlay's writes.
OMS_RATE_LIMITER = RateLimiterState(tokens=2.5, capacity=5.0, last_refill_epoch_seconds=123.0)


def _config(**overrides: float | bool) -> QuantRiskConfig:
    """Permissive config: only the control under test is isolated by default."""
    base: dict[str, float | bool] = {
        "volatility_targeting_enabled": False,
        "max_single_instrument_weight": 10.0,
        "max_sector_weight": 10.0,
        "max_gross_leverage": 10.0,
        "trailing_stop_pct": 0.99,
        "drawdown_limit_pct": 0.99,
        "daily_max_loss_pct": 0.03,
    }
    base.update(overrides)
    return QuantRiskConfig(**base)  # type: ignore[arg-type]


def _overlay(
    store: InMemoryRiskStateStore | FileRiskStateStore | None = None,
    **overrides: float | bool,
) -> QuantRiskOverlay:
    return QuantRiskOverlay(
        config=_config(**overrides),
        clock=lambda: FIXED_TIMESTAMP,
        state_store=store,
    )


def _state(
    *,
    equity: float = 1_000_000.0,
    prices: dict[str, float] | None = None,
    timestamp: datetime = FIXED_TIMESTAMP,
) -> MarketState:
    return make_market_state(
        equity=equity,
        prices=prices or {},
        with_covariance=False,
        with_returns=False,
        timestamp=timestamp,
    )


class _ExplodingStore:
    """A store whose every operation fails; must never crash the overlay."""

    def load(self) -> PersistedRiskState | None:
        raise RuntimeError("store unavailable")

    def save(self, state: PersistedRiskState) -> None:
        raise RuntimeError("disk full")

    def clear(self) -> None:
        raise RuntimeError("store unavailable")


# ---------------------------------------------------------------------------
# Baseline: no store configured is unchanged
# ---------------------------------------------------------------------------


def test_store_none_keeps_existing_behaviour() -> None:
    overlay = _overlay(None, daily_max_loss_pct=0.03)
    assert overlay.state_store is None
    assert overlay.state_store_errors == 0
    assert overlay.last_state_store_error is None
    assert overlay.restore_from_store() is False
    overlay.update_equity(1_000_000.0, timestamp=FIXED_TIMESTAMP)
    overlay.update_equity(960_000.0, timestamp=FIXED_TIMESTAMP)
    assert overlay.kill_switch_active is True
    assert overlay.apply_risk(make_weight_vector({"ES": 0.5}), _state(equity=960_000.0)).is_flat()
    # persist() still returns the durable representation even without a store.
    snapshot = overlay.persist()
    assert isinstance(snapshot, PersistedRiskState)
    assert snapshot.starting_daily_nlv == pytest.approx(1_000_000.0)
    assert snapshot.daily_loss_locked is True
    assert overlay.persisted_state.daily_loss_locked is True


def test_invalid_state_store_rejected() -> None:
    with pytest.raises(TypeError, match="RiskStateStore"):
        QuantRiskOverlay(config=_config(), state_store=object())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Auto-persist on equity / HWM / session mutations
# ---------------------------------------------------------------------------


def test_update_equity_persists_daily_nlv_hwm_and_trading_day() -> None:
    store = InMemoryRiskStateStore()
    overlay = _overlay(store)
    overlay.update_equity(1_000_000.0, timestamp=FIXED_TIMESTAMP)
    overlay.update_equity(1_050_000.0, timestamp=LATER_INTRADAY)
    # Re-marking the same equity is a no-op (no needless persistence write).
    overlay.update_equity(1_050_000.0, timestamp=LATER_INTRADAY)

    persisted = store.load()
    assert persisted is not None
    assert persisted.starting_daily_nlv == pytest.approx(1_000_000.0)
    assert persisted.portfolio_high_water_mark == pytest.approx(1_050_000.0)
    assert persisted.trading_day == date(2024, 1, 2)
    assert persisted.kill_switch_locked is False
    assert persisted.daily_loss_locked is False
    assert persisted.updated_at is not None
    json.dumps(persisted.to_dict())  # must be strictly JSON-serialisable


def test_apply_risk_persists_instrument_high_water_marks() -> None:
    store = InMemoryRiskStateStore()
    overlay = _overlay(store)
    overlay.update_equity(1_000_000.0, timestamp=FIXED_TIMESTAMP)
    overlay.apply_risk(make_weight_vector({"ES": 0.5}), _state(prices={"ES": 4_900.0}))

    persisted = store.load()
    assert persisted is not None
    assert persisted.instrument_high_water_marks["ES"] == pytest.approx(4_900.0)

    # A higher mark raises the stored HWM.
    overlay.apply_risk(make_weight_vector({"ES": 0.5}), _state(prices={"ES": 5_000.0}))
    updated = store.load()
    assert updated is not None
    assert updated.instrument_high_water_marks["ES"] == pytest.approx(5_000.0)


def test_persist_preserves_oms_owned_fields() -> None:
    store = InMemoryRiskStateStore(
        PersistedRiskState(rate_limiter=OMS_RATE_LIMITER, extra={"oms": "keep-me"})
    )
    overlay = _overlay(store)
    overlay.update_equity(1_000_000.0, timestamp=FIXED_TIMESTAMP)
    overlay.apply_risk(make_weight_vector({"ES": 0.5}), _state(prices={"ES": 4_850.0}))

    persisted = store.load()
    assert persisted is not None
    # The OMS token bucket and unrelated forward-compatible keys survive.
    assert persisted.rate_limiter == OMS_RATE_LIMITER
    assert persisted.extra["oms"] == "keep-me"
    # ... while the overlay's own direction metadata is added.
    assert persisted.extra["quant_overlay.hwm_direction:ES"] == 1


# ---------------------------------------------------------------------------
# Simulated restart: daily NLV / HWM / kill switch
# ---------------------------------------------------------------------------


def test_restart_restores_daily_nlv_and_portfolio_hwm() -> None:
    store = InMemoryRiskStateStore()
    first = _overlay(store)
    first.update_equity(1_000_000.0, timestamp=FIXED_TIMESTAMP)
    first.update_equity(1_050_000.0, timestamp=LATER_INTRADAY)

    restarted = _overlay(store)
    snapshot = restarted.snapshot_state()
    assert snapshot["starting_daily_equity"] == pytest.approx(1_000_000.0)
    assert snapshot["portfolio_high_water_mark"] == pytest.approx(1_050_000.0)
    assert snapshot["session_date"] == "2024-01-02"
    assert restarted.kill_switch_active is False


def test_daily_max_loss_kill_switch_stays_latched_across_restart() -> None:
    store = InMemoryRiskStateStore()
    first = _overlay(store, daily_max_loss_pct=0.03)
    first.update_equity(1_000_000.0, timestamp=FIXED_TIMESTAMP)
    first.update_equity(970_000.0, timestamp=FIXED_TIMESTAMP)
    assert first.kill_switch_active is True

    restarted = _overlay(store, daily_max_loss_pct=0.03)
    assert restarted.kill_switch_active is True
    assert restarted.daily_loss_latched is True
    # A restart (or a partial recovery) must NOT re-arm the consumed risk budget.
    weights = make_weight_vector({"ES": 0.5, "NQ": 0.25})
    assert restarted.apply_risk(weights, _state(equity=970_000.0)).is_flat()
    assert restarted.apply_risk(weights, _state(equity=1_000_000.0)).is_flat()


def test_operator_reset_clears_persisted_lock() -> None:
    store = InMemoryRiskStateStore()
    first = _overlay(store, daily_max_loss_pct=0.03)
    first.update_equity(1_000_000.0, timestamp=FIXED_TIMESTAMP)
    first.update_equity(970_000.0, timestamp=FIXED_TIMESTAMP)
    restarted = _overlay(store, daily_max_loss_pct=0.03)
    assert restarted.kill_switch_active is True

    restarted.reset_kill_switch()
    assert restarted.kill_switch_active is False
    persisted = store.load()
    assert persisted is not None
    assert persisted.kill_switch_locked is False
    assert persisted.daily_loss_locked is False
    assert persisted.kill_switch_reason == ""

    third = _overlay(store, daily_max_loss_pct=0.03)
    assert third.kill_switch_active is False
    assert third.apply_risk(make_weight_vector({"ES": 0.5}), _state()).get("ES") == pytest.approx(
        0.5
    )


def test_engage_kill_switch_latches_and_persists_reason() -> None:
    store = InMemoryRiskStateStore()
    overlay = _overlay(store)
    overlay.update_equity(1_000_000.0, timestamp=FIXED_TIMESTAMP)
    overlay.engage_kill_switch("operator emergency flatten")

    assert overlay.kill_switch_active is True
    assert overlay.kill_switch_reason == "operator emergency flatten"
    persisted = store.load()
    assert persisted is not None
    assert persisted.kill_switch_locked is True
    assert persisted.kill_switch_reason == "operator emergency flatten"
    assert overlay.apply_risk(make_weight_vector({"ES": 0.5}), _state()).is_flat()

    restarted = _overlay(store)
    assert restarted.kill_switch_active is True
    assert restarted.kill_switch_reason == "operator emergency flatten"

    with pytest.raises(ValueError, match="reason"):
        restarted.engage_kill_switch("   ")


def test_day_roll_resets_daily_nlv_but_keeps_hwm_and_kill_latch() -> None:
    store = InMemoryRiskStateStore()
    overlay = _overlay(store, daily_max_loss_pct=0.03)
    overlay.update_equity(1_000_000.0, timestamp=FIXED_TIMESTAMP)
    overlay.update_equity(1_100_000.0, timestamp=LATER_INTRADAY)
    overlay.engage_kill_switch("trailing stop halted automation")

    overlay.reset_day(1_050_000.0, timestamp=NEXT_DAY)
    # The daily loss latch is session-scoped, the explicit kill switch is not.
    assert overlay.persisted_state.daily_loss_locked is False
    persisted = store.load()
    assert persisted is not None
    assert persisted.trading_day == date(2024, 1, 3)
    assert persisted.starting_daily_nlv == pytest.approx(1_050_000.0)
    assert persisted.portfolio_high_water_mark == pytest.approx(1_100_000.0)
    assert persisted.kill_switch_locked is True
    assert persisted.kill_switch_reason == "trailing stop halted automation"
    assert persisted.daily_loss_locked is False

    # The kill-switch latch survives the day roll, exactly like rolled_to_day().
    restarted = _overlay(store, daily_max_loss_pct=0.03)
    assert restarted.kill_switch_active is True
    assert restarted.apply_risk(
        make_weight_vector({"ES": 0.5}), _state(timestamp=NEXT_DAY)
    ).is_flat()

    restarted.reset_kill_switch()
    unlocked = store.load()
    assert unlocked is not None
    assert unlocked.kill_switch_locked is False
    assert unlocked.portfolio_high_water_mark == pytest.approx(1_100_000.0)


def test_automatic_session_roll_keeps_portfolio_hwm() -> None:
    store = InMemoryRiskStateStore()
    overlay = _overlay(store, daily_max_loss_pct=0.03)
    overlay.update_equity(1_000_000.0, timestamp=FIXED_TIMESTAMP)
    overlay.update_equity(1_080_000.0, timestamp=LATER_INTRADAY)
    # First observation of the next session rolls the daily counters.
    overlay.apply_risk(
        make_weight_vector({"ES": 0.5}),
        _state(equity=1_020_000.0, timestamp=NEXT_DAY),
    )
    persisted = store.load()
    assert persisted is not None
    assert persisted.trading_day == date(2024, 1, 3)
    assert persisted.starting_daily_nlv == pytest.approx(1_020_000.0)
    assert persisted.portfolio_high_water_mark == pytest.approx(1_080_000.0)


# ---------------------------------------------------------------------------
# Instrument HWM + trailing-stop latch restoration
# ---------------------------------------------------------------------------


def test_legacy_persisted_hwm_without_direction_defaults_long() -> None:
    store = InMemoryRiskStateStore(
        PersistedRiskState(
            starting_daily_nlv=1_000_000.0,
            portfolio_high_water_mark=1_000_000.0,
            instrument_high_water_marks={"ES": 4_800.0},
            trading_day=date(2024, 1, 2),
        )
    )
    overlay = _overlay(store, trailing_stop_pct=0.10)
    assert overlay.restore_from_store() is True
    snapshot = overlay.snapshot_state()
    assert snapshot["instrument_high_water_marks"]["ES"]["direction"] == 1
    crashed = _state(prices={"ES": 4_800.0 * 0.89})
    assert overlay.apply_risk(make_weight_vector({"ES": 0.5}), crashed).get("ES") == 0.0
    persisted = store.load()
    assert persisted is not None
    assert persisted.extra["quant_overlay.hwm_direction:ES"] == 1


def test_instrument_hwm_restoration_keeps_trailing_stop_latched() -> None:
    store = InMemoryRiskStateStore()
    first = _overlay(store, trailing_stop_pct=0.10)
    first.update_equity(1_000_000.0, timestamp=FIXED_TIMESTAMP)
    first.apply_risk(make_weight_vector({"ES": 0.5}), _state(prices={"ES": 4_900.0}))
    persisted = store.load()
    assert persisted is not None
    assert persisted.instrument_high_water_marks["ES"] == pytest.approx(4_900.0)

    # Simulated restart: the long HWM at 4900 must be restored (not reset to the
    # crashed mark), so the 11% drop below the peak trips the 10% trailing stop.
    restarted = _overlay(store, trailing_stop_pct=0.10)
    crashed = _state(prices={"ES": 4_900.0 * 0.89})
    stopped = restarted.apply_risk(make_weight_vector({"ES": 0.5}), crashed)
    assert stopped["ES"] == 0.0
    assert "ES" in restarted.snapshot_state()["stopped_roots"]

    # The trailing-stop latch itself is durable: a second restart is still flat
    # even after a full recovery, until the session roll / operator reset.
    third = _overlay(store, trailing_stop_pct=0.10)
    assert third.apply_risk(make_weight_vector({"ES": 0.5}), _state(prices={"ES": 4_900.0})).get(
        "ES"
    ) == pytest.approx(0.0)
    third.reset_day(1_000_000.0, timestamp=FIXED_TIMESTAMP)
    assert third.apply_risk(make_weight_vector({"ES": 0.5}), _state(prices={"ES": 4_900.0})).get(
        "ES"
    ) == pytest.approx(0.5)

    # A fresh overlay without a store treats the crashed mark as the entry HWM.
    fresh = _overlay(None, trailing_stop_pct=0.10)
    fresh.update_equity(1_000_000.0, timestamp=FIXED_TIMESTAMP)
    assert fresh.apply_risk(make_weight_vector({"ES": 0.5}), crashed).get("ES") == pytest.approx(
        0.5
    )


def test_short_instrument_hwm_direction_restored() -> None:
    store = InMemoryRiskStateStore()
    first = _overlay(store, trailing_stop_pct=0.10)
    first.update_equity(1_000_000.0, timestamp=FIXED_TIMESTAMP)
    first.apply_risk(make_weight_vector({"ES": -0.5}), _state(prices={"ES": 4_800.0}))
    persisted = store.load()
    assert persisted is not None
    assert persisted.instrument_high_water_marks["ES"] == pytest.approx(4_800.0)
    assert persisted.extra["quant_overlay.hwm_direction:ES"] == -1

    restarted = _overlay(store, trailing_stop_pct=0.10)
    rallied = _state(prices={"ES": 4_800.0 * 1.12})
    assert restarted.apply_risk(make_weight_vector({"ES": -0.5}), rallied).get("ES") == (
        pytest.approx(0.0)
    )


# ---------------------------------------------------------------------------
# Store failures are contained
# ---------------------------------------------------------------------------


def test_failing_store_never_raises_into_the_trading_loop() -> None:
    store = _ExplodingStore()
    overlay = QuantRiskOverlay(
        config=_config(),
        clock=lambda: FIXED_TIMESTAMP,
        state_store=store,  # type: ignore[arg-type]
    )
    assert overlay.state_store_errors >= 1
    assert overlay.last_state_store_error is not None
    assert "load" in overlay.last_state_store_error

    overlay.update_equity(1_000_000.0, timestamp=FIXED_TIMESTAMP)
    assert overlay.state_store_errors >= 2
    assert "RuntimeError" in (overlay.last_state_store_error or "")
    # In-memory state remains fully functional.
    snapshot = overlay.snapshot_state()
    assert snapshot["starting_daily_equity"] == pytest.approx(1_000_000.0)
    assert overlay.restore_from_store() is False
    assert overlay.persist() is not None


def test_corrupt_file_store_is_quarantined_and_ignored(tmp_path: Path) -> None:
    path = tmp_path / "risk_state.json"
    path.write_text("{this is not valid json", encoding="utf-8")
    store = FileRiskStateStore(path)
    overlay = _overlay(store)
    assert overlay.state_store_errors == 0  # the store itself handles corruption
    assert overlay.kill_switch_active is False
    overlay.update_equity(1_000_000.0, timestamp=FIXED_TIMESTAMP)
    assert store.load() is not None  # a clean document replaced the corrupt one


def test_restore_from_store_returns_false_when_empty() -> None:
    store = InMemoryRiskStateStore()
    overlay = _overlay(store)
    assert overlay.restore_from_store() is False
    overlay.update_equity(1_000_000.0, timestamp=FIXED_TIMESTAMP)
    assert overlay.restore_from_store() is True


# ---------------------------------------------------------------------------
# Config coercion keeps the existing RiskSettings mapping
# ---------------------------------------------------------------------------


def test_config_from_risk_settings_still_maps_all_fields() -> None:
    settings = RiskSettings()
    config = QuantRiskConfig.from_risk_settings(settings)
    assert config.daily_max_loss_pct == settings.daily_max_loss_pct
    assert config.trailing_stop_pct == settings.trailing_stop_pct
    overlay = QuantRiskOverlay(config=settings, state_store=InMemoryRiskStateStore())
    assert overlay.config.drawdown_limit_pct == settings.drawdown_limit_pct
