"""Unit tests for the runner <-> API bridge (Phase 9).

Every test is offline and deterministic: the bridge is exercised with hand-built
snapshots, temp store paths and fake OMS / ``SimulatedBroker`` shapes; nothing
touches IBKR, Databento or the network.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from configs.instruments import InstrumentsConfig
from configs.settings import Settings
from src.backtest.simulated_broker import BarQuote, SimulatedBroker
from src.core.types import (
    Fill,
    OrderEvent,
    OrderSide,
    OrderStatus,
    OrderType,
    PipelineResult,
    Position,
)
from src.execution.runner import RunnerStatus
from src.execution.runner_bridge import (
    BRIDGE_FILENAME,
    BRIDGE_MODES,
    BRIDGE_SCHEMA_VERSION,
    DEFAULT_STALE_AFTER_SECONDS,
    MAX_BRIDGE_FILLS,
    MAX_BRIDGE_ORDERS,
    BridgeFill,
    BridgePosition,
    BridgeSchemaError,
    BridgeWorkingOrder,
    RunnerBridgeSnapshot,
    RunnerBridgeStore,
    snapshot_from_runner,
)

from tests.fixtures.market import (
    FIXED_TIMESTAMP,
    make_contract_targets,
    make_weight_vector,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


def make_position(
    instrument_id: str = "ES",
    quantity: int = 2,
    *,
    average_price: float = 4_800.0,
    market_price: float = 4_820.0,
    multiplier: float = 50.0,
) -> Position:
    return Position(
        instrument_id=instrument_id,
        quantity=quantity,
        average_price=average_price,
        market_price=market_price,
        multiplier=multiplier,
    )


def make_order(
    order_id: str = "o-1",
    *,
    instrument_id: str = "ES",
    side: OrderSide = OrderSide.BUY,
    quantity: int = 2,
    order_type: OrderType = OrderType.MARKET,
    limit_price: float | None = None,
    timestamp: datetime = FIXED_TIMESTAMP,
) -> OrderEvent:
    return OrderEvent(
        order_id=order_id,
        timestamp=timestamp,
        instrument_id=instrument_id,
        side=side,
        quantity=quantity,
        order_type=order_type,
        limit_price=limit_price,
    )


def make_fill(
    order_id: str = "o-1",
    *,
    instrument_id: str = "ES",
    side: OrderSide = OrderSide.BUY,
    quantity: int = 1,
    price: float = 4_800.0,
    timestamp: datetime = FIXED_TIMESTAMP,
    commission: float = 1.0,
    slippage_ticks: float = 0.5,
) -> Fill:
    return Fill(
        order_id=order_id,
        instrument_id=instrument_id,
        side=side,
        quantity=quantity,
        price=price,
        timestamp=timestamp,
        commission=commission,
        slippage_ticks=slippage_ticks,
    )


def make_pipeline_result(
    *,
    timestamp: datetime = FIXED_TIMESTAMP,
    diagnostics: dict[str, object] | None = None,
) -> PipelineResult:
    weights = make_weight_vector({"ES": 0.5, "NQ": 0.25})
    return PipelineResult(
        timestamp=timestamp,
        universe=("ES", "NQ"),
        base_weights=weights,
        timing_weights=weights,
        risk_weights=weights,
        final_weights=weights,
        targets=make_contract_targets({"ES": 1}),
        diagnostics=diagnostics if diagnostics is not None else {},
        elapsed_ms=3.0,
    )


def make_snapshot(
    sequence: int = 1,
    *,
    updated_at: datetime = FIXED_TIMESTAMP,
    stale_after_seconds: float = DEFAULT_STALE_AFTER_SECONDS,
    mode: str = "offline",
    runner_id: str = "runner-existing",
) -> RunnerBridgeSnapshot:
    return RunnerBridgeSnapshot(
        schema_version=BRIDGE_SCHEMA_VERSION,
        runner_id=runner_id,
        mode=mode,  # type: ignore[arg-type]
        sequence=sequence,
        updated_at=updated_at,
        stale_after_seconds=stale_after_seconds,
        status={"hydrated": True, "states_processed": sequence, "halted": False},
    )


class FakeRunner:
    """Duck-typed runner providing the caches ``snapshot_from_runner`` reads."""

    def __init__(
        self,
        *,
        settings: Settings,
        status: RunnerStatus | None = None,
        runner_id: str = "runner-test",
        clock: Any = None,
        last_positions: dict[str, Position] | None = None,
        last_account: dict[str, float] | None = None,
        last_pipeline_result: PipelineResult | None = None,
        last_bridge_snapshot: RunnerBridgeSnapshot | None = None,
        mode: str | None = None,
    ) -> None:
        self.settings = settings
        self.status = status if status is not None else RunnerStatus()
        self.runner_id = runner_id
        self.clock = clock if clock is not None else (lambda: FIXED_TIMESTAMP)
        self.last_positions = last_positions
        self.last_account = last_account
        self.last_pipeline_result = last_pipeline_result
        self.last_bridge_snapshot = last_bridge_snapshot
        if mode is not None:
            self.mode = mode


class FakeOMS:
    """Minimal Phase-5 OMS surface (``orders`` / ``open_order_ids`` / ``all_fills``)."""

    def __init__(
        self,
        *,
        orders: dict[str, Any] | None = None,
        open_order_ids: tuple[str, ...] = (),
        fills: list[Fill] | None = None,
    ) -> None:
        self.orders = dict(orders or {})
        self.open_order_ids = open_order_ids
        self._fills = list(fills or [])
        self.is_locked = False

    def all_fills(self) -> list[Fill]:
        return list(self._fills)


def managed(
    event: OrderEvent,
    *,
    status: OrderStatus = OrderStatus.SUBMITTED,
    filled_quantity: int = 0,
    receipt: Any = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        event=event,
        status=status,
        filled_quantity=filled_quantity,
        receipt=receipt,
        is_terminal=status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED),
    )


# ---------------------------------------------------------------------------
# Snapshot contract: round-trip + validation
# ---------------------------------------------------------------------------


def _full_snapshot() -> RunnerBridgeSnapshot:
    return RunnerBridgeSnapshot(
        schema_version=BRIDGE_SCHEMA_VERSION,
        runner_id="runner-roundtrip",
        mode="paper",
        sequence=3,
        updated_at=FIXED_TIMESTAMP,
        stale_after_seconds=12.5,
        status={"hydrated": True, "states_processed": 3, "halted": False, "roots": "ES,NQ"},
        account={"net_liquidation": 1_000_000.0, "margin_utilization": 0.12},
        pipeline={"universe": "ES,NQ", "gross_leverage": 0.75, "targets": 1},
        positions=(
            BridgePosition(
                instrument_id="NQ",
                contract_symbol="NQ_TEST",
                quantity=-1,
                average_price=16_800.0,
                market_price=16_900.0,
                multiplier=20.0,
                unrealized_pnl=-2_000.0,
                initial_margin=21_000.0,
            ),
            BridgePosition(
                instrument_id="ES",
                contract_symbol=None,
                quantity=2,
                average_price=4_800.0,
                market_price=4_820.0,
                multiplier=50.0,
                unrealized_pnl=2_000.0,
                initial_margin=13_000.0,
            ),
        ),
        working_orders=(
            BridgeWorkingOrder(
                order_id="wo-1",
                instrument_id="ES",
                contract_symbol="ES_TEST",
                side="BUY",
                quantity=2,
                filled_quantity=1,
                remaining_quantity=1,
                order_type="LIMIT",
                limit_price=4_790.0,
                status="PARTIALLY_FILLED",
                updated_at=FIXED_TIMESTAMP,
            ),
        ),
        fills=(
            BridgeFill(
                order_id="f-1",
                instrument_id="ES",
                side="SELL",
                quantity=3,
                price=4_810.0,
                commission=1.5,
                slippage_ticks=0.25,
                timestamp=FIXED_TIMESTAMP,
            ),
        ),
    )


def test_snapshot_round_trips_through_json() -> None:
    snapshot = _full_snapshot()
    payload = snapshot.to_dict()
    serialised = json.dumps(payload)  # must be JSON-serialisable
    restored = RunnerBridgeSnapshot.from_dict(json.loads(serialised))
    assert restored.to_dict() == payload
    assert restored.schema_version == BRIDGE_SCHEMA_VERSION
    assert restored.mode == "paper"
    assert restored.sequence == 3
    assert [position.instrument_id for position in restored.positions] == ["ES", "NQ"]
    assert restored.working_orders[0].remaining_quantity == 1
    assert restored.fills[0].side == "SELL"
    assert isinstance(restored.updated_at, datetime)
    assert restored.updated_at.tzinfo is not None


def test_snapshot_from_dict_rejects_newer_schema_and_bad_fields() -> None:
    payload = _full_snapshot().to_dict()
    newer = {**payload, "schema_version": BRIDGE_SCHEMA_VERSION + 1}
    with pytest.raises(BridgeSchemaError, match="newer than supported"):
        RunnerBridgeSnapshot.from_dict(newer)
    bad_mode = {**payload, "mode": "simulated"}
    with pytest.raises(BridgeSchemaError, match="mode"):
        RunnerBridgeSnapshot.from_dict(bad_mode)
    missing_id = {k: v for k, v in payload.items() if k != "runner_id"}
    with pytest.raises(BridgeSchemaError, match="runner_id"):
        RunnerBridgeSnapshot.from_dict(missing_id)
    bad_position = {
        **payload,
        "positions": [{**payload["positions"][0], "quantity": "two"}],
    }
    with pytest.raises(BridgeSchemaError):
        RunnerBridgeSnapshot.from_dict(bad_position)


def test_snapshot_from_dict_ignores_forward_compatible_unknown_keys() -> None:
    payload = {**_full_snapshot().to_dict(), "future_field": {"anything": 1}}
    restored = RunnerBridgeSnapshot.from_dict(payload)
    assert restored.sequence == 3


def test_snapshot_is_stale_boundaries_are_strict() -> None:
    snapshot = make_snapshot(updated_at=FIXED_TIMESTAMP, stale_after_seconds=10.0)
    assert snapshot.is_stale(now=FIXED_TIMESTAMP + timedelta(seconds=9.999)) is False
    assert snapshot.is_stale(now=FIXED_TIMESTAMP + timedelta(seconds=10.0)) is False
    assert snapshot.is_stale(now=FIXED_TIMESTAMP + timedelta(seconds=10.001)) is True
    # A future timestamp (clock skew) is not stale.
    assert snapshot.is_stale(now=FIXED_TIMESTAMP - timedelta(seconds=5)) is False
    # Naive "now" is rejected: a futures timestamp must never be guessed.
    with pytest.raises(ValueError):
        snapshot.is_stale(now=datetime(2024, 1, 2, 21, 0))


def test_snapshot_validation_rejects_bad_modes_versions_and_caps() -> None:
    good = make_snapshot()
    with pytest.raises(BridgeSchemaError, match="mode"):
        RunnerBridgeSnapshot(
            schema_version=BRIDGE_SCHEMA_VERSION,
            runner_id="r",
            mode="simulated",  # type: ignore[arg-type]
            sequence=1,
            updated_at=FIXED_TIMESTAMP,
            stale_after_seconds=15.0,
        )
    with pytest.raises(BridgeSchemaError, match="schema_version"):
        RunnerBridgeSnapshot(
            schema_version=2,
            runner_id="r",
            mode="offline",
            sequence=1,
            updated_at=FIXED_TIMESTAMP,
            stale_after_seconds=15.0,
        )
    with pytest.raises(BridgeSchemaError, match="sequence"):
        RunnerBridgeSnapshot(
            schema_version=BRIDGE_SCHEMA_VERSION,
            runner_id="r",
            mode="offline",
            sequence=-1,
            updated_at=FIXED_TIMESTAMP,
            stale_after_seconds=15.0,
        )
    too_many_orders = tuple(
        BridgeWorkingOrder(
            order_id=f"wo-{index}",
            instrument_id="ES",
            contract_symbol=None,
            side="BUY",
            quantity=1,
            filled_quantity=0,
            remaining_quantity=1,
            order_type="LIMIT",
            limit_price=4_800.0,
            status="SUBMITTED",
            updated_at=FIXED_TIMESTAMP,
        )
        for index in range(MAX_BRIDGE_ORDERS + 1)
    )
    with pytest.raises(BridgeSchemaError, match="cap"):
        RunnerBridgeSnapshot(
            schema_version=BRIDGE_SCHEMA_VERSION,
            runner_id="r",
            mode="offline",
            sequence=1,
            updated_at=FIXED_TIMESTAMP,
            stale_after_seconds=15.0,
            working_orders=too_many_orders,
        )
    assert good.stale_after_seconds == DEFAULT_STALE_AFTER_SECONDS


def test_bridge_dataclass_validation_and_normalisation() -> None:
    with pytest.raises(BridgeSchemaError):
        BridgePosition(
            instrument_id="",
            contract_symbol=None,
            quantity=1,
            average_price=1.0,
            market_price=1.0,
            multiplier=1.0,
            unrealized_pnl=0.0,
            initial_margin=0.0,
        )
    with pytest.raises(BridgeSchemaError, match="side"):
        BridgeWorkingOrder(
            order_id="o",
            instrument_id="ES",
            contract_symbol=None,
            side="HOLD",
            quantity=1,
            filled_quantity=0,
            remaining_quantity=1,
            order_type="LIMIT",
            limit_price=1.0,
            status="SUBMITTED",
            updated_at=FIXED_TIMESTAMP,
        )
    with pytest.raises(BridgeSchemaError, match="filled_quantity"):
        BridgeWorkingOrder(
            order_id="o",
            instrument_id="ES",
            contract_symbol=None,
            side="BUY",
            quantity=1,
            filled_quantity=3,
            remaining_quantity=1,
            order_type="LIMIT",
            limit_price=1.0,
            status="SUBMITTED",
            updated_at=FIXED_TIMESTAMP,
        )
    # An inconsistent remaining quantity is repaired from quantity - filled.
    repaired = BridgeWorkingOrder(
        order_id="o",
        instrument_id="ES",
        contract_symbol=None,
        side="SELL",
        quantity=10,
        filled_quantity=8,
        remaining_quantity=5,
        order_type="LIMIT",
        limit_price=1.0,
        status="PARTIALLY_FILLED",
        updated_at=FIXED_TIMESTAMP,
    )
    assert repaired.remaining_quantity == 2
    with pytest.raises(BridgeSchemaError, match="price"):
        BridgeFill(
            order_id="f",
            instrument_id="ES",
            side="BUY",
            quantity=1,
            price=0.0,
            commission=0.0,
            slippage_ticks=0.0,
            timestamp=FIXED_TIMESTAMP,
        )


# ---------------------------------------------------------------------------
# Durable store
# ---------------------------------------------------------------------------


def test_store_default_path_is_data_dir_bridge(settings: Settings) -> None:
    store = RunnerBridgeStore(settings)
    expected = settings.data_dir / "bridge" / BRIDGE_FILENAME
    assert store.path == expected
    assert RunnerBridgeStore.default_path(settings) == expected
    assert store.stale_after_seconds == DEFAULT_STALE_AFTER_SECONDS


def test_store_saves_atomically_and_loads(settings: Settings) -> None:
    store = RunnerBridgeStore(settings)
    snapshot = make_snapshot(sequence=4)
    written = store.save(snapshot)
    assert written == store.path
    assert store.path.is_file()
    assert json.loads(store.path.read_text(encoding="utf-8"))["schema_version"] == (
        BRIDGE_SCHEMA_VERSION
    )
    # No temporary files left behind.
    leftovers = [
        entry.name for entry in store.path.parent.iterdir() if entry.name != BRIDGE_FILENAME
    ]
    assert leftovers == []
    loaded = store.load()
    assert loaded is not None
    assert loaded.to_dict() == snapshot.to_dict()


def test_store_accepts_explicit_directory_or_file_paths(settings: Settings, tmp_path: Path) -> None:
    directory_store = RunnerBridgeStore(settings, path=tmp_path / "bridge-dir")
    assert directory_store.path == tmp_path / "bridge-dir" / BRIDGE_FILENAME
    directory_store.save(make_snapshot())
    assert directory_store.path.is_file()

    file_store = RunnerBridgeStore(settings, path=tmp_path / "custom.json")
    assert file_store.path == tmp_path / "custom.json"
    file_store.save(make_snapshot(sequence=2))
    assert file_store.path.is_file()


def test_store_corrupt_payload_returns_none_without_clobbering(
    settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    store = RunnerBridgeStore(settings)
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text("{not valid json", encoding="utf-8")
    with caplog.at_level("ERROR"):
        assert store.load() is None
    assert any("corrupt" in record.getMessage() for record in caplog.records)
    # The corrupt file is reported, not silently deleted or rewritten on load.
    assert store.path.read_text(encoding="utf-8") == "{not valid json"
    assert store.load_fresh() == (None, False)
    # A subsequent valid save atomically replaces it.
    store.save(make_snapshot(sequence=9))
    loaded = store.load()
    assert loaded is not None and loaded.sequence == 9


def test_store_load_fresh_reports_staleness(settings: Settings) -> None:
    store = RunnerBridgeStore(settings, stale_after_seconds=10.0)
    stale = make_snapshot(
        updated_at=datetime.now(UTC) - timedelta(seconds=60), stale_after_seconds=10.0
    )
    store.save(stale)
    snapshot, is_stale = store.load_fresh()
    assert snapshot is not None and is_stale is True

    fresh = make_snapshot(updated_at=datetime.now(UTC), stale_after_seconds=10.0)
    store.save(fresh)
    snapshot, is_stale = store.load_fresh()
    assert snapshot is not None and is_stale is False


def test_store_clear_is_idempotent(settings: Settings) -> None:
    store = RunnerBridgeStore(settings)
    store.save(make_snapshot())
    assert store.path.is_file()
    store.clear()
    assert not store.path.is_file()
    store.clear()  # no-op


def test_store_no_clobber_between_writes(settings: Settings) -> None:
    store = RunnerBridgeStore(settings)
    first = make_snapshot(sequence=1)
    second = make_snapshot(sequence=2)
    store.save(first)
    store.save(second)
    loaded = store.load()
    assert loaded is not None
    assert loaded.sequence == 2
    assert json.loads(store.path.read_text(encoding="utf-8"))["sequence"] == 2


def test_store_validation(settings: Settings) -> None:
    with pytest.raises(ValueError):
        RunnerBridgeStore(settings, stale_after_seconds=0.0)
    with pytest.raises(ValueError):
        RunnerBridgeStore(settings, stale_after_seconds=float("nan"))
    with pytest.raises(TypeError):
        RunnerBridgeStore("not-settings")  # type: ignore[arg-type]
    store = RunnerBridgeStore(settings)
    with pytest.raises(TypeError):
        store.save("not-a-snapshot")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# snapshot_from_runner: OMS shape
# ---------------------------------------------------------------------------


def test_snapshot_from_runner_normalises_oms_shape(settings: Settings) -> None:
    orders = {
        "o-1": managed(
            make_order("o-1", quantity=2, order_type=OrderType.LIMIT, limit_price=4_790.0)
        ),
        "o-2": managed(
            make_order("o-2", instrument_id="NQ", side=OrderSide.SELL, quantity=3),
            status=OrderStatus.PARTIALLY_FILLED,
            filled_quantity=1,
            receipt=SimpleNamespace(timestamp=FIXED_TIMESTAMP),
        ),
        "o-3": managed(make_order("o-3", quantity=1), status=OrderStatus.FILLED, filled_quantity=1),
    }
    broker = FakeOMS(
        orders=orders,
        open_order_ids=("o-1", "o-2"),
        fills=[make_fill("f-1"), make_fill("f-2", side=OrderSide.SELL, quantity=2)],
    )
    runner = FakeRunner(
        settings=settings,
        status=RunnerStatus(
            hydrated=True,
            states_processed=3,
            orders_submitted=2,
            orders_rejected=1,
            roots=("ES", "NQ"),
        ),
        last_positions={"ES": make_position("ES", 2)},
        last_account={"net_liquidation": 1_000_000.0, "margin_utilization": 0.12},
        last_pipeline_result=make_pipeline_result(diagnostics={"tracking_error_pct": 1.5}),
    )
    snapshot = snapshot_from_runner(runner, broker=broker, mode="paper", stale_after_seconds=7.5)
    assert snapshot.schema_version == BRIDGE_SCHEMA_VERSION
    assert snapshot.mode == "paper"
    assert snapshot.runner_id == "runner-test"
    assert snapshot.sequence == 1
    assert snapshot.updated_at == FIXED_TIMESTAMP
    assert snapshot.stale_after_seconds == 7.5
    assert snapshot.status["states_processed"] == 3
    assert snapshot.status["orders_rejected"] == 1
    assert snapshot.status["roots"] == "ES,NQ"
    assert snapshot.account == {
        "net_liquidation": 1_000_000.0,
        "margin_utilization": 0.12,
    }
    assert snapshot.pipeline is not None
    assert snapshot.pipeline["universe"] == "ES,NQ"
    assert snapshot.pipeline["gross_leverage"] == pytest.approx(0.75)
    assert snapshot.pipeline["targets"] == 1
    assert snapshot.pipeline["tracking_error_pct"] == pytest.approx(1.5)
    assert snapshot.pipeline["target_book"] == "ES:1"
    assert [position.instrument_id for position in snapshot.positions] == ["ES"]
    position = snapshot.positions[0]
    assert position.quantity == 2
    assert position.unrealized_pnl == pytest.approx(2_000.0)
    assert position.initial_margin == pytest.approx(0.0)  # Position has no initial margin field
    working = {order.order_id: order for order in snapshot.working_orders}
    assert set(working) == {"o-1", "o-2"}
    assert working["o-1"].side == "BUY"
    assert working["o-1"].quantity == 2
    assert working["o-1"].remaining_quantity == 2
    assert working["o-1"].order_type == "LMT"
    assert working["o-1"].limit_price == pytest.approx(4_790.0)
    assert working["o-2"].side == "SELL"
    assert working["o-2"].filled_quantity == 1
    assert working["o-2"].remaining_quantity == 2
    assert working["o-2"].status == "PARTIALLY_FILLED"
    assert [fill.order_id for fill in snapshot.fills] == ["f-1", "f-2"]
    assert snapshot.fills[1].side == "SELL"
    assert snapshot.fills[0].commission == pytest.approx(1.0)
    assert snapshot.fills[0].slippage_ticks == pytest.approx(0.5)


def test_snapshot_from_runner_uses_runner_caches_when_available(settings: Settings) -> None:
    runner = FakeRunner(
        settings=settings,
        last_account={"net_liquidation": 500_000.0, "cash_balance": 250_000.0},
        last_positions={"NQ": make_position("NQ", -2, average_price=16_800.0)},
    )
    snapshot = snapshot_from_runner(runner, broker=object())
    assert snapshot.account is not None
    assert snapshot.account["net_liquidation"] == pytest.approx(500_000.0)
    assert snapshot.positions[0].instrument_id == "NQ"
    assert snapshot.positions[0].quantity == -2
    assert snapshot.pipeline is None
    assert snapshot.fills == ()
    assert snapshot.working_orders == ()
    assert snapshot.status["hydrated"] is False  # default RunnerStatus


def test_snapshot_from_runner_pipeline_without_tracking_error(settings: Settings) -> None:
    runner = FakeRunner(
        settings=settings,
        last_pipeline_result=make_pipeline_result(diagnostics={"other": "value"}),
    )
    snapshot = snapshot_from_runner(runner, broker=object(), mode="live")
    assert snapshot.pipeline is not None
    assert "tracking_error_pct" not in snapshot.pipeline
    assert snapshot.pipeline["diagnostics_keys"] == "other"


def test_snapshot_from_runner_sequence_increments(settings: Settings) -> None:
    runner = FakeRunner(settings=settings)
    first = snapshot_from_runner(runner, broker=object())
    assert first.sequence == 1
    runner.last_bridge_snapshot = first
    second = snapshot_from_runner(runner, broker=object())
    assert second.sequence == 2
    runner.last_bridge_snapshot = second
    assert snapshot_from_runner(runner, broker=object()).sequence == 3


def test_snapshot_from_runner_caps_orders_and_fills_to_most_recent(
    settings: Settings,
) -> None:
    start = datetime(2024, 1, 2, 21, 0, tzinfo=UTC)
    orders = {
        f"o-{index}": managed(
            make_order(f"o-{index}", timestamp=start + timedelta(seconds=index)),
            status=OrderStatus.SUBMITTED,
        )
        for index in range(MAX_BRIDGE_ORDERS + 20)
    }
    fills = [
        make_fill(
            f"f-{index}",
            timestamp=start + timedelta(seconds=index),
            price=4_800.0 + index,
        )
        for index in range(MAX_BRIDGE_FILLS + 20)
    ]
    broker = FakeOMS(
        orders=orders,
        open_order_ids=tuple(orders),
        fills=fills,
    )
    runner = FakeRunner(settings=settings)
    snapshot = snapshot_from_runner(runner, broker=broker)
    assert len(snapshot.working_orders) == MAX_BRIDGE_ORDERS
    assert len(snapshot.fills) == MAX_BRIDGE_FILLS
    # The most recent entries are kept (not the oldest).
    assert snapshot.working_orders[-1].order_id == f"o-{MAX_BRIDGE_ORDERS + 19}"
    assert snapshot.working_orders[0].order_id == "o-20"
    assert snapshot.fills[-1].order_id == f"f-{MAX_BRIDGE_FILLS + 19}"
    assert snapshot.fills[0].order_id == "f-20"


def test_snapshot_from_runner_validation(settings: Settings) -> None:
    runner = FakeRunner(settings=settings)
    with pytest.raises(BridgeSchemaError, match="mode"):
        snapshot_from_runner(runner, broker=object(), mode="simulated")
    with pytest.raises(ValueError):
        snapshot_from_runner(runner, broker=object(), stale_after_seconds=0.0)
    with pytest.raises(ValueError):
        snapshot_from_runner(runner, broker=object(), stale_after_seconds=float("inf"))


# ---------------------------------------------------------------------------
# snapshot_from_runner: SimulatedBroker shape
# ---------------------------------------------------------------------------


def test_snapshot_from_runner_normalises_simulated_broker(
    settings: Settings, instrument_master: InstrumentsConfig
) -> None:
    simulated = SimulatedBroker(instrument_master, initial_cash=1_000_000.0)
    timestamp = datetime(2024, 1, 2, 21, 0, tzinfo=UTC)
    quotes = {
        "ES": BarQuote(root="ES", open=4_800.0, high=4_830.0, low=4_790.0, close=4_820.0),
        "NQ": BarQuote(root="NQ", open=16_800.0, high=16_900.0, low=16_700.0, close=16_850.0),
    }
    simulated.set_market(timestamp, quotes)
    simulated.place_order(make_order("filled-1", quantity=2, timestamp=timestamp))
    runner = FakeRunner(
        settings=settings,
        status=RunnerStatus(hydrated=True, states_processed=1, orders_submitted=1),
        clock=lambda: timestamp,
    )
    snapshot = snapshot_from_runner(runner, broker=simulated, mode="offline")
    assert snapshot.account is not None
    assert snapshot.account["net_liquidation"] > 0.0
    assert snapshot.positions and snapshot.positions[0].instrument_id == "ES"
    assert snapshot.positions[0].quantity == 2
    assert snapshot.positions[0].market_price == pytest.approx(4_820.0)
    assert snapshot.fills and snapshot.fills[0].order_id == "filled-1"
    assert snapshot.fills[0].commission >= 0.0
    assert snapshot.working_orders == ()
    # Closing the position removes it from the published open book.
    simulated.place_order(
        make_order(
            "close-1",
            side=OrderSide.SELL,
            quantity=2,
            timestamp=timestamp,
        )
    )
    flattened = snapshot_from_runner(runner, broker=simulated, mode="offline")
    assert all(position.quantity != 0 for position in flattened.positions)
    assert flattened.positions == ()


def test_snapshot_from_runner_normalises_simulated_working_orders(
    settings: Settings, instrument_master: InstrumentsConfig
) -> None:
    # A participation cap of 0.1% against a thin bar leaves the order working.
    simulated = SimulatedBroker(
        instrument_master, initial_cash=1_000_000.0, max_participation=0.001
    )
    timestamp = datetime(2024, 1, 2, 21, 0, tzinfo=UTC)
    simulated.set_market(
        timestamp,
        {
            "ES": BarQuote(
                root="ES", open=4_800.0, high=4_830.0, low=4_790.0, close=4_820.0, volume=10.0
            )
        },
    )
    simulated.place_order(
        make_order("working-1", side=OrderSide.BUY, quantity=4, timestamp=timestamp)
    )
    assert len(simulated.working_orders) == 1
    runner = FakeRunner(settings=settings, clock=lambda: timestamp)
    snapshot = snapshot_from_runner(runner, broker=simulated, mode="offline")
    assert len(snapshot.working_orders) == 1
    working = snapshot.working_orders[0]
    assert working.order_id == "working-1"
    assert working.instrument_id == "ES"
    assert working.side == "BUY"
    assert working.quantity == 4
    assert working.remaining_quantity == 4
    assert working.status
    assert working.updated_at == timestamp


def test_snapshot_from_runner_defaults_mode_from_settings_environment(
    settings: Settings,
) -> None:
    runner = FakeRunner(settings=settings)  # backtest environment, no runner.mode
    assert snapshot_from_runner(runner, broker=object()).mode == "offline"

    live_settings = settings.model_copy(update={"environment": "paper", "ibkr_paper": True})
    paper_runner = FakeRunner(settings=live_settings)
    assert snapshot_from_runner(paper_runner, broker=object()).mode == "paper"


def test_snapshot_from_runner_accepts_runner_mode(settings: Settings) -> None:
    runner = FakeRunner(settings=settings, mode="live")
    snapshot = snapshot_from_runner(runner, broker=object())
    assert snapshot.mode == "live"


# ---------------------------------------------------------------------------
# Settings integration
# ---------------------------------------------------------------------------


def test_runner_bridge_dir_property_is_used_for_default_path(settings: Settings) -> None:
    assert settings.runner_bridge_dir == settings.data_dir / "bridge"
    store = RunnerBridgeStore(settings)
    store.save(make_snapshot())
    assert store.path == settings.runner_bridge_dir / BRIDGE_FILENAME
    assert store.path.is_file()
    assert BRIDGE_MODES == ("offline", "paper", "live")
    assert not (math.isnan(store.stale_after_seconds) or store.stale_after_seconds <= 0.0)


def test_runner_bridge_settings_flag_defaults_on(settings: Settings) -> None:
    assert settings.runner_bridge_enabled is True


# ---------------------------------------------------------------------------
# Package-level lazy exports (execution-eng consumes these from the API)
# ---------------------------------------------------------------------------


def test_bridge_exports_resolve_from_the_execution_package() -> None:
    import src.execution as package

    assert package.RunnerBridgeStore is RunnerBridgeStore
    assert package.RunnerBridgeSnapshot is RunnerBridgeSnapshot
    assert package.snapshot_from_runner is snapshot_from_runner
    assert package.BRIDGE_SCHEMA_VERSION == BRIDGE_SCHEMA_VERSION
    from src.execution import OfflineRunnerSession

    assert OfflineRunnerSession.mode == "offline"
    assert "RunnerBridgeStore" in dir(package)
    assert "OfflineRunnerSession" in dir(package)
