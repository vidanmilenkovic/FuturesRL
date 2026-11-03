"""Phase-11A runner --rebalance-once + /api/runner rebalance preview/execute tests.

Fully hermetic: deterministic offline market/catalog, fake execution adapters and a
temporary data/config root. No IBKR/TWS or market-data stream is touched.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from configs.settings import Settings
from fastapi.testclient import TestClient
from src.api.main import create_app
from src.api.phase11_schemas import RebalanceExecuteRequest
from src.api.rebalance_service import RebalanceError, RebalanceService
from src.api.schemas import AccountSnapshotResponse
from src.core.types import (
    InstrumentId,
    OrderEvent,
    OrderReceipt,
    OrderStatus,
    Position,
)
from src.execution.runner import OfflineRunnerSession, _build_parser

pytestmark = pytest.mark.integration

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
FIXED_TIMESTAMP = datetime(2026, 11, 3, 18, 0, tzinfo=UTC)
TELEMETRY_INTERVAL = 0.02


def _settings(tmp_path: Path) -> Settings:
    configs_dir = tmp_path / "configs"
    strategies_dir = configs_dir / "strategies"
    strategies_dir.mkdir(parents=True, exist_ok=True)
    for source in sorted((REPO_ROOT / "configs" / "strategies").glob("*.yaml")):
        shutil.copyfile(source, strategies_dir / source.name)
    return Settings(
        environment="backtest",
        configs_dir=configs_dir,
        data_dir=tmp_path / "data",
        parquet_dir=tmp_path / "data" / "parquet",
        duckdb_path=tmp_path / "data" / "api.duckdb",
        report_dir=tmp_path / "artifacts",
        instruments_path=REPO_ROOT / "configs" / "instruments.yaml",
        telemetry_interval_seconds=TELEMETRY_INTERVAL,
        api_cors_origins=("http://localhost:3080",),
    )


def _account(nlv: float = 1_000_000.0, initial_margin: float = 0.0) -> AccountSnapshotResponse:
    return AccountSnapshotResponse(
        timestamp=FIXED_TIMESTAMP,
        net_liquidation=nlv,
        cash_balance=nlv,
        buying_power=nlv,
        initial_margin_used=initial_margin,
        maintenance_margin_used=initial_margin * 0.9,
        excess_liquidity=nlv - initial_margin,
        margin_utilization=initial_margin / nlv if nlv else 0.0,
        day_pnl=0.0,
        day_pnl_pct=0.0,
        gross_leverage=0.0,
        net_exposure=0.0,
        source="simulated",
    )


class FakeExecution:
    """Minimal execution adapter surface used by RebalanceService."""

    def __init__(
        self,
        *,
        ibkr: bool = False,
        connected: bool = False,
        positions: dict[InstrumentId, Position] | None = None,
        locked: bool = False,
    ) -> None:
        self._ibkr = ibkr
        self._connected = connected
        self._positions = positions or {}
        self._locked = locked
        self.submitted: list[OrderEvent] = []
        self.submit_calls = 0
        self.translator = type("_Translator", (), {"max_margin_utilization": 0.60})()

    @property
    def ibkr_enabled(self) -> bool:
        return self._ibkr

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def is_locked(self) -> bool:
        return self._locked

    async def account(self) -> AccountSnapshotResponse:
        return _account(initial_margin=1_500.0)

    async def positions(self) -> dict[InstrumentId, Position]:
        return dict(self._positions)

    async def submit_orders(self, orders: Sequence[OrderEvent]) -> list[OrderReceipt]:
        self.submit_calls += 1
        receipts: list[OrderReceipt] = []
        for order in orders:
            self.submitted.append(order)
            receipts.append(
                OrderReceipt(
                    order_id=order.order_id,
                    status=OrderStatus.FILLED,
                    filled_quantity=order.quantity,
                    average_price=order.limit_price or 100.0,
                    message="filled",
                    timestamp=FIXED_TIMESTAMP,
                )
            )
        return receipts


def _app(settings: Settings) -> TestClient:
    return TestClient(create_app(settings, clock=lambda: FIXED_TIMESTAMP))


@pytest.fixture()
def client(tmp_path: Path) -> Iterator[TestClient]:
    with _app(_settings(tmp_path)) as test_client:
        yield test_client


def test_rebalance_preview_offline_contract(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with _app(settings) as client:
        container = client.app.state.container
        service = RebalanceService(
            settings=settings,
            market=container.market,
            catalog=container.strategies,
            execution=FakeExecution(),
            clock=lambda: FIXED_TIMESTAMP,
        )
        client.app.state.rebalance_service = service
        response = client.post(
            "/api/runner/rebalance-preview",
            json={"strategy_id": "tsmom_vol_target", "dry_run": True},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["strategy_id"] == "tsmom_vol_target"
        assert body["mode"] == "offline"
        assert body["account_nlv"] == pytest.approx(1_000_000.0)
        assert body["account_currency"] == settings.account_currency
        assert body["margin_utilization_cap"] == pytest.approx(0.60)
        assert body["projected_margin_utilization"] >= 0.0
        assert body["tracking_error_pct"] >= 0.0
        assert isinstance(body["current_positions"], list)
        assert isinstance(body["target_contracts"], list)
        assert isinstance(body["proposed_orders"], list)
        for order in body["proposed_orders"]:
            assert order["side"] in {"BUY", "SELL"}
            assert order["quantity"] > 0
            assert "pre_trade_gate_passed" in order
        # Preview must never submit.
        assert service._execution.submit_calls == 0  # type: ignore[attr-defined]


def test_rebalance_preview_tolerates_masterless_market_adapter(tmp_path: Path) -> None:
    """Live container market services may expose only latest_state(); preview must load the master."""
    settings = _settings(tmp_path)
    with _app(settings) as client:
        container = client.app.state.container
        inner = container.market

        class _MasterlessMarket:
            def latest_state(self):
                return inner.latest_state()

        service = RebalanceService(
            settings=settings,
            market=_MasterlessMarket(),  # type: ignore[arg-type]
            catalog=container.strategies,
            execution=FakeExecution(),
            clock=lambda: FIXED_TIMESTAMP,
        )
        client.app.state.rebalance_service = service
        response = client.post(
            "/api/runner/rebalance-preview",
            json={"strategy_id": "tsmom_vol_target", "roots": ["ES", "NQ"], "dry_run": True},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["strategy_id"] == "tsmom_vol_target"
        assert body["mode"] == "offline"
        assert isinstance(body["proposed_orders"], list)


def test_rebalance_preview_unknown_strategy_is_404(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with _app(settings) as client:
        container = client.app.state.container
        client.app.state.rebalance_service = RebalanceService(
            settings=settings,
            market=container.market,
            catalog=container.strategies,
            execution=FakeExecution(),
            clock=lambda: FIXED_TIMESTAMP,
        )
        response = client.post(
            "/api/runner/rebalance-preview", json={"strategy_id": "does_not_exist"}
        )
        assert response.status_code == 404


def test_rebalance_execute_offline_submits_and_publishes_bridge(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    execution = FakeExecution(
        positions={
            "ES": Position(
                instrument_id="ES",
                quantity=2,
                average_price=5_000.0,
                market_price=5_000.0,
                multiplier=50.0,
                initial_margin=12_000.0,
                maintenance_margin=10_000.0,
            )
        }
    )
    with _app(settings) as client:
        container = client.app.state.container
        service = RebalanceService(
            settings=settings,
            market=container.market,
            catalog=container.strategies,
            execution=execution,
            clock=lambda: FIXED_TIMESTAMP,
        )
        client.app.state.rebalance_service = service
        first = client.post(
            "/api/runner/rebalance-execute",
            json={"strategy_id": "tsmom_vol_target", "confirm": True},
        )
        assert first.status_code == 200, first.text
        body = first.json()
        assert body["mode"] == "offline"
        assert body["status"] in {"submitted", "no-op"}
        assert body["account_nlv"] == pytest.approx(1_000_000.0)
        assert body["submitted_order_ids"]
        assert body["fills"]
        assert "updated_positions" in body
        bridge_path = settings.runner_bridge_dir / "runner_state.json"
        assert bridge_path.is_file()
        snapshot = json.loads(bridge_path.read_text(encoding="utf-8"))
        assert snapshot["mode"] == "offline"
        assert snapshot["schema_version"] == 1
        assert body["bridge_updated"] is True

        # Repeated identical execution is an idempotent replay (no duplicate submission).
        calls_after_first = execution.submit_calls
        second = client.post(
            "/api/runner/rebalance-execute",
            json={"strategy_id": "tsmom_vol_target", "confirm": True},
        )
        assert second.status_code == 200, second.text
        assert second.json()["status"] == "idempotent-replay"
        assert execution.submit_calls == calls_after_first


def test_rebalance_execute_requires_confirmation_and_refuses_live(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with _app(settings) as client:
        container = client.app.state.container
        client.app.state.rebalance_service = RebalanceService(
            settings=settings,
            market=container.market,
            catalog=container.strategies,
            execution=FakeExecution(),
            clock=lambda: FIXED_TIMESTAMP,
        )
        unconfirmed = client.post(
            "/api/runner/rebalance-execute",
            json={"strategy_id": "tsmom_vol_target", "confirm": False},
        )
        assert unconfirmed.status_code == 400

    live_settings = settings.model_copy(update={"environment": "live", "ibkr_port": 7496})
    live_service = RebalanceService(
        settings=live_settings,
        market=container.market,
        catalog=container.strategies,
        execution=FakeExecution(ibkr=True, connected=True),
        clock=lambda: FIXED_TIMESTAMP,
    )

    async def _run() -> None:
        with pytest.raises(RebalanceError) as excinfo:
            await live_service.execute(
                RebalanceExecuteRequest(strategy_id="tsmom_vol_target", confirm=True)
            )
        assert excinfo.value.status_code == 409

    import asyncio

    asyncio.run(_run())


def test_rebalance_execute_refuses_unconnected_ibkr(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with _app(settings) as client:
        container = client.app.state.container
        service = RebalanceService(
            settings=settings,
            market=container.market,
            catalog=container.strategies,
            execution=FakeExecution(ibkr=True, connected=False),
            clock=lambda: FIXED_TIMESTAMP,
        )

        import asyncio

        async def _run() -> None:
            with pytest.raises(RebalanceError) as excinfo:
                await service.execute(
                    RebalanceExecuteRequest(strategy_id="tsmom_vol_target", confirm=True)
                )
            assert excinfo.value.status_code == 409
            assert "IBKR paper session" in str(excinfo.value)

        asyncio.run(_run())


def test_rebalance_once_cli_parser_flag() -> None:
    args = _build_parser().parse_args(["--rebalance-once", "--roots", "ES,NQ"])
    assert args.rebalance_once is True
    assert args.roots == "ES,NQ"


def test_runner_rebalance_once_offline_dry_run(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    session = OfflineRunnerSession(
        settings,
        strategy_path=str(settings.configs_dir / "strategies" / "tsmom_vol_target.yaml"),
        roots=("ES", "NQ"),
    )

    import asyncio

    async def _run() -> Any:
        try:
            await session.warm_start()

            # The streamer must never be consumed by the rebalance-once path.
            async def _forbidden() -> Any:
                raise AssertionError("rebalance-once must not stream market data")
                yield  # pragma: no cover

            session._streamer.stream = _forbidden  # type: ignore[method-assign]
            return await session.run_rebalance_once(dry_run=True)
        finally:
            await session.close()

    result = asyncio.run(_run())
    assert result.dry_run is True
    assert result.halted is False
    assert result.nlv == pytest.approx(1_000_000.0)
    assert result.account_currency == settings.account_currency
    assert result.message
    assert result.submitted_orders == ()
    assert result.tracking_error_pct is not None
    # The offline broker is connected/disconnected only (no streamer subscription).
    assert session.snapshot() is not None
