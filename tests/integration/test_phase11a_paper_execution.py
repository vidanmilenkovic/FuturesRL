"""Independent Phase 11A verification: IBKR paper execution hardening.

Verification-only suite for branch ``feat/phase-11a-ibkr-paper-execution``
(tasks 58-61); no production edits unless a real bug is found (reported first).
Everything is hermetic: fake IBKR clients (never TWS), ``tmp_path`` settings,
fake execution/streamer objects for the runner/API surface.

1. **Front-month qualification** (task 58): explicit ``YYYYMM`` month from the
   instrument roll rules, ``reqContractDetailsAsync``/``qualifyContractsAsync``
   before any market-data request, ``conId > 0`` enforced, per ``(root, month)``
   cache, actionable failure names root/exchange/month.
2. **Zero-subscription fallback** (task 58): ``reqMarketDataType(3)`` on connect;
   synchronous permission exception and asynchronous error codes 10189/10090/354
   degrade only the affected root to delayed/historical polling without crashing;
   the remaining roots keep real-time subscriptions; delayed bars are ingested.
3+): runner ``--rebalance-once`` (task 59), API rebalance preview/execute
   (task 59) and DRL checkpoint universe/observation alignment (task 60) are
   covered in the sections below.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from configs.instruments import load_instruments
from configs.settings import Settings
from fastapi.testclient import TestClient
from pydantic import ValidationError
from src.api.main import create_app
from src.data.ibkr_contracts import (
    clear_front_future_cache,
    front_contract_month,
)
from src.data.ibkr_data import IBKRMarketDataStreamer

from tests.fixtures.market import (
    make_account_snapshot,
    make_bars_frame,
    make_contract_targets,
    make_market_state,
    make_weight_vector,
)

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
INSTRUMENTS_PATH = REPO_ROOT / "configs" / "instruments.yaml"
FIXED_CLOCK = datetime(2026, 11, 3, 12, 0, tzinfo=UTC)


def _settings(tmp_path: Path, *, market_data_type: int = 3) -> Settings:
    return Settings(
        environment="paper",
        data_dir=tmp_path / "data",
        parquet_dir=tmp_path / "data" / "parquet",
        duckdb_path=tmp_path / "data" / "phase11a.duckdb",
        report_dir=tmp_path / "artifacts",
        configs_dir=REPO_ROOT / "configs",
        instruments_path=INSTRUMENTS_PATH,
        telemetry_interval_seconds=0.02,
        api_enable_ibkr=False,
        ibkr_paper=True,
        ibkr_market_data_type=market_data_type,
    )


# ---------------------------------------------------------------------------
# Fake IBKR client (front-month qualification + subscriptions + fallback)
# ---------------------------------------------------------------------------


class _FakeContract:
    def __init__(
        self,
        symbol: str,
        *,
        exchange: str = "CME",
        currency: str = "USD",
        month: str | None = None,
        con_id: int | None = None,
    ) -> None:
        self.symbol = symbol
        self.exchange = exchange
        self.currency = currency
        self.lastTradeDateOrContractMonth = month
        self.conId = con_id
        self.includeExpired = False

    def qualified(self, con_id: int) -> _FakeContract:
        return _FakeContract(
            self.symbol,
            exchange=self.exchange,
            currency=self.currency,
            month=self.lastTradeDateOrContractMonth,
            con_id=con_id,
        )


class _FakeDetails:
    def __init__(self, contract: _FakeContract) -> None:
        self.contract = contract


class _FakeSubscription:
    def __init__(self, req_id: int) -> None:
        self.reqId = req_id


class _FakeSignal:
    """Minimal ``ib_async`` event signal supporting ``signal += handler``."""

    def __init__(self) -> None:
        self.handlers: list[Any] = []

    def __iadd__(self, handler: Any) -> _FakeSignal:
        self.handlers.append(handler)
        return self

    def emit(self, *args: object) -> None:
        for handler in self.handlers:
            handler(*args)


class _FakeIB:
    """Recording fake IB client (no SDK, no network, never TWS)."""

    def __init__(
        self,
        *,
        fail_contract_details: bool = False,
        tick_permission_error: Exception | None = None,
        permission_failing_roots: frozenset[str] = frozenset(),
        delayed_rows: list[dict[str, object]] | None = None,
    ) -> None:
        self.fail_contract_details = fail_contract_details
        self.tick_permission_error = tick_permission_error
        self.permission_failing_roots = permission_failing_roots
        self.delayed_rows = delayed_rows or []
        self.events: list[tuple[str, object]] = []
        self.contract_requests: list[_FakeContract] = []
        self.subscriptions: list[tuple[str, _FakeContract]] = []
        self.cancelled: list[tuple[str, object]] = []
        self.market_data_types: list[int] = []
        self.con_id = 9000
        self.errorEvent = _FakeSignal()
        self.error = _FakeSignal()

    # -- connection -----------------------------------------------------------
    def isConnected(self) -> bool:  # noqa: N802
        return True

    async def connectAsync(self, host: str, port: int, client_id: int, timeout: float) -> None:  # noqa: N802
        self.events.append(("connect", host))

    def disconnect(self) -> None:
        self.events.append(("disconnect", ""))

    def reqMarketDataType(self, value: int) -> None:  # noqa: N802
        self.market_data_types.append(value)
        self.events.append(("market_data_type", value))

    # -- qualification --------------------------------------------------------
    async def reqContractDetailsAsync(self, contract: _FakeContract) -> list[_FakeDetails]:  # noqa: N802
        self.contract_requests.append(contract)
        self.events.append(("reqContractDetails", contract))
        if self.fail_contract_details:
            return []
        self.con_id += 1
        return [
            _FakeDetails(
                _FakeContract(
                    contract.symbol,
                    exchange=contract.exchange,
                    currency=contract.currency,
                    month=contract.lastTradeDateOrContractMonth,
                    con_id=self.con_id,
                )
            )
        ]

    async def qualifyContractsAsync(self, *contracts: _FakeContract) -> list[_FakeContract]:  # noqa: N802
        self.events.append(("qualifyContracts", contracts))
        if self.fail_contract_details:
            return []
        qualified: list[_FakeContract] = []
        for contract in contracts:
            self.con_id += 1
            qualified.append(contract.qualified(self.con_id))
        return qualified

    # -- subscriptions --------------------------------------------------------
    def _next_req_id(self) -> int:
        self._req_id = getattr(self, "_req_id", 200) + 1
        return self._req_id

    def _record(self, kind: str, contract: _FakeContract) -> _FakeSubscription:
        subscription = _FakeSubscription(self._next_req_id())
        self.subscriptions.append((kind, contract))
        self.events.append((kind, contract))
        return subscription

    def reqRealTimeBars(  # noqa: N802
        self, contract: _FakeContract, bar_size: int, what: str, rth: bool
    ) -> _FakeSubscription:
        return self._record("bars", contract)

    def reqTickByTickData(  # noqa: N802
        self, contract: _FakeContract, tick_type: str, n: int, ignore: bool
    ) -> _FakeSubscription:
        if self.tick_permission_error is not None and (
            not self.permission_failing_roots or contract.symbol in self.permission_failing_roots
        ):
            raise self.tick_permission_error
        return self._record("ticks", contract)

    def reqMktData(  # noqa: N802
        self, contract: _FakeContract, generic: str, snapshot: bool, regulatory: bool
    ) -> _FakeSubscription:
        return self._record("quotes", contract)

    def reqMktDepth(self, contract: _FakeContract, rows: int, smart: bool) -> _FakeSubscription:  # noqa: N802
        return self._record("depth", contract)

    def reqHistoricalDataAsync(  # noqa: N802
        self, contract: _FakeContract, *args: object, **kwargs: object
    ) -> list[object]:
        return list(self.delayed_rows)

    def cancelRealTimeBars(self, bars: object) -> None:  # noqa: N802
        self.cancelled.append(("bars", bars))

    def cancelTickByTickData(self, contract: object, tick_type: str) -> None:  # noqa: N802
        self.cancelled.append(("ticks", contract))

    def cancelMktData(self, contract: object) -> None:  # noqa: N802
        self.cancelled.append(("quotes", contract))

    def cancelMktDepth(self, contract: object, smart: bool) -> None:  # noqa: N802
        self.cancelled.append(("depth", contract))


def _streamer(
    tmp_path: Path,
    client: _FakeIB,
    *,
    roots: tuple[str, ...] = ("ES",),
    market_data_type: int = 3,
    delayed_bar_source: Any | None = None,
) -> IBKRMarketDataStreamer:
    async def _no_sleep(_delay: float) -> None:
        return None

    def contract_factory(root: str) -> _FakeContract:
        return _FakeContract(root, exchange="CME")

    clear_front_future_cache()
    return IBKRMarketDataStreamer(
        settings=_settings(tmp_path, market_data_type=market_data_type),
        roots=roots,
        instrument_master=load_instruments(INSTRUMENTS_PATH),
        ib_factory=lambda: client,
        contract_factory=contract_factory,
        sleep=_no_sleep,
        clock=lambda: FIXED_CLOCK,
        delayed_poll_interval_seconds=None,  # deterministic: no background task
        delayed_bar_source=delayed_bar_source,
    )


# ---------------------------------------------------------------------------
# 1. Front-month qualification (task 58)
# ---------------------------------------------------------------------------


def test_front_month_contract_month_and_qualification_order(tmp_path: Path) -> None:
    client = _FakeIB()
    streamer = _streamer(tmp_path, client)
    master = load_instruments(INSTRUMENTS_PATH)
    expected_month, expected_expiration = front_contract_month(
        "ES", reference=FIXED_CLOCK, instrument_master=master
    )
    asyncio.run(streamer.connect())

    assert client.market_data_types == [3]
    events = [kind for kind, _ in client.events]
    assert events.index("market_data_type") < events.index("reqContractDetails")
    assert events.index("reqContractDetails") < events.index("bars")
    # Explicit month resolved from roll rules, and the qualified conId used.
    request = client.contract_requests[0]
    assert request.symbol == "ES"
    assert request.conId is None, "a bare/unqualified contract must be resolved first"
    assert request.lastTradeDateOrContractMonth is None  # raw factory contract
    assert expected_month == f"{expected_expiration.year:04d}{expected_expiration.month:02d}"
    diagnostics = streamer.diagnostics()
    qualified = diagnostics["qualified_contracts"]["ES"]
    assert qualified["contract_month"] == expected_month
    assert qualified["con_id"] > 0
    assert qualified["qualified"] is True
    assert qualified["exchange"] == "CME"
    for kind, contract in client.subscriptions:
        assert contract.conId is not None and contract.conId > 0, (
            f"{kind} subscribed with an unqualified contract"
        )


def test_front_month_cache_avoids_repeated_qualification(tmp_path: Path) -> None:
    client = _FakeIB()
    streamer = _streamer(tmp_path, client)
    asyncio.run(streamer.connect())
    qualification_calls = len(client.contract_requests)
    assert qualification_calls == 1
    # A second subscription pass for the same (root, month) hits the cache.
    asyncio.run(streamer._subscribe(client))
    assert len(client.contract_requests) == qualification_calls
    diagnostics = streamer.diagnostics()
    assert diagnostics["qualified_contracts"]["ES"]["con_id"] > 0


def test_front_month_resolution_failure_is_actionable(tmp_path: Path) -> None:
    client = _FakeIB(fail_contract_details=True)
    streamer = _streamer(tmp_path, client, roots=("ES", "NQ"))
    master = load_instruments(INSTRUMENTS_PATH)
    expected_month, _ = front_contract_month("ES", reference=FIXED_CLOCK, instrument_master=master)
    with pytest.raises(RuntimeError) as excinfo:
        asyncio.run(streamer.connect())
    message = str(excinfo.value)
    assert "could not qualify any market-data contract" in message
    assert "root 'ES' on CME" in message
    assert f"contract month {expected_month}" in message
    assert client.subscriptions == [], "no unqualified contract may be subscribed"


def test_market_data_type_setting_validation_and_request(tmp_path: Path) -> None:
    for valid in (1, 2, 3, 4):
        assert _settings(tmp_path, market_data_type=valid).ibkr_market_data_type == valid
    with pytest.raises(ValidationError):
        _settings(tmp_path, market_data_type=5)
    client = _FakeIB()
    streamer = _streamer(tmp_path, client, market_data_type=4)
    asyncio.run(streamer.connect())
    assert client.market_data_types == [4]
    assert streamer.diagnostics()["market_data_type"]["requested"] == 4


# ---------------------------------------------------------------------------
# 2. Zero-subscription delayed fallback (task 58)
# ---------------------------------------------------------------------------


def test_tick_permission_exception_degrades_only_that_root(tmp_path: Path) -> None:
    client = _FakeIB(
        tick_permission_error=RuntimeError("error 10189: no market data permissions"),
        permission_failing_roots=frozenset({"ES"}),
    )
    streamer = _streamer(tmp_path, client, roots=("ES", "NQ"))
    asyncio.run(streamer.connect())  # must not crash

    diagnostics = streamer.diagnostics()
    fallback = diagnostics["market_data_fallback"]
    assert fallback["active"] is True
    assert fallback["roots"] == ["ES"]
    assert "10189" in str(fallback["reasons"]["ES"])
    active = streamer.diagnostics()["subscriptions"]
    assert active == ["NQ"], "the healthy root keeps real-time subscriptions"

    # The paid subscriptions created for the degraded root were cancelled.
    assert any(kind == "bars" for kind, _ in client.cancelled)


def test_async_error_code_degrades_root_and_cancels(tmp_path: Path) -> None:
    client = _FakeIB()
    streamer = _streamer(tmp_path, client, roots=("ES",))
    asyncio.run(streamer.connect())
    assert streamer.diagnostics()["market_data_fallback"]["active"] is False
    contract = client.subscriptions[0][1]
    error = 10090
    req_id = 201  # first subscription handle reqId assigned by the fake
    streamer._on_error_event(req_id, error, "partial subscription", contract)
    diagnostics = streamer.diagnostics()
    assert diagnostics["market_data_fallback"]["active"] is True
    assert diagnostics["market_data_fallback"]["roots"] == ["ES"]
    assert "10090" in str(diagnostics["market_data_fallback"]["reasons"]["ES"])
    assert client.cancelled, "degraded root's subscriptions must be cancelled"


@pytest.mark.parametrize("code", [10189, 10090, 354])
def test_all_permission_error_codes_trigger_fallback(tmp_path: Path, code: int) -> None:
    client = _FakeIB()
    streamer = _streamer(tmp_path, client, roots=("ES",))
    asyncio.run(streamer.connect())
    contract = client.subscriptions[0][1]
    streamer._on_error_event(201, code, f"permission code {code}", contract)
    assert streamer.diagnostics()["market_data_fallback"]["roots"] == ["ES"]


def test_delayed_fallback_ingests_historical_rows(tmp_path: Path) -> None:
    rows = [
        {
            "timestamp_utc": FIXED_CLOCK.replace(minute=55).isoformat(),
            "open": 4800.0,
            "high": 4810.0,
            "low": 4795.0,
            "close": 4805.0,
            "volume": 1234.0,
            "bid": 4804.5,
            "ask": 4805.5,
        }
    ]

    async def delayed_source(
        roots: tuple[str, ...], start: datetime, end: datetime
    ) -> list[dict[str, object]]:
        return rows

    client = _FakeIB(tick_permission_error=RuntimeError("error 354: not subscribed"))
    streamer = _streamer(tmp_path, client, roots=("ES",), delayed_bar_source=delayed_source)
    asyncio.run(streamer.connect())
    ingested = asyncio.run(streamer.poll_delayed_roots())
    assert ingested == {"ES": 1}
    diagnostics = streamer.diagnostics()
    assert diagnostics["market_data_fallback"]["unsupported"] is False
    assert diagnostics["market_data_fallback"]["roots"] == ["ES"]


# ---------------------------------------------------------------------------
# 3. Runner --rebalance-once (task 59): broker only, NLV enriched, bridge
# ---------------------------------------------------------------------------


class _FakeRunnerStreamer:
    """Must never be connected/subscribed by the rebalance-once path."""

    def __init__(self) -> None:
        self.connect_calls = 0
        self.stream_calls = 0
        self.is_connected = False

    async def connect(self) -> None:
        self.connect_calls += 1

    async def disconnect(self) -> None:
        pass

    async def stream(self) -> Any:
        self.stream_calls += 1
        raise AssertionError("--rebalance-once must not open a market-data stream")
        yield  # pragma: no cover


class _FakeRunnerHydrator:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], datetime, float | None]] = []

    def hydrate(
        self,
        roots: tuple[str, ...],
        as_of: datetime,
        *,
        equity: float | None = None,
        allow_backfill: bool | None = None,
    ) -> Any:
        from src.execution.hydrator import HydrationResult

        del allow_backfill
        self.calls.append((tuple(roots), as_of, equity))
        state = make_market_state(roots=tuple(roots), timestamp=as_of)
        return HydrationResult(
            states=(state,),
            bars=make_bars_frame(roots=tuple(roots), periods=2),
            roots=tuple(roots),
            lookback_days=400,
            coverage=dict.fromkeys(roots, 300),
            required_bars=260,
            backfilled=False,
            warnings=(),
            as_of=as_of,
        )


class _FakeRunnerBroker:
    def __init__(self, *, net_liquidation: float = 2_000_000.0) -> None:
        self.net_liquidation = net_liquidation
        self.connect_calls = 0
        self.disconnect_calls = 0
        self.submit_calls = 0
        self.submitted: list[Any] = []
        self.is_locked = False
        self.positions: dict[str, Any] = {}

    async def connect(self) -> None:
        self.connect_calls += 1

    async def disconnect(self) -> None:
        self.disconnect_calls += 1

    async def get_account(self) -> Any:
        return make_account_snapshot(
            net_liquidation=self.net_liquidation, initial_margin_used=100_000.0
        )

    async def get_positions(self) -> dict[str, Any]:
        return dict(self.positions)

    async def submit_targets(
        self,
        targets: Any,
        current_positions: dict[str, int],
        timestamp: datetime,
        *,
        flatten: bool = False,
    ) -> list[Any]:
        from src.core.types import OrderReceipt, OrderStatus

        del flatten
        self.submit_calls += 1
        self.submitted.append((targets, dict(current_positions), timestamp))
        return [
            OrderReceipt(
                order_id=f"reb-{index}",
                status=OrderStatus.FILLED,
                filled_quantity=target.quantity,
                average_price=target.price,
                timestamp=timestamp,
            )
            for index, target in enumerate(targets)
        ]


class _FakeRunnerPipeline:
    name = "fake-rebalance-pipeline"

    def __init__(self) -> None:
        self.states: list[Any] = []

    def run(self, state: Any) -> Any:
        from src.core.types import PipelineResult

        self.states.append(state)
        weights = make_weight_vector({"ES": 0.5})
        return PipelineResult(
            timestamp=state.timestamp,
            universe=("ES",),
            base_weights=weights,
            timing_weights=weights,
            risk_weights=weights,
            final_weights=weights,
            targets=make_contract_targets({"ES": 2}),
        )


def _runner_for_rebalance(
    tmp_path: Path, *, dry_run: bool = False
) -> tuple[
    Any, _FakeRunnerBroker, _FakeRunnerStreamer, _FakeRunnerHydrator, _FakeRunnerPipeline, Any
]:
    from src.execution.runner import TradingRunner
    from src.execution.runner_bridge import RunnerBridgeStore

    settings = _settings(tmp_path)
    settings = settings.model_copy(update={"runner_bridge_enabled": True})
    streamer = _FakeRunnerStreamer()
    broker = _FakeRunnerBroker()
    hydrator = _FakeRunnerHydrator()
    pipeline = _FakeRunnerPipeline()
    bridge = RunnerBridgeStore(settings)
    runner = TradingRunner(
        settings=settings,
        pipeline=pipeline,
        streamer=streamer,
        broker=broker,
        hydrator=hydrator,
        instrument_master=load_instruments(INSTRUMENTS_PATH),
        clock=lambda: FIXED_CLOCK,
        dry_run=dry_run,
        bridge_store=bridge,
    )
    return runner, broker, streamer, hydrator, pipeline, bridge


def test_rebalance_once_cli_parser_flag(tmp_path: Path) -> None:
    from src.execution.runner import _build_parser

    parser = _build_parser()
    help_text = parser.format_help()
    assert "--rebalance-once" in help_text
    assert "--dry-run" in help_text
    args = parser.parse_args(["--rebalance-once", "--dry-run", "--roots", "ES,NQ"])
    assert args.rebalance_once is True
    assert args.dry_run is True
    assert args.roots == "ES,NQ"


def test_runner_rebalance_once_broker_only_nlv_enrichment_and_bridge(tmp_path: Path) -> None:
    runner, broker, streamer, hydrator, pipeline, bridge = _runner_for_rebalance(tmp_path)
    result = asyncio.run(runner.run_rebalance_once(roots=("ES",)))

    assert streamer.connect_calls == 0, "market-data streamer must never be connected"
    assert streamer.stream_calls == 0, "market-data streamer must never be streamed"
    assert broker.connect_calls == 1 and broker.disconnect_calls == 1
    assert hydrator.calls, "rebalance-once must hydrate before computing targets"
    assert result.nlv == pytest.approx(2_000_000.0)
    assert result.halted is False
    assert result.positions == {}
    assert result.target_contracts == {"ES": 2}
    # NLV enrichment: the pipeline saw the live account NLV, not a cold state.
    assert pipeline.states and pipeline.states[-1].equity == pytest.approx(2_000_000.0)
    assert result.proposed_orders, "delta orders should be proposed"
    proposed = result.proposed_orders[0]
    assert proposed["side"] == "BUY"
    assert proposed["quantity"] == 2
    assert result.submitted_orders and broker.submit_calls == 1
    assert result.margin_utilization is not None
    assert result.tracking_error_pct is not None
    assert result.bridge_published is True
    snapshot = bridge.load()
    assert snapshot is not None
    assert snapshot.account["net_liquidation"] == pytest.approx(2_000_000.0)


def test_runner_rebalance_once_dry_run_submits_nothing(tmp_path: Path) -> None:
    runner, broker, streamer, _hydrator, _pipeline, bridge = _runner_for_rebalance(
        tmp_path, dry_run=True
    )
    result = asyncio.run(runner.run_rebalance_once(roots=("ES",)))
    assert streamer.connect_calls == 0
    assert result.dry_run is True
    assert result.proposed_orders
    assert result.submitted_orders == ()
    assert broker.submit_calls == 0
    assert bridge.load() is not None, "dry run still refreshes the bridge snapshot"


# ---------------------------------------------------------------------------
# 4. API rebalance preview/execute (task 59)
# ---------------------------------------------------------------------------


class _FakeRebalanceExecution:
    """Execution adapter fake for the dashboard rebalance endpoints."""

    def __init__(
        self,
        *,
        ibkr: bool = False,
        connected: bool = False,
        locked: bool = False,
    ) -> None:
        from types import SimpleNamespace

        self._ibkr = ibkr
        self._connected = connected
        self._locked = locked
        self.submit_calls = 0
        self.submitted: list[Any] = []
        self.translator = SimpleNamespace(max_margin_utilization=0.60)

    @property
    def ibkr_enabled(self) -> bool:
        return self._ibkr

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def is_locked(self) -> bool:
        return self._locked

    async def account(self) -> Any:
        from src.api.schemas import AccountSnapshotResponse

        nlv = 2_000_000.0
        return AccountSnapshotResponse(
            timestamp=FIXED_CLOCK,
            net_liquidation=nlv,
            cash_balance=nlv,
            buying_power=nlv,
            initial_margin_used=100_000.0,
            maintenance_margin_used=90_000.0,
            excess_liquidity=nlv - 100_000.0,
            margin_utilization=0.05,
            day_pnl=0.0,
            day_pnl_pct=0.0,
            gross_leverage=0.0,
            net_exposure=0.0,
            source="simulated" if not self._ibkr else "engine",
        )

    async def positions(self) -> dict[str, Any]:
        from src.core.types import Position

        return {
            "ES": Position(
                instrument_id="ES",
                quantity=10,
                average_price=4800.0,
                market_price=4800.0,
                multiplier=50.0,
                initial_margin=15_000.0,
                contract_symbol="ES-202612",
            )
        }

    async def submit_orders(self, orders: Any) -> list[Any]:
        from src.core.types import OrderReceipt, OrderStatus

        self.submit_calls += 1
        self.submitted.extend(orders)
        return [
            OrderReceipt(
                order_id=order.order_id,
                status=OrderStatus.FILLED,
                filled_quantity=order.quantity,
                average_price=order.limit_price or 4800.0,
                commission=2.0,
                timestamp=FIXED_CLOCK,
            )
            for order in orders
        ]


@contextmanager
def _rebalance_app(
    tmp_path: Path,
    execution: _FakeRebalanceExecution,
    *,
    environment: str = "backtest",
) -> Any:
    from src.api.main import create_app as _create_app
    from src.api.rebalance_service import RebalanceService

    settings = _settings(tmp_path)
    settings = settings.model_copy(update={"environment": environment})
    app = _create_app(settings, clock=lambda: FIXED_CLOCK)
    with TestClient(app) as client:
        container = app.state.container
        service = RebalanceService(
            settings=settings,
            market=container.market,
            catalog=container.strategies,
            execution=execution,
            clock=lambda: FIXED_CLOCK,
        )
        app.state.rebalance_service = service
        yield client, service


def test_api_rebalance_preview_offline_contract_submits_nothing(tmp_path: Path) -> None:
    execution = _FakeRebalanceExecution()
    with _rebalance_app(tmp_path, execution) as (client, _service):
        response = client.post(
            "/api/runner/rebalance-preview", json={"strategy_id": "tsmom_vol_target"}
        )
        assert response.status_code == 200, response.text
        body = response.json()
        for field in (
            "strategy_id",
            "account_nlv",
            "account_currency",
            "current_positions",
            "target_contracts",
            "proposed_orders",
            "projected_margin_utilization",
            "margin_utilization_cap",
            "tracking_error_pct",
            "mode",
            "generated_at",
            "strategy_roots",
            "warnings",
        ):
            assert field in body
        assert body["strategy_id"] == "tsmom_vol_target"
        assert body["account_nlv"] == pytest.approx(2_000_000.0)
        assert body["account_currency"]
        assert body["mode"] == "offline"
        assert body["margin_utilization_cap"] == pytest.approx(0.60)
        assert body["current_positions"]
        assert body["current_positions"][0]["instrument_id"] == "ES"
        assert isinstance(body["target_contracts"], list)
        assert isinstance(body["proposed_orders"], list)
        for order in body["proposed_orders"]:
            assert order["pre_trade_gate_passed"] is True
            assert order["rejection_reason"] is None
        assert 0.0 <= body["projected_margin_utilization"] <= 1.0
        assert execution.submit_calls == 0, "preview must never submit orders"

        # Idempotent preview: repeated calls submit nothing.
        again = client.post(
            "/api/runner/rebalance-preview", json={"strategy_id": "tsmom_vol_target"}
        )
        assert again.status_code == 200
        assert execution.submit_calls == 0

        unknown = client.post(
            "/api/runner/rebalance-preview", json={"strategy_id": "nope_strategy"}
        )
        assert unknown.status_code == 404


def test_api_rebalance_execute_submits_once_and_is_idempotent(tmp_path: Path) -> None:
    execution = _FakeRebalanceExecution()
    with _rebalance_app(tmp_path, execution) as (client, _service):
        missing_confirm = client.post(
            "/api/runner/rebalance-execute",
            json={"strategy_id": "tsmom_vol_target", "confirm": False},
        )
        assert missing_confirm.status_code == 400
        assert execution.submit_calls == 0, "no confirmation -> no submission"

        response = client.post(
            "/api/runner/rebalance-execute",
            json={"strategy_id": "tsmom_vol_target", "confirm": True},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["mode"] == "offline"
        assert body["status"] in {"submitted", "no-op"}
        if body["status"] == "submitted":
            assert body["submitted_order_ids"]
            assert body["orders"]
            assert body["fills"]
            assert execution.submit_calls == 1
            assert body["bridge_updated"] is True
            snapshot_path = (
                client.app.state.container.settings.runner_bridge_dir / "runner_state.json"
            )
            assert snapshot_path.is_file(), "execute must publish a bridge snapshot"

            replay = client.post(
                "/api/runner/rebalance-execute",
                json={"strategy_id": "tsmom_vol_target", "confirm": True},
            )
            assert replay.status_code == 200
            replay_body = replay.json()
            assert replay_body["status"] == "idempotent-replay"
            assert execution.submit_calls == 1, "identical execution must not duplicate orders"
        else:  # pragma: no cover - defensive: real offline pipeline should trade
            assert body["message"]


def test_api_rebalance_execute_refuses_live_and_unconnected_ibkr(tmp_path: Path) -> None:
    # Live environment: execute is refused with 409 (preview would be read-only).
    live_execution = _FakeRebalanceExecution()
    with _rebalance_app(tmp_path / "live", live_execution, environment="live") as (client, _svc):
        refused = client.post(
            "/api/runner/rebalance-execute",
            json={"strategy_id": "tsmom_vol_target", "confirm": True},
        )
        assert refused.status_code == 409, refused.text
        assert "live environment" in refused.json()["detail"]
        assert live_execution.submit_calls == 0

    # IBKR adapter enabled but not connected: no simulated substitution.
    unconnected = _FakeRebalanceExecution(ibkr=True, connected=False)
    with _rebalance_app(tmp_path / "unconnected", unconnected) as (client2, _svc2):
        preview = client2.post(
            "/api/runner/rebalance-preview", json={"strategy_id": "tsmom_vol_target"}
        )
        assert preview.status_code == 200
        assert preview.json()["mode"] == "offline"
        assert any("not connected" in warning for warning in preview.json()["warnings"])
        refused = client2.post(
            "/api/runner/rebalance-execute",
            json={"strategy_id": "tsmom_vol_target", "confirm": True},
        )
        assert refused.status_code == 409, refused.text
        assert unconnected.submit_calls == 0


def test_api_rebalance_ibkr_paper_mode_executes(tmp_path: Path) -> None:
    execution = _FakeRebalanceExecution(ibkr=True, connected=True)
    with _rebalance_app(tmp_path, execution) as (client, _service):
        preview = client.post(
            "/api/runner/rebalance-preview", json={"strategy_id": "tsmom_vol_target"}
        )
        assert preview.status_code == 200
        assert preview.json()["mode"] == "ibkr-paper"
        executed = client.post(
            "/api/runner/rebalance-execute",
            json={"strategy_id": "tsmom_vol_target", "confirm": True},
        )
        assert executed.status_code == 200, executed.text
        body = executed.json()
        assert body["mode"] == "ibkr-paper"
        if body["status"] == "submitted":
            assert execution.submit_calls == 1
            assert body["submitted_order_ids"]


# ---------------------------------------------------------------------------
# 5. DRL checkpoint meta universe + observation alignment (task 60)
# ---------------------------------------------------------------------------


def _write_drl_meta(
    checkpoint: Path,
    *,
    universe: tuple[str, ...] = ("ES", "NQ", "CL"),
    feature_names: tuple[str, ...] = ("ret_1d",),
    observation_shape: tuple[int, ...] | None = None,
    action_shape: tuple[int, ...] | None = None,
) -> Path:
    import json

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"fake-sb3-zip")
    n_assets = len(universe)
    payload: dict[str, object] = {
        "universe": list(universe),
        "feature_names": list(feature_names),
        "observation_shape": list(
            observation_shape or (n_assets * len(feature_names) + n_assets + 1,)
        ),
        "action_shape": list(action_shape or (n_assets,)),
        "algorithm": "PPO",
        "total_timesteps": 64,
        "strategy_id": "drl_ppo_llm_risk",
        "trained_at": "2026-01-01T00:00:00+00:00",
    }
    meta_path = Path(f"{checkpoint}.meta.json")
    meta_path.write_text(json.dumps(payload), encoding="utf-8")
    return meta_path


class _FakeDRLModel:
    def __init__(self, action: object, *, observation_shape: tuple[int, ...] | None = None) -> None:
        from types import SimpleNamespace

        self.action = action
        self.observation_space = (
            None if observation_shape is None else SimpleNamespace(shape=observation_shape)
        )

    def predict(self, observation: object, deterministic: bool = True) -> tuple[object, None]:
        assert deterministic is True
        assert observation is not None
        return self.action, None


def test_drl_bind_preview_and_params_patch(tmp_path: Path) -> None:
    from src.strategies.allocators.drl import (
        checkpoint_bind_preview,
        checkpoint_strategy_params_patch,
        checkpoint_universe_bind_patch,
    )

    checkpoint = tmp_path / "models" / "ppo-test.zip"
    _write_drl_meta(checkpoint, universe=("ES", "NQ", "CL"))
    preview = checkpoint_bind_preview(checkpoint)
    assert preview["metadata_available"] is True
    assert preview["universe"] == ["ES", "NQ", "CL"]
    assert preview["feature_names"] == ["ret_1d"]
    assert preview["roots_patch"] == {"roots": ["ES", "NQ", "CL"], "top_n": 3}
    assert preview["params_patch"] == {"roots": "ES,NQ,CL", "top_n": 3}
    assert preview["warning"] is None
    assert checkpoint_universe_bind_patch(checkpoint) == {
        "roots": ["ES", "NQ", "CL"],
        "top_n": 3,
    }
    assert checkpoint_strategy_params_patch(checkpoint) == {"roots": "ES,NQ,CL", "top_n": 3}

    missing = checkpoint_bind_preview(tmp_path / "models" / "missing.zip")
    assert missing["metadata_available"] is False
    assert missing["roots_patch"] is None and missing["params_patch"] is None
    assert missing["warning"], "missing metadata must surface a warning, not raise"


def test_drl_observation_alignment_and_action_mapping(tmp_path: Path) -> None:
    from src.strategies.allocators.drl import DRLAllocator

    checkpoint = tmp_path / "models" / "ppo-align.zip"
    _write_drl_meta(checkpoint, universe=("ES", "NQ", "CL"), feature_names=("ret_1d",))
    fake_model = _FakeDRLModel(action=[0.1, 0.2, 0.3])
    allocator = DRLAllocator(
        model_path=str(checkpoint), model=fake_model, max_instrument_weight=0.5
    )

    state = make_market_state(
        roots=("ES", "NQ", "CL", "GC"),
        features={
            "ES": {"ret_1d": 0.11},
            "NQ": {"ret_1d": 0.22},
            "CL": {"ret_1d": 0.33},
            "GC": {"ret_1d": 0.44},
        },
    )
    # Candidate order differs from the checkpoint order; GC is extra, CL missing.
    observation = allocator.build_observation(state, ("NQ", "ES", "GC"))
    assert observation.shape == (3 * 1 + 3 + 1,)
    # Checkpoint rows are ES, NQ, CL (checkpoint order), not candidate order.
    assert observation[0] == pytest.approx(0.11)  # ES features
    assert observation[1] == pytest.approx(0.22)  # NQ features
    assert observation[2] == pytest.approx(0.0)  # CL absent -> flat
    assert observation[4] == pytest.approx(0.0)  # CL held weight stays zero

    weights = allocator.weights_from_action(
        ("NQ", "ES", "GC"),
        [0.1, 0.2, 0.3],
        action_candidates=("ES", "NQ", "CL"),
    )
    assert weights.weights["ES"] == pytest.approx(0.1)
    assert weights.weights["NQ"] == pytest.approx(0.2)
    assert weights.weights["GC"] == pytest.approx(0.0)

    with pytest.raises(ValueError):
        allocator.weights_from_action(("ES",), [0.1], action_candidates=("ES", "NQ"))


def test_drl_observation_shape_mismatch_is_actionable(tmp_path: Path) -> None:
    from src.strategies.allocators.drl import DRLAllocator

    checkpoint = tmp_path / "models" / "ppo-bad-meta.zip"
    _write_drl_meta(
        checkpoint,
        universe=("ES", "NQ"),
        feature_names=("ret_1d",),
        observation_shape=(99,),
    )
    fake_model = _FakeDRLModel(action=[0.1, 0.2])
    allocator = DRLAllocator(model_path=str(checkpoint), model=fake_model)
    state = make_market_state(roots=("ES", "NQ"))
    with pytest.raises(RuntimeError) as excinfo:
        allocator.allocate(("ES", "NQ"), state)
    assert "observation" in str(excinfo.value)
    assert "99" in str(excinfo.value)


def test_drl_missing_metadata_falls_back_with_warning(tmp_path: Path) -> None:
    from src.strategies.allocators.drl import DRLAllocator

    checkpoint = tmp_path / "models" / "ppo-no-meta.zip"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"fake-sb3-zip")
    fake_model = _FakeDRLModel(action=[0.1, 0.2])
    allocator = DRLAllocator(model_path=str(checkpoint), model=fake_model)
    state = make_market_state(roots=("ES", "NQ"))
    with pytest.warns(UserWarning, match="no metadata sidecar"):
        observation = allocator.build_observation(state, ("NQ", "ES"))
    assert observation.shape == (2 * 7 + 2 + 1,), "legacy candidate-order layout"


def test_api_drl_models_exposes_checkpoint_universe(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.drl_models_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(settings.drl_models_dir) / "ppo-api.zip"
    _write_drl_meta(checkpoint, universe=("ES", "NQ", "CL"))
    app = create_app(settings, clock=lambda: FIXED_CLOCK)
    with TestClient(app) as client:
        response = client.get("/api/drl/models")
        assert response.status_code == 200, response.text
        body = response.json()
        models = {row["checkpoint_name"]: row for row in body["models"]}
        assert "ppo-api" in models
        assert models["ppo-api"]["universe"] == ["ES", "NQ", "CL"]
        assert models["ppo-api"]["metadata_available"] is True
