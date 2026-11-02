"""Concrete service adapters for the FuturesRL-X telemetry API.

Wiring philosophy
-----------------
Each adapter eagerly uses the **real** module it fronts and only degrades to a
deterministic fixture when that module is missing or a call fails. Degradation is
never silent: :class:`Container` records a human-readable warning, ``/health``
reports ``status="degraded"``/``offline_mode=true`` and backtest runs carry
``data_source="fixture"``.

Real wiring
-----------
* :class:`EngineMarketService` — real ``src.data.market_state_builder`` +
  ``src.strategies.build_pipeline`` + ``src.risk.contract_translator`` +
  ``src.risk.quant_risk_overlay`` (LLM overlay used when importable, skipped
  otherwise and shown in the pipeline diagnostics).
* :class:`SimulatedExecutionService` — real ``src.backtest.simulated_broker``
  (``SimulatedBroker``) seeded with the demo book; the kill switch calls the
  real ``flatten_all_sync`` path and locks execution.
* :class:`IBKRExecutionService` — opt-in real ``IBKRConnectionManager`` +
  ``OrderManagementSystem`` wiring (``FUTURESRLX_API_ENABLE_IBKR=1``); connects in
  the lifespan and transparently uses the offline simulated broker while the
  session is unavailable.
* :class:`EngineRiskService` — real ``RiskSettings`` + deterministic Risk-First
  assessment/projection (same Module-3 formulas as the LLM overlay).
* Backtests — see :mod:`src.api.backtests_service` (real ``BacktestEngine``).
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import polars as pl
from configs.instruments import InstrumentsConfig
from configs.settings import Settings
from configs.strategy import StrategyConfig, discover_strategy_configs, parse_strategy_config

from src.api import fixtures
from src.api.protocols import (
    BacktestService,
    ExecutionService,
    MarketDataService,
    RiskService,
    StrategyCatalog,
)
from src.api.schemas import (
    AccountSnapshotResponse,
    EquityPoint,
    FlattenAllResponse,
    FlattenReceipt,
    HealthResponse,
    PipelineDiagnostic,
    PipelineInstrumentRow,
    PipelineStage,
    PipelineWaterfallResponse,
    PositionRow,
    RiskStateResponse,
    RiskThresholdOverride,
    RiskThresholds,
    StageConfig,
    StrategyConfigUpdate,
    StrategyConfigView,
    TelemetryStatusResponse,
    TermStructureResponse,
)
from src.api.telemetry import TelemetryBroadcaster
from src.core.interfaces import ContractTranslatorProtocol, RiskOverlay, StrategyPipeline
from src.core.types import (
    AccountSnapshot,
    ContractTargetSet,
    InstrumentId,
    MarketState,
    OrderSide,
    OrderStatus,
    PipelineResult,
    Position,
    WeightVector,
)

if TYPE_CHECKING:  # pragma: no cover - typing-only imports of optional layers
    from src.backtest.simulated_broker import BarQuote, SimulatedBroker
    from src.execution.ibkr_client import IBKRConnectionManager
    from src.execution.oms import OrderManagementSystem
    from src.risk.llm_risk_overlay import LLMRiskOverlay, MacroEventCalendar

logger = logging.getLogger(__name__)

__all__ = [
    "Container",
    "EngineMarketService",
    "EngineRiskService",
    "FixtureExecutionService",
    "FixtureMarketService",
    "IBKRExecutionService",
    "InMemoryStrategyCatalog",
    "SimulatedExecutionService",
    "build_container",
]

#: Env flag that opts into real IBKR wiring instead of the simulated broker.
IBKR_ENABLE_ENV: str = "FUTURESRLX_API_ENABLE_IBKR"

#: How long the kill switch keeps execution locked after a flatten.
KILL_SWITCH_LOCK_MINUTES: int = 30

#: Default strategy shown by the pipeline inspector when available.
DEFAULT_STRATEGY_ID: str = "tsmom_vol_target"

#: Default initial equity for the offline account/pipeline fixtures.
INITIAL_EQUITY: float = fixtures.DEFAULT_INITIAL_EQUITY

#: Deterministic average-price offsets for the fixture book (entry slippage).
_ENTRY_OFFSETS: Mapping[str, float] = {
    "ES": 0.0079,
    "NQ": 0.0061,
    "CL": -0.0043,
    "GC": 0.0112,
    "ZN": -0.0018,
    "MES": 0.0042,
}


def _utc(value: datetime) -> datetime:
    return value.astimezone(UTC)


def _metadata_float(diagnostics: Mapping[str, object], key: str) -> float | None:
    value = diagnostics.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _metadata_names(diagnostics: Mapping[str, object], key: str) -> list[str]:
    value = diagnostics.get(key)
    if isinstance(value, str):
        return [value]
    if isinstance(value, tuple | list):
        return [str(item) for item in cast(Sequence[object], value)]
    return []


def _metadata_mapping(diagnostics: Mapping[str, object], key: str) -> dict[str, object]:
    value = diagnostics.get(key)
    if isinstance(value, Mapping):
        return {
            str(item_key): item for item_key, item in cast(Mapping[object, object], value).items()
        }
    return {}


class _NamedRiskOverlay:
    """Adapter exposing a real risk overlay under its declarative YAML stage name.

    ``build_pipeline`` matches injected overlays to the ``risk:`` stages declared
    in the strategy YAML by name (``quant_risk`` / ``llm_risk``); the production
    overlays use their own class names, so this thin adapter provides the
    declarative alias without touching either module.
    """

    __slots__ = ("_inner", "_name")

    def __init__(self, inner: RiskOverlay, name: str) -> None:
        self._inner = inner
        self._name = name

    @property
    def name(self) -> str:
        """Declarative stage name (``quant_risk`` / ``llm_risk``)."""
        return self._name

    def apply_risk(self, weights: WeightVector, state: MarketState) -> WeightVector:
        """Delegate to the wrapped production overlay."""
        return self._inner.apply_risk(weights, state)


# ---------------------------------------------------------------------------
# Strategy catalog (declarative YAML editor, in memory)
# ---------------------------------------------------------------------------


class _StageLike(Protocol):
    """Structural view of a ``configs.strategy.StageConfig``."""

    @property
    def name(self) -> str: ...

    @property
    def params(self) -> Mapping[str, object]: ...


def _stage_config(stage: object) -> StageConfig:
    model = cast(_StageLike, stage)
    params: dict[str, bool | int | float | str] = {}
    for key, value in model.params.items():
        if isinstance(value, bool | int | float | str):
            params[str(key)] = value
    return StageConfig(name=str(model.name), params=params)


class InMemoryStrategyCatalog:
    """Validated strategy configs discovered from ``configs/strategies/*.yaml``.

    ``PUT /api/strategies/{id}`` merges the edit, re-validates the whole pipeline
    with the frozen ``configs.strategy`` models and keeps the result **in
    memory** (never writing the committed YAML) — the API is a runtime editor,
    not a config writer.
    """

    def __init__(self, *, settings: Settings, clock: fixtures.Clock = fixtures.utc_now) -> None:
        self._settings = settings
        self._clock = clock
        self._configs: dict[str, StrategyConfig] = {}
        self._updated_at: dict[str, datetime] = {}
        directory = settings.strategies_dir
        if directory.is_dir():
            for strategy_id, config in discover_strategy_configs(directory).items():
                self._configs[strategy_id] = config
                self._updated_at[strategy_id] = datetime.fromtimestamp(
                    (directory / f"{strategy_id}.yaml").stat().st_mtime, tz=UTC
                )
        else:  # pragma: no cover - committed configs are always present
            logger.warning(
                "strategy directory %s is missing; API starts with no configs", directory
            )

    def list_configs(self) -> list[StrategyConfigView]:
        """Every config, sorted by strategy id."""
        return [self._view(self._configs[key]) for key in sorted(self._configs)]

    def get(self, strategy_id: str) -> StrategyConfigView | None:
        """One strategy view or ``None``."""
        config = self._configs.get(strategy_id.strip().lower())
        return self._view(config) if config is not None else None

    def update(self, strategy_id: str, update: StrategyConfigUpdate) -> StrategyConfigView | None:
        """Merge, validate and store an edited config (in memory)."""
        key = strategy_id.strip().lower()
        current = self._configs.get(key)
        if current is None:
            return None
        payload = current.model_dump()
        changes = update.model_dump(exclude_none=True)
        for name, value in changes.items():
            payload[name] = value
        payload["id"] = current.id
        validated = parse_strategy_config(payload)
        self._configs[key] = validated
        self._updated_at[key] = self._clock()
        return self._view(validated)

    def core_config(self, strategy_id: str) -> StrategyConfig | None:
        """Validated core config used by the backtest engine."""
        return self._configs.get(strategy_id.strip().lower())

    def available_ids(self) -> list[str]:
        """All strategy ids, sorted."""
        return sorted(self._configs)

    def _view(self, config: StrategyConfig) -> StrategyConfigView:
        return StrategyConfigView(
            id=config.id,
            description=config.description,
            tags=list(config.tags),
            universe=_stage_config(config.universe),
            allocator=_stage_config(config.allocator),
            timing=[_stage_config(stage) for stage in config.timing],
            risk=[_stage_config(stage) for stage in config.risk],
            updated_at=self._updated_at.get(config.id, self._clock()),
        )


# ---------------------------------------------------------------------------
# Execution services
# ---------------------------------------------------------------------------


def _quotes_for_market(market: fixtures.OfflineMarket) -> dict[str, BarQuote]:
    """Latest deterministic bar per root as execution-layer ``BarQuote``s."""
    from src.backtest.simulated_broker import BarQuote

    latest = (
        market.bars.filter(pl.col("timestamp_utc") == pl.lit(market.as_of))
        .sort("root_symbol")
        .iter_rows(named=True)
    )
    quotes: dict[str, BarQuote] = {}
    for record in latest:
        root = str(record["root_symbol"])
        expiration = record.get("expiration_date")
        quotes[root] = BarQuote(
            root=root,
            open=float(record["open"]),
            high=float(record["high"]),
            low=float(record["low"]),
            close=float(record["close"]),
            volume=float(record["volume"]),
            expiration=expiration if isinstance(expiration, date) else None,
            contract_symbol=f"{root}_FIXTURE",
            settlement=float(record["settlement"]),
        )
    return quotes


def _receipt_status(status: OrderStatus) -> str:
    if status is OrderStatus.FILLED:
        return "FILLED"
    if status is OrderStatus.REJECTED:
        return "REJECTED"
    if status in (OrderStatus.SUBMITTED, OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED):
        return "SUBMITTED"
    return "UNKNOWN"


def _account_response(
    *,
    snapshot: AccountSnapshot,
    positions: Mapping[InstrumentId, Position],
    master: InstrumentsConfig,
    market: fixtures.OfflineMarket,
    curve: Sequence[EquityPoint],
    day_pnl: float,
) -> AccountSnapshotResponse:
    """Map an account snapshot + positions into the frozen dashboard contract."""
    rows: list[PositionRow] = fixtures.position_rows(
        master, positions, snapshot.net_liquidation, as_of=market.as_of.date()
    )
    gross_notional = sum(abs(row.notional) for row in rows)
    net_exposure = sum(row.notional for row in rows)
    return AccountSnapshotResponse(
        timestamp=snapshot.timestamp,
        net_liquidation=snapshot.net_liquidation,
        cash_balance=snapshot.cash_balance,
        buying_power=snapshot.buying_power,
        initial_margin_used=snapshot.initial_margin_used,
        maintenance_margin_used=snapshot.maintenance_margin_used,
        excess_liquidity=snapshot.excess_liquidity,
        margin_utilization=snapshot.margin_utilization,
        day_pnl=day_pnl,
        day_pnl_pct=day_pnl / snapshot.net_liquidation if snapshot.net_liquidation > 0 else 0.0,
        gross_leverage=gross_notional / snapshot.net_liquidation
        if snapshot.net_liquidation
        else 0.0,
        net_exposure=net_exposure,
        equity_curve=list(curve),
        positions=rows,
    )


class _KillSwitchState:
    """Shared kill-switch latch used by every execution adapter."""

    def __init__(self, clock: fixtures.Clock) -> None:
        self._clock = clock
        self._locked_until: datetime | None = None
        self._reason = ""

    @property
    def is_locked(self) -> bool:
        """Whether execution is currently halted."""
        if self._locked_until is None:
            return False
        return self._clock() < self._locked_until

    @property
    def reason(self) -> str:
        """Human-readable halt reason (empty when unlocked)."""
        return self._reason if self.is_locked else ""

    @property
    def locked_until(self) -> datetime | None:
        """Lock expiry timestamp (``None`` when never engaged)."""
        return self._locked_until

    def engage(self, reason: str, *, minutes: int = KILL_SWITCH_LOCK_MINUTES) -> datetime:
        """Engage the kill switch and return the lock expiry."""
        self._reason = reason
        self._locked_until = self._clock() + timedelta(minutes=minutes)
        logger.critical("API kill switch engaged: %s (until %s)", reason, self._locked_until)
        return self._locked_until

    def release(self) -> None:
        """Release the kill switch (manual reset)."""
        self._locked_until = None
        self._reason = ""


class SimulatedExecutionService:
    """Offline account/execution service backed by the real ``SimulatedBroker``.

    The broker is seeded with the deterministic demo book by submitting real
    market orders through its matching engine, so account valuation, margin
    requirements and the emergency flatten path all exercise production code.
    """

    def __init__(
        self,
        *,
        market: fixtures.OfflineMarket,
        settings: Settings,
        clock: fixtures.Clock = fixtures.utc_now,
    ) -> None:
        from src.backtest.simulated_broker import (
            SimulatedBroker,
            TradeKind,
        )

        self._market = market
        self._master = market.instrument_master
        self._settings = settings
        self._clock = clock
        self._broker: SimulatedBroker = SimulatedBroker(
            self._master,
            initial_cash=INITIAL_EQUITY,
            timestamp=market.as_of,
            name="simulated-offline",
        )
        self._broker.set_market(market.as_of, _quotes_for_market(market))
        self._broker._connected = True
        self._trade_kind = TradeKind
        self._kill_switch = _KillSwitchState(clock)
        self._seed_book()

    def _seed_book(self) -> None:
        from src.backtest.simulated_broker import TradeKind
        from src.core.types import OrderEvent

        for index, (root, quantity) in enumerate(fixtures.fixture_book()):
            if root not in self._market.prices or quantity == 0:
                continue
            order = OrderEvent.market(
                order_id=f"seed-{root}-{index}",
                timestamp=self._market.as_of,
                instrument_id=root,
                side=OrderSide.BUY if quantity > 0 else OrderSide.SELL,
                quantity=abs(quantity),
                contract_symbol=f"{root}_FIXTURE",
            )
            receipt = self._broker.place_order(order, kind=TradeKind.ENTRY)
            if receipt.status is OrderStatus.REJECTED:
                logger.warning("seed order %s rejected: %s", order.order_id, receipt.message)

    async def account(self) -> AccountSnapshotResponse:
        """Account snapshot from the simulated broker state."""
        snapshot = self._broker.get_account_sync(self._market.as_of)
        positions = self._broker.get_positions_sync()
        curve = fixtures.equity_history(self._market, snapshot.net_liquidation, now=self._clock())
        day_pnl = curve[-1].value - curve[-2].value if len(curve) > 1 else 0.0
        return _account_response(
            snapshot=snapshot,
            positions=positions,
            master=self._master,
            market=self._market,
            curve=curve,
            day_pnl=day_pnl,
        )

    async def positions(self) -> Mapping[InstrumentId, Position]:
        """Current broker positions keyed by root."""
        return self._broker.get_positions_sync()

    @property
    def is_locked(self) -> bool:
        """Whether the kill switch currently halts execution."""
        return self._kill_switch.is_locked

    @property
    def lock_reason(self) -> str:
        """Kill-switch reason (empty when unlocked)."""
        return self._kill_switch.reason

    async def flatten_all(self) -> FlattenAllResponse:
        """Market-close every position, cancel orders and lock execution."""
        timestamp = self._clock()
        positions = self._broker.get_positions_sync()
        targets = [
            (root, position.quantity)
            for root, position in sorted(positions.items())
            if position.quantity != 0
        ]
        cancelled = [
            receipt
            for receipt in self._broker.cancel_all_sync()
            if receipt.status is OrderStatus.CANCELLED
        ]
        receipts = self._broker.flatten_all_sync(kind=self._trade_kind.FLATTEN)
        mapped: list[FlattenReceipt] = []
        for index, (root, quantity) in enumerate(targets):
            receipt = receipts[index] if index < len(receipts) else None
            mapped.append(
                FlattenReceipt(
                    order_id=receipt.order_id if receipt is not None else f"flatten-{root}",
                    instrument_id=root,
                    side="SELL" if quantity > 0 else "BUY",
                    quantity=abs(quantity),
                    status=_receipt_status(receipt.status) if receipt is not None else "UNKNOWN",  # type: ignore[arg-type]
                    average_price=receipt.average_price if receipt is not None else None,
                    message=receipt.message if receipt is not None else "no broker receipt",
                )
            )
        locked_until: datetime | None = None
        if mapped:
            locked_until = self._kill_switch.engage("emergency flatten_all")
        status: str
        if not targets:
            status = "no_positions"
        elif any(receipt.status == "REJECTED" for receipt in mapped):
            status = "halted"
        else:
            status = "flattened"
        return FlattenAllResponse(
            status=status,  # type: ignore[arg-type]
            timestamp=timestamp,
            flattened_positions=len(mapped),
            cancelled_orders=len(cancelled),
            receipts=mapped,
            execution_locked_until=locked_until,
            message=(
                "offline simulated broker: positions market-closed and execution locked"
                if mapped
                else "no open positions to flatten"
            ),
        )


class FixtureExecutionService:
    """Pure-arithmetic execution fallback (used only if the broker is absent)."""

    def __init__(
        self,
        *,
        market: fixtures.OfflineMarket,
        settings: Settings,
        clock: fixtures.Clock = fixtures.utc_now,
    ) -> None:
        self._market = market
        self._master = market.instrument_master
        self._settings = settings
        self._clock = clock
        self._kill_switch = _KillSwitchState(clock)

    def _positions(self) -> dict[InstrumentId, Position]:
        positions: dict[InstrumentId, Position] = {}
        as_of = self._market.as_of.date()
        for root, quantity in fixtures.fixture_book():
            if root not in self._market.prices or quantity == 0:
                continue
            spec = self._master.get(root)
            price = self._market.price(root)
            average = price * (1.0 - _ENTRY_OFFSETS.get(root, 0.0))
            roll_date, _ = fixtures.roll_window(spec, as_of)
            positions[root] = Position(
                instrument_id=root,
                quantity=quantity,
                average_price=average,
                market_price=price,
                multiplier=spec.multiplier,
                expiration=roll_date + timedelta(days=spec.roll.days_before_fnd),
                initial_margin=spec.margins.initial,
                maintenance_margin=spec.margins.maintenance,
                contract_symbol=f"{root}_{roll_date.year}{roll_date.month:02d}",
            )
        return positions

    def _snapshot(self) -> AccountSnapshot:
        positions = self._positions()
        unrealized = sum(position.unrealized_pnl for position in positions.values())
        net_liquidation = INITIAL_EQUITY + unrealized
        initial = sum(
            abs(position.quantity) * position.initial_margin for position in positions.values()
        )
        maintenance = sum(
            abs(position.quantity) * position.maintenance_margin for position in positions.values()
        )
        return AccountSnapshot(
            timestamp=self._market.as_of,
            net_liquidation=net_liquidation,
            cash_balance=INITIAL_EQUITY,
            buying_power=max(0.0, net_liquidation - initial),
            initial_margin_used=initial,
            maintenance_margin_used=maintenance,
            excess_liquidity=net_liquidation - initial,
        )

    async def account(self) -> AccountSnapshotResponse:
        """Deterministic arithmetic account snapshot."""
        snapshot = self._snapshot()
        curve = fixtures.equity_history(self._market, snapshot.net_liquidation, now=self._clock())
        day_pnl = curve[-1].value - curve[-2].value if len(curve) > 1 else 0.0
        return _account_response(
            snapshot=snapshot,
            positions=self._positions(),
            master=self._master,
            market=self._market,
            curve=curve,
            day_pnl=day_pnl,
        )

    async def positions(self) -> Mapping[InstrumentId, Position]:
        """Synthetic positions keyed by root."""
        return self._positions()

    @property
    def is_locked(self) -> bool:
        """Whether the kill switch currently halts execution."""
        return self._kill_switch.is_locked

    @property
    def lock_reason(self) -> str:
        """Kill-switch reason (empty when unlocked)."""
        return self._kill_switch.reason

    async def flatten_all(self) -> FlattenAllResponse:
        """Deterministically acknowledge a flatten without a broker."""
        timestamp = self._clock()
        positions = self._positions()
        receipts: list[FlattenReceipt] = []
        for root, position in sorted(positions.items()):
            receipts.append(
                FlattenReceipt(
                    order_id=f"fixture-flatten-{root}",
                    instrument_id=root,
                    side="SELL" if position.quantity > 0 else "BUY",
                    quantity=abs(position.quantity),
                    status="FILLED",
                    average_price=position.market_price,
                    message="arithmetic fixture fill (no broker module available)",
                )
            )
        locked_until = (
            self._kill_switch.engage("emergency flatten_all fixture") if receipts else None
        )
        return FlattenAllResponse(
            status="flattened" if receipts else "no_positions",
            timestamp=timestamp,
            flattened_positions=len(receipts),
            cancelled_orders=0,
            receipts=receipts,
            execution_locked_until=locked_until,
            message="fixture flatten path (SimulatedBroker unavailable)",
        )


class IBKRExecutionService:
    """Opt-in real IBKR wiring with a transparent offline fallback.

    Enabled with ``FUTURESRLX_API_ENABLE_IBKR=1``. The lifespan calls
    :meth:`connect`, which uses the production ``IBKRConnectionManager`` with a
    bounded timeout; while the session is unavailable every call falls through
    to the deterministic simulated broker, and ``/health`` reports
    ``offline_mode=true``.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        market: fixtures.OfflineMarket,
        clock: fixtures.Clock = fixtures.utc_now,
    ) -> None:
        from src.execution.ibkr_client import IBKRConnectionManager
        from src.execution.oms import OrderManagementSystem

        self._settings = settings
        self._market = market
        self._clock = clock
        self._fallback = SimulatedExecutionService(market=market, settings=settings, clock=clock)
        self._manager: IBKRConnectionManager = IBKRConnectionManager(
            settings, instrument_master=market.instrument_master, clock=clock
        )
        self._oms: OrderManagementSystem = OrderManagementSystem(
            manager=self._manager,
            settings=settings,
            instrument_master=market.instrument_master,
            clock=clock,
        )
        self._connect_error: str | None = None

    @property
    def connected(self) -> bool:
        """Whether the IBKR session is established."""
        return self._manager.is_connected

    @property
    def connection_error(self) -> str | None:
        """Last connection error (``None`` when connected)."""
        return self._connect_error

    async def connect(self) -> bool:
        """Attempt the IBKR connection within the configured timeout."""
        try:
            await asyncio.wait_for(
                self._manager.connect(), timeout=self._settings.ibkr_connect_timeout_seconds
            )
        except (TimeoutError, Exception) as exc:
            self._connect_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "IBKR session unavailable, using offline fallback: %s", self._connect_error
            )
            return False
        self._connect_error = None
        return self._manager.is_connected

    async def disconnect(self) -> None:
        """Close the IBKR session if one was established."""
        try:
            await self._manager.disconnect()
        except Exception:
            logger.exception("error while disconnecting IBKR session")

    async def account(self) -> AccountSnapshotResponse:
        """Live IBKR account when connected, offline simulated account otherwise."""
        if not self._manager.is_connected:
            return await self._fallback.account()
        snapshot = await self._manager.fetch_account()
        positions = await self._manager.fetch_positions()
        curve = fixtures.equity_history(self._market, snapshot.net_liquidation, now=self._clock())
        day_pnl = curve[-1].value - curve[-2].value if len(curve) > 1 else 0.0
        return _account_response(
            snapshot=snapshot,
            positions=positions,
            master=self._market.instrument_master,
            market=self._market,
            curve=curve,
            day_pnl=day_pnl,
        )

    async def positions(self) -> Mapping[InstrumentId, Position]:
        """Live IBKR positions when connected, simulated otherwise."""
        if not self._manager.is_connected:
            return await self._fallback.positions()
        return await self._manager.fetch_positions()

    @property
    def is_locked(self) -> bool:
        """Whether the OMS gate or the offline kill switch is latched."""
        return self._oms.is_locked or self._fallback.is_locked

    @property
    def lock_reason(self) -> str:
        """Kill-switch reason from the active adapter."""
        return self._oms.gate.lock_reason if self._oms.is_locked else self._fallback.lock_reason

    async def flatten_all(self) -> FlattenAllResponse:
        """Kill switch: live OMS flatten when connected, simulated otherwise."""
        if not self._manager.is_connected:
            return await self._fallback.flatten_all()
        timestamp = self._clock()
        positions = await self._manager.fetch_positions()
        targets = [
            (root, position.quantity)
            for root, position in sorted(positions.items())
            if position.quantity != 0
        ]
        receipts = await self._oms.flatten_all()
        mapped: list[FlattenReceipt] = []
        for index, (root, quantity) in enumerate(targets):
            receipt = receipts[index] if index < len(receipts) else None
            mapped.append(
                FlattenReceipt(
                    order_id=receipt.order_id if receipt is not None else f"flatten-{root}",
                    instrument_id=root,
                    side="SELL" if quantity > 0 else "BUY",
                    quantity=abs(quantity),
                    status=_receipt_status(receipt.status) if receipt is not None else "UNKNOWN",  # type: ignore[arg-type]
                    average_price=receipt.average_price if receipt is not None else None,
                    message=receipt.message if receipt is not None else "no IBKR receipt",
                )
            )
        return FlattenAllResponse(
            status="flattened" if mapped else "no_positions",
            timestamp=timestamp,
            flattened_positions=len(mapped),
            cancelled_orders=len(receipts) - len(mapped),
            receipts=mapped,
            execution_locked_until=None,
            message="IBKR OMS flatten_all executed; pre-trade gate locked",
        )


# ---------------------------------------------------------------------------
# Market data / pipeline service
# ---------------------------------------------------------------------------


def _build_translator(
    config: StrategyConfig, master: InstrumentsConfig
) -> ContractTranslatorProtocol:
    from src.risk.contract_translator import ContractTranslator

    return cast(
        ContractTranslatorProtocol,
        ContractTranslator.from_config(config.translator, instruments=master),
    )


def thresholds_from_settings(settings: Settings) -> RiskThresholds:
    """API threshold view of the central :class:`RiskSettings` contract."""
    risk = settings.risk
    return RiskThresholds(
        llm_confidence_threshold=risk.llm_confidence_threshold,
        llm_safe_score=risk.llm_safe_score,
        llm_critical_score=risk.llm_critical_score,
        max_margin_utilization=risk.max_margin_utilization,
        margin_warning_threshold=risk.margin_warning_threshold,
        margin_critical_threshold=risk.margin_critical_threshold,
        daily_max_loss_pct=risk.daily_max_loss_pct,
        target_annual_volatility=risk.target_annual_volatility,
        macro_lockout_minutes_before=risk.macro_lockout_minutes_before,
        macro_lockout_minutes_after=risk.macro_lockout_minutes_after,
    )


@lru_cache(maxsize=4)
def _macro_calendar(path_str: str) -> MacroEventCalendar | None:
    """Load and cache the configured Tier-1 macro calendar (``None`` on failure)."""
    try:
        from src.risk.llm_risk_overlay import MacroEventCalendar
    except ImportError:  # pragma: no cover - real overlay ships in src/risk
        return None
    try:
        return MacroEventCalendar.from_json(Path(path_str))
    except (FileNotFoundError, ValueError, TypeError) as exc:
        logger.warning("macro calendar unavailable at %s: %s", path_str, exc)
        return None


def build_llm_risk_overlay(
    settings: Settings,
    master: InstrumentsConfig,
    clock: fixtures.Clock,
    state: MarketState,
) -> LLMRiskOverlay | None:
    """Wire the **real** :class:`LLMRiskOverlay` with the offline LLM client.

    The deterministic :class:`~src.api.fixtures.OfflineLLMClient` implements the
    production ``AsyncLLMClient`` protocol (strict JSON completions, no network);
    the overlay's assessment cache is seeded from the same deterministic
    assessments so ``apply_risk`` exercises the real Module-1/Module-3 projection
    and macro-lockout calendar. Returns ``None`` only when the risk module is
    genuinely missing/uninitialisable (the pipeline diagnostics then report the
    stage as skipped).
    """
    try:
        from src.risk.llm_risk_overlay import LLMRiskOverlay
    except ImportError:
        logger.warning("src.risk.llm_risk_overlay is unavailable; LLM stage will be skipped")
        return None
    thresholds = thresholds_from_settings(settings)
    client = fixtures.OfflineLLMClient(state, thresholds, clock())
    calendar = _macro_calendar(str(settings.configs_dir / "macro_calendar.yaml"))
    try:
        overlay = LLMRiskOverlay(
            client,
            settings.risk,
            calendar=calendar,
            instruments=master,
        )
    except Exception as exc:
        logger.exception("LLM risk overlay failed to initialise: %s", exc)
        return None
    overlay.set_cached_assessments(fixtures.risk_assessment_core(state, thresholds, clock()))
    return overlay


def build_pipeline_parts(
    config: StrategyConfig,
    master: InstrumentsConfig,
    settings: Settings,
    *,
    clock: fixtures.Clock = fixtures.utc_now,
    state: MarketState | None = None,
) -> tuple[StrategyPipeline, ContractTranslatorProtocol]:
    """Build the real FinRL-X pipeline + contract translator for ``config``.

    Risk overlays are injected by declarative name (``quant_risk`` /
    ``llm_risk``). When a market ``state`` is supplied the real LLM Risk-First
    overlay is wired with the deterministic offline client and a seeded
    assessment cache; without it (or when the risk module is missing) the LLM
    stage is reported as skipped inside the pipeline diagnostics.
    """
    from src.strategies import build_pipeline

    translator = _build_translator(config, master)
    overlays: list[RiskOverlay] = []
    try:
        from src.risk.quant_risk_overlay import QuantRiskOverlay

        overlays.append(
            _NamedRiskOverlay(
                QuantRiskOverlay(settings.risk, instruments=master, clock=clock),
                "quant_risk",
            )
        )
    except Exception as exc:
        logger.warning("quant risk overlay unavailable: %s", exc)
    if state is not None:
        llm = build_llm_risk_overlay(settings, master, clock, state)
        if llm is not None:
            overlays.append(_NamedRiskOverlay(llm, "llm_risk"))
    return build_pipeline(config, master, translator, risk_overlays=overlays), translator


class EngineMarketService:
    """Real FinRL-X pipeline executed against the deterministic offline market."""

    def __init__(
        self,
        *,
        market: fixtures.OfflineMarket,
        settings: Settings,
        catalog: StrategyCatalog,
        clock: fixtures.Clock = fixtures.utc_now,
    ) -> None:
        self._market = market
        self._master = market.instrument_master
        self._settings = settings
        self._catalog = catalog
        self._clock = clock
        self._state_cache: dict[date, MarketState] = {}
        self._pipeline_cache: dict[date, PipelineWaterfallResponse] = {}
        self._term_cache: dict[date, TermStructureResponse] = {}
        self._warnings: list[str] = []

    @property
    def warnings(self) -> tuple[str, ...]:
        """Non-fatal wiring warnings recorded while building the view."""
        return tuple(self._warnings)

    def latest_state(self) -> MarketState:
        """Point-in-time ``MarketState`` for the latest offline bar."""
        key = self._market.as_of.date()
        cached = self._state_cache.get(key)
        if cached is not None:
            return cached
        from src.data.market_state_builder import MarketStateBuilder

        builder = MarketStateBuilder(instrument_master=self._master, initial_equity=INITIAL_EQUITY)
        state = builder.build(self._market.bars, self._market.as_of, equity=INITIAL_EQUITY)
        self._state_cache[key] = state
        return state

    def _default_config(self) -> StrategyConfig | None:
        config = self._catalog.core_config(DEFAULT_STRATEGY_ID)
        if config is not None:
            return config
        for strategy_id in self._catalog.available_ids():
            config = self._catalog.core_config(strategy_id)
            if config is not None:
                return config
        return None

    def _pipeline_for(self, config: StrategyConfig, state: MarketState) -> StrategyPipeline:
        pipeline, _ = build_pipeline_parts(
            config,
            self._master,
            self._settings,
            clock=self._clock,
            state=state,
        )
        return pipeline

    def pipeline(self) -> PipelineWaterfallResponse:
        """Run the real FinRL-X pipeline on the latest offline state."""
        key = self._market.as_of.date()
        cached = self._pipeline_cache.get(key)
        if cached is not None:
            return cached
        config = self._default_config()
        if config is None:
            raise RuntimeError(
                "no declarative strategy config available under configs/strategies; "
                "the pipeline inspector requires at least one YAML config"
            )
        state = self.latest_state()
        pipeline = self._pipeline_for(config, state)
        result = pipeline.run(state)
        response = self._map_pipeline(config, state, result)
        self._pipeline_cache[key] = response
        return response

    def _map_pipeline(
        self,
        config: StrategyConfig,
        state: MarketState,
        result: PipelineResult,
    ) -> PipelineWaterfallResponse:
        targets = result.targets if result.targets is not None else ContractTargetSet.empty()
        target_rows = targets.by_instrument
        selected = set(result.universe)
        equity = state.equity or INITIAL_EQUITY
        instruments: list[PipelineInstrumentRow] = []
        for root in sorted(state.instruments):
            base = result.base_weights.get(root, 0.0)
            timing = result.timing_weights.get(root, 0.0)
            risk = result.risk_weights.get(root, 0.0)
            final = result.final_weights.get(root, 0.0)
            target = target_rows.get(root)
            instruments.append(
                PipelineInstrumentRow(
                    instrument_id=root,
                    asset_class=state.instruments[root].asset_class,
                    selected=root in selected,
                    weight_base=round(base, 8),
                    weight_timing=round(timing, 8),
                    weight_risk=round(risk, 8),
                    weight_final=round(final, 8),
                    delta_timing=round(timing - base, 8),
                    delta_risk=round(risk - timing, 8),
                    target_contracts=target.quantity if target is not None else None,
                    notional=round(final * equity, 2),
                    reason=_adjustment_reason(
                        root in selected, base, timing, risk, target.quantity if target else 0
                    ),
                )
            )
        stages = [
            PipelineStage(
                key="universe",
                label="S_t · Universe",
                formula="S_t(X≤t) → roots",
                gross=0.0,
                net=0.0,
                instrument_count=len(result.universe),
                note=f"selector: {config.universe.name}",
            ),
            PipelineStage(
                key="base",
                label="w_base · Allocation",
                formula=f"A_t(candidates) · {config.allocator.name}",
                gross=round(result.base_weights.gross, 8),
                net=round(result.base_weights.net, 8),
                instrument_count=len(result.base_weights),
                note=f"allocator: {config.allocator.name}",
            ),
            PipelineStage(
                key="timing",
                label="w_timing · Overlay",
                formula="T_t(w_base)",
                gross=round(result.timing_weights.gross, 8),
                net=round(result.timing_weights.net, 8),
                instrument_count=len(result.timing_weights),
                note="timing: "
                + (", ".join(stage.name for stage in config.timing) if config.timing else "none"),
            ),
            PipelineStage(
                key="risk",
                label="w_risk · Approved",
                formula="R_t(w_timing) → w_t",
                gross=round(result.risk_weights.gross, 8),
                net=round(result.risk_weights.net, 8),
                instrument_count=len(result.risk_weights),
                note="risk: "
                + (", ".join(stage.name for stage in config.risk) if config.risk else "none"),
            ),
            PipelineStage(
                key="targets",
                label="N_i,t · Contracts",
                formula="N = w_t·E_t / (P·M)",
                gross=round(result.final_weights.gross, 8),
                net=round(result.final_weights.net, 8),
                instrument_count=sum(abs(target.quantity) for target in targets),
                note=(
                    f"translator: {config.translator.rounding_mode.value} rounding, "
                    f"buffer {config.translator.rebalance_buffer_contracts:g}"
                ),
            ),
        ]
        diagnostics = self._diagnostics(config, state, result, targets, equity)
        elapsed = result.elapsed_ms
        if elapsed is None:
            timings = _metadata_mapping(result.diagnostics, "stage_timings_ms")
            elapsed = sum(
                float(value) for value in timings.values() if isinstance(value, int | float)
            )
        return PipelineWaterfallResponse(
            timestamp=state.timestamp,
            cycle_id=f"rc-{state.timestamp:%Y%m%d}-001",
            universe=list(result.universe),
            stages=stages,
            instruments=instruments,
            diagnostics=diagnostics,
            elapsed_ms=round(float(elapsed or 0.0), 3),
            data_source="engine",
        )

    def _diagnostics(
        self,
        config: StrategyConfig,
        state: MarketState,
        result: PipelineResult,
        targets: ContractTargetSet,
        equity: float,
    ) -> list[PipelineDiagnostic]:
        universe_size = len(result.universe)
        eligible = len(state.instruments)
        applied_risk = _metadata_names(result.diagnostics, "risk_overlays")
        skipped_risk = _metadata_names(result.diagnostics, "skipped_risk_stages")
        diagnostics: list[PipelineDiagnostic] = [
            PipelineDiagnostic(
                stage="Universe selector",
                detail=(
                    f"{config.universe.name} selected {universe_size}/{eligible} roots from the "
                    "point-in-time liquidity/OI/regime filter"
                ),
                value=f"{universe_size}/{eligible}",
                status="ok" if universe_size > 0 else "warn",
            ),
            PipelineDiagnostic(
                stage="Portfolio allocator",
                detail=f"{config.allocator.name} produced gross {result.base_weights.gross:.3f}x",
                value=f"{result.base_weights.gross:.3f}x",
                status="ok",
            ),
            PipelineDiagnostic(
                stage="Timing overlay",
                detail=", ".join(stage.name for stage in config.timing) or "no timing overlays",
                value=f"{result.timing_weights.gross:.3f}x",
                status="ok",
            ),
        ]
        if applied_risk:
            diagnostics.append(
                PipelineDiagnostic(
                    stage="Risk overlays",
                    detail="applied: " + ", ".join(applied_risk),
                    value=f"{len(applied_risk)} applied",
                    status="ok",
                )
            )
        else:
            diagnostics.append(
                PipelineDiagnostic(
                    stage="Risk overlays",
                    detail="no injected overlay matched the declared risk stages",
                    value="0 applied",
                    status="warn",
                )
            )
        if skipped_risk:
            diagnostics.append(
                PipelineDiagnostic(
                    stage="Risk stage skipped",
                    detail=(
                        "declared in YAML but not importable yet: "
                        + ", ".join(skipped_risk)
                        + " (pipelines run without that constraint)"
                    ),
                    value=f"{len(skipped_risk)} skipped",
                    status="warn",
                )
            )
        projected_margin = targets.total_initial_margin
        utilization = projected_margin / equity if equity > 0 else 0.0
        eta = config.translator.max_margin_utilization
        diagnostics.append(
            PipelineDiagnostic(
                stage="Margin feasibility",
                detail=(
                    f"projected IM {projected_margin:,.0f} / equity {equity:,.0f} = "
                    f"{utilization:.1%} vs η_max {eta:.1%}"
                ),
                value=f"{utilization:.1%}",
                status="critical"
                if utilization > eta
                else "warn"
                if utilization > eta * 0.8
                else "ok",
            )
        )
        diagnostics.append(
            PipelineDiagnostic(
                stage="Contract translator",
                detail=(
                    f"{config.translator.rounding_mode.value} rounding, buffer "
                    f"{config.translator.rebalance_buffer_contracts:g} contracts, micro routing "
                    f"{'on' if config.translator.enable_micro_routing else 'off'}"
                ),
                value=f"{sum(abs(target.quantity) for target in targets)} lots",
                status="ok",
            )
        )
        timings = _metadata_mapping(result.diagnostics, "stage_timings_ms")
        elapsed = result.elapsed_ms
        if elapsed is None:
            elapsed = sum(
                float(value) for value in timings.values() if isinstance(value, int | float)
            )
        note = _metadata_mapping(result.diagnostics, "stage_notes").get("sanitised_weights")
        diagnostics.append(
            PipelineDiagnostic(
                stage="Pipeline trace",
                detail=(
                    f"stages={len(timings)} stage timings summed to {float(elapsed):.2f} ms"
                    + (f"; sanitised={note}" if note is not None else "")
                ),
                value=f"{len(applied_risk) + len(skipped_risk)} R_t",
                status="info",
            )
        )
        return diagnostics

    def term_structure(self) -> TermStructureResponse:
        """Deterministic forward curves + roll calendar (real instrument master)."""
        key = self._market.as_of.date()
        cached = self._term_cache.get(key)
        if cached is None:
            cached = fixtures.term_structure(self._market)
            self._term_cache[key] = cached
        return cached


class FixtureMarketService:
    """Deterministic pipeline fallback used when the strategy/data layers fail.

    Every stage of the waterfall is synthesised from the deterministic fixture
    inputs and the response is marked ``data_source="fixture"``; the diagnostics
    record why the real engine could not be used.
    """

    def __init__(
        self,
        *,
        market: fixtures.OfflineMarket,
        settings: Settings,
        reason: str,
        clock: fixtures.Clock = fixtures.utc_now,
    ) -> None:
        self._market = market
        self._master = market.instrument_master
        self._settings = settings
        self._reason = reason
        self._clock = clock
        self._state: MarketState | None = None

    def latest_state(self) -> MarketState:
        """Lightweight state from the deterministic fixtures."""
        if self._state is None:
            from src.core.types import InstrumentSnapshot

            snapshots: dict[str, InstrumentSnapshot] = {}
            for root in self._market.roots:
                spec = self._master.get(root)
                price = self._market.price(root)
                snapshots[root] = InstrumentSnapshot(
                    instrument_id=root,
                    root_symbol=root,
                    price=price,
                    asset_class=spec.asset_class,
                    expiration=fixtures.roll_window(spec, self._market.as_of.date())[0]
                    + timedelta(days=spec.roll.days_before_fnd),
                    volume=3_000_000.0,
                    open_interest=2_000_000.0,
                )
            self._state = MarketState(
                timestamp=self._market.as_of,
                instruments=snapshots,
                equity=INITIAL_EQUITY,
            )
        return self._state

    def pipeline(self) -> PipelineWaterfallResponse:
        """Deterministic synthetic waterfall (explicitly marked as fixture)."""
        state = self.latest_state()
        roots = sorted(state.instruments)
        selected = tuple(roots[:5])
        instruments: list[PipelineInstrumentRow] = []
        for index, root in enumerate(roots):
            base = 0.3 if root in selected else 0.0
            timing = base * (1.0 + 0.08 * math.sin(index + 1.0))
            risk = timing * 0.92
            quantity = (
                round(risk * INITIAL_EQUITY / (state.price(root) * 50.0))
                if root in selected
                else None
            )
            instruments.append(
                PipelineInstrumentRow(
                    instrument_id=root,
                    asset_class=state.instruments[root].asset_class,
                    selected=root in selected,
                    weight_base=base,
                    weight_timing=timing,
                    weight_risk=risk,
                    weight_final=risk,
                    delta_timing=timing - base,
                    delta_risk=risk - timing,
                    target_contracts=quantity,
                    notional=risk * INITIAL_EQUITY,
                    reason="fixture fallback: real pipeline layer unavailable",
                )
            )
        gross = sum(abs(row.weight_final) for row in instruments)
        stages = [
            PipelineStage(
                key="universe",
                label="S_t · Universe",
                formula="S_t(X≤t) → roots",
                gross=0.0,
                net=0.0,
                instrument_count=len(selected),
                note="fixture fallback",
            ),
            PipelineStage(
                key="base",
                label="w_base · Allocation",
                formula="A_t(candidates)",
                gross=round(sum(abs(row.weight_base) for row in instruments), 8),
                net=round(sum(row.weight_base for row in instruments), 8),
                instrument_count=len(selected),
                note="fixture fallback",
            ),
            PipelineStage(
                key="timing",
                label="w_timing · Overlay",
                formula="T_t(w_base)",
                gross=round(sum(abs(row.weight_timing) for row in instruments), 8),
                net=round(sum(row.weight_timing for row in instruments), 8),
                instrument_count=len(selected),
                note="fixture fallback",
            ),
            PipelineStage(
                key="risk",
                label="w_risk · Approved",
                formula="R_t(w_timing) → w_t",
                gross=round(gross, 8),
                net=round(sum(row.weight_risk for row in instruments), 8),
                instrument_count=len(selected),
                note="fixture fallback",
            ),
            PipelineStage(
                key="targets",
                label="N_i,t · Contracts",
                formula="N = w_t·E_t / (P·M)",
                gross=round(gross, 8),
                net=round(sum(row.weight_final for row in instruments), 8),
                instrument_count=sum(abs(row.target_contracts or 0) for row in instruments),
                note="fixture fallback",
            ),
        ]
        return PipelineWaterfallResponse(
            timestamp=state.timestamp,
            cycle_id=f"rc-{state.timestamp:%Y%m%d}-fx",
            universe=list(selected),
            stages=stages,
            instruments=instruments,
            diagnostics=[
                PipelineDiagnostic(
                    stage="Pipeline wiring",
                    detail=f"real FinRL-X pipeline unavailable — {self._reason}",
                    value="fixture",
                    status="warn",
                )
            ],
            elapsed_ms=0.0,
            data_source="fixture",
        )

    def term_structure(self) -> TermStructureResponse:
        """Forward curves + roll calendar from the deterministic fixtures."""
        return fixtures.term_structure(self._market)


def _adjustment_reason(
    selected: bool, base: float, timing: float, risk: float, quantity: int
) -> str:
    if not selected:
        return "filtered out by the universe selector"
    if timing != base:
        return "scaled by the timing/term-structure overlay"
    if risk != timing:
        return "projected by the risk overlays (vol target / LLM breaker)"
    if quantity == 0:
        return "rounded flat by the contract translator (below buffer)"
    return "passed through unchanged"


# ---------------------------------------------------------------------------
# LLM Risk-First service
# ---------------------------------------------------------------------------


class EngineRiskService:
    """Deterministic Risk-First state on top of the real risk settings."""

    def __init__(
        self,
        *,
        settings: Settings,
        market: MarketDataService,
        execution: ExecutionService,
        clock: fixtures.Clock = fixtures.utc_now,
    ) -> None:
        self._settings = settings
        self._market = market
        self._execution = execution
        self._clock = clock
        self._thresholds = thresholds_from_settings(settings)

    def thresholds(self) -> RiskThresholds:
        """Effective thresholds (settings + API overrides)."""
        return self._thresholds

    def update_thresholds(self, override: RiskThresholdOverride) -> RiskThresholds:
        """Validate and apply a partial override (raises ``ValueError``)."""
        changes = override.model_dump(exclude_none=True)
        merged = self._thresholds.model_copy(update=changes)
        if merged.llm_critical_score <= merged.llm_safe_score:
            raise ValueError("llm_critical_score must exceed llm_safe_score")
        if merged.margin_critical_threshold <= merged.margin_warning_threshold:
            raise ValueError("margin_critical_threshold must exceed margin_warning_threshold")
        self._thresholds = merged
        logger.warning("risk thresholds overridden: %s", sorted(changes))
        return self._thresholds

    def state(self) -> RiskStateResponse:
        """Latest assessments, breaker state, macro lockout and news feed."""
        now = self._clock()
        thresholds = self._thresholds
        state = self._market.latest_state()
        # Prefer the real LLM Risk-First overlay's assessment cache (seeded from the
        # deterministic offline client); fall back to the raw fixture assessments
        # only when the risk layer is genuinely unavailable.
        overlay = build_llm_risk_overlay(
            self._settings, self._settings.instruments, self._clock, state
        )
        if overlay is not None:
            overlay.set_cached_assessments(fixtures.risk_assessment_core(state, thresholds, now))
            core = tuple(
                overlay.latest_assessments[root] for root in sorted(overlay.latest_assessments)
            )
        else:
            core = fixtures.risk_assessment_core(state, thresholds, now)
        assessments = fixtures.risk_assessment_rows(core, thresholds, now)
        events = fixtures.macro_calendar(now)
        lockout = fixtures.macro_lockout(events, thresholds, now)
        news = fixtures.news_feed(assessments, now)
        liquidated = [row for row in assessments if row.action == "liquidate"]
        coerced = [row for row in assessments if row.action == "neutralized"]
        if liquidated:
            reasons = ", ".join(
                f"{row.instrument_id} S={row.semantic_risk_score:.2f}/c={row.confidence:.2f}"
                for row in liquidated
            )
            breaker_active = True
            breaker_reason = f"LLM circuit breaker forced flat: {reasons}"
        elif self._execution.is_locked:
            breaker_active = True
            breaker_reason = f"execution halted by kill switch: {self._execution.lock_reason}"
        else:
            breaker_active = False
            breaker_reason = (
                f"no root above κ={thresholds.llm_critical_score:.1f} with "
                f"c ≥ τ={thresholds.llm_confidence_threshold:.2f}"
            )
        return RiskStateResponse(
            timestamp=now,
            thresholds=thresholds,
            circuit_breaker_active=breaker_active,
            circuit_breaker_reason=breaker_reason,
            macro_lockout=lockout,
            assessments=assessments,
            macro_events=events,
            news=news,
            neutralized_instruments=[row.instrument_id for row in coerced],
        )


# ---------------------------------------------------------------------------
# Container / dependency wiring
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Container:
    """Process-wide service container injected into every route."""

    settings: Settings
    clock: fixtures.Clock
    started_at: datetime
    instrument_master: InstrumentsConfig
    market: MarketDataService
    strategies: StrategyCatalog
    backtests: BacktestService
    execution: ExecutionService
    risk: RiskService
    telemetry: TelemetryBroadcaster
    offline_mode: bool
    warnings: list[str] = field(default_factory=list)

    def health(self) -> HealthResponse:
        """Liveness payload (``degraded`` when any fallback is active)."""
        now = self.clock()
        return HealthResponse(
            status="degraded" if self.warnings or self.offline_mode else "ok",
            app_name=self.settings.app_name,
            version="1.0.0",
            environment=self.settings.environment,
            offline_mode=self.offline_mode,
            uptime_seconds=max(0.0, (now - self.started_at).total_seconds()),
            server_time=now,
        )

    def telemetry_status(self) -> TelemetryStatusResponse:
        """WebSocket broadcaster status."""
        return self.telemetry.status()

    async def start(self) -> None:
        """Start the telemetry publisher and any live execution session."""
        connector = getattr(self.execution, "connect", None)
        if callable(connector):
            outcome = connector()
            connected = await outcome if inspect.isawaitable(outcome) else bool(outcome)
            if not connected:
                self.warnings.append("IBKR session unavailable — offline execution fallback active")
                self.offline_mode = True
        await self.telemetry.start()

    async def stop(self) -> None:
        """Stop telemetry and disconnect live sessions."""
        await self.telemetry.stop()
        disconnector = getattr(self.execution, "disconnect", None)
        if callable(disconnector):
            outcome = disconnector()
            if inspect.isawaitable(outcome):
                await outcome


def _build_execution_service(
    *,
    settings: Settings,
    market: fixtures.OfflineMarket,
    clock: fixtures.Clock,
    warnings: list[str],
) -> ExecutionService:
    if os.environ.get(IBKR_ENABLE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}:
        try:
            return IBKRExecutionService(settings=settings, market=market, clock=clock)
        except Exception as exc:
            warnings.append(f"IBKR execution wiring failed: {type(exc).__name__}: {exc}")
            logger.exception("IBKR execution wiring failed; using simulated broker")
    try:
        return SimulatedExecutionService(market=market, settings=settings, clock=clock)
    except Exception as exc:
        warnings.append(f"SimulatedBroker unavailable: {type(exc).__name__}: {exc}")
        logger.exception("SimulatedBroker unavailable; using arithmetic fixture execution")
        return FixtureExecutionService(market=market, settings=settings, clock=clock)


def _build_market_service(
    *,
    settings: Settings,
    market: fixtures.OfflineMarket,
    catalog: StrategyCatalog,
    clock: fixtures.Clock,
    warnings: list[str],
) -> MarketDataService:
    try:
        return EngineMarketService(market=market, settings=settings, catalog=catalog, clock=clock)
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        warnings.append(f"FinRL-X pipeline wiring failed: {reason}")
        logger.exception("FinRL-X pipeline wiring failed; using fixture pipeline")
        return FixtureMarketService(market=market, settings=settings, reason=reason, clock=clock)


def build_container(
    settings: Settings | None = None,
    *,
    clock: fixtures.Clock | None = None,
) -> Container:
    """Build the service container (real adapters with deterministic fallbacks)."""
    resolved_settings = settings if settings is not None else Settings()
    resolved_clock = clock if clock is not None else fixtures.utc_now
    warnings: list[str] = []
    master = resolved_settings.instruments
    market = fixtures.build_offline_market(master, clock=resolved_clock)
    catalog = InMemoryStrategyCatalog(settings=resolved_settings, clock=resolved_clock)
    execution = _build_execution_service(
        settings=resolved_settings, market=market, clock=resolved_clock, warnings=warnings
    )
    market_service = _build_market_service(
        settings=resolved_settings,
        market=market,
        catalog=catalog,
        clock=resolved_clock,
        warnings=warnings,
    )
    risk = EngineRiskService(
        settings=resolved_settings,
        market=market_service,
        execution=execution,
        clock=resolved_clock,
    )
    from src.api.backtests_service import EngineBacktestService

    backtests = EngineBacktestService(
        settings=resolved_settings,
        market=market,
        catalog=catalog,
        clock=resolved_clock,
    )
    telemetry = TelemetryBroadcaster(
        settings=resolved_settings,
        clock=resolved_clock,
        execution=execution,
        risk=risk,
        market=market_service,
    )
    offline_mode = not isinstance(execution, IBKRExecutionService)
    return Container(
        settings=resolved_settings,
        clock=resolved_clock,
        started_at=resolved_clock(),
        instrument_master=master,
        market=market_service,
        strategies=catalog,
        backtests=backtests,
        execution=execution,
        risk=risk,
        telemetry=telemetry,
        offline_mode=offline_mode,
        warnings=warnings,
    )
