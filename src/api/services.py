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
  the lifespan, keeps a background liveness/reconnect probe running and serves the
  last-known-good IBKR book (marked ``stale``) while TWS is unavailable -- it never
  silently substitutes simulated positions.
* :class:`EngineRiskService` — real ``RiskSettings`` + deterministic Risk-First
  assessment/projection (same Module-3 formulas as the LLM overlay).
* Backtests — see :mod:`src.api.backtests_service` (real ``BacktestEngine``).
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import math
import os
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

import polars as pl
from configs.instruments import InstrumentsConfig
from configs.settings import Settings
from configs.strategy import StrategyConfig, discover_strategy_configs, parse_strategy_config

from src.api import fixtures
from src.api.live import (
    BRIDGE_FILE_NAME,
    EQUITY_HISTORY_FILE_NAME,
    LiveEquityHistoryStore,
    LiveStateService,
    micro_routing_from_payload,
)
from src.api.news import NewsFeedService
from src.api.protocols import (
    BacktestService,
    ExecutionService,
    MarketDataService,
    RiskService,
    StrategyCatalog,
)
from src.api.schemas import (
    PAPER_STATE_RESET_CONFIRM,
    RISK_LOCK_RESET_CONFIRM,
    AccountSnapshotResponse,
    EquityPoint,
    FlattenAllResponse,
    FlattenReceipt,
    HealthResponse,
    MicroRoutingEntry,
    PaperStateResetReceipts,
    PaperStateResetRequest,
    PaperStateResetResponse,
    PipelineDiagnostic,
    PipelineInstrumentRow,
    PipelineStage,
    PipelineWaterfallResponse,
    PositionRow,
    RiskLockResetRequest,
    RiskLockResetResponse,
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
from src.api.version import API_VERSION
from src.core.interfaces import ContractTranslatorProtocol, RiskOverlay, StrategyPipeline
from src.core.types import (
    AccountSnapshot,
    ContractTargetSet,
    InstrumentId,
    MarketState,
    OrderEvent,
    OrderReceipt,
    OrderSide,
    OrderStatus,
    PipelineResult,
    Position,
    WeightVector,
)
from src.risk.state_store import PersistedRiskState, RiskStateStore, build_risk_state_store

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
    "IBKRBookUnavailableError",
    "IBKRExecutionService",
    "InMemoryStrategyCatalog",
    "LiveEquityHistoryStore",
    "OperatorActionUnavailableError",
    "OperatorConfirmationError",
    "OperatorResetError",
    "OperatorTokenError",
    "SimulatedExecutionService",
    "build_container",
]

#: Env flag that opts into real IBKR wiring instead of the simulated broker.
IBKR_ENABLE_ENV: str = "FUTURESRLX_API_ENABLE_IBKR"

#: How long the kill switch keeps execution locked after a flatten.
KILL_SWITCH_LOCK_MINUTES: int = 30

#: Env var holding the optional operator reset token for ``POST /api/risk/reset-lock``.
OPERATOR_RESET_TOKEN_ENV: str = "FUTURESRLX_OPERATOR_RESET_TOKEN"

#: Environments where the destructive operator paper-state reset is allowed.
PAPER_STATE_RESET_ENVIRONMENTS: frozenset[str] = frozenset({"development", "paper", "backtest"})

#: ``PersistedRiskState.extra`` keys owned by the quant risk overlay (cleared on reset).
QUANT_OVERLAY_EXTRA_PREFIX: str = "quant_overlay."


class OperatorResetError(RuntimeError):
    """Base class for guarded operator reset failures (mapped to 4xx by routes)."""


class OperatorConfirmationError(OperatorResetError, ValueError):
    """The request body confirmation phrase is missing or wrong (HTTP 400)."""


class OperatorTokenError(OperatorResetError, PermissionError):
    """The required ``FUTURESRLX_OPERATOR_RESET_TOKEN`` is missing/wrong (HTTP 403)."""


class OperatorActionUnavailableError(OperatorResetError):
    """The reset cannot run now (active flatten / live environment / unsupported)."""


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
        self._lock = threading.RLock()
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
        with self._lock:
            return [self._view(self._configs[key]) for key in sorted(self._configs)]

    def get(self, strategy_id: str) -> StrategyConfigView | None:
        """One strategy view or ``None``."""
        with self._lock:
            config = self._configs.get(strategy_id.strip().lower())
            return self._view(config) if config is not None else None

    def update(self, strategy_id: str, update: StrategyConfigUpdate) -> StrategyConfigView | None:
        """Merge, validate and store an edited config (in memory)."""
        key = strategy_id.strip().lower()
        with self._lock:
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

    def upsert(self, config: StrategyConfig) -> StrategyConfigView:
        """Register a validated config in memory so it is immediately runnable.

        This is the runtime bridge used by the Strategy Studio / optimizer apply
        flow: a strategy created or tuned through the API becomes available to the
        backtest service without a process restart. It never touches disk (the
        committed YAML stays immutable; Studio writes are handled elsewhere).
        """
        if not isinstance(config, StrategyConfig):
            raise TypeError(f"config must be a StrategyConfig, got {type(config)!r}")
        with self._lock:
            self._configs[config.id] = config
            self._updated_at[config.id] = self._clock()
            return self._view(config)

    def remove(self, strategy_id: str) -> bool:
        """Remove a strategy from the runtime catalog (delete/rename QoL flows).

        Additive runtime hook: drops the in-memory entry only; on-disk deletion is
        owned by the Strategy Studio service. Returns whether an entry was removed.
        """
        key = strategy_id.strip().lower()
        with self._lock:
            removed = self._configs.pop(key, None) is not None
            self._updated_at.pop(key, None)
            return removed

    def reload(self) -> tuple[str, ...]:
        """Re-discover the committed strategy directory (additive refresh hook)."""
        directory = self._settings.strategies_dir
        discovered = discover_strategy_configs(directory) if directory.is_dir() else {}
        with self._lock:
            for strategy_id, config in discovered.items():
                self._configs[strategy_id] = config
                self._updated_at[strategy_id] = datetime.fromtimestamp(
                    (directory / f"{strategy_id}.yaml").stat().st_mtime, tz=UTC
                )
            return tuple(sorted(self._configs))

    def core_config(self, strategy_id: str) -> StrategyConfig | None:
        """Validated core config used by the backtest engine."""
        with self._lock:
            return self._configs.get(strategy_id.strip().lower())

    def available_ids(self) -> list[str]:
        """All strategy ids, sorted."""
        with self._lock:
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
        self._flatten_in_progress = False
        self._seed_book()

    @property
    def source_name(self) -> str:
        """Provenance reported by the API for this execution adapter."""
        return "simulated"

    def working_orders_snapshot(self) -> list[dict[str, object]]:
        """Working orders as plain rows for ``GET /api/orders/working``."""
        as_of = self._market.as_of
        return [
            {
                "order_id": order.order_id,
                "instrument_id": order.instrument_id,
                "side": order.side.value,
                "quantity": order.quantity,
                "filled_quantity": 0,
                "remaining_quantity": order.quantity,
                "status": "SUBMITTED",
                "order_type": order.order_type.value,
                "limit_price": order.limit_price,
                "updated_at": as_of,
            }
            for order in self._broker.working_orders
        ]

    def recent_fills_snapshot(self, limit: int = 50) -> list[dict[str, object]]:
        """Most recent simulated fills as plain rows for ``GET /api/fills/recent``."""
        fills = list(self._broker.fills)[-max(1, limit) :]
        return [
            {
                "order_id": fill.order_id,
                "instrument_id": fill.instrument_id,
                "side": fill.side.value,
                "quantity": fill.quantity,
                "price": fill.price,
                "commission": fill.cost.commission,
                "timestamp": fill.timestamp,
            }
            for fill in fills
        ]

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

    @property
    def lock_until(self) -> datetime | None:
        """Kill-switch expiry timestamp (``None`` when never engaged)."""
        return self._kill_switch.locked_until

    @property
    def flatten_in_progress(self) -> bool:
        """Whether an emergency flatten is currently executing."""
        return self._flatten_in_progress

    def reset_kill_switch(self) -> None:
        """Release the kill switch (operator ``POST /api/risk/reset-lock``)."""
        self._kill_switch.release()

    def reset_paper_state(self) -> None:
        """Rebuild the deterministic simulated paper book and release the latch."""
        from src.backtest.simulated_broker import SimulatedBroker

        self._broker = SimulatedBroker(
            self._master,
            initial_cash=INITIAL_EQUITY,
            timestamp=self._market.as_of,
            name="simulated-offline",
        )
        self._broker.set_market(self._market.as_of, _quotes_for_market(self._market))
        self._broker._connected = True
        self._kill_switch = _KillSwitchState(self._clock)
        self._flatten_in_progress = False
        self._seed_book()

    async def submit_orders(self, orders: Sequence[OrderEvent]) -> list[OrderReceipt]:
        """Submit delta orders directly to the simulated matching engine."""
        return [self._broker.place_order(order) for order in orders]

    async def flatten_all(self) -> FlattenAllResponse:
        """Market-close every position, cancel orders and lock execution."""
        self._flatten_in_progress = True
        try:
            return await self._flatten_all_impl()
        finally:
            self._flatten_in_progress = False

    async def _flatten_all_impl(self) -> FlattenAllResponse:
        """Implementation of :meth:`flatten_all` (tracked by ``flatten_in_progress``)."""
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
        self._flatten_in_progress = False

    @property
    def source_name(self) -> str:
        """Provenance reported by the API for this execution adapter."""
        return "fixture"

    def working_orders_snapshot(self) -> list[dict[str, object]]:
        """A pure fixture adapter never carries working orders."""
        return []

    def recent_fills_snapshot(self, limit: int = 50) -> list[dict[str, object]]:
        """A pure fixture adapter never carries real fills."""
        del limit
        return []

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

    @property
    def lock_until(self) -> datetime | None:
        """Kill-switch expiry timestamp (``None`` when never engaged)."""
        return self._kill_switch.locked_until

    @property
    def flatten_in_progress(self) -> bool:
        """Whether an emergency flatten is currently executing."""
        return self._flatten_in_progress

    def reset_kill_switch(self) -> None:
        """Release the kill switch (operator ``POST /api/risk/reset-lock``)."""
        self._kill_switch.release()

    def reset_paper_state(self) -> None:
        """Reset the arithmetic fixture paper latch (positions are stateless)."""
        self._kill_switch = _KillSwitchState(self._clock)

    async def submit_orders(self, orders: Sequence[OrderEvent]) -> list[OrderReceipt]:
        """Refuse execution honestly: the fixture adapter has no matching engine."""
        del orders
        raise RuntimeError(
            "fixture execution adapter cannot submit orders; deploy the simulated broker"
        )

    async def flatten_all(self) -> FlattenAllResponse:
        """Deterministically acknowledge a flatten without a broker."""
        self._flatten_in_progress = True
        try:
            return await self._flatten_all_impl()
        finally:
            self._flatten_in_progress = False

    async def _flatten_all_impl(self) -> FlattenAllResponse:
        """Implementation of :meth:`flatten_all` (tracked by ``flatten_in_progress``)."""
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


class IBKRBookUnavailableError(RuntimeError):
    """No IBKR account book is available (and simulated data must never be used)."""


class IBKRExecutionService:
    """Opt-in real IBKR wiring that never substitutes a simulated book.

    Enabled with ``FUTURESRLX_API_ENABLE_IBKR=1`` (or ``settings.api_enable_ibkr``).
    The lifespan calls :meth:`connect` once, then :meth:`start_background` starts a
    throttled liveness probe, so a TWS restart (or the IBKR nightly server reset)
    is recovered automatically without restarting the API process.

    Degradation is honest:

    * while the session is down the service serves the **last-known-good IBKR
      book** with ``stale=True`` / ``stale_reason`` -- never the deterministic
      simulated broker;
    * before the first successful fetch it raises
      :class:`IBKRBookUnavailableError`, mapped by the routes to a clear 503;
    * :meth:`account` / :meth:`positions` invalidate a dead socket on request-path
      failure and schedule a bounded background reconnect.
    """

    #: Minimum seconds between background/request-path reconnect attempts.
    _MIN_PROBE_INTERVAL_SECONDS: float = 2.0

    def __init__(
        self,
        *,
        settings: Settings,
        market: fixtures.OfflineMarket,
        clock: fixtures.Clock = fixtures.utc_now,
        connection_manager: IBKRConnectionManager | None = None,
        equity_history_store: LiveEquityHistoryStore | None = None,
    ) -> None:
        from src.execution.ibkr_client import IBKRConnectionManager
        from src.execution.oms import OrderManagementSystem

        self._settings = settings
        self._market = market
        self._clock = clock
        manager = (
            connection_manager
            if connection_manager is not None
            else IBKRConnectionManager(
                settings, instrument_master=market.instrument_master, clock=clock
            )
        )
        self._manager: IBKRConnectionManager = manager
        self._oms: OrderManagementSystem = OrderManagementSystem(
            manager=manager,
            settings=settings,
            instrument_master=market.instrument_master,
            clock=clock,
        )
        self._connect_error: str | None = None
        self._ever_connected: bool = False
        self._last_good_account: AccountSnapshotResponse | None = None
        self._last_good_positions: dict[InstrumentId, Position] | None = None
        self._last_good_at: datetime | None = None
        self._liveness_task: asyncio.Task[None] | None = None
        self._ensure_task: asyncio.Task[bool] | None = None
        self._last_ensure_at: float | None = None
        self._flatten_in_progress: bool = False
        # One IBKR book refresh at a time: concurrent account-summary requests on
        # the same client can wedge TWS after a reconnect.
        self._refresh_lock = asyncio.Lock()
        self._probe_interval: float = max(
            self._MIN_PROBE_INTERVAL_SECONDS,
            float(settings.ibkr_heartbeat_interval_seconds),
        )
        #: Durable history of real observed IBKR NLV points (never synthetic).
        self._equity_history: LiveEquityHistoryStore = (
            equity_history_store
            if equity_history_store is not None
            else LiveEquityHistoryStore(
                settings.data_dir / EQUITY_HISTORY_FILE_NAME,
                clock=clock,
            )
        )

    # -- introspection ---------------------------------------------------------

    @property
    def ibkr_enabled(self) -> bool:
        """Whether this adapter is the real IBKR execution engine."""
        return True

    @property
    def connected(self) -> bool:
        """Whether the IBKR session is currently established."""
        return bool(self._manager.is_connected)

    @property
    def reconnecting(self) -> bool:
        """Whether a reconnect attempt (heartbeat, liveness or request) is in flight."""
        task = self._ensure_task
        return bool(self._manager.is_reconnecting or (task is not None and not task.done()))

    @property
    def last_error(self) -> str | None:
        """Most recent connection/fetch failure (``None`` while healthy)."""
        if self._connect_error is not None:
            return self._connect_error
        manager_error = getattr(self._manager, "last_error", None)
        return manager_error if isinstance(manager_error, str) else None

    @property
    def last_book_at(self) -> datetime | None:
        """Timestamp of the last successful live account/position fetch."""
        return self._last_good_at

    @property
    def has_last_good_book(self) -> bool:
        """Whether at least one live IBKR account snapshot has been cached."""
        return self._last_good_account is not None

    @property
    def source_name(self) -> str:
        """Provenance reported by the API (the IBKR book is always ``engine``)."""
        return "engine"

    @property
    def live_equity_points(self) -> tuple[EquityPoint, ...]:
        """Persisted real IBKR NLV points (oldest first; ``()`` when absent)."""
        try:
            return tuple(self._equity_history.load())
        except Exception:
            logger.warning("failed to load IBKR equity history", exc_info=True)
            return ()

    @property
    def equity_history_path(self) -> Path:
        """Path of the durable IBKR NLV history document."""
        return self._equity_history.path

    @property
    def connection_error(self) -> str | None:
        """Last connection error (``None`` when healthy)."""
        return self.last_error

    def working_orders_snapshot(self) -> list[dict[str, object]]:
        """Working OMS orders as plain rows for ``GET /api/orders/working``."""
        rows: list[dict[str, object]] = []
        timestamp = self._clock()
        for managed in self._oms.orders.values():
            if managed.is_terminal:
                continue
            event = managed.event
            rows.append(
                {
                    "order_id": event.order_id,
                    "instrument_id": event.instrument_id,
                    "side": event.side.value,
                    "quantity": event.quantity,
                    "filled_quantity": managed.filled_quantity,
                    "remaining_quantity": max(0, event.quantity - managed.filled_quantity),
                    "status": managed.status.value,
                    "order_type": event.order_type.value,
                    "limit_price": event.limit_price,
                    "updated_at": timestamp,
                }
            )
        return rows

    def recent_fills_snapshot(self, limit: int = 50) -> list[dict[str, object]]:
        """Most recent OMS fills as plain rows for ``GET /api/fills/recent``."""
        fills = self._oms.all_fills()[-max(1, limit) :]
        return [
            {
                "order_id": fill.order_id,
                "instrument_id": fill.instrument_id,
                "side": fill.side.value,
                "quantity": fill.quantity,
                "price": fill.price,
                "commission": fill.commission,
                "timestamp": fill.timestamp,
            }
            for fill in fills
        ]

    # -- lifecycle / reconnect -------------------------------------------------

    async def connect(self) -> bool:
        """Attempt the IBKR connection within the configured timeout."""
        try:
            await asyncio.wait_for(
                self._manager.connect(), timeout=self._settings.ibkr_connect_timeout_seconds
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._connect_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "IBKR session unavailable; background reconnect active: %s", self._connect_error
            )
            return False
        self._connect_error = None
        self._ever_connected = True
        logger.info("IBKR execution session established")
        return bool(self._manager.is_connected)

    async def ensure_connected(self) -> bool:
        """Throttled, bounded reconnect probe; never raises into the caller."""
        if self._manager.is_connected:
            return True
        now = time.monotonic()
        last = self._last_ensure_at
        if last is not None and (now - last) < self._probe_interval:
            return False
        self._last_ensure_at = now
        timeout = max(0.1, float(self._settings.ibkr_connect_timeout_seconds))
        try:
            connected = await asyncio.wait_for(
                self._manager.ensure_connected(max_attempts=1), timeout=timeout
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._connect_error = f"{type(exc).__name__}: {exc}"
            logger.warning("IBKR reconnect probe failed: %s", self._connect_error)
            return False
        if connected:
            self._connect_error = None
            self._ever_connected = True
            return True
        manager_error = getattr(self._manager, "last_error", None)
        if isinstance(manager_error, str):
            self._connect_error = manager_error
        return False

    async def start_background(self) -> None:
        """Start the throttled liveness probe that reconnects after a TWS restart."""
        task = self._liveness_task
        if task is not None and not task.done():
            return
        self._liveness_task = asyncio.create_task(self._liveness_loop(), name="ibkr-liveness")

    async def stop_background(self) -> None:
        """Cancel the liveness probe and any in-flight request-path reconnect."""
        tasks = [self._liveness_task, self._ensure_task]
        self._liveness_task = None
        self._ensure_task = None
        for task in tasks:
            if task is None or task.done():
                continue
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def disconnect(self) -> None:
        """Stop background probes and close the IBKR session if established."""
        await self.stop_background()
        try:
            await self._manager.disconnect()
        except Exception:
            logger.exception("error while disconnecting IBKR session")

    async def _liveness_loop(self) -> None:
        try:
            while True:
                try:
                    if self._manager.is_connected:
                        self._manager.ensure_heartbeat()
                    elif not self._manager.is_reconnecting:
                        await self.ensure_connected()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("IBKR liveness probe failed")
                await asyncio.sleep(self._probe_interval)
        except asyncio.CancelledError:
            raise
        finally:
            self._liveness_task = None

    async def _attempt_connect_now(self) -> bool:
        """Operator-intent reconnect (bypasses the liveness throttle)."""
        self._last_ensure_at = None
        return await self.ensure_connected()

    def _schedule_ensure(self) -> None:
        """Trigger one bounded background reconnect from a failed request path."""
        task = self._ensure_task
        if task is not None and not task.done():
            return
        if self._manager.is_connected or self._manager.is_reconnecting:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # pragma: no cover - synchronous caller (tests/CLI)
            return
        scheduled = loop.create_task(self.ensure_connected(), name="ibkr-request-ensure")
        self._ensure_task = scheduled
        scheduled.add_done_callback(self._clear_ensure_task)

    def _clear_ensure_task(self, task: asyncio.Task[bool]) -> None:
        if task is self._ensure_task:
            self._ensure_task = None
        if not task.cancelled():
            task.exception()  # retrieved; failures are recorded on the service

    # -- book fetching ---------------------------------------------------------

    async def _refresh_book(self) -> bool:
        """Fetch the live book; cache it as last-known-good. Never raises."""
        async with self._refresh_lock:
            return await self._refresh_book_locked()

    async def _refresh_book_locked(self) -> bool:
        """Single-flight body of :meth:`_refresh_book`."""
        if not self._manager.is_connected:
            return False
        try:
            snapshot = await self._manager.fetch_account()
            positions = await self._manager.fetch_positions()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._connect_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "IBKR account/position fetch failed; invalidating session: %s", self._connect_error
            )
            self._manager.invalidate(self._connect_error)
            self._schedule_ensure()
            return False
        self._connect_error = None
        self._ever_connected = True
        curve = fixtures.equity_history(self._market, snapshot.net_liquidation, now=self._clock())
        day_pnl = curve[-1].value - curve[-2].value if len(curve) > 1 else 0.0
        self._last_good_account = _account_response(
            snapshot=snapshot,
            positions=positions,
            master=self._market.instrument_master,
            market=self._market,
            curve=curve,
            day_pnl=day_pnl,
        ).model_copy(update={"source": "engine"})
        self._last_good_positions = dict(positions)
        self._last_good_at = self._clock()
        self._record_equity_point(
            EquityPoint(time=snapshot.timestamp, value=snapshot.net_liquidation)
        )
        return True

    def _record_equity_point(self, point: EquityPoint) -> None:
        """Persist one observed NLV point; attribution I/O must never break trading."""
        try:
            self._equity_history.append(point)
        except Exception:
            logger.warning("failed to persist IBKR equity history point", exc_info=True)

    def _unavailable_reason(self) -> str:
        detail = self.last_error or "IBKR session not connected"
        return (
            "IBKR execution is enabled but no live account book is available yet "
            f"(no simulated fallback): {detail}"
        )

    def _stale_book(self, cached: AccountSnapshotResponse) -> AccountSnapshotResponse:
        moment = self._last_good_at.isoformat() if self._last_good_at is not None else "unknown"
        reason = (
            f"IBKR session unavailable ({self.last_error or 'disconnected'}); "
            f"serving last-known-good book fetched at {moment}"
        )
        return cached.model_copy(update={"source": "engine", "stale": True, "stale_reason": reason})

    async def account(self) -> AccountSnapshotResponse:
        """Live IBKR account, or the last-known-good book marked stale."""
        if await self._refresh_book():
            cached = self._last_good_account
            assert cached is not None
            return cached
        self._schedule_ensure()
        if self._last_good_account is None:
            raise IBKRBookUnavailableError(self._unavailable_reason())
        return self._stale_book(self._last_good_account)

    async def positions(self) -> Mapping[InstrumentId, Position]:
        """Live IBKR positions, or the last-known-good book marked stale."""
        if await self._refresh_book():
            return dict(self._last_good_positions or {})
        self._schedule_ensure()
        if self._last_good_positions is not None:
            return dict(self._last_good_positions)
        raise IBKRBookUnavailableError(self._unavailable_reason())

    @property
    def is_locked(self) -> bool:
        """Whether the OMS gate kill switch is latched."""
        return self._oms.is_locked

    @property
    def lock_reason(self) -> str:
        """Kill-switch reason (empty when unlocked)."""
        return self._oms.gate.lock_reason if self._oms.is_locked else ""

    @property
    def lock_until(self) -> datetime | None:
        """OMS kill switch has no wall-clock expiry (``None``)."""
        return None

    @property
    def flatten_in_progress(self) -> bool:
        """Whether an emergency IBKR flatten is currently executing."""
        return self._flatten_in_progress

    def reset_kill_switch(self) -> None:
        """Release the OMS pre-trade kill switch and its persisted latch."""
        self._oms.reset_kill_switch()

    async def submit_orders(self, orders: Sequence[OrderEvent]) -> list[OrderReceipt]:
        """Submit delta orders through the live OMS (kill-switch/whatIf gated)."""
        return list(await self._oms.submit_orders(list(orders)))

    async def flatten_all(self) -> FlattenAllResponse:
        """Kill switch: live OMS flatten (never a simulated fallback)."""
        self._flatten_in_progress = True
        try:
            return await self._flatten_all_impl()
        finally:
            self._flatten_in_progress = False

    async def _flatten_all_impl(self) -> FlattenAllResponse:
        """Implementation of :meth:`flatten_all` (tracked by ``flatten_in_progress``)."""
        if not self._manager.is_connected:
            # Operator intent: attempt a bounded reconnect immediately, even if a
            # background probe ran a moment ago.
            await self._attempt_connect_now()
        if not self._manager.is_connected:
            raise IBKRBookUnavailableError(
                "cannot flatten: IBKR session unavailable and simulated positions must never "
                f"be used ({self.last_error or 'disconnected'})"
            )
        try:
            positions = await self._manager.fetch_positions()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._connect_error = f"{type(exc).__name__}: {exc}"
            self._manager.invalidate(self._connect_error)
            self._schedule_ensure()
            raise IBKRBookUnavailableError(
                f"cannot flatten: failed to fetch IBKR positions ({self._connect_error})"
            ) from exc
        timestamp = self._clock()
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
    config: StrategyConfig, master: InstrumentsConfig, settings: Settings | None = None
) -> ContractTranslatorProtocol:
    from src.risk.contract_translator import ContractTranslator

    return cast(
        ContractTranslatorProtocol,
        ContractTranslator.from_config(
            config.translator,
            instruments=master,
            account_currency=settings.account_currency if settings is not None else None,
            fx_rates=settings.fx_rates if settings is not None else None,
        ),
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
    client: object = fixtures.OfflineLLMClient(state, thresholds, clock())
    # Phase-9: prefer the risk layer's provider factory when it is importable and a
    # real provider is configured; the offline deterministic client stays the
    # default so unit/integration runs never touch the network.
    if settings.llm_provider != "offline":
        try:
            from src.risk.llm_clients import build_llm_client as _build_llm_client

            wired_client = _build_llm_client(settings)
            if wired_client is not None:
                client = wired_client
        except Exception as exc:
            logger.warning(
                "src.risk.llm_clients is unavailable (%s); using the offline LLM client",
                exc,
            )
    calendar = _macro_calendar(str(settings.configs_dir / "macro_calendar.yaml"))
    try:
        overlay = LLMRiskOverlay(
            cast("Any", client),
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

    translator = _build_translator(config, master, settings)
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


def _report_field(report: object, name: str, default: object = None) -> object:
    """Field access on a discretization report (attribute or mapping based)."""
    if isinstance(report, Mapping):
        return cast(Mapping[str, object], report).get(name, default)
    return getattr(report, name, default)


def _report_float(report: object, name: str) -> float | None:
    value = _report_field(report, name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _report_weights(report: object, name: str) -> dict[str, float] | None:
    value = _report_field(report, name)
    if not isinstance(value, Mapping):
        return None
    weights: dict[str, float] = {}
    for key, weight in cast(Mapping[object, object], value).items():
        if isinstance(weight, bool) or not isinstance(weight, int | float):
            continue
        result = float(weight)
        if math.isfinite(result):
            weights[str(key).upper()] = result
    return weights or None


def achieved_weights(
    targets: ContractTargetSet, state: MarketState, equity: float
) -> dict[str, float]:
    """Signed notional weight actually achieved by the discrete contract book."""
    achieved: dict[str, float] = {}
    target_rows = targets.by_instrument
    for root in state.instruments:
        target = target_rows.get(root)
        if target is None or equity <= 0.0:
            achieved[root] = 0.0
        else:
            achieved[root] = target.signed_notional / equity
    return achieved


def discretization_details(
    translator: ContractTranslatorProtocol | None,
    weights: WeightVector,
    targets: ContractTargetSet,
    state: MarketState,
    equity: float,
) -> tuple[float, list[MicroRoutingEntry], dict[str, float], str]:
    """Return ``(tracking_error_pct, micro_routing, achieved_weights, source)``.

    Uses the real ``ContractTranslator.discretization_report`` when the risk layer
    exposes it; otherwise computes the tracking error arithmetically from the
    discrete contract book, so the Pipeline Inspector never shows a fabricated
    zero or crashes on a missing translator API.
    """
    fallback_achieved = achieved_weights(targets, state, equity)
    fallback_error = sum(
        abs(weights.get(root, 0.0) - fallback_achieved.get(root, 0.0))
        for root in set(weights) | set(fallback_achieved)
    )
    report: object | None = None
    if translator is not None:
        reporter = getattr(translator, "discretization_report", None)
        if callable(reporter):
            attempts: tuple[dict[str, object], ...] = (
                {"equity": equity},
                {},
            )
            for extra in attempts:
                try:
                    report = reporter(weights, targets, state, **extra)
                    break
                except TypeError:
                    continue
                except Exception as exc:
                    logger.warning("discretization_report failed: %s", exc)
                    report = None
                    break
    if report is None:
        return (round(max(0.0, fallback_error) * 100.0, 6), [], fallback_achieved, "fallback")
    tracking = _report_float(report, "tracking_error_pct")
    if tracking is None:
        raw_error = _report_float(report, "tracking_error")
        tracking = raw_error * 100.0 if raw_error is not None else fallback_error * 100.0
    micro_routes = _report_field(report, "micro_routes")
    if micro_routes is None:
        micro_routes = _report_field(report, "routes", ())
    routing = micro_routing_from_payload(micro_routes)
    report_achieved = _report_weights(report, "achieved_weights")
    achieved = report_achieved if report_achieved is not None else fallback_achieved
    return (round(max(0.0, float(tracking)), 6), routing, achieved, "translator")


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

    def _pipeline_for(
        self, config: StrategyConfig, state: MarketState
    ) -> tuple[StrategyPipeline, ContractTranslatorProtocol]:
        pipeline, translator = build_pipeline_parts(
            config,
            self._master,
            self._settings,
            clock=self._clock,
            state=state,
        )
        return pipeline, translator

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
        pipeline, translator = self._pipeline_for(config, state)
        result = pipeline.run(state)
        response = self._map_pipeline(config, state, result, translator)
        self._pipeline_cache[key] = response
        return response

    def _map_pipeline(
        self,
        config: StrategyConfig,
        state: MarketState,
        result: PipelineResult,
        translator: ContractTranslatorProtocol | None = None,
    ) -> PipelineWaterfallResponse:
        targets = result.targets if result.targets is not None else ContractTargetSet.empty()
        target_rows = targets.by_instrument
        selected = set(result.universe)
        equity = state.equity or INITIAL_EQUITY
        tracking_error_pct, micro_routing, achieved, tracking_source = discretization_details(
            translator, result.final_weights, targets, state, equity
        )
        instruments: list[PipelineInstrumentRow] = []
        for root in sorted(state.instruments):
            base = result.base_weights.get(root, 0.0)
            timing = result.timing_weights.get(root, 0.0)
            risk = result.risk_weights.get(root, 0.0)
            final = result.final_weights.get(root, 0.0)
            target = target_rows.get(root)
            driver = _adjustment_reason(
                root in selected, base, timing, risk, target.quantity if target else 0
            )
            instruments.append(
                PipelineInstrumentRow(
                    instrument_id=root,
                    asset_class=state.instruments[root].asset_class,
                    selected=root in selected,
                    weight_base=round(base, 8),
                    weight_timing=round(timing, 8),
                    weight_risk=round(risk, 8),
                    weight_final=round(final, 8),
                    weight_actual=round(achieved.get(root, 0.0), 8),
                    delta_timing=round(timing - base, 8),
                    delta_risk=round(risk - timing, 8),
                    target_contracts=target.quantity if target is not None else None,
                    notional=round(final * equity, 2),
                    driver=driver,
                    regime=state.regime,
                    reason=driver,
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
            tracking_error_pct=tracking_error_pct,
            micro_routing=micro_routing,
            tracking_error_source=cast('Literal["translator", "fallback"]', tracking_source),
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
        equity = state.equity or INITIAL_EQUITY
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
            spec = self._master.get(root) if self._master.has(root) else None
            multiplier = spec.multiplier if spec is not None else 50.0
            achieved = (
                (quantity or 0) * state.price(root) * multiplier / INITIAL_EQUITY
                if equity > 0
                else 0.0
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
                    weight_actual=round(achieved, 8),
                    delta_timing=timing - base,
                    delta_risk=risk - timing,
                    target_contracts=quantity,
                    notional=risk * INITIAL_EQUITY,
                    driver="fixture fallback: real pipeline layer unavailable",
                    regime=state.regime,
                    reason="fixture fallback: real pipeline layer unavailable",
                )
            )
        gross = sum(abs(row.weight_final) for row in instruments)
        equity = state.equity or INITIAL_EQUITY
        tracking_error_pct = round(
            sum(abs(row.weight_final - (row.weight_actual or 0.0)) for row in instruments) * 100.0,
            6,
        )
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
            tracking_error_pct=tracking_error_pct,
            micro_routing=[],
            tracking_error_source="fallback",
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
        news: NewsFeedService | None = None,
        state_store: RiskStateStore | None = None,
    ) -> None:
        self._settings = settings
        self._market = market
        self._execution = execution
        self._clock = clock
        self._news = news
        self._state_store = state_store
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

    # -- operator kill-switch status / reset (QoL) -----------------------------

    @staticmethod
    def operator_reset_token_required() -> bool:
        """Whether ``FUTURESRLX_OPERATOR_RESET_TOKEN`` is configured."""
        return bool(os.environ.get(OPERATOR_RESET_TOKEN_ENV, "").strip())

    @staticmethod
    def _configured_operator_token() -> str:
        return os.environ.get(OPERATOR_RESET_TOKEN_ENV, "").strip()

    def _persisted_state(self) -> PersistedRiskState | None:
        """Durable risk state, or ``None`` when no store/state exists."""
        store = self._state_store
        if store is None:
            return None
        try:
            return store.load()
        except Exception:
            logger.warning("risk state store could not be read", exc_info=True)
            return None

    def _kill_switch_snapshot(self) -> dict[str, object]:
        """Honest kill-switch status merged from execution and the risk state store."""
        persisted = self._persisted_state()
        execution_locked = bool(self._execution.is_locked)
        execution_reason = str(self._execution.lock_reason or "") if execution_locked else ""
        store_locked = bool(persisted.kill_switch_locked) if persisted is not None else False
        store_reason = (
            str(persisted.kill_switch_reason) if persisted is not None and store_locked else ""
        )
        daily_latched = bool(persisted.daily_loss_locked) if persisted is not None else False
        raw_lock_until = getattr(self._execution, "lock_until", None)
        lock_until = raw_lock_until if isinstance(raw_lock_until, datetime) else None
        flattening = bool(getattr(self._execution, "flatten_in_progress", False))
        reset_available = (
            callable(getattr(self._execution, "reset_kill_switch", None)) and not flattening
        )
        return {
            "kill_switch_active": execution_locked or store_locked or daily_latched,
            "kill_switch_reason": execution_reason or store_reason,
            "daily_loss_breaker_active": daily_latched,
            "lock_until": lock_until,
            "reset_available": reset_available,
            "operator_reset_token_required": self.operator_reset_token_required(),
            "flattening": flattening,
        }

    def _clear_persisted_risk_state(self, now: datetime) -> bool:
        """Operator reset of the durable risk state (kill switch/daily/HWM latches).

        The load-modify-save cycle preserves OMS-owned fields (the rate-limiter
        bucket) while clearing the kill-switch latch, the daily-loss latch, the
        starting daily NLV, the portfolio/instrument high-water marks and the
        quant-overlay trailing-stop metadata so the re-armed engine resumes cleanly.
        Returns whether any durable latch/HWM state was actually cleared.
        """
        store = self._state_store
        if store is None:
            return False
        base = store.load()
        if base is None:
            return False
        overlay_extra = any(key.startswith(QUANT_OVERLAY_EXTRA_PREFIX) for key in base.extra)
        cleared = bool(
            base.kill_switch_locked
            or base.daily_loss_locked
            or base.starting_daily_nlv is not None
            or base.portfolio_high_water_mark is not None
            or base.instrument_high_water_marks
            or overlay_extra
        )
        extra = {
            key: value
            for key, value in base.extra.items()
            if not key.startswith(QUANT_OVERLAY_EXTRA_PREFIX)
        }
        updated = base.unlocked().updated(
            starting_daily_nlv=None,
            portfolio_high_water_mark=None,
            instrument_high_water_marks={},
            trading_day=now.date(),
            updated_at=now,
            extra=extra,
        )
        store.save(updated)
        return cleared

    def reset_risk_lock(
        self,
        request: RiskLockResetRequest,
        *,
        header_token: str | None = None,
    ) -> RiskLockResetResponse:
        """Operator kill-switch reset with confirmation phrase + optional token."""
        now = self._clock()
        if request.confirm.strip() != RISK_LOCK_RESET_CONFIRM:
            raise OperatorConfirmationError(f'confirm must be exactly "{RISK_LOCK_RESET_CONFIRM}"')
        required_token = self._configured_operator_token()
        supplied_token = (request.operator_token or request.token or header_token or "").strip()
        if required_token and supplied_token != required_token:
            raise OperatorTokenError(
                f"operator reset token is required and must match {OPERATOR_RESET_TOKEN_ENV}"
            )
        before = self._kill_switch_snapshot()
        if bool(before["flattening"]):
            raise OperatorActionUnavailableError(
                "refusing to reset the risk lock while an emergency flatten is in progress"
            )
        resetter = getattr(self._execution, "reset_kill_switch", None)
        if not callable(resetter):
            raise OperatorActionUnavailableError(
                "the active execution adapter does not support a manual kill-switch reset"
            )
        warnings: list[str] = []
        try:
            resetter()
            execution_reset = True
        except Exception as exc:
            raise OperatorActionUnavailableError(
                f"execution kill-switch reset failed: {type(exc).__name__}: {exc}"
            ) from exc
        day_resetter = getattr(self._execution, "reset_day", None)
        if callable(day_resetter):
            try:
                day_resetter()
            except Exception:
                logger.warning("execution reset_day() failed", exc_info=True)
        persisted_reset = False
        if request.reset_persisted_state:
            try:
                persisted_reset = self._clear_persisted_risk_state(now)
            except Exception as exc:
                warnings.append(f"persisted risk state reset failed: {type(exc).__name__}: {exc}")
                logger.warning("persisted risk state reset failed", exc_info=True)
        after = self._kill_switch_snapshot()
        changed = bool(before["kill_switch_active"]) or persisted_reset
        status: Literal["reset", "already_unlocked"] = "reset" if changed else "already_unlocked"
        return RiskLockResetResponse(
            status=status,
            timestamp=now,
            execution_reset=execution_reset,
            persisted_state_reset=persisted_reset,
            kill_switch_active=bool(after["kill_switch_active"]),
            kill_switch_reason=str(after["kill_switch_reason"]),
            daily_loss_breaker_active=bool(after["daily_loss_breaker_active"]),
            lock_until=cast(datetime | None, after["lock_until"]),
            warnings=warnings,
            message=(
                "risk lock cleared; execution resumed"
                if execution_reset
                else "risk lock was not latched"
            ),
        )

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
        ingested_news = self._news.items() if self._news is not None else []
        if ingested_news:
            news = ingested_news
            news_source = self._news.source if self._news is not None else "fixture"
        else:
            news = fixtures.news_feed(assessments, now)
            news_source = "fixture"
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
        lock = self._kill_switch_snapshot()
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
            news_source=news_source,
            kill_switch_active=bool(lock["kill_switch_active"]),
            kill_switch_reason=str(lock["kill_switch_reason"]),
            daily_loss_breaker_active=bool(lock["daily_loss_breaker_active"]),
            lock_until=cast(datetime | None, lock["lock_until"]),
            reset_available=bool(lock["reset_available"]),
            operator_reset_token_required=bool(lock["operator_reset_token_required"]),
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
    live: LiveStateService
    news: NewsFeedService
    offline_mode: bool
    warnings: list[str] = field(default_factory=list)
    #: Durable risk/kill-switch state shared with the live runner (file/DuckDB/Redis).
    risk_state_store: RiskStateStore | None = None

    def health(self) -> HealthResponse:
        """Liveness payload with dynamically derived IBKR connectivity state.

        ``offline_mode`` is recomputed on every request: for an IBKR-enabled
        process it is ``not ibkr_connected``; for a simulated/backtest process it
        stays ``True``. A TWS restart therefore flips the dashboard badge back to
        online automatically once the background reconnect succeeds.
        """
        now = self.clock()
        bridge_available = self.live.bridge_available()
        ibkr_enabled = bool(getattr(self.execution, "ibkr_enabled", False))
        ibkr_connected = bool(getattr(self.execution, "connected", False))
        ibkr_reconnecting = bool(getattr(self.execution, "reconnecting", False))
        raw_last_error = getattr(self.execution, "last_error", None)
        ibkr_last_error = raw_last_error if isinstance(raw_last_error, str) else None
        offline_mode = (not ibkr_connected) if ibkr_enabled else self.offline_mode
        return HealthResponse(
            status="degraded" if self.warnings or offline_mode else "ok",
            app_name=self.settings.app_name,
            version=API_VERSION,
            environment=self.settings.environment,
            offline_mode=offline_mode,
            uptime_seconds=max(0.0, (now - self.started_at).total_seconds()),
            server_time=now,
            ibkr_connected=ibkr_connected,
            ibkr_enabled=ibkr_enabled,
            ibkr_reconnecting=ibkr_reconnecting,
            ibkr_last_error=ibkr_last_error,
            runner_bridge_available=bridge_available,
            runner_bridge_stale=self.live.bridge_stale(),
            runner_mode=self.live.runner_mode(),
        )

    def telemetry_status(self) -> TelemetryStatusResponse:
        """WebSocket broadcaster status."""
        return self.telemetry.status()

    # -- operator resets (QoL) -------------------------------------------------

    def reset_risk_lock(
        self,
        request: RiskLockResetRequest,
        *,
        header_token: str | None = None,
    ) -> RiskLockResetResponse:
        """Delegate the guarded operator kill-switch reset to the risk service."""
        resetter = getattr(self.risk, "reset_risk_lock", None)
        if not callable(resetter):
            raise OperatorActionUnavailableError(
                "the configured risk service does not support a manual lock reset"
            )
        return cast(RiskLockResetResponse, resetter(request, header_token=header_token))

    async def reset_paper_state(self, request: PaperStateResetRequest) -> PaperStateResetResponse:
        """Clear paper/backtest operator state (bridge file, risk store, paper book).

        Never available in the ``live`` environment. Every requested clear is
        attempted independently so one failure cannot block the others; the
        response receipts record exactly what happened and repeated calls are safe.
        """
        now = self.clock()
        environment = self.settings.environment
        if environment not in PAPER_STATE_RESET_ENVIRONMENTS:
            raise OperatorActionUnavailableError(
                "reset-paper-state is only available in non-live environments "
                f"(environment={environment!r})"
            )
        if request.confirm.strip() != PAPER_STATE_RESET_CONFIRM:
            raise OperatorConfirmationError(
                f'confirm must be exactly "{PAPER_STATE_RESET_CONFIRM}"'
            )
        receipts = PaperStateResetReceipts(
            risk_state_store_uri=self.settings.risk_state_store_uri or "memory:"
        )
        warnings: list[str] = []
        actions = 0
        if request.clear_runner_bridge:
            removed = self._remove_runner_bridge_snapshot()
            receipts.files_removed.extend(removed)
            actions += len(removed)
        if request.clear_risk_state:
            store = self.risk_state_store
            if store is None:
                warnings.append("no persisted risk state store is configured")
            else:
                try:
                    existing = store.load()
                    store.clear()
                    receipts.risk_state_store_cleared = existing is not None
                    actions += 1 if existing is not None else 0
                except Exception as exc:
                    warnings.append(f"risk state store reset failed: {type(exc).__name__}: {exc}")
                    logger.warning("paper-state risk store reset failed", exc_info=True)
        if request.clear_paper_cache:
            try:
                await self.live.close()
                receipts.paper_session_reset = True
                actions += 1
            except Exception as exc:
                warnings.append(f"paper runner session reset failed: {type(exc).__name__}: {exc}")
                logger.warning("paper runner session reset failed", exc_info=True)
            resetter = getattr(self.execution, "reset_paper_state", None)
            if callable(resetter):
                try:
                    resetter()
                except Exception as exc:
                    warnings.append(
                        f"paper execution cache reset failed: {type(exc).__name__}: {exc}"
                    )
                    logger.warning("paper execution cache reset failed", exc_info=True)
            elif bool(getattr(self.execution, "ibkr_enabled", False)):
                warnings.append(
                    "IBKR adapter: IBKR session/book left untouched by the paper-state reset"
                )
        if warnings and actions:
            status: Literal["reset", "noop", "partial"] = "partial"
        elif warnings:
            status = "noop"
        elif actions:
            status = "reset"
        else:
            status = "noop"
        return PaperStateResetResponse(
            status=status,
            timestamp=now,
            environment=environment,
            confirm_ok=True,
            receipts=receipts,
            warnings=warnings,
            message=(
                "paper state cleared"
                if status == "reset"
                else "nothing to clear (idempotent no-op)"
                if status == "noop"
                else "paper state partially cleared; see warnings"
            ),
        )

    def _remove_runner_bridge_snapshot(self) -> list[str]:
        """Remove the runner bridge snapshot (and its atomic temp files)."""
        bridge_dir = self.settings.runner_bridge_dir
        removed: list[str] = []
        candidates = [bridge_dir / BRIDGE_FILE_NAME]
        with contextlib.suppress(OSError):
            candidates.extend(bridge_dir.glob(f".{BRIDGE_FILE_NAME}.*.tmp"))
        for path in candidates:
            try:
                if path.is_file():
                    path.unlink()
                    removed.append(str(path))
            except OSError as exc:
                logger.warning("could not remove runner bridge file %s: %s", path, exc)
        return removed

    async def start(self) -> None:
        """Start the telemetry publisher, live execution session and reconnect probe."""
        connector = getattr(self.execution, "connect", None)
        if callable(connector):
            outcome = connector()
            connected = await outcome if inspect.isawaitable(outcome) else bool(outcome)
            if not connected:
                # Never latch offline_mode here: /health derives it dynamically and a
                # background liveness probe reconnects once TWS is back. No simulated
                # book is ever served in the meantime.
                logger.warning(
                    "IBKR session unavailable at startup; background reconnect probe active"
                )
        starter = getattr(self.execution, "start_background", None)
        if callable(starter):
            outcome = starter()
            if inspect.isawaitable(outcome):
                await outcome
        await self.telemetry.start()
        await self.news.start()
        await self.live.start()

    async def stop(self) -> None:
        """Stop telemetry, the news poller, runner session and live sessions."""
        await self.telemetry.stop()
        await self.news.stop()
        await self.live.close()
        stopper = getattr(self.execution, "stop_background", None)
        if callable(stopper):
            outcome = stopper()
            if inspect.isawaitable(outcome):
                await outcome
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
    enabled = settings.api_enable_ibkr or (
        os.environ.get(IBKR_ENABLE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}
    )
    if enabled:
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


def _warm_optional_imports() -> None:
    """Import the optional strategy stack before any background thread starts.

    The telemetry publisher imports ``src.strategies`` to build its first pipeline
    frame, while Studio requests import ``src.strategies.parameter_catalog``; Python's
    per-module import locks can deadlock if both first imports race in different
    threads. Container construction is single-threaded and runs before the telemetry
    task starts, so warming the imports here removes the contention window.
    """
    import importlib

    for module_name in ("src.strategies", "src.strategies.parameter_catalog"):
        try:
            importlib.import_module(module_name)
        except Exception:  # pragma: no cover - strategy layer always present
            logger.debug("optional strategy import %s unavailable", module_name)


def build_container(
    settings: Settings | None = None,
    *,
    clock: fixtures.Clock | None = None,
) -> Container:
    """Build the service container (real adapters with deterministic fallbacks)."""
    _warm_optional_imports()
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
    try:
        risk_state_store: RiskStateStore | None = build_risk_state_store(
            resolved_settings.risk_state_store_uri,
            duckdb_path=resolved_settings.duckdb_path,
        )
    except Exception as exc:
        warnings.append(f"risk state store unavailable: {type(exc).__name__}: {exc}")
        logger.exception("risk state store could not be constructed")
        risk_state_store = None
    news = NewsFeedService(settings=resolved_settings, clock=resolved_clock)
    risk = EngineRiskService(
        settings=resolved_settings,
        market=market_service,
        execution=execution,
        clock=resolved_clock,
        news=news,
        state_store=risk_state_store,
    )
    from src.api.backtests_service import EngineBacktestService

    backtests = EngineBacktestService(
        settings=resolved_settings,
        market=market,
        catalog=catalog,
        clock=resolved_clock,
    )
    # LiveStateService is built first so the telemetry broadcaster consumes the
    # SAME enriched account as GET /api/positions (IBKR last-known-good book plus
    # curve/volatility attribution, and no demo jitter in live mode).
    live = LiveStateService(
        settings=resolved_settings,
        clock=resolved_clock,
        execution=execution,
        instrument_master=master,
        market=market_service,
    )
    telemetry = TelemetryBroadcaster(
        settings=resolved_settings,
        clock=resolved_clock,
        execution=execution,
        risk=risk,
        market=market_service,
        account_provider=live.account,
    )
    warnings.extend(live.warnings)
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
        live=live,
        news=news,
        offline_mode=offline_mode,
        warnings=warnings,
        risk_state_store=risk_state_store,
    )
