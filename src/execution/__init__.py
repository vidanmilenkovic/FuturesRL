"""FuturesRL-X IBKR execution engine.

Public surface
-------------
Connection / state
    :class:`IBKRConnectionManager`, :class:`IBClientLike`, the pure ``Venue*``
    transport dataclasses and the :class:`MarginImpact` margin-model container.
Orders
    :class:`OrderManagementSystem`, :class:`PreTradeRiskGate`,
    :class:`ManagedOrder`, :class:`GateDecision`, :class:`IBOrderRequest` and the
    deterministic id/delta helpers :func:`deterministic_order_id` /
    :func:`build_orders_from_targets`.
Rolls
    :class:`ContractRoller`, :class:`RollPlan`, :class:`RollExecution`,
    :class:`BagOrderSpec`, :func:`select_roll_expirations`.
Reconciliation
    :class:`PositionReconciler`, :class:`WorkingOrderSnapshot`.
Cold-start hydration (Phase 7)
    :class:`ColdStartHydrator`, :class:`HydrationResult`,
    :class:`HydrationError`, :data:`BackfillFn`.
Live trading runner (Phase 7)
    :class:`TradingRunner`, :class:`RunnerStatus`, :class:`TargetBroker`,
    :class:`RunnerNotHydratedError`, :class:`RunnerConfigError`, :func:`main`,
    :class:`OfflineRunnerSession` (deterministic API/offline controller session).
Runner <-> API bridge (Phase 9)
    :class:`RunnerBridgeSnapshot`, :class:`RunnerBridgeStore`,
    :func:`snapshot_from_runner`, the bridge dataclasses, :data:`BRIDGE_SCHEMA_VERSION`
    and the payload entry caps.

The runner exports (:class:`TradingRunner`, :class:`RunnerStatus`,
:class:`TargetBroker`, :class:`OfflineMarketStateStreamer`,
:class:`OfflineTargetBroker`, the runner errors and :func:`main`) are resolved
**lazily** through PEP 562 ``__getattr__``. Importing ``src.execution`` therefore
must not import :mod:`src.execution.runner`: otherwise ``python -m
src.execution.runner`` would find the target module in ``sys.modules`` before
``runpy`` executes it and emit ``<frozen runpy>:128: RuntimeWarning``. Every
eager export remains importable as before; only the runner module is deferred
until an attribute is actually accessed.

Everything imported here is SDK-free; ``ib_async`` is only imported lazily when
the live :class:`IBKRConnectionManager` client is instantiated (or by the
runner's ``main`` production wiring), so importing this package never touches the
network.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from src.execution.hydrator import (
    DEFAULT_MAX_WARMUP_STATES,
    HYDRATION_SOURCE_HINTS,
    HYDRATION_SOURCES,
    MIN_REQUIRED_BARS,
    NEUTRAL_TERM_STRUCTURE_DAYS,
    BackfillFn,
    BackfillSource,
    ColdStartHydrator,
    HydrationError,
    HydrationResult,
    MultiSourceBackfill,
    add_neutral_term_structure,
    build_backfill_chain,
    offline_backfill,
)
from src.execution.ibkr_client import (
    BagLeg,
    BagOrderSpec,
    ContractRequest,
    IBClientLike,
    IBKRBagUnsupportedError,
    IBKRConnectionError,
    IBKRConnectionManager,
    IBKRDataError,
    IBKRError,
    IBOrderRequest,
    MarginImpact,
    VenueAccountValue,
    VenueContract,
    VenueFill,
    VenueOrderSnapshot,
    VenuePosition,
    VenueTrade,
    active_contract_expiration,
    contract_month_string,
    map_order_status,
    margin_impact_from_order_state,
    normalize_expiration,
    parse_contract_month,
    third_friday,
    upcoming_contract_expirations,
)
from src.execution.ibkr_roller import (
    ContractRoller,
    RollExecution,
    RollPlan,
    select_roll_expirations,
)
from src.execution.oms import (
    ORDER_ID_NAMESPACE,
    GateDecision,
    ManagedOrder,
    OrderManagementSystem,
    PreTradeRiskGate,
    build_orders_from_targets,
    deterministic_order_id,
)
from src.execution.reconciler import PositionReconciler, WorkingOrderSnapshot

if TYPE_CHECKING:
    from src.execution.runner import (
        DEFAULT_TELEMETRY_TOPIC,
        OFFLINE_DEFAULT_ROOTS,
        OFFLINE_INITIAL_CASH,
        OFFLINE_REPLAY_STATES,
        OFFLINE_SIMULATION_END,
        OfflineMarketStateStreamer,
        OfflineRunnerSession,
        OfflineTargetBroker,
        RunnerConfigError,
        RunnerNotHydratedError,
        RunnerStatus,
        TargetBroker,
        TradingRunner,
        build_hydration_chain,
        describe_hydration_sources,
        main,
    )
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

#: Export name -> defining module for lazily resolved attributes. Deferring both
#: the runner and the runner-bridge keeps ``python -m src.execution.runner`` free
#: of the runpy ``RuntimeWarning`` and keeps ``import src.execution`` light.
_LAZY_MODULE_ATTRS: Final[dict[str, str]] = {
    **dict.fromkeys(
        (
            "DEFAULT_TELEMETRY_TOPIC",
            "OFFLINE_DEFAULT_ROOTS",
            "OFFLINE_INITIAL_CASH",
            "OFFLINE_REPLAY_STATES",
            "OFFLINE_SIMULATION_END",
            "OfflineMarketStateStreamer",
            "OfflineRunnerSession",
            "OfflineTargetBroker",
            "RunnerConfigError",
            "RunnerNotHydratedError",
            "RunnerStatus",
            "TargetBroker",
            "TradingRunner",
            "build_hydration_chain",
            "describe_hydration_sources",
            "main",
        ),
        "src.execution.runner",
    ),
    **dict.fromkeys(
        (
            "BRIDGE_FILENAME",
            "BRIDGE_MODES",
            "BRIDGE_SCHEMA_VERSION",
            "DEFAULT_STALE_AFTER_SECONDS",
            "MAX_BRIDGE_FILLS",
            "MAX_BRIDGE_ORDERS",
            "BridgeFill",
            "BridgePosition",
            "BridgeSchemaError",
            "BridgeWorkingOrder",
            "RunnerBridgeSnapshot",
            "RunnerBridgeStore",
            "snapshot_from_runner",
        ),
        "src.execution.runner_bridge",
    ),
}


def __getattr__(name: str) -> object:
    """Resolve a lazy export on first access (PEP 562 deferred import)."""
    module_name = _LAZY_MODULE_ATTRS.get(name)
    if module_name is not None:
        import importlib

        module = importlib.import_module(module_name)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Expose the full public surface to ``dir(src.execution)``."""
    return sorted(__all__)


__all__ = [
    "BRIDGE_FILENAME",
    "BRIDGE_MODES",
    "BRIDGE_SCHEMA_VERSION",
    "DEFAULT_MAX_WARMUP_STATES",
    "DEFAULT_STALE_AFTER_SECONDS",
    "DEFAULT_TELEMETRY_TOPIC",
    "HYDRATION_SOURCES",
    "HYDRATION_SOURCE_HINTS",
    "MAX_BRIDGE_FILLS",
    "MAX_BRIDGE_ORDERS",
    "MIN_REQUIRED_BARS",
    "NEUTRAL_TERM_STRUCTURE_DAYS",
    "OFFLINE_DEFAULT_ROOTS",
    "OFFLINE_INITIAL_CASH",
    "OFFLINE_REPLAY_STATES",
    "OFFLINE_SIMULATION_END",
    "ORDER_ID_NAMESPACE",
    "BackfillFn",
    "BackfillSource",
    "BagLeg",
    "BagOrderSpec",
    "BridgeFill",
    "BridgePosition",
    "BridgeSchemaError",
    "BridgeWorkingOrder",
    "ColdStartHydrator",
    "ContractRequest",
    "ContractRoller",
    "GateDecision",
    "HydrationError",
    "HydrationResult",
    "IBClientLike",
    "IBKRBagUnsupportedError",
    "IBKRConnectionError",
    "IBKRConnectionManager",
    "IBKRDataError",
    "IBKRError",
    "IBOrderRequest",
    "ManagedOrder",
    "MarginImpact",
    "MultiSourceBackfill",
    "OfflineMarketStateStreamer",
    "OfflineRunnerSession",
    "OfflineTargetBroker",
    "OrderManagementSystem",
    "PositionReconciler",
    "PreTradeRiskGate",
    "RollExecution",
    "RollPlan",
    "RunnerBridgeSnapshot",
    "RunnerBridgeStore",
    "RunnerConfigError",
    "RunnerNotHydratedError",
    "RunnerStatus",
    "TargetBroker",
    "TradingRunner",
    "VenueAccountValue",
    "VenueContract",
    "VenueFill",
    "VenueOrderSnapshot",
    "VenuePosition",
    "VenueTrade",
    "WorkingOrderSnapshot",
    "active_contract_expiration",
    "add_neutral_term_structure",
    "build_backfill_chain",
    "build_hydration_chain",
    "build_orders_from_targets",
    "contract_month_string",
    "describe_hydration_sources",
    "deterministic_order_id",
    "main",
    "map_order_status",
    "margin_impact_from_order_state",
    "normalize_expiration",
    "offline_backfill",
    "parse_contract_month",
    "select_roll_expirations",
    "snapshot_from_runner",
    "third_friday",
    "upcoming_contract_expirations",
]
