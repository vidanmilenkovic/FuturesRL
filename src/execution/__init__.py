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
    :class:`RunnerNotHydratedError`, :class:`RunnerConfigError`, :func:`main`.

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

#: Runner names resolved lazily (PEP 562) so ``python -m src.execution.runner``
#: never triggers the runpy ``RuntimeWarning``. Attribute access imports the
#: runner module on demand and caches the value in the module globals.
_LAZY_RUNNER_ATTRS: Final[frozenset[str]] = frozenset(
    {
        "DEFAULT_TELEMETRY_TOPIC",
        "OFFLINE_DEFAULT_ROOTS",
        "OFFLINE_INITIAL_CASH",
        "OFFLINE_REPLAY_STATES",
        "OFFLINE_SIMULATION_END",
        "OfflineMarketStateStreamer",
        "OfflineTargetBroker",
        "RunnerConfigError",
        "RunnerNotHydratedError",
        "RunnerStatus",
        "TargetBroker",
        "TradingRunner",
        "build_hydration_chain",
        "describe_hydration_sources",
        "main",
    }
)


def __getattr__(name: str) -> object:
    """Resolve a runner export on first access (PEP 562 lazy import)."""
    if name in _LAZY_RUNNER_ATTRS:
        import importlib

        module = importlib.import_module("src.execution.runner")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Expose the full public surface to ``dir(src.execution)``."""
    return sorted(__all__)


__all__ = [
    "DEFAULT_MAX_WARMUP_STATES",
    "DEFAULT_TELEMETRY_TOPIC",
    "HYDRATION_SOURCES",
    "HYDRATION_SOURCE_HINTS",
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
    "OfflineTargetBroker",
    "OrderManagementSystem",
    "PositionReconciler",
    "PreTradeRiskGate",
    "RollExecution",
    "RollPlan",
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
    "third_friday",
    "upcoming_contract_expirations",
]
