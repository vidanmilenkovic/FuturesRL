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

Everything imported here is SDK-free; ``ib_async`` is only imported lazily when
the live :class:`IBKRConnectionManager` client is instantiated (or by the
runner's ``main`` production wiring), so importing this package never touches the
network.
"""

from __future__ import annotations

from src.execution.hydrator import (
    DEFAULT_MAX_WARMUP_STATES,
    MIN_REQUIRED_BARS,
    BackfillFn,
    ColdStartHydrator,
    HydrationError,
    HydrationResult,
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
from src.execution.runner import (
    DEFAULT_TELEMETRY_TOPIC,
    RunnerConfigError,
    RunnerNotHydratedError,
    RunnerStatus,
    TargetBroker,
    TradingRunner,
    main,
)

__all__ = [
    "DEFAULT_MAX_WARMUP_STATES",
    "DEFAULT_TELEMETRY_TOPIC",
    "MIN_REQUIRED_BARS",
    "ORDER_ID_NAMESPACE",
    "BackfillFn",
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
    "build_orders_from_targets",
    "contract_month_string",
    "deterministic_order_id",
    "main",
    "map_order_status",
    "margin_impact_from_order_state",
    "normalize_expiration",
    "parse_contract_month",
    "select_roll_expirations",
    "third_friday",
    "upcoming_contract_expirations",
]
