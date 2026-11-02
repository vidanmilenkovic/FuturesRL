"""FuturesRL-X IBKR execution engine.

Public surface
--------------
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
    :class:`PositionReconciler`.

Everything imported here is SDK-free; ``ib_async`` is only imported lazily when
the live :class:`IBKRConnectionManager` client is instantiated.
"""

from __future__ import annotations

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
from src.execution.reconciler import PositionReconciler

__all__ = [
    # orders / risk gate
    "ORDER_ID_NAMESPACE",
    # connection / transport
    "BagLeg",
    "BagOrderSpec",
    "ContractRequest",
    # rolls
    "ContractRoller",
    "GateDecision",
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
    # reconciliation
    "PositionReconciler",
    "PreTradeRiskGate",
    "RollExecution",
    "RollPlan",
    "VenueAccountValue",
    "VenueContract",
    "VenueFill",
    "VenueOrderSnapshot",
    "VenuePosition",
    "VenueTrade",
    "active_contract_expiration",
    "build_orders_from_targets",
    "contract_month_string",
    "deterministic_order_id",
    "map_order_status",
    "margin_impact_from_order_state",
    "normalize_expiration",
    "parse_contract_month",
    "select_roll_expirations",
    "third_friday",
    "upcoming_contract_expirations",
]
