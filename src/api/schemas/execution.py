"""Phase-11A rebalance preview/execute REST contract (dashboard runner book)."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import Field

from src.api.schemas.core import APIModel, StrictModel

# ---------------------------------------------------------------------------
# IBKR port discovery / guarded connection switch (workstation UX)
# ---------------------------------------------------------------------------


class IbkrPortInfo(APIModel):
    """One probed IBKR TWS/IB Gateway TCP port."""

    port: int = Field(ge=1, le=65535)
    mode: Literal["paper", "live"]
    gateway: Literal["tws", "ib_gateway"]
    reachable: bool
    latency_ms: float | None = None


class IbkrActiveConnection(APIModel):
    """Currently configured/active IBKR connection (settings/runtime)."""

    port: int | None = None
    client_id: int | None = None
    environment: str = "backtest"
    mode: Literal["paper", "live"] | None = None
    gateway: Literal["tws", "ib_gateway"] | None = None


class IbkrPortsResponse(APIModel):
    """``GET /api/runner/ibkr-ports`` payload."""

    host: str
    ports: list[IbkrPortInfo] = Field(default_factory=list)
    active: IbkrActiveConnection = Field(default_factory=IbkrActiveConnection)
    warnings: list[str] = Field(default_factory=list)
    probed_at: datetime


class IbkrConnectionSwitchRequest(StrictModel):
    """``POST /api/runner/ibkr-connection`` body."""

    port: int = Field(ge=1, le=65535)
    client_id: int | None = Field(default=None, ge=0, le=999)
    #: Required exactly ``"LIVE"`` when requesting a live port (7496/4001).
    confirmation: str | None = None


class IbkrRequestedConnection(APIModel):
    """Normalised connection requested by the switch endpoint."""

    port: int
    mode: Literal["paper", "live"]
    gateway: Literal["tws", "ib_gateway"]
    client_id: int


class IbkrConnectionSwitchResponse(APIModel):
    """Result of a guarded IBKR connection switch (never claims an unsafe switch)."""

    status: Literal["applied", "restart_required", "refused"]
    applied: bool = False
    restart_required: bool = False
    requested: IbkrRequestedConnection
    active: IbkrActiveConnection
    environment: str
    warnings: list[str] = Field(default_factory=list)
    message: str = ""


# ---------------------------------------------------------------------------
# IBKR whatIf cost/margin calibration (debloat/realism)
# ---------------------------------------------------------------------------


class CalibratedCostRoot(APIModel):
    """Per-root IBKR whatIf-calibrated fees/margins (USD per contract)."""

    broker_commission_per_contract: float = Field(ge=0.0)
    exchange_and_reg_fee_per_contract: float = Field(ge=0.0)
    initial_margin: float = Field(default=0.0, ge=0.0)
    maintenance_margin: float = Field(default=0.0, ge=0.0)
    commission_source: str = "ibkr_whatif"
    margin_source: str = "ibkr_whatif"


class CalibrateCostsRequest(StrictModel):
    """``POST /api/runner/calibrate-costs`` body."""

    roots: list[str] | None = None


class CalibrateCostsResponse(APIModel):
    """Result of an IBKR whatIf cost/margin calibration run."""

    status: Literal["calibrated", "partial", "unavailable"]
    account_id: str | None = None
    calibrated_at: datetime
    roots: dict[str, CalibratedCostRoot] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    path: str | None = None
    source: Literal["ibkr"] = "ibkr"


class CalibratedCostsStatusResponse(APIModel):
    """``GET /api/runner/calibrated-costs`` persisted-override status."""

    available: bool = False
    path: str | None = None
    account_id: str | None = None
    calibrated_at: datetime | None = None
    age_seconds: float | None = None
    roots: dict[str, CalibratedCostRoot] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    source: Literal["ibkr", "none"] = "none"


__all__ = [
    "CalibrateCostsRequest",
    "CalibrateCostsResponse",
    "CalibratedCostRoot",
    "CalibratedCostsStatusResponse",
    "IbkrActiveConnection",
    "IbkrConnectionSwitchRequest",
    "IbkrConnectionSwitchResponse",
    "IbkrPortInfo",
    "IbkrPortsResponse",
    "IbkrRequestedConnection",
    "RebalanceExecuteRequest",
    "RebalanceExecuteResponse",
    "RebalanceFillRow",
    "RebalanceOrderRow",
    "RebalancePositionRow",
    "RebalancePreviewRequest",
    "RebalancePreviewResponse",
    "RebalanceTargetRow",
]

type RebalanceMode = Literal["ibkr-paper", "offline"]


class RebalancePreviewRequest(StrictModel):
    """``POST /api/runner/rebalance-preview`` body."""

    strategy_id: str = Field(min_length=1, max_length=128)
    roots: list[str] | None = None
    dry_run: bool = True


class RebalanceExecuteRequest(StrictModel):
    """``POST /api/runner/rebalance-execute`` body."""

    strategy_id: str = Field(min_length=1, max_length=128)
    roots: list[str] | None = None
    #: Mandatory confirmation; ``false`` returns 400 (no orders are submitted).
    confirm: bool = True


class RebalancePositionRow(APIModel):
    """One current broker position."""

    instrument_id: str
    quantity: int
    average_price: float
    market_price: float
    multiplier: float
    signed_notional: float
    initial_margin: float
    contract_symbol: str | None = None


class RebalanceTargetRow(APIModel):
    """One discrete target contract leg produced by the translator."""

    instrument_id: str
    quantity: int
    price: float
    multiplier: float
    initial_margin: float
    signed_notional: float
    contract_symbol: str | None = None
    expiration: date | None = None


class RebalanceOrderRow(APIModel):
    """One proposed/executed delta order with its pre-trade gate verdict."""

    order_id: str
    symbol: str
    instrument_id: str
    side: Literal["BUY", "SELL"]
    quantity: int
    order_type: str = "MARKET"
    est_price: float | None = None
    est_margin_impact: float = 0.0
    pre_trade_gate_passed: bool = True
    rejection_reason: str | None = None


class RebalanceFillRow(APIModel):
    """One execution report returned by ``/rebalance-execute``."""

    order_id: str
    instrument_id: str
    side: Literal["BUY", "SELL"]
    quantity: int
    price: float
    commission: float = 0.0
    status: str = "FILLED"
    timestamp: datetime | None = None


class RebalancePreviewResponse(APIModel):
    """``POST /api/runner/rebalance-preview`` response (submits nothing)."""

    strategy_id: str
    account_nlv: float
    account_currency: str
    current_positions: list[RebalancePositionRow] = Field(default_factory=list)
    target_contracts: list[RebalanceTargetRow] = Field(default_factory=list)
    proposed_orders: list[RebalanceOrderRow] = Field(default_factory=list)
    projected_margin_utilization: float = 0.0
    margin_utilization_cap: float = 0.60
    tracking_error_pct: float = 0.0
    mode: RebalanceMode = "offline"
    generated_at: datetime
    strategy_roots: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class RebalanceExecuteResponse(APIModel):
    """``POST /api/runner/rebalance-execute`` response."""

    strategy_id: str
    mode: RebalanceMode = "offline"
    status: Literal["submitted", "no-op", "idempotent-replay"] = "submitted"
    submitted_order_ids: list[str] = Field(default_factory=list)
    orders: list[RebalanceOrderRow] = Field(default_factory=list)
    fills: list[RebalanceFillRow] = Field(default_factory=list)
    rejected_orders: list[RebalanceOrderRow] = Field(default_factory=list)
    account_nlv: float
    account_currency: str
    projected_margin_utilization: float = 0.0
    margin_utilization_cap: float = 0.60
    tracking_error_pct: float = 0.0
    updated_positions: list[RebalancePositionRow] = Field(default_factory=list)
    bridge_updated: bool = False
    message: str = ""
    generated_at: datetime
