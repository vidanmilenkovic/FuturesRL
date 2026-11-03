"""Durable runner <-> API bridge for live/paper/offline book telemetry.

The live/paper trading runner and the FastAPI controller are separate processes
(and the API may outlive any single runner). This module is the on-disk contract
between them: after every processed state the runner atomically publishes a
:class:`RunnerBridgeSnapshot` to ``data_dir/bridge/runner_state.json`` and the API
reads it, reports staleness and renders the book.

Design rules
------------
* **Versioned JSON** — :data:`BRIDGE_SCHEMA_VERSION` guards the payload; a newer
  payload is rejected (never silently mis-rendered by an older API) and a corrupt
  file is logged and treated as "no snapshot" (the dashboard shows "offline"
  rather than crashing).
* **Atomic publish** — snapshots are written to a temp file in the same directory,
  ``fsync``-ed and ``os.replace``-d, so a reader never observes a partial JSON
  document even while the runner restarts.
* **Staleness contract** — every snapshot carries ``stale_after_seconds``; the API
  calls :meth:`RunnerBridgeSnapshot.is_stale` to distinguish "runner silent" from
  "runner healthy".
* **Duck-typed extraction** — :func:`snapshot_from_runner` normalises both the
  Phase-5 IBKR OMS (``orders`` / ``open_order_ids`` / ``all_fills`` / async
  ``get_positions`` / ``get_account``) and
  :class:`src.backtest.simulated_broker.SimulatedBroker` (``working_orders`` /
  ``fills`` / ``get_positions_sync`` / ``net_liquidation``) without importing
  either: the runner is the only component that knows which venue it drives.
* **Bounded payload** — working orders and fills are capped at
  :data:`MAX_BRIDGE_ORDERS` / :data:`MAX_BRIDGE_FILLS` (most recent, chronological)
  so the JSON stays small enough for the dashboard to poll.
* **Full pipeline breakdown (Phase 10)** — alongside the flat ``pipeline`` scalar
  summary the snapshot carries ``instruments`` (one :class:`BridgePipelineRow`
  per root: base/timing/risk/final/actual weights, target contracts, notional,
  driver and regime), ``micro_routing`` (:class:`BridgeMicroRouting` pairs),
  ``tracking_error_pct`` and ``tracking_error_source`` (``translator`` when the
  :class:`~src.risk.contract_translator.ContractTranslator` discretization report
  was available, else ``fallback``). This is purely additive: payloads written by
  older runners still load (new fields default) and older readers ignore the new
  keys, so :data:`BRIDGE_SCHEMA_VERSION` stays ``1``.
* **No heavyweight imports** — this module never imports the runner, the OMS or
  the backtester, so the lazy ``src.execution.__init__`` stays runpy-warning free
  and the API can import it without pulling the IBKR stack in.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Literal, cast

from configs.settings import Settings, get_settings

from src.core.types import ScalarValue, ensure_utc

__all__ = [
    "BRIDGE_FILENAME",
    "BRIDGE_MODES",
    "BRIDGE_SCHEMA_VERSION",
    "DEFAULT_STALE_AFTER_SECONDS",
    "MAX_BRIDGE_FILLS",
    "MAX_BRIDGE_INSTRUMENTS",
    "MAX_BRIDGE_MICRO_ROUTES",
    "MAX_BRIDGE_ORDERS",
    "BridgeFill",
    "BridgeMicroRouting",
    "BridgePipelineRow",
    "BridgePosition",
    "BridgeSchemaError",
    "BridgeWorkingOrder",
    "RunnerBridgeSnapshot",
    "RunnerBridgeStore",
    "snapshot_from_runner",
]

logger = logging.getLogger(__name__)

#: Current on-disk payload schema version (bump on incompatible field changes).
BRIDGE_SCHEMA_VERSION: int = 1

#: File name used inside ``Settings.runner_bridge_dir``.
BRIDGE_FILENAME: str = "runner_state.json"

#: Runner modes the bridge contract understands.
type BridgeMode = Literal["offline", "paper", "live"]
BRIDGE_MODES: tuple[BridgeMode, ...] = ("offline", "paper", "live")

#: Default age after which a snapshot is considered stale by the API.
DEFAULT_STALE_AFTER_SECONDS: float = 15.0

#: Maximum number of in-flight orders / fills published in one snapshot.
MAX_BRIDGE_ORDERS: int = 200
MAX_BRIDGE_FILLS: int = 200

#: Maximum per-instrument pipeline rows / micro-routing pairs published.
MAX_BRIDGE_INSTRUMENTS: int = 200
MAX_BRIDGE_MICRO_ROUTES: int = 200

_SIDES: frozenset[str] = frozenset({"BUY", "SELL"})


class BridgeSchemaError(ValueError):
    """Raised when a bridge payload (or dataclass field) violates the contract."""


# ---------------------------------------------------------------------------
# Small validation helpers (JSON payloads arrive as ``Mapping[str, object]``)
# ---------------------------------------------------------------------------


def _require_str(payload: Mapping[str, object], key: str, *, context: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise BridgeSchemaError(f"{context}.{key} must be a non-empty string, got {value!r}")
    return value


def _require_int(payload: Mapping[str, object], key: str, *, context: str) -> int:
    value = payload.get(key)
    result = _as_int(value)
    if result is None:
        raise BridgeSchemaError(f"{context}.{key} must be an int, got {value!r}")
    return result


def _require_float(
    payload: Mapping[str, object],
    key: str,
    *,
    context: str,
    positive: bool = False,
) -> float:
    value = payload.get(key)
    result = _as_float(value)
    if result is None or (positive and result <= 0.0):
        raise BridgeSchemaError(f"{context}.{key} must be a finite number, got {value!r}")
    return result


def _require_datetime(payload: Mapping[str, object], key: str, *, context: str) -> datetime:
    value = payload.get(key)
    if isinstance(value, datetime):
        return ensure_utc(value)
    if not isinstance(value, str):
        raise BridgeSchemaError(f"{context}.{key} must be an ISO-8601 string, got {value!r}")
    try:
        return ensure_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
    except (ValueError, TypeError) as exc:
        raise BridgeSchemaError(f"{context}.{key} is not a valid ISO-8601 timestamp") from exc


def _optional_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return ensure_utc(value)
    if isinstance(value, str):
        try:
            return ensure_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
        except (ValueError, TypeError):
            return None
    return None


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_float(payload: Mapping[str, object], key: str) -> float | None:
    return _as_float(payload.get(key))


def _as_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return None


def _as_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _as_positive(value: object) -> float | None:
    result = _as_float(value)
    return result if result is not None and result > 0.0 else None


def _enum_text(value: object) -> str | None:
    """Enum (``StrEnum``/``Enum``) or plain string as text."""
    text = getattr(value, "value", None)
    if isinstance(text, str):
        return text
    return _optional_str(value)


def _side_text(value: object) -> str | None:
    text = _enum_text(value)
    if text is None:
        return None
    upper = text.strip().upper()
    return upper if upper in _SIDES else None


def _as_sequence(value: object) -> tuple[object, ...] | None:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return None
    return tuple(value)


def _as_iterable_sequence(value: object) -> tuple[object, ...] | None:
    """Like :func:`_as_sequence` but also accepts iterable-only containers.

    ``ContractTargetSet`` is iterable/sized but deliberately not a
    ``collections.abc.Sequence`` (it has no ``__getitem__``), so the pipeline
    breakdown extractors use this tolerant helper.
    """
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        return None
    if isinstance(value, Sequence):
        return tuple(value)
    if isinstance(value, Iterable):
        return tuple(value)
    return None


def _coerce_scalar(value: object) -> ScalarValue:
    """Coerce a telemetry-style value to a JSON scalar (``None`` when not scalar)."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, datetime):
        return value.isoformat()
    return None


def _scalar_mapping(value: object, *, name: str) -> dict[str, ScalarValue]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise BridgeSchemaError(f"{name} must be a mapping, got {type(value)!r}")
    result: dict[str, ScalarValue] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise BridgeSchemaError(f"{name} keys must be strings, got {key!r}")
        scalar = _coerce_scalar(item)
        if scalar is None and item is not None:
            raise BridgeSchemaError(f"{name}[{key!r}] must be a JSON scalar, got {item!r}")
        result[key] = scalar
    return result


def _float_mapping(value: object, *, name: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise BridgeSchemaError(f"{name} must be a mapping, got {type(value)!r}")
    result: dict[str, float] = {}
    for key, item in value.items():
        number = _as_float(item)
        if not isinstance(key, str) or number is None:
            raise BridgeSchemaError(f"{name}[{key!r}] must be a finite number, got {item!r}")
        result[key] = number
    return result


def _cap_latest(
    items: Sequence[Mapping[str, object]],
    *,
    limit: int,
    key: str = "updated_at",
) -> tuple[Mapping[str, object], ...]:
    """Return the most recent ``limit`` payloads, in chronological order."""
    if len(items) <= limit:
        return tuple(items)
    ordered = sorted(items, key=lambda item: str(item.get(key, "")))
    return tuple(ordered[-limit:])


# ---------------------------------------------------------------------------
# Bridge dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BridgePosition:
    """One open futures position as published on the bridge."""

    instrument_id: str
    contract_symbol: str | None
    quantity: int
    average_price: float
    market_price: float
    multiplier: float
    unrealized_pnl: float
    initial_margin: float

    def __post_init__(self) -> None:
        if not isinstance(self.instrument_id, str) or not self.instrument_id:
            raise BridgeSchemaError("position.instrument_id must be a non-empty string")
        if self.contract_symbol is not None and not isinstance(self.contract_symbol, str):
            raise BridgeSchemaError("position.contract_symbol must be a string or None")
        quantity = _as_int(self.quantity)
        if quantity is None:
            raise BridgeSchemaError(f"position.quantity must be an int, got {self.quantity!r}")
        object.__setattr__(self, "quantity", quantity)
        for name in ("average_price", "market_price", "multiplier"):
            value = _as_positive(getattr(self, name))
            if value is None:
                raise BridgeSchemaError(
                    f"position.{name} must be positive, got {getattr(self, name)!r}"
                )
            object.__setattr__(self, name, value)
        unrealized = _as_float(self.unrealized_pnl)
        if unrealized is None:
            raise BridgeSchemaError("position.unrealized_pnl must be a finite number")
        object.__setattr__(self, "unrealized_pnl", unrealized)
        margin = _as_float(self.initial_margin)
        if margin is None or margin < 0.0:
            raise BridgeSchemaError("position.initial_margin must be finite and non-negative")
        object.__setattr__(self, "initial_margin", margin)

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable representation."""
        return {
            "instrument_id": self.instrument_id,
            "contract_symbol": self.contract_symbol,
            "quantity": self.quantity,
            "average_price": self.average_price,
            "market_price": self.market_price,
            "multiplier": self.multiplier,
            "unrealized_pnl": self.unrealized_pnl,
            "initial_margin": self.initial_margin,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> BridgePosition:
        """Parse a bridge position payload."""
        if not isinstance(payload, Mapping):
            raise BridgeSchemaError("position payload must be a mapping")
        return cls(
            instrument_id=_require_str(payload, "instrument_id", context="position"),
            contract_symbol=_optional_str(payload.get("contract_symbol")),
            quantity=_require_int(payload, "quantity", context="position"),
            average_price=_require_float(
                payload, "average_price", context="position", positive=True
            ),
            market_price=_require_float(payload, "market_price", context="position", positive=True),
            multiplier=_require_float(payload, "multiplier", context="position", positive=True),
            unrealized_pnl=_require_float(payload, "unrealized_pnl", context="position"),
            initial_margin=_require_float(payload, "initial_margin", context="position"),
        )


@dataclass(frozen=True, slots=True)
class BridgeWorkingOrder:
    """One in-flight order as published on the bridge.

    ``quantity`` / ``filled_quantity`` / ``remaining_quantity`` are absolute
    quantities; direction is carried by ``side`` (``"BUY"``/``"SELL"``).
    """

    order_id: str
    instrument_id: str
    contract_symbol: str | None
    side: str
    quantity: int
    filled_quantity: int
    remaining_quantity: int
    order_type: str
    limit_price: float | None
    status: str
    updated_at: datetime

    def __post_init__(self) -> None:
        for name in ("order_id", "instrument_id", "order_type", "status"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise BridgeSchemaError(f"working_order.{name} must be a non-empty string")
        if self.contract_symbol is not None and not isinstance(self.contract_symbol, str):
            raise BridgeSchemaError("working_order.contract_symbol must be a string or None")
        side = _side_text(self.side)
        if side is None:
            raise BridgeSchemaError("working_order.side must be 'BUY' or 'SELL'")
        object.__setattr__(self, "side", side)
        quantity = _as_int(self.quantity)
        filled = _as_int(self.filled_quantity)
        remaining = _as_int(self.remaining_quantity)
        if quantity is None or quantity <= 0:
            raise BridgeSchemaError("working_order.quantity must be a positive int")
        if filled is None or filled < 0 or filled > quantity:
            raise BridgeSchemaError("working_order.filled_quantity must be within [0, quantity]")
        if remaining is None or remaining < 0 or remaining > quantity - filled + 0:
            # A working order keeps a positive remainder; tolerate off-by-fill reports.
            remaining = quantity - filled
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "filled_quantity", filled)
        object.__setattr__(self, "remaining_quantity", remaining)
        if remaining == 0:
            raise BridgeSchemaError("working_order.remaining_quantity must be positive")
        if self.limit_price is not None:
            limit = _as_positive(self.limit_price)
            if limit is None:
                raise BridgeSchemaError("working_order.limit_price must be positive or None")
            object.__setattr__(self, "limit_price", limit)
        object.__setattr__(self, "updated_at", ensure_utc(self.updated_at))

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable representation."""
        return {
            "order_id": self.order_id,
            "instrument_id": self.instrument_id,
            "contract_symbol": self.contract_symbol,
            "side": self.side,
            "quantity": self.quantity,
            "filled_quantity": self.filled_quantity,
            "remaining_quantity": self.remaining_quantity,
            "order_type": self.order_type,
            "limit_price": self.limit_price,
            "status": self.status,
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> BridgeWorkingOrder:
        """Parse a bridge working-order payload."""
        if not isinstance(payload, Mapping):
            raise BridgeSchemaError("working_order payload must be a mapping")
        return cls(
            order_id=_require_str(payload, "order_id", context="working_order"),
            instrument_id=_require_str(payload, "instrument_id", context="working_order"),
            contract_symbol=_optional_str(payload.get("contract_symbol")),
            side=_require_str(payload, "side", context="working_order"),
            quantity=_require_int(payload, "quantity", context="working_order"),
            filled_quantity=_require_int(payload, "filled_quantity", context="working_order"),
            remaining_quantity=_require_int(payload, "remaining_quantity", context="working_order"),
            order_type=_require_str(payload, "order_type", context="working_order"),
            limit_price=_optional_float(payload, "limit_price"),
            status=_require_str(payload, "status", context="working_order"),
            updated_at=_require_datetime(payload, "updated_at", context="working_order"),
        )


@dataclass(frozen=True, slots=True)
class BridgeFill:
    """One execution report as published on the bridge."""

    order_id: str
    instrument_id: str
    side: str
    quantity: int
    price: float
    commission: float
    slippage_ticks: float
    timestamp: datetime

    def __post_init__(self) -> None:
        for name in ("order_id", "instrument_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise BridgeSchemaError(f"fill.{name} must be a non-empty string")
        side = _side_text(self.side)
        if side is None:
            raise BridgeSchemaError("fill.side must be 'BUY' or 'SELL'")
        object.__setattr__(self, "side", side)
        quantity = _as_int(self.quantity)
        if quantity is None or quantity <= 0:
            raise BridgeSchemaError("fill.quantity must be a positive int")
        object.__setattr__(self, "quantity", quantity)
        price = _as_positive(self.price)
        if price is None:
            raise BridgeSchemaError("fill.price must be a finite strictly-positive number")
        object.__setattr__(self, "price", price)
        for name in ("commission", "slippage_ticks"):
            value = _as_float(getattr(self, name))
            if value is None:
                raise BridgeSchemaError(f"fill.{name} must be a finite number")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "timestamp", ensure_utc(self.timestamp))

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable representation."""
        return {
            "order_id": self.order_id,
            "instrument_id": self.instrument_id,
            "side": self.side,
            "quantity": self.quantity,
            "price": self.price,
            "commission": self.commission,
            "slippage_ticks": self.slippage_ticks,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> BridgeFill:
        """Parse a bridge fill payload."""
        if not isinstance(payload, Mapping):
            raise BridgeSchemaError("fill payload must be a mapping")
        return cls(
            order_id=_require_str(payload, "order_id", context="fill"),
            instrument_id=_require_str(payload, "instrument_id", context="fill"),
            side=_require_str(payload, "side", context="fill"),
            quantity=_require_int(payload, "quantity", context="fill"),
            price=_require_float(payload, "price", context="fill", positive=True),
            commission=_require_float(payload, "commission", context="fill"),
            slippage_ticks=_require_float(payload, "slippage_ticks", context="fill"),
            timestamp=_require_datetime(payload, "timestamp", context="fill"),
        )


@dataclass(frozen=True, slots=True)
class BridgePipelineRow:
    """One per-instrument weight trace from the latest pipeline cycle.

    The field names intentionally mirror the frozen API waterfall contract
    (:class:`src.api.schemas.PipelineInstrumentRow`) so the API can render the
    Weight matrix from the bridge row dicts almost verbatim; ``root``, ``driver``
    and ``regime`` are additive bridge-only context (``reason`` is serialised as
    the API-compatible alias of ``driver``).

    ``weight_actual`` is the weight actually achieved by the discrete contract
    book (signed notional / equity) when a discretization report/fallback is
    available, else ``None``. ``notional`` is the target contract notional (or
    ``w_final * equity`` when no contract target exists).
    """

    instrument_id: str
    root: str
    asset_class: str
    selected: bool
    weight_base: float
    weight_timing: float
    weight_risk: float
    weight_final: float
    weight_actual: float | None
    delta_timing: float
    delta_risk: float
    target_contracts: int | None
    notional: float | None
    driver: str
    regime: str | None

    def __post_init__(self) -> None:
        for name in ("instrument_id", "root", "asset_class", "driver"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise BridgeSchemaError(f"pipeline_row.{name} must be a non-empty string")
        if not isinstance(self.selected, bool):
            raise BridgeSchemaError("pipeline_row.selected must be a bool")
        for name in (
            "weight_base",
            "weight_timing",
            "weight_risk",
            "weight_final",
            "delta_timing",
            "delta_risk",
        ):
            value = _as_float(getattr(self, name))
            if value is None:
                raise BridgeSchemaError(f"pipeline_row.{name} must be a finite number")
            object.__setattr__(self, name, value)
        if self.weight_actual is not None:
            actual = _as_float(self.weight_actual)
            if actual is None:
                raise BridgeSchemaError("pipeline_row.weight_actual must be finite or None")
            object.__setattr__(self, "weight_actual", actual)
        if self.target_contracts is not None:
            contracts = _as_int(self.target_contracts)
            if contracts is None:
                raise BridgeSchemaError("pipeline_row.target_contracts must be an int or None")
            object.__setattr__(self, "target_contracts", contracts)
        if self.notional is not None:
            notional = _as_float(self.notional)
            if notional is None:
                raise BridgeSchemaError("pipeline_row.notional must be finite or None")
            object.__setattr__(self, "notional", notional)
        if self.regime is not None and not isinstance(self.regime, str):
            raise BridgeSchemaError("pipeline_row.regime must be a string or None")

    @property
    def reason(self) -> str:
        """API-compatible alias of :attr:`driver`."""
        return self.driver

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable representation (API-compatible keys)."""
        return {
            "instrument_id": self.instrument_id,
            "root": self.root,
            "asset_class": self.asset_class,
            "selected": self.selected,
            "weight_base": round(self.weight_base, 8),
            "weight_timing": round(self.weight_timing, 8),
            "weight_risk": round(self.weight_risk, 8),
            "weight_final": round(self.weight_final, 8),
            "weight_actual": (
                round(self.weight_actual, 8) if self.weight_actual is not None else None
            ),
            "delta_timing": round(self.delta_timing, 8),
            "delta_risk": round(self.delta_risk, 8),
            "target_contracts": self.target_contracts,
            "notional": round(self.notional, 2) if self.notional is not None else None,
            "driver": self.driver,
            "reason": self.driver,
            "regime": self.regime,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> BridgePipelineRow:
        """Parse a bridge pipeline row (missing optional fields degrade gracefully)."""
        if not isinstance(payload, Mapping):
            raise BridgeSchemaError("pipeline_row payload must be a mapping")
        instrument_id = _require_str(payload, "instrument_id", context="pipeline_row")
        weight_base = _require_float(payload, "weight_base", context="pipeline_row")
        weight_timing = _require_float(payload, "weight_timing", context="pipeline_row")
        weight_risk = _require_float(payload, "weight_risk", context="pipeline_row")
        weight_final = _require_float(payload, "weight_final", context="pipeline_row")
        raw_selected = payload.get("selected", True)
        raw_actual = payload.get("weight_actual")
        raw_contracts = payload.get("target_contracts")
        raw_notional = payload.get("notional")
        driver = _optional_str(payload.get("driver")) or _optional_str(payload.get("reason"))
        return cls(
            instrument_id=instrument_id,
            root=_optional_str(payload.get("root")) or instrument_id,
            asset_class=_optional_str(payload.get("asset_class")) or "unknown",
            selected=bool(raw_selected),
            weight_base=weight_base,
            weight_timing=weight_timing,
            weight_risk=weight_risk,
            weight_final=weight_final,
            weight_actual=None
            if raw_actual is None
            else _require_float(payload, "weight_actual", context="pipeline_row"),
            delta_timing=(
                _require_float(payload, "delta_timing", context="pipeline_row")
                if payload.get("delta_timing") is not None
                else weight_timing - weight_base
            ),
            delta_risk=(
                _require_float(payload, "delta_risk", context="pipeline_row")
                if payload.get("delta_risk") is not None
                else weight_risk - weight_timing
            ),
            target_contracts=(
                None
                if raw_contracts is None
                else _require_int(payload, "target_contracts", context="pipeline_row")
            ),
            notional=(
                None
                if raw_notional is None
                else _require_float(payload, "notional", context="pipeline_row")
            ),
            driver=driver or "passed through unchanged",
            regime=_optional_str(payload.get("regime")),
        )


@dataclass(frozen=True, slots=True)
class BridgeMicroRouting:
    """One achieved standard/micro contract routing pair from the translator."""

    family: str
    standard_root: str
    standard_quantity: int
    micro_root: str
    micro_quantity: int

    def __post_init__(self) -> None:
        for name in ("family", "standard_root", "micro_root"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise BridgeSchemaError(f"micro_routing.{name} must be a non-empty string")
        for name in ("standard_quantity", "micro_quantity"):
            value = _as_int(getattr(self, name))
            if value is None:
                raise BridgeSchemaError(f"micro_routing.{name} must be an int")
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable representation (API ``MicroRoutingEntry`` keys)."""
        return {
            "family": self.family,
            "standard_root": self.standard_root,
            "standard_quantity": self.standard_quantity,
            "micro_root": self.micro_root,
            "micro_quantity": self.micro_quantity,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> BridgeMicroRouting:
        """Parse a bridge micro-routing payload."""
        if not isinstance(payload, Mapping):
            raise BridgeSchemaError("micro_routing payload must be a mapping")
        standard_root = _require_str(payload, "standard_root", context="micro_routing")
        return cls(
            family=_optional_str(payload.get("family")) or standard_root,
            standard_root=standard_root,
            standard_quantity=(
                _require_int(payload, "standard_quantity", context="micro_routing")
                if payload.get("standard_quantity") is not None
                else 0
            ),
            micro_root=_require_str(payload, "micro_root", context="micro_routing"),
            micro_quantity=(
                _require_int(payload, "micro_quantity", context="micro_routing")
                if payload.get("micro_quantity") is not None
                else 0
            ),
        )


type _PayloadParser = Callable[[Mapping[str, object]], object]


def _sequence_payload(value: object, *, name: str, parser: _PayloadParser) -> tuple[object, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise BridgeSchemaError(f"{name} must be a JSON array")
    parsed: list[object] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise BridgeSchemaError(f"{name} entries must be mappings")
        parsed.append(parser(cast("Mapping[str, object]", item)))
    return tuple(parsed)


@dataclass(frozen=True, slots=True)
class RunnerBridgeSnapshot:
    """A complete, versioned view of the runner book for the API/dashboard.

    Attributes
    ----------
    schema_version:
        On-disk payload version (:data:`BRIDGE_SCHEMA_VERSION`).
    runner_id:
        Identifier of the publishing runner process (stable for its lifetime).
    mode:
        ``offline`` / ``paper`` / ``live`` execution mode.
    sequence:
        Monotonically increasing publish counter (continues across restarts).
    updated_at:
        Publication timestamp (timezone-aware UTC).
    stale_after_seconds:
        Age after which the API must consider the snapshot stale.
    status:
        Flattened :class:`src.execution.runner.RunnerStatus` scalars.
    account:
        Account valuation/margin scalars (``None`` when unavailable).
    pipeline:
        Latest pipeline/translation summary scalars (``None`` before the first run).
    positions:
        Open positions, sorted by instrument id.
    working_orders:
        In-flight orders (most recent :data:`MAX_BRIDGE_ORDERS`).
    fills:
        Execution reports (most recent :data:`MAX_BRIDGE_FILLS`).
    instruments:
        Full per-instrument pipeline weight matrix rows (Phase 10, additive).
    micro_routing:
        Achieved standard -> micro contract routing pairs (Phase 10, additive).
    tracking_error_pct:
        Discretization tracking error in percent (``None`` when unavailable).
    tracking_error_source:
        ``translator`` when the real discretization report was used, else
        ``fallback`` (arithmetic estimate from the discrete book).
    """

    schema_version: int
    runner_id: str
    mode: BridgeMode
    sequence: int
    updated_at: datetime
    stale_after_seconds: float
    status: Mapping[str, ScalarValue] = field(default_factory=dict)
    account: Mapping[str, float] | None = None
    pipeline: Mapping[str, ScalarValue] | None = None
    positions: tuple[BridgePosition, ...] = ()
    working_orders: tuple[BridgeWorkingOrder, ...] = ()
    fills: tuple[BridgeFill, ...] = ()
    instruments: tuple[BridgePipelineRow, ...] = ()
    micro_routing: tuple[BridgeMicroRouting, ...] = ()
    tracking_error_pct: float | None = None
    tracking_error_source: str = "fallback"

    def __post_init__(self) -> None:
        version = _as_int(self.schema_version)
        if version is None or version != BRIDGE_SCHEMA_VERSION:
            raise BridgeSchemaError(
                f"snapshot.schema_version must be {BRIDGE_SCHEMA_VERSION}, got {self.schema_version!r}"
            )
        if not isinstance(self.runner_id, str) or not self.runner_id:
            raise BridgeSchemaError("snapshot.runner_id must be a non-empty string")
        if self.mode not in BRIDGE_MODES:
            raise BridgeSchemaError(
                f"snapshot.mode must be one of {list(BRIDGE_MODES)}, got {self.mode!r}"
            )
        sequence = _as_int(self.sequence)
        if sequence is None or sequence < 0:
            raise BridgeSchemaError("snapshot.sequence must be a non-negative int")
        object.__setattr__(self, "sequence", sequence)
        object.__setattr__(self, "updated_at", ensure_utc(self.updated_at))
        stale_after = _as_float(self.stale_after_seconds)
        if stale_after is None or stale_after <= 0.0:
            raise BridgeSchemaError("snapshot.stale_after_seconds must be strictly positive")
        object.__setattr__(self, "stale_after_seconds", stale_after)
        object.__setattr__(self, "status", MappingProxyType(dict(self.status)))
        if self.account is not None:
            account = _float_mapping(self.account, name="snapshot.account")
            object.__setattr__(self, "account", MappingProxyType(account))
        if self.pipeline is not None:
            pipeline = _scalar_mapping(self.pipeline, name="snapshot.pipeline")
            object.__setattr__(self, "pipeline", MappingProxyType(pipeline))
        positions = tuple(self.positions)
        working_orders = tuple(self.working_orders)
        fills = tuple(self.fills)
        for position in positions:
            if not isinstance(position, BridgePosition):
                raise BridgeSchemaError("snapshot.positions entries must be BridgePosition")
        for order in working_orders:
            if not isinstance(order, BridgeWorkingOrder):
                raise BridgeSchemaError(
                    "snapshot.working_orders entries must be BridgeWorkingOrder"
                )
        for fill in fills:
            if not isinstance(fill, BridgeFill):
                raise BridgeSchemaError("snapshot.fills entries must be BridgeFill")
        if len(working_orders) > MAX_BRIDGE_ORDERS:
            raise BridgeSchemaError(
                f"snapshot.working_orders exceeds the {MAX_BRIDGE_ORDERS} entry cap"
            )
        if len(fills) > MAX_BRIDGE_FILLS:
            raise BridgeSchemaError(f"snapshot.fills exceeds the {MAX_BRIDGE_FILLS} entry cap")
        instruments = tuple(self.instruments)
        micro_routing = tuple(self.micro_routing)
        for row in instruments:
            if not isinstance(row, BridgePipelineRow):
                raise BridgeSchemaError("snapshot.instruments entries must be BridgePipelineRow")
        for route in micro_routing:
            if not isinstance(route, BridgeMicroRouting):
                raise BridgeSchemaError("snapshot.micro_routing entries must be BridgeMicroRouting")
        if len(instruments) > MAX_BRIDGE_INSTRUMENTS:
            raise BridgeSchemaError(
                f"snapshot.instruments exceeds the {MAX_BRIDGE_INSTRUMENTS} entry cap"
            )
        if len(micro_routing) > MAX_BRIDGE_MICRO_ROUTES:
            raise BridgeSchemaError(
                f"snapshot.micro_routing exceeds the {MAX_BRIDGE_MICRO_ROUTES} entry cap"
            )
        if self.tracking_error_pct is not None:
            tracking = _as_float(self.tracking_error_pct)
            if tracking is None or tracking < 0.0:
                raise BridgeSchemaError(
                    "snapshot.tracking_error_pct must be a finite non-negative number or None"
                )
            object.__setattr__(self, "tracking_error_pct", tracking)
        if self.tracking_error_source not in ("translator", "fallback"):
            raise BridgeSchemaError(
                "snapshot.tracking_error_source must be 'translator' or 'fallback'"
            )
        object.__setattr__(
            self, "positions", tuple(sorted(positions, key=lambda item: item.instrument_id))
        )
        object.__setattr__(self, "working_orders", working_orders)
        object.__setattr__(self, "fills", fills)
        object.__setattr__(
            self, "instruments", tuple(sorted(instruments, key=lambda row: row.instrument_id))
        )
        object.__setattr__(self, "micro_routing", micro_routing)

    # -- staleness -------------------------------------------------------------

    def is_stale(self, now: datetime | None = None) -> bool:
        """Whether the snapshot is older than its ``stale_after_seconds`` budget."""
        moment = datetime.now(UTC) if now is None else ensure_utc(now)
        return (moment - self.updated_at) > timedelta(seconds=self.stale_after_seconds)

    # -- serialisation ---------------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable representation (all timestamps ISO-8601 UTC)."""
        return {
            "schema_version": self.schema_version,
            "runner_id": self.runner_id,
            "mode": self.mode,
            "sequence": self.sequence,
            "updated_at": self.updated_at.isoformat(),
            "stale_after_seconds": self.stale_after_seconds,
            "status": dict(self.status),
            "account": dict(self.account) if self.account is not None else None,
            "pipeline": dict(self.pipeline) if self.pipeline is not None else None,
            "positions": [position.to_dict() for position in self.positions],
            "working_orders": [order.to_dict() for order in self.working_orders],
            "fills": [fill.to_dict() for fill in self.fills],
            "instruments": [row.to_dict() for row in self.instruments],
            "micro_routing": [route.to_dict() for route in self.micro_routing],
            "tracking_error_pct": self.tracking_error_pct,
            "tracking_error_source": self.tracking_error_source,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> RunnerBridgeSnapshot:
        """Parse a persisted bridge payload (rejecting newer schema versions)."""
        if not isinstance(payload, Mapping):
            raise BridgeSchemaError(f"snapshot payload must be a mapping, got {type(payload)!r}")
        version = _as_int(payload.get("schema_version"))
        if version is None:
            raise BridgeSchemaError("snapshot.schema_version is missing or not an int")
        if version > BRIDGE_SCHEMA_VERSION:
            raise BridgeSchemaError(
                f"snapshot schema_version {version} is newer than supported {BRIDGE_SCHEMA_VERSION}"
            )
        mode = payload.get("mode")
        if mode not in BRIDGE_MODES:
            raise BridgeSchemaError(
                f"snapshot.mode must be one of {list(BRIDGE_MODES)}, got {mode!r}"
            )
        raw_account = payload.get("account")
        account: Mapping[str, float] | None = None
        if raw_account is not None:
            account = _float_mapping(raw_account, name="snapshot.account")
        raw_pipeline = payload.get("pipeline")
        pipeline: Mapping[str, ScalarValue] | None = None
        if raw_pipeline is not None:
            pipeline = _scalar_mapping(raw_pipeline, name="snapshot.pipeline")
        return cls(
            schema_version=version,
            runner_id=_require_str(payload, "runner_id", context="snapshot"),
            mode=mode,
            sequence=_require_int(payload, "sequence", context="snapshot"),
            updated_at=_require_datetime(payload, "updated_at", context="snapshot"),
            stale_after_seconds=_require_float(
                payload, "stale_after_seconds", context="snapshot", positive=True
            ),
            status=_scalar_mapping(payload.get("status", {}), name="snapshot.status"),
            account=account,
            pipeline=pipeline,
            positions=cast(
                "tuple[BridgePosition, ...]",
                _sequence_payload(
                    payload.get("positions", ()), name="positions", parser=BridgePosition.from_dict
                ),
            ),
            working_orders=cast(
                "tuple[BridgeWorkingOrder, ...]",
                _sequence_payload(
                    payload.get("working_orders", ()),
                    name="working_orders",
                    parser=BridgeWorkingOrder.from_dict,
                ),
            ),
            fills=cast(
                "tuple[BridgeFill, ...]",
                _sequence_payload(
                    payload.get("fills", ()), name="fills", parser=BridgeFill.from_dict
                ),
            ),
            # Phase-10 additive fields: absent in v1 snapshots written by older
            # runners, so they default to empty/None and keep loading.
            instruments=cast(
                "tuple[BridgePipelineRow, ...]",
                _sequence_payload(
                    payload.get("instruments", ()),
                    name="instruments",
                    parser=BridgePipelineRow.from_dict,
                ),
            ),
            micro_routing=cast(
                "tuple[BridgeMicroRouting, ...]",
                _sequence_payload(
                    payload.get("micro_routing", ()),
                    name="micro_routing",
                    parser=BridgeMicroRouting.from_dict,
                ),
            ),
            tracking_error_pct=_optional_float(payload, "tracking_error_pct"),
            tracking_error_source=_optional_str(payload.get("tracking_error_source")) or "fallback",
        )


# ---------------------------------------------------------------------------
# Durable store
# ---------------------------------------------------------------------------


class RunnerBridgeStore:
    """Atomic JSON store for the runner bridge snapshot.

    Parameters
    ----------
    settings:
        Runtime settings; ``None`` uses the process-wide settings singleton.
    path:
        Explicit target (file path, or a directory to append
        :data:`BRIDGE_FILENAME` to). Defaults to
        :meth:`default_path` (``<data_dir>/bridge/runner_state.json``).
    stale_after_seconds:
        Default staleness budget stamped into snapshots created by callers that
        use this store (also exposed for the API).
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        path: Path | str | None = None,
        stale_after_seconds: float = DEFAULT_STALE_AFTER_SECONDS,
    ) -> None:
        resolved = settings if settings is not None else get_settings()
        if not isinstance(resolved, Settings):
            raise TypeError(f"settings must be a Settings, got {type(resolved)!r}")
        stale = _as_float(stale_after_seconds)
        if stale is None or stale <= 0.0:
            raise ValueError("stale_after_seconds must be a finite strictly-positive number")
        target = Path(path) if path is not None else self.default_path(resolved)
        if target.suffix == "":
            target = target / BRIDGE_FILENAME
        self._settings = resolved
        self._path = target
        self._stale_after_seconds = stale

    # -- introspection ---------------------------------------------------------

    @staticmethod
    def default_path(settings: Settings) -> Path:
        """Canonical snapshot path for ``settings`` (``data_dir/bridge/runner_state.json``)."""
        return Path(settings.runner_bridge_dir) / BRIDGE_FILENAME

    @property
    def settings(self) -> Settings:
        """Settings the store was resolved from."""
        return self._settings

    @property
    def path(self) -> Path:
        """Target JSON path."""
        return self._path

    @property
    def stale_after_seconds(self) -> float:
        """Default staleness budget stamped into published snapshots."""
        return self._stale_after_seconds

    # -- persistence -----------------------------------------------------------

    def save(self, snapshot: RunnerBridgeSnapshot) -> Path:
        """Atomically persist ``snapshot`` (temp file + ``fsync`` + ``os.replace``)."""
        if not isinstance(snapshot, RunnerBridgeSnapshot):
            raise TypeError(f"snapshot must be a RunnerBridgeSnapshot, got {type(snapshot)!r}")
        serialised = json.dumps(snapshot.to_dict(), sort_keys=True, indent=2)
        target = self._path
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(serialised)
                handle.flush()
                os.fsync(handle.fileno())
            Path(temporary_name).replace(target)
            self._fsync_directory(target.parent)
        except BaseException:
            with contextlib.suppress(OSError):
                Path(temporary_name).unlink(missing_ok=True)
            raise
        return target

    def load(self) -> RunnerBridgeSnapshot | None:
        """Read the published snapshot, or ``None`` when absent/corrupt."""
        if not self._path.is_file():
            return None
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                raise BridgeSchemaError("runner bridge payload must be a JSON object")
            return RunnerBridgeSnapshot.from_dict(cast("Mapping[str, object]", payload))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            logger.error(
                "runner bridge snapshot %s is corrupt (%s: %s); ignoring it",
                self._path,
                type(exc).__name__,
                exc,
            )
            return None

    def load_fresh(self) -> tuple[RunnerBridgeSnapshot | None, bool]:
        """Load the snapshot plus its staleness (``(None, False)`` when absent)."""
        snapshot = self.load()
        if snapshot is None:
            return (None, False)
        return (snapshot, snapshot.is_stale())

    def clear(self) -> None:
        """Delete the published snapshot (no-op when absent)."""
        with contextlib.suppress(FileNotFoundError):
            self._path.unlink()

    # -- internals -------------------------------------------------------------

    @staticmethod
    def _fsync_directory(directory: Path) -> None:
        """Best-effort directory fsync so the rename survives a crash."""
        try:
            descriptor = os.open(directory, os.O_RDONLY)
        except OSError:  # pragma: no cover - platform dependent
            return
        try:
            os.fsync(descriptor)
        except OSError:  # pragma: no cover - platform dependent
            pass
        finally:
            os.close(descriptor)


# ---------------------------------------------------------------------------
# Duck-typed extraction from the runner / broker
# ---------------------------------------------------------------------------


def _position_from_object(value: object) -> BridgePosition | None:
    """Normalise a broker/OMS position object into :class:`BridgePosition`."""
    instrument_id = getattr(value, "instrument_id", None)
    if not isinstance(instrument_id, str) or not instrument_id:
        return None
    quantity = _as_int(getattr(value, "quantity", None))
    if quantity is None:
        return None
    market_price = _as_positive(getattr(value, "market_price", None))
    if market_price is None:
        return None
    average_price = _as_positive(getattr(value, "average_price", None)) or market_price
    multiplier = _as_positive(getattr(value, "multiplier", None)) or 1.0
    unrealized = _as_float(getattr(value, "unrealized_pnl", None))
    if unrealized is None:
        unrealized = (market_price - average_price) * float(quantity) * multiplier
    initial_margin = _as_float(getattr(value, "initial_margin_required", None))
    if initial_margin is None:
        per_contract = _as_float(getattr(value, "initial_margin", None)) or 0.0
        initial_margin = abs(float(quantity)) * per_contract
    contract_symbol = _optional_str(getattr(value, "contract_symbol", None))
    return BridgePosition(
        instrument_id=instrument_id,
        contract_symbol=contract_symbol,
        quantity=quantity,
        average_price=average_price,
        market_price=market_price,
        multiplier=multiplier,
        unrealized_pnl=unrealized,
        initial_margin=max(initial_margin, 0.0),
    )


def _working_order_from_managed(value: object, order_id: str) -> BridgeWorkingOrder | None:
    event = getattr(value, "event", None)
    if event is None:
        return None
    return _working_order_from_event(event, order_id=order_id, managed=value)


def _working_order_from_event(
    event: object,
    *,
    order_id: str | None = None,
    managed: object | None = None,
) -> BridgeWorkingOrder | None:
    resolved_order_id = order_id or _optional_str(getattr(event, "order_id", None))
    if resolved_order_id is None:
        return None
    instrument_id = _optional_str(getattr(event, "instrument_id", None))
    side = _side_text(getattr(event, "side", None))
    quantity = _as_int(getattr(event, "quantity", None))
    if instrument_id is None or side is None or quantity is None or quantity <= 0:
        return None
    filled = 0
    status = "WORKING"
    receipt: object | None = None
    if managed is not None:
        filled = max(_as_int(getattr(managed, "filled_quantity", None)) or 0, 0)
        receipt = getattr(managed, "receipt", None)
        status_text = _enum_text(getattr(managed, "status", None))
        if status_text is not None:
            status = status_text
    remaining = quantity - min(filled, quantity)
    if remaining <= 0:
        return None
    limit_price = _as_positive(getattr(event, "limit_price", None))
    order_type = _enum_text(getattr(event, "order_type", None)) or "MARKET"
    contract_symbol = _optional_str(getattr(event, "contract_symbol", None))
    updated_at = _optional_datetime(getattr(receipt, "timestamp", None))
    if updated_at is None:
        updated_at = _optional_datetime(getattr(event, "timestamp", None)) or datetime.now(UTC)
    return BridgeWorkingOrder(
        order_id=resolved_order_id,
        instrument_id=instrument_id,
        contract_symbol=contract_symbol,
        side=side,
        quantity=quantity,
        filled_quantity=min(filled, quantity),
        remaining_quantity=remaining,
        order_type=order_type,
        limit_price=limit_price,
        status=status,
        updated_at=updated_at,
    )


def _working_orders_from_broker(broker: object) -> list[BridgeWorkingOrder]:
    """Extract in-flight orders from an OMS (``orders``) or a simulated broker."""
    orders = getattr(broker, "orders", None)
    if isinstance(orders, Mapping):
        open_order_ids = getattr(broker, "open_order_ids", None)
        candidate_ids: tuple[str, ...]
        if isinstance(open_order_ids, Iterable) and not isinstance(open_order_ids, (str, bytes)):
            candidate_ids = tuple(str(order_id) for order_id in open_order_ids)
        else:
            candidate_ids = tuple(
                str(order_id)
                for order_id, managed in orders.items()
                if not bool(getattr(managed, "is_terminal", False))
            )
        result: list[BridgeWorkingOrder] = []
        for order_id in candidate_ids:
            managed = orders.get(order_id)
            if managed is None:
                continue
            item = _working_order_from_managed(managed, order_id)
            if item is not None:
                result.append(item)
        return result
    events = getattr(broker, "working_orders", None)
    sequence = _as_sequence(events)
    if sequence is None:
        return []
    result = []
    for event in sequence:
        item = _working_order_from_event(event)
        if item is not None:
            result.append(item)
    return result


def _fill_from_object(value: object) -> BridgeFill | None:
    order_id = _optional_str(getattr(value, "order_id", None))
    instrument_id = _optional_str(getattr(value, "instrument_id", None))
    side = _side_text(getattr(value, "side", None))
    quantity = _as_int(getattr(value, "quantity", None))
    price = _as_positive(getattr(value, "price", None))
    timestamp = _optional_datetime(getattr(value, "timestamp", None))
    if (
        order_id is None
        or instrument_id is None
        or side is None
        or quantity is None
        or quantity <= 0
        or price is None
        or timestamp is None
    ):
        return None
    commission = _as_float(getattr(value, "commission", None))
    if commission is None:
        commission = _as_float(getattr(value, "commissions", None)) or 0.0
    slippage = _as_float(getattr(value, "slippage_ticks", None))
    if slippage is None:
        slippage = _as_float(getattr(value, "execution_cost_ticks", None)) or 0.0
    return BridgeFill(
        order_id=order_id,
        instrument_id=instrument_id,
        side=side,
        quantity=quantity,
        price=price,
        commission=commission,
        slippage_ticks=slippage,
        timestamp=timestamp,
    )


def _fills_from_broker(broker: object) -> list[BridgeFill]:
    """Extract execution reports from an OMS or a simulated broker."""
    raw_items: list[object] = []
    all_fills = getattr(broker, "all_fills", None)
    if callable(all_fills):
        outcome = all_fills()
        sequence = _as_sequence(outcome)
        if sequence is not None:
            raw_items.extend(sequence)
    orders = getattr(broker, "orders", None)
    if not raw_items and isinstance(orders, Mapping):
        for managed in orders.values():
            managed_fills = _as_sequence(getattr(managed, "fills", None))
            if managed_fills is not None:
                raw_items.extend(managed_fills)
    if not raw_items:
        broker_fills = _as_sequence(getattr(broker, "fills", None))
        if broker_fills is not None:
            for item in broker_fills:
                converter = getattr(item, "to_core_fill", None)
                raw_items.append(converter() if callable(converter) else item)
    result: list[BridgeFill] = []
    for item in raw_items:
        parsed = _fill_from_object(item)
        if parsed is not None:
            result.append(parsed)
    return result


def _account_from_runner(runner: object, broker: object) -> Mapping[str, float] | None:
    """Account valuation/margin scalars from the runner cache or the broker."""
    cached = getattr(runner, "last_account", None)
    if isinstance(cached, Mapping):
        account = {str(key): value for key, value in cached.items() if _as_float(value) is not None}
        if account:
            return MappingProxyType({key: float(value) for key, value in account.items()})
    getter = getattr(broker, "get_account_sync", None)
    if callable(getter):
        snapshot = getter()
        return _account_from_snapshot(snapshot)
    net_liquidation = _as_positive(getattr(broker, "net_liquidation", None))
    if net_liquidation is None:
        return None
    result: dict[str, float] = {"net_liquidation": net_liquidation}
    cash = _as_float(getattr(broker, "cash", None))
    if cash is not None:
        result["cash_balance"] = cash
    return MappingProxyType(result)


def _account_from_snapshot(snapshot: object) -> Mapping[str, float] | None:
    result: dict[str, float] = {}
    for name in (
        "net_liquidation",
        "cash_balance",
        "buying_power",
        "initial_margin_used",
        "maintenance_margin_used",
        "excess_liquidity",
        "margin_utilization",
    ):
        value = _as_float(getattr(snapshot, name, None))
        if value is not None:
            result[name] = value
    return MappingProxyType(result) if result else None


def _positions_from_runner(runner: object, broker: object) -> tuple[BridgePosition, ...]:
    """Open positions from the runner's latest fetch, else the broker (sync APIs).

    Flat (zero-quantity) rows are dropped: the simulated venue keeps closed
    positions around for accounting, but the bridge publishes the *open* book the
    dashboard renders.
    """
    source = getattr(runner, "last_positions", None)
    if not isinstance(source, Mapping):
        getter = getattr(broker, "get_positions_sync", None)
        if callable(getter):
            source = getter()
        else:
            candidate = getattr(broker, "positions", None)
            source = candidate if isinstance(candidate, Mapping) else None
    if not isinstance(source, Mapping):
        return ()
    positions: list[BridgePosition] = []
    for key, value in source.items():
        parsed = _position_from_object(value)
        if parsed is None:
            continue
        if parsed.instrument_id != str(key):
            parsed = BridgePosition(
                instrument_id=str(key),
                contract_symbol=parsed.contract_symbol,
                quantity=parsed.quantity,
                average_price=parsed.average_price,
                market_price=parsed.market_price,
                multiplier=parsed.multiplier,
                unrealized_pnl=parsed.unrealized_pnl,
                initial_margin=parsed.initial_margin,
            )
        if parsed.quantity == 0:
            continue
        positions.append(parsed)
    return tuple(positions)


def _status_from_runner(runner: object) -> dict[str, ScalarValue]:
    """Flatten the runner status (dataclass or mapping) into bridge scalars."""
    raw = getattr(runner, "status", None)
    if isinstance(raw, Mapping):
        return _scalar_mapping(raw, name="runner.status")
    if raw is None:
        return {}
    as_dict = getattr(raw, "as_dict", None)
    if callable(as_dict):
        payload = cast("Mapping[str, object]", as_dict())
    else:
        payload = {
            name: getattr(raw, name, None)
            for name in (
                "hydrated",
                "states_processed",
                "orders_submitted",
                "orders_rejected",
                "halted",
                "locked",
                "last_error",
                "started_at",
                "ended_at",
                "roots",
            )
        }
    status: dict[str, ScalarValue] = {}
    for key, value in payload.items():
        if key == "roots":
            sequence = _as_sequence(value)
            status[key] = (
                ",".join(str(root) for root in sequence)
                if sequence is not None
                else str(value or "")
            )
            continue
        scalar = _coerce_scalar(value)
        if scalar is None and value is not None:
            continue
        status[key] = scalar
    for name in ("dry_run", "halted"):
        value = getattr(runner, name, None)
        if isinstance(value, bool):
            status.setdefault(name, value)
    return status


def _pipeline_from_result(result: object | None) -> Mapping[str, ScalarValue] | None:
    """Flatten a :class:`~src.core.types.PipelineResult` into bridge scalars."""
    if result is None:
        return None
    pipeline: dict[str, ScalarValue] = {}
    timestamp = _optional_datetime(getattr(result, "timestamp", None))
    if timestamp is not None:
        pipeline["timestamp"] = timestamp.isoformat()
    universe = _as_sequence(getattr(result, "universe", None))
    if universe is not None:
        pipeline["universe"] = ",".join(str(root) for root in universe)
        pipeline["universe_size"] = len(universe)
    final_weights = getattr(result, "final_weights", None)
    weights = getattr(final_weights, "weights", None)
    if isinstance(weights, Mapping):
        numeric = {
            str(key): float(value) for key, value in weights.items() if _as_float(value) is not None
        }
        pipeline["final_weights"] = len(numeric)
        pipeline["gross_leverage"] = sum(abs(value) for value in numeric.values())
        pipeline["net_leverage"] = sum(numeric.values())
    targets = getattr(result, "targets", None)
    if targets is not None:
        try:
            pipeline["targets"] = len(targets)
        except TypeError:  # pragma: no cover - defensive against odd doubles
            pipeline["targets"] = None
        book: list[str] = []
        for target in _as_iterable_sequence(targets) or ():
            instrument_id = _optional_str(getattr(target, "instrument_id", None))
            quantity = _as_int(getattr(target, "quantity", None))
            if instrument_id is not None and quantity is not None:
                book.append(f"{instrument_id}:{quantity}")
        if book:
            pipeline["target_book"] = ",".join(book)[:512]
    elapsed = _as_float(getattr(result, "elapsed_ms", None))
    if elapsed is not None:
        pipeline["elapsed_ms"] = elapsed
    diagnostics = getattr(result, "diagnostics", None)
    if isinstance(diagnostics, Mapping):
        pipeline["diagnostics_keys"] = ",".join(sorted(str(key) for key in diagnostics))[:512]
        tracking = _tracking_error_from_diagnostics(diagnostics)
        if tracking is not None:
            pipeline["tracking_error_pct"] = tracking
    return MappingProxyType(pipeline) if pipeline else None


def _tracking_error_from_diagnostics(diagnostics: Mapping[object, object]) -> float | None:
    for known_key in ("tracking_error_pct", "tracking_error", "tracking_error_percent"):
        known_value = _as_float(diagnostics.get(known_key))
        if known_value is not None:
            return known_value
    for diagnostic_key, diagnostic_value in diagnostics.items():
        if isinstance(diagnostic_key, str) and "tracking_error" in diagnostic_key:
            number = _as_float(diagnostic_value)
            if number is not None:
                return number
    return None


# ---------------------------------------------------------------------------
# Pipeline weight-matrix extraction (Phase 10, duck-typed)
# ---------------------------------------------------------------------------


def _field(value: object, name: str, default: object = None) -> object:
    """Field access supporting both mapping- and attribute-based duck types."""
    if isinstance(value, Mapping):
        return cast("Mapping[str, object]", value).get(name, default)
    return getattr(value, name, default)


def _weight_map(vector: object) -> dict[str, float]:
    """Finite ``instrument -> weight`` mapping from a WeightVector-like object."""
    raw = _field(vector, "weights")
    if not isinstance(raw, Mapping):
        return {}
    weights: dict[str, float] = {}
    for key, value in raw.items():
        number = _as_float(value)
        if isinstance(key, str) and number is not None:
            weights[key] = number
    return weights


def _targets_map(result: object) -> dict[str, object]:
    """``instrument -> ContractTarget`` mapping from a PipelineResult-like object."""
    sequence = _as_iterable_sequence(_field(result, "targets"))
    if sequence is None:
        return {}
    targets: dict[str, object] = {}
    for target in sequence:
        instrument_id = _optional_str(_field(target, "instrument_id"))
        if instrument_id is not None:
            targets[instrument_id] = target
    return targets


def _universe_roots(result: object) -> set[str]:
    sequence = _as_sequence(_field(result, "universe"))
    if sequence is None:
        return set()
    return {str(root) for root in sequence if str(root)}


def _asset_class_text(master: object, root: str) -> str:
    """Asset-class enum value for ``root`` via the instrument master (duck-typed)."""
    unknown = "unknown"
    if master is None:
        return unknown
    try:
        has = getattr(master, "has", None)
        get = getattr(master, "get", None)
        if callable(has) and callable(get) and bool(has(root)):
            spec = get(root)
            text = _enum_text(_field(spec, "asset_class"))
            if text:
                return text
    except Exception:  # pragma: no cover - defensive against exotic masters
        return unknown
    return unknown


def _adjustment_driver(
    selected: bool, base: float, timing: float, risk: float, quantity: int
) -> str:
    """Per-instrument driver label (identical semantics to the API waterfall)."""
    if not selected:
        return "filtered out by the universe selector"
    if timing != base:
        return "scaled by the timing/term-structure overlay"
    if risk != timing:
        return "projected by the risk overlays (vol target / LLM breaker)"
    if quantity == 0:
        return "rounded flat by the contract translator (below buffer)"
    return "passed through unchanged"


def _fallback_achieved_weights(
    targets: Mapping[str, object], equity: float | None
) -> dict[str, float] | None:
    """Arithmetic ``signed notional / equity`` achieved weights from the book.

    An empty target set is a genuinely flat achieved book (``{}``), not "unknown":
    the tracking error then measures the unachieved requested weights, which is
    the information the dashboard needs after a risk overlay flattens the book.
    """
    if equity is None or equity <= 0.0:
        return None
    achieved: dict[str, float] = {}
    for root, target in targets.items():
        notional = _as_float(_field(target, "signed_notional"))
        if notional is None:
            quantity = _as_int(_field(target, "quantity"))
            price = _as_positive(_field(target, "price"))
            multiplier = _as_positive(_field(target, "multiplier"))
            if quantity is None or price is None or multiplier is None:
                continue
            notional = float(quantity) * price * multiplier
        achieved[root] = notional / equity
    return achieved


def _tracking_error_from_weights(
    final_weights: Mapping[str, float], achieved_weights: Mapping[str, float]
) -> float:
    """Half-L1 notional tracking error in percent (the translator's convention)."""
    roots = set(final_weights) | set(achieved_weights)
    if not roots:
        return 0.0
    error = 0.5 * math.fsum(
        abs(final_weights.get(root, 0.0) - achieved_weights.get(root, 0.0)) for root in roots
    )
    return round(max(0.0, error) * 100.0, 6)


def _micro_routing_from_targets(
    targets: Mapping[str, object], master: object
) -> tuple[BridgeMicroRouting, ...]:
    """Derive standard/micro pairs from the achieved discrete contract book."""
    if not targets or master is None:
        return ()
    get = getattr(master, "get", None)
    has = getattr(master, "has", None)
    if not callable(get) or not callable(has):
        return ()
    routes: list[BridgeMicroRouting] = []
    seen_families: set[str] = set()
    for root in sorted(targets):
        family = root
        try:
            if bool(has(root)):
                spec = get(root)
                standard_root = _optional_str(_field(spec, "standard_root"))
                family = standard_root or _optional_str(_field(spec, "root")) or root
        except Exception:  # pragma: no cover - defensive against exotic masters
            family = root
        if family in seen_families:
            continue
        seen_families.add(family)
        try:
            if not bool(has(family)):
                continue
            micro_root = _optional_str(_field(get(family), "micro_root"))
            micro_quantity = _as_int(_field(targets.get(micro_root or ""), "quantity")) or 0
            standard_quantity = _as_int(_field(targets.get(family), "quantity")) or 0
        except Exception:  # pragma: no cover - defensive against exotic masters
            continue
        if micro_root is None or micro_quantity == 0:
            continue
        routes.append(
            BridgeMicroRouting(
                family=family,
                standard_root=family,
                standard_quantity=standard_quantity,
                micro_root=micro_root,
                micro_quantity=micro_quantity,
            )
        )
    return tuple(routes)


def _micro_routing_from_report(report: object) -> tuple[BridgeMicroRouting, ...]:
    """Read ``micro_routes`` (or ``routes``) from a duck-typed discretization report."""
    raw = _field(report, "micro_routes")
    if raw is None:
        raw = _field(report, "routes")
    sequence = _as_sequence(raw)
    if sequence is None:
        return ()
    routes: list[BridgeMicroRouting] = []
    for entry in sequence:
        standard_root = _optional_str(_field(entry, "standard_root"))
        micro_root = _optional_str(_field(entry, "micro_root"))
        if standard_root is None or micro_root is None:
            continue
        routes.append(
            BridgeMicroRouting(
                family=_optional_str(_field(entry, "family")) or standard_root,
                standard_root=standard_root,
                standard_quantity=_as_int(_field(entry, "standard_quantity")) or 0,
                micro_root=micro_root,
                micro_quantity=_as_int(_field(entry, "micro_quantity")) or 0,
            )
        )
    return tuple(routes)


def _achieved_from_report(report: object) -> dict[str, float] | None:
    raw = _field(report, "achieved_weights")
    if not isinstance(raw, Mapping):
        return None
    achieved: dict[str, float] = {}
    for key, value in raw.items():
        number = _as_float(value)
        if isinstance(key, str) and number is not None:
            achieved[key] = number
    return achieved or None


def _translator_report(
    runner: object, result: object | None, equity: float | None
) -> object | None:
    """Duck-type the runner's latest translator/discretization report.

    Preferred sources, in order:

    1. ``runner.last_discretization_report`` / ``runner.translator_report``
       (a runner that already computed the report once per state);
    2. ``runner.pipeline.translator.discretization_report(...)`` invoked against
       ``runner.last_market_state`` / the pipeline result (tolerant: the runner,
       pipeline and state are all duck-typed, and any failure degrades to the
       arithmetic ``fallback`` rather than failing the snapshot).
    """
    for attribute in ("last_discretization_report", "translator_report"):
        candidate = getattr(runner, attribute, None)
        if candidate is not None and not callable(candidate):
            return cast("object", candidate)
    if result is None:
        return None
    pipeline = getattr(runner, "pipeline", None)
    translator = getattr(pipeline, "translator", None)
    reporter = getattr(translator, "discretization_report", None)
    state = getattr(runner, "last_market_state", None)
    weights = _field(result, "final_weights")
    targets = _field(result, "targets")
    if not callable(reporter) or state is None or targets is None or weights is None:
        return None
    call = cast("Callable[..., object]", reporter)
    kwargs: dict[str, object] = {"equity": equity} if equity is not None else {}
    try:
        return call(weights, targets, state, **kwargs)
    except TypeError:
        try:
            return call(weights, targets, state)
        except Exception as exc:  # pragma: no cover - defensive vendor-free fallback
            logger.warning("translator discretization report unavailable: %s", exc)
            return None
    except Exception as exc:  # pragma: no cover - defensive vendor-free fallback
        logger.warning("translator discretization report failed: %s", exc)
        return None


def _runner_equity(runner: object, account: Mapping[str, float] | None) -> float | None:
    if account is not None:
        net_liquidation = _as_positive(account.get("net_liquidation"))
        if net_liquidation is not None:
            return net_liquidation
    for attribute in ("equity", "net_liquidation"):
        value = _as_positive(getattr(runner, attribute, None))
        if value is not None:
            return value
    state = getattr(runner, "last_market_state", None)
    return _as_positive(_field(state, "equity"))


def _runner_regime(runner: object) -> str | None:
    for attribute in ("last_regime", "regime"):
        value = _optional_str(getattr(runner, attribute, None))
        if value is not None:
            return value
    state = getattr(runner, "last_market_state", None)
    return _optional_str(_field(state, "regime"))


def _pipeline_rows_from_result(
    result: object | None,
    *,
    master: object,
    achieved: Mapping[str, float] | None,
    equity: float | None,
    regime: str | None,
) -> tuple[BridgePipelineRow, ...]:
    """Build the per-instrument weight-matrix rows from a pipeline result."""
    if result is None:
        return ()
    base = _weight_map(_field(result, "base_weights"))
    timing = _weight_map(_field(result, "timing_weights"))
    risk = _weight_map(_field(result, "risk_weights"))
    final = _weight_map(_field(result, "final_weights"))
    targets = _targets_map(result)
    selected = _universe_roots(result)
    roots = set(base) | set(timing) | set(risk) | set(final) | set(targets) | selected
    if not roots:
        return ()
    rows: list[BridgePipelineRow] = []
    for root in sorted(roots):
        target = targets.get(root)
        quantity = None if target is None else _as_int(_field(target, "quantity"))
        notional = None if target is None else _as_float(_field(target, "signed_notional"))
        if notional is None and equity is not None:
            notional = final.get(root, 0.0) * equity
        row_base = base.get(root, 0.0)
        row_timing = timing.get(root, 0.0)
        row_risk = risk.get(root, 0.0)
        row_final = final.get(root, 0.0)
        rows.append(
            BridgePipelineRow(
                instrument_id=root,
                root=root,
                asset_class=_asset_class_text(master, root),
                selected=root in selected,
                weight_base=round(row_base, 8),
                weight_timing=round(row_timing, 8),
                weight_risk=round(row_risk, 8),
                weight_final=round(row_final, 8),
                weight_actual=(round(achieved.get(root, 0.0), 8) if achieved is not None else None),
                delta_timing=round(row_timing - row_base, 8),
                delta_risk=round(row_risk - row_timing, 8),
                target_contracts=quantity,
                notional=round(notional, 2) if notional is not None else None,
                driver=_adjustment_driver(
                    root in selected, row_base, row_timing, row_risk, quantity or 0
                ),
                regime=regime,
            )
        )
    return tuple(rows)


def _pipeline_detail(
    runner: object, result: object | None, *, account: Mapping[str, float] | None
) -> tuple[
    tuple[BridgePipelineRow, ...],
    tuple[BridgeMicroRouting, ...],
    float | None,
    str,
]:
    """Return ``(rows, micro_routing, tracking_error_pct, tracking_error_source)``."""
    if result is None:
        return ((), (), None, "fallback")
    equity = _runner_equity(runner, account)
    master = getattr(runner, "instrument_master", None)
    targets = _targets_map(result)
    final_weights = _weight_map(_field(result, "final_weights"))
    report = _translator_report(runner, result, equity)
    achieved = _achieved_from_report(report) if report is not None else None
    if achieved is None:
        achieved = _fallback_achieved_weights(targets, equity)
    tracking: float | None = None
    if report is not None:
        tracking = _as_float(_field(report, "tracking_error_pct"))
        if tracking is None:
            raw_error = _as_float(_field(report, "tracking_error"))
            tracking = raw_error * 100.0 if raw_error is not None else None
    if tracking is None:
        diagnostics = _field(result, "diagnostics")
        if isinstance(diagnostics, Mapping):
            tracking = _tracking_error_from_diagnostics(
                cast("Mapping[object, object]", diagnostics)
            )
    if tracking is None and achieved is not None:
        tracking = _tracking_error_from_weights(final_weights, achieved)
    if tracking is not None:
        tracking = round(max(0.0, float(tracking)), 6)
    routes = _micro_routing_from_report(report) if report is not None else ()
    if not routes:
        routes = _micro_routing_from_targets(targets, master)
    rows = _pipeline_rows_from_result(
        result,
        master=master,
        achieved=achieved,
        equity=equity,
        regime=_runner_regime(runner),
    )
    return (rows, routes, tracking, "translator" if report is not None else "fallback")


def _runner_mode(runner: object, explicit: str | None) -> BridgeMode:
    if explicit is not None:
        if explicit not in BRIDGE_MODES:
            raise BridgeSchemaError(f"mode must be one of {list(BRIDGE_MODES)}, got {explicit!r}")
        return explicit
    candidate = getattr(runner, "mode", None)
    if isinstance(candidate, str) and candidate in BRIDGE_MODES:
        return candidate
    settings = getattr(runner, "settings", None)
    environment = getattr(settings, "environment", None)
    if environment in ("paper", "live"):
        return cast("BridgeMode", environment)
    return "offline"


def _runner_clock(runner: object) -> datetime:
    clock = getattr(runner, "clock", None)
    if callable(clock):
        return ensure_utc(cast("datetime", clock()))
    return datetime.now(UTC)


def _runner_identifier(runner: object) -> str:
    candidate = getattr(runner, "runner_id", None)
    if isinstance(candidate, str) and candidate:
        return candidate
    return f"runner-{os.getpid()}"


def snapshot_from_runner(
    runner: object,
    *,
    broker: object,
    pipeline_result: object | None = None,
    mode: str | None = None,
    stale_after_seconds: float = DEFAULT_STALE_AFTER_SECONDS,
) -> RunnerBridgeSnapshot:
    """Build a :class:`RunnerBridgeSnapshot` from a runner and its broker.

    The helper is deliberately duck-typed: it accepts the Phase-8
    :class:`~src.execution.runner.TradingRunner` (or any object exposing
    ``status`` / ``runner_id`` / ``mode`` / ``clock`` / ``last_positions`` /
    ``last_account`` / ``last_pipeline_result`` / ``last_bridge_snapshot``) and
    either the Phase-5 OMS or the simulated broker. Missing capabilities are
    skipped gracefully (``None``/empty collections) instead of failing a live
    state transition.

    The Phase-10 pipeline weight matrix is serialised from the latest pipeline
    result plus the best available discretization report: ``runner
    .last_discretization_report`` / ``runner.translator_report`` when the runner
    exposes one, else ``runner.pipeline.translator.discretization_report(...)``
    against ``runner.last_market_state`` (all duck-typed, no import of the risk
    layer); failing that an arithmetic ``fallback`` computes achieved weights /
    tracking error from the discrete target book and reports
    ``tracking_error_source="fallback"``.

    Parameters
    ----------
    runner:
        Runner-like object providing the status/book caches.
    broker:
        Broker/OMS-like object providing positions, working orders and fills.
    pipeline_result:
        Latest :class:`~src.core.types.PipelineResult`; falls back to
        ``runner.last_pipeline_result`` when omitted.
    mode:
        Explicit ``offline``/``paper``/``live`` override; defaults to
        ``runner.mode`` and finally to the settings environment.
    stale_after_seconds:
        Staleness budget stamped into the snapshot.
    """
    stale = _as_float(stale_after_seconds)
    if stale is None or stale <= 0.0:
        raise ValueError("stale_after_seconds must be a finite strictly-positive number")
    resolved_result = (
        pipeline_result
        if pipeline_result is not None
        else getattr(runner, "last_pipeline_result", None)
    )
    previous = getattr(runner, "last_bridge_snapshot", None)
    sequence = 1
    if isinstance(previous, RunnerBridgeSnapshot):
        sequence = previous.sequence + 1
    working_orders = _cap_latest(
        [order.to_dict() for order in _working_orders_from_broker(broker)],
        limit=MAX_BRIDGE_ORDERS,
    )
    fills = _cap_latest(
        [fill.to_dict() for fill in _fills_from_broker(broker)],
        limit=MAX_BRIDGE_FILLS,
        key="timestamp",
    )
    account = _account_from_runner(runner, broker)
    rows, micro_routing, tracking_error_pct, tracking_error_source = _pipeline_detail(
        runner, resolved_result, account=account
    )
    pipeline = _pipeline_from_result(resolved_result)
    if pipeline is not None:
        enriched = dict(pipeline)
        enriched["instruments"] = len(rows)
        enriched["micro_routes"] = len(micro_routing)
        enriched["tracking_error_source"] = tracking_error_source
        if tracking_error_pct is not None:
            enriched["tracking_error_pct"] = tracking_error_pct
        pipeline = MappingProxyType(enriched)
    return RunnerBridgeSnapshot(
        schema_version=BRIDGE_SCHEMA_VERSION,
        runner_id=_runner_identifier(runner),
        mode=_runner_mode(runner, mode),
        sequence=sequence,
        updated_at=_runner_clock(runner),
        stale_after_seconds=stale,
        status=_status_from_runner(runner),
        account=account,
        pipeline=pipeline,
        positions=_positions_from_runner(runner, broker),
        working_orders=tuple(BridgeWorkingOrder.from_dict(payload) for payload in working_orders),
        fills=tuple(BridgeFill.from_dict(payload) for payload in fills),
        instruments=rows,
        micro_routing=micro_routing,
        tracking_error_pct=tracking_error_pct,
        tracking_error_source=tracking_error_source,
    )
