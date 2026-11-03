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
    "MAX_BRIDGE_ORDERS",
    "BridgeFill",
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
        object.__setattr__(
            self, "positions", tuple(sorted(positions, key=lambda item: item.instrument_id))
        )
        object.__setattr__(self, "working_orders", working_orders)
        object.__setattr__(self, "fills", fills)

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
        for target in cast("Iterable[object]", targets):
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
    return RunnerBridgeSnapshot(
        schema_version=BRIDGE_SCHEMA_VERSION,
        runner_id=_runner_identifier(runner),
        mode=_runner_mode(runner, mode),
        sequence=sequence,
        updated_at=_runner_clock(runner),
        stale_after_seconds=stale,
        status=_status_from_runner(runner),
        account=_account_from_runner(runner, broker),
        pipeline=_pipeline_from_result(resolved_result),
        positions=_positions_from_runner(runner, broker),
        working_orders=tuple(BridgeWorkingOrder.from_dict(payload) for payload in working_orders),
        fills=tuple(BridgeFill.from_dict(payload) for payload in fills),
    )
