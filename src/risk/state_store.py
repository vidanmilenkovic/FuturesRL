"""Durable risk & execution state so restarts never reset intraday risk tracking.

A live futures process can be restarted at any time: a container redeploy, an
operator restart, or IBKR's nightly server reset (approx. 23:45-00:45 ET). If
the in-memory :class:`~src.risk.quant_risk_overlay.QuantRiskOverlay` daily loss
tracker, high-water marks or kill-switch latch were lost on restart, the next
session would silently *re-arm* a risk budget that has already been consumed —
that is an unacceptable production failure mode.

This module defines a small, dependency-free persistence contract
(:class:`RiskStateStore`) plus three concrete backends:

* :class:`FileRiskStateStore` - atomic JSON file (single host, default).
* :class:`DuckDBRiskStateStore` - single-row JSON document in DuckDB (the
  platform's existing analytical store).
* :class:`RedisRiskStateStore` - injected Redis-compatible client (multi-host
  deployments / compose stack).
* :class:`InMemoryRiskStateStore` - deterministic tests and ephemeral runs.

Design rules
------------
* Every persisted timestamp is timezone-aware UTC; naive timestamps are rejected.
* Writes are atomic (temp file + ``os.replace``), versioned and forward
  compatible (unknown keys are retained in ``extra``).
* A corrupt store is quarantined (``*.corrupt-<timestamp>``) and reported as
  "no state" rather than crashing the trading process - a missing state is
  recoverable, a wedged trader is not.
* Rate-limiter state is anchored to *wall-clock epoch seconds* (not monotonic
  time) so token-bucket refills remain meaningful across a restart.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import tempfile
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Protocol, cast, runtime_checkable

from src.core.types import ScalarValue, ensure_utc

logger = logging.getLogger(__name__)

#: Current persisted payload schema version.
STATE_STORE_VERSION = 1

#: Default file-store path relative to the repo/artefact directory.
DEFAULT_STATE_FILENAME = "risk_state.json"


class StateStoreError(RuntimeError):
    """Raised when a risk-state payload is structurally invalid."""


def _require_finite(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StateStoreError(f"{name} must be a real number, got {type(value)!r}")
    result = float(value)
    if not math.isfinite(result):
        raise StateStoreError(f"{name} must be finite, got {result!r}")
    return result


def _coerce_float(value: object, *, name: str) -> float:
    """Coerce a stored JSON scalar to float, raising a domain error otherwise."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StateStoreError(f"{name} must be a number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise StateStoreError(f"{name} must be finite, got {result!r}")
    return result


def _coerce_scalar(value: object, *, name: str) -> ScalarValue:
    """Validate a JSON scalar stored in the forward-compatibility bag."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return value
    raise StateStoreError(f"{name} must be a JSON scalar, got {type(value)!r}")


def _isoformat(value: datetime | date | None) -> str | None:
    return value.isoformat() if value is not None else None


def _parse_datetime(value: object, *, name: str) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise StateStoreError(f"{name} must be an ISO-8601 string or null, got {value!r}")
    try:
        return ensure_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
    except (TypeError, ValueError) as exc:
        raise StateStoreError(f"{name} is not a valid ISO-8601 timestamp: {value!r}") from exc


def _parse_date(value: object, *, name: str) -> date | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise StateStoreError(f"{name} must be an ISO-8601 date string or null, got {value!r}")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise StateStoreError(f"{name} is not a valid ISO-8601 date: {value!r}") from exc


@dataclass(frozen=True, slots=True)
class RateLimiterState:
    """Persistable token-bucket state for the OMS rapid-fire / duplicate limiter.

    ``last_refill_epoch_seconds`` is a wall-clock anchor (UTC epoch seconds) so a
    restart mid-session refills exactly the tokens earned while the process was
    down, instead of resetting the bucket to full capacity.
    """

    tokens: float
    capacity: float
    last_refill_epoch_seconds: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.capacity) or self.capacity <= 0.0:
            raise StateStoreError("rate-limiter capacity must be finite and strictly positive")
        if not math.isfinite(self.tokens) or self.tokens < 0.0:
            raise StateStoreError("rate-limiter tokens must be finite and non-negative")
        if self.tokens > self.capacity:
            raise StateStoreError("rate-limiter tokens cannot exceed capacity")
        if (
            not math.isfinite(self.last_refill_epoch_seconds)
            or self.last_refill_epoch_seconds < 0.0
        ):
            raise StateStoreError("rate-limiter refill anchor must be finite and non-negative")

    def to_dict(self) -> dict[str, ScalarValue]:
        """JSON-serialisable representation."""
        return {
            "tokens": self.tokens,
            "capacity": self.capacity,
            "last_refill_epoch_seconds": self.last_refill_epoch_seconds,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> RateLimiterState:
        """Parse a stored rate-limiter payload."""
        if not isinstance(payload, Mapping):
            raise StateStoreError(f"rate_limiter must be a mapping, got {type(payload)!r}")
        try:
            tokens = _coerce_float(payload["tokens"], name="rate_limiter.tokens")
            capacity = _coerce_float(payload["capacity"], name="rate_limiter.capacity")
            anchor = _coerce_float(
                payload["last_refill_epoch_seconds"], name="rate_limiter.last_refill_epoch_seconds"
            )
        except (KeyError, TypeError) as exc:
            raise StateStoreError(f"invalid rate_limiter payload: {dict(payload)!r}") from exc
        return cls(tokens=tokens, capacity=capacity, last_refill_epoch_seconds=anchor)

    @classmethod
    def full(cls, capacity: float, *, now_epoch_seconds: float) -> RateLimiterState:
        """A bucket initialised to full capacity at ``now_epoch_seconds``."""
        return cls(tokens=capacity, capacity=capacity, last_refill_epoch_seconds=now_epoch_seconds)

    def refilled(
        self,
        *,
        now_epoch_seconds: float,
        rate_per_second: float,
        capacity: float | None = None,
    ) -> RateLimiterState:
        """Return tokens earned since the stored anchor (clamped to capacity)."""
        if not math.isfinite(rate_per_second) or rate_per_second <= 0.0:
            raise StateStoreError("rate_per_second must be finite and strictly positive")
        if not math.isfinite(now_epoch_seconds):
            raise StateStoreError("now_epoch_seconds must be finite")
        cap = float(capacity) if capacity is not None else self.capacity
        elapsed = max(0.0, now_epoch_seconds - self.last_refill_epoch_seconds)
        tokens = min(cap, max(0.0, self.tokens) + elapsed * rate_per_second)
        return RateLimiterState(
            tokens=tokens,
            capacity=cap,
            last_refill_epoch_seconds=now_epoch_seconds,
        )


@dataclass(frozen=True, slots=True)
class PersistedRiskState:
    """The durable slice of intraday risk/execution state."""

    #: Net liquidation value at the start of the current trading day (account currency).
    starting_daily_nlv: float | None = None
    #: Portfolio-level high-water mark used by trailing-stop / drawdown logic.
    portfolio_high_water_mark: float | None = None
    #: Per-instrument high-water marks (peak mark value or price, by root).
    instrument_high_water_marks: Mapping[str, float] = field(default_factory=dict)
    #: Latched kill switch (daily max-loss, drawdown stop, or operator emergency).
    kill_switch_locked: bool = False
    kill_switch_reason: str = ""
    #: Quant overlay latching daily-loss breaker (flatten until next session).
    daily_loss_locked: bool = False
    #: Trading day the daily counters belong to (UTC date); a new day resets them.
    trading_day: date | None = None
    #: OMS token-bucket state (persisted across container restarts).
    rate_limiter: RateLimiterState | None = None
    #: When this state was last written.
    updated_at: datetime | None = None
    #: Payload schema version.
    version: int = STATE_STORE_VERSION
    #: Forward-compatible extension bag for future persisted fields.
    extra: Mapping[str, ScalarValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "starting_daily_nlv",
            _require_finite("starting_daily_nlv", self.starting_daily_nlv),
        )
        if self.starting_daily_nlv is not None and self.starting_daily_nlv <= 0.0:
            raise StateStoreError("starting_daily_nlv must be strictly positive")
        object.__setattr__(
            self,
            "portfolio_high_water_mark",
            _require_finite("portfolio_high_water_mark", self.portfolio_high_water_mark),
        )
        if self.portfolio_high_water_mark is not None and self.portfolio_high_water_mark <= 0.0:
            raise StateStoreError("portfolio_high_water_mark must be strictly positive")
        hwm: dict[str, float] = {}
        for instrument_id, value in self.instrument_high_water_marks.items():
            if not isinstance(instrument_id, str) or not instrument_id:
                raise StateStoreError("instrument_high_water_marks keys must be non-empty strings")
            checked = _require_finite(f"instrument_hwm[{instrument_id!r}]", value)
            assert checked is not None
            if checked <= 0.0:
                raise StateStoreError(
                    f"instrument_high_water_marks[{instrument_id!r}] must be strictly positive"
                )
            hwm[instrument_id] = checked
        object.__setattr__(self, "instrument_high_water_marks", MappingProxyType(hwm))
        if self.updated_at is not None:
            try:
                object.__setattr__(self, "updated_at", ensure_utc(self.updated_at))
            except ValueError as exc:
                raise StateStoreError(f"updated_at must be timezone-aware UTC: {exc}") from exc
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version < 1:
            raise StateStoreError("version must be a positive int")
        if self.kill_switch_locked and not self.kill_switch_reason:
            raise StateStoreError("a locked kill switch must carry a non-empty reason")
        object.__setattr__(self, "extra", MappingProxyType(dict(self.extra)))

    # -- mutation helpers ------------------------------------------------------

    def updated(self, **changes: object) -> PersistedRiskState:
        """Return a copy with ``changes`` applied (dataclass ``replace``)."""
        return replace(self, **changes)  # type: ignore[arg-type]

    def locked(self, reason: str) -> PersistedRiskState:
        """Return a copy with the kill switch engaged for ``reason``."""
        if not reason:
            raise StateStoreError("kill switch reason must be non-empty")
        return replace(self, kill_switch_locked=True, kill_switch_reason=reason)

    def unlocked(self) -> PersistedRiskState:
        """Return a copy with the kill switch released."""
        return replace(
            self, kill_switch_locked=False, kill_switch_reason="", daily_loss_locked=False
        )

    def rolled_to_day(self, trading_day: date) -> PersistedRiskState:
        """Reset the daily counter fields for a new ``trading_day``.

        High-water marks and the kill-switch latch deliberately survive the day
        roll: a trailing-stop halt must not auto-clear at midnight.
        """
        if not isinstance(trading_day, date):
            raise StateStoreError("trading_day must be a date")
        return replace(
            self,
            trading_day=trading_day,
            starting_daily_nlv=None,
            daily_loss_locked=False,
        )

    # -- serialisation ---------------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable representation (all timestamps ISO-8601 UTC)."""
        return {
            "version": self.version,
            "starting_daily_nlv": self.starting_daily_nlv,
            "portfolio_high_water_mark": self.portfolio_high_water_mark,
            "instrument_high_water_marks": dict(self.instrument_high_water_marks),
            "kill_switch_locked": self.kill_switch_locked,
            "kill_switch_reason": self.kill_switch_reason,
            "daily_loss_locked": self.daily_loss_locked,
            "trading_day": _isoformat(self.trading_day),
            "rate_limiter": self.rate_limiter.to_dict() if self.rate_limiter else None,
            "updated_at": _isoformat(self.updated_at),
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> PersistedRiskState:
        """Parse a persisted payload, rejecting malformed values explicitly."""
        if not isinstance(payload, Mapping):
            raise StateStoreError(f"risk state must be a mapping, got {type(payload)!r}")
        version = payload.get("version", STATE_STORE_VERSION)
        if isinstance(version, bool) or not isinstance(version, int):
            raise StateStoreError(f"version must be an int, got {version!r}")
        if version > STATE_STORE_VERSION:
            raise StateStoreError(
                f"risk state version {version} is newer than supported {STATE_STORE_VERSION}"
            )
        marks = payload.get("instrument_high_water_marks", {})
        if not isinstance(marks, Mapping):
            raise StateStoreError("instrument_high_water_marks must be a mapping")
        extra = payload.get("extra", {})
        if not isinstance(extra, Mapping):
            raise StateStoreError("extra must be a mapping")
        raw_rate_limiter = payload.get("rate_limiter")
        if raw_rate_limiter is not None and not isinstance(raw_rate_limiter, Mapping):
            raise StateStoreError("rate_limiter must be a mapping or null")
        rate_limiter = (
            RateLimiterState.from_dict(cast("Mapping[str, object]", raw_rate_limiter))
            if raw_rate_limiter is not None
            else None
        )
        locked = payload.get("kill_switch_locked", False)
        if not isinstance(locked, bool):
            raise StateStoreError("kill_switch_locked must be a bool")
        daily_locked = payload.get("daily_loss_locked", False)
        if not isinstance(daily_locked, bool):
            raise StateStoreError("daily_loss_locked must be a bool")
        reason = payload.get("kill_switch_reason", "")
        if not isinstance(reason, str):
            raise StateStoreError("kill_switch_reason must be a string")
        starting = payload.get("starting_daily_nlv")
        portfolio_hwm = payload.get("portfolio_high_water_mark")
        return cls(
            starting_daily_nlv=None
            if starting is None
            else _coerce_float(starting, name="starting_daily_nlv"),
            portfolio_high_water_mark=(
                None
                if portfolio_hwm is None
                else _coerce_float(portfolio_hwm, name="portfolio_high_water_mark")
            ),
            instrument_high_water_marks={
                str(k): _coerce_float(v, name=f"instrument_high_water_marks[{k!r}]")
                for k, v in marks.items()
            },
            kill_switch_locked=locked,
            kill_switch_reason=reason,
            daily_loss_locked=daily_locked,
            trading_day=_parse_date(payload.get("trading_day"), name="trading_day"),
            rate_limiter=rate_limiter,
            updated_at=_parse_datetime(payload.get("updated_at"), name="updated_at"),
            version=version,
            extra={str(k): _coerce_scalar(v, name=f"extra[{k!r}]") for k, v in extra.items()},
        )


#: A risk-state backend: durable, synchronous and process-safe enough for one
#: writer per key (multi-writer deployments should use Redis or DuckDB).
@runtime_checkable
class RiskStateStore(Protocol):
    """Persistence contract for :class:`PersistedRiskState`."""

    def load(self) -> PersistedRiskState | None:
        """Return the persisted state, or ``None`` when nothing is stored."""
        ...

    def save(self, state: PersistedRiskState) -> None:
        """Atomically persist ``state``."""
        ...

    def clear(self) -> None:
        """Delete the persisted state (operator reset / test helper)."""
        ...


# ---------------------------------------------------------------------------
# In-memory backend
# ---------------------------------------------------------------------------


class InMemoryRiskStateStore:
    """Volatile store for tests, dry runs and single-process ephemeral sessions."""

    def __init__(self, initial: PersistedRiskState | None = None) -> None:
        self._state = initial
        self._lock = threading.Lock()

    def load(self) -> PersistedRiskState | None:
        with self._lock:
            return self._state

    def save(self, state: PersistedRiskState) -> None:
        if not isinstance(state, PersistedRiskState):
            raise TypeError(f"state must be PersistedRiskState, got {type(state)!r}")
        with self._lock:
            self._state = state

    def clear(self) -> None:
        with self._lock:
            self._state = None


# ---------------------------------------------------------------------------
# JSON file backend
# ---------------------------------------------------------------------------


class FileRiskStateStore:
    """Atomic single-file JSON store (default backend for single-host deploys)."""

    def __init__(
        self,
        path: Path | str,
        *,
        namespace_key: str = "risk_state",
        fsync: bool = True,
    ) -> None:
        self._path = Path(path)
        if self._path.suffix == "":
            self._path = self._path / DEFAULT_STATE_FILENAME
        if not namespace_key:
            raise ValueError("namespace_key must be non-empty")
        self._namespace_key = namespace_key
        self._fsync = fsync
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        """Absolute path of the JSON file."""
        return self._path.resolve()

    def load(self) -> PersistedRiskState | None:
        with self._lock:
            if not self._path.is_file():
                return None
            try:
                payload = json.loads(self._path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                self._quarantine(exc)
                return None
            if not isinstance(payload, dict):
                self._quarantine(StateStoreError("payload is not a JSON object"))
                return None
            raw = payload.get(self._namespace_key)
            if raw is None:
                return None
            try:
                return PersistedRiskState.from_dict(raw)
            except (StateStoreError, TypeError, ValueError) as exc:
                self._quarantine(exc)
                return None

    def save(self, state: PersistedRiskState) -> None:
        if not isinstance(state, PersistedRiskState):
            raise TypeError(f"state must be PersistedRiskState, got {type(state)!r}")
        stamped = state.updated_at if state.updated_at is not None else datetime.now(UTC)
        stamped_state = replace(state, updated_at=stamped)
        payload = {self._namespace_key: stamped_state.to_dict()}
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            serialised = json.dumps(payload, sort_keys=True, indent=2)
            fd, temp_name = tempfile.mkstemp(
                prefix=f".{self._path.name}.", suffix=".tmp", dir=str(self._path.parent)
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(serialised)
                    handle.flush()
                    if self._fsync:
                        os.fsync(handle.fileno())
                Path(temp_name).replace(self._path)
                if self._fsync:
                    self._fsync_directory(self._path.parent)
            except BaseException:
                with contextlib.suppress(OSError):
                    Path(temp_name).unlink(missing_ok=True)
                raise

    def clear(self) -> None:
        with self._lock:
            try:
                self._path.unlink()
            except FileNotFoundError:
                return

    # -- internals -------------------------------------------------------------

    @staticmethod
    def _fsync_directory(directory: Path) -> None:
        """Best-effort directory fsync so the rename survives a power loss."""
        try:
            fd = os.open(directory, os.O_RDONLY)
        except OSError:  # pragma: no cover - platform dependent
            return
        try:
            os.fsync(fd)
        except OSError:  # pragma: no cover - platform dependent
            pass
        finally:
            os.close(fd)

    def _quarantine(self, exc: BaseException) -> None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        target = self._path.with_name(f"{self._path.name}.corrupt-{stamp}")
        logger.error(
            "risk-state store %s is corrupt (%s); quarantining to %s", self._path, exc, target
        )
        try:
            Path(self._path).replace(target)
        except OSError:  # pragma: no cover - best effort quarantine
            logger.exception("failed to quarantine corrupt risk-state store %s", self._path)


# ---------------------------------------------------------------------------
# DuckDB backend
# ---------------------------------------------------------------------------


class DuckDBRiskStateStore:
    """Single-row JSON document in DuckDB (shares the platform's analytical file)."""

    def __init__(
        self,
        db_path: Path | str,
        *,
        table: str = "risk_state",
        namespace: str = "default",
    ) -> None:
        if not table or not table.replace("_", "").isalnum():
            raise ValueError("table must be a simple identifier")
        if not namespace:
            raise ValueError("namespace must be non-empty")
        self._db_path = Path(db_path)
        self._table = table
        self._namespace = namespace
        self._lock = threading.Lock()
        self._connection: object | None = None

    @property
    def db_path(self) -> Path:
        """Absolute DuckDB file path."""
        return self._db_path.resolve()

    def _connect(self) -> object:  # pragma: no cover - thin duckdb wrapper
        if self._connection is None:
            import duckdb

            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = duckdb.connect(str(self._db_path))
            self._connection.execute(
                f"CREATE TABLE IF NOT EXISTS {self._table} ("
                "namespace VARCHAR PRIMARY KEY, payload VARCHAR, updated_at TIMESTAMP)"
            )
        return self._connection

    def load(self) -> PersistedRiskState | None:
        with self._lock:
            connection = self._connect()
            row = connection.execute(  # type: ignore[attr-defined]
                f"SELECT payload FROM {self._table} WHERE namespace = ?", [self._namespace]
            ).fetchone()
            if row is None or not row[0]:
                return None
            try:
                payload = json.loads(str(row[0]))
                return PersistedRiskState.from_dict(payload)
            except (ValueError, TypeError, StateStoreError) as exc:
                logger.error("corrupt DuckDB risk state (%s); ignoring stored row", exc)
                return None

    def save(self, state: PersistedRiskState) -> None:
        if not isinstance(state, PersistedRiskState):
            raise TypeError(f"state must be PersistedRiskState, got {type(state)!r}")
        stamped = replace(state, updated_at=state.updated_at or datetime.now(UTC))
        serialised = json.dumps(stamped.to_dict(), sort_keys=True)
        with self._lock:
            connection = self._connect()
            connection.execute(  # type: ignore[attr-defined]
                f"DELETE FROM {self._table} WHERE namespace = ?", [self._namespace]
            )
            connection.execute(  # type: ignore[attr-defined]
                f"INSERT INTO {self._table} (namespace, payload, updated_at) VALUES (?, ?, ?)",
                [self._namespace, serialised, datetime.now(UTC)],
            )

    def clear(self) -> None:
        with self._lock:
            connection = self._connect()
            connection.execute(  # type: ignore[attr-defined]
                f"DELETE FROM {self._table} WHERE namespace = ?", [self._namespace]
            )


# ---------------------------------------------------------------------------
# Redis backend
# ---------------------------------------------------------------------------


class RedisRiskStateStore:
    """Redis-backed store using an injected Redis-compatible client.

    The client only needs ``get(key)`` / ``set(key, value)`` / ``delete(key)``
    returning bytes or strings; ``redis.asyncio`` clients also work through the
    sync shim, and tests inject a tiny fake.
    """

    def __init__(self, client: object, *, key: str = "futuresrl:risk_state") -> None:
        if not key:
            raise ValueError("key must be non-empty")
        for method in ("get", "set", "delete"):
            if not callable(getattr(client, method, None)):
                raise TypeError(f"redis client must provide {method}()")
        self._client = client
        self._key = key

    @property
    def key(self) -> str:
        """Redis key holding the JSON payload."""
        return self._key

    def load(self) -> PersistedRiskState | None:
        raw = self._client.get(self._key)  # type: ignore[attr-defined]
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        if not isinstance(raw, str):
            raise StateStoreError(f"redis payload must be str/bytes, got {type(raw)!r}")
        try:
            payload = json.loads(raw)
        except ValueError as exc:
            logger.error("corrupt Redis risk state (%s); ignoring stored key", exc)
            return None
        if not isinstance(payload, Mapping):
            logger.error("corrupt Redis risk state (not a JSON object); ignoring stored key")
            return None
        try:
            return PersistedRiskState.from_dict(payload)
        except (StateStoreError, TypeError, ValueError) as exc:
            logger.error("corrupt Redis risk state (%s); ignoring stored key", exc)
            return None

    def save(self, state: PersistedRiskState) -> None:
        if not isinstance(state, PersistedRiskState):
            raise TypeError(f"state must be PersistedRiskState, got {type(state)!r}")
        stamped = replace(state, updated_at=state.updated_at or datetime.now(UTC))
        self._client.set(self._key, json.dumps(stamped.to_dict(), sort_keys=True))  # type: ignore[attr-defined]

    def clear(self) -> None:
        self._client.delete(self._key)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

type StateStoreLocation = str | Path | None


def build_risk_state_store(
    location: str | Path | None,
    *,
    duckdb_path: Path | str | None = None,
    redis_client: object | None = None,
) -> RiskStateStore:
    """Build a store from a URI/path.

    Accepted forms::

        None | "" | "memory:"            -> InMemoryRiskStateStore
        "file:/path/risk_state.json"     -> FileRiskStateStore
        "/path/risk_state.json"          -> FileRiskStateStore
        "duckdb:"  (uses duckdb_path)    -> DuckDBRiskStateStore
        "redis:"  (uses redis_client)    -> RedisRiskStateStore
    """
    if location is None or location == "" or location == "memory:":
        return InMemoryRiskStateStore()
    text = str(location)
    if text.startswith("file:"):
        return FileRiskStateStore(text[len("file:") :])
    if text.startswith("duckdb:"):
        if duckdb_path is None:
            raise ValueError("duckdb: location requires duckdb_path")
        return DuckDBRiskStateStore(duckdb_path, namespace=text[len("duckdb:") :] or "default")
    if text.startswith("redis:"):
        if redis_client is None:
            raise ValueError("redis: location requires an injected redis client")
        return RedisRiskStateStore(
            redis_client, key=text[len("redis:") :] or "futuresrl:risk_state"
        )
    return FileRiskStateStore(text)
