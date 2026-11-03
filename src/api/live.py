"""Live runner bridge integration for the telemetry API (Phase 9).

The production process is split in two: the **live/paper runner** publishes
snapshots to a small durable bridge file (``settings.runner_bridge_dir /
runner_state.json``, implemented by ``src.execution.runner_bridge``), and this
**API process** reads them. This module is the API side of that bridge:

* :class:`RunnerBridgeReader` — loads a fresh bridge snapshot, preferring the
  durable :class:`~src.execution.runner_bridge.RunnerBridgeStore` when that
  module is importable and falling back to a direct JSON read of the documented
  bridge payload otherwise (so the dashboard still sees runner state even when
  the execution package is partially deployed).
* :class:`RunnerSessionController` — owns the singleton ``OfflineRunnerSession``
  used by ``POST /api/runner/step``; the session is constructed lazily (never at
  import time), and the endpoint returns an actionable 503 when the execution
  package predates the offline session.
* :class:`LiveStateService` — maps bridge/fallback data onto the frozen API
  schemas for orders, fills, account/positions, the pipeline waterfall and the
  runner status endpoints. Every bridge payload is parsed tolerantly: an
  incompatible or missing payload degrades to the existing simulated/fixture
  behaviour instead of breaking the endpoint.

Everything here is offline-testable: no network, no IBKR, and the runner bridge
may be a synthetic JSON file written by a test.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import importlib
import inspect
import json
import logging
import math
import os
import tempfile
import threading
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from configs.instruments import InstrumentsConfig
from configs.settings import Settings
from pydantic import ValidationError

from src.api.schemas import (
    AccountSnapshotResponse,
    AccountSource,
    EquityPoint,
    FillRow,
    MicroRoutingEntry,
    PipelineDiagnostic,
    PipelineStage,
    PipelineWaterfallResponse,
    PositionRow,
    RecentFillsResponse,
    RunnerBridgeSummary,
    RunnerStatusResponse,
    RunnerStepRequest,
    RunnerStepResponse,
    SnapshotSource,
    WorkingOrderRow,
    WorkingOrdersResponse,
)
from src.core.types import AssetClass, InstrumentId, ensure_utc

if TYPE_CHECKING:  # pragma: no cover - typing-only imports
    from src.api.protocols import MarketDataService
    from src.core.types import MarketState

logger = logging.getLogger(__name__)

__all__ = [
    "MIN_LIVE_CURVE_DAYS",
    "BridgeSnapshot",
    "LiveEquityHistoryStore",
    "LiveStateService",
    "RunnerBridgeReader",
    "RunnerSessionController",
    "RunnerSessionUnavailable",
    "RunnerSessionUnavailableError",
    "ex_ante_portfolio_volatility",
    "history_distinct_utc_days",
    "history_intraday_points",
    "load_optional_module",
]

#: File name of the runner -> API bridge snapshot inside ``runner_bridge_dir``.
BRIDGE_FILE_NAME: str = "runner_state.json"

#: File name of the durable IBKR live-NLV history inside ``settings.data_dir``.
EQUITY_HISTORY_FILE_NAME: str = "ibkr_equity_history.json"

#: Minimum number of real IBKR NLV points before the curve is labelled live.
MIN_LIVE_CURVE_POINTS: int = 5

#: Minimum number of DISTINCT UTC calendar days of real IBKR history before the
#: multi-day curve is labelled ``ibkr-live`` (intraday points alone are not enough).
MIN_LIVE_CURVE_DAYS: int = 5

#: Bound on the persisted live NLV history (newest wins).
MAX_EQUITY_HISTORY_POINTS: int = 5000

#: Bound on the age of persisted live NLV points (days).
MAX_EQUITY_HISTORY_AGE_DAYS: float = 30.0

#: Persisted history payload schema version.
EQUITY_HISTORY_VERSION: int = 1

#: Default staleness horizon when the snapshot does not declare its own.
DEFAULT_STALE_AFTER_SECONDS: float = 30.0

#: Lazily-importable optional execution module paths.
RUNNER_BRIDGE_MODULE = "src.execution.runner_bridge"
RUNNER_MODULE = "src.execution.runner"


class RunnerSessionUnavailableError(RuntimeError):
    """The offline runner session cannot be constructed in this deployment."""


#: Backwards/forwards compatible shorter alias.
RunnerSessionUnavailable = RunnerSessionUnavailableError


# ---------------------------------------------------------------------------
# Durable live IBKR NLV history (attribution)
# ---------------------------------------------------------------------------


class LiveEquityHistoryStore:
    """Atomic, bounded and deduped JSON history of real observed IBKR NLV points.

    The dashboard must never present a synthetic equity curve as live. This store
    keeps the *real* points fetched from an IBKR session so the account snapshot
    can switch from the synthetic backfill curve to a genuine ``ibkr-live`` curve
    once at least :data:`MIN_LIVE_CURVE_POINTS` observations exist.

    Design rules
    ------------
    * Atomic writes (temp file + ``os.replace``) so a crash can never truncate a
      previously good history document.
    * One point per UTC timestamp (dedupe; later value wins).
    * Bounded by :data:`MAX_EQUITY_HISTORY_POINTS` and
      :data:`MAX_EQUITY_HISTORY_AGE_DAYS` so the file cannot grow forever.
    * A corrupt file is quarantined (``*.corrupt-*``) and treated as "no history"
      instead of crashing the trading/API process.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        clock: Callable[[], datetime],
        max_points: int = MAX_EQUITY_HISTORY_POINTS,
        max_age_days: float = MAX_EQUITY_HISTORY_AGE_DAYS,
    ) -> None:
        resolved = Path(path)
        if resolved.suffix == "":
            resolved = resolved / EQUITY_HISTORY_FILE_NAME
        self._path = resolved
        self._clock = clock
        self._max_points = max(1, int(max_points))
        if not math.isfinite(max_age_days) or max_age_days < 0.0:
            raise ValueError("max_age_days must be a finite non-negative number")
        self._max_age_days = float(max_age_days)
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        """Absolute path of the persisted history JSON document."""
        return self._path.resolve()

    # -- public API ------------------------------------------------------------

    def load(self) -> list[EquityPoint]:
        """Return the persisted points (oldest first, ``[]`` when absent/corrupt)."""
        with self._lock:
            return self._load_locked()

    def append(self, point: EquityPoint) -> list[EquityPoint]:
        """Append one observed NLV point (deduped/bounded) and persist atomically."""
        with self._lock:
            existing = self._load_locked()
            moment = ensure_utc(point.time)
            merged: dict[datetime, float] = {
                existing_point.time: existing_point.value for existing_point in existing
            }
            merged[moment] = float(point.value)
            points = self._trim(
                [EquityPoint(time=timestamp, value=value) for timestamp, value in merged.items()]
            )
            self._save_locked(points)
            return points

    def clear(self) -> None:
        """Delete the persisted history (operator reset / tests)."""
        with self._lock, contextlib.suppress(FileNotFoundError):
            self._path.unlink()

    def distinct_utc_days(self) -> int:
        """Number of distinct UTC calendar days covered by the persisted points."""
        return history_distinct_utc_days(self.load())

    def intraday_points(self) -> list[EquityPoint]:
        """Real points of the latest recorded UTC day (for the 1D chart)."""
        return history_intraday_points(self.load())

    # -- internals -------------------------------------------------------------

    def _load_locked(self) -> list[EquityPoint]:
        if not self._path.is_file():
            return []
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            self._quarantine(exc)
            return []
        if not isinstance(payload, Mapping):
            self._quarantine(ValueError("payload is not a JSON object"))
            return []
        raw_points = cast("Mapping[str, object]", payload).get("points")
        if not isinstance(raw_points, Sequence) or isinstance(raw_points, (str, bytes)):
            return []
        points: list[EquityPoint] = []
        for entry in raw_points:
            parsed = self._parse_point(entry)
            if parsed is not None:
                points.append(parsed)
        points.sort(key=lambda item: item.time)
        return points

    @staticmethod
    def _parse_point(entry: object) -> EquityPoint | None:
        if not isinstance(entry, Mapping):
            return None
        mapping = cast("Mapping[str, object]", entry)
        raw_time = mapping.get("time")
        raw_value = mapping.get("value")
        if not isinstance(raw_time, str) or isinstance(raw_value, bool):
            return None
        if not isinstance(raw_value, (int, float)):
            return None
        value = float(raw_value)
        if not math.isfinite(value) or value <= 0.0:
            return None
        try:
            moment = ensure_utc(datetime.fromisoformat(raw_time.replace("Z", "+00:00")))
        except (TypeError, ValueError):
            return None
        return EquityPoint(time=moment, value=value)

    def _trim(self, points: Sequence[EquityPoint]) -> list[EquityPoint]:
        moment = ensure_utc(self._clock())
        cutoff = moment - timedelta(days=self._max_age_days)
        fresh = sorted(
            (point for point in points if point.time >= cutoff),
            key=lambda item: item.time,
        )
        if len(fresh) > self._max_points:
            fresh = fresh[-self._max_points :]
        return fresh

    def _save_locked(self, points: Sequence[EquityPoint]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": EQUITY_HISTORY_VERSION,
            "points": [{"time": point.time.isoformat(), "value": point.value} for point in points],
        }
        serialised = json.dumps(payload, sort_keys=True, indent=2)
        descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{self._path.name}.", suffix=".tmp", dir=str(self._path.parent)
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(serialised)
                handle.flush()
                os.fsync(handle.fileno())
            Path(temp_name).replace(self._path)
        except BaseException:
            with contextlib.suppress(OSError):
                Path(temp_name).unlink(missing_ok=True)
            raise

    def _quarantine(self, exc: BaseException) -> None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        target = self._path.with_name(f"{self._path.name}.corrupt-{stamp}")
        logger.error(
            "IBKR equity history %s is corrupt (%s); quarantining to %s",
            self._path,
            exc,
            target,
        )
        with contextlib.suppress(OSError):
            self._path.replace(target)


# ---------------------------------------------------------------------------
# Live-equity history provenance helpers
# ---------------------------------------------------------------------------


def history_distinct_utc_days(points: Sequence[EquityPoint]) -> int:
    """Distinct UTC calendar days covered by the real IBKR NLV observations."""
    return len({point.time.astimezone(UTC).date() for point in points})


def history_intraday_points(points: Sequence[EquityPoint]) -> list[EquityPoint]:
    """Real points of the latest recorded UTC day (deterministic 1D curve).

    The latest *recorded* day is used rather than the wall clock so a replay or a
    test clock behaves deterministically; with no history the list is empty.
    """
    ordered = sorted(points, key=lambda point: point.time)
    if not ordered:
        return []
    latest_day = ordered[-1].time.astimezone(UTC).date()
    return [
        point.model_copy() for point in ordered if point.time.astimezone(UTC).date() == latest_day
    ]


# ---------------------------------------------------------------------------
# Ex-ante portfolio volatility from the live book
# ---------------------------------------------------------------------------


def ex_ante_portfolio_volatility(
    weights: Mapping[InstrumentId, float],
    state: MarketState | None,
) -> tuple[float | None, str]:
    """Annualised ``sqrt(w' Σ w)`` from signed notional weights and market state.

    Returns ``(0.0, "ex-ante-book")`` for a flat book (no open positions),
    ``(None, "unavailable")`` when the covariance matrix cannot value the book,
    and ``(volatility, "ex-ante-book")`` otherwise. The covariance carried by
    :class:`~src.core.types.MarketState` is already annualised.
    """
    active = {
        instrument_id: float(weight)
        for instrument_id, weight in weights.items()
        if math.isfinite(float(weight)) and abs(float(weight)) > 1e-12
    }
    if not active:
        return 0.0, "ex-ante-book"
    if state is None or state.covariance is None:
        return None, "unavailable"
    instruments = tuple(sorted(active))
    covariance = state.covariance_submatrix(instruments)
    if covariance is None:
        return None, "unavailable"
    vector = np.array([active[instrument] for instrument in instruments], dtype=np.float64)
    variance = float(vector @ covariance @ vector)
    if not math.isfinite(variance):
        return None, "unavailable"
    return math.sqrt(max(variance, 0.0)), "ex-ante-book"


def load_optional_module(path: str) -> object | None:
    """Import an optional module, returning ``None`` instead of raising."""
    try:
        return importlib.import_module(path)
    except Exception:  # ImportError and any import-time wiring failure
        return None


# ---------------------------------------------------------------------------
# Tolerant mapping helpers (the bridge payload is versioned but plain JSON)
# ---------------------------------------------------------------------------


def _get(source: object, name: str, default: object = None) -> object:
    if isinstance(source, Mapping):
        return cast(Mapping[str, object], source).get(name, default)
    return getattr(source, name, default)


def _as_mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return cast(Mapping[str, object], value)
    return {}


def _entry_mapping(value: object) -> Mapping[str, object]:
    """Mapping view of a bridge row (plain dict, ``to_dict()`` or dataclass)."""
    mapping = _as_mapping(value)
    if mapping:
        return mapping
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            result = to_dict()
        except Exception:  # pragma: no cover - defensive bridge adapter
            return {}
        if isinstance(result, Mapping):
            return cast(Mapping[str, object], result)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        try:
            result = dataclasses.asdict(value)
        except Exception:  # pragma: no cover - exotic dataclass
            return {}
        return cast(Mapping[str, object], result)
    return {}


def _as_sequence(value: object) -> list[object]:
    if isinstance(value, (str | bytes | bytearray, Mapping)):
        return []
    if isinstance(value, Sequence):
        return list(cast(Sequence[object], value))
    return []


def _as_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _as_int(value: object) -> int | None:
    number = _as_float(value)
    return None if number is None else round(number)


def _as_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        try:
            return ensure_utc(value)
        except ValueError:
            return None
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time(), tzinfo=UTC)
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        return ensure_utc(datetime.fromisoformat(text))
    except ValueError:
        return None


def _text(value: object, default: str = "") -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if value is None:
        return default
    return str(value)


# ---------------------------------------------------------------------------
# Bridge snapshot view
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BridgeSnapshot:
    """Tolerant, read-only view over one runner bridge snapshot."""

    raw: object
    stale: bool

    def get(self, name: str, default: object = None) -> object:
        """Field access supporting both attribute- and mapping-based snapshots."""
        return _get(self.raw, name, default)

    @property
    def runner_id(self) -> str | None:
        value = self.get("runner_id")
        return _text(value) or None

    @property
    def mode(self) -> str | None:
        value = self.get("mode")
        return _text(value) or None

    @property
    def sequence(self) -> int | None:
        return _as_int(self.get("sequence"))

    @property
    def updated_at(self) -> datetime | None:
        return _as_datetime(self.get("updated_at"))

    @property
    def stale_after_seconds(self) -> float | None:
        return _as_float(self.get("stale_after_seconds"))

    @property
    def status(self) -> str | None:
        value = self.get("status")
        if isinstance(value, Mapping):
            payload = cast(Mapping[str, object], value)
            for key in ("state", "status"):
                text = _text(payload.get(key), "")
                if text:
                    return text
            if bool(payload.get("locked")):
                return "locked"
            if bool(payload.get("halted")):
                return "halted"
            if bool(payload.get("hydrated")):
                return "running"
            return "unknown"
        return _text(value) or None

    def mapping(self, name: str) -> Mapping[str, object]:
        """Nested mapping field (empty mapping when absent/wrong-typed)."""
        value = self.get(name)
        if isinstance(value, Mapping) and name in {"pipeline"}:
            nested = _get(value, "pipeline")
            if isinstance(nested, Mapping):
                return cast(Mapping[str, object], nested)
        return _as_mapping(value)

    def sequence_field(self, name: str) -> list[object]:
        """Nested sequence field (empty list when absent/wrong-typed)."""
        value = self.get(name)
        if isinstance(value, Mapping):
            nested = _get(value, "rows", _get(value, "items", value))
            return _as_sequence(nested)
        return _as_sequence(value)


# ---------------------------------------------------------------------------
# Bridge reader
# ---------------------------------------------------------------------------


class RunnerBridgeReader:
    """Read the latest runner bridge snapshot (store when importable, else file)."""

    def __init__(self, settings: Settings, *, clock: object) -> None:
        self._settings = settings
        self._clock = clock
        self._error: str | None = None
        self._store: object | None = None
        self._module = load_optional_module(RUNNER_BRIDGE_MODULE)
        if not settings.runner_bridge_enabled:
            self._error = (
                "runner bridge is disabled (settings.runner_bridge_enabled=False); "
                "orders/fills/positions use the local simulated/fixture adapters"
            )
        elif self._module is not None:
            store_class = getattr(self._module, "RunnerBridgeStore", None)
            if callable(store_class):
                try:
                    self._store = store_class(settings)
                except Exception as exc:  # pragma: no cover - defensive wiring
                    logger.warning("RunnerBridgeStore could not be constructed: %s", exc)
                    self._store = None
        if self._module is None:
            self._error = (
                f"{RUNNER_BRIDGE_MODULE} is not available; the API reads "
                f"{self.path} directly (documented bridge payload)"
            )

    @property
    def path(self) -> Path:
        """Bridge snapshot path (``settings.runner_bridge_dir/runner_state.json``)."""
        return self._settings.runner_bridge_dir / BRIDGE_FILE_NAME

    @property
    def error(self) -> str | None:
        """Why the durable store is unavailable (``None`` when fully wired)."""
        return self._error

    @property
    def store_available(self) -> bool:
        """Whether the durable ``RunnerBridgeStore`` implementation is importable."""
        return self._store is not None

    def _now(self) -> datetime:
        return cast(datetime, self._clock())  # type: ignore[operator]

    def _compute_stale(self, raw: object) -> bool:
        updated = _as_datetime(_get(raw, "updated_at"))
        if updated is None:
            return True
        horizon = _as_float(_get(raw, "stale_after_seconds"))
        if horizon is None:
            horizon = DEFAULT_STALE_AFTER_SECONDS
        return (self._now() - updated).total_seconds() > max(0.0, horizon)

    def load(self) -> BridgeSnapshot | None:
        """Return the latest snapshot (or ``None`` when the bridge is absent)."""
        if not self._settings.runner_bridge_enabled:
            return None
        if self._store is not None:
            loader = getattr(self._store, "load_fresh", None)
            if not callable(loader):
                return self._load_file()
            try:
                outcome = loader()
            except Exception as exc:
                logger.warning("RunnerBridgeStore.load_fresh() failed: %s", exc)
                return self._load_file()
            snapshot, _store_stale = self._split_outcome(outcome)
            if snapshot is None:
                return None
            # Recompute against the API's injected clock so a deterministic test
            # clock and the runner's published timestamp agree exactly.
            return BridgeSnapshot(raw=snapshot, stale=self._compute_stale(snapshot))
        return self._load_file()

    @staticmethod
    def _split_outcome(outcome: object) -> tuple[object | None, object | None]:
        if isinstance(outcome, tuple) and len(outcome) == 2:
            return outcome[0], outcome[1]
        if outcome is None:
            return None, None
        return outcome, None

    def _load_file(self) -> BridgeSnapshot | None:
        path = self.path
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("runner bridge file %s is unreadable: %s", path, exc)
            return None
        if not isinstance(payload, Mapping):
            logger.warning("runner bridge file %s is not a JSON object", path)
            return None
        mapping = cast(Mapping[str, object], payload)
        snapshot: object = mapping.get("snapshot", mapping)
        if not isinstance(snapshot, Mapping) and not hasattr(snapshot, "__dict__"):
            return None
        return BridgeSnapshot(raw=snapshot, stale=self._compute_stale(snapshot))


# ---------------------------------------------------------------------------
# Offline runner session controller
# ---------------------------------------------------------------------------


class RunnerSessionController:
    """Singleton manager for ``src.execution.runner.OfflineRunnerSession``."""

    def __init__(
        self,
        *,
        settings: Settings,
        clock: object,
        instrument_master: InstrumentsConfig,
    ) -> None:
        self._settings = settings
        self._clock = clock
        self._master = instrument_master
        self._session: object | None = None
        self._lock = asyncio.Lock()
        self._module = load_optional_module(RUNNER_MODULE)
        session_class = (
            getattr(self._module, "OfflineRunnerSession", None)
            if self._module is not None
            else None
        )
        self._session_class = session_class if callable(session_class) else None
        self._error: str | None = None
        if self._session_class is None:
            self._error = (
                f"{RUNNER_MODULE}.OfflineRunnerSession is not available in this build; "
                "deploy the Phase-9 execution package (or use GET /api/runner/status) "
                "to drive the runner bridge"
            )

    @property
    def available(self) -> bool:
        """Whether the offline runner session class is importable."""
        return self._session_class is not None

    @property
    def session_active(self) -> bool:
        """Whether a session has been constructed and warmed."""
        return self._session is not None

    @property
    def error(self) -> str | None:
        """Human-readable reason the session is unavailable."""
        return self._error

    async def step(self, states: int) -> dict[str, object]:
        """Warm-start (once) and advance the offline runner by ``states`` states."""
        async with self._lock:
            session = await self._ensure_session()
            result = await _resolve(_invoke_step(session, states))
            snapshot = await _resolve(_invoke_session_snapshot(session))
            payload = _as_mapping(result)
            snapshot_mapping = _as_mapping(snapshot)
            processed = (
                _as_int(_get(result, "states_processed"))
                or _as_int(_get(payload, "states_processed"))
                or _as_int(_get(snapshot_mapping, "states_processed"))
                or _as_int(_get(snapshot_mapping, "states"))
                or states
            )
            mode = _text(_get(session, "mode", "offline"), "offline")
            return {
                "states_processed": processed,
                "snapshot": snapshot_mapping,
                "mode": mode,
            }

    async def _ensure_session(self) -> object:
        if self._session is not None:
            return self._session
        if self._session_class is None:
            raise RunnerSessionUnavailableError(self._error or "offline runner session unavailable")
        try:
            session = _construct_session(
                self._session_class,
                settings=self._settings,
                clock=self._clock,
                instrument_master=self._master,
            )
            warm_start = getattr(session, "warm_start", None)
            if not callable(warm_start):
                raise RunnerSessionUnavailableError(
                    "OfflineRunnerSession does not expose warm_start(); cannot guarantee "
                    "the runner hydrates before stepping"
                )
            await _resolve(warm_start())
        except RunnerSessionUnavailableError:
            raise
        except Exception as exc:
            self._error = f"OfflineRunnerSession initialisation failed: {type(exc).__name__}: {exc}"
            raise RunnerSessionUnavailableError(self._error) from exc
        self._session = session
        return session

    async def close(self) -> None:
        """Close and forget the singleton session (graceful shutdown / reset)."""
        session = self._session
        self._session = None
        if session is None:
            return
        closer = getattr(session, "close", None)
        if callable(closer):
            try:
                await _resolve(closer())
            except Exception:  # pragma: no cover - best-effort shutdown
                logger.exception("error while closing the offline runner session")


def _invoke_step(session: object, states: int) -> object:
    step = getattr(session, "step", None)
    if not callable(step):
        raise RunnerSessionUnavailableError("OfflineRunnerSession does not expose step()")
    try:
        return step(states=states)
    except TypeError:
        return step(states)


def _invoke_session_snapshot(session: object) -> object:
    snapshot = getattr(session, "snapshot", None)
    if not callable(snapshot):
        return {}
    try:
        return snapshot()
    except Exception as exc:  # pragma: no cover - defensive: snapshot is observable data
        logger.warning("OfflineRunnerSession.snapshot() failed: %s", exc)
        return {}


def _construct_session(
    session_class: object,
    *,
    settings: Settings,
    clock: object,
    instrument_master: InstrumentsConfig,
) -> object:
    """Construct the offline session, adapting to its (versioned) constructor."""
    candidates: dict[str, object] = {
        "settings": settings,
        "clock": clock,
        "instrument_master": instrument_master,
        "instruments": instrument_master,
    }
    errors: list[str] = []
    constructor = cast(type[object], session_class)
    try:
        signature = inspect.signature(constructor)
    except (TypeError, ValueError):  # pragma: no cover - builtins only
        signature = None
    if signature is not None:
        kwargs: dict[str, object] = {}
        usable = True
        for name, parameter in signature.parameters.items():
            if name == "self":
                continue
            if name in candidates:
                kwargs[name] = candidates[name]
            elif parameter.default is inspect.Parameter.empty and parameter.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
                inspect.Parameter.POSITIONAL_ONLY,
            ):
                usable = False
                break
        if usable:
            try:
                return constructor(**kwargs)
            except Exception as exc:
                errors.append(f"kwargs: {type(exc).__name__}: {exc}")
    attempts: list[tuple[tuple[object, ...], str]] = [
        ((settings,), "settings"),
        ((settings, clock), "settings, clock"),
        ((settings, clock, instrument_master), "settings, clock, instrument_master"),
        ((), "()"),
    ]
    for arguments, label in attempts:
        try:
            return constructor(*arguments)
        except Exception as exc:
            errors.append(f"{label}: {type(exc).__name__}: {exc}")
    raise RunnerSessionUnavailableError(
        "could not construct OfflineRunnerSession; attempted signatures: " + "; ".join(errors)
    )


async def _resolve(value: object) -> object:
    """Await ``value`` when it is awaitable (sync/async session APIs are both accepted)."""
    if inspect.isawaitable(value):
        return await cast(Awaitable[object], value)
    return value


# ---------------------------------------------------------------------------
# Row mapping helpers
# ---------------------------------------------------------------------------


_SIDE_VALUES = {"BUY", "SELL"}


def _side(raw: object, signed_quantity: int | None = None) -> str:
    text = _text(raw).upper()
    if text in _SIDE_VALUES:
        return text
    if signed_quantity is not None:
        return "BUY" if signed_quantity >= 0 else "SELL"
    return "BUY"


def _instrument_id(raw: Mapping[str, object]) -> str:
    for key in ("instrument_id", "root_symbol", "root", "symbol"):
        text = _text(raw.get(key))
        if text:
            return text.upper()
    return "UNKNOWN"


def _working_row_from_mapping(raw: Mapping[str, object], index: int) -> WorkingOrderRow:
    signed = _as_int(raw.get("signed_remaining"))
    quantity = (
        _as_int(raw.get("quantity"))
        or (abs(signed) if signed is not None else None)
        or _as_int(raw.get("remaining_quantity"))
        or 0
    )
    filled = max(0, _as_int(raw.get("filled_quantity")) or 0)
    remaining = _as_int(raw.get("remaining_quantity"))
    if remaining is None:
        remaining = max(0, quantity - filled)
    return WorkingOrderRow(
        order_id=_text(raw.get("order_id") or raw.get("id"), f"bridge-order-{index}"),
        instrument_id=_instrument_id(raw),
        side=cast(Literal["BUY", "SELL"], _side(raw.get("side"), signed)),
        quantity=max(0, quantity),
        filled_quantity=filled,
        remaining_quantity=max(0, remaining),
        status=_text(raw.get("status"), "SUBMITTED"),
        order_type=_text(raw.get("order_type")) or None,
        limit_price=_as_float(raw.get("limit_price")),
        updated_at=_as_datetime(raw.get("updated_at")),
    )


def _fill_row_from_mapping(raw: Mapping[str, object], index: int) -> FillRow | None:
    price = _as_float(raw.get("price"))
    quantity = _as_int(raw.get("quantity")) or _as_int(raw.get("shares"))
    if price is None or price <= 0.0 or quantity is None or quantity <= 0:
        return None
    timestamp = _as_datetime(raw.get("timestamp")) or _as_datetime(raw.get("time"))
    if timestamp is None:
        return None
    commission = _as_float(raw.get("commission")) or 0.0
    return FillRow(
        order_id=_text(raw.get("order_id") or raw.get("id"), f"bridge-fill-{index}"),
        instrument_id=_instrument_id(raw),
        side=cast(
            Literal["BUY", "SELL"], _side(raw.get("side"), _as_int(raw.get("signed_quantity")))
        ),
        quantity=quantity,
        price=price,
        commission=commission,
        timestamp=timestamp,
    )


def _working_rows(entries: Sequence[object]) -> list[WorkingOrderRow]:
    rows: list[WorkingOrderRow] = []
    for index, entry in enumerate(entries):
        mapping = _entry_mapping(entry)
        if mapping:
            rows.append(_working_row_from_mapping(mapping, index))
    return rows


def _fill_rows(entries: Sequence[object]) -> list[FillRow]:
    rows: list[FillRow] = []
    for index, entry in enumerate(entries):
        mapping = _entry_mapping(entry)
        if mapping:
            row = _fill_row_from_mapping(mapping, index)
            if row is not None:
                rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Live state service
# ---------------------------------------------------------------------------


class LiveStateService:
    """Bridge-first live state adapter consumed by the orders/runner endpoints."""

    def __init__(
        self,
        *,
        settings: Settings,
        clock: object,
        execution: object,
        instrument_master: InstrumentsConfig,
        market: MarketDataService | None = None,
    ) -> None:
        self._settings = settings
        self._clock = clock
        self._execution = execution
        self._master = instrument_master
        self._market = market
        self._reader = RunnerBridgeReader(settings, clock=clock)
        self._session = RunnerSessionController(
            settings=settings, clock=clock, instrument_master=instrument_master
        )
        self._warnings: list[str] = []
        if self._reader.error is not None:
            self._warnings.append(self._reader.error)
        if self._session.error is not None:
            self._warnings.append(self._session.error)

    # -- introspection ---------------------------------------------------------

    @property
    def warnings(self) -> tuple[str, ...]:
        """Non-fatal bridge/session wiring warnings."""
        return tuple(self._warnings)

    @property
    def bridge_error(self) -> str | None:
        """Why the durable bridge store is unavailable (``None`` when wired)."""
        return self._reader.error

    def bridge_available(self) -> bool:
        """Whether any runner bridge snapshot is currently readable."""
        return self._reader.load() is not None

    def bridge_stale(self) -> bool:
        """Whether the latest bridge snapshot is stale (or absent)."""
        snapshot = self._reader.load()
        return True if snapshot is None else snapshot.stale

    def runner_mode(self) -> str | None:
        """Runner mode reported by the bridge (``None`` when absent)."""
        snapshot = self._reader.load()
        return None if snapshot is None else snapshot.mode

    def runner_status(self) -> RunnerStatusResponse:
        """Bridge staleness + offline session state for ``GET /api/runner/status``."""
        snapshot = self._reader.load()
        session_available = self._session.available
        return RunnerStatusResponse(
            timestamp=self._now(),
            bridge_available=snapshot is not None,
            bridge_stale=True if snapshot is None else snapshot.stale,
            runner_mode=None if snapshot is None else snapshot.mode,
            mode=None if snapshot is None else snapshot.mode,
            stale=True if snapshot is None else snapshot.stale,
            session_active=self._session.session_active,
            session_available=session_available,
            runner_id=None if snapshot is None else snapshot.runner_id,
            sequence=None if snapshot is None else snapshot.sequence,
            updated_at=None if snapshot is None else snapshot.updated_at,
            stale_after_seconds=None if snapshot is None else snapshot.stale_after_seconds,
            status=None if snapshot is None else snapshot.status,
            source="runner-bridge" if snapshot is not None else "unavailable",
            message=self._session.error or self._reader.error or "",
        )

    # -- bridge-backed views ---------------------------------------------------

    def _now(self) -> datetime:
        return cast(datetime, self._clock())  # type: ignore[operator]

    def _fresh(self) -> BridgeSnapshot | None:
        snapshot = self._reader.load()
        if snapshot is None or snapshot.stale:
            return None
        return snapshot

    def _summary(self, snapshot: BridgeSnapshot | None) -> RunnerBridgeSummary | None:
        if snapshot is None:
            return None
        return RunnerBridgeSummary(
            runner_id=snapshot.runner_id,
            mode=snapshot.mode,
            sequence=snapshot.sequence,
            updated_at=snapshot.updated_at,
            stale_after_seconds=snapshot.stale_after_seconds,
            stale=snapshot.stale,
            status=snapshot.status,
        )

    def bridge_pipeline(self) -> PipelineWaterfallResponse | None:
        """Fresh runner pipeline snapshot mapped onto the waterfall contract."""
        snapshot = self._fresh()
        if snapshot is None:
            return None
        payload = snapshot.mapping("pipeline")
        if not payload:
            return None
        candidate: dict[str, object] = dict(payload)
        candidate.setdefault("timestamp", snapshot.updated_at or self._now())
        candidate.setdefault("cycle_id", f"bridge-{snapshot.sequence or 0}")
        candidate["data_source"] = "runner-bridge"
        try:
            return PipelineWaterfallResponse.model_validate(candidate)
        except ValidationError:
            return self._bridge_pipeline_summary(snapshot, payload)

    def _bridge_pipeline_summary(
        self, snapshot: BridgeSnapshot, payload: Mapping[str, object]
    ) -> PipelineWaterfallResponse:
        """Build a compact waterfall from the runner's flat pipeline scalars."""
        universe_raw = _text(payload.get("universe"))
        universe = [root.strip().upper() for root in universe_raw.split(",") if root.strip()]
        elapsed = _as_float(payload.get("elapsed_ms")) or 0.0
        gross = _as_float(payload.get("gross_leverage")) or 0.0
        net = _as_float(payload.get("net_leverage")) or 0.0
        targets = _as_int(payload.get("targets")) or 0
        tracking = _as_float(payload.get("tracking_error_pct"))
        stages = [
            PipelineStage(
                key="universe",
                label="S_t · Universe",
                formula="S_t(X≤t) → roots",
                gross=0.0,
                net=0.0,
                instrument_count=len(universe),
                note="live runner bridge snapshot",
            ),
            PipelineStage(
                key="base",
                label="w_base · Allocation",
                formula="A_t(candidates)",
                gross=gross,
                net=net,
                instrument_count=len(universe),
                note="bridge scalar summary",
            ),
            PipelineStage(
                key="timing",
                label="w_timing · Overlay",
                formula="T_t(w_base)",
                gross=gross,
                net=net,
                instrument_count=len(universe),
                note="bridge scalar summary",
            ),
            PipelineStage(
                key="risk",
                label="w_risk · Approved",
                formula="R_t(w_timing) → w_t",
                gross=gross,
                net=net,
                instrument_count=len(universe),
                note="bridge scalar summary",
            ),
            PipelineStage(
                key="targets",
                label="N_i,t · Contracts",
                formula="N = w_t·E_t / (P·M)",
                gross=gross,
                net=net,
                instrument_count=targets,
                note="live runner bridge target book",
            ),
        ]
        diagnostics = [
            PipelineDiagnostic(
                stage="Live runner bridge",
                detail=(
                    "compact pipeline summary published by the live runner "
                    f"(sequence {snapshot.sequence or 0}); full per-stage "
                    "waterfall is available from the engine when the bridge is absent"
                ),
                value="runner-bridge",
                status="info",
            )
        ]
        return PipelineWaterfallResponse(
            timestamp=snapshot.updated_at or self._now(),
            cycle_id=f"bridge-{snapshot.sequence or 0}",
            universe=universe,
            stages=stages,
            instruments=[],
            diagnostics=diagnostics,
            elapsed_ms=round(elapsed, 3),
            data_source="runner-bridge",
            tracking_error_pct=max(0.0, tracking) if tracking is not None else 0.0,
            micro_routing=[],
            tracking_error_source="translator" if tracking is not None else "fallback",
        )

    def bridge_account(self) -> AccountSnapshotResponse | None:
        """Fresh runner account/positions snapshot mapped onto the account contract."""
        snapshot = self._fresh()
        if snapshot is None:
            return None
        account = snapshot.mapping("account")
        net_liquidation = _as_float(
            account.get("net_liquidation") or account.get("NetLiquidation") or account.get("nlv")
        )
        if net_liquidation is None or net_liquidation <= 0.0:
            return None
        positions = self._bridge_position_rows(snapshot, net_liquidation)
        initial = _as_float(account.get("initial_margin_used")) or sum(
            row.initial_margin for row in positions
        )
        maintenance = _as_float(account.get("maintenance_margin_used")) or sum(
            row.maintenance_margin for row in positions
        )
        cash = _as_float(account.get("cash_balance")) or 0.0
        timestamp = snapshot.updated_at or self._now()
        rows = positions
        gross_notional = sum(abs(row.notional) for row in rows)
        net_exposure = sum(row.notional for row in rows)
        return AccountSnapshotResponse(
            timestamp=timestamp,
            net_liquidation=net_liquidation,
            cash_balance=cash,
            buying_power=_as_float(account.get("buying_power")) or 0.0,
            initial_margin_used=max(0.0, initial),
            maintenance_margin_used=max(0.0, maintenance),
            excess_liquidity=max(0.0, net_liquidation - initial),
            margin_utilization=initial / net_liquidation if net_liquidation > 0 else 0.0,
            day_pnl=_as_float(account.get("day_pnl")) or 0.0,
            day_pnl_pct=_as_float(account.get("day_pnl_pct")) or 0.0,
            gross_leverage=gross_notional / net_liquidation if net_liquidation > 0 else 0.0,
            net_exposure=net_exposure,
            equity_curve=self._bridge_equity_curve(snapshot)
            or [EquityPoint(time=timestamp, value=net_liquidation)],
            positions=rows,
            source="runner-bridge",
        )

    def _bridge_position_rows(
        self, snapshot: BridgeSnapshot, net_liquidation: float
    ) -> list[PositionRow]:
        entries: list[object]
        raw_positions = snapshot.get("positions")
        if isinstance(raw_positions, Mapping):
            entries = [
                {**cast(Mapping[str, object], value), "instrument_id": key}
                if isinstance(value, Mapping)
                else {"instrument_id": key, "quantity": value}
                for key, value in cast(Mapping[str, object], raw_positions).items()
            ]
        else:
            entries = _as_sequence(raw_positions)
        rows: list[PositionRow] = []
        for entry in entries:
            mapping = _entry_mapping(entry)
            if not mapping:
                continue
            quantity = _as_int(mapping.get("quantity")) or _as_int(mapping.get("qty")) or 0
            if quantity == 0:
                continue
            root = _instrument_id(mapping)
            spec = self._master.get(root) if self._master.has(root) else None
            multiplier = _as_float(mapping.get("multiplier")) or (
                spec.multiplier if spec is not None else 1.0
            )
            market_price = (
                _as_float(mapping.get("market_price")) or _as_float(mapping.get("price")) or 0.0
            )
            average_price = _as_float(mapping.get("average_price")) or market_price
            if market_price <= 0.0 or average_price <= 0.0:
                continue
            initial_margin = _as_float(mapping.get("initial_margin")) or (
                spec.margins.initial if spec is not None else 0.0
            )
            maintenance_margin = _as_float(mapping.get("maintenance_margin")) or (
                spec.margins.maintenance if spec is not None else 0.0
            )
            notional = quantity * market_price * multiplier
            asset_class = spec.asset_class if spec is not None else AssetClass.UNKNOWN
            rows.append(
                PositionRow(
                    instrument_id=root,
                    root_symbol=root,
                    asset_class=asset_class,
                    contract_symbol=_text(mapping.get("contract_symbol")) or None,
                    expiration=_as_date(mapping.get("expiration")),
                    days_to_roll=_as_int(mapping.get("days_to_roll")),
                    quantity=quantity,
                    average_price=average_price,
                    market_price=market_price,
                    multiplier=multiplier,
                    notional=notional,
                    weight=notional / net_liquidation if net_liquidation > 0 else 0.0,
                    unrealized_pnl=quantity * (market_price - average_price) * multiplier,
                    initial_margin=abs(quantity) * initial_margin,
                    maintenance_margin=abs(quantity) * maintenance_margin,
                )
            )
        return sorted(rows, key=lambda row: row.instrument_id)

    def _bridge_equity_curve(self, snapshot: BridgeSnapshot) -> list[EquityPoint]:
        raw_curve = snapshot.get("equity_curve")
        points: list[EquityPoint] = []
        for entry in _as_sequence(raw_curve):
            mapping = _as_mapping(entry)
            time = _as_datetime(mapping.get("time") or mapping.get("timestamp"))
            value = _as_float(mapping.get("value") or mapping.get("equity"))
            if time is not None and value is not None:
                points.append(EquityPoint(time=time, value=value))
        return points

    # -- Protocol surface ------------------------------------------------------

    async def account(self) -> AccountSnapshotResponse:
        """Bridge account when fresh, otherwise the execution adapter snapshot.

        The adapter's additive ``stale`` / ``stale_reason`` provenance is preserved:
        for an IBKR-enabled process this is either the live book or the last-known-good
        IBKR book (never the simulated fixture book). IBKR books are additionally
        enriched with the real persisted equity curve provenance and the ex-ante
        portfolio volatility of the current book; bridge/offline responses keep
        their existing semantics.
        """
        bridged = self.bridge_account()
        if bridged is not None:
            return bridged
        snapshot = await cast(Awaitable[AccountSnapshotResponse], self._execution.account())  # type: ignore[attr-defined]
        snapshot = snapshot.model_copy(update={"source": self._execution_source()})
        if self._ibkr_enabled():
            return self._enrich_ibkr_account(snapshot)
        return snapshot

    # -- IBKR attribution (v1.5.1) -------------------------------------------

    def _ibkr_enabled(self) -> bool:
        return bool(getattr(self._execution, "ibkr_enabled", False))

    def _live_equity_points(self) -> tuple[EquityPoint, ...]:
        provider = getattr(self._execution, "live_equity_points", None)
        if provider is None:
            return ()
        try:
            raw = provider() if callable(provider) else provider
        except Exception:
            logger.warning("execution live_equity_points() failed", exc_info=True)
            return ()
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            return ()
        points: list[EquityPoint] = []
        for entry in raw:
            if isinstance(entry, EquityPoint) and math.isfinite(entry.value):
                points.append(entry)
        points.sort(key=lambda point: point.time)
        return tuple(points)

    def _market_state(self) -> MarketState | None:
        market = self._market
        if market is None:
            return None
        try:
            return market.latest_state()
        except Exception:
            logger.warning("market latest_state() unavailable for volatility", exc_info=True)
            return None

    def _enrich_ibkr_account(self, snapshot: AccountSnapshotResponse) -> AccountSnapshotResponse:
        """Attach real-curve provenance and ex-ante book volatility for IBKR.

        The multi-day curve only becomes ``ibkr-live`` once the persisted history
        spans at least :data:`MIN_LIVE_CURVE_DAYS` distinct UTC calendar days; a
        burst of same-day points stays ``simulated-backfill`` (the 1D chart uses
        ``equity_curve_intraday`` instead).
        """
        points = self._live_equity_points()
        live_days = history_distinct_utc_days(points)
        updates: dict[str, object] = {
            "equity_curve_live_points": len(points),
            "equity_curve_live_days": live_days,
            "equity_curve_intraday": history_intraday_points(points),
            "equity_curve_source": "simulated-backfill",
        }
        if live_days >= MIN_LIVE_CURVE_DAYS:
            # Real recorded points only -- never prepend the synthetic spline.
            updates["equity_curve"] = [point.model_copy() for point in points]
            updates["equity_curve_source"] = "ibkr-live"
        weights = {
            row.instrument_id: row.weight
            for row in snapshot.positions
            if math.isfinite(row.weight) and abs(row.weight) > 1e-12
        }
        volatility, volatility_source = ex_ante_portfolio_volatility(weights, self._market_state())
        updates["portfolio_volatility"] = volatility
        updates["portfolio_volatility_source"] = volatility_source
        return snapshot.model_copy(update=updates)

    async def working_orders(self) -> WorkingOrdersResponse:
        """Working orders: fresh bridge first, then the execution adapter."""
        bridge = self._reader.load()
        if bridge is not None and not bridge.stale:
            return WorkingOrdersResponse(
                timestamp=self._now(),
                source="runner-bridge",
                orders=_working_rows(bridge.sequence_field("working_orders")),
                bridge_stale=False,
            )
        rows, source = self._execution_working_orders()
        return WorkingOrdersResponse(
            timestamp=self._now(),
            source=source,
            orders=rows,
            bridge_stale=None if bridge is None else bridge.stale,
        )

    async def recent_fills(self, limit: int = 50) -> RecentFillsResponse:
        """Recent fills (newest first): fresh bridge first, then the execution adapter."""
        bridge = self._reader.load()
        if bridge is not None and not bridge.stale:
            fills = _fill_rows(bridge.sequence_field("fills"))
            fills.sort(key=lambda row: row.timestamp, reverse=True)
            return RecentFillsResponse(
                timestamp=self._now(),
                source="runner-bridge",
                fills=fills[:limit],
                limit=limit,
                bridge_stale=False,
            )
        fills, source = self._execution_fills(limit)
        return RecentFillsResponse(
            timestamp=self._now(),
            source=source,
            fills=fills,
            limit=limit,
            bridge_stale=None if bridge is None else bridge.stale,
        )

    async def runner_step(self, request: RunnerStepRequest) -> RunnerStepResponse:
        """Warm-start (once) and advance the singleton offline runner session."""
        try:
            outcome = await self._session.step(request.states)
        except RunnerSessionUnavailableError:
            raise
        except Exception as exc:
            logger.exception("offline runner step failed")
            return RunnerStepResponse(
                status="error",
                states_processed=0,
                mode=request.mode,
                source="unavailable",
                message=f"offline runner step failed: {type(exc).__name__}: {exc}",
                timestamp=self._now(),
            )
        snapshot = self._reader.load()
        states_processed = _as_int(outcome.get("states_processed")) or request.states
        return RunnerStepResponse(
            status="completed",
            states_processed=max(0, states_processed),
            mode=_text(outcome.get("mode"), request.mode),
            source="runner-bridge" if snapshot is not None else self._snapshot_source(),
            bridge=self._summary(snapshot),
            message="" if snapshot is not None else "runner bridge snapshot not found after step",
            timestamp=self._now(),
        )

    async def runner_reset(self) -> RunnerStatusResponse:
        """Close the singleton offline session (bridge file is left untouched)."""
        await self._session.close()
        return self.runner_status()

    async def start(self) -> None:
        """Lifecycle hook (the offline session is created lazily on first step)."""

    async def close(self) -> None:
        """Graceful shutdown: close the singleton offline session."""
        await self._session.close()

    # -- execution adapter fallbacks ------------------------------------------

    def _execution_source(self) -> AccountSource:
        name = getattr(self._execution, "source_name", None)
        if name == "fixture":
            return "fixture"
        if name == "engine":
            return "engine"
        if name == "simulated":
            return "simulated"
        return "fixture" if "Fixture" in type(self._execution).__name__ else "simulated"

    def _snapshot_source(self) -> SnapshotSource:
        source = self._execution_source()
        if source == "fixture":
            return "fixture"
        if source == "engine":
            # Live IBKR OMS orders/fills must never be labelled as simulated data.
            return "engine"
        return "simulated"

    def _execution_working_orders(self) -> tuple[list[WorkingOrderRow], SnapshotSource]:
        provider = getattr(self._execution, "working_orders_snapshot", None)
        if not callable(provider):
            return [], self._snapshot_source()
        try:
            return _working_rows(_as_sequence(provider())), self._snapshot_source()
        except Exception:
            logger.exception("execution working_orders_snapshot() failed")
            return [], self._snapshot_source()

    def _execution_fills(self, limit: int) -> tuple[list[FillRow], SnapshotSource]:
        provider = getattr(self._execution, "recent_fills_snapshot", None)
        if not callable(provider):
            return [], self._snapshot_source()
        try:
            rows = _fill_rows(_as_sequence(provider(limit)))
        except Exception:
            logger.exception("execution recent_fills_snapshot() failed")
            return [], self._snapshot_source()
        rows.sort(key=lambda row: row.timestamp, reverse=True)
        return rows[:limit], self._snapshot_source()


def _as_date(value: object) -> date | None:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str) and value.strip():
        try:
            return date.fromisoformat(value.strip())
        except ValueError:
            return None
    return None


#: Module-level helper retained for micro-routing payload conversion by services.
def micro_routing_from_payload(payload: object) -> list[MicroRoutingEntry]:
    """Map a translator/bridge micro-routing payload onto the API schema."""
    entries: list[MicroRoutingEntry] = []
    for entry in _as_sequence(payload):
        mapping = _as_mapping(entry)
        if not mapping:
            mapping = {
                field: getattr(entry, field)
                for field in (
                    "family",
                    "standard_root",
                    "standard_quantity",
                    "micro_root",
                    "micro_quantity",
                )
                if hasattr(entry, field)
            }
        if not mapping:
            continue
        family = _text(mapping.get("family"), "unknown")
        standard_root = _text(mapping.get("standard_root"))
        micro_root = _text(mapping.get("micro_root"))
        if not standard_root or not micro_root:
            continue
        entries.append(
            MicroRoutingEntry(
                family=family,
                standard_root=standard_root.upper(),
                standard_quantity=_as_int(mapping.get("standard_quantity")) or 0,
                micro_root=micro_root.upper(),
                micro_quantity=_as_int(mapping.get("micro_quantity")) or 0,
            )
        )
    return entries
