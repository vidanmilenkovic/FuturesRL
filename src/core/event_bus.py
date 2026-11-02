"""Async pub/sub message bus used identically by live trading and backtests.

The bus is intentionally tiny and dependency-free: it provides topic-based
fan-out with sync *and* async handlers, deterministic in-flight delivery, and
full isolation of handler failures (one broken subscriber must never stall the
trading loop). Backtest parity relies on the same bus: the event-driven
backtester publishes the exact same event types that the live engine publishes,
so strategy/risk code cannot tell which engine is driving it.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from src.core.types import ScalarValue

logger = logging.getLogger(__name__)

type Handler[EventT] = Callable[[EventT], Awaitable[None] | None]


@dataclass(frozen=True, slots=True)
class Event:
    """A bus envelope carrying an arbitrary typed payload for one topic."""

    topic: str
    payload: object
    timestamp_ns: int = 0

    def __post_init__(self) -> None:
        if not self.topic:
            raise ValueError("event topic must be a non-empty string")


class EventBus[EventT]:
    """In-memory asynchronous pub/sub bus.

    ``publish`` delivers to every subscriber sequentially in subscription order
    (deterministic, important for backtest parity) and awaits async handlers so
    that ordering is preserved. Handler exceptions are logged and swallowed:
    a telemetry consumer failing must never abort a rebalance cycle.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[Handler[EventT]]] = {}
        self._lock = asyncio.Lock()
        self._published = 0
        self._failures = 0

    @property
    def published_count(self) -> int:
        """Number of events published since construction."""
        return self._published

    @property
    def failure_count(self) -> int:
        """Number of handler failures observed since construction."""
        return self._failures

    @property
    def topics(self) -> tuple[str, ...]:
        """Topics that currently have subscribers."""
        return tuple(self._handlers)

    def subscribe(self, topic: str, handler: Handler[EventT]) -> None:
        """Register ``handler`` for ``topic`` (idempotent per handler instance)."""
        if not topic:
            raise ValueError("topic must be a non-empty string")
        if not callable(handler):
            raise TypeError("handler must be callable")
        handlers = self._handlers.setdefault(topic, [])
        if handler not in handlers:
            handlers.append(handler)

    def unsubscribe(self, topic: str, handler: Handler[EventT]) -> None:
        """Remove ``handler`` from ``topic`` (no-op if absent)."""
        handlers = self._handlers.get(topic)
        if not handlers:
            return
        if handler in handlers:
            handlers.remove(handler)
        if not handlers:
            self._handlers.pop(topic, None)

    def subscriber_count(self, topic: str) -> int:
        """Number of subscribers registered for ``topic``."""
        return len(self._handlers.get(topic, ()))

    async def publish(self, topic: str, event: EventT) -> int:
        """Publish ``event`` to every subscriber of ``topic``.

        Returns the number of handlers invoked successfully.
        """
        if not topic:
            raise ValueError("topic must be a non-empty string")
        self._published += 1
        handlers = tuple(self._handlers.get(topic, ()))
        delivered = 0
        for handler in handlers:
            try:
                outcome = handler(event)
                if inspect.isawaitable(outcome):
                    await outcome
                delivered += 1
            except Exception:
                self._failures += 1
                logger.exception("event handler failed on topic %r", topic)
        return delivered

    async def publish_many(self, events: Mapping[str, EventT]) -> int:
        """Publish one event per topic, preserving mapping insertion order."""
        delivered = 0
        for topic, event in events.items():
            delivered += await self.publish(topic, event)
        return delivered

    async def drain(self, timeout: float = 5.0) -> None:
        """Best-effort wait until pending event-loop callbacks have run.

        Useful before asserting on telemetry captured by subscribers in tests
        and in the live engine's shutdown path.
        """
        await asyncio.wait_for(asyncio.sleep(0), timeout=timeout)

    def clear(self) -> None:
        """Remove every subscription and reset counters (test helper)."""
        self._handlers.clear()
        self._published = 0
        self._failures = 0


@dataclass(frozen=True, slots=True)
class TelemetryEvent:
    """Payload used on the telemetry topic for the WebSocket broadcaster."""

    kind: str
    timestamp_ns: int
    values: Mapping[str, ScalarValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("telemetry kind must be a non-empty string")
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))

    def as_dict(self) -> dict[str, object]:
        """JSON-serialisable representation."""
        return {"kind": self.kind, "timestamp_ns": self.timestamp_ns, "values": dict(self.values)}
