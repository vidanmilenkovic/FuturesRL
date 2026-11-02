"""Unit tests for the asynchronous event bus and telemetry envelope."""

from __future__ import annotations

import pytest
from src.core.event_bus import Event, EventBus, TelemetryEvent


@pytest.mark.asyncio
async def test_publish_delivers_to_sync_and_async_handlers_in_order() -> None:
    bus: EventBus[str] = EventBus()
    received: list[str] = []

    def sync_handler(event: str) -> None:
        received.append(f"sync:{event}")

    async def async_handler(event: str) -> None:
        received.append(f"async:{event}")

    bus.subscribe("orders", sync_handler)
    bus.subscribe("orders", async_handler)
    delivered = await bus.publish("orders", "ES")
    assert delivered == 2
    assert received == ["sync:ES", "async:ES"]
    assert bus.published_count == 1
    assert bus.failure_count == 0


@pytest.mark.asyncio
async def test_handler_failure_is_isolated() -> None:
    bus: EventBus[dict[str, int]] = EventBus()
    received: list[int] = []

    def broken(event: dict[str, int]) -> None:
        raise RuntimeError("telemetry consumer exploded")

    def healthy(event: dict[str, int]) -> None:
        received.append(event["value"])

    bus.subscribe("telemetry", broken)
    bus.subscribe("telemetry", healthy)
    delivered = await bus.publish("telemetry", {"value": 7})
    assert delivered == 1
    assert received == [7]
    assert bus.failure_count == 1


@pytest.mark.asyncio
async def test_subscribe_is_idempotent_and_unsubscribe_works() -> None:
    bus: EventBus[int] = EventBus()
    received: list[int] = []

    def handler(event: int) -> None:
        received.append(event)

    bus.subscribe("pnl", handler)
    bus.subscribe("pnl", handler)
    assert bus.subscriber_count("pnl") == 1
    assert bus.topics == ("pnl",)
    await bus.publish("pnl", 1)
    bus.unsubscribe("pnl", handler)
    await bus.publish("pnl", 2)
    assert received == [1]
    assert bus.subscriber_count("pnl") == 0


@pytest.mark.asyncio
async def test_publish_many_preserves_topic_order() -> None:
    bus: EventBus[str] = EventBus()
    received: list[str] = []
    bus.subscribe("a", lambda event: received.append(f"a:{event}"))  # type: ignore[arg-type]
    bus.subscribe("b", lambda event: received.append(f"b:{event}"))  # type: ignore[arg-type]
    delivered = await bus.publish_many({"a": "one", "b": "two"})
    assert delivered == 2
    assert received == ["a:one", "b:two"]


@pytest.mark.asyncio
async def test_publish_unknown_topic_is_noop() -> None:
    bus: EventBus[str] = EventBus()
    assert await bus.publish("missing", "x") == 0
    assert bus.published_count == 1


def test_subscribe_validates_arguments() -> None:
    bus: EventBus[str] = EventBus()
    with pytest.raises(ValueError, match="topic"):
        bus.subscribe("", lambda event: None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="callable"):
        bus.subscribe("topic", "not-callable")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_drain_and_clear() -> None:
    bus: EventBus[str] = EventBus()
    bus.subscribe("topic", lambda event: None)  # type: ignore[arg-type]
    await bus.publish("topic", "x")
    await bus.drain()
    bus.clear()
    assert bus.topics == ()
    assert bus.published_count == 0
    assert bus.failure_count == 0


def test_event_envelope_validation() -> None:
    event = Event(topic="orders", payload={"root": "ES"}, timestamp_ns=123)
    assert event.payload == {"root": "ES"}
    with pytest.raises(ValueError, match="topic"):
        Event(topic="", payload=None)


def test_telemetry_event_is_immutable_and_serialisable() -> None:
    event = TelemetryEvent(kind="pnl", timestamp_ns=1, values={"net_liquidation": 1_000_000.0})
    assert event.as_dict()["kind"] == "pnl"
    with pytest.raises(TypeError):
        event.values["net_liquidation"] = 0.0  # type: ignore[index]
    with pytest.raises(ValueError, match="kind"):
        TelemetryEvent(kind="", timestamp_ns=0)
