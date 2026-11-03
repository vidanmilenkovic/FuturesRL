"""News feed adapter for the LLM Risk-First monitor (Phase 9).

``src.data.news_ingestor.NewsIngestor`` (delivered concurrently by the data
layer) owns the real RSS/IBKR headline ingestion. This module is the API-side
adapter:

* constructed once when the container is built (never on the request path),
* refreshed **in the background** on ``settings.news_poll_interval_seconds`` so
  ``GET /api/risk/state`` never blocks the event loop on network I/O,
* falls back to the deterministic ``src.api.fixtures.news_feed`` when the
  ingestor is unavailable or has not produced anything yet, and reports the
  provenance through ``RiskStateResponse.news_source``.

The ingestor is imported lazily and its (versioned) async API is adapted
tolerantly; tests inject deterministic doubles or run with the module absent.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
from collections.abc import Awaitable, Mapping, Sequence
from datetime import UTC, datetime
from typing import cast

from configs.settings import Settings

from src.api import fixtures
from src.api.live import load_optional_module
from src.api.schemas import NewsEvent, RiskAssessmentRow, Severity

logger = logging.getLogger(__name__)

__all__ = ["NewsFeedService"]

NEWS_INGESTOR_MODULE = "src.data.news_ingestor"

_SEVERITY_VALUES = {"ok", "info", "warn", "critical"}


def _as_mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return cast(Mapping[str, object], value)
    return {}


def _get(source: object, name: str, default: object = None) -> object:
    if isinstance(source, Mapping):
        return cast(Mapping[str, object], source).get(name, default)
    return getattr(source, name, default)


def _text(value: object, default: str = "") -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def _float(value: object, default: float) -> float:
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return default
    return default


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _datetime(value: object, default: datetime) -> datetime:
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo is not None else default
    if isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return default
        return parsed.astimezone(UTC) if parsed.tzinfo is not None else default
    return default


def _roots(value: object) -> list[str]:
    if isinstance(value, str):
        return [value.upper()] if value.strip() else []
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        return [str(item).upper() for item in cast(Sequence[object], value) if str(item).strip()]
    return []


def _severity(value: object) -> str:
    text = _text(value).lower()
    return text if text in _SEVERITY_VALUES else "info"


def _news_event(item: object, index: int, now: datetime) -> NewsEvent | None:
    mapping = _as_mapping(item)
    source = mapping if mapping else item
    headline = _text(_get(source, "headline")) or _text(_get(source, "title"))
    if not headline:
        return None
    event_id = _text(_get(source, "id"), f"news-ingested-{index}")
    timestamp = _datetime(_get(source, "timestamp") or _get(source, "published_at"), now)
    sentiment = _clamp(_float(_get(source, "sentiment_score"), 3.0), 1.0, 5.0)
    return NewsEvent(
        id=event_id,
        timestamp=timestamp,
        headline=headline,
        source=_text(_get(source, "source"), "news-ingestor"),
        roots=_roots(_get(source, "roots") or _get(source, "instruments")),
        sentiment_score=sentiment,
        severity=cast(Severity, _severity(_get(source, "severity"))),
        summary=_text(_get(source, "summary"), headline),
    )


def _coerce_items(payload: object, now: datetime) -> list[NewsEvent]:
    if payload is None:
        return []
    if isinstance(payload, Mapping):
        nested = _as_mapping(payload)
        for key in ("items", "news", "headlines", "rows"):
            candidate = nested.get(key)
            if candidate is not None:
                payload = candidate
                break
    if isinstance(payload, Sequence) and not isinstance(payload, bytes | bytearray | str):
        items = cast(Sequence[object], payload)
    else:
        return []
    events: list[NewsEvent] = []
    for index, item in enumerate(items):
        event = _news_event(item, index, now)
        if event is not None:
            events.append(event)
    return events


async def _resolve(value: object) -> object:
    if inspect.isawaitable(value):
        return await cast(Awaitable[object], value)
    return value


class NewsFeedService:
    """Background-refreshed news cache with an offline deterministic fallback."""

    def __init__(self, *, settings: Settings, clock: fixtures.Clock = fixtures.utc_now) -> None:
        self._settings = settings
        self._clock = clock
        self._ingestor: object | None = None
        self._module = load_optional_module(NEWS_INGESTOR_MODULE)
        self._error: str | None = None
        self._items: list[NewsEvent] = []
        self._source = "fixture"
        self._last_refresh: datetime | None = None
        self._task: asyncio.Task[None] | None = None
        if self._module is not None:
            ingestor_class = getattr(self._module, "NewsIngestor", None)
            if callable(ingestor_class):
                try:
                    self._ingestor = ingestor_class(settings)
                    self._source = str(getattr(self._ingestor, "source_kind", "news-ingestor"))
                except Exception as exc:  # pragma: no cover - defensive wiring
                    logger.warning("NewsIngestor could not be constructed: %s", exc)
                    self._ingestor = None
            else:
                self._error = f"{NEWS_INGESTOR_MODULE} does not expose NewsIngestor"
        else:
            self._error = f"{NEWS_INGESTOR_MODULE} is not available; using fixture headlines"

    # -- introspection ---------------------------------------------------------

    @property
    def available(self) -> bool:
        """Whether the real news ingestor is wired."""
        return self._ingestor is not None

    @property
    def source(self) -> str:
        """Provenance label reported as ``RiskStateResponse.news_source``."""
        return self._source if self._items else "fixture"

    @property
    def error(self) -> str | None:
        """Why the ingestor is unavailable (``None`` when wired)."""
        return self._error

    @property
    def last_refresh(self) -> datetime | None:
        """Timestamp of the last successful ingestion refresh."""
        return self._last_refresh

    def items(self) -> list[NewsEvent]:
        """Cached ingested headlines (empty until the first successful refresh)."""
        return list(self._items)

    # -- lifecycle -------------------------------------------------------------

    async def start(self) -> None:
        """Refresh once and start the background polling loop (best effort)."""
        if self._ingestor is None:
            return
        await self.refresh()
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._poll_loop(), name="api-news-poll")

    async def stop(self) -> None:
        """Stop the background polling loop."""
        task = self._task
        self._task = None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def refresh(self) -> list[NewsEvent]:
        """Best-effort ingestion refresh; failures keep the last known-good cache."""
        ingestor = self._ingestor
        if ingestor is None:
            return []
        fetch = getattr(ingestor, "fetch_async", None)
        if not callable(fetch):
            self._error = "NewsIngestor does not expose fetch_async()"
            return []
        try:
            payload = await _resolve(fetch(limit=self._settings.news_max_items, roots=None))
        except Exception as exc:
            self._error = f"NewsIngestor.fetch_async() failed: {type(exc).__name__}: {exc}"
            logger.warning(self._error)
            return []
        events = _coerce_items(payload, self._clock())
        if events:
            self._items = events
            self._last_refresh = self._clock()
            self._source = str(getattr(ingestor, "source_kind", self._source) or self._source)
        return events

    async def _poll_loop(self) -> None:
        interval = self._settings.news_poll_interval_seconds
        try:
            while True:
                await asyncio.sleep(interval)
                await self.refresh()
        except asyncio.CancelledError:
            raise


def fallback_news(assessments: Sequence[RiskAssessmentRow], now: datetime) -> list[NewsEvent]:
    """Deterministic fixture headlines (used when no ingested items are cached)."""
    return fixtures.news_feed(assessments, now)
