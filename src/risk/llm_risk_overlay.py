"""Tier A of the FinRL Risk-First engine: LLM structural constraint overlay.

This module implements the three modules of the FinRL Risk-First LLM architecture
described in ``AGENTS.md`` Module 3:

Module 1 - Empirical confidence & variance filter (anti-hallucination)
    Each root is sampled ``K`` times from the LLM. If the mean confidence is below
    ``tau`` or the cross-sample variance of the semantic risk score exceeds
    ``sigma2_max``, the signal is neutralised to the safe baseline
    ``S = theta_safe`` (default 1.5) so a hallucination can never move risk.

Module 2 - DRL reward shaping
    :func:`semantic_risk_penalty` implements exactly

    ``R_penalty = lambda_ * sum_i |w_i| * I(c_i >= tau) * max(0, S_i - theta_safe)``

    so ``FuturesTradingEnv`` can subtract it from the net-PnL step reward.

Module 3 - Deterministic post-hoc circuit breaker & projection
    ``apply_risk`` is synchronous, consumes the latest cached assessments and
    enforces the hard projection: ``S_i >= kappa`` with ``c_i >= tau`` zeroes the
    weight, while ``theta_safe < S_i < kappa`` scales it linearly by
    ``(kappa - S_i) / (kappa - theta_safe)``. Tier-1 macro event lockout windows
    (FOMC, CPI, NFP, EIA crude inventory, WASDE, OPEC, geopolitical shocks) either
    flatten the book or cap leverage around the scheduled release.

The LLM itself is accessed through the injectable :class:`AsyncLLMClient`
protocol; concrete HTTP adapters for OpenAI, Anthropic, DeepSeek and a local
vLLM server use ``httpx`` (imported lazily, never at module import time) with
robust JSON extraction, request retry and exponential backoff. Unit tests inject
a deterministic fake client and never touch the network.

Phase-7 async cache
-------------------
``apply_risk`` is synchronous and never awaits LLM I/O. The assessment set is a
TTL-managed :class:`LLMRiskCacheEntry` (``cache_ttl_seconds``); when it goes
stale the synchronous path schedules a fire-and-forget refresh on the running
event loop and immediately projects with the last-known-good cache.
:class:`AsyncLLMRiskCacheWorker` is the production mechanism that keeps that
cache warm off the trading/WebSocket loops (single-flight, exponential backoff,
last-known-good cache retained on failure, clean cancellation).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from configs.instruments import InstrumentsConfig, load_instruments
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from src.core.types import (
    InstrumentId,
    MarketState,
    MetadataValue,
    RiskAssessment,
    WeightVector,
    ensure_utc,
)

if TYPE_CHECKING:
    from configs.settings import RiskSettings

logger = logging.getLogger(__name__)

__all__ = [
    "ANTHROPIC_VERSION",
    "DEFAULT_LLM_SYSTEM_PROMPT",
    "AnthropicClient",
    "AsyncLLMClient",
    "AsyncLLMRiskCacheWorker",
    "DeepSeekClient",
    "GeminiClient",
    "LLMClientError",
    "LLMResponseError",
    "LLMRiskAssessmentBatch",
    "LLMRiskCacheEntry",
    "LLMRiskConfig",
    "LLMRiskOverlay",
    "LLMRiskResponse",
    "MacroEvent",
    "MacroEventCalendar",
    "OpenAIClient",
    "VLLMClient",
    "build_risk_prompt",
    "extract_json_payload",
    "semantic_risk_penalty",
]

#: Anthropic Messages API version header value.
ANTHROPIC_VERSION: str = "2023-06-01"

#: Maximum number of characters of an LLM response embedded in error messages.
_MAX_ERROR_SNIPPET: int = 200

#: Upper bound for the async worker's exponential failure backoff (seconds).
_MAX_WORKER_BACKOFF_SECONDS: float = 300.0

#: Default system prompt: strict JSON contract tailored to futures macro catalysts.
DEFAULT_LLM_SYSTEM_PROMPT = (
    "You are the FinRL-X Risk-First macro analyst for a systematic futures book. "
    "Assess every requested futures root against imminent scheduled macro catalysts "
    "(FOMC rate decisions, CPI and NFP releases, EIA crude inventory, WASDE, OPEC "
    "decisions, central-bank speeches) and unscheduled geopolitical tail events. "
    "Respond with STRICT JSON only, without markdown fences or commentary, using "
    'exactly this schema: {"assessments": [{"instrument_id": "<ROOT>", '
    '"sentiment_score": <number 1-5>, "semantic_risk_score": <number 1-5>, '
    '"confidence": <number 0-1>, "reasoning_summary": "<concise audit note>"}]}. '
    "sentiment_score is 1 (very bearish) to 5 (very bullish). semantic_risk_score "
    "is 1 (benign) to 5 (imminent macro shock, geopolitical tail event or extreme "
    "illiquidity). confidence is the calibrated probability that the assessment is "
    "correct; report low confidence when the evidence is ambiguous. Emit one record "
    "per requested root and no other keys."
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class LLMClientError(RuntimeError):
    """Raised when the LLM transport fails after all retry attempts."""


class LLMResponseError(ValueError):
    """Raised when an LLM response cannot be parsed into the strict schema."""


# ---------------------------------------------------------------------------
# Structured output schemas (Pydantic v2, strict)
# ---------------------------------------------------------------------------


class LLMRiskResponse(BaseModel):
    """One strictly validated per-root LLM risk assessment record."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    instrument_id: str = Field(min_length=1)
    sentiment_score: float = Field(ge=1.0, le=5.0)
    semantic_risk_score: float = Field(ge=1.0, le=5.0)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning_summary: str = ""


class LLMRiskAssessmentBatch(BaseModel):
    """Strictly validated batch of per-root LLM assessments (one LLM completion)."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    assessments: tuple[LLMRiskResponse, ...] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _normalise_assessments(cls, data: object) -> object:
        # JSON decodes arrays to lists; strict mode requires the declared tuple type.
        if isinstance(data, Mapping):
            assessments = data.get("assessments")
            if isinstance(assessments, list):
                normalised = dict(data)
                normalised["assessments"] = tuple(assessments)
                return normalised
        return data

    @model_validator(mode="after")
    def _unique_instruments(self) -> LLMRiskAssessmentBatch:
        seen: set[str] = set()
        for response in self.assessments:
            if response.instrument_id in seen:
                raise ValueError(f"duplicate instrument_id {response.instrument_id!r} in LLM batch")
            seen.add(response.instrument_id)
        return self

    @classmethod
    def from_json_text(cls, text: str) -> LLMRiskAssessmentBatch:
        """Parse an LLM completion into the strict batch schema.

        Tolerates markdown code fences and surrounding prose (common LLM noise)
        but never tolerates out-of-range values, missing fields or extra keys.
        """
        if not isinstance(text, str):
            raise LLMResponseError(f"LLM response must be text, got {type(text)!r}")
        payload = extract_json_payload(text)
        if isinstance(payload, Mapping):
            if "assessments" in payload:
                data: object = dict(payload)
            elif "instrument_id" in payload:
                data = {"assessments": [dict(payload)]}
            else:
                raise LLMResponseError(
                    "LLM JSON object must contain an 'assessments' list or a single "
                    "instrument_id record"
                )
        elif isinstance(payload, list):
            data = {"assessments": payload}
        else:
            raise LLMResponseError(
                f"LLM JSON payload must be an object or list, got {type(payload)!r}"
            )
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            raise LLMResponseError(f"invalid LLM risk JSON: {exc}") from exc

    def to_risk_assessments(
        self, *, timestamp: datetime, source: str = "llm"
    ) -> tuple[RiskAssessment, ...]:
        """Convert the validated batch into core :class:`RiskAssessment` objects."""
        moment = ensure_utc(timestamp)
        return tuple(
            RiskAssessment(
                instrument_id=response.instrument_id,
                timestamp=moment,
                sentiment_score=response.sentiment_score,
                semantic_risk_score=response.semantic_risk_score,
                confidence=response.confidence,
                reasoning_summary=response.reasoning_summary,
                source=source,
            )
            for response in self.assessments
        )


_JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(?P<body>.*?)```", re.DOTALL)


def _try_json(text: str) -> object:
    try:
        return json.loads(text)
    except (ValueError, TypeError) as exc:
        raise ValueError(str(exc)) from exc


def extract_json_payload(text: str) -> object:
    """Extract the first JSON payload from an LLM completion.

    Handles bare JSON, markdown-fenced JSON and JSON embedded in prose. Raises
    :class:`LLMResponseError` when no parseable payload exists.
    """
    if not isinstance(text, str) or not text.strip():
        raise LLMResponseError("LLM response is empty")
    candidates: list[str] = [match.group("body") for match in _JSON_FENCE_PATTERN.finditer(text)]
    candidates.append(text)
    for candidate in candidates:
        stripped = candidate.strip()
        if not stripped:
            continue
        try:
            return _try_json(stripped)
        except ValueError:
            pass
        for opener, closer in (("{", "}"), ("[", "]")):
            start = stripped.find(opener)
            end = stripped.rfind(closer)
            if start != -1 and end > start:
                try:
                    return _try_json(stripped[start : end + 1])
                except ValueError:
                    continue
    snippet = text[:_MAX_ERROR_SNIPPET]
    raise LLMResponseError(f"could not extract valid JSON from LLM response: {snippet!r}")


# ---------------------------------------------------------------------------
# Async LLM client protocol and HTTP adapters
# ---------------------------------------------------------------------------


@runtime_checkable
class AsyncLLMClient(Protocol):
    """Transport contract for one asynchronous LLM completion."""

    async def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        """Return the raw text completion for ``prompt``."""
        ...  # pragma: no cover - protocol declaration only


def _load_httpx() -> Any:
    """Lazily import ``httpx`` so importing this module never requires it."""
    import httpx

    return httpx


class _OpenAICompatibleClient:
    """Async chat-completions adapter shared by OpenAI-compatible providers.

    Subclasses override the endpoint/headers/payload/parse hooks; the retry and
    backoff logic lives here once.
    """

    provider: str = "openai-compatible"
    _endpoint_suffix: str = "/chat/completions"

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str | None,
        provider: str,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        backoff_base_seconds: float = 0.5,
        httpx_loader: Callable[[], Any] | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("model must be a non-empty string")
        if not base_url.strip():
            raise ValueError("base_url must be a non-empty string")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be a finite positive number")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if not math.isfinite(backoff_base_seconds) or backoff_base_seconds < 0.0:
            raise ValueError("backoff_base_seconds must be a finite non-negative number")
        self.provider = provider
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._backoff_base_seconds = backoff_base_seconds
        self._httpx_loader = httpx_loader or _load_httpx

    # -- observability ---------------------------------------------------------

    @property
    def model(self) -> str:
        """Model identifier sent to the provider."""
        return self._model

    @property
    def base_url(self) -> str:
        """Provider base URL."""
        return self._base_url

    # -- provider hooks --------------------------------------------------------

    def _endpoint(self) -> str:
        return f"{self._base_url}{self._endpoint_suffix}"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _payload(
        self, prompt: str, system_prompt: str | None, temperature: float
    ) -> dict[str, object]:
        messages: list[dict[str, object]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }

    def _parse_response(self, data: object) -> str:
        if not isinstance(data, Mapping):
            raise ValueError(f"LLM response payload must be an object, got {type(data)!r}")
        choices = data.get("choices")
        if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)) or not choices:
            raise ValueError("LLM response has no choices")
        first = choices[0]
        if not isinstance(first, Mapping):
            raise ValueError("LLM response choice is not an object")
        message = first.get("message")
        if not isinstance(message, Mapping):
            raise ValueError("LLM response choice has no message")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("LLM response message content is empty")
        return content

    # -- completion with retry/backoff ----------------------------------------

    async def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        if not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        httpx = self._httpx_loader()
        payload = self._payload(prompt, system_prompt, temperature)
        headers = self._headers()
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
                    response = await client.post(self._endpoint(), headers=headers, json=payload)
                    response.raise_for_status()
                    return self._parse_response(response.json())
            except (
                httpx.HTTPError,
                ValueError,
                KeyError,
                IndexError,
                TypeError,
            ) as exc:  # transport and malformed-payload failures are retryable
                last_error = exc
                if attempt >= self._max_retries:
                    break
                await asyncio.sleep(self._backoff_base_seconds * (2**attempt))
        raise LLMClientError(
            f"{self.provider} completion failed after {self._max_retries + 1} attempts: "
            f"{last_error!r}"
        ) from last_error


class OpenAIClient(_OpenAICompatibleClient):
    """Async adapter for the OpenAI Chat Completions API."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        backoff_base_seconds: float = 0.5,
        httpx_loader: Callable[[], Any] | None = None,
    ) -> None:
        super().__init__(
            model=model,
            base_url=base_url,
            api_key=api_key,
            provider="openai",
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            backoff_base_seconds=backoff_base_seconds,
            httpx_loader=httpx_loader,
        )


class DeepSeekClient(_OpenAICompatibleClient):
    """Async adapter for the DeepSeek OpenAI-compatible chat API."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com/v1",
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        backoff_base_seconds: float = 0.5,
        httpx_loader: Callable[[], Any] | None = None,
    ) -> None:
        super().__init__(
            model=model,
            base_url=base_url,
            api_key=api_key,
            provider="deepseek",
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            backoff_base_seconds=backoff_base_seconds,
            httpx_loader=httpx_loader,
        )


class GeminiClient(_OpenAICompatibleClient):
    """Async adapter for Google Gemini's OpenAI-compatible endpoint.

    Gemini exposes an OpenAI-compatible Chat Completions API at
    ``https://generativelanguage.googleapis.com/v1beta/openai/``; the API key is
    sent as a bearer token, so the existing OpenAI transport is reused.
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "gemini-2.5-flash",
        base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai",
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        backoff_base_seconds: float = 0.5,
        httpx_loader: Callable[[], Any] | None = None,
    ) -> None:
        super().__init__(
            model=model,
            base_url=base_url,
            api_key=api_key,
            provider="gemini",
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            backoff_base_seconds=backoff_base_seconds,
            httpx_loader=httpx_loader,
        )


class VLLMClient(_OpenAICompatibleClient):
    """Async adapter for a locally hosted vLLM OpenAI-compatible server."""

    def __init__(
        self,
        *,
        model: str = "local-model",
        base_url: str = "http://localhost:8000/v1",
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        backoff_base_seconds: float = 0.5,
        httpx_loader: Callable[[], Any] | None = None,
    ) -> None:
        super().__init__(
            model=model,
            base_url=base_url,
            api_key=api_key,
            provider="vllm",
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            backoff_base_seconds=backoff_base_seconds,
            httpx_loader=httpx_loader,
        )


class AnthropicClient(_OpenAICompatibleClient):
    """Async adapter for the Anthropic Messages API."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "claude-3-5-sonnet-20241022",
        base_url: str = "https://api.anthropic.com/v1",
        anthropic_version: str = ANTHROPIC_VERSION,
        max_tokens: int = 1024,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        backoff_base_seconds: float = 0.5,
        httpx_loader: Callable[[], Any] | None = None,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be strictly positive")
        self._anthropic_version = anthropic_version
        self._max_tokens = max_tokens
        super().__init__(
            model=model,
            base_url=base_url,
            api_key=api_key,
            provider="anthropic",
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            backoff_base_seconds=backoff_base_seconds,
            httpx_loader=httpx_loader,
        )

    def _endpoint(self) -> str:
        return f"{self._base_url}/messages"

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self._api_key or "",
            "anthropic-version": self._anthropic_version,
            "Content-Type": "application/json",
        }

    def _payload(
        self, prompt: str, system_prompt: str | None, temperature: float
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            payload["system"] = system_prompt
        return payload

    def _parse_response(self, data: object) -> str:
        if not isinstance(data, Mapping):
            raise ValueError(f"Anthropic response payload must be an object, got {type(data)!r}")
        content = data.get("content")
        if not isinstance(content, Sequence) or isinstance(content, (str, bytes)) or not content:
            raise ValueError("Anthropic response has no content blocks")
        first = content[0]
        if not isinstance(first, Mapping):
            raise ValueError("Anthropic content block is not an object")
        text = first.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Anthropic response text is empty")
        return text


# ---------------------------------------------------------------------------
# Macro event calendar (Tier-1 lockout windows)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MacroEvent:
    """One scheduled macro release with a Tier-1/2/3 importance level."""

    name: str
    timestamp: datetime
    tier: int = 1

    def __post_init__(self) -> None:
        cleaned = self.name.strip()
        if not cleaned:
            raise ValueError("macro event name must be non-empty")
        object.__setattr__(self, "name", cleaned)
        object.__setattr__(self, "timestamp", ensure_utc(self.timestamp))
        if isinstance(self.tier, bool) or not isinstance(self.tier, int):
            raise TypeError("macro event tier must be an int")
        if self.tier < 1:
            raise ValueError("macro event tier must be >= 1")


class MacroEventCalendar:
    """Injectable macro calendar used for deterministic Tier-1 lockout windows.

    Supports construction from :class:`MacroEvent` objects, from ``(name,
    timestamp, tier)`` tuples, and from a JSON/YAML configuration payload::

        {"events": [{"name": "FOMC", "timestamp": "2024-01-31T19:00:00Z", "tier": 1}]}

    All timestamps are normalised to timezone-aware UTC; naive timestamps are
    rejected (a lockout window computed in the wrong timezone is a silent risk bug).
    """

    def __init__(self, events: Sequence[MacroEvent] = ()) -> None:
        self._events: tuple[MacroEvent, ...] = tuple(
            sorted(events, key=lambda event: (event.timestamp, event.name))
        )

    @property
    def events(self) -> tuple[MacroEvent, ...]:
        """Chronologically sorted events."""
        return self._events

    @classmethod
    def from_events(
        cls, events: Sequence[MacroEvent | tuple[str, datetime, int]]
    ) -> MacroEventCalendar:
        """Build a calendar from typed events or ``(name, timestamp, tier)`` tuples."""
        parsed: list[MacroEvent] = []
        for event in events:
            if isinstance(event, MacroEvent):
                parsed.append(event)
            else:
                name, timestamp, tier = event
                parsed.append(MacroEvent(name=name, timestamp=timestamp, tier=tier))
        return cls(parsed)

    @classmethod
    def from_json(cls, payload: str | Path | Mapping[str, object]) -> MacroEventCalendar:
        """Parse a macro calendar from a JSON/YAML string, file path or mapping."""
        if isinstance(payload, Mapping):
            return cls(_parse_macro_events(payload))
        if isinstance(payload, Path):
            return cls(_parse_macro_events(_load_calendar_file(payload)))
        if not isinstance(payload, str):
            raise TypeError(f"payload must be str, Path or mapping, got {type(payload)!r}")
        stripped = payload.strip()
        if stripped.startswith(("{", "[")):
            try:
                data = json.loads(stripped)
            except ValueError as exc:
                raise ValueError(f"invalid macro calendar JSON: {exc}") from exc
            return cls(_parse_macro_events(data))
        return cls(_parse_macro_events(_load_calendar_file(Path(stripped))))

    def upcoming_events(
        self, start: datetime, end: datetime
    ) -> tuple[tuple[str, datetime, int], ...]:
        """Return ``(event_name, event_time, tier)`` triples within ``[start, end]``."""
        lower = ensure_utc(start)
        upper = ensure_utc(end)
        return tuple(
            (event.name, event.timestamp, event.tier)
            for event in self._events
            if lower <= event.timestamp <= upper
        )

    def next_event(self, now: datetime) -> MacroEvent | None:
        """Earliest scheduled event strictly after ``now``."""
        moment = ensure_utc(now)
        for event in self._events:
            if event.timestamp > moment:
                return event
        return None

    def lockout_active(
        self,
        now: datetime,
        *,
        minutes_before: int,
        minutes_after: int,
        tier: int = 1,
    ) -> bool:
        """Whether ``now`` sits inside a Tier-``tier`` lockout window.

        The window is inclusive at both boundaries:
        ``[event_time - minutes_before, event_time + minutes_after]``.
        """
        if minutes_before < 0 or minutes_after < 0:
            raise ValueError("lockout minutes must be non-negative")
        moment = ensure_utc(now)
        before = timedelta(minutes=minutes_before)
        after = timedelta(minutes=minutes_after)
        return any(
            event.tier == tier and event.timestamp - before <= moment <= event.timestamp + after
            for event in self._events
        )


def _load_calendar_file(path: Path) -> object:
    resolved = path.expanduser()
    if not resolved.is_file():
        raise FileNotFoundError(f"macro calendar file not found: {resolved}")
    text = resolved.read_text(encoding="utf-8")
    if resolved.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        return yaml.safe_load(text)
    return json.loads(text)


def _parse_event_timestamp(value: object) -> datetime:
    if isinstance(value, datetime):
        return ensure_utc(value)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"macro event timestamp must be an ISO date-time, got {value!r}")
    cleaned = value.strip()
    try:
        parsed = datetime.fromisoformat(cleaned)
    except ValueError as exc:
        raise ValueError(f"invalid macro event timestamp {value!r}: {exc}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"macro event timestamp must be timezone-aware, got {value!r}")
    return parsed.astimezone(UTC)


def _parse_macro_events(payload: object) -> tuple[MacroEvent, ...]:
    """Normalise several accepted macro-calendar payload shapes into events."""
    entries: object = payload
    if isinstance(payload, Mapping):
        for key in ("events", "macro_events", "calendar"):
            if key in payload:
                entries = payload[key]
                break
        else:
            entries = [payload] if "name" in payload or "event" in payload else []
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise ValueError("macro calendar payload must contain an event sequence")
    events: list[MacroEvent] = []
    for entry in entries:
        events.append(_parse_macro_event(entry))
    return tuple(events)


def _parse_macro_event(entry: object) -> MacroEvent:
    if isinstance(entry, Mapping):
        name = entry.get("name", entry.get("event", entry.get("type")))
        timestamp = entry.get(
            "timestamp", entry.get("datetime", entry.get("time", entry.get("date")))
        )
        tier = entry.get("tier", 1)
    elif isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)):
        if len(entry) < 2:
            raise ValueError(f"macro event tuple must be (name, timestamp[, tier]), got {entry!r}")
        name = entry[0]
        timestamp = entry[1]
        tier = entry[2] if len(entry) > 2 else 1
    else:
        raise ValueError(f"invalid macro event entry: {entry!r}")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"macro event name must be a non-empty string, got {name!r}")
    if isinstance(tier, bool) or not isinstance(tier, int):
        raise ValueError(f"macro event tier must be an int, got {tier!r}")
    return MacroEvent(name=name, timestamp=_parse_event_timestamp(timestamp), tier=tier)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_PROMPT_FEATURES: tuple[str, ...] = (
    "vol_20d",
    "vol_60d",
    "term_slope",
    "carry_annualized",
    "volume_oi_ratio",
    "turbulence",
    "ofi",
)

_SCALAR_METADATA_TYPES: tuple[type, ...] = (str, int, float, bool)


def _format_number(value: float) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.6g}"


def _format_feature(snapshot_features: Mapping[str, float], name: str) -> str:
    value = snapshot_features.get(name)
    if value is None:
        return "n/a"
    return _format_number(float(value))


def build_risk_prompt(
    state: MarketState,
    candidates: Sequence[InstrumentId],
    *,
    calendar: MacroEventCalendar | None = None,
    lookahead_hours: float = 48.0,
) -> str:
    """Build the deterministic user prompt for the Tier-A risk assessment.

    The prompt carries the as-of timestamp, the macro state metadata, upcoming
    Tier-1 scheduled events (FOMC, CPI, NFP, EIA, WASDE, OPEC, ...) and per-root
    market context, then restates the strict JSON contract.
    """
    if lookahead_hours <= 0.0 or not math.isfinite(lookahead_hours):
        raise ValueError("lookahead_hours must be a finite positive number")
    roots = tuple(dict.fromkeys(candidates))
    timestamp = state.timestamp
    lines: list[str] = [
        f"As-of UTC timestamp: {timestamp.isoformat()}",
        f"Macro regime: {state.regime or 'unknown'}",
    ]
    macro_bits: list[str] = []
    for key in sorted(state.metadata):
        value = state.metadata[key]
        if isinstance(value, _SCALAR_METADATA_TYPES) and value is not None:
            rendered = _format_number(float(value)) if isinstance(value, float) else str(value)
            macro_bits.append(f"{key}={rendered}")
    lines.append("Macro state: " + (", ".join(macro_bits) if macro_bits else "n/a"))

    horizon = timestamp + timedelta(hours=lookahead_hours)
    scheduled = calendar.upcoming_events(timestamp, horizon) if calendar is not None else ()
    if scheduled:
        lines.append("Tier-1 scheduled macro events in the horizon:")
        for name, event_time, tier in scheduled:
            lines.append(f"- {name} at {event_time.isoformat()} (tier {tier})")
    else:
        lines.append("Tier-1 scheduled macro events in the horizon: none in the injected calendar")

    lines.append("Futures roots to assess:")
    for root in roots:
        snapshot = state.instruments.get(root)
        if snapshot is None:
            lines.append(f"- {root}: no point-in-time snapshot available")
            continue
        context = [
            f"price={_format_number(snapshot.price)}",
            f"asset_class={snapshot.asset_class.value}",
        ]
        context.extend(
            f"{name}={_format_feature(snapshot.features, name)}" for name in _PROMPT_FEATURES
        )
        if snapshot.expiration is not None:
            context.append(f"expiration={snapshot.expiration.isoformat()}")
        lines.append(f"- {root}: " + ", ".join(context))

    lines.append(
        "Assess every root above against imminent macro catalysts (FOMC, CPI, NFP, EIA crude "
        "inventory, WASDE, OPEC, central-bank speeches, geopolitical shocks). Reply with the "
        "strict JSON object defined in the system prompt: one assessment per requested root, "
        "sentiment_score 1-5, semantic_risk_score 1-5, confidence 0-1 and a concise "
        "reasoning_summary. No markdown, no extra keys."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Overlay configuration
# ---------------------------------------------------------------------------


class LLMRiskConfig(BaseModel):
    """Typed configuration for :class:`LLMRiskOverlay`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    confidence_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    variance_max: float = Field(default=0.25, gt=0.0)
    sample_count: int = Field(default=3, ge=1, le=10)
    safe_score: float = Field(default=1.5, ge=1.0, le=5.0)
    critical_score: float = Field(default=4.5, ge=1.0, le=5.0)
    penalty_lambda: float = Field(default=0.5, ge=0.0)
    macro_lockout_minutes_before: int = Field(default=30, ge=0)
    macro_lockout_minutes_after: int = Field(default=30, ge=0)
    macro_lockout_action: Literal["flatten", "cap_leverage"] = "flatten"
    macro_lockout_leverage_cap: float = Field(default=0.25, gt=0.0)
    sampling_temperature: float = Field(default=0.7, ge=0.0)
    #: Time-to-live of the cached assessment set (seconds). The synchronous
    #: pipeline never awaits LLM I/O: stale caches are refreshed in the background.
    cache_ttl_seconds: float = Field(default=60.0, gt=0.0)

    @model_validator(mode="after")
    def _thresholds_ordered(self) -> LLMRiskConfig:
        if self.critical_score <= self.safe_score:
            raise ValueError("critical_score must exceed safe_score")
        return self

    @classmethod
    def from_risk_settings(cls, settings: RiskSettings) -> LLMRiskConfig:
        """Build the overlay configuration from the central risk settings."""
        return cls(
            confidence_threshold=settings.llm_confidence_threshold,
            variance_max=settings.llm_semantic_variance_max,
            sample_count=settings.llm_sample_count,
            safe_score=settings.llm_safe_score,
            critical_score=settings.llm_critical_score,
            penalty_lambda=settings.llm_risk_penalty_lambda,
            macro_lockout_minutes_before=settings.macro_lockout_minutes_before,
            macro_lockout_minutes_after=settings.macro_lockout_minutes_after,
            cache_ttl_seconds=settings.llm_cache_ttl_seconds,
        )


# ---------------------------------------------------------------------------
# Module 2: pure reward-shaping penalty
# ---------------------------------------------------------------------------


def semantic_risk_penalty(
    weights: WeightVector,
    assessments: Sequence[RiskAssessment] | Mapping[InstrumentId, RiskAssessment],
    lambda_: float = 0.5,
    theta_safe: float = 1.5,
    tau: float = 0.70,
) -> float:
    """Exact FinRL Risk-First reward penalty.

    ``R_penalty = lambda_ * sum_i |w_i| * I(c_i >= tau) * max(0, S_i - theta_safe)``

    Pure, deterministic and used by ``FuturesTradingEnv`` to shape the step reward.
    """
    penalty_weight = float(lambda_)
    if not math.isfinite(penalty_weight) or penalty_weight < 0.0:
        raise ValueError("lambda_ must be a finite non-negative number")
    safe = float(theta_safe)
    if not math.isfinite(safe) or not 1.0 <= safe <= 5.0:
        raise ValueError("theta_safe must be within [1, 5]")
    threshold = float(tau)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("tau must be within [0, 1]")
    values = assessments.values() if isinstance(assessments, Mapping) else assessments
    total = 0.0
    for assessment in values:
        if assessment.confidence < threshold:
            continue
        excess = assessment.semantic_risk_score - safe
        if excess <= 0.0:
            continue
        total += abs(weights.get(assessment.instrument_id, 0.0)) * excess
    return penalty_weight * total


# ---------------------------------------------------------------------------
# TTL-managed assessment cache
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LLMRiskCacheEntry:
    """One cached assessment set with the wall-clock window it is valid for.

    The synchronous risk projection (:meth:`LLMRiskOverlay.apply_risk`) always
    reads this last-known-good entry and never awaits LLM I/O. Background workers
    replace the whole entry atomically only after a successful sample/filter run,
    so a failed or slow completion can never degrade the cached signal.
    """

    assessments: Mapping[InstrumentId, RiskAssessment]
    refreshed_at_utc: datetime
    valid_until_utc: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "assessments", MappingProxyType(dict(self.assessments)))
        object.__setattr__(self, "refreshed_at_utc", ensure_utc(self.refreshed_at_utc))
        object.__setattr__(self, "valid_until_utc", ensure_utc(self.valid_until_utc))
        if self.valid_until_utc <= self.refreshed_at_utc:
            raise ValueError("valid_until_utc must be after refreshed_at_utc")


# ---------------------------------------------------------------------------
# LLM Risk-First overlay (Modules 1 & 3)
# ---------------------------------------------------------------------------


class LLMRiskOverlay:
    """FinRL Risk-First Tier-A overlay.

    Parameters
    ----------
    client:
        Async LLM transport used by :meth:`refresh`. Tests inject deterministic
        fake clients; no network access happens anywhere else.
    config:
        :class:`LLMRiskConfig`, a central ``RiskSettings`` instance, or ``None``.
    calendar:
        Injectable :class:`MacroEventCalendar` used for Tier-1 lockout windows.
    instruments:
        Instrument master used to validate roots. Defaults to
        ``configs/instruments.yaml``.
    source:
        ``RiskAssessment.source`` label recorded on the produced assessments.
    clock:
        Injectable wall clock (timezone-aware) used for cache TTL bookkeeping.
        Production keeps the default; tests inject a deterministic fake.
    """

    def __init__(
        self,
        client: AsyncLLMClient,
        config: LLMRiskConfig | RiskSettings | None = None,
        *,
        calendar: MacroEventCalendar | None = None,
        instruments: InstrumentsConfig | None = None,
        source: str = "llm",
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._client = client
        self._config = self._coerce_config(config)
        self._calendar = calendar
        self._instruments = instruments
        self._source = source
        self._clock: Callable[[], datetime] = clock or (lambda: datetime.now(UTC))
        self._cache_entry: LLMRiskCacheEntry | None = None
        self._last_refresh: datetime | None = None

        # -- async cache-refresh bookkeeping (fire-and-forget, never blocking) ---
        self._refresh_task: asyncio.Task[None] | None = None
        self._background_tasks: set[asyncio.Task[None]] = set()
        self._refresh_in_flight: int = 0
        self._refresh_errors: int = 0
        self._last_refresh_error: str | None = None
        self._staleness_count: int = 0
        self._last_stale_at: datetime | None = None

    # -- public API ------------------------------------------------------------

    @property
    def name(self) -> str:
        """Overlay name used in telemetry / pipeline traces."""
        return "llm_risk"

    @property
    def config(self) -> LLMRiskConfig:
        """Effective overlay configuration."""
        return self._config

    @property
    def calendar(self) -> MacroEventCalendar | None:
        """Injected macro event calendar, if any."""
        return self._calendar

    @property
    def latest_assessments(self) -> Mapping[InstrumentId, RiskAssessment]:
        """Latest cached (already filtered) assessments keyed by futures root."""
        entry = self._cache_entry
        return dict(entry.assessments) if entry is not None else {}

    @property
    def cache_entry(self) -> LLMRiskCacheEntry | None:
        """The current TTL-managed cache entry, if anything has been cached yet."""
        return self._cache_entry

    @property
    def refreshed_at_utc(self) -> datetime | None:
        """Wall-clock timestamp of the last successful cache update."""
        entry = self._cache_entry
        return entry.refreshed_at_utc if entry is not None else None

    @property
    def valid_until_utc(self) -> datetime | None:
        """Wall-clock expiry of the cached assessment set (``None`` when empty)."""
        entry = self._cache_entry
        return entry.valid_until_utc if entry is not None else None

    @property
    def last_refresh(self) -> datetime | None:
        """Timestamp of the last successful :meth:`refresh` call."""
        return self._last_refresh

    @property
    def pending_refresh_count(self) -> int:
        """Number of in-flight fire-and-forget cache refresh tasks."""
        return sum(1 for task in self._background_tasks if not task.done())

    @property
    def refresh_in_flight(self) -> int:
        """Number of :meth:`refresh` calls currently awaiting LLM I/O."""
        return self._refresh_in_flight

    @property
    def refresh_errors(self) -> int:
        """Total failed background/direct refreshes observed since construction."""
        return self._refresh_errors

    @property
    def last_refresh_error(self) -> str | None:
        """Description of the most recent refresh failure, if any."""
        return self._last_refresh_error

    @property
    def staleness_count(self) -> int:
        """How many times the synchronous path observed a stale/absent cache."""
        return self._staleness_count

    @property
    def last_stale_at(self) -> datetime | None:
        """Wall-clock time of the most recent stale-cache observation."""
        return self._last_stale_at

    def is_cache_fresh(self, now: datetime | None = None) -> bool:
        """Whether the cached assessment set is still inside its TTL window."""
        entry = self._cache_entry
        if entry is None:
            return False
        moment = ensure_utc(now) if now is not None else self._now()
        return moment < entry.valid_until_utc

    def cache_age_seconds(self, now: datetime | None = None) -> float:
        """Age of the cached assessment set in seconds (``inf`` when empty)."""
        entry = self._cache_entry
        if entry is None:
            return math.inf
        moment = ensure_utc(now) if now is not None else self._now()
        return max(0.0, (moment - entry.refreshed_at_utc).total_seconds())

    def set_cached_assessments(self, assessments: Sequence[RiskAssessment]) -> None:
        """Seed the assessment cache (deterministic replay / tests).

        The seeded set gets a fresh TTL window from the overlay clock, so the
        synchronous path does not immediately schedule a background refresh.
        """
        self._store_cache(assessments, self._now())

    async def refresh(
        self, state: MarketState, candidates: Sequence[InstrumentId]
    ) -> tuple[RiskAssessment, ...]:
        """Sample the LLM ``K`` times, run Module 1 filtering and refresh the cache."""
        self._refresh_in_flight += 1
        try:
            filtered = await self.sample_assessments(state, candidates)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._refresh_errors += 1
            self._last_refresh_error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._refresh_in_flight -= 1
        self._store_cache(filtered, self._now())
        self._last_refresh = state.timestamp
        return filtered

    async def sample_assessments(
        self,
        state: MarketState,
        candidates: Sequence[InstrumentId],
        k: int | None = None,
    ) -> tuple[RiskAssessment, ...]:
        """Sample ``K`` completions and aggregate them with the Module 1 filter."""
        roots = tuple(dict.fromkeys(candidates))
        if not roots:
            return ()
        sample_count = self._config.sample_count if k is None else int(k)
        if sample_count < 1:
            raise ValueError("k (sample count) must be >= 1")
        prompt = build_risk_prompt(state, roots, calendar=self._calendar)
        tasks = [
            self._client.complete(
                prompt,
                system_prompt=DEFAULT_LLM_SYSTEM_PROMPT,
                temperature=self._config.sampling_temperature,
            )
            for _ in range(sample_count)
        ]
        raw_responses: list[str] = list(await asyncio.gather(*tasks))
        samples: dict[InstrumentId, list[LLMRiskResponse]] = {root: [] for root in roots}
        for text in raw_responses:
            try:
                batch = LLMRiskAssessmentBatch.from_json_text(text)
            except LLMResponseError:
                continue
            for response in batch.assessments:
                if response.instrument_id in samples:
                    samples[response.instrument_id].append(response)
        return tuple(self._filter_samples(root, samples[root], state.timestamp) for root in roots)

    def apply_risk(self, weights: WeightVector, state: MarketState) -> WeightVector:
        """Deterministic post-hoc circuit breaker / macro lockout projection.

        This method is strictly synchronous and NEVER awaits LLM I/O. When the
        cached assessment set is stale/absent and an asyncio event loop is
        running, a fire-and-forget background refresh is scheduled and the
        projection immediately proceeds with the last-known-good cache (or the
        existing pass-through behaviour when nothing has been cached yet).
        """
        keys: tuple[InstrumentId, ...] = tuple(weights)
        if not weights.weights or weights.is_flat():
            return WeightVector(dict.fromkeys(keys, 0.0))
        self._maybe_schedule_refresh(state, tuple(weights))
        if self._macro_lockout_active(state.timestamp):
            if self._config.macro_lockout_action == "flatten":
                return WeightVector(dict.fromkeys(keys, 0.0))
            return self._cap_leverage(weights)

        entry = self._cache_entry
        master = self._instrument_master()
        tau = self._config.confidence_threshold
        safe = self._config.safe_score
        critical = self._config.critical_score
        output: dict[InstrumentId, float] = {}
        for root, value in weights.items():
            try:
                canonical = master.get(root).root
            except KeyError as exc:
                raise ValueError(
                    f"unknown futures root {root!r}: not declared in the instrument master"
                ) from exc
            assessment = entry.assessments.get(canonical) if entry is not None else None
            if assessment is None or assessment.confidence < tau:
                output[canonical] = value
                continue
            score = assessment.semantic_risk_score
            if score >= critical:
                output[canonical] = 0.0
            elif score > safe:
                factor = (critical - score) / (critical - safe)
                output[canonical] = value * factor
            else:
                output[canonical] = value
        return WeightVector(output)

    def penalty(
        self,
        weights: WeightVector,
        assessments: Sequence[RiskAssessment] | Mapping[InstrumentId, RiskAssessment] | None = None,
    ) -> float:
        """Apply :func:`semantic_risk_penalty` with the overlay's configured parameters."""
        selected = self.latest_assessments if assessments is None else assessments
        return semantic_risk_penalty(
            weights,
            selected,
            lambda_=self._config.penalty_lambda,
            theta_safe=self._config.safe_score,
            tau=self._config.confidence_threshold,
        )

    # -- async cache refresh ---------------------------------------------------

    async def drain_pending_refresh(self, timeout: float | None = None) -> None:
        """Await in-flight fire-and-forget cache refreshes (clean shutdown helper).

        A ``TimeoutError``/deadline expiry leaves the refreshes running (the next
        tick, a worker or :meth:`stop_background_cache` will finish them).
        """
        if timeout is not None and timeout < 0.0:
            raise ValueError("timeout must be non-negative")
        tasks = [task for task in self._background_tasks if not task.done()]
        if not tasks:
            return
        if timeout is None:
            await asyncio.gather(*tasks, return_exceptions=True)
            return
        try:
            # ``shield`` keeps the refresh tasks alive when the caller's deadline
            # expires; a drain timeout must not corrupt the cache or cancel work.
            await asyncio.wait_for(
                asyncio.shield(asyncio.gather(*tasks, return_exceptions=True)), timeout
            )
        except TimeoutError:
            return

    async def stop_background_cache(self) -> None:
        """Cancel every in-flight fire-and-forget refresh and wait for cancellation."""
        tasks = [task for task in self._background_tasks if not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._background_tasks.clear()
        self._refresh_task = None

    # -- internals -------------------------------------------------------------

    def _now(self) -> datetime:
        return ensure_utc(self._clock())

    def _store_cache(
        self, assessments: Sequence[RiskAssessment], refreshed_at_utc: datetime
    ) -> None:
        moment = ensure_utc(refreshed_at_utc)
        self._cache_entry = LLMRiskCacheEntry(
            assessments={assessment.instrument_id: assessment for assessment in assessments},
            refreshed_at_utc=moment,
            valid_until_utc=moment + timedelta(seconds=self._config.cache_ttl_seconds),
        )

    def _maybe_schedule_refresh(
        self, state: MarketState, candidates: Sequence[InstrumentId]
    ) -> bool:
        """Schedule a non-blocking cache refresh when stale; return whether scheduled.

        With no running event loop (synchronous backtest runner) this records the
        staleness and returns ``False`` instead of raising -- the synchronous
        pipeline simply keeps using the cache it has.
        """
        moment = self._now()
        if self.is_cache_fresh(moment):
            return False
        self._staleness_count += 1
        self._last_stale_at = moment
        if not candidates:
            return False
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return False
        if self._refresh_in_flight > 0:
            return False
        if self._refresh_task is not None and not self._refresh_task.done():
            return False
        task = loop.create_task(self._background_refresh(state, tuple(candidates)))
        self._refresh_task = task
        self._background_tasks.add(task)
        task.add_done_callback(self._on_background_refresh_done)
        return True

    async def _background_refresh(
        self, state: MarketState, candidates: tuple[InstrumentId, ...]
    ) -> None:
        try:
            await self.refresh(state, candidates)
        except asyncio.CancelledError:
            raise
        except Exception:
            return

    def _on_background_refresh_done(self, task: asyncio.Task[None]) -> None:
        self._background_tasks.discard(task)
        if task is self._refresh_task:
            self._refresh_task = None
        if not task.cancelled():
            # Retrieve the exception (if any) so asyncio never warns about it;
            # refresh() has already counted the failure.
            task.exception()

    @staticmethod
    def _coerce_config(config: LLMRiskConfig | RiskSettings | None) -> LLMRiskConfig:
        if config is None:
            return LLMRiskConfig()
        if isinstance(config, LLMRiskConfig):
            return config
        if hasattr(config, "llm_confidence_threshold"):
            return LLMRiskConfig.from_risk_settings(config)
        raise TypeError(
            f"config must be an LLMRiskConfig or configs.settings.RiskSettings, "
            f"got {type(config)!r}"
        )

    def _instrument_master(self) -> InstrumentsConfig:
        return self._instruments if self._instruments is not None else load_instruments()

    def _macro_lockout_active(self, timestamp: datetime) -> bool:
        if self._calendar is None:
            return False
        return self._calendar.lockout_active(
            timestamp,
            minutes_before=self._config.macro_lockout_minutes_before,
            minutes_after=self._config.macro_lockout_minutes_after,
            tier=1,
        )

    def _cap_leverage(self, weights: WeightVector) -> WeightVector:
        cap = self._config.macro_lockout_leverage_cap
        gross = weights.gross
        if gross <= cap or gross <= 0.0:
            return weights
        return weights.scaled(cap / gross)

    def _filter_samples(
        self,
        root: InstrumentId,
        samples: Sequence[LLMRiskResponse],
        timestamp: datetime,
    ) -> RiskAssessment:
        """Module 1: confidence & cross-sample variance anti-hallucination filter."""
        if not samples:
            metadata: dict[str, MetadataValue] = {
                "filtered": True,
                "sample_count": 0.0,
                "reason": "no valid LLM response",
            }
            return RiskAssessment(
                instrument_id=root,
                timestamp=timestamp,
                sentiment_score=3.0,
                semantic_risk_score=self._config.safe_score,
                confidence=0.0,
                reasoning_summary="no valid LLM response; neutralised to safe baseline",
                source=self._source,
                metadata=metadata,
            )
        count = len(samples)
        mean_confidence = math.fsum(sample.confidence for sample in samples) / count
        mean_sentiment = math.fsum(sample.sentiment_score for sample in samples) / count
        mean_score = math.fsum(sample.semantic_risk_score for sample in samples) / count
        variance = (
            math.fsum((sample.semantic_risk_score - mean_score) ** 2 for sample in samples) / count
        )
        filtered = (
            mean_confidence < self._config.confidence_threshold
            or variance > self._config.variance_max
        )
        metadata = {
            "filtered": filtered,
            "sample_count": float(count),
            "mean_confidence": mean_confidence,
            "mean_semantic_risk_score": mean_score,
            "semantic_risk_variance": variance,
        }
        if filtered:
            reasoning = (
                "filtered: LLM confidence below threshold or cross-sample variance too high; "
                "signal neutralised to safe baseline"
            )
            score = self._config.safe_score
        else:
            reasoning = samples[0].reasoning_summary or "aggregated LLM risk assessment"
            score = mean_score
        return RiskAssessment(
            instrument_id=root,
            timestamp=timestamp,
            sentiment_score=mean_sentiment,
            semantic_risk_score=score,
            confidence=mean_confidence,
            reasoning_summary=reasoning,
            source=self._source,
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# Background TTL cache worker (production mechanism)
# ---------------------------------------------------------------------------


class AsyncLLMRiskCacheWorker:
    """Single-flight background refresher that keeps the LLM cache warm.

    The synchronous :meth:`LLMRiskOverlay.apply_risk` projection must never wait
    on a slow K-sample multi-provider LLM completion (a 3-sample call can take
    seconds and would stall the WebSocket event loop / rebalance tick). This
    worker refreshes the overlay's TTL-managed cache on an interval derived from
    the configured cache TTL and keeps the last-known-good assessments on failure,
    backing off exponentially (capped) instead of hammering the provider.

    Parameters
    ----------
    overlay:
        The :class:`LLMRiskOverlay` whose cache is refreshed.
    state_provider:
        Callable returning the latest :class:`MarketState` (or ``None`` when no
        state is available yet; the tick is skipped).
    candidates_provider:
        Callable deriving the roots to assess from the current state.
    interval_seconds:
        Optional refresh cadence. The effective cadence is
        ``min(cache_ttl_seconds, interval_seconds)`` (TTL-based when omitted).
    sleep, clock:
        Injectable async sleep / wall clock for deterministic tests.
    max_failures:
        Optional consecutive-failure threshold after which the loop stops.
    stop_event:
        Optional external stop signal; :meth:`stop` also sets it.
    """

    def __init__(
        self,
        overlay: LLMRiskOverlay,
        *,
        state_provider: Callable[[], MarketState | None],
        candidates_provider: Callable[[MarketState], Sequence[InstrumentId]],
        interval_seconds: float | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        clock: Callable[[], datetime] | None = None,
        max_failures: int | None = None,
        stop_event: asyncio.Event | None = None,
    ) -> None:
        if not isinstance(overlay, LLMRiskOverlay):
            raise TypeError(f"overlay must be an LLMRiskOverlay, got {type(overlay)!r}")
        if not callable(state_provider):
            raise TypeError("state_provider must be callable")
        if not callable(candidates_provider):
            raise TypeError("candidates_provider must be callable")
        ttl = overlay.config.cache_ttl_seconds
        if interval_seconds is None:
            interval = ttl
        else:
            if not math.isfinite(interval_seconds) or interval_seconds <= 0.0:
                raise ValueError("interval_seconds must be a finite positive number")
            interval = min(ttl, float(interval_seconds))
        if max_failures is not None and (
            isinstance(max_failures, bool) or not isinstance(max_failures, int) or max_failures < 1
        ):
            raise ValueError("max_failures must be a positive int or None")
        self._overlay = overlay
        self._state_provider = state_provider
        self._candidates_provider = candidates_provider
        self._interval = interval
        self._sleep: Callable[[float], Awaitable[None]] = sleep or asyncio.sleep
        self._clock: Callable[[], datetime] = clock or (lambda: datetime.now(UTC))
        self._max_failures = max_failures
        self._stop_event = stop_event if stop_event is not None else asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._running_once = False
        self._refreshes = 0
        self._failures = 0
        self._consecutive_failures = 0
        self._last_error: str | None = None
        self._last_run_at: datetime | None = None

    # -- lifecycle -------------------------------------------------------------

    def start(self) -> None:
        """Start the background loop (requires a running asyncio event loop)."""
        if self.is_running:
            return
        loop = asyncio.get_running_loop()
        self._stop_event.clear()
        self._task = loop.create_task(self._run_loop())

    async def stop(self) -> None:
        """Stop the loop, cancel any in-flight refresh and await clean teardown."""
        self._stop_event.set()
        task = self._task
        self._task = None
        if task is not None:
            if not task.done():
                task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await self._overlay.stop_background_cache()

    async def run_once(self) -> bool:
        """Run exactly one cache refresh; ``False`` when skipped or failed.

        Single-flight: concurrent callers get ``False`` instead of stacking
        overlapping LLM completion batches.
        """
        if self._running_once:
            return False
        self._running_once = True
        try:
            try:
                state = self._state_provider()
            except Exception as exc:
                self._record_failure(exc)
                return False
            if state is None:
                return False
            try:
                candidates = tuple(self._candidates_provider(state))
            except Exception as exc:
                self._record_failure(exc)
                return False
            if not candidates:
                return False
            if self._overlay.refresh_in_flight > 0:
                return False
            try:
                await self._overlay.refresh(state, candidates)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._record_failure(exc)
                return False
            self._refreshes += 1
            self._consecutive_failures = 0
            self._last_run_at = self._clock()
            return True
        finally:
            self._running_once = False

    # -- observability ---------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """Whether the background loop task is alive."""
        return self._task is not None and not self._task.done()

    @property
    def interval_seconds(self) -> float:
        """Effective refresh cadence (``min(ttl, configured interval)``)."""
        return self._interval

    @property
    def refreshes(self) -> int:
        """Number of successful cache refreshes performed by this worker."""
        return self._refreshes

    @property
    def failures(self) -> int:
        """Number of failed refresh attempts (provider/LLM/transport errors)."""
        return self._failures

    @property
    def consecutive_failures(self) -> int:
        """Current run of consecutive failures (drives the exponential backoff)."""
        return self._consecutive_failures

    @property
    def last_error(self) -> str | None:
        """Description of the most recent worker failure, if any."""
        return self._last_error

    @property
    def last_run_at(self) -> datetime | None:
        """Wall-clock time of the last successful worker refresh."""
        return self._last_run_at

    @property
    def stop_event(self) -> asyncio.Event:
        """The stop event; ``set()`` it to request an orderly loop exit."""
        return self._stop_event

    # -- internals -------------------------------------------------------------

    async def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            await self.run_once()
            if self._stop_event.is_set():
                break
            if self._max_failures is not None and self._consecutive_failures >= self._max_failures:
                break
            await self._sleep(self._next_delay())

    def _next_delay(self) -> float:
        if self._consecutive_failures <= 0:
            return self._interval
        exponent = min(self._consecutive_failures, 16)
        return min(_MAX_WORKER_BACKOFF_SECONDS, self._interval * (2.0**exponent))

    def _record_failure(self, exc: BaseException) -> None:
        self._failures += 1
        self._consecutive_failures += 1
        self._last_error = f"{type(exc).__name__}: {exc}"
