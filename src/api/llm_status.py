"""LLM provider health/status and bounded, side-effect-free ping (workstation UX).

The Risk Monitor must be able to answer "is the configured LLM actually usable?"
without ever inventing health. This module derives the provider/model/key state
from :class:`~configs.settings.Settings`, never builds a transport for offline or
key-less configurations, and performs an explicitly bounded ping against the live
transport when asked.

Design rules
------------
* ``offline`` (or a key-required provider without a key) never touches the
  network: mode/status are ``offline_stub`` and ``POST /api/risk/llm-ping`` is a
  pure state update.
* A live ping is bounded (default 5s) and classifies failures honestly:
  ``auth_error`` for 401/403-style authentication failures, ``unreachable`` for
  timeouts/transport/other errors.
* No order, risk-budget or trading side effects: the ping only issues a tiny
  completion request; it never mutates the overlays' assessment cache.
* ``cache_entries`` comes from the risk overlay's assessment cache through an
  injected provider (0 when unavailable).
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Callable
from datetime import UTC, datetime

from configs.settings import Settings

from src.api.schemas import LLMHealthStatus, LLMProviderMode, LLMStatusResponse
from src.risk.llm_clients import build_llm_client
from src.risk.llm_risk_overlay import AsyncLLMClient

logger = logging.getLogger(__name__)

__all__ = [
    "LLM_PING_PROMPT",
    "LLM_PING_TIMEOUT_SECONDS",
    "LLMStatusService",
]

#: Tiny, deterministic prompt used only to prove the transport answers.
LLM_PING_PROMPT: str = 'Health check only. Reply with exactly {"ok": true} and nothing else.'

#: Hard upper bound on the network probe (seconds).
LLM_PING_TIMEOUT_SECONDS: float = 5.0

#: Providers whose transport requires an API key in this platform.
_KEY_REQUIRED_PROVIDERS: frozenset[str] = frozenset({"openai", "anthropic", "deepseek", "gemini"})

#: Error text markers that classify a failed completion as an auth problem.
_AUTH_MARKERS: tuple[str, ...] = (
    "401",
    "403",
    "unauthorized",
    "unauthenticated",
    "authentication",
    "invalid api key",
    "invalid_api_key",
    "api key",
    "forbidden",
    "permission",
)


class LLMStatusService:
    """Derive and probe the configured LLM provider status (never invents health)."""

    def __init__(
        self,
        *,
        settings: Settings,
        clock: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] | None = None,
        client_factory: Callable[[], AsyncLLMClient] | None = None,
        ping_timeout_seconds: float = LLM_PING_TIMEOUT_SECONDS,
        cache_entries_provider: Callable[[], int] | None = None,
    ) -> None:
        if not math.isfinite(ping_timeout_seconds) or ping_timeout_seconds <= 0.0:
            raise ValueError("ping_timeout_seconds must be a finite positive number")
        self._settings = settings
        self._clock: Callable[[], datetime] = clock or (lambda: datetime.now(UTC))
        self._monotonic: Callable[[], float] = monotonic or (lambda: 0.0)
        self._client_factory = client_factory or (lambda: build_llm_client(self._settings))
        self._ping_timeout = float(ping_timeout_seconds)
        self._cache_entries_provider = cache_entries_provider
        self._provider = self._resolve_provider()
        self._model = str(getattr(settings, "llm_model", "") or "").strip()
        self._api_key_configured = bool(self._resolve_api_key())
        self._latency_ms: float | None = None
        self._checked_at: datetime | None = None
        self._error_detail: str | None = None
        self._status: LLMHealthStatus = (
            "offline_stub" if self.mode == "offline_stub" else "unreachable"
        )
        if self.mode == "live_api":
            self._error_detail = "LLM endpoint has not been pinged yet"
        elif self._provider != "offline":
            self._error_detail = (
                f"API key not configured for provider {self._provider!r}; "
                "offline deterministic stub in use"
            )

    # -- derivation ------------------------------------------------------------

    def _resolve_provider(self) -> str:
        raw = getattr(self._settings, "llm_provider_value", None)
        if not isinstance(raw, str) or not raw.strip():
            raw = getattr(self._settings, "llm_provider", "offline")
        return str(raw or "offline").strip().lower() or "offline"

    def _resolve_api_key(self) -> str | None:
        getter = getattr(self._settings, "llm_api_key_value", None)
        value = getter() if callable(getter) else getter
        return str(value) if value else None

    @property
    def provider(self) -> str:
        """Configured provider id (lower-case, never empty)."""
        return self._provider

    @property
    def model(self) -> str:
        """Configured model identifier (may be empty for malformed settings)."""
        return self._model

    @property
    def api_key_configured(self) -> bool:
        """Whether an API key is configured (keyless vLLM reports ``False``)."""
        return self._api_key_configured

    @property
    def mode(self) -> LLMProviderMode:
        """``offline_stub`` when no live transport can be used, else ``live_api``."""
        if self._provider == "offline":
            return "offline_stub"
        if self._provider in _KEY_REQUIRED_PROVIDERS and not self._api_key_configured:
            return "offline_stub"
        return "live_api"

    def _cache_entries(self) -> int:
        provider = self._cache_entries_provider
        if provider is None:
            return 0
        try:
            value = int(provider())
        except Exception:
            logger.warning("LLM cache-entry provider failed", exc_info=True)
            return 0
        return max(0, value)

    # -- public API ------------------------------------------------------------

    def status(self) -> LLMStatusResponse:
        """Current status snapshot (no network I/O, never invents health)."""
        return LLMStatusResponse(
            provider=self._provider,
            model=self._model,
            api_key_configured=self._api_key_configured,
            mode=self.mode,
            status=self._status,
            last_latency_ms=self._latency_ms,
            last_checked_at=self._checked_at,
            cache_entries=self._cache_entries(),
            error_detail=self._error_detail,
        )

    async def ping(self) -> LLMStatusResponse:
        """Probe the configured transport once (bounded) and update the status."""
        checked_at = self._clock()
        if self.mode == "offline_stub":
            self._status = "offline_stub"
            self._latency_ms = None
            self._checked_at = checked_at
            self._error_detail = (
                None
                if self._provider == "offline"
                else (
                    "API key not configured for provider "
                    f"{self._provider!r}; offline deterministic stub in use"
                )
            )
            return self.status()
        started = self._monotonic()
        try:
            client = self._client_factory()
            response = await asyncio.wait_for(
                client.complete(
                    LLM_PING_PROMPT,
                    system_prompt=None,
                    temperature=0.0,
                ),
                timeout=self._ping_timeout,
            )
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            self._record_failure(
                "unreachable",
                f"LLM ping timed out after {self._ping_timeout:g}s",
                checked_at,
            )
        except Exception as exc:
            status, detail = self._classify_failure(exc)
            self._record_failure(status, detail, checked_at)
        else:
            elapsed_ms = max(0.0, (self._monotonic() - started) * 1000.0)
            if isinstance(response, str) and response.strip():
                self._status = "online"
                self._latency_ms = round(elapsed_ms, 3)
                self._checked_at = checked_at
                self._error_detail = None
            else:
                self._record_failure(
                    "unreachable", "LLM ping returned an empty completion", checked_at
                )
        return self.status()

    # -- internals -------------------------------------------------------------

    def _record_failure(self, status: LLMHealthStatus, detail: str, checked_at: datetime) -> None:
        self._status = status
        self._latency_ms = None
        self._checked_at = checked_at
        self._error_detail = detail
        logger.warning("LLM health probe failed (%s): %s", status, detail)

    @staticmethod
    def _classify_failure(exc: BaseException) -> tuple[LLMHealthStatus, str]:
        """Classify a failed completion as ``auth_error`` or ``unreachable``."""
        text = LLMStatusService._error_chain_text(exc).lower()
        if any(marker in text for marker in _AUTH_MARKERS):
            return "auth_error", f"{type(exc).__name__}: {exc}"
        if isinstance(exc, TimeoutError) or "timeout" in text or "timed out" in text:
            return "unreachable", f"{type(exc).__name__}: {exc}"
        return "unreachable", f"{type(exc).__name__}: {exc}"

    @staticmethod
    def _error_chain_text(exc: BaseException) -> str:
        parts: list[str] = []
        current: BaseException | None = exc
        for _ in range(6):
            if current is None:
                break
            parts.append(f"{type(current).__name__}: {current}")
            current = current.__cause__ or current.__context__
        return " | ".join(parts)
