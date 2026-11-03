"""LLM provider construction for the Tier-A Risk-First engine.

Phase-9 live inference switch: :func:`build_llm_client` maps the process
:class:`~configs.settings.Settings` onto one concrete
:class:`~src.risk.llm_risk_overlay.AsyncLLMClient` implementation.

* ``offline`` (default) - :class:`OfflineLLMClient`, a deterministic, in-process
  client that never touches the network. It parses the futures roots out of the
  risk prompt and answers with the strict JSON contract consumed by
  :class:`~src.risk.llm_risk_overlay.LLMRiskAssessmentBatch.from_json_text`.
* ``openai`` / ``anthropic`` / ``deepseek`` - the existing async HTTP adapters
  with ``settings.llm_model`` and the API key from ``FUTURESRLX_LLM_API_KEY``.
* ``vllm`` - the OpenAI-compatible local vLLM adapter (default base URL
  ``http://localhost:8000/v1``).

``Settings.llm_base_url`` (``FUTURESRLX_LLM_BASE_URL``) optionally overrides the
endpoint of any HTTP adapter (OpenAI-compatible proxy, self-hosted gateway,
custom vLLM host); an empty value keeps the adapter's documented default. The
offline provider ignores it entirely (no transport is created).

The HTTP adapters import ``httpx`` lazily at request time, so importing this
module (or constructing any client) performs no network I/O and does not import
``httpx``/``asyncio`` transports at module load. Missing credentials, unsupported
providers and blank model names raise actionable :class:`ValueError` messages
that name the exact environment variable to fix.

Notes
-----
* Base URLs come from each adapter's documented constructor default; API keys,
  models and the provider id are owned by :class:`~configs.settings.Settings`.
* The offline client is deterministic: repeated completions for the same prompt
  are byte-identical, which makes live-paper demos, API fixtures and tests
  reproducible without credentials or network access.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, ClassVar

from configs.instruments import load_instruments

from src.risk.llm_risk_overlay import (
    AnthropicClient,
    AsyncLLMClient,
    DeepSeekClient,
    OpenAIClient,
    VLLMClient,
)

if TYPE_CHECKING:
    from configs.instruments import InstrumentsConfig
    from configs.settings import Settings

__all__ = ["OfflineLLMClient", "build_llm_client"]

#: Providers accepted by :func:`build_llm_client` (mirrors ``Settings.llm_provider``).
_SUPPORTED_PROVIDERS: tuple[str, ...] = ("offline", "openai", "anthropic", "deepseek", "vllm")

#: Upper-case token in the prompt; only tokens present in the instrument master
#: are treated as futures roots (macro event names such as FOMC/CPI never match).
_ROOT_TOKEN_PATTERN = re.compile(r"(?<![A-Za-z0-9])([A-Z0-9]{1,6})(?![A-Za-z0-9])")


class OfflineLLMClient:
    """Deterministic, no-network :class:`AsyncLLMClient` for offline mode.

    The risk prompt built by :func:`~src.risk.llm_risk_overlay.build_risk_prompt`
    lists every requested futures root as ``- <ROOT>: ...``. The offline client
    extracts those roots (filtered against the instrument master so macro event
    names like ``FOMC``/``CPI`` are never mistaken for instruments) and returns a
    strict JSON assessment batch with deliberately benign, fixed scores:
    frozen sentiment 3.0, semantic risk 1.0 (at/below ``theta_safe`` so the
    deterministic Module-3 projection passes upstream weights through unchanged)
    and confidence 0.9. Every completion for the same prompt is byte-identical.
    """

    provider: ClassVar[str] = "offline"
    sentiment_score: ClassVar[float] = 3.0
    semantic_risk_score: ClassVar[float] = 1.0
    confidence: ClassVar[float] = 0.9

    def __init__(self, *, instruments: InstrumentsConfig | None = None) -> None:
        master = instruments if instruments is not None else load_instruments()
        self._known_roots: frozenset[str] = frozenset(master.roots())

    @property
    def known_roots(self) -> frozenset[str]:
        """Futures roots this client recognises when parsing a prompt."""
        return self._known_roots

    def roots_from_prompt(self, prompt: str) -> tuple[str, ...]:
        """Extract requested futures roots from ``prompt`` in first-seen order."""
        if not isinstance(prompt, str):
            raise TypeError(f"prompt must be a string, got {type(prompt)!r}")
        found: list[str] = []
        seen: set[str] = set()
        for match in _ROOT_TOKEN_PATTERN.finditer(prompt):
            token = match.group(1)
            if token in self._known_roots and token not in seen:
                seen.add(token)
                found.append(token)
        return tuple(found)

    def payload_for_prompt(self, prompt: str) -> str:
        """Deterministic strict-JSON completion for ``prompt``.

        Returns ``{"assessments": [...]}`` with one record per detected root, or
        ``{"assessments": []}`` when the prompt lists no known futures root.
        """
        assessments = [
            {
                "instrument_id": root,
                "sentiment_score": self.sentiment_score,
                "semantic_risk_score": self.semantic_risk_score,
                "confidence": self.confidence,
                "reasoning_summary": (
                    f"Offline deterministic assessment for {root}: "
                    "no live LLM provider or macro/news feed is configured."
                ),
            }
            for root in self.roots_from_prompt(prompt)
        ]
        return json.dumps({"assessments": assessments}, sort_keys=True, separators=(",", ":"))

    async def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        """Return :meth:`payload_for_prompt` (arguments other than prompt ignored)."""
        del system_prompt, temperature
        return self.payload_for_prompt(prompt)


def _normalise_provider(settings: Settings) -> str:
    provider = settings.llm_provider_value
    return provider.strip().lower() if isinstance(provider, str) else ""


def build_llm_client(settings: Settings) -> AsyncLLMClient:
    """Build the configured :class:`AsyncLLMClient` from process settings.

    Parameters
    ----------
    settings:
        Central :class:`~configs.settings.Settings`. ``llm_provider`` selects the
        adapter, ``llm_model`` the model identifier and ``llm_api_key`` the
        ``FUTURESRLX_LLM_API_KEY`` secret.

    Returns
    -------
    AsyncLLMClient
        An ``OfflineLLMClient`` for ``offline``, else the matching HTTP adapter
        (with ``settings.llm_base_url`` as ``base_url=`` when configured).

    Raises
    ------
    ValueError
        For an unsupported provider, a blank model identifier (only reachable via
        ``model_construct`` settings) or a missing API key on a non-offline
        provider. The message names the environment variable to set.
    """
    provider = _normalise_provider(settings)
    if provider not in _SUPPORTED_PROVIDERS:
        raise ValueError(
            f"unsupported llm_provider {provider!r}; supported providers: "
            f"{list(_SUPPORTED_PROVIDERS)} (set FUTURESRLX_LLM_PROVIDER)"
        )
    model = settings.llm_model.strip()
    if not model:
        raise ValueError("llm_model must be a non-empty model identifier (FUTURESRLX_LLM_MODEL)")
    if provider == "offline":
        return OfflineLLMClient()
    api_key = settings.llm_api_key_value
    base_url = settings.llm_base_url_value
    if provider == "vllm":
        # Local vLLM servers are normally keyless: the adapter accepts api_key=None.
        if base_url is None:
            return VLLMClient(model=model, api_key=api_key)
        return VLLMClient(model=model, api_key=api_key, base_url=base_url)
    if not api_key:
        raise ValueError(
            f"llm_provider={provider!r} requires an API key: set FUTURESRLX_LLM_API_KEY "
            "(or use llm_provider='offline' for deterministic fixture assessments, or "
            "llm_provider='vllm' for a keyless local server)"
        )
    if provider == "openai":
        if base_url is None:
            return OpenAIClient(api_key, model=model)
        return OpenAIClient(api_key, model=model, base_url=base_url)
    if provider == "anthropic":
        if base_url is None:
            return AnthropicClient(api_key, model=model)
        return AnthropicClient(api_key, model=model, base_url=base_url)
    if provider == "deepseek":
        if base_url is None:
            return DeepSeekClient(api_key, model=model)
        return DeepSeekClient(api_key, model=model, base_url=base_url)
    # Local vLLM servers are normally keyless: the adapter accepts api_key=None.
    if base_url is None:
        return VLLMClient(model=model, api_key=api_key)
    return VLLMClient(model=model, api_key=api_key, base_url=base_url)
