"""Unit tests for the Phase-9 LLM provider factory (``src.risk.llm_clients``).

All tests are offline and deterministic: the offline client performs no network
I/O, HTTP adapters are only constructed (never called), and the lazy-import test
runs in a fresh subprocess so it cannot be polluted by other tests' imports.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from configs.instruments import load_instruments
from configs.settings import Settings
from pydantic import SecretStr, ValidationError
from src.risk.llm_clients import OfflineLLMClient, build_llm_client
from src.risk.llm_risk_overlay import (
    AnthropicClient,
    AsyncLLMClient,
    DeepSeekClient,
    LLMResponseError,
    LLMRiskAssessmentBatch,
    LLMRiskConfig,
    LLMRiskOverlay,
    OpenAIClient,
    VLLMClient,
    build_risk_prompt,
)

from tests.fixtures.market import FIXED_TIMESTAMP, make_market_state, make_weight_vector

#: Repo root used by the lazy-import subprocess test.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Provider -> (adapter class, documented default base URL).
PROVIDER_ADAPTERS: dict[str, tuple[type[object], str]] = {
    "openai": (OpenAIClient, "https://api.openai.com/v1"),
    "anthropic": (AnthropicClient, "https://api.anthropic.com/v1"),
    "deepseek": (DeepSeekClient, "https://api.deepseek.com/v1"),
    "vllm": (VLLMClient, "http://localhost:8000/v1"),
}


def _prompt(roots: tuple[str, ...] = ("ES", "NQ", "CL")) -> str:
    state = make_market_state(roots=roots, with_covariance=False, with_returns=False)
    return build_risk_prompt(state, list(roots))


# ---------------------------------------------------------------------------
# OfflineLLMClient
# ---------------------------------------------------------------------------


def test_offline_client_parses_roots_and_returns_strict_json() -> None:
    client = OfflineLLMClient()
    prompt = _prompt()
    assert client.roots_from_prompt(prompt) == ("ES", "NQ", "CL")

    text = client.payload_for_prompt(prompt)
    payload = json.loads(text)
    assert set(payload) == {"assessments"}
    records = payload["assessments"]
    assert [record["instrument_id"] for record in records] == ["ES", "NQ", "CL"]
    for record in records:
        assert set(record) == {
            "instrument_id",
            "sentiment_score",
            "semantic_risk_score",
            "confidence",
            "reasoning_summary",
        }
        assert record["sentiment_score"] == 3.0
        assert record["semantic_risk_score"] == 1.0
        assert record["confidence"] == 0.9
        assert record["instrument_id"] in record["reasoning_summary"]
    # Deterministic: identical prompt -> byte-identical completion.
    assert client.payload_for_prompt(prompt) == text

    batch = LLMRiskAssessmentBatch.from_json_text(text)
    assert [response.instrument_id for response in batch.assessments] == ["ES", "NQ", "CL"]
    assessments = batch.to_risk_assessments(timestamp=FIXED_TIMESTAMP, source="offline")
    assert [assessment.source for assessment in assessments] == ["offline"] * 3
    assert all(assessment.timestamp == FIXED_TIMESTAMP for assessment in assessments)


def test_offline_client_ignores_macro_tokens_and_unknown_symbols() -> None:
    prompt = (
        "As-of UTC timestamp: 2024-01-02T21:00:00+00:00\n"
        "Tier-1 scheduled macro events: FOMC, CPI, NFP, EIA, WASDE, OPEC, GDP, XYZ\n"
        "- ES: price=4800, asset_class=equity_index\n"
        "- GC: price=2050, asset_class=metals\n"
    )
    client = OfflineLLMClient()
    assert client.roots_from_prompt(prompt) == ("ES", "GC")


def test_offline_client_no_roots_returns_empty_assessment_list() -> None:
    client = OfflineLLMClient()
    text = client.payload_for_prompt("macro backdrop is mixed; no instruments requested")
    assert json.loads(text) == {"assessments": []}
    # An empty batch is intentionally invalid under the strict schema.
    with pytest.raises(LLMResponseError):
        LLMRiskAssessmentBatch.from_json_text(text)


async def test_offline_complete_is_deterministic_and_ignores_sampling_arguments() -> None:
    client = OfflineLLMClient()
    prompt = _prompt()
    first = await client.complete(prompt)
    second = await client.complete(prompt, system_prompt="different system", temperature=0.9)
    assert first == second == client.payload_for_prompt(prompt)


def test_offline_client_satisfies_protocol_and_known_roots() -> None:
    client = OfflineLLMClient()
    assert isinstance(client, AsyncLLMClient)
    assert client.known_roots == frozenset(load_instruments().roots())
    custom = OfflineLLMClient(instruments=load_instruments())
    assert custom.roots_from_prompt("- ES: price=4800") == ("ES",)
    with pytest.raises(TypeError):
        custom.roots_from_prompt(None)  # type: ignore[arg-type]


async def test_offline_client_drives_overlay_refresh_and_passes_weights_through() -> None:
    settings = Settings(llm_provider="offline")
    client = build_llm_client(settings)
    overlay = LLMRiskOverlay(
        client,
        LLMRiskConfig(sample_count=2),
        clock=lambda: FIXED_TIMESTAMP,
    )
    state = make_market_state(roots=("ES", "NQ"), with_covariance=False, with_returns=False)
    filtered = await overlay.refresh(state, ["ES", "NQ"])
    assert [assessment.instrument_id for assessment in filtered] == ["ES", "NQ"]
    for assessment in filtered:
        assert assessment.semantic_risk_score == 1.0
        assert assessment.confidence == 0.9
        assert assessment.metadata["filtered"] is False
    assert overlay.is_cache_fresh()
    assert overlay.valid_until_utc is not None
    # S = 1.0 <= theta_safe -> the deterministic projection passes weights through.
    result = overlay.apply_risk(make_weight_vector({"ES": 0.4, "NQ": -0.2}), state)
    assert result["ES"] == pytest.approx(0.4)
    assert result["NQ"] == pytest.approx(-0.2)
    await overlay.stop_background_cache()


# ---------------------------------------------------------------------------
# build_llm_client provider mapping
# ---------------------------------------------------------------------------


def test_build_llm_client_offline_provider() -> None:
    client = build_llm_client(Settings(llm_provider="offline"))
    assert isinstance(client, OfflineLLMClient)
    assert isinstance(client, AsyncLLMClient)


@pytest.mark.parametrize("provider", sorted(PROVIDER_ADAPTERS))
def test_build_llm_client_constructs_http_adapters(provider: str) -> None:
    expected_type, expected_base_url = PROVIDER_ADAPTERS[provider]
    settings = Settings(
        llm_provider=provider,  # type: ignore[arg-type]
        llm_model="unit-test-model",
        llm_api_key=SecretStr("unit-test-key"),
    )
    client = build_llm_client(settings)
    assert isinstance(client, expected_type)
    assert isinstance(client, AsyncLLMClient)
    assert client.model == "unit-test-model"  # type: ignore[attr-defined]
    assert client.base_url == expected_base_url  # type: ignore[attr-defined]
    assert client.provider == provider  # type: ignore[attr-defined]


@pytest.mark.parametrize("provider", sorted(PROVIDER_ADAPTERS))
def test_build_llm_client_propagates_custom_base_url(provider: str) -> None:
    """A configured ``llm_base_url`` overrides every HTTP adapter's default."""
    custom_base_url = "https://llm-proxy.internal:8443/v1"
    settings = Settings(
        llm_provider=provider,  # type: ignore[arg-type]
        llm_model="unit-test-model",
        llm_api_key=SecretStr("unit-test-key"),
        llm_base_url=custom_base_url,
    )
    expected_type, default_base_url = PROVIDER_ADAPTERS[provider]
    client = build_llm_client(settings)
    assert isinstance(client, expected_type)
    assert client.base_url == custom_base_url  # type: ignore[attr-defined]
    assert client.base_url != default_base_url  # type: ignore[attr-defined]
    assert settings.llm_base_url_value == custom_base_url


def test_offline_provider_ignores_custom_base_url() -> None:
    """Offline mode creates no transport and is unaffected by llm_base_url."""
    settings = Settings(llm_provider="offline", llm_base_url="https://unused.example/v1")
    client = build_llm_client(settings)
    assert isinstance(client, OfflineLLMClient)
    assert settings.llm_base_url_value == "https://unused.example/v1"
    assert not hasattr(client, "base_url")


def test_empty_base_url_keeps_adapter_defaults() -> None:
    """An empty llm_base_url leaves the adapter's documented default endpoint."""
    assert Settings(llm_provider="offline").llm_base_url_value is None
    settings = Settings(
        llm_provider="vllm",
        llm_model="unit-test-model",
        llm_api_key=SecretStr("unit-test-key"),
        llm_base_url="",
    )
    client = build_llm_client(settings)
    assert isinstance(client, VLLMClient)
    assert client.base_url == "http://localhost:8000/v1"
    assert settings.llm_base_url_value is None


@pytest.mark.parametrize("provider", sorted(PROVIDER_ADAPTERS))
def test_missing_api_key_raises_actionable_error(provider: str) -> None:
    # ``model_construct`` bypasses the Settings validator so the factory's own
    # defensive check is exercised (production settings already fail closed).
    settings = Settings.model_construct(
        llm_provider=provider, llm_model="unit-test-model", llm_api_key=None
    )
    with pytest.raises(ValueError, match="FUTURESRLX_LLM_API_KEY"):
        build_llm_client(settings)


def test_settings_requires_api_key_for_non_offline_provider() -> None:
    with pytest.raises(ValidationError, match="FUTURESRLX_LLM_API_KEY"):
        Settings(llm_provider="openai")
    offline = Settings(llm_provider="offline")
    assert offline.llm_api_key_value is None


def test_unsupported_provider_raises_actionable_error() -> None:
    settings = Settings.model_construct(llm_provider="bogus", llm_model="unit-test-model")
    with pytest.raises(ValueError, match="unsupported llm_provider"):
        build_llm_client(settings)


def test_blank_model_identifier_raises() -> None:
    settings = Settings.model_construct(
        llm_provider="openai", llm_model="   ", llm_api_key=SecretStr("unit-test-key")
    )
    with pytest.raises(ValueError, match="llm_model"):
        build_llm_client(settings)


# ---------------------------------------------------------------------------
# Lazy imports: no httpx / network stack at module import or client construction
# ---------------------------------------------------------------------------


def test_import_and_client_construction_do_not_import_httpx() -> None:
    code = """
import sys
from pydantic import SecretStr
from configs.settings import Settings
from src.risk.llm_clients import OfflineLLMClient, build_llm_client

assert "httpx" not in sys.modules, "httpx imported by src.risk package import"
offline = build_llm_client(Settings(llm_provider="offline"))
assert isinstance(offline, OfflineLLMClient)
assert "httpx" not in sys.modules, "offline construction imported httpx"
for provider in ("openai", "anthropic", "deepseek", "vllm"):
    build_llm_client(
        Settings(llm_provider=provider, llm_model="unit-model", llm_api_key=SecretStr("k"))
    )
assert "httpx" not in sys.modules, "HTTP adapter construction imported httpx eagerly"
print("ok")
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"
