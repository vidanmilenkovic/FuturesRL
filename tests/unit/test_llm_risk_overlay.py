"""Unit tests for the FinRL Risk-First Tier A LLM overlay.

The suite is fully offline and deterministic: the LLM transport is a scripted
fake implementing :class:`AsyncLLMClient`, and the HTTP adapters are exercised
against an injected fake ``httpx`` module. No network call is ever made.
"""

from __future__ import annotations

import json
import math
import types
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from configs.settings import RiskSettings
from pydantic import ValidationError
from src.core.interfaces import RiskOverlay
from src.core.types import (
    MarketState,
    RiskAssessment,
    WeightVector,
)
from src.risk import llm_risk_overlay
from src.risk.llm_risk_overlay import (
    DEFAULT_LLM_SYSTEM_PROMPT,
    AnthropicClient,
    AsyncLLMClient,
    DeepSeekClient,
    LLMClientError,
    LLMResponseError,
    LLMRiskAssessmentBatch,
    LLMRiskConfig,
    LLMRiskOverlay,
    LLMRiskResponse,
    MacroEvent,
    MacroEventCalendar,
    OpenAIClient,
    VLLMClient,
    build_risk_prompt,
    extract_json_payload,
    semantic_risk_penalty,
)

from tests.fixtures.market import (
    FIXED_TIMESTAMP,
    make_market_state,
    make_risk_assessment,
    make_weight_vector,
)

ONE_SAMPLE = LLMRiskConfig(sample_count=1)


# ---------------------------------------------------------------------------
# Deterministic fakes
# ---------------------------------------------------------------------------


class ScriptedClient:
    """AsyncLLMClient returning scripted responses in call order."""

    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    async def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        self.calls.append(
            {"prompt": prompt, "system_prompt": system_prompt, "temperature": temperature}
        )
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        return self.responses[index]


def _record(
    root: str,
    score: float = 2.0,
    confidence: float = 0.9,
    sentiment: float = 3.0,
    reasoning: str = "deterministic fake reasoning",
) -> dict[str, object]:
    return {
        "instrument_id": root,
        "sentiment_score": sentiment,
        "semantic_risk_score": score,
        "confidence": confidence,
        "reasoning_summary": reasoning,
    }


def _json_batch(*records: dict[str, object]) -> str:
    return json.dumps({"assessments": list(records)})


def _overlay(
    client: AsyncLLMClient,
    *,
    calendar: MacroEventCalendar | None = None,
    config: LLMRiskConfig | RiskSettings | None = None,
    source: str = "llm",
) -> LLMRiskOverlay:
    return LLMRiskOverlay(
        client,
        config if config is not None else ONE_SAMPLE,
        calendar=calendar,
        source=source,
    )


def _state(timestamp: datetime = FIXED_TIMESTAMP, *, equity: float = 1_000_000.0) -> MarketState:
    return make_market_state(timestamp=timestamp, equity=equity)


class FakeHTTPError(Exception):
    """Stands in for ``httpx.HTTPError`` in adapter tests."""


class FakeResponse:
    def __init__(self, payload: object | None = None, error: Exception | None = None) -> None:
        self._payload = payload
        self._error = error

    def raise_for_status(self) -> None:
        if self._error is not None:
            raise self._error

    def json(self) -> object:
        if self._payload is None:
            raise ValueError("no JSON payload")
        return self._payload


class FakeAsyncClient:
    def __init__(self, queue: list[object], recorder: list[dict[str, object]]) -> None:
        self._queue = queue
        self._recorder = recorder

    async def __aenter__(self) -> FakeAsyncClient:
        return self

    async def __aexit__(self, *exc_info: object) -> bool:
        return False

    async def post(self, url: str, *, headers: dict[str, str], json: object) -> FakeResponse:
        self._recorder.append({"url": url, "headers": headers, "json": json})
        item = self._queue.pop(0) if len(self._queue) > 1 else self._queue[0]
        if isinstance(item, Exception):
            raise item
        assert isinstance(item, FakeResponse)
        return item


def _fake_httpx(*items: object) -> tuple[types.SimpleNamespace, list[dict[str, object]]]:
    recorder: list[dict[str, object]] = []
    queue = list(items)

    def async_client(*, timeout: float) -> FakeAsyncClient:
        return FakeAsyncClient(queue, recorder)

    module = types.SimpleNamespace(AsyncClient=async_client, HTTPError=FakeHTTPError)
    return module, recorder


def _openai_payload(content: str) -> dict[str, object]:
    return {"choices": [{"message": {"content": content}}]}


def _anthropic_payload(content: str) -> dict[str, object]:
    return {"content": [{"type": "text", "text": content}]}


# ---------------------------------------------------------------------------
# Pydantic structured-output schema
# ---------------------------------------------------------------------------


def test_llm_risk_response_validates_ranges() -> None:
    record = LLMRiskResponse(
        instrument_id="ES",
        sentiment_score=3.0,
        semantic_risk_score=4.0,
        confidence=0.9,
    )
    assert record.semantic_risk_score == 4.0
    for bad in (
        {"sentiment_score": 0.5},
        {"semantic_risk_score": 5.5},
        {"confidence": 1.5},
        {"confidence": -0.1},
    ):
        payload = {
            "instrument_id": "ES",
            "sentiment_score": 3.0,
            "semantic_risk_score": 2.0,
            "confidence": 0.8,
            **bad,
        }
        with pytest.raises(ValidationError):
            LLMRiskResponse.model_validate(payload)


def test_llm_risk_response_rejects_extra_keys_and_missing_fields() -> None:
    with pytest.raises(ValidationError):
        LLMRiskResponse.model_validate(
            {
                "instrument_id": "ES",
                "sentiment_score": 3.0,
                "semantic_risk_score": 2.0,
                "confidence": 0.8,
                "hallucinated": True,
            }
        )
    with pytest.raises(ValidationError):
        LLMRiskResponse.model_validate({"instrument_id": "ES"})


def test_batch_requires_non_empty_and_unique_ids() -> None:
    with pytest.raises(ValidationError):
        LLMRiskAssessmentBatch.model_validate({"assessments": []})
    with pytest.raises(LLMResponseError, match="duplicate"):
        LLMRiskAssessmentBatch.from_json_text(_json_batch(_record("ES"), _record("ES")))


def test_from_json_text_bare_object() -> None:
    batch = LLMRiskAssessmentBatch.from_json_text(_json_batch(_record("ES"), _record("NQ")))
    assert {response.instrument_id for response in batch.assessments} == {"ES", "NQ"}


def test_from_json_text_markdown_fence_with_prose() -> None:
    text = (
        "Here is my macro assessment of the book.\n\n"
        "```json\n" + _json_batch(_record("ES", score=4.9, confidence=0.95)) + "\n```\n"
        "Stay risk-first!"
    )
    batch = LLMRiskAssessmentBatch.from_json_text(text)
    assert len(batch.assessments) == 1
    assert batch.assessments[0].semantic_risk_score == pytest.approx(4.9)


def test_from_json_text_bare_list_and_single_record() -> None:
    list_batch = LLMRiskAssessmentBatch.from_json_text(json.dumps([_record("CL")]))
    assert list_batch.assessments[0].instrument_id == "CL"
    single_batch = LLMRiskAssessmentBatch.from_json_text(json.dumps(_record("GC")))
    assert single_batch.assessments[0].instrument_id == "GC"


def test_from_json_text_embedded_object_in_prose() -> None:
    text = "Assessment follows: " + _json_batch(_record("ES")) + " end of note."
    batch = LLMRiskAssessmentBatch.from_json_text(text)
    assert batch.assessments[0].instrument_id == "ES"


def test_from_json_text_rejects_invalid_payloads() -> None:
    with pytest.raises(LLMResponseError):
        LLMRiskAssessmentBatch.from_json_text("not json at all")
    with pytest.raises(LLMResponseError):
        LLMRiskAssessmentBatch.from_json_text("")
    with pytest.raises(LLMResponseError):
        LLMRiskAssessmentBatch.from_json_text(json.dumps({"unexpected": []}))
    with pytest.raises(LLMResponseError):
        LLMRiskAssessmentBatch.from_json_text(json.dumps({"assessments": "nope"}))
    with pytest.raises(LLMResponseError):
        LLMRiskAssessmentBatch.from_json_text(json.dumps(42))
    with pytest.raises(LLMResponseError):
        LLMRiskAssessmentBatch.from_json_text(_json_batch(_record("ES", score=9.0)))
    with pytest.raises(LLMResponseError):
        LLMRiskAssessmentBatch.from_json_text(None)  # type: ignore[arg-type]


def test_extract_json_payload_handles_fences_and_prose() -> None:
    payload = extract_json_payload('prefix ```json {"assessments": []} ``` suffix')
    assert payload == {"assessments": []}
    assert extract_json_payload("[1, 2, 3]") == [1, 2, 3]


def test_batch_to_risk_assessments() -> None:
    batch = LLMRiskAssessmentBatch.from_json_text(
        _json_batch(_record("ES", score=2.5, confidence=0.8, sentiment=4.0, reasoning="CPI risk"))
    )
    assessments = batch.to_risk_assessments(timestamp=FIXED_TIMESTAMP, source="unit-test")
    assert len(assessments) == 1
    assessment = assessments[0]
    assert assessment.instrument_id == "ES"
    assert assessment.semantic_risk_score == pytest.approx(2.5)
    assert assessment.confidence == pytest.approx(0.8)
    assert assessment.sentiment_score == pytest.approx(4.0)
    assert assessment.source == "unit-test"
    assert assessment.timestamp == FIXED_TIMESTAMP


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------


def test_system_prompt_mentions_macro_catalysts_and_json_contract() -> None:
    for catalyst in ("FOMC", "CPI", "NFP", "EIA", "WASDE", "OPEC", "geopolitical", "JSON"):
        assert catalyst in DEFAULT_LLM_SYSTEM_PROMPT
    assert "semantic_risk_score" in DEFAULT_LLM_SYSTEM_PROMPT


def test_build_risk_prompt_includes_roots_events_and_contract() -> None:
    calendar = MacroEventCalendar.from_events([("FOMC", FIXED_TIMESTAMP + timedelta(hours=3), 1)])
    prompt = build_risk_prompt(_state(), ["ES", "NQ", "CL"], calendar=calendar)
    assert "FOMC" in prompt
    for root in ("ES", "NQ", "CL"):
        assert f"- {root}:" in prompt
    assert "strict JSON" in prompt
    assert "2024-01-02T21:00:00+00:00" in prompt


def test_build_risk_prompt_handles_missing_snapshot_and_empty_calendar() -> None:
    prompt = build_risk_prompt(_state(), ["ZZ"], calendar=MacroEventCalendar())
    assert "ZZ: no point-in-time snapshot available" in prompt
    assert "none in the injected calendar" in prompt


def test_build_risk_prompt_rejects_bad_lookahead() -> None:
    with pytest.raises(ValueError):
        build_risk_prompt(_state(), ["ES"], lookahead_hours=0.0)


# ---------------------------------------------------------------------------
# Macro event calendar
# ---------------------------------------------------------------------------


def test_calendar_upcoming_events_window_is_inclusive_and_sorted() -> None:
    calendar = MacroEventCalendar.from_events(
        [
            ("CPI", FIXED_TIMESTAMP + timedelta(hours=2), 1),
            ("FOMC", FIXED_TIMESTAMP, 2),
            ("NFP", FIXED_TIMESTAMP + timedelta(days=3), 1),
        ]
    )
    assert [event.name for event in calendar.events] == ["FOMC", "CPI", "NFP"]
    events = calendar.upcoming_events(FIXED_TIMESTAMP, FIXED_TIMESTAMP + timedelta(hours=2))
    assert events == (
        ("FOMC", FIXED_TIMESTAMP, 2),
        ("CPI", FIXED_TIMESTAMP + timedelta(hours=2), 1),
    )
    assert calendar.next_event(FIXED_TIMESTAMP) == calendar.events[1]
    assert calendar.next_event(FIXED_TIMESTAMP + timedelta(days=10)) is None


def test_calendar_lockout_window_boundaries_are_inclusive() -> None:
    event_time = FIXED_TIMESTAMP
    calendar = MacroEventCalendar.from_events([("FOMC", event_time, 1)])
    before = timedelta(minutes=30)
    after = timedelta(minutes=30)
    assert calendar.lockout_active(event_time - before, minutes_before=30, minutes_after=30)
    assert not calendar.lockout_active(
        event_time - before - timedelta(seconds=1), minutes_before=30, minutes_after=30
    )
    assert calendar.lockout_active(event_time + after, minutes_before=30, minutes_after=30)
    assert not calendar.lockout_active(
        event_time + after + timedelta(seconds=1), minutes_before=30, minutes_after=30
    )
    # Tier filtering: only Tier-1 events lock out the book.
    assert not calendar.lockout_active(event_time, minutes_before=30, minutes_after=30, tier=2)
    with pytest.raises(ValueError):
        calendar.lockout_active(event_time, minutes_before=-1, minutes_after=30)


def test_calendar_from_json_mapping_text_and_file(tmp_path: Path) -> None:
    payload = {
        "events": [
            {"name": "CPI", "timestamp": "2024-01-02T21:00:00Z", "tier": 1},
            {"name": "OPEC", "timestamp": "2024-01-03T12:00:00+01:00", "tier": 2},
        ]
    }
    from_mapping = MacroEventCalendar.from_json(payload)
    assert len(from_mapping.events) == 2
    assert from_mapping.events[0].timestamp == FIXED_TIMESTAMP
    from_text = MacroEventCalendar.from_json(json.dumps(payload))
    assert from_text.events == from_mapping.events
    json_path = tmp_path / "calendar.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    from_path = MacroEventCalendar.from_json(str(json_path))
    assert from_path.events == from_mapping.events
    from_path_object = MacroEventCalendar.from_json(json_path)
    assert from_path_object.events == from_mapping.events
    yaml_path = tmp_path / "calendar.yaml"
    yaml_path.write_text(
        "events:\n  - name: NFP\n    timestamp: 2024-01-05T13:30:00Z\n    tier: 1\n",
        encoding="utf-8",
    )
    assert MacroEventCalendar.from_json(str(yaml_path)).events[0].name == "NFP"


def test_calendar_from_json_single_event_and_tuple_entries() -> None:
    single = MacroEventCalendar.from_json({"name": "FOMC", "timestamp": "2024-01-02T21:00:00Z"})
    assert single.events[0].name == "FOMC"
    tuple_calendar = MacroEventCalendar.from_json({"events": [["CPI", "2024-01-02T21:00:00Z", 1]]})
    assert tuple_calendar.events[0].tier == 1


def test_calendar_invalid_payloads() -> None:
    with pytest.raises(ValueError):
        MacroEventCalendar.from_json(
            {"events": [{"name": "CPI", "timestamp": "2024-01-02T21:00:00"}]}
        )
    with pytest.raises(ValueError):
        MacroEventCalendar.from_json({"events": ["bad-entry"]})
    with pytest.raises(ValueError):
        MacroEventCalendar.from_json({"events": [["CPI"]]})
    with pytest.raises(ValueError):
        MacroEventCalendar.from_json(
            {"events": [{"name": "", "timestamp": "2024-01-02T00:00:00Z"}]}
        )
    with pytest.raises(ValueError):
        MacroEventCalendar.from_json(
            {"events": [{"name": "CPI", "timestamp": "2024-01-02T00:00:00Z", "tier": True}]}
        )
    with pytest.raises(ValueError):
        MacroEventCalendar.from_json({"events": "not-a-list"})
    with pytest.raises(ValueError):
        MacroEventCalendar.from_json("{not valid json")
    with pytest.raises(TypeError):
        MacroEventCalendar.from_json(42)  # type: ignore[arg-type]
    with pytest.raises(FileNotFoundError):
        MacroEventCalendar.from_json("/nonexistent/calendar.json")
    with pytest.raises(ValueError):
        MacroEventCalendar.from_json({"events": [{"name": "CPI", "timestamp": 12345}]})


def test_macro_event_validation() -> None:
    with pytest.raises(ValueError):
        MacroEvent(name=" ", timestamp=FIXED_TIMESTAMP)
    with pytest.raises(ValueError):
        MacroEvent(name="CPI", timestamp=datetime(2024, 1, 2, 21, 0))
    with pytest.raises(TypeError):
        MacroEvent(name="CPI", timestamp=FIXED_TIMESTAMP, tier=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        MacroEvent(name="CPI", timestamp=FIXED_TIMESTAMP, tier=0)


def test_batch_before_validator_tolerates_non_mapping_input() -> None:
    with pytest.raises(ValidationError):
        LLMRiskAssessmentBatch.model_validate([_record("ES")])


def test_extract_json_payload_skips_empty_fence_and_broken_objects() -> None:
    payload = extract_json_payload("```json\n\n```\n" + _json_batch(_record("ES")))
    assert isinstance(payload, dict)
    with pytest.raises(LLMResponseError):
        extract_json_payload("prefix {broken json} suffix")
    with pytest.raises(llm_risk_overlay.LLMResponseError):
        extract_json_payload("only prose here")


def test_response_parse_rejects_missing_message() -> None:
    client = OpenAIClient("key", httpx_loader=lambda: None)
    with pytest.raises(ValueError):
        client._parse_response({"choices": [{"no_message": True}]})


async def test_client_retries_multiple_times_before_success() -> None:
    content = _json_batch(_record("ES"))
    module, recorder = _fake_httpx(
        FakeHTTPError("first"),
        FakeHTTPError("second"),
        FakeResponse(_openai_payload(content)),
    )
    client = OpenAIClient(
        "key", max_retries=2, backoff_base_seconds=0.0, httpx_loader=lambda: module
    )
    assert await client.complete("prompt") == content
    assert len(recorder) == 3


def test_calendar_from_events_accepts_macro_event_objects() -> None:
    event = MacroEvent(name="FOMC", timestamp=FIXED_TIMESTAMP)
    calendar = MacroEventCalendar.from_events([event])
    assert calendar.events == (event,)


def test_calendar_rejects_invalid_iso_timestamp() -> None:
    with pytest.raises(ValueError):
        MacroEventCalendar.from_json({"events": [{"name": "CPI", "timestamp": "not-a-date"}]})


def test_calendar_empty_mapping_payload() -> None:
    calendar = MacroEventCalendar.from_json({"unrelated": 1})
    assert calendar.events == ()
    assert MacroEventCalendar.from_json("[]").events == ()
    with pytest.raises(FileNotFoundError):
        MacroEventCalendar.from_json("42")


def test_prompt_renders_unknown_metrics_and_missing_features() -> None:
    state = make_market_state(
        metadata={"non_scalar": (1, 2), "bad_metric": math.nan, "good_metric": 0.5},
        features={"ES": {}},
    )
    prompt = build_risk_prompt(state, ["ES"])
    assert "bad_metric=n/a" in prompt
    assert "good_metric=0.5" in prompt
    assert "non_scalar" not in prompt


def test_prompt_without_expiration() -> None:
    from tests.unit.test_contract_translator import make_state as make_compact_state

    state = make_compact_state(roots=("ES",))
    snapshot = state.instruments["ES"]
    from src.core.types import InstrumentSnapshot

    no_expiry = InstrumentSnapshot(
        instrument_id=snapshot.instrument_id,
        root_symbol=snapshot.root_symbol,
        price=snapshot.price,
        asset_class=snapshot.asset_class,
        expiration=None,
        features={},
    )
    state = MarketState(
        timestamp=FIXED_TIMESTAMP,
        instruments={"ES": no_expiry},
        equity=1_000_000.0,
    )
    prompt = build_risk_prompt(state, ["ES"])
    assert "expiration=" not in prompt
    assert "vol_20d=n/a" in prompt


def test_lockout_cap_leverage_keeps_small_book_unchanged() -> None:
    calendar = MacroEventCalendar.from_events([("FOMC", FIXED_TIMESTAMP, 1)])
    config = LLMRiskConfig(sample_count=1, macro_lockout_action="cap_leverage")
    overlay = _overlay(ScriptedClient(_json_batch(_record("ES"))), calendar=calendar, config=config)
    result = overlay.apply_risk(make_weight_vector({"ES": 0.1}), _state(FIXED_TIMESTAMP))
    assert result["ES"] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Module 1: confidence / variance anti-hallucination filter
# ---------------------------------------------------------------------------


async def test_low_confidence_hallucination_is_neutralised() -> None:
    client = ScriptedClient(_json_batch(_record("ES", score=5.0, confidence=0.20)))
    overlay = _overlay(client)
    filtered = await overlay.refresh(_state(), ["ES"])
    assessment = filtered[0]
    assert assessment.semantic_risk_score == pytest.approx(1.5)
    assert assessment.metadata["filtered"] is True
    # The hallucinated critical score must not move exposure.
    result = overlay.apply_risk(make_weight_vector({"ES": 0.5}), _state())
    assert result["ES"] == pytest.approx(0.5)


async def test_high_variance_across_samples_is_neutralised() -> None:
    client = ScriptedClient(
        _json_batch(_record("ES", score=1.0, confidence=0.95)),
        _json_batch(_record("ES", score=5.0, confidence=0.95)),
        _json_batch(_record("ES", score=1.0, confidence=0.95)),
    )
    overlay = _overlay(client, config=LLMRiskConfig(sample_count=3, variance_max=0.25))
    assessment = (await overlay.refresh(_state(), ["ES"]))[0]
    assert assessment.metadata["filtered"] is True
    assert assessment.metadata["semantic_risk_variance"] == pytest.approx(3.5555556, rel=1e-6)
    assert assessment.semantic_risk_score == pytest.approx(1.5)


async def test_low_variance_samples_are_mean_aggregated() -> None:
    client = ScriptedClient(
        _json_batch(_record("ES", score=2.0, confidence=0.9, sentiment=2.0)),
        _json_batch(_record("ES", score=2.0, confidence=0.9, sentiment=4.0)),
        _json_batch(_record("ES", score=2.0, confidence=0.9, sentiment=3.0)),
    )
    overlay = _overlay(client, config=LLMRiskConfig(sample_count=3))
    assessment = (await overlay.refresh(_state(), ["ES"]))[0]
    assert assessment.metadata["filtered"] is False
    assert assessment.semantic_risk_score == pytest.approx(2.0)
    assert assessment.sentiment_score == pytest.approx(3.0)
    assert assessment.metadata["sample_count"] == 3.0
    assert len(client.calls) == 3
    assert client.calls[0]["system_prompt"] == DEFAULT_LLM_SYSTEM_PROMPT
    assert client.calls[0]["temperature"] == pytest.approx(0.7)


async def test_one_invalid_sample_is_skipped() -> None:
    client = ScriptedClient("not json", _json_batch(_record("ES", score=2.5, confidence=0.9)))
    overlay = _overlay(client, config=LLMRiskConfig(sample_count=2))
    assessment = (await overlay.refresh(_state(), ["ES"]))[0]
    assert assessment.metadata["filtered"] is False
    assert assessment.metadata["sample_count"] == 1.0
    assert assessment.semantic_risk_score == pytest.approx(2.5)


async def test_all_samples_invalid_yields_neutral_fallback() -> None:
    client = ScriptedClient("garbage", "also garbage")
    overlay = _overlay(client, config=LLMRiskConfig(sample_count=2))
    assessments = await overlay.refresh(_state(), ["ES", "NQ"])
    assert len(assessments) == 2
    for assessment in assessments:
        assert assessment.semantic_risk_score == pytest.approx(1.5)
        assert assessment.confidence == 0.0
        assert assessment.metadata["filtered"] is True
        assert "no valid LLM response" in assessment.reasoning_summary


async def test_unknown_llm_root_is_ignored_and_duplicates_deduped() -> None:
    client = ScriptedClient(
        _json_batch(_record("ES", score=2.0), _record("ZZ", score=5.0, confidence=0.99))
    )
    overlay = _overlay(client)
    assessments = await overlay.refresh(_state(), ["ES", "ES"])
    assert len(assessments) == 1
    assert assessments[0].instrument_id == "ES"
    assert overlay.latest_assessments["ES"].semantic_risk_score == pytest.approx(2.0)


async def test_refresh_empty_candidates_and_k_validation() -> None:
    client = ScriptedClient(_json_batch(_record("ES")))
    overlay = _overlay(client)
    assert await overlay.refresh(_state(), []) == ()
    assert client.calls == []
    with pytest.raises(ValueError, match="k"):
        await overlay.sample_assessments(_state(), ["ES"], k=0)
    assert await overlay.sample_assessments(_state(), [], k=3) == ()


async def test_refresh_records_timestamp_and_cache() -> None:
    client = ScriptedClient(_json_batch(_record("ES", score=2.0)))
    overlay = _overlay(client, source="unit-test")
    await overlay.refresh(_state(), ["ES"])
    assert overlay.last_refresh == FIXED_TIMESTAMP
    assert overlay.latest_assessments["ES"].source == "unit-test"
    overlay.set_cached_assessments([make_risk_assessment("ES", semantic_risk_score=4.9)])
    assert overlay.latest_assessments["ES"].semantic_risk_score == pytest.approx(4.9)


# ---------------------------------------------------------------------------
# Module 3: deterministic circuit breaker and projection
# ---------------------------------------------------------------------------


def test_critical_alert_zeroes_exposure_regardless_of_upstream_weight() -> None:
    client = ScriptedClient(_json_batch(_record("ES")))
    overlay = _overlay(client)
    overlay.set_cached_assessments(
        [make_risk_assessment("ES", semantic_risk_score=4.5, confidence=0.95)]
    )
    result = overlay.apply_risk(make_weight_vector({"ES": 0.9}), _state())
    assert result["ES"] == 0.0


def test_linear_scaling_is_exact() -> None:
    client = ScriptedClient(_json_batch(_record("ES")))
    overlay = _overlay(client)
    overlay.set_cached_assessments(
        [make_risk_assessment("ES", semantic_risk_score=3.0, confidence=0.9)]
    )
    # (kappa - S) / (kappa - theta_safe) = (4.5 - 3.0) / (4.5 - 1.5) = 0.5.
    result = overlay.apply_risk(make_weight_vector({"ES": 0.4}), _state())
    assert result["ES"] == pytest.approx(0.2)


def test_scaling_boundaries_theta_and_kappa() -> None:
    client = ScriptedClient(_json_batch(_record("ES")))
    overlay = _overlay(client)
    overlay.set_cached_assessments(
        [make_risk_assessment("ES", semantic_risk_score=1.5, confidence=0.99)]
    )
    assert overlay.apply_risk(make_weight_vector({"ES": 0.3}), _state())["ES"] == pytest.approx(0.3)
    overlay.set_cached_assessments(
        [make_risk_assessment("ES", semantic_risk_score=4.5, confidence=0.99)]
    )
    assert overlay.apply_risk(make_weight_vector({"ES": 0.3}), _state())["ES"] == 0.0


def test_low_confidence_assessment_does_not_move_weights() -> None:
    client = ScriptedClient(_json_batch(_record("ES")))
    overlay = _overlay(client)
    overlay.set_cached_assessments(
        [make_risk_assessment("ES", semantic_risk_score=5.0, confidence=0.5)]
    )
    result = overlay.apply_risk(make_weight_vector({"ES": 0.3}), _state())
    assert result["ES"] == pytest.approx(0.3)


def test_no_assessment_keeps_weight_and_unknown_root_raises() -> None:
    client = ScriptedClient(_json_batch(_record("ES")))
    overlay = _overlay(client)
    result = overlay.apply_risk(make_weight_vector({"NQ": -0.2}), _state())
    assert result["NQ"] == pytest.approx(-0.2)
    with pytest.raises(ValueError, match="XX"):
        overlay.apply_risk(make_weight_vector({"XX": 0.1}), _state())


def test_apply_risk_flat_and_canonical_roots() -> None:
    client = ScriptedClient(_json_batch(_record("ES")))
    overlay = _overlay(client)
    assert overlay.apply_risk(WeightVector.empty(), _state()) == WeightVector.empty()
    flat = overlay.apply_risk(make_weight_vector({"ES": 0.0}), _state())
    assert flat.is_flat()
    overlay.set_cached_assessments(
        [make_risk_assessment("ES", semantic_risk_score=3.0, confidence=0.9)]
    )
    canonical = overlay.apply_risk(make_weight_vector({"es": 0.4}), _state())
    assert set(canonical.as_dict()) == {"ES"}
    assert canonical["ES"] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Module 3: macro event lockout windows
# ---------------------------------------------------------------------------


def _lockout_overlay(
    action: str = "flatten", *, actual_timestamp: datetime = FIXED_TIMESTAMP
) -> tuple[LLMRiskOverlay, MacroEventCalendar]:
    event_time = actual_timestamp
    calendar = MacroEventCalendar.from_events([("FOMC", event_time, 1)])
    config = LLMRiskConfig(
        sample_count=1,
        macro_lockout_action=action,  # type: ignore[arg-type]
        macro_lockout_leverage_cap=0.25,
        macro_lockout_minutes_before=30,
        macro_lockout_minutes_after=30,
    )
    client = ScriptedClient(_json_batch(_record("ES")))
    return _overlay(client, calendar=calendar, config=config), calendar


@pytest.mark.parametrize("minutes", [30, 0])
def test_macro_lockout_flattens_inclusive_window(minutes: int) -> None:
    state = _state(FIXED_TIMESTAMP)
    overlay, _ = _lockout_overlay()
    offset = timedelta(minutes=minutes)
    inside = _state(FIXED_TIMESTAMP - offset)
    result = overlay.apply_risk(make_weight_vector({"ES": 0.5}), inside)
    assert result.is_flat()
    assert overlay.apply_risk(make_weight_vector({"ES": 0.5}), state).is_flat()


def test_macro_lockout_outside_window_is_inactive() -> None:
    overlay, _ = _lockout_overlay()
    before_outside = _state(FIXED_TIMESTAMP - timedelta(minutes=30, seconds=1))
    after_outside = _state(FIXED_TIMESTAMP + timedelta(minutes=30, seconds=1))
    assert overlay.apply_risk(make_weight_vector({"ES": 0.5}), before_outside)["ES"] == (
        pytest.approx(0.5)
    )
    assert overlay.apply_risk(make_weight_vector({"ES": 0.5}), after_outside)["ES"] == (
        pytest.approx(0.5)
    )


def test_macro_lockout_cap_leverage_action() -> None:
    overlay, _ = _lockout_overlay("cap_leverage")
    result = overlay.apply_risk(
        make_weight_vector({"ES": 0.75, "NQ": 0.75}), _state(FIXED_TIMESTAMP)
    )
    assert result.gross == pytest.approx(0.25)
    assert result["ES"] == pytest.approx(0.125)
    assert result["NQ"] == pytest.approx(0.125)


def test_macro_lockout_ignores_tier_two_events() -> None:
    calendar = MacroEventCalendar.from_events([("OPEC", FIXED_TIMESTAMP, 2)])
    config = LLMRiskConfig(sample_count=1)
    overlay = _overlay(ScriptedClient(_json_batch(_record("ES"))), calendar=calendar, config=config)
    result = overlay.apply_risk(make_weight_vector({"ES": 0.5}), _state(FIXED_TIMESTAMP))
    assert result["ES"] == pytest.approx(0.5)


def test_no_calendar_means_no_lockout() -> None:
    overlay = _overlay(ScriptedClient(_json_batch(_record("ES"))))
    assert overlay.calendar is None
    assert overlay.apply_risk(make_weight_vector({"ES": 0.5}), _state())["ES"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Module 2: reward-shaping penalty
# ---------------------------------------------------------------------------


def test_semantic_risk_penalty_formula_exact() -> None:
    weights = make_weight_vector({"ES": 0.3, "NQ": -0.5, "CL": 0.2})
    assessments = [
        make_risk_assessment("ES", semantic_risk_score=3.0, confidence=0.9),
        make_risk_assessment("NQ", semantic_risk_score=5.0, confidence=0.9),
        make_risk_assessment("CL", semantic_risk_score=2.0, confidence=0.5),
    ]
    penalty = semantic_risk_penalty(weights, assessments, 0.5, 1.5, 0.7)
    # 0.5 * (0.3 * 1.5 + 0.5 * 3.5 + 0.2 * 0.0) = 0.5 * 2.2
    assert penalty == pytest.approx(1.1)


def test_semantic_risk_penalty_ignores_below_threshold_scores() -> None:
    weights = make_weight_vector({"ES": 0.5})
    assessments = [make_risk_assessment("ES", semantic_risk_score=1.5, confidence=1.0)]
    assert semantic_risk_penalty(weights, assessments) == pytest.approx(0.0)
    # Confidence exactly at tau is still included (I(c >= tau)).
    assessments = [make_risk_assessment("ES", semantic_risk_score=2.5, confidence=0.7)]
    assert semantic_risk_penalty(weights, assessments, 1.0, 1.5, 0.7) == pytest.approx(0.5)


def test_semantic_risk_penalty_accepts_mapping_and_ignores_missing_weights() -> None:
    weights = make_weight_vector({"ES": 0.5})
    assessments = {
        "ES": make_risk_assessment("ES", semantic_risk_score=2.5, confidence=0.9),
        "CL": make_risk_assessment("CL", semantic_risk_score=5.0, confidence=0.9),
    }
    assert semantic_risk_penalty(weights, assessments, 1.0, 1.5, 0.7) == pytest.approx(0.5)


def test_semantic_risk_penalty_validation() -> None:
    weights = make_weight_vector({"ES": 0.5})
    assessments = [make_risk_assessment("ES", semantic_risk_score=3.0)]
    with pytest.raises(ValueError):
        semantic_risk_penalty(weights, assessments, -0.1)
    with pytest.raises(ValueError):
        semantic_risk_penalty(weights, assessments, 0.5, 0.5)
    with pytest.raises(ValueError):
        semantic_risk_penalty(weights, assessments, 0.5, 1.5, 1.5)


def test_overlay_penalty_uses_configured_parameters() -> None:
    config = LLMRiskConfig(sample_count=1, penalty_lambda=2.0, safe_score=1.5)
    overlay = _overlay(ScriptedClient(_json_batch(_record("ES"))), config=config)
    overlay.set_cached_assessments(
        [make_risk_assessment("ES", semantic_risk_score=2.5, confidence=0.9)]
    )
    # 2.0 * |0.5| * (2.5 - 1.5) = 1.0
    assert overlay.penalty(make_weight_vector({"ES": 0.5})) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# LLM configuration / overlay plumbing
# ---------------------------------------------------------------------------


def test_overlay_satisfies_risk_overlay_protocol_and_names() -> None:
    overlay = _overlay(ScriptedClient(_json_batch(_record("ES"))))
    assert isinstance(overlay, RiskOverlay)
    assert overlay.name == "llm_risk"
    assert isinstance(ScriptedClient(_json_batch(_record("ES"))), AsyncLLMClient)


def test_config_from_risk_settings() -> None:
    settings = RiskSettings()
    config = LLMRiskConfig.from_risk_settings(settings)
    assert config.confidence_threshold == settings.llm_confidence_threshold
    assert config.variance_max == settings.llm_semantic_variance_max
    assert config.sample_count == settings.llm_sample_count
    assert config.safe_score == settings.llm_safe_score
    assert config.critical_score == settings.llm_critical_score
    assert config.penalty_lambda == settings.llm_risk_penalty_lambda
    assert config.macro_lockout_minutes_before == settings.macro_lockout_minutes_before
    assert config.macro_lockout_minutes_after == settings.macro_lockout_minutes_after
    overlay = _overlay(ScriptedClient(_json_batch(_record("ES"))), config=settings)
    assert overlay.config.confidence_threshold == settings.llm_confidence_threshold


def test_config_validation_and_coercion_errors() -> None:
    with pytest.raises(ValidationError):
        LLMRiskConfig(critical_score=1.4, safe_score=1.5)
    with pytest.raises(ValidationError):
        LLMRiskConfig(sample_count=0)
    with pytest.raises(TypeError, match="LLMRiskConfig"):
        LLMRiskOverlay(ScriptedClient(_json_batch(_record("ES"))), object())  # type: ignore[arg-type]
    defaults = LLMRiskOverlay(ScriptedClient(_json_batch(_record("ES"))))
    assert defaults.config.confidence_threshold == 0.70
    assert defaults.config.macro_lockout_action == "flatten"


# ---------------------------------------------------------------------------
# HTTP adapters (fake httpx module; no network)
# ---------------------------------------------------------------------------


async def test_openai_client_builds_request_and_parses_response() -> None:
    content = _json_batch(_record("ES", score=2.0))
    module, recorder = _fake_httpx(FakeResponse(_openai_payload(content)))
    client = OpenAIClient("secret-key", httpx_loader=lambda: module)
    result = await client.complete("assess ES", system_prompt="system", temperature=0.5)
    assert result == content
    assert recorder[0]["url"] == "https://api.openai.com/v1/chat/completions"
    headers = recorder[0]["headers"]
    assert headers["Authorization"] == "Bearer secret-key"
    payload = recorder[0]["json"]
    assert payload["model"] == "gpt-4o-mini"
    assert payload["temperature"] == 0.5
    assert payload["messages"][0] == {"role": "system", "content": "system"}
    assert payload["messages"][1] == {"role": "user", "content": "assess ES"}
    assert client.provider == "openai"
    assert client.model == "gpt-4o-mini"
    assert client.base_url == "https://api.openai.com/v1"


async def test_deepseek_and_vllm_clients() -> None:
    content = _json_batch(_record("ES"))
    module, recorder = _fake_httpx(FakeResponse(_openai_payload(content)))
    deepseek = DeepSeekClient("ds-key", httpx_loader=lambda: module)
    await deepseek.complete("prompt")
    assert recorder[0]["url"] == "https://api.deepseek.com/v1/chat/completions"
    assert recorder[0]["headers"]["Authorization"] == "Bearer ds-key"
    assert recorder[0]["json"]["model"] == "deepseek-chat"

    module, recorder = _fake_httpx(FakeResponse(_openai_payload(content)))
    vllm = VLLMClient(httpx_loader=lambda: module)
    await vllm.complete("prompt")
    assert recorder[0]["url"] == "http://localhost:8000/v1/chat/completions"
    assert "Authorization" not in recorder[0]["headers"]
    assert recorder[0]["json"]["model"] == "local-model"


async def test_anthropic_client_builds_request_and_parses_response() -> None:
    content = _json_batch(_record("ES"))
    module, recorder = _fake_httpx(FakeResponse(_anthropic_payload(content)))
    client = AnthropicClient("anthropic-key", httpx_loader=lambda: module)
    result = await client.complete("assess roots", system_prompt="system", temperature=0.3)
    assert result == content
    assert recorder[0]["url"] == "https://api.anthropic.com/v1/messages"
    headers = recorder[0]["headers"]
    assert headers["x-api-key"] == "anthropic-key"
    assert headers["anthropic-version"] == llm_risk_overlay.ANTHROPIC_VERSION
    payload = recorder[0]["json"]
    assert payload["max_tokens"] == 1024
    assert payload["system"] == "system"
    assert payload["messages"] == [{"role": "user", "content": "assess roots"}]
    assert client.provider == "anthropic"


async def test_client_retries_transport_errors_then_succeeds() -> None:
    content = _json_batch(_record("ES"))
    module, recorder = _fake_httpx(
        FakeHTTPError("connection reset"), FakeResponse(_openai_payload(content))
    )
    client = OpenAIClient(
        "key", max_retries=2, backoff_base_seconds=0.0, httpx_loader=lambda: module
    )
    assert await client.complete("prompt") == content
    assert len(recorder) == 2


async def test_client_retries_malformed_payloads_then_succeeds() -> None:
    content = _json_batch(_record("ES"))
    module, recorder = _fake_httpx(
        FakeResponse({"unexpected": True}), FakeResponse(_openai_payload(content))
    )
    client = OpenAIClient(
        "key", max_retries=1, backoff_base_seconds=0.0, httpx_loader=lambda: module
    )
    assert await client.complete("prompt") == content
    assert len(recorder) == 2


async def test_client_raises_after_exhausted_retries() -> None:
    module, recorder = _fake_httpx(FakeHTTPError("down"))
    client = OpenAIClient(
        "key", max_retries=2, backoff_base_seconds=0.0, httpx_loader=lambda: module
    )
    with pytest.raises(LLMClientError, match="after 3 attempts"):
        await client.complete("prompt")
    assert len(recorder) == 3


async def test_anthropic_malformed_response_raises_after_retries() -> None:
    module, recorder = _fake_httpx(FakeResponse({"content": []}))
    client = AnthropicClient(
        "key", max_retries=1, backoff_base_seconds=0.0, httpx_loader=lambda: module
    )
    with pytest.raises(LLMClientError):
        await client.complete("prompt")
    assert len(recorder) == 2


async def test_client_rejects_empty_prompt() -> None:
    module, _ = _fake_httpx(FakeResponse(_openai_payload("{}")))
    client = OpenAIClient("key", httpx_loader=lambda: module)
    with pytest.raises(ValueError, match="prompt"):
        await client.complete("   ")


def test_client_constructor_validation() -> None:
    with pytest.raises(ValueError):
        OpenAIClient("key", model=" ")
    with pytest.raises(ValueError):
        OpenAIClient("key", base_url=" ")
    with pytest.raises(ValueError):
        OpenAIClient("key", timeout_seconds=0.0)
    with pytest.raises(ValueError):
        OpenAIClient("key", max_retries=-1)
    with pytest.raises(ValueError):
        OpenAIClient("key", backoff_base_seconds=-0.5)
    with pytest.raises(ValueError):
        AnthropicClient("key", max_tokens=0)


def test_anthropic_headers_without_api_key() -> None:
    client = AnthropicClient("", httpx_loader=lambda: None)
    assert client._headers()["x-api-key"] == ""


def test_lazy_httpx_import_is_real() -> None:
    module = llm_risk_overlay._load_httpx()
    assert hasattr(module, "AsyncClient")
    assert issubclass(module.HTTPError, Exception)


def test_response_parse_rejects_non_mapping_and_empty_content() -> None:
    client = OpenAIClient("key", httpx_loader=lambda: None)
    with pytest.raises(ValueError):
        client._parse_response([1, 2])
    with pytest.raises(ValueError):
        client._parse_response({})
    with pytest.raises(ValueError):
        client._parse_response({"choices": []})
    with pytest.raises(ValueError):
        client._parse_response({"choices": ["bad"]})
    with pytest.raises(ValueError):
        client._parse_response({"choices": [{"message": {}}]})
    with pytest.raises(ValueError):
        client._parse_response({"choices": [{"message": {"content": "  "}}]})
    anthropic = AnthropicClient("key", httpx_loader=lambda: None)
    with pytest.raises(ValueError):
        anthropic._parse_response({"content": [{"text": ""}]})
    with pytest.raises(ValueError):
        anthropic._parse_response({"content": ["bad"]})
    with pytest.raises(ValueError):
        anthropic._parse_response({"content": []})
    with pytest.raises(ValueError):
        anthropic._parse_response([1])


def test_macro_event_from_sequence_with_default_tier() -> None:
    event = llm_risk_overlay._parse_macro_event(["CPI", "2024-01-02T21:00:00Z"])
    assert event.tier == 1
    assert math.isfinite(event.timestamp.timestamp())


def test_openai_payload_without_system_prompt() -> None:
    module, _recorder = _fake_httpx(FakeResponse(_openai_payload(_json_batch(_record("ES")))))
    client = OpenAIClient("key", httpx_loader=lambda: module)
    payload = client._payload("prompt", None, 0.0)
    assert payload["messages"] == [{"role": "user", "content": "prompt"}]
    anthropic = AnthropicClient("key", httpx_loader=lambda: module)
    anthropic_payload = anthropic._payload("prompt", None, 0.0)
    assert "system" not in anthropic_payload


def test_assessment_reasoning_and_sentiment_from_samples() -> None:
    assessment = make_risk_assessment("ES", semantic_risk_score=2.0, confidence=0.9)
    assert isinstance(assessment, RiskAssessment)
    assert assessment.semantic_risk_score == pytest.approx(2.0)
