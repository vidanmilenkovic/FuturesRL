"""Risk engine package: LLM Risk-First Tier A, quant Tier B and the contract translator.

The public surface is intentionally small and stable: the FinRL-X pipeline, the
backtest engine, the IBKR execution engine and the FastAPI telemetry layer import
these names to wire the ``R_t`` stage and the discrete ``w_t -> N_i,t`` translation.

Typical usage
-------------
>>> from src.risk import ContractTranslator, QuantRiskOverlay
>>> translator = ContractTranslator()
>>> targets = translator.translate(weights, state)
"""

from __future__ import annotations

from src.risk.contract_translator import ContractTranslator
from src.risk.llm_risk_overlay import (
    DEFAULT_LLM_SYSTEM_PROMPT,
    AnthropicClient,
    AsyncLLMClient,
    AsyncLLMRiskCacheWorker,
    DeepSeekClient,
    LLMClientError,
    LLMResponseError,
    LLMRiskAssessmentBatch,
    LLMRiskCacheEntry,
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
from src.risk.quant_risk_overlay import QuantRiskConfig, QuantRiskOverlay
from src.risk.state_store import PersistedRiskState, RiskStateStore

__all__ = [
    "DEFAULT_LLM_SYSTEM_PROMPT",
    "AnthropicClient",
    "AsyncLLMClient",
    "AsyncLLMRiskCacheWorker",
    "ContractTranslator",
    "DeepSeekClient",
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
    "PersistedRiskState",
    "QuantRiskConfig",
    "QuantRiskOverlay",
    "RiskStateStore",
    "VLLMClient",
    "build_risk_prompt",
    "extract_json_payload",
    "semantic_risk_penalty",
]
