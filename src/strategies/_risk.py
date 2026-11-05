"""FinRL Risk-First "Module 2" reward-shaping penalty.

The penalty is the exact Lagrangian term specified by the FuturesRL
architecture:

    ``penalty = lambda * sum_i |w_i| * I(c_i >= tau) * max(0, S_i - theta_safe)``

where ``S_i`` is the semantic risk score (1 = benign, 5 = imminent macro shock),
``c_i`` the LLM confidence and ``tau`` the anti-hallucination confidence floor.
A low-confidence assessment is ignored, so a hallucinating model can never move
the reward signal, and an assessment at or below ``theta_safe`` is neutral.

Both the Gymnasium environment (training) and :class:`DRLAllocator` (the
CPPO/CPRL hook exposed for external trainers) call this single implementation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from src.core.types import InstrumentId, RiskAssessment

#: Default anti-hallucination confidence floor ``tau``.
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.70

#: Default neutral semantic-risk score ``theta_safe``.
DEFAULT_SAFE_RISK_SCORE: float = 1.5


def assessments_by_instrument(
    assessments: Sequence[RiskAssessment] | Mapping[InstrumentId, RiskAssessment] | None,
) -> dict[InstrumentId, RiskAssessment]:
    """Normalise assessments to an ``instrument -> assessment`` mapping."""
    if assessments is None:
        return {}
    if isinstance(assessments, Mapping):
        return dict(assessments)
    return {assessment.instrument_id: assessment for assessment in assessments}


def semantic_risk_penalty(
    instruments: Sequence[InstrumentId],
    weights: NDArray[np.float64],
    assessments: Sequence[RiskAssessment] | Mapping[InstrumentId, RiskAssessment] | None,
    *,
    risk_penalty_lambda: float,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    safe_risk_score: float = DEFAULT_SAFE_RISK_SCORE,
) -> float:
    """Return the Lagrangian semantic-risk penalty for one weight vector.

    ``instruments`` and ``weights`` are aligned positionally (the dense weight
    vector layout); assessments absent for an instrument contribute nothing.
    """
    if risk_penalty_lambda <= 0.0:
        return 0.0
    mapping = assessments_by_instrument(assessments)
    if not mapping:
        return 0.0
    if len(instruments) != int(weights.size):
        raise ValueError(
            f"instruments ({len(instruments)}) and weights ({int(weights.size)}) must align"
        )
    total = 0.0
    for index, instrument_id in enumerate(instruments):
        assessment = mapping.get(instrument_id)
        if assessment is None:
            continue
        if assessment.confidence < confidence_threshold:
            continue
        excess = assessment.semantic_risk_score - safe_risk_score
        if excess <= 0.0:
            continue
        total += abs(float(weights[index])) * excess
    return float(risk_penalty_lambda) * total


def semantic_risk_penalty_mapping(
    weights: Mapping[InstrumentId, float],
    assessments: Sequence[RiskAssessment] | Mapping[InstrumentId, RiskAssessment] | None,
    *,
    risk_penalty_lambda: float,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    safe_risk_score: float = DEFAULT_SAFE_RISK_SCORE,
) -> float:
    """Mapping-keyed variant of :func:`semantic_risk_penalty`."""
    instruments = tuple(weights)
    array = np.array([float(weights[instrument_id]) for instrument_id in instruments])
    return semantic_risk_penalty(
        instruments,
        array,
        assessments,
        risk_penalty_lambda=risk_penalty_lambda,
        confidence_threshold=confidence_threshold,
        safe_risk_score=safe_risk_score,
    )
