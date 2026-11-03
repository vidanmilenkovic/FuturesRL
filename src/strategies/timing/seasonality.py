"""Seasonality timing overlay (``T_t``) — calendar tilts for commodity cycles.

Futures markets have well-documented calendar rhythms (energy demand seasons,
precious-metal safe-haven demand, grain harvest cycles, year-end rate plumbing).
:class:`SeasonalityTiming` applies a deterministic, UTC-month tilt to the base
weights:

``w_i^timing = w_i^base * (1 + root_month_biases[root][month])``

for an **active** month, and multiplied by ``out_of_season_scale`` for a month
outside ``active_months`` (so a summer-energy book can be de-risked in winter
without flattening it). The month is taken from ``MarketState.timestamp``, which
:class:`~src.core.types.MarketState` normalises to UTC at construction, so the
overlay is timezone-proof and cannot look ahead.

Configured biases
-----------------
``root_month_biases`` maps a futures root to ``{month: bias}`` with ``bias`` a
multiplicative tilt (``0.25`` = +25% notional in that month, ``-0.5`` = -50%).
A small library of conservative, well-known commodity seasonality curves is
provided in :data:`DEFAULT_ROOT_MONTH_BIASES`; strategies reference them from
YAML with the comma-separated root list form (``root_month_biases: "CL,GC"``),
while programmatic callers and generated strategy drafts may pass the full
mapping directly (the YAML form additionally accepts a JSON object string, e.g.
``'{"CL": {"1": 0.2, "7": 0.3}}'``, because strategy stage parameters are
scalars).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType

from configs.strategy import StageParam

from src.core.interfaces import TimingOverlay
from src.core.types import InstrumentId, MarketState, WeightVector
from src.strategies._params import ParamError, ParamReader

#: All calendar months (the default active window).
ALL_MONTHS: tuple[int, ...] = tuple(range(1, 13))

#: Conservative built-in seasonal tilt curves (multiplicative bias per month).
#:
#: These are deliberately small (|bias| <= 0.35) illustrative curves, not
#: optimised parameters: energy demand peaks into US summer driving season, gold
#: tends to firm into autumn/winter, equity indices have a weak year-end and
#: September effect, and rates see quarter-end/repo pressure. Strategies usually
#: worth tuning them per backtest (or passing an explicit JSON mapping).
DEFAULT_ROOT_MONTH_BIASES: Mapping[str, Mapping[int, float]] = MappingProxyType(
    {
        "CL": MappingProxyType(
            {2: 0.10, 3: 0.15, 4: 0.20, 5: 0.25, 6: 0.20, 7: 0.10, 9: -0.05, 10: -0.10}
        ),
        "MCL": MappingProxyType(
            {2: 0.10, 3: 0.15, 4: 0.20, 5: 0.25, 6: 0.20, 7: 0.10, 9: -0.05, 10: -0.10}
        ),
        "GC": MappingProxyType({1: 0.10, 8: 0.05, 9: 0.10, 10: 0.10, 11: 0.05, 12: 0.10}),
        "MGC": MappingProxyType({1: 0.10, 8: 0.05, 9: 0.10, 10: 0.10, 11: 0.05, 12: 0.10}),
        "ES": MappingProxyType({1: 0.05, 9: -0.10, 10: 0.05, 11: 0.10, 12: 0.10}),
        "NQ": MappingProxyType({1: 0.05, 9: -0.10, 10: 0.05, 11: 0.10, 12: 0.10}),
        "MES": MappingProxyType({1: 0.05, 9: -0.10, 10: 0.05, 11: 0.10, 12: 0.10}),
        "MNQ": MappingProxyType({1: 0.05, 9: -0.10, 10: 0.05, 11: 0.10, 12: 0.10}),
        "ZN": MappingProxyType({3: -0.05, 6: -0.05, 9: -0.05, 12: -0.05}),
        "ZT": MappingProxyType({3: -0.05, 6: -0.05, 9: -0.05, 12: -0.05}),
    }
)


def _normalise_months(months: Sequence[int], *, name: str) -> tuple[int, ...]:
    cleaned: list[int] = []
    for month in months:
        if isinstance(month, bool) or not isinstance(month, int):
            raise ValueError(f"{name} must contain integer calendar months, got {month!r}")
        if not 1 <= month <= 12:
            raise ValueError(f"{name} months must be within 1..12, got {month!r}")
        if month in cleaned:
            raise ValueError(f"{name} contains duplicate month {month!r}")
        cleaned.append(month)
    if not cleaned:
        raise ValueError(f"{name} must contain at least one month")
    return tuple(sorted(cleaned))


def _normalise_biases(
    biases: Mapping[str, Mapping[int, float]] | None,
) -> Mapping[str, Mapping[int, float]]:
    frozen: dict[str, Mapping[int, float]] = {}
    for raw_root, curve in (biases or {}).items():
        if not isinstance(raw_root, str) or not raw_root.strip():
            raise ValueError(
                f"root_month_biases keys must be non-empty root symbols, got {raw_root!r}"
            )
        root = raw_root.strip().upper()
        cleaned: dict[int, float] = {}
        for raw_month, raw_bias in curve.items():
            month = int(raw_month)
            if not 1 <= month <= 12:
                raise ValueError(
                    f"root_month_biases[{root!r}] month must be within 1..12, got {month!r}"
                )
            bias = float(raw_bias)
            if not math.isfinite(bias):
                raise ValueError(
                    f"root_month_biases[{root!r}][{month!r}] must be finite, got {bias!r}"
                )
            if bias < -1.0:
                raise ValueError(
                    f"root_month_biases[{root!r}][{month!r}] must be >= -1.0 (a leg cannot be "
                    f"negatively sized by the seasonality overlay), got {bias!r}"
                )
            cleaned[month] = bias
        frozen[root] = MappingProxyType(cleaned)
    return MappingProxyType(frozen)


class SeasonalityTiming:
    """Tilt base weights by UTC-month seasonal bias and out-of-season scaling."""

    __slots__ = ("_active_months", "_biases", "_out_of_season_scale")

    def __init__(
        self,
        *,
        active_months: Sequence[int] | None = None,
        root_month_biases: Mapping[str, Mapping[int, float]] | None = None,
        out_of_season_scale: float = 1.0,
    ) -> None:
        months = ALL_MONTHS if active_months is None else tuple(active_months)
        if not math.isfinite(out_of_season_scale) or out_of_season_scale < 0.0:
            raise ValueError("out_of_season_scale must be a finite non-negative number")
        self._active_months = _normalise_months(months, name="active_months")
        self._biases = _normalise_biases(root_month_biases)
        self._out_of_season_scale = float(out_of_season_scale)

    @property
    def name(self) -> str:
        """Registry name of this overlay."""
        return "seasonality"

    @property
    def active_months(self) -> tuple[int, ...]:
        """Months in which the seasonal bias is applied."""
        return self._active_months

    @property
    def root_month_biases(self) -> Mapping[str, Mapping[int, float]]:
        """Configured per-root month bias curves (frozen)."""
        return self._biases

    def month_for(self, state: MarketState) -> int:
        """UTC calendar month of ``state`` (1..12)."""
        return state.timestamp.month

    def bias_for(self, instrument_id: InstrumentId, month: int) -> float:
        """Configured bias for ``instrument_id`` in ``month`` (``0.0`` when absent)."""
        return self._biases.get(instrument_id.strip().upper(), {}).get(month, 0.0)

    def scaling_factor(self, state: MarketState, instrument_id: InstrumentId) -> float:
        """Multiplicative tilt for one leg in the state's UTC month."""
        month = self.month_for(state)
        factor = 1.0 + self.bias_for(instrument_id, month)
        if month not in self._active_months:
            factor *= self._out_of_season_scale
        return max(0.0, factor)

    def adjust_timing(self, weights: WeightVector, state: MarketState) -> WeightVector:
        """Apply the monthly/out-of-season tilt to every selected leg."""
        adjusted: dict[InstrumentId, float] = {}
        for instrument_id, weight in weights.items():
            adjusted[instrument_id] = weight * self.scaling_factor(state, instrument_id)
        return WeightVector(adjusted)


def _parse_root_month_biases(raw: str, stage: str) -> Mapping[str, Mapping[int, float]]:
    """Parse the YAML scalar form of ``root_month_biases``.

    Accepted forms: empty (no biases), a JSON object string
    (``'{"CL": {"1": 0.2}}'``) for a full custom curve, or a comma-separated root
    list (``"CL,GC"``) selecting the built-in :data:`DEFAULT_ROOT_MONTH_BIASES`.
    """
    text = raw.strip()
    if not text:
        return {}
    if text.startswith("{"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ParamError(
                f"{stage}.root_month_biases: invalid JSON object ({exc}); expected e.g. "
                '\'{"CL": {"1": 0.2, "7": 0.3}}\''
            ) from exc
        if not isinstance(payload, dict):
            raise ParamError(
                f"{stage}.root_month_biases: JSON form must be an object of root -> {{month: bias}}"
            )
        parsed: dict[str, dict[int, float]] = {}
        for root, curve in payload.items():
            if not isinstance(root, str) or not isinstance(curve, dict):
                raise ParamError(
                    f"{stage}.root_month_biases: entry {root!r} must be root -> {{month: bias}}"
                )
            parsed[root] = {int(month): float(bias) for month, bias in curve.items()}
        return _normalise_biases(parsed)
    roots = tuple(part.strip().upper() for part in text.split(",") if part.strip())
    missing = [root for root in roots if root not in DEFAULT_ROOT_MONTH_BIASES]
    if missing:
        raise ParamError(
            f"{stage}.root_month_biases: no built-in seasonal curve for roots {missing}; "
            "provide an explicit JSON object string with month biases instead"
        )
    return _normalise_biases({root: DEFAULT_ROOT_MONTH_BIASES[root] for root in roots})


def build_seasonality_timing(params: Mapping[str, StageParam]) -> TimingOverlay:
    """Build a :class:`SeasonalityTiming` overlay from declarative stage parameters."""
    reader = ParamReader("timing.seasonality", params)
    active_months: Sequence[int] = ALL_MONTHS
    if reader.has("active_months"):
        raw_months = reader.as_sequence("active_months")
        if not raw_months:
            raise ParamError("timing.seasonality.active_months: must contain at least one month")
        active_months = tuple(int(month) for month in raw_months)
    overlay = SeasonalityTiming(
        active_months=active_months,
        root_month_biases=_parse_root_month_biases(
            reader.as_str("root_month_biases", ""), "timing.seasonality"
        ),
        out_of_season_scale=reader.as_float("out_of_season_scale", 1.0),
    )
    reader.check()
    return overlay


__all__ = [
    "ALL_MONTHS",
    "DEFAULT_ROOT_MONTH_BIASES",
    "SeasonalityTiming",
    "build_seasonality_timing",
]
