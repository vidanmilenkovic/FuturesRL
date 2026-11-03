"""Liquidity / open-interest universe selector (``S_t``).

The selector is intentionally simple and fully deterministic: it ranks the
tradable futures roots by traded dollar volume so that the allocator only ever
sees roots whose mark-to-market can actually be traded at the target size.
Ranking uses ``(-dollar_volume, -open_interest, root)`` so equal-liquidity roots
always resolve in a stable, reproducible order (a backtest that reshuffles its
universe on re-runs is unreproducible and un-auditable).

The snapshot carries the venue's session/rolling volume; the data layer owns the
multi-day averaging, so this selector never needs to look at history beyond the
``min_history_days`` sanity check on the return series.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from configs.instruments import InstrumentsConfig
from configs.strategy import StageParam

from src.core.interfaces import UniverseSelector
from src.core.types import InstrumentId, MarketState
from src.strategies._params import ParamReader
from src.strategies.selectors._roots import (
    normalise_requested_roots,
    validate_requested_roots,
)


class LiquiditySelector:
    """Rank futures roots by dollar volume and keep the most liquid ``top_n``."""

    __slots__ = (
        "_min_avg_dollar_volume",
        "_min_history_days",
        "_min_open_interest",
        "_roots",
        "_top_n",
    )

    def __init__(
        self,
        *,
        min_avg_dollar_volume: float = 0.0,
        min_open_interest: float = 0.0,
        top_n: int | None = None,
        min_history_days: int = 0,
        roots: Sequence[str] | str = (),
        instrument_master: InstrumentsConfig | None = None,
    ) -> None:
        if min_avg_dollar_volume < 0.0:
            raise ValueError("min_avg_dollar_volume must be non-negative")
        if min_open_interest < 0.0:
            raise ValueError("min_open_interest must be non-negative")
        if top_n is not None and top_n < 1:
            raise ValueError("top_n must be a positive integer or None")
        if min_history_days < 0:
            raise ValueError("min_history_days must be non-negative")
        self._roots = normalise_requested_roots(roots)
        validate_requested_roots(self._roots, instrument_master, stage="liquidity")
        self._min_avg_dollar_volume = float(min_avg_dollar_volume)
        self._min_open_interest = float(min_open_interest)
        self._top_n = top_n
        self._min_history_days = int(min_history_days)

    @property
    def name(self) -> str:
        """Registry name of this selector."""
        return "liquidity"

    @property
    def top_n(self) -> int | None:
        """Maximum universe size (``None`` keeps every passing root)."""
        return self._top_n

    @property
    def roots(self) -> tuple[InstrumentId, ...]:
        """Explicit root allow-list from ``universe.params.roots`` (``()`` = all)."""
        return self._roots

    def dollar_volume(self, state: MarketState) -> dict[InstrumentId, float]:
        """Traded dollar volume proxy ``volume * mid`` per root."""
        return {
            instrument_id: snapshot.volume * snapshot.mid
            for instrument_id, snapshot in state.instruments.items()
        }

    def _history_is_sufficient(self, state: MarketState, instrument_id: InstrumentId) -> bool:
        """Whether a *known* return history satisfies ``min_history_days``.

        An instrument without a return series is treated as sufficient: history
        completeness is a data-layer responsibility, and silently dropping every
        root when a caller builds a feature-only state would be worse than
        trusting the caller. An explicitly short series is rejected.
        """
        if self._min_history_days <= 0:
            return True
        series = state.returns.get(instrument_id)
        if series is None:
            return True
        return series.size >= self._min_history_days

    def select(self, state: MarketState) -> list[InstrumentId]:
        """Return the liquid tradable universe, most liquid first."""
        allowed = set(self._roots) if self._roots else None
        rows: list[tuple[float, float, InstrumentId]] = []
        for instrument_id, snapshot in state.instruments.items():
            if allowed is not None and instrument_id not in allowed:
                continue
            dollar_volume = snapshot.volume * snapshot.mid
            if not dollar_volume >= self._min_avg_dollar_volume:
                continue
            if not snapshot.open_interest >= self._min_open_interest:
                continue
            if not self._history_is_sufficient(state, instrument_id):
                continue
            rows.append((-dollar_volume, -snapshot.open_interest, instrument_id))
        rows.sort()
        selected = [instrument_id for _, _, instrument_id in rows]
        return selected[: self._top_n] if self._top_n is not None else selected


def build_liquidity_selector(
    params: Mapping[str, StageParam],
    instruments: InstrumentsConfig | None = None,
) -> UniverseSelector:
    """Build a :class:`LiquiditySelector` from declarative stage parameters."""
    reader = ParamReader("selector.liquidity", params)
    selector = LiquiditySelector(
        min_avg_dollar_volume=reader.as_float("min_avg_dollar_volume", 0.0),
        min_open_interest=reader.as_float("min_open_interest", 0.0),
        top_n=reader.as_optional_int("top_n"),
        min_history_days=reader.as_int("min_history_days", 0),
        # Studio-saved configs persist an explicit root filter alongside the
        # thresholds; it restricts the candidate universe before ranking.
        roots=reader.as_sequence("roots"),
        instrument_master=instruments,
    )
    reader.check()
    return selector


__all__ = ["LiquiditySelector", "build_liquidity_selector"]
