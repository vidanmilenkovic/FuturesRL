"""Explicit universe allow-list (``universe.params.roots``) shared by selectors.

Strategy Studio persists an explicit root filter (``roots: ES,NQ,...``) next to
the selector's numeric thresholds. The selectors treat it as an *optional
allow-list*: the candidate universe is restricted to the requested roots before
the existing thresholds/ranking/``top_n`` logic runs, and can never be widened
by it.

Two distinct failure modes are handled deliberately:

* roots that are not in the instrument master (typos, delisted contracts) raise
  an actionable :class:`ValueError` naming them as soon as the selector is built
  with a master (the declarative builder always passes one);
* roots that are simply absent from a particular point-in-time ``MarketState``
  (warm-up, partial data, a fixture state with a subset of roots) are intersected
  out silently — the state remains the hard upper bound, and a backtest must not
  crash because an early window has no history for one requested root yet.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from configs.instruments import InstrumentsConfig

from src.core.types import InstrumentId


def normalise_requested_roots(
    roots: Sequence[str] | str | None,
) -> tuple[InstrumentId, ...]:
    """Normalise a comma-separated string / sequence into a canonical root tuple.

    Symbols are upper-cased, stripped and de-duplicated while preserving the
    configured order (the selector itself applies deterministic ranking later).
    ``None``/empty means "no explicit restriction" and normalises to ``()``.
    """
    if roots is None:
        return ()
    if isinstance(roots, str):
        parts: Iterable[str] = roots.split(",")
    else:
        parts = (str(item) for item in roots)
    normalised: list[InstrumentId] = []
    for part in parts:
        token = part.strip().upper()
        if token and token not in normalised:
            normalised.append(token)
    return tuple(normalised)


def validate_requested_roots(
    roots: tuple[InstrumentId, ...],
    instrument_master: InstrumentsConfig | None,
    *,
    stage: str,
) -> None:
    """Reject explicit roots that the instrument master does not know."""
    if instrument_master is None or not roots:
        return
    unknown = [root for root in roots if not instrument_master.has(root)]
    if unknown:
        raise ValueError(
            f"{stage} selector: explicit roots {unknown!r} are not in the instrument master; "
            "use canonical root symbols (see /api/instruments)"
        )


def restrict_roots(
    roots: Iterable[InstrumentId],
    requested: tuple[InstrumentId, ...],
) -> list[InstrumentId]:
    """Intersect ``roots`` (preserving order) with the requested allow-list."""
    if not requested:
        return list(roots)
    allowed = set(requested)
    return [root for root in roots if root in allowed]


__all__ = [
    "normalise_requested_roots",
    "restrict_roots",
    "validate_requested_roots",
]
