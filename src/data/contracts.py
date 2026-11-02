"""Futures contract calendar and symbology helpers.

These helpers are shared by the Databento historical loader, the deterministic
offline bar source and the continuous-roll engine. They deliberately contain
*no* market data: they only translate between futures roots, contract-month
codes, representative expiration dates and Databento's native continuous
symbology.

Important
---------
Real expiration dates (and first notice dates) always come from the vendor
``definition`` schema. The calendar helpers here are only a *deterministic
fallback* for:

* synthetic/offline data (which has no vendor definition record), and
* parsing a raw CME symbol (``"ESM4"``) when no instrument-id definition join is
  available. In that case the expiry is approximated as the third Friday of the
  contract month and is flagged as approximate by the caller's documentation.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, timedelta

__all__ = [
    "INVERSE_MONTH_CODES",
    "MONTH_CODES",
    "NATIVE_CONTINUOUS_PATTERN",
    "contract_symbol",
    "month_code",
    "native_continuous_symbol",
    "next_contract_expiration",
    "parse_contract_symbol",
    "parse_native_continuous_symbol",
    "representative_expiration",
    "third_friday",
]

#: CME/Globex futures month codes.
MONTH_CODES: Mapping[str, int] = {
    "F": 1,
    "G": 2,
    "H": 3,
    "J": 4,
    "K": 5,
    "M": 6,
    "N": 7,
    "Q": 8,
    "U": 9,
    "V": 10,
    "X": 11,
    "Z": 12,
}

#: Inverse mapping (month number -> month code).
INVERSE_MONTH_CODES: Mapping[int, str] = {month: code for code, month in MONTH_CODES.items()}

#: Databento native continuous symbology pattern (``ROOT.v.0`` / ``ROOT.c.0``).
NATIVE_CONTINUOUS_PATTERN = re.compile(
    r"^(?P<root>[A-Z0-9]{1,6})\.(?P<kind>[vVcC])\.(?P<index>\d+)$"
)

_CONTRACT_SUFFIX_PATTERN = re.compile(r"^([FGHJKMNQUVXZ])(\d{1,2})$")
_GENERIC_CONTRACT_PATTERN = re.compile(r"^([A-Z0-9]{1,4}?)([FGHJKMNQUVXZ])(\d{1,2})$")


def month_code(month: int) -> str:
    """Return the futures month code for a calendar ``month`` (1..12)."""
    try:
        return INVERSE_MONTH_CODES[month]
    except KeyError as exc:
        raise ValueError(f"month must be within 1..12, got {month!r}") from exc


def third_friday(year: int, month: int) -> date:
    """Third Friday of ``month`` — a representative CME futures expiration."""
    first = date(year, month, 1)
    # Friday == 4 in ``date.weekday`` (Monday == 0).
    offset = (4 - first.weekday()) % 7
    return first + timedelta(days=offset + 14)


def representative_expiration(year: int, month: int) -> date:
    """Deterministic representative expiration used when no vendor date exists."""
    return third_friday(year, month)


def contract_symbol(root: str, expiration: date) -> str:
    """Build a CME-style contract symbol (``root='ES'``, 2024-03 -> ``ESM4``)."""
    normalised = root.strip().upper()
    if not normalised:
        raise ValueError("root must be a non-empty string")
    return f"{normalised}{month_code(expiration.month)}{expiration.year % 10}"


def native_continuous_symbol(root: str, methodology: str = "volume") -> str:
    """Databento native continuous symbol for ``root`` (``ES.v.0`` / ``ES.c.0``)."""
    normalised = root.strip().upper()
    if methodology not in ("volume", "calendar"):
        raise ValueError(f"methodology must be 'volume' or 'calendar', got {methodology!r}")
    suffix = ".v.0" if methodology == "volume" else ".c.0"
    return f"{normalised}{suffix}"


def parse_native_continuous_symbol(symbol: str) -> tuple[str, str] | None:
    """Split ``ES.v.0`` into ``("ES", "volume")``; ``None`` for non-native symbols."""
    match = NATIVE_CONTINUOUS_PATTERN.match(symbol.strip().upper())
    if match is None:
        return None
    methodology = "volume" if match.group("kind").lower() == "v" else "calendar"
    return (match.group("root"), methodology)


def next_contract_expiration(contract_months: Sequence[int], reference: date | datetime) -> date:
    """First representative expiration strictly after ``reference``.

    ``contract_months`` is the instrument's declared delivery cycle (e.g.
    ``(3, 6, 9, 12)`` for equity index futures, every month for WTI).
    """
    if not contract_months:
        raise ValueError("contract_months must not be empty")
    reference_date = reference.date() if isinstance(reference, datetime) else reference
    candidates: list[date] = []
    for year in (reference_date.year, reference_date.year + 1, reference_date.year + 2):
        for month in contract_months:
            expiration = representative_expiration(year, month)
            if expiration > reference_date:
                candidates.append(expiration)
    if not candidates:
        raise ValueError(  # pragma: no cover - contract_months is validated non-empty
            f"could not derive a future expiration after {reference_date.isoformat()}"
        )
    return min(candidates)


def _infer_expiration_year(month: int, year_digit: int, reference: date) -> int:
    """Resolve a single-digit contract year (``M4``) relative to ``reference``.

    The nearest year whose representative expiration is not more than ~13 months
    before ``reference`` wins, which matches how CME symbols roll over decades.
    """
    year = (reference.year // 10) * 10 + year_digit
    while representative_expiration(year, month) < reference - timedelta(days=400):
        year += 10
    while representative_expiration(year, month) > reference + timedelta(days=365 * 2):
        year -= 10
    return year


def parse_contract_symbol(
    symbol: str,
    *,
    roots: Sequence[str] | None = None,
    reference: date | datetime | None = None,
) -> tuple[str, date] | None:
    """Parse a raw futures contract symbol into ``(root, expiration)``.

    Returns ``None`` when the symbol is not a concrete dated contract (for
    example the parent symbol ``ES.FUT`` or a native continuous symbol
    ``ES.v.0``). The expiration is the representative third-Friday expiry — a
    caller that has vendor definitions must prefer those.
    """
    text = symbol.strip().upper()
    if not text or parse_native_continuous_symbol(text) is not None:
        return None
    if text.endswith(".FUT") or "." in text:
        return None
    moment = reference or datetime.now(UTC)
    reference_date = moment.date() if isinstance(moment, datetime) else moment

    candidates = (
        sorted({root.strip().upper() for root in roots}, key=len, reverse=True) if roots else []
    )
    for root in candidates:
        if not text.startswith(root):
            continue
        match = _CONTRACT_SUFFIX_PATTERN.match(text[len(root) :])
        if match is not None:
            return (root, _expiration_from_match(match, reference_date))
    match = _GENERIC_CONTRACT_PATTERN.match(text)
    if match is None:
        return None
    suffix = _CONTRACT_SUFFIX_PATTERN.match(match.group(2) + match.group(3))
    if suffix is None:  # pragma: no cover - guarded by the generic pattern
        return None
    return (match.group(1), _expiration_from_match(suffix, reference_date))


def _expiration_from_match(match: re.Match[str], reference: date) -> date:
    """Convert a ``(month code, year digits)`` regex match to an expiration date."""
    month = MONTH_CODES[match.group(1)]
    year_text = match.group(2)
    if len(year_text) == 2:
        year = 2000 + int(year_text)
    else:
        year = _infer_expiration_year(month, int(year_text), reference)
    return representative_expiration(year, month)
