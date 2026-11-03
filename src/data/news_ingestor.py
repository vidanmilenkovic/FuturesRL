"""Pluggable live news & macro headline ingestion (Phase 9, Part B3).

The LLM Risk-First engine needs a stream of *macro-relevant headlines* to feed its
semantic-risk assessments and the Risk Monitor live feed. This module provides one
stable contract (:class:`NewsHeadline`) and one resolver (:class:`NewsIngestor`)
with a strictly ordered, pluggable source strategy:

1. **RSS** — every configured ``settings.news_rss_feeds`` URL is fetched
   concurrently through an injectable async ``fetcher`` and parsed with the
   standard-library :mod:`xml.etree.ElementTree` parser (RSS 2.0/1.0 and Atom,
   namespace-agnostic). The default fetcher lazily imports ``httpx``; unit tests
   always inject a fetcher and never touch the network.
2. **IBKR** — when no RSS feeds are configured (or all RSS feeds fail) and an
   ``ib_client`` is injected, headline articles are requested through the
   duck-typed ``ib_async`` historical-news API (``reqNewsProvidersAsync`` /
   ``reqHistoricalNewsAsync``). Any failure degrades gracefully to the next
   source.
3. **Deterministic fixtures** — a fixed, clock-anchored set of macro/commodity
   headlines derived from the macro-calendar style of ``configs/macro_calendar.yaml``
   (FOMC, CPI, NFP, EIA, OPEC, WASDE, ...). No network, stable across calls for an
   injected clock, so offline backtests/CI are reproducible.

Every headline is classified into futures roots (keyword map), a severity bucket
and a simple sentiment score in ``[-1, 1]``. Industry conventions
------------------------------------------------------------------

* ``roots == ()`` means *market-wide*: the headline is relevant to every root and
  a ``roots`` filter keeps it.
* ``published_at`` is always timezone-aware UTC. Naive timestamps are rejected by
  :class:`NewsHeadline`; feed strings without a zone are interpreted as UTC
  (documented tolerance for otherwise-valid RSS/Atom payloads).
* ``source_kind`` records the *actual* source of a headline (``rss``, ``ibkr`` or
  ``fixture``), so API/telemetry consumers can label the feed and detect a
  fallback.
* Dedupe is by ``(headline, source)`` (case-insensitive); the newest occurrence
  wins and the result is sorted newest-first before ``limit`` is applied.
"""

from __future__ import annotations

import asyncio
import contextlib
import email.utils
import html
import inspect
import logging
import math
import re
from collections.abc import Awaitable, Callable, Coroutine, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, cast
from urllib.parse import urlparse
from xml.etree import ElementTree

from configs.settings import Settings, get_settings

from src.core.types import ensure_utc

__all__ = [
    "DEFAULT_MACRO_KEYWORDS",
    "DEFAULT_ROOT_KEYWORDS",
    "NewsHeadline",
    "NewsIngestor",
    "Severity",
    "SourceKind",
]

_LOGGER = logging.getLogger(__name__)

#: Severity bucket understood by the LLM risk overlay and the Risk Monitor feed.
type Severity = Literal["info", "warn", "critical"]

#: Where a headline actually came from.
type SourceKind = Literal["rss", "ibkr", "fixture"]

#: Injectable async HTTP fetcher: URL in, decoded response body out.
type NewsFetcher = Callable[[str], Awaitable[str]]

#: Injectable clock returning a timezone-aware datetime.
type Clock = Callable[[], datetime]

_SEVERITIES: frozenset[str] = frozenset({"info", "warn", "critical"})
_SOURCE_KINDS: frozenset[str] = frozenset({"rss", "ibkr", "fixture"})

_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_HTTP_PREFIXES: tuple[str, ...] = ("https://", "http://")

#: How far back IBKR historical news is requested.
_IBKR_LOOKBACK = timedelta(days=7)

# ---------------------------------------------------------------------------
# Classification lexicons
# ---------------------------------------------------------------------------

DEFAULT_MACRO_KEYWORDS: tuple[str, ...] = (
    "fomc",
    "fomc minutes",
    "federal reserve",
    "fed",
    "rate decision",
    "interest rate decision",
    "policy decision",
    "cpi",
    "consumer price index",
    "consumer prices",
    "inflation",
    "nfp",
    "nonfarm payrolls",
    "non-farm payrolls",
    "payrolls",
    "jobs report",
    "unemployment rate",
    "wasde",
    "usda",
    "macro data",
    "geopolitical",
)

#: Energy/metals keywords are shared by the standard and micro contracts.
_ENERGY_KEYWORDS: tuple[str, ...] = (
    "crude",
    "wti",
    "oil",
    "opec",
    "eia",
    "petroleum",
    "barrel",
    "cushing",
    "refinery",
    "energy",
)
_METALS_KEYWORDS: tuple[str, ...] = (
    "gold",
    "bullion",
    "precious metals",
    "xau",
    "silver",
    "safe haven",
)
_EQUITY_KEYWORDS: tuple[str, ...] = (
    "s&p",
    "sp500",
    "s&p 500",
    "e-mini",
    "emini",
    "equity index",
    "equities",
    "equity futures",
    "stock market",
)

#: Default root -> keyword map used to classify a headline into futures roots.
DEFAULT_ROOT_KEYWORDS: Mapping[str, tuple[str, ...]] = {
    "ES": (*_EQUITY_KEYWORDS, "spx"),
    "NQ": ("nasdaq", "tech stocks", "growth stocks", "mega-cap", "mega cap"),
    "YM": ("dow", "djia", "dow jones"),
    "RTY": ("russell", "small-cap", "small cap"),
    "CL": _ENERGY_KEYWORDS,
    "MCL": _ENERGY_KEYWORDS,
    "GC": _METALS_KEYWORDS,
    "MGC": _METALS_KEYWORDS,
    "ZN": (
        "treasury",
        "treasuries",
        "10-year",
        "10y",
        "yield curve",
        "duration",
        "government bonds",
        "note futures",
    ),
    "ZT": ("2-year", "2y", "short-dated", "front-end yields", "treasury bill"),
    "6E": ("euro", "eur/usd", "eurusd", "ecb", "fx", "currency", "dollar index"),
}

#: Every root of the default universe, used for macro headlines ("all roots").
_DEFAULT_ALL_ROOTS: tuple[str, ...] = tuple(sorted(DEFAULT_ROOT_KEYWORDS))

_CRITICAL_KEYWORDS: tuple[str, ...] = (
    "war",
    "invasion",
    "default",
    "bankruptcy",
    "collapse",
    "crash",
    "flash crash",
    "circuit breaker",
    "halted",
    "emergency",
    "crisis",
    "meltdown",
    "sanctions",
    "embargo",
    "terror",
    "black swan",
    "contagion",
    "geopolitical shock",
)

_WARN_KEYWORDS: tuple[str, ...] = (
    "risk",
    "warning",
    "selloff",
    "sell-off",
    "volatility",
    "downgrade",
    "recession",
    "slowdown",
    "tariff",
    "escalation",
    "uncertainty",
    "investigation",
    "shortage",
    "outage",
    "strike",
    "misses",
    "missed estimates",
    "below expectations",
    "cuts forecast",
    "cuts guidance",
    "layoffs",
    "probe",
    "inflation",
    "higher-for-longer",
)

#: Simple sentiment lexicon: keyword -> signed weight. Summed and clamped to [-1, 1].
_SENTIMENT_LEXICON: Mapping[str, float] = {
    "beats": 0.6,
    "beat": 0.5,
    "beat expectations": 0.7,
    "beat estimates": 0.7,
    "rally": 0.7,
    "rallies": 0.7,
    "rallied": 0.7,
    "surge": 0.6,
    "surges": 0.6,
    "soars": 0.8,
    "jumps": 0.5,
    "jumped": 0.4,
    "gains": 0.4,
    "rise": 0.3,
    "rises": 0.3,
    "upgrade": 0.6,
    "upgraded": 0.6,
    "record high": 0.8,
    "strong": 0.5,
    "optimism": 0.6,
    "optimistic": 0.6,
    "improves": 0.4,
    "rebound": 0.5,
    "boost": 0.5,
    "misses": -0.6,
    "missed estimates": -0.7,
    "plunge": -0.8,
    "plunges": -0.8,
    "slump": -0.7,
    "tumbles": -0.7,
    "selloff": -0.7,
    "sell-off": -0.7,
    "downgrade": -0.6,
    "downgraded": -0.6,
    "recession": -0.8,
    "crisis": -0.9,
    "crash": -0.9,
    "fall": -0.3,
    "slips": -0.3,
    "slip": -0.3,
    "slumped": -0.6,
    "falls": -0.3,
    "drop": -0.4,
    "drops": -0.4,
    "weak": -0.5,
    "warning": -0.4,
    "risk-off": -0.6,
    "layoffs": -0.6,
    "tariff": -0.5,
    "war": -0.9,
    "invasion": -0.9,
    "default": -0.8,
    "sanctions": -0.6,
    "inflation": -0.3,
    "accelerates": -0.3,
    "accelerating": -0.3,
    "slows": -0.4,
    "slowdown": -0.6,
    "fear": -0.5,
    "uncertainty": -0.3,
    "volatility": -0.2,
}


def _compile_keywords(keywords: Sequence[str]) -> tuple[re.Pattern[str], ...]:
    """Compile case-insensitive whole-token/phrase patterns for ``keywords``."""
    patterns: list[re.Pattern[str]] = []
    seen: set[str] = set()
    for keyword in keywords:
        phrase = keyword.strip().lower()
        if not phrase or phrase in seen:
            continue
        seen.add(phrase)
        escaped = re.escape(phrase)
        patterns.append(re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])"))
    return tuple(patterns)


MACRO_PATTERNS: tuple[re.Pattern[str], ...] = _compile_keywords(DEFAULT_MACRO_KEYWORDS)
_CRITICAL_PATTERNS: tuple[re.Pattern[str], ...] = _compile_keywords(_CRITICAL_KEYWORDS)
_WARN_PATTERNS: tuple[re.Pattern[str], ...] = _compile_keywords(_WARN_KEYWORDS)
_SENTIMENT_PATTERNS: tuple[tuple[re.Pattern[str], float], ...] = tuple(
    (pattern, _SENTIMENT_LEXICON[keyword])
    for keyword in _SENTIMENT_LEXICON
    for pattern in _compile_keywords((keyword,))
)

# ---------------------------------------------------------------------------
# Fixture headlines (deterministic, clock-anchored; no network)
# ---------------------------------------------------------------------------

#: (headline, source, summary, roots, severity, sentiment, minutes_ago)
_FIXTURE_TEMPLATES: tuple[tuple[str, str, str, tuple[str, ...], Severity, float, int], ...] = (
    (
        "FOMC minutes: officials see policy staying restrictive for longer",
        "Macro Wire",
        "Minutes show broad support for holding the policy rate at current levels.",
        _DEFAULT_ALL_ROOTS,
        "warn",
        -0.25,
        8,
    ),
    (
        "Core CPI lands in line with consensus; services disinflation continues",
        "Macro Wire",
        "Core consumer prices rose in line with expectations, easing policy pressure.",
        _DEFAULT_ALL_ROOTS,
        "info",
        0.3,
        47,
    ),
    (
        "OPEC+ delegates signal steady output; WTI crude holds above $78",
        "Energy Desk",
        "No change to production quotas is expected at this week's meeting.",
        ("CL", "MCL"),
        "info",
        0.15,
        96,
    ),
    (
        "EIA weekly crude inventories build 3.4M bbl vs -0.8M expected",
        "Energy Desk",
        "A larger-than-expected build weighs on the front of the crude curve.",
        ("CL", "MCL"),
        "warn",
        -0.45,
        151,
    ),
    (
        "Gold extends record run as real yields slip",
        "Metals Desk",
        "Bullion inflows accelerate for a third consecutive session.",
        ("GC", "MGC"),
        "info",
        0.55,
        212,
    ),
    (
        "Treasury 10Y auction tails; duration complex under pressure",
        "Rates Desk",
        "The long end cheapened after a soft afternoon auction.",
        ("ZN", "ZT"),
        "warn",
        -0.35,
        268,
    ),
    (
        "WASDE lifts US corn carryout; ag complex softer",
        "Ag Wire",
        "The USDA raised its carryout estimate for the new crop.",
        _DEFAULT_ALL_ROOTS,
        "info",
        -0.1,
        327,
    ),
    (
        "Geopolitical escalation: strikes halt a key oil export terminal",
        "Macro Wire",
        "Supply disruption risk spikes as exports are suspended indefinitely.",
        ("CL", "MCL", "GC", "MGC"),
        "critical",
        -0.9,
        392,
    ),
)


# ---------------------------------------------------------------------------
# NewsHeadline contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NewsHeadline:
    """One normalised market/macro headline consumed by the risk engine.

    ``published_at`` is always timezone-aware UTC (naive datetimes are rejected:
    a headline timestamp in the wrong zone would silently mis-age the risk feed).
    ``roots`` names the affected futures roots; an empty tuple means market-wide.
    ``sentiment`` is a simple lexicon score in ``[-1, 1]`` (negative = risk-off)
    or ``None`` when a source genuinely does not provide one.
    """

    headline: str
    source: str
    published_at: datetime
    summary: str = ""
    url: str | None = None
    roots: tuple[str, ...] = ()
    severity: Severity = "info"
    sentiment: float | None = None
    source_kind: SourceKind = "rss"

    def __post_init__(self) -> None:
        headline = self.headline.strip()
        if not headline:
            raise ValueError("news headline must be non-empty")
        object.__setattr__(self, "headline", headline)
        source = self.source.strip()
        if not source:
            raise ValueError("news source must be non-empty")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "published_at", ensure_utc(self.published_at))
        object.__setattr__(self, "summary", _collapse_whitespace(self.summary))
        if self.url is not None:
            url = self.url.strip()
            object.__setattr__(self, "url", url or None)
        cleaned_roots: set[str] = set()
        for root in self.roots:
            if not isinstance(root, str):
                raise TypeError(f"news roots must be strings, got {type(root)!r}")
            cleaned = root.strip().upper()
            if not cleaned:
                raise ValueError("news roots must be non-empty strings")
            cleaned_roots.add(cleaned)
        object.__setattr__(self, "roots", tuple(sorted(cleaned_roots)))
        if self.severity not in _SEVERITIES:
            raise ValueError(
                f"news severity must be one of {sorted(_SEVERITIES)}, got {self.severity!r}"
            )
        if self.source_kind not in _SOURCE_KINDS:
            raise ValueError(
                f"news source_kind must be one of {sorted(_SOURCE_KINDS)}, got {self.source_kind!r}"
            )
        if self.sentiment is not None:
            if isinstance(self.sentiment, bool) or not isinstance(self.sentiment, (int, float)):
                raise TypeError(f"news sentiment must be a float, got {type(self.sentiment)!r}")
            sentiment = float(self.sentiment)
            if not math.isfinite(sentiment) or not -1.0 <= sentiment <= 1.0:
                raise ValueError(f"news sentiment must be within [-1, 1], got {sentiment!r}")
            object.__setattr__(self, "sentiment", sentiment)

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable representation (``published_at`` becomes ISO-8601 UTC)."""
        return {
            "headline": self.headline,
            "source": self.source,
            "published_at": self.published_at.isoformat(),
            "summary": self.summary,
            "url": self.url,
            "roots": list(self.roots),
            "severity": self.severity,
            "sentiment": self.sentiment,
            "source_kind": self.source_kind,
        }

    def __repr__(self) -> str:
        return (
            f"NewsHeadline(headline={self.headline!r}, source={self.source!r}, "
            f"published_at={self.published_at.isoformat()}, roots={self.roots!r}, "
            f"severity={self.severity!r}, sentiment={self.sentiment!r}, "
            f"source_kind={self.source_kind!r})"
        )


# ---------------------------------------------------------------------------
# Raw article parsing (RSS 2.0 / RSS 1.0 / Atom)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _RawArticle:
    """One pre-classification article extracted from a feed or IBKR response."""

    headline: str
    source: str
    summary: str
    published_raw: str | None
    url: str | None


def _collapse_whitespace(text: str) -> str:
    """Collapse runs of whitespace to single spaces (also strips)."""
    return _WHITESPACE_PATTERN.sub(" ", text).strip()


def _strip_html(text: str) -> str:
    """Strip HTML tags/entities from a feed summary/title."""
    return _collapse_whitespace(html.unescape(_HTML_TAG_PATTERN.sub(" ", text)))


def _local_name(tag: object) -> str:
    """Namespace-agnostic lowercase XML tag name."""
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1].lower()


def _first_child_text(element: ElementTree.Element, names: Sequence[str]) -> str | None:
    """First non-blank direct-child text among ``names`` (local-name match)."""
    wanted = tuple(name.lower() for name in names)
    for name in wanted:
        for child in element:
            if _local_name(child.tag) == name and child.text and child.text.strip():
                return child.text
    return None


def _extract_link(element: ElementTree.Element) -> str | None:
    """Extract an RSS ``<link>`` text or Atom ``<link href=...>`` attribute."""
    for child in element:
        if _local_name(child.tag) != "link":
            continue
        href = child.get("href")
        if href and href.strip():
            return href.strip()
        if child.text and child.text.strip():
            return child.text.strip()
    return None


def _parse_timestamp(raw: str | None) -> datetime | None:
    """Parse an RFC-822 or ISO-8601 timestamp into an aware UTC datetime.

    Supports the variants seen in financial RSS/Atom feeds (RFC-822 via
    :func:`email.utils.parsedate_to_datetime`, ISO-8601 with/without ``Z``,
    date-only). Zone-less strings are interpreted as UTC; unparseable strings
    return ``None`` so the caller can fall back to its clock.
    """
    if raw is None:
        return None
    text = html.unescape(raw.strip())
    if not text:
        return None
    parsed: datetime | None = None
    with contextlib.suppress(TypeError, ValueError):
        parsed = email.utils.parsedate_to_datetime(text)
    if parsed is None:
        candidate = text
        if candidate.endswith(("Z", "z")):
            candidate = f"{candidate[:-1]}+00:00"
        with contextlib.suppress(ValueError):
            parsed = datetime.fromisoformat(candidate)
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _coerce_datetime(value: object) -> datetime | None:
    """Coerce a datetime/string value to aware UTC (``None`` when impossible)."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    if isinstance(value, str):
        return _parse_timestamp(value)
    return None


def _parse_feed_document(xml_text: str, *, fallback_source: str) -> list[_RawArticle]:
    """Parse an RSS 2.0/1.0 or Atom document into raw articles.

    Raises:
        ValueError: the payload is not well-formed XML.
    """
    if not xml_text or not xml_text.strip():
        raise ValueError("empty feed document")
    try:
        root = ElementTree.fromstring(xml_text.encode("utf-8"))
    except ElementTree.ParseError as exc:
        raise ValueError(f"invalid XML feed: {exc}") from exc
    feed_title = _first_child_text(root, ("title",))
    if feed_title is None:
        # RSS 2.0/1.0 nests the channel title one level below the document root.
        for child in root:
            if _local_name(child.tag) in ("channel", "feed") and _local_name(child.tag) != "feed":
                feed_title = _first_child_text(child, ("title",))
                if feed_title is not None:
                    break
    source = _strip_html(feed_title) if feed_title else fallback_source
    articles: list[_RawArticle] = []
    for element in root.iter():
        if _local_name(element.tag) not in ("item", "entry"):
            continue
        title = _first_child_text(element, ("title",))
        if title is None or not _strip_html(title):
            continue
        summary = _first_child_text(element, ("description", "summary", "content", "encoded"))
        published = _first_child_text(
            element, ("pubdate", "published", "updated", "date", "issued")
        )
        articles.append(
            _RawArticle(
                headline=_strip_html(title),
                source=source,
                summary=_strip_html(summary) if summary else "",
                published_raw=published,
                url=_extract_link(element),
            )
        )
    return articles


def _fallback_source(url: str) -> str:
    """Human-readable feed source when the document has no channel title."""
    netloc = urlparse(url).netloc
    return netloc or url


def _run_sync[T](coro: Coroutine[Any, Any, T]) -> T:
    """Run ``coro`` to completion from sync code, even inside a running loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    results: list[T] = []
    errors: list[BaseException] = []

    def runner() -> None:
        try:
            results.append(asyncio.run(coro))
        except BaseException as exc:
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(runner).result()
    if errors:
        raise errors[0]
    if not results:  # pragma: no cover - runner always appends or raises
        raise RuntimeError("coroutine produced no result")
    return results[0]


async def _default_fetch(url: str) -> str:
    """Default async RSS fetcher; imports ``httpx`` lazily and raises if absent."""
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - httpx is a project dependency
        raise RuntimeError(
            "RSS news ingestion requires the httpx package: install it with "
            "`pip install httpx`, or inject a custom `fetcher`."
        ) from exc
    async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.text


# ---------------------------------------------------------------------------
# NewsIngestor
# ---------------------------------------------------------------------------


class NewsIngestor:
    """Resolve live/fixture news headlines implementing the B3 data contract.

    Parameters
    ----------
    settings:
        Runtime settings; supplies ``news_rss_feeds``, ``news_max_items`` and
        ``news_poll_interval_seconds``. Defaults to the process singleton.
    feeds:
        Explicit RSS feed override (``None`` = use settings; empty tuple forces
        the IBKR/fixture path).
    fetcher:
        Injectable async HTTP fetcher (``Callable[[str], Awaitable[str]]``). The
        default lazily imports ``httpx``; tests always inject a fake so the
        network is never touched.
    ib_client:
        Duck-typed IBKR client exposing ``reqNewsProvidersAsync`` and/or
        ``reqHistoricalNewsAsync`` (ib_async-style async methods). Failures fall
        back gracefully to deterministic fixtures.
    clock:
        Injectable clock returning an aware datetime (fixtures/undated articles).
    fixture_headlines:
        Explicit deterministic fixtures overriding the built-in macro/commodity
        set (source_kind is normalised to ``"fixture"``).
    root_keywords:
        Root -> keyword overrides for headline classification (defaults to
        :data:`DEFAULT_ROOT_KEYWORDS`). Macro keywords always map to all roots.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        feeds: Sequence[str] | None = None,
        fetcher: NewsFetcher | None = None,
        ib_client: object | None = None,
        clock: Clock | None = None,
        fixture_headlines: Sequence[NewsHeadline] | None = None,
        root_keywords: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        self._settings = settings if settings is not None else get_settings()
        configured = feeds if feeds is not None else self._settings.news_rss_feeds
        cleaned_feeds: list[str] = []
        for feed in configured:
            if not isinstance(feed, str):
                raise TypeError(f"news feed URLs must be strings, got {type(feed)!r}")
            url = feed.strip()
            if not url:
                continue
            if not url.startswith(_HTTP_PREFIXES):
                raise ValueError(f"news RSS feed must be an http(s) URL, got {url!r}")
            cleaned_feeds.append(url)
        self._feeds: tuple[str, ...] = tuple(dict.fromkeys(cleaned_feeds))
        self._fetcher: NewsFetcher = fetcher if fetcher is not None else _default_fetch
        self._ib_client = ib_client
        self._clock: Clock = clock if clock is not None else _utc_now
        self._explicit_fixtures = _validate_fixtures(fixture_headlines)
        self._root_keywords, self._root_patterns = _build_root_keywords(root_keywords)
        self._all_roots: tuple[str, ...] = tuple(sorted(self._root_keywords))

    # -- introspection ---------------------------------------------------------

    @property
    def source_kind(self) -> SourceKind:
        """Preferred source kind before fallbacks: ``rss`` > ``ibkr`` > ``fixture``."""
        if self._feeds:
            return "rss"
        if self._ib_client is not None:
            return "ibkr"
        return "fixture"

    @property
    def feeds(self) -> tuple[str, ...]:
        """Configured RSS feed URLs (empty when RSS is not used)."""
        return self._feeds

    @property
    def max_items(self) -> int:
        """Default result limit (``settings.news_max_items``)."""
        return int(self._settings.news_max_items)

    @property
    def poll_interval_seconds(self) -> float:
        """Recommended polling interval for the API/live loop."""
        return float(self._settings.news_poll_interval_seconds)

    def __repr__(self) -> str:
        return (
            f"NewsIngestor(source_kind={self.source_kind!r}, feeds={len(self._feeds)}, "
            f"ib_client={type(self._ib_client).__name__ if self._ib_client is not None else None})"
        )

    # -- public API ------------------------------------------------------------

    def fetch(
        self, *, limit: int | None = None, roots: Sequence[str] | None = None
    ) -> tuple[NewsHeadline, ...]:
        """Synchronously resolve the newest headlines (safe inside a running loop)."""
        return _run_sync(self.fetch_async(limit=limit, roots=roots))

    async def fetch_async(
        self, *, limit: int | None = None, roots: Sequence[str] | None = None
    ) -> tuple[NewsHeadline, ...]:
        """Asynchronously resolve the newest headlines (RSS -> IBKR -> fixtures)."""
        resolved_limit = self._resolve_limit(limit)
        requested_roots = self._normalise_roots_filter(roots)
        if self._feeds:
            headlines = await self._fetch_rss()
            if headlines:
                return self._finalise(headlines, resolved_limit, requested_roots)
            _LOGGER.warning(
                "RSS news unavailable (%d configured feed(s)); trying the next source",
                len(self._feeds),
            )
        if self._ib_client is not None:
            headlines = await self._fetch_ibkr()
            if headlines:
                return self._finalise(headlines, resolved_limit, requested_roots)
            _LOGGER.warning("IBKR news unavailable; using deterministic fixture headlines")
        return self._finalise(self._fixture_headlines(), resolved_limit, requested_roots)

    def to_dict(self, headlines: Sequence[NewsHeadline]) -> list[dict[str, object]]:
        """JSON-serialisable list of headline dictionaries."""
        return [headline.to_dict() for headline in headlines]

    # -- resolution: RSS -------------------------------------------------------

    async def _fetch_rss(self) -> list[NewsHeadline]:
        """Fetch and parse every configured feed concurrently (partial failure ok)."""
        results = await asyncio.gather(
            *(self._fetch_feed(feed) for feed in self._feeds),
            return_exceptions=True,
        )
        headlines: list[NewsHeadline] = []
        for feed, result in zip(self._feeds, results, strict=True):
            if isinstance(result, BaseException):
                _LOGGER.warning("RSS feed %s failed: %s", feed, result)
                continue
            headlines.extend(result)
        return headlines

    async def _fetch_feed(self, url: str) -> list[NewsHeadline]:
        """Fetch and normalise one RSS/Atom feed; raises on transport/parse errors."""
        body = await self._fetcher(url)
        if not isinstance(body, str):
            raise TypeError(f"news fetcher returned {type(body)!r}, expected str")
        articles = _parse_feed_document(body, fallback_source=_fallback_source(url))
        now = self._now()
        headlines: list[NewsHeadline] = []
        for article in articles:
            published = _parse_timestamp(article.published_raw) or now
            headlines.append(self._build_headline(article, "rss", published))
        return headlines

    def _build_headline(
        self, article: _RawArticle, kind: SourceKind, published: datetime
    ) -> NewsHeadline:
        """Classify a raw article and build the frozen headline contract."""
        text = f"{article.headline} {article.summary}"
        return NewsHeadline(
            headline=article.headline,
            source=article.source,
            published_at=published,
            summary=article.summary,
            url=article.url,
            roots=self._classify_roots(text),
            severity=self._classify_severity(text),
            sentiment=self._classify_sentiment(text),
            source_kind=kind,
        )

    # -- resolution: IBKR ------------------------------------------------------

    async def _fetch_ibkr(self) -> list[NewsHeadline]:
        """Best-effort IBKR news ingestion; ``None``/``[]`` means "use fixtures"."""
        client = self._ib_client
        if client is None:  # pragma: no cover - caller checked
            return []
        try:
            providers = await self._ibkr_providers(client)
            raw_articles = await self._ibkr_articles(client, providers)
        except Exception as exc:
            _LOGGER.warning("IBKR news request failed: %s", exc)
            return []
        now = self._now()
        headlines: list[NewsHeadline] = []
        for article in raw_articles:
            title_value = _field(article, ("headline", "title"))
            if not isinstance(title_value, str) or not _strip_html(title_value):
                continue
            headline = _strip_html(title_value)
            summary_value = _field(article, ("summary", "extra_data", "extradata"))
            summary = _strip_html(summary_value) if isinstance(summary_value, str) else ""
            published = (
                _coerce_datetime(_field(article, ("time", "published_at", "published", "date")))
                or now
            )
            provider_code = _field(article, ("provider_code", "providerCode"))
            source = (
                providers.get(str(provider_code).strip(), str(provider_code).strip())
                if isinstance(provider_code, str) and provider_code.strip()
                else "IBKR"
            )
            url_value = _field(article, ("url",))
            url = url_value.strip() if isinstance(url_value, str) and url_value.strip() else None
            raw = _RawArticle(
                headline=headline,
                source=source,
                summary=summary,
                published_raw=None,
                url=url,
            )
            headlines.append(self._build_headline(raw, "ibkr", published))
        if not headlines:
            _LOGGER.warning("IBKR news returned no usable articles")
        return headlines

    async def _ibkr_providers(self, client: object) -> dict[str, str]:
        """Provider code -> display-name map (empty on unavailable/failure)."""
        method = _client_method(client, "reqNewsProvidersAsync")
        if method is None:
            return {}
        try:
            result = await _maybe_await(method())
        except Exception as exc:
            _LOGGER.warning("IBKR news providers unavailable: %s", exc)
            return {}
        providers: dict[str, str] = {}
        for provider in _as_sequence(result):
            code = _field(provider, ("code", "provider_code", "providerCode"))
            name = _field(provider, ("name", "provider_name", "providerName"))
            if not isinstance(code, str) or not code.strip():
                continue
            cleaned_code = code.strip()
            providers[cleaned_code] = (
                name.strip() if isinstance(name, str) and name.strip() else cleaned_code
            )
        return providers

    async def _ibkr_articles(self, client: object, providers: Mapping[str, str]) -> list[object]:
        """Request historical news articles from the duck-typed IBKR client."""
        now = self._now()
        total_results = max(int(self._settings.news_max_items), 10)
        historical = _client_method(client, "reqHistoricalNewsAsync")
        if historical is None:
            for fallback_name in ("get_news_async", "news_async", "fetch_news_async"):
                generic = _client_method(client, fallback_name)
                if generic is None:
                    continue
                result = await _maybe_await(generic())
                return _as_sequence(result)
            return []
        provider_codes: tuple[str, ...] = tuple(providers) or ("",)
        results = await asyncio.gather(
            *(
                _call_historical_news(
                    historical,
                    provider_code,
                    now - _IBKR_LOOKBACK,
                    now,
                    total_results,
                )
                for provider_code in provider_codes
            ),
            return_exceptions=True,
        )
        articles: list[object] = []
        failures = 0
        for result in results:
            if isinstance(result, BaseException):
                failures += 1
                _LOGGER.warning("IBKR historical news request failed: %s", result)
                continue
            articles.extend(_articles_from_response(result))
        if failures == len(results) and results:
            return []
        return articles

    # -- resolution: fixtures --------------------------------------------------

    def _fixture_headlines(self) -> list[NewsHeadline]:
        """Deterministic fixture headlines (explicit override or built-in set)."""
        now = self._now()
        if self._explicit_fixtures is not None:
            return [
                replace(headline, source_kind="fixture")
                if headline.source_kind != "fixture"
                else headline
                for headline in self._explicit_fixtures
            ]
        return [
            NewsHeadline(
                headline=headline,
                source=source,
                published_at=now - timedelta(minutes=minutes_ago),
                summary=summary,
                roots=roots,
                severity=severity,
                sentiment=sentiment,
                source_kind="fixture",
            )
            for headline, source, summary, roots, severity, sentiment, minutes_ago in (
                _FIXTURE_TEMPLATES
            )
        ]

    # -- classification --------------------------------------------------------

    def _classify_roots(self, text: str) -> tuple[str, ...]:
        """Map a headline to futures roots; macro keywords map to every root."""
        lowered = text.lower()
        if any(pattern.search(lowered) for pattern in MACRO_PATTERNS):
            return self._all_roots
        matched: list[str] = []
        for root in sorted(self._root_patterns):
            if any(pattern.search(lowered) for pattern in self._root_patterns[root]):
                matched.append(root)
        return tuple(matched)

    @staticmethod
    def _classify_severity(text: str) -> Severity:
        """Classify severity from critical/warn keyword lexicons."""
        lowered = text.lower()
        if any(pattern.search(lowered) for pattern in _CRITICAL_PATTERNS):
            return "critical"
        if any(pattern.search(lowered) for pattern in _WARN_PATTERNS):
            return "warn"
        return "info"

    @staticmethod
    def _classify_sentiment(text: str) -> float:
        """Simple lexicon sentiment score clamped to ``[-1, 1]``."""
        lowered = text.lower()
        score = sum(weight for pattern, weight in _SENTIMENT_PATTERNS if pattern.search(lowered))
        return round(max(-1.0, min(1.0, score)), 6)

    # -- helpers ---------------------------------------------------------------

    def _now(self) -> datetime:
        """Current time from the injectable clock, normalised to UTC."""
        moment = self._clock()
        if not isinstance(moment, datetime):
            raise TypeError(f"news clock must return a datetime, got {type(moment)!r}")
        return ensure_utc(moment)

    def _resolve_limit(self, limit: int | None) -> int:
        """Validate an explicit limit or fall back to ``settings.news_max_items``."""
        if limit is None:
            return self.max_items
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError(f"news limit must be an int, got {type(limit)!r}")
        if limit < 1:
            raise ValueError(f"news limit must be >= 1, got {limit}")
        return limit

    @staticmethod
    def _normalise_roots_filter(roots: Sequence[str] | None) -> tuple[str, ...] | None:
        """Normalise a roots filter (``None``/empty = no filter)."""
        if roots is None:
            return None
        if isinstance(roots, str):
            raise TypeError("news roots filter must be a sequence of root symbols")
        requested = sorted({root.strip().upper() for root in roots if root.strip()})
        return tuple(requested) or None

    def _finalise(
        self,
        headlines: Sequence[NewsHeadline],
        limit: int,
        roots: Sequence[str] | None,
    ) -> tuple[NewsHeadline, ...]:
        """Dedupe (headline+source), sort newest-first, filter roots, apply limit."""
        ordered = sorted(
            headlines,
            key=lambda headline: (headline.published_at, headline.headline),
            reverse=True,
        )
        deduped: dict[tuple[str, str], NewsHeadline] = {}
        for headline in ordered:
            key = (headline.headline.casefold(), headline.source.casefold())
            deduped.setdefault(key, headline)
        selected = list(deduped.values())
        if roots is not None:
            requested = set(roots)
            selected = [
                headline
                for headline in selected
                if not headline.roots or requested.intersection(headline.roots)
            ]
        return tuple(selected[:limit])


# ---------------------------------------------------------------------------
# Constructor / IBKR helper functions
# ---------------------------------------------------------------------------


def _utc_now() -> datetime:
    """Default clock: current timezone-aware UTC time."""
    return datetime.now(UTC)


def _validate_fixtures(
    fixtures: Sequence[NewsHeadline] | None,
) -> tuple[NewsHeadline, ...] | None:
    """Validate explicit fixture headlines (``None`` = use built-ins)."""
    if fixtures is None:
        return None
    cleaned: list[NewsHeadline] = []
    for fixture in fixtures:
        if not isinstance(fixture, NewsHeadline):
            raise TypeError(
                f"fixture_headlines must contain NewsHeadline objects, got {type(fixture)!r}"
            )
        cleaned.append(fixture)
    if not cleaned:
        raise ValueError("fixture_headlines must not be empty when provided")
    return tuple(cleaned)


def _build_root_keywords(
    overrides: Mapping[str, Sequence[str]] | None,
) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[re.Pattern[str], ...]]]:
    """Normalise a root keyword map and compile its patterns."""
    source: Mapping[str, Sequence[str]] = (
        overrides if overrides is not None else (DEFAULT_ROOT_KEYWORDS)
    )
    if not source:
        raise ValueError("root_keywords must declare at least one root")
    keywords: dict[str, tuple[str, ...]] = {}
    patterns: dict[str, tuple[re.Pattern[str], ...]] = {}
    for root, words in source.items():
        if not isinstance(root, str) or not root.strip():
            raise ValueError(f"root_keywords keys must be non-empty strings, got {root!r}")
        cleaned_root = root.strip().upper()
        if isinstance(words, str) or not isinstance(words, Sequence):
            raise TypeError(f"root_keywords[{root!r}] must be a sequence of keywords")
        cleaned_words = tuple(
            word.strip() for word in words if isinstance(word, str) and word.strip()
        )
        if not cleaned_words:
            raise ValueError(f"root_keywords[{root!r}] must declare at least one keyword")
        if cleaned_root in keywords:
            raise ValueError(f"duplicate root {cleaned_root!r} in root_keywords")
        keywords[cleaned_root] = cleaned_words
        patterns[cleaned_root] = _compile_keywords(cleaned_words)
    return keywords, patterns


def _client_method(client: object, name: str) -> Callable[..., object] | None:
    """Return a callable client method by name, or ``None`` when unavailable."""
    candidate = getattr(client, name, None)
    if callable(candidate):
        return cast(Callable[..., object], candidate)
    return None


async def _maybe_await(value: object) -> object:
    """Await ``value`` when it is awaitable (duck-typed async/sync client APIs)."""
    if inspect.isawaitable(value):
        return await value
    return value


async def _call_historical_news(
    method: Callable[..., object],
    provider_code: str,
    start: datetime,
    end: datetime,
    total_results: int,
) -> object:
    """Call an ib_async-style ``reqHistoricalNewsAsync`` (positional signature)."""
    return await _maybe_await(method(0, provider_code, start, end, total_results))


def _field(item: object, names: Sequence[str]) -> object:
    """Read the first present attribute/key among ``names``."""
    if isinstance(item, Mapping):
        for name in names:
            value = item.get(name)
            if value is not None:
                return cast(object, value)
        return None
    for name in names:
        value = getattr(item, name, None)
        if value is not None:
            return value
    return None


def _as_sequence(value: object) -> list[object]:
    """Normalise an iterable/list response into a list of items."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, (str, bytes)):
        return []
    try:
        return list(cast(Iterable[object], value))
    except TypeError:
        return []


def _articles_from_response(response: object) -> list[object]:
    """Extract article objects from an IBKR historical-news response."""
    articles = getattr(response, "articles", None)
    if articles is None and isinstance(response, Mapping):
        articles = response.get("articles")
    if articles is not None:
        return _as_sequence(articles)
    return _as_sequence(response)
