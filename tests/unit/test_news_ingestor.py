"""Unit tests for the Phase-9 pluggable news/macro headline ingestor.

Every test is offline and deterministic: RSS/Atom payloads are injected through a
fake async fetcher (the default ``httpx`` fetcher is only exercised through a fake
``httpx`` module), IBKR news uses an injected duck-typed fake client, and fixtures
use an injected clock. No network access is performed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest
from configs.settings import Settings
from src.data.news_ingestor import (
    DEFAULT_MACRO_KEYWORDS,
    DEFAULT_ROOT_KEYWORDS,
    NewsHeadline,
    NewsIngestor,
    Severity,
    SourceKind,
)

FIXED_NOW = datetime(2024, 10, 2, 18, 0, tzinfo=UTC)


def fixed_clock() -> datetime:
    """Deterministic UTC clock for fixture/undated-article tests."""
    return FIXED_NOW


RSS2_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Macro Wire</title>
    <link>https://macro.example.test/</link>
    <item>
      <title>Fed signals higher-for-longer rates as CPI inflation stays sticky</title>
      <description>&lt;p&gt;Consumer prices jumped more than expected.&lt;/p&gt;</description>
      <pubDate>Wed, 02 Oct 2024 12:30:00 +0000</pubDate>
      <link>https://macro.example.test/fed-cpi</link>
      <guid>macro-1</guid>
    </item>
    <item>
      <title>OPEC+ agrees to extend output cuts; WTI crude rallies</title>
      <description>Barrels extend gains after the cartel decision.</description>
      <pubDate>Tue, 01 Oct 2024 15:00:00 GMT</pubDate>
    </item>
    <item>
      <title>Gold surges as war risk spooks investors</title>
      <pubDate>2024-09-30T13:30:00Z</pubDate>
    </item>
    <item>
      <title>Nasdaq tech stocks slump as layoffs mount</title>
      <pubDate>2024-09-30T09:15:00Z</pubDate>
    </item>
  </channel>
</rss>"""

ATOM_XML = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Atom Macro Desk</title>
  <entry>
    <title>Treasury 10Y auction tails; duration under pressure</title>
    <summary>The long end cheapened after a soft auction.</summary>
    <link href="https://atom.example.test/10y-auction"/>
    <published>2024-10-02T15:00:00Z</published>
    <id>atom-1</id>
  </entry>
  <entry>
    <title>Euro slips as ECB officials hint at earlier cuts</title>
    <link href="https://atom.example.test/ecb"/>
    <updated>2024-10-01T08:00:00+02:00</updated>
  </entry>
  <entry>
    <title>No publication date here</title>
  </entry>
</feed>"""

RSS1_XML = """<?xml version="1.0" encoding="utf-8"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:dc="http://purl.org/dc/elements/1.1/"
         xmlns="http://purl.org/rss/1.0/">
  <channel rdf:about="https://rss1.example.test/">
    <title>RSS One Wire</title>
  </channel>
  <item rdf:about="https://rss1.example.test/item/1">
    <title>WASDE raises US corn carryout</title>
    <description>USDA data adds to the ag complex bearish tone.</description>
    <dc:date>2024-09-28T16:00:00Z</dc:date>
    <link>https://rss1.example.test/item/1</link>
  </item>
</rdf:RDF>"""


def make_ingestor(
    settings: Settings,
    *,
    feeds: tuple[str, ...] | None = None,
    fetcher: Any = None,
    ib_client: object | None = None,
    clock: Any = fixed_clock,
    fixture_headlines: tuple[NewsHeadline, ...] | None = None,
    root_keywords: dict[str, tuple[str, ...]] | None = None,
) -> NewsIngestor:
    """Build an ingestor with a deterministic clock and injected sources."""
    return NewsIngestor(
        settings,
        feeds=feeds,
        fetcher=fetcher,
        ib_client=ib_client,
        clock=clock,
        fixture_headlines=fixture_headlines,
        root_keywords=root_keywords,
    )


async def xml_fetcher(xml: str) -> Any:
    """Return a fetcher that always serves ``xml``."""

    async def fetch(url: str) -> str:
        del url
        return xml

    return fetch


# ---------------------------------------------------------------------------
# Source resolution order / construction
# ---------------------------------------------------------------------------


def test_source_kind_resolution_order(settings: Settings) -> None:
    assert make_ingestor(settings).source_kind == "fixture"
    assert make_ingestor(settings, ib_client=object()).source_kind == "ibkr"
    assert (
        make_ingestor(settings, feeds=("https://a.test/rss",), ib_client=object()).source_kind
        == "rss"
    )
    # Explicit feeds=() overrides configured settings feeds.
    configured = settings.model_copy(update={"news_rss_feeds": ("https://a.test/rss",)})
    assert make_ingestor(configured).source_kind == "rss"
    assert make_ingestor(configured, feeds=()).source_kind == "fixture"
    assert make_ingestor(configured).feeds == ("https://a.test/rss",)
    assert make_ingestor(configured).max_items == settings.news_max_items
    assert make_ingestor(configured).poll_interval_seconds == settings.news_poll_interval_seconds


def test_settings_feeds_are_used(settings: Settings) -> None:
    configured = settings.model_copy(
        update={"news_rss_feeds": ("https://feeds.example.test/rss",), "news_max_items": 1}
    )
    calls: list[str] = []

    async def fetcher(url: str) -> str:
        calls.append(url)
        return RSS2_XML

    ingestor = make_ingestor(configured, fetcher=fetcher)
    headlines = ingestor.fetch()
    assert calls == ["https://feeds.example.test/rss"]
    assert len(headlines) == 1  # settings.news_max_items applies as the default limit


def test_feed_validation(settings: Settings) -> None:
    with pytest.raises(ValueError):
        make_ingestor(settings, feeds=("ftp://not-http.test/rss",))
    with pytest.raises(TypeError):
        make_ingestor(settings, feeds=(42,))  # type: ignore[arg-type]
    ingestor = make_ingestor(settings, feeds=(" https://a.test/rss ", "", "https://a.test/rss"))
    assert ingestor.feeds == ("https://a.test/rss",)


# ---------------------------------------------------------------------------
# RSS 2.0 / Atom / RSS 1.0 parsing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rss2_parsing_and_classification(settings: Settings) -> None:
    ingestor = make_ingestor(
        settings, feeds=("https://feed.example.test/rss",), fetcher=await xml_fetcher(RSS2_XML)
    )
    headlines = await ingestor.fetch_async()
    assert len(headlines) == 4
    assert {headline.source for headline in headlines} == {"Macro Wire"}
    assert {headline.source_kind for headline in headlines} == {"rss"}

    fed = next(h for h in headlines if "higher-for-longer" in h.headline)
    assert fed.roots == tuple(sorted(DEFAULT_ROOT_KEYWORDS))  # macro -> all roots
    assert fed.severity == "warn"
    assert fed.url == "https://macro.example.test/fed-cpi"
    assert "Consumer prices" in fed.summary
    assert "<p>" not in fed.summary
    assert fed.published_at == datetime(2024, 10, 2, 12, 30, tzinfo=UTC)

    opec = next(h for h in headlines if "OPEC" in h.headline)
    assert opec.roots == ("CL", "MCL")
    assert opec.sentiment is not None and opec.sentiment > 0.0

    gold_war = next(h for h in headlines if "Gold" in h.headline)
    assert gold_war.severity == "critical"
    assert gold_war.sentiment is not None and gold_war.sentiment < 0.0

    tech = next(h for h in headlines if "Nasdaq" in h.headline)
    assert tech.roots == ("NQ",)
    assert tech.severity == "warn"

    # Newest-first ordering.
    timestamps = [headline.published_at for headline in headlines]
    assert timestamps == sorted(timestamps, reverse=True)


@pytest.mark.asyncio
async def test_atom_parsing(settings: Settings) -> None:
    ingestor = make_ingestor(
        settings, feeds=("https://atom.example.test/feed",), fetcher=await xml_fetcher(ATOM_XML)
    )
    headlines = await ingestor.fetch_async()
    assert len(headlines) == 3
    assert {headline.source for headline in headlines} == {"Atom Macro Desk"}

    auction = next(h for h in headlines if "10Y auction" in h.headline)
    assert auction.url == "https://atom.example.test/10y-auction"
    assert auction.summary == "The long end cheapened after a soft auction."
    assert auction.published_at == datetime(2024, 10, 2, 15, 0, tzinfo=UTC)
    assert auction.roots == ("ZN",)

    ecb = next(h for h in headlines if "Euro" in h.headline)
    assert ecb.published_at == datetime(2024, 10, 1, 6, 0, tzinfo=UTC)  # +02:00 -> UTC
    assert ecb.roots == ("6E",)

    undated = next(h for h in headlines if "No publication date" in h.headline)
    assert undated.published_at == FIXED_NOW


@pytest.mark.asyncio
async def test_rss1_rdf_parsing(settings: Settings) -> None:
    ingestor = make_ingestor(
        settings, feeds=("https://rss1.example.test/feed",), fetcher=await xml_fetcher(RSS1_XML)
    )
    headlines = await ingestor.fetch_async()
    assert len(headlines) == 1
    headline = headlines[0]
    assert headline.headline == "WASDE raises US corn carryout"
    assert headline.source == "RSS One Wire"
    assert headline.published_at == datetime(2024, 9, 28, 16, 0, tzinfo=UTC)
    assert headline.roots == tuple(sorted(DEFAULT_ROOT_KEYWORDS))  # WASDE -> all roots


@pytest.mark.asyncio
async def test_timestamp_normalisation_variants(settings: Settings) -> None:
    xml = """<?xml version="1.0"?><rss version="2.0"><channel><title>T</title>
    <item><title>rfc822 aware</title><pubDate>Wed, 02 Oct 2024 12:30:00 +0000</pubDate></item>
    <item><title>rfc822 naive</title><pubDate>Wed, 02 Oct 2024 12:30:00</pubDate></item>
    <item><title>iso zulu</title><pubDate>2024-10-02T12:30:00Z</pubDate></item>
    <item><title>iso offset</title><pubDate>2024-10-02T14:30:00+02:00</pubDate></item>
    <item><title>date only</title><pubDate>2024-10-02</pubDate></item>
    <item><title>garbage date</title><pubDate>not-a-date</pubDate></item>
    <item><title>missing date</title></item>
    </channel></rss>"""
    ingestor = make_ingestor(
        settings, feeds=("https://t.example.test/rss",), fetcher=await xml_fetcher(xml)
    )
    headlines = {headline.headline: headline.published_at for headline in ingestor.fetch()}
    expected = datetime(2024, 10, 2, 12, 30, tzinfo=UTC)
    assert headlines["rfc822 aware"] == expected
    assert headlines["rfc822 naive"] == expected  # naive strings interpreted as UTC
    assert headlines["iso zulu"] == expected
    assert headlines["iso offset"] == expected
    assert headlines["date only"] == datetime(2024, 10, 2, 0, 0, tzinfo=UTC)
    assert headlines["garbage date"] == FIXED_NOW
    assert headlines["missing date"] == FIXED_NOW


# ---------------------------------------------------------------------------
# Failure handling / fetcher injection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invalid_xml_falls_back_to_fixtures(
    settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    async def bad_fetcher(url: str) -> str:
        del url
        return "<not-really-xml"

    ingestor = make_ingestor(settings, feeds=("https://bad.example.test/rss",), fetcher=bad_fetcher)
    with caplog.at_level(logging.WARNING, logger="src.data.news_ingestor"):
        headlines = await ingestor.fetch_async()
    assert ingestor.source_kind == "rss"
    assert headlines
    assert {headline.source_kind for headline in headlines} == {"fixture"}
    assert any("invalid XML" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_empty_feed_falls_back_to_fixtures(settings: Settings) -> None:
    empty = '<?xml version="1.0"?><rss version="2.0"><channel><title>Empty</title></channel></rss>'
    ingestor = make_ingestor(
        settings,
        feeds=("https://empty.example.test/rss",),
        fetcher=await xml_fetcher(empty),
    )
    headlines = await ingestor.fetch_async(limit=3)
    assert len(headlines) == 3
    assert {headline.source_kind for headline in headlines} == {"fixture"}


@pytest.mark.asyncio
async def test_partial_feed_failure_keeps_successful_items(settings: Settings) -> None:
    calls: list[str] = []

    async def fetcher(url: str) -> str:
        calls.append(url)
        if "broken" in url:
            raise ConnectionError("feed unreachable")
        return RSS2_XML

    ingestor = make_ingestor(
        settings,
        feeds=("https://broken.example.test/rss", "https://good.example.test/rss"),
        fetcher=fetcher,
    )
    headlines = await ingestor.fetch_async()
    assert calls == [
        "https://broken.example.test/rss",
        "https://good.example.test/rss",
    ]
    assert len(headlines) == 4
    assert {headline.source_kind for headline in headlines} == {"rss"}


@pytest.mark.asyncio
async def test_invalid_fetcher_return_type_falls_back(settings: Settings) -> None:
    async def bytes_fetcher(url: str) -> Any:
        del url
        return b"<rss/>"

    ingestor = make_ingestor(
        settings, feeds=("https://bytes.example.test/rss",), fetcher=bytes_fetcher
    )
    headlines = await ingestor.fetch_async(limit=1)
    assert headlines[0].source_kind == "fixture"


@pytest.mark.asyncio
async def test_default_fetcher_imports_httpx_lazily(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    requested: list[str] = []

    class FakeResponse:
        text = RSS2_XML

        def raise_for_status(self) -> None:
            return None

    class FakeAsyncClient:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        async def __aenter__(self) -> FakeAsyncClient:
            return self

        async def __aexit__(self, *exc_info: object) -> bool:
            return False

        async def get(self, url: str) -> FakeResponse:
            requested.append(url)
            return FakeResponse()

    fake_httpx = SimpleNamespace(AsyncClient=FakeAsyncClient)
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)
    ingestor = make_ingestor(settings, feeds=("https://lazy.example.test/rss",))
    headlines = await ingestor.fetch_async(limit=2)
    assert requested == ["https://lazy.example.test/rss"]
    assert len(headlines) == 2
    assert {headline.source_kind for headline in headlines} == {"rss"}


@pytest.mark.asyncio
async def test_missing_httpx_falls_back_to_fixtures(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setitem(sys.modules, "httpx", None)
    ingestor = make_ingestor(settings, feeds=("https://httpx.example.test/rss",))
    with caplog.at_level(logging.WARNING, logger="src.data.news_ingestor"):
        headlines = await ingestor.fetch_async(limit=2)
    assert len(headlines) == 2
    assert {headline.source_kind for headline in headlines} == {"fixture"}
    assert any("httpx" in record.message.lower() for record in caplog.records)


# ---------------------------------------------------------------------------
# IBKR path
# ---------------------------------------------------------------------------


class FakeIBKRClient:
    """Duck-typed fake of the ib_async historical-news API."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.provider_calls = 0
        self.historical_calls: list[tuple[Any, ...]] = []

    async def reqNewsProvidersAsync(self) -> list[object]:  # noqa: N802
        self.provider_calls += 1
        if self.fail:
            raise RuntimeError("providers unavailable")
        return [
            SimpleNamespace(code="DJ-N", name="Dow Jones Newswires"),
            SimpleNamespace(code="BRFG", name="Briefing.com"),
        ]

    async def reqHistoricalNewsAsync(  # noqa: N802
        self,
        con_id: int,
        provider_code: str,
        start: datetime,
        end: datetime,
        total: int,
    ) -> object:
        self.historical_calls.append((con_id, provider_code, start, end, total))
        if self.fail:
            raise RuntimeError("historical news unavailable")
        return SimpleNamespace(
            articles=[
                SimpleNamespace(
                    headline="FOMC holds rates steady as inflation cools",
                    time=datetime(2024, 10, 2, 18, 0, tzinfo=UTC),
                    providerCode=provider_code,
                    articleId="news-1",
                ),
                SimpleNamespace(
                    headline="Crude oil slips on demand fears",
                    time="2024-10-02T15:00:00Z",
                    providerCode=provider_code,
                ),
                SimpleNamespace(headline="   ", time=FIXED_NOW, providerCode=provider_code),
            ]
        )


@pytest.mark.asyncio
async def test_ibkr_success_path(settings: Settings) -> None:
    client = FakeIBKRClient()
    ingestor = make_ingestor(settings, ib_client=client)
    assert ingestor.source_kind == "ibkr"
    headlines = await ingestor.fetch_async()
    assert len(headlines) == 4  # 2 providers x 2 usable articles
    assert {headline.source_kind for headline in headlines} == {"ibkr"}
    assert {headline.source for headline in headlines} == {
        "Dow Jones Newswires",
        "Briefing.com",
    }
    assert client.provider_calls == 1
    assert len(client.historical_calls) == 2
    for call in client.historical_calls:
        assert call[0] == 0  # broad-market conId
        assert call[3] == FIXED_NOW
    fed = next(h for h in headlines if "FOMC" in h.headline)
    assert fed.roots == tuple(sorted(DEFAULT_ROOT_KEYWORDS))
    assert fed.published_at == datetime(2024, 10, 2, 18, 0, tzinfo=UTC)


@pytest.mark.asyncio
async def test_ibkr_failure_falls_back_to_fixtures(
    settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    ingestor = make_ingestor(settings, ib_client=FakeIBKRClient(fail=True))
    with caplog.at_level(logging.WARNING, logger="src.data.news_ingestor"):
        headlines = await ingestor.fetch_async(limit=2)
    assert len(headlines) == 2
    assert {headline.source_kind for headline in headlines} == {"fixture"}
    assert any("IBKR" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_ibkr_without_methods_falls_back(settings: Settings) -> None:
    ingestor = make_ingestor(settings, ib_client=object())
    headlines = await ingestor.fetch_async(limit=1)
    assert headlines[0].source_kind == "fixture"


@pytest.mark.asyncio
async def test_ibkr_generic_news_method(settings: Settings) -> None:
    class GenericClient:
        async def get_news_async(self) -> list[dict[str, object]]:
            return [
                {
                    "headline": "Gold rallies on safe-haven demand",
                    "summary": "Bullion gains as real yields ease.",
                    "published_at": "2024-10-02T17:00:00Z",
                    "source": "IBKR",
                }
            ]

    ingestor = make_ingestor(settings, ib_client=GenericClient())
    headlines = await ingestor.fetch_async()
    assert len(headlines) == 1
    assert headlines[0].source_kind == "ibkr"
    assert headlines[0].roots == ("GC", "MGC")
    assert headlines[0].published_at == datetime(2024, 10, 2, 17, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Deterministic fixture path
# ---------------------------------------------------------------------------


def test_fixture_mode_is_deterministic(settings: Settings) -> None:
    first = make_ingestor(settings)
    second = make_ingestor(settings)
    assert first.source_kind == "fixture"
    headlines_a = first.fetch()
    headlines_b = second.fetch()
    assert headlines_a == headlines_b
    assert first.fetch() == headlines_a
    assert len(headlines_a) == len(headlines_b) == 8
    assert {headline.source_kind for headline in headlines_a} == {"fixture"}
    assert {headline.severity for headline in headlines_a} == {"info", "warn", "critical"}
    assert any(headline.roots == tuple(sorted(DEFAULT_ROOT_KEYWORDS)) for headline in headlines_a)
    assert [headline.published_at for headline in headlines_a] == sorted(
        (headline.published_at for headline in headlines_a), reverse=True
    )


def test_fixture_timestamps_follow_injected_clock(settings: Settings) -> None:
    baseline = make_ingestor(settings).fetch()
    shifted_now = FIXED_NOW + timedelta(hours=3)
    shifted = make_ingestor(settings, clock=lambda: shifted_now).fetch()
    assert len(shifted) == len(baseline)
    for original, moved in zip(baseline, shifted, strict=True):
        assert moved.published_at - original.published_at == timedelta(hours=3)


def test_injected_fixture_headlines(settings: Settings) -> None:
    custom = (
        NewsHeadline(
            headline="Custom crude headline",
            source="Test Desk",
            published_at=FIXED_NOW - timedelta(minutes=1),
            roots=("CL",),
            severity="info",
            sentiment=0.1,
            source_kind="rss",  # normalised to fixture
        ),
        NewsHeadline(
            headline="Custom rates headline",
            source="Test Desk",
            published_at=FIXED_NOW - timedelta(minutes=2),
            roots=("ZN",),
            severity="warn",
        ),
    )
    ingestor = make_ingestor(settings, fixture_headlines=custom)
    headlines = ingestor.fetch()
    assert len(headlines) == 2
    assert [headline.headline for headline in headlines] == [
        "Custom crude headline",
        "Custom rates headline",
    ]
    assert {headline.source_kind for headline in headlines} == {"fixture"}
    assert ingestor.fetch(limit=1)[0].headline == "Custom crude headline"
    assert ingestor.fetch(roots=["ZN"])[0].headline == "Custom rates headline"

    with pytest.raises(TypeError):
        make_ingestor(settings, fixture_headlines=(object(),))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        make_ingestor(settings, fixture_headlines=())


# ---------------------------------------------------------------------------
# Dedupe / sorting / limit / roots filtering
# ---------------------------------------------------------------------------


DUPLICATE_XML = """<?xml version="1.0"?><rss version="2.0"><channel><title>Dedupe Wire</title>
<item><title>Crude oil rallies on OPEC cuts</title><pubDate>Mon, 30 Sep 2024 10:00:00 GMT</pubDate></item>
<item><title>Crude oil rallies on OPEC cuts</title><pubDate>Wed, 02 Oct 2024 10:00:00 GMT</pubDate></item>
<item><title>Nasdaq rallies on earnings beats</title><pubDate>Wed, 02 Oct 2024 09:00:00 GMT</pubDate></item>
<item><title>Fed warns on inflation risk</title><pubDate>Tue, 01 Oct 2024 12:00:00 GMT</pubDate></item>
</channel></rss>"""


@pytest.mark.asyncio
async def test_dedupe_sort_limit_and_roots_filter(settings: Settings) -> None:
    ingestor = make_ingestor(
        settings,
        feeds=("https://dedupe.example.test/rss",),
        fetcher=await xml_fetcher(DUPLICATE_XML),
    )
    headlines = await ingestor.fetch_async()
    assert len(headlines) == 3  # duplicate (headline, source) collapsed
    crude = [headline for headline in headlines if "Crude" in headline.headline]
    assert len(crude) == 1
    assert crude[0].published_at == datetime(2024, 10, 2, 10, 0, tzinfo=UTC)  # newest wins

    limited = await ingestor.fetch_async(limit=2)
    assert len(limited) == 2
    assert limited[0].published_at >= limited[1].published_at

    energy_only = await ingestor.fetch_async(roots=["CL"])
    assert {headline.headline for headline in energy_only} == {
        "Crude oil rallies on OPEC cuts",
        "Fed warns on inflation risk",  # macro -> all roots / market-wide
    }
    nasdaq_only = await ingestor.fetch_async(roots=["NQ", "ES"])
    assert {headline.headline for headline in nasdaq_only} == {
        "Nasdaq rallies on earnings beats",
        "Fed warns on inflation risk",
    }
    # Empty roots sequence behaves like None (no filter).
    assert len(await ingestor.fetch_async(roots=())) == 3


def test_limit_validation(settings: Settings) -> None:
    ingestor = make_ingestor(settings)
    with pytest.raises(ValueError):
        ingestor.fetch(limit=0)
    with pytest.raises(TypeError):
        ingestor.fetch(limit=True)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ingestor.fetch(roots="CL")  # type: ignore[arg-type]
    assert len(ingestor.fetch(limit=3)) == 3


# ---------------------------------------------------------------------------
# Contract / serialisation
# ---------------------------------------------------------------------------


def test_to_dict_is_json_serialisable(settings: Settings) -> None:
    ingestor = make_ingestor(settings)
    headlines = ingestor.fetch(limit=2)
    payload = ingestor.to_dict(headlines)
    rendered = json.dumps(payload, sort_keys=True)
    assert len(payload) == 2
    decoded = json.loads(rendered)
    first = decoded[0]
    assert set(first) == {
        "headline",
        "source",
        "published_at",
        "summary",
        "url",
        "roots",
        "severity",
        "sentiment",
        "source_kind",
    }
    assert first["published_at"].endswith("+00:00")
    assert isinstance(first["roots"], list)
    assert first["source_kind"] == "fixture"
    assert NewsHeadline(**{**first, "published_at": headlines[0].published_at}) is not None


def test_news_headline_validation() -> None:
    aware = datetime(2024, 10, 2, 12, 0, tzinfo=UTC)
    with pytest.raises(ValueError):
        NewsHeadline(headline=" ", source="S", published_at=aware)
    with pytest.raises(ValueError):
        NewsHeadline(headline="H", source=" ", published_at=aware)
    with pytest.raises(ValueError):
        NewsHeadline(headline="H", source="S", published_at=datetime(2024, 10, 2, 12, 0))
    with pytest.raises(ValueError):
        NewsHeadline(
            headline="H",
            source="S",
            published_at=aware,
            severity="panic",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError):
        NewsHeadline(
            headline="H",
            source="S",
            published_at=aware,
            source_kind="api",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError):
        NewsHeadline(headline="H", source="S", published_at=aware, sentiment=1.5)
    with pytest.raises(TypeError):
        NewsHeadline(
            headline="H",
            source="S",
            published_at=aware,
            sentiment=True,  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError):
        NewsHeadline(
            headline="H",
            source="S",
            published_at=aware,
            roots=(1,),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError):
        NewsHeadline(headline="H", source="S", published_at=aware, roots=(" ",))

    headline = NewsHeadline(
        headline="  Gold rallies  ",
        source=" Desk ",
        published_at=aware,
        summary="  line one\n line two ",
        url="  https://x.test/a ",
        roots=("gc", "GC", "mgc"),
        severity="info",
        sentiment=-0.25,
        source_kind="fixture",
    )
    assert headline.headline == "Gold rallies"
    assert headline.source == "Desk"
    assert headline.summary == "line one line two"
    assert headline.url == "https://x.test/a"
    assert headline.roots == ("GC", "MGC")
    assert headline.to_dict()["published_at"] == aware.isoformat()
    assert "Gold rallies" in repr(headline)

    neutral = NewsHeadline(headline="H", source="S", published_at=aware)
    assert neutral.severity == "info"
    assert neutral.sentiment is None
    assert neutral.roots == ()
    assert neutral.url is None


def test_classification_lexicons_are_exported_and_sane() -> None:
    assert {"ES", "NQ", "YM", "RTY", "CL", "MCL", "GC", "MGC", "ZN", "ZT", "6E"} <= set(
        DEFAULT_ROOT_KEYWORDS
    )
    assert {"fomc", "cpi", "nfp", "wasde"} <= set(DEFAULT_MACRO_KEYWORDS)
    assert isinstance(make_ingestor.__name__, str)


# ---------------------------------------------------------------------------
# Async / sync equivalence and custom classification
# ---------------------------------------------------------------------------


def test_async_and_sync_paths_agree(settings: Settings) -> None:
    async def run() -> tuple[tuple[NewsHeadline, ...], tuple[NewsHeadline, ...]]:
        fetcher = await xml_fetcher(RSS2_XML)
        ingestor = make_ingestor(
            settings, feeds=("https://sync.example.test/rss",), fetcher=fetcher
        )
        return await ingestor.fetch_async(), ingestor.fetch()  # sync inside a loop

    async_headlines, sync_headlines = asyncio.run(run())
    assert async_headlines == sync_headlines


@pytest.mark.asyncio
async def test_sync_fetch_works_inside_running_loop(settings: Settings) -> None:
    fetcher = await xml_fetcher(RSS2_XML)
    ingestor = make_ingestor(settings, feeds=("https://loop.example.test/rss",), fetcher=fetcher)
    assert len(ingestor.fetch(limit=2)) == 2


@pytest.mark.asyncio
async def test_custom_root_keywords(settings: Settings) -> None:
    xml = """<?xml version="1.0"?><rss version="2.0"><channel><title>Custom</title>
    <item><title>Banana exports surge to a record high</title><pubDate>2024-10-02T10:00:00Z</pubDate></item>
    <item><title>FOMC minutes point to a longer hold</title><pubDate>2024-10-02T09:00:00Z</pubDate></item>
    </channel></rss>"""
    ingestor = make_ingestor(
        settings,
        feeds=("https://custom.example.test/rss",),
        fetcher=await xml_fetcher(xml),
        root_keywords={"MYROOT": ("banana",)},
    )
    headlines = {headline.headline: headline for headline in await ingestor.fetch_async()}
    assert headlines["Banana exports surge to a record high"].roots == ("MYROOT",)
    assert headlines["FOMC minutes point to a longer hold"].roots == ("MYROOT",)


def test_custom_root_keywords_validation(settings: Settings) -> None:
    with pytest.raises(ValueError):
        make_ingestor(settings, root_keywords={})
    with pytest.raises(ValueError):
        make_ingestor(settings, root_keywords={"ES": ()})
    with pytest.raises(ValueError):
        make_ingestor(settings, root_keywords={"es": ("a",), "ES": ("b",)})
    with pytest.raises(TypeError):
        make_ingestor(settings, root_keywords={"ES": "not-a-sequence"})  # type: ignore[dict-item]


def test_clock_validation(settings: Settings) -> None:
    ingestor = make_ingestor(settings, clock=lambda: "not-a-datetime")  # type: ignore[arg-type,return-value]
    with pytest.raises(TypeError):
        ingestor.fetch()


def test_zero_network_default_construction(settings: Settings) -> None:
    """Constructing/default-fixture fetching must never open a socket."""
    ingestor = NewsIngestor(settings)
    assert ingestor.source_kind == "fixture"
    headlines = ingestor.fetch()
    assert all(isinstance(headline, NewsHeadline) for headline in headlines)


def test_severity_and_source_kind_aliases() -> None:
    severity: Severity = "critical"
    source_kind: SourceKind = "ibkr"
    headline = NewsHeadline(
        headline="H",
        source="S",
        published_at=datetime(2024, 10, 2, tzinfo=UTC),
        severity=severity,
        source_kind=source_kind,
    )
    assert headline.severity == "critical"
    assert headline.source_kind == "ibkr"
