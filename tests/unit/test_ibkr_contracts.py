"""Unit tests for the shared IBKR front-month contract qualification helper.

No live TWS/Gateway: every client is an in-memory duck-typed fake. The tests pin
the contract-month derivation from the instrument roll rules, the explicit
``lastTradeDateOrContractMonth`` contract construction, conId verification, the
reqContractDetailsAsync -> qualifyContractsAsync fallbacks, the per-(root, month)
cache and the actionable failure messages.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from configs.instruments import InstrumentsConfig, load_instruments
from src.data.ibkr_contracts import (
    FrontFuture,
    build_front_contract,
    clear_front_future_cache,
    front_contract_month,
    qualify_front_future,
)

REFERENCE = datetime(2024, 1, 2, 21, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def master() -> InstrumentsConfig:
    return load_instruments()


class FakeContractClient:
    """Duck-typed IBKR client recording contract-resolution calls."""

    def __init__(
        self,
        *,
        details: list[Any] | None = None,
        qualified: list[Any] | None = None,
        details_error: Exception | None = None,
        qualify_error: Exception | None = None,
    ) -> None:
        self.details = details
        self.qualified = qualified
        self.details_error = details_error
        self.qualify_error = qualify_error
        self.details_calls: list[object] = []
        self.qualify_calls: list[object] = []

    async def reqContractDetailsAsync(self, contract: object) -> object:  # noqa: N802
        self.details_calls.append(contract)
        if self.details_error is not None:
            raise self.details_error
        return self.details

    async def qualifyContractsAsync(self, *contracts: object) -> object:  # noqa: N802
        self.qualify_calls.extend(contracts)
        if self.qualify_error is not None:
            raise self.qualify_error
        return self.qualified


class DetailsOnlyFake(FakeContractClient):
    """Client exposing reqContractDetailsAsync but no qualifyContractsAsync."""

    qualifyContractsAsync = None  # type: ignore[assignment]  # noqa: N815


class QualifyOnlyFake(FakeContractClient):
    """Client exposing only qualifyContractsAsync."""

    reqContractDetailsAsync = None  # type: ignore[assignment]  # noqa: N815


class SyncOnlyFake(QualifyOnlyFake):
    """Client exposing only the synchronous qualifyContracts fallback."""

    qualifyContractsAsync = None  # type: ignore[assignment]  # noqa: N815

    def qualifyContracts(self, *contracts: object) -> list[object]:  # noqa: N802
        self.qualify_calls.extend(contracts)
        if self.qualify_error is not None:
            raise self.qualify_error
        return list(self.qualified or [])


def contract(con_id: int, *, symbol: str = "ES") -> SimpleNamespace:
    return SimpleNamespace(symbol=symbol, conId=con_id, exchange="CME")


def detail(con_id: int, *, symbol: str = "ES") -> SimpleNamespace:
    return SimpleNamespace(contract=contract(con_id, symbol=symbol))


# ---------------------------------------------------------------------------
# Contract-month derivation from the roll rules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("root", "month"),
    [
        ("ES", "202403"),
        ("NQ", "202403"),
        ("ZN", "202403"),
        ("6E", "202403"),
        ("CL", "202401"),
        ("GC", "202402"),
        ("ZC", "202403"),
        ("SR3", "202403"),
        ("MBT", "202401"),
    ],
)
def test_front_contract_month_uses_delivery_cycle(
    master: InstrumentsConfig, root: str, month: str
) -> None:
    resolved_month, expiration = front_contract_month(
        root, reference=REFERENCE, instrument_master=master
    )
    assert resolved_month == month
    assert f"{expiration.year:04d}{expiration.month:02d}" == month
    assert expiration > REFERENCE.date()


def test_roll_window_moves_to_next_delivery_month(master: InstrumentsConfig) -> None:
    # ES rolls 5 days before expiry: 2024-03-10 is inside the window -> June.
    assert (
        front_contract_month(
            "ES", reference=datetime(2024, 3, 8, tzinfo=UTC), instrument_master=master
        )[0]
        == "202403"
    )
    assert (
        front_contract_month(
            "ES", reference=datetime(2024, 3, 10, tzinfo=UTC), instrument_master=master
        )[0]
        == "202406"
    )
    assert (
        front_contract_month(
            "ES", reference=datetime(2024, 3, 12, tzinfo=UTC), instrument_master=master
        )[0]
        == "202406"
    )
    # CL rolls 5 days before its 2024-01-19 expiry -> 2024-01-14 flips to Feb.
    assert (
        front_contract_month(
            "CL", reference=datetime(2024, 1, 10, tzinfo=UTC), instrument_master=master
        )[0]
        == "202401"
    )
    assert (
        front_contract_month(
            "CL", reference=datetime(2024, 1, 14, tzinfo=UTC), instrument_master=master
        )[0]
        == "202402"
    )


def test_front_contract_month_rejects_naive_reference(master: InstrumentsConfig) -> None:
    with pytest.raises(ValueError):
        front_contract_month("ES", reference=datetime(2024, 1, 2), instrument_master=master)


def test_front_contract_month_accepts_spec_instance(master: InstrumentsConfig) -> None:
    spec = master.get("NQ")
    month, expiration = front_contract_month(spec, reference=REFERENCE)
    assert month == "202403"
    assert expiration.year == 2024


# ---------------------------------------------------------------------------
# Explicit contract construction (never a bare Future)
# ---------------------------------------------------------------------------


def test_build_front_contract_is_explicit_and_never_bare(
    master: InstrumentsConfig,
) -> None:
    contract_obj, month, expiration = build_front_contract(
        "ES", reference=REFERENCE, instrument_master=master
    )
    assert month == "202403"
    assert expiration.year == 2024
    assert contract_obj.symbol == "ES"
    assert contract_obj.lastTradeDateOrContractMonth == "202403"
    assert contract_obj.exchange == "CME"
    assert contract_obj.currency == "USD"
    assert contract_obj.includeExpired is False


def test_build_front_contract_supports_raw_factory() -> None:
    contract_obj, month, _expiration = build_front_contract(
        "CL",
        reference=REFERENCE,
        raw_contract_factory=lambda root: SimpleNamespace(symbol=root, localSymbol=f"{root}F4"),
    )
    assert month == "202401"
    assert contract_obj.symbol == "CL"
    assert contract_obj.localSymbol == "CLF4"


def test_build_front_contract_unknown_root_uses_defaults() -> None:
    contract_obj, month, _expiration = build_front_contract(
        "ZZZ", reference=REFERENCE, raw_contract_factory=lambda root: SimpleNamespace(symbol=root)
    )
    assert month == "202403"  # default quarterly cycle
    assert contract_obj.symbol == "ZZZ"


# ---------------------------------------------------------------------------
# Qualification, fallbacks, cache and verification
# ---------------------------------------------------------------------------


async def test_qualify_via_contract_details_prefers_exchange_contract(
    master: InstrumentsConfig,
) -> None:
    client = DetailsOnlyFake(details=[None, detail(123456)])
    cache: dict[tuple[str, str], FrontFuture] = {}
    front = await qualify_front_future(
        client, "ES", reference=REFERENCE, cache=cache, instrument_master=master
    )
    assert isinstance(front, FrontFuture)
    assert front.con_id == 123456
    assert front.contract_month == "202403"
    assert front.qualified is True
    assert front.has_con_id is True
    assert front.exchange == "CME"
    assert client.details_calls and client.details_calls[0].symbol == "ES"
    assert cache == {("ES", "202403"): front}


async def test_qualify_falls_back_to_qualify_contracts_async(
    master: InstrumentsConfig,
) -> None:
    client = QualifyOnlyFake(qualified=[None, contract(777)])
    front = await qualify_front_future(
        client, "CL", reference=REFERENCE, cache={}, instrument_master=master
    )
    assert front.con_id == 777
    assert front.contract_month == "202401"
    assert client.qualify_calls


async def test_qualify_falls_back_to_sync_qualification(
    master: InstrumentsConfig,
) -> None:
    client = SyncOnlyFake(qualified=[contract(888, symbol="NQ")])
    front = await qualify_front_future(
        client, "NQ", reference=REFERENCE, cache={}, instrument_master=master
    )
    assert front.con_id == 888
    assert front.contract_month == "202403"


async def test_details_failure_falls_back_to_qualification(
    master: InstrumentsConfig,
) -> None:
    client = FakeContractClient(
        details_error=RuntimeError("details unavailable"), qualified=[contract(999)]
    )
    front = await qualify_front_future(
        client, "ES", reference=REFERENCE, cache={}, instrument_master=master
    )
    assert front.con_id == 999
    assert client.details_calls and client.qualify_calls


async def test_missing_con_id_raises_actionable_error(master: InstrumentsConfig) -> None:
    client = QualifyOnlyFake(qualified=[contract(0), None])
    with pytest.raises(RuntimeError) as excinfo:
        await qualify_front_future(
            client, "ES", reference=REFERENCE, cache={}, instrument_master=master
        )
    message = str(excinfo.value)
    assert "could not resolve a conId for root 'ES' on CME" in message
    assert "contract month 202403" in message
    assert "TWS market-data subscriptions" in message


async def test_qualification_exception_surfaces_as_actionable_error(
    master: InstrumentsConfig,
) -> None:
    client = QualifyOnlyFake(qualify_error=RuntimeError("TWS timeout"))
    with pytest.raises(RuntimeError, match="could not resolve a conId"):
        await qualify_front_future(
            client, "ES", reference=REFERENCE, cache={}, instrument_master=master
        )


async def test_cache_hit_avoids_requalification(master: InstrumentsConfig) -> None:
    client = QualifyOnlyFake(qualified=[contract(42)])
    cache: dict[tuple[str, str], FrontFuture] = {}
    first = await qualify_front_future(
        client, "ES", reference=REFERENCE, cache=cache, instrument_master=master
    )
    second = await qualify_front_future(
        client, "ES", reference=REFERENCE, cache=cache, instrument_master=master
    )
    assert second is first
    assert len(client.qualify_calls) == 1
    # A different contract month gets its own cache entry (rolls re-qualify).
    next_month = await qualify_front_future(
        client,
        "ES",
        reference=datetime(2024, 6, 25, tzinfo=UTC),
        cache=cache,
        instrument_master=master,
    )
    assert next_month.contract_month == "202409"
    assert len(client.qualify_calls) == 2
    assert set(cache) == {("ES", "202403"), ("ES", "202409")}


async def test_clear_front_future_cache(master: InstrumentsConfig) -> None:
    client = QualifyOnlyFake(qualified=[contract(42)])
    cache: dict[tuple[str, str], FrontFuture] = {}
    await qualify_front_future(
        client, "ES", reference=REFERENCE, cache=cache, instrument_master=master
    )
    assert cache
    clear_front_future_cache(cache)
    assert cache == {}
    # The process-wide default cache is also clearable.
    clear_front_future_cache()


async def test_legacy_client_without_qualification_keeps_raw_contract(
    master: InstrumentsConfig,
) -> None:
    front = await qualify_front_future(
        SimpleNamespace(), "ES", reference=REFERENCE, cache={}, instrument_master=master
    )
    assert front.qualified is False
    assert front.con_id == 0
    assert front.contract_month == "202403"
    assert front.contract.lastTradeDateOrContractMonth == "202403"


async def test_front_future_to_dict_is_json_serialisable(master: InstrumentsConfig) -> None:
    client = QualifyOnlyFake(qualified=[contract(42)])
    front = await qualify_front_future(
        client, "CL", reference=REFERENCE, cache={}, instrument_master=master
    )
    payload = front.to_dict()
    assert set(payload) == {
        "root",
        "con_id",
        "contract_month",
        "expiration",
        "exchange",
        "qualified",
    }
    assert payload["con_id"] == 42
    assert payload["expiration"] == "2024-01-19"
    assert json.dumps(payload)


def test_front_future_cache_alias_is_typing_only() -> None:
    # FrontFutureCache is a public type alias usable by streamers/connection managers.
    from src.data.ibkr_contracts import FrontFutureCache

    cache: FrontFutureCache = {}
    assert isinstance(cache, dict)
