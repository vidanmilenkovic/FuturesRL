"""IBKR front-month futures contract resolution and qualification.

IBKR cannot subscribe to tick-by-tick / market-by-price data (or hash an order
key) with a *bare* ``ib_async.Future``: the contract must be qualified and carry a
positive ``conId``. A bare ``Future(symbol='ES')`` also has no contract month, so
IBKR may resolve a far/ambiguous series, and every reconnect re-resolves it.

This module is the single place that turns a FuturesRL root into the **active
front-month** IBKR contract:

1. derive the explicit contract month (``YYYYMM``) from the instrument master's
   roll rules (``roll.contract_months`` / ``roll.days_before_fnd``), rolling early
   when the reference timestamp is already inside the roll window;
2. build an explicit ``Future(symbol=root, lastTradeDateOrContractMonth=YYYYMM,
   exchange=spec.ibkr_exchange, currency=spec.currency, includeExpired=False)``;
3. resolve it through ``reqContractDetailsAsync`` (preferred, gives the
   exchange-listed contract) or ``qualifyContractsAsync``/``qualifyContracts``;
4. verify ``conId > 0`` and return the qualified contract together with its
   month/conId metadata;
5. cache the result per ``(root, contract_month)`` on a supplied cache dict (or a
   process-wide default), so reconnects and repeated subscriptions never
   re-qualify the same front month.

Everything is duck-typed and injectable: fake clients only need
``reqContractDetailsAsync``/``qualifyContractsAsync``/``qualifyContracts`` and a
``raw_contract_factory``; ``ib_async`` is imported lazily only when the default
contract builder is used, so importing this module never requires the live SDK.
"""

from __future__ import annotations

import inspect
import logging
import threading
from collections.abc import Callable, MutableMapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Protocol, cast

from configs.instruments import InstrumentsConfig, InstrumentSpec, load_instruments

from src.core.types import InstrumentId, ensure_utc
from src.data.contracts import next_contract_expiration

__all__ = [
    "FrontFuture",
    "FrontFutureCache",
    "build_front_contract",
    "clear_front_future_cache",
    "front_contract_month",
    "qualify_front_future",
]

logger = logging.getLogger(__name__)

#: Default delivery cycle when a root is not declared in the instrument master.
_DEFAULT_CONTRACT_MONTHS: tuple[int, ...] = (3, 6, 9, 12)
#: Default roll buffer (calendar days before expiry) when the root is unknown.
_DEFAULT_ROLL_DAYS: int = 5
#: Default fallback exchange/currency for undeclared roots.
_DEFAULT_EXCHANGE: str = "CME"
_DEFAULT_CURRENCY: str = "USD"

#: Process-wide front-future cache used when the caller does not supply one.
_DEFAULT_CACHE: dict[tuple[str, str], FrontFuture] = {}
_CACHE_LOCK = threading.RLock()


@dataclass(frozen=True, slots=True)
class FrontFuture:
    """One resolved front-month IBKR contract with its conId metadata."""

    root: InstrumentId
    contract: object
    con_id: int
    contract_month: str
    expiration: date
    exchange: str
    qualified: bool = True

    @property
    def has_con_id(self) -> bool:
        """Whether the contract was qualified to a positive ``conId``."""
        return self.con_id > 0

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly diagnostics for telemetry/API/runner status."""
        return {
            "root": self.root,
            "con_id": self.con_id,
            "contract_month": self.contract_month,
            "expiration": self.expiration.isoformat(),
            "exchange": self.exchange,
            "qualified": self.qualified,
        }


#: Cache type shared by streamers/fetchers/connection managers.
type FrontFutureCache = MutableMapping[tuple[str, str], FrontFuture]


class _ContractResolutionClient(Protocol):
    """The duck-typed IBKR client calls used for front-month resolution."""

    async def reqContractDetailsAsync(self, contract: object) -> object:  # noqa: N802
        """Return contract-detail objects for ``contract``."""
        ...

    async def qualifyContractsAsync(self, *contracts: object) -> object:  # noqa: N802
        """Return qualified contracts (with ``conId``) for ``contracts``."""
        ...

    def qualifyContracts(self, *contracts: object) -> object:  # noqa: N802
        """Synchronous qualification fallback."""
        ...


async def _maybe_await(value: object) -> object:
    """Await ``value`` when it is awaitable (async/sync duck-typed clients)."""
    if inspect.isawaitable(value):
        return await value
    return value


def _spec_or_none(
    root_or_spec: InstrumentId | InstrumentSpec,
    instrument_master: InstrumentsConfig | None,
) -> InstrumentSpec | None:
    """Resolve an ``InstrumentSpec`` from a spec instance, a root symbol or the master."""
    if isinstance(root_or_spec, InstrumentSpec):
        return root_or_spec
    master = instrument_master if instrument_master is not None else load_instruments()
    normalised = root_or_spec.strip().upper()
    return master.get(normalised) if master.has(normalised) else None


def _root_symbol(root_or_spec: InstrumentId | InstrumentSpec) -> str:
    """Normalised root symbol for a spec or root string."""
    if isinstance(root_or_spec, InstrumentSpec):
        return root_or_spec.root
    return root_or_spec.strip().upper()


def _reference_date(reference: datetime | None) -> datetime:
    """Validate/normalise the reference timestamp used for roll resolution."""
    if reference is None:
        return datetime.now(UTC)
    return ensure_utc(reference)


def front_contract_month(
    root_or_spec: InstrumentId | InstrumentSpec,
    *,
    reference: datetime | None = None,
    instrument_master: InstrumentsConfig | None = None,
) -> tuple[str, date]:
    """Return the active front contract month ``("YYYYMM", representative_expiry)``.

    The roll follows the instrument master: the nearest declared delivery month is
    the front, unless ``reference`` is already inside the ``days_before_fnd`` roll
    window, in which case the following delivery month becomes the front (the
    same convention used by the continuous roll engine).
    """
    spec = _spec_or_none(root_or_spec, instrument_master)
    months = tuple(spec.roll.contract_months) if spec is not None else _DEFAULT_CONTRACT_MONTHS
    roll_days = spec.roll.days_before_fnd if spec is not None else _DEFAULT_ROLL_DAYS
    reference_date = _reference_date(reference)
    expiration = next_contract_expiration(months, reference_date)
    roll_start = expiration - timedelta(days=roll_days)
    if reference_date.date() >= roll_start:
        expiration = next_contract_expiration(months, expiration)
    return (f"{expiration.year:04d}{expiration.month:02d}", expiration)


def build_front_contract(
    root_or_spec: InstrumentId | InstrumentSpec,
    *,
    reference: datetime | None = None,
    instrument_master: InstrumentsConfig | None = None,
    raw_contract_factory: Callable[[InstrumentId], object] | None = None,
) -> tuple[object, str, date]:
    """Build the explicit front-month IBKR contract for ``root_or_spec``.

    Returns ``(contract, contract_month, representative_expiration)``. The default
    builder lazily imports ``ib_async`` and always passes an explicit
    ``lastTradeDateOrContractMonth`` (never a bare ``Future``); tests/integrations
    can inject ``raw_contract_factory`` (root -> contract) instead.
    """
    root = _root_symbol(root_or_spec)
    spec = _spec_or_none(root_or_spec, instrument_master)
    contract_month, expiration = front_contract_month(
        root_or_spec, reference=reference, instrument_master=instrument_master
    )
    if raw_contract_factory is not None:
        return (raw_contract_factory(root), contract_month, expiration)
    try:
        import ib_async
    except ImportError as exc:  # pragma: no cover - SDK installed in CI/live
        raise RuntimeError(
            "The ib_async SDK is not installed; install the optional live extra "
            "(`pip install 'futuresrl[live]'` or `pip install ib_async`) or inject "
            "a raw_contract_factory."
        ) from exc
    exchange = spec.ibkr_exchange if spec is not None else _DEFAULT_EXCHANGE
    currency = spec.currency if spec is not None else _DEFAULT_CURRENCY
    ibkr_symbol = spec.ibkr_future_symbol if spec is not None else root
    contract = ib_async.Future(
        symbol=ibkr_symbol,
        lastTradeDateOrContractMonth=contract_month,
        exchange=exchange,
        currency=currency,
        includeExpired=False,
    )
    return (contract, contract_month, expiration)


def _has_con_id(contract: object) -> bool:
    """Whether ``contract`` carries a positive IBKR contract id."""
    con_id = getattr(contract, "conId", None)
    if con_id is None:
        con_id = getattr(contract, "con_id", None)
    return isinstance(con_id, int) and not isinstance(con_id, bool) and con_id > 0


def _contract_from(item: object) -> object:
    """Extract the contract object from a ContractDetails item (or the item itself)."""
    detail_contract = getattr(item, "contract", None)
    return detail_contract if detail_contract is not None else item


def _first_contract_with_con_id(result: object) -> object | None:
    """First non-null item of a qualification/details result carrying a ``conId``."""
    if result is None:
        return None
    items = tuple(result) if isinstance(result, (list, tuple)) else (result,)
    for item in items:
        if item is None:
            continue
        contract = _contract_from(item)
        if _has_con_id(contract):
            return contract
    return None


async def _resolve_qualified_contract(client: object, raw_contract: object) -> object | None:
    """Resolve ``raw_contract`` via ``reqContractDetailsAsync`` or qualification APIs."""
    details_method = getattr(client, "reqContractDetailsAsync", None)
    if callable(details_method):
        try:
            details = await _maybe_await(details_method(raw_contract))
        except Exception as exc:
            logger.warning("IBKR reqContractDetailsAsync failed: %s", exc)
        else:
            resolved = _first_contract_with_con_id(details)
            if resolved is not None:
                return resolved
    async_qualify = getattr(client, "qualifyContractsAsync", None)
    if callable(async_qualify):
        try:
            qualified = await _maybe_await(async_qualify(raw_contract))
        except Exception as exc:
            logger.warning("IBKR qualifyContractsAsync failed: %s", exc)
        else:
            resolved = _first_contract_with_con_id(qualified)
            if resolved is not None:
                return resolved
    sync_qualify = getattr(client, "qualifyContracts", None)
    if callable(sync_qualify):
        try:
            qualified = await _maybe_await(sync_qualify(raw_contract))
        except Exception as exc:
            logger.warning("IBKR qualifyContracts failed: %s", exc)
        else:
            resolved = _first_contract_with_con_id(qualified)
            if resolved is not None:
                return resolved
    return None


def _client_has_qualification(client: object) -> bool:
    """Whether ``client`` exposes any contract-resolution API at all."""
    return any(
        callable(getattr(client, name, None))
        for name in ("reqContractDetailsAsync", "qualifyContractsAsync", "qualifyContracts")
    )


async def qualify_front_future(
    client: object,
    root_or_spec: InstrumentId | InstrumentSpec,
    *,
    reference: datetime | None = None,
    cache: FrontFutureCache | None = None,
    instrument_master: InstrumentsConfig | None = None,
    raw_contract_factory: Callable[[InstrumentId], object] | None = None,
) -> FrontFuture:
    """Resolve the qualified front-month IBKR contract for one root.

    Parameters
    ----------
    client:
        Duck-typed IBKR client (``ib_async.IB`` or a fake). Qualification is
        attempted with ``reqContractDetailsAsync`` first, then
        ``qualifyContractsAsync``/``qualifyContracts``.
    root_or_spec:
        Futures root symbol or its :class:`~configs.instruments.InstrumentSpec`.
    reference:
        Point-in-time used for roll resolution (tests inject a fixed clock).
    cache:
        Optional ``(root, contract_month) -> FrontFuture`` mapping owned by the
        caller (streamer/connection manager). A process-wide cache is used when
        omitted.
    instrument_master:
        Instrument master used for roll rules / exchange / currency resolution.
    raw_contract_factory:
        Optional legacy root -> contract factory (injected fakes/tests); the
        explicit front-month contract is built otherwise.

    Raises
    ------
    RuntimeError
        The client supports qualification but returned no usable ``conId``; the
        message names the root, exchange and contract month.
    """
    root = _root_symbol(root_or_spec)
    raw_contract, contract_month, expiration = build_front_contract(
        root_or_spec,
        reference=reference,
        instrument_master=instrument_master,
        raw_contract_factory=raw_contract_factory,
    )
    spec = _spec_or_none(root_or_spec, instrument_master)
    exchange = spec.ibkr_exchange if spec is not None else _DEFAULT_EXCHANGE
    cache_store: FrontFutureCache = (
        cache if cache is not None else cast(FrontFutureCache, _DEFAULT_CACHE)
    )
    key = (root, contract_month)
    with _CACHE_LOCK:
        cached = cache_store.get(key)
    if cached is not None and cached.qualified:
        return cached
    if not _client_has_qualification(client):
        # Duck-typed legacy/fake clients: keep the raw contract so offline harnesses
        # and old adapters remain usable (no conId verification possible).
        logger.debug(
            "IBKR client exposes no qualification API; using raw front contract for "
            "root %r month %s",
            root,
            contract_month,
        )
        resolved = FrontFuture(
            root=root,
            contract=raw_contract,
            con_id=0,
            contract_month=contract_month,
            expiration=expiration,
            exchange=exchange,
            qualified=False,
        )
    else:
        qualified = await _resolve_qualified_contract(client, raw_contract)
        if qualified is None or not _has_con_id(qualified):
            raise RuntimeError(
                f"IBKR could not resolve a conId for root {root!r} on {exchange} for "
                f"contract month {contract_month}; check the contract month/exchange and "
                "TWS market-data subscriptions"
            )
        con_id = getattr(qualified, "conId", getattr(qualified, "con_id", 0))
        resolved = FrontFuture(
            root=root,
            contract=qualified,
            con_id=int(con_id),
            contract_month=contract_month,
            expiration=expiration,
            exchange=exchange,
            qualified=True,
        )
    with _CACHE_LOCK:
        cache_store[key] = resolved
    return resolved


def clear_front_future_cache(cache: FrontFutureCache | None = None) -> None:
    """Clear a supplied front-future cache (or the process-wide default)."""
    with _CACHE_LOCK:
        target = cache if cache is not None else _DEFAULT_CACHE
        target.clear()
