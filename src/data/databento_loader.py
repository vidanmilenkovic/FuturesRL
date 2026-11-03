"""Databento historical ingestion and the deterministic offline bar source.

Two data sources share the canonical :mod:`src.data.schema` contract:

:class:`DatabentoLoader`
    Historical ingestion from the paid Databento API (``GLBX.MDP3`` for
    CME/CBOT/NYMEX/COMEX). The ``databento`` SDK is imported **lazily** inside
    the methods that need it, and a missing SDK or API key raises an actionable
    :class:`RuntimeError` — the loader never fabricates vendor data. Supported
    schemas: ``ohlcv-1m``, ``ohlcv-1h``, ``ohlcv-1d``, ``tbbo``, ``mbp-1``,
    ``mbp-10`` and ``definition``.

:class:`OfflineBarSource`
    A fully deterministic, seeded synthetic GBM source implementing
    :class:`src.core.interfaces.HistoricalDataSource`. It exists so backtests,
    tests and CI can run the *entire* pipeline with no network, no credentials
    and reproducible numbers. It generates a deterministic, roll-aware
    front (F1) **and** synthetic second-month (F2) bar pair per root/timestamp
    from :class:`~src.data.term_structure.SyntheticTermStructure` (contango for
    equity/rates/metals, backwardation for energy, FX near flat, with a slow
    calendar regime cycle), so ``term_slope``/``carry_annualized`` are finite,
    non-zero and regime-consistent instead of NaN/zero-filled. It is **clearly
    synthetic**: every contract id is suffixed ``_SYN<yyyymmdd>``,
    ``is_synthetic`` is ``True`` and the class must never be used to impersonate
    Databento data in paper/live mode (the live loop consumes
    :class:`~src.data.ibkr_data.IBKRMarketDataStreamer`). Pass
    ``term_structure=False`` for the legacy single-series behaviour.

Returned frames are always canonical: bars pass
:func:`src.data.schema.ensure_bars_schema`, quotes pass
``ensure_quote_schema``, and definitions pass ``ensure_definitions_schema``
before they are persisted by :class:`~src.data.storage.ParquetRepository`.

Contract identity resolution
----------------------------
A Databento frame carries an opaque ``instrument_id`` plus the raw CME symbol
(``ESM4``). The loader first tries the stored ``definition`` schema
(``instrument_id -> root/expiration``) and only then falls back to parsing the
raw symbol. Continuous native symbols (``ES.v.0``) *require* a stored
definition; otherwise the loader raises an actionable error instead of guessing
an expiry.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, cast

import numpy as np
import polars as pl
from configs.instruments import InstrumentsConfig, InstrumentSpec, load_instruments
from configs.settings import Settings, get_settings
from numpy.typing import NDArray

from src.core.types import InstrumentId
from src.data import schema as s
from src.data.contracts import (
    native_continuous_symbol,
    next_contract_expiration,
    parse_contract_symbol,
)
from src.data.storage import ParquetRepository, WriteMode
from src.data.term_structure import SyntheticTermStructure

#: Deterministic volume / open-interest scale of the synthetic second month.
_SECOND_MONTH_VOLUME_SCALE = 0.40
_SECOND_MONTH_OI_SCALE = 0.45
#: Calendar buffer (days) added when building the synthetic expiry chain.
_EXPIRY_CHAIN_BUFFER_DAYS = 400
#: Minimum definition horizon (days) so generated expiries always resolve.
_DEFINITION_HORIZON_PERIODS = 512

__all__ = [
    "DatabentoIngestError",
    "DatabentoLoader",
    "OfflineBarSource",
    "synthetic_reference_price",
]

#: Reference prices used by the deterministic synthetic source (every root).
SYNTHETIC_REFERENCE_PRICES: dict[str, float] = {
    # equity index
    "ES": 4_800.0,
    "MES": 4_800.0,
    "NQ": 16_800.0,
    "MNQ": 16_800.0,
    "YM": 38_000.0,
    "MYM": 38_000.0,
    "RTY": 2_000.0,
    "M2K": 2_000.0,
    "NKD": 38_000.0,
    # rates
    "ZT": 102.75,
    "ZF": 107.5,
    "ZN": 111.5,
    "TN": 114.5,
    "ZB": 120.0,
    "UB": 130.0,
    "SR3": 95.0,
    # energy
    "CL": 72.5,
    "MCL": 72.5,
    "NG": 2.80,
    "MNG": 2.80,
    "RB": 2.40,
    "HO": 2.60,
    # metals
    "GC": 2_050.0,
    "MGC": 2_050.0,
    "SI": 25.0,
    "SIL": 25.0,
    "HG": 4.10,
    "MHG": 4.10,
    "PL": 950.0,
    "PA": 1_000.0,
    # fx
    "6E": 1.095,
    "6J": 0.0068,
    "6B": 1.27,
    "6A": 0.66,
    "6C": 0.74,
    "6S": 1.13,
    # agriculture
    "ZC": 480.0,
    "ZS": 1_200.0,
    "ZW": 600.0,
    "ZM": 350.0,
    "ZL": 45.0,
    "LE": 180.0,
    "HE": 85.0,
    # crypto
    "MBT": 65_000.0,
    "MET": 3_200.0,
}

#: Annualised volatility per asset class for the synthetic GBM source.
SYNTHETIC_ANNUAL_VOL: dict[str, float] = {
    "equity_index": 0.18,
    "rates": 0.06,
    "energy": 0.35,
    "metals": 0.15,
    "fx": 0.08,
    "agriculture": 0.25,
    "crypto": 0.60,
    "unknown": 0.15,
}

#: Optional per-root volatility overrides for the synthetic GBM source (the
#: expanded universe has very different risk profiles inside one asset class,
#: e.g. SOFR vs Ultra T-Bond, Micro Bitcoin vs Micro Ether, natural gas vs WTI).
SYNTHETIC_ROOT_ANNUAL_VOL: dict[str, float] = {
    "NKD": 0.20,
    "SR3": 0.02,
    "ZT": 0.03,
    "ZF": 0.05,
    "TN": 0.08,
    "ZB": 0.10,
    "UB": 0.11,
    "NG": 0.50,
    "MNG": 0.50,
    "RB": 0.40,
    "HO": 0.38,
    "SI": 0.25,
    "SIL": 0.25,
    "HG": 0.22,
    "MHG": 0.22,
    "PL": 0.20,
    "PA": 0.28,
    "6J": 0.10,
    "6B": 0.09,
    "6A": 0.10,
    "6C": 0.07,
    "6S": 0.08,
    "ZC": 0.24,
    "ZS": 0.22,
    "ZW": 0.26,
    "ZM": 0.24,
    "ZL": 0.22,
    "LE": 0.15,
    "HE": 0.22,
    "MBT": 0.65,
    "MET": 0.70,
}

#: Relative synthetic volume / open-interest scale per root (1.0 = legacy
#: baseline 250k volume, 1.5M OI). The first 13 roots keep 1.0 so historical
#: offline price/volume paths and strategy universes are byte-compatible.
SYNTHETIC_VOLUME_SCALE: dict[str, float] = {
    "NKD": 0.24,
    "MYM": 0.32,
    "M2K": 0.48,
    "ZF": 1.60,
    "TN": 1.00,
    "ZB": 0.80,
    "UB": 0.50,
    "SR3": 2.40,
    "NG": 0.70,
    "MNG": 1.20,
    "RB": 0.50,
    "HO": 0.55,
    "SI": 0.36,
    "SIL": 0.60,
    "HG": 0.32,
    "MHG": 0.48,
    "PL": 0.16,
    "PA": 0.08,
    "6J": 0.48,
    "6B": 0.36,
    "6A": 0.28,
    "6C": 0.24,
    "6S": 0.20,
    "ZC": 1.20,
    "ZS": 0.80,
    "ZW": 0.50,
    "ZM": 0.40,
    "ZL": 0.40,
    "LE": 0.24,
    "HE": 0.20,
    "MBT": 0.08,
    "MET": 0.08,
}

_SYNTHETIC_ROOTS: tuple[str, ...] = ("ES", "NQ", "CL", "GC", "ZN")


class DatabentoIngestError(RuntimeError):
    """Raised when a Databento ingestion request cannot be satisfied."""


def synthetic_reference_price(root: InstrumentId) -> float:
    """Deterministic reference price for the synthetic source (fallback 1000.0)."""
    return SYNTHETIC_REFERENCE_PRICES.get(root.strip().upper(), 1_000.0)


def _validate_window(start: datetime, end: datetime) -> tuple[datetime, datetime]:
    """Validate a half-open UTC request window."""
    if not isinstance(start, datetime) or not isinstance(end, datetime):
        raise TypeError("start and end must be datetime instances")
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("start and end must be timezone-aware (UTC expected)")
    start_utc = start.astimezone(UTC)
    end_utc = end.astimezone(UTC)
    if start_utc >= end_utc:
        raise ValueError(
            f"start ({start_utc.isoformat()}) must be before end ({end_utc.isoformat()})"
        )
    return (start_utc, end_utc)


def _as_polars(store: Any) -> pl.DataFrame:
    """Convert a Databento ``DBNStore`` (or any table-like object) to Polars."""
    if isinstance(store, pl.DataFrame):
        return store
    to_df = getattr(store, "to_df", None)
    if callable(to_df):
        data = to_df()
        if isinstance(data, pl.DataFrame):
            return data
        return pl.from_pandas(data)
    to_arrow = getattr(store, "to_arrow", None)
    if callable(to_arrow):
        return cast(pl.DataFrame, pl.from_arrow(to_arrow()))
    if isinstance(store, dict):
        return pl.DataFrame(store)
    raise DatabentoIngestError(
        f"unsupported Databento response object {type(store)!r}: expected a DBNStore "
        "with .to_df()/.to_arrow() or a Polars DataFrame"
    )


def _pick_timestamp_column(frame: pl.DataFrame) -> str:
    """Pick the event timestamp column of a Databento frame."""
    for candidate in ("ts_event", "ts_recv", s.TIMESTAMP_COLUMN):
        if candidate in frame.columns:
            return candidate
    raise DatabentoIngestError(
        f"Databento frame has no timestamp column (looked for ts_event/ts_recv); "
        f"columns={frame.columns}"
    )


def _coerce_expiration(value: object) -> date:
    """Coerce a vendor expiration value (ns int, datetime, date, str) to a date."""
    if isinstance(value, datetime):
        return value.astimezone(UTC).date() if value.tzinfo else value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, (int, np.integer)):
        return datetime.fromtimestamp(int(value) / 1e9, UTC).date()
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError as exc:
            raise DatabentoIngestError(f"cannot parse expiration {value!r}") from exc
    raise DatabentoIngestError(f"unsupported expiration value {value!r} ({type(value)!r})")


class DatabentoLoader:
    """Historical Databento ingestion into the partitioned Parquet cache."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        repository: ParquetRepository | None = None,
        api_key: str | None = None,
        dataset: str | None = None,
        instrument_master: InstrumentsConfig | None = None,
        definition_lookback_days: int = 365,
    ) -> None:
        resolved = settings if settings is not None else get_settings()
        self._settings = resolved
        self._repository = repository if repository is not None else ParquetRepository(resolved)
        self._api_key = api_key if api_key is not None else resolved.databento_api_key_value
        self._dataset = dataset or resolved.databento_dataset
        self._master = instrument_master
        self._definition_lookback_days = definition_lookback_days

    # -- properties ------------------------------------------------------------

    @property
    def repository(self) -> ParquetRepository:
        """Underlying partitioned Parquet cache."""
        return self._repository

    @property
    def dataset(self) -> str:
        """Databento dataset id (``GLBX.MDP3`` by default)."""
        return self._dataset

    @property
    def instrument_master(self) -> InstrumentsConfig:
        """Instrument master used to map roots to Databento parent symbols."""
        if self._master is None:
            self._master = load_instruments(self._settings.instruments_path)
        return self._master

    # -- lazy SDK / credential access -----------------------------------------

    def _import_sdk(self) -> Any:
        """Import the optional Databento SDK, or raise an actionable error."""
        try:
            import databento
        except ImportError as exc:  # pragma: no cover - SDK is installed in CI
            raise RuntimeError(
                "The Databento SDK is not installed. Install the optional extra "
                "(`pip install 'futuresrl-x[databento]'` or `pip install databento`) "
                "or use `OfflineBarSource` for deterministic offline research."
            ) from exc
        return cast(Any, databento)

    def _require_api_key(self) -> str:
        """Return the configured Databento API key or raise an actionable error."""
        if not self._api_key:
            raise RuntimeError(
                "No Databento API key configured: set FUTURESRLX_DATABENTO_API_KEY "
                "(or pass api_key=...) before running historical ingestion. "
                "Offline research/tests must use `OfflineBarSource` instead."
            )
        return self._api_key

    def _client(self) -> Any:
        """Build a ``databento.Historical`` client (lazy import, credentials checked)."""
        sdk = self._import_sdk()
        return sdk.Historical(api_key=self._require_api_key())

    # -- symbol resolution -----------------------------------------------------

    def _symbols_for(self, roots: Sequence[InstrumentId], *, stype_in: str) -> list[str]:
        """Map canonical roots to the Databento symbols used by ``stype_in``."""
        if stype_in == "continuous":
            return [native_continuous_symbol(root, "volume") for root in roots]
        if stype_in in ("parent", ""):
            return [self.instrument_master.get(root).databento_symbol for root in roots]
        return [str(root) for root in roots]

    def _definition_lookup(self) -> dict[str, tuple[str, date]]:
        """``instrument_id -> (root, expiration)`` from the cached definition schema."""
        definitions = self._repository.load_definitions()
        if definitions.is_empty():
            return {}
        lookup: dict[str, tuple[str, date]] = {}
        for row in definitions.iter_rows(named=True):
            instrument = str(row[s.INSTRUMENT_ID_COLUMN])
            expiration = row[s.EXPIRATION_COLUMN]
            if not isinstance(expiration, date):  # pragma: no cover - schema guarantees it
                continue
            lookup[instrument] = (str(row[s.ROOT_SYMBOL_COLUMN]), expiration)
        return lookup

    def _resolve_identity(
        self,
        symbol: str,
        *,
        definitions: dict[str, tuple[str, date]],
        roots: Sequence[InstrumentId],
        reference: datetime,
    ) -> tuple[str, str, date]:
        """Resolve ``(instrument_id, root, expiration)`` for one vendor symbol."""
        cached = definitions.get(symbol)
        if cached is not None:
            return (symbol, cached[0], cached[1])
        parsed = parse_contract_symbol(
            symbol,
            roots=[self.instrument_master.get(root).root for root in roots],
            reference=reference,
        )
        if parsed is None:
            raise DatabentoIngestError(
                f"cannot resolve the contract identity of symbol {symbol!r}. Ingest the "
                "`definition` schema first (`loader.ingest(..., schema='definition')`) so "
                "instrument ids can be joined to root/expiration, or query with "
                "stype_in='parent'."
            )
        root, expiration = parsed
        return (symbol, root, expiration)

    def _resolve_identities(
        self,
        frame: pl.DataFrame,
        *,
        symbol_column: str,
        roots: Sequence[InstrumentId],
        reference: datetime,
    ) -> dict[str, tuple[str, str, date]]:
        """Resolve every unique symbol of ``frame`` to ``(id, root, expiration)``."""
        symbols = [str(value) for value in frame[symbol_column].cast(pl.Utf8).to_list()]
        definitions = self._definition_lookup()
        resolved: dict[str, tuple[str, str, date]] = {}
        for symbol in dict.fromkeys(symbols):
            resolved[symbol] = self._resolve_identity(
                symbol, definitions=definitions, roots=roots, reference=reference
            )
        return resolved

    # -- public ingestion API ---------------------------------------------------

    def ingest(
        self,
        roots: Sequence[InstrumentId],
        start: datetime,
        end: datetime,
        schema: str = s.DEFAULT_BAR_SCHEMA,
        *,
        dataset: str | None = None,
        stype_in: str = "parent",
        mode: WriteMode = "append",
        client: Any | None = None,
        symbols: Sequence[str] | None = None,
    ) -> pl.DataFrame:
        """Fetch ``schema`` for ``roots`` in ``[start, end)`` and cache it.

        Returns the normalised canonical frame (also persisted to the repository).
        Raises :class:`DatabentoIngestError` on an empty/failed vendor response
        and :class:`RuntimeError` when the SDK or API key is unavailable.
        """
        normalised_roots = s.validate_roots(roots)
        start_utc, end_utc = _validate_window(start, end)
        if schema not in s.DATABENTO_SCHEMAS:
            raise ValueError(
                f"unsupported Databento schema {schema!r}; expected one of "
                f"{list(s.DATABENTO_SCHEMAS)}"
            )
        resolved_dataset = dataset or self._dataset
        requested_symbols = (
            list(symbols)
            if symbols is not None
            else self._symbols_for(normalised_roots, stype_in=stype_in)
        )
        client_object = client if client is not None else self._client()
        try:
            store = client_object.timeseries.get_range(
                dataset=resolved_dataset,
                schema=schema,
                symbols=requested_symbols,
                start=start_utc,
                end=end_utc,
                stype_in=stype_in,
            )
        except Exception as exc:
            raise DatabentoIngestError(
                f"Databento get_range failed (dataset={resolved_dataset!r}, schema={schema!r}, "
                f"roots={normalised_roots!r}, window=[{start_utc.isoformat()}, "
                f"{end_utc.isoformat()})): {exc}"
            ) from exc
        raw = _as_polars(store)
        if raw.is_empty():
            raise DatabentoIngestError(
                f"Databento returned no {schema!r} rows for {normalised_roots!r} in "
                f"[{start_utc.isoformat()}, {end_utc.isoformat()}); check the request "
                "window/entitlements or widen the date range."
            )
        frame = self._normalise(raw, schema=schema, roots=normalised_roots, reference=end_utc)
        if schema == s.DATABENTO_DEFINITION_SCHEMA:
            self._repository.store_definitions(frame, dataset=resolved_dataset, mode=mode)
        elif s.is_bar_schema(schema):
            self._repository.write_bars(frame, dataset=resolved_dataset, schema=schema, mode=mode)
        else:
            self._repository.write_frame(frame, schema=schema, dataset=resolved_dataset, mode=mode)
        return frame

    def ingest_definitions(
        self,
        roots: Sequence[InstrumentId],
        start: datetime,
        end: datetime,
        *,
        dataset: str | None = None,
        mode: WriteMode = "append",
        client: Any | None = None,
    ) -> pl.DataFrame:
        """Fetch and cache point-in-time contract definitions (multipliers, expiries)."""
        return self.ingest(
            roots,
            start,
            end,
            s.DATABENTO_DEFINITION_SCHEMA,
            dataset=dataset,
            mode=mode,
            client=client,
        )

    # -- HistoricalDataSource protocol -----------------------------------------

    def load(
        self,
        roots: Sequence[InstrumentId],
        start: datetime,
        end: datetime,
        *,
        schema: str = s.DEFAULT_BAR_SCHEMA,
    ) -> pl.DataFrame:
        """Load bars from the cache when complete, otherwise ingest from Databento."""
        normalised_roots = s.validate_roots(roots)
        start_utc, end_utc = _validate_window(start, end)
        if not s.is_bar_schema(schema):
            raise ValueError(
                f"DatabentoLoader.load only serves bar schemas {list(s.DATABENTO_BAR_SCHEMAS)}, "
                f"got {schema!r}"
            )
        cached = self._repository.read_bars(normalised_roots, start_utc, end_utc, schema=schema)
        if self._covers(cached, start_utc, end_utc, schema):
            return cached
        return self.ingest(normalised_roots, start_utc, end_utc, schema=schema)

    def load_definitions(self, roots: Sequence[InstrumentId], *, as_of: datetime) -> pl.DataFrame:
        """Load point-in-time definitions, ingesting the lookback window on a cache miss."""
        normalised_roots = s.validate_roots(roots)
        if as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware (UTC expected)")
        cached = self._repository.load_definitions(normalised_roots, as_of=as_of)
        if not cached.is_empty():
            return cached
        lookback = as_of - timedelta(days=self._definition_lookback_days)
        return self.ingest_definitions(normalised_roots, lookback, as_of)

    def available_roots(self) -> Sequence[InstrumentId]:
        """Roots present in the cache for any bar schema."""
        roots: set[str] = set()
        for schema in s.DATABENTO_BAR_SCHEMAS:
            roots.update(self._repository.available_roots(schema))
        return tuple(sorted(roots))

    # -- normalisation ---------------------------------------------------------

    def _covers(self, frame: pl.DataFrame, start: datetime, end: datetime, schema: str) -> bool:
        """Whether ``frame`` already covers the request window for ``schema``."""
        if frame.is_empty():
            return False
        interval = s.bar_interval(schema)
        column = frame[s.TIMESTAMP_COLUMN]
        first = cast(datetime, column.min())
        last = cast(datetime, column.max())
        return first <= start + interval and last >= end - interval

    def _normalise(
        self,
        raw: pl.DataFrame,
        *,
        schema: str,
        roots: Sequence[InstrumentId],
        reference: datetime,
    ) -> pl.DataFrame:
        """Dispatch a raw vendor frame to the matching canonical normaliser."""
        if schema == s.DATABENTO_DEFINITION_SCHEMA:
            return self._normalise_definitions(raw, roots=roots, reference=reference)
        if s.is_bar_schema(schema):
            return self._normalise_bars(raw, roots=roots, reference=reference)
        if schema in s.DATABENTO_QUOTE_SCHEMAS:
            return self._normalise_quotes(raw, roots=roots, reference=reference)
        return self._normalise_book(raw, roots=roots, reference=reference)

    def _symbol_column(self, raw: pl.DataFrame) -> str:
        """Databento exposes both ``symbol`` and ``raw_symbol``; prefer the raw one."""
        if "raw_symbol" in raw.columns:
            return "raw_symbol"
        if "symbol" in raw.columns:
            return "symbol"
        raise DatabentoIngestError(f"Databento frame has no symbol column; columns={raw.columns}")

    def _identity_columns(
        self,
        raw: pl.DataFrame,
        *,
        roots: Sequence[InstrumentId],
        reference: datetime,
        timestamp_column: str | None = None,
    ) -> pl.DataFrame:
        """Attach canonical ``instrument_id``/``root_symbol``/``expiration_date`` columns."""
        frame = s.normalise_timestamp_column(raw, timestamp_column or _pick_timestamp_column(raw))
        if s.TIMESTAMP_COLUMN not in frame.columns:
            source = timestamp_column or _pick_timestamp_column(raw)
            frame = frame.with_columns(pl.col(source).alias(s.TIMESTAMP_COLUMN))
        symbol_column = self._symbol_column(frame)
        resolved = self._resolve_identities(
            frame, symbol_column=symbol_column, roots=roots, reference=reference
        )
        symbols = [str(value) for value in frame[symbol_column].cast(pl.Utf8).to_list()]
        identities = [resolved[symbol] for symbol in symbols]
        return frame.with_columns(
            pl.Series(s.INSTRUMENT_ID_COLUMN, [entry[0] for entry in identities], dtype=pl.Utf8),
            pl.Series(s.ROOT_SYMBOL_COLUMN, [entry[1] for entry in identities], dtype=pl.Utf8),
            pl.Series(
                s.EXPIRATION_COLUMN,
                [entry[2] for entry in identities],
                dtype=pl.Date,
            ),
        )

    def _normalise_bars(
        self, raw: pl.DataFrame, *, roots: Sequence[InstrumentId], reference: datetime
    ) -> pl.DataFrame:
        """Convert an OHLCV vendor frame to the canonical bar schema."""
        frame = self._identity_columns(raw, roots=roots, reference=reference)
        required = (s.OPEN_COLUMN, s.HIGH_COLUMN, s.LOW_COLUMN, s.CLOSE_COLUMN)
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise DatabentoIngestError(f"OHLCV frame is missing columns {missing}")
        if s.VOLUME_COLUMN not in frame.columns:
            frame = frame.with_columns(pl.lit(0.0, dtype=pl.Float64).alias(s.VOLUME_COLUMN))
        if s.OPEN_INTEREST_COLUMN not in frame.columns:
            frame = frame.with_columns(pl.lit(0.0, dtype=pl.Float64).alias(s.OPEN_INTEREST_COLUMN))
        if s.SETTLEMENT_COLUMN not in frame.columns:
            frame = frame.with_columns(pl.lit(None, dtype=pl.Float64).alias(s.SETTLEMENT_COLUMN))
        columns = [*s.BAR_COLUMNS]
        normalised = frame.select(columns)
        return s.ensure_bars_schema(normalised, context="databento-bars")

    def _normalise_quotes(
        self, raw: pl.DataFrame, *, roots: Sequence[InstrumentId], reference: datetime
    ) -> pl.DataFrame:
        """Convert a ``tbbo`` vendor frame to the canonical quote schema."""
        frame = self._identity_columns(raw, roots=roots, reference=reference)
        for required in (s.BID_PRICE_COLUMN, s.ASK_PRICE_COLUMN):
            if required not in frame.columns:
                raise DatabentoIngestError(
                    f"tbbo frame is missing {required!r}; columns={frame.columns}"
                )
        for optional in (
            s.BID_SIZE_COLUMN,
            s.ASK_SIZE_COLUMN,
            s.BID_COUNT_COLUMN,
            s.ASK_COUNT_COLUMN,
        ):
            if optional not in frame.columns:
                frame = frame.with_columns(pl.lit(0.0, dtype=pl.Float64).alias(optional))
        extra = [column for column in frame.columns if column not in s.QUOTE_COLUMNS]
        normalised = frame.select([*s.QUOTE_COLUMNS, *extra])
        return s.ensure_quote_schema(normalised, context="databento-tbbo")

    def _normalise_book(
        self, raw: pl.DataFrame, *, roots: Sequence[InstrumentId], reference: datetime
    ) -> pl.DataFrame:
        """Convert an ``mbp-1``/``mbp-10`` vendor frame to the canonical book frame."""
        frame = self._identity_columns(raw, roots=roots, reference=reference)
        normalised = frame
        for column in normalised.columns:
            if column.startswith(("bid_px_", "ask_px_", "bid_sz_", "ask_sz_")):
                normalised = normalised.with_columns(pl.col(column).cast(pl.Float64))
        return s.ensure_key_schema(normalised, context="databento-mbp")

    def _normalise_definitions(
        self, raw: pl.DataFrame, *, roots: Sequence[InstrumentId], reference: datetime
    ) -> pl.DataFrame:
        """Convert a ``definition`` vendor frame to the canonical definition schema."""
        frame = raw
        symbol_column = self._symbol_column(frame)
        timestamp_column = (
            _pick_timestamp_column(frame)
            if any(column in frame.columns for column in ("ts_event", "ts_recv"))
            else None
        )
        if timestamp_column is not None:
            frame = s.normalise_timestamp_column(frame, timestamp_column)
        else:
            frame = frame.with_columns(
                pl.lit(reference, dtype=pl.Datetime("us", "UTC")).alias(s.TIMESTAMP_COLUMN)
            )
        definitions = self._definition_lookup()
        rows: list[dict[str, Any]] = []
        for row in frame.iter_rows(named=True):
            symbol = str(row[symbol_column])
            instrument_id, root, parsed_expiration = self._resolve_identity(
                symbol, definitions=definitions, roots=roots, reference=reference
            )
            expiration = (
                _coerce_expiration(row["expiration"])
                if row.get("expiration") is not None
                else parsed_expiration
            )
            spec = self._spec_or_none(root)
            multiplier = row.get("unit_of_measure_qty")
            if multiplier is None:
                multiplier = row.get("multiplier")
            if multiplier is None and spec is not None:
                multiplier = spec.multiplier
            tick_size = row.get("min_price_increment")
            if tick_size is None:
                tick_size = row.get("tick_size")
            if tick_size is None and spec is not None:
                tick_size = spec.tick_size
            activation = row.get("activation")
            if activation is None:
                activation = row.get(timestamp_column) if timestamp_column else None
            if activation is None:
                activation = reference
            rows.append(
                {
                    s.INSTRUMENT_ID_COLUMN: instrument_id,
                    s.ROOT_SYMBOL_COLUMN: root,
                    s.EXPIRATION_COLUMN: expiration,
                    "multiplier": float(multiplier) if multiplier is not None else None,
                    "tick_size": float(tick_size) if tick_size is not None else None,
                    "exchange": row.get("exchange"),
                    "currency": row.get("currency", "USD"),
                    "activation": activation,
                }
            )
        return s.ensure_definitions_schema(pl.DataFrame(rows), context="databento-definition")

    def _spec_or_none(self, root: str) -> InstrumentSpec | None:
        """Instrument spec for ``root`` if the master declares it."""
        master = self.instrument_master
        return master.get(root) if master.has(root) else None


@dataclass(frozen=True, slots=True)
class _SyntheticPath:
    """Deterministic price/volume path of one synthetic contract family.

    A *family* is one underlying plus its paired micro contract (``ES``/``MES``,
    ``CL``/``MCL``, ...). The paired roots share this single path so the
    documented ``P_micro == P_standard`` invariant (used by the
    ContractTranslator's family/micro sizing) always holds; only the root's own
    volume/open-interest and contract identity (ids/expiries) differ.
    """

    root: str
    opens: NDArray[np.float64]
    highs: NDArray[np.float64]
    lows: NDArray[np.float64]
    closes: NDArray[np.float64]
    volumes: NDArray[np.float64]
    open_interest: NDArray[np.float64]
    front_expirations: tuple[date, ...]
    second_expirations: tuple[date, ...] = ()
    second_opens: NDArray[np.float64] | None = None
    second_highs: NDArray[np.float64] | None = None
    second_lows: NDArray[np.float64] | None = None
    second_closes: NDArray[np.float64] | None = None


class OfflineBarSource:
    """Deterministic synthetic OHLCV source for offline backtests and CI.

    The path is a seeded geometric Brownian motion per root with volatility
    derived from the root's asset class. By default every bar is emitted as a
    **front/second-month pair**: F1 follows the seeded GBM and F2 is the same bar
    scaled by ``1 + term_slope(t)`` from
    :class:`~src.data.term_structure.SyntheticTermStructure`, whose carry is
    root-specific (equity/rates/metals contango, energy backwardation, FX near
    flat) and modulated by a slow deterministic calendar regime cycle. The
    expiry pair follows the instrument master's delivery cycle and rolls with the
    configured calendar buffer (``days_before_fnd`` + physical-delivery safety),
    so ``FeatureProcessor`` derives finite, non-zero ``term_slope`` and
    ``carry_annualized`` without any look-ahead.

    Everything is *entirely synthetic*: contract ids are suffixed
    ``_SYN<yyyymmdd>`` and ``is_synthetic`` is ``True``. Use it for deterministic
    tests and offline research only — never in paper/live mode and never as a
    substitute for Databento vendor data in reported results. Pass
    ``term_structure=False`` to reproduce the legacy single-expiration series.
    """

    #: Marker consumed by guard-rails that must reject synthetic data in live mode.
    is_synthetic: bool = True

    def __init__(
        self,
        roots: Sequence[InstrumentId] | None = None,
        *,
        seed: int = 7,
        start: datetime | None = None,
        periods: int = 512,
        instrument_master: InstrumentsConfig | None = None,
        term_structure: bool = True,
        term_structure_model: SyntheticTermStructure | None = None,
    ) -> None:
        self._roots = s.validate_roots(tuple(roots) if roots is not None else _SYNTHETIC_ROOTS)
        if start is None:
            start = datetime(2024, 1, 2, 21, 0, tzinfo=UTC)
        if start.tzinfo is None:
            raise ValueError("start must be timezone-aware (UTC expected)")
        self._start = start.astimezone(UTC)
        if periods < 2:
            raise ValueError("periods must be >= 2")
        self._periods = periods
        self._seed = seed
        self._master = instrument_master
        if not isinstance(term_structure, bool):
            raise TypeError(f"term_structure must be a bool, got {type(term_structure)!r}")
        self._term_structure_enabled = term_structure
        if term_structure_model is not None and not isinstance(
            term_structure_model, SyntheticTermStructure
        ):
            raise TypeError(
                "term_structure_model must be a SyntheticTermStructure, got "
                f"{type(term_structure_model)!r}"
            )
        self._term_structure_model = term_structure_model

    @property
    def roots(self) -> tuple[InstrumentId, ...]:
        """Roots this source can generate."""
        return self._roots

    @property
    def start(self) -> datetime:
        """Timestamp of the first generated bar."""
        return self._start

    @property
    def instrument_master(self) -> InstrumentsConfig:
        """Instrument master used for multipliers and delivery cycles."""
        if self._master is None:
            self._master = load_instruments()
        return self._master

    @property
    def term_structure_enabled(self) -> bool:
        """Whether the synthetic F1/F2 curve is generated (default ``True``)."""
        return self._term_structure_enabled

    @property
    def term_structure_model(self) -> SyntheticTermStructure:
        """Deterministic synthetic curve model used for the F1/F2 spread."""
        if self._term_structure_model is None:
            self._term_structure_model = SyntheticTermStructure(instrument_master=self._master)
        return self._term_structure_model

    def available_roots(self) -> Sequence[InstrumentId]:
        """Roots this source can generate (mirrors the cache contract)."""
        return self._roots

    # -- HistoricalDataSource protocol -----------------------------------------

    def load(
        self,
        roots: Sequence[InstrumentId],
        start: datetime,
        end: datetime,
        *,
        schema: str = s.DEFAULT_BAR_SCHEMA,
    ) -> pl.DataFrame:
        """Generate deterministic synthetic bars in the half-open window ``[start, end)``.

        With the default ``term_structure=True`` each root/timestamp carries two
        rows (front + synthetic second month); ``term_structure=False`` keeps the
        legacy one-row-per-root/timestamp contract.
        """
        if not s.is_bar_schema(schema):
            raise ValueError(
                f"OfflineBarSource only produces OHLCV bars {list(s.DATABENTO_BAR_SCHEMAS)}, "
                f"got {schema!r}"
            )
        normalised_roots = s.validate_roots(roots)
        start_utc, end_utc = _validate_window(start, end)
        interval = s.bar_interval(schema)
        origin = min(start_utc, self._start)
        needed = math.ceil((end_utc - origin) / interval) + 2
        frame = self._generate(normalised_roots, max(self._periods, needed), interval, origin)
        window = frame.filter(
            (pl.col(s.TIMESTAMP_COLUMN) >= pl.lit(start_utc))
            & (pl.col(s.TIMESTAMP_COLUMN) < pl.lit(end_utc))
        ).sort([s.TIMESTAMP_COLUMN, s.ROOT_SYMBOL_COLUMN, s.EXPIRATION_COLUMN])
        return s.ensure_bars_schema(window, context="offline-bars")

    def load_definitions(self, roots: Sequence[InstrumentId], *, as_of: datetime) -> pl.DataFrame:
        """Synthetic definition rows for every contract generated by :meth:`load`.

        The generated expiry chain is deterministic for ``(roots, start, periods)``;
        each expiration (front *and* synthetic second month) gets one definition
        row so a join on ``instrument_id`` always resolves.
        """
        normalised_roots = s.validate_roots(roots)
        if as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware (UTC expected)")
        rows: list[dict[str, Any]] = []
        for root in normalised_roots:
            spec = self._spec_or_none(root)
            for expiration in self._expirations_for(root, spec):
                rows.append(
                    {
                        s.INSTRUMENT_ID_COLUMN: self._contract_id(root, expiration),
                        s.ROOT_SYMBOL_COLUMN: root,
                        s.EXPIRATION_COLUMN: expiration,
                        "multiplier": spec.multiplier if spec is not None else 1.0,
                        "tick_size": spec.tick_size if spec is not None else 0.01,
                        "exchange": spec.exchange if spec is not None else "SYNTHETIC",
                        "currency": spec.currency if spec is not None else "USD",
                        "activation": self._start - timedelta(days=30),
                    }
                )
        return s.ensure_definitions_schema(pl.DataFrame(rows), context="offline-definitions")

    # -- internals -------------------------------------------------------------

    def _spec_or_none(self, root: str) -> InstrumentSpec | None:
        master = self.instrument_master
        return master.get(root) if master.has(root) else None

    def _contract_id(self, root: str, expiration: date) -> str:
        """Unambiguous synthetic contract id (never mistakable for vendor data)."""
        return f"{root}_SYN{expiration:%Y%m%d}"

    def _front_expiration(self, spec: InstrumentSpec | None) -> date:
        """Legacy single-series front expiry (no term structure mode)."""
        months = spec.roll.contract_months if spec is not None else (3, 6, 9, 12)
        return next_contract_expiration(months, self._start)

    def _expirations_for(self, root: str, spec: InstrumentSpec | None) -> tuple[date, ...]:
        """Expiries emitted by this source for ``root`` (front/second chain)."""
        if not self._term_structure_enabled:
            return (self._front_expiration(spec),)
        horizon = max(self._periods, _DEFINITION_HORIZON_PERIODS) + _EXPIRY_CHAIN_BUFFER_DAYS
        return self.term_structure_model.expiration_chain(
            root, reference=self._start, horizon_days=horizon
        )

    def _root_seed(self, root: str) -> int:
        """Stable per-root RNG seed (independent of dict/iteration order)."""
        return int(self._seed) * 1_000_003 + sum(
            (index + 1) * ord(char) for index, char in enumerate(root)
        )

    def _annual_vol(self, spec: InstrumentSpec | None) -> float:
        if spec is None:
            return SYNTHETIC_ANNUAL_VOL["unknown"]
        # A paired micro contract follows its standard contract's risk profile.
        if spec.standard_root is not None:
            standard = self._spec_or_none(spec.standard_root)
            if standard is not None:
                return self._annual_vol(standard)
        if spec.root in SYNTHETIC_ROOT_ANNUAL_VOL:
            return SYNTHETIC_ROOT_ANNUAL_VOL[spec.root]
        return SYNTHETIC_ANNUAL_VOL.get(str(spec.asset_class), SYNTHETIC_ANNUAL_VOL["unknown"])

    @staticmethod
    def _volume_scale(root: str) -> float:
        """Relative synthetic volume / open-interest baseline for ``root``."""
        return SYNTHETIC_VOLUME_SCALE.get(root.strip().upper(), 1.0)

    @staticmethod
    def _leg_frame(
        root: str,
        timestamps: Sequence[datetime],
        opens: NDArray[np.float64],
        highs: NDArray[np.float64],
        lows: NDArray[np.float64],
        closes: NDArray[np.float64],
        volumes: NDArray[np.float64],
        open_interest: NDArray[np.float64],
        expirations: Sequence[date],
    ) -> pl.DataFrame:
        """Assemble one canonical-schema contract leg for ``root``."""
        contract_ids = [f"{root}_SYN{expiration:%Y%m%d}" for expiration in expirations]
        return pl.DataFrame(
            {
                s.TIMESTAMP_COLUMN: list(timestamps),
                s.INSTRUMENT_ID_COLUMN: contract_ids,
                s.ROOT_SYMBOL_COLUMN: [root] * len(timestamps),
                s.EXPIRATION_COLUMN: list(expirations),
                s.OPEN_COLUMN: opens.tolist(),
                s.HIGH_COLUMN: highs.tolist(),
                s.LOW_COLUMN: lows.tolist(),
                s.CLOSE_COLUMN: closes.tolist(),
                s.VOLUME_COLUMN: volumes.tolist(),
                s.OPEN_INTEREST_COLUMN: open_interest.tolist(),
                s.SETTLEMENT_COLUMN: closes.tolist(),
            }
        )

    def _family_root(self, root: str, spec: InstrumentSpec | None) -> str:
        """Underlying family root for ``root`` (standard root for micros)."""
        if spec is not None and spec.standard_root is not None:
            return spec.standard_root
        return root

    def _build_path(
        self,
        root: str,
        spec: InstrumentSpec | None,
        *,
        periods: int,
        interval: timedelta,
        timestamps: Sequence[datetime],
        horizon_days: int,
        reuse_prices: _SyntheticPath | None = None,
    ) -> _SyntheticPath:
        """Generate the deterministic path of one root (or reuse family prices).

        ``reuse_prices`` is used for a paired micro root: the price columns (front
        and second month, OHLC + settlement) are the standard contract's exact
        path, while the micro keeps its own deterministic volume/open-interest and
        its own contract ids/expiries.
        """
        bar_days = interval.total_seconds() / 86_400.0
        # Single seeded draw order keeps the historical synthetic price path
        # unchanged (chart/test baselines) while the term structure is layered
        # on top purely from the calendar.
        rng = np.random.default_rng(self._root_seed(root))
        reference_price = synthetic_reference_price(root)
        annual_vol = self._annual_vol(spec)
        sigma = annual_vol * math.sqrt(max(bar_days, 1e-9) / 252.0)
        returns = rng.normal(0.0, sigma, size=periods)
        closes = reference_price * np.cumprod(1.0 + returns)
        opens = np.concatenate([[reference_price], closes[:-1]])
        intrabar = rng.uniform(0.0, sigma, size=periods)
        highs = np.maximum(opens, closes) * (1.0 + intrabar)
        lows = np.minimum(opens, closes) * (1.0 - intrabar)
        volume_scale = self._volume_scale(root)
        volumes = np.abs(
            rng.normal(250_000.0 * volume_scale, 25_000.0 * volume_scale, size=periods)
        )
        open_interest = np.abs(
            rng.normal(1_500_000.0 * volume_scale, 50_000.0 * volume_scale, size=periods)
        )
        if not self._term_structure_enabled:
            expiration = self._front_expiration(spec)
            front_expirations = tuple([expiration] * periods)
            if reuse_prices is not None:
                assert reuse_prices.second_closes is None
                opens = reuse_prices.opens
                highs = reuse_prices.highs
                lows = reuse_prices.lows
                closes = reuse_prices.closes
            return _SyntheticPath(
                root=root,
                opens=opens,
                highs=highs,
                lows=lows,
                closes=closes,
                volumes=volumes,
                open_interest=open_interest,
                front_expirations=front_expirations,
            )
        model = self.term_structure_model
        chain = model.expiration_chain(root, reference=timestamps[0], horizon_days=horizon_days)
        front_expirations_list: list[date] = []
        second_expirations_list: list[date] = []
        second_ratios: list[float] = []
        for timestamp in timestamps:
            pair = model.pair_at(root, timestamp, chain)
            if pair is None:  # pragma: no cover - chain always has >= 2 expiries
                raise RuntimeError(f"synthetic term structure produced no expiry pair for {root!r}")
            curve = model.curve(
                root,
                timestamp,
                front_price=1.0,
                front_expiration=pair[0],
                second_expiration=pair[1],
            )
            front_expirations_list.append(pair[0])
            second_expirations_list.append(pair[1])
            second_ratios.append(curve.second_price / curve.front_price)
        ratios = np.asarray(second_ratios, dtype=np.float64)
        second_opens = opens * ratios
        second_highs = highs * ratios
        second_lows = lows * ratios
        second_closes = closes * ratios
        if reuse_prices is not None:
            # Paired standard <-> micro shares one underlying: reuse the standard
            # family's exact prices (front + second month) for both legs.
            assert reuse_prices.second_opens is not None
            assert reuse_prices.second_highs is not None
            assert reuse_prices.second_lows is not None
            assert reuse_prices.second_closes is not None
            opens = reuse_prices.opens
            highs = reuse_prices.highs
            lows = reuse_prices.lows
            closes = reuse_prices.closes
            second_opens = reuse_prices.second_opens
            second_highs = reuse_prices.second_highs
            second_lows = reuse_prices.second_lows
            second_closes = reuse_prices.second_closes
        return _SyntheticPath(
            root=root,
            opens=opens,
            highs=highs,
            lows=lows,
            closes=closes,
            volumes=volumes,
            open_interest=open_interest,
            front_expirations=tuple(front_expirations_list),
            second_expirations=tuple(second_expirations_list),
            second_opens=second_opens,
            second_highs=second_highs,
            second_lows=second_lows,
            second_closes=second_closes,
        )

    def _generate(
        self,
        roots: Sequence[InstrumentId],
        periods: int,
        interval: timedelta,
        origin: datetime,
    ) -> pl.DataFrame:
        """Generate the deterministic synthetic bar frame for ``roots``."""
        frames: list[pl.DataFrame] = []
        timestamps = [origin + interval * step for step in range(periods)]
        horizon_days = (
            math.ceil((timestamps[-1] - origin).total_seconds() / 86_400.0)
            + _EXPIRY_CHAIN_BUFFER_DAYS
        )
        family_paths: dict[str, _SyntheticPath] = {}
        for root in roots:
            spec = self._spec_or_none(root)
            family_root = self._family_root(root, spec)
            owner = family_paths.get(family_root)
            if owner is None:
                owner = self._build_path(
                    family_root,
                    self._spec_or_none(family_root),
                    periods=periods,
                    interval=interval,
                    timestamps=timestamps,
                    horizon_days=horizon_days,
                )
                family_paths[family_root] = owner
            if root == family_root:
                path = owner
            else:
                path = self._build_path(
                    root,
                    spec,
                    periods=periods,
                    interval=interval,
                    timestamps=timestamps,
                    horizon_days=horizon_days,
                    reuse_prices=owner,
                )
                # A paired micro must share the standard contract's delivery
                # cycle/roll config; otherwise price/expiry labels would disagree.
                if (
                    path.front_expirations != owner.front_expirations
                    or path.second_expirations != owner.second_expirations
                ):
                    raise RuntimeError(
                        f"synthetic family {family_root!r}: micro root {root!r} has a "
                        "different delivery/roll calendar than its standard contract; "
                        "align roll.contract_months/days_before_fnd/physical_delivery"
                    )
            frames.append(
                self._leg_frame(
                    root,
                    timestamps,
                    path.opens,
                    path.highs,
                    path.lows,
                    path.closes,
                    path.volumes,
                    path.open_interest,
                    path.front_expirations,
                )
            )
            if path.second_closes is not None:
                assert path.second_opens is not None
                assert path.second_highs is not None
                assert path.second_lows is not None
                frames.append(
                    self._leg_frame(
                        root,
                        timestamps,
                        path.second_opens,
                        path.second_highs,
                        path.second_lows,
                        path.second_closes,
                        path.volumes * _SECOND_MONTH_VOLUME_SCALE,
                        path.open_interest * _SECOND_MONTH_OI_SCALE,
                        path.second_expirations,
                    )
                )
        frame = pl.concat(frames, how="vertical_relaxed")
        return s.ensure_bars_schema(frame, context="offline-generate").sort(
            [s.TIMESTAMP_COLUMN, s.ROOT_SYMBOL_COLUMN, s.EXPIRATION_COLUMN]
        )
