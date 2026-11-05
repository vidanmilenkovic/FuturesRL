"""Immutable domain contracts shared by every FuturesRL-X module.

This module is the narrow waist of the platform: the data layer, the FinRL-X
composable pipeline, the LLM/quant risk overlays, the futures contract
translator, the backtester, the IBKR execution engine and the FastAPI
telemetry layer all exchange *only* the types defined here.

Design rules
------------
* Every container is an immutable (frozen) dataclass; where a ``Mapping`` or
  ``ndarray`` is embedded it is defensively copied and made read-only in
  ``__post_init__`` so that no downstream consumer can mutate shared state.
* Every numeric invariant that would silently corrupt a backtest (negative
  prices, naive timestamps, ``NaN`` weights, negative margin, bid > ask, ...)
  is validated at construction time with an explicit ``ValueError``.
* The futures weight is the only interface contract: nothing in this file
  knows about brokers, LLMs or storage.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime
from enum import StrEnum
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Primitive aliases / value unions
# ---------------------------------------------------------------------------

#: Canonical identifier of a futures *root* (e.g. ``"ES"``, ``"MES"``, ``"CL"``).
type InstrumentId = str

#: A scalar that can be exported through JSON/telemetry channels.
type ScalarValue = str | int | float | bool | None

#: Immutable metadata value (scalars, scalar tuples or scalar mappings only).
type MetadataValue = ScalarValue | tuple[ScalarValue, ...] | Mapping[str, ScalarValue]

#: Annualisation factor used when a daily volatility is annualised.
TRADING_DAYS_PER_YEAR: int = 252

#: Number of months in a year (term-structure / roll arithmetic).
MONTHS_PER_YEAR: int = 12


def ensure_utc(timestamp: datetime) -> datetime:
    """Return ``timestamp`` normalised to a timezone-aware UTC datetime.

    Naive datetimes are rejected: a futures platform must never guess whether a
    timestamp is New York, Chicago or UTC time (that is a classic look-ahead /
    settlement bug).
    """
    if not isinstance(timestamp, datetime):
        raise TypeError(f"timestamp must be a datetime, got {type(timestamp)!r}")
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError(
            f"timestamp must be timezone-aware (UTC expected), got naive {timestamp!r}"
        )
    return timestamp.astimezone(UTC)


def _validate_finite(name: str, value: float, *, allow_zero: bool = True) -> float:
    """Validate a finite float and return it as ``float``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value)!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result!r}")
    if not allow_zero and result == 0.0:
        raise ValueError(f"{name} must be non-zero")
    return result


def _freeze_float_mapping(weights: Mapping[InstrumentId, float]) -> Mapping[InstrumentId, float]:
    """Validate, copy and freeze an ``instrument -> weight`` mapping."""
    cleaned: dict[InstrumentId, float] = {}
    for instrument_id, value in weights.items():
        if not isinstance(instrument_id, str) or not instrument_id:
            raise ValueError(f"instrument ids must be non-empty strings, got {instrument_id!r}")
        cleaned[instrument_id] = _validate_finite(f"weight[{instrument_id!r}]", value)
    return MappingProxyType(cleaned)


def _readonly_array(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return a read-only copy of ``values``."""
    array = np.array(values, dtype=np.float64, copy=True)
    array.setflags(write=False)
    return array


def normalise_currency(code: str, *, name: str = "currency") -> str:
    """Validate an ISO-4217-style 3-letter currency code and upper-case it."""
    if not isinstance(code, str):
        raise TypeError(f"{name} must be a string, got {type(code)!r}")
    cleaned = code.strip().upper()
    if len(cleaned) != 3 or not cleaned.isalpha():
        raise ValueError(f"{name} must be a 3-letter ISO-4217 code, got {code!r}")
    return cleaned


def freeze_fx_rates(rates: Mapping[str, float]) -> Mapping[str, float]:
    """Validate, copy and freeze ``currency -> USD per unit`` FX rates.

    Convention: ``fx_rates["EUR"] == 1.10`` means one euro buys 1.10 US dollars,
    i.e. the value is the USD amount per one unit of the quoted currency. USD is
    implicitly 1.0 and, if supplied explicitly, must equal 1.0.
    """
    cleaned: dict[str, float] = {}
    for currency, rate in rates.items():
        code = normalise_currency(currency, name="fx_rates currency")
        value = _validate_finite(f"fx_rates[{code!r}]", rate, allow_zero=False)
        if value <= 0.0:
            raise ValueError(f"fx_rates[{code!r}] must be strictly positive, got {value!r}")
        if code == "USD" and value != 1.0:
            raise ValueError("fx_rates['USD'] must be exactly 1.0 (USD is the base currency)")
        cleaned[code] = value
    return MappingProxyType(cleaned)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class AssetClass(StrEnum):
    """Coarse asset-class bucket used by universe selectors and risk clusters."""

    EQUITY_INDEX = "equity_index"
    RATES = "rates"
    ENERGY = "energy"
    METALS = "metals"
    FX = "fx"
    AGRICULTURE = "agriculture"
    CRYPTO = "crypto"
    UNKNOWN = "unknown"


class OrderSide(StrEnum):
    """Direction of an order/fill."""

    BUY = "BUY"
    SELL = "SELL"

    @property
    def multiplier(self) -> int:
        """Signed multiplier (+1 for buys, -1 for sells)."""
        return 1 if self is OrderSide.BUY else -1


class OrderType(StrEnum):
    """Supported order types (MKT/LMT plus IBKR algo and spread variants)."""

    MARKET = "MKT"
    LIMIT = "LMT"
    STOP = "STP"
    STOP_LIMIT = "STP_LMT"
    ADAPTIVE = "IBALGO"
    TWAP = "TWAP"
    VWAP = "VWAP"
    CALENDAR_SPREAD = "BAG"


class TimeInForce(StrEnum):
    """Order time-in-force policies."""

    DAY = "DAY"
    GTC = "GTC"
    IOC = "IOC"
    FOK = "FOK"
    GTD = "GTD"


class OrderStatus(StrEnum):
    """Lifecycle states reported by the broker / OMS."""

    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    UNKNOWN = "UNKNOWN"


class RoundingMode(StrEnum):
    """How fractional contracts are rounded to discrete contracts."""

    #: Round toward zero (conservative risk: never overshoot the target).
    FLOOR = "floor"
    #: Round to nearest integer (with a configurable rebalance buffer).
    NEAREST = "nearest"
    #: Round away from zero (aggressive; only used for explicit stress tests).
    CEIL = "ceil"


class RollTrigger(StrEnum):
    """Contract-roll trigger methodology."""

    VOLUME_OI_CROSSOVER = "volume_oi_crossover"
    CALENDAR_DAYS_BEFORE_FND = "calendar_days_before_fnd"
    DATABENTO_NATIVE = "databento_native"


class PriceAdjustment(StrEnum):
    """Continuous-series price adjustment methodology."""

    PANAMA = "panama"
    PROPORTIONAL = "proportional"
    UNADJUSTED = "unadjusted"


# ---------------------------------------------------------------------------
# Market data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class InstrumentSnapshot:
    """Point-in-time state of one futures root's *active* contract.

    ``price`` is the mark/last price of the front (or second-month) contract
    that the translator will use to convert notional into contracts.
    """

    instrument_id: InstrumentId
    root_symbol: str
    price: float
    asset_class: AssetClass = AssetClass.UNKNOWN
    expiration: date | None = None
    bid: float | None = None
    ask: float | None = None
    volume: float = 0.0
    open_interest: float = 0.0
    settlement: float | None = None
    features: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.instrument_id:
            raise ValueError("instrument_id must be a non-empty string")
        if not self.root_symbol:
            raise ValueError("root_symbol must be a non-empty string")
        object.__setattr__(self, "price", _validate_finite("price", self.price, allow_zero=False))
        if self.bid is not None:
            object.__setattr__(self, "bid", _validate_finite("bid", self.bid, allow_zero=False))
        if self.ask is not None:
            object.__setattr__(self, "ask", _validate_finite("ask", self.ask, allow_zero=False))
        if self.bid is not None and self.ask is not None and self.bid > self.ask:
            raise ValueError(f"bid {self.bid} > ask {self.ask} for {self.instrument_id}")
        object.__setattr__(self, "volume", _validate_finite("volume", self.volume))
        object.__setattr__(
            self, "open_interest", _validate_finite("open_interest", self.open_interest)
        )
        if self.volume < 0 or self.open_interest < 0:
            raise ValueError("volume and open_interest must be non-negative")
        if self.settlement is not None:
            object.__setattr__(self, "settlement", _validate_finite("settlement", self.settlement))
        frozen_features: dict[str, float] = {}
        for key, value in self.features.items():
            if not isinstance(key, str):
                raise TypeError(f"feature names must be strings, got {key!r}")
            frozen_features[key] = _validate_finite(f"feature[{key!r}]", value)
        object.__setattr__(self, "features", MappingProxyType(frozen_features))

    @property
    def mid(self) -> float:
        """Mid price when a two-sided quote is available, otherwise ``price``."""
        if self.bid is not None and self.ask is not None:
            return 0.5 * (self.bid + self.ask)
        return self.price

    @property
    def spread(self) -> float | None:
        """Absolute bid/ask spread, or ``None`` if a quote is unavailable."""
        if self.bid is None or self.ask is None:
            return None
        return self.ask - self.bid

    def feature(self, name: str, default: float = math.nan) -> float:
        """Return a named feature value (``default`` when missing)."""
        return self.features.get(name, default)


@dataclass(frozen=True, slots=True)
class MarketState:
    """Everything the ``w_t = R_t(T_t(A_t(S_t(X))))`` pipeline may observe at ``t``.

    The state deliberately carries no future information: every field is a
    point-in-time observation and the pipeline is forbidden from fetching more.
    ``returns`` / ``covariance`` are used by the risk overlays, ``features`` per
    instrument by selectors and timing overlays, ``metadata`` (VIX term slope,
    2Y/10Y curve, DXY, macro flags) by the regime and LLM overlays.
    """

    timestamp: datetime
    instruments: Mapping[InstrumentId, InstrumentSnapshot]
    equity: float | None = None
    covariance: NDArray[np.float64] | None = None
    covariance_instruments: tuple[InstrumentId, ...] = ()
    returns: Mapping[InstrumentId, NDArray[np.float64]] = field(default_factory=dict)
    regime: str | None = None
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)
    #: IBKR account base currency; ``equity`` is denominated in this currency.
    account_currency: str = "USD"
    #: ``currency -> USD per one unit`` rates used to normalise non-USD accounts.
    fx_rates: Mapping[str, float] = field(default_factory=dict)
    #: Bar timeframe of the state (canonical registry name; ``"1d"`` = daily).
    #: ``returns`` and ``covariance``/volatility features are in *bar* units, so
    #: annualisation must use ``bars_per_year(timeframe)`` (1d keeps 252).
    timeframe: str = "1d"

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", ensure_utc(self.timestamp))
        if not self.instruments:
            raise ValueError("MarketState requires at least one instrument snapshot")
        frozen: dict[InstrumentId, InstrumentSnapshot] = {}
        for instrument_id, snapshot in self.instruments.items():
            if instrument_id != snapshot.instrument_id:
                raise ValueError(
                    f"instrument key {instrument_id!r} does not match snapshot "
                    f"instrument_id {snapshot.instrument_id!r}"
                )
            frozen[instrument_id] = snapshot
        object.__setattr__(self, "instruments", MappingProxyType(frozen))
        if self.equity is not None:
            object.__setattr__(self, "equity", _validate_finite("equity", self.equity))
            if self.equity <= 0:
                raise ValueError("equity must be strictly positive")
        if self.covariance is not None:
            cov = np.array(self.covariance, dtype=np.float64, copy=True)
            if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
                raise ValueError(f"covariance must be square, got shape {cov.shape}")
            if len(self.covariance_instruments) != cov.shape[0]:
                raise ValueError(
                    "covariance_instruments length "
                    f"({len(self.covariance_instruments)}) must match covariance size "
                    f"({cov.shape[0]})"
                )
            if not np.all(np.isfinite(cov)):
                raise ValueError("covariance contains non-finite values")
            cov.setflags(write=False)
            object.__setattr__(self, "covariance", cov)
        elif self.covariance_instruments:
            raise ValueError("covariance_instruments given without a covariance matrix")
        frozen_returns: dict[InstrumentId, NDArray[np.float64]] = {}
        for instrument_id, series in self.returns.items():
            array = np.asarray(series, dtype=np.float64)
            if array.ndim != 1:
                raise ValueError(f"returns[{instrument_id!r}] must be a 1-D series")
            frozen_returns[instrument_id] = _readonly_array(array)
        object.__setattr__(self, "returns", MappingProxyType(frozen_returns))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        object.__setattr__(
            self,
            "account_currency",
            normalise_currency(self.account_currency, name="account_currency"),
        )
        object.__setattr__(self, "fx_rates", freeze_fx_rates(self.fx_rates))
        # Lazy import: configs/data modules import this contract module, and the
        # canonical timeframe registry is dependency-free, but keeping the import
        # inside __post_init__ avoids any import cycle at module load time.
        from src.data.timeframes import get_timeframe

        object.__setattr__(self, "timeframe", get_timeframe(self.timeframe).name)

    # -- convenience accessors -------------------------------------------------

    @property
    def ids(self) -> tuple[InstrumentId, ...]:
        """Instrument ids in insertion order."""
        return tuple(self.instruments)

    def __contains__(self, instrument_id: object) -> bool:
        return instrument_id in self.instruments

    def __len__(self) -> int:
        return len(self.instruments)

    def snapshot(self, instrument_id: InstrumentId) -> InstrumentSnapshot:
        """Return the snapshot for ``instrument_id`` (raising ``KeyError``)."""
        try:
            return self.instruments[instrument_id]
        except KeyError as exc:
            raise KeyError(
                f"instrument {instrument_id!r} is not part of the market state "
                f"({sorted(self.instruments)!r})"
            ) from exc

    def price(self, instrument_id: InstrumentId) -> float:
        """Mark price for ``instrument_id``."""
        return self.snapshot(instrument_id).price

    def feature(self, instrument_id: InstrumentId, name: str, default: float = math.nan) -> float:
        """Named feature for ``instrument_id``."""
        return self.snapshot(instrument_id).feature(name, default)

    def has(self, instrument_id: InstrumentId) -> bool:
        """Whether the state covers ``instrument_id``."""
        return instrument_id in self.instruments

    # -- base-currency normalisation ------------------------------------------

    def fx_rate_to_usd(self, currency: str | None = None) -> float:
        """USD value of one unit of ``currency`` (defaults to the account currency).

        USD itself is always 1.0. A non-USD currency without a configured rate
        raises a ``KeyError`` with actionable guidance: silently assuming a rate
        of 1.0 would mis-size every contract on a EUR/GBP/JPY account.
        """
        code = normalise_currency(
            currency if currency is not None else self.account_currency, name="currency"
        )
        if code == "USD":
            return 1.0
        try:
            return self.fx_rates[code]
        except KeyError:
            raise KeyError(
                f"no FX rate for account/contract currency {code!r}; provide "
                f"fx_rates[{code!r}] = USD per 1 {code} (available: {sorted(self.fx_rates)!r})"
            ) from None

    @property
    def equity_in_usd(self) -> float | None:
        """``equity`` converted into USD using the configured account FX rate."""
        if self.equity is None:
            return None
        return self.equity * self.fx_rate_to_usd()

    def converted_to_currency(
        self,
        account_currency: str,
        *,
        fx_rates: Mapping[str, float] | None = None,
    ) -> MarketState:
        """Return a copy denominated in ``account_currency`` (equity converted).

        The stored ``equity`` is first interpreted in the current account
        currency, converted through USD, then re-expressed in the target
        currency. Used when the runner's IBKR base currency differs from the
        state's recorded account currency.
        """
        target = normalise_currency(account_currency, name="account_currency")
        merged_rates: dict[str, float] = dict(self.fx_rates)
        if fx_rates:
            merged_rates.update(fx_rates)
        equity_usd = self.equity_in_usd
        new_equity: float | None = None
        if equity_usd is not None:
            if target == "USD":
                new_equity = equity_usd
            else:
                if target not in merged_rates:
                    raise KeyError(
                        f"cannot convert equity into {target!r}: no fx_rates[{target!r}]"
                    )
                new_equity = equity_usd / merged_rates[target]
        return replace(
            self,
            equity=new_equity,
            account_currency=target,
            fx_rates=merged_rates,
        )

    def subset(self, instrument_ids: Sequence[InstrumentId]) -> MarketState:
        """Return a new state restricted to ``instrument_ids`` (order preserved)."""
        unknown = [i for i in instrument_ids if i not in self.instruments]
        if unknown:
            raise KeyError(f"cannot subset to unknown instruments: {unknown!r}")
        selected = tuple(instrument_ids)
        instruments = {i: self.instruments[i] for i in selected}
        returns = {i: self.returns[i] for i in selected if i in self.returns}
        covariance: NDArray[np.float64] | None = None
        covariance_instruments: tuple[InstrumentId, ...] = ()
        if self.covariance is not None:
            positions = {name: idx for idx, name in enumerate(self.covariance_instruments)}
            missing = [i for i in selected if i not in positions]
            if missing:
                raise KeyError(f"covariance has no entry for {missing!r}")
            index = [positions[i] for i in selected]
            covariance = self.covariance[np.ix_(index, index)]
            covariance_instruments = selected
        return MarketState(
            timestamp=self.timestamp,
            instruments=instruments,
            equity=self.equity,
            covariance=covariance,
            covariance_instruments=covariance_instruments,
            returns=returns,
            regime=self.regime,
            metadata=self.metadata,
            account_currency=self.account_currency,
            fx_rates=self.fx_rates,
        )

    def covariance_submatrix(
        self, instrument_ids: Sequence[InstrumentId]
    ) -> NDArray[np.float64] | None:
        """Aligned covariance sub-matrix for ``instrument_ids`` (``None`` if absent)."""
        if self.covariance is None:
            return None
        positions = {name: idx for idx, name in enumerate(self.covariance_instruments)}
        missing = [i for i in instrument_ids if i not in positions]
        if missing:
            return None
        index = [positions[i] for i in instrument_ids]
        return self.covariance[np.ix_(index, index)]

    def annualized_vol(self, instrument_id: InstrumentId) -> float:
        """Annualised volatility for one instrument (``nan`` when unknown)."""
        if self.covariance is None:
            return math.nan
        try:
            index = self.covariance_instruments.index(instrument_id)
        except ValueError:
            return math.nan
        variance = float(self.covariance[index, index])
        return math.sqrt(variance) if variance > 0 else 0.0

    @classmethod
    def from_snapshots(
        cls,
        timestamp: datetime,
        snapshots: Sequence[InstrumentSnapshot],
        *,
        equity: float | None = None,
        regime: str | None = None,
        metadata: Mapping[str, MetadataValue] | None = None,
        account_currency: str = "USD",
        fx_rates: Mapping[str, float] | None = None,
    ) -> MarketState:
        """Build a state from a snapshot sequence."""
        if not snapshots:
            raise ValueError("from_snapshots requires at least one snapshot")
        return cls(
            timestamp=timestamp,
            instruments={snapshot.instrument_id: snapshot for snapshot in snapshots},
            equity=equity,
            regime=regime,
            metadata=metadata or {},
            account_currency=account_currency,
            fx_rates=fx_rates or {},
        )


# ---------------------------------------------------------------------------
# Portfolio weight vector (the FinRL-X interface contract)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WeightVector:
    """Immutable signed notional target weights ``w_t``.

    Positive weights are long notional, negative weights are short notional and
    ``sum(abs(w))`` is gross leverage. Every stage of the pipeline consumes and
    returns this type; the translator is the only component that converts it
    into discrete contracts.
    """

    weights: Mapping[InstrumentId, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "weights", _freeze_float_mapping(self.weights))

    # -- mapping protocol ------------------------------------------------------

    def __getitem__(self, instrument_id: InstrumentId) -> float:
        return self.weights[instrument_id]

    def __iter__(self) -> Iterator[InstrumentId]:
        return iter(self.weights)

    def __len__(self) -> int:
        return len(self.weights)

    def __contains__(self, instrument_id: object) -> bool:
        return instrument_id in self.weights

    def get(self, instrument_id: InstrumentId, default: float = 0.0) -> float:
        """Weight for ``instrument_id`` (``default`` when absent)."""
        return self.weights.get(instrument_id, default)

    def items(self) -> Iterator[tuple[InstrumentId, float]]:
        """Iterate ``(instrument, weight)`` pairs."""
        return iter(self.weights.items())

    def as_dict(self) -> dict[InstrumentId, float]:
        """Mutable copy of the weights."""
        return dict(self.weights)

    # -- risk / leverage statistics -------------------------------------------

    @property
    def gross(self) -> float:
        """Gross leverage ``sum(|w_i|)``."""
        return sum(abs(value) for value in self.weights.values())

    @property
    def net(self) -> float:
        """Net exposure ``sum(w_i)``."""
        return sum(self.weights.values())

    @property
    def l2(self) -> float:
        """Euclidean norm of the weight vector."""
        return math.sqrt(sum(value * value for value in self.weights.values()))

    @property
    def active(self) -> dict[InstrumentId, float]:
        """Weights whose magnitude is non-negligible."""
        return {key: value for key, value in self.weights.items() if abs(value) > 1e-12}

    def is_flat(self, tolerance: float = 1e-9) -> bool:
        """Whether every weight is (numerically) zero."""
        return all(abs(value) <= tolerance for value in self.weights.values())

    def max_abs(self) -> float:
        """Largest absolute weight (0 for an empty vector)."""
        return max((abs(value) for value in self.weights.values()), default=0.0)

    # -- transformations -------------------------------------------------------

    def scaled(self, factor: float) -> WeightVector:
        """Return ``factor * w``."""
        factor = _validate_finite("factor", factor)
        return WeightVector({key: value * factor for key, value in self.weights.items()})

    def shifted(self, delta: Mapping[InstrumentId, float]) -> WeightVector:
        """Return ``w + delta`` over the union of keys."""
        merged = dict(self.weights)
        for key, value in delta.items():
            merged[key] = merged.get(key, 0.0) + _validate_finite(f"delta[{key!r}]", value)
        return WeightVector(merged)

    def subset(self, instrument_ids: Sequence[InstrumentId] | set[InstrumentId]) -> WeightVector:
        """Restrict to ``instrument_ids`` (missing ids are treated as zero)."""
        allowed = set(instrument_ids)
        return WeightVector({k: v for k, v in self.weights.items() if k in allowed})

    def reindexed(self, instrument_ids: Sequence[InstrumentId]) -> WeightVector:
        """Reindex to exactly ``instrument_ids`` inserting zeros where needed."""
        return WeightVector({i: self.get(i, 0.0) for i in instrument_ids})

    def clipped(self, max_abs_weight: float) -> WeightVector:
        """Clip every |w_i| to ``max_abs_weight``."""
        limit = _validate_finite("max_abs_weight", max_abs_weight)
        if limit < 0:
            raise ValueError("max_abs_weight must be non-negative")
        return WeightVector(
            {key: max(-limit, min(limit, value)) for key, value in self.weights.items()}
        )

    def normalized_gross(self, target_gross: float) -> WeightVector:
        """Scale the vector so that gross leverage equals ``target_gross``."""
        target = _validate_finite("target_gross", target_gross)
        if target < 0:
            raise ValueError("target_gross must be non-negative")
        current = self.gross
        if current <= 1e-12 or target == 0.0:
            return WeightVector(dict.fromkeys(self.weights, 0.0))
        return self.scaled(target / current)

    def to_array(self, order: Sequence[InstrumentId]) -> NDArray[np.float64]:
        """Dense vector aligned to ``order`` (absent instruments map to 0.0)."""
        return np.array([self.get(instrument_id, 0.0) for instrument_id in order], dtype=np.float64)

    # -- dunder arithmetic -----------------------------------------------------

    def __add__(self, other: WeightVector | Mapping[InstrumentId, float]) -> WeightVector:
        if isinstance(other, WeightVector):
            return self.shifted(other.weights)
        if isinstance(other, Mapping):
            return self.shifted(other)
        return NotImplemented

    def __sub__(self, other: WeightVector | Mapping[InstrumentId, float]) -> WeightVector:
        if isinstance(other, WeightVector):
            return self.shifted({k: -v for k, v in other.weights.items()})
        if isinstance(other, Mapping):
            return self.shifted({k: -v for k, v in other.items()})
        return NotImplemented

    def __mul__(self, factor: float) -> WeightVector:
        return self.scaled(float(factor))

    def __rmul__(self, factor: float) -> WeightVector:
        return self.scaled(float(factor))

    def __neg__(self) -> WeightVector:
        return self.scaled(-1.0)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, WeightVector):
            return self.as_dict() == other.as_dict()
        if isinstance(other, Mapping):
            return self.as_dict() == dict(other)
        return NotImplemented

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.weights.items())))

    def __repr__(self) -> str:
        body = ", ".join(f"{key}={value:+.4f}" for key, value in sorted(self.weights.items()))
        return f"WeightVector(gross={self.gross:.4f}, net={self.net:+.4f}, {{{body}}})"

    # -- constructors ----------------------------------------------------------

    @classmethod
    def empty(cls) -> WeightVector:
        """A flat (all-zero) vector with no instruments."""
        return cls({})

    @classmethod
    def from_mapping(cls, weights: Mapping[InstrumentId, float]) -> WeightVector:
        """Build from an ``instrument -> weight`` mapping."""
        return cls(weights)

    @classmethod
    def from_array(
        cls, instrument_ids: Sequence[InstrumentId], values: Sequence[float] | NDArray[np.float64]
    ) -> WeightVector:
        """Build from a dense array aligned to ``instrument_ids``."""
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (len(instrument_ids),):
            raise ValueError(
                f"values shape {array.shape} does not match {len(instrument_ids)} instruments"
            )
        return cls(
            {
                instrument_id: float(value)
                for instrument_id, value in zip(instrument_ids, array, strict=True)
            }
        )

    @classmethod
    def equal_weight(
        cls, instrument_ids: Sequence[InstrumentId], gross: float = 1.0
    ) -> WeightVector:
        """Equal-weight vector with total gross leverage ``gross``."""
        if not instrument_ids:
            return cls.empty()
        if len(set(instrument_ids)) != len(instrument_ids):
            raise ValueError("instrument_ids must be unique")
        per_instrument = float(gross) / len(instrument_ids)
        return cls(dict.fromkeys(instrument_ids, per_instrument))


# ---------------------------------------------------------------------------
# Discrete futures contract targets (output of the ContractTranslator)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ContractTarget:
    """A discrete, signed number of contracts for one futures instrument."""

    instrument_id: InstrumentId
    quantity: int
    price: float
    multiplier: float
    expiration: date | None = None
    initial_margin: float = 0.0
    maintenance_margin: float = 0.0
    contract_symbol: str | None = None
    con_id: str | None = None
    asset_class: AssetClass = AssetClass.UNKNOWN

    def __post_init__(self) -> None:
        if not self.instrument_id:
            raise ValueError("instrument_id must be a non-empty string")
        if isinstance(self.quantity, bool) or not isinstance(self.quantity, int):
            raise TypeError(f"quantity must be an int, got {type(self.quantity)!r}")
        object.__setattr__(self, "price", _validate_finite("price", self.price, allow_zero=False))
        object.__setattr__(
            self, "multiplier", _validate_finite("multiplier", self.multiplier, allow_zero=False)
        )
        if self.multiplier <= 0:
            raise ValueError("multiplier must be strictly positive")
        object.__setattr__(
            self, "initial_margin", _validate_finite("initial_margin", self.initial_margin)
        )
        object.__setattr__(
            self,
            "maintenance_margin",
            _validate_finite("maintenance_margin", self.maintenance_margin),
        )
        if self.initial_margin < 0 or self.maintenance_margin < 0:
            raise ValueError("margins must be non-negative")

    @property
    def signed_notional(self) -> float:
        """Signed notional exposure ``N * P * M``."""
        return self.quantity * self.price * self.multiplier

    @property
    def abs_notional(self) -> float:
        """Absolute notional exposure ``|N| * P * M``."""
        return abs(self.signed_notional)

    @property
    def initial_margin_required(self) -> float:
        """Initial margin required for the position."""
        return abs(self.quantity) * self.initial_margin

    @property
    def maintenance_margin_required(self) -> float:
        """Maintenance margin required for the position."""
        return abs(self.quantity) * self.maintenance_margin

    @property
    def side(self) -> OrderSide:
        """Order side implied by the target quantity."""
        return OrderSide.BUY if self.quantity >= 0 else OrderSide.SELL

    def to_dict(self) -> dict[str, ScalarValue]:
        """JSON-serialisable representation."""
        return {
            "instrument_id": self.instrument_id,
            "quantity": self.quantity,
            "price": self.price,
            "multiplier": self.multiplier,
            "expiration": self.expiration.isoformat() if self.expiration else None,
            "initial_margin": self.initial_margin,
            "maintenance_margin": self.maintenance_margin,
            "contract_symbol": self.contract_symbol,
            "con_id": self.con_id,
            "asset_class": self.asset_class.value,
        }


@dataclass(frozen=True, slots=True)
class ContractTargetSet:
    """Immutable collection of :class:`ContractTarget` with portfolio aggregates."""

    targets: tuple[ContractTarget, ...] = ()

    def __post_init__(self) -> None:
        seen: set[InstrumentId] = set()
        for target in self.targets:
            if target.instrument_id in seen:
                raise ValueError(f"duplicate contract target for {target.instrument_id!r}")
            seen.add(target.instrument_id)

    def __iter__(self) -> Iterator[ContractTarget]:
        return iter(self.targets)

    def __len__(self) -> int:
        return len(self.targets)

    def __bool__(self) -> bool:
        return any(target.quantity != 0 for target in self.targets)

    def get(self, instrument_id: InstrumentId) -> ContractTarget | None:
        """Target for ``instrument_id`` if present."""
        for target in self.targets:
            if target.instrument_id == instrument_id:
                return target
        return None

    @property
    def by_instrument(self) -> Mapping[InstrumentId, ContractTarget]:
        """Mapping from instrument id to target."""
        return MappingProxyType({target.instrument_id: target for target in self.targets})

    @property
    def quantities(self) -> Mapping[InstrumentId, int]:
        """Mapping from instrument id to signed contract quantity."""
        return MappingProxyType({target.instrument_id: target.quantity for target in self.targets})

    @property
    def gross_notional(self) -> float:
        """Gross notional exposure across all legs."""
        return sum(target.abs_notional for target in self.targets)

    @property
    def net_notional(self) -> float:
        """Net notional exposure across all legs."""
        return sum(target.signed_notional for target in self.targets)

    @property
    def total_initial_margin(self) -> float:
        """Total initial margin required across all legs."""
        return sum(target.initial_margin_required for target in self.targets)

    @property
    def total_maintenance_margin(self) -> float:
        """Total maintenance margin required across all legs."""
        return sum(target.maintenance_margin_required for target in self.targets)

    def nonzero(self) -> ContractTargetSet:
        """Targets with a non-zero quantity."""
        return ContractTargetSet(tuple(t for t in self.targets if t.quantity != 0))

    @classmethod
    def empty(cls) -> ContractTargetSet:
        """An empty target set (fully flat)."""
        return cls(())


# ---------------------------------------------------------------------------
# Account / position / reconciliation containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Position:
    """A broker position in one futures instrument."""

    instrument_id: InstrumentId
    quantity: int
    average_price: float
    market_price: float
    multiplier: float
    expiration: date | None = None
    initial_margin: float = 0.0
    maintenance_margin: float = 0.0
    contract_symbol: str | None = None

    def __post_init__(self) -> None:
        if not self.instrument_id:
            raise ValueError("instrument_id must be a non-empty string")
        if isinstance(self.quantity, bool) or not isinstance(self.quantity, int):
            raise TypeError("quantity must be an int")
        object.__setattr__(
            self,
            "market_price",
            _validate_finite("market_price", self.market_price, allow_zero=False),
        )
        object.__setattr__(
            self,
            "average_price",
            _validate_finite("average_price", self.average_price, allow_zero=False),
        )
        object.__setattr__(
            self, "multiplier", _validate_finite("multiplier", self.multiplier, allow_zero=False)
        )
        if self.multiplier <= 0:
            raise ValueError("multiplier must be strictly positive")

    @property
    def signed_notional(self) -> float:
        """Signed market value of the position."""
        return self.quantity * self.market_price * self.multiplier

    @property
    def unrealized_pnl(self) -> float:
        """Mark-to-market unrealised PnL."""
        return self.quantity * (self.market_price - self.average_price) * self.multiplier

    @property
    def initial_margin_required(self) -> float:
        """Initial margin consumed by the position."""
        return abs(self.quantity) * self.initial_margin


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    """Broker account valuation at one instant."""

    timestamp: datetime
    net_liquidation: float
    cash_balance: float = 0.0
    buying_power: float = 0.0
    initial_margin_used: float = 0.0
    maintenance_margin_used: float = 0.0
    excess_liquidity: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", ensure_utc(self.timestamp))
        object.__setattr__(
            self,
            "net_liquidation",
            _validate_finite("net_liquidation", self.net_liquidation, allow_zero=False),
        )
        if self.net_liquidation <= 0:
            raise ValueError("net_liquidation must be strictly positive")
        for name in (
            "cash_balance",
            "buying_power",
            "initial_margin_used",
            "maintenance_margin_used",
            "excess_liquidity",
        ):
            object.__setattr__(self, name, _validate_finite(name, getattr(self, name)))

    @property
    def margin_utilization(self) -> float:
        """Initial margin utilisation as a fraction of net liquidation value."""
        return self.initial_margin_used / self.net_liquidation


# ---------------------------------------------------------------------------
# Orders / fills / reconciliation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OrderEvent:
    """An order instruction emitted by the OMS (or the simulated broker)."""

    order_id: str
    timestamp: datetime
    instrument_id: InstrumentId
    side: OrderSide
    quantity: int
    order_type: OrderType = OrderType.MARKET
    time_in_force: TimeInForce = TimeInForce.DAY
    limit_price: float | None = None
    stop_price: float | None = None
    contract_symbol: str | None = None
    expiration: date | None = None
    parent_order_id: str | None = None
    tag: str = ""

    def __post_init__(self) -> None:
        if not self.order_id:
            raise ValueError("order_id must be a non-empty string")
        object.__setattr__(self, "timestamp", ensure_utc(self.timestamp))
        if not self.instrument_id:
            raise ValueError("instrument_id must be a non-empty string")
        if isinstance(self.quantity, bool) or not isinstance(self.quantity, int):
            raise TypeError("quantity must be an int")
        if self.quantity <= 0:
            raise ValueError("quantity must be strictly positive; use OrderSide for direction")
        if self.limit_price is not None:
            object.__setattr__(
                self,
                "limit_price",
                _validate_finite("limit_price", self.limit_price, allow_zero=False),
            )
        if self.stop_price is not None:
            object.__setattr__(
                self,
                "stop_price",
                _validate_finite("stop_price", self.stop_price, allow_zero=False),
            )
        if self.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT) and self.limit_price is None:
            raise ValueError(f"{self.order_type.value} order requires limit_price")
        if self.order_type in (OrderType.STOP, OrderType.STOP_LIMIT) and self.stop_price is None:
            raise ValueError(f"{self.order_type.value} order requires stop_price")

    @property
    def signed_quantity(self) -> int:
        """Positive for buys, negative for sells."""
        return self.side.multiplier * self.quantity

    @classmethod
    def market(
        cls,
        *,
        order_id: str,
        timestamp: datetime,
        instrument_id: InstrumentId,
        side: OrderSide,
        quantity: int,
        contract_symbol: str | None = None,
        expiration: date | None = None,
        tag: str = "",
    ) -> OrderEvent:
        """Convenience constructor for a market order."""
        return cls(
            order_id=order_id,
            timestamp=timestamp,
            instrument_id=instrument_id,
            side=side,
            quantity=quantity,
            order_type=OrderType.MARKET,
            contract_symbol=contract_symbol,
            expiration=expiration,
            tag=tag,
        )


@dataclass(frozen=True, slots=True)
class Fill:
    """Execution report for a (possibly partial) fill."""

    order_id: str
    instrument_id: InstrumentId
    side: OrderSide
    quantity: int
    price: float
    timestamp: datetime
    commission: float = 0.0
    slippage_ticks: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", ensure_utc(self.timestamp))
        object.__setattr__(self, "price", _validate_finite("price", self.price, allow_zero=False))
        object.__setattr__(self, "commission", _validate_finite("commission", self.commission))
        object.__setattr__(
            self, "slippage_ticks", _validate_finite("slippage_ticks", self.slippage_ticks)
        )
        if self.quantity <= 0:
            raise ValueError("fill quantity must be strictly positive")

    @property
    def signed_quantity(self) -> int:
        """Positive for buys, negative for sells."""
        return self.side.multiplier * self.quantity


@dataclass(frozen=True, slots=True)
class OrderReceipt:
    """Normalised broker acknowledgement for one order."""

    order_id: str
    status: OrderStatus
    filled_quantity: int = 0
    average_price: float | None = None
    commission: float = 0.0
    message: str = ""
    timestamp: datetime | None = None

    def __post_init__(self) -> None:
        if self.timestamp is not None:
            object.__setattr__(self, "timestamp", ensure_utc(self.timestamp))
        if self.filled_quantity < 0:
            raise ValueError("filled_quantity must be non-negative")
        object.__setattr__(self, "commission", _validate_finite("commission", self.commission))
        if self.average_price is not None:
            object.__setattr__(
                self,
                "average_price",
                _validate_finite("average_price", self.average_price, allow_zero=False),
            )


@dataclass(frozen=True, slots=True)
class ReconcileReport:
    """Result of comparing internal expected state against the broker's state."""

    timestamp: datetime
    ok: bool
    expected_positions: Mapping[InstrumentId, int] = field(default_factory=dict)
    actual_positions: Mapping[InstrumentId, int] = field(default_factory=dict)
    drift: Mapping[InstrumentId, int] = field(default_factory=dict)
    halted: bool = False
    message: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", ensure_utc(self.timestamp))
        object.__setattr__(
            self, "expected_positions", MappingProxyType(dict(self.expected_positions))
        )
        object.__setattr__(self, "actual_positions", MappingProxyType(dict(self.actual_positions)))
        object.__setattr__(self, "drift", MappingProxyType(dict(self.drift)))


# ---------------------------------------------------------------------------
# Pipeline trace (FinRL-X waterfall inspector contract)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """One full ``w_t = R_t(T_t(A_t(S_t(X))))`` rebalance cycle, step by step."""

    timestamp: datetime
    universe: tuple[InstrumentId, ...]
    base_weights: WeightVector
    timing_weights: WeightVector
    risk_weights: WeightVector
    final_weights: WeightVector
    targets: ContractTargetSet | None = None
    diagnostics: Mapping[str, MetadataValue] = field(default_factory=dict)
    elapsed_ms: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", ensure_utc(self.timestamp))
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))
        if self.elapsed_ms is not None and self.elapsed_ms < 0:
            raise ValueError("elapsed_ms must be non-negative")


# ---------------------------------------------------------------------------
# LLM risk assessment (Tier A contract, produced by the risk overlay)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RiskAssessment:
    """Structured LLM risk output for one futures root.

    ``semantic_risk_score`` follows the FinRL Risk-First convention: 1 = benign,
    5 = imminent macro shock / geopolitical tail event / extreme illiquidity.
    """

    instrument_id: InstrumentId
    timestamp: datetime
    sentiment_score: float
    semantic_risk_score: float
    confidence: float
    reasoning_summary: str = ""
    source: str = "llm"
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.instrument_id:
            raise ValueError("instrument_id must be a non-empty string")
        object.__setattr__(self, "timestamp", ensure_utc(self.timestamp))
        object.__setattr__(
            self, "sentiment_score", _validate_finite("sentiment_score", self.sentiment_score)
        )
        object.__setattr__(
            self,
            "semantic_risk_score",
            _validate_finite("semantic_risk_score", self.semantic_risk_score),
        )
        object.__setattr__(self, "confidence", _validate_finite("confidence", self.confidence))
        if not 1.0 <= self.sentiment_score <= 5.0:
            raise ValueError("sentiment_score must be within [1, 5]")
        if not 1.0 <= self.semantic_risk_score <= 5.0:
            raise ValueError("semantic_risk_score must be within [1, 5]")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be within [0, 1]")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def is_critical(self) -> bool:
        """Whether the assessment is above the critical semantic-risk threshold."""
        return self.semantic_risk_score >= 4.5

    def to_dict(self) -> dict[str, ScalarValue]:
        """JSON-serialisable representation."""
        return {
            "instrument_id": self.instrument_id,
            "timestamp": self.timestamp.isoformat(),
            "sentiment_score": self.sentiment_score,
            "semantic_risk_score": self.semantic_risk_score,
            "confidence": self.confidence,
            "reasoning_summary": self.reasoning_summary,
            "source": self.source,
        }


# ---------------------------------------------------------------------------
# Bar / roll primitives (data layer + simulator contract)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Bar:
    """One OHLCV bar for one specific (instrument, expiry) contract."""

    timestamp: datetime
    instrument_id: InstrumentId
    root_symbol: str
    expiration: date | None
    open: float
    high: float
    low: float
    close: float
    volume: float
    open_interest: float = 0.0
    settlement: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", ensure_utc(self.timestamp))
        for name in ("open", "high", "low", "close"):
            object.__setattr__(self, name, _validate_finite(name, getattr(self, name)))
        if not self.low <= min(self.open, self.close) <= max(self.open, self.close) <= self.high:
            raise ValueError(f"inconsistent OHLC bar for {self.instrument_id}: {self!r}")
        object.__setattr__(self, "volume", _validate_finite("volume", self.volume))
        object.__setattr__(
            self, "open_interest", _validate_finite("open_interest", self.open_interest)
        )
        if self.volume < 0 or self.open_interest < 0:
            raise ValueError("volume and open_interest must be non-negative")
        if self.settlement is not None:
            object.__setattr__(self, "settlement", _validate_finite("settlement", self.settlement))


@dataclass(frozen=True, slots=True)
class RollEvent:
    """A scheduled/executed roll from the front to the next contract."""

    root_symbol: str
    roll_datetime: datetime
    from_expiration: date
    to_expiration: date
    trigger: RollTrigger
    from_price: float
    to_price: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "roll_datetime", ensure_utc(self.roll_datetime))
        object.__setattr__(self, "from_price", _validate_finite("from_price", self.from_price))
        object.__setattr__(self, "to_price", _validate_finite("to_price", self.to_price))
        if self.to_expiration <= self.from_expiration:
            raise ValueError("to_expiration must be after from_expiration")

    @property
    def roll_gap(self) -> float:
        """Price gap ``F_next - F_front`` at the roll instant."""
        return self.to_price - self.from_price

    @property
    def roll_yield_annualized(self) -> float:
        """Annualised roll yield implied by the gap and the expiry distance."""
        days = max((self.to_expiration - self.from_expiration).days, 1)
        if self.from_price <= 0:
            return 0.0
        return -(self.roll_gap / self.from_price) * (365.0 / days)
