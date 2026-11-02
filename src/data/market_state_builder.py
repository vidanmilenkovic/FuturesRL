"""Point-in-time :class:`~src.core.types.MarketState` construction.

:class:`MarketStateBuilder` is the bridge between the data layer
(bars/features) and the FinRL-X strategy pipeline. At a decision timestamp ``t``
it selects **only** rows with ``timestamp_utc <= t`` and produces an immutable
:class:`~src.core.types.MarketState` containing:

* one :class:`~src.core.types.InstrumentSnapshot` per futures root (price, bid /
  ask, expiry, asset class and the canonical feature set),
* an EWMA (RiskMetrics-style) annualised covariance matrix with an explicit
  ``covariance_instruments`` ordering,
* the trailing return window per root,
* account equity and macro metadata (VIX term slope, 2s10s curve, DXY,
  turbulence, ...).

Point-in-time guarantee
-----------------------
Rows strictly after ``t`` are dropped *before* any aggregation. Appending or
mutating future rows therefore cannot change the state built at ``t`` — the
backtest engine may safely run this builder over a frame that extends beyond the
simulation end.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import date, datetime

import numpy as np
import polars as pl
from configs.instruments import InstrumentsConfig, InstrumentSpec, load_instruments
from numpy.typing import NDArray

from src.core.types import (
    AssetClass,
    InstrumentId,
    InstrumentSnapshot,
    MarketState,
    MetadataValue,
    ensure_utc,
)
from src.data import schema as s
from src.data.feature_processor import FeatureProcessor

__all__ = ["MarketStateBuilder"]


class MarketStateBuilder:
    """Convert a bar/feature frame at ``t`` into an immutable ``MarketState``.

    Parameters
    ----------
    instrument_master:
        Instrument master used for asset classes (unknown roots degrade to
        :attr:`~src.core.types.AssetClass.UNKNOWN`).
    feature_processor:
        Processor used when the input frame does not already carry the
        canonical features (raw bars are accepted). Defaults to a fresh
        :class:`~src.data.feature_processor.FeatureProcessor`.
    covariance_halflife, covariance_window, covariance_min_periods:
        EWMA covariance estimation controls (all use strictly past returns).
    returns_window:
        Number of trailing 1-day returns exposed per instrument in
        ``MarketState.returns``.
    annualize_covariance:
        Multiply the daily EWMA covariance by 252 (annualised, as required by
        the volatility-targeting risk overlay).
    initial_equity:
        Default equity assigned to every state when the caller does not pass
        ``equity=``.
    """

    def __init__(
        self,
        *,
        instrument_master: InstrumentsConfig | None = None,
        feature_processor: FeatureProcessor | None = None,
        covariance_halflife: float = 20.0,
        covariance_window: int = 252,
        covariance_min_periods: int = 20,
        returns_window: int = 252,
        annualize_covariance: bool = True,
        initial_equity: float | None = None,
    ) -> None:
        if covariance_halflife <= 0:
            raise ValueError("covariance_halflife must be strictly positive")
        if covariance_window < 2:
            raise ValueError("covariance_window must be >= 2")
        if covariance_min_periods < 2:
            raise ValueError("covariance_min_periods must be >= 2")
        if returns_window < 1:
            raise ValueError("returns_window must be >= 1")
        if initial_equity is not None and initial_equity <= 0:
            raise ValueError("initial_equity must be strictly positive")
        self._master = instrument_master
        self._feature_processor = feature_processor
        self._covariance_halflife = float(covariance_halflife)
        self._covariance_window = covariance_window
        self._covariance_min_periods = covariance_min_periods
        self._returns_window = returns_window
        self._annualize_covariance = annualize_covariance
        self._initial_equity = initial_equity

    # -- properties ------------------------------------------------------------

    @property
    def instrument_master(self) -> InstrumentsConfig:
        """Instrument master used for asset-class lookups."""
        if self._master is None:
            self._master = load_instruments()
        return self._master

    @property
    def feature_processor(self) -> FeatureProcessor:
        """Processor used to derive features when the input frame lacks them."""
        if self._feature_processor is None:
            self._feature_processor = FeatureProcessor(instrument_master=self._master)
        return self._feature_processor

    # -- public API ------------------------------------------------------------

    def build(
        self,
        frame: pl.DataFrame,
        timestamp: datetime,
        *,
        equity: float | None = None,
        regime: str | None = None,
        metadata: Mapping[str, MetadataValue] | None = None,
    ) -> MarketState:
        """Build the immutable state observable at ``timestamp`` (inclusive)."""
        prepared = self._prepare(frame)
        return self._build_prepared(
            prepared, timestamp, equity=equity, regime=regime, metadata=metadata
        )

    def build_series(
        self, frame: pl.DataFrame, timestamps: Sequence[datetime]
    ) -> list[MarketState]:
        """Build one state per timestamp in ``timestamps`` (features computed once)."""
        prepared = self._prepare(frame)
        return [self._build_prepared(prepared, timestamp) for timestamp in timestamps]

    def timestamps(self, frame: pl.DataFrame) -> tuple[datetime, ...]:
        """Sorted unique timestamps present in ``frame``."""
        if frame.is_empty():
            return ()
        return tuple(
            sorted(
                {
                    value
                    for value in frame[s.TIMESTAMP_COLUMN].cast(pl.Datetime("us", "UTC")).to_list()
                    if isinstance(value, datetime)
                }
            )
        )

    def latest_timestamp(self, frame: pl.DataFrame) -> datetime:
        """Latest timestamp present in ``frame`` (``ValueError`` when empty)."""
        stamps = self.timestamps(frame)
        if not stamps:
            raise ValueError("cannot derive a timestamp from an empty frame")
        return stamps[-1]

    # -- preparation -----------------------------------------------------------

    def _prepare(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Normalise the input frame, computing features only when required."""
        if not isinstance(frame, pl.DataFrame):
            raise TypeError(f"frame must be a polars DataFrame, got {type(frame)!r}")
        if frame.is_empty():
            raise ValueError("MarketStateBuilder received an empty frame")
        has_features = all(name in frame.columns for name in s.REQUIRED_FEATURES)
        if has_features:
            return s.normalise_timestamp_column(frame, s.TIMESTAMP_COLUMN)
        return self.feature_processor.compute(frame)

    def _point_in_time(self, prepared: pl.DataFrame, timestamp: datetime) -> pl.DataFrame:
        """Rows observable at ``timestamp`` (inclusive), sorted by root/timestamp."""
        moment = ensure_utc(timestamp)
        point_in_time = prepared.filter(pl.col(s.TIMESTAMP_COLUMN) <= pl.lit(moment))
        if point_in_time.is_empty():
            raise ValueError(
                f"no rows at or before {moment.isoformat()}; the frame starts later "
                "than the requested decision time"
            )
        return point_in_time.sort([s.ROOT_SYMBOL_COLUMN, s.TIMESTAMP_COLUMN])

    def _build_prepared(
        self,
        prepared: pl.DataFrame,
        timestamp: datetime,
        *,
        equity: float | None = None,
        regime: str | None = None,
        metadata: Mapping[str, MetadataValue] | None = None,
    ) -> MarketState:
        moment = ensure_utc(timestamp)
        point_in_time = self._point_in_time(prepared, moment)
        latest = point_in_time.unique(
            subset=[s.ROOT_SYMBOL_COLUMN], keep="last", maintain_order=True
        )
        snapshots = self._snapshots(latest)
        returns = self._returns(point_in_time)
        covariance, covariance_instruments = self._covariance(point_in_time)
        state_metadata = self._metadata(latest, point_in_time)
        if metadata:
            state_metadata.update(metadata)
        resolved_equity = equity if equity is not None else self._initial_equity
        return MarketState(
            timestamp=moment,
            instruments={snapshot.instrument_id: snapshot for snapshot in snapshots},
            equity=resolved_equity,
            covariance=covariance,
            covariance_instruments=covariance_instruments,
            returns=returns,
            regime=regime,
            metadata=state_metadata,
        )

    # -- snapshots -------------------------------------------------------------

    def _snapshots(self, latest: pl.DataFrame) -> tuple[InstrumentSnapshot, ...]:
        """Build one :class:`InstrumentSnapshot` per root from its latest row."""
        snapshots: list[InstrumentSnapshot] = []
        for record in latest.iter_rows(named=True):
            root = str(record[s.ROOT_SYMBOL_COLUMN])
            spec = self._spec_or_none(root)
            features = self._feature_dict(record)
            price = float(record[s.CLOSE_COLUMN])
            bid = self._optional_float(record.get(s.BID_PRICE_COLUMN))
            ask = self._optional_float(record.get(s.ASK_PRICE_COLUMN))
            if bid is not None and ask is not None and bid > ask:
                bid = ask = None
            expiration = record.get(s.EXPIRATION_COLUMN)
            snapshots.append(
                InstrumentSnapshot(
                    instrument_id=root,
                    root_symbol=root,
                    price=price,
                    asset_class=spec.asset_class if spec is not None else AssetClass.UNKNOWN,
                    expiration=expiration if isinstance(expiration, date) else None,
                    bid=bid,
                    ask=ask,
                    volume=self._float_or_zero(record.get(s.VOLUME_COLUMN)),
                    open_interest=self._float_or_zero(record.get(s.OPEN_INTEREST_COLUMN)),
                    settlement=self._optional_float(record.get(s.SETTLEMENT_COLUMN)),
                    features=features,
                )
            )
        return tuple(snapshots)

    def _feature_dict(self, record: Mapping[str, object]) -> dict[str, float]:
        """Finite canonical features of one row (non-finite values are dropped)."""
        features: dict[str, float] = {}
        for name in s.SNAPSHOT_FEATURES:
            value = self._optional_float(record.get(name))
            if value is not None:
                features[name] = value
        return features

    def _spec_or_none(self, root: str) -> InstrumentSpec | None:
        master = self.instrument_master
        return master.get(root) if master.has(root) else None

    @staticmethod
    def _optional_float(value: object) -> float | None:
        """Coerce a value to a finite float (``None`` otherwise)."""
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        result = float(value)
        return result if math.isfinite(result) else None

    @staticmethod
    def _float_or_zero(value: object) -> float:
        """Coerce a value to a finite non-negative float (default ``0.0``)."""
        if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
            return 0.0
        result = float(value)
        return result if math.isfinite(result) else 0.0

    # -- returns / covariance --------------------------------------------------

    def _returns(self, point_in_time: pl.DataFrame) -> dict[InstrumentId, NDArray[np.float64]]:
        """Trailing 1-day return series per instrument."""
        returns: dict[InstrumentId, NDArray[np.float64]] = {}
        for key, group in point_in_time.partition_by(s.ROOT_SYMBOL_COLUMN, as_dict=True).items():
            root = str(key[0] if isinstance(key, tuple) else key)
            ordered = group.sort(s.TIMESTAMP_COLUMN)
            if s.RET_1D in ordered.columns:
                values = np.asarray(ordered[s.RET_1D].cast(pl.Float64).to_numpy(), dtype=np.float64)
            else:  # pragma: no cover - prepared features always carry ret_1d
                close = np.asarray(
                    ordered[s.CLOSE_COLUMN].cast(pl.Float64).to_numpy(), dtype=np.float64
                )
                values = np.concatenate([[np.nan], close[1:] / close[:-1] - 1.0])
            finite = values[np.isfinite(values)]
            returns[root] = finite[-self._returns_window :]
        return returns

    def _covariance(
        self, point_in_time: pl.DataFrame
    ) -> tuple[NDArray[np.float64] | None, tuple[InstrumentId, ...]]:
        """EWMA covariance of the trailing cross-sectional return matrix."""
        pivot = point_in_time.pivot(
            on=s.ROOT_SYMBOL_COLUMN,
            index=s.TIMESTAMP_COLUMN,
            values=s.RET_1D,
            aggregate_function="first",
        ).sort(s.TIMESTAMP_COLUMN)
        roots = tuple(str(column) for column in pivot.columns if column != s.TIMESTAMP_COLUMN)
        if not roots:
            return (None, ())
        matrix = np.asarray(pivot.select(list(roots)).to_numpy(), dtype=np.float64)
        matrix = matrix[np.isfinite(matrix).all(axis=1)]
        if matrix.shape[0] < self._covariance_min_periods:
            return (None, ())
        window = matrix[-self._covariance_window :]
        mean = window.mean(axis=0)
        centred = window - mean
        decay = 0.5 ** (1.0 / self._covariance_halflife)
        weights = decay ** np.arange(window.shape[0] - 1, -1, -1, dtype=np.float64)
        weights = weights / weights.sum()
        covariance = (centred * weights[:, None]).T @ centred
        if self._annualize_covariance:
            covariance = covariance * float(_TRADING_DAYS)
        covariance = (covariance + covariance.T) / 2.0
        return (covariance, roots)

    # -- metadata --------------------------------------------------------------

    def _metadata(
        self, latest: pl.DataFrame, point_in_time: pl.DataFrame
    ) -> dict[str, MetadataValue]:
        """Scalar macro/regime metadata attached to the state."""
        metadata: dict[str, MetadataValue] = {}
        if latest.height:
            record = latest.row(0, named=True)
            for name in s.MACRO_FEATURES:
                value = self._optional_float(record.get(name))
                if value is not None:
                    metadata[name] = value
            turbulence = self._optional_float(record.get(s.TURBULENCE))
            if turbulence is not None:
                metadata[s.TURBULENCE] = turbulence
            roots = latest[s.ROOT_SYMBOL_COLUMN].cast(pl.Utf8).to_list()
            metadata["universe"] = tuple(str(root) for root in roots)
        metadata["point_in_time_rows"] = point_in_time.height
        return metadata


#: Trading days used when annualising the covariance matrix.
_TRADING_DAYS = 252
