"""FuturesRL-X data layer: canonical schema, Parquet cache, ingestion and features.

Public API
----------
Canonical contracts
    :mod:`src.data.schema` — the frozen column/feature names every adapter and
    consumer shares (see also :data:`src.data.schema.ALL_FEATURES`).

Persistence
    :class:`~src.data.storage.ParquetRepository` — partitioned Parquet cache at
    ``{parquet_dir}/{dataset}/{schema}/{root}/`` with atomic writes, dedupe,
    append/overwrite modes and a DuckDB SQL facade.

Historical ingestion
    :class:`~src.data.databento_loader.DatabentoLoader` — Databento historical
    ingestion (``ohlcv-1m/1h/1d``, ``tbbo``, ``mbp-1/10``, ``definition``).
    :class:`~src.data.databento_loader.OfflineBarSource` — deterministic
    synthetic source implementing ``HistoricalDataSource`` for offline tests.

Research features
    :class:`~src.data.continuous_roller.ContinuousContractBuilder` — roll engine
    (volume/OI crossover, calendar-before-LTD, Databento native cross-check)
    producing Panama, proportional and unadjusted series.
    :class:`~src.data.feature_processor.FeatureProcessor` — FinRL technicals,
    term structure, carry, volume/OI, OFI, macro regime and Mahalanobis
    turbulence.
    :class:`~src.data.market_state_builder.MarketStateBuilder` — strictly
    point-in-time :class:`~src.core.types.MarketState` construction.

Live/paper
    :class:`~src.data.ibkr_data.IBKRMarketDataStreamer` — ``ib_async`` 5-second
    bars, ticks and L1/L2 depth folded into ``MarketState`` snapshots.
"""

from __future__ import annotations

from src.data import schema
from src.data.continuous_roller import (
    CalendarRollSchedule,
    ContinuousContractBuilder,
    DatabentoNativeRollSchedule,
    NativeRollCrossCheck,
    RollScheduleCalculator,
    VolumeOICrossoverRollSchedule,
)
from src.data.databento_loader import (
    DatabentoIngestError,
    DatabentoLoader,
    OfflineBarSource,
    synthetic_reference_price,
)
from src.data.feature_processor import FeatureProcessor
from src.data.ibkr_data import IBKRMarketDataStreamer
from src.data.market_state_builder import MarketStateBuilder
from src.data.storage import ParquetRepository, StorageError, WriteMode

__all__ = [
    "CalendarRollSchedule",
    "ContinuousContractBuilder",
    "DatabentoIngestError",
    "DatabentoLoader",
    "DatabentoNativeRollSchedule",
    "FeatureProcessor",
    "IBKRMarketDataStreamer",
    "MarketStateBuilder",
    "NativeRollCrossCheck",
    "OfflineBarSource",
    "ParquetRepository",
    "RollScheduleCalculator",
    "StorageError",
    "VolumeOICrossoverRollSchedule",
    "WriteMode",
    "schema",
    "synthetic_reference_price",
]
