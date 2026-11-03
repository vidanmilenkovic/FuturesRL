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
    :func:`~src.data.ibkr_contracts.front_contract_month` — pure, non-network
    front-month ``YYYYMM`` resolution from the instrument roll rules (used for
    runner/API rebalance previews); :func:`~src.data.ibkr_contracts.qualify_front_future`
    resolves and caches the qualified (conId) IBKR contract used by the streamer
    and OMS.

Live news / macro headlines
    :class:`~src.data.news_ingestor.NewsIngestor` — pluggable RSS -> IBKR ->
    deterministic-fixture headline resolution returning immutable
    :class:`~src.data.news_ingestor.NewsHeadline` objects (root/severity/sentiment
    classified) for the LLM Risk-First engine and the Risk Monitor live feed.

Synthetic offline term structure
    :class:`~src.data.term_structure.SyntheticTermStructure` — deterministic,
    roll-aware F1/F2 curve (root/regime-specific contango/backwardation) used by
    :class:`~src.data.databento_loader.OfflineBarSource` so offline/DRL training
    derives finite, non-zero ``term_slope``/``carry_annualized`` instead of
    zero-filled fallbacks. :func:`~src.data.term_structure.offline_term_structure_metadata`
    reports the per-root curve metadata for trainer/strategy checkpoints.
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
from src.data.ibkr_contracts import (
    FrontFuture,
    build_front_contract,
    clear_front_future_cache,
    front_contract_month,
    qualify_front_future,
)
from src.data.ibkr_data import IBKRMarketDataStreamer
from src.data.market_state_builder import MarketStateBuilder
from src.data.news_ingestor import NewsHeadline, NewsIngestor, Severity, SourceKind
from src.data.storage import ParquetRepository, StorageError, WriteMode
from src.data.term_structure import (
    CarryProfile,
    SyntheticTermStructure,
    TermRegime,
    TermStructureCurve,
    offline_term_structure_metadata,
)

__all__ = [
    "CalendarRollSchedule",
    "CarryProfile",
    "ContinuousContractBuilder",
    "DatabentoIngestError",
    "DatabentoLoader",
    "DatabentoNativeRollSchedule",
    "FeatureProcessor",
    "FrontFuture",
    "IBKRMarketDataStreamer",
    "MarketStateBuilder",
    "NativeRollCrossCheck",
    "NewsHeadline",
    "NewsIngestor",
    "OfflineBarSource",
    "ParquetRepository",
    "RollScheduleCalculator",
    "Severity",
    "SourceKind",
    "StorageError",
    "SyntheticTermStructure",
    "TermRegime",
    "TermStructureCurve",
    "VolumeOICrossoverRollSchedule",
    "WriteMode",
    "build_front_contract",
    "clear_front_future_cache",
    "front_contract_month",
    "offline_term_structure_metadata",
    "qualify_front_future",
    "schema",
    "synthetic_reference_price",
]
