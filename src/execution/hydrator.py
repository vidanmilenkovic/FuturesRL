"""Cold-start state hydration: warm every indicator before the first live bar.

A live/paper trading session that starts from an empty process has **no history**.
Every strategy feature the FinRL-X pipeline consumes (``ret_126d``, ``vol_20d``,
the EWMA covariance used by volatility targeting, ``term_slope``, ...) needs a
trailing window of bars; feeding the first live IBKR bar into a cold pipeline
produces ``NaN`` features, a missing covariance matrix and therefore an
un-sizable target book. In production that failure mode is silent: the pipeline
either raises deep inside a strategy stage or, worse, trades a degenerate book.

:class:`ColdStartHydrator` closes that gap. It rebuilds the exact point-in-time
:class:`~src.core.types.MarketState` warm-up window the pipeline would have
observed had the process been running for the whole lookback period:

1. read canonical bars for the requested roots from the durable
   :class:`~src.data.storage.ParquetRepository` over the lookback window,
2. if a root's stored coverage is short, optionally invoke an injected
   :data:`BackfillFn` (production wiring: ``DatabentoLoader.ingest`` or an IBKR
   ``reqHistoricalData`` fetch; tests inject a deterministic fake), validate the
   returned frame with :func:`src.data.schema.ensure_bars_schema`, persist it,
   and re-read the window,
3. compute the canonical features with
   :class:`~src.data.feature_processor.FeatureProcessor`,
4. build one :class:`~src.core.types.MarketState` per bar timestamp (plus a final
   state at ``as_of``) with :class:`~src.data.market_state_builder.MarketStateBuilder`,
5. validate the warm-up **on the final state**: every
   :data:`src.data.schema.REQUIRED_FEATURES` value finite for every root, an EWMA
   covariance matrix present with aligned instruments, and a non-empty trailing
   return window. Anything less raises :class:`HydrationError` — the trader must
   fail closed rather than start cold.

Coverage contract
-----------------
At least ``max(260, min_bars_per_root)`` **observed bars per root** are required.
Because futures data is sparse (weekends, CME holidays), a calendar-day window
is derived from that bar count::

    required_lookback = ceil(required_bars * 7 / 5) + CALENDAR_SAFETY_DAYS

(``7/5`` = calendar days per trading day, ``CALENDAR_SAFETY_DAYS`` absorbs
holidays and the occasional short week.) The effective window is
``max(configured lookback_days, required_lookback)`` and is reported in
:attr:`HydrationResult.lookback_days`. The formula assumes daily bars (the
default :data:`src.data.schema.DEFAULT_BAR_SCHEMA`), which is the hydration
frequency the live strategies decide on.

Multi-source backfill (Phase 8)
-------------------------------
A cold start should not depend on one vendor. :class:`MultiSourceBackfill` is a
:data:`BackfillFn` that tries an ordered chain of history providers and returns
the first canonical frame covering each root:
``auto`` = Databento (API key) -> IBKR historical (paper/live) -> offline
synthetic history (only when the operator explicitly allows it, e.g.
``--offline``); a forced ``--backfill-source databento|ibkr|offline`` fails
closed with :data:`HYDRATION_SOURCE_HINTS` when that source is unavailable.
Sources that only expose a single series per root (IBKR continuous futures, the
seeded offline source) are given a neutrally-priced second expiration
(:func:`add_neutral_term_structure`) so ``term_slope``/``carry_annualized`` are
``0.0`` instead of ``NaN``; the assumption is logged as a warning.

Determinism
-----------
Hydration is pure given the repository contents, ``as_of`` and the injected
callables: no network access happens unless a :data:`BackfillFn` is injected, the
read window is half-open ``[as_of - lookback, as_of]`` (the ``as_of`` bar is
included) and the persisted backfill is idempotent (the repository dedupes on the
canonical primary key), so calling :meth:`ColdStartHydrator.hydrate` twice
returns the same warm-up window.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from types import MappingProxyType

import polars as pl
from configs.instruments import InstrumentsConfig, load_instruments

from src.core.types import InstrumentId, MarketState, ensure_utc
from src.data import schema as s
from src.data.feature_processor import FeatureProcessor
from src.data.market_state_builder import MarketStateBuilder
from src.data.storage import ParquetRepository, StorageError

__all__ = [
    "CALENDAR_SAFETY_DAYS",
    "DEFAULT_MAX_WARMUP_STATES",
    "HYDRATION_SOURCES",
    "HYDRATION_SOURCE_HINTS",
    "MIN_REQUIRED_BARS",
    "NEUTRAL_TERM_STRUCTURE_DAYS",
    "BackfillFn",
    "BackfillSource",
    "ColdStartHydrator",
    "HydrationError",
    "HydrationResult",
    "MultiSourceBackfill",
    "add_neutral_term_structure",
    "build_backfill_chain",
    "offline_backfill",
]

logger = logging.getLogger(__name__)

#: Absolute minimum number of observed bars per root needed to warm every
#: canonical feature (``ret_126d`` needs 127 bars, ``vol_20d`` 21, the EWMA
#: covariance 20 cross-sectional observations; 260 gives a full year of margin).
MIN_REQUIRED_BARS: int = 260

#: Calendar days per trading day (5 sessions per 7 calendar days).
_CALENDAR_DAYS_PER_TRADING_DAY: float = 7.0 / 5.0

#: Extra calendar days added to the derived window to absorb CME holidays,
#: half sessions and short weeks (purely a safety margin, never a substitute for
#: the observed-bar coverage check).
CALENDAR_SAFETY_DAYS: int = 10

#: ``read_bars`` uses a half-open ``[start, end)`` window; widen ``end`` by one
#: microsecond so a bar stamped exactly at ``as_of`` is part of the warm-up.
_INCLUSIVE_END: timedelta = timedelta(microseconds=1)

#: Default cap on materialised warm-up states (see ``max_warmup_states``).
DEFAULT_MAX_WARMUP_STATES: int = 64


class HydrationError(RuntimeError):
    """Raised when the cold-start warm-up window cannot be reconstructed.

    The message always names the offending roots, the observed coverage and the
    actionable remedy (ingest history or inject a backfill callable): a live
    trader must never start on a cold pipeline.
    """


#: Function used to fetch missing history: ``(roots, start, end) -> bars``.
#:
#: Production wiring passes :meth:`src.data.databento_loader.DatabentoLoader.ingest`
#: (or an IBKR ``reqHistoricalData`` adapter); tests inject a deterministic fake.
#: ``start`` is inclusive and ``end`` is exclusive apart from a one-microsecond
#: widening so a bar stamped exactly at ``as_of`` is included.
type BackfillFn = Callable[[Sequence[InstrumentId], datetime, datetime], pl.DataFrame]


@dataclass(frozen=True, slots=True)
class HydrationResult:
    """The reconstructed warm-up window handed to the live trading runner.

    Attributes
    ----------
    states:
        Chronologically ordered point-in-time market states; the last one is the
        state observed at ``as_of``.
    bars:
        Canonical bar history actually used for the warm-up (validated schema).
    roots:
        Normalised, sorted, de-duplicated futures roots that were hydrated.
    lookback_days:
        Effective calendar-day lookback used for the read (``>=`` the configured
        lookback because the observed-bar requirement may extend it).
    coverage:
        Number of observed bars per root in the read window.
    required_bars:
        Minimum observed bars per root that was enforced.
    backfilled:
        Whether missing history was fetched and persisted during this call.
    warnings:
        Non-fatal observations (lookback expansion, ignored backfill roots, ...).
    as_of:
        Decision timestamp the warm-up ends at (timezone-aware UTC).
    """

    states: tuple[MarketState, ...]
    bars: pl.DataFrame
    roots: tuple[InstrumentId, ...]
    lookback_days: int
    coverage: Mapping[InstrumentId, int]
    required_bars: int
    backfilled: bool
    warnings: tuple[str, ...]
    as_of: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.bars, pl.DataFrame):
            raise TypeError(f"bars must be a polars DataFrame, got {type(self.bars)!r}")
        if not self.states:
            raise ValueError("HydrationResult requires at least one hydrated market state")
        moment = ensure_utc(self.as_of)
        object.__setattr__(self, "as_of", moment)
        roots = tuple(self.roots)
        if not roots:
            raise ValueError("HydrationResult requires at least one root")
        if len(set(roots)) != len(roots):
            raise ValueError(f"HydrationResult roots must be unique, got {roots!r}")
        if any(not root for root in roots):
            raise ValueError("HydrationResult roots must be non-empty strings")
        object.__setattr__(self, "roots", roots)
        if isinstance(self.required_bars, bool) or self.required_bars < 1:
            raise ValueError("required_bars must be a positive int")
        if isinstance(self.lookback_days, bool) or self.lookback_days < 1:
            raise ValueError("lookback_days must be a positive int")
        previous: datetime | None = None
        for state in self.states:
            if previous is not None and state.timestamp < previous:
                raise ValueError(
                    "HydrationResult states must be chronologically ordered; "
                    f"{state.timestamp.isoformat()} follows {previous.isoformat()}"
                )
            previous = state.timestamp
        if previous != moment:
            raise ValueError(
                "the last hydrated state must be the state observed at as_of; "
                f"last={previous!r} as_of={moment!r}"
            )
        coverage = {str(root): int(count) for root, count in self.coverage.items()}
        missing = sorted(set(roots) - set(coverage))
        if missing:
            raise ValueError(f"HydrationResult coverage is missing roots {missing!r}")
        if any(count < 0 for count in coverage.values()):
            raise ValueError("HydrationResult coverage counts must be non-negative")
        object.__setattr__(self, "coverage", MappingProxyType(coverage))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings))
        object.__setattr__(self, "backfilled", bool(self.backfilled))

    # -- convenience -----------------------------------------------------------

    @property
    def final_state(self) -> MarketState:
        """The state observed at ``as_of`` (the last warm-up state)."""
        return self.states[-1]

    @property
    def bar_count(self) -> int:
        """Total number of bars used for the warm-up."""
        return self.bars.height


class ColdStartHydrator:
    """Rebuild the point-in-time warm-up window a live pipeline must observe.

    Parameters
    ----------
    repository:
        Durable bar store (:class:`~src.data.storage.ParquetRepository`).
    instrument_master:
        Instrument master for asset classes / margin metadata. Defaults to the
        committed ``configs/instruments.yaml``.
    feature_processor:
        Feature engine. Defaults to the injected ``state_builder``'s processor,
        else a fresh :class:`~src.data.feature_processor.FeatureProcessor`.
    state_builder:
        Point-in-time state builder. Defaults to
        ``MarketStateBuilder(instrument_master=..., feature_processor=...,
        initial_equity=equity)``. An injected builder is reused as-is with the
        requested ``equity`` applied to every hydrated state.
    lookback_days:
        Configured calendar-day lookback floor (``400`` by default, i.e. a full
        year plus holidays). :attr:`required_lookback` may extend it so the
        observed-bar requirement is satisfied.
    min_bars_per_root:
        Override of the per-root observed-bar requirement (never below
        :data:`MIN_REQUIRED_BARS`).
    backfill:
        Optional missing-history provider. Only invoked when stored coverage is
        short **and** backfill is enabled (constructor flag or ``allow_backfill``
        on :meth:`hydrate`) **and** this callable is present.
    clock:
        Injectable UTC clock used by :meth:`hydrate_now`.
    backfill_enabled:
        Operator/configuration switch mirroring
        ``Settings.hydration_backfill_enabled``. When ``False`` a short window
        fails closed even if a provider is configured.
    schema:
        Repository bar schema to read (daily bars by default).
    max_warmup_states:
        Cap on how many of the newest bar timestamps are materialised into
        :attr:`HydrationResult.states` (``None`` materialises every timestamp in
        the window). Features are always computed over the full window, so every
        returned state is exactly what the pipeline would have observed; the cap
        only bounds the latency/memory of the audit trail (the state builder
        re-estimates the EWMA covariance for every state, ~10-20ms each, and the
        live runner consumes only the final state). Defaults to
        :data:`DEFAULT_MAX_WARMUP_STATES`.
    """

    def __init__(
        self,
        *,
        repository: ParquetRepository,
        instrument_master: InstrumentsConfig | None = None,
        feature_processor: FeatureProcessor | None = None,
        state_builder: MarketStateBuilder | None = None,
        lookback_days: int = 400,
        min_bars_per_root: int | None = None,
        backfill: BackfillFn | None = None,
        clock: Callable[[], datetime] | None = None,
        backfill_enabled: bool = True,
        schema: str = s.DEFAULT_BAR_SCHEMA,
        max_warmup_states: int | None = DEFAULT_MAX_WARMUP_STATES,
    ) -> None:
        if not isinstance(repository, ParquetRepository):
            raise TypeError(f"repository must be a ParquetRepository, got {type(repository)!r}")
        if (
            isinstance(lookback_days, bool)
            or not isinstance(lookback_days, int)
            or lookback_days < 1
        ):
            raise ValueError("lookback_days must be a positive int")
        if min_bars_per_root is not None and (
            isinstance(min_bars_per_root, bool)
            or not isinstance(min_bars_per_root, int)
            or min_bars_per_root < 1
        ):
            raise ValueError("min_bars_per_root must be a positive int or None")
        if max_warmup_states is not None and (
            isinstance(max_warmup_states, bool)
            or not isinstance(max_warmup_states, int)
            or max_warmup_states < 1
        ):
            raise ValueError("max_warmup_states must be a positive int or None")
        if not s.is_bar_schema(schema):
            raise ValueError(
                f"schema must be one of the canonical bar schemas "
                f"{list(s.DATABENTO_BAR_SCHEMAS)}, got {schema!r}"
            )
        self._repository = repository
        self._master = instrument_master
        self._feature_processor = feature_processor
        self._state_builder = state_builder
        self._lookback_days = int(lookback_days)
        self._min_bars_per_root = min_bars_per_root
        self._backfill = backfill
        self._clock: Callable[[], datetime] = clock if clock is not None else _utc_now
        self._backfill_enabled = bool(backfill_enabled)
        self._schema = schema
        self._max_warmup_states = max_warmup_states

    # -- introspection ---------------------------------------------------------

    @property
    def repository(self) -> ParquetRepository:
        """Durable bar store this hydrator reads from / persists backfills into."""
        return self._repository

    @property
    def instrument_master(self) -> InstrumentsConfig:
        """Instrument master used for asset classes and the default roots."""
        if self._master is None:
            self._master = load_instruments()
        return self._master

    @property
    def feature_processor(self) -> FeatureProcessor:
        """Feature engine used to derive the canonical warm-up features."""
        if self._feature_processor is None:
            builder = self._state_builder
            self._feature_processor = (
                builder.feature_processor
                if builder is not None
                else FeatureProcessor(instrument_master=self._master)
            )
        return self._feature_processor

    @property
    def lookback_days(self) -> int:
        """Configured calendar-day lookback floor."""
        return self._lookback_days

    @property
    def required_bars(self) -> int:
        """Observed bars per root that :meth:`hydrate` enforces."""
        floor = self._min_bars_per_root
        return max(MIN_REQUIRED_BARS, floor if floor is not None else MIN_REQUIRED_BARS)

    @property
    def required_lookback(self) -> int:
        """Calendar-day window needed to observe :attr:`required_bars` daily bars.

        ``ceil(required_bars * 7 / 5) + CALENDAR_SAFETY_DAYS``: five trading
        sessions per seven calendar days plus a holiday/short-week safety margin.
        The effective read window is ``max(lookback_days, required_lookback)``.
        """
        derived = math.ceil(self.required_bars * _CALENDAR_DAYS_PER_TRADING_DAY)
        return derived + CALENDAR_SAFETY_DAYS

    @property
    def backfill_enabled(self) -> bool:
        """Whether the hydrator may fetch missing history at all."""
        return self._backfill_enabled

    @property
    def max_warmup_states(self) -> int | None:
        """Cap on materialised warm-up states (``None`` = every bar timestamp)."""
        return self._max_warmup_states

    def effective_lookback(self) -> int:
        """Calendar-day window :meth:`hydrate` actually reads."""
        return max(self._lookback_days, self.required_lookback)

    # -- warm-up validation ----------------------------------------------------

    def is_warm(self, state: MarketState) -> bool:
        """Whether ``state`` is a fully warmed pipeline input.

        Checks every instrument of ``state`` for finite
        :data:`~src.data.schema.REQUIRED_FEATURES`, a non-empty returns window,
        and a present EWMA covariance aligned with the state's instruments.
        """
        if not isinstance(state, MarketState):
            raise TypeError(f"state must be a MarketState, got {type(state)!r}")
        return not self._warm_problems(state, tuple(state.instruments))

    def _warm_problems(self, state: MarketState, roots: Sequence[InstrumentId]) -> list[str]:
        """Return the reasons ``state`` is not warm (empty list when warm)."""
        problems: list[str] = []
        for root in roots:
            snapshot = state.instruments.get(root)
            if snapshot is None:
                problems.append(f"{root}: missing from the final state")
                continue
            missing = [
                name for name in s.REQUIRED_FEATURES if not _is_finite(snapshot.features.get(name))
            ]
            if missing:
                problems.append(
                    f"{root}: required features missing or non-finite after the warm-up {missing}"
                )
            series = state.returns.get(root)
            if series is None or series.size == 0:
                problems.append(f"{root}: empty trailing returns window")
        covariance = state.covariance
        if covariance is None:
            problems.append(
                "EWMA covariance matrix is unavailable (too few cross-sectional returns "
                "in the warm-up window)"
            )
        else:
            expected = (len(state.covariance_instruments), len(state.covariance_instruments))
            if covariance.shape != expected:
                problems.append(
                    f"covariance shape {covariance.shape} does not match covariance "
                    f"instruments {state.covariance_instruments!r}"
                )
            aligned = set(state.covariance_instruments)
            missing_covariance = sorted(set(roots) - aligned)
            extra_covariance = sorted(aligned - set(roots))
            if missing_covariance or extra_covariance:
                problems.append(
                    "covariance instruments are not aligned with the hydrated roots "
                    f"(missing={missing_covariance!r}, extra={extra_covariance!r})"
                )
            if not bool(pl.Series(covariance.flatten().tolist()).is_finite().all()):
                problems.append("covariance matrix contains non-finite values")
        return problems

    # -- hydration -------------------------------------------------------------

    def hydrate(
        self,
        roots: Sequence[InstrumentId],
        as_of: datetime,
        *,
        equity: float | None = None,
        allow_backfill: bool | None = None,
    ) -> HydrationResult:
        """Rebuild and validate the warm-up window ending at ``as_of``.

        Parameters
        ----------
        roots:
            Futures roots to hydrate (normalised, sorted, de-duplicated).
        as_of:
            Decision timestamp; the last hydrated state is observed exactly here.
        equity:
            Account equity attached to every hydrated state (``None`` leaves it
            unset so the state builder's ``initial_equity`` applies).
        allow_backfill:
            Per-call override of the constructor's ``backfill_enabled`` switch.

        Raises
        ------
        HydrationError
            When coverage is short and cannot be restored, when the backfill
            provider fails or returns unusable data, or when the final state is
            not warm.
        """
        moment = ensure_utc(as_of)
        try:
            normalised_roots = s.validate_roots(tuple(roots))
        except ValueError as exc:
            raise HydrationError(f"cold-start hydration rejected the root list: {exc}") from exc
        lookback = self.effective_lookback()
        window_start = moment - timedelta(days=lookback)
        window_end = moment + _INCLUSIVE_END
        warnings: list[str] = []
        if lookback > self._lookback_days:
            warnings.append(
                f"lookback expanded from {self._lookback_days} to {lookback} calendar days so "
                f"{self.required_bars} observed daily bars per root can be hydrated"
            )
        bars = self._read_window(normalised_roots, window_start, window_end)
        coverage = self._coverage(bars, normalised_roots)
        short = tuple(root for root in normalised_roots if coverage[root] < self.required_bars)
        backfilled = False
        backfill_allowed = (
            self._backfill_enabled if allow_backfill is None else bool(allow_backfill)
        )
        if short:
            if not backfill_allowed:
                raise HydrationError(
                    self._insufficient_message(
                        moment,
                        lookback,
                        normalised_roots,
                        coverage,
                        short,
                        reason=(
                            "backfill is disabled (settings.hydration_backfill_enabled / "
                            "allow_backfill=False)"
                        ),
                    )
                )
            if self._backfill is None:
                raise HydrationError(
                    self._insufficient_message(
                        moment,
                        lookback,
                        normalised_roots,
                        coverage,
                        short,
                        reason=(
                            "no backfill provider is configured; inject "
                            "ColdStartHydrator(backfill=...) wired to "
                            "DatabentoLoader.ingest or an IBKR historical-data fetch"
                        ),
                    )
                )
            bars = self._backfill_window(
                short,
                normalised_roots,
                coverage,
                window_start,
                window_end,
                warnings,
            )
            coverage = self._coverage(bars, normalised_roots)
            backfilled = True
            still_short = tuple(
                root for root in normalised_roots if coverage[root] < self.required_bars
            )
            if still_short:
                raise HydrationError(
                    self._insufficient_message(
                        moment,
                        lookback,
                        normalised_roots,
                        coverage,
                        still_short,
                        reason="the backfill provider returned but coverage is still short",
                    )
                )
        features = self._compute_features(bars, normalised_roots, moment)
        states = self._build_states(features, normalised_roots, moment, equity)
        final_state = states[-1]
        problems = self._warm_problems(final_state, normalised_roots)
        if problems:
            raise HydrationError(
                "cold-start hydration refused: the warm-up window is not warm at "
                f"{moment.isoformat()} for roots {list(normalised_roots)!r} "
                f"({self.required_bars} bars/root required; observed coverage {dict(coverage)!r}): "
                + "; ".join(problems)
            )
        logger.info(
            "hydrated %d roots over %d calendar days (%d states, %d bars, backfilled=%s)",
            len(normalised_roots),
            lookback,
            len(states),
            bars.height,
            backfilled,
        )
        return HydrationResult(
            states=tuple(states),
            bars=bars,
            roots=normalised_roots,
            lookback_days=lookback,
            coverage=coverage,
            required_bars=self.required_bars,
            backfilled=backfilled,
            warnings=tuple(warnings),
            as_of=moment,
        )

    def hydrate_now(
        self,
        roots: Sequence[InstrumentId],
        *,
        equity: float | None = None,
        allow_backfill: bool | None = None,
    ) -> HydrationResult:
        """Hydrate the warm-up window ending at the injected clock's current time."""
        return self.hydrate(
            roots,
            self._clock(),
            equity=equity,
            allow_backfill=allow_backfill,
        )

    # -- internals -------------------------------------------------------------

    def _read_window(
        self,
        roots: Sequence[InstrumentId],
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """Read the canonical bars for the warm-up window from the repository."""
        try:
            bars = self._repository.read_bars(roots, start, end, schema=self._schema)
            return s.ensure_bars_schema(bars, context="hydration-read")
        except (StorageError, ValueError, pl.exceptions.PolarsError) as exc:
            raise HydrationError(
                f"cold-start hydration could not read the {self._schema!r} history for "
                f"roots {list(roots)!r} in [{start.isoformat()}, {end.isoformat()}): {exc}"
            ) from exc

    @staticmethod
    def _coverage(bars: pl.DataFrame, roots: Sequence[InstrumentId]) -> dict[str, int]:
        """Observed bar count per root (``0`` for a root without stored bars).

        Counts *distinct bar timestamps* per root, not raw rows: canonical history
        may carry several expirations per timestamp (front + second month from
        Databento parent symbology or the synthetic offline curve), and the warm-up
        requirement is expressed in decision bars, not contract legs. For
        single-expiration sources the result is identical to a row count.
        """
        counts: dict[str, int] = dict.fromkeys(roots, 0)
        if bars.is_empty():
            return counts
        grouped = bars.group_by(s.ROOT_SYMBOL_COLUMN).agg(
            pl.col(s.TIMESTAMP_COLUMN).n_unique().alias("bars")
        )
        for record in grouped.iter_rows(named=True):
            root = str(record[s.ROOT_SYMBOL_COLUMN])
            if root in counts:
                counts[root] = int(record["bars"])
        return counts

    def _backfill_window(
        self,
        short_roots: Sequence[InstrumentId],
        requested_roots: Sequence[InstrumentId],
        coverage: Mapping[InstrumentId, int],
        start: datetime,
        end: datetime,
        warnings: list[str],
    ) -> pl.DataFrame:
        """Fetch, validate and persist missing history, then re-read the window."""
        provider = self._backfill
        if provider is None:  # pragma: no cover - guarded by the caller
            raise HydrationError("no backfill provider is configured")
        logger.warning(
            "cold-start hydration backfilling roots %s (%s) over %d calendar days",
            list(short_roots),
            dict(coverage),
            max((end - start).days, 1),
        )
        try:
            frame = provider(short_roots, start, end)
        except HydrationError:
            raise
        except Exception as exc:
            raise HydrationError(
                "cold-start hydration backfill failed for roots "
                f"{list(short_roots)!r} in [{start.isoformat()}, {end.isoformat()}): "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(frame, pl.DataFrame):
            raise HydrationError(
                f"cold-start hydration backfill must return a polars DataFrame, got {type(frame)!r}"
            )
        if frame.is_empty():
            raise HydrationError(
                "cold-start hydration backfill returned an empty frame for roots "
                f"{list(short_roots)!r}; the vendor/IBKR request produced no history"
            )
        try:
            validated = s.ensure_bars_schema(frame, context="hydration-backfill")
        except s.SchemaError as exc:
            raise HydrationError(
                f"cold-start hydration backfill returned a non-canonical frame: {exc}"
            ) from exc
        available = {str(root) for root in validated[s.ROOT_SYMBOL_COLUMN].unique().to_list()}
        missing = sorted(set(short_roots) - available)
        if missing:
            raise HydrationError(
                "cold-start hydration backfill did not return any rows for roots "
                f"{missing!r}; refusing to start cold"
            )
        extra = sorted(available - set(requested_roots))
        if extra:
            warnings.append(
                f"ignored backfill rows for roots outside the requested universe {extra!r}"
            )
            validated = validated.filter(pl.col(s.ROOT_SYMBOL_COLUMN).is_in(list(requested_roots)))
        try:
            written = self._repository.write_bars(validated, schema=self._schema, mode="append")
        except (StorageError, ValueError, TypeError) as exc:
            raise HydrationError(
                f"cold-start hydration could not persist the backfilled history: {exc}"
            ) from exc
        if written <= 0:  # pragma: no cover - the repository reports input rows
            raise HydrationError(
                "cold-start hydration backfill persisted zero rows; refusing to start cold"
            )
        warnings.append(
            f"backfilled {written} rows for roots {list(short_roots)!r} and persisted them to "
            f"the repository"
        )
        return self._read_window(requested_roots, start, end)

    def _compute_features(
        self,
        bars: pl.DataFrame,
        roots: Sequence[InstrumentId],
        moment: datetime,
    ) -> pl.DataFrame:
        """Compute the canonical feature frame for the warm-up window."""
        try:
            features = self.feature_processor.compute(bars)
        except (ValueError, s.SchemaError, TypeError, pl.exceptions.PolarsError) as exc:
            raise HydrationError(
                f"cold-start hydration feature computation failed for roots {list(roots)!r} "
                f"at {moment.isoformat()}: {type(exc).__name__}: {exc}"
            ) from exc
        if features.is_empty():
            raise HydrationError(
                f"cold-start hydration produced an empty feature frame for roots "
                f"{list(roots)!r} at {moment.isoformat()}"
            )
        return features

    def _build_states(
        self,
        features: pl.DataFrame,
        roots: Sequence[InstrumentId],
        moment: datetime,
        equity: float | None,
    ) -> list[MarketState]:
        """Build point-in-time states for the newest warm-up timestamps.

        One state is built per bar timestamp (plus the ``as_of`` state) up to the
        configured ``max_warmup_states`` cap; the final state is always the one
        observed at ``moment``.
        """
        timestamps = sorted(
            {
                value
                for value in features[s.TIMESTAMP_COLUMN].to_list()
                if isinstance(value, datetime)
            }
        )
        if not timestamps:
            raise HydrationError(
                f"cold-start hydration found no timestamps for roots {list(roots)!r} "
                f"at {moment.isoformat()}"
            )
        decision_times = [ensure_utc(stamp) for stamp in timestamps if stamp <= moment]
        if not decision_times:
            raise HydrationError(
                f"cold-start hydration has no bars at or before {moment.isoformat()} for roots "
                f"{list(roots)!r}"
            )
        if decision_times[-1] != moment:
            decision_times.append(moment)
        cap = self._max_warmup_states
        if cap is not None:
            decision_times = decision_times[-cap:]
        builder = self._state_builder
        try:
            if builder is None:
                builder = MarketStateBuilder(
                    instrument_master=self._master,
                    feature_processor=self.feature_processor,
                    initial_equity=equity,
                )
                return builder.build_series(features, decision_times)
            states = builder.build_series(features, decision_times)
            if equity is not None:
                states = [replace(state, equity=equity) for state in states]
            return states
        except (ValueError, TypeError, pl.exceptions.PolarsError) as exc:
            raise HydrationError(
                f"cold-start hydration could not build point-in-time states for roots "
                f"{list(roots)!r} at {moment.isoformat()}: {type(exc).__name__}: {exc}"
            ) from exc

    def _insufficient_message(
        self,
        moment: datetime,
        lookback: int,
        roots: Sequence[InstrumentId],
        coverage: Mapping[InstrumentId, int],
        short_roots: Sequence[InstrumentId],
        *,
        reason: str,
    ) -> str:
        """Actionable ``HydrationError`` message for a cold window."""
        detail = ", ".join(f"{root}={coverage.get(root, 0)}" for root in roots)
        return (
            "cold-start hydration refused: insufficient stored history for roots "
            f"{list(short_roots)!r} (need at least "
            f"{self.required_bars} observed bars per root over a {lookback}-day lookback ending "
            f"{moment.isoformat()}; observed coverage: {detail}). {reason}. "
            "Ingest the missing history with DatabentoLoader.ingest(roots, start, end) / IBKR "
            "historical data, or inject a ColdStartHydrator(backfill=...) callable."
        )


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _is_finite(value: float | None) -> bool:
    """Whether ``value`` is a finite real number."""
    return value is not None and math.isfinite(float(value))


# ---------------------------------------------------------------------------
# Multi-source backfill chain (Phase 8)
# ---------------------------------------------------------------------------
#
# A production cold start should not depend on a single vendor. The chain below
# tries sources in a configured order and returns the first canonical frame that
# covers each requested root:
#
#     auto:     Databento (API key configured) -> IBKR historical (paper/live
#               environment reachable) -> OfflineBarSource (only when the
#               operator explicitly allows offline history, e.g. ``--offline``)
#     databento/ibkr/offline: force that single source (fail closed when it is
#               not configured instead of silently trading on another vendor).
#
# The chain itself is dependency-free: the concrete vendors are injected as
# :data:`BackfillFn` callables, so production wiring (and tests) stay offline.

#: Supported hydration backfill source selectors (mirrors ``Settings.hydration_source``).
HYDRATION_SOURCES: tuple[str, ...] = ("auto", "databento", "ibkr", "offline")

#: Calendar days between a real expiration and the neutrally-priced synthetic
#: next expiration added when a source only exposes a single series per root.
NEUTRAL_TERM_STRUCTURE_DAYS: int = 90

#: Actionable guidance appended to every source-availability failure. The CLI
#: also prints this block so an operator never sees a bare traceback.
HYDRATION_SOURCE_HINTS: str = (
    "Options: (1) rerun with --offline for deterministic synthetic history; "
    "(2) start IBKR TWS or IB Gateway and export FUTURESRLX_ENVIRONMENT=paper|live plus "
    "FUTURESRLX_IBKR_PORT (paper 7497/4002, live 7496/4001); "
    "(3) set FUTURESRLX_DATABENTO_API_KEY for Databento GLBX.MDP3 history."
)


@dataclass(frozen=True, slots=True)
class BackfillSource:
    """One named history provider in a :class:`MultiSourceBackfill` chain."""

    name: str
    fetch: BackfillFn

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("BackfillSource.name must be a non-empty string")
        if not callable(self.fetch):
            raise TypeError("BackfillSource.fetch must be a callable")


def _normalise_backfill_frame(frame: object, *, source: str) -> pl.DataFrame:
    """Validate one source response and neutralise a missing term structure.

    Every source must return canonical bars; a source that exposes only one
    series per root (continuous/front-month history such as IBKR's continuous
    futures or the synthetic offline source) cannot provide ``term_slope`` /
    ``carry_annualized``. Rather than allowing ``NaN`` to poison the warm-up,
    a neutrally-priced next expiration is added (flat curve, ``term_slope = 0``)
    and the assumption is logged.
    """
    if not isinstance(frame, pl.DataFrame):
        raise HydrationError(
            f"hydration source {source!r} must return a polars DataFrame, got {type(frame)!r}"
        )
    validated = s.ensure_bars_schema(frame, context=f"hydration-source[{source}]")
    if validated.is_empty():
        return validated
    return add_neutral_term_structure(validated, context=f"hydration-source[{source}]")


def add_neutral_term_structure(frame: pl.DataFrame, *, context: str = "hydration") -> pl.DataFrame:
    """Make the required term-structure features computable for single-series data.

    ``FeatureProcessor`` derives ``term_slope = (F2 - F1) / F1`` and
    ``carry_annualized`` from the front and second contract observed at the same
    timestamp. A source that only ever returns one expiration per root therefore
    produces ``NaN`` for two :data:`src.data.schema.REQUIRED_FEATURES` and the
    hydrator fails closed. For every ``(root, timestamp)`` that has exactly one
    expiration this function appends a neutrally-priced next expiration
    (:data:`NEUTRAL_TERM_STRUCTURE_DAYS` further out, identical OHLC), so the
    implied curve is flat and the features are ``0.0`` instead of ``NaN``.

    This is an explicit *neutral assumption*, never a directional carry signal:
    sources that provide a real second expiration (e.g. Databento ``parent``
    symbology) are returned unchanged.
    """
    validated = s.ensure_bars_schema(frame, context=context)
    if validated.is_empty():
        return validated
    missing_expiration = validated.filter(pl.col(s.EXPIRATION_COLUMN).is_null()).height
    if missing_expiration:
        raise HydrationError(
            f"{context}: {missing_expiration} rows have no expiration_date; a canonical "
            "history source must stamp the contract expiry so term-structure features can "
            "be derived"
        )
    counts = validated.group_by([s.ROOT_SYMBOL_COLUMN, s.TIMESTAMP_COLUMN]).agg(
        pl.col(s.EXPIRATION_COLUMN).n_unique().alias("_expirations")
    )
    single = counts.filter(pl.col("_expirations") == 1).select(
        [s.ROOT_SYMBOL_COLUMN, s.TIMESTAMP_COLUMN]
    )
    if single.is_empty():
        return validated
    next_leg = (
        validated.join(single, on=[s.ROOT_SYMBOL_COLUMN, s.TIMESTAMP_COLUMN], how="inner")
        .with_columns(
            pl.col(s.EXPIRATION_COLUMN)
            .dt.offset_by(f"{NEUTRAL_TERM_STRUCTURE_DAYS}d")
            .alias(s.EXPIRATION_COLUMN),
            (pl.col(s.INSTRUMENT_ID_COLUMN) + "_NEXT").alias(s.INSTRUMENT_ID_COLUMN),
            (pl.col(s.VOLUME_COLUMN) * 0.5).alias(s.VOLUME_COLUMN),
            (pl.col(s.OPEN_INTEREST_COLUMN) * 0.5).alias(s.OPEN_INTEREST_COLUMN),
        )
        .select(validated.columns)
    )
    augmented = s.ensure_bars_schema(
        pl.concat([validated, next_leg], how="vertical_relaxed"),
        context=f"{context}/term-structure",
    ).sort(list(s.BAR_KEY_COLUMNS))
    logger.warning(
        "%s: %d root/timestamp row(s) had a single expiration; added neutrally-priced "
        "next-expiration rows so term_slope/carry_annualized are 0.0 instead of NaN",
        context,
        single.height,
    )
    return augmented


def offline_backfill(roots: Sequence[InstrumentId], start: datetime, end: datetime) -> pl.DataFrame:
    """Deterministic synthetic history for ``--offline`` mode.

    Uses the seeded :class:`~src.data.databento_loader.OfflineBarSource` (clearly
    synthetic: contract ids carry the ``_SYN`` suffix) and adds the neutral
    second expiration required by the warm-up validation. No network access.
    """
    from src.data.databento_loader import OfflineBarSource

    if end <= start:
        raise HydrationError(
            f"offline backfill needs end ({end.isoformat()}) after start ({start.isoformat()})"
        )
    source = OfflineBarSource(roots, periods=2, start=start)
    frame = source.load(roots, start, end)
    return add_neutral_term_structure(frame, context="offline-backfill")


class MultiSourceBackfill:
    """Ordered fallback chain of hydration history providers.

    Implements the :data:`BackfillFn` signature so it can be injected directly
    into :class:`ColdStartHydrator`. Sources are tried in order; a source that
    errors, returns an empty/unusable frame or misses roots is logged as a
    fallback warning and the next source is attempted. Only when every root is
    still uncovered does the chain raise :class:`HydrationError` — with the
    per-source failure reasons and :data:`HYDRATION_SOURCE_HINTS`.
    """

    def __init__(self, sources: Sequence[BackfillSource], *, label: str = "hydration") -> None:
        if not isinstance(label, str) or not label:
            raise ValueError("label must be a non-empty string")
        self._sources = tuple(sources)
        self._label = label

    @property
    def sources(self) -> tuple[str, ...]:
        """Source names in fallback order."""
        return tuple(source.name for source in self._sources)

    def __bool__(self) -> bool:
        return bool(self._sources)

    def __len__(self) -> int:
        return len(self._sources)

    def __repr__(self) -> str:
        return f"MultiSourceBackfill({self.sources!r})"

    def __call__(
        self,
        roots: Sequence[InstrumentId],
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """Fetch every requested root from the first source(s) that can supply it."""
        requested = tuple(roots)
        if not requested:
            raise HydrationError("hydration backfill requires at least one root")
        if not self._sources:
            raise HydrationError(
                "no hydration backfill source is available: Databento is not configured, IBKR "
                "is not configured (paper/live only) and offline history was not allowed. "
                + HYDRATION_SOURCE_HINTS
            )
        missing = requested
        frames: list[pl.DataFrame] = []
        failures: list[str] = []
        for source in self._sources:
            if not missing:
                break
            try:
                raw = source.fetch(missing, start, end)
                frame = _normalise_backfill_frame(raw, source=source.name)
            except HydrationError as exc:
                failures.append(f"{source.name}: {exc}")
                logger.warning(
                    "%s source %r could not backfill %s (%s); falling back",
                    self._label,
                    source.name,
                    list(missing),
                    exc,
                )
                continue
            except Exception as exc:
                failures.append(f"{source.name}: {type(exc).__name__}: {exc}")
                logger.warning(
                    "%s source %r failed for %s (%s: %s); falling back",
                    self._label,
                    source.name,
                    list(missing),
                    type(exc).__name__,
                    exc,
                )
                continue
            if frame.is_empty():
                failures.append(f"{source.name}: returned no rows")
                logger.warning(
                    "%s source %r returned no rows for %s; falling back",
                    self._label,
                    source.name,
                    list(missing),
                )
                continue
            present = {str(root) for root in frame[s.ROOT_SYMBOL_COLUMN].unique().to_list()}
            supplied = tuple(root for root in missing if root in present)
            if not supplied:
                failures.append(f"{source.name}: returned no rows for the requested roots")
                logger.warning(
                    "%s source %r returned no rows for any of %s; falling back",
                    self._label,
                    source.name,
                    list(missing),
                )
                continue
            frames.append(frame.filter(pl.col(s.ROOT_SYMBOL_COLUMN).is_in(list(supplied))))
            missing = tuple(root for root in missing if root not in present)
            logger.info(
                "%s source %r supplied %d row(s) for roots %s",
                self._label,
                source.name,
                frame.height,
                list(supplied),
            )
            if missing:
                logger.warning(
                    "%s source %r returned no history for roots %s; falling back",
                    self._label,
                    source.name,
                    list(missing),
                )
        if missing:
            raise HydrationError(
                "no hydration source could supply history for roots "
                f"{list(missing)!r} between {start.isoformat()} and {end.isoformat()}: "
                + ("; ".join(failures) if failures else "no source was configured")
                + ". "
                + HYDRATION_SOURCE_HINTS
            )
        if len(frames) == 1:
            return frames[0]
        return s.ensure_bars_schema(
            pl.concat(frames, how="vertical_relaxed"), context=f"{self._label}-chain"
        ).sort(list(s.BAR_KEY_COLUMNS))


def build_backfill_chain(
    *,
    databento: BackfillFn | None = None,
    ibkr: BackfillFn | None = None,
    offline: BackfillFn | None = None,
    source: str = "auto",
    offline_allowed: bool = False,
) -> MultiSourceBackfill:
    """Compose the ordered fallback chain for a hydration ``source`` selector.

    Parameters
    ----------
    databento, ibkr, offline:
        Vendor backfill callables. ``None`` means the vendor is not configured
        (or was intentionally not wired); it is skipped in ``auto`` mode and
        makes that source unavailable in forced mode.
    source:
        ``auto`` (Databento -> IBKR -> offline-if-allowed) or a forced vendor
        (``databento`` / ``ibkr`` / ``offline``).
    offline_allowed:
        Whether synthetic offline history may be used as a fallback. ``True``
        for ``--offline`` / ``FUTURESRLX_HYDRATION_SOURCE=offline`` or when the
        operator explicitly opts in; never silently enabled.

    Raises
    ------
    HydrationError
        When an explicitly forced source is not configured (fail closed with
        :data:`HYDRATION_SOURCE_HINTS`) or the selector is unknown.
    """
    resolved = str(source or "auto").strip().lower()
    if resolved not in HYDRATION_SOURCES:
        raise HydrationError(
            f"unknown hydration source {source!r}; expected one of {list(HYDRATION_SOURCES)}. "
            + HYDRATION_SOURCE_HINTS
        )
    configured: dict[str, BackfillFn | None] = {
        "databento": databento,
        "ibkr": ibkr,
        "offline": offline,
    }
    if resolved != "auto":
        if configured[resolved] is None:
            reasons = {
                "databento": (
                    "FUTURESRLX_DATABENTO_API_KEY is not configured (the Databento SDK/API "
                    "key is required to fetch GLBX.MDP3 history)"
                ),
                "ibkr": (
                    "IBKR historical data requires a paper/live environment with TWS or IB "
                    "Gateway running: set FUTURESRLX_ENVIRONMENT=paper|live and "
                    "FUTURESRLX_IBKR_PORT (paper 7497/4002, live 7496/4001)"
                ),
                "offline": (
                    "no offline history provider is wired; offline history is only used with "
                    "--offline or --backfill-source offline"
                ),
            }
            raise HydrationError(
                f"forced hydration source {resolved!r} is unavailable: {reasons[resolved]}. "
                + HYDRATION_SOURCE_HINTS
            )
        order: list[str] = [resolved]
    else:
        order = ["databento", "ibkr"]
        if offline_allowed:
            order.append("offline")
    sources = [
        BackfillSource(name=name, fetch=fetch)
        for name in order
        if (fetch := configured[name]) is not None
    ]
    chain = MultiSourceBackfill(sources)
    if not sources:
        logger.warning(
            "no hydration backfill source is configured (auto mode); a cold repository "
            "cannot be warmed. %s",
            HYDRATION_SOURCE_HINTS,
        )
    return chain
