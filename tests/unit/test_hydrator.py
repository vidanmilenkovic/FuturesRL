"""Unit tests for cold-start state hydration (offline, deterministic).

The hydrator is exercised against a real partitioned ``ParquetRepository`` seeded
with the deterministic :class:`~src.data.databento_loader.OfflineBarSource`, so
the whole read -> feature -> point-in-time-state path runs exactly as it does in
production, with no network and no credentials.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from configs.settings import Settings
from src.data import schema as s
from src.data.databento_loader import OfflineBarSource
from src.data.storage import ParquetRepository
from src.execution.hydrator import (
    CALENDAR_SAFETY_DAYS,
    DEFAULT_MAX_WARMUP_STATES,
    MIN_REQUIRED_BARS,
    ColdStartHydrator,
    HydrationError,
    HydrationResult,
)

from tests.fixtures.market import make_market_state

ROOTS: tuple[str, ...] = ("ES", "NQ", "CL")
START: datetime = datetime(2024, 1, 2, 21, 0, tzinfo=UTC)
PERIODS: int = 512


def make_history(
    roots: tuple[str, ...] = ROOTS,
    *,
    periods: int = PERIODS,
) -> tuple[pl.DataFrame, datetime]:
    """Deterministic synthetic canonical bar history ending at ``as_of``."""
    source = OfflineBarSource(roots, periods=periods, start=START)
    end = START + timedelta(days=periods)
    # OfflineBarSource now emits a roll-aware front (F1) + synthetic second
    # month (F2) pair per root/timestamp, so term structure/carry are real.
    bars = source.load(roots, start=START, end=end)
    as_of = bars[s.TIMESTAMP_COLUMN].max()
    assert isinstance(as_of, datetime)
    return bars, as_of


@pytest.fixture(scope="module")
def history() -> tuple[pl.DataFrame, datetime]:
    return make_history()


@pytest.fixture()
def repository(settings: Settings) -> ParquetRepository:
    return ParquetRepository(settings)


def make_hydrator(repository: ParquetRepository, **kwargs: object) -> ColdStartHydrator:
    options: dict[str, object] = {"repository": repository, "lookback_days": 400}
    options.update(kwargs)
    return ColdStartHydrator(**options)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Happy path: warm, chronological, deterministic warm-up window
# ---------------------------------------------------------------------------


def test_hydrate_returns_warm_chronological_states(
    repository: ParquetRepository, history: tuple[pl.DataFrame, datetime]
) -> None:
    bars, as_of = history
    repository.write_bars(bars)
    hydrator = make_hydrator(repository)
    result = hydrator.hydrate(ROOTS, as_of, equity=2_500_000.0)
    assert isinstance(result, HydrationResult)
    assert result.roots == ("CL", "ES", "NQ")  # normalised + sorted
    assert result.as_of == as_of
    assert result.required_bars == MIN_REQUIRED_BARS
    assert result.lookback_days == 400
    assert result.backfilled is False
    assert result.warnings == ()
    assert set(result.coverage) == set(ROOTS)
    assert all(count >= result.required_bars for count in result.coverage.values())
    assert len(set(result.coverage.values())) == 1  # equal decision timestamps per root
    # Coverage counts decision timestamps; the offline source emits two synthetic
    # legs (front F1 + second month F2) for every root/timestamp.
    assert result.bars.height == 2 * sum(result.coverage.values())
    assert result.bar_count == result.bars.height
    assert set(s.BAR_COLUMNS).issubset(result.bars.columns)
    assert result.bars[s.TIMESTAMP_COLUMN].max() == as_of
    # Chronological, bounded, and ending exactly at as_of.
    assert 1 < len(result.states) <= DEFAULT_MAX_WARMUP_STATES
    assert len(result.states) == DEFAULT_MAX_WARMUP_STATES
    stamps = [state.timestamp for state in result.states]
    assert stamps == sorted(stamps)
    assert stamps[-1] == as_of
    assert result.final_state is result.states[-1]
    for state in result.states:
        assert state.equity == 2_500_000.0
    final = result.final_state
    assert final.covariance is not None
    assert set(final.covariance_instruments) == set(result.roots)
    for root in result.roots:
        snapshot = final.snapshot(root)
        for feature in s.REQUIRED_FEATURES:
            value = snapshot.features.get(feature)
            assert value is not None and math.isfinite(value), (root, feature, value)
        assert final.returns[root].size > 0
    assert hydrator.is_warm(final)
    assert hydrator.required_lookback == math.ceil(MIN_REQUIRED_BARS * 7 / 5) + CALENDAR_SAFETY_DAYS
    assert hydrator.max_warmup_states == DEFAULT_MAX_WARMUP_STATES


def test_hydrate_state_window_is_bounded_by_max_warmup_states(
    repository: ParquetRepository, history: tuple[pl.DataFrame, datetime]
) -> None:
    bars, as_of = history
    repository.write_bars(bars)
    hydrator = make_hydrator(repository, max_warmup_states=8)
    result = hydrator.hydrate(ROOTS, as_of)
    assert len(result.states) == 8
    assert result.states[-1].timestamp == as_of


def test_hydrate_without_equity_leaves_states_unpriced(
    repository: ParquetRepository, history: tuple[pl.DataFrame, datetime]
) -> None:
    bars, as_of = history
    repository.write_bars(bars)
    result = make_hydrator(repository).hydrate(ROOTS, as_of)
    assert all(state.equity is None for state in result.states)


# ---------------------------------------------------------------------------
# Coverage failure (fail closed, actionable message)
# ---------------------------------------------------------------------------


def _recent_only(bars: pl.DataFrame, as_of: datetime, days: int = 90) -> pl.DataFrame:
    cutoff = as_of - timedelta(days=days)
    return bars.filter(pl.col(s.TIMESTAMP_COLUMN) > cutoff)


def test_insufficient_history_without_backfill_fails_closed(
    repository: ParquetRepository, history: tuple[pl.DataFrame, datetime]
) -> None:
    bars, as_of = history
    repository.write_bars(_recent_only(bars, as_of, days=90))
    hydrator = make_hydrator(repository)
    with pytest.raises(HydrationError) as excinfo:
        hydrator.hydrate(ROOTS, as_of)
    message = str(excinfo.value)
    assert "insufficient stored history" in message
    assert "observed coverage:" in message
    assert "CL=" in message and "ES=" in message and "NQ=" in message
    assert str(MIN_REQUIRED_BARS) in message
    assert "backfill provider is configured" in message
    assert "DatabentoLoader.ingest" in message


def test_empty_repository_fails_closed(
    repository: ParquetRepository, history: tuple[pl.DataFrame, datetime]
) -> None:
    _, as_of = history
    with pytest.raises(HydrationError) as excinfo:
        make_hydrator(repository).hydrate(ROOTS, as_of)
    assert "insufficient stored history" in str(excinfo.value)
    assert "ES=0" in str(excinfo.value)


def test_unknown_root_fails_closed(repository: ParquetRepository) -> None:
    _, as_of = make_history()
    with pytest.raises(HydrationError) as excinfo:
        make_hydrator(repository).hydrate(("ZZZ",), as_of)
    assert "ZZZ=0" in str(excinfo.value)


def test_hydrate_rejects_invalid_root_list(repository: ParquetRepository) -> None:
    _, as_of = make_history()
    with pytest.raises(HydrationError):
        make_hydrator(repository).hydrate((), as_of)


# ---------------------------------------------------------------------------
# Backfill
# ---------------------------------------------------------------------------


def test_backfill_is_invoked_persisted_and_reported(
    repository: ParquetRepository, history: tuple[pl.DataFrame, datetime]
) -> None:
    bars, as_of = history
    repository.write_bars(_recent_only(bars, as_of))
    calls: list[tuple[tuple[str, ...], datetime, datetime]] = []

    def fake_backfill(roots: list[str], start: datetime, end: datetime) -> pl.DataFrame:
        calls.append((tuple(roots), start, end))
        return bars

    hydrator = make_hydrator(repository, backfill=fake_backfill)
    result = hydrator.hydrate(ROOTS, as_of)
    assert result.backfilled is True
    assert len(calls) == 1
    assert calls[0][0] == ("CL", "ES", "NQ")
    assert calls[0][2] - calls[0][1] >= timedelta(days=result.lookback_days)
    assert any("backfilled" in warning for warning in result.warnings)
    # The backfilled history was persisted (dedupe makes re-persist safe).
    assert repository.read_bars(ROOTS).height == bars.height
    # A later hydrate finds full coverage and does not need the provider again.
    plain = make_hydrator(repository)
    again = plain.hydrate(ROOTS, as_of)
    assert again.backfilled is False
    assert dict(again.coverage) == dict(result.coverage)


def test_backfill_disabled_switch_is_honoured(
    repository: ParquetRepository, history: tuple[pl.DataFrame, datetime]
) -> None:
    bars, as_of = history
    repository.write_bars(_recent_only(bars, as_of))
    calls: list[object] = []

    def fake_backfill(*_args: object) -> pl.DataFrame:
        calls.append(_args)
        return bars

    disabled = make_hydrator(repository, backfill=fake_backfill, backfill_enabled=False)
    with pytest.raises(HydrationError) as excinfo:
        disabled.hydrate(ROOTS, as_of)
    assert "backfill is disabled" in str(excinfo.value)
    assert calls == []
    # A per-call override can re-enable backfilling.
    enabled = disabled.hydrate(ROOTS, as_of, allow_backfill=True)
    assert enabled.backfilled is True
    assert len(calls) == 1


def test_backfill_provider_failure_fails_closed(
    repository: ParquetRepository, history: tuple[pl.DataFrame, datetime]
) -> None:
    bars, as_of = history
    repository.write_bars(_recent_only(bars, as_of))

    def exploding(*_args: object) -> pl.DataFrame:
        raise ConnectionError("Databento unavailable")

    with pytest.raises(HydrationError) as excinfo:
        make_hydrator(repository, backfill=exploding).hydrate(ROOTS, as_of)
    assert "backfill failed" in str(excinfo.value)
    assert "Databento unavailable" in str(excinfo.value)

    def empty(*_args: object) -> pl.DataFrame:
        return s.empty_bars_frame()

    with pytest.raises(HydrationError) as empty_exc:
        make_hydrator(repository, backfill=empty).hydrate(ROOTS, as_of)
    assert "empty frame" in str(empty_exc.value)


def test_backfill_returning_still_short_history_fails_closed(
    repository: ParquetRepository, history: tuple[pl.DataFrame, datetime]
) -> None:
    bars, as_of = history
    repository.write_bars(_recent_only(bars, as_of, days=30))
    partial = _recent_only(bars, as_of, days=120)
    with pytest.raises(HydrationError) as excinfo:
        make_hydrator(repository, backfill=lambda *_args: partial).hydrate(ROOTS, as_of)
    assert "coverage is still short" in str(excinfo.value)


def test_backfill_ignores_rows_outside_the_requested_universe(
    repository: ParquetRepository, history: tuple[pl.DataFrame, datetime]
) -> None:
    bars, as_of = history
    repository.write_bars(_recent_only(bars, as_of))
    extra, _ = make_history(("GC",))
    payload = pl.concat([bars, extra], how="vertical_relaxed")
    result = make_hydrator(repository, backfill=lambda *_args: payload).hydrate(ROOTS, as_of)
    assert result.backfilled is True
    assert any("outside the requested universe" in warning for warning in result.warnings)
    assert set(result.roots) == set(ROOTS)


# ---------------------------------------------------------------------------
# Determinism / idempotency / no lookahead
# ---------------------------------------------------------------------------


def test_hydrate_is_idempotent(
    repository: ParquetRepository, history: tuple[pl.DataFrame, datetime]
) -> None:
    bars, as_of = history
    repository.write_bars(bars)
    hydrator = make_hydrator(repository)
    first = hydrator.hydrate(ROOTS, as_of)
    stored_rows = repository.read_bars(ROOTS).height
    second = hydrator.hydrate(ROOTS, as_of)
    assert repository.read_bars(ROOTS).height == stored_rows
    assert [state.timestamp for state in first.states] == [
        state.timestamp for state in second.states
    ]
    assert dict(first.coverage) == dict(second.coverage)
    assert first.backfilled is second.backfilled is False
    for root in first.roots:
        left = first.final_state.snapshot(root).features
        right = second.final_state.snapshot(root).features
        assert left == pytest.approx(right)


def test_hydration_has_no_lookahead(
    repository: ParquetRepository, history: tuple[pl.DataFrame, datetime]
) -> None:
    bars, as_of = history
    repository.write_bars(bars)
    hydrator = make_hydrator(repository)
    before = hydrator.hydrate(ROOTS, as_of)
    assert before.bars[s.TIMESTAMP_COLUMN].max() == as_of
    # Point-in-time invariant: everything in the window is observable at as_of.
    window_timestamps = before.bars[s.TIMESTAMP_COLUMN].n_unique()
    assert before.final_state.metadata["point_in_time_rows"] == (
        window_timestamps * len(before.roots)
    )
    rows = [state.metadata["point_in_time_rows"] for state in before.states]
    assert rows == sorted(rows)
    # Append aggressive future bars, then hydrate the same as_of again.
    future_base = bars.filter(pl.col(s.TIMESTAMP_COLUMN) == as_of)
    future_rows = pl.concat(
        [
            future_base.with_columns(
                (pl.col(s.TIMESTAMP_COLUMN) + pl.duration(days=offset)).alias(s.TIMESTAMP_COLUMN),
                (pl.col(s.CLOSE_COLUMN) * 3.0).alias(s.CLOSE_COLUMN),
                (pl.col(s.OPEN_COLUMN) * 3.0).alias(s.OPEN_COLUMN),
                (pl.col(s.HIGH_COLUMN) * 3.0).alias(s.HIGH_COLUMN),
                (pl.col(s.LOW_COLUMN) * 3.0).alias(s.LOW_COLUMN),
            )
            for offset in range(1, 31)
        ]
    )
    repository.write_bars(s.ensure_bars_schema(future_rows, context="future"))
    after = hydrator.hydrate(ROOTS, as_of)
    assert after.bars[s.TIMESTAMP_COLUMN].max() == as_of
    assert [state.timestamp for state in after.states] == [
        state.timestamp for state in before.states
    ]
    for root in after.roots:
        assert after.final_state.snapshot(root).features == pytest.approx(
            before.final_state.snapshot(root).features
        )


# ---------------------------------------------------------------------------
# Warm-state validation helper
# ---------------------------------------------------------------------------


def test_is_warm_accepts_a_complete_state_and_rejects_a_cold_one(
    repository: ParquetRepository,
) -> None:
    hydrator = make_hydrator(repository)
    warm = make_market_state(roots=("ES", "NQ"))
    assert hydrator.is_warm(warm) is True
    cold = make_market_state(roots=("ES",), with_covariance=False, with_returns=False)
    assert hydrator.is_warm(cold) is False
    with pytest.raises(TypeError):
        hydrator.is_warm("not-a-state")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Construction validation / introspection
# ---------------------------------------------------------------------------


def test_constructor_validation(repository: ParquetRepository) -> None:
    with pytest.raises(TypeError):
        ColdStartHydrator(repository="not-a-repository")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        make_hydrator(repository, lookback_days=0)
    with pytest.raises(ValueError):
        make_hydrator(repository, min_bars_per_root=0)
    with pytest.raises(ValueError):
        make_hydrator(repository, max_warmup_states=0)
    with pytest.raises(ValueError):
        make_hydrator(repository, schema="tbbo")


def test_required_bars_and_lookback_introspection(repository: ParquetRepository) -> None:
    default = make_hydrator(repository)
    assert default.required_bars == MIN_REQUIRED_BARS
    assert default.required_lookback == (
        math.ceil(MIN_REQUIRED_BARS * 7 / 5) + CALENDAR_SAFETY_DAYS
    )
    stricter = make_hydrator(repository, min_bars_per_root=400)
    assert stricter.required_bars == 400
    assert stricter.required_lookback > default.required_lookback
    relaxed = make_hydrator(repository, min_bars_per_root=100)
    assert relaxed.required_bars == MIN_REQUIRED_BARS
    assert relaxed.effective_lookback() == max(400, default.required_lookback)
    assert default.repository is repository


def test_hydrate_now_uses_the_injected_clock(
    repository: ParquetRepository, history: tuple[pl.DataFrame, datetime]
) -> None:
    bars, as_of = history
    repository.write_bars(bars)
    hydrator = make_hydrator(repository, clock=lambda: as_of)
    result = hydrator.hydrate_now(ROOTS)
    assert result.as_of == as_of


def test_lookback_is_expanded_when_configured_too_small(
    repository: ParquetRepository, history: tuple[pl.DataFrame, datetime]
) -> None:
    bars, as_of = history
    repository.write_bars(bars)
    hydrator = make_hydrator(repository, lookback_days=MIN_REQUIRED_BARS)
    result = hydrator.hydrate(ROOTS, as_of)
    assert result.lookback_days == hydrator.required_lookback
    assert any("lookback expanded" in warning for warning in result.warnings)
    assert all(count >= MIN_REQUIRED_BARS for count in result.coverage.values())
