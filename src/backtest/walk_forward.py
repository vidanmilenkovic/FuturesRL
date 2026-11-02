"""Purged and embargoed walk-forward cross-validation over integer index ranges.

Walk-forward validation is the anti-look-ahead gate of the backtester: a model is
only ever trained on observations that are strictly in the past relative to the
test window, with a ``purge`` gap around the boundary (to remove label overlap
across the split) and an ``embargo`` window after each test block (to remove
serial-correlation bleed-through into the next training window).

The splitter is a pure function of ``n_samples`` and its configuration. That
property is what makes leakage testable: mutating any future observation cannot
change the index sets of an earlier fold, because the split never looks at the
data itself.

Indices are half-open ``[start, end)`` integer ranges. A fold guarantees:

* ``max(train_indices) < min(test_indices)`` -- no training observation is at or
  after the first test observation;
* ``train_indices`` and ``test_indices`` are disjoint;
* ``test_start - train_end >= purge_days`` -- the purge gap is respected;
* no training index falls inside a previous fold's embargo window.
"""

from __future__ import annotations

from dataclasses import dataclass

from configs.strategy import WalkForwardConfig


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    """One purged/embargoed train-test fold expressed as index sets."""

    fold_index: int
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    train_start: int
    train_end: int
    test_start: int
    test_end: int

    def __post_init__(self) -> None:
        if isinstance(self.fold_index, bool) or not isinstance(self.fold_index, int):
            raise TypeError("fold_index must be an int")
        if self.fold_index < 0:
            raise ValueError("fold_index must be non-negative")
        for name in ("train_start", "train_end", "test_start", "test_end"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.train_start > self.train_end:
            raise ValueError("train_start must not exceed train_end")
        if self.test_start > self.test_end:
            raise ValueError("test_start must not exceed test_end")
        if self.train_end > self.test_start:
            raise ValueError("training data must end at or before the test window starts")
        if not self.train_indices:
            raise ValueError("a fold must have at least one training index")
        if not self.test_indices:
            raise ValueError("a fold must have at least one test index")
        if tuple(sorted(set(self.train_indices))) != self.train_indices:
            raise ValueError("train_indices must be sorted and unique")
        if tuple(sorted(set(self.test_indices))) != self.test_indices:
            raise ValueError("test_indices must be sorted and unique")
        if self.train_indices[0] < self.train_start or self.train_indices[-1] >= self.train_end:
            raise ValueError("train_indices fall outside [train_start, train_end)")
        if self.test_indices[0] < self.test_start or self.test_indices[-1] >= self.test_end:
            raise ValueError("test_indices fall outside [test_start, test_end)")
        if self.train_indices[-1] >= self.test_indices[0]:
            raise ValueError("training indices must all precede the test indices")

    @property
    def train_size(self) -> int:
        """Number of training observations in the fold."""
        return len(self.train_indices)

    @property
    def test_size(self) -> int:
        """Number of test observations in the fold."""
        return len(self.test_indices)

    @property
    def purge_gap(self) -> int:
        """Number of excluded samples between the training and test windows."""
        return self.test_start - self.train_end

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable representation (index lists are materialised)."""
        return {
            "fold_index": self.fold_index,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "train_size": self.train_size,
            "test_size": self.test_size,
            "purge_gap": self.purge_gap,
            "train_indices": list(self.train_indices),
            "test_indices": list(self.test_indices),
        }


class PurgedWalkForwardSplit:
    """Expanding or sliding purged/embargoed walk-forward splitter.

    Parameters mirror :class:`configs.strategy.WalkForwardConfig`:

    ``train_days``
        Number of observations in the initial training window (and in every
        sliding training window).
    ``test_days``
        Number of observations in each test window.
    ``purge_days``
        Observations dropped before every test window (label-overlap purge).
    ``embargo_days``
        Observations dropped immediately after each test window before they may
        appear in a later training window.
    ``step_days``
        Distance between consecutive test starts (defaults to ``test_days``,
        i.e. back-to-back test windows).
    """

    def __init__(
        self,
        *,
        train_days: int = 252,
        test_days: int = 63,
        purge_days: int = 5,
        embargo_days: int = 5,
        step_days: int | None = None,
        expanding: bool = True,
    ) -> None:
        for name, value, minimum in (
            ("train_days", train_days, 1),
            ("test_days", test_days, 1),
            ("purge_days", purge_days, 0),
            ("embargo_days", embargo_days, 0),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int")
            if value < minimum:
                raise ValueError(f"{name} must be >= {minimum}")
        if purge_days >= train_days:
            raise ValueError("purge_days must be smaller than train_days")
        if embargo_days >= test_days:
            raise ValueError("embargo_days must be smaller than test_days")
        if step_days is None:
            step_days = test_days
        if isinstance(step_days, bool) or not isinstance(step_days, int):
            raise TypeError("step_days must be an int")
        if step_days < 1:
            raise ValueError("step_days must be >= 1")
        if not isinstance(expanding, bool):
            raise TypeError("expanding must be a bool")
        self.train_days = train_days
        self.test_days = test_days
        self.purge_days = purge_days
        self.embargo_days = embargo_days
        self.step_days = step_days
        self.expanding = expanding

    @classmethod
    def from_config(
        cls, config: WalkForwardConfig, *, expanding: bool = True
    ) -> PurgedWalkForwardSplit:
        """Build a splitter from a validated declarative walk-forward config."""
        return cls(
            train_days=config.train_days,
            test_days=config.test_days,
            purge_days=config.purge_days,
            embargo_days=config.embargo_days,
            step_days=config.step_days,
            expanding=expanding,
        )

    # -- splitting ------------------------------------------------------------

    def n_splits(self, n_samples: int) -> int:
        """Number of complete folds available for ``n_samples`` observations."""
        self._validate_n_samples(n_samples)
        first_test_start = self.train_days + self.purge_days
        if n_samples < first_test_start + self.test_days:
            return 0
        return 1 + (n_samples - first_test_start - self.test_days) // self.step_days

    def split(self, n_samples: int) -> list[WalkForwardFold]:
        """Return every complete walk-forward fold for ``n_samples``.

        An empty list is returned when there is not enough history for a single
        complete train/purge/test cycle; that is a configuration outcome, not an
        error (callers decide whether to skip the run or extend the data).
        """
        self._validate_n_samples(n_samples)
        folds: list[WalkForwardFold] = []
        embargo_windows: list[tuple[int, int]] = []
        for fold_index in range(self.n_splits(n_samples)):
            test_start = self.train_days + self.purge_days + fold_index * self.step_days
            test_end = test_start + self.test_days
            train_end = test_start - self.purge_days
            train_start = 0 if self.expanding else max(0, train_end - self.train_days)
            train_indices = self._train_indices(train_start, train_end, embargo_windows)
            test_indices = tuple(range(test_start, test_end))
            folds.append(
                WalkForwardFold(
                    fold_index=fold_index,
                    train_indices=train_indices,
                    test_indices=test_indices,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                )
            )
            embargo_windows.append((test_end, test_end + self.embargo_days))
        return folds

    def split_frame(self, frame_length: int) -> list[WalkForwardFold]:
        """Alias of :meth:`split` using an explicit sample count."""
        return self.split(frame_length)

    def _train_indices(
        self,
        train_start: int,
        train_end: int,
        embargo_windows: list[tuple[int, int]],
    ) -> tuple[int, ...]:
        """Materialise training indices minus all previous embargo windows."""
        excluded: set[int] = set()
        for window_start, window_end in embargo_windows:
            excluded.update(range(window_start, min(window_end, train_end)))
        return tuple(index for index in range(train_start, train_end) if index not in excluded)

    @staticmethod
    def _validate_n_samples(n_samples: int) -> None:
        if isinstance(n_samples, bool) or not isinstance(n_samples, int):
            raise TypeError("n_samples must be an int")
        if n_samples < 0:
            raise ValueError("n_samples must be non-negative")

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable configuration description (run metadata)."""
        return {
            "train_days": self.train_days,
            "test_days": self.test_days,
            "purge_days": self.purge_days,
            "embargo_days": self.embargo_days,
            "step_days": self.step_days,
            "expanding": self.expanding,
        }
