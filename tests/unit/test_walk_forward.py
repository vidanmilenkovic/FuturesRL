"""Unit tests for purged/embargoed walk-forward cross-validation."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
from configs.strategy import WalkForwardConfig
from src.backtest.walk_forward import PurgedWalkForwardSplit, WalkForwardFold


def test_expanding_split_respects_purge_and_no_overlap() -> None:
    splitter = PurgedWalkForwardSplit(train_days=30, test_days=10, purge_days=3, embargo_days=2)
    folds = splitter.split(120)
    assert folds
    for fold in folds:
        assert isinstance(fold, WalkForwardFold)
        assert fold.train_indices
        assert fold.test_indices
        assert fold.test_start - fold.train_end == 3
        assert fold.purge_gap == 3
        assert max(fold.train_indices) < min(fold.test_indices)
        assert not set(fold.train_indices) & set(fold.test_indices)
        assert fold.train_indices == tuple(sorted(set(fold.train_indices)))
        assert fold.test_indices == tuple(sorted(set(fold.test_indices)))
        assert fold.train_start == 0  # expanding
        assert fold.train_end == fold.test_start - 3
        assert fold.test_end == fold.test_start + 10
        assert fold.test_end <= 120
        assert fold.train_size + fold.test_size > 0
    # First fold is the initial full training window.
    assert folds[0].train_indices == tuple(range(30))
    assert folds[0].test_indices == tuple(range(33, 43))
    # Fold indices and test windows advance by step_days (= test_days).
    assert [fold.fold_index for fold in folds] == list(range(len(folds)))
    assert folds[1].test_start - folds[0].test_start == 10


def test_n_splits_and_insufficient_history() -> None:
    splitter = PurgedWalkForwardSplit(train_days=30, test_days=10, purge_days=3, embargo_days=2)
    assert splitter.n_splits(30 + 3 + 10 - 1) == 0
    assert splitter.n_splits(30 + 3 + 10) == 1
    assert splitter.split(10) == []
    assert splitter.n_splits(120) == len(splitter.split(120))
    with pytest.raises(TypeError):
        splitter.split(120.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        splitter.split(-1)
    with pytest.raises(TypeError):
        splitter.n_splits(True)


def test_sliding_window_split() -> None:
    splitter = PurgedWalkForwardSplit(
        train_days=30,
        test_days=10,
        purge_days=2,
        embargo_days=1,
        expanding=False,
    )
    folds = splitter.split(120)
    for fold in folds:
        assert fold.train_end - fold.train_start <= 30
        assert fold.test_start - fold.train_end == 2
        assert max(fold.train_indices) < min(fold.test_indices)
    assert folds[-1].train_start > 0
    assert folds[0].train_start == 0


def test_embargo_excludes_post_test_observations() -> None:
    # A step larger than the test window leaves observed data between test
    # blocks; the embargo must remove the first ``embargo_days`` of it from the
    # next training set.
    splitter = PurgedWalkForwardSplit(
        train_days=20,
        test_days=10,
        purge_days=2,
        embargo_days=4,
        step_days=20,
    )
    folds = splitter.split(200)
    assert len(folds) >= 3
    for previous, current in pairwise(folds):
        embargo_window = set(range(previous.test_end, previous.test_end + splitter.embargo_days))
        assert not embargo_window & set(current.train_indices)
        # Data beyond the embargo window may be trained on (it is in the past
        # relative to the current test block).
        assert any(index >= previous.test_end for index in current.train_indices)


def test_step_days_controls_spacing() -> None:
    splitter = PurgedWalkForwardSplit(
        train_days=20, test_days=10, purge_days=2, embargo_days=1, step_days=15
    )
    folds = splitter.split(100)
    assert folds[1].test_start - folds[0].test_start == 15


def test_from_config_matches_manual_construction() -> None:
    config = WalkForwardConfig(
        train_days=40, test_days=12, purge_days=4, embargo_days=3, step_days=12
    )
    from_config = PurgedWalkForwardSplit.from_config(config)
    manual = PurgedWalkForwardSplit(
        train_days=40, test_days=12, purge_days=4, embargo_days=3, step_days=12
    )
    assert from_config.to_dict() == manual.to_dict()
    assert [fold.to_dict() for fold in from_config.split(150)] == [
        fold.to_dict() for fold in manual.split(150)
    ]
    sliding = PurgedWalkForwardSplit.from_config(config, expanding=False)
    assert sliding.expanding is False


def test_split_frame_alias_and_to_dict() -> None:
    splitter = PurgedWalkForwardSplit(train_days=10, test_days=5, purge_days=1, embargo_days=1)
    assert [f.to_dict() for f in splitter.split_frame(40)] == [
        f.to_dict() for f in splitter.split(40)
    ]
    payload = splitter.to_dict()
    assert payload["train_days"] == 10
    assert payload["test_days"] == 5
    assert payload["expanding"] is True


def test_future_data_mutation_cannot_change_earlier_folds() -> None:
    """The split is a pure function of the sample count: no data leakage."""
    splitter = PurgedWalkForwardSplit(train_days=30, test_days=10, purge_days=3, embargo_days=2)
    data = np.arange(120.0)
    original = splitter.split(len(data))
    fold = original[0]
    before = {
        "train": tuple(fold.train_indices),
        "test": tuple(fold.test_indices),
    }
    # Mutate every observation that is *in the future* of the first fold.
    mutated = data.copy()
    mutated[fold.test_end :] = np.nan
    mutated[fold.test_start :] = 1e12
    rebuilt = splitter.split(len(mutated))
    rebuilt_fold = rebuilt[0]
    assert {
        "train": tuple(rebuilt_fold.train_indices),
        "test": tuple(rebuilt_fold.test_indices),
    } == before
    # Earlier folds are identical even when a later test block is extended.
    for original_fold, rebuilt_original in zip(original, rebuilt, strict=True):
        assert tuple(rebuilt_original.train_indices) == tuple(original_fold.train_indices)
        assert tuple(rebuilt_original.test_indices) == tuple(original_fold.test_indices)


def test_splitter_parameter_validation() -> None:
    with pytest.raises(ValueError):
        PurgedWalkForwardSplit(train_days=0)
    with pytest.raises(ValueError):
        PurgedWalkForwardSplit(test_days=0)
    with pytest.raises(ValueError):
        PurgedWalkForwardSplit(train_days=5, purge_days=5)
    with pytest.raises(ValueError):
        PurgedWalkForwardSplit(test_days=5, embargo_days=5)
    with pytest.raises(ValueError):
        PurgedWalkForwardSplit(step_days=0)
    with pytest.raises(TypeError):
        PurgedWalkForwardSplit(train_days=10.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        PurgedWalkForwardSplit(expanding=1)  # type: ignore[arg-type]


def test_fold_validation() -> None:
    with pytest.raises(TypeError):
        WalkForwardFold(
            fold_index=1.5,  # type: ignore[arg-type]
            train_indices=(0,),
            test_indices=(2,),
            train_start=0,
            train_end=1,
            test_start=1,
            test_end=2,
        )
    valid = {
        "fold_index": 0,
        "train_indices": (0, 1),
        "test_indices": (3, 4),
        "train_start": 0,
        "train_end": 3,
        "test_start": 3,
        "test_end": 5,
    }
    assert WalkForwardFold(**valid).train_size == 2
    assert WalkForwardFold(**valid).test_size == 2
    with pytest.raises(ValueError):
        WalkForwardFold(**{**valid, "train_indices": ()})
    with pytest.raises(ValueError):
        WalkForwardFold(**{**valid, "test_indices": ()})
    with pytest.raises(ValueError):
        WalkForwardFold(**{**valid, "train_indices": (1, 0)})
    with pytest.raises(ValueError):
        WalkForwardFold(**{**valid, "train_indices": (0, 5)})
    with pytest.raises(ValueError):
        WalkForwardFold(**{**valid, "test_indices": (3, 9)})
    with pytest.raises(ValueError):
        WalkForwardFold(**{**valid, "test_start": 1})  # overlaps training
    with pytest.raises(ValueError):
        WalkForwardFold(**{**valid, "train_start": 3})
    with pytest.raises(ValueError):
        WalkForwardFold(**{**valid, "train_end": 10, "train_indices": (0, 1)})
    with pytest.raises(ValueError):
        WalkForwardFold(**{**valid, "fold_index": -1})
    with pytest.raises(TypeError):
        WalkForwardFold(**{**valid, "train_start": 1.5})
