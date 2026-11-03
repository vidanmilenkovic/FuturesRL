"""Unit tests for DRL checkpoint metadata, universe guard and observation alignment.

Phase 11A: a trained checkpoint must be consumed in the exact root/feature order it
was trained on. The sidecar written by the trainer (``<checkpoint>.zip.meta.json``)
is the source of truth; these tests pin the contract without importing torch.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from src.strategies.allocators.drl import (
    DRLAllocator,
    checkpoint_bind_preview,
    checkpoint_strategy_params_patch,
    checkpoint_universe_bind_patch,
    load_checkpoint_metadata,
)

from tests.fixtures.market import make_market_state


class _FakeModel:
    """Duck-typed SB3 model: fixed action plus an optional observation space."""

    def __init__(self, action: object, *, observation_shape: tuple[int, ...] | None = None) -> None:
        self.action = action
        self.observation_space = (
            None if observation_shape is None else SimpleNamespace(shape=observation_shape)
        )

    def predict(self, observation: object, deterministic: bool = True) -> tuple[object, None]:
        assert deterministic is True
        assert observation is not None
        return self.action, None


def _write_meta(
    model_path: Path,
    *,
    universe: tuple[str, ...] = ("ES", "NQ", "CL"),
    feature_names: tuple[str, ...] = ("ret_1d",),
    observation_shape: tuple[int, ...] | None = None,
    action_shape: tuple[int, ...] | None = None,
) -> Path:
    """Write a trainer-compatible sidecar next to ``model_path``."""
    n_assets = len(universe)
    n_features = len(feature_names)
    payload: dict[str, object] = {
        "universe": list(universe),
        "feature_names": list(feature_names),
        "observation_shape": list(observation_shape or (n_assets * n_features + n_assets + 1,)),
        "action_shape": list(action_shape or (n_assets,)),
        "algorithm": "PPO",
        "total_timesteps": 64,
        "strategy_id": "drl_ppo_llm_risk",
        "trained_at": "2026-01-01T00:00:00+00:00",
    }
    meta_path = Path(f"{model_path}.meta.json")
    meta_path.write_text(json.dumps(payload), encoding="utf-8")
    return meta_path


def _state_with_features(
    roots: tuple[str, ...] = ("ES", "NQ", "CL"),
    *,
    features: dict[str, dict[str, float]] | None = None,
) -> Any:
    return make_market_state(roots=roots, features=features)


# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------


def test_checkpoint_metadata_loads_universe_features_and_shapes(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.zip"
    _write_meta(
        checkpoint,
        universe=("ES", "NQ", "CL"),
        feature_names=("ret_1d", "vol_20d"),
    )
    meta = load_checkpoint_metadata(checkpoint)
    assert meta.universe == ("ES", "NQ", "CL")
    assert meta.feature_names == ("ret_1d", "vol_20d")
    assert meta.observation_dimension == 3 * 2 + 3 + 1
    assert meta.algorithm == "PPO"
    assert meta.total_timesteps == 64
    assert meta.meta_path == Path(f"{checkpoint}.meta.json")
    assert meta.to_dict()["universe"] == ["ES", "NQ", "CL"]

    allocator = DRLAllocator(model=_FakeModel(np.zeros(3)), model_path=checkpoint)
    assert allocator.expected_roots == ("ES", "NQ", "CL")
    assert allocator.expected_feature_names == ("ret_1d", "vol_20d")
    assert allocator.checkpoint_metadata is not None


def test_metadata_missing_preserves_legacy_behaviour_with_warning(tmp_path: Path) -> None:
    checkpoint = tmp_path / "legacy.zip"  # no sidecar written
    model = _FakeModel(np.array([0.1, 0.2]))
    allocator = DRLAllocator(model=model, model_path=checkpoint)
    state = _state_with_features(("ES", "NQ"))
    with pytest.warns(UserWarning, match="no metadata sidecar"):
        observation = allocator.build_observation(state, ("ES", "NQ"))
    # Legacy layout: candidates order, configured feature names, not checkpoint metadata.
    assert observation.size == 2 * len(allocator.feature_names) + 2 + 1
    assert allocator.expected_roots == ()
    weights = allocator.allocate(("ES", "NQ"), state)
    assert set(weights.as_dict()) == {"ES", "NQ"}
    with pytest.raises(RuntimeError, match="no metadata sidecar"):
        allocator.universe_bind_patch()


def test_malformed_metadata_is_actionable(tmp_path: Path) -> None:
    checkpoint = tmp_path / "broken.zip"
    Path(f"{checkpoint}.meta.json").write_text(json.dumps({"feature_names": ["ret_1d"]}))
    with pytest.raises(ValueError, match="universe"):
        load_checkpoint_metadata(checkpoint)
    allocator = DRLAllocator(model=_FakeModel(np.zeros(3)), model_path=checkpoint)
    with pytest.raises(RuntimeError, match="invalid DRL checkpoint metadata"):
        _ = allocator.expected_roots


# ---------------------------------------------------------------------------
# Observation alignment
# ---------------------------------------------------------------------------


def test_observation_uses_checkpoint_root_order(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.zip"
    _write_meta(checkpoint, universe=("ES", "NQ", "CL"), feature_names=("ret_1d",))
    state = _state_with_features(
        features={
            "ES": {"ret_1d": 0.01},
            "NQ": {"ret_1d": 0.02},
            "CL": {"ret_1d": -0.03},
        }
    )
    allocator = DRLAllocator(model=_FakeModel(np.zeros(3)), model_path=checkpoint)
    # Reordered candidates: observation still follows ES, NQ, CL (NQ not selected).
    observation = allocator.build_observation(state, ("CL", "ES"))
    assert observation.size == 3 + 3 + 1
    features = observation[:3]
    assert features == pytest.approx([0.01, 0.0, -0.03])
    assert observation[3:6] == pytest.approx([0.0, 0.0, 0.0])  # held weights flat
    assert observation[-1] == pytest.approx(1.0)


def test_missing_roots_are_padded_flat(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.zip"
    _write_meta(checkpoint, universe=("ES", "NQ", "CL"), feature_names=("ret_1d",))
    # State has no NQ row at all (partial data/warm-up): its block must be flat.
    state = _state_with_features(
        ("ES", "CL"), features={"ES": {"ret_1d": 0.01}, "CL": {"ret_1d": -0.03}}
    )
    allocator = DRLAllocator(model=_FakeModel(np.zeros(3)), model_path=checkpoint)
    observation = allocator.build_observation(state, ("ES", "CL"))
    assert observation[:3] == pytest.approx([0.01, 0.0, -0.03])


def test_feature_order_follows_checkpoint_metadata(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.zip"
    _write_meta(checkpoint, universe=("ES", "NQ"), feature_names=("vol_20d", "ret_1d"))
    state = _state_with_features(
        ("ES", "NQ"),
        features={"ES": {"ret_1d": 0.01, "vol_20d": 0.4}, "NQ": {"ret_1d": 0.02, "vol_20d": 0.5}},
    )
    allocator = DRLAllocator(model=_FakeModel(np.zeros(2)), model_path=checkpoint)
    observation = allocator.build_observation(state, ("NQ", "ES"))
    features = observation[:4].reshape(2, 2)
    # metadata column order: vol_20d, ret_1d (not the configured feature order)
    assert features[0] == pytest.approx([0.4, 0.01])
    assert features[1] == pytest.approx([0.5, 0.02])


def test_observation_shape_mismatch_is_actionable(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.zip"
    _write_meta(checkpoint, observation_shape=(999,))
    state = _state_with_features()
    allocator = DRLAllocator(model=_FakeModel(np.zeros(3)), model_path=checkpoint)
    with pytest.raises(RuntimeError, match=r"observation mismatch.*999.*7 entries"):
        allocator.allocate(("ES", "NQ", "CL"), state)

    # Model observation-space differs from the aligned observation.
    checkpoint2 = tmp_path / "model2.zip"
    _write_meta(checkpoint2, universe=("ES", "NQ", "CL"), feature_names=("ret_1d",))
    state_allocator = DRLAllocator(
        model=_FakeModel(np.zeros(3), observation_shape=(9,)), model_path=checkpoint2
    )
    with pytest.raises(RuntimeError, match=r"observation-space mismatch.*9.*7"):
        state_allocator.allocate(("ES", "NQ", "CL"), state)


# ---------------------------------------------------------------------------
# Action mapping
# ---------------------------------------------------------------------------


def test_action_maps_back_to_actual_candidates_and_ignores_extra(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.zip"
    _write_meta(checkpoint, universe=("ES", "NQ", "CL"), feature_names=("ret_1d",))
    state = _state_with_features(("ES", "NQ", "CL", "ZN"))
    allocator = DRLAllocator(
        model=_FakeModel(np.array([0.5, -0.25, 0.1])),
        model_path=checkpoint,
        gross_leverage=1.0,
    )
    # Reordered + extra candidate: action indexes ES, NQ, CL; ZN is ignored.
    weights = allocator.allocate(("ZN", "CL", "ES", "NQ"), state)
    assert weights.as_dict() == pytest.approx({"ZN": 0.0, "CL": 0.1, "ES": 0.5, "NQ": -0.25})
    # Missing checkpoint root in candidates: its action is discarded.
    weights2 = allocator.allocate(("ES", "CL"), state)
    assert weights2.as_dict() == pytest.approx({"ES": 0.5, "CL": 0.1})
    # Gross-target semantics preserved: normalized mode scales mapped weights.
    normalised = DRLAllocator(
        model=_FakeModel(np.array([2.0, 1.0, 1.0])),
        model_path=checkpoint,
        action_mode="normalized",
        gross_leverage=2.0,
    )
    assert normalised.allocate(("ES", "CL"), state).gross == pytest.approx(4.0 * 2.0 / 4.0)


def test_action_shape_mismatch_is_actionable(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.zip"
    _write_meta(checkpoint, universe=("ES", "NQ", "CL"))
    state = _state_with_features()
    allocator = DRLAllocator(model=_FakeModel(np.zeros(4)), model_path=checkpoint)
    with pytest.raises(ValueError, match=r"checkpoint expects 3 roots"):
        allocator.allocate(("ES", "NQ", "CL"), state)


# ---------------------------------------------------------------------------
# Bind patch surface
# ---------------------------------------------------------------------------


def test_bind_patch_and_preview(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.zip"
    _write_meta(checkpoint, universe=("ES", "NQ", "CL"))
    assert checkpoint_universe_bind_patch(checkpoint) == {
        "roots": ["ES", "NQ", "CL"],
        "top_n": 3,
    }
    assert checkpoint_strategy_params_patch(checkpoint) == {"roots": "ES,NQ,CL", "top_n": 3}
    preview = checkpoint_bind_preview(checkpoint)
    assert preview["metadata_available"] is True
    assert preview["universe"] == ["ES", "NQ", "CL"]
    assert preview["roots_patch"] == {"roots": ["ES", "NQ", "CL"], "top_n": 3}
    assert preview["params_patch"] == {"roots": "ES,NQ,CL", "top_n": 3}
    assert preview["warning"] is None
    assert preview["checkpoint_name"] == "model"
    # Allocator exposes the same patch via the loaded sidecar.
    allocator = DRLAllocator(model=_FakeModel(np.zeros(3)), model_path=checkpoint)
    assert allocator.universe_bind_patch() == {"roots": ["ES", "NQ", "CL"], "top_n": 3}


def test_bind_preview_missing_metadata_is_non_raising(tmp_path: Path) -> None:
    checkpoint = tmp_path / "legacy.zip"
    preview = checkpoint_bind_preview(checkpoint)
    assert preview["metadata_available"] is False
    assert preview["roots_patch"] is None
    assert preview["params_patch"] is None
    assert preview["universe"] == []
    assert "metadata" in str(preview["warning"]).lower()
    with pytest.raises(FileNotFoundError):
        checkpoint_universe_bind_patch(checkpoint)
