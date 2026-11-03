"""Unit tests for the ``python -m src.strategies.train_drl`` CLI (Phase 9)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest
from src.strategies._observation import DEFAULT_OBSERVATION_FEATURES

REPO_ROOT = Path(__file__).resolve().parents[2]
STRATEGY_PATH = REPO_ROOT / "configs" / "strategies" / "drl_ppo_llm_risk.yaml"


def _run_cli(
    *arguments: str,
    env: dict[str, str] | None = None,
    cwd: Path = REPO_ROOT,
) -> subprocess.CompletedProcess[str]:
    """Run the training CLI as a subprocess from the repository root."""
    return subprocess.run(
        [sys.executable, "-m", "src.strategies.train_drl", *arguments],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        env=env,
    )


@pytest.fixture(scope="module")
def trained_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, object]]:
    """Train one tiny PPO checkpoint offline and return ``(checkpoint, summary)``."""
    output_dir = tmp_path_factory.mktemp("train_drl")
    checkpoint = output_dir / "model.zip"
    completed = _run_cli(
        "--strategy",
        str(STRATEGY_PATH),
        "--offline",
        "--timesteps",
        "64",
        "--output",
        str(checkpoint),
        "--json",
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip(), "CLI printed no JSON summary"
    summary = json.loads(completed.stdout)
    assert isinstance(summary, dict)
    return checkpoint, summary


def test_cli_writes_checkpoint_and_metadata(
    trained_checkpoint: tuple[Path, dict[str, object]],
) -> None:
    checkpoint, summary = trained_checkpoint
    meta_path = Path(f"{checkpoint}.meta.json")
    assert checkpoint.is_file()
    assert checkpoint.stat().st_size > 0
    assert meta_path.is_file()
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    # The sidecar records the exact training provenance (checkpoint/model contract).
    assert metadata["status"] == "ok"
    assert metadata["strategy_id"] == "drl_ppo_llm_risk"
    assert metadata["strategy_path"].endswith("drl_ppo_llm_risk.yaml")
    assert metadata["algorithm"] == "PPO"
    assert metadata["sb3_algorithm"] == "PPO"
    assert metadata["policy"] == "MlpPolicy"
    assert metadata["total_timesteps"] == 64
    assert metadata["seed"] == 7
    assert metadata["device"] == "cpu"
    assert metadata["offline"] is True
    assert metadata["model_path"] == str(checkpoint)
    assert metadata["meta_path"] == str(meta_path)
    assert metadata["feature_names"] == list(DEFAULT_OBSERVATION_FEATURES)
    # Offline synthetic bars are single-expiration: term structure/carry are the
    # only canonical features the source cannot produce and they are zero-filled.
    assert set(metadata["feature_fallbacks"]) == {"term_slope", "carry_annualized"}
    universe = metadata["universe"]
    assert isinstance(universe, list) and universe
    # observation = n_assets * n_features + n_assets (weights) + 1 (equity ratio)
    expected_observation = len(universe) * len(DEFAULT_OBSERVATION_FEATURES) + len(universe) + 1
    assert metadata["observation_shape"] == [expected_observation]
    assert metadata["observation_bounds"] == {
        "low": -1_000_000.0,
        "high": 1_000_000.0,
        "dtype": "float32",
        "clipped": True,
    }
    assert metadata["action_shape"] == [len(universe)]
    assert int(metadata["usable_rows"]) >= 2
    assert int(metadata["periods"]) >= 2
    assert float(metadata["duration_seconds"]) >= 0.0
    assert float(metadata["total_duration_seconds"]) >= 0.0
    assert isinstance(metadata["sb3_version"], str) and metadata["sb3_version"]
    assert metadata["git_commit"] is None or isinstance(metadata["git_commit"], str)
    trained_at = datetime.fromisoformat(str(metadata["trained_at"]))
    assert trained_at.tzinfo is not None
    assert trained_at.utcoffset() is not None

    # stdout JSON summary and sidecar describe the same run.
    assert summary == metadata


def test_cli_creates_missing_output_directories(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "deeper" / "model"  # no .zip suffix on purpose
    completed = _run_cli(
        "--offline",
        "--timesteps",
        "8",
        "--periods",
        "160",
        "--output",
        str(output),
        "--json",
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    checkpoint = Path(str(output) + ".zip")
    assert checkpoint.is_file()
    assert Path(f"{checkpoint}.meta.json").is_file()
    assert summary["model_path"] == str(checkpoint)
    assert summary["total_timesteps"] == 8


@pytest.mark.parametrize("timesteps", ["0", "-3", "abc", "1.5"])
def test_cli_rejects_invalid_timesteps(timesteps: str, tmp_path: Path) -> None:
    completed = _run_cli("--timesteps", timesteps, "--output", str(tmp_path / "model.zip"))
    assert completed.returncode == 2
    assert "Traceback" not in completed.stderr
    assert "timesteps" in completed.stderr.lower()


def test_cli_rejects_unknown_algorithm(tmp_path: Path) -> None:
    completed = _run_cli("--algorithm", "DQN", "--output", str(tmp_path / "model.zip"), "--json")
    assert completed.returncode == 2
    assert "Traceback" not in completed.stderr
    assert "unsupported algorithm" in completed.stderr.lower()


def test_cli_missing_strategy_config_is_actionable(tmp_path: Path) -> None:
    completed = _run_cli(
        "--strategy",
        str(tmp_path / "does_not_exist.yaml"),
        "--output",
        str(tmp_path / "model.zip"),
        "--json",
    )
    assert completed.returncode == 2
    assert "Traceback" not in completed.stderr
    assert "strategy config not found" in completed.stderr.lower()


def test_cli_unwritable_output_path_is_actionable(tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    completed = _run_cli(
        "--offline",
        "--timesteps",
        "8",
        "--output",
        str(blocker / "model.zip"),
        "--json",
    )
    assert completed.returncode == 2
    assert "Traceback" not in completed.stderr
    assert "not writable" in completed.stderr.lower()


def test_cli_missing_stable_baselines3_is_actionable(tmp_path: Path) -> None:
    """Simulate an environment without the optional DRL stack (clean exit 2)."""
    stub_package = tmp_path / "stubs" / "stable_baselines3"
    stub_package.mkdir(parents=True)
    (stub_package / "__init__.py").write_text(
        "raise ImportError('simulated missing stable_baselines3')\n", encoding="utf-8"
    )
    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(stub_package.parent) if not existing else f"{stub_package.parent}{os.pathsep}{existing}"
    )
    output = tmp_path / "model.zip"
    completed = _run_cli(
        "--offline",
        "--timesteps",
        "8",
        "--output",
        str(output),
        "--json",
        env=environment,
    )
    assert completed.returncode == 2
    assert "Traceback" not in completed.stderr
    assert "stable-baselines3" in completed.stderr
    assert not output.exists()


def test_cli_explicit_algorithm_override(
    trained_checkpoint: tuple[Path, dict[str, object]],
) -> None:
    """``--algorithm`` normalises case and is recorded in the summary."""
    _, summary = trained_checkpoint
    assert summary["algorithm"] == "PPO"


def test_cli_help_exits_zero() -> None:
    completed = _run_cli("--help")
    assert completed.returncode == 0
    assert "--offline" in completed.stdout
    assert "--timesteps" in completed.stdout
    assert "--output" in completed.stdout


def test_cli_cppo_maps_to_ppo(tmp_path: Path) -> None:
    """``CPPO`` (constrained PPO) trains as PPO with the risk-shaped reward."""
    completed = _run_cli(
        "--offline",
        "--algorithm",
        "CPPO",
        "--timesteps",
        "4",
        "--periods",
        "160",
        "--output",
        str(tmp_path / "cppo.zip"),
        "--json",
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["algorithm"] == "CPPO"
    assert summary["sb3_algorithm"] == "PPO"
    assert Path(str(tmp_path / "cppo.zip")).is_file()
