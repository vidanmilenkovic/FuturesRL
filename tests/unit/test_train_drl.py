"""Unit tests for the ``python -m src.strategies.train_drl`` CLI (Phase 9)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from src.strategies import train_drl as trainer
from src.strategies._observation import DEFAULT_OBSERVATION_FEATURES
from src.strategies.envs import FuturesTradingEnv

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
    # Phase-10 task-24: the offline source now emits a synthetic F1/F2 curve, so
    # every canonical feature (incl. term_slope/carry_annualized) is available.
    assert metadata["feature_fallbacks"] == []
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


# ---------------------------------------------------------------------------
# Phase-10: reusable callable, hyperparameter plumbing, reward curve, CLI flags
# ---------------------------------------------------------------------------

MINI_INSTRUMENTS: tuple[str, ...] = ("ES", "NQ")
MINI_FEATURES: tuple[str, ...] = ("ret_1d",)
_CAPTURED_ALLOCATOR_KWARGS: dict[str, object] = {}


def _mini_training_data(periods: int = 80) -> trainer.TrainingData:
    """Tiny deterministic TrainingData used to test plumbing without real bars."""
    rows = periods
    prices = np.empty((rows, len(MINI_INSTRUMENTS)), dtype=np.float64)
    prices[:, 0] = 4800.0 * np.cumprod(np.full(rows, 1.0002))
    prices[:, 1] = 16800.0 * np.cumprod(np.full(rows, 1.0001))
    features = np.zeros((rows, len(MINI_INSTRUMENTS), len(MINI_FEATURES)), dtype=np.float64)
    start = datetime(2024, 1, 2, tzinfo=UTC)
    timestamps = tuple(start + timedelta(days=index) for index in range(rows))
    return trainer.TrainingData(
        prices=prices,
        features=features,
        timestamps=timestamps,
        instrument_ids=MINI_INSTRUMENTS,
        feature_names=MINI_FEATURES,
        feature_fallbacks=(),
        multipliers=(50.0, 20.0),
        initial_margins=(15_000.0, 20_000.0),
        maintenance_margins=(13_600.0, 18_100.0),
        tick_sizes=(0.25, 0.25),
        periods=periods,
    )


def _mini_env(data: trainer.TrainingData) -> FuturesTradingEnv:
    return FuturesTradingEnv(
        prices=data.prices,
        instrument_ids=data.instrument_ids,
        features=data.features,
        feature_names=data.feature_names,
        multipliers=data.multipliers,
        initial_margins=data.initial_margins,
        maintenance_margins=data.maintenance_margins,
        tick_sizes=data.tick_sizes,
        initial_equity=1_000_000.0,
        commission_bps=1.0,
        slippage_ticks=1.0,
        max_gross_leverage=1.0,
        max_margin_utilization=0.6,
        risk_penalty_lambda=0.5,
    )


class _FakeModel:
    """SB3 stand-in that persists a dummy checkpoint file."""

    def save(self, path: str) -> None:
        Path(path).write_bytes(b"fake-sb3-checkpoint")


class _FakeAllocator:
    """DRLAllocator stand-in capturing kwargs and stepping the env deterministically."""

    def __init__(self, **kwargs: object) -> None:
        _CAPTURED_ALLOCATOR_KWARGS.clear()
        _CAPTURED_ALLOCATOR_KWARGS.update(kwargs)

    def train(self, env: object, total_timesteps: int | None = None) -> _FakeModel:
        assert total_timesteps is not None
        env.reset(seed=7)  # type: ignore[attr-defined]
        zeros = np.zeros(len(MINI_INSTRUMENTS), dtype=np.float32)
        for _ in range(int(total_timesteps)):
            _, _, terminated, truncated, _ = env.step(zeros)  # type: ignore[attr-defined]
            if terminated or truncated:
                env.reset()  # type: ignore[attr-defined]
        return _FakeModel()


@pytest.fixture()
def patched_trainer(monkeypatch: pytest.MonkeyPatch) -> trainer.TrainingData:
    """Patch the heavy SB3/data stages, keeping the real train_drl orchestration."""
    data = _mini_training_data()
    _CAPTURED_ALLOCATOR_KWARGS.clear()
    monkeypatch.setattr(trainer, "_sb3_version", lambda: "test-sb3-1.0")
    monkeypatch.setattr(trainer, "_git_commit", lambda: "a" * 40)
    monkeypatch.setattr(
        trainer,
        "build_training_data",
        lambda config, instruments, *, seed, periods: data,
    )
    monkeypatch.setattr(trainer, "build_env", lambda config, data, *, drl: _mini_env(data))
    monkeypatch.setattr(trainer, "DRLAllocator", _FakeAllocator)
    return data


def test_train_drl_callable_plumbs_algorithm_and_hyperparameters(
    patched_trainer: trainer.TrainingData, tmp_path: Path
) -> None:
    summary = trainer.train_drl(
        strategy=STRATEGY_PATH,
        algorithm="cppo",
        timesteps=8,
        learning_rate=0.0007,
        gamma=0.9,
        llm_reward_penalty_lambda=0.25,
        checkpoint_name="unit_plumb",
        output_dir=tmp_path,
        periods=40,
        seed=11,
        device="cpu",
        reward_curve_windows=4,
    )

    captured = _CAPTURED_ALLOCATOR_KWARGS
    assert captured["algorithm"] == "CPPO"
    assert captured["policy"] == "MlpPolicy"
    assert captured["total_timesteps"] == 8
    assert captured["learning_rate"] == pytest.approx(0.0007)
    assert captured["gamma"] == pytest.approx(0.9)
    assert captured["risk_penalty_lambda"] == pytest.approx(0.25)
    assert captured["seed"] == 11
    assert captured["device"] == "cpu"
    assert captured["feature_names"] == MINI_FEATURES

    assert summary["status"] == "ok"
    assert summary["algorithm"] == "CPPO"
    assert summary["sb3_algorithm"] == "PPO"
    assert summary["total_timesteps"] == 8
    assert summary["learning_rate"] == pytest.approx(0.0007)
    assert summary["gamma"] == pytest.approx(0.9)
    assert summary["llm_reward_penalty_lambda"] == pytest.approx(0.25)
    assert summary["checkpoint_name"] == "unit_plumb"
    assert summary["hyperparameters"] == {
        "algorithm": "CPPO",
        "sb3_algorithm": "PPO",
        "policy": "MlpPolicy",
        "total_timesteps": 8,
        "learning_rate": 0.0007,
        "gamma": 0.9,
        "llm_reward_penalty_lambda": 0.25,
        "seed": 11,
        "device": "cpu",
        "reward_curve_windows": 4,
    }

    checkpoint = Path(str(summary["model_path"]))
    meta_path = Path(str(summary["meta_path"]))
    curve_path = Path(str(summary["reward_curve_path"]))
    assert checkpoint == tmp_path / "unit_plumb.zip"
    assert checkpoint.is_file()
    assert meta_path == tmp_path / "unit_plumb.zip.meta.json"
    assert meta_path.is_file()
    assert curve_path == tmp_path / "unit_plumb.zip.reward_curve.json"
    assert curve_path.is_file()

    curve = summary["reward_curve"]
    assert isinstance(curve, dict)
    assert curve["env_steps"] == 8
    assert curve["samples"] <= 4
    assert len(curve["points"]) == curve["samples"]
    assert layout_ok(curve)
    assert json.loads(curve_path.read_text(encoding="utf-8")) == curve
    assert json.loads(meta_path.read_text(encoding="utf-8")) == summary


def layout_ok(curve: dict[str, object]) -> bool:
    """Sanity check for the bounded reward-curve payload layout."""
    points = curve["points"]
    means = curve["window_mean_rewards"]
    assert isinstance(points, list) and isinstance(means, list)
    assert len(points) == len(means) == int(curve["samples"])  # type: ignore[arg-type]
    for point, mean in zip(points, means, strict=True):
        assert isinstance(point, dict)
        assert point["mean_reward"] == pytest.approx(float(mean))
    assert curve["final_mean_reward"] == pytest.approx(float(means[-1]))
    assert curve["best_mean_reward"] == pytest.approx(max(float(value) for value in means))
    return True


def test_train_drl_cppo_maps_to_sb3_ppo(
    patched_trainer: trainer.TrainingData, tmp_path: Path
) -> None:
    summary = trainer.train_drl(
        strategy=STRATEGY_PATH,
        algorithm="cppo",
        timesteps=4,
        checkpoint_name="unit_cppo",
        output_dir=tmp_path,
    )
    assert _CAPTURED_ALLOCATOR_KWARGS["algorithm"] == "CPPO"
    assert summary["algorithm"] == "CPPO"
    assert summary["sb3_algorithm"] == "PPO"
    assert summary["hyperparameters"]["sb3_algorithm"] == "PPO"


def test_train_drl_checkpoint_naming_precedence(
    patched_trainer: trainer.TrainingData, tmp_path: Path
) -> None:
    named = trainer.train_drl(
        strategy=STRATEGY_PATH,
        timesteps=4,
        checkpoint_name="named_model",
        output_dir=tmp_path / "named",
    )
    assert Path(str(named["model_path"])) == tmp_path / "named" / "named_model.zip"

    explicit = trainer.train_drl(
        strategy=STRATEGY_PATH,
        timesteps=4,
        checkpoint_name="ignored_name",
        output=tmp_path / "explicit_model",
        output_dir=tmp_path / "unused",
    )
    assert Path(str(explicit["model_path"])) == tmp_path / "explicit_model.zip"
    assert explicit["checkpoint_name"] == "explicit_model"
    assert not (tmp_path / "unused" / "ignored_name.zip").exists()

    defaulted = trainer.train_drl(
        strategy=STRATEGY_PATH,
        algorithm="cppo",
        timesteps=4,
        output_dir=tmp_path / "defaulted",
    )
    assert (
        Path(str(defaulted["model_path"])) == tmp_path / "defaulted" / "drl_ppo_llm_risk_cppo.zip"
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"algorithm": "DQN"}, "algorithm"),
        ({"timesteps": 0}, "timesteps"),
        ({"timesteps": trainer.MAX_TIMESTEPS + 1}, "timesteps"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"learning_rate": 1.5}, "learning_rate"),
        ({"gamma": -0.1}, "gamma"),
        ({"gamma": 1.5}, "gamma"),
        ({"llm_reward_penalty_lambda": -0.1}, "llm_reward_penalty_lambda"),
        ({"checkpoint_name": "../evil"}, "checkpoint_name"),
        ({"checkpoint_name": ""}, "checkpoint_name"),
        ({"device": "   "}, "device"),
        ({"periods": 1}, "periods"),
        ({"seed": -1}, "seed"),
        ({"reward_curve_windows": 0}, "reward_curve_windows"),
        ({"offline": False}, "offline"),
    ],
)
def test_train_drl_rejects_invalid_parameters(
    kwargs: dict[str, object], match: str, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match=match):
        trainer.train_drl(
            strategy=STRATEGY_PATH,
            output=tmp_path / "never.zip",
            **kwargs,  # type: ignore[arg-type]
        )
    assert not (tmp_path / "never.zip").exists()


def test_train_drl_reward_curve_is_downsampled_to_requested_windows(
    patched_trainer: trainer.TrainingData, tmp_path: Path
) -> None:
    summary = trainer.train_drl(
        strategy=STRATEGY_PATH,
        timesteps=8,
        reward_curve_windows=3,
        checkpoint_name="windows",
        output_dir=tmp_path,
    )
    curve = summary["reward_curve"]
    assert isinstance(curve, dict)
    assert curve["window_steps"] == 2  # max(1, 8 // 3)
    assert curve["samples"] <= 3
    assert curve["env_steps"] == 8
    assert curve["episode_count"] == 0  # episode is longer than the toy budget


def test_train_drl_real_tiny_run_writes_reward_curve(tmp_path: Path) -> None:
    """Real SB3 run (small budget) exercises the shared callable end to end."""
    summary = trainer.train_drl(
        strategy=STRATEGY_PATH,
        timesteps=8,
        periods=160,
        checkpoint_name="real_callable",
        output_dir=tmp_path,
        reward_curve_windows=4,
    )
    checkpoint = Path(str(summary["model_path"]))
    meta_path = Path(str(summary["meta_path"]))
    curve_path = Path(str(summary["reward_curve_path"]))
    assert checkpoint.is_file() and checkpoint.stat().st_size > 0
    assert meta_path.is_file() and curve_path.is_file()
    curve = summary["reward_curve"]
    assert isinstance(curve, dict)
    assert int(curve["samples"]) >= 1  # type: ignore[arg-type]
    assert int(curve["env_steps"]) >= 1  # type: ignore[arg-type]
    assert summary["feature_fallbacks"] == []
    assert json.loads(meta_path.read_text(encoding="utf-8")) == summary
    assert json.loads(curve_path.read_text(encoding="utf-8")) == curve


def test_cli_plumbs_new_training_options(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    captured: dict[str, object] = {}

    def fake_train(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(trainer, "train_drl", fake_train)
    exit_code = trainer.main(
        [
            "--checkpoint-name",
            "cli_named",
            "--output-dir",
            "artifacts/custom_models",
            "--learning-rate",
            "0.001",
            "--gamma",
            "0.9",
            "--llm-reward-penalty-lambda",
            "0.2",
            "--reward-curve-windows",
            "5",
            "--timesteps",
            "4",
            "--seed",
            "3",
            "--json",
        ]
    )
    assert exit_code == 0
    assert captured["checkpoint_name"] == "cli_named"
    assert captured["output_dir"] == Path("artifacts/custom_models")
    assert captured["learning_rate"] == pytest.approx(0.001)
    assert captured["gamma"] == pytest.approx(0.9)
    assert captured["llm_reward_penalty_lambda"] == pytest.approx(0.2)
    assert captured["reward_curve_windows"] == 5
    assert captured["timesteps"] == 4
    assert captured["seed"] == 3
    assert captured["offline"] is True
    assert json.loads(capsys.readouterr().out) == {"status": "ok"}


@pytest.mark.parametrize(
    "arguments",
    [
        ("--checkpoint-name", "../evil"),
        ("--checkpoint-name", "bad name"),
        ("--learning-rate", "0"),
        ("--gamma", "1.5"),
        ("--llm-reward-penalty-lambda", "-1"),
        ("--reward-curve-windows", "0"),
    ],
)
def test_cli_parser_rejects_invalid_new_options(
    arguments: tuple[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    parser = trainer.build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args([*arguments, "--output", "model.zip"])
    assert excinfo.value.code == 2
    assert "Traceback" not in capsys.readouterr().err


def test_cli_invalid_checkpoint_name_exits_two_without_traceback(tmp_path: Path) -> None:
    completed = _run_cli("--checkpoint-name", "../evil", "--output", str(tmp_path / "model.zip"))
    assert completed.returncode == 2
    assert "Traceback" not in completed.stderr
    assert "checkpoint" in completed.stderr.lower()


def test_cli_help_lists_new_training_options() -> None:
    completed = _run_cli("--help")
    assert completed.returncode == 0
    for option in (
        "--checkpoint-name",
        "--output-dir",
        "--learning-rate",
        "--gamma",
        "--llm-reward-penalty-lambda",
        "--reward-curve-windows",
    ):
        assert option in completed.stdout


# ---------------------------------------------------------------------------
# Phase-StudioUX: bounded offline training-data cache (45-root runtime control)
# ---------------------------------------------------------------------------


def _fake_uncached_builder(
    calls: list[tuple[int, int, tuple[str, ...]]],
) -> Any:
    """Cheap stand-in for the expensive 45-root offline build (records calls)."""

    def fake(
        config: object,
        instruments: object,
        *,
        seed: int,
        periods: int = trainer.DEFAULT_PERIODS,
    ) -> trainer.TrainingData:
        assert config is not None
        roots = tuple(instruments.roots())
        calls.append((seed, periods, roots))
        rows = 4
        prices = np.full((rows, len(roots)), 1_000.0 + seed + periods, dtype=np.float64)
        features = np.full(
            (rows, len(roots), len(DEFAULT_OBSERVATION_FEATURES)),
            float(periods),
            dtype=np.float64,
        )
        start = datetime(2024, 1, 2, tzinfo=UTC)
        timestamps = tuple(start + timedelta(days=index) for index in range(rows))
        return trainer.TrainingData(
            prices=prices,
            features=features,
            timestamps=timestamps,
            instrument_ids=roots,
            feature_names=DEFAULT_OBSERVATION_FEATURES,
            feature_fallbacks=(),
            multipliers=tuple(1.0 for _ in roots),
            initial_margins=tuple(0.0 for _ in roots),
            maintenance_margins=tuple(0.0 for _ in roots),
            tick_sizes=tuple(0.01 for _ in roots),
            periods=periods,
        )

    return fake


def _custom_cache_master(root: str = "ZC9") -> object:
    """A one-root instrument master that must not share cache entries with the real one."""
    from configs.instruments import load_instruments, parse_instruments_config

    master = load_instruments(REPO_ROOT / "configs" / "instruments.yaml")
    entry = master.get("ES").model_dump(mode="json")
    entry.pop("micro_root", None)
    entry.pop("standard_root", None)
    entry.update(
        {
            "root": root,
            "name": "Corn (cache isolation test)",
            "asset_class": "agriculture",
            "sector": "agri",
            "exchange": "CBOT",
            "databento_symbol": f"{root}.FUT",
            "ibkr_exchange": "CBOT",
        }
    )
    return parse_instruments_config(
        {
            "version": master.version,
            "defaults": master.defaults.model_dump(mode="json"),
            "instruments": {root: entry},
        }
    )


def test_training_data_cache_hits_and_returns_defensive_copies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from configs.instruments import load_instruments
    from configs.strategy import load_strategy_config

    trainer.clear_training_data_cache()
    calls: list[tuple[int, int, tuple[str, ...]]] = []
    monkeypatch.setattr(trainer, "_build_training_data_uncached", _fake_uncached_builder(calls))
    config = load_strategy_config(STRATEGY_PATH)
    master = load_instruments(REPO_ROOT / "configs" / "instruments.yaml")

    first = trainer.build_training_data(config, master, seed=7, periods=64)
    second = trainer.build_training_data(config, master, seed=7, periods=64)
    assert len(calls) == 1  # second call was a cache hit (no rebuild)
    info = trainer.training_data_cache_info()
    assert info == {
        "size": 1,
        "maxsize": trainer.TRAINING_DATA_CACHE_MAXSIZE,
        "hits": 1,
        "misses": 1,
        "evictions": 0,
    }
    # Same outputs, but caller-owned writable arrays (defensive copies).
    assert np.array_equal(first.prices, second.prices)
    assert np.array_equal(first.features, second.features)
    assert first.instrument_ids == second.instrument_ids
    assert first.feature_names == second.feature_names
    assert first.timestamps == second.timestamps
    assert first.prices is not second.prices
    assert first.features is not second.features
    assert first.prices.flags.writeable and second.prices.flags.writeable
    # Mutating a returned copy can never corrupt the cached snapshot.
    snapshot = second.features.copy()
    first.features[0, 0, 0] = 123_456.0
    first.prices[0, 0] = -1.0
    third = trainer.build_training_data(config, master, seed=7, periods=64)
    assert np.array_equal(third.features, snapshot)
    assert third.prices[0, 0] != -1.0
    # The cached arrays themselves are read-only.
    cached = next(iter(trainer._TRAINING_DATA_CACHE.values()))
    assert cached.prices.flags.writeable is False
    assert cached.features.flags.writeable is False
    trainer.clear_training_data_cache()


def test_training_data_cache_separates_seed_periods_master_and_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from configs.instruments import load_instruments
    from configs.strategy import load_strategy_config

    trainer.clear_training_data_cache()
    calls: list[tuple[int, int, tuple[str, ...]]] = []
    monkeypatch.setattr(trainer, "_build_training_data_uncached", _fake_uncached_builder(calls))
    config = load_strategy_config(STRATEGY_PATH)
    master = load_instruments(REPO_ROOT / "configs" / "instruments.yaml")
    custom_master = _custom_cache_master()

    base = trainer.build_training_data(config, master, seed=7, periods=64)
    other_seed = trainer.build_training_data(config, master, seed=8, periods=64)
    other_periods = trainer.build_training_data(config, master, seed=7, periods=128)
    other_master = trainer.build_training_data(config, custom_master, seed=7, periods=64)
    assert len(calls) == 4
    assert trainer.training_data_cache_info()["misses"] == 4
    assert base.prices[0, 0] != other_seed.prices[0, 0]
    assert base.features[0, 0, 0] != other_periods.features[0, 0, 0]
    assert custom_master.roots() == ("ZC9",)
    assert other_master.instrument_ids == ("ZC9",)
    # Four distinct keys fit in the bounded cache (maxsize=4): every repeat is a hit.
    for builder_master, seed, periods in (
        (master, 7, 64),
        (master, 8, 64),
        (master, 7, 128),
        (custom_master, 7, 64),
    ):
        trainer.build_training_data(config, builder_master, seed=seed, periods=periods)
    info = trainer.training_data_cache_info()
    assert len(calls) == 4
    assert info["hits"] == 4
    assert info["misses"] == 4
    assert info["size"] == trainer.TRAINING_DATA_CACHE_MAXSIZE
    trainer.clear_training_data_cache()


def test_training_data_cache_separates_universe_and_equity_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from configs.instruments import load_instruments
    from configs.strategy import StageConfig, load_strategy_config

    trainer.clear_training_data_cache()
    calls: list[tuple[int, int, tuple[str, ...]]] = []
    monkeypatch.setattr(trainer, "_build_training_data_uncached", _fake_uncached_builder(calls))
    config = load_strategy_config(STRATEGY_PATH)
    master = load_instruments(REPO_ROOT / "configs" / "instruments.yaml")
    assert config.backtest is not None
    variants = (
        config,
        config.model_copy(update={"universe": StageConfig(name="liquidity", params={"top_n": 3})}),
        config.model_copy(
            update={"backtest": config.backtest.model_copy(update={"initial_equity": 2_000_000.0})}
        ),
    )
    # Each config variant affects the build inputs and must be keyed separately.
    for variant in variants:
        trainer.build_training_data(variant, master, seed=7, periods=64)
    assert len(calls) == 3
    for variant in variants:
        trainer.build_training_data(variant, master, seed=7, periods=64)
    info = trainer.training_data_cache_info()
    assert len(calls) == 3
    assert info["hits"] == 3 and info["misses"] == 3 and info["size"] == 3
    trainer.clear_training_data_cache()


def test_training_data_cache_is_bounded_and_evictable(monkeypatch: pytest.MonkeyPatch) -> None:
    from configs.instruments import load_instruments
    from configs.strategy import load_strategy_config

    trainer.clear_training_data_cache()
    calls: list[tuple[int, int, tuple[str, ...]]] = []
    monkeypatch.setattr(trainer, "_build_training_data_uncached", _fake_uncached_builder(calls))
    config = load_strategy_config(STRATEGY_PATH)
    master = load_instruments(REPO_ROOT / "configs" / "instruments.yaml")

    for seed in range(6):
        trainer.build_training_data(config, master, seed=seed, periods=64)
    info = trainer.training_data_cache_info()
    assert len(calls) == 6
    assert info["size"] == trainer.TRAINING_DATA_CACHE_MAXSIZE == 4
    assert info["evictions"] == 2
    # The two oldest entries (seeds 0,1) were evicted; seed 5 is still cached.
    trainer.build_training_data(config, master, seed=5, periods=64)
    assert len(calls) == 6
    trainer.build_training_data(config, master, seed=0, periods=64)
    assert len(calls) == 7  # evicted -> rebuilt
    assert trainer.training_data_cache_info()["evictions"] == 3
    # Clearing the cache resets entries and counters (API reload hook).
    trainer.clear_training_data_cache()
    cleared = trainer.training_data_cache_info()
    assert cleared == {
        "size": 0,
        "maxsize": trainer.TRAINING_DATA_CACHE_MAXSIZE,
        "hits": 0,
        "misses": 0,
        "evictions": 0,
    }


def test_training_data_cache_real_build_is_identical_on_hit() -> None:
    """Real 45-root build: cached hit returns byte-identical matrices/metadata."""
    from configs.instruments import load_instruments
    from configs.strategy import load_strategy_config

    trainer.clear_training_data_cache()
    config = load_strategy_config(STRATEGY_PATH)
    master = load_instruments(REPO_ROOT / "configs" / "instruments.yaml")
    first = trainer.build_training_data(config, master, seed=7, periods=160)
    snapshot_prices = first.prices.copy()
    snapshot_features = first.features.copy()
    first_info = trainer.training_data_cache_info()
    assert first_info["misses"] == 1 and first_info["hits"] == 0

    second = trainer.build_training_data(config, master, seed=7, periods=160)
    second_info = trainer.training_data_cache_info()
    assert second_info["hits"] == 1 and second_info["misses"] == 1
    assert np.array_equal(second.prices, snapshot_prices)
    assert np.array_equal(second.features, snapshot_features)
    assert second.instrument_ids == first.instrument_ids
    assert second.feature_names == DEFAULT_OBSERVATION_FEATURES
    assert second.feature_fallbacks == ()  # canonical features complete (A2)
    assert second.timestamps == first.timestamps
    assert second.periods == first.periods
    assert np.all(second.prices > 0.0) and np.all(np.isfinite(second.features))
    # Cache hits are still safe to mutate (own copy) and never leak into later hits.
    first.features[0, 0, 0] += 1.0
    third = trainer.build_training_data(config, master, seed=7, periods=160)
    assert np.array_equal(third.features, snapshot_features)
    trainer.clear_training_data_cache()
