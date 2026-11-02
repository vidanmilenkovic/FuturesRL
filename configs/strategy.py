"""Typed declarative strategy pipeline configuration (FinRL-X style YAML).

A strategy YAML is a pure description of the composable pipeline
``w_t = R_t(T_t(A_t(S_t(X))))``: which universe selector, allocator, timing
overlays and risk overlays to compose, plus the translator/backtest parameters.
No executable code lives in a strategy config, which makes parameter sweeps and
experiment tracking reproducible.
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from src.core.types import RoundingMode

type StageParam = float | int | str | bool


class StageConfig(BaseModel):
    """One pipeline stage: a registry name plus its parameter dictionary."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    params: dict[str, StageParam] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _known_name(cls, name: str) -> str:
        cleaned = name.strip()
        if not cleaned:
            raise ValueError("stage name must be non-empty")
        return cleaned


class TranslatorConfig(BaseModel):
    """Declarative parameters for the deterministic ContractTranslator."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rounding_mode: RoundingMode = RoundingMode.NEAREST
    rebalance_buffer_contracts: float = Field(default=0.0, ge=0.0)
    max_margin_utilization: float = Field(default=0.60, gt=0.0, le=1.0)
    enable_micro_routing: bool = True
    min_micro_notional_ratio: float = Field(default=0.0, ge=0.0, le=1.0)


class WalkForwardConfig(BaseModel):
    """Purged and embargoed walk-forward validation parameters."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    train_days: int = Field(default=252, ge=1)
    test_days: int = Field(default=63, ge=1)
    purge_days: int = Field(default=5, ge=0)
    embargo_days: int = Field(default=5, ge=0)
    step_days: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _consistent_windows(self) -> WalkForwardConfig:
        if self.purge_days >= self.train_days:
            raise ValueError("purge_days must be smaller than train_days")
        if self.embargo_days >= self.test_days:
            raise ValueError("embargo_days must be smaller than test_days")
        return self


class BacktestSpec(BaseModel):
    """Declarative backtest run specification."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    start: date
    end: date
    initial_equity: float = Field(default=1_000_000.0, gt=0.0)
    base_currency: str = "USD"
    rebalance_frequency: str = "W-FRI"
    commission_bps: float = Field(default=0.0, ge=0.0)
    slippage_ticks: float = Field(default=1.0, ge=0.0)
    random_seed: int = 7
    walk_forward: WalkForwardConfig | None = None

    @model_validator(mode="after")
    def _ordered_dates(self) -> BacktestSpec:
        if self.end <= self.start:
            raise ValueError("backtest end must be after start")
        return self


class DRLSpec(BaseModel):
    """Optional DRL allocator configuration (PPO / CPPO / SAC / TD3 / A2C)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    algorithm: str = "PPO"
    policy: str = "MlpPolicy"
    total_timesteps: int = Field(default=100_000, ge=1)
    learning_rate: float = Field(default=3e-4, gt=0.0)
    gamma: float = Field(default=0.99, gt=0.0, le=1.0)
    seed: int = 7
    reward_risk_penalty_lambda: float = Field(default=0.5, ge=0.0)
    normalize_observations: bool = True

    @model_validator(mode="after")
    def _known_algorithm(self) -> DRLSpec:
        allowed = {"PPO", "CPPO", "SAC", "TD3", "A2C"}
        if self.algorithm.upper() not in allowed:
            raise ValueError(f"algorithm must be one of {sorted(allowed)}")
        return self


class StrategyConfig(BaseModel):
    """A complete declarative strategy pipeline definition."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    description: str = ""
    tags: tuple[str, ...] = ()
    universe: StageConfig
    allocator: StageConfig
    timing: tuple[StageConfig, ...] = ()
    risk: tuple[StageConfig, ...] = ()
    translator: TranslatorConfig = TranslatorConfig()
    backtest: BacktestSpec | None = None
    drl: DRLSpec | None = None

    @field_validator("id")
    @classmethod
    def _slug_id(cls, value: str) -> str:
        cleaned = value.strip().lower().replace(" ", "_")
        if not cleaned:
            raise ValueError("strategy id must be non-empty")
        return cleaned


def parse_strategy_config(payload: dict[str, object]) -> StrategyConfig:
    """Validate a raw YAML payload into a :class:`StrategyConfig`.

    The ``universe`` stage may be written as ``selector: liquidity`` (the domain
    wording used in the architecture spec); it is normalised to the generic
    stage shape ``name: liquidity`` before validation.
    """
    if not isinstance(payload, dict):
        raise ValueError("strategy YAML must be a mapping")
    raw = dict(payload)
    universe = raw.get("universe")
    if isinstance(universe, dict) and "selector" in universe:
        normalised = dict(universe)
        normalised.setdefault("name", normalised.pop("selector"))
        raw["universe"] = normalised
    return StrategyConfig.model_validate(raw)


@lru_cache(maxsize=32)
def _load_strategy_cached(path: Path) -> StrategyConfig:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return parse_strategy_config(payload)


def load_strategy_config(path: Path | str) -> StrategyConfig:
    """Load and cache a strategy YAML file."""
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"strategy config not found: {resolved}")
    return _load_strategy_cached(resolved)


def discover_strategy_configs(directory: Path | str | None = None) -> dict[str, StrategyConfig]:
    """Load every ``*.yaml`` strategy config in ``directory`` keyed by strategy id."""
    base = (
        Path(directory) if directory is not None else Path(__file__).resolve().parent / "strategies"
    )
    if not base.is_dir():
        raise FileNotFoundError(f"strategy directory not found: {base}")
    configs: dict[str, StrategyConfig] = {}
    for path in sorted(base.glob("*.yaml")):
        config = load_strategy_config(path)
        if config.id in configs:
            raise ValueError(f"duplicate strategy id {config.id!r} in {base}")
        configs[config.id] = config
    return configs


def clear_strategy_cache() -> None:
    """Clear the strategy config cache (test helper)."""
    _load_strategy_cached.cache_clear()
