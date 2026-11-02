"""Shared pytest fixtures.

All fixtures are offline and deterministic: the test suite must pass with no
network access, no Databento credentials and no IBKR/TWS connection.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from configs.instruments import InstrumentsConfig, load_instruments
from configs.settings import Settings
from src.core.types import MarketState, WeightVector

from tests.fixtures.market import (
    FIXED_TIMESTAMP,
    make_account_snapshot,
    make_contract_targets,
    make_market_state,
    make_risk_assessment,
    make_weight_vector,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolate_futuresrlx_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Hermetic tests: the developer's shell must never configure the suite.

    Exported ``FUTURESRLX_*`` variables (e.g. ``FUTURESRLX_ENVIRONMENT=paper``
    or a ``duckdb:`` risk-state URI used for live trading) would otherwise leak
    into ``Settings()`` constructions and change test behaviour depending on the
    machine. This autouse fixture strips every ``FUTURESRLX_*`` variable and
    disables the repo ``.env`` file before each test; a test that *wants* a
    variable can still set it explicitly via its own ``monkeypatch.setenv``.

    The settings singleton is cleared on entry and exit so cached settings from
    a previous test can never observe a different environment.
    """
    from configs.settings import get_settings

    for variable in list(os.environ):
        if variable.startswith("FUTURESRLX_"):
            monkeypatch.delenv(variable, raising=False)
    # Ignore the developer's committed/untracked ``.env`` file too, unless a
    # test deliberately opts back in (none currently do).
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(scope="session")
def instrument_master() -> InstrumentsConfig:
    """The committed instrument master."""
    return load_instruments(REPO_ROOT / "configs" / "instruments.yaml")


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    """Isolated settings pointing every writable path at ``tmp_path``."""
    return Settings(
        environment="backtest",
        data_dir=tmp_path / "data",
        parquet_dir=tmp_path / "data" / "parquet",
        duckdb_path=tmp_path / "data" / "test.duckdb",
        report_dir=tmp_path / "artifacts",
        instruments_path=REPO_ROOT / "configs" / "instruments.yaml",
    )


@pytest.fixture()
def market_state() -> MarketState:
    """A five-root synthetic market state with covariance and returns."""
    return make_market_state()


@pytest.fixture()
def market_state_factory():
    """Factory fixture: ``market_state_factory(roots=..., seed=...)``."""
    return make_market_state


@pytest.fixture()
def weights() -> WeightVector:
    """A small long/short weight vector for pipeline/translator tests."""
    return make_weight_vector({"ES": 0.30, "NQ": 0.20, "CL": -0.15, "GC": 0.10, "ZN": -0.05})


@pytest.fixture()
def weight_factory():
    """Factory fixture for :class:`WeightVector` construction."""
    return make_weight_vector


@pytest.fixture()
def contract_targets():
    """A contract target set aligned with the default instrument master."""
    return make_contract_targets({"ES": 2, "NQ": 1, "CL": -3, "GC": 4, "ZN": -2})


@pytest.fixture()
def risk_assessments():
    """Neutral LLM risk assessments for the default universe."""
    return [
        make_risk_assessment(root, semantic_risk_score=1.8, confidence=0.92)
        for root in ("ES", "NQ", "CL", "GC", "ZN")
    ]


@pytest.fixture()
def account_snapshot():
    """A standard account snapshot fixture."""
    return make_account_snapshot()


@pytest.fixture()
def fixed_timestamp() -> datetime:
    """Deterministic UTC timestamp used across fixtures."""
    return FIXED_TIMESTAMP


@pytest.fixture()
def equity_curve() -> np.ndarray:
    """A deterministic daily equity curve for analytics tests."""
    rng = np.random.default_rng(11)
    returns = rng.normal(0.0004, 0.009, size=756)
    return 1_000_000.0 * np.cumprod(1.0 + returns)
