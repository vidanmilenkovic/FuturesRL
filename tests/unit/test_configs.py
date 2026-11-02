"""Unit tests for the typed configuration layer (instrument master + settings)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from configs.instruments import (
    InstrumentsConfig,
    clear_instruments_cache,
    load_instruments,
    parse_instruments_config,
)
from configs.settings import RiskSettings, Settings, clear_settings_cache, get_settings
from configs.strategy import (
    BacktestSpec,
    StrategyConfig,
    clear_strategy_cache,
    discover_strategy_configs,
    load_strategy_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
INSTRUMENTS_PATH = REPO_ROOT / "configs" / "instruments.yaml"
STRATEGIES_DIR = REPO_ROOT / "configs" / "strategies"


# ---------------------------------------------------------------------------
# Instrument master
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def master() -> InstrumentsConfig:
    clear_instruments_cache()
    return load_instruments(INSTRUMENTS_PATH)


def test_instrument_master_contains_required_roots(master: InstrumentsConfig) -> None:
    for root in ("ES", "MES", "NQ", "MNQ", "CL", "MCL", "GC", "MGC", "ZN", "ZT"):
        assert master.has(root)
    assert master.get("es").root == "ES"


def test_instrument_spec_derived_values(master: InstrumentsConfig) -> None:
    es = master.get("ES")
    assert es.multiplier == pytest.approx(50.0)
    assert es.tick_size == pytest.approx(0.25)
    assert es.point_value == pytest.approx(50.0)
    assert es.tick_value == pytest.approx(12.5)
    assert es.ticks_to_price(2) == pytest.approx(0.5)
    assert es.price_to_ticks(0.5) == pytest.approx(2.0)
    assert not es.is_micro


def test_micro_routing_pairs(master: InstrumentsConfig) -> None:
    assert master.micro_for("ES").root == "MES"
    assert master.micro_for("CL").root == "MCL"
    assert master.micro_for("GC").root == "MGC"
    assert master.micro_for("ZN") is None
    assert master.standard_for("MES").root == "ES"
    assert master.standard_for("ES") is None
    assert master.get("MES").is_micro
    assert master.get("MES").multiplier < master.get("ES").multiplier


def test_asset_class_and_sector_queries(master: InstrumentsConfig) -> None:
    equity_roots = {spec.root for spec in master.by_asset_class(master.get("ES").asset_class)}
    assert {"ES", "MES", "NQ", "MNQ"}.issubset(equity_roots)
    assert {spec.root for spec in master.by_sector("energy")} == {"CL", "MCL"}
    assert "equity" in master.sectors()
    assert len(master.roots()) >= 10


def test_unknown_root_raises(master: InstrumentsConfig) -> None:
    with pytest.raises(KeyError, match="unknown futures root"):
        master.get("NOPE")


def test_missing_instrument_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_instruments(tmp_path / "missing.yaml")


def test_parse_rejects_broken_cross_reference() -> None:
    with pytest.raises(ValueError, match="unknown micro_root"):
        parse_instruments_config(
            {
                "instruments": {
                    "ES": {
                        "name": "E-mini",
                        "asset_class": "equity_index",
                        "sector": "equity",
                        "exchange": "CME",
                        "multiplier": 50.0,
                        "tick_size": 0.25,
                        "databento_symbol": "ES.FUT",
                        "ibkr_exchange": "CME",
                        "micro_root": "MES",
                        "margins": {"initial": 15000.0, "maintenance": 13600.0},
                        "roll": {"contract_months": [3, 6, 9, 12]},
                    }
                }
            }
        )


def test_parse_injects_root_from_key() -> None:
    config = parse_instruments_config(
        {
            "instruments": {
                "CL": {
                    "name": "WTI",
                    "asset_class": "energy",
                    "sector": "energy",
                    "exchange": "NYMEX",
                    "multiplier": 1000.0,
                    "tick_size": 0.01,
                    "databento_symbol": "CL.FUT",
                    "ibkr_exchange": "NYMEX",
                    "margins": {"initial": 6500.0, "maintenance": 5900.0},
                    "roll": {"contract_months": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]},
                }
            }
        }
    )
    assert config.get("CL").root == "CL"
    assert config.get("CL").roll.physical_delivery is False


def test_roll_spec_validation() -> None:
    from configs.instruments import RollSpec

    with pytest.raises(ValueError, match="contract_months"):
        RollSpec(contract_months=())
    with pytest.raises(ValueError, match=r"1\.\.12"):
        RollSpec(contract_months=(0, 3))
    spec = RollSpec(contract_months=(12, 3, 6, 3))
    assert spec.contract_months == (3, 6, 12)


def test_margin_spec_validation() -> None:
    from configs.instruments import MarginSpec

    with pytest.raises(ValueError, match="maintenance margin"):
        MarginSpec(initial=100.0, maintenance=200.0)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def test_settings_defaults(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        parquet_dir=tmp_path / "data" / "parquet",
        duckdb_path=tmp_path / "data" / "x.duckdb",
        report_dir=tmp_path / "artifacts",
    )
    assert settings.environment == "development"
    assert settings.is_live is False
    assert settings.risk.max_margin_utilization == pytest.approx(0.60)
    assert settings.risk.llm_confidence_threshold == pytest.approx(0.70)
    assert settings.risk.llm_critical_score == pytest.approx(4.5)
    assert settings.api_base_url == "http://0.0.0.0:8000" or settings.api_base_url.startswith(
        "http"
    )


def test_settings_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_settings_cache()
    monkeypatch.setenv("FUTURESRLX_ENVIRONMENT", "backtest")
    monkeypatch.setenv("FUTURESRLX_API_PORT", "9123")
    monkeypatch.setenv("FUTURESRLX_RISK__MAX_MARGIN_UTILIZATION", "0.45")
    monkeypatch.setenv("FUTURESRLX_DATABENTO_API_KEY", "secret-key")
    settings = Settings(_env_file=None)
    assert settings.environment == "backtest"
    assert settings.api_port == 9123
    assert settings.risk.max_margin_utilization == pytest.approx(0.45)
    assert "secret-key" not in str(settings.model_dump_redacted())
    assert settings.databento_api_key_value == "secret-key"
    clear_settings_cache()


def test_settings_paper_port_validation() -> None:
    with pytest.raises(ValueError, match="paper port"):
        Settings(_env_file=None, environment="paper", ibkr_port=7496)
    with pytest.raises(ValueError, match="ibkr_paper"):
        Settings(_env_file=None, environment="paper", ibkr_port=7497, ibkr_paper=False)


def test_settings_live_port_validation() -> None:
    with pytest.raises(ValueError, match="live port"):
        Settings(_env_file=None, environment="live", ibkr_port=7497)
    live = Settings(_env_file=None, environment="live", ibkr_port=4001, ibkr_paper=False)
    assert live.is_live


def test_settings_ensure_directories(settings: Settings) -> None:
    settings.ensure_directories()
    assert settings.parquet_dir.is_dir()
    assert settings.report_dir.is_dir()
    assert settings.dataset_dir() == settings.parquet_dir / settings.databento_dataset


def test_settings_risk_validators() -> None:
    with pytest.raises(ValueError, match="margin_critical_threshold"):
        RiskSettings(margin_warning_threshold=0.7, margin_critical_threshold=0.6)
    with pytest.raises(ValueError, match="llm_critical_score"):
        RiskSettings(llm_safe_score=4.0, llm_critical_score=3.0)


# ---------------------------------------------------------------------------
# Strategy configs
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def strategies() -> dict[str, StrategyConfig]:
    clear_strategy_cache()
    return discover_strategy_configs(STRATEGIES_DIR)


def test_discover_strategy_configs(strategies: dict[str, StrategyConfig]) -> None:
    assert set(strategies) == {"tsmom_vol_target", "drl_ppo_llm_risk"}
    tsmom = strategies["tsmom_vol_target"]
    assert tsmom.universe.name == "liquidity"
    assert tsmom.allocator.name == "volatility_target"
    assert [stage.name for stage in tsmom.timing] == ["tsmom", "kama", "carry_filter"]
    assert [stage.name for stage in tsmom.risk] == ["quant_risk", "llm_risk"]
    assert tsmom.translator.max_margin_utilization == pytest.approx(0.60)
    assert tsmom.backtest is not None
    assert tsmom.backtest.start == date(2018, 1, 1)
    assert tsmom.backtest.walk_forward is not None
    assert tsmom.backtest.walk_forward.purge_days == 5


def test_drl_strategy_declares_drl_section(strategies: dict[str, StrategyConfig]) -> None:
    drl = strategies["drl_ppo_llm_risk"]
    assert drl.drl is not None
    assert drl.drl.algorithm == "PPO"
    assert drl.allocator.name == "drl"


def test_strategy_config_validation(strategies: dict[str, StrategyConfig]) -> None:
    base = strategies["tsmom_vol_target"].model_dump()
    with pytest.raises(ValueError, match="end must be after start"):
        BacktestSpec(start=date(2024, 1, 1), end=date(2023, 1, 1))
    with pytest.raises(ValueError, match="purge_days"):
        BacktestSpec(
            start=date(2020, 1, 1),
            end=date(2021, 1, 1),
            walk_forward={"train_days": 10, "test_days": 5, "purge_days": 10, "embargo_days": 1},
        )
    with pytest.raises(ValueError, match="algorithm"):
        from configs.strategy import DRLSpec

        DRLSpec(algorithm="NOT_AN_ALGO")
    assert base["id"] == "tsmom_vol_target"


def test_load_strategy_config_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_strategy_config(tmp_path / "nope.yaml")


def test_get_settings_cached() -> None:
    clear_settings_cache()
    assert get_settings() is get_settings()
    clear_settings_cache()
