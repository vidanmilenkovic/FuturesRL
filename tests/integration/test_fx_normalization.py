"""Phase-7 integration test 5: base-currency FX normalisation end-to-end.

Proves that a non-USD IBKR base account (EUR) flows correctly through the real
FinRL-X pipeline and futures contract translator: sizing uses USD-normalised
equity, margin utilisation is computed against USD equity, and the USD path is
unchanged. Also checks the fail-closed behaviour when no FX rate is configured.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from configs.instruments import InstrumentsConfig, load_instruments
from configs.strategy import StageConfig, StrategyConfig, load_strategy_config
from src.backtest.engine import EventBacktester
from src.risk.contract_translator import ContractTranslator
from src.strategies import build_pipeline

from tests.fixtures.market import make_market_state

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOTS = ("ES", "NQ", "CL")
TIMESTAMP = datetime(2024, 1, 2, 21, 0, tzinfo=UTC)
EUR_RATE = 1.2
EUR_EQUITY = 100_000.0
USD_EQUITY = EUR_EQUITY * EUR_RATE  # 120k USD equivalent


@pytest.fixture(scope="module")
def instrument_master() -> InstrumentsConfig:
    return load_instruments(REPO_ROOT / "configs" / "instruments.yaml")


def _strategy_config():
    config = load_strategy_config(REPO_ROOT / "configs" / "strategies" / "tsmom_vol_target.yaml")
    return config.model_copy(
        update={
            "universe": StageConfig(
                name="liquidity",
                params={
                    "min_avg_dollar_volume": 0.0,
                    "min_open_interest": 0.0,
                    "top_n": 3,
                    "min_history_days": 0,
                },
            )
        }
    )


def _states(*, equity: float, account_currency: str, fx_rates: dict[str, float], count: int = 2):
    return [
        make_market_state(
            ROOTS,
            timestamp=TIMESTAMP + timedelta(days=offset),
            equity=equity,
            account_currency=account_currency,
            fx_rates=fx_rates,
        )
        for offset in range(count)
    ]


def _pipeline(
    translator: ContractTranslator,
    master: InstrumentsConfig,
    config: StrategyConfig | None = None,
):
    return build_pipeline(
        config or _strategy_config(), master, translator=translator, risk_overlays=[]
    )


def _backtest_config():
    """TSMOM-only variant so the synthetic series produces a first-bar trade."""
    config = _strategy_config()
    return config.model_copy(
        update={"timing": tuple(stage for stage in config.timing if stage.name == "tsmom")}
    )


def test_eur_pipeline_matches_usd_equivalent(instrument_master: InstrumentsConfig) -> None:
    eur_translator = ContractTranslator(instruments=instrument_master)
    usd_translator = ContractTranslator(instruments=instrument_master)
    eur_result = _pipeline(eur_translator, instrument_master).run(
        _states(equity=EUR_EQUITY, account_currency="EUR", fx_rates={"EUR": EUR_RATE})[0]
    )
    usd_result = _pipeline(usd_translator, instrument_master).run(
        _states(equity=USD_EQUITY, account_currency="USD", fx_rates={})[0]
    )
    # The approved weights are currency-independent ...
    assert eur_result.final_weights == usd_result.final_weights
    # ... and the discrete targets must be identical after USD normalisation.
    assert eur_result.targets is not None and usd_result.targets is not None
    assert eur_result.targets.quantities == usd_result.targets.quantities
    # Margin utilisation is IM / equity_usd, identical in both accounting bases.
    eur_util = eur_translator.margin_utilization(eur_result.targets, EUR_EQUITY)
    usd_util = usd_translator.margin_utilization(usd_result.targets, USD_EQUITY)
    assert eur_util == pytest.approx(usd_util, rel=1e-12)
    assert eur_util <= 0.60 + 1e-9


def test_eur_equity_actually_changes_sizing(instrument_master: InstrumentsConfig) -> None:
    """A EUR account is NOT sized as if its NLV were USD (the bug this fixes)."""
    weights_state = _states(equity=EUR_EQUITY, account_currency="EUR", fx_rates={"EUR": EUR_RATE})[
        0
    ]
    translator = ContractTranslator(instruments=instrument_master)
    from src.core.types import WeightVector

    weights = WeightVector({"ES": 1.0})
    eur_targets = translator.translate(weights, weights_state)
    # Same numeric NLV but a USD account: target notional is 20% smaller.
    usd_state = _states(equity=EUR_EQUITY, account_currency="USD", fx_rates={})[0]
    usd_targets = translator.translate(weights, usd_state)
    assert eur_targets.gross_notional > usd_targets.gross_notional
    assert eur_targets.gross_notional == pytest.approx(
        usd_targets.gross_notional * EUR_RATE, rel=0.25
    )


def test_missing_fx_rate_fails_closed_through_pipeline(
    instrument_master: InstrumentsConfig,
) -> None:
    translator = ContractTranslator(instruments=instrument_master)
    pipeline = _pipeline(translator, instrument_master)
    state = _states(equity=EUR_EQUITY, account_currency="EUR", fx_rates={})[0]
    with pytest.raises(KeyError, match="no FX rate"):
        pipeline.run(state)


def test_constructor_fx_applies_without_state_rates(
    instrument_master: InstrumentsConfig,
) -> None:
    translator = ContractTranslator(
        instruments=instrument_master,
        account_currency="EUR",
        fx_rates={"EUR": EUR_RATE},
    )
    state = _states(equity=EUR_EQUITY, account_currency="USD", fx_rates={})[0]
    from src.core.types import WeightVector

    targets = translator.translate(WeightVector({"ES": 1.0}), state)
    usd_translator = ContractTranslator(instruments=instrument_master)
    usd_targets = usd_translator.translate(
        WeightVector({"ES": 1.0}), replace(state, equity=USD_EQUITY)
    )
    assert targets.quantities == usd_targets.quantities


def test_backtest_first_rebalance_eur_matches_usd_equivalent(
    instrument_master: InstrumentsConfig,
) -> None:
    """The event backtester's first rebalance sizes EUR equity exactly like 120k USD."""
    eur_states = _states(equity=EUR_EQUITY, account_currency="EUR", fx_rates={"EUR": EUR_RATE})
    usd_states = _states(equity=USD_EQUITY, account_currency="USD", fx_rates={})

    eur_engine = EventBacktester(
        _pipeline(
            ContractTranslator(instruments=instrument_master),
            instrument_master,
            _backtest_config(),
        ),
        ContractTranslator(instruments=instrument_master),
        instrument_master=instrument_master,
        initial_equity=EUR_EQUITY,
    )
    usd_engine = EventBacktester(
        _pipeline(
            ContractTranslator(instruments=instrument_master),
            instrument_master,
            _backtest_config(),
        ),
        ContractTranslator(instruments=instrument_master),
        instrument_master=instrument_master,
        initial_equity=USD_EQUITY,
    )
    eur_result = eur_engine.run(states=eur_states)
    usd_result = usd_engine.run(states=usd_states)
    first_ts = eur_states[0].timestamp

    def first_deltas(result) -> dict[str, int]:
        deltas: dict[str, int] = {}
        for trade in result.trades:
            if trade.timestamp == first_ts:
                deltas[trade.instrument_id] = (
                    deltas.get(trade.instrument_id, 0) + trade.signed_quantity
                )
        return deltas

    assert first_deltas(eur_result) == first_deltas(usd_result)
    assert first_deltas(eur_result), "the first rebalance must trade"
    # No translator failures were recorded (fail-closed conversion worked).
    assert not [warning for warning in eur_result.warnings if "translator failed" in warning]
