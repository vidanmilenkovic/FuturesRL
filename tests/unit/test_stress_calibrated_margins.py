"""Stress-regime margin multiplier, calibrated margin overrides and cost breakdown (task-86)."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from configs.instruments import InstrumentsConfig
from src.backtest.engine import EventBacktester, SimpleContractTranslator
from src.backtest.margin_simulator import (
    DEFAULT_STRESS_MARGIN_MULTIPLIER,
    CalibratedMargin,
    MarginSimulator,
    load_calibrated_margins,
)
from src.core.types import Position
from src.risk.contract_translator import ContractTranslator

from tests.fixtures.market import make_bars_frame, make_market_state

POSITION = Position(
    instrument_id="ES",
    quantity=2,
    average_price=4800.0,
    market_price=4800.0,
    multiplier=50.0,
    initial_margin=10_000.0,
    maintenance_margin=9_000.0,
    expiration=date(2024, 3, 15),
)


def _write_calibration(
    path: Path, *, initial: float = 20_000.0, maintenance: float = 18_000.0
) -> Path:
    payload = {
        "schema_version": 1,
        "account_id": "DU123",
        "calibrated_at": datetime.now(UTC).isoformat(),
        "roots": {
            "ES": {
                "broker_commission_per_contract": 0.85,
                "exchange_and_reg_fee_per_contract": 1.40,
                "initial_margin": initial,
                "maintenance_margin": maintenance,
                "commission_source": "ibkr_whatif",
                "margin_source": "ibkr_whatif",
            }
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class _ConstantPipeline:
    def __init__(self, weights: dict[str, float]) -> None:
        from src.core.types import WeightVector

        self._weights = WeightVector(weights)

    @property
    def name(self) -> str:
        return "constant"

    def run(self, state: Any) -> Any:
        from src.core.types import PipelineResult

        return PipelineResult(
            timestamp=state.timestamp,
            universe=state.ids,
            base_weights=self._weights,
            timing_weights=self._weights,
            risk_weights=self._weights,
            final_weights=self._weights,
        )


def test_normal_regime_margin_is_unchanged() -> None:
    simulator = MarginSimulator()
    normal = simulator.margin_requirements({"ES": POSITION})
    assert normal.initial_margin == pytest.approx(20_000.0)
    assert normal.maintenance_margin == pytest.approx(18_000.0)
    assert normal.stress_applied is False
    assert normal.stress_multiplier == pytest.approx(1.0)
    assert normal.calibrated is False


def test_stress_multiplier_applies_above_vol_threshold() -> None:
    simulator = MarginSimulator(stress_margin_multiplier=1.35, stress_vol_threshold=0.30)
    simulator.set_volatilities({"ES": 0.29})
    normal = simulator.margin_requirements({"ES": POSITION})
    assert normal.initial_margin == pytest.approx(20_000.0)
    assert normal.stress_applied is False

    simulator.set_volatilities({"ES": 0.45})
    stressed = simulator.margin_requirements({"ES": POSITION})
    assert stressed.initial_margin == pytest.approx(20_000.0 * 1.35)
    assert stressed.maintenance_margin == pytest.approx(18_000.0 * 1.35)
    assert stressed.stress_applied is True
    assert stressed.stress_multiplier == pytest.approx(1.35)
    assert stressed.stress_roots == ("ES",)
    payload = stressed.to_dict()
    assert payload["stress_applied"] is True
    assert payload["stress_multiplier"] == pytest.approx(1.35)


def test_stress_multiplier_configurable_and_default() -> None:
    assert MarginSimulator().stress_margin_multiplier == pytest.approx(
        DEFAULT_STRESS_MARGIN_MULTIPLIER
    )
    custom = MarginSimulator(stress_margin_multiplier=1.6, stress_vol_threshold=0.2)
    custom.set_volatilities({"ES": 0.25})
    requirements = custom.margin_requirements({"ES": POSITION})
    assert requirements.initial_margin == pytest.approx(20_000.0 * 1.6)
    with pytest.raises(ValueError):
        MarginSimulator(stress_margin_multiplier=0.9)
    with pytest.raises(ValueError):
        MarginSimulator(stress_vol_threshold=0.0)


def test_calibrated_margins_loader_and_apply(tmp_path: Path) -> None:
    path = _write_calibration(tmp_path / "cache" / "ibkr_calibrated_costs.json")
    loaded = load_calibrated_margins(path)
    assert loaded["ES"].initial_margin == pytest.approx(20_000.0)
    assert loaded["ES"].maintenance_margin == pytest.approx(18_000.0)
    assert loaded["ES"].margin_source == "ibkr_whatif"

    simulator = MarginSimulator(calibrated_costs_path=path)
    requirements = simulator.margin_requirements({"ES": POSITION})
    # POSITION has quantity 2; calibrated per-contract margins are authoritative.
    assert requirements.initial_margin == pytest.approx(40_000.0)
    assert requirements.calibrated is True
    assert requirements.calibrated_roots == ("ES",)

    # Calibrated override is applied BEFORE the stress multiplier.
    simulator.set_volatilities({"ES": 0.50})
    stressed = simulator.margin_requirements({"ES": POSITION})
    assert stressed.initial_margin == pytest.approx(40_000.0 * 1.35)
    assert stressed.calibrated is True
    assert stressed.stress_applied is True


def test_calibration_absent_corrupt_and_empty_fall_back(tmp_path: Path) -> None:
    missing = tmp_path / "cache" / "does_not_exist.json"
    assert load_calibrated_margins(missing) == {}
    corrupt = tmp_path / "cache" / "corrupt.json"
    corrupt.parent.mkdir(parents=True)
    corrupt.write_text("{not-json", encoding="utf-8")
    assert load_calibrated_margins(corrupt) == {}
    empty = tmp_path / "cache" / "empty.json"
    empty.parent.mkdir(parents=True, exist_ok=True)
    empty.write_text(json.dumps({"roots": {}}), encoding="utf-8")
    assert load_calibrated_margins(empty) == {}
    simulator = MarginSimulator(calibrated_costs_path=corrupt)
    requirements = simulator.margin_requirements({"ES": POSITION})
    assert requirements.initial_margin == pytest.approx(20_000.0)
    assert requirements.calibrated is False


def _high_vol_state(*, volatility: float = 0.45) -> Any:
    state = make_market_state(("ES",), equity=1_000_000.0)
    return replace(
        state,
        covariance=np.array([[volatility * volatility]]),
        covariance_instruments=("ES",),
    )


def test_contract_translator_applies_calibrated_and_stress_margins(
    instrument_master: InstrumentsConfig,
) -> None:
    state = make_market_state(("ES",), equity=1_000_000.0)
    calibrated = {"ES": CalibratedMargin(initial_margin=20_000.0, maintenance_margin=18_000.0)}
    translator = ContractTranslator(
        instruments=instrument_master,
        calibrated_margins=calibrated,
        load_calibrated_costs=False,
        stress_vol_threshold=0.30,
        stress_margin_multiplier=1.35,
    )
    targets = translator.translate({"ES": 0.30}, state)
    assert targets.targets
    target = targets.targets[0]
    # State fixture covariance is below the stress threshold: calibrated only.
    assert target.initial_margin == pytest.approx(20_000.0)
    assert target.maintenance_margin == pytest.approx(18_000.0)

    stressed_state = _high_vol_state(volatility=0.45)
    stressed_targets = translator.translate({"ES": 0.30}, stressed_state)
    stressed = stressed_targets.targets[0]
    assert stressed.initial_margin == pytest.approx(20_000.0 * 1.35)
    assert stressed.maintenance_margin == pytest.approx(18_000.0 * 1.35)


def test_event_engine_reports_stress_margin_and_cost_breakdown(
    instrument_master: InstrumentsConfig,
) -> None:
    timestamps = [
        make_market_state(("ES",), equity=1_000_000.0).timestamp,
    ]
    from datetime import timedelta

    states = [
        replace(
            _high_vol_state(volatility=0.45),
            timestamp=timestamps[0] + timedelta(days=index),
        )
        for index in range(3)
    ]
    engine = EventBacktester(
        _ConstantPipeline({"ES": 0.30}),
        SimpleContractTranslator(instrument_master),
        instrument_master=instrument_master,
        initial_equity=1_000_000.0,
    )
    result = engine.run(states=states)
    assert result.margin_stress_applied is True
    assert result.margin_stress_multiplier == pytest.approx(1.35)
    assert result.margin_stress_roots == ("ES",)
    breakdown = result.cost_breakdown
    assert breakdown["total_costs"] == pytest.approx(result.total_costs)
    assert breakdown["total_commissions"] == pytest.approx(result.total_commissions)
    assert breakdown["total_execution_costs"] == pytest.approx(
        result.total_commissions + result.slippage
    )
    assert breakdown["roll_cost_total"] == pytest.approx(result.roll_cost_total)
    assert breakdown["roll_commissions"] + breakdown[
        "roll_exchange_and_regulatory_fees"
    ] + breakdown["roll_slippage"] == pytest.approx(result.roll_cost_total)
    assert "cost_breakdown" in result.to_dict()
    assert result.to_dict()["margin_stress_applied"] is True


def test_event_engine_calibrated_margin_metadata(
    instrument_master: InstrumentsConfig, tmp_path: Path
) -> None:
    path = _write_calibration(tmp_path / "cache" / "ibkr_calibrated_costs.json")
    from src.backtest.margin_simulator import MarginSimulator as Sim

    simulator = Sim(calibrated_costs_path=path)
    timestamps = [make_market_state(("ES",), equity=1_000_000.0).timestamp]
    from datetime import timedelta

    states = [
        replace(
            make_market_state(("ES",), equity=1_000_000.0),
            timestamp=timestamps[0] + timedelta(days=i),
        )
        for i in range(2)
    ]
    engine = EventBacktester(
        _ConstantPipeline({"ES": 0.30}),
        SimpleContractTranslator(instrument_master),
        instrument_master=instrument_master,
        initial_equity=1_000_000.0,
        margin_simulator=simulator,
    )
    result = engine.run(states=states)
    assert result.margin_calibrated is True
    assert result.margin_calibrated_roots == ("ES",)


def test_daily_cost_breakdown_is_accounted(
    instrument_master: InstrumentsConfig,
) -> None:
    bars = make_bars_frame(("ES", "NQ"), periods=12, seed=5)
    engine = EventBacktester(
        _ConstantPipeline({"ES": 0.20, "NQ": 0.10}),
        SimpleContractTranslator(instrument_master),
        instrument_master=instrument_master,
        initial_equity=1_000_000.0,
    )
    result = engine.run(bars=bars)
    breakdown = result.cost_breakdown
    assert breakdown["broker_commissions"] == pytest.approx(result.commissions)
    assert breakdown["exchange_and_regulatory_fees"] == pytest.approx(result.fees)
    assert breakdown["slippage"] == pytest.approx(result.slippage)
    assert breakdown["total_costs"] == pytest.approx(result.total_costs)
    result.check_accounting_identity()


def test_roll_slippage_multiplier_increases_result_roll_costs(
    instrument_master: InstrumentsConfig,
) -> None:
    """Task-84 multiplier must visibly flow through the task-86 roll breakdown."""
    from datetime import timedelta

    import polars as pl
    from src.backtest.cost_model import CostModel

    first = make_bars_frame(("ES",), periods=3, expiration=date(2024, 3, 15), seed=3)
    second = make_bars_frame(
        ("ES",),
        periods=3,
        start=first["timestamp_utc"][0] + timedelta(days=3),
        expiration=date(2024, 6, 21),
        seed=3,
    )
    bars = pl.concat([first, second]).sort(["timestamp_utc", "root_symbol"])

    def run(multiplier: float) -> Any:
        cost_model = CostModel(
            instrument_master=instrument_master,
            roll_slippage_multiplier=multiplier,
        )
        engine = EventBacktester(
            _ConstantPipeline({"ES": 0.50}),
            SimpleContractTranslator(instrument_master),
            instrument_master=instrument_master,
            cost_model=cost_model,
            initial_equity=1_000_000.0,
        )
        return engine.run(bars=bars)

    baseline = run(1.0)
    multiplied = run(1.5)
    assert len(baseline.roll_events) == len(multiplied.roll_events) == 1
    assert multiplied.roll_slippage > baseline.roll_slippage
    assert multiplied.roll_cost_total > baseline.roll_cost_total
    assert multiplied.cost_breakdown["roll_slippage"] > baseline.cost_breakdown["roll_slippage"]
