"""Unit tests for the immutable core contracts in :mod:`src.core.types`."""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, timedelta, timezone

import numpy as np
import pytest
from src.core.types import (
    AccountSnapshot,
    AssetClass,
    Bar,
    ContractTarget,
    ContractTargetSet,
    Fill,
    InstrumentSnapshot,
    MarketState,
    OrderEvent,
    OrderReceipt,
    OrderSide,
    OrderStatus,
    OrderType,
    PipelineResult,
    Position,
    ReconcileReport,
    RiskAssessment,
    RollEvent,
    RollTrigger,
    RoundingMode,
    TimeInForce,
    WeightVector,
    ensure_utc,
)

from tests.fixtures.market import (
    FIXED_TIMESTAMP,
    make_contract_target,
    make_market_state,
    make_snapshot,
)

NAIVE = datetime(2024, 1, 2, 21, 0)
BERLIN = timezone(timedelta(hours=1))


# ---------------------------------------------------------------------------
# ensure_utc
# ---------------------------------------------------------------------------


def test_ensure_utc_rejects_naive_datetime() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        ensure_utc(NAIVE)


def test_ensure_utc_converts_other_timezones() -> None:
    converted = ensure_utc(datetime(2024, 1, 2, 22, 0, tzinfo=BERLIN))
    assert converted.tzinfo is UTC
    assert converted.hour == 21


def test_ensure_utc_rejects_non_datetime() -> None:
    with pytest.raises(TypeError, match="datetime"):
        ensure_utc("2024-01-02")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# InstrumentSnapshot
# ---------------------------------------------------------------------------


def test_snapshot_rejects_non_positive_price() -> None:
    with pytest.raises(ValueError, match="price must be non-zero"):
        make_snapshot("ES", price=0.0)
    with pytest.raises(ValueError, match="finite"):
        make_snapshot("ES", price=math.nan)


def test_snapshot_rejects_crossed_quote() -> None:
    with pytest.raises(ValueError, match=r"bid .* > ask"):
        InstrumentSnapshot(
            instrument_id="ES",
            root_symbol="ES",
            price=4800.0,
            bid=4801.0,
            ask=4800.5,
        )


def test_snapshot_features_are_immutable_and_accessors_work() -> None:
    snapshot = make_snapshot("ES", features={"ret_21d": 0.03})
    assert snapshot.feature("ret_21d") == pytest.approx(0.03)
    assert math.isnan(snapshot.feature("missing"))
    assert snapshot.feature("missing", 1.5) == 1.5
    with pytest.raises(TypeError):
        snapshot.features["ret_21d"] = 0.5  # type: ignore[index]


def test_snapshot_mid_and_spread_use_two_sided_quote() -> None:
    snapshot = make_snapshot("ES", price=4800.0, bid_ask_ticks=2.0)
    assert snapshot.spread == pytest.approx(0.5)
    assert snapshot.mid == pytest.approx(4800.0)


def test_snapshot_rejects_negative_volume() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        make_snapshot("ES", volume=-1.0)


# ---------------------------------------------------------------------------
# MarketState
# ---------------------------------------------------------------------------


def test_market_state_rejects_naive_timestamp() -> None:
    snapshot = make_snapshot("ES")
    with pytest.raises(ValueError, match="timezone-aware"):
        MarketState(timestamp=NAIVE, instruments={"ES": snapshot})


def test_market_state_rejects_non_positive_equity() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        make_market_state(equity=0.0)


def test_market_state_rejects_key_mismatch() -> None:
    snapshot = make_snapshot("ES")
    with pytest.raises(ValueError, match="does not match snapshot"):
        MarketState(timestamp=FIXED_TIMESTAMP, instruments={"NQ": snapshot})


def test_market_state_rejects_covariance_without_labels() -> None:
    snapshot = make_snapshot("ES")
    with pytest.raises(ValueError, match="covariance_instruments"):
        MarketState(
            timestamp=FIXED_TIMESTAMP,
            instruments={"ES": snapshot},
            covariance=np.eye(1),
        )


def test_market_state_covariance_is_read_only_and_labels_aligned() -> None:
    state = make_market_state(roots=("ES", "NQ"))
    assert state.covariance is not None
    assert state.covariance_instruments == ("ES", "NQ")
    assert not state.covariance.flags.writeable
    with pytest.raises(ValueError):
        state.covariance[0, 0] = 1.0  # type: ignore[index]


def test_market_state_accessors_and_volatility() -> None:
    state = make_market_state(roots=("ES", "NQ"))
    assert "ES" in state
    assert len(state) == 2
    assert state.price("ES") == pytest.approx(4800.0)
    assert state.feature("ES", "ret_21d") == pytest.approx(state.snapshot("ES").feature("ret_21d"))
    assert state.annualized_vol("ES") > 0.0
    assert math.isnan(state.annualized_vol("CL"))


def test_market_state_subset_preserves_alignment() -> None:
    state = make_market_state(roots=("ES", "NQ", "CL"))
    subset = state.subset(("CL", "ES"))
    assert subset.ids == ("CL", "ES")
    assert subset.covariance is not None
    assert subset.covariance_instruments == ("CL", "ES")
    assert subset.covariance.shape == (2, 2)
    assert set(subset.returns) == {"CL", "ES"}


def test_market_state_unknown_instrument_raises() -> None:
    state = make_market_state(roots=("ES",))
    with pytest.raises(KeyError, match="not part of the market state"):
        state.snapshot("CL")
    with pytest.raises(KeyError):
        state.subset(("CL",))


def test_market_state_metadata_is_frozen() -> None:
    state = make_market_state()
    with pytest.raises(TypeError):
        state.metadata["vix_term_slope"] = 0.5  # type: ignore[index]


def test_market_state_from_snapshots() -> None:
    snapshots = [make_snapshot("ES"), make_snapshot("CL")]
    state = MarketState.from_snapshots(FIXED_TIMESTAMP, snapshots, equity=500_000.0)
    assert state.ids == ("ES", "CL")
    assert state.equity == pytest.approx(500_000.0)


# ---------------------------------------------------------------------------
# WeightVector
# ---------------------------------------------------------------------------


def test_weight_vector_rejects_nan() -> None:
    with pytest.raises(ValueError, match="finite"):
        WeightVector({"ES": math.nan})


def test_weight_vector_statistics() -> None:
    vector = WeightVector({"ES": 0.4, "CL": -0.2, "GC": 0.1})
    assert vector.gross == pytest.approx(0.7)
    assert vector.net == pytest.approx(0.3)
    assert vector.l2 == pytest.approx(math.sqrt(0.16 + 0.04 + 0.01))
    assert vector.max_abs() == pytest.approx(0.4)
    assert not vector.is_flat()
    assert WeightVector.empty().is_flat()


def test_weight_vector_arithmetic() -> None:
    left = WeightVector({"ES": 0.3, "CL": -0.1})
    right = WeightVector({"ES": 0.1, "GC": 0.2})
    added = left + right
    assert added["ES"] == pytest.approx(0.4)
    assert added["CL"] == pytest.approx(-0.1)
    assert added["GC"] == pytest.approx(0.2)
    assert (left - right)["ES"] == pytest.approx(0.2)
    assert (2.0 * left)["ES"] == pytest.approx(0.6)
    assert (-left)["CL"] == pytest.approx(0.1)
    assert left == {"ES": 0.3, "CL": -0.1}


def test_weight_vector_transformations() -> None:
    vector = WeightVector({"ES": 0.6, "CL": -0.2})
    assert vector.scaled(0.5).gross == pytest.approx(0.4)
    assert vector.clipped(0.3)["ES"] == pytest.approx(0.3)
    assert vector.clipped(0.3)["CL"] == pytest.approx(-0.2)
    normalized = vector.normalized_gross(1.0)
    assert normalized.gross == pytest.approx(1.0)
    assert WeightVector.empty().normalized_gross(1.0).is_flat()
    assert vector.subset({"ES"}).as_dict() == {"ES": 0.6}
    assert vector.reindexed(("ES", "GC")).as_dict() == {"ES": 0.6, "GC": 0.0}


def test_weight_vector_to_array_and_constructors() -> None:
    vector = WeightVector.from_array(("ES", "CL"), [0.25, -0.75])
    assert vector.as_dict() == {"ES": 0.25, "CL": -0.75}
    assert vector.to_array(("CL", "ES")).tolist() == [-0.75, 0.25]
    equal = WeightVector.equal_weight(("ES", "CL"), gross=2.0)
    assert equal["ES"] == pytest.approx(1.0)
    assert equal["CL"] == pytest.approx(1.0)


def test_weight_vector_from_array_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="does not match"):
        WeightVector.from_array(("ES", "CL"), [0.1])


def test_weight_vector_is_hashable_and_repr() -> None:
    vector = WeightVector({"ES": 0.1})
    assert isinstance(hash(vector), int)
    assert "gross=" in repr(vector)


# ---------------------------------------------------------------------------
# ContractTarget / ContractTargetSet
# ---------------------------------------------------------------------------


def test_contract_target_computes_notional_and_margin() -> None:
    target = make_contract_target("ES", 3)
    assert target.signed_notional == pytest.approx(3 * 4800.0 * 50.0)
    assert target.abs_notional == pytest.approx(target.signed_notional)
    assert target.initial_margin_required == pytest.approx(3 * 15000.0)
    assert target.side is OrderSide.BUY
    assert make_contract_target("CL", -2).signed_notional < 0
    assert make_contract_target("CL", -2).side is OrderSide.SELL


def test_contract_target_validations() -> None:
    with pytest.raises(TypeError, match="quantity must be an int"):
        ContractTarget(instrument_id="ES", quantity=1.5, price=4800.0, multiplier=50.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="multiplier"):
        ContractTarget(instrument_id="ES", quantity=1, price=4800.0, multiplier=0.0)
    with pytest.raises(ValueError, match="margins must be non-negative"):
        ContractTarget(
            instrument_id="ES",
            quantity=1,
            price=4800.0,
            multiplier=50.0,
            initial_margin=-1.0,
        )


def test_contract_target_set_aggregates() -> None:
    targets = ContractTargetSet((make_contract_target("ES", 2), make_contract_target("CL", -1)))
    assert len(targets) == 2
    assert bool(targets)
    assert targets.quantities == {"ES": 2, "CL": -1}
    assert targets.gross_notional == pytest.approx(2 * 4800.0 * 50.0 + 1 * 72.5 * 1000.0)
    assert targets.net_notional == pytest.approx(2 * 4800.0 * 50.0 - 72.5 * 1000.0)
    assert targets.total_initial_margin == pytest.approx(2 * 15000.0 + 6500.0)
    assert targets.get("ES") is not None
    assert targets.get("ZN") is None
    assert not bool(ContractTargetSet.empty())
    assert len(targets.nonzero()) == 2


def test_contract_target_set_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="duplicate contract target"):
        ContractTargetSet((make_contract_target("ES", 1), make_contract_target("ES", 2)))


# ---------------------------------------------------------------------------
# Position / AccountSnapshot
# ---------------------------------------------------------------------------


def test_position_pnl() -> None:
    position = Position(
        instrument_id="ES",
        quantity=2,
        average_price=4750.0,
        market_price=4800.0,
        multiplier=50.0,
    )
    assert position.unrealized_pnl == pytest.approx(2 * 50.0 * 50.0)
    assert position.signed_notional == pytest.approx(2 * 4800.0 * 50.0)


def test_account_snapshot_margin_utilization() -> None:
    account = AccountSnapshot(
        timestamp=FIXED_TIMESTAMP,
        net_liquidation=1_000_000.0,
        initial_margin_used=150_000.0,
    )
    assert account.margin_utilization == pytest.approx(0.15)
    with pytest.raises(ValueError, match="strictly positive"):
        AccountSnapshot(timestamp=FIXED_TIMESTAMP, net_liquidation=-5.0)


# ---------------------------------------------------------------------------
# Orders / fills / receipts
# ---------------------------------------------------------------------------


def test_order_event_validation() -> None:
    with pytest.raises(ValueError, match="limit_price"):
        OrderEvent(
            order_id="o-1",
            timestamp=FIXED_TIMESTAMP,
            instrument_id="ES",
            side=OrderSide.BUY,
            quantity=1,
            order_type=OrderType.LIMIT,
        )
    with pytest.raises(ValueError, match="quantity must be strictly positive"):
        OrderEvent(
            order_id="o-1",
            timestamp=FIXED_TIMESTAMP,
            instrument_id="ES",
            side=OrderSide.BUY,
            quantity=0,
        )
    with pytest.raises(ValueError, match="stop_price"):
        OrderEvent(
            order_id="o-1",
            timestamp=FIXED_TIMESTAMP,
            instrument_id="ES",
            side=OrderSide.SELL,
            quantity=1,
            order_type=OrderType.STOP,
        )


def test_order_event_signed_quantity_and_market_factory() -> None:
    order = OrderEvent.market(
        order_id="o-1",
        timestamp=FIXED_TIMESTAMP,
        instrument_id="ES",
        side=OrderSide.SELL,
        quantity=3,
    )
    assert order.signed_quantity == -3
    assert order.time_in_force is TimeInForce.DAY


def test_fill_and_receipt_validation() -> None:
    fill = Fill(
        order_id="o-1",
        instrument_id="ES",
        side=OrderSide.BUY,
        quantity=2,
        price=4800.25,
        timestamp=FIXED_TIMESTAMP,
        commission=4.5,
        slippage_ticks=1.0,
    )
    assert fill.signed_quantity == 2
    with pytest.raises(ValueError, match="strictly positive"):
        Fill(
            order_id="o-1",
            instrument_id="ES",
            side=OrderSide.BUY,
            quantity=0,
            price=4800.0,
            timestamp=FIXED_TIMESTAMP,
        )
    receipt = OrderReceipt(
        order_id="o-1",
        status=OrderStatus.FILLED,
        filled_quantity=2,
        average_price=4800.25,
    )
    assert receipt.status is OrderStatus.FILLED
    with pytest.raises(ValueError, match="non-negative"):
        OrderReceipt(order_id="o-1", status=OrderStatus.REJECTED, filled_quantity=-1)


# ---------------------------------------------------------------------------
# Risk assessment
# ---------------------------------------------------------------------------


def test_risk_assessment_ranges_and_criticality() -> None:
    assessment = RiskAssessment(
        instrument_id="ES",
        timestamp=FIXED_TIMESTAMP,
        sentiment_score=2.0,
        semantic_risk_score=4.8,
        confidence=0.95,
    )
    assert assessment.is_critical
    assert assessment.to_dict()["semantic_risk_score"] == pytest.approx(4.8)
    with pytest.raises(ValueError, match="confidence"):
        RiskAssessment(
            instrument_id="ES",
            timestamp=FIXED_TIMESTAMP,
            sentiment_score=3.0,
            semantic_risk_score=3.0,
            confidence=1.5,
        )
    with pytest.raises(ValueError, match="sentiment_score"):
        RiskAssessment(
            instrument_id="ES",
            timestamp=FIXED_TIMESTAMP,
            sentiment_score=6.0,
            semantic_risk_score=3.0,
            confidence=0.5,
        )


# ---------------------------------------------------------------------------
# Bar / roll
# ---------------------------------------------------------------------------


def test_bar_validation() -> None:
    bar = Bar(
        timestamp=FIXED_TIMESTAMP,
        instrument_id="ES",
        root_symbol="ES",
        expiration=date(2024, 3, 15),
        open=4800.0,
        high=4810.0,
        low=4790.0,
        close=4805.0,
        volume=1000.0,
    )
    assert bar.close == pytest.approx(4805.0)
    with pytest.raises(ValueError, match="inconsistent OHLC"):
        Bar(
            timestamp=FIXED_TIMESTAMP,
            instrument_id="ES",
            root_symbol="ES",
            expiration=date(2024, 3, 15),
            open=4800.0,
            high=4790.0,
            low=4795.0,
            close=4805.0,
            volume=1000.0,
        )


def test_roll_event_gap_and_yield() -> None:
    roll = RollEvent(
        root_symbol="CL",
        roll_datetime=FIXED_TIMESTAMP,
        from_expiration=date(2024, 2, 20),
        to_expiration=date(2024, 3, 20),
        trigger=RollTrigger.CALENDAR_DAYS_BEFORE_FND,
        from_price=72.0,
        to_price=72.6,
    )
    assert roll.roll_gap == pytest.approx(0.6)
    assert roll.roll_yield_annualized == pytest.approx(-(0.6 / 72.0) * (365.0 / 29))
    with pytest.raises(ValueError, match="after from_expiration"):
        RollEvent(
            root_symbol="CL",
            roll_datetime=FIXED_TIMESTAMP,
            from_expiration=date(2024, 3, 20),
            to_expiration=date(2024, 2, 20),
            trigger=RollTrigger.CALENDAR_DAYS_BEFORE_FND,
            from_price=72.0,
            to_price=72.6,
        )


# ---------------------------------------------------------------------------
# Pipeline result / reconcile report
# ---------------------------------------------------------------------------


def test_pipeline_result_freezes_diagnostics() -> None:
    result = PipelineResult(
        timestamp=FIXED_TIMESTAMP,
        universe=("ES", "CL"),
        base_weights=WeightVector({"ES": 0.5}),
        timing_weights=WeightVector({"ES": 0.4}),
        risk_weights=WeightVector({"ES": 0.3}),
        final_weights=WeightVector({"ES": 0.3}),
        diagnostics={"stage": "risk", "notes": "ok"},
    )
    assert result.final_weights["ES"] == pytest.approx(0.3)
    with pytest.raises(TypeError):
        result.diagnostics["new"] = "x"  # type: ignore[index]
    with pytest.raises(ValueError, match="elapsed_ms"):
        PipelineResult(
            timestamp=FIXED_TIMESTAMP,
            universe=("ES",),
            base_weights=WeightVector.empty(),
            timing_weights=WeightVector.empty(),
            risk_weights=WeightVector.empty(),
            final_weights=WeightVector.empty(),
            elapsed_ms=-1.0,
        )


def test_reconcile_report_is_immutable() -> None:
    report = ReconcileReport(
        timestamp=FIXED_TIMESTAMP,
        ok=False,
        expected_positions={"ES": 2},
        actual_positions={"ES": 1},
        drift={"ES": 1},
        halted=True,
        message="drift",
    )
    assert report.drift["ES"] == 1
    with pytest.raises(TypeError):
        report.drift["ES"] = 0  # type: ignore[index]


def test_rounding_mode_enum() -> None:
    assert RoundingMode.FLOOR.value == "floor"
    assert AssetClass.EQUITY_INDEX.value == "equity_index"


# ---------------------------------------------------------------------------
# Base-currency FX normalisation (Phase 7)
# ---------------------------------------------------------------------------


def test_market_state_defaults_to_usd_account() -> None:
    state = make_market_state(roots=("ES",), equity=1_000_000.0)
    assert state.account_currency == "USD"
    assert state.fx_rates == {}
    assert state.fx_rate_to_usd() == pytest.approx(1.0)
    assert state.equity_in_usd == pytest.approx(1_000_000.0)


def test_market_state_converts_non_usd_equity() -> None:
    state = make_market_state(
        roots=("ES",), equity=100_000.0, account_currency="eur", fx_rates={"EUR": 1.1}
    )
    assert state.account_currency == "EUR"
    assert state.fx_rate_to_usd() == pytest.approx(1.1)
    assert state.fx_rate_to_usd("eur") == pytest.approx(1.1)
    assert state.fx_rate_to_usd("USD") == pytest.approx(1.0)
    assert state.equity_in_usd == pytest.approx(110_000.0)


def test_market_state_missing_fx_rate_is_actionable() -> None:
    state = make_market_state(roots=("ES",), equity=100_000.0, account_currency="GBP")
    assert state.equity and state.equity > 0
    with pytest.raises(KeyError, match="no FX rate"):
        state.fx_rate_to_usd()
    with pytest.raises(KeyError, match="no FX rate"):
        _ = state.equity_in_usd


def test_market_state_rejects_invalid_currency_and_rates() -> None:
    snapshot = make_snapshot("ES")
    with pytest.raises(ValueError, match="3-letter"):
        MarketState(
            timestamp=FIXED_TIMESTAMP, instruments={"ES": snapshot}, account_currency="EURO"
        )
    with pytest.raises(ValueError, match="strictly positive"):
        MarketState(
            timestamp=FIXED_TIMESTAMP,
            instruments={"ES": snapshot},
            account_currency="EUR",
            fx_rates={"EUR": -1.0},
        )
    with pytest.raises(ValueError, match="USD"):
        MarketState(
            timestamp=FIXED_TIMESTAMP,
            instruments={"ES": snapshot},
            fx_rates={"USD": 1.2},
        )


def test_market_state_fx_rates_are_frozen() -> None:
    state = make_market_state(
        roots=("ES",), equity=100_000.0, account_currency="EUR", fx_rates={"EUR": 1.1}
    )
    with pytest.raises(TypeError):
        state.fx_rates["EUR"] = 1.2  # type: ignore[index]


def test_market_state_subset_and_from_snapshots_propagate_fx() -> None:
    state = make_market_state(
        roots=("ES", "CL"),
        equity=100_000.0,
        account_currency="EUR",
        fx_rates={"EUR": 1.1, "GBP": 1.25},
    )
    subset = state.subset(("CL",))
    assert subset.account_currency == "EUR"
    assert dict(subset.fx_rates) == {"EUR": 1.1, "GBP": 1.25}
    rebuilt = MarketState.from_snapshots(
        FIXED_TIMESTAMP,
        [make_snapshot("ES")],
        equity=50_000.0,
        account_currency="GBP",
        fx_rates={"GBP": 1.25},
    )
    assert rebuilt.equity_in_usd == pytest.approx(62_500.0)


def test_market_state_converted_to_currency_round_trips() -> None:
    eur_state = make_market_state(
        roots=("ES",), equity=100_000.0, account_currency="EUR", fx_rates={"EUR": 1.1}
    )
    usd_state = eur_state.converted_to_currency("USD")
    assert usd_state.account_currency == "USD"
    assert usd_state.equity == pytest.approx(110_000.0)
    assert usd_state.equity_in_usd == pytest.approx(110_000.0)
    back = usd_state.converted_to_currency("EUR")
    assert back.equity == pytest.approx(100_000.0)
    with pytest.raises(KeyError, match="no fx_rates"):
        eur_state.converted_to_currency("JPY")
