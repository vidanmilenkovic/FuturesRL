"""Unit tests for base-currency FX normalisation in :class:`ContractTranslator`.

All expectations are hand-computed from the committed instrument master (ES at
4800 with a 50x multiplier is a 240,000 USD contract; MES is 24,000 USD) and the
deterministic pytest fixtures. Every test is offline and deterministic.

Convention used throughout: ``fx_rates["EUR"] == 1.20`` means one EUR buys 1.20
USD, so a 100,000 EUR account has 120,000 USD of sizing equity.
"""

from __future__ import annotations

import math

import pytest
from configs.instruments import InstrumentsConfig, parse_instruments_config
from configs.strategy import TranslatorConfig
from hypothesis import given, settings
from hypothesis import strategies as st
from src.core.interfaces import ContractTranslatorProtocol
from src.core.types import (
    ContractTargetSet,
    MarketState,
    RoundingMode,
    WeightVector,
)
from src.risk.contract_translator import ContractTranslator

from tests.fixtures.market import make_market_state, make_weight_vector

DEFAULT_ROOTS: tuple[str, ...] = ("ES", "NQ", "CL", "GC", "ZN")
MICRO_PAIR_ROOTS: tuple[str, ...] = ("ES", "MES")

#: ES at 4,800 x 50 = 240,000 USD per standard contract; MES = 24,000 USD.
ES_CONTRACT_NOTIONAL_USD: float = 240_000.0
MES_CONTRACT_NOTIONAL_USD: float = 24_000.0
#: Initial margins from the committed instrument master.
ES_INITIAL_MARGIN_USD: float = 15_000.0
MES_INITIAL_MARGIN_USD: float = 1_500.0


def _translator(**kwargs: object) -> ContractTranslator:
    defaults: dict[str, object] = {"rounding_mode": RoundingMode.NEAREST}
    defaults.update(kwargs)
    return ContractTranslator(**defaults)  # type: ignore[arg-type]


def _eur_state(
    *, equity: float = 100_000.0, rate: float = 1.2, roots: tuple[str, ...] = MICRO_PAIR_ROOTS
) -> MarketState:
    return make_market_state(
        roots=roots,
        equity=equity,
        account_currency="EUR",
        fx_rates={"EUR": rate},
    )


def _usd_state(
    *, equity: float = 1_000_000.0, roots: tuple[str, ...] = MICRO_PAIR_ROOTS
) -> MarketState:
    return make_market_state(roots=roots, equity=equity)


def _eur_contract_master() -> InstrumentsConfig:
    """A minimal instrument master whose only root settles in EUR."""
    return parse_instruments_config(
        {
            "version": 1,
            "defaults": {"currency": "USD"},
            "instruments": {
                "ES": {
                    "name": "E-mini S&P 500 (EUR test contract)",
                    "asset_class": "equity_index",
                    "sector": "equity",
                    "exchange": "CME",
                    "multiplier": 50.0,
                    "tick_size": 0.25,
                    "currency": "EUR",
                    "databento_symbol": "ES.FUT",
                    "ibkr_exchange": "CME",
                    "margins": {"initial": 15_000.0, "maintenance": 13_600.0},
                    "roll": {
                        "trigger": "calendar_days_before_fnd",
                        "days_before_fnd": 5,
                        "contract_months": [3, 6, 9, 12],
                    },
                }
            },
        }
    )


# ---------------------------------------------------------------------------
# Exact EUR sizing math
# ---------------------------------------------------------------------------


def test_eur_account_sizes_from_usd_converted_equity() -> None:
    """100,000 EUR at 1.20 is 120,000 USD of sizing equity, not 100,000 USD.

    2.5 x 120,000 = 300,000 USD target -> 1.25 ES raw -> FLOOR 1 ES (240,000)
    with a 60,000 USD residual -> 2.5 MES -> FLOOR 2 MES.
    """
    translator = _translator(
        rounding_mode=RoundingMode.FLOOR, account_currency="EUR", fx_rates={"EUR": 1.2}
    )
    state = _eur_state()
    targets = translator.translate(make_weight_vector({"ES": 2.5}), state)
    assert targets.quantities == {"ES": 1, "MES": 2}
    es = targets.get("ES")
    mes = targets.get("MES")
    assert es is not None
    assert mes is not None
    # Native marks are unchanged by FX normalisation (USD-settled contracts).
    assert es.price == 4800.0
    assert mes.price == 4800.0
    assert es.signed_notional == pytest.approx(ES_CONTRACT_NOTIONAL_USD)
    assert mes.signed_notional == pytest.approx(2 * MES_CONTRACT_NOTIONAL_USD)
    assert targets.total_initial_margin == pytest.approx(
        ES_INITIAL_MARGIN_USD + 2 * MES_INITIAL_MARGIN_USD
    )
    # The same weight on a 100,000 USD account would size ~1.04 ES and no micro leg.
    usd_targets = _translator(rounding_mode=RoundingMode.FLOOR).translate(
        make_weight_vector({"ES": 2.5}), _usd_state(equity=100_000.0)
    )
    assert usd_targets.quantities == {"ES": 1}
    assert usd_targets.quantities != targets.quantities


def test_eur_translation_matches_usd_equivalent_equity() -> None:
    """EUR translation equals USD translation with equity converted at the FX rate."""
    eur_translator = _translator(
        rounding_mode=RoundingMode.NEAREST, account_currency="EUR", fx_rates={"EUR": 1.2}
    )
    weights = make_weight_vector({"ES": 1.1, "NQ": 0.4, "CL": -0.2})
    eur_targets = eur_translator.translate(weights, _eur_state(roots=DEFAULT_ROOTS))
    usd_targets = _translator(rounding_mode=RoundingMode.NEAREST).translate(
        weights, _usd_state(equity=120_000.0, roots=DEFAULT_ROOTS)
    )
    assert eur_targets == usd_targets
    assert eur_targets.quantities == usd_targets.quantities


def test_per_call_equity_is_in_effective_account_currency() -> None:
    """An explicit ``equity=`` is local currency, exactly like ``state.equity``."""
    translator = _translator(
        rounding_mode=RoundingMode.FLOOR, account_currency="EUR", fx_rates={"EUR": 1.2}
    )
    state = _eur_state(equity=500_000.0)
    from_state = translator.translate(make_weight_vector({"ES": 0.3}), state)
    explicit = translator.translate(make_weight_vector({"ES": 0.3}), state, equity=100_000.0)
    expected = _translator(rounding_mode=RoundingMode.FLOOR).translate(
        make_weight_vector({"ES": 0.3}), _usd_state(equity=120_000.0)
    )
    assert explicit == expected
    assert from_state != explicit


def test_shorts_and_micro_routing_under_non_usd_account() -> None:
    translator = _translator(
        rounding_mode=RoundingMode.FLOOR, account_currency="EUR", fx_rates={"EUR": 1.2}
    )
    targets = translator.translate(make_weight_vector({"ES": -2.5}), _eur_state())
    assert targets.quantities == {"ES": -1, "MES": -2}
    es = targets.get("ES")
    mes = targets.get("MES")
    assert es is not None
    assert mes is not None
    assert es.price == 4800.0
    assert es.signed_notional < 0.0
    assert mes.signed_notional < 0.0
    # Margin utilisation is sign agnostic and still measured against USD equity.
    assert translator.margin_utilization(targets, 100_000.0) == pytest.approx(0.15)


# ---------------------------------------------------------------------------
# Margin utilisation in USD
# ---------------------------------------------------------------------------


def test_margin_utilization_uses_usd_equity_exactly() -> None:
    translator = _translator(
        rounding_mode=RoundingMode.FLOOR, account_currency="EUR", fx_rates={"EUR": 1.2}
    )
    targets = translator.translate(make_weight_vector({"ES": 2.5}), _eur_state())
    total_margin = ES_INITIAL_MARGIN_USD + 2 * MES_INITIAL_MARGIN_USD
    # 18,000 USD of margin against 120,000 USD of equity = 0.15 exactly.
    assert translator.margin_utilization(targets, 100_000.0) == pytest.approx(
        total_margin / 120_000.0
    )
    assert translator.margin_utilization(targets, 100_000.0) == pytest.approx(0.15)
    # Constructor-level FX is equivalent to the same explicit USD equity.
    assert translator.margin_utilization(targets, 100_000.0) == pytest.approx(
        _translator().margin_utilization(targets, 120_000.0)
    )
    # Per-call overrides work without a constructor-level currency.
    assert _translator().margin_utilization(
        targets, 100_000.0, account_currency="EUR", fx_rates={"EUR": 1.2}
    ) == pytest.approx(0.15)
    # USD accounts are unchanged: 18,000 / 100,000.
    assert _translator().margin_utilization(targets, 100_000.0) == pytest.approx(0.18)


def test_margin_ceiling_uses_usd_equity_for_non_usd_accounts() -> None:
    translator = _translator(account_currency="EUR", fx_rates={"EUR": 1.2})
    targets = translator.translate(make_weight_vector({"ES": 20.0}), _eur_state())
    utilization = translator.margin_utilization(targets, 100_000.0)
    assert utilization <= translator.max_margin_utilization + 1e-12
    assert targets.total_initial_margin <= 0.60 * 120_000.0
    # Identical to the USD translation at the converted equity (validates the budget).
    usd_targets = _translator().translate(
        make_weight_vector({"ES": 20.0}), _usd_state(equity=120_000.0)
    )
    assert targets == usd_targets


def test_resolve_account_equity_helper() -> None:
    translator = _translator(account_currency="EUR", fx_rates={"EUR": 1.2})
    state = _eur_state(equity=100_000.0)
    assert translator.resolve_account_equity(state) == ("EUR", 1.2, pytest.approx(120_000.0))
    currency, usd_per_unit, equity_usd = translator.resolve_account_equity(state, equity=50_000.0)
    assert currency == "EUR"
    assert usd_per_unit == pytest.approx(1.2)
    assert equity_usd == pytest.approx(60_000.0)
    # USD default: identity conversion.
    assert _translator().resolve_account_equity(_usd_state(equity=250_000.0)) == (
        "USD",
        1.0,
        pytest.approx(250_000.0),
    )
    with pytest.raises(ValueError, match="equity"):
        _translator().resolve_account_equity(make_market_state(roots=("ES",), equity=None))


# ---------------------------------------------------------------------------
# Override resolution order: per-call > constructor > MarketState > USD
# ---------------------------------------------------------------------------


def test_fx_resolution_priority_changes_sizing() -> None:
    """State 1.10 vs constructor 1.20 vs per-call 1.30 produce different integers."""
    state = make_market_state(
        roots=MICRO_PAIR_ROOTS,
        equity=100_000.0,
        account_currency="EUR",
        fx_rates={"EUR": 1.1},
    )
    weights = make_weight_vector({"ES": 2.5})
    # 1.10 -> 110,000 USD -> 275,000 target -> 1 ES + 35,000 residual / 1.458 MES
    state_only = _translator(rounding_mode=RoundingMode.FLOOR).translate(weights, state)
    assert state_only.quantities == {"ES": 1, "MES": 1}
    # 1.20 -> 120,000 USD -> 300,000 target -> 1 ES + 2.5 MES (floored to 2).
    constructor = _translator(
        rounding_mode=RoundingMode.FLOOR, account_currency="EUR", fx_rates={"EUR": 1.2}
    )
    assert constructor.translate(weights, state).quantities == {"ES": 1, "MES": 2}
    # 1.30 -> 130,000 USD -> 325,000 target -> 1 ES + 85,000 residual / 3.54 MES.
    assert constructor.translate(weights, state, fx_rates={"EUR": 1.3}).quantities == {
        "ES": 1,
        "MES": 3,
    }


def test_currency_resolution_priority() -> None:
    state = make_market_state(
        roots=("ES",),
        equity=100_000.0,
        account_currency="EUR",
        fx_rates={"EUR": 1.1, "GBP": 1.25},
    )
    constructor = _translator(account_currency="EUR", fx_rates={"EUR": 1.2})
    assert constructor.effective_account_currency(state) == "EUR"
    assert constructor.resolve_account_equity(state)[:2] == ("EUR", pytest.approx(1.2))
    # Per-call currency wins over the constructor, and per-call rates win over both.
    assert constructor.effective_account_currency(state, account_currency="gbp") == "GBP"
    assert constructor.resolve_account_equity(
        state, account_currency="GBP", fx_rates={"GBP": 1.3}
    ) == ("GBP", pytest.approx(1.3), pytest.approx(130_000.0))
    # State fields are used when neither override is configured.
    assert _translator().effective_account_currency(state) == "EUR"
    # USD is the final fallback.
    usd_state = make_market_state(roots=("ES",), equity=100_000.0)
    assert _translator().effective_account_currency(usd_state) == "USD"


def test_fx_rate_table_merges_with_per_call_winning() -> None:
    state = make_market_state(
        roots=("ES",),
        equity=100_000.0,
        account_currency="EUR",
        fx_rates={"EUR": 1.1, "GBP": 1.25},
    )
    constructor = _translator(fx_rates={"EUR": 1.2})
    assert dict(constructor.effective_fx_rates(state)) == {"EUR": 1.2, "GBP": 1.25}
    merged = constructor.effective_fx_rates(state, fx_rates={"EUR": 1.3, "JPY": 150.0})
    assert dict(merged) == {"EUR": 1.3, "GBP": 1.25, "JPY": 150.0}
    with pytest.raises(TypeError):
        merged["EUR"] = 1.4  # type: ignore[index]
    # USD is implicitly 1.0 and never needs a table entry.
    usd_state = make_market_state(roots=("ES",))
    assert dict(_translator().effective_fx_rates(usd_state)) == {}


# ---------------------------------------------------------------------------
# Fail-closed behaviour
# ---------------------------------------------------------------------------


def test_missing_account_fx_rate_fails_closed() -> None:
    translator = _translator()
    state = make_market_state(roots=("ES",), equity=100_000.0, account_currency="EUR")
    with pytest.raises(KeyError, match=r"fx_rates\['EUR'\] = USD per 1 EUR"):
        translator.translate(make_weight_vector({"ES": 1.0}), state)
    with pytest.raises(KeyError, match="no FX rate"):
        translator.resolve_account_equity(state)


def test_missing_constructor_currency_rate_fails_closed() -> None:
    translator = _translator(account_currency="GBP")
    state = make_market_state(roots=("ES",), equity=100_000.0)
    with pytest.raises(KeyError, match=r"fx_rates\['GBP'\]"):
        translator.translate(make_weight_vector({"ES": 1.0}), state)
    with pytest.raises(KeyError, match=r"fx_rates\['GBP'\]"):
        translator.margin_utilization(ContractTargetSet.empty(), 100_000.0)


def test_flat_weights_need_no_equity_or_fx() -> None:
    translator = _translator(account_currency="EUR")  # deliberately no EUR rate
    state = make_market_state(roots=("ES",), equity=None, account_currency="JPY", fx_rates={})
    empty = ContractTargetSet.empty()
    assert translator.translate(WeightVector.empty(), state) == empty
    assert translator.translate(make_weight_vector({"ES": 0.0, "NQ": 0.0}), state) == empty


def test_non_usd_contract_currency_requires_rate_and_keeps_native_price() -> None:
    master = _eur_contract_master()
    state = _usd_state()  # USD account, USD 1,000,000 equity
    weights = make_weight_vector({"ES": 0.30})
    # 0.30 x 1,000,000 = 300,000 USD target; one EUR contract is
    # 4,800 EUR -> 5,760 USD x 50 = 288,000 USD -> FLOOR 1 contract.
    priced = _translator(
        instruments=master, rounding_mode=RoundingMode.FLOOR, fx_rates={"EUR": 1.2}
    )
    targets = priced.translate(weights, state)
    assert targets.quantities == {"ES": 1}
    target = targets.get("ES")
    assert target is not None
    # Execution semantics stay native: the emitted mark is 4,800, not 5,760.
    assert target.price == 4800.0
    assert target.multiplier == 50.0
    # 0.24 x 1,000,000 = 240,000 USD fits exactly one USD contract but only 0.83
    # EUR contracts, so the same weight floors to a flat book.
    small = make_weight_vector({"ES": 0.24})
    assert priced.translate(small, state) == ContractTargetSet.empty()
    usd_master_translator = _translator(rounding_mode=RoundingMode.FLOOR)
    assert usd_master_translator.translate(small, state).quantities == {"ES": 1}
    # Fail closed when the EUR contract currency has no rate.
    unpriced = _translator(instruments=master, rounding_mode=RoundingMode.FLOOR)
    with pytest.raises(KeyError, match=r"fx_rates\['EUR'\] = USD per 1 EUR"):
        unpriced.translate(weights, state)


# ---------------------------------------------------------------------------
# USD regression: FX support must not change USD behaviour
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rounding_mode", list(RoundingMode))
@pytest.mark.parametrize("buffer", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("micro_routing", [True, False])
def test_usd_translation_is_unchanged_by_fx_configuration(
    rounding_mode: RoundingMode, buffer: float, micro_routing: bool
) -> None:
    state = make_market_state(roots=DEFAULT_ROOTS, equity=1_000_000.0)
    weights = make_weight_vector({"ES": 0.30, "NQ": -0.20, "CL": 0.15, "ZN": 0.10})
    baseline = _translator(
        rounding_mode=rounding_mode,
        rebalance_buffer_contracts=buffer,
        enable_micro_routing=micro_routing,
    )
    fx_configured = _translator(
        rounding_mode=rounding_mode,
        rebalance_buffer_contracts=buffer,
        enable_micro_routing=micro_routing,
        account_currency="USD",
        fx_rates={"EUR": 1.2, "GBP": 1.25},
    )
    baseline_targets = baseline.translate(weights, state)
    fx_targets = fx_configured.translate(weights, state)
    assert fx_targets == baseline_targets
    assert fx_targets.quantities == baseline_targets.quantities
    assert baseline.margin_utilization(baseline_targets, 1_000_000.0) == pytest.approx(
        fx_configured.margin_utilization(fx_targets, 1_000_000.0)
    )


@pytest.mark.parametrize(
    ("rounding_mode", "expected"),
    [
        (RoundingMode.FLOOR, {"ES": 1, "MES": 2}),
        (RoundingMode.NEAREST, {"ES": 1, "MES": 3}),
    ],
)
def test_usd_hand_computed_quantities_still_hold(
    rounding_mode: RoundingMode, expected: dict[str, int]
) -> None:
    """The documented pre-FX USD quantities for ES 0.30 at 1,000,000 equity."""
    translator = _translator(
        rounding_mode=rounding_mode, account_currency="USD", fx_rates={"EUR": 1.2}
    )
    targets = translator.translate(
        make_weight_vector({"ES": 0.30}), make_market_state(roots=MICRO_PAIR_ROOTS)
    )
    assert targets.quantities == expected


def test_from_config_defaults_to_no_fx_override() -> None:
    config = TranslatorConfig(rounding_mode=RoundingMode.FLOOR, max_margin_utilization=0.30)
    translator = ContractTranslator.from_config(config)
    assert translator.account_currency is None
    assert dict(translator.fx_rates) == {}
    assert isinstance(translator, ContractTranslatorProtocol)
    state = make_market_state(roots=MICRO_PAIR_ROOTS, equity=1_000_000.0)
    assert translator.translate(make_weight_vector({"ES": 0.30}), state).quantities == {
        "ES": 1,
        "MES": 2,
    }


# ---------------------------------------------------------------------------
# Constructor / per-call validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"account_currency": "EURO"}, ValueError, "3-letter"),
        ({"account_currency": ""}, ValueError, "3-letter"),
        ({"account_currency": "12"}, ValueError, "3-letter"),
        ({"account_currency": " eur "}, None, None),  # normalised, not rejected
        ({"fx_rates": {"EUR": 0.0}}, ValueError, "non-zero"),
        ({"fx_rates": {"EUR": -1.0}}, ValueError, "strictly positive"),
        ({"fx_rates": {"EUR": math.nan}}, ValueError, "finite"),
        ({"fx_rates": {"EUR": math.inf}}, ValueError, "finite"),
        ({"fx_rates": {"USD": 1.2}}, ValueError, "USD"),
        ({"fx_rates": {"EURO": 1.2}}, ValueError, "3-letter"),
    ],
)
def test_constructor_fx_validation(
    kwargs: dict[str, object], error: type[Exception] | None, message: str | None
) -> None:
    if error is None:
        assert _translator(**kwargs).account_currency == "EUR"
        return
    with pytest.raises(error, match=message):
        _translator(**kwargs)


def test_constructor_rejects_non_string_currency() -> None:
    with pytest.raises(TypeError, match="account_currency"):
        ContractTranslator(account_currency=123)  # type: ignore[arg-type]


def test_constructor_fx_fields_are_frozen() -> None:
    translator = ContractTranslator(account_currency="eur", fx_rates={"EUR": 1.2})
    assert translator.account_currency == "EUR"
    assert translator.fx_rates["EUR"] == pytest.approx(1.2)
    with pytest.raises(TypeError):
        translator.fx_rates["EUR"] = 1.3  # type: ignore[index]


def test_per_call_fx_overrides_are_validated() -> None:
    translator = _translator()
    state = make_market_state(roots=("ES",), equity=100_000.0)
    with pytest.raises(ValueError, match="3-letter"):
        translator.translate(make_weight_vector({"ES": 1.0}), state, account_currency="EURO")
    with pytest.raises(ValueError, match="strictly positive"):
        translator.translate(make_weight_vector({"ES": 1.0}), state, fx_rates={"EUR": -1.0})
    with pytest.raises(ValueError, match="USD"):
        translator.margin_utilization(ContractTargetSet.empty(), 100_000.0, fx_rates={"USD": 1.5})


def test_margin_utilization_rejects_bad_equity_before_fx_lookup() -> None:
    translator = _translator(account_currency="EUR")  # no EUR rate configured
    for bad_equity in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(ValueError, match="equity"):
            translator.margin_utilization(ContractTargetSet.empty(), bad_equity)


# ---------------------------------------------------------------------------
# Hypothesis properties: FX equivalence and USD invariant preservation
# ---------------------------------------------------------------------------


@settings(max_examples=40, deadline=None)
@given(
    equity=st.floats(
        min_value=10_000.0, max_value=2_000_000.0, allow_nan=False, allow_infinity=False
    ),
    eur_per_usd=st.floats(min_value=0.5, max_value=2.0, allow_nan=False, allow_infinity=False),
    weights=st.dictionaries(
        keys=st.sampled_from(["ES", "NQ", "CL", "GC", "ZN"]),
        values=st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=5,
    ),
    rounding_mode=st.sampled_from(list(RoundingMode)),
)
def test_property_non_usd_equals_usd_translation_with_converted_equity(
    equity: float,
    eur_per_usd: float,
    weights: dict[str, float],
    rounding_mode: RoundingMode,
) -> None:
    """Property: EUR translation(w, equity) == USD translation(w, equity * rate).

    This is the core FX-normalisation invariant: converting the account equity to
    USD once, up front, must be equivalent to running the whole pre-FX translator
    on a USD account funded with that converted equity.
    """
    eur_state = make_market_state(
        roots=DEFAULT_ROOTS,
        equity=equity,
        account_currency="EUR",
        fx_rates={"EUR": eur_per_usd},
    )
    usd_state = make_market_state(roots=DEFAULT_ROOTS, equity=equity * eur_per_usd)
    eur_translator = ContractTranslator(
        rounding_mode=rounding_mode,
        account_currency="EUR",
        fx_rates={"EUR": eur_per_usd},
    )
    usd_translator = ContractTranslator(rounding_mode=rounding_mode)
    eur_targets = eur_translator.translate(WeightVector(weights), eur_state)
    usd_targets = usd_translator.translate(WeightVector(weights), usd_state)
    assert eur_targets == usd_targets
    assert eur_targets.quantities == usd_targets.quantities
    # The margin ceiling must be enforced against the same USD equity.
    assert eur_translator.margin_utilization(eur_targets, equity) == pytest.approx(
        usd_translator.margin_utilization(usd_targets, equity * eur_per_usd)
    )


@settings(max_examples=40, deadline=None)
@given(
    equity=st.floats(
        min_value=10_000.0, max_value=2_000_000.0, allow_nan=False, allow_infinity=False
    ),
    weights=st.dictionaries(
        keys=st.sampled_from(["ES", "MES", "NQ", "MNQ", "CL", "MCL", "GC", "MGC", "ZN"]),
        values=st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=5,
    ),
    rounding_mode=st.sampled_from(list(RoundingMode)),
)
def test_property_usd_margin_ceiling_survives_fx_configuration(
    equity: float, weights: dict[str, float], rounding_mode: RoundingMode
) -> None:
    """Property: configuring unused FX rates/currency never breaks the USD invariants."""
    state = make_market_state(roots=DEFAULT_ROOTS, equity=equity)
    translator = ContractTranslator(
        rounding_mode=rounding_mode,
        account_currency="USD",
        fx_rates={"EUR": 1.2, "GBP": 1.25, "JPY": 150.0},
    )
    targets = translator.translate(WeightVector(weights), state)
    utilization = translator.margin_utilization(targets, equity)
    assert utilization <= translator.max_margin_utilization + 1e-9
    assert translator.resolve_account_equity(state) == ("USD", 1.0, pytest.approx(equity))
