"""Deterministic test-data factories shared by the whole test suite.

Every fixture here is fully deterministic (seeded) and offline: unit tests must
never touch Databento, IBKR or the network. Import these factories directly in
tests (``from tests.fixtures.market import make_market_state``) or use the
pytest fixtures registered in ``tests/conftest.py``.
"""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, timedelta

import numpy as np
import polars as pl
from configs.instruments import InstrumentsConfig, load_instruments
from numpy.typing import NDArray
from src.core.types import (
    AccountSnapshot,
    AssetClass,
    Bar,
    ContractTarget,
    ContractTargetSet,
    InstrumentId,
    InstrumentSnapshot,
    MarketState,
    OrderEvent,
    OrderSide,
    OrderType,
    RiskAssessment,
    WeightVector,
)
from src.data.schema import ensure_bars_schema

#: Fixed, deterministic point-in-time used by default (a US close, 21:00 UTC).
FIXED_TIMESTAMP: datetime = datetime(2024, 1, 2, 21, 0, tzinfo=UTC)

#: Reference mark prices for the instrument master's roots.
DEFAULT_PRICES: dict[InstrumentId, float] = {
    "ES": 4800.0,
    "MES": 4800.0,
    "NQ": 16800.0,
    "MNQ": 16800.0,
    "YM": 38000.0,
    "RTY": 2000.0,
    "CL": 72.5,
    "MCL": 72.5,
    "GC": 2050.0,
    "MGC": 2050.0,
    "ZN": 111.5,
    "ZT": 102.75,
    "6E": 1.095,
}

#: Default synthetic universe for pipeline / risk / backtest tests.
DEFAULT_ROOTS: tuple[InstrumentId, ...] = ("ES", "NQ", "CL", "GC", "ZN")

#: Annualised volatility per risk-cluster sector (used for synthetic covariance).
SECTOR_ANNUAL_VOL: dict[str, float] = {
    "equity": 0.18,
    "rates": 0.06,
    "energy": 0.35,
    "metals": 0.15,
    "fx": 0.08,
}

_SECTOR_CORRELATION = 0.80
_CROSS_SECTOR_CORRELATION = 0.10

#: Default expiration used for synthetic bars/contracts.
DEFAULT_EXPIRATION: date = date(2024, 3, 15)


def _master(instrument_master: InstrumentsConfig | None) -> InstrumentsConfig:
    return instrument_master if instrument_master is not None else load_instruments()


def default_price(root: InstrumentId) -> float:
    """Reference mark price for ``root`` (deterministic fallback 1000.0)."""
    return DEFAULT_PRICES.get(root.upper(), 1000.0)


def make_snapshot(
    root: InstrumentId,
    *,
    instrument_master: InstrumentsConfig | None = None,
    price: float | None = None,
    volume: float = 250_000.0,
    open_interest: float = 1_500_000.0,
    features: dict[str, float] | None = None,
    expiration: date | None = DEFAULT_EXPIRATION,
    bid_ask_ticks: float = 1.0,
) -> InstrumentSnapshot:
    """Build one :class:`InstrumentSnapshot` for ``root``."""
    master = _master(instrument_master)
    spec = master.get(root)
    mark = price if price is not None else default_price(spec.root)
    half_spread = 0.5 * spec.tick_size * bid_ask_ticks
    return InstrumentSnapshot(
        instrument_id=spec.root,
        root_symbol=spec.root,
        price=mark,
        asset_class=spec.asset_class,
        expiration=expiration,
        bid=mark - half_spread,
        ask=mark + half_spread,
        volume=volume,
        open_interest=open_interest,
        settlement=mark,
        features=features or {},
    )


def _aggregate_returns(returns: NDArray[np.float64], window: int) -> float:
    if returns.size == 0:
        return 0.0
    used = returns[-min(window, returns.size) :]
    return float(np.sum(used))


def _features_for(
    returns: NDArray[np.float64],
    root: InstrumentId,
    spec_sector: str,
    *,
    index: int,
    n_roots: int,
) -> dict[str, float]:
    """Deterministic feature dictionary matching ``src.data.schema`` names."""
    daily_vol = float(np.std(returns, ddof=1)) if returns.size > 2 else 0.01
    annual_vol = daily_vol * math.sqrt(252.0)
    # Term structure: equity index mildly negative carry (contango), energy and
    # rates positive carry in backwardation-like fixtures; deterministic by root.
    sector_bias = {"equity": -0.004, "rates": 0.002, "energy": 0.010, "metals": 0.001, "fx": 0.0}
    term_slope = sector_bias.get(spec_sector, 0.0) + 0.0005 * math.sin(index + 1.0)
    carry = -term_slope * (365.0 / 30.0)
    tsmom_21 = _aggregate_returns(returns, 21)
    tsmom_126 = _aggregate_returns(returns, 126)
    return {
        "ret_1d": float(returns[-1]) if returns.size else 0.0,
        "ret_5d": _aggregate_returns(returns, 5),
        "ret_21d": tsmom_21,
        "ret_63d": _aggregate_returns(returns, 63),
        "ret_126d": tsmom_126,
        "ret_252d": _aggregate_returns(returns, 252),
        "vol_20d": float(np.std(returns[-20:], ddof=1))
        if returns.size > 21
        else annual_vol / math.sqrt(252),
        "vol_60d": annual_vol / math.sqrt(252),
        "atr": max(annual_vol / math.sqrt(252) * float(DEFAULT_PRICES.get(root, 1000.0)), 1e-6),
        "garman_klass_vol": annual_vol,
        "macd": tsmom_21 * 0.5,
        "macd_signal": tsmom_21 * 0.4,
        "rsi": 50.0 + 10.0 * math.tanh(tsmom_21 * 100.0),
        "cci": 100.0 * math.tanh(tsmom_21 * 50.0),
        "adx": 20.0 + 15.0 * abs(math.tanh(tsmom_126 * 20.0)),
        "bb_upper": 1.02,
        "bb_lower": 0.98,
        "bb_width": 0.04,
        "term_slope": term_slope,
        "carry_annualized": carry,
        "volume_oi_ratio": 0.15 + 0.01 * index,
        "ofi": 0.01 * (index - (n_roots - 1) / 2.0),
        "turbulence": 0.5 + 0.1 * (index % 3),
        "kama": 1.0 + tsmom_21,
        "tsmom_21": tsmom_21,
        "tsmom_126": tsmom_126,
    }


def make_returns(
    n_roots: int,
    *,
    length: int = 260,
    seed: int = 7,
    daily_vol: float = 0.01,
    drift: float = 0.0002,
) -> NDArray[np.float64]:
    """Synthetic daily return matrix ``(length, n_roots)``."""
    rng = np.random.default_rng(seed)
    return rng.normal(drift, daily_vol, size=(length, n_roots))


def make_covariance(
    roots: tuple[InstrumentId, ...],
    *,
    instrument_master: InstrumentsConfig | None = None,
) -> tuple[NDArray[np.float64], tuple[InstrumentId, ...]]:
    """Annualised synthetic covariance matrix with sector-clustered correlation."""
    master = _master(instrument_master)
    sectors = [master.get(root).sector for root in roots]
    vols = np.array([SECTOR_ANNUAL_VOL.get(sector, 0.15) for sector in sectors], dtype=np.float64)
    correlations = np.full((len(roots), len(roots)), _CROSS_SECTOR_CORRELATION, dtype=np.float64)
    for i, left in enumerate(sectors):
        correlations[i, i] = 1.0
        for j, right in enumerate(sectors):
            if i != j and left == right:
                correlations[i, j] = _SECTOR_CORRELATION
    covariance = correlations * np.outer(vols, vols)
    return covariance, roots


def make_market_state(
    roots: tuple[InstrumentId, ...] | None = None,
    *,
    timestamp: datetime = FIXED_TIMESTAMP,
    equity: float | None = 1_000_000.0,
    prices: dict[InstrumentId, float] | None = None,
    features: dict[InstrumentId, dict[str, float]] | None = None,
    instrument_master: InstrumentsConfig | None = None,
    with_covariance: bool = True,
    with_returns: bool = True,
    returns_length: int = 260,
    seed: int = 7,
    regime: str | None = "neutral",
    metadata: dict[str, object] | None = None,
) -> MarketState:
    """Build a complete synthetic :class:`MarketState` for tests."""
    master = _master(instrument_master)
    selected = roots or DEFAULT_ROOTS
    price_overrides = prices or {}
    feature_overrides = features or {}
    all_returns = make_returns(len(selected), length=returns_length, seed=seed)
    snapshots: list[InstrumentSnapshot] = []
    returns_map: dict[InstrumentId, NDArray[np.float64]] = {}
    for index, root in enumerate(selected):
        spec = master.get(root)
        root_returns = all_returns[:, index]
        root_features = _features_for(
            root_returns, spec.root, spec.sector, index=index, n_roots=len(selected)
        )
        root_features.update(feature_overrides.get(spec.root, {}))
        returns_map[spec.root] = root_returns
        snapshots.append(
            make_snapshot(
                spec.root,
                instrument_master=master,
                price=price_overrides.get(spec.root) or default_price(spec.root),
                features=root_features,
            )
        )
    covariance: NDArray[np.float64] | None = None
    covariance_instruments: tuple[InstrumentId, ...] = ()
    if with_covariance:
        covariance, covariance_instruments = make_covariance(selected, instrument_master=master)
    state_metadata: dict[str, object] = {
        "vix_term_slope": 0.02,
        "yield_curve_slope_2s10s": 0.35,
        "dxy_return_5d": -0.01,
        "macro_lockout": False,
    }
    if metadata:
        state_metadata.update(metadata)
    return MarketState(
        timestamp=timestamp,
        instruments={snapshot.instrument_id: snapshot for snapshot in snapshots},
        equity=equity,
        covariance=covariance,
        covariance_instruments=covariance_instruments,
        returns=returns_map if with_returns else {},
        regime=regime,
        metadata=state_metadata,  # type: ignore[arg-type]
    )


def make_weight_vector(mapping: dict[InstrumentId, float]) -> WeightVector:
    """Build a :class:`WeightVector` from a plain mapping."""
    return WeightVector.from_mapping(mapping)


def make_contract_target(
    root: InstrumentId,
    quantity: int,
    *,
    price: float | None = None,
    instrument_master: InstrumentsConfig | None = None,
    expiration: date | None = DEFAULT_EXPIRATION,
) -> ContractTarget:
    """Build one :class:`ContractTarget` using the real instrument spec."""
    master = _master(instrument_master)
    spec = master.get(root)
    mark = price if price is not None else default_price(spec.root)
    return ContractTarget(
        instrument_id=spec.root,
        quantity=quantity,
        price=mark,
        multiplier=spec.multiplier,
        expiration=expiration,
        initial_margin=spec.margins.initial,
        maintenance_margin=spec.margins.maintenance,
        contract_symbol=f"{spec.root}_TEST",
        asset_class=spec.asset_class,
    )


def make_contract_targets(
    quantities: dict[InstrumentId, int],
    *,
    instrument_master: InstrumentsConfig | None = None,
    expiration: date | None = DEFAULT_EXPIRATION,
) -> ContractTargetSet:
    """Build a :class:`ContractTargetSet` from ``root -> quantity``."""
    return ContractTargetSet(
        tuple(
            make_contract_target(
                root,
                quantity,
                instrument_master=instrument_master,
                expiration=expiration,
            )
            for root, quantity in quantities.items()
            if quantity != 0
        )
    )


def make_bars_frame(
    roots: tuple[InstrumentId, ...] | None = None,
    *,
    periods: int = 252,
    start: datetime = FIXED_TIMESTAMP,
    interval_minutes: int = 1440,
    seed: int = 7,
    instrument_master: InstrumentsConfig | None = None,
    expiration: date = DEFAULT_EXPIRATION,
) -> pl.DataFrame:
    """Build a deterministic, schema-valid daily OHLCV frame for ``roots``.

    One contract per root is generated with a geometric random walk; the frame
    is normalised through the canonical :mod:`src.data.schema` validator.
    """
    master = _master(instrument_master)
    selected = roots or DEFAULT_ROOTS
    rng = np.random.default_rng(seed)
    rows: dict[str, list[object]] = {
        "timestamp_utc": [],
        "instrument_id": [],
        "root_symbol": [],
        "expiration_date": [],
        "open": [],
        "high": [],
        "low": [],
        "close": [],
        "volume": [],
        "open_interest": [],
        "settlement": [],
    }
    for index, root in enumerate(selected):
        spec = master.get(root)
        price = default_price(spec.root)
        daily_vol = SECTOR_ANNUAL_VOL.get(spec.sector, 0.15) / math.sqrt(252.0)
        returns = rng.normal(0.0002, daily_vol, size=periods)
        closes = price * np.cumprod(1.0 + returns)
        for step in range(periods):
            close = float(closes[step])
            open_price = float(closes[step - 1]) if step > 0 else price
            high = max(open_price, close) * (1.0 + abs(rng.normal(0.0, daily_vol * 0.5)))
            low = min(open_price, close) * (1.0 - abs(rng.normal(0.0, daily_vol * 0.5)))
            volume = float(abs(rng.normal(250_000.0, 25_000.0)))
            timestamp = start + timedelta(minutes=interval_minutes * step)
            rows["timestamp_utc"].append(timestamp)
            rows["instrument_id"].append(f"{spec.root}_TEST")
            rows["root_symbol"].append(spec.root)
            rows["expiration_date"].append(expiration)
            rows["open"].append(open_price)
            rows["high"].append(high)
            rows["low"].append(low)
            rows["close"].append(close)
            rows["volume"].append(volume)
            rows["open_interest"].append(float(1_000_000 + 10_000 * index + step))
            rows["settlement"].append(close)
    frame = pl.DataFrame(rows)
    return ensure_bars_schema(frame, context="make_bars_frame")


def make_bars(
    roots: tuple[InstrumentId, ...] | None = None,
    *,
    periods: int = 60,
    start: datetime = FIXED_TIMESTAMP,
) -> list[Bar]:
    """Build a list of :class:`Bar` objects (single contract per root)."""
    frame = make_bars_frame(roots, periods=periods, start=start)
    bars: list[Bar] = []
    for row in frame.iter_rows(named=True):
        bars.append(
            Bar(
                timestamp=row["timestamp_utc"],
                instrument_id=row["root_symbol"],
                root_symbol=row["root_symbol"],
                expiration=row["expiration_date"],
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
                open_interest=row["open_interest"],
                settlement=row["settlement"],
            )
        )
    return bars


def make_risk_assessment(
    root: InstrumentId,
    *,
    semantic_risk_score: float = 2.0,
    confidence: float = 0.9,
    sentiment_score: float = 3.0,
    timestamp: datetime = FIXED_TIMESTAMP,
    reasoning_summary: str = "deterministic fixture",
    source: str = "fixture",
) -> RiskAssessment:
    """Build a deterministic LLM risk assessment fixture."""
    return RiskAssessment(
        instrument_id=root,
        timestamp=timestamp,
        sentiment_score=sentiment_score,
        semantic_risk_score=semantic_risk_score,
        confidence=confidence,
        reasoning_summary=reasoning_summary,
        source=source,
    )


def make_order_event(
    root: InstrumentId,
    quantity: int,
    side: OrderSide = OrderSide.BUY,
    *,
    order_type: OrderType = OrderType.MARKET,
    limit_price: float | None = None,
    timestamp: datetime = FIXED_TIMESTAMP,
    order_id: str = "test-order-1",
) -> OrderEvent:
    """Build an :class:`OrderEvent` fixture."""
    return OrderEvent(
        order_id=order_id,
        timestamp=timestamp,
        instrument_id=root,
        side=side,
        quantity=quantity,
        order_type=order_type,
        limit_price=limit_price,
    )


def make_account_snapshot(
    *,
    net_liquidation: float = 1_000_000.0,
    initial_margin_used: float = 120_000.0,
    timestamp: datetime = FIXED_TIMESTAMP,
) -> AccountSnapshot:
    """Build an :class:`AccountSnapshot` fixture."""
    return AccountSnapshot(
        timestamp=timestamp,
        net_liquidation=net_liquidation,
        cash_balance=net_liquidation,
        buying_power=net_liquidation * 2.0,
        initial_margin_used=initial_margin_used,
        maintenance_margin_used=initial_margin_used * 0.9,
        excess_liquidity=net_liquidation - initial_margin_used,
    )


def asset_class_of(root: InstrumentId) -> AssetClass:
    """Asset class for ``root`` from the instrument master."""
    return load_instruments().get(root).asset_class
