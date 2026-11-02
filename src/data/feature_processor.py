"""FinRL-X feature engineering for futures: technicals, term structure, OFI, macro.

The processor consumes canonical bars (raw multi-expiration bars *and* the
continuous-series frame produced by :mod:`src.data.continuous_roller` are both
accepted) and returns the same canonical bar schema plus the canonical feature
columns declared in :mod:`src.data.schema`.

Features
--------
Technical (FinRL-X style): ``macd``/``macd_signal``, ``rsi``, ``cci``, ``adx``,
``bb_upper``/``bb_lower``/``bb_width``, ``atr``, ``garman_klass_vol``, ``kama``
and the TSMOM/return/volatility families.

Futures-specific: ``term_slope = (F2 - F1) / F1``, ``carry_annualized``
(annualised roll yield, sign convention ``-term_slope * 365 / days``),
``volume_oi_ratio`` and the order-flow imbalance ``ofi`` built from ``tbbo``
quotes.

Cross-asset macro regime inputs (from an optional exogenous macro frame):
``macro_vix_term_slope``, ``macro_yield_curve_slope``, ``macro_dxy_return_5d``,
``macro_dxy_trend_21d``.

Structural break detector: the FinRL Mahalanobis ``turbulence`` index computed
from the *strictly trailing* cross-sectional return covariance.

Causality
---------
Every window is strictly trailing: return/volatility windows use ``shift`` and
rolling reducers that require a full historical window, EWM/Wilder smoothers are
one-sided, ``term_slope`` uses simultaneous quotes, and the turbulence mean /
covariance are estimated from returns strictly *before* ``t``. Changing any
future row can therefore never change an earlier feature value.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from datetime import date, datetime
from typing import Any

import numpy as np
import polars as pl
from configs.instruments import InstrumentsConfig
from numpy.typing import NDArray

from src.data import schema as s

__all__ = ["FeatureProcessor"]

_FloatArray = NDArray[np.float64]
_TWO_LN2_MINUS_ONE = 2.0 * math.log(2.0) - 1.0


# ---------------------------------------------------------------------------
# Small causal array helpers (all vectorised / one-sided)
# ---------------------------------------------------------------------------


def _float_array(series: pl.Series) -> _FloatArray:
    """Cast a Polars series to a float64 numpy array (nulls become ``NaN``)."""
    return np.asarray(series.cast(pl.Float64).to_numpy(), dtype=np.float64)


def _shift(values: _FloatArray, periods: int) -> _FloatArray:
    """Shift right by ``periods`` filling the head with ``NaN``."""
    if periods < 0:
        raise ValueError("periods must be non-negative")
    out = np.full(values.shape, np.nan, dtype=np.float64)
    if periods == 0:
        out[:] = values
    elif periods < values.size:
        out[periods:] = values[:-periods]
    return out


def _safe_div(numerator: _FloatArray | float, denominator: _FloatArray | float) -> _FloatArray:
    """Elementwise divide returning ``NaN`` where the denominator is 0/NaN."""
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    out = np.full(np.broadcast(num, den).shape, np.nan, dtype=np.float64)
    np.divide(num, den, out=out, where=np.isfinite(den) & (den != 0.0))
    out[~np.isfinite(num)] = np.nan
    return out


def _rolling_apply(
    values: _FloatArray, window: int, reducer: Callable[[_FloatArray], _FloatArray]
) -> _FloatArray:
    """Apply ``reducer`` to every *complete* trailing window (strictly causal)."""
    if window < 1:
        raise ValueError("window must be >= 1")
    out = np.full(values.shape, np.nan, dtype=np.float64)
    if values.size < window:
        return out
    windows = np.lib.stride_tricks.sliding_window_view(values, window)
    finite = np.isfinite(windows).all(axis=-1)
    reduced = reducer(windows)
    out[window - 1 :] = np.where(finite, reduced, np.nan)
    return out


def _rolling_mean(values: _FloatArray, window: int) -> _FloatArray:
    return _rolling_apply(values, window, lambda windows: np.mean(windows, axis=-1))


def _rolling_std(values: _FloatArray, window: int) -> _FloatArray:
    return _rolling_apply(values, window, lambda windows: np.std(windows, axis=-1, ddof=1))


def _ewm(values: _FloatArray, alpha: float) -> _FloatArray:
    """One-sided exponentially weighted mean (``adjust=False`` semantics)."""
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be within (0, 1]")
    out = np.full(values.shape, np.nan, dtype=np.float64)
    valid = np.flatnonzero(np.isfinite(values))
    if valid.size == 0:
        return out
    start = int(valid[0])
    out[start] = values[start]
    for index in range(start + 1, values.size):
        current = values[index]
        out[index] = (
            out[index - 1]
            if not np.isfinite(current)
            else ((1.0 - alpha) * out[index - 1] + alpha * current)
        )
    return out


def _ema(values: _FloatArray, span: int) -> _FloatArray:
    """Standard EMA (``span`` periods) implemented as one-sided EWM."""
    return _ewm(values, 2.0 / (span + 1.0))


def _kama(close: _FloatArray, window: int, fast: int, slow: int) -> _FloatArray:
    """Kaufman Adaptive Moving Average (strictly causal loop)."""
    size = close.size
    out = np.full(size, np.nan, dtype=np.float64)
    if size < window + 1:
        return out
    absolute_change = np.abs(close - _shift(close, window))
    one_step_change = np.abs(close - _shift(close, 1))
    one_step_change[0] = 0.0
    volatility = _rolling_apply(one_step_change, window, lambda windows: np.sum(windows, axis=-1))
    efficiency = _safe_div(absolute_change, volatility)
    fast_constant = 2.0 / (fast + 1.0)
    slow_constant = 2.0 / (slow + 1.0)
    smoothing = (efficiency * (fast_constant - slow_constant) + slow_constant) ** 2
    out[window - 1] = close[window - 1]
    for index in range(window, size):
        if not np.isfinite(smoothing[index]):
            out[index] = out[index - 1]
            continue
        out[index] = out[index - 1] + smoothing[index] * (close[index] - out[index - 1])
    return out


# ---------------------------------------------------------------------------
# Feature processor
# ---------------------------------------------------------------------------


class FeatureProcessor:
    """Compute the canonical FinRL-X futures feature frame.

    Parameters
    ----------
    instrument_master:
        Optional instrument master. Only used to resolve asset classes for
        diagnostics; the processor is root-agnostic.
    turbulence_window, volatility_window, long_volatility_window, atr_window,
    rsi_window, cci_window, adx_window, bollinger_window, kama_window,
    slow_momentum_window, annualization_factor:
        Windows of the corresponding features (all strictly trailing).
    """

    def __init__(
        self,
        *,
        instrument_master: InstrumentsConfig | None = None,
        turbulence_window: int = 252,
        volatility_window: int = 20,
        long_volatility_window: int = 60,
        atr_window: int = 14,
        rsi_window: int = 14,
        cci_window: int = 20,
        adx_window: int = 14,
        bollinger_window: int = 20,
        kama_window: int = 10,
        slow_momentum_window: int = 126,
        annualization_factor: int = 252,
    ) -> None:
        self._master = instrument_master
        self._turbulence_window = turbulence_window
        self._volatility_window = volatility_window
        self._long_volatility_window = long_volatility_window
        self._atr_window = atr_window
        self._rsi_window = rsi_window
        self._cci_window = cci_window
        self._adx_window = adx_window
        self._bollinger_window = bollinger_window
        self._kama_window = kama_window
        self._slow_momentum_window = slow_momentum_window
        self._annualization_factor = annualization_factor
        for name in (
            "turbulence_window",
            "volatility_window",
            "long_volatility_window",
            "atr_window",
            "rsi_window",
            "cci_window",
            "adx_window",
            "bollinger_window",
            "kama_window",
            "slow_momentum_window",
            "annualization_factor",
        ):
            if int(getattr(self, f"_{name}")) < 1:
                raise ValueError(f"{name} must be >= 1")

    # -- public API ------------------------------------------------------------

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Canonical feature names emitted by :meth:`compute`."""
        return s.ALL_FEATURES

    def compute(
        self,
        bars: pl.DataFrame,
        *,
        exogenous: pl.DataFrame | None = None,
        quotes: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Return canonical bars plus every feature column.

        ``bars`` may be raw multi-expiration canonical bars or the continuous
        frame produced by :class:`~src.data.continuous_roller.ContinuousContractBuilder`.
        ``exogenous`` supplies cross-asset macro inputs and ``quotes`` a
        ``tbbo`` quote frame used for the order-flow-imbalance feature.
        """
        frame = self._normalise_input(bars)
        front = self._front_frame(frame)
        frames = [self._root_features(group) for _root, group in self._group_by_root(front)]
        combined = pl.concat(frames, how="vertical_relaxed").sort(
            [s.TIMESTAMP_COLUMN, s.ROOT_SYMBOL_COLUMN]
        )
        combined = self._add_turbulence(combined)
        combined = self._add_ofi(combined, quotes)
        return self._add_macro(combined, exogenous)

    def compute_order_flow_imbalance(self, quotes: pl.DataFrame) -> pl.DataFrame:
        """Order-flow imbalance per top-of-book quote (Cont et al. convention).

        ``OFI_t = bid_sz_t * 1{b_t >= b_{t-1}} - bid_sz_{t-1} * 1{b_t <= b_{t-1}}
        - ask_sz_t * 1{a_t <= a_{t-1}} + ask_sz_{t-1} * 1{a_t >= a_{t-1}}``

        Positive OFI is net buy-side pressure. Returns one row per quote with
        ``timestamp_utc``, ``root_symbol`` and ``ofi`` (the first quote of each
        root is ``NaN`` because it has no predecessor).
        """
        frame = s.ensure_quote_schema(quotes, context="ofi-quotes")
        outputs: list[pl.DataFrame] = []
        for key, group in frame.partition_by(s.ROOT_SYMBOL_COLUMN, as_dict=True).items():
            root = str(key[0] if isinstance(key, tuple) else key)
            ordered = self._ensure_sorted(group, [s.ROOT_SYMBOL_COLUMN, s.TIMESTAMP_COLUMN])
            bid = _float_array(ordered[s.BID_PRICE_COLUMN])
            ask = _float_array(ordered[s.ASK_PRICE_COLUMN])
            bid_size = _float_array(ordered[s.BID_SIZE_COLUMN])
            ask_size = _float_array(ordered[s.ASK_SIZE_COLUMN])
            previous_bid = _shift(bid, 1)
            previous_ask = _shift(ask, 1)
            previous_bid_size = _shift(bid_size, 1)
            previous_ask_size = _shift(ask_size, 1)
            bid_flow = bid_size * (bid >= previous_bid) - previous_bid_size * (bid <= previous_bid)
            ask_flow = ask_size * (ask <= previous_ask) - previous_ask_size * (ask >= previous_ask)
            ofi = bid_flow - ask_flow
            ofi[~np.isfinite(previous_bid)] = np.nan
            outputs.append(
                pl.DataFrame(
                    {
                        s.TIMESTAMP_COLUMN: ordered[s.TIMESTAMP_COLUMN].to_list(),
                        s.ROOT_SYMBOL_COLUMN: [root] * ordered.height,
                        s.OFI: ofi.tolist(),
                    }
                )
            )
        if not outputs:
            return pl.DataFrame(
                {
                    s.TIMESTAMP_COLUMN: [],
                    s.ROOT_SYMBOL_COLUMN: [],
                    s.OFI: [],
                },
                schema={
                    s.TIMESTAMP_COLUMN: pl.Datetime("us", "UTC"),
                    s.ROOT_SYMBOL_COLUMN: pl.Utf8,
                    s.OFI: pl.Float64,
                },
            )
        return pl.concat(outputs, how="vertical_relaxed").sort(
            [s.TIMESTAMP_COLUMN, s.ROOT_SYMBOL_COLUMN]
        )

    # -- input normalisation ---------------------------------------------------

    @staticmethod
    def _ensure_sorted(frame: pl.DataFrame, columns: Sequence[str]) -> pl.DataFrame:
        """Sort by ``columns`` when they exist (stable, cheap no-op when sorted)."""
        keys = [column for column in columns if column in frame.columns]
        return frame.sort(keys) if keys else frame

    def _normalise_input(self, bars: pl.DataFrame) -> pl.DataFrame:
        """Accept raw bars or continuous frames and return canonical bars."""
        if not isinstance(bars, pl.DataFrame):
            raise TypeError(f"bars must be a polars DataFrame, got {type(bars)!r}")
        if bars.is_empty():
            raise ValueError("FeatureProcessor.compute received an empty frame")
        frame = s.normalise_timestamp_column(bars, s.TIMESTAMP_COLUMN)
        if s.ROOT_SYMBOL_COLUMN not in frame.columns:
            raise s.SchemaError(
                f"feature input needs a {s.ROOT_SYMBOL_COLUMN!r} column; got {frame.columns}"
            )
        if s.INSTRUMENT_ID_COLUMN not in frame.columns:
            source = "front_symbol" if "front_symbol" in frame.columns else s.ROOT_SYMBOL_COLUMN
            frame = frame.with_columns(pl.col(source).alias(s.INSTRUMENT_ID_COLUMN))
        if s.EXPIRATION_COLUMN not in frame.columns:
            if "front_expiration" in frame.columns:
                frame = frame.with_columns(pl.col("front_expiration").alias(s.EXPIRATION_COLUMN))
            else:
                raise s.SchemaError(
                    "feature input needs an expiration_date (bars) or front_expiration "
                    "(continuous series) column"
                )
        if s.OPEN_INTEREST_COLUMN not in frame.columns:
            frame = frame.with_columns(pl.lit(0.0, dtype=pl.Float64).alias(s.OPEN_INTEREST_COLUMN))
        if s.SETTLEMENT_COLUMN not in frame.columns:
            frame = frame.with_columns(pl.lit(None, dtype=pl.Float64).alias(s.SETTLEMENT_COLUMN))
        return s.ensure_bars_schema(frame, context="feature-input")

    @staticmethod
    def _group_by_root(frame: pl.DataFrame) -> list[tuple[str, pl.DataFrame]]:
        """Deterministically group a frame by root symbol."""
        grouped: list[tuple[str, pl.DataFrame]] = []
        for key, group in (
            frame.sort(s.ROOT_SYMBOL_COLUMN)
            .partition_by(s.ROOT_SYMBOL_COLUMN, as_dict=True)
            .items()
        ):
            root = str(key[0] if isinstance(key, tuple) else key)
            grouped.append((root, group))
        return grouped

    def _front_frame(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Collapse multi-expiration bars to one front-contract row per root/time."""
        records: list[dict[str, Any]] = []
        for _root, group in self._group_by_root(frame):
            ordered = group.sort([s.TIMESTAMP_COLUMN, s.EXPIRATION_COLUMN])
            by_expiration: dict[date, dict[datetime, dict[str, Any]]] = {}
            for record in ordered.iter_rows(named=True):
                timestamp = record[s.TIMESTAMP_COLUMN]
                expiration = record[s.EXPIRATION_COLUMN]
                if not isinstance(timestamp, datetime) or not isinstance(expiration, date):
                    raise s.SchemaError("feature input has invalid timestamp/expiration dtypes")
                by_expiration.setdefault(expiration, {})[timestamp] = record
            expirations = sorted(by_expiration)
            timestamps = sorted(
                {timestamp for rows in by_expiration.values() for timestamp in rows}
            )
            for timestamp in timestamps:
                available = [exp for exp in expirations if timestamp in by_expiration[exp]]
                front_expiration = available[0]
                front = by_expiration[front_expiration][timestamp]
                next_close = None
                next_expiration: date | None = None
                if len(available) > 1:
                    next_expiration = available[1]
                    next_close = by_expiration[next_expiration][timestamp].get(s.CLOSE_COLUMN)
                explicit_next = front.get("next_close")
                if explicit_next is not None:
                    next_close = explicit_next
                explicit_next_expiration = front.get("next_expiration")
                if explicit_next_expiration is not None:
                    next_expiration = explicit_next_expiration
                row: dict[str, Any] = {
                    key: front.get(key)
                    for key in (
                        s.TIMESTAMP_COLUMN,
                        s.INSTRUMENT_ID_COLUMN,
                        s.ROOT_SYMBOL_COLUMN,
                        s.EXPIRATION_COLUMN,
                        s.OPEN_COLUMN,
                        s.HIGH_COLUMN,
                        s.LOW_COLUMN,
                        s.CLOSE_COLUMN,
                        s.VOLUME_COLUMN,
                        s.OPEN_INTEREST_COLUMN,
                        s.SETTLEMENT_COLUMN,
                    )
                }
                row["_next_close"] = next_close
                row["_next_expiration"] = next_expiration
                if s.TERM_SLOPE in front:
                    row["_input_term_slope"] = front[s.TERM_SLOPE]
                if s.CARRY_ANNUALIZED in front:
                    row["_input_carry_annualized"] = front[s.CARRY_ANNUALIZED]
                records.append(row)
        return pl.DataFrame(records, infer_schema_length=None).sort(
            [s.ROOT_SYMBOL_COLUMN, s.TIMESTAMP_COLUMN]
        )

    # -- per-root feature computation -----------------------------------------

    def _root_features(self, group: pl.DataFrame) -> pl.DataFrame:
        """Compute every feature for one root's front-contract series."""
        ordered = group.sort(s.TIMESTAMP_COLUMN)
        open_price = _float_array(ordered[s.OPEN_COLUMN])
        high = _float_array(ordered[s.HIGH_COLUMN])
        low = _float_array(ordered[s.LOW_COLUMN])
        close = _float_array(ordered[s.CLOSE_COLUMN])
        volume = _float_array(ordered[s.VOLUME_COLUMN])
        open_interest = _float_array(ordered[s.OPEN_INTEREST_COLUMN])
        previous_close = _shift(close, 1)
        annualization = math.sqrt(float(self._annualization_factor))

        def simple_return(periods: int) -> _FloatArray:
            return _safe_div(close, _shift(close, periods)) - 1.0

        ret_1d = simple_return(1)
        ret_5d = simple_return(5)
        ret_21d = simple_return(21)
        ret_63d = simple_return(63)
        ret_126d = simple_return(self._slow_momentum_window)
        ret_252d = simple_return(252)
        vol_short = _rolling_std(ret_1d, self._volatility_window) * annualization
        vol_long = _rolling_std(ret_1d, self._long_volatility_window) * annualization

        true_range = np.full(close.shape, np.nan, dtype=np.float64)
        true_range[0] = high[0] - low[0]
        true_range[1:] = np.maximum.reduce(
            [
                high[1:] - low[1:],
                np.abs(high[1:] - previous_close[1:]),
                np.abs(low[1:] - previous_close[1:]),
            ]
        )
        atr = _ewm(true_range, 1.0 / self._atr_window)

        safe_high = np.where(high > 0, high, np.nan)
        safe_low = np.where(low > 0, low, np.nan)
        safe_open = np.where(open_price > 0, open_price, np.nan)
        safe_close = np.where(close > 0, close, np.nan)
        log_range = np.log(_safe_div(safe_high, safe_low))
        log_body = np.log(_safe_div(safe_close, safe_open))
        garman_klass_bar = np.sqrt(
            np.maximum(0.5 * log_range**2 - _TWO_LN2_MINUS_ONE * log_body**2, 0.0)
        )
        garman_klass = _rolling_mean(garman_klass_bar, self._volatility_window) * annualization

        macd = _ema(close, 12) - _ema(close, 26)
        macd_signal = _ema(macd, 9)

        delta = close - previous_close
        gains = np.where(delta > 0.0, delta, 0.0)
        losses = np.where(delta < 0.0, -delta, 0.0)
        gains[0] = 0.0
        losses[0] = 0.0
        average_gain = _ewm(gains, 1.0 / self._rsi_window)
        average_loss = _ewm(losses, 1.0 / self._rsi_window)
        relative_strength = _safe_div(average_gain, average_loss)
        rsi = 100.0 - _safe_div(100.0, 1.0 + relative_strength)
        rsi = np.where(np.isfinite(rsi) & (average_loss == 0.0), 100.0, rsi)

        typical_price = (high + low + close) / 3.0
        typical_sma = _rolling_mean(typical_price, self._cci_window)
        mean_deviation = _rolling_apply(
            typical_price,
            self._cci_window,
            lambda windows: np.mean(
                np.abs(windows - np.mean(windows, axis=-1, keepdims=True)), axis=-1
            ),
        )
        cci = _safe_div(typical_price - typical_sma, 0.015 * mean_deviation)

        up_move = high - _shift(high, 1)
        down_move = _shift(low, 1) - low
        plus_dm = np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0)
        plus_dm[0] = 0.0
        minus_dm[0] = 0.0
        smoothed_plus = _ewm(plus_dm, 1.0 / self._adx_window)
        smoothed_minus = _ewm(minus_dm, 1.0 / self._adx_window)
        plus_di = 100.0 * _safe_div(smoothed_plus, atr)
        minus_di = 100.0 * _safe_div(smoothed_minus, atr)
        directional = 100.0 * _safe_div(np.abs(plus_di - minus_di), plus_di + minus_di)
        adx = _ewm(np.where(np.isfinite(directional), directional, 0.0), 1.0 / self._adx_window)

        bollinger_mid = _rolling_mean(close, self._bollinger_window)
        bollinger_std = _rolling_std(close, self._bollinger_window)
        bb_upper = _safe_div(bollinger_mid + 2.0 * bollinger_std, close)
        bb_lower = _safe_div(bollinger_mid - 2.0 * bollinger_std, close)
        bb_width = _safe_div(4.0 * bollinger_std, bollinger_mid)

        kama = _kama(close, self._kama_window, 2, 30)
        kama_ratio = _safe_div(kama, close)
        volume_oi_ratio = _safe_div(volume, open_interest)

        next_close = _float_array(
            ordered["_next_close"].cast(pl.Float64)
            if "_next_close" in ordered.columns
            else pl.Series("_next_close", [None] * ordered.height, dtype=pl.Float64)
        )
        term_slope = _safe_div(next_close - close, close)
        days_between = np.full(close.shape, np.nan, dtype=np.float64)
        if "_next_expiration" in ordered.columns:
            expiration_values = ordered[s.EXPIRATION_COLUMN].to_list()
            next_expiration_values = ordered["_next_expiration"].to_list()
            for index, (front_expiration, next_exp) in enumerate(
                zip(expiration_values, next_expiration_values, strict=True)
            ):
                if isinstance(front_expiration, date) and isinstance(next_exp, date):
                    days_between[index] = max((next_exp - front_expiration).days, 1)
        carry_annualized = -term_slope * _safe_div(np.full(close.shape, 365.0), days_between)
        if "_input_term_slope" in ordered.columns:
            input_term = _float_array(ordered["_input_term_slope"])
            term_slope = np.where(np.isfinite(term_slope), term_slope, input_term)
        if "_input_carry_annualized" in ordered.columns:
            input_carry = _float_array(ordered["_input_carry_annualized"])
            carry_annualized = np.where(
                np.isfinite(carry_annualized), carry_annualized, input_carry
            )

        features: dict[str, _FloatArray] = {
            s.RET_1D: ret_1d,
            s.RET_5D: ret_5d,
            s.RET_21D: ret_21d,
            s.RET_63D: ret_63d,
            s.RET_126D: ret_126d,
            s.RET_252D: ret_252d,
            s.VOL_20D: vol_short,
            s.VOL_60D: vol_long,
            s.ATR: atr,
            s.GARMAN_KLASS_VOL: garman_klass,
            s.MACD: macd,
            s.MACD_SIGNAL: macd_signal,
            s.RSI: rsi,
            s.CCI: cci,
            s.ADX: adx,
            s.BB_UPPER: bb_upper,
            s.BB_LOWER: bb_lower,
            s.BB_WIDTH: bb_width,
            s.TERM_SLOPE: term_slope,
            s.CARRY_ANNUALIZED: carry_annualized,
            s.VOLUME_OI_RATIO: volume_oi_ratio,
            s.KAMA: kama_ratio,
            s.TSMOM_21: ret_21d,
            s.TSMOM_126: ret_126d,
        }
        output = ordered.select(
            [
                s.TIMESTAMP_COLUMN,
                s.INSTRUMENT_ID_COLUMN,
                s.ROOT_SYMBOL_COLUMN,
                s.EXPIRATION_COLUMN,
                s.OPEN_COLUMN,
                s.HIGH_COLUMN,
                s.LOW_COLUMN,
                s.CLOSE_COLUMN,
                s.VOLUME_COLUMN,
                s.OPEN_INTEREST_COLUMN,
                s.SETTLEMENT_COLUMN,
            ]
        )
        return output.with_columns(
            [pl.Series(name, array.tolist(), dtype=pl.Float64) for name, array in features.items()]
        )

    # -- cross-sectional / external features -----------------------------------

    def _add_turbulence(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Attach the FinRL Mahalanobis turbulence index (strictly trailing moments)."""
        pivot = frame.pivot(
            on=s.ROOT_SYMBOL_COLUMN,
            index=s.TIMESTAMP_COLUMN,
            values=s.RET_1D,
            aggregate_function="first",
        ).sort(s.TIMESTAMP_COLUMN)
        roots = [column for column in pivot.columns if column != s.TIMESTAMP_COLUMN]
        turbulence = np.full(pivot.height, np.nan, dtype=np.float64)
        if roots:
            returns = np.asarray(pivot.select(roots).to_numpy(), dtype=np.float64)
            minimum_history = max(10, len(roots) + 2)
            window = self._turbulence_window
            for index in range(pivot.height):
                history = returns[max(0, index - window) : index]
                history = history[np.isfinite(history).all(axis=1)]
                if history.shape[0] < minimum_history:
                    continue
                current = returns[index]
                if not np.isfinite(current).all():
                    continue
                mean = history.mean(axis=0)
                centred = history - mean
                covariance = (centred.T @ centred) / float(history.shape[0] - 1)
                covariance = covariance + np.eye(covariance.shape[0]) * 1e-12
                deviation = current - mean
                turbulence[index] = float(deviation @ np.linalg.pinv(covariance) @ deviation)
        turbulence_frame = pl.DataFrame(
            {
                s.TIMESTAMP_COLUMN: pivot[s.TIMESTAMP_COLUMN].to_list(),
                s.TURBULENCE: turbulence.tolist(),
            }
        )
        return frame.join(turbulence_frame, on=s.TIMESTAMP_COLUMN, how="left")

    def _add_ofi(self, frame: pl.DataFrame, quotes: pl.DataFrame | None) -> pl.DataFrame:
        """Attach per-bar order-flow imbalance (bar-window sum of quote OFI)."""
        if quotes is None:
            if s.OFI in frame.columns:
                return frame
            return frame.with_columns(pl.lit(None, dtype=pl.Float64).alias(s.OFI))
        if quotes.is_empty():
            return frame.with_columns(pl.lit(None, dtype=pl.Float64).alias(s.OFI))
        ofi_frame = self.compute_order_flow_imbalance(quotes)
        outputs: list[pl.DataFrame] = []
        for root, group in self._group_by_root(frame):
            ordered = group.sort(s.TIMESTAMP_COLUMN)
            root_ofi = ofi_frame.filter(pl.col(s.ROOT_SYMBOL_COLUMN) == root).sort(
                s.TIMESTAMP_COLUMN
            )
            if root_ofi.is_empty():
                outputs.append(ordered.with_columns(pl.lit(None, dtype=pl.Float64).alias(s.OFI)))
                continue
            bar_ns = np.asarray(
                ordered[s.TIMESTAMP_COLUMN].dt.epoch("ns").to_numpy(), dtype=np.int64
            )
            quote_ns = np.asarray(
                root_ofi[s.TIMESTAMP_COLUMN].dt.epoch("ns").to_numpy(), dtype=np.int64
            )
            ofi_values = _float_array(root_ofi[s.OFI])
            cumulative = np.concatenate([[0.0], np.cumsum(np.nan_to_num(ofi_values, nan=0.0))])
            upper = np.searchsorted(quote_ns, bar_ns, side="right")
            lower = np.concatenate([[0], upper[:-1]])
            bar_ofi = cumulative[upper] - cumulative[lower]
            # No quote has ever been seen by the first bar: unknown, not zero flow.
            bar_ofi[upper == 0] = np.nan
            outputs.append(
                ordered.with_columns(pl.Series(s.OFI, bar_ofi.tolist(), dtype=pl.Float64))
            )
        return pl.concat(outputs, how="vertical_relaxed")

    def _macro_series(self, exogenous: pl.DataFrame) -> dict[str, _FloatArray]:
        """Derive the canonical macro features from a flexible exogenous frame."""
        frame = s.normalise_timestamp_column(exogenous, s.TIMESTAMP_COLUMN).sort(s.TIMESTAMP_COLUMN)
        columns = frame.columns
        result: dict[str, _FloatArray] = {}

        def series(name: str) -> _FloatArray | None:
            if name in columns:
                return _float_array(frame[name])
            return None

        if s.MACRO_VIX_TERM_SLOPE in columns:
            result[s.MACRO_VIX_TERM_SLOPE] = _float_array(frame[s.MACRO_VIX_TERM_SLOPE])
        elif series("vix_front") is not None and series("vix_back") is not None:
            front = _float_array(frame["vix_front"])
            back = _float_array(frame["vix_back"])
            result[s.MACRO_VIX_TERM_SLOPE] = _safe_div(
                np.log(np.where(back > 0, back, np.nan)),
                np.log(np.where(front > 0, front, np.nan)),
            )
        if s.MACRO_YIELD_CURVE_SLOPE in columns:
            result[s.MACRO_YIELD_CURVE_SLOPE] = _float_array(frame[s.MACRO_YIELD_CURVE_SLOPE])
        elif series("yield_10y") is not None and series("yield_2y") is not None:
            result[s.MACRO_YIELD_CURVE_SLOPE] = _float_array(frame["yield_10y"]) - _float_array(
                frame["yield_2y"]
            )
        dxy = series("dxy")
        if s.MACRO_DXY_RETURN_5D in columns:
            result[s.MACRO_DXY_RETURN_5D] = _float_array(frame[s.MACRO_DXY_RETURN_5D])
        elif dxy is not None:
            result[s.MACRO_DXY_RETURN_5D] = _safe_div(dxy, _shift(dxy, 5)) - 1.0
        if s.MACRO_DXY_TREND_21D in columns:
            result[s.MACRO_DXY_TREND_21D] = _float_array(frame[s.MACRO_DXY_TREND_21D])
        elif dxy is not None:
            result[s.MACRO_DXY_TREND_21D] = _safe_div(dxy, _shift(dxy, 21)) - 1.0
        return result

    def _add_macro(self, frame: pl.DataFrame, exogenous: pl.DataFrame | None) -> pl.DataFrame:
        """Attach macro features by as-of (backward) alignment to bar timestamps."""
        macro_names = [
            s.MACRO_VIX_TERM_SLOPE,
            s.MACRO_YIELD_CURVE_SLOPE,
            s.MACRO_DXY_RETURN_5D,
            s.MACRO_DXY_TREND_21D,
        ]
        if exogenous is None or exogenous.is_empty():
            return frame.with_columns(
                [pl.lit(None, dtype=pl.Float64).alias(name) for name in macro_names]
            )
        macro_frame = s.normalise_timestamp_column(exogenous, s.TIMESTAMP_COLUMN).sort(
            s.TIMESTAMP_COLUMN
        )
        macro_series = self._macro_series(macro_frame)
        exogenous_ns = np.asarray(
            macro_frame[s.TIMESTAMP_COLUMN].dt.epoch("ns").to_numpy(), dtype=np.int64
        )
        bar_ns = np.asarray(frame[s.TIMESTAMP_COLUMN].dt.epoch("ns").to_numpy(), dtype=np.int64)
        positions = np.searchsorted(exogenous_ns, bar_ns, side="right") - 1
        for name in macro_names:
            values = macro_series.get(name)
            if values is None:
                frame = frame.with_columns(pl.lit(None, dtype=pl.Float64).alias(name))
                continue
            aligned = np.where(positions >= 0, values[np.clip(positions, 0, None)], np.nan)
            frame = frame.with_columns(pl.Series(name, aligned.tolist(), dtype=pl.Float64))
        return frame
