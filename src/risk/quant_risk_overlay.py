"""Tier B quantitative futures risk overlay (FinRL Risk-First, statistical controls).

The overlay implements the ``R_t`` statistical side of the FinRL-X risk stage:

* **Ex-ante volatility targeting** - scale the signed notional weights so that the
  annualised portfolio volatility implied by the EWMA covariance matrix matches
  ``target_annual_volatility``. The covariance source is, in order: the state's
  point-in-time covariance matrix, an EWMA covariance estimated from
  ``state.returns`` with ``ewma_halflife_days``, or a diagonal (correlation-free)
  volatility fallback. A singular / non-positive-semidefinite covariance matrix
  can never raise or produce ``inf``/``nan`` weights.
* **Gross leverage, single-instrument and sector caps** - the correlation-cluster
  guard caps ``sum(|w|)`` per instrument master sector (e.g. ES + NQ share the
  ``equity`` beta bucket) and scales the offending sector down proportionally.
* **High-water-mark trailing stops** - per instrument (from the favourable price
  extreme while a position is held) and at the portfolio level (from the equity
  high-water mark). A stopped instrument stays flat until the next session roll.
* **Intraday daily max-loss kill switch** - when equity falls by
  ``daily_max_loss_pct`` from the starting daily NLV the overlay flattens the
  book and latches until :meth:`QuantRiskOverlay.reset_day` or the next trading
  session.
* **Drawdown limit** - a portfolio drawdown from the equity high-water mark of
  ``drawdown_limit_pct`` flattens the book while the breach persists.

Everything is deterministic, stateful and driven exclusively by the immutable
:class:`~src.core.types.MarketState` plus injected equity updates. A clock can
be injected for reproducible reset/day-roll timestamps.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

import numpy as np
from configs.instruments import InstrumentsConfig, InstrumentSpec, load_instruments
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from src.core.types import TRADING_DAYS_PER_YEAR, InstrumentId, MarketState, WeightVector

if TYPE_CHECKING:
    from configs.settings import RiskSettings

__all__ = ["QuantRiskConfig", "QuantRiskOverlay"]

#: Weight magnitudes below this threshold are treated as flat.
_WEIGHT_EPSILON: float = 1e-12


class QuantRiskConfig(BaseModel):
    """Typed configuration for :class:`QuantRiskOverlay`.

    ``from_risk_settings`` maps the central :class:`configs.settings.RiskSettings`
    onto this model so the overlay stays a single-source-of-truth consumer.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    target_annual_volatility: float = Field(default=0.12, gt=0.0)
    ewma_halflife_days: float = Field(default=20.0, gt=0.0)
    max_gross_leverage: float = Field(default=3.0, gt=0.0)
    max_single_instrument_weight: float = Field(default=0.50, gt=0.0)
    max_sector_weight: float = Field(default=0.40, gt=0.0)
    daily_max_loss_pct: float = Field(default=0.03, gt=0.0, le=1.0)
    trailing_stop_pct: float = Field(default=0.10, gt=0.0, le=1.0)
    drawdown_limit_pct: float = Field(default=0.20, gt=0.0, le=1.0)
    volatility_targeting_enabled: bool = True
    daily_loss_kill_switch_enabled: bool = True
    min_portfolio_volatility: float = Field(default=1e-8, gt=0.0)
    annualization_factor: int = Field(default=TRADING_DAYS_PER_YEAR, gt=0)

    @classmethod
    def from_risk_settings(cls, settings: RiskSettings) -> QuantRiskConfig:
        """Build the overlay configuration from the central risk settings."""
        return cls(
            target_annual_volatility=settings.target_annual_volatility,
            ewma_halflife_days=settings.ewma_halflife_days,
            max_gross_leverage=settings.max_gross_leverage,
            max_single_instrument_weight=settings.max_single_instrument_weight,
            max_sector_weight=settings.max_sector_weight,
            daily_max_loss_pct=settings.daily_max_loss_pct,
            trailing_stop_pct=settings.trailing_stop_pct,
            drawdown_limit_pct=settings.drawdown_limit_pct,
        )


class QuantRiskOverlay:
    """Stateful statistical risk overlay implementing the ``RiskOverlay`` protocol.

    Parameters
    ----------
    config:
        A :class:`QuantRiskConfig`, a central ``RiskSettings`` instance, or
        ``None`` for the defaults.
    target_annual_volatility, ewma_halflife_days, max_gross_leverage, ...
        Explicit per-parameter overrides applied on top of ``config``.
    instruments:
        Instrument master used to resolve asset-class clusters. Defaults to
        ``configs/instruments.yaml``.
    clock:
        Injectable clock returning a timezone-aware ``datetime`` (used only when
        :meth:`update_equity` / :meth:`reset_day` are called without a timestamp).
    """

    def __init__(
        self,
        config: QuantRiskConfig | RiskSettings | None = None,
        *,
        target_annual_volatility: float | None = None,
        ewma_halflife_days: float | None = None,
        max_gross_leverage: float | None = None,
        max_single_instrument_weight: float | None = None,
        max_sector_weight: float | None = None,
        daily_max_loss_pct: float | None = None,
        trailing_stop_pct: float | None = None,
        drawdown_limit_pct: float | None = None,
        volatility_targeting_enabled: bool | None = None,
        daily_loss_kill_switch_enabled: bool | None = None,
        instruments: InstrumentsConfig | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        base = self._coerce_config(config)
        overrides: dict[str, float | bool] = {}
        for name, value in (
            ("target_annual_volatility", target_annual_volatility),
            ("ewma_halflife_days", ewma_halflife_days),
            ("max_gross_leverage", max_gross_leverage),
            ("max_single_instrument_weight", max_single_instrument_weight),
            ("max_sector_weight", max_sector_weight),
            ("daily_max_loss_pct", daily_max_loss_pct),
            ("trailing_stop_pct", trailing_stop_pct),
            ("drawdown_limit_pct", drawdown_limit_pct),
            ("volatility_targeting_enabled", volatility_targeting_enabled),
            ("daily_loss_kill_switch_enabled", daily_loss_kill_switch_enabled),
        ):
            if value is not None:
                overrides[name] = value
        self._config: QuantRiskConfig = base.model_copy(update=overrides) if overrides else base
        self._instruments = instruments
        self._clock: Callable[[], datetime] = clock or (lambda: datetime.now(UTC))

        # -- mutable session / equity state --------------------------------------
        self._equity: float | None = None
        self._starting_daily_equity: float | None = None
        self._portfolio_hwm: float | None = None
        self._session_date: date | None = None
        self._last_timestamp: datetime | None = None
        self._daily_loss_latched: bool = False
        self._stopped_roots: set[InstrumentId] = set()
        self._instrument_hwm: dict[InstrumentId, tuple[float, int]] = {}
        self._last_directions: dict[InstrumentId, int] = {}

    # -- public API ------------------------------------------------------------

    @property
    def name(self) -> str:
        """Overlay name used in telemetry / pipeline traces."""
        return "quant_risk"

    @property
    def config(self) -> QuantRiskConfig:
        """Effective overlay configuration."""
        return self._config

    @property
    def kill_switch_active(self) -> bool:
        """Whether the intraday daily max-loss kill switch is latched."""
        return self._daily_loss_latched

    def apply_risk(self, weights: WeightVector, state: MarketState) -> WeightVector:
        """Project ``weights`` onto the quantitative risk constraints."""
        self._observe(state)
        keys: tuple[InstrumentId, ...] = tuple(weights)
        if not weights.weights or weights.is_flat():
            return WeightVector(dict.fromkeys(keys, 0.0))
        if self._daily_loss_latched or self._portfolio_risk_breached():
            return WeightVector(dict.fromkeys(keys, 0.0))

        normalised, specs = self._normalise(weights)
        current = self._apply_instrument_stops(normalised, state)
        scaled = self._apply_volatility_target(current, state)
        scaled = self._apply_single_instrument_cap(scaled)
        scaled = self._apply_sector_cap(scaled, specs)
        scaled = self._apply_gross_leverage_cap(scaled)
        return WeightVector(scaled)

    def update_equity(self, equity: float, timestamp: datetime | None = None) -> None:
        """Mark the account equity to market and evaluate the daily loss limit."""
        value = self._validate_equity(equity)
        moment = self._timestamp(timestamp)
        self._roll_session(moment, value)
        self._set_equity(value, moment)

    def reset_day(self, equity: float | None = None, timestamp: datetime | None = None) -> None:
        """Reset the daily loss baseline, kill switch and instrument stops.

        This models the next trading session roll: the daily max-loss latch is
        cleared, the starting daily NLV is re-based and trailing-stopped
        instruments may be re-entered. The portfolio high-water mark is preserved
        so the portfolio drawdown limit keeps its history.
        """
        value = self._validate_equity(equity if equity is not None else self._equity)
        moment = self._timestamp(timestamp)
        self._session_date = moment.date()
        self._starting_daily_equity = value
        self._equity = value
        self._portfolio_hwm = max(self._portfolio_hwm or value, value)
        self._daily_loss_latched = False
        self._stopped_roots.clear()
        self._last_timestamp = moment

    def reset_kill_switch(self) -> None:
        """Manually clear the latched daily max-loss kill switch."""
        self._daily_loss_latched = False

    def reset(self) -> None:
        """Clear every piece of mutable overlay state (test / new-session helper)."""
        self._equity = None
        self._starting_daily_equity = None
        self._portfolio_hwm = None
        self._session_date = None
        self._last_timestamp = None
        self._daily_loss_latched = False
        self._stopped_roots.clear()
        self._instrument_hwm.clear()
        self._last_directions.clear()

    def snapshot_state(self) -> dict[str, object]:
        """JSON-serialisable snapshot of the overlay's risk state."""
        daily_pnl: float | None = None
        if self._equity is not None and self._starting_daily_equity:
            daily_pnl = (self._equity - self._starting_daily_equity) / self._starting_daily_equity
        drawdown: float | None = None
        if self._equity is not None and self._portfolio_hwm:
            drawdown = (self._portfolio_hwm - self._equity) / self._portfolio_hwm
        return {
            "equity": self._equity,
            "starting_daily_equity": self._starting_daily_equity,
            "session_date": self._session_date.isoformat() if self._session_date else None,
            "last_timestamp": (self._last_timestamp.isoformat() if self._last_timestamp else None),
            "portfolio_high_water_mark": self._portfolio_hwm,
            "daily_pnl_pct": daily_pnl,
            "drawdown_pct": drawdown,
            "kill_switch_active": self._daily_loss_latched,
            "stopped_roots": sorted(self._stopped_roots),
            "instrument_high_water_marks": {
                root: {"price": hwm, "direction": direction}
                for root, (hwm, direction) in sorted(self._instrument_hwm.items())
            },
        }

    # -- configuration / normalisation ----------------------------------------

    @staticmethod
    def _coerce_config(config: QuantRiskConfig | RiskSettings | None) -> QuantRiskConfig:
        if config is None:
            return QuantRiskConfig()
        if isinstance(config, QuantRiskConfig):
            return config
        if hasattr(config, "target_annual_volatility"):
            return QuantRiskConfig.from_risk_settings(config)
        raise TypeError(
            "config must be a QuantRiskConfig or configs.settings.RiskSettings, "
            f"got {type(config)!r}"
        )

    def _instrument_master(self) -> InstrumentsConfig:
        return self._instruments if self._instruments is not None else load_instruments()

    def _normalise(
        self, weights: WeightVector
    ) -> tuple[dict[InstrumentId, float], dict[InstrumentId, InstrumentSpec]]:
        """Canonicalise roots and resolve their instrument specs."""
        master = self._instrument_master()
        normalised: dict[InstrumentId, float] = {}
        specs: dict[InstrumentId, InstrumentSpec] = {}
        for root, value in weights.items():
            try:
                spec = master.get(root)
            except KeyError as exc:
                raise ValueError(
                    f"unknown futures root {root!r}: not declared in the instrument master"
                ) from exc
            canonical = spec.root
            normalised[canonical] = normalised.get(canonical, 0.0) + value
            specs[canonical] = spec
        return normalised, specs

    # -- equity / session state ------------------------------------------------

    @staticmethod
    def _validate_equity(equity: float | None) -> float:
        if equity is None:
            raise ValueError(
                "equity must be provided: pass update_equity(...)/reset_day(...) or set "
                "MarketState.equity"
            )
        value = float(equity)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"equity must be a finite strictly positive number, got {value!r}")
        return value

    def _timestamp(self, timestamp: datetime | None) -> datetime:
        moment = timestamp if timestamp is not None else self._clock()
        if moment.tzinfo is None or moment.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        return moment.astimezone(UTC)

    def _observe(self, state: MarketState) -> None:
        moment = state.timestamp
        equity = state.equity
        self._roll_session(moment, equity)
        if equity is not None:
            self._set_equity(self._validate_equity(equity), moment)
        self._last_timestamp = moment

    def _roll_session(self, moment: datetime, equity: float | None) -> None:
        session = moment.astimezone(UTC).date()
        if self._session_date is None:
            self._session_date = session
            return
        if session == self._session_date:
            return
        # New trading session: clear daily counters/stops but keep the portfolio HWM.
        self._session_date = session
        self._daily_loss_latched = False
        self._stopped_roots.clear()
        new_equity = equity if equity is not None else self._equity
        if new_equity is not None:
            value = self._validate_equity(new_equity)
            self._starting_daily_equity = value
            self._equity = value
            self._portfolio_hwm = max(self._portfolio_hwm or value, value)

    def _set_equity(self, equity: float, moment: datetime) -> None:
        self._equity = equity
        if self._starting_daily_equity is None:
            self._starting_daily_equity = equity
        self._portfolio_hwm = (
            equity if self._portfolio_hwm is None else max(self._portfolio_hwm, equity)
        )
        self._last_timestamp = moment
        if self._config.daily_loss_kill_switch_enabled:
            starting = self._starting_daily_equity
            if starting and (starting - equity) / starting >= self._config.daily_max_loss_pct:
                self._daily_loss_latched = True

    def _portfolio_risk_breached(self) -> bool:
        equity = self._equity
        hwm = self._portfolio_hwm
        if equity is None or hwm is None or hwm <= 0.0:
            return False
        drawdown = (hwm - equity) / hwm
        if drawdown >= self._config.drawdown_limit_pct:
            return True
        return drawdown >= self._config.trailing_stop_pct

    # -- trailing stops --------------------------------------------------------

    def _apply_instrument_stops(
        self, weights: dict[InstrumentId, float], state: MarketState
    ) -> dict[InstrumentId, float]:
        output = dict(weights)
        threshold = self._config.trailing_stop_pct
        for root, value in weights.items():
            direction = 1 if value > 0.0 else (-1 if value < 0.0 else 0)
            if root in self._stopped_roots:
                output[root] = 0.0
                continue
            if direction == 0:
                self._last_directions[root] = 0
                continue
            snapshot = state.instruments.get(root)
            if snapshot is None:
                # No mark price: leave the upstream weight untouched.
                self._last_directions[root] = direction
                continue
            price = snapshot.price
            previous_direction = self._last_directions.get(root, 0)
            if previous_direction != direction:
                self._instrument_hwm[root] = (price, direction)
            else:
                hwm = self._instrument_hwm.get(root, (price, direction))[0]
                if direction > 0:
                    hwm = max(hwm, price)
                    triggered = (hwm - price) / hwm >= threshold
                else:
                    hwm = min(hwm, price)
                    triggered = (price - hwm) / hwm >= threshold
                self._instrument_hwm[root] = (hwm, direction)
                if triggered:
                    self._stopped_roots.add(root)
                    output[root] = 0.0
            self._last_directions[root] = direction
        return output

    # -- volatility targeting --------------------------------------------------

    def _apply_volatility_target(
        self, weights: dict[InstrumentId, float], state: MarketState
    ) -> dict[InstrumentId, float]:
        if not self._config.volatility_targeting_enabled:
            return dict(weights)
        sigma = self._portfolio_volatility(tuple(weights), weights, state)
        if sigma is None or sigma <= self._config.min_portfolio_volatility:
            return dict(weights)
        scale = self._config.target_annual_volatility / sigma
        if not math.isfinite(scale) or scale <= 0.0:  # pragma: no cover - sigma floor
            return dict(weights)
        return {root: value * scale for root, value in weights.items()}

    def _portfolio_volatility(
        self,
        instruments: tuple[InstrumentId, ...],
        weights: Mapping[InstrumentId, float],
        state: MarketState,
    ) -> float | None:
        vector = self._position_vector(instruments, weights)
        covariance = state.covariance_submatrix(instruments)
        if covariance is not None:
            sigma = self._quadratic_form_volatility(vector, covariance)
            if sigma is not None:
                return sigma
        ewma = self._ewma_covariance(instruments, state)
        if ewma is not None:
            sigma = self._quadratic_form_volatility(vector, ewma)
            if sigma is not None:
                return sigma
        return self._diagonal_volatility(vector, instruments, state)

    @staticmethod
    def _position_vector(
        instruments: tuple[InstrumentId, ...], weights: Mapping[InstrumentId, float]
    ) -> NDArray[np.float64]:
        return np.array([weights.get(root, 0.0) for root in instruments], dtype=np.float64)

    @staticmethod
    def _quadratic_form_volatility(
        vector: NDArray[np.float64], covariance: NDArray[np.float64]
    ) -> float | None:
        variance = float(vector @ covariance @ vector)
        if not math.isfinite(variance) or variance <= 0.0:
            return None
        return math.sqrt(variance)

    def _ewma_covariance(
        self, instruments: tuple[InstrumentId, ...], state: MarketState
    ) -> NDArray[np.float64] | None:
        columns: list[NDArray[np.float64]] = []
        for root in instruments:
            series = state.returns.get(root)
            if series is None or series.size < 2:
                return None
            columns.append(np.asarray(series, dtype=np.float64))
        length = min(column.size for column in columns)
        if length < 2:  # pragma: no cover - each column was already checked
            return None
        matrix = np.column_stack([column[-length:] for column in columns])
        halflife = self._config.ewma_halflife_days
        decay = 0.5 ** (1.0 / halflife)
        raw_weights = decay ** np.arange(length - 1, -1, -1, dtype=np.float64)
        weights = raw_weights / float(np.sum(raw_weights))
        means = weights @ matrix
        centered = matrix - means
        covariance = (centered * weights[:, None]).T @ centered
        return np.asarray(covariance * float(self._config.annualization_factor), dtype=np.float64)

    def _diagonal_volatility(
        self,
        vector: NDArray[np.float64],
        instruments: tuple[InstrumentId, ...],
        state: MarketState,
    ) -> float | None:
        variances: list[float] = []
        covariance = state.covariance
        positions = {name: index for index, name in enumerate(state.covariance_instruments)}
        for root in instruments:
            variance: float | None = None
            if covariance is not None and root in positions:
                candidate = float(covariance[positions[root], positions[root]])
                variance = candidate if math.isfinite(candidate) else None
            if variance is None:
                series = state.returns.get(root)
                if series is not None and series.size >= 2:
                    values = np.asarray(series, dtype=np.float64)
                    variance = float(np.var(values, ddof=1)) * float(
                        self._config.annualization_factor
                    )
            if variance is None:
                return None
            variances.append(max(variance, 0.0))
        weighted = vector * np.array(variances, dtype=np.float64)
        total = float(np.sum(vector * weighted))
        if not math.isfinite(total) or total <= 0.0:
            return None
        return math.sqrt(total)

    # -- weight caps -----------------------------------------------------------

    def _apply_single_instrument_cap(
        self, weights: dict[InstrumentId, float]
    ) -> dict[InstrumentId, float]:
        cap = self._config.max_single_instrument_weight
        return {root: max(-cap, min(cap, value)) for root, value in weights.items()}

    def _apply_sector_cap(
        self,
        weights: dict[InstrumentId, float],
        specs: dict[InstrumentId, InstrumentSpec],
    ) -> dict[InstrumentId, float]:
        sector_gross: dict[str, float] = {}
        for root, value in weights.items():
            sector = specs[root].sector
            sector_gross[sector] = sector_gross.get(sector, 0.0) + abs(value)
        factors: dict[str, float] = {}
        cap = self._config.max_sector_weight
        for sector, gross in sector_gross.items():
            factors[sector] = cap / gross if gross > cap and gross > 0.0 else 1.0
        return {root: value * factors[specs[root].sector] for root, value in weights.items()}

    def _apply_gross_leverage_cap(
        self, weights: dict[InstrumentId, float]
    ) -> dict[InstrumentId, float]:
        gross = sum(abs(value) for value in weights.values())
        limit = self._config.max_gross_leverage
        if gross <= limit or gross <= _WEIGHT_EPSILON:
            return dict(weights)
        factor = limit / gross
        return {root: value * factor for root, value in weights.items()}
