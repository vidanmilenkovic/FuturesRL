"""Deterministic translation of target notional weights into discrete futures contracts.

This module implements the ``w_t -> N_i,t`` stage of the FinRL-X pipeline described
in ``AGENTS.md`` section 1.2. It is the single place where a continuous, signed
notional weight vector becomes an executable set of integer futures contracts, so
the backtest engine and the live IBKR execution engine observe *identical* targets
for an identical :class:`~src.core.types.MarketState`.

Algorithm
---------
1. ``TargetNotional_i = w_i * E`` where ``E`` is the supplied equity (falling back
   to ``state.equity``).
2. ``N_raw_i = TargetNotional_i / (P_i * M_i)`` using the point-in-time mark price
   and the contract multiplier from the instrument master.
3. Discrete rounding (``floor`` toward zero, ``ceil`` away from zero, or ``nearest``
   with a configurable rebalance buffer) plus automatic standard/micro routing
   (``ES -> MES``, ``NQ -> MNQ``, ``CL -> MCL``, ``GC -> MGC``) that pushes the
   residual fractional exposure into the micro root to minimise tracking error.
   Several deterministic decompositions of one standard/micro family are compared
   (configured rounding, truncation + same-sign residual, micro-only) and the one
   with the smallest absolute notional tracking error wins (ties prefer fewer
   legs). The translator never manufactures an opposite-signed micro leg to hedge
   its own standard leg: a family leg pair always carries the same directional
   exposure, so commissions, slippage, basis risk and OMS/reconciler noise from a
   self-cancelling spread are avoided.
4. A margin feasibility projection enforces ``IM_proj / E <= eta_max`` by uniformly
   scaling the raw targets down and re-rounding, with a deterministic contract-drop
   fallback that guarantees the returned target set never violates the ceiling.

Design notes
------------
* The translator is stateless and deterministic: the same inputs always produce
  the same :class:`~src.core.types.ContractTargetSet`.
* The ``rebalance_buffer_contracts`` parameter raises the round-up threshold in
  ``NEAREST`` mode to ``0.5 + min(buffer, 1.0) / 2`` contracts, which biases the
  discrete target toward zero so marginal increments do not generate churn trades.
* Auto-micro routing is only possible when the market state carries a snapshot for
  the micro root (the translator never fabricates a price). A requested root that
  is absent from the market state is a hard error.
* The returned set contains only non-zero contract targets; a flat weight vector
  produces the empty (flat) target set.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

from configs.instruments import InstrumentsConfig, InstrumentSpec, load_instruments

from src.core.types import (
    ContractTarget,
    ContractTargetSet,
    InstrumentId,
    InstrumentSnapshot,
    MarketState,
    RoundingMode,
    WeightVector,
)

if TYPE_CHECKING:
    from configs.strategy import TranslatorConfig

__all__ = ["ContractTranslator"]

#: Iteration cap for the proportional margin-scaling loop.
_MARGIN_SCALING_ITERATIONS: int = 64
#: Defensive cap for the greedy contract-drop fallback (never reached in practice).
_MAX_GREEDY_DROPS: int = 100_000


@dataclass(slots=True)
class _Leg:
    """Internal mutable representation of one candidate contract leg."""

    instrument_id: InstrumentId
    spec: InstrumentSpec
    price: float
    expiration: date | None
    raw_quantity: float

    @property
    def initial_margin(self) -> float:
        return self.spec.margins.initial


@dataclass(frozen=True, slots=True)
class ContractTranslator:
    """Convert approved weight vectors into discrete futures contract targets.

    Parameters
    ----------
    instruments:
        Instrument master providing multipliers, tick sizes, margins and the
        standard <-> micro routing map. Defaults to ``configs/instruments.yaml``.
    rounding_mode:
        ``FLOOR`` (toward zero), ``CEIL`` (away from zero) or ``NEAREST``.
    rebalance_buffer_contracts:
        Rebalance dead-band, in contracts, applied to ``NEAREST`` rounding.
    max_margin_utilization:
        Hard upper bound on ``IM_proj / equity`` (default ``0.60``).
    enable_micro_routing:
        Whether residual fractional standard-contract exposure may be routed into
        the paired micro root.
    min_micro_notional_ratio:
        Minimum residual notional, expressed as a fraction of one micro contract's
        notional, required before the residual is routed into micro contracts.
    """

    instruments: InstrumentsConfig | None = None
    rounding_mode: RoundingMode = RoundingMode.NEAREST
    rebalance_buffer_contracts: float = 0.0
    max_margin_utilization: float = 0.60
    enable_micro_routing: bool = True
    min_micro_notional_ratio: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.rounding_mode, RoundingMode):
            raise TypeError(
                f"rounding_mode must be a RoundingMode, got {type(self.rounding_mode)!r}"
            )
        buffer = float(self.rebalance_buffer_contracts)
        if not math.isfinite(buffer) or buffer < 0.0:
            raise ValueError("rebalance_buffer_contracts must be a finite non-negative number")
        utilization = float(self.max_margin_utilization)
        if not math.isfinite(utilization) or not 0.0 < utilization <= 1.0:
            raise ValueError("max_margin_utilization must be within (0, 1]")
        ratio = float(self.min_micro_notional_ratio)
        if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
            raise ValueError("min_micro_notional_ratio must be within [0, 1]")

    # -- public API ------------------------------------------------------------

    @classmethod
    def from_config(
        cls, config: TranslatorConfig, *, instruments: InstrumentsConfig | None = None
    ) -> ContractTranslator:
        """Build a translator from a declarative :class:`TranslatorConfig`."""
        return cls(
            instruments=instruments,
            rounding_mode=config.rounding_mode,
            rebalance_buffer_contracts=config.rebalance_buffer_contracts,
            max_margin_utilization=config.max_margin_utilization,
            enable_micro_routing=config.enable_micro_routing,
            min_micro_notional_ratio=config.min_micro_notional_ratio,
        )

    def translate(
        self, weights: WeightVector, state: MarketState, *, equity: float | None = None
    ) -> ContractTargetSet:
        """Convert ``weights`` into discrete contract targets.

        Parameters
        ----------
        weights:
            Signed notional target weights. Negative values are shorts.
        state:
            Point-in-time market state providing mark prices, expirations and the
            default equity.
        equity:
            Explicit net liquidation value. When omitted, ``state.equity`` is used;
            when neither is available a :class:`ValueError` is raised.

        Returns
        -------
        ContractTargetSet
            Non-zero integer contract targets. A flat/zero weight vector yields the
            empty target set.
        """
        master = self._instrument_master()
        active = self._normalise_weights(weights, master)
        if not active:
            return ContractTargetSet.empty()
        resolved_equity = self._resolve_equity(equity, state)
        specs = self._resolve_specs(active, master)
        legs = self._build_legs(active, specs, master, state, resolved_equity)
        quantities = self._resolve_quantities(legs, resolved_equity)
        return self._materialise(legs, quantities)

    def margin_utilization(self, targets: ContractTargetSet, equity: float) -> float:
        """Projected initial-margin utilisation of ``targets`` at ``equity``."""
        value = float(equity)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("equity must be a finite strictly positive number")
        return targets.total_initial_margin / value

    # -- normalisation / validation -------------------------------------------

    def _instrument_master(self) -> InstrumentsConfig:
        return self.instruments if self.instruments is not None else load_instruments()

    @staticmethod
    def _normalise_weights(
        weights: WeightVector, master: InstrumentsConfig
    ) -> dict[InstrumentId, float]:
        """Resolve every weight to its canonical root and aggregate duplicates."""
        active: dict[InstrumentId, float] = {}
        for root, value in weights.items():
            if value == 0.0:
                continue
            try:
                spec = master.get(root)
            except KeyError as exc:
                raise ValueError(
                    f"unknown futures root {root!r}: not declared in the instrument master"
                ) from exc
            active[spec.root] = active.get(spec.root, 0.0) + value
        return active

    @staticmethod
    def _resolve_specs(
        active: dict[InstrumentId, float], master: InstrumentsConfig
    ) -> dict[InstrumentId, InstrumentSpec]:
        return {root: master.get(root) for root in active}

    @staticmethod
    def _resolve_equity(equity: float | None, state: MarketState) -> float:
        candidate = equity if equity is not None else state.equity
        if candidate is None:
            raise ValueError(
                "ContractTranslator requires equity: pass equity=... explicitly or populate "
                "MarketState.equity with the account net liquidation value"
            )
        value = float(candidate)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"equity must be a finite strictly positive number, got {value!r}")
        return value

    # -- contract quantity construction ---------------------------------------

    def _build_legs(
        self,
        active: dict[InstrumentId, float],
        specs: dict[InstrumentId, InstrumentSpec],
        master: InstrumentsConfig,
        state: MarketState,
        equity: float,
    ) -> list[_Leg]:
        """Build raw (rounding-aware) contract legs, including micro routing."""
        family_order: list[InstrumentId] = []
        families: dict[InstrumentId, dict[InstrumentId, float]] = {}
        for root, weight in active.items():
            spec = specs[root]
            family_key = spec.standard_root if spec.standard_root is not None else spec.root
            if family_key not in families:
                families[family_key] = {}
                family_order.append(family_key)
            families[family_key][root] = weight

        legs: list[_Leg] = []
        for family_key in family_order:
            family_weights = families[family_key]
            legs.extend(
                self._build_family_legs(family_key, family_weights, specs, master, state, equity)
            )
        return legs

    def _build_family_legs(
        self,
        family_key: InstrumentId,
        family_weights: dict[InstrumentId, float],
        specs: dict[InstrumentId, InstrumentSpec],
        master: InstrumentsConfig,
        state: MarketState,
        equity: float,
    ) -> list[_Leg]:
        standard_spec = self._family_standard_spec(family_key, master)
        micro_root = standard_spec.micro_root
        has_standard = family_key in family_weights

        if not self.enable_micro_routing or micro_root is None or not has_standard:
            # Routing disabled, no paired micro root, or a micro-only request:
            # round each explicitly requested root independently. Micro-only
            # requests are honoured as micro contracts and never upsampled.
            return [
                self._plain_leg(root, weight, specs[root], state, equity)
                for root, weight in family_weights.items()
            ]

        standard_price, standard_expiration = self._price_and_expiration(
            state, family_key, standard_spec
        )
        micro_spec = master.instruments.get(micro_root)
        if micro_spec is None:  # pragma: no cover - validated by InstrumentsConfig
            raise ValueError(f"instrument master is missing the micro root {micro_root!r}")
        # The paired standard and micro roots reference the same underlying, so when
        # the state does not carry a micro snapshot its mark price and expiration are
        # obtained from the standard contract (never a fabricated, arbitrary price).
        micro_snapshot = state.instruments.get(micro_root)
        if micro_snapshot is not None:
            micro_price = micro_snapshot.price
            micro_expiration = micro_snapshot.expiration
        else:
            micro_price = standard_price
            micro_expiration = standard_expiration

        family_notional = sum(family_weights.values()) * equity
        candidates = self._family_candidates(
            family_key,
            family_notional,
            micro_root,
            standard_spec,
            standard_price,
            standard_expiration,
            micro_spec,
            micro_price,
            micro_expiration,
        )
        return self._select_family_candidate(family_notional, candidates)

    def _plain_leg(
        self,
        root: InstrumentId,
        weight: float,
        spec: InstrumentSpec,
        state: MarketState,
        equity: float,
    ) -> _Leg:
        price, expiration = self._price_and_expiration(state, root, spec)
        raw = weight * equity / (price * spec.multiplier)
        return _Leg(root, spec, price, expiration, float(self._round_quantity(raw)))

    def _family_candidates(
        self,
        family_key: InstrumentId,
        family_notional: float,
        micro_root: InstrumentId,
        standard_spec: InstrumentSpec,
        standard_price: float,
        standard_expiration: date | None,
        micro_spec: InstrumentSpec,
        micro_price: float,
        micro_expiration: date | None,
    ) -> list[list[_Leg]]:
        """Enumerate deterministic same-direction decompositions of one family.

        Candidate decompositions are:

        1. ``configured``: the configured rounding of the standard-contract raw
           quantity plus a same-sign micro residual (no residual at all when the
           configured rounding overshoots the requested notional).
        2. ``truncated``: the standard leg truncated toward zero plus a same-sign
           micro residual, which removes overshoot while keeping the exposure
           direction intact.
        3. ``micro_only``: the family notional rounded directly into micro
           contracts (only relevant when the standard raw quantity is below one
           contract and the mode is not ``FLOOR``).

        ``_select_family_candidate`` then picks the decomposition with the
        smallest absolute notional tracking error, preferring fewer legs on ties.
        """
        standard_notional = standard_price * standard_spec.multiplier
        micro_notional = micro_price * micro_spec.multiplier
        if family_notional == 0.0 or micro_notional <= 0.0:
            return [[]]
        standard_raw = family_notional / standard_notional

        def build(standard_quantity: int) -> list[_Leg]:
            legs: list[_Leg] = []
            if standard_quantity != 0:
                legs.append(
                    _Leg(
                        family_key,
                        standard_spec,
                        standard_price,
                        standard_expiration,
                        float(standard_quantity),
                    )
                )
            residual = family_notional - standard_quantity * standard_notional
            micro_leg = self._same_sign_residual_micro(
                micro_root,
                micro_spec,
                micro_price,
                micro_expiration,
                residual,
                family_notional,
                micro_notional,
            )
            if micro_leg is not None:
                legs.append(micro_leg)
            return legs

        configured = self._round_quantity(standard_raw)
        candidates: list[list[_Leg]] = [build(configured)]
        truncated = self._truncate_quantity(standard_raw)
        if truncated != configured:
            candidates.append(build(truncated))
        if abs(standard_raw) < 1.0 and self.rounding_mode is not RoundingMode.FLOOR:
            micro_quantity = self._round_quantity(family_notional / micro_notional)
            if micro_quantity != 0 and abs(family_notional) >= (
                self.min_micro_notional_ratio * micro_notional
            ):
                candidates.append(
                    [
                        _Leg(
                            micro_root,
                            micro_spec,
                            micro_price,
                            micro_expiration,
                            float(micro_quantity),
                        )
                    ]
                )
        return candidates

    @staticmethod
    def _select_family_candidate(
        family_notional: float, candidates: list[list[_Leg]]
    ) -> list[_Leg]:
        """Choose the decomposition with minimal tracking error (fewest legs on ties)."""
        best: list[_Leg] = []
        best_key: tuple[float, int] | None = None
        for legs in candidates:
            achieved = sum(leg.raw_quantity * leg.price * leg.spec.multiplier for leg in legs)
            key = (abs(family_notional - achieved), len(legs))
            if best_key is None or key < best_key:
                best = legs
                best_key = key
        return best

    def _same_sign_residual_micro(
        self,
        micro_root: InstrumentId,
        micro_spec: InstrumentSpec,
        micro_price: float,
        micro_expiration: date | None,
        residual_notional: float,
        family_notional: float,
        micro_notional: float,
    ) -> _Leg | None:
        """Route a residual into the paired micro root, never against the family.

        A residual with the opposite sign of the requested family exposure would
        hedge the standard leg against itself (double commission/slippage and a
        self-cancelling basis position), so it is deliberately dropped instead.
        """
        if residual_notional == 0.0 or family_notional == 0.0:
            return None
        if (residual_notional > 0.0) != (family_notional > 0.0):
            return None
        if abs(residual_notional) < self.min_micro_notional_ratio * micro_notional:
            return None
        quantity = self._round_quantity(residual_notional / micro_notional)
        if quantity == 0:
            return None
        if (quantity > 0) != (family_notional > 0.0):  # pragma: no cover - sign preserving
            return None
        return _Leg(micro_root, micro_spec, micro_price, micro_expiration, float(quantity))

    @staticmethod
    def _truncate_quantity(raw: float) -> int:
        """Truncate a raw quantity toward zero."""
        if raw == 0.0 or not math.isfinite(raw):
            return 0
        magnitude = math.floor(abs(raw))
        return magnitude if raw > 0.0 else -magnitude

    @staticmethod
    def _family_standard_spec(
        family_key: InstrumentId, master: InstrumentsConfig
    ) -> InstrumentSpec:
        standard = master.instruments.get(family_key)
        if standard is None:  # pragma: no cover - validated by InstrumentsConfig
            raise ValueError(f"instrument master is missing the family root {family_key!r}")
        return standard

    @staticmethod
    def _price_and_expiration(
        state: MarketState, root: InstrumentId, spec: InstrumentSpec
    ) -> tuple[float, date | None]:
        """Mark price/expiration for ``root``, resolving a paired counterpart if needed.

        A standard contract and its micro pair reference the same underlying and
        therefore trade at the same mark price, so a market state that carries only
        one leg of the pair is still usable. When neither leg is present the
        translation is impossible and a clear ``ValueError`` is raised.
        """
        snapshot: InstrumentSnapshot | None = state.instruments.get(root)
        if snapshot is not None:
            return snapshot.price, snapshot.expiration
        counterpart_root = spec.standard_root or spec.micro_root
        if counterpart_root is not None:
            counterpart = state.instruments.get(counterpart_root)
            if counterpart is not None:
                return counterpart.price, counterpart.expiration
        raise ValueError(
            f"instrument {root!r} is not present in the market state; cannot translate its "
            "weight without a point-in-time mark price (and no standard/micro counterpart "
            "is available either)"
        )

    # -- rounding --------------------------------------------------------------

    def _round_quantity(self, raw: float) -> int:
        """Round a raw contract quantity according to the configured mode."""
        if raw == 0.0 or not math.isfinite(raw):
            return 0
        magnitude = abs(raw)
        sign = 1 if raw > 0.0 else -1
        if self.rounding_mode is RoundingMode.CEIL:
            return sign * math.ceil(magnitude)
        if self.rounding_mode is RoundingMode.FLOOR:
            return sign * math.floor(magnitude)
        base = math.floor(magnitude)
        fraction = magnitude - base
        threshold = 0.5 + 0.5 * min(self.rebalance_buffer_contracts, 1.0)
        return sign * (base + 1 if fraction >= threshold else base)

    # -- margin feasibility ----------------------------------------------------

    def _resolve_quantities(self, legs: list[_Leg], equity: float) -> dict[InstrumentId, int]:
        """Round legs and enforce the hard initial-margin utilisation ceiling."""
        quantities: dict[InstrumentId, int] = {}
        for leg in legs:
            quantities[leg.instrument_id] = self._round_quantity(leg.raw_quantity)
        if not legs:
            return quantities
        budget = self.max_margin_utilization * equity
        margin_by_root = {leg.instrument_id: leg.initial_margin for leg in legs}
        projected = self._projected_initial_margin(quantities, margin_by_root)
        if projected <= budget:
            return quantities

        raw = {leg.instrument_id: leg.raw_quantity for leg in legs}
        for _ in range(_MARGIN_SCALING_ITERATIONS):
            projected = self._projected_initial_margin(quantities, margin_by_root)
            if projected <= budget:
                return quantities
            factor = budget / projected if projected > 0.0 else 0.0
            for root in raw:
                raw[root] *= factor
            quantities = {root: self._round_quantity(value) for root, value in raw.items()}

        return self._drop_until_feasible(quantities, margin_by_root, budget)

    @staticmethod
    def _projected_initial_margin(
        quantities: dict[InstrumentId, int], margin_by_root: dict[InstrumentId, float]
    ) -> float:
        return sum(abs(quantity) * margin_by_root[root] for root, quantity in quantities.items())

    @staticmethod
    def _drop_until_feasible(
        quantities: dict[InstrumentId, int],
        margin_by_root: dict[InstrumentId, float],
        budget: float,
    ) -> dict[InstrumentId, int]:
        """Deterministically drop contracts until the margin ceiling holds.

        Drops the largest-margin leg first (lexicographic root tie-break), so the
        process is deterministic. Zero contracts require zero margin, therefore
        the loop always terminates with a feasible (possibly flat) target set.
        """
        drops = 0
        while True:
            projected = ContractTranslator._projected_initial_margin(quantities, margin_by_root)
            if projected <= budget:
                return quantities
            candidates = [
                (abs(quantity) * margin_by_root[root], root)
                for root, quantity in quantities.items()
                if quantity != 0
            ]
            if not candidates:  # pragma: no cover - zero contracts require zero margin
                return quantities
            _, root = sorted(candidates, key=lambda item: (-item[0], item[1]))[0]
            quantities[root] -= 1 if quantities[root] > 0 else -1
            drops += 1
            if drops > _MAX_GREEDY_DROPS:  # pragma: no cover - defensive guard
                return dict.fromkeys(quantities, 0)

    # -- output ----------------------------------------------------------------

    def _materialise(
        self, legs: list[_Leg], quantities: dict[InstrumentId, int]
    ) -> ContractTargetSet:
        targets: list[ContractTarget] = []
        for leg in legs:
            quantity = quantities.get(leg.instrument_id, 0)
            if quantity == 0:
                continue
            targets.append(
                ContractTarget(
                    instrument_id=leg.instrument_id,
                    quantity=quantity,
                    price=leg.price,
                    multiplier=leg.spec.multiplier,
                    expiration=leg.expiration,
                    initial_margin=leg.spec.margins.initial,
                    maintenance_margin=leg.spec.margins.maintenance,
                    contract_symbol=self._contract_symbol(leg.instrument_id, leg.expiration),
                    asset_class=leg.spec.asset_class,
                )
            )
        return ContractTargetSet(tuple(targets))

    @staticmethod
    def _contract_symbol(root: InstrumentId, expiration: date | None) -> str:
        """Deterministic contract symbol, e.g. ``ES_20240315``."""
        if expiration is None:
            return root
        return f"{root}_{expiration.strftime('%Y%m%d')}"
