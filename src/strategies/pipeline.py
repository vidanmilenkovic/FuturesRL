"""FinRL-X composable strategy pipeline (``w_t = R_t(T_t(A_t(S_t(X))))``).

:class:`ComposablePipeline` is the narrow implementation of the FinRL-X
weight-centric contract: every rebalance cycle threads an immutable
:class:`~src.core.types.MarketState` through four contract-preserving stages and
emits a fully auditable :class:`~src.core.types.PipelineResult`:

    ``universe = S_t(state)``
    ``w_base   = A_t(universe, state)``
    ``w_timing = T_t^1(T_t^2(... A_t ...))``      (configured order)
    ``w_risk   = R_t^1(R_t^2(... w_timing ...))`` (configured order)
    ``w_t      = sanitize(w_risk)``
    ``targets  = translator.translate(w_t, state)``  (when a translator is injected)

Guarantees (all enforced in :meth:`run`, never silently repaired):

* every stage output must be a :class:`WeightVector` (finite weights by
  construction) referring only to instruments present in the state **and** in the
  selector's universe — a stage cannot smuggle in an unselected root;
* a selector returning a duplicate or unknown instrument is rejected;
* weights below ``negligible_weight`` (default ``1e-9``) are removed from the
  final book so the waterfall and the contract translator never see numerical
  dust;
* every stage is timed and annotated in ``PipelineResult.diagnostics`` for the
  FinRL-X waterfall inspector.

Risk overlays are **injected**, not imported: ``src/risk`` owns the LLM Risk-First
and quantitative overlays, and this module never imports it (that keeps the
strategy package free of circular dependencies and lets tests inject a
deterministic fake). ``build_pipeline`` matches injected overlays to the
``risk:`` stages declared in the YAML by name.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from configs.instruments import InstrumentsConfig
from configs.strategy import StrategyConfig

from src.core.interfaces import (
    ContractTranslatorProtocol,
    PortfolioAllocator,
    RiskOverlay,
    StrategyPipeline,
    TimingOverlay,
    UniverseSelector,
)
from src.core.types import (
    ContractTargetSet,
    InstrumentId,
    MarketState,
    MetadataValue,
    PipelineResult,
    WeightVector,
)
from src.strategies._numeric import require_known_instruments, sanitize_weights
from src.strategies.allocators import build_allocator
from src.strategies.selectors import build_selector
from src.strategies.timing import build_timing

#: Weights at or below this magnitude are dropped from the final book.
DEFAULT_NEGLIGIBLE_WEIGHT: float = 1e-9


class IdentityRiskOverlay:
    """A real pass-through ``R_t`` stage (no risk constraints applied).

    Useful for tests, benchmarks, ablations and for running a declarative YAML
    that declares risk stages before ``src/risk`` overlays are wired in. It is a
    genuine implementation of the :class:`RiskOverlay` protocol, not a stub: it
    returns the very same ``WeightVector`` object, so it is allocation-free and
    provably identity.
    """

    __slots__ = ()

    @property
    def name(self) -> str:
        """Registry name of this overlay."""
        return "identity"

    def apply_risk(self, weights: WeightVector, state: MarketState) -> WeightVector:
        """Return ``weights`` unchanged."""
        del state  # the identity overlay is state independent
        return weights


class ComposablePipeline:
    """``w_t = R_t(T_t(A_t(S_t(X))))`` executed stage by stage, with a trace."""

    __slots__ = (
        "_allocator",
        "_name",
        "_negligible_weight",
        "_risk_overlays",
        "_selector",
        "_skipped_risk_stages",
        "_timing_overlays",
        "_translator",
        "_unused_risk_overlays",
    )

    def __init__(
        self,
        *,
        name: str,
        selector: UniverseSelector,
        allocator: PortfolioAllocator,
        timing: Sequence[TimingOverlay] = (),
        risk: Sequence[RiskOverlay] = (),
        translator: ContractTranslatorProtocol | None = None,
        skipped_risk_stages: Sequence[str] = (),
        unused_risk_overlays: Sequence[str] = (),
        negligible_weight: float = DEFAULT_NEGLIGIBLE_WEIGHT,
    ) -> None:
        if not name or not name.strip():
            raise ValueError("pipeline name must be a non-empty string")
        if not isinstance(selector, UniverseSelector):
            raise TypeError(
                f"selector must satisfy the UniverseSelector protocol, got {type(selector)!r}"
            )
        if not isinstance(allocator, PortfolioAllocator):
            raise TypeError(
                f"allocator must satisfy the PortfolioAllocator protocol, got {type(allocator)!r}"
            )
        timing_overlays = tuple(timing)
        for timing_overlay in timing_overlays:
            if not isinstance(timing_overlay, TimingOverlay):
                raise TypeError(
                    f"timing overlay {type(timing_overlay)!r} does not satisfy the "
                    "TimingOverlay protocol"
                )
        risk_overlays = tuple(risk)
        for risk_overlay in risk_overlays:
            if not isinstance(risk_overlay, RiskOverlay):
                raise TypeError(
                    f"risk overlay {type(risk_overlay)!r} does not satisfy the RiskOverlay protocol"
                )
        if negligible_weight < 0.0:
            raise ValueError("negligible_weight must be non-negative")
        self._name = name.strip()
        self._selector = selector
        self._allocator = allocator
        self._timing_overlays = timing_overlays
        self._risk_overlays = risk_overlays
        self._translator = translator
        self._skipped_risk_stages = tuple(skipped_risk_stages)
        self._unused_risk_overlays = tuple(unused_risk_overlays)
        self._negligible_weight = float(negligible_weight)

    # -- introspection --------------------------------------------------------

    @property
    def name(self) -> str:
        """Pipeline name (matches the declarative YAML ``id``)."""
        return self._name

    @property
    def selector(self) -> UniverseSelector:
        """The ``S_t`` universe selector."""
        return self._selector

    @property
    def allocator(self) -> PortfolioAllocator:
        """The ``A_t`` portfolio allocator."""
        return self._allocator

    @property
    def timing_overlays(self) -> tuple[TimingOverlay, ...]:
        """The ``T_t`` overlays, in application order."""
        return self._timing_overlays

    @property
    def risk_overlays(self) -> tuple[RiskOverlay, ...]:
        """The injected ``R_t`` overlays, in application order."""
        return self._risk_overlays

    @property
    def translator(self) -> ContractTranslatorProtocol | None:
        """The injected contract translator (``None`` when not wired)."""
        return self._translator

    @property
    def stage_names(self) -> tuple[str, ...]:
        """Human-readable stage chain for telemetry (``S`` then ``A``/``T``/``R``)."""
        timing = [
            f"timing[{index}].{overlay.name}" for index, overlay in enumerate(self._timing_overlays)
        ]
        risk = [
            f"risk[{index}].{overlay.name}" for index, overlay in enumerate(self._risk_overlays)
        ]
        return (
            f"selector.{self._selector.name}",
            f"allocator.{self._allocator.name}",
            *timing,
            *risk,
        )

    # -- validation -----------------------------------------------------------

    def _validate_universe(self, selected: object, state: MarketState) -> tuple[InstrumentId, ...]:
        if not isinstance(selected, Sequence):
            raise TypeError(
                "universe selector must return a sequence of instrument ids, "
                f"got {type(selected).__name__}"
            )
        universe: list[InstrumentId] = []
        seen: set[InstrumentId] = set()
        for instrument_id in selected:
            if not isinstance(instrument_id, str) or not instrument_id:
                raise ValueError(
                    f"selector.{self._selector.name}: invalid instrument id {instrument_id!r}"
                )
            if instrument_id not in state.instruments:
                raise ValueError(
                    f"selector.{self._selector.name}: selected unknown instrument {instrument_id!r}"
                )
            if instrument_id in seen:
                raise ValueError(
                    f"selector.{self._selector.name}: duplicate instrument {instrument_id!r}"
                )
            seen.add(instrument_id)
            universe.append(instrument_id)
        return tuple(universe)

    def _validate_stage_output(
        self,
        stage: str,
        output: object,
        universe: Sequence[InstrumentId],
        state: MarketState,
    ) -> WeightVector:
        if not isinstance(output, WeightVector):
            raise TypeError(
                f"{stage}: stage must return a WeightVector, got {type(output).__name__}"
            )
        require_known_instruments(output.weights, state, stage=stage)
        permitted = set(universe)
        extraneous = sorted(key for key in output.weights if key not in permitted)
        if extraneous:
            raise ValueError(
                f"{stage}: produced weights outside the selected universe: {extraneous!r}"
            )
        return output

    # -- execution ------------------------------------------------------------

    def run(self, state: MarketState) -> PipelineResult:
        """Execute one full rebalance cycle and return the staged trace."""
        if not isinstance(state, MarketState):
            raise TypeError(f"state must be a MarketState, got {type(state)!r}")
        started = time.perf_counter()
        stage_notes: dict[str, str] = {}
        stage_timings: dict[str, float] = {}

        # -- S_t: universe selection ---------------------------------------
        segment = time.perf_counter()
        universe = self._validate_universe(self._selector.select(state), state)
        selector_key = f"selector.{self._selector.name}"
        stage_timings[selector_key] = _elapsed_ms(segment)
        stage_notes[selector_key] = (
            f"{len(state.instruments)} available -> {len(universe)} selected"
        )

        base_weights = WeightVector.empty()
        timing_weights = WeightVector.empty()
        risk_weights = WeightVector.empty()
        final_weights = WeightVector.empty()
        targets: ContractTargetSet | None = None
        sanitised = 0
        overlays_skipped = not universe

        if universe:
            # -- A_t: base allocation --------------------------------------
            segment = time.perf_counter()
            allocator_key = f"allocator.{self._allocator.name}"
            base_output = self._allocator.allocate(universe, state)
            base_weights = self._validate_stage_output(allocator_key, base_output, universe, state)
            stage_timings[allocator_key] = _elapsed_ms(segment)
            stage_notes[allocator_key] = (
                f"{len(base_weights)} legs, gross={base_weights.gross:.6f}, "
                f"net={base_weights.net:+.6f}"
            )

            # -- T_t: timing overlays, in configured order -----------------
            timing_weights = base_weights
            for index, timing_overlay in enumerate(self._timing_overlays):
                segment = time.perf_counter()
                key = f"timing[{index}].{timing_overlay.name}"
                timing_output = timing_overlay.adjust_timing(timing_weights, state)
                previous_gross = timing_weights.gross
                timing_weights = self._validate_stage_output(key, timing_output, universe, state)
                stage_timings[key] = _elapsed_ms(segment)
                stage_notes[key] = f"gross {previous_gross:.6f} -> {timing_weights.gross:.6f}"

            # -- R_t: injected risk overlays, in configured order ----------
            risk_weights = timing_weights
            for index, risk_overlay in enumerate(self._risk_overlays):
                segment = time.perf_counter()
                key = f"risk[{index}].{risk_overlay.name}"
                risk_output = risk_overlay.apply_risk(risk_weights, state)
                previous_gross = risk_weights.gross
                risk_weights = self._validate_stage_output(key, risk_output, universe, state)
                stage_timings[key] = _elapsed_ms(segment)
                stage_notes[key] = f"gross {previous_gross:.6f} -> {risk_weights.gross:.6f}"

            # -- final hygiene + contract translation ----------------------
            final_weights = sanitize_weights(
                risk_weights,
                threshold=self._negligible_weight,
                allowed=universe,
            )
            sanitised = len(risk_weights) - len(final_weights)

        if self._translator is not None:
            targets = self._translator.translate(final_weights, state, equity=state.equity)

        diagnostics: dict[str, MetadataValue] = {
            "pipeline": self._name,
            "selector": self._selector.name,
            "allocator": self._allocator.name,
            "timing_overlays": tuple(overlay.name for overlay in self._timing_overlays),
            "risk_overlays": tuple(overlay.name for overlay in self._risk_overlays),
            "skipped_risk_stages": self._skipped_risk_stages,
            "unused_risk_overlays": self._unused_risk_overlays,
            "universe_size": len(universe),
            "empty_universe": not universe,
            "overlays_skipped": overlays_skipped,
            "stage_notes": stage_notes,
            "stage_timings_ms": stage_timings,
            "base_gross": base_weights.gross,
            "timing_gross": timing_weights.gross,
            "risk_gross": risk_weights.gross,
            "final_gross": final_weights.gross,
            "final_net": final_weights.net,
            "sanitised_weights": sanitised,
            "targets_attached": targets is not None,
        }
        return PipelineResult(
            timestamp=state.timestamp,
            universe=universe,
            base_weights=base_weights,
            timing_weights=timing_weights,
            risk_weights=risk_weights,
            final_weights=final_weights,
            targets=targets,
            diagnostics=diagnostics,
            elapsed_ms=_elapsed_ms(started),
        )


def _elapsed_ms(started: float) -> float:
    """Wall-clock milliseconds since ``started`` (never negative)."""
    return max((time.perf_counter() - started) * 1000.0, 0.0)


def _resolve_risk_overlays(
    declared: Sequence[str],
    provided: Sequence[RiskOverlay],
) -> tuple[tuple[RiskOverlay, ...], tuple[str, ...], tuple[str, ...]]:
    """Match injected overlays to the declared YAML risk stages by name.

    Returns ``(overlays_in_order, skipped_stage_names, unused_overlay_names)``.
    When the configured names do not match any injected overlay (callers may name
    their overlays differently), the injected overlays are used as-is in the
    provided order: the caller injected them for a reason, and silently dropping
    every ``R_t`` stage would be far more dangerous than an order mismatch.
    """
    matched: list[RiskOverlay] = []
    used: set[int] = set()
    skipped: list[str] = []
    for stage_name in declared:
        found: RiskOverlay | None = None
        for index, candidate in enumerate(provided):
            if index in used:
                continue
            if candidate.name == stage_name:
                found = candidate
                used.add(index)
                break
        if found is None:
            skipped.append(stage_name)
        else:
            matched.append(found)
    if not matched and provided:
        # No name matched: honour the caller's injection order, but still report
        # the declared stage names as unmatched so the waterfall is honest.
        return tuple(provided), tuple(skipped), ()
    unused = tuple(overlay.name for index, overlay in enumerate(provided) if index not in used)
    return tuple(matched), tuple(skipped), unused


def build_pipeline(
    config: StrategyConfig,
    instruments: InstrumentsConfig,
    translator: ContractTranslatorProtocol | None = None,
    risk_overlays: Sequence[RiskOverlay] | None = None,
) -> ComposablePipeline:
    """Build a :class:`ComposablePipeline` from a declarative strategy config.

    The universe selector, allocator and timing overlays are resolved from the
    YAML stage names via their registries. Risk overlays are injected (owned by
    ``src/risk``); a declared ``risk:`` stage with no matching injected overlay
    is recorded in the pipeline diagnostics as skipped so the waterfall shows
    that the final book was produced without it.
    """
    selector = build_selector(config.universe.name, config.universe.params, instruments)
    allocator = build_allocator(config.allocator.name, config.allocator.params)
    timing = tuple(build_timing(stage.name, stage.params) for stage in config.timing)
    provided = tuple(risk_overlays or ())
    risk, skipped, unused = _resolve_risk_overlays(
        tuple(stage.name for stage in config.risk), provided
    )
    return ComposablePipeline(
        name=config.id,
        selector=selector,
        allocator=allocator,
        timing=timing,
        risk=risk,
        translator=translator,
        skipped_risk_stages=skipped,
        unused_risk_overlays=unused,
    )


__all__ = [
    "DEFAULT_NEGLIGIBLE_WEIGHT",
    "ComposablePipeline",
    "IdentityRiskOverlay",
    "StrategyPipeline",
    "build_pipeline",
]
