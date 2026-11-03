"""Runner-bridge -> pipeline waterfall mapping (Phase-10 Part A1).

The live/paper runner publishes a per-instrument weight matrix on the bridge
(exec-hardening task-23) so ``GET /api/pipeline/latest`` can render the *exact*
same Weight matrix with ``data_source="runner-bridge"`` as the engine waterfall:
``w_base``, ``w_timing``, ``w_risk``, ``w_final``, ``weight_actual``,
``target_contracts``, ``notional``, ``driver`` and ``regime``.

Honest-fallback contract
------------------------
* Only a **fresh** bridge snapshot (per ``stale_after_seconds``) is consumed.
* If the snapshot carries the flat scalar summary but no per-instrument rows, the
  mapper returns ``None`` and the endpoint keeps the existing compact bridge
  summary / engine waterfall instead of fabricating rows.
* Fields are read through small alias sets so an additive bridge schema update
  cannot break the dashboard; a row missing its weights is skipped, never zeroed.

Expected bridge keys (documented contract with exec-hardening task-23), inside
``snapshot.pipeline``::

    "instruments"/"instrument_rows"/"weight_matrix": [
        {"instrument_id"|"root_symbol": "ES",
         "weight_base"|"w_base": 0.10,
         "weight_timing"|"w_timing"/"timing": 0.08,
         "weight_risk"|"w_risk"/"risk": 0.07,
         "weight_final"|"w_final"/"weighted"/"weight": 0.07,
         "weight_actual"|"achieved_weight": 0.07,
         "target_contracts"|"contracts"|"quantity": 1,
         "notional": 336000.0,
         "driver"|"reason": "...",
         "regime": "contango",
         "asset_class": "equity_index",
         "selected": true}
    ]
    "tracking_error_pct": 1.25,
    "tracking_error_source": "translator",
    "micro_routing": [{"family","standard_root","standard_quantity","micro_root","micro_quantity"}]
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast

from configs.instruments import InstrumentsConfig

from src.api.live import micro_routing_from_payload
from src.api.schemas import (
    PipelineDiagnostic,
    PipelineInstrumentRow,
    PipelineStage,
    PipelineWaterfallResponse,
)
from src.core.types import AssetClass

logger = logging.getLogger(__name__)

__all__ = [
    "BridgeInstrumentWeight",
    "BridgePipelineService",
    "extract_bridge_instruments",
    "pipeline_from_bridge_payload",
]

#: Bridge payload keys carrying the per-instrument weight matrix, in priority order.
_ROW_KEYS: tuple[str, ...] = (
    "instruments",
    "instrument_rows",
    "weight_matrix",
    "pipeline_instruments",
    "per_instrument",
)

_WEIGHT_BASE_KEYS: tuple[str, ...] = ("weight_base", "w_base", "base_weight", "base")
_WEIGHT_TIMING_KEYS: tuple[str, ...] = ("weight_timing", "w_timing", "timing_weight", "timing")
_WEIGHT_RISK_KEYS: tuple[str, ...] = ("weight_risk", "w_risk", "risk_weight", "risk")
_WEIGHT_FINAL_KEYS: tuple[str, ...] = (
    "weight_final",
    "w_final",
    "weighted",
    "weight",
    "final_weight",
    "final",
)
_WEIGHT_ACTUAL_KEYS: tuple[str, ...] = ("weight_actual", "achieved_weight", "actual_weight")
_CONTRACTS_KEYS: tuple[str, ...] = (
    "target_contracts",
    "contracts",
    "quantity",
    "target_quantity",
    "n_contracts",
)
_DRIVER_KEYS: tuple[str, ...] = ("driver", "reason", "note", "explanation")
_REGIME_KEYS: tuple[str, ...] = ("regime", "market_regime")


def _mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return cast(Mapping[str, object], value)
    for attribute in ("model_dump", "to_dict"):
        dumper = getattr(value, attribute, None)
        if callable(dumper):
            try:
                dumped = dumper()
            except Exception:  # pragma: no cover - defensive adapter
                return {}
            return cast(Mapping[str, object], dumped) if isinstance(dumped, Mapping) else {}
    return {}


def _sequence(value: object) -> list[object]:
    if isinstance(value, Mapping):
        for key in _ROW_KEYS:
            if key in cast(Mapping[str, object], value):
                return _sequence(cast(Mapping[str, object], value)[key])
        return []
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return list(cast(Sequence[object], value))
    return []


def _text(value: object, default: str = "") -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if value is None:
        return default
    return str(value)


def _float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, str):
        try:
            result = float(value.strip())
        except ValueError:
            return None
        return result if math.isfinite(result) else None
    return None


def _first_float(mapping: Mapping[str, object], keys: Sequence[str]) -> float | None:
    for key in keys:
        resolved = _float(mapping.get(key))
        if resolved is not None:
            return resolved
    return None


def _int(value: object) -> int | None:
    number = _float(value)
    return None if number is None else round(number)


def _first_int(mapping: Mapping[str, object], keys: Sequence[str]) -> int | None:
    for key in keys:
        resolved = _int(mapping.get(key))
        if resolved is not None:
            return resolved
    return None


def _first_text(mapping: Mapping[str, object], keys: Sequence[str]) -> str:
    for key in keys:
        resolved = _text(mapping.get(key))
        if resolved:
            return resolved
    return ""


def _asset_class(value: object, root: str, master: InstrumentsConfig) -> AssetClass:
    text = _text(value).lower()
    try:
        return AssetClass(text)
    except ValueError:
        pass
    if master.has(root):
        return master.get(root).asset_class
    return AssetClass.UNKNOWN


@dataclass(frozen=True, slots=True)
class BridgeInstrumentWeight:
    """One per-instrument weight-matrix row published on the runner bridge."""

    instrument_id: str
    asset_class: AssetClass
    selected: bool
    weight_base: float
    weight_timing: float
    weight_risk: float
    weight_final: float
    weight_actual: float | None = None
    target_contracts: int | None = None
    notional: float | None = None
    driver: str = ""
    regime: str | None = None

    @property
    def delta_timing(self) -> float:
        return self.weight_timing - self.weight_base

    @property
    def delta_risk(self) -> float:
        return self.weight_risk - self.weight_timing


def extract_bridge_instruments(
    payload: Mapping[str, object], *, master: InstrumentsConfig
) -> list[BridgeInstrumentWeight]:
    """Per-instrument rows from a bridge pipeline payload (empty when absent)."""
    entries: list[object] = []
    for key in _ROW_KEYS:
        candidate = payload.get(key)
        if candidate is not None:
            entries = _sequence(candidate)
            if entries:
                break
    rows: list[BridgeInstrumentWeight] = []
    for entry in entries:
        mapping = _mapping(entry)
        if not mapping:
            continue
        root = _first_text(
            mapping, ("instrument_id", "root_symbol", "root", "symbol", "id")
        ).upper()
        weight_final = _first_float(mapping, _WEIGHT_FINAL_KEYS)
        weight_risk = _first_float(mapping, _WEIGHT_RISK_KEYS)
        if root == "" or (weight_final is None and weight_risk is None):
            continue
        base = _first_float(mapping, _WEIGHT_BASE_KEYS)
        timing = _first_float(mapping, _WEIGHT_TIMING_KEYS)
        risk = weight_risk if weight_risk is not None else weight_final
        final = weight_final if weight_final is not None else risk
        assert base is not None or timing is not None or final is not None
        base_value = base if base is not None else 0.0
        timing_value = timing if timing is not None else base_value
        final_value = final if final is not None else timing_value
        selected = mapping.get("selected")
        rows.append(
            BridgeInstrumentWeight(
                instrument_id=root,
                asset_class=_asset_class(mapping.get("asset_class"), root, master),
                selected=bool(selected) if selected is not None else abs(final_value) > 0.0,
                weight_base=base_value,
                weight_timing=timing_value,
                weight_risk=risk if risk is not None else final_value,
                weight_final=final_value,
                weight_actual=_first_float(mapping, _WEIGHT_ACTUAL_KEYS),
                target_contracts=_first_int(mapping, _CONTRACTS_KEYS),
                notional=_float(mapping.get("notional")),
                driver=_first_text(mapping, _DRIVER_KEYS),
                regime=_first_text(mapping, _REGIME_KEYS) or None,
            )
        )
    return rows


def _universe(payload: Mapping[str, object], rows: Sequence[BridgeInstrumentWeight]) -> list[str]:
    raw = payload.get("universe")
    if isinstance(raw, str) and raw.strip():
        return [root.strip().upper() for root in raw.split(",") if root.strip()]
    if isinstance(raw, Sequence) and not isinstance(raw, str | bytes):
        return [
            str(root).strip().upper() for root in cast(Sequence[object], raw) if str(root).strip()
        ]
    return [row.instrument_id for row in rows]


def _stage(
    key: str,
    label: str,
    formula: str,
    *,
    gross: float,
    net: float,
    instrument_count: int,
    note: str,
) -> PipelineStage:
    return PipelineStage(
        key=cast("object", key),  # type: ignore[arg-type]
        label=label,
        formula=formula,
        gross=round(gross, 8),
        net=round(net, 8),
        instrument_count=instrument_count,
        note=note,
    )


def pipeline_from_bridge_payload(
    payload: Mapping[str, object],
    *,
    master: InstrumentsConfig,
    timestamp: datetime,
    sequence: int | None = None,
) -> PipelineWaterfallResponse | None:
    """Map a bridge pipeline payload onto the waterfall contract.

    Returns ``None`` when the payload does not carry per-instrument rows (the
    caller then keeps the compact bridge summary or the engine waterfall).
    """
    rows = extract_bridge_instruments(payload, master=master)
    if not rows:
        return None
    rows.sort(key=lambda row: row.instrument_id)
    universe = _universe(payload, rows)
    gross_base = sum(abs(row.weight_base) for row in rows)
    gross_timing = sum(abs(row.weight_timing) for row in rows)
    gross_risk = sum(abs(row.weight_risk) for row in rows)
    gross_final = sum(abs(row.weight_final) for row in rows)
    net_base = sum(row.weight_base for row in rows)
    net_timing = sum(row.weight_timing for row in rows)
    net_risk = sum(row.weight_risk for row in rows)
    net_final = sum(row.weight_final for row in rows)
    tracking = _float(payload.get("tracking_error_pct")) or 0.0
    tracking_raw = _text(payload.get("tracking_error_source"))
    micro_payload = payload.get("micro_routing")
    if micro_payload is None:
        micro_payload = payload.get("micro_routes")
    micro = micro_routing_from_payload(micro_payload)
    target_count = sum(abs(row.target_contracts or 0) for row in rows)
    elapsed = _float(payload.get("elapsed_ms")) or 0.0
    instruments = [
        PipelineInstrumentRow(
            instrument_id=row.instrument_id,
            asset_class=row.asset_class,
            selected=row.selected,
            weight_base=round(row.weight_base, 8),
            weight_timing=round(row.weight_timing, 8),
            weight_risk=round(row.weight_risk, 8),
            weight_final=round(row.weight_final, 8),
            weight_actual=None if row.weight_actual is None else round(row.weight_actual, 8),
            delta_timing=round(row.delta_timing, 8),
            delta_risk=round(row.delta_risk, 8),
            target_contracts=row.target_contracts,
            notional=row.notional,
            driver=row.driver,
            regime=row.regime,
            reason=row.driver,
        )
        for row in rows
    ]
    stages = [
        _stage(
            "universe",
            "S_t · Universe",
            "S_t(X≤t) → roots",
            gross=0.0,
            net=0.0,
            instrument_count=len(universe),
            note="runner-bridge snapshot",
        ),
        _stage(
            "base",
            "w_base · Allocation",
            "A_t(candidates)",
            gross=gross_base,
            net=net_base,
            instrument_count=sum(1 for row in rows if row.weight_base != 0.0),
            note="runner-bridge snapshot",
        ),
        _stage(
            "timing",
            "w_timing · Overlay",
            "T_t(w_base)",
            gross=gross_timing,
            net=net_timing,
            instrument_count=sum(1 for row in rows if row.weight_timing != 0.0),
            note="runner-bridge snapshot",
        ),
        _stage(
            "risk",
            "w_risk · Approved",
            "R_t(w_timing) → w_t",
            gross=gross_risk,
            net=net_risk,
            instrument_count=sum(1 for row in rows if row.weight_risk != 0.0),
            note="runner-bridge snapshot",
        ),
        _stage(
            "targets",
            "N_i,t · Contracts",
            "N = w_t·E_t / (P·M)",
            gross=gross_final,
            net=net_final,
            instrument_count=target_count,
            note="runner-bridge target book",
        ),
    ]
    diagnostics = [
        PipelineDiagnostic(
            stage="Live runner bridge",
            detail=(
                f"{len(rows)} per-instrument rows published by the live runner "
                f"(sequence {sequence or 0}); weights identical to the engine waterfall"
            ),
            value=f"{len(rows)} roots",
            status="info",
        )
    ]
    return PipelineWaterfallResponse(
        timestamp=timestamp,
        cycle_id=f"bridge-{sequence or 0}",
        universe=universe,
        stages=stages,
        instruments=instruments,
        diagnostics=diagnostics,
        elapsed_ms=round(elapsed, 3),
        data_source="runner-bridge",
        tracking_error_pct=max(0.0, tracking),
        micro_routing=micro,
        tracking_error_source="translator" if tracking_raw == "translator" else "fallback",
    )


class BridgePipelineService:
    """Bridge-first waterfall provider used by ``/api/pipeline/latest``.

    Reads the exact same ``RunnerBridgeStore`` snapshot as the live-state service,
    but only consumes it when it is fresh and carries per-instrument rows.
    """

    def __init__(
        self, *, settings: object, clock: object, instrument_master: InstrumentsConfig
    ) -> None:
        from src.api.live import RunnerBridgeReader

        self._reader = RunnerBridgeReader(settings, clock=clock)  # type: ignore[arg-type]
        self._master = instrument_master

    def pipeline(self) -> PipelineWaterfallResponse | None:
        """Fresh runner-bridge matrix waterfall, or ``None`` for honest fallback.

        The final bridge schema (exec-hardening task-23) publishes``instruments``,
        ``micro_routing`` and the tracking fields **top-level** on the snapshot while
        ``pipeline`` keeps the flat scalar summary; this method merges both views.
        Old Phase-9 snapshots with ``instruments=[]`` return ``None`` so the caller
        falls back to the compact summary / engine waterfall.
        """
        snapshot = self._reader.load()
        if snapshot is None or snapshot.stale:
            return None
        payload: dict[str, object] = dict(snapshot.mapping("pipeline"))
        rows = snapshot.sequence_field("instruments")
        if not rows:
            return None
        payload["instruments"] = rows
        for key in (
            "micro_routing",
            "tracking_error_pct",
            "tracking_error_source",
            "universe",
            "regime",
            "elapsed_ms",
        ):
            value = snapshot.get(key)
            if value is not None and key not in payload:
                payload[key] = value
        return pipeline_from_bridge_payload(
            payload,
            master=self._master,
            timestamp=snapshot.updated_at or datetime.now(UTC),
            sequence=snapshot.sequence,
        )
