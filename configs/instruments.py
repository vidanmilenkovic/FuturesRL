"""Instrument master: typed access to ``configs/instruments.yaml``.

The instrument master is the single source of truth for contract multipliers,
tick sizes, margin estimates, roll rules and standard <-> micro mappings. The
``ContractTranslator``, backtest simulators and IBKR contract resolver all read
from :class:`InstrumentsConfig`, so a spec change is a one-file change.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from src.core.types import AssetClass, RollTrigger, RoundingMode

CONFIGS_DIR: Path = Path(__file__).resolve().parent
DEFAULT_INSTRUMENTS_PATH: Path = CONFIGS_DIR / "instruments.yaml"


class MarginSpec(BaseModel):
    """Exchange/broker margin estimates for one contract, in account currency."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    initial: float = Field(ge=0.0)
    maintenance: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _maintenance_le_initial(self) -> MarginSpec:
        if self.maintenance > self.initial:
            raise ValueError("maintenance margin must not exceed initial margin")
        return self


class RollSpec(BaseModel):
    """Roll methodology for one futures root."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trigger: RollTrigger = RollTrigger.CALENDAR_DAYS_BEFORE_FND
    days_before_fnd: int = Field(default=5, ge=0, le=60)
    use_volume_crossover: bool = True
    contract_months: tuple[int, ...] = Field(default=(3, 6, 9, 12))
    physical_delivery: bool = False
    min_volume_ratio: float = Field(default=1.0, gt=0.0)

    @field_validator("contract_months")
    @classmethod
    def _valid_months(cls, months: tuple[int, ...]) -> tuple[int, ...]:
        if not months:
            raise ValueError("contract_months must not be empty")
        if any(not 1 <= month <= 12 for month in months):
            raise ValueError("contract_months must be within 1..12")
        return tuple(sorted(set(months)))


class InstrumentDefaults(BaseModel):
    """Fleet-wide defaults applied when an instrument omits a field."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    currency: str = "USD"
    price_precision: int = Field(default=2, ge=0, le=8)
    rounding_mode: RoundingMode = RoundingMode.NEAREST
    rebalance_buffer_contracts: float = Field(default=0.0, ge=0.0)
    max_margin_utilization: float = Field(default=0.60, gt=0.0, le=1.0)
    max_order_contracts: int = Field(default=200, gt=0)


class InstrumentSpec(BaseModel):
    """Point-in-time specification of one futures root."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    root: str
    name: str
    asset_class: AssetClass
    sector: str
    exchange: str
    multiplier: float = Field(gt=0.0)
    tick_size: float = Field(gt=0.0)
    currency: str = "USD"
    price_precision: int = Field(default=2, ge=0, le=8)
    databento_symbol: str
    ibkr_exchange: str
    micro_root: str | None = None
    standard_root: str | None = None
    #: IBKR one-way broker commission per contract for this exact root
    #: (standard ~0.85; micro equity/energy/metals/rates ~0.25; micro FX ~0.15).
    broker_commission_per_contract: float = Field(default=0.85, ge=0.0)
    #: Exchange + clearing + regulatory fees per contract per side for this exact
    #: root (standard CME equity ~1.40; micro MES/MNQ ~0.37; CL ~1.52; MCL ~0.52).
    exchange_and_reg_fee_per_contract: float = Field(default=1.70, ge=0.0)
    #: Baseline slippage (ticks per side) for a liquid outright order.
    default_slippage_ticks: float = Field(default=0.75, ge=0.0)
    margins: MarginSpec
    roll: RollSpec

    @field_validator("root")
    @classmethod
    def _upper_root(cls, root: str) -> str:
        root = root.strip().upper()
        if not root:
            raise ValueError("instrument root must be non-empty")
        return root

    @property
    def point_value(self) -> float:
        """Account-currency value of one full contract point."""
        return self.multiplier

    @property
    def transaction_fee_per_contract(self) -> float:
        """Broker commission + exchange/regulatory fees per contract per side."""
        return self.broker_commission_per_contract + self.exchange_and_reg_fee_per_contract

    @property
    def tick_value(self) -> float:
        """Account-currency value of one tick."""
        return self.multiplier * self.tick_size

    @property
    def is_micro(self) -> bool:
        """Whether this root is a micro contract."""
        return self.standard_root is not None

    def ticks_to_price(self, ticks: float) -> float:
        """Convert a number of ticks to a price delta."""
        return ticks * self.tick_size

    def price_to_ticks(self, delta: float) -> float:
        """Convert a price delta to (fractional) ticks."""
        return delta / self.tick_size


class InstrumentsConfig(BaseModel):
    """The full instrument master, with cross-reference validation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: int = Field(default=1, ge=1)
    defaults: InstrumentDefaults = InstrumentDefaults()
    instruments: dict[str, InstrumentSpec]

    @model_validator(mode="after")
    def _validate_cross_references(self) -> InstrumentsConfig:
        if not self.instruments:
            raise ValueError("instrument master must declare at least one instrument")
        for root, spec in self.instruments.items():
            if root != spec.root:
                raise ValueError(f"instrument key {root!r} does not match spec root {spec.root!r}")
            if spec.micro_root is not None:
                micro = self.instruments.get(spec.micro_root)
                if micro is None:
                    raise ValueError(f"{root}: unknown micro_root {spec.micro_root!r}")
                if micro.standard_root != root:
                    raise ValueError(
                        f"{root}: micro contract {spec.micro_root!r} must declare "
                        f"standard_root={root!r}, got {micro.standard_root!r}"
                    )
                if micro.multiplier >= spec.multiplier:
                    raise ValueError(f"{root}: micro multiplier must be smaller than standard")
            if spec.standard_root is not None:
                standard = self.instruments.get(spec.standard_root)
                if standard is None:
                    raise ValueError(f"{root}: unknown standard_root {spec.standard_root!r}")
                if standard.micro_root != root:
                    raise ValueError(
                        f"{root}: standard contract {spec.standard_root!r} must declare "
                        f"micro_root={root!r}"
                    )
        return self

    def get(self, root: str) -> InstrumentSpec:
        """Return the spec for ``root`` (case-insensitive), raising ``KeyError``."""
        key = root.strip().upper()
        try:
            return self.instruments[key]
        except KeyError as exc:
            raise KeyError(
                f"unknown futures root {root!r}; known: {sorted(self.instruments)}"
            ) from exc

    def has(self, root: str) -> bool:
        """Whether ``root`` is declared."""
        return root.strip().upper() in self.instruments

    def roots(self) -> tuple[str, ...]:
        """All declared roots in declaration order."""
        return tuple(self.instruments)

    def by_asset_class(self, asset_class: AssetClass) -> tuple[InstrumentSpec, ...]:
        """Specs belonging to one asset class."""
        return tuple(spec for spec in self.instruments.values() if spec.asset_class is asset_class)

    def by_sector(self, sector: str) -> tuple[InstrumentSpec, ...]:
        """Specs belonging to one risk-cluster sector (equity, rates, energy, ...)."""
        return tuple(spec for spec in self.instruments.values() if spec.sector == sector)

    def micro_for(self, root: str) -> InstrumentSpec | None:
        """Micro contract routing target for ``root``, if any."""
        spec = self.get(root)
        return self.instruments.get(spec.micro_root) if spec.micro_root else None

    def standard_for(self, root: str) -> InstrumentSpec | None:
        """Standard contract that a micro ``root`` rolls up to, if any."""
        spec = self.get(root)
        return self.instruments.get(spec.standard_root) if spec.standard_root else None

    def sectors(self) -> tuple[str, ...]:
        """Distinct risk-cluster sectors in declaration order."""
        seen: dict[str, None] = {}
        for spec in self.instruments.values():
            seen.setdefault(spec.sector, None)
        return tuple(seen)


def parse_instruments_config(payload: dict[str, object]) -> InstrumentsConfig:
    """Parse and validate a raw YAML payload into the instrument master.

    Instrument keys in YAML are shorthand; the ``root`` field is injected from
    the mapping key so the same spec can never be declared under two roots.
    """
    raw_instruments = payload.get("instruments")
    if not isinstance(raw_instruments, dict):
        raise ValueError("instruments.yaml must contain an 'instruments' mapping")
    instruments: dict[str, dict[str, object]] = {}
    for root, raw in raw_instruments.items():
        if not isinstance(root, str) or not isinstance(raw, dict):
            raise ValueError(f"invalid instrument entry for {root!r}")
        entry = dict(raw)
        entry.setdefault("root", root)
        instruments[root] = entry
    return InstrumentsConfig.model_validate(
        {
            "version": payload.get("version", 1),
            "defaults": payload.get("defaults", {}),
            "instruments": instruments,
        }
    )


@lru_cache(maxsize=8)
def _load_instruments_cached(path: Path) -> InstrumentsConfig:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return parse_instruments_config(payload)


def load_instruments(path: Path | str | None = None) -> InstrumentsConfig:
    """Load (and cache) the instrument master from ``path``.

    Defaults to ``configs/instruments.yaml``; genuinely missing files raise a
    ``FileNotFoundError`` rather than silently degrading (a missing instrument
    master must never be interpreted as "no tradable instruments").
    """
    resolved = Path(path) if path is not None else DEFAULT_INSTRUMENTS_PATH
    resolved = resolved.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"instrument master not found: {resolved}")
    return _load_instruments_cached(resolved)


def clear_instruments_cache() -> None:
    """Clear the module cache (test/configuration-reload helper)."""
    _load_instruments_cached.cache_clear()
