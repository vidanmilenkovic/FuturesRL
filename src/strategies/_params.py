"""Typed parsing of declarative stage parameters.

Strategy pipelines are declared in YAML as a sequence of stages, each holding a
flat ``name -> scalar`` parameter mapping (see :mod:`configs.strategy`). The
:class:`ParamReader` is the single place that turns those untyped scalars into
the typed keyword arguments the strategy classes expect:

* conversions are explicit (``float``/``int``/``bool``/``str``/comma-separated
  sequences) and raise :class:`ParamError` with the stage name on bad input;
* every accessor records the key it consumed, so :meth:`ParamReader.check`
  rejects leftovers: a typo in a strategy YAML fails loudly instead of being
  silently ignored (a silently ignored ``top_n`` is a trading bug, not a typo).

The reader never mutates the caller's mapping and is safe to share: it only
records consumed keys.
"""

from __future__ import annotations

from collections.abc import Mapping

from configs.strategy import StageParam

type StageParams = Mapping[str, StageParam]


class ParamError(ValueError):
    """Raised when a declarative stage parameter is invalid or unknown."""


def _as_float(stage: str, name: str, value: StageParam) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError as exc:
            raise ParamError(f"{stage}.{name}: expected a number, got {value!r}") from exc
    raise ParamError(f"{stage}.{name}: expected a number, got {value!r}")


def _as_int(stage: str, name: str, value: StageParam) -> int:
    if isinstance(value, bool):
        raise ParamError(f"{stage}.{name}: expected an integer, got boolean {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not float(value).is_integer():
            raise ParamError(f"{stage}.{name}: expected an integer, got {value!r}")
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        try:
            return int(text)
        except ValueError as exc:
            raise ParamError(f"{stage}.{name}: expected an integer, got {value!r}") from exc
    raise ParamError(f"{stage}.{name}: expected an integer, got {value!r}")


def _as_bool(stage: str, name: str, value: StageParam) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalised = value.strip().lower()
        if normalised in {"true", "1", "yes", "on"}:
            return True
        if normalised in {"false", "0", "no", "off"}:
            return False
    raise ParamError(f"{stage}.{name}: expected a boolean, got {value!r}")


def _as_sequence(stage: str, name: str, value: StageParam) -> tuple[str, ...]:
    if not isinstance(value, str):
        raise ParamError(
            f"{stage}.{name}: expected a comma-separated string or a list, got {value!r}"
        )
    if not value.strip():
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


class ParamReader:
    """Read one stage's parameters while tracking consumed keys."""

    __slots__ = ("_consumed", "_params", "_stage")

    def __init__(self, stage: str, params: Mapping[str, StageParam] | None) -> None:
        self._stage = stage
        self._params: Mapping[str, StageParam] = dict(params or {})
        self._consumed: set[str] = set()

    @property
    def stage(self) -> str:
        """Name of the stage being parsed (used in error messages)."""
        return self._stage

    def has(self, name: str) -> bool:
        """Whether ``name`` was supplied by the config (does not consume it)."""
        return name in self._params

    def _lookup(self, name: str) -> StageParam | None:
        self._consumed.add(name)
        return self._params.get(name)

    def as_float(self, name: str, default: float) -> float:
        """Read a float parameter (``default`` when absent)."""
        value = self._lookup(name)
        return default if value is None else _as_float(self._stage, name, value)

    def as_optional_float(self, name: str, default: float | None = None) -> float | None:
        """Read a float parameter, preserving ``None`` when absent."""
        value = self._lookup(name)
        return default if value is None else _as_float(self._stage, name, value)

    def as_int(self, name: str, default: int) -> int:
        """Read an int parameter (``default`` when absent)."""
        value = self._lookup(name)
        return default if value is None else _as_int(self._stage, name, value)

    def as_optional_int(self, name: str, default: int | None = None) -> int | None:
        """Read an int parameter, preserving ``None`` when absent."""
        value = self._lookup(name)
        return default if value is None else _as_int(self._stage, name, value)

    def as_bool(self, name: str, default: bool) -> bool:
        """Read a boolean parameter (``default`` when absent)."""
        value = self._lookup(name)
        return default if value is None else _as_bool(self._stage, name, value)

    def as_str(self, name: str, default: str) -> str:
        """Read a string parameter (``default`` when absent)."""
        value = self._lookup(name)
        if value is None:
            return default
        if not isinstance(value, str):
            raise ParamError(f"{self._stage}.{name}: expected a string, got {value!r}")
        return value

    def as_optional_str(self, name: str, default: str | None = None) -> str | None:
        """Read a string parameter, preserving ``None`` when absent."""
        value = self._lookup(name)
        if value is None:
            return default
        if not isinstance(value, str):
            raise ParamError(f"{self._stage}.{name}: expected a string, got {value!r}")
        return value

    def as_sequence(self, name: str, default: tuple[str, ...] = ()) -> tuple[str, ...]:
        """Read a comma-separated sequence parameter."""
        value = self._lookup(name)
        return default if value is None else _as_sequence(self._stage, name, value)

    def check(self) -> None:
        """Raise :class:`ParamError` if any parameter was not consumed."""
        unknown = sorted(set(self._params) - self._consumed)
        if unknown:
            raise ParamError(
                f"{self._stage}: unknown parameter(s) {unknown!r}; "
                f"known keys are the ones the stage implementation reads"
            )
