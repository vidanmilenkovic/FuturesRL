"""Independent v1.5.1 verification: Studio Optuna defaults + IBKR attribution contract.

Verification-only suite (no production edits). It pins the v1.5.1 contracts after
tasks 36/37/38/40:

1. Catalog: exactly the five curated ``optuna_default_selected`` parameters, the
   data-hygiene/structural parameters non-tunable, active-strategy filtering,
   preset/config root symbols all canonical (no delisted GE/ZQ/KE/BTC), and no
   "75 defaults" regression (6, never 75).
2. Optimizer: active-only default search space for TSMOM, panic mean-reversion
   default excludes TSMOM/KAMA/carry and includes mean-reversion params, explicit
   inactive paths are filtered out (no inactive trial evaluation).
3. API catalog: ``GET /api/strategies/schema/catalog`` preserves the boolean
   ``optuna_default_selected``/``optuna_tunable`` fields over JSON.
4. IBKR attribution: persisted live-NLV history (append/dedupe/bounded/age/
   corrupt-safe), ``equity_curve_source``/``equity_curve_live_points``/
   ``portfolio_volatility``/``portfolio_volatility_source`` on
   ``GET /api/positions``; flat IBKR book -> exactly 0.0 vol; <5 distinct UTC
   days -> simulated-backfill; >=5 distinct days -> ibkr-live (v1.5.2 rule:
   distinct days, not point count); ex-ante covariance vol finite; offline
   unchanged.
5. Hostile-environment gate contract: the quality-gate script unsets
   ``FUTURESRL_*`` and the app settings/state-store construct under
   ``FUTURESRL_ENVIRONMENT=paper`` + ``FUTURESRL_RISK_STATE_STORE_URI=duckdb:``.

Everything is offline/deterministic (fakes + ``tmp_path``; never TWS/Databento).
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from configs.instruments import InstrumentsConfig, load_instruments
from configs.settings import Settings
from configs.strategy import load_strategy_config
from fastapi.testclient import TestClient
from src.api.live import (
    EQUITY_HISTORY_FILE_NAME,
    MAX_EQUITY_HISTORY_AGE_DAYS,
    MAX_EQUITY_HISTORY_POINTS,
    LiveEquityHistoryStore,
    LiveStateService,
    ex_ante_portfolio_volatility,
)
from src.api.main import create_app
from src.api.schemas import AccountSnapshotResponse, EquityPoint, PositionRow
from src.api.services import Container, build_container
from src.core.types import AssetClass, MarketState
from src.strategies.parameter_catalog import (
    ROOT_LIST_PARAMETER_KEYS,
    active_parameter_keys,
    catalog_dict,
    default_selected_parameter_keys,
    parameter_catalog,
    parameters_for_strategy,
    strategy_root_symbols,
)

from tests.fixtures.market import FIXED_TIMESTAMP, make_market_state

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
STRATEGY_DIR = REPO_ROOT / "configs" / "strategies"
INSTRUMENTS_PATH = REPO_ROOT / "configs" / "instruments.yaml"

#: The v1.5.3 curated Optuna pre-selection (exactly five keys, never 75).
#: ``timing.carry_filter.min_carry_bps`` stays tunable but is no longer pre-selected.
CURATED_DEFAULT_SELECTED: frozenset[str] = frozenset(
    {
        "timing.tsmom.lookback_fast",
        "timing.tsmom.lookback_slow",
        "timing.kama.window",
        "allocator.volatility_target.target_annual_volatility",
        "risk.quant_risk.trailing_stop_pct",
    }
)

#: Structural/data-hygiene parameters that must never be Optuna-tunable.
HYGIENE_NON_TUNABLE: tuple[str, ...] = (
    "universe.liquidity.min_avg_dollar_volume",
    "universe.liquidity.min_open_interest",
    "universe.liquidity.min_history_days",
    "universe.liquidity.top_n",
    "universe.regime.top_n",
    "risk.llm_risk.sample_count",
    "risk.llm_risk.cache_ttl_seconds",
    "risk.llm_risk.macro_lockout_minutes_before",
    "risk.llm_risk.macro_lockout_minutes_after",
    "risk.llm_risk.macro_lockout_action",
    "risk.llm_risk.macro_lockout_leverage_cap",
)

#: Delisted/non-canonical root tokens that must not appear as strategy roots.
DELISTED_ROOT_TOKENS: frozenset[str] = frozenset({"GE", "ZQ", "KE", "BTC", "BTCUSD"})

#: Known core param names allowed in the curated Optuna defaults.
CORE_PARAM_NAMES: frozenset[str] = frozenset(
    {
        "target_annual_volatility",
        "lookback_fast",
        "lookback_slow",
        "window",
        "min_carry_bps",
        "trailing_stop_pct",
    }
)

FIXED_CLOCK = datetime(2026, 11, 3, 12, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def master() -> InstrumentsConfig:
    return load_instruments(INSTRUMENTS_PATH)


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        environment="backtest",
        data_dir=tmp_path / "data",
        parquet_dir=tmp_path / "data" / "parquet",
        duckdb_path=tmp_path / "data" / "v151.duckdb",
        report_dir=tmp_path / "artifacts",
        instruments_path=INSTRUMENTS_PATH,
        telemetry_interval_seconds=0.05,
    )


@pytest.fixture(scope="module")
def offline_client(tmp_path_factory: pytest.TempPathFactory) -> TestClient:
    """Default offline app (simulated execution, fixture market)."""
    tmp_path = tmp_path_factory.mktemp("v151-offline")
    app = create_app(_settings(tmp_path), clock=lambda: FIXED_CLOCK)
    with TestClient(app) as client:
        yield client


# ---------------------------------------------------------------------------
# 1. Catalog: curated defaults, hygiene exclusion, active filtering, roots
# ---------------------------------------------------------------------------


def test_catalog_has_exactly_five_curated_defaults() -> None:
    catalog = parameter_catalog()
    selected = tuple(spec.key for spec in catalog.parameters if spec.optuna_default_selected)
    assert set(selected) == CURATED_DEFAULT_SELECTED, (
        f"curated Optuna defaults changed; got {sorted(selected)!r}"
    )
    assert len(selected) == 5, f"expected exactly 5 curated defaults, got {len(selected)}"
    for spec in catalog.parameters:
        assert isinstance(spec.optuna_default_selected, bool)
        if spec.optuna_default_selected:
            assert spec.optuna_tunable is True, f"{spec.key}: default-selected must be tunable"
    # No "75 defaults" leak: every non-curated parameter is explicitly false.
    leaked = [
        spec.key
        for spec in catalog.parameters
        if spec.optuna_default_selected and spec.key not in CURATED_DEFAULT_SELECTED
    ]
    assert not leaked, f"unexpected default-selected parameters: {leaked!r}"


def test_catalog_hygiene_parameters_are_not_tunable() -> None:
    catalog = parameter_catalog()
    for key in HYGIENE_NON_TUNABLE:
        spec = catalog.get(key)
        assert spec.optuna_tunable is False, f"{key}: structural/hygiene param must not be tunable"
        assert spec.optuna_default_selected is False
    # Structural params are also excluded from every active strategy's Optuna space.
    tsmom = load_strategy_config(STRATEGY_DIR / "tsmom_vol_target.yaml")
    active = set(active_parameter_keys(tsmom))
    assert active.isdisjoint(HYGIENE_NON_TUNABLE)


def test_catalog_dict_json_preserves_booleans() -> None:
    payload = catalog_dict()
    round_tripped = json.loads(json.dumps(payload))
    parameters = round_tripped["parameters"]
    assert isinstance(parameters, list) and parameters
    assert all(isinstance(entry["optuna_tunable"], bool) for entry in parameters)
    assert all(isinstance(entry["optuna_default_selected"], bool) for entry in parameters)
    defaults = {entry["key"] for entry in parameters if entry["optuna_default_selected"] is True}
    assert defaults == CURATED_DEFAULT_SELECTED
    hygiene = {
        entry["key"]: entry["optuna_tunable"]
        for entry in parameters
        if entry["key"] in HYGIENE_NON_TUNABLE
    }
    assert hygiene and all(value is False for value in hygiene.values())
    assert len(parameters) > 75  # the catalog is large; only the selection is curated


def test_active_filtering_tsmom_vs_panic_mean_reversion() -> None:
    catalog = parameter_catalog()
    tsmom = load_strategy_config(STRATEGY_DIR / "tsmom_vol_target.yaml")
    panic = next(
        preset for preset in catalog.presets if preset.id == "panic_mean_reversion"
    ).to_config()

    tsmom_active = set(active_parameter_keys(tsmom))
    assert "timing.tsmom.lookback_fast" in tsmom_active
    assert "timing.tsmom.lookback_slow" in tsmom_active
    assert "timing.kama.window" in tsmom_active
    assert "timing.carry_filter.min_carry_bps" in tsmom_active
    assert "risk.quant_risk.trailing_stop_pct" in tsmom_active
    # Inactive overlays/selectors are excluded.
    assert not any("mean_reversion" in key for key in tsmom_active)
    assert not any("risk_parity" in key for key in tsmom_active)
    assert not any("mean_variance" in key for key in tsmom_active)
    # Non-tunable hygiene params are excluded from the active/tunable space.
    assert tsmom_active.isdisjoint(HYGIENE_NON_TUNABLE)

    panic_active = set(active_parameter_keys(panic))
    assert "timing.mean_reversion.rsi_period" in panic_active
    assert "timing.mean_reversion.oversold_threshold" in panic_active
    assert "timing.mean_reversion.reversion_strength" in panic_active
    assert not any("tsmom" in key for key in panic_active)
    assert not any("kama" in key for key in panic_active)
    assert not any("carry_filter" in key for key in panic_active)
    assert "allocator.risk_parity.target_annual_volatility" in panic_active

    # parameters_for_strategy mirrors active_parameter_keys exactly and only
    # returns tunable specs.
    assert tuple(spec.key for spec in parameters_for_strategy(tsmom)) == tuple(
        active_parameter_keys(tsmom)
    )
    assert all(spec.optuna_tunable for spec in parameters_for_strategy(panic))


def test_default_selected_keys_follow_active_stages() -> None:
    catalog = parameter_catalog()
    tsmom = load_strategy_config(STRATEGY_DIR / "tsmom_vol_target.yaml")
    panic = next(
        preset for preset in catalog.presets if preset.id == "panic_mean_reversion"
    ).to_config()

    tsmom_defaults = set(default_selected_parameter_keys(tsmom))
    assert tsmom_defaults == CURATED_DEFAULT_SELECTED
    # Panic mean-reversion has no TSMOM/KAMA/carry stages: only the risk default
    # that is actually active may remain.
    panic_defaults = set(default_selected_parameter_keys(panic))
    assert panic_defaults == {"risk.quant_risk.trailing_stop_pct"}
    assert panic_defaults <= set(active_parameter_keys(panic))
    assert panic_defaults <= CURATED_DEFAULT_SELECTED


def test_preset_and_config_root_symbols_are_canonical(master: InstrumentsConfig) -> None:
    known_roots = frozenset(master.roots())
    catalog = parameter_catalog()
    referenced: set[str] = set()
    for preset in catalog.presets:
        strategy = preset.to_config()
        symbols = set(strategy_root_symbols(strategy))
        referenced |= symbols
        unknown = symbols - known_roots
        assert not unknown, f"preset {preset.id!r} references unknown roots {sorted(unknown)!r}"
        assert symbols.isdisjoint(DELISTED_ROOT_TOKENS), (
            f"preset {preset.id!r} still references delisted roots "
            f"{sorted(symbols & DELISTED_ROOT_TOKENS)!r}"
        )

    from configs.strategy import discover_strategy_configs

    for strategy_id, config in discover_strategy_configs(STRATEGY_DIR).items():
        symbols = set(strategy_root_symbols(config))
        unknown = symbols - known_roots
        assert not unknown, f"config {strategy_id!r} references unknown roots {sorted(unknown)!r}"
        assert symbols.isdisjoint(DELISTED_ROOT_TOKENS), (
            f"config {strategy_id!r} still references delisted roots "
            f"{sorted(symbols & DELISTED_ROOT_TOKENS)!r}"
        )

    # No remaining hardcoded delisted root tokens in the catalog/API sources.
    import re

    token_pattern = re.compile(r'["\'](?:GE|ZQ|KE|BTC|BTCUSD)["\']')
    for source in (
        REPO_ROOT / "src" / "strategies" / "parameter_catalog.py",
        REPO_ROOT / "src" / "api" / "studio_service.py",
    ):
        assert token_pattern.search(source.read_text(encoding="utf-8")) is None, (
            f"{source.name} still hardcodes a delisted root token"
        )
    assert ROOT_LIST_PARAMETER_KEYS  # contract helper remains exported


# ---------------------------------------------------------------------------
# 2. Optimizer: active-only default space + defensive filtering
# ---------------------------------------------------------------------------


def _param_paths(space: object) -> list[str]:
    paths: list[str] = []
    for entry in space:  # type: ignore[union-attr]
        path = getattr(entry, "param_path", None)
        assert isinstance(path, str)
        paths.append(path)
    return paths


def _last_segment(path: str) -> str:
    return path.rsplit(".", 1)[-1]


def test_default_search_space_tsmom_is_active_only() -> None:
    from src.backtest.optimizer import default_search_space

    tsmom = load_strategy_config(STRATEGY_DIR / "tsmom_vol_target.yaml")
    space = default_search_space(tsmom)
    paths = _param_paths(space)
    assert paths, "default search space must not be empty"
    assert len(paths) == len(set(paths)), f"duplicate search-space paths: {paths!r}"
    joined = " ".join(paths).lower()
    for forbidden in (
        "min_avg_dollar_volume",
        "min_open_interest",
        "min_history_days",
        "top_n",
        "llm_risk",
        "sample_count",
        "cache_ttl",
        "macro_lockout",
        "translator",
        "max_margin_utilization",
        "risk_parity",
        "mean_reversion",
    ):
        assert forbidden not in joined, (
            f"inactive/hygiene path leaked into default space: {forbidden}"
        )
    allowed_names = {_last_segment(path) for path in paths}
    assert allowed_names <= CORE_PARAM_NAMES, (
        f"curated default space must only contain core params, got {sorted(allowed_names)!r}"
    )
    assert 5 <= len(paths) <= 6, f"expected ~5-6 curated params, got {len(paths)}: {paths!r}"
    # Active TSMOM defaults are present.
    assert "lookback_fast" in allowed_names
    assert "lookback_slow" in allowed_names
    assert "target_annual_volatility" in allowed_names


def test_default_search_space_panic_preset_uses_mean_reversion() -> None:
    from src.backtest.optimizer import default_search_space

    catalog = parameter_catalog()
    panic = next(
        preset for preset in catalog.presets if preset.id == "panic_mean_reversion"
    ).to_config()
    space = default_search_space(panic)
    paths = _param_paths(space)
    joined = " ".join(paths).lower()
    assert paths, "panic mean-reversion default space must not be empty"
    assert "mean_reversion" in joined or any(
        _last_segment(path)
        in {
            "rsi_period",
            "oversold_threshold",
            "overbought_threshold",
            "reversion_strength",
        }
        for path in paths
    ), f"panic preset default space must include mean-reversion params, got {paths!r}"
    for inactive in ("tsmom", "kama", "carry_filter"):
        assert inactive not in joined, f"inactive overlay {inactive!r} leaked into panic default"
    assert "lookback_fast" not in {_last_segment(path) for path in paths}
    assert "min_carry_bps" not in {_last_segment(path) for path in paths}


def test_filter_search_space_drops_inactive_and_hygiene_paths() -> None:
    from src.backtest.optimizer import (
        ParameterRange,
        default_search_space,
        filter_search_space_for_config,
    )

    catalog = parameter_catalog()
    panic = next(
        preset for preset in catalog.presets if preset.id == "panic_mean_reversion"
    ).to_config()
    valid = default_search_space(panic)
    inactive_ranges = (
        ParameterRange(param_path="timing.0.params.lookback_fast", type="int", low=5, high=60),
        ParameterRange(param_path="universe.params.top_n", type="int", low=2, high=20),
        ParameterRange(param_path="risk.0.params.sample_count", type="int", low=1, high=10),
    )
    mixed = tuple(inactive_ranges) + tuple(valid)
    filtered = filter_search_space_for_config(mixed, panic)
    filtered_paths = _param_paths(filtered)
    assert filtered_paths, "filter must keep the active panic params"
    joined = " ".join(filtered_paths).lower()
    assert "lookback_fast" not in joined
    assert "top_n" not in joined
    assert "sample_count" not in joined
    assert set(filtered_paths) == set(_param_paths(valid)), (
        f"filtered space {filtered_paths!r} must equal the curated active space"
    )
    assert len(filtered_paths) == len(set(filtered_paths)), "filter must dedupe paths"
    with pytest.raises(ValueError):
        filter_search_space_for_config(inactive_ranges, panic)


def test_tsmom_default_space_has_no_inactive_overlay_trials() -> None:
    """A default optimization trial must never evaluate an inactive overlay path."""
    from src.backtest.optimizer import default_search_space

    tsmom = load_strategy_config(STRATEGY_DIR / "tsmom_vol_target.yaml")
    active = set(active_parameter_keys(tsmom))
    for entry in default_search_space(tsmom):
        path = str(entry.param_path)
        name = _last_segment(path)
        assert any(key.endswith(name) and key in active for key in active), (
            f"default trial path {path!r} is not an active tunable parameter"
        )


# ---------------------------------------------------------------------------
# 3. API catalog: booleans preserved over JSON (task 40)
# ---------------------------------------------------------------------------


def test_api_catalog_preserves_optuna_fields(offline_client: TestClient) -> None:
    response = offline_client.get("/api/strategies/schema/catalog")
    assert response.status_code == 200, response.text
    raw = json.loads(response.text)  # JSON round-trip: booleans stay booleans
    parameters = raw["parameters"]
    assert isinstance(parameters, list) and len(parameters) > 75
    for entry in parameters:
        assert isinstance(entry["optuna_tunable"], bool)
        assert isinstance(entry["optuna_default_selected"], bool)
    selected = {entry["key"] for entry in parameters if entry["optuna_default_selected"] is True}
    assert selected == CURATED_DEFAULT_SELECTED
    assert len(selected) == 5, f"API must expose exactly 5 curated defaults, got {len(selected)}"
    by_key = {entry["key"]: entry for entry in parameters}
    for key in HYGIENE_NON_TUNABLE:
        assert by_key[key]["optuna_tunable"] is False, f"API {key} must be non-tunable"


# ---------------------------------------------------------------------------
# 4. IBKR attribution: history store, ex-ante volatility, API fields
# ---------------------------------------------------------------------------


def _store(
    tmp_path: Path, *, max_points: int = MAX_EQUITY_HISTORY_POINTS
) -> LiveEquityHistoryStore:
    return LiveEquityHistoryStore(
        tmp_path / "data" / EQUITY_HISTORY_FILE_NAME,
        clock=lambda: FIXED_CLOCK,
        max_points=max_points,
    )


def test_live_history_append_reload_dedupe_and_bounds(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert store.load() == []
    assert store.path.name == EQUITY_HISTORY_FILE_NAME
    first = FIXED_CLOCK - timedelta(minutes=3)
    second = FIXED_CLOCK - timedelta(minutes=2)
    third = FIXED_CLOCK - timedelta(minutes=1)
    store.append(EquityPoint(time=first, value=100.0))
    store.append(EquityPoint(time=second, value=110.0))
    store.append(EquityPoint(time=first, value=101.0))  # dedupe: later value wins
    points = store.load()
    assert [point.time for point in points] == [first, second]
    assert points[0].value == pytest.approx(101.0)
    # Reload through a fresh instance (persistence).
    reloaded = _store(tmp_path).load()
    assert [(point.time, point.value) for point in reloaded] == [
        (point.time, point.value) for point in points
    ]
    # Bounded: newest max_points retained.
    bounded = _store(tmp_path, max_points=2)
    bounded.append(EquityPoint(time=third, value=120.0))
    kept = bounded.load()
    assert len(kept) == 2
    assert [point.time for point in kept] == [second, third]


def test_live_history_age_bound_and_corrupt_safety(tmp_path: Path) -> None:
    store = _store(tmp_path)
    stale = FIXED_CLOCK - timedelta(days=MAX_EQUITY_HISTORY_AGE_DAYS + 5)
    fresh = FIXED_CLOCK - timedelta(minutes=1)
    store.append(EquityPoint(time=stale, value=50.0))  # trimmed out immediately
    store.append(EquityPoint(time=fresh, value=100.0))
    points = store.load()
    assert [point.time for point in points] == [fresh]

    # Corrupt payload: treated as empty history and quarantined, never raises.
    store.path.write_text("{not valid json", encoding="utf-8")
    assert store.load() == []
    assert not store.path.exists()
    quarantined = list(store.path.parent.glob(f"{store.path.name}.corrupt-*"))
    assert quarantined, "corrupt history must be quarantined next to the original"

    # Non-object payload and invalid entries are tolerated too.
    store.path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
    assert store.load() == []
    store.path.write_text(
        json.dumps(
            {
                "version": 1,
                "points": [
                    {"time": fresh.isoformat(), "value": 123.0},
                    {"time": "not-a-timestamp", "value": 1.0},
                    {"time": (fresh + timedelta(seconds=1)).isoformat(), "value": -5.0},
                ],
            }
        ),
        encoding="utf-8",
    )
    assert [point.value for point in store.load()] == [123.0]
    store.clear()
    assert store.load() == []


def test_ex_ante_portfolio_volatility_contract() -> None:
    assert ex_ante_portfolio_volatility({}, make_market_state(roots=("ES", "NQ"))) == (
        0.0,
        "ex-ante-book",
    )
    assert ex_ante_portfolio_volatility({"ES": 0.2}, None) == (None, "unavailable")
    no_cov = make_market_state(roots=("ES", "NQ"), with_covariance=False)
    assert ex_ante_portfolio_volatility({"ES": 0.2}, no_cov) == (None, "unavailable")

    state = make_market_state(roots=("ES", "NQ"))
    weights = {"ES": 0.3, "NQ": -0.2}
    volatility, source = ex_ante_portfolio_volatility(weights, state)
    assert source == "ex-ante-book"
    assert volatility is not None and math.isfinite(volatility) and volatility > 0.0
    covariance = state.covariance_submatrix(("ES", "NQ"))
    assert covariance is not None
    vector = np.array([0.3, -0.2])
    assert volatility == pytest.approx(math.sqrt(float(vector @ covariance @ vector)))


class _FakeMarket:
    """Minimal MarketDataService stand-in exposing ``latest_state``."""

    def __init__(self, state: MarketState | None) -> None:
        self._state = state

    def latest_state(self) -> MarketState | None:
        return self._state


class _FakeIBKRExecution:
    """Fake IBKR adapter: never talks to TWS, serves a deterministic book."""

    def __init__(
        self,
        *,
        snapshot: AccountSnapshotResponse,
        points: tuple[EquityPoint, ...] = (),
    ) -> None:
        self._snapshot = snapshot
        self._points = points
        self.ibkr_enabled = True
        self.source_name = "engine"
        self.connected = True

    async def account(self) -> AccountSnapshotResponse:
        return self._snapshot

    @property
    def live_equity_points(self) -> tuple[EquityPoint, ...]:
        return self._points


def _position_row(root: str = "ES", *, weight: float = 0.24, quantity: int = 1) -> PositionRow:
    notional = weight * 1_000_000.0
    return PositionRow(
        instrument_id=root,
        root_symbol=root,
        asset_class=AssetClass.EQUITY_INDEX,
        contract_symbol=f"{root}_20240315",
        expiration=None,
        days_to_roll=None,
        quantity=quantity,
        average_price=4800.0,
        market_price=4800.0,
        multiplier=50.0,
        notional=notional,
        weight=weight,
        unrealized_pnl=0.0,
        initial_margin=15_000.0,
        maintenance_margin=13_600.0,
    )


def _account_snapshot(
    positions: list[PositionRow], *, curve: list[EquityPoint] | None = None
) -> AccountSnapshotResponse:
    return AccountSnapshotResponse(
        timestamp=FIXED_TIMESTAMP,
        net_liquidation=1_000_000.0,
        cash_balance=1_000_000.0,
        buying_power=2_000_000.0,
        initial_margin_used=sum(row.initial_margin for row in positions),
        maintenance_margin_used=sum(row.maintenance_margin for row in positions),
        excess_liquidity=1_000_000.0,
        margin_utilization=0.0,
        day_pnl=0.0,
        day_pnl_pct=0.0,
        gross_leverage=sum(abs(row.weight) for row in positions),
        net_exposure=sum(row.weight for row in positions),
        equity_curve=curve or [EquityPoint(time=FIXED_TIMESTAMP, value=1_000_000.0)],
        positions=list(positions),
        source="engine",
    )


def _attribution_client(
    tmp_path: Path,
    *,
    execution: _FakeIBKRExecution,
    market_state: MarketState | None = None,
) -> TestClient:
    settings = _settings(tmp_path)
    clock = lambda: FIXED_CLOCK  # noqa: E731 - deterministic fixture clock
    container: Container = build_container(settings, clock=clock)
    live = LiveStateService(
        settings=settings,
        clock=clock,
        execution=execution,
        instrument_master=settings.instruments,
        market=_FakeMarket(market_state) if market_state is not None else _FakeMarket(None),
    )
    container = replace(container, execution=execution, live=live)
    app = create_app(settings, clock=clock, container=container)
    return TestClient(app)


def _live_points(count: int) -> tuple[EquityPoint, ...]:
    """``count`` real NLV points all on the SAME UTC calendar day."""
    return tuple(
        EquityPoint(time=FIXED_CLOCK - timedelta(minutes=count - index), value=1_000_000.0 + index)
        for index in range(count)
    )


def _live_points_across_days(days: int, per_day: int = 2) -> tuple[EquityPoint, ...]:
    """Real NLV points spanning ``days`` distinct UTC calendar days (v1.5.2 rule)."""
    points: list[EquityPoint] = []
    for day in range(days):
        day_start = FIXED_CLOCK - timedelta(days=day)
        for index in range(per_day):
            points.append(
                EquityPoint(
                    time=day_start - timedelta(minutes=index),
                    value=1_000_000.0 + day * 1_000.0 + index,
                )
            )
    return tuple(sorted(points, key=lambda point: point.time))


def test_positions_attribution_flat_ibkr_book_has_zero_volatility(tmp_path: Path) -> None:
    # v1.5.2: five SAME-DAY points are not enough for the multi-day ``ibkr-live``
    # badge (that needs >=5 distinct UTC calendar days), but the flat-book risk
    # values and the 1D intraday provenance must still be exact.
    execution = _FakeIBKRExecution(snapshot=_account_snapshot([], curve=[]), points=_live_points(5))
    with _attribution_client(tmp_path, execution=execution) as client:
        response = client.get("/api/positions")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["portfolio_volatility"] == 0.0, "a flat IBKR book must report exactly 0.0 vol"
    assert body["portfolio_volatility_source"] == "ex-ante-book"
    assert body["equity_curve_source"] == "simulated-backfill"
    assert body["equity_curve_live_points"] == 5
    assert body["equity_curve_live_days"] == 1
    assert len(body["equity_curve_intraday"]) == 5
    # The <5-day path keeps the snapshot's synthetic backfill curve (here the
    # account helper's single default point), NOT the real intraday points.
    assert [point["value"] for point in body["equity_curve"]] == [pytest.approx(1_000_000.0)]
    assert len(body["equity_curve"]) != 5


@pytest.mark.parametrize("live_count", [0, 1, 4])
def test_positions_attribution_backfill_when_few_live_points(
    live_count: int, tmp_path: Path
) -> None:
    backfill = [
        EquityPoint(time=FIXED_TIMESTAMP - timedelta(minutes=index), value=999_000.0 + index)
        for index in range(3)
    ]
    execution = _FakeIBKRExecution(
        snapshot=_account_snapshot([], curve=backfill), points=_live_points(live_count)
    )
    with _attribution_client(tmp_path, execution=execution) as client:
        body = client.get("/api/positions").json()
    assert body["equity_curve_source"] == "simulated-backfill"
    assert body["equity_curve_live_points"] == live_count
    assert len(body["equity_curve"]) == len(backfill), "synthetic backfill curve is preserved"
    assert body["equity_curve_source"] != "ibkr-live"


def test_positions_attribution_live_curve_and_ex_ante_volatility(tmp_path: Path) -> None:
    state = make_market_state(roots=("ES", "NQ"))
    positions = [_position_row("ES", weight=0.3), _position_row("NQ", weight=0.2, quantity=2)]
    real_points = _live_points_across_days(days=5, per_day=2)
    execution = _FakeIBKRExecution(snapshot=_account_snapshot(positions), points=real_points)
    with _attribution_client(tmp_path, execution=execution, market_state=state) as client:
        body = client.get("/api/positions").json()
    # v1.5.2: live multi-day provenance now requires >=5 DISTINCT UTC days.
    assert body["equity_curve_source"] == "ibkr-live"
    assert body["equity_curve_live_points"] == len(real_points)
    assert body["equity_curve_live_days"] == 5
    assert [point["value"] for point in body["equity_curve"]] == [
        pytest.approx(point.value) for point in real_points
    ], "live mode must plot ONLY real recorded points (no synthetic spline prepended)"
    intraday = [point["value"] for point in body["equity_curve_intraday"]]
    assert intraday == [pytest.approx(point.value) for point in real_points[-2:]]
    volatility = body["portfolio_volatility"]
    assert isinstance(volatility, float) and math.isfinite(volatility) and volatility > 0.0
    assert body["portfolio_volatility_source"] == "ex-ante-book"
    weights = {"ES": 0.3, "NQ": 0.2}
    expected, _ = ex_ante_portfolio_volatility(weights, state)
    assert expected is not None
    assert volatility == pytest.approx(expected, rel=1e-9)


def test_positions_attribution_covariance_unavailable_is_honest(tmp_path: Path) -> None:
    positions = [_position_row("ES", weight=0.3)]
    execution = _FakeIBKRExecution(snapshot=_account_snapshot(positions), points=_live_points(2))
    with _attribution_client(tmp_path, execution=execution) as client:
        body = client.get("/api/positions").json()
    assert body["portfolio_volatility"] is None, "no covariance -> never a synthetic vol"
    assert body["portfolio_volatility_source"] == "unavailable"
    assert body["equity_curve_source"] == "simulated-backfill"


def test_offline_positions_attribution_unchanged(offline_client: TestClient) -> None:
    response = offline_client.get("/api/positions")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["equity_curve_source"] == "fixture"
    assert body["equity_curve_live_points"] == 0
    assert body["portfolio_volatility"] is None
    assert body["portfolio_volatility_source"] is None


# ---------------------------------------------------------------------------
# 5. Hostile-environment quality gate contract
# ---------------------------------------------------------------------------


def test_quality_gate_script_unsets_hostile_environment() -> None:
    script = (REPO_ROOT / "scripts" / "quality_gate.sh").read_text(encoding="utf-8")
    assert "FUTURESRL_" in script
    assert "compgen -e" in script
    assert 'unset "$variable"' in script
    # The unset loop must run before any ruff/mypy/pytest invocation.
    unset_index = script.index('unset "$variable"')
    for command in ("ruff check", "ruff format --check", "mypy src configs", "pytest"):
        command_index = script.find(command)
        assert command_index != -1
        assert unset_index < command_index, f"hostile env not cleared before {command!r}"


def test_hostile_environment_settings_and_state_store(tmp_path: Path) -> None:
    """The app settings + durable risk store construct under the paper/duckdb env."""
    code = """
import os
from configs.settings import Settings
from src.risk.state_store import PersistedRiskState, build_risk_state_store

settings = Settings()
assert settings.environment == "paper", settings.environment
assert settings.risk_state_store_uri == "duckdb:"
store = build_risk_state_store(
    settings.risk_state_store_uri,
    duckdb_path=os.environ["V151_HOSTILE_DUCKDB"],
)
store.save(PersistedRiskState(portfolio_high_water_mark=123.0))
loaded = store.load()
assert loaded is not None and loaded.portfolio_high_water_mark == 123.0
print("ok")
"""
    environment = dict(os.environ)
    environment.update(
        {
            "FUTURESRL_ENVIRONMENT": "paper",
            "FUTURESRL_RISK_STATE_STORE_URI": "duckdb:",
            "V151_HOSTILE_DUCKDB": str(tmp_path / "hostile.duckdb"),
        }
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"


def test_no_hardcoded_delisted_roots_in_catalog_payload() -> None:
    payload = catalog_dict()
    serialised = json.dumps(payload).upper()
    for token in DELISTED_ROOT_TOKENS:
        assert f'"{token}"' not in serialised, (
            f"catalog payload still references delisted root {token}"
        )
