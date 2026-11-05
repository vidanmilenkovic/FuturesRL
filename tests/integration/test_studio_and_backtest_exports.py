"""Independent QoL release verification: conId qualification, artifact management,
strategy clone/rename/export, backtest exports/diff snapshots and operator resets.

Verification-only suite for branch ``feat/qol-artifact-management`` (tasks 52-55);
no production edits unless a real bug is found (reported first). Everything is
hermetic: fake IBKR clients (never TWS), ``tmp_path`` settings and temp API apps.

1. **IBKR conId qualification** (task 52): every root is qualified before any
   ``reqRealTimeBars``/``reqTickByTickData``/``reqMktData``/``reqMktDepth`` call,
   only the qualified (conId-bearing) contract is used for subscriptions, and a
   resolution failure raises an actionable error / skips cleanly without ever
   subscribing an unqualified contract.
2. **Artifact management** (task 53): DELETE optimization studies, backtest runs
   and custom strategies remove memory + disk, vanish from lists, 404 for unknown
   ids, and committed built-in strategies are protected.
3. **Strategy clone/rename/export** (task 53): clone/rename create/remove ids and
   refresh the catalog; GET ``/yaml`` returns a valid downloadable StrategyConfig.
4. **Backtest exports/diff snapshots** (task 53): export.json has summary +
   tear sheet + strategy_config; trades.csv headers/rows match trades; config
   snapshots are persisted for new runs and old runs tolerate missing snapshots.
5. **Risk/maintenance resets** (task 54): risk state kill-switch fields;
   ``reset-lock`` requires confirmation/operator token; paper-state reset clears
   bridge/risk state in tmp paths, refuses live, is idempotent; unknown/bad
   requests are clear 400/403/409s.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
from configs.instruments import load_instruments
from configs.settings import Settings
from configs.strategy import load_strategy_config
from fastapi.testclient import TestClient
from src.api.main import create_app
from src.data.ibkr_data import IBKRMarketDataStreamer

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"
INSTRUMENTS_PATH = CONFIGS_DIR / "instruments.yaml"
FIXED_CLOCK = datetime(2026, 11, 3, 12, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Shared hermetic settings / API helpers
# ---------------------------------------------------------------------------


def _settings(
    tmp_path: Path,
    *,
    environment: str = "backtest",
    configs_dir: Path | None = None,
    ibkr_paper: bool = True,
    ibkr_port: int | None = None,
) -> Settings:
    """Hermetic settings with TWS wiring explicitly disabled."""
    kwargs: dict[str, Any] = {
        "environment": environment,
        "data_dir": tmp_path / "data",
        "parquet_dir": tmp_path / "data" / "parquet",
        "duckdb_path": tmp_path / "data" / "qol.duckdb",
        "report_dir": tmp_path / "artifacts",
        "configs_dir": configs_dir or CONFIGS_DIR,
        "instruments_path": INSTRUMENTS_PATH,
        "telemetry_interval_seconds": 0.02,
        "api_enable_ibkr": False,
        "ibkr_paper": ibkr_paper,
    }
    if ibkr_port is not None:
        kwargs["ibkr_port"] = ibkr_port
    return Settings(**kwargs)


def _paper_settings(tmp_path: Path) -> Settings:
    """Paper-like hermetic settings (tmp built-in config tree, no committed writes)."""
    configs_dir = tmp_path / "configs"
    strategies = configs_dir / "strategies"
    strategies.mkdir(parents=True, exist_ok=True)
    for name in ("tsmom_vol_target.yaml", "drl_ppo_llm_risk.yaml"):
        source = CONFIGS_DIR / "strategies" / name
        if source.is_file():
            (strategies / name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return _settings(tmp_path, environment="paper", configs_dir=configs_dir)


# ---------------------------------------------------------------------------
# 1. IBKR conId qualification (task 52) — fake clients, never TWS
# ---------------------------------------------------------------------------


class _FakeFuture:
    """Minimal ib_async-contract lookalike (``conId`` present only when qualified)."""

    def __init__(self, symbol: str, exchange: str, *, con_id: int | None = None) -> None:
        self.symbol = symbol
        self.exchange = exchange
        self.conId = con_id

    def qualified(self, con_id: int) -> _FakeFuture:
        return _FakeFuture(self.symbol, self.exchange, con_id=con_id)


class _FakeSubscription:
    def __init__(self, req_id: int) -> None:
        self.reqId = req_id


class _RecordingQualifyingIB:
    """Fake IB client recording qualification/subscription ordering."""

    def __init__(
        self,
        *,
        failing_roots: frozenset[str] = frozenset(),
        qualify_sync: bool = False,
        no_qualify_api: bool = False,
    ) -> None:
        self.failing_roots = failing_roots
        self.qualify_sync = qualify_sync
        self.no_qualify_api = no_qualify_api
        self.events: list[tuple[str, str, _FakeFuture | None, int | None]] = []
        self.qualified_contracts: dict[str, _FakeFuture] = {}
        self._next_req_id = 100

    # -- connection surface ----------------------------------------------------
    def isConnected(self) -> bool:  # noqa: N802
        return True

    async def connectAsync(  # noqa: N802
        self, host: str, port: int, client_id: int, timeout: float
    ) -> None:
        self.events.append(("connect", host, None, None))

    def disconnect(self) -> None:
        self.events.append(("disconnect", "", None, None))

    # -- qualification --------------------------------------------------------
    def _qualify(self, contract: _FakeFuture) -> list[_FakeFuture]:
        if contract.symbol in self.failing_roots:
            return []
        qualified = contract.qualified(9000 + len(self.qualified_contracts))
        self.qualified_contracts[contract.symbol] = qualified
        return [qualified]

    async def qualifyContractsAsync(  # noqa: N802
        self, contract: _FakeFuture
    ) -> list[_FakeFuture]:
        self.events.append(("qualify", contract.symbol, contract, contract.conId))
        return self._qualify(contract)

    def qualifyContracts(self, contract: _FakeFuture) -> list[_FakeFuture]:  # noqa: N802
        self.events.append(("qualify", contract.symbol, contract, contract.conId))
        return self._qualify(contract)

    def __getattribute__(self, name: str) -> Any:
        # Hide qualification APIs when emulating a legacy duck-typed client.
        if name in {"qualifyContractsAsync", "qualifyContracts"} and object.__getattribute__(
            self, "no_qualify_api"
        ):
            raise AttributeError(name)
        return object.__getattribute__(self, name)

    # -- subscriptions --------------------------------------------------------
    def _subscribe_event(self, kind: str, contract: _FakeFuture) -> _FakeSubscription:
        self.events.append((kind, contract.symbol, contract, contract.conId))
        self._next_req_id += 1
        return _FakeSubscription(self._next_req_id)

    def reqRealTimeBars(  # noqa: N802
        self, contract: _FakeFuture, bar_size: int, what: str, rth: bool
    ) -> _FakeSubscription:
        return self._subscribe_event("bars", contract)

    def reqTickByTickData(  # noqa: N802
        self, contract: _FakeFuture, tick_type: str, n: int, ignore: bool
    ) -> _FakeSubscription:
        return self._subscribe_event("ticks", contract)

    def reqMktData(  # noqa: N802
        self, contract: _FakeFuture, generic: str, snapshot: bool, regulatory: bool
    ) -> _FakeSubscription:
        return self._subscribe_event("quotes", contract)

    def reqMktDepth(  # noqa: N802
        self, contract: _FakeFuture, rows: int, smart: bool
    ) -> _FakeSubscription:
        return self._subscribe_event("depth", contract)

    def cancelRealTimeBars(self, bars: object) -> None:  # noqa: N802
        pass

    def cancelTickByTickData(self, contract: object, tick_type: str) -> None:  # noqa: N802
        pass

    def cancelMktData(self, contract: object) -> None:  # noqa: N802
        pass

    def cancelMktDepth(self, contract: object, smart: bool) -> None:  # noqa: N802
        pass


def _streamer(
    tmp_path: Path,
    client: _RecordingQualifyingIB,
    *,
    roots: tuple[str, ...] = ("ES", "NQ"),
) -> IBKRMarketDataStreamer:
    async def _no_sleep(_delay: float) -> None:
        return None

    def contract_factory(root: str) -> _FakeFuture:
        return _FakeFuture(root, "CME")

    return IBKRMarketDataStreamer(
        settings=_settings(tmp_path),
        roots=roots,
        instrument_master=load_instruments(INSTRUMENTS_PATH),
        ib_factory=lambda: client,
        contract_factory=contract_factory,
        sleep=_no_sleep,
    )


def test_ibkr_qualifies_every_contract_before_subscribing(tmp_path: Path) -> None:
    client = _RecordingQualifyingIB()
    streamer = _streamer(tmp_path, client, roots=("ES", "NQ"))
    asyncio.run(streamer.connect())

    events = client.events
    assert events[0][0] == "connect"
    kinds = [event[0] for event in events]
    assert kinds.count("qualify") == 2
    assert kinds.count("bars") == 2 and kinds.count("ticks") == 2
    assert kinds.count("quotes") == 2 and kinds.count("depth") == 2
    # Per root: qualification strictly before every subscription, and only the
    # qualified conId-bearing contract is ever passed to IBKR subscription calls.
    for root in ("ES", "NQ"):
        root_events = [event for event in events if event[1] == root]
        qualify_index = next(i for i, event in enumerate(root_events) if event[0] == "qualify")
        subscription_events = [
            event for event in root_events if event[0] in {"bars", "ticks", "quotes", "depth"}
        ]
        assert len(subscription_events) == 4
        assert all(root_events.index(event) > qualify_index for event in subscription_events), (
            f"subscription before qualification for {root}: {root_events}"
        )
        qualified = client.qualified_contracts[root]
        assert qualified.conId is not None
        assert all(event[3] == qualified.conId for event in subscription_events)


def test_ibkr_qualification_failure_is_actionable_and_skips(tmp_path: Path) -> None:
    client = _RecordingQualifyingIB(failing_roots=frozenset({"ES", "NQ"}))
    streamer = _streamer(tmp_path, client, roots=("ES", "NQ"))
    with pytest.raises(RuntimeError) as excinfo:
        asyncio.run(streamer.connect())
    message = str(excinfo.value)
    assert "could not qualify any market-data contract" in message
    assert "could not resolve a conId for root 'ES' on CME" in message
    assert "could not resolve a conId for root 'NQ' on CME" in message
    subscriptions = [
        event for event in client.events if event[0] in {"bars", "ticks", "quotes", "depth"}
    ]
    assert subscriptions == [], "an unqualified/failed root must never be subscribed"


def test_ibkr_partial_qualification_failure_skips_only_that_root(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    client = _RecordingQualifyingIB(failing_roots=frozenset({"NQ"}))
    streamer = _streamer(tmp_path, client, roots=("ES", "NQ"))
    with caplog.at_level("WARNING", logger="src.data.ibkr_data"):
        asyncio.run(streamer.connect())
    subscribed_roots = {
        event[1] for event in client.events if event[0] in {"bars", "ticks", "quotes", "depth"}
    }
    assert subscribed_roots == {"ES"}
    assert client.qualified_contracts["ES"].conId is not None
    assert any(
        "could not resolve a conId for root 'NQ'" in record.message for record in caplog.records
    )


def test_ibkr_legacy_client_without_qualification_api_uses_raw_contract(
    tmp_path: Path,
) -> None:
    client = _RecordingQualifyingIB(no_qualify_api=True)
    streamer = _streamer(tmp_path, client, roots=("ES",))
    asyncio.run(streamer.connect())
    kinds = [event[0] for event in client.events]
    assert "qualify" not in kinds, "legacy clients without the API must not be queried"
    assert kinds.count("bars") == 1


# ---------------------------------------------------------------------------
# Shared API app helpers (sections 2-5)
# ---------------------------------------------------------------------------


def _artifact_dirs(settings: Settings) -> tuple[Path, Path]:
    settings.optimization_artifacts_dir.mkdir(parents=True, exist_ok=True)
    settings.backtest_artifacts_dir.mkdir(parents=True, exist_ok=True)
    return settings.optimization_artifacts_dir, settings.backtest_artifacts_dir


def _write_clean_study(settings: Settings, *, study_id: str = "qol-clean-study") -> Path:
    """Persist one v1.5.3-clean optimizer study in the API artifact store."""
    from src.api.optimize_service import OptimizationStudy
    from src.backtest.optimizer import default_search_space

    config = load_strategy_config(CONFIGS_DIR / "strategies" / "tsmom_vol_target.yaml")
    space = [
        {"param_path": entry.param_path, "type": entry.type, "low": entry.low, "high": entry.high}
        for entry in default_search_space(config)
    ]
    study = OptimizationStudy.model_validate(
        {
            "study_id": study_id,
            "base_strategy_id": config.id,
            "objective_metric": "sharpe",
            "n_trials": 5,
            "completed_trials": 5,
            "created_at": FIXED_CLOCK.isoformat(),
            "best_params": {
                "timing.0.params.lookback_fast": 21,
                "timing.0.params.lookback_slow": 126,
            },
            "best_value": 0.5,
        }
    )
    payload = study.model_dump(mode="json")
    payload.pop("provenance", None)
    payload["seed"] = 7
    payload["request"] = {"base_strategy_id": config.id, "search_space": space}
    path = settings.optimization_artifacts_dir / f"{study_id}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _run_backtest(client: TestClient, *, strategy_id: str = "tsmom_vol_target") -> dict[str, Any]:
    response = client.post(
        "/api/backtests",
        json={
            "strategy_id": strategy_id,
            "start": "2024-01-02",
            "end": "2024-06-28",
            "initial_equity": 1_000_000.0,
            "walk_forward": True,
            "wf_splits": 3,
            "purge_bars": 2,
            "embargo_bars": 2,
        },
    )
    assert response.status_code == 200, response.text
    return cast(dict[str, Any], response.json())


# ---------------------------------------------------------------------------
# 2. Artifact management: DELETE optimize/backtest/strategy (task 53)
# ---------------------------------------------------------------------------


def test_delete_optimization_study_removes_memory_and_disk(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _artifact_dirs(settings)
    artifact = _write_clean_study(settings)
    app = create_app(settings, clock=lambda: FIXED_CLOCK)
    with TestClient(app) as client:
        listed = client.get("/api/optimize")
        assert [row["study_id"] for row in listed.json()] == ["qol-clean-study"]
        assert client.get("/api/optimize/qol-clean-study").status_code == 200

        deleted = client.delete("/api/optimize/qol-clean-study")
        assert deleted.status_code == 200, deleted.text
        receipt = deleted.json()
        assert receipt["id"] == "qol-clean-study"
        assert receipt["deleted"] is True
        assert receipt["artifact_deleted"] is True
        assert not artifact.exists()
        assert client.get("/api/optimize").json() == []
        assert client.get("/api/optimize/qol-clean-study").status_code == 404
        # Second delete is a clear 404, never a silent success.
        assert client.delete("/api/optimize/qol-clean-study").status_code == 404
        assert client.delete("/api/optimize/unknown-study").status_code == 404


def test_delete_backtest_run_removes_memory_and_artifact(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _artifact_dirs(settings)
    app = create_app(settings, clock=lambda: FIXED_CLOCK)
    with TestClient(app) as client:
        run = _run_backtest(client)
        run_id = run["id"]
        artifact = settings.backtest_artifacts_dir / f"{run_id}.json"
        assert artifact.is_file(), "custom run must be persisted before deletion"
        assert run_id in {row["id"] for row in client.get("/api/backtests").json()["runs"]}

        deleted = client.delete(f"/api/backtests/{run_id}")
        assert deleted.status_code == 200, deleted.text
        receipt = deleted.json()
        assert receipt["id"] == run_id and receipt["deleted"] is True
        assert receipt["artifact_deleted"] is True
        assert not artifact.exists()
        assert run_id not in {row["id"] for row in client.get("/api/backtests").json()["runs"]}
        assert client.get(f"/api/backtests/{run_id}").status_code == 404
        assert client.delete("/api/backtests/bt-9999").status_code == 404


def test_strategy_delete_protects_builtins_and_removes_custom(tmp_path: Path) -> None:
    from src.api.main import create_app as _create_app  # noqa: F401  (explicit helper scope)

    settings = _paper_settings(tmp_path)
    app = create_app(settings, clock=lambda: FIXED_CLOCK)
    with TestClient(app) as client:
        # Committed built-ins are protected.
        builtin = client.delete("/api/strategies/tsmom_vol_target")
        assert builtin.status_code == 409, builtin.text
        assert "built-in" in builtin.json()["detail"]

        # A custom clone is deletable and disappears from the catalog.
        cloned = client.post(
            "/api/strategies/tsmom_vol_target/clone",
            json={"new_strategy_id": "qol_custom_clone"},
        )
        assert cloned.status_code == 200, cloned.text
        custom_path = settings.configs_dir / "strategies" / "qol_custom_clone.yaml"
        assert custom_path.is_file()
        assert "qol_custom_clone" in {row["id"] for row in client.get("/api/strategies").json()}

        deleted = client.delete("/api/strategies/qol_custom_clone")
        assert deleted.status_code == 200, deleted.text
        receipt = deleted.json()
        assert receipt["artifact_deleted"] is True
        assert receipt["path"] == str(custom_path)
        assert not custom_path.exists()
        assert "qol_custom_clone" not in {row["id"] for row in client.get("/api/strategies").json()}
        assert client.delete("/api/strategies/qol_custom_clone").status_code == 404
        assert client.delete("/api/strategies/unknown_strategy").status_code == 404


# ---------------------------------------------------------------------------
# 3. Strategy clone / rename / YAML export (task 53)
# ---------------------------------------------------------------------------


def test_strategy_clone_rename_and_yaml_export(tmp_path: Path) -> None:
    import yaml
    from configs.strategy import parse_strategy_config

    settings = _paper_settings(tmp_path)
    app = create_app(settings, clock=lambda: FIXED_CLOCK)
    with TestClient(app) as client:
        cloned = client.post(
            "/api/strategies/tsmom_vol_target/clone",
            json={"new_strategy_id": "qol_clone_a"},
        )
        assert cloned.status_code == 200, cloned.text
        assert cloned.json()["id"] == "qol_clone_a"
        ids = {row["id"] for row in client.get("/api/strategies").json()}
        assert {"tsmom_vol_target", "qol_clone_a"} <= ids
        assert (settings.configs_dir / "strategies" / "qol_clone_a.yaml").is_file()

        # Existing target id is a clear 409; unknown source is 404.
        assert (
            client.post(
                "/api/strategies/tsmom_vol_target/clone",
                json={"new_strategy_id": "qol_clone_a"},
            ).status_code
            == 409
        )
        assert (
            client.post(
                "/api/strategies/unknown_strategy/clone",
                json={"new_strategy_id": "qol_clone_b"},
            ).status_code
            == 404
        )

        # Committed built-ins cannot be renamed (clone instead).
        assert (
            client.post(
                "/api/strategies/tsmom_vol_target/rename",
                json={"new_strategy_id": "qol_builtin_renamed"},
            ).status_code
            == 409
        )
        renamed = client.post(
            "/api/strategies/qol_clone_a/rename",
            json={"new_strategy_id": "qol_clone_b"},
        )
        assert renamed.status_code == 200, renamed.text
        assert renamed.json()["id"] == "qol_clone_b"
        ids = {row["id"] for row in client.get("/api/strategies").json()}
        assert "qol_clone_b" in ids and "qol_clone_a" not in ids
        assert not (settings.configs_dir / "strategies" / "qol_clone_a.yaml").exists()
        assert (settings.configs_dir / "strategies" / "qol_clone_b.yaml").is_file()

        # YAML download is an attachment and a valid StrategyConfig document.
        yaml_response = client.get("/api/strategies/qol_clone_b/yaml")
        assert yaml_response.status_code == 200
        assert yaml_response.headers["content-type"].startswith("text/yaml")
        assert 'filename="qol_clone_b.yaml"' in yaml_response.headers["content-disposition"]
        parsed = parse_strategy_config(yaml.safe_load(yaml_response.text))
        assert parsed.id == "qol_clone_b"
        assert client.get("/api/strategies/unknown_strategy/yaml").status_code == 404


# ---------------------------------------------------------------------------
# 4. Backtest exports + strategy-config snapshots (task 53)
# ---------------------------------------------------------------------------


def test_backtest_export_json_trades_csv_and_snapshots(tmp_path: Path) -> None:
    import csv
    import io

    settings = _settings(tmp_path)
    _artifact_dirs(settings)
    app = create_app(settings, clock=lambda: FIXED_CLOCK)
    with TestClient(app) as client:
        run = _run_backtest(client)
        run_id = run["id"]
        assert run["strategy_config"], "new runs must persist a strategy-config snapshot"
        snapshot = run["strategy_config"]
        assert snapshot["id"] == "tsmom_vol_target"

        detail = client.get(f"/api/backtests/{run_id}")
        assert detail.status_code == 200
        assert detail.json()["strategy_config"] == snapshot
        tearsheet = client.get(f"/api/backtests/{run_id}/tearsheet")
        assert tearsheet.status_code == 200
        trades = tearsheet.json()["trades"]
        assert isinstance(trades, list)

        export = client.get(f"/api/backtests/{run_id}/export.json")
        assert export.status_code == 200, export.text
        assert export.headers["content-type"].startswith("application/json")
        assert 'filename="' in export.headers["content-disposition"]
        payload = export.json()
        assert payload["summary"]["id"] == run_id
        assert payload["summary"]["strategy_config"] == snapshot
        assert payload["strategy_config"] == snapshot
        assert payload["tearsheet"]["run_id"] == run_id
        assert payload["tearsheet"]["trades"] == trades

        csv_response = client.get(f"/api/backtests/{run_id}/trades.csv")
        assert csv_response.status_code == 200, csv_response.text
        assert csv_response.headers["content-type"].startswith("text/csv")
        assert 'filename="' in csv_response.headers["content-disposition"]
        rows = list(csv.reader(io.StringIO(csv_response.text)))
        expected_header = [
            "timestamp",
            "order_id",
            "instrument_id",
            "side",
            "quantity",
            "price",
            "notional",
            "commissions",
            "slippage",
            "kind",
        ]
        assert rows[0] == expected_header
        assert len(rows) - 1 == len(trades), (
            f"CSV has {len(rows) - 1} rows but tearsheet has {len(trades)} trades"
        )
        if trades:
            first = trades[0]
            csv_first = rows[1]
            assert csv_first[1] == first["order_id"]
            assert csv_first[2] == first["instrument_id"]
            assert int(csv_first[4]) == first["quantity"]

        # Snapshots survive an API restart (persisted artifact reload).
        restarted = create_app(settings, clock=lambda: FIXED_CLOCK)
        with TestClient(restarted) as second:
            reloaded = second.get(f"/api/backtests/{run_id}")
            assert reloaded.status_code == 200
            assert reloaded.json()["strategy_config"] == snapshot
            reloaded_tearsheet = second.get(f"/api/backtests/{run_id}/tearsheet").json()
            assert reloaded_tearsheet["trades"] == trades

        # Unknown ids are 404 on every export endpoint.
        for suffix in ("export.json", "trades.csv"):
            assert client.get(f"/api/backtests/bt-9999/{suffix}").status_code == 404


def test_legacy_backtest_artifacts_tolerate_missing_snapshots(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _artifact_dirs(settings)
    app = create_app(settings, clock=lambda: FIXED_CLOCK)
    with TestClient(app) as client:
        run = _run_backtest(client)
        run_id = run["id"]
        artifact = settings.backtest_artifacts_dir / f"{run_id}.json"
        payload = json.loads(artifact.read_text(encoding="utf-8"))

    legacy_id = "bt-9001"
    legacy_payload = json.loads(json.dumps(payload))
    legacy_payload["summary"]["id"] = legacy_id
    legacy_payload["summary"].pop("strategy_config", None)
    legacy_payload["tearsheet"]["run_id"] = legacy_id
    legacy_payload["tearsheet"].pop("trades", None)
    (settings.backtest_artifacts_dir / f"{legacy_id}.json").write_text(
        json.dumps(legacy_payload), encoding="utf-8"
    )

    restarted = create_app(settings, clock=lambda: FIXED_CLOCK)
    with TestClient(restarted) as client:
        listed = {row["id"]: row for row in client.get("/api/backtests").json()["runs"]}
        assert legacy_id in listed
        assert listed[legacy_id]["strategy_config"] is None
        detail = client.get(f"/api/backtests/{legacy_id}")
        assert detail.status_code == 200
        assert detail.json()["strategy_config"] is None
        tearsheet = client.get(f"/api/backtests/{legacy_id}/tearsheet").json()
        assert tearsheet["trades"] == []
        export = client.get(f"/api/backtests/{legacy_id}/export.json")
        assert export.status_code == 200
        assert export.json()["strategy_config"] is None
        csv_response = client.get(f"/api/backtests/{legacy_id}/trades.csv")
        assert csv_response.status_code == 200
        lines = csv_response.text.strip().splitlines()
        assert len(lines) == 1, "legacy run without trades must return header-only CSV"


# ---------------------------------------------------------------------------
# 5. Risk/maintenance operator resets (task 54)
# ---------------------------------------------------------------------------


def _reset_settings(tmp_path: Path, *, environment: str = "backtest") -> Settings:
    settings = _settings(tmp_path, environment=environment)
    return settings.model_copy(update={"risk_state_store_uri": str(tmp_path / "risk_state.json")})


def _seed_risk_state(settings: Settings) -> None:
    from src.risk.state_store import PersistedRiskState, build_risk_state_store

    store = build_risk_state_store(settings.risk_state_store_uri)
    store.save(
        PersistedRiskState(
            kill_switch_locked=True,
            kill_switch_reason="operator test lock",
            daily_loss_locked=True,
            starting_daily_nlv=500_000.0,
            portfolio_high_water_mark=123_456.0,
            instrument_high_water_marks={"ES": 1.0},
            extra={"ratelimit.bucket": "keep-me", "quant_overlay.trailing": "drop-me"},
        )
    )


def test_risk_state_fields_and_reset_lock_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FUTURESRL_OPERATOR_RESET_TOKEN", raising=False)
    settings = _reset_settings(tmp_path)
    _seed_risk_state(settings)
    app = create_app(settings, clock=lambda: FIXED_CLOCK)
    with TestClient(app) as client:
        state = client.get("/api/risk/state")
        assert state.status_code == 200
        body = state.json()
        for field in (
            "kill_switch_active",
            "kill_switch_reason",
            "daily_loss_breaker_active",
            "lock_until",
            "reset_available",
            "operator_reset_token_required",
        ):
            assert field in body
        assert body["kill_switch_active"] is True
        assert body["kill_switch_reason"] == "operator test lock"
        assert body["daily_loss_breaker_active"] is True
        assert body["reset_available"] is True
        assert body["operator_reset_token_required"] is False

        # Exact confirmation phrase is required (400), no destructive action.
        bad = client.post("/api/risk/reset-lock", json={"confirm": "reset risk lock"})
        assert bad.status_code == 400
        assert "RESET RISK LOCK" in bad.json()["detail"]

        reset = client.post("/api/risk/reset-lock", json={"confirm": "RESET RISK LOCK"})
        assert reset.status_code == 200, reset.text
        receipt = reset.json()
        assert receipt["status"] == "reset"
        assert receipt["execution_reset"] is True
        assert receipt["persisted_state_reset"] is True
        assert receipt["kill_switch_active"] is False
        assert receipt["daily_loss_breaker_active"] is False

        after = client.get("/api/risk/state").json()
        assert after["kill_switch_active"] is False
        assert after["daily_loss_breaker_active"] is False

        from src.risk.state_store import build_risk_state_store

        persisted = build_risk_state_store(settings.risk_state_store_uri).load()
        assert persisted is not None
        assert persisted.kill_switch_locked is False
        assert persisted.daily_loss_locked is False
        assert persisted.starting_daily_nlv is None
        assert persisted.portfolio_high_water_mark is None
        assert persisted.instrument_high_water_marks == {}
        assert persisted.extra.get("ratelimit.bucket") == "keep-me"
        assert not any(key.startswith("quant_overlay.") for key in persisted.extra)

        again = client.post("/api/risk/reset-lock", json={"confirm": "RESET RISK LOCK"})
        assert again.status_code == 200
        assert again.json()["status"] == "already_unlocked"


def test_reset_lock_operator_token_enforcement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FUTURESRL_OPERATOR_RESET_TOKEN", "s3cret-token")
    try:
        settings = _reset_settings(tmp_path)
        app = create_app(settings, clock=lambda: FIXED_CLOCK)
        with TestClient(app) as client:
            state = client.get("/api/risk/state").json()
            assert state["operator_reset_token_required"] is True
            missing = client.post("/api/risk/reset-lock", json={"confirm": "RESET RISK LOCK"})
            assert missing.status_code == 403
            wrong = client.post(
                "/api/risk/reset-lock",
                json={"confirm": "RESET RISK LOCK", "operator_token": "nope"},
            )
            assert wrong.status_code == 403
            body_token = client.post(
                "/api/risk/reset-lock",
                json={"confirm": "RESET RISK LOCK", "operator_token": "s3cret-token"},
            )
            assert body_token.status_code == 200, body_token.text
            # Header token is accepted too.
            (tmp_path / "risk_state.json").unlink(missing_ok=True)
            header_token = client.post(
                "/api/risk/reset-lock",
                json={"confirm": "RESET RISK LOCK"},
                headers={"X-Operator-Token": "s3cret-token"},
            )
            assert header_token.status_code == 200, header_token.text
    finally:
        monkeypatch.delenv("FUTURESRL_OPERATOR_RESET_TOKEN", raising=False)


def test_paper_state_reset_clears_bridge_and_store(tmp_path: Path) -> None:
    settings = _reset_settings(tmp_path, environment="paper")
    settings.runner_bridge_dir.mkdir(parents=True, exist_ok=True)
    bridge_file = settings.runner_bridge_dir / "runner_state.json"
    bridge_file.write_text('{"runner_id": "qol", "stale": false}', encoding="utf-8")
    _seed_risk_state(settings)
    app = create_app(settings, clock=lambda: FIXED_CLOCK)
    with TestClient(app) as client:
        bad = client.post(
            "/api/maintenance/reset-paper-state", json={"confirm": "reset paper state"}
        )
        assert bad.status_code == 400
        assert bridge_file.is_file(), "bad confirmation must be non-destructive"

        response = client.post(
            "/api/maintenance/reset-paper-state", json={"confirm": "RESET PAPER STATE"}
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["confirm_ok"] is True
        assert body["environment"] == "paper"
        assert body["receipts"]["files_removed"] == [str(bridge_file)]
        assert body["receipts"]["risk_state_store_cleared"] is True
        assert body["receipts"]["paper_session_reset"] is True
        assert body["receipts"]["risk_state_store_uri"] == settings.risk_state_store_uri
        assert not bridge_file.exists()

        from src.risk.state_store import build_risk_state_store

        assert build_risk_state_store(settings.risk_state_store_uri).load() is None

        # Idempotent second call: no files, no store to clear, still a safe 200.
        second = client.post(
            "/api/maintenance/reset-paper-state", json={"confirm": "RESET PAPER STATE"}
        )
        assert second.status_code == 200, second.text
        second_body = second.json()
        assert second_body["receipts"]["files_removed"] == []
        assert second_body["receipts"]["risk_state_store_cleared"] is False


def test_paper_state_reset_refuses_live_environment(tmp_path: Path) -> None:
    settings = _settings(tmp_path, environment="live", ibkr_paper=False, ibkr_port=7496)
    settings = settings.model_copy(
        update={"risk_state_store_uri": str(tmp_path / "risk_state.json")}
    )
    settings.runner_bridge_dir.mkdir(parents=True, exist_ok=True)
    bridge_file = settings.runner_bridge_dir / "runner_state.json"
    bridge_file.write_text('{"runner_id": "live"}', encoding="utf-8")
    app = create_app(settings, clock=lambda: FIXED_CLOCK)
    with TestClient(app) as client:
        response = client.post(
            "/api/maintenance/reset-paper-state", json={"confirm": "RESET PAPER STATE"}
        )
        assert response.status_code == 409, response.text
        assert "non-live" in response.json()["detail"]
    assert bridge_file.is_file(), "live refusal must never destroy operator state"
