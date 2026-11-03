"""Phase-10.1 API integration tests: sector-grouped instrument universe endpoint.

The endpoint derives everything from the committed instrument master, so these
tests intentionally avoid hardcoded root lists: they validate the contract,
ordering, totals, micro/standard pair links and JSON serialisability against the
*current* config (which data-eng is expanding concurrently — the ">= 40 roots"
assertion activates automatically once the expansion lands).
"""

from __future__ import annotations

import copy
import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml
from configs.instruments import load_instruments
from configs.settings import Settings
from fastapi.testclient import TestClient
from src.api.main import create_app

pytestmark = pytest.mark.integration

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
FIXED_TIMESTAMP = datetime(2026, 11, 3, 12, 0, tzinfo=UTC)
TELEMETRY_INTERVAL = 0.02

#: Fixed Studio sector order (the UI renders these lanes in this order).
EXPECTED_SECTOR_ORDER: list[str] = [
    "equities",
    "rates",
    "energy",
    "metals",
    "fx",
    "agri",
    "crypto",
]

INSTRUMENT_FIELDS: set[str] = {
    "root",
    "name",
    "asset_class",
    "sector",
    "exchange",
    "multiplier",
    "tick_size",
    "currency",
    "micro_root",
    "standard_root",
}


def _expanded_master_payload() -> dict[str, Any]:
    """Synthesise a 40+ root / 7-sector instrument master from the committed one.

    This keeps the endpoint's expanded-universe behaviour testable before data-eng's
    real task-30 expansion lands (the live-config test still skips until then); the
    payload preserves the instrument master's cross-reference invariants
    (standard <-> micro pairs, micro multiplier < standard multiplier).
    """
    payload: dict[str, Any] = yaml.safe_load(
        (REPO_ROOT / "configs" / "instruments.yaml").read_text(encoding="utf-8")
    )
    instruments: dict[str, Any] = payload["instruments"]

    def clone(
        base_root: str,
        new_root: str,
        *,
        sector: str,
        asset_class: str | None = None,
        exchange: str | None = None,
        micro_root: str | None = None,
        standard_root: str | None = None,
        multiplier: float | None = None,
        tick_size: float | None = None,
    ) -> None:
        entry = copy.deepcopy(instruments[base_root])
        entry["root"] = new_root
        entry["name"] = f"Synthetic {new_root} test contract"
        entry["sector"] = sector
        if asset_class is not None:
            entry["asset_class"] = asset_class
        if exchange is not None:
            entry["exchange"] = exchange
        entry["databento_symbol"] = f"{new_root}.FUT"
        entry["ibkr_exchange"] = exchange or entry["exchange"]
        entry["micro_root"] = micro_root
        entry["standard_root"] = standard_root
        if multiplier is not None:
            entry["multiplier"] = multiplier
        if tick_size is not None:
            entry["tick_size"] = tick_size
        instruments[new_root] = entry

    # equities: 3 standard/micro pairs
    for index in range(3):
        standard, micro = f"EQP{index}", f"EQM{index}"
        clone("ES", standard, sector="equity", micro_root=micro)
        clone("MES", micro, sector="equity", standard_root=standard)
    # rates: 5 singles (incl. one extra to cross 40 total)
    for index in range(5):
        clone("ZN", f"RT{index}", sector="rates")
    # energy: 2 pairs
    for index in range(2):
        standard, micro = f"ENP{index}", f"ENM{index}"
        clone("CL", standard, sector="energy", micro_root=micro)
        clone("MCL", micro, sector="energy", standard_root=standard)
    # metals: 2 pairs
    for index in range(2):
        standard, micro = f"MTP{index}", f"MTM{index}"
        clone("GC", standard, sector="metals", micro_root=micro)
        clone("MGC", micro, sector="metals", standard_root=standard)
    # fx: 3 singles
    for index in range(3):
        clone("6E", f"FX{index}", sector="fx")
    # agriculture: 3 singles (no committed agri root to clone structurally from)
    for index in range(3):
        clone(
            "CL",
            f"AG{index}",
            sector="agriculture",
            asset_class="agriculture",
            exchange="CBOT",
        )
    # crypto: 3 singles
    for index in range(3):
        clone(
            "GC",
            f"CR{index}",
            sector="crypto",
            asset_class="crypto",
            exchange="CME",
        )
    return payload


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        environment="backtest",
        configs_dir=REPO_ROOT / "configs",
        data_dir=tmp_path / "data",
        parquet_dir=tmp_path / "data" / "parquet",
        duckdb_path=tmp_path / "data" / "api.duckdb",
        report_dir=tmp_path / "artifacts",
        instruments_path=REPO_ROOT / "configs" / "instruments.yaml",
        telemetry_interval_seconds=TELEMETRY_INTERVAL,
        api_cors_origins=("http://localhost:3080",),
    )


@pytest.fixture()
def client(tmp_path: Path) -> Iterator[TestClient]:
    with TestClient(create_app(_settings(tmp_path), clock=lambda: FIXED_TIMESTAMP)) as test_client:
        yield test_client


def _expanded_settings(tmp_path: Path) -> Settings:
    """Settings pointing at the synthetic 40+/7-sector instrument master."""
    master_path = tmp_path / "instruments-expanded.yaml"
    master_path.write_text(
        yaml.safe_dump(_expanded_master_payload(), sort_keys=False), encoding="utf-8"
    )
    return Settings(
        environment="backtest",
        configs_dir=REPO_ROOT / "configs",
        data_dir=tmp_path / "data",
        parquet_dir=tmp_path / "data" / "parquet",
        duckdb_path=tmp_path / "data" / "api.duckdb",
        report_dir=tmp_path / "artifacts",
        instruments_path=master_path,
        telemetry_interval_seconds=TELEMETRY_INTERVAL,
        api_cors_origins=("http://localhost:3080",),
    )


def _instruments(client: TestClient) -> dict[str, Any]:
    response = client.get("/api/instruments")
    assert response.status_code == 200, response.text
    body = response.json()
    assert isinstance(body, dict)
    return body


def _instrument_rows(client: TestClient) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    body = _instruments(client)
    by_root = {row["root"]: row for row in body["roots"]}
    return body, by_root


# ---------------------------------------------------------------------------
# Contract / ordering / totals
# ---------------------------------------------------------------------------


def test_instruments_response_contract(client: TestClient) -> None:
    body = _instruments(client)
    assert set(body) >= {"generated_at", "count", "sectors", "roots"}
    assert datetime.fromisoformat(body["generated_at"].replace("Z", "+00:00")).tzinfo is not None
    assert body["count"] > 0
    assert body["count"] == len(body["roots"])
    assert body["count"] == sum(sector["count"] for sector in body["sectors"])

    sector_ids = [sector["id"] for sector in body["sectors"]]
    assert sector_ids[:7] == EXPECTED_SECTOR_ORDER
    assert len(body["sectors"]) >= 7

    for sector in body["sectors"]:
        assert isinstance(sector["id"], str) and sector["id"]
        assert isinstance(sector["label"], str) and sector["label"]
        assert sector["count"] == len(sector["roots"])
        roots = [row["root"] for row in sector["roots"]]
        assert roots == sorted(roots), f"sector {sector['id']} roots are not alphabetical"
        for row in sector["roots"]:
            assert row["sector"] == sector["id"]
            assert set(row) >= INSTRUMENT_FIELDS

    flat_roots = [row["root"] for row in body["roots"]]
    assert flat_roots == sorted(flat_roots)
    assert len(flat_roots) == len(set(flat_roots))

    for row in body["roots"]:
        assert row["root"]
        assert row["name"]
        assert row["asset_class"]
        assert row["exchange"]
        assert row["multiplier"] > 0.0
        assert row["tick_size"] > 0.0
        assert len(row["currency"]) == 3

    # The payload must be plain JSON-serialisable data (no dataclasses/enums leaking).
    serialised = json.dumps(body, sort_keys=True)
    assert json.loads(serialised)["count"] == body["count"]


def test_instruments_match_instrument_master(client: TestClient) -> None:
    body = _instruments(client)
    master = load_instruments(REPO_ROOT / "configs" / "instruments.yaml")
    expected_roots = set(master.roots())
    assert {row["root"] for row in body["roots"]} == expected_roots
    assert body["count"] == len(expected_roots)
    for row in body["roots"]:
        spec = master.get(row["root"])
        assert row["name"] == spec.name
        assert row["exchange"] == spec.exchange
        assert row["multiplier"] == pytest.approx(spec.multiplier)
        assert row["tick_size"] == pytest.approx(spec.tick_size)
        assert row["micro_root"] == spec.micro_root
        assert row["standard_root"] == spec.standard_root


# ---------------------------------------------------------------------------
# Micro / standard pair links (both directions)
# ---------------------------------------------------------------------------


def test_micro_standard_links_both_directions(client: TestClient) -> None:
    _body, by_root = _instrument_rows(client)
    standard_links = [
        (row["root"], row["micro_root"]) for row in by_root.values() if row["micro_root"]
    ]
    micro_links = [
        (row["root"], row["standard_root"]) for row in by_root.values() if row["standard_root"]
    ]
    assert standard_links, "expected at least one standard -> micro pair"
    assert micro_links, "expected at least one micro -> standard pair"
    for standard, micro in standard_links:
        assert micro in by_root, f"micro root {micro} missing from universe"
        assert by_root[micro]["standard_root"] == standard
    for micro, standard in micro_links:
        assert standard in by_root, f"standard root {standard} missing from universe"
        assert by_root[standard]["micro_root"] == micro


def test_sector_groups_cover_the_flat_universe(client: TestClient) -> None:
    body = _instruments(client)
    grouped = {row["root"] for sector in body["sectors"] for row in sector["roots"]}
    assert grouped == {row["root"] for row in body["roots"]}
    # Every known Studio sector is present even before it has roots (empty lanes).
    counts = {sector["id"]: sector["count"] for sector in body["sectors"]}
    for sector_id in EXPECTED_SECTOR_ORDER:
        assert sector_id in counts


# ---------------------------------------------------------------------------
# Expanded universe (activates when data-eng's task-30 expansion lands)
# ---------------------------------------------------------------------------


def test_expanded_synthetic_master_serves_40_plus_roots(tmp_path: Path) -> None:
    """The endpoint scales to the expanded 7-sector/40+ root universe dynamically."""
    settings = _expanded_settings(tmp_path)
    with TestClient(create_app(settings, clock=lambda: FIXED_TIMESTAMP)) as expanded_client:
        body = _instruments(expanded_client)
    assert body["count"] >= 40, body["count"]
    sector_counts = {sector["id"]: sector["count"] for sector in body["sectors"]}
    for sector_id in EXPECTED_SECTOR_ORDER:
        assert sector_counts.get(sector_id, 0) > 0, f"sector {sector_id} has no roots"
    by_root = {row["root"]: row for row in body["roots"]}
    pair_count = 0
    for row in body["roots"]:
        if row["micro_root"]:
            assert by_root[row["micro_root"]]["standard_root"] == row["root"]
            pair_count += 1
    assert pair_count >= 7  # 3 equity + 2 energy + 2 metals pairs


def test_catalog_universe_tracks_expanded_instrument_master(tmp_path: Path) -> None:
    """The catalog's universe/stage metadata follows the current instrument master."""
    settings = _expanded_settings(tmp_path)
    with TestClient(create_app(settings, clock=lambda: FIXED_TIMESTAMP)) as expanded_client:
        catalog = expanded_client.get("/api/strategies/schema/catalog").json()
        instruments = expanded_client.get("/api/instruments").json()
    universe = catalog["universe"]
    assert len(universe["roots"]) == instruments["count"] >= 40
    assert {row["root"] for row in universe["roots"]} == {
        row["root"] for row in instruments["roots"]
    }
    all_sector = universe["sectors"][0]
    assert all_sector["id"] == "all"
    assert all_sector["count"] == len(universe["roots"])
    group_counts = {
        sector["id"]: sector["count"] for sector in universe["sectors"] if sector["id"] != "all"
    }
    for sector in instruments["sectors"]:
        assert group_counts.get(sector["id"]) == sector["count"]
    assert catalog["stage_choices"]["universe"] == [
        "liquidity",
        "regime",
        "composite",
    ]
    assert catalog["stage_choices"]["allocator"] == [
        "volatility_target",
        "risk_parity",
        "equal_weight",
        "mean_variance",
        "drl",
    ]
    # Dynamic root multiselect choices mirror the expanded master (fingerprinted
    # catalog build), not the import-time 45-root snapshot.
    parameters = {parameter["key"]: parameter for parameter in catalog["parameters"]}
    synthetic_seen = False
    for key in (
        "universe.regime.risk_off_roots",
        "timing.seasonality.root_month_biases",
    ):
        parameter = parameters.get(key)
        if parameter is None or not parameter["choices"]:
            continue
        assert any(
            str(choice).upper().startswith(("EQP", "CR", "AG")) for choice in parameter["choices"]
        ), key
        synthetic_seen = True
    assert synthetic_seen, "expected at least one root-multiselect parameter"
    assert json.loads(json.dumps(catalog)) == catalog


def test_expanded_universe_has_at_least_40_roots(client: TestClient) -> None:
    body = _instruments(client)
    if body["count"] < 40:  # pragma: no cover - phase-10.1 expansion in flight
        pytest.skip(
            "instrument master expansion not landed yet "
            f"(count={body['count']} < 40); this test activates with data-eng task-30"
        )
    assert body["count"] >= 40
    sector_counts = {sector["id"]: sector["count"] for sector in body["sectors"]}
    for sector_id in EXPECTED_SECTOR_ORDER:
        assert sector_counts.get(sector_id, 0) > 0, f"sector {sector_id} has no roots"


def test_instruments_endpoint_in_openapi(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert "/api/instruments" in paths
    operation = paths["/api/instruments"]["get"]
    assert operation["tags"] == ["instruments"]
    assert "200" in operation["responses"]
