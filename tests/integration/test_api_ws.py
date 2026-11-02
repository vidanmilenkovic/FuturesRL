"""Integration tests for the ``/ws/telemetry`` WebSocket broadcaster.

The socket is driven by the real ``TelemetryBroadcaster`` (EventBus + background
publisher task) at a 20 ms interval with a fixed clock, so the tests are fully
offline, deterministic and fast.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from configs.settings import Settings
from fastapi.testclient import TestClient
from src.api.main import create_app

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]

FIXED_TIMESTAMP = datetime(2026, 11, 2, 17, 0, tzinfo=UTC)
TELEMETRY_INTERVAL = 0.02

#: How many frames to read before asserting the publisher cycle is complete.
FRAME_BUDGET = 24

#: Payload keys required by each telemetry frame kind (the frozen frontend contract).
REQUIRED_PAYLOAD_KEYS: dict[str, set[str]] = {
    "pnl": {"net_liquidation", "day_pnl", "day_pnl_pct", "point"},
    "margin": {
        "initial_margin_used",
        "maintenance_margin_used",
        "margin_utilization",
        "warning_threshold",
        "critical_threshold",
    },
    "positions": {"positions", "gross_leverage", "net_exposure"},
    "risk_alert": {
        "assessments",
        "circuit_breaker_active",
        "circuit_breaker_reason",
        "macro_lockout",
    },
    "pipeline": {"pipeline"},
    "heartbeat": {"status", "sequence"},
}


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TestClient]:
    tmp_path = tmp_path_factory.mktemp("api-ws")
    settings = Settings(
        environment="backtest",
        data_dir=tmp_path / "data",
        parquet_dir=tmp_path / "data" / "parquet",
        duckdb_path=tmp_path / "data" / "api.duckdb",
        report_dir=tmp_path / "artifacts",
        instruments_path=REPO_ROOT / "configs" / "instruments.yaml",
        telemetry_interval_seconds=TELEMETRY_INTERVAL,
        api_cors_origins=("http://localhost:3080",),
    )
    app = create_app(settings, clock=lambda: FIXED_TIMESTAMP)
    with TestClient(app) as test_client:
        yield test_client


def test_websocket_handshake_and_envelope_contract(client: TestClient) -> None:
    with client.websocket_connect("/ws/telemetry") as websocket:
        first = websocket.receive_json()
        assert first["type"] == "telemetry"
        assert first["kind"] == "heartbeat"
        assert first["sequence"] >= 1
        assert first["timestamp"].startswith("2026-11-02T17:00:00")
        assert set(first["payload"]) >= REQUIRED_PAYLOAD_KEYS["heartbeat"]

        seen_kinds: dict[str, dict[str, object]] = {}
        previous_sequence = first["sequence"]
        for _ in range(FRAME_BUDGET):
            frame = websocket.receive_json()
            assert frame["type"] == "telemetry"
            assert frame["kind"] in REQUIRED_PAYLOAD_KEYS
            assert frame["sequence"] > previous_sequence
            previous_sequence = frame["sequence"]
            assert frame["timestamp"].startswith("2026-11-02T17:00:00")
            assert set(frame["payload"]) >= REQUIRED_PAYLOAD_KEYS[frame["kind"]]
            seen_kinds.setdefault(frame["kind"], frame["payload"])

    # The publisher cycle must cover the full telemetry surface within one budget.
    assert {"heartbeat", "pnl", "margin", "positions"} <= set(seen_kinds)
    pnl = seen_kinds["pnl"]
    assert isinstance(pnl["net_liquidation"], float | int)
    assert pnl["net_liquidation"] > 0
    margin = seen_kinds["margin"]
    assert 0.0 <= float(margin["margin_utilization"]) < 1.0  # type: ignore[arg-type]
    positions = seen_kinds["positions"]
    assert isinstance(positions["positions"], list)


def test_websocket_reports_connected_client_and_cleans_up(client: TestClient) -> None:
    baseline = client.get("/api/telemetry/status").json()
    assert baseline["stream_active"] is True
    assert baseline["connected_clients"] == 0

    with client.websocket_connect("/ws/telemetry") as websocket:
        assert websocket.receive_json()["kind"] == "heartbeat"
        live = client.get("/api/telemetry/status").json()
        assert live["connected_clients"] == 1
        assert live["published_events"] >= baseline["published_events"]

    # Closing the socket must unregister the client queue.
    after = client.get("/api/telemetry/status").json()
    assert after["connected_clients"] == 0


def test_websocket_streams_risk_and_pipeline_frames(client: TestClient) -> None:
    """The slower-cadence frames (risk_alert / pipeline) must reach subscribers."""
    with client.websocket_connect("/ws/telemetry") as websocket:
        websocket.receive_json()  # immediate heartbeat
        seen: set[str] = set()
        for _ in range(FRAME_BUDGET * 2):
            frame = websocket.receive_json()
            seen.add(frame["kind"])
            if {"risk_alert", "pipeline"} <= seen:
                break
        assert {"risk_alert", "pipeline"} <= seen

    # Frames must be JSON-serialisable and carry the envelope sequence.
    assert seen
