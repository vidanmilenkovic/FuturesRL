"""FuturesRL telemetry API package.

FastAPI application exposing the FuturesRL platform over REST + WebSocket:

* real-time account/position telemetry and the emergency flatten-all kill switch,
* the FinRL-X weight pipeline waterfall (``S_t → w_base → w_timing → w_risk → N_i,t``),
* the LLM Risk-First assessments, threshold overrides and macro lockout,
* futures term structure and roll calendar,
* declarative strategy configs and deterministic offline backtest tear sheets,
* ``/ws/telemetry`` pushed through :class:`src.core.event_bus.EventBus`.

Public surface::

    from src.api import create_app, app

Run with ``uvicorn src.api.main:app --port 8000`` or the ``futuresrl-api`` script.
"""

from __future__ import annotations

from src.api.main import app, create_app, main
from src.api.version import API_VERSION

__all__ = ["API_VERSION", "app", "create_app", "main"]
