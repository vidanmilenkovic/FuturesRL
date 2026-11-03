"""FuturesRL-X: modular algorithmic futures trading platform.

Package layout mirrors the architecture specification: ``core`` holds the
immutable contracts, ``data`` the Databento/IBKR market data layer, and the
remaining packages implement the FinRL-X pipeline, risk overlays, backtesting,
IBKR execution, analytics and the FastAPI telemetry API.
"""

__version__ = "1.5.3"
