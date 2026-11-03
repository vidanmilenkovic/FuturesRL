"""Single source of truth for the API version surfaced in ``/health``.

Kept in a leaf module (no FastAPI/app imports) so both ``src.api.main`` and the
service container can import it without creating an import cycle.
"""

from __future__ import annotations

from src import __version__

#: API version surfaced in ``/health`` and the OpenAPI document.
API_VERSION: str = __version__
