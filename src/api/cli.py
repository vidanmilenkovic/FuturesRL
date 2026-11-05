"""Console entry point for the FuturesRL-X FastAPI service."""

from __future__ import annotations

import sys

from src.api.main import main as _serve


def main() -> int:
    """Run the API server (``futuresrlx-api``) with configured settings."""
    _serve()
    return 0


if __name__ == "__main__":  # pragma: no cover - console entry
    sys.exit(main())
