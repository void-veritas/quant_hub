"""Project configuration: paths and environment variables.

Secrets live in the gitignored `.env` at the repo root (template: `.env.example`).
Call `get_env("FMP_API_KEY")` from anywhere — notebooks, connectors, scripts —
and the .env file is loaded on first use.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@lru_cache(maxsize=1)
def load_env() -> bool:
    """Load the repo-root .env into os.environ (no-op if already loaded)."""
    return load_dotenv(PROJECT_ROOT / ".env")


def get_env(key: str, default: str | None = None) -> str:
    """Return an environment variable, loading .env first.

    Raises KeyError with a helpful message if the variable is missing
    and no default is given.
    """
    load_env()
    value = os.environ.get(key, default)
    if value is None:
        raise KeyError(f"{key} not set. Add it to {PROJECT_ROOT / '.env'} (see .env.example).")
    return value
