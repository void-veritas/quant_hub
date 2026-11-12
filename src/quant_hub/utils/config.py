"""Project configuration: paths and environment variables.

quant_hub is used two ways, and the home (data layer + `.env`) follows:

- from its own checkout (`uv sync` here, the server image, CI): the checkout's root, as always;
- installed as a dependency of another project (`uv add` from git): `~/.quant_hub`, one data layer
  per machine that every project shares (`~/.quant_hub/data`, secrets in `~/.quant_hub/.env`).

`QUANT_HUB_HOME` overrides both. Call `get_env("FMP_API_KEY")` from anywhere — notebooks,
connectors, scripts — and the .env file is loaded on first use.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

_SOURCE_ROOT = Path(__file__).resolve().parents[3]


def _home() -> Path:
    if os.environ.get("QUANT_HUB_HOME"):
        return Path(os.environ["QUANT_HUB_HOME"]).expanduser()
    if (_SOURCE_ROOT / "pyproject.toml").exists() and (_SOURCE_ROOT / "src" / "quant_hub").is_dir():
        return _SOURCE_ROOT  # a checkout of this repo (editable install)
    return Path.home() / ".quant_hub"  # installed into another project's environment


PROJECT_ROOT = _home()


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
