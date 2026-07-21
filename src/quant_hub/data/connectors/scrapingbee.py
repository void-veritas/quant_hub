"""Optional ScrapingBee proxy transport for connectors.

Routes connector HTTP calls through ScrapingBee's IP pool instead of the
local IP — avoids per-IP venue rate limits (notably Hyperliquid's, which is
shared with the user's browser session) and allows concurrent fetching up to
the plan's stream limit.

Enabled per-run with the ingestion CLI flag `--proxy scrapingbee`; requires
SCRAPING_BEE_KEY in .env. Costs 1 credit per request (render_js=false).
Check remaining credits with `usage()`.
"""

from __future__ import annotations

import httpx

from quant_hub.utils.config import get_env

API = "https://app.scrapingbee.com/api/v1"

enabled = False  # toggled by the ingestion CLI, read by the venue connectors

_client = httpx.Client(timeout=90)


def request(
    method: str, url: str, *, params: dict | None = None, content: bytes | str | None = None
) -> httpx.Response:
    """Proxy one HTTP request through ScrapingBee, returning the target's response.

    `transparent_status_code` passes the target's real status through (a 404
    from a missing dump file stays a 404 instead of a ScrapingBee 500).
    """
    target = str(httpx.URL(url, params=params or {}))
    headers = {"Spb-Content-Type": "application/json"} if content is not None else {}
    return _client.request(
        method,
        API,
        params={
            "api_key": get_env("SCRAPING_BEE_KEY"),
            "url": target,
            "render_js": "false",
            "forward_headers": "true",
            "transparent_status_code": "true",
        },
        headers=headers,
        content=content,
    )


def usage() -> dict:
    """Remaining credits and concurrency allowance (free call)."""
    resp = httpx.get(f"{API}/usage", params={"api_key": get_env("SCRAPING_BEE_KEY")}, timeout=30)
    resp.raise_for_status()
    return resp.json()
