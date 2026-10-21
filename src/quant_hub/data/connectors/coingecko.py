"""CoinGecko public API (Demo key): current markets and daily market-cap history.

Auth: header ``x-cg-demo-api-key`` from env ``COINGECKO_API_KEY`` (never logged).
Demo limits: ~30 calls/min, 10,000 calls/month, history only for the last 365 days.
``market_chart`` daily points are stamped 00:00 UTC of each day (value as of that
time, so known before a 09:00 decision on the same day).
"""

from __future__ import annotations

import os
import time

import httpx
import pandas as pd

BASE = "https://api.coingecko.com/api/v3"
_client = httpx.Client(timeout=60)
_MIN_INTERVAL = 2.2  # seconds between calls: ~27/min, under the Demo cap
_last = 0.0


class CoinGeckoError(RuntimeError):
    """HTTP error from CoinGecko (the key travels in a header, never in the URL)."""


def _key() -> str:
    key = os.environ.get("COINGECKO_API_KEY", "").strip().strip("\"'")
    if not key:
        raise RuntimeError("COINGECKO_API_KEY is not set in the environment")
    return key


def _get(path: str, params: dict | None = None, retries: int = 5):
    global _last
    for attempt in range(retries):
        wait = _MIN_INTERVAL - (time.monotonic() - _last)
        if wait > 0:
            time.sleep(wait)
        _last = time.monotonic()
        resp = _client.get(f"{BASE}{path}", params=params, headers={"x-cg-demo-api-key": _key()})
        if resp.status_code == 429 and attempt < retries - 1:
            time.sleep(30 * (attempt + 1))
            continue
        if resp.status_code >= 400:
            raise CoinGeckoError(f"{resp.status_code} for {path}: {resp.text[:200]}")
        return resp.json()
    raise RuntimeError("unreachable")


def markets(pages: int = 2, per_page: int = 250, ids: list[str] | None = None) -> pd.DataFrame:
    """Current market data ranked by market cap (or for the given ids, up to 250 per call)."""
    rows = []
    if ids:
        for i in range(0, len(ids), 250):
            rows += _get(
                "/coins/markets",
                {"vs_currency": "usd", "ids": ",".join(ids[i : i + 250]), "per_page": 250},
            )
    else:
        for page in range(1, pages + 1):
            rows += _get(
                "/coins/markets",
                {
                    "vs_currency": "usd",
                    "order": "market_cap_desc",
                    "per_page": per_page,
                    "page": page,
                },
            )
    cols = [
        "id",
        "symbol",
        "name",
        "market_cap",
        "market_cap_rank",
        "current_price",
        "total_volume",
        "last_updated",
    ]
    df = pd.DataFrame(rows)
    return df.reindex(columns=cols) if not df.empty else pd.DataFrame(columns=cols)


def coins_list() -> pd.DataFrame:
    """Every coin id with its symbol and name (one call)."""
    return pd.DataFrame(_get("/coins/list"))


def market_chart(coin_id: str, days: int = 365) -> pd.DataFrame:
    """Daily history (Date, Price, Volume, MarketCapUSD) for the last `days` (Demo: <= 365)."""
    j = _get(
        f"/coins/{coin_id}/market_chart", {"vs_currency": "usd", "days": days, "interval": "daily"}
    )

    def frame(key: str, name: str) -> pd.DataFrame:
        d = pd.DataFrame(j.get(key, []), columns=["ms", name])
        d["Date"] = pd.to_datetime(d["ms"], unit="ms", utc=True).dt.floor("D").dt.tz_localize(None)
        return d.drop(columns="ms").drop_duplicates("Date", keep="first")

    out = frame("market_caps", "MarketCapUSD").merge(
        frame("prices", "Price"), on="Date", how="outer"
    )
    out = out.merge(frame("total_volumes", "Volume"), on="Date", how="outer").sort_values("Date")
    return out[["Date", "Price", "Volume", "MarketCapUSD"]].reset_index(drop=True)
