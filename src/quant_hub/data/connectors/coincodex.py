"""CoinCodex public endpoints (no key): coin metadata and price/volume/market-cap history.

Undocumented and unversioned: the same source as the RW ``coincodex.marketcap`` snapshot,
so it keeps the universe's coin set and cap definitions across the snapshot's end. Calls
are spaced politely (1/s); errors carry no credentials (there are none).

    get_coin/{SYMBOL}                            name, market_cap_rank, ...
    get_coin_history/{SYMBOL}/{from}/{to}/{n}    [[unix_ts, price, volume, market_cap], ...]
"""

from __future__ import annotations

import time

import httpx
import pandas as pd

BASE = "https://coincodex.com/api/coincodex"
_client = httpx.Client(
    timeout=60, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0 (quant_hub)"}
)
_MIN_INTERVAL = 1.0
_last = 0.0


class CoinCodexError(RuntimeError):
    """HTTP or payload error from CoinCodex."""


def _get(path: str, retries: int = 4):
    global _last
    for attempt in range(retries):
        wait = _MIN_INTERVAL - (time.monotonic() - _last)
        if wait > 0:
            time.sleep(wait)
        _last = time.monotonic()
        try:
            resp = _client.get(f"{BASE}{path}")
        except httpx.TransportError as err:
            if attempt == retries - 1:
                raise CoinCodexError(f"transport error for {path}: {err}") from None
            time.sleep(5 * (attempt + 1))
            continue
        if resp.status_code in (429, 502, 503) and attempt < retries - 1:
            time.sleep(10 * (attempt + 1))
            continue
        if resp.status_code >= 400:
            raise CoinCodexError(f"{resp.status_code} for {path}")
        return resp.json()
    raise RuntimeError("unreachable")


def coin(symbol: str) -> dict | None:
    """Coin metadata, or None when CoinCodex does not know the symbol."""
    try:
        j = _get(f"/get_coin/{symbol}")
    except CoinCodexError:
        return None
    return j if isinstance(j, dict) and j.get("symbol") else None


def daily_history(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Daily rows (Date, Price, Volume, MarketCapUSD) at 00:00 UTC of each day in [start, end].

    The endpoint returns ``n`` roughly evenly spaced samples (not always on the hour); with
    four per day the first sample of each day falls in its first hours. That first sample is
    kept (stamped with the day): it is known before a 09:00 decision on the same day.
    """
    days = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
    j = _get(f"/get_coin_history/{symbol}/{start}/{end}/{4 * days}")
    rows = j.get(symbol) if isinstance(j, dict) else None
    cols = ["Date", "Price", "Volume", "MarketCapUSD"]
    if not rows:
        return pd.DataFrame(columns=cols)
    d = pd.DataFrame(rows, columns=["ts", "Price", "Volume", "MarketCapUSD"])
    d["t"] = pd.to_datetime(d["ts"], unit="s", utc=True).dt.tz_localize(None)
    d["Date"] = d["t"].dt.normalize()
    d = (
        d[d["t"] - d["Date"] < pd.Timedelta(hours=8)]
        .sort_values("t")
        .drop_duplicates("Date", keep="first")
    )
    return d[cols].sort_values("Date").reset_index(drop=True)
