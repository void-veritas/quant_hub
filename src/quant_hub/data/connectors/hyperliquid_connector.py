"""Hyperliquid perps via the public info API.

Note: the candle API serves limited recent history only — deep history comes
from the S3 archives (hyperliquid-archive / Reservoir), wired up separately.
Funding history is fully paginatable through the API.

HL API timestamps are epoch ms; candle "t" is bar OPEN time, matching our
project-wide convention. All frames return a UTC `ts` column.
"""

from __future__ import annotations

import json
import time
from datetime import datetime

import httpx
import pandas as pd

from quant_hub.data.connectors import scrapingbee

INFO_URL = "https://api.hyperliquid.xyz/info"

_client = httpx.Client(timeout=60)


def _post(payload: dict, retries: int = 6) -> httpx.Response:
    """POST to the info API, via ScrapingBee's IP pool when enabled."""
    for attempt in range(retries):
        try:
            if scrapingbee.enabled:
                resp = scrapingbee.request("POST", INFO_URL, content=json.dumps(payload))
            else:
                resp = _client.post(INFO_URL, json=payload)
            resp.raise_for_status()
            return resp
        except (httpx.TransportError, httpx.HTTPStatusError) as err:
            if attempt == retries - 1:
                raise
            rate_limited = (
                isinstance(err, httpx.HTTPStatusError) and err.response.status_code == 429
            )
            # HL allows ~1200 weight/min per IP; a 429 means back off hard
            time.sleep(10 * 2**attempt if rate_limited else 2**attempt)
    raise RuntimeError("unreachable")


def fetch_meta() -> list[dict]:
    """Perp universe: name, szDecimals, maxLeverage, isDelisted flag."""
    return _post({"type": "meta"}).json()["universe"]


def fetch_funding(coin: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Hourly funding history; paginates (API caps ~500 rows per call)."""
    rows: list[dict] = []
    cursor = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    while cursor < end_ms:
        batch = _post(
            {"type": "fundingHistory", "coin": coin, "startTime": cursor, "endTime": end_ms}
        ).json()
        if not batch:
            break
        rows.extend(batch)
        last = batch[-1]["time"]
        if last <= cursor:
            break
        cursor = last + 1
        if not scrapingbee.enabled:  # proxied requests come from rotating IPs
            time.sleep(1.1)  # info requests weigh 20 of the 1200/min budget
    if not rows:
        return pd.DataFrame(columns=["ts", "rate", "premium", "symbol"])
    df = pd.DataFrame(rows)
    out = pd.DataFrame(
        {
            "ts": pd.to_datetime(df["time"].astype("int64"), unit="ms", utc=True),
            "rate": pd.to_numeric(df["fundingRate"]),
            "premium": pd.to_numeric(df["premium"], errors="coerce"),
            "symbol": coin,
        }
    )
    return out.drop_duplicates(subset="ts").sort_values("ts")


def fetch_candles(coin: str, interval: str, start: datetime, end: datetime) -> pd.DataFrame:
    """OHLCV candles (recent history only — see module docstring)."""
    batch = _post(
        {
            "type": "candleSnapshot",
            "req": {
                "coin": coin,
                "interval": interval,
                "startTime": int(start.timestamp() * 1000),
                "endTime": int(end.timestamp() * 1000),
            },
        }
    ).json()
    if not batch:
        return pd.DataFrame(
            columns=["ts", "open", "high", "low", "close", "volume", "trades", "symbol"]
        )
    df = pd.DataFrame(batch)
    out = pd.DataFrame(
        {
            "ts": pd.to_datetime(df["t"].astype("int64"), unit="ms", utc=True),
            "open": pd.to_numeric(df["o"]),
            "high": pd.to_numeric(df["h"]),
            "low": pd.to_numeric(df["l"]),
            "close": pd.to_numeric(df["c"]),
            "volume": pd.to_numeric(df["v"]),
            "trades": pd.to_numeric(df["n"]).astype("int64"),
            "symbol": coin,
        }
    )
    return out.drop_duplicates(subset="ts").sort_values("ts")
