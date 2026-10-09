"""Robot Wealth API (api.robotwealth.com/v1): YOLO reference data and bulk datasets.

Auth: query parameter `api_key`, read from env `RW_PRO` (never logged). Rate
limits: ~10 requests/min on YOLO endpoints; bulk files are served through
short-lived (5 min) signed CDN URLs with a 20 GiB cumulative bandwidth cap per
key, so download each file once and keep it (sync to the bucket).

Endpoints used:
    /yolo/weights, /yolo/factors, /yolo/volatilities   latest production values
    /yolo/historical?days=N                            megafactors, combo weight,
                                                       EWMA vol, arrival price per day
    /crypto/datasets                                   catalogue of bulk datasets
    /crypto/file?dataset=&schema=                      signed URL for one file
    /crypto/binance/perps/funding?gte=&lte=&ticker=    live funding (cursor-paginated)
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import httpx
import pandas as pd

BASE = "https://api.robotwealth.com/v1"
_client = httpx.Client(timeout=120, follow_redirects=True)


def _key() -> str:
    key = os.environ.get("RW_PRO", "").strip().strip('"')
    if not key:
        raise RuntimeError("RW_PRO (Robot Wealth API key) is not set in the environment")
    return key


def _get(path: str, params: dict | None = None, retries: int = 4) -> dict:
    params = {"api_key": _key(), **(params or {})}
    for attempt in range(retries):
        resp = _client.get(f"{BASE}{path}", params=params)
        if resp.status_code == 429 and attempt < retries - 1:
            time.sleep(15 * (attempt + 1))
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError("unreachable")


def yolo_weights() -> pd.DataFrame:
    """Latest production weights: ticker, date, arrival_price, momentum/trend/carry, combo."""
    return pd.DataFrame(_get("/yolo/weights")["data"])


def yolo_factors() -> pd.DataFrame:
    """Latest sub-factor values (long: ticker, date, factor_name, value)."""
    return pd.DataFrame(_get("/yolo/factors")["data"])


def yolo_volatilities() -> pd.DataFrame:
    return pd.DataFrame(_get("/yolo/volatilities")["data"])


def yolo_historical(days: int = 365) -> pd.DataFrame:
    """Daily megafactors, combo weight, EWMA vol and arrival price for the last `days`."""
    out = _get("/yolo/historical", {"days": days})
    df = pd.DataFrame(out["data"])
    df.attrs["last_updated"] = out.get("last_updated")
    return df


def list_datasets(namespace: str = "crypto") -> pd.DataFrame:
    """Catalogue of (dataset, schema) tuples the bulk endpoint can serve."""
    out = _get(f"/{namespace}/datasets")
    rows = out.get("data", out.get("datasets", out))
    return pd.DataFrame(rows)


def download_dataset(dataset: str, schema: str, dest: Path, namespace: str = "crypto") -> Path:
    """Fetch a signed URL and stream the file to `dest` (bandwidth is charged once)."""
    out = _get(f"/{namespace}/file", {"dataset": dataset, "schema": schema})
    url = out.get("url") or out.get("signed_url") or out.get("data", {}).get("url")
    if not url:
        raise RuntimeError(f"no url in response: {list(out)}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with _client.stream("GET", url) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
    return dest


def binance_perps_funding(gte: str, lte: str, ticker: str | None = None) -> pd.DataFrame:
    """Live Binance USDT-M funding between dates (inclusive), all pages."""
    rows, cursor = [], None
    while True:
        params = {"gte": gte, "lte": lte}
        if ticker:
            params["ticker"] = ticker
        if cursor:
            params["cursor"] = cursor
        out = _get("/crypto/binance/perps/funding", params)
        rows.extend(out.get("data", []))
        cursor = (out.get("pagination") or {}).get("next_cursor")
        if not cursor:
            break
        time.sleep(0.5)
    return pd.DataFrame(rows)
