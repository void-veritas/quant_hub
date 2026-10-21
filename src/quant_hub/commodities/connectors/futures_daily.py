"""Daily bars of exchange-traded commodity futures from free sources (Yahoo, Stooq).

Both serve a *continuous front-month* series, not individual contracts: the price
jumps at each roll by the spread between the expiring and the next contract. Returns
across a roll date are therefore not tradeable returns; use them for trend/vol
research with that caveat, never for carry (which needs at least two contracts).
`ts` = trading date (UTC midnight), bar = that session.
"""

from __future__ import annotations

import io
import time

import httpx
import pandas as pd

# canonical root -> (yahoo symbol, stooq symbol, exchange, description)
COMMODITIES = {
    "HG": ("HG=F", "hg.f", "COMEX", "copper"),
    "GC": ("GC=F", "gc.f", "COMEX", "gold"),
    "SI": ("SI=F", "si.f", "COMEX", "silver"),
    "PL": ("PL=F", "pl.f", "NYMEX", "platinum"),
    "PA": ("PA=F", "pa.f", "NYMEX", "palladium"),
    "ALI": ("ALI=F", None, "COMEX", "aluminum"),
    "CL": ("CL=F", "cl.f", "NYMEX", "WTI crude"),
    "BZ": ("BZ=F", "cb.f", "NYMEX", "Brent crude"),
    "NG": ("NG=F", "ng.f", "NYMEX", "natural gas"),
    "HO": ("HO=F", "ho.f", "NYMEX", "heating oil"),
    "RB": ("RB=F", "rb.f", "NYMEX", "RBOB gasoline"),
    "ZC": ("ZC=F", "zc.f", "CBOT", "corn"),
    "ZW": ("ZW=F", "zw.f", "CBOT", "wheat"),
    "ZS": ("ZS=F", "zs.f", "CBOT", "soybeans"),
    "ZM": ("ZM=F", "zm.f", "CBOT", "soybean meal"),
    "ZL": ("ZL=F", "zl.f", "CBOT", "soybean oil"),
    "KC": ("KC=F", "kc.f", "ICE US", "coffee"),
    "SB": ("SB=F", "sb.f", "ICE US", "sugar no. 11"),
    "CC": ("CC=F", "cc.f", "ICE US", "cocoa"),
    "CT": ("CT=F", "ct.f", "ICE US", "cotton"),
    "LE": ("LE=F", "le.f", "CME", "live cattle"),
    "HE": ("HE=F", "he.f", "CME", "lean hogs"),
    "GF": ("GF=F", "gf.f", "CME", "feeder cattle"),
}

COLUMNS = ["ts", "open", "high", "low", "close", "volume", "symbol"]
YAHOO = "https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
STOOQ = "https://stooq.com/q/d/l/?s={symbol}&i=d"

_client = httpx.Client(
    timeout=60, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0 (quant_hub research)"}
)


def _get(url: str, params: dict | None = None, retries: int = 4) -> httpx.Response:
    for attempt in range(retries):
        try:
            resp = _client.get(url, params=params)
            resp.raise_for_status()
            return resp
        except (httpx.TransportError, httpx.HTTPStatusError):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError("unreachable")


def parse_yahoo(payload: dict, symbol: str) -> pd.DataFrame:
    """Yahoo chart API JSON -> daily bars; rows with a missing close are dropped."""
    res = (payload.get("chart") or {}).get("result") or []
    if not res or not res[0].get("timestamp"):
        return pd.DataFrame(columns=COLUMNS)
    r = res[0]
    q = r["indicators"]["quote"][0]
    tz = r.get("meta", {}).get("exchangeTimezoneName", "America/New_York")
    # Yahoo stamps a daily bar at the session open in exchange time; keep the date only
    dates = (
        pd.to_datetime(r["timestamp"], unit="s", utc=True)
        .tz_convert(tz)
        .normalize()
        .tz_localize(None)
    )
    out = pd.DataFrame(
        {
            "ts": pd.to_datetime(dates).tz_localize("UTC"),
            "open": q.get("open"),
            "high": q.get("high"),
            "low": q.get("low"),
            "close": q.get("close"),
            "volume": q.get("volume"),
            "symbol": symbol,
        }
    )
    out = out.dropna(subset=["close"])
    return out.drop_duplicates(subset="ts", keep="last").sort_values("ts").reset_index(drop=True)


def parse_stooq(text: str, symbol: str) -> pd.DataFrame:
    """Stooq CSV (Date,Open,High,Low,Close,Volume) -> daily bars."""
    if not text.lstrip().lower().startswith("date"):
        raise ValueError(f"stooq returned no CSV for {symbol}: {text[:120]!r}")
    df = pd.read_csv(io.StringIO(text))
    df.columns = [c.strip().lower() for c in df.columns]
    out = pd.DataFrame(
        {
            "ts": pd.to_datetime(df["date"]).dt.tz_localize("UTC"),
            "open": df["open"],
            "high": df["high"],
            "low": df["low"],
            "close": df["close"],
            "volume": df.get("volume"),
            "symbol": symbol,
        }
    )
    return out.dropna(subset=["close"]).sort_values("ts").reset_index(drop=True)


def fetch_yahoo(symbol: str, start: str = "1990-01-01", end: str | None = None) -> pd.DataFrame:
    """Daily bars from the Yahoo chart API (front-month continuous, from ~2000)."""
    p1 = int(pd.Timestamp(start, tz="UTC").timestamp())
    p2 = int((pd.Timestamp(end, tz="UTC") if end else pd.Timestamp.now(tz="UTC")).timestamp())
    resp = _get(
        YAHOO.format(symbol=symbol), params={"period1": p1, "period2": p2, "interval": "1d"}
    )
    return parse_yahoo(resp.json(), symbol)


def fetch_stooq(symbol: str) -> pd.DataFrame:
    """Full daily history from Stooq (front-month continuous; often longer than Yahoo)."""
    return parse_stooq(_get(STOOQ.format(symbol=symbol)).text, symbol)
