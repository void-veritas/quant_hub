"""Binance USDT-M perps: bulk dumps (data.binance.vision) + light REST.

Bulk dumps are the primary backfill path (klines, metrics/open-interest,
monthly funding); REST is used where dumps don't exist (current-month funding,
exchangeInfo). Note that fapi.binance.com is geo-blocked (HTTP 451) from some
locations, while data.binance.vision is not — prefer dumps wherever they exist.
All returned frames use a UTC `ts` column = bar OPEN time / event time,
and keep venue-native column values otherwise.
"""

from __future__ import annotations

import io
import time
import zipfile
from datetime import UTC, date, datetime, timedelta
from xml.etree import ElementTree

import httpx
import pandas as pd

from quant_hub.data.connectors import scrapingbee

VISION = "https://data.binance.vision/data"
VISION_S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
FAPI = "https://fapi.binance.com/fapi/v1"

KLINE_COLS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]

_client = httpx.Client(timeout=60, follow_redirects=True)


def _get(url: str, params: dict | None = None, retries: int = 3) -> httpx.Response | None:
    """GET with retries; returns None on 404 (missing dump file, not an error)."""
    for attempt in range(retries):
        try:
            if scrapingbee.enabled:
                resp = scrapingbee.request("GET", url, params=params)
            else:
                resp = _client.get(url, params=params)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp
        except (httpx.TransportError, httpx.HTTPStatusError):
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
    return None


def _read_zipped_csv(content: bytes, names: list[str] | None = None) -> pd.DataFrame:
    """Read the single CSV inside a dump zip, tolerating optional header rows."""
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        raw = zf.read(zf.namelist()[0])
    first_line = raw.split(b"\n", 1)[0]
    has_header = not first_line.split(b",")[0].strip().replace(b".", b"").isdigit()
    if names is not None:
        return pd.read_csv(io.BytesIO(raw), names=names, header=0 if has_header else None)
    return pd.read_csv(io.BytesIO(raw), header=0 if has_header else None)


def _to_utc(series: pd.Series) -> pd.Series:
    """Epoch ints in ms (pre-2025 dumps) or µs (2025+) -> UTC timestamps."""
    unit = "us" if series.iloc[0] > 1e14 else "ms"
    return pd.to_datetime(series, unit=unit, utc=True)


def _parse_klines(content: bytes, symbol: str) -> pd.DataFrame:
    df = _read_zipped_csv(content, names=KLINE_COLS)
    out = pd.DataFrame({"ts": _to_utc(df["open_time"])})
    for col in [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "taker_buy_volume",
        "taker_buy_quote_volume",
    ]:
        out[col] = pd.to_numeric(df[col])
    out["trades"] = pd.to_numeric(df["trades"]).astype("int64")
    out["symbol"] = symbol
    return out


def fetch_klines_month(symbol: str, interval: str, year: int, month: int) -> pd.DataFrame | None:
    """Fetch one month of klines from the monthly dump; None if the file is absent."""
    base = f"{VISION}/futures/um/monthly/klines/{symbol}/{interval}"
    resp = _get(f"{base}/{symbol}-{interval}-{year}-{month:02d}.zip")
    return _parse_klines(resp.content, symbol) if resp else None


def fetch_klines_day(symbol: str, interval: str, day: date) -> pd.DataFrame | None:
    """Fetch one day of klines from the daily dump; None if the file is absent."""
    base = f"{VISION}/futures/um/daily/klines/{symbol}/{interval}"
    resp = _get(f"{base}/{symbol}-{interval}-{day.isoformat()}.zip")
    return _parse_klines(resp.content, symbol) if resp else None


def fetch_metrics_day(symbol: str, day: date) -> pd.DataFrame | None:
    """Daily metrics dump: 5-min open interest + long/short ratio snapshots."""
    url = f"{VISION}/futures/um/daily/metrics/{symbol}/{symbol}-metrics-{day.isoformat()}.zip"
    resp = _get(url)
    if resp is None:
        return None
    df = _read_zipped_csv(resp.content)
    df.columns = [c.strip() for c in df.columns]
    out = pd.DataFrame({"ts": pd.to_datetime(df["create_time"], utc=True, format="mixed")})
    for col in df.columns:
        if col not in ("create_time", "symbol"):
            out[col] = pd.to_numeric(df[col], errors="coerce")
    out["symbol"] = symbol
    return out


def fetch_funding_month(symbol: str, year: int, month: int) -> pd.DataFrame | None:
    """One month of funding events from the monthly dump; None if absent.

    Dump columns: calc_time (epoch ms), funding_interval_hours, last_funding_rate.
    Monthly files appear a few days after month end; there are no daily dumps,
    so the current month is never available from this path.
    """
    base = f"{VISION}/futures/um/monthly/fundingRate/{symbol}"
    resp = _get(f"{base}/{symbol}-fundingRate-{year}-{month:02d}.zip")
    if resp is None:
        return None
    df = _read_zipped_csv(resp.content)
    df.columns = [c.strip() for c in df.columns]
    out = pd.DataFrame(
        {
            "ts": pd.to_datetime(df["calc_time"].astype("int64"), unit="ms", utc=True),
            "rate": pd.to_numeric(df["last_funding_rate"]),
            "interval_hours": pd.to_numeric(df["funding_interval_hours"]).astype("int64"),
            "symbol": symbol,
        }
    )
    # calc_time carries a few ms of jitter (…00001, …00002); settlement is on the hour
    out["ts"] = out["ts"].dt.round("1min")
    return out.drop_duplicates(subset="ts").sort_values("ts")


def fetch_funding_dumps(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Funding events from monthly dumps covering [start, end); missing months skipped."""
    frames = []
    for year, month in month_range(start.date(), end.date()):
        df = fetch_funding_month(symbol, year, month)
        if df is not None:
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["ts", "rate", "interval_hours", "symbol"])
    out = pd.concat(frames, ignore_index=True)
    return out[(out["ts"] >= start) & (out["ts"] < end)].reset_index(drop=True)


def fetch_funding(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Funding history via REST (fapi). Paginates in 1000s. Geo-blocked in some regions."""
    rows: list[dict] = []
    cursor = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    while cursor < end_ms:
        resp = _get(
            f"{FAPI}/fundingRate",
            params={"symbol": symbol, "startTime": cursor, "endTime": end_ms, "limit": 1000},
        )
        batch = resp.json() if resp else []
        if not batch:
            break
        rows.extend(batch)
        last = batch[-1]["fundingTime"]
        if len(batch) < 1000:
            break
        cursor = last + 1
        time.sleep(0.25)  # stay well under fapi rate limits
    if not rows:
        return pd.DataFrame(columns=["ts", "rate", "mark_price", "symbol"])
    df = pd.DataFrame(rows)
    out = pd.DataFrame(
        {
            "ts": pd.to_datetime(df["fundingTime"].astype("int64"), unit="ms", utc=True),
            "rate": pd.to_numeric(df["fundingRate"]),
            "mark_price": pd.to_numeric(df.get("markPrice"), errors="coerce"),
            "symbol": symbol,
        }
    )
    return out.drop_duplicates(subset="ts").sort_values("ts")


def fetch_exchange_info() -> list[dict]:
    """Current USDT-M symbols with metadata (onboardDate, status, filters)."""
    resp = _get(f"{FAPI}/exchangeInfo")
    return resp.json()["symbols"]


def list_dump_symbols(prefix: str = "data/futures/um/monthly/klines/") -> list[str]:
    """Every symbol that ever produced a dump — includes delisted perps."""
    symbols: list[str] = []
    marker = ""
    while True:
        resp = _client.get(VISION_S3, params={"delimiter": "/", "prefix": prefix, "marker": marker})
        resp.raise_for_status()
        root = ElementTree.fromstring(resp.content)
        ns = {"s3": root.tag.split("}")[0].strip("{")}
        prefixes = [el.text for el in root.findall(".//s3:CommonPrefixes/s3:Prefix", ns)]
        symbols.extend(p.rstrip("/").rsplit("/", 1)[-1] for p in prefixes)
        if root.findtext("s3:IsTruncated", "false", ns) != "true":
            break
        marker = root.findtext("s3:NextMarker", "", ns) or (prefixes[-1] if prefixes else "")
        if not marker:
            break
    return symbols


def month_range(start: date, end: date) -> list[tuple[int, int]]:
    """Inclusive (year, month) pairs from start's month through end's month."""
    months = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        months.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return months


def day_range(start: date, end: date) -> list[date]:
    """Inclusive list of dates from `start` through `end`."""
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def utc_today() -> date:
    """Today's date in UTC."""
    return datetime.now(UTC).date()
