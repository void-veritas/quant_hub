"""Hyperliquid historical L2 snapshots from the public S3 archive.

Bucket `hyperliquid-archive` (requester pays; AWS credentials required):

    market_data/<YYYYMMDD>/<hour>/l2Book/<COIN>.lz4

One file per coin per hour: newline-delimited JSON, one `l2Book` websocket
message per line (~0.55 s cadence, 20 levels per side), lz4-frame compressed,
~1 MB each for majors. Coverage starts 2023-04-15 and is updated daily.
Trades/fills live in `hl-mainnet-node-data` and are handled separately.

Nothing here is cached raw: an hour file is downloaded, reduced to the rows the
caller asked for, and discarded. Egress is billed to the requester (~$0.09/GB),
so callers should budget: `estimate_bytes()` before a large pull.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime

import boto3
import lz4.frame
import pandas as pd
from botocore.exceptions import BotoCoreError, ClientError

BUCKET = "hyperliquid-archive"
ARCHIVE_START = date(2023, 4, 15)
_RP = {"RequestPayer": "requester"}

_session = boto3.session.Session()


def _client():
    return _session.client("s3", region_name="us-east-1")


def key_for(day: date, hour: int, coin: str) -> str:
    """S3 key of the hour file."""
    return f"market_data/{day:%Y%m%d}/{hour}/l2Book/{coin}.lz4"


def fetch_hour_raw(day: date, hour: int, coin: str, retries: int = 4) -> bytes | None:
    """Download and decompress one hour file; None if the key does not exist.

    Transient transport errors (truncated streams, throttling) are retried with
    exponential backoff; a missing key is not an error.
    """
    for attempt in range(retries):
        try:
            obj = _client().get_object(Bucket=BUCKET, Key=key_for(day, hour, coin), **_RP)
            return lz4.frame.decompress(obj["Body"].read())
        except ClientError as err:
            if err.response["Error"]["Code"] in ("NoSuchKey", "404"):
                return None
            if attempt == retries - 1:
                raise
        except (BotoCoreError, OSError):
            if attempt == retries - 1:
                raise
        time.sleep(2**attempt)
    return None


def parse_snapshots(raw: bytes, coin: str, depth_bps=(10, 25, 50)) -> pd.DataFrame:
    """Reduce an hour of L2 messages to one row per snapshot.

    Columns: ts (exchange time, ms), bid, ask, mid, spread_bps, bid_sz0, ask_sz0,
    n_bid0, n_ask0, and for each d in depth_bps the base-asset size resting within
    d bps of mid on each side (bid_d{d}, ask_d{d}). Sizes are in coin units.
    """
    rows = []
    for line in raw.decode().splitlines():
        if not line:
            continue
        msg = json.loads(line)
        data = msg["raw"]["data"]
        bids, asks = data["levels"]
        if not bids or not asks:
            continue
        bid, ask = float(bids[0]["px"]), float(asks[0]["px"])
        mid = 0.5 * (bid + ask)
        row = {
            "ts": data["time"],
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "spread_bps": 1e4 * (ask - bid) / mid,
            "bid_sz0": float(bids[0]["sz"]),
            "ask_sz0": float(asks[0]["sz"]),
            "n_bid0": int(bids[0]["n"]),
            "n_ask0": int(asks[0]["n"]),
        }
        for d in depth_bps:
            lim = mid * d * 1e-4
            row[f"bid_d{d}"] = sum(float(x["sz"]) for x in bids if mid - float(x["px"]) <= lim)
            row[f"ask_d{d}"] = sum(float(x["sz"]) for x in asks if float(x["px"]) - mid <= lim)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df["symbol"] = coin
    return df.sort_values("ts").reset_index(drop=True)


def fetch_hour(day: date, hour: int, coin: str) -> pd.DataFrame | None:
    """One hour of reduced snapshots, or None if the archive has no file."""
    raw = fetch_hour_raw(day, hour, coin)
    return None if raw is None else parse_snapshots(raw, coin)


def hour_summary(snaps: pd.DataFrame, archive_hour: pd.Timestamp | None = None) -> dict:
    """Per-hour aggregates used by the daily decision-time dataset.

    `archive_hour` is the file's nominal hour (day + hour, UTC); the first
    message in an hour file can be stamped a few hundred ms before the hour, so
    the row's `ts` must come from the file name, not from the message.
    """
    first = snaps.iloc[0]
    ts = pd.Timestamp(archive_hour) if archive_hour is not None else snaps["ts"].iloc[-1].floor("h")
    out = {
        "ts": ts.tz_localize("UTC") if ts.tzinfo is None else ts,
        "n_snapshots": len(snaps),
        "first_ts": first["ts"],
        "open_mid": first["mid"],
        "open_bid": first["bid"],
        "open_ask": first["ask"],
        "open_spread_bps": first["spread_bps"],
        "close_mid": snaps["mid"].iloc[-1],
        "med_spread_bps": snaps["spread_bps"].median(),
        "mean_spread_bps": snaps["spread_bps"].mean(),
        "symbol": first["symbol"],
    }
    for col in snaps.columns:
        if col.startswith(("bid_d", "ask_d")):
            out[f"med_{col}"] = snaps[col].median()
            out[f"open_{col}"] = first[col]
    return out


def list_days(start: date | None = None, end: date | None = None) -> list[date]:
    """Days present in the archive (one LIST per 1000 prefixes)."""
    s3 = _client()
    days: list[date] = []
    token = None
    while True:
        kw = dict(Bucket=BUCKET, Prefix="market_data/", Delimiter="/", MaxKeys=1000, **_RP)
        if token:
            kw["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kw)
        for p in resp.get("CommonPrefixes", []):
            d = datetime.strptime(p["Prefix"].split("/")[1], "%Y%m%d").date()
            if (start is None or d >= start) and (end is None or d <= end):
                days.append(d)
        if not resp.get("IsTruncated"):
            return days
        token = resp["NextContinuationToken"]


def estimate_bytes(days: list[date], coins: list[str], hour: int, sample: int = 3) -> int:
    """Rough egress estimate: HEAD a few files and extrapolate."""
    s3 = _client()
    sizes = []
    for d in days[:: max(1, len(days) // sample)][:sample]:
        for c in coins:
            try:
                sizes.append(
                    s3.head_object(Bucket=BUCKET, Key=key_for(d, hour, c), **_RP)["ContentLength"]
                )
            except ClientError:
                pass
    if not sizes:
        return 0
    return int(sum(sizes) / len(sizes) * len(days) * len(coins))


def fetch_many(jobs, fn, workers: int = 8):
    """Run fn(job) over jobs with a thread pool, yielding (job, result)."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        yield from zip(jobs, pool.map(fn, jobs), strict=True)


def utc_today() -> date:
    """Today's date in UTC."""
    return datetime.now(UTC).date()
