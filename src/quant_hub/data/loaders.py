"""Research-facing data access. Notebooks import from here and get pandas back.

DuckDB is an implementation detail: it queries the hive-partitioned parquet
files directly (predicate pushdown on exchange/asset/year), no database
server or import step involved.

    from quant_hub.data.loaders import load_ohlcv, load_funding

    df = load_ohlcv(["BTC", "ETH"], start="2024-01-01", end="2024-06-30")
    wide = load_ohlcv(["BTC", "ETH"], field="close")   # ts x asset matrix
"""

from __future__ import annotations

import duckdb
import pandas as pd

from quant_hub.data.instruments import load_instruments  # re-export for convenience
from quant_hub.data.storage import RAW_ROOT, dataset_glob

__all__ = ["load_ohlcv", "load_funding", "load_oi", "load_instruments"]


def _load(
    dataset: str,
    assets: list[str] | None,
    start: str | None,
    end: str | None,
    exchange: str | None,
) -> pd.DataFrame:
    glob = dataset_glob(dataset, RAW_ROOT)
    conditions, params = [], []
    if assets:
        conditions.append(f"asset IN ({','.join('?' * len(assets))})")
        params.extend(assets)
    if exchange:
        conditions.append("exchange = ?")
        params.append(exchange)
    if start:
        conditions.append("ts >= ?")
        params.append(pd.Timestamp(start, tz="UTC"))
    if end:
        # inclusive through end-of-day when a bare date is given
        end_ts = pd.Timestamp(end, tz="UTC")
        if end_ts == end_ts.normalize():
            end_ts += pd.Timedelta(days=1)
        conditions.append("ts < ?")
        params.append(end_ts)
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    query = f"""
        SELECT * EXCLUDE (year)
        FROM read_parquet('{glob}', hive_partitioning = true, union_by_name = true)
        {where}
        ORDER BY exchange, asset, ts
        """
    try:
        df = duckdb.sql(query, params=params).df()
        # DuckDB localizes timestamptz to the session timezone; keep UTC everywhere
        return df.assign(ts=df["ts"].dt.tz_convert("UTC"))
    except duckdb.IOException as err:
        raise FileNotFoundError(
            f"no data stored for dataset '{dataset}' — run the ingestion CLI first "
            f"(python -m quant_hub.data.ingestion)"
        ) from err


def load_ohlcv(
    assets: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    exchange: str = "binance",
    field: str | None = None,
    interval: str = "15m",
) -> pd.DataFrame:
    """Bars (15m, or 1d for hyperliquid). `field="close"` pivots to a ts x asset wide matrix."""
    df = _load("ohlcv_1d" if interval == "1d" else "ohlcv_15m", assets, start, end, exchange)
    if field is None:
        return df
    return df.pivot_table(index="ts", columns="asset", values=field)


def load_funding(
    assets: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    exchange: str = "hyperliquid",
) -> pd.DataFrame:
    """Venue-native funding events (hyperliquid hourly, binance 8h/4h)."""
    return _load("funding", assets, start, end, exchange)


def load_oi(
    assets: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    exchange: str = "binance",
) -> pd.DataFrame:
    """5-min open interest snapshots (binance metrics dumps)."""
    return _load("open_interest", assets, start, end, exchange)
