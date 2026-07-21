"""Partitioned parquet storage for market data.

Layout (hive-style, readable by DuckDB/pyarrow natively):

    data/raw/<dataset>/exchange=<exchange>/asset=<asset>/year=<year>/data.parquet

Conventions:
- `ts` column: UTC timestamp, bar OPEN time for bars, event time otherwise.
- Raw datasets store venue-native values; normalization happens in processed.
- Writes are idempotent: rows are deduplicated on `ts` within a partition,
  with newly written rows winning over existing ones.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_hub.utils.config import PROJECT_ROOT

DATA_ROOT = PROJECT_ROOT / "data"
RAW_ROOT = DATA_ROOT / "raw"
PROCESSED_ROOT = DATA_ROOT / "processed"


def partition_path(
    dataset: str, exchange: str, asset: str, year: int, root: Path = RAW_ROOT
) -> Path:
    """Return the parquet file path for one (dataset, exchange, asset, year) partition."""
    return (
        root / dataset / f"exchange={exchange}" / f"asset={asset}" / f"year={year}" / "data.parquet"
    )


def dataset_glob(dataset: str, root: Path = RAW_ROOT) -> str:
    """Glob matching every partition file of a dataset (for DuckDB/pyarrow)."""
    return str(root / dataset / "**" / "data.parquet")


def write_partition(
    df: pd.DataFrame,
    dataset: str,
    exchange: str,
    asset: str,
    root: Path = RAW_ROOT,
) -> int:
    """Merge `df` into the dataset's partitions, deduplicating on `ts`.

    `df` needs a UTC-aware `ts` column; partition columns (exchange/asset/year)
    must NOT be in the frame — they live in the path. Returns rows written.
    """
    if df.empty:
        return 0
    if "ts" not in df.columns:
        raise ValueError("expected a 'ts' column")
    ts = pd.to_datetime(df["ts"], utc=True)
    df = df.assign(ts=ts).sort_values("ts")

    written = 0
    for year, chunk in df.groupby(ts.dt.year):
        path = partition_path(dataset, exchange, asset, int(year), root)
        if path.exists():
            existing = pd.read_parquet(path)
            existing["ts"] = pd.to_datetime(existing["ts"], utc=True)
            chunk = pd.concat([existing, chunk], ignore_index=True)
            chunk = chunk.drop_duplicates(subset="ts", keep="last").sort_values("ts")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".parquet.tmp")
        chunk.to_parquet(tmp, index=False)
        tmp.replace(path)  # atomic swap: readers never see a half-written file
        written += len(chunk)
    return written


def list_partitions(dataset: str, root: Path = RAW_ROOT) -> pd.DataFrame:
    """Inventory of existing partitions: exchange, asset, year, path."""
    rows = []
    for path in sorted((root / dataset).glob("exchange=*/asset=*/year=*/data.parquet")):
        exchange, asset, year = (p.split("=", 1)[1] for p in path.parts[-4:-1])
        rows.append({"exchange": exchange, "asset": asset, "year": int(year), "path": str(path)})
    return pd.DataFrame(rows, columns=["exchange", "asset", "year", "path"])


def coverage(dataset: str, root: Path = RAW_ROOT) -> pd.DataFrame:
    """Per exchange/asset: first ts, last ts, and row count across partitions."""
    parts = list_partitions(dataset, root)
    if parts.empty:
        return pd.DataFrame(columns=["exchange", "asset", "first_ts", "last_ts", "rows"])
    rows = []
    for (exchange, asset), grp in parts.groupby(["exchange", "asset"]):
        first_ts, last_ts, n = None, None, 0
        for path in grp["path"]:
            ts = pd.read_parquet(path, columns=["ts"])["ts"]
            ts = pd.to_datetime(ts, utc=True)
            n += len(ts)
            first_ts = ts.min() if first_ts is None else min(first_ts, ts.min())
            last_ts = ts.max() if last_ts is None else max(last_ts, ts.max())
        rows.append(
            {
                "exchange": exchange,
                "asset": asset,
                "first_ts": first_ts,
                "last_ts": last_ts,
                "rows": n,
            }
        )
    return pd.DataFrame(rows)
