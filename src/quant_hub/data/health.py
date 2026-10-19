r"""One-command health report of the raw data layer.

    uv run python -m quant_hub.data.health [--dataset ohlcv_1d] [--exchange binance] \\
        [--assets BTC,ETH] [--stale-days 3] [--all]

Per (dataset, exchange, asset): first/last timestamp, rows, duplicate or unsorted
timestamps, OHLCV sanity problems, gaps against the dataset's bar schedule, and
staleness of the last bar. `--cross-venue` adds the HL vs Binance daily-close
comparison (an asset whose closes disagree by more than 3% on over 10% of common
days is not the same instrument on the two venues, e.g. FTT). Exit code 1 when any
problem is found, so it can gate a push to the bucket. Pre-listing synthetic HL daily
bars (zero volume) and the known 2% of venue days missing from the HL L2 archive are
reported, not hidden.
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import UTC, datetime

import pandas as pd

from quant_hub.data import storage, validation

# expected spacing of timestamps; None = infer from the median spacing
SCHEDULE = {
    "ohlcv_15m": "15min",
    "ohlcv_1d": "1D",
    "hl_l2_hour": "1D",
    "open_interest": "5min",
    "funding": None,  # hourly on HL, 8h/4h on Binance: inferred per asset
}
OHLCV = {"ohlcv_15m", "ohlcv_1d"}
# datasets whose source is known to lack some days: flag only above this many missing
KNOWN_MISSING_DAYS = {"hl_l2_hour": 30}  # the HL archive lacks ~25 venue days (2% of history)


def _event_gaps(ts: pd.Series, tolerance: float = 2.5) -> pd.DataFrame:
    """Gaps in event data without a fixed grid (funding: hourly on HL, 8h/4h on Binance).

    Timestamps carry millisecond jitter and venues change cadence, so a gap is a
    spacing wider than `tolerance` times the local (rolling median) spacing; the
    missing count is the number of skipped settlements at that cadence.
    """
    ts = pd.to_datetime(ts, utc=True).sort_values().reset_index(drop=True)
    if len(ts) < 3:
        return pd.DataFrame(columns=["gap_start", "gap_end", "n_missing"])
    d = ts.diff().dt.total_seconds()
    # local cadence: HL funding went from 8h to hourly, Binance from 8h to 4h on some names
    med = d.rolling(51, center=True, min_periods=5).median().bfill().ffill()
    wide = d > tolerance * med
    rows = [
        dict(
            gap_start=ts[i - 1] + pd.Timedelta(seconds=med[i]),
            gap_end=ts[i] - pd.Timedelta(seconds=med[i]),
            n_missing=int(round(d[i] / med[i])) - 1,
        )
        for i in ts.index[wide]
    ]
    return pd.DataFrame(rows, columns=["gap_start", "gap_end", "n_missing"])


def _asset_report(dataset: str, exchange: str, asset: str, paths, now: pd.Timestamp) -> dict:
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    problems: list[str] = []
    dup = int(df["ts"].duplicated().sum())
    if dup:
        problems.append(f"{dup} duplicate ts")
    if not df["ts"].is_monotonic_increasing:
        problems.append("unsorted")
    df = df.sort_values("ts").drop_duplicates("ts")
    if dataset in OHLCV:
        synthetic = int((df.get("volume", pd.Series(1.0, index=df.index)) <= 0).sum())
        bars = df[df["volume"] > 0] if "volume" in df else df
        problems += [
            x for x in validation.check_bars(bars) if "duplicate" not in x and "sorted" not in x
        ]
    else:
        synthetic, bars = 0, df
    freq = SCHEDULE.get(dataset)
    if freq is not None and len(bars) > 1:
        gaps = validation.find_gaps(bars["ts"], freq)
    else:
        gaps = _event_gaps(bars["ts"])
    last = bars["ts"].max() if len(bars) else df["ts"].max()
    return dict(
        dataset=dataset,
        exchange=exchange,
        asset=asset,
        first=bars["ts"].min() if len(bars) else df["ts"].min(),
        last=last,
        rows=len(df),
        synthetic_rows=synthetic,
        gaps=len(gaps),
        missing=int(gaps["n_missing"].sum()) if len(gaps) else 0,
        stale_days=round((now - last).total_seconds() / 86400, 1),
        problems="; ".join(problems),
    )


def _statuses() -> dict:
    try:
        from quant_hub.data.instruments import load_instruments

        inst = load_instruments()
        return dict(
            zip(
                zip(inst["exchange"], inst["canonical_id"], strict=True),
                inst["status"],
                strict=True,
            )
        )
    except Exception:  # noqa: BLE001 - the registry is optional for the report
        return {}


def report(
    datasets: list[str] | None = None,
    exchange: str | None = None,
    assets: list[str] | None = None,
    root=None,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Health table over every stored (dataset, exchange, asset)."""
    root = root or storage.RAW_ROOT
    now = now or pd.Timestamp(datetime.now(UTC))
    datasets = datasets or sorted(d.name for d in root.iterdir() if d.is_dir())
    status = _statuses()
    rows = []
    for ds in datasets:
        parts = storage.list_partitions(ds, root=root)
        if exchange:
            parts = parts[parts["exchange"] == exchange]
        if assets:
            parts = parts[parts["asset"].isin(assets)]
        for (ex, asset), grp in parts.groupby(["exchange", "asset"]):
            r = _asset_report(ds, ex, asset, list(grp["path"]), now)
            r["status"] = status.get((ex, asset), "")
            rows.append(r)
    cols = [
        "dataset",
        "exchange",
        "asset",
        "status",
        "first",
        "last",
        "rows",
        "synthetic_rows",
        "gaps",
        "missing",
        "stale_days",
        "problems",
    ]
    return pd.DataFrame(rows, columns=cols)


def flag(df: pd.DataFrame, stale_days: float = 3.0) -> pd.DataFrame:
    """Rows that need attention: problems, gaps, or a last bar older than `stale_days`."""
    if df.empty:
        return df
    live = df["status"].isin(["", "trading"])  # delisted instruments are stale by nature
    tol = df["dataset"].map(KNOWN_MISSING_DAYS).fillna(0)
    bad = (df["problems"] != "") | (df["missing"] > tol) | ((df["stale_days"] > stale_days) & live)
    return df[bad]


def cross_venue(
    root=None,
    start: str | None = None,
    threshold: float = 0.03,
    max_share: float = 0.10,
) -> pd.DataFrame:
    """Daily closes of assets stored on both venues: share of days off by > `threshold`.

    Returns one row per common asset (common_days, off_days, off_share, median_logdiff,
    flagged) sorted by off_share. A flagged asset (off_share > `max_share`) must not
    be stitched across venues: its canonical id maps to different instruments or one
    venue's series is stale.
    """
    from quant_hub.data.loaders import load_ohlcv

    root = root or storage.RAW_ROOT
    frames = {}
    for venue in ("hyperliquid", "binance"):
        if not (root / "ohlcv_1d" / f"exchange={venue}").exists():
            return pd.DataFrame(
                columns=[
                    "asset",
                    "common_days",
                    "off_days",
                    "off_share",
                    "median_logdiff",
                    "flagged",
                ]
            )
        df = (
            load_ohlcv(None, start=start, exchange=venue, interval="1d")
            if root == storage.RAW_ROOT
            else _read_daily(root, venue, start)
        )
        frames[venue] = df[df["volume"] > 0][["ts", "asset", "close"]]
    m = frames["hyperliquid"].merge(frames["binance"], on=["ts", "asset"], suffixes=("_hl", "_bn"))
    if m.empty:
        return pd.DataFrame(
            columns=["asset", "common_days", "off_days", "off_share", "median_logdiff", "flagged"]
        )
    m["ldiff"] = (m["close_hl"] / m["close_bn"]).map(math.log)
    g = m.groupby("asset")
    out = pd.DataFrame(
        {
            "common_days": g.size(),
            "off_days": g["ldiff"].apply(lambda s: int((s.abs() > threshold).sum())),
            "median_logdiff": g["ldiff"].median().round(5),
        }
    )
    out["off_share"] = (out["off_days"] / out["common_days"]).round(4)
    out["flagged"] = out["off_share"] > max_share
    return out.reset_index().sort_values("off_share", ascending=False).reset_index(drop=True)


def _read_daily(root, venue: str, start: str | None) -> pd.DataFrame:
    parts = storage.list_partitions("ohlcv_1d", root=root)
    parts = parts[parts["exchange"] == venue]
    frames = []
    for _, p in parts.iterrows():
        df = pd.read_parquet(p["path"])
        df["asset"] = p["asset"]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    if start:
        df = df[df["ts"] >= pd.Timestamp(start, tz="UTC")]
    return df


def main(argv=None) -> int:
    """CLI entry point; exit code 1 when any row is flagged."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", action="append", help="repeatable; default all")
    ap.add_argument("--exchange")
    ap.add_argument("--assets", help="comma-separated canonical ids")
    ap.add_argument("--stale-days", type=float, default=3.0)
    ap.add_argument("--all", action="store_true", help="print every row, not only flagged ones")
    ap.add_argument(
        "--cross-venue",
        action="store_true",
        help="also compare HL and Binance daily closes per asset (ohlcv_1d on both venues)",
    )
    args = ap.parse_args(argv)
    df = report(args.dataset, args.exchange, args.assets.split(",") if args.assets else None)
    bad = flag(df, args.stale_days)
    summary = df.groupby(["dataset", "exchange"]).agg(
        assets=("asset", "nunique"),
        last=("last", "max"),
        rows=("rows", "sum"),
        gaps=("missing", "sum"),
    )
    print(summary.to_string())
    print()
    if args.all:
        print(df.to_string(index=False))
    elif bad.empty:
        print("no problems found")
    else:
        print(
            f"{len(bad)} rows need attention (problems, gaps, or stale > {args.stale_days} days):"
        )
        print(bad.to_string(index=False))
    rc = 1 if not bad.empty else 0
    if args.cross_venue:
        cv = cross_venue()
        print("\ncross-venue daily closes (HL vs Binance):")
        print(cv[cv["flagged"] | (cv["off_days"] > 0)].to_string(index=False))
        if cv["flagged"].any():
            bad_assets = cv[cv["flagged"]]["asset"].tolist()
            print(f"flagged (not the same instrument on both venues): {bad_assets}")
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
