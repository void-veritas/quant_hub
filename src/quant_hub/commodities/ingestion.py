r"""Backfill CLI of the commodities domain. Manual, idempotent, resume-safe.

    uv run python -m quant_hub.commodities.ingestion hip3      [--dexes xyz,flx] [--all-markets]
    uv run python -m quant_hub.commodities.ingestion databento --symbols HG.FUT --schema ohlcv-1d \\
        --start 2010-06-06 [--end YYYY-MM-DD] --max-usd 1.74 [--estimate-only]
    uv run python -m quant_hub.commodities.ingestion cot       [--start-year 2006] [--codes 085692]
    uv run python -m quant_hub.commodities.ingestion futures   --assets HG,GC [--source yahoo|stooq]

`databento` never buys without a free cost estimate first and refuses above
`--max-usd`; the rule of the house is: cost table to Egor, download after his yes.
Datasets and layout: see the package docstring.
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from quant_hub.commodities import hip3, storage
from quant_hub.commodities.connectors import cftc, futures_daily
from quant_hub.commodities.connectors import databento as dbn


def cmd_hip3(args) -> None:
    """Daily bars and funding of HIP-3 builder perps; refreshes `reference/hip3_markets.csv`."""
    table = hip3.ingest(
        dexes=[d.strip() for d in args.dexes.split(",")], all_markets=args.all_markets
    )
    n_del = int(table["delisted"].sum())
    print(f"hip3: {len(table)} markets listed, {n_del} delisted; reference/hip3_markets.csv")


def cmd_databento(args) -> None:
    """Estimate, then (unless --estimate-only) buy one request and store it."""
    cli = dbn.client()
    end = args.end or dbn.dataset_end(args.dataset, cli)
    req = dbn.request(
        [s.strip() for s in args.symbols.split(",")],
        args.schema,
        args.start,
        end,
        dataset=args.dataset,
        stype_in=args.stype_in,
    )
    est = dbn.estimate(req, cli)
    print(f"{dbn.file_stem(req)}: {est.usd:.2f} USD, {est.mb:.1f} MB (limit {args.max_usd:.2f})")
    if args.estimate_only:
        return
    store, path = dbn.fetch(req, max_usd=args.max_usd, cli=cli)
    print(f"stored {path} ({path.stat().st_size / 1e6:.1f} MB on disk)")
    if args.schema.startswith("ohlcv"):
        bars = dbn.bars_to_frame(store)
        n = dbn.write_bars(bars, dataset="futures_1d" if args.schema == "ohlcv-1d" else args.schema)
        spreads = bars["symbol"].map(dbn.is_spread)
        print(
            f"futures_1d: {n} rows written, {bars['symbol'].nunique()} symbols "
            f"({int((~spreads).sum())} outright rows, {int(spreads.sum())} spread rows), "
            f"{bars['ts'].min().date()} -> {bars['ts'].max().date()}"
        )


def cmd_cot(args) -> None:
    """CFTC disaggregated COT, all markets, one partition set per CFTC code."""
    last = args.end_year or pd.Timestamp.now(tz="UTC").year
    df = cftc.fetch(args.start_year, last)
    if args.codes:
        df = df[df["code"].isin([c.strip().zfill(6) for c in args.codes.split(",")])]
    n = 0
    for code, sub in df.groupby("code"):
        n += storage.write_partition(sub.drop(columns=["code"]), "cot_disagg", "cftc", str(code))
    cu = df[df["code"] == cftc.COPPER_CODE]
    span = f"{cu['ts'].min()} -> {cu['ts'].max()}" if len(cu) else "-"
    print(f"cot: {df['code'].nunique()} markets, {n} rows; copper {len(cu)} weeks {span}")


def cmd_futures(args) -> None:
    """Continuous front-month commodity futures from free sources (`futures_cont_1d`)."""
    roots = (
        list(futures_daily.COMMODITIES)
        if args.all
        else [a.strip().upper() for a in args.assets.split(",")]
    )
    for root in roots:
        if root not in futures_daily.COMMODITIES:
            raise SystemExit(f"unknown root {root}; known: {sorted(futures_daily.COMMODITIES)}")
        ysym, ssym, _, desc = futures_daily.COMMODITIES[root]
        if args.source == "yahoo":
            df = futures_daily.fetch_yahoo(ysym, start=args.start or "1990-01-01", end=args.end)
        else:
            if ssym is None:
                print(f"futures stooq {root:4s} no stooq symbol, skipped", flush=True)
                continue
            df = futures_daily.fetch_stooq(ssym)
        if not df.empty:
            storage.write_partition(df, "futures_cont_1d", args.source, root)
        span = f"{df['ts'].min().date()} -> {df['ts'].max().date()}" if len(df) else "-"
        print(f"futures {args.source} {root:4s} {desc:16s} {len(df)} rows {span}", flush=True)
        time.sleep(1.0)


def main(argv=None) -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("hip3", help="HIP-3 builder perps on Hyperliquid: daily bars + funding")
    p.add_argument("--dexes", default=",".join(hip3.DEXES))
    p.add_argument("--all-markets", action="store_true", help="not only commodities")

    p = sub.add_parser("databento", help="buy one Databento request (cost-gated)")
    p.add_argument("--symbols", required=True, help="comma-separated, e.g. HG.FUT")
    p.add_argument("--schema", default="ohlcv-1d")
    p.add_argument("--start", default=dbn.DATASET_START)
    p.add_argument("--end", help="YYYY-MM-DD; default: the dataset's last available day")
    p.add_argument("--dataset", default=dbn.DATASET)
    p.add_argument("--stype-in", default="parent")
    p.add_argument("--max-usd", type=float, required=True, help="refuse above this estimate")
    p.add_argument("--estimate-only", action="store_true", help="print the estimate and exit")

    p = sub.add_parser("cot", help="CFTC disaggregated Commitments of Traders (weekly, 2006+)")
    p.add_argument("--start-year", type=int, default=cftc.FIRST_YEAR)
    p.add_argument("--end-year", type=int)
    p.add_argument("--codes", help="comma-separated CFTC codes to keep (default: all markets)")

    p = sub.add_parser("futures", help="continuous commodity futures (yahoo / stooq)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--assets", help="roots, e.g. HG,GC,SI (see connectors/futures_daily.py)")
    g.add_argument("--all", action="store_true", help="the whole commodity basket")
    p.add_argument("--source", choices=["yahoo", "stooq"], default="yahoo")
    p.add_argument("--start", help="YYYY-MM-DD (yahoo only; default 1990-01-01)")
    p.add_argument("--end", help="YYYY-MM-DD (yahoo only)")

    args = ap.parse_args(argv)
    {"hip3": cmd_hip3, "databento": cmd_databento, "cot": cmd_cot, "futures": cmd_futures}[
        args.command
    ](args)


if __name__ == "__main__":
    main()
