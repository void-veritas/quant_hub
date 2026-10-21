"""Backfill CLI. Manual, idempotent, resume-safe.

    uv run python -m quant_hub.data.ingestion instruments
    uv run python -m quant_hub.data.ingestion ohlcv   --exchange binance --assets BTC,ETH
    uv run python -m quant_hub.data.ingestion ohlcv   --exchange binance --all
    uv run python -m quant_hub.data.ingestion funding --exchange hyperliquid --assets BTC
    uv run python -m quant_hub.data.ingestion oi      --assets BTC --start 2024-01-01
    uv run python -m quant_hub.data.ingestion gaps    --dataset ohlcv_15m

Re-runs skip complete months / already-stored rows, so interrupting and
restarting a backfill is always safe.

Datasets written (all raw, venue-native):
    ohlcv_15m      15m bars        (binance: bulk dumps; hyperliquid: API, recent only)
    ohlcv_1d       daily bars      (hyperliquid API, full history; `--interval 1d`)
    funding        funding events  (binance monthly dumps + optional REST tail, hyperliquid API)
    open_interest  5-min OI snapshots (binance metrics dumps, exist since ~2021-12)
    hl_l2_hour     per-hour L2 summaries from the HL S3 archive (requester pays):
                   first snapshot of the hour (decision-time mid/spread/depth) and
                   hour medians; `l2hour --hour 9` builds the YOLO 09:00 panel
    cot_disagg     CFTC disaggregated Commitments of Traders, weekly from 2006, every
                   market (exchange=cftc, asset=CFTC code; copper 085692): `cot`
    futures_1d     daily continuous front-month commodity futures (exchange=yahoo or
                   stooq, asset=root, e.g. HG copper): `futures --assets HG,GC`
    ohlcv_1d,      HIP-3 builder-deployed perps on Hyperliquid (exchange=hyperliquid_hip3,
    funding        asset=<dex>-<NAME>, e.g. xyz-COPPER): `hip3 --dexes xyz,flx`

    uv run python -m quant_hub.data.ingestion cot     [--start-year 2006]
    uv run python -m quant_hub.data.ingestion futures --source yahoo --assets HG,GC,SI
    uv run python -m quant_hub.data.ingestion hip3    --dexes xyz,flx [--all-markets]
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from datetime import time as dtime

import pandas as pd

from quant_hub.data import storage, validation
from quant_hub.data.connectors import binance_connector as bnc
from quant_hub.data.connectors import cftc, futures_daily, hl_archive, scrapingbee
from quant_hub.data.connectors import hyperliquid_connector as hl
from quant_hub.data.instruments import build_instruments, load_instruments, save_instruments

DEFAULT_START = {"binance": date(2019, 9, 8), "hyperliquid": date(2023, 3, 1)}


def _utc(day: date) -> datetime:
    return datetime.combine(day, dtime.min, tzinfo=UTC)


def _existing_ts(dataset: str, exchange: str, asset: str) -> pd.DatetimeIndex:
    parts = storage.list_partitions(dataset, root=storage.RAW_ROOT)
    if parts.empty:
        return pd.DatetimeIndex([], tz="UTC")
    parts = parts[(parts["exchange"] == exchange) & (parts["asset"] == asset)]
    if parts.empty:
        return pd.DatetimeIndex([], tz="UTC")
    ts = pd.concat([pd.read_parquet(p, columns=["ts"])["ts"] for p in parts["path"]])
    return pd.DatetimeIndex(pd.to_datetime(ts, utc=True)).sort_values()


def _resolve_assets(args, exchange: str) -> pd.DataFrame:
    inst = load_instruments(exchange=exchange)
    if args.all:
        if exchange == "binance":
            # HL is the trading venue: backfill cross-listed assets first so the
            # tradeable universe becomes usable before the Binance-only tail
            on_hl = inst["canonical_id"].isin(
                load_instruments(exchange="hyperliquid")["canonical_id"]
            )
            inst = pd.concat([inst[on_hl], inst[~on_hl]])
        return inst
    wanted = [a.strip().upper() for a in args.assets.split(",")]
    missing = set(wanted) - set(inst["canonical_id"])
    if missing:
        raise SystemExit(f"not in instruments table for {exchange}: {sorted(missing)}")
    return inst[inst["canonical_id"].isin(wanted)]


def cmd_instruments(_args) -> None:
    """Rebuild config/instruments.csv from live venue metadata."""
    df = build_instruments()
    save_instruments(df)
    counts = df.groupby(["exchange", "status"]).size()
    print(f"instruments table written ({len(df)} rows)\n{counts.to_string()}")


def _ohlcv_binance_1d(inst: pd.DataFrame, start: date, end: date) -> None:
    """Daily bars (dataset ohlcv_1d, binance): monthly dumps, daily files for the open month.

    Daily klines carry taker_buy_volume / taker_buy_quote_volume, the aggressor
    imbalance input (Robot Wealth's volume feature) at a fraction of the 15m cost.
    """
    today = bnc.utc_today()
    daily_cutoff = min(end, today - timedelta(days=1))
    for _, row in inst.iterrows():
        asset, symbol = row["canonical_id"], row["symbol"]
        have = _existing_ts("ohlcv_1d", "binance", asset)
        n = 0
        for year, month in bnc.month_range(start, end):
            month_start = pd.Timestamp(year, month, 1, tz="UTC")
            next_month = month_start + pd.offsets.MonthBegin(1)
            if (year, month) == (today.year, today.month):
                for day in bnc.day_range(max(start, date(year, month, 1)), daily_cutoff):
                    if have[(have >= _utc(day)) & (have < _utc(day + timedelta(days=1)))].size:
                        continue
                    df = bnc.fetch_klines_day(symbol, "1d", day)
                    if df is not None:
                        storage.write_partition(df, "ohlcv_1d", "binance", asset)
                        n += len(df)
            else:
                month_have = have[(have >= month_start) & (have < next_month)]
                if month_have.size and month_have.max() >= next_month - pd.Timedelta(days=1):
                    continue
                df = bnc.fetch_klines_month(symbol, "1d", year, month)
                if df is not None:
                    storage.write_partition(df, "ohlcv_1d", "binance", asset)
                    n += len(df)
        print(f"ohlcv(1d) binance {asset:12s} +{n} rows", flush=True)


def _ohlcv_binance(inst: pd.DataFrame, start: date, end: date) -> None:
    """Backfill 15m bars from bulk dumps: monthly zips, daily for the current month."""
    today = bnc.utc_today()
    daily_cutoff = min(end, today - timedelta(days=1))
    for _, row in inst.iterrows():
        asset, symbol = row["canonical_id"], row["symbol"]
        have = _existing_ts("ohlcv_15m", "binance", asset)
        n = 0
        for year, month in bnc.month_range(start, end):
            month_start = pd.Timestamp(year, month, 1, tz="UTC")
            next_month = month_start + pd.offsets.MonthBegin(1)
            if (year, month) == (today.year, today.month):
                for day in bnc.day_range(max(start, date(year, month, 1)), daily_cutoff):
                    if (
                        have[(have >= _utc(day)) & (have < _utc(day + timedelta(days=1)))].size
                        >= 96
                    ):
                        continue
                    df = bnc.fetch_klines_day(symbol, "15m", day)
                    if df is not None:
                        storage.write_partition(df, "ohlcv_15m", "binance", asset)
                        n += len(df)
            else:
                # month is complete if its final 15m bar is already stored
                month_have = have[(have >= month_start) & (have < next_month)]
                if month_have.size and month_have.max() >= next_month - pd.Timedelta("15min"):
                    continue
                df = bnc.fetch_klines_month(symbol, "15m", year, month)
                if df is not None:
                    storage.write_partition(df, "ohlcv_15m", "binance", asset)
                    n += len(df)
        # Monthly dumps are occasionally truncated (e.g. SOL/TRX/XRP/ZEC lack
        # 2022-02-26..28 and 2022-04-01..02); daily dumps exist for those days.
        n += _fill_binance_day_gaps(asset, symbol, start, daily_cutoff)
        print(f"ohlcv binance {asset:12s} +{n} rows", flush=True)


def _fill_binance_day_gaps(asset: str, symbol: str, start: date, end: date) -> int:
    """Fetch daily dumps for days inside the stored range that hold fewer than 96 bars."""
    have = _existing_ts("ohlcv_15m", "binance", asset)
    if not have.size:
        return 0
    first_day = max(start, have.min().date())
    counts = pd.Series(1, index=have).groupby(have.floor("D")).size()
    n = 0
    for day in bnc.day_range(first_day, min(end, have.max().date())):
        if counts.get(pd.Timestamp(day, tz="UTC"), 0) >= 96:
            continue
        df = bnc.fetch_klines_day(symbol, "15m", day)
        if df is not None:
            storage.write_partition(df, "ohlcv_15m", "binance", asset)
            n += len(df)
    return n


def _ohlcv_hyperliquid(
    inst: pd.DataFrame,
    start: date,
    end: date,
    interval: str = "15m",
    refresh: int = 2,
    full: bool = False,
    extend_back: bool = False,
) -> None:
    """Backfill bars from the HL API.

    Intraday intervals serve only the most recent ~5000 bars; "1d" serves the full
    history (with synthetic zero-volume bars before real trading began), which is
    what point-in-time universe selection by dollar volume needs.

    The API also returns the bar that is still forming. It is never stored: a bar
    is kept only when its whole interval lies in the past. The last `refresh`
    stored bars are re-fetched and overwritten on every run, so a bar that was
    revised at the venue is picked up on the next pull. A `start` before the first
    stored bar re-fetches from `start` when `extend_back` is set (an explicit
    `--start`; the default start would refetch everything on every run); `full`
    ignores the stored state and re-fetches [start, end] (the idempotent write
    dedups), which is the way to repair holes inside the stored range.
    """
    dataset = "ohlcv_1d" if interval == "1d" else "ohlcv_15m"
    step = pd.Timedelta(days=1) if interval == "1d" else pd.Timedelta("15min")
    closed_before = pd.Timestamp.now(tz="UTC").floor(step)  # bars stamped here or later are open
    for _, row in inst.iterrows():
        asset, symbol = row["canonical_id"], row["symbol"]
        have = _existing_ts(dataset, "hyperliquid", asset)
        fetch_from = _utc(start)
        if have.size and not full and (fetch_from >= have.min() or not extend_back):
            fetch_from = max(fetch_from, (have.max() - refresh * step).to_pydatetime())
        df = hl.fetch_candles(symbol, interval, fetch_from, _utc(end + timedelta(days=1)))
        df = df[df["ts"] < closed_before]
        if not df.empty:
            storage.write_partition(df, dataset, "hyperliquid", asset, root=storage.RAW_ROOT)
        print(f"ohlcv({interval}) hyperliquid {asset:12s} +{len(df)} rows", flush=True)
        time.sleep(1.1)  # candleSnapshot weighs 20 of the 1200/min budget


def cmd_ohlcv(args) -> None:
    """Backfill 15m bars for the selected exchange and assets."""
    inst = _resolve_assets(args, args.exchange)
    start = date.fromisoformat(args.start) if args.start else DEFAULT_START[args.exchange]
    end = date.fromisoformat(args.end) if args.end else bnc.utc_today()
    if args.exchange == "binance":
        if args.interval == "1d":
            _ohlcv_binance_1d(inst, start, end)
        else:
            _ohlcv_binance(inst, start, end)
    else:
        if args.interval == "1d":
            start = date.fromisoformat(args.start) if args.start else date(2020, 1, 1)
        _ohlcv_hyperliquid(
            inst,
            start,
            end,
            interval=args.interval,
            refresh=args.refresh,
            full=args.full,
            extend_back=bool(args.start),
        )


def cmd_funding(args) -> None:
    """Backfill funding events, extending forward and filling backward gaps.

    Normally fetches only from the last stored timestamp forward. If `--start`
    predates the earliest stored bar (a backward gap), re-fetches from `start`
    instead and lets the idempotent write dedup the overlap.

    Binance: monthly bulk dumps are the primary source (no geo-block, no rate
    limits); the current month has no dump, so its tail is fetched via REST only
    when `--source rest` or `--source both` is given (fapi may be geo-blocked).
    """
    inst = _resolve_assets(args, args.exchange)
    start = date.fromisoformat(args.start) if args.start else DEFAULT_START[args.exchange]
    end = date.fromisoformat(args.end) if args.end else bnc.utc_today()
    source = getattr(args, "source", "dumps")
    for _, row in inst.iterrows():
        asset, symbol = row["canonical_id"], row["symbol"]
        have = _existing_ts("funding", args.exchange, asset)
        if have.size and _utc(start) >= have.min():
            fetch_from = have.max() + pd.Timedelta("1ms")  # forward increment
        else:
            fetch_from = _utc(start)  # empty, or a backward gap to fill
        fetch_to = _utc(end + timedelta(days=1))
        if args.exchange == "hyperliquid":
            df = hl.fetch_funding(symbol, fetch_from, fetch_to)
        elif source == "rest":
            df = bnc.fetch_funding(symbol, fetch_from, fetch_to)
        else:
            df = bnc.fetch_funding_dumps(symbol, fetch_from, fetch_to)
            if source == "both":
                tail_from = df["ts"].max() + pd.Timedelta("1ms") if not df.empty else fetch_from
                tail = bnc.fetch_funding(symbol, tail_from, fetch_to)
                df = pd.concat([df, tail], ignore_index=True)
        if not df.empty:
            storage.write_partition(df, "funding", args.exchange, asset)
        print(f"funding {args.exchange} {asset:12s} +{len(df)} rows", flush=True)


def cmd_oi(args) -> None:
    """Backfill Binance 5-min open-interest snapshots, skipping stored days."""
    inst = _resolve_assets(args, "binance")
    start = date.fromisoformat(args.start) if args.start else date(2021, 12, 1)
    end = date.fromisoformat(args.end) if args.end else bnc.utc_today() - timedelta(days=1)
    for _, row in inst.iterrows():
        asset, symbol = row["canonical_id"], row["symbol"]
        first = start
        if row["listed_at"]:  # no point requesting days before the perp existed
            first = max(first, date.fromisoformat(row["listed_at"]))
        have_days = {t.date() for t in _existing_ts("open_interest", "binance", asset)}
        days = [d for d in bnc.day_range(first, end) if d not in have_days]
        n = 0
        if args.concurrency > 1 and days:
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                frames = pool.map(lambda d, sym=symbol: bnc.fetch_metrics_day(sym, d), days)
                frames = [df for df in frames if df is not None]
            if frames:
                combined = pd.concat(frames, ignore_index=True)
                storage.write_partition(combined, "open_interest", "binance", asset)
                n = len(combined)
        else:
            for day in days:
                df = bnc.fetch_metrics_day(symbol, day)
                if df is not None:
                    storage.write_partition(df, "open_interest", "binance", asset)
                    n += len(df)
        print(f"oi binance {asset:12s} +{n} rows", flush=True)


def cmd_l2hour(args) -> None:
    """Summarise one archive hour per day per asset from HL L2 snapshots.

    Writes dataset `hl_l2_hour` (exchange=hyperliquid). Only days not yet stored
    for that hour are fetched, so re-runs are cheap. Egress is billed to the
    requester; the estimate is printed before any download.
    """
    inst = _resolve_assets(args, "hyperliquid")
    start = date.fromisoformat(args.start) if args.start else hl_archive.ARCHIVE_START
    end = date.fromisoformat(args.end) if args.end else hl_archive.utc_today()
    days = hl_archive.list_days(start, end)
    coins = list(inst["symbol"])
    est = hl_archive.estimate_bytes(days, coins, args.hour)
    print(
        f"l2hour: {len(days)} days x {len(coins)} coins, hour {args.hour}; "
        f"~{est / 1e9:.1f} GB egress (~${est / 1e9 * 0.09:.2f})",
        flush=True,
    )
    if args.dry_run:
        return
    for _, row in inst.iterrows():
        asset, symbol = row["canonical_id"], row["symbol"]
        have = _existing_ts("hl_l2_hour", "hyperliquid", asset)
        have_days = {t.date() for t in have if t.hour == args.hour}
        todo = [d for d in days if d not in have_days]
        rows = []

        def _one(day, sym=symbol):
            snaps = hl_archive.fetch_hour(day, args.hour, sym)
            if snaps is None or snaps.empty:
                return None
            nominal = pd.Timestamp(day) + pd.Timedelta(hours=args.hour)
            return hl_archive.hour_summary(snaps, archive_hour=nominal)

        for _day, summary in hl_archive.fetch_many(todo, _one, workers=args.concurrency):
            if summary is not None:
                rows.append(summary)
            if len(rows) >= 200:
                storage.write_partition(pd.DataFrame(rows), "hl_l2_hour", "hyperliquid", asset)
                rows = []
        if rows:
            storage.write_partition(pd.DataFrame(rows), "hl_l2_hour", "hyperliquid", asset)
        print(f"l2hour hyperliquid {asset:12s} {len(todo)} days fetched", flush=True)


def cmd_cot(args) -> None:
    """CFTC disaggregated COT, all markets, one partition set per CFTC code."""
    last = args.end_year or pd.Timestamp.now(tz="UTC").year
    df = cftc.fetch(args.start_year, last)
    if args.codes:
        df = df[df["code"].isin([c.strip().zfill(6) for c in args.codes.split(",")])]
    n = 0
    for code, sub in df.groupby("code"):
        storage.write_partition(sub.drop(columns=["code"]), "cot_disagg", "cftc", code)
        n += len(sub)
    cu = df[df["code"] == cftc.COPPER_CODE]
    print(
        f"cot: {df['code'].nunique()} markets, {n} rows; copper {len(cu)} weeks "
        f"{cu['ts'].min()} -> {cu['ts'].max()}",
        flush=True,
    )


def cmd_futures(args) -> None:
    """Daily continuous front-month commodity futures (free sources; see connector caveats)."""
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
            storage.write_partition(df, "futures_1d", args.source, root)
        span = f"{df['ts'].min().date()} -> {df['ts'].max().date()}" if len(df) else "-"
        print(f"futures {args.source} {root:4s} {desc:16s} {len(df)} rows {span}", flush=True)
        time.sleep(1.0)


HIP3_COMMODITY_WORDS = (
    "COPPER",
    "GOLD",
    "SILVER",
    "PLATINUM",
    "PALLADIUM",
    "ALUMIN",
    "NICKEL",
    "ZINC",
    "OIL",
    "WTI",
    "BRENT",
    "NATGAS",
    "GAS",
    "URANIUM",
    "CORN",
    "WHEAT",
    "SOY",
    "COFFEE",
)


def hip3_asset(coin: str) -> str:
    """'xyz:COPPER' -> 'xyz-COPPER' (a colon is a poor path component on macOS)."""
    return coin.replace(":", "-")


def cmd_hip3(args) -> None:
    """Daily bars and funding of HIP-3 builder-deployed perps (commodities by default)."""
    end = _utc(bnc.utc_today() + timedelta(days=1))
    closed_before = pd.Timestamp.now(tz="UTC").floor("D")
    for dex in [d.strip() for d in args.dexes.split(",")]:
        markets = hl.fetch_meta(dex=dex)
        names = [m["name"] for m in markets]
        if not args.all_markets:
            names = [
                n for n in names if any(w in n.split(":")[-1].upper() for w in HIP3_COMMODITY_WORDS)
            ]
        for coin in names:
            asset = hip3_asset(coin)
            bars = hl.fetch_candles(coin, "1d", _utc(date(2023, 1, 1)), end)
            bars = bars[bars["ts"] < closed_before]
            if not bars.empty:
                storage.write_partition(bars, "ohlcv_1d", "hyperliquid_hip3", asset)
            have = _existing_ts("funding", "hyperliquid_hip3", asset)
            f_from = have.max() + pd.Timedelta("1ms") if have.size else _utc(date(2023, 1, 1))
            fund = hl.fetch_funding(coin, f_from, end)
            if not fund.empty:
                storage.write_partition(fund, "funding", "hyperliquid_hip3", asset)
            print(f"hip3 {coin:18s} bars {len(bars):4d} funding +{len(fund)}", flush=True)
            time.sleep(1.1)


FREQ = {"ohlcv_15m": "15min", "ohlcv_1d": "1D", "open_interest": "5min"}


def cmd_gaps(args) -> None:
    """Report contiguous missing-bar ranges per exchange/asset for a dataset."""
    inst = load_instruments()
    keys = zip(inst["exchange"], inst["canonical_id"], strict=True)
    intervals = dict(zip(keys, inst["funding_interval_hours"], strict=True))
    parts = storage.list_partitions(args.dataset)
    if args.exchange:
        parts = parts[parts["exchange"] == args.exchange]
    if parts.empty:
        print(f"no data stored for dataset {args.dataset}")
        return
    clean = True
    for (exchange, asset), grp in parts.groupby(["exchange", "asset"]):
        ts = pd.concat([pd.read_parquet(p, columns=["ts"])["ts"] for p in grp["path"]])
        if args.dataset == "funding":
            hours = intervals.get((exchange, asset), 8)
            freq = f"{int(hours)}h"
        else:
            freq = FREQ[args.dataset]
        # venues stamp funding events a few ms off the hour — snap to the grid
        ts = pd.to_datetime(ts, utc=True).dt.floor(freq)
        gaps = validation.find_gaps(ts, freq)
        if not gaps.empty:
            clean = False
            print(
                f"\n{exchange} {asset}: {len(gaps)} gap(s), {gaps['n_missing'].sum()} bars missing"
            )
            print(gaps.to_string(index=False, max_rows=20))
    if clean:
        print(f"{args.dataset}: no gaps found")


def main() -> None:
    """CLI entry point (see module docstring for usage)."""
    parser = argparse.ArgumentParser(prog="quant_hub.data.ingestion", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("instruments", help="(re)build config/instruments.csv from venue metadata")

    def _add_common(p, exchanges):
        p.add_argument("--exchange", choices=exchanges, default=exchanges[0])
        group = p.add_mutually_exclusive_group(required=True)
        group.add_argument("--assets", help="comma-separated canonical ids, e.g. BTC,ETH")
        group.add_argument("--all", action="store_true", help="every instrument on the exchange")
        p.add_argument("--start", help="YYYY-MM-DD (default: venue launch)")
        p.add_argument("--end", help="YYYY-MM-DD (default: today UTC)")
        p.add_argument(
            "--proxy",
            choices=["direct", "scrapingbee"],
            default="direct",
            help="scrapingbee routes requests through the ScrapingBee IP pool "
            "(needs SCRAPING_BEE_KEY; spares the local IP's venue rate limits)",
        )

    ohlcv = sub.add_parser("ohlcv", help="backfill bars (15m; 1d on both venues)")
    _add_common(ohlcv, ["binance", "hyperliquid"])
    ohlcv.add_argument("--interval", choices=["15m", "1d"], default="15m")
    ohlcv.add_argument(
        "--refresh",
        type=int,
        default=2,
        help="hyperliquid: re-fetch and overwrite the last N stored bars (default 2)",
    )
    ohlcv.add_argument(
        "--full",
        action="store_true",
        help="hyperliquid: ignore stored state and re-fetch [start, end] (repairs holes)",
    )
    funding = sub.add_parser("funding", help="backfill funding")
    _add_common(funding, ["binance", "hyperliquid"])
    funding.add_argument(
        "--source",
        choices=["dumps", "rest", "both"],
        default="dumps",
        help="binance only: monthly dumps (default, lags up to a month), REST (fapi, "
        "may be geo-blocked), or dumps plus a REST tail for the current month",
    )
    oi = sub.add_parser("oi", help="backfill open interest (binance)")
    _add_common(oi, ["binance"])
    oi.add_argument(
        "--concurrency", type=int, default=1, help="parallel downloads per asset (default 1)"
    )

    l2 = sub.add_parser("l2hour", help="HL archive: summarise one L2 hour per day (requester pays)")
    _add_common(l2, ["hyperliquid"])
    l2.add_argument("--hour", type=int, default=9, help="UTC hour of the archive file (default 9)")
    l2.add_argument("--concurrency", type=int, default=8)
    l2.add_argument("--dry-run", action="store_true", help="print the egress estimate and exit")

    cot = sub.add_parser("cot", help="CFTC disaggregated Commitments of Traders (weekly, 2006+)")
    cot.add_argument("--start-year", type=int, default=cftc.FIRST_YEAR)
    cot.add_argument("--end-year", type=int)
    cot.add_argument("--codes", help="comma-separated CFTC codes to keep (default: all markets)")

    fut = sub.add_parser("futures", help="daily continuous commodity futures (yahoo / stooq)")
    fgroup = fut.add_mutually_exclusive_group(required=True)
    fgroup.add_argument("--assets", help="roots, e.g. HG,GC,SI (see connectors/futures_daily.py)")
    fgroup.add_argument("--all", action="store_true", help="the whole commodity basket")
    fut.add_argument("--source", choices=["yahoo", "stooq"], default="yahoo")
    fut.add_argument("--start", help="YYYY-MM-DD (yahoo only; default 1990-01-01)")
    fut.add_argument("--end", help="YYYY-MM-DD (yahoo only)")

    hip3 = sub.add_parser("hip3", help="HIP-3 builder perps on Hyperliquid: daily bars + funding")
    hip3.add_argument("--dexes", default="xyz,flx", help="comma-separated builder dex names")
    hip3.add_argument("--all-markets", action="store_true", help="not only commodities")

    gaps = sub.add_parser("gaps", help="report missing bars per asset")
    gaps.add_argument(
        "--dataset", choices=["ohlcv_15m", "ohlcv_1d", "funding", "open_interest"], required=True
    )
    gaps.add_argument("--exchange")

    args = parser.parse_args()
    if getattr(args, "proxy", "direct") == "scrapingbee":
        scrapingbee.enabled = True
    {
        "instruments": cmd_instruments,
        "ohlcv": cmd_ohlcv,
        "funding": cmd_funding,
        "oi": cmd_oi,
        "gaps": cmd_gaps,
        "l2hour": cmd_l2hour,
        "cot": cmd_cot,
        "futures": cmd_futures,
        "hip3": cmd_hip3,
    }[args.command](args)


if __name__ == "__main__":
    main()
