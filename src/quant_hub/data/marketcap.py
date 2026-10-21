r"""Daily market caps for the point-in-time universe: CoinCodex (RW) spliced with CoinGecko.

The RW ``coincodex.marketcap`` snapshot stops at 2026-01-22 and is no longer updated.
After its last date the caps come from CoinGecko (Demo API). One loader for every
consumer (production, shadow book, backtests):

    from quant_hub.data.marketcap import load_marketcap
    mc = load_marketcap()        # Ticker, Date, Price, Volume, MarketCapUSD, source

Maintenance CLI (needs COINGECKO_API_KEY):

    python -m quant_hub.data.marketcap map        # ticker -> CoinGecko id map
    python -m quant_hub.data.marketcap backfill   # daily history of mapped coins (<= 365 days)
    python -m quant_hub.data.marketcap snapshot   # today's caps, top 500 (2 calls; daily job)
    python -m quant_hub.data.marketcap exclusions # CoinGecko ids of stablecoins / tokenized assets
    python -m quant_hub.data.marketcap check      # overlap with CoinCodex: level ratio per ticker

Files (data/external, synced to the bucket like the RW files):
    rw/coincodex_marketcap.feather          RW snapshot, read-only
    coingecko/id_map.csv                    Ticker, cg_id, name, rule
    coingecko/marketcap_daily.parquet       Ticker, Date, Price, Volume, MarketCapUSD, cg_id, kind
    coingecko/excluded.csv                  cg_id, category (non-coins: pegs, tokenized, wrappers)

What is ranked must be the same kind of thing on both sides of the seam. CoinGecko lists
many stablecoins and tokenized real-world assets that CoinCodex did not track (or that the
universe's exclusion list does not name); left in, they push real coins out of the top N.
``load_marketcap`` therefore drops (1) CoinGecko ids in non-coin categories, with the
CoinCodex tickers mapped to them, and (2) on both sources any ticker that trades at
1 USD +- 3% on 90% of its days (a USD peg, whatever its name).
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from quant_hub.data.storage import DATA_ROOT

EXTERNAL = DATA_ROOT / "external"
COINCODEX = EXTERNAL / "rw" / "coincodex_marketcap.feather"
CG_DIR = EXTERNAL / "coingecko"
ID_MAP = CG_DIR / "id_map.csv"
CG_DAILY = CG_DIR / "marketcap_daily.parquet"
EXCLUDED = CG_DIR / "excluded.csv"
COLS = ["Ticker", "Date", "Price", "Volume", "MarketCapUSD"]
# CoinGecko renamed Toncoin to GRAM (HL renamed TON to GRAM in July 2026); keep the CoinCodex
# ticker so the series stays one asset across the seam (the universe maps TONCOIN -> TON)
CG_ALIASES = {"GRAM": "TONCOIN"}
# CoinGecko categories that are not coins for a market-cap universe
NON_COIN_PREFIXES = ("tokenized-", "bridged-")
NON_COIN_CATEGORIES = {
    "stablecoins",
    "yield-bearing-stablecoins",
    "us-treasury-backed-stablecoin",
    "algorithmic-stablecoin",
    "bank-issued-stablecoin",
    "commodity-backed-stablecoin",
    "crypto-backed-stablecoin",
    "fiat-backed-stablecoin",
    "genius-act-compliant-stablecoin",
    "mica-compliant-stablecoin",
    "try-stablecoins",
    "wrapped-tokens",
    "liquid-staking-tokens",
    "ondo-tokenized-assets",
    "remona-tokenized-stocks",
    "remora-markets-tokenized-rstocks",
    "republic-tokenized-pre-ipo-assets",
    "tessera-tokenized-pre-ipo-assets",
    "four-meme-tokenized-stocks-4stock",
}


def non_coin_category(category_id: str) -> bool:
    """True for stablecoin, tokenized-asset, bridged/wrapped and liquid-staking categories."""
    return (
        category_id in NON_COIN_CATEGORIES
        or category_id.startswith(NON_COIN_PREFIXES)
        or category_id.endswith("-stablecoin")
    )


def usd_pegged(mc: pd.DataFrame, band: float = 0.03, share: float = 0.9) -> set[str]:
    """Tickers whose price is within 1 +- `band` USD on at least `share` of their days."""
    near = (mc["Price"] - 1.0).abs() <= band
    s = near.groupby(mc["Ticker"]).mean()
    return set(s[s >= share].index)


# ----------------------------------------------------------------------------- loading
def load_coincodex(path=COINCODEX) -> pd.DataFrame:
    """The RW CoinCodex snapshot with a datetime ``Date`` and a ``source`` column."""
    mc = pd.read_feather(path)
    mc["Date"] = pd.to_datetime(mc["Date"])
    mc["Ticker"] = mc["Ticker"].astype(str)
    return mc[COLS].assign(source="coincodex")


def load_marketcap(
    coincodex_path=COINCODEX, coingecko_path=CG_DAILY, excluded_path=EXCLUDED, id_map_path=ID_MAP
) -> pd.DataFrame:
    """CoinCodex through its last date, CoinGecko strictly after it (no overlap, no rescaling).

    Non-coins are dropped on both sides (module docstring): CoinGecko ids in
    ``excluded.csv`` and the tickers mapped to them, and USD-pegged tickers.
    """
    cc = load_coincodex(coincodex_path)
    cut = cc["Date"].max()
    if coingecko_path.exists():
        cg = pd.read_parquet(coingecko_path)
        bad_ids = set(pd.read_csv(excluded_path)["cg_id"]) if excluded_path.exists() else set()
        bad_tickers = set()
        if bad_ids and id_map_path.exists():
            idm = pd.read_csv(id_map_path)
            bad_tickers = set(idm.loc[idm["cg_id"].isin(bad_ids), "Ticker"])
        cg = cg[(cg["Date"] > cut) & ~cg["cg_id"].isin(bad_ids)].copy()
        cg["Ticker"] = cg["Ticker"].replace(CG_ALIASES)
        out = pd.concat([cc, cg[COLS].assign(source="coingecko")], ignore_index=True)
        out = out[~out["Ticker"].isin(bad_tickers)]
    else:
        out = cc
    out = out[~out["Ticker"].isin(usd_pegged(out))]
    return out.sort_values(["Ticker", "Date"]).reset_index(drop=True)


# ----------------------------------------------------------------------------- mapping
def build_id_map(
    cc_last: pd.DataFrame, top: pd.DataFrame, extra_markets: pd.DataFrame
) -> pd.DataFrame:
    """Ticker -> CoinGecko id.

    Universe of tickers: the CoinCodex tickers of its last date (what the universe knew
    in January) plus CoinGecko's current top 500 (what it must know now). A symbol shared
    by several coins resolves to the one with the largest current market cap (`rule`).
    """
    cand = pd.concat([top, extra_markets], ignore_index=True).drop_duplicates("id")
    cand = cand.assign(Ticker=cand["symbol"].str.upper()).dropna(subset=["market_cap"])
    cand = cand.sort_values("market_cap", ascending=False)
    n_ids = cand.groupby("Ticker")["id"].transform("nunique")
    best = cand.drop_duplicates("Ticker", keep="first").copy()
    best["rule"] = np.where(
        n_ids.loc[best.index] > 1, "largest cap among same symbol", "unique symbol"
    )
    want = set(cc_last["Ticker"].str.upper()) | set(top["symbol"].str.upper())
    best = best[best["Ticker"].isin(want)]
    return best.rename(columns={"id": "cg_id"})[["Ticker", "cg_id", "name", "rule"]].reset_index(
        drop=True
    )


def overlap_check(cc: pd.DataFrame, cg: pd.DataFrame, tol: float = 0.2) -> pd.DataFrame:
    """Per ticker on common dates: median CoinGecko/CoinCodex cap ratio, its spread, a flag."""
    m = cc[["Ticker", "Date", "MarketCapUSD"]].merge(
        cg[["Ticker", "Date", "MarketCapUSD"]], on=["Ticker", "Date"], suffixes=("_cc", "_cg")
    )
    m = m[(m["MarketCapUSD_cc"] > 0) & (m["MarketCapUSD_cg"] > 0)]
    m["lr"] = np.log(m["MarketCapUSD_cg"] / m["MarketCapUSD_cc"])
    g = m.groupby("Ticker")["lr"]
    out = pd.DataFrame(
        {
            "days": g.size(),
            "median_ratio": np.exp(g.median()),
            "lr_iqr": g.quantile(0.75) - g.quantile(0.25),
        }
    )
    out["flag"] = (out["median_ratio"] - 1).abs() > tol
    return out.sort_values("median_ratio").reset_index()


# ----------------------------------------------------------------------------- CLI
def _save_daily(new: pd.DataFrame) -> int:
    CG_DIR.mkdir(parents=True, exist_ok=True)
    if CG_DAILY.exists():
        old = pd.read_parquet(CG_DAILY)
        # history points (00:00 UTC) win over same-day snapshots taken later in the day
        new = pd.concat([old, new], ignore_index=True)
        new = new.sort_values("kind").drop_duplicates(["Ticker", "Date"], keep="first")
    new = new.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    tmp = CG_DAILY.with_suffix(".parquet.tmp")
    new.to_parquet(tmp, index=False)
    tmp.replace(CG_DAILY)
    return len(new)


def cmd_map(_args) -> None:
    """Build the ticker -> CoinGecko id map and write ``coingecko/id_map.csv``."""
    from quant_hub.data.connectors import coingecko as cg

    cc = load_coincodex()
    cc_last = cc[cc["Date"] == cc["Date"].max()]
    top = cg.markets(pages=2)
    missing = sorted(set(cc_last["Ticker"].str.upper()) - set(top["symbol"].str.upper()))
    allc = cg.coins_list()
    ids = allc[allc["symbol"].str.upper().isin(missing)]["id"].tolist()
    extra = cg.markets(ids=ids) if ids else pd.DataFrame(columns=top.columns)
    idm = build_id_map(cc_last, top, extra)
    CG_DIR.mkdir(parents=True, exist_ok=True)
    idm.to_csv(ID_MAP, index=False)
    amb = (idm["rule"] != "unique symbol").sum()
    unmapped = sorted(set(cc_last["Ticker"].str.upper()) - set(idm["Ticker"]))
    print(
        f"map: {len(idm)} tickers ({amb} resolved among same-symbol coins); "
        f"{len(unmapped)} CoinCodex tickers without a CoinGecko coin: {unmapped[:20]}"
    )


def cmd_backfill(args) -> None:
    """Daily history of every mapped coin into ``coingecko/marketcap_daily.parquet``."""
    from quant_hub.data.connectors import coingecko as cg

    idm = pd.read_csv(ID_MAP)
    frames, failed = [], []
    for i, r in enumerate(idm.itertuples(index=False), 1):
        try:
            h = cg.market_chart(r.cg_id, days=args.days)
        except Exception as e:  # noqa: BLE001 - keep going, report at the end
            failed.append(f"{r.Ticker}/{r.cg_id}: {str(e)[:80]}")
            continue
        frames.append(h.assign(Ticker=r.Ticker, cg_id=r.cg_id, kind="history"))
        if i % 50 == 0:
            print(f"backfill: {i}/{len(idm)}", flush=True)
    n = _save_daily(pd.concat(frames, ignore_index=True)) if frames else 0
    print(f"backfill: {len(frames)} coins, file rows {n}; failed {len(failed)}: {failed[:10]}")


def cmd_snapshot(_args) -> None:
    """Append today's caps of the current top 500 (the daily job)."""
    from quant_hub.data.connectors import coingecko as cg

    top = cg.markets(pages=2)
    today = pd.Timestamp(datetime.now(UTC).date())
    idm = pd.read_csv(ID_MAP) if ID_MAP.exists() else pd.DataFrame(columns=["Ticker", "cg_id"])
    known = dict(zip(idm["cg_id"], idm["Ticker"], strict=True))
    top = top[top["id"].isin(known) | ~top["symbol"].str.upper().isin(idm["Ticker"])]
    snap = pd.DataFrame(
        {
            "Ticker": top["id"].map(known).fillna(top["symbol"].str.upper()),
            "Date": today,
            "Price": top["current_price"],
            "Volume": top["total_volume"],
            "MarketCapUSD": top["market_cap"],
            "cg_id": top["id"],
            "kind": "snapshot",
        }
    ).drop_duplicates("Ticker")
    n = _save_daily(snap)
    print(f"snapshot {today.date()}: {len(snap)} coins, file rows {n}")


def cmd_exclusions(_args) -> None:
    """Write the CoinGecko ids of every non-coin category to ``coingecko/excluded.csv``."""
    from quant_hub.data.connectors import coingecko as cg

    cats = pd.DataFrame(cg._get("/coins/categories/list"))
    cats = cats[cats["category_id"].map(non_coin_category)]
    rows = []
    for cid in cats["category_id"]:
        for page in (1, 2):
            got = cg._get(
                "/coins/markets",
                {
                    "vs_currency": "usd",
                    "category": cid,
                    "order": "market_cap_desc",
                    "per_page": 250,
                    "page": page,
                },
            )
            rows += [dict(cg_id=c["id"], category=cid) for c in got]
            if len(got) < 250:
                break
    out = pd.DataFrame(rows).drop_duplicates("cg_id")
    CG_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(EXCLUDED, index=False)
    print(f"exclusions: {len(cats)} categories, {len(out)} coin ids")


def cmd_check(args) -> None:
    """Print the overlap statistics against CoinCodex and the flagged tickers."""
    cc = load_coincodex()
    cg = pd.read_parquet(CG_DAILY)
    chk = overlap_check(cc, cg, args.tol)
    print(
        f"overlap {cg['Date'].min().date()} -> {cc['Date'].max().date()}: {len(chk)} tickers, "
        f"median ratio {chk['median_ratio'].median():.3f}, "
        f"flagged (>{args.tol:.0%}) {int(chk['flag'].sum())}"
    )
    print(chk[chk["flag"]].to_string(index=False))


def main(argv=None) -> int:
    """CLI entry point (see module docstring)."""
    from quant_hub.utils.config import load_env

    load_env()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("map")
    b = sub.add_parser("backfill")
    b.add_argument("--days", type=int, default=365)
    sub.add_parser("snapshot")
    sub.add_parser("exclusions")
    c = sub.add_parser("check")
    c.add_argument("--tol", type=float, default=0.2)
    args = ap.parse_args(argv)
    cmds = {
        "map": cmd_map,
        "backfill": cmd_backfill,
        "snapshot": cmd_snapshot,
        "exclusions": cmd_exclusions,
        "check": cmd_check,
    }
    cmds[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
