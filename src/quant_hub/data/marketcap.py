r"""Daily market caps for the point-in-time universe: the RW CoinCodex snapshot, extended.

The RW ``coincodex.marketcap`` snapshot stops at 2026-01-22 and is no longer updated.
After its last date the caps come, by default, from CoinCodex's own public endpoints:
the same source, so the same coin set and cap definitions on both sides of the seam
(``source="coincodex"``). CoinGecko (Demo API) is kept as a cross-check and fallback
(``source="coingecko"``); on the same dates its top 50 differs by ~12 names because it
ranks many stablecoins and tokenized assets that CoinCodex does not. One loader for every
consumer (production, shadow book, backtests):

    from quant_hub.data.marketcap import load_marketcap
    mc = load_marketcap()        # Ticker, Date, Price, Volume, MarketCapUSD, source

Maintenance CLI (needs COINGECKO_API_KEY):

    python -m quant_hub.data.marketcap cx-candidates  # CoinCodex tickers to track beyond the RW 500
    python -m quant_hub.data.marketcap cx-backfill    # CoinCodex daily history after the snapshot
    python -m quant_hub.data.marketcap cx-snapshot    # last 3 days of the top 300 (daily job)
    python -m quant_hub.data.marketcap map        # ticker -> CoinGecko id map
    python -m quant_hub.data.marketcap backfill   # daily history of mapped coins (<= 365 days)
    python -m quant_hub.data.marketcap snapshot   # today's caps, top 500 (2 calls; daily job)
    python -m quant_hub.data.marketcap exclusions # CoinGecko ids of stablecoins / tokenized assets
    python -m quant_hub.data.marketcap check      # overlap with CoinCodex: level ratio per ticker

Files (data/external, synced to the bucket like the RW files):
    rw/coincodex_marketcap.feather          RW snapshot, read-only
    coincodex/tickers.csv                   Ticker, coin_name, rank, how (RW 500 + newcomers)
    coincodex/marketcap_daily.parquet       Ticker, Date, Price, Volume, MarketCapUSD
    coingecko/id_map.csv                    Ticker, cg_id, name, rule
    coingecko/marketcap_daily.parquet       Ticker, Date, Price, Volume, MarketCapUSD, cg_id, kind
    coingecko/excluded.csv                  cg_id, category (non-coins: pegs, tokenized, wrappers)

What is ranked must be the same kind of thing on both sides of the seam. CoinGecko lists
many stablecoins and tokenized real-world assets that CoinCodex did not track (or that the
universe's exclusion list does not name); left in, they push real coins out of the top N.
With ``exclude_non_coins=True`` the loader drops (1) CoinGecko ids in non-coin categories,
with the CoinCodex tickers mapped to them, and (2) on both sources any ticker that trades
at 1 USD +- 3% on 90% of its days (a USD peg, whatever its name). This is a change of the
universe rule (RW ranks everything CoinCodex ranks, minus a fixed exclusion list), so it is
off by default and evaluated as its own candidate.
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
CX_DIR = EXTERNAL / "coincodex"
CX_TICKERS = CX_DIR / "tickers.csv"
CX_DAILY = CX_DIR / "marketcap_daily.parquet"
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
    source: str = "coincodex",
    exclude_non_coins: bool = False,
    coincodex_path=COINCODEX,
    extension_path=None,
    excluded_path=EXCLUDED,
    id_map_path=ID_MAP,
) -> pd.DataFrame:
    """The RW snapshot through its last date, then ``source`` strictly after it.

    ``source``: "coincodex" (default, same source as the snapshot) or "coingecko". No
    overlap and no rescaling at the seam. ``exclude_non_coins`` see module docstring.
    """
    cc = load_coincodex(coincodex_path)
    cut = cc["Date"].max()
    path = extension_path or (CX_DAILY if source == "coincodex" else CG_DAILY)
    bad_tickers: set[str] = set()
    if path.exists():
        ext = pd.read_parquet(path)
        ext = ext[ext["Date"] > cut].copy()
        if source == "coingecko":
            bad_ids = set()
            if exclude_non_coins and excluded_path.exists():
                bad_ids = set(pd.read_csv(excluded_path)["cg_id"])
            if bad_ids and id_map_path.exists():
                idm = pd.read_csv(id_map_path)
                bad_tickers = set(idm.loc[idm["cg_id"].isin(bad_ids), "Ticker"])
            ext = ext[~ext["cg_id"].isin(bad_ids)]
            ext["Ticker"] = ext["Ticker"].replace(CG_ALIASES)
        elif source != "coincodex":
            raise ValueError(f"unknown source {source!r}")
        out = pd.concat([cc, ext[COLS].assign(source=source)], ignore_index=True)
    else:
        out = cc
    if exclude_non_coins:
        out = out[~out["Ticker"].isin(bad_tickers | usd_pegged(out))]
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


def _cx_save(new: pd.DataFrame) -> int:
    CX_DIR.mkdir(parents=True, exist_ok=True)
    if CX_DAILY.exists():
        new = pd.concat([pd.read_parquet(CX_DAILY), new], ignore_index=True)
        new = new.drop_duplicates(["Ticker", "Date"], keep="last")  # re-fetched days win
    new = new.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    tmp = CX_DAILY.with_suffix(".parquet.tmp")
    new.to_parquet(tmp, index=False)
    tmp.replace(CX_DAILY)
    return len(new)


def _name_match(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    na, nb = (" ".join(x.lower().replace("(", " ").replace(")", " ").split()) for x in (a, b))
    return na in nb or nb in na or na.split()[0] == nb.split()[0]


def cmd_cx_candidates(args) -> None:
    """Tickers to track: the RW snapshot's 500 plus confirmed newcomers.

    CoinCodex's coin list endpoint is unreliable, so newcomers are probed: every HL symbol
    and every CoinGecko top-300 ticker that the snapshot does not know. A probe tries the
    symbol and its numbered variants (CoinCodex disambiguates with suffixes, e.g. HYPE8);
    a hit needs a market-cap rank <= ``--max-rank`` and, when CoinGecko has a name for the
    ticker, a matching coin name.
    """
    from quant_hub.data.connectors import coincodex as cx
    from quant_hub.data.instruments import load_instruments

    cc = load_coincodex()
    rw_t = set(cc["Ticker"])
    known = rw_t | {t.rstrip("0123456789") for t in rw_t}
    names = {}
    if ID_MAP.exists():
        idm = pd.read_csv(ID_MAP)
        names = dict(zip(idm["Ticker"], idm["name"], strict=True))
    cands = set(load_instruments(exchange="hyperliquid")["canonical_id"]) - known
    if CG_DAILY.exists():
        cg = pd.read_parquet(CG_DAILY)
        last = cg[cg["Date"] == cg["Date"].max()].sort_values("MarketCapUSD", ascending=False)
        cands |= set(last.head(300)["Ticker"]) - known
    rows = [dict(Ticker=t, coin_name="", rank=None, how="rw snapshot") for t in sorted(rw_t)]
    for t in sorted(c for c in cands if c.isascii() and c.isalnum()):
        hit = None
        for sym in [t] + [f"{t}{i}" for i in range(2, 10)]:
            meta = cx.coin(sym)
            if meta is None:
                if sym != t:
                    break  # numbered variants are consecutive
                continue
            rank = meta.get("market_cap_rank")
            if (
                rank
                and rank <= args.max_rank
                and (t not in names or _name_match(meta.get("coin_name"), names[t]))
            ):
                hit = dict(
                    Ticker=sym, coin_name=meta.get("coin_name"), rank=rank, how=f"newcomer ({t})"
                )
                break
        if hit:
            rows.append(hit)
    out = pd.DataFrame(rows)
    CX_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(CX_TICKERS, index=False)
    new = out[out["how"] != "rw snapshot"]
    print(
        f"cx-candidates: {len(cands)} probed, {len(new)} newcomers kept: "
        f"{new['Ticker'].tolist()[:40]}"
    )


def cmd_cx_backfill(args) -> None:
    """Daily CoinCodex history from the day after the snapshot to today for every tracked ticker."""
    from quant_hub.data.connectors import coincodex as cx

    tickers = pd.read_csv(CX_TICKERS)["Ticker"].tolist()
    start = (load_coincodex()["Date"].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end = args.end or datetime.now(UTC).strftime("%Y-%m-%d")
    frames, empty, failed = [], [], []
    for i, t in enumerate(tickers, 1):
        try:
            h = cx.daily_history(t, start, end)
        except Exception as e:  # noqa: BLE001 - keep going, report at the end
            failed.append(f"{t}: {str(e)[:60]}")
            continue
        if h.empty:
            empty.append(t)
        else:
            frames.append(h.assign(Ticker=t))
        if i % 50 == 0:
            print(f"cx-backfill: {i}/{len(tickers)}", flush=True)
    n = _cx_save(pd.concat(frames, ignore_index=True)) if frames else 0
    print(
        f"cx-backfill {start} -> {end}: {len(frames)} tickers, file rows {n}; "
        f"no data {len(empty)}: {empty[:15]}; failed {len(failed)}: {failed[:5]}"
    )


def cmd_cx_snapshot(args) -> None:
    """Re-fetch the last 3 days for the top ``--top`` tickers by latest cap (daily job)."""
    from quant_hub.data.connectors import coincodex as cx

    m = load_marketcap()
    last = m[m["Date"] == m["Date"].max()].sort_values("MarketCapUSD", ascending=False)
    tracked = set(pd.read_csv(CX_TICKERS)["Ticker"]) if CX_TICKERS.exists() else set()
    tickers = [t for t in last["Ticker"] if t in tracked][: args.top]
    today = datetime.now(UTC).date()
    start = (pd.Timestamp(today) - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    frames = [cx.daily_history(t, start, today.isoformat()).assign(Ticker=t) for t in tickers]
    n = _cx_save(pd.concat(frames, ignore_index=True)) if frames else 0
    print(f"cx-snapshot {start} -> {today}: {len(tickers)} tickers, file rows {n}")


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
    cxc = sub.add_parser("cx-candidates")
    cxc.add_argument("--max-rank", type=int, default=500)
    cxb = sub.add_parser("cx-backfill")
    cxb.add_argument("--end", help="YYYY-MM-DD (default today UTC)")
    cxs = sub.add_parser("cx-snapshot")
    cxs.add_argument("--top", type=int, default=300)
    sub.add_parser("map")
    b = sub.add_parser("backfill")
    b.add_argument("--days", type=int, default=365)
    sub.add_parser("snapshot")
    sub.add_parser("exclusions")
    c = sub.add_parser("check")
    c.add_argument("--tol", type=float, default=0.2)
    args = ap.parse_args(argv)
    cmds = {
        "cx-candidates": cmd_cx_candidates,
        "cx-backfill": cmd_cx_backfill,
        "cx-snapshot": cmd_cx_snapshot,
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
