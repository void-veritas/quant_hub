r"""Health report of the commodities raw layer.

    uv run python -m quant_hub.commodities.health [--dataset futures_1d] [--exchange cme] \\
        [--assets xyz-COPPER] [--all]

Same per-asset table as `quant_hub.data.health` (first/last, rows, duplicates, OHLCV
sanity, gaps, staleness) over `data/commodities/raw`, with rules that fit the domain:

- HIP-3 perps (`ohlcv_1d`, `funding`): daily grid; a market flagged delisted in
  `reference/hip3_markets.csv` is stale by nature and not reported as such; a day
  without trades is stored as a zero-volume bar and is not a gap;
- `futures_1d` (one asset per contract): bars exist only on days with trades, so gaps
  are counted but never flagged; staleness is judged per product root (the newest bar
  across all its contracts), because an expired contract is not a stale feed; calendar
  spreads (`HGZ6-HGH7`) legitimately trade at negative prices;
- `options_1d`: one row per strike and day, so duplicate timestamps are by design; an
  expired option series is not stale, the newest series is;
- `futures_cont_1d`, `cot_disagg`: weekends and holidays are not gaps; staleness
  thresholds are 5 and 14 days (COT is weekly and published with a 3-day lag); COT
  staleness is judged on the newest market (hundreds of CFTC markets are dead), and
  Yahoo's OHLC glitches are reported but not flagged (cross-check data only).

Exit code 1 when any row is flagged, so it can gate `commodities.sync push`.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from quant_hub.commodities import hip3, storage
from quant_hub.commodities.connectors.databento import is_spread, root_of
from quant_hub.data import health as _health

STALE_DAYS = {
    "ohlcv_1d": 3.0,
    "funding": 3.0,
    "futures_1d": 5.0,
    "futures_1h": 5.0,
    "options_1d": 5.0,
    "futures_cont_1d": 5.0,
    "cot_disagg": 14.0,
}
GAPS_FLAGGED = {"ohlcv_1d", "funding"}  # the others are sparse or on trading days
# cross-check sources with known OHLC glitches (Yahoo): sanity problems are reported, not flagged
SANITY_INFORMATIONAL = {"futures_cont_1d"}
# datasets whose staleness is judged on the newest row only (expired contracts, dead CFTC markets)
STALE_BY_NEWEST = {"futures_1d", "futures_1h", "cot_disagg", "options_1d"}
# several rows per timestamp by design (one per strike): the duplicate/sorted checks do not apply
MULTI_ROW = {"options_1d"}
COLUMNS = [
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


def _hip3_status() -> dict[tuple[str, str], str]:
    m = hip3.load_markets()
    if m.empty:
        return {}
    return {
        (hip3.EXCHANGE, a): ("delisted" if d else "trading")
        for a, d in zip(m["asset"], m["delisted"].astype(bool), strict=True)
    }


def _drop_multirow_problems(problems: str) -> str:
    return "; ".join(
        p for p in problems.split("; ") if p and "duplicate" not in p and "unsorted" not in p
    )


def _drop_sign_problems(problems: str) -> str:
    return "; ".join(p for p in problems.split("; ") if p and "non-positive" not in p)


def report(
    datasets: list[str] | None = None,
    exchange: str | None = None,
    assets: list[str] | None = None,
    now: pd.Timestamp | None = None,
    root=None,
) -> pd.DataFrame:
    """Health table over every stored (dataset, exchange, asset) of the commodities raw layer."""
    root = root or storage.RAW_ROOT
    if not root.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = _health.report(datasets, exchange, assets, root=root, now=now)
    if df.empty:
        return df
    status = _hip3_status()
    multi = df["dataset"].isin(MULTI_ROW)
    df.loc[multi, "problems"] = df.loc[multi, "problems"].map(_drop_multirow_problems)
    # calendar spreads trade at negative prices: the sign check does not apply to them
    spread = df["dataset"].isin(["futures_1d", "futures_1h"]) & df["asset"].map(is_spread)
    df.loc[spread, "problems"] = df.loc[spread, "problems"].map(_drop_sign_problems)
    df["status"] = [
        status.get(
            (ex, a),
            {
                "futures_1d": "contract",
                "futures_1h": "contract",
                "cot_disagg": "market",
                "options_1d": "series",
            }.get(ds, ""),
        )
        for ds, ex, a in zip(df["dataset"], df["exchange"], df["asset"], strict=True)
    ]
    return df


def flag(df: pd.DataFrame, stale_days: float | None = None) -> pd.DataFrame:
    """Rows that need attention under the domain rules (see module docstring)."""
    if df.empty:
        return df
    limit = df["dataset"].map(STALE_DAYS).fillna(3.0) if stale_days is None else stale_days
    bad = (df["problems"] != "") & ~df["dataset"].isin(SANITY_INFORMATIONAL)
    # a zero-volume day of an illiquid perp is stored (synthetic row), not missing
    bad |= (df["missing"] > df["synthetic_rows"]) & df["dataset"].isin(GAPS_FLAGGED)
    live = ~df["status"].isin(["delisted", "contract", "market", "series"])
    bad |= (df["stale_days"] > limit) & live
    # futures and COT: staleness per product root (or per dataset), attached to the newest row
    for ds in STALE_BY_NEWEST:
        sel = df["dataset"] == ds
        if not sel.any():
            continue
        if ds in ("futures_1d", "futures_1h"):
            groups = df.loc[sel, "asset"].map(lambda a: root_of(a) or a)
        else:
            groups = pd.Series(ds, index=df.index[sel])
        newest = df.loc[sel, "stale_days"].groupby(groups).transform("min")
        lim = limit if isinstance(limit, int | float) else limit[sel]
        bad.loc[sel] |= (newest > lim) & (df.loc[sel, "stale_days"] == newest)
    return df[bad]


def main(argv=None) -> int:
    """CLI entry point; exit code 1 when any row is flagged."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", action="append", help="repeatable; default all")
    ap.add_argument("--exchange")
    ap.add_argument("--assets", help="comma-separated asset ids")
    ap.add_argument("--stale-days", type=float, help="override the per-dataset thresholds")
    ap.add_argument("--all", action="store_true", help="print every row, not only flagged ones")
    args = ap.parse_args(argv)
    df = report(args.dataset, args.exchange, args.assets.split(",") if args.assets else None)
    if df.empty:
        print("no commodities data stored")
        return 0
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
        print(f"{len(bad)} rows need attention:")
        print(bad.to_string(index=False))
    return 1 if not bad.empty else 0


if __name__ == "__main__":
    sys.exit(main())
