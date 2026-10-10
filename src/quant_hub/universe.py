"""Point-in-time universe selection for daily cross-sectional strategies.

Moved from yolo-microstructure (notes 006, 010, 013).

Rule: on each rebalance date R (first calendar day of a month), rank assets by
the median daily dollar volume over the `lookback` days strictly before R,
require at least `min_age_days` of real trading before R, and keep the top `n`
until the next rebalance. Membership on any date therefore depends only on
data up to the previous day. Delisted assets drop out when their bars stop.

Input is the long daily-bar frame from quant_hub (`ohlcv_1d`): ts, asset,
close, volume (base units). Real trading starts at the first bar with
volume > 0 (HL serves synthetic zero-volume bars before launch).
"""

from __future__ import annotations

import pandas as pd


def dollar_volume(bars: pd.DataFrame) -> pd.DataFrame:
    """Wide (date x asset) daily dollar volume; NaN before real trading starts."""
    b = bars.copy()
    b["date"] = pd.to_datetime(b["ts"], utc=True).dt.floor("D").dt.tz_localize(None)
    b["dv"] = b["volume"] * b["close"]
    wide = b.pivot_table(index="date", columns="asset", values="dv", aggfunc="sum")
    started = (wide > 0).cumsum() > 0
    return wide.where(started)


def liquidity_universe(
    bars: pd.DataFrame,
    n: int = 10,
    lookback: int = 30,
    min_age_days: int = 60,
    rebalance: str = "MS",
    exclude: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Long frame (date, asset, in_universe) with monthly point-in-time membership."""
    dv = dollar_volume(bars).drop(columns=list(exclude), errors="ignore")
    first_real = dv.notna().idxmax()  # first date with a real bar per asset
    dates = dv.index
    rebal = pd.date_range(dates[0], dates[-1], freq=rebalance)
    rebal = rebal[rebal > dates[0]]
    member = pd.DataFrame(False, index=dates, columns=dv.columns)
    for i, R in enumerate(rebal):
        window = dv.loc[(dv.index < R) & (dv.index >= R - pd.Timedelta(days=lookback))]
        if len(window) < max(5, lookback // 2):
            continue
        score = window.median(axis=0, skipna=True)
        age_ok = (R - first_real).dt.days >= min_age_days
        has_data = window.notna().sum() >= lookback // 2
        score = score.where(age_ok & has_data).dropna()
        top = score.sort_values(ascending=False).head(n).index
        end = rebal[i + 1] if i + 1 < len(rebal) else dates[-1] + pd.Timedelta(days=1)
        member.loc[(member.index >= R) & (member.index < end), top] = True
    # an asset whose bars stopped (delisted) cannot be a member
    member &= dv.notna()
    out = member.stack(future_stack=True).rename("in_universe").reset_index()
    out.columns = ["date", "asset", "in_universe"]
    return out


def membership_summary(universe: pd.DataFrame) -> pd.DataFrame:
    """Per asset: first/last membership date and number of member days."""
    m = universe[universe["in_universe"]]
    return (
        m.groupby("asset")["date"]
        .agg(first="min", last="max", days="count")
        .sort_values("days", ascending=False)
    )


# ---------------------------------------------------------------- market-cap universe (v1.2)

STABLECOINS = frozenset(
    "USDT USDC BUSD DAI TUSD USDP UST USTC USDE USDS FDUSD PYUSD USDD FRAX LUSD GUSD USDN "
    "HUSD SUSD "
    "EURS USDJ USDX USD1 RLUSD USD0 CUSD MIM DOLA USDY FEI USDL AUSD XAUT PAXG USDK USDQ EURC GHO "
    "CRVUSD USDB BUIDL USDE2 USDS5 USDX3 EUROC".split()
)
DUPLICATES = frozenset(
    "BTCD HBTC IBBTC RBTC SBTC3 WNXM WBTC WETH STETH WSTETH WBETH WEETH RETH CBBTC BTCB "
    "EZETH RSETH "
    "BNSOL JITOSOL MSOL FBTC LBTC TBTC".split()
)
TICKER_MAP = {"TONCOIN": "TON"}


def normalise_ticker(t: str, known: set[str] | None = None) -> str:
    """Coincodex symbol -> venue symbol: TONCOIN->TON; HYPE8->HYPE when the base is known."""
    if t in TICKER_MAP:
        return TICKER_MAP[t]
    base = t.rstrip("0123456789")
    if base != t and known is not None and base in known and t not in known:
        return base
    return t


def cap_universe(
    mc: pd.DataFrame,
    n: int = 10,
    smooth: int = 7,
    min_history: int = 30,
    exclude: frozenset[str] = STABLECOINS | DUPLICATES,
    known: set[str] | None = None,
    extend_prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Long frame (date, asset, in_universe): RW's rule, top-n by market cap on the previous day.

    RW (crypto-pod notebook 1): drop stablecoins and duplicate/wrapped coins, rank
    `MarketCapUSD` within each date, member on date d iff rank on d-1 <= n.
    Robustness added here because the coincodex snapshot has days where large
    coins carry a missing cap: the cap is forward-filled up to 3 days and replaced
    by its `smooth`-day rolling median before ranking, and a ticker needs
    `min_history` days of data before it can rank. Point-in-time by construction.

    `extend_prices` (long: ts/date, asset, close) extends the cap series past the
    snapshot's last date as cap_last × close_t / close_last (supply held fixed),
    so the rule keeps running on live prices when the cap file is stale.
    """
    m = mc[["Ticker", "Date", "MarketCapUSD"]].copy()
    m["Date"] = pd.to_datetime(m["Date"])
    m = m[~m["Ticker"].isin(exclude)]
    m["asset"] = m["Ticker"].map(lambda t: normalise_ticker(t, known))
    wide = m.pivot_table(index="Date", columns="asset", values="MarketCapUSD", aggfunc="max")
    wide = wide.asfreq("D").ffill(limit=3)
    if extend_prices is not None:
        px = extend_prices.copy()
        px["date"] = (
            pd.to_datetime(px["ts"] if "ts" in px else px["date"], utc=True)
            .dt.floor("D")
            .dt.tz_localize(None)
        )
        px = (
            px.pivot_table(index="date", columns="asset", values="close", aggfunc="last")
            .asfreq("D")
            .ffill(limit=3)
        )
        last = wide.index[-1]
        if px.index[-1] > last:
            base_cap = wide.loc[last].dropna()
            cols = [
                c
                for c in base_cap.index
                if c in px.columns and pd.notna(px.loc[last, c])
                if last in px.index
            ]
            future = px.loc[px.index > last, cols]
            ext = future / px.loc[last, cols] * base_cap[cols]
            wide = pd.concat([wide, ext.reindex(columns=wide.columns)])
    cap = wide.rolling(smooth, min_periods=max(1, smooth // 2)).median()
    history = wide.notna().cumsum()
    cap = cap.where(history >= min_history)
    rank = cap.rank(axis=1, ascending=False, method="first")
    member = (rank <= n).shift(1).fillna(False).astype(bool)  # yesterday's rank decides today
    out = member.stack(future_stack=True).rename("in_universe").reset_index()
    out.columns = ["date", "asset", "in_universe"]
    return out
