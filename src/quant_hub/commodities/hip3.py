"""HIP-3 builder-deployed perps on Hyperliquid (commodities by default).

The venue is Hyperliquid, but the underlying and the reference prices are
commodities, so the data lives in the commodities domain
(`ohlcv_1d` and `funding`, exchange=hyperliquid_hip3, asset=<dex>-<NAME>).
Market metadata (delisted flag, lot size, leverage) is written to
`reference/hip3_markets.csv`; `health` reads the delisted flag from there.
"""

from __future__ import annotations

import time
from datetime import UTC, date, datetime, timedelta
from datetime import time as dtime

import pandas as pd

from quant_hub.commodities import storage
from quant_hub.data.connectors import hyperliquid_connector as hl

EXCHANGE = "hyperliquid_hip3"
DEXES = ("xyz", "flx")
FIRST_DAY = date(2023, 1, 1)
COMMODITY_WORDS = (
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
COMMODITY_EXACT = ("CL",)  # xyz:CL is WTI crude
MARKETS_COLUMNS = ["coin", "dex", "asset", "delisted", "sz_decimals", "max_leverage", "fetched_at"]


def asset_of(coin: str) -> str:
    """'xyz:COPPER' -> 'xyz-COPPER' (a colon is a poor path component on macOS)."""
    return coin.replace(":", "-")


def is_commodity(coin: str) -> bool:
    """Whether a HIP-3 market name looks like a commodity (not an equity, index or FX)."""
    name = coin.split(":")[-1].upper()
    return name in COMMODITY_EXACT or any(w in name for w in COMMODITY_WORDS)


def markets_table(meta_by_dex: dict[str, list[dict]], fetched_at: datetime) -> pd.DataFrame:
    """`reference/hip3_markets.csv` rows from `fetch_meta(dex=...)` payloads."""
    rows = [
        dict(
            coin=m["name"],
            dex=dex,
            asset=asset_of(m["name"]),
            delisted=bool(m.get("isDelisted", False)),
            sz_decimals=m.get("szDecimals"),
            max_leverage=m.get("maxLeverage"),
            fetched_at=fetched_at.isoformat(timespec="seconds"),
        )
        for dex, markets in meta_by_dex.items()
        for m in markets
    ]
    return pd.DataFrame(rows, columns=MARKETS_COLUMNS)


def markets_path():
    """Location of the HIP-3 market list in the reference layer."""
    return storage.layer_path("reference", "hip3_markets.csv")


def load_markets() -> pd.DataFrame:
    """The stored market list, or an empty frame when none was written yet."""
    path = markets_path()
    if not path.exists():
        return pd.DataFrame(columns=MARKETS_COLUMNS)
    return pd.read_csv(path)


def _utc(day: date) -> datetime:
    return datetime.combine(day, dtime.min, tzinfo=UTC)


def _existing_funding_end(asset: str) -> pd.Timestamp | None:
    parts = storage.list_partitions("funding")
    parts = parts[(parts["exchange"] == EXCHANGE) & (parts["asset"] == asset)]
    if parts.empty:
        return None
    ts = pd.concat([pd.read_parquet(p, columns=["ts"])["ts"] for p in parts["path"]])
    return pd.to_datetime(ts, utc=True).max()


def ingest(dexes=DEXES, all_markets: bool = False, pause: float = 1.1) -> pd.DataFrame:
    """Daily bars and funding of HIP-3 markets; refreshes the market list. Returns it."""
    now = datetime.now(UTC)
    meta = {dex: hl.fetch_meta(dex=dex) for dex in dexes}
    table = markets_table(meta, now)
    table.to_csv(markets_path(), index=False)
    end = _utc(now.date() + timedelta(days=1))
    closed_before = pd.Timestamp(now).floor("D")
    for _, m in table.iterrows():
        coin, asset = m["coin"], m["asset"]
        if not all_markets and not is_commodity(coin):
            continue
        bars = hl.fetch_candles(coin, "1d", _utc(FIRST_DAY), end)
        bars = bars[bars["ts"] < closed_before]
        if not bars.empty:
            storage.write_partition(bars, "ohlcv_1d", EXCHANGE, asset)
        have = _existing_funding_end(asset)
        f_from = have + pd.Timedelta("1ms") if have is not None else _utc(FIRST_DAY)
        fund = hl.fetch_funding(coin, f_from, end)
        if not fund.empty:
            storage.write_partition(fund, "funding", EXCHANGE, asset)
        print(f"hip3 {coin:18s} bars {len(bars):4d} funding +{len(fund)}", flush=True)
        time.sleep(pause)
    return table
