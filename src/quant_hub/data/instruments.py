"""The instruments table: canonical asset <-> venue symbol mapping.

Git-tracked at `config/instruments.csv`. Everything downstream keys on
`canonical_id`; only connectors deal in venue symbols. Multiplier handles
denomination quirks (Binance 1000PEPEUSDT and HL kPEPE both = 1000x PEPE).

`listed_at` comes from venue metadata where available; `delisted_at` is left
empty until inferred from actual data coverage (a delisted perp's last bar).
Rebuild with `python -m quant_hub.data.ingestion instruments` — rebuilding
never removes rows, so delisted symbols stay in the table permanently.

Two copies: the git-tracked seed `config/instruments.csv` and the live table
`data/reference/instruments.csv` that the daily refresh writes (a production
image is read-only; the data directory is a volume). Readers use the live table
when it exists, else the seed; ``QUANT_HUB_INSTRUMENTS`` points at any other file.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from quant_hub.utils.config import PROJECT_ROOT

INSTRUMENTS_PATH = PROJECT_ROOT / "config" / "instruments.csv"  # git-tracked seed
LIVE_INSTRUMENTS_PATH = PROJECT_ROOT / "data" / "reference" / "instruments.csv"


def instruments_path(write: bool = False) -> Path:
    """The table to read (live copy if present, else the seed) or to write (the live copy)."""
    if os.environ.get("QUANT_HUB_INSTRUMENTS"):
        return Path(os.environ["QUANT_HUB_INSTRUMENTS"])
    if write or LIVE_INSTRUMENTS_PATH.exists():
        return LIVE_INSTRUMENTS_PATH
    return INSTRUMENTS_PATH


COLUMNS = [
    "canonical_id",
    "exchange",
    "symbol",
    "multiplier",
    "listed_at",
    "delisted_at",
    "status",
    "funding_interval_hours",
]


def _canonical_binance(symbol: str) -> tuple[str, int]:
    base = symbol.removesuffix("USDT")
    if base.startswith("1000000"):
        return base[7:], 1_000_000
    if base.startswith("1M") and len(base) > 2 and not base[2].isdigit():
        return base[2:], 1_000_000
    if base.startswith("1000"):
        return base[4:], 1000
    return base, 1


def _canonical_hyperliquid(coin: str) -> tuple[str, int]:
    if coin.startswith("k") and coin[1:].isupper():
        return coin[1:], 1000
    return coin.upper(), 1


def build_instruments(binance: bool = True) -> pd.DataFrame:
    """Fetch venue metadata and build the full instruments frame.

    ``binance=False`` (or an unreachable Binance, e.g. geo-blocked) builds the
    Hyperliquid rows only; ``save_instruments`` then keeps the Binance rows it has.
    """
    from quant_hub.data.connectors import hyperliquid_connector as hl

    rows: list[dict] = []
    if binance:
        try:
            rows += _binance_rows()
        except Exception as err:  # noqa: BLE001 (HL is the trading venue: it must still refresh)
            print(f"binance metadata unavailable, keeping the stored rows: {err}")
    for asset in hl.fetch_meta():
        canonical, multiplier = _canonical_hyperliquid(asset["name"])
        rows.append(
            {
                "canonical_id": canonical,
                "exchange": "hyperliquid",
                "symbol": asset["name"],
                "multiplier": multiplier,
                "listed_at": "",
                "delisted_at": "",
                "status": "delisted" if asset.get("isDelisted") else "trading",
                "funding_interval_hours": 1,
            }
        )
    return _dedupe_canonicals(pd.DataFrame(rows, columns=COLUMNS))


def _binance_rows() -> list[dict]:
    from quant_hub.data.connectors import binance_connector as bnc

    rows: list[dict] = []
    info = {s["symbol"]: s for s in bnc.fetch_exchange_info()}
    funding_intervals = {
        f["symbol"]: int(f["fundingIntervalHours"])
        for f in bnc._get(f"{bnc.FAPI}/fundingInfo").json()
    }
    dump_symbols = [s for s in bnc.list_dump_symbols() if s.endswith("USDT")]
    for symbol in sorted(set(dump_symbols) | {s for s in info if s.endswith("USDT")}):
        meta = info.get(symbol)
        is_perp = meta is None or meta.get("contractType") == "PERPETUAL"
        if not is_perp:
            continue
        canonical, multiplier = _canonical_binance(symbol)
        listed_at = ""
        if meta and meta.get("onboardDate"):
            listed_at = pd.Timestamp(meta["onboardDate"], unit="ms", tz="UTC").date().isoformat()
        rows.append(
            {
                "canonical_id": canonical,
                "exchange": "binance",
                "symbol": symbol,
                "multiplier": multiplier,
                "listed_at": listed_at,
                "delisted_at": "",
                "status": "trading" if meta and meta.get("status") == "TRADING" else "delisted",
                "funding_interval_hours": funding_intervals.get(symbol, 8),
            }
        )
    return rows


def _dedupe_canonicals(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce one canonical_id per exchange (storage partitions key on it).

    Venues occasionally carry the same asset under two denominations (e.g.
    Binance BOBUSDT and 1000000BOBUSDT). The trading / most recently listed
    symbol keeps the clean canonical id; the rest get a `<id>-<symbol>` id.
    """
    for (_, cid), grp in df.groupby(["exchange", "canonical_id"]):
        if len(grp) == 1:
            continue
        ranked = grp.sort_values(["status", "listed_at"], ascending=False)  # trading > delisted
        losers = ranked.index[1:]
        df.loc[losers, "canonical_id"] = cid + "-" + df.loc[losers, "symbol"]
    return df


def changes(old: pd.DataFrame, new: pd.DataFrame, exchange: str) -> dict[str, list[str]]:
    """Canonical ids newly trading and newly delisted on one venue between two tables."""

    def trading(df):
        return set(df[(df["exchange"] == exchange) & (df["status"] == "trading")]["canonical_id"])

    def known(df):
        return set(df[df["exchange"] == exchange]["canonical_id"])

    return {
        "listed": sorted(trading(new) - trading(old)),
        "delisted": sorted((trading(old) - trading(new)) & known(new)),
    }


def save_instruments(df: pd.DataFrame) -> pd.DataFrame:
    """Merge with the existing table (rows never disappear), write the live copy, return it."""
    if instruments_path().exists():
        existing = load_instruments()
        merged = pd.concat([existing, df], ignore_index=True)
        # fresh rows win, but a symbol that vanished from venue metadata survives
        df = merged.drop_duplicates(subset=["exchange", "symbol"], keep="last")
    dup = df.duplicated(subset=["exchange", "canonical_id"])
    if dup.any():
        raise ValueError(f"canonical_id collisions: {df.loc[dup, 'symbol'].tolist()}")
    out = instruments_path(write=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = df.sort_values(["exchange", "canonical_id", "symbol"]).reset_index(drop=True)
    df.to_csv(out, index=False)
    return df


def load_instruments(exchange: str | None = None, status: str | None = None) -> pd.DataFrame:
    """Read `config/instruments.csv`, optionally filtered by exchange and/or status.

    Parameters
    ----------
    exchange : str, optional
        Keep only rows for this venue (e.g. ``"binance"``, ``"hyperliquid"``).
    status : str, optional
        Keep only rows with this status (e.g. ``"trading"``, ``"delisted"``).

    Returns
    -------
    pandas.DataFrame
        The instruments table with an integer index reset after filtering.
    """
    df = pd.read_csv(instruments_path(), dtype={"listed_at": str, "delisted_at": str}).fillna(
        {"listed_at": "", "delisted_at": ""}
    )
    if exchange:
        df = df[df["exchange"] == exchange]
    if status:
        df = df[df["status"] == status]
    return df.reset_index(drop=True)


def symbol_map(exchange: str) -> dict[str, str]:
    """canonical_id -> venue symbol for one exchange (all statuses)."""
    df = load_instruments(exchange=exchange)
    return dict(zip(df["canonical_id"], df["symbol"], strict=True))
