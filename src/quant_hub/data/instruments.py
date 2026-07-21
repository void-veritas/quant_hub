"""The instruments table: canonical asset <-> venue symbol mapping.

Git-tracked at `config/instruments.csv`. Everything downstream keys on
`canonical_id`; only connectors deal in venue symbols. Multiplier handles
denomination quirks (Binance 1000PEPEUSDT and HL kPEPE both = 1000x PEPE).

`listed_at` comes from venue metadata where available; `delisted_at` is left
empty until inferred from actual data coverage (a delisted perp's last bar).
Rebuild with `python -m quant_hub.data.ingestion instruments` — rebuilding
never removes rows, so delisted symbols stay in the table permanently.
"""

from __future__ import annotations

import pandas as pd

from quant_hub.utils.config import PROJECT_ROOT

INSTRUMENTS_PATH = PROJECT_ROOT / "config" / "instruments.csv"

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


def build_instruments() -> pd.DataFrame:
    """Fetch venue metadata and build the full instruments frame."""
    from quant_hub.data.connectors import binance_connector as bnc
    from quant_hub.data.connectors import hyperliquid_connector as hl

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


def save_instruments(df: pd.DataFrame) -> None:
    """Merge with the existing table (rows never disappear) and write."""
    if INSTRUMENTS_PATH.exists():
        existing = load_instruments()
        merged = pd.concat([existing, df], ignore_index=True)
        # fresh rows win, but a symbol that vanished from venue metadata survives
        df = merged.drop_duplicates(subset=["exchange", "symbol"], keep="last")
    dup = df.duplicated(subset=["exchange", "canonical_id"])
    if dup.any():
        raise ValueError(f"canonical_id collisions: {df.loc[dup, 'symbol'].tolist()}")
    INSTRUMENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values(["exchange", "canonical_id", "symbol"]).to_csv(INSTRUMENTS_PATH, index=False)


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
    df = pd.read_csv(INSTRUMENTS_PATH, dtype={"listed_at": str, "delisted_at": str}).fillna(
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
