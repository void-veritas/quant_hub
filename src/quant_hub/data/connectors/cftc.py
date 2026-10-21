"""CFTC Commitments of Traders, disaggregated futures-only report (weekly, from 2006).

Bulk history files on www.cftc.gov: one zip for 2006-2016, one per year after.
Each holds a CSV with every reported market; rows are keyed by the CFTC contract
market code (copper on COMEX: 085692). Positions are in contracts. `ts` is the
report's "as of" Tuesday (UTC midnight); the report is published the following
Friday, so a signal built on it is known from that Friday's close at the earliest.
"""

from __future__ import annotations

import io
import time
import zipfile

import httpx
import pandas as pd

BASE = "https://www.cftc.gov/files/dea/history"
FIRST_YEAR = 2006
COPPER_CODE = "085692"

# normalised column name -> our column
FIELDS = {
    "market_and_exchange_names": "market",
    "report_date_as_yyyy_mm_dd": "ts",
    "cftc_contract_market_code": "code",
    "open_interest_all": "open_interest",
    "prod_merc_positions_long_all": "producer_long",
    "prod_merc_positions_short_all": "producer_short",
    "swap_positions_long_all": "swap_long",
    "swap_positions_short_all": "swap_short",
    "swap_positions_spread_all": "swap_spread",
    "m_money_positions_long_all": "mm_long",
    "m_money_positions_short_all": "mm_short",
    "m_money_positions_spread_all": "mm_spread",
    "other_rept_positions_long_all": "other_long",
    "other_rept_positions_short_all": "other_short",
    "other_rept_positions_spread_all": "other_spread",
    "nonrept_positions_long_all": "nonrept_long",
    "nonrept_positions_short_all": "nonrept_short",
    "traders_tot_all": "traders",
}

_client = httpx.Client(timeout=120, follow_redirects=True, headers={"User-Agent": "quant_hub"})


def file_urls(first: int, last: int) -> list[str]:
    """Bulk files covering [first, last]: the 2006-2016 archive plus one zip per later year."""
    urls = []
    if first <= 2016:
        urls.append(f"{BASE}/fut_disagg_txt_hist_2006_2016.zip")
    urls += [f"{BASE}/fut_disagg_txt_{y}.zip" for y in range(max(first, 2017), last + 1)]
    return urls


def _norm(col: str) -> str:
    c = col.strip().lower().replace(" ", "_").replace("-", "_")
    while "__" in c:  # the CFTC header has "Swap__Positions_Short_All"
        c = c.replace("__", "_")
    return c


def parse(content: bytes) -> pd.DataFrame:
    """Disaggregated zip -> long frame (ts, code, market, positions...) for every market."""
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        frames = [pd.read_csv(zf.open(n), dtype=str, low_memory=False) for n in zf.namelist()]
    raw = pd.concat(frames, ignore_index=True)
    raw.columns = [_norm(c) for c in raw.columns]
    missing = [
        k for k in ("report_date_as_yyyy_mm_dd", "cftc_contract_market_code") if k not in raw
    ]
    if missing:
        raise ValueError(
            f"unexpected CFTC layout, missing {missing}; columns: {list(raw.columns)[:12]}"
        )
    out = raw[[k for k in FIELDS if k in raw]].rename(columns=FIELDS)
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    out["code"] = out["code"].str.strip().str.zfill(6)
    out["market"] = out["market"].str.strip()
    for c in out.columns:
        if c not in ("ts", "code", "market"):
            out[c] = pd.to_numeric(out[c].str.replace(",", ""), errors="coerce")
    return (
        out.drop_duplicates(subset=["code", "ts"])
        .sort_values(["code", "ts"])
        .reset_index(drop=True)
    )


def fetch(first: int = FIRST_YEAR, last: int | None = None, retries: int = 3) -> pd.DataFrame:
    """Download and parse every bulk file for [first, last]."""
    last = last or pd.Timestamp.now(tz="UTC").year
    frames = []
    for url in file_urls(first, last):
        for attempt in range(retries):
            try:
                resp = _client.get(url)
                if resp.status_code == 404:
                    break
                resp.raise_for_status()
                frames.append(parse(resp.content))
                break
            except (httpx.TransportError, httpx.HTTPStatusError):
                if attempt == retries - 1:
                    raise
                time.sleep(2**attempt)
    if not frames:
        return pd.DataFrame(columns=["ts", "code", "market"])
    out = pd.concat(frames, ignore_index=True)
    out = out[(out["ts"].dt.year >= first) & (out["ts"].dt.year <= last)]
    return out.drop_duplicates(subset=["code", "ts"], keep="last").reset_index(drop=True)
