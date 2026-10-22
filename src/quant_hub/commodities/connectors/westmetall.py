"""LME copper daily reference prices and stocks from Westmetall (free, one HTML table per year).

`https://www.westmetall.com/en/markdaten.php?action=table&field=LME_Cu_cash&year=YYYY`
lists, per LME business day, the cash settlement, the 3-month price (USD/t) and the
LME stock (t). Years run from 2008. `ts` is the LME date at UTC midnight; the cash
and 3M prices are the LME official settlements fixed around 12:30-13:15 London, so
a same-day COMEX close (00:00 UTC next day in Databento's UTC bars) is later.
Other metals use the same page with `LME_Al_cash`, `LME_Ni_cash`, etc.
"""

from __future__ import annotations

import io
import time

import httpx
import pandas as pd

URL = "https://www.westmetall.com/en/markdaten.php"
FIRST_YEAR = 2008
COLUMNS = ["ts", "cash", "m3", "stock"]

_client = httpx.Client(
    timeout=60, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0 (quant_hub research)"}
)


def parse(html: str) -> pd.DataFrame:
    """The year table -> (ts, cash, m3, stock); columns matched by name, not position."""
    tables = pd.read_html(io.StringIO(html))
    t = max(tables, key=len)
    cols = {c: str(c).lower() for c in t.columns}
    pick = {}
    for c, low in cols.items():
        if low == "date":
            pick["ts"] = c
        elif "cash" in low:
            pick["cash"] = c
        elif "3-month" in low or "3 month" in low:
            pick["m3"] = c
        elif "stock" in low:
            pick["stock"] = c
    missing = [k for k in COLUMNS if k not in pick]
    if missing:
        raise ValueError(f"unexpected Westmetall layout, missing {missing}: {list(t.columns)}")
    t = t[t[pick["ts"]].astype(str) != "date"]  # the header row repeats inside the table
    out = pd.DataFrame(
        {
            "ts": pd.to_datetime(t[pick["ts"]], format="%d. %B %Y", utc=True, errors="coerce"),
            "cash": pd.to_numeric(
                t[pick["cash"]].astype(str).str.replace(",", ""), errors="coerce"
            ),
            "m3": pd.to_numeric(t[pick["m3"]].astype(str).str.replace(",", ""), errors="coerce"),
            "stock": pd.to_numeric(
                t[pick["stock"]].astype(str).str.replace(",", ""), errors="coerce"
            ),
        }
    )
    return out.dropna(subset=["ts", "cash"]).sort_values("ts").reset_index(drop=True)


def fetch_year(field: str, year: int, retries: int = 3) -> pd.DataFrame:
    """One year of one metal's table (`field` such as `LME_Cu_cash`)."""
    for attempt in range(retries):
        try:
            r = _client.get(URL, params={"action": "table", "field": field, "year": year})
            r.raise_for_status()
            return parse(r.text)
        except (httpx.TransportError, httpx.HTTPStatusError):
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def fetch(
    field: str = "LME_Cu_cash", first: int = FIRST_YEAR, last: int | None = None
) -> pd.DataFrame:
    """All years [first, last] of one metal, concatenated and deduplicated on `ts`."""
    last = last or pd.Timestamp.now(tz="UTC").year
    frames = []
    for year in range(first, last + 1):
        frames.append(fetch_year(field, year))
        time.sleep(0.5)
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates("ts", keep="last").sort_values("ts").reset_index(drop=True)
