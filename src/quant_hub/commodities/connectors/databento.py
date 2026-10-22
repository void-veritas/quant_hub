"""Databento Historical: paid data behind a mandatory cost estimate.

    from quant_hub.commodities.connectors import databento as dbn
    req = dbn.request(["HG.FUT"], "ohlcv-1d", "2010-06-06", "2026-10-21")
    dbn.estimate(req)                      # free: USD and billable bytes, nothing downloaded
    store, path = dbn.fetch(req, max_usd=1.74)   # refuses when the estimate exceeds max_usd

Every download is stored as the DBN file Databento delivered
(`data/commodities/raw/databento/<stem>.dbn.zst`) next to a JSON sidecar with the
request, the estimate, the billable size, the time of purchase and the licence note.
Daily bars are then laid out per contract in the `futures_1d` dataset
(exchange=cme, asset=contract symbol); spreads keep their `A-B` symbol.

Quirks of GLBX.MDP3 (CME Globex) worth knowing:
- parent symbology `HG.FUT` resolves to every copper outright future AND every
  calendar spread of the root; `is_spread` tells them apart;
- `ohlcv-1d` bars are aggregated per UTC day, not per CME trading session
  (17:00-16:00 Chicago), and the close is the last trade, not the settlement.
  Settlement and open interest come from the `statistics` schema (step 2 of the plan);
- bars exist only for days with trades: far-dated contracts are sparse by nature.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from quant_hub.commodities import storage
from quant_hub.utils.config import get_env

DATASET = "GLBX.MDP3"
DATASET_START = "2010-06-06"  # first day of GLBX.MDP3 history
EXCHANGE = "cme"
LICENCE = (
    "Databento Historical, licensed to the account holder for internal research use; "
    "not for redistribution."
)
BARS = ["ts", "open", "high", "low", "close", "volume", "symbol", "instrument_id"]
_MONTHS = "FGHJKMNQUVXZ"
_CONTRACT = re.compile(rf"^(?P<root>[A-Z0-9]+?)(?P<month>[{_MONTHS}])(?P<year>\d{{1,4}})$")


class CostLimitExceeded(RuntimeError):
    """The estimated price of a request is above the caller's limit; nothing was bought."""


@dataclass(frozen=True)
class Estimate:
    """Price and billable size of one request, from Databento's free metadata endpoints."""

    usd: float
    billable_bytes: int

    @property
    def mb(self) -> float:
        """Billable size in megabytes."""
        return self.billable_bytes / 1e6


def client():
    """Databento Historical client; the key comes from `DATABENTO_API_KEY` (env or .env)."""
    import databento as db

    return db.Historical(get_env("DATABENTO_API_KEY"))


def request(
    symbols: list[str],
    schema: str,
    start: str,
    end: str,
    dataset: str = DATASET,
    stype_in: str = "parent",
) -> dict:
    """The keyword set shared by `metadata.get_cost` and `timeseries.get_range`."""
    return dict(
        dataset=dataset,
        symbols=list(symbols),
        schema=schema,
        start=start,
        end=end,
        stype_in=stype_in,
    )


def dataset_end(dataset: str = DATASET, cli=None) -> str:
    """Last available date of a dataset (YYYY-MM-DD), the natural `end` of a backfill."""
    cli = cli or client()
    return str(cli.metadata.get_dataset_range(dataset)["end"])[:10]


def estimate(req: dict, cli=None) -> Estimate:
    """Price in USD and billable bytes of `req`; free, downloads nothing."""
    cli = cli or client()
    usd = float(cli.metadata.get_cost(**req))
    size = int(cli.metadata.get_billable_size(**req))
    return Estimate(usd, size)


def file_stem(req: dict) -> str:
    """`GLBX.MDP3_ohlcv-1d_HG.FUT_2010-06-06_2026-10-21`: one file per request."""
    syms = "+".join(req["symbols"]).replace("/", "_")
    return f"{req['dataset']}_{req['schema']}_{syms}_{req['start']}_{req['end']}"


def fetch(
    req: dict,
    max_usd: float,
    cli=None,
    store_dir: Path | None = None,
):
    """Buy `req` only if its estimate is at most `max_usd`; keep the DBN file and a sidecar.

    Returns `(DBNStore, path)`. Raises `CostLimitExceeded` before any paid call when
    the estimate, rounded to the cent (what Databento bills), is above the limit.
    The sidecar `<stem>.json` records what was bought.
    """
    cli = cli or client()
    est = estimate(req, cli)
    if round(est.usd, 2) > round(max_usd, 2):  # Databento bills in cents; compare at that precision
        raise CostLimitExceeded(
            f"estimate {est.usd:.4f} USD ({est.mb:.1f} MB) exceeds the limit {max_usd:.2f} USD; "
            "nothing downloaded"
        )
    store_dir = store_dir or storage.layer_path("raw", "databento").parent / "databento"
    store_dir.mkdir(parents=True, exist_ok=True)
    path = store_dir / f"{file_stem(req)}.dbn.zst"
    data = cli.timeseries.get_range(**req, path=str(path))
    sidecar = {
        "request": req,
        "estimate": asdict(est),
        "max_usd": max_usd,
        "purchased_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "file": path.name,
        "licence": LICENCE,
    }
    path.with_suffix("").with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n")
    return data, path


def bars_to_frame(store) -> pd.DataFrame:
    """OHLCV DBN store (or its `to_df()` frame) -> `BARS` columns, `ts` = bar open, UTC."""
    df = store if isinstance(store, pd.DataFrame) else store.to_df()
    if "ts_event" in df.columns:
        df = df.set_index("ts_event")
    out = df.reset_index().rename(columns={"ts_event": "ts"})
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    for col in ("open", "high", "low", "close"):
        out[col] = out[col].astype(float)
    out["volume"] = out["volume"].astype("int64")
    if "instrument_id" not in out:
        out["instrument_id"] = pd.NA
    out["symbol"] = out["symbol"].astype(str)
    return out[BARS].sort_values(["symbol", "ts"]).reset_index(drop=True)


def resolve_year(code: str, first_bar) -> int:
    """Four-digit delivery year of a month-year code (`Z6`, `Z26`, `Z2026`) seen on `first_bar`.

    Globex symbols carry one digit and are reused every ten years (HGZ5 was December
    2015 and is December 2025 again), so the decade is the earliest one in which the
    delivery month is not before the contract's first bar (minus a month of slack).
    """
    month = _MONTHS.index(code[0]) + 1
    digits = code[1:]
    if len(digits) == 4:
        return int(digits)
    first = pd.Timestamp(first_bar).tz_localize(None).normalize()
    if len(digits) == 2:
        return 2000 + int(digits) if int(digits) < 70 else 1900 + int(digits)
    base = first.year - first.year % 10
    for year in (base + int(digits) - 10, base + int(digits), base + int(digits) + 10):
        if pd.Timestamp(year=year, month=month, day=1) >= first - pd.Timedelta(days=31):
            return year
    raise ValueError(f"cannot place {code} seen first on {first.date()}")


def contract_id(symbol: str, first_bar) -> str:
    """Unambiguous asset id: `HGZ6` seen in 2024 -> `HGZ2026`; spreads leg by leg."""
    legs = []
    for leg in symbol.split("-"):
        m = _CONTRACT.match(leg)
        if m is None:
            return symbol
        legs.append(f"{m['root']}{m['month']}{resolve_year(m['month'] + m['year'], first_bar)}")
    return "-".join(legs)


def write_bars(df: pd.DataFrame, exchange: str = EXCHANGE, dataset: str = "futures_1d") -> int:
    """Lay daily bars out per contract: `futures_1d/exchange=cme/asset=<contract id>/year=.../`.

    One Databento instrument_id is one contract; its asset id carries the full delivery
    year (`contract_id`), so the two instruments that shared the raw symbol `HGZ5` land
    in `HGZ2015` and `HGZ2025`. The raw `symbol` column is kept inside the rows.
    """
    n = 0
    keys = ["symbol", "instrument_id"] if df["instrument_id"].notna().any() else ["symbol"]
    for _, sub in df.groupby(keys, dropna=False):
        asset = contract_id(str(sub["symbol"].iloc[0]), sub["ts"].min())
        n += storage.write_partition(sub, dataset, exchange, asset)
    return n


def read_file(path: Path):
    """The DBN store of a file bought earlier (no network, no cost)."""
    import databento as db

    return db.DBNStore.from_file(path)


def is_spread(symbol: str) -> bool:
    """`HGZ6-HGH7` is a calendar spread, `HGZ6` an outright."""
    return "-" in symbol


def root_of(symbol: str) -> str | None:
    """Product root of a contract or spread symbol (`HGZ6` -> `HG`); None if not a contract."""
    m = _CONTRACT.match(symbol.split("-")[0])
    return m["root"] if m else None


def expiry_code(symbol: str) -> str | None:
    """Month-year code of an outright (`HGZ2026` -> `Z2026`), None for spreads or non-contracts."""
    if is_spread(symbol):
        return None
    m = _CONTRACT.match(symbol)
    return f"{m['month']}{m['year']}" if m else None
