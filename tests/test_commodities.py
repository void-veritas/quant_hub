"""Commodities domain: storage roots, sync layers, cost-gated Databento, HIP-3, health, migrate."""

import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest
from tests.test_data_layer import _FakeS3

from quant_hub.commodities import curated, health, hip3, migrate, storage
from quant_hub.commodities.connectors import cftc, futures_daily
from quant_hub.commodities.connectors import databento as dbn


@pytest.fixture
def croot(tmp_path, monkeypatch):
    root = tmp_path / "commodities"
    monkeypatch.setattr(storage, "COMMODITIES_ROOT", root)
    monkeypatch.setattr(storage, "RAW_ROOT", root / "raw")
    monkeypatch.setattr(storage, "REFERENCE_ROOT", root / "reference")
    monkeypatch.setattr(storage, "CURATED_ROOT", root / "curated")
    return root


def _bars(symbol, start, n, freq="1D"):
    ts = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    return pd.DataFrame(
        dict(ts=ts, open=4.0, high=4.1, low=3.9, close=4.05, volume=10, symbol=symbol)
    )


# --- storage and sync -------------------------------------------------------------


def test_storage_layers_and_partitions(croot):
    assert storage.write_partition(_bars("HGZ6", "2026-01-01", 3), "futures_1d", "cme", "HGZ6") == 3
    parts = storage.list_partitions("futures_1d")
    assert parts.iloc[0]["path"].startswith(str(croot / "raw" / "futures_1d"))
    assert storage.coverage("futures_1d").iloc[0]["rows"] == 3
    assert storage.list_partitions("nothing").empty
    p = storage.layer_path("reference", "tcrc", "benchmarks.csv")
    assert p == croot / "reference" / "tcrc" / "benchmarks.csv" and p.parent.exists()
    with pytest.raises(ValueError, match="unknown layer"):
        storage.layer_path("external", "x")


def test_sync_mirrors_the_three_layers_only(tmp_path):
    from quant_hub.commodities import sync

    root = tmp_path / "commodities"
    for rel in (
        "raw/futures_1d/exchange=cme/asset=HGZ6/year=2026/data.parquet",
        "reference/tcrc/benchmarks.csv",
        "curated/copper_front.parquet",
        "external/not_a_layer.csv",
        "raw/databento/x.dbn.zst",
    ):
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"x" * 10)
    s3 = _FakeS3()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(storage, "COMMODITIES_ROOT", root)
        mp.setattr(sync, "COMMODITIES_ROOT", root)
        assert sync.push("b", s3=s3) == 4
        assert set(s3.objects) == {
            "raw/futures_1d/exchange=cme/asset=HGZ6/year=2026/data.parquet",
            "reference/tcrc/benchmarks.csv",
            "curated/copper_front.parquet",
            "raw/databento/x.dbn.zst",
        }
        assert sync.push("b", dataset="futures_1d", s3=s3) == 0
        (root / "curated/copper_front.parquet").unlink()
        assert sync.pull("b", s3=s3) == 1 and (root / "curated/copper_front.parquet").exists()


# --- Databento -------------------------------------------------------------------


class _Meta:
    def __init__(self, usd, size):
        self.usd, self.size = usd, size
        self.calls = []

    def get_cost(self, **kw):
        self.calls.append(("cost", kw))
        return self.usd

    def get_billable_size(self, **kw):
        return self.size

    def get_dataset_range(self, dataset):
        return {"start": "2010-06-06T00:00:00", "end": "2026-10-21T00:00:00"}


class _Timeseries:
    def __init__(self):
        self.calls = []

    def get_range(self, path=None, **kw):
        self.calls.append(kw)
        with open(path, "wb") as fh:
            fh.write(b"DBN")
        df = _bars("HGZ6", "2026-01-01", 2).drop(columns="symbol")
        df["symbol"] = ["HGZ6", "HGZ6"]
        df["instrument_id"] = 42
        return df.rename(columns={"ts": "ts_event"}).set_index("ts_event")


class _FakeClient:
    def __init__(self, usd=1.74, size=10_000_000):
        self.metadata = _Meta(usd, size)
        self.timeseries = _Timeseries()


def test_databento_refuses_above_the_limit_without_downloading(tmp_path):
    cli = _FakeClient(usd=1.745)
    req = dbn.request(["HG.FUT"], "ohlcv-1d", "2010-06-06", "2026-10-21")
    est = dbn.estimate(req, cli)
    assert est.usd == 1.745 and est.mb == 10.0
    with pytest.raises(dbn.CostLimitExceeded, match="1.7450 USD"):
        dbn.fetch(req, max_usd=1.74, cli=cli, store_dir=tmp_path)
    assert cli.timeseries.calls == [] and not list(tmp_path.iterdir())
    # sub-cent excess is not an excess: Databento bills in cents
    cli = _FakeClient(usd=1.7412)
    dbn.fetch(req, max_usd=1.74, cli=cli, store_dir=tmp_path)
    assert len(cli.timeseries.calls) == 1


def test_databento_fetch_keeps_the_file_and_a_sidecar(tmp_path):
    cli = _FakeClient(usd=1.74)
    req = dbn.request(["HG.FUT"], "ohlcv-1d", "2010-06-06", "2026-10-21")
    store, path = dbn.fetch(req, max_usd=1.74, cli=cli, store_dir=tmp_path)
    assert path.name == "GLBX.MDP3_ohlcv-1d_HG.FUT_2010-06-06_2026-10-21.dbn.zst"
    assert path.read_bytes() == b"DBN"
    assert cli.timeseries.calls == [req]
    side = json.loads(path.with_suffix("").with_suffix(".json").read_text())
    assert side["request"] == req and side["estimate"]["usd"] == 1.74 and "licence" in side
    assert side["max_usd"] == 1.74
    assert dbn.dataset_end(cli=cli) == "2026-10-21"


def test_databento_bars_per_contract_with_unambiguous_ids(croot):
    # the raw symbol HGZ5 is two instruments ten years apart; the spread resolves leg by leg
    a = _bars("HGZ5", "2015-06-01", 2).assign(instrument_id=199046)
    b = _bars("HGZ5", "2025-06-01", 2).assign(instrument_id=6572)
    c = _bars("HGZ6-HGH7", "2026-01-01", 2).assign(instrument_id=3)
    df = pd.concat([a, b, c], ignore_index=True)
    bars = dbn.bars_to_frame(df.rename(columns={"ts": "ts_event"}).set_index("ts_event"))
    assert list(bars.columns) == dbn.BARS and bars["ts"].dt.tz is not None
    assert dbn.write_bars(bars) == 6
    parts = storage.list_partitions("futures_1d")
    assert sorted(set(parts["asset"])) == ["HGZ2015", "HGZ2025", "HGZ2026-HGH2027"]
    assert set(parts["exchange"]) == {"cme"}
    stored = pd.read_parquet(parts[parts["asset"] == "HGZ2025"]["path"].iloc[0])
    assert set(stored["symbol"]) == {"HGZ5"}
    # one instrument id reused a decade later (gold): the gap splits it into two contracts
    g = pd.concat([_bars("GCM1", "2010-06-07", 2), _bars("GCM1", "2021-03-01", 2)]).assign(
        instrument_id=77
    )
    junk = _bars("ZW:CF Z4H5K5N5", "2024-07-17", 2).assign(instrument_id=78)  # butterfly: skipped
    frame = pd.concat([g, junk]).rename(columns={"ts": "ts_event"}).set_index("ts_event")
    bars = dbn.bars_to_frame(frame)
    assert dbn.write_bars(bars) == 4
    assets = set(storage.list_partitions("futures_1d")["asset"])
    assert {"GCM2011", "GCM2021"} <= assets and not any(":" in a for a in assets)


def test_databento_symbol_helpers():
    assert dbn.root_of("HGZ6") == "HG" and dbn.root_of("HGZ2026") == "HG"
    assert dbn.root_of("HGZ2026-HGH2027") == "HG" and dbn.is_spread("HGZ6-HGH7")
    assert dbn.root_of("ZSX6") == "ZS" and dbn.expiry_code("ZSX2026") == "X2026"
    assert dbn.expiry_code("HGZ6-HGH7") is None and dbn.root_of("HG.FUT") is None
    assert dbn.resolve_year("M0", "2010-06-06") == 2010
    assert dbn.resolve_year("F0", "2018-06-29") == 2020
    assert (
        dbn.resolve_year("Z5", "2011-01-31") == 2015
        and dbn.resolve_year("Z5", "2023-12-22") == 2025
    )
    assert dbn.contract_id("HGZ6-HGH7", "2024-05-01") == "HGZ2026-HGH2027"
    assert dbn.contract_id("HG.FUT", "2024-05-01") == "HG.FUT"


# --- HIP-3 -----------------------------------------------------------------------


def test_hip3_names_and_markets_table():
    assert hip3.asset_of("xyz:COPPER") == "xyz-COPPER"
    assert hip3.is_commodity("xyz:COPPER") and hip3.is_commodity("xyz:CL")
    assert hip3.is_commodity("xyz:BRENTOIL") and not hip3.is_commodity("xyz:TSLA")
    meta = {
        "xyz": [{"name": "xyz:COPPER", "isDelisted": False, "szDecimals": 2, "maxLeverage": 10}],
        "flx": [{"name": "flx:COPPER", "isDelisted": True, "szDecimals": 1, "maxLeverage": 5}],
    }
    t = hip3.markets_table(meta, pd.Timestamp("2026-10-21T12:00:00Z"))
    assert list(t.columns) == hip3.MARKETS_COLUMNS
    assert t.set_index("asset")["delisted"].to_dict() == {"xyz-COPPER": False, "flx-COPPER": True}


# --- health ----------------------------------------------------------------------


def test_health_rules_of_the_domain(croot):
    now = pd.Timestamp("2026-10-21", tz="UTC")
    # HIP-3: live xyz stale, delisted flx stale, xyz with a gap
    storage.write_partition(
        _bars("xyz:GOLD", "2026-10-01", 20), "ohlcv_1d", hip3.EXCHANGE, "xyz-GOLD"
    )
    storage.write_partition(
        _bars("xyz:COPPER", "2026-10-01", 10), "ohlcv_1d", hip3.EXCHANGE, "xyz-COPPER"
    )
    storage.write_partition(
        _bars("flx:COPPER", "2026-06-01", 10), "ohlcv_1d", hip3.EXCHANGE, "flx-COPPER"
    )
    quiet = _bars("xyz:NATGAS", "2026-10-01", 20)
    quiet.loc[quiet.index[5], "volume"] = 0  # a day without trades is not a gap
    storage.write_partition(quiet, "ohlcv_1d", hip3.EXCHANGE, "xyz-NATGAS")
    gap = pd.concat(
        [
            _bars("xyz:SILVER", "2026-10-01", 20).iloc[:5],
            _bars("xyz:SILVER", "2026-10-01", 20).iloc[8:],
        ]
    )
    storage.write_partition(gap, "ohlcv_1d", hip3.EXCHANGE, "xyz-SILVER")
    hip3.markets_table(
        {
            "xyz": [
                {"name": "xyz:COPPER"},
                {"name": "xyz:GOLD"},
                {"name": "xyz:SILVER"},
                {"name": "xyz:NATGAS"},
            ],
            "flx": [{"name": "flx:COPPER", "isDelisted": True}],
        },
        now,
    ).to_csv(hip3.markets_path(), index=False)
    # futures: expired contract (old) and a live one; a sparse far contract; a spread
    storage.write_partition(_bars("HGU6", "2026-08-01", 30), "futures_1d", "cme", "HGU6")
    storage.write_partition(_bars("HGZ6", "2026-09-22", 30), "futures_1d", "cme", "HGZ6")
    storage.write_partition(_bars("HGZ7", "2026-09-01", 10, freq="5D"), "futures_1d", "cme", "HGZ7")
    neg = _bars("HGZ6-HGH7", "2026-10-01", 10).assign(open=-0.1, high=-0.05, low=-0.2, close=-0.1)
    storage.write_partition(neg, "futures_1d", "cme", "HGZ6-HGH7")
    # COT: weekly, last report 10 days ago -> fine under the 14-day rule
    storage.write_partition(
        _bars("c", "2026-07-07", 15, freq="7D").drop(columns=["open", "high", "low", "close"]),
        "cot_disagg",
        "cftc",
        "085692",
    )
    storage.write_partition(  # a market CFTC stopped reporting: not a stale feed
        _bars("c", "2016-07-07", 15, freq="7D").drop(columns=["open", "high", "low", "close"]),
        "cot_disagg",
        "cftc",
        "967654",
    )

    opt = pd.concat([_bars("HXEZ6 C650", "2026-10-11", 10), _bars("HXEZ6 C700", "2026-10-11", 10)])
    storage.write_partition(opt, "options_1d", "cme", "HXEZ2026", key=("ts", "symbol"))
    old_opt = _bars("HXEZ5 C400", "2025-09-01", 10)
    storage.write_partition(old_opt, "options_1d", "cme", "HXEZ2025", key=("ts", "symbol"))

    df = health.report(now=now)
    st = df.set_index(["dataset", "asset"])["status"]
    assert (
        st[("ohlcv_1d", "flx-COPPER")] == "delisted" and st[("ohlcv_1d", "xyz-GOLD")] == "trading"
    )
    assert st[("futures_1d", "HGZ6")] == "contract"
    bad = health.flag(df)
    flagged = set(zip(bad["dataset"], bad["asset"], strict=True))
    assert ("ohlcv_1d", "xyz-COPPER") in flagged  # live and stale
    assert ("ohlcv_1d", "xyz-SILVER") in flagged  # gap on the daily grid
    assert ("ohlcv_1d", "xyz-GOLD") not in flagged
    assert ("ohlcv_1d", "xyz-NATGAS") not in flagged  # zero-volume day, stored
    assert ("ohlcv_1d", "flx-COPPER") not in flagged  # delisted
    assert not any(d == "futures_1d" for d, _ in flagged)  # root HG is fresh; negative spread ok
    assert df.set_index("asset").loc["HGZ6-HGH7", "problems"] == ""
    assert not any(d == "options_1d" for d, _ in flagged)  # strikes share a day; old series expired
    assert st[("options_1d", "HXEZ2026")] == "series"
    assert ("cot_disagg", "085692") not in flagged
    assert ("cot_disagg", "967654") not in flagged  # dead market: not a stale feed
    assert st[("cot_disagg", "967654")] == "market"
    # when every contract of the root goes quiet, exactly the newest one is flagged
    late = health.report(now=now + pd.Timedelta(days=10))
    fut = health.flag(late)
    assert fut[fut["dataset"] == "futures_1d"]["asset"].tolist() == ["HGZ6"]
    assert health.flag(late, stale_days=100.0).empty is False  # xyz-SILVER's gap stays flagged


def test_health_without_data_is_empty(croot):
    assert health.report().empty and health.main([]) == 0


# --- migrate ---------------------------------------------------------------------


class _MoveS3(_FakeS3):
    def __init__(self):
        super().__init__()
        self.deleted = []

    def copy_object(self, Bucket, Key, CopySource, MetadataDirective):
        body, meta, t = self.objects[("src", CopySource["Key"])]
        self.objects[(Bucket, Key)] = (body, dict(meta), t)

    def head_object(self, Bucket, Key):
        body, meta, _ = self.objects[(Bucket, Key)]
        return {"Metadata": meta, "ContentLength": len(body)}

    def delete_object(self, Bucket, Key):
        self.deleted.append((Bucket, Key))
        del self.objects[(Bucket, Key)]

    def list_objects_v2(self, Bucket, Prefix="", MaxKeys=1000, ContinuationToken=None):
        items = [
            {"Key": k, "Size": len(b), "LastModified": t.to_pydatetime()}
            for (bk, k), (b, _, t) in self.objects.items()
            if bk == Bucket and k.startswith(Prefix)
        ]
        return {"Contents": items, "IsTruncated": False}


def test_migrate_moves_only_commodity_keys_and_verifies(tmp_path):
    s3 = _MoveS3()
    keys = [
        "raw/ohlcv_1d/exchange=hyperliquid_hip3/asset=xyz-COPPER/year=2026/data.parquet",
        "raw/funding/exchange=hyperliquid_hip3/asset=flx-GAS/year=2026/data.parquet",
        "external/tcrc/benchmarks.csv",
        "raw/ohlcv_1d/exchange=hyperliquid/asset=BTC/year=2026/data.parquet",
        "external/rw/x.feather",
    ]
    for k in keys:
        s3.objects[("src", k)] = (b"abc", {"sha256": "h"}, pd.Timestamp.now(tz="UTC"))
    assert migrate.destination_key(keys[2]) == "reference/tcrc/benchmarks.csv"
    assert migrate.destination_key(keys[3]) is None
    assert len(migrate.move_objects(s3, "src", "dst", dry_run=True)) == 3 and not s3.deleted
    moved = migrate.move_objects(s3, "src", "dst")
    assert moved == sorted(keys[:3])
    assert ("dst", "reference/tcrc/benchmarks.csv") in s3.objects
    assert ("src", keys[0]) not in s3.objects and ("src", keys[3]) in s3.objects
    # local move
    root, dest = tmp_path / "data", tmp_path / "data" / "commodities"
    for rel in (keys[0], keys[2], keys[3]):
        f = root / rel
        f.parent.mkdir(parents=True)
        f.write_bytes(b"x")
    assert sorted(migrate.move_local(root, dest)) == [
        "asset=xyz-COPPER/year=2026/data.parquet",
        "benchmarks.csv",
    ]
    assert (dest / keys[0]).exists() and (dest / "reference/tcrc/benchmarks.csv").exists()
    assert (root / keys[3]).exists() and not (root / "external/tcrc").exists()


# --- free connectors (parsers on fixtures shaped like the real payloads) ---------

CFTC_HEADER = (
    "Market_and_Exchange_Names,As_of_Date_In_Form_YYMMDD,Report_Date_as_YYYY-MM-DD,"
    "CFTC_Contract_Market_Code,CFTC_Market_Code,Open_Interest_All,Prod_Merc_Positions_Long_All,"
    "Prod_Merc_Positions_Short_All,Swap_Positions_Long_All,Swap__Positions_Short_All,"
    "Swap__Positions_Spread_All,M_Money_Positions_Long_All,M_Money_Positions_Short_All,"
    "M_Money_Positions_Spread_All,Other_Rept_Positions_Long_All,Other_Rept_Positions_Short_All,"
    "Other_Rept_Positions_Spread_All,NonRept_Positions_Long_All,NonRept_Positions_Short_All,"
    "Traders_Tot_All"
)


def _zip(rows: list[str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("f_year.txt", "\n".join([CFTC_HEADER, *rows]) + "\n")
    return buf.getvalue()


def test_cftc_parse_normalises_columns_and_codes():
    rows = [
        '"COPPER- #1 - COMMODITY EXCHANGE INC.",241015,2024-10-15,85692,CMX,'
        "250000,40000,90000,20000,15000,3000,70000,25000,9000,10000,8000,2000,6000,5000,210",
        '"GOLD - COMMODITY EXCHANGE INC.",241015,2024-10-15,088691,CMX,'
        "500000,1,2,3,4,5,6,7,8,9,10,11,12,13,300",
        '"COPPER- #1 - COMMODITY EXCHANGE INC.",241015,2024-10-15,085692,CMX,'
        "250000,40000,90000,20000,15000,3000,70000,25000,9000,10000,8000,2000,6000,5000,210",
    ]
    df = cftc.parse(_zip(rows))
    assert set(df["code"]) == {"085692", "088691"}
    cu = df[df["code"] == cftc.COPPER_CODE].iloc[0]
    assert cu["ts"] == pd.Timestamp("2024-10-15", tz="UTC")
    assert cu["mm_long"] == 70000 and cu["mm_short"] == 25000 and cu["swap_short"] == 15000
    assert cu["open_interest"] == 250000 and len(df) == 2
    urls = cftc.file_urls(2006, 2018)
    assert urls[0].endswith("fut_disagg_txt_hist_2006_2016.zip") and len(urls) == 3
    with pytest.raises(ValueError, match="unexpected CFTC layout"):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("x.txt", "a,b\n1,2\n")
        cftc.parse(buf.getvalue())


def test_free_futures_parsers():
    payload = {
        "chart": {
            "result": [
                {
                    "meta": {"exchangeTimezoneName": "America/New_York"},
                    "timestamp": [1728964800, 1729051200, 1729137600],
                    "indicators": {
                        "quote": [
                            {
                                "open": [4.3, 4.31, None],
                                "high": [4.4, 4.35, None],
                                "low": [4.2, 4.28, None],
                                "close": [4.35, 4.3, None],
                                "volume": [100, 120, None],
                            }
                        ]
                    },
                }
            ]
        }
    }
    df = futures_daily.parse_yahoo(payload, "HG=F")
    assert list(df["ts"]) == [
        pd.Timestamp("2024-10-15", tz="UTC"),
        pd.Timestamp("2024-10-16", tz="UTC"),
    ]
    text = "Date,Open,High,Low,Close,Volume\n2024-10-15,4.3,4.4,4.2,4.35,100\n"
    assert len(futures_daily.parse_stooq(text, "hg.f")) == 1
    with pytest.raises(ValueError, match="no CSV"):
        futures_daily.parse_stooq("Exceeded the daily hits limit", "hg.f")


# --- curated copper layer ------------------------------------------------------


def _contract(symbol, start, n, price, vol):
    ts = pd.bdate_range(start, periods=n, tz="UTC")
    p = np.linspace(price, price * 1.02, n)
    return pd.DataFrame(
        dict(asset=symbol, ts=ts, open=p, high=p * 1.01, low=p * 0.99, close=p, volume=float(vol))
    )


def test_curated_calendar_and_expiry():
    assert curated.resolve_expiry_month("HGM0", pd.Timestamp("2010-06-06")) == pd.Timestamp(
        "2010-06-01"
    )
    assert curated.resolve_expiry_month("HGZ2026", pd.Timestamp("2024-03-01")) == pd.Timestamp(
        "2026-12-01"
    )
    assert curated._third_last_business_day(2026, 12) == pd.Timestamp("2026-12-29")
    with pytest.raises(ValueError, match="not an outright"):
        curated.resolve_expiry_month("HGZ6-HGH7", pd.Timestamp("2026-01-01"))


def test_curated_roll_by_volume_with_hysteresis_and_returns():
    # Z6 is front until H7's volume takes over; H7 then stays front even when Z6 is louder
    z6 = _contract("HGZ2026", "2026-10-01", 40, 6.6, 100)
    z6.loc[z6.index[30:], "volume"] = 20.0
    z6.loc[z6.index[36:], "volume"] = 500.0  # noise after the roll: must not roll back
    h7 = _contract("HGH2027", "2026-10-01", 60, 6.7, 10)
    h7.loc[h7.index[30:], "volume"] = 200.0
    k7 = _contract("HGK2027", "2026-10-01", 60, 6.8, 20)
    j7 = _contract("HGJ2027", "2026-10-01", 60, 6.75, 0.5)  # thin serial month: never the second
    sunday = pd.DataFrame(
        dict(
            asset="HGZ2026",
            ts=[pd.Timestamp("2026-10-04", tz="UTC")],
            open=1,
            high=1,
            low=1,
            close=1,
            volume=1.0,
        )
    )
    bars = pd.concat([z6, h7, k7, j7, sunday], ignore_index=True)
    curve, table = curated.build_curve(bars)
    assert list(table.index) == ["HGZ2026", "HGH2027", "HGJ2027", "HGK2027"]
    assert curve.index.dayofweek.max() < 5  # the Sunday bar is gone
    assert curve["front"].iloc[0] == "HGZ2026" and curve["second"].iloc[0] == "HGH2027"
    assert curve["front"].iloc[-1] == "HGH2027" and curve["second"].iloc[-1] == "HGK2027"
    rolls = curve.index[curve["roll"]]
    assert len(rolls) == 1
    # on the roll day the return is still earned on Z6 (held from t-1)
    r = curve.loc[rolls[0]]
    i = z6.index[z6["ts"] == rolls[0]][0]
    assert r["ret_front"] == pytest.approx(z6["close"].iloc[i] / z6["close"].iloc[i - 1] - 1)
    # carry: Z6 cheaper than H7 -> contango -> negative carry; days to second ~ 3 months
    first = curve.iloc[0]
    assert first["carry_ann"] < 0 and 80 < first["days_to_second"] < 100
    assert curve["ret_spread"].iloc[1] == pytest.approx(
        curve["ret_front"].iloc[1] - curve["ret_second"].iloc[1]
    )
    # Z6 drops out of the front at its guard even if still loud; after its last bar its price is NaN
    close = curated.engine_inputs(bars)
    assert close["HGZ2026"].isna().iloc[-1] and close["HGH2027"].notna().all()


def test_westmetall_parse_skips_repeated_headers():
    from quant_hub.commodities.connectors import westmetall

    html = (
        "<table><tr><th>date</th><th>LME Copper Cash-Settlement</th><th>LME Copper 3-month</th>"
        "<th>LME Copper stock</th></tr>"
        "<tr><td>21. October 2026</td><td>14441.00</td><td>14315.00</td><td>245275</td></tr>"
        "<tr><td>date</td><td>LME Copper Cash-Settlement</td><td>LME Copper 3-month</td>"
        "<td>LME Copper stock</td></tr>"
        "<tr><td>20. October 2026</td><td>14,540.00</td><td>14357.00</td><td>233800</td></tr>"
        "</table>"
    )
    df = westmetall.parse(html)
    assert list(df.columns) == westmetall.COLUMNS and len(df) == 2
    assert (
        df["ts"].iloc[0] == pd.Timestamp("2026-10-20", tz="UTC") and df["cash"].iloc[0] == 14540.0
    )
    assert df["stock"].iloc[1] == 245275
    with pytest.raises(ValueError, match="unexpected Westmetall"):
        westmetall.parse(
            "<table><tr><th>a</th><th>b</th></tr><tr><td>1</td><td>2</td></tr></table>"
        )
