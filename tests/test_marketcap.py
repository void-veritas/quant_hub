"""Market-cap splice (CoinCodex -> CoinGecko), id mapping, overlap check, RW key redaction."""

import pandas as pd

from quant_hub.data import marketcap as mcap
from quant_hub.data.connectors import robotwealth as rw


def _cc(tmp_path):
    rows = []
    for d in pd.date_range("2026-01-20", "2026-01-22"):
        rows += [
            dict(
                Ticker="BTC", Date=d.strftime("%Y-%m-%d"), Price=9e4, Volume=1.0, MarketCapUSD=100.0
            ),
            dict(
                Ticker="BONK2",
                Date=d.strftime("%Y-%m-%d"),
                Price=2e-5,
                Volume=1.0,
                MarketCapUSD=5.0,
            ),
        ]
    p = tmp_path / "cc.feather"
    pd.DataFrame(rows).to_feather(p)
    return p


def test_splice_takes_coingecko_strictly_after_the_coincodex_end(tmp_path):
    cc = _cc(tmp_path)
    cg = pd.DataFrame(
        {
            "Ticker": ["BTC", "BTC", "NEW"],
            "Date": pd.to_datetime(["2026-01-22", "2026-01-23", "2026-01-23"]),
            "Price": [9e4, 9.1e4, 3.0],
            "Volume": 1.0,
            "MarketCapUSD": [999.0, 110.0, 7.0],
            "cg_id": ["bitcoin", "bitcoin", "new-coin"],
            "kind": "history",
        }
    )
    cgp = tmp_path / "cg.parquet"
    cg.to_parquet(cgp)
    m = mcap.load_marketcap(cc, cgp)
    btc = m[m["Ticker"] == "BTC"].set_index("Date")
    assert btc.loc["2026-01-22", "MarketCapUSD"] == 100.0  # CoinCodex wins on its own dates
    assert (
        btc.loc["2026-01-23", "MarketCapUSD"] == 110.0
        and btc.loc["2026-01-23", "source"] == "coingecko"
    )
    assert "NEW" in set(m["Ticker"])  # a coin listed after January enters
    assert m.duplicated(["Ticker", "Date"]).sum() == 0


def test_usd_pegs_are_dropped_on_both_sources(tmp_path):
    days = pd.date_range("2026-01-20", "2026-01-22").strftime("%Y-%m-%d")
    rows = [
        dict(Ticker=t, Date=d, Price=px, Volume=1.0, MarketCapUSD=50.0)
        for d in days
        for t, px in (("USDG4", 1.001), ("ETH", 3000.0))
    ]
    cc = tmp_path / "cc.feather"
    pd.DataFrame(rows).to_feather(cc)
    cg = pd.DataFrame(
        {
            "Ticker": ["USYC", "ETH"],
            "Date": pd.to_datetime(["2026-01-23"] * 2),
            "Price": [1.0, 3100.0],
            "Volume": 1.0,
            "MarketCapUSD": [9e9, 4e11],
            "cg_id": ["hashnote-usyc", "ethereum"],
            "kind": "history",
        }
    )
    cgp = tmp_path / "cg.parquet"
    cg.to_parquet(cgp)
    m = mcap.load_marketcap(cc, cgp, tmp_path / "none.csv", tmp_path / "none_map.csv")
    assert set(m["Ticker"]) == {"ETH"}


def test_id_map_resolves_shared_symbols_by_cap():
    cc_last = pd.DataFrame({"Ticker": ["BTC", "PUMP"]})
    top = pd.DataFrame(
        {
            "id": ["bitcoin", "pump-fun", "pump-other"],
            "symbol": ["btc", "pump", "pump"],
            "name": ["", "", ""],
            "market_cap": [1e12, 5e9, 1e6],
        }
    )
    idm = mcap.build_id_map(cc_last, top, pd.DataFrame(columns=top.columns)).set_index("Ticker")
    assert idm.loc["PUMP", "cg_id"] == "pump-fun" and idm.loc["PUMP", "rule"].startswith("largest")
    assert idm.loc["BTC", "rule"] == "unique symbol"


def test_overlap_check_flags_level_mismatches():
    d = pd.to_datetime(["2026-01-01", "2026-01-02"])
    cc = pd.DataFrame(
        {"Ticker": ["A", "A", "B", "B"], "Date": list(d) * 2, "MarketCapUSD": [100, 100, 100, 100]}
    )
    cg = pd.DataFrame(
        {"Ticker": ["A", "A", "B", "B"], "Date": list(d) * 2, "MarketCapUSD": [101, 99, 150, 150]}
    )
    out = mcap.overlap_check(cc, cg, tol=0.2).set_index("Ticker")
    assert not out.loc["A", "flag"] and out.loc["B", "flag"]


def test_rw_errors_never_carry_the_key():
    msg = "Client error '404' for url 'https://api.robotwealth.com/v1/x?api_key=abc123&dataset=y'"
    assert "abc123" not in rw.redact(msg) and "api_key=***" in rw.redact(msg)
