"""Daily instruments refresh: live copy on the data volume, change report, Binance optional."""

from __future__ import annotations

import pandas as pd

from quant_hub.data import instruments as I


def _row(cid, ex, status, symbol=None):
    return {
        "canonical_id": cid,
        "exchange": ex,
        "symbol": symbol or cid,
        "multiplier": 1,
        "listed_at": "",
        "delisted_at": "",
        "status": status,
        "funding_interval_hours": 1,
    }


def test_changes_name_new_listings_and_delistings():
    old = pd.DataFrame(
        [_row("BTC", "hyperliquid", "trading"), _row("VINE", "hyperliquid", "trading")]
    )
    new = pd.DataFrame(
        [
            _row("BTC", "hyperliquid", "trading"),
            _row("VINE", "hyperliquid", "delisted"),
            _row("PEPE", "hyperliquid", "trading", "kPEPE"),
        ]
    )
    assert I.changes(old, new, "hyperliquid") == {"listed": ["PEPE"], "delisted": ["VINE"]}
    assert I.changes(new, new, "hyperliquid") == {"listed": [], "delisted": []}


def test_refresh_writes_the_live_copy_and_keeps_rows_a_venue_dropped(tmp_path, monkeypatch):
    live = tmp_path / "reference" / "instruments.csv"
    monkeypatch.setenv("QUANT_HUB_INSTRUMENTS", str(live))
    I.save_instruments(
        pd.DataFrame(
            [_row("BTC", "hyperliquid", "trading"), _row("BTC", "binance", "trading", "BTCUSDT")]
        )
    )
    assert live.exists() and len(I.load_instruments()) == 2
    # a Hyperliquid-only refresh (Binance unreachable) keeps the stored Binance rows
    out = I.save_instruments(
        pd.DataFrame(
            [_row("BTC", "hyperliquid", "trading"), _row("XTEN", "hyperliquid", "trading")]
        )
    )
    assert set(out["canonical_id"] + "@" + out["exchange"]) == {
        "BTC@hyperliquid",
        "XTEN@hyperliquid",
        "BTC@binance",
    }


def test_build_survives_an_unreachable_binance(monkeypatch):
    from quant_hub.data.connectors import hyperliquid_connector as hl

    monkeypatch.setattr(I, "_binance_rows", lambda: (_ for _ in ()).throw(ConnectionError("451")))
    monkeypatch.setattr(
        hl,
        "fetch_meta",
        lambda: [{"name": "BTC"}, {"name": "kPEPE"}, {"name": "VINE", "isDelisted": True}],
    )
    df = I.build_instruments()
    assert set(df["exchange"]) == {"hyperliquid"}
    assert dict(zip(df["canonical_id"], df["status"], strict=True)) == {
        "BTC": "trading",
        "PEPE": "trading",
        "VINE": "delisted",
    }


def test_readers_use_the_live_copy_once_it_exists(tmp_path, monkeypatch):
    monkeypatch.delenv("QUANT_HUB_INSTRUMENTS", raising=False)
    monkeypatch.setattr(I, "LIVE_INSTRUMENTS_PATH", tmp_path / "live.csv")
    assert (
        I.instruments_path() == I.INSTRUMENTS_PATH
        and I.instruments_path(write=True) == tmp_path / "live.csv"
    )
    (tmp_path / "live.csv").write_text("x")
    assert I.instruments_path() == tmp_path / "live.csv"
