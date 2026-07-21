"""Storage round-trip, idempotency, gap detection, and loader queries."""

import pandas as pd
import pytest

from quant_hub.data import loaders, storage, validation


def _bars(start: str, periods: int, freq: str = "15min") -> pd.DataFrame:
    ts = pd.date_range(start, periods=periods, freq=freq, tz="UTC")
    return pd.DataFrame(
        {
            "ts": ts,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1.0,
        }
    )


def test_write_read_round_trip(tmp_path):
    df = _bars("2024-01-01", 96)
    storage.write_partition(df, "ohlcv_15m", "binance", "BTC", root=tmp_path)
    path = storage.partition_path("ohlcv_15m", "binance", "BTC", 2024, root=tmp_path)
    assert path.exists()
    back = pd.read_parquet(path)
    assert len(back) == 96
    assert back["ts"].is_monotonic_increasing


def test_rewrite_is_idempotent(tmp_path):
    df = _bars("2024-01-01", 96)
    storage.write_partition(df, "ohlcv_15m", "binance", "BTC", root=tmp_path)
    storage.write_partition(df, "ohlcv_15m", "binance", "BTC", root=tmp_path)
    path = storage.partition_path("ohlcv_15m", "binance", "BTC", 2024, root=tmp_path)
    assert len(pd.read_parquet(path)) == 96  # no duplicates


def test_overlapping_writes_merge_and_new_rows_win(tmp_path):
    first = _bars("2024-01-01", 96)
    storage.write_partition(first, "ohlcv_15m", "binance", "BTC", root=tmp_path)
    overlap = _bars("2024-01-01 12:00", 96)
    overlap["close"] = 200.0
    storage.write_partition(overlap, "ohlcv_15m", "binance", "BTC", root=tmp_path)
    back = pd.read_parquet(
        storage.partition_path("ohlcv_15m", "binance", "BTC", 2024, root=tmp_path)
    )
    assert len(back) == 96 + 48  # union, not sum
    ts_noon = pd.Timestamp("2024-01-01 12:00", tz="UTC")
    assert back.loc[back["ts"] == ts_noon, "close"].item() == 200.0


def test_year_boundary_split(tmp_path):
    df = _bars("2023-12-31 23:00", 8)  # crosses midnight into 2024
    storage.write_partition(df, "ohlcv_15m", "binance", "BTC", root=tmp_path)
    parts = storage.list_partitions("ohlcv_15m", root=tmp_path)
    assert sorted(parts["year"]) == [2023, 2024]


def test_find_gaps():
    ts = pd.date_range("2024-01-01", periods=100, freq="15min", tz="UTC")
    holed = ts.delete([10, 11, 12, 50])
    gaps = validation.find_gaps(pd.Series(holed), "15min")
    assert len(gaps) == 2
    assert gaps["n_missing"].tolist() == [3, 1]
    assert gaps.loc[0, "gap_start"] == ts[10]


def test_find_gaps_clean():
    ts = pd.date_range("2024-01-01", periods=50, freq="1h", tz="UTC")
    assert validation.find_gaps(pd.Series(ts), "1h").empty


def test_check_bars_flags_problems():
    df = _bars("2024-01-01", 10)
    df.loc[3, "high"] = 1.0  # high < low
    df.loc[5, "open"] = -1.0
    problems = validation.check_bars(df)
    assert any("high < low" in p for p in problems)
    assert any("non-positive" in p for p in problems)
    assert validation.check_bars(_bars("2024-01-01", 10)) == []


def test_loaders_filter_and_pivot(tmp_path, monkeypatch):
    monkeypatch.setattr(loaders, "RAW_ROOT", tmp_path)
    for asset in ("BTC", "ETH"):
        storage.write_partition(
            _bars("2024-01-01", 192), "ohlcv_15m", "binance", asset, root=tmp_path
        )

    df = loaders.load_ohlcv(["BTC"], start="2024-01-01", end="2024-01-01")
    assert set(df["asset"]) == {"BTC"}
    assert len(df) == 96  # end date inclusive through end of day
    assert df["exchange"].unique().tolist() == ["binance"]
    assert "year" not in df.columns

    wide = loaders.load_ohlcv(["BTC", "ETH"], field="close")
    assert list(wide.columns) == ["BTC", "ETH"]
    assert len(wide) == 192


def test_loaders_missing_dataset_message(tmp_path, monkeypatch):
    monkeypatch.setattr(loaders, "RAW_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="ingestion"):
        loaders.load_funding(["BTC"])
