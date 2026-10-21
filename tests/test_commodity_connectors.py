"""Parsers of the commodity data connectors on fixtures shaped like the real payloads."""

import io
import zipfile

import pandas as pd
import pytest

from quant_hub.data.connectors import cftc, futures_daily

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
        # the same copper week again (archives overlap): deduplicated
        '"COPPER- #1 - COMMODITY EXCHANGE INC.",241015,2024-10-15,085692,CMX,'
        "250000,40000,90000,20000,15000,3000,70000,25000,9000,10000,8000,2000,6000,5000,210",
    ]
    df = cftc.parse(_zip(rows))
    assert set(df["code"]) == {"085692", "088691"}
    cu = df[df["code"] == cftc.COPPER_CODE].iloc[0]
    assert cu["ts"] == pd.Timestamp("2024-10-15", tz="UTC")
    assert cu["mm_long"] == 70000 and cu["mm_short"] == 25000 and cu["swap_short"] == 15000
    assert cu["open_interest"] == 250000 and len(df) == 2


def test_cftc_file_urls_cover_the_archive_and_years():
    urls = cftc.file_urls(2006, 2018)
    assert urls[0].endswith("fut_disagg_txt_hist_2006_2016.zip")
    assert urls[-1].endswith("fut_disagg_txt_2018.zip") and len(urls) == 3
    assert cftc.file_urls(2020, 2021)[0].endswith("fut_disagg_txt_2020.zip")


def test_cftc_parse_rejects_an_unknown_layout():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("x.txt", "a,b\n1,2\n")
    with pytest.raises(ValueError, match="unexpected CFTC layout"):
        cftc.parse(buf.getvalue())


def test_yahoo_parse_keeps_the_session_date_and_drops_empty_closes():
    payload = {
        "chart": {
            "result": [
                {
                    "meta": {"exchangeTimezoneName": "America/New_York"},
                    # sessions stamped at 00:00 New York (04:00/05:00 UTC)
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
    assert df["close"].tolist() == [4.35, 4.3]


def test_stooq_parse_and_error_on_non_csv():
    text = (
        "Date,Open,High,Low,Close,Volume\n"
        "2024-10-15,4.3,4.4,4.2,4.35,100\n"
        "2024-10-16,4.31,4.35,4.28,4.3,120\n"
    )
    df = futures_daily.parse_stooq(text, "hg.f")
    assert len(df) == 2 and df["ts"].iloc[0] == pd.Timestamp("2024-10-15", tz="UTC")
    with pytest.raises(ValueError, match="no CSV"):
        futures_daily.parse_stooq("Exceeded the daily hits limit", "hg.f")


def test_hip3_asset_names_are_path_safe():
    from quant_hub.data.ingestion import hip3_asset

    assert hip3_asset("xyz:COPPER") == "xyz-COPPER"
