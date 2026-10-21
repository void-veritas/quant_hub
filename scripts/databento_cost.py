"""Databento cost estimate for the commodities plan: nothing is downloaded.

    DATABENTO_API_KEY=... uv run --with databento python cost.py

Prints USD and MB per (symbols, schema) for CME Globex (GLBX.MDP3), parent symbology
(HG.FUT = every copper future). The key is read from the environment by the client.
"""

import os
import sys

import databento as db

DATASET = "GLBX.MDP3"
START = "2010-06-06"  # GLBX.MDP3 history starts here
COPPER = ["HG.FUT"]
BASKET = [
    f"{r}.FUT"
    for r in [
        "HG",
        "GC",
        "SI",
        "PL",
        "PA",  # metals
        "CL",
        "NG",
        "HO",
        "RB",
        "BZ",  # energy
        "ZC",
        "ZS",
        "ZW",
        "ZM",
        "ZL",
        "KE",
        "ZO",  # grains
        "LE",
        "HE",
        "GF",  # livestock
    ]
]
PLAN = [
    ("copper futures, daily bars", COPPER, "ohlcv-1d"),
    ("copper futures, statistics (settlement, open interest)", COPPER, "statistics"),
    ("copper futures, contract definitions", COPPER, "definition"),
    ("basket of 20, daily bars", BASKET, "ohlcv-1d"),
    ("basket of 20, statistics", BASKET, "statistics"),
    ("basket of 20, contract definitions", BASKET, "definition"),
    ("copper options, daily bars (implied vol later)", ["HG.OPT"], "ohlcv-1d"),
    ("copper options, statistics", ["HG.OPT"], "statistics"),
]


def main() -> int:
    """Print the cost and billable size of every planned request; download nothing."""
    if not os.environ.get("DATABENTO_API_KEY"):
        print("DATABENTO_API_KEY is not set", file=sys.stderr)
        return 2
    client = db.Historical()
    rng = client.metadata.get_dataset_range(DATASET)
    end = str(rng["end"])[:10]
    print(f"{DATASET}: {rng['start']} -> {rng['end']}; request window {START} -> {end}")
    total = 0.0
    for label, symbols, schema in PLAN:
        kw = dict(
            dataset=DATASET, symbols=symbols, schema=schema, start=START, end=end, stype_in="parent"
        )
        try:
            usd = client.metadata.get_cost(**kw)
            mb = client.metadata.get_billable_size(**kw) / 1e6
        except Exception as e:  # noqa: BLE001
            print(f"{label:58s} ERROR {type(e).__name__}: {str(e)[:120]}")
            continue
        total += usd
        print(f"{label:58s} {usd:8.2f} USD  {mb:9.1f} MB")
    print(f"{'total':58s} {total:8.2f} USD")
    return 0


if __name__ == "__main__":
    sys.exit(main())
