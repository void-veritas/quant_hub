"""Commodities data domain: a separate layer next to the crypto one.

Why separate: paid data (Databento) must not be redistributed and lives in its own
bucket with its own IAM; the crypto `sync pull` must not drag commodity gigabytes;
the crypto code reads only `data/raw` and `data/external`, so the two cannot cross.

Layout (`quant_hub.commodities.storage`):

    data/commodities/raw/        as the source delivered it, plus source and licence
    data/commodities/reference/  contract specs, units, calendars, annual TC/RC, market lists
    data/commodities/curated/    built by code from raw: continuous series with the roll rule
                                 written down, carry, cash-3M, COMEX-LME spread in common units

Raw datasets keep the hive layout of the crypto layer
(`<dataset>/exchange=<x>/asset=<a>/year=<y>/data.parquet`, `ts` UTC) so the same
`quant_hub.data.storage` functions read and write them:

    ohlcv_1d, funding   HIP-3 builder perps on Hyperliquid (exchange=hyperliquid_hip3,
                        asset=<dex>-<NAME>, e.g. xyz-COPPER)
    futures_1d          daily bars per contract from Databento GLBX.MDP3 (exchange=cme,
                        asset=contract symbol such as HGZ6 or a spread HGZ6-HGH7)
    futures_cont_1d     continuous front-month series from Yahoo / Stooq (exchange=yahoo|stooq,
                        asset=root): cross-checks only, not tradeable returns
    cot_disagg          CFTC disaggregated Commitments of Traders (exchange=cftc, asset=code)
    databento/          the DBN files Databento delivered, each with a JSON sidecar
                        (request, cost, billable size, licence)

Principles: store by contract, not only the continuous series; positions and
fundamentals carry two dates (period and publication); units are kept source-native
in raw and normalised in curated; COMEX, LME and SHFE close at different times, so
cross-exchange spreads are built only on synchronised prices.

Commands: `python -m quant_hub.commodities.ingestion ...`, `... .sync push|pull`,
`... .health`, `... .migrate` (one-off move of HIP-3 out of the crypto bucket).
"""
