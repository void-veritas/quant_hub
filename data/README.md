# data/: what is on disk and where it comes from

Everything under `data/` is gitignored; it is rebuilt from the S3 mirror
(`uv run python -m quant_hub.data.sync pull --bucket yolo-research-ep`, minutes) or from
the venues (`python -m quant_hub.data.ingestion ...`, hours). After every backfill: `sync push`.

## Raw datasets (`data/raw/<dataset>/exchange=<x>/asset=<A>/year=<Y>/data.parquet`)

Coverage on 2026-10-12 (`quant_hub.data.storage.coverage(dataset)` prints the live table):

| dataset | exchange | assets | first | last | source | used for |
|---|---|---|---|---|---|---|
| `ohlcv_1d` | hyperliquid | 232 | 2020-08-19 (see quirk 1) | 2026-10-08 | HL API `candleSnapshot` | cap-universe extension past 2026-01-22, delisting coverage |
| `ohlcv_1d` | binance | 100 | 2023-01-01 | 2026-10-09 | Vision daily/monthly dumps, carries `taker_buy_volume` | volume factor (v1.4) |
| `ohlcv_15m` | binance | 16 | 2020-01-01 | 2026-10-08 | Vision dumps | 08:00 decision prices for the Binance replications |
| `ohlcv_15m` | hyperliquid | 12 | 2026-08-17 | 2026-10-08 | HL API (recent history only) | spot checks |
| `funding` | hyperliquid | 94 | 2023-05-12 | 2026-10-09 | HL API `fundingHistory` (hourly) | HL carry variants |
| `funding` | binance | 100 | 2020-01-01 | 2026-09-30 | Vision monthly dumps; the current-month tail needs `--source both` (fapi REST, geo-blocked here) or the RW API on the VPS | production carry (RW uses Binance funding) |
| `hl_l2_hour` | hyperliquid | 94 | 2023-04-15 | 2026-10-08 | `hyperliquid-archive` S3 (requester pays) | 09:00 decision mid, spread, depth |
| `open_interest`, `liquidations`, `exchanges`, `hyperliquid` | | | | | | empty placeholders |

External layer (`data/external`, synced like raw): `rw/` (RW bulk files; `coincodex_marketcap.feather` is
frozen at 2026-01-22, RW no longer updates it), `coincodex/` (its continuation from CoinCodex's public
endpoints: `tickers.csv`, `marketcap_daily.parquet`; `python -m quant_hub.data.marketcap cx-snapshot` daily),
`coingecko/` (one year of CoinGecko caps for 682 coins, cross-check only).
`quant_hub.data.marketcap.load_marketcap()` is the one loader for the universe.

Checks: `python -m quant_hub.data.health` (gaps, duplicates, sanity, staleness; exit 1 if flagged;
`--cross-venue` compares HL and Binance daily closes per asset). Sync is content-based and protects
newer local files; `pull --merge` unions a partition both machines appended to; see the repo README.

Rules that keep two machines consistent (one bucket, several writers):
- `pull` before you ingest, `push` after. Sync copies whole partition files, not rows: two machines
  that both appended to the same `asset/year` file without a pull in between hold different files,
  and `pull` then keeps the newer local one and reports it. Resolve with `pull --merge` (union on
  `ts`, local rows win) and `push`.
- Re-running any ingestion is safe: writes merge on `ts` within a partition, new rows win. A bar is
  stored only when its interval is closed; the last two HL bars are re-fetched on every run.
- Health before push: `python -m quant_hub.data.health && python -m quant_hub.data.sync push ...`.

Quirks worth knowing:
1. HL daily candles start in 2020 for the majors although Hyperliquid launched in 2023: the API serves
   pre-launch history seeded from index prices. Use them only to extend market-cap ranks, never as
   tradeable returns. Tradeable HL history starts with the L2 archive (2023-04-15).
2. Binance funding on Vision lags by weeks (last 2026-09-30 here). The tail comes from the Robot Wealth
   API (`robotwealth.binance_perps_funding`) or from `fapi.binance.com`, which is geo-blocked (451) from
   some containers. Production (`prod/analytics/produce_weights.py` in yolo-microstructure) refuses to
   write weights when this is stale.
3. Funding sign convention in raw data is venue-native; the YOLO panel converts to "paid to the long".
4. `hl_l2_hour` rows are stamped with the archive file's nominal hour; ~2% of venue days are missing
   and are filled from Binance at the same hour by `yolo.panel.hl_decision_prices` (flag `filled`).

## Commodities domain (`data/commodities/`, bucket `s3://yolo-commodities-ep`)

Separate from the crypto layers above (paid data, own IAM, own sync): `raw/`, `reference/`,
`curated/`, mirrored with `python -m quant_hub.commodities.sync pull|push`, checked with
`python -m quant_hub.commodities.health`. Moved here on 2026-10-21 from the crypto bucket
(`quant_hub.commodities.migrate`): HIP-3 perps and the TC/RC benchmark file.

| layer/dataset | exchange | assets | source | notes |
|---|---|---|---|---|
| `raw/ohlcv_1d`, `raw/funding` | hyperliquid_hip3 | commodity markets of the builder dexes `xyz` (live) and `flx` (delisted 2026-06) | HL API, `ingestion hip3` | asset = `<dex>-<NAME>` (`xyz-COPPER`); the L2 archive does not cover these markets |
| `raw/futures_1d` | cme | one asset per contract (`HGZ6`) or calendar spread (`HGZ6-HGH7`) | Databento GLBX.MDP3 `ohlcv-1d`, parent symbology (`HG.FUT`), `ingestion databento` | bars per UTC day, close = last trade (settlement and OI: `statistics` schema, not bought yet); bars only on days with trades |
| `raw/databento/` | | | the DBN files as delivered + JSON sidecar (request, estimate, purchase time, licence) | not for redistribution |
| `raw/futures_cont_1d` | yahoo, stooq | roots (`HG`) | `ingestion futures` | continuous front month: roll jumps, cross-checks only |
| `raw/cot_disagg` | cftc | CFTC market code (copper `085692`) | `ingestion cot` | `ts` = report Tuesday; published the Friday after |
| `reference/hip3_markets.csv` | | | written by `ingestion hip3` | delisted flag, lot size, leverage (health reads it) |
| `reference/tcrc/benchmarks.csv` | | | annual TC/RC benchmark 2010–2026 (hand-collected, `verified` column) | |

## External files (`data/external/rw/`, from the Robot Wealth API and CDN)

| file | content | used for |
|---|---|---|
| `coincodex_marketcap.feather` | daily market caps to 2026-01-22 (`Ticker, Date, Price, Volume, MarketCapUSD`) | cap universe (top-n by yesterday's cap) |
| `binance_perps_funding.feather` | RW's bulk Binance funding (213 settlements missing Jul–Oct 2026) | cross-check only |
| `yolo_historical_365.parquet`, `yolo_weights_latest.parquet`, `yolo_factors_latest.parquet`, `yolo_volatilities_latest.parquet` | RW production values | replication tests (1e-9) |
| `binance_funding_live_*.parquet` | RW API funding tails pulled on given dates | carry_factor investigation |

## Config (`config/`, git-tracked)
`instruments.csv`: canonical asset id ↔ venue symbols and status (rebuild: `ingestion instruments`).
`hl_lot_sizes.json`: HL `szDecimals` per coin as a position step (rebuild from `fetch_meta()`).
