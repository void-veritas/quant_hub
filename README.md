# quant_hub

Crypto quant research hub — perpetual futures strategies (funding carry, momentum) with Hyperliquid and Binance as primary venues.

## Setup

Requires Python >= 3.13 and [uv](https://docs.astral.sh/uv/):

```sh
uv sync
```

The `quant_hub` package is installed editable, so it's importable from notebooks and tests directly.

## Layout

```
src/quant_hub/
  data/          ingestion, loaders, storage, validation
    connectors/  hyperliquid, binance, ccxt
    assets/      prices, funding, open interest
  alpha/         signals, features, momentum, funding signals
  indicators/    technical, volatility, market structure
  backtest/      metrics, blotter
    engines/     vectorized, event-driven
    strategies/  base, momentum, funding carry, hyperliquid btc
  portfolio/     construction, sizing, positions, constraints
  risk/          volatility, drawdown, leverage, exposures, liquidation
  costs/         fees, slippage, funding
  execution/     orders, paper broker
  attribution/   pnl decomposition, factor/funding attribution
  reporting/     charts, tables, reports
  utils/         config, dates, logging

data/            raw -> interim -> processed (contents gitignored)
notebooks/       exploratory work
research_memos/  written-up findings
reports/         generated figures and tables (gitignored)
config/          strategy/run configuration
tests/           pytest suite
```

## Data layer

Raw market data is stored as hive-partitioned parquet under
`data/raw/<dataset>/exchange=…/asset=…/year=…/data.parquet`. Conventions:
timestamps are UTC, bar timestamps are OPEN time, raw values are venue-native.
Assets are keyed by canonical id via the git-tracked `config/instruments.csv`
(rebuild: `uv run python -m quant_hub.data.ingestion instruments`).

### Backfilling

Backfills are manual, idempotent, and resume-safe: re-runs skip complete
months / already-stored rows, so interrupting and restarting is always fine.
With `--all`, Binance assets cross-listed on Hyperliquid are backfilled first
(the tradeable universe becomes usable before the Binance-only tail).

Full backfill, in recommended order:

```sh
uv run python -m quant_hub.data.ingestion instruments                        # refresh instrument table (fast)
uv run python -m quant_hub.data.ingestion ohlcv   --exchange binance --all   # 15m bars, bulk dumps (hours)
uv run python -m quant_hub.data.ingestion funding --exchange hyperliquid --all   # hourly funding (~30 min)
uv run python -m quant_hub.data.ingestion funding --exchange binance --all   # 8h/4h funding (~1 h)
uv run python -m quant_hub.data.ingestion ohlcv   --exchange hyperliquid --all   # HL bars, recent history only (fast)
uv run python -m quant_hub.data.ingestion oi      --all                      # 5-min OI snapshots (very long — run overnight, or per-asset as needed)
```

Targeted backfills and quality checks:

```sh
uv run python -m quant_hub.data.ingestion ohlcv   --exchange binance --assets BTC,ETH --start 2023-01-01
uv run python -m quant_hub.data.ingestion oi      --assets BTC --start 2024-01-01
uv run python -m quant_hub.data.ingestion gaps    --dataset ohlcv_15m        # missing-bar report (also: funding, open_interest)
```

Any backfill accepts `--proxy scrapingbee` to route requests through the
ScrapingBee IP pool (needs `SCRAPING_BEE_KEY` in `.env`; 1 credit/request).
Use it for Hyperliquid pulls — HL rate limits are per-IP and shared with the
web UI, so a direct backfill throttles a browser trading session on the same
connection. `oi` also takes `--concurrency N` for parallel downloads (pair
with the proxy; the current plan allows 100 concurrent streams):

```sh
uv run python -m quant_hub.data.ingestion funding --exchange hyperliquid --all --proxy scrapingbee
uv run python -m quant_hub.data.ingestion oi --all --proxy scrapingbee --concurrency 80
```

### Reading data (notebooks / research)

Everything goes through `quant_hub.data.loaders` — DuckDB under the hood,
pandas out. Assets are canonical ids (`BTC`, `ETH`, `PEPE` — never venue
symbols like `1000PEPEUSDT` or `kPEPE`).

```python
from quant_hub.data.loaders import load_ohlcv, load_funding, load_oi, load_instruments

bars    = load_ohlcv(["BTC", "ETH"], start="2024-01-01", end="2024-06-30")   # long format
closes  = load_ohlcv(["BTC", "ETH"], field="close")                          # ts x asset wide matrix
hl_fund = load_funding(["BTC"], exchange="hyperliquid")                      # hourly, venue-native
bn_fund = load_funding(["BTC"], exchange="binance")                          # 8h/4h, venue-native
oi      = load_oi(["BTC"], start="2024-01-01")                               # 5-min OI snapshots
inst    = load_instruments(exchange="binance", status="trading")             # instruments table
```

### Module reference (`src/quant_hub/data/`)

| Function | Purpose |
|---|---|
| `loaders.load_ohlcv(assets, start, end, exchange="binance", field=None)` | 15m bars; `field=` pivots to a ts × asset matrix |
| `loaders.load_funding(assets, start, end, exchange="hyperliquid")` | funding events at each venue's native cadence |
| `loaders.load_oi(assets, start, end)` | Binance 5-min open-interest snapshots |
| `instruments.load_instruments(exchange=None, status=None)` | the instruments table as a DataFrame |
| `instruments.symbol_map(exchange)` | canonical id → venue symbol dict |
| `storage.write_partition(df, dataset, exchange, asset)` | idempotent partition write (dedup on `ts`, atomic) |
| `storage.coverage(dataset)` | per-asset first/last timestamp and row counts |
| `validation.find_gaps(ts, freq)` | contiguous missing-timestamp ranges |
| `validation.check_bars(df)` | OHLCV sanity check (duplicates, high<low, non-positive prices) |

Notes: all timestamps UTC with bar OPEN time convention; the `funding` dataset's
columns differ per venue (Binance has `mark_price`, HL has `premium`); Binance OI
dumps exist since ~2021-12; HL deep candle history (S3 archive) not wired up yet.

## Tooling

- `uv run pytest` — tests
- `uv run ruff check .` — lint
- `uv run black .` — format (line length 100)
