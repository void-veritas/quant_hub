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
  backtest/      metrics, multiple_testing, blotter
    engines/     vectorized, event-driven
    strategies/  base, momentum, funding carry, hyperliquid btc
  portfolio/     construction, sizing, positions, constraints
  risk/          volatility, covariance, factor_model, hedging,
                 exposures, leverage, drawdown, liquidation
  costs/         fees, slippage, funding
  execution/     orders, paper broker
  attribution/   factor attribution, performance, pnl decomposition
  reporting/     charts, tables, reports
  utils/         config, dates, logging, state_space

data/            raw -> interim -> processed (contents gitignored)
docs/            book summaries, toolkit_map.md, design notes
notebooks/       exploratory work
research_memos/  written-up findings
reports/         generated figures and tables (gitignored)
config/          strategy/run configuration
tests/           pytest suite
```

The research toolkit (`risk`, `portfolio`, `alpha`, `attribution`, `costs`,
`backtest`) implements the math from two Paleologo texts, mapped module-by-module
in [docs/toolkit_map.md](docs/toolkit_map.md). Worked examples are below.

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
dumps exist since ~2021-12.

### Hyperliquid L2 archive (`data/connectors/hl_archive.py`, dataset `hl_l2_hour`)

`hyperliquid-archive` S3 bucket (requester pays, AWS keys in env): 20-level L2
snapshots every ~0.55 s since 2023-04-15. `ingestion l2hour --assets A,B --hour 9`
summarises one hour per day per asset (first/median mid, spread in bps, depth
within 10/25/50 bps in base units) so a daily-rebalance strategy gets a true
decision-time price and spread without storing the raw book.

### Robot Wealth API (`data/connectors/robotwealth.py`, key in env `RW_PRO`)

`yolo_weights/factors/volatilities/historical(days)` for live and year-long
reference values of the YOLO strategy; `list_datasets()` + `download_dataset()`
for the bulk files (`coincodex.marketcap`, `binance.perps` funding/ohlcv-1h,
served from `cdn.data.robotwealth.com`); `binance_perps_funding(gte, lte)` for
funding newer than the Vision monthly dumps. Downloads go to `data/external/rw/`.

### Mirror on S3 (`data/sync.py`)

The raw and external layers are mirrored to `s3://yolo-research-ep` (us-east-1):

```
uv run python -m quant_hub.data.sync pull --bucket yolo-research-ep   # fresh container: minutes, not hours
uv run python -m quant_hub.data.sync push --bucket yolo-research-ep   # after every backfill
```

`push`/`pull` copy only files missing or differing in size; `--dataset` limits to
one dataset. Same AWS keys as the HL archive.

## Research toolkit

Panel tools for daily cross-sectional books (long frames `date, asset, ...`), moved
in from the YOLO research repo (M6):

| Module | Purpose |
|---|---|
| `universe.cap_universe / liquidity_universe` | point-in-time universes (yesterday's market-cap rank, trailing dollar volume) |
| `portfolio.neutral.rolling_betas / neutralise` | lagged betas to the EW basket; dollar / beta / hedge neutralisation |
| `portfolio.pit_construction.build` | vol target, mean-variance (Ledoit-Wolf / Marchenko-Pastur), HRP on point-in-time covariances |
| `attribution.panel_attribution` | PnL by weight source (with overlay), time-series and cross-sectional factor attribution |
| `backtest.validation` | walk-forward, purged k-fold, CPCV, variant table with Rademacher terms, regime table |
| `backtest.engines.daily_rebalance` | rsims-style daily rebalance engine with venue constraints |


The `risk`, `portfolio`, `alpha`, `attribution`, `costs`, and `backtest` packages
are a factor-model-centred quant toolkit (full module map:
[docs/toolkit_map.md](docs/toolkit_map.md)). Everything takes numpy/pandas and
returns the same; the risk machinery keys on a `FactorModel` object. Shape
conventions: `n` assets, `m` factors, `T` periods; wide return matrices are
`(T, n)`.

### End-to-end: a cross-sectional momentum book

Load data → build a signal → fit a risk model → construct and size the portfolio.
This runs as-is against the backfilled data:

```python
import numpy as np
from quant_hub.data.loaders import load_ohlcv
from quant_hub.alpha import momentum, signals
from quant_hub.risk import covariance
from quant_hub.risk.factor_model import FactorModel
from quant_hub.portfolio import construction

# 1. Wide close-price matrix -> daily log returns
universe = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX", "LINK", "LTC"]
closes = load_ohlcv(universe, start="2023-06-01", end="2024-12-31", field="close")
daily  = closes.resample("1D").last().dropna(how="any")
rets   = np.log(daily).diff().dropna()

# 2. Cross-sectional momentum at the last date, z-scored into an alpha
mom   = momentum.cross_sectional_momentum(daily, lookback=60, skip=5).iloc[-1]
alpha = signals.zscore(mom)

# 3. A 2-factor statistical risk model (PPCA) from the return history
loadings, factor_cov, idio = covariance.ppca(rets.to_numpy(), n_factors=2)
fm = FactorModel(loadings, factor_cov, np.full(rets.shape[1], idio))

# 4. Mean-variance weights at a 10%-vol budget, then inspect the risk
cov_r = fm.asset_covariance()
w = construction.mean_variance_weights(alpha.to_numpy(), cov_r, vol_target=0.10)
print({a: round(float(x), 3) for a, x in zip(daily.columns, w)})
print("pct idio variance: %.2f" % fm.pct_idio_variance(w))   # ~0.68
```

### Building blocks

Short, self-contained fragments for each area (the objects `daily`, `rets`,
`alpha`, `cov_r`, `loadings` come from the example above).

**Volatility & risk models** (`quant_hub.risk`)

```python
from quant_hub.risk import volatility, covariance

btc_ret = rets["BTC"]
vol = volatility.ewma_volatility(btc_ret, halflife=20)          # filtered EWMA vol
volatility.annualize_volatility(vol.iloc[-1], 252)             # daily -> annual
garch = volatility.GARCH11().fit(btc_ret); garch.forecast(10)  # conditional-var forecast

shrunk, intensity = covariance.ledoit_wolf_shrinkage(rets.to_numpy())  # well-conditioned cov
fm.factor_mimicking_portfolios()          # min-idio-risk unit-exposure portfolios (n x m)
```

**Signals & alpha** (`quant_hub.alpha`)

```python
from quant_hub.alpha import signals

fwd = rets.shift(-1).iloc[-2]                                   # next-period returns
ic  = signals.information_coefficient(alpha, fwd)              # cross-sectional IC
signals.ic_to_sharpe(ic, breadth=len(alpha))                  # fundamental law: IC * sqrt(n)
signals.orthogonalize(alpha.to_numpy(), loadings)             # strip factor exposure from a signal
```

**Portfolio construction & sizing** (`quant_hub.portfolio`)

```python
from quant_hub.portfolio import construction, sizing

construction.factor_neutral_weights(alpha.to_numpy(), loadings)   # proportional, B'w = 0
construction.robust_mvo_weights(alpha.to_numpy(), cov_r, 1e-4, vol_target=0.10)  # alpha-uncertainty
sizing.proportional_size(alpha.to_numpy(), gross=1.0)            # APM's robust default
sizing.kelly_fraction(mu=0.001, sigma=0.02)                     # growth-optimal capital fraction
```

**Evaluation & backtest hygiene** (`quant_hub.backtest`)

```python
from quant_hub.backtest import metrics, multiple_testing

strat_ret = (rets * (w / np.abs(w).sum())).sum(axis=1).to_numpy()
eq = metrics.equity_from_returns(strat_ret)
sr = metrics.sharpe_ratio(strat_ret, periods_per_year=252)
metrics.sharpe_confidence_interval(sr, n_obs=len(strat_ret))    # Lo (2002) — SR error bars
metrics.max_drawdown(eq)
# haircut an in-sample Sharpe for how many strategies you searched
multiple_testing.rademacher_haircut_sharpe(sr, np.column_stack([strat_ret] * 20))
# probabilistic-forecast scoring (also used by the Polymarket work)
metrics.brier_score(prob=[0.6, 0.3], outcome=[1, 0])
```

**Funding — the perp-native carry** (`quant_hub.alpha.funding_signals`)

```python
from quant_hub.data.loaders import load_funding
from quant_hub.alpha import funding_signals as fs

f = load_funding(["BTC"], exchange="hyperliquid", start="2024-01-01")
fs.funding_carry(f["rate"], interval_hours=1)          # annualized carry a short position earns
fs.cross_venue_funding_spread(f["rate"], 1, f["rate"], 8)   # HL(1h) vs Binance(8h) spread
```

**Attribution & risk reporting** (`quant_hub.attribution`, `quant_hub.risk`)

```python
from quant_hub.risk import exposures
from quant_hub.attribution import performance

exposures.risk_decomposition(fm, w)                    # per-factor %Var / $Vol / MCFR table
performance.effective_breadth(w)                       # 1 / Herfindahl of the book
performance.selection_sizing_timing(weights_ts, idio_returns_ts)   # skill decomposition (T x n)
```

## Tooling

- `uv run pytest` — tests (`test_data_layer.py`, `test_toolkit.py`)
- `uv run ruff check .` — lint (PEP 8 + numpy-style docstrings)
- `uv run black .` — format (line length 100)
