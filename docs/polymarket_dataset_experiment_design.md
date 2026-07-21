# Dataset and Experiment Design

## 1. Purpose

The dataset must let us answer three separate questions:

1. **Forecast:** what is the probability that the currently winning side remains the terminal winner?
2. **Trade:** does that probability exceed the all-in executable market cost?
3. **Capacity:** how much can be executed before spread, book walking and latency eliminate the edge?

The dataset is not simply a collection of five-minute BTC candles. The critical object is a synchronised record of Chainlink, external BTC prices and the Polymarket order book during each market.

## 2. Storage structure

Keep the physical structure deliberately simple:

```text
data/
  raw/
    chainlink/
    polymarket/
    external_btc/
    market_metadata/
    executions/

  processed/
    markets/
    decision_snapshots/
    executions/
```

`raw` is append-only source data. `processed` contains reproducible tables built from raw data. Feature construction happens in scripts; no additional storage layers are required initially.

## 3. Raw datasets

### 3.1 Chainlink BTC/USD

Record every available BTC/USD update used by Polymarket, including:

- price value;
- symbol/feed identifier;
- source timestamp;
- local receipt timestamp;
- message sequence or unique identifier when available;
- freshness and data-quality status.

Chainlink defines:

- the opening reference price `S_0`;
- the live distance from the reference `S_t - S_0`;
- the terminal price `S_T`;
- the final market label.

External exchange prices must never be substituted for a missing Chainlink settlement observation.

### 3.2 Polymarket market metadata

Store one record per five-minute market:

- event, market and condition IDs;
- Up and Down token IDs;
- market slug and title;
- scheduled opening and closing timestamps;
- actual resolution timestamp and outcome;
- fee parameters;
- tick size and other CLOB parameters;
- market status and any suspension or resolution flags.

Fee parameters are queried and stored for every market instead of being assumed permanent.

### 3.3 Polymarket CLOB

Preferred collection:

- initial full order-book snapshot for both tokens;
- every subsequent book delta;
- every trade event;
- tick-size or market-status changes;
- source/message timestamp and local receipt timestamp;
- sequence identifiers where available.

The raw stream should make it possible to reconstruct the book at any historical millisecond.

If full depth becomes impractical, the minimum acceptable depth is enough to simulate the complete proposed size grid up to at least $1,000. Best bid/ask alone is insufficient for capacity analysis.

Periodic full snapshots should be stored as recovery points so a missing delta does not corrupt the rest of a market.

### 3.4 External BTC markets

The minimum external set is:

- one liquid BTC spot market;
- one liquid BTC perpetual market.

A practical MVP could use Binance BTCUSDT spot plus Binance, Bybit or Hyperliquid perpetuals. Coinbase BTCUSD can be added to distinguish a genuine USD move from USDT-specific basis.

Record:

- trades or best bid/ask updates;
- source timestamp and local receipt timestamp;
- bid, ask, mid and spread;
- trade price, size and aggressor where available;
- optional shallow order-book depth and imbalance.

Updates around 100–250 ms are sufficient for the initial forecasting model. Raw event capture is preferable to fixed candles because jumps and feed divergence occur inside a one-second interval.

### 3.5 Decisions and executions

Every evaluated opportunity is recorded, including no-trades:

- market and checkpoint;
- model and feature-set versions;
- feature cutoff timestamp;
- point probability and conservative bounds;
- proposed side and size;
- observed book and simulated all-in cost;
- accept/reject decision and reason codes;
- order submission, acknowledgement and match timestamps;
- filled size, average price and fee;
- final outcome and realised P&L.

Recording rejected opportunities is necessary to detect selection bias and strategy drift.

## 4. Timestamp standard

Each live message should contain at least:

```text
event_timestamp
received_timestamp
```

where:

- `event_timestamp` is supplied by the source;
- `received_timestamp` is assigned by the collector when the message arrives.

Use UTC internally and retain millisecond precision. The collector clock must be synchronised and monitored. Nanosecond precision is unnecessary, but one-second candles are inadequate for reliable execution replay.

The processed model may operate on a one-second grid, while execution replay continues to use raw event timestamps.

## 5. Market-level processed table

Create one row per completed five-minute market:

```text
market_id
condition_id
up_token_id
down_token_id
scheduled_start
scheduled_end
resolution_time
chainlink_open_price
chainlink_terminal_price
terminal_return
winning_outcome
fee_parameters
data_quality_flags
```

The canonical Up label is:

```text
y_up = 1 if S_T >= S_0 else 0
```

The equality rule must be implemented exactly.

## 6. Decision-snapshot processed table

The main modelling table contains one row per market per predefined checkpoint.

Initial checkpoints:

- 60 seconds remaining;
- 30 seconds remaining.

Raw data may be collected continuously, but adding dozens of modelling checkpoints initially would create correlated observations and multiple-testing risk.

### 6.1 State and labels

```text
market_id
checkpoint
feature_cutoff_time
seconds_remaining
S_0
S_t
S_T
current_side
terminal_winner
y_persist
remaining_return
required_reversal_return
```

where:

```text
remaining_return = log(S_T / S_t)
y_persist = 1 if current_side equals terminal_winner else 0
```

### 6.2 Structural features

- signed log-distance `log(S_t / S_0)`;
- absolute distance from the reference;
- forecast volatility over the exact remaining seconds;
- volatility-scaled distance `z`;
- current-side market price and all-in taker cost;
- spread and qualifying ask depth.

### 6.3 Path and regime features

- returns over 5, 15, 30 and 60 seconds;
- realised volatility over short and medium windows;
- volatility acceleration;
- jump size and time since jump;
- gradual move versus discrete jump;
- distance from recent high and low;
- time-of-day, weekday/weekend and scheduled-event flags.

### 6.4 Cross-venue and microstructure features

- Chainlink divergence from spot and perpetual markets;
- spot/perpetual basis;
- external spread, order-book imbalance and aggressive flow;
- Polymarket bid/ask, microprice, depth imbalance and recent trade direction;
- source-feed ages and collector latency.

## 7. Continuous historical context

Although the contract lasts five minutes, volatility estimation requires data preceding the market. External BTC collection therefore runs continuously.

Initial trailing windows may include:

- 5, 15, 30 and 60 seconds;
- 2, 5, 15 and 60 minutes.

This distinguishes a large gap in a quiet market from an ordinary gap during a volatility shock.

The path must also be retained. The same terminal distance can result from a gradual trend, one abrupt jump or violent oscillation around the reference, with materially different reversal risks.

## 8. Execution and capacity labels

At each checkpoint, reconstruct hypothetical execution for the following cash sizes:

```text
$5, $10, $25, $50, $100, $250, $500, $1,000
```

For each size record:

- executable shares;
- average displayed price;
- fee;
- average all-in cost per share;
- percentage filled;
- spread and book walking;
- edge before and after execution costs;
- realised P&L if held to settlement.

Configured taker delay must be applied before reading the simulated execution book. Capacity is the largest size for which the conservative edge remains above the required buffer.

## 9. Sample-size expectations

There are 288 five-minute markets per day:

| Collection period | Complete markets |
| --- | ---: |
| 14 days | 4,032 |
| 30 days | 8,640 |
| 60 days | 17,280 |
| 90 days | 25,920 |

Interpretation:

- **Two weeks:** data-pipeline validation and preliminary baselines.
- **One month:** initial modelling and calibration diagnostics.
- **Two to three months:** initial evidence across more than one volatility state.
- **Longer:** necessary for credible inference about genuinely rare 1% tail events.

If only 5% of markets pass the strategy filter, one month produces approximately 430 accepted opportunities. This is useful but insufficient to claim that a 1% reversal tail has been characterised precisely.

The effective sample is the number of independent markets, not the number of messages or snapshots.

## 10. Experiment sequence

### Stage 1: data-quality and accounting test

- Run the recorder continuously.
- Reconcile Chainlink opening and terminal prices with Polymarket outcomes.
- Reconstruct user trades and platform fees exactly.
- Verify that book snapshots and deltas remain consistent.
- Measure missing-message, feed-age and clock-error rates.

No model result is trusted until this stage passes.

### Stage 2: model-free diagnostic

Create an empirical reversal table by:

```text
checkpoint
x volatility-scaled distance
x volatility regime
x path type
```

For each cell compare:

- observed persistence frequency;
- executable market break-even probability;
- simple diffusion or empirical-volatility probability.

If the market already dominates the simple models and shows no systematic mispricing, more complex ML is unlikely to rescue the strategy.

### Stage 3: probabilistic models

Compare, in order:

1. executable market probability;
2. diffusion/local-volatility baseline;
3. empirical state table;
4. regularised logistic regression;
5. gradient-boosted probability and quantile models;
6. conformalised adverse-return bounds.

Complexity is added only when it produces stable out-of-sample improvement.

### Stage 4: chronological validation

Use complete chronological folds:

```text
training -> conformal calibration -> untouched test
```

Then roll the structure forward. All observations from a market stay in the same fold. Purge boundaries when trailing feature windows would leak information.

Never use a random snapshot split.

### Stage 5: freeze the policy

Before evaluating the final test period, freeze:

- decision checkpoints;
- features and model version;
- conformal method and calibration window;
- minimum edge and uncertainty buffer;
- proposed size grid;
- fee and execution-delay treatment;
- exclusions and kill switches.

Any later modification is evaluated on new future data.

### Stage 6: execution replay

For every historical checkpoint:

1. freeze data at its actual receipt time;
2. generate probability and conformal outputs;
3. apply the configured delay;
4. reconstruct the new ask book;
5. walk the book for each size;
6. apply the exact fee;
7. reject the trade if the remaining edge is insufficient;
8. settle filled shares against the Chainlink outcome.

### Stage 7: shadow and tiny live trading

Shadow trading validates the complete live decision path without capital. Several hundred independent accepted markets should be observed before a tiny live pilot.

The live pilot validates:

- actual delay and fill behaviour;
- fee accounting;
- difference between simulated and realised execution;
- operational failures;
- whether discretionary trades remain excluded.

Live trading is not used to compensate for an inconclusive historical test.

## 11. Evaluation

### Forecast quality

- Brier score and log loss;
- calibration/reliability by probability, checkpoint and regime;
- adverse-tail coverage and conformal interval width;
- improvement over market and simple-model baselines.

### Economic quality

- net P&L after every cost;
- return on dollars deployed and on capital tied up;
- hit rate together with average win and loss;
- maximum drawdown and expected shortfall;
- loss clustering by hour, day and volatility event;
- stability across thresholds and chronological folds.

### Execution and capacity

- fill rate and latency;
- expected versus realised slippage;
- edge captured versus edge predicted;
- P&L and confidence interval at every proposed size;
- capacity as a rolling edge-decay curve.

Use day-level or other appropriate block resampling when constructing uncertainty intervals. Do not treat adjacent markets as fully independent during volatility events.

## 12. Minimum credible experiment

A credible initial result requires:

- at least 30 consecutive days of correctly synchronised data, preferably 60–90;
- every five-minute market retained, including every no-trade;
- fixed 60- and 30-second checkpoints;
- exact Chainlink labels;
- historical Polymarket L2 reconstruction;
- executable market, volatility and empirical-state baselines;
- realistic delay, fees, spread and book walking;
- one completely untouched chronological test period;
- several hundred independent accepted out-of-sample signals before tiny live trading;
- continued collection long enough to observe rare reversal clusters.

The most difficult and irreplaceable component is historical Polymarket L2 aligned with Chainlink receipt times. Outcomes and public trades can often be backfilled; exact book state and our real receipt latency generally cannot. Starting the recorder is therefore the first implementation priority.

## 13. What is deliberately excluded from the MVP

- multiple crypto assets;
- news and social sentiment;
- on-chain analytics;
- options surfaces;
- deep neural networks;
- adaptive every-second trading;
- maker quoting;
- early-exit logic;
- leveraged or martingale sizing.

Each may become a separate experiment later. None is necessary to answer whether the core late-window persistence strategy has an edge.
