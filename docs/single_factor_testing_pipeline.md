# Single-Factor Research: Testing, Attribution, and Dynamic Trading

This is the smallest useful research loop: one instrument, one signal, one forward
return, and one position rule. Start here before moving to a cross-sectional factor
model.

## The 30-second version

```text
factor known at t
      -> forecast return from t to t+1
      -> choose position
      -> observe realised return
      -> attribute forecast, P&L, and costs separately
      -> update factor-health estimate using data now available
      -> choose the next position
```

Keep three questions separate:

| Question | Correct tool |
|---|---|
| What drove the model's forecast? | Coefficients or SHAP values |
| What drove realised P&L? | Factor/idio/cost accounting |
| Should exposure change now? | A causal rolling or state-space estimate of factor efficacy |

SHAP values do **not** allocate realised P&L. They explain a model prediction.

## 1. Freeze the first experiment

Example:

- instrument: Binance BTC perpetual;
- decision frequency: daily;
- signal: 20-day momentum, excluding the most recent day;
- target: next-day BTC return;
- hypothesis: higher momentum predicts a higher next-day return;
- initial rule: long when momentum is positive, short when negative;
- cost assumption: fixed basis points per unit of turnover.

Write these choices down before inspecting performance. Record every later variation as
a separate experiment so that weak ideas are not rescued by silently changing them.

## 2. Build one point-in-time table

One row should contain only information known when the position was chosen:

| Column | Meaning |
|---|---|
| `decision_ts` | Time the decision could actually be made |
| `factor` | Signal known at `decision_ts` |
| `forward_return` | Return earned after that decision |
| `forecast` | Model prediction made at that time |
| `position` | Position selected from the forecast |
| `turnover` | Absolute change in position |
| `gross_pnl` | `position * forward_return` |
| `cost` | Fees, spread, funding, and impact |
| `net_pnl` | `gross_pnl - cost` |
| `split` | Development, validation, or locked test |

Manually inspect several adjacent rows before running statistics. For each row, ask:
"Could I really have known this factor value before earning this return?"

## 3. Attribute the forecast

For the first linear model,

```text
forecast_t = intercept + theta * factor_t
```

the factor's contribution relative to a baseline is simply

```text
factor_contribution_t = theta * (factor_t - baseline_factor)
```

For a centred factor, the baseline is zero. This is also what one-factor linear SHAP
reduces to. Therefore a SHAP dependency would add ceremony but no information to the
first experiment.

SHAP becomes useful when the forecasting model is nonlinear or contains several
inputs, for example momentum, funding, open interest, volatility, and their
interactions. Then calculate SHAP values with these rules:

1. Fit the model on development data only.
2. Use development observations as the SHAP background distribution.
3. Explain validation/test predictions out of sample.
4. Check the additivity identity:
   `prediction = baseline prediction + sum(SHAP values)`.
5. Plot signed SHAP values through time as well as mean absolute SHAP importance.
6. Treat SHAP as model explanation, not causality and not realised-PnL attribution.

## 4. Attribute realised performance

If the instrument return is described by a risk-factor model,

```text
return_t = beta_t * risk_factor_return_t + idiosyncratic_return_t
```

then the strategy's realised P&L is

```text
gross P&L  = position * return
factor P&L = position * beta * risk-factor return
idio P&L   = position * idiosyncratic return
net P&L    = factor P&L + idio P&L - costs
```

This is an accounting identity. It answers whether performance came from an intended
alpha or from an incidental market/style exposure. Report confidence intervals because
estimated factor returns and betas introduce attribution error; zero inside the interval
means the claimed contribution is not distinguishable from estimation noise.

The repository already contains the multi-asset implementation:

- `attribution.factor_attribution.factor_idio_pnl` for factor versus idio P&L;
- `attribution.factor_attribution.attribution_confidence_interval` for uncertainty;
- `attribution.factor_attribution.maximal_attribution` for correlated factors;
- `risk.exposures.risk_decomposition` for exposure and risk contribution;
- `attribution.performance.selection_sizing_timing` for counterfactual attribution.

For a single instrument, selection and cross-sectional sizing attribution are not yet
meaningful: there is nothing to select among. At this stage report forecast contribution,
gross P&L, costs, and exposure to a chosen risk factor such as the crypto market. Add
selection/sizing attribution only when the experiment expands to several instruments.

## 5. Measure changing factor efficacy

Do not decide that a factor is "working" merely because its cumulative P&L recently
rose. Track the relationship it was designed to exploit.

For a factor observed at `t` and a return realised at `t+1`, define its payoff as:

```text
factor_payoff_(t+1) = standardised_factor_t * return_(t+1)
```

Monitor, using past data only:

- rolling or EWMA mean factor payoff;
- rolling regression slope and its standard error;
- directional hit rate;
- turnover and net payoff after costs;
- stability by volatility/trend/funding regime;
- concentration: whether a few dates explain most of the result.

A rolling estimate is the simple starting point. A state-space version follows the
books' Kalman-filter treatment:

```text
return_(t+1) = beta_t * factor_t + observation noise
beta_t       = beta_(t-1) + state noise
```

Here `beta_t` is the latent, changing effectiveness of the factor. The state-noise
variance controls how quickly the estimate can change; the observation-noise variance
controls how much it trusts each new return. `utils.state_space.KalmanFilter` provides
the filtering machinery, although a dynamic-regression wrapper still needs to be added.

## 6. Turn factor health into exposure

At decision time `t`:

1. Update the efficacy estimate only with returns realised through `t`.
2. Form `expected_return = estimated_beta * current_factor`.
3. Shrink the estimate for uncertainty.
4. Divide by forecast risk and enforce position/turnover limits.
5. Apply the resulting position to the return after `t`.

A conservative positive-factor estimate is:

```text
beta_effective = max(0, beta_estimate - confidence_multiplier * beta_standard_error)
expected_return = beta_effective * current_factor
```

This automatically reduces exposure when the estimated edge weakens or becomes
uncertain. It is preferable to repeatedly switching the factor on and off after short
losing streaks. Predefine the estimation window/half-life, confidence multiplier,
position cap, turnover cap, and minimum holding/cooldown rules; otherwise the dynamic
overlay becomes another overfit strategy.

## 7. The report card

Every run should produce one compact report:

- hypothesis and exact factor definition;
- development, validation, and untouched test dates;
- coefficient, HAC standard error, confidence interval, and sample size;
- out-of-sample forecast contribution (SHAP only when useful);
- factor P&L, idio P&L, attribution uncertainty, costs, and net P&L;
- Sharpe, drawdown, turnover, hit rate, and yearly results;
- rolling factor-efficacy estimate and the exposure it produced;
- complete list of tried variations.

The governing principle from both books is simple: use the least complicated model
that can test the idea, and carry estimation error into attribution and sizing rather
than treating estimates as truth.

## Book-to-code map

- *The Elements of Quantitative Investing*, Ch. 2: EWMA/Kalman updating.
- *The Elements of Quantitative Investing*, Ch. 8: multiple-testing control.
- *The Elements of Quantitative Investing*, Ch. 10/13: uncertainty-aware sizing.
- *The Elements of Quantitative Investing*, Ch. 14: factor/idio attribution with errors.
- *Advanced Portfolio Management*, Ch. 8: factor attribution and
  selection/sizing/timing counterfactuals.
- See `docs/toolkit_map.md` for the corresponding repository modules.
