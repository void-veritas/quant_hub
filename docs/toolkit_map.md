# Quant Toolkit — Module Map

Where each result from the two Paleologo summaries lives in `src/quant_hub/`.
`EQI` = *Elements of Quantitative Investing* notes; `APM` = *Advanced Portfolio
Management* notes. The math is asset-class-agnostic (factor models, vol, Kelly,
MVO all apply to crypto perps); equity-only descriptors (country/industry/13f)
are noted as such and kept thin.

For the practical workflow from a single-factor hypothesis through forecast
explanation, realised-PnL attribution, and dynamic exposure, see
[Single-Factor Research: Testing, Attribution, and Dynamic Trading](single_factor_testing_pipeline.md).

## Foundation (Tier 1) — everything builds on these

| Module | Contents | Source |
|---|---|---|
| `utils/state_space.py` *(new)* | `KalmanFilter` (predict/update/steady-state), `muth_gain` | EQI Ch2 appx |
| `risk/volatility.py` | EWMA var, half-life ↔ decay, `GARCH11`, realized variance, Roll spread, Harvey–Shephard filter, annualize | EQI Ch2 |
| `risk/covariance.py` *(new)* | sample/EWMA covariance, Ledoit–Wolf shrinkage, MP/BBP thresholds, eigenvalue denoising, PCA & PPCA factor models, Procrustes rotation | EQI Ch6–7 |
| `risk/factor_model.py` *(new)* | `FactorModel` (Ω_r=BΩ_fBᵀ+Ω_ε, exposures, %idio var, MCFR, FMPs), cross-sectional WLS regression, factor-cov estimation w/ noise subtraction, winsorization | EQI Ch4/6, APM Ch11 |
| `backtest/metrics.py` | Sharpe/IR (+Lo standard error, CI), t-stat, Cantelli bound, drawdown, Brier/log-loss/calibration, QLIKE/MSE vol-forecast loss | EQI Ch3/5, APM Ch6 |

## Construction & sizing (Tier 2)

| Module | Contents | Source |
|---|---|---|
| `alpha/signals.py` | z-score, rank, winsorize, IC, `ic_to_sharpe`/`ic_to_information_ratio` (fundamental law), orthogonalize-against-factors, combine signals | EQI Ch9, APM Ch6/8 |
| `portfolio/construction.py` | MVO (`w*=(σ/√αᵀΩ⁻¹α)Ω⁻¹α`), vol/correlation form `v*∝C⁻¹s`, min-variance, robust MVO (cov inflation), factor-neutral proportional weights | EQI Ch9–10, APM Ch11 |
| `portfolio/sizing.py` | proportional / risk-parity / MV / shrunk-MV sizing, vol targeting, Kelly (`μ/(μ²+σ²)`, binary, fractional, growth rate, uncertainty-adjusted) | EQI Ch13, APM Ch6 |

## Risk management & attribution (Tier 3)

| Module | Contents | Source |
|---|---|---|
| `risk/exposures.py` | risk-decomposition report (%Var/$Vol/MCFR), portfolio beta, dollar-beta arithmetic | APM Ch3/7 |
| `risk/leverage.py` | leverage floor/ceiling & feasibility band, market variance/GMV share, single-stock cap (CPR), single-factor cap | APM Ch7/10 |
| `risk/drawdown.py` | drawdown series, high-watermark, Grossman–Zhou fraction, stop-loss efficiency | EQI Ch13, APM Ch9 |
| `risk/hedging.py` *(new)* | min-var hedge ratio (−β), shrunk hedge ratio, factor hedge via FMPs, internal/external hedge | EQI Ch12, APM Ch11 |
| `attribution/factor_attribution.py` | factor vs idio PnL, attribution confidence intervals, maximal attribution | EQI Ch14 |
| `attribution/performance.py` | selection/sizing/timing (XSE/XSTSE counterfactuals + EQI Thm 14.1 identity), effective breadth, IR-from-hit-rate | EQI Ch14, APM Ch8 |
| `costs/slippage.py` | square-root / Almgren / Obizhaeva–Wang / Gatheral impact, optimal event-trade size | EQI Ch11, APM Ch8 |
| `backtest/multiple_testing.py` *(new)* | Rademacher complexity + RAS haircut, Massart bound | EQI Ch8 |

## Crypto alpha (Tier 4)

| Module | Contents |
|---|---|
| `alpha/momentum.py` | cross-sectional & time-series momentum (Novy-Marx term structure), short-term reversal |
| `alpha/funding_signals.py` | funding carry, funding momentum, cross-venue funding spread (the perp-native alphas) |
| `alpha/features.py` | feature transforms: log/rank/arctan, normalize-by, term-structure, change-vs-average |

## Deliberately deferred / equity-only

Multi-period impact MPC (EQI Ch11 finite-horizon), currency rebasing & model
linking (EQI Ch6 ★), DCC/STVU dynamic factor-cov updating, 13f-crowding and
borrow-rate factors — implement if a strategy needs them.

## Added after the YOLO work (M5–M7), not from the Paleologo summaries

| Module | Contents |
|---|---|
| `universe.py` | point-in-time market-cap and liquidity universes (yesterday's rank, smoothing, minimum history, extension by price ratios) |
| `portfolio/neutral.py` | lagged rolling betas to the equal-weight basket; dollar, beta and hedge neutralisation |
| `portfolio/pit_construction.py` | point-in-time covariances (sample, Ledoit–Wolf, Marchenko–Pastur), vol targeting, MV and HRP weights |
| `backtest/engines/daily_rebalance.py` | rsims-style daily engine: funding, commissions, no-trade buffer, margin call, delisting, venue lots and minimums |
| `backtest/validation.py` | walk-forward, purged k-fold, CPCV paths, variant comparison with Rademacher terms, regime tables |
| `attribution/panel_attribution.py` | PnL by weight source with overlay, time-series and cross-sectional factor attribution on long panels |
| `live/` | production execution layer (see README "Layout") |

## Added in M8 (PCRM / Vorobets, Steiner, Palomar, López de Prado), 2026-10-19

| Module | Contents | Source |
|---|---|---|
| `risk/panel_risk_model.py` | point-in-time style risk model on long panels: daily WLS on market + size/vol/liquidity/momentum/funding, EWMA factor and idio variances, predicted portfolio vol, Barra bias statistic, PnL attribution factor vs residual | EQI Ch 4–7, Barra |
| `risk/tail.py` | VaR/CVaR on scenarios (Rockafellar–Uryasev), Euler CVaR contributions, Entropy Pooling views (mean, second moment, probability) and stress tables via `fortitudo.tech` | PCRM Ch 5–7 |
| `backtest/metrics.py` (additions) | moment ratios (skew, kurtosis), Mertens standard error, PSR, DSR, circular block-bootstrap Sharpe CI | Mertens 2002; Bailey & López de Prado |
| `backtest/simulation.py` | block-bootstrap and state-conditioned resampled paths, path statistics, effective history | PCRM Ch 3 (time- and state-dependent resampling, simplified) |
| `portfolio/pit_construction.py` (additions) | EWMA covariance option, Euler variance-share cap, market-vol state and regime multiplier of the target vol (`market_vol_state`, `regime_scale`), two-state HMM alternative (`hmm_regime_multipliers`, tested and rejected, note 024); the `mp` covariance option is `risk.covariance.denoise_covariance` (rejected for the YOLO book, note 024) | Steiner essays 4, 6; PCRM 6.3 |
| `portfolio/neutral.py` (addition) | `project_out`: zero the portfolio's loadings on chosen style exposures | EQI Ch 10 (factor-neutral) |
| `data/health.py`, `data/sync.py` | data-layer health report; content-based S3 mirror with local-change protection | — |
