"""Ex-post attribution of a long-short factor book from a long daily panel.

Moved from yolo-microstructure (M0 §4, notes 004, 017). `factors` are panel
columns holding each factor's raw weight contribution; the raw combined weight
is `weight` = sum(factors) / divisor.

Two decompositions on *ideal* weights (pre-buffer, pre-cost), both on daily
total returns r_{i,t} = price return + funding, with weights w_{i,t-1} set at the
previous decision snapshot:

A. By weight source (exact, no residual). The scaled weight is a per-asset
   multiple of (momo + trend + carry)/3, so each factor's share of the final
   weight is its share of the raw weight. Also splits price vs funding and long
   vs short legs.

B. By risk model (with residual).
   - Time-series: regress the strategy's daily return on the returns of the pure
     factor portfolios (momo, trend, carry weights as they are) and the
     equal-weighted market. Robust on n=10 assets.
   - Cross-sectional (Elements ch. 14): daily WLS of asset returns on loadings
     [1, z(momo), z(trend), z(carry)] known at t-1 → factor returns f_t, idio ε_t,
     exposures b_t = Bᵀw, factor/idio PnL with estimation-error CIs.

Sub-factor efficacy: daily cross-sectional rank IC of each sub-feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from quant_hub.attribution.performance import selection_sizing_timing
from quant_hub.risk.factor_model import cross_sectional_regression


def _factor_cols(panel: pd.DataFrame, factors) -> list[str]:
    if factors is not None:
        return list(factors)
    return [c for c in panel.columns if c.endswith("_megafactor")]


def _short(name: str) -> str:
    return name.replace("_megafactor", "")


def _wide(panel: pd.DataFrame, col: str) -> pd.DataFrame:
    return panel.pivot(index="date", columns="asset", values=col)


def _lagged_weights(panel: pd.DataFrame, col: str = "scaled_weight") -> pd.DataFrame:
    """Weights that earn day t's return: set at t-1, zero where undefined."""
    return _wide(panel, col).shift(1).fillna(0.0)


# ------------------------------------------------------------------ A. by source


def decompose_by_source(panel: pd.DataFrame, factors=None, divisor: float = 3.0) -> dict:
    """Daily PnL (in weight units, i.e. return on capital) split by factor, leg and asset."""
    w = _lagged_weights(panel)
    r_px = _wide(panel, "price_returns").fillna(0.0)
    fund = _wide(panel, "funding").fillna(0.0)
    r_tot = r_px + fund
    raw = _wide(panel, "weight").shift(1)
    # The factor split is exact only for the production construction (scaled weight =
    # per-asset multiple of the raw combo). With overlays (neutralisation, vol target)
    # the panel carries `prod_weight`: the split is done on it and the remainder
    # (scaled - prod) is reported as "overlay".
    base_w = _lagged_weights(panel, "prod_weight") if "prod_weight" in panel.columns else w
    ratio = (base_w / raw).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    contrib = {}
    fcols = _factor_cols(panel, factors)
    for f in fcols:
        comp = _wide(panel, f).shift(1).fillna(0.0) / divisor
        contrib[_short(f)] = (comp * ratio * r_tot).sum(axis=1)
    daily = pd.DataFrame(contrib)
    daily["overlay"] = ((w - base_w.reindex_like(w).fillna(0.0)) * r_tot).sum(axis=1)
    daily["total"] = (w * r_tot).sum(axis=1)
    daily["price"] = (w * r_px).sum(axis=1)
    daily["funding"] = (w * fund).sum(axis=1)
    daily["long_leg"] = (w.clip(lower=0) * r_tot).sum(axis=1)
    daily["short_leg"] = (w.clip(upper=0) * r_tot).sum(axis=1)
    daily["gross"] = w.abs().sum(axis=1)
    daily["net"] = w.sum(axis=1)
    by_asset = (w * r_tot).sum(axis=0).sort_values(ascending=False)
    # factor gross: how much of the book each factor drives, in |weight| units
    factor_gross = pd.DataFrame(
        {
            _short(f): (_wide(panel, f).shift(1).abs() / divisor * ratio.abs()).sum(axis=1)
            for f in fcols
        }
    )
    return {"daily": daily, "by_asset": by_asset, "factor_gross": factor_gross}


# ------------------------------------------------------------------ B1. time-series


def factor_portfolio_returns(
    panel: pd.DataFrame, vol_scaled: bool = False, factors=None
) -> pd.DataFrame:
    """Daily returns of the pure factor portfolios and the equal-weighted market.

    `vol_scaled=True` applies the strategy's own per-asset scaling (scaled/raw
    ratio, i.e. inverse vol, cap and gross normalisation) to each factor's
    weights, so the regression residual isolates interaction effects only.
    """
    r_tot = (_wide(panel, "price_returns") + _wide(panel, "funding")).fillna(0.0)
    uni = _wide(panel, "in_universe").fillna(False).astype(bool).shift(1).fillna(False)
    ratio = 1.0
    if vol_scaled:
        raw = _wide(panel, "weight").shift(1)
        ratio = (_lagged_weights(panel) / raw).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out = {}
    for f in _factor_cols(panel, factors):
        wf = _wide(panel, f).shift(1).fillna(0.0) * ratio
        out[_short(f)] = (wf * r_tot).sum(axis=1)
    mkt = (r_tot.where(uni)).mean(axis=1)
    out["market"] = mkt.fillna(0.0)
    return pd.DataFrame(out)


def time_series_attribution(
    panel: pd.DataFrame,
    window: int = 180,
    vol_scaled: bool = False,
    factors=None,
    divisor: float = 3.0,
) -> dict:
    """OLS of the strategy return on factor-portfolio returns, HAC t-stats, rolling betas."""
    y = decompose_by_source(panel, factors, divisor)["daily"]["total"]
    X = factor_portfolio_returns(panel, vol_scaled=vol_scaled, factors=factors)
    X = X.loc[y.index]
    Xc = sm.add_constant(X)
    model = sm.OLS(y, Xc, missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": 10})
    coef = pd.DataFrame({"beta": model.params, "t_hac": model.tvalues, "p": model.pvalues})
    coef.loc["const", "beta_annual_%"] = 100 * 365 * model.params["const"]
    pnl_share = {}
    for c in X.columns:
        pnl_share[c] = float(model.params[c] * X[c].sum())
    pnl_share["residual_alpha"] = float(model.params["const"] * len(y))
    pnl_share["residual_noise"] = float(model.resid.sum())
    roll = pd.DataFrame(index=y.index, columns=X.columns, dtype=float)
    for i in range(window, len(y) + 1):
        sl = slice(i - window, i)
        b = np.linalg.lstsq(Xc.iloc[sl].to_numpy(), y.iloc[sl].to_numpy(), rcond=None)[0]
        roll.iloc[i - 1] = b[1:]
    return {
        "coef": coef,
        "r2": float(model.rsquared),
        "pnl_share": pd.Series(pnl_share),
        "rolling_betas": roll,
        "factor_returns": X,
        "strategy_return": y,
    }


# ------------------------------------------------------------------ B2. cross-sectional


def _zscore_xs(df: pd.DataFrame) -> pd.DataFrame:
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1, ddof=1), axis=0)


def cross_sectional_attribution(
    panel: pd.DataFrame, ewma_lambda: float = 0.94, factors=None
) -> dict:
    """Daily WLS factor model on [1, z(momo), z(trend), z(carry)] known at t-1.

    Idio variances for the WLS weights come from an EWMA of squared residuals of
    a first OLS pass (two-stage, Elements ch. 6). Returns factor returns, idio
    returns, exposures, PnL by factor and idio, and the estimation-error CI.
    """
    r_tot = _wide(panel, "price_returns") + _wide(panel, "funding")
    w = _lagged_weights(panel)
    fcols = _factor_cols(panel, factors)
    shorts = [_short(f) for f in fcols]
    load = {_short(f): _zscore_xs(_wide(panel, f)).shift(1) for f in fcols}
    names = ["market"] + shorts
    dates = r_tot.index
    assets = list(r_tot.columns)
    T, n, m = len(dates), len(assets), len(names)

    def _design(t):
        B = np.column_stack([np.ones(n)] + [load[k].iloc[t].to_numpy() for k in shorts])
        r = r_tot.iloc[t].to_numpy()
        ok = np.isfinite(r) & np.isfinite(B).all(axis=1)
        return B, r, ok

    # pass 1: OLS residual variances
    resid = np.full((T, n), np.nan)
    for t in range(T):
        B, r, ok = _design(t)
        if ok.sum() < m + 2:
            continue
        _, e, _ = cross_sectional_regression(r[ok], B[ok])
        resid[t, ok] = e
    var = pd.DataFrame(resid, index=dates, columns=assets) ** 2
    idio_var = var.ewm(alpha=1 - ewma_lambda, min_periods=20).mean().shift(1)
    floor = np.nanmedian(idio_var.to_numpy())
    # pass 2: WLS with lagged idio variances
    f_ret = np.full((T, m), np.nan)
    eps = np.full((T, n), np.nan)
    expo = np.zeros((T, m))
    noise_var = np.zeros(T)
    for t in range(T):
        B, r, ok = _design(t)
        if ok.sum() < m + 2:
            continue
        iv = idio_var.iloc[t].to_numpy()[ok]
        iv = np.where(np.isfinite(iv) & (iv > 0), iv, floor)
        f, e, _ = cross_sectional_regression(r[ok], B[ok], idio_var=iv)
        f_ret[t], eps[t, ok] = f, e
        b = B[ok].T @ w.iloc[t].to_numpy()[ok]
        expo[t] = b
        gram_inv = np.linalg.pinv((B[ok].T / iv) @ B[ok])
        noise_var[t] = float(b @ gram_inv @ b)
    f_ret = pd.DataFrame(f_ret, index=dates, columns=names)
    eps = pd.DataFrame(eps, index=dates, columns=assets)
    expo = pd.DataFrame(expo, index=dates, columns=names)
    factor_pnl_daily = expo * f_ret
    idio_pnl_daily = (w * eps.fillna(0.0)).sum(axis=1)
    se = float(np.sqrt(np.nansum(noise_var)))
    sst = selection_sizing_timing(w.to_numpy(), eps.fillna(0.0).to_numpy())
    return {
        "factor_returns": f_ret,
        "idio_returns": eps,
        "exposures": expo,
        "factor_pnl_daily": factor_pnl_daily,
        "idio_pnl_daily": idio_pnl_daily,
        "factor_pnl": factor_pnl_daily.sum(),
        "idio_pnl": float(idio_pnl_daily.sum()),
        "pnl_std_error": se,
        "selection_sizing_timing": sst,
    }
