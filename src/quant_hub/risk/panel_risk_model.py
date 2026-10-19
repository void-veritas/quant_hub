"""Point-in-time style risk model on a long daily panel (EQI Ch 4–7, Barra-style).

Each day ``d`` the cross-section of next-day returns is regressed (WLS) on exposures
known at ``d``: a market intercept and standardised style characteristics. The
outputs are daily factor returns, residuals, exposures and ``R²``; from them the
model forecasts the volatility of any weight vector and attributes its PnL to
factors and residual. Everything is causal: a forecast for the return earned from
``d`` to ``d+1`` uses factor returns and residuals through ``d-1`` only.

Vocabulary: *exposures* are per-asset characteristics (one column per factor),
*loadings* of a portfolio are ``x = B' w``.

Typical use::

    expo = build_exposures(panel, caps=cap_wide, dollar_volume=dv_wide)
    model = fit(panel, expo)
    pred = predicted_vol(model, weights_long)           # annualised, by date
    bias = bias_statistic(pred, realised_returns)       # ~1 when calibrated
    attr = attribute(model, weights_long)               # daily PnL by factor + idio
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_hub.risk.factor_model import cross_sectional_regression

STYLES = ("size", "vol", "liquidity", "momentum", "funding")
MARKET = "market"


# ----------------------------------------------------------------------------- exposures
def _zscore_xs(s: pd.Series, clip: float = 3.0) -> pd.Series:
    """Cross-sectional standardisation within a date, winsorised at ±clip."""
    mu, sd = s.mean(), s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return s * 0.0
    return ((s - mu) / sd).clip(-clip, clip)


def build_exposures(
    panel: pd.DataFrame,
    caps: pd.DataFrame | None = None,
    dollar_volume: pd.DataFrame | None = None,
    momentum_days: int = 30,
    funding_days: int = 30,
    liquidity_days: int = 30,
    vol_col: str = "ewma_vol",
    funding_col: str = "carry_funding",
    universe_col: str | None = "in_universe",
) -> pd.DataFrame:
    """Long frame (date, asset, market, size, vol, liquidity, momentum, funding).

    Characteristics known at the decision of day ``d``:
    - ``size``: log market cap on ``d-1`` (``caps`` wide date x asset);
    - ``vol``: log of the panel's EWMA volatility on ``d``;
    - ``liquidity``: log mean dollar volume over the ``liquidity_days`` bars ending ``d-1``
      (``dollar_volume`` wide, bar-dated);
    - ``momentum``: log price return over ``momentum_days`` days to ``d``;
    - ``funding``: mean funding over ``funding_days`` days to ``d`` (sign as in the panel).
    Styles are z-scored across the day's universe and winsorised at ±3; the market
    exposure is 1. Missing characteristics are set to 0 (neutral) after scaling.
    """
    df = panel[["date", "asset", "close", vol_col, funding_col]].copy()
    if universe_col and universe_col in panel:
        df["_u"] = panel[universe_col].astype(bool)
    else:
        df["_u"] = True
    df = df.sort_values(["asset", "date"])
    g = df.groupby("asset", sort=False)
    df["momentum"] = np.log(df["close"]) - g["close"].transform(
        lambda s: np.log(s).shift(momentum_days)
    )
    df["funding"] = g[funding_col].transform(
        lambda s: s.rolling(funding_days, min_periods=funding_days // 2).mean()
    )
    df["vol"] = np.log(df[vol_col].where(df[vol_col] > 0))
    if caps is not None:
        c = caps.shift(1).stack(future_stack=True).rename("size").reset_index()
        c.columns = ["date", "asset", "size"]
        c["size"] = np.log(c["size"].where(c["size"] > 0))
        df = df.merge(c, on=["date", "asset"], how="left")
    else:
        df["size"] = np.nan
    if dollar_volume is not None:
        dv = dollar_volume.rolling(liquidity_days, min_periods=liquidity_days // 2).mean()
        dv = dv.shift(1).stack(future_stack=True).rename("liquidity").reset_index()
        dv.columns = ["date", "asset", "liquidity"]
        dv["liquidity"] = np.log(dv["liquidity"].where(dv["liquidity"] > 0))
        df = df.merge(dv, on=["date", "asset"], how="left")
    else:
        df["liquidity"] = np.nan
    df = df[df["_u"]]
    out = df[["date", "asset"]].copy()
    out[MARKET] = 1.0
    for k in STYLES:
        out[k] = df.groupby("date")[k].transform(_zscore_xs).fillna(0.0)
    return out.reset_index(drop=True)


# ----------------------------------------------------------------------------- fit
@dataclass
class PanelRiskModel:
    factors: list[str]
    factor_returns: pd.DataFrame  # date x factor, return earned from d to d+1
    residuals: pd.DataFrame  # date x asset, same timing
    exposures: pd.DataFrame  # long (date, asset, factors...)
    r2: pd.Series  # by date
    n: pd.Series  # assets per day

    def loadings(self, weights: pd.DataFrame) -> pd.DataFrame:
        """Portfolio loadings ``x_k(d) = sum_i w_i(d) B_ik(d)`` for long weights (date, asset, weight)."""
        m = weights.merge(self.exposures, on=["date", "asset"], how="inner")
        for k in self.factors:
            m[k] = m[k] * m["weight"]
        return m.groupby("date")[self.factors].sum()


def fit(
    panel: pd.DataFrame,
    exposures: pd.DataFrame,
    return_col: str = "total_returns",
    vol_col: str = "ewma_vol",
    factors: tuple[str, ...] = (MARKET, *STYLES),
    winsor_sd: float = 8.0,
) -> PanelRiskModel:
    """Daily WLS of next-day returns on today's exposures; weights ``1/vol²``.

    Returns of day ``d+1`` in the panel are the return earned from the decision at
    ``d`` to the decision at ``d+1``, so the regression for date ``d`` pairs
    ``exposures[d]`` with ``returns[d+1]``. Returns are winsorised at ``winsor_sd``
    daily volatilities before the fit (EQI: robust factor returns).
    """
    p = panel[["date", "asset", return_col, vol_col]].sort_values(["asset", "date"]).copy()
    p["y"] = p.groupby("asset")[return_col].shift(-1)
    p = p.merge(exposures, on=["date", "asset"], how="inner").dropna(subset=["y", vol_col])
    p = p[p[vol_col] > 0]
    lim = winsor_sd * p[vol_col]
    p["y"] = p["y"].clip(-lim, lim)
    f_rows, r2_rows, n_rows, eps_frames = [], [], [], []
    cols = list(factors)
    for d, g in p.groupby("date", sort=True):
        if len(g) < len(cols) + 3:
            continue
        b = g[cols].to_numpy()
        y = g["y"].to_numpy()
        f, eps, _ = cross_sectional_regression(y, b, idio_var=g[vol_col].to_numpy() ** 2)
        f_rows.append(pd.Series(f, index=cols, name=d))
        w = 1.0 / g[vol_col].to_numpy() ** 2
        tss = float(np.sum(w * (y - np.average(y, weights=w)) ** 2))
        r2_rows.append(1.0 - float(np.sum(w * eps**2)) / tss if tss > 0 else np.nan)
        n_rows.append(len(g))
        eps_frames.append(pd.Series(eps, index=g["asset"].to_numpy(), name=d))
    fr = pd.DataFrame(f_rows)
    fr.index.name = "date"
    res = pd.DataFrame(eps_frames)
    res.index.name = "date"
    dates = fr.index
    return PanelRiskModel(
        cols,
        fr,
        res,
        exposures,
        pd.Series(r2_rows, index=dates, name="r2"),
        pd.Series(n_rows, index=dates, name="n"),
    )


# ----------------------------------------------------------------------------- forecasts
def _ewma_cov(f: pd.DataFrame, halflife: float) -> dict[pd.Timestamp, np.ndarray]:
    """EWMA factor covariance using returns strictly before each date."""
    lam = 0.5 ** (1.0 / halflife)
    out: dict[pd.Timestamp, np.ndarray] = {}
    cov = None
    prev = None
    for d, row in f.iterrows():
        if cov is not None:
            out[d] = cov.copy()
        x = row.to_numpy(float)[:, None]
        if np.isnan(x).any():
            continue
        if cov is None:
            cov = x @ x.T
        else:
            cov = lam * cov + (1 - lam) * (x @ x.T)
        prev = d
    _ = prev
    return out


def predicted_vol(
    model: PanelRiskModel,
    weights: pd.DataFrame,
    halflife_factor: float = 60.0,
    halflife_idio: float = 30.0,
    periods_per_year: int = 365,
    min_obs: int = 20,
) -> pd.DataFrame:
    """Annualised forecast of the portfolio's volatility for each date's weights.

    ``var = x'Ω_f x + Σ w_i² σ_ε,i²`` with the factor covariance an EWMA of factor
    returns strictly before ``d`` and idio variances an EWMA of squared residuals
    strictly before ``d`` (a residual of day ``d`` is only known at ``d+1``).
    Returns columns ``total``, ``factor``, ``idio`` (annualised vol, idio share in
    ``idio_share``).
    """
    x = model.loadings(weights)
    covs = _ewma_cov(model.factor_returns, halflife_factor)
    lam = 0.5 ** (1.0 / halflife_idio)
    eps2 = (model.residuals**2).ewm(alpha=1 - lam, adjust=False, min_periods=5).mean().shift(1)
    w_wide = weights.pivot(index="date", columns="asset", values="weight")
    rows = []
    for i, d in enumerate(x.index):
        if d not in covs or i < min_obs:
            continue
        xv = x.loc[d].to_numpy()
        fv = float(xv @ covs[d] @ xv)
        wd = w_wide.loc[d].dropna() if d in w_wide.index else pd.Series(dtype=float)
        e = eps2.loc[d].reindex(wd.index) if d in eps2.index else pd.Series(dtype=float)
        e = e.fillna(e.median() if e.notna().any() else 0.0)
        iv = float((wd**2 * e).sum())
        tot = fv + iv
        rows.append(
            dict(
                date=d,
                total=np.sqrt(tot * periods_per_year),
                factor=np.sqrt(fv * periods_per_year),
                idio=np.sqrt(iv * periods_per_year),
                idio_share=iv / tot if tot > 0 else np.nan,
            )
        )
    return pd.DataFrame(rows).set_index("date")


def bias_statistic(predicted: pd.Series, realised: pd.Series, window: int = 60) -> dict:
    """Barra bias statistic: std of standardised returns ``r_t / σ_pred,t``.

    ``predicted`` is the annualised forecast for the return earned from ``t`` to
    ``t+1``; ``realised`` is that return (daily). A calibrated model gives ≈ 1;
    > 1 means the model under-forecasts risk. Also returns the rolling ratio of
    realised to predicted vol over ``window`` days and the share of 95% exceedances.
    """
    df = pd.concat({"pred": predicted, "r": realised}, axis=1, sort=True).dropna()
    z = df["r"] / (df["pred"] / np.sqrt(365))
    roll = (df["r"].rolling(window).std() * np.sqrt(365)) / df["pred"].rolling(window).mean()
    return {
        "bias": float(z.std(ddof=1)),
        "n": int(len(z)),
        "exceed_95": float((z.abs() > 1.96).mean()),
        "rolling_ratio": roll.dropna(),
        "z": z,
    }


# ----------------------------------------------------------------------------- attribution
def attribute(model: PanelRiskModel, weights: pd.DataFrame) -> pd.DataFrame:
    """Daily PnL by factor and residual: ``pnl_k = x_k f_k``, ``idio = Σ w_i ε_i``.

    Rows are decision dates; the PnL is earned from ``d`` to ``d+1``. Columns:
    the factors, ``idio`` and ``total`` (which equals the realised portfolio return
    on the assets covered by the model, up to the winsorisation of returns).
    """
    x = model.loadings(weights)
    f = model.factor_returns.reindex(x.index)
    pnl = x * f
    w_wide = weights.pivot(index="date", columns="asset", values="weight").reindex(x.index)
    eps = model.residuals.reindex(index=x.index, columns=w_wide.columns)
    pnl["idio"] = (w_wide * eps).sum(axis=1, min_count=1)
    pnl["total"] = pnl.sum(axis=1, min_count=1)
    return pnl.dropna(how="all")


def summarise_attribution(pnl: pd.DataFrame, periods_per_year: int = 365) -> pd.DataFrame:
    """Per column: total PnL, share of total, annualised mean, vol, t-stat, Sharpe."""
    tot = pnl["total"].sum()
    rows = []
    for c in pnl.columns:
        s = pnl[c].dropna()
        mu, sd = s.mean(), s.std(ddof=1)
        rows.append(
            dict(
                source=c,
                pnl=s.sum(),
                share=s.sum() / tot if tot else np.nan,
                ann_mean=mu * periods_per_year,
                ann_vol=sd * np.sqrt(periods_per_year),
                t_stat=mu / sd * np.sqrt(len(s)) if sd > 0 else np.nan,
                sharpe=mu / sd * np.sqrt(periods_per_year) if sd > 0 else np.nan,
            )
        )
    return pd.DataFrame(rows).set_index("source")
