"""Market neutrality for a long-short book: rolling betas and three neutralisation modes.

Moved from yolo-microstructure (note 014). Panels are long frames with columns
`date`, `asset`, `price_returns`, `funding`, `in_universe`.
"""

from __future__ import annotations

import pandas as pd


def rolling_betas(panel: pd.DataFrame, window: int = 90, min_periods: int = 45) -> pd.Series:
    """Per (date, asset) beta to the equal-weight basket of in-universe names, known at t-1.

    OLS slope of the asset's daily total return on the basket's return over the
    trailing `window` days, shifted by one day so the weight on day d uses only
    returns up to d-1. Aligned to `panel.index`.
    """
    r = panel["price_returns"] + panel["funding"]
    wide = pd.DataFrame({"r": r, "date": panel["date"], "asset": panel["asset"]}).pivot(
        index="date", columns="asset", values="r"
    )
    uni = (
        pd.DataFrame({"u": panel["in_universe"], "date": panel["date"], "asset": panel["asset"]})
        .pivot(index="date", columns="asset", values="u")
        .fillna(False)
    )
    basket = wide.where(uni).mean(axis=1)
    cov = wide.rolling(window, min_periods=min_periods).cov(basket)
    var = basket.rolling(window, min_periods=min_periods).var()
    beta = cov.div(var, axis=0).shift(1)
    long = beta.stack(future_stack=True).rename("beta").reset_index()
    long.columns = ["date", "asset", "beta"]
    out = panel[["date", "asset"]].merge(long, on=["date", "asset"], how="left")["beta"]
    out.index = panel.index
    return out


def neutralise(
    weight: pd.Series,
    dates: pd.Series,
    mode: str,
    beta: pd.Series | None = None,
    assets: pd.Series | None = None,
    hedge_asset: str = "BTC",
    cap: float = 0.25,
) -> pd.Series:
    """Remove net market exposure from scaled weights, one of three ways.

    "dollar": subtract the cross-sectional mean of the held weights (Σw = 0).
    "beta":   project the weight vector onto the complement of the beta vector
              (Σ β_i w_i = 0); names without a beta keep their weight.
    "hedge":  keep every weight, add -Σ β_i w_i / β_hedge to `hedge_asset`
              (the hedge leg is exempt from the per-asset cap).
    After "dollar" and "beta" the gross is renormalised to ≤ 1 and the cap
    re-applied, as in the production clamp. NaN weights are left as NaN.
    """
    w = weight.copy()
    held = w.notna() & (w != 0)
    if mode == "dollar":
        for _ in range(4):  # demean, cap, repeat: the cap can re-introduce a small net
            mean = w.where(held).groupby(dates).transform("mean")
            w = w.where(~held, w - mean).clip(-cap, cap)
    elif mode in ("beta", "hedge"):
        if beta is None:
            raise ValueError("beta series required")
        b = beta.where(held)
        bw = (b * w).groupby(dates).transform("sum")  # portfolio beta per date
        if mode == "beta":
            bb = (b * b).groupby(dates).transform("sum")
            adj = (bw / bb).where(bb > 0, 0.0) * b
            w = w.where(~held | adj.isna(), w - adj.fillna(0.0))
        else:
            if assets is None:
                raise ValueError("assets series required for hedge mode")
            is_hedge = assets == hedge_asset
            b_h = b.where(is_hedge).groupby(dates).transform("max")
            hedge = (-bw / b_h).where(b_h.abs() > 0.2, 0.0)
            w = w.where(~is_hedge, w.fillna(0.0) + hedge)
            return w.rename("scaled_weight")
    else:
        raise ValueError(mode)
    w = w.clip(-cap, cap)
    gross = w.abs().groupby(dates).transform("sum")
    w = w.where(gross <= 1, w / gross)
    return w.rename("scaled_weight")
