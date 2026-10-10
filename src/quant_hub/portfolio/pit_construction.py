"""Point-in-time portfolio construction on a long daily panel: vol target, mean-variance, HRP.

Moved from yolo-microstructure (note 015).

Every builder takes the long panel (with `weight` = raw combo, `in_universe`,
`price_returns`, `funding`, `ewma_vol`) and returns a `scaled_weight` series.
Covariances are point-in-time: estimated on the trailing `window` days ending
at d-1, so the weight on day d uses only returns up to d-1.

Builders:
- vol_target:  production weights rescaled so the predicted portfolio vol hits
               `target_vol` (annualised), gross capped at `max_gross`.
- mv:          w ∝ Σ⁻¹ μ with μ = raw combo, Σ = Ledoit-Wolf ("lw") or
               Marchenko-Pastur denoised ("mp") covariance; then cap, gross ≤ 1.
- hrp:         hierarchical risk parity (López de Prado 2016) on the correlation
               matrix for the magnitudes, sign from the raw combo; cap, gross ≤ 1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform

from quant_hub.risk.covariance import denoise_covariance, ledoit_wolf_shrinkage


def _returns_wide(panel: pd.DataFrame) -> pd.DataFrame:
    r = panel["price_returns"] + panel["funding"]
    return pd.DataFrame({"r": r, "date": panel["date"], "asset": panel["asset"]}).pivot(
        index="date", columns="asset", values="r"
    )


def pit_covariance(ret: pd.DataFrame, date, window: int, method: str) -> pd.DataFrame | None:
    """Covariance from the `window` rows strictly before `date`; None if too short."""
    hist = ret.loc[ret.index < date].tail(window)
    hist = hist.dropna(axis=1, thresh=int(0.8 * window)).fillna(0.0)
    if len(hist) < max(40, window // 2) or hist.shape[1] < 3:
        return None
    if method == "lw":
        cov, _ = ledoit_wolf_shrinkage(hist.to_numpy())
    elif method == "mp":
        cov = denoise_covariance(hist.to_numpy())
    elif method == "sample":
        cov = np.cov(hist.to_numpy(), rowvar=False)
    else:
        raise ValueError(method)
    return pd.DataFrame(cov, index=hist.columns, columns=hist.columns)


def _cap_and_normalise(w: pd.Series, cap: float) -> pd.Series:
    w = w.clip(-cap, cap)
    g = w.abs().sum()
    return w / g if g > 1 else w


def hrp_weights(cov: pd.DataFrame) -> pd.Series:
    """Hierarchical risk parity magnitudes (sum to 1) from a covariance matrix."""
    std = np.sqrt(np.diag(cov.to_numpy()))
    corr = cov.to_numpy() / np.outer(std, std)
    corr = np.clip(np.nan_to_num(corr), -1, 1)
    dist = np.sqrt(0.5 * (1 - corr))
    np.fill_diagonal(dist, 0.0)
    link = linkage(squareform(dist, checks=False), method="single")
    order = to_tree(link, rd=False).pre_order()
    w = pd.Series(1.0, index=cov.index[order])
    clusters = [list(w.index)]
    while clusters:
        nxt = []
        for c in clusters:
            if len(c) < 2:
                continue
            half = len(c) // 2
            a, b = c[:half], c[half:]
            for grp in (a, b):
                sub = cov.loc[grp, grp].to_numpy()
                ivp = 1 / np.diag(sub)
                ivp /= ivp.sum()
                grp_var = ivp @ sub @ ivp
                grp.append(grp_var)  # stash the cluster variance at the end
            va, vb = a.pop(), b.pop()
            alpha = 1 - va / (va + vb)
            w[a] *= alpha
            w[b] *= 1 - alpha
            nxt += [a, b]
        clusters = nxt
    return w


def build(
    panel: pd.DataFrame,
    method: str,
    base_weight: pd.Series,
    window: int = 120,
    cov_method: str = "lw",
    target_vol: float = 0.12,
    max_gross: float = 1.5,
    cap: float = 0.25,
) -> pd.Series:
    """Return a scaled_weight series for `method` in {"vol_target", "mv", "hrp"}."""
    ret = _returns_wide(panel)
    out = pd.Series(np.nan, index=panel.index, name="scaled_weight")
    for date, idx in panel.groupby("date").indices.items():
        rows = panel.iloc[idx]
        held = rows["in_universe"].to_numpy() & rows["weight"].notna().to_numpy()
        if held.sum() < 3:
            continue
        assets = rows["asset"].to_numpy()[held]
        cov = pit_covariance(ret, date, window, cov_method)
        if cov is None:
            continue
        common = [a for a in assets if a in cov.index]
        if len(common) < 3:
            continue
        cov = cov.loc[common, common]
        pos = {a: i for a, i in zip(rows["asset"], idx, strict=True)}
        if method == "vol_target":
            w = base_weight.loc[[pos[a] for a in common]].to_numpy()
            w = np.nan_to_num(w)
            cur = np.sqrt(w @ cov.to_numpy() @ w * 365)
            if cur <= 0:
                continue
            scale = min(target_vol / cur, max_gross / max(np.abs(w).sum(), 1e-9))
            new = pd.Series(w * scale, index=common)
        elif method == "mv":
            mu = rows.set_index("asset").loc[common, "weight"].to_numpy()
            try:
                direction = np.linalg.solve(cov.to_numpy() + 1e-10 * np.eye(len(common)), mu)
            except np.linalg.LinAlgError:
                continue
            new = _cap_and_normalise(
                pd.Series(direction / np.abs(direction).sum(), index=common), cap
            )
        elif method == "hrp":
            mag = hrp_weights(cov)
            sign = np.sign(rows.set_index("asset").loc[mag.index, "weight"])
            new = _cap_and_normalise(mag * sign, cap)
        else:
            raise ValueError(method)
        out.loc[[pos[a] for a in new.index]] = new.to_numpy()
    return out
