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


def pit_covariance(
    ret: pd.DataFrame, date, window: int, method: str, halflife: float | None = None
) -> pd.DataFrame | None:
    """Covariance from the `window` rows strictly before `date`; None if too short.

    Methods: "lw" Ledoit-Wolf, "mp" Marchenko-Pastur denoised, "sample", "ewma"
    (exponentially weighted with `halflife` days, shrunk towards its own diagonal with
    the Ledoit-Wolf intensity of the equal-weighted sample: reacts to a volatility
    regime within ~2 half-lives instead of `window` days).
    """
    hist = ret.loc[ret.index < date].tail(window)
    hist = hist.dropna(axis=1, thresh=int(0.8 * window)).fillna(0.0)
    if len(hist) < max(40, window // 2) or hist.shape[1] < 3:
        return None
    x = hist.to_numpy()
    if method == "lw":
        cov, _ = ledoit_wolf_shrinkage(x)
    elif method == "mp":
        cov = denoise_covariance(x)
    elif method == "sample":
        cov = np.cov(x, rowvar=False)
    elif method == "ewma":
        hl = halflife or window / 4
        lam = 0.5 ** (1.0 / hl)
        wts = lam ** np.arange(len(x) - 1, -1, -1)
        wts = wts / wts.sum()
        xc = x - wts @ x
        cov = (xc * wts[:, None]).T @ xc
        _, delta = ledoit_wolf_shrinkage(x)
        cov = (1 - delta) * cov + delta * np.diag(np.diag(cov))
    else:
        raise ValueError(method)
    return pd.DataFrame(cov, index=hist.columns, columns=hist.columns)


def cap_risk_share(w: np.ndarray, cov: np.ndarray, max_share: float, iters: int = 20) -> np.ndarray:
    """Shrink names whose Euler variance contribution exceeds `max_share` of the total.

    Contribution_i = w_i (Σw)_i / (w'Σw). Offenders are scaled down towards the cap,
    the rest is left alone; repeated until no name exceeds the cap (risk budgeting as
    a constraint, variance version of PCRM 6.3). Signs are preserved.
    """
    w = w.astype(float).copy()
    for _ in range(iters):
        tot = float(w @ cov @ w)
        if tot <= 0:
            return w
        share = w * (cov @ w) / tot
        over = share > max_share * (1 + 1e-9)
        if not over.any():
            return w
        w[over] *= np.sqrt(max_share / share[over])
    return w


def _hmm2_fit(x: np.ndarray, iters: int = 100, tol: float = 1e-6) -> dict:
    """Two-state Gaussian HMM (shared variance) by Baum-Welch with a sticky initialisation."""
    lo, hi = np.quantile(x, [0.25, 0.75])
    mu = np.array([lo, hi])
    sig = max(np.std(x) / 2, 1e-6)
    a = np.array([[0.95, 0.05], [0.05, 0.95]])
    pi = np.array([0.5, 0.5])
    ll_old = -np.inf
    for _ in range(iters):
        alpha, c = _hmm2_forward(x, mu, sig, a, pi)
        beta = _hmm2_backward(x, mu, sig, a, c)
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        b = _hmm2_emission(x, mu, sig)
        xi = alpha[:-1, :, None] * a[None, :, :] * (b[1:, None, :] * beta[1:, None, :])
        xi /= xi.sum(axis=(1, 2), keepdims=True)
        a = xi.sum(axis=0) / gamma[:-1].sum(axis=0)[:, None]
        pi = gamma[0]
        mu = (gamma * x[:, None]).sum(axis=0) / gamma.sum(axis=0)
        sig = np.sqrt((gamma * (x[:, None] - mu) ** 2).sum() / len(x))
        ll = np.log(c).sum()
        if abs(ll - ll_old) < tol:
            break
        ll_old = ll
    return {"mu": mu, "sig": sig, "a": a, "pi": pi}


def _hmm2_emission(x, mu, sig):
    z = (x[:, None] - mu[None, :]) / sig
    return np.exp(-0.5 * z**2) / (sig * np.sqrt(2 * np.pi))


def _hmm2_forward(x, mu, sig, a, pi):
    b = _hmm2_emission(x, mu, sig)
    alpha = np.zeros((len(x), 2))
    c = np.zeros(len(x))
    alpha[0] = pi * b[0]
    c[0] = alpha[0].sum()
    alpha[0] /= c[0]
    for t in range(1, len(x)):
        alpha[t] = (alpha[t - 1] @ a) * b[t]
        c[t] = alpha[t].sum()
        alpha[t] /= c[t]
    return alpha, c


def _hmm2_backward(x, mu, sig, a, c):
    b = _hmm2_emission(x, mu, sig)
    beta = np.ones((len(x), 2))
    for t in range(len(x) - 2, -1, -1):
        beta[t] = (a @ (b[t + 1] * beta[t + 1])) / c[t + 1]
    return beta


def hmm_regime_multipliers(
    panel: pd.DataFrame,
    refit_every: int = 21,
    min_history: int = 250,
    threshold: float = 0.5,
    scale: float = 0.5,
    vol_window: int = 10,
    min_vol_ratio: float = 1.5,
    min_persistence: float = 0.8,
) -> pd.Series:
    """Per-date multiplier from a two-state hidden Markov model of realised volatility.

    Observable: log of the ``vol_window``-day rolling volatility of the equal-weighted
    universe return (persistent, so two levels are identifiable; a switching-variance
    model on raw returns splits iid noise into spurious states). A two-state Gaussian
    HMM is refitted by Baum-Welch every ``refit_every`` days on data strictly before
    the date; between refits the *filtered* probability of the high-vol state is
    updated with the fixed parameters (forward pass only, no smoothing). Multiplier
    ``scale`` when that probability, lagged one day, exceeds ``threshold``, else 1.
    The states count as identified only when the high state's volatility level is at
    least ``min_vol_ratio`` times the low one and both states are sticky
    (``a_ii >= min_persistence``); otherwise (typically before the first high-vol
    episode in the history, when EM splits noise) the multiplier stays 1. Point in
    time by construction.
    """
    r = _returns_wide(panel)
    mkt = r.mean(axis=1, skipna=True)
    obs = np.log(mkt.rolling(vol_window).std()).dropna()
    x = obs.to_numpy()
    mult = pd.Series(1.0, index=mkt.index)
    for i in range(min_history, len(x), refit_every):
        prm = _hmm2_fit(x[:i])
        end = min(i + refit_every, len(x))
        sel = obs.index[i:end]
        high = int(np.argmax(prm["mu"]))
        separated = abs(prm["mu"][1] - prm["mu"][0]) >= np.log(min_vol_ratio)
        sticky = np.diag(prm["a"]).min() >= min_persistence
        if not (separated and sticky):
            continue
        alpha, _ = _hmm2_forward(x[:end], prm["mu"], prm["sig"], prm["a"], prm["pi"])
        p_high = pd.Series(alpha[:, high], index=obs.index[:end]).shift(1)  # known at the decision
        mult.loc[sel] = np.where(p_high.loc[sel] > threshold, scale, 1.0)
    return mult


def market_vol_state(
    panel: pd.DataFrame, window: int = 30, lookback: int = 365, quantile: float = 0.8
) -> pd.DataFrame:
    """Market-volatility state per date: columns `state`, `threshold`.

    `state` = annualised EWMA (`window` half-life) vol of the equal-weighted total
    return (price + funding) of every asset in the panel with a return that day, lagged
    one day. `threshold` = its trailing `lookback`-day `quantile`, lagged one day
    (point in time). The panel, not the universe, defines the market on purpose: the
    state must not change when the universe rule does. Production reports exactly this
    frame, so the published state and the applied multiplier always agree.
    """
    r = _returns_wide(panel)
    mkt = r.mean(axis=1, skipna=True)
    state = (mkt.ewm(halflife=window, min_periods=window).std() * np.sqrt(365)).shift(1)
    thr = state.rolling(lookback, min_periods=lookback // 2).quantile(quantile).shift(1)
    return pd.DataFrame({"state": state, "threshold": thr})


def regime_scale(
    panel: pd.DataFrame,
    window: int = 30,
    lookback: int = 365,
    quantile: float = 0.8,
    scale: float = 0.5,
) -> pd.Series:
    """Per-date multiplier of the target vol.

    `scale` while the market-vol state (`market_vol_state`) exceeds its trailing
    quantile, else 1.
    """
    s = market_vol_state(panel, window, lookback, quantile)
    mult = pd.Series(1.0, index=s.index)
    mult[(s["state"] > s["threshold"]) & s["threshold"].notna()] = scale
    return mult


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
    cov_halflife: float | None = None,
    max_risk_share: float | None = None,
    regime: dict | None = None,
) -> pd.Series:
    """Return a scaled_weight series for `method` in {"vol_target", "mv", "hrp"}.

    Optional overlays (vol_target only): `max_risk_share` caps each name's Euler
    variance contribution before the vol scaling; `regime` (kwargs of
    :func:`regime_scale`) multiplies the target vol by the state multiplier.
    """
    ret = _returns_wide(panel)
    mult = None
    if regime:
        regime = dict(regime)
        mult = regime.pop("multipliers", None)  # precomputed per-date multipliers win
        if mult is None:
            mult = regime_scale(panel, **regime)
    out = pd.Series(np.nan, index=panel.index, name="scaled_weight")
    for date, idx in panel.groupby("date").indices.items():
        rows = panel.iloc[idx]
        held = rows["in_universe"].to_numpy() & rows["weight"].notna().to_numpy()
        if held.sum() < 3:
            continue
        assets = rows["asset"].to_numpy()[held]
        cov = pit_covariance(ret, date, window, cov_method, cov_halflife)
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
            if max_risk_share is not None:
                w = cap_risk_share(w, cov.to_numpy(), max_risk_share)
            cur = np.sqrt(w @ cov.to_numpy() @ w * 365)
            if cur <= 0:
                continue
            tv = target_vol * (float(mult.get(date, 1.0)) if mult is not None else 1.0)
            scale = min(tv / cur, max_gross / max(np.abs(w).sum(), 1e-9))
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
