"""Tail risk on scenarios: CVaR, Euler contributions, Entropy Pooling stress (PCRM Ch 5–7).

Everything works on a scenario matrix ``R`` (S scenarios x I components) with
probabilities ``p`` (default uniform). For a dynamic strategy the natural scenarios
are its realised daily PnL *components* (factor PnL and residual from
:mod:`quant_hub.risk.panel_risk_model`), so that stress views can be put on factor
returns; for "today's book" the scenarios are historical asset returns and the
components are ``w_i r_i``.

CVaR here is the expected loss beyond VaR at level ``alpha`` (a positive number, in
return units), computed on demeaned or raw PnL as asked. Euler contributions
``E[c_k | loss in the tail]`` sum to the portfolio CVaR exactly (Tasche).
Entropy Pooling (Meucci; fortitudo.tech implementation) finds the posterior
probabilities closest to the prior that satisfy moment views, so a stress test is a
re-weighting of history rather than an invented path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_ALPHA = 0.95


def _p(p, n: int) -> np.ndarray:
    if p is None:
        return np.full(n, 1.0 / n)
    p = np.asarray(p, float).reshape(-1)
    return p / p.sum()


def var_cvar(pnl, alpha: float = _ALPHA, p=None, demean: bool = False) -> tuple[float, float]:
    """(VaR, CVaR) of a PnL vector at level ``alpha`` under probabilities ``p``.

    Losses are ``-pnl``; VaR is the ``alpha`` quantile of the loss, CVaR the
    probability-weighted mean loss at and beyond it (with the boundary scenario
    split, as in Rockafellar–Uryasev). Positive numbers mean losses.
    """
    x = np.asarray(pnl, float).reshape(-1)
    p = _p(p, len(x))
    if demean:
        x = x - float(x @ p)
    loss = -x
    order = np.argsort(loss)[::-1]  # worst first
    ls, ps = loss[order], p[order]
    cum = np.cumsum(ps)
    tail = 1.0 - alpha
    k = int(np.searchsorted(cum, tail, side="right"))
    k = min(k, len(ls) - 1)
    covered = cum[k - 1] if k > 0 else 0.0
    cvar = (ls[:k] @ ps[:k] + (tail - covered) * ls[k]) / tail
    return float(ls[k]), float(cvar)


def euler_cvar_contributions(
    components: pd.DataFrame, alpha: float = _ALPHA, p=None, demean: bool = False
) -> pd.Series:
    """Contribution of each component to the CVaR of their sum (sums to the CVaR).

    ``contribution_k = -E[c_k | sum <= -VaR]`` with the boundary scenario weighted
    fractionally, so that the contributions add up to ``var_cvar(sum)[1]``.
    """
    c = components.to_numpy(float)
    x = c.sum(axis=1)
    p = _p(p, len(x))
    if demean:
        mu = p @ c
        c = c - mu
        x = c.sum(axis=1)
    loss = -x
    order = np.argsort(loss)[::-1]
    cum = np.cumsum(p[order])
    tail = 1.0 - alpha
    k = int(np.searchsorted(cum, tail, side="right"))
    k = min(k, len(order) - 1)
    weights = np.zeros(len(x))
    weights[order[:k]] = p[order[:k]]
    covered = cum[k - 1] if k > 0 else 0.0
    weights[order[k]] += tail - covered
    contrib = -(weights @ c) / tail
    return pd.Series(contrib, index=components.columns, name="cvar_contribution")


def tail_summary(pnl, p=None, periods_per_year: int = 365) -> dict:
    """Mean, vol, Sharpe, CVaR 95/99 (daily, in return units) and the worst scenario."""
    x = np.asarray(pnl, float).reshape(-1)
    p = _p(p, len(x))
    mu = float(x @ p)
    sd = float(np.sqrt(p @ (x - mu) ** 2))
    return {
        "mean_ann": mu * periods_per_year,
        "vol_ann": sd * np.sqrt(periods_per_year),
        "sharpe": mu / sd * np.sqrt(periods_per_year) if sd > 0 else np.nan,
        "cvar95": var_cvar(x, 0.95, p)[1],
        "cvar99": var_cvar(x, 0.99, p)[1],
        "worst": float(-x.min()),
        "cvar95_over_vol": var_cvar(x, 0.95, p)[1] / sd if sd > 0 else np.nan,
    }


# ----------------------------------------------------------------------------- entropy pooling
def mean_view(R: pd.DataFrame, column: str, value: float) -> tuple[np.ndarray, np.ndarray]:
    """Equality view ``E[R[column]] = value`` as an (A, b) row pair."""
    a = R[column].to_numpy(float)[None, :]
    return a, np.array([[value]])


def vol_view(R: pd.DataFrame, column: str, vol: float, mean: float = 0.0):
    """Equality view on the second moment: ``E[R[column]²] = vol² + mean²``."""
    a = (R[column].to_numpy(float) ** 2)[None, :]
    return a, np.array([[vol**2 + mean**2]])


def prob_view(R: pd.DataFrame, mask, prob: float) -> tuple[np.ndarray, np.ndarray]:
    """Equality view on the probability of a scenario set: ``P[mask] = prob``."""
    a = np.asarray(mask, float)[None, :]
    return a, np.array([[prob]])


def entropy_pooling(
    R: pd.DataFrame, views: list[tuple[np.ndarray, np.ndarray]], p=None
) -> np.ndarray:
    """Posterior probabilities satisfying the equality views, closest to the prior (KL)."""
    from fortitudo.tech import entropy_pooling as _ep

    n = len(R)
    prior = _p(p, n)[:, None]
    A = np.vstack([a for a, _ in views])
    b = np.vstack([v for _, v in views])
    # the probabilities must still sum to one
    A = np.vstack([A, np.ones((1, n))])
    b = np.vstack([b, np.array([[1.0]])])
    q = np.asarray(_ep(prior, A, b), float).reshape(-1)
    q = np.clip(q, 0.0, None)
    return q / q.sum()  # the solver meets the constraints to ~1e-8; renormalise exactly


def effective_scenarios(p) -> float:
    """Effective number of scenarios ``exp(entropy)`` under ``p`` (how extreme a stress is)."""
    p = np.asarray(p, float)
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def stress_table(
    components: pd.DataFrame, scenarios: dict[str, list], p=None, alpha: float = _ALPHA
) -> pd.DataFrame:
    """Strategy statistics under the prior and under each Entropy-Pooling scenario.

    ``components`` are daily PnL components (sum = strategy PnL); ``scenarios`` maps a
    name to a list of views built with :func:`mean_view`/:func:`vol_view`/:func:`prob_view`.
    """
    total = components.sum(axis=1).to_numpy()
    rows = {
        "prior": {**tail_summary(total, p), "eff_scenarios": effective_scenarios(_p(p, len(total)))}
    }
    for name, views in scenarios.items():
        q = entropy_pooling(components, views, p)
        rows[name] = {**tail_summary(total, q), "eff_scenarios": effective_scenarios(q)}
        contrib = euler_cvar_contributions(components, alpha, q)
        rows[name]["top_cvar_source"] = str(contrib.idxmax())
    return pd.DataFrame(rows).T
