"""Idiosyncratic-skill decomposition: selection / sizing / timing.

Implements two complementary decompositions of idiosyncratic PnL: the
counterfactual-book method (XSE / XSTSE) of *Advanced Portfolio Management* Ch 8,
and the closed-form selection x diversification + sizing identity of *The Elements
of Quantitative Investing* Theorem 14.1. Also the effective-breadth measure and the
information-ratio-from-hit-rate fundamental law.

Shape conventions: ``n`` assets, ``T`` periods; panel inputs are ``(T, n)``.
"""

from __future__ import annotations

import numpy as np


def effective_breadth(weights) -> float:
    """Effective number of positions ``||w||_1^2 / ||w||_2^2 = 1 / Herfindahl`` (EQI 14 / APM 8).

    Parameters
    ----------
    weights : array_like, shape (n,)
        Position weights (or dollar-vol weights).

    Returns
    -------
    float
        The effective breadth, between 1 and ``n``.
    """
    w = np.abs(np.asarray(weights, float))
    l2sq = np.sum(w**2)
    return float(np.sum(w) ** 2 / l2sq) if l2sq > 0 else 0.0


def ir_from_hit_rate(hit_rate: float, n_eff: float, periods_per_year: float = 252.0) -> float:
    """Annualized IR ``(2*hit - 1) * sqrt(periods * N_eff)`` (APM 8.2.2).

    Parameters
    ----------
    hit_rate : float
        Fraction of correct directional bets.
    n_eff : float
        Effective number of independent positions (see :func:`effective_breadth`).
    periods_per_year : float, default 252.0
        Number of independent decision periods per year.

    Returns
    -------
    float
        The implied annualized information ratio.
    """
    return (2.0 * hit_rate - 1.0) * np.sqrt(periods_per_year * n_eff)


def selection_sizing_decomposition(weights, idio_returns, idio_vol):
    """One-period idiosyncratic information-ratio identity (EQI Theorem 14.1).

    Uses dollar-vol positions ``w~ = sigma_i w_i`` and z-scored idio returns
    ``eps~ = eps_i / sigma_i`` so ``IR ~ selection * diversification + sizing``.

    Parameters
    ----------
    weights : array_like, shape (n,)
        Portfolio weights.
    idio_returns : array_like, shape (n,)
        Realized idiosyncratic returns.
    idio_vol : array_like, shape (n,)
        Per-asset idiosyncratic volatilities.

    Returns
    -------
    dict
        Keys ``idio_ir``, ``selection``, ``diversification``, ``sizing``,
        ``effective_positions``, and ``n``.
    """
    w = np.asarray(weights, float)
    eps = np.asarray(idio_returns, float)
    sig = np.asarray(idio_vol, float)
    n = len(w)
    w_tilde = sig * w
    eps_tilde = eps / sig
    norm2 = np.sqrt(np.sum(w_tilde**2))
    if norm2 == 0:
        return {"idio_ir": 0.0, "selection": 0.0, "diversification": 0.0, "sizing": 0.0}
    idio_ir = float(w_tilde @ eps_tilde / norm2)
    selection = float(np.mean(eps_tilde * np.sign(w_tilde)))
    diversification = float(np.sum(np.abs(w_tilde)) / norm2)  # in [1, sqrt(n)]
    sizing = idio_ir - selection * diversification
    return {
        "idio_ir": idio_ir,
        "selection": selection,
        "diversification": diversification,
        "sizing": sizing,
        "effective_positions": diversification**2,
        "n": n,
    }


def _equalize_within_date(weights):
    """Same signs, equal ``|NMV|`` within a date, same gross (removes sizing)."""
    w = np.asarray(weights, float)
    gross = np.sum(np.abs(w))
    active = w != 0
    if not active.any():
        return w.copy()
    return np.where(active, np.sign(w) * gross / active.sum(), 0.0)


def selection_sizing_timing(weights_ts, idio_returns_ts):
    """Counterfactual-book decomposition of idiosyncratic PnL (APM Ch 8).

    Builds progressively simplified books: XSE equalizes bet sizes within each date;
    XSTSE also equalizes gross across dates. Then ``sizing = live - XSE``,
    ``timing = XSE - XSTSE``, ``selection = XSTSE``.

    Parameters
    ----------
    weights_ts : array_like, shape (T, n)
        Portfolio weights each period.
    idio_returns_ts : array_like, shape (T, n)
        Idiosyncratic returns each period.

    Returns
    -------
    dict
        Keys ``total_idio_pnl``, ``sizing``, ``timing``, ``selection`` (the last
        three sum to ``total_idio_pnl``).
    """
    w = np.atleast_2d(np.asarray(weights_ts, float))
    eps = np.atleast_2d(np.asarray(idio_returns_ts, float))
    live = float(np.sum(w * eps))

    xse = np.vstack([_equalize_within_date(w[t]) for t in range(len(w))])
    xse_pnl = float(np.sum(xse * eps))

    # XSTSE: also equalize gross across dates (constant per-date gross of 1)
    signs = np.sign(w)
    active_counts = np.sum(w != 0, axis=1, keepdims=True)
    xstse = np.divide(signs, active_counts, out=np.zeros_like(signs), where=active_counts > 0)
    xstse_pnl = float(np.sum(xstse * eps))

    return {
        "total_idio_pnl": live,
        "sizing": live - xse_pnl,
        "timing": xse_pnl - xstse_pnl,
        "selection": xstse_pnl,
    }
