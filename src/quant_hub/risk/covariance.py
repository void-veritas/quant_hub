"""Covariance estimation for factor risk models.

Implements the covariance-estimation material of *The Elements of Quantitative
Investing* Ch 6-7: sample and exponentially weighted covariance, Ledoit-Wolf
linear shrinkage, Marchenko-Pastur / BBP spectral thresholds, eigenvalue denoising
for spiked covariances, PCA and PPCA statistical factor models, and Procrustes
rotation for eigenvector stability across periods.

All estimators take a ``(T, n)`` return matrix (rows = time, columns = assets).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _as_matrix(returns) -> np.ndarray:
    """Coerce a return input to a 2-D float array of shape (T, n)."""
    arr = (
        returns.to_numpy()
        if isinstance(returns, (pd.DataFrame, pd.Series))
        else np.asarray(returns)
    )
    return np.atleast_2d(arr.astype(float))


def sample_covariance(returns) -> np.ndarray:
    """Plain sample covariance of asset returns.

    Parameters
    ----------
    returns : array_like, shape (T, n)
        Return matrix.

    Returns
    -------
    numpy.ndarray, shape (n, n)
        The sample covariance matrix.
    """
    return np.cov(_as_matrix(returns), rowvar=False)


def ewma_covariance(returns, halflife: float) -> np.ndarray:
    """Exponentially weighted covariance, weighting recent observations most.

    Parameters
    ----------
    returns : array_like, shape (T, n)
        Return matrix.
    halflife : float
        EWMA half-life in periods.

    Returns
    -------
    numpy.ndarray, shape (n, n)
        The EWMA covariance matrix as of the last observation.
    """
    r = _as_matrix(returns)
    decay = 0.5 ** (1.0 / halflife)
    centered = r - r.mean(axis=0)
    cov = np.cov(r, rowvar=False)
    for t in range(len(centered)):
        x = centered[t][:, None]
        cov = (1.0 - decay) * (x @ x.T) + decay * cov
    return cov


def ledoit_wolf_shrinkage(returns):
    """Ledoit-Wolf (2004) linear shrinkage toward a scaled identity.

    Shrinks the sample covariance toward ``(trace(S)/n) I`` with a closed-form
    optimal intensity, giving a well-conditioned estimate even when ``n ~ T``.

    Parameters
    ----------
    returns : array_like, shape (T, n)
        Return matrix.

    Returns
    -------
    shrunk_cov : numpy.ndarray, shape (n, n)
        The shrunk covariance matrix.
    intensity : float
        The estimated shrinkage intensity in ``[0, 1]``.
    """
    r = _as_matrix(returns)
    n_obs, n = r.shape
    x = r - r.mean(axis=0)
    sample = (x.T @ x) / n_obs
    mu = np.trace(sample) / n
    target = mu * np.eye(n)
    dispersion = np.sum((sample - target) ** 2)  # ||S - muI||_F^2
    noise = 0.0
    for t in range(n_obs):
        outer = np.outer(x[t], x[t])
        noise += np.sum((outer - sample) ** 2)
    noise /= n_obs**2
    noise = min(noise, dispersion)
    intensity = noise / dispersion if dispersion > 0 else 0.0
    shrunk = intensity * target + (1.0 - intensity) * sample
    return shrunk, intensity


def marchenko_pastur_edges(n: int, n_obs: int) -> tuple[float, float]:
    """Bulk eigenvalue support of a pure-noise correlation matrix.

    Parameters
    ----------
    n : int
        Number of assets.
    n_obs : int
        Number of observations ``T``.

    Returns
    -------
    tuple of float
        ``(lambda_minus, lambda_plus) = ((1 - sqrt(g))^2, (1 + sqrt(g))^2)`` with
        ``g = n / T``; eigenvalues inside this band are indistinguishable from noise.
    """
    gamma = n / n_obs
    root = np.sqrt(gamma)
    return (1.0 - root) ** 2, (1.0 + root) ** 2


def spike_threshold(n: int, n_obs: int) -> float:
    """BBP detectability threshold ``1 + sqrt(n / T)`` for a spiked eigenvalue.

    Parameters
    ----------
    n : int
        Number of assets.
    n_obs : int
        Number of observations ``T``.

    Returns
    -------
    float
        The threshold below which an eigenvalue carries no reliable signal.
    """
    return 1.0 + np.sqrt(n / n_obs)


def denoise_covariance(returns) -> np.ndarray:
    """Marchenko-Pastur eigenvalue clipping on the correlation matrix (EQI Ch 7).

    Eigenvalues inside the Marchenko-Pastur bulk are replaced by their common
    average (trace-preserving), isolating the informative spikes, then rescaled back
    to a covariance using the sample volatilities.

    Parameters
    ----------
    returns : array_like, shape (T, n)
        Return matrix.

    Returns
    -------
    numpy.ndarray, shape (n, n)
        The denoised covariance matrix.
    """
    r = _as_matrix(returns)
    n_obs, n = r.shape
    std = r.std(axis=0, ddof=1)
    corr = np.corrcoef(r, rowvar=False)
    vals, vecs = np.linalg.eigh(corr)
    _, edge_hi = marchenko_pastur_edges(n, n_obs)
    bulk = vals < edge_hi
    if bulk.any():
        vals = vals.copy()
        vals[bulk] = vals[bulk].mean()
    corr_clean = (vecs * vals) @ vecs.T
    np.fill_diagonal(corr_clean, 1.0)
    return corr_clean * np.outer(std, std)


def pca_factor_model(returns, n_factors: int, standardize: bool = False):
    """Statistical factor model via SVD -- the best rank-``m`` approximation (EQI Ch 7).

    Parameters
    ----------
    returns : array_like, shape (T, n)
        Return matrix.
    n_factors : int
        Number of factors ``m`` to retain.
    standardize : bool, default False
        If True, standardize each asset's returns before the decomposition.

    Returns
    -------
    loadings : numpy.ndarray, shape (n, m)
        Leading eigenvectors of the asset covariance (eigenportfolios).
    factor_returns : numpy.ndarray, shape (T, m)
        Factor return time series (projections ``R @ loadings``).
    explained_variance_ratio : numpy.ndarray, shape (m,)
        Fraction of total variance explained by each retained factor.
    """
    r = _as_matrix(returns)
    centered = r - r.mean(axis=0)
    if standardize:
        centered = centered / centered.std(axis=0, ddof=1)
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    loadings = vt[:n_factors].T
    factor_returns = u[:, :n_factors] * s[:n_factors]
    explained = (s**2 / np.sum(s**2))[:n_factors]
    return loadings, factor_returns, explained


def ppca(returns, n_factors: int):
    """Probabilistic PCA (Tipping-Bishop MLE) with isotropic noise (EQI Ch 7).

    Models ``Sigma_r = B Omega_f B' + sigma^2 I`` and returns a factorization that
    plugs straight into :class:`~quant_hub.risk.factor_model.FactorModel`.

    Parameters
    ----------
    returns : array_like, shape (T, n)
        Return matrix.
    n_factors : int
        Number of factors ``m``.

    Returns
    -------
    loadings : numpy.ndarray, shape (n, m)
        Leading eigenvectors of the sample covariance.
    factor_cov : numpy.ndarray, shape (m, m)
        Diagonal of the leading eigenvalues minus ``sigma^2`` (shrunk).
    idio_var : float
        The isotropic idiosyncratic variance ``sigma^2`` (mean of the trailing
        ``n - m`` eigenvalues).
    """
    sample = sample_covariance(returns)
    vals, vecs = np.linalg.eigh(sample)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    m = n_factors
    sigma2 = float(np.mean(vals[m:])) if m < len(vals) else 0.0
    loadings = vecs[:, :m]
    factor_cov = np.diag(np.maximum(vals[:m] - sigma2, 0.0))
    return loadings, factor_cov, sigma2


def procrustes_rotation(loadings_old: np.ndarray, loadings_new: np.ndarray):
    """Orthogonal rotation aligning new loadings to old ones (Wahba's problem, EQI Ch 7).

    Solves ``min ||B_old - B_new X||_F`` subject to ``X'X = I``, fixing eigenvector
    sign flips and subspace rotation between consecutive periods.

    Parameters
    ----------
    loadings_old : numpy.ndarray, shape (n, m)
        Reference loadings.
    loadings_new : numpy.ndarray, shape (n, m)
        Loadings to align to the reference.

    Returns
    -------
    rotation : numpy.ndarray, shape (m, m)
        The orthogonal rotation ``X``.
    rotated : numpy.ndarray, shape (n, m)
        ``loadings_new @ rotation``, aligned to ``loadings_old``.
    """
    u, _, vt = np.linalg.svd(loadings_old.T @ loadings_new)
    rotation = vt.T @ u.T
    return rotation, loadings_new @ rotation
