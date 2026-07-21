"""The factor risk model ``Omega_r = B Omega_f B' + Omega_e``.

Implements the factor-model core of *The Elements of Quantitative Investing*
Ch 4/6 (and *Advanced Portfolio Management* Ch 11). :class:`FactorModel` bundles
loadings, factor covariance, and idiosyncratic variances and exposes the risk
decomposition, factor-mimicking portfolios, and marginal risk consumed by the
construction, hedging, and attribution modules. The estimation helpers cover the
fundamental cross-sectional regression and factor-covariance cleanup; build a
statistical model instead with :func:`quant_hub.risk.covariance.ppca`.

Shape conventions: ``n`` assets, ``m`` factors, ``T`` periods; loadings ``B`` are
``(n, m)``, factor covariance ``Omega_f`` is ``(m, m)``, idiosyncratic variances
are length ``n``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FactorModel:
    """A linear factor risk model ``Omega_r = B Omega_f B' + diag(idio_var)``.

    Parameters
    ----------
    loadings : array_like, shape (n, m)
        Factor loadings ``B``.
    factor_cov : array_like, shape (m, m)
        Factor return covariance ``Omega_f``.
    idio_var : array_like, shape (n,)
        Diagonal of the idiosyncratic covariance ``Omega_e``.
    alpha : array_like, shape (n,), optional
        Expected idiosyncratic returns, carried for convenience.
    """

    loadings: np.ndarray
    factor_cov: np.ndarray
    idio_var: np.ndarray
    alpha: np.ndarray | None = None

    def __post_init__(self):
        """Coerce inputs to float arrays with the expected dimensionality."""
        self.loadings = np.atleast_2d(np.asarray(self.loadings, float))
        self.factor_cov = np.atleast_2d(np.asarray(self.factor_cov, float))
        self.idio_var = np.asarray(self.idio_var, float).reshape(-1)

    @property
    def n_assets(self) -> int:
        """Number of assets ``n``."""
        return self.loadings.shape[0]

    @property
    def n_factors(self) -> int:
        """Number of factors ``m``."""
        return self.loadings.shape[1]

    def asset_covariance(self) -> np.ndarray:
        """Full asset covariance ``Omega_r = B Omega_f B' + diag(idio_var)``.

        Returns
        -------
        numpy.ndarray, shape (n, n)
            The asset return covariance matrix.
        """
        return self.loadings @ self.factor_cov @ self.loadings.T + np.diag(self.idio_var)

    def exposures(self, weights) -> np.ndarray:
        """Factor exposures ``b = B' w`` of a portfolio.

        Parameters
        ----------
        weights : array_like, shape (n,)
            Portfolio weights (or dollar positions).

        Returns
        -------
        numpy.ndarray, shape (m,)
            Exposure to each factor.
        """
        return self.loadings.T @ np.asarray(weights, float)

    def factor_variance(self, weights) -> float:
        """Factor-driven variance ``b' Omega_f b``.

        Parameters
        ----------
        weights : array_like, shape (n,)
            Portfolio weights.

        Returns
        -------
        float
            The factor component of portfolio variance.
        """
        b = self.exposures(weights)
        return float(b @ self.factor_cov @ b)

    def idio_variance(self, weights) -> float:
        """Idiosyncratic variance ``sum(idio_var * w^2)``.

        Parameters
        ----------
        weights : array_like, shape (n,)
            Portfolio weights.

        Returns
        -------
        float
            The idiosyncratic component of portfolio variance.
        """
        w = np.asarray(weights, float)
        return float(np.sum(self.idio_var * w**2))

    def portfolio_variance(self, weights) -> float:
        """Total portfolio variance = factor + idiosyncratic.

        Parameters
        ----------
        weights : array_like, shape (n,)
            Portfolio weights.

        Returns
        -------
        float
            The total predicted variance.
        """
        return self.factor_variance(weights) + self.idio_variance(weights)

    def pct_idio_variance(self, weights) -> float:
        """Idiosyncratic share of total variance (APM keeps this >= ~75%).

        Parameters
        ----------
        weights : array_like, shape (n,)
            Portfolio weights.

        Returns
        -------
        float
            Idiosyncratic variance divided by total variance, in ``[0, 1]``.
        """
        idio = self.idio_variance(weights)
        total = idio + self.factor_variance(weights)
        return idio / total if total > 0 else 0.0

    def factor_mimicking_portfolios(self) -> np.ndarray:
        """Factor-mimicking portfolios ``P = Omega_e^-1 B (B' Omega_e^-1 B)^-1``.

        Column ``i`` is the minimum-idiosyncratic-variance portfolio with unit
        exposure to factor ``i`` and zero exposure to the others (EQI Ch 9,
        APM Ch 11).

        Returns
        -------
        numpy.ndarray, shape (n, m)
            The FMP weight matrix, one portfolio per column.
        """
        w_inv = 1.0 / self.idio_var
        weighted_b = self.loadings * w_inv[:, None]  # Omega_e^-1 B
        return weighted_b @ np.linalg.inv(self.loadings.T @ weighted_b)

    def marginal_contribution_to_factor_risk(self, weights) -> np.ndarray:
        """Per-factor marginal contribution to factor risk.

        ``MCFR = Omega_f b / sqrt(b' Omega_f b)`` -- the sensitivity of factor risk
        to each factor exposure (APM Ch 7).

        Parameters
        ----------
        weights : array_like, shape (n,)
            Portfolio weights.

        Returns
        -------
        numpy.ndarray, shape (m,)
            Marginal contribution to factor risk per factor (zeros if factor
            variance is zero).
        """
        b = self.exposures(weights)
        fac_var = float(b @ self.factor_cov @ b)
        if fac_var <= 0:
            return np.zeros_like(b)
        return (self.factor_cov @ b) / np.sqrt(fac_var)

    def factor_covariance_noise(self) -> np.ndarray:
        """Estimation-noise covariance ``(B' Omega_e^-1 B)^-1`` of FMP factor returns.

        Subtract this from ``var(f_hat)`` to de-bias the factor covariance, and use
        it for the confidence bands in attribution (EQI Ch 6/14).

        Returns
        -------
        numpy.ndarray, shape (m, m)
            The FMP factor-return estimation-noise covariance.
        """
        w_inv = 1.0 / self.idio_var
        return np.linalg.inv(self.loadings.T @ (self.loadings * w_inv[:, None]))


def cross_sectional_regression(returns, loadings, idio_var=None):
    """One-period WLS factor estimation ``f_hat = (B'WB)^-1 B'W r`` (EQI Ch 6).

    Parameters
    ----------
    returns : array_like, shape (n,)
        Cross-section of asset returns for one period.
    loadings : array_like, shape (n, m)
        Factor loadings ``B`` (known at the start of the period).
    idio_var : array_like, shape (n,), optional
        Idiosyncratic variances giving the WLS weights ``W = Omega_e^-1``. When
        ``None``, ordinary least squares is used.

    Returns
    -------
    factor_returns : numpy.ndarray, shape (m,)
        Estimated factor returns ``f_hat``.
    residuals : numpy.ndarray, shape (n,)
        Idiosyncratic returns ``eps = r - B f_hat``.
    fmp_weights : numpy.ndarray, shape (n, m)
        Factor-mimicking portfolio weights, one portfolio per column, so that
        ``fmp_weights.T @ returns == factor_returns``.

    Notes
    -----
    Uses a pseudo-inverse, so rank-deficient loadings are handled gracefully.
    """
    r = np.asarray(returns, float)
    b = np.atleast_2d(np.asarray(loadings, float))
    w = np.ones(len(r)) if idio_var is None else 1.0 / np.asarray(idio_var, float)
    bt_w = b.T * w  # m x n
    gram_inv = np.linalg.pinv(bt_w @ b)
    fmp = gram_inv @ bt_w  # m x n
    factor_returns = fmp @ r
    residuals = r - b @ factor_returns
    return factor_returns, residuals, fmp.T


def estimate_factor_covariance(factor_returns, estimation_noise=None, shrinkage: float = 0.0):
    """Estimate the factor covariance from a history of estimated factor returns.

    Parameters
    ----------
    factor_returns : array_like, shape (T, m)
        Time series of estimated factor returns ``f_hat``.
    estimation_noise : array_like, shape (m, m), optional
        The FMP estimation-noise covariance (see
        :meth:`FactorModel.factor_covariance_noise`); subtracted to de-bias.
    shrinkage : float, default 0.0
        Linear shrinkage intensity toward a scaled identity, in ``[0, 1]``.

    Returns
    -------
    numpy.ndarray, shape (m, m)
        The estimated factor covariance ``Omega_f``.
    """
    f = np.atleast_2d(np.asarray(factor_returns, float))
    cov = np.atleast_2d(np.cov(f, rowvar=False))
    if estimation_noise is not None:
        cov = cov - np.asarray(estimation_noise, float)
    if shrinkage > 0:
        m = cov.shape[0]
        mu = np.trace(cov) / m
        cov = (1.0 - shrinkage) * cov + shrinkage * mu * np.eye(m)
    return cov


def winsorize_returns(returns, d_max: float = 8.0):
    """Robustly winsorize returns by a median-scaled log return (EQI Ch 6).

    Computes ``d = |log(1 + r)| / median(|log(1 + r)|)`` and clips values with
    ``d > d_max``, returning a mask of flagged observations so genuine jumps can be
    reviewed rather than silently dropped.

    Parameters
    ----------
    returns : array_like, shape (T,)
        Return series.
    d_max : float, default 8.0
        Robust-z threshold above which returns are clipped (typically 5-10).

    Returns
    -------
    winsorized : numpy.ndarray, shape (T,)
        The winsorized returns.
    flagged : numpy.ndarray of bool, shape (T,)
        True where an observation exceeded ``d_max`` and was clipped.
    """
    r = np.asarray(returns, float)
    log_r = np.log1p(r)
    median_abs = np.median(np.abs(log_r))
    if median_abs == 0:
        return r.copy(), np.zeros(len(r), dtype=bool)
    d = np.abs(log_r) / median_abs
    flagged = d > d_max
    clipped = np.clip(log_r, -d_max * median_abs, d_max * median_abs)
    return np.expm1(clipped), flagged
