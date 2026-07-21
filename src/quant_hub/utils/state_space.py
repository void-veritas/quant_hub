"""Linear-Gaussian state-space tools: the Kalman filter and the Muth/EWMA link.

Implements the state-space material from *The Elements of Quantitative Investing*
Ch 2 (appendix). The Kalman filter underlies the state-space volatility models in
:mod:`quant_hub.risk.volatility`, and -- through the state <-> factor-return
analogy of Ch 4 -- factor estimation.

The model, in the book's notation, is::

    x_{t+1} = A x_t + eps,   eps ~ N(0, Q)   (state transition)
    y_t     = H x_t + eta,   eta ~ N(0, R)   (observation)

Public objects use descriptive names (``transition`` for A, ``observation`` for H,
and so on); the single-letter symbols above are kept only in the documentation.
"""

from __future__ import annotations

import numpy as np


class KalmanFilter:
    """Time-invariant linear-Gaussian Kalman filter.

    Parameters
    ----------
    transition : array_like, shape (n, n)
        State transition matrix ``A``: how the latent state evolves each step.
    observation : array_like, shape (k, n)
        Observation matrix ``H`` mapping state to measurement.
    state_cov : array_like, shape (n, n)
        Covariance ``Q`` of the state-transition noise ``eps``.
    obs_cov : array_like, shape (k, k)
        Covariance ``R`` of the observation noise ``eta``.
    initial_state : array_like, shape (n,)
        Prior mean of the state at t = 0.
    initial_cov : array_like, shape (n, n)
        Prior covariance of the state at t = 0.

    Notes
    -----
    Scalars are accepted for the 1-D case and promoted to 1x1 matrices.
    """

    def __init__(self, transition, observation, state_cov, obs_cov, initial_state, initial_cov):
        self.transition = np.atleast_2d(np.asarray(transition, float))
        self.observation = np.atleast_2d(np.asarray(observation, float))
        self.state_cov = np.atleast_2d(np.asarray(state_cov, float))
        self.obs_cov = np.atleast_2d(np.asarray(obs_cov, float))
        self.initial_state = np.asarray(initial_state, float).reshape(-1)
        self.initial_cov = np.atleast_2d(np.asarray(initial_cov, float))

    def filter(self, observations):
        """Run the forward filter over a sequence of observations.

        Parameters
        ----------
        observations : array_like, shape (T,) or (T, k)
            Measurements ``y_t``. A 1-D input is treated as scalar observations.

        Returns
        -------
        states : numpy.ndarray, shape (T, n)
            Filtered state means ``x_hat_{t|t}``.
        covariances : numpy.ndarray, shape (T, n, n)
            Filtered state covariances ``P_{t|t}``.
        """
        ys = np.asarray(observations, float)
        if ys.ndim == 1:
            ys = ys[:, None]
        n_steps = ys.shape[0]
        n_state = self.transition.shape[0]
        eye = np.eye(n_state)
        states = np.zeros((n_steps, n_state))
        covariances = np.zeros((n_steps, n_state, n_state))
        state_pred, cov_pred = self.initial_state.copy(), self.initial_cov.copy()
        for t in range(n_steps):
            innovation_cov = self.observation @ cov_pred @ self.observation.T + self.obs_cov
            gain = cov_pred @ self.observation.T @ np.linalg.inv(innovation_cov)
            residual = ys[t] - self.observation @ state_pred
            state_upd = state_pred + gain @ residual
            cov_upd = (eye - gain @ self.observation) @ cov_pred
            states[t], covariances[t] = state_upd, cov_upd
            state_pred = self.transition @ state_upd
            cov_pred = self.transition @ cov_upd @ self.transition.T + self.state_cov
        return states, covariances

    def steady_state_gain(self, max_iter: int = 10_000, tol: float = 1e-12):
        """Compute the stationary Kalman gain by iterating the Riccati recursion.

        Parameters
        ----------
        max_iter : int, default 10000
            Maximum number of Riccati iterations.
        tol : float, default 1e-12
            Convergence tolerance on the max absolute change of the covariance.

        Returns
        -------
        numpy.ndarray, shape (n, k)
            The stationary Kalman gain ``K``.
        """
        n_state = self.transition.shape[0]
        eye = np.eye(n_state)
        cov = self.initial_cov.copy()
        for _ in range(max_iter):
            innovation_cov = self.observation @ cov @ self.observation.T + self.obs_cov
            gain = cov @ self.observation.T @ np.linalg.inv(innovation_cov)
            cov_upd = (eye - gain @ self.observation) @ cov
            cov_next = self.transition @ cov_upd @ self.transition.T + self.state_cov
            if np.max(np.abs(cov_next - cov)) < tol:
                cov = cov_next
                break
            cov = cov_next
        innovation_cov = self.observation @ cov @ self.observation.T + self.obs_cov
        return cov @ self.observation.T @ np.linalg.inv(innovation_cov)


def muth_gain(noise_ratio: float) -> float:
    """EWMA memory weight implied by the Muth model's noise ratio.

    In the Muth model a random-walk state is observed with noise; the stationary
    Kalman filter reduces to an EWMA whose weight on the *past* estimate depends
    only on ``kappa = tau_eta / tau_eps`` (observation-noise over state-noise
    standard deviations). A noisy observation (large ``kappa``) means the new
    reading is trusted less, so the filter has long memory (weight -> 1); a
    volatile state (small ``kappa``) discounts the past quickly (weight -> 0).

    Parameters
    ----------
    noise_ratio : float
        The ratio ``kappa = tau_eta / tau_eps`` (>= 0).

    Returns
    -------
    float
        The EWMA weight ``K`` on the previous estimate, in ``(0, 1)``. This is
        ``1 - gain``, where ``gain`` is the Kalman weight on the new observation.
    """
    kappa = float(noise_ratio)
    state_var_ratio = (1.0 + np.sqrt(4.0 * kappa**2 + 1.0)) / 2.0  # sigma_hat^2 / tau_eps^2
    return kappa**2 / (state_var_ratio + kappa**2)
