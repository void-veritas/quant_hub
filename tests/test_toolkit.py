"""Property tests for the quant toolkit.

Check mathematical invariants, not just that code runs. Grouped by module cluster.
"""

import numpy as np
import pytest

from quant_hub.alpha import funding_signals as fund
from quant_hub.alpha import momentum as mom
from quant_hub.alpha import signals
from quant_hub.attribution import factor_attribution as fattr
from quant_hub.attribution import performance as perf
from quant_hub.backtest import metrics as m
from quant_hub.backtest import multiple_testing as mt
from quant_hub.costs import slippage
from quant_hub.portfolio import construction as con
from quant_hub.portfolio import sizing
from quant_hub.risk import covariance as cov
from quant_hub.risk import drawdown, hedging, leverage
from quant_hub.risk.factor_model import FactorModel, cross_sectional_regression, winsorize_returns
from quant_hub.risk.volatility import GARCH11, ewma_variance, roll_effective_spread
from quant_hub.utils.state_space import KalmanFilter, muth_gain


@pytest.fixture
def factor_setup():
    rng = np.random.default_rng(0)
    n, t, k = 30, 400, 3
    B = rng.standard_normal((n, k))
    f = rng.standard_normal((t, k)) * 0.02
    eps = rng.standard_normal((t, n)) * 0.01
    R = f @ B.T + eps
    return B, f, eps, R


# --- volatility ---------------------------------------------------------------


def test_ewma_variance_tracks_level():
    rng = np.random.default_rng(1)
    r = rng.standard_normal(3000) * 0.02
    v = np.asarray(ewma_variance(r, halflife=50))
    assert abs(np.sqrt(v[-500:].mean()) - 0.02) < 0.004


def test_roll_spread_recovers_bounce():
    rng = np.random.default_rng(2)
    efficient = 100 + np.cumsum(rng.standard_normal(20000) * 0.05)
    half_spread = 0.10
    observed = efficient + half_spread * rng.choice([-1.0, 1.0], size=len(efficient))
    # Roll model recovers the full spread (2 * half_spread) from the bid-ask bounce
    assert abs(roll_effective_spread(observed) - 2 * half_spread) < 0.03
    # a pure random walk (no bounce) implies a far smaller spread
    assert roll_effective_spread(efficient) < half_spread


def test_garch_recovers_persistence():
    rng = np.random.default_rng(3)
    t, omega, alpha, beta = 6000, 1e-6, 0.08, 0.90
    r = np.zeros(t)
    h2 = np.full(t, omega / (1 - alpha - beta))
    for i in range(1, t):
        h2[i] = omega + alpha * r[i - 1] ** 2 + beta * h2[i - 1]
        r[i] = np.sqrt(h2[i]) * rng.standard_normal()
    g = GARCH11().fit(r)
    assert 0.9 < g.alpha + g.beta < 0.999
    assert g.kurtosis() > 3
    # multi-step forecasts mean-revert toward the unconditional variance
    uncond = g.unconditional_variance()
    fc = g.forecast(100)
    assert abs(fc[-1] - uncond) < abs(fc[0] - uncond)


# --- state space --------------------------------------------------------------


def test_kalman_tracks_constant_state():
    rng = np.random.default_rng(4)
    true_x = 5.0
    ys = true_x + rng.standard_normal(500) * 0.5
    kf = KalmanFilter(
        transition=1.0,
        observation=1.0,
        state_cov=1e-6,
        obs_cov=0.25,
        initial_state=0.0,
        initial_cov=1.0,
    )
    states, _ = kf.filter(ys)
    assert abs(states[-1, 0] - true_x) < 0.2


def test_muth_gain_in_unit_interval():
    for kappa in (0.1, 1.0, 5.0, 20.0):
        assert 0.0 < muth_gain(kappa) < 1.0
    assert muth_gain(20.0) > muth_gain(0.1)  # noisier obs -> longer memory


# --- covariance / factor model ------------------------------------------------


def test_pca_explains_variance(factor_setup):
    _, _, _, R = factor_setup
    _, _, ev = cov.pca_factor_model(R, 3)
    assert ev.sum() > 0.6  # 3 real factors dominate


def test_ppca_reconstructs_covariance(factor_setup):
    _, _, _, R = factor_setup
    B, of, s2 = cov.ppca(R, 3)
    reconstructed = B @ of @ B.T + s2 * np.eye(R.shape[1])
    assert np.allclose(reconstructed, np.cov(R, rowvar=False), atol=5e-4)


def test_ledoit_wolf_psd_and_intensity(factor_setup):
    _, _, _, R = factor_setup
    shrunk, intensity = cov.ledoit_wolf_shrinkage(R)
    assert 0.0 <= intensity <= 1.0
    assert np.min(np.linalg.eigvalsh(shrunk)) > 0  # positive definite


def test_procrustes_recovers_rotation():
    rng = np.random.default_rng(5)
    B = rng.standard_normal((20, 3))
    q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    rotation, aligned = cov.procrustes_rotation(B, B @ q)
    assert np.allclose(aligned, B, atol=1e-8)


def test_fmp_unit_exposure_and_zero_residual_pnl(factor_setup):
    B, _, _, R = factor_setup
    fm = FactorModel(B, np.eye(3) * 4e-4, np.full(B.shape[0], 1e-4))
    P = fm.factor_mimicking_portfolios()
    assert np.allclose(fm.loadings.T @ P, np.eye(3), atol=1e-8)
    # residuals from the cross-sectional regression are orthogonal to FMP weights
    fhat, resid, fmp = cross_sectional_regression(R[0], B, idio_var=fm.idio_var)
    assert np.allclose(fmp.T @ resid, 0.0, atol=1e-8)


def test_factor_model_variance_decomposition(factor_setup):
    B, _, _, _ = factor_setup
    fm = FactorModel(B, np.eye(3) * 4e-4, np.full(B.shape[0], 1e-4))
    w = np.full(B.shape[0], 1.0 / B.shape[0])
    assert np.isclose(fm.portfolio_variance(w), fm.factor_variance(w) + fm.idio_variance(w))
    assert 0.0 <= fm.pct_idio_variance(w) <= 1.0
    assert np.isclose(w @ fm.asset_covariance() @ w, fm.portfolio_variance(w))


def test_winsorize_flags_jumps():
    r = np.concatenate([np.full(99, 0.01), [0.9]])
    w, flagged = winsorize_returns(r, d_max=8.0)
    assert flagged[-1] and not flagged[:-1].any()
    assert w[-1] < 0.9


# --- metrics ------------------------------------------------------------------


def test_sharpe_and_confidence_interval():
    rng = np.random.default_rng(6)
    r = rng.standard_normal(2000) * 0.01 + 0.0005
    sr = m.sharpe_ratio(r)
    lo, hi = m.sharpe_confidence_interval(sr, len(r))
    assert lo < sr < hi
    assert np.isclose(m.t_statistic(r), sr * np.sqrt(len(r)), rtol=1e-6)


def test_cantelli_bound_known_value():
    assert abs(m.cantelli_loss_bound(2.0, 3.0) - 1.0 / 26.0) < 1e-9


def test_qlike_minimized_at_truth():
    rng = np.random.default_rng(7)
    rv = np.abs(rng.standard_normal(500)) + 0.5
    assert abs(m.qlike(rv, rv)) < 1e-9
    assert m.qlike(rv, rv * 1.5) > 0


def test_calibration_curve_on_calibrated_data():
    rng = np.random.default_rng(8)
    p = rng.uniform(0, 1, 20000)
    y = (rng.uniform(0, 1, 20000) < p).astype(float)
    curve = m.calibration_curve(p, y, n_bins=10)
    assert np.allclose(curve["mean_pred"], curve["mean_obs"], atol=0.03)


def test_max_drawdown():
    eq = np.array([1.0, 1.2, 0.9, 1.1, 0.6])
    assert np.isclose(m.max_drawdown(eq), 0.6 / 1.2 - 1.0)


# --- signals ------------------------------------------------------------------


def test_orthogonalize_removes_factor(factor_setup):
    B, _, _, _ = factor_setup
    signal = B @ np.array([1.0, -2.0, 0.5]) + 0.01  # perfectly spanned + tiny const
    resid = signals.orthogonalize(signal, B)
    assert np.allclose(B.T @ resid, 0.0, atol=1e-8)


def test_ic_to_sharpe():
    assert np.isclose(signals.ic_to_sharpe(0.05, 400), 0.05 * 20)


def test_zscore_properties():
    z = signals.zscore(np.array([1.0, 2, 3, 4, 5]))
    assert abs(z.mean()) < 1e-12 and abs(z.std(ddof=1) - 1.0) < 1e-9


# --- construction / sizing ----------------------------------------------------


def test_mvo_realizes_max_sharpe(factor_setup):
    B, _, _, _ = factor_setup
    fm = FactorModel(B, np.eye(3) * 4e-4, np.full(B.shape[0], 1e-4))
    omega = fm.asset_covariance()
    alpha = np.linspace(-0.02, 0.02, B.shape[0])
    w = con.mean_variance_weights(alpha, omega, vol_target=0.1)
    realized_sr = (w @ alpha) / np.sqrt(w @ omega @ w)
    assert np.isclose(realized_sr, con.max_sharpe(alpha, omega), rtol=1e-6)
    assert np.isclose(np.sqrt(w @ omega @ w), 0.1, rtol=1e-6)  # hits vol target


def test_factor_neutral_weights_zero_exposure(factor_setup):
    B, _, _, _ = factor_setup
    alpha = np.linspace(-1, 1, B.shape[0])
    w = con.factor_neutral_weights(alpha, B, gross=1.0)
    assert np.allclose(B.T @ w, 0.0, atol=1e-8)
    assert np.isclose(np.sum(np.abs(w)), 1.0)


def test_uncorrelated_mvo_dollar_vol_proportional_to_sharpe():
    s = np.array([0.5, 1.0, 1.5])
    v = con.mvo_vol_correlation_form(s, np.eye(3))
    assert np.allclose(v, s)


def test_kelly_binary_two_to_one_coin():
    assert np.isclose(sizing.kelly_binary(0.5, 1.0, 0.5), 0.5)


def test_kelly_uncertainty_shrinks():
    assert sizing.kelly_with_uncertainty(0.1, 0.2, 0.05) < sizing.kelly_fraction(0.1, 0.2)


def test_vol_target_scale():
    cov_m = np.array([[0.04, 0.0], [0.0, 0.09]])
    w = sizing.vol_target_scale(np.array([1.0, 1.0]), cov_m, target_vol=0.1)
    assert np.isclose(np.sqrt(w @ cov_m @ w), 0.1)


# --- risk management ----------------------------------------------------------


def test_leverage_band_feasibility():
    floor, ceiling, feasible = leverage.leverage_band(
        min_return=0.15, max_drawdown=0.2, n_positions=100, idio_vol=0.2, sharpe=1.5
    )
    assert feasible == (floor <= ceiling)
    assert floor > 0 and ceiling > 0


def test_market_variance_share_monotone():
    assert leverage.market_variance_share(3.0, 0.5) < leverage.market_variance_share(1.0, 0.5)


def test_hedge_ratio_recovers_beta():
    rng = np.random.default_rng(9)
    r_h = rng.standard_normal(5000) * 0.02
    beta = 1.3
    r_c = beta * r_h + rng.standard_normal(5000) * 0.01
    h = hedging.hedge_ratio(np.cov(r_c, r_h)[0, 1], np.var(r_h))
    assert abs(h - (-beta)) < 0.05


def test_factor_hedge_neutralizes(factor_setup):
    B, _, _, _ = factor_setup
    fm = FactorModel(B, np.eye(3) * 4e-4, np.full(B.shape[0], 1e-4))
    w = np.linspace(-1, 1, B.shape[0]) / B.shape[0]
    overlay = hedging.factor_hedge_overlay(fm, w)
    assert np.allclose(fm.exposures(w + overlay), 0.0, atol=1e-8)


def test_grossman_zhou_liquidates_at_limit():
    assert drawdown.grossman_zhou_fraction(0.1, 0.2, 0.2, 0.2) == 0.0  # d == D
    assert drawdown.grossman_zhou_fraction(0.1, 0.2, 0.2, 0.0) > 0


# --- attribution --------------------------------------------------------------


def test_effective_breadth_bounds():
    assert np.isclose(perf.effective_breadth(np.ones(50)), 50)
    assert np.isclose(perf.effective_breadth(np.array([1.0, 0, 0, 0])), 1.0)


def test_selection_sizing_timing_sums_to_total():
    rng = np.random.default_rng(10)
    w = rng.standard_normal((100, 20))
    eps = rng.standard_normal((100, 20)) * 0.01
    d = perf.selection_sizing_timing(w, eps)
    assert np.isclose(d["sizing"] + d["timing"] + d["selection"], d["total_idio_pnl"])


def test_factor_idio_pnl_adds_up(factor_setup):
    B, f, eps, _ = factor_setup
    rng = np.random.default_rng(11)
    w = rng.standard_normal((f.shape[0], B.shape[0])) / B.shape[0]
    fac, idio, per_factor = fattr.factor_idio_pnl(w, f, eps, B)
    assert np.isclose(per_factor.sum(), fac)
    # total position PnL = sum of asset returns * weights
    asset_returns = f @ B.T + eps
    assert np.isclose(fac + idio, np.sum(w * asset_returns))


# --- multiple testing ---------------------------------------------------------


def test_rademacher_haircut_reduces_sharpe():
    rng = np.random.default_rng(12)
    x = rng.standard_normal((250, 100))  # 100 useless strategies
    best = x.mean(axis=0).max()  # best in-sample per-period Sharpe
    haircut = mt.rademacher_haircut_sharpe(best, x, seed=0)
    assert haircut < best
    assert mt.rademacher_complexity(x, seed=0) > 0


# --- slippage -----------------------------------------------------------------


def test_square_root_impact_scales():
    assert np.isclose(slippage.square_root_impact(0.02, 0.04), 0.02 * 0.2)
    assert slippage.square_root_impact(0.02, 0.16) == 2 * slippage.square_root_impact(0.02, 0.04)


# --- crypto alpha -------------------------------------------------------------


def test_momentum_skip():
    import pandas as pd

    prices = pd.Series(np.linspace(100, 200, 30))
    mo = mom.momentum(prices, lookback=10, skip=1)
    assert mo.iloc[-1] > 0  # uptrend


def test_funding_carry_annualizes():
    # 0.01% hourly funding on Hyperliquid -> ~87.6% annualized
    assert np.isclose(fund.annualize_funding(0.0001, 1.0), 0.0001 * 24 * 365)
