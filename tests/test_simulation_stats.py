import numpy as np

from quant_hub.backtest import metrics, simulation


def test_mertens_reduces_to_lo_for_normal_and_psr_dsr_order():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 2000)
    sr = x.mean() / x.std(ddof=1)
    lo = metrics.sharpe_standard_error(sr, len(x))
    me = metrics.sharpe_standard_error_mertens(x)
    assert abs(lo - me) / lo < 0.05
    psr = metrics.probabilistic_sharpe_ratio(x)
    dsr = metrics.deflated_sharpe_ratio(x, n_trials=30, trial_sharpe_variance=0.02**2)
    assert psr > 0.99 and dsr["sr0"] > 0 and dsr["dsr"] < psr
    lo_ci, hi_ci = metrics.block_bootstrap_sharpe_ci(x, block=20, n_boot=500)
    ann = sr * np.sqrt(365)
    assert lo_ci < ann < hi_ci
    # fat left tail lowers the PSR relative to a normal sample with the same SR
    y = np.concatenate([x[:-20], np.full(20, -0.08)])
    y = y - y.mean() + x.mean() * (y.std(ddof=1) / x.std(ddof=1))
    assert metrics.return_moments(y)["skew"] < -0.5
    assert metrics.probabilistic_sharpe_ratio(y) <= psr


def test_paths_preserve_moments_and_state_conditioning_concentrates():
    rng = np.random.default_rng(1)
    T = 1000
    vol = np.where(np.arange(T) < 500, 0.01, 0.03)  # two regimes
    r = rng.normal(0.0005, vol)
    paths = simulation.block_paths(r, horizon=365, n_paths=2000, block=20, seed=2)
    assert paths.shape == (2000, 365, 1)
    st = simulation.path_statistics(paths)
    summ = simulation.summarise_paths(st)
    assert 0.0 <= summ.loc["total_return", "P(loss)"] <= 1.0
    chk = simulation.moments_check(paths, r)
    assert abs(chk.loc["simulated", "vol"] / chk.loc["history", "vol"] - 1) < 0.1
    # condition on the high-vol state: simulated vol approaches 0.03, not the 0.022 mixture
    p = simulation.state_probabilities(vol, target=0.03, bandwidth=0.002)
    assert simulation.effective_history(p) < 600
    hp = simulation.block_paths(r, horizon=365, n_paths=2000, block=20, p=p, seed=3)
    assert simulation.path_statistics(hp)["vol_ann"].median() > 0.025 * np.sqrt(365)
