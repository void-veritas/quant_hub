import numpy as np
import pandas as pd
from scipy import stats

from quant_hub.risk import tail


def test_cvar_matches_normal_theory_and_contributions_add_up():
    rng = np.random.default_rng(0)
    n = 200_000
    a = rng.normal(0.0, 0.01, n)
    b = rng.normal(0.0, 0.02, n)
    comp = pd.DataFrame({"a": a, "b": b})
    var95, cvar95 = tail.var_cvar(comp.sum(axis=1), 0.95)
    sd = np.sqrt(0.01**2 + 0.02**2)
    theory = sd * stats.norm.pdf(stats.norm.ppf(0.95)) / 0.05
    assert abs(cvar95 - theory) / theory < 0.02
    assert abs(var95 - sd * stats.norm.ppf(0.95)) / (sd * 1.645) < 0.02
    contrib = tail.euler_cvar_contributions(comp, 0.95)
    assert abs(contrib.sum() - cvar95) < 1e-12
    # independent normals: contributions proportional to variance shares
    assert abs(contrib["b"] / contrib.sum() - 0.8) < 0.03


def test_entropy_pooling_mean_view_shifts_the_mean_only_as_needed():
    rng = np.random.default_rng(1)
    R = pd.DataFrame({"f": rng.normal(0.001, 0.02, 5000), "g": rng.normal(0.0, 0.01, 5000)})
    q = tail.entropy_pooling(R, [tail.mean_view(R, "f", -0.003)])
    assert abs(q.sum() - 1) < 1e-12 and (q >= 0).all()
    assert abs(float(R["f"].to_numpy() @ q) + 0.003) < 1e-6
    # the untouched component keeps roughly its prior mean; the stress is not extreme
    assert abs(float(R["g"].to_numpy() @ q)) < 0.001
    assert tail.effective_scenarios(q) > 0.9 * len(R) * 0.9
    t = tail.stress_table(R, {"f_down": [tail.mean_view(R, "f", -0.003)]})
    assert t.loc["f_down", "mean_ann"] < t.loc["prior", "mean_ann"]
    assert t.loc["f_down", "top_cvar_source"] == "f"
