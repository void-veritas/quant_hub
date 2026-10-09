"""Validation splits: disjointness, purging, CPCV path assembly, regime table."""

import numpy as np
import pandas as pd

from quant_hub.backtest.validation import (
    compare_variants,
    cpcv,
    group_sharpes,
    purged_kfold,
    regime_table,
    walk_forward,
)

DATES = pd.date_range("2024-01-01", periods=120, freq="D")


def test_walk_forward_windows_do_not_overlap_and_test_follows_train():
    splits = walk_forward(DATES, train=30, test=10)
    assert len(splits) == 9
    for tr, te in splits:
        assert tr.max() < te.min() and len(te) == 10 and len(tr) == 30


def test_purged_kfold_drops_rows_near_the_test_block():
    splits = purged_kfold(DATES, k=4, purge=3, embargo=2)
    assert len(splits) == 4
    for tr, te in splits:
        assert not np.intersect1d(tr, te).size
        assert not ((tr >= te.min() - 3) & (tr <= te.max() + 2)).any()


def test_cpcv_paths_cover_every_group_once():
    cv = cpcv(DATES, n_groups=6, n_test=2, purge=1, embargo=1)
    assert len(cv["splits"]) == 15 and cv["n_paths"] == 5
    for path in cv["paths"]:
        assert all(s is not None for s in path)
        for g, s_i in enumerate(path):
            assert g in cv["splits"][s_i]["test_groups"]
    for s in cv["splits"]:
        assert not np.intersect1d(s["train"], s["test"]).size


def test_group_sharpes_and_compare_variants_shape():
    rng = np.random.default_rng(0)
    r = pd.DataFrame(rng.normal(0.001, 0.01, (120, 3)), index=DATES, columns=["a", "b", "c"])
    cv = cpcv(DATES, n_groups=4, n_test=1)
    assert len(group_sharpes(r["a"], cv)) == 4
    t = compare_variants(r, n_groups=4, n_draws=200)
    assert list(t.index) == ["a", "b", "c"]
    assert (t["sharpe_after_haircut"] < t["sharpe"]).all()


def test_regime_table_splits_by_lagged_trailing_benchmark():
    bench = pd.Series(np.r_[np.full(60, 0.01), np.full(60, -0.01)], index=DATES)
    strat = pd.Series(0.001, index=DATES)
    t = regime_table(strat, bench, window=10)
    assert set(t.index) == {"bull", "bear"}
    assert t.loc["bull", "benchmark_ann_%"] > 0 > t.loc["bear", "benchmark_ann_%"]
