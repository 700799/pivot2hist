import numpy as np
import pandas as pd
import pytest

import pivot2hist as p2h
from pivot2hist._mixture import GMMFit, choose_gmm_k, fit_gmm, mixture_cutpoints

rng = np.random.default_rng(1)


def test_recovers_bimodal():
    x = np.concatenate([rng.normal(200, 30, 1500), rng.normal(5000, 500, 1500)]).reshape(-1, 1)
    assert choose_gmm_k(x, k_max=8) == 2
    fit = fit_gmm(x, 2)
    means = sorted(c["mean"] for c in fit.components())
    assert abs(means[0] - 200) < 40 and abs(means[1] - 5000) < 200


def test_recovers_trimodal():
    x = np.concatenate([rng.normal(0, 1, 800), rng.normal(20, 1, 800), rng.normal(50, 2, 800)]).reshape(-1, 1)
    assert choose_gmm_k(x, k_max=6) == 3


def test_unimodal_picks_one_component():
    x = rng.normal(0, 1, 2000).reshape(-1, 1)
    assert choose_gmm_k(x, k_max=5) == 1
    fit = fit_gmm(x, 1)
    assert fit.k == 1 and abs(fit.components()[0]["mean"]) < 0.5


def test_multivariate_clustering():
    X = np.vstack([rng.normal([0, 0], 0.5, (300, 2)), rng.normal([6, 6], 0.5, (300, 2)), rng.normal([0, 6], 0.5, (300, 2))])
    k = choose_gmm_k(X, k_max=6)
    assert k == 3
    fit = fit_gmm(X, k)
    labels = fit.predict(X)
    sizes = sorted(np.bincount(labels).tolist())
    assert sizes == [300, 300, 300]
    resp = fit.responsibilities(X)
    assert np.allclose(resp.sum(axis=1), 1.0)


def test_components_sorted_by_mean_and_weights_sum_to_one():
    x = np.concatenate([rng.normal(0, 1, 600), rng.normal(50, 1, 400)]).reshape(-1, 1)
    fit = fit_gmm(x, 2)
    comps = fit.components()
    assert comps[0]["mean"] < comps[1]["mean"]
    assert abs(sum(c["weight"] for c in comps) - 1) < 1e-6
    assert repr(fit).startswith("<GMMFit")


def test_degenerate_constant_data():
    deg = np.ones((100, 1))
    assert choose_gmm_k(deg, k_max=5) == 1
    fit = fit_gmm(deg, 1)
    assert fit.k == 1 and abs(fit.components()[0]["mean"] - 1) < 1e-6


def test_mixture_cutpoints_between_modes():
    x = np.concatenate([rng.normal(10, 2, 1500), rng.normal(60, 3, 1000)]).reshape(-1, 1)
    fit = fit_gmm(x, 2)
    cuts = mixture_cutpoints(fit)
    assert len(cuts) == 1 and 20 < cuts[0] < 50
    with pytest.raises(ValueError):
        mixture_cutpoints(fit_gmm(np.vstack([rng.normal(size=(50, 2))]), 2))


def test_bin_rule_mixture_on_bimodal_column():
    bimodal = np.concatenate([rng.normal(200, 30, 1500), rng.normal(5000, 500, 1500)])
    df = pd.DataFrame({"v": bimodal, "k": rng.choice(list("ab"), 3000)})
    h = p2h.fit(df).histogram("v", bins="mixture")
    assert h.layout.rows[0].kind == "binned"
    edges = h.layout.rows[0].edges
    assert len(edges) - 1 <= 3  # roughly one bin per mode, not a dozen equal-width slices
    assert h.bins()["count"].sum() == len(df)


def test_bin_rule_mixture_registered_in_rules():
    assert "mixture" in p2h.RULES


def test_cluster_method_gmm(fw):
    v = p2h.fit(fw, max_rows=12, max_cols=5, agg="count")
    c = v.cluster(method="gmm")
    assert c.layout.rows[0].column == "cluster"
    assert c.pivot().to_numpy().sum() == len(fw)
    rec = v.cluster(on=["bytes", "duration"], method="gmm")
    assert rec.pivot().to_numpy().sum() == len(fw)
    assert "gmm" in p2h.METHODS


def test_view_modes_and_top_level(fw):
    v = p2h.fit(fw, max_rows=8, max_cols=4)
    m = v.modes("bytes")
    assert m is None or (isinstance(m, list) and all({"weight", "mean", "std"} <= set(c) for c in m))
    with pytest.raises(KeyError):
        v.modes("nope")
    m2 = p2h.modes(fw, "bytes")
    assert m2 is None or isinstance(m2, list)
    with pytest.raises(KeyError):
        p2h.modes(fw, "nope")
    tiny = p2h.fit(fw.head(3))
    assert tiny.modes("bytes") is None


def test_fit_gmm_caps_k_at_n_points():
    x = rng.normal(size=(3, 1))
    fit = fit_gmm(x, 10)
    assert fit.k <= 3


def test_gmm_performance_bounded():
    import time

    x = rng.normal(size=(20000, 1))
    t = time.time()
    fit_gmm(x, 3, n_init=4)
    assert time.time() - t < 5.0


def test_isinstance_gmmfit():
    fit = fit_gmm(rng.normal(size=(200, 1)), 1)
    assert isinstance(fit, GMMFit)
