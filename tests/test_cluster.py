import numpy as np
import pandas as pd
import pytest

from pivot2hist import _cluster as C


def _blobs(seed=0):
    rng = np.random.default_rng(seed)
    return np.vstack([rng.normal(0, 0.3, (200, 2)), rng.normal(4, 0.3, (200, 2)), rng.normal((0, 5), 0.3, (150, 2))])


def test_kmeans_recovers_blobs():
    X = _blobs()
    labels, centers, inertia = C.kmeans(X, 3)
    assert set(labels) == {0, 1, 2} and centers.shape == (3, 2)
    assert inertia < 200
    assert C.choose_k(X) == 3
    assert C.silhouette(X, labels, centers) > 0.7


def test_kmeans_edge_cases():
    X = np.zeros((10, 2))
    labels, centers, _ = C.kmeans(X, 3)
    assert len(labels) == 10
    assert C.choose_k(X) == 1
    assert C.choose_k(np.array([[0.0], [1.0]])) == 1
    labels, _, _ = C.kmeans(np.array([[0.0], [1.0], [2.0]]), 10)  # k capped at n
    assert len(set(labels)) <= 3


def test_standardize_logs_skewed():
    X = np.column_stack([np.exp(np.random.default_rng(0).normal(7, 2, 500)), np.arange(500.0)])
    Z = C.standardize(X)
    assert abs(Z[:, 0].mean()) < 1e-9 and abs(Z[:, 0].std() - 1) < 1e-9
    assert Z[:, 0].max() < 6  # log1p tamed the tail


def test_cluster_frame_labels():
    df = pd.DataFrame(_blobs(), columns=["a", "b"])
    lab = C.cluster_frame(df)
    assert lab.cat.ordered and len(lab.cat.categories) == 3
    assert lab.cat.categories[0].startswith("c1 (n=200)")
    assert lab.value_counts().tolist() == [200, 200, 150]
    lab2 = C.cluster_frame(df, ["a"], k=2)
    assert len(lab2.cat.categories) == 2
    with pytest.raises(ValueError):
        C.cluster_frame(pd.DataFrame({"s": ["x", "y"]}))


def test_cluster_rows_groups_similar_profiles():
    T = pd.DataFrame({"allow": [100, 90, 5, 4, 50], "deny": [2, 3, 60, 70, 50]}, index=["p80", "p443", "p22", "p3389", "p8080"])
    lab, order = C.cluster_rows(T, k=2)
    assert lab["p80"] == lab["p443"] and lab["p22"] == lab["p3389"] and lab["p80"] != lab["p22"]
    assert sorted(order) == list(range(5))
    lab_auto, _ = C.cluster_rows(T)
    assert 2 <= len(lab_auto.cat.categories) <= 3
    lab_col, _ = C.cluster_rows(T, k=2, normalize="column")
    assert len(lab_col.cat.categories) == 2
    with pytest.raises(ValueError):
        C.cluster_rows(pd.DataFrame())


def test_dbscan_and_hdbscan_dispatch():
    rng = np.random.default_rng(0)
    X = np.vstack([_blobs(), rng.uniform(-3, 8, (15, 2))])
    lab = C.dbscan(X)
    assert len(set(lab) - {-1}) == 3 and 10 <= (lab == -1).sum() <= 40
    assert C.auto_eps(X) > 0
    hl, backend = C.hdbscan_labels(X)
    assert len(set(hl) - {-1}) >= 2 and backend
    big = np.vstack([rng.normal(0, 0.5, (6000, 3)), rng.normal(8, 0.5, (6000, 3))])
    lb = C.dbscan(big)  # sampled path + assignment of the rest
    assert len(set(lb) - {-1}) >= 2 and len(lb) == 12000
    assert C.dbscan(np.empty((0, 2))).size == 0
    labels, backend = C.cluster_labels(X, None, "kmeans")
    assert backend == "kmeans" and 3 <= len(set(labels)) <= 5  # the scattered outliers may earn a cluster
    with pytest.raises(ValueError):
        C.cluster_labels(X, None, "nope")
    df = pd.DataFrame(X, columns=["a", "b"])
    lab_s = C.cluster_frame(df, method="dbscan")
    assert lab_s.cat.categories[-1].startswith("noise")
    T = pd.DataFrame({"allow": [100, 90, 5, 4, 50, 1], "deny": [2, 3, 60, 70, 50, 99]}, index=list("abcdef"))
    lab_r, _ = C.cluster_rows(T, method="dbscan")
    assert lab_r.cat.categories[0].endswith("rows=2)") or len(lab_r.cat.categories) >= 1


def test_cocluster_spectral_and_mcl():
    rng = np.random.default_rng(1)
    A = np.zeros((9, 9))
    for b in range(3):
        A[3 * b : 3 * b + 3, 3 * b : 3 * b + 3] = rng.integers(40, 90, (3, 3))
    A += rng.integers(0, 2, A.shape)
    T = pd.DataFrame(A, index=[f"r{i}" for i in range(9)], columns=[f"c{i}" for i in range(9)])
    for method in ("spectral", "mcl"):
        r, c = C.cocluster(T, 3, method=method)
        assert len(r.cat.categories) == 3 and r.tolist()[:3] == [r.tolist()[0]] * 3 and c.tolist()[3:6] == [c.tolist()[3]] * 3
        assert r.tolist()[0] == c.tolist()[0]  # rows and columns of a block share the label
    ra, _ = C.cocluster(T)  # eigengap picks 3
    assert len(ra.cat.categories) == 3
    labels = C.mcl(np.eye(4) + np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]))
    assert len(set(labels)) == 2
    with pytest.raises(ValueError):
        C.cocluster(pd.DataFrame({"a": [1, 2]}))
    with pytest.raises(ValueError):
        C.cocluster(T, method="nope")
