"""Small, dependency-light clustering for reducing and explaining a pivot.

* :func:`cluster_rows` groups the *rows of a pivot table* by the shape of their column
  profile (which ports behave alike across actions?).
* :func:`cluster_frame` clusters the *records* of a frame on numeric columns and returns a
  label column that can be used as a pivot dimension.
* :func:`cocluster` groups rows **and** columns of a matrix together (spectral
  co-clustering, or Markov clustering on the bipartite graph), so the heatmap can be
  reordered into blocks.

Methods: ``kmeans`` (k-means++, silhouette-chosen k), ``dbscan`` (density, auto eps,
noise label), ``hdbscan`` (scikit-learn's or the ``hdbscan`` package when installed,
else DBSCAN), ``spectral`` and ``mcl`` for co-clustering. Everything but HDBSCAN is numpy.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from ._log import log

NOISE = "noise"
METHODS = ("kmeans", "dbscan", "hdbscan", "gmm")
COMETHODS = ("spectral", "mcl")


def _kmeans_pp_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = X.shape[0]
    centers = np.empty((k, X.shape[1]), dtype=float)
    centers[0] = X[rng.integers(n)]
    d2 = ((X - centers[0]) ** 2).sum(axis=1)
    for i in range(1, k):
        total = d2.sum()
        if total <= 0:
            centers[i:] = X[rng.integers(n, size=k - i)]
            break
        idx = rng.choice(n, p=d2 / total)
        centers[i] = X[idx]
        d2 = np.minimum(d2, ((X - centers[i]) ** 2).sum(axis=1))
    return centers


def kmeans(
    X: np.ndarray, k: int, *, n_init: int = 4, max_iter: int = 60, seed: int = 0, tol: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Lloyd's k-means with k-means++ seeding. Returns ``(labels, centers, inertia)``."""
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    k = max(1, min(int(k), n))
    rng = np.random.default_rng(seed)
    best: Optional[Tuple[np.ndarray, np.ndarray, float]] = None
    for _ in range(n_init):
        centers = _kmeans_pp_init(X, k, rng)
        labels = np.zeros(n, dtype=int)
        for _ in range(max_iter):
            d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels = d2.argmin(axis=1)
            new = centers.copy()
            for j in range(k):
                m = labels == j
                if m.any():
                    new[j] = X[m].mean(axis=0)
                else:  # empty cluster: re-seed on the farthest point
                    new[j] = X[d2.min(axis=1).argmax()]
            shift = float(((new - centers) ** 2).sum())
            centers = new
            if shift <= tol:
                break
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = d2.argmin(axis=1)
        inertia = float(d2[np.arange(n), labels].sum())
        if best is None or inertia < best[2]:
            best = (labels, centers, inertia)
    assert best is not None
    return best


def silhouette(X: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> float:
    """Simplified (centroid-based) silhouette in [-1, 1]; fast and good enough to pick k."""
    k = centers.shape[0]
    if k < 2:
        return -1.0
    d = np.sqrt(((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))
    a = d[np.arange(len(X)), labels]
    d_other = d.copy()
    d_other[np.arange(len(X)), labels] = np.inf
    b = d_other.min(axis=1)
    denom = np.maximum(a, b)
    s = np.where(denom > 0, (b - a) / np.where(denom > 0, denom, 1), 0.0)
    return float(s.mean())


def choose_k(X: np.ndarray, k_max: int = 8, *, seed: int = 0, sample: int = 4000) -> int:
    """Best k in ``2..k_max`` by silhouette (1 when the data has no spread)."""
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if n < 4 or np.allclose(X, X[0]):
        return 1
    rng = np.random.default_rng(seed)
    S = X[rng.choice(n, size=min(n, sample), replace=False)] if n > sample else X
    best_k, best_s = 1, -1.0
    for k in range(2, min(k_max, len(np.unique(S, axis=0))) + 1):
        labels, centers, _ = kmeans(S, k, n_init=2, seed=seed)
        s = silhouette(S, labels, centers)
        if s > best_s + 1e-9:
            best_k, best_s = k, s
    return best_k


def standardize(X: np.ndarray, *, log_skewed: bool = True) -> np.ndarray:
    """Column-wise z-scores; heavily right-skewed non-negative columns are log1p'ed first."""
    X = np.asarray(X, dtype=float).copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        finite = col[np.isfinite(col)]
        if finite.size and log_skewed and finite.min() >= 0:
            p50, p95 = np.percentile(finite, [50, 95])
            if p95 > 20 * max(p50, 1e-12):
                col = np.log1p(col)
        mu = np.nanmean(col) if np.isfinite(col).any() else 0.0
        sd = np.nanstd(col) if np.isfinite(col).any() else 1.0
        col = np.where(np.isfinite(col), col, mu)
        X[:, j] = (col - mu) / (sd if sd > 0 else 1.0)
    return X


def _pairwise(A: np.ndarray, B: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """Euclidean distances via |a|^2 + |b|^2 - 2 a.b (a matmul, chunked over rows of A)."""
    A = np.ascontiguousarray(A, dtype=float)
    B = np.ascontiguousarray(B, dtype=float)
    b2 = (B * B).sum(axis=1)
    out = np.empty((A.shape[0], B.shape[0]), dtype=float)
    for i in range(0, A.shape[0], chunk):
        a = A[i : i + chunk]
        d2 = (a * a).sum(axis=1)[:, None] + b2[None, :] - 2.0 * (a @ B.T)
        out[i : i + chunk] = np.sqrt(np.maximum(d2, 0.0))
    return out


def auto_eps(X: np.ndarray, min_samples: int = 5) -> float:
    """DBSCAN radius from the knee of the sorted k-nearest-neighbour distance curve."""
    n = X.shape[0]
    k = min(max(2, min_samples), n - 1)
    D = _pairwise(X, X)
    kd = np.sort(np.partition(D, k, axis=1)[:, k])
    if kd[-1] <= 0:
        return 1e-9
    # knee: farthest point from the chord between the curve's ends
    x = np.linspace(0, 1, n)
    y = (kd - kd[0]) / (kd[-1] - kd[0] + 1e-12)
    dist = np.abs(y - x) / math.sqrt(2)
    knee = int(np.argmax(dist))
    return float(max(kd[knee], np.percentile(kd, 10), 1e-9))


def dbscan(X: np.ndarray, eps: Optional[float] = None, min_samples: int = 5, *, max_points: int = 4000, seed: int = 0) -> np.ndarray:
    """DBSCAN labels (``-1`` = noise). Larger inputs are clustered on a sample and the
    rest assigned to the nearest core point within ``eps``."""
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if n == 0:
        return np.array([], dtype=int)
    rng = np.random.default_rng(seed)
    idx = np.arange(n) if n <= max_points else np.sort(rng.choice(n, max_points, replace=False))
    S = X[idx]
    m = S.shape[0]
    if eps is None:
        eps = auto_eps(S, min_samples) if m > min_samples else 1.0
    D = _pairwise(S, S)
    neigh = D <= eps
    core = neigh.sum(axis=1) >= min_samples
    labels = np.full(m, -1, dtype=int)
    cid = 0
    for i in range(m):
        if labels[i] != -1 or not core[i]:
            continue
        labels[i] = cid
        stack = [i]
        while stack:
            j = stack.pop()
            if not core[j]:
                continue
            for q in np.nonzero(neigh[j])[0]:
                if labels[q] == -1:
                    labels[q] = cid
                    if core[q]:
                        stack.append(int(q))
        cid += 1
    if m == n:
        return labels
    out = np.full(n, -1, dtype=int)
    out[idx] = labels
    core_idx = np.nonzero(core & (labels >= 0))[0]
    rest = np.setdiff1d(np.arange(n), idx)
    if core_idx.size and rest.size:
        Dr = _pairwise(X[rest], S[core_idx])
        nearest = Dr.argmin(axis=1)
        ok = Dr[np.arange(rest.size), nearest] <= eps
        out[rest[ok]] = labels[core_idx[nearest[ok]]]
    return out


def hdbscan_labels(X: np.ndarray, min_cluster_size: Optional[int] = None, *, seed: int = 0) -> Tuple[np.ndarray, str]:
    """HDBSCAN labels via scikit-learn or the ``hdbscan`` package; DBSCAN when neither is installed.

    Returns ``(labels, backend)`` where backend names what actually ran.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    mcs = int(min_cluster_size or max(5, min(50, round(math.sqrt(n)))))
    try:
        from sklearn.cluster import HDBSCAN  # type: ignore

        return np.asarray(HDBSCAN(min_cluster_size=min(mcs, max(2, n // 2))).fit_predict(X)), "sklearn.cluster.HDBSCAN"
    except Exception:  # noqa: BLE001 - not installed or too few points
        pass
    try:
        import hdbscan  # type: ignore

        return np.asarray(hdbscan.HDBSCAN(min_cluster_size=min(mcs, max(2, n // 2))).fit_predict(X)), "hdbscan.HDBSCAN"
    except Exception:  # noqa: BLE001
        pass
    log.info("cluster", "hdbscan not installed (pip install scikit-learn); using DBSCAN with an automatic radius")
    return dbscan(X, min_samples=max(3, min(mcs, 10)), seed=seed), "dbscan (fallback)"


def gmm_labels(X: np.ndarray, k: Optional[int], *, seed: int = 0) -> np.ndarray:
    """Hard assignment (highest-responsibility component) from a fitted Gaussian mixture."""
    from ._mixture import choose_gmm_k, fit_gmm

    kk = k if k is not None else choose_gmm_k(X, k_max=8, seed=seed)
    return fit_gmm(X, max(1, kk), seed=seed).predict(X)


def cluster_labels(X: np.ndarray, k: Optional[int], method: str = "kmeans", *, seed: int = 0) -> Tuple[np.ndarray, str]:
    """Dispatch to a method; returns ``(labels, backend)``. ``-1`` marks noise."""
    if method == "kmeans":
        kk = k if k is not None else choose_k(X, seed=seed)
        return kmeans(X, kk, seed=seed)[0], "kmeans"
    if method == "dbscan":
        return dbscan(X, min_samples=max(3, k or 5), seed=seed), "dbscan"
    if method == "hdbscan":
        return hdbscan_labels(X, k, seed=seed)
    if method == "gmm":
        return gmm_labels(X, k, seed=seed), "gmm"
    raise ValueError(f"unknown method {method!r}; use one of {METHODS}")


def cluster_frame(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    k: Optional[int] = None,
    *,
    method: str = "kmeans",
    seed: int = 0,
    name: str = "cluster",
) -> pd.Series:
    """Cluster the records of ``df`` on numeric ``columns`` (default: all numeric).

    Returns an ordered categorical Series of labels ``c1, c2 ...`` numbered by cluster
    size (largest first), each carrying its member count, aligned to ``df.index``. With
    density methods a ``noise (n=...)`` label collects the outliers.
    """
    if columns is None:
        columns = [c for c in df.columns if pdt.is_numeric_dtype(df[c]) and not pdt.is_bool_dtype(df[c])]
    columns = list(columns)
    if not columns:
        raise ValueError("cluster_frame needs at least one numeric column")
    X = np.column_stack([pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float) for c in columns])
    Z = standardize(X)
    with log.step("cluster", f"{method} on {len(df):,} records x {len(columns)} columns") as st:
        labels, backend = cluster_labels(Z, k, method, seed=seed)
        st.detail += f" -> {len(set(labels) - {-1})} clusters ({backend})"
    return _label_series(labels, df.index, name)


def _label_series(labels: np.ndarray, index: pd.Index, name: str, unit: str = "n") -> pd.Series:
    labels = np.asarray(labels)
    counts = pd.Series(labels[labels >= 0]).value_counts()
    order = {old: i + 1 for i, old in enumerate(counts.index)}
    text = [f"c{order[l]} ({unit}={counts[l]:,})" for l in counts.index]
    remap = {old: text[i] for i, old in enumerate(counts.index)}
    n_noise = int((labels < 0).sum())
    if n_noise:
        noise_label = f"{NOISE} ({unit}={n_noise:,})"
        text.append(noise_label)
        remap[-1] = noise_label
    out = pd.Series([remap[l] for l in labels], index=index, dtype=object, name=name)
    return pd.Series(pd.Categorical(out, categories=text, ordered=True), index=index, name=name)


def _row_features(table: pd.DataFrame, normalize: str) -> np.ndarray:
    X = table.to_numpy(dtype=float)
    X = np.where(np.isfinite(X), X, 0.0)
    if X.shape[1] == 0 or X.shape[0] == 0:
        raise ValueError("cannot cluster an empty table")
    if normalize == "row":
        sums = np.abs(X).sum(axis=1, keepdims=True)
        Z = np.where(sums > 0, X / np.where(sums > 0, sums, 1), 0.0)
        mag = np.log1p(np.abs(X).sum(axis=1))
        return np.column_stack([Z, (mag - mag.mean()) / (mag.std() or 1.0) * 0.5]) if X.shape[1] > 1 else standardize(X)
    if normalize == "column":
        return standardize(X)
    return np.log1p(np.abs(X)) * np.sign(X) if (X >= 0).all() and X.max() > 1000 else X


def cluster_rows(
    table: pd.DataFrame, k: Optional[int] = None, *, method: str = "kmeans", seed: int = 0, normalize: str = "row"
) -> Tuple[pd.Series, List[int]]:
    """Cluster the rows of a pivot table by the shape of their column profile.

    ``normalize="row"`` compares proportions (each row scaled to sum 1, so a big and a
    small row with the same mix cluster together); ``"column"`` uses z-scores per column;
    ``"none"`` uses raw values (log1p'ed when skewed). ``method`` is ``kmeans``,
    ``dbscan`` or ``hdbscan`` (density methods may label rows ``noise``).

    Returns ``(labels aligned to table.index, row order grouping similar rows)``.
    """
    Z = _row_features(table, normalize)
    with log.step("cluster", f"{method} on {len(table)} pivot rows") as st:
        if method == "kmeans":
            if k is None:
                k = choose_k(Z, k_max=min(8, max(2, len(table) // 2)), seed=seed)
            labels, centers, _ = kmeans(Z, k, seed=seed)
            d = np.sqrt(((Z - centers[labels]) ** 2).sum(axis=1))
        else:
            labels, backend = cluster_labels(Z, k, method, seed=seed)
            d = np.zeros(len(table))
            for c in set(labels):
                m = labels == c
                d[m] = np.sqrt(((Z[m] - Z[m].mean(axis=0)) ** 2).sum(axis=1))
        st.detail += f" -> {len(set(labels) - {-1})} clusters"
    ser = _label_series(labels, table.index, "cluster", unit="rows")
    codes = ser.cat.codes.to_numpy()
    order = sorted(range(len(table)), key=lambda i: (codes[i], d[i]))
    return ser, order


# --------------------------------------------------------------------------- co-clustering


def spectral_cocluster(A: np.ndarray, k: Optional[int] = None, *, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Spectral co-clustering (Dhillon 2001): rows and columns of a non-negative matrix
    are embedded together from the singular vectors of the normalised matrix and
    k-means'd jointly. Returns ``(row_labels, col_labels)``."""
    A = np.abs(np.asarray(A, dtype=float))
    A = np.where(np.isfinite(A), A, 0.0) + 1e-9
    r, c = A.shape
    d1 = 1.0 / np.sqrt(A.sum(axis=1))
    d2 = 1.0 / np.sqrt(A.sum(axis=0))
    An = (A * d1[:, None]) * d2[None, :]
    U, s, Vt = np.linalg.svd(An, full_matrices=False)
    kmax = max(2, min(r, c, 8))
    if k is None:
        # eigengap: the block count is where the singular values drop the most
        tail = s[1 : kmax + 1]
        if tail.size >= 2:
            ratios = tail[:-1] / np.maximum(tail[1:], 1e-12)
            k = int(np.argmax(ratios)) + 2
        else:
            k = 2
    k = max(2, min(int(k), kmax))
    l = max(1, int(math.ceil(math.log2(k))))
    l = min(l, U.shape[1] - 1) if U.shape[1] > 1 else 1
    Zr = U[:, 1 : l + 1] * d1[:, None]
    Zc = Vt.T[:, 1 : l + 1] * d2[:, None]
    Z = np.vstack([Zr, Zc])
    labels, _, _ = kmeans(Z, k, seed=seed)
    return labels[:r], labels[r:]


def mcl(M: np.ndarray, *, inflation: float = 2.0, expansion: int = 2, iters: int = 100, tol: float = 1e-6, prune: float = 1e-5) -> np.ndarray:
    """Markov Cluster Algorithm on a (weighted, symmetric) adjacency matrix with self loops.

    Alternates expansion (random-walk chains: matrix powers) and inflation (element-wise
    powers that sharpen the walk) until the matrix stops changing; each node joins the
    attractor it flows to. Returns integer labels.
    """
    M = np.abs(np.asarray(M, dtype=float))
    n = M.shape[0]
    M = M + np.eye(n) * (M.max(axis=1, keepdims=True) if n else 1.0).clip(min=1e-9)
    M = M / M.sum(axis=0, keepdims=True)
    for _ in range(iters):
        prev = M
        M = np.linalg.matrix_power(M, expansion)
        M = M ** inflation
        M[M < prune] = 0.0
        colsum = M.sum(axis=0, keepdims=True)
        M = np.where(colsum > 0, M / np.where(colsum > 0, colsum, 1), 0.0)
        if np.abs(M - prev).max() < tol:
            break
    # clusters: every node joins the attractors it still flows to; nodes sharing an
    # attractor (and attractors sharing a node) are merged, as in the reference MCL
    attractors = np.nonzero(np.diag(M) > 1e-6)[0]
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for j in range(n):
        col = M[:, j]
        cand = attractors[col[attractors] > 1e-6] if attractors.size else np.array([], dtype=int)
        if cand.size == 0:
            cand = np.array([int(np.argmax(col))])
        for a in cand:
            union(int(a), j)
    roots = [find(i) for i in range(n)]
    seen: Dict[int, int] = {}
    return np.array([seen.setdefault(r, len(seen)) for r in roots], dtype=int)


def mcl_cocluster(A: np.ndarray, k: Optional[int] = None, *, inflation: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Markov clustering of the bipartite graph rows <-> columns weighted by the matrix.
    With ``k`` the inflation is searched so the cluster count comes closest to ``k``."""
    A = np.abs(np.asarray(A, dtype=float))
    A = np.where(np.isfinite(A), A, 0.0)
    r, c = A.shape
    scale = A.max() or 1.0
    B = np.zeros((r + c, r + c))
    B[:r, r:] = A / scale
    B[r:, :r] = A.T / scale
    if inflation is not None:
        labels = mcl(B, inflation=inflation)
    else:
        best: Optional[Tuple[float, np.ndarray]] = None
        for infl in (1.3, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0):
            labels = mcl(B, inflation=infl)
            n_cl = len(set(labels))
            score = abs(n_cl - k) if k is not None else abs(n_cl - min(max(2, min(r, c) // 3), 8))
            if best is None or score < best[0]:
                best = (score, labels)
            if score == 0:
                break
        labels = best[1] if best is not None else labels
    return labels[:r], labels[r:]


def cocluster(table: pd.DataFrame, k: Optional[int] = None, *, method: str = "spectral", seed: int = 0) -> Tuple[pd.Series, pd.Series]:
    """Group the rows and the columns of a pivot table into matching blocks.

    Returns ``(row_labels, col_labels)`` as ordered categoricals ``b1, b2 ...`` (by size).
    """
    if table.shape[0] < 2 or table.shape[1] < 2:
        raise ValueError("co-clustering needs at least 2 rows and 2 columns")
    A = table.to_numpy(dtype=float)
    with log.step("cocluster", f"{method} on {A.shape[0]} x {A.shape[1]}") as st:
        if method == "spectral":
            rl, cl = spectral_cocluster(A, k, seed=seed)
        elif method == "mcl":
            rl, cl = mcl_cocluster(A, k)
        else:
            raise ValueError(f"unknown co-clustering method {method!r}; use one of {COMETHODS}")
        st.detail += f" -> {len(set(rl) | set(cl))} blocks"
    both = np.concatenate([rl, cl])
    counts = pd.Series(both).value_counts()
    order = {old: i + 1 for i, old in enumerate(counts.index)}
    text = [f"b{order[l]}" for l in counts.index]
    remap = {old: text[i] for i, old in enumerate(counts.index)}
    rows = pd.Series(pd.Categorical([remap[l] for l in rl], categories=text, ordered=True), index=table.index, name="block")
    cols = pd.Series(pd.Categorical([remap[l] for l in cl], categories=text, ordered=True), index=table.columns, name="block")
    return rows, cols


__all__ = ["kmeans", "choose_k", "silhouette", "standardize", "dbscan", "auto_eps", "hdbscan_labels", "gmm_labels", "cluster_labels",
           "cluster_frame", "cluster_rows", "cocluster", "spectral_cocluster", "mcl", "mcl_cocluster", "METHODS", "COMETHODS", "NOISE"]
