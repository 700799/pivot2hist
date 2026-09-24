"""Small, dependency-free clustering: k-means++ with automatic k, for reducing a pivot.

Two uses:

* :func:`cluster_rows` groups the *rows of a pivot table* by the shape of their column
  profile (which ports behave alike across actions?), so a 40-row table can be shown as
  5 groups, or merely re-ordered so similar rows sit together.
* :func:`cluster_frame` clusters the *records* of a frame on numeric columns and returns a
  label column that can be used as a pivot dimension.

Everything is numpy; k is chosen by a centroid-based silhouette on a sample when not given.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pandas.api import types as pdt


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


def cluster_frame(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    k: Optional[int] = None,
    *,
    seed: int = 0,
    name: str = "cluster",
) -> pd.Series:
    """Cluster the records of ``df`` on numeric ``columns`` (default: all numeric).

    Returns an ordered categorical Series of labels ``c1, c2 ...`` numbered by cluster
    size (largest first), each carrying its member count, aligned to ``df.index``.
    """
    if columns is None:
        columns = [c for c in df.columns if pdt.is_numeric_dtype(df[c]) and not pdt.is_bool_dtype(df[c])]
    columns = list(columns)
    if not columns:
        raise ValueError("cluster_frame needs at least one numeric column")
    X = np.column_stack([pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float) for c in columns])
    Z = standardize(X)
    if k is None:
        k = choose_k(Z, seed=seed)
    labels, _, _ = kmeans(Z, k, seed=seed)
    return _label_series(labels, df.index, name)


def _label_series(labels: np.ndarray, index: pd.Index, name: str) -> pd.Series:
    counts = pd.Series(labels).value_counts()
    order = {old: i + 1 for i, old in enumerate(counts.index)}
    text = np.array([f"c{order[l]} (n={counts[l]:,})" for l in counts.index], dtype=object)
    remap = {old: text[i] for i, old in enumerate(counts.index)}
    out = pd.Series([remap[l] for l in labels], index=index, dtype=object, name=name)
    return pd.Series(pd.Categorical(out, categories=list(text), ordered=True), index=index, name=name)


def cluster_rows(
    table: pd.DataFrame, k: Optional[int] = None, *, seed: int = 0, normalize: str = "row"
) -> Tuple[pd.Series, List[int]]:
    """Cluster the rows of a pivot table by the shape of their column profile.

    ``normalize="row"`` compares proportions (each row scaled to sum 1, so a big and a
    small row with the same mix cluster together); ``"column"`` uses z-scores per column;
    ``"none"`` uses raw values (log1p'ed when skewed).

    Returns ``(labels aligned to table.index, row order grouping similar rows)``.
    """
    X = table.to_numpy(dtype=float)
    X = np.where(np.isfinite(X), X, 0.0)
    if X.shape[1] == 0 or X.shape[0] == 0:
        raise ValueError("cannot cluster an empty table")
    if normalize == "row":
        sums = np.abs(X).sum(axis=1, keepdims=True)
        Z = np.where(sums > 0, X / np.where(sums > 0, sums, 1), 0.0)
        mag = np.log1p(np.abs(X).sum(axis=1))
        Z = np.column_stack([Z, (mag - mag.mean()) / (mag.std() or 1.0) * 0.5]) if X.shape[1] > 1 else standardize(X)
    elif normalize == "column":
        Z = standardize(X)
    else:
        Z = np.log1p(np.abs(X)) * np.sign(X) if (X >= 0).all() and X.max() > 1000 else X
    if k is None:
        k = choose_k(Z, k_max=min(8, max(2, len(table) // 2)), seed=seed)
    labels, centers, _ = kmeans(Z, k, seed=seed)
    ser = _label_series(labels, table.index, "cluster")
    # order: clusters by size, rows within a cluster by distance to the centre
    d = np.sqrt(((Z - centers[labels]) ** 2).sum(axis=1))
    codes = ser.cat.codes.to_numpy()
    order = sorted(range(len(table)), key=lambda i: (codes[i], d[i]))
    return ser, order


__all__ = ["kmeans", "choose_k", "silhouette", "standardize", "cluster_frame", "cluster_rows"]
