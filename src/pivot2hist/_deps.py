"""Dependency map: pairwise normalized mutual information between columns, no scipy.

Every column is discretized (numeric/datetime: quantile bins; categorical/boolean:
top-N + "(other)") and mutual information is computed on the joint histogram, with a
Miller-Madow bias correction (the same one used for auto-fit layout scoring in
:mod:`pivot2hist._fit`) so noisy high-cardinality pairs don't look falsely associated.
MI is normalized to ``[0, 1]`` (MI / min(H(a), H(b))) so pairs of columns with very
different cardinalities stay comparable.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from ._profile import CATEGORICAL, CONSTANT, DATETIME, ID, NUMERIC, Profile, profile


def _entropy(counts: np.ndarray) -> float:
    counts = counts[counts > 0]
    if counts.size == 0:
        return 0.0
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


def _discretize(s: pd.Series, kind: str, bins: int) -> np.ndarray:
    """Column values as small non-negative integer codes, -1 for missing."""
    if kind in (NUMERIC, DATETIME):
        x = pd.to_numeric(s, errors="coerce") if kind == NUMERIC else pd.to_datetime(s, errors="coerce").astype("int64")
        nun = x.nunique(dropna=True)
        if nun == 0:
            return np.full(len(s), -1, dtype=np.int64)
        try:
            codes = pd.qcut(x, min(max(bins, 2), nun), duplicates="drop").cat.codes.to_numpy()
        except (ValueError, IndexError):
            codes = pd.Categorical(x).codes
        return codes.astype(np.int64)
    vals = s.astype(str)
    top = vals.value_counts().index[: max(1, bins)]
    capped = vals.where(vals.isin(top), "(other)")
    return pd.Categorical(capped).codes.astype(np.int64)


def _pair_mi(a: np.ndarray, b: np.ndarray) -> float:
    """Mutual information between two discretized columns, normalized to [0, 1]."""
    mask = (a >= 0) & (b >= 0)
    a, b = a[mask], b[mask]
    n = a.size
    if n < 2:
        return 0.0
    na, nb = int(a.max()) + 1, int(b.max()) + 1
    if na <= 1 or nb <= 1:
        return 0.0
    joint = np.zeros(na * nb, dtype=np.int64)
    np.add.at(joint, a * nb + b, 1)
    joint = joint.reshape(na, nb)
    row, col = joint.sum(axis=1), joint.sum(axis=0)
    h_r, h_c = _entropy(row), _entropy(col)
    if h_r <= 0 or h_c <= 0:
        return 0.0
    h_rc = _entropy(joint.ravel())
    bias = (np.count_nonzero(row) - 1) * (np.count_nonzero(col) - 1) / (2.0 * n)
    mi = max(0.0, h_r + h_c - h_rc - bias)
    return float(min(1.0, mi / min(h_r, h_c)))


def _eligible_columns(prof: Profile, columns: Optional[Sequence[str]], max_cols: int) -> List[str]:
    if columns is not None:
        return list(columns)
    cols = [c.name for c in prof if c.kind not in (CONSTANT, ID)]
    if len(cols) > max_cols:
        cols = sorted(cols, key=lambda c: -prof[c].nunique)[:max_cols]
    return cols


def mutual_info_matrix(
    df: pd.DataFrame, columns: Optional[Sequence[str]] = None, *, bins: int = 10, max_cols: int = 30
) -> pd.DataFrame:
    """Symmetric matrix of normalized mutual information (0..1) between columns.

    Diagonal is 1.0 (a column is fully "dependent" on itself). ``columns`` defaults to
    every non-constant, non-id column, capped to the ``max_cols`` highest-cardinality
    ones (dependency structure is more interesting there than on near-constant fields).
    """
    prof = profile(df)
    cols = _eligible_columns(prof, columns, max_cols)
    if len(cols) < 2:
        raise ValueError("need at least two usable columns to map dependencies")
    codes = {c: _discretize(df[c], prof[c].kind if c in prof else CATEGORICAL, bins) for c in cols}
    out = np.eye(len(cols), dtype=float)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            mi = _pair_mi(codes[cols[i]], codes[cols[j]])
            out[i, j] = out[j, i] = mi
    return pd.DataFrame(out, index=cols, columns=cols)


def dependency_pairs(
    df: pd.DataFrame, columns: Optional[Sequence[str]] = None, *, bins: int = 10, max_cols: int = 30
) -> pd.DataFrame:
    """Long-format ``column_a``/``column_b``/``association`` (0..1), both directions,
    ready to pivot. See :func:`mutual_info_matrix`."""
    m = mutual_info_matrix(df, columns, bins=bins, max_cols=max_cols)
    long = m.stack().rename("association").rename_axis(["column_a", "column_b"]).reset_index()
    return long[long["column_a"] != long["column_b"]].reset_index(drop=True)


__all__ = ["mutual_info_matrix", "dependency_pairs"]
