"""Matrices and chains: Markov transition matrices and frequent sequences of states.

For event logs the interesting structure is often *what follows what*: which action
follows a failed login for the same user, which port a scanner tries after 22. A
transition matrix is just a pivot table (rows = from-state, columns = to-state) so it
gets the whole toolkit: heatmap, toggle, slicing, clustering.

::

    v = p2h.chains(df, "event", by="user", time="timestamp")   # View: from x to
    p2h.sequences(df, "event", by="user", time="timestamp", length=3)
"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence

import numpy as np
import pandas as pd

ARROW = " → "


def _ordered(df: pd.DataFrame, by: Optional[Sequence[str]], time: Optional[str]) -> pd.DataFrame:
    keys = list(by or [])
    if time is not None:
        keys.append(time)
    if keys:
        return df.sort_values(keys, kind="stable")
    return df


def _as_list(x: Optional[Any]) -> Optional[List[str]]:
    if x is None:
        return None
    return [x] if isinstance(x, str) else list(x)


def transitions(
    df: pd.DataFrame,
    state: str,
    *,
    by: Optional[Any] = None,
    time: Optional[str] = None,
    order: int = 1,
    dropna: bool = True,
) -> pd.DataFrame:
    """Long-form transition counts: columns ``from``, ``to``, ``count``, ``prob``.

    ``by`` groups the sequences (a user, a source IP); transitions never cross groups.
    ``time`` sorts within groups. ``order=2`` uses the previous two states as ``from``.
    """
    by_l = _as_list(by)
    d = _ordered(df, by_l, time)
    s = d[state].astype(object)
    if dropna:
        keep = s.notna()
        d, s = d[keep], s[keep]
    if len(d) == 0:
        return pd.DataFrame(columns=["from", "to", "count", "prob"])
    g = d.groupby(by_l, sort=False, observed=True, dropna=False)[state] if by_l else None
    if g is None:
        nxt = s.shift(-1)
        prevs = [s.shift(k) for k in range(order - 1, 0, -1)]
    else:
        nxt = g.shift(-1).astype(object)
        prevs = [g.shift(k).astype(object) for k in range(order - 1, 0, -1)]
    frm = s.astype(str)
    for p in prevs:
        frm = p.astype(str) + ARROW + frm
    ok = nxt.notna()
    for p in prevs:
        ok &= p.notna()
    long = pd.DataFrame({"from": frm[ok].to_numpy(), "to": nxt[ok].astype(str).to_numpy()})
    counts = long.value_counts().rename("count").reset_index()
    counts["prob"] = counts["count"] / counts.groupby("from")["count"].transform("sum")
    return counts.sort_values(["from", "count"], ascending=[True, False]).reset_index(drop=True)


def transition_matrix(df: pd.DataFrame, state: str, *, by: Optional[Any] = None, time: Optional[str] = None,
                      order: int = 1, normalize: bool = False) -> pd.DataFrame:
    """Square-ish matrix of transition counts (or row probabilities with ``normalize``)."""
    t = transitions(df, state, by=by, time=time, order=order)
    if t.empty:
        return pd.DataFrame()
    m = t.pivot_table(index="from", columns="to", values="count", aggfunc="sum", fill_value=0)
    if normalize:
        m = m.div(m.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    return m


def steady_state(matrix: pd.DataFrame, *, iters: int = 500, tol: float = 1e-10) -> pd.Series:
    """Stationary distribution of a transition matrix (power iteration, row-stochastic)."""
    states = sorted(set(matrix.index) | set(matrix.columns))
    P = matrix.reindex(index=states, columns=states, fill_value=0).to_numpy(dtype=float)
    rows = P.sum(axis=1, keepdims=True)
    P = np.where(rows > 0, P / np.where(rows > 0, rows, 1), 1.0 / len(states))
    pi = np.full(len(states), 1.0 / len(states))
    for _ in range(iters):
        new = pi @ P
        if np.abs(new - pi).sum() < tol:
            pi = new
            break
        pi = new
    return pd.Series(pi, index=states, name="steady_state").sort_values(ascending=False)


def sequences(
    df: pd.DataFrame,
    state: str,
    *,
    by: Optional[Any] = None,
    time: Optional[str] = None,
    length: int = 3,
    n: int = 10,
    dropna: bool = True,
) -> pd.DataFrame:
    """The ``n`` most frequent chains of ``length`` consecutive states within each group.

    Columns: ``chain`` (``"a → b → c"``), ``count``, ``share`` (of all chains), and
    ``groups`` (how many distinct ``by`` groups showed it) when ``by`` is given.
    """
    by_l = _as_list(by)
    d = _ordered(df, by_l, time)
    s = d[state].astype(object)
    if dropna:
        keep = s.notna()
        d, s = d[keep], s[keep]
    if len(d) < length:
        return pd.DataFrame(columns=["chain", "count", "share"] + (["groups"] if by_l else []))
    g = d.groupby(by_l, sort=False, observed=True, dropna=False)[state] if by_l else None
    cols = []
    for k in range(length):
        shifted = (g.shift(-k) if g is not None else s.shift(-k)).astype(object)
        cols.append(shifted)
    ok = np.ones(len(d), dtype=bool)
    for c in cols:
        ok &= c.notna().to_numpy()
    chain = cols[0].astype(str)
    for c in cols[1:]:
        chain = chain + ARROW + c.astype(str)
    chain = chain[ok]
    counts = chain.value_counts()
    out = counts.rename("count").reset_index()
    out.columns = ["chain", "count"]
    out["share"] = out["count"] / out["count"].sum()
    if by_l:
        grp = d.loc[chain.index, by_l].astype(str).agg("|".join, axis=1) if len(by_l) > 1 else d.loc[chain.index, by_l[0]].astype(str)
        groups = pd.DataFrame({"chain": chain.to_numpy(), "g": grp.to_numpy()}).drop_duplicates().groupby("chain").size()
        out["groups"] = out["chain"].map(groups).fillna(0).astype(int)
    return out.head(n).reset_index(drop=True)


__all__ = ["transitions", "transition_matrix", "steady_state", "sequences", "ARROW"]
