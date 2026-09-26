"""Novelty per entity: what is *new* in the recent part of the data - entities never seen
before, pairs an entity had never made before, values nobody had ever used, and entities
whose fan-out suddenly grew.

The data in view is split at a point in time into *before* (the history) and *after* (the
recent window) and each entity of the ``entity`` column (a source IP, a user, a host ...)
seen after is checked against the history:

* **new entity** - never seen before the split;
* **new pair** - the entity existed, but its combination with this ``attr`` value did
  not (a source that starts talking to a port it never used); ``peers`` says how many
  other entities had used that value before - the fewer, the rarer the pair;
* **new value** - a pair whose value nobody had used before at all;
* **fan-out** - the entity's distinct ``attr`` values in the recent window against its
  history (a scanner's port count jumping from 3 to 40).

Everything is counted, nothing is modelled: scores are in bits-like units (log2 of the
rows behind the finding, plus log2 of how rare the value was among entities), so a
finding on many rows about a rare value ranks first and the kinds rank against each
other. Pairs with the spike detector (:mod:`pivot2hist._spikes`): spikes say what grew,
novelty says what appeared.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from pandas.api import types as pdt

from . import _binning as B
from ._log import log
from ._profile import CATEGORICAL, DATETIME

COLUMNS = ["kind", "entity", "value", "before", "after", "peers", "score", "text"]
KINDS = ("new entity", "new pair", "new value", "fan-out")
_ENTITY_SEMANTICS = ("ipv4", "ipv6", "email", "domain", "mac", "url")
_DURATION = re.compile(r"^\s*\d+(\.\d+)?\s*[a-zA-Z]+\s*$")


def default_time_column(view: Any) -> Optional[str]:
    for cp in view.profile:
        if cp.kind == DATETIME:
            return cp.name
    return None


def default_entity_columns(view: Any) -> Tuple[Optional[str], Optional[str]]:
    """A guess at (entity, attr) for :func:`novelty` when the caller gives none: the
    entity is the address-like column (IP, email, domain, MAC, URL) with the most distinct
    values, else the label column with the most distinct values that is not id-like; the
    attribute is the label column with the most distinct values after that."""
    n = max(1, len(view.data))
    cands = [cp for cp in view.profile if cp.kind == CATEGORICAL and 2 <= cp.nunique <= max(2, n // 2)]
    if not cands:
        return None, None
    addr = [cp for cp in cands if cp.semantic in _ENTITY_SEMANTICS]
    pool = addr or [cp for cp in cands if cp.nunique >= 10] or cands
    entity = max(pool, key=lambda cp: cp.nunique)
    rest = [cp for cp in cands if cp.name != entity.name]
    attr = max(rest, key=lambda cp: cp.nunique) if rest else None
    return entity.name, (attr.name if attr else None)


def _as_time(s: pd.Series) -> pd.Series:
    if pdt.is_datetime64_any_dtype(s):
        return s
    if isinstance(s.dtype, pd.PeriodDtype):
        return s.dt.to_timestamp()
    return pd.to_datetime(s, errors="coerce")


def split_point(t: pd.Series, since: Union[float, str, pd.Timestamp, None]) -> pd.Timestamp:
    """Where the recent window starts: a fraction of the span (``0.25`` = the last
    quarter), a duration back from the end (``"24h"``, ``"7D"``), or a moment
    (``"2026-03-06"``, a ``Timestamp``)."""
    valid = t.dropna()
    if valid.empty:
        raise ValueError("no usable datetime values to split on")
    t_min, t_max = valid.min(), valid.max()
    if since is None:
        since = 0.25
    if isinstance(since, (int, float)) and not isinstance(since, bool):
        if not 0 < since < 1:
            raise ValueError("since as a number is the fraction of the time span that counts as recent: 0 < since < 1")
        return t_max - (t_max - t_min) * float(since)
    if isinstance(since, str) and _DURATION.match(since):
        try:
            return t_max - pd.Timedelta(since)
        except ValueError:
            pass
    point = pd.Timestamp(since)
    if point.tzinfo is not None and getattr(t_max, "tzinfo", None) is None:
        point = point.tz_localize(None)
    elif point.tzinfo is None and getattr(t_max, "tzinfo", None) is not None:
        point = point.tz_localize(t_max.tzinfo)
    return point


def _key(s: pd.Series) -> pd.Series:
    return B._hashable_column(s)


def _label(x: Any) -> Optional[str]:
    return None if x is None else str(x)


def _fmt_when(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%d %H:%M") if (ts.hour or ts.minute) else ts.strftime("%Y-%m-%d")


def novelty(
    view: Any,
    entity: Optional[str] = None,
    attr: Optional[str] = None,
    *,
    since: Union[float, str, pd.Timestamp, None] = 0.25,
    time: Optional[str] = None,
    n: int = 10,
    min_support: int = 3,
) -> pd.DataFrame:
    """Everything :meth:`View.novel` returns - see that method for the contract."""
    df = view.data
    if entity is None:
        entity, guess_attr = default_entity_columns(view)
        if entity is None:
            raise ValueError("no label column to treat as the entity; pass entity=")
        if attr is None:
            attr = guess_attr
    if entity not in df.columns:
        raise KeyError(f"unknown column {entity!r}")
    if attr is not None and attr not in df.columns:
        raise KeyError(f"unknown column {attr!r}")
    if attr == entity:
        raise ValueError("attr must be a different column from entity")
    if time is None:
        time = default_time_column(view)
        if time is None:
            raise ValueError("no datetime column to split on; pass time=")
    if time not in df.columns:
        raise KeyError(f"unknown column {time!r}")
    t = _as_time(df[time])
    point = split_point(t, since)
    when = _fmt_when(point)

    with log.step("novelty", f"{entity}" + (f" x {attr}" if attr else "") + f" since {when}"):
        keep = t.notna() & df[entity].notna()
        ent = _key(df[entity])[keep]
        recent = (t[keep] >= point).to_numpy()
        ent_b, ent_a = ent[~recent], ent[recent]
        before_counts = ent_b.value_counts()
        after_counts = ent_a.value_counts()
        n_entities_before = int(before_counts.size)
        rows: List[Dict[str, Any]] = []

        distinct_after = pd.Series(dtype=int)
        if attr is not None:
            ok = keep & df[attr].notna()
            ent2 = _key(df[entity])[ok]
            val2 = _key(df[attr])[ok]
            rec2 = (t[ok] >= point).to_numpy()
            pairs_b = pd.DataFrame({"e": ent2[~rec2].to_numpy(), "v": val2[~rec2].to_numpy()})
            pairs_a = pd.DataFrame({"e": ent2[rec2].to_numpy(), "v": val2[rec2].to_numpy()})
            if not pairs_a.empty:
                distinct_after = pairs_a.groupby("e")["v"].nunique()

        # new entities (a newcomer that immediately spreads over many values ranks higher)
        new_ent = after_counts[~after_counts.index.isin(before_counts.index)]
        for e, c in new_ent.items():
            if c < min_support:
                continue
            spread = int(distinct_after.get(e, 0))
            rows.append({
                "kind": "new entity", "entity": _label(e), "value": None, "before": 0, "after": int(c), "peers": None,
                "score": round(math.log2(1 + c) + (math.log2(spread) if spread > 1 else 0.0), 3),
                "text": f"{entity} {e!s} is new: first seen since {when}, {c:,} row(s)"
                        + (f" over {spread:,} distinct {attr}" if spread > 1 else ""),
            })

        if attr is not None:
            if not pairs_a.empty:
                seen_pairs = set(map(tuple, pairs_b.drop_duplicates().itertuples(index=False, name=None)))
                value_peers = pairs_b.drop_duplicates().groupby("v")["e"].size() if not pairs_b.empty else pd.Series(dtype=int)
                known_entities = set(before_counts.index)
                pair_counts = pairs_a.groupby(["e", "v"]).size()
                for (e, v), c in pair_counts.items():
                    if c < min_support or e not in known_entities or (e, v) in seen_pairs:
                        continue
                    peers = int(value_peers.get(v, 0))
                    rarity = math.log2((n_entities_before + 1) / (peers + 1))
                    if peers == 0:
                        kind = "new value"
                        text = f"{attr} {v!s} is a new value: never seen before {when}; {entity} {e!s} uses it, {c:,} row(s)"
                    else:
                        kind = "new pair"
                        text = (f"{entity} {e!s} → {attr} {v!s} is a new pair: not before {when}; {v!s} had been seen with "
                                f"{peers:,} of {n_entities_before:,} {entity}s, {c:,} row(s)")
                    rows.append({"kind": kind, "entity": _label(e), "value": _label(v), "before": 0, "after": int(c), "peers": peers,
                                 "score": round(math.log2(1 + c) + rarity, 3), "text": text})
                # fan-out: distinct values now vs then
                fan_b = pairs_b.groupby("e")["v"].nunique() if not pairs_b.empty else pd.Series(dtype=int)
                fan_a = pairs_a.groupby("e")["v"].nunique()
                for e, fa in fan_a.items():
                    fb = int(fan_b.get(e, 0))
                    fa = int(fa)
                    if fb < 1 or fa < max(3, min_support):
                        continue
                    ratio = (fa + 1) / (fb + 1)
                    if ratio < 2:
                        continue
                    rows.append({
                        "kind": "fan-out", "entity": _label(e), "value": None, "before": fb, "after": fa, "peers": None,
                        "score": round(math.log2(ratio) * math.log2(1 + fa), 3),
                        "text": f"{entity} {e!s} fans out: {fa:,} distinct {attr} since {when} vs {fb:,} before (×{fa / fb:.2g})",
                    })

        out = pd.DataFrame(rows, columns=COLUMNS)
        if out.empty:
            return out
        out = out.sort_values("score", ascending=False, kind="stable").head(max(0, int(n))).reset_index(drop=True)
        out.attrs["since"] = point
        return out


__all__ = ["novelty", "split_point", "default_entity_columns", "default_time_column", "COLUMNS", "KINDS"]
