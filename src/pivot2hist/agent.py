"""A plain-JSON surface for LLM agents and other tool-calling frameworks.

Every function here takes and returns only JSON-safe types (``str``, ``int``, ``float``,
``bool``, ``None``, ``list``, ``dict``) — never a DataFrame, a numpy scalar or a pandas
Timestamp — so an agent's tool-calling layer can pass its model-generated arguments
straight through and hand the result straight back to the model, with no pandas import
on the caller's side. Docstrings double as the tool descriptions :mod:`pivot2hist.mcp_server`
registers, so read them as the contract an agent sees.

A typical exploration loop for an agent::

    describe("events.parquet")                              # what is this data?
    pivot("events.parquet", agg="count")                     # an auto-fitted first look
    pivot("events.parquet", rows=["src_ip"], cols=["action"],
          filters=[{"column": "action", "in": ["deny", "drop"]}])
    suggest("events.parquet", n=5)                            # other layouts worth trying
    slicers("events.parquet", columns=["action", "country"]) # values to filter on

Everything routes through :func:`pivot2hist.fit`, so large files are surveyed and paged
the same way as in a notebook: a call never has to load more than the memory budget.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Union

Filter = Dict[str, Any]
Source = Union[str, os.PathLike, List[Dict[str, Any]], Dict[str, List[Any]], Any]

#: JSON Schema-ish description of one filter dict, for building an agent's tool schema.
FILTER_OPS = ("eq", "not_eq", "in", "range", "gt", "gte", "lt", "lte", "regex", "since", "on", "query")


def _is_path_like(source: Any) -> bool:
    return isinstance(source, (str, os.PathLike)) or type(source).__name__ == "DuckDBPyConnection"


def _apply_filters(v: Any, filters: Optional[Sequence[Filter]]) -> Any:
    """Apply a list of JSON filter dicts to a View via :meth:`View.slice`/:meth:`View.exclude`.

    Each filter is ``{"query": "<pandas expression>"}`` or ``{"column": "<name>", <op>: <value>}``
    with exactly one of: ``eq``, ``not_eq``, ``in`` (a list), ``range`` (a ``[lo, hi]`` pair),
    ``gt``/``gte``/``lt``/``lte`` (a number), ``regex`` (a pattern), ``since`` (a relative
    time span like ``"24h"`` or ``"7d"``, for a datetime column), ``on`` (a date/month/year
    string). Unknown keys raise ``ValueError`` naming the offending filter.
    """
    if not filters:
        return v
    for f in filters:
        if not isinstance(f, dict):
            raise ValueError(f"each filter must be an object, got {f!r}")
        if "query" in f:
            v = v.slice(f["query"])
            continue
        col = f.get("column")
        if col is None:
            raise ValueError(f"filter needs 'column' (or 'query'): {f!r}")
        ops = [k for k in f if k != "column"]
        if len(ops) != 1:
            raise ValueError(f"filter on {col!r} needs exactly one of {FILTER_OPS}, got {ops or 'none'}: {f!r}")
        op = ops[0]
        val = f[op]
        if op == "eq":
            v = v.slice(**{col: val})
        elif op == "not_eq":
            v = v.exclude(**{col: val})
        elif op == "in":
            v = v.slice(**{col: list(val)})
        elif op == "range":
            lo, hi = val
            v = v.slice(**{col: (lo, hi)})
        elif op == "gt":
            v = v.slice(**{col: f"> {val}"})
        elif op == "gte":
            v = v.slice(**{col: f">= {val}"})
        elif op == "lt":
            v = v.slice(**{col: f"< {val}"})
        elif op == "lte":
            v = v.slice(**{col: f"<= {val}"})
        elif op == "regex":
            v = v.slice(**{col: "~" + str(val)})
        elif op == "since":
            v = v.slice(**{col: f"last {val}"})
        elif op == "on":
            v = v.slice(**{col: val})
        else:
            raise ValueError(f"unknown filter op {op!r} on {col!r}; use one of {FILTER_OPS}")
    return v


def describe(
    source: Source,
    *,
    columns: Optional[Sequence[str]] = None,
    memory_budget_mb: Optional[float] = None,
    distributions: bool = False,
    **load_kwargs: Any,
) -> Dict[str, Any]:
    """What is this data? Column kinds, semantic types, cardinality, nulls, time series.

    Cheap and safe on large files: a path or DuckDB source is only *surveyed* (rows on
    disk, size, a bounded probe sample) rather than fully loaded, so this never reads more
    than a few tens of thousands of rows regardless of the source's real size. Call this
    first; call :func:`pivot` to actually see the data.

    With ``distributions=True``, each numeric column also gets a ``distribution`` entry —
    the closed-form family (normal, lognormal, exponential, gamma, uniform, poisson,
    geometric, bernoulli, or discrete-uniform) that best explains it by BIC, e.g.
    ``{"family": "lognormal", "params": {"mu": 7.0, "sigma": 1.8}}``. Off by default: it's
    a further, slower pass over every numeric column, not part of the cheap survey.

    Returns a dict with ``columns`` (one entry per column: ``name``, ``kind``,
    ``semantic`` type or ``None``, ``nunique``, ``null_frac``, ``examples``, and
    ``distribution`` when requested), ``rows_profiled``, ``is_time_series``,
    ``time_column``, and — for a file or DuckDB source — ``survey`` (disk size, estimated
    memory, the load plan pivot2hist would use).
    """
    from ._density import fit_distribution
    from ._io import load as _load
    from ._profile import NUMERIC, profile as _profile_fn
    from ._survey import survey as _survey

    out: Dict[str, Any] = {}
    if _is_path_like(source):
        sv = _survey(source, columns=columns, memory_budget_mb=memory_budget_mb, **load_kwargs)
        out["survey"] = {
            "source": sv.source,
            "format": sv.format,
            "disk_mb": round(sv.disk_mb, 2) if sv.disk_mb is not None else None,
            "rows": sv.rows,
            "rows_exact": sv.rows_exact,
            "estimated_memory_mb": round(sv.est_memory_mb, 1),
            "plan": sv.plan.mode,
            "plan_summary": sv.plan.summary(),
        }
        frame = sv.probe if sv.probe is not None else _load(source, columns=list(columns) if columns else None)
    else:
        frame = _load(source)
        if columns:
            frame = frame[list(columns)]
    prof = _profile_fn(frame)
    cols = []
    for c in prof:
        entry: Dict[str, Any] = {
            "name": c.name,
            "kind": c.kind,
            "semantic": c.semantic,
            "nunique": int(c.nunique),
            "null_frac": round(c.null_frac, 4),
            "examples": list(c.examples),
        }
        if distributions and c.kind == NUMERIC:
            fit = fit_distribution(frame[c.name])
            entry["distribution"] = {"family": fit.name, "params": {k: round(v, 6) for k, v in fit.params.items()}, "bic": round(fit.bic, 2)} if fit else None
        cols.append(entry)
    out["columns"] = cols
    out["rows_profiled"] = int(len(frame))
    out["is_time_series"] = bool(prof.is_time_series)
    out["time_column"] = prof.time_column
    return out


def pivot(
    source: Source,
    *,
    rows: Optional[Sequence[str]] = None,
    cols: Optional[Sequence[str]] = None,
    values: Optional[str] = None,
    agg: Optional[str] = None,
    filters: Optional[Sequence[Filter]] = None,
    mode: str = "pivot",
    on: Optional[str] = None,
    by: Optional[Union[str, Sequence[str]]] = None,
    bins: Optional[Union[str, int]] = None,
    max_rows: int = 40,
    max_cols: int = 12,
    layers: int = 2,
    aspect: Optional[float] = None,
    memory_budget_mb: Optional[float] = None,
    columns: Optional[Sequence[str]] = None,
    **opts: Any,
) -> Dict[str, Any]:
    """Build a pivot table (or its histogram) and return it as JSON.

    Leave ``rows``/``cols``/``values`` unset to auto-fit; name them to fix an axis and let
    the rest be chosen. ``filters`` is a list of filter objects (see :func:`describe`'s
    sibling constant ``FILTER_OPS`` for the operators, or just pass
    ``[{"query": "bytes > 1000 and action == 'deny'"}]``). Set ``mode="hist"`` (or give
    ``on``/``by``) for a histogram of one column instead of a cross-tab.

    Returns ``mode``, ``layout`` (measure, row/column dimensions), ``shape``, ``table``
    (a list of row records — column names are the pivot's column labels), ``rows`` (rows
    matched after filtering), ``source_rows``, ``slices`` (the filters actually applied,
    as text), and — for data too large to hold in memory — ``paged: true`` (count/sum/min/
    max/mean are still exact; other aggregations fall back to a sample and set
    ``approximate: true``).
    """
    from . import fit as _fit

    v = _fit(
        source,
        rows=list(rows) if rows is not None else None,
        cols=list(cols) if cols is not None else None,
        values=values,
        agg=agg,
        max_rows=max_rows,
        max_cols=max_cols,
        layers=layers,
        aspect=aspect,
        memory_budget_mb=memory_budget_mb,
        columns=columns,
        **opts,
    )
    v = _apply_filters(v, filters)
    if mode == "hist" or on is not None or by is not None:
        v = v.histogram(on, by, bins=bins)
    return v.to_dict()


def suggest(
    source: Source,
    n: int = 5,
    *,
    rows: Optional[Sequence[str]] = None,
    cols: Optional[Sequence[str]] = None,
    values: Optional[str] = None,
    agg: Optional[str] = None,
    filters: Optional[Sequence[Filter]] = None,
    max_rows: int = 40,
    max_cols: int = 12,
    memory_budget_mb: Optional[float] = None,
    **opts: Any,
) -> Dict[str, Any]:
    """The ``n`` best layouts for this data, best first — the auto-guess menu.

    Use this when unsure which columns make a good pivot. Each candidate has a
    ``description`` (e.g. ``"sum(bytes) by dst_ip (top 39) x rule"``), a raw fit
    ``score`` and a ``confidence`` (``"high"``/``"medium"``/``"low"``, from the gap to the
    next-best candidate — ``"low"`` means the alternatives are genuinely close and worth
    a look, not just a formality) suitable for picking by name and passing back into
    :func:`pivot` as explicit ``rows``/``cols``.
    """
    from . import fit as _fit

    v = _fit(
        source, rows=list(rows) if rows is not None else None, cols=list(cols) if cols is not None else None,
        values=values, agg=agg, max_rows=max_rows, max_cols=max_cols, memory_budget_mb=memory_budget_mb, **opts,
    )
    v = _apply_filters(v, filters)
    ranked = v.suggest_ranked(n)
    return {
        "current": v.layout.describe(),
        "confidence": v.confidence,
        "alternatives": [
            {"rank": i, "description": lay.describe(), "score": round(sc, 3), "rows": [d.column for d in lay.rows], "cols": [d.column for d in lay.cols], "measure": lay.measure}
            for i, (sc, lay) in enumerate(ranked)
        ],
    }


def anomalies(
    source: Source,
    n: int = 10,
    *,
    rows: Optional[Sequence[str]] = None,
    cols: Optional[Sequence[str]] = None,
    values: Optional[str] = None,
    agg: Optional[str] = None,
    filters: Optional[Sequence[Filter]] = None,
    max_rows: int = 40,
    max_cols: int = 12,
    memory_budget_mb: Optional[float] = None,
    **opts: Any,
) -> Dict[str, Any]:
    """The ``n`` most surprising cells of the pivot: the biggest deviation from what
    independence of the row and column axes would predict (a Pearson-style residual
    against ``row_total x col_total / grand_total``). Needs an additive measure
    (``sum``/``count``, the default). Positive ``residual`` = more than expected
    (an unusually heavy combination — a way to spot, say, a port/action pair that fires
    far more than its marginals alone would suggest); negative = less."""
    from . import fit as _fit

    v = _fit(
        source, rows=list(rows) if rows is not None else None, cols=list(cols) if cols is not None else None,
        values=values, agg=agg, max_rows=max_rows, max_cols=max_cols, memory_budget_mb=memory_budget_mb, **opts,
    )
    v = _apply_filters(v, filters)
    df = v.anomalies(n)
    return {"layout": v.layout.describe(), "cells": df.to_dict(orient="records")}


def slicers(
    source: Source,
    *,
    columns: Optional[Sequence[str]] = None,
    top: int = 10,
    memory_budget_mb: Optional[float] = None,
    **opts: Any,
) -> Dict[str, Any]:
    """Values worth filtering on: the ``top`` most frequent values (with counts) per
    label-like column, for building :func:`pivot`'s ``filters``."""
    from . import fit as _fit

    v = _fit(source, memory_budget_mb=memory_budget_mb, **opts)
    raw = v.slicers(top)
    if columns:
        raw = {k: vals for k, vals in raw.items() if k in set(columns)}
    return {col: [{"value": val, "count": int(c)} for val, c in vals] for col, vals in raw.items()}


__all__ = ["describe", "pivot", "suggest", "slicers", "anomalies", "FILTER_OPS"]
