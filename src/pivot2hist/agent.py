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

import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

Filter = Dict[str, Any]
Source = Union[str, os.PathLike, List[Dict[str, Any]], Dict[str, List[Any]], Any]

#: JSON Schema-ish description of one filter dict, for building an agent's tool schema.
FILTER_OPS = ("eq", "not_eq", "in", "range", "gt", "gte", "lt", "lte", "regex", "since", "on", "query")


def _is_path_like(source: Any) -> bool:
    return isinstance(source, (str, os.PathLike)) or type(source).__name__ == "DuckDBPyConnection"


def _filter_spec(f: Filter) -> Tuple[str, Any]:
    """One JSON filter dict as ``("query", expr)``, ``("slice", {col: spec})`` or
    ``("exclude", {col: spec})`` in :meth:`View.slice`'s own grammar."""
    if not isinstance(f, dict):
        raise ValueError(f"each filter must be an object, got {f!r}")
    if "query" in f:
        return "query", f["query"]
    col = f.get("column")
    if col is None:
        raise ValueError(f"filter needs 'column' (or 'query'): {f!r}")
    ops = [k for k in f if k != "column"]
    if len(ops) != 1:
        raise ValueError(f"filter on {col!r} needs exactly one of {FILTER_OPS}, got {ops or 'none'}: {f!r}")
    op = ops[0]
    val = f[op]
    if op == "eq":
        return "slice", {col: val}
    if op == "not_eq":
        return "exclude", {col: val}
    if op == "in":
        return "slice", {col: list(val)}
    if op == "range":
        lo, hi = val
        return "slice", {col: (lo, hi)}
    if op in ("gt", "gte", "lt", "lte"):
        return "slice", {col: f"{ {'gt': '>', 'gte': '>=', 'lt': '<', 'lte': '<='}[op] } {val}"}
    if op == "regex":
        return "slice", {col: "~" + str(val)}
    if op == "since":
        return "slice", {col: f"last {val}"}
    if op == "on":
        return "slice", {col: val}
    raise ValueError(f"unknown filter op {op!r} on {col!r}; use one of {FILTER_OPS}")


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
        kind, spec = _filter_spec(f)
        if kind == "query":
            v = v.slice(spec)
        elif kind == "slice":
            v = v.slice(**spec)
        else:
            v = v.exclude(**spec)
    return v


def _side_spec(f: Filter) -> Union[str, Dict[str, Any]]:
    """A filter dict as one side of a comparison: a query string or a ``{col: spec}``."""
    kind, spec = _filter_spec(f)
    if kind in ("query", "slice"):
        return spec
    (col, val), = spec.items()
    return f"`{col}` != {val!r}"


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


def llm_context(
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
    table_max_rows: int = 30,
    table_max_cols: int = 12,
    layers: int = 2,
    aspect: Optional[float] = None,
    memory_budget_mb: Optional[float] = None,
    columns: Optional[Sequence[str]] = None,
    notes: bool = True,
    **opts: Any,
) -> Dict[str, Any]:
    """Build a pivot/histogram (same arguments as :func:`pivot`) and return it packaged
    for an LLM instead of as a raw table: ``{"description", "metadata", "table"}``.

    ``description`` is a short natural-language summary (what the table shows, its
    shape, active filters, whether it's approximate, the most surprising cell if cheap
    to compute); ``metadata`` is the same facts as plain JSON, scoped to just the
    columns actually used (not a full profile dump); ``table`` is a GitHub-flavored
    markdown rendering, truncated (not sampled) to ``table_max_rows`` x
    ``table_max_cols`` — deliberately markdown rather than a list of per-cell dicts,
    since it's both what a model has seen the most of and the cheapest in tokens. Prefer
    this over :func:`pivot` when the result is headed into a model's context rather than
    back into your own code.
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
    return v.llm_context(max_rows=table_max_rows, max_cols=table_max_cols, notes=notes)


def insights(
    source: Source,
    *,
    rows: Optional[Sequence[str]] = None,
    cols: Optional[Sequence[str]] = None,
    values: Optional[str] = None,
    agg: Optional[str] = None,
    filters: Optional[Sequence[Filter]] = None,
    sensitivity: float = 0.5,
    max_findings: int = 15,
    max_pairs: int = 5,
    max_rows: int = 40,
    max_cols: int = 12,
    layers: int = 2,
    aspect: Optional[float] = None,
    memory_budget_mb: Optional[float] = None,
    columns: Optional[Sequence[str]] = None,
    **opts: Any,
) -> Dict[str, Any]:
    """A rich, local, non-LLM analysis of the (optionally filtered/laid-out) data:
    column summaries (kind, semantic type, distribution, cardinality), and ranked
    findings - skew, a mixture-model check for multiple populations in one numeric
    column, concentration, outliers, correlated column pairs, and (when a 2-D layout is
    available) the most surprising cells.

    Nothing here calls a model: every score is a plain, deterministic computation, cheap
    enough to call again after every new ``filters``. ``sensitivity`` (0..1, default
    0.5) sets how much of the ranked findings list surfaces - higher shows more
    (including weaker findings), lower shows only the strongest; it is not a
    "temperature" (no sampling is involved, so that word would be misleading here).
    ``rows``/``cols``/``values``/``agg`` are optional - leave them unset and a layout is
    still auto-fitted, purely so the surprising-cell check has a 2-D table to look at.
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
    return v.insights(sensitivity=sensitivity, max_findings=max_findings, max_pairs=max_pairs)


def compare(
    source: Source,
    *,
    split: Filter,
    vs: Optional[Filter] = None,
    metric: Optional[str] = None,
    n: int = 10,
    rows: Optional[Sequence[str]] = None,
    cols: Optional[Sequence[str]] = None,
    values: Optional[str] = None,
    agg: Optional[str] = None,
    filters: Optional[Sequence[Filter]] = None,
    max_rows: int = 40,
    max_cols: int = 12,
    layers: int = 2,
    aspect: Optional[float] = None,
    memory_budget_mb: Optional[float] = None,
    columns: Optional[Sequence[str]] = None,
    **opts: Any,
) -> Dict[str, Any]:
    """Compare two sides of the data cell by cell on one shared pivot layout.

    ``split`` is one filter object (same forms as ``filters``: ``{"column": "action",
    "eq": "deny"}``, ``{"column": "bytes", "gt": 1000}``, ``{"query": "..."}``) naming
    side A; side B is everything else, or ``vs`` (a second filter object) - e.g. ``split =
    {"column": "timestamp", "on": "2026-03-02"}``, ``vs = {"column": "timestamp", "on":
    "2026-03-01"}`` for today against yesterday. ``filters`` apply to both sides first.
    The layout is auto-fitted once (or fixed via ``rows``/``cols``/``values``/``agg``) and
    then *frozen* for both sides - same dimensions, bins and top-N labels - so every cell
    means the same thing on each side; if the split column sits on an axis it is taken off
    it and that axis refilled.

    ``metric``: ``"lift"`` (default for an additive measure: A's share of its own total
    over B's share, so a small slice compares fairly against a large one; >1 means
    over-represented in A), ``"delta"`` (A - B; default otherwise), ``"ratio"``,
    ``"pct_change"``, ``"share_delta"`` (percentage points), or ``"a"``/``"b"``/
    ``"share_a"``/``"share_b"``.

    Returns ``description``, ``metric`` and ``metric_meaning``, ``a``/``b`` (name, rows,
    slices, total), ``layout``, ``shape``, ``table`` (the metric per cell, as records),
    ``top`` - the ``n`` cells that differ most, each with both raw values, ``delta``,
    ``ratio``, ``lift``, ``only_in`` (set when the other side has nothing there) and, for
    a count measure, ``p`` (a per-cell G-test of the cell's share of its side, A vs B;
    raw, so compare it with ``0.05 / cells``) - and ``drivers``: what else differs between
    the sides beyond the table's axes (see :meth:`Comparison.drivers`), each with a
    ready-made ``text``.
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
    sa = _side_spec(split)
    c = v.compare(sa) if vs is None else v.compare(sa, _side_spec(vs))
    if metric is not None:
        c = c.with_metric(metric)
    return c.to_dict(n)


def _view_for(source: Source, *, rows: Any, cols: Any, values: Any, agg: Any, filters: Any, mode: str, on: Any, by: Any,
              bins: Any, max_rows: int, max_cols: int, memory_budget_mb: Any, columns: Any, opts: Dict[str, Any]) -> Any:
    from . import fit as _fit

    v = _fit(
        source, rows=list(rows) if rows is not None else None, cols=list(cols) if cols is not None else None,
        values=values, agg=agg, max_rows=max_rows, max_cols=max_cols, memory_budget_mb=memory_budget_mb, columns=columns, **opts,
    )
    v = _apply_filters(v, filters)
    if mode == "hist" or on is not None or by is not None:
        v = v.histogram(on, by, bins=bins)
    return v


def rows(
    source: Source,
    *,
    cell: Dict[str, Any],
    n: int = 50,
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
    memory_budget_mb: Optional[float] = None,
    columns: Optional[Sequence[str]] = None,
    **opts: Any,
) -> Dict[str, Any]:
    """The raw rows behind one cell of the pivot (or one bin of the histogram), with every
    filter applied - the step after :func:`pivot` when a cell looks interesting.

    ``cell`` is ``{column: label}`` for any axis column, using the labels :func:`pivot`
    shows - a bin like ``"[1K, 2K)"``, a time bucket like ``"13:00"``, ``"(other)"`` for
    the folded top-N remainder, ``"(null)"``, a roll-up like ``"10.0.1.0/24"`` - or a plain
    value. Give one axis only to get a whole row or column. A column that is not on an
    axis is applied as an ordinary filter. Same layout arguments as :func:`pivot`, so the
    labels line up with the table you were just looking at. Returns ``cell``, ``layout``,
    ``n_returned`` (capped at ``n``), ``n_total`` (all matching rows; ``None`` for a paged
    source, which is scanned only up to ``n``) and ``rows`` as records.
    """
    v = _view_for(source, rows=rows, cols=cols, values=values, agg=agg, filters=filters, mode=mode, on=on, by=by, bins=bins,
                  max_rows=max_rows, max_cols=max_cols, memory_budget_mb=memory_budget_mb, columns=columns, opts=opts)
    from ._explain import _records

    df = v.rows(n=n, **cell)
    return {
        "cell": {str(k): str(x) for k, x in cell.items()},
        "layout": v.layout.describe(),
        "n_returned": int(len(df)),
        "n_total": (int(len(v.cell(**cell).data)) if v.paged is None else None),
        "rows": _records(df),
    }


def explain(
    source: Source,
    *,
    cell: Dict[str, Any],
    n_rows: int = 10,
    k: int = 5,
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
    memory_budget_mb: Optional[float] = None,
    columns: Optional[Sequence[str]] = None,
    **opts: Any,
) -> Dict[str, Any]:
    """Why does this cell look the way it does? ``cell`` names it exactly as in
    :func:`rows`. Returns the cell's ``observed`` value; for a count/sum on a 2-D table its
    ``expected`` value under independence of the axes, ``ratio_to_expected`` and
    ``direction`` (over/under); its shares of its row, column and the table; its ``rank``
    among all cells; ``n_rows`` behind it; ``distinguishing`` - the ``k`` columns that most
    set those rows apart from the rest of the data in view (a label over-represented here
    with its share here vs elsewhere and a lift; a numeric whose median differs, as a
    ratio), each with a ready-made ``text``; the first ``n_rows`` rows; and a one-paragraph
    ``text`` saying all of it. Deterministic and local - no model call.
    """
    v = _view_for(source, rows=rows, cols=cols, values=values, agg=agg, filters=filters, mode=mode, on=on, by=by, bins=bins,
                  max_rows=max_rows, max_cols=max_cols, memory_budget_mb=memory_budget_mb, columns=columns, opts=opts)
    return dict(v.explain(n_rows=n_rows, k=k, **cell))


def prompt(
    source: Source,
    *,
    question: Optional[str] = None,
    table: bool = True,
    profile: bool = True,
    anomalies: bool = True,
    insights: bool = False,
    sensitivity: float = 0.5,
    split: Optional[Filter] = None,
    vs: Optional[Filter] = None,
    metric: Optional[str] = None,
    cell: Optional[Dict[str, Any]] = None,
    table_max_rows: int = 30,
    table_max_cols: int = 12,
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
    memory_budget_mb: Optional[float] = None,
    columns: Optional[Sequence[str]] = None,
    **opts: Any,
) -> Dict[str, Any]:
    """One self-contained analysis prompt, as text, built from facts computed locally -
    for handing to another model, a colleague's chat, or a file. It states that nothing
    in it is invented and that the model should reason only from it, then gives the
    dataset's columns (``profile``), the table with its filters (``table``), the most
    surprising cells (``anomalies``), optionally the ranked insights (``insights``, at
    ``sensitivity``), a comparison (``split``/``vs``/``metric`` as in :func:`compare`), a
    cell in focus (``cell`` as in :func:`explain`), and ``question`` (default: what stands
    out, three next steps, data-quality flags). Same layout arguments as :func:`pivot`.
    Returns ``prompt`` (markdown), ``chars``, ``tokens_estimate`` and the ``question`` used.
    """
    from ._prompt import DEFAULT_QUESTION

    v = _view_for(source, rows=rows, cols=cols, values=values, agg=agg, filters=filters, mode=mode, on=on, by=by, bins=bins,
                  max_rows=max_rows, max_cols=max_cols, memory_budget_mb=memory_budget_mb, columns=columns, opts=opts)
    comparison = None
    if split is not None:
        sa = _side_spec(split)
        comparison = v.compare(sa) if vs is None else v.compare(sa, _side_spec(vs))
        if metric is not None:
            comparison = comparison.with_metric(metric)
    p = v.prompt(
        question, table=table, profile=profile, anomalies=anomalies, insights=bool(insights), sensitivity=sensitivity,
        compare=comparison, explain=dict(cell) if cell else None, max_rows=table_max_rows, max_cols=table_max_cols,
    )
    return {"prompt": str(p), "chars": p.chars, "tokens_estimate": p.tokens,
            "question": question.strip() if isinstance(question, str) and question.strip() else DEFAULT_QUESTION}


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


def spikes(
    source: Source,
    column: Optional[str] = None,
    n: int = 10,
    *,
    z: float = 3.0,
    min_support: int = 5,
    baseline: str = "auto",
    shifts: bool = True,
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
    """Which rows of the pivot moved over time, when, and by how much against their own
    history: the ``n`` strongest spikes, drops and step changes per row of the table,
    over the datetime ``column`` (default: the first one), each with the row, the time
    bucket, the ``kind``, ``observed`` vs ``expected``, the ``ratio``, a signed robust
    ``score`` (baseline spreads away), the rows behind it and a ready-made sentence. The
    baseline is seasonal when three periods exist (same hours on other days, same weekday
    in other weeks ...), else the row's share of the bucket total, else its median; see
    :meth:`View.spikes`. Same layout arguments as :func:`pivot`."""
    from . import fit as _fit
    from ._spikes import describe_spike

    v = _fit(
        source, rows=list(rows) if rows is not None else None, cols=list(cols) if cols is not None else None,
        values=values, agg=agg, max_rows=max_rows, max_cols=max_cols, memory_budget_mb=memory_budget_mb, **opts,
    )
    v = _apply_filters(v, filters)
    df = v.spikes(column, n=n, z=z, min_support=min_support, baseline=baseline, shifts=shifts)
    from ._spikes import default_time_column

    col = column or default_time_column(v)
    recs = []
    for r in df.itertuples():
        d = {k: (None if isinstance(x, float) and not math.isfinite(x) else x) for k, x in r._asdict().items() if k != "Index"}
        d["text"] = describe_spike(r, v.layout.measure, str(col))
        recs.append(d)
    return {"layout": v.layout.describe(), "column": col, "measure": v.layout.measure, "n": len(recs), "spikes": recs}


def novel(
    source: Source,
    entity: Optional[str] = None,
    attr: Optional[str] = None,
    *,
    since: Any = 0.25,
    time: Optional[str] = None,
    n: int = 10,
    min_support: int = 3,
    filters: Optional[Sequence[Filter]] = None,
    max_rows: int = 40,
    max_cols: int = 12,
    memory_budget_mb: Optional[float] = None,
    **opts: Any,
) -> Dict[str, Any]:
    """What is new in the recent part of the data, per entity: the rows are split at
    ``since`` (a fraction of the time span such as ``0.25``, a duration back from the end
    such as ``"24h"``, or a moment such as ``"2026-03-06"``) on the datetime ``time``
    column (default: the first), and every ``entity`` value seen since (default: the
    address-like column with the most distinct values) is checked against the history:
    ``new entity`` (never seen before), ``new pair`` (an entity carrying an ``attr`` value
    it never had; ``peers`` = how many other entities had that value before), ``new
    value`` (a value nobody had used) and ``fan-out`` (distinct ``attr`` values since vs
    before, when at least doubled). Returns the ``n`` strongest with a bits-like ``score``
    and a ready-made ``text`` each; see :meth:`View.novel`."""
    from . import fit as _fit

    v = _fit(source, max_rows=max_rows, max_cols=max_cols, memory_budget_mb=memory_budget_mb, **opts)
    v = _apply_filters(v, filters)
    df = v.novel(entity, attr, since=since, time=time, n=n, min_support=min_support)
    recs = []
    for r in df.itertuples():
        d = {k: x for k, x in r._asdict().items() if k != "Index"}
        d["peers"] = None if d["peers"] is None or (isinstance(d["peers"], float) and math.isnan(d["peers"])) else int(d["peers"])
        d["before"], d["after"] = int(d["before"]), int(d["after"])
        recs.append(d)
    since_at = df.attrs.get("since")
    return {"entity": entity, "attr": attr, "since": (str(since_at) if since_at is not None else None),
            "n": len(recs), "findings": recs}


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


__all__ = ["describe", "pivot", "llm_context", "insights", "compare", "rows", "explain", "prompt", "suggest", "slicers", "anomalies", "spikes", "novel", "FILTER_OPS"]
