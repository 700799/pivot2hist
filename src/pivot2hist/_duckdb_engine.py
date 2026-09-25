"""Optional DuckDB pushdown for :func:`pivot2hist._fit.build_table`'s group-by step.

Layout planning (which dims, bin edges, top-N budgets, time frequency) stays in pandas
on a bounded sample - it's already cheap. What DuckDB buys is the scan + group-by +
aggregate over the *actual* rows being pivoted, using DuckDB's vectorized engine instead
of ``pandas.pivot_table``. That part is pushed down for the dim kinds it can express in
SQL; the rest of the pipeline (labeling, ordering, the final pivot reshape) stays exactly
the pandas code every other path already uses, so results are identical either way.

Falls back to the ordinary pandas path (returns ``None``, and :func:`build_table` retries
with ``engine="pandas"``) whenever a dim needs something DuckDB can't express here: a
semantic drill level (IP ``/24``, port class ...), a cyclic time bucket (hour of day,
weekday), a frozen ``keep`` list (paged sources), or a non-string/number/bool categorical
value.
"""
from __future__ import annotations

import weakref
from collections import OrderedDict
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from . import _binning as B
from ._binning import _natural_sort_key, _time_keys
from ._fit import COUNT, Dim, Layout, effective_label

# Registering a DataFrame with DuckDB converts it into DuckDB's own columnar format,
# which costs roughly as much as the aggregation query itself - a real tax on a one-off
# call. Views are immutable and often queried repeatedly (pivot(), bins(), a slice, a
# handful of suggest() layouts all touch the same frame), so one connection per distinct
# DataFrame is kept alive and reused. It's a bounded LRU, not a weakref cache: DuckDB's
# register() keeps its own strong reference to the DataFrame for as long as it's
# registered, so a cache entry that itself holds the connection can never become weakly
# unreachable - the entry would leak forever. DataFrames aren't hashable, so the cache is
# keyed by id() with an identity check (via a weakref, used only to detect a reused id)
# before trusting a hit.
_CONN_CACHE_MAX = 8
_CONN_CACHE: "OrderedDict[int, Tuple[Any, weakref.ref]]" = OrderedDict()


def _close_entry(entry: Tuple[Any, Any]) -> None:
    con, _ref = entry
    try:
        if hasattr(con, "unregister"):
            con.unregister("t")  # drops DuckDB's own reference to the DataFrame
    except Exception:
        pass
    try:
        con.close()
    except Exception:
        pass


def _connection_for(df: pd.DataFrame, duckdb_module: Any) -> Any:
    key = id(df)
    entry = _CONN_CACHE.get(key)
    if entry is not None:
        con, ref = entry
        if ref() is df:
            _CONN_CACHE.move_to_end(key)
            return con
        _close_entry(entry)  # id reused by an unrelated, now-different DataFrame
        del _CONN_CACHE[key]

    con = duckdb_module.connect()
    con.register("t", df)
    _CONN_CACHE[key] = (con, weakref.ref(df))
    while len(_CONN_CACHE) > _CONN_CACHE_MAX:
        _, oldest = _CONN_CACHE.popitem(last=False)
        _close_entry(oldest)
    return con

_PLAIN_TIME_SQL = {"min": "minute", "h": "hour", "D": "day", "M": "month", "Q": "quarter", "Y": "year"}
_BUCKET_TIME_SQL = {"5min": "5 minutes", "15min": "15 minutes", "6h": "6 hours"}
_SCALAR_AGGS = {"sum": "sum", "mean": "avg", "count": "count", "min": "min", "max": "max",
                "median": "median", "std": "stddev_samp"}


def supported(layout: Layout) -> bool:
    """Whether every dim in this layout, and the measure, can be pushed down."""
    for d in list(layout.rows) + list(layout.cols):
        if d.is_coarse or d.keep is not None:
            return False
        if d.kind == "time" and (d.freq or "D") not in _PLAIN_TIME_SQL and (d.freq or "D") not in _BUCKET_TIME_SQL:
            return False
    return True


def _quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _sql_literal(v: Any) -> Optional[str]:
    """A SQL literal for ``v``, or ``None`` if its type isn't safe to embed directly."""
    if isinstance(v, (bool, np.bool_)):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return repr(float(v))
    if isinstance(v, str):
        return "'" + v.replace("'", "''") + "'"
    return None


def _binned_expr(col: str, d: Dim) -> Tuple[str, List[str]]:
    edges = d.edges or (0.0, 1.0)
    n = len(edges) - 1
    parts = [f"WHEN {col} IS NULL OR isnan(CAST({col} AS DOUBLE)) OR isinf(CAST({col} AS DOUBLE)) THEN NULL"]
    for i in range(n):
        parts.append(f"ELSE {i}" if i == n - 1 else f"WHEN {col} < {float(edges[i + 1])!r} THEN {i}")
    return "CASE " + " ".join(parts) + " END", B.bin_labels(edges, integer=bool(d.integer))


def _time_expr(col: str, d: Dim) -> str:
    freq = d.freq or "D"
    ts = f"CAST({col} AS TIMESTAMP)"
    if freq in _PLAIN_TIME_SQL:
        return f"date_trunc('{_PLAIN_TIME_SQL[freq]}', {ts})"
    return f"time_bucket(INTERVAL '{_BUCKET_TIME_SQL[freq]}', {ts})"


def _categorical_expr(con: Any, table: str, col: str, d: Dim) -> Optional[Tuple[str, List[str]]]:
    """SQL CASE expression folding to top-N/"(other)", and the resulting category order.

    ``d.top is None`` means "no cap" (mirrors ``categorize(top=None)``): every distinct
    value is fetched, unbounded, and nothing ever folds into "(other)". Only when ``d.top``
    is set do we probe one row past it (a bounded, small LIMIT) to learn whether folding
    is actually needed, matching ``categorize()``'s own ``len(counts) > top`` check.
    """
    limit_sql = f" LIMIT {int(d.top) + 1}" if d.top is not None else ""
    rows = con.execute(
        f"SELECT {col} AS v, COUNT(*) AS n FROM {table} WHERE {col} IS NOT NULL "
        f"GROUP BY {col} ORDER BY n DESC{limit_sql}"
    ).fetchall()
    if not rows:
        return None
    values = [r[0] for r in rows]
    if any(_sql_literal(v) is None for v in values):
        return None  # a value type we don't know how to embed as a SQL literal (date, bytes, ...)
    folding = d.top is not None and len(values) > d.top
    if folding:
        values = values[: d.top]
    order = d.order if d.order in ("natural", "frequency") else ("natural" if all(isinstance(v, (int, float, bool)) for v in values[:50]) else "frequency")
    cats = _natural_sort_key(list(values)) if order == "natural" else list(values)  # already n DESC from SQL
    labels = [str(v) for v in cats]
    branches = " ".join(f"WHEN {col} = {_sql_literal(v)} THEN {_sql_literal(str(v))}" for v in values)
    other = _sql_literal(B.OTHER)
    expr = f"CASE WHEN {col} IS NULL THEN NULL {branches} ELSE {other} END"
    if folding:
        labels.append(B.OTHER)
    return expr, labels


def _agg_sql(vname: str, agg: str) -> Tuple[str, str]:
    if agg == "nunique":
        return f'COUNT(DISTINCT {_quote_ident(vname)}) AS "__v"', "__v"
    fn = _SCALAR_AGGS.get(agg)
    if fn is None:
        raise ValueError(f"unsupported agg {agg!r}")
    return f'{fn}({_quote_ident(vname)}) AS "__v"', "__v"


def aggregate(
    df: pd.DataFrame, layout: Layout, *, observed: bool
) -> Optional[Tuple[pd.DataFrame, List[str], List[str], str, str]]:
    """DuckDB-backed equivalent of ``_fit._materialize_table``: same return shape, tiny frame.

    Returns ``None`` (caller falls back to pandas) when this layout, or the value column's
    dtype/aggregation, isn't one this module can express in SQL. Raises ``ImportError`` if
    the ``duckdb`` package itself isn't installed - the caller asked for it explicitly.
    """
    try:
        import duckdb
    except ImportError as e:
        raise ImportError("engine='duckdb' needs the duckdb package: pip install duckdb") from e
    if not supported(layout):
        return None
    if layout.values is not None:
        v = df[layout.values]
        if pdt.is_timedelta64_dtype(v):
            return None  # SQL INTERVAL handling is a narrow case; pandas already does this cheaply
        if not pdt.is_bool_dtype(v) and not pdt.is_numeric_dtype(v) and layout.agg in ("sum", "mean", "min", "max", "median", "std"):
            raise ValueError(f"cannot {layout.agg} non-numeric column {layout.values!r}; use agg='count' or 'nunique'")

    con = _connection_for(df, duckdb)
    exprs: List[str] = []
    aliases: List[str] = []
    cats: dict = {}
    row_keys: List[str] = []
    col_keys: List[str] = []
    for axis_dims, key_list in ((layout.rows, row_keys), (layout.cols, col_keys)):
        for d in axis_dims:
            col = _quote_ident(d.column)
            alias = f"k{len(aliases)}"
            if d.kind == "binned":
                expr, labels = _binned_expr(col, d)
            elif d.kind == "time":
                expr = _time_expr(col, d)
                labels = None  # ordered from the observed data itself, see below
            else:
                got = _categorical_expr(con, "t", col, d)
                if got is None:
                    return None
                expr, labels = got
            exprs.append(f"{expr} AS {alias}")
            aliases.append(alias)
            cats[alias] = (d, labels)
            # dim.top (and so the "(other)" fold check inside effective_label) is only
            # ever set on categorical dims, so an empty-but-correctly-categorized dummy
            # is enough here - binned/time dims never read past the short-circuited check.
            dummy = pd.Categorical([], categories=labels if labels is not None else [])
            label = effective_label(d, pd.Series(dummy))
            if label in key_list or label in row_keys or label in col_keys:
                label += " "
            key_list.append(label)

    if layout.values is None:
        vname, agg = COUNT, "sum"
        agg_sql, vcol = "COUNT(*) AS \"__v\"", "__v"
    else:
        vname, agg = layout.values, layout.agg
        agg_sql, vcol = _agg_sql(layout.values, agg)

    select = ", ".join(exprs + [agg_sql]) if exprs else agg_sql
    group_by = ", ".join(aliases)
    sql = f"SELECT {select} FROM t" + (f" GROUP BY {group_by}" if aliases else "")
    result = con.execute(sql).fetchdf()

    work: dict = {}
    for alias, key in zip(aliases, row_keys + col_keys):
        d, labels = cats[alias]
        raw = result[alias]
        has_null = bool(raw.isna().any())
        if d.kind == "binned":
            code = raw.astype("Int64")
            full_cats = list(labels) + ([B.NULL] if has_null else [])
            str_labels = [full_cats[int(c)] if pd.notna(c) else B.NULL for c in code]
            work[key] = pd.Series(pd.Categorical(str_labels, categories=full_cats, ordered=True))
        elif d.kind == "time":
            valid = raw.dropna()
            if valid.empty:
                order_cats: List[str] = []
                label_map = {}
            else:
                keyed, labeled = _time_keys(pd.Series(pd.to_datetime(valid.to_numpy())), d.freq or "D")
                lut = pd.DataFrame({"k": keyed.to_numpy(), "l": labeled.to_numpy()}).drop_duplicates("l").sort_values("k")
                order_cats = lut["l"].tolist()
                label_map = dict(zip(valid.to_numpy(), labeled.to_numpy()))
            full_cats = order_cats + ([B.NULL] if has_null else [])
            str_labels = [label_map.get(v, B.NULL) if pd.notna(v) else B.NULL for v in raw.to_numpy()]
            work[key] = pd.Series(pd.Categorical(str_labels, categories=full_cats, ordered=True))
        else:
            full_cats = list(labels) + ([B.NULL] if has_null else [])
            str_labels = [v if pd.notna(v) else B.NULL for v in raw]
            work[key] = pd.Series(pd.Categorical(str_labels, categories=full_cats, ordered=True))
    work[vname] = result[vcol]
    frame = pd.DataFrame(work)
    return frame, row_keys, col_keys, vname, agg


__all__ = ["aggregate", "supported"]
