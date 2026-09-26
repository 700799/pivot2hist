"""Per-row trend sparklines: how each row of the current table moved over time.

A pivot cell answers "how much" for one (row, column) combination; it says nothing about
*trend*. Seeing whether traffic to port 3389 is rising currently means putting time on
the column axis and reading two dozen numbers per row. :func:`sparkline_table` instead
collapses the column axis and gives each row a short trend series of the view's own
measure over an auto-bucketed time column, aligned to the row order the table already
has - a fast "which rows are moving" scan that a static cross-tab can't offer.
"""
from __future__ import annotations

import warnings
from typing import Any, List, Tuple

import pandas as pd
from pandas.api import types as pdt

from . import _binning as B
from ._fit import _materialize_table


def sparkline_table(view: Any, column: str, *, max_points: int = 24) -> Tuple[pd.DataFrame, List[str]]:
    """One trend series per row of ``view``'s current table (pivot or histogram): the
    view's own measure, aggregated into up to ``max_points`` time buckets of the datetime
    column ``column``, with the column axis collapsed (a sparkline is per row, not per
    cell - if you want it per cell, facet on the column values instead and sparkline each
    facet).

    Returns ``(table, bucket_labels)``: ``table``'s row index matches ``view.table()``'s
    (reindexed, so a row with nothing in view still gets a row of the fill value, same as
    every other cell) and its columns are the bucket labels in time order, coarsened
    (minute up to year) until they fit in ``max_points`` - the same ladder
    :meth:`View.histogram` uses for a time column.
    """
    layout = view.layout
    if not layout.rows:
        raise ValueError("sparklines need at least one row dimension")
    df = view.data
    if column not in df.columns:
        raise KeyError(f"unknown column {column!r}")
    s = df[column]
    if not (pdt.is_datetime64_any_dtype(s) or isinstance(s.dtype, pd.PeriodDtype)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # "could not infer format": expected for a non-time column
            s = pd.to_datetime(s, errors="coerce")
    if not s.notna().any():
        raise ValueError(f"{column!r} has no usable datetime values")
    freq = B.time_freq_for(s, max_points)
    bucket = B.bucket_time(s, freq)
    cats = [c for c in bucket.cat.categories if c != B.NULL]
    ref_index = view.table().index

    frame, row_keys, _col_keys, vname, agg = _materialize_table(df, layout)
    frame = frame.assign(__bucket__=bucket)
    frame = frame[frame["__bucket__"] != B.NULL]
    fill_value = 0 if agg in ("sum", "count", "nunique") else None
    if frame.empty or not cats:
        return pd.DataFrame(index=ref_index, columns=cats, dtype=float), cats

    table = pd.pivot_table(
        frame, index=row_keys, columns="__bucket__", values=vname, aggfunc=agg,
        observed=True, dropna=False, fill_value=fill_value,
    )
    table = table.reindex(columns=cats, fill_value=fill_value)
    table = table.reindex(ref_index, fill_value=fill_value)
    return table, cats


__all__ = ["sparkline_table"]
