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
from dataclasses import dataclass
from typing import Any, List, Tuple

import pandas as pd
from pandas.api import types as pdt

from . import _binning as B
from ._fit import _materialize_table


@dataclass(frozen=True)
class Trend:
    """Per-row trend series (rows x time buckets) with what :func:`pivot2hist.spikes`
    needs on top of the sparkline itself: ``counts`` (rows behind each cell, same shape as
    ``table``), the bucket ``starts`` (one timestamp per label) and the ``freq`` they were
    cut at."""

    table: pd.DataFrame
    counts: pd.DataFrame
    labels: List[str]
    starts: List[pd.Timestamp]
    freq: str


def trend_tables(view: Any, column: str, *, max_points: int = 24) -> Trend:
    """The machinery behind :func:`sparkline_table`, plus the support: see :class:`Trend`.
    ``counts`` keeps a "spike" built on two rows from ranking above one built on two
    thousand."""
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
    key, label = B._time_keys(s[s.notna()], freq)
    first = pd.Series(key.to_numpy(), index=label.to_numpy()).groupby(level=0).min()
    starts = [pd.Timestamp(first[c]) for c in cats]
    ref_index = view.table().index

    frame, row_keys, _col_keys, vname, agg = _materialize_table(df, layout)
    frame = frame.assign(__bucket__=bucket, __n__=1)
    frame = frame[frame["__bucket__"] != B.NULL]
    fill_value = 0 if agg in ("sum", "count", "nunique") else None
    if frame.empty or not cats:
        empty = pd.DataFrame(index=ref_index, columns=cats, dtype=float)
        return Trend(empty, empty.fillna(0), cats, starts, freq)

    def pivot(values: str, how: str, fill: Any) -> pd.DataFrame:
        t = pd.pivot_table(
            frame, index=row_keys, columns="__bucket__", values=values, aggfunc=how,
            observed=True, dropna=False, fill_value=fill,
        )
        t = t.reindex(columns=cats, fill_value=fill)
        return t.reindex(ref_index, fill_value=fill)

    return Trend(pivot(vname, agg, fill_value), pivot("__n__", "sum", 0), cats, starts, freq)


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
    t = trend_tables(view, column, max_points=max_points)
    return t.table, t.labels


__all__ = ["sparkline_table", "trend_tables", "Trend"]
