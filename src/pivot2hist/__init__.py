"""pivot2hist: auto-fitted pivot tables that toggle to histograms and back, with slicing.

Quick start::

    import pivot2hist as p2h

    v = p2h.fit(df)               # auto-chooses rows/cols/measure to fit a 40x12 box
    print(v)                      # pivot table
    print(v.toggle())             # the same data as a histogram
    print(v.slice(action="deny")) # sliced pivot
    print(v.histogram("bytes"))   # histogram of a specific column
    v.suggest()                   # alternative layouts
    v.cluster(4)                  # group similar rows
    p2h.explore(df)               # Jupyter menus (ipywidgets)
"""
from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import pandas as pd


from . import sample
from ._binning import RULES, bin_count, bin_edges, bin_labels, kde
from ._cluster import cluster_frame, cluster_rows, kmeans
from ._fit import AGGS, DEFAULT_WEIGHTS, Dim, DimSpec, FitOptions, Layout, build_table, fit_layout, suggest_layouts
from ._io import load
from ._profile import ColumnProfile, Profile
from ._profile import profile as _profile
from ._semantic import HIERARCHY, infer_semantic
from ._view import HIST, PIVOT, Filter, View

__version__ = "0.2.0"


def fit(
    data: Any,
    *,
    rows: Optional[Sequence[DimSpec]] = None,
    cols: Optional[Sequence[DimSpec]] = None,
    values: Optional[str] = None,
    agg: Optional[str] = None,
    options: Optional[FitOptions] = None,
    **opts: Any,
) -> View:
    """Auto-fit ``data`` (frame, path, records ...) into a pivot :class:`View`.

    Fix any of ``rows``/``cols``/``values``/``agg`` and the rest is chosen for you.
    Keyword options (``max_rows``, ``max_cols``, ``layers``, ``aspect``, ``bins``,
    ``scale`` ...) are :class:`FitOptions` fields.
    """
    df = load(data)
    return View.fit(df, rows=rows, cols=cols, values=values, agg=agg, options=options, **opts)


pivot = fit


def histogram(
    data: Any,
    on: Optional[str] = None,
    by: Optional[Union[str, Sequence[DimSpec]]] = None,
    *,
    bins: Optional[Union[str, int]] = None,
    values: Optional[str] = None,
    agg: Optional[str] = None,
    scale: Optional[str] = None,
    options: Optional[FitOptions] = None,
    **opts: Any,
) -> View:
    """A histogram :class:`View` of ``data`` (``toggle()`` gives the matching pivot)."""
    df = load(data)
    base = View.fit(df, options=options, **opts)
    if on is None and by is None and values is None and bins is None and scale is None:
        return base.toggle()
    return base.histogram(on, by, bins=bins, values=values, agg=agg, scale=scale)


def profile(data: Any, **kw: Any) -> Profile:
    """Profile the columns of ``data`` (kinds, semantic types, cardinality, nulls, time series)."""
    return _profile(load(data), **kw)


def suggest(data: Any, n: int = 5, **opts: Any) -> list:
    """The ``n`` best distinct layouts for ``data``, best first (the auto-guess menu)."""
    return [lay for _, lay in suggest_layouts(load(data), FitOptions().replace(**opts) if opts else None, n)]


def explore(data: Any, **kw: Any) -> Any:
    """Interactive Jupyter explorer (needs ``ipywidgets``): menus to alter, slice, best-fit,
    reduce and cluster, with heatmap pivots and SVG histograms."""
    from .ui import explore as _explore

    return _explore(data, **kw)


def cluster(data: Any, columns: Optional[Sequence[str]] = None, k: Optional[int] = None, *, name: str = "cluster") -> pd.DataFrame:
    """``data`` with an extra k-means ``cluster`` column over numeric ``columns`` (default: all)."""
    df = load(data)
    out = df.copy()
    out[name] = cluster_frame(df, columns, k, name=name)
    return out


__all__ = [
    "fit", "pivot", "histogram", "profile", "load", "suggest", "explore", "cluster",
    "View", "Layout", "Dim", "FitOptions", "Filter", "Profile", "ColumnProfile",
    "build_table", "fit_layout", "suggest_layouts", "bin_edges", "bin_count", "bin_labels", "kde",
    "cluster_frame", "cluster_rows", "kmeans", "infer_semantic", "HIERARCHY", "DEFAULT_WEIGHTS",
    "RULES", "AGGS", "PIVOT", "HIST", "sample", "__version__",
]
