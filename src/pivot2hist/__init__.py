"""pivot2hist: auto-fitted pivot tables that toggle to histograms and back, with slicing.

Quick start::

    import pivot2hist as p2h

    v = p2h.fit(df)               # auto-chooses rows/cols/measure to fit a 40x12 box
    print(v)                      # pivot table
    print(v.toggle())             # the same data as a histogram
    print(v.slice(action="deny")) # sliced pivot
    print(v.histogram("bytes"))   # histogram of a specific column
"""
from __future__ import annotations

from typing import Any, Optional, Sequence, Union


from . import sample
from ._binning import RULES, bin_count, bin_edges, bin_labels
from ._fit import AGGS, Dim, DimSpec, FitOptions, Layout, build_table, fit_layout
from ._io import load
from ._profile import ColumnProfile, Profile
from ._profile import profile as _profile
from ._view import HIST, PIVOT, Filter, View

__version__ = "0.1.0"


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
    """Profile the columns of ``data`` (kinds, cardinality, nulls)."""
    return _profile(load(data), **kw)


__all__ = [
    "fit", "pivot", "histogram", "profile", "load",
    "View", "Layout", "Dim", "FitOptions", "Filter", "Profile", "ColumnProfile",
    "build_table", "fit_layout", "bin_edges", "bin_count", "bin_labels",
    "RULES", "AGGS", "PIVOT", "HIST", "sample", "__version__",
]
