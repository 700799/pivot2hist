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
    p2h.fit("huge.parquet")       # surveyed, fitted on a sample, aggregated page by page
    p2h.verbose(); p2h.stats(7)   # scrolling step log; the seven costliest steps
"""
from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import pandas as pd


from . import agent, sample
from ._binning import RULES, bin_count, bin_edges, bin_labels, kde
from ._chains import sequences, steady_state, transition_matrix, transitions
from ._cluster import COMETHODS, METHODS, cluster_frame, cluster_rows, cocluster, dbscan, kmeans
from ._density import DistFit, fit_distribution, rank_distributions
from ._density import FAMILIES as DIST_FAMILIES
from ._mixture import GMMFit, choose_gmm_k, fit_gmm, mixture_cutpoints
from ._hmm import HMMFit, choose_hmm_states, decode_regimes, fit_hmm
from ._deps import dependency_pairs, mutual_info_matrix
from ._fit import AGGS, DEFAULT_WEIGHTS, Dim, DimSpec, FitOptions, Layout, build_table, fit_layout, suggest_layouts
from ._io import load
from ._log import log, stats, verbose
from ._profile import ColumnProfile, Profile
from ._profile import profile as _profile
from ._semantic import HIERARCHY, infer_semantic
from ._survey import Machine, PagedSource, Plan, Survey, downcast, load_planned, survey
from ._view import HIST, PIVOT, Derived, Filter, View

__version__ = "0.7.0"

_PLANNED_KEYS = ("memory_budget_mb", "mode", "columns", "query", "table", "sample_rows", "page_rows")


def _needs_plan(data: Any) -> bool:
    """Paths and DuckDB sources go through the survey; frames and records load directly."""
    if isinstance(data, (str, bytes)) or hasattr(data, "__fspath__"):
        return True
    return type(data).__name__ == "DuckDBPyConnection"


def _load_for_fit(data: Any, opts: dict):
    """Split fit() kwargs into loading knobs and FitOptions; return (frame_or_paged, survey)."""
    planned = {k: opts.pop(k) for k in _PLANNED_KEYS if k in opts}
    if _needs_plan(data) or planned:
        if not _needs_plan(data) and planned.get("mode", "auto") == "auto" and "memory_budget_mb" not in planned:
            frame = load(data)
            cols = planned.get("columns")
            if cols:
                missing = [c for c in cols if c not in frame.columns]
                if missing:
                    raise KeyError(f"unknown column(s) {missing}; available: {list(frame.columns)[:20]}")
                frame = frame[list(cols)]
            return frame, None
        frame, sv = load_planned(data, **planned)
        return frame, sv
    return load(data), None


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
    """Auto-fit ``data`` (frame, path, records, ``duckdb://`` URL ...) into a pivot :class:`View`.

    Fix any of ``rows``/``cols``/``values``/``agg`` and the rest is chosen for you.
    Keyword options (``max_rows``, ``max_cols``, ``layers``, ``aspect``, ``bins``,
    ``scale`` ...) are :class:`FitOptions` fields.

    Files and DuckDB sources are surveyed first (rows, size on disk, estimated memory
    against the machine's RAM); too-big data is fitted on a sample and aggregated page by
    page. Loading knobs: ``memory_budget_mb`` (default: half the free RAM), ``mode``
    (``"auto"`` | ``"full"`` | ``"downcast"`` | ``"sample"`` | ``"paged"``), ``columns``,
    ``query`` / ``table`` for DuckDB, ``sample_rows``, ``page_rows``.
    """
    frame, sv = _load_for_fit(data, opts)
    v = View.fit(frame, rows=rows, cols=cols, values=values, agg=agg, options=options, **opts)
    return v if sv is None else View(frame, v.layout, options=v.options, spec=v._spec, survey=sv)


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
    frame, sv = _load_for_fit(data, opts)
    base = View.fit(frame, options=options, **opts)
    if sv is not None:
        base = View(frame, base.layout, options=base.options, spec=base._spec, survey=sv)
    if on is None and by is None and values is None and bins is None and scale is None:
        return base.toggle()
    return base.histogram(on, by, bins=bins, values=values, agg=agg, scale=scale)


def profile(data: Any, **kw: Any) -> Profile:
    """Profile the columns of ``data`` (kinds, semantic types, cardinality, nulls, time series)."""
    return _profile(load(data), **kw)


def suggest(data: Any, n: int = 5, **opts: Any) -> list:
    """The ``n`` best distinct layouts for ``data``, best first (the auto-guess menu)."""
    return [lay for _, lay in suggest_layouts(load(data), FitOptions().replace(**opts) if opts else None, n)]


def distribution(data: Any, column: str, **kw: Any):
    """Best-fitting probability distribution for one numeric column of ``data`` (BIC over
    normal/lognormal/exponential/gamma/uniform/poisson/geometric/bernoulli/discrete-uniform).

    ``kw`` forwards to :func:`fit_distribution` (``families=``, ``discrete_max=``, ``min_n=``).
    """
    df = load(data)
    if column not in df.columns:
        raise KeyError(f"unknown column {column!r}")
    return fit_distribution(df[column], **kw)


def modes(data: Any, column: str, k: Optional[int] = None, **kw: Any) -> Optional[list]:
    """How many peaks does this numeric column have, and where? A Gaussian mixture fit
    (component count chosen by BIC unless ``k`` is given), as a list of
    ``{"weight", "mean", "std"}`` dicts sorted by mean, or ``None`` if there isn't enough
    data. No scipy/sklearn: EM from scratch, see :mod:`pivot2hist._mixture`.
    """
    df = load(data)
    if column not in df.columns:
        raise KeyError(f"unknown column {column!r}")
    import numpy as _np

    x = _np.asarray(df[column], dtype=float)
    x = x[_np.isfinite(x)]
    if x.size < 8:
        return None
    kk = k if k is not None else choose_gmm_k(x.reshape(-1, 1), **kw)
    fit = fit_gmm(x.reshape(-1, 1), max(1, kk), **{k2: v for k2, v in kw.items() if k2 != "k_max"})
    return fit.components()


def explore(data: Any, **kw: Any) -> Any:
    """Interactive Jupyter explorer (needs ``ipywidgets``): menus to alter, slice, best-fit,
    reduce and cluster, with heatmap pivots and SVG histograms."""
    from .ui import explore as _explore

    return _explore(data, **kw)


def cluster(data: Any, columns: Optional[Sequence[str]] = None, k: Optional[int] = None, *, method: str = "kmeans", name: str = "cluster") -> pd.DataFrame:
    """``data`` with an extra ``cluster`` column over numeric ``columns`` (default: all).

    ``method``: ``"kmeans"`` (auto k), ``"dbscan"`` or ``"hdbscan"`` (outliers -> ``noise``)."""
    df = load(data)
    out = df.copy()
    out[name] = cluster_frame(df, columns, k, method=method, name=name)
    return out


def chains(
    data: Any,
    state: str,
    *,
    by: Optional[Union[str, Sequence[str]]] = None,
    time: Optional[str] = None,
    order: int = 1,
    normalize: bool = False,
    **opts: Any,
) -> View:
    """Markov transition matrix of ``state`` as a :class:`View` (rows = from, columns = to).

    ``by`` keeps sequences inside an entity (user, source IP); ``time`` orders them;
    ``order=2`` conditions on the previous two states; ``normalize`` shows row
    probabilities instead of counts. Everything else (toggle, slice, cluster, cocluster,
    style) works as on any pivot. See :func:`sequences` for the most frequent chains.
    """
    df = load(data)
    long = transitions(df, state, by=by, time=time, order=order)
    if long.empty:
        raise ValueError("no transitions found (need at least two consecutive states per group)")
    values, agg = ("prob", "sum") if normalize else ("count", "sum")
    max_states = max(opts.pop("max_rows", 40), opts.pop("max_cols", 12))
    v = View.fit(long, rows=[{"column": "from", "top": max_states - 1}] if long["from"].nunique() > max_states else ["from"],
                 cols=[{"column": "to", "top": max_states - 1}] if long["to"].nunique() > max_states else ["to"],
                 values=values, agg=agg, max_rows=max_states, max_cols=max_states, **opts)
    return v.style(heat="row" if normalize else "table")


def regimes(
    data: Any,
    state: str,
    *,
    by: Optional[Union[str, Sequence[str]]] = None,
    time: Optional[str] = None,
    n_states: Optional[int] = None,
    k_max: int = 4,
    seed: int = 0,
    **opts: Any,
) -> View:
    """Hidden Markov regimes over ``state`` as a :class:`View` (rows = regime, columns =
    ``state``), so you can see what each regime looks like and ``toggle()`` to a
    histogram of it.

    A Baum-Welch fit (no hmmlearn/scipy, see :mod:`pivot2hist._hmm`) decodes each row
    into one of a small number of hidden regimes from the sequence of ``state`` values,
    e.g. a user's logins drifting from a "normal" regime into a "credential-stuffing"
    regime. ``by`` keeps sequences inside an entity (user, source IP); ``time`` orders
    them; ``n_states`` fixes the regime count (default: chosen by BIC, up to ``k_max``).
    Regimes are numbered by how common they are (``"regime 1"`` = most common).
    """
    df = load(data)
    regime = decode_regimes(df, state, by=by, time=time, n_states=n_states, k_max=k_max, seed=seed)
    out = df.copy()
    out["regime"] = regime.values
    max_cols = opts.pop("max_cols", 12)
    cols = [{"column": state, "top": max_cols - 1}] if df[state].nunique() > max_cols else [state]
    v = View.fit(out, rows=["regime"], cols=cols, values=None, agg="count", max_cols=max_cols, **opts)
    return v.style(heat="row")


def dependencies(data: Any, columns: Optional[Sequence[str]] = None, *, bins: int = 10, max_cols: int = 30, **opts: Any) -> View:
    """Which columns of ``data`` move together, as a square :class:`View` (rows = cols =
    column names, cells = normalized mutual information, 0..1).

    No correlation-matrix assumption of linearity or numeric-only columns: every column
    is discretized (numeric/datetime into quantile bins, categorical/boolean by top-N)
    and scored by bias-corrected mutual information, so a categorical/numeric pair (e.g.
    ``protocol`` and ``dst_port``) shows up just as well as two numeric ones. ``.toggle()``
    turns it into a histogram of each column's total association with everything else.
    See :func:`pivot2hist.mutual_info_matrix` for the plain matrix.
    """
    df = load(data)
    long = dependency_pairs(df, columns, bins=bins, max_cols=max_cols)
    v = View.fit(long, rows=["column_a"], cols=["column_b"], values="association", agg="max",
                 max_rows=max_cols, max_cols=max_cols, **opts)
    return v.style(heat="table")


__all__ = [
    "fit", "pivot", "histogram", "profile", "load", "suggest", "explore", "cluster", "chains", "regimes", "dependencies",
    "survey", "load_planned", "downcast", "stats", "verbose", "log", "distribution",
    "sequences", "transitions", "transition_matrix", "steady_state",
    "View", "Layout", "Dim", "FitOptions", "Filter", "Derived", "Profile", "ColumnProfile",
    "Survey", "Plan", "Machine", "PagedSource", "DistFit",
    "build_table", "fit_layout", "suggest_layouts", "bin_edges", "bin_count", "bin_labels", "kde",
    "cluster_frame", "cluster_rows", "cocluster", "kmeans", "dbscan", "METHODS", "COMETHODS",
    "infer_semantic", "HIERARCHY", "DEFAULT_WEIGHTS", "fit_distribution", "rank_distributions", "DIST_FAMILIES",
    "modes", "GMMFit", "fit_gmm", "choose_gmm_k", "mixture_cutpoints",
    "HMMFit", "fit_hmm", "choose_hmm_states", "decode_regimes",
    "mutual_info_matrix", "dependency_pairs",
    "RULES", "AGGS", "PIVOT", "HIST", "sample", "agent", "__version__",
]
