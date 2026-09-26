"""Comparisons and facets: two sides of a dataset on one shared layout, cell by cell, and
small multiples that share one layout and one colour scale.

The auto-fit is deliberately *frozen* here. Both sides of a comparison (or every facet)
are laid out with the same dimensions, the same bin edges and the same kept top-N labels,
so row 3 / column 2 means the same thing on every panel and cells can be subtracted. A
refit per side would pick whatever suits each side best - and make them incomparable.
"""
from __future__ import annotations

import json
import math
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from . import _binning as B
from ._fit import COUNT, Layout, fit_layout, freeze, order_index
from ._html import grid_html, hist_svg, pivot_html
from ._llm import markdown_table
from ._log import log
from ._render import fmt_cell, render_hist, render_pivot
from ._view import HIST, PIVOT, Filter, View

#: Measures whose cells add up, so a share-of-total (``lift``, ``share_delta``) is meaningful.
ADDITIVE = ("sum", "count")
#: Measures for which a cell one side never had is 0 rather than unknown.
FILLABLE = ("sum", "count", "nunique")
METRICS = ("delta", "ratio", "pct_change", "lift", "share_delta", "a", "b", "share_a", "share_b")
METRIC_HELP: Dict[str, str] = {
    "delta": "A − B",
    "ratio": "A ÷ B ('new' where only A has it)",
    "pct_change": "(A − B) ÷ B, in %",
    "lift": "A's share of its own total ÷ B's share of its total (>1: over-represented in A)",
    "share_delta": "A's share of its total − B's share of its total, in percentage points",
    "a": "side A as is",
    "b": "side B as is",
    "share_a": "% of side A's total",
    "share_b": "% of side B's total",
}
REST = "rest"
SHRUNK_NOTE = ("colour and ranking use a shrunk log-ratio: a cell with little data on either side is pulled "
               "toward ×1, so a big ratio on a few rows reads paler than a modest one on many - the values shown are exact")

Spec = Union[str, Mapping[str, Any]]


# --------------------------------------------------------------------------- helpers


def _agg_of(layout: Layout) -> str:
    return "count" if layout.values is None else layout.agg


def _on_layout(view: View, layout: Layout, **more: Any) -> View:
    """``view`` with ``layout`` imposed (its slices kept, no refit)."""
    if view.layout is layout and not more:
        return view
    spec = {"rows": list(layout.rows), "cols": list(layout.cols), "values": layout.values,
            "agg": COUNT if layout.values is None else layout.agg}
    return view._clone(layout=layout, spec=spec, **more)


def _check_columns(view: View, layout: Layout, side: str) -> None:
    have = set(view.source.columns) | set(view.derived)
    missing = [c for c in layout.columns_used if c not in have]
    if missing:
        raise KeyError(f"side {side} has no column(s) {missing} needed by the shared layout ({layout.describe()})")


def _layout_without(view: View, columns: Sequence[str]) -> Layout:
    """``view``'s layout minus ``columns`` as dimensions. An axis that empties is refilled
    by one auto-fit on the view's data (never picking those columns again), so a split on
    a column that sat on an axis still yields a 2-D table to compare."""
    lay = view.layout
    drop = {c for c in columns if c in {d.column for d in lay.dims}}
    if not drop:
        return lay
    rows = [d for d in lay.rows if d.column not in drop]
    cols = [d for d in lay.cols if d.column not in drop]
    data = view.data
    if data.empty:
        raise ValueError("nothing to compare: the view has no rows")
    opts = view.options.replace(exclude=tuple(view.options.exclude) + tuple(sorted(drop)))
    with log.step("fit", f"compare: layout without {sorted(drop)}") as st:
        layout = fit_layout(
            data, opts,
            rows=rows or None,
            cols=cols if (cols or not lay.cols) else None,
            values=lay.values, agg=COUNT if lay.values is None else lay.agg,
        )
        st.detail += f" -> {layout.describe()}"
    return layout


def _new_filters(base: View, spec: Spec) -> Tuple[Filter, ...]:
    tmp = base.slice(spec, refit=False) if isinstance(spec, str) else base.slice(refit=False, **dict(spec))
    return tmp.filters[len(base.filters):]


def _spec_columns(spec: Spec) -> List[str]:
    return [] if isinstance(spec, str) else list(spec)


def _spec_from_args(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Spec:
    """The slice() argument forms (query string, mapping, column/value, keywords) as one spec."""
    if len(args) == 1 and isinstance(args[0], str):
        if kwargs:
            raise TypeError("a query string cannot be combined with column=value keywords")
        return args[0]
    if len(args) == 1 and isinstance(args[0], Mapping):
        return {**args[0], **kwargs}
    if len(args) == 2:
        return {args[0]: args[1], **kwargs}
    if args:
        raise TypeError("takes a query string, (column, value) or column=value keywords")
    return dict(kwargs)


def _label(filters: Sequence[Filter]) -> str:
    return " & ".join(f.label for f in filters)


def _flat(index: pd.Index) -> List[str]:
    if isinstance(index, pd.MultiIndex):
        return ["/".join(str(x) for x in t) for t in index]
    return [str(x) for x in index]


def _isnull(x: Any) -> bool:
    try:
        return bool(pd.isna(x))
    except (TypeError, ValueError):
        return False


def _json_safe(v: Any) -> Any:
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, dict):
        return {str(k): _json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    if v is None or isinstance(v, (str, int, bool)):
        return v
    return str(v)


def _records(table: pd.DataFrame) -> List[Dict[str, Any]]:
    t = table.copy()
    t.columns = [" / ".join(str(x) for x in c) if isinstance(c, tuple) else str(c) for c in t.columns]
    flat = t.reset_index()
    flat.columns = [str(c) for c in flat.columns]
    return _json_safe(json.loads(flat.replace([np.inf, -np.inf], np.nan).to_json(orient="records", date_format="iso")))


def align_tables(tables: Sequence[pd.DataFrame], layout: Layout, frames: Sequence[pd.DataFrame]) -> List[pd.DataFrame]:
    """The same tables on the union of their row labels and of their column labels, in the
    layout's level order, so cell ``[i, j]`` is the same (row, column) on every one.

    A cell one side never had is 0 for an additive measure (count/sum/nunique: nothing
    happened there) and NaN otherwise (a mean of nothing is unknown, not 0). ``frames``
    supply the level order (each dim materialized on them, see :func:`order_index`).
    """
    fill = 0.0 if _agg_of(layout) in FILLABLE else np.nan
    live = [t for t in tables if not t.empty]
    if not live:
        return [pd.DataFrame(index=t.index[:0], columns=t.columns[:0], dtype=float) for t in tables]
    idx, cols = live[0].index, live[0].columns
    for t in live[1:]:
        idx = idx.union(t.index, sort=False)
        cols = cols.union(t.columns, sort=False)
    dimcols = list(dict.fromkeys(d.column for d in layout.dims))  # a column can sit on both axes (variants)
    parts = [f[[c for c in dimcols if c in f.columns]] for f in frames if not f.empty]
    if parts:
        both = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
        idx = order_index(idx, layout.rows, both)
        if layout.cols:
            cols = order_index(cols, layout.cols, both)
    out: List[pd.DataFrame] = []
    for t in tables:
        if t.empty:
            out.append(pd.DataFrame(fill, index=idx, columns=cols, dtype=float))
        else:
            r = t.reindex(index=idx, columns=cols).astype(float)
            out.append(r.fillna(fill) if fill == 0.0 else r)
    return out


def _joint_trim_index(tables: Sequence[pd.DataFrame], layout: Layout) -> pd.Index:
    """Bins empty on *every* table, trimmed off the ends (the joint version of ``View._trim``)."""
    t = tables[0]
    if t.empty:
        return t.index
    empty = np.ones(len(t), dtype=bool)
    for x in tables:
        v = x.to_numpy(dtype=float)
        empty &= (~np.isfinite(v) | (v == 0)).all(axis=1)
    if t.index.nlevels > 1:
        return t.index[~empty]
    if layout.rows and layout.rows[0].kind in ("binned", "time"):
        keep = ~empty
        if keep.any():
            first, last = int(np.argmax(keep)), len(keep) - int(np.argmax(keep[::-1]))
            return t.index[first:last]
    return t.index


def _shared_scale(arrays: Sequence[np.ndarray]) -> Tuple[List[np.ndarray], float]:
    """(transformed arrays, peak): one magnitude scale across several tables, log-compressed
    when heavy-tailed, so their heat colours mean the same thing panel to panel."""
    finite = [a[np.isfinite(a)] for a in arrays]
    allv = np.concatenate(finite) if finite else np.array([])
    nz = np.abs(allv[allv != 0])
    use_log = nz.size > 4 and nz.max() / nz.min() > 1000
    out: List[np.ndarray] = []
    for a in arrays:
        with np.errstate(all="ignore"):
            out.append(np.sign(a) * np.log1p(np.abs(a)) if use_log else np.array(a, dtype=float))
    peaks = [float(np.abs(o[np.isfinite(o)]).max()) for o in out if np.isfinite(o).any()]
    return out, (max(peaks) if peaks else 0.0)


# --------------------------------------------------------------------------- metrics


def _ratio(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = np.full(x.shape, np.nan)
    ok = np.isfinite(x) & np.isfinite(y)
    nz = ok & (y != 0)
    out[nz] = x[nz] / y[nz]
    new = ok & (y == 0) & (x != 0)
    out[new] = np.inf * np.sign(x[new])  # only on this side; 0/0 stays NaN (nothing on either)
    return out


def _share(x: np.ndarray) -> np.ndarray:
    fin = np.where(np.isfinite(x), x, 0.0)
    tot = float(fin.sum())
    if tot == 0:
        return np.full(x.shape, np.nan)
    return np.where(np.isfinite(x), x / tot * 100.0, np.nan)


def _metric_values(a: np.ndarray, b: np.ndarray, metric: str) -> np.ndarray:
    with np.errstate(all="ignore"):
        if metric == "a":
            return a.copy()
        if metric == "b":
            return b.copy()
        if metric == "delta":
            return a - b
        if metric == "ratio":
            return _ratio(a, b)
        if metric == "pct_change":
            return (_ratio(a, b) - 1.0) * 100.0
        sa, sb = _share(a), _share(b)
        if metric == "share_a":
            return sa
        if metric == "share_b":
            return sb
        if metric == "share_delta":
            return sa - sb
        return _ratio(sa, sb)  # lift


def _score_values(metric: str, a: np.ndarray, b: np.ndarray, m: np.ndarray) -> Optional[np.ndarray]:
    """What a metric table is coloured *and* ranked by: a signed, symmetric magnitude, so
    the reddest/bluest cell is also the top mover. ``None`` means plain sequential colouring
    (the raw-side metrics).

    Ratio-like metrics use a *shrunk* log2 ratio, ``log2((a + e) / (b + e))`` with ``e``
    the median positive cell value in play: x4 and x0.25 are equally strong; a cell built
    on a handful of rows is pulled toward "no change" (a x142 on 30 bytes ranks below a x6
    on millions); and a cell present on one side only ranks by how much is actually there
    instead of every such cell tying at infinity. The displayed metric stays exact."""
    if metric in ("a", "b", "share_a", "share_b"):
        return None
    if metric == "share_delta":
        return m
    if metric == "delta":
        nz = np.abs(m[np.isfinite(m) & (m != 0)])
        if nz.size > 4 and nz.max() / nz.min() > 1000:
            with np.errstate(all="ignore"):
                return np.sign(m) * np.log1p(np.abs(m))
        return m
    x, y = (_share(a), _share(b)) if metric == "lift" else (a, b)
    pos = np.concatenate([x[np.isfinite(x) & (x > 0)], y[np.isfinite(y) & (y > 0)]])
    eps = float(np.median(pos)) if pos.size else 1.0
    with np.errstate(all="ignore"):
        return np.log2((x + eps) / (y + eps))


def _num(v: Any, fmt: Callable[[float], str]) -> str:
    if v is None:
        return ""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if math.isnan(f):
        return ""
    return fmt(f)


def _fmt_ratio(f: float) -> str:
    if math.isinf(f):
        return "new" if f > 0 else "gone"
    if f == 0:
        return "×0"
    return f"×{f:.2g}" if abs(f) < 10 else f"×{f:,.0f}"


def _fmt_delta(f: float) -> str:
    return ("+" if f > 0 else "") + fmt_cell(f)


def _formatter(metric: str) -> Callable[[Any], str]:
    if metric in ("a", "b"):
        return lambda v: fmt_cell(v)
    if metric == "delta":
        return lambda v: _num(v, _fmt_delta)
    if metric in ("ratio", "lift"):
        return lambda v: _num(v, _fmt_ratio)
    if metric == "pct_change":
        return lambda v: _num(v, lambda f: "new" if math.isinf(f) else f"{f:+.0f}%")
    if metric in ("share_a", "share_b"):
        return lambda v: _num(v, lambda f: f"{f:.1f}%")
    return lambda v: _num(v, lambda f: f"{f:+.1f} pp")  # share_delta


# --------------------------------------------------------------------------- Comparison


class Comparison:
    """Two views - side A and side B - on one shared layout, compared cell by cell.

    Build one with :meth:`View.compare`; the sides are :attr:`a` and :attr:`b` (ordinary
    :class:`View` objects on the shared layout). :meth:`table` is the comparison itself
    under the current :attr:`metric`; :meth:`top` lists the cells that moved most;
    :meth:`sides` gives the two aligned raw tables. Slicing (:meth:`slice`,
    :meth:`exclude`, ...) applies to both sides at once and :meth:`toggle` compares the
    histograms instead, so a comparison can be interrogated like a view.
    """

    def __init__(
        self,
        a: View,
        b: View,
        *,
        names: Optional[Sequence[str]] = None,
        metric: Optional[str] = None,
        base: Optional[View] = None,
        _split: bool = False,
        _aligned: bool = False,
        _parent: Optional["Comparison"] = None,
    ):
        if not _aligned:
            layout = freeze(a.layout, a.data)
            _check_columns(b, layout, "B")
            a = _on_layout(a, layout)
            b = _on_layout(b, layout, mode=a.mode)
        self.a: View = a
        self.b: View = b
        self._base = base if base is not None else a
        self._split = _split
        self._parent = _parent
        self.names: Tuple[str, str] = self._default_names(names)
        default = "lift" if (_split and _agg_of(self.layout) in ADDITIVE) else "delta"
        self.metric: str = self._check_metric(default if metric is None else metric)
        self._cache: Dict[Any, Any] = {}

    # ------------------------------------------------------------------ state

    @property
    def layout(self) -> Layout:
        return self.a.layout

    @property
    def mode(self) -> str:
        return self.a.mode

    @property
    def is_hist(self) -> bool:
        return self.a.is_hist

    @property
    def display(self) -> Dict[str, Any]:
        return self.a.display

    @property
    def additive(self) -> bool:
        """Whether the measure adds up (count/sum), i.e. shares and ``lift`` make sense."""
        return _agg_of(self.layout) in ADDITIVE

    @property
    def metric_help(self) -> str:
        return METRIC_HELP[self.metric]

    def _common_filters(self) -> Tuple[Filter, ...]:
        """Slices both sides carry (whatever their position: a slice added after the split
        sits behind the defining one on each side)."""
        return tuple(f for f in self.a.filters if f in self.b.filters)

    def _own_filters(self, side: View) -> Tuple[Filter, ...]:
        common = self._common_filters()
        return tuple(f for f in side.filters if f not in common)

    def _default_names(self, names: Optional[Sequence[str]]) -> Tuple[str, str]:
        if names is not None:
            if len(names) != 2:
                raise ValueError("names must be a pair: (name of side A, name of side B)")
            return str(names[0]), str(names[1])
        na, nb = _label(self._own_filters(self.a)) or "A", _label(self._own_filters(self.b)) or "B"
        return (na, nb) if na != nb else (f"{na} (A)", f"{nb} (B)")

    def _check_metric(self, m: str) -> str:
        if m not in METRICS:
            raise ValueError(f"unknown metric {m!r}; use one of {METRICS}")
        if m in ("lift", "share_delta", "share_a", "share_b") and not self.additive:
            raise ValueError(
                f"{m!r} compares shares of a total, which needs an additive measure (sum/count); this layout's "
                f"measure is {self.layout.measure} - use 'delta', 'ratio' or 'pct_change'"
            )
        return m

    def _clone(self, **changes: Any) -> "Comparison":
        kw: Dict[str, Any] = dict(a=self.a, b=self.b, names=self.names, metric=self.metric, base=self._base,
                                  _split=self._split, _aligned=True, _parent=self._parent)
        kw.update(changes)
        return Comparison(**kw)

    def with_metric(self, metric: str) -> "Comparison":
        """The same comparison under another metric (see :data:`METRICS` / :attr:`metric_help`)."""
        return self._clone(metric=metric)

    def swap(self) -> "Comparison":
        """Sides exchanged: B becomes the reference."""
        return self._clone(a=self.b, b=self.a, names=(self.names[1], self.names[0]))

    # ------------------------------------------------------------------ tables

    def sides(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """The two raw tables, aligned: same row labels, same column labels, same order."""
        if "sides" not in self._cache:
            with log.step("compare", self.describe()):
                frames = [self._base.data] if self._split else [self.a.data, self.b.data]
                if self.is_hist:
                    ta, tb = align_tables([self.a._bins_full(), self.b._bins_full()], self.layout, frames)
                    keep = _joint_trim_index([ta, tb], self.layout)
                    ta, tb = ta.loc[keep], tb.loc[keep]
                else:
                    ta, tb = align_tables([self.a.pivot(), self.b.pivot()], self.layout, frames)
                self._cache["sides"] = (ta, tb)
        return self._cache["sides"]

    def _arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        ta, tb = self.sides()
        return ta.to_numpy(dtype=float), tb.to_numpy(dtype=float)

    def table(self, metric: Optional[str] = None) -> pd.DataFrame:
        """The comparison as one table of the metric (default: :attr:`metric`)."""
        m = self.metric if metric is None else self._check_metric(metric)
        key = ("table", m)
        if key not in self._cache:
            a, b = self._arrays()
            ta, _ = self.sides()
            self._cache[key] = pd.DataFrame(_metric_values(a, b, m), index=ta.index, columns=ta.columns)
        return self._cache[key]

    @property
    def shape(self) -> Tuple[int, int]:
        return self.table().shape

    def top(self, n: int = 10, metric: Optional[str] = None) -> pd.DataFrame:
        """The ``n`` cells that differ most between the sides, biggest first: ``row``,
        ``col``, both raw values (columns named after the sides), ``delta``, ``ratio``,
        ``lift`` (additive measures only), the ranking metric when it is another one, and
        ``only_in`` (the side's name when the other side has nothing there).

        Ratio-like metrics (``ratio``, ``lift``, ``pct_change``) rank by a *shrunk* log
        ratio - the same score the heatmap is coloured by - so a x4 and a x0.25 are
        equally big moves, a x142 built on a handful of rows ranks below a x6 built on
        millions, and a cell present on one side only ranks by how much is actually
        there. The values listed are the exact ones.
        """
        m = self.metric if metric is None else self._check_metric(metric)
        a, b = self._arrays()
        ta, _ = self.sides()
        mv = _metric_values(a, b, m)
        sc = _score_values(m, a, b, mv)
        score = np.abs(mv if sc is None else sc)
        score = np.where(np.isfinite(score), score, -1.0)
        empty_a = ~np.isfinite(a) | (a == 0)
        empty_b = ~np.isfinite(b) | (b == 0)
        score[empty_a & empty_b] = -1.0
        na, nb = self.names
        rl, cl = _flat(ta.index), _flat(ta.columns)
        with np.errstate(all="ignore"):
            delta, ratio = a - b, _ratio(a, b)
            lift = _ratio(_share(a), _share(b)) if self.additive else None
        rows: List[Dict[str, Any]] = []
        for flat in np.argsort(-score, axis=None)[: max(0, int(n))]:
            if score.flat[flat] < 0:
                break
            i, j = np.unravel_index(int(flat), score.shape)
            rec: Dict[str, Any] = {"row": rl[i], "col": cl[j], na: a[i, j], nb: b[i, j],
                                   "delta": float(delta[i, j]), "ratio": float(ratio[i, j])}
            if lift is not None:
                rec["lift"] = float(lift[i, j])
            if m not in ("delta", "ratio", "lift", "a", "b"):
                rec[m] = float(mv[i, j])
            rec["only_in"] = na if (empty_b[i, j] and not empty_a[i, j]) else (nb if (empty_a[i, j] and not empty_b[i, j]) else None)
            rows.append(rec)
        cols = ["row", "col", na, nb, "delta", "ratio"] + (["lift"] if lift is not None else []) + \
               ([m] if m not in ("delta", "ratio", "lift", "a", "b") else []) + ["only_in"]
        return pd.DataFrame(rows, columns=cols)

    # ------------------------------------------------------------------ slicing (both sides)

    def slice(self, *args: Any, **kwargs: Any) -> "Comparison":
        """Slice both sides (same forms as :meth:`View.slice`); the layout stays put."""
        return self._clone(a=self.a.slice(*args, refit=False, **kwargs), b=self.b.slice(*args, refit=False, **kwargs))

    def where(self, expr: str, **kw: Any) -> "Comparison":
        return self.slice(expr, **kw)

    def exclude(self, *args: Any, **kwargs: Any) -> "Comparison":
        """Drop matching rows from both sides."""
        new = tuple(f.negate() for f in _new_filters(self.a, _spec_from_args(args, kwargs)))
        return self._clone(a=self.a._clone(filters=self.a.filters + new), b=self.b._clone(filters=self.b.filters + new))

    def unslice(self, *columns: str) -> "Comparison":
        """Drop the slices shared by both sides on ``columns`` (all shared slices when none
        are given); the slices that *define* the sides are kept."""
        common = self._common_filters()
        drop = list(common) if not columns else [f for f in common if f.column in columns or f.label in columns]
        return self._clone(a=self.a._clone(filters=tuple(f for f in self.a.filters if f not in drop)),
                           b=self.b._clone(filters=tuple(f for f in self.b.filters if f not in drop)))

    def style(self, **display: Any) -> "Comparison":
        """Display options for both sides (see :meth:`View.style`); for a histogram
        comparison ``normalize=False`` plots raw values instead of % of each side."""
        return self._clone(a=self.a.style(**display), b=self.b.style(**display))

    # ------------------------------------------------------------------ modes

    def toggle(self, on: Optional[str] = None) -> "Comparison":
        """Compare the histograms instead (and back), like :meth:`View.toggle`."""
        if on is not None:
            return self.histogram(on)
        if not self.is_hist:
            return self._clone(a=self.a._clone(mode=HIST), b=self.b._clone(mode=HIST), _parent=self)
        if self._parent is not None:
            p = self._parent
            return p._clone(a=p.a._clone(filters=self.a.filters), b=p.b._clone(filters=self.b.filters), _parent=None)
        return self._clone(a=self.a._clone(mode=PIVOT), b=self.b._clone(mode=PIVOT), _parent=None)

    def as_hist(self) -> "Comparison":
        return self if self.is_hist else self.toggle()

    def as_pivot(self) -> "Comparison":
        return self.toggle() if self.is_hist else self

    def histogram(self, on: Optional[str] = None, by: Any = None, **kw: Any) -> "Comparison":
        """Compare histograms of ``on`` (see :meth:`View.histogram`); the bins are planned
        on both sides' data together, so they line up."""
        if self._split:
            planner = self._base
        else:
            both = pd.concat([self.a.data, self.b.data], ignore_index=True)
            planner = View(both, self.layout, options=self.a.options, display=self.a.display)
        h = planner.histogram(on, by, **kw)
        layout = freeze(h.layout, planner.data)
        _check_columns(self.b, layout, "B")
        parent = self._parent if self.is_hist else self
        return self._clone(a=_on_layout(self.a, layout, mode=HIST), b=_on_layout(self.b, layout, mode=HIST), _parent=parent)

    # ------------------------------------------------------------------ rendering

    def describe(self) -> str:
        na, nb = self.names
        return (f"{self.layout.describe()} · {na} ({len(self.a.data):,} rows) vs {nb} "
                f"({len(self.b.data):,} rows) · {self.metric}")

    def title(self) -> str:
        s = f"compare{' (hist)' if self.is_hist else ''} · {self.describe()}"
        common = self._common_filters()
        if common:
            s += " · slices: " + " & ".join(f.label for f in common)
        return s

    def _tooltips(self) -> np.ndarray:
        a, b = self._arrays()
        na, nb = self.names
        with np.errstate(all="ignore"):
            delta, ratio = a - b, _ratio(a, b)
            lift = _ratio(_share(a), _share(b)) if self.additive else None
        out = np.empty(a.shape, dtype=object)
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                parts = [f"{na}: {fmt_cell(a[i, j])}", f"{nb}: {fmt_cell(b[i, j])}",
                         "Δ " + _num(delta[i, j], _fmt_delta), _num(ratio[i, j], _fmt_ratio)]
                if lift is not None:
                    parts.append("lift " + _num(lift[i, j], _fmt_ratio))
                out[i, j] = " · ".join(p for p in parts if p.strip() not in ("Δ", ""))
        return out

    def _hist_table(self) -> Tuple[pd.DataFrame, bool]:
        """Both sides as series of one histogram table (and whether it is % of each side)."""
        ta, tb = self.sides()
        na, nb = self.names
        normalize = self.display.get("normalize")
        normalize = self.additive if normalize is None else (bool(normalize) and self.additive)
        a, b = self._arrays()
        if normalize:
            a, b = _share(a), _share(b)
        if ta.shape[1] == 1:
            table = pd.DataFrame({na: a[:, 0], nb: b[:, 0]}, index=ta.index)
        else:
            table = pd.concat({na: pd.DataFrame(a, index=ta.index, columns=ta.columns),
                               nb: pd.DataFrame(b, index=ta.index, columns=ta.columns)}, axis=1)
        return table, normalize

    def html(self, *, side_by_side: bool = False, title: bool = True) -> str:
        """Rich HTML: a diverging heatmap of the metric (blue = more in A, red = less),
        every cell's hover text carrying both raw values, the delta, the ratio and the
        lift. ``side_by_side=True`` adds the two raw tables on one shared colour scale.
        Histogram comparisons draw both sides as paired bars (% of each side by default).
        """
        d = self.display
        theme = str(d.get("theme", "light"))
        t = self.title() if title else None
        with log.step("render", "compare html"):
            if self.is_hist:
                table, normalized = self._hist_table()
                return hist_svg(
                    table, title=(t + " · % of each side") if (t and normalized) else t,
                    width=int(d.get("width", 760)), height=int(d.get("height", 340)), stacked=False,
                    log_y=bool(d.get("log_y", False)), show_values=d.get("show_values"), theme=theme,
                )
            m = self.table()
            a, b = self._arrays()
            hv = _score_values(self.metric, a, b, m.to_numpy(dtype=float))
            heat = str(d.get("heat", "table"))
            if heat != "none" and hv is not None:
                heat = "table"
            common = dict(compact=bool(d.get("compact", False)), max_rows=d.get("max_rows"), agg=_agg_of(self.layout), theme=theme)
            note = SHRUNK_NOTE if (self.metric in ("ratio", "lift", "pct_change") and heat != "none") else None
            main = pivot_html(m, title=t, heat=heat, heat_values=hv, cell_format=_formatter(self.metric),
                              tooltips=self._tooltips(), note=note, **common)
            if not side_by_side:
                return main
            ta, tb = self.sides()
            (ha, hb), peak = _shared_scale(list(self._arrays()))
            na, nb = self.names
            pa = pivot_html(ta, title=f"{na} · {len(self.a.data):,} rows", heat=heat, heat_values=ha, heat_vmax=peak, **common)
            pb = pivot_html(tb, title=f"{nb} · {len(self.b.data):,} rows", heat=heat, heat_values=hb, heat_vmax=peak, **common)
            return grid_html([pa, pb, main], theme=theme)

    def _repr_html_(self) -> str:
        return self.html()

    def render(self, *, width: Optional[int] = None, max_rows: Optional[int] = None, top: int = 5,
               title: bool = True, ascii_only: bool = False) -> str:
        """Text rendering: the metric table, then the ``top`` biggest movers."""
        head = self.title() if title else None
        with log.step("render", "compare text"):
            if self.is_hist:
                table, normalized = self._hist_table()
                return render_hist(table, title=(head + " · % of each side") if (head and normalized) else head,
                                   width=width, ascii_only=ascii_only, compact=True, max_rows=max_rows)
            body = render_pivot(self.table(), width=width, max_rows=max_rows, formatter=_formatter(self.metric))
            out = (head + "\n" if head else "") + body
            if top:
                movers = self.top(top)
                if not movers.empty:
                    shown = movers.drop(columns=["only_in"]).copy()
                    for c in shown.columns:
                        if c in ("delta",):
                            shown[c] = shown[c].map(lambda v: _num(v, _fmt_delta))
                        elif c in ("ratio", "lift"):
                            shown[c] = shown[c].map(lambda v: _num(v, _fmt_ratio))
                        elif c in self.names:
                            shown[c] = shown[c].map(fmt_cell)
                        elif c not in ("row", "col"):
                            shown[c] = shown[c].map(_formatter(c))
                    out += f"\n\ntop {len(shown)} by |{self.metric}| ({self.metric_help}):\n" + shown.to_string(index=False)
            return out

    def show(self, **kw: Any) -> None:
        print(self.render(**kw))

    def __repr__(self) -> str:
        return self.render()

    __str__ = __repr__

    # ------------------------------------------------------------------ export

    def to_dict(self, n: int = 10) -> Dict[str, Any]:
        """JSON-friendly: both sides (name, rows, slices, total), the metric and what it
        means, the metric table as records, and the ``n`` biggest movers."""
        na, nb = self.names
        a, b = self._arrays()
        m = self.table()

        def side(name: str, v: View, arr: np.ndarray) -> Dict[str, Any]:
            fin = arr[np.isfinite(arr)]
            return {"name": name, "rows": int(len(v.data)), "slices": list(v.slices),
                    "total": (float(fin.sum()) if (self.additive and fin.size) else None)}

        return {
            "description": self.describe(),
            "mode": self.mode,
            "layout": self.layout.to_dict(),
            "metric": self.metric,
            "metric_meaning": self.metric_help,
            "a": side(na, self.a, a),
            "b": side(nb, self.b, b),
            "shape": [int(x) for x in m.shape],
            "table": _records(m),
            "top": _json_safe(self.top(n).to_dict(orient="records")),
        }

    def to_json(self, **kw: Any) -> str:
        kw.setdefault("indent", 2)
        return json.dumps(self.to_dict(), **kw)

    def prompt(self, question: Optional[str] = None, **kw: Any) -> Any:
        """This comparison as a self-contained LLM prompt (see :meth:`View.prompt`): the
        dataset, the comparison (both sides, the metric table, the biggest movers) and
        the question; the plain table and anomaly sections are off unless asked for."""
        from ._prompt import build_prompt

        base = self._base if self._split else self.a
        kw.setdefault("table", False)
        kw.setdefault("anomalies", False)
        return build_prompt(base, question, compare=self, **kw)

    def llm_context(self, *, max_rows: int = 30, max_cols: int = 12, top: int = 5) -> Dict[str, Any]:
        """``{"description", "metadata", "table"}`` for a model: what was compared and
        how, the biggest movers in words, and the metric table as markdown."""
        md, truncated = markdown_table(self.table(), max_rows=max_rows, max_cols=max_cols)
        movers = _json_safe(self.top(top).to_dict(orient="records"))
        na, nb = self.names
        parts = [f"Comparison of {na} ({len(self.a.data):,} rows) against {nb} ({len(self.b.data):,} rows), "
                 f"laid out as {self.layout.describe()}, under metric '{self.metric}' ({self.metric_help})."]
        if movers:
            fmt = _formatter(self.metric)
            key = self.metric if self.metric in ("delta", "ratio", "lift") else (self.metric if self.metric in movers[0] else "delta")
            words = "; ".join(
                f"{r['row']} / {r['col']}: {fmt_cell(r[na])} vs {fmt_cell(r[nb])} ({fmt(r[key]) or 'n/a'})"
                for r in movers
            )
            parts.append(f"Biggest movers: {words}.")
        if truncated:
            parts.append("The table below is truncated.")
        return {
            "description": " ".join(parts),
            "metadata": {
                "mode": self.mode, "measure": self.layout.measure, "metric": self.metric, "metric_meaning": self.metric_help,
                "rows": [d.label for d in self.layout.rows], "cols": [d.label for d in self.layout.cols],
                "a": {"name": na, "rows": int(len(self.a.data)), "slices": list(self.a.slices)},
                "b": {"name": nb, "rows": int(len(self.b.data)), "slices": list(self.b.slices)},
                "shape": {"rows": int(self.shape[0]), "cols": int(self.shape[1])}, "truncated": truncated,
                "top": movers,
            },
            "table": md,
        }


# --------------------------------------------------------------------------- Facets


class Facets:
    """Small multiples: one panel per value of a column, all on the same layout with one
    shared colour scale (or y axis), so the panels can be read against each other.
    Build with :meth:`View.facet`.
    """

    def __init__(self, base: View, column: str, values: Sequence[Any], views: Sequence[View], *, hidden: int = 0):
        self.base = base
        self.column = column
        self.values: List[Any] = list(values)
        self.views: List[View] = list(views)
        self.labels: List[str] = [B.NULL if v is None else str(v) for v in self.values]
        self.hidden = int(hidden)
        self._cache: Dict[str, Any] = {}

    def __len__(self) -> int:
        return len(self.views)

    def __iter__(self) -> Iterator[View]:
        return iter(self.views)

    def _index(self, key: Union[int, str]) -> int:
        if isinstance(key, str):
            if key not in self.labels:
                raise KeyError(f"no facet {key!r}; have {self.labels}")
            return self.labels.index(key)
        return range(len(self.views))[key]

    def __getitem__(self, key: Union[int, str]) -> View:
        return self.views[self._index(key)]

    @property
    def layout(self) -> Layout:
        return self.base.layout

    @property
    def mode(self) -> str:
        return self.base.mode

    @property
    def is_hist(self) -> bool:
        return self.base.is_hist

    @property
    def display(self) -> Dict[str, Any]:
        return self.base.display

    def _clone(self, base: View, views: Sequence[View]) -> "Facets":
        return Facets(base, self.column, self.values, views, hidden=self.hidden)

    def tables(self) -> List[pd.DataFrame]:
        """One table per facet, aligned to the same row/column labels."""
        if "tables" not in self._cache:
            with log.step("facet", self.describe()):
                if self.is_hist:
                    ts = align_tables([v._bins_full() for v in self.views], self.layout, [self.base.data])
                    keep = _joint_trim_index(ts, self.layout)
                    ts = [t.loc[keep] for t in ts]
                else:
                    ts = align_tables([v.pivot() for v in self.views], self.layout, [self.base.data])
                self._cache["tables"] = ts
        return self._cache["tables"]

    def compare(self, a: Union[int, str], b: Optional[Union[int, str]] = None, *, metric: Optional[str] = None) -> Comparison:
        """Facet ``a`` against facet ``b`` (by index or label), or against everything
        else in the data when ``b`` is omitted."""
        ia = self._index(a)
        if b is None:
            return compare(self.base, **{self.column: self.values[ia]}, metric=metric)
        ib = self._index(b)
        return Comparison(self.views[ia], self.views[ib], names=(self.labels[ia], self.labels[ib]), metric=metric,
                          base=self.base, _split=True, _aligned=True)

    # -- slicing applies to every facet
    def slice(self, *args: Any, **kwargs: Any) -> "Facets":
        return self._clone(self.base.slice(*args, refit=False, **kwargs), [v.slice(*args, refit=False, **kwargs) for v in self.views])

    def where(self, expr: str, **kw: Any) -> "Facets":
        return self.slice(expr, **kw)

    def exclude(self, *args: Any, **kwargs: Any) -> "Facets":
        new = tuple(f.negate() for f in _new_filters(self.base, _spec_from_args(args, kwargs)))
        return self._clone(self.base._clone(filters=self.base.filters + new), [v._clone(filters=v.filters + new) for v in self.views])

    def style(self, **display: Any) -> "Facets":
        return self._clone(self.base.style(**display), [v.style(**display) for v in self.views])

    def toggle(self, on: Optional[str] = None) -> "Facets":
        """Histograms per facet instead of pivots (and back), on shared bins."""
        if on is not None:
            return self.histogram(on)
        if not self.is_hist:
            return self._clone(self.base._clone(mode=HIST), [v._clone(mode=HIST) for v in self.views])
        return self._clone(self.base._clone(mode=PIVOT), [v._clone(mode=PIVOT) for v in self.views])

    def as_hist(self) -> "Facets":
        return self if self.is_hist else self.toggle()

    def as_pivot(self) -> "Facets":
        return self.toggle() if self.is_hist else self

    def histogram(self, on: Optional[str] = None, by: Any = None, **kw: Any) -> "Facets":
        h = self.base.histogram(on, by, **kw)
        layout = freeze(h.layout, self.base.data)
        return self._clone(_on_layout(self.base, layout, mode=HIST), [_on_layout(v, layout, mode=HIST) for v in self.views])

    # -- rendering
    def describe(self) -> str:
        return f"{self.layout.describe()} · by {self.column} ({len(self.views)} facets)"

    def title(self) -> str:
        s = f"facets{' (hist)' if self.is_hist else ''} · {self.describe()}"
        if self.base.filters:
            s += " · slices: " + " & ".join(f.label for f in self.base.filters)
        return s

    def _panel_titles(self) -> List[str]:
        return [f"{self.column} = {lab} · {len(v.data):,} rows" for lab, v in zip(self.labels, self.views)]

    def html(self, *, title: bool = True) -> str:
        """All facets side by side on one colour scale (pivots) or one y axis (histograms)."""
        d = self.display
        theme = str(d.get("theme", "light"))
        tables = self.tables()
        titles = self._panel_titles()
        note = f"{self.hidden:,} more value(s) of {self.column} not shown" if self.hidden else None
        with log.step("render", "facets html"):
            if self.is_hist:
                arrays = [t.to_numpy(dtype=float) for t in tables]
                fin = [np.abs(x[np.isfinite(x)]) for x in arrays]
                vmax = max((float(f.max()) for f in fin if f.size), default=0.0)
                width = max(300, int(d.get("width", 760)) // 2)
                panels = [
                    hist_svg(t, title=ttl, width=width, height=int(d.get("height", 340)), stacked=bool(d.get("stacked", False)),
                             log_y=bool(d.get("log_y", False)), show_values=d.get("show_values"), theme=theme, vmax=vmax)
                    for t, ttl in zip(tables, titles)
                ]
            else:
                hvs, peak = _shared_scale([t.to_numpy(dtype=float) for t in tables])
                heat = str(d.get("heat", "table"))
                panels = [
                    pivot_html(t, title=ttl, heat=heat, heat_values=hv, heat_vmax=peak, compact=bool(d.get("compact", False)),
                               max_rows=d.get("max_rows"), agg=_agg_of(self.layout), theme=theme)
                    for t, ttl, hv in zip(tables, titles, hvs)
                ]
            return grid_html(panels, title=self.title() if title else None, note=note, theme=theme)

    def _repr_html_(self) -> str:
        return self.html()

    def render(self, *, width: Optional[int] = None, max_rows: Optional[int] = None, title: bool = True, ascii_only: bool = False) -> str:
        tables = self.tables()
        out: List[str] = [self.title()] if title else []
        for t, ttl in zip(tables, self._panel_titles()):
            if self.is_hist:
                out.append(render_hist(t, title=ttl, width=width, ascii_only=ascii_only, compact=True, max_rows=max_rows))
            else:
                out.append(ttl + "\n" + render_pivot(t, width=width, max_rows=max_rows, compact=bool(self.display.get("compact", False))))
        if self.hidden:
            out.append(f"... {self.hidden:,} more value(s) of {self.column} not shown")
        return "\n\n".join(out)

    def show(self, **kw: Any) -> None:
        print(self.render(**kw))

    def __repr__(self) -> str:
        return self.render()

    __str__ = __repr__

    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.describe(),
            "mode": self.mode,
            "column": self.column,
            "layout": self.layout.to_dict(),
            "hidden": self.hidden,
            "facets": [
                {"value": _json_safe(v), "label": lab, "rows": int(len(view.data)), "shape": [int(x) for x in t.shape], "table": _records(t)}
                for v, lab, view, t in zip(self.values, self.labels, self.views, self.tables())
            ],
        }


# --------------------------------------------------------------------------- entry points


def _split(base: View, *args: Any, **kwargs: Any) -> Tuple[View, View, Tuple[str, str]]:
    """(a, b, names) for the split forms of :meth:`View.compare`."""
    columns = set(base.data.columns) | set(base.derived)
    if len(args) == 3 and isinstance(args[0], str):
        if kwargs:
            raise TypeError("compare(column, a, b) takes no extra keywords")
        col, va, vb = args
        return _two_sided(base, {col: va}, {col: vb})
    if len(args) == 2:
        x, y = args
        if isinstance(x, str) and x in columns and not isinstance(y, Mapping):
            kwargs = {x: y, **kwargs}
            args = ()
        elif isinstance(x, (str, Mapping)) and isinstance(y, (str, Mapping)):
            if kwargs:
                raise TypeError("compare(side_a, side_b) takes no extra keywords")
            return _two_sided(base, x, y)
        else:
            raise TypeError("compare() takes (column, value), (column, a, b), two side specs, a query string, a View or column=value")
    if len(args) == 1:
        x = args[0]
        if isinstance(x, str):
            if kwargs:
                raise TypeError("a query string cannot be combined with column=value keywords")
            return _one_sided(base, x)
        if isinstance(x, Mapping):
            kwargs = {**x, **kwargs}
        else:
            raise TypeError(f"cannot compare on {type(x).__name__}: pass a View, a query string, or column=value")
    elif args:
        raise TypeError("compare() takes (column, value), (column, a, b), two side specs, a query string, a View or column=value")
    if len(kwargs) != 1:
        raise TypeError(f"compare() splits on exactly one column=value (got {sorted(kwargs) or 'none'}); "
                        "to split on several at once pass a query string, or two side specs for A vs B")
    return _one_sided(base, kwargs)


def _one_sided(base: View, spec: Spec) -> Tuple[View, View, Tuple[str, str]]:
    new = _new_filters(base, spec)
    if not new:
        raise ValueError("nothing to split on")
    layout = freeze(_layout_without(base, _spec_columns(spec)), base.data)
    on = _on_layout(base, layout)
    a = on._clone(filters=on.filters + new)
    b = on._clone(filters=on.filters + tuple(f.negate() for f in new))
    return a, b, (_label(new), REST)


def _two_sided(base: View, sa: Spec, sb: Spec) -> Tuple[View, View, Tuple[str, str]]:
    fa, fb = _new_filters(base, sa), _new_filters(base, sb)
    layout = freeze(_layout_without(base, _spec_columns(sa) + _spec_columns(sb)), base.data)
    on = _on_layout(base, layout)
    return (on._clone(filters=on.filters + fa), on._clone(filters=on.filters + fb), (_label(fa) or "A", _label(fb) or "B"))


def compare(base: View, *args: Any, metric: Optional[str] = None, names: Optional[Sequence[str]] = None, **kwargs: Any) -> Comparison:
    """See :meth:`View.compare`."""
    if len(args) == 1 and isinstance(args[0], View):
        if kwargs:
            raise TypeError("compare(view) takes no column=value keywords")
        return Comparison(base, args[0], metric=metric, names=names)
    a, b, auto = _split(base, *args, **kwargs)
    return Comparison(a, b, names=names if names is not None else auto, metric=metric, base=base, _split=True, _aligned=True)


def facet(base: View, column: str, n: int = 6, *, levels: Optional[Sequence[Any]] = None) -> Facets:
    """See :meth:`View.facet`."""
    if column not in base.data.columns:
        raise KeyError(f"unknown column {column!r}")
    if levels is None:
        vc = base.data[column].value_counts(dropna=False)
        vals = [None if _isnull(k) else k for k in vc.index[: max(1, int(n))]]
        hidden = max(0, len(vc) - len(vals))
    else:
        vals = list(levels)
        hidden = 0
    if not vals:
        raise ValueError(f"{column!r} has no values to facet on")
    layout = freeze(_layout_without(base, [column]), base.data)
    on = _on_layout(base, layout)
    return Facets(on, column, vals, [on.slice(refit=False, **{column: v}) for v in vals], hidden=hidden)


__all__ = ["Comparison", "Facets", "compare", "facet", "align_tables", "METRICS", "METRIC_HELP", "ADDITIVE"]
