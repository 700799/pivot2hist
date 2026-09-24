"""The View: one dataset, one layout, a mode (``pivot`` or ``hist``) and a stack of slices.

Views are immutable: every operation returns a new View sharing the same source frame.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from . import _binning as B
from ._fit import COUNT, Dim, DimSpec, FitOptions, Layout, build_table, default_agg, fit_layout, plan_dim
from ._fit import _resolve_axis
from ._profile import BOOLEAN, CATEGORICAL, DATETIME, NUMERIC, Profile, profile
from ._render import hist_html, render_hist, render_pivot

PIVOT = "pivot"
HIST = "hist"
MODES = (PIVOT, HIST)

_OP_RE = re.compile(r"^\s*(==|!=|>=|<=|>|<|~|=)\s*(.+?)\s*$")


# --------------------------------------------------------------------------- filters


@dataclass(frozen=True)
class Filter:
    """One slice: a predicate over the source frame."""

    column: Optional[str]
    kind: str  # eq | in | range | op | regex | callable | query
    value: Any
    label: str

    def mask(self, df: pd.DataFrame) -> np.ndarray:
        if self.kind == "query":
            return df.eval(self.value).to_numpy(dtype=bool)
        s = df[self.column]
        if self.kind == "callable":
            m = self.value(s)
            return np.asarray(m, dtype=bool)
        if self.kind == "eq":
            v = self.value
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return s.isna().to_numpy()
            return (s == v).to_numpy(dtype=bool)
        if self.kind == "in":
            vals = list(self.value)
            m = s.isin([v for v in vals if v is not None]).to_numpy(dtype=bool)
            if any(v is None for v in vals):
                m |= s.isna().to_numpy()
            return m
        if self.kind == "range":
            lo, hi = self.value
            m = np.ones(len(s), dtype=bool)
            if lo is not None:
                m &= (s >= lo).to_numpy(dtype=bool)
            if hi is not None:
                m &= (s < hi).to_numpy(dtype=bool)
            return m
        if self.kind == "op":
            op, v = self.value
            ops: Dict[str, Callable[[pd.Series, Any], pd.Series]] = {
                "==": lambda a, b: a == b, "!=": lambda a, b: a != b, ">=": lambda a, b: a >= b,
                "<=": lambda a, b: a <= b, ">": lambda a, b: a > b, "<": lambda a, b: a < b,
            }
            return ops[op](s, v).fillna(False).to_numpy(dtype=bool)
        if self.kind == "regex":
            return s.astype(str).str.contains(self.value, regex=True, na=False).to_numpy(dtype=bool)
        raise ValueError(f"unknown filter kind {self.kind!r}")  # pragma: no cover

    def __str__(self) -> str:
        return self.label


def _coerce_scalar(s: pd.Series, v: Any) -> Any:
    """Make ``v`` comparable with column ``s`` (timestamps, numbers from strings)."""
    if pdt.is_datetime64_any_dtype(s) and isinstance(v, str):
        ts = pd.Timestamp(v)
        tz = getattr(s.dt, "tz", None)
        if tz is not None and ts.tzinfo is None:
            ts = ts.tz_localize(tz)
        return ts
    if pdt.is_numeric_dtype(s) and isinstance(v, str):
        try:
            f = float(v)
            return int(f) if f.is_integer() and pdt.is_integer_dtype(s) else f
        except ValueError:
            return v
    if pdt.is_bool_dtype(s) and isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y", "t")
    return v


def _fmt(v: Any) -> str:
    if isinstance(v, str):
        return v
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def make_filter(column: str, spec: Any, df: pd.DataFrame) -> Filter:
    """Build a :class:`Filter` from the many shorthand forms ``View.slice`` accepts."""
    if column not in df.columns:
        raise KeyError(f"unknown column {column!r}; available: {list(df.columns)[:20]}")
    s = df[column]
    if callable(spec):
        return Filter(column, "callable", spec, f"{column}=<fn>")
    if isinstance(spec, slice):
        lo = _coerce_scalar(s, spec.start) if spec.start is not None else None
        hi = _coerce_scalar(s, spec.stop) if spec.stop is not None else None
        return Filter(column, "range", (lo, hi), f"{column}∈[{_fmt(lo) if lo is not None else '-∞'}, {_fmt(hi) if hi is not None else '∞'})")
    if isinstance(spec, tuple) and len(spec) == 2 and not isinstance(spec[0], (list, tuple)):
        return make_filter(column, slice(spec[0], spec[1]), df)
    if isinstance(spec, (list, set, frozenset, np.ndarray, pd.Index, pd.Series)):
        vals = [_coerce_scalar(s, v) for v in list(spec)]
        shown = ",".join(_fmt(v) for v in vals[:5]) + (",…" if len(vals) > 5 else "")
        return Filter(column, "in", vals, f"{column}∈{{{shown}}}")
    if isinstance(spec, str):
        m = _OP_RE.match(spec)
        if m and m.group(1) != "=" or (m and m.group(1) == "=" and not pdt.is_string_dtype(s)):
            op, raw = m.group(1), m.group(2)
            if op == "~":
                return Filter(column, "regex", raw, f"{column}~/{raw}/")
            if op == "=":
                op = "=="
            v = _coerce_scalar(s, raw.strip("'\""))
            if op == "==":
                return make_filter(column, v, df)
            return Filter(column, "op", (op, v), f"{column}{op}{_fmt(v)}")
        if pdt.is_datetime64_any_dtype(s) and re.fullmatch(r"\d{4}(-\d{2}(-\d{2})?)?", spec.strip()):
            start = pd.Timestamp(spec.strip())
            parts = spec.strip().count("-")
            end = start + pd.DateOffset(**{("years", "months", "days")[parts]: 1})
            tz = getattr(s.dt, "tz", None)
            if tz is not None:
                start, end = start.tz_localize(tz), end.tz_localize(tz)
            return Filter(column, "range", (start, end), f"{column}={spec.strip()}")
        spec = _coerce_scalar(s, spec)
    v = _coerce_scalar(s, spec)
    return Filter(column, "eq", v, f"{column}={_fmt(v)}")


# --------------------------------------------------------------------------- view


class View:
    """A pivot table (or its histogram twin) over a frame, with slices.

    Build one with :func:`pivot2hist.fit`; then chain :meth:`slice`, :meth:`toggle`,
    :meth:`histogram`, :meth:`refit`. ``str(view)`` renders it as text.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        layout: Layout,
        *,
        options: Optional[FitOptions] = None,
        mode: str = PIVOT,
        filters: Sequence[Filter] = (),
        parent: Optional["View"] = None,
        spec: Optional[Mapping[str, Any]] = None,
    ):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self._source = data
        self._layout = layout
        self._options = options or FitOptions()
        self._mode = mode
        self._filters: Tuple[Filter, ...] = tuple(filters)
        self._parent = parent
        self._spec: Dict[str, Any] = dict(spec or {})
        self._cache: Dict[str, Any] = {}

    # ------------------------------------------------------------------ construction

    @classmethod
    def fit(
        cls,
        data: pd.DataFrame,
        *,
        rows: Optional[Sequence[DimSpec]] = None,
        cols: Optional[Sequence[DimSpec]] = None,
        values: Optional[str] = None,
        agg: Optional[str] = None,
        options: Optional[FitOptions] = None,
        mode: str = PIVOT,
        **opts: Any,
    ) -> "View":
        options = (options or FitOptions()).replace(**opts) if opts else (options or FitOptions())
        layout = fit_layout(data, options, rows=rows, cols=cols, values=values, agg=agg)
        spec = {"rows": rows, "cols": cols, "values": values, "agg": agg}
        return cls(data, layout, options=options, mode=mode, spec=spec)

    def _clone(self, **changes: Any) -> "View":
        kw = dict(
            data=self._source, layout=self._layout, options=self._options, mode=self._mode,
            filters=self._filters, parent=self._parent, spec=self._spec,
        )
        kw.update(changes)
        return View(**kw)

    # ------------------------------------------------------------------ state

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def layout(self) -> Layout:
        return self._layout

    @property
    def options(self) -> FitOptions:
        return self._options

    @property
    def filters(self) -> Tuple[Filter, ...]:
        """Active slices, oldest first."""
        return self._filters

    @property
    def slices(self) -> List[str]:
        """Human-readable labels of the active slices."""
        return [f.label for f in self._filters]

    @property
    def source(self) -> pd.DataFrame:
        """The full, unsliced frame."""
        return self._source

    @property
    def data(self) -> pd.DataFrame:
        """The sliced frame."""
        if "data" not in self._cache:
            if not self._filters:
                self._cache["data"] = self._source
            else:
                m = np.ones(len(self._source), dtype=bool)
                for f in self._filters:
                    m &= f.mask(self._source)
                self._cache["data"] = self._source[m]
        return self._cache["data"]

    @property
    def profile(self) -> Profile:
        if "profile" not in self._cache:
            o = self._options
            self._cache["profile"] = profile(
                self.data, max_categories=o.max_categories, discrete_max=o.discrete_max, id_ratio=o.id_ratio
            )
        return self._cache["profile"]

    @property
    def shape(self) -> Tuple[int, int]:
        return self.table().shape

    @property
    def is_hist(self) -> bool:
        return self._mode == HIST

    # ------------------------------------------------------------------ tables

    def pivot(self) -> pd.DataFrame:
        """The pivot table (only observed level combinations)."""
        if "pivot" not in self._cache:
            self._cache["pivot"] = build_table(self.data, self._layout, observed=True)
        return self._cache["pivot"]

    def bins(self) -> pd.DataFrame:
        """The histogram table: every planned bin/level on the row axis, series as columns."""
        if "bins" not in self._cache:
            t = build_table(self.data, self._layout, observed=False)
            self._cache["bins"] = self._trim(t)
        return self._cache["bins"]

    def _trim(self, t: pd.DataFrame) -> pd.DataFrame:
        if t.empty:
            return t
        empty = t.isna() | (t == 0)
        if t.index.nlevels > 1:
            t = t.loc[~empty.all(axis=1)]
        elif self._layout.rows and self._layout.rows[0].kind in ("binned", "time"):
            keep = ~empty.all(axis=1).to_numpy()
            if keep.any():
                first, last = int(np.argmax(keep)), len(keep) - int(np.argmax(keep[::-1]))
                t = t.iloc[first:last]
        if isinstance(t.columns, pd.MultiIndex) and len(t.columns) > 1:
            cempty = t.isna() | (t == 0)
            t = t.loc[:, ~cempty.all(axis=0)]
        return t

    def table(self) -> pd.DataFrame:
        """The frame behind the current mode: :meth:`pivot` or :meth:`bins`."""
        return self.bins() if self.is_hist else self.pivot()

    # ------------------------------------------------------------------ slicing

    def slice(self, *args: Any, refit: Union[bool, str] = "auto", **kwargs: Any) -> "View":
        """Add slices (filters) and return the sliced view.

        Forms::

            v.slice(action="deny")                    # equality
            v.slice(dst_port=[22, 3389])              # membership
            v.slice(bytes=(1000, 50000))              # range [lo, hi)
            v.slice(bytes=slice(1000, None))          # open range
            v.slice(bytes=">= 1000")                  # comparison string (== != > >= < <=)
            v.slice(src_ip="~^10\\.0\\.1\\.")             # regex (leading ~)
            v.slice(timestamp="2026-03-02")           # whole day/month/year on datetimes
            v.slice(bytes=lambda s: s > s.median())   # callable mask
            v.slice("bytes > 1000 and action == 'deny'")  # pandas query expression
            v.slice("action", "deny")                 # positional column/value

        ``refit="auto"`` re-runs the auto-fit when a slice collapses one of the layout's
        dimensions to a single value; ``True`` always refits, ``False`` never.
        """
        new: List[Filter] = []
        if len(args) == 1 and isinstance(args[0], str):
            expr = args[0]
            new.append(Filter(None, "query", expr, expr))
        elif len(args) == 1 and isinstance(args[0], Mapping):
            kwargs = {**args[0], **kwargs}
        elif len(args) == 2:
            kwargs = {args[0]: args[1], **kwargs}
        elif args:
            raise TypeError("slice() takes a query string, (column, value) or column=value keywords")
        for col, spec in kwargs.items():
            new.append(make_filter(col, spec, self._source))
        v = self._clone(filters=self._filters + tuple(new))
        if refit is True or (refit == "auto" and v._layout_stale()):
            v = v.refit()
        return v

    def where(self, expr: str, **kw: Any) -> "View":
        """Slice with a pandas query expression (alias for ``slice(expr)``)."""
        return self.slice(expr, **kw)

    def unslice(self, *columns: str, refit: Union[bool, str] = "auto") -> "View":
        """Drop slices on the given columns (all slices when none are given)."""
        if not columns:
            keep: Tuple[Filter, ...] = ()
        else:
            keep = tuple(f for f in self._filters if f.column not in columns and f.label not in columns)
        v = self._clone(filters=keep)
        if refit is True or (refit == "auto" and v._layout_stale()):
            v = v.refit()
        return v

    def slicers(self, n: int = 10) -> Dict[str, List[Tuple[Any, int]]]:
        """Values to slice on: top-``n`` values (with counts) of every label-like column."""
        out: Dict[str, List[Tuple[Any, int]]] = {}
        data = self.data
        seen = []
        for d in self._layout.dims:
            if d.column not in seen:
                seen.append(d.column)
        for cp in self.profile:
            if cp.kind in (CATEGORICAL, BOOLEAN) and cp.name not in seen:
                seen.append(cp.name)
        for col in seen:
            if col not in data.columns:
                continue
            vc = data[col].value_counts(dropna=False).head(n)
            out[col] = [(None if (isinstance(k, float) and np.isnan(k)) else k, int(c)) for k, c in vc.items()]
        return out

    def _layout_stale(self) -> bool:
        data = self.data
        if data.empty:
            return False
        for d in self._layout.dims:
            if d.column in data.columns and data[d.column].nunique(dropna=False) <= 1:
                return True
        return False

    # ------------------------------------------------------------------ refit / relayout

    def refit(self, **opts: Any) -> "View":
        """Re-run the auto-fit on the sliced data, optionally with changed options.

        Axes the user fixed are kept unless a slice collapsed them to a single value.
        """
        options = self._options.replace(**opts) if opts else self._options
        data = self.data
        spec = dict(self._spec)
        for axis in ("rows", "cols"):
            specs = spec.get(axis)
            if specs is None:
                continue
            if isinstance(specs, (str, Dim, dict)):
                specs = [specs]
            kept = []
            for sp in specs:
                col = sp.column if isinstance(sp, Dim) else sp["column"] if isinstance(sp, dict) else str(sp)
                if col in data.columns and data[col].nunique(dropna=False) > 1:
                    kept.append(sp)
            spec[axis] = kept if kept or specs == [] else None
        if data.empty:
            return self._clone(options=options, spec=spec)
        layout = fit_layout(data, options, rows=spec.get("rows"), cols=spec.get("cols"),
                            values=spec.get("values"), agg=spec.get("agg"))
        return self._clone(layout=layout, options=options, spec=spec)

    def relayout(
        self,
        *,
        rows: Optional[Sequence[DimSpec]] = None,
        cols: Optional[Sequence[DimSpec]] = None,
        values: Optional[str] = None,
        agg: Optional[str] = None,
        **opts: Any,
    ) -> "View":
        """Fix some axes/measure by hand and auto-fit the rest (``rows=[]`` clears an axis)."""
        options = self._options.replace(**opts) if opts else self._options
        spec = {"rows": rows, "cols": cols, "values": values, "agg": agg}
        layout = fit_layout(self.data, options, **spec)
        return self._clone(layout=layout, options=options, spec=spec)

    def layers(self, n: int) -> "View":
        """Refit with at most ``n`` stacked dimensions per axis."""
        return self.refit(layers=int(n))

    def fit_to(self, max_rows: Optional[int] = None, max_cols: Optional[int] = None, aspect: Optional[float] = None) -> "View":
        """Refit into a different box."""
        kw: Dict[str, Any] = {}
        if max_rows is not None:
            kw["max_rows"] = int(max_rows)
        if max_cols is not None:
            kw["max_cols"] = int(max_cols)
        if aspect is not None:
            kw["aspect"] = float(aspect)
        return self.refit(**kw)

    # ------------------------------------------------------------------ modes

    def toggle(self, on: Optional[str] = None) -> "View":
        """Flip between the pivot and its histogram.

        From a pivot: the row axis becomes the bins and the column axis the series.
        From a histogram: back to the pivot it came from (keeping any slices added since).
        ``on`` targets a specific column, see :meth:`histogram`.
        """
        if on is not None:
            return self.histogram(on)
        if self._mode == PIVOT:
            return self._clone(mode=HIST, parent=self)
        if self._parent is not None:
            return self._parent._clone(filters=self._filters, parent=None)
        return self._clone(mode=PIVOT, parent=None)

    def as_pivot(self) -> "View":
        return self if self._mode == PIVOT else self.toggle()

    def as_hist(self) -> "View":
        return self if self._mode == HIST else self.toggle()

    def histogram(
        self,
        on: Optional[str] = None,
        by: Optional[Union[str, Sequence[DimSpec]]] = None,
        *,
        bins: Optional[B.BinRule] = None,
        values: Optional[str] = None,
        agg: Optional[str] = None,
        scale: Optional[str] = None,
    ) -> "View":
        """A histogram view.

        With no arguments this is :meth:`toggle`. ``on`` picks the column to bin
        (numeric -> nice bins, datetime -> time buckets, labels -> top-N bars); ``by``
        adds one series per level of another column; ``values``/``agg`` replace the
        default row count (e.g. ``values="bytes", agg="sum"``); ``bins`` is a rule name
        or a count; ``scale`` forces ``"linear"``/``"log"`` bins.
        """
        if on is None and by is None and values is None and bins is None and scale is None:
            return self.toggle() if self._mode == PIVOT else self
        options = self._options.replace(scale=scale) if scale else self._options
        data = self.data
        prof = self.profile
        if on is None:
            on = self._default_hist_column(prof)
        if on not in prof:
            raise KeyError(f"unknown column {on!r}")
        cp = prof[on]
        budget = int(bins) if isinstance(bins, (int, np.integer)) and not isinstance(bins, bool) else options.max_bins
        rule: Optional[B.BinRule] = bins if isinstance(bins, str) else (int(bins) if isinstance(bins, (int, np.integer)) else None)
        kind = "categorical" if cp.kind not in (NUMERIC, DATETIME) else None
        row = plan_dim(data[on], cp, budget, options, bins=rule, kind=kind)
        if row is None:
            raise ValueError(f"cannot bin {on!r}")
        cols = _resolve_axis(by, data, prof, options.max_cols, options) or []
        if values is not None and agg is None:
            agg = default_agg(prof[values]) if values in prof else "sum"
        if values is None and agg not in (None, COUNT):
            values = self._layout.values
        if agg == COUNT:
            values, agg = None, "sum"
        layout = Layout((row,), tuple(cols), values, agg or "sum")
        spec = {"rows": [row], "cols": list(cols), "values": values, "agg": agg}
        parent = self if self._mode == PIVOT else self._parent
        return View(self._source, layout, options=options, mode=HIST, filters=self._filters, parent=parent, spec=spec)

    def _default_hist_column(self, prof: Profile) -> str:
        if self._layout.values is not None and self._layout.values in prof and prof[self._layout.values].kind == NUMERIC:
            return self._layout.values
        for cp in prof:
            if cp.kind == NUMERIC:
                return cp.name
        if self._layout.rows:
            return self._layout.rows[0].column
        return next(iter(prof)).name

    # ------------------------------------------------------------------ rendering

    def describe(self) -> str:
        """The layout as realised on the sliced data (e.g. without a stale "(top N)")."""
        t = self.table()
        rows = " > ".join(str(n) for n in t.index.names if n is not None) or "-"
        cols = " > ".join(str(n).rstrip() for n in t.columns.names if n is not None)
        s = f"{self._layout.measure} by {rows}"
        if cols:
            s += f" x {cols}"
        return s

    def title(self) -> str:
        n, total = len(self.data), len(self._source)
        s = f"{self._mode} \u00b7 {self.describe()} \u00b7 {n:,} rows"
        if n != total:
            s += f" of {total:,}"
        if self._filters:
            s += " \u00b7 slices: " + " & ".join(f.label for f in self._filters)
        return s

    def render(self, *, width: Optional[int] = None, ascii_only: bool = False, compact: Optional[bool] = None,
               max_rows: Optional[int] = None, title: bool = True) -> str:
        """Text rendering of the current mode."""
        head = self.title() + "\n" if title else ""
        if self.is_hist:
            return render_hist(self.bins(), title=head.rstrip("\n") or None, width=width, ascii_only=ascii_only,
                               compact=True if compact is None else compact, max_rows=max_rows)
        return head + render_pivot(self.pivot(), width=width, max_rows=max_rows, compact=bool(compact))

    def show(self, **kw: Any) -> None:
        print(self.render(**kw))

    def __str__(self) -> str:
        return self.render()

    def __repr__(self) -> str:
        return self.render()

    def _repr_html_(self) -> str:
        if self.is_hist:
            return hist_html(self.bins(), title=self.title())
        from ._render import format_table

        return f"<div style='font-family:monospace;margin-bottom:4px'>{self.title()}</div>" + format_table(self.pivot()).to_html()

    def plot(self, ax: Any = None, **kw: Any) -> Any:
        """Draw with matplotlib (optional dependency): bars for histograms, a heatmap for pivots."""
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:  # pragma: no cover
            raise ImportError("plotting needs matplotlib: pip install 'pivot2hist[plot]'") from e
        t = self.table()
        if ax is None:
            _, ax = plt.subplots(figsize=kw.pop("figsize", (9, 5)))
        if self.is_hist:
            plot_t = t.copy()
            plot_t.index = [" / ".join(map(str, i)) if isinstance(i, tuple) else str(i) for i in t.index]
            plot_t.columns = [" / ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in t.columns]
            plot_t.plot.bar(ax=ax, width=0.85, **kw)
            ax.set_ylabel(self._layout.measure)
            if len(plot_t.columns) == 1:
                ax.get_legend().remove()
        else:
            vals = t.to_numpy(dtype=float)
            im = ax.imshow(vals, aspect="auto", cmap=kw.pop("cmap", "Blues"))
            ax.set_yticks(range(len(t.index)))
            ax.set_yticklabels([" / ".join(map(str, i)) if isinstance(i, tuple) else str(i) for i in t.index])
            ax.set_xticks(range(len(t.columns)))
            ax.set_xticklabels([" / ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in t.columns], rotation=45, ha="right")
            ax.figure.colorbar(im, ax=ax, label=self._layout.measure)
        ax.set_title(self.title(), fontsize=9)
        return ax

    # ------------------------------------------------------------------ export

    def to_dict(self) -> Dict[str, Any]:
        """JSON-friendly description: mode, layout, slices and the table as records."""
        t = self.table().copy()
        if isinstance(t.columns, pd.MultiIndex):
            t.columns = [" / ".join(str(x) for x in c) for c in t.columns]
        else:
            t.columns = [str(c) for c in t.columns]
        flat = t.reset_index()
        flat.columns = [str(c) for c in flat.columns]
        records = json.loads(flat.to_json(orient="records", date_format="iso"))
        return {
            "mode": self._mode,
            "layout": self._layout.to_dict(),
            "slices": self.slices,
            "rows": int(len(self.data)),
            "source_rows": int(len(self._source)),
            "shape": list(self.table().shape),
            "table": records,
        }

    def to_json(self, **kw: Any) -> str:
        kw.setdefault("indent", 2)
        return json.dumps(self.to_dict(), **kw)

    def to_csv(self, path: Optional[str] = None, **kw: Any) -> Optional[str]:
        return self.table().to_csv(path, **kw)


__all__ = ["View", "Filter", "make_filter", "PIVOT", "HIST"]
