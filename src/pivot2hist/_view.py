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

from dataclasses import replace as _replace

from . import _binning as B
from ._cluster import METHODS, cluster_frame, cluster_rows, cocluster as _cocluster
from ._fit import COUNT, Dim, DimSpec, FitOptions, Layout, build_table, default_agg, fit_layout, freeze, materialize, order_index, plan_dim
from ._fit import _resolve_axis, suggest_layouts
from ._density import DistFit
from ._html import expected_independence, hist_svg, pivot_html, surprise_residuals
from ._log import log
from ._log import stats as _log_stats
from ._profile import BOOLEAN, CATEGORICAL, DATETIME, NUMERIC, Profile, profile
from ._render import render_hist, render_pivot
from ._survey import PagedSource, Survey

EXACT_AGGS = ("sum", "count", "min", "max", "mean")

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
        if self.kind == "not":
            return ~self.value.mask(df)
        if self.kind == "query":
            return _bools(df.eval(self.value))
        s = df[self.column]
        if self.kind == "callable":
            return _bools(self.value(s))
        if self.kind == "eq":
            v = self.value
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return s.isna().to_numpy()
            return _bools(s == v)
        if self.kind == "in":
            vals = list(self.value)
            m = _bools(s.isin([v for v in vals if v is not None]))
            if any(v is None for v in vals):
                m |= s.isna().to_numpy()
            return m
        if self.kind == "range":
            lo, hi = self.value
            m = np.ones(len(s), dtype=bool)
            if lo is not None:
                m &= _bools(s >= lo)
            if hi is not None:
                m &= _bools(s < hi)
            return m
        if self.kind == "op":
            op, v = self.value
            ops: Dict[str, Callable[[pd.Series, Any], pd.Series]] = {
                "==": lambda a, b: a == b, "!=": lambda a, b: a != b, ">=": lambda a, b: a >= b,
                "<=": lambda a, b: a <= b, ">": lambda a, b: a > b, "<": lambda a, b: a < b,
            }
            return _bools(ops[op](s, v))
        if self.kind == "regex":
            return _bools(s.astype(str).str.contains(self.value, regex=True, na=False))
        raise ValueError(f"unknown filter kind {self.kind!r}")  # pragma: no cover

    def negate(self) -> "Filter":
        if self.kind == "not":
            return self.value
        return Filter(self.column, "not", self, f"not {self.label}")

    def __str__(self) -> str:
        return self.label


_RELATIVE_RE = re.compile(r"^\s*last\s+(\d+(?:\.\d+)?)\s*(s|sec|secs|m|min|mins|h|hr|hrs|hour|hours|d|day|days|w|wk|week|weeks)\s*$", re.I)
_PCT_RE = re.compile(r"^p(\d{1,2}(?:\.\d+)?)$", re.I)
_UNIT = {"s": "s", "sec": "s", "secs": "s", "m": "min", "min": "min", "mins": "min", "h": "h", "hr": "h", "hrs": "h",
         "hour": "h", "hours": "h", "d": "D", "day": "D", "days": "D", "w": "W", "wk": "W", "week": "W", "weeks": "W"}


def _axis_scale(edges: np.ndarray) -> Optional[str]:
    """``"linear"`` for equal widths, ``"log"`` for (roughly) equal ratios, else ``None``."""
    if edges.size < 3:
        return "linear"
    widths = np.diff(edges)
    if np.allclose(widths, widths[0], rtol=1e-6, atol=0):
        return "linear"
    pos = edges[edges > 0]
    if pos.size >= 3 and (edges <= 0).sum() <= 1:
        ratios = pos[1:] / pos[:-1]
        if ratios.min() > 1.3 and ratios.max() / ratios.min() < 3.0:
            return "log"
    return None


def _bools(m: Any) -> np.ndarray:
    """A plain boolean numpy mask; missing values (nullable dtypes) count as False."""
    if isinstance(m, pd.Series):
        if m.dtype == bool:
            return m.to_numpy()
        return m.fillna(False).astype(bool).to_numpy()
    arr = np.asarray(m)
    if arr.dtype == object:
        return np.array([bool(x) if x is not None and x == x else False for x in arr], dtype=bool)
    return arr.astype(bool)


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
        rel = _RELATIVE_RE.match(spec) if pdt.is_datetime64_any_dtype(s) else None
        if rel:
            amount, unit = float(rel.group(1)), _UNIT[rel.group(2).lower()]
            delta = pd.Timedelta(amount * 7, unit="D") if unit == "W" else pd.Timedelta(amount, unit=unit)
            end = s.max()
            start = end - delta
            return Filter(column, "range", (start, end + pd.Timedelta(nanoseconds=1)), f"{column}={spec.strip()}")
        m = _OP_RE.match(spec)
        if m and m.group(1) != "=" or (m and m.group(1) == "=" and not pdt.is_string_dtype(s)):
            op, raw = m.group(1), m.group(2)
            if op == "~":
                return Filter(column, "regex", raw, f"{column}~/{raw}/")
            if op == "=":
                op = "=="
            raw = raw.strip("'\"")
            pct = _PCT_RE.match(raw) if pdt.is_numeric_dtype(s) else None
            if pct:
                q = float(pct.group(1))
                v = float(np.nanpercentile(s.to_numpy(dtype=float, na_value=np.nan), q))
                if op == "==":
                    op = ">="
                return Filter(column, "op", (op, v), f"{column}{op}p{pct.group(1)} ({B.human(v)})")
            v = _coerce_scalar(s, raw)
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


@dataclass(frozen=True)
class Derived:
    """A column computed on the fly: cluster/block labels looked up from a row key, or an
    index-aligned series. Never written into the source frame, so it works page by page."""

    name: str
    dims: Tuple[Dim, ...] = ()
    lookup: Optional[Dict[Any, Any]] = None
    categories: Tuple[Any, ...] = ()
    series: Optional[pd.Series] = None

    def compute(self, df: pd.DataFrame) -> pd.Series:
        if self.series is not None:
            vals = self.series.reindex(df.index)
        else:
            keys = [materialize(df, d).astype(object) for d in self.dims]
            if len(keys) == 1:
                vals = keys[0].map(self.lookup or {})
            else:
                tuples = pd.Series(list(zip(*[k.to_numpy() for k in keys])), index=df.index)
                vals = tuples.map(self.lookup or {})
        return pd.Series(pd.Categorical(vals, categories=list(self.categories), ordered=True), index=df.index, name=self.name)

    def dim(self) -> Dim:
        return Dim(self.name, "categorical", len(self.categories), keep=tuple(self.categories), order="keep", natural=len(self.categories), quality=1.0)


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
        display: Optional[Mapping[str, Any]] = None,
        derived: Optional[Mapping[str, Derived]] = None,
        survey: Optional[Survey] = None,
    ):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self._paged: Optional[PagedSource] = data if isinstance(data, PagedSource) else None
        self._source: pd.DataFrame = data.sample if isinstance(data, PagedSource) else data
        self._survey = survey if survey is not None else (data.survey if isinstance(data, PagedSource) else None)
        self._derived: Dict[str, Derived] = dict(derived or {})
        self._layout = layout
        self._options = options or FitOptions()
        self._mode = mode
        self._filters: Tuple[Filter, ...] = tuple(filters)
        self._parent = parent
        self._spec: Dict[str, Any] = dict(spec or {})
        self._display: Dict[str, Any] = dict(display or {})
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
        frame = data.sample if isinstance(data, PagedSource) else data
        with log.step("fit", f"{len(frame):,} rows x {frame.shape[1]} cols") as st:
            layout = fit_layout(frame, options, rows=rows, cols=cols, values=values, agg=agg)
            st.detail += f" -> {layout.describe()}"
        if isinstance(data, PagedSource):
            layout = freeze(layout, frame)
        spec = {"rows": rows, "cols": cols, "values": values, "agg": agg}
        return cls(data, layout, options=options, mode=mode, spec=spec)

    def _clone(self, **changes: Any) -> "View":
        kw = dict(
            data=self._paged if self._paged is not None else self._source, layout=self._layout, options=self._options,
            mode=self._mode, filters=self._filters, parent=self._parent, spec=self._spec, display=self._display,
            derived=self._derived, survey=self._survey,
        )
        kw.update(changes)
        return View(**kw)

    def clone(self) -> "View":
        """An independent copy of this View, for zero extra memory or disk.

        Views are already immutable, so this is the same object graph with a fresh,
        empty render cache: the underlying frame (or paged source/survey) is *shared by
        reference*, not copied, and every field on the clone (layout, options, filters,
        spec, display, derived columns) is its own independent value, not aliased to the
        original - changing one (e.g. ``.slice()``, ``.style()``) never affects the
        other. Handy for fanning one loaded/profiled dataset out into several notebook
        cells that each explore a different layout, without re-reading the source or
        duplicating it in RAM::

            base = p2h.fit("huge.parquet")   # surveyed, profiled, fitted once
            a = base.clone().slice(action="deny").style(theme="graphite")
            b = base.clone().histogram("bytes")
            # `a` and `b` share the same underlying data; neither's cache or
            # slices/style touch the other, or `base`.
        """
        return self._clone()

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
        """The full, unsliced frame (for a paged source: its fitting sample)."""
        return self._source

    @property
    def paged(self) -> Optional[PagedSource]:
        """The paged source behind this view, or ``None`` when the data is in memory."""
        return self._paged

    @property
    def survey(self) -> Optional[Survey]:
        """Size survey of the source, when it was loaded through :func:`pivot2hist.fit`."""
        return self._survey

    @property
    def derived(self) -> Dict[str, Derived]:
        return dict(self._derived)

    def _with_derived(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived columns in insertion order (a later one may build on an earlier one)."""
        for name, d in self._derived.items():
            if name not in df.columns:
                df = df.assign(**{name: d.compute(df)})
        return df

    def _mask(self, df: pd.DataFrame) -> np.ndarray:
        m = np.ones(len(df), dtype=bool)
        for f in self._filters:
            m &= f.mask(df)
        return m

    @property
    def data(self) -> pd.DataFrame:
        """The sliced frame (for a paged source: the sliced sample)."""
        if "data" not in self._cache:
            base = self._with_derived(self._source)
            self._cache["data"] = base if not self._filters else base[self._mask(base)]
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
            with log.step("pivot", self._layout.describe()) as st:
                if self._paged is not None:
                    self._cache["pivot"] = self._paged_table(observed=True)
                else:
                    self._cache["pivot"] = build_table(self.data, self._layout, observed=True, engine=self._options.engine)
                st.detail += f" -> {self._cache['pivot'].shape[0]} x {self._cache['pivot'].shape[1]}"
        return self._cache["pivot"]

    def bins(self) -> pd.DataFrame:
        """The histogram table: every planned bin/level on the row axis, series as columns."""
        if "bins" not in self._cache:
            with log.step("bins", self._layout.describe()):
                self._cache["bins"] = self._trim(self._bins_full())
        return self._cache["bins"]

    def _bins_full(self) -> pd.DataFrame:
        """:meth:`bins` before empty leading/trailing bins are trimmed: every planned bin."""
        if "bins_full" not in self._cache:
            if self._paged is not None:
                self._cache["bins_full"] = self._paged_table(observed=False)
            else:
                self._cache["bins_full"] = build_table(self.data, self._layout, observed=False, engine=self._options.engine)
        return self._cache["bins_full"]

    @property
    def approximate(self) -> bool:
        """True when a paged view had to compute its measure on the sample (median, std, nunique)."""
        return bool(self._cache.get("approx", False)) or (
            self._paged is not None and self._layout.values is not None and self._layout.agg not in EXACT_AGGS
        )

    def _paged_table(self, *, observed: bool) -> pd.DataFrame:
        """Aggregate page by page. Exact for count/sum/min/max/mean; sample-based otherwise."""
        assert self._paged is not None
        layout = self._layout
        agg = layout.agg if layout.values is not None else "sum"
        if layout.values is not None and agg not in EXACT_AGGS:
            self._cache["approx"] = True
            log.info("pivot", f"{agg} cannot be combined across pages: computed on the {len(self.data):,}-row sample")
            return build_table(self.data, layout, observed=observed, engine=self._options.engine)
        mean = layout.values is not None and agg == "mean"
        sum_layout = _replace(layout, agg="sum") if mean else layout
        cnt_layout = _replace(layout, agg="count") if mean else None
        sums: List[pd.DataFrame] = []
        cnts: List[pd.DataFrame] = []
        seen = 0
        for page in self._paged.pages():
            page = self._with_derived(page)
            if self._filters:
                page = page[self._mask(page)]
            seen += len(page)
            if page.empty:
                continue
            sums.append(build_table(page, sum_layout, observed=observed, engine=self._options.engine))
            if cnt_layout is not None:
                cnts.append(build_table(page, cnt_layout, observed=observed, engine=self._options.engine))
        self._cache["paged_rows"] = seen
        if not sums:
            return build_table(self.data.iloc[0:0], layout, observed=observed, engine=self._options.engine)
        how = {"sum": "sum", "count": "sum", "min": "min", "max": "max", "mean": "sum"}[agg]
        total = self._combine(sums, how)
        if mean:
            n = self._combine(cnts, "sum")
            total = total / n.replace(0, np.nan).reindex_like(total)
        return self._reorder(total)

    @staticmethod
    def _combine(parts: List[pd.DataFrame], how: str) -> pd.DataFrame:
        big = pd.concat(parts, axis=0, sort=False)
        if how == "sum":
            big = big.fillna(0)
        levels = list(range(big.index.nlevels))
        out = big.groupby(level=levels, observed=True, sort=False).agg(how)
        return out

    def _reorder(self, table: pd.DataFrame) -> pd.DataFrame:
        """Order index/columns the way the sample's categories are ordered (unseen labels last)."""
        table = table.reindex(order_index(table.index, self._layout.rows, self.data))
        if self._layout.cols:
            table = table.reindex(columns=order_index(table.columns, self._layout.cols, self.data))
        return table

    def materialize(self, max_rows: Optional[int] = None) -> "View":
        """Pull a paged source into memory (sliced rows only) and return an ordinary view."""
        if self._paged is None:
            return self
        parts: List[pd.DataFrame] = []
        got = 0
        with log.step("materialize", "paged -> memory") as st:
            for page in self._paged.pages():
                if self._filters:
                    page = page[self._mask(self._with_derived(page))]
                parts.append(page)
                got += len(page)
                if max_rows is not None and got >= max_rows:
                    break
            df = pd.concat(parts, ignore_index=True) if parts else self._source.iloc[0:0]
            if max_rows is not None:
                df = df.head(max_rows)
            st.detail += f": {len(df):,} rows"
        return View(df, self._layout, options=self._options, mode=self._mode, filters=(), parent=None,
                    spec=self._spec, display=self._display, derived=self._derived, survey=self._survey)

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
        base = self._with_derived(self._source) if any(c in self._derived for c in kwargs) else self._source
        for col, spec in kwargs.items():
            new.append(make_filter(col, spec, base))
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

    def exclude(self, *args: Any, **kwargs: Any) -> "View":
        """The complement of :meth:`slice`: drop rows matching the given slices."""
        tmp = self.slice(*args, refit=False, **kwargs)
        new = tuple(f.negate() for f in tmp.filters[len(self._filters):])
        v = self._clone(filters=self._filters + new)
        return v.refit() if v._layout_stale() else v

    def top(self, column: str, n: int = 10, *, by: Optional[str] = None) -> "View":
        """Keep only the ``n`` heaviest values of ``column`` (by row count, or by ``sum(by)``)."""
        data = self.data
        if by is None:
            keep = data[column].value_counts().index[:n]
        else:
            keep = data.groupby(column, observed=True)[by].sum().sort_values(ascending=False).index[:n]
        f = Filter(column, "in", list(keep), f"{column}\u2208top{n}" + (f"[{by}]" if by else ""))
        v = self._clone(filters=self._filters + (f,))
        return v.refit() if v._layout_stale() else v

    def drill(self, *args: Any, **kwargs: Any) -> "View":
        """Slice and refit: zoom into a cell/row and let the fit pick the next dimensions."""
        return self.slice(*args, refit=True, **kwargs)

    def sample(self, n: Optional[int] = None, frac: Optional[float] = None, *, seed: int = 0) -> "View":
        """Reduce size: the same view over a random sample of the sliced rows (in memory)."""
        data = self.data
        if n is None and frac is None:
            n = min(len(data), 10_000)
        if n is not None:
            n = min(int(n), len(data))
            src = data.sample(n=n, random_state=seed)
        else:
            src = data.sample(frac=float(frac), random_state=seed)
        return View(src.sort_index(), self._layout, options=self._options, mode=self._mode, filters=self._filters,
                    parent=None, spec=self._spec, display=self._display, derived={}, survey=self._survey)

    def _dim_for(self, column: str) -> Tuple[str, int, Dim]:
        for axis in ("rows", "cols"):
            dims = getattr(self._layout, axis)
            for i, d in enumerate(dims):
                if d.column == column:
                    return axis, i, d
        raise KeyError(f"{column!r} is not a dimension of this layout ({self._layout.describe()})")

    def level(self, column: str, level: Union[str, int]) -> "View":
        """Change how a dimension is bucketed: a semantic level (``"/24"``, ``"class"``,
        ``"host"``), a time bucket (``"h"``, ``"D"``, ``"hour_of_day"``, ``"weekday"``) or a
        bin count for numeric dimensions."""
        axis, i, d = self._dim_for(column)
        if d.kind == "time":
            spec: Dict[str, Any] = {"column": column, "freq": str(level)}
        elif d.kind == "binned":
            n = int(level)
            spec = {"column": column, "bins": n, "levels": max(2, n + 1)}
        else:
            spec = {"column": column, "level": str(level)}
        rows = [x for x in self._layout.rows]
        cols = [x for x in self._layout.cols]
        (rows if axis == "rows" else cols)[i] = spec  # type: ignore[call-overload]
        return self.relayout(rows=rows, cols=cols, **self._measure_spec())

    def _measure_spec(self) -> Dict[str, Any]:
        """values/agg keywords that reproduce this layout's measure (count stays count)."""
        if self._layout.values is None:
            return {"values": None, "agg": COUNT}
        return {"values": self._layout.values, "agg": self._layout.agg}

    def _step(self, column: str, direction: int) -> "View":
        axis, i, d = self._dim_for(column)
        cp = self.profile[column] if column in self.profile else None
        if d.kind == "time":
            freqs = list(B.TIME_FREQS)  # fine -> coarse, so coarser means a later entry
            cur = freqs.index(d.freq) if d.freq in freqs else freqs.index("D")
            nxt = min(max(cur - direction, 0), len(freqs) - 1)
            return self.level(column, freqs[nxt])
        if d.kind == "binned":
            n = max(2, len(d.edges or ()) - 1)
            return self.level(column, max(2, n // 2) if direction < 0 else n * 2)
        levels = list(cp.hierarchy) if cp is not None else []
        if not levels:
            raise ValueError(f"{column!r} has no drill hierarchy")
        cur = levels.index(d.level) if d.level in levels else len(levels) - 1
        nxt = min(max(cur + direction, 0), len(levels) - 1)
        return self.level(column, levels[nxt])

    def coarser(self, column: str) -> "View":
        """Roll a dimension up one level (``/24`` -> ``/16``, hour -> 6 h, fewer bins)."""
        return self._step(column, -1)

    def finer(self, column: str) -> "View":
        """Drill a dimension down one level (``/16`` -> ``/24``, day -> 6 h, more bins)."""
        return self._step(column, +1)

    # ------------------------------------------------------------------ auto-guess

    def suggest_ranked(self, n: int = 5) -> List[Tuple[float, Layout]]:
        """Like :meth:`suggest` but keeps each candidate's raw fit score, best first.

        The score is the auto-fit's internal objective (entropy plus mutual information,
        minus sparsity/aspect/layer/other-bucket penalties, see
        ``pivot2hist.DEFAULT_WEIGHTS``): higher is better, comparable only *within* one
        call on one dataset, not across datasets or box sizes.
        """
        spec = {k: self._spec.get(k) for k in ("rows", "cols", "values", "agg")}
        if self.data.empty:
            return [(0.0, self._layout)]
        out = suggest_layouts(self.data, self._options, n, prof=self.profile, **spec)
        return out or [(0.0, self._layout)]

    def suggest(self, n: int = 5) -> List[Layout]:
        """The ``n`` best distinct layouts for the sliced data, best first."""
        return [lay for _, lay in self.suggest_ranked(n)]

    @property
    def confidence(self) -> str:
        """How much better the current layout scores than the runner-up: ``"high"``,
        ``"medium"`` or ``"low"`` — a small gap means alternatives are worth a look
        (see :meth:`suggest`)."""
        ranked = self.suggest_ranked(2)
        if len(ranked) < 2:
            return "high"
        gap = ranked[0][0] - ranked[1][0]
        if gap >= 0.75:
            return "high"
        if gap >= 0.25:
            return "medium"
        return "low"

    def alternatives(self, n: int = 5) -> List["View"]:
        """:meth:`suggest` as ready-made views."""
        return [self.use(lay) for lay in self.suggest(n)]

    def use(self, layout: Union[Layout, int]) -> "View":
        """Switch to a suggested layout (a :class:`Layout` or its index in :meth:`suggest`)."""
        if isinstance(layout, int):
            layout = self.suggest(layout + 1)[layout]
        spec = {"rows": list(layout.rows), "cols": list(layout.cols), "values": layout.values,
                "agg": COUNT if layout.values is None else layout.agg}
        return self._clone(layout=layout, spec=spec)

    # ------------------------------------------------------------------ clustering

    def _fresh_name(self, name: str) -> str:
        taken = set(self._source.columns) | set(self._derived)
        if name not in taken:
            return name
        i = 2
        while f"{name}{i}" in taken:
            i += 1
        return f"{name}{i}"

    def cluster(
        self,
        k: Optional[int] = None,
        *,
        on: Optional[Union[str, Sequence[str]]] = None,
        method: str = "kmeans",
        collapse: bool = False,
        name: str = "cluster",
        normalize: str = "row",
    ) -> "View":
        """Group rows into clusters and use the groups as a dimension.

        With ``on=None`` the *rows of the current pivot* are clustered by the shape of their
        column profile (ports that get denied alike, hosts with the same method mix ...).
        The cluster becomes the outer row level, or the only row level with
        ``collapse=True`` (reduce size). With ``on`` = numeric column(s), the *records* are
        clustered on those columns instead (in-memory data only).

        ``method`` is ``"kmeans"`` (``k`` defaults to a silhouette-chosen value),
        ``"dbscan"`` (density, automatic radius, outliers become ``noise``) or
        ``"hdbscan"`` (scikit-learn's HDBSCAN when installed, else DBSCAN).
        """
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}; use one of {METHODS}")
        name = self._fresh_name(name)
        if on is None:
            table = self.pivot()
            if table.shape[0] < 3:
                raise ValueError("need at least 3 pivot rows to cluster")
            labels, _ = cluster_rows(table, k, method=method, normalize=normalize, seed=self._options.seed)
            lookup = dict(zip(table.index, labels.tolist()))
            der = Derived(name, dims=tuple(self._layout.rows), lookup=lookup, categories=tuple(labels.cat.categories))
            # nested under the cluster, the existing rows are partitioned, not multiplied
            rows: List[Any] = [der.dim()] + ([] if collapse else list(self._layout.rows))
        else:
            if self._paged is not None:
                raise ValueError("record clustering needs the data in memory: use .sample() or .materialize() first")
            cols = [on] if isinstance(on, str) else list(on)
            lab = cluster_frame(self.data, cols, k, method=method, seed=self._options.seed, name=name)
            der = Derived(name, series=lab, categories=tuple(lab.cat.categories))
            rows = [der.dim()]
        base = self._clone(derived={**self._derived, name: der})
        return base.relayout(rows=rows, cols=list(self._layout.cols), **self._measure_spec())

    def cocluster(self, k: Optional[int] = None, *, method: str = "spectral", nest: bool = True, name: str = "block") -> "View":
        """Group rows *and* columns into matching blocks (spectral co-clustering or Markov
        clustering of the bipartite row-column graph).

        With ``nest=True`` a block level is added to both axes; with ``nest=False`` the axes
        are only reordered so the blocks show up along the diagonal of the heatmap.
        """
        table = self.pivot()
        rl, cl = _cocluster(table, k, method=method, seed=self._options.seed)
        rname, cname = self._fresh_name(f"{name}_r"), self._fresh_name(f"{name}_c")
        der_r = Derived(rname, dims=tuple(self._layout.rows), lookup=dict(zip(table.index, rl.tolist())), categories=tuple(rl.cat.categories))
        der_c = Derived(cname, dims=tuple(self._layout.cols), lookup=dict(zip(table.columns, cl.tolist())), categories=tuple(cl.cat.categories))
        if nest or len(self._layout.rows) > 1 or len(self._layout.cols) > 1:
            base = self._clone(derived={**self._derived, rname: der_r, cname: der_c})
            return base.relayout(rows=[der_r.dim()] + list(self._layout.rows), cols=[der_c.dim()] + list(self._layout.cols), **self._measure_spec())
        # reorder only: freeze each axis' single dimension in block order
        def ordered(labels: pd.Series) -> Tuple[Any, ...]:
            codes = labels.cat.codes.to_numpy()
            return tuple(labels.index[i] for i in sorted(range(len(labels)), key=lambda i: codes[i]))

        rows = [_replace(self._layout.rows[0], keep=ordered(rl), order="keep", top=None)]
        cols = [_replace(self._layout.cols[0], keep=ordered(cl), order="keep", top=None)]
        return self.relayout(rows=rows, cols=cols, **self._measure_spec())

    def stats(self, n: int = 7) -> pd.DataFrame:
        """The ``n`` costliest kinds of step logged so far (time, CPU, memory)."""
        return _log_stats(n)

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
        with log.step("fit", f"refit on {len(data):,} rows") as st:
            layout = fit_layout(data, options, rows=spec.get("rows"), cols=spec.get("cols"),
                                values=spec.get("values"), agg=spec.get("agg"))
            st.detail += f" -> {layout.describe()}"
        if self._paged is not None:
            layout = freeze(layout, data)
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
        if self._paged is not None:
            layout = freeze(layout, self.data)
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
        if self._paged is not None:
            layout = freeze(layout, data)
        spec = {"rows": [row], "cols": list(cols), "values": values, "agg": agg}
        parent = self if self._mode == PIVOT else self._parent
        return View(self._paged if self._paged is not None else self._source, layout, options=options, mode=HIST,
                    filters=self._filters, parent=parent, spec=spec, display=self._display, derived=self._derived, survey=self._survey)

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

    def title(self, *, slices: bool = True) -> str:
        """One line: mode, layout, row counts and (unless ``slices=False``) the active slices."""
        if self._paged is not None:
            total = len(self._paged)
            seen = self._cache.get("paged_rows")
            s = f"{self._mode} \u00b7 {self.describe()} \u00b7 "
            s += f"{seen:,} of \u2248{total:,} rows" if (self._filters and seen is not None) else f"\u2248{total:,} rows"
            s += f" (paged, {self._paged.survey.plan.n_pages} pages)"
            if self.approximate:
                s += " \u00b7 \u2248 measure from sample"
        else:
            n, total = len(self.data), len(self._source)
            s = f"{self._mode} \u00b7 {self.describe()} \u00b7 {n:,} rows"
            if n != total:
                s += f" of {total:,}"
        if self._filters and slices:
            s += " \u00b7 slices: " + " & ".join(f.label for f in self._filters)
        return s

    def render(self, *, width: Optional[int] = None, ascii_only: bool = False, compact: Optional[bool] = None,
               max_rows: Optional[int] = None, title: bool = True) -> str:
        """Text rendering of the current mode."""
        with log.step("render", "text"):
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

    def style(self, **display: Any) -> "View":
        """Display options for :meth:`html` / notebooks.

        ``theme``: ``"light"`` (default) or ``"graphite"`` (a dark, modern theme), both
        apply to pivots and histograms alike. Pivot: ``heat`` (``"table"`` | ``"column"``
        | ``"row"`` | ``"none"``), ``bars``, ``totals``, ``compact``, ``max_rows``,
        ``subtotals`` (a subtotal row after each outer row group), ``outline`` (nested
        rows as collapsible groups). Histogram: ``stacked``, ``density``, ``log_y``,
        ``width``, ``height``, ``show_values``.
        """
        return self._clone(display={**self._display, **display})

    @property
    def display(self) -> Dict[str, Any]:
        return dict(self._display)

    def html(self, title: bool = True, *, removable_slices: bool = False) -> str:
        """Rich HTML: a heatmap table for pivots, an SVG bar chart for histograms. With
        ``title``, the active slices are drawn as chips under the title line
        (``removable_slices`` adds a x to each, for the explorer, whose output pane turns a
        click on a chip into :meth:`unslice`)."""
        with log.step("render", "svg" if self.is_hist else "html"):
            return self._html(title, removable_slices=removable_slices)

    def _html(self, title: bool = True, *, removable_slices: bool = False) -> str:
        d = self._display
        t = self.title(slices=False) if title else None
        chips = self.slices if title else None
        if self.is_hist:
            table = self.bins()
            density = None
            rows = self._layout.rows
            if d.get("density", True) and len(rows) == 1 and rows[0].kind == "binned" and rows[0].column in self.data:
                edges = np.asarray(rows[0].edges or (0, 1), dtype=float)
                scale = _axis_scale(edges)  # the curve only makes sense on a linear or log axis
                if scale is not None:
                    density = B.kde(self.data[rows[0].column], log=(scale == "log"), seed=self._options.seed)
            return hist_svg(
                table, title=t, width=int(d.get("width", 760)), height=int(d.get("height", 340)), density=density,
                stacked=bool(d.get("stacked", False)), log_y=bool(d.get("log_y", False)), show_values=d.get("show_values"),
                theme=str(d.get("theme", "light")), chips=chips, removable_chips=removable_slices,
            )
        return pivot_html(
            self.pivot(), title=t, heat=str(d.get("heat", "table")), bars=bool(d.get("bars", False)),
            totals=bool(d.get("totals", False)), compact=bool(d.get("compact", False)), max_rows=d.get("max_rows"),
            subtotals=bool(d.get("subtotals", False)), outline=bool(d.get("outline", False)),
            agg=self._layout.agg if self._layout.values is not None else "count", theme=str(d.get("theme", "light")),
            chips=chips, removable_chips=removable_slices,
        )

    def svg(self) -> str:
        """The histogram of this view as SVG (toggles to histogram mode if needed)."""
        return self.as_hist().html()

    def distribution(self, column: str, **kw: Any) -> Optional[DistFit]:
        """Best-fitting probability distribution for a numeric column of the sliced data
        (BIC over normal/lognormal/exponential/gamma/uniform/poisson/geometric/bernoulli/
        discrete-uniform), or ``None`` if there isn't enough data. See
        :func:`pivot2hist.fit_distribution`."""
        from ._density import fit_distribution

        if column not in self.data.columns:
            raise KeyError(f"unknown column {column!r}")
        return fit_distribution(self.data[column], **kw)

    def modes(self, column: str, k: Optional[int] = None, **kw: Any) -> Optional[List[Dict[str, Any]]]:
        """How many peaks does this numeric column have, and where? A Gaussian mixture
        fit (component count chosen by BIC unless ``k`` is given), reported as a list of
        ``{"weight", "mean", "std"}`` dicts sorted by mean — e.g. two components at
        ~200 B and ~5 KB for a bimodal transfer-size column. ``None`` if there isn't
        enough data. See :func:`pivot2hist.modes`."""
        from ._mixture import choose_gmm_k, fit_gmm

        if column not in self.data.columns:
            raise KeyError(f"unknown column {column!r}")
        x = pd.to_numeric(self.data[column], errors="coerce").to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        if x.size < 8:
            return None
        kk = k if k is not None else choose_gmm_k(x.reshape(-1, 1), **kw)
        fit = fit_gmm(x.reshape(-1, 1), max(1, kk), **{k2: v for k2, v in kw.items() if k2 != "k_max"})
        return fit.components()

    def anomalies(self, n: int = 10) -> pd.DataFrame:
        """The ``n`` most surprising cells: biggest deviation from what independence of
        the row and column axes would predict (``row_total x col_total / grand_total``),
        as a Pearson-style residual. Needs a 2-D pivot (both axes present) and an additive
        measure (``sum``/``count`` — the only aggregations independence expectation means
        anything for). Positive ``residual`` = more than expected, negative = less.

        Pairs with ``v.style(heat="surprise")``, which colours every cell by the same
        residual instead of raw magnitude.
        """
        t = self.pivot()
        if t.shape[0] < 2 or t.shape[1] < 2:
            raise ValueError("anomalies() needs at least 2 rows and 2 columns")
        agg = self._layout.agg if self._layout.values is not None else "count"
        if agg not in ("sum", "count"):
            raise ValueError(f"anomalies() needs an additive measure (sum/count), not {agg!r}")
        values = t.to_numpy(dtype=float)
        expected = expected_independence(values)
        resid = surprise_residuals(values)
        row_label = ["/".join(map(str, t.index[i])) if t.index.nlevels > 1 else str(t.index[i]) for i in range(len(t.index))]
        col_label = ["/".join(map(str, t.columns[j])) if t.columns.nlevels > 1 else str(t.columns[j]) for j in range(len(t.columns))]
        order = np.argsort(-np.abs(resid), axis=None)[: max(1, n)]
        rows = []
        for flat in order:
            i, j = np.unravel_index(int(flat), resid.shape)
            v = float(values[i, j])
            if not np.isfinite(v):
                continue
            rows.append({
                "row": row_label[i], "col": col_label[j], "observed": v,
                "expected": round(float(expected[i, j]), 3), "residual": round(float(resid[i, j]), 3),
                "direction": "over" if resid[i, j] > 0 else "under",
            })
        return pd.DataFrame(rows, columns=["row", "col", "observed", "expected", "residual", "direction"])

    def llm_context(self, *, max_rows: int = 30, max_cols: int = 12, notes: bool = True) -> Dict[str, Any]:
        """This view as context for an LLM: ``{"description", "metadata", "table"}``.

        ``description`` is a short natural-language summary (what the table shows, its
        shape, active slices, whether it's approximate, the most surprising cell if one
        is cheap to compute); ``metadata`` is the same facts as plain JSON (measure,
        dims, shape, slices, per-column kind/semantic/cardinality for just the columns
        involved - not a full profile dump); ``table`` is the data itself as a GitHub-
        flavored markdown table, truncated (not sampled) to ``max_rows`` x ``max_cols``.
        Markdown, not HTML or a list of per-cell dicts, because it's both what a model
        has seen the most of and the cheapest in tokens. See
        :func:`pivot2hist.llm_context` for the plain function, and
        :func:`pivot2hist.agent.llm_context` for the JSON-source version.

        ``notes=False`` skips the bonus anomaly check (cheap, but not free, and not
        every layout supports it - a 1-D pivot or a non-additive measure just skip it
        either way).
        """
        from ._llm import llm_context as _llm_context

        return _llm_context(self, max_rows=max_rows, max_cols=max_cols, notes=notes)

    def insights(self, *, sensitivity: float = 0.5, max_findings: int = 15, max_pairs: int = 5) -> Dict[str, Any]:
        """A rich, local, non-LLM analysis of *this view's currently-filtered data* -
        every slice you've applied is already baked in, so re-slicing and calling this
        again is exactly "recalculate on what I'm looking at now".

        Every numeric/categorical/datetime column gets a summary (mean/median/std and
        best-fit distribution for numeric; top value and its share for categorical/
        boolean; range for datetime) in ``"columns"``. On top of that, ``"findings"`` is
        a ranked list of what's actually notable: skew, a Gaussian-mixture check for
        multiple distinct populations in one numeric column (the "mixle"-inspired bit -
        see :mod:`pivot2hist._mixture`), how concentrated a category is versus an even
        split, outlier share, near-constant columns, id-like cardinality, the most
        mutually-informative column pairs, and - when the current layout is a real 2-D
        pivot - the most surprising cells (see :meth:`anomalies`). Each finding carries
        a 0..1 ``significance``; ``sensitivity`` (0..1, default 0.5) sets how much of
        that ranked list actually surfaces - higher shows more (including weaker
        findings), lower shows only the strongest. It's not called "temperature": this
        is a deterministic computation over the data, not sampling from a model, and
        that word would suggest a kind of randomness this doesn't have.

        Cheap enough to call after every slice: distribution/mixture fits sample down to
        20k rows, and nothing here is more expensive than the pivot itself. See
        :func:`pivot2hist.insights` for the plain function and
        :func:`pivot2hist.agent.insights` for the JSON-source version.
        """
        from ._insights import insights as _insights

        return _insights(self, sensitivity=sensitivity, max_findings=max_findings, max_pairs=max_pairs)

    def _repr_html_(self) -> str:
        return self.html()

    # ------------------------------------------------------------------ comparing

    def compare(self, *args: Any, metric: Optional[str] = None, names: Optional[Sequence[str]] = None, **kwargs: Any) -> Any:
        """Two sides of this data on one shared layout, cell by cell - a :class:`Comparison`.

        Forms::

            v.compare(action="deny")                     # deny vs the rest (any slice() form)
            v.compare("bytes > 1000")                    # a query vs its complement
            v.compare("action", "deny")                  # positional column/value, vs the rest
            v.compare("action", "deny", "allow")         # deny vs allow
            v.compare({"timestamp": "2026-03-02"},
                      {"timestamp": "2026-03-01"})       # any two side specs (mappings or queries)
            today.compare(yesterday)                     # two Views (yesterday laid out like today)

        The auto-fit is frozen for the comparison: both sides get *this* view's dimensions,
        bin edges and kept top-N labels, so every cell means the same thing on both sides
        (a per-side refit would pick whatever suits each side and make them incomparable).
        If the split column is itself on an axis it is taken off it - splitting on
        ``action`` when ``action`` is the column axis would leave nothing to compare - and
        that axis is refilled by one fit on this view's data.

        ``metric`` is what the comparison table shows: ``"lift"`` (default for a split of
        an additive measure - A's share of its own total over B's share, so a 5%-of-traffic
        slice compares fairly against the other 95%), ``"delta"`` (default otherwise: A -
        B), ``"ratio"``, ``"pct_change"``, ``"share_delta"`` (percentage points), or the
        raw ``"a"``/``"b"``/``"share_a"``/``"share_b"``. ``names`` labels the sides. The
        result renders as a diverging heatmap (blue = more on side A, red = less), lists
        the biggest movers via ``.top()``, slices both sides at once, and ``.toggle()``\\ s
        to paired histograms on shared bins.
        """
        from ._compare import compare as _compare

        return _compare(self, *args, metric=metric, names=names, **kwargs)

    def facet(self, column: str, n: int = 6, *, levels: Optional[Sequence[Any]] = None) -> Any:
        """Small multiples: this view once per value of ``column`` (the ``n`` most frequent,
        or the given ``levels``), all on the same layout and one shared colour scale, as
        a :class:`Facets` grid. The faceted column is taken off the axes (if it was on one)
        and every panel shares the same row/column labels, so panel 2's top-left cell is
        the same (row, column) as panel 1's. ``facets.compare("deny", "allow")`` turns two
        panels into a :class:`Comparison`; slicing and ``toggle()`` apply to every panel.
        """
        from ._compare import facet as _facet

        return _facet(self, column, n, levels=levels)

    # ------------------------------------------------------------------ interrogating a cell

    def cell(self, *args: Any, **labels: Any) -> "View":
        """This view narrowed to one cell (or one row / one column) of its table, named by
        *label* - what the table shows - rather than by raw value::

            v.cell("22", "deny")                 # row label, column label (tuples for nested levels)
            v.cell(dst_port=22, action="deny")   # column=label; a label of any axis level
            v.cell(bytes="[1K, 2K)")             # a bin of a histogram / binned axis
            v.cell(timestamp="13:00")            # a time bucket, as labelled
            v.cell(dst_port="(other)")           # the folded top-N remainder; "(null)" for nulls
            v.cell(src_ip="10.0.1.0/24")         # a semantic roll-up level

        Labels are matched by materializing each dimension exactly as the pivot did, so
        every kind of level works the same way. ``None`` for an axis (or a level) leaves
        it open, giving a whole row or column. A ``column=value`` for a column that is
        *not* on an axis is an ordinary :meth:`slice`. The layout is kept (no refit): the
        result's table is that cell; ``.data`` is the rows behind it (see :meth:`rows`).
        """
        from ._explain import cell_view

        return cell_view(self, args, labels)[0]

    def rows(self, *args: Any, n: Optional[int] = None, **labels: Any) -> pd.DataFrame:
        """The raw rows behind a cell (same forms as :meth:`cell`), with every active
        slice applied; ``n`` caps them. A paged source is scanned page by page for them.
        """
        from ._explain import rows as _rows

        return _rows(self, args, n, labels)

    def explain(self, *args: Any, n_rows: int = 10, k: int = 5, **labels: Any) -> Any:
        """Why does this cell look the way it does? (Same forms as :meth:`cell`.)

        Returns an :class:`Explanation` - a JSON-safe dict that reads well in a notebook -
        with the cell's ``observed`` value; for an additive measure on a 2-D table its
        ``expected`` value under independence of the axes, the ``ratio_to_expected`` and
        ``direction`` (the same residual :meth:`anomalies` ranks by); its shares of its row,
        column and the whole table; its ``rank`` among all cells; how many rows it holds;
        ``distinguishing`` - the ``k`` columns that most set those rows apart from the rest
        of the data in view (a label over-represented here, as its share here vs elsewhere;
        a numeric whose median differs, as a ratio); the first ``n_rows`` rows; and a
        one-paragraph ``text`` saying all of that. Nothing here calls a model.
        """
        from ._explain import explain as _explain

        return _explain(self, args, labels, n_rows=n_rows, k=k)

    # ------------------------------------------------------------------ export

    def prompt(self, question: Optional[str] = None, **kw: Any) -> Any:
        """Everything on screen as one self-contained prompt for any LLM - a
        :class:`Prompt` (a ``str`` that previews as markdown, with ``.tokens`` and
        ``.save(path)``)::

            p = v.prompt("Which destination ports deserve a firewall rule, and why?")
            print(p)                        # or just `p` in a notebook - paste it into any chat
            p.save("prompt.md")

        The prompt states that every fact was computed locally and that the model should
        reason only from them, then gives (each switchable): the **dataset** (rows in view,
        every column with kind, semantic type, cardinality, nulls, examples; ``profile=``),
        the **current view** (its description, active slices and the table as markdown,
        ``table=``, ``max_rows=``/``max_cols=``), **computed findings** (the most surprising
        cells, ``anomalies=``/``n_anomalies=``; the ranked :meth:`insights`, ``insights=True``
        or a report you already have, at ``sensitivity=``), a **comparison** (``compare=``:
        a :class:`Comparison`, or a split such as ``{"action": "deny"}`` or a query string),
        a **cell in focus** (``explain=``: an :class:`Explanation`, ``{column: label}``, or a
        ``(row_label, col_label)`` tuple), and **your task** - ``question``, or a default
        asking for what stands out, three next steps and data-quality flags. No model is
        called here; this is the packaging.
        """
        from ._prompt import build_prompt

        return build_prompt(self, question, **kw)

    def explore(self, **kw: Any) -> Any:
        """Open the interactive Jupyter explorer (needs ``ipywidgets``)."""
        from .ui import explore

        return explore(self, **kw)

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
        d = {
            "mode": self._mode,
            "layout": self._layout.to_dict(),
            "slices": self.slices,
            "rows": int(len(self.data)),
            "source_rows": int(len(self._paged) if self._paged is not None else len(self._source)),
            "shape": list(self.table().shape),
            "table": records,
        }
        if self._paged is not None:
            d["paged"] = True
            d["source_rows_exact"] = False
            d["pages"] = self._paged.survey.plan.n_pages
        if self.approximate:
            d["approximate"] = True
        return d

    def to_json(self, **kw: Any) -> str:
        kw.setdefault("indent", 2)
        return json.dumps(self.to_dict(), **kw)

    def to_csv(self, path: Optional[str] = None, **kw: Any) -> Optional[str]:
        return self.table().to_csv(path, **kw)


__all__ = ["View", "Filter", "Derived", "make_filter", "PIVOT", "HIST"]
