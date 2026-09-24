"""Interactive Jupyter explorer: menus to alter the layout, slice, best-fit, reduce and
cluster, rendered with the dependency-free HTML/SVG graphics.

::

    import pivot2hist as p2h
    p2h.explore(df)            # or view.explore()

Needs ``ipywidgets`` (``pip install "pivot2hist[jupyter]"``). The explorer keeps a small
*recipe* (layout spec, options, slices, mode, histogram/cluster/style settings) and
rebuilds the :class:`~pivot2hist.View` from it on every change, so the "Code" tab can
always show the equivalent Python.
"""
from __future__ import annotations

import html as _html
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import ipywidgets as W
except ImportError as e:  # pragma: no cover
    raise ImportError("pivot2hist.ui needs ipywidgets: pip install 'pivot2hist[jupyter]'") from e

from ._binning import RULES, human
from ._fit import AGGS, COUNT, FitOptions, Layout
from ._io import load
from ._profile import BOOLEAN, CATEGORICAL, DATETIME, NUMERIC, profile
from ._view import HIST, PIVOT, View

_NUM_STEPS = 24


def _kw(d: Dict[str, Any]) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in d.items())


class Explorer:
    """The widget app. ``explorer.view`` is the current :class:`View`; ``explorer.code()``
    is the Python that reproduces it."""

    def __init__(self, view: View, *, width: int = 760, height: int = 340, max_slicers: int = 8):
        self._source = view.source
        self._profile = profile(self._source)
        self.options: FitOptions = view.options
        self.spec: Dict[str, Any] = {k: view._spec.get(k) for k in ("rows", "cols", "values", "agg")}
        self.slices: List[Tuple[str, Any]] = []  # ("slice", {col: spec}) | ("query", expr) | ("top", (col, n))
        self.mode = view.mode
        self.hist: Dict[str, Any] = {"on": None, "by": None, "bins": None}
        self.reduce: Dict[str, Any] = {"sample": None, "cluster": None, "cluster_on": None, "collapse": False}
        self.display: Dict[str, Any] = {**view.display, "width": width, "height": height}
        self.refit_sliced = False
        self.history: List[Dict[str, Any]] = []
        self.view: View = view
        self._syncing = False
        self._root_cache: Dict[Tuple, View] = {}
        self._max_slicers = max_slicers
        self._build_widgets()
        for f in view.filters:  # carry existing slices along as an opaque query-ish step
            self.slices.append(("filter", f))
        self._rebuild()

    # ------------------------------------------------------------------ widgets

    def _build_widgets(self) -> None:
        prof = self._profile
        columns = list(self._source.columns)
        numeric = [c.name for c in prof if c.kind == NUMERIC]
        labels = [c.name for c in prof if c.kind in (CATEGORICAL, BOOLEAN)]
        st = {"description_width": "80px"}

        # -- top bar
        self.w_mode = W.ToggleButtons(options=[("Pivot", PIVOT), ("Histogram", HIST)], value=self.mode, tooltips=["table", "bars"])
        self.w_best = W.Button(description="Best fit", icon="magic", tooltip="auto-fit rows/cols/measure on the sliced data")
        self.w_suggest_btn = W.Button(description="Suggest", icon="lightbulb-o", tooltip="rank alternative layouts")
        self.w_suggest = W.Dropdown(options=[("alternatives…", None)], value=None, layout=W.Layout(width="360px"))
        self.w_undo = W.Button(description="Undo", icon="undo")
        self.w_reset = W.Button(description="Reset", icon="refresh")

        # -- layout tab
        self.w_rows = W.SelectMultiple(options=columns, rows=min(8, len(columns)), description="Rows", style=st)
        self.w_cols = W.SelectMultiple(options=columns, rows=min(8, len(columns)), description="Columns", style=st)
        self.w_values = W.Dropdown(options=[("count", COUNT)] + [(c, c) for c in numeric], description="Values", style=st)
        self.w_agg = W.Dropdown(options=[a for a in AGGS if a != COUNT], value="sum", description="Agg", style=st)
        self.w_layers = W.IntSlider(value=self.options.layers, min=1, max=3, description="Layers", style=st)
        self.w_max_rows = W.IntSlider(value=self.options.max_rows, min=3, max=120, description="Max rows", style=st)
        self.w_max_cols = W.IntSlider(value=self.options.max_cols, min=1, max=40, description="Max cols", style=st)
        self.w_bins_rule = W.Dropdown(options=list(RULES), value=str(self.options.bins) if self.options.bins in RULES else "auto", description="Bin rule", style=st)
        self.w_scale = W.Dropdown(options=["auto", "linear", "log"], value=self.options.scale, description="Scale", style=st)
        self.w_order = W.Dropdown(options=["auto", "natural", "frequency"], value=self.options.order, description="Order", style=st)
        self.w_variants = W.Checkbox(value=self.options.variants, description="drill variants (/24, hour of day …)")
        layout_tab = W.VBox([
            W.HTML("<i>Leave rows/columns empty to let the fit choose.</i>"),
            W.HBox([self.w_rows, self.w_cols, W.VBox([self.w_values, self.w_agg, self.w_layers])]),
            W.HBox([self.w_max_rows, self.w_max_cols]),
            W.HBox([self.w_bins_rule, self.w_scale, self.w_order]),
            self.w_variants,
        ])

        # -- histogram tab
        self.w_on = W.Dropdown(options=[("(rows of the pivot)", None)] + [(c, c) for c in columns], value=None, description="On", style=st)
        self.w_by = W.Dropdown(options=[("(none)", None)] + [(c, c) for c in columns], value=None, description="By", style=st)
        self.w_nbins = W.IntSlider(value=0, min=0, max=60, description="Bins (0=auto)", style=st)
        self.w_stacked = W.Checkbox(value=False, description="stacked")
        self.w_density = W.Checkbox(value=True, description="density curve")
        self.w_logy = W.Checkbox(value=False, description="log y")
        hist_tab = W.VBox([W.HBox([self.w_on, self.w_by]), self.w_nbins, W.HBox([self.w_stacked, self.w_density, self.w_logy])])

        # -- slicers tab
        self.w_slicers: Dict[str, Any] = {}
        boxes: List[Any] = []
        ranked = sorted([c for c in prof if c.kind in (CATEGORICAL, BOOLEAN)], key=lambda c: (c.position,))
        for cp in ranked[: self._max_slicers]:
            vc = self._source[cp.name].value_counts(dropna=False).head(30)
            opts = [(f"{'(null)' if (isinstance(k, float) and np.isnan(k)) else k} ({n:,})", None if (isinstance(k, float) and np.isnan(k)) else k) for k, n in vc.items()]
            w = W.SelectMultiple(options=opts, rows=min(6, len(opts)), description=cp.name[:12], style=st, layout=W.Layout(width="260px"))
            self.w_slicers[cp.name] = ("in", w)
            boxes.append(w)
        for cp in [c for c in prof if c.kind == NUMERIC][:4]:
            col = pd.to_numeric(self._source[cp.name], errors="coerce").to_numpy(dtype=float)
            col = col[np.isfinite(col)]
            if col.size == 0:
                continue
            qs = np.unique(np.percentile(col, np.linspace(0, 100, _NUM_STEPS + 1)))
            if qs.size < 2:
                continue
            opts = [(human(q), float(q)) for q in qs]
            w = W.SelectionRangeSlider(options=opts, index=(0, len(opts) - 1), description=cp.name[:12], style=st, layout=W.Layout(width="360px"), continuous_update=False)
            self.w_slicers[cp.name] = ("range", w)
            boxes.append(w)
        for cp in [c for c in prof if c.kind == DATETIME][:2]:
            days = self._source[cp.name].dropna().dt.floor("D").drop_duplicates().sort_values()
            if len(days) < 2:
                continue
            opts = [(d.strftime("%Y-%m-%d"), d) for d in days]
            w = W.SelectionRangeSlider(options=opts, index=(0, len(opts) - 1), description=cp.name[:12], style=st, layout=W.Layout(width="360px"), continuous_update=False)
            self.w_slicers[cp.name] = ("days", w)
            boxes.append(w)
        self.w_query = W.Text(placeholder="pandas query, e.g. bytes > 1000 and action == 'deny'  (Enter applies)", description="Query",
                              style=st, layout=W.Layout(width="520px"), continuous_update=False)
        self.w_top_col = W.Dropdown(options=[("(no top-N)", None)] + [(c, c) for c in labels], value=None, description="Top-N of", style=st)
        self.w_top_n = W.IntSlider(value=10, min=2, max=100, description="N", style=st)
        self.w_clear = W.Button(description="Clear slices", icon="eraser")
        slice_tab = W.VBox([W.HBox(boxes[:3]), W.HBox(boxes[3:6]), W.HBox(boxes[6:]), W.HBox([self.w_query]), W.HBox([self.w_top_col, self.w_top_n, self.w_clear])])

        # -- reduce & cluster tab
        self.w_sample = W.Dropdown(options=[("all rows", None), ("100k rows", 100_000), ("10k rows", 10_000), ("1k rows", 1_000)], value=None, description="Sample", style=st)
        self.w_cluster = W.Dropdown(options=[("off", None), ("auto k", "auto")] + [(str(k), k) for k in range(2, 9)], value=None, description="Cluster", style=st)
        self.w_cluster_on = W.Dropdown(options=[("pivot rows", None)] + [(c, c) for c in numeric], value=None, description="on", style=st)
        self.w_collapse = W.Checkbox(value=False, description="collapse rows into clusters")
        self.w_dim = W.Dropdown(options=[], description="Dimension", style=st)
        self.w_coarser = W.Button(description="Coarser", icon="compress", tooltip="/24 -> /16, hour -> day, fewer bins")
        self.w_finer = W.Button(description="Finer", icon="expand")
        reduce_tab = W.VBox([
            W.HTML("<i>Reduce size: sample rows, shrink the box (Layout tab), roll dimensions up, or cluster similar rows.</i>"),
            self.w_sample,
            W.HBox([self.w_cluster, self.w_cluster_on, self.w_collapse]),
            W.HBox([self.w_dim, self.w_coarser, self.w_finer]),
        ])

        # -- style tab
        self.w_heat = W.Dropdown(options=["table", "column", "row", "none"], value=str(self.display.get("heat", "table")), description="Heat", style=st)
        self.w_totals = W.Checkbox(value=bool(self.display.get("totals", False)), description="totals")
        self.w_bars = W.Checkbox(value=bool(self.display.get("bars", False)), description="in-cell bars")
        self.w_compact = W.Checkbox(value=bool(self.display.get("compact", False)), description="compact numbers")
        self.w_width = W.IntSlider(value=int(self.display.get("width", 760)), min=300, max=1600, step=20, description="Width", style=st)
        self.w_height = W.IntSlider(value=int(self.display.get("height", 340)), min=160, max=900, step=20, description="Height", style=st)
        style_tab = W.VBox([W.HBox([self.w_heat, self.w_totals, self.w_bars, self.w_compact]), W.HBox([self.w_width, self.w_height])])

        # -- code tab, output
        self.w_code = W.HTML()
        self.w_out = W.HTML()
        self.w_status = W.HTML()
        self.w_profile = W.HTML("<pre style='font-size:11px'>" + _html.escape(self._profile.summary().to_string(index=False)) + "</pre>")

        tabs = W.Tab(children=[layout_tab, hist_tab, slice_tab, reduce_tab, style_tab, self.w_code, self.w_profile])
        for i, t in enumerate(["Layout", "Histogram", "Slicers", "Reduce & cluster", "Style", "Code", "Profile"]):
            tabs.set_title(i, t)
        top = W.HBox([self.w_mode, self.w_best, self.w_suggest_btn, self.w_suggest, self.w_undo, self.w_reset])
        self.box = W.VBox([top, tabs, self.w_status, self.w_out])

        # -- wiring
        self.w_mode.observe(self._on_mode, names="value")
        self.w_best.on_click(lambda _: self._act(self._best_fit))
        self.w_suggest_btn.on_click(lambda _: self._act(self._fill_suggestions))
        self.w_suggest.observe(self._on_suggest, names="value")
        self.w_undo.on_click(lambda _: self._act(self._undo))
        self.w_reset.on_click(lambda _: self._act(self._reset))
        for w in (self.w_rows, self.w_cols, self.w_values, self.w_agg):
            w.observe(self._on_layout, names="value")
        for w in (self.w_layers, self.w_max_rows, self.w_max_cols, self.w_bins_rule, self.w_scale, self.w_order, self.w_variants):
            w.observe(self._on_options, names="value")
        for w in (self.w_on, self.w_by, self.w_nbins):
            w.observe(self._on_hist, names="value")
        for w in (self.w_stacked, self.w_density, self.w_logy, self.w_heat, self.w_totals, self.w_bars, self.w_compact, self.w_width, self.w_height):
            w.observe(self._on_style, names="value")
        for _, (kind, w) in self.w_slicers.items():
            w.observe(self._on_slicers, names="value" if kind != "range" and kind != "days" else "index")
        self.w_query.observe(self._on_query, names="value")  # continuous_update=False: fires on Enter/blur
        self.w_top_col.observe(self._on_slicers, names="value")
        self.w_top_n.observe(self._on_slicers, names="value")
        self.w_clear.on_click(lambda _: self._act(self._clear_slices))
        for w in (self.w_sample, self.w_cluster, self.w_cluster_on, self.w_collapse):
            w.observe(self._on_reduce, names="value")
        self.w_coarser.on_click(lambda _: self._act(lambda: self._step(-1)))
        self.w_finer.on_click(lambda _: self._act(lambda: self._step(+1)))

    # ------------------------------------------------------------------ recipe -> view

    def _snapshot(self) -> Dict[str, Any]:
        return {
            "spec": dict(self.spec), "slices": list(self.slices), "mode": self.mode, "hist": dict(self.hist),
            "reduce": dict(self.reduce), "display": dict(self.display), "options": self.options, "refit": self.refit_sliced,
        }

    def _restore(self, snap: Dict[str, Any]) -> None:
        self.spec, self.slices, self.mode = dict(snap["spec"]), list(snap["slices"]), snap["mode"]
        self.hist, self.reduce, self.display = dict(snap["hist"]), dict(snap["reduce"]), dict(snap["display"])
        self.options, self.refit_sliced = snap["options"], snap["refit"]
        self._sync_recipe_widgets()

    def _sync_recipe_widgets(self) -> None:
        """Push recipe state into the widgets (after undo/reset) without firing handlers."""
        self._syncing = True
        try:
            o, h, r, d = self.options, self.hist, self.reduce, self.display
            self.w_layers.value, self.w_max_rows.value, self.w_max_cols.value = o.layers, o.max_rows, o.max_cols
            self.w_bins_rule.value = str(o.bins) if o.bins in RULES else "auto"
            self.w_scale.value, self.w_order.value, self.w_variants.value = o.scale, o.order, o.variants
            self.w_on.value, self.w_by.value, self.w_nbins.value = h.get("on"), h.get("by"), int(h.get("bins") or 0)
            self.w_sample.value, self.w_cluster.value = r.get("sample"), r.get("cluster")
            self.w_cluster_on.value, self.w_collapse.value = r.get("cluster_on"), bool(r.get("collapse"))
            self.w_stacked.value, self.w_density.value = bool(d.get("stacked")), bool(d.get("density", True))
            self.w_logy.value, self.w_heat.value = bool(d.get("log_y")), str(d.get("heat", "table"))
            self.w_totals.value, self.w_bars.value, self.w_compact.value = bool(d.get("totals")), bool(d.get("bars")), bool(d.get("compact"))
            self.w_width.value, self.w_height.value = int(d.get("width", 760)), int(d.get("height", 340))
            rows, cols = self.spec.get("rows"), self.spec.get("cols")
            def names(specs: Any) -> Tuple[str, ...]:
                out = []
                for x in specs or ():
                    c = x if isinstance(x, str) else (x.get("column") if isinstance(x, dict) else getattr(x, "column", None))
                    if c in self.w_rows.options:
                        out.append(c)
                return tuple(out)

            self.w_rows.value, self.w_cols.value = names(rows), names(cols)
            self.w_mode.value = self.mode
        finally:
            self._syncing = False

    def _act(self, fn) -> None:
        """Run a state change with undo support and error reporting."""
        if self._syncing:
            return
        self.history.append(self._snapshot())
        try:
            fn()
            self._rebuild()
        except Exception as e:  # noqa: BLE001 - surface any failure in the status line
            self._restore(self.history.pop())
            self.w_status.value = f"<span style='color:#b00020'>{_html.escape(type(e).__name__)}: {_html.escape(str(e))}</span>"

    def _root(self) -> View:
        src = self._source
        n = self.reduce["sample"]
        key = (id(src), n, repr(self.spec), repr(self.options))
        if key not in self._root_cache:
            if n and n < len(src):
                src = src.sample(int(n), random_state=self.options.seed).sort_index()
            self._root_cache[key] = View.fit(src, options=self.options, **self.spec)
        return self._root_cache[key]

    def build(self) -> View:
        """The :class:`View` described by the current recipe."""
        v = self._root()
        for kind, payload in self.slices:
            if kind == "slice":
                v = v.slice(refit=False, **payload)
            elif kind == "query":
                v = v.slice(payload, refit=False)
            elif kind == "top":
                v = v.top(*payload)
            elif kind == "filter":
                v = v._clone(filters=v.filters + (payload,))
        if self.refit_sliced or (self.slices and v._layout_stale()):
            v = v.refit()
        k = self.reduce["cluster"]
        if k is not None:
            v = v.cluster(None if k == "auto" else int(k), on=self.reduce["cluster_on"], collapse=bool(self.reduce["collapse"]))
        if self.mode == HIST:
            h = self.hist
            if h["on"] or h["by"] or h["bins"]:
                v = v.histogram(h["on"], h["by"], bins=h["bins"] or None)
            else:
                v = v.toggle()
        return v.style(**self.display)

    def code(self) -> str:
        """Python that reproduces the current view."""
        spec = {k: v for k, v in self.spec.items() if v is not None}
        opts = {}
        d = FitOptions()
        for f in ("max_rows", "max_cols", "layers", "bins", "scale", "order", "variants"):
            if getattr(self.options, f) != getattr(d, f):
                opts[f] = getattr(self.options, f)
        lines = ["import pivot2hist as p2h", ""]
        src = "df"
        if self.reduce["sample"]:
            src = f"df.sample({self.reduce['sample']}, random_state=0)"
        args = ", ".join(x for x in (src, _kw(spec), _kw(opts)) if x)
        lines.append(f"v = p2h.fit({args})")
        for kind, payload in self.slices:
            if kind == "slice":
                lines.append(f"v = v.slice({_kw(payload)})")
            elif kind == "query":
                lines.append(f"v = v.slice({payload!r})")
            elif kind == "top":
                lines.append(f"v = v.top({payload[0]!r}, {payload[1]})")
            elif kind == "filter":
                lines.append(f"# slice: {payload.label}")
        if self.refit_sliced:
            lines.append("v = v.refit()")
        if self.reduce["cluster"] is not None:
            k = self.reduce["cluster"]
            kw = {"on": self.reduce["cluster_on"]} if self.reduce["cluster_on"] else {}
            if self.reduce["collapse"]:
                kw["collapse"] = True
            lines.append(f"v = v.cluster({'' if k == 'auto' else k}{', ' if kw and k != 'auto' else ''}{_kw(kw)})")
        if self.mode == HIST:
            h = {k: v for k, v in self.hist.items() if v}
            lines.append(f"v = v.histogram({_kw(h)})" if h else "v = v.toggle()")
        defaults = {"density": True, "heat": "table"}
        disp = {k: v for k, v in self.display.items()
                if k not in ("width", "height") and v not in (None, False) and defaults.get(k) != v}
        if disp:
            lines.append(f"v = v.style({_kw(disp)})")
        lines.append("v")
        return "\n".join(lines)

    def _rebuild(self) -> None:
        self.view = self.build()
        self.w_out.value = self.view.html()
        self.w_status.value = f"<span style='color:#555;font-size:12px'>{_html.escape(self.view.title())}</span>"
        self.w_code.value = "<pre style='font-size:12px'>" + _html.escape(self.code()) + "</pre>"
        self._sync_widgets()

    def _sync_widgets(self) -> None:
        self._syncing = True
        try:
            lay = self.view.layout
            self.w_mode.value = self.view.mode
            dims = [d.column for d in lay.dims]
            self.w_dim.options = dims
            if dims and self.w_dim.value not in dims:
                self.w_dim.value = dims[0]
            if self.spec.get("rows") is None:
                self.w_rows.value = tuple(d.column for d in lay.rows if d.column in self.w_rows.options)
            if self.spec.get("cols") is None:
                self.w_cols.value = tuple(d.column for d in lay.cols if d.column in self.w_cols.options)
            if self.spec.get("values") is None:
                self.w_values.value = lay.values if lay.values in [o[1] for o in self.w_values.options] else COUNT
                if lay.values is not None:
                    self.w_agg.value = lay.agg
        finally:
            self._syncing = False

    # ------------------------------------------------------------------ handlers

    def _on_mode(self, change: Any) -> None:
        def go() -> None:
            self.mode = change["new"]
        self._act(go)

    def _best_fit(self) -> None:
        self.spec = {"rows": None, "cols": None, "values": None, "agg": None}
        self.refit_sliced = True

    def _fill_suggestions(self) -> None:
        sugg = self.view.suggest(8)
        self._suggestions = sugg
        self._syncing = True
        try:
            self.w_suggest.options = [("alternatives…", None)] + [(lay.describe(), i) for i, lay in enumerate(sugg)]
            self.w_suggest.value = None
        finally:
            self._syncing = False

    def _on_suggest(self, change: Any) -> None:
        if change["new"] is None:
            return
        lay: Layout = self._suggestions[change["new"]]

        def go() -> None:
            self.spec = {"rows": list(lay.rows), "cols": list(lay.cols), "values": lay.values, "agg": COUNT if lay.values is None else lay.agg}
            self.refit_sliced = False
        self._act(go)

    def _on_layout(self, change: Any) -> None:
        def go() -> None:
            rows = list(self.w_rows.value)
            cols = [c for c in self.w_cols.value if c not in rows]
            values = self.w_values.value
            self.spec = {
                "rows": rows or None,
                "cols": cols or None,
                "values": None if values == COUNT else values,
                "agg": COUNT if values == COUNT else self.w_agg.value,
            }
            self.refit_sliced = False
        self._act(go)

    def _on_options(self, change: Any) -> None:
        def go() -> None:
            self.options = self.options.replace(
                layers=self.w_layers.value, max_rows=self.w_max_rows.value, max_cols=self.w_max_cols.value,
                bins=self.w_bins_rule.value, scale=self.w_scale.value, order=self.w_order.value, variants=self.w_variants.value,
            )
        self._act(go)

    def _on_hist(self, change: Any) -> None:
        def go() -> None:
            self.hist = {"on": self.w_on.value, "by": self.w_by.value, "bins": self.w_nbins.value or None}
            self.mode = HIST
        self._act(go)

    def _on_style(self, change: Any) -> None:
        def go() -> None:
            self.display.update(
                stacked=self.w_stacked.value, density=self.w_density.value, log_y=self.w_logy.value,
                heat=self.w_heat.value, totals=self.w_totals.value, bars=self.w_bars.value, compact=self.w_compact.value,
                width=self.w_width.value, height=self.w_height.value,
            )
        self._act(go)

    def _slices_from_widgets(self) -> List[Tuple[str, Any]]:
        out: List[Tuple[str, Any]] = [s for s in self.slices if s[0] in ("query", "filter")]
        for col, (kind, w) in self.w_slicers.items():
            if kind == "in":
                if w.value:
                    out.append(("slice", {col: list(w.value)}))
            else:
                lo, hi = w.index
                if lo != 0 or hi != len(w.options) - 1:
                    a, b = w.options[lo][1], w.options[hi][1]
                    if kind == "days":
                        b = b + pd.Timedelta(days=1)
                        out.append(("slice", {col: (str(a.date()), str(b.date()))}))
                    else:
                        out.append(("slice", {col: (a, b + 1e-9 if hi == len(w.options) - 1 else b)}))
        if self.w_top_col.value:
            out.append(("top", (self.w_top_col.value, int(self.w_top_n.value))))
        return out

    def _on_slicers(self, change: Any) -> None:
        def go() -> None:
            self.slices = self._slices_from_widgets()
        self._act(go)

    def _on_query(self, change: Any) -> None:
        self._act(self._apply_query)

    def _apply_query(self) -> None:
        expr = self.w_query.value.strip()
        self.slices = [s for s in self.slices if s[0] != "query"]
        if expr:
            self.slices.append(("query", expr))

    def _clear_slices(self) -> None:
        self._syncing = True
        try:
            for col, (kind, w) in self.w_slicers.items():
                if kind == "in":
                    w.value = ()
                else:
                    w.index = (0, len(w.options) - 1)
            self.w_query.value = ""
            self.w_top_col.value = None
        finally:
            self._syncing = False
        self.slices = []

    def _on_reduce(self, change: Any) -> None:
        def go() -> None:
            self.reduce = {
                "sample": self.w_sample.value, "cluster": self.w_cluster.value,
                "cluster_on": self.w_cluster_on.value, "collapse": self.w_collapse.value,
            }
        self._act(go)

    def _step(self, direction: int) -> None:
        col = self.w_dim.value
        if not col:
            raise ValueError("pick a dimension first")
        v = self.view.coarser(col) if direction < 0 else self.view.finer(col)
        lay = v.layout
        self.spec = {"rows": list(lay.rows), "cols": list(lay.cols), "values": lay.values, "agg": COUNT if lay.values is None else lay.agg}
        self.refit_sliced = False

    def _undo(self) -> None:
        if len(self.history) >= 2:
            self.history.pop()  # the snapshot _act just pushed
            self._restore(self.history.pop())

    def _reset(self) -> None:
        first = self.history[0] if self.history else self._snapshot()
        self._restore(first)
        self._clear_slices()

    # ------------------------------------------------------------------ display

    def _repr_mimebundle_(self, **kw: Any) -> Any:
        return self.box._repr_mimebundle_(**kw)

    def _ipython_display_(self) -> None:  # pragma: no cover - notebook only
        from IPython.display import display

        display(self.box)


def explore(data: Any, **fit_kwargs: Any) -> Explorer:
    """Open the interactive explorer on a frame/path/records or an existing :class:`View`."""
    ui_kw = {k: fit_kwargs.pop(k) for k in ("width", "height", "max_slicers") if k in fit_kwargs}
    view = data if isinstance(data, View) else View.fit(load(data), **fit_kwargs)
    return Explorer(view, **ui_kw)


__all__ = ["Explorer", "explore"]
