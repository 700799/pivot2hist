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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import ipywidgets as W
except ImportError as e:  # pragma: no cover
    raise ImportError("pivot2hist.ui needs ipywidgets: pip install 'pivot2hist[jupyter]'") from e

from ._binning import RULES, human
from ._chains import sequences as _sequences
from ._cluster import COMETHODS, METHODS
from ._fit import AGGS, COUNT, FitOptions, Layout
from ._log import log
from ._profile import BOOLEAN, CATEGORICAL, DATETIME, NUMERIC, profile
from ._view import HIST, PIVOT, View
from .ui_fields import make_field_list

CHAINS = "chains"

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
        self.reduce: Dict[str, Any] = {"sample": None, "cluster": None, "cluster_on": None, "collapse": False,
                                       "method": "kmeans", "cocluster": None}
        self.chain: Dict[str, Any] = {"state": None, "by": None, "time": None, "normalize": False}
        self.display: Dict[str, Any] = {**view.display, "width": width, "height": height}
        self.field_slicers: List[str] = []
        self.refit_sliced = False
        self.history: List[Dict[str, Any]] = []
        self.view: View = view
        self._root_view = view  # keeps a paged source alive across rebuilds
        self._syncing = False
        self._root_cache: Dict[Tuple, View] = {}
        self._max_slicers = max_slicers
        self._log_lines: List[str] = []
        self._build_widgets()
        self._unlisten = log.listen(self._on_log)
        for f in view.filters:  # carry existing slices along as an opaque query-ish step
            self.slices.append(("filter", f))
        self._rebuild()

    def close(self) -> None:
        """Stop listening to the step log."""
        self._unlisten()

    def _on_log(self, entry: Any) -> None:
        self._log_lines.append(entry.format())
        self._log_lines = self._log_lines[-200:]
        self._refresh_log()

    _LOG_THEME = {
        "light": {"text": "#333", "bg": "#fafafa", "border": "#e5e5e5"},
        "graphite": {"text": "#c7ccd6", "bg": "#1b2128", "border": "#333a45"},
    }

    def _refresh_log(self) -> None:
        lines = "\n".join(_html.escape(l) for l in self._log_lines[-60:])
        c = self._LOG_THEME.get(self.display.get("theme", "light"), self._LOG_THEME["light"])
        self.w_log.value = (
            f"<div style='font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11px;color:{c['text']};background:{c['bg']};"
            f"border:1px solid {c['border']};border-radius:6px;padding:6px 8px;height:120px;overflow-y:auto;white-space:pre;"
            "display:flex;flex-direction:column-reverse'>"
            f"<div>{lines}</div></div>"
        )

    # ------------------------------------------------------------------ widgets

    def _build_widgets(self) -> None:
        prof = self._profile
        columns = list(self._source.columns)
        numeric = [c.name for c in prof if c.kind == NUMERIC]
        labels = [c.name for c in prof if c.kind in (CATEGORICAL, BOOLEAN)]
        st = {"description_width": "80px"}

        # -- top bar
        self.w_mode = W.ToggleButtons(options=[("Pivot", PIVOT), ("Histogram", HIST), ("Chains", CHAINS)], value=self.mode,
                                      tooltips=["heatmap table", "bar chart", "Markov transition matrix of a state column"])
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
        self.w_method = W.Dropdown(options=list(METHODS), value="kmeans", description="method", style=st)
        self.w_cluster_on = W.Dropdown(options=[("pivot rows", None)] + [(c, c) for c in numeric], value=None, description="on", style=st)
        self.w_collapse = W.Checkbox(value=False, description="collapse rows into clusters")
        self.w_cocluster = W.Dropdown(options=[("off", None)] + [(m, m) for m in COMETHODS], value=None, description="Co-cluster", style=st)
        self.w_dim = W.Dropdown(options=[], description="Dimension", style=st)
        self.w_coarser = W.Button(description="Coarser", icon="compress", tooltip="/24 -> /16, hour -> day, fewer bins")
        self.w_finer = W.Button(description="Finer", icon="expand")
        reduce_tab = W.VBox([
            W.HTML("<i>Reduce size: sample rows, shrink the box (Layout tab), roll dimensions up, cluster similar rows, "
                   "or co-cluster rows and columns into blocks.</i>"),
            self.w_sample,
            W.HBox([self.w_cluster, self.w_method, self.w_cluster_on, self.w_collapse]),
            W.HBox([self.w_cocluster]),
            W.HBox([self.w_dim, self.w_coarser, self.w_finer]),
        ])

        # -- chains tab
        self.w_state = W.Dropdown(options=[("(pick a state column)", None)] + [(c, c) for c in labels], value=None, description="State", style=st)
        self.w_chain_by = W.Dropdown(options=[("(none)", None)] + [(c, c) for c in labels], value=None, description="Entity", style=st)
        self.w_chain_time = W.Dropdown(options=[("(row order)", None)] + [(c.name, c.name) for c in prof if c.kind == DATETIME], value=None, description="Time", style=st)
        self.w_chain_norm = W.Checkbox(value=False, description="row probabilities")
        self.w_sequences = W.HTML()
        chains_tab = W.VBox([
            W.HTML("<i>Markov chains: what follows what within an entity. Rows = from, columns = to.</i>"),
            W.HBox([self.w_state, self.w_chain_by, self.w_chain_time, self.w_chain_norm]),
            self.w_sequences,
        ])

        # -- style tab
        self.w_theme = W.Dropdown(options=["light", "graphite"], value=str(self.display.get("theme", "light")), description="Theme", style=st)
        self.w_heat = W.Dropdown(options=["table", "column", "row", "surprise", "none"], value=str(self.display.get("heat", "table")), description="Heat", style=st)
        self.w_totals = W.Checkbox(value=bool(self.display.get("totals", False)), description="totals")
        self.w_bars = W.Checkbox(value=bool(self.display.get("bars", False)), description="in-cell bars")
        self.w_compact = W.Checkbox(value=bool(self.display.get("compact", False)), description="compact numbers")
        self.w_subtotals = W.Checkbox(value=bool(self.display.get("subtotals", False)), description="subtotals (nested rows)")
        self.w_outline = W.Checkbox(value=bool(self.display.get("outline", False)), description="outline / collapsible groups")
        self.w_width = W.IntSlider(value=int(self.display.get("width", 760)), min=300, max=1600, step=20, description="Width", style=st)
        self.w_height = W.IntSlider(value=int(self.display.get("height", 340)), min=160, max=900, step=20, description="Height", style=st)
        style_tab = W.VBox([
            W.HBox([self.w_theme, self.w_heat, self.w_totals, self.w_bars, self.w_compact]),
            W.HBox([self.w_subtotals, self.w_outline]),
            W.HBox([self.w_width, self.w_height]),
        ])

        # -- fields tab: draggable columns with stats, dropped into Rows/Columns/Values/Slicers
        def _col_of(x: Any) -> Optional[str]:
            return x if isinstance(x, str) else (x.get("column") if isinstance(x, dict) else getattr(x, "column", None))

        init_rows = [c for c in (_col_of(x) for x in (self.spec.get("rows") or [])) if c]
        init_cols = [c for c in (_col_of(x) for x in (self.spec.get("cols") or [])) if c]
        init_values = [self.spec["values"]] if self.spec.get("values") else []
        self.w_fields = make_field_list(prof, theme=str(self.display.get("theme", "light")), rows=init_rows, cols=init_cols, values=init_values, slicers=[])
        self.w_fields.observe(self._on_fields, names=["rows", "cols", "values", "slicers"])
        self.w_field_slicer_box = W.VBox()
        fields_tab = W.VBox([
            W.HTML(
                "<i>Drag fields into Rows / Columns / Values / Slicers — rows within rows and columns within "
                "columns by dropping more than one field on an axis, in the order you drop them. Each chip shows "
                "kind, semantic type, non-null count (n), distinct count (≠) and variety: distinct ÷ non-null, "
                "as a small bar (a handful of repeated labels reads near-empty; an id-like column reads full).</i>"
            ),
            self.w_fields,
            self.w_field_slicer_box,
        ])

        # -- code, profile, data, stats tabs; output; log panel
        self.w_code = W.HTML()
        self.w_out = W.HTML()
        self.w_status = W.HTML()
        self.w_log = W.HTML()
        self.w_profile = W.HTML("<pre style='font-size:11px'>" + _html.escape(self._profile.summary().to_string(index=False)) + "</pre>")
        self.w_data = W.HTML()
        self.w_stats = W.HTML()
        self.w_stats_btn = W.Button(description="Refresh stats", icon="clock-o")
        stats_tab = W.VBox([W.HTML("<i>The seven costliest kinds of step so far (wall time, CPU time, peak memory delta).</i>"), self.w_stats_btn, self.w_stats])

        self.tabs = W.Tab(children=[layout_tab, hist_tab, slice_tab, reduce_tab, chains_tab, style_tab, self.w_code, self.w_profile, self.w_data, stats_tab, fields_tab])
        for i, t in enumerate(["Layout", "Histogram", "Slicers", "Reduce & cluster", "Chains", "Style", "Code", "Profile", "Data", "Stats", "Fields"]):
            self.tabs.set_title(i, t)
        self.FIELDS_TAB = 10
        top = W.HBox([self.w_mode, self.w_best, self.w_suggest_btn, self.w_suggest, self.w_undo, self.w_reset])
        self.box = W.VBox([top, self.tabs, self.w_status, self.w_out, W.HTML("<b style='font-size:11px;color:#666'>log</b>"), self.w_log])
        self._refresh_data_tab()
        self._refresh_log()

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
        for w in (self.w_stacked, self.w_density, self.w_logy, self.w_theme, self.w_heat, self.w_totals, self.w_bars,
                  self.w_compact, self.w_subtotals, self.w_outline, self.w_width, self.w_height):
            w.observe(self._on_style, names="value")
        for _, (kind, w) in self.w_slicers.items():
            w.observe(self._on_slicers, names="value" if kind != "range" and kind != "days" else "index")
        self.w_query.observe(self._on_query, names="value")  # continuous_update=False: fires on Enter/blur
        self.w_top_col.observe(self._on_slicers, names="value")
        self.w_top_n.observe(self._on_slicers, names="value")
        self.w_clear.on_click(lambda _: self._act(self._clear_slices))
        for w in (self.w_sample, self.w_cluster, self.w_method, self.w_cluster_on, self.w_collapse, self.w_cocluster):
            w.observe(self._on_reduce, names="value")
        self.w_coarser.on_click(lambda _: self._act(lambda: self._step(-1)))
        self.w_finer.on_click(lambda _: self._act(lambda: self._step(+1)))
        for w in (self.w_state, self.w_chain_by, self.w_chain_time, self.w_chain_norm):
            w.observe(self._on_chain, names="value")
        self.w_stats_btn.on_click(lambda _: self._refresh_stats())

    # ------------------------------------------------------------------ recipe -> view

    def _snapshot(self) -> Dict[str, Any]:
        return {
            "spec": dict(self.spec), "slices": list(self.slices), "mode": self.mode, "hist": dict(self.hist),
            "reduce": dict(self.reduce), "display": dict(self.display), "options": self.options, "refit": self.refit_sliced,
            "chain": dict(self.chain),
        }

    def _restore(self, snap: Dict[str, Any]) -> None:
        self.spec, self.slices, self.mode = dict(snap["spec"]), list(snap["slices"]), snap["mode"]
        self.hist, self.reduce, self.display = dict(snap["hist"]), dict(snap["reduce"]), dict(snap["display"])
        self.options, self.refit_sliced = snap["options"], snap["refit"]
        self.chain = dict(snap.get("chain", self.chain))
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
            self.w_method.value, self.w_cocluster.value = r.get("method", "kmeans"), r.get("cocluster")
            c = self.chain
            self.w_state.value, self.w_chain_by.value, self.w_chain_time.value, self.w_chain_norm.value = c.get("state"), c.get("by"), c.get("time"), bool(c.get("normalize"))
            self.w_stacked.value, self.w_density.value = bool(d.get("stacked")), bool(d.get("density", True))
            self.w_logy.value, self.w_heat.value = bool(d.get("log_y")), str(d.get("heat", "table"))
            self.w_totals.value, self.w_bars.value, self.w_compact.value = bool(d.get("totals")), bool(d.get("bars")), bool(d.get("compact"))
            self.w_width.value, self.w_height.value = int(d.get("width", 760)), int(d.get("height", 340))
            self.w_theme.value = str(d.get("theme", "light"))
            self.w_subtotals.value, self.w_outline.value = bool(d.get("subtotals")), bool(d.get("outline"))
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
            self.w_fields.theme = str(d.get("theme", "light"))
            self.w_fields.set_zones(rows=list(names(rows)), cols=list(names(cols)),
                                    values=[self.spec["values"]] if self.spec.get("values") else [], slicers=self.field_slicers)
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
        n = self.reduce["sample"]
        key = (n, repr(self.spec), repr(self.options))
        if key not in self._root_cache:
            base = self._root_view
            if n and n < len(base.data):
                base = base.sample(int(n), seed=self.options.seed)
            if base.paged is not None and not n:
                self._root_cache[key] = View.fit(base.paged, options=self.options, **self.spec)
            else:
                self._root_cache[key] = View.fit(base.source, options=self.options, **self.spec)
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
        if self.mode == CHAINS:
            c = self.chain
            if not c.get("state"):
                raise ValueError("Chains: pick a state column in the Chains tab")
            from . import chains as _chains

            v = _chains(v.data, c["state"], by=c.get("by"), time=c.get("time"), normalize=bool(c.get("normalize")),
                        max_rows=self.options.max_rows, max_cols=self.options.max_cols)
            self._refresh_sequences(c)
            return v.style(**{k: x for k, x in self.display.items() if k != "heat"})
        k = self.reduce["cluster"]
        if k is not None:
            v = v.cluster(None if k == "auto" else int(k), on=self.reduce["cluster_on"], method=self.reduce.get("method", "kmeans"),
                          collapse=bool(self.reduce["collapse"]))
        if self.reduce.get("cocluster"):
            v = v.cocluster(method=self.reduce["cocluster"])
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
        if self.mode == CHAINS:
            c = {k: x for k, x in self.chain.items() if x}
            state = c.pop("state", None)
            lines.append(f"v = p2h.chains(v.data, {state!r}{', ' if c else ''}{_kw(c)})")
        if self.reduce["cluster"] is not None:
            k = self.reduce["cluster"]
            kw = {"on": self.reduce["cluster_on"]} if self.reduce["cluster_on"] else {}
            if self.reduce.get("method", "kmeans") != "kmeans":
                kw["method"] = self.reduce["method"]
            if self.reduce["collapse"]:
                kw["collapse"] = True
            lines.append(f"v = v.cluster({'' if k == 'auto' else k}{', ' if kw and k != 'auto' else ''}{_kw(kw)})")
        if self.reduce.get("cocluster"):
            lines.append(f"v = v.cocluster(method={self.reduce['cocluster']!r})")
        if self.mode == HIST:
            h = {k: v for k, v in self.hist.items() if v}
            lines.append(f"v = v.histogram({_kw(h)})" if h else "v = v.toggle()")
        defaults = {"density": True, "heat": "table", "theme": "light"}
        disp = {k: v for k, v in self.display.items()
                if k not in ("width", "height") and v not in (None, False) and defaults.get(k) != v}
        if disp:
            lines.append(f"v = v.style({_kw(disp)})")
        lines.append("v")
        return "\n".join(lines)

    def _rebuild(self) -> None:
        with log.step("explorer", "rebuild") as st:
            self.view = self.build()
            self.w_out.value = self.view.html(title=False)  # the status line already shows it
            st.detail = f"rebuild -> {self.view.layout.describe()}"
        self.w_status.value = f"<span style='color:#555;font-size:12px'>{_html.escape(self.view.title())}</span>"
        self.w_code.value = "<pre style='font-size:12px'>" + _html.escape(self.code()) + "</pre>"
        self._sync_widgets()
        self._refresh_stats()
        self._refresh_field_slicers()
        self._refresh_log()  # picks up a theme change immediately, not just on the next log line

    def _refresh_stats(self) -> None:
        df = log.stats(7)
        if df.empty:
            self.w_stats.value = "<i>nothing logged yet</i>"
            return
        rows = "".join(
            f"<tr><td style='padding:2px 8px'>{_html.escape(str(r.step))}</td><td style='text-align:right;padding:2px 8px'>{int(r.calls)}</td>"
            f"<td style='text-align:right;padding:2px 8px'>{r.seconds:.2f}</td><td style='text-align:right;padding:2px 8px'>{r.cpu:.2f}</td>"
            f"<td style='text-align:right;padding:2px 8px'>{r.mem_mb:+.0f}</td><td style='padding:2px 8px;color:#666'>{_html.escape(str(r.last_detail))[:70]}</td></tr>"
            for r in df.itertuples()
        )
        self.w_stats.value = (
            "<table style='font-size:12px;border-collapse:collapse'><tr style='background:#f5f6f8'><th style='padding:2px 8px;text-align:left'>step</th>"
            "<th style='padding:2px 8px'>calls</th><th style='padding:2px 8px'>seconds</th><th style='padding:2px 8px'>cpu s</th>"
            "<th style='padding:2px 8px'>peak MB</th><th style='padding:2px 8px;text-align:left'>last</th></tr>" + rows + "</table>"
        )

    def _refresh_data_tab(self) -> None:
        sv = self._root_view.survey
        if sv is None:
            from ._survey import survey as _survey

            try:
                sv = _survey(self._source)
            except Exception:  # noqa: BLE001
                sv = None
        if sv is None:
            self.w_data.value = "<i>no survey available</i>"
            return
        paged = self._root_view.paged
        head = f"<b>{'paged source' if paged else 'in memory'}</b><br>" + _html.escape(sv.summary()).replace("\n", "<br>")
        self.w_data.value = f"<div style='font-size:12px;line-height:1.5'>{head}</div>"

    def _refresh_sequences(self, c: Dict[str, Any]) -> None:
        try:
            seq = _sequences(self.view.data if self.view.mode != CHAINS else self._root().data, c["state"], by=c.get("by"), time=c.get("time"), length=3, n=8)
        except Exception:  # noqa: BLE001
            seq = None
        if seq is None or seq.empty:
            self.w_sequences.value = ""
            return
        rows = "".join(f"<tr><td style='padding:2px 8px'>{_html.escape(str(r.chain))}</td><td style='text-align:right;padding:2px 8px'>{int(r.count):,}</td><td style='text-align:right;padding:2px 8px'>{r.share:.1%}</td></tr>" for r in seq.itertuples())
        self.w_sequences.value = "<div style='font-size:12px'><b>most frequent 3-step chains</b><table style='border-collapse:collapse'>" + rows + "</table></div>"

    def _sync_widgets(self) -> None:
        self._syncing = True
        try:
            lay = self.view.layout
            self.w_mode.value = self.mode
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
            if self.mode != CHAINS:
                self.w_fields.set_zones(
                    rows=[d.column for d in lay.rows if d.column in self.w_rows.options],
                    cols=[d.column for d in lay.cols if d.column in self.w_cols.options],
                    values=[lay.values] if lay.values else [],
                    slicers=self.field_slicers,
                )
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

    def _on_fields(self, change: Any) -> None:
        def go() -> None:
            w = self.w_fields
            rows, cols, values, slicers = list(w.rows), list(w.cols), list(w.values), list(w.slicers)
            cols = [c for c in cols if c not in rows]
            value = values[0] if values else None
            self.spec = {
                "rows": rows or None,
                "cols": cols or None,
                "values": value,
                "agg": COUNT if value is None else self.w_agg.value,
            }
            self.field_slicers = slicers
            self.refit_sliced = False
            self._syncing = True
            try:
                self.w_rows.value = tuple(c for c in rows if c in self.w_rows.options)
                self.w_cols.value = tuple(c for c in cols if c in self.w_cols.options)
                if value is not None:
                    self.w_values.value = value
            finally:
                self._syncing = False
        self._act(go)

    def _refresh_field_slicers(self) -> None:
        rows = []
        for name in self.field_slicers:
            entry = self.w_slicers.get(name)
            if entry is not None:
                rows.append(W.VBox([W.HTML(f"<b style='font-size:11px'>{_html.escape(name)}</b>"), entry[1]]))
            else:
                rows.append(W.HTML(f"<span style='font-size:11px;color:#889'>{_html.escape(name)}: no quick filter built for this column — use the Slicers tab (query or top-N)</span>"))
        self.w_field_slicer_box.children = rows

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
                theme=self.w_theme.value, heat=self.w_heat.value, totals=self.w_totals.value, bars=self.w_bars.value,
                compact=self.w_compact.value, subtotals=self.w_subtotals.value, outline=self.w_outline.value,
                width=self.w_width.value, height=self.w_height.value,
            )
            self.w_fields.theme = self.w_theme.value
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
                "method": self.w_method.value, "cocluster": self.w_cocluster.value,
            }
        self._act(go)

    def _on_chain(self, change: Any) -> None:
        def go() -> None:
            self.chain = {"state": self.w_state.value, "by": self.w_chain_by.value, "time": self.w_chain_time.value,
                          "normalize": bool(self.w_chain_norm.value)}
            if self.chain["state"]:
                self.mode = CHAINS
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

    # ------------------------------------------------------------------ static snapshot

    def snapshot_html(self, *, active_tab: Optional[int] = None) -> str:
        """A static HTML picture of the interface (for docs, screenshots, sharing).

        Widgets are drawn as plain form elements with their current values; the output,
        status line and log are the real thing.
        """
        tab = self.tabs.selected_index if active_tab is None else active_tab
        if tab is None:
            tab = 0
        theme = str(self.display.get("theme", "light"))
        c = self._CHROME_THEME.get(theme, self._CHROME_THEME["light"])

        def render(w: Any) -> str:
            name = type(w).__name__
            if name in ("FieldList", "FieldListFallback"):
                return w.snapshot_html()
            if name in ("VBox", "HBox", "Box"):
                direction = "column" if name == "VBox" else "row"
                inner = "".join(render(x) for x in w.children)
                return f"<div style='display:flex;flex-direction:{direction};flex-wrap:wrap;gap:6px;align-items:flex-start;margin:2px 0'>{inner}</div>"
            if name == "Tab":
                heads = "".join(
                    f"<span style='padding:5px 12px;border:1px solid {c['border']};border-bottom:{'none' if i == tab else '1px solid ' + c['border']};"
                    f"border-radius:8px 8px 0 0;background:{c['panel'] if i == tab else c['bg']};color:{c['text']};"
                    f"font-weight:{600 if i == tab else 400}'>{_html.escape(w.get_title(i) or str(i))}</span>"
                    for i in range(len(w.children))
                )
                body = render(w.children[tab]) if w.children else ""
                return (
                    f"<div><div style='display:flex;gap:2px'>{heads}</div>"
                    f"<div style='border:1px solid {c['border']};border-radius:0 8px 8px 8px;padding:10px;background:{c['panel']}'>{body}</div></div>"
                )
            if name == "HTML":
                return f"<div style='color:{c['text']}'>{w.value}</div>"
            desc = _html.escape(str(getattr(w, "description", "") or ""))
            label = f"<label style='color:{c['muted']};margin-right:4px'>{desc}</label>" if desc else ""
            if name == "Button":
                return f"<button style='padding:4px 12px;border:1px solid {c['border']};border-radius:8px;background:{c['field']};color:{c['text']}'>{desc}</button>"
            if name == "ToggleButtons":
                btns = "".join(
                    f"<span style='padding:4px 12px;border:1px solid {c['border']};background:{c['accent'] if v == w.value else c['field']};"
                    f"color:{c['on_accent'] if v == w.value else c['text']}'>{_html.escape(str(lab))}</span>"
                    for lab, v in (w.options if isinstance(w.options[0], tuple) else [(o, o) for o in w.options])
                )
                return f"<span style='display:inline-flex;border-radius:8px;overflow:hidden'>{btns}</span>"
            if name in ("Dropdown",):
                opts = w.options if (w.options and isinstance(w.options[0], tuple)) else [(o, o) for o in w.options]
                items = "".join(f"<option{' selected' if v == w.value else ''}>{_html.escape(str(lab))}</option>" for lab, v in opts[:60])
                return f"<span>{label}<select style='{c['input_style']}'>{items}</select></span>"
            if name == "SelectMultiple":
                opts = w.options if (w.options and isinstance(w.options[0], tuple)) else [(o, o) for o in w.options]
                items = "".join(f"<option{' selected' if v in w.value else ''}>{_html.escape(str(lab))}</option>" for lab, v in opts[:60])
                return f"<span>{label}<select multiple size='{min(6, max(2, len(opts)))}' style='{c['input_style']}'>{items}</select></span>"
            if name == "Checkbox":
                return f"<label style='color:{c['text']}'><input type='checkbox'{' checked' if w.value else ''}> {desc}</label>"
            if name in ("IntSlider", "FloatSlider"):
                return f"<span style='color:{c['text']}'>{label}<input type='range' min='{w.min}' max='{w.max}' value='{w.value}'> <b>{w.value}</b></span>"
            if name == "SelectionRangeSlider":
                lo, hi = w.index
                labs = [o[0] if isinstance(o, tuple) else str(o) for o in w.options]
                return f"<span style='color:{c['text']}'>{label}<input type='range' min='0' max='{len(labs) - 1}' value='{lo}'> <b>{_html.escape(labs[lo])} .. {_html.escape(labs[hi])}</b></span>"
            if name == "Text":
                return f"<span>{label}<input type='text' value='{_html.escape(w.value)}' placeholder='{_html.escape(w.placeholder or '')}' size='48' style='{c['input_style']}'></span>"
            return f"<span style='color:{c['text']}'>{label}{_html.escape(str(getattr(w, 'value', '')))}</span>"

        body = render(self.box)
        return (
            f"<div style='font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;font-size:12px;"
            f"max-width:1150px;background:{c['bg']};color:{c['text']};padding:14px;border-radius:12px'>{body}</div>"
        )

    _CHROME_THEME = {
        "light": {
            "bg": "#ffffff", "panel": "#ffffff", "field": "#f7f7f7", "border": "#ccc",
            "text": "#222", "muted": "#555", "accent": "#4a7ebb", "on_accent": "#fff",
            "input_style": "border:1px solid #ccc;border-radius:6px;padding:2px 4px;background:#fff;color:#222",
        },
        "graphite": {
            "bg": "#161a20", "panel": "#1b2128", "field": "#20262e", "border": "#333a45",
            "text": "#e5e9ef", "muted": "#98a1b0", "accent": "#5b9be0", "on_accent": "#0f1216",
            "input_style": "border:1px solid #333a45;border-radius:6px;padding:2px 4px;background:#20262e;color:#e5e9ef",
        },
    }

    # ------------------------------------------------------------------ display

    def _repr_mimebundle_(self, **kw: Any) -> Any:
        return self.box._repr_mimebundle_(**kw)

    def _ipython_display_(self) -> None:  # pragma: no cover - notebook only
        from IPython.display import display

        display(self.box)


def explore(data: Any, **fit_kwargs: Any) -> Explorer:
    """Open the interactive explorer on a frame/path/records or an existing :class:`View`."""
    from . import fit as _fit  # package-level fit: surveys files/DuckDB and pages big sources

    ui_kw = {k: fit_kwargs.pop(k) for k in ("width", "height", "max_slicers") if k in fit_kwargs}
    view = data if isinstance(data, View) else _fit(data, **fit_kwargs)
    return Explorer(view, **ui_kw)


__all__ = ["Explorer", "explore"]
