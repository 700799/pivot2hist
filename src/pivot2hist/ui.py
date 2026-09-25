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

import datetime as _dt
import html as _html
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import quote as _quote

import numpy as np
import pandas as pd

try:
    import ipywidgets as W
except ImportError as e:  # pragma: no cover
    raise ImportError("pivot2hist.ui needs ipywidgets: pip install 'pivot2hist[jupyter]'") from e

from ._binning import RULES, human
from ._chains import sequences as _sequences
from ._cluster import COMETHODS, METHODS
from ._compare import METRICS, METRIC_HELP, Comparison, Facets
from ._explain import Explanation, cell_filters, parse_cell
from ._prompt import DEFAULT_QUESTION, Prompt
from ._fit import AGGS, COUNT, FitOptions, Layout
from ._log import log
from ._profile import BOOLEAN, CATEGORICAL, DATETIME, NUMERIC, Profile, profile
from ._view import HIST, PIVOT, View
from .ui_fields import _set_mark, make_field_list
from .ui_output import make_copy_button, make_output

CHAINS = "chains"

_NUM_STEPS = 24


def _kw(d: Dict[str, Any]) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in d.items())


class Explorer:
    """The widget app. ``explorer.view`` is the current :class:`View`; ``explorer.code()``
    is the Python that reproduces it."""

    def __init__(self, view: View, *, width: int = 760, height: int = 340, max_slicers: int = 8,
                 marks: Optional[Dict[str, str]] = None, _profile: Optional[Profile] = None):
        self._source = view.source
        self._profile = _profile if _profile is not None else profile(self._source)
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
        self.marks: Dict[str, str] = dict(marks) if marks else {}
        self.refit_sliced = False
        self.history: List[Dict[str, Any]] = []
        self.checkpoints: List[Dict[str, Any]] = []
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

    # ------------------------------------------------------------------ chrome

    @staticmethod
    def _cluster(caption: str, *widgets: Any) -> Any:
        """A labelled group of top-bar controls (a caption plus the widgets, in one frame)."""
        label = W.HTML(
            f"<span class='p2h-caption' style='font-size:10px;letter-spacing:.08em;text-transform:uppercase;"
            f"color:#889;margin:0 4px 0 2px'>{_html.escape(caption)}</span>",
            layout=W.Layout(align_self="center"),
        )
        box = W.HBox([label, *widgets], layout=W.Layout(align_items="center"))
        box.add_class("p2h-cluster")
        return box

    def _chrome_css(self) -> str:
        """The stylesheet that gives the explorer its visual hierarchy: framed control
        clusters in the top bar, section-styled group tabs, plain inner tabs."""
        c = self._CHROME_THEME.get(str(self.display.get("theme", "light")), self._CHROME_THEME["light"])
        return (
            "<style>"
            ".p2h-topbar{gap:10px;flex-wrap:wrap;margin:2px 0 6px 0}"
            f".p2h-cluster{{padding:3px 8px 3px 4px;border:1px solid {c['border']};border-radius:8px;background:{c['field']};gap:4px}}"
            ".p2h-groups .lm-TabBar-tab,.p2h-groups .p-TabBar-tab{font-weight:600;text-transform:uppercase;letter-spacing:.05em;font-size:12px}"
            f".p2h-groups .lm-TabBar-tab.lm-mod-current,.p2h-groups .p-TabBar-tab.p-mod-current{{box-shadow:inset 0 -3px 0 {c['accent']}}}"
            ".p2h-subtabs .lm-TabBar-tab,.p2h-subtabs .p-TabBar-tab{font-weight:400;text-transform:none;letter-spacing:0;font-size:13px}"
            ".p2h-subtabs .lm-TabBar-tab.lm-mod-current,.p2h-subtabs .p-TabBar-tab.p-mod-current{box-shadow:none;font-weight:600}"
            "</style>"
        )

    # ------------------------------------------------------------------ tabs

    def tab_names(self) -> List[str]:
        """Every tab, in display order (grouped: Explore, Analyze, Output, Session)."""
        return list(self._tab_index)

    def select_tab(self, name: str) -> None:
        """Bring tab ``name`` (e.g. ``"Inspect"``) to the front, whichever group it is in."""
        if name not in self._tab_index:
            raise KeyError(f"no tab {name!r}; have {self.tab_names()}")
        gi, i = self._tab_index[name]
        self.tabs.selected_index = gi
        self.tabs.children[gi].selected_index = i

    def current_tab(self) -> str:
        """The name of the tab in front."""
        gi = self.tabs.selected_index or 0
        inner = self.tabs.children[gi]
        return inner.get_title(inner.selected_index or 0)

    def tab_group(self, name: str) -> str:
        """The group tab ``name`` lives in."""
        gi, _ = self._tab_index[name]
        return self.tabs.get_title(gi)

    def clone(self, view: Optional[View] = None) -> "Explorer":
        """A second, independent explorer over the same data, for a new notebook cell.

        Re-profiling a wide or large frame is real work, and ``Explorer.__init__``
        normally does it every time; ``clone()`` skips it - the new explorer shares this
        one's already-computed :class:`~pivot2hist.Profile` and underlying source frame
        (no copy) - and carries over the field highlights (:meth:`paint`) since those are
        about the columns, not this cell's particular layout. Everything else (the
        layout/slices/style recipe, undo history, checkpoints) starts fresh, so the two
        cells can't step on each other::

            explorer = p2h.explore(df)                 # cell 1: explore freely
            cell2 = explorer.clone()                    # cell 2: same data, no re-profiling
            cell2.w_rows.value = ("dst_port",)           # diverges independently

        Pass ``view`` to start the clone from a different layout instead of this one's.
        """
        return Explorer(
            view if view is not None else self.view, width=int(self.display.get("width", 760)),
            height=int(self.display.get("height", 340)), max_slicers=self._max_slicers,
            marks=dict(self.marks), _profile=self._profile,
        )

    def paint(self, name: str, color: Optional[str] = None) -> None:
        """Highlight field ``name`` in the Fields tab with ``color`` (a name from
        :data:`pivot2hist.ui_fields.MARK_PALETTE` - ``"red"``, ``"orange"``, ``"yellow"``,
        ``"green"``, ``"blue"``, ``"purple"`` - or any CSS color string); ``color=None``
        clears it. Marks are cosmetic bookkeeping only (they never affect the fitted
        layout or the data) and are carried along by :meth:`clone` and saved checkpoints.
        """
        if name not in self._profile:
            raise KeyError(f"unknown column {name!r}")
        self.marks = _set_mark(self.marks, name, color)
        self.w_fields.marks = dict(self.marks)

    def unmark(self, name: Optional[str] = None) -> None:
        """Clear one field's highlight (:meth:`paint`), or every one when ``name`` is
        ``None``."""
        self.marks = {} if name is None else {k: v for k, v in self.marks.items() if k != name}
        self.w_fields.marks = dict(self.marks)

    # ------------------------------------------------------------------ timeline / checkpoints

    def save_checkpoint(self, note: str = "") -> None:
        """Bookmark the current recipe (layout, slices, style, cluster/chain settings,
        field marks — everything the undo history tracks) as a labeled point on the
        Timeline tab, with an optional ``note`` describing what it shows.

        Move the Timeline tab's slider (or click its Play button to step through every
        checkpoint automatically) to jump back to any of them — the view is rebuilt
        exactly as it was at that point every time, like undo but to a *named*,
        permanent point rather than just the last change. Checkpoints are cheap: like
        everything else here they hold the recipe, not a copy of the data.

        ::

            explorer = p2h.explore(df)
            # ... adjust rows/cols/slices/style until it looks right ...
            explorer.save_checkpoint("clean baseline, all traffic")
            # ... slice down to a suspicious host, try a few things ...
            explorer.save_checkpoint("host 10.0.4.12 spike investigation")
            explorer.goto_checkpoint(0)   # back to the baseline, or drag the slider
        """
        entry = {
            "snapshot": self._snapshot(),
            "note": str(note).strip(),
            "label": self.view.describe(),
            "time": _dt.datetime.now().strftime("%H:%M:%S"),
        }
        self.checkpoints.append(entry)
        self._syncing = True
        try:
            self.w_checkpoint_note.value = ""
        finally:
            self._syncing = False
        self._refresh_timeline(select=len(self.checkpoints) - 1)

    def goto_checkpoint(self, idx: int) -> None:
        """Jump to checkpoint ``idx`` (0-based, oldest first; negative indexes from the
        end like a list) and re-render the view as it was then — the same thing moving
        the Timeline tab's slider does."""
        if not self.checkpoints:
            raise IndexError("no checkpoints saved yet")
        n = len(self.checkpoints)
        idx = idx + n if idx < 0 else idx
        if not 0 <= idx < n:
            raise IndexError(f"checkpoint index {idx} out of range for {n} checkpoint(s)")
        self._syncing = True
        try:
            self.w_timeline_slider.value = idx
            self.w_timeline_play.value = idx
        finally:
            self._syncing = False
        self._apply_checkpoint(idx)

    def _delete_checkpoint(self) -> None:
        if not self.checkpoints:
            return
        idx = max(0, min(self.w_timeline_slider.value, len(self.checkpoints) - 1))
        del self.checkpoints[idx]
        self._refresh_timeline()

    def _on_timeline(self, change: Any) -> None:
        if self._syncing or not self.checkpoints:
            return
        idx = max(0, min(int(self.w_timeline_slider.value), len(self.checkpoints) - 1))
        self._apply_checkpoint(idx)

    def _apply_checkpoint(self, idx: int) -> None:
        try:
            self._restore(self.checkpoints[idx]["snapshot"])
            self._rebuild()
        except Exception as e:  # noqa: BLE001 - surface any failure in the status line
            self.w_status.value = f"<span style='color:#b00020'>{_html.escape(type(e).__name__)}: {_html.escape(str(e))}</span>"
        self._refresh_timeline_label(idx)

    def _refresh_timeline(self, select: Optional[int] = None) -> None:
        n = len(self.checkpoints)
        idx = select if select is not None else min(self.w_timeline_slider.value, max(0, n - 1))
        idx = max(0, min(idx, max(0, n - 1)))
        self._syncing = True
        try:
            self.w_timeline_slider.max = max(0, n - 1)
            self.w_timeline_play.max = max(0, n - 1)
            self.w_timeline_slider.value = idx
            self.w_timeline_play.value = idx
        finally:
            self._syncing = False
        self._refresh_timeline_label(idx)
        rows = []
        for i, c in enumerate(self.checkpoints):
            note = _html.escape(c["note"]) if c["note"] else "<i style='color:#999'>(no note)</i>"
            marker = " style='background:#eef3ff'" if n and i == idx else ""
            rows.append(
                f"<tr{marker}><td style='padding:2px 8px;color:#888'>{i}</td>"
                f"<td style='padding:2px 8px'>{_html.escape(c['time'])}</td>"
                f"<td style='padding:2px 8px'>{_html.escape(c['label'])}</td>"
                f"<td style='padding:2px 8px'>{note}</td></tr>"
            )
        body = "".join(rows) if rows else (
            "<tr><td colspan='4' style='padding:6px;color:#999'><i>no checkpoints yet — set the layout how "
            "you want it and click Save checkpoint</i></td></tr>"
        )
        self.w_timeline_list.value = (
            "<table style='font-size:11px;border-collapse:collapse;width:100%'>"
            "<tr style='background:#f5f6f8'><th style='padding:2px 8px;text-align:left'>#</th>"
            "<th style='padding:2px 8px;text-align:left'>time</th><th style='padding:2px 8px;text-align:left'>state</th>"
            "<th style='padding:2px 8px;text-align:left'>note</th></tr>" + body + "</table>"
        )

    def _refresh_timeline_label(self, idx: int) -> None:
        if not self.checkpoints:
            self.w_timeline_note.value = "<i style='color:#999'>no checkpoints saved yet</i>"
            return
        c = self.checkpoints[idx]
        note = _html.escape(c["note"]) if c["note"] else "<i style='color:#999'>(no note)</i>"
        self.w_timeline_note.value = f"<div style='font-size:12px'><b>#{idx}</b> · {_html.escape(c['time'])} · {_html.escape(c['label'])}<br>{note}</div>"

    # ------------------------------------------------------------------ insights

    def calculate(self, *, sensitivity: Optional[float] = None) -> Dict[str, Any]:
        """Run :meth:`View.insights` on the data currently in view (every slice already
        applied) and refresh the Insights tab - the same thing its Calculate button
        does. Local and non-LLM: distributions, a mixture-model check for multiple
        populations in one numeric column, skew, concentration, outliers, correlated
        columns, and surprising pivot cells, ranked by significance. Does not run on its
        own on every change (it's real work); call it again - "recalculate" is just
        calling this again - whenever the filtered data has moved on. Returns the report
        dict (``{"summary", "shape", "columns", "findings", "sensitivity"}``).

        ``sensitivity`` (0..1, default: the Insights tab's slider, itself defaulting to
        0.5) sets how much of the ranked findings list actually surfaces - higher shows
        more, including weaker findings; lower shows only the strongest. It isn't called
        "temperature": this is a deterministic threshold on a computed score, not
        sampling from a model.
        """
        s = self.w_sensitivity.value if sensitivity is None else sensitivity
        if sensitivity is not None:
            self._syncing = True
            try:
                self.w_sensitivity.value = s
            finally:
                self._syncing = False
        try:
            self._insights_report = self.view.insights(sensitivity=s)
        except Exception as e:  # noqa: BLE001 - surface any failure in the Insights tab, not a traceback
            self._insights_report = None
            self.w_insights.value = f"<span style='color:#b00020'>{_html.escape(type(e).__name__)}: {_html.escape(str(e))}</span>"
            return {}
        self._insights_view_id = id(self.view)
        self._render_insights()
        return self._insights_report

    def _render_insights(self) -> None:
        report = self._insights_report
        if report is None:
            self.w_insights.value = "<i style='color:#999'>click Calculate to analyze the data currently in view</i>"
            return
        stale = id(self.view) != self._insights_view_id
        banner = (
            "<div style='color:#a15c00;font-size:11px;margin-bottom:6px'>&#9888; the view has changed since "
            "this was calculated — click Calculate to refresh</div>"
        ) if stale else ""
        if report["findings"]:
            rows = "".join(
                f"<tr><td style='padding:2px 8px;text-align:right;color:#888'>{f['significance']:.2f}</td>"
                f"<td style='padding:2px 8px;color:#667'>{_html.escape(f['kind'])}</td>"
                f"<td style='padding:2px 8px'>{_html.escape(f['text'])}</td></tr>"
                for f in report["findings"]
            )
            findings_html = (
                "<table style='font-size:12px;border-collapse:collapse;width:100%'>"
                "<tr style='background:#f5f6f8'><th style='padding:2px 8px'>sig</th>"
                "<th style='padding:2px 8px;text-align:left'>kind</th>"
                "<th style='padding:2px 8px;text-align:left'>observation</th></tr>" + rows + "</table>"
            )
        else:
            findings_html = "<i style='color:#999'>nothing crossed the significance bar — try raising sensitivity</i>"
        self.w_insights.value = (
            banner + f"<div style='font-size:12px;margin-bottom:8px'>{_html.escape(report['summary'])}</div>" + findings_html
        )

    # ------------------------------------------------------------------ export

    def prompt(self, question: Optional[str] = None, **overrides: Any) -> Prompt:
        """Build the LLM prompt from what is on screen - :meth:`View.prompt` on the current
        view with the Export tab's toggles (the table, dataset columns, surprising cells,
        the Insights report when computed for this view, the Compare tab's comparison, the
        Inspect tab's cell; any of them overridable by keyword) - show it in the tab, wire
        the Copy / Download buttons, and return it. ``question`` defaults to the tab's
        question box, which defaults to a general "what stands out, what next" ask."""
        use_insights: Any = False
        if self.w_prompt_insights.value:
            fresh = self._insights_report is not None and self._insights_view_id == id(self.view)
            use_insights = self._insights_report if fresh else True
        kw: Dict[str, Any] = dict(
            table=self.w_prompt_table.value, profile=self.w_prompt_profile.value, anomalies=self.w_prompt_anomalies.value,
            insights=use_insights, sensitivity=self.w_sensitivity.value, max_rows=int(self.w_prompt_rows.value),
            compare=self.comparison if (self.w_prompt_compare.value and isinstance(self.comparison, Comparison)) else None,
            explain=self.cell if (self.w_prompt_cell.value and self.cell is not None) else None,
        )
        kw.update(overrides)
        q = question if question is not None else (self.w_prompt_question.value.strip() or None)
        p = self.view.prompt(q, **kw)
        self.prompt_text = p
        self.w_prompt_out.value = str(p)
        if hasattr(self.w_prompt_copy, "text"):
            self.w_prompt_copy.text = str(p)
        href = "data:text/markdown;charset=utf-8," + _quote(str(p))
        self.w_prompt_download.value = (
            f"<a download='pivot2hist-prompt.md' href='{href}' style='font-size:12px;margin-left:8px'>Download .md</a>"
        )
        self.w_prompt_meta.value = f"<span style='font-size:11px;color:#667'>{p.chars:,} characters ≈ {p.tokens:,} tokens</span>"
        return p

    def _build_prompt_clicked(self) -> None:
        try:
            self.prompt()
        except Exception as e:  # noqa: BLE001 - surface any failure in the tab, not a traceback
            self.w_prompt_meta.value = f"<span style='color:#b00020'>{_html.escape(type(e).__name__)}: {_html.escape(str(e))}</span>"

    # ------------------------------------------------------------------ compare

    def pin(self) -> View:
        """Remember the current view as the comparison *baseline* (side B). Change the
        view - add a slice, move to another day, refit - then :meth:`compare_with_baseline`
        (or the tab's button) puts the new view (side A) against it, cell by cell, on the
        baseline's layout."""
        self._baseline = self.view
        self.w_cmp_status.value = f"<span style='font-size:12px;color:#555'>baseline: {_html.escape(self.view.title())}</span>"
        return self._baseline

    def compare_with_baseline(self, *, metric: Optional[str] = None) -> Optional[Comparison]:
        """The current view (side A) against the pinned baseline (side B) - see :meth:`pin`.
        Renders into the Compare tab and returns the :class:`~pivot2hist.Comparison`."""
        if self._baseline is None:
            self.w_cmp_status.value = "<span style='color:#b00020;font-size:12px'>pin a baseline first</span>"
            return None
        if metric is not None:
            self._set_metric(metric)
        self._compare_recipe = {"kind": "baseline"}
        self._refresh_compare()
        return self.comparison if isinstance(self.comparison, Comparison) else None

    def compare(self, *args: Any, metric: Optional[str] = None, **kwargs: Any) -> Comparison:
        """:meth:`View.compare` on the current view, rendered into the Compare tab and kept
        in step with it - every later change to the view re-runs the same comparison.
        Same forms: ``explorer.compare(action="deny")``, ``explorer.compare("action",
        "deny", "allow")``, ``explorer.compare("bytes > 1000")``. Returns the
        :class:`~pivot2hist.Comparison`."""
        if metric is not None:
            self._set_metric(metric)
        self._compare_recipe = {"kind": "custom", "args": args, "kwargs": kwargs}
        self._refresh_compare()
        if not isinstance(self.comparison, Comparison):
            raise ValueError(self._compare_error or "comparison failed")
        return self.comparison

    def facet(self, column: str, n: int = 6) -> Facets:
        """:meth:`View.facet` on the current view, rendered into the Compare tab and kept in
        step with it. Returns the :class:`~pivot2hist.Facets`."""
        self._compare_recipe = {"kind": "facet", "column": column, "n": n}
        self._refresh_compare()
        if not isinstance(self.comparison, Facets):
            raise ValueError(self._compare_error or "faceting failed")
        return self.comparison

    def _set_metric(self, metric: str) -> None:
        if metric not in METRICS:
            raise ValueError(f"unknown metric {metric!r}; use one of {METRICS}")
        self._syncing = True
        try:
            self.w_cmp_metric.value = metric
        finally:
            self._syncing = False

    def _on_cmp_col(self, change: Dict[str, Any]) -> None:
        if self._syncing:
            return
        col = change["new"]
        self._syncing = True
        try:
            opts: List[Tuple[str, Any]] = [("(every value: one panel each)", None)]
            if col is not None:
                vc = self.view.data[col].value_counts(dropna=False).head(12)
                opts += [(f"{'(null)' if (isinstance(k, float) and np.isnan(k)) else k} ({int(c):,})",
                          None if (isinstance(k, float) and np.isnan(k)) else k) for k, c in vc.items()]
            self.w_cmp_val.options = opts
            self.w_cmp_val.value = None
            self.w_cmp_query.value = ""
        finally:
            self._syncing = False
        self._compare_recipe = None if col is None else {"kind": "facet", "column": col, "n": 6}
        self._refresh_compare()

    def _on_cmp_change(self, change: Dict[str, Any]) -> None:
        if self._syncing:
            return
        col = self.w_cmp_col.value
        if col is not None and change["owner"] is self.w_cmp_val:
            val = self.w_cmp_val.value
            self._compare_recipe = {"kind": "facet", "column": col, "n": 6} if val is None else {"kind": "split", "column": col, "value": val}
        self._refresh_compare()

    def _on_cmp_query(self, change: Dict[str, Any]) -> None:
        if self._syncing:
            return
        q = (change["new"] or "").strip()
        if q:
            self._syncing = True
            try:
                self.w_cmp_col.value = None
                self.w_cmp_val.options = [("(every value: one panel each)", None)]
                self.w_cmp_val.value = None
            finally:
                self._syncing = False
            self._compare_recipe = {"kind": "query", "query": q}
        elif self._compare_recipe and self._compare_recipe.get("kind") == "query":
            self._compare_recipe = None
        self._refresh_compare()

    def _clear_compare(self) -> None:
        self._syncing = True
        try:
            self.w_cmp_col.value = None
            self.w_cmp_val.options = [("(every value: one panel each)", None)]
            self.w_cmp_val.value = None
            self.w_cmp_query.value = ""
        finally:
            self._syncing = False
        self._compare_recipe = None
        self._refresh_compare()

    def _build_compare(self, recipe: Dict[str, Any]) -> Any:
        kind = recipe["kind"]
        if kind == "facet":
            return self.view.facet(recipe["column"], recipe.get("n", 6))
        if kind == "split":
            c = self.view.compare(**{recipe["column"]: recipe["value"]})
        elif kind == "query":
            c = self.view.compare(recipe["query"])
        elif kind == "baseline":
            c = self.view.compare(self._baseline, names=("current", "baseline"))
        else:
            c = self.view.compare(*recipe["args"], **recipe["kwargs"])
        want = self.w_cmp_metric.value
        try:
            return c.with_metric(want)
        except ValueError:
            # lift/shares need an additive measure; fall back rather than blank the tab
            self._set_metric("delta")
            self.w_cmp_status.value = (
                f"<span style='font-size:12px;color:#a15c00'>{_html.escape(want)} needs a count/sum measure; "
                f"showing delta for {_html.escape(c.layout.measure)}</span>"
            )
            return c.with_metric("delta")

    def _refresh_compare(self) -> None:
        recipe = self._compare_recipe
        self._compare_error: Optional[str] = None
        if recipe is None:
            self.comparison = None
            self.w_cmp_out.value = (
                "<i style='color:#999'>pick a column to split on (with a value: that value vs the rest; alone: "
                "one panel per value), type a query, or pin a baseline and compare the current view with it</i>"
            )
            return
        with log.step("explorer", f"compare: {recipe['kind']}"):
            try:
                obj = self._build_compare(recipe)
            except Exception as e:  # noqa: BLE001 - surface any failure in the tab, not a traceback
                self.comparison = None
                self._compare_error = f"{type(e).__name__}: {e}"
                self.w_cmp_out.value = f"<span style='color:#b00020'>{_html.escape(self._compare_error)}</span>"
                return
            self.comparison = obj
            if isinstance(obj, Comparison):
                self.w_cmp_out.value = obj.html(side_by_side=bool(self.w_cmp_side.value))
            else:
                self.w_cmp_out.value = obj.html()

    # ------------------------------------------------------------------ inspect

    def _picked_cell(self) -> Tuple[Any, ...]:
        row, col = self.w_cell_row.value, self.w_cell_col.value
        if row is None and col is None:
            raise ValueError("pick a row and/or a column in the Inspect tab first, or click a cell")
        return (row, col) if col is not None else (row,)

    def _show_cell_hint(self) -> None:
        self.w_cell_out.value = (
            "<i style='color:#999'>click a cell of the heatmap"
            + ("" if hasattr(self.w_out, "clicked") else " (needs anywidget)")
            + ", or pick a row / column above, then Explain</i>"
        )

    def explain(self, *args: Any, n_rows: Optional[int] = None, **labels: Any) -> Explanation:
        """:meth:`View.explain` on the current view, shown in the Inspect tab. With no cell
        given, explains the cell picked there (or the one last clicked). Returns the
        :class:`~pivot2hist.Explanation`."""
        if not args and not labels:
            args = self._picked_cell()
        n = self.w_cell_n.value if n_rows is None else int(n_rows)
        ex = self.view.explain(*args, n_rows=n, **labels)
        self.cell = ex
        self.w_cell_out.value = ex._repr_html_()
        return ex

    def rows(self, *args: Any, n: Optional[int] = None, **labels: Any) -> pd.DataFrame:
        """:meth:`View.rows` on the current view (the cell picked in the Inspect tab when
        none is given)."""
        if not args and not labels:
            args = self._picked_cell()
        return self.view.rows(*args, n=n, **labels)

    def drill_into(self, *args: Any, **labels: Any) -> None:
        """Slice to a cell by label and refit the layout on just those rows - the Inspect
        tab's **Drill in**. With no cell given, the picked / last-clicked one. Undo
        reverses it; the Code tab shows it as ``v.cell(...)``."""
        if not args and not labels:
            args = self._picked_cell()

        def go() -> None:
            given, extra = parse_cell(self.view, args, labels)
            filters = cell_filters(self.view, given, extra)
            payload = {**{c: str(v) for c, v in given.items()}, **extra}
            self.slices.append(("cell", (payload, filters)))
            self.refit_sliced = True

        self._act(go)

    def _explain_clicked(self) -> None:
        try:
            self.explain()
        except Exception as e:  # noqa: BLE001 - surface any failure in the tab, not a traceback
            self.w_cell_out.value = f"<span style='color:#b00020'>{_html.escape(type(e).__name__)}: {_html.escape(str(e))}</span>"

    def _on_cell_click(self, change: Dict[str, Any]) -> None:
        c = change["new"] or {}
        if c.get("slice"):
            self.remove_slice(str(c["slice"]))
            return
        row = tuple(str(x) for x in c.get("row") or ()) or None
        col = tuple(str(x) for x in c.get("col") or ()) or None
        if not self.view.layout.cols:
            col = None  # a 1-D table's only column is the measure, not a level
        self._syncing = True
        try:
            if row is not None and row in [v for _, v in self.w_cell_row.options]:
                self.w_cell_row.value = row
            if col is not None and col in [v for _, v in self.w_cell_col.options]:
                self.w_cell_col.value = col
        finally:
            self._syncing = False
        self.select_tab("Inspect")
        self._explain_clicked()

    def _refresh_cell_pickers(self) -> None:
        """Row / column pickers follow the current table's labels (keeping a still-valid pick)."""
        t = self.view.table()
        lay = self.view.layout

        def opts(index: pd.Index, hint: str) -> List[Tuple[str, Any]]:
            out: List[Tuple[str, Any]] = [(hint, None)]
            for x in list(index)[:200]:
                tup = tuple(str(y) for y in (x if isinstance(x, tuple) else (x,)))
                out.append((" / ".join(tup), tup))
            return out

        self._syncing = True
        try:
            row_prev, col_prev = self.w_cell_row.value, self.w_cell_col.value
            self.w_cell_row.options = opts(t.index, "(any row)")
            self.w_cell_col.options = opts(t.columns, "(any column)") if lay.cols else [("(any column)", None)]
            self.w_cell_row.value = row_prev if row_prev in [v for _, v in self.w_cell_row.options] else None
            self.w_cell_col.value = col_prev if col_prev in [v for _, v in self.w_cell_col.options] else None
        finally:
            self._syncing = False

    def _compare_code(self) -> List[str]:
        r = self._compare_recipe
        if not r:
            return []
        metric = self.w_cmp_metric.value
        m = f", metric={metric!r}" if metric != "lift" else ""
        if r["kind"] == "facet":
            return [f"f = v.facet({r['column']!r}{'' if r.get('n', 6) == 6 else ', ' + str(r['n'])})", "f"]
        if r["kind"] == "split":
            return [f"c = v.compare({r['column']}={r['value']!r}{m})", "c"]
        if r["kind"] == "query":
            return [f"c = v.compare({r['query']!r}{m})", "c"]
        if r["kind"] == "baseline":
            return ["# baseline = the view as it was when pinned", f"c = v.compare(baseline, names=('current', 'baseline'){m})", "c"]
        args = ", ".join(x for x in (", ".join(repr(a) for a in r["args"]), _kw(r["kwargs"])) if x)
        return [f"c = v.compare({args}{m})", "c"]

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
        self._widget_style = st

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

        # -- slicers tab: a fixed, capped grid of quick filters (avoids clutter on wide
        # tables). Any OTHER column still gets a widget the moment it's dragged into the
        # Fields tab's Slicers zone - see _make_slicer_widget / _refresh_field_slicers.
        self.w_slicers: Dict[str, Any] = {}
        boxes: List[Any] = []
        ranked = sorted([c for c in prof if c.kind in (CATEGORICAL, BOOLEAN)], key=lambda c: (c.position,))
        capped = ranked[: self._max_slicers] + [c for c in prof if c.kind == NUMERIC][:4] + [c for c in prof if c.kind == DATETIME][:2]
        for cp in capped:
            entry = self._make_slicer_widget(cp.name)
            if entry is not None:
                boxes.append(entry[1])
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
        self.w_fields = make_field_list(prof, theme=str(self.display.get("theme", "light")), rows=init_rows, cols=init_cols,
                                        values=init_values, slicers=list(self.field_slicers), marks=dict(self.marks))
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

        # -- timeline tab: save/note/replay checkpoints of the whole recipe
        self.w_checkpoint_note = W.Text(placeholder="what does this state show? (optional note)", description="Note",
                                        style=st, layout=W.Layout(width="420px"))
        self.w_checkpoint_save = W.Button(description="Save checkpoint", icon="bookmark", button_style="primary")
        self.w_checkpoint_delete = W.Button(description="Delete current", icon="trash")
        self.w_timeline_slider = W.IntSlider(min=0, max=0, value=0, description="Checkpoint", style=st, layout=W.Layout(width="420px"))
        self.w_timeline_play = W.Play(min=0, max=0, value=0, interval=1200)
        W.jslink((self.w_timeline_play, "value"), (self.w_timeline_slider, "value"))
        self.w_timeline_note = W.HTML()
        self.w_timeline_list = W.HTML()
        timeline_tab = W.VBox([
            W.HTML(
                "<i>Save a labeled checkpoint of the whole recipe (layout, slices, style, marks ...) at any "
                "point, then scrub or play back through them — each step re-renders the view as it was at that "
                "point, with the note you wrote for it.</i>"
            ),
            W.HBox([self.w_checkpoint_note, self.w_checkpoint_save, self.w_checkpoint_delete]),
            W.HBox([self.w_timeline_play, self.w_timeline_slider]),
            self.w_timeline_note,
            self.w_timeline_list,
        ])

        # -- insights tab: local, non-LLM analysis of the data currently in view
        self.w_sensitivity = W.FloatSlider(value=0.5, min=0.0, max=1.0, step=0.05, description="Sensitivity",
                                           style=st, layout=W.Layout(width="360px"))
        self.w_calculate_btn = W.Button(description="Calculate", icon="magic", button_style="primary")
        self.w_insights = W.HTML()
        self._insights_report: Optional[Dict[str, Any]] = None
        self._insights_view_id: Optional[int] = None
        insights_tab = W.VBox([
            W.HTML(
                "<i>A local, non-LLM read of whatever's currently in view (every slice already applied): "
                "column types, distributions, a mixture-model check for multiple populations in one numeric "
                "column, skew, concentration, outliers, correlated columns, and the most surprising pivot "
                "cells — ranked by how notable they are, not just listed. Click Calculate any time the data "
                "changes (it does not run itself, since it's real work on every slice).</i>"
            ),
            W.HBox([self.w_sensitivity, self.w_calculate_btn]),
            W.HTML(
                "<span style='font-size:11px;color:#889'>0 = only the strongest finding or two; 1 = "
                "everything that crosses any bar at all. Not called \"temperature\": this is a deterministic "
                "threshold on a computed significance score, not sampling from a model.</span>"
            ),
            self.w_insights,
        ])

        # -- compare tab: A vs B on one shared layout, or one panel per value
        self.w_cmp_col = W.Dropdown(options=[("(pick a column)", None)] + [(c, c) for c in labels], value=None,
                                    description="Split on", style=st)
        self.w_cmp_val = W.Dropdown(options=[("(every value: one panel each)", None)], value=None, description="Value",
                                    style=st, layout=W.Layout(width="300px"))
        self.w_cmp_query = W.Text(placeholder="or a query for side A, e.g. bytes > 1000 - side B is the rest  (Enter applies)",
                                  description="Query", style=st, layout=W.Layout(width="560px"), continuous_update=False)
        self.w_cmp_metric = W.Dropdown(options=[(f"{m}: {METRIC_HELP[m]}", m) for m in METRICS], value="lift",
                                       description="Metric", style=st, layout=W.Layout(width="520px"))
        self.w_cmp_side = W.Checkbox(value=False, description="show both sides too")
        self.w_cmp_pin = W.Button(description="Pin as baseline", icon="thumb-tack", tooltip="remember the current view as side B")
        self.w_cmp_vs_pin = W.Button(description="Compare with baseline", icon="exchange",
                                     tooltip="the current view (side A) against the pinned baseline (side B)")
        self.w_cmp_clear = W.Button(description="Clear", icon="eraser")
        self.w_cmp_status = W.HTML()
        self.w_cmp_out = W.HTML()
        self._baseline: Optional[View] = None
        self._compare_recipe: Optional[Dict[str, Any]] = None
        self.comparison: Optional[Any] = None
        compare_tab = W.VBox([
            W.HTML(
                "<i>Put two sides of the data on one shared layout and read them cell by cell. Pick a column "
                "and a value (that value vs the rest), a column alone (one panel per value, one colour scale), "
                "a query (its rows vs the rest), or pin the current view as a baseline, change it, and compare "
                "the new view against the pinned one - today vs yesterday, before vs after a slice. The layout "
                "is frozen across both sides so every cell means the same thing on each; blue = more on side A, "
                "red = less; hover a cell for both raw values.</i>"
            ),
            W.HBox([self.w_cmp_col, self.w_cmp_val]),
            W.HBox([self.w_cmp_query]),
            W.HBox([self.w_cmp_metric, self.w_cmp_side]),
            W.HBox([self.w_cmp_pin, self.w_cmp_vs_pin, self.w_cmp_clear]),
            self.w_cmp_status,
            self.w_cmp_out,
        ])

        # -- inspect tab: one cell, interrogated (click a cell, or pick a row and a column)
        self.w_cell_row = W.Dropdown(options=[("(any row)", None)], value=None, description="Row", style=st, layout=W.Layout(width="340px"))
        self.w_cell_col = W.Dropdown(options=[("(any column)", None)], value=None, description="Column", style=st, layout=W.Layout(width="280px"))
        self.w_cell_n = W.IntSlider(value=10, min=0, max=100, step=5, description="Rows shown", style=st)
        self.w_cell_explain = W.Button(description="Explain", icon="search", button_style="primary",
                                       tooltip="what's in this cell, and what sets its rows apart")
        self.w_cell_drill = W.Button(description="Drill in", icon="level-down", tooltip="slice to this cell and refit the layout on it")
        self.w_cell_out = W.HTML()
        self.cell: Optional[Explanation] = None
        inspect_tab = W.VBox([
            W.HTML(
                "<i>Click any cell of the heatmap (or bar of the histogram), or pick a row and a column here, "
                "then <b>Explain</b>: the cell's value against what independence of the axes would predict, its "
                "share of its row, column and the table, its rank, the rows behind it, and what sets those rows "
                "apart from the rest of the data in view. <b>Drill in</b> slices to the cell and refits, so the "
                "next layout is chosen for just those rows.</i>"
            ),
            W.HBox([self.w_cell_row, self.w_cell_col, self.w_cell_n]),
            W.HBox([self.w_cell_explain, self.w_cell_drill]),
            self.w_cell_out,
        ])

        # -- export tab: everything on screen as one prompt for any LLM
        self.w_prompt_question = W.Textarea(placeholder="your question (blank: " + DEFAULT_QUESTION[:60] + "...)",
                                            description="Question", style=st, layout=W.Layout(width="760px", height="64px"))
        self.w_prompt_table = W.Checkbox(value=True, description="the table (current view)")
        self.w_prompt_profile = W.Checkbox(value=True, description="dataset columns")
        self.w_prompt_anomalies = W.Checkbox(value=True, description="surprising cells")
        self.w_prompt_insights = W.Checkbox(value=False, description="insights (last Calculate, else computed now)")
        self.w_prompt_compare = W.Checkbox(value=True, description="the Compare tab's comparison, if any")
        self.w_prompt_cell = W.Checkbox(value=True, description="the Inspect tab's cell, if any")
        self.w_prompt_rows = W.IntSlider(value=30, min=5, max=100, step=5, description="Table rows", style=st)
        self.w_prompt_build = W.Button(description="Build prompt", icon="file-text-o", button_style="primary")
        self.w_prompt_copy = make_copy_button("Copy prompt")
        self.w_prompt_download = W.HTML()
        self.w_prompt_meta = W.HTML()
        self.w_prompt_out = W.Textarea(placeholder="click Build prompt", layout=W.Layout(width="100%", height="320px"))
        self.prompt_text: Optional[Prompt] = None
        export_tab = W.VBox([
            W.HTML(
                "<i>Everything on screen as one self-contained prompt for any LLM: the dataset's columns, the "
                "current table with its slices, the most surprising cells, optionally the Insights report, the "
                "Compare tab's comparison and the Inspect tab's cell, then your question. The prompt tells the "
                "model that every fact was computed locally and to reason only from them. Nothing here calls a "
                "model - copy it into whichever chat or agent you use.</i>"
            ),
            self.w_prompt_question,
            W.HBox([self.w_prompt_table, self.w_prompt_profile, self.w_prompt_anomalies]),
            W.HBox([self.w_prompt_insights, self.w_prompt_compare, self.w_prompt_cell]),
            W.HBox([self.w_prompt_rows, self.w_prompt_build, self.w_prompt_copy, self.w_prompt_download]),
            self.w_prompt_meta,
            self.w_prompt_out,
        ])

        # -- code, profile, data, stats tabs; output; log panel
        self.w_code = W.HTML()
        self.w_out = make_output()
        self.w_status = W.HTML()
        self.w_log = W.HTML()
        self.w_profile = W.HTML("<pre style='font-size:11px'>" + _html.escape(self._profile.summary().to_string(index=False)) + "</pre>")
        self.w_data = W.HTML()
        self.w_stats = W.HTML()
        self.w_stats_btn = W.Button(description="Refresh stats", icon="clock-o")
        stats_tab = W.VBox([W.HTML("<i>The seven costliest kinds of step so far (wall time, CPU time, peak memory delta).</i>"), self.w_stats_btn, self.w_stats])

        # -- tabs in four groups, the everyday loop (shape it, slice it, ask about it) up front
        groups: List[Tuple[str, List[Tuple[str, Any]]]] = [
            ("Explore", [("Fields", fields_tab), ("Layout", layout_tab), ("Slicers", slice_tab), ("Histogram", hist_tab)]),
            ("Analyze", [("Inspect", inspect_tab), ("Compare", compare_tab), ("Insights", insights_tab),
                         ("Reduce & cluster", reduce_tab), ("Chains", chains_tab)]),
            ("Output", [("Style", style_tab), ("Code", self.w_code), ("Export", export_tab)]),
            ("Session", [("Timeline", timeline_tab), ("Profile", self.w_profile), ("Data", self.w_data), ("Stats", stats_tab)]),
        ]
        self.tab_groups: Dict[str, Any] = {}
        self._tab_index: Dict[str, Tuple[int, int]] = {}
        outer: List[Any] = []
        for gi, (gname, items) in enumerate(groups):
            inner = W.Tab(children=[w for _, w in items])
            for i, (name, _) in enumerate(items):
                inner.set_title(i, name)
                self._tab_index[name] = (gi, i)
            self.tab_groups[gname] = inner
            outer.append(inner)
        self.tabs = W.Tab(children=outer)
        for gi, (gname, _) in enumerate(groups):
            self.tabs.set_title(gi, gname)
        self.FIELDS_TAB, self.TIMELINE_TAB, self.INSIGHTS_TAB, self.COMPARE_TAB, self.INSPECT_TAB = "Fields", "Timeline", "Insights", "Compare", "Inspect"
        # the top bar's controls in three labelled clusters, so the row reads as view / layout / history
        # rather than six unrelated buttons; the group tabs get a section look distinct from the inner tabs
        self.top = W.HBox([
            self._cluster("view", self.w_mode),
            self._cluster("layout", self.w_best, self.w_suggest_btn, self.w_suggest),
            self._cluster("history", self.w_undo, self.w_reset),
        ])
        self.top.add_class("p2h-topbar")
        self.tabs.add_class("p2h-groups")
        for inner in outer:
            inner.add_class("p2h-subtabs")
        self.w_css = W.HTML(self._chrome_css())
        self.box = W.VBox([self.w_css, self.top, self.tabs, self.w_status, self.w_out, W.HTML("<b style='font-size:11px;color:#666'>log</b>"), self.w_log])
        self._refresh_data_tab()
        self._refresh_log()
        self._refresh_timeline()

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
        # slicer widgets wire themselves up in _make_slicer_widget, whether built here
        # (the capped Slicers tab) or on demand later (a field dropped into Fields tab)
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
        self.w_checkpoint_save.on_click(lambda _: self.save_checkpoint(self.w_checkpoint_note.value))
        self.w_checkpoint_delete.on_click(lambda _: self._delete_checkpoint())
        self.w_timeline_slider.observe(self._on_timeline, names="value")
        self.w_calculate_btn.on_click(lambda _: self.calculate())
        self.w_cmp_col.observe(self._on_cmp_col, names="value")
        for w in (self.w_cmp_val, self.w_cmp_metric, self.w_cmp_side):
            w.observe(self._on_cmp_change, names="value")
        self.w_cmp_query.observe(self._on_cmp_query, names="value")  # continuous_update=False: fires on Enter/blur
        self.w_cmp_pin.on_click(lambda _: self.pin())
        self.w_cmp_vs_pin.on_click(lambda _: self.compare_with_baseline())
        self.w_cmp_clear.on_click(lambda _: self._clear_compare())
        self._refresh_compare()
        self.w_cell_explain.on_click(lambda _: self._explain_clicked())
        self.w_cell_drill.on_click(lambda _: self.drill_into())
        self.w_prompt_build.on_click(lambda _: self._build_prompt_clicked())
        if hasattr(self.w_out, "clicked"):
            self.w_out.observe(self._on_cell_click, names="clicked")
        self._show_cell_hint()

    # ------------------------------------------------------------------ recipe -> view

    def _snapshot(self) -> Dict[str, Any]:
        return {
            "spec": dict(self.spec), "slices": list(self.slices), "mode": self.mode, "hist": dict(self.hist),
            "reduce": dict(self.reduce), "display": dict(self.display), "options": self.options, "refit": self.refit_sliced,
            "chain": dict(self.chain), "field_slicers": list(self.field_slicers), "marks": dict(self.marks),
        }

    def _restore(self, snap: Dict[str, Any]) -> None:
        self.spec, self.slices, self.mode = dict(snap["spec"]), list(snap["slices"]), snap["mode"]
        self.hist, self.reduce, self.display = dict(snap["hist"]), dict(snap["reduce"]), dict(snap["display"])
        self.options, self.refit_sliced = snap["options"], snap["refit"]
        self.chain = dict(snap.get("chain", self.chain))
        self.field_slicers = list(snap.get("field_slicers", self.field_slicers))
        self.marks = dict(snap.get("marks", self.marks))
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
            self.w_fields.marks = dict(self.marks)
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

    @staticmethod
    def _apply_entry(v: View, kind: str, payload: Any) -> View:
        """One recipe slice entry applied to ``v`` (no refit)."""
        if kind == "slice":
            return v.slice(refit=False, **payload)
        if kind == "query":
            return v.slice(payload, refit=False)
        if kind == "top":
            return v.top(*payload)
        if kind == "filter":
            return v._clone(filters=v.filters + (payload,))
        if kind == "cell":
            return v._clone(filters=v.filters + tuple(payload[1]))
        return v

    def _recipe_labels(self) -> List[List[str]]:
        """The slice labels (as in ``view.slices``) each recipe entry contributes."""
        v = self._root()
        out: List[List[str]] = []
        for kind, payload in self.slices:
            before = len(v.filters)
            v = self._apply_entry(v, kind, payload)
            out.append([f.label for f in v.filters[before:]])
        return out

    def remove_slice(self, label: str) -> None:
        """Drop one active slice by its label (as shown on its chip / in ``view.slices``),
        whatever put it there - a slicer widget (which is reset), the query box, top-N, a
        drilled-in cell. The explorer's output pane calls this when a chip is clicked."""
        def go() -> None:
            hits = [i for i, labels in enumerate(self._recipe_labels()) if label in labels]
            if not hits:
                raise KeyError(f"no active slice {label!r}; have {self.view.slices}")
            for i in sorted(hits, reverse=True):
                kind, payload = self.slices.pop(i)
                self._reset_slice_widget(kind, payload)

        self._act(go)  # a bad label lands in the status line, like any other failed step

    def _reset_slice_widget(self, kind: str, payload: Any) -> None:
        self._syncing = True
        try:
            if kind == "slice":
                for col in payload:
                    entry = self.w_slicers.get(col)
                    if entry is None:
                        continue
                    wkind, w = entry
                    if wkind == "in":
                        w.value = ()
                    else:
                        w.index = (0, len(w.options) - 1)
            elif kind == "query":
                self.w_query.value = ""
            elif kind == "top":
                self.w_top_col.value = None
        finally:
            self._syncing = False

    def build(self) -> View:
        """The :class:`View` described by the current recipe."""
        v = self._root()
        for kind, payload in self.slices:
            v = self._apply_entry(v, kind, payload)
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
            elif kind == "cell":
                lines.append(f"v = v.cell({_kw(payload[0])})")
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
        lines.extend(self._compare_code())
        return "\n".join(lines)

    def _rebuild(self) -> None:
        with log.step("explorer", "rebuild") as st:
            self.view = self.build()
            # the title (with the slices as chips, removable when the pane can report clicks) lives in the pane
            self.w_out.value = self.view.html(title=True, removable_slices=hasattr(self.w_out, "clicked"))
            st.detail = f"rebuild -> {self.view.layout.describe()}"
        self.w_status.value = ""  # errors only; the pane carries the title and the slice chips
        self.w_code.value = "<pre style='font-size:12px'>" + _html.escape(self.code()) + "</pre>"
        self._sync_widgets()
        self._refresh_stats()
        self._refresh_field_slicers()
        self._render_insights()  # updates the "view changed since last Calculate" banner
        if getattr(self, "_compare_recipe", None):
            self._refresh_compare()  # the comparison is a lens on the view: keep it in step
        if hasattr(self, "w_cell_row"):
            self._refresh_cell_pickers()
        self._refresh_log()  # picks up a theme change immediately, not just on the next log line
        if hasattr(self, "w_css"):
            self.w_css.value = self._chrome_css()

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

    def _make_slicer_widget(self, name: str) -> Optional[Tuple[str, Any]]:
        """The (kind, widget) quick filter for ``name``, building and wiring it up the
        first time it's needed. Backs both the capped Slicers tab grid and any field
        dropped into the Fields tab's Slicers zone - see :meth:`_refresh_field_slicers`."""
        entry = self.w_slicers.get(name)
        if entry is not None:
            return entry
        if name not in self._profile:
            return None
        cp = self._profile[name]
        st = self._widget_style
        if cp.kind in (CATEGORICAL, BOOLEAN):
            vc = self._source[name].value_counts(dropna=False).head(30)
            opts = [(f"{'(null)' if (isinstance(k, float) and np.isnan(k)) else k} ({n:,})",
                     None if (isinstance(k, float) and np.isnan(k)) else k) for k, n in vc.items()]
            if not opts:
                return None
            w = W.SelectMultiple(options=opts, rows=min(6, len(opts)), description=name[:12], style=st, layout=W.Layout(width="260px"))
            entry = ("in", w)
        elif cp.kind == NUMERIC:
            col = pd.to_numeric(self._source[name], errors="coerce").to_numpy(dtype=float)
            col = col[np.isfinite(col)]
            if col.size == 0:
                return None
            qs = np.unique(np.percentile(col, np.linspace(0, 100, _NUM_STEPS + 1)))
            if qs.size < 2:
                return None
            opts = [(human(q), float(q)) for q in qs]
            w = W.SelectionRangeSlider(options=opts, index=(0, len(opts) - 1), description=name[:12], style=st,
                                        layout=W.Layout(width="360px"), continuous_update=False)
            entry = ("range", w)
        elif cp.kind == DATETIME:
            days = self._source[name].dropna().dt.floor("D").drop_duplicates().sort_values()
            if len(days) < 2:
                return None
            opts = [(d.strftime("%Y-%m-%d"), d) for d in days]
            w = W.SelectionRangeSlider(options=opts, index=(0, len(opts) - 1), description=name[:12], style=st,
                                        layout=W.Layout(width="360px"), continuous_update=False)
            entry = ("days", w)
        else:
            return None  # id / constant: no sensible quick filter
        entry[1].observe(self._on_slicers, names="value" if entry[0] == "in" else "index")
        self.w_slicers[name] = entry
        return entry

    def _refresh_field_slicers(self) -> None:
        rows = []
        for name in self.field_slicers:
            entry = self._make_slicer_widget(name)
            if entry is not None:
                rows.append(W.VBox([W.HTML(f"<b style='font-size:11px'>{_html.escape(name)}</b>"), entry[1]]))
            else:
                rows.append(W.HTML(f"<span style='font-size:11px;color:#889'>{_html.escape(name)}: no quick filter for this kind of column — use the Slicers tab (query or top-N)</span>"))
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
        out: List[Tuple[str, Any]] = [s for s in self.slices if s[0] in ("query", "filter", "cell")]
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

    def snapshot_html(self, *, active_tab: Optional[Union[int, str]] = None) -> str:
        """A static HTML picture of the interface (for docs, screenshots, sharing).

        Widgets are drawn as plain form elements with their current values; the output,
        status line and log are the real thing. ``active_tab`` brings a tab to the front
        first: a name (``"Inspect"``) or a group index.
        """
        if isinstance(active_tab, str):
            self.select_tab(active_tab)
        elif active_tab is not None:
            self.tabs.selected_index = int(active_tab)
        theme = str(self.display.get("theme", "light"))
        c = self._CHROME_THEME.get(theme, self._CHROME_THEME["light"])

        def render(w: Any) -> str:
            name = type(w).__name__
            if name in ("FieldList", "FieldListFallback"):
                return w.snapshot_html()
            classes = list(getattr(w, "_dom_classes", ()))
            if name in ("VBox", "HBox", "Box"):
                direction = "column" if name == "VBox" else "row"
                inner = "".join(render(x) for x in w.children)
                if "p2h-cluster" in classes:
                    return (
                        f"<div style='display:inline-flex;align-items:center;gap:4px;padding:3px 8px 3px 4px;border:1px solid {c['border']};"
                        f"border-radius:8px;background:{c['field']};margin:2px 0'>{inner}</div>"
                    )
                gap = "10px" if "p2h-topbar" in classes else "6px"
                return f"<div style='display:flex;flex-direction:{direction};flex-wrap:wrap;gap:{gap};align-items:flex-start;margin:2px 0'>{inner}</div>"
            if name == "Tab":
                tab = w.selected_index or 0
                section = "p2h-groups" in classes  # the outer, grouping tabs read as sections
                heads = "".join(
                    f"<span style='padding:5px 12px;border:1px solid {c['border']};border-bottom:{'none' if i == tab else '1px solid ' + c['border']};"
                    f"border-radius:8px 8px 0 0;background:{c['panel'] if i == tab else c['bg']};color:{c['text']};"
                    f"font-weight:{600 if (i == tab or section) else 400}"
                    + (";text-transform:uppercase;letter-spacing:.05em;font-size:11px" if section else "")
                    + (f";box-shadow:inset 0 -3px 0 {c['accent']}" if (section and i == tab) else "")
                    + f"'>{_html.escape(w.get_title(i) or str(i))}</span>"
                    for i in range(len(w.children))
                )
                body = render(w.children[tab]) if w.children else ""
                return (
                    f"<div><div style='display:flex;gap:2px'>{heads}</div>"
                    f"<div style='border:1px solid {c['border']};border-radius:0 8px 8px 8px;padding:10px;background:{c['panel']}'>{body}</div></div>"
                )
            if name in ("HTML", "ClickableHTML"):
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
