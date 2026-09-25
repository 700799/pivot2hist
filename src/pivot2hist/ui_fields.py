"""The fields pane: every column as a draggable chip with its stats, and the drop zones
(Rows, Columns, Values, Slicers) that define the pivot.

Two implementations share one Python interface (the ``rows`` / ``cols`` / ``values`` /
``slicers`` traits):

* :class:`FieldList` is an ``anywidget`` with real HTML5 drag-and-drop and reordering,
  rendered by a few hundred lines of plain JavaScript (no build step, works in Jupyter,
  Lab, VS Code and Colab);
* :class:`FieldListFallback` is plain ipywidgets: dropdowns to add, buttons to move and
  remove. Used when ``anywidget`` is not installed.

Chip stats: kind (and semantic type), non-null count, distinct count, and *variety*:
distinct values as a share of the rows (1.0 = every row different, like an id;
0.0001 = a handful of labels), shown as a small bar.
"""
from __future__ import annotations

import html as _html
from typing import Any, Callable, Dict, List, Optional, Sequence

import ipywidgets as W
import traitlets as T

from ._profile import Profile

KIND_COLORS = {"categorical": "#0072B2", "numeric": "#009E73", "datetime": "#E69F00", "boolean": "#CC79A7", "id": "#999999", "constant": "#bbbbbb"}
ZONES = ("rows", "cols", "values", "slicers")
ZONE_TITLES = {"rows": "Rows", "cols": "Columns", "values": "Values", "slicers": "Slicers"}


def field_stats(profile: Profile) -> List[Dict[str, Any]]:
    """One dict per column: name, kind, semantic, count, distinct, variety, nulls, examples."""
    out = []
    for c in profile:
        count = int(round(c.n * (1 - c.null_frac)))
        out.append(
            {
                "name": c.name,
                "kind": c.kind,
                "semantic": c.semantic or "",
                "count": count,
                "distinct": int(c.nunique),
                "variety": round(c.nunique / count, 4) if count else 0.0,
                "nulls": round(c.null_frac, 4),
                "examples": ", ".join(c.examples[:3]),
                "ts": c.ts_freq or "",
            }
        )
    return out


def _fmt_n(n: float) -> str:
    if n >= 1e6:
        return f"{n / 1e6:.3g}M"
    if n >= 1e3:
        return f"{n / 1e3:.3g}K"
    return f"{int(n)}"


def chip_html(f: Dict[str, Any], *, removable: bool = False, drag: bool = True) -> str:
    """Static HTML for one field chip (used by the fallback and by snapshots)."""
    color = KIND_COLORS.get(f["kind"], "#888")
    kind = f["kind"] + (f" · {f['semantic']}" if f.get("semantic") else "") + (f" · {f['ts']}" if f.get("ts") else "")
    pct = max(2, min(100, int(round(f["variety"] * 100)))) if f["distinct"] > 1 else 2
    title = _html.escape(f"{f['name']}: {kind}; {f['count']:,} non-null, {f['distinct']:,} distinct (variety {f['variety']:.2%}), nulls {f['nulls']:.1%}; e.g. {f.get('examples', '')}")
    x = "<span class='p2h-x' title='remove'>×</span>" if removable else ""
    return (
        f"<div class='p2h-chip' draggable='{'true' if drag else 'false'}' data-name='{_html.escape(f['name'])}' title='{title}'>"
        f"<span class='p2h-dot' style='background:{color}'></span>"
        f"<span class='p2h-name'>{_html.escape(f['name'])}</span>"
        f"<span class='p2h-kind'>{_html.escape(kind)}</span>"
        f"<span class='p2h-stat' title='non-null count'>n {_fmt_n(f['count'])}</span>"
        f"<span class='p2h-stat' title='distinct values'>≠ {_fmt_n(f['distinct'])}</span>"
        f"<span class='p2h-var' title='variety: distinct / rows'><span style='width:{pct}%'></span></span>{x}</div>"
    )


def zones_html(fields: List[Dict[str, Any]], zones: Dict[str, List[str]], *, interactive: bool = True) -> str:
    """Static HTML for the pane: the pool of unused fields plus the four zones."""
    by = {f["name"]: f for f in fields}
    used = {n for z in zones.values() for n in z}
    pool = "".join(chip_html(f, drag=interactive) for f in fields if f["name"] not in used)
    zone_blocks = []
    for z in ZONES:
        chips = "".join(chip_html(by[n], removable=interactive, drag=interactive) for n in zones.get(z, []) if n in by)
        hint = {"rows": "drag fields here: rows within rows", "cols": "columns within columns", "values": "measure (count when empty)", "slicers": "filter fields"}[z]
        body = chips if chips else f"<span class='p2h-hint'>{hint}</span>"
        zone_blocks.append(
            f"<div class='p2h-zone' data-zone='{z}'><div class='p2h-zone-title'>{ZONE_TITLES[z]}</div>"
            f"<div class='p2h-zone-body'>{body}</div></div>"
        )
    return (
        "<div class='p2h-fields'>"
        "<div class='p2h-pool'><div class='p2h-zone-title'>Fields <input class='p2h-search' placeholder='search'/></div>"
        f"<div class='p2h-pool-body'>{pool}</div></div>"
        f"<div class='p2h-zones'>{''.join(zone_blocks)}</div></div>"
    )


FIELDS_CSS = """
.p2h-fields{display:flex;gap:10px;align-items:stretch;font-size:12px}
.p2h-pool{flex:0 0 250px;display:flex;flex-direction:column;min-height:120px}
.p2h-pool-body{overflow-y:auto;max-height:330px;display:flex;flex-direction:column;gap:4px;padding:6px;border:1px dashed var(--p2h-border,#cfd6df);border-radius:8px;background:var(--p2h-panel,#fbfcfd)}
.p2h-zones{flex:1;display:grid;grid-template-columns:1fr 1fr;gap:8px}
.p2h-zone{display:flex;flex-direction:column;min-height:60px}
.p2h-zone-title{font-weight:600;color:var(--p2h-muted,#556);margin:0 0 4px 2px;display:flex;justify-content:space-between;align-items:center}
.p2h-zone-body{flex:1;display:flex;flex-direction:column;gap:4px;padding:6px;border:1px dashed var(--p2h-border,#cfd6df);border-radius:8px;background:var(--p2h-panel,#fbfcfd);min-height:40px}
.p2h-zone-body.p2h-over,.p2h-pool-body.p2h-over{border-color:var(--p2h-accent,#0072B2);background:var(--p2h-accent-soft,#e8f1fa)}
.p2h-hint{color:#99a;font-style:italic;padding:2px 4px}
.p2h-chip{display:grid;grid-template-columns:8px minmax(70px,1.4fr) minmax(60px,1fr) auto auto 46px auto;gap:6px;align-items:center;padding:4px 8px;border-radius:6px;background:var(--p2h-chip,#fff);border:1px solid var(--p2h-border,#dfe4ea);cursor:grab;box-shadow:0 1px 1px rgba(0,0,0,.04)}
.p2h-chip.p2h-dragging{opacity:.5}
.p2h-dot{width:8px;height:8px;border-radius:50%}
.p2h-name{font-weight:600;color:var(--p2h-text,#1f2937);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.p2h-kind{color:var(--p2h-muted,#667);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-size:11px}
.p2h-stat{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11px;color:var(--p2h-muted,#556);white-space:nowrap}
.p2h-var{display:inline-block;height:6px;background:#e5eaf0;border-radius:3px;overflow:hidden}
.p2h-var span{display:block;height:6px;background:linear-gradient(90deg,#56B4E9,#0072B2)}
.p2h-x{cursor:pointer;color:#99a;font-weight:700;padding:0 2px}.p2h-x:hover{color:#c00}
.p2h-search{font-size:11px;padding:2px 6px;border:1px solid var(--p2h-border,#dfe4ea);border-radius:6px;width:110px}
"""

_ESM = r"""
function render({ model, el }) {
  const ZONES = ["rows", "cols", "values", "slicers"];
  const TITLES = { rows: "Rows", cols: "Columns", values: "Values", slicers: "Slicers" };
  const HINTS = { rows: "drag fields here: rows within rows", cols: "columns within columns", values: "measure (count when empty)", slicers: "filter fields" };
  const COLORS = { categorical: "#0072B2", numeric: "#009E73", datetime: "#E69F00", boolean: "#CC79A7", id: "#999999", constant: "#bbbbbb" };
  let search = "";
  const fmt = (n) => (n >= 1e6 ? (n / 1e6).toPrecision(3) + "M" : n >= 1e3 ? (n / 1e3).toPrecision(3) + "K" : String(n));
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  function chip(f, removable) {
    const d = document.createElement("div");
    d.className = "p2h-chip"; d.draggable = true; d.dataset.name = f.name;
    const kind = f.kind + (f.semantic ? " · " + f.semantic : "") + (f.ts ? " · " + f.ts : "");
    const pct = f.distinct > 1 ? Math.max(2, Math.min(100, Math.round(f.variety * 100))) : 2;
    d.title = `${f.name}: ${kind}; ${f.count.toLocaleString()} non-null, ${f.distinct.toLocaleString()} distinct (variety ${(f.variety * 100).toFixed(2)}%), nulls ${(f.nulls * 100).toFixed(1)}%; e.g. ${f.examples || ""}`;
    d.innerHTML = `<span class='p2h-dot' style='background:${COLORS[f.kind] || "#888"}'></span><span class='p2h-name'>${esc(f.name)}</span><span class='p2h-kind'>${esc(kind)}</span><span class='p2h-stat'>n ${fmt(f.count)}</span><span class='p2h-stat'>≠ ${fmt(f.distinct)}</span><span class='p2h-var'><span style='width:${pct}%'></span></span>${removable ? "<span class='p2h-x' title='remove'>×</span>" : ""}`;
    d.addEventListener("dragstart", (e) => { e.dataTransfer.setData("text/plain", f.name); e.dataTransfer.effectAllowed = "move"; d.classList.add("p2h-dragging"); });
    d.addEventListener("dragend", () => d.classList.remove("p2h-dragging"));
    if (removable) d.querySelector(".p2h-x").addEventListener("click", () => { removeEverywhere(f.name); commit(); });
    return d;
  }
  function state() { const s = {}; for (const z of ZONES) s[z] = [...(model.get(z) || [])]; return s; }
  function removeEverywhere(name) { for (const z of ZONES) { const arr = (model.get(z) || []).filter((n) => n !== name); model.set(z, arr); } }
  function commit() { model.save_changes(); draw(); }
  function dropInto(zone, name, beforeName) {
    removeEverywhere(name);
    if (zone === "pool") { return; }
    const arr = [...(model.get(zone) || [])];
    if (zone === "values") arr.length = 0;  // one measure at a time
    const idx = beforeName ? arr.indexOf(beforeName) : -1;
    if (idx >= 0) arr.splice(idx, 0, name); else arr.push(name);
    model.set(zone, arr);
  }
  function wireZone(body, zone) {
    body.addEventListener("dragover", (e) => { e.preventDefault(); body.classList.add("p2h-over"); e.dataTransfer.dropEffect = "move"; });
    body.addEventListener("dragleave", () => body.classList.remove("p2h-over"));
    body.addEventListener("drop", (e) => {
      e.preventDefault(); body.classList.remove("p2h-over");
      const name = e.dataTransfer.getData("text/plain"); if (!name) return;
      const target = e.target.closest(".p2h-chip");
      const before = target && target.dataset.name !== name ? target.dataset.name : null;
      dropInto(zone, name, before); commit();
    });
  }
  function draw() {
    const fields = model.get("fields") || [];
    const by = Object.fromEntries(fields.map((f) => [f.name, f]));
    const s = state();
    const used = new Set(ZONES.flatMap((z) => s[z]));
    el.innerHTML = "";
    const root = document.createElement("div"); root.className = "p2h-fields";
    const pool = document.createElement("div"); pool.className = "p2h-pool";
    pool.innerHTML = `<div class='p2h-zone-title'>Fields <input class='p2h-search' placeholder='search' value='${esc(search)}'/></div>`;
    const poolBody = document.createElement("div"); poolBody.className = "p2h-pool-body";
    for (const f of fields) if (!used.has(f.name) && f.name.toLowerCase().includes(search.toLowerCase())) poolBody.appendChild(chip(f, false));
    wireZone(poolBody, "pool"); pool.appendChild(poolBody); root.appendChild(pool);
    pool.querySelector(".p2h-search").addEventListener("input", (e) => { search = e.target.value; draw(); pool.querySelector(".p2h-search").focus(); });
    const zones = document.createElement("div"); zones.className = "p2h-zones";
    for (const z of ZONES) {
      const zone = document.createElement("div"); zone.className = "p2h-zone"; zone.dataset.zone = z;
      zone.innerHTML = `<div class='p2h-zone-title'>${TITLES[z]}</div>`;
      const body = document.createElement("div"); body.className = "p2h-zone-body";
      if (!s[z].length) body.innerHTML = `<span class='p2h-hint'>${HINTS[z]}</span>`;
      for (const n of s[z]) if (by[n]) body.appendChild(chip(by[n], true));
      wireZone(body, z); zone.appendChild(body); zones.appendChild(zone);
    }
    root.appendChild(zones); el.appendChild(root);
  }
  for (const k of ["fields", ...ZONES]) model.on(`change:${k}`, draw);
  draw();
}
export default { render };
"""


class _FieldTraits(T.HasTraits):
    fields = T.List(T.Dict()).tag(sync=True)
    rows = T.List(T.Unicode()).tag(sync=True)
    cols = T.List(T.Unicode()).tag(sync=True)
    values = T.List(T.Unicode()).tag(sync=True)
    slicers = T.List(T.Unicode()).tag(sync=True)


try:
    import anywidget

    class FieldList(anywidget.AnyWidget):  # type: ignore[misc]
        """Drag-and-drop fields pane. Observe ``rows`` / ``cols`` / ``values`` / ``slicers``."""

        _esm = _ESM
        _css = FIELDS_CSS
        fields = T.List(T.Dict()).tag(sync=True)
        rows = T.List(T.Unicode()).tag(sync=True)
        cols = T.List(T.Unicode()).tag(sync=True)
        values = T.List(T.Unicode()).tag(sync=True)
        slicers = T.List(T.Unicode()).tag(sync=True)

        def set_zones(self, **zones: Sequence[str]) -> None:
            """Set several zones at once without firing a change per zone."""
            with self.hold_trait_notifications():
                for k, v in zones.items():
                    setattr(self, k, list(v))

        def snapshot_html(self) -> str:
            return zones_html(self.fields, {z: list(getattr(self, z)) for z in ZONES})

    HAS_ANYWIDGET = True
except ImportError:  # pragma: no cover - exercised only without anywidget
    HAS_ANYWIDGET = False
    FieldList = None  # type: ignore[assignment,misc]


class FieldListFallback(W.VBox):
    """The same pane with plain ipywidgets: add via dropdown, reorder with buttons.

    Exposes the same ``rows`` / ``cols`` / ``values`` / ``slicers`` traits as
    :class:`FieldList` so the explorer does not care which one it got.
    """

    fields = T.List(T.Dict())
    rows = T.List(T.Unicode())
    cols = T.List(T.Unicode())
    values = T.List(T.Unicode())
    slicers = T.List(T.Unicode())

    def __init__(self, fields: List[Dict[str, Any]], **zones: Sequence[str]):
        super().__init__()
        self._by = {f["name"]: f for f in fields}
        self._syncing = False
        self.fields = list(fields)
        self._boxes: Dict[str, W.VBox] = {}
        self._adds: Dict[str, W.Dropdown] = {}
        panes = []
        for z in ZONES:
            add = W.Dropdown(options=[("add …", None)] + [(self._label(f), f["name"]) for f in fields], value=None, layout=W.Layout(width="260px"))
            add.observe(self._on_add(z), names="value")
            box = W.VBox()
            self._boxes[z], self._adds[z] = box, add
            panes.append(W.VBox([W.HTML(f"<b>{ZONE_TITLES[z]}</b>"), add, box], layout=W.Layout(border="1px dashed #cfd6df", padding="4px", margin="2px", min_width="280px")))
        self.children = [W.HTML("<i>anywidget not installed: pick fields from the dropdowns, reorder with the arrows.</i>"), W.HBox(panes[:2]), W.HBox(panes[2:])]
        with self.hold_trait_notifications():
            for z, v in zones.items():
                setattr(self, z, list(v))
        self.observe(lambda _: self._redraw(), names=list(ZONES))
        self._redraw()

    @staticmethod
    def _label(f: Dict[str, Any]) -> str:
        return f"{f['name']}  [{f['kind']}{' ' + f['semantic'] if f.get('semantic') else ''}]  n={_fmt_n(f['count'])} ≠{_fmt_n(f['distinct'])} var={f['variety']:.0%}"

    def _on_add(self, zone: str) -> Callable[[Any], None]:
        def handler(change: Any) -> None:
            name = change["new"]
            if not name or self._syncing:
                return
            self._syncing = True
            try:
                self._adds[zone].value = None
            finally:
                self._syncing = False
            self.move(name, zone)
        return handler

    def move(self, name: str, zone: Optional[str], *, before: Optional[str] = None) -> None:
        """Put ``name`` into ``zone`` (``None`` removes it from every zone)."""
        with self.hold_trait_notifications():
            for z in ZONES:
                if name in getattr(self, z):
                    setattr(self, z, [n for n in getattr(self, z) if n != name])
            if zone:
                arr = [] if zone == "values" else list(getattr(self, zone))
                idx = arr.index(before) if before in arr else len(arr)
                arr.insert(idx, name)
                setattr(self, zone, arr)

    def _shift(self, zone: str, name: str, delta: int) -> None:
        arr = list(getattr(self, zone))
        i = arr.index(name)
        j = max(0, min(len(arr) - 1, i + delta))
        arr.insert(j, arr.pop(i))
        setattr(self, zone, arr)

    def _redraw(self) -> None:
        for z in ZONES:
            rows = []
            for name in getattr(self, z):
                f = self._by.get(name)
                if f is None:
                    continue
                up = W.Button(icon="arrow-up", layout=W.Layout(width="30px"))
                down = W.Button(icon="arrow-down", layout=W.Layout(width="30px"))
                rm = W.Button(icon="times", layout=W.Layout(width="30px"))
                up.on_click(lambda _, z=z, n=name: self._shift(z, n, -1))
                down.on_click(lambda _, z=z, n=name: self._shift(z, n, +1))
                rm.on_click(lambda _, n=name: self.move(n, None))
                rows.append(W.HBox([W.HTML(f"<style>{FIELDS_CSS}</style>" + chip_html(f, drag=False)), up, down, rm]))
            self._boxes[z].children = rows or [W.HTML("<span style='color:#99a;font-style:italic'>empty</span>")]

    def set_zones(self, **zones: Sequence[str]) -> None:
        with self.hold_trait_notifications():
            for k, v in zones.items():
                setattr(self, k, list(v))

    def snapshot_html(self) -> str:
        return zones_html(self.fields, {z: list(getattr(self, z)) for z in ZONES}, interactive=False)


def make_field_list(profile: Profile, *, prefer_anywidget: bool = True, **zones: Sequence[str]):
    """The best available fields pane for ``profile``."""
    fields = field_stats(profile)
    if prefer_anywidget and HAS_ANYWIDGET and FieldList is not None:
        w = FieldList(fields=fields)
        w.set_zones(**zones)
        return w
    return FieldListFallback(fields, **zones)


__all__ = ["FieldList", "FieldListFallback", "make_field_list", "field_stats", "chip_html", "zones_html", "FIELDS_CSS", "ZONES", "HAS_ANYWIDGET"]
