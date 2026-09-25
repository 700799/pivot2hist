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

#: Named highlight colors for :func:`chip_html`'s ``mark``/the ``marks`` trait. Any CSS
#: color string works too (``"#123abc"``, ``"tomato"``) - these are just a convenient,
#: discoverable palette; clicking a chip's dot in the anywidget UI cycles through them.
MARK_PALETTE: Dict[str, str] = {
    "red": "#e5484d", "orange": "#f76b15", "yellow": "#f5c518",
    "green": "#30a46c", "blue": "#3b82f6", "purple": "#8b5cf6",
}


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


def _set_mark(marks: Dict[str, str], name: str, color: Optional[str]) -> Dict[str, str]:
    """A new ``marks`` dict with ``name`` painted ``color`` (a :data:`MARK_PALETTE` name
    or any CSS color string), or cleared when ``color`` is ``None``/empty."""
    out = dict(marks)
    resolved = MARK_PALETTE.get(color, color) if color else None
    if resolved:
        out[name] = resolved
    else:
        out.pop(name, None)
    return out


def _fmt_n(n: float) -> str:
    if n >= 1e6:
        return f"{n / 1e6:.3g}M"
    if n >= 1e3:
        return f"{n / 1e3:.3g}K"
    return f"{int(n)}"


def chip_html(f: Dict[str, Any], *, removable: bool = False, drag: bool = True, mark: Optional[str] = None) -> str:
    """Static HTML for one field chip (used by the fallback and by snapshots).

    ``mark`` is a CSS color (see :data:`MARK_PALETTE` for named ones) painted as a left
    border and a faint background tint, so a highlighted field stays visible even in a
    plain HTML snapshot with no interactivity.
    """
    color = KIND_COLORS.get(f["kind"], "#888")
    kind = f["kind"] + (f" · {f['semantic']}" if f.get("semantic") else "") + (f" · {f['ts']}" if f.get("ts") else "")
    pct = max(2, min(100, int(round(f["variety"] * 100)))) if f["distinct"] > 1 else 2
    title = _html.escape(f"{f['name']}: {kind}; {f['count']:,} non-null, {f['distinct']:,} distinct (variety {f['variety']:.2%}), nulls {f['nulls']:.1%}; e.g. {f.get('examples', '')}")
    if mark:
        title += _html.escape(f" — marked {mark}")
    x = "<span class='p2h-x' title='remove'>×</span>" if removable else ""
    mark_style = f"border-left:4px solid {_html.escape(mark)};box-shadow:0 0 0 1px {_html.escape(mark)} inset" if mark else ""
    return (
        f"<div class='p2h-chip' draggable='{'true' if drag else 'false'}' data-name='{_html.escape(f['name'])}' title='{title}' style='{mark_style}'>"
        f"<span class='p2h-dot' style='background:{color}'></span>"
        f"<span class='p2h-name'>{_html.escape(f['name'])}</span>"
        f"<span class='p2h-kind'>{_html.escape(kind)}</span>"
        f"<span class='p2h-stat' title='non-null count'>n {_fmt_n(f['count'])}</span>"
        f"<span class='p2h-stat' title='distinct values'>≠ {_fmt_n(f['distinct'])}</span>"
        f"<span class='p2h-var' title='variety: distinct / rows'><span style='width:{pct}%'></span></span>{x}</div>"
    )


def zones_html(fields: List[Dict[str, Any]], zones: Dict[str, List[str]], *, interactive: bool = True,
                theme: str = "light", marks: Optional[Dict[str, str]] = None) -> str:
    """Static HTML for the pane: the pool of unused fields plus the four zones."""
    marks = marks or {}
    by = {f["name"]: f for f in fields}
    used = {n for z in zones.values() for n in z}
    pool = "".join(chip_html(f, drag=interactive, mark=marks.get(f["name"])) for f in fields if f["name"] not in used)
    zone_blocks = []
    for z in ZONES:
        chips = "".join(chip_html(by[n], removable=interactive, drag=interactive, mark=marks.get(n)) for n in zones.get(z, []) if n in by)
        hint = {"rows": "drag fields here: rows within rows", "cols": "columns within columns", "values": "measure (count when empty)", "slicers": "filter fields"}[z]
        body = chips if chips else f"<span class='p2h-hint'>{hint}</span>"
        zone_blocks.append(
            f"<div class='p2h-zone' data-zone='{z}'><div class='p2h-zone-title'>{ZONE_TITLES[z]}</div>"
            f"<div class='p2h-zone-body'>{body}</div></div>"
        )
    css = theme_style_block(theme, selector=".p2h-fields-scope") + f"<style>{FIELDS_CSS}</style>"
    return (
        f"<div class='p2h-fields-scope' style='background:var(--p2h-bg,#fff);padding:8px;border-radius:12px'>{css}"
        "<div class='p2h-fields'>"
        "<div class='p2h-pool'><div class='p2h-zone-title'>Fields <input class='p2h-search' placeholder='search'/></div>"
        f"<div class='p2h-pool-body'>{pool}</div></div>"
        f"<div class='p2h-zones'>{''.join(zone_blocks)}</div></div></div>"
    )


#: CSS custom properties per theme; :func:`theme_style_block` emits one as a scoped
#: ``<style>`` block so the field list (and anything else using the same ``--p2h-*``
#: tokens) can switch between a light and a graphite look without editing markup.
FIELD_THEMES = {
    "light": {
        "bg": "#ffffff", "panel": "#fbfcfd", "border": "#dfe4ea", "text": "#1f2937",
        "muted": "#667", "accent": "#0072B2", "accent-soft": "#e8f1fa", "chip": "#ffffff",
        "var-track": "#e5eaf0", "danger": "#c00", "hint": "#99a",
    },
    "graphite": {
        "bg": "#161a20", "panel": "#1b2128", "border": "#333a45", "text": "#e5e9ef",
        "muted": "#98a1b0", "accent": "#5b9be0", "accent-soft": "#22384d", "chip": "#1e242c",
        "var-track": "#2a313b", "danger": "#e8785a", "hint": "#7a8494",
    },
}


def theme_style_block(theme: str = "light", *, selector: str = ":root") -> str:
    """A ``<style>`` block defining the ``--p2h-*`` custom properties for ``theme``."""
    vars_ = FIELD_THEMES.get(theme, FIELD_THEMES["light"])
    decls = ";".join(f"--p2h-{k}:{v}" for k, v in vars_.items())
    return f"<style>{selector}{{{decls}}}</style>"


FIELDS_CSS = """
.p2h-fields{display:flex;gap:10px;align-items:stretch;font-size:12px}
.p2h-pool{flex:0 0 250px;display:flex;flex-direction:column;min-height:120px}
.p2h-pool-body{overflow-y:auto;max-height:330px;display:flex;flex-direction:column;gap:4px;padding:6px;border:1px dashed var(--p2h-border,#cfd6df);border-radius:10px;background:var(--p2h-panel,#fbfcfd)}
.p2h-zones{flex:1;display:grid;grid-template-columns:1fr 1fr;gap:8px}
.p2h-zone{display:flex;flex-direction:column;min-height:60px}
.p2h-zone-title{font-weight:600;color:var(--p2h-muted,#556);margin:0 0 4px 2px;display:flex;justify-content:space-between;align-items:center}
.p2h-zone-body{flex:1;display:flex;flex-direction:column;gap:4px;padding:6px;border:1px dashed var(--p2h-border,#cfd6df);border-radius:10px;background:var(--p2h-panel,#fbfcfd);min-height:40px;transition:border-color .12s,background .12s}
.p2h-zone-body.p2h-over,.p2h-pool-body.p2h-over{border-color:var(--p2h-accent,#0072B2);background:var(--p2h-accent-soft,#e8f1fa)}
.p2h-hint{color:var(--p2h-hint,#99a);font-style:italic;padding:2px 4px}
.p2h-chip{display:grid;grid-template-columns:8px minmax(70px,1.4fr) minmax(60px,1fr) auto auto 46px auto;gap:6px;align-items:center;padding:4px 8px;border-radius:8px;background:var(--p2h-chip,#fff);color:var(--p2h-text,#1f2937);border:1px solid var(--p2h-border,#dfe4ea);cursor:grab;box-shadow:0 1px 2px rgba(0,0,0,.06);transition:box-shadow .12s,transform .12s}
.p2h-chip:hover{box-shadow:0 2px 6px rgba(0,0,0,.12)}
.p2h-chip.p2h-dragging{opacity:.5}
.p2h-dot{width:8px;height:8px;border-radius:50%}
.p2h-name{font-weight:600;color:var(--p2h-text,#1f2937);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.p2h-kind{color:var(--p2h-muted,#667);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-size:11px}
.p2h-stat{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11px;color:var(--p2h-muted,#556);white-space:nowrap}
.p2h-var{display:inline-block;height:6px;background:var(--p2h-var-track,#e5eaf0);border-radius:3px;overflow:hidden}
.p2h-var span{display:block;height:6px;background:linear-gradient(90deg,#56B4E9,var(--p2h-accent,#0072B2))}
.p2h-x{cursor:pointer;color:var(--p2h-hint,#99a);font-weight:700;padding:0 2px}.p2h-x:hover{color:var(--p2h-danger,#c00)}
.p2h-search{font-size:11px;padding:3px 8px;border:1px solid var(--p2h-border,#dfe4ea);border-radius:8px;width:110px;background:var(--p2h-bg,#fff);color:var(--p2h-text,#1f2937)}
"""

_ESM = r"""
const THEME_VARS = {
  light: { bg: "#ffffff", panel: "#fbfcfd", border: "#dfe4ea", text: "#1f2937", muted: "#667", accent: "#0072B2", "accent-soft": "#e8f1fa", chip: "#ffffff", "var-track": "#e5eaf0", danger: "#c00", hint: "#99a" },
  graphite: { bg: "#161a20", panel: "#1b2128", border: "#333a45", text: "#e5e9ef", muted: "#98a1b0", accent: "#5b9be0", "accent-soft": "#22384d", chip: "#1e242c", "var-track": "#2a313b", danger: "#e8785a", hint: "#7a8494" },
};
function applyTheme(el, theme) {
  const vars = THEME_VARS[theme] || THEME_VARS.light;
  for (const [k, v] of Object.entries(vars)) el.style.setProperty(`--p2h-${k}`, v);
  el.style.background = "var(--p2h-bg)";
  el.style.borderRadius = "12px";
  el.style.padding = "8px";
}
function render({ model, el }) {
  const ZONES = ["rows", "cols", "values", "slicers"];
  const TITLES = { rows: "Rows", cols: "Columns", values: "Values", slicers: "Slicers" };
  const HINTS = { rows: "drag fields here: rows within rows", cols: "columns within columns", values: "measure (count when empty)", slicers: "filter fields" };
  const COLORS = { categorical: "#0072B2", numeric: "#009E73", datetime: "#E69F00", boolean: "#CC79A7", id: "#999999", constant: "#bbbbbb" };
  const MARK_CYCLE = [null, "#e5484d", "#f76b15", "#f5c518", "#30a46c", "#3b82f6", "#8b5cf6"];
  let search = "";
  const fmt = (n) => (n >= 1e6 ? (n / 1e6).toPrecision(3) + "M" : n >= 1e3 ? (n / 1e3).toPrecision(3) + "K" : String(n));
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  function cycleMark(name) {
    const marks = { ...(model.get("marks") || {}) };
    const cur = marks[name] || null;
    const next = MARK_CYCLE[(MARK_CYCLE.indexOf(cur) + 1) % MARK_CYCLE.length];
    if (next) marks[name] = next; else delete marks[name];
    model.set("marks", marks); model.save_changes(); draw();
  }
  function chip(f, removable) {
    const d = document.createElement("div");
    d.className = "p2h-chip"; d.draggable = true; d.dataset.name = f.name;
    const kind = f.kind + (f.semantic ? " · " + f.semantic : "") + (f.ts ? " · " + f.ts : "");
    const pct = f.distinct > 1 ? Math.max(2, Math.min(100, Math.round(f.variety * 100))) : 2;
    const mark = (model.get("marks") || {})[f.name];
    if (mark) { d.style.borderLeft = `4px solid ${mark}`; d.style.boxShadow = `0 0 0 1px ${mark} inset`; }
    d.title = `${f.name}: ${kind}; ${f.count.toLocaleString()} non-null, ${f.distinct.toLocaleString()} distinct (variety ${(f.variety * 100).toFixed(2)}%), nulls ${(f.nulls * 100).toFixed(1)}%; e.g. ${f.examples || ""}` + (mark ? ` — marked (click the dot to cycle/clear)` : ` — click the dot to highlight`);
    d.innerHTML = `<span class='p2h-dot' style='background:${COLORS[f.kind] || "#888"};cursor:pointer' title='click to highlight'></span><span class='p2h-name'>${esc(f.name)}</span><span class='p2h-kind'>${esc(kind)}</span><span class='p2h-stat'>n ${fmt(f.count)}</span><span class='p2h-stat'>≠ ${fmt(f.distinct)}</span><span class='p2h-var'><span style='width:${pct}%'></span></span>${removable ? "<span class='p2h-x' title='remove'>×</span>" : ""}`;
    d.addEventListener("dragstart", (e) => { e.dataTransfer.setData("text/plain", f.name); e.dataTransfer.effectAllowed = "move"; d.classList.add("p2h-dragging"); });
    d.addEventListener("dragend", () => d.classList.remove("p2h-dragging"));
    d.querySelector(".p2h-dot").addEventListener("click", (e) => { e.stopPropagation(); cycleMark(f.name); });
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
    applyTheme(el, model.get("theme") || "light");
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
  for (const k of ["fields", ...ZONES, "theme", "marks"]) model.on(`change:${k}`, draw);
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
        theme = T.Unicode("light").tag(sync=True)
        marks = T.Dict().tag(sync=True)  # column name -> CSS color; clicking a chip's dot also cycles this

        def set_zones(self, **zones: Sequence[str]) -> None:
            """Set several zones at once without firing a change per zone."""
            with self.hold_trait_notifications():
                for k, v in zones.items():
                    setattr(self, k, list(v))

        def paint(self, name: str, color: Optional[str]) -> None:
            """Highlight field ``name`` with ``color`` (a name from :data:`MARK_PALETTE`
            or any CSS color); ``color=None`` clears its mark. No-op for an unknown field."""
            self.marks = _set_mark(self.marks, name, color)

        def unmark(self, name: Optional[str] = None) -> None:
            """Clear one field's mark, or every mark when ``name`` is ``None``."""
            self.marks = {} if name is None else {k: v for k, v in self.marks.items() if k != name}

        def snapshot_html(self) -> str:
            return zones_html(self.fields, {z: list(getattr(self, z)) for z in ZONES}, theme=self.theme, marks=self.marks)

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
    marks = T.Dict()

    def __init__(self, fields: List[Dict[str, Any]], *, theme: str = "light", **zones: Sequence[str]):
        super().__init__()
        self._by = {f["name"]: f for f in fields}
        self._syncing = False
        self.theme = theme
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
        self.observe(lambda _: self._redraw(), names=list(ZONES) + ["marks"])
        self._redraw()

    def paint(self, name: str, color: Optional[str]) -> None:
        """Highlight field ``name`` with ``color`` (a name from :data:`MARK_PALETTE`
        or any CSS color); ``color=None`` clears its mark. No-op for an unknown field."""
        self.marks = _set_mark(self.marks, name, color)

    def unmark(self, name: Optional[str] = None) -> None:
        """Clear one field's mark, or every mark when ``name`` is ``None``."""
        self.marks = {} if name is None else {k: v for k, v in self.marks.items() if k != name}

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
        mark_opts = [("mark…", "__unset__"), ("—", None)] + [(c, c) for c in MARK_PALETTE]
        for z in ZONES:
            rows = []
            for name in getattr(self, z):
                f = self._by.get(name)
                if f is None:
                    continue
                up = W.Button(icon="arrow-up", layout=W.Layout(width="30px"))
                down = W.Button(icon="arrow-down", layout=W.Layout(width="30px"))
                rm = W.Button(icon="times", layout=W.Layout(width="30px"))
                mark = W.Dropdown(options=mark_opts, value="__unset__", layout=W.Layout(width="80px"))
                up.on_click(lambda _, z=z, n=name: self._shift(z, n, -1))
                down.on_click(lambda _, z=z, n=name: self._shift(z, n, +1))
                rm.on_click(lambda _, n=name: self.move(n, None))
                mark.observe(lambda ch, n=name: self.paint(n, ch["new"]) if ch["new"] != "__unset__" else None, names="value")
                preview = theme_style_block(self.theme, selector=".p2h-fb-chip") + f"<style>{FIELDS_CSS}</style><div class='p2h-fb-chip'>{chip_html(f, drag=False, mark=self.marks.get(name))}</div>"
                rows.append(W.HBox([W.HTML(preview), mark, up, down, rm]))
            self._boxes[z].children = rows or [W.HTML("<span style='color:#99a;font-style:italic'>empty</span>")]

    def set_zones(self, **zones: Sequence[str]) -> None:
        with self.hold_trait_notifications():
            for k, v in zones.items():
                setattr(self, k, list(v))

    def snapshot_html(self) -> str:
        return zones_html(self.fields, {z: list(getattr(self, z)) for z in ZONES}, interactive=False, theme=self.theme, marks=self.marks)


def make_field_list(profile: Profile, *, prefer_anywidget: bool = True, theme: str = "light",
                     marks: Optional[Dict[str, str]] = None, **zones: Sequence[str]):
    """The best available fields pane for ``profile``. ``marks``: column name -> CSS
    color (or a :data:`MARK_PALETTE` name), for fields already highlighted."""
    fields = field_stats(profile)
    if prefer_anywidget and HAS_ANYWIDGET and FieldList is not None:
        w = FieldList(fields=fields, theme=theme)
        w.set_zones(**zones)
        if marks:
            w.marks = {k: MARK_PALETTE.get(v, v) for k, v in marks.items()}
        return w
    w = FieldListFallback(fields, theme=theme, **zones)
    if marks:
        w.marks = {k: MARK_PALETTE.get(v, v) for k, v in marks.items()}
    return w


__all__ = [
    "FieldList", "FieldListFallback", "make_field_list", "field_stats", "chip_html", "zones_html",
    "theme_style_block", "FIELDS_CSS", "FIELD_THEMES", "ZONES", "HAS_ANYWIDGET", "MARK_PALETTE",
]
