"""The explorer's output pane: the rendered heatmap / histogram, with clickable cells.

:class:`ClickableHTML` is an ``anywidget`` that shows the HTML the explorer renders and
reports which cell (``data-p2h-row`` / ``data-p2h-col`` attributes on heatmap cells and
histogram bars) was clicked through its ``clicked`` trait. Without ``anywidget`` the pane
is a plain ``ipywidgets.HTML`` (same ``value`` interface, no clicks); the Inspect tab's
row/column pickers cover that case.
"""
from __future__ import annotations

from typing import Any

import ipywidgets as W
import traitlets as T

_ESM = """
function render({ model, el }) {
  const style = document.createElement("style");
  style.textContent =
    "[data-p2h-row]{cursor:pointer}" +
    "td[data-p2h-row]:hover{outline:2px solid #4a7ebb;outline-offset:-2px}" +
    "rect[data-p2h-row]:hover{stroke:#222;stroke-width:1.5}";
  el.appendChild(style);
  const box = document.createElement("div");
  el.appendChild(box);
  const draw = () => { box.innerHTML = model.get("value"); };
  draw();
  model.on("change:value", draw);
  box.addEventListener("click", (ev) => {
    const prev = model.get("clicked") || {};
    const s = ev.target.closest("[data-p2h-slice]");
    if (s) {
      model.set("clicked", { slice: s.dataset.p2hSlice, n: (prev.n || 0) + 1 });
      model.save_changes();
      return;
    }
    const t = ev.target.closest("[data-p2h-row]");
    if (!t) return;
    model.set("clicked", {
      row: JSON.parse(t.dataset.p2hRow),
      col: JSON.parse(t.dataset.p2hCol),
      n: (prev.n || 0) + 1,
    });
    model.save_changes();
  });
}
export default { render };
"""

_COPY_ESM = """
function render({ model, el }) {
  const b = document.createElement("button");
  b.className = "jupyter-button widget-button";
  b.style.cssText = "padding:4px 12px;border-radius:6px;cursor:pointer";
  const label = () => { b.textContent = model.get("label"); };
  label();
  model.on("change:label", label);
  b.addEventListener("click", async () => {
    const text = model.get("text") || "";
    try {
      await navigator.clipboard.writeText(text);
    } catch (e) {
      const ta = document.createElement("textarea");
      ta.value = text;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      ta.remove();
    }
    b.textContent = "Copied";
    setTimeout(label, 1500);
    model.set("copies", (model.get("copies") || 0) + 1);
    model.save_changes();
  });
  el.appendChild(b);
}
export default { render };
"""

try:
    import anywidget

    class CopyButton(anywidget.AnyWidget):  # type: ignore[misc]
        """A button that copies its ``text`` to the clipboard; ``copies`` counts clicks."""

        _esm = _COPY_ESM
        text = T.Unicode("").tag(sync=True)
        label = T.Unicode("Copy").tag(sync=True)
        copies = T.Int(0).tag(sync=True)

    class ClickableHTML(anywidget.AnyWidget):  # type: ignore[misc]
        """Rendered HTML whose cells report clicks: observe ``clicked`` for
        ``{"row": [labels...], "col": [labels...], "n": click count}`` (a cell or a bar) or
        ``{"slice": label, "n": ...}`` (a slice chip)."""

        _esm = _ESM
        value = T.Unicode("").tag(sync=True)
        clicked = T.Dict().tag(sync=True)

    HAS_ANYWIDGET = True
except ImportError:  # pragma: no cover - exercised only without anywidget
    HAS_ANYWIDGET = False
    ClickableHTML = None  # type: ignore[assignment,misc]
    CopyButton = None  # type: ignore[assignment,misc]


def make_output(*, prefer_anywidget: bool = True) -> Any:
    """The best available output pane: clickable when ``anywidget`` is installed."""
    if prefer_anywidget and HAS_ANYWIDGET and ClickableHTML is not None:
        return ClickableHTML()
    return W.HTML()


def make_copy_button(label: str = "Copy", *, prefer_anywidget: bool = True) -> Any:
    """A one-click copy-to-clipboard button when ``anywidget`` is installed; otherwise a
    disabled placeholder that says to select the text and copy it."""
    if prefer_anywidget and HAS_ANYWIDGET and CopyButton is not None:
        return CopyButton(label=label)
    return W.Button(description=label, disabled=True, tooltip="one-click copy needs anywidget; select the text and copy it")


__all__ = ["ClickableHTML", "CopyButton", "make_output", "make_copy_button", "HAS_ANYWIDGET"]
