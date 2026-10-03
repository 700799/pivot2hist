"""Start screen for :func:`bts_pivot.explore` when it is called without data.

::

    import bts_pivot as bp
    bp.explore()                      # upload files or a folder, type a path, or open a demo
    bp.explore("/content/drive/MyDrive/logs")   # a folder: every data file in it, combined

Uploads go to a temporary folder on the machine running the notebook (the Colab VM,
say), are read with :func:`bts_pivot.load`, and open in the explorer in the same output.
Several files become one table with a ``source_file`` column naming each row's file.
"""
from __future__ import annotations

import html as _html
import os
import shutil
import tempfile
import warnings
import weakref
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple

import ipywidgets as W
import pandas as pd
import traitlets as T

from ._io import load

#: File types offered for upload and picked up from folders. Pickle is left out on
#: purpose: unpickling a file can run arbitrary code.
EXTENSIONS = (".csv", ".tsv", ".txt", ".json", ".jsonl", ".ndjson", ".parquet", ".pq",
              ".xlsx", ".xls", ".feather", ".ft")
COMPRESSED = (".gz", ".bz2", ".zip", ".xz", ".zst")
COMPRESSIBLE = (".csv", ".tsv", ".txt", ".json", ".jsonl", ".ndjson")
SOURCE_COLUMN = "source_file"
_SKIP_DIRS = ("__MACOSX", "__pycache__")


def is_supported(name: str) -> bool:
    """Whether a file name is one the start screen and folder loading will read."""
    p = Path(name.lower())
    if p.suffix in EXTENSIONS:
        return True
    return p.suffix in COMPRESSED and Path(p.stem).suffix in COMPRESSIBLE


def _hidden(parts: Sequence[str]) -> bool:
    return any(p.startswith(".") or p in _SKIP_DIRS for p in parts)


def folder_files(folder: "str | os.PathLike[str]") -> List[Path]:
    """Every readable data file under ``folder`` (recursively), skipping hidden files."""
    root = Path(folder).expanduser()
    return sorted(p for p in root.rglob("*")
                  if p.is_file() and is_supported(p.name) and not _hidden(p.relative_to(root).parts))


def load_files(paths: Sequence[Path], root: Optional[Path] = None) -> Tuple[pd.DataFrame, List[str]]:
    """Read several files into one frame with a ``source_file`` column (the file's path
    relative to ``root``). Returns the frame and a ``"name: error"`` line per file that
    could not be read; raises ``ValueError`` when none could."""
    frames, names, failed = [], [], []
    for p in paths:
        name = p.relative_to(root).as_posix() if root is not None else p.name
        try:
            frames.append(load(p))
            names.append(name)
        except Exception as e:  # noqa: BLE001 - one unreadable file shouldn't sink the rest
            failed.append(f"{name}: {e}")
    if not frames:
        raise ValueError("none of the files could be read:\n" + "\n".join(failed))
    col = SOURCE_COLUMN
    while any(col in f.columns for f in frames):
        col = "_" + col
    frames = [f.assign(**{col: n})[[col, *f.columns]] for f, n in zip(frames, names)]
    return pd.concat(frames, ignore_index=True, sort=False), failed


def resolve_source(data: Any) -> Any:
    """A folder path becomes the combined frame of its data files (a lone file stays a path,
    so big files still get surveyed and paged); anything else passes through unchanged."""
    if not isinstance(data, (str, os.PathLike)):
        return data
    path = Path(data).expanduser()
    if not path.is_dir():  # also leaves duckdb:// URLs alone, which Path() would mangle
        return str(path) if isinstance(data, str) and data.startswith("~") else data
    files = folder_files(path)
    if not files:
        raise FileNotFoundError(f"no data files in {path} (looked for {', '.join(EXTENSIONS)}, also compressed)")
    if len(files) == 1:
        return str(files[0])
    df, failed = load_files(files, root=path)
    if failed:
        warnings.warn("skipped files that could not be read:\n" + "\n".join(failed), stacklevel=3)
    return df


def _safe_relpath(name: str) -> Path:
    """An uploaded file's name as a path that cannot leave the upload folder."""
    parts = [p for p in name.replace("\\", "/").split("/") if p not in ("", ".", "..")]
    return Path(*parts) if parts else Path("upload")


_UPLOAD_CSS = """
.bp-upload{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.bp-upload-btn{font:inherit;font-size:13px;padding:5px 12px;border:1px solid #b9c2cc;border-radius:6px;background:#f6f8fa;cursor:pointer}
.bp-upload-btn:hover{background:#eaeef2}
.bp-upload-btn:disabled{opacity:.55;cursor:default}
.bp-upload-status{font-size:12px;color:#57606a}
"""

_UPLOAD_ESM = r"""
const CHUNK = 2 * 1024 * 1024;
const fmt = (n) => n >= 1048576 ? (n / 1048576).toFixed(1) + " MB" : Math.max(1, Math.round(n / 1024)) + " KB";

function render({ model, el }) {
  el.classList.add("bp-upload");
  const status = document.createElement("span");
  status.className = "bp-upload-status";
  let waiting = null;
  model.on("msg:custom", (m) => { if (m && m.kind === "ack" && waiting) { const w = waiting; waiting = null; w(); } });
  const ack = () => new Promise((resolve, reject) => {
    waiting = resolve;
    setTimeout(() => { if (waiting === resolve) { waiting = null; reject(new Error("the kernel stopped answering")); } }, 120000);
  });
  const usable = (name) => {
    const n = name.toLowerCase();
    if (n.split("/").some((p) => p.startsWith(".") || p === "__macosx" || p === "__pycache__")) return false;
    const dot = n.lastIndexOf("."), ext = n.slice(dot);
    if (model.get("extensions").includes(ext)) return true;
    if (!model.get("compressed").includes(ext)) return false;
    const stem = n.slice(0, dot);
    return model.get("compressible").includes(stem.slice(stem.lastIndexOf(".")));
  };
  const buttons = [];
  async function send(list, isFolder) {
    if (!list.length) return;
    const chosen = list.filter((f) => usable(f.webkitRelativePath || f.name));
    const skipped = list.length - chosen.length;
    if (!chosen.length) { status.textContent = `no data files among the ${list.length} chosen`; return; }
    buttons.forEach((b) => (b.disabled = true));
    const total = chosen.reduce((s, f) => s + f.size, 0);
    let sent = 0;
    try {
      model.send({ kind: "begin", files: chosen.length, skipped, folder: isFolder });
      for (let i = 0; i < chosen.length; i++) {
        const f = chosen[i], name = f.webkitRelativePath || f.name;
        let off = 0;
        do {
          const buf = await f.slice(off, off + CHUNK).arrayBuffer();
          model.send({ kind: "chunk", name, offset: off }, undefined, [buf]);
          await ack();
          off += buf.byteLength; sent += buf.byteLength;
          status.textContent = `uploading ${i + 1} of ${chosen.length} · ${fmt(sent)} of ${fmt(total)}`;
        } while (off < f.size);
      }
      model.send({ kind: "end" });
      status.textContent = `uploaded ${chosen.length} file${chosen.length > 1 ? "s" : ""} (${fmt(total)})` +
        (skipped ? `, skipped ${skipped} that aren't data files` : "");
    } catch (e) {
      status.textContent = "upload failed: " + e.message;
    } finally {
      buttons.forEach((b) => (b.disabled = false));
    }
  }
  const add = (label, isFolder) => {
    const input = document.createElement("input");
    input.type = "file"; input.multiple = true; input.style.display = "none";
    input.className = isFolder ? "bp-upload-folder" : "bp-upload-files";
    if (isFolder) { input.webkitdirectory = true; input.setAttribute("webkitdirectory", ""); }
    else input.accept = model.get("accept");
    const b = document.createElement("button");
    b.className = "bp-upload-btn"; b.textContent = label;
    b.addEventListener("click", () => input.click());
    input.addEventListener("change", () => { const files = [...input.files]; input.value = ""; send(files, isFolder); });
    el.appendChild(b); el.appendChild(input); buttons.push(b);
  };
  add("Upload files…", false);
  add("Upload folder…", true);
  el.appendChild(status);
}
export default { render };
"""


class _Receiver:
    """Writes uploaded files into a fresh temporary folder, then hands their paths on."""

    def __init__(self, done: Callable[[List[Path], Path, int], None]):
        self._done = done
        self.folder: Optional[Path] = None
        self.paths: List[Path] = []
        self.skipped = 0

    def begin(self, skipped: int = 0) -> None:
        self.folder = Path(tempfile.mkdtemp(prefix="bts-pivot-upload-"))
        self.paths, self.skipped = [], skipped

    def write(self, name: str, data: bytes, offset: int = 0) -> None:
        if self.folder is None:
            self.begin()
        path = self.folder / _safe_relpath(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb" if offset == 0 else "ab") as fh:
            fh.write(data)
        if offset == 0:
            self.paths.append(path)

    def end(self) -> None:
        if self.folder is not None and self.paths:
            self._done(list(self.paths), self.folder, self.skipped)


try:
    import anywidget

    class Uploader(anywidget.AnyWidget):  # type: ignore[misc]
        """The "Upload files…" and "Upload folder…" buttons. Files go up in 2 MB chunks,
        each acknowledged by the kernel, so big uploads neither stall the browser nor
        exceed a message size limit."""

        _esm = _UPLOAD_ESM
        _css = _UPLOAD_CSS
        extensions = T.List(T.Unicode(), list(EXTENSIONS)).tag(sync=True)
        compressed = T.List(T.Unicode(), list(COMPRESSED)).tag(sync=True)
        compressible = T.List(T.Unicode(), list(COMPRESSIBLE)).tag(sync=True)
        accept = T.Unicode(",".join(EXTENSIONS + COMPRESSED)).tag(sync=True)

        def __init__(self, done: Callable[[List[Path], Path, int], None], **kw: Any):
            super().__init__(**kw)
            self._rx = _Receiver(done)
            self.on_msg(self._on_msg)

        def _on_msg(self, _widget: Any, content: Any, buffers: Sequence[Any]) -> None:
            kind = content.get("kind") if isinstance(content, dict) else None
            if kind == "begin":
                self._rx.begin(int(content.get("skipped", 0)))
            elif kind == "chunk":
                self._rx.write(str(content.get("name", "upload")), bytes(buffers[0]) if buffers else b"",
                               int(content.get("offset", 0)))
                self.send({"kind": "ack"})
            elif kind == "end":
                self._rx.end()

    HAS_ANYWIDGET = True
except ImportError:  # pragma: no cover - exercised only without anywidget
    HAS_ANYWIDGET = False
    Uploader = None  # type: ignore[assignment,misc]


def _upload_items(value: Any) -> List[Tuple[str, bytes]]:
    """(name, bytes) pairs from a ``FileUpload`` value: a tuple of dicts in ipywidgets 8,
    a ``{name: {"content": ...}}`` dict in ipywidgets 7."""
    if isinstance(value, dict):
        return [(name, bytes(v["content"])) for name, v in value.items()]
    return [(v["name"], bytes(v["content"])) for v in value]


def _fallback_uploader(done: Callable[[List[Path], Path, int], None]) -> W.Widget:
    """Plain-ipywidgets file upload for when anywidget isn't installed (no folder picker)."""
    up = W.FileUpload(accept=",".join(EXTENSIONS + COMPRESSED), multiple=True, description="Upload files")
    rx = _Receiver(done)

    def changed(change: Any) -> None:
        items = [(n, b) for n, b in _upload_items(change["new"]) if is_supported(n)]
        if not items:
            return
        rx.begin(skipped=len(_upload_items(change["new"])) - len(items))
        for name, data in items:
            rx.write(name, data)
        rx.end()

    up.observe(changed, names="value")
    return W.VBox([up, W.HTML("<span style='font-size:12px;color:#57606a'>Folder upload needs anywidget "
                              "(<code>pip install \"bts-pivot[jupyter]\"</code>); or type a folder path below.</span>")])


_DEMOS = {"firewall": "firewall logs (5,000 rows)", "auth": "auth logs (3,000 rows)"}
_HINT_STYLE = "font-size:12px;color:#57606a"
_LABEL_STYLE = "font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:#57606a;font-weight:600"


class Launcher:
    """What ``bp.explore()`` shows with no data: pick a source, then the explorer opens in
    the same output. ``launcher.explorer`` is that :class:`~bts_pivot.ui.Explorer` once
    one is open (``None`` before), ``launcher.view`` its current view."""

    def __init__(self, opener: Callable[[Any], Any]):
        self._opener = opener
        self.explorer: Any = None
        self._tmp: List[Path] = []
        self._finalizer = weakref.finalize(self, _remove_all, self._tmp)
        self.w_status = W.HTML()
        upload = Uploader(self._uploaded) if HAS_ANYWIDGET else _fallback_uploader(self._uploaded)
        self.uploader = upload
        self.w_path = W.Text(placeholder="data/events.parquet  or  /content/drive/MyDrive/logs",
                             layout=W.Layout(width="440px"))
        self.w_path_open = W.Button(description="Open", icon="folder-open", layout=W.Layout(width="90px"))
        self.w_demo = W.Dropdown(options=[(label, key) for key, label in _DEMOS.items()], layout=W.Layout(width="220px"))
        self.w_demo_open = W.Button(description="Open demo", icon="play", layout=W.Layout(width="120px"))
        self.w_path_open.on_click(lambda _b: self.open_path(self.w_path.value))
        self.w_demo_open.on_click(lambda _b: self.open_demo(self.w_demo.value))
        types = ", ".join(e.lstrip(".") for e in ("csv", "tsv", "json", "jsonl", "parquet", "xlsx", "feather"))
        self.start = W.VBox([
            W.HTML(f"<div style='font-size:16px;font-weight:600;margin:2px 0'>bts-pivot explorer</div>"
                   f"<div style='{_HINT_STYLE}'>Choose data to explore: {types}, also gzip/zip-compressed. "
                   "Several files or a folder open as one table with a <code>source_file</code> column.</div>"),
            W.HTML(f"<div style='{_LABEL_STYLE};margin-top:8px'>Upload from this computer</div>"),
            upload,
            W.HTML(f"<div style='{_LABEL_STYLE};margin-top:8px'>File or folder on the notebook's machine</div>"),
            W.HBox([self.w_path, self.w_path_open]),
            W.HTML(f"<div style='{_HINT_STYLE}'>Big files load faster from a path than through the browser. In Colab, "
                   "mount Drive first (<code>from google.colab import drive; drive.mount('/content/drive')</code>), "
                   "then use <code>/content/drive/MyDrive/…</code>.</div>"),
            W.HTML(f"<div style='{_LABEL_STYLE};margin-top:8px'>Or try a demo</div>"),
            W.HBox([self.w_demo, self.w_demo_open]),
            self.w_status,
        ], layout=W.Layout(padding="6px 4px"))
        self.w_back = W.Button(description="Choose other data", icon="arrow-left",
                                 layout=W.Layout(width="170px", margin="0 10px 0 0"))
        self.w_back.on_click(lambda _b: self.show_start())
        self.w_loaded = W.HTML()
        self.box = W.VBox([self.start])

    # ------------------------------------------------------------------ sources

    def open_path(self, path: str) -> None:
        """Open a file or folder on the notebook's machine."""
        path = path.strip().strip("'\"")
        if not path:
            self._error("type the path of a file or folder first")
            return
        p = Path(path).expanduser()
        if not p.exists():
            self._error(f"no such file or folder: {path}")
            return
        if p.is_dir():
            files = folder_files(p)
            if not files:
                self._error(f"no data files in {path} (looked for {', '.join(EXTENSIONS)}, also compressed)")
                return
            self._open_many(files, p, f"{p.name or p}/")
            return
        if not is_supported(p.name):
            self._error(f"{p.name} isn't a supported data file ({', '.join(EXTENSIONS)}, also compressed)")
            return
        self._open(str(p), p.name)

    def open_demo(self, name: str) -> None:
        """Open one of the bundled sample datasets (``"firewall"`` or ``"auth"``)."""
        from . import sample

        self._open({"firewall": sample.firewall_logs, "auth": sample.auth_logs}[name](), _DEMOS[name])

    def _uploaded(self, paths: List[Path], folder: Path, skipped: int) -> None:
        self._tmp.append(folder)
        rel = [p.relative_to(folder) for p in paths]
        if len(rel) == 1:
            label = rel[0].name
        elif len({r.parts[0] for r in rel}) == 1 and all(len(r.parts) > 1 for r in rel):
            label = f"{rel[0].parts[0]}/"  # an uploaded folder
        else:
            label = f"{len(rel)} files"
        note = f" · skipped {skipped} non-data file{'s' if skipped != 1 else ''}" if skipped else ""
        self._open_many(paths, folder, label, note)

    def _open_many(self, paths: List[Path], root: Path, label: str, note: str = "") -> None:
        if len(paths) == 1:  # stays a path, so a big file is surveyed and paged
            self._open(str(paths[0]), label, note)
            return
        try:
            df, failed = load_files(paths, root=root)
        except Exception as e:  # noqa: BLE001 - shown in the widget instead of a traceback
            self._error(str(e))
            return
        if failed:
            note += f" · couldn't read {len(failed)}: " + "; ".join(failed)
        self._open(df, label, note)

    def _open(self, source: Any, label: str, note: str = "") -> None:
        self.w_status.value = f"<span style='{_HINT_STYLE}'>loading {_html.escape(label)}…</span>"
        try:
            explorer = self._opener(source)
        except Exception as e:  # noqa: BLE001
            self._error(f"couldn't open {label}: {e}")
            return
        self.explorer = explorer
        self.w_status.value = ""
        self.w_loaded.value = (f"<span style='font-size:13px'>Exploring <b>{_html.escape(label)}</b></span>"
                               f"<span style='{_HINT_STYLE}'>{_html.escape(note)}</span>")
        self.box.children = [W.HBox([self.w_back, self.w_loaded], layout=W.Layout(align_items="center")),
                             explorer.box]

    def show_start(self) -> None:
        """Back to the start screen (the current explorer stays in ``launcher.explorer``)."""
        self.box.children = [self.start]

    def _error(self, message: str) -> None:
        self.w_status.value = f"<span style='font-size:12px;color:#b42318;white-space:pre-wrap'>{_html.escape(message)}</span>"

    # ------------------------------------------------------------------ access

    @property
    def view(self) -> Any:
        return None if self.explorer is None else self.explorer.view

    def _repr_mimebundle_(self, **kw: Any) -> Any:
        return self.box._repr_mimebundle_(**kw)

    def _ipython_display_(self) -> None:  # pragma: no cover - notebook only
        from IPython.display import display

        display(self.box)


def _remove_all(folders: List[Path]) -> None:
    for f in folders:
        shutil.rmtree(f, ignore_errors=True)


__all__ = ["Launcher", "Uploader", "EXTENSIONS", "SOURCE_COLUMN", "folder_files", "is_supported",
           "load_files", "resolve_source"]
