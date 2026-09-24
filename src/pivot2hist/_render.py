"""Plain-text rendering of pivot tables and histograms (Unicode bars, ASCII fallback)."""
from __future__ import annotations

import math
import shutil
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from ._binning import human

FULL = "█"
PARTIALS = ["", "▏", "▎", "▍", "▌", "▋", "▊", "▉"]


def terminal_width(default: int = 100) -> int:
    try:
        return max(40, shutil.get_terminal_size((default, 24)).columns)
    except Exception:  # pragma: no cover
        return default


def fmt_cell(x: Any, *, compact: bool = False) -> str:
    """Format a table cell: blanks for NaN, thousands separators, trimmed decimals."""
    if x is None:
        return ""
    if isinstance(x, (float, np.floating)):
        if math.isnan(x):
            return ""
        if compact:
            return human(float(x))
        if float(x).is_integer() and abs(x) < 1e15:
            return f"{int(x):,}"
        return f"{x:,.2f}"
    if isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_)):
        return human(int(x), integer=True) if compact else f"{int(x):,}"
    try:
        if pd.isna(x):
            return ""
    except (TypeError, ValueError):
        pass
    return str(x)


def format_table(table: pd.DataFrame, *, compact: bool = False) -> pd.DataFrame:
    """Copy of ``table`` with every cell formatted as a string."""
    return table.apply(lambda col: col.map(lambda v: fmt_cell(v, compact=compact)))


def render_pivot(table: pd.DataFrame, *, width: Optional[int] = None, max_rows: Optional[int] = None,
                 compact: bool = False) -> str:
    """Text rendering of a pivot table."""
    width = width or terminal_width()
    if table.empty:
        return "(empty)"
    shown = table if max_rows is None or len(table) <= max_rows else table.head(max_rows)
    text = format_table(shown, compact=compact).to_string(line_width=width, justify="right")
    if len(shown) < len(table):
        text += f"\n... {len(table) - len(shown):,} more rows"
    return text


def bar(value: float, vmax: float, width: int, *, ascii_only: bool = False) -> str:
    """A horizontal bar for ``value`` scaled so ``vmax`` fills ``width`` cells."""
    if not (vmax > 0) or width <= 0 or not (value > 0):
        return ""
    frac = min(1.0, value / vmax) * width
    if ascii_only:
        return "#" * max(1, int(round(frac)))
    full = int(frac)
    rem = int(round((frac - full) * 8))
    if rem == 8:
        full, rem = full + 1, 0
    s = FULL * full + PARTIALS[rem]
    return s or PARTIALS[1]


def _series_labels(columns: pd.Index) -> List[str]:
    if isinstance(columns, pd.MultiIndex):
        return [" / ".join(str(x) for x in tup) for tup in columns]
    return [str(c) for c in columns]


def render_hist(
    table: pd.DataFrame,
    *,
    title: Optional[str] = None,
    width: Optional[int] = None,
    ascii_only: bool = False,
    compact: bool = True,
    max_rows: Optional[int] = None,
) -> str:
    """Text histogram: one line per bin (row), one bar per series (column).

    Multi-level rows are drawn nested: outer levels become group headers and inner
    levels are indented under them.
    """
    width = width or terminal_width()
    if table.empty:
        return (title + "\n" if title else "") + "(empty)"
    series = _series_labels(table.columns)
    n_series = len(series)
    values = table.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    vmax = float(np.nanmax(np.abs(finite))) if finite.size else 0.0

    index = table.index
    nlev = index.nlevels
    if nlev == 1:
        labels = [str(x) for x in index]
        groups: List[Tuple[Tuple[str, ...], int]] = [((), i) for i in range(len(index))]
    else:
        labels = [str(t[-1]) for t in index]
        groups = [(tuple(str(x) for x in t[:-1]), i) for i, t in enumerate(index)]
    indent = "  " if nlev > 1 else ""
    label_w = max([len(indent + l) for l in labels] + [len(str(index.names[-1] or ""))])
    label_w = min(label_w, max(12, width // 3))

    num_strs = [[fmt_cell(v, compact=compact) for v in row] for row in values]
    num_w = max([len(s) for row in num_strs for s in row] + [1])
    avail = width - label_w - 2 - n_series * (num_w + 3)
    bar_w = max(6, avail // max(1, n_series))
    col_w = bar_w + 1 + num_w

    lines: List[str] = []
    if title:
        lines.append(title)
    head_label = str(index.names[-1] or "")
    if n_series == 1:
        lines.append(f"{head_label:<{label_w}}  {series[0]}")
    else:
        lines.append(f"{head_label:<{label_w}}  " + "  ".join(f"{s[:col_w]:<{col_w}}" for s in series))
    if max_rows is not None and len(index) > max_rows:
        rows_iter = list(range(max_rows))
    else:
        rows_iter = list(range(len(index)))
    prev_group: Optional[Tuple[str, ...]] = None
    outer_names = [str(n or "") for n in index.names[:-1]]
    for i in rows_iter:
        group, _ = groups[i]
        if nlev > 1 and group != prev_group:
            head = " > ".join(f"{n}={g}" if n else g for n, g in zip(outer_names, group))
            lines.append(head)
            prev_group = group
        lab = (indent + labels[i])
        if len(lab) > label_w:
            lab = lab[: label_w - 1] + "…"
        cells = []
        for j in range(n_series):
            v = values[i, j]
            b = bar(v if np.isfinite(v) else 0.0, vmax, bar_w, ascii_only=ascii_only)
            cells.append(f"{b:<{bar_w}} {num_strs[i][j]:>{num_w}}")
        lines.append(f"{lab:<{label_w}}  " + "  ".join(cells).rstrip())
    if len(rows_iter) < len(index):
        lines.append(f"... {len(index) - len(rows_iter):,} more bins")
    return "\n".join(lines)


def hist_html(table: pd.DataFrame, *, title: Optional[str] = None) -> str:
    """Minimal HTML rendering of a histogram table with inline bars (for notebooks)."""
    series = _series_labels(table.columns)
    values = table.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    vmax = float(np.nanmax(np.abs(finite))) if finite.size else 0.0
    out = ["<div class='pivot2hist'>"]
    if title:
        out.append(f"<div style='font-family:monospace;margin-bottom:4px'>{title}</div>")
    out.append("<table style='border-collapse:collapse;font-family:monospace;font-size:12px'>")
    out.append("<tr><th style='text-align:left;padding:2px 8px'>" + " / ".join(str(n or "") for n in table.index.names) + "</th>"
               + "".join(f"<th style='text-align:left;padding:2px 8px'>{s}</th>" for s in series) + "</tr>")
    for i, idx in enumerate(table.index):
        label = " / ".join(str(x) for x in idx) if isinstance(idx, tuple) else str(idx)
        cells = []
        for j in range(len(series)):
            v = values[i, j]
            pct = 0.0 if not (vmax > 0) or not np.isfinite(v) else min(100.0, abs(v) / vmax * 100)
            cells.append(
                f"<td style='padding:2px 8px;white-space:nowrap'><span style='display:inline-block;width:{pct:.1f}px;"
                f"max-width:200px;min-width:{'1' if pct>0 else '0'}px;height:10px;background:#4a7ebb;vertical-align:middle'></span> "
                f"{fmt_cell(v)}</td>"
            )
        out.append(f"<tr><td style='padding:2px 8px'>{label}</td>" + "".join(cells) + "</tr>")
    out.append("</table></div>")
    return "".join(out)
