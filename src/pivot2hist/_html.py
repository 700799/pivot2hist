"""Dependency-free graphics: an HTML heatmap for pivot tables and an SVG histogram.

Both render in any notebook front-end (Jupyter, Lab, VS Code, Colab, nbviewer, GitHub)
because they are plain HTML/SVG with inline styles: no widget extension, no JS, no CSS
framework. Colours are a colour-blind-safe categorical palette (Okabe-Ito) and a single
sequential ramp for heat.
"""
from __future__ import annotations

import html
import math
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from ._binning import human
from ._render import fmt_cell

PALETTE = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#999999")
FONT = "font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif"
MONO = "font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"


def _esc(x: Any) -> str:
    return html.escape(str(x), quote=True)


def _labels(index: pd.Index) -> List[Tuple[str, ...]]:
    if isinstance(index, pd.MultiIndex):
        return [tuple(str(x) for x in t) for t in index]
    return [(str(x),) for x in index]


def heat_color(t: float, *, negative: bool = False) -> str:
    """Sequential ramp: white -> deep blue (or white -> deep red for negatives)."""
    t = 0.0 if not (t == t) else max(0.0, min(1.0, t))
    if negative:
        r, g, b = 255 - int(60 * t), 255 - int(180 * t), 255 - int(200 * t)
    else:
        r, g, b = 255 - int(240 * t), 255 - int(160 * t), 255 - int(80 * t)
    return f"rgb({r},{g},{b})"


def _text_color(t: float) -> str:
    return "#fff" if t > 0.62 else "#111"


def _scale(values: np.ndarray, mode: str) -> np.ndarray:
    """Map values to [0,1] for colouring, globally or per column; log when heavy-tailed."""
    v = values.astype(float)
    out = np.zeros_like(v)
    finite = np.isfinite(v)
    if not finite.any():
        return out
    absv = np.abs(v)
    pos = absv[finite & (absv > 0)]
    use_log = pos.size > 4 and pos.max() / pos.min() > 1000
    x = np.log1p(absv) if use_log else absv

    def norm(block: np.ndarray, mask: np.ndarray) -> np.ndarray:
        res = np.zeros_like(block)
        if mask.any():
            hi = block[mask].max()
            lo = block[mask].min()
            span = hi - lo
            res[mask] = (block[mask] - lo) / span if span > 0 else (1.0 if hi > 0 else 0.0)
        return res

    if mode == "column" and v.ndim == 2:
        for j in range(v.shape[1]):
            out[:, j] = norm(x[:, j], finite[:, j])
    elif mode == "row" and v.ndim == 2:
        for i in range(v.shape[0]):
            out[i, :] = norm(x[i, :], finite[i, :])
    else:
        out = norm(x, finite)
    return out


def pivot_html(
    table: pd.DataFrame,
    *,
    title: Optional[str] = None,
    heat: str = "table",  # "table" | "column" | "row" | "none"
    bars: bool = False,
    totals: bool = False,
    compact: bool = False,
    max_rows: Optional[int] = None,
) -> str:
    """Pivot table as an HTML heatmap. Cells carry the exact value as a tooltip."""
    if table.empty:
        return _wrap(title, "<div style='color:#888;padding:8px'>(empty)</div>")
    t = table
    if max_rows is not None and len(t) > max_rows:
        t = t.head(max_rows)
    values = t.to_numpy(dtype=float)
    shade = _scale(values, heat) if heat != "none" else np.zeros_like(values)
    row_labels = _labels(t.index)
    col_labels = _labels(t.columns)
    n_row_levels = t.index.nlevels
    n_col_levels = t.columns.nlevels
    row_names = [str(n) if n is not None else "" for n in t.index.names]
    col_names = [str(n).rstrip() if n is not None else "" for n in t.columns.names]

    out: List[str] = []
    out.append(
        f"<table style='border-collapse:separate;border-spacing:0;{FONT};font-size:12px;"
        f"border:1px solid #ddd;border-radius:6px;overflow:hidden'>"
    )
    # ---- column headers, one row per column level, with colspan grouping
    for lvl in range(n_col_levels):
        out.append("<tr>")
        for r in range(n_row_levels):
            head = col_names[lvl] if r == n_row_levels - 1 else ""
            out.append(
                f"<th style='text-align:left;padding:4px 8px;background:#f5f6f8;color:#555;font-weight:600'>{_esc(head)}</th>"
            )
        j = 0
        while j < len(col_labels):
            k = j
            while k + 1 < len(col_labels) and col_labels[k + 1][: lvl + 1] == col_labels[j][: lvl + 1]:
                k += 1
            span = k - j + 1
            out.append(
                f"<th colspan='{span}' style='text-align:right;padding:4px 8px;background:#f5f6f8;color:#222;"
                f"font-weight:600;border-left:1px solid #e6e6e6'>{_esc(col_labels[j][lvl])}</th>"
            )
            j = k + 1
        if totals:
            out.append("<th style='padding:4px 8px;background:#f5f6f8;text-align:right'>" + ("total" if lvl == n_col_levels - 1 else "") + "</th>")
        out.append("</tr>")
    if any(row_names):
        out.append("<tr>")
        for r in range(n_row_levels):
            out.append(f"<th style='text-align:left;padding:2px 8px;background:#fafafa;color:#777;font-weight:500'>{_esc(row_names[r])}</th>")
        out.append(f"<th colspan='{len(col_labels) + (1 if totals else 0)}' style='background:#fafafa'></th></tr>")
    # ---- body
    row_sum = np.nansum(np.where(np.isfinite(values), values, 0.0), axis=1) if totals else None
    prev: Tuple[str, ...] = ()
    vmax_bar = np.nanmax(np.abs(values)) if bars and np.isfinite(values).any() else 0.0
    for i, labels in enumerate(row_labels):
        out.append("<tr>")
        for r in range(n_row_levels):
            same = prev[: r + 1] == labels[: r + 1] if prev else False
            txt = "" if (same and r < n_row_levels - 1) else labels[r]
            style = "text-align:left;padding:3px 8px;white-space:nowrap;color:#222;background:#fff"
            if r < n_row_levels - 1:
                style += ";font-weight:600;border-top:1px solid #eee" if not same else ""
            out.append(f"<td style='{style}'>{_esc(txt)}</td>")
        for j in range(values.shape[1]):
            v = values[i, j]
            raw = t.iat[i, j]
            text = fmt_cell(raw, compact=compact)
            sh = float(shade[i, j])
            bg = heat_color(sh, negative=(v < 0)) if heat != "none" and np.isfinite(v) else "#fff"
            fg = _text_color(sh) if heat != "none" else "#222"
            cell = _esc(text)
            if bars and vmax_bar > 0 and np.isfinite(v):
                pct = min(100.0, abs(v) / vmax_bar * 100)
                bar = (
                    f"<span style='display:inline-block;width:60px;height:8px;background:#e8edf3;vertical-align:middle;margin-right:6px;border-radius:2px'>"
                    f"<span style='display:block;width:{pct:.1f}%;height:8px;background:{PALETTE[0]};border-radius:2px'></span></span>"
                )
                cell = bar + cell
            out.append(
                f"<td title='{_esc(fmt_cell(raw))}' style='text-align:right;padding:3px 8px;{MONO};white-space:nowrap;"
                f"background:{bg};color:{fg};border-left:1px solid #f0f0f0'>{cell}</td>"
            )
        if totals:
            out.append(f"<td style='text-align:right;padding:3px 8px;{MONO};font-weight:600;background:#fafafa'>{_esc(fmt_cell(row_sum[i], compact=compact))}</td>")
        out.append("</tr>")
        prev = labels
    if totals:
        col_sum = np.nansum(np.where(np.isfinite(values), values, 0.0), axis=0)
        out.append("<tr>")
        out.append(f"<td colspan='{n_row_levels}' style='padding:3px 8px;font-weight:600;background:#fafafa;border-top:1px solid #ddd'>total</td>")
        for j in range(values.shape[1]):
            out.append(f"<td style='text-align:right;padding:3px 8px;{MONO};font-weight:600;background:#fafafa;border-top:1px solid #ddd'>{_esc(fmt_cell(col_sum[j], compact=compact))}</td>")
        out.append(f"<td style='text-align:right;padding:3px 8px;{MONO};font-weight:700;background:#f0f0f0;border-top:1px solid #ddd'>{_esc(fmt_cell(float(col_sum.sum()), compact=compact))}</td>")
        out.append("</tr>")
    out.append("</table>")
    if len(t) < len(table):
        out.append(f"<div style='color:#888;font-size:11px;padding:4px 2px'>... {len(table) - len(t):,} more rows</div>")
    return _wrap(title, "".join(out))


def _wrap(title: Optional[str], body: str) -> str:
    head = f"<div style='{FONT};font-size:12px;color:#444;margin:0 0 6px 0'>{_esc(title)}</div>" if title else ""
    return f"<div class='pivot2hist' style='display:inline-block;max-width:100%;overflow-x:auto'>{head}{body}</div>"


def _ticks(vmax: float, n: int = 5) -> List[float]:
    if not (vmax > 0):
        return [0.0]
    raw = vmax / n
    exp = math.floor(math.log10(raw))
    base = 10 ** exp
    step = next(m * base for m in (1, 2, 2.5, 5, 10) if m * base >= raw)
    return [i * step for i in range(int(vmax // step) + 1)]


def hist_svg(
    table: pd.DataFrame,
    *,
    title: Optional[str] = None,
    width: int = 760,
    height: int = 340,
    density: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    stacked: bool = False,
    log_y: bool = False,
    show_values: Optional[bool] = None,
) -> str:
    """Histogram / bar chart as inline SVG.

    Rows are the bins on the x axis (multi-level rows are drawn as labelled groups),
    columns are the series (one colour each, side by side or stacked), and every bar
    carries a tooltip. ``density`` is an optional ``(x, y)`` curve drawn over a numeric
    axis, e.g. from :func:`pivot2hist.bin_edges`'s sibling :func:`kde`.
    """
    if table.empty:
        return _wrap(title, "<div style='color:#888;padding:8px'>(empty)</div>")
    values = np.where(np.isfinite(table.to_numpy(dtype=float)), table.to_numpy(dtype=float), 0.0)
    n_bins, n_series = values.shape
    series = [" / ".join(l) for l in _labels(table.columns)]
    bins = _labels(table.index)
    n_levels = table.index.nlevels
    outer = [b[:-1] for b in bins]
    leaf = [b[-1] for b in bins]

    left, right, top = 56, 16, 28 if title else 14
    legend_h = 22 if n_series > 1 else 0
    label_h = 44 if n_bins > 12 else 26
    group_h = 16 * (n_levels - 1)
    bottom = label_h + group_h + legend_h + 10
    plot_w = max(60, width - left - right)
    plot_h = max(60, height - top - bottom)

    if stacked and n_series > 1:
        tops = values.clip(min=0).sum(axis=1)
        vmax = float(tops.max()) if tops.size else 0.0
    else:
        vmax = float(np.abs(values).max()) if values.size else 0.0
    if vmax <= 0:
        vmax = 1.0

    def y_of(v: float) -> float:
        if log_y:
            return top + plot_h - plot_h * (math.log1p(max(v, 0)) / math.log1p(vmax))
        return top + plot_h - plot_h * (max(v, 0) / vmax)

    slot = plot_w / n_bins
    gap = min(6.0, slot * 0.15)
    inner = slot - gap
    bar_w = inner if (stacked or n_series == 1) else inner / n_series
    if show_values is None:
        show_values = n_bins * max(1, 1 if stacked else n_series) <= 24

    out: List[str] = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}' "
        f"style='{FONT};font-size:11px;max-width:100%;height:auto'>"
    ]
    if title:
        out.append(f"<text x='{left}' y='16' fill='#444' font-size='12'>{_esc(title)}</text>")
    # y grid + ticks
    for tv in _ticks(vmax):
        y = y_of(tv)
        out.append(f"<line x1='{left}' x2='{left + plot_w}' y1='{y:.1f}' y2='{y:.1f}' stroke='#eee'/>")
        out.append(f"<text x='{left - 6}' y='{y + 4:.1f}' text-anchor='end' fill='#666'>{_esc(human(tv))}</text>")
    out.append(f"<line x1='{left}' x2='{left + plot_w}' y1='{top + plot_h:.1f}' y2='{top + plot_h:.1f}' stroke='#bbb'/>")
    # bars
    for i in range(n_bins):
        x0 = left + i * slot + gap / 2
        base = 0.0
        for j in range(n_series):
            v = float(values[i, j])
            color = PALETTE[j % len(PALETTE)]
            if stacked and n_series > 1:
                y1, y0 = y_of(base + max(v, 0)), y_of(base)
                x = x0
                base += max(v, 0)
            else:
                y1, y0 = y_of(abs(v)), y_of(0)
                x = x0 + j * bar_w
            h = max(0.0, y0 - y1)
            tip = f"{' / '.join(bins[i])}" + (f" · {series[j]}" if n_series > 1 else "") + f": {fmt_cell(table.iat[i, j])}"
            out.append(
                f"<rect x='{x:.1f}' y='{y1:.1f}' width='{max(bar_w, 0.5):.1f}' height='{h:.1f}' fill='{color}' "
                f"opacity='{0.55 if v < 0 else 0.9}' rx='1.5'><title>{_esc(tip)}</title></rect>"
            )
            if show_values and h > 0 and not stacked:
                out.append(
                    f"<text x='{x + bar_w / 2:.1f}' y='{y1 - 3:.1f}' text-anchor='middle' fill='#333' font-size='10'>{_esc(human(abs(v)))}</text>"
                )
    # x labels
    step = 1 if n_bins <= 12 else max(1, math.ceil(n_bins / 24))
    for i in range(0, n_bins, step):
        xc = left + i * slot + slot / 2
        lab = leaf[i]
        if len(lab) > 18:
            lab = lab[:17] + "…"
        if n_bins > 12:
            out.append(
                f"<text x='{xc:.1f}' y='{top + plot_h + 12:.1f}' transform='rotate(-40 {xc:.1f} {top + plot_h + 12:.1f})' "
                f"text-anchor='end' fill='#444'>{_esc(lab)}</text>"
            )
        else:
            out.append(f"<text x='{xc:.1f}' y='{top + plot_h + 16:.1f}' text-anchor='middle' fill='#444'>{_esc(lab)}</text>")
    # group brackets for multi-level rows
    if n_levels > 1:
        y = top + plot_h + label_h + 4
        i = 0
        while i < n_bins:
            k = i
            while k + 1 < n_bins and outer[k + 1] == outer[i]:
                k += 1
            xa, xb = left + i * slot + 2, left + (k + 1) * slot - 2
            out.append(f"<line x1='{xa:.1f}' x2='{xb:.1f}' y1='{y}' y2='{y}' stroke='#999'/>")
            out.append(f"<text x='{(xa + xb) / 2:.1f}' y='{y + 12}' text-anchor='middle' fill='#333' font-weight='600'>{_esc(' / '.join(outer[i]))}</text>")
            i = k + 1
    # density curve
    if density is not None and len(density[0]) > 1 and n_levels == 1:
        dx, dy = np.asarray(density[0], dtype=float), np.asarray(density[1], dtype=float)
        if dy.max() > 0:
            xs = left + (np.arange(n_bins) + 0.5) * slot
            # map density x onto bin positions: assume bins are ordered numeric ranges, spread evenly
            gx = np.linspace(xs[0] - slot / 2, xs[-1] + slot / 2, len(dx))
            gy = top + plot_h - (dy / dy.max()) * plot_h * 0.95
            pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(gx, gy))
            out.append(f"<polyline points='{pts}' fill='none' stroke='#333' stroke-width='1.5' stroke-dasharray='4 3' opacity='0.8'><title>density (KDE)</title></polyline>")
    # legend
    if n_series > 1:
        y = height - 8
        x = left
        for j, name in enumerate(series):
            color = PALETTE[j % len(PALETTE)]
            out.append(f"<rect x='{x}' y='{y - 9}' width='10' height='10' fill='{color}' rx='2'/>")
            label = name if len(name) <= 22 else name[:21] + "…"
            out.append(f"<text x='{x + 14}' y='{y}' fill='#333'>{_esc(label)}</text>")
            x += 14 + 6.5 * len(label) + 14
            if x > width - 60:
                break
    out.append("</svg>")
    return _wrap(None, "".join(out))


__all__ = ["pivot_html", "hist_svg", "heat_color", "PALETTE"]
