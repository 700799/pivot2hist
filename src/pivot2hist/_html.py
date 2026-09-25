"""Dependency-free graphics: an HTML heatmap for pivot tables and an SVG histogram.

Both render in any notebook front-end (Jupyter, Lab, VS Code, Colab, nbviewer, GitHub)
because they are plain HTML/SVG with inline styles: no widget extension, no JS, no CSS
framework. Colours are a colour-blind-safe categorical palette (Okabe-Ito) for series and
a single sequential ramp for heat, in two themes: ``"light"`` (the default, unchanged from
earlier releases) and ``"graphite"`` (a modern dark theme: charcoal panels, soft borders,
a blue accent) — pass ``theme="graphite"`` to any function here, or ``view.style(theme=...)``.
"""
from __future__ import annotations

import html
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ._binning import human
from ._render import fmt_cell

PALETTE = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#999999")
#: The same series colours, tuned a little brighter for a dark background.
PALETTE_GRAPHITE = ("#5B9BD5", "#F2B035", "#3FC79A", "#E8785A", "#D98CC0", "#7FC8F0", "#E8DE5A", "#9AA3B1")
FONT = "font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif"
MONO = "font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"
THEMES: Tuple[str, ...] = ("light", "graphite")

#: One colour token set per theme. ``light`` reproduces the literal colours this module
#: always used, so passing no ``theme`` (or ``theme="light"``) renders byte-identical output.
PALETTES: Dict[str, Dict[str, str]] = {
    "light": {
        "page_bg": None, "panel_pad": "0", "panel_radius": "6px",
        "border": "#ddd", "border_soft": "#f0f0f0", "row_border": "#eee",
        "head_bg": "#f5f6f8", "head_text": "#222", "head_muted": "#555",
        "subhead_bg": "#fafafa", "subhead_text": "#777",
        "total_bg": "#fafafa", "total_bg_strong": "#f0f0f0",
        "sub_bg": "#eef2f7", "sub_border": "#d9e1ea", "sub_text": "#334",
        "cell_bg": "#fff", "cell_text": "#222", "cell_border": "#f0f0f0",
        "bar_track": "#e8edf3", "white": "#fff", "dark_text": "#111",
        "outline_bg": "#eef2f7", "outline_chevron": "#667", "outline_count": "#778",
        "outline_border": "#d9e1ea", "outline_outer_border": "#ddd",
        "svg_title": "#444", "svg_axis_text": "#666", "svg_axis_line": "#bbb",
        "svg_grid": "#eee", "svg_xlabel": "#444", "svg_group_line": "#999",
        "svg_group_text": "#333", "svg_value_text": "#333", "svg_legend_text": "#333",
        "svg_density": "#333", "empty_text": "#888",
        "heat_pos_from": (255, 255, 255), "heat_pos_to": (15, 95, 175),
        "heat_neg_from": (255, 255, 255), "heat_neg_to": (195, 75, 55),
        "text_on_heat_light": "#111", "text_on_heat_dark": "#fff", "heat_text_switch": 0.62,
        "palette": PALETTE,
    },
    "graphite": {
        "page_bg": "#161a20", "panel_pad": "2px", "panel_radius": "10px",
        "border": "#333a45", "border_soft": "#262c35", "row_border": "#262c35",
        "head_bg": "#1e242c", "head_text": "#e5e9ef", "head_muted": "#98a1b0",
        "subhead_bg": "#1a1f26", "subhead_text": "#98a1b0",
        "total_bg": "#1d232b", "total_bg_strong": "#242b35",
        "sub_bg": "#20272f", "sub_border": "#3a4250", "sub_text": "#c4cbd6",
        "cell_bg": "#1b2128", "cell_text": "#dfe3ea", "cell_border": "#262c35",
        "bar_track": "#2a313b", "white": "#eef1f5", "dark_text": "#0f1216",
        "outline_bg": "#20272f", "outline_chevron": "#8fa8d8", "outline_count": "#8891a0",
        "outline_border": "#3a4250", "outline_outer_border": "#333a45",
        "svg_title": "#c7ccd6", "svg_axis_text": "#98a1b0", "svg_axis_line": "#4a5261",
        "svg_grid": "#262c35", "svg_xlabel": "#c7ccd6", "svg_group_line": "#5b6577",
        "svg_group_text": "#c7ccd6", "svg_value_text": "#c7ccd6", "svg_legend_text": "#c7ccd6",
        "svg_density": "#eef1f5", "empty_text": "#8891a0",
        "heat_pos_from": (27, 33, 40), "heat_pos_to": (67, 143, 224),
        "heat_neg_from": (27, 33, 40), "heat_neg_to": (216, 96, 82),
        "text_on_heat_light": "#0f1216", "text_on_heat_dark": "#eef1f5", "heat_text_switch": 1.1,
        "palette": PALETTE_GRAPHITE,
    },
}


def _pal(theme: str) -> Dict[str, Any]:
    try:
        return PALETTES[theme]
    except KeyError:
        raise ValueError(f"unknown theme {theme!r}; use one of {THEMES}") from None


def _esc(x: Any) -> str:
    return html.escape(str(x), quote=True)


def _labels(index: pd.Index) -> List[Tuple[str, ...]]:
    if isinstance(index, pd.MultiIndex):
        return [tuple(str(x) for x in t) for t in index]
    return [(str(x),) for x in index]


def heat_color(t: float, *, negative: bool = False, theme: str = "light") -> str:
    """Sequential ramp for one theme: pale/dark base -> deep blue (or red for negatives)."""
    p = _pal(theme)
    t = 0.0 if not (t == t) else max(0.0, min(1.0, t))
    (r0, g0, b0), (r1, g1, b1) = (p["heat_neg_from"], p["heat_neg_to"]) if negative else (p["heat_pos_from"], p["heat_pos_to"])
    r, g, b = int(round(r0 + (r1 - r0) * t)), int(round(g0 + (g1 - g0) * t)), int(round(b0 + (b1 - b0) * t))
    return f"rgb({r},{g},{b})"


def _text_color(t: float, *, theme: str = "light") -> str:
    p = _pal(theme)
    return p["text_on_heat_dark"] if t > p["heat_text_switch"] else p["text_on_heat_light"]


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


def expected_independence(values: np.ndarray) -> np.ndarray:
    """Expected cell values under independence of the row and column axes: the classic
    contingency-table expectation ``row_total x col_total / grand_total``, applied to any
    additive (sum/count) 2-D table — not just frequency counts."""
    v = np.where(np.isfinite(values), values, 0.0).astype(float)
    grand = v.sum()
    if grand == 0 or v.ndim != 2:
        return np.zeros_like(v)
    row_sums = v.sum(axis=1, keepdims=True)
    col_sums = v.sum(axis=0, keepdims=True)
    return (row_sums @ col_sums) / grand


def surprise_residuals(values: np.ndarray) -> np.ndarray:
    """Pearson-style residual per cell: ``(observed - expected) / sqrt(|expected|)`` under
    independence. Positive = more than expected, negative = less; magnitude ~ how surprising."""
    v = np.where(np.isfinite(values), values, 0.0).astype(float)
    expected = expected_independence(v)
    denom = np.sqrt(np.maximum(np.abs(expected), 1.0))
    return (v - expected) / denom


def _heat_fields(values: np.ndarray, mode: str) -> Tuple[np.ndarray, np.ndarray]:
    """(shade in [0,1], negative mask) for colouring, for any ``heat`` mode including
    ``"surprise"`` (residual from independence rather than raw magnitude)."""
    if mode == "surprise":
        resid = surprise_residuals(values)
        peak = np.nanmax(np.abs(resid)) if np.isfinite(resid).any() else 0.0
        shade = np.abs(resid) / peak if peak > 0 else np.zeros_like(resid)
        return shade, resid < 0
    return _scale(values, mode), values < 0


_GROUP_AGG = {"sum": "sum", "count": "sum", "nunique": "sum", "min": "min", "max": "max", "mean": "mean", "median": "median", "std": None}


def _group_reduce(block: pd.DataFrame, agg: str) -> Optional[np.ndarray]:
    """Subtotal of a group's rows under the table's aggregation (None when it has no meaning)."""
    how = _GROUP_AGG.get(agg, "sum")
    if how is None:
        return None
    vals = block.to_numpy(dtype=float)
    with np.errstate(all="ignore"):
        if how == "sum":
            return np.nansum(np.where(np.isfinite(vals), vals, 0.0), axis=0)
        if how == "min":
            return np.nanmin(vals, axis=0)
        if how == "max":
            return np.nanmax(vals, axis=0)
        if how == "median":
            return np.nanmedian(vals, axis=0)
        return np.nanmean(vals, axis=0)


def pivot_html(
    table: pd.DataFrame,
    *,
    title: Optional[str] = None,
    heat: str = "table",  # "table" | "column" | "row" | "surprise" | "none"
    bars: bool = False,
    totals: bool = False,
    compact: bool = False,
    max_rows: Optional[int] = None,
    subtotals: bool = False,
    outline: bool = False,
    agg: str = "sum",
    theme: str = "light",
) -> str:
    """Pivot table as an HTML heatmap. Cells carry the exact value as a tooltip.

    ``heat="surprise"`` colours by how far each cell is from what independence of the row
    and column axes would predict (a Pearson residual against ``row_total x col_total /
    grand_total``), rather than by raw magnitude — a table can look uneven under
    ``heat="table"`` while nothing in it is actually surprising, or vice versa. See
    :func:`pivot2hist.View.anomalies` for the same computation as a ranked list.

    With nested rows, ``subtotals`` adds a subtotal row after every outer group (using the
    table's ``agg``: sums add up, min/max/mean roll up accordingly) and ``outline`` draws
    each group as a collapsible block whose header carries the subtotal, so rows within
    rows can be folded and unfolded in the notebook without any JavaScript. ``theme`` is
    ``"light"`` (default) or ``"graphite"`` (dark).
    """
    p = _pal(theme)
    if table.empty:
        return _wrap(title, f"<div style='color:{p['empty_text']};padding:8px'>(empty)</div>", theme=theme)
    if outline and table.index.nlevels > 1:
        return _wrap(title, _outline_html(table, heat=heat, compact=compact, agg=agg, totals=totals, theme=theme), theme=theme)
    t = table
    if max_rows is not None and len(t) > max_rows:
        t = t.head(max_rows)
    values = t.to_numpy(dtype=float)
    shade, neg = _heat_fields(values, heat) if heat != "none" else (np.zeros_like(values), values < 0)
    row_labels = _labels(t.index)
    col_labels = _labels(t.columns)
    n_row_levels = t.index.nlevels
    n_col_levels = t.columns.nlevels
    row_names = [str(n) if n is not None else "" for n in t.index.names]
    col_names = [str(n).rstrip() if n is not None else "" for n in t.columns.names]

    out: List[str] = []
    out.append(
        f"<table style='border-collapse:separate;border-spacing:0;{FONT};font-size:12px;"
        f"border:1px solid {p['border']};border-radius:{p['panel_radius']};overflow:hidden'>"
    )
    # ---- column headers, one row per column level, with colspan grouping
    for lvl in range(n_col_levels):
        out.append("<tr>")
        for r in range(n_row_levels):
            head = col_names[lvl] if r == n_row_levels - 1 else ""
            out.append(
                f"<th style='text-align:left;padding:4px 8px;background:{p['head_bg']};color:{p['head_muted']};font-weight:600'>{_esc(head)}</th>"
            )
        j = 0
        while j < len(col_labels):
            k = j
            while k + 1 < len(col_labels) and col_labels[k + 1][: lvl + 1] == col_labels[j][: lvl + 1]:
                k += 1
            span = k - j + 1
            out.append(
                f"<th colspan='{span}' style='text-align:right;padding:4px 8px;background:{p['head_bg']};color:{p['head_text']};"
                f"font-weight:600;border-left:1px solid {p['border_soft']}'>{_esc(col_labels[j][lvl])}</th>"
            )
            j = k + 1
        if totals:
            out.append(f"<th style='padding:4px 8px;background:{p['head_bg']};text-align:right'>" + ("total" if lvl == n_col_levels - 1 else "") + "</th>")
        out.append("</tr>")
    if any(row_names):
        out.append("<tr>")
        for r in range(n_row_levels):
            out.append(f"<th style='text-align:left;padding:2px 8px;background:{p['subhead_bg']};color:{p['subhead_text']};font-weight:500'>{_esc(row_names[r])}</th>")
        out.append(f"<th colspan='{len(col_labels) + (1 if totals else 0)}' style='background:{p['subhead_bg']}'></th></tr>")
    # ---- body
    row_sum = np.nansum(np.where(np.isfinite(values), values, 0.0), axis=1) if totals else None
    prev: Tuple[str, ...] = ()
    vmax_bar = np.nanmax(np.abs(values)) if bars and np.isfinite(values).any() else 0.0
    sub_style = f"text-align:right;padding:3px 8px;{MONO};font-weight:600;background:{p['sub_bg']};border-top:1px solid {p['sub_border']}"

    def subtotal_row(start: int, stop: int) -> str:
        block = _group_reduce(t.iloc[start:stop], agg)
        if block is None:
            return ""
        label = " / ".join(row_labels[start][:-1])
        cells = "".join(f"<td style='{sub_style}'>{_esc(fmt_cell(float(v), compact=compact))}</td>" for v in block)
        extra = f"<td style='{sub_style}'>{_esc(fmt_cell(float(np.nansum(block)), compact=compact))}</td>" if totals else ""
        return (
            f"<tr><td colspan='{n_row_levels}' style='padding:3px 8px;font-weight:600;background:{p['sub_bg']};border-top:1px solid {p['sub_border']};color:{p['sub_text']}'>"
            f"{_esc(label)} ∑</td>{cells}{extra}</tr>"
        )

    group_start = 0
    for i, labels in enumerate(row_labels):
        if subtotals and n_row_levels > 1 and i > 0 and labels[:-1] != row_labels[i - 1][:-1]:
            out.append(subtotal_row(group_start, i))
            group_start = i
        out.append("<tr>")
        for r in range(n_row_levels):
            same = prev[: r + 1] == labels[: r + 1] if prev else False
            txt = "" if (same and r < n_row_levels - 1) else labels[r]
            style = f"text-align:left;padding:3px 8px;white-space:nowrap;color:{p['cell_text']};background:{p['cell_bg']}"
            if r < n_row_levels - 1:
                style += f";font-weight:600;border-top:1px solid {p['row_border']}" if not same else ""
            out.append(f"<td style='{style}'>{_esc(txt)}</td>")
        for j in range(values.shape[1]):
            v = values[i, j]
            raw = t.iat[i, j]
            text = fmt_cell(raw, compact=compact)
            sh = float(shade[i, j])
            bg = heat_color(sh, negative=bool(neg[i, j]), theme=theme) if heat != "none" and np.isfinite(v) else p["cell_bg"]
            fg = _text_color(sh, theme=theme) if heat != "none" else p["cell_text"]
            cell = _esc(text)
            if bars and vmax_bar > 0 and np.isfinite(v):
                pct = min(100.0, abs(v) / vmax_bar * 100)
                bar = (
                    f"<span style='display:inline-block;width:60px;height:8px;background:{p['bar_track']};vertical-align:middle;margin-right:6px;border-radius:2px'>"
                    f"<span style='display:block;width:{pct:.1f}%;height:8px;background:{p['palette'][0]};border-radius:2px'></span></span>"
                )
                cell = bar + cell
            out.append(
                f"<td title='{_esc(fmt_cell(raw))}' style='text-align:right;padding:3px 8px;{MONO};white-space:nowrap;"
                f"background:{bg};color:{fg};border-left:1px solid {p['cell_border']}'>{cell}</td>"
            )
        if totals:
            out.append(f"<td style='text-align:right;padding:3px 8px;{MONO};font-weight:600;background:{p['total_bg']};color:{p['cell_text']}'>{_esc(fmt_cell(row_sum[i], compact=compact))}</td>")
        out.append("</tr>")
        prev = labels
    if subtotals and n_row_levels > 1 and len(row_labels):
        out.append(subtotal_row(group_start, len(row_labels)))
    if totals:
        col_sum = np.nansum(np.where(np.isfinite(values), values, 0.0), axis=0)
        out.append("<tr>")
        out.append(f"<td colspan='{n_row_levels}' style='padding:3px 8px;font-weight:600;background:{p['total_bg']};color:{p['cell_text']};border-top:1px solid {p['border']}'>total</td>")
        for j in range(values.shape[1]):
            out.append(f"<td style='text-align:right;padding:3px 8px;{MONO};font-weight:600;background:{p['total_bg']};color:{p['cell_text']};border-top:1px solid {p['border']}'>{_esc(fmt_cell(col_sum[j], compact=compact))}</td>")
        out.append(f"<td style='text-align:right;padding:3px 8px;{MONO};font-weight:700;background:{p['total_bg_strong']};color:{p['cell_text']};border-top:1px solid {p['border']}'>{_esc(fmt_cell(float(col_sum.sum()), compact=compact))}</td>")
        out.append("</tr>")
    out.append("</table>")
    if len(t) < len(table):
        out.append(f"<div style='color:{p['empty_text']};font-size:11px;padding:4px 2px'>... {len(table) - len(t):,} more rows</div>")
    return _wrap(title, "".join(out), theme=theme)


def _outline_html(table: pd.DataFrame, *, heat: str, compact: bool, agg: str, totals: bool, theme: str = "light", label_w: int = 190, col_w: int = 96) -> str:
    """Nested rows as collapsible ``<details>`` blocks with subtotals in the headers."""
    p = _pal(theme)
    values = table.to_numpy(dtype=float)
    shade, neg = _heat_fields(values, heat) if heat != "none" else (np.zeros_like(values), values < 0)
    col_labels = _labels(table.columns)
    n_cols = len(col_labels)
    width = label_w + col_w * (n_cols + (1 if totals else 0))
    colgroup = f"<colgroup><col style='width:{label_w}px'/>" + "".join(f"<col style='width:{col_w}px'/>" for _ in range(n_cols + (1 if totals else 0))) + "</colgroup>"
    tstyle = f"table-layout:fixed;width:{width}px;border-collapse:collapse;{FONT};font-size:12px"

    def cell(v: float, sh: float, *, bold: bool = False, negative: Optional[bool] = None) -> str:
        is_neg = (v < 0) if negative is None else negative
        bg = heat_color(sh, negative=is_neg, theme=theme) if heat != "none" and np.isfinite(v) else p["cell_bg"]
        fg = _text_color(sh, theme=theme) if heat != "none" else p["cell_text"]
        return (
            f"<td title='{_esc(fmt_cell(v))}' style='text-align:right;padding:3px 8px;{MONO};white-space:nowrap;overflow:hidden;"
            f"background:{bg};color:{fg};font-weight:{600 if bold else 400};border-left:1px solid {p['cell_border']}'>{_esc(fmt_cell(v, compact=compact))}</td>"
        )

    def shade_of(row: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if heat == "none":
            return np.zeros(len(row)), row < 0
        return _heat_fields(row.reshape(1, -1), heat if heat != "surprise" else "table")[0][0], row < 0

    def block(sub: pd.DataFrame, sub_shade: np.ndarray, sub_neg: np.ndarray, depth: int) -> str:
        parts: List[str] = []
        if sub.index.nlevels == 1:
            rows = []
            for i, lab in enumerate(sub.index):
                cells = "".join(cell(float(sub.iat[i, j]), float(sub_shade[i, j]), negative=bool(sub_neg[i, j])) for j in range(n_cols))
                extra = cell(float(np.nansum(np.where(np.isfinite(sub.iloc[i].to_numpy(dtype=float)), sub.iloc[i].to_numpy(dtype=float), 0.0))), 0.0, bold=True) if totals else ""
                rows.append(
                    f"<tr><td style='padding:3px 8px 3px {8 + 14 * depth}px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;"
                    f"background:{p['cell_bg']};color:{p['cell_text']}'>{_esc(str(lab))}</td>{cells}{extra}</tr>"
                )
            return f"<table style='{tstyle}'>{colgroup}{''.join(rows)}</table>"
        keys = sub.index.get_level_values(0)
        seen: List[Any] = []
        for k in keys:
            if k not in seen:
                seen.append(k)
        for k in seen:
            mask = keys == k
            inner = sub[mask].droplevel(0)
            inner_shade = sub_shade[mask]
            tot = _group_reduce(sub[mask], agg)
            if tot is not None:
                tot_shade, tot_neg = shade_of(tot)
                head_cells = "".join(cell(float(v), float(sh), bold=True, negative=bool(ng)) for v, sh, ng in zip(tot, tot_shade, tot_neg))
            else:
                head_cells = "".join("<td></td>" for _ in range(n_cols))
            extra = cell(float(np.nansum(tot)), 0.0, bold=True) if (totals and tot is not None) else ""
            summary = (
                f"<summary style='cursor:pointer;list-style:none;display:block'><table style='{tstyle}'>{colgroup}<tr style='background:{p['outline_bg']}'>"
                f"<td style='padding:4px 8px 4px {8 + 14 * depth}px;font-weight:600;color:{p['head_text']};white-space:nowrap;overflow:hidden;text-overflow:ellipsis'>"
                f"<span style='display:inline-block;width:12px;color:{p['outline_chevron']}'>▸</span>{_esc(str(k))} <span style='color:{p['outline_count']};font-weight:400'>({int(mask.sum())})</span></td>{head_cells}{extra}</tr></table></summary>"
            )
            inner_neg = (inner.to_numpy(dtype=float) < 0) if heat != "surprise" else (surprise_residuals(inner.to_numpy(dtype=float)) < 0)
            parts.append(
                f"<details open='open' style='margin:0'>{summary}<div style='border-left:2px solid {p['outline_border']};margin-left:{6 + 14 * depth}px'>"
                f"{block(inner, inner_shade, inner_neg, depth + 1)}</div></details>"
            )
        return "".join(parts)

    head = "".join(
        f"<th style='text-align:right;padding:4px 8px;background:{p['head_bg']};color:{p['head_text']};font-weight:600;white-space:nowrap;overflow:hidden'>{_esc(' / '.join(c))}</th>"
        for c in col_labels
    ) + (f"<th style='text-align:right;padding:4px 8px;background:{p['head_bg']};color:{p['head_text']}'>total</th>" if totals else "")
    names = " > ".join(str(n) for n in table.index.names if n is not None)
    header = f"<table style='{tstyle}'>{colgroup}<tr><th style='text-align:left;padding:4px 8px;background:{p['head_bg']};color:{p['head_muted']}'>{_esc(names)}</th>{head}</tr></table>"
    css = "<style>.p2h-outline details>summary::-webkit-details-marker{display:none}.p2h-outline details:not([open])>summary span:first-child{transform:rotate(0deg)}.p2h-outline details[open]>summary span:first-child{display:inline-block;transform:rotate(90deg)}</style>"
    return (
        f"<div class='p2h-outline' style='border:1px solid {p['outline_outer_border']};border-radius:{p['panel_radius']};overflow:auto;"
        f"display:inline-block;max-width:100%'>{css}{header}{block(table, shade, neg, 0)}</div>"
    )


def _wrap(title: Optional[str], body: str, *, theme: str = "light") -> str:
    p = _pal(theme)
    head = f"<div style='{FONT};font-size:12px;color:{p['head_muted']};margin:0 0 6px 0'>{_esc(title)}</div>" if title else ""
    outer = ""
    if p["page_bg"]:
        outer = f";background:{p['page_bg']};padding:{('10px' if p['panel_pad'] != '0' else '0')};border-radius:{p['panel_radius']}"
    return f"<div class='pivot2hist' data-p2h-theme='{theme}' style='display:inline-block;max-width:100%;overflow-x:auto{outer}'>{head}{body}</div>"


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
    theme: str = "light",
) -> str:
    """Histogram / bar chart as inline SVG.

    Rows are the bins on the x axis (multi-level rows are drawn as labelled groups),
    columns are the series (one colour each, side by side or stacked), and every bar
    carries a tooltip. ``density`` is an optional ``(x, y)`` curve drawn over a numeric
    axis, e.g. from :func:`pivot2hist.bin_edges`'s sibling :func:`kde`. ``theme`` is
    ``"light"`` (default) or ``"graphite"`` (dark).
    """
    p = _pal(theme)
    if table.empty:
        return _wrap(title, f"<div style='color:{p['empty_text']};padding:8px'>(empty)</div>", theme=theme)
    values = np.where(np.isfinite(table.to_numpy(dtype=float)), table.to_numpy(dtype=float), 0.0)
    n_bins, n_series = values.shape
    series = [" / ".join(l) for l in _labels(table.columns)]
    bins = _labels(table.index)
    n_levels = table.index.nlevels
    outer = [b[:-1] for b in bins]
    leaf = [b[-1] for b in bins]
    palette = p["palette"]

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

    svg_style = f"{FONT};font-size:11px;max-width:100%;height:auto"
    if p["page_bg"]:
        svg_style += f";background:{p['page_bg']};border-radius:{p['panel_radius']}"
    out: List[str] = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}' style='{svg_style}'>"
    ]
    if title:
        out.append(f"<text x='{left}' y='16' fill='{p['svg_title']}' font-size='12'>{_esc(title)}</text>")
    # y grid + ticks
    for tv in _ticks(vmax):
        y = y_of(tv)
        out.append(f"<line x1='{left}' x2='{left + plot_w}' y1='{y:.1f}' y2='{y:.1f}' stroke='{p['svg_grid']}'/>")
        out.append(f"<text x='{left - 6}' y='{y + 4:.1f}' text-anchor='end' fill='{p['svg_axis_text']}'>{_esc(human(tv))}</text>")
    out.append(f"<line x1='{left}' x2='{left + plot_w}' y1='{top + plot_h:.1f}' y2='{top + plot_h:.1f}' stroke='{p['svg_axis_line']}'/>")
    # bars
    for i in range(n_bins):
        x0 = left + i * slot + gap / 2
        base = 0.0
        for j in range(n_series):
            v = float(values[i, j])
            color = palette[j % len(palette)]
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
                    f"<text x='{x + bar_w / 2:.1f}' y='{y1 - 3:.1f}' text-anchor='middle' fill='{p['svg_value_text']}' font-size='10'>{_esc(human(abs(v)))}</text>"
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
                f"text-anchor='end' fill='{p['svg_xlabel']}'>{_esc(lab)}</text>"
            )
        else:
            out.append(f"<text x='{xc:.1f}' y='{top + plot_h + 16:.1f}' text-anchor='middle' fill='{p['svg_xlabel']}'>{_esc(lab)}</text>")
    # group brackets for multi-level rows
    if n_levels > 1:
        y = top + plot_h + label_h + 4
        i = 0
        while i < n_bins:
            k = i
            while k + 1 < n_bins and outer[k + 1] == outer[i]:
                k += 1
            xa, xb = left + i * slot + 2, left + (k + 1) * slot - 2
            out.append(f"<line x1='{xa:.1f}' x2='{xb:.1f}' y1='{y}' y2='{y}' stroke='{p['svg_group_line']}'/>")
            out.append(f"<text x='{(xa + xb) / 2:.1f}' y='{y + 12}' text-anchor='middle' fill='{p['svg_group_text']}' font-weight='600'>{_esc(' / '.join(outer[i]))}</text>")
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
            out.append(f"<polyline points='{pts}' fill='none' stroke='{p['svg_density']}' stroke-width='1.5' stroke-dasharray='4 3' opacity='0.8'><title>density (KDE)</title></polyline>")
    # legend
    if n_series > 1:
        y = height - 8
        x = left
        for j, name in enumerate(series):
            color = palette[j % len(palette)]
            out.append(f"<rect x='{x}' y='{y - 9}' width='10' height='10' fill='{color}' rx='2'/>")
            label = name if len(name) <= 22 else name[:21] + "…"
            out.append(f"<text x='{x + 14}' y='{y}' fill='{p['svg_legend_text']}'>{_esc(label)}</text>")
            x += 14 + 6.5 * len(label) + 14
            if x > width - 60:
                break
    out.append("</svg>")
    return _wrap(None, "".join(out), theme=theme)


__all__ = ["pivot_html", "hist_svg", "heat_color", "PALETTE", "PALETTE_GRAPHITE", "PALETTES", "THEMES"]
