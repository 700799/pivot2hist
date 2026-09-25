"""Package a :class:`~pivot2hist.View` as context for an LLM: a short natural-language
description, compact structured metadata, and the table itself as GitHub-flavored
markdown - the three things a model needs to reason about the data without re-deriving
them from a raw dump, and without spending tokens on a full ``repr`` or a wall of JSON.

Markdown is the deliberate choice for the table: it's what every current model has seen
the most of, it's far more token-efficient than a list of per-cell dicts, and unlike HTML
it needs no dependency (no ``tabulate``, no ``lxml``) to produce.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def _esc(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ").strip()


def _flat_labels(idx: pd.Index) -> List[str]:
    if isinstance(idx, pd.MultiIndex):
        return [_esc(" / ".join(str(x) for x in tup)) for tup in idx]
    return [_esc(str(x)) for x in idx]


def _fmt_cell(v: Any) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, (float, np.floating)):
        return _esc(f"{v:,.4g}")
    return _esc(str(v))


def markdown_table(df: pd.DataFrame, *, max_rows: int = 30, max_cols: int = 12) -> Tuple[str, bool]:
    """``df`` as a GitHub-flavored markdown table, capped to ``max_rows`` rows and
    ``max_cols`` columns - a truncation, not a sample, so whatever ordering the caller
    already chose (top-N, sorted, ranked) survives. Returns ``(markdown, truncated)``.
    """
    if df.shape[0] == 0 or df.shape[1] == 0:
        return "_(empty table)_", False
    truncated = df.shape[0] > max_rows or df.shape[1] > max_cols
    shown = df.iloc[: max(1, max_rows), : max(1, max_cols)]
    row_labels = _flat_labels(shown.index)
    col_labels = _flat_labels(shown.columns)
    index_name = _esc(" / ".join(str(n) for n in (shown.index.names or ()) if n)) or " "
    lines = [
        "| " + index_name + " | " + " | ".join(col_labels) + " |",
        "|" + "---|" * (len(col_labels) + 1),
    ]
    for label, (_, row) in zip(row_labels, shown.iterrows()):
        lines.append("| " + label + " | " + " | ".join(_fmt_cell(v) for v in row) + " |")
    if truncated:
        lines.append(f"\n_truncated to {shown.shape[0]} of {df.shape[0]} rows x {shown.shape[1]} of {df.shape[1]} columns_")
    return "\n".join(lines), truncated


def llm_context(view: Any, *, max_rows: int = 30, max_cols: int = 12, notes: bool = True) -> Dict[str, Any]:
    """``{"description", "metadata", "table"}`` for ``view`` - see :func:`pivot2hist.llm_context`."""
    layout = view.layout
    prof = view.profile
    table = view.table()
    md, truncated = markdown_table(table, max_rows=max_rows, max_cols=max_cols)

    columns_meta: Dict[str, Any] = {}
    for name in layout.columns_used:
        if name in prof:
            c = prof[name]
            columns_meta[name] = {
                "kind": c.kind, "semantic": c.semantic, "nunique": int(c.nunique),
                "null_frac": round(c.null_frac, 4),
            }

    metadata: Dict[str, Any] = {
        "mode": view.mode,
        "measure": layout.measure,
        "rows": [d.label for d in layout.rows],
        "cols": [d.label for d in layout.cols],
        "shape": {"rows": int(table.shape[0]), "cols": int(table.shape[1])},
        "source_rows": int(len(view.data)),
        "slices": list(view.slices),
        "columns": columns_meta,
        "truncated": truncated,
    }
    if view.paged is not None:
        metadata["paged"] = True
        metadata["approximate"] = view.approximate

    if notes:
        notable: Dict[str, Any] = {}
        try:
            top = view.anomalies(3)
            if not top.empty:
                notable["anomalies"] = top.to_dict("records")
        except Exception:
            pass  # anomalies() is a bonus fact, not required (needs a 2x2+ additive pivot)
        if notable:
            metadata["notable"] = notable

    return {"description": _describe(view, metadata), "metadata": metadata, "table": md}


def _describe(view: Any, metadata: Dict[str, Any]) -> str:
    layout = view.layout
    kind = "Histogram" if view.is_hist else "Pivot table"
    parts = [f"{kind} of {metadata['source_rows']:,} rows: {layout.describe()}."]
    shape = metadata["shape"]
    if shape["rows"] or shape["cols"]:
        tail = " (table below is truncated)." if metadata["truncated"] else "."
        parts.append(f"Shown as {shape['rows']} row(s) x {shape['cols']} column(s){tail}")
    if metadata["slices"]:
        parts.append("Sliced to: " + "; ".join(metadata["slices"]) + ".")
    if metadata.get("paged"):
        if metadata.get("approximate"):
            parts.append("Source is larger than the memory budget and paged; this measure was approximated from a sample.")
        else:
            parts.append("Source is larger than the memory budget and paged, but this measure is exact across all pages.")
    anomalies = metadata.get("notable", {}).get("anomalies")
    if anomalies:
        a = anomalies[0]
        parts.append(
            f"Most surprising cell: {a['row']} / {a['col']} ({a['direction']} expectation - observed "
            f"{a['observed']:,.4g} vs. expected {a['expected']:,.4g})."
        )
    return " ".join(parts)


__all__ = ["llm_context", "markdown_table"]
