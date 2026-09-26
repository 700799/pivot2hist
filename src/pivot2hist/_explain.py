"""Interrogate one cell of a pivot (or one bin of a histogram): the rows behind it, and
why it looks the way it does.

A cell is named by its *labels* - what the table shows - not by raw values, so the same
call works for a bin (``bytes="[1K, 2K)"``), a time bucket (``timestamp="13:00"``), a
folded top-N label (``dst_port="(other)"``), a semantic roll-up (``src_ip="10.0.1.0/24"``)
or a null (``"(null)"``): each dimension is materialized exactly as the pivot did it and
matched on the label.
"""
from __future__ import annotations

import html as _html
import json
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import _binning as B
from ._binning import human
from ._fit import Dim, materialize
from ._html import expected_independence, surprise_residuals
from ._log import log
from ._profile import BOOLEAN, CATEGORICAL, NUMERIC
from ._render import fmt_cell
from ._view import Filter, View, make_filter

_SAMPLE_CAP = 50_000
_EPS_SHARE = 0.005  # half a percent: keeps a 0%-elsewhere value from reading as infinitely over-represented


# --------------------------------------------------------------------------- naming a cell


def _label_filter(dim: Dim, label: Any) -> Filter:
    """A slice keeping the rows whose materialized level for ``dim`` is ``label``."""
    lab = str(label)

    def hit(s: pd.Series) -> np.ndarray:
        m = materialize(s.to_frame(dim.column), dim)
        out = np.asarray(m.astype(str) == lab, dtype=bool)
        if lab == B.NULL:
            out = out | np.asarray(m.isna(), dtype=bool)
        return out

    return Filter(dim.column, "callable", hit, f"{dim.column}={lab}")


def parse_cell(view: View, args: Sequence[Any], labels: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """``(axis labels by column, other column=spec slices)`` from the ``cell()`` forms:
    positional ``(row_label[, col_label])`` - a tuple per axis for nested levels, ``None``
    to leave an axis open - and/or ``column=label`` keywords."""
    layout = view.layout
    dim_cols = [d.column for d in layout.dims]
    given: Dict[str, Any] = {}
    if len(args) > 2:
        raise TypeError("cell takes (row_label[, col_label]) and/or column=label keywords")
    for axis_dims, lab in zip((layout.rows, layout.cols), args):
        if lab is None:
            continue
        labs = lab if isinstance(lab, tuple) else (lab,)
        if len(labs) > len(axis_dims):
            raise ValueError(f"{len(labs)} label(s) given for an axis with {len(axis_dims)} level(s): {[d.label for d in axis_dims]}")
        for d, x in zip(axis_dims, labs):
            if x is not None:
                given[d.column] = x
    extra: Dict[str, Any] = {}
    for col, lab in labels.items():
        if col in dim_cols:
            given[col] = lab
        else:
            extra[col] = lab
    if not given and not extra:
        raise TypeError("say which cell: v.cell(row_label, col_label) or v.cell(column=label)")
    return given, extra


def cell_filters(view: View, given: Dict[str, Any], extra: Dict[str, Any]) -> Tuple[Filter, ...]:
    out: List[Filter] = []
    for d in view.layout.dims:
        if d.column in given and not any(f.column == d.column for f in out):
            out.append(_label_filter(d, given[d.column]))
    if extra:
        base = view.data
        for col, spec in extra.items():
            out.append(make_filter(col, spec, base))
    return tuple(out)


def cell_view(view: View, args: Sequence[Any], labels: Dict[str, Any]) -> Tuple[View, Dict[str, Any]]:
    given, extra = parse_cell(view, args, labels)
    filters = cell_filters(view, given, extra)
    return view._clone(filters=view.filters + filters), given


def rows(view: View, args: Sequence[Any], n: Optional[int], labels: Dict[str, Any]) -> pd.DataFrame:
    cv, _ = cell_view(view, args, labels)
    if cv.paged is not None:
        return cv.materialize(max_rows=n).data
    df = cv.data
    return df if n is None else df.head(int(n))


# --------------------------------------------------------------------------- explaining it


def _axis_mask(index: pd.Index, dims: Sequence[Dim], given: Dict[str, Any]) -> np.ndarray:
    mask = np.ones(len(index), dtype=bool)
    for lvl, d in enumerate(dims):
        if d.column not in given or lvl >= index.nlevels:
            continue
        vals = index.get_level_values(lvl) if isinstance(index, pd.MultiIndex) else index
        lab = str(given[d.column])
        mask &= np.fromiter((str(x) == lab for x in vals), dtype=bool, count=len(index))
    return mask


def _sample(df: pd.DataFrame) -> pd.DataFrame:
    return df.sample(_SAMPLE_CAP, random_state=0) if len(df) > _SAMPLE_CAP else df


def _fmt_ratio(r: float) -> str:
    if math.isinf(r):
        return "only here"
    return f"×{r:.2g}" if r < 10 else f"×{r:,.0f}"


def distinguishing(cell: pd.DataFrame, rest: pd.DataFrame, profile: Any, *, exclude: Sequence[str] = (), k: int = 5) -> List[Dict[str, Any]]:
    """What sets ``cell``'s rows apart from ``rest``, best first: for a label column, the
    value most over-represented in the cell (its share here vs elsewhere, as a lift); for
    a numeric column, the median here vs elsewhere. Columns that name the cell are skipped.
    Scores: a label's share in the cell times ``|log2 lift|`` (common *and* distinctive
    wins); a numeric's ``|log2(median ratio)|``. Small differences are dropped."""
    if cell.empty or rest.empty:
        return []
    cell, rest = _sample(cell), _sample(rest)
    skip = set(exclude)
    feats: List[Dict[str, Any]] = []
    for cp in profile:
        name = cp.name
        if name in skip or name not in cell.columns or name not in rest.columns:
            continue
        if cp.kind in (CATEGORICAL, BOOLEAN):
            vc = B._hashable_column(cell[name]).value_counts(normalize=True, dropna=False).head(3)
            vr = B._hashable_column(rest[name]).value_counts(normalize=True, dropna=False)
            for val, sc in vc.items():
                sr = float(vr.get(val, 0.0))
                sc = float(sc)
                lift = (sc + _EPS_SHARE) / (sr + _EPS_SHARE)
                if 0.8 <= lift <= 1.25 or sc < 0.05:
                    continue
                shown = B.NULL if (isinstance(val, float) and np.isnan(val)) or val is None else str(val)
                feats.append({
                    "column": name, "kind": "label", "value": shown,
                    "share_in_cell": round(sc, 4), "share_in_rest": round(sr, 4), "lift": round(lift, 3),
                    "score": round(sc * abs(math.log2(lift)), 4),
                    "text": f"`{name}` is {shown!r} for {sc:.0%} of these rows vs {sr:.0%} elsewhere ({_fmt_ratio(lift)})",
                })
        elif cp.kind == NUMERIC:
            xc = pd.to_numeric(cell[name], errors="coerce").to_numpy(dtype=float)
            xr = pd.to_numeric(rest[name], errors="coerce").to_numpy(dtype=float)
            xc, xr = xc[np.isfinite(xc)], xr[np.isfinite(xr)]
            if xc.size < 3 or xr.size < 3:
                continue
            mc, mr = float(np.median(xc)), float(np.median(xr))
            if mc == mr or (mc <= 0 and mr <= 0):
                continue
            if mr > 0 and mc > 0:
                ratio = mc / mr
            elif mr == 0 and mc > 0:
                ratio = math.inf
            elif mc == 0 and mr > 0:
                ratio = 0.0
            else:
                continue  # signs differ: a ratio would mislead
            score = min(abs(math.log2(ratio)) if 0 < ratio < math.inf else 6.0, 6.0)
            if score < 0.15:
                continue
            feats.append({
                "column": name, "kind": "numeric", "median_in_cell": mc, "median_in_rest": mr,
                "ratio": (None if math.isinf(ratio) else round(ratio, 4)), "score": round(score, 4),
                "text": f"`{name}` median {human(mc)} here vs {human(mr)} elsewhere ({_fmt_ratio(ratio) if ratio else '×0'})",
            })
    feats.sort(key=lambda f: -f["score"])
    return feats[: max(0, int(k))]


def _records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    flat = df.reset_index(drop=True)
    flat.columns = [str(c) for c in flat.columns]
    return json.loads(flat.to_json(orient="records", date_format="iso", default_handler=str))


class Explanation(dict):
    """What :meth:`View.explain` returns: a plain dict (JSON-safe, so it travels through
    the agent surface unchanged) that also reads well in a notebook."""

    def __str__(self) -> str:
        return str(self.get("text", ""))

    __repr__ = __str__

    def _repr_html_(self) -> str:
        e = _html.escape
        where = " × ".join(f"{k}={v}" for k, v in self["cell"].items())
        facts: List[Tuple[str, str]] = [(self["measure"], fmt_cell(self.get("observed")))]
        if self.get("expected") is not None:
            facts.append(("expected under independence", fmt_cell(self["expected"])))
            if self.get("ratio_to_expected") is not None:
                facts.append(("observed / expected", f"×{self['ratio_to_expected']:.2g} ({self['direction']})"))
        for key, lab in (("share_of_row", "share of its row"), ("share_of_col", "share of its column"), ("share_of_total", "share of the table")):
            if self.get(key) is not None:
                facts.append((lab, f"{self[key]:.1%}"))
        if self.get("rank") is not None:
            facts.append(("rank", f"{self['rank']} of {self['n_cells']} cells"))
        facts.append(("rows", f"{self['n_rows']:,} of {self['rows_total']:,}" + (" (sampled)" if self.get("sampled") else "")))
        fact_rows = "".join(f"<tr><td style='padding:2px 10px 2px 0;color:#667'>{e(k)}</td><td style='padding:2px 0;font-family:ui-monospace,Menlo,Consolas,monospace'>{e(v)}</td></tr>" for k, v in facts)
        out = ["<div style='font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;font-size:12px'>",
               f"<div style='font-weight:600;margin-bottom:4px'>{e(where)}</div>",
               f"<table style='border-collapse:collapse;margin-bottom:6px'>{fact_rows}</table>"]
        if self.get("distinguishing"):
            items = "".join(f"<li>{e(f['text'])}</li>" for f in self["distinguishing"])
            out.append(f"<div style='color:#667'>what sets these rows apart</div><ul style='margin:2px 0 6px 18px;padding:0'>{items}</ul>")
        elif self["n_rows"] == self["rows_total"]:
            out.append("<div style='color:#889'>these are all the rows in view - nothing to contrast against</div>")
        recs = self.get("rows") or []
        if recs:
            cols = list(recs[0].keys())
            head = "".join(f"<th style='text-align:left;padding:2px 8px;background:#f5f6f8;font-weight:600'>{e(c)}</th>" for c in cols)
            body = "".join("<tr>" + "".join(f"<td style='padding:2px 8px;white-space:nowrap;border-top:1px solid #f0f0f0'>{e('' if r[c] is None else str(r[c]))}</td>" for c in cols) + "</tr>" for r in recs)
            out.append(f"<div style='color:#667;margin-top:4px'>first {len(recs)} row(s)</div>"
                       f"<div style='overflow-x:auto'><table style='border-collapse:collapse;font-size:11px'><tr>{head}</tr>{body}</table></div>")
        out.append("</div>")
        return "".join(out)


def _text(out: Dict[str, Any]) -> str:
    where = " × ".join(f"{k}={v}" for k, v in out["cell"].items())
    parts = [f"{where}: {out['measure']} = {fmt_cell(out.get('observed'))}"]
    if out.get("ratio_to_expected") is not None:
        parts.append(f"×{out['ratio_to_expected']:.2g} what independence of the axes predicts ({fmt_cell(out['expected'])}) - {out['direction']}")
    shares = [f"{out[k]:.0%} of its {lab}" for k, lab in (("share_of_row", "row"), ("share_of_col", "column")) if out.get(k) is not None]
    if out.get("share_of_total") is not None:
        shares.append(f"{out['share_of_total']:.1%} of the table")
    if shares:
        parts.append(", ".join(shares))
    if out.get("rank") is not None:
        parts.append(f"rank {out['rank']} of {out['n_cells']} cells")
    parts.append(f"{out['n_rows']:,} of {out['rows_total']:,} rows" + (" (sampled)" if out.get("sampled") else ""))
    text = " · ".join(parts)
    if out.get("distinguishing"):
        text += "\nWhat sets these rows apart: " + "; ".join(f["text"] for f in out["distinguishing"]) + "."
    elif out["n_rows"] == out["rows_total"]:
        text += "\n(these are all the rows in view - nothing to contrast against)"
    return text


def explain(view: View, args: Sequence[Any], labels: Dict[str, Any], *, n_rows: int = 10, k: int = 5) -> Explanation:
    given, extra = parse_cell(view, args, labels)
    filters = cell_filters(view, given, extra)
    layout = view.layout
    with log.step("explain", " x ".join(f"{c}={v}" for c, v in given.items()) or "cell"):
        table = view.table()
        values = table.to_numpy(dtype=float)
        rm = _axis_mask(table.index, layout.rows, given)
        cm = _axis_mask(table.columns, layout.cols, given) if (layout.cols and values.ndim == 2) else np.ones(values.shape[1] if values.ndim == 2 else 1, dtype=bool)
        if values.size == 0 or not rm.any() or not cm.any():
            raise KeyError(
                f"no cell matches {given}; row labels look like {[str(x) for x in table.index[:5]]}, "
                f"column labels like {[str(x) for x in table.columns[:5]]}"
            )
        block = values[np.ix_(rm, cm)]
        single = block.size == 1
        agg = layout.agg if layout.values is not None else "count"
        additive = agg in ("sum", "count")
        finite = np.isfinite(values)
        observed: Optional[float]
        if additive:
            observed = float(np.nansum(np.where(np.isfinite(block), block, 0.0)))
        else:
            observed = float(block[0, 0]) if (single and np.isfinite(block[0, 0])) else None
        cell_labels = {c: (B.NULL if v is None else str(v)) for c, v in given.items()}
        for col, spec in extra.items():
            cell_labels[col] = str(spec)
        vd = view.data
        mask = np.ones(len(vd), dtype=bool)
        for f in filters:
            mask &= f.mask(vd)
        cell_df, rest_df = vd[mask], vd[~mask]
        out: Dict[str, Any] = {
            "cell": cell_labels, "measure": layout.measure, "layout": layout.describe(), "mode": view.mode,
            "observed": observed, "single_cell": single,
            "n_rows": int(len(cell_df)), "rows_total": int(len(vd)),
        }
        if view.paged is not None:
            out["sampled"] = True
        # a block spanning a whole axis equals its own marginal: expectation and that share are trivial
        sub_block = values.ndim == 2 and not rm.all() and not cm.all()
        if additive and sub_block and values.shape[0] >= 2 and values.shape[1] >= 2 and observed is not None:
            exp = expected_independence(values)
            e = float(np.nansum(exp[np.ix_(rm, cm)]))
            out["expected"] = e
            out["ratio_to_expected"] = (observed / e) if e > 0 else None
            out["direction"] = "over" if observed > e else ("under" if observed < e else "as expected")
            if single:
                out["residual"] = float(surprise_residuals(values)[np.ix_(rm, cm)][0, 0])
        if additive and observed is not None:
            safe = np.where(finite, values, 0.0)
            grand, row_tot, col_tot = float(safe.sum()), float(safe[rm, :].sum()), float(safe[:, cm].sum())
            if not cm.all():
                out["share_of_row"] = observed / row_tot if row_tot else None
            if not rm.all():
                out["share_of_col"] = observed / col_tot if col_tot else None
            out["share_of_total"] = observed / grand if grand else None
        if single and observed is not None:
            out["rank"] = int((values[finite] > observed).sum()) + 1
            out["n_cells"] = int(finite.sum())
        out["distinguishing"] = distinguishing(cell_df, rest_df, view.profile, exclude=list(given) + list(extra), k=k)
        out["rows"] = _records(cell_df.head(max(0, int(n_rows))))
        out["text"] = _text(out)
    return Explanation(out)


__all__ = ["Explanation", "explain", "rows", "cell_view", "distinguishing"]
