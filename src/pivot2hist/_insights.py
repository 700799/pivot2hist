"""Rich, local, non-LLM analysis of whatever data is currently in view: distributions,
modality, skew, concentration, outliers, correlated columns, surprising pivot cells -
scored and ranked, not just listed. Nothing here calls a language model or scipy/sklearn;
every score is a plain numpy/pandas computation, cheap enough to re-run on every filter
change (see ``View.insights()``'s ``sensitivity`` knob for how many findings surface).

"mixle inspired": the modality check reuses the from-scratch Gaussian-mixture fitting in
:mod:`pivot2hist._mixture` (BIC-selected component count) to notice a column with two or
more distinct populations - the same automatic-model-selection idea :mod:`_density`
already borrows for single-family fits, applied here to "how many groups is this,
really?" instead of "what shape is this?".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ._profile import BOOLEAN, CATEGORICAL, CONSTANT, DATETIME, ID, NUMERIC, ColumnProfile, Profile

_SAMPLE_CAP = 20_000  # rows used for the expensive per-column fits (distribution, mixture)


@dataclass(frozen=True)
class Insight:
    """One ranked observation. ``significance`` is 0..1 - see :func:`insights` for how
    it's computed per ``kind`` and how ``sensitivity`` filters on it."""

    kind: str
    columns: Tuple[str, ...]
    significance: float
    text: str
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind, "columns": list(self.columns),
            "significance": round(self.significance, 3), "text": self.text, "detail": self.detail,
        }


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _numeric_sample(s: pd.Series, seed: int) -> np.ndarray:
    x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if x.size > _SAMPLE_CAP:
        rng = np.random.default_rng(seed)
        x = rng.choice(x, _SAMPLE_CAP, replace=False)
    return x


def _skewness(x: np.ndarray) -> float:
    if x.size < 3:
        return 0.0
    m = x.mean()
    std = x.std()
    if std <= 0:
        return 0.0
    return float(np.mean(((x - m) / std) ** 3))


def _column_summary(df: pd.DataFrame, cp: ColumnProfile, *, seed: int) -> Dict[str, Any]:
    """The always-shown facts about one column, no thresholding."""
    out: Dict[str, Any] = {
        "kind": cp.kind, "semantic": cp.semantic, "nunique": int(cp.nunique),
        "null_frac": round(cp.null_frac, 4),
    }
    if cp.kind == NUMERIC:
        x = _numeric_sample(df[cp.name], seed)
        if x.size:
            out.update(
                mean=round(float(x.mean()), 6), median=round(float(np.median(x)), 6),
                std=round(float(x.std()), 6), min=round(float(x.min()), 6), max=round(float(x.max()), 6),
            )
            try:
                from ._density import fit_distribution

                fit = fit_distribution(x)
                if fit is not None:
                    out["distribution"] = fit.describe()
            except Exception:
                pass
    elif cp.kind in (CATEGORICAL, BOOLEAN):
        try:
            vc = df[cp.name].value_counts(dropna=True)
            if len(vc):
                out["top_value"] = str(vc.index[0])
                out["top_share"] = round(float(vc.iloc[0] / vc.sum()), 4)
        except Exception:
            pass
    elif cp.kind == DATETIME:
        s = pd.to_datetime(df[cp.name], errors="coerce").dropna()
        if len(s):
            out["min"] = str(s.min())
            out["max"] = str(s.max())
        if cp.is_time_series and cp.ts_freq:
            out["sampled_every"] = cp.ts_freq
    return out


def _finding_skew(name: str, cp: ColumnProfile, df: pd.DataFrame, seed: int) -> Optional[Insight]:
    x = _numeric_sample(df[name], seed)
    if x.size < 20:
        return None
    g1 = _skewness(x)
    sig = _clip01(abs(g1) / 3.0)
    if sig <= 0:
        return None
    direction = "right" if g1 > 0 else "left"
    tail = "a few very large values pull the mean up" if g1 > 0 else "a few very small values pull the mean down"
    return Insight(
        "skew", (name,), sig,
        f"`{name}` is strongly {direction}-skewed (skewness {g1:.2f}) - {tail}; median is a better "
        f"summary than mean here.",
        {"skewness": round(g1, 3)},
    )


def _finding_concentration(name: str, cp: ColumnProfile, df: pd.DataFrame) -> Optional[Insight]:
    try:
        vc = df[name].value_counts(dropna=True)
    except Exception:
        return None
    if len(vc) < 2:
        return None
    total = int(vc.sum())
    if total == 0:
        return None
    top_share = float(vc.iloc[0] / total)
    expected = 1.0 / len(vc)
    sig = _clip01((top_share - expected) / max(1e-9, 1 - expected))
    if sig <= 0:
        return None
    return Insight(
        "concentration", (name,), sig,
        f"`{name}` is dominated by {vc.index[0]!r}: {top_share:.0%} of rows, vs. {expected:.0%} expected "
        f"if the {len(vc)} values were even.",
        {"top_value": str(vc.index[0]), "top_share": round(top_share, 4), "nunique": len(vc)},
    )


def _finding_modality(name: str, cp: ColumnProfile, df: pd.DataFrame, seed: int) -> Optional[Insight]:
    x = _numeric_sample(df[name], seed)
    if x.size < 40:
        return None
    try:
        from ._mixture import choose_gmm_k, fit_gmm

        k = choose_gmm_k(x.reshape(-1, 1), k_max=4, seed=seed)
        if k < 2:
            return None
        fit = fit_gmm(x.reshape(-1, 1), k, seed=seed)
        comps = fit.components()
    except Exception:
        return None
    weights = sorted((c["weight"] for c in comps), reverse=True)
    sig = _clip01(2 * weights[-1] if len(weights) >= 2 else 0.0)  # balanced split -> high; one tiny sliver -> low
    if sig <= 0:
        return None
    parts = ", ".join(f"~{c['mean']:.3g} ({c['weight']:.0%})" for c in comps)
    return Insight(
        "modality", (name,), sig,
        f"`{name}` looks like {len(comps)} distinct groups, not one: {parts}.",
        {"components": comps},
    )


def _finding_nulls(name: str, cp: ColumnProfile) -> Optional[Insight]:
    if cp.null_frac <= 0:
        return None
    sig = _clip01(cp.null_frac)
    return Insight(
        "nulls", (name,), sig,
        f"`{name}` is {cp.null_frac:.0%} null.",
        {"null_frac": round(cp.null_frac, 4)},
    )


def _finding_constant(name: str, cp: ColumnProfile, df: pd.DataFrame) -> Optional[Insight]:
    if cp.kind == CONSTANT or cp.nunique <= 1:
        return Insight("constant", (name,), 1.0, f"`{name}` has a single value for every row - useless as a dimension or measure.", {})
    if cp.kind != NUMERIC:
        return None
    x = _numeric_sample(df[name], 0)
    if x.size < 10:
        return None
    # the middle 80% of the data, not min/max: a couple of extreme outliers on an
    # otherwise skewed-but-genuinely-varying column (bytes, latency ...) must not make
    # std/range look tiny and falsely read as "constant" - see test_insights.py.
    lo, hi = np.percentile(x, [10, 90])
    spread = hi - lo
    scale = max(abs(float(np.median(x))), abs(hi), abs(lo), 1e-9)
    ratio = spread / scale
    sig = _clip01(1 - ratio / 0.05)
    if sig <= 0.3:
        return None
    return Insight(
        "constant", (name,), sig,
        f"`{name}` barely varies (the middle 80% of values span only {ratio:.1%} of its typical "
        "magnitude) - close to constant.",
        {"p10_p90_over_scale": round(float(ratio), 4)},
    )


def _finding_high_cardinality(name: str, cp: ColumnProfile) -> Optional[Insight]:
    if cp.kind not in (CATEGORICAL, BOOLEAN) or cp.n < 20:
        return None
    ratio = cp.unique_ratio
    sig = _clip01((ratio - 0.5) * 2)  # only worth flagging once more than half the rows are distinct
    if sig <= 0:
        return None
    return Insight(
        "high_cardinality", (name,), sig,
        f"`{name}` has {cp.nunique:,} distinct values across {cp.n:,} rows ({ratio:.0%}) - "
        "looks more like an identifier than a category to group by.",
        {"nunique": int(cp.nunique), "unique_ratio": round(ratio, 4)},
    )


def _finding_outliers(name: str, cp: ColumnProfile, df: pd.DataFrame, seed: int) -> Optional[Insight]:
    x = _numeric_sample(df[name], seed)
    if x.size < 20:
        return None
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    if iqr <= 0:
        return None
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    frac = float(np.mean((x < lo) | (x > hi)))
    sig = _clip01(frac / 0.15)
    if sig <= 0:
        return None
    return Insight(
        "outliers", (name,), sig,
        f"`{name}` has {frac:.1%} of values outside 1.5x the interquartile range "
        f"(outside [{lo:.3g}, {hi:.3g}]).",
        {"outlier_frac": round(frac, 4), "lo": round(float(lo), 4), "hi": round(float(hi), 4)},
    )


def _finding_correlations(df: pd.DataFrame, columns: Sequence[str], *, max_pairs: int = 5) -> List[Insight]:
    if len(columns) < 2:
        return []
    try:
        from ._deps import mutual_info_matrix

        m = mutual_info_matrix(df, list(columns))
    except Exception:
        return []
    pairs = []
    cols = list(m.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = float(m.iat[i, j])
            if v > 0:
                pairs.append((v, cols[i], cols[j]))
    pairs.sort(key=lambda t: -t[0])
    out = []
    for v, a, b in pairs[:max_pairs]:
        out.append(Insight(
            "correlation", (a, b), _clip01(v),
            f"`{a}` and `{b}` move together (normalized mutual information {v:.2f}).",
            {"mutual_info": round(v, 4)},
        ))
    return out


def _finding_anomalies(view: Any, *, max_cells: int = 3) -> List[Insight]:
    try:
        top = view.anomalies(max_cells)
    except Exception:
        return []
    if top.empty:
        return []
    max_abs = float(top["residual"].abs().max()) or 1.0
    out = []
    for r in top.itertuples():
        sig = _clip01(abs(r.residual) / max(10.0, max_abs))
        out.append(Insight(
            "anomaly", (), sig,
            f"{r.row} / {r.col} is {r.direction} expectation: observed {r.observed:,.4g} vs. "
            f"expected {r.expected:,.4g} (residual {r.residual:+.1f}).",
            {"row": r.row, "col": r.col, "observed": r.observed, "expected": r.expected, "residual": r.residual},
        ))
    return out


def _finding_spikes(view: Any, *, max_items: int = 3) -> List[Insight]:
    """The strongest per-row movements over the first datetime column (see
    :meth:`View.spikes`); nothing when the view has no time column to scan, or its rows
    are that column's own buckets."""
    from ._spikes import default_time_column, describe_spike
    from ._spikes import spikes as _spikes

    column = default_time_column(view)
    if column is None or not view.layout.rows:
        return []
    try:
        top = _spikes(view, column, n=max_items)
    except Exception:
        return []
    out = []
    for r in top.itertuples():
        out.append(Insight(
            "spike", (column,), _clip01(abs(r.score) / 8.0),
            describe_spike(r, view.layout.measure, column) + ".",
            {"row": r.row, "bucket": r.bucket, "kind": r.kind, "observed": r.observed, "expected": r.expected,
             "ratio": (None if not np.isfinite(r.ratio) else r.ratio), "score": r.score, "support": int(r.support),
             "baseline": r.baseline},
        ))
    return out


def _finding_novelty(view: Any, *, max_items: int = 3) -> List[Insight]:
    """What appeared in the last quarter of the time span (see :meth:`View.novel`), on the
    guessed entity/attribute columns; nothing without a time column or an entity-like one."""
    from ._novelty import default_entity_columns, default_time_column
    from ._novelty import novelty as _novelty

    if default_time_column(view) is None:
        return []
    entity, attr = default_entity_columns(view)
    if entity is None:
        return []
    try:
        top = _novelty(view, entity, attr, n=max_items)
    except Exception:
        return []
    out = []
    for r in top.itertuples():
        out.append(Insight(
            "novelty", tuple(c for c in (entity, attr) if c), _clip01(r.score / 12.0),
            r.text + ".",
            {"kind": r.kind, "entity": r.entity, "value": r.value, "before": int(r.before), "after": int(r.after),
             "peers": (None if pd.isna(r.peers) else int(r.peers)), "score": r.score,
             "since": str(top.attrs.get("since", ""))},
        ))
    return out


def insights(
    view: Any, *, sensitivity: float = 0.5, max_findings: int = 15, max_pairs: int = 5, seed: int = 0
) -> Dict[str, Any]:
    """Everything :func:`View.insights` returns - see that method for the full contract."""
    if not 0.0 <= sensitivity <= 1.0:
        raise ValueError(f"sensitivity must be between 0 and 1, got {sensitivity}")
    df = view.data
    prof: Profile = view.profile
    n_rows = len(df)

    columns: Dict[str, Any] = {}
    findings: List[Insight] = []
    numeric_cols: List[str] = []
    for cp in prof:
        if cp.kind == ID:
            columns[cp.name] = {"kind": cp.kind, "semantic": cp.semantic, "nunique": int(cp.nunique), "null_frac": round(cp.null_frac, 4)}
            continue
        columns[cp.name] = _column_summary(df, cp, seed=seed)
        f = _finding_constant(cp.name, cp, df)
        if f:
            findings.append(f)
        f = _finding_nulls(cp.name, cp)
        if f:
            findings.append(f)
        if cp.kind == NUMERIC and n_rows >= 20:
            numeric_cols.append(cp.name)
            for fn in (_finding_skew, _finding_modality, _finding_outliers):
                f = fn(cp.name, cp, df, seed)
                if f:
                    findings.append(f)
        elif cp.kind in (CATEGORICAL, BOOLEAN):
            f = _finding_concentration(cp.name, cp, df)
            if f:
                findings.append(f)
            f = _finding_high_cardinality(cp.name, cp)
            if f:
                findings.append(f)

    corr_cols = [c.name for c in prof if c.kind != ID and c.nunique >= 2]
    findings.extend(_finding_correlations(df, corr_cols, max_pairs=max_pairs))
    findings.extend(_finding_anomalies(view))
    findings.extend(_finding_spikes(view))
    findings.extend(_finding_novelty(view))

    threshold = 1.0 - sensitivity
    kept = sorted((f for f in findings if f.significance >= threshold), key=lambda f: -f.significance)[:max_findings]

    kinds = pd.Series([cp.kind for cp in prof]).value_counts().to_dict()
    summary = (
        f"{n_rows:,} rows x {len(prof)} columns ({', '.join(f'{v} {k}' for k, v in kinds.items())}). "
        + (f"{len(kept)} notable finding(s) at sensitivity {sensitivity:.2f}." if kept else
           f"Nothing crossed the significance bar at sensitivity {sensitivity:.2f} - try a higher value, "
           "or the data may simply be unremarkable (well-behaved, evenly spread, few nulls).")
    )

    return {
        "summary": summary,
        "shape": {"rows": n_rows, "cols": len(prof)},
        "columns": columns,
        "findings": [f.to_dict() for f in kept],
        "sensitivity": sensitivity,
    }


__all__ = ["Insight", "insights"]
