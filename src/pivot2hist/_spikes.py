"""Spike and change detection on the per-row trends: which rows of the table moved, when,
and by how much against their *own* history.

A sparkline shows the shape; this scores it. Each row of the current table gets a time
series of the view's measure (the same buckets :func:`pivot2hist.sparkline_table` draws)
and every bucket is compared with a baseline built from that row's other buckets. The
baseline is robust - a median, with the spread measured by the median absolute deviation
(MAD) - so the spike itself doesn't inflate the yardstick it is judged by, and it is
*seasonal* when the data allows: a 6-hour bucket is compared with the same six hours on
the other days, a daily bucket with the same weekday in other weeks, a monthly one with
the same month in other years, so the daily peak is not reported as a spike every day.
Without three periods to learn a season from, an additive measure is judged by the row's
share of the bucket's total (a peak everyone has is not this row's spike); anything else
falls back to the row's plain median.

Two kinds of movement are reported:

* a **spike** (or **drop**): one bucket far above (below) its baseline, and
* a **shift**: a step change - after some bucket the row stays higher (lower) than its
  baseline predicts, found as the split point whose standardized residuals differ most
  before and after.

Scores are robust z-scores ("spreads": how many baseline spreads away), floored so a flat
row with a tiny spread cannot make everything a spike (Poisson-like for counts, a tenth of
the level otherwise), with a minimum number of rows behind the bucket. Sums and means of
heavy-tailed quantities (bytes, latencies) are scored on a log scale so one big transfer
reads as ×20, not as +25 spreads. Nothing is tuned per dataset; scores compare across rows
and across kinds.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ._log import log
from ._profile import DATETIME

_MAD_TO_SIGMA = 1.4826  # MAD of a normal sample x 1.4826 = its standard deviation
_MIN_BUCKETS = 5  # a baseline needs at least this many finite buckets in the row
_MIN_SEASON = 3  # same-phase peers (periods) needed before a seasonal baseline is trusted
_MIN_SIDE = 4  # a shift needs at least this many buckets on each side of the step
_LOG_RANGE = 100.0  # non-negative measures spanning more than this ratio are scored on log scale
_MIN_SHARE_ROWS = 6  # fewer rows than this and one row's spike is every other row's "drop" in share terms
_MIN_EFFECT = 1.25  # observed/expected must be at least this far from x1 (either way) to be reported
BASELINES = ("auto", "seasonal", "share", "median")
COLUMNS = ["row", "bucket", "kind", "observed", "expected", "ratio", "score", "support", "baseline"]

_SUB_DAILY = ("min", "5min", "15min", "h", "6h")


# --------------------------------------------------------------------------- baselines


def _row_label(index: pd.Index, i: int) -> str:
    return "/".join(map(str, index[i])) if index.nlevels > 1 else str(index[i])


def _floor(level: np.ndarray, counts: bool) -> np.ndarray:
    """Smallest spread we are willing to believe: sqrt(level) for counts (Poisson), a tenth
    of the level otherwise (a tenth of a unit on the log scale), never zero."""
    lv = np.abs(np.where(np.isfinite(level), level, 0.0))
    f = np.sqrt(np.maximum(lv, 1.0)) if counts else 0.1 * np.maximum(lv, 1.0)
    return np.maximum(f, 1e-9)


def _robust(values: np.ndarray, counts: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Per-row median and floored, MAD-based spread (rows x buckets in, two vectors out)."""
    with np.errstate(all="ignore"):
        med = np.nanmedian(values, axis=1)
        mad = np.nanmedian(np.abs(values - med[:, None]), axis=1) * _MAD_TO_SIGMA
    med = np.where(np.isfinite(med), med, 0.0)
    mad = np.where(np.isfinite(mad), mad, 0.0)
    return med, np.maximum(mad, _floor(med, counts))


def phase_keys(starts: List[pd.Timestamp], freq: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """(phase, period) per bucket for the season a bucket frequency implies: time of day
    within a day for sub-daily buckets, weekday within a week for daily ones, month or
    quarter within a year for those. ``(None, None)`` when the frequency has no natural
    season (weeks, years)."""
    ts = pd.DatetimeIndex(starts)
    if freq in _SUB_DAILY:
        return (ts.hour * 60 + ts.minute).to_numpy(), ts.normalize().to_numpy()
    if freq == "D":
        return ts.dayofweek.to_numpy(), ts.to_period("W").start_time.to_numpy()
    if freq in ("M", "Q"):
        return (ts.month if freq == "M" else ts.quarter).to_numpy(), ts.year.to_numpy()
    return None, None


def _median_baseline(values: np.ndarray, counts: bool) -> Tuple[np.ndarray, np.ndarray]:
    med, scale = _robust(values, counts)
    return np.repeat(med[:, None], values.shape[1], axis=1), np.repeat(scale[:, None], values.shape[1], axis=1)


def _seasonal_baseline(values: np.ndarray, phase: np.ndarray, counts: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Same-phase median/spread per bucket; a phase with too few members uses the row's
    plain median."""
    expected, scale = _median_baseline(values, counts)
    for ph in np.unique(phase):
        cols = np.nonzero(phase == ph)[0]
        if cols.size < _MIN_SEASON:
            continue
        med, sc = _robust(values[:, cols], counts)
        expected[:, cols] = med[:, None]
        scale[:, cols] = sc[:, None]
    return expected, scale


def _share_baseline(values: np.ndarray, counts: bool) -> Tuple[np.ndarray, np.ndarray]:
    """The row's usual share of each bucket's total, back in the measure's units: expected
    = median share x that bucket's total, spread likewise (floored on the raw scale)."""
    safe = np.where(np.isfinite(values), values, 0.0)
    total = safe.sum(axis=0)
    with np.errstate(all="ignore"):
        share = np.where(total > 0, safe / total, np.nan)
        share = np.where(np.isfinite(values), share, np.nan)
    med, mad = _robust(share, counts=False)
    expected = med[:, None] * total[None, :]
    spread = np.maximum(mad, 1e-9)[:, None] * total[None, :]
    return expected, np.maximum(spread, _floor(expected, counts))


def _pick_baseline(baseline: str, *, additive: bool, n_rows: int, phase: Optional[np.ndarray], period: Optional[np.ndarray]) -> str:
    seasonal_ok = phase is not None and period is not None and len(np.unique(period)) >= _MIN_SEASON
    if baseline == "auto":
        if seasonal_ok:
            return "seasonal"
        if additive and n_rows >= _MIN_SHARE_ROWS:
            return "share"
        return "median"
    if baseline == "seasonal" and not seasonal_ok:
        return "median"
    if baseline == "share" and not additive:
        raise ValueError("baseline='share' needs an additive measure (count/sum); use 'seasonal' or 'median'")
    return baseline


# --------------------------------------------------------------------------- detection


def _spike_rows(score: np.ndarray, finite: np.ndarray, support: np.ndarray, *, z: float, min_support: int) -> List[Tuple[int, int, str, float]]:
    enough = finite.sum(axis=1) >= _MIN_BUCKETS
    ok = finite & enough[:, None] & (support >= min_support) & (np.abs(score) >= z)
    return [(int(i), int(t), "spike" if score[i, t] > 0 else "drop", float(score[i, t])) for i, t in zip(*np.nonzero(ok))]


def _shift_rows(score: np.ndarray, support: np.ndarray, *, z: float, min_support: int) -> List[Tuple[int, int, str, float]]:
    """The best step per row on the standardized residuals: the split whose after-median
    differs most from its before-median, as a two-sample z (medians, pooled unit spread)."""
    n_rows, n_buckets = score.shape
    if n_buckets < 2 * _MIN_SIDE:
        return []
    best = np.zeros(n_rows)
    best_t = np.full(n_rows, -1)
    finite = np.isfinite(score)
    for t in range(_MIN_SIDE, n_buckets - _MIN_SIDE + 1):
        nb, na = finite[:, :t].sum(axis=1), finite[:, t:].sum(axis=1)
        ok = (nb >= _MIN_SIDE) & (na >= _MIN_SIDE)
        with np.errstate(all="ignore"):
            mb, ma = np.nanmedian(score[:, :t], axis=1), np.nanmedian(score[:, t:], axis=1)
            s = np.where(ok, (ma - mb) * np.sqrt(nb * na / np.maximum(nb + na, 1)), 0.0)
        s = np.where(np.isfinite(s), s, 0.0)
        better = np.abs(s) > np.abs(best)
        best, best_t = np.where(better, s, best), np.where(better, t, best_t)
    out = []
    for i in np.nonzero((np.abs(best) >= z) & (best_t >= 0))[0]:
        t = int(best_t[i])
        if support[i].sum() < min_support:  # the whole row: a step down to nothing has no rows after it
            continue
        out.append((int(i), t, "shift up" if best[i] > 0 else "shift down", float(best[i])))
    return out


def _measure_is_count(view: Any) -> bool:
    layout = view.layout
    return layout.values is None or layout.agg == "count"


def _use_log(values: np.ndarray, counts: bool) -> bool:
    if counts:
        return False
    fin = values[np.isfinite(values)]
    pos = fin[fin > 0]
    return bool(fin.size and fin.min() >= 0 and pos.size >= 2 and pos.max() / pos.min() > _LOG_RANGE)


def default_time_column(view: Any) -> Optional[str]:
    """The first datetime column that is not on the row axis (a row dim's own buckets make
    a one-bucket "trend" per row - nothing to score against)."""
    on_rows = {d.column for d in view.layout.rows}
    for cp in view.profile:
        if cp.kind == DATETIME and cp.name not in on_rows:
            return cp.name
    return None


def spikes(
    view: Any,
    column: Optional[str] = None,
    *,
    n: int = 10,
    z: float = 3.0,
    min_support: int = 5,
    max_points: int = 48,
    baseline: str = "auto",
    shifts: bool = True,
) -> pd.DataFrame:
    """Everything :meth:`View.spikes` returns - see that method for the contract."""
    if column is None:
        column = default_time_column(view)
        if column is None:
            raise ValueError("no datetime column to scan for spikes; pass column=")
    if column in {d.column for d in view.layout.rows}:
        raise ValueError(
            f"{column!r} is on the row axis, so each row covers one bucket of it and has no history to compare with; "
            "put another column on rows, or use anomalies() for cell-vs-marginal surprise"
        )
    if z <= 0:
        raise ValueError("z must be positive")
    if baseline not in BASELINES:
        raise ValueError(f"baseline must be one of {BASELINES}, got {baseline!r}")
    from ._sparkline import trend_tables

    with log.step("spikes", f"{view.layout.describe()} over {column}"):
        trend = trend_tables(view, column, max_points=max_points)
        table, cats = trend.table, trend.labels
        if table.empty or not cats:
            return pd.DataFrame(columns=COLUMNS)
        raw = table.to_numpy(dtype=float, copy=True)
        support = trend.counts.to_numpy(dtype=float, copy=True)
        is_count = _measure_is_count(view)
        additive = is_count or view.layout.agg == "sum"
        phase, period = phase_keys(trend.starts, trend.freq)
        how = _pick_baseline(baseline, additive=additive, n_rows=raw.shape[0], phase=phase, period=period)
        logged = _use_log(raw, is_count)
        values = np.log1p(raw) if logged else raw
        counts = is_count and not logged
        if how == "seasonal":
            expected, scale = _seasonal_baseline(values, phase, counts)
        elif how == "share":
            expected, scale = _share_baseline(values, counts)
        else:
            expected, scale = _median_baseline(values, counts)
        finite = np.isfinite(values)
        with np.errstate(all="ignore"):
            score = np.where(finite, (values - expected) / scale, np.nan)
        found = _spike_rows(score, finite, support, z=z, min_support=min_support)
        if shifts:
            found.extend(_shift_rows(score, support, z=z, min_support=min_support))
        expected_raw = np.expm1(expected) if logged else expected
        rows: List[Dict[str, Any]] = []
        for i, t, kind, s in found:
            if kind.startswith("shift"):
                # a step is judged on everything after it: the sum for an additive measure, the mean otherwise
                fold = np.nansum if additive else np.nanmean
                obs_v, exp_v = fold(raw[i, t:]), fold(expected_raw[i, t:][finite[i, t:]])
                sup = float(support[i, t:].sum())
            else:
                obs_v, exp_v, sup = raw[i, t], expected_raw[i, t], float(support[i, t])
            observed, exp_raw = float(obs_v), float(exp_v)
            with np.errstate(all="ignore"):
                ratio = observed / exp_raw if exp_raw else (np.inf if observed > 0 else np.nan)
            if np.isfinite(ratio) and 1.0 / _MIN_EFFECT <= ratio <= _MIN_EFFECT:
                continue  # statistically loud but practically flat (a busy row's +5%): not worth a line
            rows.append({
                "row": _row_label(table.index, i), "bucket": cats[t], "kind": kind,
                "observed": observed, "expected": round(exp_raw, 4), "ratio": float(ratio), "score": round(s, 2),
                "support": int(sup), "baseline": how + (" (log)" if logged else ""),
            })
        out = pd.DataFrame(rows, columns=COLUMNS)
        if out.empty:
            return out
        out = out.reindex(out["score"].abs().sort_values(ascending=False, kind="stable").index)
        return out.head(max(0, int(n))).reset_index(drop=True)


def describe_spike(r: Any, measure: str, column: str) -> str:
    """One sentence for a row of :func:`spikes` (an itertuple or a mapping)."""
    g = r.__getitem__ if isinstance(r, dict) else (lambda k: getattr(r, k))
    kind, ratio = g("kind"), g("ratio")
    times = f" (×{ratio:.2g})" if np.isfinite(ratio) else ""
    if kind.startswith("shift"):
        return (f"{g('row')}: {measure} steps {'up' if kind.endswith('up') else 'down'} from {column} {g('bucket')} on - "
                f"{g('observed'):,.4g} since vs {g('expected'):,.4g} expected{times}, {g('score'):+.1f} spreads, {g('support'):,} row(s)")
    verb = "spikes to" if kind == "spike" else "drops to"
    return (f"{g('row')}: {measure} {verb} {g('observed'):,.4g} at {column} {g('bucket')} vs {g('expected'):,.4g} "
            f"expected{times}, {g('score'):+.1f} spreads, {g('support'):,} row(s)")


__all__ = ["spikes", "describe_spike", "default_time_column", "phase_keys", "BASELINES", "COLUMNS"]
