"""Bin-width and bucket selection: the "how many levels, and where do the cuts go" half.

* :func:`bin_count`   - classic rules (Freedman-Diaconis, Sturges, Scott, sqrt, rice, auto)
                        computed arithmetically so heavy tails can't allocate huge arrays.
* :func:`bin_edges`   - human-friendly edges: 1/2/2.5/5 x 10^k widths, integer-aware,
                        and automatic 1-2-5 log bins when the data spans many decades.
* :func:`bucket_time` - minute/hour/day/week/month/quarter/year buckets for timestamps.
* :func:`categorize`  - top-N + "(other)" bucketing and ordering for labels.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from pandas.api import types as pdt

OTHER = "(other)"
NULL = "(null)"

BinRule = Union[str, int]
RULES = ("auto", "fd", "sturges", "scott", "sqrt", "rice", "kmeans", "quantile")
_COUNT_RULES = ("auto", "fd", "sturges", "scott", "sqrt", "rice")

_NICE_FLOAT = (1.0, 2.0, 2.5, 5.0, 10.0)
_NICE_INT = (1.0, 2.0, 5.0, 10.0)


# --------------------------------------------------------------------------- numbers


def _clean(values) -> np.ndarray:
    if isinstance(values, pd.Series):
        if pdt.is_timedelta64_dtype(values):
            values = values.dt.total_seconds()
        values = values.to_numpy(dtype=float, na_value=np.nan)
    v = np.asarray(values, dtype=float).ravel()
    return v[np.isfinite(v)]


def bin_count(values, rule: BinRule = "auto") -> int:
    """Number of bins suggested by ``rule`` for ``values``.

    ``auto`` is Freedman-Diaconis clamped to [Sturges, 4 x Sturges], which behaves on
    heavy-tailed data (bytes, durations) where plain FD explodes.
    """
    if isinstance(rule, (int, np.integer)) and not isinstance(rule, bool):
        return max(1, int(rule))
    if rule in ("kmeans", "quantile"):
        rule = "auto"  # placement rules use the auto count
    v = _clean(values)
    n = v.size
    if n < 2:
        return 1
    lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        return 1
    span = hi - lo
    sturges = int(math.ceil(math.log2(n))) + 1
    if rule == "sturges":
        return sturges
    if rule == "sqrt":
        return int(math.ceil(math.sqrt(n)))
    if rule == "rice":
        return int(math.ceil(2 * n ** (1 / 3)))
    if rule == "scott":
        w = 3.49 * float(v.std()) * n ** (-1 / 3)
        return max(1, int(math.ceil(span / w))) if w > 0 else sturges
    if rule in ("fd", "auto"):
        q75, q25 = np.percentile(v, [75, 25])
        w = 2 * float(q75 - q25) * n ** (-1 / 3)
        fd = int(math.ceil(span / w)) if w > 0 else None
        if rule == "fd":
            return fd if fd else sturges
        if fd is None:
            return sturges
        return int(min(max(fd, sturges), 4 * sturges))
    raise ValueError(f"unknown bin rule {rule!r}; use one of {RULES} or an int")


def nice_number(x: float, *, integer: bool = False) -> float:
    """Smallest 'nice' number (1, 2, 2.5, 5 x 10^k; 1, 2, 5 x 10^k for integers) >= x."""
    if not (x > 0) or not math.isfinite(x):
        return 1.0
    if integer and x <= 1:
        return 1.0
    exp = math.floor(math.log10(x))
    base = 10.0 ** exp
    for m in _NICE_INT if integer else _NICE_FLOAT:
        cand = m * base
        if cand >= x * (1 - 1e-9):
            return float(cand)
    return float(10.0 * base)  # pragma: no cover - unreachable, 10*base always >= x


def _next_nice(w: float, integer: bool) -> float:
    return nice_number(w * (1 + 1e-6), integer=integer)


def _tidy(edges: np.ndarray, w: float) -> np.ndarray:
    decimals = max(0, -int(math.floor(math.log10(w))) + 1) if w > 0 else 0
    return np.round(edges, decimals)


def linear_edges(lo: float, hi: float, n_bins: int, *, integer: bool = False) -> np.ndarray:
    """At most ``n_bins`` equal-width bins with nice widths covering [lo, hi].

    For integer data the last edge is ``> hi`` so every bin is a half-open integer
    range; for floats the maximum lands in the last (closed) bin.
    """
    n_bins = max(1, int(n_bins))
    lo, hi = float(lo), float(hi)
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise ValueError("bin range must be finite")
    if integer:
        hi = hi + 1.0  # cover the max value with a half-open bin
    if hi <= lo:
        w = 1.0
        start = math.floor(lo)
        return np.array([start, start + w], dtype=float)
    w = nice_number((hi - lo) / n_bins, integer=integer)
    for _ in range(12):
        start = math.floor(lo / w + 1e-9) * w
        count = max(1, int(math.ceil((hi - start) / w - 1e-9)))
        if count <= n_bins:
            break
        w = _next_nice(w, integer)
    edges = start + w * np.arange(count + 1)
    return _tidy(edges, w)


def log_edges(lo: float, hi: float, n_bins: int) -> np.ndarray:
    """At most ``n_bins`` log-spaced nice edges (1-2-5 per decade, then 1-3, then decades)."""
    if not (lo > 0 and hi >= lo):
        raise ValueError("log bins need positive values")
    n_bins = max(1, int(n_bins))
    e0 = math.floor(math.log10(lo) + 1e-9)
    e1 = math.ceil(math.log10(hi) - 1e-9)
    for mant in ((1, 2, 5), (1, 3), (1,)):
        cand = sorted({m * 10.0 ** e for e in range(e0, e1 + 1) for m in mant})
        left = max(i for i, c in enumerate(cand) if c <= lo * (1 + 1e-9))
        right = min(i for i, c in enumerate(cand) if c >= hi * (1 - 1e-9))
        edges = cand[left : right + 1]
        if len(edges) < 2:
            edges = cand[left : left + 2] if left + 1 < len(cand) else [cand[left], cand[left] * 10]
        if len(edges) - 1 <= n_bins:
            return np.array(edges, dtype=float)
    k = 2
    while True:
        cand = [10.0 ** e for e in range(e0, e1 + k, k)]
        right = min(i for i, c in enumerate(cand) if c >= hi * (1 - 1e-9))
        edges = cand[: right + 1]
        if len(edges) - 1 <= n_bins or k > 64:
            return np.array(edges, dtype=float)
        k += 1


def should_log(values, scale: str = "auto") -> bool:
    """Decide whether log-spaced bins suit ``values`` (many decades, heavy tail)."""
    v = _clean(values)
    if v.size == 0 or v.min() < 0:
        return False
    pos = v[v > 0]
    if scale == "log":
        return pos.size > 0
    if scale != "auto":
        return False
    if pos.size < 20:
        return False
    if pos.max() / pos.min() < 1000:
        return False
    p50, p95 = np.percentile(pos, [50, 95])
    return bool(p95 / max(p50, 1e-12) > 20)


def bin_edges(
    values,
    rule: BinRule = "auto",
    *,
    max_bins: Optional[int] = None,
    scale: str = "auto",
    integer: Optional[bool] = None,
) -> np.ndarray:
    """Human-friendly bin edges for ``values``.

    Parameters
    ----------
    rule:      a rule name (see :data:`RULES`) or an explicit bin count.
    max_bins:  hard cap on the number of bins (a pivot axis budget, typically).
    scale:     ``"auto"`` (log bins when the data is heavy-tailed), ``"linear"`` or ``"log"``.
    integer:   treat the data as integers (nice integer widths). Auto-detected by default.
    """
    v = _clean(values)
    if v.size == 0:
        return np.array([0.0, 1.0])
    lo, hi = float(v.min()), float(v.max())
    if integer is None:
        integer = bool(np.all(np.mod(v, 1) == 0))
    n = bin_count(v, rule)
    if max_bins is not None:
        n = min(n, int(max_bins))
    n = max(1, n)
    if rule == "quantile":
        return quantile_edges(v, n, integer=integer)
    if rule == "kmeans":
        return natural_breaks(v, n, integer=integer)
    if should_log(v, scale):
        pos = v[v > 0]
        has_zero = pos.size < v.size
        edges = log_edges(float(pos.min()), float(pos.max()), max(1, n - (1 if has_zero else 0)))
        if has_zero:
            edges = np.concatenate([[0.0], edges])
        if integer and edges[-1] <= hi:
            edges = np.append(edges, edges[-1] * 10)
        return edges
    return linear_edges(lo, hi, n, integer=integer)


def _round_sig(x: np.ndarray, sig: int = 3) -> np.ndarray:
    """Round each value to ``sig`` significant digits (vectorised)."""
    out = np.asarray(x, dtype=float).copy()
    nz = out != 0
    mag = np.floor(np.log10(np.abs(out[nz])))
    scale = 10.0 ** (sig - 1 - mag)
    out[nz] = np.round(out[nz] * scale) / scale
    return out


def _dedupe_edges(edges: np.ndarray, lo: float, hi: float, integer: bool) -> np.ndarray:
    e = np.unique(np.round(edges, 0) if integer else edges)
    if e.size < 2:
        return linear_edges(lo, hi, 1, integer=integer)
    e[0] = min(e[0], lo)
    if integer:
        e[-1] = max(e[-1], hi + 1)
    else:
        e[-1] = max(e[-1], hi)
    return e


def quantile_edges(values, n_bins: int, *, integer: bool = False) -> np.ndarray:
    """Equal-frequency bins: each bin holds about the same number of values."""
    v = _clean(values)
    if v.size == 0:
        return np.array([0.0, 1.0])
    n_bins = max(1, int(n_bins))
    q = np.percentile(v, np.linspace(0, 100, n_bins + 1))
    q = _round_sig(q) if not integer else np.round(q)
    return _dedupe_edges(q, float(v.min()), float(v.max()), integer)


def natural_breaks(values, n_bins: int, *, integer: bool = False, seed: int = 0) -> np.ndarray:
    """Density-driven bins (1-D k-means, a.k.a. Jenks-style natural breaks).

    Cut points fall in the gaps between clusters of values, so multi-modal data gets one
    bin per mode instead of arbitrary equal widths.
    """
    from ._cluster import kmeans

    v = _clean(values)
    if v.size == 0:
        return np.array([0.0, 1.0])
    n_bins = max(1, int(n_bins))
    uniq = np.unique(v)
    if uniq.size <= n_bins:
        edges = np.concatenate([uniq, [uniq[-1] + (1.0 if integer else 0.0)]])
        return _dedupe_edges(edges, float(v.min()), float(v.max()), integer)
    sample = v if v.size <= 20000 else np.random.default_rng(seed).choice(v, 20000, replace=False)
    x = np.log1p(sample) if sample.min() >= 0 and should_log(sample, "auto") else sample
    labels, centers, _ = kmeans(x.reshape(-1, 1), n_bins, n_init=2, seed=seed)
    order = np.argsort(centers[:, 0])
    cuts = []
    for a, b in zip(order[:-1], order[1:]):
        hi_a = x[labels == a].max()
        lo_b = x[labels == b].min()
        cuts.append((hi_a + lo_b) / 2.0)
    cuts = np.array(cuts)
    if x is not sample:
        cuts = np.expm1(cuts)
    edges = np.concatenate([[v.min()], cuts, [v.max()]])
    edges = np.round(edges) if integer else _round_sig(edges)
    return _dedupe_edges(edges, float(v.min()), float(v.max()), integer)


def kde(values, grid: Optional[np.ndarray] = None, *, points: int = 128, sample: int = 5000, log: bool = False, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Gaussian kernel density estimate (Scott's bandwidth) on a sample. Returns ``(x, density)``.

    With ``log=True`` the estimate is done in log10 space (for heavy-tailed data) and the
    returned x values are on the original scale.
    """
    v = _clean(values)
    if v.size == 0:
        return np.array([]), np.array([])
    if v.size > sample:
        v = np.random.default_rng(seed).choice(v, sample, replace=False)
    if log:
        v = np.log10(v[v > 0]) if (v > 0).any() else v
    if grid is None:
        lo, hi = float(v.min()), float(v.max())
        pad = (hi - lo) * 0.05 or 0.5
        grid = np.linspace(lo - pad, hi + pad, points)
    elif log:
        grid = np.log10(np.clip(np.asarray(grid, dtype=float), 1e-12, None))
    sd = float(v.std())
    n = v.size
    bw = 1.06 * sd * n ** (-1 / 5) if sd > 0 else 1.0
    bw = max(bw, 1e-9)
    z = (grid[:, None] - v[None, :]) / bw
    dens = np.exp(-0.5 * z * z).sum(axis=1) / (n * bw * math.sqrt(2 * math.pi))
    x = 10 ** grid if log else grid
    return x, dens


def digitize(values, edges: Sequence[float]) -> np.ndarray:
    """Bin index for every value (``-1`` for NaN/inf). Last bin is closed on the right."""
    edges = np.asarray(edges, dtype=float)
    if isinstance(values, pd.Series):
        if pdt.is_timedelta64_dtype(values):
            values = values.dt.total_seconds()
        values = values.to_numpy(dtype=float, na_value=np.nan)
    v = np.asarray(values, dtype=float)
    idx = np.searchsorted(edges, v, side="right") - 1
    idx = np.clip(idx, 0, len(edges) - 2)
    idx = np.where(np.isfinite(v), idx, -1)
    return idx.astype(int)


def human(x: float, *, integer: bool = False) -> str:
    """Compact number: 950, 1.2K, 34M, 0.25 ..."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "nan"
    x = float(x)
    sign = "-" if x < 0 else ""
    a = abs(x)
    for div, suf in ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "K")):
        if a >= div * 0.9995:
            val = a / div
            s = f"{val:.3g}"
            if "e" in s:
                s = f"{val:.0f}"
            return f"{sign}{s}{suf}"
    if integer or a == int(a):
        return f"{sign}{int(round(a))}"
    s = f"{a:.3g}"
    if "e" in s:
        s = f"{a:.4f}".rstrip("0").rstrip(".")
    return sign + s


def bin_labels(edges: Sequence[float], *, integer: bool = False) -> List[str]:
    """Labels for consecutive bins.

    Equal-width integer bins read ``0-9``, ``10-19``; everything else (floats, log bins)
    uses interval notation ``[0, 10)`` with a closed last bin.
    """
    e = np.asarray(edges, dtype=float)
    out: List[str] = []
    n = len(e) - 1
    widths = np.diff(e)
    uniform = n < 2 or bool(np.allclose(widths, widths[0], rtol=1e-6, atol=0))
    for i in range(n):
        lo, hi = e[i], e[i + 1]
        if integer and uniform:
            top = hi - 1
            if top <= lo:
                out.append(human(lo, integer=True))
            else:
                out.append(f"{human(lo, integer=True)}-{human(top, integer=True)}")
        else:
            close = "]" if i == n - 1 else ")"
            out.append(f"[{human(lo)}, {human(hi)}{close}")
    return out


def bin_series(s: pd.Series, edges: Sequence[float], *, integer: Optional[bool] = None) -> pd.Series:
    """Ordered categorical of bin labels for ``s`` (nulls become ``"(null)"``)."""
    if integer is None:
        v = _clean(s)
        integer = bool(v.size) and bool(np.all(np.mod(v, 1) == 0))
    labels = bin_labels(edges, integer=integer)
    idx = digitize(s, edges)
    cats = list(labels)
    has_null = bool((idx < 0).any())
    if has_null:
        cats.append(NULL)
    codes = np.where(idx < 0, len(labels), idx)
    cat = pd.Categorical.from_codes(codes, categories=cats, ordered=True)
    return pd.Series(cat, index=s.index, name=s.name)


# --------------------------------------------------------------------------- time

TIME_FREQS: Tuple[str, ...] = ("min", "5min", "15min", "h", "6h", "D", "W", "M", "Q", "Y")
CYCLIC_FREQS: Tuple[str, ...] = ("hour_of_day", "weekday", "month_of_year")
_FREQ_TITLE = {
    "min": "minute", "5min": "5 min", "15min": "15 min", "h": "hour", "6h": "6 h",
    "D": "day", "W": "week", "M": "month", "Q": "quarter", "Y": "year",
    "hour_of_day": "hour of day", "weekday": "weekday", "month_of_year": "month of year",
}
_WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def freq_title(freq: str) -> str:
    return _FREQ_TITLE.get(freq, freq)


def _floor(s: pd.Series, freq: str) -> pd.Series:
    try:
        return s.dt.floor(freq)
    except (ValueError, TypeError):  # pandas < 2.2 spells hours "H"
        return s.dt.floor(freq.upper())


def _time_keys(s: pd.Series, freq: str) -> Tuple[pd.Series, pd.Series]:
    """(sortable key, label) per row for a datetime series bucketed at ``freq``."""
    if isinstance(s.dtype, pd.PeriodDtype):
        s = s.dt.to_timestamp()
    if freq == "hour_of_day":
        key = s.dt.hour
        return key, key.map(lambda h: f"{int(h):02d}h")
    if freq == "weekday":
        key = s.dt.dayofweek
        return key, key.map(lambda d: _WEEKDAYS[int(d)])
    if freq == "month_of_year":
        key = s.dt.month
        return key, key.map(lambda m: _MONTHS[int(m) - 1])
    if freq in ("M", "Q", "Y", "W"):
        p = s.dt.tz_localize(None).dt.to_period(freq) if getattr(s.dt, "tz", None) is not None else s.dt.to_period(freq)
        key = p.dt.start_time
        if freq == "W":
            label = "wk " + key.dt.strftime("%Y-%m-%d")
        else:
            label = p.astype(str)
        return key, label
    key = _floor(s, freq)
    fmt = "%Y-%m-%d" if freq == "D" else "%Y-%m-%d %H:%M"
    return key, key.dt.strftime(fmt)


def time_bucket_count(s: pd.Series, freq: str) -> int:
    key, _ = _time_keys(s.dropna(), freq)
    return int(key.nunique())


def time_freq_for(s: pd.Series, max_buckets: int) -> str:
    """Finest frequency whose bucket count fits in ``max_buckets``."""
    s = s.dropna()
    if s.empty:
        return "D"
    for f in TIME_FREQS:
        if time_bucket_count(s, f) <= max_buckets:
            return f
    return TIME_FREQS[-1]


def bucket_time(s: pd.Series, freq: str) -> pd.Series:
    """Ordered categorical of time-bucket labels for ``s``."""
    valid = s.notna()
    key, label = _time_keys(s[valid], freq)
    order = pd.DataFrame({"k": key.to_numpy(), "l": label.to_numpy()}).drop_duplicates("l").sort_values("k")
    cats = order["l"].tolist()
    has_null = bool((~valid).any())
    if has_null:
        cats.append(NULL)
    out = pd.Series(NULL, index=s.index, dtype=object)
    out[valid] = label
    cat = pd.Categorical(out, categories=cats, ordered=True)
    return pd.Series(cat, index=s.index, name=s.name)


# --------------------------------------------------------------------------- categories


def _natural_sort_key(values: List) -> List:
    try:
        return sorted(values)
    except TypeError:
        return sorted(values, key=lambda x: (str(type(x)), str(x)))


def categorize(
    s: pd.Series,
    *,
    top: Optional[int] = None,
    order: str = "auto",
    other: str = OTHER,
) -> pd.Series:
    """Ordered categorical for a label column.

    ``top`` keeps the ``top`` most frequent values and folds the rest into ``"(other)"``.
    ``order`` is ``"natural"`` (sorted), ``"frequency"`` (most common first) or ``"auto"``
    (natural for numbers/booleans, frequency for text). ``"(other)"`` and ``"(null)"``
    always go last.
    """
    if isinstance(s.dtype, pd.CategoricalDtype):
        s = s.astype(object)
    if pdt.is_bool_dtype(s):
        s = s.astype(object)
    valid = s.notna()
    vals = s[valid]
    try:
        counts = vals.value_counts()
    except TypeError:
        vals = vals.astype(str)
        counts = vals.value_counts()
    numeric_like = pdt.is_numeric_dtype(vals) or all(isinstance(x, (int, float, bool, np.number)) for x in counts.index[:50])
    if order == "auto":
        order = "natural" if numeric_like else "frequency"

    used_other = False
    if top is not None and len(counts) > top:
        keep = counts.index[:top]
        vals = vals.where(vals.isin(keep), other).astype(object)
        counts = counts.loc[keep]
        used_other = True

    if order == "frequency":
        cats = list(counts.index)
    else:
        cats = _natural_sort_key(list(counts.index))
    if used_other:
        cats.append(other)
    has_null = bool((~valid).any())
    if has_null:
        cats.append(NULL)

    out = pd.Series(NULL, index=s.index, dtype=object)
    out[valid] = vals.to_numpy(dtype=object)
    cat = pd.Categorical(out, categories=cats, ordered=True)
    return pd.Series(cat, index=s.index, name=s.name)
