"""Auto-fit: choose rows, columns, measure and level budgets so the data fits a pivot.

The fit answers "what is the best pivot table for this frame in a box of
``max_rows`` x ``max_cols`` cells?" by:

1. profiling columns (:mod:`._profile`);
2. picking a measure (an additive numeric column such as ``bytes``, else row count);
3. enumerating small combinations of candidate dimensions for the row and column
   axes (up to ``layers`` per axis), planning a level budget for each (top-N,
   numeric bins, time buckets) so the planned shape fits the box;
4. scoring each candidate on a sample of the data: information (non-empty cells),
   sparsity, distance from the target aspect ratio, and dimension quality.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from . import _binning as B
from ._profile import BOOLEAN, CATEGORICAL, CONSTANT, DATETIME, ID, NUMERIC, ColumnProfile, Profile, profile

AGGS = ("sum", "mean", "count", "min", "max", "median", "nunique", "std")
COUNT = "count"


# --------------------------------------------------------------------------- options


@dataclass
class FitOptions:
    """Knobs for the auto-fit. All have sensible defaults.

    max_rows / max_cols:
        The box the pivot must fit in (rows x columns of the table body).
    layers:
        Maximum number of stacked dimensions per axis (``2`` gives e.g. ``src_ip > dst_port``).
    aspect:
        Target rows/columns ratio. ``None`` uses the box's own ratio (``max_rows / max_cols``).
    max_sparsity:
        Fraction of empty cells tolerated before a layout is penalised.
    max_bins:
        Bin cap for standalone histograms (``View.histogram``).
    bins:
        Bin rule for numeric dimensions: ``"auto"``, ``"fd"``, ``"sturges"``, ``"scott"``,
        ``"sqrt"``, ``"rice"`` or an int.
    scale:
        ``"auto"`` / ``"linear"`` / ``"log"`` numeric binning.
    order:
        Level order for label dimensions: ``"auto"``, ``"natural"`` or ``"frequency"``.
    sample:
        Rows used to score candidate layouts (the final table always uses all rows).
    search_width:
        How many of the best-looking dimension candidates enter the combinatorial search.
    exclude:
        Column names never used as dimensions or measures.
    max_categories / discrete_max / id_ratio:
        Profiling thresholds, see :func:`pivot2hist.profile`.
    """

    max_rows: int = 40
    max_cols: int = 12
    layers: int = 2
    aspect: Optional[float] = None
    max_sparsity: float = 0.6
    max_bins: int = 30
    bins: B.BinRule = "auto"
    scale: str = "auto"
    order: str = "auto"
    sample: int = 50_000
    search_width: int = 7
    exclude: Sequence[str] = ()
    max_categories: int = 50
    discrete_max: int = 20
    id_ratio: float = 0.5
    seed: int = 0

    @property
    def target_aspect(self) -> float:
        if self.aspect and self.aspect > 0:
            return float(self.aspect)
        return max(self.max_rows, 1) / max(self.max_cols, 1)

    def replace(self, **kw: Any) -> "FitOptions":
        unknown = set(kw) - set(self.__dataclass_fields__)
        if unknown:
            raise TypeError(f"unknown option(s): {sorted(unknown)}")
        return replace(self, **kw)


# --------------------------------------------------------------------------- dims / layout


@dataclass(frozen=True)
class Dim:
    """One pivot dimension: a column plus how it is bucketed into levels."""

    column: str
    kind: str  # "categorical" | "binned" | "time"
    levels: int  # planned level count (observed may be lower)
    edges: Optional[Tuple[float, ...]] = None  # binned
    integer: bool = False  # binned: integer-valued data
    freq: Optional[str] = None  # time
    top: Optional[int] = None  # categorical: top-N kept, rest -> "(other)"
    order: str = "auto"
    quality: float = 1.0  # profiling preference, feeds the score
    entity: bool = False  # names an entity (ip, host, user ...): reads best as a row
    position: int = 0  # source column index
    natural: int = 0  # natural level count before any budget was applied

    @property
    def label(self) -> str:
        if self.kind == "binned":
            return f"{self.column} (bins)"
        if self.kind == "time":
            return f"{self.column} ({B.freq_title(self.freq or '')})"
        if self.top is not None:
            return f"{self.column} (top {self.top})"
        return self.column

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"column": self.column, "kind": self.kind, "levels": self.levels, "label": self.label}
        if self.edges is not None:
            d["edges"] = list(self.edges)
        if self.freq:
            d["freq"] = self.freq
        if self.top is not None:
            d["top"] = self.top
        return d


@dataclass(frozen=True)
class Layout:
    """Rows, columns and measure of a pivot table."""

    rows: Tuple[Dim, ...]
    cols: Tuple[Dim, ...] = ()
    values: Optional[str] = None  # None -> row count
    agg: str = "sum"

    @property
    def dims(self) -> Tuple[Dim, ...]:
        return self.rows + self.cols

    @property
    def columns_used(self) -> List[str]:
        out = [d.column for d in self.dims]
        if self.values is not None:
            out.append(self.values)
        return out

    @property
    def measure(self) -> str:
        return COUNT if self.values is None else f"{self.agg}({self.values})"

    def describe(self) -> str:
        rows = " > ".join(d.label for d in self.rows) or "-"
        cols = " > ".join(d.label for d in self.cols)
        s = f"{self.measure} by {rows}"
        if cols:
            s += f" x {cols}"
        return s

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rows": [d.to_dict() for d in self.rows],
            "cols": [d.to_dict() for d in self.cols],
            "values": self.values,
            "agg": None if self.values is None else self.agg,
            "measure": self.measure,
        }

    def __repr__(self) -> str:
        return f"<Layout {self.describe()}>"


# --------------------------------------------------------------------------- helpers

DimSpec = Union[str, Dim, Dict[str, Any]]


def _prof_quality(cp: ColumnProfile) -> float:
    base = {CATEGORICAL: 1.0, BOOLEAN: 0.55, DATETIME: 0.9, NUMERIC: 0.35, ID: 0.05}.get(cp.kind, 0.0)
    if cp.kind == CATEGORICAL and cp.discrete_hint:
        base += 0.1
    if cp.kind == CATEGORICAL and cp.nunique > 200:
        base -= 0.2
    if cp.kind == CATEGORICAL and cp.is_integer and not cp.discrete_hint:
        base -= 0.2  # small integers with no label-ish name are often quantities
    if cp.additive_hint:
        base -= 0.35  # looks like a measure, not a dimension
    return max(0.0, base - cp.null_frac)


def plan_dim(
    series: pd.Series,
    cp: ColumnProfile,
    budget: int,
    options: FitOptions,
    *,
    bins: Optional[B.BinRule] = None,
    kind: Optional[str] = None,
) -> Optional[Dim]:
    """Plan how ``series`` becomes a dimension with at most ``budget`` levels."""
    budget = int(budget)
    if budget < 1 or cp.kind == CONSTANT and kind is None:
        return None
    q = _prof_quality(cp)
    natural = cp.natural_levels
    kind = kind or {
        NUMERIC: "binned",
        DATETIME: "time",
    }.get(cp.kind, "categorical")

    meta = dict(quality=q, entity=cp.entity_hint, position=cp.position, natural=natural)
    if kind == "categorical":
        if natural <= budget:
            return Dim(cp.name, "categorical", natural, order=options.order, **meta)
        if budget < 2:
            return None
        return Dim(cp.name, "categorical", budget, top=budget - 1, order=options.order, **meta)

    if kind == "time":
        freq = B.time_freq_for(series, budget)
        n = B.time_bucket_count(series, freq) + (1 if cp.null_frac > 0 else 0)
        if n > budget:
            return None
        return Dim(cp.name, "time", max(1, n), freq=freq, **meta)

    if kind == "binned":
        rule = options.bins if bins is None else bins
        cap = budget - (1 if cp.null_frac > 0 else 0)
        if cap < 1:
            return None
        edges = B.bin_edges(series, rule, max_bins=cap, scale=options.scale, integer=cp.is_integer or None)
        n = len(edges) - 1 + (1 if cp.null_frac > 0 else 0)
        if n > budget:
            return None
        return Dim(cp.name, "binned", n, edges=tuple(float(e) for e in edges), integer=cp.is_integer, **meta)

    raise ValueError(f"unknown dim kind {kind!r}")


def materialize(df: pd.DataFrame, dim: Dim) -> pd.Series:
    """Turn a dimension into an ordered categorical Series of level labels."""
    s = df[dim.column]
    if dim.kind == "binned":
        return B.bin_series(s, dim.edges or (0.0, 1.0), integer=dim.integer)
    if dim.kind == "time":
        if not pdt.is_datetime64_any_dtype(s) and not isinstance(s.dtype, pd.PeriodDtype):
            s = pd.to_datetime(s, errors="coerce")
        return B.bucket_time(s, dim.freq or "D")
    return B.categorize(s, top=dim.top, order=dim.order)


# --------------------------------------------------------------------------- measure


def _measure_score(cp: ColumnProfile) -> float:
    score = 0.0
    if cp.additive_hint:
        score += 2.0
    score += 1.0 - cp.null_frac
    score += min(math.log1p(cp.nunique), 8) / 8
    if cp.discrete_hint or cp.id_hint:
        score -= 1.5
    return score


def choose_measure(prof: Profile, exclude: Iterable[str] = ()) -> Tuple[Optional[str], str]:
    """Best (values, agg) for the frame: sum of an additive-looking numeric column, else count."""
    ex = set(exclude)
    cands = [c for c in prof.measures if c.name not in ex and c.additive_hint and c.null_frac < 0.5]
    if not cands:
        return None, "sum"
    best = max(cands, key=_measure_score)
    return best.name, "sum"


def default_agg(cp: Optional[ColumnProfile]) -> str:
    if cp is None:
        return "sum"
    return "sum" if cp.additive_hint or cp.is_integer else "mean"


# --------------------------------------------------------------------------- search


def _allocate(
    dims: List[Tuple[ColumnProfile, pd.Series]], budget: int, options: FitOptions, *, slack: Optional[float] = None
) -> Optional[List[Dim]]:
    """Plan levels for an ordered list of dims so the product roughly fits ``budget``."""
    if not dims:
        return []
    naturals = []
    for cp, s in dims:
        if cp.kind == NUMERIC:
            naturals.append(min(cp.natural_levels, B.bin_count(s, options.bins), options.max_bins))
        elif cp.kind == DATETIME:
            naturals.append(min(cp.natural_levels, budget))
        else:
            naturals.append(cp.natural_levels)
    if slack is None:
        slack = 1.5 if len(dims) > 1 else 1.0
    caps = list(naturals)
    for _ in range(64):
        prod = float(np.prod(caps))
        if prod <= budget * slack:
            break
        i = int(np.argmax(caps))
        others = prod / caps[i]
        new_cap = int(math.floor(budget * slack / others))
        if new_cap >= caps[i]:
            new_cap = caps[i] - 1
        if new_cap < 2:
            return None
        caps[i] = new_cap
    out: List[Dim] = []
    for (cp, s), cap in zip(dims, caps):
        d = plan_dim(s, cp, cap, options)
        if d is None:
            return None
        out.append(d)
    return out


def _order_for_axis(dims: List[Tuple[ColumnProfile, pd.Series]]) -> List[Tuple[ColumnProfile, pd.Series]]:
    """Outer level = fewest natural levels; time goes innermost among equals."""
    return sorted(dims, key=lambda t: (t[0].natural_levels, t[0].kind == DATETIME))


def _entropy(counts: np.ndarray) -> float:
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


class _Evaluator:
    """Materialises dims on a sample once and scores candidate layouts."""

    def __init__(self, sample: pd.DataFrame, options: FitOptions):
        self.sample = sample
        self.options = options
        self._codes: Dict[Tuple, np.ndarray] = {}

    def _key(self, dim: Dim) -> Tuple:
        return (dim.column, dim.kind, dim.top, dim.freq, dim.edges)

    def codes(self, dim: Dim) -> np.ndarray:
        k = self._key(dim)
        if k not in self._codes:
            self._codes[k] = materialize(self.sample, dim).cat.codes.to_numpy()
        return self._codes[k]

    def shape(self, rows: Sequence[Dim], cols: Sequence[Dim]) -> "Shape":
        """Observed rows/cols, non-empty cells, cell entropy and the mass folded into (other)."""
        n = len(self.sample)
        if n == 0:
            return Shape(0, 0, 0, 0.0, 0.0, 0.0)
        r_key = self._combo(rows)
        c_key = self._combo(cols) if cols else np.zeros(n, dtype=np.int64)
        n_rows = int(np.unique(r_key).size)
        n_cols = int(np.unique(c_key).size) if cols else 1
        _, counts = np.unique(r_key.astype(np.int64) * (int(c_key.max()) + 1) + c_key, return_counts=True)
        h_rc = _entropy(counts)
        mi = 0.0
        if cols:
            h_r = _entropy(np.unique(r_key, return_counts=True)[1])
            h_c = _entropy(np.unique(c_key, return_counts=True)[1])
            # finite-sample bias correction (Miller-Madow) so big tables don't fake structure
            bias = (n_rows - 1) * (n_cols - 1) / (2.0 * n)
            mi = max(0.0, h_r + h_c - h_rc - bias)
        other = sum(self.other_frac(d) for d in list(rows) + list(cols))
        return Shape(n_rows, n_cols, int(counts.size), h_rc, mi, other)

    def other_frac(self, dim: Dim) -> float:
        """Fraction of sample rows folded into the "(other)" bucket of a top-N dimension."""
        if dim.top is None:
            return 0.0
        k = ("other",) + self._key(dim)
        if k not in self._codes:
            cat = materialize(self.sample, dim)
            cats = list(cat.cat.categories)
            if B.OTHER in cats:
                code = cats.index(B.OTHER)
                frac = float((cat.cat.codes.to_numpy() == code).mean())
            else:
                frac = 0.0
            self._codes[k] = np.array([frac])
        return float(self._codes[k][0])

    def _combo(self, dims: Sequence[Dim]) -> np.ndarray:
        key = np.zeros(len(self.sample), dtype=np.int64)
        for d in dims:
            key = key * (d.levels + 1) + (self.codes(d) + 1)
        return key


@dataclass(frozen=True)
class Shape:
    """What a candidate layout looks like on the sample."""

    n_rows: int
    n_cols: int
    cells: int
    entropy: float  # of the row-count distribution over cells, in nats
    mutual_info: float  # between the row axis and the column axis, in nats
    other_frac: float  # summed over dims: share of rows folded into "(other)"

    @property
    def sparsity(self) -> float:
        total = self.n_rows * self.n_cols
        return 1.0 - self.cells / total if total else 1.0


def score_layout(shape: Shape, rows: Sequence[Dim], cols: Sequence[Dim], options: FitOptions) -> float:
    """Higher is better.

    Information is the entropy of how rows spread over cells (many, evenly used cells)
    plus the mutual information between the axes (tables that show structure); it is
    traded off against sparsity, distance from the target aspect ratio, extra layers,
    low-quality dimensions and mass hidden in "(other)" buckets.
    """
    if shape.n_rows < 1 or shape.n_cols < 1 or shape.cells < 2:
        return -math.inf
    if shape.n_rows > options.max_rows or shape.n_cols > options.max_cols:
        return -math.inf
    sparsity = shape.sparsity
    score = shape.entropy + 2.5 * shape.mutual_info
    score -= max(0.0, sparsity - options.max_sparsity) * 8.0
    score -= sparsity * 0.6
    ratio = shape.n_rows / shape.n_cols
    aspect_w = 1.5 if options.aspect else 0.35  # an explicit target ratio is a request, not a hint
    score -= aspect_w * abs(math.log(ratio / options.target_aspect))
    score -= 0.7 * (len(rows) + len(cols) - 1)
    score += 0.5 * sum(d.quality for d in rows) + 0.5 * sum(d.quality for d in cols)
    score -= 1.2 * shape.other_frac
    if cols and shape.n_cols >= 2:
        score += 0.4
    # Pivot-structure priors: entities down the side, small categories across the top,
    # and a weak preference for columns that come first in the source.
    for d in rows:
        if d.entity:
            score += 0.5
        if d.natural and d.natural <= 4:
            score -= 0.2
    for d in cols:
        if d.natural and d.natural <= 8:
            score += 0.35
        if d.entity and d.natural > 8:
            score -= 0.3
    for d in list(rows) + list(cols):
        score -= 0.05 * d.position
        if d.levels < 2:
            score -= 1.0
    return score


def _resolve_spec(spec: DimSpec, df: pd.DataFrame, prof: Profile, budget: int, options: FitOptions) -> Dim:
    """Turn a user-supplied row/col spec into a Dim."""
    if isinstance(spec, Dim):
        return spec
    if isinstance(spec, dict):
        spec = dict(spec)
        col = spec.pop("column")
        cp = prof[col]
        kind = spec.pop("kind", None)
        bins = spec.pop("bins", None)
        if "edges" in spec:
            edges = tuple(float(e) for e in spec.pop("edges"))
            return Dim(col, "binned", len(edges) - 1, edges=edges, integer=cp.is_integer, quality=_prof_quality(cp))
        if "freq" in spec:
            freq = spec.pop("freq")
            return Dim(col, "time", B.time_bucket_count(df[col], freq), freq=freq, quality=_prof_quality(cp))
        if "top" in spec:
            top = int(spec.pop("top"))
            return Dim(col, "categorical", top + 1, top=top, order=options.order, quality=_prof_quality(cp))
        d = plan_dim(df[col], cp, int(spec.pop("levels", budget)), options, bins=bins, kind=kind)
        if d is None:
            raise ValueError(f"cannot use {col!r} as a dimension")
        return d
    col = str(spec)
    if col not in prof:
        raise KeyError(f"unknown column {col!r}")
    cp = prof[col]
    kind = None
    if cp.kind == CONSTANT:
        kind = "categorical"
    elif cp.kind == ID:
        kind = "categorical"
    d = plan_dim(df[col], cp, budget, options, kind=kind)
    if d is None:
        raise ValueError(f"cannot fit {col!r} into {budget} levels")
    return d


def _resolve_axis(specs: Optional[Sequence[DimSpec]], df, prof, budget, options) -> Optional[List[Dim]]:
    """Resolve a user-supplied axis. Plain column names share the level budget; explicit
    Dim/dict specs keep whatever levels they ask for. User order is preserved."""
    if specs is None:
        return None
    if isinstance(specs, (str, Dim, dict)):
        specs = [specs]
    specs = list(specs)
    if not specs:
        return []
    resolved: Dict[int, Dim] = {}
    for i, spec in enumerate(specs):
        if not isinstance(spec, str):
            resolved[i] = _resolve_spec(spec, df, prof, budget, options)
    explicit = float(np.prod([d.levels for d in resolved.values()])) if resolved else 1.0
    plain = [(i, str(spec)) for i, spec in enumerate(specs) if isinstance(spec, str)]
    if plain:
        remaining = max(2, int(budget / explicit))
        for _, col in plain:
            if col not in prof:
                raise KeyError(f"unknown column {col!r}")
        pairs = [(prof[col], df[col]) for _, col in plain]
        if any(cp.kind in (CONSTANT, ID) for cp, _ in pairs):
            planned = None
        else:
            planned = _allocate(pairs, remaining, options, slack=1.0)
        if planned is None:
            share = max(2, int(remaining ** (1.0 / len(plain))))
            planned = [_resolve_spec(col, df, prof, share if len(plain) > 1 else remaining, options) for _, col in plain]
        for (i, _), d in zip(plain, planned):
            resolved[i] = d
    return [resolved[i] for i in range(len(specs))]


def fit_layout(
    df: pd.DataFrame,
    options: Optional[FitOptions] = None,
    *,
    rows: Optional[Sequence[DimSpec]] = None,
    cols: Optional[Sequence[DimSpec]] = None,
    values: Optional[str] = None,
    agg: Optional[str] = None,
    prof: Optional[Profile] = None,
    ranked: Optional[List[Tuple[float, "Layout"]]] = None,
) -> Layout:
    """Find the best :class:`Layout` for ``df``. Any of rows/cols/values/agg may be fixed.

    Pass a list as ``ranked`` to receive every scored candidate, best first.
    """
    options = options or FitOptions()
    if df.shape[1] == 0:
        raise ValueError("cannot fit a pivot on a frame with no columns")
    if len(df) == 0:
        raise ValueError("cannot fit a pivot on an empty frame")
    prof = prof or profile(
        df, max_categories=options.max_categories, discrete_max=options.discrete_max, id_ratio=options.id_ratio
    )
    excluded = set(options.exclude)

    # ---- measure
    if values is not None:
        if values not in prof:
            raise KeyError(f"unknown values column {values!r}")
        if values == COUNT and COUNT not in df.columns:
            values = None
    fixed_dim_cols = set()
    for specs in (rows, cols):
        for sp in list(specs) if isinstance(specs, (list, tuple)) else ([specs] if specs is not None else []):
            fixed_dim_cols.add(sp.column if isinstance(sp, Dim) else sp["column"] if isinstance(sp, dict) else str(sp))
    if values is None and agg in (None, "sum", COUNT):
        auto_values, auto_agg = choose_measure(prof, excluded | fixed_dim_cols)
        if agg == COUNT:
            auto_values = None
        values, agg = auto_values, (agg if agg not in (None, COUNT) else auto_agg)
    elif values is not None and agg is None:
        agg = default_agg(prof[values])
    if values is not None and prof[values].kind not in (NUMERIC, CATEGORICAL, BOOLEAN, CONSTANT, ID):
        if agg not in ("count", "nunique"):
            raise ValueError(f"{values!r} is {prof[values].kind}; use agg='count' or 'nunique'")
    agg = agg or "sum"
    if agg not in AGGS:
        raise ValueError(f"unknown agg {agg!r}; use one of {AGGS}")

    # ---- fixed axes
    fixed_rows = _resolve_axis(rows, df, prof, options.max_rows, options)
    fixed_cols = _resolve_axis(cols, df, prof, options.max_cols, options)
    if fixed_rows is not None and fixed_cols is not None:
        return Layout(tuple(fixed_rows), tuple(fixed_cols), values, agg)

    # ---- candidates
    used = set(excluded)
    if values is not None:
        used.add(values)
    for d in (fixed_rows or []) + (fixed_cols or []):
        used.add(d.column)
    cands = [
        c
        for c in prof
        if c.name not in used and c.kind in (CATEGORICAL, BOOLEAN, DATETIME, NUMERIC, ID) and c.nunique >= 2
    ]
    cands.sort(key=lambda c: (-_prof_quality(c), -min(c.nunique, 500)))
    cands = cands[: max(1, options.search_width)]

    n = len(df)
    sample = df if n <= options.sample else df.sample(options.sample, random_state=options.seed)
    ev = _Evaluator(sample, options)
    series = {c.name: sample[c.name] for c in cands}

    def combos(pool: List[ColumnProfile], min_k: int, max_k: int):
        for k in range(min_k, max_k + 1):
            for combo in itertools.combinations(pool, k):
                yield list(combo)

    best: Optional[Tuple[float, Layout]] = None
    max_layers = max(1, options.layers)

    row_choices: List[Optional[List[Dim]]]
    if fixed_rows is not None:
        row_choices = [fixed_rows]
    else:
        row_choices = []
        for combo in combos(cands, 1, max_layers):
            dims = _order_for_axis([(c, series[c.name]) for c in combo])
            planned = _allocate(dims, options.max_rows, options)
            if planned:
                row_choices.append(planned)

    for r_dims in row_choices:
        if r_dims is None:
            continue
        r_cols = {d.column for d in r_dims}
        if fixed_cols is not None:
            col_choices: List[List[Dim]] = [fixed_cols]
        else:
            col_choices = [[]]
            pool = [c for c in cands if c.name not in r_cols]
            for combo in combos(pool, 1, max_layers):
                dims = _order_for_axis([(c, series[c.name]) for c in combo])
                planned = _allocate(dims, options.max_cols, options)
                if planned:
                    col_choices.append(planned)
        for c_dims in col_choices:
            sc = score_layout(ev.shape(r_dims, c_dims), r_dims, c_dims, options)
            if sc == -math.inf:
                continue
            if ranked is not None:
                ranked.append((sc, Layout(tuple(r_dims), tuple(c_dims), values, agg)))
            if best is None or sc > best[0]:
                best = (sc, Layout(tuple(r_dims), tuple(c_dims), values, agg))

    if ranked is not None:
        ranked.sort(key=lambda t: -t[0])
    if best is None:
        # Nothing scored: fall back to the single best-quality candidate, or a constant.
        if fixed_rows is not None:
            return Layout(tuple(fixed_rows), tuple(fixed_cols or ()), values, agg)
        for c in cands:
            d = plan_dim(series[c.name], c, options.max_rows, options)
            if d is not None:
                return Layout((d,), tuple(fixed_cols or ()), values, agg)
        first = str(df.columns[0])
        d = Dim(first, "categorical", max(1, prof[first].natural_levels), order=options.order)
        return Layout((d,), tuple(fixed_cols or ()), values, agg)
    return best[1]


# --------------------------------------------------------------------------- tables


def effective_label(dim: Dim, materialized: pd.Series) -> str:
    """The dim's label, minus a "(top N)" tag when nothing was folded into "(other)"."""
    if dim.top is not None and B.OTHER not in materialized.cat.categories:
        return dim.column
    return dim.label



def build_table(
    df: pd.DataFrame,
    layout: Layout,
    *,
    observed: bool = True,
    fill: bool = True,
) -> pd.DataFrame:
    """Pivot ``df`` according to ``layout``.

    ``observed=False`` keeps every planned level (all bins, all top-N values), which the
    histogram view wants; ``observed=True`` keeps only combinations present in the data.
    """
    work: Dict[str, pd.Series] = {}
    row_keys: List[str] = []
    col_keys: List[str] = []
    for d in layout.rows:
        m = materialize(df, d)
        work[effective_label(d, m)] = m
        row_keys.append(effective_label(d, m))
    for d in layout.cols:
        m = materialize(df, d)
        key = effective_label(d, m)
        if key in work:
            key += " "
        work[key] = m
        col_keys.append(key)
    if layout.values is None:
        vname, agg = COUNT, "sum"
        work[vname] = pd.Series(1, index=df.index, dtype="int64")
    else:
        vname, agg = layout.values, layout.agg
        v = df[layout.values]
        if pdt.is_timedelta64_dtype(v):
            v = v.dt.total_seconds()
        elif pdt.is_bool_dtype(v):
            v = v.astype(int)
        elif not pdt.is_numeric_dtype(v) and agg in ("sum", "mean", "min", "max", "median", "std"):
            raise ValueError(f"cannot {agg} non-numeric column {layout.values!r}; use agg='count' or 'nunique'")
        work[vname] = v
    frame = pd.DataFrame(work, index=df.index)
    if frame.empty:
        idx = pd.MultiIndex.from_arrays([[] for _ in row_keys], names=row_keys) if len(row_keys) > 1 else pd.Index([], name=row_keys[0] if row_keys else None)
        return pd.DataFrame(index=idx)
    fill_value: Any = 0 if (fill and agg in ("sum", "count", "nunique")) else None
    table = pd.pivot_table(
        frame,
        index=row_keys,
        columns=col_keys or None,
        values=vname,
        aggfunc=agg,
        observed=observed,
        dropna=observed,  # pivot: only observed combos; hist: keep empty bins (gaps matter)
        fill_value=fill_value,
    )
    if isinstance(table, pd.Series):
        table = table.to_frame(vname)
    if not col_keys:
        table.columns = [layout.measure]
        table.columns.name = None
    return table
