"""Column profiling: work out what each column *is* before fitting a pivot.

Every column is assigned one of a handful of kinds:

``categorical``  a discrete label (action, protocol, dst_port, country ...)
``numeric``      a continuous quantity (bytes, duration, score ...)
``boolean``      two-valued
``datetime``     timestamps (bucketed into minutes/hours/days ... when used as a dimension)
``id``           high-cardinality identifiers (session ids, hashes) - poor dimensions
``constant``     one or zero distinct values - useless for a pivot

The kind, cardinality and a few column-name hints drive the auto-fit.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pandas.api import types as pdt

NUMERIC = "numeric"
CATEGORICAL = "categorical"
BOOLEAN = "boolean"
DATETIME = "datetime"
ID = "id"
CONSTANT = "constant"

KINDS = (NUMERIC, CATEGORICAL, BOOLEAN, DATETIME, ID, CONSTANT)

# Column-name hints. Cheap, forgiving, and easy to extend.
_ID_HINTS = re.compile(
    r"(^|[_\-. ])(id|uuid|guid|hash|md5|sha\d*|session|token|key|serial|msgid|eventid)([_\-. ]|$)"
    r"|(_|-)id$|^id$",
    re.I,
)
_DISCRETE_HINTS = re.compile(
    r"(port|code|status|severity|level|priority|proto|protocol|vlan|asn|ttl|flag|type|class|"
    r"category|zone|region|country|action|verdict|result|method|rule|state|phase|tier|grade|"
    r"year|month|weekday|dow|hour_of|group|team|dept|department|host|user|name|ip|addr|domain|url|path)",
    re.I,
)
_ADDITIVE_HINTS = re.compile(
    r"(bytes|byte|count|cnt|packets|pkts|size|total|amount|sum|hits|requests|"
    r"errors|qty|quantity|volume|sales|revenue|cost|price|events|sessions|conn|connections|octets|"
    r"payload|attempts|failures|retries|units)",
    re.I,
)
_ENTITY_HINTS = re.compile(
    r"(ip|host|user|name|domain|url|addr|address|src|dst|source|dest|account|device|asset|"
    r"client|server|process|file|path|email|customer|vendor|product|item|sku|city|company)",
    re.I,
)
_TIME_HINTS = re.compile(r"(time|date|stamp|_at$|^ts$|_ts$|epoch|when|seen|created|updated)", re.I)


@dataclass(frozen=True)
class ColumnProfile:
    """What we know about a single column."""

    name: str
    kind: str
    dtype: str
    n: int
    nunique: int
    null_frac: float
    is_integer: bool
    additive_hint: bool
    discrete_hint: bool
    id_hint: bool
    time_hint: bool
    examples: Tuple[str, ...]
    entity_hint: bool = False
    position: int = 0  # column index in the source frame (earlier columns are weakly preferred)

    @property
    def unique_ratio(self) -> float:
        return self.nunique / self.n if self.n else 0.0

    @property
    def dimension_like(self) -> bool:
        """Usable as a pivot axis without binning."""
        return self.kind in (CATEGORICAL, BOOLEAN, DATETIME)

    @property
    def binnable(self) -> bool:
        return self.kind == NUMERIC

    @property
    def measure_like(self) -> bool:
        return self.kind == NUMERIC

    @property
    def natural_levels(self) -> int:
        """Distinct values plus one for nulls, when present."""
        return self.nunique + (1 if self.null_frac > 0 else 0)


@dataclass(frozen=True)
class Profile:
    """Profiles for every column of a frame."""

    columns: Dict[str, ColumnProfile]
    n_rows: int

    def __getitem__(self, name: str) -> ColumnProfile:
        return self.columns[name]

    def __contains__(self, name: object) -> bool:
        return name in self.columns

    def __iter__(self):
        return iter(self.columns.values())

    def __len__(self) -> int:
        return len(self.columns)

    def by_kind(self, *kinds: str) -> List[ColumnProfile]:
        return [c for c in self.columns.values() if c.kind in kinds]

    @property
    def dimensions(self) -> List[ColumnProfile]:
        return self.by_kind(CATEGORICAL, BOOLEAN, DATETIME)

    @property
    def measures(self) -> List[ColumnProfile]:
        return self.by_kind(NUMERIC)

    def summary(self) -> pd.DataFrame:
        """One row per column: kind, dtype, cardinality, nulls, examples."""
        rows = []
        for c in self.columns.values():
            rows.append(
                {
                    "column": c.name,
                    "kind": c.kind,
                    "dtype": c.dtype,
                    "nunique": c.nunique,
                    "null_frac": round(c.null_frac, 4),
                    "examples": ", ".join(c.examples),
                }
            )
        return pd.DataFrame(rows, columns=["column", "kind", "dtype", "nunique", "null_frac", "examples"])

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<Profile {self.n_rows:,} rows x {len(self)} columns>\n" + self.summary().to_string(index=False)


def _examples(non_null: pd.Series, k: int = 3) -> Tuple[str, ...]:
    out: List[str] = []
    try:
        uniques = non_null.drop_duplicates().head(k)
    except TypeError:  # unhashable cells (lists/dicts)
        uniques = non_null.astype(str).drop_duplicates().head(k)
    for v in uniques:
        s = str(v)
        out.append(s if len(s) <= 24 else s[:21] + "...")
    return tuple(out)


def _nunique(non_null: pd.Series) -> int:
    try:
        return int(non_null.nunique())
    except TypeError:
        return int(non_null.astype(str).nunique())


def profile_column(
    name: str,
    s: pd.Series,
    *,
    max_categories: int = 50,
    discrete_max: int = 20,
    id_ratio: float = 0.5,
    position: int = 0,
) -> ColumnProfile:
    """Profile a single column. See :func:`profile` for the knobs."""
    n = int(len(s))
    non_null = s.dropna()
    null_frac = 1.0 - (len(non_null) / n) if n else 0.0
    nunique = _nunique(non_null)
    dtype = str(s.dtype)

    is_int = bool(pdt.is_integer_dtype(s))
    if not is_int and pdt.is_float_dtype(s) and len(non_null):
        vals = non_null.to_numpy(dtype=float)
        finite = vals[np.isfinite(vals)]
        is_int = bool(finite.size) and bool(np.all(np.mod(finite, 1) == 0))

    additive = bool(_ADDITIVE_HINTS.search(name))
    discrete = bool(_DISCRETE_HINTS.search(name))
    id_hint = bool(_ID_HINTS.search(name))
    time_hint = bool(_TIME_HINTS.search(name))
    ratio = nunique / n if n else 0.0

    if nunique <= 1:
        kind = CONSTANT
    elif pdt.is_bool_dtype(s):
        kind = BOOLEAN
    elif pdt.is_datetime64_any_dtype(s) or isinstance(s.dtype, pd.PeriodDtype):
        kind = DATETIME
    elif pdt.is_timedelta64_dtype(s):
        kind = NUMERIC
    elif pdt.is_numeric_dtype(s):
        uniq = set(non_null.unique().tolist()) if nunique <= 2 else set()
        if nunique == 2 and uniq <= {0, 1, 0.0, 1.0}:
            kind = BOOLEAN
        elif discrete and (nunique <= max_categories * 4 or ratio <= id_ratio):
            kind = CATEGORICAL  # ports, status codes, severities: numbers that are labels
        elif is_int and nunique <= discrete_max:
            kind = CATEGORICAL
        elif not is_int and nunique <= max(2, discrete_max // 4):
            kind = CATEGORICAL
        elif id_hint and ratio > id_ratio:
            kind = ID
        else:
            kind = NUMERIC
    else:
        if isinstance(s.dtype, pd.CategoricalDtype):
            kind = CATEGORICAL if ratio <= id_ratio or nunique <= max_categories else ID
        elif nunique <= max_categories:
            kind = CATEGORICAL
        elif id_hint and ratio > 0.2:
            kind = ID
        elif ratio > id_ratio:
            kind = ID
        else:
            kind = CATEGORICAL  # high cardinality but repetitive: usable with top-N bucketing

    return ColumnProfile(
        name=name,
        kind=kind,
        dtype=dtype,
        n=n,
        nunique=nunique,
        null_frac=float(null_frac),
        is_integer=is_int,
        additive_hint=additive,
        discrete_hint=discrete,
        id_hint=id_hint,
        time_hint=time_hint,
        examples=_examples(non_null),
        entity_hint=bool(_ENTITY_HINTS.search(name)),
        position=int(position),
    )


def profile(
    df: pd.DataFrame,
    *,
    max_categories: int = 50,
    discrete_max: int = 20,
    id_ratio: float = 0.5,
    columns: Optional[Sequence[str]] = None,
) -> Profile:
    """Profile every column of ``df``.

    Parameters
    ----------
    max_categories:
        Text columns with at most this many distinct values are categorical outright.
        Above it they are still categorical when values repeat a lot (see ``id_ratio``).
    discrete_max:
        Integer columns with at most this many distinct values are treated as labels,
        not quantities (severity 1-5, HTTP status codes ...).
    id_ratio:
        Text columns whose distinct-count / row-count exceeds this are identifiers.
    """
    names = [str(c) for c in (columns if columns is not None else df.columns)]
    cols = {
        name: profile_column(
            name, df[name], max_categories=max_categories, discrete_max=discrete_max, id_ratio=id_ratio, position=i
        )
        for i, name in enumerate(names)
    }
    return Profile(columns=cols, n_rows=int(len(df)))
