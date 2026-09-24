"""Loading tabular data from files, records and frames, with light type inference."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from ._profile import _TIME_HINTS

Source = Union[pd.DataFrame, pd.Series, str, "os.PathLike[str]", Sequence[Mapping[str, Any]], Mapping[str, Sequence[Any]], np.ndarray]


def _read_path(path: Path, **kw: Any) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext in (".csv", ".txt"):
        return pd.read_csv(path, **kw)
    if ext == ".tsv":
        kw.setdefault("sep", "\t")
        return pd.read_csv(path, **kw)
    if ext in (".gz", ".bz2", ".zip", ".xz", ".zst"):
        inner = path.with_suffix("").suffix.lower()
        if inner == ".tsv":
            kw.setdefault("sep", "\t")
        if inner in (".json", ".jsonl", ".ndjson"):
            return pd.read_json(path, lines=inner != ".json", **kw)
        return pd.read_csv(path, **kw)
    if ext in (".jsonl", ".ndjson"):
        return pd.read_json(path, lines=True, **kw)
    if ext == ".json":
        try:
            return pd.read_json(path, **kw)
        except ValueError:
            return pd.read_json(path, lines=True, **kw)
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path, **kw)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path, **kw)
    if ext in (".feather", ".ft"):
        return pd.read_feather(path, **kw)
    if ext == ".pkl":
        return pd.read_pickle(path, **kw)
    return pd.read_csv(path, sep=None, engine="python", **kw)


def infer_datetimes(df: pd.DataFrame, columns: Optional[Iterable[str]] = None, *, min_frac: float = 0.9) -> pd.DataFrame:
    """Parse text columns that look like timestamps (by name or by content) into datetimes."""
    out = df
    cols = list(columns) if columns is not None else [c for c in df.columns if pdt.is_string_dtype(df[c]) or pdt.is_object_dtype(df[c])]
    for c in cols:
        s = df[c]
        if pdt.is_datetime64_any_dtype(s):
            continue
        non_null = s.dropna()
        if non_null.empty:
            continue
        hinted = bool(_TIME_HINTS.search(str(c)))
        probe = non_null.head(200).astype(str)
        if not hinted and not probe.str.contains(r"\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|\d{2}:\d{2}", regex=True).mean() > 0.8:
            continue
        parsed = _to_datetime(probe)
        if parsed.notna().mean() < min_frac:
            continue
        full = _to_datetime(non_null.astype(str))
        if full.notna().mean() >= min_frac:
            if out is df:
                out = df.copy()
            out[c] = _to_datetime(s.astype("string"))
    return out


def _to_datetime(s: pd.Series) -> pd.Series:
    for kw in ({"format": "ISO8601"}, {"format": "mixed"}, {}):
        try:
            return pd.to_datetime(s, errors="coerce", utc=False, **kw)
        except (ValueError, TypeError):
            continue
    return pd.to_datetime(s, errors="coerce")


_BOOL_WORDS = {"true": True, "false": False, "yes": True, "no": False, "y": True, "n": False, "t": True, "f": False, "on": True, "off": False}


def infer_scalars(df: pd.DataFrame, *, min_frac: float = 0.98) -> pd.DataFrame:
    """Text columns holding numbers (``"12"``, ``"3.5"``, ``"1,024"``) or booleans
    (``yes/no``, ``true/false``) become numeric / boolean columns."""
    out = df
    for c in df.columns:
        s = df[c]
        if not (pdt.is_string_dtype(s) or pdt.is_object_dtype(s)):
            continue
        non_null = s.dropna()
        if non_null.empty:
            continue
        try:
            probe = non_null.head(500).astype(str).str.strip()
        except (TypeError, ValueError):
            continue
        low = probe.str.lower()
        if low.isin(_BOOL_WORDS.keys()).mean() >= min_frac and low.nunique() <= 2:
            full = non_null.astype(str).str.strip().str.lower()
            if full.isin(_BOOL_WORDS.keys()).mean() >= min_frac:
                if out is df:
                    out = df.copy()
                mapped = s.astype("string").str.strip().str.lower().map(_BOOL_WORDS)
                out[c] = mapped.astype("boolean") if mapped.isna().any() else mapped.astype(bool)
            continue
        cleaned = probe.str.replace(",", "", regex=False)
        num = pd.to_numeric(cleaned, errors="coerce")
        if num.notna().mean() >= min_frac and not probe.str.match(r"^0\d+$").any():
            full = pd.to_numeric(non_null.astype(str).str.strip().str.replace(",", "", regex=False), errors="coerce")
            if full.notna().mean() >= min_frac:
                if out is df:
                    out = df.copy()
                col = pd.to_numeric(s.astype("string").str.strip().str.replace(",", "", regex=False), errors="coerce")
                if col.dropna().mod(1).eq(0).all() and col.notna().all():
                    col = col.astype("int64")
                out[c] = col
    return out


def _epoch_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Integer columns named like timestamps holding epoch seconds/millis become datetimes."""
    out = df
    for c in df.columns:
        s = df[c]
        if not pdt.is_numeric_dtype(s) or not _TIME_HINTS.search(str(c)):
            continue
        v = s.dropna()
        if v.empty:
            continue
        lo, hi = float(v.min()), float(v.max())
        unit = None
        if 1e9 <= lo and hi < 1e10:
            unit = "s"
        elif 1e12 <= lo and hi < 1e13:
            unit = "ms"
        if unit:
            if out is df:
                out = df.copy()
            out[c] = pd.to_datetime(s, unit=unit, errors="coerce")
    return out


def load(source: Source, *, parse_dates: bool = True, infer_types: bool = True, **read_kwargs: Any) -> pd.DataFrame:
    """Turn almost anything tabular into a DataFrame.

    Accepts a DataFrame/Series, a file path (csv/tsv/json/jsonl/parquet/xlsx/feather,
    optionally compressed), a list of dicts, a dict of lists, or a 2-D numpy array.
    A DatetimeIndex becomes a column. With ``infer_types`` text columns holding numbers
    or yes/no words are converted; with ``parse_dates`` text that looks like timestamps
    (and epoch integers in time-named columns) is parsed. Nothing is copied unless a
    column actually changes.
    """
    if isinstance(source, pd.DataFrame):
        df = source
    elif isinstance(source, pd.Series):
        df = source.to_frame(source.name if source.name is not None else "value")
    elif isinstance(source, (str, os.PathLike)):
        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(str(p))
        df = _read_path(p, **read_kwargs)
    elif isinstance(source, np.ndarray):
        arr = source if source.ndim == 2 else source.reshape(-1, 1)
        df = pd.DataFrame(arr, columns=[f"c{i}" for i in range(arr.shape[1])])
    elif isinstance(source, Mapping):
        df = pd.DataFrame(dict(source))
    else:
        df = pd.DataFrame(list(source))
    if isinstance(df.index, pd.DatetimeIndex) or (isinstance(df.index, pd.MultiIndex) and any(
        isinstance(lvl, pd.DatetimeIndex) for lvl in df.index.levels
    )):
        df = df.reset_index()  # a time index becomes a regular (timestamp) column
    if any(not isinstance(c, str) for c in df.columns):
        df = df.copy() if df is source else df
        df.columns = [str(c) for c in df.columns]
    if len(set(df.columns)) != len(df.columns):
        df = df.copy() if df is source else df
        seen: dict = {}
        cols = []
        for c in df.columns:
            seen[c] = seen.get(c, 0) + 1
            cols.append(c if seen[c] == 1 else f"{c}.{seen[c] - 1}")
        df.columns = cols
    periods = [c for c in df.columns if isinstance(df[c].dtype, pd.PeriodDtype)]
    if periods:
        df = df.copy() if df is source else df
        for c in periods:
            df[c] = df[c].dt.to_timestamp()  # periods behave like timestamps everywhere downstream
    if infer_types:
        df = infer_scalars(df)
    if parse_dates:
        df = infer_datetimes(_epoch_columns(df))
    return df


__all__ = ["load", "infer_datetimes", "infer_scalars"]
