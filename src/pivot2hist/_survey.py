"""Survey the data and the machine, then plan countermeasures before the notebook chokes.

:func:`survey` looks at a source (frame, CSV, JSONL, Parquet, DuckDB) and reports rows,
columns, size on disk and the estimated in-memory size, next to the machine's RAM, CPU
and this process's footprint. The :class:`Plan` it produces is one of

``full``      load everything;
``downcast``  load everything, shrink dtypes (categories, smaller ints);
``paged``     never hold the whole thing: fit on a sample, aggregate page by page.

:class:`PagedSource` is what the paged plan hands to :class:`~pivot2hist.View`.
"""
from __future__ import annotations

import io
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from ._log import log, rss_bytes

SAMPLE_ROWS = 200_000
PROBE_ROWS = 20_000


# --------------------------------------------------------------------------- machine


@dataclass
class Machine:
    """RAM, CPU and this process, as far as the platform tells us."""

    ram_total_mb: Optional[float]
    ram_available_mb: Optional[float]
    cpu_count: int
    load_1m: Optional[float]
    rss_mb: Optional[float]

    @classmethod
    def probe(cls) -> "Machine":
        total = avail = None
        try:
            info: Dict[str, float] = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    k, v = line.split(":", 1)
                    info[k] = float(v.strip().split()[0]) / 1024.0  # kB -> MB
            total, avail = info.get("MemTotal"), info.get("MemAvailable")
            # containers: cgroup limits beat the host numbers
            for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
                try:
                    raw = open(p).read().strip()
                    if raw.isdigit() and int(raw) < 1 << 60:
                        limit = int(raw) / 1e6
                        if total is None or limit < total:
                            used = None
                            for q in ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
                                try:
                                    used = int(open(q).read().strip()) / 1e6
                                    break
                                except OSError:
                                    continue
                            total = limit
                            avail = max(0.0, limit - used) if used is not None else min(avail or limit, limit)
                    break
                except OSError:
                    continue
        except OSError:
            pass
        if total is None:
            try:
                import psutil

                vm = psutil.virtual_memory()
                total, avail = vm.total / 1e6, vm.available / 1e6
            except Exception:  # noqa: BLE001
                try:
                    total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e6
                except (ValueError, OSError, AttributeError):
                    total = None
        try:
            load = os.getloadavg()[0]
        except (OSError, AttributeError):
            load = None
        rss = rss_bytes()
        return cls(total, avail, os.cpu_count() or 1, load, (rss / 1e6) if rss else None)

    def summary(self) -> str:
        ram = f"{self.ram_available_mb:,.0f} MB free of {self.ram_total_mb:,.0f} MB" if self.ram_total_mb and self.ram_available_mb is not None else "RAM unknown"
        load = f", load {self.load_1m:.2f}" if self.load_1m is not None else ""
        rss = f", this process {self.rss_mb:,.0f} MB" if self.rss_mb else ""
        return f"{ram}; {self.cpu_count} CPUs{load}{rss}"


# --------------------------------------------------------------------------- survey / plan


@dataclass
class Plan:
    """What to do about the data."""

    mode: str  # full | downcast | paged
    page_rows: Optional[int] = None
    n_pages: Optional[int] = None
    sample_rows: Optional[int] = None
    columns: Optional[List[str]] = None
    reasons: List[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.mode == "paged":
            s = f"paged: {self.n_pages} pages of {self.page_rows:,} rows, fit on a {self.sample_rows:,}-row sample"
        elif self.mode == "downcast":
            s = "full load with dtype downcasting"
        else:
            s = "full load"
        if self.columns:
            s += f", {len(self.columns)} columns"
        return s + ("; " + "; ".join(self.reasons) if self.reasons else "")


@dataclass
class Survey:
    """Data size on disk and in memory, next to the machine, and the resulting plan."""

    source: str
    format: str
    disk_mb: Optional[float]
    rows: Optional[int]
    rows_exact: bool
    columns: List[str]
    dtypes: Dict[str, str]
    bytes_per_row: float
    est_memory_mb: float
    machine: Machine
    budget_mb: float
    verdict: str  # fits | tight | choke
    plan: Plan
    probe: Optional[pd.DataFrame] = None  # the rows used to calibrate (not part of the report)

    def summary(self) -> str:
        rows = f"{self.rows:,}" if self.rows is not None else "unknown"
        rows += "" if self.rows_exact else " (est.)"
        disk = f"{self.disk_mb:,.1f} MB on disk, " if self.disk_mb is not None else ""
        return (
            f"{self.format}: {disk}{rows} rows x {len(self.columns)} columns, "
            f"~{self.est_memory_mb:,.0f} MB in memory ({self.bytes_per_row:,.0f} B/row)\n"
            f"machine: {self.machine.summary()}\n"
            f"budget: {self.budget_mb:,.0f} MB -> {self.verdict}; plan: {self.plan.summary()}"
        )

    def to_dict(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k not in ("probe", "machine", "plan")}
        d["machine"] = dict(self.machine.__dict__)
        d["plan"] = dict(self.plan.__dict__)
        return d

    def __repr__(self) -> str:
        return f"<Survey {self.source}>\n{self.summary()}"


def _mem_per_row(df: pd.DataFrame) -> float:
    if len(df) == 0:
        return 64.0
    return float(df.memory_usage(deep=True).sum()) / len(df)


def _plan(rows: Optional[int], bytes_per_row: float, est_mb: float, budget_mb: float, *, pageable: bool,
          sample_rows: int, page_rows: Optional[int], columns: Optional[List[str]], disk_mb: Optional[float]) -> Tuple[str, Plan]:
    reasons: List[str] = []
    if est_mb <= 0.3 * budget_mb:
        return "fits", Plan("full", columns=columns, reasons=reasons)
    if est_mb <= budget_mb or not pageable:
        if not pageable and est_mb > budget_mb:
            reasons.append("format cannot be paged; loading anyway")
        reasons.append(f"~{est_mb:,.0f} MB vs {budget_mb:,.0f} MB budget: shrinking dtypes")
        return ("tight" if est_mb <= budget_mb else "choke"), Plan("downcast", columns=columns, reasons=reasons)
    reasons.append(f"~{est_mb:,.0f} MB exceeds the {budget_mb:,.0f} MB budget")
    if page_rows is None:
        page_rows = int(max(20_000, min(2_000_000, budget_mb * 1e6 * 0.15 / max(bytes_per_row, 1))))
    n_rows = rows if rows is not None else int(est_mb * 1e6 / max(bytes_per_row, 1))
    n_pages = max(1, math.ceil(n_rows / page_rows))
    fit_rows = int(max(5_000, min(sample_rows, n_rows, budget_mb * 1e6 * 0.3 / max(bytes_per_row, 1))))
    return "choke", Plan("paged", page_rows=page_rows, n_pages=n_pages, sample_rows=fit_rows, columns=columns, reasons=reasons)


def _budget(machine: Machine, memory_budget_mb: Optional[float]) -> float:
    if memory_budget_mb:
        return float(memory_budget_mb)
    if machine.ram_available_mb:
        return max(256.0, 0.5 * machine.ram_available_mb)
    return 2048.0


# --------------------------------------------------------------------------- sources

_TEXT_EXT = {".csv", ".tsv", ".txt", ".jsonl", ".ndjson"}
_COMPRESSED = {".gz", ".bz2", ".xz", ".zst", ".zip"}


def _fmt_of(path: Path) -> Tuple[str, bool]:
    ext = path.suffix.lower()
    compressed = ext in _COMPRESSED
    inner = path.with_suffix("").suffix.lower() if compressed else ext
    if inner in (".parquet", ".pq"):
        return "parquet", compressed
    if inner in (".jsonl", ".ndjson"):
        return "jsonl", compressed
    if inner in (".duckdb", ".ddb"):
        return "duckdb", compressed
    if inner in (".csv", ".tsv", ".txt"):
        return "csv", compressed
    return "other", compressed


def _is_duckdb_spec(source: Any) -> bool:
    if isinstance(source, str) and source.startswith("duckdb://"):
        return True
    return type(source).__name__ == "DuckDBPyConnection"


def _duckdb_parts(source: Any, query: Optional[str], table: Optional[str]) -> Tuple[Any, str, str]:
    """(connection, query, label) for a duckdb source."""
    import duckdb

    if isinstance(source, str):
        u = urlparse(source)
        path = (u.netloc + u.path) if u.netloc else u.path
        if path.startswith("/") and not os.path.exists(path) and os.path.exists(path[1:]):
            path = path[1:]
        qs = parse_qs(u.query)
        query = query or (qs.get("query") or [None])[0]
        table = table or (qs.get("table") or [None])[0]
        con = duckdb.connect(path or ":memory:", read_only=bool(path) and os.path.exists(path))
        label = source
    else:
        con, label = source, "duckdb connection"
    if query is None:
        if table is None:
            raise ValueError("duckdb sources need query=... or table=... (or ?table= in the URL)")
        query = f"SELECT * FROM {table}"
    return con, query, label


def _line_bytes(path: Path, n: int = 2000) -> float:
    """Average bytes per line from the head of a text file."""
    with open(path, "rb") as f:
        head = f.read(4 * 1024 * 1024)
    lines = head.split(b"\n")
    if len(lines) <= 2:
        return float(len(head)) or 1.0
    body = lines[1 : min(len(lines) - 1, n + 1)]
    return max(1.0, sum(len(l) + 1 for l in body) / len(body))


def _read_head(path: Path, fmt: str, n: int, columns: Optional[List[str]], read_kwargs: Dict[str, Any]) -> pd.DataFrame:
    if fmt == "csv":
        kw = dict(read_kwargs)
        if path.suffix.lower() == ".tsv" or path.with_suffix("").suffix.lower() == ".tsv":
            kw.setdefault("sep", "\t")
        return pd.read_csv(path, nrows=n, usecols=columns, **kw)
    if fmt == "jsonl":
        return pd.read_json(path, lines=True, nrows=n, **read_kwargs)
    if fmt == "parquet":
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        batch = next(pf.iter_batches(batch_size=n, columns=columns), None)
        return batch.to_pandas() if batch is not None else pf.schema_arrow.empty_table().to_pandas()
    raise ValueError(fmt)


def survey(
    source: Any,
    *,
    columns: Optional[Sequence[str]] = None,
    memory_budget_mb: Optional[float] = None,
    sample_rows: int = SAMPLE_ROWS,
    page_rows: Optional[int] = None,
    query: Optional[str] = None,
    table: Optional[str] = None,
    **read_kwargs: Any,
) -> Survey:
    """Size up ``source`` and the machine, and decide how to load it.

    ``source`` is a DataFrame, a path (csv / tsv / jsonl / parquet, optionally compressed),
    a ``duckdb://file.duckdb?table=t`` URL or a DuckDB connection with ``query=`` /
    ``table=``. ``memory_budget_mb`` defaults to half of the available RAM.
    """
    machine = Machine.probe()
    budget = _budget(machine, memory_budget_mb)
    cols = list(columns) if columns is not None else None

    with log.step("survey") as s:
        if isinstance(source, pd.DataFrame):
            df = source if cols is None else source[cols]
            bpr = _mem_per_row(df)
            est = bpr * len(df) / 1e6
            verdict, plan = _plan(len(df), bpr, est, budget, pageable=False, sample_rows=sample_rows, page_rows=page_rows, columns=cols, disk_mb=None)
            sv = Survey("DataFrame", "frame", None, len(df), True, [str(c) for c in df.columns], {str(c): str(t) for c, t in df.dtypes.items()}, bpr, est, machine, budget, verdict, plan, probe=None)
        elif _is_duckdb_spec(source):
            con, q, label = _duckdb_parts(source, query, table)
            n = int(con.execute(f"SELECT count(*) FROM ({q}) AS _t").fetchone()[0])
            sel = ", ".join(f'"{c}"' for c in cols) if cols else "*"
            probe = con.execute(f"SELECT {sel} FROM ({q}) AS _t LIMIT {PROBE_ROWS}").df()
            bpr = _mem_per_row(probe)
            est = bpr * n / 1e6
            disk = None
            if isinstance(source, str):
                p = urlparse(source)
                path = (p.netloc + p.path) if p.netloc else p.path
                if path and os.path.exists(path):
                    disk = os.path.getsize(path) / 1e6
            verdict, plan = _plan(n, bpr, est, budget, pageable=True, sample_rows=sample_rows, page_rows=page_rows, columns=cols, disk_mb=disk)
            sv = Survey(label, "duckdb", disk, n, True, [str(c) for c in probe.columns], {str(c): str(t) for c, t in probe.dtypes.items()}, bpr, est, machine, budget, verdict, plan, probe=probe)
        elif isinstance(source, (str, os.PathLike)):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(str(path))
            fmt, compressed = _fmt_of(path)
            disk = path.stat().st_size / 1e6
            if fmt == "duckdb":
                return survey(f"duckdb://{path}", columns=cols, memory_budget_mb=memory_budget_mb, sample_rows=sample_rows, page_rows=page_rows, query=query, table=table)
            if fmt == "other":
                from ._io import load

                df = load(path, **read_kwargs)
                sv = survey(df, columns=cols, memory_budget_mb=memory_budget_mb, sample_rows=sample_rows, page_rows=page_rows)
                sv.source, sv.format, sv.disk_mb = str(path), path.suffix.lstrip(".") or "file", disk
                sv.probe = df
                return sv
            probe = _read_head(path, fmt, PROBE_ROWS, cols, read_kwargs)
            bpr = _mem_per_row(probe)
            if fmt == "parquet":
                import pyarrow.parquet as pq

                n, exact = int(pq.ParquetFile(path).metadata.num_rows), True
            else:
                if compressed:
                    n, exact = int(disk * 1e6 * 4 / _line_bytes_probe(probe)), False  # ~4x text compression
                else:
                    n, exact = int(disk * 1e6 / _line_bytes(path)), False
                n = max(n, len(probe))
            est = bpr * n / 1e6
            verdict, plan = _plan(n, bpr, est, budget, pageable=True, sample_rows=sample_rows, page_rows=page_rows, columns=cols, disk_mb=disk)
            sv = Survey(str(path), fmt + (" (compressed)" if compressed else ""), disk, n, exact, [str(c) for c in probe.columns], {str(c): str(t) for c, t in probe.dtypes.items()}, bpr, est, machine, budget, verdict, plan, probe=probe)
        else:
            from ._io import load

            df = load(source, **read_kwargs)
            sv = survey(df, columns=cols, memory_budget_mb=memory_budget_mb, sample_rows=sample_rows, page_rows=page_rows)
            sv.source = type(source).__name__
            sv.probe = df
            return sv
        s.detail = f"{sv.format}: {sv.rows:,} rows{'' if sv.rows_exact else ' est.'}, ~{sv.est_memory_mb:,.0f} MB -> {sv.verdict}, {sv.plan.mode}"
    return sv


def _line_bytes_probe(probe: pd.DataFrame) -> float:
    buf = io.StringIO()
    probe.head(500).to_csv(buf, index=False, header=False)
    text = buf.getvalue()
    return max(1.0, len(text.encode()) / max(1, min(500, len(probe))))


# --------------------------------------------------------------------------- downcast


def downcast(df: pd.DataFrame, *, category_ratio: float = 0.5) -> pd.DataFrame:
    """Shrink a frame in place-ish: repetitive text -> category, ints -> smallest width, float64 -> float32."""
    out = df
    for c in df.columns:
        s = df[c]
        new = None
        if pdt.is_integer_dtype(s) and not isinstance(s.dtype, pd.CategoricalDtype):
            new = pd.to_numeric(s, downcast="integer") if s.isna().sum() == 0 else None
        elif pdt.is_float_dtype(s) and s.dtype != np.float32:
            new = s.astype(np.float32) if np.isfinite(s.to_numpy(dtype=float, na_value=np.nan)).all() or True else None
        elif (pdt.is_string_dtype(s) or pdt.is_object_dtype(s)) and len(s):
            try:
                if s.nunique(dropna=True) <= category_ratio * len(s):
                    new = s.astype("category")
            except TypeError:
                new = None
        if new is not None and new.dtype != s.dtype:
            if out is df:
                out = df.copy()
            out[c] = new
    return out


# --------------------------------------------------------------------------- paging


def _conform(page: pd.DataFrame, dtypes: Dict[str, Any]) -> pd.DataFrame:
    """Give a page the same column dtypes the sample got, so pages aggregate consistently."""
    out = page
    for c, dt in dtypes.items():
        if c not in page.columns or str(page[c].dtype) == str(dt):
            continue
        s = page[c]
        try:
            if pdt.is_datetime64_any_dtype(dt):
                new = pd.to_datetime(s, errors="coerce", utc=getattr(dt, "tz", None) is not None)
                if getattr(dt, "tz", None) is not None:
                    new = new.dt.tz_convert(dt.tz)
            elif pdt.is_bool_dtype(dt):
                if pdt.is_string_dtype(s) or pdt.is_object_dtype(s):
                    from ._io import _BOOL_WORDS

                    mapped = s.astype("string").str.strip().str.lower().map(_BOOL_WORDS)
                    new = mapped.astype("boolean") if mapped.isna().any() else mapped.astype(bool)
                else:
                    new = s.astype("boolean").astype(dt) if s.isna().any() else s.astype(bool)
            elif pdt.is_numeric_dtype(dt):
                new = pd.to_numeric(s.astype("string").str.replace(",", "", regex=False) if pdt.is_string_dtype(s) or pdt.is_object_dtype(s) else s, errors="coerce")
                new = new.astype(dt) if not new.isna().any() or pdt.is_float_dtype(dt) else new
            elif isinstance(dt, pd.CategoricalDtype):
                new = s.astype(object)
            else:
                new = s.astype(dt)
        except (TypeError, ValueError):
            continue
        if out is page:
            out = page.copy()
        out[c] = new
    return out


class PagedSource:
    """Data too big to hold: a calibrated sample plus a way to stream pages.

    ``sample`` is what fitting, profiling and slicers see; :meth:`pages` streams the whole
    source in ``plan.page_rows`` chunks, each conformed to the sample's dtypes, for exact
    page-wise aggregation.
    """

    def __init__(self, survey: Survey, opener: Callable[[], Iterator[pd.DataFrame]], sample: pd.DataFrame):
        self.survey = survey
        self._opener = opener
        self.sample = sample
        self.dtypes = {str(c): sample[c].dtype for c in sample.columns}
        self.rows_seen: Optional[int] = None

    @property
    def columns(self) -> List[str]:
        return [str(c) for c in self.sample.columns]

    def __len__(self) -> int:
        return int(self.survey.rows or len(self.sample))

    def pages(self) -> Iterator[pd.DataFrame]:
        from ._io import load

        n_pages = self.survey.plan.n_pages or 0
        total = 0
        for i, raw in enumerate(self._opener(), start=1):
            with log.step("page", f"{i}/{n_pages or '?'}") as s:
                page = _conform(load(raw), self.dtypes)
                page.columns = [str(c) for c in page.columns]
                total += len(page)
                s.detail += f": {len(page):,} rows"
            yield page
        self.rows_seen = total

    def __repr__(self) -> str:
        return f"<PagedSource {self.survey.source}: {len(self):,} rows in {self.survey.plan.n_pages} pages; sample {len(self.sample):,} rows>"


def _opener_for(sv: Survey, source: Any, query: Optional[str], table: Optional[str], read_kwargs: Dict[str, Any]) -> Callable[[], Iterator[pd.DataFrame]]:
    plan = sv.plan
    cols = plan.columns
    page_rows = int(plan.page_rows or 100_000)
    fmt = sv.format.split(" ")[0]
    if fmt == "csv":
        path = Path(sv.source)
        kw = dict(read_kwargs)
        if path.suffix.lower() == ".tsv" or path.with_suffix("").suffix.lower() == ".tsv":
            kw.setdefault("sep", "\t")

        def open_csv() -> Iterator[pd.DataFrame]:
            yield from pd.read_csv(path, chunksize=page_rows, usecols=cols, **kw)

        return open_csv
    if fmt == "jsonl":
        path = Path(sv.source)

        def open_jsonl() -> Iterator[pd.DataFrame]:
            for chunk in pd.read_json(path, lines=True, chunksize=page_rows, **read_kwargs):
                yield chunk if cols is None else chunk[[c for c in cols if c in chunk.columns]]

        return open_jsonl
    if fmt == "parquet":
        path = Path(sv.source)

        def open_parquet() -> Iterator[pd.DataFrame]:
            import pyarrow.parquet as pq

            for batch in pq.ParquetFile(path).iter_batches(batch_size=page_rows, columns=cols):
                yield batch.to_pandas()

        return open_parquet
    if fmt == "duckdb":
        con, q, _ = _duckdb_parts(source, query, table)
        sel = ", ".join(f'"{c}"' for c in cols) if cols else "*"

        def open_duckdb() -> Iterator[pd.DataFrame]:
            res = con.execute(f"SELECT {sel} FROM ({q}) AS _t")
            vectors = max(1, page_rows // 2048)
            while True:
                chunk = res.fetch_df_chunk(vectors)
                if chunk is None or len(chunk) == 0:
                    break
                yield chunk

        return open_duckdb
    raise ValueError(f"cannot page a {sv.format} source")


def _sample_for(sv: Survey, source: Any, opener: Callable[[], Iterator[pd.DataFrame]], query: Optional[str], table: Optional[str]) -> pd.DataFrame:
    """A calibration/fitting sample: spread across the source when the format allows."""
    n = int(sv.plan.sample_rows or SAMPLE_ROWS)
    fmt = sv.format.split(" ")[0]
    if fmt == "parquet":
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(sv.source)
        groups = pf.num_row_groups
        take = max(1, min(groups, 8))
        idx = sorted(set(int(round(i * (groups - 1) / max(1, take - 1))) for i in range(take))) if groups > 1 else [0]
        per = max(1, -(-n // len(idx)))
        parts = []
        for g in idx:
            tbl = pf.read_row_group(g, columns=sv.plan.columns)
            df = tbl.to_pandas()
            parts.append(df.sample(min(per, len(df)), random_state=0) if len(df) > per else df)
        return pd.concat(parts, ignore_index=True).head(n)
    if fmt == "duckdb":
        con, q, _ = _duckdb_parts(source, query, table)
        sel = ", ".join(f'"{c}"' for c in sv.plan.columns) if sv.plan.columns else "*"
        return con.execute(f"SELECT {sel} FROM ({q}) AS _t USING SAMPLE reservoir({n} ROWS) REPEATABLE (0)").df()
    parts, got = [], 0
    for chunk in opener():
        parts.append(chunk)
        got += len(chunk)
        if got >= n:
            break
    df = pd.concat(parts, ignore_index=True) if parts else (sv.probe if sv.probe is not None else pd.DataFrame())
    return df.head(n)


def load_planned(
    source: Any,
    *,
    columns: Optional[Sequence[str]] = None,
    memory_budget_mb: Optional[float] = None,
    mode: str = "auto",
    sample_rows: int = SAMPLE_ROWS,
    page_rows: Optional[int] = None,
    query: Optional[str] = None,
    table: Optional[str] = None,
    **read_kwargs: Any,
) -> Tuple[Union[pd.DataFrame, PagedSource], Survey]:
    """Survey ``source`` and load it the planned way.

    Returns ``(frame_or_paged_source, survey)``. ``mode`` forces ``"full"``, ``"downcast"``,
    ``"sample"`` (a frame of ``sample_rows``) or ``"paged"`` instead of the plan.
    """
    from ._io import load

    sv = survey(source, columns=columns, memory_budget_mb=memory_budget_mb, sample_rows=sample_rows, page_rows=page_rows, query=query, table=table, **read_kwargs)
    plan = sv.plan
    chosen = plan.mode if mode == "auto" else mode
    if chosen == "paged" and sv.format in ("frame",) or (chosen == "paged" and sv.format.split(" ")[0] not in ("csv", "jsonl", "parquet", "duckdb")):
        chosen = "downcast"
    if chosen == "paged" and plan.mode != "paged":
        # forced paging on data that would fit: still needs a page size and a sample size
        bpr = max(sv.bytes_per_row, 1.0)
        plan.page_rows = page_rows or int(max(20_000, sv.budget_mb * 1e6 * 0.15 / bpr))
        plan.n_pages = max(1, math.ceil((sv.rows or 0) / plan.page_rows))
        plan.sample_rows = min(sample_rows, sv.rows or sample_rows)
        plan.mode = "paged"
    log.info("plan", plan.summary() if chosen == plan.mode else f"{chosen} (forced; plan was {plan.mode}: {plan.summary()})")

    if chosen == "paged":
        if sv.format == "duckdb":  # duckdb hands out whole vectors of 2048 rows
            plan.page_rows = max(2048, (int(plan.page_rows or 100_000) // 2048) * 2048)
            plan.n_pages = max(1, math.ceil((sv.rows or 0) / plan.page_rows))
        opener = _opener_for(sv, source, query, table, read_kwargs)
        with log.step("sample", f"{plan.sample_rows:,} rows") as s:
            sample = load(_sample_for(sv, source, opener, query, table))
            sample.columns = [str(c) for c in sample.columns]
            s.detail = f"{len(sample):,} rows, {_mem_per_row(sample) * len(sample) / 1e6:,.0f} MB"
        return PagedSource(sv, opener, sample), sv

    with log.step("load", sv.source) as s:
        if sv.format == "frame":
            df = source if columns is None else source[list(columns)]
            df = load(df)
        elif sv.format == "duckdb":
            con, q, _ = _duckdb_parts(source, query, table)
            sel = ", ".join(f'"{c}"' for c in columns) if columns else "*"
            if chosen == "sample":
                df = con.execute(f"SELECT {sel} FROM ({q}) AS _t USING SAMPLE reservoir({sample_rows} ROWS) REPEATABLE (0)").df()
            else:
                df = con.execute(f"SELECT {sel} FROM ({q}) AS _t").df()
            df = load(df)
        elif sv.probe is not None and sv.format not in ("csv", "jsonl", "parquet") and not sv.format.startswith(("csv", "jsonl", "parquet")):
            df = load(sv.probe if columns is None else sv.probe[list(columns)])
        else:
            fmt = sv.format.split(" ")[0]
            path = Path(sv.source)
            if chosen == "sample":
                opener = _opener_for(sv, source, query, table, read_kwargs)
                sv.plan.page_rows = sv.plan.page_rows or min(sample_rows, 200_000)
                sv.plan.sample_rows = sample_rows
                df = load(_sample_for(sv, source, opener, query, table))
            elif fmt == "parquet":
                df = load(pd.read_parquet(path, columns=list(columns) if columns else None))
            elif fmt == "jsonl":
                df = load(pd.read_json(path, lines=True, **read_kwargs))
                if columns:
                    df = df[[c for c in columns if c in df.columns]]
            else:
                kw = dict(read_kwargs)
                if path.suffix.lower() == ".tsv" or path.with_suffix("").suffix.lower() == ".tsv":
                    kw.setdefault("sep", "\t")
                df = load(pd.read_csv(path, usecols=list(columns) if columns else None, **kw))
        df.columns = [str(c) for c in df.columns]
        if chosen == "downcast":
            df = downcast(df)
        s.detail = f"{len(df):,} rows x {df.shape[1]} cols, {_mem_per_row(df) * len(df) / 1e6:,.0f} MB" + (" (downcast)" if chosen == "downcast" else "")
    return df, sv


__all__ = ["survey", "Survey", "Plan", "Machine", "PagedSource", "load_planned", "downcast"]
