"""Command line: ``pivot2hist data.csv [--hist] [--slice col=val] ...``"""
from __future__ import annotations

import argparse
import sys
from typing import Any, List, Optional, Sequence

from . import __version__, chains, fit, load, sample, stats, survey, verbose
from ._fit import AGGS
from ._profile import profile


def _parse_value(raw: str) -> Any:
    raw = raw.strip()
    if raw.lower() in ("null", "none", "nan"):
        return None
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def parse_slice(text: str) -> tuple:
    """``col=val`` | ``col=a,b,c`` | ``col=lo..hi`` | ``col>=val`` | ``col~regex``."""
    for op in (">=", "<=", "!=", "==", ">", "<", "~", "="):
        if op in text:
            col, raw = text.split(op, 1)
            col = col.strip()
            if op in ("=", "=="):
                if ".." in raw:
                    lo, hi = raw.split("..", 1)
                    return col, (_parse_value(lo) if lo.strip() else None, _parse_value(hi) if hi.strip() else None)
                if "," in raw:
                    return col, [_parse_value(v) for v in raw.split(",")]
                return col, _parse_value(raw)
            if op == "~":
                return col, "~" + raw
            return col, f"{op}{raw}"
    raise argparse.ArgumentTypeError(f"bad slice {text!r}; use col=val, col=a,b, col=lo..hi, col>=val or col~regex")


def _csv_list(text: Optional[str]) -> Optional[List[str]]:
    if text is None:
        return None
    return [t.strip() for t in text.split(",") if t.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pivot2hist",
        description="Auto-fit tabular data into a pivot table, toggle to a histogram, slice it.",
    )
    p.add_argument("file", nargs="?", help="csv/tsv/json/jsonl/parquet/xlsx file, duckdb://db?table=t (or - for stdin csv)")
    p.add_argument("--demo", choices=["firewall", "auth"], help="use a built-in sample dataset")
    p.add_argument("-V", "--version", action="version", version=f"pivot2hist {__version__}")

    d = p.add_argument_group("data size")
    d.add_argument("--survey", action="store_true", help="print the size survey and plan, then exit")
    d.add_argument("--budget", type=float, metavar="MB", help="memory budget in MB (default: half the free RAM)")
    d.add_argument("--mode", choices=["auto", "full", "downcast", "sample", "paged"], default="auto", help="how to load")
    d.add_argument("--columns", help="only load these columns (comma separated)")
    d.add_argument("--page-rows", type=int, help="rows per page in paged mode")
    d.add_argument("--table", help="duckdb table name")
    d.add_argument("--query", help="duckdb query")
    d.add_argument("-v", "--verbose", action="store_true", help="log major steps to stderr as they happen")
    d.add_argument("--stats", action="store_true", help="after the output, print the 7 costliest steps")

    c = p.add_argument_group("chains")
    c.add_argument("--chains", metavar="STATE", help="transition matrix of this state column instead of a pivot (--by = entity, --time = order)")
    c.add_argument("--time", dest="chain_time", help="time column that orders the sequences for --chains")

    g = p.add_argument_group("layout")
    g.add_argument("--rows", help="fix row dimension(s), comma separated")
    g.add_argument("--cols", help="fix column dimension(s), comma separated")
    g.add_argument("--values", help="measure column (default: an additive numeric column, else row count)")
    g.add_argument("--agg", choices=AGGS, help="aggregation for --values")
    g.add_argument("--count", action="store_true", help="measure = row count")
    g.add_argument("--max-rows", type=int, default=40)
    g.add_argument("--max-cols", type=int, default=12)
    g.add_argument("--layers", type=int, default=2, help="max stacked dimensions per axis")
    g.add_argument("--aspect", type=float, help="target rows/cols ratio")

    h = p.add_argument_group("histogram")
    h.add_argument("--hist", action="store_true", help="show the histogram instead of the pivot")
    h.add_argument("--on", help="column to bin for the histogram")
    h.add_argument("--by", help="series column(s) for the histogram")
    h.add_argument("--bins", help="bin rule (auto, fd, sturges, scott, sqrt, rice) or a count")
    h.add_argument("--scale", choices=["auto", "linear", "log"], default="auto")

    s = p.add_argument_group("slicing")
    s.add_argument("--slice", action="append", default=[], metavar="COL=VAL",
                   help="col=val | col=a,b | col=lo..hi | col>=val | col~regex (repeatable)")
    s.add_argument("--where", help="pandas query expression")

    o = p.add_argument_group("output")
    o.add_argument("--width", type=int, help="text width")
    o.add_argument("--ascii", action="store_true", help="ASCII bars")
    o.add_argument("--json", action="store_true", help="emit JSON")
    o.add_argument("--csv", action="store_true", help="emit the table as CSV")
    o.add_argument("--profile", action="store_true", help="print the column profile and exit")
    o.add_argument("--layout", action="store_true", help="print the chosen layout only")
    o.add_argument("--slicers", action="store_true", help="print available slice values")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = build_parser()
    a = p.parse_args(argv)
    if a.verbose:
        verbose()

    load_kw: dict = {}
    if a.budget:
        load_kw["memory_budget_mb"] = a.budget
    if a.mode != "auto":
        load_kw["mode"] = a.mode
    if a.columns:
        load_kw["columns"] = _csv_list(a.columns)
    if a.page_rows:
        load_kw["page_rows"] = a.page_rows
    if a.table:
        load_kw["table"] = a.table
    if a.query:
        load_kw["query"] = a.query

    if a.demo:
        df = sample.firewall_logs() if a.demo == "firewall" else sample.auth_logs()
    elif a.file == "-" or (a.file is None and not sys.stdin.isatty()):
        import pandas as pd

        df = load(pd.read_csv(sys.stdin))
    elif a.file:
        df = a.file  # surveyed and loaded (or paged) by fit()
    else:
        p.print_help()
        return 2

    if a.survey:
        print(survey(df, **{k: v for k, v in load_kw.items() if k != "mode"}).summary())
        return 0
    if a.profile:
        print(profile(load(df) if isinstance(df, str) else df).summary().to_string(index=False))
        return 0

    values = a.values
    agg = "count" if a.count else a.agg
    if a.chains:
        v = chains(load(df) if isinstance(df, str) else df, a.chains, by=_csv_list(a.by), time=a.chain_time,
                   max_rows=a.max_rows, max_cols=a.max_cols)
    else:
        v = fit(
            df, rows=_csv_list(a.rows), cols=_csv_list(a.cols), values=values, agg=agg,
            max_rows=a.max_rows, max_cols=a.max_cols, layers=a.layers, aspect=a.aspect, scale=a.scale, **load_kw,
        )
    for text in a.slice:
        col, spec = parse_slice(text)
        v = v.slice(**{col: spec})
    if a.where:
        v = v.slice(a.where)

    if a.hist or a.on or (a.by and not a.chains):
        bins: Any = a.bins
        if bins is not None and bins.isdigit():
            bins = int(bins)
        if a.on or a.by or bins is not None:
            v = v.histogram(a.on, _csv_list(a.by), bins=bins, values=values if a.on else None, agg=agg if a.on else None)
        else:
            v = v.toggle()

    if a.layout:
        print(v.layout.describe())
        return 0
    if a.slicers:
        for col, vals in v.slicers().items():
            print(f"{col}: " + ", ".join(f"{k} ({c})" for k, c in vals))
        return 0
    if a.json:
        print(v.to_json())
        return 0
    if a.csv:
        sys.stdout.write(v.table().to_csv())
        return 0
    print(v.render(width=a.width, ascii_only=a.ascii))
    if a.stats:
        print()
        print("costliest steps:")
        print(stats(7).to_string(index=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
