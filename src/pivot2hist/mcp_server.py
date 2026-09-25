"""An MCP (Model Context Protocol) server exposing pivot2hist to any MCP-speaking agent
(Claude Code, Claude Desktop, and other MCP clients).

::

    pip install "pivot2hist[mcp]"
    pivot2hist-mcp                    # stdio server, add it to your MCP client's config
    python -m pivot2hist.mcp_server   # same thing

Each tool is a thin wrapper around :mod:`pivot2hist.agent`: JSON in, JSON out, so the
tool's declared schema (built from the function's type hints) and its docstring are the
whole contract an agent sees. Works against both the 1.x (``mcp.server.fastmcp.FastMCP``)
and 2.x (``mcp.server.mcpserver.MCPServer``) releases of the ``mcp`` SDK, whichever is
installed, since the ``@server.tool()`` / ``.run()`` surface the two share is enough here.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from mcp.server.fastmcp import FastMCP as _Server  # mcp < 2.0
except ImportError:
    try:
        from mcp.server.mcpserver import MCPServer as _Server  # mcp >= 2.0
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "pivot2hist.mcp_server needs the mcp package: pip install 'pivot2hist[mcp]'"
        ) from e

from . import agent as _agent

server = _Server(
    "pivot2hist",
    instructions=(
        "Pivot and explore tabular data (CSV/TSV/JSON/JSONL/Parquet/DuckDB, or inline "
        "records). Start with describe() to see the columns; call pivot() with no "
        "rows/cols to get an auto-fitted first look; refine with filters, or call "
        "suggest() for alternative layouts and slicers() for values to filter on. Prefer "
        "llm_context() over pivot() when you're about to reason over or quote the result "
        "yourself, rather than hand it to other code - it returns a natural-language "
        "description and a markdown table instead of raw JSON rows. Files are surveyed "
        "against available memory and streamed page by page when too large to hold in "
        "memory, so any source can be passed directly by path."
    ),
)


@server.tool()
def describe(
    source: str,
    columns: Optional[List[str]] = None,
    memory_budget_mb: Optional[float] = None,
    distributions: bool = False,
) -> Dict[str, Any]:
    """Profile a data source: column kinds, semantic types (IP, port, URL, email ...),
    cardinality, nulls, and whether it's a regular time series. Cheap even on huge files:
    reads only a bounded sample, never the whole thing. ``source`` is a file path
    (csv/tsv/json/jsonl/parquet) or a ``duckdb://path?table=name`` URL. With
    distributions=True, also fits a probability distribution (normal, lognormal,
    exponential, gamma, uniform, poisson, geometric, bernoulli...) to each numeric
    column by BIC — slower, so off by default."""
    return _agent.describe(source, columns=columns, memory_budget_mb=memory_budget_mb, distributions=distributions)


@server.tool()
def pivot(
    source: str,
    rows: Optional[List[str]] = None,
    cols: Optional[List[str]] = None,
    values: Optional[str] = None,
    agg: Optional[str] = None,
    filters: Optional[List[Dict[str, Any]]] = None,
    mode: str = "pivot",
    on: Optional[str] = None,
    by: Optional[str] = None,
    bins: Optional[str] = None,
    max_rows: int = 40,
    max_cols: int = 12,
    memory_budget_mb: Optional[float] = None,
) -> Dict[str, Any]:
    """Build a pivot table (rows x columns of an aggregated measure) or, with
    mode="hist" (or by giving `on`), a histogram of one column, and return it as JSON.

    Leave rows/cols/values empty to auto-fit the best layout. `filters` is a list of
    objects like {"column": "action", "in": ["deny", "drop"]} or
    {"column": "bytes", "gt": 1000} or {"query": "bytes > 1000 and action == 'deny'"} —
    call describe() first to see available columns, or slicers() for values to filter on.
    """
    return _agent.pivot(
        source, rows=rows, cols=cols, values=values, agg=agg, filters=filters, mode=mode,
        on=on, by=by, bins=bins, max_rows=max_rows, max_cols=max_cols, memory_budget_mb=memory_budget_mb,
    )


@server.tool()
def llm_context(
    source: str,
    rows: Optional[List[str]] = None,
    cols: Optional[List[str]] = None,
    values: Optional[str] = None,
    agg: Optional[str] = None,
    filters: Optional[List[Dict[str, Any]]] = None,
    mode: str = "pivot",
    on: Optional[str] = None,
    by: Optional[str] = None,
    bins: Optional[str] = None,
    max_rows: int = 40,
    max_cols: int = 12,
    table_max_rows: int = 30,
    table_max_cols: int = 12,
    memory_budget_mb: Optional[float] = None,
) -> Dict[str, Any]:
    """Same arguments as pivot(), but returns {"description", "metadata", "table"}
    instead of a raw table: a short natural-language summary, compact structured facts
    (measure, dims, shape, active filters, per-column kind/semantic/cardinality), and the
    table itself as a markdown string truncated to table_max_rows x table_max_cols.
    Prefer this over pivot() when you (the calling model) are about to reason over or
    quote the result, rather than pass it on to other code."""
    return _agent.llm_context(
        source, rows=rows, cols=cols, values=values, agg=agg, filters=filters, mode=mode,
        on=on, by=by, bins=bins, max_rows=max_rows, max_cols=max_cols,
        table_max_rows=table_max_rows, table_max_cols=table_max_cols, memory_budget_mb=memory_budget_mb,
    )


@server.tool()
def suggest(source: str, n: int = 5, filters: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """List the n best pivot layouts for this data, best first, each with a description
    you can turn into pivot()'s rows/cols. Use this when unsure which columns to pick."""
    return _agent.suggest(source, n, filters=filters)


@server.tool()
def slicers(source: str, columns: Optional[List[str]] = None, top: int = 10) -> Dict[str, Any]:
    """The most frequent values (with counts) per label-like column, for building
    pivot()'s `filters`."""
    return _agent.slicers(source, columns=columns, top=top)


@server.tool()
def anomalies(
    source: str,
    n: int = 10,
    rows: Optional[List[str]] = None,
    cols: Optional[List[str]] = None,
    filters: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """The n most surprising cells of a pivot: the biggest deviation between what's
    actually there and what independence of the row and column axes would predict (a
    heavy port/action combination that fires far more than its row and column totals
    alone would suggest, for instance). Needs at least 2 rows and 2 columns."""
    return _agent.anomalies(source, n, rows=rows, cols=cols, filters=filters)


def main() -> None:
    server.run()


if __name__ == "__main__":  # pragma: no cover
    main()
