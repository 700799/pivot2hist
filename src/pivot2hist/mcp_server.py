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
        "description and a markdown table instead of raw JSON rows. Call insights() when "
        "the question is 'what's actually interesting here' rather than a specific "
        "cross-tab - it's a local, non-LLM statistical pass (distributions, skew, "
        "correlated columns, surprising cells) over whatever filters you've applied. Call "
        "compare() when the question is 'how does X differ from Y' (deny vs allow, today vs "
        "yesterday, one host vs the rest): both sides land on one shared layout and it "
        "returns the cells that moved most. When one cell of a table looks interesting, "
        "explain() says what's in it (vs what independence would predict, its shares, and "
        "what sets its rows apart) and rows() returns the raw rows behind it - name the "
        "cell by the labels pivot() showed. prompt() packages any of that as one "
        "self-contained markdown prompt, for briefing another model or a sub-agent. Files "
        "are surveyed against available memory "
        "and streamed page by page when too large to hold in memory, so any source can be "
        "passed directly by path."
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
def insights(
    source: str,
    rows: Optional[List[str]] = None,
    cols: Optional[List[str]] = None,
    values: Optional[str] = None,
    agg: Optional[str] = None,
    filters: Optional[List[Dict[str, Any]]] = None,
    sensitivity: float = 0.5,
    max_findings: int = 15,
    memory_budget_mb: Optional[float] = None,
) -> Dict[str, Any]:
    """A rich, local, non-LLM analysis of the data (optionally filtered): column
    summaries (kind, semantic type, best-fit distribution, cardinality) plus ranked
    findings - skew, a mixture-model check for multiple populations in one numeric
    column, concentration, outliers, correlated column pairs, and surprising pivot
    cells. Every score is a deterministic computation over the data itself, not a model
    call. sensitivity (0..1, default 0.5) controls how much of the ranked list surfaces:
    higher shows more (including weaker findings), lower shows only the strongest. Call
    this instead of - or after - pivot() when the question is "what's actually
    interesting in this data" rather than "show me this specific cross-tab"."""
    return _agent.insights(
        source, rows=rows, cols=cols, values=values, agg=agg, filters=filters,
        sensitivity=sensitivity, max_findings=max_findings, memory_budget_mb=memory_budget_mb,
    )


@server.tool()
def compare(
    source: str,
    split: Dict[str, Any],
    vs: Optional[Dict[str, Any]] = None,
    metric: Optional[str] = None,
    n: int = 10,
    rows: Optional[List[str]] = None,
    cols: Optional[List[str]] = None,
    values: Optional[str] = None,
    agg: Optional[str] = None,
    filters: Optional[List[Dict[str, Any]]] = None,
    max_rows: int = 40,
    max_cols: int = 12,
    memory_budget_mb: Optional[float] = None,
) -> Dict[str, Any]:
    """Compare two sides of the data cell by cell on one shared pivot layout - "how does
    deny differ from allow", "today vs yesterday", "this host vs the rest". `split` is one
    filter object naming side A ({"column": "action", "eq": "deny"}, {"column": "bytes",
    "gt": 1000}, {"query": "..."}); side B is everything else, or `vs` (a second filter
    object, e.g. {"column": "timestamp", "on": "2026-03-01"}). `filters` narrow both sides
    first. The layout is auto-fitted once (or fixed via rows/cols/values/agg) and frozen
    for both sides so every cell means the same thing on each. metric: "lift" (default for
    count/sum: A's share of its total over B's share - size-independent, >1 = over-
    represented in A), "delta" (A - B, default otherwise), "ratio", "pct_change",
    "share_delta". Returns both sides' totals, the metric table, and `top`: the n cells
    that differ most with both raw values, delta, ratio, lift and only_in."""
    return _agent.compare(
        source, split=split, vs=vs, metric=metric, n=n, rows=rows, cols=cols, values=values, agg=agg,
        filters=filters, max_rows=max_rows, max_cols=max_cols, memory_budget_mb=memory_budget_mb,
    )


@server.tool()
def rows(
    source: str,
    cell: Dict[str, Any],
    n: int = 50,
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
    """The raw rows behind one cell of a pivot() table (or one bar of its histogram), with
    every filter applied - call this when a cell looks interesting. `cell` is
    {column: label} for the axis columns, using the labels pivot() showed: a plain value,
    a bin like "[1K, 2K)", a time bucket like "13:00", "(other)", "(null)", a roll-up like
    "10.0.1.0/24". One axis alone gives a whole row or column. Pass the same
    rows/cols/values/agg/filters you passed to pivot() so the labels line up. Returns
    n_returned (capped at n), n_total, and the rows as records."""
    return _agent.rows(
        source, cell=cell, n=n, rows=rows, cols=cols, values=values, agg=agg, filters=filters, mode=mode, on=on, by=by,
        bins=bins, max_rows=max_rows, max_cols=max_cols, memory_budget_mb=memory_budget_mb,
    )


@server.tool()
def explain(
    source: str,
    cell: Dict[str, Any],
    n_rows: int = 10,
    k: int = 5,
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
    """Why does this cell look the way it does? `cell` names it as in rows(). Returns the
    observed value; for a count/sum on a 2-D table the value independence of the axes
    would predict, the ratio to it and the direction (over/under); the cell's share of
    its row, column and the whole table; its rank among all cells; how many rows it holds;
    `distinguishing` - the k columns that most set those rows apart from the rest of the
    data (a label over-represented here, with its share here vs elsewhere and a lift; a
    numeric whose median differs, as a ratio), each with a ready-made sentence; the first
    n_rows rows; and a one-paragraph `text` you can quote. Deterministic and local - no
    model call - so it's cheap to run on every cell you wonder about."""
    return _agent.explain(
        source, cell=cell, n_rows=n_rows, k=k, rows=rows, cols=cols, values=values, agg=agg, filters=filters, mode=mode,
        on=on, by=by, bins=bins, max_rows=max_rows, max_cols=max_cols, memory_budget_mb=memory_budget_mb,
    )


@server.tool()
def prompt(
    source: str,
    question: Optional[str] = None,
    table: bool = True,
    profile: bool = True,
    anomalies: bool = True,
    insights: bool = False,
    sensitivity: float = 0.5,
    split: Optional[Dict[str, Any]] = None,
    vs: Optional[Dict[str, Any]] = None,
    metric: Optional[str] = None,
    cell: Optional[Dict[str, Any]] = None,
    rows: Optional[List[str]] = None,
    cols: Optional[List[str]] = None,
    values: Optional[str] = None,
    agg: Optional[str] = None,
    filters: Optional[List[Dict[str, Any]]] = None,
    mode: str = "pivot",
    on: Optional[str] = None,
    by: Optional[str] = None,
    bins: Optional[str] = None,
    table_max_rows: int = 30,
    table_max_cols: int = 12,
    max_rows: int = 40,
    max_cols: int = 12,
    memory_budget_mb: Optional[float] = None,
) -> Dict[str, Any]:
    """One self-contained analysis prompt as markdown text, built from facts computed
    locally: the dataset's columns, the pivot table with its filters, the most surprising
    cells, optionally the ranked insights, a comparison (split/vs/metric as in compare())
    and a cell in focus (cell as in explain()), then the question. Use it to brief another
    model or a sub-agent, to hand a colleague a ready-made ask, or to save an analysis
    packet - the model that receives it gets exact numbers and is told to reason only from
    them. Returns the prompt, its size in characters and a rough token estimate."""
    return _agent.prompt(
        source, question=question, table=table, profile=profile, anomalies=anomalies, insights=insights,
        sensitivity=sensitivity, split=split, vs=vs, metric=metric, cell=cell, table_max_rows=table_max_rows,
        table_max_cols=table_max_cols, rows=rows, cols=cols, values=values, agg=agg, filters=filters, mode=mode,
        on=on, by=by, bins=bins, max_rows=max_rows, max_cols=max_cols, memory_budget_mb=memory_budget_mb,
    )


if hasattr(server, "prompt"):  # MCP prompts are first-class in the protocol; register one where the SDK supports it

    @server.prompt()
    def analyze(source: str, question: str = "") -> str:
        """A ready-to-use analysis prompt for a data file: its columns, an auto-fitted
        pivot table, the most surprising cells, and your question (or a default asking
        what stands out, what to look at next, and what looks like a data-quality issue).
        Everything in it was computed locally from the data."""
        return _agent.prompt(source, question=question or None)["prompt"]


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
