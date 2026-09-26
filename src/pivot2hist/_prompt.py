"""Package what is on screen as one self-contained prompt for any LLM.

pivot2hist never calls a model. This turns the facts it computed locally - the dataset's
shape and column types, the current table with its slices, the surprising cells and
ranked insights, a comparison, an explained cell - into one markdown document a model
can reason over, with the question at the end. Paste it into any chat, hand it to an
agent, or save it: the model gets exact numbers instead of a screenshot or a re-typed
summary, and it is told which facts it may rely on.
"""
from __future__ import annotations

import os
from typing import Any, Mapping, Optional, Union

from ._log import log
from ._profile import DATETIME
from ._render import fmt_cell

DEFAULT_QUESTION = (
    "Summarise what stands out in this view in a few bullet points, referring to specific cells and "
    "numbers. Then suggest the three most useful next steps - a slice, a comparison, a cell to drill "
    "into - and say why each would be informative. Flag anything that looks like a data-quality "
    "problem. If something is not supported by the facts above, say so rather than guessing."
)

HEADER = (
    "# Data analysis request\n\n"
    "The facts below were computed locally by pivot2hist from the actual data; nothing here is invented "
    "or sampled from a model. Reason only from these facts, and quote specific cells and numbers when "
    "you do."
)


class Prompt(str):
    """A prompt: a plain string that previews as markdown in a notebook, knows its rough
    token count and can :meth:`save` itself."""

    def _repr_markdown_(self) -> str:
        return str(self)

    @property
    def chars(self) -> int:
        return len(self)

    @property
    def tokens(self) -> int:
        """A rough estimate (about four characters per token for English and markdown)."""
        return max(1, round(len(self) / 4))

    def save(self, path: Union[str, "os.PathLike[str]"]) -> str:
        """Write the prompt to ``path`` (markdown) and return the path."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(self))
        return os.fspath(path)


def _dataset(view: Any, *, max_columns: int = 40) -> str:
    prof = view.profile
    n = len(view.data)
    total = len(view.paged) if view.paged is not None else len(view.source)
    head = f"{n:,} rows in view" + (f" (of {total:,} in the source)" if n != total else "") + f", {len(prof)} columns."
    if view.paged is not None:
        head += " The source is larger than the memory budget and was aggregated page by page; the column facts below come from its sample."
    lines = ["## Dataset", "", head]
    if prof.is_time_series and prof.time_column:
        lines.append(f"Regular time series on `{prof.time_column}`.")
    lines.append("")
    cols = list(prof)
    for cp in cols[:max_columns]:
        bits = [cp.kind]
        if cp.semantic:
            bits.append(cp.semantic)
        if cp.kind == DATETIME and cp.ts_freq:
            bits.append(f"~every {cp.ts_freq}")
        line = f"- `{cp.name}`: {', '.join(bits)}; {cp.nunique:,} distinct"
        if cp.null_frac > 0:
            line += f"; {cp.null_frac:.0%} null"
        ex = ", ".join(str(x) for x in cp.examples[:3])
        if ex:
            line += f"; e.g. {ex}"
        lines.append(line)
    if len(cols) > max_columns:
        lines.append(f"- ... {len(cols) - max_columns} more columns")
    return "\n".join(lines)


def _view_section(view: Any, *, max_rows: int, max_cols: int) -> str:
    ctx = view.llm_context(max_rows=max_rows, max_cols=max_cols, notes=False)
    return "## Current view\n\n" + ctx["description"] + "\n\n" + ctx["table"]


def _findings_section(view: Any, *, anomalies: bool, n_anomalies: int, insights: Any, sensitivity: float,
                      spikes: bool = True, n_spikes: int = 3) -> str:
    items = []
    if spikes:
        from ._spikes import default_time_column, describe_spike

        column = default_time_column(view)
        top = None
        if column is not None and view.layout.rows:
            try:
                top = view.spikes(column, n=n_spikes)
            except Exception:  # noqa: BLE001 - a bonus fact, not a requirement
                top = None
        if top is not None:
            for r in top.itertuples():
                items.append("- over time: " + describe_spike(r, view.layout.measure, column))
    if anomalies:
        try:
            top = view.anomalies(n_anomalies)
        except Exception:  # noqa: BLE001 - needs a 2x2+ additive pivot; a bonus fact, not a requirement
            top = None
        if top is not None:
            for r in top.itertuples():
                items.append(
                    f"- surprising cell {r.row} / {r.col}: {fmt_cell(r.observed)} observed vs {fmt_cell(r.expected)} "
                    f"expected if the axes were independent ({r.direction}, residual {r.residual:+.1f})"
                )
    report = None
    if insights is True:
        report = view.insights(sensitivity=sensitivity)
    elif isinstance(insights, Mapping):
        report = insights
    if report is not None:
        findings = report.get("findings") or []
        for f in findings:
            items.append(f"- [{f['kind']}, significance {f['significance']:.2f}] {f['text']}")
        if not findings and report.get("summary"):
            items.append(f"- insights: {report['summary']}")
    if not items:
        return ""
    return "## Computed findings\n\n" + "\n".join(items)


def _comparison_section(c: Any, *, max_rows: int, max_cols: int, top: int) -> str:
    ctx = c.llm_context(max_rows=max_rows, max_cols=max_cols, top=top)
    return "## Comparison\n\n" + ctx["description"] + "\n\n" + ctx["table"]


def _cell_section(e: Mapping[str, Any]) -> str:
    return "## Cell in focus\n\n" + str(e["text"])


def _resolve_compare(view: Any, compare: Any) -> Any:
    from ._compare import Comparison

    if compare is None or compare is False:
        return None
    if isinstance(compare, Comparison):
        return compare
    if isinstance(compare, (str, Mapping)):
        return view.compare(compare)
    if isinstance(compare, tuple):
        return view.compare(*compare)
    raise TypeError("compare must be a Comparison, a split (column=value mapping or query string) or a tuple of compare() arguments")


def _resolve_explain(view: Any, explain: Any) -> Any:
    from ._explain import Explanation

    if explain is None or explain is False:
        return None
    if isinstance(explain, Explanation):
        return explain
    if isinstance(explain, Mapping):
        return view.explain(**explain)
    if isinstance(explain, tuple):
        return view.explain(*explain)
    if isinstance(explain, str):
        return view.explain(explain)
    raise TypeError("explain must be an Explanation, a {column: label} mapping, a (row_label, col_label) tuple or a row label")


def build_prompt(
    view: Any,
    question: Optional[str] = None,
    *,
    table: bool = True,
    profile: bool = True,
    anomalies: bool = True,
    n_anomalies: int = 3,
    spikes: bool = True,
    n_spikes: int = 3,
    insights: Any = False,
    sensitivity: float = 0.5,
    compare: Any = None,
    explain: Any = None,
    max_rows: int = 30,
    max_cols: int = 12,
    top: int = 5,
    header: bool = True,
) -> Prompt:
    """See :meth:`View.prompt`."""
    with log.step("prompt", view.layout.describe()):
        parts = [HEADER] if header else []
        if profile:
            parts.append(_dataset(view))
        if table:
            parts.append(_view_section(view, max_rows=max_rows, max_cols=max_cols))
        if anomalies or insights or spikes:
            found = _findings_section(view, anomalies=anomalies, n_anomalies=n_anomalies, insights=insights, sensitivity=sensitivity,
                                      spikes=spikes, n_spikes=n_spikes)
            if found:
                parts.append(found)
        c = _resolve_compare(view, compare)
        if c is not None:
            parts.append(_comparison_section(c, max_rows=max_rows, max_cols=max_cols, top=top))
        e = _resolve_explain(view, explain)
        if e is not None:
            parts.append(_cell_section(e))
        q = question.strip() if isinstance(question, str) and question.strip() else DEFAULT_QUESTION
        parts.append("## Your task\n\n" + q)
        return Prompt("\n\n".join(parts) + "\n")


__all__ = ["Prompt", "build_prompt", "DEFAULT_QUESTION", "HEADER"]
