import json

import pandas as pd
import pytest

import pivot2hist as p2h
from pivot2hist._llm import markdown_table


@pytest.fixture(scope="module")
def fw():
    return p2h.sample.firewall_logs(3000, seed=1)


def test_llm_context_shape(fw):
    v = p2h.fit(fw)
    ctx = v.llm_context()
    assert set(ctx) == {"description", "metadata", "table"}
    assert isinstance(ctx["description"], str) and ctx["description"]
    assert isinstance(ctx["metadata"], dict)
    assert isinstance(ctx["table"], str) and ctx["table"].startswith("|")


def test_llm_context_is_json_safe(fw):
    v = p2h.fit(fw)
    ctx = v.llm_context()
    # round-trips through json with no TypeError (no Timestamp/np.int64/etc left over)
    round_tripped = json.loads(json.dumps(ctx))
    assert round_tripped["description"] == ctx["description"]


def test_llm_context_metadata_fields(fw):
    v = p2h.fit(fw, rows=["dst_ip"], cols=["rule"], values="bytes", agg="sum")
    ctx = v.llm_context()
    m = ctx["metadata"]
    assert m["mode"] == "pivot"
    assert m["measure"] == "sum(bytes)"
    assert m["source_rows"] == len(fw)
    assert m["shape"]["rows"] > 0 and m["shape"]["cols"] > 0
    assert "dst_ip" in m["columns"] and m["columns"]["dst_ip"]["semantic"] == "ipv4"
    assert "rule" in m["columns"] and "bytes" in m["columns"]
    # metadata is scoped to columns actually used, not a full profile dump
    assert "timestamp" not in m["columns"]


def test_llm_context_description_mentions_shape_and_measure(fw):
    v = p2h.fit(fw, rows=["action"], cols=["protocol"], values="bytes", agg="sum")
    ctx = v.llm_context()
    assert "sum(bytes)" in ctx["description"]
    assert "3,000 rows" in ctx["description"]


def test_llm_context_notes_toggle(fw):
    v = p2h.fit(fw, rows=["action"], cols=["protocol"])
    with_notes = v.llm_context(notes=True)
    without_notes = v.llm_context(notes=False)
    assert "notable" in with_notes["metadata"]
    assert "notable" not in without_notes["metadata"]


def test_llm_context_no_anomalies_for_1d_layout(fw):
    v = p2h.fit(fw, rows=["action"], cols=[])
    ctx = v.llm_context()
    assert "notable" not in ctx["metadata"]
    assert "surprising" not in ctx["description"]


def test_llm_context_reflects_slices(fw):
    v = p2h.fit(fw).slice(action="deny")
    ctx = v.llm_context()
    assert ctx["metadata"]["slices"] == v.slices
    assert "Sliced to" in ctx["description"]


def test_llm_context_histogram_mode(fw):
    h = p2h.fit(fw, rows=["bytes"], values=None, agg=None).toggle()
    ctx = h.llm_context()
    assert ctx["metadata"]["mode"] == "hist"
    assert "Histogram" in ctx["description"]


def test_top_level_llm_context_matches_view_method(fw):
    a = p2h.llm_context(fw, rows=["action"], cols=["protocol"])
    b = p2h.fit(fw, rows=["action"], cols=["protocol"]).llm_context()
    assert a["metadata"]["measure"] == b["metadata"]["measure"]
    assert a["table"] == b["table"]


def test_agent_llm_context_is_json_safe(fw):
    ctx = p2h.agent.llm_context(fw, rows=["action"])
    json.dumps(ctx)  # must not raise
    assert set(ctx) == {"description", "metadata", "table"}


def test_agent_llm_context_with_filters(fw):
    ctx = p2h.agent.llm_context(fw, rows=["action"], filters=[{"column": "action", "eq": "deny"}])
    assert "action=deny" in ctx["metadata"]["slices"][0] or "deny" in ctx["metadata"]["slices"][0]


# --------------------------------------------------------------------------- markdown_table


def test_markdown_table_basic_rendering():
    df = pd.DataFrame({"x": [1, 2], "y": [3.14159, 2.71828]}, index=pd.Index(["a", "b"], name="key"))
    md, truncated = markdown_table(df)
    assert not truncated
    assert "| key | x | y |" in md
    assert "| a | 1 | 3.142 |" in md


def test_markdown_table_escapes_pipes_and_newlines():
    df = pd.DataFrame({"x|y": ["has|pipe", "has\nnewline"]})
    md, _ = markdown_table(df)
    assert "has\\|pipe" in md
    assert "\n\n" not in md.split("\n", 2)[2]  # the embedded newline was flattened, not left raw


def test_markdown_table_truncates_not_samples():
    df = pd.DataFrame({f"c{i}": range(50) for i in range(20)})
    md, truncated = markdown_table(df, max_rows=5, max_cols=3)
    assert truncated
    assert "truncated to 5 of 50 rows x 3 of 20 columns" in md
    lines = [l for l in md.split("\n") if l.startswith("|")]
    assert len(lines) == 2 + 5  # header + separator + 5 data rows


def test_markdown_table_empty_frame():
    md, truncated = markdown_table(pd.DataFrame())
    assert not truncated
    assert "empty" in md


def test_markdown_table_multiindex_rows_and_cols():
    idx = pd.MultiIndex.from_tuples([("a", 1), ("b", 2)], names=["outer", "inner"])
    cols = pd.MultiIndex.from_tuples([("m", "x"), ("m", "y")])
    df = pd.DataFrame([[1, 2], [3, 4]], index=idx, columns=cols)
    md, _ = markdown_table(df)
    assert "outer / inner" in md
    assert "m / x" in md


def test_markdown_table_nan_renders_blank():
    import numpy as np

    df = pd.DataFrame({"a": [1.0, np.nan]})
    md, _ = markdown_table(df)
    lines = md.split("\n")
    assert lines[2] == "| 0 | 1 |"
    assert lines[3] == "| 1 |  |"
