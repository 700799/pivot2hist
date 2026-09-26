import pytest

import pivot2hist as p2h
from pivot2hist import DEFAULT_QUESTION, Prompt


@pytest.fixture(scope="module")
def v(fw):
    return p2h.fit(fw, max_rows=12, max_cols=5).slice(protocol="TCP")


def _sections(p):
    return [line for line in str(p).splitlines() if line.startswith("## ")]


def test_prompt_default_sections_and_shape(v):
    p = v.prompt()
    assert isinstance(p, Prompt) and isinstance(p, str)
    assert p.startswith("# Data analysis request") and "computed locally" in p
    assert _sections(p) == ["## Dataset", "## Current view", "## Computed findings", "## Your task"]
    assert p.rstrip().endswith(DEFAULT_QUESTION)
    assert p.chars == len(p) and p.tokens == round(len(p) / 4) and p._repr_markdown_() == str(p)
    assert "rows in view (of 3,000 in the source), 11 columns" in p and "- `dst_port`: categorical, port" in p
    assert "Sliced to: protocol=TCP" in p and "| dst_port" in p
    assert "surprising cell" in p and "expected if the axes were independent" in p


def test_prompt_toggles(v):
    assert _sections(v.prompt(table=False)) == ["## Dataset", "## Computed findings", "## Your task"]
    assert _sections(v.prompt(profile=False, anomalies=False)) == ["## Current view", "## Your task"]
    bare = v.prompt(profile=False, table=False, anomalies=False, header=False)
    assert _sections(bare) == ["## Your task"] and bare.startswith("## Your task")
    assert "| dst_port" in v.prompt(max_rows=3) and "truncated to 3" in v.prompt(max_rows=3)


def test_prompt_question(v, tmp_path):
    p = v.prompt("Why is port 22 hot?")
    assert p.rstrip().endswith("## Your task\n\nWhy is port 22 hot?")
    assert v.prompt("   ").rstrip().endswith(DEFAULT_QUESTION)
    path = p.save(tmp_path / "p.md")
    assert open(path, encoding="utf-8").read() == str(p)


def test_prompt_insights_true_or_a_report(v):
    p = v.prompt(insights=True, sensitivity=1.0, anomalies=False)
    assert "## Computed findings" in p and "significance" in p
    planted = {"summary": "s", "findings": [{"kind": "test", "significance": 0.9, "text": "planted finding"}]}
    p2 = v.prompt(insights=planted, anomalies=False)
    assert "planted finding" in p2 and "[test, significance 0.90]" in p2
    empty = v.prompt(insights={"summary": "nothing crossed the bar", "findings": []}, anomalies=False)
    assert "- insights: nothing crossed the bar" in empty


def test_prompt_compare_forms(v):
    for spec in ({"action": "deny"}, "bytes > 5000", ("action", "deny", "allow"), v.compare(action="deny")):
        p = v.prompt(compare=spec, table=False, anomalies=False, profile=False)
        assert "## Comparison" in p and "Biggest movers" in p and "| dst_port" in p
    assert "action=deny" in v.prompt(compare={"action": "deny"}) and "action=allow" in v.prompt(compare=("action", "deny", "allow"))
    with pytest.raises(TypeError):
        v.prompt(compare=123)


def test_prompt_explain_forms(v):
    e = v.explain("22", "3")
    for spec in (e, {"dst_port": "22", "severity": "3"}, ("22", "3"), "22"):
        p = v.prompt(explain=spec, table=False, anomalies=False, profile=False)
        assert "## Cell in focus" in p and "dst_port=22" in p
    assert "What sets these rows apart" in v.prompt(explain=e)
    with pytest.raises(TypeError):
        v.prompt(explain=123)


def test_comparison_prompt(v):
    c = v.compare(action="deny")
    p = c.prompt("What changed?")
    assert _sections(p) == ["## Dataset", "## Comparison", "## Your task"] and p.rstrip().endswith("What changed?")
    assert "Comparison of action=deny (" in p and "against rest (" in p and "Biggest movers" in p
    assert "## Current view" in c.prompt(table=True)


def test_top_level_prompt_routes_fit_and_prompt_kwargs(fw):
    p = p2h.prompt(fw, "q?", rows=["dst_port"], cols=["action"], max_rows=6, table_max_rows=4, anomalies=False, insights=False, spikes=False)
    assert _sections(p) == ["## Dataset", "## Current view", "## Your task"] and p.rstrip().endswith("q?")
    assert "by dst_port (top 5) x action" in p and "truncated to 4" in p
    assert "## Comparison" in p2h.prompt(fw, compare={"action": "deny"}, anomalies=False)


def test_prompt_on_histogram_and_paged_source(fw, tmp_path):
    h = p2h.fit(fw).histogram("bytes")
    ph = h.prompt(anomalies=False)
    assert "Histogram of" in ph and "| bytes (bins)" in ph
    path = tmp_path / "fw.csv"
    fw.to_csv(path, index=False)
    paged = p2h.fit(str(path), rows=["dst_port"], cols=["action"], agg="count", mode="paged", page_rows=700)
    pp = paged.prompt(anomalies=False)
    assert "aggregated page by page" in pp and "## Current view" in pp
