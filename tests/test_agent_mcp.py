import json

import pytest

import pivot2hist as p2h
from pivot2hist import agent


def _assert_json_safe(obj):
    json.dumps(obj)  # raises TypeError on non-JSON-safe values (Timestamp, numpy scalar, ...)


def test_columns_kwarg_honored_for_frames(fw):
    v = p2h.fit(fw, columns=["action", "bytes", "dst_port"], agg="count")
    assert sorted(v.data.columns) == ["action", "bytes", "dst_port"]
    with pytest.raises(KeyError):
        p2h.fit(fw, columns=["nope"])


def test_agent_describe(fw):
    d = agent.describe(fw)
    _assert_json_safe(d)
    assert d["rows_profiled"] == len(fw)
    names = {c["name"] for c in d["columns"]}
    assert names == set(fw.columns)
    ip = next(c for c in d["columns"] if c["name"] == "src_ip")
    assert ip["kind"] == "categorical" and ip["semantic"] == "ipv4"
    assert d["time_column"] == "timestamp"
    assert "survey" not in d  # in-memory frame: no survey block


def test_agent_describe_path_is_cheap(tmp_path, fw):
    pytest.importorskip("pyarrow")
    path = tmp_path / "fw.parquet"
    fw.to_parquet(path, index=False)
    d = agent.describe(str(path))
    _assert_json_safe(d)
    assert d["survey"]["format"] == "parquet" and d["survey"]["rows"] == len(fw)
    assert d["rows_profiled"] <= len(fw)  # profiled from the probe, not a full load
    d2 = agent.describe(str(path), columns=["action", "bytes"])
    assert {c["name"] for c in d2["columns"]} == {"action", "bytes"}


def test_agent_pivot_auto_and_explicit(fw):
    auto = agent.pivot(fw, agg="count")
    _assert_json_safe(auto)
    assert auto["mode"] == "pivot" and auto["shape"][0] <= 40 and auto["shape"][1] <= 12
    explicit = agent.pivot(fw, rows=["action"], cols=["protocol"], agg="count")
    assert explicit["layout"]["rows"][0]["column"] == "action"
    assert explicit["layout"]["cols"][0]["column"] == "protocol"
    assert sum(sum(v for k, v in row.items() if k != "action") for row in explicit["table"]) == len(fw)


def test_agent_pivot_filters(fw):
    r = agent.pivot(fw, rows=["src_ip"], cols=["action"], agg="count", filters=[{"column": "action", "in": ["deny", "drop"]}])
    assert r["rows"] == int(((fw["action"] == "deny") | (fw["action"] == "drop")).sum())
    assert "action∈{deny,drop}" in r["slices"]
    r2 = agent.pivot(fw, filters=[{"query": "bytes > 1000 and action == 'deny'"}])
    assert r2["rows"] == int(((fw["bytes"] > 1000) & (fw["action"] == "deny")).sum())
    r3 = agent.pivot(fw, filters=[{"column": "bytes", "gt": 1000}])
    assert r3["rows"] == int((fw["bytes"] > 1000).sum())
    r4 = agent.pivot(fw, filters=[{"column": "bytes", "range": [100, 200]}])
    assert r4["rows"] == int(((fw["bytes"] >= 100) & (fw["bytes"] < 200)).sum())
    r5 = agent.pivot(fw, filters=[{"column": "action", "not_eq": "allow"}])
    assert r5["rows"] == int((fw["action"] != "allow").sum())
    with pytest.raises(ValueError):
        agent.pivot(fw, filters=[{"column": "bytes"}])
    with pytest.raises(ValueError):
        agent.pivot(fw, filters=[{"column": "bytes", "gt": 1, "lt": 2}])
    with pytest.raises(ValueError):
        agent.pivot(fw, filters=[{"nope": 1}])


def test_agent_pivot_hist_mode(fw):
    h = agent.pivot(fw, mode="hist", on="bytes", bins=6)
    _assert_json_safe(h)
    assert h["mode"] == "hist" and h["shape"][0] <= 6
    h2 = agent.pivot(fw, on="dst_port", by="action")
    assert h2["mode"] == "hist"


def test_agent_pivot_paged_source(tmp_path, fw):
    pytest.importorskip("pyarrow")
    path = tmp_path / "fw.parquet"
    fw.to_parquet(path, index=False)
    r = agent.pivot(str(path), agg="count", memory_budget_mb=0.05)
    _assert_json_safe(r)
    assert r.get("paged") is True and r["pages"] >= 1
    assert sum(sum(v for k, v in row.items() if not isinstance(v, str)) for row in r["table"]) == len(fw)


def test_agent_suggest(fw):
    s = agent.suggest(fw, 4)
    _assert_json_safe(s)
    assert len(s["alternatives"]) <= 4
    assert s["alternatives"][0]["description"] == s["current"]
    ranks = [a["rank"] for a in s["alternatives"]]
    assert ranks == sorted(ranks)


def test_agent_slicers(fw):
    s = agent.slicers(fw, columns=["action"], top=2)
    _assert_json_safe(s)
    assert list(s.keys()) == ["action"]
    assert len(s["action"]) <= 2
    assert s["action"][0]["count"] >= s["action"][-1]["count"]


def test_agent_records_source():
    recs = [{"a": "x", "n": 1}, {"a": "y", "n": 2}, {"a": "x", "n": 3}]
    r = agent.pivot(recs, rows=["a"], agg="count")
    _assert_json_safe(r)
    assert r["rows"] == 3


mcp = pytest.importorskip("mcp")


@pytest.fixture(scope="module")
def mcp_server_module():
    from pivot2hist import mcp_server

    return mcp_server


def test_mcp_tools_registered(mcp_server_module):
    import asyncio

    tools = asyncio.run(mcp_server_module.server.list_tools())
    names = {t.name for t in tools}
    assert names == {"describe", "pivot", "llm_context", "suggest", "slicers", "anomalies"}
    for t in tools:
        assert t.description and len(t.description) > 10


def test_mcp_call_tool_roundtrip(mcp_server_module, tmp_path, fw):
    import asyncio

    path = tmp_path / "fw.csv"
    fw.to_csv(path, index=False)

    async def run():
        r1 = await mcp_server_module.server.call_tool("describe", {"source": str(path)})
        r2 = await mcp_server_module.server.call_tool(
            "pivot", {"source": str(path), "rows": ["action"], "cols": ["protocol"], "agg": "count"}
        )
        r3 = await mcp_server_module.server.call_tool("suggest", {"source": str(path), "n": 3})
        r4 = await mcp_server_module.server.call_tool("slicers", {"source": str(path), "columns": ["action"]})
        return r1, r2, r3, r4

    r1, r2, r3, r4 = asyncio.run(run())
    d1 = json.loads(r1.content[0].text)
    assert d1["survey"]["format"] == "csv"
    d2 = json.loads(r2.content[0].text)
    assert d2["mode"] == "pivot" and d2["layout"]["rows"][0]["column"] == "action"
    d3 = json.loads(r3.content[0].text)
    assert "alternatives" in d3 and len(d3["alternatives"]) <= 3
    d4 = json.loads(r4.content[0].text)
    assert "action" in d4


def test_mcp_llm_context_tool(mcp_server_module, tmp_path, fw):
    import asyncio

    path = tmp_path / "fw.csv"
    fw.to_csv(path, index=False)

    async def run():
        return await mcp_server_module.server.call_tool(
            "llm_context", {"source": str(path), "rows": ["action"], "cols": ["protocol"], "agg": "count"}
        )

    r = asyncio.run(run())
    d = json.loads(r.content[0].text)
    assert set(d) == {"description", "metadata", "table"}
    assert d["metadata"]["measure"] == "count"
    assert d["table"].startswith("|")


def test_agent_describe_distributions(fw):
    d = agent.describe(fw, distributions=True)
    _assert_json_safe(d)
    bytes_col = next(c for c in d["columns"] if c["name"] == "bytes")
    assert bytes_col["distribution"] is not None
    assert bytes_col["distribution"]["family"] in p2h.DIST_FAMILIES
    assert isinstance(bytes_col["distribution"]["params"], dict)
    non_numeric = next(c for c in d["columns"] if c["name"] == "action")
    assert "distribution" not in non_numeric
    plain = agent.describe(fw)
    assert "distribution" not in plain["columns"][0]


def test_agent_suggest_has_confidence_and_scores(fw):
    s = agent.suggest(fw, 4)
    _assert_json_safe(s)
    assert s["confidence"] in ("high", "medium", "low")
    scores = [a["score"] for a in s["alternatives"]]
    assert scores == sorted(scores, reverse=True)


def test_agent_anomalies(fw):
    a = agent.anomalies(fw, 5, rows=["dst_port"], cols=["action"], agg="count")
    _assert_json_safe(a)
    assert len(a["cells"]) == 5
    assert set(a["cells"][0].keys()) == {"row", "col", "observed", "expected", "residual", "direction"}
    with pytest.raises(ValueError):
        agent.anomalies(fw, rows=["dst_port"], cols=[], agg="count")


def test_mcp_anomalies_and_distributions_tools(mcp_server_module, tmp_path, fw):
    import asyncio

    path = tmp_path / "fw.csv"
    fw.to_csv(path, index=False)

    async def run():
        r1 = await mcp_server_module.server.call_tool(
            "anomalies", {"source": str(path), "n": 3, "rows": ["dst_port"], "cols": ["action"]}
        )
        r2 = await mcp_server_module.server.call_tool("describe", {"source": str(path), "distributions": True})
        return r1, r2

    r1, r2 = asyncio.run(run())
    d1 = json.loads(r1.content[0].text)
    assert len(d1["cells"]) == 3
    d2 = json.loads(r2.content[0].text)
    bytes_col = next(c for c in d2["columns"] if c["name"] == "bytes")
    assert bytes_col["distribution"]["family"] in p2h.DIST_FAMILIES
