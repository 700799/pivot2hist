import json

import numpy as np
import pandas as pd
import pytest

import pivot2hist as p2h
from pivot2hist import agent
from pivot2hist._novelty import COLUMNS, default_entity_columns, split_point


@pytest.fixture(scope="module")
def novel_df(fw):
    """The firewall sample with, on the last day: a brand-new source scanning 40 ports, and
    the busiest known source using a port nobody had ever used."""
    rng = np.random.default_rng(5)
    scan = fw.sample(60, random_state=2).copy()
    scan["timestamp"] = pd.Timestamp("2026-03-07 09:00") + pd.to_timedelta(rng.uniform(0, 7200, 60), unit="s")
    scan["src_ip"] = "10.9.9.9"
    scan["dst_port"] = rng.choice(np.arange(20000, 20040), 60)
    known = fw.sample(8, random_state=3).copy()
    known["timestamp"] = pd.Timestamp("2026-03-07 10:00") + pd.to_timedelta(rng.uniform(0, 3600, 8), unit="s")
    known["src_ip"] = fw["src_ip"].value_counts().index[0]
    known["dst_port"] = 31337
    return pd.concat([fw, scan, known], ignore_index=True)


@pytest.fixture(scope="module")
def v(novel_df):
    return p2h.fit(novel_df)


def test_new_entity_and_new_value_lead(v, novel_df):
    r = v.novel("src_ip", "dst_port", since="24h")
    assert list(r.columns) == COLUMNS
    kinds = dict(zip(r["entity"], r["kind"]))
    assert kinds["10.9.9.9"] == "new entity"
    busiest = str(novel_df["src_ip"].value_counts().index[0])
    assert kinds[busiest] == "new value"
    scan = r[r["entity"] == "10.9.9.9"].iloc[0]
    spread = novel_df.loc[novel_df["src_ip"] == "10.9.9.9", "dst_port"].nunique()
    assert scan["after"] == 60 and scan["before"] == 0 and f"{spread} distinct dst_port" in scan["text"]
    port = r[r["entity"] == busiest].iloc[0]
    assert port["value"] == "31337" and port["peers"] == 0 and port["after"] == 8
    assert (r["score"].diff().dropna() <= 0).all()
    assert r.attrs["since"] == novel_df["timestamp"].max() - pd.Timedelta("24h")


def test_default_columns_and_split(v):
    assert default_entity_columns(v) == ("src_ip", "dst_port")
    r = v.novel(n=5)
    assert not r.empty and r.iloc[0]["kind"] in ("new entity", "new value")
    assert set(r["kind"]) <= {"new entity", "new pair", "new value", "fan-out"}
    assert v.novel().equals(v.novel("src_ip", "dst_port", since=0.25))


def test_new_pair_reports_peers(v):
    r = v.novel("src_ip", "dst_port", since=0.25, n=50)
    pairs = r[r["kind"] == "new pair"]
    assert not pairs.empty
    assert (pairs["peers"] >= 1).all() and (pairs["before"] == 0).all() and (pairs["after"] >= 3).all()
    # a value many peers already used scores below a value few used, at equal support
    if len(pairs) > 1:
        same = pairs[pairs["after"] == pairs["after"].iloc[0]]
        if len(same) > 1:
            assert (same.sort_values("peers")["score"].diff().dropna() <= 0).all()


def test_fan_out(fw):
    rng = np.random.default_rng(7)
    history = fw[fw["timestamp"] < fw["timestamp"].max() - pd.Timedelta("24h")]
    spread = history.groupby("src_ip")["dst_port"].nunique()
    quiet = spread[(spread <= 3) & (history["src_ip"].value_counts().reindex(spread.index) >= 3)].index[0]  # a known, narrow source
    burst = fw.sample(30, random_state=4).copy()
    burst["timestamp"] = pd.Timestamp("2026-03-07 12:00") + pd.to_timedelta(rng.uniform(0, 3600, 30), unit="s")
    burst["src_ip"] = quiet
    burst["dst_port"] = rng.choice(np.arange(40000, 40030), 30)
    r = p2h.fit(pd.concat([fw, burst], ignore_index=True)).novel("src_ip", "dst_port", since="24h", n=50)
    fan = r[(r["kind"] == "fan-out") & (r["entity"] == str(quiet))]
    assert len(fan) == 1 and fan.iloc[0]["after"] >= fan.iloc[0]["before"] * 2 and "fans out" in fan.iloc[0]["text"]
    assert fan.iloc[0]["before"] == spread[quiet]


def test_no_attr_only_new_entities(v):
    r = v.novel("src_ip", since="24h")
    assert set(r["kind"]) == {"new entity"} and r["value"].isna().all()


def test_split_point_forms():
    t = pd.Series(pd.date_range("2026-03-01", "2026-03-09", freq="h"))
    assert split_point(t, 0.5) == pd.Timestamp("2026-03-05")
    assert split_point(t, "24h") == pd.Timestamp("2026-03-08")
    assert split_point(t, "2D") == pd.Timestamp("2026-03-07")
    assert split_point(t, "2026-03-06") == pd.Timestamp("2026-03-06")
    assert split_point(t, pd.Timestamp("2026-03-06 12:00")) == pd.Timestamp("2026-03-06 12:00")
    assert split_point(t, None) == split_point(t, 0.25)
    with pytest.raises(ValueError, match="fraction"):
        split_point(t, 1.5)
    with pytest.raises(ValueError, match="no usable"):
        split_point(pd.Series([pd.NaT, pd.NaT]), 0.5)


def test_bad_arguments(v, fw):
    with pytest.raises(KeyError):
        v.novel("nope")
    with pytest.raises(KeyError):
        v.novel("src_ip", "nope")
    with pytest.raises(ValueError, match="different"):
        v.novel("src_ip", "src_ip")
    with pytest.raises(ValueError, match="no datetime"):
        p2h.fit(fw.drop(columns=["timestamp"])).novel("src_ip")
    with pytest.raises(KeyError):
        v.novel("src_ip", time="nope")


def test_min_support_and_empty(v):
    r = v.novel("src_ip", "dst_port", since="24h", min_support=100)
    assert r.empty and list(r.columns) == COLUMNS


def test_string_columns_survive_mixed_types(fw):
    df = fw.copy()
    df["dst_port"] = df["dst_port"].astype(object)
    df.loc[df.index[:5], "dst_port"] = "http"
    r = p2h.fit(df).novel("src_ip", "dst_port", n=5)
    assert list(r.columns) == COLUMNS


def test_top_level_function(novel_df):
    r = p2h.novel(novel_df, "src_ip", "dst_port", since="24h", n=2)
    assert len(r) == 2 and "new entity" in set(r["kind"])
    assert set(p2h.NOVELTY_KINDS) == {"new entity", "new pair", "new value", "fan-out"}


def test_insights_include_novelty(v):
    report = v.insights()
    hits = [f for f in report["findings"] if f["kind"] == "novelty"]
    assert hits and hits[0]["columns"] == ["src_ip", "dst_port"] and 0 < hits[0]["significance"] <= 1
    assert hits[0]["detail"]["kind"] in ("new entity", "new value") and hits[0]["detail"]["since"]
    json.dumps(report)


def test_agent_and_mcp_novel(novel_df, tmp_path):
    r = agent.novel(novel_df, "src_ip", "dst_port", since="24h", n=3)
    assert set(r) == {"entity", "attr", "since", "n", "findings"} and 1 <= r["n"] == len(r["findings"]) <= 3
    assert r["findings"][0]["kind"] in ("new entity", "new value") and "text" in r["findings"][0]
    json.dumps(r)
    auto = agent.novel(novel_df, n=2)
    assert auto["entity"] is None and len(auto["findings"]) == 2
    pytest.importorskip("mcp")
    import asyncio

    from pivot2hist import mcp_server

    path = tmp_path / "fw.csv"
    novel_df.to_csv(path, index=False)

    async def run():
        a = await mcp_server.server.call_tool("novel", {"source": str(path), "since": "24h", "n": 2})
        b = await mcp_server.server.call_tool("novel", {"source": str(path), "since": "0.5", "entity": "src_ip", "n": 1})
        return a, b

    a, b = asyncio.run(run())
    ra, rb = json.loads(a.content[0].text), json.loads(b.content[0].text)
    assert ra["n"] == 2 and rb["n"] == 1 and rb["entity"] == "src_ip"
