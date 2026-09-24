import numpy as np
import pandas as pd
import pytest

import pivot2hist as p2h


def test_suggest_use_alternatives(fw):
    v = p2h.fit(fw, max_rows=12, max_cols=5)
    sugg = v.suggest(4)
    assert 1 <= len(sugg) <= 4 and sugg[0].describe() == v.layout.describe()
    assert len({l.describe() for l in sugg}) == len(sugg)
    alt = v.alternatives(3)
    assert all(a.mode == "pivot" for a in alt) and alt[1].layout == sugg[1]
    used = v.use(1)
    assert used.layout == sugg[1] and used.pivot().shape[0] <= 12
    assert v.use(sugg[0]).layout == sugg[0]
    counted = p2h.fit(fw, agg="count", max_rows=12, max_cols=5)
    assert all(l.values is None for l in counted.suggest(3))


def test_exclude_top_drill(fw):
    v = p2h.fit(fw, max_rows=12, max_cols=5)
    ex = v.exclude(action="allow")
    assert len(ex.data) == (fw["action"] != "allow").sum() and ex.slices == ["not action=allow"]
    assert len(v.exclude(action="allow").exclude(action="deny").data) == (fw["action"] == "drop").sum()
    t = v.top("src_ip", 5)
    assert t.data["src_ip"].nunique() == 5 and "src_ip\u2208top5" in t.slices
    tb = v.top("dst_port", 3, by="bytes")
    assert tb.data["dst_port"].nunique() == 3
    d = v.drill(dst_port=443)
    assert "dst_port" not in {x.column for x in d.layout.dims} and len(d.data) == (fw["dst_port"] == 443).sum()


def test_relative_time_and_percentile_slices(fw):
    v = p2h.fit(fw)
    last = v.slice(timestamp="last 24h")
    assert last.data["timestamp"].min() >= fw["timestamp"].max() - pd.Timedelta("24h")
    assert len(v.slice(timestamp="last 2 days").data) > len(last.data)
    p = v.slice(bytes=">= p95")
    assert abs(len(p.data) - 0.05 * len(fw)) < 0.01 * len(fw) + 2
    assert "p95" in p.slices[0]
    assert len(v.slice(bytes="< p50").data) < len(fw) / 2 + 2


def test_sample(fw):
    v = p2h.fit(fw).slice(action="deny")
    s = v.sample(100)
    assert len(s.data) == 100 and len(s.source) == 100 and s.slices == ["action=deny"]
    assert len(v.sample(frac=0.5).data) == pytest.approx(len(v.data) * 0.5, abs=1)


def test_level_coarser_finer_semantic_time_bins(fw):
    ip = p2h.fit(fw, rows=["src_ip"], cols=["action"], agg="count")
    c = ip.coarser("src_ip")
    assert c.layout.rows[0].level == "/24" and c.pivot().shape[0] == 2
    assert c.coarser("src_ip").layout.rows[0].level == "/16"
    assert c.finer("src_ip").layout.rows[0].level == "host"
    assert ip.level("src_ip", "/16").layout.rows[0].label == "src_ip (/16)"
    assert c.layout.measure == "count"
    tv = p2h.fit(fw, rows=["timestamp"], cols=["action"], agg="count")
    assert tv.layout.rows[0].freq == "6h"
    assert tv.coarser("timestamp").layout.rows[0].freq == "D"
    assert tv.finer("timestamp").layout.rows[0].freq == "h"
    hod = tv.level("timestamp", "hour_of_day")
    assert hod.pivot().shape[0] <= 24 and hod.layout.rows[0].label == "timestamp (hour of day)"
    wd = tv.level("timestamp", "weekday")
    assert list(wd.pivot().index)[:2] == ["Sun", "Mon"] or len(wd.pivot()) <= 7
    hb = p2h.fit(fw).histogram("bytes")
    assert len(hb.coarser("bytes").bins()) < len(hb.bins())
    assert len(hb.level("bytes", 4).bins()) <= 4
    with pytest.raises(KeyError):
        ip.level("country", "/24")
    with pytest.raises(ValueError):
        p2h.fit(fw, rows=["country"], cols=["action"]).coarser("country")


def test_cluster_rows_and_records(fw):
    v = p2h.fit(fw, max_rows=12, max_cols=5, agg="count")
    c = v.cluster(3)
    assert c.layout.rows[0].column == "cluster" and c.layout.rows[1].column == v.layout.rows[0].column
    assert c.pivot().shape == v.pivot().shape  # partitioned, not multiplied
    assert c.pivot().to_numpy().sum() == len(fw)
    assert c.layout.measure == "count"
    cc = v.cluster(3, collapse=True)
    assert cc.pivot().shape[0] == 3 and cc.pivot().to_numpy().sum() == len(fw)
    auto = v.cluster()
    assert 2 <= auto.pivot().index.get_level_values(0).nunique() <= 8
    rec = v.cluster(4, on=["bytes", "duration"])
    assert rec.layout.rows[0].column == "cluster" and rec.pivot().shape[0] == 4
    rec2 = rec.cluster(2, on="bytes")
    assert rec2.layout.rows[0].column == "cluster2"
    sliced = v.slice(action="deny").cluster(2, collapse=True)
    assert sliced.pivot().to_numpy().sum() == (fw["action"] == "deny").sum()
    assert "cluster" not in fw.columns  # source untouched
    with pytest.raises(ValueError):
        p2h.fit(fw, rows=["action"], cols=["protocol"]).slice(action="deny", refit=False).cluster(2)


def test_fit_options_pin_prefer_weights(fw):
    pinned = p2h.fit(fw, pin=["country"])
    assert "country" in {d.column for d in pinned.layout.dims}
    pref = p2h.fit(fw, prefer_rows=["timestamp"], layers=1)
    assert pref.layout.rows[0].column == "timestamp"
    prefc = p2h.fit(fw, prefer_cols=["action"], layers=1)
    assert prefc.layout.cols and prefc.layout.cols[0].column == "action"
    flat = p2h.fit(fw, weights={"layers": 5.0})
    assert len(flat.layout.dims) == 1  # every extra dimension costs 5: a single axis wins
    deep = p2h.fit(fw, weights={"layers": 0.0}, layers=2)
    assert len(deep.layout.dims) >= 3
    with pytest.raises(ValueError):
        p2h.fit(pd.DataFrame({"k": [1] * 20, "a": list("ab") * 10}), pin=["k"])
    assert "entropy" in p2h.DEFAULT_WEIGHTS


def test_variants_semantic_and_cyclic(fw):
    ranked = []
    p2h.fit_layout(fw, p2h.FitOptions(layers=1), ranked=ranked)
    descs = " | ".join(l.describe() for _, l in ranked)
    assert "hour of day" in descs or "weekday" in descs
    assert any(tag in descs for tag in ("(/8", "(/16", "(/24", "(class"))
    ranked2 = []
    p2h.fit_layout(fw, p2h.FitOptions(layers=1, variants=False), ranked=ranked2)
    assert "hour of day" not in " ".join(l.describe() for _, l in ranked2)
    lvl = p2h.fit(fw, rows=[{"column": "src_ip", "level": "/24"}], cols=["action"], agg="count")
    assert lvl.pivot().index.name == "src_ip (/24)"
    cyc = p2h.fit(fw, rows=[{"column": "timestamp", "freq": "hour_of_day"}], cols=[{"column": "timestamp", "freq": "weekday"}], agg="count")
    assert cyc.pivot().shape[1] <= 7 and cyc.pivot().to_numpy().sum() == len(fw)


def test_time_series_measure():
    rng = np.random.default_rng(0)
    ts = pd.DataFrame({"t": pd.date_range("2026-01-01", periods=2000, freq="15min"), "host": rng.choice(list("abcd"), 2000), "cpu": rng.random(2000)})
    v = p2h.fit(ts)
    assert v.profile.is_time_series and v.layout.measure == "mean(cpu)"
    assert any(d.column == "t" for d in v.layout.rows)
    log = p2h.sample.firewall_logs(500)
    assert not p2h.profile(log).is_time_series


def test_cluster_methods_and_cocluster_views(fw):
    v = p2h.fit(fw, max_rows=12, max_cols=5, agg="count")
    d = v.cluster(method="dbscan")
    assert d.layout.rows[0].column == "cluster" and d.pivot().to_numpy().sum() == len(fw)
    h = v.cluster(method="hdbscan", collapse=True)
    assert h.pivot().to_numpy().sum() == len(fw)
    r = v.cluster(on="bytes", method="dbscan")
    assert r.pivot().index.str.startswith(("c", "noise")).all()
    with pytest.raises(ValueError):
        v.cluster(method="nope")
    cc = v.cocluster()
    assert cc.layout.rows[0].column == "block_r" and cc.layout.cols[0].column == "block_c"
    assert cc.pivot().to_numpy().sum() == len(fw) and cc.pivot().shape[0] == v.pivot().shape[0]
    flat = v.cocluster(2, nest=False)
    assert flat.pivot().shape == v.pivot().shape and flat.layout.rows[0].keep is not None
    mcl = v.cocluster(method="mcl")
    assert mcl.pivot().to_numpy().sum() == len(fw)
    again = cc.cocluster()
    assert again.layout.rows[0].column == "block_r2"
    assert "cluster" not in fw.columns and "block_r" not in fw.columns
    assert cc.slice(block_r=cc.pivot().index[0][0]).pivot().shape[0] >= 1  # derived columns are sliceable
    assert cc.to_dict()["layout"]["rows"][0]["column"] == "block_r"
