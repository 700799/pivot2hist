import json
import math

import numpy as np
import pandas as pd
import pytest

import pivot2hist as p2h
from pivot2hist import Comparison, Facets, agent


@pytest.fixture(scope="module")
def v(fw):
    return p2h.fit(fw, max_rows=12, max_cols=5)


@pytest.fixture
def tiny():
    # x is the layout row; g is the split. A = g=a: p x3, q x1. B = g=b: p x2, q x6.
    return pd.DataFrame({"g": ["a"] * 4 + ["b"] * 8, "x": ["p", "p", "p", "q"] + ["p", "p"] + ["q"] * 6})


# --------------------------------------------------------------------------- construction


def test_split_vs_rest_shares_one_layout(v):
    c = v.compare(action="deny")
    assert isinstance(c, Comparison)
    assert c.names == ("action=deny", "rest") and c.metric == "lift"
    assert c.a.layout == c.b.layout == c.layout
    ta, tb = c.sides()
    assert ta.shape == tb.shape == c.table().shape
    assert list(ta.index) == list(tb.index) and list(ta.columns) == list(tb.columns)
    assert len(c.a.data) + len(c.b.data) == len(v.data)


def test_split_column_on_an_axis_is_taken_off_and_axis_refilled(v):
    assert any(d.column == "severity" for d in v.layout.cols)
    c = v.compare(severity=5)
    assert "severity" not in {d.column for d in c.layout.dims}
    assert c.layout.cols  # refilled, still 2-D
    assert c.table().shape[1] > 1


def test_all_split_forms(v, fw):
    a = v.compare("action", "deny")
    b = v.compare("action", "deny", "allow")
    q = v.compare("bytes > 5000")
    two = v.compare({"timestamp": "2026-03-02"}, {"timestamp": "2026-03-01"})
    qq = v.compare("action == 'deny'", "action == 'drop'")
    assert a.names == ("action=deny", "rest")
    assert b.names == ("action=deny", "action=allow") and len(b.b.data) == (fw["action"] == "allow").sum()
    assert q.names == ("bytes > 5000", "rest") and len(q.a.data) == (fw["bytes"] > 5000).sum()
    assert two.names == ("timestamp=2026-03-02", "timestamp=2026-03-01")
    assert qq.names == ("action == 'deny'", "action == 'drop'")


def test_two_views_other_is_laid_out_like_self(v):
    today, yesterday = v.slice(timestamp="2026-03-02"), v.slice(timestamp="2026-03-01").relayout(rows=["country"])
    c = today.compare(yesterday)
    dims = [d.column for d in c.b.layout.dims]
    assert c.b.layout == c.a.layout and dims == [d.column for d in today.layout.dims]
    assert dims != [d.column for d in yesterday.layout.dims]
    assert c.metric == "delta"
    assert c.names == ("timestamp=2026-03-02", "timestamp=2026-03-01")


def test_two_frames_missing_column_is_a_clear_error(fw):
    a = p2h.fit(fw, rows=["dst_port"], cols=["action"])
    b = p2h.fit(fw.drop(columns=["action"]), rows=["dst_port"])
    with pytest.raises(KeyError, match="action"):
        a.compare(b)
    ok = a.compare(p2h.fit(fw.sample(500, random_state=1), rows=["dst_port"]))
    assert ok.b.layout == ok.a.layout and [d.column for d in ok.layout.dims] == ["dst_port", "action"]


def test_names_override_and_identical_names_disambiguated(v, fw):
    assert v.compare(action="deny", names=("blocked", "allowed")).names == ("blocked", "allowed")
    same = p2h.fit(fw, rows=["dst_port"]).compare(p2h.fit(fw, rows=["dst_port"]))
    assert same.names == ("A", "B")
    s = v.slice(protocol="TCP")
    assert s.compare(s.clone()).names == ("A", "B")
    with pytest.raises(ValueError):
        v.compare(action="deny", names=("only one",))


@pytest.mark.parametrize("bad", [
    lambda v: v.compare(action="deny", protocol="TCP"),
    lambda v: v.compare(123),
    lambda v: v.compare(action="deny", metric="nope"),
    lambda v: v.compare(v, action="deny"),
    lambda v: v.compare(),
    lambda v: v.compare("action", "deny", "allow", protocol="TCP"),
])
def test_bad_forms_raise(v, bad):
    with pytest.raises((TypeError, ValueError)):
        bad(v)


# --------------------------------------------------------------------------- alignment & metrics


def test_metric_math_on_a_hand_built_case(tiny):
    c = p2h.fit(tiny, rows=["x"], cols=[], agg="count").compare(g="a")
    ta, tb = c.sides()
    assert set(ta.index) == {"p", "q"}
    assert ta["count"].loc[["p", "q"]].tolist() == [3.0, 1.0] and tb["count"].loc[["p", "q"]].tolist() == [2.0, 6.0]
    got = {m: c.table(m)["count"].loc[["p", "q"]].round(4).tolist() for m in p2h.METRICS}
    assert got["delta"] == [1.0, -5.0]
    assert got["ratio"] == [1.5, 0.1667]
    assert got["pct_change"] == [50.0, -83.3333]
    assert got["share_a"] == [75.0, 25.0] and got["share_b"] == [25.0, 75.0]
    assert got["share_delta"] == [50.0, -50.0]
    assert got["lift"] == [3.0, 0.3333]
    assert got["a"] == [3.0, 1.0] and got["b"] == [2.0, 6.0]
    assert c.additive and c.metric == "lift"


def test_union_of_labels_fills_zero_for_counts_and_nan_for_means():
    df = pd.DataFrame({"g": ["a"] * 3 + ["b"] * 3, "x": ["only_a", "both", "both", "both", "only_b", "only_b"], "n": [1, 2, 3, 4, 5, 6]})
    c = p2h.fit(df, rows=["x"], cols=[], agg="count").compare(g="a")
    ta, tb = c.sides()
    assert set(ta.index) == {"only_a", "both", "only_b"} and list(ta.index) == list(tb.index)
    assert ta.loc["only_b", "count"] == 0 and tb.loc["only_a", "count"] == 0
    cm = p2h.fit(df, rows=["x"], cols=[], values="n", agg="mean").compare(g="a")
    ma, mb = cm.sides()
    assert np.isnan(ma.loc["only_b"].iloc[0]) and np.isnan(mb.loc["only_a"].iloc[0])
    assert cm.metric == "delta"


def test_new_and_gone_cells():
    df = pd.DataFrame({"g": ["a"] * 3 + ["b"] * 3, "x": ["only_a", "both", "both", "both", "only_b", "only_b"]})
    c = p2h.fit(df, rows=["x"], cols=[], agg="count").compare(g="a").with_metric("ratio")
    t = c.table()["count"]
    assert np.isposinf(t["only_a"]) and t["only_b"] == 0
    text = c.render()
    assert "new" in text and "×0" in text
    top = c.top()
    assert set(top["only_in"].dropna()) == {"g=a", "rest"}
    d = c.to_dict()
    json.dumps(d)  # inf must have been sanitised
    only_a = next(r for r in d["top"] if r["row"] == "only_a")
    assert only_a["ratio"] is None and only_a["only_in"] == "g=a"


def test_share_metrics_need_an_additive_measure(fw):
    vm = p2h.fit(fw, rows=["dst_port"], cols=["action"], values="bytes", agg="mean", max_rows=12)
    c = vm.compare(protocol="TCP")
    assert not c.additive and c.metric == "delta"
    for m in ("lift", "share_delta", "share_a", "share_b"):
        with pytest.raises(ValueError, match="additive"):
            c.with_metric(m)
    assert "lift" not in c.top().columns
    c.with_metric("ratio").html()


def test_top_is_ranked_by_a_shrunk_log_ratio_and_lists_exact_values():
    # p: x30 built on 31 rows; q: x6 built on 70,000 rows -> q must rank first, exact values listed.
    df = pd.DataFrame({"g": ["a"] * 60_030 + ["b"] * 10_001, "x": ["p"] * 30 + ["q"] * 60_000 + ["p"] * 1 + ["q"] * 10_000})
    c = p2h.fit(df, rows=["x"], cols=[], agg="count").compare(g="a").with_metric("ratio")
    top = c.top()
    assert list(top.columns) == ["row", "col", "g=a", "rest", "delta", "ratio", "lift", "p", "only_in"]
    assert top["row"].tolist() == ["q", "p"] and top["ratio"].tolist() == [6.0, 30.0]
    assert "×30" in c.render() and "×6" in c.render()  # displayed values stay exact
    df2 = pd.DataFrame({"g": ["a"] * 8 + ["b"] * 10_000, "x": ["p"] * 2 + ["q"] * 6 + ["p"] * 1 + ["q"] * 9_999})
    c2 = p2h.fit(df2, rows=["x"], cols=[], agg="count").compare(g="a").with_metric("lift")
    t2 = c2.top()
    assert t2.iloc[0]["row"] == "p"  # the only cell over-represented in A; q is under-represented
    assert t2["lift"].round(2).tolist() == c2.table()["count"].round(2).loc[t2["row"]].tolist()


def test_top_n_zero_and_empty_side(v):
    assert v.compare(action="deny").top(0).empty
    e = v.compare(action="nope")
    ta, tb = e.sides()
    assert ta.shape == tb.shape and (ta.to_numpy() == 0).all()
    assert e.top().empty and "(empty)" not in e.html() and len(e.a.data) == 0


# --------------------------------------------------------------------------- slicing / modes


def test_slice_both_sides_and_unslice_keeps_defining_filters(v, fw):
    c = v.compare(action="deny")
    s = c.slice(protocol="TCP")
    assert len(s.a.data) == ((fw["action"] == "deny") & (fw["protocol"] == "TCP")).sum()
    assert len(s.b.data) == ((fw["action"] != "deny") & (fw["protocol"] == "TCP")).sum()
    assert s.layout == c.layout and "slices: protocol=TCP" in s.title()
    u = s.unslice("protocol")
    assert u.sides()[0].equals(c.sides()[0]) and u.names == c.names
    assert s.unslice().names == c.names and len(s.unslice().a.data) == len(c.a.data)
    x = c.exclude(protocol="UDP")
    assert (x.a.data["protocol"] != "UDP").all() and (x.b.data["protocol"] != "UDP").all()
    w = c.where("bytes > 100")
    assert (w.a.data["bytes"] > 100).all()


def test_toggle_to_paired_histograms_and_back(v):
    c = v.compare(action="deny")
    h = c.toggle()
    assert h.is_hist and h.a.is_hist and h.b.is_hist and "(hist)" in h.title()
    ta, tb = h.sides()
    assert list(ta.index) == list(tb.index)
    svg = h.html()
    assert "<svg" in svg and "% of each side" in svg
    back = h.slice(protocol="TCP").toggle()
    assert not back.is_hist and back.layout == c.layout and "protocol=TCP" in back.title()
    raw = h.style(normalize=False).html()
    assert "% of each side" not in raw


def test_histogram_of_a_column_plans_bins_on_both_sides(v, fw):
    c = v.compare(action="deny").histogram("bytes")
    assert c.is_hist and c.layout.rows[0].column == "bytes"
    ta, tb = c.sides()
    assert list(ta.index) == list(tb.index) and ta.shape[1] == 1
    # the rest's bytes reach far higher than deny's: shared edges must cover both
    edges = c.layout.edges if hasattr(c.layout, "edges") else c.layout.rows[0].edges
    assert edges[-1] >= fw["bytes"].max()
    today, yesterday = v.slice(timestamp="2026-03-02"), v.slice(timestamp="2026-03-01")
    tv = today.compare(yesterday).histogram("bytes")
    assert list(tv.sides()[0].index) == list(tv.sides()[1].index)
    assert not tv.toggle().is_hist


def test_swap_with_metric_and_style(v):
    c = v.compare(action="deny")
    s = c.swap()
    assert s.names == ("rest", "action=deny")
    np.testing.assert_allclose(s.table("delta").to_numpy(), -c.table("delta").to_numpy())
    assert c.with_metric("pct_change").metric == "pct_change" and c.metric == "lift"
    g = c.style(theme="graphite")
    assert "graphite" in g.html() and g.a.display["theme"] == "graphite"


# --------------------------------------------------------------------------- rendering / export


def test_html_is_diverging_and_tooltips_carry_both_sides(v):
    c = v.compare(action="deny")
    html = c.html()
    assert "action=deny:" in html and "rest:" in html and "lift" in html
    assert "×" in html  # ratio formatting
    both = c.html(side_by_side=True)
    assert both.count("<table") >= 3 and "413 rows" in both or "rows" in both
    none = c.style(heat="none").html()
    assert "rgb(" not in none.split("</div>", 1)[1] or "background:#fff" in none
    assert c.with_metric("delta").html().count("+") > 0


def test_render_text_and_repr(v):
    c = v.compare(action="deny")
    text = c.render(top=3)
    assert text.startswith("compare ·") and "top 3 by |lift|" in text and "×" in text
    assert repr(c) == str(c) == c.render()
    assert "by |lift|" not in c.render(top=0)


def test_to_dict_and_llm_context_are_json_safe(v):
    c = v.compare(action="deny")
    d = c.to_dict(4)
    s = json.dumps(d)
    assert "Infinity" not in s and "NaN" not in s
    assert set(d) == {"description", "mode", "layout", "metric", "metric_meaning", "a", "b", "shape", "table", "top", "drivers"}
    assert d["a"]["rows"] + d["b"]["rows"] == len(v.data) and len(d["top"]) <= 4
    assert d["a"]["total"] is not None
    ctx = c.llm_context(top=2)
    json.dumps(ctx)
    assert set(ctx) == {"description", "metadata", "table"}
    assert ctx["table"].startswith("|") and "Biggest movers" in ctx["description"]
    assert ctx["metadata"]["metric"] == "lift" and len(ctx["metadata"]["top"]) == 2
    assert json.loads(c.to_json())["metric"] == "lift"


def test_top_level_functions_route_split_kwargs(fw):
    c = p2h.compare(fw, action="deny", max_rows=10, max_cols=4)
    assert isinstance(c, Comparison) and c.a.options.max_rows == 10
    assert p2h.compare(fw, "action", "deny", "allow", rows=["dst_port"], metric="delta").metric == "delta"
    f = p2h.facet(fw, "protocol", max_rows=10)
    assert isinstance(f, Facets) and f.column == "protocol"


def test_paged_source_compare_matches_in_memory(fw, tmp_path):
    path = tmp_path / "fw.csv"
    fw.to_csv(path, index=False)
    paged = p2h.fit(str(path), rows=["dst_port"], cols=["action"], agg="count", mode="paged", page_rows=700)
    assert paged.paged is not None
    mem = p2h.fit(fw, rows=["dst_port"], cols=["action"], agg="count")
    cp, cm = paged.compare(protocol="TCP"), mem.compare(protocol="TCP")
    ta, tb = cp.sides()
    ma, mb = cm.sides()
    assert ta.reindex_like(ma).fillna(0).equals(ma) and tb.reindex_like(mb).fillna(0).equals(mb)


# --------------------------------------------------------------------------- facets


def test_facets_share_layout_labels_and_scale(v, fw):
    f = v.facet("action")
    assert isinstance(f, Facets) and f.labels == ["allow", "deny", "drop"] and f.hidden == 0
    ts = f.tables()
    assert all(list(t.index) == list(ts[0].index) and list(t.columns) == list(ts[0].columns) for t in ts)
    assert all(view.layout == f.layout for view in f)
    assert sum(len(view.data) for view in f) == len(v.data)
    assert f["deny"].slices[-1] == "action=deny" and f[1] is f["deny"]
    html = f.html()
    assert html.count("action = ") == 3 and "not shown" not in html
    small = v.facet("dst_port", 2)
    assert len(small) == 2 and small.hidden == fw["dst_port"].nunique() - 2 and "more value" in small.html()
    assert "more value" in small.render()


def test_facets_explicit_levels_and_axis_column(v):
    f = v.facet("action", levels=["deny", "nope"])
    assert f.labels == ["deny", "nope"] and len(f["nope"].data) == 0 and f.tables()[1].shape == f.tables()[0].shape
    on_axis = v.facet("severity", 3)
    assert "severity" not in {d.column for d in on_axis.layout.dims} and on_axis.layout.cols
    with pytest.raises(KeyError):
        v.facet("nope")


def test_facets_compare_slice_toggle(v, fw):
    f = v.facet("action")
    c = f.compare("deny", "allow")
    assert c.names == ("deny", "allow") and c.layout == f.layout
    r = f.compare("drop")
    assert r.names == ("action=drop", "rest") and len(r.b.data) == (fw["action"] != "drop").sum()
    assert f.compare(1, 0).names == ("deny", "allow")
    s = f.slice(protocol="TCP")
    assert all((view.data["protocol"] == "TCP").all() for view in s) and "protocol=TCP" in s.title()
    h = f.toggle()
    assert h.is_hist and "<svg" in h.html() and not h.toggle().is_hist
    hb = f.histogram("bytes")
    assert all(view.layout.rows[0].column == "bytes" for view in hb)
    x = f.exclude(protocol="UDP")
    assert all((view.data["protocol"] != "UDP").all() for view in x)
    d = f.to_dict()
    json.dumps(d)
    assert [x["label"] for x in d["facets"]] == f.labels
    with pytest.raises(KeyError):
        f["nope"]


# --------------------------------------------------------------------------- adversarial


@pytest.mark.parametrize("df_fn, split", [
    (lambda: pd.DataFrame({"a": [[1, 2]] * 50 + [[3, 4]] * 50, "b": range(100), "g": ["x", "y"] * 50}), {"g": "x"}),
    (lambda: pd.DataFrame({"a": [None] * 100, "b": range(100), "g": ["x", "y"] * 50}), {"a": None}),
    (lambda: pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}), {"b": "x"}),
    (lambda: pd.DataFrame({"a": pd.array([1, 2, None, 4] * 25, dtype="Int64"), "b": range(100)}), {"a": None}),
    (lambda: pd.DataFrame({"a": [np.inf, -np.inf, 1.0, 2.0] * 25, "b": range(100)}), {"a": 1.0}),
    (lambda: pd.DataFrame({"a": [f"id{i}" for i in range(200)], "b": range(200)}), {"b": "> 100"}),
    (lambda: pd.DataFrame({"a": [1] * 100, "b": [2] * 100}), {"a": 1}),
    (lambda: pd.DataFrame({"a": [True, False] * 50, "b": range(100)}), {"a": True}),
    (lambda: pd.DataFrame({"t": pd.date_range("2026-01-01", periods=100, freq="h", tz="UTC"), "b": range(100)}), {"t": "2026-01-02"}),
    (lambda: pd.DataFrame({"a": ["x"] * 100, "b": range(100), "n": np.random.default_rng(0).normal(size=100)}), {"n": "> 0"}),
])
def test_compare_and_facet_do_not_crash_on_adversarial_input(df_fn, split):
    v = p2h.fit(df_fn())
    c = v.compare(**split)
    c.html()
    c.html(side_by_side=True)
    c.render()
    c.top()
    json.dumps(c.to_dict())
    c.toggle().html()
    for m in p2h.METRICS:
        try:
            c.with_metric(m).html()
        except ValueError as e:
            assert "additive" in str(e)
    (col,) = split
    f = v.facet(col, 3)
    f.html()
    f.render()
    json.dumps(f.to_dict())
    f.toggle().html()


# --------------------------------------------------------------------------- significance and drivers


def test_top_has_p_for_count_measures_only(fw):
    c = p2h.compare(fw, action="deny", rows=["dst_port"], cols=["protocol"], agg="count")
    t = c.top(5)
    assert "p" in t.columns and list(t.columns)[-2:] == ["p", "only_in"]
    assert ((t["p"] >= 0) & (t["p"] <= 1)).all()
    assert t["p"].min() < 0.05 / c.shape[0] / c.shape[1]  # the top mover survives Bonferroni
    assert c.is_count
    s = p2h.compare(fw, action="deny", rows=["dst_port"], cols=["protocol"], values="bytes", agg="sum")
    assert "p" not in s.top(5).columns and not s.is_count


def test_gtest_matches_hand_computation():
    from pivot2hist._compare import _gtest_p

    a = np.array([[30.0, 70.0]])
    b = np.array([[10.0, 90.0]])
    p = _gtest_p(a, b)
    # 2x2 for cell (0,0): [[30, 70], [10, 90]] -> G = 2 * sum(O ln(O/E))
    obs = np.array([30, 70, 10, 90], dtype=float)
    exp = np.array([100 * 40 / 200, 100 * 160 / 200, 100 * 40 / 200, 100 * 160 / 200])
    g = 2 * np.sum(obs * np.log(obs / exp))
    assert p[0, 0] == pytest.approx(math.erfc(math.sqrt(g / 2)), rel=1e-9)
    assert p[0, 0] < 0.001 and p[0, 1] == pytest.approx(p[0, 0])  # the complement cell is the same test
    # identical shares: p = 1; a cell empty on both sides: NaN; an empty side: all NaN
    assert _gtest_p(np.array([[5.0, 5.0]]), np.array([[50.0, 50.0]]))[0, 0] == pytest.approx(1.0)
    assert np.isnan(_gtest_p(np.array([[0.0, 5.0]]), np.array([[0.0, 5.0]]))[0, 0])
    assert np.isnan(_gtest_p(np.array([[1.0, 2.0]]), np.array([[0.0, 0.0]]))).all()


def test_drivers_name_columns_off_the_table(fw):
    c = p2h.compare(fw, action="deny", rows=["dst_port"], cols=["protocol"], agg="count")
    d = c.drivers()
    assert list(d.columns) == ["column", "kind", "value", "share_a", "share_b", "lift", "median_a", "median_b", "ratio", "score", "text"]
    assert not d.empty and len(d) <= 8
    assert not (set(d["column"]) & {"dst_port", "protocol", "action"})  # axes and the split column are excluded
    assert (d["score"].diff().dropna() <= 0).all()  # best first
    assert "bytes" in set(d["column"])  # denies are small transfers in the sample
    row = d[d["column"] == "bytes"].iloc[0]
    assert row["kind"] == "numeric" and row["median_a"] < row["median_b"] and "action=deny" in row["text"] and "rest" in row["text"]
    labels = d[d["kind"] == "label"]
    assert (labels["share_a"].notna() & labels["share_b"].notna() & labels["lift"].notna()).all()
    assert c.drivers(2).equals(d.head(2)) or len(c.drivers(2)) == 2


def test_drivers_are_symmetric(fw):
    # a value that is common on side B but rare on side A is a driver too
    c = p2h.compare(fw, action="deny", rows=["dst_port"], cols=["protocol"], agg="count")
    d = c.drivers(20)
    assert (d["lift"].dropna() < 1).any() and (d["lift"].dropna() > 1).any()


def test_drivers_skip_query_columns(fw):
    v = p2h.fit(fw, rows=["dst_port"], cols=["action"], agg="count")
    c = v.compare("bytes > 5000 and duration < 2")
    assert c._split_columns() == ["bytes", "duration"]
    assert not (set(c.drivers()["column"]) & {"bytes", "duration"})


def test_drivers_of_a_two_view_comparison(fw):
    v = p2h.fit(fw, rows=["dst_port"], cols=["action"], agg="count")
    c = v.slice(country="US").compare(v.slice(country="CN"))
    assert c._split_columns() == ["country"]
    assert "country" not in set(c.drivers()["column"])
    assert "p" in c.top(3).columns


def test_to_dict_and_llm_context_carry_p_and_drivers(fw):
    c = p2h.compare(fw, action="deny", rows=["dst_port"], cols=["protocol"], agg="count")
    d = c.to_dict(3)
    assert "drivers" in d and d["drivers"] and {"column", "kind", "score", "text"} <= set(d["drivers"][0])
    assert all("p" in r for r in d["top"])
    json.dumps(d)
    ctx = c.llm_context(top=3)
    assert "p=" in ctx["description"] and "what else differs" in ctx["description"]
    assert ctx["metadata"]["drivers"] and "p" in ctx["metadata"]["top"][0]
    assert "what else differs" in str(c.prompt("q?"))


def test_agent_compare_returns_drivers(fw):
    r = agent.compare(fw, split={"column": "action", "eq": "deny"}, rows=["dst_port"], cols=["protocol"], agg="count", n=3)
    assert r["drivers"] and "text" in r["drivers"][0] and all("p" in t for t in r["top"])
    json.dumps(r)
