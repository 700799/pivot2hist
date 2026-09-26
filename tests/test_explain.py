import json

import numpy as np
import pandas as pd
import pytest

import pivot2hist as p2h
from pivot2hist import Explanation
from pivot2hist._html import expected_independence


@pytest.fixture(scope="module")
def v(fw):
    return p2h.fit(fw, max_rows=12, max_cols=5)  # sum(bytes) by dst_port (top 11) x severity


# --------------------------------------------------------------------------- cell / rows


def test_cell_and_rows_by_label(v, fw):
    c = v.cell("22", "3")
    assert c.layout == v.layout and c.pivot().shape == (1, 1) and c.slices[-2:] == ["dst_port=22", "severity=3"]
    want = ((fw["dst_port"] == 22) & (fw["severity"] == 3)).sum()
    assert len(c.data) == want == len(v.rows("22", "3")) == len(v.rows(dst_port=22, severity=3)) == len(v.rows(dst_port="22", severity="3"))
    assert len(v.rows("22")) == (fw["dst_port"] == 22).sum()            # whole row
    assert len(v.rows(None, "3")) == (fw["severity"] == 3).sum()         # whole column
    assert len(v.rows("22", n=5)) == 5
    assert len(v.rows("22", protocol="TCP")) == ((fw["dst_port"] == 22) & (fw["protocol"] == "TCP")).sum()  # non-axis: slice grammar


def test_rows_for_other_null_bins_time_nested_and_rollup(fw):
    v = p2h.fit(fw, max_rows=12, max_cols=5)
    kept = set(v.pivot().index) - {"(other)"}
    assert len(v.rows(dst_port="(other)")) == (~fw["dst_port"].isin(kept)).sum()
    h = v.histogram("bytes")
    lab = str(h.bins().index[3])
    assert len(h.rows(bytes=lab)) == int(h.bins().iloc[3, 0]) > 0
    t = v.histogram("timestamp")
    tl = str(t.bins().index[0])
    assert len(t.rows(timestamp=tl)) == int(t.bins().iloc[0, 0]) > 0
    n = p2h.fit(fw, rows=["protocol", "action"], cols=["severity"], agg="count", max_rows=20)
    want = ((fw["protocol"] == "TCP") & (fw["action"] == "deny")).sum()
    assert len(n.rows(("TCP", "deny"))) == len(n.rows(protocol="TCP", action="deny")) == want
    assert len(n.rows(("TCP", "deny"), "3")) == (fw["severity"][(fw["protocol"] == "TCP") & (fw["action"] == "deny")] == 3).sum()
    r = p2h.fit(fw, rows=["src_ip"], cols=["action"], agg="count").coarser("src_ip")
    lab = str(r.pivot().index[0])
    assert lab.endswith("/24") and len(r.rows(src_ip=lab)) == int(r.pivot().loc[lab].sum())
    withnull = fw.copy()
    withnull.loc[withnull.index[:7], "protocol"] = None
    vn = p2h.fit(withnull, rows=["protocol"], cols=["action"], agg="count")
    assert len(vn.rows(protocol="(null)")) == 7


def test_cell_errors(v):
    with pytest.raises(TypeError):
        v.rows()
    with pytest.raises(KeyError, match="no cell matches"):
        v.explain("nope", "3")
    with pytest.raises(ValueError, match="label"):
        v.cell(("a", "b", "c"))
    with pytest.raises(KeyError, match="unknown column"):
        v.rows(nope=1)
    assert len(v.rows("nope")) == 0  # a label no row has: empty, not an error


# --------------------------------------------------------------------------- explain


def test_explain_facts_match_the_pivot(v):
    t = v.pivot()
    e = v.explain("22", "3")
    assert isinstance(e, Explanation) and isinstance(e, dict)
    assert e["cell"] == {"dst_port": "22", "severity": "3"} and e["single_cell"]
    assert e["observed"] == pytest.approx(float(t.loc[22, 3]))
    vals = t.to_numpy(dtype=float)
    i, j = list(t.index).index(22), list(t.columns).index(3)
    assert e["expected"] == pytest.approx(float(expected_independence(vals)[i, j]))
    assert e["ratio_to_expected"] == pytest.approx(e["observed"] / e["expected"])
    assert e["direction"] in ("over", "under", "as expected")
    assert e["share_of_row"] == pytest.approx(e["observed"] / vals[i].sum())
    assert e["share_of_col"] == pytest.approx(e["observed"] / vals[:, j].sum())
    assert e["share_of_total"] == pytest.approx(e["observed"] / vals.sum())
    assert e["rank"] == int((vals > e["observed"]).sum()) + 1 and e["n_cells"] == vals.size
    assert e["n_rows"] == len(v.rows("22", "3")) and e["rows_total"] == len(v.data)
    assert len(e["rows"]) == 10 and set(e["rows"][0]) == set(v.data.columns)
    assert e["text"].startswith("dst_port=22 × severity=3: sum(bytes) = ")
    assert str(e) == e["text"] and repr(e) == e["text"]
    json.dumps(e)
    html = e._repr_html_()
    assert "expected under independence" in html and "what sets these rows apart" in html and "<table" in html


def test_explain_whole_row_skips_trivial_facts(v):
    e = v.explain("445")
    assert not e["single_cell"] and "expected" not in e and "share_of_row" not in e and "rank" not in e
    assert e["share_of_col"] is not None and e["share_of_total"] is not None
    assert e["n_rows"] == len(v.rows("445"))
    e1 = p2h.fit(p2h.sample.firewall_logs(500), rows=["action"], cols=[], agg="count").explain("deny")
    assert "share_of_row" not in e1 and e1["share_of_col"] is not None and e1["rank"] is not None


def test_explain_non_additive_and_hist(fw):
    vm = p2h.fit(fw, rows=["dst_port"], cols=["action"], values="bytes", agg="mean", max_rows=8)
    e = vm.explain("22", "deny")
    assert e["observed"] == pytest.approx(float(vm.pivot().loc[22, "deny"]))
    assert "expected" not in e and "share_of_row" not in e and e["rank"] is not None
    h = p2h.fit(fw).histogram("bytes")
    lab = str(h.bins().index[2])
    eh = h.explain(lab)
    assert eh["mode"] == "hist" and eh["cell"] == {"bytes": lab} and eh["observed"] == float(h.bins().iloc[2, 0])
    assert eh["n_rows"] == eh["observed"]


def test_distinguishing_finds_the_planted_signal():
    rng = np.random.default_rng(0)
    n = 2000
    df = pd.DataFrame({
        "r": rng.choice(["a", "b", "c"], n), "c": rng.choice(["x", "y"], n),
        "tag": rng.choice(["p", "q", "s"], n), "num": rng.normal(100, 5, n), "noise": rng.normal(0, 1, n),
    })
    cell = (df["r"] == "a") & (df["c"] == "x")
    df.loc[cell, "tag"] = "p"                # planted label
    df.loc[cell, "num"] = 400 + rng.normal(0, 5, cell.sum())  # planted numeric shift
    e = p2h.fit(df, rows=["r"], cols=["c"], agg="count").explain("a", "x")
    feats = e["distinguishing"]
    assert {f["column"] for f in feats[:2]} == {"tag", "num"}
    tag = next(f for f in feats if f["column"] == "tag")
    assert tag["value"] == "p" and tag["share_in_cell"] == 1.0 and tag["lift"] > 2
    num = next(f for f in feats if f["column"] == "num")
    assert num["median_in_cell"] > 350 and num["ratio"] > 3
    assert "noise" not in {f["column"] for f in feats}
    assert "`tag` is 'p' for 100% of these rows" in e["text"]


def test_explain_everything_has_nothing_to_contrast(fw):
    v = p2h.fit(fw, rows=["protocol"], cols=[], agg="count").slice(protocol="TCP", refit=False)
    e = v.explain("TCP")
    assert e["n_rows"] == e["rows_total"] and e["distinguishing"] == [] and "nothing to contrast" in e["text"]


def test_explain_k_and_n_rows(v):
    e = v.explain("22", "3", n_rows=3, k=2)
    assert len(e["rows"]) == 3 and len(e["distinguishing"]) <= 2
    assert v.explain("22", "3", n_rows=0)["rows"] == []


def test_rows_and_explain_on_paged_source(fw, tmp_path):
    path = tmp_path / "fw.csv"
    fw.to_csv(path, index=False)
    paged = p2h.fit(str(path), rows=["dst_port"], cols=["action"], agg="count", mode="paged", page_rows=700)
    mem = p2h.fit(fw, rows=["dst_port"], cols=["action"], agg="count")
    assert len(paged.rows("22", "deny")) == len(mem.rows("22", "deny")) > 0
    assert len(paged.rows("22", "deny", n=5)) == 5
    e = paged.explain("22", "deny")
    assert e["sampled"] and e["observed"] == mem.explain("22", "deny")["observed"]


def test_facets_and_comparison_sides_explain(v):
    f = v.facet("action")
    e = f["deny"].explain("22", "3")
    assert e["rows_total"] == len(f["deny"].data) and all(r["action"] == "deny" for r in e["rows"])
    c = v.compare(action="deny")
    assert c.a.explain("22", "3")["n_rows"] == len(c.a.rows("22", "3"))


@pytest.mark.parametrize("df_fn", [
    lambda: pd.DataFrame({"a": [[1, 2]] * 50 + [[3, 4]] * 50, "b": range(100)}),
    lambda: pd.DataFrame({"a": [None] * 100, "b": range(100)}),
    lambda: pd.DataFrame({"a": [1], "b": ["x"]}),
    lambda: pd.DataFrame({"a": pd.array([1, 2, None, 4] * 25, dtype="Int64"), "b": range(100)}),
    lambda: pd.DataFrame({"a": [np.inf, -np.inf, 1.0, 2.0] * 25, "b": range(100)}),
    lambda: pd.DataFrame({"a": [f"id{i}" for i in range(200)], "b": range(200)}),
    lambda: pd.DataFrame({"a": [1] * 100, "b": [2] * 100}),
    lambda: pd.DataFrame({"a": [True, False] * 50, "b": range(100)}),
    lambda: pd.DataFrame({"t": pd.date_range("2026-01-01", periods=100, freq="h", tz="UTC"), "b": range(100)}),
])
def test_explain_does_not_crash_on_adversarial_input(df_fn):
    v = p2h.fit(df_fn())
    t = v.table()
    first = t.index[0]
    row = tuple(str(x) for x in first) if isinstance(first, tuple) else str(first)
    e = v.explain(row)
    json.dumps(e)
    e._repr_html_()
    v.rows(row)
    if v.layout.cols:
        col = t.columns[0]
        v.explain(row, tuple(str(x) for x in col) if isinstance(col, tuple) else str(col))
