import json

import numpy as np
import pandas as pd
import pytest

import pivot2hist as p2h


def test_clone_shares_source_but_is_independent(fw):
    base = p2h.fit(fw, max_rows=12, max_cols=6)
    a = base.clone().slice(action="deny")
    b = base.clone().toggle()
    assert a is not base and b is not base
    assert a.source is base.source and b.source is base.source  # no data copy
    assert base.filters == () and a.filters != ()
    assert base.mode == "pivot" and b.mode == "hist"
    assert base._cache is not a._cache and a._cache is not b._cache
    assert a.pivot().shape[0] <= base.pivot().shape[0]


def test_toggle_round_trip(fw):
    v = p2h.fit(fw, max_rows=12, max_cols=6)
    h = v.toggle()
    assert h.mode == "hist" and v.mode == "pivot"
    assert h.layout == v.layout
    back = h.toggle()
    assert back.mode == "pivot" and back.layout == v.layout
    assert back.pivot().equals(v.pivot())
    assert v.as_hist().mode == "hist" and v.as_pivot() is v
    assert h.as_hist() is h


def test_hist_bins_match_pivot(fw):
    v = p2h.fit(fw, max_rows=12, max_cols=6, agg="count")
    h = v.toggle()
    b = h.bins()
    assert b.to_numpy().sum() == len(fw)
    assert b.shape[1] == v.pivot().shape[1]


def test_histogram_on_numeric(fw):
    h = p2h.fit(fw).histogram("bytes")
    assert h.mode == "hist"
    assert h.layout.rows[0].kind == "binned" and h.layout.measure == "count"
    assert h.bins()["count"].sum() == len(fw)
    assert len(h.bins()) <= 30
    h5 = p2h.fit(fw).histogram("bytes", bins=5)
    assert len(h5.bins()) <= 5
    lin = p2h.fit(fw).histogram("bytes", bins=8, scale="linear")
    edges = np.array(lin.layout.rows[0].edges)
    assert np.allclose(np.diff(edges), np.diff(edges)[0])


def test_histogram_back_to_parent(fw):
    v = p2h.fit(fw)
    h = v.histogram("duration")
    assert h.toggle().layout == v.layout
    # toggling twice from a hist made with `on` stays on the parent pivot
    assert h.toggle().toggle().layout == v.layout


def test_histogram_by_and_values(fw):
    h = p2h.fit(fw).histogram("bytes", by="action", values="bytes", agg="sum")
    b = h.bins()
    assert list(b.columns) == ["allow", "deny", "drop"]
    assert b.to_numpy().sum() == fw["bytes"].sum()
    h2 = p2h.fit(fw).histogram("dst_port", by=["protocol", "action"])
    assert isinstance(h2.bins().columns, pd.MultiIndex)


def test_histogram_on_time_and_categorical(fw):
    ht = p2h.fit(fw).histogram("timestamp")
    assert ht.layout.rows[0].kind == "time"
    assert ht.bins()["count"].sum() == len(fw)
    hc = p2h.fit(fw).histogram("country")
    assert hc.layout.rows[0].kind == "categorical"
    assert hc.bins()["count"].sum() == len(fw)


def test_top_level_histogram_helper(fw):
    h = p2h.histogram(fw, "bytes", bins=6)
    assert h.mode == "hist" and len(h.bins()) <= 6
    h2 = p2h.histogram(fw)
    assert h2.mode == "hist" and h2.toggle().mode == "pivot"


def test_slice_forms(fw):
    v = p2h.fit(fw)
    assert len(v.slice(action="deny").data) == (fw["action"] == "deny").sum()
    assert len(v.slice(dst_port=[22, 443]).data) == fw["dst_port"].isin([22, 443]).sum()
    assert len(v.slice(bytes=(1000, 5000)).data) == ((fw["bytes"] >= 1000) & (fw["bytes"] < 5000)).sum()
    assert len(v.slice(bytes=slice(1000, None)).data) == (fw["bytes"] >= 1000).sum()
    assert len(v.slice(bytes=">= 1000").data) == (fw["bytes"] >= 1000).sum()
    assert len(v.slice(bytes="!=0").data) == (fw["bytes"] != 0).sum()
    assert len(v.slice(src_ip="~^10\\.0\\.0\\.1$").data) == (fw["src_ip"] == "10.0.0.1").sum()
    assert len(v.slice(bytes=lambda s: s > 100).data) == (fw["bytes"] > 100).sum()
    assert len(v.slice("bytes > 100 and action == 'deny'").data) == ((fw["bytes"] > 100) & (fw["action"] == "deny")).sum()
    assert len(v.slice("action", "drop").data) == (fw["action"] == "drop").sum()
    assert len(v.slice({"action": "drop"}).data) == (fw["action"] == "drop").sum()
    day = v.slice(timestamp="2026-03-02")
    assert len(day.data) == (fw["timestamp"].dt.normalize() == pd.Timestamp("2026-03-02")).sum()
    month = v.slice(timestamp="2026-03")
    assert len(month.data) == len(fw)
    rng = v.slice(timestamp=("2026-03-02", "2026-03-04"))
    assert len(rng.data) == ((fw["timestamp"] >= "2026-03-02") & (fw["timestamp"] < "2026-03-04")).sum()
    assert len(v.slice(dst_port="443").data) == (fw["dst_port"] == 443).sum()  # coerced


def test_slice_chain_and_unslice(fw):
    v = p2h.fit(fw)
    s = v.slice(action="deny").slice(dst_port=[22, 3389])
    assert len(s.filters) == 2
    assert s.slices == ["action=deny", "dst_port∈{22,3389}"]
    assert "slices:" in s.title()
    u = s.unslice("action")
    assert [f.column for f in u.filters] == ["dst_port"]
    assert u.unslice().filters == ()
    assert len(v.unslice().data) == len(fw)
    with pytest.raises(KeyError):
        v.slice(nope=1)


def test_slice_refits_when_dim_collapses(fw):
    v = p2h.fit(fw, rows=["src_ip"], cols=["action"], agg="count")
    s = v.slice(action="deny")
    assert "action" not in {d.column for d in s.layout.dims}  # refit dropped the constant column
    assert s.layout.rows[0].column == "src_ip"  # the other fixed axis survived
    kept = v.slice(action="deny", refit=False)
    assert kept.layout == v.layout
    assert kept.pivot().shape[1] == 1


def test_slice_persists_through_toggle(fw):
    v = p2h.fit(fw, max_rows=10, max_cols=4).slice(action="deny")
    h = v.toggle().slice(protocol="TCP")
    back = h.toggle()
    assert back.mode == "pivot" and len(back.filters) == 2
    assert len(back.data) == ((fw["action"] == "deny") & (fw["protocol"] == "TCP")).sum()


def test_empty_slice_renders(fw):
    v = p2h.fit(fw).slice(action="nope")
    assert len(v.data) == 0
    assert "(empty)" in v.render()
    assert "(empty)" in v.toggle().render()


def test_slicers(fw):
    v = p2h.fit(fw)
    s = v.slicers(3)
    assert all(len(vals) <= 3 for vals in s.values())
    assert "action" in s and s["action"][0][1] >= s["action"][-1][1]


def test_refit_relayout_layers_fit_to(fw):
    v = p2h.fit(fw)
    assert v.layers(1).layout.rows and len(v.layers(1).layout.rows) == 1
    small = v.fit_to(6, 3)
    assert small.pivot().shape[0] <= 6 and small.pivot().shape[1] <= 3
    r = v.relayout(rows=["country"], agg="count")
    assert r.layout.rows[0].column == "country" and r.layout.measure == "count"
    r2 = v.relayout(rows=["country"], cols=[], agg="count")
    assert r2.layout.cols == ()


def test_render_modes(fw):
    v = p2h.fit(fw, max_rows=10, max_cols=4)
    txt = v.render(width=100)
    assert txt.startswith("pivot ·") and "\n" in txt
    ascii_hist = v.toggle().render(width=100, ascii_only=True)
    assert "#" in ascii_hist and "█" not in ascii_hist
    uni = v.toggle().render(width=100)
    assert "█" in uni or "▏" in uni
    assert str(v) == repr(v) == v.render()
    html = v._repr_html_()
    assert "<table" in html and "<svg" in v.toggle()._repr_html_()
    assert "... " in v.render(max_rows=3)


def test_render_nested_rows(auth):
    v = p2h.fit(auth, rows=["event", "method"], cols=["mfa"], agg="count").toggle()
    txt = v.render(width=100)
    assert "event=login_failure" in txt or "event=login_success" in txt
    assert "  password" in txt


def test_to_dict_json_csv(fw):
    v = p2h.fit(fw, max_rows=8, max_cols=3).slice(action="allow")
    d = v.to_dict()
    assert d["mode"] == "pivot" and d["slices"] == ["action=allow"]
    assert d["layout"]["measure"].startswith("sum(")
    assert len(d["table"]) == d["shape"][0]
    json.loads(v.to_json())
    assert v.toggle().to_dict()["mode"] == "hist"
    assert v.to_csv().count("\n") >= d["shape"][0]


def test_pandas_string_dtype_and_bool_measure():
    df = pd.DataFrame(
        {
            "user": pd.Series(["a", "b", "a", "c"] * 25, dtype="string"),
            "ok": [True, False, True, True] * 25,
            "lat": np.arange(100) * 1.0,
        }
    )
    v = p2h.fit(df, rows=["user"], values="ok", agg="sum")
    assert v.pivot().to_numpy().sum() == 75
    v2 = p2h.fit(df, rows=["user"], values="lat", agg="mean")
    assert v2.pivot().shape[0] == 3


def test_timedelta_measure_and_bins():
    df = pd.DataFrame({"who": list("abab") * 10, "dur": pd.to_timedelta(np.arange(40), unit="s")})
    v = p2h.fit(df, rows=["who"], values="dur", agg="sum")
    assert v.pivot().to_numpy().sum() == np.arange(40).sum()
    h = v.histogram("dur", bins=4)
    assert h.bins()["count"].sum() == 40


def test_source_is_untouched(fw):
    before = fw.copy()
    v = p2h.fit(fw)
    v.slice(action="deny").toggle().histogram("bytes").pivot()
    pd.testing.assert_frame_equal(fw, before)
    assert v.source is fw
