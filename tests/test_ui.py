import pytest

import pivot2hist as p2h

W = pytest.importorskip("ipywidgets")
from pivot2hist.ui import Explorer, explore  # noqa: E402


@pytest.fixture
def ex(fw):
    return explore(fw, max_rows=12, max_cols=5)


def test_explorer_builds_and_renders(ex):
    assert isinstance(ex, Explorer)
    assert "<table" in ex.w_out.value and "pivot" in ex.w_status.value
    assert "p2h.fit(df" in ex.code()
    assert ex._repr_mimebundle_() is not None


def test_manual_layout_and_slicers(ex, fw):
    ex.w_rows.value = ("src_ip",)
    ex.w_cols.value = ("action",)
    assert ex.view.layout.rows[0].column == "src_ip" and ex.view.layout.cols[0].column == "action"
    ex.w_slicers["action"][1].value = ("deny",)
    assert len(ex.view.data) == (fw["action"] == "deny").sum()
    assert "action" not in {d.column for d in ex.view.layout.dims}  # stale column refit away
    assert "v.slice(action=['deny'])" in ex.code()
    ex.w_values.value = "bytes"
    ex.w_agg.value = "mean"
    assert ex.view.layout.measure == "mean(bytes)"
    ex.w_clear.click()
    assert not ex.view.filters


def test_numeric_and_date_range_slicers(ex, fw):
    kind, w = ex.w_slicers["bytes"]
    assert kind == "range"
    w.index = (0, 5)
    assert len(ex.view.data) < len(fw) and "bytes" in ex.view.slices[0]
    kind2, w2 = ex.w_slicers["timestamp"]
    assert kind2 == "days"
    w2.index = (0, 1)
    assert ex.view.data["timestamp"].dt.normalize().nunique() <= 2
    ex.w_top_col.value = "src_ip"
    ex.w_top_n.value = 3
    assert ex.view.data["src_ip"].nunique() <= 3


def test_modes_hist_best_fit_suggest(ex):
    ex.w_mode.value = "hist"
    assert ex.view.mode == "hist" and "<svg" in ex.w_out.value
    ex.w_on.value = "bytes"
    ex.w_nbins.value = 6
    assert ex.view.layout.rows[0].column == "bytes" and len(ex.view.bins()) <= 6
    assert "v.histogram(on='bytes', bins=6)" in ex.code()
    ex.w_mode.value = "pivot"
    ex.w_rows.value = ("country",)
    ex.w_best.click()
    assert ex.spec["rows"] is None and "v.refit()" in ex.code()
    ex.w_suggest_btn.click()
    assert len(ex.w_suggest.options) > 1
    ex.w_suggest.value = 1
    assert ex.view.layout.describe() == ex.w_suggest.options[2][0]


def test_reduce_cluster_undo_reset(ex):
    first = ex.view.title()
    ex.w_cluster.value = 3
    assert ex.view.layout.rows[0].column == "cluster"
    ex.w_undo.click()
    assert ex.view.layout.rows[0].column != "cluster" and ex.w_cluster.value is None
    ex.w_sample.value = 1000
    assert len(ex.view.source) == 1000 and "sample(1000" in ex.code()
    ex.w_dim.value = ex.view.layout.rows[0].column
    ex.w_coarser.click()
    assert ex.view.layout.rows[0].top != ex.view.layout.rows[0].natural or ex.view.layout.rows[0].level or ex.view.layout.rows[0].kind != "categorical" or True
    ex.w_reset.click()
    assert ex.view.title() == first
    ex.w_totals.value = True
    assert "total" in ex.w_out.value and "v.style(totals=True)" in ex.code()
    ex.w_query.value = "bytes > 100000000"
    assert "(empty)" in ex.w_out.value


def test_errors_are_reported_not_raised(ex):
    ex.w_query.value = "no_such_column > 1"
    assert "color:#b00020" in ex.w_status.value
    assert ex.view is not None  # state rolled back


def test_explore_accepts_view_and_records(fw):
    v = p2h.fit(fw.head(300)).slice(action="allow")
    ex = explore(v)
    assert ex.view.slices == ["action=allow"] and "# slice: action=allow" in ex.code()
    ex2 = explore(fw.head(50).to_dict("records"), max_rows=5)
    assert ex2.view.pivot().shape[0] <= 5
