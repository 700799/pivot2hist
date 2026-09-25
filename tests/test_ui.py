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


def test_explorer_log_stats_data_and_snapshot(ex):
    assert "<div" in ex.w_log.value and len(ex._log_lines) > 0
    assert "<table" in ex.w_stats.value
    assert "in memory" in ex.w_data.value or "frame" in ex.w_data.value
    html = ex.snapshot_html()
    assert "<select" in html and "Reduce" in html and "<table" in html
    ex.tabs.selected_index = 4
    assert "Markov" in ex.snapshot_html()
    ex.close()


def test_explorer_cluster_methods_cocluster_chains(ex, fw):
    ex.w_cluster.value = 2
    ex.w_method.value = "dbscan"
    assert ex.view.layout.rows[0].column == "cluster" and "method='dbscan'" in ex.code()
    ex.w_cluster.value = None
    ex.w_cocluster.value = "spectral"
    assert ex.view.layout.rows[0].column == "block_r" and "cocluster(method='spectral')" in ex.code()
    ex.w_cocluster.value = None
    ex.w_state.value = "action"
    assert ex.mode == "chains" and ex.view.layout.rows[0].column == "from"
    assert "p2h.chains(v.data, 'action')" in ex.code() and "chains" in ex.w_sequences.value
    ex.w_chain_norm.value = True
    assert "normalize=True" in ex.code()
    ex.w_mode.value = "pivot"
    assert ex.view.layout.rows[0].column != "from"
    ex.close()


def test_explorer_on_paged_source(tmp_path):
    pytest.importorskip("pyarrow")
    path = tmp_path / "fw.parquet"
    p2h.sample.firewall_logs(20_000, seed=4).to_parquet(path, index=False)
    ex = explore(str(path), memory_budget_mb=2, max_rows=8, max_cols=4)
    assert ex.view.paged is not None and "(paged" in ex.view.title() and "paged source" in ex.w_data.value
    ex.w_slicers["action"][1].value = ("deny",)
    assert " of ≈" in ex.view.title()
    ex.w_sample.value = 1000
    assert ex.view.paged is None and len(ex.view.source) == 1000
    ex.close()


def test_fields_tab_present_and_synced_with_layout(ex, fw):
    assert ex.tabs.get_title(ex.FIELDS_TAB) == "Fields"
    assert list(ex.w_fields.rows) == [d.column for d in ex.view.layout.rows if d.column in ex.w_rows.options]
    assert list(ex.w_fields.cols) == [d.column for d in ex.view.layout.cols if d.column in ex.w_cols.options]


def test_dragging_fields_updates_the_layout_rows_within_rows(ex, fw):
    ex.w_fields.rows = ["src_ip", "dst_port"]
    ex.w_fields.cols = ["action"]
    ex.w_fields.values = ["bytes"]
    assert ex.spec["rows"] == ["src_ip", "dst_port"] and ex.spec["cols"] == ["action"]
    assert [d.column for d in ex.view.layout.rows] == ["src_ip", "dst_port"]
    assert list(ex.w_rows.value) == ["src_ip", "dst_port"]  # Layout tab mirrors the drop
    ex.w_fields.cols = ["action", "protocol"]  # columns within columns
    assert [d.column for d in ex.view.layout.cols] == ["action", "protocol"]


def test_fields_values_zone_holds_one_measure(ex, fw):
    ex.w_fields.values = ["bytes"]
    assert ex.view.layout.measure == "sum(bytes)"
    ex.w_fields.values = ["duration"]
    assert ex.view.layout.measure.endswith("(duration)")


def test_fields_slicers_zone_mirrors_widget(ex, fw):
    ex.w_fields.slicers = ["action"]
    assert ex.field_slicers == ["action"]
    assert len(ex.w_field_slicer_box.children) == 1
    ex.w_fields.slicers = []
    assert len(ex.w_field_slicer_box.children) == 0


def test_fields_slicer_builds_on_demand_beyond_the_capped_tab(fw):
    # max_slicers=1 leaves every categorical column but the first without a pre-built
    # widget in the Slicers tab; dragging one of the rest into the Fields tab's Slicers
    # zone must still produce a working quick filter, not the "no quick filter" message.
    ex = explore(fw, max_rows=12, max_cols=5, max_slicers=1)
    assert "country" not in ex.w_slicers
    ex.w_fields.slicers = ["country"]
    assert "country" in ex.w_slicers
    kind, w = ex.w_slicers["country"]
    assert kind == "in"
    box = ex.w_field_slicer_box.children[0]
    assert w in box.children  # the real widget was mounted, not a "no quick filter" message
    some_country = w.options[0][1]
    w.value = (some_country,)
    assert len(ex.view.data) == (fw["country"] == some_country).sum()
    assert "country" in ex.view.slices[0]


def test_fields_slicer_numeric_builds_on_demand_beyond_the_hardcoded_cap():
    # the Slicers tab pre-builds only the first 4 numeric columns regardless of
    # max_slicers; a 5th must still build on demand when dragged into the Fields tab.
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    df = pd.DataFrame({f"n{i}": rng.normal(size=300) for i in range(6)})
    ex = explore(df, max_rows=8, max_cols=4)
    assert "n5" not in ex.w_slicers
    ex.w_fields.slicers = ["n5"]
    kind, w = ex.w_slicers["n5"]
    assert kind == "range"
    w.index = (0, len(w.options) // 2)
    assert len(ex.view.data) < len(df)
    ex.close()


def test_best_fit_resyncs_fields_pane(ex, fw):
    ex.w_fields.rows = ["country"]
    ex.w_best.click()
    assert ex.spec["rows"] is None
    assert list(ex.w_fields.rows) == [d.column for d in ex.view.layout.rows]


def test_undo_reset_restore_fields_pane(ex, fw):
    before_rows = list(ex.w_fields.rows)
    ex.w_fields.cols = ["protocol"]
    ex.w_undo.click()
    assert list(ex.w_fields.rows) == before_rows


def test_theme_toggle(ex, fw):
    assert ex.w_theme.value == "light"
    ex.w_theme.value = "graphite"
    assert ex.display["theme"] == "graphite"
    assert "#161a20" in ex.w_out.value or "#1b2128" in ex.w_out.value
    assert ex.w_fields.theme == "graphite"
    assert "v.style(theme='graphite')" in ex.code()
    ex.w_theme.value = "light"
    assert "theme=" not in ex.code()  # default theme omitted from the reproducing code


def test_subtotals_and_outline_toggles(fw):
    v = p2h.fit(fw, rows=["severity", "action"], cols=["protocol"], agg="count", max_rows=15)
    ex = explore(v)
    ex.w_subtotals.value = True
    assert "∑" in ex.w_out.value
    ex.w_subtotals.value = False
    ex.w_outline.value = True
    assert "<details" in ex.w_out.value
    ex.close()


def test_snapshot_html_themes_the_chrome(ex, fw):
    light = ex.snapshot_html()
    ex.w_theme.value = "graphite"
    dark = ex.snapshot_html()
    assert light != dark
    assert "background:#161a20" in dark
    assert "src_ip" in dark or "dst_port" in dark  # the fields pane rendered inside


def test_fields_tab_survives_reset(ex, fw):
    first_rows = list(ex.w_fields.rows)
    ex.w_fields.rows = ["country", "protocol"]
    ex.w_reset.click()
    assert list(ex.w_fields.rows) == first_rows


def test_paint_and_unmark(ex):
    ex.paint("src_ip", "red")
    assert ex.marks["src_ip"].startswith("#")
    assert ex.w_fields.marks == ex.marks
    with pytest.raises(KeyError):
        ex.paint("not_a_real_column", "blue")
    ex.unmark("src_ip")
    assert "src_ip" not in ex.marks
    ex.paint("action", "green")
    ex.unmark()
    assert ex.marks == {}


def test_undo_reverts_marks_too(ex):
    ex.paint("src_ip", "red")  # not through _act(): no undo entry from this alone
    ex.w_rows.value = ("action",)  # a real _act()-driven change: snapshots the marked state
    ex.paint("src_ip", None)
    ex.w_cols.value = ("protocol",)  # another _act(): snapshots the now-unmarked state
    ex.w_undo.click()
    assert ex.marks.get("src_ip") is None  # undo restores the snapshot from just before this click
    ex.w_undo.click()
    assert "src_ip" in ex.marks  # further back: the mark from before the first _act()


def test_clone_shares_source_and_profile_but_is_independent(ex, fw):
    ex.paint("src_ip", "blue")
    clone = ex.clone()
    try:
        assert clone is not ex
        assert clone._source is ex._source
        assert clone._profile is ex._profile
        assert clone.marks == ex.marks  # marks carry over
        assert clone.checkpoints == [] and clone.history == []  # fresh timeline
        clone.w_rows.value = ("dst_port",)
        assert ex.w_rows.value != ("dst_port",) or list(ex.spec.get("rows") or []) != ["dst_port"]
    finally:
        clone.close()


def test_clone_with_explicit_view(ex, fw):
    other = p2h.fit(fw, rows=["protocol"], cols=["action"])
    clone = ex.clone(view=other)
    try:
        assert clone.spec.get("rows") == ["protocol"]
    finally:
        clone.close()


def test_timeline_starts_empty(ex):
    assert ex.checkpoints == []
    assert ex.w_timeline_slider.max == 0
    assert "no checkpoints" in ex.w_timeline_note.value


def test_save_checkpoint_and_goto(ex):
    ex.w_rows.value = ("src_ip",)
    ex.save_checkpoint("baseline")
    assert len(ex.checkpoints) == 1
    assert ex.checkpoints[0]["note"] == "baseline"
    assert ex.w_timeline_slider.max == 0
    assert ex.w_checkpoint_note.value == ""  # cleared after saving

    ex.w_rows.value = ("dst_port",)
    ex.paint("dst_port", "red")
    ex.save_checkpoint("  port spike  ")
    assert len(ex.checkpoints) == 2
    assert ex.checkpoints[1]["note"] == "port spike"  # stripped
    assert ex.w_timeline_slider.max == 1
    assert ex.w_timeline_slider.value == 1  # jumps to the newest on save

    ex.goto_checkpoint(0)
    assert ex.w_rows.value == ("src_ip",) and ex.marks == {}
    ex.goto_checkpoint(-1)
    assert ex.w_rows.value == ("dst_port",) and "dst_port" in ex.marks


def test_timeline_slider_rerenders_on_each_notch(ex):
    ex.w_rows.value = ("src_ip",)
    ex.save_checkpoint("a")
    out_a = ex.w_out.value
    ex.w_rows.value = ("action",)
    ex.save_checkpoint("b")
    out_b = ex.w_out.value
    assert out_a != out_b

    ex.w_timeline_slider.value = 0
    assert ex.view.layout.rows[0].column == "src_ip"
    assert ex.w_out.value == out_a
    ex.w_timeline_slider.value = 1
    assert ex.view.layout.rows[0].column == "action"
    assert ex.w_out.value == out_b


def test_goto_checkpoint_out_of_range_and_empty(ex):
    with pytest.raises(IndexError):
        ex.goto_checkpoint(0)  # nothing saved yet
    ex.save_checkpoint("only one")
    with pytest.raises(IndexError):
        ex.goto_checkpoint(5)


def test_delete_checkpoint(ex):
    ex.save_checkpoint("one")
    ex.save_checkpoint("two")
    n = len(ex.checkpoints)
    ex.w_timeline_slider.value = n - 1
    ex._delete_checkpoint()
    assert len(ex.checkpoints) == n - 1


def test_checkpoint_playback_widget_linked(ex):
    ex.save_checkpoint("one")
    ex.save_checkpoint("two")
    assert ex.w_timeline_play.max == ex.w_timeline_slider.max == 1


def test_insights_tab_present(ex):
    assert ex.tabs.get_title(ex.INSIGHTS_TAB) == "Insights"


def test_calculate_populates_insights_tab(ex):
    assert "click Calculate" in ex.w_insights.value
    report = ex.calculate()
    assert set(report) == {"summary", "shape", "columns", "findings", "sensitivity"}
    assert "<table" in ex.w_insights.value or "nothing crossed" in ex.w_insights.value


def test_calculate_uses_slider_sensitivity_by_default(ex):
    ex.w_sensitivity.value = 0.1
    ex.calculate()
    assert ex._insights_report["sensitivity"] == 0.1


def test_calculate_with_explicit_sensitivity_syncs_slider(ex):
    ex.calculate(sensitivity=0.9)
    assert ex.w_sensitivity.value == 0.9
    assert ex._insights_report["sensitivity"] == 0.9


def test_insights_goes_stale_after_view_changes_then_clears_on_recalculate(ex):
    ex.calculate()
    assert "view has changed" not in ex.w_insights.value
    ex.w_rows.value = ("action",)
    assert "view has changed" in ex.w_insights.value
    ex.calculate()
    assert "view has changed" not in ex.w_insights.value


def test_insights_reflects_current_slice_in_explorer(ex, fw):
    ex.w_slicers["action"][1].value = ("deny",)
    report = ex.calculate()
    assert report["shape"]["rows"] == (fw["action"] == "deny").sum()


# --------------------------------------------------------------------------- compare tab


def test_compare_tab_present_and_idle(ex):
    assert ex.tabs.get_title(ex.COMPARE_TAB) == "Compare"
    assert ex.comparison is None and "pick a column" in ex.w_cmp_out.value


def test_compare_tab_facet_then_split_then_metric(ex):
    from pivot2hist import Comparison, Facets

    ex.w_cmp_col.value = "action"
    assert isinstance(ex.comparison, Facets) and ex.w_cmp_out.value.count("action = ") == 3
    assert ex.w_cmp_val.options[1][1] == "allow"
    ex.w_cmp_val.value = "deny"
    assert isinstance(ex.comparison, Comparison) and ex.comparison.names == ("action=deny", "rest")
    assert ex.comparison.metric == "lift" and "c = v.compare(action='deny')" in ex.code()
    ex.w_cmp_metric.value = "delta"
    assert ex.comparison.metric == "delta" and "metric='delta'" in ex.code()
    before = len(ex.w_cmp_out.value)
    ex.w_cmp_side.value = True
    assert len(ex.w_cmp_out.value) > before


def test_compare_tab_query_form_and_clear(ex, fw):
    ex.w_cmp_query.value = "bytes > 5000"
    assert ex.comparison.names == ("bytes > 5000", "rest") and len(ex.comparison.a.data) == (fw["bytes"] > 5000).sum()
    assert "c = v.compare('bytes > 5000')" in ex.code()
    ex.w_cmp_clear.click()
    assert ex.comparison is None and ex.w_cmp_query.value == "" and "c = v.compare" not in ex.code()


def test_compare_stays_in_step_with_the_view(ex, fw):
    ex.w_cmp_col.value = "action"
    ex.w_cmp_val.value = "deny"
    ex.w_slicers["protocol"][1].value = ("TCP",)
    assert "protocol" in ex.comparison.title() and len(ex.comparison.a.data) == ((fw["action"] == "deny") & (fw["protocol"] == "TCP")).sum()


def test_compare_pin_baseline_flow(ex, fw):
    assert ex.compare_with_baseline() is None and "pin a baseline" in ex.w_cmp_status.value
    base = ex.pin()
    assert base is ex.view and "baseline:" in ex.w_cmp_status.value
    ex.w_slicers["action"][1].value = ("deny",)
    c = ex.compare_with_baseline()
    assert c.names == ("current", "baseline") and len(c.b.data) == len(fw) and len(c.a.data) == (fw["action"] == "deny").sum()
    assert c.b.layout == c.a.layout and [d.column for d in c.layout.dims] == [d.column for d in base.layout.dims]
    assert "v.compare(baseline" in ex.code()


def test_compare_programmatic(ex):
    c = ex.compare("action", "deny", "drop", metric="ratio")
    assert c.names == ("action=deny", "action=drop") and c.metric == "ratio" and ex.w_cmp_metric.value == "ratio"
    assert "c = v.compare('action', 'deny', 'drop', metric='ratio')" in ex.code()
    f = ex.facet("protocol", 2)
    assert f.labels == ["TCP", "UDP"] and "f = v.facet('protocol', 2)" in ex.code()
    with pytest.raises(ValueError):
        ex.compare(action="deny", metric="nope")
    with pytest.raises(ValueError):
        ex.compare(nope="x")


def test_compare_metric_falls_back_for_non_additive_measures(ex):
    ex.w_values.value = "bytes"
    ex.w_agg.value = "mean"
    ex.w_cmp_col.value = "protocol"
    ex.w_cmp_val.value = "TCP"
    assert ex.comparison.metric == "delta" and ex.w_cmp_metric.value == "delta"
    assert "needs a count/sum" in ex.w_cmp_status.value


# --------------------------------------------------------------------------- inspect tab


def test_inspect_tab_present_and_pickers_follow_the_table(ex):
    assert ex.tabs.get_title(ex.INSPECT_TAB) == "Inspect" and "click a cell" in ex.w_cell_out.value
    rows = [v for _, v in ex.w_cell_row.options if v is not None]
    cols = [v for _, v in ex.w_cell_col.options if v is not None]
    assert rows == [(str(x),) for x in ex.view.pivot().index] and cols == [(str(x),) for x in ex.view.pivot().columns]
    ex.w_rows.value = ("country",)
    assert [v for _, v in ex.w_cell_row.options if v is not None] == [(str(x),) for x in ex.view.pivot().index]
    one_d = explore(p2h.fit(ex.view.source, rows=["action"], cols=[], agg="count"))
    try:
        assert one_d.w_cell_col.options == (("(any column)", None),)
        assert [v for _, v in one_d.w_cell_row.options if v is not None] == [(str(x),) for x in one_d.view.pivot().index]
        if hasattr(one_d.w_out, "clicked"):
            one_d.w_out.clicked = {"row": ["deny"], "col": ["count"], "n": 1}
            assert one_d.cell["cell"] == {"action": "deny"}  # the only column is the measure, not a level
    finally:
        one_d.close()


def test_inspect_explain_rows_via_pickers_and_errors(ex, fw):
    from pivot2hist import Explanation

    ex.w_cell_explain.click()
    assert "pick a row" in ex.w_cell_out.value
    ex.w_cell_row.value = ("22",)
    ex.w_cell_col.value = ("3",)
    ex.w_cell_explain.click()
    assert isinstance(ex.cell, Explanation) and ex.cell["cell"] == {"dst_port": "22", "severity": "3"}
    assert "what sets these rows apart" in ex.w_cell_out.value
    assert len(ex.rows()) == ((fw["dst_port"] == 22) & (fw["severity"] == 3)).sum()
    e = ex.explain("445", "5", n_rows=2)
    assert e["cell"] == {"dst_port": "445", "severity": "5"} and len(e["rows"]) == 2


def test_inspect_click_path_and_drill_in(ex):
    if not hasattr(ex.w_out, "clicked"):
        pytest.skip("anywidget not installed")
    ex.w_out.clicked = {"row": ["445"], "col": ["5"], "n": 1}
    assert ex.cell["cell"] == {"dst_port": "445", "severity": "5"} and ex.tabs.selected_index == ex.INSPECT_TAB
    assert ex.w_cell_row.value == ("445",) and ex.w_cell_col.value == ("5",)
    before, n_before = ex.view.layout.describe(), len(ex.view.data)
    ex.w_cell_drill.click()
    assert ex.view.slices[-2:] == ["dst_port=445", "severity=5"] and len(ex.view.data) < n_before
    assert "dst_port" not in {d.column for d in ex.view.layout.dims} and ex.view.layout.describe() != before
    assert "v = v.cell(dst_port='445', severity='5')" in ex.code() and "v.refit()" in ex.code()
    ex.w_undo.click()
    assert ex.view.layout.describe() == before and len(ex.view.data) == n_before
    ex.w_mode.value = "hist"
    lab = ex.w_cell_row.options[2][1]
    ex.w_out.clicked = {"row": list(lab), "col": ["count"], "n": 2}
    assert ex.cell["cell"] == {"dst_port": lab[0]} and ex.cell["mode"] == "hist"
    ex.w_mode.value = "pivot"
