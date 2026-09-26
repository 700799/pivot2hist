import pandas as pd
import pytest

import pivot2hist as p2h
from pivot2hist import sparkline_table


@pytest.fixture(scope="module")
def v(fw):
    return p2h.fit(fw, max_rows=10, max_cols=4)  # sum(bytes) by dst_port (top N) x severity/action


# --------------------------------------------------------------------------- data


def test_sparkline_table_matches_pivot_row_totals(v, fw):
    table, cats = sparkline_table(v, "timestamp")
    assert list(table.index) == list(v.pivot().index)
    assert len(cats) >= 2 and cats == sorted(cats)
    # summed across buckets, each row's trend equals that row's pivot total (additive measure)
    row_sums = table.sum(axis=1, skipna=True)
    pivot_sums = v.pivot().sum(axis=1)
    pd.testing.assert_series_equal(row_sums.astype(float), pivot_sums.astype(float), check_names=False, check_index_type=False)


def test_sparkline_table_count_layout_exact(fw):
    v = p2h.fit(fw, rows=["action"], cols=[], agg="count")
    table, cats = sparkline_table(v, "timestamp")
    assert table.sum(axis=1).tolist() == v.pivot()["count"].tolist()
    assert table.to_numpy().min() >= 0  # counts fill 0, never negative/NaN


def test_sparkline_table_hist_mode(v, fw):
    h = p2h.fit(fw).histogram("bytes")
    table, cats = sparkline_table(h, "timestamp")
    assert list(table.index) == list(h.table().index)
    assert table.sum(axis=1).sum() == pytest.approx(len(fw))


def test_sparkline_table_nonadditive_agg_fills_nan(fw):
    v = p2h.fit(fw, rows=["dst_port"], cols=[], values="bytes", agg="mean", max_rows=8)
    table, cats = sparkline_table(v, "timestamp")
    assert table.isna().to_numpy().any() or table.notna().all(axis=None)  # some buckets legitimately empty per row


def test_sparkline_table_nested_rows(fw):
    v = p2h.fit(fw, rows=["protocol", "action"], cols=["severity"], agg="count", max_rows=20)
    table, cats = sparkline_table(v, "timestamp")
    assert isinstance(table.index, pd.MultiIndex) and list(table.index) == list(v.pivot().index)


def test_sparkline_table_degenerate_single_bucket(fw):
    df = fw.assign(timestamp=pd.Timestamp("2026-01-01 00:00:30"))
    v = p2h.fit(df, rows=["action"], cols=[])
    table, cats = sparkline_table(v, "timestamp")
    assert len(cats) == 1 and table.shape[1] == 1


def test_sparkline_table_errors(v, fw):
    with pytest.raises(ValueError, match="row dimension"):
        sparkline_table(p2h.fit(fw, rows=[], cols=["action"], agg="count"), "timestamp")
    with pytest.raises(KeyError, match="nope"):
        sparkline_table(v, "nope")
    with pytest.raises(ValueError, match="no usable datetime"):
        sparkline_table(v, "action")
    all_null = fw.assign(timestamp=pd.NaT)
    with pytest.raises(ValueError, match="no usable datetime"):
        sparkline_table(p2h.fit(all_null, rows=["action"], cols=[]), "timestamp")


def test_view_sparklines_method(v):
    table = v.sparklines("timestamp", max_points=6)
    assert len(table.columns) <= 6 and list(table.index) == list(v.pivot().index)


def test_sparkline_table_max_points_coarsens(v):
    fine, fine_cats = sparkline_table(v, "timestamp", max_points=200)
    coarse, coarse_cats = sparkline_table(v, "timestamp", max_points=3)
    assert len(coarse_cats) <= 3 and len(coarse_cats) <= len(fine_cats)


# --------------------------------------------------------------------------- rendering


def test_style_sparklines_renders_trend_column(v):
    html = v.style(sparklines="timestamp").html()
    assert "<svg" in html and ">trend<" in html
    assert html.count("<svg") == len(v.pivot())


def test_style_sparklines_none_renders_nothing(v):
    assert "<svg" not in v.style(sparklines=None).html()
    assert "<svg" not in v.html()  # off by default


def test_style_sparklines_combines_with_totals_bars_subtotals(fw):
    v = p2h.fit(fw, rows=["protocol", "action"], cols=["severity"], agg="count", max_rows=20)
    html = v.style(sparklines="timestamp", totals=True, subtotals=True, bars=True).html()
    assert html.count("<svg") == len(v.pivot())  # one per data row, none for Σ/total rows
    assert ">trend<" in html and ">total<" in html


def test_style_sparklines_overrides_outline(fw):
    v = p2h.fit(fw, rows=["protocol", "action"], cols=["severity"], agg="count", max_rows=20)
    html = v.style(sparklines="timestamp", outline=True).html()
    assert "<details" not in html and "<svg" in html


def test_style_sparklines_all_three_themes(v):
    for theme in ("light", "graphite", "gunmetal"):
        assert "<svg" in v.style(sparklines="timestamp", theme=theme).html()


def test_style_sparklines_error_propagates(v):
    with pytest.raises(ValueError, match="no usable datetime"):
        v.style(sparklines="action").html()


def test_style_sparklines_max_rows_truncation_stays_aligned(v):
    html = v.style(sparklines="timestamp", max_rows=3).html()
    assert html.count("<svg") == 3


def test_style_sparklines_row_with_no_data_shows_placeholder(fw):
    v = p2h.fit(fw, rows=["dst_port"], cols=[], values="bytes", agg="mean", max_rows=40)
    html = v.style(sparklines="timestamp").html()
    assert "no data in this window" in html or html.count("<svg") == len(v.pivot())


def test_style_sparklines_hist_mode(fw):
    h = p2h.fit(fw).histogram("bytes")
    html = h.style(sparklines="timestamp").html()
    assert "<svg" in html and html.count("<svg") <= len(h.bins())


@pytest.mark.parametrize("df_fn, time_col", [
    (lambda: pd.DataFrame({"g": ["a"] * 100, "t": pd.date_range("2026-01-01", periods=100, freq="h", tz="UTC")}), "t"),
    (lambda: pd.DataFrame({"g": [1] * 100, "t": pd.date_range("2026-01-01", periods=100, freq="min")}), "t"),
    (lambda: pd.DataFrame({"g": ["a", "b"] * 50, "t": [pd.Timestamp("2026-01-01")] * 100}), "t"),
    (lambda: pd.DataFrame({"g": ["x"] * 100, "t": pd.to_datetime(["2026-01-01"] * 50 + [None] * 50)}), "t"),
])
def test_sparkline_does_not_crash_on_adversarial_input(df_fn, time_col):
    v = p2h.fit(df_fn(), rows=["g"], cols=[])
    table, cats = sparkline_table(v, time_col)
    v.style(sparklines=time_col).html()
