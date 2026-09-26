import xml.dom.minidom

import pandas as pd
import pytest

import pivot2hist as p2h
from pivot2hist.ui_fields import (
    FIELD_THEMES,
    HAS_ANYWIDGET,
    MARK_PALETTE,
    ZONES,
    FieldListFallback,
    chip_html,
    field_stats,
    glyph_svg,
    make_field_list,
    theme_style_block,
    zones_html,
)


def _xml_fragment(s: str) -> None:
    xml.dom.minidom.parseString(f"<root>{s}</root>")


@pytest.fixture(scope="module")
def prof():
    return p2h.profile(p2h.sample.firewall_logs(2000, seed=1))


def test_field_stats_shape_and_variety(prof):
    stats = field_stats(prof)
    assert {f["name"] for f in stats} == set(prof.summary()["column"])
    ip = next(f for f in stats if f["name"] == "src_ip")
    assert ip["kind"] == "categorical" and ip["semantic"] == "ipv4"
    assert 0 < ip["variety"] < 1
    assert ip["count"] == 2000
    action = next(f for f in stats if f["name"] == "action")
    assert action["distinct"] == 3 and action["variety"] < 0.01  # a handful of repeated labels


def test_chip_html_escapes_and_marks_removable():
    f = {"name": "<x>", "kind": "categorical", "semantic": "", "count": 10, "distinct": 3, "variety": 0.3, "nulls": 0.0, "examples": "a, b", "ts": ""}
    h = chip_html(f, removable=True)
    _xml_fragment(h)
    assert "&lt;x&gt;" in h and "p2h-x" in h
    assert "p2h-x" not in chip_html(f, removable=False)


def test_zones_html_light_and_graphite(prof):
    fields = field_stats(prof)
    zones = {"rows": ["src_ip"], "cols": ["action"], "values": ["bytes"], "slicers": []}
    light = zones_html(fields, zones)
    dark = zones_html(fields, zones, theme="graphite")
    _xml_fragment(light)
    _xml_fragment(dark)
    assert light != dark and "#161a20" in dark and "#161a20" not in light
    assert "src_ip" in light and "action" in light and "bytes" in light
    non_interactive = zones_html(fields, zones, interactive=False)
    assert "class='p2h-x'" not in non_interactive  # the CSS rule for the class is still emitted; only the chip markup is gone


def test_theme_style_block():
    css = theme_style_block("graphite", selector=".scope")
    assert css.startswith("<style>.scope{") and "--p2h-accent:" in css
    assert theme_style_block("light") != theme_style_block("graphite")
    with pytest.raises(KeyError):
        _ = FIELD_THEMES["not-a-theme"]
    # unknown theme falls back to light rather than raising
    assert theme_style_block("not-a-theme") == theme_style_block("light")


def test_field_list_fallback_move_and_zones(prof):
    fields = field_stats(prof)
    fb = FieldListFallback(fields, rows=["src_ip"])
    assert fb.rows == ("src_ip",) or list(fb.rows) == ["src_ip"]
    fb.move("action", "cols")
    fb.move("dst_port", "rows")
    assert list(fb.cols) == ["action"] and list(fb.rows) == ["src_ip", "dst_port"]
    fb._shift("rows", "dst_port", -1)
    assert list(fb.rows) == ["dst_port", "src_ip"]
    fb.move("dst_port", None)  # remove
    assert "dst_port" not in list(fb.rows)
    fb.set_zones(rows=["country"], cols=[])
    assert list(fb.rows) == ["country"] and list(fb.cols) == []
    assert "country" in fb.snapshot_html()
    fb2 = FieldListFallback(fields, theme="graphite")
    assert "#161a20" in fb2.snapshot_html()


def test_field_list_fallback_values_single_slot(prof):
    fields = field_stats(prof)
    fb = FieldListFallback(fields)
    fb.move("bytes", "values")
    fb.move("duration", "values")
    assert list(fb.values) == ["duration"]  # one measure at a time


def test_make_field_list_dispatch(prof):
    w = make_field_list(prof, rows=["src_ip"])
    assert list(w.rows) == ["src_ip"]
    assert type(w).__name__ in ("FieldList", "FieldListFallback")
    forced = make_field_list(prof, prefer_anywidget=False, rows=["src_ip"])
    assert isinstance(forced, FieldListFallback)


@pytest.mark.skipif(not HAS_ANYWIDGET, reason="anywidget not installed")
def test_field_list_anywidget_traits_and_events(prof):
    from pivot2hist.ui_fields import FieldList

    fields = field_stats(prof)
    w = FieldList(fields=fields)
    seen = []
    w.observe(lambda ch: seen.append((ch["name"], list(ch["new"]))), names=list(ZONES))
    w.rows = ["src_ip", "dst_port"]
    w.cols = ["action"]
    assert seen == [("rows", ["src_ip", "dst_port"]), ("cols", ["action"])]
    w.set_zones(values=["bytes"], slicers=["country"])
    assert list(w.values) == ["bytes"] and list(w.slicers) == ["country"]
    w.theme = "graphite"
    assert w.theme == "graphite"
    snap = w.snapshot_html()
    _xml_fragment(snap)
    assert "src_ip" in snap and "#161a20" in snap
    assert "_esm" in dir(FieldList) or hasattr(FieldList, "_esm")
    assert "applyTheme" in FieldList._esm and "dragstart" in FieldList._esm


def test_chip_html_mark_styling():
    f = {"name": "src_ip", "kind": "categorical", "semantic": "", "count": 100, "distinct": 5,
         "variety": 0.05, "nulls": 0.0, "examples": "1.2.3.4"}
    plain = chip_html(f)
    marked = chip_html(f, mark="#e5484d")
    assert "border-left" not in plain
    assert "border-left:4px solid #e5484d" in marked
    _xml_fragment(marked)


def test_zones_html_threads_marks_through(prof):
    fields = field_stats(prof)
    html = zones_html(fields, {"rows": ["src_ip"]}, marks={"src_ip": "#3b82f6"})
    _xml_fragment(html)
    assert "#3b82f6" in html


def test_fallback_paint_and_unmark(prof):
    fields = field_stats(prof)
    w = FieldListFallback(fields, rows=["src_ip"])
    assert w.marks == {}
    w.paint("src_ip", "red")
    assert w.marks == {"src_ip": MARK_PALETTE["red"]}
    w.paint("src_ip", "#123456")  # raw CSS color, not just named palette
    assert w.marks == {"src_ip": "#123456"}
    w.paint("action", "blue")
    w.unmark("src_ip")
    assert w.marks == {"action": MARK_PALETTE["blue"]}
    w.unmark()
    assert w.marks == {}
    snap = w.snapshot_html()
    _xml_fragment(snap)


def test_make_field_list_constructor_marks(prof):
    w = make_field_list(prof, rows=["src_ip"], marks={"src_ip": "yellow"})
    assert w.marks == {"src_ip": MARK_PALETTE["yellow"]}


@pytest.mark.skipif(not HAS_ANYWIDGET, reason="anywidget not installed")
def test_field_list_anywidget_paint_and_unmark(prof):
    from pivot2hist.ui_fields import FieldList

    w = FieldList(fields=field_stats(prof))
    seen = []
    w.observe(lambda ch: seen.append(dict(ch["new"])), names="marks")
    w.paint("src_ip", "red")
    assert w.marks == {"src_ip": MARK_PALETTE["red"]}
    assert seen == [{"src_ip": MARK_PALETTE["red"]}]
    w.paint("src_ip", None)
    assert w.marks == {}
    w.paint("dst_port", "green")
    w.paint("action", "blue")
    w.unmark()
    assert w.marks == {}
    assert "cycleMark" in FieldList._esm  # click-to-highlight is wired in the JS too


def test_gunmetal_field_theme_and_js_table():
    from pivot2hist.ui_fields import FIELD_THEMES, _ESM, theme_style_block

    assert set(FIELD_THEMES["gunmetal"]) == set(FIELD_THEMES["graphite"])
    assert "--p2h-bg:#1f262d" in theme_style_block("gunmetal")
    assert "gunmetal: {" in _ESM and '"#1f262d"' in _ESM


# --------------------------------------------------------------------------- chip glyphs


@pytest.fixture(scope="module")
def fw_df():
    return p2h.sample.firewall_logs(2000, seed=1)


def test_field_stats_glyph_only_for_numeric_and_datetime(prof, fw_df):
    stats = field_stats(prof, fw_df)
    by = {f["name"]: f for f in stats}
    assert by["bytes"]["glyph_kind"] == "hist" and len(by["bytes"]["glyph"]) == 12
    assert all(0.0 <= v <= 1.0 for v in by["bytes"]["glyph"]) and max(by["bytes"]["glyph"]) == 1.0
    assert by["timestamp"]["glyph_kind"] == "trend" and len(by["timestamp"]["glyph"]) >= 2
    for name in ("action", "src_ip", "protocol"):
        assert by[name]["glyph"] is None and by[name]["glyph_kind"] is None


def test_field_stats_without_df_has_no_glyphs(prof):
    stats = field_stats(prof)
    assert all(f["glyph"] is None and f["glyph_kind"] is None for f in stats)


def test_field_stats_glyph_edge_cases():
    const_df = pd.DataFrame({"a": [5] * 100, "b": range(100)})
    prof_c = p2h.profile(const_df)
    by = {f["name"]: f for f in field_stats(prof_c, const_df)}
    assert by["a"]["glyph"] is None  # no spread: nothing to draw

    tiny = pd.DataFrame({"a": [1, 2, 3]})
    by_tiny = {f["name"]: f for f in field_stats(p2h.profile(tiny), tiny)}
    assert by_tiny["a"]["glyph"] is None  # too few points

    null_df = pd.DataFrame({"a": [None] * 50, "b": range(50)})
    by_null = {f["name"]: f for f in field_stats(p2h.profile(null_df), null_df)}
    assert by_null["a"]["glyph"] is None

    single_time = pd.DataFrame({"t": [pd.Timestamp("2026-01-01")] * 50, "b": range(50)})
    by_t = {f["name"]: f for f in field_stats(p2h.profile(single_time), single_time)}
    assert by_t["t"]["glyph"] is None  # one bucket only: not a trend worth drawing


def test_glyph_svg_hist_and_trend_and_empty():
    hist = glyph_svg([0.2, 1.0, 0.5], "hist", color="#123")
    assert hist.count("<rect") == 3 and "#123" in hist
    trend = glyph_svg([0.0, 0.5, 1.0], "trend", color="#456")
    assert "<polyline" in trend and "#456" in trend
    assert glyph_svg(None, "hist") == "" and glyph_svg([], "trend") == ""


def test_chip_html_renders_glyph_and_tolerates_missing_keys():
    f = {"name": "bytes", "kind": "numeric", "semantic": "", "count": 100, "distinct": 80, "variety": 0.8,
         "nulls": 0.0, "examples": "1, 2, 3", "ts": "", "glyph": [0.1, 0.5, 1.0], "glyph_kind": "hist"}
    h = chip_html(f)
    _xml_fragment(h)
    assert "<svg" in h and "p2h-glyph" in h
    no_glyph = {"name": "x", "kind": "categorical", "semantic": "", "count": 10, "distinct": 3, "variety": 0.3, "nulls": 0.0, "examples": "", "ts": ""}
    h2 = chip_html(no_glyph)  # no "glyph"/"glyph_kind" keys at all
    _xml_fragment(h2)
    assert "<svg" not in h2


def test_make_field_list_threads_df_to_glyphs(prof, fw_df):
    fb = make_field_list(prof, fw_df, prefer_anywidget=False)
    by = {f["name"]: f for f in fb.fields}
    assert by["bytes"]["glyph_kind"] == "hist"
    snap = fb.snapshot_html()
    _xml_fragment(snap)
    assert "<svg" in snap

    fb_none = make_field_list(prof, prefer_anywidget=False)
    assert all(f["glyph"] is None for f in fb_none.fields)


@pytest.mark.skipif(not HAS_ANYWIDGET, reason="needs anywidget")
def test_make_field_list_anywidget_has_glyphs(prof, fw_df):
    w = make_field_list(prof, fw_df, prefer_anywidget=True)
    by = {f["name"]: f for f in w.fields}
    assert by["bytes"]["glyph_kind"] == "hist" and by["timestamp"]["glyph_kind"] == "trend"
