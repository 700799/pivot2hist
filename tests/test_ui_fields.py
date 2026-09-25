import xml.dom.minidom

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
