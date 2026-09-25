import xml.dom.minidom

import pandas as pd

import pivot2hist as p2h
from pivot2hist._html import heat_color, hist_svg, pivot_html


def _xml(s: str) -> None:
    xml.dom.minidom.parseString(s)


def test_pivot_html_structure(fw):
    v = p2h.fit(fw, max_rows=8, max_cols=4)
    h = pivot_html(v.pivot(), title=v.title(), totals=True, bars=True)
    _xml(h)
    assert "total" in h and "background:rgb(" in h and "<title" not in h
    assert h.count("<tr>") == 8 + 2 + 1  # rows + header + totals (+ row-name header)
    plain = pivot_html(v.pivot(), heat="none")
    assert "rgb(" not in plain


def test_pivot_html_multiindex_and_escaping():
    df = pd.DataFrame({"a": ["<b>", "<b>", "y"], "b": ["s&t", "u", "u"], "n": [1, 2, 3]})
    v = p2h.fit(df, rows=["a", "b"], cols=[], agg="count")
    h = pivot_html(v.pivot())
    _xml(h)
    assert "&lt;b&gt;" in h and "s&amp;t" in h
    m = p2h.fit(p2h.sample.firewall_logs(500), rows=["severity", "action"], cols=["protocol", "action"], agg="count")
    h2 = pivot_html(m.pivot(), heat="column")
    _xml(h2)
    assert "colspan=" in h2


def test_pivot_html_empty_and_max_rows(fw):
    v = p2h.fit(fw).slice(action="nope")
    assert "(empty)" in pivot_html(v.pivot())
    h = pivot_html(p2h.fit(fw).pivot(), max_rows=3)
    assert "more rows" in h


def test_heat_color_range():
    assert heat_color(0) == "rgb(255,255,255)"
    assert heat_color(1) == "rgb(15,95,175)"
    assert heat_color(float("nan")) == "rgb(255,255,255)"
    assert heat_color(0.5, negative=True) != heat_color(0.5)


def test_hist_svg_variants(fw):
    v = p2h.fit(fw, max_rows=8, max_cols=4)
    s = hist_svg(v.toggle().bins(), title="t")
    _xml(s[s.index("<svg"):s.index("</svg>") + 6])
    assert s.count("<rect") >= 8 and "<title>" in s
    stacked = hist_svg(v.toggle().bins(), stacked=True, log_y=True)
    _xml(stacked[stacked.index("<svg"):stacked.index("</svg>") + 6])
    nested = p2h.fit(fw, rows=["severity", "action"], cols=["protocol"], agg="count").toggle()
    s3 = hist_svg(nested.bins())
    assert "stroke='#999'" in s3  # group brackets
    hb = v.histogram("bytes")
    s4 = hist_svg(hb.bins(), density=p2h._binning.kde(fw["bytes"], log=True))
    assert "polyline" in s4
    assert "(empty)" in hist_svg(pd.DataFrame())


def test_view_html_and_svg(fw):
    v = p2h.fit(fw, max_rows=8, max_cols=4)
    assert "<table" in v.html() and "<table" in v._repr_html_()
    assert "<svg" in v.svg() and "<svg" in v.toggle().html()
    styled = v.style(totals=True, heat="row", bars=True, compact=True)
    assert styled.display["totals"] is True and "total" in styled.html()
    assert "polyline" in v.histogram("duration").html()
    assert "polyline" not in v.histogram("duration").style(density=False).html()
    big = v.toggle().style(width=1000, height=500, stacked=True).html()
    assert "width='1000'" in big


def test_theme_default_is_byte_identical_to_light():
    df = p2h.sample.firewall_logs(300)
    v = p2h.fit(df, max_rows=8, max_cols=4)
    assert pivot_html(v.pivot()) == pivot_html(v.pivot(), theme="light")
    h = v.toggle()
    assert hist_svg(h.bins()) == hist_svg(h.bins(), theme="light")


def test_graphite_theme_produces_dark_output(fw):
    v = p2h.fit(fw, max_rows=8, max_cols=4)
    light = pivot_html(v.pivot())
    dark = pivot_html(v.pivot(), theme="graphite")
    _xml(dark)
    assert light != dark
    assert "#161a20" not in light and ("#1b2128" in dark or "#161a20" in dark)
    hd = hist_svg(v.toggle().bins(), theme="graphite")
    _xml(hd[hd.index("<svg"):hd.index("</svg>") + 6])
    assert "#161a20" in hd or "#1b2128" in hd


def test_unknown_theme_raises():
    import pytest as _pytest

    with _pytest.raises(ValueError):
        heat_color(0.5, theme="neon")
    with _pytest.raises(ValueError):
        pivot_html(p2h.fit(p2h.sample.firewall_logs(50)).pivot(), theme="neon")


def test_subtotals_and_outline(fw):
    v = p2h.fit(fw, rows=["severity", "action"], cols=["protocol"], agg="count", max_rows=15)
    plain = pivot_html(v.pivot(), agg="count")
    sub = pivot_html(v.pivot(), subtotals=True, totals=True, agg="count")
    _xml(sub)
    assert sub.count("∑") == v.pivot().index.get_level_values(0).nunique()
    assert plain.count("<tr>") < sub.count("<tr>")
    out = pivot_html(v.pivot(), outline=True, totals=True, agg="count")
    _xml(out)
    assert out.count("<details") == v.pivot().index.get_level_values(0).nunique()
    assert "p2h-outline" in out
    # subtotal values actually sum the group (agg=count -> sum; agg=mean -> mean)
    m = p2h.fit(fw, rows=["severity", "action"], cols=["protocol"], values="bytes", agg="mean")
    msub = pivot_html(m.pivot(), subtotals=True, agg="mean")
    assert "∑" in msub
    # single-level rows: subtotals/outline are no-ops, never crash
    flat = p2h.fit(fw, rows=["action"], cols=["protocol"], agg="count")
    assert pivot_html(flat.pivot(), subtotals=True, agg="count") == pivot_html(flat.pivot(), agg="count")
    assert pivot_html(flat.pivot(), outline=True, agg="count") == pivot_html(flat.pivot(), agg="count")


def test_view_style_theme_subtotals_outline(fw):
    v = p2h.fit(fw, rows=["severity", "action"], cols=["protocol"], agg="count", max_rows=15)
    assert v.style(theme="graphite").html() != v.html()
    assert "∑" in v.style(subtotals=True, totals=True).html()
    assert "<details" in v.style(outline=True).html()
    assert v.toggle().style(theme="graphite").html() != v.toggle().html()
