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
