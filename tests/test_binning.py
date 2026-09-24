import numpy as np
import pandas as pd
import pytest

from pivot2hist import _binning as B


def test_nice_numbers():
    assert B.nice_number(3.2) == 5.0
    assert B.nice_number(0.11) == 0.2
    assert B.nice_number(2.2, integer=True) == 5.0
    assert B.nice_number(0.3, integer=True) == 1.0
    assert B.nice_number(1000) == 1000.0


def test_linear_edges_cover_range_and_respect_cap():
    e = B.linear_edges(0, 100, 10)
    assert e[0] == 0 and e[-1] == 100 and len(e) - 1 <= 10
    e = B.linear_edges(1, 5, 10, integer=True)
    assert list(e) == [1, 2, 3, 4, 5, 6]
    e = B.linear_edges(-3.3, 47.1, 6)
    assert e[0] <= -3.3 and e[-1] >= 47.1 and len(e) - 1 <= 6
    e = B.linear_edges(7, 7, 5)
    assert len(e) == 2


def test_log_edges_are_nice_and_capped():
    e = B.log_edges(3, 45000, 10)
    assert e[0] <= 3 and e[-1] >= 45000 and len(e) - 1 <= 10
    e = B.log_edges(1, 1e9, 4)
    assert len(e) - 1 <= 4
    with pytest.raises(ValueError):
        B.log_edges(0, 10, 5)


@pytest.mark.parametrize("rule", ["auto", "fd", "sturges", "scott", "sqrt", "rice", 7])
def test_bin_count_rules(rule):
    v = np.random.default_rng(0).normal(size=1000)
    n = B.bin_count(v, rule)
    assert 1 <= n <= 1000
    if rule == 7:
        assert n == 7


def test_bin_count_heavy_tail_is_bounded():
    v = np.concatenate([np.zeros(10000), [1e9]])
    assert B.bin_count(v, "auto") <= 4 * (int(np.ceil(np.log2(10001))) + 1)


def test_bin_edges_auto_log_for_heavy_tail():
    rng = np.random.default_rng(0)
    v = np.exp(rng.normal(7, 2, 5000)).round()
    e = B.bin_edges(v, max_bins=12)
    assert len(e) - 1 <= 12
    assert e[0] <= v.min() and e[-1] >= v.max()
    ratios = np.diff(e)[1:] / np.diff(e)[:-1]
    assert (ratios > 1.5).any()  # log-spaced
    lin = B.bin_edges(v, max_bins=12, scale="linear")
    assert np.allclose(np.diff(lin), np.diff(lin)[0])


def test_digitize_and_labels():
    e = np.array([0.0, 10.0, 20.0])
    assert list(B.digitize([0, 5, 10, 19.9, 20, np.nan], e)) == [0, 0, 1, 1, 1, -1]
    assert B.bin_labels(e, integer=True) == ["0-9", "10-19"]
    assert B.bin_labels(e) == ["[0, 10)", "[10, 20]"]
    assert B.bin_labels([1, 2, 3, 4], integer=True) == ["1", "2", "3"]
    assert B.bin_labels([0, 1, 10, 100], integer=True) == ["[0, 1)", "[1, 10)", "[10, 100]"]


def test_bin_series_with_nulls():
    s = pd.Series([1.0, 5.0, np.nan, 9.0])
    b = B.bin_series(s, [0, 5, 10], integer=False)
    assert b.cat.ordered
    assert list(b.astype(str)) == ["[0, 5)", "[5, 10]", "(null)", "[5, 10]"]
    assert list(B.bin_series(s, [0, 5, 10]).astype(str)) == ["0-4", "5-9", "(null)", "5-9"]


def test_human():
    assert B.human(950) == "950"
    assert B.human(1234) == "1.23K"
    assert B.human(12345678) == "12.3M"
    assert B.human(0.25) == "0.25"
    assert B.human(-2500) == "-2.5K"
    assert B.human(float("nan")) == "nan"


def test_time_buckets():
    ts = pd.Series(pd.date_range("2026-01-01", periods=5000, freq="2min"))
    assert B.time_freq_for(ts, 12) == "D"
    assert B.time_freq_for(ts, 40) == "6h"
    b = B.bucket_time(ts, "D")
    assert b.cat.ordered and len(b.cat.categories) == 7
    assert b.cat.categories[0] == "2026-01-01"
    m = B.bucket_time(pd.Series(pd.date_range("2025-01-01", periods=400, freq="D")), "M")
    assert list(m.cat.categories[:2]) == ["2025-01", "2025-02"]
    w = B.bucket_time(pd.Series(pd.date_range("2025-01-01", periods=30, freq="D")), "W")
    assert w.cat.categories[0].startswith("wk ")
    tz = pd.Series(pd.date_range("2026-01-01", periods=48, freq="h", tz="UTC"))
    assert len(B.bucket_time(tz, "D").cat.categories) == 2
    with_null = pd.Series([pd.Timestamp("2026-01-01"), pd.NaT])
    assert list(B.bucket_time(with_null, "D").astype(str)) == ["2026-01-01", "(null)"]


def test_categorize_top_n_and_order():
    s = pd.Series(["a"] * 5 + ["b"] * 3 + ["c"] * 2 + ["d"] + [None])
    c = B.categorize(s, top=2)
    assert list(c.cat.categories) == ["a", "b", "(other)", "(null)"]
    assert (c == "(other)").sum() == 3
    ports = B.categorize(pd.Series([443, 80, 22, 22, 443]), top=2)
    assert list(ports.cat.categories) == [22, 443, "(other)"]
    nat = B.categorize(pd.Series(["b", "a", "b"]), order="natural")
    assert list(nat.cat.categories) == ["a", "b"]
    boo = B.categorize(pd.Series([True, False, True]))
    assert list(boo.cat.categories) == [False, True]
