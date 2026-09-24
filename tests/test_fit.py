import numpy as np
import pandas as pd
import pytest

import pivot2hist as p2h
from pivot2hist import FitOptions, Layout
from pivot2hist._fit import build_table, fit_layout


@pytest.mark.parametrize("box", [(40, 12), (20, 6), (10, 4), (8, 3)])
def test_fit_respects_box(fw, box):
    r, c = box
    v = p2h.fit(fw, max_rows=r, max_cols=c)
    n_r, n_c = v.pivot().shape
    assert n_r <= r and n_c <= c
    assert n_r >= 2 and n_c >= 1


def test_default_layout_is_sensible(fw):
    v = p2h.fit(fw)
    lay = v.layout
    assert lay.values == "bytes" and lay.agg == "sum"
    assert lay.rows and lay.cols
    dims = {d.column for d in lay.dims}
    assert "bytes" not in dims
    assert v.pivot().shape[0] <= 40 and v.pivot().shape[1] <= 12


def test_layers(fw):
    v1 = p2h.fit(fw, layers=1)
    assert len(v1.layout.rows) == 1 and len(v1.layout.cols) <= 1
    v2 = p2h.fit(fw, layers=2)
    assert all(len(ax) <= 2 for ax in (v2.layout.rows, v2.layout.cols))
    assert v2.pivot().shape[0] <= 40


def test_multi_layer_when_natural(auth):
    v = p2h.fit(auth)
    assert v.layout.values is None  # no additive column -> count
    assert len(v.layout.dims) >= 3  # host x event > method (or similar)
    assert isinstance(v.pivot().columns, pd.MultiIndex) or isinstance(v.pivot().index, pd.MultiIndex)


def test_fixed_axes(fw):
    v = p2h.fit(fw, rows=["src_ip"], cols=["action"], values="duration", agg="mean")
    t = v.pivot()
    assert t.index.name.startswith("src_ip")
    assert list(t.columns) == ["allow", "deny", "drop"]
    assert t.shape[0] <= 40
    assert v.layout.measure == "mean(duration)"


def test_fixed_rows_only_auto_cols(fw):
    v = p2h.fit(fw, rows=["timestamp"])
    assert v.layout.rows[0].kind == "time"
    assert v.pivot().shape[0] <= 40
    assert v.layout.cols  # something was chosen for columns


def test_count_measure(fw):
    v = p2h.fit(fw, agg="count")
    assert v.layout.values is None and v.layout.measure == "count"
    assert v.pivot().to_numpy().sum() == len(fw)


def test_dim_specs(fw):
    v = p2h.fit(fw, rows=[{"column": "bytes", "bins": 5}], cols=[{"column": "src_ip", "top": 3}], agg="count")
    t = v.pivot()
    assert t.shape[0] <= 5
    assert list(t.columns)[-1] == "(other)" and t.shape[1] == 4
    v2 = p2h.fit(fw, rows=[{"column": "timestamp", "freq": "D"}], cols=[], agg="count")
    assert v2.pivot().shape == (7, 1) or v2.pivot().shape[0] in (7, 8)


def test_aspect_changes_shape(fw):
    tall = p2h.fit(fw, max_rows=40, max_cols=12, aspect=20, layers=1).pivot().shape
    wide = p2h.fit(fw, max_rows=40, max_cols=12, aspect=0.5, layers=1).pivot().shape
    assert tall[0] / tall[1] > wide[0] / wide[1]


def test_no_categoricals_single_numeric():
    df = pd.DataFrame({"x": np.random.default_rng(0).normal(size=500)})
    v = p2h.fit(df)
    assert v.layout.rows[0].kind == "binned" and v.layout.measure == "count"
    assert v.pivot()["count"].sum() == 500


def test_nulls_become_a_level():
    df = pd.DataFrame({"a": ["x", "y", None] * 50, "b": [1, 2, 3] * 50, "n": range(150)})
    v = p2h.fit(df, rows=["a"], cols=["b"], agg="count")
    assert "(null)" in list(v.pivot().index)


def test_empty_and_degenerate():
    with pytest.raises(ValueError):
        p2h.fit(pd.DataFrame({"a": []}))
    with pytest.raises(ValueError):
        p2h.fit(pd.DataFrame(index=range(3)))
    const = pd.DataFrame({"a": [1] * 10, "b": ["k"] * 10})
    v = p2h.fit(const)
    assert v.pivot().shape[0] >= 1


def test_ranked_candidates(fw):
    ranked = []
    fit_layout(fw, FitOptions(layers=1), ranked=ranked)
    assert ranked and ranked[0][0] >= ranked[-1][0]
    assert all(isinstance(l, Layout) for _, l in ranked)


def test_build_table_observed_false_keeps_all_bins(fw):
    v = p2h.fit(fw, rows=[{"column": "bytes", "bins": 6}], cols=["action"], agg="count")
    full = build_table(fw, v.layout, observed=False)
    assert len(full) == len(v.layout.rows[0].edges) - 1


def test_options_replace_rejects_unknown():
    with pytest.raises(TypeError):
        FitOptions().replace(nope=1)


def test_exclude(fw):
    v = p2h.fit(fw, exclude=["dst_ip", "rule", "country"])
    assert not {"dst_ip", "rule", "country"} & {d.column for d in v.layout.dims}


def test_large_frame_uses_sample_quickly():
    import time

    df = p2h.sample.firewall_logs(200_000, seed=3)
    t = time.time()
    v = p2h.fit(df, sample=20_000)
    v.pivot()
    assert time.time() - t < 20
