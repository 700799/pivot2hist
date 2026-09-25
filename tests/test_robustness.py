"""Safety net for unusual/bad input: things that used to crash with an opaque, low-level
error now either work or fail with a clear, pivot2hist-level message. Each test here
traces back to a bug found by adversarial probing, not a hypothetical.
"""
import numpy as np
import pandas as pd
import pytest

import pivot2hist as p2h
from pivot2hist._fit import FitOptions


def _round_trip(df, **kw):
    """profile -> fit -> pivot -> toggle -> bins -> render, the full pipeline."""
    p2h.profile(df)
    v = p2h.fit(df, **kw)
    v.pivot()
    v.toggle().bins()
    str(v)
    return v


# --------------------------------------------------------------------------- unhashable cells


def test_unhashable_list_cells_do_not_crash():
    df = pd.DataFrame({"a": [[1, 2]] * 50 + [[3, 4]] * 50, "b": range(100)})
    v = _round_trip(df)
    t = v.pivot()
    assert t.to_numpy().sum() == 100


def test_unhashable_dict_cells_do_not_crash():
    df = pd.DataFrame({"a": [{"x": 1}] * 100, "b": range(100)})
    _round_trip(df)


def test_unhashable_cells_nulls_stay_null_not_stringified():
    df = pd.DataFrame({"a": [[1, 2]] * 40 + [None] * 40 + [[3, 4]] * 20, "b": range(100)})
    v = p2h.fit(df, rows=["a"], cols=None, values=None, agg=None)
    labels = [str(x) for x in v.pivot().index]
    assert "(null)" in labels
    assert "None" not in labels  # nulls must not get stringified into the literal "None"


def test_unhashable_values_column_gives_clear_error_not_a_crash():
    df = pd.DataFrame({"a": ["x", "y"] * 50, "b": [[1, 2]] * 100})
    with pytest.raises(ValueError, match="non-numeric"):
        p2h.fit(df, rows=["a"], values="b", agg="sum").pivot()


def test_unhashable_cells_via_duckdb_engine_fall_back_cleanly():
    duckdb = pytest.importorskip("duckdb")
    del duckdb
    df = pd.DataFrame({"a": [[1, 2]] * 50 + [[3, 4]] * 50, "b": range(100)})
    v_pd = p2h.fit(df)
    v_dd = p2h.fit(df, engine="duckdb")
    assert v_pd.pivot().to_numpy().tolist() == v_dd.pivot().sort_index().reindex(v_pd.pivot().index).to_numpy().tolist()


# --------------------------------------------------------------------------- bad/None/wrong-type input


@pytest.mark.parametrize("bad", [None, 12345, 3.14, True, b"bytes"])
def test_load_rejects_non_tabular_scalars_clearly(bad):
    with pytest.raises(TypeError, match="tabular data"):
        p2h.load(bad)


def test_fit_none_raises_clear_type_error():
    with pytest.raises(TypeError, match="tabular data"):
        p2h.fit(None)


def test_agent_describe_bad_source_type_raises_clear_error():
    with pytest.raises(TypeError, match="tabular data"):
        p2h.agent.describe(12345)


# --------------------------------------------------------------------------- degenerate FitOptions


def test_sample_zero_raises_clear_error_instead_of_pandas_indexerror():
    with pytest.raises(ValueError, match="sample must be >= 1"):
        FitOptions(sample=0)


def test_sample_negative_raises():
    with pytest.raises(ValueError, match="sample must be >= 1"):
        FitOptions(sample=-5)


def test_sample_one_still_works():
    fw = p2h.sample.firewall_logs(300, seed=1)
    v = p2h.fit(fw, sample=1)
    v.pivot()


@pytest.mark.parametrize("kw", [
    {"max_rows": 0}, {"max_cols": 0}, {"max_rows": -5}, {"layers": 0}, {"layers": -1},
    {"aspect": 0}, {"aspect": -2}, {"max_bins": 0}, {"search_width": 0}, {"discrete_max": 0},
    {"max_categories": 0}, {"id_ratio": -1}, {"max_sparsity": -1},
])
def test_other_degenerate_options_degrade_gracefully(kw):
    fw = p2h.sample.firewall_logs(300, seed=1)
    v = p2h.fit(fw, **kw)
    v.pivot()  # must not raise; the exact layout chosen is not the point here


# --------------------------------------------------------------------------- mixture / GMM


def test_fit_gmm_empty_array_raises_clear_error():
    from pivot2hist import fit_gmm

    with pytest.raises(ValueError, match="at least 1 data point"):
        fit_gmm(np.empty((0, 1)), 2)


def test_choose_gmm_k_empty_array_does_not_crash():
    from pivot2hist import choose_gmm_k

    assert choose_gmm_k(np.empty((0, 1))) == 1


# --------------------------------------------------------------------------- structural edge cases


def test_empty_frame_raises_clear_error():
    with pytest.raises(ValueError, match="no columns"):
        p2h.fit(pd.DataFrame())


def test_zero_row_frame_raises_clear_error():
    with pytest.raises(ValueError, match="empty frame"):
        p2h.fit(pd.DataFrame({"a": pd.Series(dtype=float), "b": pd.Series(dtype=object)}))


def test_single_row_frame_does_not_crash():
    _round_trip(pd.DataFrame({"a": [1], "b": ["x"]}))


def test_all_null_column_does_not_crash():
    df = pd.DataFrame({"a": [None] * 100, "b": range(100)})
    _round_trip(df)


def test_all_null_frame_does_not_crash():
    df = pd.DataFrame({"a": [None] * 50, "b": [np.nan] * 50})
    _round_trip(df)


def test_duplicate_column_names_do_not_crash():
    df = pd.DataFrame([[1, 2]] * 50, columns=["a", "a"])
    _round_trip(df)


def test_non_string_column_names_do_not_crash():
    df = pd.DataFrame({1: range(100), 2.5: range(100), (1, 2): ["x"] * 100})
    _round_trip(df)


@pytest.mark.parametrize("dtype,values", [
    ("Int64", [1, 2, None, 4] * 25),
    ("boolean", [True, False, None] * 33 + [True]),
    ("string", ["x", "y", None, "z"] * 25),
    ("Float64", [1.5, 2.5, None] * 33 + [1.0]),
])
def test_pandas_nullable_extension_dtypes_do_not_crash(dtype, values):
    df = pd.DataFrame({"a": pd.array(values, dtype=dtype), "b": range(100)})
    _round_trip(df)


def test_categorical_with_unused_categories_does_not_crash():
    df = pd.DataFrame({"a": pd.Categorical(["x"] * 100, categories=["x", "y", "z", "unused"]), "b": range(100)})
    _round_trip(df)


def test_inf_values_do_not_crash():
    df = pd.DataFrame({"a": [np.inf, -np.inf, 1.0, 2.0] * 25, "b": range(100)})
    _round_trip(df)


def test_extreme_magnitude_numbers_do_not_crash():
    df = pd.DataFrame({"a": [1e300, -1e300, 1e-300] * 33 + [0.0], "b": range(100)})
    _round_trip(df)


def test_unicode_and_emoji_do_not_crash():
    df = pd.DataFrame({"名前": ["日本語", "🎉emoji", "normal"] * 33 + ["x"], "b": range(100)})
    _round_trip(df)


def test_tz_aware_datetime_does_not_crash():
    df = pd.DataFrame({"a": pd.date_range("2024-01-01", periods=100, tz="UTC"), "b": range(100)})
    _round_trip(df)


def test_nat_in_datetime_does_not_crash():
    dates = list(pd.date_range("2024-01-01", periods=90)) + [pd.NaT] * 10
    df = pd.DataFrame({"a": dates, "b": range(100)})
    _round_trip(df)


def test_mixed_date_and_string_object_column_does_not_crash():
    import datetime

    df = pd.DataFrame({"a": [datetime.date(2024, 1, 1)] * 50 + ["not a date"] * 50, "b": range(100)})
    _round_trip(df)


def test_bytes_column_does_not_crash():
    df = pd.DataFrame({"a": [b"hello"] * 50 + [b"world"] * 50, "b": range(100)})
    _round_trip(df)


def test_sparse_dtype_column_does_not_crash():
    df = pd.DataFrame({"a": pd.arrays.SparseArray([0] * 90 + [1] * 10), "b": range(100)})
    _round_trip(df)


def test_complex_number_column_does_not_crash():
    df = pd.DataFrame({"a": [complex(i, i) for i in range(100)], "b": range(100)})
    with pytest.warns(Warning):  # pandas warns when casting complex -> real for binning
        _round_trip(df)


def test_series_input_does_not_crash():
    _round_trip(pd.Series(range(100), name="a"))


def test_2d_numpy_array_input_does_not_crash():
    _round_trip(np.random.default_rng(0).random((100, 5)))


def test_empty_string_column_name_does_not_crash():
    df = pd.DataFrame({"": range(100), "b": range(100)})
    _round_trip(df)


def test_all_unique_id_like_column_does_not_crash():
    df = pd.DataFrame({"a": [f"id{i}" for i in range(200)], "b": range(200)})
    _round_trip(df)
