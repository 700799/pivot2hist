import numpy as np
import pandas as pd
import pytest

import pivot2hist as p2h

rng = np.random.default_rng(0)


@pytest.fixture(scope="module")
def fw():
    return p2h.sample.firewall_logs(4000, seed=1)


def _kinds(report, kind):
    return [f for f in report["findings"] if f["kind"] == kind]


def test_insights_shape(fw):
    v = p2h.fit(fw)
    r = v.insights()
    assert set(r) == {"summary", "shape", "columns", "findings", "sensitivity"}
    assert isinstance(r["summary"], str) and r["summary"]
    assert r["shape"] == {"rows": len(fw), "cols": fw.shape[1]}
    assert set(r["columns"]) == set(fw.columns)


def test_findings_sorted_by_significance_desc(fw):
    r = p2h.fit(fw).insights(sensitivity=1.0)
    sigs = [f["significance"] for f in r["findings"]]
    assert sigs == sorted(sigs, reverse=True)
    assert all(0 <= s <= 1 for s in sigs)


def test_sensitivity_monotonically_increases_findings(fw):
    v = p2h.fit(fw)
    counts = [len(v.insights(sensitivity=s)["findings"]) for s in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert counts == sorted(counts)


def test_sensitivity_out_of_range_raises(fw):
    v = p2h.fit(fw)
    with pytest.raises(ValueError, match="sensitivity"):
        v.insights(sensitivity=1.5)
    with pytest.raises(ValueError, match="sensitivity"):
        v.insights(sensitivity=-0.1)


def test_max_findings_cap(fw):
    r = p2h.fit(fw).insights(sensitivity=1.0, max_findings=3)
    assert len(r["findings"]) <= 3


def test_column_summary_numeric_has_distribution_and_moments(fw):
    r = p2h.fit(fw).insights()
    c = r["columns"]["bytes"]
    assert c["kind"] == "numeric"
    assert {"mean", "median", "std", "min", "max"} <= set(c)
    assert "distribution" in c and isinstance(c["distribution"], str)


def test_column_summary_categorical_has_top_value(fw):
    r = p2h.fit(fw).insights()
    c = r["columns"]["action"]
    assert c["kind"] == "categorical"
    assert "top_value" in c and "top_share" in c
    assert 0 <= c["top_share"] <= 1


def test_column_summary_datetime_has_range(fw):
    r = p2h.fit(fw).insights()
    c = r["columns"]["timestamp"]
    assert c["kind"] == "datetime"
    assert "min" in c and "max" in c


def test_skew_detects_right_skewed_column(fw):
    r = p2h.fit(fw).insights(sensitivity=1.0)
    skews = _kinds(r, "skew")
    assert any(f["columns"] == ["bytes"] for f in skews)


def test_concentration_detects_dominant_category(fw):
    r = p2h.fit(fw).insights(sensitivity=1.0)
    conc = _kinds(r, "concentration")
    assert any(f["columns"] == ["action"] for f in conc)


def test_correlation_detects_known_association(fw):
    # firewall_logs generates dst_port from protocol, so this pair should surface
    r = p2h.fit(fw).insights(sensitivity=1.0, max_pairs=10)
    corr = _kinds(r, "correlation")
    pairs = {tuple(sorted(f["columns"])) for f in corr}
    assert any({"dst_port", "protocol"} == set(p) for p in pairs)


def test_modality_detects_bimodal_numeric_column():
    df = pd.DataFrame({
        "x": np.concatenate([rng.normal(100, 5, 500), rng.normal(900, 5, 500)]),
        "y": range(1000),
    })
    v = p2h.fit(df, rows=["x"], cols=None, values=None, agg=None)
    r = v.insights(sensitivity=1.0)
    modality = _kinds(r, "modality")
    assert any(f["columns"] == ["x"] for f in modality)
    m = next(f for f in modality if f["columns"] == ["x"])
    assert len(m["detail"]["components"]) == 2


def test_anomalies_appear_for_2d_pivot(fw):
    r = p2h.fit(fw).insights(sensitivity=1.0)
    assert _kinds(r, "anomaly")  # firewall_logs has a real spike by construction


def test_constant_column_flagged_with_full_significance():
    df = pd.DataFrame({"a": [5] * 200, "b": range(200)})
    v = p2h.fit(df, rows=["a"], cols=None, values=None, agg=None)
    r = v.insights()
    const = _kinds(r, "constant")
    assert const and const[0]["significance"] == 1.0


def test_null_column_flagged_proportional_to_null_frac():
    df = pd.DataFrame({"a": [None] * 60 + list(range(40)), "b": range(100)})
    v = p2h.fit(df, rows=["b"], cols=None, values=None, agg=None)
    r = v.insights(sensitivity=1.0)
    nulls = _kinds(r, "nulls")
    a_null = next(f for f in nulls if f["columns"] == ["a"])
    assert abs(a_null["significance"] - 0.6) < 0.01


# --------------------------------------------------------------------------- regression: near-constant false positive


def test_skewed_but_genuinely_varying_column_not_flagged_constant():
    # a heavy-tailed (exponential-like) column has a huge max/min range, which made an
    # earlier std/range heuristic falsely call it "near constant" - the real bug found
    # against pivot2hist.sample.firewall_logs()'s `bytes` column.
    x = rng.exponential(1000, 4000)
    df = pd.DataFrame({"a": x, "b": range(4000)})
    v = p2h.fit(df, rows=["a"], cols=None, values=None, agg=None)
    r = v.insights(sensitivity=1.0)
    assert not _kinds(r, "constant")


def test_genuinely_near_constant_column_with_outliers_still_flagged():
    x = np.concatenate([np.full(3950, 100.0) + rng.normal(0, 0.5, 3950), rng.uniform(0, 1e6, 50)])
    df = pd.DataFrame({"a": x, "b": range(4000)})
    v = p2h.fit(df, rows=["a"], cols=None, values=None, agg=None)
    r = v.insights(sensitivity=1.0)
    const = _kinds(r, "constant")
    assert const and const[0]["significance"] > 0.5


def test_bytes_column_from_sample_not_falsely_flagged_constant(fw):
    r = p2h.fit(fw).insights(sensitivity=1.0)
    const = _kinds(r, "constant")
    assert not any(f["columns"] == ["bytes"] for f in const)


# --------------------------------------------------------------------------- top-level / robustness


def test_top_level_insights_matches_view_method(fw):
    a = p2h.insights(fw, rows=["action"], cols=["protocol"])
    b = p2h.fit(fw, rows=["action"], cols=["protocol"]).insights()
    assert a["shape"] == b["shape"]


def test_insights_reflects_current_slice(fw):
    full = p2h.fit(fw).insights()
    sliced = p2h.fit(fw).slice(action="deny").insights()
    assert sliced["shape"]["rows"] < full["shape"]["rows"]


@pytest.mark.parametrize("df_fn", [
    lambda: pd.DataFrame({"a": [[1, 2]] * 50 + [[3, 4]] * 50, "b": range(100)}),
    lambda: pd.DataFrame({"a": [None] * 100, "b": range(100)}),
    lambda: pd.DataFrame({"a": [1], "b": ["x"]}),
    lambda: pd.DataFrame([[1, 2]] * 50, columns=["a", "a"]),
    lambda: pd.DataFrame({"a": pd.array([1, 2, None, 4] * 25, dtype="Int64"), "b": range(100)}),
    lambda: pd.DataFrame({"a": [np.inf, -np.inf, 1.0, 2.0] * 25, "b": range(100)}),
    lambda: pd.DataFrame({"a": [f"id{i}" for i in range(200)], "b": range(200)}),
    lambda: pd.DataFrame({"a": [1] * 100, "b": [2] * 100}),
])
def test_insights_does_not_crash_on_adversarial_input(df_fn):
    p2h.fit(df_fn()).insights()
