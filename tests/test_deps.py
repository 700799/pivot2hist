import numpy as np
import pandas as pd
import pytest

import pivot2hist as p2h
from pivot2hist._deps import dependency_pairs, mutual_info_matrix

rng = np.random.default_rng(3)


@pytest.fixture(scope="module")
def correlated():
    n = 4000
    x = rng.normal(0, 1, n)
    y = x * 2 + rng.normal(0, 0.1, n)
    z = rng.normal(0, 1, n)
    cat = np.where(x > 0, "hi", "lo")
    return pd.DataFrame({"x": x, "y": y, "z": z, "cat": cat})


def test_matrix_is_symmetric_with_unit_diagonal(correlated):
    m = mutual_info_matrix(correlated)
    assert list(m.index) == list(m.columns)
    assert np.allclose(np.diag(m.to_numpy()), 1.0)
    assert np.allclose(m.to_numpy(), m.to_numpy().T)
    assert ((m.to_numpy() >= 0) & (m.to_numpy() <= 1.000001)).all()


def test_matrix_recovers_known_dependencies(correlated):
    m = mutual_info_matrix(correlated)
    assert m.loc["x", "y"] > 0.5  # tightly correlated
    assert m.loc["x", "z"] < 0.1  # independent
    assert m.loc["x", "cat"] > 0.5  # cat is literally derived from x


def test_constant_and_id_excluded_by_default():
    n = 200
    df = pd.DataFrame({
        "a": rng.integers(0, 5, n),
        "b": rng.integers(0, 5, n),
        "const": np.ones(n),
        "uid": [f"id{i}" for i in range(n)],
    })
    m = mutual_info_matrix(df)
    assert set(m.columns) == {"a", "b"}


def test_explicit_columns_overrides_exclusion():
    n = 200
    df = pd.DataFrame({"a": rng.integers(0, 5, n), "const": np.ones(n)})
    m = mutual_info_matrix(df, columns=["a", "const"])
    assert set(m.columns) == {"a", "const"}
    assert m.loc["a", "const"] == 0.0  # zero entropy column carries no information


def test_too_few_usable_columns_raises():
    with pytest.raises(ValueError):
        mutual_info_matrix(pd.DataFrame({"a": [1] * 10, "const": [1] * 10}))


def test_max_cols_caps_wide_frames():
    wide = pd.DataFrame({f"c{i}": rng.integers(0, 5, 300) for i in range(50)})
    m = mutual_info_matrix(wide, max_cols=8)
    assert m.shape == (8, 8)


def test_dependency_pairs_excludes_self_pairs(correlated):
    long = dependency_pairs(correlated)
    assert (long["column_a"] != long["column_b"]).all()
    assert set(long.columns) == {"column_a", "column_b", "association"}
    n = correlated.shape[1] if "cat" not in correlated else len(mutual_info_matrix(correlated).columns)
    assert len(long) == n * (n - 1)


def test_p2h_dependencies_returns_view_and_toggles():
    auth = p2h.sample.auth_logs(3000, seed=1)
    v = p2h.dependencies(auth)
    assert isinstance(v, p2h.View)
    t = v.pivot()
    assert t.shape[0] == t.shape[1]
    # the synthetic auth_logs mfa rate depends on method (sso vs others)
    assert t.loc["method", "mfa"] > 0.05
    h = v.toggle()
    assert h.mode == "hist"
    assert "<svg" in v.svg()


def test_p2h_dependencies_columns_arg():
    auth = p2h.sample.auth_logs(1000, seed=1)
    v = p2h.dependencies(auth, columns=["method", "mfa", "event"])
    assert set(v.pivot().index) == {"method", "mfa", "event"}
