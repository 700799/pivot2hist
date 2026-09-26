import numpy as np
import pandas as pd
import pytest

import pivot2hist as p2h

duckdb = pytest.importorskip("duckdb")


def _destringify(t: pd.DataFrame) -> pd.DataFrame:
    """Normalize a pivot's index/columns to plain sorted strings, so two tables built
    with different (but equally valid) category orderings compare cell-for-cell."""
    t = t.copy()
    if isinstance(t.index, pd.MultiIndex):
        t.index = pd.MultiIndex.from_tuples([tuple(map(str, x)) for x in t.index], names=t.index.names)
    else:
        t.index = t.index.astype(str)
    if isinstance(t.columns, pd.MultiIndex):
        t.columns = pd.MultiIndex.from_tuples([tuple(map(str, x)) for x in t.columns], names=t.columns.names)
    else:
        t.columns = t.columns.astype(str)
    return t.sort_index().sort_index(axis=1)


def assert_same(v_pd, v_dd):
    a, b = _destringify(v_pd.pivot()), _destringify(v_dd.pivot())
    assert a.shape == b.shape, (a.shape, b.shape)
    assert np.allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float), equal_nan=True)


@pytest.fixture(scope="module")
def auth():
    return p2h.sample.auth_logs(4000, seed=1)


@pytest.fixture(scope="module")
def fw():
    return p2h.sample.firewall_logs(4000, seed=2)


@pytest.mark.parametrize("agg", ["sum", "mean", "min", "max", "median", "std", "nunique"])
def test_agg_matches_pandas(auth, agg):
    v1 = p2h.fit(auth, rows=["host"], cols=["event"], values="latency_ms", agg=agg)
    v2 = p2h.fit(auth, rows=["host"], cols=["event"], values="latency_ms", agg=agg, engine="duckdb")
    assert_same(v1, v2)


def test_count_matches_pandas(auth):
    v1 = p2h.fit(auth, rows=["host"], cols=["event"], values=None, agg=None)
    v2 = p2h.fit(auth, rows=["host"], cols=["event"], values=None, agg=None, engine="duckdb")
    assert_same(v1, v2)
    # explicit agg="count" on a real value column is a different code path than the
    # values=None row-count convention - both must reshape correctly
    v3 = p2h.fit(auth, rows=["host"], cols=["event"], values="latency_ms", agg="count")
    v4 = p2h.fit(auth, rows=["host"], cols=["event"], values="latency_ms", agg="count", engine="duckdb")
    assert_same(v3, v4)


def test_binned_dim_matches_pandas(auth, fw):
    v1 = p2h.fit(auth, rows=["latency_ms"], cols=["event"], values=None, agg=None)
    v2 = p2h.fit(auth, rows=["latency_ms"], cols=["event"], values=None, agg=None, engine="duckdb")
    assert_same(v1, v2)
    v3 = p2h.fit(fw, rows=["bytes"], cols=["duration"], values=None, agg=None)
    v4 = p2h.fit(fw, rows=["bytes"], cols=["duration"], values=None, agg=None, engine="duckdb")
    assert_same(v3, v4)


@pytest.mark.parametrize("max_rows", [8, 40])
def test_time_dim_matches_pandas(auth, max_rows):
    v1 = p2h.fit(auth, rows=["timestamp"], cols=["event"], values=None, agg=None, max_rows=max_rows)
    v2 = p2h.fit(auth, rows=["timestamp"], cols=["event"], values=None, agg=None, max_rows=max_rows, engine="duckdb")
    assert_same(v1, v2)


def test_histogram_observed_false_matches_pandas(auth):
    h1 = p2h.histogram(auth, on="latency_ms")
    h2 = p2h.fit(auth, rows=["latency_ms"], values=None, agg=None, engine="duckdb").histogram("latency_ms")
    b1, b2 = h1.bins(), h2.bins()
    assert b1.shape == b2.shape
    assert np.allclose(b1.to_numpy(dtype=float), b2.to_numpy(dtype=float))


def test_top_n_folding_matches_pandas(auth):
    v1 = p2h.fit(auth, rows=["user"], cols=["event"], values=None, agg=None, max_rows=10)
    v2 = p2h.fit(auth, rows=["user"], cols=["event"], values=None, agg=None, max_rows=10, engine="duckdb")
    assert_same(v1, v2)
    assert "(other)" in v2.pivot().index.astype(str).tolist()


def test_stacked_dims_matches_pandas(auth):
    v1 = p2h.fit(auth, rows=["host", "event"], cols=["method"], values=None, agg=None)
    v2 = p2h.fit(auth, rows=["host", "event"], cols=["method"], values=None, agg=None, engine="duckdb")
    assert_same(v1, v2)


def test_slice_after_duckdb_fit_matches_pandas(auth):
    v1 = p2h.fit(auth, rows=["host"], cols=["event"], values=None, agg=None).slice(method="sso")
    v2 = p2h.fit(auth, rows=["host"], cols=["event"], values=None, agg=None, engine="duckdb").slice(method="sso")
    assert_same(v1, v2)


def test_semantic_level_falls_back_to_pandas(fw):
    v1 = p2h.fit(fw, rows=[{"column": "src_ip", "level": "/24"}], cols=["action"], values=None, agg=None)
    v2 = p2h.fit(fw, rows=[{"column": "src_ip", "level": "/24"}], cols=["action"], values=None, agg=None, engine="duckdb")
    assert v1.pivot().equals(v2.pivot())


def test_weekly_freq_falls_back_to_pandas(auth):
    v1 = p2h.fit(auth, rows=[{"column": "timestamp", "freq": "W"}], cols=["event"], values=None, agg=None)
    v2 = p2h.fit(auth, rows=[{"column": "timestamp", "freq": "W"}], cols=["event"], values=None, agg=None, engine="duckdb")
    assert v1.pivot().equals(v2.pivot())


def test_unknown_engine_raises(auth):
    with pytest.raises(ValueError):
        p2h.fit(auth, engine="bogus").pivot()


def test_connection_cache_reused_for_the_same_frame(auth):
    from pivot2hist import _duckdb_engine as ddb

    before = len(ddb._CONN_CACHE)
    df = auth.copy()
    v1 = p2h.fit(df, rows=["host"], cols=["event"], values=None, agg=None, engine="duckdb")
    v1.pivot()
    assert len(ddb._CONN_CACHE) == before + 1
    v2 = p2h.fit(df, rows=["user"], cols=["method"], values=None, agg=None, engine="duckdb")
    v2.pivot()
    assert len(ddb._CONN_CACHE) == before + 1  # same frame, same cached connection


def test_connection_cache_is_bounded_lru(auth):
    # DuckDB's register() keeps its own strong reference to a registered frame, so the
    # cache can't rely on garbage collection to free entries (see _connection_for's
    # docstring) - it must evict on its own once the cache is full.
    from pivot2hist import _duckdb_engine as ddb

    frames = [auth.sample(200, random_state=i).copy() for i in range(ddb._CONN_CACHE_MAX + 3)]
    for f in frames:
        p2h.fit(f, rows=["host"], cols=["event"], values=None, agg=None, engine="duckdb").pivot()
    assert len(ddb._CONN_CACHE) <= ddb._CONN_CACHE_MAX
    # the connections for the earliest, evicted frames must have been closed, not leaked
    live_keys = set(ddb._CONN_CACHE)
    assert id(frames[0]) not in live_keys
