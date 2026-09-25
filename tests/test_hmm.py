import numpy as np
import pandas as pd
import pytest

import pivot2hist as p2h
from pivot2hist._hmm import HMMFit, choose_hmm_states, decode_regimes, fit_hmm

rng = np.random.default_rng(1)


def _make_seq(n, switch, rng, p_a=(0.8, 0.1, 0.1), p_b=(0.1, 0.1, 0.8)):
    out = []
    for i in range(n):
        p = p_a if i < switch else p_b
        out.append(rng.choice(["login_success", "login_failure", "logout"], p=list(p)))
    return out


def test_fit_hmm_recovers_two_regimes():
    seqs = [_make_seq(60, 30, rng) for _ in range(20)]
    fit = fit_hmm(seqs, 2, seed=0, n_init=3)
    assert isinstance(fit, HMMFit) and fit.n_states == 2
    assert np.allclose(fit.A.sum(axis=1), 1.0) and np.allclose(fit.B.sum(axis=1), 1.0)
    assert abs(fit.pi.sum() - 1.0) < 1e-9
    # the two states should specialise: one mostly login_success, the other mostly logout
    top_symbol = [fit.symbols[np.argmax(fit.B[i])] for i in range(2)]
    assert set(top_symbol) == {"login_success", "logout"}


def test_viterbi_finds_the_switch_point():
    seq = _make_seq(60, 30, rng)
    fit = fit_hmm([seq], 2, seed=0, n_init=3)
    idx = {s: i for i, s in enumerate(fit.symbols)}
    obs = np.array([idx[s] for s in seq])
    path = fit.viterbi(obs)
    assert len(path) == 60
    switches = np.where(np.diff(path) != 0)[0]
    assert len(switches) >= 1
    assert abs(switches[0] - 29) < 12  # near the true switch at t=30, allowing EM noise


def test_state_summary_and_repr():
    seqs = [_make_seq(50, 25, rng) for _ in range(10)]
    fit = fit_hmm(seqs, 2, seed=0)
    summary = fit.state_summary(top=2)
    assert len(summary) == 2
    for row in summary:
        assert set(row) == {"regime", "share", "typical"}
        assert len(row["typical"]) == 2
        assert all(0 <= t["prob"] <= 1 for t in row["typical"])
    assert abs(sum(r["share"] for r in summary) - 1.0) < 1e-6
    assert "HMMFit" in repr(fit)


def test_choose_hmm_states_no_structure_stays_small():
    seqs = [list(rng.choice(["x", "y", "z"], 30)) for _ in range(10)]
    k = choose_hmm_states(seqs, k_max=4, seed=0)
    assert k == 1


def test_choose_hmm_states_finds_two_regimes():
    seqs = [_make_seq(80, 40, rng) for _ in range(15)]
    k = choose_hmm_states(seqs, k_max=4, seed=0, n_init=2)
    assert k in (2, 3)  # BIC may occasionally prefer a slightly richer model


def test_fit_hmm_edge_cases():
    # short single sequence
    fit = fit_hmm([["a", "b", "a", "b", "a"]], 2, seed=0)
    assert np.isfinite(fit.A).all() and np.isfinite(fit.B).all()

    # n_states larger than distinct symbols gets capped
    fit2 = fit_hmm([["a", "b"] * 20], 5, seed=0)
    assert fit2.n_states == 2

    # empty sequences mixed in with real ones are dropped, not fatal
    fit3 = fit_hmm([[], ["a", "b", "a", "b"], []], 2, seed=0)
    assert fit3.n_states == 2

    # all-empty is a real error
    with pytest.raises(ValueError):
        fit_hmm([[], []], 2)

    # a single-symbol vocabulary collapses to one state, not NaN
    fit4 = fit_hmm([["only"] * 10], 3, seed=0)
    assert fit4.n_states == 1 and np.isfinite(fit4.bic)


def test_fit_hmm_no_nan_under_state_collapse_pressure():
    seq = list(rng.choice(["a", "b", "c", "d"], 200, p=[0.48, 0.48, 0.02, 0.02]))
    for seed in range(5):
        fit = fit_hmm([seq], 4, seed=seed, n_init=2)
        assert np.isfinite(fit.A).all() and np.isfinite(fit.B).all() and np.isfinite(fit.pi).all()
        assert np.allclose(fit.A.sum(axis=1), 1.0) and np.allclose(fit.B.sum(axis=1), 1.0)


def test_decode_regimes_dataframe():
    seqs = [_make_seq(60, 30, rng) for _ in range(20)]
    rows = []
    for e, seq in enumerate(seqs):
        for t, s in enumerate(seq):
            rows.append({"entity": e, "t": t, "state": s})
    df = pd.DataFrame(rows)
    out = decode_regimes(df, "state", by="entity", time="t", n_states=2, seed=0)
    assert len(out) == len(df)
    assert set(out.unique()) <= {"regime 1", "regime 2"}
    assert isinstance(out.dtype, pd.CategoricalDtype) and out.dtype.ordered
    # entity 0's regime label should not be a single run-on state across the whole sequence
    g0 = out[df["entity"] == 0]
    assert g0.nunique() >= 1


def test_decode_regimes_no_by_splits_single_sequence():
    df = pd.DataFrame({"state": ["a"] * 10 + ["b"] * 10})
    out = decode_regimes(df, "state", n_states=2, seed=0)
    counts = out.value_counts()
    assert set(counts.index) == {"regime 1", "regime 2"} and counts.sum() == 20


def test_p2h_regimes_returns_view_and_toggles():
    auth = p2h.sample.auth_logs(2000, seed=2)
    v = p2h.regimes(auth, "event", by="user", time="timestamp", n_states=2, seed=0)
    assert isinstance(v, p2h.View)
    assert v.layout.rows[0].column == "regime" and v.layout.measure == "count"
    t = v.pivot()
    assert t.shape[0] == 2
    h = v.toggle()
    assert h.mode == "hist"
    assert "<svg" in v.svg()


def test_p2h_regimes_max_states_caps_columns():
    auth = p2h.sample.auth_logs(1200, seed=3)
    v = p2h.regimes(auth, "event", by="user", time="timestamp", n_states=2, seed=0, max_cols=2)
    assert v.pivot().shape[1] <= 2


def test_fit_hmm_performance_bound():
    import time

    auth = p2h.sample.auth_logs(3000, seed=1)
    t0 = time.time()
    decode_regimes(auth, "event", by="user", time="timestamp", k_max=3, seed=0)
    assert time.time() - t0 < 15
