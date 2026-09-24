import time

import numpy as np
import pandas as pd
import pytest

import pivot2hist as p2h
from pivot2hist._chains import sequences, steady_state, transition_matrix, transitions
from pivot2hist._log import Log


@pytest.fixture(scope="module")
def auth():
    return p2h.sample.auth_logs(2000, seed=2)


def test_transitions_long_and_matrix(auth):
    t = transitions(auth, "event", by="user", time="timestamp")
    assert set(t.columns) == {"from", "to", "count", "prob"}
    assert np.allclose(t.groupby("from")["prob"].sum(), 1.0)
    n_pairs = sum(max(0, n - 1) for n in auth.groupby("user").size())
    assert t["count"].sum() == n_pairs  # no transition crosses a user
    m = transition_matrix(auth, "event", by="user", time="timestamp")
    assert m.shape == (3, 3) and m.to_numpy().sum() == n_pairs
    p = transition_matrix(auth, "event", by="user", time="timestamp", normalize=True)
    assert np.allclose(p.sum(axis=1), 1.0)
    t2 = transitions(auth, "event", by="user", time="timestamp", order=2)
    assert t2["from"].str.contains("→").all()
    plain = transitions(pd.DataFrame({"s": list("aabab")}), "s")
    assert plain.set_index(["from", "to"])["count"].to_dict() == {("a", "a"): 1, ("a", "b"): 2, ("b", "a"): 1}
    assert transitions(pd.DataFrame({"s": ["a"]}), "s").empty
    ss = steady_state(m)
    assert abs(ss.sum() - 1) < 1e-9 and ss.index[0] == "login_success"


def test_sequences(auth):
    s = sequences(auth, "event", by="user", time="timestamp", length=3, n=5)
    assert len(s) == 5 and list(s.columns) == ["chain", "count", "share", "groups"]
    assert s["chain"].str.count("→").eq(2).all() and s["count"].is_monotonic_decreasing
    assert (s["groups"] <= auth["user"].nunique()).all()
    assert sequences(pd.DataFrame({"s": ["a", "b"]}), "s", length=3).empty


def test_chains_view(auth):
    v = p2h.chains(auth, "event", by="user", time="timestamp")
    assert v.layout.rows[0].column == "from" and v.layout.cols[0].column == "to" and v.layout.measure == "sum(count)"
    assert v.pivot().shape == (3, 3) and "<table" in v.html() and v.toggle().mode == "hist"
    n = p2h.chains(auth, "event", by="user", time="timestamp", normalize=True)
    assert np.allclose(n.pivot().sum(axis=1), 1.0) and n.display["heat"] == "row"
    assert v.cocluster(2).pivot().shape[0] == 3
    with pytest.raises(ValueError):
        p2h.chains(pd.DataFrame({"s": ["a"]}), "s")


def test_log_steps_and_stats():
    lg = Log()
    seen = []
    off = lg.listen(seen.append)
    with lg.step("outer", "x") as s:
        s.detail += "!"
        with lg.step("inner"):
            time.sleep(0.01)
    lg.info("note", "hello")
    assert [e.step for e in lg.entries] == ["inner", "outer", "note"]
    assert lg.entries[1].detail == "x!" and lg.entries[0].depth == 1 and lg.entries[1].seconds >= 0.01
    assert len(seen) == 3 and "outer" in lg.lines(2)[0]
    st = lg.stats(2)
    assert list(st["step"]) == ["outer", "inner"] and st.iloc[0]["seconds"] >= st.iloc[1]["seconds"]
    off()
    lg.info("after")
    assert len(seen) == 3
    assert "outer" in repr(lg)
    empty = Log()
    assert empty.stats().empty and repr(empty) == "<Log: empty>"


def test_global_log_records_pipeline(capsys):
    p2h.log.clear()
    p2h.verbose()
    try:
        v = p2h.fit(p2h.sample.firewall_logs(500))
        v.pivot()
        v.html()
        v.toggle().render()
    finally:
        p2h.verbose(False)
    steps = {e.step for e in p2h.log.entries}
    assert {"fit", "pivot", "render", "bins"} <= steps
    st = p2h.stats(7)
    assert len(st) <= 7 and "fit" in set(st["step"]) and v.stats(3).shape[0] <= 3
    err = capsys.readouterr().err
    assert "fit" in err and "pivot" in err
