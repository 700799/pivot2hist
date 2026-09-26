import numpy as np
import pandas as pd

import pivot2hist as p2h
from pivot2hist._profile import BOOLEAN, CATEGORICAL, CONSTANT, DATETIME, ID, NUMERIC


def test_kinds_on_firewall(fw):
    prof = p2h.profile(fw)
    assert prof["timestamp"].kind == DATETIME
    assert prof["src_ip"].kind == CATEGORICAL
    assert prof["dst_port"].kind == CATEGORICAL  # numeric but a label
    assert prof["action"].kind == CATEGORICAL
    assert prof["bytes"].kind == NUMERIC
    assert prof["duration"].kind == NUMERIC
    assert prof["severity"].kind == CATEGORICAL
    assert prof["bytes"].additive_hint
    assert prof["bytes"].is_integer and not prof["duration"].is_integer


def test_edge_kinds():
    n = 400
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "const": 1,
            "flag": rng.random(n) < 0.5,
            "zero_one": rng.integers(0, 2, n),
            "session_id": [f"s{i}" for i in range(n)],
            "small_int": rng.integers(1, 6, n),
            "big_float": rng.normal(size=n),
            "allnull": [None] * n,
            "text_id": [f"{i:05x}" for i in range(n)],
            "cat": pd.Categorical(rng.choice(list("abc"), n)),
        }
    )
    prof = p2h.profile(df)
    assert prof["const"].kind == CONSTANT
    assert prof["flag"].kind == BOOLEAN
    assert prof["zero_one"].kind == BOOLEAN
    assert prof["session_id"].kind == ID
    assert prof["small_int"].kind == CATEGORICAL
    assert prof["big_float"].kind == NUMERIC
    assert prof["allnull"].kind == CONSTANT
    assert prof["text_id"].kind == ID
    assert prof["cat"].kind == CATEGORICAL
    assert prof.summary().shape[0] == 9
    assert prof["allnull"].null_frac == 1.0


def test_high_cardinality_but_repetitive_is_categorical():
    vals = np.repeat([f"10.0.0.{i}" for i in range(200)], 10)
    prof = p2h.profile(pd.DataFrame({"src_ip": vals}))
    assert prof["src_ip"].kind == CATEGORICAL
    assert prof["src_ip"].entity_hint


def test_semantic_and_time_series_fields(fw):
    prof = p2h.profile(fw)
    assert prof["src_ip"].semantic == "ipv4" and prof["src_ip"].hierarchy == ("/8", "/16", "/24", "host")
    assert prof["dst_port"].semantic == "port" and prof["dst_port"].hierarchy == ("class", "port")
    assert prof["action"].semantic is None and prof["action"].hierarchy == ()
    assert prof["src_ip"].entity_hint
    assert prof.time_column == "timestamp" and not prof.is_time_series
    assert prof["timestamp"].ts_freq is not None
    assert "semantic" in prof.summary().columns
    ts = pd.DataFrame({"t": pd.date_range("2026-01-01", periods=300, freq="5min"), "v": np.arange(300.0)})
    pt = p2h.profile(ts)
    assert pt.is_time_series and pt["t"].ts_freq == "5min" and pt["t"].ts_regularity == 1.0 and pt["t"].ts_sorted
    irregular = pd.DataFrame({"t": [pd.Timestamp(x) for x in ("2026-01-01", "2026-01-01 00:00:07", "2026-01-03", "2026-02-01")], "v": [1, 2, 3, 4]})
    assert not p2h.profile(irregular).is_time_series
    assert p2h.profile(pd.DataFrame({"x": [1, 2]})).time_column is None
