"""Many data shapes, one contract: fit never crashes, fits the box, toggles round-trip,
slices, renders as text/HTML/SVG and exports JSON."""
import json
import xml.dom.minidom

import numpy as np
import pandas as pd
import pytest

import pivot2hist as p2h

rng = np.random.default_rng(42)
N = 600


def _dates(n=N, freq="7min", start="2026-02-01"):
    return pd.date_range(start, periods=n, freq=freq)


SCENARIOS = {
    "all_numeric": lambda: pd.DataFrame({f"x{i}": rng.normal(i, 1 + i, N) for i in range(5)}),
    "all_categorical": lambda: pd.DataFrame({c: rng.choice(list("abcdef")[: k + 2], N) for k, c in enumerate("pqrs")}),
    "single_row": lambda: pd.DataFrame({"a": ["x"], "b": [1], "t": [pd.Timestamp("2026-01-01")]}),
    "two_rows": lambda: pd.DataFrame({"a": ["x", "y"], "b": [1, 2]}),
    "one_categorical_column": lambda: pd.DataFrame({"kind": rng.choice(["a", "b", "c"], N)}),
    "one_datetime_column": lambda: pd.DataFrame({"when": _dates()}),
    "one_numeric_column": lambda: pd.DataFrame({"v": rng.exponential(5, N)}),
    "wide_mixed": lambda: pd.DataFrame({**{f"n{i}": rng.normal(size=N) for i in range(30)}, **{f"c{i}": rng.choice(list("xyz"), N) for i in range(30)}}),
    "nulls_heavy": lambda: pd.DataFrame({"a": rng.choice(["x", "y", None], N), "b": np.where(rng.random(N) < 0.5, np.nan, rng.normal(size=N)), "z": [None] * N}),
    "constants_plus_one": lambda: pd.DataFrame({"k": 1, "s": "same", "v": rng.choice(["a", "b"], N)}),
    "ids_only": lambda: pd.DataFrame({"session_id": [f"s{i}" for i in range(N)], "uuid": [f"{i:08x}-0000-4000-8000-{i:012x}" for i in range(N)]}),
    "booleans_only": lambda: pd.DataFrame({"a": rng.random(N) < 0.5, "b": rng.random(N) < 0.2}),
    "tz_aware": lambda: pd.DataFrame({"t": _dates().tz_localize("UTC"), "host": rng.choice(list("ab"), N), "n": rng.integers(0, 9, N)}),
    "epoch_millis": lambda: pd.DataFrame({"ts": (1_700_000_000 + np.arange(N) * 300) * 1000, "ev": rng.choice(["login", "logout"], N)}),
    "iso_strings": lambda: pd.DataFrame({"time": _dates().strftime("%Y-%m-%dT%H:%M:%S").tolist(), "ev": rng.choice(list("abc"), N)}),
    "numeric_and_bool_strings": lambda: pd.DataFrame({"n": rng.integers(0, 100, N).astype(str), "ok": rng.choice(["yes", "no"], N), "k": rng.choice(list("ab"), N)}),
    "mixed_object_column": lambda: pd.DataFrame({"m": [1, "a", 2.5, None, "b"] * (N // 5), "k": rng.choice(list("xy"), N)}),
    "unicode_labels": lambda: pd.DataFrame({"país": rng.choice(["España", "日本", "🇩🇪", "Ünïcødé"], N), "α": rng.normal(size=N)}),
    "negative_measure": lambda: pd.DataFrame({"account": rng.choice(list("abcd"), N), "amount": rng.normal(0, 100, N), "day": _dates(freq="1D").repeat(1)[:N] if N <= 1 else _dates(N, "h")}),
    "heavy_tail": lambda: pd.DataFrame({"bytes": np.exp(rng.normal(8, 3, N)).round(), "proto": rng.choice(["tcp", "udp"], N)}),
    "regular_time_series": lambda: pd.DataFrame({"t": _dates(freq="5min"), "cpu": rng.random(N), "mem": rng.random(N) * 100, "host": rng.choice(list("abc"), N)}),
    "irregular_event_log": lambda: pd.DataFrame({"t": pd.Timestamp("2026-01-01") + pd.to_timedelta(np.sort(rng.exponential(600, N)).cumsum(), unit="s"), "src": rng.choice([f"10.0.0.{i}" for i in range(40)], N), "action": rng.choice(["allow", "deny"], N, p=[0.9, 0.1])}),
    "duplicate_columns": lambda: pd.DataFrame(np.column_stack([rng.choice(list("ab"), N), rng.choice(list("cd"), N), rng.integers(0, 5, N)]), columns=["a", "a", "b"]),
    "datetime_index": lambda: pd.DataFrame({"v": rng.normal(size=N), "g": rng.choice(list("ab"), N)}, index=_dates()),
    "categorical_dtype": lambda: pd.DataFrame({"c": pd.Categorical(rng.choice(list("abc"), N)), "d": pd.Categorical(rng.choice(list("xy"), N), ordered=True), "n": rng.integers(0, 3, N)}),
    "nullable_dtypes": lambda: pd.DataFrame({"i": pd.array(np.where(rng.random(N) < 0.1, None, rng.integers(0, 50, N)), dtype="Int64"), "b": pd.array(rng.choice([True, False, None], N), dtype="boolean"), "s": pd.array(rng.choice(["p", "q", None], N), dtype="string")}),
    "long_strings": lambda: pd.DataFrame({"msg": rng.choice(["x" * 300, "y" * 250, "z" * 200], N), "k": rng.choice(list("ab"), N)}),
    "cyber_dns": lambda: pd.DataFrame({"ts": _dates(freq="2s"), "client": rng.choice([f"192.168.1.{i}" for i in range(1, 60)], N), "qname": rng.choice(["www.example.com", "api.corp.local", "cdn.evil.biz", "mail.google.co.uk"], N), "qtype": rng.choice(["A", "AAAA", "TXT"], N), "rcode": rng.choice([0, 3], N, p=[0.9, 0.1]), "url": rng.choice(["https://a.com/x", "https://b.org/y/z", "http://c.net/"], N), "user": rng.choice([f"u{i}@corp.local" for i in range(8)], N)}),
    "high_cardinality_text": lambda: pd.DataFrame({"path": [f"/var/app/{i % 700}/{i}" for i in range(N)], "k": rng.choice(list("abc"), N)}),
    "records_missing_keys": lambda: p2h.load([{"a": 1, "b": "x"}, {"a": 2}, {"b": "y", "c": 3.0}] * (N // 3)),
    "wide_short": lambda: pd.DataFrame({f"c{i}": rng.choice(list("ab"), 50) for i in range(150)}),
    "inf_values": lambda: pd.DataFrame({"v": np.where(rng.random(N) < 0.05, np.inf, rng.normal(size=N)), "k": rng.choice(list("ab"), N)}),
    "big_ints": lambda: pd.DataFrame({"big": rng.integers(10**15, 10**18, N), "k": rng.choice(list("abc"), N)}),
    "bool_and_numeric": lambda: pd.DataFrame({"flag": rng.random(N) < 0.3, "v": rng.normal(size=N)}),
    "named_index": lambda: pd.DataFrame({"a": rng.choice(list("ab"), N), "v": rng.normal(size=N)}, index=pd.Index(rng.permutation(N) + 1000, name="rowid")),
    "ports_and_ips": lambda: pd.DataFrame({"src_ip": rng.choice([f"10.{i}.{j}.{k}" for i in range(3) for j in range(4) for k in range(1, 6)], N), "dst_port": rng.choice([22, 80, 443, 3389, 8080, 50000, 60123], N), "bytes": rng.integers(0, 10**6, N)}),
    "single_value_numeric": lambda: pd.DataFrame({"v": np.ones(N), "k": rng.choice(list("ab"), N)}),
    "timedelta_column": lambda: pd.DataFrame({"dur": pd.to_timedelta(rng.exponential(30, N), unit="s"), "k": rng.choice(list("abc"), N)}),
    "period_column": lambda: pd.DataFrame({"m": pd.period_range("2024-01", periods=N, freq="M"), "v": rng.normal(size=N)}),
}


def _xml(s: str) -> None:
    xml.dom.minidom.parseString(s)


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_scenario(name):
    df = SCENARIOS[name]()
    box = (12, 5)
    v = p2h.fit(df, max_rows=box[0], max_cols=box[1])
    t = v.pivot()
    assert t.shape[0] <= box[0] and t.shape[1] <= box[1], (name, t.shape, v.layout)
    assert t.shape[0] >= 1
    # text, html, svg
    txt = v.render(width=100)
    assert txt.startswith("pivot") and v.title()
    _xml(v.html())
    h = v.toggle()
    assert h.mode == "hist" and h.toggle().layout == v.layout
    svg = h.html()
    _xml(svg[svg.index("<svg"):svg.index("</svg>") + 6]) if "<svg" in svg else None
    assert h.render(width=100)
    # slicing on the first dimension's first observed value
    d = v.layout.rows[0]
    col = d.column
    data = v.data
    if col in data.columns and data[col].notna().any():
        val = data[col].dropna().iloc[0]
        if not isinstance(val, (list, dict)):
            s = v.slice(**{col: val})
            assert len(s.data) >= 1 and s.slices
            assert s.toggle().mode == "hist"
    # export, suggestions, profile
    json.loads(v.to_json())
    assert v.suggest(3)
    assert len(v.profile) == len(v.data.columns)
    assert v.profile.summary().shape[0] == len(v.data.columns)


@pytest.mark.parametrize("name", ["cyber_dns", "ports_and_ips", "regular_time_series", "heavy_tail", "all_numeric"])
def test_scenario_extras(name):
    df = SCENARIOS[name]()
    v = p2h.fit(df, max_rows=12, max_cols=5)
    if v.pivot().shape[0] >= 3:
        c = v.cluster(2)
        assert c.pivot().shape[0] <= 12 and c.layout.rows[0].column == "cluster"
    numeric = [c.name for c in v.profile if c.kind == "numeric"]
    if numeric:
        hb = v.histogram(numeric[0], bins=6)
        assert len(hb.bins()) <= 6 and "<svg" in hb.html()
        assert len(v.histogram(numeric[0], bins="kmeans").bins()) >= 1
        assert len(v.histogram(numeric[0], bins="quantile").bins()) >= 1
    for cp in v.profile:
        if cp.hierarchy and any(d.column == cp.name for d in v.layout.dims):
            assert v.coarser(cp.name).pivot().shape[0] <= 12
    assert v.sample(50).pivot().shape[0] <= 12


def test_scenario_registry_covers_semantic_and_time_series():
    dns = p2h.profile(SCENARIOS["cyber_dns"]())
    assert dns["client"].semantic == "ipv4" and dns["url"].semantic == "url"
    assert dns["user"].semantic == "email" and dns["qname"].semantic == "domain"
    assert dns.is_time_series
    assert p2h.profile(SCENARIOS["regular_time_series"]()).is_time_series
    assert not p2h.profile(SCENARIOS["irregular_event_log"]()).is_time_series
    assert p2h.profile(SCENARIOS["ports_and_ips"]())["dst_port"].semantic == "port"


def test_large_frame_end_to_end():
    df = p2h.sample.firewall_logs(150_000, seed=9)
    v = p2h.fit(df)
    assert v.pivot().shape[0] <= 40
    assert v.toggle().bins().shape[0] <= 40
    assert v.slice(action="deny").histogram("bytes").bins()["count"].sum() == (df["action"] == "deny").sum()
    assert v.cluster(3, collapse=True).pivot().shape[0] == 3
