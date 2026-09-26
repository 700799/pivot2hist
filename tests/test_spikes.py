import json

import numpy as np
import pandas as pd
import pytest

import pivot2hist as p2h
from pivot2hist import agent
from pivot2hist._spikes import COLUMNS, describe_spike, phase_keys


@pytest.fixture(scope="module")
def bursty(fw):
    """The firewall sample with a burst of 3389 denies on day 4 (14:00-15:00) and port 445
    going silent from day 5 on."""
    burst = fw.sample(200, random_state=1).copy()
    burst["timestamp"] = pd.Timestamp("2026-03-04 14:00") + pd.to_timedelta(np.random.default_rng(0).uniform(0, 3600, 200), unit="s")
    burst["dst_port"] = 3389
    burst["action"] = "deny"
    df = pd.concat([fw, burst], ignore_index=True)
    return df[~((df["dst_port"] == 445) & (df["timestamp"] >= "2026-03-05"))].reset_index(drop=True)


@pytest.fixture(scope="module")
def v(bursty):
    return p2h.fit(bursty, rows=["dst_port"], cols=["action"], agg="count")


# --------------------------------------------------------------------------- detection


def test_injected_burst_is_the_top_spike(v):
    s = v.spikes("timestamp")
    assert list(s.columns) == COLUMNS
    top = s.iloc[0]
    assert top["row"] == "3389" and top["kind"] == "spike" and top["bucket"].startswith("2026-03-04")
    assert top["support"] == pytest.approx(200, abs=15)  # the burst, plus the few organic 3389 events in that bucket
    assert top["ratio"] > 5 and top["score"] > 10
    assert top["baseline"] == "seasonal"  # 7 days of 6-hour buckets: three-plus periods to learn the daily cycle


def test_silenced_port_is_a_shift_down(v):
    s = v.spikes("timestamp", n=20)
    hit = s[(s["row"] == "445") & (s["kind"] == "shift down")]
    assert len(hit) == 1
    r = hit.iloc[0]
    assert r["observed"] < r["expected"] and r["ratio"] < 0.7 and r["score"] < -3
    assert r["support"] == r["observed"]  # count measure: what was observed since the step is the rows behind it


def test_seasonal_baseline_does_not_flag_the_daily_peak(fw):
    # untouched data: the daytime peak recurs every day, so nothing should read as ×10+
    s = p2h.fit(fw, rows=["dst_port"], cols=["action"], agg="count").spikes("timestamp", n=50)
    assert s.empty or s["ratio"].max() < 10
    assert (s["score"].abs() >= 3).all()


def test_median_baseline_by_contrast_flags_the_peak(v):
    s = v.spikes("timestamp", n=50, baseline="median")
    assert (s["baseline"] == "median").all()
    assert len(s) > len(v.spikes("timestamp", n=50))  # the daily cycle now counts as movement


def test_share_baseline_on_a_single_day(bursty):
    day = bursty[(bursty["timestamp"] >= "2026-03-04") & (bursty["timestamp"] < "2026-03-05")]
    s = p2h.fit(day, rows=["dst_port"], agg="count").spikes("timestamp")
    assert not s.empty
    top = s.iloc[0]
    assert top["baseline"] == "share" and top["row"] == "3389" and top["kind"] == "spike" and "14:00" in top["bucket"]


def test_few_rows_fall_back_to_median(bursty):
    day = bursty[(bursty["timestamp"] >= "2026-03-04") & (bursty["timestamp"] < "2026-03-05")]
    s = p2h.fit(day, rows=["action"], agg="count").spikes("timestamp")
    assert not s.empty and (s["baseline"] == "median").all()
    assert not ((s["kind"] == "drop") & (s["row"] == "allow")).any()  # no mirror-image "drop" from share arithmetic


def test_share_needs_additive_measure(fw):
    with pytest.raises(ValueError, match="additive"):
        p2h.fit(fw, rows=["dst_port"], values="bytes", agg="mean").spikes("timestamp", baseline="share")


def test_bad_arguments(v, fw):
    with pytest.raises(ValueError, match="baseline"):
        v.spikes("timestamp", baseline="magic")
    with pytest.raises(ValueError, match="positive"):
        v.spikes("timestamp", z=0)
    with pytest.raises(KeyError):
        v.spikes("nope")
    with pytest.raises(ValueError, match="row axis"):
        p2h.fit(fw, rows=["timestamp"], cols=["action"]).spikes("timestamp")
    with pytest.raises(ValueError, match="no datetime column"):
        p2h.fit(fw.drop(columns=["timestamp"]), rows=["dst_port"]).spikes()
    with pytest.raises(ValueError, match="no usable datetime"):
        v.spikes("protocol")


def test_default_column_is_the_first_datetime_off_the_rows(v, fw):
    assert v.spikes().equals(v.spikes("timestamp"))
    # time on the column axis is fine: the column axis is collapsed anyway
    s = p2h.fit(fw, rows=["dst_port"], cols=["timestamp"], agg="count").spikes()
    assert list(s.columns) == COLUMNS


def test_heavy_tailed_sums_are_scored_on_log_scale(bursty):
    s = p2h.fit(bursty, rows=["dst_port"], cols=["action"], values="bytes", agg="sum").spikes("timestamp")
    assert not s.empty and s["baseline"].str.endswith("(log)").all()
    assert ((s["row"] == "3389") & (s["kind"] == "spike")).any()


def test_flat_busy_row_is_not_a_spike():
    # 1000 +- a little per bucket, one bucket at 1150: loud in Poisson terms, flat in practice
    rng = np.random.default_rng(3)
    t0 = pd.Timestamp("2026-01-01")
    counts = np.full(28, 1000) + rng.integers(-20, 20, 28)
    counts[10] = 1150
    ts = np.concatenate([np.full(c, t0 + pd.Timedelta(hours=6 * i)) for i, c in enumerate(counts)])
    df = pd.DataFrame({"t": ts, "k": "a"})
    s = p2h.fit(df, rows=["k"], agg="count").spikes("t", baseline="median")
    assert s.empty


def test_shifts_can_be_switched_off(v):
    s = v.spikes("timestamp", n=50, shifts=False)
    assert set(s["kind"]) <= {"spike", "drop"}


def test_hist_and_one_d_layouts(bursty):
    h = p2h.fit(bursty).histogram("bytes").spikes(n=3)
    assert list(h.columns) == COLUMNS
    one = p2h.fit(bursty, rows=["dst_port"], cols=[], agg="count").spikes(n=3)
    assert one.iloc[0]["row"] == "3389"


def test_top_level_function(bursty):
    s = p2h.spikes(bursty, rows=["dst_port"], cols=["action"], agg="count", n=1)
    assert s.iloc[0]["row"] == "3389"
    assert "auto" in p2h.SPIKE_BASELINES


def test_phase_keys():
    days = [pd.Timestamp("2026-03-01") + pd.Timedelta(hours=6 * i) for i in range(8)]
    phase, period = phase_keys(days, "6h")
    assert list(phase) == [0, 360, 720, 1080] * 2 and len(set(period)) == 2
    phase, period = phase_keys([pd.Timestamp("2026-03-02") + pd.Timedelta(days=i) for i in range(14)], "D")
    assert list(phase) == list(range(7)) * 2 and len(set(period)) == 2
    phase, period = phase_keys([pd.Timestamp(f"2025-{m:02d}-01") for m in range(1, 13)], "M")
    assert list(phase) == list(range(1, 13)) and len(set(period)) == 1
    assert phase_keys([pd.Timestamp("2026-01-05")], "W") == (None, None)


def test_describe_spike_wording(v):
    s = v.spikes("timestamp", n=20)
    for r in s.itertuples():
        text = describe_spike(r, "count", "timestamp")
        assert text.startswith(r.row + ":") and "timestamp" in text and "spreads" in text
        assert ("steps" in text) == r.kind.startswith("shift")


# --------------------------------------------------------------------------- surfaces


def test_insights_include_spike_finding(v):
    report = v.insights()
    hits = [f for f in report["findings"] if f["kind"] == "spike"]
    assert hits and hits[0]["columns"] == ["timestamp"] and hits[0]["detail"]["row"] == "3389"
    assert 0 < hits[0]["significance"] <= 1
    json.dumps(report)  # JSON-safe


def test_insights_skip_spikes_without_a_time_column(fw):
    report = p2h.fit(fw.drop(columns=["timestamp"]), rows=["dst_port"]).insights(sensitivity=1.0)
    assert not [f for f in report["findings"] if f["kind"] == "spike"]


def test_prompt_lists_movements(v):
    p = v.prompt("q?")
    lines = [ln for ln in str(p).splitlines() if ln.startswith("- over time:")]
    assert lines and "3389" in lines[0]
    assert "- over time:" not in str(v.prompt("q?", spikes=False))
    assert "- over time:" not in str(v.compare(action="deny").prompt("q?"))  # off by default in a comparison prompt


def test_agent_spikes(bursty):
    r = agent.spikes(bursty, rows=["dst_port"], cols=["action"], agg="count", n=2)
    assert set(r) == {"layout", "column", "measure", "n", "spikes"} and r["column"] == "timestamp" and r["n"] == 2
    assert r["spikes"][0]["row"] == "3389" and r["spikes"][0]["text"].startswith("3389: count spikes")
    json.dumps(r)
    filtered = agent.spikes(bursty, rows=["dst_port"], cols=["action"], agg="count", filters=[{"column": "action", "eq": "allow"}], n=5)
    assert all(s["support"] < 100 for s in filtered["spikes"])  # the burst is all denies: filtered out


def test_mcp_spikes_tool(tmp_path, bursty):
    pytest.importorskip("mcp")
    import asyncio

    from pivot2hist import mcp_server

    path = tmp_path / "fw.csv"
    bursty.to_csv(path, index=False)

    async def run():
        return await mcp_server.server.call_tool("spikes", {"source": str(path), "rows": ["dst_port"], "cols": ["action"], "agg": "count", "n": 1})

    out = asyncio.run(run())
    r = json.loads(out.content[0].text)
    assert r["spikes"][0]["row"] == "3389" and r["spikes"][0]["kind"] == "spike"
