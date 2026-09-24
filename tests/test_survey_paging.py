import numpy as np
import pandas as pd
import pytest

import pivot2hist as p2h
from pivot2hist._survey import Machine, PagedSource, _conform, downcast, load_planned, survey

pq = pytest.importorskip("pyarrow")


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    df = p2h.sample.firewall_logs(30_000, seed=3)
    csv, parquet, jsonl = d / "fw.csv", d / "fw.parquet", d / "fw.jsonl"
    df.to_csv(csv, index=False)
    df.to_parquet(parquet, index=False, row_group_size=5_000)
    df.head(5000).to_json(jsonl, orient="records", lines=True, date_format="iso")
    return {"df": df, "csv": csv, "parquet": parquet, "jsonl": jsonl}


def test_machine_probe():
    m = Machine.probe()
    assert m.cpu_count >= 1 and "CPU" in m.summary()


def test_survey_frame_and_files(files):
    sv = survey(files["df"])
    assert sv.format == "frame" and sv.rows == 30_000 and sv.rows_exact and sv.plan.mode == "full"
    assert "in memory" in sv.summary() and sv.to_dict()["plan"]["mode"] == "full"
    sp = survey(files["parquet"], memory_budget_mb=3)
    assert sp.format == "parquet" and sp.rows == 30_000 and sp.rows_exact and sp.plan.mode == "paged"
    assert sp.plan.n_pages >= 2 and sp.plan.sample_rows <= 30_000 and sp.disk_mb > 0
    sc = survey(files["csv"], memory_budget_mb=3)
    assert sc.format == "csv" and not sc.rows_exact and abs(sc.rows - 30_000) / 30_000 < 0.05 and sc.plan.mode == "paged"
    tight = survey(files["csv"], memory_budget_mb=survey(files["csv"]).est_memory_mb * 1.5)
    assert tight.plan.mode == "downcast" and tight.verdict == "tight"
    assert survey(files["csv"], columns=["action", "bytes"]).columns == ["action", "bytes"]
    with pytest.raises(FileNotFoundError):
        survey("/nope/missing.csv")


def test_load_planned_modes(files):
    df, sv = load_planned(files["csv"])
    assert isinstance(df, pd.DataFrame) and len(df) == 30_000 and sv.plan.mode == "full"
    dc, _ = load_planned(files["csv"], mode="downcast")
    assert str(dc["src_ip"].dtype) == "category" and dc.memory_usage(deep=True).sum() < df.memory_usage(deep=True).sum()
    smp, _ = load_planned(files["parquet"], mode="sample", sample_rows=2_000)
    assert len(smp) == 2_000
    pg, sv2 = load_planned(files["parquet"], memory_budget_mb=3)
    assert isinstance(pg, PagedSource) and len(pg) == 30_000 and len(pg.sample) <= sv2.plan.sample_rows
    assert sum(len(p) for p in pg.pages()) == 30_000 and pg.rows_seen == 30_000
    forced, _ = load_planned(files["csv"], mode="paged", page_rows=10_000)
    assert isinstance(forced, PagedSource) and forced.survey.plan.n_pages == 3
    page = next(forced.pages())
    assert pd.api.types.is_datetime64_any_dtype(page["timestamp"]) and len(page) == 10_000
    js, _ = load_planned(files["jsonl"], mode="paged", page_rows=2_000)
    assert sum(len(p) for p in js.pages()) == 5_000


def test_conform_and_downcast():
    page = pd.DataFrame({"t": ["2026-01-01", "2026-01-02"], "n": ["1", "2"], "b": ["yes", "no"], "s": ["a", "b"]})
    dt = {"t": np.dtype("datetime64[ns]"), "n": np.dtype("int64"), "b": np.dtype("bool"), "s": pd.CategoricalDtype(["a", "b"])}
    out = _conform(page, dt)
    assert pd.api.types.is_datetime64_any_dtype(out["t"]) and out["n"].tolist() == [1, 2] and out["b"].tolist() == [True, False]
    df = pd.DataFrame({"i": np.arange(100, dtype="int64"), "f": np.arange(100, dtype="float64"), "s": ["x", "y"] * 50})
    d = downcast(df)
    assert d["i"].dtype.itemsize < 8 and d["f"].dtype == np.float32 and str(d["s"].dtype) == "category"


def test_duckdb_source(files, tmp_path):
    duckdb = pytest.importorskip("duckdb")
    path = tmp_path / "ev.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE events AS SELECT * FROM read_parquet(?)", [str(files["parquet"])])
    con.close()
    url = f"duckdb://{path}?table=events"
    sv = survey(url)
    assert sv.format == "duckdb" and sv.rows == 30_000 and sv.rows_exact
    pg, _ = load_planned(url, memory_budget_mb=3, page_rows=8_192)
    assert isinstance(pg, PagedSource) and sum(len(p) for p in pg.pages()) == 30_000
    con = duckdb.connect(str(path), read_only=True)
    v = p2h.fit(con, query="SELECT action, bytes, dst_port FROM events", memory_budget_mb=0.5)
    assert v.paged is not None and v.pivot().to_numpy().sum() > 0
    full = p2h.fit(con, table="events")
    assert full.paged is None and len(full.data) == 30_000
    con.close()


def test_paged_view_exactness(files):
    df = files["df"]
    full = p2h.fit(files["parquet"])
    assert full.survey is not None and full.paged is None
    pg = p2h.fit(files["parquet"], memory_budget_mb=3)
    assert pg.paged is not None and "(paged" in pg.title() and "≈" in pg.title()
    ref = full.relayout(rows=list(pg.layout.rows), cols=list(pg.layout.cols), **pg._measure_spec()).pivot()
    got = pg.pivot()
    assert ref.shape == got.shape and np.allclose(ref.reindex_like(got).fillna(0).to_numpy(), got.fillna(0).to_numpy())
    # count / mean / max / min via pages equal the in-memory numbers
    cnt = pg.relayout(rows=["action"], cols=["protocol"], agg="count").pivot()
    assert cnt.to_numpy().sum() == 30_000
    mean_pg = pg.relayout(rows=["action"], cols=["protocol"], values="bytes", agg="mean")
    mean_full = full.relayout(rows=list(mean_pg.layout.rows), cols=list(mean_pg.layout.cols), values="bytes", agg="mean")
    assert np.allclose(mean_full.pivot().reindex_like(mean_pg.pivot()).to_numpy(), mean_pg.pivot().to_numpy(), equal_nan=True)
    for agg in ("max", "min"):
        a = pg.relayout(rows=["action"], cols=[], values="bytes", agg=agg).pivot()
        b = full.relayout(rows=["action"], cols=[], values="bytes", agg=agg).pivot()
        assert a.iloc[:, 0].tolist() == b.iloc[:, 0].tolist()
    # slices apply per page
    s = pg.slice(action="deny")
    assert s.pivot().to_numpy().sum() == df.loc[df["action"] == "deny", "bytes"].sum()
    assert " of ≈" in s.title()
    # histogram and toggle stay paged
    h = pg.histogram("bytes")
    assert h.paged is not None and int(h.bins()["count"].sum()) == 30_000 and h.toggle().layout == pg.layout
    assert pg.toggle().bins().shape[1] == pg.pivot().shape[1]
    # non-combinable aggregations fall back to the sample and say so
    med = pg.relayout(rows=["action"], cols=[], values="bytes", agg="median")
    assert med.approximate and "sample" in med.title()
    # clustering pivot rows works page-wise through derived columns
    c = pg.cluster(3)
    assert c.layout.rows[0].column == "cluster" and c.pivot().to_numpy().sum() == got.to_numpy().sum()
    with pytest.raises(ValueError):
        pg.cluster(on=["bytes", "duration"])
    # sample / materialize give in-memory views
    assert pg.sample(500).paged is None and len(pg.sample(500).data) == 500
    m = pg.slice(action="drop").materialize()
    assert m.paged is None and len(m.data) == (df["action"] == "drop").sum()
    assert pg.suggest(2) and pg.use(0).paged is not None
    assert "paged source" not in pg.html() and "<table" in pg.html()


def test_fit_frame_with_budget_uses_survey(files):
    v = p2h.fit(files["df"], memory_budget_mb=1_000)
    assert v.survey is not None and v.survey.format == "frame"
    assert p2h.fit(files["df"]).survey is None
