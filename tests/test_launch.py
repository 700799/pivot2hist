import datetime as dt
import html

import pandas as pd
import pytest

import bts_pivot as bp

W = pytest.importorskip("ipywidgets")
from bts_pivot import ui_launch  # noqa: E402
from bts_pivot.ui import Explorer, explore  # noqa: E402
from bts_pivot.ui_launch import (  # noqa: E402
    SOURCE_COLUMN, Launcher, _Receiver, _safe_relpath, _upload_items, folder_files, is_supported, load_files,
)


@pytest.fixture
def folder(tmp_path, fw):
    """A folder like a user's log dump: nested data files plus junk that must be skipped."""
    (tmp_path / "day2").mkdir()
    fw.iloc[:100].to_csv(tmp_path / "day1.csv", index=False)
    fw.iloc[100:200].to_json(tmp_path / "day2" / "part.jsonl", orient="records", lines=True, date_format="iso")
    (tmp_path / "notes.md").write_text("not data")
    (tmp_path / ".hidden.csv").write_text("a\n1\n")
    (tmp_path / "model.pkl").write_bytes(b"never unpickled")
    (tmp_path / ".ipynb_checkpoints").mkdir()
    (tmp_path / ".ipynb_checkpoints" / "day1-checkpoint.csv").write_text("a\n1\n")
    return tmp_path


def test_explorer_alias():
    assert bp.explorer is bp.explore
    assert "explorer" in bp.__all__


@pytest.mark.parametrize("name, ok", [
    ("a.csv", True), ("A.CSV", True), ("x.tsv", True), ("x.jsonl", True), ("x.parquet", True), ("x.xlsx", True),
    ("x.csv.gz", True), ("x.json.zip", True), ("x.parquet.gz", False), ("x.gz", False),
    ("x.pkl", False), ("x.md", False), ("x", False),
])
def test_is_supported(name, ok):
    assert is_supported(name) is ok


def test_folder_files_recurses_and_skips_hidden_and_unsupported(folder):
    assert [p.relative_to(folder).as_posix() for p in folder_files(folder)] == ["day1.csv", "day2/part.jsonl"]


def test_load_files_adds_source_column_and_reports_failures(folder, tmp_path_factory):
    bad = folder / "broken.parquet"
    bad.write_bytes(b"not parquet")
    df, failed = load_files(folder_files(folder), root=folder)
    assert len(df) == 200
    assert df.columns[0] == SOURCE_COLUMN
    assert sorted(df[SOURCE_COLUMN].unique()) == ["day1.csv", "day2/part.jsonl"]
    assert len(failed) == 1 and failed[0].startswith("broken.parquet:")


def test_load_files_keeps_an_existing_source_file_column(tmp_path):
    pd.DataFrame({SOURCE_COLUMN: ["orig"], "x": [1]}).to_csv(tmp_path / "a.csv", index=False)
    pd.DataFrame({SOURCE_COLUMN: ["orig"], "x": [2]}).to_csv(tmp_path / "b.csv", index=False)
    df, _ = load_files(folder_files(tmp_path), root=tmp_path)
    assert list(df[SOURCE_COLUMN]) == ["orig", "orig"]  # the data's own column is untouched
    assert list(df["_" + SOURCE_COLUMN]) == ["a.csv", "b.csv"]


def test_load_files_all_unreadable_raises(tmp_path):
    (tmp_path / "a.parquet").write_bytes(b"nope")
    with pytest.raises(ValueError, match="none of the files could be read"):
        load_files([tmp_path / "a.parquet"])


def test_explore_folder_path_combines_files(folder):
    ex = explore(str(folder), max_rows=8, max_cols=4)
    assert isinstance(ex, Explorer)
    assert len(ex.view.source) == 200 and SOURCE_COLUMN in ex.view.source.columns


def test_explore_folder_with_one_file_stays_a_path(tmp_path, fw):
    fw.head(50).to_csv(tmp_path / "only.csv", index=False)
    assert ui_launch.resolve_source(str(tmp_path)) == str(tmp_path / "only.csv")


def test_explore_empty_folder_says_why(tmp_path):
    (tmp_path / "readme.md").write_text("x")
    with pytest.raises(FileNotFoundError, match="no data files"):
        explore(str(tmp_path))


def test_resolve_source_leaves_non_folders_alone(fw):
    assert ui_launch.resolve_source(fw) is fw
    assert ui_launch.resolve_source("duckdb://db.duckdb") == "duckdb://db.duckdb"


def test_explore_without_data_opens_the_start_screen():
    L = explore()
    assert isinstance(L, Launcher) and L.explorer is None and L.view is None
    assert L.box.children == (L.start,)


def test_launcher_demo_then_back(fw):
    L = explore(max_rows=8, max_cols=4)
    L.open_demo("auth")
    assert isinstance(L.explorer, Explorer)
    assert L.box.children[1] is L.explorer.box
    assert "auth logs" in L.w_loaded.value
    L.show_start()
    assert L.box.children == (L.start,)
    assert L.explorer is not None  # kept, so a handle to it doesn't go stale


def test_launcher_open_path_file_folder_and_errors(folder, fw):
    L = explore(max_rows=8, max_cols=4)
    L.open_path(str(folder / "day1.csv"))
    assert len(L.view.source) == 100 and "day1.csv" in L.w_loaded.value
    L.open_path(f"  '{folder}'  ")  # pasted with quotes and spaces
    assert len(L.view.source) == 200 and f"{folder.name}/" in L.w_loaded.value
    for bad, msg in [("", "type the path"), (str(folder / "nope.csv"), "no such file"),
                     (str(folder / "notes.md"), "isn't a supported"), (str(folder / "model.pkl"), "isn't a supported")]:
        L.open_path(bad)
        assert msg in html.unescape(L.w_status.value)


def test_launcher_shows_load_errors_instead_of_raising(tmp_path):
    (tmp_path / "bad.parquet").write_bytes(b"not parquet")
    L = explore()
    L.open_path(str(tmp_path / "bad.parquet"))
    assert "couldn't open bad.parquet" in html.unescape(L.w_status.value) and L.explorer is None


def test_safe_relpath_cannot_escape():
    assert _safe_relpath("../../etc/passwd.csv").as_posix() == "etc/passwd.csv"
    assert _safe_relpath("/abs/x.csv").as_posix() == "abs/x.csv"
    assert _safe_relpath("logs\\day1.csv").as_posix() == "logs/day1.csv"
    assert _safe_relpath("..").as_posix() == "upload"


def _chunked_upload(uploader, files, chunk=7):
    """Drive the uploader's message protocol the way the browser does, in small chunks."""
    acks = []
    uploader.send = lambda content, buffers=None: acks.append(content)
    uploader._on_msg(uploader, {"kind": "begin", "files": len(files), "skipped": 1, "folder": True}, [])
    for name, data in files:
        off = 0
        while True:
            uploader._on_msg(uploader, {"kind": "chunk", "name": name, "offset": off}, [memoryview(data[off:off + chunk])])
            off += chunk
            if off >= len(data):
                break
    uploader._on_msg(uploader, {"kind": "end"}, [])
    return acks


def test_uploader_reassembles_a_chunked_folder_upload(fw):
    if not ui_launch.HAS_ANYWIDGET:
        pytest.skip("anywidget not installed")
    L = explore(max_rows=8, max_cols=4)
    a = fw.iloc[:60].to_csv(index=False).encode()
    b = fw.iloc[60:100].to_csv(index=False).encode()
    acks = _chunked_upload(L.uploader, [("logs/a.csv", a), ("logs/sub/b.csv", b)])
    assert all(m == {"kind": "ack"} for m in acks) and len(acks) == -(-len(a) // 7) + -(-len(b) // 7)
    assert len(L.view.source) == 100
    assert sorted(L.view.source[SOURCE_COLUMN].unique()) == ["logs/a.csv", "logs/sub/b.csv"]
    assert "logs/" in L.w_loaded.value and "skipped 1 non-data file" in L.w_loaded.value


def test_uploader_keeps_traversal_names_inside_its_folder(fw):
    if not ui_launch.HAS_ANYWIDGET:
        pytest.skip("anywidget not installed")
    L = explore(max_rows=8, max_cols=4)
    _chunked_upload(L.uploader, [("../../escape.csv", fw.head(20).to_csv(index=False).encode())])
    (folder,) = L._tmp
    written = [p for p in folder.rglob("*") if p.is_file()]
    assert [p.relative_to(folder).as_posix() for p in written] == ["escape.csv"]
    assert len(L.view.source) == 20


def test_uploaded_empty_file_reports_an_error_instead_of_raising():
    if not ui_launch.HAS_ANYWIDGET:
        pytest.skip("anywidget not installed")
    L = explore()
    _chunked_upload(L.uploader, [("empty.csv", b"")])
    assert "couldn't open empty.csv" in html.unescape(L.w_status.value)


@pytest.mark.parametrize("value", [
    ({"name": "a.csv", "content": memoryview(b"x,y\n1,2\n3,4\n")},),  # ipywidgets 8
    {"a.csv": {"metadata": {"name": "a.csv"}, "content": b"x,y\n1,2\n3,4\n"}},  # ipywidgets 7
])
def test_upload_items_reads_both_fileupload_formats(value):
    assert _upload_items(value) == [("a.csv", b"x,y\n1,2\n3,4\n")]


def test_fallback_uploader_without_anywidget(fw):
    opened = []
    up = ui_launch._fallback_uploader(lambda paths, folder, skipped: opened.append((paths, skipped)))
    upload = up.children[0]
    data = fw.head(10).to_csv(index=False).encode()
    files = {"a.csv": data, "pic.png": b"x"}
    if int(W.__version__.split(".")[0]) >= 8:
        now = dt.datetime.now(dt.timezone.utc)
        value = tuple({"name": n, "type": "", "size": len(b), "content": memoryview(b), "last_modified": now}
                      for n, b in files.items())
    else:
        value = {n: {"metadata": {"name": n, "size": len(b)}, "content": b} for n, b in files.items()}
    upload.set_trait("value", value)
    (paths, skipped), = opened
    assert [p.name for p in paths] == ["a.csv"] and skipped == 1
    assert pd.read_csv(paths[0]).shape == (10, fw.shape[1])


def test_receiver_ignores_an_end_without_files():
    called = []
    rx = _Receiver(lambda *a: called.append(a))
    rx.end()
    rx.begin()
    rx.end()
    assert called == []
