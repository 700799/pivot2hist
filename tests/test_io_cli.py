import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import pivot2hist as p2h
from pivot2hist.cli import main, parse_slice


def test_load_records_and_dict_and_array():
    recs = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    assert list(p2h.load(recs).columns) == ["a", "b"]
    assert p2h.load({"a": [1, 2], "b": ["x", "y"]}).shape == (2, 2)
    assert list(p2h.load(np.zeros((3, 2))).columns) == ["c0", "c1"]
    assert p2h.load(pd.Series([1, 2], name="v")).columns.tolist() == ["v"]


def test_load_csv_parses_dates(tmp_path, fw):
    path = tmp_path / "fw.csv"
    fw.head(200).to_csv(path, index=False)
    df = p2h.load(path)
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    assert df["dst_port"].dtype.kind == "i"
    raw = p2h.load(path, parse_dates=False)
    assert not pd.api.types.is_datetime64_any_dtype(raw["timestamp"])


def test_load_jsonl_and_epoch(tmp_path):
    path = tmp_path / "ev.jsonl"
    rows = [{"ts": 1_700_000_000 + i * 60, "kind": "a" if i % 2 else "b", "n": i} for i in range(50)]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    df = p2h.load(path)
    assert pd.api.types.is_datetime64_any_dtype(df["ts"])
    assert p2h.fit(df).pivot().shape[0] >= 2


def test_load_missing_file():
    with pytest.raises(FileNotFoundError):
        p2h.load("/nonexistent/file.csv")


def test_parse_slice():
    assert parse_slice("action=deny") == ("action", "deny")
    assert parse_slice("port=22,443") == ("port", [22, 443])
    assert parse_slice("bytes=100..200") == ("bytes", (100, 200))
    assert parse_slice("bytes=100..") == ("bytes", (100, None))
    assert parse_slice("bytes>=5") == ("bytes", ">=5")
    assert parse_slice("ip~^10") == ("ip", "~^10")
    assert parse_slice("flag=true") == ("flag", True)


def test_cli_demo_and_file(tmp_path, capsys, fw):
    assert main(["--demo", "firewall", "--max-rows", "6", "--max-cols", "3", "--width", "100"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("pivot ·")
    path = tmp_path / "fw.csv"
    fw.to_csv(path, index=False)
    assert main([str(path), "--hist", "--on", "bytes", "--bins", "5", "--ascii", "--width", "80"]) == 0
    out = capsys.readouterr().out
    assert "hist · count by bytes" in out and "#" in out
    assert main([str(path), "--slice", "action=deny", "--slice", "dst_port=22,3389", "--count", "--layout"]) == 0
    assert "count by" in capsys.readouterr().out
    assert main([str(path), "--json", "--max-rows", "5"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["shape"][0] <= 5
    assert main([str(path), "--csv", "--rows", "action", "--cols", "protocol", "--count"]) == 0
    assert "allow" in capsys.readouterr().out
    assert main([str(path), "--profile"]) == 0
    assert "categorical" in capsys.readouterr().out
    assert main([str(path), "--slicers"]) == 0
    assert "action:" in capsys.readouterr().out
    assert main([str(path), "--where", "bytes > 1000", "--hist", "--by", "action"]) == 0
    assert "allow" in capsys.readouterr().out


def test_cli_no_args_prints_help(capsys, monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    assert main([]) == 2
    assert "usage" in capsys.readouterr().out.lower()


def test_cli_entry_point_installed():
    r = subprocess.run([sys.executable, "-m", "pivot2hist", "--version"], capture_output=True, text=True)
    assert r.returncode == 0 and "pivot2hist" in r.stdout
