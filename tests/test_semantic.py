import numpy as np
import pandas as pd
import pytest

from pivot2hist import _semantic as S


@pytest.mark.parametrize(
    "values,expected",
    [
        (["10.0.1.5", "192.168.0.1", "8.8.8.8"], "ipv4"),
        (["fe80::1", "2001:db8::ff00:42:8329", "::1"], "ipv6"),
        (["aa:bb:cc:dd:ee:ff", "00-11-22-33-44-55"], "mac"),
        (["a@x.com", "bob.smith@corp.example.org"], "email"),
        (["https://ex.com/a", "http://ex.org:8080/x?y=1"], "url"),
        (["550e8400-e29b-41d4-a716-446655440000", "123e4567-e89b-12d3-a456-426614174000"], "uuid"),
        (["d41d8cd98f00b204e9800998ecf8427e", "9e107d9d372bb6826bd81d3542a419d6"], "hash"),
        (["/var/log/auth.log", "/etc/passwd", "C:\\Windows\\System32\\cmd.exe"], "path"),
        (["www.example.com", "mail.google.co.uk", "cdn.net"], "domain"),
        (["allow", "deny", "drop"], None),
        (["10.0.1.5", "not an ip", "also not"], None),
    ],
)
def test_infer_semantic(values, expected):
    assert S.infer_semantic(pd.Series(values)) == expected


def test_port_needs_name_hint_and_range():
    assert S.infer_semantic(pd.Series([443, 80, 22]), name="dst_port") == "port"
    assert S.infer_semantic(pd.Series([443, 80, 22]), name="count") is None
    assert S.infer_semantic(pd.Series([443, 80, 70000]), name="dst_port") is None


def test_bucket_levels():
    ips = pd.Series(["10.0.1.5", "10.0.1.9", "192.168.4.1", None])
    assert S.bucket(ips, "ipv4", "/24").tolist() == ["10.0.1.0/24", "10.0.1.0/24", "192.168.4.0/24", None]
    assert S.bucket(ips, "ipv4", "/16").tolist()[0] == "10.0.0.0/16"
    assert S.bucket(ips, "ipv4", "/8").tolist()[2] == "192.0.0.0/8"
    assert S.bucket(ips, "ipv4", "host") is ips
    assert S.bucket(pd.Series(["a@X.com"]), "email", "domain").tolist() == ["x.com"]
    u = pd.Series(["https://ex.com/a/b", "http://u@ex.org:8080/", "ftp://f.net"])
    assert S.bucket(u, "url", "host").tolist() == ["ex.com", "ex.org", "f.net"]
    assert S.bucket(u, "url", "path").tolist() == ["ex.com/a", "ex.org/", "f.net/"]
    assert S.bucket(pd.Series([443, 8080, 60000]), "port", "class").tolist() == [
        "well-known (<1024)", "registered (1024-49151)", "ephemeral (49152+)"
    ]
    assert S.bucket(pd.Series(["/var/log/x", "/etc/passwd"]), "path", "top").tolist() == ["/var", "/etc"]
    assert S.bucket(pd.Series(["/var/log/x", "/etc/passwd"]), "path", "dir").tolist() == ["/var/log", "/etc"]
    assert S.bucket(pd.Series(["www.example.com", "mail.google.co.uk"]), "domain", "site").tolist() == ["example.com", "google.co.uk"]
    assert S.bucket(pd.Series(["fe80::1", "2001:db8::1"]), "ipv6", "/32").tolist() == ["fe80:0::/32", "2001:db8::/32"]
    assert S.bucket(pd.Series(["AA-BB-CC-DD-EE-FF"]), "mac", "oui").tolist() == ["aa:bb:cc"]
    with pytest.raises(ValueError):
        S.bucket(ips, "ipv4", "/12")


def test_bucket_uses_distinct_fast_path():
    ips = pd.Series(np.random.default_rng(0).choice([f"10.0.{i}.1" for i in range(50)], 5000))
    out = S.bucket(ips, "ipv4", "/24")
    assert len(out) == 5000 and out.str.endswith("/24").all()
    assert (out == "10.0.7.0/24").sum() == (ips == "10.0.7.1").sum()


def test_port_name():
    assert S.port_name(443) == "443 https"
    assert S.port_name(4444) == "4444"
    assert S.port_name("x") == "x"
    assert S.levels_for("ipv4") == ("/8", "/16", "/24", "host") and S.levels_for(None) == ()
