"""Semantic types on top of storage kinds, and the drill hierarchies they unlock.

A column that *is* text may *also be* an IP address, a URL or an email; a column that is
an integer may be a port. Knowing that lets the fit bucket values along a meaningful
hierarchy (``10.0.1.0/24`` rather than "top 39 IPs + other") and lets the UI drill.

The inference is deliberately simple and written from scratch: a handful of anchored
regexes tested on a sample of distinct values, claiming a type only when nearly every
value fits. This is the same *idea* as relation-based type systems used by pandas
profilers, not their code.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api import types as pdt

IPV4 = "ipv4"
IPV6 = "ipv6"
MAC = "mac"
EMAIL = "email"
URL = "url"
DOMAIN = "domain"
UUID = "uuid"
HASH = "hash"
PATH = "path"
PORT = "port"

_IPV4 = re.compile(r"^(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)$")
_IPV6 = re.compile(r"^(?=.*:.*:)[0-9a-f:]{2,39}(?:%[a-z0-9]+)?$", re.I)
_MAC = re.compile(r"^(?:[0-9a-f]{2}[:-]){5}[0-9a-f]{2}$", re.I)
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_URL = re.compile(r"^[a-z][a-z0-9+.\-]*://\S+$", re.I)
_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
_HASH = re.compile(r"^(?:[0-9a-f]{32}|[0-9a-f]{40}|[0-9a-f]{64})$", re.I)
_PATH = re.compile(r"^(?:/[^/\s]+)+/?$|^[a-z]:\\[^\s]*$", re.I)
_DOMAIN = re.compile(r"^(?=.{4,253}$)(?:[a-z0-9](?:[a-z0-9\-]{0,61}[a-z0-9])?\.)+[a-z]{2,24}$", re.I)

# Most specific first: a UUID is also a "hash-like" string, an email also has a domain.
_CHECKS: List[Tuple[str, "re.Pattern[str]"]] = [
    (UUID, _UUID), (IPV4, _IPV4), (MAC, _MAC), (EMAIL, _EMAIL), (URL, _URL),
    (HASH, _HASH), (IPV6, _IPV6), (PATH, _PATH), (DOMAIN, _DOMAIN),
]

#: Drill levels per semantic type, coarse -> fine. The last level is the raw value.
HIERARCHY: Dict[str, Tuple[str, ...]] = {
    IPV4: ("/8", "/16", "/24", "host"),
    IPV6: ("/32", "/64", "host"),
    URL: ("host", "path", "full"),
    EMAIL: ("domain", "full"),
    DOMAIN: ("site", "full"),
    PATH: ("top", "dir", "full"),
    PORT: ("class", "port"),
    MAC: ("oui", "full"),
}

_TWO_LEVEL_TLDS = {
    "co.uk", "org.uk", "ac.uk", "gov.uk", "com.au", "net.au", "org.au", "co.jp", "co.nz", "co.za",
    "com.br", "com.cn", "com.mx", "com.sg", "co.in", "co.kr", "com.tr", "com.ar", "co.il", "com.hk",
}

_PORT_NAMES = {
    20: "ftp-data", 21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp", 53: "dns", 67: "dhcp", 68: "dhcp",
    69: "tftp", 80: "http", 88: "kerberos", 110: "pop3", 123: "ntp", 135: "msrpc", 137: "netbios",
    138: "netbios", 139: "netbios", 143: "imap", 161: "snmp", 162: "snmp-trap", 179: "bgp", 389: "ldap",
    443: "https", 445: "smb", 465: "smtps", 514: "syslog", 587: "submission", 636: "ldaps", 993: "imaps",
    995: "pop3s", 1433: "mssql", 1521: "oracle", 1723: "pptp", 2049: "nfs", 3306: "mysql", 3389: "rdp",
    5432: "postgres", 5900: "vnc", 5985: "winrm", 5986: "winrm", 6379: "redis", 8080: "http-alt",
    8443: "https-alt", 8888: "http-alt", 9200: "elastic", 27017: "mongodb",
}


def infer_semantic(s: pd.Series, *, name: str = "", min_frac: float = 0.9, sample: int = 500) -> Optional[str]:
    """Semantic type of a column, or ``None``.

    Text columns are matched against the anchored patterns above on a sample of distinct
    values; integer columns named like a port in ``0..65535`` are ports.
    """
    non_null = s.dropna()
    if non_null.empty:
        return None
    if pdt.is_numeric_dtype(non_null) and not pdt.is_bool_dtype(non_null):
        if re.search(r"port", name, re.I):
            v = non_null.to_numpy(dtype=float)
            if np.all((v >= 0) & (v <= 65535) & (np.mod(v, 1) == 0)):
                return PORT
        return None
    if not (pdt.is_string_dtype(non_null) or pdt.is_object_dtype(non_null) or isinstance(non_null.dtype, pd.CategoricalDtype)):
        return None
    try:
        distinct = non_null.drop_duplicates()
    except TypeError:
        return None
    if len(distinct) > sample:
        distinct = distinct.sample(sample, random_state=0)
    vals = [v for v in distinct.tolist() if isinstance(v, str)]
    if len(vals) < max(2, int(0.9 * len(distinct))):
        return None
    for sem, rx in _CHECKS:
        hits = sum(1 for v in vals if rx.match(v.strip()))
        if hits >= min_frac * len(vals):
            return sem
    return None


def levels_for(semantic: Optional[str]) -> Tuple[str, ...]:
    return HIERARCHY.get(semantic or "", ())


def bucket(s: pd.Series, semantic: str, level: str) -> pd.Series:
    """Map raw values to the given hierarchy level (nulls stay null).

    Repeated values are bucketed once: the work is done on the distinct values and
    mapped back, which keeps this cheap on long log frames.
    """
    levels = HIERARCHY.get(semantic)
    if not levels or level not in levels:
        raise ValueError(f"unknown level {level!r} for {semantic!r}; use one of {levels}")
    if level == levels[-1]:
        return s
    if len(s) > 2000:
        try:
            uniq = pd.Series(s.dropna().unique())
        except TypeError:
            uniq = None
        if uniq is not None and len(uniq) < len(s) * 0.5:
            mapped = _bucket_values(uniq, semantic, level)
            lookup = dict(zip(uniq.tolist(), mapped.tolist()))
            return s.map(lookup).astype(object).where(s.notna(), None)
    return _bucket_values(s, semantic, level)


def _bucket_values(s: pd.Series, semantic: str, level: str) -> pd.Series:
    if semantic == IPV4:
        n = {"/8": 1, "/16": 2, "/24": 3}[level]
        parts = s.astype("string").str.split(".", n=n, expand=True)
        out = parts.iloc[:, 0].astype("string")
        for i in range(1, n):
            out = out + "." + parts.iloc[:, i].astype("string")
        return (out + ".0" * (4 - n) + level).astype(object).where(s.notna(), None)
    if semantic == IPV6:
        n = {"/32": 2, "/64": 4}[level]
        exp = s.astype("string").str.lower().str.split(":", expand=True)
        cols = [
            exp.iloc[:, i].fillna("0").replace("", "0").astype("string") if i < exp.shape[1] else "0"
            for i in range(n)
        ]
        out = cols[0]
        for c in cols[1:]:
            out = out + ":" + c
        return (out + "::" + level).astype(object).where(s.notna(), None)
    if semantic == URL:
        rest = s.astype("string").str.replace(r"^[a-z][a-z0-9+.\-]*://", "", regex=True, case=False)
        host = rest.str.split("/", n=1).str[0].str.split("@").str[-1].str.split(":").str[0]
        if level == "host":
            return host.astype(object).where(s.notna(), None)
        first = rest.str.split("/", n=2).str[1].fillna("")
        return (host + "/" + first).astype(object).where(s.notna(), None)
    if semantic == EMAIL:
        return s.astype("string").str.split("@").str[-1].str.lower().astype(object).where(s.notna(), None)
    if semantic == DOMAIN:
        labels = s.astype("string").str.lower().str.split(".")

        def site(p):
            if not isinstance(p, list):
                return p
            keep = 3 if len(p) >= 3 and ".".join(p[-2:]) in _TWO_LEVEL_TLDS else 2
            return ".".join(p[-keep:])

        return labels.map(site).astype(object).where(s.notna(), None)
    if semantic == PATH:
        norm = s.astype("string").str.replace("\\", "/", regex=False)
        if level == "top":
            top = norm.str.extract(r"^(/[^/]+|[a-zA-Z]:)", expand=False).fillna("/")
            return top.astype(object).where(s.notna(), None)
        d = norm.str.rsplit("/", n=1).str[0]
        return d.where(d != "", "/").astype(object).where(s.notna(), None)
    if semantic == MAC:
        return s.astype("string").str.replace("-", ":", regex=False).str.lower().str[:8].astype(object).where(s.notna(), None)
    if semantic == PORT:
        v = pd.to_numeric(s, errors="coerce")
        cls = np.select([v < 1024, v < 49152], ["well-known (<1024)", "registered (1024-49151)"], default="ephemeral (49152+)")
        return pd.Series(cls, index=s.index, dtype=object).where(s.notna(), None)
    raise ValueError(f"no bucketing for {semantic!r}")  # pragma: no cover


def port_name(port: object) -> str:
    """``443`` -> ``"443 https"``; unknown ports come back unchanged as text."""
    try:
        p = int(port)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(port)
    name = _PORT_NAMES.get(p)
    return f"{p} {name}" if name else str(p)


__all__ = ["infer_semantic", "bucket", "levels_for", "port_name", "HIERARCHY"]
