"""Synthetic datasets for demos and tests (cyber-flavoured, but the library is generic)."""
from __future__ import annotations

import numpy as np
import pandas as pd

_PORTS = np.array([443, 80, 22, 53, 3389, 8080, 25, 445, 3306, 21, 123, 161, 1433, 5900, 8443])
_PORT_W = np.array([0.30, 0.20, 0.08, 0.12, 0.04, 0.05, 0.03, 0.05, 0.02, 0.02, 0.03, 0.02, 0.01, 0.01, 0.02])
_COUNTRIES = np.array(["US", "DE", "CN", "RU", "BR", "IN", "GB", "NL", "FR", "KR"])
_COUNTRY_W = np.array([0.35, 0.10, 0.12, 0.08, 0.06, 0.07, 0.08, 0.05, 0.05, 0.04])


def _zipf_choice(rng: np.random.Generator, pool: np.ndarray, n: int, a: float = 1.2) -> np.ndarray:
    ranks = np.arange(1, len(pool) + 1)
    w = ranks ** (-a)
    w /= w.sum()
    return rng.choice(pool, size=n, p=w)


def firewall_logs(n: int = 5000, seed: int = 0, *, start: str = "2026-03-01") -> pd.DataFrame:
    """Firewall connection events: timestamp, src_ip, dst_ip, dst_port, protocol, action, bytes ..."""
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp(start)
    # 7 days with a daytime peak
    days = rng.integers(0, 7, n)
    hours = np.clip(rng.normal(13, 4.5, n), 0, 23.999)
    ts = t0 + pd.to_timedelta(days * 86400 + hours * 3600 + rng.uniform(0, 60, n), unit="s")

    src_pool = np.array([f"10.0.{i // 250}.{i % 250 + 1}" for i in range(300)])
    dst_pool = np.array([f"{rng.integers(1, 223)}.{rng.integers(0, 255)}.{rng.integers(0, 255)}.{rng.integers(1, 254)}" for _ in range(80)])
    src_ip = _zipf_choice(rng, src_pool, n, 1.1)
    dst_ip = _zipf_choice(rng, dst_pool, n, 1.0)
    dst_port = rng.choice(_PORTS, size=n, p=_PORT_W)
    high = rng.random(n) < 0.06
    dst_port = np.where(high, rng.integers(1024, 65535, n), dst_port)

    protocol = np.where(np.isin(dst_port, [53, 123, 161]), "UDP", "TCP")
    protocol = np.where(rng.random(n) < 0.03, "ICMP", protocol)

    risky = np.isin(dst_port, [22, 3389, 445, 1433, 5900, 3306]) | high
    p_deny = np.where(risky, 0.55, 0.08)
    action = np.where(rng.random(n) < p_deny, "deny", "allow")
    action = np.where((action == "deny") & (rng.random(n) < 0.3), "drop", action)

    bytes_ = np.exp(rng.normal(7.0, 1.8, n)).round().astype(np.int64)
    bytes_ = np.where(action != "allow", (bytes_ * 0.05).astype(np.int64), bytes_)
    duration = np.round(rng.exponential(2.5, n) * np.where(action == "allow", 1.0, 0.1), 3)
    severity = rng.choice([1, 2, 3, 4, 5], size=n, p=[0.45, 0.28, 0.15, 0.08, 0.04])
    severity = np.where(action != "allow", np.minimum(severity + 1, 5), severity)
    rule = np.where(
        action == "allow",
        np.char.add("fw-allow-", rng.integers(1, 6, n).astype(str)),
        np.char.add("fw-block-", rng.integers(1, 4, n).astype(str)),
    )
    country = rng.choice(_COUNTRIES, size=n, p=_COUNTRY_W)

    df = pd.DataFrame(
        {
            "timestamp": ts,
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "dst_port": dst_port.astype(int),
            "protocol": protocol,
            "action": action,
            "bytes": bytes_,
            "duration": duration,
            "severity": severity.astype(int),
            "rule": rule,
            "country": country,
        }
    )
    return df.sort_values("timestamp").reset_index(drop=True)


def auth_logs(n: int = 3000, seed: int = 1, *, start: str = "2026-03-01") -> pd.DataFrame:
    """Authentication events: timestamp, user, host, event, method, src_ip, latency_ms, mfa."""
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp(start)
    ts = t0 + pd.to_timedelta(rng.uniform(0, 14 * 86400, n), unit="s")
    users = np.array([f"user{i:03d}" for i in range(120)])
    hosts = np.array([f"host-{k}-{i:02d}" for k in ("web", "db", "vpn", "jump") for i in range(6)])
    user = _zipf_choice(rng, users, n, 1.0)
    host = _zipf_choice(rng, hosts, n, 0.8)
    method = rng.choice(["password", "sso", "ssh-key", "token"], size=n, p=[0.45, 0.3, 0.2, 0.05])
    attacker = rng.random(n) < 0.08
    p_fail = np.where(attacker, 0.9, 0.06)
    event = np.where(rng.random(n) < p_fail, "login_failure", "login_success")
    event = np.where((event == "login_success") & (rng.random(n) < 0.25), "logout", event)
    def _ips(prefix: str, third_hi: int) -> np.ndarray:
        third = rng.integers(0, third_hi, n).astype(str)
        fourth = rng.integers(1, 250, n).astype(str)
        return np.char.add(np.char.add(np.char.add(prefix, third), "."), fourth)

    src_ip = np.where(attacker, _ips("185.220.", 4), _ips("10.1.", 8))
    latency = np.round(rng.lognormal(4.5, 0.6, n), 1)
    mfa = rng.random(n) < np.where(method == "sso", 0.95, 0.4)
    return (
        pd.DataFrame(
            {
                "timestamp": ts,
                "user": user,
                "host": host,
                "event": event,
                "method": method,
                "src_ip": src_ip,
                "latency_ms": latency,
                "mfa": mfa,
            }
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


__all__ = ["firewall_logs", "auth_logs"]
