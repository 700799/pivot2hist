"""A scrolling log of major steps, with wall time, CPU time and memory deltas per step,
and a report of what cost the most.

::

    import pivot2hist as p2h
    p2h.verbose()                 # print every major step to stderr as it happens
    v = p2h.fit("big.parquet")    # survey -> plan -> pages -> fit -> pivot ...
    p2h.stats(7)                  # the seven costliest steps (time, cpu, memory)
    p2h.log.tail(12)              # the last twelve log lines
"""
from __future__ import annotations

import os
import sys
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, Iterator, List, Optional, TextIO

import pandas as pd


def rss_bytes() -> Optional[int]:
    """Resident set size of this process, or ``None`` when it cannot be read."""
    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError, AttributeError):
        pass
    try:
        import psutil  # optional

        return int(psutil.Process().memory_info().rss)
    except Exception:  # noqa: BLE001
        return None


@dataclass
class Entry:
    """One logged step."""

    when: float  # epoch seconds at completion
    step: str
    detail: str
    seconds: float
    cpu: float
    mem_mb: float  # RSS delta during the step
    rss_mb: float  # RSS after the step (0 when unknown)
    depth: int

    def format(self) -> str:
        indent = "  " * self.depth
        t = time.strftime("%H:%M:%S", time.localtime(self.when))
        cost = f"{self.seconds:6.2f}s  cpu {self.cpu:5.2f}s  mem {self.mem_mb:+6.0f} MB" if self.seconds or self.cpu else ""
        return f"{t}  {indent}{self.step:<12} {self.detail}  {cost}".rstrip()


class _Holder:
    """Lets a step set its detail message while it runs."""

    def __init__(self, detail: str = ""):
        self.detail = detail


class Log:
    """Ring buffer of :class:`Entry` plus optional live printing and listeners."""

    def __init__(self, maxlen: int = 2000):
        self.entries: Deque[Entry] = deque(maxlen=maxlen)
        self.listeners: List[Callable[[Entry], None]] = []
        self.stream: Optional[TextIO] = None
        self._depth = 0

    # ---- control
    def enable(self, stream: Optional[TextIO] = None) -> "Log":
        self.stream = stream or sys.stderr
        return self

    def disable(self) -> "Log":
        self.stream = None
        return self

    def clear(self) -> None:
        self.entries.clear()

    def listen(self, fn: Callable[[Entry], None]) -> Callable[[], None]:
        """Register a listener called for every entry; returns an unsubscribe function."""
        self.listeners.append(fn)

        def off() -> None:
            if fn in self.listeners:
                self.listeners.remove(fn)

        return off

    # ---- recording
    def _emit(self, e: Entry) -> None:
        self.entries.append(e)
        if self.stream is not None:
            try:
                print(e.format(), file=self.stream, flush=True)
            except Exception:  # noqa: BLE001 - never let logging break work
                pass
        for fn in list(self.listeners):
            try:
                fn(e)
            except Exception:  # noqa: BLE001
                pass

    def info(self, step: str, detail: str = "") -> None:
        """An instantaneous entry (no timing)."""
        rss = rss_bytes()
        self._emit(Entry(time.time(), step, detail, 0.0, 0.0, 0.0, (rss or 0) / 1e6, self._depth))

    @contextmanager
    def step(self, name: str, detail: str = "") -> Iterator[_Holder]:
        """Time a step: ``with log.step("fit", "5,000 rows") as s: ...; s.detail += " -> layout"``."""
        holder = _Holder(detail)
        t0, c0, m0 = time.perf_counter(), time.process_time(), rss_bytes()
        self._depth += 1
        try:
            yield holder
        finally:
            self._depth -= 1
            m1 = rss_bytes()
            mem = ((m1 or 0) - (m0 or 0)) / 1e6 if (m0 is not None and m1 is not None) else 0.0
            self._emit(
                Entry(time.time(), name, holder.detail, time.perf_counter() - t0, time.process_time() - c0, mem, (m1 or 0) / 1e6, self._depth)
            )

    # ---- reading
    def tail(self, n: int = 12) -> List[Entry]:
        return list(self.entries)[-n:]

    def lines(self, n: int = 12) -> List[str]:
        return [e.format() for e in self.tail(n)]

    def stats(self, n: int = 7, by: str = "seconds") -> pd.DataFrame:
        """The ``n`` costliest step names: calls, total wall time, CPU time, peak memory delta.

        Nested steps (a page inside a pivot) are counted under their own name; the outer
        step's time includes them, so the table reads as "where did the time go" per kind
        of work, not as a sum.
        """
        if not self.entries:
            return pd.DataFrame(columns=["step", "calls", "seconds", "cpu", "mem_mb", "last_detail"])
        rows: Dict[str, Dict[str, Any]] = {}
        for e in self.entries:
            r = rows.setdefault(e.step, {"step": e.step, "calls": 0, "seconds": 0.0, "cpu": 0.0, "mem_mb": 0.0, "last_detail": ""})
            r["calls"] += 1
            r["seconds"] += e.seconds
            r["cpu"] += e.cpu
            r["mem_mb"] = max(r["mem_mb"], e.mem_mb)
            r["last_detail"] = e.detail or r["last_detail"]
        df = pd.DataFrame(list(rows.values()))
        df = df[df["seconds"] > 0] if (df["seconds"] > 0).any() else df
        return df.sort_values(by, ascending=False).head(n).reset_index(drop=True)

    def __repr__(self) -> str:
        return "\n".join(self.lines(20)) or "<Log: empty>"


#: The module-wide log every pivot2hist step writes to.
log = Log()


def step(name: str, detail: str = ""):
    """Shortcut for ``log.step``."""
    return log.step(name, detail)


def verbose(on: bool = True, stream: Optional[TextIO] = None) -> Log:
    """Print each major step (survey, pages, fit, pivot, render ...) as it completes."""
    return log.enable(stream) if on else log.disable()


def stats(n: int = 7, by: str = "seconds") -> pd.DataFrame:
    """The ``n`` costliest kinds of step so far (see :meth:`Log.stats`)."""
    return log.stats(n, by)


__all__ = ["Log", "Entry", "log", "step", "verbose", "stats", "rss_bytes"]
