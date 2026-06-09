from __future__ import annotations

import cProfile
import ctypes
import io
import os
import pstats
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class ProfileResult:
    elapsed_s: float
    hotspots: list[dict[str, object]]
    stats_text: str


def current_memory_rss_mb() -> float | None:
    try:
        import psutil  # type: ignore

        process = psutil.Process()
        return round(float(process.memory_info().rss) / (1024.0 * 1024.0), 3)
    except Exception:
        if os.name == "nt":
            return _current_memory_rss_mb_windows()
        return None


def _current_memory_rss_mb_windows() -> float | None:
    from ctypes import wintypes

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    try:
        ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE  # type: ignore[attr-defined]
        ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [  # type: ignore[attr-defined]
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            wintypes.DWORD,
        ]
        ctypes.windll.psapi.GetProcessMemoryInfo.restype = wintypes.BOOL  # type: ignore[attr-defined]
        handle = ctypes.windll.kernel32.GetCurrentProcess()  # type: ignore[attr-defined]
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(  # type: ignore[attr-defined]
            handle,
            ctypes.byref(counters),
            counters.cb,
        )
    except Exception:
        return None
    if not ok:
        return None
    return round(float(counters.WorkingSetSize) / (1024.0 * 1024.0), 3)


def profile_call(fn, *, sort_by: str = "cumulative", limit: int = 20):
    profiler = cProfile.Profile()
    started = time.perf_counter()
    result = profiler.runcall(fn)
    elapsed_s = round(time.perf_counter() - started, 6)
    stats = pstats.Stats(profiler)
    stats.sort_stats(sort_by)
    hotspots = _top_stats(stats, limit=limit)
    text_buffer = io.StringIO()
    stats.stream = text_buffer
    stats.print_stats(limit)
    return result, ProfileResult(
        elapsed_s=elapsed_s,
        hotspots=hotspots,
        stats_text=text_buffer.getvalue(),
    )


def _top_stats(stats: pstats.Stats, *, limit: int) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for index, (func, raw) in enumerate(stats.stats.items()):
        _cc, nc, tt, ct, callers = raw
        filename, lineno, func_name = func
        items.append(
            {
                "index": index,
                "function": f"{filename}:{lineno}::{func_name}",
                "ncalls": int(nc),
                "tottime_s": round(float(tt), 6),
                "cumtime_s": round(float(ct), 6),
                "callers": len(callers),
            }
        )
    items.sort(key=lambda item: float(item["cumtime_s"]), reverse=True)
    return items[: max(1, int(limit))]
