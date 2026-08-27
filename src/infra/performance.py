from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class SystemResources:
    logical_cpu_count: int
    total_memory_bytes: int
    available_memory_bytes: int


@dataclass(frozen=True)
class PerformanceProfile:
    name: str
    logical_cpu_count: int
    total_memory_bytes: int
    available_memory_bytes: int
    thumbnail_workers: int
    thumbnail_prefetch_rows: int
    embedding_preprocess_workers: int
    embedding_memory_cache_size: int
    qimage_cache_size: int
    pixmap_cache_size: int
    backend_worker_cap: int | None
    cpu_batch_size: int
    gpu_batch_size: int
    vram_headroom_mb: int

    def backend_workers_for(self, backend_count: int) -> int:
        backend_count = max(1, int(backend_count))
        if self.backend_worker_cap is None:
            return backend_count
        return max(1, min(int(self.backend_worker_cap), backend_count))


def detect_system_resources() -> SystemResources:
    logical_cpus = max(1, int(os.cpu_count() or 1))
    total_memory = 0
    available_memory = 0

    try:
        import psutil  # type: ignore

        memory = psutil.virtual_memory()
        total_memory = int(memory.total or 0)
        available_memory = int(memory.available or 0)
    except Exception:
        if os.name == "nt":
            total_memory, available_memory = _windows_memory()
        else:
            total_memory, available_memory = _posix_memory()

    return SystemResources(
        logical_cpu_count=logical_cpus,
        total_memory_bytes=max(0, int(total_memory)),
        available_memory_bytes=max(0, int(available_memory)),
    )


def select_performance_profile(profile_name: str | None, resources: SystemResources | None = None) -> PerformanceProfile:
    resources = resources or detect_system_resources()
    profile = str(profile_name or "balanced").strip().lower()
    if profile not in {"low_memory", "balanced", "max_speed"}:
        profile = "balanced"

    logical = max(1, int(resources.logical_cpu_count))
    available_gb = max(1.0, float(resources.available_memory_bytes or resources.total_memory_bytes or 0) / (1024**3))

    if profile == "low_memory":
        return PerformanceProfile(
            name="low_memory",
            logical_cpu_count=logical,
            total_memory_bytes=resources.total_memory_bytes,
            available_memory_bytes=resources.available_memory_bytes,
            thumbnail_workers=max(1, min(2, logical)),
            thumbnail_prefetch_rows=2,
            embedding_preprocess_workers=max(1, min(2, logical)),
            embedding_memory_cache_size=max(512, int(available_gb * 192)),
            qimage_cache_size=max(80, int(available_gb * 24)),
            pixmap_cache_size=max(120, int(available_gb * 32)),
            backend_worker_cap=1,
            cpu_batch_size=max(4, min(8, logical)),
            gpu_batch_size=max(8, min(16, logical * 2)),
            vram_headroom_mb=1536,
        )

    if profile == "max_speed":
        return PerformanceProfile(
            name="max_speed",
            logical_cpu_count=logical,
            total_memory_bytes=resources.total_memory_bytes,
            available_memory_bytes=resources.available_memory_bytes,
            thumbnail_workers=max(1, min(logical, 12)),
            thumbnail_prefetch_rows=max(8, min(max(8, logical), 16)),
            embedding_preprocess_workers=max(1, min(logical, 8)),
            embedding_memory_cache_size=max(8192, int(available_gb * 2048)),
            qimage_cache_size=max(600, int(available_gb * 180)),
            pixmap_cache_size=max(800, int(available_gb * 220)),
            backend_worker_cap=None,
            cpu_batch_size=max(16, min(24, logical)),
            gpu_batch_size=max(24, logical * 2),
            vram_headroom_mb=1024,
        )

    return PerformanceProfile(
        name="balanced",
        logical_cpu_count=logical,
        total_memory_bytes=resources.total_memory_bytes,
        available_memory_bytes=resources.available_memory_bytes,
        thumbnail_workers=max(2, min(logical, max(3, logical // 2))),
        thumbnail_prefetch_rows=max(4, min(12, max(4, logical // 2))),
        embedding_preprocess_workers=max(2, min(logical, max(3, logical // 2))),
        embedding_memory_cache_size=max(2048, int(available_gb * 768)),
        qimage_cache_size=max(250, int(available_gb * 90)),
        pixmap_cache_size=max(320, int(available_gb * 120)),
        backend_worker_cap=max(1, min(3, max(1, logical // 4))),
        cpu_batch_size=max(8, min(24, logical)),
        gpu_batch_size=max(16, min(48, logical * 2)),
        vram_headroom_mb=1024,
    )


def apply_performance_overrides(
    profile: PerformanceProfile,
    *,
    cpu_batch_size: int | None = None,
    gpu_batch_size: int | None = None,
    embedding_preprocess_workers: int | None = None,
    thumbnail_workers: int | None = None,
    thumbnail_prefetch_rows: int | None = None,
    vram_headroom_mb: int | None = None,
) -> PerformanceProfile:
    values: dict[str, int] = {}
    if cpu_batch_size is not None:
        values["cpu_batch_size"] = max(1, min(512, int(cpu_batch_size)))
    if gpu_batch_size is not None:
        values["gpu_batch_size"] = max(1, min(1024, int(gpu_batch_size)))
    if embedding_preprocess_workers is not None:
        values["embedding_preprocess_workers"] = max(1, min(128, int(embedding_preprocess_workers)))
    if thumbnail_workers is not None:
        values["thumbnail_workers"] = max(1, min(64, int(thumbnail_workers)))
    if thumbnail_prefetch_rows is not None:
        values["thumbnail_prefetch_rows"] = max(0, min(128, int(thumbnail_prefetch_rows)))
    if vram_headroom_mb is not None:
        values["vram_headroom_mb"] = max(256, min(65536, int(vram_headroom_mb)))
    return replace(profile, **values) if values else profile


def _windows_memory() -> tuple[int, int]:
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):  # type: ignore[attr-defined]
        return 0, 0
    return int(stat.ullTotalPhys), int(stat.ullAvailPhys)


def _posix_memory() -> tuple[int, int]:
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        total_pages = int(os.sysconf("SC_PHYS_PAGES"))
        avail_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        return page_size * total_pages, page_size * avail_pages
    except Exception:
        return 0, 0
