from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path

from infra.settings import get_settings


@dataclass(frozen=True)
class DiscoveryResult:
    paths: tuple[str, ...]
    snapshot_key: str
    image_count: int
    fingerprints: tuple[tuple[str, int, int], ...] = ()


class ImageDiscoveryService:
    def __init__(self) -> None:
        self.settings = get_settings()

    def discover(self, directory: str, recursive: bool | None = None) -> list[str]:
        return list(self.discover_result(directory, recursive=recursive).paths)

    def discover_result(self, directory: str, recursive: bool | None = None, progress_callback=None) -> DiscoveryResult:
        recursive = self.settings.recursive_scan if recursive is None else recursive
        root = Path(directory)
        if not root.exists():
            return DiscoveryResult(paths=(), snapshot_key=hashlib.sha256(b"").hexdigest(), image_count=0, fingerprints=())

        records: list[tuple[str, int, int]] = []
        stack = [str(root)]
        extensions = set(self.settings.image_extensions)
        scanned_dirs = 0
        last_progress_s = 0.0

        def emit_progress(*, force: bool = False) -> None:
            nonlocal last_progress_s
            if progress_callback is None:
                return
            now = time.monotonic()
            if not force and now - last_progress_s < self.settings.progress_emit_interval_s:
                return
            last_progress_s = now
            progress_callback(
                -1,
                f"Scanning folder: {len(records)} image(s) found in {scanned_dirs} folder(s)",
            )

        emit_progress(force=True)

        while stack:
            current = stack.pop()
            try:
                with os.scandir(current) as entries:
                    scanned_dirs += 1
                    child_dirs: list[str] = []
                    for entry in entries:
                        try:
                            if entry.is_dir(follow_symlinks=False):
                                if recursive:
                                    child_dirs.append(entry.path)
                                continue
                            if not entry.is_file(follow_symlinks=False):
                                continue
                            if Path(entry.name).suffix.lower() not in extensions:
                                continue
                            stat = entry.stat(follow_symlinks=False)
                            records.append((os.path.abspath(entry.path), int(stat.st_mtime_ns), int(stat.st_size)))
                            emit_progress()
                        except OSError:
                            continue
                    if recursive:
                        child_dirs.sort(reverse=True)
                        stack.extend(child_dirs)
                    emit_progress()
            except OSError:
                continue

        records.sort(key=lambda item: item[0].lower())
        payload = "\n".join(f"{path}|{mtime_ns}|{size}" for path, mtime_ns, size in records)
        if progress_callback is not None:
            progress_callback(0, f"Folder scan complete: {len(records)} image(s) discovered.")
        return DiscoveryResult(
            paths=tuple(path for path, _mtime_ns, _size in records),
            snapshot_key=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            image_count=len(records),
            fingerprints=tuple(records),
        )

    @staticmethod
    def collect_fingerprints(image_paths: list[str]) -> tuple[tuple[str, int, int], ...]:
        records: list[tuple[str, int, int]] = []
        for image_path in sorted({str(path) for path in image_paths if path}, key=str.lower):
            path = Path(image_path)
            try:
                stat = path.stat()
            except OSError:
                continue
            records.append((os.path.abspath(str(path)), int(stat.st_mtime_ns), int(stat.st_size)))
        return tuple(records)
