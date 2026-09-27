from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path

from app.path_scope import PathScope
from app.services.source_admission import SourceAdmissionPolicy
from infra.cancel import Cancelled
from infra.settings import get_settings


@dataclass(frozen=True)
class DiscoveryResult:
    paths: tuple[str, ...]
    snapshot_key: str
    image_count: int
    fingerprints: tuple[tuple[str, int, int], ...] = ()
    complete: bool = True
    warning_count: int = 0
    filtered_count: int = 0


class ImageDiscoveryService:
    def __init__(self, *, admission_policy: SourceAdmissionPolicy | None = None) -> None:
        self.settings = get_settings()
        # Settings are snapshotted when a background operation is created, so
        # a scan cannot change its own source set halfway through.
        self.admission_policy = admission_policy or SourceAdmissionPolicy.from_global_settings()

    def discover(self, directory: str, recursive: bool | None = None) -> list[str]:
        return list(self.discover_result(directory, recursive=recursive).paths)

    def discover_roots_result(
        self,
        roots: list[str] | tuple[str, ...],
        recursive: bool | None = None,
        progress_callback=None,
        cancel_check=None,
    ) -> DiscoveryResult:
        """Discover a deterministic, de-duplicated union of active roots.

        This intentionally composes the existing cancellable single-root walk
        instead of making Qt enumerate a filesystem model.  ``PathScope`` has
        already removed ancestor/descendant overlap; inode/realpath keys still
        protect against symlink aliases across distinct roots.
        """

        scope = PathScope.from_paths(roots)
        if scope.is_empty:
            return DiscoveryResult(paths=(), snapshot_key=hashlib.sha256(b"").hexdigest(), image_count=0, fingerprints=())

        records_by_identity: dict[object, tuple[str, int, int]] = {}
        complete = True
        warning_count = 0
        filtered_count = 0
        total_roots = len(scope.roots)
        for root_index, root in enumerate(scope.roots, start=1):
            if callable(cancel_check) and cancel_check():
                raise Cancelled()
            if not Path(root).is_dir():
                complete = False
                warning_count += 1
                if progress_callback:
                    progress_callback(-1, f"Skipping unavailable root {root_index}/{total_roots}: {root}")
                continue

            def _progress(value: int, message: str, *, index=root_index, current_root=root) -> None:
                if progress_callback is None:
                    return
                prefix = f"Root {index}/{total_roots} ({Path(current_root).name or current_root})"
                progress_callback(value, f"{prefix}: {message}")

            result = self.discover_result(
                root,
                recursive=recursive,
                progress_callback=_progress,
                cancel_check=cancel_check,
            )
            complete = bool(complete and result.complete)
            warning_count += int(result.warning_count)
            filtered_count += int(result.filtered_count)
            for path, mtime_ns, size in result.fingerprints:
                if callable(cancel_check) and cancel_check():
                    raise Cancelled()
                try:
                    stat_result = os.stat(path, follow_symlinks=True)
                    inode = int(getattr(stat_result, "st_ino", 0) or 0)
                    identity: object = (int(getattr(stat_result, "st_dev", 0) or 0), inode) if inode else os.path.realpath(path)
                except OSError:
                    identity = os.path.realpath(path)
                records_by_identity.setdefault(identity, (path, int(mtime_ns), int(size)))

        records = sorted(records_by_identity.values(), key=lambda item: item[0].casefold())
        payload = "\n".join(f"{path}|{mtime_ns}|{size}" for path, mtime_ns, size in records)
        if progress_callback is not None:
            message = f"Active-root scan complete: {len(records)} image(s) discovered across {total_roots} root(s)."
            if filtered_count:
                message += f" {filtered_count} file(s) were excluded by source filters."
            if not complete:
                message += " Some paths could not be read."
            progress_callback(0, message)
        return DiscoveryResult(
            paths=tuple(path for path, _mtime_ns, _size in records),
            snapshot_key=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            image_count=len(records),
            fingerprints=tuple(records),
            complete=complete,
            warning_count=warning_count,
            filtered_count=filtered_count,
        )

    def discover_result(self, directory: str, recursive: bool | None = None, progress_callback=None, cancel_check=None) -> DiscoveryResult:
        recursive = self.settings.recursive_scan if recursive is None else recursive
        root = Path(directory)
        if not root.exists():
            return DiscoveryResult(paths=(), snapshot_key=hashlib.sha256(b"").hexdigest(), image_count=0, fingerprints=())

        records_by_key: dict[object, tuple[str, int, int]] = {}
        stack = [os.path.abspath(str(root))]
        extensions = {str(item).lower() for item in self.settings.image_extensions}
        scanned_dirs = 0
        last_progress_s = 0.0
        complete = True
        warning_count = 0
        filtered_count = 0
        visited_dirs: set[object] = set()

        def mark_warning() -> None:
            nonlocal complete, warning_count
            complete = False
            warning_count += 1

        def identity_key(path: str, stat_result: os.stat_result) -> object:
            inode = int(getattr(stat_result, "st_ino", 0) or 0)
            device = int(getattr(stat_result, "st_dev", 0) or 0)
            if inode > 0:
                return (device, inode)
            return os.path.realpath(path)

        def emit_progress(*, force: bool = False) -> None:
            nonlocal last_progress_s
            if callable(cancel_check) and cancel_check():
                raise Cancelled()
            if progress_callback is None:
                return
            now = time.monotonic()
            if not force and now - last_progress_s < self.settings.progress_emit_interval_s:
                return
            last_progress_s = now
            progress_callback(
                -1,
                f"Scanning folder: {len(records_by_key)} image(s) found in {scanned_dirs} folder(s)",
            )

        emit_progress(force=True)

        while stack:
            if callable(cancel_check) and cancel_check():
                raise Cancelled()
            current = stack.pop()
            try:
                current_stat = os.stat(current, follow_symlinks=True)
                current_key = identity_key(current, current_stat)
                if current_key in visited_dirs:
                    continue
                visited_dirs.add(current_key)
                with os.scandir(current) as entries:
                    scanned_dirs += 1
                    child_dirs: list[str] = []
                    for entry in entries:
                        if callable(cancel_check) and cancel_check():
                            raise Cancelled()
                        try:
                            entry_path = os.path.abspath(entry.path)
                            is_dir = entry.is_dir(follow_symlinks=False)
                            is_file = entry.is_file(follow_symlinks=False)
                            if entry.is_symlink() and not is_dir and not is_file:
                                is_dir = entry.is_dir(follow_symlinks=True)
                                is_file = False if is_dir else entry.is_file(follow_symlinks=True)
                            if is_dir:
                                if recursive:
                                    child_dirs.append(entry_path)
                                continue
                            if not is_file:
                                continue
                            if Path(entry.name).suffix.lower() not in extensions:
                                continue
                            stat = entry.stat(follow_symlinks=True)
                            decision = self.admission_policy.decide(entry_path, stat_result=stat)
                            if not decision.admitted:
                                filtered_count += 1
                                continue
                            file_key = identity_key(entry_path, stat)
                            if file_key not in records_by_key:
                                records_by_key[file_key] = (entry_path, int(stat.st_mtime_ns), int(stat.st_size))
                            emit_progress()
                        except OSError:
                            mark_warning()
                            continue
                    if recursive:
                        child_dirs.sort(reverse=True)
                        stack.extend(child_dirs)
                    emit_progress()
            except OSError:
                mark_warning()
                continue

        records = sorted(records_by_key.values(), key=lambda item: item[0].lower())
        payload = "\n".join(f"{path}|{mtime_ns}|{size}" for path, mtime_ns, size in records)
        if progress_callback is not None:
            message = f"Folder scan complete: {len(records)} image(s) discovered."
            if filtered_count:
                message += f" {filtered_count} file(s) were excluded by source filters."
            if not complete:
                message = f"{message} Some paths could not be read."
            progress_callback(0, message)
        return DiscoveryResult(
            paths=tuple(path for path, _mtime_ns, _size in records),
            snapshot_key=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            image_count=len(records),
            fingerprints=tuple(records),
            complete=bool(complete),
            warning_count=int(warning_count),
            filtered_count=int(filtered_count),
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
