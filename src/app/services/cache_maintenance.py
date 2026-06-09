from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from infra.settings import AppSettings, get_settings


@dataclass(frozen=True)
class CacheUsageSummary:
    cache_root: str
    target_bytes: dict[str, int]
    total_bytes: int


@dataclass(frozen=True)
class CacheClearResult:
    cleared_targets: tuple[str, ...]
    freed_bytes: int
    failures: tuple[str, ...]


@dataclass(frozen=True)
class RuntimeStorageSummary:
    runtime_root: str
    target_bytes: dict[str, int]
    total_bytes: int


class CacheMaintenanceService:
    DIRECTORY_TARGETS = (
        "cluster_results/",
        "cluster_meanings/",
        "embedding_indexes/",
        "thumbnails/",
        "onnx_models/",
        "tmp/",
    )

    def __init__(self, settings: AppSettings | None = None) -> None:
        self.settings = settings or get_settings()

    def rebuildable_targets(self) -> dict[str, Path]:
        cache_dir = self.settings.cache_dir
        return {
            "embeddings.sqlite3": self.settings.embedding_cache_db,
            "cluster_results/": cache_dir / "cluster_results",
            "cluster_meanings/": cache_dir / "cluster_meanings",
            "embedding_indexes/": cache_dir / "embedding_indexes",
            "thumbnails/": self.settings.thumbnail_cache_dir,
            "onnx_models/": cache_dir / "onnx_models",
            "tmp/": cache_dir / "tmp",
        }

    def describe_rebuildable_caches(self) -> CacheUsageSummary:
        targets = self.rebuildable_targets()
        target_bytes = {name: self._path_size(path) for name, path in targets.items()}
        return CacheUsageSummary(
            cache_root=str(self.settings.cache_dir),
            target_bytes=target_bytes,
            total_bytes=sum(target_bytes.values()),
        )

    def describe_runtime_storage(self, runtime_layout) -> RuntimeStorageSummary:
        rebuildable_summary = self.describe_rebuildable_caches()
        runtime_temp = int(rebuildable_summary.target_bytes.get("tmp/", 0))
        rebuildable = max(0, int(rebuildable_summary.total_bytes - runtime_temp))
        tag_db = self._path_size(self.settings.image_tags_db)
        cache_total = self._path_size(self.settings.cache_dir)
        other_cache = max(0, int(cache_total - rebuildable - runtime_temp - tag_db))
        target_bytes = {
            "rebuildable_caches": rebuildable,
            "runtime_temp_files": runtime_temp,
            "tag_database": tag_db,
            "other_cache_data": other_cache,
            "logs": self._path_size(runtime_layout.logs_dir),
            "crash_reports": self._path_size(runtime_layout.crash_dir),
            "benchmarks": self._path_size(runtime_layout.benchmarks_dir),
            "support_bundles": self._path_size(runtime_layout.support_dir),
            "model_assets": self._path_size(runtime_layout.model_assets_dir),
        }
        return RuntimeStorageSummary(
            runtime_root=str(runtime_layout.root),
            target_bytes=target_bytes,
            total_bytes=sum(target_bytes.values()),
        )

    def clear_rebuildable_disk_targets(self, *, exclude: set[str] | None = None) -> tuple[tuple[str, ...], tuple[str, ...]]:
        exclude = {str(name) for name in (exclude or set())}
        cleared: list[str] = []
        failures: list[str] = []
        for name, path in self.rebuildable_targets().items():
            if name in exclude:
                continue
            try:
                self._remove_target(path)
                cleared.append(name)
            except OSError as exc:
                failures.append(f"{name}: {exc}")
        self._recreate_directories(exclude=exclude)
        return tuple(cleared), tuple(failures)

    def clear_runtime_cleanup_targets(self, runtime_layout, *, exclude: set[str] | None = None) -> tuple[tuple[str, ...], tuple[str, ...]]:
        cleared_targets, failures = self.clear_rebuildable_disk_targets(exclude=exclude)
        cleared = list(cleared_targets)
        failure_messages = list(failures)
        extra_targets = {
            "benchmarks/": runtime_layout.benchmarks_dir,
            "support/": runtime_layout.support_dir,
        }
        for name, path in extra_targets.items():
            try:
                self._remove_target(path)
                path.mkdir(parents=True, exist_ok=True)
                cleared.append(name)
            except OSError as exc:
                failure_messages.append(f"{name}: {exc}")
        return tuple(cleared), tuple(failure_messages)

    def clear_runtime_temp_files(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        cleared: list[str] = []
        failures: list[str] = []
        tmp_path = self.rebuildable_targets()["tmp/"]
        try:
            self._remove_target(tmp_path)
            tmp_path.mkdir(parents=True, exist_ok=True)
            cleared.append("tmp/")
        except OSError as exc:
            failures.append(f"tmp/: {exc}")

        download_cleared, download_failures = self.clear_partial_model_downloads()
        cleared.extend(download_cleared)
        failures.extend(download_failures)
        return tuple(cleared), tuple(failures)

    def clear_partial_model_downloads(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        candidates = (
            self.settings.cache_dir / "huggingface" / "hub",
            self.settings.cache_dir / "torch" / "hub" / "checkpoints",
        )
        removed: list[str] = []
        failures: list[str] = []
        suffixes = (".incomplete", ".lock", ".tmp", ".partial")
        for root in candidates:
            if not root.exists():
                continue
            for path in root.rglob("*"):
                try:
                    if not path.name.endswith(suffixes):
                        continue
                    if not path.is_file():
                        continue
                    size = path.stat().st_size
                    path.unlink(missing_ok=True)
                    removed.append(f"{path.relative_to(self.settings.cache_dir)} ({size} bytes)")
                except OSError as exc:
                    failures.append(f"{path}: {exc}")
        return tuple(removed), tuple(failures)

    def _recreate_directories(self, *, exclude: set[str]) -> None:
        self.settings.cache_dir.mkdir(parents=True, exist_ok=True)
        for name in self.DIRECTORY_TARGETS:
            if name in exclude:
                continue
            path = self.rebuildable_targets()[name]
            path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _remove_target(path: Path) -> None:
        if not path.exists():
            return
        if path.is_dir():
            shutil.rmtree(path)
            return
        path.unlink(missing_ok=True)

    @staticmethod
    def _path_size(path: Path) -> int:
        if not path.exists():
            return 0
        if path.is_file():
            try:
                return int(path.stat().st_size)
            except OSError:
                return 0
        total = 0
        for child in path.rglob("*"):
            try:
                if child.is_file():
                    total += int(child.stat().st_size)
            except OSError:
                continue
        return total
