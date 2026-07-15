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


@dataclass(frozen=True)
class GeneratedStorageSummary:
    runtime_root: str
    config_location: str
    cache_root: str
    target_bytes: dict[str, int]
    target_paths: dict[str, tuple[str, ...]]
    total_bytes: int


class CacheMaintenanceService:
    DIRECTORY_TARGETS = (
        "cluster_results/",
        "cluster_meanings/",
        "embedding_indexes/",
        "face_model_assets/",
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
            "face_model_assets/": cache_dir / "face_model_assets",
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

    def describe_generated_storage(self, *, config_location: str = "") -> GeneratedStorageSummary:
        target_paths = self.generated_storage_targets()
        target_bytes = {
            name: sum(self._path_size(Path(path)) for path in paths)
            for name, paths in target_paths.items()
        }
        return GeneratedStorageSummary(
            runtime_root=str(self.settings.base_dir),
            config_location=str(config_location or ""),
            cache_root=str(self.settings.cache_dir),
            target_bytes=target_bytes,
            target_paths={name: tuple(str(path) for path in paths) for name, paths in target_paths.items()},
            total_bytes=sum(target_bytes.values()),
        )

    def generated_storage_targets(self) -> dict[str, tuple[Path, ...]]:
        cache_dir = self.settings.cache_dir
        rebuildable = self.rebuildable_targets()
        return {
            "logs": (self.settings.log_dir,),
            "thumbnails": (self.settings.thumbnail_cache_dir,),
            "rebuildable_caches": (
                self.settings.embedding_cache_db,
                rebuildable["cluster_results/"],
                rebuildable["cluster_meanings/"],
                rebuildable["embedding_indexes/"],
            ),
            "face_databases": self._face_database_paths(),
            "ann_files": self._ann_file_paths(),
            "model_caches": (
                cache_dir / "face_model_assets",
                cache_dir / "onnx_models",
                cache_dir / "huggingface",
                cache_dir / "torch",
            ),
            "temp_files": (cache_dir / "tmp",),
        }

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

    def clear_face_storage_targets(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        return self._clear_named_paths([*self._face_database_paths(), *self._ann_file_paths()], recreate_dirs=())

    def clear_model_cache_targets(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        cache_dir = self.settings.cache_dir
        paths = (
            cache_dir / "face_model_assets",
            cache_dir / "onnx_models",
            cache_dir / "huggingface",
            cache_dir / "torch",
        )
        return self._clear_named_paths(paths, recreate_dirs=(cache_dir / "face_model_assets", cache_dir / "onnx_models"))

    def clear_log_files(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        log_dir = self.settings.log_dir
        if not log_dir.exists():
            log_dir.mkdir(parents=True, exist_ok=True)
            return (), ()
        return self._clear_named_paths(tuple(log_dir.iterdir()), recreate_dirs=(log_dir,))

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

    def _face_database_paths(self) -> tuple[Path, ...]:
        cache_dir = self.settings.cache_dir
        return tuple(sorted(cache_dir.glob("face_search*.db*")))

    def _ann_file_paths(self) -> tuple[Path, ...]:
        cache_dir = self.settings.cache_dir
        return tuple(sorted(cache_dir.glob("face_search*.faiss*")))

    def _clear_named_paths(self, paths, *, recreate_dirs: tuple[Path, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
        cleared: list[str] = []
        failures: list[str] = []
        for path in tuple(dict.fromkeys(Path(item) for item in paths)):
            try:
                if path.exists():
                    self._remove_target(path)
                    cleared.append(self._display_target_name(path))
            except OSError as exc:
                failures.append(f"{self._display_target_name(path)}: {exc}")
        for path in recreate_dirs:
            try:
                path.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                failures.append(f"{self._display_target_name(path)}: {exc}")
        return tuple(cleared), tuple(failures)

    def _display_target_name(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.settings.cache_dir))
        except ValueError:
            try:
                return str(path.relative_to(self.settings.base_dir))
            except ValueError:
                return str(path)

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
