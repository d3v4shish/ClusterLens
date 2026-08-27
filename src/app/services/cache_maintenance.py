from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from infra.settings import AppSettings, get_settings
from infra.cancel import raise_if_cancelled


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

    def describe_rebuildable_caches(self, *, progress_callback=None, cancel_check=None) -> CacheUsageSummary:
        targets = self.rebuildable_targets()
        target_bytes: dict[str, int] = {}
        total_targets = max(1, len(targets))
        for index, (name, path) in enumerate(targets.items()):
            raise_if_cancelled(cancel_check)
            if progress_callback:
                progress_callback(int(index * 100 / total_targets), f"Scanning {name}")
            target_bytes[name] = self._path_size(path, cancel_check=cancel_check)
        if progress_callback:
            progress_callback(100, "Cache usage scan complete")
        return CacheUsageSummary(
            cache_root=str(self.settings.cache_dir),
            target_bytes=target_bytes,
            total_bytes=sum(target_bytes.values()),
        )

    def describe_runtime_storage(self, runtime_layout, *, progress_callback=None, cancel_check=None) -> RuntimeStorageSummary:
        rebuildable_summary = self.describe_rebuildable_caches(
            progress_callback=progress_callback,
            cancel_check=cancel_check,
        )
        raise_if_cancelled(cancel_check)
        runtime_temp = int(rebuildable_summary.target_bytes.get("tmp/", 0))
        rebuildable = max(0, int(rebuildable_summary.total_bytes - runtime_temp))
        tag_db = self._path_size(self.settings.image_tags_db, cancel_check=cancel_check)
        cache_total = self._path_size(self.settings.cache_dir, cancel_check=cancel_check)
        other_cache = max(0, int(cache_total - rebuildable - runtime_temp - tag_db))
        target_bytes = {
            "rebuildable_caches": rebuildable,
            "runtime_temp_files": runtime_temp,
            "tag_database": tag_db,
            "other_cache_data": other_cache,
            "logs": self._path_size(runtime_layout.logs_dir, cancel_check=cancel_check),
            "crash_reports": self._path_size(runtime_layout.crash_dir, cancel_check=cancel_check),
            "benchmarks": self._path_size(runtime_layout.benchmarks_dir, cancel_check=cancel_check),
            "support_bundles": self._path_size(runtime_layout.support_dir, cancel_check=cancel_check),
            "model_assets": self._path_size(runtime_layout.model_assets_dir, cancel_check=cancel_check),
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
                cache_dir / "face_model_downloads",
                cache_dir / "onnx_models",
                cache_dir / "huggingface",
                cache_dir / "torch",
            ),
            "temp_files": (cache_dir / "tmp",),
        }

    def clear_rebuildable_disk_targets(
        self,
        *,
        exclude: set[str] | None = None,
        progress_callback=None,
        cancel_check=None,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        exclude = {str(name) for name in (exclude or set())}
        cleared: list[str] = []
        failures: list[str] = []
        targets = [(name, path) for name, path in self.rebuildable_targets().items() if name not in exclude]
        total = max(1, len(targets))
        for index, (name, path) in enumerate(targets):
            raise_if_cancelled(cancel_check)
            if progress_callback:
                progress_callback(int(index * 100 / total), f"Clearing {name}")
            try:
                self._remove_target(path, cancel_check=cancel_check)
                cleared.append(name)
            except OSError as exc:
                failures.append(f"{name}: {exc}")
        self._recreate_directories(exclude=exclude, cancel_check=cancel_check)
        if progress_callback:
            progress_callback(100, "Rebuildable cache clear complete")
        return tuple(cleared), tuple(failures)

    def clear_runtime_cleanup_targets(
        self,
        runtime_layout,
        *,
        exclude: set[str] | None = None,
        progress_callback=None,
        cancel_check=None,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        cleared_targets, failures = self.clear_rebuildable_disk_targets(
            exclude=exclude,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
        )
        cleared = list(cleared_targets)
        failure_messages = list(failures)
        extra_targets = {
            "benchmarks/": runtime_layout.benchmarks_dir,
            "support/": runtime_layout.support_dir,
        }
        for name, path in extra_targets.items():
            raise_if_cancelled(cancel_check)
            try:
                self._remove_target(path, cancel_check=cancel_check)
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

        # Hugging Face and Torch partial downloads are resumable cache state,
        # not disposable runtime temp files. Preserve them across app restarts.
        return tuple(cleared), tuple(failures)

    def clear_face_storage_targets(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        return self._clear_named_paths([*self._face_database_paths(), *self._ann_file_paths()], recreate_dirs=())

    def clear_model_cache_targets(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        cache_dir = self.settings.cache_dir
        paths = (
            cache_dir / "face_model_assets",
            cache_dir / "face_model_downloads",
            cache_dir / "onnx_models",
            cache_dir / "huggingface",
            cache_dir / "torch",
        )
        return self._clear_named_paths(
            paths,
            recreate_dirs=(
                cache_dir / "face_model_assets",
                cache_dir / "face_model_downloads",
                cache_dir / "onnx_models",
            ),
        )

    def clear_log_files(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        log_dir = self.settings.log_dir
        if not log_dir.exists():
            log_dir.mkdir(parents=True, exist_ok=True)
            return (), ()
        protected_names = {
            "file_operations.jsonl",
            "file_operations.sqlite3",
            "file_operations.sqlite3-shm",
            "file_operations.sqlite3-wal",
        }
        return self._clear_named_paths(
            tuple(path for path in log_dir.iterdir() if path.name not in protected_names),
            recreate_dirs=(log_dir,),
        )

    def clear_partial_model_downloads(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        candidates = (
            self.settings.cache_dir / "huggingface" / "hub",
            self.settings.cache_dir / "torch" / "checkpoints",
            self.settings.cache_dir / "torch" / "hub" / "checkpoints",
            self.settings.cache_dir / "face_model_downloads",
        )
        removed: list[str] = []
        failures: list[str] = []
        # Never remove lock files here: another app process may own them.
        suffixes = (".incomplete", ".tmp", ".partial")
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

    def _recreate_directories(self, *, exclude: set[str], cancel_check=None) -> None:
        self.settings.cache_dir.mkdir(parents=True, exist_ok=True)
        for name in self.DIRECTORY_TARGETS:
            raise_if_cancelled(cancel_check)
            if name in exclude:
                continue
            path = self.rebuildable_targets()[name]
            path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _remove_target(path: Path, *, cancel_check=None) -> None:
        raise_if_cancelled(cancel_check)
        if not path.exists():
            return
        if path.is_symlink() or not path.is_dir():
            path.unlink(missing_ok=True)
            return
        for root, directories, files in os.walk(path, topdown=False):
            raise_if_cancelled(cancel_check)
            root_path = Path(root)
            for filename in files:
                raise_if_cancelled(cancel_check)
                (root_path / filename).unlink(missing_ok=True)
            for directory in directories:
                raise_if_cancelled(cancel_check)
                child = root_path / directory
                if child.is_symlink():
                    child.unlink(missing_ok=True)
                else:
                    child.rmdir()
        path.rmdir()

    @staticmethod
    def _path_size(path: Path, *, cancel_check=None) -> int:
        raise_if_cancelled(cancel_check)
        if not path.exists():
            return 0
        if path.is_file():
            try:
                return int(path.stat().st_size)
            except OSError:
                return 0
        total = 0
        for child in path.rglob("*"):
            raise_if_cancelled(cancel_check)
            try:
                if child.is_file():
                    total += int(child.stat().st_size)
            except OSError:
                continue
        return total
