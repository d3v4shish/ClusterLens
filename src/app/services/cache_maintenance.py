from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from infra.settings import AppSettings, get_settings
from infra.cancel import Cancelled, raise_if_cancelled
from app.services.face_storage_recovery import FaceStorageRemovalService


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
    completion_survives_cancellation: bool = False
    retry_targets: tuple[str, ...] = ()


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


class _CacheDirectoryClearInterrupted(Cancelled):
    def __init__(self, target: str, *, committed: bool) -> None:
        super().__init__(target)
        self.target = str(target)
        self.committed = bool(committed)


class _CacheDirectoryClearFailed(OSError):
    def __init__(self, target: str, message: str, *, committed: bool) -> None:
        super().__init__(message)
        self.target = str(target)
        self.committed = bool(committed)


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
        library_catalog = self._library_catalog_size(cancel_check=cancel_check)
        cache_total = self._path_size(self.settings.cache_dir, cancel_check=cancel_check)
        other_cache = max(0, int(cache_total - rebuildable - runtime_temp - tag_db - library_catalog))
        target_bytes = {
            "rebuildable_caches": rebuildable,
            "runtime_temp_files": runtime_temp,
            "tag_database": tag_db,
            "library_catalog": library_catalog,
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

    def describe_generated_storage(
        self,
        *,
        config_location: str = "",
        runtime_layout=None,
        progress_callback=None,
        cancel_check=None,
    ) -> GeneratedStorageSummary:
        target_paths = self.generated_storage_targets(runtime_layout=runtime_layout)
        target_bytes: dict[str, int] = {}
        total_targets = max(1, len(target_paths))
        for index, (name, paths) in enumerate(target_paths.items()):
            raise_if_cancelled(cancel_check)
            if progress_callback:
                progress_callback(int(index * 100 / total_targets), f"Scanning {name}")
            target_bytes[name] = sum(self._path_size(Path(path), cancel_check=cancel_check) for path in paths)
        if progress_callback:
            progress_callback(100, "Generated storage scan complete")
        return GeneratedStorageSummary(
            runtime_root=str(getattr(runtime_layout, "root", self.settings.base_dir)),
            config_location=str(config_location or ""),
            cache_root=str(self.settings.cache_dir),
            target_bytes=target_bytes,
            target_paths={name: tuple(str(path) for path in paths) for name, paths in target_paths.items()},
            total_bytes=sum(target_bytes.values()),
        )

    def generated_storage_targets(self, *, runtime_layout=None) -> dict[str, tuple[Path, ...]]:
        cache_dir = self.settings.cache_dir
        rebuildable = self.rebuildable_targets()
        runtime_targets = {
            "crash_reports": self._runtime_layout_target(runtime_layout, "crash_dir"),
            "support_bundles": self._runtime_layout_target(runtime_layout, "support_dir"),
            "benchmarks": self._runtime_layout_target(runtime_layout, "benchmarks_dir"),
            "model_assets": self._runtime_layout_target(runtime_layout, "model_assets_dir"),
        }
        return {
            "logs": (self.settings.log_dir,),
            "thumbnails": (self.settings.thumbnail_cache_dir,),
            "rebuildable_caches": (
                self.settings.embedding_cache_db,
                rebuildable["cluster_results/"],
                rebuildable["cluster_meanings/"],
                rebuildable["embedding_indexes/"],
            ),
            "library_catalog": self._library_catalog_paths(),
            "face_databases": self._face_database_paths(),
            "ann_files": self._ann_file_paths(),
            "model_caches": (
                cache_dir / "face_model_assets",
                cache_dir / "face_model_state.json",
                cache_dir / "face_model_downloads",
                cache_dir / "onnx_models",
                cache_dir / "huggingface",
                cache_dir / "torch",
            ),
            "temp_files": (cache_dir / "tmp",),
            **{
                name: (path,) if path is not None else ()
                for name, path in runtime_targets.items()
            },
        }

    def clear_generated_storage_category(
        self,
        category: str,
        *,
        runtime_layout=None,
        progress_callback=None,
        cancel_check=None,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Remove one explicitly selected generated-data category.

        User-authored source files, tags, recovery history, and settings are
        intentionally not a category here. Face databases are an explicit
        high-friction exception because they also contain saved labels and
        identities; their clear path is recoverable.
        """
        category = str(category or "").strip()
        if progress_callback:
            progress_callback(0, f"Clearing {category or 'selected storage'}")
        raise_if_cancelled(cancel_check)
        if category == "logs":
            result = self.clear_log_files(cancel_check=cancel_check)
        elif category == "thumbnails":
            result = self._clear_named_paths(
                (self.settings.thumbnail_cache_dir,),
                recreate_dirs=(self.settings.thumbnail_cache_dir,),
                cancel_check=cancel_check,
            )
        elif category == "rebuildable_caches":
            # The Storage page reports thumbnails, model caches, and temporary
            # files separately. A category clear must not silently expand into
            # those other visible categories.
            rebuildable = self.rebuildable_targets()
            paths = (
                self.settings.embedding_cache_db,
                rebuildable["cluster_results/"],
                rebuildable["cluster_meanings/"],
                rebuildable["embedding_indexes/"],
            )
            result = self._clear_named_paths(paths, recreate_dirs=paths[1:], cancel_check=cancel_check)
        elif category == "face_databases":
            result = self.clear_face_storage_targets(cancel_check=cancel_check)
            # This batch has a durable commit point. A Cancel click arriving
            # after it must not be reported as if no files changed.
            return result
        elif category == "ann_files":
            result = self._clear_named_paths(self._ann_file_paths(), recreate_dirs=(), cancel_check=cancel_check)
        elif category == "model_caches":
            result = self.clear_model_cache_targets(cancel_check=cancel_check)
        elif category == "temp_files":
            result = self.clear_runtime_temp_files(cancel_check=cancel_check)
        elif category in {"crash_reports", "support_bundles", "benchmarks", "model_assets"}:
            attribute = {
                "crash_reports": "crash_dir",
                "support_bundles": "support_dir",
                "benchmarks": "benchmarks_dir",
                "model_assets": "model_assets_dir",
            }[category]
            path = self._runtime_layout_target(runtime_layout, attribute)
            if path is None:
                result = (), (f"{category} is unavailable without a runtime layout.",)
            else:
                result = self._clear_named_paths((path,), recreate_dirs=(path,), cancel_check=cancel_check)
        else:
            result = (), (f"Unknown generated-storage category: {category}",)
        raise_if_cancelled(cancel_check)
        if progress_callback:
            progress_callback(100, f"Finished clearing {category}")
        return result

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
        committed_any = False
        for index, (name, path) in enumerate(targets):
            try:
                raise_if_cancelled(cancel_check)
            except Cancelled:
                if not committed_any:
                    raise
                failures.append("Cancellation acknowledged after committed cache targets; choose Clear Rebuildable Caches again if cleanup remains pending.")
                break
            if progress_callback:
                progress_callback(int(index * 100 / total), f"Clearing {name}")
            try:
                if name.endswith("/"):
                    committed = self._clear_rebuildable_directory(name, path, cancel_check=cancel_check)
                    committed_any = committed_any or committed
                else:
                    self._remove_target(path, cancel_check=cancel_check)
                    committed_any = committed_any or not path.exists()
                cleared.append(name)
            except _CacheDirectoryClearInterrupted as exc:
                if not (committed_any or exc.committed):
                    raise Cancelled() from exc
                committed_any = True
                cleared.append(name)
                failures.append(
                    f"{name}: cleanup paused after the cache was detached; choose Clear Rebuildable Caches again to retry."
                )
                break
            except _CacheDirectoryClearFailed as exc:
                if exc.committed:
                    committed_any = True
                    cleared.append(name)
                failures.append(f"{name}: {exc}")
            except OSError as exc:
                failures.append(f"{name}: {exc}")
        self._recreate_directories(exclude=exclude, cancel_check=None if committed_any else cancel_check)
        if progress_callback:
            try:
                progress_callback(100, "Rebuildable cache clear complete")
            except Cancelled:
                if not committed_any:
                    raise
        return tuple(cleared), tuple(failures)

    def pending_rebuildable_cleanup_targets(self) -> tuple[str, ...]:
        pending: list[str] = []
        targets = self.rebuildable_targets()
        for name in self.DIRECTORY_TARGETS:
            path = targets[name]
            staging = self._cache_clear_staging_path(path)
            if staging.exists() or staging.is_symlink():
                pending.append(name)
        return tuple(pending)

    def _clear_rebuildable_directory(self, name: str, path: Path, *, cancel_check=None) -> bool:
        """Atomically detach one cache tree before cancellable physical cleanup."""

        staging = self._cache_clear_staging_path(path)
        committed = bool(staging.exists() and staging.is_dir() and not staging.is_symlink())
        try:
            raise_if_cancelled(cancel_check)
            if staging.is_symlink() or (staging.exists() and not staging.is_dir()):
                raise OSError(f"unsafe pending cache cleanup path: {staging}")
            if staging.exists():
                self._remove_target(staging, cancel_check=cancel_check)
            raise_if_cancelled(cancel_check)
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                return committed
            if path.is_symlink() or not path.is_dir():
                self._remove_target(path, cancel_check=cancel_check)
                return True
            self._cache_directory_clear_checkpoint("before_detach", path, staging)
            os.replace(path, staging)
            committed = True
            self._cache_directory_clear_checkpoint("after_detach", path, staging)
            path.mkdir(parents=True, exist_ok=True)
            self._cache_directory_clear_checkpoint("after_recreate", path, staging)
            self._remove_target(staging, cancel_check=cancel_check)
            return True
        except Cancelled as exc:
            raise _CacheDirectoryClearInterrupted(name, committed=committed) from exc
        except OSError as exc:
            raise _CacheDirectoryClearFailed(name, str(exc), committed=committed) from exc

    @staticmethod
    def _cache_clear_staging_path(path: Path) -> Path:
        return path.parent / f".{path.name}.clusterlens-clear-staging"

    def _cache_directory_clear_checkpoint(self, name: str, source: Path, staging: Path) -> None:
        """Fault-injection seam around the atomic directory detach boundary."""

        _ = (name, source, staging)

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

    def clear_runtime_temp_files(self, *, cancel_check=None) -> tuple[tuple[str, ...], tuple[str, ...]]:
        cleared: list[str] = []
        failures: list[str] = []
        tmp_path = self.rebuildable_targets()["tmp/"]
        try:
            self._remove_target(tmp_path, cancel_check=cancel_check)
            raise_if_cancelled(cancel_check)
            tmp_path.mkdir(parents=True, exist_ok=True)
            cleared.append("tmp/")
        except OSError as exc:
            failures.append(f"tmp/: {exc}")

        # Hugging Face and Torch partial downloads are resumable cache state,
        # not disposable runtime temp files. Preserve them across app restarts.
        return tuple(cleared), tuple(failures)

    def clear_sqlite_cache_bundle(
        self,
        database: str | Path,
        *,
        cancel_check=None,
    ) -> tuple[bool, tuple[str, ...]]:
        """Remove a rebuildable SQLite cache without exposing a torn bundle.

        A best-effort WAL checkpoint makes the main database independently
        usable. WAL/SHM sidecars are removed first; the main file is the commit
        point, so cancellation or process exit before it leaves a usable cache.
        """

        path = Path(database)
        failures: list[str] = []
        raise_if_cancelled(cancel_check)
        if path.exists():
            try:
                connection = sqlite3.connect(str(path), timeout=1.0)
                try:
                    connection.execute("PRAGMA wal_checkpoint(TRUNCATE);").fetchone()
                finally:
                    connection.close()
            except sqlite3.DatabaseError:
                # A corrupt derived cache is still safe to clear. It was not a
                # usable pre-state, so deletion remains the recovery action.
                pass
        for suffix in ("-wal", "-shm"):
            raise_if_cancelled(cancel_check)
            sidecar = path.with_name(path.name + suffix)
            try:
                sidecar.unlink(missing_ok=True)
            except OSError as exc:
                failures.append(f"{sidecar.name}: {exc}")
        raise_if_cancelled(cancel_check)
        self._sqlite_cache_clear_checkpoint("before_remove_main", path)
        removed = False
        try:
            if path.exists():
                path.unlink()
                removed = True
        except OSError as exc:
            failures.append(f"{path.name}: {exc}")
        if removed:
            self._sqlite_cache_clear_checkpoint("after_remove_main", path)
        return removed, tuple(failures)

    def _sqlite_cache_clear_checkpoint(self, name: str, database: Path) -> None:
        """Fault-injection seam around the rebuildable SQLite commit point."""

        _ = (name, database)

    def clear_face_storage_targets(self, *, cancel_check=None) -> tuple[tuple[str, ...], tuple[str, ...]]:
        cleared: list[str] = []
        failures: list[str] = []
        removal_service = FaceStorageRemovalService(self.settings.cache_dir)
        batch_committed = False
        for database in self._face_database_roots():
            if not batch_committed:
                raise_if_cancelled(cancel_check)
            try:
                removal_service.checkpoint_database(database)
                result = removal_service.remove(
                    database,
                    cancel_check=None if batch_committed else cancel_check,
                )
                batch_committed = True
                cleared.extend(self._display_target_name(Path(path)) for path in result.staged_paths)
            except Cancelled:
                raise
            except Exception as exc:
                failures.append(f"{self._display_target_name(database)}: {exc}")
        return tuple(cleared), tuple(failures)

    def clear_model_cache_targets(self, *, cancel_check=None) -> tuple[tuple[str, ...], tuple[str, ...]]:
        cache_dir = self.settings.cache_dir
        paths = (
            cache_dir / "face_model_assets",
            cache_dir / "face_model_state.json",
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
            cancel_check=cancel_check,
        )

    def clear_log_files(self, *, cancel_check=None) -> tuple[tuple[str, ...], tuple[str, ...]]:
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
            cancel_check=cancel_check,
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

    def _face_database_roots(self) -> tuple[Path, ...]:
        candidates = [*self._face_database_paths(), *self._ann_file_paths()]
        roots: set[Path] = set()
        for path in candidates:
            marker = path.name.find(".db")
            if marker < 0:
                continue
            roots.add(path.with_name(path.name[: marker + len(".db")]))
        return tuple(sorted(roots))

    def _library_catalog_paths(self) -> tuple[Path, ...]:
        catalog = self.settings.cache_dir / "library_catalog.sqlite3"
        return (catalog, catalog.with_name(f"{catalog.name}-wal"), catalog.with_name(f"{catalog.name}-shm"))

    def _library_catalog_size(self, *, cancel_check=None) -> int:
        return sum(self._path_size(path, cancel_check=cancel_check) for path in self._library_catalog_paths())

    def _ann_file_paths(self) -> tuple[Path, ...]:
        cache_dir = self.settings.cache_dir
        return tuple(sorted(cache_dir.glob("face_search*.faiss*")))

    def _clear_named_paths(
        self,
        paths,
        *,
        recreate_dirs: tuple[Path, ...],
        cancel_check=None,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        cleared: list[str] = []
        failures: list[str] = []
        for path in tuple(dict.fromkeys(Path(item) for item in paths)):
            raise_if_cancelled(cancel_check)
            try:
                if path.exists():
                    self._remove_target(path, cancel_check=cancel_check)
                    cleared.append(self._display_target_name(path))
            except OSError as exc:
                failures.append(f"{self._display_target_name(path)}: {exc}")
        for path in recreate_dirs:
            raise_if_cancelled(cancel_check)
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

    @staticmethod
    def _runtime_layout_target(runtime_layout, attribute: str) -> Path | None:
        value = getattr(runtime_layout, attribute, None)
        return Path(value) if value is not None else None

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
