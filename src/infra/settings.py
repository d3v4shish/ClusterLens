from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppSettings:
    app_name: str = "ClusterLens"
    recursive_scan: bool = True
    image_extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    default_model: str = "dino"
    default_cluster_backend: str = "cosine-kmeans"
    default_similarity_mode: str = "semantic"
    default_outlier_policy: str = "assign"
    default_use_onnx: bool = False
    default_reuse_result_cache: bool = True
    default_cluster_count: int = 12
    min_cluster_count: int = 2
    max_thumbnail_workers: int = 2
    thumbnail_size: int = 250
    thumbnail_cache_size: int = 250
    thumbnail_cache_max_bytes: int = 536870912
    thumbnail_prefetch_rows: int = 4
    gallery_flush_interval_ms: int = 33
    embedding_memory_cache_size: int = 2048
    progress_emit_interval_s: float = 0.12
    minibatch_kmeans_threshold: int = 2000
    silhouette_sample_size: int = 2000
    outlier_quantile: float = 0.97
    min_graph_similarity: float = 0.86
    min_graph_cluster_size: int = 2
    tiny_cluster_min_size: int = 2
    batch_size_cpu: int = 8
    batch_size_gpu: int = 16
    preferred_execution_mode: str = "auto"
    default_performance_profile: str = "balanced"
    allow_gpu_warmup: bool = True
    show_runtime_badge: bool = True
    default_workspace_view: str = "clusters"
    default_dense_ui: bool = True

    def preferred_base_dirs(self) -> list[Path]:
        dirs: list[Path] = []
        override = os.environ.get("IMAGE_CLUSTERING_APP_DIR")
        if override:
            dirs.append(Path(override))
        dirs.extend(_platform_base_dirs(self.app_name))
        dirs.append(Path.cwd() / ".runtime" / self.app_name)
        return dirs


_SETTINGS = AppSettings()
_RUNTIME_BASE_DIR: Path | None = None


def _is_writable_runtime_dir(candidate: Path) -> bool:
    try:
        (candidate / "cache" / "thumbnails").mkdir(parents=True, exist_ok=True)
        (candidate / "logs").mkdir(parents=True, exist_ok=True)
        probe = candidate / "cache" / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def get_settings() -> AppSettings:
    global _RUNTIME_BASE_DIR
    if _RUNTIME_BASE_DIR is None:
        for candidate in _SETTINGS.preferred_base_dirs():
            if _is_writable_runtime_dir(candidate):
                _RUNTIME_BASE_DIR = candidate
                break
        if _RUNTIME_BASE_DIR is None:
            raise PermissionError("Unable to create an application data directory.")
    return _SETTINGS


def get_runtime_base_dir() -> Path:
    get_settings()
    return _RUNTIME_BASE_DIR


def _platform_base_dirs(app_name: str) -> list[Path]:
    dirs: list[Path] = []
    if os.name == "nt":
        for env_name in ("LOCALAPPDATA", "APPDATA"):
            env_value = os.environ.get(env_name)
            if env_value:
                dirs.append(Path(env_value) / app_name)
        return dirs

    home = _home_path()
    if sys.platform == "darwin":
        if home is not None:
            dirs.append(home / "Library" / "Application Support" / app_name)
        return dirs

    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    if xdg_data_home:
        dirs.append(Path(xdg_data_home) / app_name)
    elif home is not None:
        dirs.append(home / ".local" / "share" / app_name)
    return dirs


def _home_path() -> Path | None:
    raw_home = os.path.expanduser("~")
    if not raw_home or raw_home == "~":
        return None
    return Path(raw_home)


AppSettings.base_dir = property(lambda self: get_runtime_base_dir())
AppSettings.cache_dir = property(lambda self: self.base_dir / "cache")
AppSettings.thumbnail_cache_dir = property(lambda self: self.cache_dir / "thumbnails")
AppSettings.embedding_cache_db = property(lambda self: self.cache_dir / "embeddings.sqlite3")
AppSettings.image_tags_db = property(lambda self: self.cache_dir / "image_tags.sqlite3")
AppSettings.log_dir = property(lambda self: self.base_dir / "logs")
AppSettings.log_file = property(lambda self: self.log_dir / "cluster_gallery.log")
