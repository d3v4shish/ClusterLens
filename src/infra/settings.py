from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class AppSettings:
    app_name: str = "ClusterLens"
    recursive_scan: bool = True
    image_extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    default_model: str = "siglip"
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
        override = runtime_root_override()
        if override:
            dirs.append(Path(override))
        dirs.extend(_platform_base_dirs(self.app_name))
        dirs.append(Path.cwd() / ".runtime" / self.app_name)
        return dirs


SettingValueType = Literal["bool", "int", "float", "str"]


@dataclass(frozen=True)
class SettingSpec:
    key: str
    value_type: SettingValueType
    default: object
    description: str
    choices: tuple[object, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    user_data: bool = False

    def validate(self, value: object) -> object:
        typed = self._coerce(value)
        if self.choices and typed not in self.choices:
            return self.default
        if self.value_type in {"int", "float"}:
            numeric = float(typed)
            if self.minimum is not None and numeric < self.minimum:
                typed = int(self.minimum) if self.value_type == "int" else float(self.minimum)
            if self.maximum is not None and numeric > self.maximum:
                typed = int(self.maximum) if self.value_type == "int" else float(self.maximum)
        return typed

    @property
    def qt_type(self):
        return {"bool": bool, "int": int, "float": float, "str": str}[self.value_type]

    def _coerce(self, value: object) -> object:
        if value is None:
            return self.default
        if self.value_type == "bool":
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"1", "true", "yes", "on"}:
                    return True
                if normalized in {"0", "false", "no", "off"}:
                    return False
            return bool(value)
        if self.value_type == "int":
            try:
                return int(value)
            except (TypeError, ValueError):
                return int(self.default)
        if self.value_type == "float":
            try:
                return float(value)
            except (TypeError, ValueError):
                return float(self.default)
        return str(value or "")


class SettingsRegistry:
    def __init__(self, specs: tuple[SettingSpec, ...]) -> None:
        self._specs = {spec.key: spec for spec in specs}

    def spec(self, key: str) -> SettingSpec | None:
        return self._specs.get(str(key))

    def keys(self) -> tuple[str, ...]:
        return tuple(self._specs)

    def user_data_keys(self) -> tuple[str, ...]:
        return tuple(key for key, spec in self._specs.items() if spec.user_data)

    def default(self, key: str, fallback: object | None = None) -> object:
        spec = self.spec(key)
        if spec is None:
            return fallback
        return spec.default

    def validate(self, key: str, value: object, fallback: object | None = None) -> object:
        spec = self.spec(key)
        if spec is None:
            return fallback if value is None else value
        return spec.validate(value)

    def get(self, store: object, key: str, fallback: object | None = None) -> object:
        spec = self.spec(key)
        default = fallback if fallback is not None else (spec.default if spec is not None else None)
        value_type = spec.qt_type if spec is not None else None
        try:
            if value_type is None:
                raw = store.value(key, default)
            else:
                raw = store.value(key, default, value_type)
        except TypeError:
            raw = store.value(key, default)
        if raw is None:
            raw = default
        return self.validate(key, raw, fallback=default)

    def set(self, store: object, key: str, value: object) -> object:
        validated = self.validate(key, value)
        store.setValue(key, validated)
        return validated

    def validate_values(self, values: dict[str, object]) -> dict[str, object]:
        return {str(key): self.validate(str(key), value) for key, value in values.items()}


PRODUCTION_SETTING_SPECS: tuple[SettingSpec, ...] = (
    SettingSpec("runtime/preferred_mode", "str", "auto", "Runtime backend preference.", choices=("auto", "cuda", "cpu")),
    SettingSpec("runtime/precision", "str", "auto", "Inference precision preference.", choices=("auto", "fp32", "fp16")),
    SettingSpec("runtime/cuda_device", "str", "cuda:0", "CUDA device identifier."),
    SettingSpec("runtime/allow_gpu_warmup", "bool", True, "Allow model warmup after folder changes."),
    SettingSpec("runtime/show_badge", "bool", True, "Show the resolved runtime badge."),
    SettingSpec("performance/profile", "str", "balanced", "Performance profile.", choices=("low_memory", "balanced", "max_speed")),
    SettingSpec("performance/keep_worker_warm", "bool", False, "Keep the clustering worker process warm."),
    SettingSpec("performance/batch_size_cpu", "int", 8, "CPU inference batch size.", minimum=1, maximum=512),
    SettingSpec("performance/batch_size_gpu", "int", 16, "CUDA inference batch size.", minimum=1, maximum=1024),
    SettingSpec("performance/decode_workers", "int", 4, "Bounded image decode worker count.", minimum=1, maximum=128),
    SettingSpec("performance/preprocess_workers", "int", 4, "Bounded preprocessing worker count.", minimum=1, maximum=128),
    SettingSpec("performance/vram_headroom_mb", "int", 1024, "Reserved CUDA VRAM headroom before batch selection.", minimum=256, maximum=65536),
    SettingSpec("gallery/thumbnail_size", "int", 250, "Gallery thumbnail size.", minimum=96, maximum=512),
    SettingSpec("gallery/thumbnail_workers", "int", 2, "Thumbnail worker count.", minimum=1, maximum=64),
    SettingSpec("gallery/prefetch_rows", "int", 4, "Gallery prefetch rows.", minimum=0, maximum=128),
    SettingSpec("gallery/confirm_large_runs", "bool", True, "Require confirmation before large gallery/clustering runs."),
    SettingSpec("gallery/operation_journal_retention_days", "int", 0, "Operation journal retention; zero means keep indefinitely.", minimum=0, maximum=3650, user_data=True),
    SettingSpec("workspace/default_view", "str", "gallery", "Default workspace.", choices=("gallery", "clustering", "faces", "names", "tags", "library")),
    SettingSpec("workspace/faces_mode", "str", "basic", "Faces UI density/mode.", choices=("basic", "advanced")),
    SettingSpec("workspace/dense_ui", "bool", True, "Use dense desktop spacing."),
    SettingSpec("workspace/recent_folders_v1", "str", "", "Versioned recent-folder history.", user_data=True),
    SettingSpec("library/vision_provider", "str", "ollama", "Cluster-context vision provider.", choices=("ollama", "openai-compatible")),
    SettingSpec("library/vision_model", "str", "", "Cluster-context vision model name."),
    SettingSpec("library/vision_endpoint", "str", "http://127.0.0.1:11434", "Local Ollama URL or explicitly configured remote endpoint."),
    SettingSpec("library/vision_api_key_environment", "str", "CLUSTERLENS_LLM_API_KEY", "Environment variable holding the remote vision API key."),
    SettingSpec("library/auto_cluster_context", "bool", False, "Automatically describe completed clusters after explicit opt-in."),
    SettingSpec("models/offline_mode", "bool", False, "Use only bundled or already cached model files."),
    SettingSpec("models/download_default_after_setup", "bool", False, "Download the default model after first-run setup."),
    SettingSpec("models/asset_root", "str", "", "External model asset root."),
    SettingSpec("faces/model_root", "str", "", "External human face model root."),
    SettingSpec("faces/default_detector/human", "str", "scrfd_10g_kps", "Default human face detector.", user_data=True),
    SettingSpec("faces/default_embedder/human", "str", "arcface_r100_glint360k", "Default human face embedder.", user_data=True),
    SettingSpec("faces/detector_score_threshold/human", "float", 0.35, "Human face detector score threshold.", minimum=0.0, maximum=1.0, user_data=True),
    SettingSpec("faces/max_detections/human", "int", 50, "Maximum human face detections per image.", minimum=1, maximum=1000, user_data=True),
    SettingSpec("faces/pipeline_preferences/human", "str", "", "Applied advanced human face-pipeline preferences.", user_data=True),
    SettingSpec("faces/identity_similarity_threshold", "float", 0.72, "Face identity similarity threshold.", minimum=0.0, maximum=1.0, user_data=True),
    SettingSpec("faces/pending_accept_threshold", "float", 0.85, "Pending face proposal accept threshold.", minimum=0.0, maximum=1.0, user_data=True),
    SettingSpec("clustering/default_model", "str", "siglip", "Default embedding model."),
    SettingSpec("clustering/default_backend", "str", "cosine-kmeans", "Default clustering backend."),
    SettingSpec("clustering/default_similarity_mode", "str", "semantic", "Default similarity mode."),
    SettingSpec("clustering/default_outlier_policy", "str", "assign", "Default outlier policy."),
    SettingSpec("clustering/default_cluster_count", "int", 12, "Default cluster count.", minimum=2, maximum=10000),
    SettingSpec("storage/thumbnail_cache_max_bytes", "int", 536870912, "Thumbnail cache byte budget.", minimum=16777216, maximum=137438953472),
    SettingSpec("storage/embedding_memory_cache_size", "int", 2048, "Embedding memory cache entries.", minimum=0, maximum=1000000),
    SettingSpec("storage/rebuildable_cache_max_bytes", "int", 0, "Rebuildable cache byte budget; zero means unlimited.", minimum=0, maximum=1099511627776),
    SettingSpec("updates/checks_enabled", "bool", False, "Enable update checks."),
    SettingSpec("updates/channel", "str", "stable", "Update channel.", choices=("stable", "beta", "nightly")),
)


_PRODUCTION_SETTINGS_REGISTRY = SettingsRegistry(PRODUCTION_SETTING_SPECS)


def get_production_settings_registry() -> SettingsRegistry:
    return _PRODUCTION_SETTINGS_REGISTRY


_SETTINGS = AppSettings()
_RUNTIME_BASE_DIR: Path | None = None
CLUSTERLENS_RUNTIME_ROOT_ENV = "CLUSTERLENS_RUNTIME_ROOT"
LEGACY_RUNTIME_ROOT_ENV = "IMAGE_CLUSTERING_APP_DIR"


def runtime_root_override() -> str:
    """Return the canonical runtime override, preserving the legacy alias."""
    return str(
        os.environ.get(CLUSTERLENS_RUNTIME_ROOT_ENV)
        or os.environ.get(LEGACY_RUNTIME_ROOT_ENV)
        or ""
    ).strip()


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


def configure_model_cache_environment(settings: AppSettings | object | None = None) -> Path:
    """Bind third-party model caches to the active ClusterLens runtime cache.

    This runs before Hugging Face or Torch download APIs are imported, so local,
    worker, and packaged processes all resolve exactly the same directories.
    """
    active_settings = settings or get_settings()
    cache_dir = Path(active_settings.cache_dir)
    hf_home = cache_dir / "huggingface"
    torch_home = cache_dir / "torch"
    for path in (hf_home, hf_home / "hub", torch_home, cache_dir / "tmp"):
        path.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_home / "hub")
    os.environ["HF_HUB_CACHE"] = str(hf_home / "hub")
    os.environ.pop("TRANSFORMERS_CACHE", None)
    os.environ["TORCH_HOME"] = str(torch_home)
    return cache_dir


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
