from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from infra.settings import AppSettings, get_settings
from infra.cancel import raise_if_cancelled


BUNDLED_ONNX_INPUT_SIZES = {
    "fast_preview": (224, 224),
    "resnet": (224, 224),
    "convnext": (224, 224),
}

BUNDLED_FALLBACK_ORDER = ("fast_preview", "resnet", "convnext")
TEXT_MODEL_ORDER = ("clip", "openclip", "siglip")
DEFAULT_CLUSTER_MEANING_MODEL = "clip"
MODEL_MANIFEST_VERSION = "model-assets-v2"

MODEL_SOURCE_LABELS = {
    "fast_preview": "TorchVision MobileNetV3 weights",
    "resnet": "TorchVision ResNet weights",
    "convnext": "TorchVision ConvNeXt weights",
    "mobileclip": "Apple MobileCLIP weights from Hugging Face/timm",
    "dino": "DINO timm weights",
    "dinov2_base": "DINOv2 Base timm weights",
    "clip": "OpenAI CLIP weights from Hugging Face",
    "openclip": "OpenCLIP LAION weights from Hugging Face",
    "siglip": "Google SigLIP weights from Hugging Face",
    "facenet": "FaceNet VGGFace2 weights from facenet-pytorch",
}

MODEL_SOURCE_URLS = {
    "fast_preview": "https://pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html",
    "resnet": "https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet101.html",
    "convnext": "https://pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny.html",
    "mobileclip": "https://huggingface.co/apple/mobileclip_s0_timm",
    "dino": "https://huggingface.co/timm/vit_small_patch16_224.dino",
    "dinov2_base": "https://huggingface.co/timm/vit_base_patch14_dinov2.lvd142m",
    "clip": "https://huggingface.co/openai/clip-vit-base-patch32",
    "openclip": "https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    "siglip": "https://huggingface.co/google/siglip-base-patch16-224",
    "facenet": "https://github.com/timesler/facenet-pytorch/releases/tag/v2.2.9",
}

# Hugging Face assets used by the managed snapshot downloader must resolve to
# one reviewed revision.  Runtime loading consumes the same mapping so a model,
# processor and tokenizer cannot silently move to different ``main`` commits.
HF_MODEL_REPOSITORIES = {
    "mobileclip": "apple/mobileclip_s0_timm",
    "dino": "timm/vit_small_patch16_224.dino",
    "dinov2_base": "timm/vit_base_patch14_dinov2.lvd142m",
    "clip": "openai/clip-vit-base-patch32",
    "openclip": "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    "siglip": "google/siglip-base-patch16-224",
}

HF_MODEL_REVISIONS = {
    "mobileclip": "7628ba98854d84a318027e036c582df9841c556b",
    "dino": "10e440b8a34dfd657a90f2fcaa988c6b6a2f0da4",
    "dinov2_base": "4685c99dabffe5affac90bd99dbffd25801ae58d",
    "clip": "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
    "openclip": "1a25a446712ba5ee05982a381eed697ef9b435cf",
    "siglip": "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed",
}

MODEL_LICENSES = {
    "fast_preview": "Unresolved for weight redistribution; TorchVision does not declare a weight-specific license.",
    "resnet": "Unresolved for weight redistribution; TorchVision does not declare a weight-specific license.",
    "convnext": "Unresolved for weight redistribution; TorchVision does not declare a weight-specific license.",
    "mobileclip": "Apple Machine Learning Research License (AMLR), per the upstream model card.",
    "dino": "Apache-2.0, per the upstream Hugging Face model card.",
    "dinov2_base": "Apache-2.0, per the upstream Hugging Face model card.",
    "clip": "Unresolved for weight redistribution; the upstream model card has no license identifier.",
    "openclip": "MIT, per the upstream Hugging Face model card; usage and dataset caveats still apply.",
    "siglip": "Apache-2.0, per the upstream Hugging Face model card.",
    "facenet": "Unresolved; facenet-pytorch code and VGGFace2-trained weight terms require review.",
}

TORCH_CACHE_PATTERNS = {
    "fast_preview": ("mobilenet_v3_small*.pth",),
    "resnet": ("resnet101*.pth",),
    "convnext": ("convnext_tiny*.pth",),
    "dino": ("*dino*.pth", "*vit_small_patch16_224_dino*"),
    "dinov2_base": ("*dinov2*base*", "*vit_base_patch14_dinov2*"),
    "facenet": ("20180402-114759-vggface2.pt", "*vggface2*.pt"),
}

HF_CACHE_DIR_NAMES = {
    "dino": ("models--timm--vit_small_patch16_224.dino",),
    "dinov2_base": ("models--timm--vit_base_patch14_dinov2.lvd142m",),
    "clip": ("models--openai--clip-vit-base-patch32",),
    "openclip": ("models--laion--CLIP-ViT-B-32-laion2B-s34B-b79K",),
    "siglip": ("models--google--siglip-base-patch16-224",),
    "mobileclip": ("models--apple--mobileclip_s0_timm",),
}

HF_CACHE_REQUIRED_FILES = {
    "dino": (("model.safetensors", "pytorch_model.bin"),),
    "dinov2_base": (("model.safetensors", "pytorch_model.bin"),),
    "clip": (
        ("config.json",),
        ("preprocessor_config.json", "processor_config.json"),
        ("model.safetensors", "pytorch_model.bin"),
    ),
    "openclip": (
        ("config.json",),
        ("preprocessor_config.json", "processor_config.json"),
        ("model.safetensors", "pytorch_model.bin", "open_clip_pytorch_model.bin"),
    ),
    "siglip": (
        ("config.json",),
        ("preprocessor_config.json", "processor_config.json"),
        ("model.safetensors", "pytorch_model.bin"),
    ),
    "mobileclip": (
        ("config.json",),
        ("model.safetensors", "pytorch_model.bin", "open_clip_pytorch_model.bin"),
    ),
}

HF_CACHE_TEXT_REQUIRED_FILES = {
    "clip": (("tokenizer.json", "vocab.json"),),
    "openclip": (("tokenizer.json", "vocab.json"),),
    "siglip": (("tokenizer.json", "spiece.model"),),
}


@dataclass(frozen=True)
class ModelAssetBundle:
    model_name: str
    model_path: Path
    metadata_path: Path | None
    input_size: tuple[int, int]
    signature: str
    root: Path | None = None


@dataclass(frozen=True)
class ModelAssetManifestV2:
    model_name: str
    purpose: str
    format: str
    target: str
    min_compute_capability: str
    precision: str
    sha256: str
    size_bytes: int
    source_label: str
    source_url: str
    license: str
    input_size: tuple[int, int] = (224, 224)
    signature: str = ""

    @classmethod
    def from_metadata(cls, model_name: str, metadata: dict[str, object]) -> "ModelAssetManifestV2":
        input_size = _metadata_input_size(metadata) or BUNDLED_ONNX_INPUT_SIZES.get(_normalize_model_name(model_name)) or (224, 224)
        normalized = _normalize_model_name(str(metadata.get("model_name") or model_name))
        return cls(
            model_name=normalized,
            purpose=str(metadata.get("purpose") or "image_embedding"),
            format=str(metadata.get("format") or "onnx"),
            target=str(metadata.get("target") or metadata.get("runtime_target") or "cpu"),
            min_compute_capability=str(metadata.get("min_compute_capability") or ""),
            precision=str(metadata.get("precision") or "fp32"),
            sha256=str(metadata.get("sha256") or metadata.get("model_sha256") or "").strip().lower(),
            size_bytes=int(metadata.get("size_bytes") or 0),
            source_label=str(metadata.get("source_label") or MODEL_SOURCE_LABELS.get(normalized, "External model weights")),
            source_url=str(metadata.get("source_url") or MODEL_SOURCE_URLS.get(normalized, "")),
            license=str(metadata.get("license") or MODEL_LICENSES.get(normalized, "Review upstream model license before distribution.")),
            input_size=input_size,
            signature=str(metadata.get("signature") or f"{normalized}:{input_size[0]}x{input_size[1]}:onnx_asset"),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "manifest_version": MODEL_MANIFEST_VERSION,
            "model_name": self.model_name,
            "purpose": self.purpose,
            "format": self.format,
            "target": self.target,
            "min_compute_capability": self.min_compute_capability,
            "precision": self.precision,
            "sha256": self.sha256,
            "size_bytes": int(self.size_bytes),
            "source_label": self.source_label,
            "source_url": self.source_url,
            "license": self.license,
            "input_size": [int(self.input_size[0]), int(self.input_size[1])],
            "signature": self.signature,
        }


@dataclass(frozen=True)
class ModelInventoryItem:
    model_name: str
    source_label: str
    source_url: str
    license: str
    packaged: bool
    cached: bool
    available_without_download: bool
    size_bytes: int
    validation_status: str
    validation_message: str
    model_path: str


@dataclass(frozen=True)
class ModelDownloadPlan:
    requested_models: tuple[str, ...]
    runnable_without_download: tuple[str, ...]
    models_requiring_download: tuple[str, ...]
    meaning_model: str | None
    meaning_requires_download: bool
    bundled_fallback_model: str | None
    bundled_models: tuple[str, ...]

    @property
    def requires_download(self) -> bool:
        return bool(self.models_requiring_download or self.meaning_requires_download)

    @property
    def unavailable_items(self) -> tuple[str, ...]:
        items = list(self.models_requiring_download)
        if (
            self.meaning_requires_download
            and self.meaning_model
            and self.meaning_model not in self.models_requiring_download
        ):
            items.append(f"{self.meaning_model} (advanced cluster naming sidecar)")
        return tuple(items)


class ModelAssetService:
    def __init__(
        self,
        *,
        runtime_model_assets_dir: str | Path | None = None,
        settings: AppSettings | None = None,
        extra_roots: tuple[str | Path, ...] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.runtime_model_assets_dir = Path(runtime_model_assets_dir) if runtime_model_assets_dir else None
        self.extra_roots = tuple(Path(root) for root in (extra_roots or ()))
        self._include_default_roots = extra_roots is None

    def build_download_plan(
        self,
        selected_models: list[str] | tuple[str, ...],
        *,
        generate_cluster_meanings: bool,
        requested_meaning_model: str = "auto",
    ) -> ModelDownloadPlan:
        requested = _normalize_model_list(selected_models)
        requiring_download = tuple(model for model in requested if not self.model_available_without_download(model))
        runnable = tuple(model for model in requested if model not in requiring_download)

        meaning_model: str | None = None
        meaning_requires_download = False
        if generate_cluster_meanings:
            meaning_model = self.resolve_meaning_model(requested_meaning_model, requested)
            meaning_requires_download = not self.model_available_without_download(meaning_model, require_text=True)

        bundled = self.bundled_model_names(valid_only=True)
        fallback = self.choose_bundled_fallback_model()
        return ModelDownloadPlan(
            requested_models=requested,
            runnable_without_download=runnable,
            models_requiring_download=requiring_download,
            meaning_model=meaning_model,
            meaning_requires_download=meaning_requires_download,
            bundled_fallback_model=fallback,
            bundled_models=bundled,
        )

    def model_available_without_download(self, model_name: str, *, require_text: bool = False) -> bool:
        normalized = _normalize_model_name(model_name)
        if self.find_bundle(normalized) is not None:
            return not require_text
        return self.local_cache_present(normalized, require_text=require_text)

    def choose_bundled_fallback_model(self) -> str | None:
        bundled = set(self.bundled_model_names(valid_only=True))
        for model_name in BUNDLED_FALLBACK_ORDER:
            if model_name in bundled:
                return model_name
        return next(iter(sorted(bundled)), None)

    def bundled_model_names(self, *, valid_only: bool = True) -> tuple[str, ...]:
        names: set[str] = set()
        for root in self.candidate_roots():
            if not root.exists():
                continue
            for child in root.iterdir():
                if child.is_dir() and (child / "model.onnx").exists():
                    if valid_only:
                        bundle = self._bundle_from_model_dir(root, child.name.strip().lower())
                        if bundle is None:
                            continue
                        status, _message = self.validate_bundle(bundle)
                        if status != "valid":
                            continue
                    names.add(child.name.strip().lower())
        return tuple(sorted(names))

    def find_bundle(self, model_name: str) -> ModelAssetBundle | None:
        bundle = self.inspect_bundle(model_name)
        if bundle is None:
            return None
        status, _message = self.validate_bundle(bundle)
        if status != "valid":
            return None
        return bundle

    def inspect_bundle(self, model_name: str) -> ModelAssetBundle | None:
        normalized = _normalize_model_name(model_name)
        for root in self.candidate_roots():
            bundle = self._bundle_from_model_dir(root, normalized)
            if bundle is None:
                continue
            return bundle
        return None

    def validate_bundle(self, bundle: ModelAssetBundle) -> tuple[str, str]:
        metadata = _read_metadata(bundle.metadata_path) if bundle.metadata_path else {}
        expected_sha = str(metadata.get("sha256") or metadata.get("model_sha256") or "").strip().lower()
        if not expected_sha and bundle.root is not None:
            manifest_metadata = self._manifest_metadata(bundle.root, bundle.model_name)
            expected_sha = str(
                manifest_metadata.get("sha256") or manifest_metadata.get("model_sha256") or ""
            ).strip().lower()
        if not expected_sha:
            return "unverified", "No checksum was found in metadata.json or manifest.json."
        actual_sha = sha256_file(bundle.model_path)
        if actual_sha.lower() != expected_sha:
            return "invalid", f"Checksum mismatch: expected {expected_sha}, got {actual_sha}."
        return "valid", f"Checksum verified: {actual_sha}."

    def model_inventory(
        self,
        model_names: tuple[str, ...] | list[str],
        *,
        progress_callback=None,
        cancel_check=None,
    ) -> tuple[ModelInventoryItem, ...]:
        items: list[ModelInventoryItem] = []
        normalized_models = _normalize_model_list(tuple(model_names))
        total = max(1, len(normalized_models))
        for index, model_name in enumerate(normalized_models):
            raise_if_cancelled(cancel_check)
            if progress_callback:
                progress_callback(int(index * 100 / total), f"Checking {model_name} model assets")
            bundle = self.inspect_bundle(model_name)
            packaged = bundle is not None
            cached = self.local_cache_present(model_name)
            status = "not_packaged"
            message = "No packaged ONNX asset found."
            model_path = ""
            size_bytes = self.local_cache_size(model_name)
            if bundle is not None:
                status, message = self.validate_bundle(bundle)
                model_path = str(bundle.model_path)
                size_bytes += _path_size(bundle.model_path)
            items.append(
                ModelInventoryItem(
                    model_name=model_name,
                    source_label=MODEL_SOURCE_LABELS.get(model_name, "External model weights"),
                    source_url=MODEL_SOURCE_URLS.get(model_name, ""),
                    license=MODEL_LICENSES.get(model_name, "Review upstream model license before distribution."),
                    packaged=bool(packaged and status == "valid"),
                    cached=bool(cached),
                    available_without_download=bool((packaged and status == "valid") or cached),
                    size_bytes=size_bytes,
                    validation_status=status,
                    validation_message=message,
                    model_path=model_path,
                )
            )
        if progress_callback:
            progress_callback(100, "Model inventory ready")
        return tuple(items)

    def local_cache_present(self, model_name: str, *, require_text: bool = False) -> bool:
        normalized = _normalize_model_name(model_name)
        cache_dir = Path(self.settings.cache_dir)
        if any(_cache_file_present(path) for path in _torch_cache_paths(cache_dir, normalized)):
            return True

        hf_hub_dir = cache_dir / "huggingface" / "hub"
        for directory_name in HF_CACHE_DIR_NAMES.get(normalized, ()):
            required_groups = HF_CACHE_REQUIRED_FILES.get(normalized, ())
            if require_text:
                required_groups = required_groups + HF_CACHE_TEXT_REQUIRED_FILES.get(normalized, ())
            if _hf_cache_snapshot_complete(hf_hub_dir / directory_name, required_groups):
                return True
        return False

    def local_cache_size(self, model_name: str) -> int:
        normalized = _normalize_model_name(model_name)
        cache_dir = Path(self.settings.cache_dir)
        total = 0
        for path in _torch_cache_paths(cache_dir, normalized):
            total += _path_size(path)
        hf_hub_dir = cache_dir / "huggingface" / "hub"
        for directory_name in HF_CACHE_DIR_NAMES.get(normalized, ()):
            total += _path_size(hf_hub_dir / directory_name)
        return total

    def local_cache_revision(self, model_name: str, *, require_text: bool = False) -> str:
        """Return a stable revision token for cache-key invalidation."""
        normalized = _normalize_model_name(model_name)
        bundle = self.find_bundle(normalized)
        if bundle is not None and not require_text:
            return str(bundle.signature)

        cache_dir = Path(self.settings.cache_dir)
        hf_hub_dir = cache_dir / "huggingface" / "hub"
        for directory_name in HF_CACHE_DIR_NAMES.get(normalized, ()):
            repo_dir = hf_hub_dir / directory_name
            required_groups = HF_CACHE_REQUIRED_FILES.get(normalized, ())
            if require_text:
                required_groups = required_groups + HF_CACHE_TEXT_REQUIRED_FILES.get(normalized, ())
            snapshot = _complete_hf_cache_snapshot(repo_dir, required_groups)
            if snapshot is not None:
                return f"hf:{directory_name}:{snapshot.name}"

        for path in _torch_cache_paths(cache_dir, normalized):
            if not _cache_file_present(path):
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            return f"torch:{path.name}:{int(stat.st_size)}:{int(stat.st_mtime_ns)}"
        return ""

    def delete_cached_model(
        self,
        model_name: str,
        *,
        progress_callback=None,
        cancel_check=None,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        normalized = _normalize_model_name(model_name)
        removed: list[str] = []
        failures: list[str] = []
        cache_dir = Path(self.settings.cache_dir)
        for path in _torch_cache_paths(cache_dir, normalized):
            raise_if_cancelled(cancel_check)
            try:
                path.unlink(missing_ok=True)
                removed.append(str(path))
            except OSError as exc:
                failures.append(f"{path}: {exc}")
        hf_hub_dir = cache_dir / "huggingface" / "hub"
        for directory_name in HF_CACHE_DIR_NAMES.get(normalized, ()):
            raise_if_cancelled(cancel_check)
            path = hf_hub_dir / directory_name
            try:
                if path.exists():
                    if progress_callback:
                        progress_callback(-1, f"Deleting cached {normalized} model files")
                    _remove_tree_cancellable(path, cancel_check=cancel_check)
                    removed.append(str(path))
            except OSError as exc:
                failures.append(f"{path}: {exc}")
        if progress_callback:
            progress_callback(100, f"Cached {normalized} files removed")
        return tuple(removed), tuple(failures)

    def candidate_roots(self) -> tuple[Path, ...]:
        roots: list[Path] = []
        if self._include_default_roots:
            env_root = os.environ.get("IMAGE_CLUSTERING_MODEL_ASSETS_DIR")
            if env_root:
                roots.append(Path(env_root))
        roots.extend(self.extra_roots)
        if self.runtime_model_assets_dir is not None:
            roots.append(self.runtime_model_assets_dir)
        try:
            roots.append(Path(self.settings.base_dir) / "model_assets")
        except Exception:
            pass
        if self._include_default_roots:
            meipass = getattr(sys, "_MEIPASS", "")
            if meipass:
                roots.append(Path(meipass) / "model_assets")
            if getattr(sys, "frozen", False):
                roots.append(Path(sys.executable).resolve().parent / "model_assets")
            roots.append(Path(__file__).resolve().parents[3] / "build" / "model_assets")
        return _dedupe_paths(roots)

    @staticmethod
    def resolve_meaning_model(requested_model: str, selected_models: tuple[str, ...]) -> str:
        requested = _normalize_model_name(requested_model)
        if requested in TEXT_MODEL_ORDER:
            return requested
        for model_name in selected_models:
            if model_name in TEXT_MODEL_ORDER:
                return model_name
        return DEFAULT_CLUSTER_MEANING_MODEL

    def _bundle_from_model_dir(self, root: Path, normalized: str) -> ModelAssetBundle | None:
        model_dir = root / normalized
        model_path = model_dir / "model.onnx"
        if not model_path.exists():
            return None
        metadata_path = model_dir / "metadata.json"
        metadata = _read_metadata(metadata_path)
        input_size = _metadata_input_size(metadata) or BUNDLED_ONNX_INPUT_SIZES.get(normalized) or (224, 224)
        signature = str(metadata.get("signature") or f"{normalized}:{input_size[0]}x{input_size[1]}:onnx_asset")
        return ModelAssetBundle(
            model_name=normalized,
            model_path=model_path,
            metadata_path=metadata_path if metadata_path.exists() else None,
            input_size=input_size,
            signature=signature,
            root=root,
        )

    @staticmethod
    def _manifest_metadata(root: Path, model_name: str) -> dict[str, object]:
        manifest = _read_metadata(root / "manifest.json")
        bundles = manifest.get("bundles")
        if not isinstance(bundles, list):
            return {}
        for item in bundles:
            if not isinstance(item, dict):
                continue
            if _normalize_model_name(str(item.get("model_name") or item.get("model") or "")) == model_name:
                return item
        return {}


def _normalize_model_name(model_name: str) -> str:
    return str(model_name or "").strip().lower()


def _normalize_model_list(model_names: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    normalized: list[str] = []
    for model_name in model_names:
        cleaned = _normalize_model_name(model_name)
        if cleaned and cleaned not in normalized:
            normalized.append(cleaned)
    return tuple(normalized)


def _dedupe_paths(paths: list[Path]) -> tuple[Path, ...]:
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        try:
            key = str(path.resolve()).casefold()
        except OSError:
            key = str(path.absolute()).casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return tuple(deduped)


def _read_metadata(metadata_path: Path) -> dict[str, object]:
    if not metadata_path.exists():
        return {}
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _metadata_input_size(metadata: dict[str, object]) -> tuple[int, int] | None:
    raw = metadata.get("input_size")
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return None
    try:
        return int(raw[0]), int(raw[1])
    except (TypeError, ValueError):
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remove_tree_cancellable(path: Path, *, cancel_check=None) -> None:
    raise_if_cancelled(cancel_check)
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


def _path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file() or path.is_symlink():
        try:
            return int(path.lstat().st_size)
        except OSError:
            return 0
    total = 0
    for child in path.rglob("*"):
        try:
            if child.is_file() or child.is_symlink():
                total += int(child.lstat().st_size)
        except OSError:
            continue
    return total


def _cache_file_present(path: Path) -> bool:
    """Return True only for a readable, non-empty cache artifact.

    Hugging Face snapshots commonly use symlinks, so stat() intentionally
    follows them and rejects broken links and zero-byte interrupted writes.
    """
    try:
        return path.is_file() and int(path.stat().st_size) > 0
    except OSError:
        return False


def _torch_cache_paths(cache_dir: Path, model_name: str) -> tuple[Path, ...]:
    """Return checkpoints written by both Torch Hub and facenet-pytorch.

    TorchVision writes below ``TORCH_HOME/hub/checkpoints`` while
    facenet-pytorch writes below ``TORCH_HOME/checkpoints``. Both locations
    belong to the same managed cache and must participate in readiness,
    accounting, revision, and deletion checks.
    """
    paths: dict[str, Path] = {}
    checkpoint_dirs = (
        Path(cache_dir) / "torch" / "checkpoints",
        Path(cache_dir) / "torch" / "hub" / "checkpoints",
    )
    for checkpoint_dir in checkpoint_dirs:
        for pattern in TORCH_CACHE_PATTERNS.get(model_name, ()):
            for path in checkpoint_dir.glob(pattern):
                paths[str(path)] = path
    return tuple(sorted(paths.values(), key=lambda item: (item.name.casefold(), str(item))))


def _hf_cache_snapshot_complete(repo_dir: Path, required_groups: tuple[tuple[str, ...], ...]) -> bool:
    return _complete_hf_cache_snapshot(repo_dir, required_groups) is not None


def _complete_hf_cache_snapshot(
    repo_dir: Path,
    required_groups: tuple[tuple[str, ...], ...],
) -> Path | None:
    if not required_groups or not repo_dir.exists():
        return None
    candidates: list[Path] = []
    snapshots_dir = repo_dir / "snapshots"
    if snapshots_dir.exists():
        preferred_revision = ""
        try:
            preferred_revision = (repo_dir / "refs" / "main").read_text(encoding="utf-8").strip()
        except OSError:
            preferred_revision = ""
        preferred_path = snapshots_dir / preferred_revision if preferred_revision else None
        if preferred_path is not None and preferred_path.is_dir():
            candidates.append(preferred_path)
        remaining = [path for path in snapshots_dir.iterdir() if path.is_dir() and path not in candidates]
        remaining.sort(key=lambda path: path.name, reverse=True)
        candidates.extend(remaining)
    if not candidates:
        candidates.append(repo_dir)
    for root in candidates:
        if all(any(_cache_file_present(root / relative_path) for relative_path in group) for group in required_groups):
            return root
    return None
