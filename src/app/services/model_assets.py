from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from infra.settings import AppSettings, get_settings


BUNDLED_ONNX_INPUT_SIZES = {
    "fast_preview": (224, 224),
    "resnet": (224, 224),
    "convnext": (224, 224),
}

BUNDLED_FALLBACK_ORDER = ("fast_preview", "resnet", "convnext")
TEXT_MODEL_ORDER = ("clip", "openclip", "siglip")
DEFAULT_CLUSTER_MEANING_MODEL = "clip"
MODEL_MANIFEST_VERSION = "model-assets-v1"

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
}

MODEL_LICENSES = {
    "fast_preview": "TorchVision model weights license; verify with packaged metadata before distribution.",
    "resnet": "TorchVision model weights license; verify with packaged metadata before distribution.",
    "convnext": "TorchVision model weights license; verify with packaged metadata before distribution.",
    "mobileclip": "Apple MobileCLIP license; verify upstream model card before distribution.",
    "dino": "timm/Hugging Face model card license; verify upstream before distribution.",
    "dinov2_base": "Meta DINOv2 license; verify upstream model card before distribution.",
    "clip": "OpenAI CLIP model card license; verify upstream before distribution.",
    "openclip": "LAION/OpenCLIP model card license; verify upstream before distribution.",
    "siglip": "Google SigLIP model card license; verify upstream before distribution.",
}

TORCH_CACHE_PATTERNS = {
    "fast_preview": ("mobilenet_v3_small*.pth",),
    "resnet": ("resnet101*.pth",),
    "convnext": ("convnext_tiny*.pth",),
    "dino": ("*dino*.pth", "*vit_small_patch16_224_dino*"),
    "dinov2_base": ("*dinov2*base*", "*vit_base_patch14_dinov2*"),
}

HF_CACHE_DIR_NAMES = {
    "clip": ("models--openai--clip-vit-base-patch32",),
    "openclip": ("models--laion--CLIP-ViT-B-32-laion2B-s34B-b79K",),
    "siglip": ("models--google--siglip-base-patch16-224",),
    "mobileclip": ("models--apple--mobileclip_s0_timm",),
}

HF_CACHE_REQUIRED_FILES = {
    "clip": (
        ("config.json",),
        ("preprocessor_config.json", "processor_config.json"),
        ("tokenizer.json", "vocab.json"),
        ("model.safetensors", "pytorch_model.bin"),
    ),
    "openclip": (
        ("config.json",),
        ("preprocessor_config.json", "processor_config.json"),
        ("tokenizer.json", "vocab.json"),
        ("model.safetensors", "pytorch_model.bin", "open_clip_pytorch_model.bin"),
    ),
    "siglip": (
        ("config.json",),
        ("preprocessor_config.json", "processor_config.json"),
        ("tokenizer.json", "spiece.model"),
        ("model.safetensors", "pytorch_model.bin"),
    ),
    "mobileclip": (
        ("config.json",),
        ("model.safetensors", "pytorch_model.bin", "open_clip_pytorch_model.bin"),
    ),
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
        if self.meaning_requires_download and self.meaning_model:
            items.append(f"{self.meaning_model} (advanced cluster naming sidecar)")
        return tuple(items)


class ModelAssetService:
    def __init__(
        self,
        *,
        runtime_model_assets_dir: str | Path | None = None,
        settings: AppSettings | None = None,
        extra_roots: tuple[str | Path, ...] = (),
    ) -> None:
        self.settings = settings or get_settings()
        self.runtime_model_assets_dir = Path(runtime_model_assets_dir) if runtime_model_assets_dir else None
        self.extra_roots = tuple(Path(root) for root in extra_roots)

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
            meaning_requires_download = not self.model_available_without_download(meaning_model)

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

    def model_available_without_download(self, model_name: str) -> bool:
        normalized = _normalize_model_name(model_name)
        if self.find_bundle(normalized) is not None:
            return True
        return self.local_cache_present(normalized)

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

    def model_inventory(self, model_names: tuple[str, ...] | list[str]) -> tuple[ModelInventoryItem, ...]:
        items: list[ModelInventoryItem] = []
        for model_name in _normalize_model_list(tuple(model_names)):
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
        return tuple(items)

    def local_cache_present(self, model_name: str) -> bool:
        normalized = _normalize_model_name(model_name)
        cache_dir = Path(self.settings.cache_dir)
        torch_checkpoint_dir = cache_dir / "torch" / "hub" / "checkpoints"
        for pattern in TORCH_CACHE_PATTERNS.get(normalized, ()):
            if any(torch_checkpoint_dir.glob(pattern)):
                return True

        hf_hub_dir = cache_dir / "huggingface" / "hub"
        for directory_name in HF_CACHE_DIR_NAMES.get(normalized, ()):
            required_groups = HF_CACHE_REQUIRED_FILES.get(normalized, ())
            if _hf_cache_snapshot_complete(hf_hub_dir / directory_name, required_groups):
                return True
        return False

    def local_cache_size(self, model_name: str) -> int:
        normalized = _normalize_model_name(model_name)
        cache_dir = Path(self.settings.cache_dir)
        total = 0
        torch_checkpoint_dir = cache_dir / "torch" / "hub" / "checkpoints"
        for pattern in TORCH_CACHE_PATTERNS.get(normalized, ()):
            for path in torch_checkpoint_dir.glob(pattern):
                total += _path_size(path)
        hf_hub_dir = cache_dir / "huggingface" / "hub"
        for directory_name in HF_CACHE_DIR_NAMES.get(normalized, ()):
            total += _path_size(hf_hub_dir / directory_name)
        return total

    def delete_cached_model(self, model_name: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
        normalized = _normalize_model_name(model_name)
        removed: list[str] = []
        failures: list[str] = []
        cache_dir = Path(self.settings.cache_dir)
        torch_checkpoint_dir = cache_dir / "torch" / "hub" / "checkpoints"
        for pattern in TORCH_CACHE_PATTERNS.get(normalized, ()):
            for path in torch_checkpoint_dir.glob(pattern):
                try:
                    path.unlink(missing_ok=True)
                    removed.append(str(path))
                except OSError as exc:
                    failures.append(f"{path}: {exc}")
        hf_hub_dir = cache_dir / "huggingface" / "hub"
        for directory_name in HF_CACHE_DIR_NAMES.get(normalized, ()):
            path = hf_hub_dir / directory_name
            try:
                if path.exists():
                    shutil.rmtree(path)
                    removed.append(str(path))
            except OSError as exc:
                failures.append(f"{path}: {exc}")
        return tuple(removed), tuple(failures)

    def candidate_roots(self) -> tuple[Path, ...]:
        roots: list[Path] = []
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


def _hf_cache_snapshot_complete(repo_dir: Path, required_groups: tuple[tuple[str, ...], ...]) -> bool:
    if not required_groups or not repo_dir.exists():
        return False
    candidates: list[Path] = []
    snapshots_dir = repo_dir / "snapshots"
    if snapshots_dir.exists():
        for path in snapshots_dir.iterdir():
            if path.is_dir():
                candidates.append(path)
    if not candidates:
        candidates.append(repo_dir)
    for root in candidates:
        if all(any((root / relative_path).exists() for relative_path in group) for group in required_groups):
            return True
    return False
