from __future__ import annotations

import os
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from PIL import Image

from app.services.model_assets import BUNDLED_ONNX_INPUT_SIZES, ModelAssetService
from app.services.onnx_models import OnnxModelService
from infra.cache import CacheService
from infra.logging_config import get_logger
from infra.performance import PerformanceProfile, select_performance_profile
from infra.runtime import ExecutionPolicy, RuntimeCapabilityService
from infra.settings import get_settings

# Ensure model download caches and temp dirs are writable in this environment.
_RUNTIME_CACHE = get_settings().cache_dir
_HF_HOME = _RUNTIME_CACHE / "huggingface"
_TORCH_HOME = _RUNTIME_CACHE / "torch"
_TMP_HOME = _RUNTIME_CACHE / "tmp"
for _p in (_HF_HOME, _TORCH_HOME, _TMP_HOME):
    _p.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(_HF_HOME)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(_HF_HOME / "hub")
os.environ["HF_HUB_CACHE"] = str(_HF_HOME / "hub")
os.environ.pop("TRANSFORMERS_CACHE", None)
os.environ["TORCH_HOME"] = str(_TORCH_HOME)
os.environ.setdefault("HOME", str(_RUNTIME_CACHE))
os.environ["TEMP"] = str(_TMP_HOME)
os.environ["TMP"] = str(_TMP_HOME)

# Torch compat: some deps (e.g. newer transformers) call torch.compiler.is_compiling(),
# which does not exist on older torch builds like 2.2.x.
try:
    import torch.compiler as _torch_compiler  # type: ignore

    if not hasattr(_torch_compiler, "is_compiling"):
        setattr(_torch_compiler, "is_compiling", lambda: False)
except Exception:
    compiler = getattr(torch, "compiler", None)
    if compiler is not None and not hasattr(compiler, "is_compiling"):
        try:
            setattr(compiler, "is_compiling", lambda: False)
        except Exception:
            pass

LOGGER = get_logger(__name__)


@contextmanager
def _huggingface_offline_context(enabled: bool):
    if not enabled:
        yield
        return
    overrides = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }
    previous = {name: os.environ.get(name) for name in overrides}
    try:
        os.environ.update(overrides)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@dataclass(frozen=True)
class ModelBundle:
    model_name: str
    model: torch.nn.Module
    preprocess: object
    input_size: tuple[int, int]
    signature: str
    family: str
    input_mode: str
    onnx_session: object | None = None
    text_tokenizer: object | None = None


class ProgressThrottler:
    def __init__(self, callback, interval_s: float) -> None:
        self.callback = callback
        self.interval_s = interval_s
        self.last_emit = 0.0

    def emit(self, done: int, total: int, stage: str, force: bool = False) -> None:
        if not self.callback:
            return
        now = time.perf_counter()
        if force or now - self.last_emit >= self.interval_s or done >= total:
            self.callback(done, total, stage)
            self.last_emit = now


class ModelManager:
    def __init__(
        self,
        use_onnx: bool = False,
        execution_policy: ExecutionPolicy | None = None,
        runtime_service: RuntimeCapabilityService | None = None,
        performance_profile: PerformanceProfile | None = None,
        allow_model_downloads: bool = True,
    ) -> None:
        self.settings = get_settings()
        hf_home = self.settings.cache_dir / "huggingface"
        hf_home.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(hf_home))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub"))
        self.runtime_service = runtime_service or RuntimeCapabilityService()
        self.execution_policy = execution_policy or self.runtime_service.select_policy(self.settings.preferred_execution_mode)
        self.performance_profile = performance_profile or select_performance_profile(self.settings.default_performance_profile)
        self.device = torch.device(self.execution_policy.torch_device)
        self.use_onnx = use_onnx
        self.allow_model_downloads = bool(allow_model_downloads)
        self.model_asset_service = ModelAssetService()
        self.onnx_service = OnnxModelService(execution_policy=self.execution_policy, runtime_service=self.runtime_service)
        self._models: dict[tuple[str, bool], ModelBundle] = {}

    def get_bundle(self, model_name: str, use_onnx: bool | None = None) -> ModelBundle:
        use_onnx = self.use_onnx if use_onnx is None else use_onnx
        cache_key = (model_name, bool(use_onnx))
        if cache_key not in self._models:
            self._models[cache_key] = self._load_bundle(model_name, bool(use_onnx))
        return self._models[cache_key]

    @staticmethod
    def static_signature(model_name: str, use_onnx: bool | None = None) -> str | None:
        if bool(use_onnx):
            return None
        signatures = {
            "fast_preview": "fast_preview:224x224:torchvision:onnx=False",
            "resnet": "resnet:224x224:torchvision:onnx=False",
            "clip": "clip:224x224:clip:onnx=False",
            "openclip": "openclip:224x224:openclip:onnx=False",
            "siglip": "siglip:224x224:siglip:onnx=False",
            "dino": "dino:224x224:timm:onnx=False",
            "dinov2_base": "dinov2_base:518x518:timm:onnx=False",
            "dino_large": "dino_large:518x518:timm:onnx=False",
            "mobileclip": "mobileclip:256x256:timm:onnx=False",
        }
        return signatures.get(str(model_name or "").strip().lower())

    def _load_bundle(self, model_name: str, use_onnx: bool) -> ModelBundle:
        base_model_name = "convnext" if model_name == "phash_embedding" else model_name
        input_size = (224, 224)
        family = "torchvision"
        input_mode = "tensor"
        text_tokenizer = None

        asset_bundle = self.model_asset_service.find_bundle(base_model_name)
        if use_onnx and asset_bundle is not None and base_model_name in BUNDLED_ONNX_INPUT_SIZES:
            onnx_session = self.onnx_service.session_from_asset(
                base_model_name,
                asset_bundle.model_path,
                self.execution_policy.onnx_provider,
            )
            if onnx_session is not None:
                input_size = asset_bundle.input_size
                preprocess = self._default_preprocess(input_size)
                signature = asset_bundle.signature
                LOGGER.info("Loaded bundled ONNX model '%s' from %s", model_name, asset_bundle.model_path)
                return ModelBundle(
                    model_name,
                    torch.nn.Identity(),
                    preprocess,
                    input_size,
                    signature,
                    family,
                    input_mode,
                    onnx_session,
                    text_tokenizer,
                )

        if not self.allow_model_downloads and not self.model_asset_service.local_cache_present(base_model_name):
            raise RuntimeError(
                f"Model '{model_name}' is not bundled with this build and downloads are disabled. "
                "Allow the download or choose a bundled model asset."
            )

        if base_model_name == "fast_preview":
            from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

            model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
            model.classifier = torch.nn.Identity()
            preprocess = self._default_preprocess(input_size)
        elif base_model_name == "resnet":
            from torchvision.models import ResNet101_Weights, resnet101

            model = resnet101(weights=ResNet101_Weights.DEFAULT)
            model.fc = torch.nn.Identity()
            preprocess = self._default_preprocess(input_size)
        elif base_model_name == "vgg":
            from torchvision.models import VGG16_Weights, vgg16

            model = vgg16(weights=VGG16_Weights.DEFAULT)
            model.classifier = torch.nn.Sequential(*list(model.classifier.children())[:-1])
            preprocess = self._default_preprocess(input_size)
        elif base_model_name == "convnext":
            from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

            model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
            model.classifier = torch.nn.Identity()
            preprocess = self._default_preprocess(input_size)
        elif base_model_name == "vit":
            from transformers import AutoImageProcessor, ViTModel

            model = ViTModel.from_pretrained(
                "google/vit-base-patch16-224-in21k",
                use_safetensors=True,
                local_files_only=not self.allow_model_downloads,
            )
            preprocess = AutoImageProcessor.from_pretrained(
                "google/vit-base-patch16-224-in21k",
                use_fast=True,
                local_files_only=not self.allow_model_downloads,
            )
            family = "hf-vision"
            input_mode = "processor"
        elif base_model_name == "clip":
            from transformers import AutoImageProcessor, AutoTokenizer, CLIPModel

            model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32",
                use_safetensors=True,
                local_files_only=not self.allow_model_downloads,
            )
            preprocess = AutoImageProcessor.from_pretrained(
                "openai/clip-vit-base-patch32",
                use_fast=True,
                local_files_only=not self.allow_model_downloads,
            )
            text_tokenizer = AutoTokenizer.from_pretrained(
                "openai/clip-vit-base-patch32",
                use_fast=True,
                local_files_only=not self.allow_model_downloads,
            )
            family = "clip"
            input_mode = "processor"
        elif base_model_name == "openclip":
            from transformers import AutoImageProcessor, AutoTokenizer, CLIPModel

            model_id = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
            model = CLIPModel.from_pretrained(model_id, use_safetensors=True, local_files_only=not self.allow_model_downloads)
            preprocess = AutoImageProcessor.from_pretrained(model_id, use_fast=True, local_files_only=not self.allow_model_downloads)
            text_tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, local_files_only=not self.allow_model_downloads)
            family = "openclip"
            input_mode = "processor"
        elif base_model_name == "siglip":
            from transformers import AutoImageProcessor, SiglipModel

            model = SiglipModel.from_pretrained(
                "google/siglip-base-patch16-224",
                use_safetensors=True,
                local_files_only=not self.allow_model_downloads,
            )
            preprocess = AutoImageProcessor.from_pretrained(
                "google/siglip-base-patch16-224",
                use_fast=True,
                local_files_only=not self.allow_model_downloads,
            )
            try:
                from transformers import AutoTokenizer

                text_tokenizer = AutoTokenizer.from_pretrained(
                    "google/siglip-base-patch16-224",
                    use_fast=True,
                    local_files_only=not self.allow_model_downloads,
                )
            except Exception:
                text_tokenizer = None
            family = "siglip"
            input_mode = "processor"
        elif base_model_name == "facenet":
            from facenet_pytorch import InceptionResnetV1

            model = InceptionResnetV1(pretrained="vggface2").eval()
            input_size = (160, 160)
            family = "facenet"
            preprocess = self._facenet_preprocess(input_size)
        elif base_model_name == "dino":
            from timm import create_model

            with _huggingface_offline_context(not self.allow_model_downloads):
                model = create_model("vit_small_patch16_224_dino", pretrained=True, num_classes=0)
            family = "timm"
            preprocess = self._default_preprocess(input_size)
        elif base_model_name == "dino_large":
            from timm import create_model

            input_size = (518, 518)
            with _huggingface_offline_context(not self.allow_model_downloads):
                model = create_model("vit_large_patch14_dinov2", pretrained=True, num_classes=0)
            family = "timm"
            preprocess, input_size = self._timm_preprocess(model, fallback_input_size=input_size)
        elif base_model_name == "dinov2_base":
            from timm import create_model

            with _huggingface_offline_context(not self.allow_model_downloads):
                model = create_model("vit_base_patch14_dinov2", pretrained=True, num_classes=0)
            family = "timm"
            preprocess, input_size = self._timm_preprocess(model, fallback_input_size=(518, 518))
        elif base_model_name == "mobileclip":
            from timm import create_model

            with _huggingface_offline_context(not self.allow_model_downloads):
                model = create_model("hf_hub:apple/mobileclip_s0_timm", pretrained=True, num_classes=0)
            family = "timm"
            preprocess, input_size = self._timm_preprocess(model, fallback_input_size=(256, 256))
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

        model = model.eval()
        onnx_session = None
        if use_onnx and family == "torchvision":
            onnx_session = self.onnx_service.session_for(base_model_name, model, input_size, self.execution_policy.onnx_provider)
        if onnx_session is None:
            model = model.to(self.device).eval()
            device_label = str(self.device)
        else:
            model = model.to("cpu").eval()
            device_label = f"onnx:{self.execution_policy.onnx_provider}"
        signature = f"{model_name}:{input_size[0]}x{input_size[1]}:{family}:onnx={bool(onnx_session)}"
        LOGGER.info("Loaded model '%s' on %s (onnx=%s)", model_name, device_label, bool(onnx_session))
        return ModelBundle(model_name, model, preprocess, input_size, signature, family, input_mode, onnx_session, text_tokenizer)

    def suggested_batch_size(self, model_name: str) -> int:
        if self.device.type != "cuda":
            return self._cpu_batch_size(model_name)
        try:
            free_bytes, _ = torch.cuda.mem_get_info()
        except Exception:
            free_bytes = 2 * 1024 * 1024 * 1024
        per_image_bytes = {
            "dino_large": 96 * 1024 * 1024,
            "dinov2_base": 72 * 1024 * 1024,
            "clip": 32 * 1024 * 1024,
            "openclip": 32 * 1024 * 1024,
            "siglip": 32 * 1024 * 1024,
            "mobileclip": 12 * 1024 * 1024,
            "vit": 48 * 1024 * 1024,
        }.get(model_name, 24 * 1024 * 1024)
        return max(1, min(self._gpu_batch_size(model_name), int((free_bytes * 0.65) // per_image_bytes)))

    def _cpu_batch_size(self, model_name: str) -> int:
        base = self.performance_profile.cpu_batch_size
        if model_name in {"dino_large", "dinov2_base", "vit", "clip", "openclip", "siglip"}:
            return max(1, base // 2)
        return base

    def _gpu_batch_size(self, model_name: str) -> int:
        base = self.performance_profile.gpu_batch_size
        if model_name in {"dino_large", "dinov2_base"}:
            return max(1, base // 2)
        return base

    @staticmethod
    def _timm_preprocess(model: torch.nn.Module, *, fallback_input_size: tuple[int, int]):
        try:
            from timm.data import create_transform, resolve_model_data_config

            data_config = resolve_model_data_config(model)
            transform = create_transform(**data_config, is_training=False)
            configured_size = data_config.get("input_size") or (3, fallback_input_size[0], fallback_input_size[1])
            if len(configured_size) >= 3:
                input_size = (int(configured_size[-2]), int(configured_size[-1]))
            else:
                input_size = fallback_input_size
            return transform, input_size
        except Exception as exc:
            LOGGER.warning("Falling back to default timm preprocessing: %s", exc)
            return ModelManager._default_preprocess(fallback_input_size), fallback_input_size

    @staticmethod
    def _default_preprocess(input_size: tuple[int, int]):
        from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

        return Compose(
            [
                Resize(input_size),
                CenterCrop(input_size),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @staticmethod
    def _facenet_preprocess(input_size: tuple[int, int]):
        from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

        return Compose(
            [
                Resize(input_size),
                CenterCrop(input_size),
                ToTensor(),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )


class EmbeddingService:
    def __init__(
        self,
        model_manager: ModelManager | None = None,
        cache_service: CacheService | None = None,
        performance_profile: PerformanceProfile | None = None,
    ) -> None:
        self.settings = get_settings()
        self.performance_profile = performance_profile or select_performance_profile(self.settings.default_performance_profile)
        self.model_manager = model_manager or ModelManager(performance_profile=self.performance_profile)
        self.cache_service = cache_service or CacheService(memory_cache_size=self.performance_profile.embedding_memory_cache_size)

    def embed_paths(
        self,
        image_paths: Iterable[str],
        model_name: str,
        progress_callback=None,
        use_onnx: bool = False,
        use_cache_lookup: bool = True,
        path_fingerprints: dict[str, tuple[int, int]] | None = None,
    ) -> tuple[list[tuple[str, np.ndarray]], dict]:
        image_paths = list(image_paths)
        static_signature_fn = getattr(self.model_manager, "static_signature", None)
        static_signature = static_signature_fn(model_name, use_onnx) if callable(static_signature_fn) else None
        if use_cache_lookup and static_signature:
            static_cache_keys = self.cache_service.build_embedding_keys(
                image_paths,
                model_name,
                static_signature,
                path_fingerprints=path_fingerprints,
            )
            cache_lookup_start = time.perf_counter()
            static_cached_vectors = self.cache_service.get_embeddings_many(static_cache_keys)
            cache_lookup_time_s = round(time.perf_counter() - cache_lookup_start, 3)
            if len(static_cached_vectors) == len(image_paths):
                finalized = [
                    (image_path, static_cached_vectors[static_cache_keys[index]])
                    for index, image_path in enumerate(image_paths)
                ]
                return finalized, {
                    "cache_hits": len(image_paths),
                    "cache_misses": 0,
                    "batch_size": 0,
                    "model_load_time_s": 0.0,
                    "cache_lookup_time_s": cache_lookup_time_s,
                    "preprocess_time_s": 0.0,
                    "inference_time_s": 0.0,
                    "oom_backoff": False,
                    "model_device": str(self.model_manager.device),
                    "amp_enabled": False,
                    "embedding_backend": "cache",
                    "effective_mode": self.model_manager.execution_policy.effective_mode,
                    "onnx_provider": self.model_manager.execution_policy.onnx_provider,
                    "runtime_reason": "Skipped model load because every embedding was cached.",
                    "embedding_cache_lookup": "enabled",
                }

        model_load_start = time.perf_counter()
        bundle = self.model_manager.get_bundle(model_name, use_onnx=use_onnx)
        model_load_time_s = round(time.perf_counter() - model_load_start, 3)
        batch_size = max(1, self.model_manager.suggested_batch_size(model_name))
        progress = ProgressThrottler(progress_callback, self.settings.progress_emit_interval_s)

        cache_keys = self.cache_service.build_embedding_keys(
            image_paths,
            model_name,
            bundle.signature,
            path_fingerprints=path_fingerprints,
        )
        cached_vectors: dict[str, np.ndarray] = {}
        cache_lookup_time_s = 0.0
        if use_cache_lookup:
            cache_lookup_start = time.perf_counter()
            cached_vectors = self.cache_service.get_embeddings_many(cache_keys)
            cache_lookup_time_s = round(time.perf_counter() - cache_lookup_start, 3)

        results: list[tuple[str, np.ndarray] | None] = [None] * len(image_paths)
        uncached: list[tuple[int, str, str]] = []
        for index, image_path in enumerate(image_paths):
            cache_key = cache_keys[index]
            cached = cached_vectors.get(cache_key)
            if cached is None:
                uncached.append((index, image_path, cache_key))
            else:
                results[index] = (image_path, cached)

        total = len(image_paths)
        done = total - len(uncached)
        if use_cache_lookup:
            progress.emit(done, total, "Embedding cache lookup", force=True)

        pending_writes: list[tuple[str, str, str, np.ndarray]] = []
        preprocess_time_s = 0.0
        inference_time_s = 0.0
        oom_backoff = False
        batches = [uncached[start : start + batch_size] for start in range(0, len(uncached), batch_size)]
        prefetch_workers = max(1, self.performance_profile.embedding_preprocess_workers)

        with ThreadPoolExecutor(max_workers=prefetch_workers) as executor:
            pending: deque[tuple[list[tuple[int, str, str]], Future]] = deque()
            batch_iter = iter(batches)

            def _submit_next() -> None:
                try:
                    next_batch = next(batch_iter)
                except StopIteration:
                    return
                pending.append((next_batch, executor.submit(self._prepare_batch, bundle, next_batch)))

            for _ in range(prefetch_workers):
                _submit_next()

            while pending:
                _batch, future = pending.popleft()
                prepared, kept, prep_elapsed = future.result()
                preprocess_time_s += prep_elapsed
                infer_start = time.perf_counter()
                embeddings, used_backoff = self._run_model_with_backoff(bundle, prepared)
                inference_time_s += time.perf_counter() - infer_start
                oom_backoff = oom_backoff or used_backoff
                done = self._store_batch_results(kept, embeddings, model_name, results, pending_writes, done, total, progress)
                _submit_next()

        self.cache_service.put_embeddings_many(pending_writes)
        finalized = [item for item in results if item is not None]
        return finalized, {
            "cache_hits": total - len(uncached) if use_cache_lookup else 0,
            "cache_misses": len(uncached) if use_cache_lookup else 0,
            "batch_size": batch_size,
            "model_load_time_s": model_load_time_s,
            "cache_lookup_time_s": cache_lookup_time_s,
            "preprocess_time_s": round(preprocess_time_s, 3),
            "inference_time_s": round(inference_time_s, 3),
            "oom_backoff": oom_backoff,
            "model_device": str(self.model_manager.device),
            "amp_enabled": self.model_manager.device.type == "cuda" and bundle.onnx_session is None,
            "embedding_backend": "onnx" if bundle.onnx_session is not None else "torch",
            "effective_mode": self.model_manager.execution_policy.effective_mode,
            "onnx_provider": self.model_manager.execution_policy.onnx_provider,
            "runtime_reason": self.model_manager.execution_policy.reason,
            "embedding_cache_lookup": "enabled" if use_cache_lookup else "disabled",
        }

    def embed_text(self, query_text: str, model_name: str = "clip", use_onnx: bool = False) -> np.ndarray:
        return self.embed_texts([query_text], model_name=model_name, use_onnx=use_onnx)[0]

    def embed_texts(self, query_texts: list[str] | tuple[str, ...], model_name: str = "clip", use_onnx: bool = False) -> list[np.ndarray]:
        bundle = self.model_manager.get_bundle(model_name, use_onnx=use_onnx)
        if bundle.family not in {"clip", "openclip", "siglip"}:
            raise ValueError(f"Text embedding is only supported for CLIP/OpenCLIP/SigLIP models, got {model_name}")
        if bundle.text_tokenizer is None:
            raise ValueError("Text embedding for SigLIP requires the tokenizer dependencies (install `sentencepiece`) or use CLIP.")
        texts = [str(text) for text in query_texts]
        if not texts:
            return []
        tokens = bundle.text_tokenizer(texts, return_tensors="pt", padding=True)
        tokens = {key: value.to(self.model_manager.device, non_blocking=True) for key, value in tokens.items()}
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.model_manager.device.type == "cuda"
            else nullcontext()
        )
        with torch.inference_mode(), autocast_context:
            embeddings = bundle.model.get_text_features(**tokens)
        embeddings = torch.nn.functional.normalize(embeddings.float(), dim=1)
        return [embedding.detach().cpu().numpy().astype(np.float32) for embedding in embeddings]

    def _store_batch_results(
        self,
        batch: list[tuple[int, str, str]],
        embeddings: list[np.ndarray],
        model_name: str,
        results: list[tuple[str, np.ndarray] | None],
        pending_writes: list[tuple[str, str, str, np.ndarray]],
        done: int,
        total: int,
        progress: ProgressThrottler,
    ) -> int:
        for (index, image_path, cache_key), embedding in zip(batch, embeddings):
            results[index] = (image_path, embedding)
            pending_writes.append((cache_key, image_path, model_name, embedding))
            done += 1
        progress.emit(done, total, f"Embedding images ({done}/{total})")
        return done

    def _prepare_batch(self, bundle: ModelBundle, batch_items: list[tuple[int, str, str]]):
        start = time.perf_counter()
        kept: list[tuple[int, str, str]] = []
        if bundle.input_mode == "tensor":
            tensors: list[torch.Tensor] = []
            kept_paths: list[str] = []
            for index, path, cache_key in batch_items:
                tensor = self._load_tensor(path, bundle.preprocess)
                if tensor is None:
                    continue
                kept.append((index, path, cache_key))
                kept_paths.append(path)
                tensors.append(tensor)
            if not tensors:
                return {"tensor": None, "phash": None}, kept, time.perf_counter() - start
            return {
                "tensor": torch.stack(tensors),
                "phash": self._phash_features(kept_paths) if bundle.model_name == "phash_embedding" else None,
            }, kept, time.perf_counter() - start

        images = []
        for index, path, cache_key in batch_items:
            image = self._load_image(path)
            if image is None:
                continue
            kept.append((index, path, cache_key))
            images.append(image)
        if not images:
            return None, kept, time.perf_counter() - start
        return bundle.preprocess(images=images, return_tensors="pt"), kept, time.perf_counter() - start

    def _run_model_with_backoff(self, bundle: ModelBundle, prepared_batch) -> tuple[list[np.ndarray], bool]:
        try:
            return self._run_model(bundle, prepared_batch), False
        except RuntimeError as exc:
            message = str(exc).lower()
            if "out of memory" not in message or self.model_manager.device.type != "cuda" or self.performance_profile.name != "max_speed":
                raise
            split = self._split_prepared_batch(bundle, prepared_batch)
            if split is None:
                raise
            torch.cuda.empty_cache()
            outputs: list[np.ndarray] = []
            for batch in split:
                outputs.extend(self._run_model(bundle, batch))
            return outputs, True

    def _split_prepared_batch(self, bundle: ModelBundle, prepared_batch):
        if prepared_batch is None:
            return None
        if bundle.input_mode == "tensor":
            tensor = prepared_batch.get("tensor")
            if tensor is None or int(getattr(tensor, "shape", [0])[0]) < 2:
                return None
            mid = int(tensor.shape[0] // 2)
            phash = prepared_batch.get("phash")
            return [
                {"tensor": tensor[:mid], "phash": phash[:mid] if phash is not None else None},
                {"tensor": tensor[mid:], "phash": phash[mid:] if phash is not None else None},
            ]
        values = list(prepared_batch.values())
        if not values or int(values[0].shape[0]) < 2:
            return None
        mid = int(values[0].shape[0] // 2)
        return [
            {key: value[:mid] for key, value in prepared_batch.items()},
            {key: value[mid:] for key, value in prepared_batch.items()},
        ]

    def _run_model(self, bundle: ModelBundle, prepared_batch) -> list[np.ndarray]:
        if prepared_batch is None:
            return []
        if bundle.input_mode == "tensor" and prepared_batch.get("tensor") is None:
            return []
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.model_manager.device.type == "cuda" and bundle.onnx_session is None
            else nullcontext()
        )
        with torch.inference_mode(), autocast_context:
            if bundle.input_mode == "tensor":
                inputs = prepared_batch["tensor"]
                if bundle.onnx_session is not None:
                    embeddings = self.model_manager.onnx_service.run(bundle.onnx_session, inputs)
                else:
                    outputs = bundle.model(inputs.to(self.model_manager.device, non_blocking=True))
                    embeddings = outputs[0] if isinstance(outputs, tuple) else outputs
                phash = prepared_batch.get("phash")
                if phash is not None:
                    hash_tensor = torch.from_numpy(phash).to(embeddings.device, dtype=embeddings.dtype)
                    embeddings = torch.cat([embeddings.flatten(start_dim=1), hash_tensor], dim=1)
            elif bundle.family in {"clip", "openclip", "siglip"}:
                prepared_batch = {
                    key: value.to(self.model_manager.device, non_blocking=True)
                    for key, value in prepared_batch.items()
                }
                embeddings = bundle.model.get_image_features(**prepared_batch)
            else:
                prepared_batch = {
                    key: value.to(self.model_manager.device, non_blocking=True)
                    for key, value in prepared_batch.items()
                }
                outputs = bundle.model(**prepared_batch)
                embeddings = outputs.last_hidden_state.mean(dim=1)
        if embeddings.ndim > 2:
            embeddings = embeddings.flatten(start_dim=1)
        if embeddings.ndim == 1:
            embeddings = embeddings.unsqueeze(0)
        embeddings = torch.nn.functional.normalize(embeddings.float(), dim=1)
        return [embedding.detach().cpu().numpy().astype(np.float32) for embedding in embeddings]

    @staticmethod
    def _load_tensor(image_path: str, preprocess) -> torch.Tensor | None:
        try:
            with Image.open(image_path) as image:
                return preprocess(image.convert("RGB"))
        except Exception:
            return None

    @staticmethod
    def _load_image(image_path: str) -> Image.Image | None:
        try:
            with Image.open(image_path) as image:
                return image.convert("RGB").copy()
        except Exception:
            return None

    @staticmethod
    def _phash_features(batch_paths: list[str]) -> np.ndarray:
        import imagehash

        vectors = []
        for image_path in batch_paths:
            with Image.open(image_path) as image:
                digest = imagehash.phash(image.convert("RGB"), hash_size=8)
            bits = np.asarray(digest.hash, dtype=np.float32).reshape(-1)
            vectors.append((bits * 2.0) - 1.0)
        return np.asarray(vectors, dtype=np.float32)
