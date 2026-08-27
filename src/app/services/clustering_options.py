from __future__ import annotations


LEGACY_MODEL_ORDER = (
    "fast_preview",
    "mobileclip",
    "clip",
    "openclip",
    "siglip",
    "convnext",
    "phash_embedding",
    "resnet",
    "vgg",
    "vit",
    "facenet",
    "dino",
    "dinov2_base",
    "dino_large",
)

PRODUCTION_MODEL_ORDER = (
    "fast_preview",
    "mobileclip",
    "dino",
    "dinov2_base",
    "clip",
    "openclip",
    "siglip",
    "resnet",
)

MODEL_DISPLAY_NAMES = {
    "fast_preview": "Fast Preview",
    "mobileclip": "MobileCLIP S0",
    "clip": "CLIP",
    "openclip": "OpenCLIP",
    "siglip": "SigLIP",
    "convnext": "ConvNeXt",
    "phash_embedding": "pHash Embedding",
    "resnet": "ResNet",
    "vgg": "VGG",
    "vit": "ViT",
    "facenet": "FaceNet",
    "dino": "DINO",
    "dinov2_base": "DINOv2 Base",
    "dino_large": "DINO Large",
}

MODEL_TOOLTIPS = {
    "fast_preview": "Fast smoke-test model for quick checks. Lower quality than DINO/CLIP, but useful for responsiveness checks. If missing from the packaged build, the app asks before downloading weights.",
    "mobileclip": "CPU-friendly MobileCLIP-S0 image model. Use this on machines without a GPU when CLIP/SigLIP are too slow. The app asks before downloading missing weights.",
    "dino": "Recommended production default for visual clustering. Image-only model; advanced naming uses a CLIP/SigLIP sidecar. The app asks before downloading missing weights.",
    "dinov2_base": "Stronger DINOv2 Base image model for higher-quality visual clustering. Heavier than DINO and MobileCLIP. The app asks before downloading missing weights.",
    "clip": "Image-text model. Useful when you want clustering plus native text-label evidence in advanced mode. The app asks before downloading missing weights.",
    "openclip": "OpenCLIP LAION ViT-B/32 image-text model. Useful for stronger semantic clustering and native text-label evidence. The app asks before downloading missing weights.",
    "siglip": "Image-text model alternative to CLIP. If text tokenization is unavailable, naming falls back visibly to CLIP. The app asks before downloading missing weights.",
    "resnet": "Stable CNN baseline for comparison runs. Useful when you want a simple non-transformer visual feature space. If bundled as ONNX, it can run without downloading weights.",
    "convnext": "Legacy CNN option. Hidden from the production UI because it overlaps with ResNet/DINO and adds another model download.",
    "phash_embedding": "Legacy pHash-assisted feature option. Hidden from production clustering to avoid mixing duplicate-search behavior into clusters.",
    "vgg": "Legacy CNN option. Hidden from production because ResNet is the simpler maintained CNN baseline.",
    "vit": "Legacy Hugging Face vision transformer. Hidden from production because DINO/CLIP cover the useful transformer paths.",
    "facenet": "Legacy face-specific embedding. Excluded from production clustering because face workflows are out of scope.",
    "dino_large": "Heavy DINOv2 model. Hidden from production UI to avoid large downloads and slow/costly default runs.",
}

ONNX_READY_MODELS = frozenset({"fast_preview", "convnext", "resnet"})

LEGACY_BACKEND_ORDER = ("cosine-kmeans", "hdbscan", "graph", "faiss", "sklearn")
PRODUCTION_BACKEND_ORDER = ("cosine-kmeans", "hdbscan", "graph")
DEFAULT_PRODUCTION_BACKENDS = ("hdbscan",)

BACKEND_TOOLTIPS = {
    "cosine-kmeans": "Fast fixed-count clustering. Best default when you want predictable cluster counts.",
    "hdbscan": "Density-based clustering. Good for natural groups and outliers when the optional dependency is available.",
    "graph": "Similarity-graph clustering. Useful for tight visual groups without forcing every image into a fixed count.",
    "faiss": "Legacy optional FAISS KMeans path. Hidden from production UI until packaging/runtime validation is complete.",
    "sklearn": "Legacy duplicate KMeans path. Hidden from production UI because cosine-kmeans already covers this behavior.",
}


def clustering_model_names(scope: str = "legacy") -> tuple[str, ...]:
    return PRODUCTION_MODEL_ORDER if _is_production_scope(scope) else LEGACY_MODEL_ORDER


def clustering_backend_names(scope: str = "legacy") -> tuple[str, ...]:
    return PRODUCTION_BACKEND_ORDER if _is_production_scope(scope) else LEGACY_BACKEND_ORDER


def default_backend_names(scope: str = "legacy") -> tuple[str, ...]:
    if _is_production_scope(scope):
        return DEFAULT_PRODUCTION_BACKENDS
    return ("cosine-kmeans", "hdbscan", "graph")


def normalize_embedding_models(
    raw_models: object,
    *,
    default_model: str = "dino",
    scope: str = "legacy",
) -> list[str]:
    allowed = set(clustering_model_names(scope))
    normalized: list[str] = []
    unsupported: list[str] = []
    for model in _iter_string_values(raw_models):
        if model in allowed and model not in normalized:
            normalized.append(model)
        elif model:
            unsupported.append(model)
    if normalized:
        return normalized

    fallback = _model_fallback_for_unsupported(unsupported, default_model, scope, allowed)
    return [fallback]


def normalize_clustering_backends(raw_backends: object, *, scope: str = "legacy") -> list[str]:
    allowed = set(clustering_backend_names(scope))
    normalized: list[str] = []
    unsupported: list[str] = []
    for backend in _iter_string_values(raw_backends):
        if backend in allowed and backend not in normalized:
            normalized.append(backend)
        elif backend:
            unsupported.append(backend)
    if normalized:
        return normalized

    fallback = _backend_fallback_for_unsupported(unsupported, scope, allowed)
    return [fallback]


def model_label(model_name: str) -> str:
    runtime = "ONNX/Torch" if model_name in ONNX_READY_MODELS else "Torch"
    return f"{MODEL_DISPLAY_NAMES.get(model_name, model_name)} ({runtime})"


def model_tooltip(model_name: str) -> str:
    return MODEL_TOOLTIPS.get(model_name, "Image embedding model.")


def backend_tooltip(backend_name: str) -> str:
    return BACKEND_TOOLTIPS.get(backend_name, "Clustering backend.")


def _model_fallback_for_unsupported(
    unsupported: list[str],
    default_model: str,
    scope: str,
    allowed: set[str],
) -> str:
    default = default_model if default_model in allowed else next(iter(clustering_model_names(scope)))
    if not _is_production_scope(scope):
        return default
    replacement_by_removed_model = {
        "convnext": "resnet",
        "convxnet": "resnet",
        "phash_embedding": "dino",
        "vgg": "resnet",
        "vit": "dino",
        "facenet": "dino",
        "dino_large": "dinov2_base",
    }
    for model in unsupported:
        replacement = replacement_by_removed_model.get(model)
        if replacement in allowed:
            return replacement
    return default


def _backend_fallback_for_unsupported(unsupported: list[str], scope: str, allowed: set[str]) -> str:
    default = "cosine-kmeans" if "cosine-kmeans" in allowed else next(iter(clustering_backend_names(scope)))
    if not _is_production_scope(scope):
        return default
    replacement_by_removed_backend = {
        "faiss": "cosine-kmeans",
        "sklearn": "cosine-kmeans",
    }
    for backend in unsupported:
        replacement = replacement_by_removed_backend.get(backend)
        if replacement in allowed:
            return replacement
    return default


def _iter_string_values(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = value.split(",")
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        values = [value]
    return [str(item or "").strip().lower() for item in values if str(item or "").strip()]


def _is_production_scope(scope: str) -> bool:
    return str(scope or "").strip().lower() == "production"
