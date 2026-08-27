from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal, TypeAlias


SUPPORTED_SIMILARITY_MODES = ("semantic", "cosine")


def _default_runtime_selection():
    try:
        from infra.runtime import RuntimeSelection

        return RuntimeSelection()
    except Exception:
        return {
            "requested_backend": "auto",
            "actual_backend": "cpu",
            "cuda_device": "",
            "precision": "fp32",
            "model_artifact": "",
            "torch_device": "cpu",
            "onnx_provider": "CPUExecutionProvider",
            "fallback_reason": "",
            "error": "",
        }


@dataclass(frozen=True)
class ProductionClusterRequest:
    directory: str
    embedding_models: list[str]
    num_clusters: int
    clustering_backends: list[str]
    recursive: bool
    similarity_mode: str
    outlier_policy: str
    use_onnx: bool
    reuse_result_cache: bool
    use_embedding_cache_lookup: bool
    source_paths: list[str] | None = None
    source_fingerprints: list[tuple[str, int, int]] | None = None
    source_snapshot_key: str = ""
    performance_profile: str = "balanced"
    batch_size_cpu: int = 0
    batch_size_gpu: int = 0
    preprocess_workers: int = 0
    vram_headroom_mb: int = 0
    preferred_execution_mode: str = "auto"
    tag_filter: list[str] = field(default_factory=list)
    tag_match: str = "Any"
    similarity_modes: list[str] = field(default_factory=list)
    generate_cluster_meanings: bool = False
    generate_cluster_explanations: bool = True
    cluster_meaning_model: str = "auto"
    allow_model_downloads: bool = True

    def __post_init__(self) -> None:
        modes = _normalize_similarity_modes(self.similarity_modes, self.similarity_mode)
        object.__setattr__(self, "similarity_modes", modes)
        object.__setattr__(self, "similarity_mode", modes[0])
        object.__setattr__(self, "allow_model_downloads", bool(self.allow_model_downloads))
        for field_name in ("batch_size_cpu", "batch_size_gpu", "preprocess_workers", "vram_headroom_mb"):
            object.__setattr__(self, field_name, max(0, int(getattr(self, field_name, 0) or 0)))

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def json_line(event_type: str, payload: dict[str, object]) -> dict[str, object]:
    return {"type": event_type, "payload": payload}


@dataclass(frozen=True)
class ClusteringWorkerRequestV2:
    type: Literal["clustering"] = "clustering"
    request: ProductionClusterRequest = field(default_factory=lambda: ProductionClusterRequest(
        directory="",
        embedding_models=[],
        num_clusters=2,
        clustering_backends=[],
        recursive=False,
        similarity_mode="semantic",
        outlier_policy="assign",
        use_onnx=False,
        reuse_result_cache=True,
        use_embedding_cache_lookup=True,
    ))
    runtime: object = field(default_factory=_default_runtime_selection)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FaceInferenceWorkerRequestV2:
    type: Literal["face_inference"] = "face_inference"
    scope: Literal["global", "session"] = "session"
    mode: Literal["human"] = "human"
    image_paths: list[str] = field(default_factory=list)
    detector_id: str = ""
    embedder_id: str = ""
    runtime: object = field(default_factory=_default_runtime_selection)
    options: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ShutdownWorkerRequestV2:
    type: Literal["shutdown"] = "shutdown"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


WorkerRequestV2: TypeAlias = ClusteringWorkerRequestV2 | FaceInferenceWorkerRequestV2 | ShutdownWorkerRequestV2


@dataclass(frozen=True)
class WorkerProgressEventV2:
    type: Literal["progress"] = "progress"
    value: int = 0
    message: str = ""
    runtime: object = field(default_factory=_default_runtime_selection)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class WorkerCompletedEventV2:
    type: Literal["completed"] = "completed"
    payload_ref: str = ""
    payload: dict[str, object] = field(default_factory=dict)
    runtime: object = field(default_factory=_default_runtime_selection)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class WorkerFailedEventV2:
    type: Literal["failed"] = "failed"
    message: str = ""
    recoverable: bool = False
    runtime: object = field(default_factory=_default_runtime_selection)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class WorkerCancelledEventV2:
    type: Literal["cancelled"] = "cancelled"
    message: str = ""
    runtime: object = field(default_factory=_default_runtime_selection)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


WorkerEventV2: TypeAlias = WorkerProgressEventV2 | WorkerCompletedEventV2 | WorkerFailedEventV2 | WorkerCancelledEventV2


def _normalize_similarity_modes(similarity_modes: list[str] | None, legacy_similarity_mode: str | None) -> list[str]:
    raw_modes = list(similarity_modes or [])
    if not raw_modes and legacy_similarity_mode:
        raw_modes = [legacy_similarity_mode]
    normalized: list[str] = []
    for mode in raw_modes or ["semantic"]:
        cleaned = str(mode or "").strip().lower()
        if cleaned not in SUPPORTED_SIMILARITY_MODES:
            cleaned = "semantic"
        if cleaned not in normalized:
            normalized.append(cleaned)
    return normalized or ["semantic"]
