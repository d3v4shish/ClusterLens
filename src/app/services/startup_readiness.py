"""Local-only runtime and model readiness checks for the desktop shell.

The check is designed for a background job after Qt is visible.  It neither
constructs inference models nor downloads assets, so it can truthfully gate
actions without creating hidden network or GPU work during startup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from infra.runtime import ExecutionPolicy, RuntimeCapabilities, RuntimeCapabilityService

if TYPE_CHECKING:
    from app.services.model_assets import ModelAssetService


@dataclass(frozen=True)
class StartupReadinessReport:
    """Readiness state derived solely from installed local resources."""

    execution_policy: ExecutionPolicy
    capabilities: RuntimeCapabilities
    clustering_models: tuple[str, ...]
    clustering_ready: bool
    clustering_message: str
    face_detector_id: str
    face_embedder_id: str
    face_ready: bool
    face_message: str


def inspect_startup_readiness(
    runtime_service: RuntimeCapabilityService,
    model_asset_service: "ModelAssetService",
    *,
    preferred_execution_mode: str,
    clustering_models: tuple[str, ...] | list[str],
    face_model_root: str,
    face_detector_id: str,
    face_embedder_id: str,
) -> StartupReadinessReport:
    """Inspect CUDA and installed assets without mutating local state."""

    policy = runtime_service.select_policy(preferred_execution_mode, refresh=True)
    capabilities = runtime_service.detect()
    normalized_models = tuple(str(model).strip().lower() for model in clustering_models if str(model).strip())

    if policy.cuda_required_unavailable:
        runtime_message = policy.error or policy.reason or "CUDA was requested but is unavailable."
        return StartupReadinessReport(
            execution_policy=policy,
            capabilities=capabilities,
            clustering_models=normalized_models,
            clustering_ready=False,
            clustering_message=runtime_message,
            face_detector_id=str(face_detector_id or ""),
            face_embedder_id=str(face_embedder_id or ""),
            face_ready=False,
            face_message=runtime_message,
        )

    missing_models = [
        model
        for model in normalized_models
        if not model_asset_service.model_available_without_download(model)
    ]
    if not normalized_models:
        clustering_ready = False
        clustering_message = "Select an embedding model in Clustering."
    elif missing_models:
        clustering_ready = False
        clustering_message = (
            "Missing clustering model" + ("s" if len(missing_models) > 1 else "") + ": "
            + ", ".join(missing_models)
            + ". Install it from Settings > Clustering Models."
        )
    else:
        clustering_ready = True
        runtime_label = "CUDA" if policy.uses_cuda else "CPU"
        clustering_message = f"{runtime_label} and clustering models are ready."

    # Optional face dependencies remain inside this worker-only path.  Shell
    # construction can therefore stay light and responsive.
    from app.services import face_search

    detector_id, embedder_id = face_search.resolve_ready_face_pipeline_ids(
        face_model_root,
        "human",
        face_detector_id,
        face_embedder_id,
    )
    try:
        detector = face_search.resolve_face_detector_bundle(face_model_root, "human", detector_id)
        embedder = face_search.resolve_face_embedder_bundle(face_model_root, "human", embedder_id)
    except Exception as exc:
        face_ready = False
        face_message = f"Face pipeline could not be resolved: {exc}"
    else:
        detector_ready = bool(detector.available)
        embedder_ready = bool(embedder.available)
        if detector.source_kind == "builtin":
            detector_ready = True
        elif detector.backend_family == "yunet":
            detector_ready = detector_ready and face_search.cv2 is not None and hasattr(face_search.cv2, "FaceDetectorYN")
        else:
            detector_ready = detector_ready and face_search.ort is not None
        if embedder.source_kind == "builtin":
            embedder_ready = embedder_ready and model_asset_service.model_available_without_download("facenet")
        else:
            embedder_ready = embedder_ready and face_search.ort is not None
        face_ready = bool(detector_ready and embedder_ready)
        if face_ready:
            face_message = f"Face pipeline ready: {detector.display_name} + {embedder.display_name}."
        else:
            unavailable: list[str] = []
            if not detector_ready:
                unavailable.append(detector.availability_message or f"{detector.display_name} is unavailable")
            if not embedder_ready:
                if embedder.source_kind == "builtin":
                    unavailable.append("FaceNet weights are not installed")
                else:
                    unavailable.append(embedder.availability_message or f"{embedder.display_name} is unavailable")
            face_message = "; ".join(unavailable) + ". Install missing face models in Settings > Face Models."

    return StartupReadinessReport(
        execution_policy=policy,
        capabilities=capabilities,
        clustering_models=normalized_models,
        clustering_ready=clustering_ready,
        clustering_message=clustering_message,
        face_detector_id=str(detector_id),
        face_embedder_id=str(embedder_id),
        face_ready=face_ready,
        face_message=face_message,
    )
