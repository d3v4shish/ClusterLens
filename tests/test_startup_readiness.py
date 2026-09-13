from __future__ import annotations

from app.services.startup_readiness import inspect_startup_readiness
from infra.runtime import ExecutionPolicy, RuntimeCapabilities


class _Runtime:
    def __init__(self, policy: ExecutionPolicy) -> None:
        self.policy = policy

    def select_policy(self, _preferred_mode: str, refresh: bool = False) -> ExecutionPolicy:
        _ = refresh
        return self.policy

    def detect(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(torch_cuda_available=self.policy.uses_cuda)


class _Assets:
    def __init__(self, available: set[str]) -> None:
        self.available = available

    def model_available_without_download(self, model_name: str, *, require_text: bool = False) -> bool:
        _ = require_text
        return model_name in self.available

    def local_cache_present(self, model_name: str) -> bool:
        return model_name in self.available


def test_startup_readiness_reports_missing_local_models_without_downloading() -> None:
    report = inspect_startup_readiness(
        _Runtime(ExecutionPolicy(preferred_mode="cpu", effective_mode="cpu")),
        _Assets(set()),
        preferred_execution_mode="cpu",
        clustering_models=("clip",),
        face_model_root="",
        face_detector_id="scrfd_10g_kps",
        face_embedder_id="arcface_r100_glint360k",
    )

    assert not report.clustering_ready
    assert "clip" in report.clustering_message
    assert not report.face_ready
    assert "FaceNet" in report.face_message


def test_startup_readiness_rejects_an_unavailable_explicit_cuda_request() -> None:
    report = inspect_startup_readiness(
        _Runtime(
            ExecutionPolicy(
                preferred_mode="cuda",
                effective_mode="cpu",
                error="CUDA is unavailable",
            )
        ),
        _Assets({"clip", "facenet"}),
        preferred_execution_mode="cuda",
        clustering_models=("clip",),
        face_model_root="",
        face_detector_id="mtcnn_builtin",
        face_embedder_id="vggface2_builtin",
    )

    assert not report.clustering_ready
    assert not report.face_ready
    assert report.clustering_message == "CUDA is unavailable"


def test_startup_readiness_accepts_downloaded_cpu_models(monkeypatch) -> None:
    from app.services import face_search

    assets = _Assets({"clip", "facenet"})
    monkeypatch.setattr(face_search, "ModelAssetService", lambda: assets)
    report = inspect_startup_readiness(
        _Runtime(ExecutionPolicy(preferred_mode="cpu", effective_mode="cpu")),
        assets,
        preferred_execution_mode="cpu",
        clustering_models=("clip",),
        face_model_root="",
        face_detector_id="mtcnn_builtin",
        face_embedder_id="vggface2_builtin",
    )

    assert report.clustering_ready
    assert report.face_ready
