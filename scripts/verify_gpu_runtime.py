from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from infra.runtime import RuntimeCapabilityService
from ml.vector_compute import VectorComputeService


def _verify_vector_compute(policy) -> tuple[dict[str, object], list[str]]:
    failures: list[str] = []
    rng = np.random.default_rng(42)
    matrix = np.ascontiguousarray(rng.normal(size=(1024, 64)).astype(np.float32))
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    service = VectorComputeService(policy)

    scores, score_info = service.cosine_scores(matrix, matrix[0])
    projected, pca_info = service.pca_project(matrix, 16)
    distances, indices, neighbor_info = service.cosine_neighbors(matrix, 12)
    labels_a, kmeans_info = service.kmeans_labels(matrix, 16, seed=42, max_iter=40, n_init=2)
    labels_b, _repeat_info = service.kmeans_labels(matrix, 16, seed=42, max_iter=40, n_init=2)
    silhouette, silhouette_info = service.silhouette_score(
        matrix,
        labels_a if labels_a is not None else np.zeros(len(matrix), dtype=np.int32),
        sample_size=512,
        seed=42,
    )
    hdbscan_matrix = np.ascontiguousarray(
        np.vstack(
            (
                rng.normal(loc=-4.0, scale=0.15, size=(256, 16)),
                rng.normal(loc=4.0, scale=0.15, size=(256, 16)),
            )
        ),
        dtype=np.float32,
    )
    hdbscan_labels, hdbscan_info = service.hdbscan_labels(
        hdbscan_matrix,
        min_cluster_size=20,
        min_samples=5,
        cluster_selection_epsilon=0.0,
        allow_single_cluster=False,
    )

    infos = {
        "dense_scores": score_info,
        "semantic_pca": pca_info,
        "graph_neighbors": neighbor_info,
        "cosine_kmeans": kmeans_info,
        "cluster_quality": silhouette_info,
        "hdbscan": hdbscan_info,
    }
    for name, info in infos.items():
        if info.device != "cuda":
            failures.append(f"{name} selected {info.device}: {info.fallback_reason or info.implementation}")
    if scores.shape != (1024,) or not np.isfinite(scores).all():
        failures.append("Dense CUDA similarity scoring returned invalid output.")
    if projected is None or projected.shape != (1024, 16) or not np.isfinite(projected).all():
        failures.append("CUDA semantic PCA returned invalid output.")
    if distances is None or indices is None or distances.shape != (1024, 12) or indices.shape != (1024, 12):
        failures.append("CUDA graph-neighbor search returned invalid output.")
    if labels_a is None or labels_b is None or labels_a.shape != (1024,):
        failures.append("CUDA K-means returned invalid output.")
    elif not np.array_equal(labels_a, labels_b):
        failures.append("CUDA K-means was not deterministic for the fixed seed.")
    if silhouette is None or not np.isfinite(silhouette):
        failures.append("CUDA silhouette scoring returned invalid output.")
    if hdbscan_labels is None or hdbscan_labels.shape != (512,):
        failures.append("CUDA HDBSCAN returned invalid output.")
    elif len({int(value) for value in hdbscan_labels if int(value) >= 0}) < 2:
        failures.append("CUDA HDBSCAN did not recover the two separated fixture groups.")

    return {
        name: {
            "device": info.device,
            "implementation": info.implementation,
            "fallback_reason": info.fallback_reason,
        }
        for name, info in infos.items()
    }, failures


def main() -> int:
    result = RuntimeCapabilityService().verify("cuda")
    capabilities = result["capabilities"]
    policy = result["policy"]
    torch_smoke = result.get("torch_smoke") or {}
    onnx_smoke = result.get("onnx_smoke") or {}
    failures: list[str] = []
    vector_smoke: dict[str, object] = {}

    if not capabilities.torch_cuda_available:
        failures.append("Torch cannot use CUDA.")
    if policy.torch_device != "cuda":
        failures.append(f"Torch policy selected {policy.torch_device}, not CUDA.")
    if policy.onnx_provider != "CUDAExecutionProvider":
        failures.append(f"ONNX policy selected {policy.onnx_provider}, not CUDAExecutionProvider.")
    if not torch_smoke.get("ok"):
        failures.append(str(torch_smoke.get("error") or "Torch CUDA smoke test failed."))
    if not onnx_smoke.get("ok") or onnx_smoke.get("provider") != "CUDAExecutionProvider":
        failures.append(str(onnx_smoke.get("error") or "ONNX CUDA smoke test failed."))
    if policy.torch_device == "cuda":
        vector_smoke, vector_failures = _verify_vector_compute(policy)
        failures.extend(vector_failures)

    print(
        json.dumps(
            {
                "capabilities": asdict(capabilities),
                "policy": asdict(policy),
                "packages": result.get("packages") or {},
                "cpu_details": result.get("cpu_details") or {},
                "torch_smoke": torch_smoke,
                "onnx_smoke": onnx_smoke,
                "vector_smoke": vector_smoke,
                "failures": failures,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
