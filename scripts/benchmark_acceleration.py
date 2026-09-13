from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time

import numpy as np
from sklearn.datasets import make_blobs

from infra.runtime import ExecutionPolicy, RuntimeCapabilityService
from ml.clustering import ClusteringService


def _matrix(samples: int, dimensions: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    matrix = np.ascontiguousarray(
        rng.normal(size=(samples, dimensions)).astype(np.float32)
    )
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix


def _run(
    matrix: np.ndarray,
    cluster_count: int,
    repeats: int,
    policy: ExecutionPolicy,
) -> dict[str, object]:
    service = ClusteringService(execution_policy=policy)
    service.cluster_prepared(
        matrix,
        cluster_count,
        backend="cosine-kmeans",
        performance_profile="max_speed",
        outlier_policy="assign",
    )
    timings_ms: list[float] = []
    metrics: dict[str, object] = {}
    cluster_total = 0
    membership_digests: list[str] = []
    for _sample in range(repeats):
        started = time.perf_counter()
        clusters, metrics = service.cluster_prepared(
            matrix,
            cluster_count,
            backend="cosine-kmeans",
            performance_profile="max_speed",
            outlier_policy="assign",
        )
        timings_ms.append((time.perf_counter() - started) * 1000.0)
        cluster_total = len(clusters)
        membership_payload = json.dumps(clusters, sort_keys=True, separators=(",", ":"))
        membership_digests.append(hashlib.sha256(membership_payload.encode("utf-8")).hexdigest()[:16])
    deterministic = len(set(membership_digests)) == 1
    if not deterministic:
        raise RuntimeError(f"Repeated clustering was not deterministic: {membership_digests}")
    return {
        "effective_mode": policy.effective_mode,
        "torch_device": policy.torch_device,
        "compute_device": metrics.get("compute_device", ""),
        "compute_implementation": metrics.get("compute_implementation", ""),
        "compute_fallback_reason": metrics.get("compute_fallback_reason", ""),
        "clusters": cluster_total,
        "cluster_quality_score": metrics.get("cluster_quality_score"),
        "deterministic_membership": deterministic,
        "membership_digest": membership_digests[0],
        "timings_ms": [round(value, 3) for value in timings_ms],
        "median_ms": round(statistics.median(timings_ms), 3),
    }


def _hdbscan_matrix(samples: int, dimensions: int, centers: int) -> np.ndarray:
    matrix, _labels = make_blobs(
        n_samples=samples,
        n_features=dimensions,
        centers=centers,
        cluster_std=0.65,
        random_state=42,
    )
    return np.ascontiguousarray(matrix, dtype=np.float32)


def _run_hdbscan(
    matrix: np.ndarray,
    repeats: int,
    policy: ExecutionPolicy,
) -> dict[str, object]:
    options = {
        "min_cluster_size": 20,
        "min_samples": 5,
        "cluster_selection_epsilon": 0.0,
        "allow_single_cluster": False,
    }
    service = ClusteringService(execution_policy=policy)
    service.cluster_prepared(
        matrix,
        24,
        backend="hdbscan",
        performance_profile="balanced",
        outlier_policy="keep",
        backend_options=options,
    )
    timings_ms: list[float] = []
    metrics: dict[str, object] = {}
    memberships: list[str] = []
    cluster_total = 0
    outlier_total = 0
    for _sample in range(repeats):
        started = time.perf_counter()
        clusters, metrics = service.cluster_prepared(
            matrix,
            24,
            backend="hdbscan",
            performance_profile="balanced",
            outlier_policy="keep",
            backend_options=options,
        )
        timings_ms.append((time.perf_counter() - started) * 1000.0)
        cluster_total = len([cluster_id for cluster_id in clusters if int(cluster_id) >= 0])
        outlier_total = len(clusters.get(-1, []))
        payload = json.dumps(clusters, sort_keys=True, separators=(",", ":"))
        memberships.append(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16])
    deterministic = len(set(memberships)) == 1
    if not deterministic:
        raise RuntimeError(f"Repeated HDBSCAN clustering was not deterministic: {memberships}")
    return {
        "effective_mode": policy.effective_mode,
        "compute_device": metrics.get("compute_device", ""),
        "compute_implementation": metrics.get("compute_implementation", ""),
        "compute_fallback_reason": metrics.get("compute_fallback_reason", ""),
        "backend_options": options,
        "clusters": cluster_total,
        "outliers": outlier_total,
        "deterministic_membership": deterministic,
        "membership_digest": memberships[0],
        "timings_ms": [round(value, 3) for value in timings_ms],
        "median_ms": round(statistics.median(timings_ms), 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Seeded ClusterLens vector-compute benchmark.")
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--dimensions", type=int, default=128)
    parser.add_argument("--clusters", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--hdbscan-samples", type=int, default=10_000)
    parser.add_argument("--hdbscan-dimensions", type=int, default=32)
    parser.add_argument("--hdbscan-centers", type=int, default=24)
    args = parser.parse_args()
    if args.samples < 2 or args.dimensions < 2 or args.clusters < 2:
        parser.error("samples, dimensions, and clusters must all be at least 2")
    if args.clusters > args.samples:
        parser.error("clusters cannot exceed samples")
    if args.repeats < 1:
        parser.error("repeats must be at least 1")
    if args.hdbscan_samples < 2 or args.hdbscan_dimensions < 2 or args.hdbscan_centers < 2:
        parser.error("HDBSCAN samples, dimensions, and centers must all be at least 2")

    matrix = _matrix(args.samples, args.dimensions)
    runtime = RuntimeCapabilityService()
    selected_policy = runtime.select_policy("auto")
    cpu_policy = ExecutionPolicy(
        preferred_mode="cpu",
        effective_mode="cpu",
        torch_device="cpu",
        onnx_provider="CPUExecutionProvider",
        reason="Deterministic benchmark CPU fallback.",
    )
    hdbscan_matrix = _hdbscan_matrix(
        args.hdbscan_samples,
        args.hdbscan_dimensions,
        args.hdbscan_centers,
    )
    payload = {
        "seed": 42,
        "samples": args.samples,
        "dimensions": args.dimensions,
        "clusters_requested": args.clusters,
        "repeats": args.repeats,
        "cpu": _run(matrix, args.clusters, args.repeats, cpu_policy),
        "selected_runtime": _run(matrix, args.clusters, args.repeats, selected_policy),
        "hdbscan": {
            "seed": 42,
            "samples": args.hdbscan_samples,
            "dimensions": args.hdbscan_dimensions,
            "centers": args.hdbscan_centers,
            "cpu": _run_hdbscan(hdbscan_matrix, args.repeats, cpu_policy),
            "selected_runtime": _run_hdbscan(hdbscan_matrix, args.repeats, selected_policy),
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
