from __future__ import annotations

import argparse
import json
import statistics
import time
from tempfile import TemporaryDirectory

import numpy as np

from app.services.face_search import FaceIndexService, IndexedFaceRecord, PersonPrototype
from infra.runtime import ExecutionPolicy, RuntimeCapabilityService
from ml.clustering import ClusteringService


def _records(samples: int, dimensions: int) -> list[IndexedFaceRecord]:
    rng = np.random.default_rng(42)
    base = np.zeros((dimensions,), dtype=np.float32)
    base[0] = 1.0
    matrix = base + (rng.normal(0.0, 0.025, size=(samples, dimensions)).astype(np.float32))
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    return [
        IndexedFaceRecord(
            image_path=f"/generated/deep-{index:05d}.jpg",
            face_index=0,
            face_bbox=(8, 8, 72, 72),
            face_confidence=0.99,
            embedding=matrix[index],
        )
        for index in range(samples)
    ]


def _run(records: list[IndexedFaceRecord], repeats: int, policy: ExecutionPolicy) -> dict[str, object]:
    with TemporaryDirectory(prefix="clusterlens-deep-face-search-") as temporary:
        service = FaceIndexService(
            clustering_service=ClusteringService(execution_policy=policy),
            execution_policy=policy,
            db_path=f"{temporary}/faces.sqlite3",
        )
        prototype = PersonPrototype("Fixture", records[0].embedding, 0.80, 1)
        service.load_person_prototypes = lambda: [prototype]  # type: ignore[method-assign]
        service.load_all_records = lambda **_kwargs: list(records)  # type: ignore[method-assign]
        service.deep_search_by_person_name("Fixture")
        timings_ms: list[float] = []
        result = None
        for _repeat in range(repeats):
            started = time.perf_counter()
            result = service.deep_search_by_person_name("Fixture")
            timings_ms.append((time.perf_counter() - started) * 1000.0)
        metrics = dict(service.last_vector_compute_metrics)
        return {
            "compute_device": metrics.get("compute_device", "cpu"),
            "compute_implementation": metrics.get("compute_implementation", ""),
            "compute_fallback_reason": metrics.get("compute_fallback_reason", ""),
            "matches": len(getattr(result, "matches", ())),
            "rounds": int(getattr(result, "round_count", 0)),
            "timings_ms": [round(value, 3) for value in timings_ms],
            "median_ms": round(statistics.median(timings_ms), 3),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Seeded full-scope deep face-search benchmark.")
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--dimensions", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.samples < 2 or args.dimensions < 2 or args.repeats < 1:
        parser.error("samples, dimensions, and repeats must be positive and samples/dimensions at least two")
    records = _records(args.samples, args.dimensions)
    runtime = RuntimeCapabilityService()
    selected = runtime.select_policy("auto")
    cpu = ExecutionPolicy("cpu", "cpu", "cpu", "CPUExecutionProvider", "Deterministic CPU control.")
    print(
        json.dumps(
            {
                "seed": 42,
                "fixture": {"indexed_faces": args.samples, "dimensions": args.dimensions, "all_faces_unlabeled": True},
                "operation": "full_scope_transitive_deep_face_search",
                "repeats": args.repeats,
                "cpu": _run(records, args.repeats, cpu),
                "selected_runtime": _run(records, args.repeats, selected),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
