#!/usr/bin/env python3
"""Benchmark the bounded face-index scheduler with generated local media only."""

from __future__ import annotations

import argparse
import cProfile
import json
import logging
import pstats
import statistics
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
from PIL import Image

from app.services.face_search import DetectedFace, FaceIndexPerformanceConfig, FaceIndexService
from infra.runtime import ExecutionPolicy


class _FixtureDetector:
    """One deterministic interior face per generated image; no model is loaded."""

    def detect_faces_in_image(self, image_path: str, rgb_image: Image.Image) -> list[DetectedFace]:
        box = (16, 16, 80, 80)
        return [DetectedFace(image_path, box, 0.99, rgb_image.crop(box))]


class _FixtureEmbedder:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def embed_faces(self, face_crops: list[Image.Image]) -> np.ndarray:
        self.batch_sizes.append(len(face_crops))
        # Keep the fixture CPU-only: this measures scheduling, decode, quality,
        # and SQLite writes independently of model quality or model download.
        return np.tile(np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (len(face_crops), 1))


def _write_fixture(root: Path, images: int) -> list[str]:
    paths: list[str] = []
    for index in range(images):
        data = np.zeros((96, 96, 3), dtype=np.uint8)
        data[:, :, 0] = (index * 17) % 255
        data[:, :, 1] = np.arange(96, dtype=np.uint8)[:, None]
        data[:, :, 2] = np.arange(96, dtype=np.uint8)[None, :]
        path = root / f"face-index-{index:04d}.jpg"
        Image.fromarray(data, mode="RGB").save(path, quality=92, subsampling=0)
        paths.append(str(path))
    return paths


def _top_profile_entries(profile: cProfile.Profile, limit: int = 8) -> list[dict[str, object]]:
    entries = []
    for (filename, line, function), (_cc, _nc, _tt, cumulative, _callers) in pstats.Stats(profile).stats.items():
        entries.append(
            {
                "function": f"{Path(filename).name}:{line}:{function}",
                "cumulative_seconds": round(float(cumulative), 6),
            }
        )
    return sorted(entries, key=lambda item: float(item["cumulative_seconds"]), reverse=True)[:limit]


def _run_once(images: int, config: FaceIndexPerformanceConfig, *, profile: bool) -> tuple[dict[str, object], list[dict[str, object]]]:
    policy = ExecutionPolicy("cpu", "cpu", "cpu", "CPUExecutionProvider", "Generated fixture CPU control.")
    with TemporaryDirectory(prefix="clusterlens-face-index-") as temporary:
        root = Path(temporary)
        paths = _write_fixture(root, images)
        embedder = _FixtureEmbedder()
        service = FaceIndexService(
            detection_service=_FixtureDetector(),
            embedding_service=embedder,
            execution_policy=policy,
            db_path=root / "faces.sqlite3",
            index_performance=config,
        )
        profiler = cProfile.Profile()
        if profile:
            profiler.enable()
        started = perf_counter()
        metrics = service.index_paths(paths, force=True)
        elapsed_ms = (perf_counter() - started) * 1000.0
        if profile:
            profiler.disable()
        output = dict(metrics)
        output["elapsed_ms"] = round(elapsed_ms, 3)
        output["embedder_batches"] = list(embedder.batch_sizes)
        output["indexed_records"] = service.count_indexed_faces(include_tiny_faces=True)
        return output, _top_profile_entries(profiler) if profile else []


def main() -> int:
    parser = argparse.ArgumentParser(description="Generated face-index scheduler benchmark.")
    parser.add_argument("--images", type=int, default=48)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--decode-workers", type=int, default=4)
    parser.add_argument("--quality-workers", type=int, default=2)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--write-batch-images", type=int, default=16)
    args = parser.parse_args()
    if min(args.images, args.repeats, args.decode_workers, args.quality_workers, args.embedding_batch_size, args.write_batch_images) < 1:
        parser.error("all counts must be at least one")
    config = FaceIndexPerformanceConfig(
        decode_workers=args.decode_workers,
        quality_workers=args.quality_workers,
        embedding_batch_size=args.embedding_batch_size,
        write_batch_images=args.write_batch_images,
        max_decoded_images=min(8, args.decode_workers),
    )
    logging.getLogger("app.services.face_search").setLevel(logging.WARNING)
    samples: list[dict[str, object]] = []
    profile_entries: list[dict[str, object]] = []
    for repeat in range(args.repeats):
        result, entries = _run_once(args.images, config, profile=repeat == 0)
        samples.append(result)
        if entries:
            profile_entries = entries
    elapsed_samples = [float(sample["elapsed_ms"]) for sample in samples]
    print(
        json.dumps(
            {
                "operation": "bounded_face_index_scheduler",
                "fixture": {"images": args.images, "faces_per_image": 1, "generated_jpegs": True},
                "execution": "CPU control; no detector/embedder model is loaded",
                "configuration": {
                    "decode_workers": config.decode_workers,
                    "quality_workers": config.quality_workers,
                    "embedding_batch_size": config.embedding_batch_size,
                    "write_batch_images": config.write_batch_images,
                    "max_decoded_images": config.max_decoded_images,
                },
                "samples": samples,
                "median_elapsed_ms": round(statistics.median(elapsed_samples), 3),
                "profile_top_cumulative": profile_entries,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
