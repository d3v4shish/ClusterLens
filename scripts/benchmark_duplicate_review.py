#!/usr/bin/env python3
"""Profile bounded near-duplicate candidate generation on a collision corpus."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from app.services.duplicate_review import DuplicateReviewService
from app.services.library_catalog import LibraryCatalogService
from apps.shared.profile_support import profile_call


def _run(paths: int, repeats: int) -> dict[str, object]:
    hashes = {f"/collision-fixture/photo-{index:04d}.jpg": 0 for index in range(paths)}
    expected_pairs = paths * (paths - 1) // 2
    samples: list[float] = []
    with TemporaryDirectory(prefix="clusterlens-duplicate-review-benchmark-") as temporary:
        catalog = LibraryCatalogService(db_path=Path(temporary) / "catalog.sqlite3")
        service = DuplicateReviewService(catalog=catalog)
        for _ in range(repeats):
            started = time.perf_counter()
            pairs = service._near_pairs(  # noqa: SLF001 - target is the LSH candidate primitive.
                hashes,
                {},
                max_hash_distance=6,
                min_visual_score=0.0,
                exact_pairs=(),
            )
            samples.append((time.perf_counter() - started) * 1000.0)
            if len(pairs) != expected_pairs:
                raise RuntimeError(f"Collision fixture returned {len(pairs)} pairs, expected {expected_pairs}")
        _result, profile = profile_call(
            lambda: service._near_pairs(hashes, {}, max_hash_distance=6, min_visual_score=0.0, exact_pairs=())
        )
    return {
        "paths": paths,
        "hash_value": "0x0000000000000000",
        "max_hash_distance": 6,
        "expected_pairs": expected_pairs,
        "timings_ms": [round(value, 3) for value in samples],
        "median_ms": round(statistics.median(samples), 3),
        "profile_hotspots": profile.hotspots,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the fixed high-collision duplicate-review corpus.")
    parser.add_argument("--paths", type=int, default=160)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.paths < 2 or args.repeats < 1:
        parser.error("paths must be at least two and repeats must be positive")
    print(json.dumps({"fixture": "all-identical-64-bit-perceptual-hashes", "operation": "near_pair_candidate_generation", "repeats": args.repeats, "result": _run(args.paths, args.repeats)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
