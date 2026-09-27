#!/usr/bin/env python3
"""Benchmark bounded progressive Faces folder-review pages.

The fixture seeds only temporary SQLite rows.  It intentionally does not create
source images, decode crops, instantiate face models, or use user runtime data:
this measures the indexed-review query and object-publication path that runs
before viewport-bounded thumbnail/crop work.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import tracemalloc
from pathlib import Path
from tempfile import TemporaryDirectory

from app.services.face_search import FACE_REVIEW_PAGE_MAX_IMAGES, FaceFolderReviewPage, FaceIndexService
from apps.shared.profile_support import profile_call


def _fixture_paths(root: Path, photos: int) -> list[str]:
    return [str(root / f"review-{index:05d}.jpg") for index in range(photos)]


def _seed_index(service: FaceIndexService, paths: list[str], faces: int) -> None:
    if faces < len(paths):
        raise ValueError("face count must be at least the number of photos")
    extra_faces = faces - len(paths)
    face_rows = []
    scan_rows = []
    for image_index, path in enumerate(paths):
        face_count = 1 + int(image_index < extra_faces)
        scan_rows.append((path, 1, 1, face_count, 100, 100))
        for face_index in range(face_count):
            face_rows.append(
                (
                    path,
                    face_index,
                    "[10, 10, 50, 60]",
                    0.99,
                    b"fixture-embedding",
                    "clean",
                    1.0,
                    "[]",
                    1,
                    0,
                    0,
                    1,
                    1,
                )
            )
    with service._connect() as connection:  # Fixture setup only; the timed path uses the public API.
        connection.executemany(
            """
            INSERT INTO face_index(
                image_path, face_index, bbox_json, face_confidence, embedding,
                quality_status, quality_score, quality_reasons_json,
                quality_revision, is_tiny, hidden, mtime_ns, file_size
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            face_rows,
        )
        connection.executemany(
            """
            INSERT INTO face_scan_images(
                image_path, mtime_ns, file_size, face_count, image_width, image_height
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            scan_rows,
        )


def _load_all_pages(
    service: FaceIndexService,
    root: Path,
    paths: list[str],
    *,
    candidate_snapshot=None,
) -> tuple[float, list[int], int]:
    started = time.perf_counter()
    offsets: list[int] = []
    loaded = 0
    offset = 0
    while True:
        page = service.load_folder_review_image_page(
            str(root),
            offset=offset,
            limit=FACE_REVIEW_PAGE_MAX_IMAGES,
            candidate_paths=paths,
            _candidate_snapshot=candidate_snapshot,
        )
        if not isinstance(page, FaceFolderReviewPage):
            raise RuntimeError("Folder-review page loader returned an invalid page")
        if page.offset != offset or len(page.items) > FACE_REVIEW_PAGE_MAX_IMAGES:
            raise RuntimeError("Folder-review page violated its bounded offset contract")
        offsets.append(int(page.offset))
        loaded += len(page.items)
        if page.next_offset is None:
            break
        offset = int(page.next_offset)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return elapsed_ms, offsets, loaded


def _run_once(
    root: Path,
    photos: int,
    faces: int,
    *,
    use_candidate_snapshot: bool,
) -> tuple[float, float, float, list[int], int]:
    root.mkdir(parents=True, exist_ok=True)
    paths = _fixture_paths(root, photos)
    service = FaceIndexService(db_path=root / "faces.sqlite3")
    _seed_index(service, paths, faces)

    first_started = time.perf_counter()
    candidate_snapshot = None
    prepare_elapsed_ms = 0.0
    if use_candidate_snapshot:
        prepare_started = time.perf_counter()
        candidate_snapshot = service.prepare_folder_review_candidate_snapshot(
            str(root), candidate_paths=paths
        )
        prepare_elapsed_ms = (time.perf_counter() - prepare_started) * 1000.0
    first_page = service.load_folder_review_image_page(
        str(root),
        offset=0,
        limit=FACE_REVIEW_PAGE_MAX_IMAGES,
        candidate_paths=paths,
        _candidate_snapshot=candidate_snapshot,
    )
    first_elapsed_ms = (time.perf_counter() - first_started) * 1000.0
    if len(first_page.items) != min(photos, FACE_REVIEW_PAGE_MAX_IMAGES):
        raise RuntimeError("First progressive page did not use the configured bound")

    all_elapsed_ms, offsets, loaded = _load_all_pages(
        service, root, paths, candidate_snapshot=candidate_snapshot
    )
    return prepare_elapsed_ms, first_elapsed_ms, all_elapsed_ms, offsets, loaded


def _measure_candidate_snapshot_memory(root: Path, paths: list[str]) -> tuple[int, int]:
    """Measure only the Python allocation retained by one active snapshot.

    ``tracemalloc`` materially changes timing, so this intentionally runs
    outside the timed benchmark repetitions.
    """

    service = FaceIndexService(db_path=root / "faces.sqlite3")
    tracemalloc.start()
    snapshot = service.prepare_folder_review_candidate_snapshot(str(root), candidate_paths=paths)
    retained_bytes, peak_bytes = tracemalloc.get_traced_memory()
    if len(snapshot.source_paths) != len(paths):
        raise RuntimeError("Candidate snapshot unexpectedly dropped fixture paths")
    tracemalloc.stop()
    return int(retained_bytes), int(peak_bytes)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generated progressive Faces review paging benchmark.")
    parser.add_argument("--photos", type=int, default=1_001)
    parser.add_argument("--faces", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    faces = int(args.faces) if args.faces is not None else int(args.photos)
    if args.photos < 1 or args.repeats < 1 or faces < args.photos:
        parser.error("photos/repeats must be positive and faces must be at least photos")

    baseline_first_page_samples: list[float] = []
    baseline_all_pages_samples: list[float] = []
    snapshot_prepare_samples: list[float] = []
    first_page_samples: list[float] = []
    all_pages_samples: list[float] = []
    page_offsets: list[int] = []
    loaded_count = 0
    profile_hotspots: list[dict[str, object]] = []
    with TemporaryDirectory(prefix="clusterlens-face-review-paging-") as temporary:
        temporary_root = Path(temporary)
        for repeat in range(args.repeats):
            _prepare_ms, baseline_first_ms, baseline_all_ms, _offsets, _loaded = _run_once(
                temporary_root / f"baseline-{repeat:02d}",
                args.photos,
                faces,
                use_candidate_snapshot=False,
            )
            prepare_ms, first_ms, all_ms, offsets, loaded = _run_once(
                temporary_root / f"snapshot-{repeat:02d}",
                args.photos,
                faces,
                use_candidate_snapshot=True,
            )
            baseline_first_page_samples.append(baseline_first_ms)
            baseline_all_pages_samples.append(baseline_all_ms)
            snapshot_prepare_samples.append(prepare_ms)
            first_page_samples.append(first_ms)
            all_pages_samples.append(all_ms)
            page_offsets = offsets
            loaded_count = loaded

        profile_root = temporary_root / "profile"
        profile_root.mkdir(parents=True, exist_ok=True)
        profile_paths = _fixture_paths(profile_root, args.photos)
        profile_service = FaceIndexService(db_path=profile_root / "faces.sqlite3")
        _seed_index(profile_service, profile_paths, faces)
        profile_snapshot = profile_service.prepare_folder_review_candidate_snapshot(
            str(profile_root), candidate_paths=profile_paths
        )
        _result, profile = profile_call(
            lambda: profile_service.load_folder_review_image_page(
                str(profile_root),
                offset=0,
                limit=FACE_REVIEW_PAGE_MAX_IMAGES,
                candidate_paths=profile_paths,
                _candidate_snapshot=profile_snapshot,
            )
        )
        profile_hotspots = profile.hotspots

        memory_root = temporary_root / "memory"
        memory_root.mkdir(parents=True, exist_ok=True)
        snapshot_retained_bytes, snapshot_peak_bytes = _measure_candidate_snapshot_memory(
            memory_root, _fixture_paths(memory_root, args.photos)
        )

    expected_offsets = list(range(0, args.photos, FACE_REVIEW_PAGE_MAX_IMAGES))
    if loaded_count != args.photos or page_offsets != expected_offsets:
        raise RuntimeError("Progressive review did not return every fixture item in source-order pages")
    print(
        json.dumps(
            {
                "operation": "progressive_face_folder_review_pages",
                "fixture": {
                    "photos": args.photos,
                    "faces": faces,
                    "source": "temporary-seeded-sqlite-no-media",
                },
                "page_size": FACE_REVIEW_PAGE_MAX_IMAGES,
                "repeats": args.repeats,
                "baseline_without_snapshot": {
                    "first_page_timings_ms": [round(value, 3) for value in baseline_first_page_samples],
                    "first_page_median_ms": round(statistics.median(baseline_first_page_samples), 3),
                    "all_pages_timings_ms": [round(value, 3) for value in baseline_all_pages_samples],
                    "all_pages_median_ms": round(statistics.median(baseline_all_pages_samples), 3),
                },
                "candidate_snapshot_prepare_timings_ms": [round(value, 3) for value in snapshot_prepare_samples],
                "candidate_snapshot_prepare_median_ms": round(statistics.median(snapshot_prepare_samples), 3),
                "candidate_snapshot_python_retained_bytes": snapshot_retained_bytes,
                "candidate_snapshot_python_peak_bytes": snapshot_peak_bytes,
                "optimized_with_snapshot": {
                    "first_page_timings_ms": [round(value, 3) for value in first_page_samples],
                    "first_page_median_ms": round(statistics.median(first_page_samples), 3),
                    "all_pages_timings_ms": [round(value, 3) for value in all_pages_samples],
                    "all_pages_median_ms": round(statistics.median(all_pages_samples), 3),
                },
                "first_page_timings_ms": [round(value, 3) for value in first_page_samples],
                "first_page_median_ms": round(statistics.median(first_page_samples), 3),
                "all_pages_timings_ms": [round(value, 3) for value in all_pages_samples],
                "all_pages_median_ms": round(statistics.median(all_pages_samples), 3),
                "page_offsets": page_offsets,
                "loaded_count": loaded_count,
                "profile_hotspots": profile_hotspots,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
