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
from pathlib import Path
from tempfile import TemporaryDirectory

from app.services.face_search import FACE_REVIEW_PAGE_MAX_IMAGES, FaceFolderReviewPage, FaceIndexService
from apps.shared.profile_support import profile_call


def _fixture_paths(root: Path, photos: int) -> list[str]:
    return [str(root / f"review-{index:05d}.jpg") for index in range(photos)]


def _seed_index(service: FaceIndexService, paths: list[str]) -> None:
    face_rows = [
        (
            path,
            0,
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
        for path in paths
    ]
    scan_rows = [(path, 1, 1, 1, 100, 100) for path in paths]
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


def _load_all_pages(service: FaceIndexService, root: Path, paths: list[str]) -> tuple[float, list[int], int]:
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


def _run_once(root: Path, photos: int) -> tuple[float, float, list[int], int]:
    root.mkdir(parents=True, exist_ok=True)
    paths = _fixture_paths(root, photos)
    service = FaceIndexService(db_path=root / "faces.sqlite3")
    _seed_index(service, paths)

    first_started = time.perf_counter()
    first_page = service.load_folder_review_image_page(
        str(root),
        offset=0,
        limit=FACE_REVIEW_PAGE_MAX_IMAGES,
        candidate_paths=paths,
    )
    first_elapsed_ms = (time.perf_counter() - first_started) * 1000.0
    if len(first_page.items) != min(photos, FACE_REVIEW_PAGE_MAX_IMAGES):
        raise RuntimeError("First progressive page did not use the configured bound")

    all_elapsed_ms, offsets, loaded = _load_all_pages(service, root, paths)
    return first_elapsed_ms, all_elapsed_ms, offsets, loaded


def main() -> int:
    parser = argparse.ArgumentParser(description="Generated progressive Faces review paging benchmark.")
    parser.add_argument("--photos", type=int, default=1_001)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.photos < 1 or args.repeats < 1:
        parser.error("photos and repeats must be positive")

    first_page_samples: list[float] = []
    all_pages_samples: list[float] = []
    page_offsets: list[int] = []
    loaded_count = 0
    profile_hotspots: list[dict[str, object]] = []
    with TemporaryDirectory(prefix="clusterlens-face-review-paging-") as temporary:
        temporary_root = Path(temporary)
        for repeat in range(args.repeats):
            first_ms, all_ms, offsets, loaded = _run_once(temporary_root / f"run-{repeat:02d}", args.photos)
            first_page_samples.append(first_ms)
            all_pages_samples.append(all_ms)
            page_offsets = offsets
            loaded_count = loaded

        profile_root = temporary_root / "profile"
        profile_root.mkdir(parents=True, exist_ok=True)
        profile_paths = _fixture_paths(profile_root, args.photos)
        profile_service = FaceIndexService(db_path=profile_root / "faces.sqlite3")
        _seed_index(profile_service, profile_paths)
        _result, profile = profile_call(
            lambda: profile_service.load_folder_review_image_page(
                str(profile_root),
                offset=0,
                limit=FACE_REVIEW_PAGE_MAX_IMAGES,
                candidate_paths=profile_paths,
            )
        )
        profile_hotspots = profile.hotspots

    expected_offsets = list(range(0, args.photos, FACE_REVIEW_PAGE_MAX_IMAGES))
    if loaded_count != args.photos or page_offsets != expected_offsets:
        raise RuntimeError("Progressive review did not return every fixture item in source-order pages")
    print(
        json.dumps(
            {
                "operation": "progressive_face_folder_review_pages",
                "fixture": {
                    "photos": args.photos,
                    "indexed_faces_per_photo": 1,
                    "source": "temporary-seeded-sqlite-no-media",
                },
                "page_size": FACE_REVIEW_PAGE_MAX_IMAGES,
                "repeats": args.repeats,
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
