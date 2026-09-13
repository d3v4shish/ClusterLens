#!/usr/bin/env python3
"""Measure deterministic SQLite-only queries used by the Tags workspace."""

from __future__ import annotations

import json
import statistics
import tempfile
import time
from pathlib import Path

from app.services.image_tags import ImageTagService


PHOTO_COUNT = 25_000
TAG_COUNT = 500
REPETITIONS = 7


def _build_fixture(db_path: Path) -> tuple[ImageTagService, str]:
    service = ImageTagService(db_path=db_path)
    rows: list[tuple[str, str, str, int, int, str, str]] = []
    folder_scope = "/generated-tags/folder-003"
    for photo_index in range(PHOTO_COUNT):
        folder_index = photo_index % 10
        image_path = f"/generated-tags/folder-{folder_index:03d}/photo-{photo_index:05d}.jpg"
        for offset in (0, 17, 89):
            tag_index = (photo_index + offset) % TAG_COUNT
            tag = f"tag-{tag_index:03d}"
            rows.append((image_path, tag, tag, 0, 0, "fixture", "2000-01-01T00:00:00+00:00"))
    with service._connection() as connection:
        connection.executemany(
            """
            INSERT INTO image_tags(image_path, tag_norm, display_tag, mtime_ns, file_size, source, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    return service, folder_scope


def _measure(callback) -> list[float]:
    callback()
    samples: list[float] = []
    for _ in range(REPETITIONS):
        started = time.perf_counter()
        callback()
        samples.append((time.perf_counter() - started) * 1000)
    return samples


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="clusterlens-tag-workspace-") as temporary:
        service, folder_scope = _build_fixture(Path(temporary) / "tags.sqlite3")
        inventory_samples = _measure(lambda: service.query_tag_inventory(limit=100, offset=0))
        inventory_followup_samples = _measure(
            lambda: service.query_tag_inventory(limit=100, offset=100, include_total=False)
        )
        inventory_followup_with_total_samples = _measure(
            lambda: service.query_tag_inventory(limit=100, offset=100)
        )
        scoped_inventory_samples = _measure(
            lambda: service.query_tag_inventory(scope_path=folder_scope, limit=100, offset=0)
        )
        photo_samples = _measure(
            lambda: service.query_tagged_paths("tag-003", scope_path=folder_scope, limit=200, offset=0)
        )
        photo_followup_samples = _measure(
            lambda: service.query_tagged_paths("tag-003", limit=50, offset=50, include_total=False)
        )
        photo_followup_with_total_samples = _measure(
            lambda: service.query_tagged_paths("tag-003", limit=50, offset=50)
        )
        inventory = service.query_tag_inventory(limit=100, offset=0)
        scoped_inventory = service.query_tag_inventory(scope_path=folder_scope, limit=100, offset=0)
        tagged_paths = service.query_tagged_paths("tag-003", scope_path=folder_scope, limit=200, offset=0)
        report = {
            "fixture": {
                "photos": PHOTO_COUNT,
                "tags": TAG_COUNT,
                "tag_rows": PHOTO_COUNT * 3,
                "repetitions": REPETITIONS,
                "media_files_created": 0,
            },
            "global_inventory_page": {
                "median_ms": round(statistics.median(inventory_samples), 3),
                "samples_ms": [round(value, 3) for value in inventory_samples],
                "total_tags": inventory.total_count,
                "returned_tags": len(inventory.items),
            },
            "global_inventory_followup_page": {
                "median_ms": round(statistics.median(inventory_followup_samples), 3),
                "samples_ms": [round(value, 3) for value in inventory_followup_samples],
                "returned_tags": len(service.query_tag_inventory(limit=100, offset=100, include_total=False).items),
            },
            "global_inventory_followup_page_with_total": {
                "median_ms": round(statistics.median(inventory_followup_with_total_samples), 3),
                "samples_ms": [round(value, 3) for value in inventory_followup_with_total_samples],
            },
            "folder_inventory_page": {
                "median_ms": round(statistics.median(scoped_inventory_samples), 3),
                "samples_ms": [round(value, 3) for value in scoped_inventory_samples],
                "total_tags": scoped_inventory.total_count,
                "returned_tags": len(scoped_inventory.items),
            },
            "folder_tagged_photo_page": {
                "median_ms": round(statistics.median(photo_samples), 3),
                "samples_ms": [round(value, 3) for value in photo_samples],
                "total_paths": tagged_paths.total_count,
                "returned_paths": len(tagged_paths.paths),
            },
            "global_tagged_photo_followup_page": {
                "median_ms": round(statistics.median(photo_followup_samples), 3),
                "samples_ms": [round(value, 3) for value in photo_followup_samples],
                "returned_paths": len(service.query_tagged_paths("tag-003", limit=50, offset=50, include_total=False).paths),
            },
            "global_tagged_photo_followup_page_with_total": {
                "median_ms": round(statistics.median(photo_followup_with_total_samples), 3),
                "samples_ms": [round(value, 3) for value in photo_followup_with_total_samples],
            },
        }
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
