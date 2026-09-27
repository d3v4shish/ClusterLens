#!/usr/bin/env python3
"""Benchmark the first useful All Faces publication query.

The fixture contains only fresh, seeded SQLite rows.  It never opens a source
photo, decodes a crop, constructs a face model, or reads user runtime data.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from app.services.face_search import FaceIndexService
from apps.shared.profile_support import profile_call


def _seed_index(service: FaceIndexService, root: Path, faces: int) -> None:
    rows = []
    labels = []
    pending = []
    for index in range(faces):
        image_path = str(root / f"face-{index:05d}.jpg")
        rows.append(
            (
                image_path, 0, "[10, 10, 52, 56]", 0.99, b"fixture-embedding",
                "clean", 1.0, "[]", 1, 0, 0, 1, 1,
            )
        )
        if index % 3 == 0:
            labels.append((image_path, 0, "Alice", 0.99))
        elif index % 11 == 0:
            pending.append((image_path, 0, "Proposed", 0.80, "fixture", "2026-01-01T00:00:00"))
    with service._connect() as connection:  # Fixture setup; timed calls use public APIs.
        connection.executemany(
            """
            INSERT INTO face_index(
                image_path, face_index, bbox_json, face_confidence, embedding,
                quality_status, quality_score, quality_reasons_json,
                quality_revision, is_tiny, hidden, mtime_ns, file_size
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        connection.executemany(
            """
            INSERT INTO face_labels(image_path, face_index, person_name, confidence)
            VALUES (?, ?, ?, ?)
            """,
            labels,
        )
        connection.executemany(
            """
            INSERT INTO pending_face_labels(image_path, face_index, person_name, confidence, source, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            pending,
        )


def _load_initial_page(service: FaceIndexService) -> tuple[float, int, int]:
    started = time.perf_counter()
    page = service.load_face_album_initial_page(
        group_limit=100,
        member_limit=500,
        group_kinds=("named", "unlabeled"),
        include_tiny_faces=False,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    group_kinds = {group.group_kind for group in page.groups.items}
    if "pending" in group_kinds or group_kinds != {"named", "unlabeled"}:
        raise RuntimeError("normal All Faces page included a pending-only group")
    if page.members is None or len(page.members.items) != min(500, faces_for_group(page.groups.items[0])):
        raise RuntimeError("first All Faces member page did not honour its 500-row bound")
    return elapsed_ms, len(page.groups.items), len(page.members.items)


def faces_for_group(group) -> int:
    return int(group.face_count)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generated People / All Faces first-page benchmark.")
    parser.add_argument("--faces", type=int, default=5_000)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.faces < 3 or args.repeats < 1:
        parser.error("faces must be at least 3 and repeats must be positive")

    samples: list[float] = []
    group_count = member_count = 0
    hotspots: list[dict[str, object]] = []
    with TemporaryDirectory(prefix="clusterlens-people-faces-") as temporary:
        temporary_root = Path(temporary)
        for repeat in range(args.repeats):
            root = temporary_root / f"run-{repeat:02d}"
            root.mkdir()
            service = FaceIndexService(db_path=root / "faces.sqlite3")
            _seed_index(service, root, args.faces)
            elapsed_ms, group_count, member_count = _load_initial_page(service)
            samples.append(elapsed_ms)

        profile_root = temporary_root / "profile"
        profile_root.mkdir()
        profile_service = FaceIndexService(db_path=profile_root / "faces.sqlite3")
        _seed_index(profile_service, profile_root, args.faces)
        _result, profile = profile_call(lambda: _load_initial_page(profile_service))
        hotspots = profile.hotspots

    print(json.dumps({
        "operation": "people_all_faces_initial_page",
        "fixture": {"faces": args.faces, "source": "temporary-seeded-sqlite-no-media"},
        "repeats": args.repeats,
        "timings_ms": [round(value, 3) for value in samples],
        "median_ms": round(statistics.median(samples), 3),
        "groups": group_count,
        "first_group_members": member_count,
        "profile_hotspots": hotspots,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
