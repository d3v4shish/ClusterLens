from __future__ import annotations

import json
from pathlib import Path
import statistics
from tempfile import TemporaryDirectory
import time

from PIL import Image

from app.services.face_region_metadata import FaceRegionMetadataService, FaceRegionUpdate
from app.services.gallery_actions import GalleryActionService


REPEATS = 5
IMAGE_SIZE = (1600, 1200)
REGION_COUNT = 8


def _updates(name_prefix: str) -> list[FaceRegionUpdate]:
    updates: list[FaceRegionUpdate] = []
    for index in range(REGION_COUNT):
        column = index % 4
        row = index // 4
        updates.append(
            FaceRegionUpdate(
                0.05 + (column * 0.23),
                0.10 + (row * 0.42),
                0.16,
                0.24,
                f"{name_prefix} {index + 1}",
            )
        )
    return updates


def main() -> int:
    """Measure only generated local JPEG/XMP region-update work."""

    with TemporaryDirectory(prefix="clusterlens-face-region-benchmark-") as temporary:
        root = Path(temporary)
        image_path = root / "fixture.jpg"
        Image.new("RGB", IMAGE_SIZE, (96, 128, 160)).save(image_path, quality=92)
        action_service = GalleryActionService(
            audit_log_path=root / "audit.jsonl",
            journal_path=root / "journal.sqlite3",
            temp_dir=root / "temporary",
        )
        service = FaceRegionMetadataService(action_service=action_service)

        warmup = service.update(image_path, _updates("Warmup"))
        if not warmup.succeeded:
            raise RuntimeError(warmup.error)

        timings_ms: list[float] = []
        for repeat in range(REPEATS):
            started = time.perf_counter()
            result = service.update(image_path, _updates(f"Run {repeat + 1}"))
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if not result.succeeded:
                raise RuntimeError(result.error)
            document = service.read(image_path)
            if len(document.regions) != REGION_COUNT or document.storage != "embedded":
                raise RuntimeError("Face-region metadata round trip did not preserve the generated fixture.")
            timings_ms.append(elapsed_ms)

        print(
            json.dumps(
                {
                    "fixture": {
                        "image_size": IMAGE_SIZE,
                        "region_count": REGION_COUNT,
                        "source": "generated-local-jpeg",
                    },
                    "operation": "embedded_jpeg_xmp_region_merge_and_readback",
                    "repeats": REPEATS,
                    "timings_ms": [round(value, 3) for value in timings_ms],
                    "median_ms": round(statistics.median(timings_ms), 3),
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
