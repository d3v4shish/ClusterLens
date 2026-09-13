from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from PIL import Image

from app.services.thumbnails import ThumbnailService
from apps.shared.profile_support import profile_call


def _thumbnail_service(cache_dir: Path) -> ThumbnailService:
    service = ThumbnailService()
    service.settings = SimpleNamespace(
        thumbnail_cache_dir=cache_dir,
        thumbnail_cache_max_bytes=1024 * 1024 * 1024,
    )
    return service


def _prepare_cache(cache_dir: Path, entry_count: int) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    template = cache_dir / "template.webp"
    Image.new("RGB", (8, 8), (48, 96, 144)).save(template, format="WEBP")
    payload = template.read_bytes()
    template.unlink()
    for index in range(max(1, int(entry_count))):
        (cache_dir / f"fixture-{index:05d}.webp").write_bytes(payload)


def _open_services(cache_dir: Path, service_count: int) -> list[float]:
    timings_ms: list[float] = []
    for _index in range(max(1, int(service_count))):
        service = _thumbnail_service(cache_dir)
        started = time.perf_counter()
        connection = service._connect_disk_index()
        connection.close()
        timings_ms.append((time.perf_counter() - started) * 1000.0)
    return timings_ms


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark bounded thumbnail-index recovery using generated WebP cache entries."
    )
    parser.add_argument("--entries", type=int, default=1024)
    parser.add_argument("--services", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.entries < 1 or args.services < 1 or args.repeats < 1:
        parser.error("entries, services, and repeats must all be positive")

    first_service_ms: list[float] = []
    total_services_ms: list[float] = []
    profile_hotspots: list[dict[str, object]] = []
    with TemporaryDirectory(prefix="clusterlens-thumbnail-index-benchmark-") as temporary:
        root = Path(temporary)
        for repeat in range(args.repeats):
            cache_dir = root / f"cache-{repeat:02d}"
            _prepare_cache(cache_dir, args.entries)
            timings = _open_services(cache_dir, args.services)
            first_service_ms.append(timings[0])
            total_services_ms.append(sum(timings))

        profile_cache = root / "profile-cache"
        _prepare_cache(profile_cache, args.entries)
        _result, profile = profile_call(lambda: _open_services(profile_cache, args.services))
        profile_hotspots = profile.hotspots

    print(
        json.dumps(
            {
                "fixture": {
                    "entries": args.entries,
                    "services": args.services,
                    "source": "generated-valid-webp-cache",
                },
                "operation": "thumbnail_index_open_and_reconcile",
                "repeats": args.repeats,
                "first_service_timings_ms": [round(value, 3) for value in first_service_ms],
                "first_service_median_ms": round(statistics.median(first_service_ms), 3),
                "all_services_timings_ms": [round(value, 3) for value in total_services_ms],
                "all_services_median_ms": round(statistics.median(total_services_ms), 3),
                "profile_hotspots": profile_hotspots,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
