#!/usr/bin/env python3
"""Benchmark the generated, source-read-only registered-library catalog."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from app.services.discovery import ImageDiscoveryService
from app.services.library_catalog import CatalogAsset, CatalogQuery, LibraryCatalogService
from apps.shared.profile_support import profile_call


def _create_fixture(root: Path, count: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        path = root / f"photo-{index:04d}.jpg"
        color = ((index * 17) % 255, (index * 47) % 255, (index * 79) % 255)
        Image.new("RGB", (64, 48), color).save(path, "JPEG", quality=84)
        if index % 8 == 0:
            path.with_suffix(".xmp").write_text(
                f"<dc:subject>ocean archive fixture {index:04d}</dc:subject>", encoding="utf-8"
            )


def _scan_fixture(root: Path, db_path: Path) -> tuple[LibraryCatalogService, float, dict[str, int]]:
    catalog = LibraryCatalogService(db_path=db_path)
    library_root = catalog.register_root(root)
    started = time.perf_counter()
    result = catalog.scan_root(library_root.root_id)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return catalog, elapsed_ms, result


def _seed_timeline_rows(catalog: LibraryCatalogService, root: Path, count: int) -> str:
    """Create deterministic catalog rows without creating or reading media files."""

    root.mkdir(parents=True, exist_ok=True)
    library_root = catalog.register_root(root)
    assets: list[tuple[CatalogAsset, int]] = []
    for index in range(count):
        month_offset = index // 100
        year = 2026 - month_offset // 12
        month = 12 - month_offset % 12
        captured = f"{year:04d}-{month:02d}-15T12:00:00+00:00"
        assets.append(
            (
                CatalogAsset(
                    str(root / f"timeline-{index:05d}.jpg"),
                    library_root.root_id,
                    captured,
                    "fixture",
                    captured,
                    "Fixture camera",
                    64,
                    48,
                    1,
                    ".jpg",
                    mtime_ns=index + 1,
                ),
                index + 1,
            )
        )
    catalog._upsert_assets(assets)  # Fixture setup; query timing stays source-read-only.
    return library_root.root_id


def _benchmark_timeline(temporary_root: Path, *, photos: int, repeats: int) -> dict[str, object]:
    samples: list[float] = []
    profile_hotspots: list[dict[str, object]] = []
    for index in range(repeats):
        catalog = LibraryCatalogService(db_path=temporary_root / f"timeline-{index:02d}.sqlite3")
        root_id = _seed_timeline_rows(catalog, temporary_root / f"timeline-root-{index:02d}", photos)
        query = CatalogQuery(root_ids=(root_id,), offset=480, limit=240)
        catalog.query_timeline(query)  # Warm SQLite structures; no source reads.
        started = time.perf_counter()
        timeline = catalog.query_timeline(query)
        samples.append((time.perf_counter() - started) * 1000.0)
        if timeline.total_count != photos or len(timeline.image_paths) != photos:
            raise RuntimeError("Timeline query did not return every deterministic fixture row")
        if index == 0:
            profile_catalog = LibraryCatalogService(db_path=temporary_root / "timeline-profile.sqlite3")
            profile_root = _seed_timeline_rows(profile_catalog, temporary_root / "timeline-profile-root", photos)
            _result, profile = profile_call(
                lambda: profile_catalog.query_timeline(CatalogQuery(root_ids=(profile_root,), limit=240))
            )
            profile_hotspots = profile.hotspots
    return {
        "photos": photos,
        "timings_ms": [round(value, 3) for value in samples],
        "median_ms": round(statistics.median(samples), 3),
        "profile_hotspots": profile_hotspots,
    }


def _benchmark_multi_root_discovery(temporary_root: Path, *, photos_per_root: int, repeats: int) -> dict[str, object]:
    """Measure deterministic root-union discovery without catalog or model work."""

    first_root = temporary_root / "active-first"
    nested_root = first_root / "nested"
    second_root = temporary_root / "active-second"
    _create_fixture(nested_root, photos_per_root)
    _create_fixture(second_root, photos_per_root)
    roots = [str(first_root), str(nested_root), str(second_root)]
    expected_count = photos_per_root * 2
    samples: list[float] = []
    profile_hotspots: list[dict[str, object]] = []
    for index in range(repeats):
        discovery = ImageDiscoveryService()
        started = time.perf_counter()
        result = discovery.discover_roots_result(roots, recursive=True)
        samples.append((time.perf_counter() - started) * 1000.0)
        if result.image_count != expected_count or len(set(result.paths)) != expected_count:
            raise RuntimeError("Multi-root discovery did not return the expected de-duplicated union")
        if index == 0:
            _result, profile = profile_call(
                lambda: ImageDiscoveryService().discover_roots_result(roots, recursive=True)
            )
            profile_hotspots = profile.hotspots
    return {
        "active_roots": 3,
        "effective_roots": 2,
        "photos_per_effective_root": photos_per_root,
        "expected_photo_count": expected_count,
        "timings_ms": [round(value, 3) for value in samples],
        "median_ms": round(statistics.median(samples), 3),
        "profile_hotspots": profile_hotspots,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the deterministic registered-root local catalog.")
    parser.add_argument("--photos", type=int, default=240)
    parser.add_argument("--timeline-photos", type=int, default=1_200)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeline-only", action="store_true")
    parser.add_argument("--multi-root-discovery-only", action="store_true")
    args = parser.parse_args()
    if args.photos < 1 or args.timeline_photos < 1 or args.repeats < 1:
        parser.error("photos, timeline-photos, and repeats must be positive")

    scan_samples: list[float] = []
    query_samples: list[float] = []
    scan_result: dict[str, int] = {}
    profile_hotspots: list[dict[str, object]] = []
    with TemporaryDirectory(prefix="clusterlens-library-catalog-benchmark-") as temporary:
        temporary_root = Path(temporary)
        if args.multi_root_discovery_only:
            report = _benchmark_multi_root_discovery(
                temporary_root,
                photos_per_root=args.photos,
                repeats=args.repeats,
            )
            print(
                json.dumps(
                    {
                        "fixture": {"source": "generated-jpeg-root-union", **report},
                        "operation": "active_root_union_discovery",
                        "repeats": args.repeats,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        timeline_report = _benchmark_timeline(
            temporary_root,
            photos=args.timeline_photos,
            repeats=args.repeats,
        )
        if args.timeline_only:
            print(
                json.dumps(
                    {
                        "fixture": {
                            "photos": args.timeline_photos,
                            "source": "generated-catalog-rows-only",
                            "query_limit_ignored": 240,
                        },
                        "operation": "full_filtered_library_timeline_query_and_year_month_grouping",
                        "repeats": args.repeats,
                        "timeline": timeline_report,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        for index in range(args.repeats):
            photos = temporary_root / f"photos-{index:02d}"
            _create_fixture(photos, args.photos)
            catalog, elapsed_ms, scan_result = _scan_fixture(photos, temporary_root / f"catalog-{index:02d}.sqlite3")
            scan_samples.append(elapsed_ms)
            if index == 0:
                catalog.query_assets(CatalogQuery(text="ocean", limit=100))
            started = time.perf_counter()
            page = catalog.query_assets(CatalogQuery(text="ocean", limit=100))
            query_samples.append((time.perf_counter() - started) * 1000.0)
            if page.total_count != (args.photos + 7) // 8:
                raise RuntimeError(f"Unexpected deterministic text-result count: {page.total_count}")

        profile_photos = temporary_root / "profile-photos"
        _create_fixture(profile_photos, args.photos)
        profile_catalog = LibraryCatalogService(db_path=temporary_root / "profile.sqlite3")
        profile_root = profile_catalog.register_root(profile_photos)
        _result, profile = profile_call(lambda: profile_catalog.scan_root(profile_root.root_id))
        profile_hotspots = profile.hotspots

    print(
        json.dumps(
            {
                "fixture": {
                    "photos": args.photos,
                    "xmp_sidecars": (args.photos + 7) // 8,
                    "image_dimensions": [64, 48],
                    "source": "generated-jpeg-and-xmp",
                },
                "operation": "registered_root_catalog_scan_and_fts_query",
                "repeats": args.repeats,
                "scan_timings_ms": [round(value, 3) for value in scan_samples],
                "scan_median_ms": round(statistics.median(scan_samples), 3),
                "query_timings_ms": [round(value, 3) for value in query_samples],
                "query_median_ms": round(statistics.median(query_samples), 3),
                "scan_result": scan_result,
                "profile_hotspots": profile_hotspots,
                "timeline": timeline_report,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
