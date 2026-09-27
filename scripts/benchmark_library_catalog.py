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
from app.services.source_admission import SourceAdmissionPolicy
from apps.shared.profile_support import profile_call


FILENAME_TIME_RULES = (
    "{date:DDMMYYYY}{sequence:6}",
    "IMG_Epoch_{epoch:s}",
    "IMG_Epoch_{epoch:ms}",
)


def _fixture_file_name(index: int) -> str:
    """Exercise each bounded filename-time parser path without random input."""

    epoch_seconds = 1_704_067_200 + index  # 2024-01-01T00:00:00Z onward.
    match index % 4:
        case 0:
            return f"06052024{index:06d}.jpg"
        case 1:
            return f"IMG_Epoch_{epoch_seconds}.jpg"
        case 2:
            return f"IMG_Epoch_{epoch_seconds * 1000}.jpg"
        case _:
            return f"hash_a{epoch_seconds}b.jpg"


def _create_fixture(
    root: Path,
    count: int,
    *,
    filename_time_rules: bool = False,
    source_filters: bool = False,
) -> tuple[int, int]:
    """Create a deterministic fixture and return admitted image/XMP counts.

    The optional source-filter fixture adds disjoint thumbnail-name,
    undersized-dimension, and undersized-byte candidates.  It measures the
    enabled policy's real admission work without reading any user media.
    """

    root.mkdir(parents=True, exist_ok=True)
    admitted = 0
    admitted_xmp = 0
    for index in range(count):
        file_name = _fixture_file_name(index) if filename_time_rules else f"photo-{index:04d}.jpg"
        kind = index % 10 if source_filters else -1
        if kind == 0:
            path = root / f"{Path(file_name).stem}_thumb.jpg"
        else:
            path = root / file_name
        color = ((index * 17) % 255, (index * 47) % 255, (index * 79) % 255)
        if kind == 1:
            Image.new("RGB", (32, 24), color).save(path, "JPEG", quality=84)
        elif kind == 2:
            path.write_bytes(b"small")
        else:
            Image.new("RGB", (64, 48), color).save(path, "JPEG", quality=84)
        if index % 8 == 0:
            path.with_suffix(".xmp").write_text(
                f"<dc:subject>ocean archive fixture {index:04d}</dc:subject>", encoding="utf-8"
            )
        if kind not in {0, 1, 2}:
            admitted += 1
            admitted_xmp += int(index % 8 == 0)
    return admitted, admitted_xmp


def _scan_fixture(
    root: Path,
    db_path: Path,
    *,
    source_admission_policy: SourceAdmissionPolicy | None = None,
) -> tuple[LibraryCatalogService, float, dict[str, int]]:
    catalog = LibraryCatalogService(
        db_path=db_path,
        filename_date_patterns=FILENAME_TIME_RULES,
        filename_epoch_heuristic=True,
        source_admission_policy=source_admission_policy,
    )
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


def _benchmark_fts_index(temporary_root: Path, *, assets: int, repeats: int) -> dict[str, object]:
    """Measure one fixed, source-free FTS corpus and a warm text lookup."""

    index_samples: list[float] = []
    query_samples: list[float] = []
    profile_hotspots: list[dict[str, object]] = []
    expected_matches = sum(index % 7 == 0 for index in range(assets))
    for repeat in range(repeats):
        catalog = LibraryCatalogService(db_path=temporary_root / f"fts-{repeat:02d}.sqlite3")
        root_path = temporary_root / f"fts-root-{repeat:02d}"
        root_path.mkdir(parents=True, exist_ok=True)
        root = catalog.register_root(root_path)
        fixture = [
            (
                CatalogAsset(
                    str(root_path / f"catalog-{index:05d}.jpg"),
                    root.root_id,
                    "2026-01-01T00:00:00+00:00",
                    "fixture",
                    "2026-01-01T00:00:00+00:00",
                    "Fixture camera",
                    64,
                    48,
                    1,
                    ".jpg",
                    exif={"subject": "harbor needle" if index % 7 == 0 else "ordinary fixture"},
                    xmp_text="harbor needle fixture" if index % 7 == 0 else "ordinary fixture",
                    mtime_ns=index + 1,
                ),
                index + 1,
            )
            for index in range(assets)
        ]
        started = time.perf_counter()
        catalog._upsert_assets(fixture)  # Fixed in-memory catalog corpus; no source media is read.
        index_samples.append((time.perf_counter() - started) * 1000.0)
        query = CatalogQuery(root_ids=(root.root_id,), text="harbor needle", limit=assets)
        catalog.query_assets(query)  # Warm the deterministic SQLite query path.
        started = time.perf_counter()
        page = catalog.query_assets(query)
        query_samples.append((time.perf_counter() - started) * 1000.0)
        if page.total_count != expected_matches:
            raise RuntimeError(f"FTS fixture returned {page.total_count}, expected {expected_matches}")
        if repeat == 0:
            profiled = LibraryCatalogService(db_path=temporary_root / "fts-profile.sqlite3")
            profile_root_path = temporary_root / "fts-profile-root"
            profile_root_path.mkdir(parents=True, exist_ok=True)
            profile_root = profiled.register_root(profile_root_path)
            profile_fixture = [
                (CatalogAsset(
                    str(profile_root_path / f"catalog-{index:05d}.jpg"), profile_root.root_id,
                    "2026-01-01T00:00:00+00:00", "fixture", "2026-01-01T00:00:00+00:00", "Fixture camera",
                    64, 48, 1, ".jpg", exif={"subject": "harbor needle" if index % 7 == 0 else "ordinary fixture"},
                    xmp_text="harbor needle fixture" if index % 7 == 0 else "ordinary fixture", mtime_ns=index + 1,
                ), index + 1)
                for index in range(assets)
            ]
            _result, profile = profile_call(lambda: profiled._upsert_assets(profile_fixture))
            profile_hotspots = profile.hotspots
    return {
        "assets": assets,
        "expected_text_matches": expected_matches,
        "fts_available": bool(catalog._fts_available),  # noqa: SLF001 - benchmark reports the selected SQLite path.
        "index_timings_ms": [round(value, 3) for value in index_samples],
        "index_median_ms": round(statistics.median(index_samples), 3),
        "warm_query_timings_ms": [round(value, 3) for value in query_samples],
        "warm_query_median_ms": round(statistics.median(query_samples), 3),
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
    parser.add_argument("--fts-assets", type=int, default=5_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeline-only", action="store_true")
    parser.add_argument("--fts-only", action="store_true")
    parser.add_argument("--multi-root-discovery-only", action="store_true")
    parser.add_argument(
        "--source-filters",
        action="store_true",
        help="Exercise thumbnail, byte-size, and dimension exclusions during generated catalog scans.",
    )
    args = parser.parse_args()
    if args.photos < 1 or args.timeline_photos < 1 or args.fts_assets < 1 or args.repeats < 1:
        parser.error("photos, timeline-photos, fts-assets, and repeats must be positive")

    scan_samples: list[float] = []
    query_samples: list[float] = []
    scan_result: dict[str, int] = {}
    profile_hotspots: list[dict[str, object]] = []
    source_admission_policy = (
        SourceAdmissionPolicy(
            ignore_thumbnail_like=True,
            minimum_width=48,
            minimum_height=36,
            minimum_file_size_bytes=512,
        )
        if args.source_filters
        else SourceAdmissionPolicy()
    )
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
        if args.fts_only:
            report = _benchmark_fts_index(temporary_root, assets=args.fts_assets, repeats=args.repeats)
            print(
                json.dumps(
                    {
                        "fixture": {"assets": args.fts_assets, "source": "fixed-catalog-rows-only", "query": "harbor needle"},
                        "operation": "batched_catalog_upsert_and_warm_fts_query",
                        "repeats": args.repeats,
                        "fts": report,
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
                        "operation": "full_filtered_library_timeline_query_and_year_month_day_bucketing",
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
            expected_photos, expected_xmp = _create_fixture(
                photos,
                args.photos,
                filename_time_rules=True,
                source_filters=args.source_filters,
            )
            catalog, elapsed_ms, scan_result = _scan_fixture(
                photos,
                temporary_root / f"catalog-{index:02d}.sqlite3",
                source_admission_policy=source_admission_policy,
            )
            scan_samples.append(elapsed_ms)
            parsed = sum(
                int(scan_result.get(key, 0))
                for key in ("filename_dates", "filename_epochs", "raw_epochs")
            )
            if parsed != expected_photos:
                raise RuntimeError(f"Expected every admitted filename-time fixture row to parse, got {parsed}/{expected_photos}")
            if int(scan_result.get("discovered", 0)) != expected_photos:
                raise RuntimeError("Source admission did not return the expected deterministic fixture count")
            if int(scan_result.get("filtered", 0)) != args.photos - expected_photos:
                raise RuntimeError("Source admission did not report every deterministic exclusion")
            if index == 0:
                catalog.query_assets(CatalogQuery(text="ocean", limit=100))
            started = time.perf_counter()
            page = catalog.query_assets(CatalogQuery(text="ocean", limit=100))
            query_samples.append((time.perf_counter() - started) * 1000.0)
            if page.total_count != expected_xmp:
                raise RuntimeError(f"Unexpected deterministic text-result count: {page.total_count}")

        profile_photos = temporary_root / "profile-photos"
        _create_fixture(profile_photos, args.photos, filename_time_rules=True, source_filters=args.source_filters)
        profile_catalog = LibraryCatalogService(
            db_path=temporary_root / "profile.sqlite3",
            filename_date_patterns=FILENAME_TIME_RULES,
            filename_epoch_heuristic=True,
            source_admission_policy=source_admission_policy,
        )
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
                    "filename_time_rules": list(FILENAME_TIME_RULES),
                    "raw_epoch_heuristic": True,
                    "source_admission_policy": {
                        "ignore_thumbnail_like": bool(args.source_filters),
                        "minimum_width": 48 if args.source_filters else 0,
                        "minimum_height": 36 if args.source_filters else 0,
                        "minimum_file_size_bytes": 512 if args.source_filters else 0,
                    },
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
                "fts": _benchmark_fts_index(temporary_root, assets=args.fts_assets, repeats=args.repeats),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
