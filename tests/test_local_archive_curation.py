from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PIL import Image

from app.services.cluster_context import ClusterContextError, ClusterContextService, VisionLanguageResponse, VisionLanguageSettings
from app.services.duplicate_review import DuplicateReviewService
from app.services.library_catalog import CatalogAsset, CatalogQuery, FilenameCapture, LibraryCatalogService
from app.services.source_admission import SourceAdmissionPolicy
from app.services.people_cleanup import PeopleCleanupService
from app.services.perceptual_hash import PerceptualHashIndexService
from app.services.photo_metadata import PhotoEditDraft, PhotoEditService
from infra.cancel import Cancelled


class _StaticVisionProvider:
    provider_id = "ollama"

    def __init__(self) -> None:
        self.calls = 0

    def describe(self, **_kwargs) -> VisionLanguageResponse:
        self.calls += 1
        return VisionLanguageResponse(
            title="Beach walk",
            description="A dog walks beside the sea.",
            keywords=("dog", "beach", "sea"),
        )


class _PromptCapturingVisionProvider(_StaticVisionProvider):
    def __init__(self) -> None:
        super().__init__()
        self.prompt = ""

    def describe(self, **kwargs) -> VisionLanguageResponse:
        self.prompt = str(kwargs["prompt"])
        return super().describe(**kwargs)


class _StaticRemoteVisionProvider(_StaticVisionProvider):
    provider_id = "openai-compatible"


class LocalArchiveCurationTests(unittest.TestCase):
    def _catalog(self, directory: Path) -> LibraryCatalogService:
        return LibraryCatalogService(db_path=directory / "catalog.sqlite3")

    @staticmethod
    def _photo(path: Path, color: tuple[int, int, int]) -> None:
        Image.new("RGB", (80, 60), color).save(path, "JPEG")

    def test_registered_root_catalogs_enabled_assets_and_searches_xmp(self) -> None:
        with TemporaryDirectory() as raw:
            directory = Path(raw)
            photos = directory / "photos"
            photos.mkdir()
            photo = photos / "beach.jpg"
            self._photo(photo, (30, 80, 170))
            photo.with_suffix(".xmp").write_text("<dc:subject>ocean holiday</dc:subject>", encoding="utf-8")
            catalog = self._catalog(directory)
            root = catalog.register_root(photos)

            result = catalog.scan_root(root.root_id)
            self.assertEqual(1, result["discovered"])
            self.assertEqual(1, result["updated"])

            page = catalog.query_assets(CatalogQuery(text="ocean", limit=10))
            self.assertEqual(1, page.total_count)
            self.assertEqual(str(photo.resolve()), page.items[0].image_path)
            self.assertIn("ocean holiday", page.items[0].xmp_text)

            # The source photo is unchanged: sidecar edits must still refresh
            # catalog text and its FTS entry on the next scan.
            photo.with_suffix(".xmp").write_text("<dc:subject>mountain expedition</dc:subject>", encoding="utf-8")
            refreshed = catalog.scan_root(root.root_id)
            self.assertEqual(1, refreshed["updated"])
            self.assertEqual(1, catalog.query_assets(CatalogQuery(text="mountain", limit=10)).total_count)
            self.assertEqual(0, catalog.query_assets(CatalogQuery(text="ocean", limit=10)).total_count)

    def test_batched_catalog_upsert_replaces_fts_text_without_stale_hits(self) -> None:
        with TemporaryDirectory() as raw:
            directory = Path(raw)
            catalog = self._catalog(directory)
            root = catalog.register_root(directory)
            path = str(directory / "fixture.jpg")
            first = CatalogAsset(path, root.root_id, "", "fixture", "", "", 64, 48, 1, ".jpg", xmp_text="ocean", mtime_ns=1)
            revised = CatalogAsset(path, root.root_id, "", "fixture", "", "", 64, 48, 1, ".jpg", xmp_text="mountain", mtime_ns=2)

            catalog._upsert_assets(((first, 1), (revised, 2)))

            self.assertEqual(0, catalog.query_assets(CatalogQuery(text="ocean", limit=10)).total_count)
            self.assertEqual(1, catalog.query_assets(CatalogQuery(text="mountain", limit=10)).total_count)

            catalog.set_root_enabled(root.root_id, False)
            self.assertEqual(0, catalog.query_assets(CatalogQuery()).total_count)

    def test_catalog_scan_reports_exact_committed_subset_before_cancellation(self) -> None:
        with TemporaryDirectory() as raw:
            directory = Path(raw)
            photos = directory / "photos"
            photos.mkdir()
            for index in range(50):
                self._photo(photos / f"photo-{index:02d}.jpg", (index, 40, 80))
            catalog = self._catalog(directory)
            root = catalog.register_root(photos)
            committed: list[dict[str, int]] = []
            cancelled = False

            def _batch_committed(root_id: str, stats: dict[str, int]) -> None:
                nonlocal cancelled
                self.assertEqual(root.root_id, root_id)
                committed.append(dict(stats))
                cancelled = True

            with self.assertRaises(Cancelled):
                catalog.scan_root(
                    root.root_id,
                    cancel_check=lambda: cancelled,
                    on_committed_batch=_batch_committed,
                )

            self.assertEqual(48, committed[-1]["updated"])
            self.assertEqual(48, catalog.query_assets(CatalogQuery(limit=100)).total_count)
            self.assertEqual("", catalog.get_root(root.root_id).last_scan_at)

    def test_catalog_batch_is_atomic_across_process_exit_commit_boundaries(self) -> None:
        class _ProcessExit(BaseException):
            pass

        for boundary, expected_assets in (("before_commit", 0), ("after_commit", 1)):
            with self.subTest(boundary=boundary), TemporaryDirectory() as raw:
                directory = Path(raw)
                catalog = self._catalog(directory)
                root = catalog.register_root(directory)
                image_path = str(directory / "fixture.jpg")
                asset = CatalogAsset(image_path, root.root_id, "", "fixture", "", "", 64, 48, 1, ".jpg")

                def _exit_at_checkpoint(name: str) -> None:
                    if name == boundary:
                        raise _ProcessExit(name)

                catalog._catalog_write_checkpoint = _exit_at_checkpoint  # type: ignore[method-assign]
                with self.assertRaises(_ProcessExit):
                    catalog._upsert_assets(((asset, 1),))

                reopened = self._catalog(directory)
                self.assertEqual(expected_assets, reopened.query_assets(CatalogQuery(limit=10)).total_count)

    def test_catalog_batch_rolls_back_sqlite_io_and_disk_full_faults(self) -> None:
        faults = (
            sqlite3.OperationalError("injected I/O error"),
            sqlite3.OperationalError("database or disk is full"),
        )
        for fault in faults:
            with self.subTest(fault=str(fault)), TemporaryDirectory() as raw:
                directory = Path(raw)
                catalog = self._catalog(directory)
                root = catalog.register_root(directory)
                image_path = str(directory / "fixture.jpg")
                asset = CatalogAsset(image_path, root.root_id, "", "fixture", "", "", 64, 48, 1, ".jpg")

                def _fail_before_commit(name: str) -> None:
                    if name == "before_commit":
                        raise fault

                catalog._catalog_write_checkpoint = _fail_before_commit  # type: ignore[method-assign]
                with self.assertRaisesRegex(sqlite3.OperationalError, str(fault)):
                    catalog._upsert_assets(((asset, 1),))

                reopened = self._catalog(directory)
                self.assertEqual(0, reopened.query_assets(CatalogQuery(limit=10)).total_count)
                with sqlite3.connect(directory / "catalog.sqlite3") as connection:
                    self.assertEqual(("ok",), connection.execute("PRAGMA integrity_check").fetchone())

    def test_catalog_retry_reconciles_a_source_that_vanishes_after_discovery(self) -> None:
        with TemporaryDirectory() as raw:
            directory = Path(raw)
            photos = directory / "photos"
            photos.mkdir()
            image = photos / "vanishing.jpg"
            self._photo(image, (30, 60, 90))
            catalog = self._catalog(directory)
            root = catalog.register_root(photos)
            self.assertEqual(1, catalog.scan_root(root.root_id)["updated"])
            original_read = catalog._read_asset
            removed = False

            def _remove_after_discovery(*args, **kwargs):
                nonlocal removed
                if not removed:
                    image.unlink()
                    removed = True
                return original_read(*args, **kwargs)

            with patch.object(catalog, "_read_asset", side_effect=_remove_after_discovery):
                first = catalog.scan_root(root.root_id, force=True)
            self.assertEqual(1, first["updated"])
            self.assertEqual(1, catalog.query_assets(CatalogQuery(limit=10)).total_count)

            retry = catalog.scan_root(root.root_id)
            self.assertEqual(1, retry["removed"])
            self.assertEqual(0, catalog.query_assets(CatalogQuery(limit=10)).total_count)
            with sqlite3.connect(directory / "catalog.sqlite3") as connection:
                self.assertEqual(("ok",), connection.execute("PRAGMA integrity_check").fetchone())

    def test_catalog_source_filters_exclude_candidates_and_remove_stale_catalog_entries(self) -> None:
        with TemporaryDirectory() as raw:
            directory = Path(raw)
            photos = directory / "photos"
            photos.mkdir()
            full = photos / "family.jpg"
            thumbnail = photos / "family_thumb.jpg"
            undersized = photos / "small.jpg"
            self._photo(full, (20, 50, 80))
            self._photo(thumbnail, (50, 80, 120))
            Image.new("RGB", (40, 30), (80, 100, 120)).save(undersized, "JPEG")

            policy = {"value": SourceAdmissionPolicy()}
            catalog = LibraryCatalogService(
                db_path=directory / "catalog.sqlite3",
                source_admission_policy_provider=lambda: policy["value"],
            )
            root = catalog.register_root(photos)
            first = catalog.scan_root(root.root_id)
            self.assertEqual(3, first["discovered"])
            self.assertEqual(3, catalog.query_assets(CatalogQuery(limit=10)).total_count)

            policy["value"] = SourceAdmissionPolicy(
                ignore_thumbnail_like=True,
                minimum_width=60,
                minimum_height=50,
            )
            filtered = catalog.scan_root(root.root_id)

            self.assertEqual(1, filtered["discovered"])
            self.assertEqual(2, filtered["filtered"])
            self.assertEqual(2, filtered["removed"])
            page = catalog.query_assets(CatalogQuery(limit=10))
            self.assertEqual((str(full.resolve()),), tuple(item.image_path for item in page.items))

    def test_catalog_query_intersects_enabled_roots_with_active_root_scope(self) -> None:
        with TemporaryDirectory() as raw:
            directory = Path(raw)
            first_root = directory / "first"
            second_root = directory / "second"
            excluded_root = directory / "excluded"
            for root in (first_root, second_root, excluded_root):
                root.mkdir()
            first = first_root / "one.jpg"
            second = second_root / "two.jpg"
            excluded = excluded_root / "outside.jpg"
            for image_path, color in ((first, (20, 40, 60)), (second, (60, 80, 100)), (excluded, (100, 120, 140))):
                self._photo(image_path, color)
            catalog = self._catalog(directory)
            roots = [catalog.register_root(root) for root in (first_root, second_root, excluded_root)]
            for root in roots:
                catalog.scan_root(root.root_id)

            page = catalog.query_assets(
                CatalogQuery(
                    root_ids=tuple(root.root_id for root in roots),
                    scope_paths=(str(first_root), str(second_root)),
                    limit=20,
                )
            )

            self.assertEqual({str(first.resolve()), str(second.resolve())}, {item.image_path for item in page.items})

    def test_timeline_reads_all_filtered_paths_in_order_and_can_cancel(self) -> None:
        with TemporaryDirectory() as raw:
            directory = Path(raw)
            photos = directory / "photos"
            photos.mkdir()
            catalog = self._catalog(directory)
            root = catalog.register_root(photos)
            periods = ((2026, 3, 400), (2026, 2, 400), (2025, 12, 201))
            assets: list[tuple[CatalogAsset, int]] = []
            sequence = 0
            for year, month, count in periods:
                for _ in range(count):
                    sequence += 1
                    captured = f"{year:04d}-{month:02d}-15T12:00:00+00:00"
                    assets.append(
                        (
                            CatalogAsset(
                                str(photos / f"fixture-{sequence:04d}.jpg"),
                                root.root_id,
                                captured,
                                "exif",
                                captured,
                                "Fixture camera",
                                64,
                                48,
                                1,
                                ".jpg",
                                mtime_ns=sequence,
                            ),
                            sequence,
                        )
                    )
            catalog._upsert_assets(assets)  # Fixture rows only; Timeline must not read source files.

            page = catalog.query_assets(CatalogQuery(root_ids=(root.root_id,), limit=240))
            timeline = catalog.query_timeline(CatalogQuery(root_ids=(root.root_id,), offset=480, limit=240))

            self.assertEqual(240, len(page.items))
            self.assertEqual(1_001, timeline.total_count)
            self.assertEqual(1_001, len(timeline.image_paths))
            self.assertEqual([2026, 2025], [year.year for year in timeline.years])
            self.assertEqual([3, 2], [month.month for month in timeline.years[0].months])
            self.assertEqual(400, len(timeline.years[0].months[0].image_paths))
            self.assertEqual([15], [day.day for day in timeline.years[0].months[0].days])
            self.assertEqual(400, len(timeline.years[0].months[0].days[0].image_paths))

            checks = 0

            def _cancel() -> bool:
                nonlocal checks
                checks += 1
                return checks >= 4

            with self.assertRaises(Cancelled):
                catalog.query_timeline(CatalogQuery(root_ids=(root.root_id,)), cancel_check=_cancel)

    def test_timeline_can_derive_capture_time_from_unambiguous_filename_on_forced_refresh(self) -> None:
        with TemporaryDirectory() as raw:
            directory = Path(raw)
            photos = directory / "photos"
            photos.mkdir()
            photo = photos / "IMG_20240506_123456.jpg"
            self._photo(photo, (30, 80, 170))
            # A known modification time makes the fallback assertion stable on
            # every host and demonstrates that a catalog policy never edits
            # the image itself.
            modified_ns = 1_577_836_800_000_000_000  # 2020-01-01T00:00:00Z
            os.utime(photo, ns=(modified_ns, modified_ns))
            catalog = self._catalog(directory)
            root = catalog.register_root(photos)

            first = catalog.scan_root(root.root_id)
            self.assertEqual(1, first["updated"])
            asset = catalog.query_assets(CatalogQuery(root_ids=(root.root_id,))).items[0]
            self.assertEqual("filename", asset.capture_source)
            self.assertEqual("2024-05-06T12:34:56+00:00", asset.captured_at)

            catalog.set_filename_date_policy("metadata_only")
            unchanged = catalog.scan_root(root.root_id)
            self.assertEqual(1, unchanged["unchanged"])
            forced = catalog.scan_root(root.root_id, force=True)
            self.assertEqual(1, forced["updated"])
            refreshed = catalog.query_assets(CatalogQuery(root_ids=(root.root_id,))).items[0]
            self.assertEqual("modified", refreshed.capture_source)
            self.assertEqual("2020-01-01T00:00:00+00:00", refreshed.captured_at)

    def test_filename_date_parser_rejects_ambiguous_or_invalid_values(self) -> None:
        parser = LibraryCatalogService.filename_capture_datetime
        self.assertEqual("2024-05-06T12:34:56+00:00", parser("IMG-2024-05-06-123456.jpg"))
        self.assertEqual("2024-05-06T00:00:00+00:00", parser("scan_20240506.jpg"))
        self.assertEqual("", parser("IMG-05-06-2024.jpg"))
        self.assertEqual("", parser("IMG-2024-02-30.jpg"))

    def test_filename_date_parser_rejects_years_outside_the_timeline_range(self) -> None:
        parser = LibraryCatalogService.filename_capture_datetime
        future_year = datetime.now(timezone.utc).year + 1

        self.assertEqual("", parser("IMG_19900506.jpg"))
        self.assertEqual("", parser("IMG_87990506.jpg", ("%Y%m%d",)))
        self.assertEqual("", parser(f"IMG_{future_year:04d}0506.jpg", ("%Y%m%d",)))

    def test_unparseable_filename_uses_valid_exif_before_a_filename_fallback(self) -> None:
        with TemporaryDirectory() as raw:
            directory = Path(raw)
            photos = directory / "photos"
            photos.mkdir()
            photo = photos / "87990101.jpg"
            exif = Image.Exif()
            exif[34665] = {36867: "2024:05:06 12:34:56"}
            Image.new("RGB", (80, 60), (30, 80, 170)).save(photo, "JPEG", exif=exif)
            catalog = self._catalog(directory)
            root = catalog.register_root(photos)
            catalog.set_filename_date_policy("prefer_filename")

            catalog.scan_root(root.root_id, force=True)
            asset = catalog.query_assets(CatalogQuery(root_ids=(root.root_id,))).items[0]

            self.assertEqual("exif", asset.capture_source)
            self.assertEqual("2024-05-06T12:34:56+00:00", asset.captured_at)

    def test_filename_only_keeps_unparseable_files_in_an_explicit_timeline_bucket(self) -> None:
        with TemporaryDirectory() as raw:
            directory = Path(raw)
            photos = directory / "photos"
            photos.mkdir()
            photo = photos / "0e5f4347-4b55-4e32-95e0-8a928c3d4f1a.jpg"
            self._photo(photo, (30, 80, 170))
            modified_ns = 1_577_836_800_000_000_000
            os.utime(photo, ns=(modified_ns, modified_ns))
            catalog = self._catalog(directory)
            root = catalog.register_root(photos)
            catalog.set_filename_date_policy("filename_only")

            catalog.scan_root(root.root_id, force=True)
            asset = catalog.query_assets(CatalogQuery(root_ids=(root.root_id,))).items[0]
            timeline = catalog.query_timeline(CatalogQuery(root_ids=(root.root_id,)))

            self.assertEqual("unparsed", asset.capture_source)
            self.assertEqual("", asset.captured_at)
            self.assertEqual([0], [year.year for year in timeline.years])
            self.assertEqual([0], [month.month for month in timeline.years[0].months])

    def test_filename_only_uses_datetime_original_when_an_unparseable_name_has_camera_metadata(self) -> None:
        with TemporaryDirectory() as raw:
            directory = Path(raw)
            photos = directory / "photos"
            photos.mkdir()
            photo = photos / "0e5f4347-4b55-4e32-95e0-8a928c3d4f1a.jpg"
            exif = Image.Exif()
            exif[36867] = "2024:05:06 12:34:56"
            Image.new("RGB", (80, 60), (30, 80, 170)).save(photo, "JPEG", exif=exif)
            catalog = self._catalog(directory)
            root = catalog.register_root(photos)
            catalog.set_filename_date_policy("filename_only")

            catalog.scan_root(root.root_id, force=True)
            asset = catalog.query_assets(CatalogQuery(root_ids=(root.root_id,))).items[0]
            timeline = catalog.query_timeline(CatalogQuery(root_ids=(root.root_id,)))

            self.assertEqual("exif_original", asset.capture_source)
            self.assertEqual("2024-05-06T12:34:56+00:00", asset.captured_at)
            self.assertEqual([2024], [year.year for year in timeline.years])

    def test_datetime_original_first_prefers_original_then_other_metadata_filename_and_modified(self) -> None:
        with TemporaryDirectory() as raw:
            catalog = self._catalog(Path(raw))
            catalog.set_filename_date_policy("datetime_original_then_metadata")
            filename = FilenameCapture("2025-06-07T08:09:10+00:00")

            captured_at, source, _sequence = catalog._resolve_capture_time(
                manual_captured_at="",
                exif_original_at="2023-04-05T06:07:08+00:00",
                exif_captured_at="2024-03-04T05:06:07+00:00",
                filename_capture=filename,
                modified_at="2022-01-02T03:04:05+00:00",
            )

            self.assertEqual("2023-04-05T06:07:08+00:00", captured_at)
            self.assertEqual("exif_original", source)

    def test_timeline_places_stale_out_of_range_catalog_rows_in_unparsed_last(self) -> None:
        with TemporaryDirectory() as raw:
            directory = Path(raw)
            photos = directory / "photos"
            photos.mkdir()
            catalog = self._catalog(directory)
            root = catalog.register_root(photos)
            valid_path = str((photos / "valid.jpg").resolve())
            stale_path = str((photos / "stale.jpg").resolve())
            catalog._upsert_assets(
                (
                    (
                        CatalogAsset(
                            valid_path, root.root_id, "2024-05-06T12:34:56+00:00", "exif",
                            "2024-05-06T12:34:56+00:00", "", 80, 60, 1, ".jpg",
                        ),
                        1,
                    ),
                    (
                        CatalogAsset(
                            stale_path, root.root_id, "8799-01-01T00:00:00+00:00", "exif",
                            "2024-05-06T12:34:56+00:00", "", 80, 60, 1, ".jpg",
                        ),
                        2,
                    ),
                )
            )

            timeline = catalog.query_timeline(CatalogQuery(root_ids=(root.root_id,)))

            self.assertEqual([2024, 0], [year.year for year in timeline.years])
            self.assertEqual((valid_path, stale_path), timeline.image_paths)

    def test_manual_capture_time_sidecar_overrides_filename_only_after_incremental_refresh(self) -> None:
        with TemporaryDirectory() as raw:
            directory = Path(raw)
            photos = directory / "photos"
            photos.mkdir()
            photo = photos / "278a7be1-0d39-43f0-b099-cca7fcd8fe4e.jpg"
            self._photo(photo, (30, 80, 170))
            catalog = self._catalog(directory)
            root = catalog.register_root(photos)
            catalog.set_filename_date_policy("filename_only")

            catalog.scan_root(root.root_id, force=True)
            before = catalog.query_assets(CatalogQuery(root_ids=(root.root_id,))).items[0]
            PhotoEditService().save_draft(
                photo,
                PhotoEditDraft(captured_at="2024-05-06 12:34:56"),
            )
            refreshed = catalog.scan_root(root.root_id)
            after = catalog.query_assets(CatalogQuery(root_ids=(root.root_id,))).items[0]
            timeline = catalog.query_timeline(CatalogQuery(root_ids=(root.root_id,)))

            self.assertEqual("unparsed", before.capture_source)
            self.assertEqual(1, refreshed["updated"])
            self.assertEqual("manual", after.capture_source)
            self.assertEqual("2024-05-06T12:34:56+00:00", after.captured_at)
            self.assertEqual([2024], [year.year for year in timeline.years])

    def test_timeline_uses_ordered_custom_filename_date_patterns_after_forced_refresh(self) -> None:
        with TemporaryDirectory() as raw:
            directory = Path(raw)
            photos = directory / "photos"
            photos.mkdir()
            photo = photos / "archive-2024x05x06--12h34m56.jpg"
            self._photo(photo, (30, 80, 170))
            modified_ns = 1_577_836_800_000_000_000  # 2020-01-01T00:00:00Z
            os.utime(photo, ns=(modified_ns, modified_ns))
            catalog = self._catalog(directory)
            root = catalog.register_root(photos)

            catalog.scan_root(root.root_id)
            before = catalog.query_assets(CatalogQuery(root_ids=(root.root_id,))).items[0]
            self.assertEqual("modified", before.capture_source)

            catalog.set_filename_date_patterns(("archive-%Yx%mx%d--%Hh%Mm%S", "%Y%m%d"))
            catalog.set_filename_date_policy("filename_only")
            refreshed = catalog.scan_root(root.root_id, force=True)
            self.assertEqual(1, refreshed["updated"])
            after = catalog.query_assets(CatalogQuery(root_ids=(root.root_id,))).items[0]
            self.assertEqual("filename", after.capture_source)
            self.assertEqual("2024-05-06T12:34:56+00:00", after.captured_at)

    def test_filename_date_patterns_reject_ambiguous_or_unsafe_formats(self) -> None:
        with TemporaryDirectory() as raw:
            catalog = self._catalog(Path(raw))
            with self.assertRaisesRegex(ValueError, "year-first"):
                catalog.set_filename_date_patterns(("%d-%m-%Y",))
            with self.assertRaisesRegex(ValueError, "supported directives"):
                catalog.set_filename_date_patterns(("%Y%m%d_%z",))

    def test_timeline_named_date_counter_uses_sequence_for_same_day_order(self) -> None:
        with TemporaryDirectory() as raw:
            directory = Path(raw)
            photos = directory / "photos"
            photos.mkdir()
            first = photos / "06052024001.jpg"
            second = photos / "06052024002.jpg"
            self._photo(first, (30, 80, 170))
            self._photo(second, (170, 80, 30))
            catalog = self._catalog(directory)
            root = catalog.register_root(photos)
            catalog.set_filename_date_patterns(("{date:DDMMYYYY}{sequence:3}",))
            catalog.set_filename_date_policy("filename_only")

            stats = catalog.scan_root(root.root_id, force=True)
            assets = catalog.query_assets(CatalogQuery(root_ids=(root.root_id,), limit=10)).items
            timeline = catalog.query_timeline(CatalogQuery(root_ids=(root.root_id,)))

            self.assertEqual(2, stats["filename_dates"])
            self.assertEqual(["06052024002.jpg", "06052024001.jpg"], [Path(asset.image_path).name for asset in assets])
            self.assertEqual([2, 1], [asset.capture_sequence for asset in assets])
            self.assertEqual({"2024-05-06T00:00:00+00:00"}, {asset.captured_at for asset in assets})
            self.assertEqual([str(second.resolve()), str(first.resolve())], list(timeline.image_paths))

    def test_timeline_named_epoch_rules_and_raw_id_fallback_are_opt_in(self) -> None:
        with TemporaryDirectory() as raw:
            directory = Path(raw)
            photos = directory / "photos"
            photos.mkdir()
            seconds = photos / "IMG_Epoch_1704067200.jpg"
            milliseconds = photos / "IMG_Epoch_1704067200123.jpg"
            counter = photos / "IMG_Epoch_001.jpg"
            raw_id = photos / "hash_a1704067200b.jpg"
            for index, path in enumerate((seconds, milliseconds, counter, raw_id), start=1):
                self._photo(path, (index * 30, 80, 170 - index * 20))
                modified_ns = 1_577_836_800_000_000_000
                os.utime(path, ns=(modified_ns, modified_ns))
            catalog = self._catalog(directory)
            root = catalog.register_root(photos)
            catalog.set_filename_date_patterns(("IMG_Epoch_{epoch:s}", "IMG_Epoch_{epoch:ms}"))
            catalog.set_filename_date_policy("filename_only")

            first_stats = catalog.scan_root(root.root_id, force=True)
            first = {Path(asset.image_path).name: asset for asset in catalog.query_assets(CatalogQuery(root_ids=(root.root_id,), limit=10)).items}
            self.assertEqual(2, first_stats["filename_epochs"])
            self.assertEqual("2024-01-01T00:00:00+00:00", first[seconds.name].captured_at)
            self.assertEqual("2024-01-01T00:00:00.123+00:00", first[milliseconds.name].captured_at)
            self.assertEqual("unparsed", first[counter.name].capture_source)
            self.assertEqual("unparsed", first[raw_id.name].capture_source)

            catalog.set_filename_epoch_heuristic(True)
            second_stats = catalog.scan_root(root.root_id, force=True)
            second = {Path(asset.image_path).name: asset for asset in catalog.query_assets(CatalogQuery(root_ids=(root.root_id,), limit=10)).items}
            self.assertEqual(1, second_stats["raw_epochs"])
            self.assertEqual("filename", second[raw_id.name].capture_source)
            self.assertEqual("2024-01-01T00:00:00+00:00", second[raw_id.name].captured_at)
            self.assertEqual(
                "filename_epoch_heuristic",
                LibraryCatalogService.filename_capture(raw_id.name, epoch_heuristic=True).strategy,
            )
            self.assertEqual(
                "2024-01-01T00:00:00.999+00:00",
                LibraryCatalogService.filename_capture_datetime(
                    "IMG_Epoch_1704067200999.jpg", ("IMG_Epoch_{epoch:ms}",)
                ),
            )
            self.assertEqual("", LibraryCatalogService.filename_capture_datetime("IMG_Epoch_001.jpg", ("IMG_Epoch_{epoch:s}",)))
            self.assertEqual(
                "",
                LibraryCatalogService.filename_capture_datetime("hash_1704067200_1704153600.jpg", epoch_heuristic=True),
            )
            self.assertEqual(
                "",
                LibraryCatalogService.filename_capture_datetime("hash_9999999999.jpg", epoch_heuristic=True),
            )

    def test_catalog_migrates_capture_sequence_for_existing_database(self) -> None:
        with TemporaryDirectory() as raw:
            database = Path(raw) / "legacy.sqlite3"
            with sqlite3.connect(database) as connection:
                connection.execute(
                    """
                    CREATE TABLE catalog_assets (
                        image_path TEXT PRIMARY KEY, root_id TEXT NOT NULL, parent_path TEXT NOT NULL,
                        file_name TEXT NOT NULL, file_ext TEXT NOT NULL, mtime_ns INTEGER NOT NULL,
                        file_size INTEGER NOT NULL, captured_at TEXT NOT NULL DEFAULT '',
                        capture_source TEXT NOT NULL DEFAULT 'modified', modified_at TEXT NOT NULL,
                        camera TEXT NOT NULL DEFAULT '', width INTEGER NOT NULL DEFAULT 0,
                        height INTEGER NOT NULL DEFAULT 0, exif_json TEXT NOT NULL DEFAULT '{}',
                        xmp_text TEXT NOT NULL DEFAULT '', metadata_mtime_ns INTEGER NOT NULL DEFAULT 0,
                        metadata_size INTEGER NOT NULL DEFAULT 0
                    )
                    """
                )

            catalog = LibraryCatalogService(db_path=database)
            with catalog._connect() as connection:
                columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(catalog_assets)").fetchall()}

            self.assertIn("capture_sequence", columns)

    def test_context_uses_catalog_capture_order_and_reuses_cache(self) -> None:
        with TemporaryDirectory() as raw:
            directory = Path(raw)
            photos = directory / "photos"
            photos.mkdir()
            early = photos / "early.jpg"
            late = photos / "late.jpg"
            self._photo(early, (20, 60, 180))
            self._photo(late, (200, 80, 20))
            catalog = self._catalog(directory)
            root = catalog.register_root(photos)
            catalog.scan_root(root.root_id)
            with catalog._connect() as connection:  # Test fixture: make ordering explicit and deterministic.
                connection.execute("UPDATE catalog_assets SET captured_at=? WHERE image_path=?", ("2020-01-01T00:00:00+00:00", str(early.resolve())))
                connection.execute("UPDATE catalog_assets SET captured_at=? WHERE image_path=?", ("2023-01-01T00:00:00+00:00", str(late.resolve())))

            service = ClusterContextService(catalog)
            provider = _StaticVisionProvider()
            service.providers["ollama"] = provider
            settings = VisionLanguageSettings(provider="ollama", ollama_model="fixture-vision")
            first = service.describe_cluster(cluster_key="fixture:0", members=[str(late), str(early)], settings=settings)
            second = service.describe_cluster(cluster_key="fixture:0", members=[str(late), str(early)], settings=settings)

            self.assertEqual(str(early.resolve()), first.representative_path)
            self.assertEqual(first.context_id, second.context_id)
            self.assertEqual(1, provider.calls)
            self.assertEqual(1, catalog.search_cluster_contexts("dog beach").total_count)

            # A representative can be edited between library scans. Its
            # context must not be reused from the prior source revision.
            Image.new("RGB", (96, 60), (20, 60, 180)).save(early, "JPEG")
            revised = service.describe_cluster(
                cluster_key="fixture:0",
                members=[str(late), str(early)],
                settings=settings,
            )
            self.assertNotEqual(first.context_id, revised.context_id)
            self.assertEqual(2, provider.calls)

    def test_context_cancellation_propagates_as_job_cancellation(self) -> None:
        with TemporaryDirectory() as raw:
            directory = Path(raw)
            photo = directory / "photo.jpg"
            self._photo(photo, (30, 80, 170))
            catalog = self._catalog(directory)
            service = ClusterContextService(catalog)

            with self.assertRaises(Cancelled):
                service.describe_cluster(
                    cluster_key="fixture:cancelled",
                    members=[str(photo)],
                    settings=VisionLanguageSettings(provider="ollama", ollama_model="fixture-vision"),
                    cancel_check=lambda: True,
                )

    def test_perceptual_hash_index_chunks_large_path_sets(self) -> None:
        with TemporaryDirectory() as raw:
            directory = Path(raw)
            paths = []
            for index in range(1_001):
                path = directory / f"fixture-{index}.jpg"
                path.touch()
                paths.append(str(path))
            service = PerceptualHashIndexService(db_path=directory / "hashes.sqlite3")

            with patch.object(service, "build_index", return_value=0) as build_index:
                self.assertEqual(0, service.ensure_index(paths))

            self.assertEqual(1_001, len(build_index.call_args.args[0]))

    def test_clearing_library_cache_preserves_roots_and_review_choices(self) -> None:
        with TemporaryDirectory() as raw:
            directory = Path(raw)
            photo = directory / "photo.jpg"
            self._photo(photo, (30, 80, 170))
            catalog = self._catalog(directory)
            root = catalog.register_root(directory)
            catalog.scan_root(root.root_id)
            catalog.set_duplicate_feedback(
                decision_key="fixture", kind="exact", left_path=str(photo), right_path=str(photo), state="not-duplicate"
            )

            self.assertEqual(1, catalog.clear_library_cache())
            self.assertEqual(1, len(catalog.list_roots()))
            self.assertEqual(0, catalog.query_assets(CatalogQuery()).total_count)
            self.assertEqual({"fixture": "not-duplicate"}, catalog.duplicate_feedback(["fixture"]))

    def test_not_duplicate_feedback_is_reconsidered_after_a_source_revision(self) -> None:
        with TemporaryDirectory() as raw:
            directory = Path(raw)
            catalog = self._catalog(directory)
            left = CatalogAsset("/fixture/a.jpg", "root", "2024-01-01T00:00:00+00:00", "exif", "", "", 100, 100, 10, ".jpg", mtime_ns=10)
            right = CatalogAsset("/fixture/b.jpg", "root", "2024-01-01T00:00:01+00:00", "exif", "", "", 90, 90, 10, ".jpg", mtime_ns=11)
            service = DuplicateReviewService(catalog=catalog)
            groups = service._groups_for_pairs("near", [(left.image_path, right.image_path, 2, 0.98)], {left.image_path: left, right.image_path: right})
            self.assertEqual(1, len(groups))
            service.mark_not_duplicate(groups[0])
            self.assertEqual({groups[0].review_keys[0]: "not-duplicate"}, catalog.duplicate_feedback(groups[0].review_keys))

            revised = CatalogAsset("/fixture/b.jpg", "root", "2024-01-01T00:00:01+00:00", "exif", "", "", 90, 90, 12, ".jpg", mtime_ns=12)
            revised_group = service._groups_for_pairs("near", [(left.image_path, revised.image_path, 2, 0.98)], {left.image_path: left, revised.image_path: revised})[0]
            self.assertNotEqual(groups[0].review_keys, revised_group.review_keys)
            self.assertEqual({}, catalog.duplicate_feedback(revised_group.review_keys))

    def test_exact_duplicate_review_never_loads_perceptual_hashes_or_vectors(self) -> None:
        with TemporaryDirectory() as raw:
            directory = Path(raw)
            first = directory / "first.jpg"
            second = directory / "second.jpg"
            first.write_bytes(b"same bytes")
            second.write_bytes(b"same bytes")
            catalog = self._catalog(directory)
            service = DuplicateReviewService(catalog=catalog)
            assets = [
                CatalogAsset(str(first), "root", "", "unparsed", "", "", 80, 60, first.stat().st_size, ".jpg"),
                CatalogAsset(str(second), "root", "", "unparsed", "", "", 80, 60, second.stat().st_size, ".jpg"),
            ]
            with (
                patch.object(service, "_all_assets", return_value=assets),
                patch.object(service.phash_service, "load_hash_values") as load_hashes,
                patch.object(service, "_vectors_for_paths") as load_vectors,
            ):
                groups = service.build_groups(kinds=("exact",))

            self.assertEqual(["exact"], [group.kind for group in groups])
            load_hashes.assert_not_called()
            load_vectors.assert_not_called()

    def test_similar_duplicate_review_requires_two_hash_backends_and_records_their_agreement(self) -> None:
        with TemporaryDirectory() as raw:
            directory = Path(raw)
            left = CatalogAsset(str(directory / "left.jpg"), "root", "", "unparsed", "", "", 80, 60, 10, ".jpg")
            right = CatalogAsset(str(directory / "right.jpg"), "root", "", "unparsed", "", "", 80, 60, 10, ".jpg")
            catalog = self._catalog(directory)
            service = DuplicateReviewService(catalog=catalog)
            with (
                patch.object(service, "_all_assets", return_value=[left, right]),
                patch.object(service.phash_service, "load_hash_values", return_value={left.image_path: 0, right.image_path: 0}) as load_hashes,
                patch.object(service, "_vectors_for_paths", return_value={}),
            ):
                groups = service.build_groups(kinds=("similar",))

            self.assertEqual(["near"], [group.kind for group in groups])
            self.assertEqual({"phash", "dhash", "whash"}, {call.kwargs["hash_backend"] for call in load_hashes.call_args_list})
            candidate = next(member for member in groups[0].members if member.image_path == right.image_path)
            self.assertEqual(("dhash", "phash", "whash"), candidate.hash_backends)

    def test_near_duplicate_radius_partitioning_does_not_miss_spread_bit_changes(self) -> None:
        """A radius-four pair can differ once in every legacy 16-bit band."""

        with TemporaryDirectory() as raw:
            service = DuplicateReviewService(catalog=self._catalog(Path(raw)))
            left = "/fixture/left.jpg"
            right = "/fixture/right.jpg"
            # Bits 0, 16, 32, and 48 make the pair invisible to four equal
            # 16-bit lookup bands, while its actual Hamming distance is four.
            pairs = service._near_pairs(
                {left: 0, right: sum(1 << bit for bit in (0, 16, 32, 48))},
                {},
                max_hash_distance=4,
                min_visual_score=0.0,
                exact_pairs=(),
            )

        self.assertEqual([(left, right, 4, None)], pairs)

    def test_remote_context_requires_explicit_consent_before_provider_execution(self) -> None:
        with TemporaryDirectory() as raw:
            directory = Path(raw)
            photo = directory / "photo.jpg"
            self._photo(photo, (30, 80, 170))
            catalog = self._catalog(directory)
            root = catalog.register_root(directory)
            catalog.scan_root(root.root_id)
            service = ClusterContextService(catalog)
            provider = _StaticRemoteVisionProvider()
            service.providers["openai-compatible"] = provider
            settings = VisionLanguageSettings(
                provider="openai-compatible",
                openai_url="https://explicit.example.invalid/v1/chat/completions",
                openai_model="fixture-vision",
                remote_consent=False,
            )

            with self.assertRaises(ClusterContextError):
                service.describe_cluster(cluster_key="fixture:remote", members=[str(photo)], settings=settings)
            self.assertEqual(0, provider.calls)

            record = service.describe_cluster(
                cluster_key="fixture:remote",
                members=[str(photo)],
                settings=VisionLanguageSettings(
                    provider=settings.provider,
                    openai_url=settings.openai_url,
                    openai_model=settings.openai_model,
                    remote_consent=True,
                ),
            )
            self.assertEqual("Beach walk", record.title)
            self.assertEqual(1, provider.calls)

    def test_manual_context_reads_xmp_for_an_unregistered_cluster_photo(self) -> None:
        with TemporaryDirectory() as raw:
            directory = Path(raw)
            photo = directory / "unregistered.jpg"
            self._photo(photo, (30, 80, 170))
            photo.with_suffix(".xmp").write_text("<dc:subject>private mountain hike</dc:subject>", encoding="utf-8")
            service = ClusterContextService(self._catalog(directory))
            provider = _PromptCapturingVisionProvider()
            service.providers["ollama"] = provider

            service.describe_cluster(
                cluster_key="manual:0",
                members=[str(photo)],
                settings=VisionLanguageSettings(provider="ollama", ollama_model="fixture-vision"),
            )
            self.assertIn("private mountain hike", provider.prompt)

    def test_split_group_records_all_pairwise_exclusions(self) -> None:
        with TemporaryDirectory() as raw:
            catalog = self._catalog(Path(raw))
            service = PeopleCleanupService(object(), catalog=catalog)  # Face service is unused by the split-only path.

            class Member:
                def __init__(self, path: str, index: int) -> None:
                    self.image_path = path
                    self.face_index = index

            members = (Member("/fixture/a.jpg", 0), Member("/fixture/a.jpg", 1), Member("/fixture/b.jpg", 0))
            service.split_group(members)
            refs = [(member.image_path, member.face_index) for member in members]
            self.assertEqual(3, len(catalog.face_cleanup_exclusions(refs)))


if __name__ == "__main__":
    unittest.main()
