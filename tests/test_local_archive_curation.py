from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PIL import Image

from app.services.cluster_context import ClusterContextError, ClusterContextService, VisionLanguageResponse, VisionLanguageSettings
from app.services.duplicate_review import DuplicateReviewService
from app.services.library_catalog import CatalogAsset, CatalogQuery, LibraryCatalogService
from app.services.people_cleanup import PeopleCleanupService
from app.services.perceptual_hash import PerceptualHashIndexService
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

            catalog.set_root_enabled(root.root_id, False)
            self.assertEqual(0, catalog.query_assets(CatalogQuery()).total_count)

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

            checks = 0

            def _cancel() -> bool:
                nonlocal checks
                checks += 1
                return checks >= 4

            with self.assertRaises(Cancelled):
                catalog.query_timeline(CatalogQuery(root_ids=(root.root_id,)), cancel_check=_cancel)

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
