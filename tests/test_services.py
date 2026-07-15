import gc
import json
import os
import sqlite3
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
from PyQt6.QtGui import QColor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.services.cache_maintenance import CacheMaintenanceService
from app.services.face_model_installer import FaceModelInstaller
from app.services.cluster_meanings import ClusterMeaningCacheService, ClusterMeaningService
from app.services.clustering_pipeline import ClusteringPipelineService, ClusteringRequest
from app.services.discovery import DiscoveryResult, ImageDiscoveryService
from app.services import face_search as face_search_module
from app.services.embedding_index import EmbeddingIndexService
from app.services.face_search import (
    BUILTIN_HUMAN_DETECTOR_ID,
    BUILTIN_HUMAN_EMBEDDER_ID,
    EditableFaceInput,
    FaceDetectionService,
    FaceIndexRecord,
    IndexedFaceRecord,
    FaceIndexService,
    FaceLabelAssignment,
    FaceLabelRequest,
    face_cluster_backend_choices,
    face_detector_choices,
    face_embedder_choices,
    inspect_animal_face_bundle,
    resolve_face_detector_bundle,
    resolve_face_embedder_bundle,
)
from app.services.gallery_actions import GalleryActionService
from app.services.image_tags import ImageTagService
from app.services.model_assets import ModelAssetService, sha256_file
from app.services.onnx_models import OnnxModelService
from app.services.perceptual_hash import PerceptualHashIndexService
from app.services.performance_dashboard import PerformanceDashboardService
from app.services.photo_metadata import MetadataSidecarService
from app.services.result_cache import ResultCacheService
from app.services.saved_searches import SavedSearchService
from app.services.similarity_graph import SimilarityGraphService
from app.services.similarity_search import GlobalImageIndexService, SearchResult, SimilaritySearchRequest, SimilaritySearchService
from app.services.thumbnails import ThumbnailService
from infra.cache import CacheService
from infra.cancel import Cancelled
from infra.performance import select_performance_profile
from infra.runtime import ExecutionPolicy, RuntimeCapabilities, RuntimeCapabilityService
from ml.clustering import ClusteringService
from ml.embeddings import EmbeddingService


class FakeEmbeddingService:
    def __init__(self):
        self.use_cache_lookup_values = []

    def embed_paths(self, image_paths, model_name, progress_callback=None, use_onnx=False, use_cache_lookup=True, path_fingerprints=None):
        _ = path_fingerprints
        self.use_cache_lookup_values.append(bool(use_cache_lookup))
        vectors = []
        for index, image_path in enumerate(image_paths):
            vectors.append((image_path, np.asarray([1.0, 0.0], dtype=np.float32) if index < 2 else np.asarray([0.0, 1.0], dtype=np.float32)))
        return vectors, {
            "cache_hits": 0,
            "cache_misses": len(vectors),
            "batch_size": 2,
            "model_device": "cpu",
            "amp_enabled": False,
            "embedding_backend": "torch",
            "cache_lookup_time_s": 0.0,
            "preprocess_time_s": 0.0,
            "inference_time_s": 0.0,
            "embedding_cache_lookup": "enabled" if use_cache_lookup else "disabled",
        }


class FakeClusteringService:
    def __init__(self):
        self.prepare_calls = 0
        self._delegate = ClusteringService()

    def prepare_matrix(self, embeddings, pca_dim=None, similarity_mode="semantic"):
        self.prepare_calls += 1
        return np.asarray(embeddings, dtype=np.float32)

    def cluster_prepared(self, embeddings, num_clusters, backend="cosine-kmeans", outlier_policy="assign", performance_profile=None):
        _ = performance_profile
        if backend == "graph":
            return {10: [0, 1], 11: [2, 3]}, {"backend": backend, "cluster_quality_score": 0.91, "outlier_count": 0}
        return {0: [0, 1], 1: [2, 3]}, {"backend": backend, "cluster_quality_score": 0.83, "outlier_count": 0}

    def build_cluster_explanations(self, metric_matrix, clusters, *, cluster_quality_score=None, prepared_info=None):
        return self._delegate.build_cluster_explanations(
            metric_matrix,
            clusters,
            cluster_quality_score=cluster_quality_score,
            prepared_info=prepared_info,
        )


class FakeDiscoveryService:
    def __init__(self, paths):
        self.paths = paths

    def discover(self, directory, recursive=True):
        return list(self.paths)

    def discover_result(self, directory, recursive=True, progress_callback=None, cancel_check=None):
        _ = (directory, recursive, progress_callback)
        if callable(cancel_check) and cancel_check():
            raise Cancelled()
        snapshot = EmbeddingIndexService.build_snapshot_key(list(self.paths))
        fingerprints = ImageDiscoveryService.collect_fingerprints(list(self.paths))
        return DiscoveryResult(paths=tuple(self.paths), snapshot_key=snapshot, image_count=len(self.paths), fingerprints=fingerprints)

    @staticmethod
    def collect_fingerprints(image_paths):
        return ImageDiscoveryService.collect_fingerprints(list(image_paths))


class FakeFace:
    def __init__(self, image_path):
        self.image_path = image_path
        self.bbox = (0, 0, 36, 36)
        self.confidence = 0.99
        self.crop = Image.new("RGB", (36, 36), (255, 255, 255))


class FakeFaceDetectionService:
    def detect_query_face(self, image_path, query_bbox=None):
        return FakeFace(image_path)

    def detect_faces(self, image_path):
        return [FakeFace(image_path)]


class FakeFaceEmbeddingService:
    def embed_faces(self, face_crops):
        return np.asarray([[1.0, 0.0, 0.0] for _ in face_crops], dtype=np.float32)


class FakeQueryEmbeddingService:
    def embed_paths(self, image_paths, model_name, progress_callback=None, use_onnx=False, use_cache_lookup=True, path_fingerprints=None):
        _ = (progress_callback, use_onnx, use_cache_lookup, path_fingerprints)
        return [(image_paths[0], np.asarray([1.0, 0.0], dtype=np.float32))], {}

    def embed_text(self, query_text, model_name, use_onnx=False):
        return np.asarray([1.0, 0.0], dtype=np.float32)


class FakeClusterMeaningEmbeddingService:
    def __init__(self):
        self.embed_paths_calls = 0
        self.embed_text_calls = 0
        self.embed_texts_calls = 0

    def embed_paths(self, image_paths, model_name, use_onnx=False, progress_callback=None, use_cache_lookup=True, path_fingerprints=None):
        _ = (use_onnx, use_cache_lookup, path_fingerprints)
        self.embed_paths_calls += 1
        vectors = []
        for image_path in image_paths:
            text = str(image_path).lower()
            if "beach" in text or "ocean" in text:
                vector = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
            elif "city" in text:
                vector = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
            else:
                vector = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
            vectors.append((image_path, vector))
        if progress_callback:
            progress_callback(len(vectors), max(1, len(vectors)), f"embedded {model_name}")
        return vectors, {"cache_hits": 0, "cache_misses": len(vectors)}

    def embed_text(self, query_text, model_name="clip", use_onnx=False):
        _ = (model_name, use_onnx)
        self.embed_text_calls += 1
        prompt = str(query_text).lower()
        if "beach" in prompt or "ocean" in prompt or "sea" in prompt:
            return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        if "city" in prompt or "building" in prompt or "architecture" in prompt:
            return np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float32)

    def embed_texts(self, query_texts, model_name="clip", use_onnx=False):
        self.embed_texts_calls += 1
        return [self.embed_text(query_text, model_name=model_name, use_onnx=use_onnx) for query_text in query_texts]


class FakeClusterMeaningService:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        clusters_by_key = kwargs["clusters_by_key"]
        return (
            {
                comparison_key: {
                    int(cluster_id): SimpleNamespace(
                        cluster_id=int(cluster_id),
                        as_context=lambda cluster_id=cluster_id: {
                            "cluster_id": int(cluster_id),
                            "labels": [],
                            "confidence": "Unavailable",
                            "status": "unavailable",
                        },
                    )
                    for cluster_id in clusters
                }
                for comparison_key, clusters in clusters_by_key.items()
            },
            {"cluster_meaning_status": "ok", "cluster_meaning_model": "clip"},
        )


class AlwaysHitResultCacheService:
    def __init__(self, paths):
        self.paths = list(paths)
        self.load_calls = []
        self.save_calls = []

    def build_result_key(self, **kwargs):
        return json.dumps(kwargs, sort_keys=True)

    def load(self, result_key):
        self.load_calls.append(result_key)
        return (
            {0: self.paths[:2], 1: self.paths[2:]},
            {
                "clustering_time_s": 0.456,
                "cluster_quality_score": 0.73,
                "similarity_space": "cached semantic/cosine space",
                "input_dimension": 4,
                "prepared_dimension": 2,
                "pca_components": 2,
            },
        )

    def save(self, result_key, clustered_images, metrics):
        self.save_calls.append((result_key, clustered_images, metrics))


class ServiceTests(unittest.TestCase):
    def test_discovery_is_recursive_and_case_insensitive(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "nested").mkdir()
            (root / "a.JPG").write_bytes(b"x")
            (root / "nested" / "b.png").write_bytes(b"y")
            paths = ImageDiscoveryService().discover(str(root), recursive=True)
            self.assertIn(str(root / "a.JPG"), paths)
            self.assertIn(str(root / "nested" / "b.png"), paths)

    def test_discovery_reports_scan_progress(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "nested").mkdir()
            (root / "a.jpg").write_bytes(b"x")
            (root / "nested" / "b.png").write_bytes(b"y")
            events: list[tuple[int, str]] = []

            result = ImageDiscoveryService().discover_result(
                str(root),
                recursive=True,
                progress_callback=lambda value, status: events.append((value, status)),
            )

            self.assertEqual(2, result.image_count)
            self.assertTrue(events)
            self.assertIn("Scanning folder", events[0][1])
            self.assertIn("Folder scan complete: 2 image(s) discovered.", events[-1][1])

    def test_discovery_honors_cancel_check(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "nested").mkdir()
            (root / "a.jpg").write_bytes(b"x")
            (root / "nested" / "b.png").write_bytes(b"y")
            cancel_requested = {"value": False}

            def progress(_value, _status):
                cancel_requested["value"] = True

            with self.assertRaises(Cancelled):
                ImageDiscoveryService().discover_result(
                    str(root),
                    recursive=True,
                    progress_callback=progress,
                    cancel_check=lambda: bool(cancel_requested["value"]),
                )

    def test_discovery_follows_symlinked_directories_with_visible_paths(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "outside-target"
            target.mkdir()
            (target / "linked.jpg").write_bytes(b"x")
            os.symlink(target, root / "linked")

            result = ImageDiscoveryService().discover_result(str(root), recursive=True)

            self.assertEqual(1, result.image_count)
            self.assertTrue(result.complete)
            self.assertEqual([str(root / "linked" / "linked.jpg")], list(result.paths))

    def test_discovery_avoids_symlink_cycles(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "nested"
            nested.mkdir()
            (nested / "photo.jpg").write_bytes(b"x")
            os.symlink(root, nested / "loop")

            result = ImageDiscoveryService().discover_result(str(root), recursive=True)

            self.assertEqual(1, result.image_count)
            self.assertTrue(result.complete)
            self.assertEqual([str(nested / "photo.jpg")], list(result.paths))

    def test_discovery_marks_incomplete_when_child_directory_is_unreadable(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            readable = root / "readable"
            blocked = root / "blocked"
            readable.mkdir()
            blocked.mkdir()
            (readable / "photo.jpg").write_bytes(b"x")
            (blocked / "secret.jpg").write_bytes(b"y")
            blocked.chmod(0)
            try:
                result = ImageDiscoveryService().discover_result(str(root), recursive=True)
            finally:
                blocked.chmod(0o755)

            self.assertEqual(1, result.image_count)
            self.assertFalse(result.complete)
            self.assertGreaterEqual(result.warning_count, 1)
            self.assertEqual([str(readable / "photo.jpg")], list(result.paths))

    def test_cache_roundtrip(self):
        with TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "image.jpg"
            image_path.write_bytes(b"123")
            cache = CacheService()
            key = cache.build_embedding_key(str(image_path), "model", "signature")
            vector = np.array([1.0, 2.0, 3.0], dtype=np.float32)
            cache.put_embedding(key, str(image_path), "model", vector)
            loaded = cache.get_embedding(key)
            self.assertTrue(np.allclose(vector, loaded))

    def test_cache_service_clear_memory_cache_empties_lru(self):
        cache = CacheService()
        cache._remember("alpha", np.asarray([1.0, 2.0], dtype=np.float32))
        self.assertTrue(cache._memory_cache)
        cache.clear_memory_cache()
        self.assertEqual({}, dict(cache._memory_cache))

    def test_saved_searches_persist_rename_and_delete_required_categories(self):
        with TemporaryDirectory() as tmp:
            store_path = Path(tmp) / "saved_searches.json"
            service = SavedSearchService(store_path)

            clustering = service.save_search(
                "Family tags",
                "clustering_filter",
                {"tags": ["family", "selfie"], "tag_match": "All"},
            )
            face_query = service.save_search(
                "Alice unknown",
                "face_query",
                {"query": 'person:"Alice" unknown:true', "include_hidden": False},
            )
            people = service.save_search("Alice and Bob", "people_together", {"names": ["Alice", "Bob"]})
            hidden_queue = service.save_search(
                "Hidden queue",
                "hidden_faces",
                {"query": "hidden:true", "include_hidden": True},
            )
            unknown_queue = service.save_search("Unknown queue", "unknown_people", {"mode": "unknown"})

            reloaded = SavedSearchService(store_path)
            by_id = {record.search_id: record for record in reloaded.list_searches()}
            self.assertEqual({"family", "selfie"}, set(by_id[clustering.search_id].payload["tags"]))
            self.assertEqual('person:"Alice" unknown:true', by_id[face_query.search_id].payload["query"])
            self.assertEqual(["Alice", "Bob"], by_id[people.search_id].payload["names"])
            self.assertTrue(by_id[hidden_queue.search_id].payload["include_hidden"])
            self.assertEqual("unknown", by_id[unknown_queue.search_id].payload["mode"])

            renamed = reloaded.rename_search(face_query.search_id, "Alice review")
            self.assertEqual("Alice review", renamed.name)
            self.assertTrue(reloaded.delete_search(people.search_id))
            final_records = {record.search_id: record for record in SavedSearchService(store_path).list_searches()}
            self.assertIn(face_query.search_id, final_records)
            self.assertEqual("Alice review", final_records[face_query.search_id].name)
            self.assertNotIn(people.search_id, final_records)

    def test_thumbnail_service_clear_memory_cache_empties_qimage_cache(self):
        service = ThumbnailService()
        service._qimage_cache[("a.jpg", 128)] = service._qimage_cache.get(("a.jpg", 128)) or None
        service.clear_memory_cache()
        self.assertEqual({}, dict(service._qimage_cache))

    def test_thumbnail_service_contact_sheet_returns_image_for_valid_paths(self):
        with TemporaryDirectory() as tmp:
            paths = []
            for index in range(4):
                image_path = Path(tmp) / f"thumb_{index}.png"
                Image.new("RGB", (40, 40), (index * 40, 80, 120)).save(image_path)
                paths.append(str(image_path))
            service = ThumbnailService()
            sheet = service.build_contact_sheet_qimage(paths, (180, 180))
            self.assertFalse(sheet.isNull())
            self.assertEqual(180, sheet.width())
            self.assertEqual(180, sheet.height())

    def test_thumbnail_service_contact_sheet_handles_empty_and_unreadable_paths(self):
        service = ThumbnailService()
        empty = service.build_contact_sheet_qimage([], 128)
        self.assertFalse(empty.isNull())
        with TemporaryDirectory() as tmp:
            valid_image = Path(tmp) / "valid.png"
            invalid_image = Path(tmp) / "invalid.png"
            Image.new("RGB", (40, 40), (50, 100, 150)).save(valid_image)
            invalid_image.write_text("not an image", encoding="utf-8")
            sheet = service.build_contact_sheet_qimage([str(invalid_image), str(valid_image)], (160, 160))
            self.assertFalse(sheet.isNull())
            self.assertEqual(160, sheet.width())
            self.assertEqual(160, sheet.height())

    def test_thumbnail_service_respects_exif_orientation(self):
        with TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "rotated.jpg"
            image = Image.new("RGB", (100, 60), (40, 80, 120))
            exif = image.getexif()
            exif[274] = 6
            image.save(image_path, exif=exif)

            service = ThumbnailService()
            thumb_path = service.ensure_thumbnail(str(image_path), 80)

            with Image.open(thumb_path) as thumb:
                self.assertEqual((80, 80), thumb.size)
                non_white = thumb.convert("RGB").point(lambda value: 255 if value < 250 else 0).getbbox()

            self.assertEqual((15, 0, 65, 80), non_white)

    def test_thumbnail_service_load_qimage_fallback_respects_exif_orientation(self):
        with TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "rotated-fallback.jpg"
            image = Image.new("RGB", (100, 60), (40, 80, 120))
            exif = image.getexif()
            exif[274] = 6
            image.save(image_path, exif=exif)

            service = ThumbnailService()
            thumb_path = service.thumbnail_path(str(image_path), 80)
            thumb_path.parent.mkdir(parents=True, exist_ok=True)
            thumb_path.write_bytes(b"not-an-image")

            thumb = service.load_qimage(str(image_path), 80)

            self.assertFalse(thumb.isNull())
            self.assertEqual(80, thumb.width())
            self.assertEqual(80, thumb.height())
            self.assertNotEqual(QColor("#FFFFFF"), thumb.pixelColor(40, 5))
            self.assertEqual(QColor("#FFFFFF"), thumb.pixelColor(5, 40))

    def test_thumbnail_service_load_qimage_cache_miss_avoids_disk_thumbnail_generation(self):
        with TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "cache-miss.jpg"
            Image.new("RGB", (160, 100), (40, 80, 120)).save(image_path)

            service = ThumbnailService()
            with patch.object(service, "ensure_thumbnail", side_effect=AssertionError("ensure_thumbnail should not run")):
                thumb = service.load_qimage(str(image_path), 80)

            self.assertFalse(thumb.isNull())
            self.assertEqual(80, thumb.width())
            self.assertEqual(80, thumb.height())

    def test_clustering_service_builds_cluster_explanations_for_cohesive_clusters(self):
        service = ClusteringService()
        matrix = np.asarray(
            [
                [1.0, 0.0],
                [0.98, 0.02],
                [0.0, 1.0],
                [0.05, 0.95],
            ],
            dtype=np.float32,
        )
        matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
        explanations = service.build_cluster_explanations(
            matrix,
            {0: [0, 1], 1: [2, 3]},
            cluster_quality_score=0.82,
        )

        self.assertEqual(2, explanations[0].cluster_size)
        self.assertFalse(explanations[0].is_outlier)
        self.assertGreater(explanations[0].cohesion_mean or 0.0, 0.9)
        self.assertEqual(1, explanations[0].nearest_cluster_id)
        self.assertIsNotNone(explanations[0].separation_margin)
        self.assertEqual(0.82, explanations[0].cluster_quality_score)

    def test_clustering_service_cluster_explanations_handle_single_cluster_and_outliers(self):
        service = ClusteringService()
        matrix = np.asarray(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )
        matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
        single_cluster = service.build_cluster_explanations(matrix[:2], {0: [0, 1]})
        self.assertIsNone(single_cluster[0].nearest_cluster_id)
        self.assertIsNone(single_cluster[0].separation_margin)

        with_outlier = service.build_cluster_explanations(matrix, {0: [0, 1], -1: [2]})
        self.assertTrue(with_outlier[-1].is_outlier)
        self.assertIsNone(with_outlier[-1].cohesion_mean)
        self.assertEqual(1, with_outlier[-1].cluster_size)

    def test_clustering_service_prepares_distinct_semantic_and_cosine_spaces(self):
        service = ClusteringService()
        matrix = np.asarray(
            [
                [1.0, 0.2, 0.1, 0.0, 0.5],
                [0.9, 0.1, 0.3, 0.2, 0.4],
                [0.1, 1.0, 0.2, 0.5, 0.0],
                [0.2, 0.8, 0.4, 0.3, 0.1],
                [0.0, 0.3, 1.0, 0.4, 0.7],
                [0.3, 0.4, 0.9, 0.2, 0.6],
            ],
            dtype=np.float32,
        )

        semantic, semantic_info = service.prepare_matrix_with_info(matrix, pca_dim=2, similarity_mode="semantic")
        cosine, cosine_info = service.prepare_matrix_with_info(matrix, pca_dim=2, similarity_mode="cosine")

        self.assertEqual((6, 2), semantic.shape)
        self.assertEqual((6, 5), cosine.shape)
        self.assertEqual(2, semantic_info.pca_components)
        self.assertIsNone(cosine_info.pca_components)
        self.assertIn("semantic projection", semantic_info.space_label)
        self.assertIn("cosine full vector", cosine_info.space_label)
        np.testing.assert_allclose(np.linalg.norm(semantic, axis=1), np.ones(6), atol=1e-5)
        np.testing.assert_allclose(np.linalg.norm(cosine, axis=1), np.ones(6), atol=1e-5)

    def test_clustering_service_semantic_projection_falls_back_for_tiny_inputs(self):
        service = ClusteringService()
        matrix = np.asarray(
            [
                [1.0, 0.0, 0.2, 0.1],
                [0.9, 0.1, 0.3, 0.0],
                [0.0, 1.0, 0.4, 0.2],
            ],
            dtype=np.float32,
        )

        semantic, semantic_info = service.prepare_matrix_with_info(matrix, pca_dim=2, similarity_mode="semantic")

        self.assertEqual((3, 4), semantic.shape)
        self.assertIsNone(semantic_info.pca_components)
        self.assertIn("fallback", semantic_info.space_label)
        np.testing.assert_allclose(np.linalg.norm(semantic, axis=1), np.ones(3), atol=1e-5)

    def test_cluster_meaning_service_ranks_model_labels_and_filename_terms(self):
        with TemporaryDirectory() as tmp:
            cache = ClusterMeaningCacheService(Path(tmp) / "meanings")
            service = ClusterMeaningService(cache)
            embeddings = FakeClusterMeaningEmbeddingService()
            clusters = {"dino::semantic::hdbscan": {3: ["beach_trip_001.jpg", "ocean_day_002.jpg"]}}

            meanings, metrics = service.generate(
                clusters_by_key=clusters,
                all_image_paths=["beach_trip_001.jpg", "ocean_day_002.jpg"],
                snapshot_key="snapshot",
                path_fingerprints={},
                embedding_service=embeddings,
                selected_models=["dino"],
                requested_model="auto",
            )

            meaning = meanings["dino::semantic::hdbscan"][3]
            self.assertEqual("ok", meaning.status)
            self.assertEqual("clip", meaning.explanation_model)
            self.assertIn(meaning.labels[0].label, {"beach", "ocean"})
            self.assertIn(("trip", 1), meaning.filename_terms)
            self.assertEqual("ok", metrics["cluster_meaning_status"])
            self.assertEqual(1, embeddings.embed_texts_calls)

    def test_cluster_meaning_service_uses_disk_cache_on_repeat(self):
        with TemporaryDirectory() as tmp:
            cache = ClusterMeaningCacheService(Path(tmp) / "meanings")
            service = ClusterMeaningService(cache)
            first_embeddings = FakeClusterMeaningEmbeddingService()
            clusters = {"clip::semantic::graph": {1: ["city_walk_001.jpg"]}}
            common = {
                "clusters_by_key": clusters,
                "all_image_paths": ["city_walk_001.jpg"],
                "snapshot_key": "snapshot",
                "path_fingerprints": {},
                "selected_models": ["clip"],
                "requested_model": "auto",
            }

            service.generate(embedding_service=first_embeddings, **common)
            second_embeddings = FakeClusterMeaningEmbeddingService()
            meanings, metrics = service.generate(embedding_service=second_embeddings, **common)

            self.assertEqual(0, second_embeddings.embed_paths_calls)
            self.assertEqual("cached", metrics["cluster_meaning_status"])
            self.assertEqual("city", meanings["clip::semantic::graph"][1].labels[0].label)

    def test_cluster_meaning_service_auto_prefers_openclip_when_selected(self):
        with TemporaryDirectory() as tmp:
            cache = ClusterMeaningCacheService(Path(tmp) / "meanings")
            service = ClusterMeaningService(cache)
            embeddings = FakeClusterMeaningEmbeddingService()
            clusters = {"openclip::semantic::graph": {2: ["city_walk_001.jpg"]}}

            meanings, metrics = service.generate(
                clusters_by_key=clusters,
                all_image_paths=["city_walk_001.jpg"],
                snapshot_key="snapshot",
                path_fingerprints={},
                embedding_service=embeddings,
                selected_models=["openclip"],
                requested_model="auto",
            )

            self.assertEqual("openclip", meanings["openclip::semantic::graph"][2].explanation_model)
            self.assertEqual("openclip", metrics["cluster_meaning_model"])

    def test_gallery_move_generates_unique_targets(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "source"
            source_dir.mkdir()
            file_a = source_dir / "image.jpg"
            file_b = source_dir / "copy.jpg"
            file_a.write_bytes(b"a")
            file_b.write_bytes(b"b")
            target_dir = root / "target"
            action_service = GalleryActionService()
            first = action_service.move_to_directory([str(file_a)], str(target_dir))
            second = action_service.move_to_directory([str(file_b)], str(target_dir))
            self.assertEqual(1, len(first.changed_paths))
            self.assertEqual(1, len(second.changed_paths))
            self.assertTrue(Path(first.changed_paths[0][1]).exists())
            self.assertTrue(Path(second.changed_paths[0][1]).exists())

            # Best-effort cleanup on Windows where filesystem locks can be flaky.
            try:
                import shutil
                import time
                for _ in range(5):
                    try:
                        shutil.rmtree(str(target_dir))
                        break
                    except OSError:
                        time.sleep(0.05)
            except Exception:
                pass

    def test_gallery_copy_generates_unique_targets(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "source"
            source_dir.mkdir()
            file_a = source_dir / "image.jpg"
            file_b = source_dir / "image_copy.jpg"
            file_a.write_bytes(b"a")
            file_b.write_bytes(b"b")
            target_dir = root / "target"
            action_service = GalleryActionService()
            first = action_service.copy_to_directory([str(file_a)], str(target_dir))
            second = action_service.copy_to_directory([str(file_b)], str(target_dir))
            self.assertEqual(1, len(first.changed_paths))
            self.assertEqual(1, len(second.changed_paths))
            self.assertTrue(Path(first.changed_paths[0][1]).exists())
            self.assertTrue(Path(second.changed_paths[0][1]).exists())

            # Best-effort cleanup on Windows where filesystem locks can be flaky.
            try:
                import shutil
                import time
                for _ in range(5):
                    try:
                        shutil.rmtree(str(target_dir))
                        break
                    except OSError:
                        time.sleep(0.05)
            except Exception:
                pass

    def test_gallery_file_operations_write_audit_and_cleanup_temp_files(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.jpg"
            source.write_bytes(b"image-bytes")
            target_dir = root / "target"
            audit_log = root / "logs" / "file_operations.jsonl"
            action_service = GalleryActionService(audit_log_path=audit_log, temp_dir=root / "cache" / "tmp" / "file_ops")

            result = action_service.copy_to_directory([str(source)], str(target_dir))

            self.assertEqual("copy", result.operation)
            self.assertFalse(result.failures)
            self.assertEqual(1, len(result.changed_paths))
            copied_path = Path(result.changed_paths[0][1])
            self.assertTrue(copied_path.exists())
            self.assertEqual(source.read_bytes(), copied_path.read_bytes())
            self.assertEqual([], list(target_dir.glob("*.ic_tmp*")))
            self.assertTrue(audit_log.exists())
            audit = json.loads(audit_log.read_text(encoding="utf-8").splitlines()[-1])
            self.assertEqual(result.operation_id, audit["operation_id"])
            self.assertEqual("copy", audit["operation"])
            self.assertEqual(1, audit["requested_count"])
            self.assertEqual(0, audit["failure_count"])

    def test_gallery_file_operation_audit_records_missing_sources(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit_log = root / "logs" / "file_operations.jsonl"
            action_service = GalleryActionService(audit_log_path=audit_log, temp_dir=root / "cache" / "tmp" / "file_ops")

            result = action_service.copy_to_directory([str(root / "missing.jpg")], str(root / "target"))

            self.assertEqual(1, len(result.failures))
            self.assertIn("source file does not exist", result.failures[0])
            audit = json.loads(audit_log.read_text(encoding="utf-8").splitlines()[-1])
            self.assertEqual(1, audit["failure_count"])
            self.assertIn("missing.jpg", audit["failures"][0])

    def test_exif_metadata_pair_writes_and_replaces_user_comment_key(self):
        with TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "photo.jpg"
            Image.new("RGB", (32, 32), (100, 120, 140)).save(image_path)
            action_service = GalleryActionService()
            action_service.write_exif_metadata_pair(str(image_path), "xyz", "test")
            action_service.write_exif_metadata_pair(str(image_path), "xyz", "updated")
            with Image.open(image_path) as image:
                comment = str(image.getexif().get(37510, ""))
            self.assertIn("xyz: updated", comment)
            self.assertNotIn("xyz: test", comment)

    def test_exif_metadata_pair_preserves_existing_plain_text_and_other_keys(self):
        service = GalleryActionService()
        merged = service.merge_comment_metadata("plain text\nalpha: one", "xyz", "test")
        self.assertIn("plain text", merged)
        self.assertIn("alpha: one", merged)
        self.assertIn("xyz: test", merged)

    def test_image_tags_add_remove_and_match_case_insensitively(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_a = root / "a.jpg"
            image_b = root / "b.jpg"
            for image_path in [image_a, image_b]:
                Image.new("RGB", (32, 32), (100, 120, 140)).save(image_path)
            service = ImageTagService(db_path=root / "image_tags.sqlite3")
            service.apply_tag_edit([str(image_a)], add_tags=["Selfie", "Family"], mirror_to_exif=False)
            service.apply_tag_edit([str(image_b)], add_tags=["selfie", "Group", "GROUP"], mirror_to_exif=False)
            service.apply_tag_edit([str(image_a)], remove_tags=["family"], mirror_to_exif=False)
            tags_by_path = service.load_tags_for_paths([str(image_a), str(image_b)])
            self.assertEqual(("Selfie",), tags_by_path[str(image_a)])
            self.assertEqual(("Group", "selfie"), tags_by_path[str(image_b)])
            any_matches = service.select_paths_by_tags([str(image_a), str(image_b)], ["SELFIE"], "Any")
            self.assertEqual([str(image_a), str(image_b)], any_matches)
            all_matches = service.select_paths_by_tags([str(image_a), str(image_b)], ["selfie", "group"], "All")
            self.assertEqual([str(image_b)], all_matches)

    def test_image_tags_exif_roundtrip_preserves_other_comment_keys(self):
        with TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "photo.jpg"
            Image.new("RGB", (32, 32), (100, 120, 140)).save(image_path)
            action_service = GalleryActionService()
            action_service.write_exif_metadata_pair(str(image_path), "xyz", "one")
            service = ImageTagService(db_path=Path(tmp) / "tags.sqlite3")
            result = service.apply_tag_edit([str(image_path)], add_tags=["family", "selfie"], mirror_to_exif=True)
            self.assertFalse(result.failures)
            payload = action_service.read_exif_metadata_value(str(image_path), "ic_tags")
            self.assertEqual('["family", "selfie"]', payload)
            comment = action_service.read_exif_metadata_value(str(image_path), "xyz")
            self.assertEqual("one", comment)
            self.assertEqual(("family", "selfie"), service.load_tags_for_paths([str(image_path)])[str(image_path)])

    def test_image_tags_lazy_import_uses_exif_then_prefers_db(self):
        with TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "photo.jpg"
            Image.new("RGB", (32, 32), (100, 120, 140)).save(image_path)
            action_service = GalleryActionService()
            action_service.write_exif_metadata_pair(str(image_path), "ic_tags", '["travel", "portrait"]')
            service = ImageTagService(db_path=Path(tmp) / "tags.sqlite3")
            self.assertEqual(("portrait", "travel"), service.load_tags_for_paths([str(image_path)])[str(image_path)])
            service.apply_tag_edit(
                [str(image_path)],
                add_tags=["db_only"],
                remove_tags=["travel"],
                mirror_to_exif=False,
            )
            self.assertEqual(("db_only", "portrait"), service.load_tags_for_paths([str(image_path)])[str(image_path)])

    def test_image_tags_prune_stale_records_when_file_changes(self):
        with TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "photo.jpg"
            Image.new("RGB", (32, 32), (100, 120, 140)).save(image_path)
            service = ImageTagService(db_path=Path(tmp) / "tags.sqlite3")
            service.apply_tag_edit([str(image_path)], add_tags=["family"], mirror_to_exif=False)
            Image.new("RGB", (48, 48), (10, 20, 30)).save(image_path)
            self.assertEqual((), service.load_tags_for_paths([str(image_path)])[str(image_path)])

    def test_image_tags_skip_unreadable_images_during_exif_import(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            valid_image = root / "valid.jpg"
            broken_image = root / "broken.jpg"
            Image.new("RGB", (32, 32), (100, 120, 140)).save(valid_image)
            broken_image.write_text("not an image", encoding="utf-8")
            action_service = GalleryActionService()
            action_service.write_exif_metadata_pair(str(valid_image), "ic_tags", '["family"]')
            service = ImageTagService(db_path=root / "tags.sqlite3", action_service=action_service)

            tags_by_path = service.load_tags_for_paths([str(valid_image), str(broken_image)])

            self.assertEqual(("family",), tags_by_path[str(valid_image)])
            self.assertEqual((), tags_by_path[str(broken_image)])
            self.assertEqual([str(valid_image)], service.select_paths_by_tags([str(valid_image), str(broken_image)], ["family"], "Any"))

    def test_image_tags_move_and_copy_helpers_preserve_tags(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_source = root / "source.jpg"
            image_copy = root / "copy.jpg"
            image_moved = root / "moved.jpg"
            Image.new("RGB", (32, 32), (100, 120, 140)).save(image_source)
            service = ImageTagService(db_path=root / "tags.sqlite3")
            service.apply_tag_edit([str(image_source)], add_tags=["family", "priority"], mirror_to_exif=False)
            shutil.copy2(str(image_source), str(image_copy))
            service.clone_tags_for_copies([(str(image_source), str(image_copy))])
            shutil.move(str(image_source), str(image_moved))
            service.sync_moved_paths([(str(image_source), str(image_moved))])
            self.assertEqual(("family", "priority"), service.load_tags_for_paths([str(image_copy)])[str(image_copy)])
            tags_by_path = service.load_tags_for_paths([str(image_source), str(image_moved)])
            self.assertEqual((), tags_by_path[str(image_source)])
            self.assertEqual(("family", "priority"), tags_by_path[str(image_moved)])

    def test_image_tags_summarize_cluster_counts_images_per_tag(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_a = root / "a.jpg"
            image_b = root / "b.jpg"
            image_c = root / "c.jpg"
            for image_path in [image_a, image_b, image_c]:
                Image.new("RGB", (32, 32), (100, 120, 140)).save(image_path)
            service = ImageTagService(db_path=root / "tags.sqlite3")
            service.apply_tag_edit([str(image_a)], add_tags=["selfie", "family"], mirror_to_exif=False)
            service.apply_tag_edit([str(image_b)], add_tags=["selfie"], mirror_to_exif=False)
            summary = service.summarize_paths([str(image_a), str(image_b), str(image_c)])
            self.assertEqual(2, summary.tagged_image_count)
            self.assertEqual(2, summary.unique_tag_count)
            self.assertEqual((("selfie", 2), ("family", 1)), summary.top_tags)

    def test_image_tags_inventory_rename_merge_and_delete(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_a = root / "a.jpg"
            image_b = root / "b.jpg"
            Image.new("RGB", (32, 32), (100, 120, 140)).save(image_a)
            Image.new("RGB", (32, 32), (80, 90, 100)).save(image_b)
            service = ImageTagService(db_path=root / "tags.sqlite3")
            service.apply_tag_edit([str(image_a)], add_tags=["Beach"], mirror_to_exif=False)
            service.apply_tag_edit([str(image_b)], add_tags=["Ocean"], mirror_to_exif=False)

            inventory = service.list_tag_inventory()
            self.assertEqual(["Beach", "Ocean"], [item.display_tag for item in inventory])

            self.assertEqual(1, service.rename_tag("Beach", "Ocean"))
            tags_by_path = service.load_tags_for_paths([str(image_a), str(image_b)], import_missing_exif=False)
            self.assertEqual(("Ocean",), tags_by_path[str(image_a)])
            self.assertEqual(("Ocean",), tags_by_path[str(image_b)])
            inventory = service.list_tag_inventory()
            self.assertEqual([("Ocean", 2)], [(item.display_tag, item.image_count) for item in inventory])

            self.assertEqual(2, service.delete_tag("Ocean"))
            self.assertEqual([], service.list_tag_inventory())

    def test_metadata_sidecar_round_trip_preserves_source_images_by_default(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "photo.jpg"
            Image.new("RGB", (32, 32), (10, 20, 30)).save(image_path)
            image_key = str(image_path)
            source_tags = ImageTagService(db_path=root / "source_tags.sqlite3")
            source_tags.apply_tag_edit([image_key], add_tags=["family", "favorite"], mirror_to_exif=False)
            before_bytes = image_path.read_bytes()
            before_mtime = image_path.stat().st_mtime_ns
            sidecar_dir = root / "sidecars"

            service = MetadataSidecarService(tag_service=source_tags)
            written = service.export_sidecars(
                [image_key],
                output_dir=sidecar_dir,
                ratings={image_key: 5},
                captions={image_key: "At the beach"},
                people={image_key: [{"person_name": "Alice", "face_index": 0}]},
                profile_refs={image_key: [{"person_name": "Alice", "profile_id": "Alice"}]},
            )

            sidecar_path = Path(written[image_key])
            payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
            self.assertEqual(1, payload["version"])
            self.assertEqual(image_key, payload["image_path"])
            self.assertEqual(["family", "favorite"], payload["tags"])
            self.assertEqual(5, payload["rating"])
            self.assertEqual("At the beach", payload["caption"])
            self.assertEqual([{"person_name": "Alice", "face_index": 0}], payload["people_labels"])
            self.assertEqual([{"person_name": "Alice", "profile_id": "Alice"}], payload["profile_refs"])

            imported_tags = ImageTagService(db_path=root / "imported_tags.sqlite3")
            imported = MetadataSidecarService(tag_service=imported_tags).import_sidecars([str(sidecar_path)])

            self.assertEqual(["family", "favorite"], imported[image_key]["tags"])
            self.assertEqual(5, imported[image_key]["rating"])
            self.assertEqual("At the beach", imported[image_key]["caption"])
            self.assertEqual(
                ("family", "favorite"),
                imported_tags.load_tags_for_paths([image_key], import_missing_exif=False)[image_key],
            )
            self.assertEqual(before_bytes, image_path.read_bytes())
            self.assertEqual(before_mtime, image_path.stat().st_mtime_ns)

    def test_performance_dashboard_snapshot_normalizes_job_and_pipeline_metrics(self):
        service = PerformanceDashboardService()
        snapshot = service.snapshot(
            {
                "model_load_time_s": 0.42,
                "cache_hits": 7,
                "cache_misses": 3,
                "skipped_unchanged": 11,
                "detector_calls": 5,
                "embedder_calls": 4,
                "ann_state": "stale",
            },
            jobs=[
                {"label": "Indexing faces", "status": "running"},
                {"label": "Searching images", "status": "cancelling"},
            ],
        )

        self.assertEqual("loaded", snapshot.model_load_state)
        self.assertEqual(7, snapshot.cache_hits)
        self.assertEqual(3, snapshot.cache_misses)
        self.assertEqual(11, snapshot.skipped_unchanged)
        self.assertEqual(5, snapshot.detector_calls)
        self.assertEqual(4, snapshot.embedder_calls)
        self.assertEqual("stale", snapshot.ann_state)
        self.assertEqual("cancelling", snapshot.cancellation_status)
        self.assertEqual(("Indexing faces", "Searching images"), snapshot.active_job_labels)
        self.assertIn("Cache: 7 hit/3 miss", snapshot.to_text())
        self.assertIn("ANN: stale", snapshot.to_text())

    def test_embedding_index_snapshot_changes_when_file_changes(self):
        with TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "image.jpg"
            image_path.write_bytes(b"123")
            first_key = EmbeddingIndexService.build_snapshot_key([str(image_path)])
            image_path.write_bytes(b"123456")
            second_key = EmbeddingIndexService.build_snapshot_key([str(image_path)])
            self.assertNotEqual(first_key, second_key)

    def test_embedding_index_reuses_existing_artifacts_for_same_snapshot(self):
        with TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "image.jpg"
            image_path.write_bytes(b"123")
            index = EmbeddingIndexService()
            snapshot = EmbeddingIndexService.build_snapshot_key([str(image_path)])
            embeddings = [(str(image_path), np.asarray([0.1, 0.2], dtype=np.float32))]
            _paths, reused = index.ensure_index(snapshot, "clip", embeddings)
            self.assertFalse(reused)
            _paths, reused = index.ensure_index(snapshot, "clip", embeddings)
            self.assertTrue(reused)

    def test_result_cache_roundtrip(self):
        service = ResultCacheService()
        result_key = service.build_result_key(
            snapshot_key="abc",
            embedding_model="clip",
            similarity_mode="semantic",
            clustering_backend="cosine-kmeans",
            num_clusters=3,
            outlier_policy="assign",
            use_onnx=False,
        )
        clusters = {0: ["a.jpg"], 1: ["b.jpg"]}
        metrics = {"cluster_quality_score": 0.5}
        service.save(result_key, clusters, metrics)
        loaded = service.load(result_key)
        self.assertIsNotNone(loaded)
        loaded_clusters, loaded_metrics = loaded
        self.assertEqual(clusters, loaded_clusters)
        self.assertEqual(metrics["cluster_quality_score"], loaded_metrics["cluster_quality_score"])

    def test_result_cache_key_includes_similarity_space_version(self):
        service = ResultCacheService()
        common = {
            "snapshot_key": "abc",
            "embedding_model": "clip",
            "similarity_mode": "semantic",
            "clustering_backend": "cosine-kmeans",
            "num_clusters": 3,
            "outlier_policy": "assign",
            "use_onnx": False,
        }
        current_key = service.build_result_key(**common)
        old_key = service.build_result_key(**common, similarity_space_version="old-semantic-cosine")
        self.assertNotEqual(current_key, old_key)

    def test_cache_maintenance_clears_only_rebuildable_targets_and_recreates_directories(self):
        with TemporaryDirectory() as tmp:
            cache_root = Path(tmp) / "cache"
            cache_root.mkdir(parents=True, exist_ok=True)
            thumbnails_dir = cache_root / "thumbnails"
            targets = {
                "embeddings.sqlite3": cache_root / "embeddings.sqlite3",
                "cluster_results/": cache_root / "cluster_results",
                "cluster_meanings/": cache_root / "cluster_meanings",
                "embedding_indexes/": cache_root / "embedding_indexes",
                "face_model_assets/": cache_root / "face_model_assets",
                "thumbnails/": thumbnails_dir,
                "onnx_models/": cache_root / "onnx_models",
                "tmp/": cache_root / "tmp",
            }
            excluded_files = [
                cache_root / "image_tags.sqlite3",
                cache_root / "face_search_index.db",
                cache_root / "face_search_session.db",
                cache_root / "global_image_search.db",
                cache_root / "global_image_search_session.db",
                cache_root / "perceptual_hashes.db",
                cache_root / "perceptual_hashes_session.db",
            ]
            excluded_dirs = [cache_root / "huggingface", cache_root / "torch"]
            for name, path in targets.items():
                if name.endswith("/"):
                    path.mkdir(parents=True, exist_ok=True)
                    (path / "payload.bin").write_bytes(b"1234")
                else:
                    path.write_bytes(b"1234")
            for path in excluded_files:
                path.write_bytes(b"keep")
            for path in excluded_dirs:
                path.mkdir(parents=True, exist_ok=True)
                (path / "keep.bin").write_bytes(b"keep")

            settings = SimpleNamespace(
                cache_dir=cache_root,
                embedding_cache_db=cache_root / "embeddings.sqlite3",
                thumbnail_cache_dir=thumbnails_dir,
            )
            service = CacheMaintenanceService(settings=settings)
            summary = service.describe_rebuildable_caches()
            self.assertGreater(summary.total_bytes, 0)

            cleared, failures = service.clear_rebuildable_disk_targets()

            self.assertEqual(set(targets.keys()), set(cleared))
            self.assertEqual((), failures)
            self.assertFalse((cache_root / "embeddings.sqlite3").exists())
            for directory_name in ("cluster_results/", "cluster_meanings/", "embedding_indexes/", "face_model_assets/", "thumbnails/", "onnx_models/", "tmp/"):
                path = targets[directory_name]
                self.assertTrue(path.exists())
                self.assertEqual([], list(path.iterdir()))
            for path in excluded_files:
                self.assertTrue(path.exists())
            for path in excluded_dirs:
                self.assertTrue((path / "keep.bin").exists())

    def test_cache_maintenance_reports_runtime_temp_files_separately(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            tmp_dir = cache_root / "tmp"
            tmp_dir.mkdir(parents=True)
            (tmp_dir / "leftover.tmp").write_bytes(b"temp")
            (cache_root / "embeddings.sqlite3").write_bytes(b"embed")
            tag_db = cache_root / "image_tags.sqlite3"
            tag_db.write_bytes(b"tags")
            runtime_layout = SimpleNamespace(
                root=root,
                logs_dir=root / "logs",
                crash_dir=root / "crash",
                benchmarks_dir=root / "benchmarks",
                support_dir=root / "support",
                model_assets_dir=root / "model_assets",
            )
            for path in (
                runtime_layout.logs_dir,
                runtime_layout.crash_dir,
                runtime_layout.benchmarks_dir,
                runtime_layout.support_dir,
                runtime_layout.model_assets_dir,
            ):
                path.mkdir(parents=True, exist_ok=True)
            settings = SimpleNamespace(
                cache_dir=cache_root,
                embedding_cache_db=cache_root / "embeddings.sqlite3",
                thumbnail_cache_dir=cache_root / "thumbnails",
                image_tags_db=tag_db,
            )

            summary = CacheMaintenanceService(settings=settings).describe_runtime_storage(runtime_layout)

            self.assertEqual(4, summary.target_bytes["runtime_temp_files"])
            self.assertEqual(4, summary.target_bytes["tag_database"])
            self.assertGreaterEqual(summary.target_bytes["rebuildable_caches"], 5)

    def test_cache_maintenance_can_clear_only_runtime_temp_files(self):
        with TemporaryDirectory() as tmp:
            cache_root = Path(tmp) / "cache"
            tmp_dir = cache_root / "tmp"
            thumbnails_dir = cache_root / "thumbnails"
            partial_dir = cache_root / "huggingface" / "hub" / "models--demo"
            tmp_dir.mkdir(parents=True)
            thumbnails_dir.mkdir(parents=True)
            partial_dir.mkdir(parents=True)
            (tmp_dir / "leftover.tmp").write_bytes(b"temp")
            (thumbnails_dir / "thumb.bin").write_bytes(b"keep")
            (partial_dir / "download.incomplete").write_bytes(b"partial")
            settings = SimpleNamespace(
                cache_dir=cache_root,
                embedding_cache_db=cache_root / "embeddings.sqlite3",
                thumbnail_cache_dir=thumbnails_dir,
            )

            cleared, failures = CacheMaintenanceService(settings=settings).clear_runtime_temp_files()

            self.assertIn("tmp/", cleared)
            self.assertTrue(any("download.incomplete" in item for item in cleared))
            self.assertEqual((), failures)
            self.assertTrue(tmp_dir.exists())
            self.assertEqual([], list(tmp_dir.iterdir()))
            self.assertTrue((thumbnails_dir / "thumb.bin").exists())
            self.assertFalse((partial_dir / "download.incomplete").exists())

    def test_cache_maintenance_reports_generated_storage_categories(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            log_dir = root / "logs"
            thumbnails_dir = cache_root / "thumbnails"
            for path in (
                log_dir,
                thumbnails_dir,
                cache_root / "cluster_results",
                cache_root / "cluster_meanings",
                cache_root / "embedding_indexes",
                cache_root / "face_model_assets",
                cache_root / "huggingface",
                cache_root / "tmp",
            ):
                path.mkdir(parents=True, exist_ok=True)
                (path / "payload.bin").write_bytes(b"1234")
            embedding_db = cache_root / "embeddings.sqlite3"
            embedding_db.write_bytes(b"embed")
            face_db = cache_root / "face_search_human.db"
            face_db.write_bytes(b"faces")
            ann_file = cache_root / "face_search_human.db.faiss"
            ann_file.write_bytes(b"ann")
            settings = SimpleNamespace(
                base_dir=root,
                cache_dir=cache_root,
                log_dir=log_dir,
                embedding_cache_db=embedding_db,
                thumbnail_cache_dir=thumbnails_dir,
            )

            summary = CacheMaintenanceService(settings=settings).describe_generated_storage(
                config_location=str(root / "settings.ini")
            )

            self.assertEqual(str(root), summary.runtime_root)
            self.assertEqual(str(root / "settings.ini"), summary.config_location)
            for key in ("logs", "thumbnails", "rebuildable_caches", "face_databases", "ann_files", "model_caches", "temp_files"):
                self.assertIn(key, summary.target_bytes)
                self.assertIn(key, summary.target_paths)
                self.assertGreater(summary.target_bytes[key], 0)
            self.assertIn(str(face_db), summary.target_paths["face_databases"])
            self.assertIn(str(ann_file), summary.target_paths["ann_files"])

    def test_cache_maintenance_clears_generated_storage_categories(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            log_dir = root / "logs"
            thumbnails_dir = cache_root / "thumbnails"
            temp_dir = cache_root / "tmp"
            model_dirs = [
                cache_root / "face_model_assets",
                cache_root / "onnx_models",
                cache_root / "huggingface",
                cache_root / "torch",
            ]
            for path in [log_dir, thumbnails_dir, temp_dir, *model_dirs]:
                path.mkdir(parents=True, exist_ok=True)
                (path / "payload.bin").write_bytes(b"1234")
            partial_dir = cache_root / "huggingface" / "hub" / "models--demo"
            partial_dir.mkdir(parents=True, exist_ok=True)
            (partial_dir / "download.incomplete").write_bytes(b"partial")
            face_db = cache_root / "face_search_human.db"
            face_db.write_bytes(b"faces")
            ann_file = cache_root / "face_search_human.db.faiss"
            ann_file.write_bytes(b"ann")
            embedding_db = cache_root / "embeddings.sqlite3"
            embedding_db.write_bytes(b"embed")
            settings = SimpleNamespace(
                base_dir=root,
                cache_dir=cache_root,
                log_dir=log_dir,
                embedding_cache_db=embedding_db,
                thumbnail_cache_dir=thumbnails_dir,
            )
            service = CacheMaintenanceService(settings=settings)

            cleared_faces, face_failures = service.clear_face_storage_targets()
            self.assertEqual((), face_failures)
            self.assertFalse(face_db.exists())
            self.assertFalse(ann_file.exists())
            self.assertTrue(any("face_search_human.db" in item for item in cleared_faces))

            cleared_models, model_failures = service.clear_model_cache_targets()
            self.assertEqual((), model_failures)
            self.assertTrue((cache_root / "face_model_assets").exists())
            self.assertTrue((cache_root / "onnx_models").exists())
            self.assertFalse((cache_root / "huggingface").exists())
            self.assertFalse((cache_root / "torch").exists())
            self.assertTrue(cleared_models)

            cleared_logs, log_failures = service.clear_log_files()
            self.assertEqual((), log_failures)
            self.assertTrue(log_dir.exists())
            self.assertEqual([], list(log_dir.iterdir()))
            self.assertTrue(cleared_logs)

            cleared_temp, temp_failures = service.clear_runtime_temp_files()
            self.assertEqual((), temp_failures)
            self.assertIn("tmp/", cleared_temp)
            self.assertTrue(temp_dir.exists())
            self.assertEqual([], list(temp_dir.iterdir()))

    def test_model_asset_service_detects_bundled_fallback_for_missing_download_model(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            assets_root = root / "model_assets"
            fast_preview = assets_root / "fast_preview"
            fast_preview.mkdir(parents=True)
            model_path = fast_preview / "model.onnx"
            model_path.write_bytes(b"onnx")
            (fast_preview / "metadata.json").write_text(
                json.dumps(
                    {
                        "model_name": "fast_preview",
                        "input_size": [224, 224],
                        "signature": "fast_preview:224x224:torchvision:onnx=True",
                        "sha256": sha256_file(model_path),
                    }
                ),
                encoding="utf-8",
            )
            settings = SimpleNamespace(cache_dir=root / "cache", base_dir=root)
            service = ModelAssetService(runtime_model_assets_dir=assets_root, settings=settings, extra_roots=())

            plan = service.build_download_plan(
                ["dino"],
                generate_cluster_meanings=False,
                requested_meaning_model="auto",
            )

            self.assertTrue(plan.requires_download)
            self.assertEqual(("dino",), plan.models_requiring_download)
            self.assertEqual("fast_preview", plan.bundled_fallback_model)
            self.assertIn("fast_preview", plan.bundled_models)

    def test_model_asset_service_validates_packaged_sha256(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            assets_root = root / "model_assets"
            model_dir = assets_root / "fast_preview"
            model_dir.mkdir(parents=True)
            model_path = model_dir / "model.onnx"
            model_path.write_bytes(b"onnx")
            (model_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "model_name": "fast_preview",
                        "input_size": [224, 224],
                        "signature": "fast_preview:224x224:torchvision:onnx=True",
                        "sha256": sha256_file(model_path),
                    }
                ),
                encoding="utf-8",
            )
            settings = SimpleNamespace(cache_dir=root / "cache", base_dir=root)
            service = ModelAssetService(runtime_model_assets_dir=assets_root, settings=settings, extra_roots=())

            bundle = service.find_bundle("fast_preview")
            self.assertIsNotNone(bundle)
            self.assertEqual(("fast_preview",), service.bundled_model_names())

            model_path.write_bytes(b"changed")
            self.assertIsNone(service.find_bundle("fast_preview"))
            inventory = service.model_inventory(["fast_preview"])
            self.assertEqual("invalid", inventory[0].validation_status)
            self.assertFalse(inventory[0].available_without_download)

    def test_model_asset_service_flags_advanced_meaning_sidecar_download(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = SimpleNamespace(cache_dir=root / "cache", base_dir=root)
            service = ModelAssetService(settings=settings, extra_roots=())

            plan = service.build_download_plan(
                ["dino"],
                generate_cluster_meanings=True,
                requested_meaning_model="auto",
            )

            self.assertTrue(plan.requires_download)
            self.assertEqual("clip", plan.meaning_model)
            self.assertTrue(plan.meaning_requires_download)
            self.assertIn("clip (advanced cluster naming sidecar)", plan.unavailable_items)

    def test_model_asset_service_requires_complete_hf_snapshot_for_offline_use(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = SimpleNamespace(cache_dir=root / "cache", base_dir=root)
            service = ModelAssetService(settings=settings, extra_roots=())
            repo_dir = root / "cache" / "huggingface" / "hub" / "models--apple--mobileclip_s0_timm"
            (repo_dir / "refs").mkdir(parents=True)
            (repo_dir / "refs" / "main").write_text("snapshot-id", encoding="utf-8")

            self.assertFalse(service.local_cache_present("mobileclip"))

            snapshot_dir = repo_dir / "snapshots" / "snapshot-id"
            snapshot_dir.mkdir(parents=True)
            (snapshot_dir / "config.json").write_text("{}", encoding="utf-8")
            (snapshot_dir / "model.safetensors").write_text("weights", encoding="utf-8")

            self.assertTrue(service.local_cache_present("mobileclip"))
            self.assertTrue(service.model_available_without_download("mobileclip"))

    def test_similarity_graph_clusters_connected_points(self):
        vectors = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.99, 0.01, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.99, 0.01],
            ],
            dtype=np.float32,
        )
        labels = SimilarityGraphService().graph_cluster(vectors, min_similarity=0.95, min_cluster_size=2, k_neighbors=2)
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[2], labels[3])
        self.assertNotEqual(labels[0], labels[2])

    def test_phash_dhash_whash_index_returns_exact_match(self):
        with TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "same.png"
            Image.new("RGB", (32, 32), (255, 0, 0)).save(image_path)
            service = PerceptualHashIndexService()
            service.build_index([str(image_path)])
            for hash_backend in ["phash", "dhash", "whash"]:
                matches = service.query_similar(str(image_path), max_distance=0, hash_backend=hash_backend)
                self.assertTrue(any(match.image_path == str(image_path) and match.hash_distance == 0 for match in matches))

    def test_global_image_index_persists_vectors_and_metadata(self):
        with TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "indexed.png"
            Image.new("RGB", (40, 20), (0, 255, 0)).save(image_path)
            service = GlobalImageIndexService()
            service.upsert_embeddings([(str(image_path), np.array([0.1, 0.2, 0.3], dtype=np.float32))], "clip")
            records = service.load_records("clip", filters={"folder": str(image_path.parent)})
            self.assertTrue(any(record.image_path == str(image_path) and record.width == 40 and record.height == 20 for record in records))

    def test_similarity_search_can_limit_results_to_candidate_paths(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_a = root / "a.png"
            image_b = root / "b.png"
            image_c = root / "c.png"
            for image_path in [image_a, image_b, image_c]:
                Image.new("RGB", (24, 24), (10, 20, 30)).save(image_path)
            index = GlobalImageIndexService(db_path=root / "similarity.sqlite3", reset_db=True)
            index.upsert_embeddings(
                [
                    (str(image_a), np.asarray([1.0, 0.0], dtype=np.float32)),
                    (str(image_b), np.asarray([1.0, 0.0], dtype=np.float32)),
                    (str(image_c), np.asarray([0.0, 1.0], dtype=np.float32)),
                ],
                "clip",
            )
            service = SimilaritySearchService(embedding_service=FakeQueryEmbeddingService(), index_service=index)
            results = service.search(
                SimilaritySearchRequest(
                    query_image_path=str(image_a),
                    search_mode="image",
                    top_k=10,
                    min_score=0.5,
                    embedding_model="clip",
                    candidate_paths=[str(image_b), str(image_c)],
                )
            )
            self.assertEqual([str(image_b)], [result.image_path for result in results])
            del service
            del index
            gc.collect()

    def test_duplicate_review_groups_and_export_decisions_do_not_modify_sources(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            query = root / "query.jpg"
            exact = root / "exact.jpg"
            near = root / "near.jpg"
            for path in (query, exact, near):
                Image.new("RGB", (24, 24), (10, 20, 30)).save(path)
            before = {path: path.stat().st_mtime_ns for path in (query, exact, near)}
            results = [
                SearchResult(str(near), 0.87, 4, "clip", "duplicate", {}, hash_distance=4, hash_backend="phash"),
                SearchResult(str(exact), 0.99, 0, "clip", "duplicate", {}, hash_distance=0, hash_backend="phash"),
            ]

            groups = SimilaritySearchService.build_duplicate_review_groups(str(query), results)
            self.assertEqual([str(exact), str(near)], [group.candidate_path for group in groups])
            self.assertTrue(groups[0].exact)
            self.assertFalse(groups[1].exact)

            out_path = root / "duplicate_decisions.json"
            payload = SimilaritySearchService.export_duplicate_decisions(
                [
                    {
                        "keeper_path": str(query),
                        "duplicate_path": str(exact),
                        "action": "trash",
                        "tag": "duplicate",
                        "score": 0.99,
                        "hash_distance": 0,
                    }
                ],
                out_path,
            )
            self.assertEqual("trash", payload["decisions"][0]["action"])
            self.assertFalse(payload["decisions"][0]["source_files_changed"])
            self.assertEqual(payload, json.loads(out_path.read_text(encoding="utf-8")))
            self.assertEqual(before, {path: path.stat().st_mtime_ns for path in (query, exact, near)})

    def test_face_index_can_store_records_cluster_and_propagate_labels(self):
        with TemporaryDirectory() as tmp:
            image_a = Path(tmp) / "person_a.jpg"
            image_b = Path(tmp) / "person_b.jpg"
            Image.new("RGB", (32, 32), (10, 20, 30)).save(image_a)
            Image.new("RGB", (32, 32), (12, 20, 30)).save(image_b)
            service = FaceIndexService(
                detection_service=FakeFaceDetectionService(),
                embedding_service=FakeFaceEmbeddingService(),
                db_path=Path(tmp) / "faces.sqlite3",
            )
            service.save_face_records(
                [
                    FaceIndexRecord(
                        image_path=str(image_a),
                        face_index=0,
                        face_bbox=(0, 0, 36, 36),
                        face_confidence=0.99,
                        embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                    )
                ],
                mtime_ns=image_a.stat().st_mtime_ns,
                file_size=image_a.stat().st_size,
                assess_quality=False,
            )
            service.save_face_records(
                [
                    FaceIndexRecord(
                        image_path=str(image_b),
                        face_index=0,
                        face_bbox=(0, 0, 36, 36),
                        face_confidence=0.98,
                        embedding=np.array([0.99, 0.01, 0.0], dtype=np.float32),
                    )
                ],
                mtime_ns=image_b.stat().st_mtime_ns,
                file_size=image_b.stat().st_size,
                assess_quality=False,
            )
            clusters = service.cluster_faces(num_clusters=2, min_face_score=0.9)
            self.assertTrue(clusters)
            service.configure_recognition_options(prototype_quality_min="reject")
            service.label_face_examples(FaceLabelRequest(person_name="Alice", example_image_paths=[str(image_a)], similarity_threshold=0.5))
            assignments = service.auto_propagate_labels()
            self.assertTrue(any(item.person_name == "Alice" and item.image_path == str(image_b) for item in assignments))

    def test_face_cluster_compare_returns_membership_explanations_and_identity_suggestions(self):
        with TemporaryDirectory() as tmp:
            image_a = Path(tmp) / "person_a.jpg"
            image_b = Path(tmp) / "person_b.jpg"
            image_c = Path(tmp) / "person_c.jpg"
            image_d = Path(tmp) / "person_d.jpg"
            for index, image_path in enumerate((image_a, image_b, image_c, image_d)):
                Image.new("RGB", (32, 32), (10 + index, 20, 30)).save(image_path)
            service = FaceIndexService(
                detection_service=FakeFaceDetectionService(),
                embedding_service=FakeFaceEmbeddingService(),
                db_path=Path(tmp) / "faces.sqlite3",
            )
            embeddings = [
                np.array([1.0, 0.0, 0.0], dtype=np.float32),
                np.array([0.99, 0.01, 0.0], dtype=np.float32),
                np.array([0.0, 1.0, 0.0], dtype=np.float32),
                np.array([0.01, 0.99, 0.0], dtype=np.float32),
            ]
            for face_index, (image_path, embedding) in enumerate(zip((image_a, image_b, image_c, image_d), embeddings, strict=True)):
                service.save_face_records(
                    [
                        FaceIndexRecord(
                            image_path=str(image_path),
                            face_index=0,
                            face_bbox=(0, 0, 16, 16),
                            face_confidence=0.99,
                            embedding=embedding,
                        )
                    ],
                    mtime_ns=image_path.stat().st_mtime_ns,
                    file_size=image_path.stat().st_size,
                    assess_quality=False,
                )
            service.configure_recognition_options(
                cluster_quality_min="reject",
                prototype_quality_min="reject",
            )
            service.label_indexed_faces("Alice", [(str(image_a), 0), (str(image_b), 0)], similarity_threshold=0.5)
            compare = service.cluster_faces_compare(
                num_clusters=2,
                min_face_score=0.9,
                backends=["cosine-kmeans", "graph"],
                outlier_policy="isolate",
            )
            self.assertEqual(["cosine-kmeans", "graph"], list(compare.clusters_by_key.keys()))
            self.assertIn((str(image_a), 0), compare.membership_by_face_ref)
            self.assertIn("cosine-kmeans", compare.membership_by_face_ref[(str(image_a), 0)])
            self.assertEqual("cosine-kmeans", compare.metrics_by_key["cosine-kmeans"]["requested_backend"])
            self.assertEqual("isolate", compare.metrics_by_key["cosine-kmeans"]["outlier_policy"])
            self.assertTrue(compare.explanations_by_key["cosine-kmeans"])
            self.assertTrue(compare.identity_suggestions_by_key["cosine-kmeans"])

    def test_face_cluster_backend_choices_follow_shared_production_set(self):
        backend_ids = [item_id for item_id, _label in face_cluster_backend_choices()]
        self.assertEqual(["cosine-kmeans", "hdbscan", "graph"], backend_ids)

    def test_clustering_service_hdbscan_backend_options_are_forwarded(self):
        service = ClusteringService()
        captured: dict[str, object] = {}

        class _FakeHdbscanEstimator:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def fit_predict(self, matrix):
                _ = matrix
                return np.asarray([0, 0, -1], dtype=np.int32)

        with patch("ml.clustering.hdbscan", SimpleNamespace(HDBSCAN=lambda **kwargs: _FakeHdbscanEstimator(**kwargs))):
            service.cluster_prepared(
                np.asarray(
                    [
                        [1.0, 0.0],
                        [0.98, 0.02],
                        [0.0, 1.0],
                    ],
                    dtype=np.float32,
                ),
                num_clusters=3,
                backend="hdbscan",
                outlier_policy="keep",
                backend_options={
                    "min_cluster_size": 6,
                    "min_samples": 3,
                    "cluster_selection_epsilon": 0.18,
                    "allow_single_cluster": True,
                },
            )

        self.assertEqual(6, captured["min_cluster_size"])
        self.assertEqual(3, captured["min_samples"])
        self.assertAlmostEqual(0.18, float(captured["cluster_selection_epsilon"]), places=6)
        self.assertTrue(captured["allow_single_cluster"])
        self.assertEqual("eom", captured["cluster_selection_method"])

    def test_face_detection_rejects_tiny_boxes_and_query_face_uses_visible_detection(self):
        class _FakeDetector:
            def detect(self, _rgb_image):
                return (
                    np.asarray([[0.0, 0.0, 18.0, 18.0], [0.0, 0.0, 42.0, 48.0]], dtype=np.float32),
                    np.asarray([0.99, 0.95], dtype=np.float32),
                )

        with TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "query.jpg"
            Image.new("RGB", (64, 64), (20, 30, 40)).save(image_path)
            service = FaceDetectionService()
            service._detector = _FakeDetector()

            faces = service.detect_faces(str(image_path))
            self.assertEqual([(0, 0, 42, 48)], [face.bbox for face in faces])

            query_face = service.detect_query_face(str(image_path))
            self.assertIsNotNone(query_face)
            self.assertEqual((0, 0, 42, 48), query_face.bbox)

    def test_face_visibility_filter_hides_tiny_records_when_requested(self):
        with TemporaryDirectory() as tmp:
            image_a = Path(tmp) / "alice_big.jpg"
            image_b = Path(tmp) / "alice_small.jpg"
            image_c = Path(tmp) / "alice_big_2.jpg"
            for image_path in [image_a, image_b, image_c]:
                Image.new("RGB", (64, 64), (10, 20, 30)).save(image_path)
            service = FaceIndexService(
                detection_service=FakeFaceDetectionService(),
                embedding_service=FakeFaceEmbeddingService(),
                db_path=Path(tmp) / "faces.sqlite3",
            )
            service.save_face_records(
                [
                    FaceIndexRecord(
                        image_path=str(image_a),
                        face_index=0,
                        face_bbox=(0, 0, 40, 42),
                        face_confidence=0.99,
                        embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                    )
                ],
                mtime_ns=image_a.stat().st_mtime_ns,
                file_size=image_a.stat().st_size,
                assess_quality=False,
            )
            service.save_face_records(
                [
                    FaceIndexRecord(
                        image_path=str(image_b),
                        face_index=0,
                        face_bbox=(0, 0, 20, 20),
                        face_confidence=0.98,
                        embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                    )
                ],
                mtime_ns=image_b.stat().st_mtime_ns,
                file_size=image_b.stat().st_size,
                assess_quality=False,
            )
            service.save_face_records(
                [
                    FaceIndexRecord(
                        image_path=str(image_c),
                        face_index=0,
                        face_bbox=(0, 0, 38, 44),
                        face_confidence=0.97,
                        embedding=np.array([0.99, 0.01, 0.0], dtype=np.float32),
                    )
                ],
                mtime_ns=image_c.stat().st_mtime_ns,
                file_size=image_c.stat().st_size,
                assess_quality=False,
            )
            service.label_indexed_faces(
                "Alice",
                [(str(image_a), 0), (str(image_b), 0), (str(image_c), 0)],
                similarity_threshold=0.5,
            )
            pending = service.load_pending_face_labels(include_tiny_faces=True)
            service.accept_pending_face_labels([assignment.proposal_id for assignment in pending])

            self.assertEqual(3, service.count_indexed_faces(include_tiny_faces=True))
            self.assertEqual(2, service.count_indexed_faces(include_tiny_faces=False))
            self.assertEqual(2, len(service.load_indexed_faces(include_tiny_faces=False, limit=10)))
            self.assertEqual(
                [str(image_a), str(image_c)],
                [result.image_path for result in service.search_by_person_name("Alice", include_tiny_faces=False)],
            )
            self.assertEqual(2, len(service.list_face_labels(include_tiny_faces=False)))

    def test_rescanning_replaces_old_face_rows_when_stricter_detection_finds_only_tiny_boxes(self):
        class _MutableDetector:
            def __init__(self):
                self.boxes = np.asarray([[0.0, 0.0, 40.0, 44.0]], dtype=np.float32)

            def detect(self, _rgb_image):
                return self.boxes, np.asarray([0.99], dtype=np.float32)

        with TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "rescan.jpg"
            Image.new("RGB", (80, 80), (30, 40, 50)).save(image_path)
            detection_service = FaceDetectionService()
            detection_service._detector = _MutableDetector()
            service = FaceIndexService(
                detection_service=detection_service,
                embedding_service=FakeFaceEmbeddingService(),
                db_path=Path(tmp) / "faces.sqlite3",
            )

            service.index_paths([str(image_path)])
            self.assertEqual(1, service.count_indexed_faces(include_tiny_faces=True))

            detection_service._detector.boxes = np.asarray([[0.0, 0.0, 18.0, 18.0]], dtype=np.float32)
            service.index_paths([str(image_path)], force=True)
            self.assertEqual(0, service.count_indexed_faces(include_tiny_faces=True))

    def test_index_paths_prefetches_existing_scan_rows_once_for_unchanged_images(self):
        class CountingDetectionService(FakeFaceDetectionService):
            def __init__(self):
                self.detect_calls = 0

            def detect_faces(self, image_path):
                self.detect_calls += 1
                return super().detect_faces(image_path)

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_paths: list[str] = []
            for index in range(10):
                image_path = root / f"prefetch-{index}.jpg"
                Image.new("RGB", (48, 48), (index, index, index)).save(image_path)
                image_paths.append(str(image_path))
            detector = CountingDetectionService()
            service = FaceIndexService(
                detection_service=detector,
                embedding_service=FakeFaceEmbeddingService(),
                db_path=root / "faces.sqlite3",
            )
            service.index_paths(image_paths)
            self.assertEqual(10, detector.detect_calls)

            real_connect = service._connect
            connect_calls = 0

            def counted_connect():
                nonlocal connect_calls
                connect_calls += 1
                return real_connect()

            service._connect = counted_connect  # type: ignore[method-assign]
            metrics = service.index_paths(image_paths)

            self.assertEqual(10, metrics["skipped_unchanged"])
            self.assertEqual(10, detector.detect_calls)
            self.assertEqual(1, connect_calls)

    def test_index_paths_throttles_progress_updates_for_large_skipped_batches(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_paths: list[str] = []
            for index in range(128):
                image_path = root / f"progress-{index}.jpg"
                Image.new("RGB", (24, 24), (index % 255, 10, 20)).save(image_path)
                image_paths.append(str(image_path))
            service = FaceIndexService(
                detection_service=FakeFaceDetectionService(),
                embedding_service=FakeFaceEmbeddingService(),
                db_path=root / "faces.sqlite3",
            )
            service.set_face_recognition_enabled(False)
            progress_events: list[tuple[int, str]] = []

            metrics = service.index_paths(
                image_paths,
                progress_callback=lambda value, status: progress_events.append((int(value), str(status))),
            )

            self.assertEqual(128, metrics["images_done"])
            self.assertEqual(128, metrics["skipped_recognition_disabled"])
            self.assertLessEqual(len(progress_events), 33)
            self.assertEqual(100, progress_events[-1][0])
            self.assertIn("skipped because face recognition is disabled", progress_events[-1][1])

    def test_save_image_faces_preserves_matching_labels_and_can_clear_one_image(self):
        class _Face:
            def __init__(self, image_path: str, bbox: tuple[int, int, int, int]):
                self.image_path = image_path
                self.bbox = bbox
                self.confidence = 0.95
                self.crop = Image.new("RGB", (max(1, bbox[2] - bbox[0]), max(1, bbox[3] - bbox[1])), (255, 255, 255))

        class _Detector:
            def detect_query_face(self, image_path, query_bbox=None):
                bbox = tuple(query_bbox or (0, 0, 32, 32))
                return _Face(str(image_path), bbox)

            def detect_faces(self, image_path):
                return [_Face(str(image_path), (0, 0, 32, 32))]

        with TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "editable.jpg"
            Image.new("RGB", (96, 96), (40, 50, 60)).save(image_path)
            service = FaceIndexService(
                detection_service=_Detector(),
                embedding_service=FakeFaceEmbeddingService(),
                db_path=Path(tmp) / "faces.sqlite3",
            )
            service.save_face_records(
                [
                    FaceIndexRecord(
                        image_path=str(image_path),
                        face_index=0,
                        face_bbox=(0, 0, 32, 32),
                        face_confidence=0.99,
                        embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                    )
                ],
                mtime_ns=image_path.stat().st_mtime_ns,
                file_size=image_path.stat().st_size,
            )
            service.configure_recognition_options(prototype_quality_min="reject")
            service.label_indexed_faces("Alice", [(str(image_path), 0)], similarity_threshold=0.5)
            pending = service.load_pending_face_labels(include_tiny_faces=True)
            service.accept_pending_face_labels([assignment.proposal_id for assignment in pending])

            detected = service.detect_faces_for_image(str(image_path))
            self.assertEqual([(0, 0, 32, 32)], [face.bbox for face in detected])

            updated = service.save_image_faces(
                str(image_path),
                [
                    EditableFaceInput(bbox=(0, 0, 32, 32), confidence=0.99),
                    EditableFaceInput(bbox=(36, 8, 72, 48), confidence=0.88),
                ],
                preserve_labels=True,
            )
            self.assertEqual(2, len(updated))
            self.assertEqual("Alice", updated[0].person_name)
            self.assertEqual("", updated[1].person_name)
            self.assertEqual([(0, 0, 32, 32), (36, 8, 72, 48)], [record.face_bbox for record in updated])

            cleared = service.save_image_faces(str(image_path), [], preserve_labels=True)
            self.assertEqual([], cleared)
            self.assertEqual([], service.load_image_faces(str(image_path), include_tiny_faces=True))

    def test_index_paths_persists_face_quality_metadata(self):
        class _Face:
            def __init__(self, image_path: str, bbox: tuple[int, int, int, int], confidence: float):
                self.image_path = image_path
                self.bbox = bbox
                self.confidence = confidence
                width = max(1, bbox[2] - bbox[0])
                height = max(1, bbox[3] - bbox[1])
                pattern = ((np.indices((height, width)).sum(axis=0) % 2) * 255).astype(np.uint8)
                self.crop = Image.fromarray(np.stack([pattern, pattern, pattern], axis=-1), mode="RGB")
                self.landmarks = ()

        class _Detector:
            def detect_faces(self, image_path):
                return [_Face(str(image_path), (0, 0, 32, 36), 0.52)]

        with TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "quality.jpg"
            checker = ((np.indices((80, 80)).sum(axis=0) % 2) * 255).astype(np.uint8)
            Image.fromarray(np.stack([checker, checker, checker], axis=-1), mode="RGB").save(image_path)
            service = FaceIndexService(
                detection_service=_Detector(),
                embedding_service=FakeFaceEmbeddingService(),
                db_path=Path(tmp) / "faces.sqlite3",
            )

            service.index_paths([str(image_path)])

            records = service.load_image_faces(str(image_path), include_tiny_faces=True)
            self.assertEqual(1, len(records))
            self.assertEqual("review", records[0].quality_status)
            self.assertIn("review_confidence", records[0].quality_reasons)

    def test_save_image_faces_persists_quality_metadata(self):
        class _Face:
            def __init__(self, image_path: str, bbox: tuple[int, int, int, int]):
                self.image_path = image_path
                self.bbox = bbox
                self.confidence = 0.25
                self.crop = Image.new("RGB", (max(1, bbox[2] - bbox[0]), max(1, bbox[3] - bbox[1])), (120, 120, 120))

        class _Detector:
            def detect_query_face(self, image_path, query_bbox=None):
                return _Face(str(image_path), tuple(query_bbox or (0, 0, 20, 20)))

            def detect_faces(self, image_path):
                return [_Face(str(image_path), (0, 0, 20, 20))]

        with TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "editable-quality.jpg"
            Image.new("RGB", (96, 96), (120, 120, 120)).save(image_path)
            service = FaceIndexService(
                detection_service=_Detector(),
                embedding_service=FakeFaceEmbeddingService(),
                db_path=Path(tmp) / "faces.sqlite3",
            )

            saved = service.save_image_faces(
                str(image_path),
                [EditableFaceInput(bbox=(0, 0, 20, 20), confidence=0.25)],
                preserve_labels=True,
            )

            self.assertEqual(1, len(saved))
            self.assertEqual("reject", saved[0].quality_status)
            self.assertIn("low_detector_score", saved[0].quality_reasons)

    def test_load_folder_review_images_backfills_legacy_quality_metadata(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "legacy.jpg"
            checker = ((np.indices((80, 80)).sum(axis=0) % 2) * 255).astype(np.uint8)
            Image.fromarray(np.stack([checker, checker, checker], axis=-1), mode="RGB").save(image_path)
            class _Detector:
                pass
            service = FaceIndexService(
                detection_service=_Detector(),
                embedding_service=FakeFaceEmbeddingService(),
                discovery_service=FakeDiscoveryService([str(image_path)]),
                db_path=root / "faces.sqlite3",
            )

            with sqlite3.connect(str(root / "faces.sqlite3")) as connection:
                connection.execute(
                    """
                    INSERT INTO face_index(
                        image_path, face_index, bbox_json, face_confidence, embedding,
                        quality_status, quality_score, quality_reasons_json, quality_revision,
                        mtime_ns, file_size
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(image_path),
                        0,
                        json.dumps([0, 0, 32, 36]),
                        0.52,
                        np.asarray([1.0, 0.0, 0.0], dtype=np.float32).tobytes(),
                        "clean",
                        1.0,
                        "[]",
                        0,
                        int(image_path.stat().st_mtime_ns),
                        int(image_path.stat().st_size),
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO face_scan_images(image_path, mtime_ns, file_size, face_count, image_width, image_height, indexed_at)
                    VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                    (
                        str(image_path),
                        int(image_path.stat().st_mtime_ns),
                        int(image_path.stat().st_size),
                        1,
                        80,
                        80,
                    ),
                )

            review = service.load_folder_review_images(str(root), include_tiny_faces=True)

            self.assertEqual(1, len(review))
            self.assertEqual("review", review[0].visible_faces[0].quality_status)
            refreshed = service.load_face_record(str(image_path), 0)
            self.assertIsNotNone(refreshed)
            self.assertEqual("review", refreshed.quality_status)
            self.assertIn("review_confidence", refreshed.quality_reasons)

    def test_load_folder_review_images_migrates_legacy_rotated_face_boxes_to_display_space(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "legacy-rotated.jpg"
            image = Image.new("RGB", (100, 60), (80, 90, 100))
            exif = image.getexif()
            exif[274] = 6
            image.save(image_path, exif=exif)

            class _Detector:
                pass

            service = FaceIndexService(
                detection_service=_Detector(),
                embedding_service=FakeFaceEmbeddingService(),
                discovery_service=FakeDiscoveryService([str(image_path)]),
                db_path=root / "faces.sqlite3",
            )

            with sqlite3.connect(str(root / "faces.sqlite3")) as connection:
                connection.execute(
                    """
                    INSERT INTO face_index(
                        image_path, face_index, bbox_json, face_confidence, embedding,
                        quality_status, quality_score, quality_reasons_json, quality_revision,
                        mtime_ns, file_size
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(image_path),
                        0,
                        json.dumps([0, 0, 20, 10]),
                        0.91,
                        np.asarray([1.0, 0.0, 0.0], dtype=np.float32).tobytes(),
                        "clean",
                        1.0,
                        "[]",
                        0,
                        int(image_path.stat().st_mtime_ns),
                        int(image_path.stat().st_size),
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO face_scan_images(image_path, mtime_ns, file_size, face_count, image_width, image_height, indexed_at)
                    VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                    (
                        str(image_path),
                        int(image_path.stat().st_mtime_ns),
                        int(image_path.stat().st_size),
                        1,
                        100,
                        60,
                    ),
                )

            review = service.load_folder_review_images(str(root), include_tiny_faces=True)

            self.assertEqual(1, len(review))
            self.assertEqual(60, review[0].image_width)
            self.assertEqual(100, review[0].image_height)
            self.assertEqual((50, 0, 60, 20), tuple(review[0].visible_faces[0].face_bbox))

            refreshed = service.load_face_record(str(image_path), 0)
            self.assertIsNotNone(refreshed)
            self.assertEqual((50, 0, 60, 20), tuple(refreshed.face_bbox))

    def test_detector_policy_rescue_on_no_faces_uses_fallback_detector(self):
        class _Face:
            def __init__(self, image_path: str, bbox: tuple[int, int, int, int], confidence: float = 0.95):
                self.image_path = image_path
                self.bbox = bbox
                self.confidence = confidence
                width = max(1, bbox[2] - bbox[0])
                height = max(1, bbox[3] - bbox[1])
                pattern = ((np.indices((height, width)).sum(axis=0) % 2) * 255).astype(np.uint8)
                self.crop = Image.fromarray(np.stack([pattern, pattern, pattern], axis=-1), mode="RGB")
                self.landmarks = ()

        class _PrimaryDetector:
            detector_id = "primary"
            def detect_faces(self, image_path):
                _ = image_path
                return []

        class _FallbackDetector:
            detector_id = "fallback"
            def detect_faces(self, image_path):
                return [_Face(str(image_path), (0, 0, 40, 44), 0.91)]

        with TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "cascade.jpg"
            Image.new("RGB", (96, 96), (50, 60, 70)).save(image_path)
            service = FaceIndexService(
                detection_service=_PrimaryDetector(),
                embedding_service=FakeFaceEmbeddingService(),
                db_path=Path(tmp) / "faces.sqlite3",
            )
            service._build_detector_service = lambda detector_id: _FallbackDetector() if detector_id == "fallback" else None
            service.configure_cascade_options(detector_policy="rescue_on_no_faces", fallback_detector_id="fallback")

            faces = service.detect_faces_for_image(str(image_path), include_tiny_faces=True)

            self.assertEqual([(0, 0, 40, 44)], [face.bbox for face in faces])

    def test_search_and_cluster_respect_quality_gates_and_rerank(self):
        with TemporaryDirectory() as tmp:
            service = FaceIndexService(
                detection_service=FakeFaceDetectionService(),
                embedding_service=FakeFaceEmbeddingService(),
                db_path=Path(tmp) / "faces.sqlite3",
            )
            service.save_face_records(
                [
                    FaceIndexRecord(
                        image_path="clean.jpg",
                        face_index=0,
                        face_bbox=(0, 0, 40, 44),
                        face_confidence=0.90,
                        embedding=np.asarray([0.8, 0.2, 0.0], dtype=np.float32),
                        quality_status="clean",
                        quality_score=0.95,
                    ),
                    FaceIndexRecord(
                        image_path="review.jpg",
                        face_index=0,
                        face_bbox=(0, 0, 40, 44),
                        face_confidence=0.90,
                        embedding=np.asarray([0.99, 0.01, 0.0], dtype=np.float32),
                        quality_status="review",
                        quality_score=0.40,
                    ),
                ],
                mtime_ns=1,
                file_size=1,
                image_width=96,
                image_height=96,
                assess_quality=False,
            )
            service._save_person_prototype("Alice", np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32), similarity_threshold=0.2)

            default_results = service.search_by_person_name("Alice", include_tiny_faces=True)
            default_clusters = service.cluster_faces(num_clusters=2, include_tiny_faces=True)

            self.assertEqual(["clean.jpg"], [result.image_path for result in default_results])
            self.assertEqual({}, default_clusters)

            service.configure_recognition_options(
                search_quality_min="review",
                cluster_quality_min="review",
                rerank_policy="quality_score",
                rerank_top_n=2,
            )
            reranked_results = service.search_by_person_name("Alice", include_tiny_faces=True)
            all_clusters = service.cluster_faces(num_clusters=2, include_tiny_faces=True)

            self.assertEqual(["clean.jpg", "review.jpg"], [result.image_path for result in reranked_results])
            self.assertEqual(
                {"clean.jpg", "review.jpg"},
                {member.image_path for members in all_clusters.values() for member in members},
            )

    def test_prototype_quality_gate_and_auto_label_threshold_apply(self):
        with TemporaryDirectory() as tmp:
            service = FaceIndexService(
                detection_service=FakeFaceDetectionService(),
                embedding_service=FakeFaceEmbeddingService(),
                db_path=Path(tmp) / "faces.sqlite3",
            )
            service.save_face_records(
                [
                    FaceIndexRecord(
                        image_path="review.jpg",
                        face_index=0,
                        face_bbox=(0, 0, 40, 44),
                        face_confidence=0.90,
                        embedding=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
                        quality_status="review",
                        quality_score=0.40,
                    )
                ],
                mtime_ns=1,
                file_size=1,
                image_width=96,
                image_height=96,
                assess_quality=False,
            )

            with self.assertRaisesRegex(ValueError, "prototype quality gate"):
                service.label_indexed_faces("Alice", [("review.jpg", 0)], similarity_threshold=0.5)

            service.configure_recognition_options(prototype_quality_min="review", auto_label_min_score=1.01)
            service.label_indexed_faces("Alice", [("review.jpg", 0)], similarity_threshold=0.5)
            assignments = service.auto_propagate_labels(include_tiny_faces=True)
            self.assertEqual([], assignments)

    def test_person_prototype_faces_can_be_loaded_pinned_removed_and_merged(self):
        with TemporaryDirectory() as tmp:
            service = FaceIndexService(
                detection_service=FakeFaceDetectionService(),
                embedding_service=FakeFaceEmbeddingService(),
                db_path=Path(tmp) / "faces.sqlite3",
            )
            service.save_face_records(
                [
                    FaceIndexRecord(
                        image_path="alice_a.jpg",
                        face_index=0,
                        face_bbox=(0, 0, 40, 44),
                        face_confidence=0.95,
                        embedding=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
                        quality_status="clean",
                        quality_score=0.95,
                    ),
                    FaceIndexRecord(
                        image_path="alice_b.jpg",
                        face_index=0,
                        face_bbox=(4, 4, 44, 48),
                        face_confidence=0.94,
                        embedding=np.asarray([0.9, 0.1, 0.0], dtype=np.float32),
                        quality_status="clean",
                        quality_score=0.91,
                    ),
                    FaceIndexRecord(
                        image_path="bob.jpg",
                        face_index=0,
                        face_bbox=(8, 8, 48, 52),
                        face_confidence=0.93,
                        embedding=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
                        quality_status="clean",
                        quality_score=0.89,
                    ),
                ],
                mtime_ns=1,
                file_size=1,
                image_width=96,
                image_height=96,
                assess_quality=False,
            )
            service.label_indexed_faces("Alice", [("alice_a.jpg", 0), ("alice_b.jpg", 0)], similarity_threshold=0.55)
            service.label_indexed_faces("Bob", [("bob.jpg", 0)], similarity_threshold=0.60)

            prototype_faces = service.load_person_prototype_faces("Alice", include_tiny_faces=True)
            self.assertEqual(
                [("alice_a.jpg", 0), ("alice_b.jpg", 0)],
                [(face.image_path, face.face_index) for face in prototype_faces],
            )
            self.assertTrue(prototype_faces[0].pinned)

            service.pin_person_prototype_face("Alice", "alice_b.jpg", 0)
            pinned_faces = service.load_person_prototype_faces("Alice", include_tiny_faces=True)
            self.assertEqual("alice_b.jpg", pinned_faces[0].image_path)
            self.assertTrue(pinned_faces[0].pinned)

            service.remove_person_prototype_face("Alice", "alice_a.jpg", 0)
            remaining_faces = service.load_person_prototype_faces("Alice", include_tiny_faces=True)
            self.assertEqual([("alice_b.jpg", 0)], [(face.image_path, face.face_index) for face in remaining_faces])
            results = service.search_by_person_name("Alice", include_tiny_faces=True)
            self.assertGreaterEqual(len(results), 1)
            self.assertEqual("alice_b.jpg", results[0].image_path)

            service.accept_pending_face_labels([assignment.proposal_id for assignment in service.load_pending_face_labels(include_tiny_faces=True)])
            service.merge_person_identities("Alice", "Bob")
            self.assertIsNone(service.find_person_prototype("Alice"))
            merged_faces = service.load_person_prototype_faces("Bob", include_tiny_faces=True)
            self.assertEqual(
                {"alice_b.jpg", "bob.jpg"},
                {face.image_path for face in merged_faces},
            )
            merged_labels = service.list_face_labels(include_tiny_faces=True)
            self.assertIn(("Bob", "alice_b.jpg"), {(item.person_name, item.image_path) for item in merged_labels})

    def test_identity_duplicate_warnings_flags_similar_saved_prototypes(self):
        with TemporaryDirectory() as tmp:
            service = FaceIndexService(
                detection_service=FakeFaceDetectionService(),
                embedding_service=FakeFaceEmbeddingService(),
                db_path=Path(tmp) / "faces.sqlite3",
            )
            service._save_person_prototype("Alice", np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32), similarity_threshold=0.55)
            service._save_person_prototype("Alicia", np.asarray([[0.99, 0.01, 0.0]], dtype=np.float32), similarity_threshold=0.55)

            warnings = service.identity_duplicate_warnings(similarity_threshold=0.95)

            self.assertIn("Alice", warnings)
            self.assertEqual("Alicia", warnings["Alice"][0][0])

    def test_face_folder_review_distinguishes_detected_no_faces_tiny_hidden_and_not_scanned(self):
        class _Face:
            def __init__(self, image_path: str, bbox: tuple[int, int, int, int]):
                self.image_path = image_path
                self.bbox = bbox
                self.confidence = 0.99
                self.crop = Image.new("RGB", (max(1, bbox[2] - bbox[0]), max(1, bbox[3] - bbox[1])), (255, 255, 255))

        class _Detector:
            def detect_query_face(self, image_path, query_bbox=None):
                _ = query_bbox
                return _Face(str(image_path), (0, 0, 40, 44))

            def detect_faces(self, image_path):
                name = Path(image_path).name
                if name == "detected.jpg":
                    return [_Face(str(image_path), (0, 0, 40, 44))]
                if name == "tiny.jpg":
                    return [_Face(str(image_path), (0, 0, 20, 20))]
                return []

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            detected = root / "detected.jpg"
            no_faces = root / "no_faces.jpg"
            tiny = root / "tiny.jpg"
            not_scanned = root / "not_scanned.jpg"
            for image_path in [detected, no_faces, tiny, not_scanned]:
                Image.new("RGB", (80, 60), (20, 30, 40)).save(image_path)
            service = FaceIndexService(
                detection_service=_Detector(),
                embedding_service=FakeFaceEmbeddingService(),
                db_path=root / "faces.sqlite3",
            )

            service.index_paths([str(detected), str(no_faces), str(tiny)])

            scan_records = {record.image_path: record for record in service.load_scan_image_records(folder_prefix=str(root))}
            self.assertEqual({str(detected), str(no_faces), str(tiny)}, set(scan_records))
            self.assertEqual(1, scan_records[str(detected)].face_count)
            self.assertEqual(0, scan_records[str(no_faces)].face_count)
            self.assertEqual(1, scan_records[str(tiny)].face_count)

            review = {item.image_path: item for item in service.load_folder_review_images(str(root), include_tiny_faces=False)}
            self.assertEqual("detected", review[str(detected)].review_status)
            self.assertEqual(1, len(review[str(detected)].visible_faces))
            self.assertEqual("no_faces", review[str(no_faces)].review_status)
            self.assertEqual("tiny_hidden", review[str(tiny)].review_status)
            self.assertEqual(1, review[str(tiny)].hidden_face_count)
            self.assertEqual("not_scanned", review[str(not_scanned)].review_status)

    def test_face_folder_review_uses_indexed_paths_when_candidate_snapshot_is_empty(self):
        class _Face:
            def __init__(self, image_path: str, bbox: tuple[int, int, int, int]):
                self.image_path = image_path
                self.bbox = bbox
                self.confidence = 0.99
                self.crop = Image.new("RGB", (max(1, bbox[2] - bbox[0]), max(1, bbox[3] - bbox[1])), (255, 255, 255))

        class _Detector:
            def detect_query_face(self, image_path, query_bbox=None):
                _ = query_bbox
                return _Face(str(image_path), (0, 0, 40, 44))

            def detect_faces(self, image_path):
                return [_Face(str(image_path), (0, 0, 40, 44))]

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            detected = root / "detected.jpg"
            Image.new("RGB", (80, 60), (20, 30, 40)).save(detected)
            service = FaceIndexService(
                detection_service=_Detector(),
                embedding_service=FakeFaceEmbeddingService(),
                db_path=root / "faces.sqlite3",
            )

            service.index_paths([str(detected)])
            review = service.load_folder_review_images(str(root), candidate_paths=[], include_tiny_faces=True)

            self.assertEqual([str(detected)], [item.image_path for item in review])
            self.assertEqual("detected", review[0].review_status)

    def test_face_folder_review_chunks_large_candidate_path_snapshots(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = FaceIndexService(
                detection_service=FakeFaceDetectionService(),
                embedding_service=FakeFaceEmbeddingService(),
                db_path=root / "faces.sqlite3",
            )
            candidate_paths = [str(root / f"image_{index:04d}.jpg") for index in range(1005)]

            review = service.load_folder_review_images(str(root), candidate_paths=candidate_paths, include_tiny_faces=True)

            self.assertEqual(1005, len(review))
            self.assertTrue(all(item.review_status == "not_scanned" for item in review))

    def test_animal_bundle_status_reports_missing_and_ready_states(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("app.services.face_search.ort", object()):
                ready, message = inspect_animal_face_bundle(root, "dog")
                self.assertFalse(ready)
                self.assertIn("Missing Dog model bundle directory", message)

                bundle_dir = root / "dog"
                bundle_dir.mkdir(parents=True, exist_ok=True)
                (bundle_dir / "detector.onnx").write_bytes(b"detector")
                (bundle_dir / "embedder.onnx").write_bytes(b"embedder")
                (bundle_dir / "metadata.json").write_text(
                    json.dumps(
                        {
                            "display_name": "Dog",
                            "detector": {
                                "input_name": "images",
                                "input_size": [320, 320],
                                "output_boxes_name": "boxes",
                                "output_scores_name": "scores",
                                "mean": [0.0, 0.0, 0.0],
                                "std": [255.0, 255.0, 255.0],
                            },
                            "embedder": {
                                "input_name": "input",
                                "input_size": [160, 160],
                                "output_name": "embedding",
                                "mean": [0.5, 0.5, 0.5],
                                "std": [0.5, 0.5, 0.5],
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                ready, message = inspect_animal_face_bundle(root, "dog")
                self.assertTrue(ready)
                self.assertIn("Dog bundle ready", message)

    def test_face_bundle_discovery_lists_builtin_and_custom_components(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            detector_dir = root / "human" / "detectors" / "yolo_face"
            embedder_dir = root / "human" / "embedders" / "arcface_small"
            detector_dir.mkdir(parents=True, exist_ok=True)
            embedder_dir.mkdir(parents=True, exist_ok=True)
            (detector_dir / "detector.onnx").write_bytes(b"detector")
            (embedder_dir / "embedder.onnx").write_bytes(b"embedder")
            (detector_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "display_name": "YOLO Face",
                        "backend_family": "yolo",
                        "supported_modes": ["human"],
                        "input_name": "images",
                        "input_size": [320, 320],
                        "output_boxes_name": "boxes",
                        "output_scores_name": "scores",
                        "mean": [0.0, 0.0, 0.0],
                        "std": [255.0, 255.0, 255.0],
                    }
                ),
                encoding="utf-8",
            )
            (embedder_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "display_name": "ArcFace Small",
                        "supported_modes": ["human"],
                        "input_name": "input",
                        "input_size": [160, 160],
                        "output_name": "embedding",
                        "mean": [0.5, 0.5, 0.5],
                        "std": [0.5, 0.5, 0.5],
                    }
                ),
                encoding="utf-8",
            )

            detector_ids = [item_id for item_id, _label in face_detector_choices(root, "human")]
            embedder_ids = [item_id for item_id, _label in face_embedder_choices(root, "human")]

            self.assertIn(BUILTIN_HUMAN_DETECTOR_ID, detector_ids)
            self.assertIn("yolo_face", detector_ids)
            self.assertIn(BUILTIN_HUMAN_EMBEDDER_ID, embedder_ids)
            self.assertIn("arcface_small", embedder_ids)

    def test_bundled_human_catalog_exposes_curated_models_without_external_root(self):
        detector_ids = [item_id for item_id, _label in face_detector_choices("", "human")]
        embedder_ids = [item_id for item_id, _label in face_embedder_choices("", "human")]

        self.assertIn(BUILTIN_HUMAN_DETECTOR_ID, detector_ids)
        self.assertIn("scrfd_2.5g_kps", detector_ids)
        self.assertIn("yolo5face_n", detector_ids)
        self.assertIn("yunet_2026may", detector_ids)
        self.assertIn("yunet_2023mar", detector_ids)
        self.assertIn("yunet_2023mar_int8bq", detector_ids)
        self.assertIn(BUILTIN_HUMAN_EMBEDDER_ID, embedder_ids)
        self.assertIn("mobilefacenet_arcface", embedder_ids)
        self.assertIn("arcface_r100_glint360k", embedder_ids)
        self.assertIn("arcface_r50", embedder_ids)
        self.assertIn("adaface_r100", embedder_ids)
        self.assertIn("sface_2021dec", embedder_ids)
        self.assertIn("sface_2021dec_int8bq", embedder_ids)
        self.assertEqual("YOLO5Face Nano", resolve_face_detector_bundle("", "human", "yolo5face_n").display_name)
        self.assertEqual("ArcFace R50", resolve_face_embedder_bundle("", "human", "arcface_r50").display_name)

    def test_face_choice_labels_include_cpu_gpu_tags(self):
        detector_labels = {item_id: label for item_id, label in face_detector_choices("", "human")}
        embedder_labels = {item_id: label for item_id, label in face_embedder_choices("", "human")}

        self.assertIn("[CPU/GPU]", detector_labels[BUILTIN_HUMAN_DETECTOR_ID])
        self.assertIn("[GPU]", detector_labels["scrfd_10g_kps"])
        self.assertIn("[CPU/GPU]", detector_labels["scrfd_500m_kps"])
        self.assertIn("[CPU/GPU]", detector_labels["scrfd_2.5g_kps"])
        self.assertIn("[CPU]", detector_labels["yunet_2026may"])
        self.assertIn("[GPU]", embedder_labels["arcface_r100_glint360k"])
        self.assertIn("[CPU/GPU]", embedder_labels[BUILTIN_HUMAN_EMBEDDER_ID])
        self.assertIn("[GPU]", embedder_labels["adaface_r100"])
        self.assertIn("[CPU/GPU]", embedder_labels["arcface_r50"])
        self.assertIn("[CPU/GPU]", embedder_labels["mobilefacenet_arcface"])
        self.assertIn("[CPU]", embedder_labels["sface_2021dec"])
        self.assertIn("[CPU]", embedder_labels["sface_2021dec_int8bq"])

    def test_selected_face_model_profile_recognizes_latest_gpu_and_opencv_cpu(self):
        self.assertEqual(
            "latest_gpu",
            face_search_module.selected_face_model_profile(
                "human",
                "scrfd_10g_kps",
                "arcface_r100_glint360k",
                score_threshold=0.35,
                max_detections=50,
            ),
        )
        self.assertEqual(
            "opencv_cpu",
            face_search_module.selected_face_model_profile(
                "human",
                "yunet_2026may",
                "sface_2021dec",
                score_threshold=0.35,
                max_detections=50,
            ),
        )

    def test_managed_face_model_root_overrides_metadata_only_catalog(self):
        with TemporaryDirectory() as tmp:
            managed_root = Path(tmp)
            detector_dir = managed_root / "human" / "detectors" / "scrfd_2.5g_kps"
            embedder_dir = managed_root / "human" / "embedders" / "arcface_r50"
            detector_dir.mkdir(parents=True, exist_ok=True)
            embedder_dir.mkdir(parents=True, exist_ok=True)
            bundled_root = Path(__file__).resolve().parents[1] / "face_model_assets"
            shutil.copyfile(
                bundled_root / "human" / "detectors" / "scrfd_2.5g_kps" / "metadata.json",
                detector_dir / "metadata.json",
            )
            shutil.copyfile(
                bundled_root / "human" / "embedders" / "arcface_r50" / "metadata.json",
                embedder_dir / "metadata.json",
            )
            (detector_dir / "detector.onnx").write_bytes(b"detector")
            (embedder_dir / "embedder.onnx").write_bytes(b"embedder")
            with patch("app.services.face_search.face_model_runtime_root_dir", return_value=managed_root):
                detector_bundle = resolve_face_detector_bundle("", "human", "scrfd_2.5g_kps")
                embedder_bundle = resolve_face_embedder_bundle("", "human", "arcface_r50")
            self.assertTrue(detector_bundle.available)
            self.assertEqual("managed", detector_bundle.source_kind)
            self.assertTrue(embedder_bundle.available)
            self.assertEqual("managed", embedder_bundle.source_kind)

    def test_face_model_installer_inventory_reflects_managed_cache(self):
        with TemporaryDirectory() as tmp:
            fake_settings = SimpleNamespace(cache_dir=Path(tmp))
            installer = FaceModelInstaller(settings=fake_settings)
            initial = {item.bundle_id: item for item in installer.inventory()}
            self.assertFalse(initial["scrfd_500m_kps"].installed)
            self.assertFalse(initial["mobilefacenet_arcface"].installed)
            self.assertFalse(initial["arcface_r100_glint360k"].installed)
            self.assertFalse(initial["sface_2021dec"].installed)
            self.assertFalse(initial["scrfd_2.5g_kps"].installed)
            self.assertFalse(initial["yunet_2026may"].installed)
            bundle_dir = installer.bundle_dir("scrfd_2.5g_kps")
            bundle_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(
                Path(__file__).resolve().parents[1] / "face_model_assets" / "human" / "detectors" / "scrfd_2.5g_kps" / "metadata.json",
                bundle_dir / "metadata.json",
            )
            (bundle_dir / "detector.onnx").write_bytes(b"detector")
            yunet_dir = installer.bundle_dir("yunet_2026may")
            yunet_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(
                Path(__file__).resolve().parents[1] / "face_model_assets" / "human" / "detectors" / "yunet_2026may" / "metadata.json",
                yunet_dir / "metadata.json",
            )
            (yunet_dir / "detector.onnx").write_bytes(b"yunet")
            refreshed = {item.bundle_id: item for item in installer.inventory()}
            self.assertTrue(refreshed["scrfd_2.5g_kps"].installed)
            self.assertGreater(refreshed["scrfd_2.5g_kps"].size_bytes, 0)
            self.assertTrue(refreshed["yunet_2026may"].installed)
            self.assertGreater(refreshed["yunet_2026may"].size_bytes, 0)

    def test_face_model_installer_install_edge_installs_all_variants(self):
        with TemporaryDirectory() as tmp:
            fake_settings = SimpleNamespace(cache_dir=Path(tmp))
            installer = FaceModelInstaller(settings=fake_settings)
            installed: list[tuple[str, str]] = []

            def fake_download(url, sha256, filename, *, progress=None, cancel_check=None, progress_prefix):
                temp_root = Path(tmp) / f"download-{filename}"
                temp_root.mkdir(parents=True, exist_ok=True)
                payload = temp_root / filename
                payload.write_bytes(f"{url}\n{sha256}\n{filename}".encode("utf-8"))
                return payload, temp_root

            def fake_install(archive_path, *, member_name, bundle_id, output_name):
                _ = archive_path
                installed.append((bundle_id, member_name))
                bundle_dir = installer.bundle_dir(bundle_id)
                bundle_dir.mkdir(parents=True, exist_ok=True)
                (bundle_dir / output_name).write_text(member_name, encoding="utf-8")

            with patch.object(installer, "_download_to_temp", side_effect=fake_download), patch.object(
                installer,
                "_install_zip_member",
                side_effect=fake_install,
            ):
                result = installer.install_edge()

            self.assertEqual(("scrfd_500m_kps", "mobilefacenet_arcface"), result)
            self.assertEqual(
                [
                    ("scrfd_500m_kps", "det_500m.onnx"),
                    ("mobilefacenet_arcface", "w600k_mbf.onnx"),
                ],
                installed,
            )

    def test_face_model_installer_install_yunet_installs_all_variants(self):
        with TemporaryDirectory() as tmp:
            fake_settings = SimpleNamespace(cache_dir=Path(tmp))
            installer = FaceModelInstaller(settings=fake_settings)
            installed: list[str] = []

            def fake_download(url, sha256, filename, *, progress=None, cancel_check=None, progress_prefix):
                temp_root = Path(tmp) / f"download-{filename}"
                temp_root.mkdir(parents=True, exist_ok=True)
                payload = temp_root / filename
                payload.write_bytes(f"{url}\n{sha256}\n{filename}".encode("utf-8"))
                return payload, temp_root

            def fake_install(payload_path, *, bundle_id, output_name):
                installed.append(bundle_id)
                bundle_dir = installer.bundle_dir(bundle_id)
                bundle_dir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(payload_path, bundle_dir / output_name)

            with patch.object(installer, "_download_to_temp", side_effect=fake_download), patch.object(
                installer,
                "_install_payload_file",
                side_effect=fake_install,
            ):
                result = installer.install_yunet()

            self.assertEqual(("yunet_2026may", "yunet_2023mar", "yunet_2023mar_int8bq"), result)
            self.assertEqual(["yunet_2026may", "yunet_2023mar", "yunet_2023mar_int8bq"], installed)

    def test_face_model_installer_install_accuracy_and_max_accuracy(self):
        with TemporaryDirectory() as tmp:
            fake_settings = SimpleNamespace(cache_dir=Path(tmp))
            installer = FaceModelInstaller(settings=fake_settings)
            installed: list[str] = []

            def fake_download(url, sha256, filename, *, progress=None, cancel_check=None, progress_prefix):
                temp_root = Path(tmp) / f"download-{filename}"
                temp_root.mkdir(parents=True, exist_ok=True)
                payload = temp_root / filename
                payload.write_bytes(f"{url}\n{sha256}\n{filename}\n{progress_prefix}".encode("utf-8"))
                return payload, temp_root

            def fake_install(payload_path, *, bundle_id, output_name):
                installed.append(bundle_id)
                bundle_dir = installer.bundle_dir(bundle_id)
                bundle_dir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(payload_path, bundle_dir / output_name)

            with patch.object(installer, "_download_to_temp", side_effect=fake_download), patch.object(
                installer,
                "_install_payload_file",
                side_effect=fake_install,
            ):
                accuracy = installer.install_accuracy()
                max_accuracy = installer.install_max_accuracy()

            self.assertEqual(("scrfd_10g_kps", "adaface_r100"), accuracy)
            self.assertEqual(("scrfd_34gf_kps", "adaface_r100"), max_accuracy)
            self.assertEqual(
                ["scrfd_10g_kps", "adaface_r100", "scrfd_34gf_kps", "adaface_r100"],
                installed,
            )

    def test_face_model_installer_install_latest_gpu_and_sface(self):
        with TemporaryDirectory() as tmp:
            fake_settings = SimpleNamespace(cache_dir=Path(tmp))
            installer = FaceModelInstaller(settings=fake_settings)
            zip_installed: list[tuple[str, str]] = []
            payload_installed: list[str] = []

            def fake_download(url, sha256, filename, *, progress=None, cancel_check=None, progress_prefix):
                temp_root = Path(tmp) / f"download-{filename}"
                temp_root.mkdir(parents=True, exist_ok=True)
                payload = temp_root / filename
                payload.write_bytes(f"{url}\n{sha256}\n{filename}\n{progress_prefix}".encode("utf-8"))
                return payload, temp_root

            def fake_zip_install(archive_path, *, member_name, bundle_id, output_name):
                _ = archive_path
                zip_installed.append((bundle_id, member_name))
                bundle_dir = installer.bundle_dir(bundle_id)
                bundle_dir.mkdir(parents=True, exist_ok=True)
                (bundle_dir / output_name).write_text(member_name, encoding="utf-8")

            def fake_payload_install(payload_path, *, bundle_id, output_name):
                payload_installed.append(bundle_id)
                bundle_dir = installer.bundle_dir(bundle_id)
                bundle_dir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(payload_path, bundle_dir / output_name)

            with patch.object(installer, "_download_to_temp", side_effect=fake_download), patch.object(
                installer,
                "_install_zip_member",
                side_effect=fake_zip_install,
            ), patch.object(
                installer,
                "_install_payload_file",
                side_effect=fake_payload_install,
            ):
                latest_gpu = installer.install_latest_gpu()
                sface = installer.install_sface()

            self.assertEqual(("scrfd_10g_kps", "arcface_r100_glint360k"), latest_gpu)
            self.assertEqual(
                [
                    ("scrfd_10g_kps", "scrfd_10g_bnkps.onnx"),
                    ("arcface_r100_glint360k", "glintr100.onnx"),
                ],
                zip_installed,
            )
            self.assertEqual(("sface_2021dec", "sface_2021dec_int8bq"), sface)
            self.assertEqual(["sface_2021dec", "sface_2021dec_int8bq"], payload_installed)

    def test_human_service_uses_onnx_path_for_non_builtin_bundles(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            detector_dir = root / "human" / "detectors" / "yolo_face"
            embedder_dir = root / "human" / "embedders" / "arcface_small"
            detector_dir.mkdir(parents=True, exist_ok=True)
            embedder_dir.mkdir(parents=True, exist_ok=True)
            (detector_dir / "detector.onnx").write_bytes(b"detector")
            (embedder_dir / "embedder.onnx").write_bytes(b"embedder")
            (detector_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "display_name": "YOLO Face",
                        "backend_family": "yolo",
                        "supported_modes": ["human"],
                        "input_name": "images",
                        "input_size": [320, 320],
                        "output_boxes_name": "boxes",
                        "output_scores_name": "scores",
                        "mean": [0.0, 0.0, 0.0],
                        "std": [255.0, 255.0, 255.0],
                    }
                ),
                encoding="utf-8",
            )
            (embedder_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "display_name": "ArcFace Small",
                        "supported_modes": ["human"],
                        "input_name": "input",
                        "input_size": [112, 112],
                        "output_name": "embedding",
                        "mean": [127.5, 127.5, 127.5],
                        "std": [128.0, 128.0, 128.0],
                    }
                ),
                encoding="utf-8",
            )

            service = FaceIndexService(
                mode="human",
                model_root=root,
                detector_id="yolo_face",
                embedder_id="arcface_small",
                db_path=root / "human_faces.sqlite3",
            )

            self.assertNotIsInstance(service.detection_service, FaceDetectionService)
            self.assertEqual("YOLO Face", getattr(service.detection_service, "display_name", ""))
            self.assertEqual("ArcFace Small", getattr(service.embedding_service, "display_name", ""))

    def test_yunet_detector_backend_uses_opencv_facadetectoryn(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            detector_dir = root / "human" / "detectors" / "yunet_2026may"
            detector_dir.mkdir(parents=True, exist_ok=True)
            bundled_root = Path(__file__).resolve().parents[1] / "face_model_assets"
            shutil.copyfile(
                bundled_root / "human" / "detectors" / "yunet_2026may" / "metadata.json",
                detector_dir / "metadata.json",
            )
            (detector_dir / "detector.onnx").write_bytes(b"detector")
            image_path = root / "face.jpg"
            Image.new("RGB", (128, 128), "white").save(image_path)

            fake_detector = SimpleNamespace(
                setInputSize=lambda *_args, **_kwargs: None,
                detect=lambda _image: (
                    1,
                    np.asarray(
                        [[16.0, 24.0, 48.0, 48.0] + [0.0] * 10 + [0.97]],
                        dtype=np.float32,
                    ),
                ),
            )

            with patch("app.services.face_search.face_model_runtime_root_dir", return_value=root), patch(
                "app.services.face_search.cv2.FaceDetectorYN.create",
                return_value=fake_detector,
            ):
                service = FaceIndexService(
                    discovery_service=FakeDiscoveryService([str(image_path)]),
                    embedding_service=FakeFaceEmbeddingService(),
                    mode="human",
                    model_root="",
                    detector_id="yunet_2026may",
                    embedder_id=BUILTIN_HUMAN_EMBEDDER_ID,
                    db_path=root / "faces.db",
                    reset_db=True,
                )
                metrics = service.index_paths([str(image_path)])

            self.assertEqual("yunet_2026may", service.detector_id)
            self.assertEqual("yunet", service.detection_service.backend_family)
            self.assertEqual(1, metrics["faces_indexed"])

    def test_animal_face_service_reports_not_ready_without_model_root(self):
        with TemporaryDirectory() as tmp:
            service = FaceIndexService(mode="dog", animal_model_root="", db_path=Path(tmp) / "dog_faces.sqlite3")
            self.assertFalse(service.is_ready())
            self.assertIn("Animal face model root is not configured", service.readiness_message())

    def test_face_profiles_persist_and_respect_candidate_scope(self):
        with TemporaryDirectory() as tmp:
            image_a = Path(tmp) / "alice_a.jpg"
            image_b = Path(tmp) / "alice_b.jpg"
            Image.new("RGB", (32, 32), (10, 20, 30)).save(image_a)
            Image.new("RGB", (32, 32), (12, 20, 30)).save(image_b)
            service = FaceIndexService(
                detection_service=FakeFaceDetectionService(),
                embedding_service=FakeFaceEmbeddingService(),
                db_path=Path(tmp) / "faces.sqlite3",
            )
            for image_path in [image_a, image_b]:
                service.save_face_records(
                    [
                        FaceIndexRecord(
                            image_path=str(image_path),
                            face_index=0,
                            face_bbox=(0, 0, 16, 16),
                            face_confidence=0.99,
                            embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                        )
                    ],
                    mtime_ns=image_path.stat().st_mtime_ns,
                    file_size=image_path.stat().st_size,
                    assess_quality=False,
                )
            service.label_indexed_faces("Alice", [(str(image_a), 0), (str(image_b), 0)], similarity_threshold=0.5)
            pending = service.load_pending_face_labels(include_tiny_faces=True)
            service.accept_pending_face_labels([assignment.proposal_id for assignment in pending])
            service.save_person_profile(
                "Alice",
                notes="Team lead",
                tags=["family", "priority"],
                cover_face_ref=(str(image_a), 0),
            )
            profiles = service.load_person_profiles(candidate_paths=[str(image_a)], limit=20)
            self.assertEqual(1, len(profiles))
            self.assertEqual("Alice", profiles[0].person_name)
            self.assertEqual("Team lead", profiles[0].notes)
            self.assertEqual(("family", "priority"), profiles[0].tags)
            self.assertEqual(1, profiles[0].visible_face_count)
            scoped_results = service.search_by_person_name("Alice", candidate_paths=[str(image_b)])
            self.assertEqual([str(image_b)], [result.image_path for result in scoped_results])
            del service
            gc.collect()

    def test_load_person_profiles_uses_scoped_counts_without_loading_examples(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_a = root / "a.jpg"
            Image.new("RGB", (80, 60), (20, 30, 40)).save(image_a)
            service = FaceIndexService(
                detection_service=FakeFaceDetectionService(),
                embedding_service=FakeFaceEmbeddingService(),
                db_path=root / "faces.sqlite3",
            )
            service.save_face_records(
                [
                    FaceIndexRecord(
                        image_path=str(image_a),
                        face_index=0,
                        face_bbox=(0, 0, 16, 16),
                        face_confidence=0.99,
                        embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                    )
                ],
                mtime_ns=image_a.stat().st_mtime_ns,
                file_size=image_a.stat().st_size,
                assess_quality=False,
            )
            service.label_indexed_faces("Alice", [(str(image_a), 0)], similarity_threshold=0.5)
            pending = service.load_pending_face_labels(include_tiny_faces=True)
            service.accept_pending_face_labels([assignment.proposal_id for assignment in pending])
            with patch.object(service, "load_profile_examples", side_effect=AssertionError("load_profile_examples should not be called")):
                profiles = service.load_person_profiles(candidate_paths=[str(image_a)], limit=20)
            self.assertEqual(1, len(profiles))
            self.assertEqual(1, profiles[0].visible_face_count)

    def test_runtime_policy_uses_directml_when_available_without_cuda(self):
        service = RuntimeCapabilityService()
        service._cached = RuntimeCapabilities(
            torch_version="2.2.2+cpu",
            torch_cuda_build=False,
            torch_cuda_available=False,
            onnx_available=True,
            onnx_version="1.24.4",
            onnx_providers=("DmlExecutionProvider", "CPUExecutionProvider"),
        )
        with patch.object(service, "_onnx_provider_usable", return_value=True):
            policy = service.select_policy("auto")
        self.assertEqual("directml", policy.effective_mode)
        self.assertEqual("cpu", policy.torch_device)
        self.assertEqual("DmlExecutionProvider", policy.onnx_provider)

    def test_runtime_policy_uses_onnx_cuda_when_torch_cuda_is_unavailable(self):
        service = RuntimeCapabilityService()
        service._cached = RuntimeCapabilities(
            torch_version="2.2.2+cpu",
            torch_cuda_build=False,
            torch_cuda_available=False,
            onnx_available=True,
            onnx_version="1.24.4",
            onnx_providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
        )
        with patch.object(service, "_onnx_provider_usable", return_value=True):
            policy = service.select_policy("auto")
        self.assertEqual("cuda", policy.effective_mode)
        self.assertEqual("cpu", policy.torch_device)
        self.assertEqual("CUDAExecutionProvider", policy.onnx_provider)
        self.assertTrue(policy.uses_onnx_cuda)

    def test_runtime_policy_rejects_unusable_onnx_cuda_provider(self):
        service = RuntimeCapabilityService()
        service._cached = RuntimeCapabilities(
            torch_version="2.2.2+cpu",
            torch_cuda_build=False,
            torch_cuda_available=False,
            onnx_available=True,
            onnx_version="1.24.4",
            onnx_providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
        )
        with patch.object(service, "_onnx_provider_usable", return_value=False):
            policy = service.select_policy("auto")
        self.assertEqual("cpu", policy.effective_mode)
        self.assertEqual("CPUExecutionProvider", policy.onnx_provider)
        self.assertFalse(policy.uses_onnx_cuda)

    def test_runtime_policy_falls_back_to_cpu_when_cuda_requested_but_unavailable(self):
        service = RuntimeCapabilityService()
        service._cached = RuntimeCapabilities(
            torch_version="2.2.2+cpu",
            torch_cuda_build=False,
            torch_cuda_available=False,
            onnx_available=True,
            onnx_version="1.24.4",
            onnx_providers=("CPUExecutionProvider",),
        )
        policy = service.select_policy("cuda")
        self.assertEqual("cpu", policy.effective_mode)
        self.assertEqual("CPUExecutionProvider", policy.onnx_provider)
        self.assertIn("CUDA requested", policy.reason)

    def test_onnx_model_service_session_from_asset_falls_back_to_cpu(self):
        calls: list[list[str]] = []

        class _FakeOrt:
            def InferenceSession(self, path, providers):
                calls.append(list(providers))
                if providers and providers[0] == "CUDAExecutionProvider":
                    raise RuntimeError("cuda init failed")
                return {"path": path, "providers": list(providers)}

        with TemporaryDirectory() as tmp:
            asset_path = Path(tmp) / "model.onnx"
            asset_path.write_bytes(b"onnx")
            service = OnnxModelService(
                execution_policy=ExecutionPolicy(
                    preferred_mode="cuda",
                    effective_mode="cuda",
                    torch_device="cpu",
                    onnx_provider="CUDAExecutionProvider",
                ),
                runtime_service=RuntimeCapabilityService(),
            )
            with patch("app.services.onnx_models.ort", _FakeOrt()), patch(
                "app.services.onnx_models.preload_onnx_cuda_runtime_libraries",
                return_value=True,
            ):
                session = service.session_from_asset("fast_preview", asset_path, "CUDAExecutionProvider")
        self.assertEqual([["CUDAExecutionProvider", "CPUExecutionProvider"], ["CPUExecutionProvider"]], calls)
        self.assertEqual(["CPUExecutionProvider"], session["providers"])

    def test_face_onnx_session_creation_falls_back_to_cpu(self):
        calls: list[list[str]] = []

        class _FakeOrt:
            def InferenceSession(self, path, providers):
                calls.append(list(providers))
                if providers and providers[0] == "CUDAExecutionProvider":
                    raise RuntimeError("cuda init failed")
                return {"path": path, "providers": list(providers)}

        with patch.object(face_search_module, "ort", _FakeOrt()):
            session = face_search_module._create_onnx_session(Path("/tmp/model.onnx"), ["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.assertEqual([["CUDAExecutionProvider", "CPUExecutionProvider"], ["CPUExecutionProvider"]], calls)
        self.assertEqual(["CPUExecutionProvider"], session["providers"])

    def test_label_indexed_faces_creates_pending_queue_before_acceptance(self):
        with TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "pending.jpg"
            Image.new("RGB", (64, 64), (12, 24, 36)).save(image_path)
            service = FaceIndexService(
                detection_service=FakeFaceDetectionService(),
                embedding_service=FakeFaceEmbeddingService(),
                db_path=Path(tmp) / "faces.sqlite3",
            )
            service.save_face_records(
                [
                    FaceIndexRecord(
                        image_path=str(image_path),
                        face_index=0,
                        face_bbox=(0, 0, 32, 32),
                        face_confidence=0.99,
                        embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                    )
                ],
                mtime_ns=image_path.stat().st_mtime_ns,
                file_size=image_path.stat().st_size,
                assess_quality=False,
            )
            service.label_indexed_faces("Alice", [(str(image_path), 0)], similarity_threshold=0.5)
            self.assertEqual([], service.list_face_labels())
            pending = service.load_pending_face_labels()
            self.assertEqual(1, len(pending))
            self.assertEqual("manual_selected_faces", pending[0].source)
            accepted = service.accept_pending_face_labels([pending[0].proposal_id])
            self.assertEqual(1, len(accepted))
            labels = service.list_face_labels()
            self.assertEqual(1, len(labels))
            self.assertEqual("Alice", labels[0].person_name)

    def test_label_indexed_faces_immediately_writes_saved_labels_and_clears_pending(self):
        with TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "immediate.jpg"
            Image.new("RGB", (64, 64), (12, 24, 36)).save(image_path)
            service = FaceIndexService(
                detection_service=FakeFaceDetectionService(),
                embedding_service=FakeFaceEmbeddingService(),
                db_path=Path(tmp) / "faces.sqlite3",
            )
            service.save_face_records(
                [
                    FaceIndexRecord(
                        image_path=str(image_path),
                        face_index=0,
                        face_bbox=(0, 0, 32, 32),
                        face_confidence=0.99,
                        embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                    )
                ],
                mtime_ns=image_path.stat().st_mtime_ns,
                file_size=image_path.stat().st_size,
                assess_quality=False,
            )
            service.label_indexed_faces("Bob", [(str(image_path), 0)], similarity_threshold=0.5)
            self.assertEqual(1, len(service.load_pending_face_labels()))

            person = service.label_indexed_faces_immediately("Alice", [(str(image_path), 0)], similarity_threshold=0.5)

            self.assertEqual("Alice", person.person_name)
            self.assertEqual([], service.load_pending_face_labels())
            labels = service.list_face_labels()
            self.assertEqual(1, len(labels))
            self.assertEqual("Alice", labels[0].person_name)
            refreshed = service.load_face_record(str(image_path), 0)
            self.assertIsNotNone(refreshed)
            self.assertEqual("Alice", refreshed.person_name)

    def test_accept_pending_face_labels_batch_can_be_undone(self):
        with TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "undo_label.jpg"
            Image.new("RGB", (64, 64), (20, 30, 40)).save(image_path)
            service = FaceIndexService(
                detection_service=FakeFaceDetectionService(),
                embedding_service=FakeFaceEmbeddingService(),
                db_path=Path(tmp) / "faces.sqlite3",
            )
            service.save_face_records(
                [
                    FaceIndexRecord(
                        image_path=str(image_path),
                        face_index=0,
                        face_bbox=(0, 0, 32, 32),
                        face_confidence=0.99,
                        embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                    )
                ],
                mtime_ns=image_path.stat().st_mtime_ns,
                file_size=image_path.stat().st_size,
            )
            service._queue_face_label_assignments(
                [
                    FaceLabelAssignment(
                        person_name="Bob",
                        image_path=str(image_path),
                        face_index=0,
                        face_bbox=(0, 0, 32, 32),
                        confidence=0.91,
                        source="seed",
                        pending=True,
                    )
                ]
            )
            seed_pending = service.load_pending_face_labels()
            service.accept_pending_face_labels([seed_pending[0].proposal_id])
            self.assertEqual("Bob", service.list_face_labels()[0].person_name)

            service._queue_face_label_assignments(
                [
                    FaceLabelAssignment(
                        person_name="Alice",
                        image_path=str(image_path),
                        face_index=0,
                        face_bbox=(0, 0, 32, 32),
                        confidence=0.97,
                        source="cluster_suggestion",
                        pending=True,
                    )
                ]
            )
            pending = service.load_pending_face_labels()
            self.assertEqual(1, len(pending))
            batch = service.accept_pending_face_labels_batch([pending[0].proposal_id])
            self.assertEqual(1, len(batch.accepted))
            self.assertEqual(1, len(batch.previous_labels))
            self.assertEqual("Alice", service.list_face_labels()[0].person_name)

            restored = service.undo_last_pending_face_acceptance()
            self.assertEqual(1, len(restored))
            self.assertEqual("Bob", service.list_face_labels()[0].person_name)
            pending_after_undo = service.load_pending_face_labels()
            self.assertEqual(1, len(pending_after_undo))
            self.assertEqual("Alice", pending_after_undo[0].person_name)

    def test_face_similarity_search_uses_explicit_faiss_sidecar_build(self):
        with TemporaryDirectory() as tmp:
            image_a = Path(tmp) / "a.jpg"
            image_b = Path(tmp) / "b.jpg"
            Image.new("RGB", (64, 64), (20, 30, 40)).save(image_a)
            Image.new("RGB", (64, 64), (30, 40, 50)).save(image_b)
            service = FaceIndexService(
                detection_service=FakeFaceDetectionService(),
                embedding_service=FakeFaceEmbeddingService(),
                db_path=Path(tmp) / "faces.sqlite3",
            )
            service.save_face_records(
                [
                    FaceIndexRecord(
                        image_path=str(image_a),
                        face_index=0,
                        face_bbox=(0, 0, 32, 32),
                        face_confidence=0.99,
                        embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                    )
                ],
                mtime_ns=image_a.stat().st_mtime_ns,
                file_size=image_a.stat().st_size,
                assess_quality=False,
            )
            service.save_face_records(
                [
                    FaceIndexRecord(
                        image_path=str(image_b),
                        face_index=0,
                        face_bbox=(0, 0, 32, 32),
                        face_confidence=0.98,
                        embedding=np.array([0.98, 0.02, 0.0], dtype=np.float32),
                    )
                ],
                mtime_ns=image_b.stat().st_mtime_ns,
                file_size=image_b.stat().st_size,
                assess_quality=False,
            )
            results = service.search_similar_face(str(image_a), 0, top_k=5, min_score=0.1)
            self.assertEqual([str(image_b)], [result.image_path for result in results])
            self.assertFalse(Path(f"{service.db_path}.faiss").exists())
            service.build_ann_index()
            if face_search_module.faiss is not None:
                self.assertTrue(Path(f"{service.db_path}.faiss").exists())
                self.assertTrue(Path(f"{service.db_path}.faiss.json").exists())

    def test_rejected_face_label_correction_suppresses_future_auto_propagation(self):
        with TemporaryDirectory() as tmp:
            image_a = Path(tmp) / "alice_a.jpg"
            image_b = Path(tmp) / "alice_b.jpg"
            Image.new("RGB", (64, 64), (20, 30, 40)).save(image_a)
            Image.new("RGB", (64, 64), (22, 32, 42)).save(image_b)
            service = FaceIndexService(
                detection_service=FakeFaceDetectionService(),
                embedding_service=FakeFaceEmbeddingService(),
                db_path=Path(tmp) / "faces.sqlite3",
            )
            for path, embedding in (
                (image_a, np.array([1.0, 0.0, 0.0], dtype=np.float32)),
                (image_b, np.array([0.99, 0.01, 0.0], dtype=np.float32)),
            ):
                service.save_face_records(
                    [FaceIndexRecord(str(path), 0, (0, 0, 36, 36), 0.99, embedding)],
                    mtime_ns=path.stat().st_mtime_ns,
                    file_size=path.stat().st_size,
                    assess_quality=False,
                )
            service.label_indexed_faces("Alice", [(str(image_a), 0)], similarity_threshold=0.5)
            service.accept_pending_face_labels([item.proposal_id for item in service.load_pending_face_labels(include_tiny_faces=True)])

            queued = service.auto_propagate_labels(include_tiny_faces=True)
            self.assertEqual([(str(image_b), 0)], [(item.image_path, item.face_index) for item in queued])
            self.assertEqual(1, service.reject_pending_face_labels([queued[0].proposal_id]))
            self.assertEqual([], service.auto_propagate_labels(include_tiny_faces=True))
            self.assertEqual([], service.load_pending_face_labels(include_tiny_faces=True))
            self.assertEqual(1, len(service.list_rejected_face_label_corrections()))

            service.clear_face_label_corrections()
            queued_again = service.auto_propagate_labels(include_tiny_faces=True)
            self.assertEqual([(str(image_b), 0)], [(item.image_path, item.face_index) for item in queued_again])

    def test_hidden_people_and_faces_persist_and_are_excluded_by_default(self):
        with TemporaryDirectory() as tmp:
            image_a = Path(tmp) / "alice.jpg"
            image_b = Path(tmp) / "bob.jpg"
            Image.new("RGB", (64, 64), (20, 30, 40)).save(image_a)
            Image.new("RGB", (64, 64), (30, 40, 50)).save(image_b)
            db_path = Path(tmp) / "faces.sqlite3"
            service = FaceIndexService(
                detection_service=FakeFaceDetectionService(),
                embedding_service=FakeFaceEmbeddingService(),
                db_path=db_path,
            )
            for path, embedding in (
                (image_a, np.array([1.0, 0.0, 0.0], dtype=np.float32)),
                (image_b, np.array([0.0, 1.0, 0.0], dtype=np.float32)),
            ):
                service.save_face_records(
                    [FaceIndexRecord(str(path), 0, (0, 0, 36, 36), 0.99, embedding)],
                    mtime_ns=path.stat().st_mtime_ns,
                    file_size=path.stat().st_size,
                    assess_quality=False,
                )
            service.label_indexed_faces("Alice", [(str(image_a), 0)], similarity_threshold=0.5)
            service.label_indexed_faces("Bob", [(str(image_b), 0)], similarity_threshold=0.5)
            service.accept_pending_face_labels([item.proposal_id for item in service.load_pending_face_labels(include_tiny_faces=True)])
            service.hide_person("Alice")
            service.hide_face(str(image_b), 0)

            self.assertEqual([], service.load_indexed_faces(include_tiny_faces=True))
            hidden_records = service.load_indexed_faces(include_tiny_faces=True, include_hidden=True)
            self.assertEqual({str(image_a), str(image_b)}, {record.image_path for record in hidden_records})
            self.assertTrue(all(record.hidden for record in hidden_records))
            self.assertEqual([], service.load_person_profiles(include_tiny_faces=True))
            self.assertEqual(["Alice", "Bob"], [profile.person_name for profile in service.load_person_profiles(include_tiny_faces=True, include_hidden=True)])

            reloaded = FaceIndexService(
                detection_service=FakeFaceDetectionService(),
                embedding_service=FakeFaceEmbeddingService(),
                db_path=db_path,
            )
            self.assertEqual([], reloaded.load_indexed_faces(include_tiny_faces=True))
            reloaded.unhide_person("Alice")
            reloaded.unhide_face(str(image_b), 0)
            self.assertEqual({str(image_a), str(image_b)}, {record.image_path for record in reloaded.load_indexed_faces(include_tiny_faces=True)})

    def test_person_profile_favorite_cover_birth_date_and_age_persist(self):
        with TemporaryDirectory() as tmp:
            image_a = Path(tmp) / "alice.jpg"
            image_b = Path(tmp) / "bob.jpg"
            Image.new("RGB", (64, 64), (20, 30, 40)).save(image_a)
            Image.new("RGB", (64, 64), (30, 40, 50)).save(image_b)
            service = FaceIndexService(
                detection_service=FakeFaceDetectionService(),
                embedding_service=FakeFaceEmbeddingService(),
                db_path=Path(tmp) / "faces.sqlite3",
            )
            for path, name, embedding in (
                (image_a, "Alice", np.array([1.0, 0.0, 0.0], dtype=np.float32)),
                (image_b, "Bob", np.array([0.0, 1.0, 0.0], dtype=np.float32)),
            ):
                service.save_face_records(
                    [FaceIndexRecord(str(path), 0, (0, 0, 36, 36), 0.99, embedding)],
                    mtime_ns=path.stat().st_mtime_ns,
                    file_size=path.stat().st_size,
                    assess_quality=False,
                )
                service.label_indexed_faces(name, [(str(path), 0)], similarity_threshold=0.5)
            service.accept_pending_face_labels([item.proposal_id for item in service.load_pending_face_labels(include_tiny_faces=True)])
            service.save_person_profile(
                "Bob",
                notes="regular",
                tags=[],
                cover_face_ref=(str(image_b), 0),
                favorite=False,
                birth_date="1990-01-01",
            )
            service.save_person_profile(
                "Alice",
                notes="priority",
                tags=["family"],
                cover_face_ref=(str(image_a), 0),
                favorite=True,
                birth_date="2000-01-01",
            )

            profiles = service.load_person_profiles(include_tiny_faces=True)
            self.assertEqual(["Alice", "Bob"], [profile.person_name for profile in profiles])
            self.assertTrue(profiles[0].favorite)
            self.assertEqual("2000-01-01", profiles[0].birth_date)
            self.assertEqual(str(image_a), profiles[0].cover_image_path)
            self.assertEqual(0, profiles[0].cover_face_index)
            self.assertIsNotNone(service.age_at_photo("Alice", str(image_a)))
            self.assertIsNone(service.age_at_photo("Alice", str(Path(tmp) / "missing.jpg")))

    def test_identity_export_import_round_trip_restores_profiles_labels_and_corrections(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_a = root / "alice_a.jpg"
            image_b = root / "alice_b.jpg"
            image_c = root / "hidden_bob.jpg"
            for path in (image_a, image_b, image_c):
                Image.new("RGB", (64, 64), (20, 30, 40)).save(path)
            source = FaceIndexService(
                detection_service=FakeFaceDetectionService(),
                embedding_service=FakeFaceEmbeddingService(),
                db_path=root / "source.sqlite3",
            )
            for path, embedding in (
                (image_a, np.array([1.0, 0.0, 0.0], dtype=np.float32)),
                (image_b, np.array([0.98, 0.02, 0.0], dtype=np.float32)),
                (image_c, np.array([0.0, 1.0, 0.0], dtype=np.float32)),
            ):
                source.save_face_records(
                    [FaceIndexRecord(str(path), 0, (0, 0, 36, 36), 0.99, embedding)],
                    mtime_ns=path.stat().st_mtime_ns,
                    file_size=path.stat().st_size,
                    assess_quality=False,
                )
            source.label_indexed_faces("Alice", [(str(image_a), 0)], similarity_threshold=0.81)
            source.label_indexed_faces("Bob", [(str(image_c), 0)], similarity_threshold=0.7)
            source.accept_pending_face_labels([item.proposal_id for item in source.load_pending_face_labels(include_tiny_faces=True)])
            source.save_person_profile(
                "Alice",
                notes="round trip",
                tags=["vip"],
                cover_face_ref=(str(image_a), 0),
                favorite=True,
                birth_date="1999-05-01",
            )
            source.hide_person("Bob")
            source.auto_propagate_labels(include_tiny_faces=True)
            pending = [item for item in source.load_pending_face_labels(include_tiny_faces=True) if item.image_path == str(image_b)]
            source.reject_pending_face_labels([pending[0].proposal_id])

            payload = source.export_identity_data()
            imported = FaceIndexService(
                detection_service=FakeFaceDetectionService(),
                embedding_service=FakeFaceEmbeddingService(),
                db_path=root / "imported.sqlite3",
            )
            imported.import_identity_data(payload)

            alice_profile = imported.load_person_profiles(include_tiny_faces=True, include_hidden=True)[0]
            self.assertEqual("Alice", alice_profile.person_name)
            self.assertTrue(alice_profile.favorite)
            self.assertEqual("1999-05-01", alice_profile.birth_date)
            self.assertEqual(str(image_a), alice_profile.cover_image_path)
            self.assertAlmostEqual(0.81, imported.find_person_prototype("Alice").similarity_threshold)
            self.assertEqual([str(image_a), str(image_b)], [result.image_path for result in imported.search_by_person_name("Alice", min_score=0.1, include_tiny_faces=True)])
            self.assertTrue([profile for profile in imported.load_person_profiles(include_hidden=True) if profile.person_name == "Bob" and profile.hidden])
            self.assertEqual(1, len(imported.list_rejected_face_label_corrections()))

    def test_identity_import_preview_merge_replace_and_cancel_semantics(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            existing_image = root / "existing.jpg"
            imported_image = root / "imported.jpg"
            unrelated_image = root / "unrelated.jpg"
            Image.new("RGB", (64, 64), (20, 30, 40)).save(existing_image)
            Image.new("RGB", (64, 64), (30, 40, 50)).save(imported_image)
            Image.new("RGB", (64, 64), (40, 50, 60)).save(unrelated_image)

            def _seed_service(db_name: str) -> FaceIndexService:
                service = FaceIndexService(
                    detection_service=FakeFaceDetectionService(),
                    embedding_service=FakeFaceEmbeddingService(),
                    db_path=root / db_name,
                )
                service.save_face_records(
                    [FaceIndexRecord(str(existing_image), 0, (0, 0, 36, 36), 0.99, np.array([1.0, 0.0], dtype=np.float32))],
                    mtime_ns=existing_image.stat().st_mtime_ns,
                    file_size=existing_image.stat().st_size,
                    assess_quality=False,
                )
                service.save_face_records(
                    [FaceIndexRecord(str(unrelated_image), 0, (0, 0, 36, 36), 0.98, np.array([0.5, 0.5], dtype=np.float32))],
                    mtime_ns=unrelated_image.stat().st_mtime_ns,
                    file_size=unrelated_image.stat().st_size,
                    assess_quality=False,
                )
                service.label_indexed_faces("Alice", [(str(existing_image), 0)], similarity_threshold=0.5)
                service.label_indexed_faces("Charlie", [(str(unrelated_image), 0)], similarity_threshold=0.5)
                service.accept_pending_face_labels([item.proposal_id for item in service.load_pending_face_labels(include_tiny_faces=True)])
                service.save_person_profile("Alice", notes="old")
                service.save_person_profile("Charlie", notes="unrelated")
                return service

            payload = {
                "version": 1,
                "identities": [
                    {
                        "person_name": "Alice",
                        "notes": "new",
                        "prototype_embedding": [1.0, 0.0],
                        "similarity_threshold": 0.8,
                        "example_count": 1,
                        "prototype_refs": [{"image_path": str(imported_image), "face_index": 0, "pinned": True}],
                    },
                    {
                        "person_name": "Bob",
                        "hidden": True,
                        "prototype_embedding": [0.0, 1.0],
                        "similarity_threshold": 0.7,
                        "example_count": 1,
                    },
                ],
                "face_index": [
                    {
                        "image_path": str(imported_image),
                        "face_index": 0,
                        "bbox": [0, 0, 40, 40],
                        "face_confidence": 0.95,
                        "embedding": [0.0, 1.0],
                        "hidden": True,
                    }
                ],
                "labels": [{"image_path": str(existing_image), "face_index": 0, "person_name": "Bob", "confidence": 0.77}],
                "corrections": [{"image_path": str(imported_image), "face_index": 0, "person_name": "Alice", "source": "test"}],
            }

            merge_service = _seed_service("merge.sqlite3")
            preview = merge_service.preview_identity_import(payload)
            self.assertEqual(2, preview["counts"]["identities"])
            self.assertEqual(1, preview["counts"]["labels"])
            self.assertEqual(3, preview["counts"]["prototypes"])
            self.assertEqual(2, preview["counts"]["hidden_items"])
            self.assertEqual(1, preview["counts"]["corrections"])
            self.assertGreaterEqual(preview["conflict_count"], 2)

            before_cancel = [profile.person_name for profile in merge_service.load_person_profiles(include_hidden=True)]
            self.assertEqual(["Alice", "Charlie"], before_cancel)

            merge_service.import_identity_data(payload, mode="merge")
            merged_profiles = {profile.person_name: profile for profile in merge_service.load_person_profiles(include_hidden=True)}
            self.assertIn("Charlie", merged_profiles)
            self.assertEqual("new", merged_profiles["Alice"].notes)
            self.assertTrue(merged_profiles["Bob"].hidden)

            replace_service = _seed_service("replace.sqlite3")
            replace_service.import_identity_data(payload, mode="replace")
            replaced_profiles = {profile.person_name: profile for profile in replace_service.load_person_profiles(include_hidden=True)}
            self.assertEqual({"Alice", "Bob"}, set(replaced_profiles))
            self.assertNotIn("Charlie", replaced_profiles)

    def test_face_search_query_grammar_filters_faces_people_unknown_and_hidden(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            group = root / "group.jpg"
            unknown = root / "unknown.jpg"
            no_faces = root / "no_faces.jpg"
            for path in (group, unknown, no_faces):
                Image.new("RGB", (80, 80), (20, 30, 40)).save(path)
            service = FaceIndexService(
                detection_service=FakeFaceDetectionService(),
                embedding_service=FakeFaceEmbeddingService(),
                db_path=root / "faces.sqlite3",
            )
            service.save_face_records(
                [
                    FaceIndexRecord(str(group), 0, (0, 0, 36, 36), 0.99, np.array([1.0, 0.0, 0.0], dtype=np.float32)),
                    FaceIndexRecord(str(group), 1, (40, 0, 76, 36), 0.98, np.array([0.0, 1.0, 0.0], dtype=np.float32)),
                ],
                mtime_ns=group.stat().st_mtime_ns,
                file_size=group.stat().st_size,
                assess_quality=False,
            )
            service.save_face_records(
                [FaceIndexRecord(str(unknown), 0, (0, 0, 36, 36), 0.97, np.array([0.0, 0.0, 1.0], dtype=np.float32))],
                mtime_ns=unknown.stat().st_mtime_ns,
                file_size=unknown.stat().st_size,
                assess_quality=False,
            )
            service._replace_image_records(str(no_faces), [], mtime_ns=no_faces.stat().st_mtime_ns, file_size=no_faces.stat().st_size)
            service.label_indexed_faces("Alice", [(str(group), 0)], similarity_threshold=0.5)
            service.label_indexed_faces("Bob", [(str(group), 1)], similarity_threshold=0.5)
            service.accept_pending_face_labels([item.proposal_id for item in service.load_pending_face_labels(include_tiny_faces=True)])
            service.hide_face(str(group), 1)

            self.assertEqual([(str(group), 0), (str(unknown), 0)], [(record.image_path, record.face_index) for record in service.search_face_query("faces:true", include_tiny_faces=True)])
            self.assertEqual([(str(no_faces), -1)], [(record.image_path, record.face_index) for record in service.search_face_query("faces:false", include_tiny_faces=True)])
            self.assertEqual([(str(group), 0), (str(group), 1)], [(record.image_path, record.face_index) for record in service.search_face_query("faces:2", include_tiny_faces=True, include_hidden=True)])
            self.assertEqual([(str(group), 0)], [(record.image_path, record.face_index) for record in service.search_face_query('people:"Ali"', include_tiny_faces=True)])
            self.assertEqual([(str(group), 0), (str(group), 1)], [(record.image_path, record.face_index) for record in service.search_face_query('person:"Alice&Bob"', include_tiny_faces=True, include_hidden=True)])
            self.assertEqual([(str(group), 0), (str(group), 1)], [(record.image_path, record.face_index) for record in service.search_face_query('person:"Alice|Bob"', include_tiny_faces=True, include_hidden=True)])
            self.assertEqual([(str(unknown), 0)], [(record.image_path, record.face_index) for record in service.search_face_query("unknown:true", include_tiny_faces=True)])
            self.assertEqual([(str(group), 1)], [(record.image_path, record.face_index) for record in service.search_face_query("hidden:true", include_tiny_faces=True)])

    def test_face_privacy_disable_and_purge_remove_face_owned_state(self):
        class CountingDetectionService(FakeFaceDetectionService):
            def __init__(self):
                self.detect_calls = 0

            def detect_faces(self, image_path):
                self.detect_calls += 1
                return super().detect_faces(image_path)

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "face.jpg"
            Image.new("RGB", (64, 64), (20, 30, 40)).save(image_path)
            detector = CountingDetectionService()
            service = FaceIndexService(
                detection_service=detector,
                embedding_service=FakeFaceEmbeddingService(),
                db_path=root / "faces.sqlite3",
            )
            service.set_face_recognition_enabled(False)
            metrics = service.index_paths([str(image_path)])
            self.assertEqual(0, detector.detect_calls)
            self.assertEqual(1, metrics["skipped_recognition_disabled"])
            service.set_face_recognition_enabled(True)
            service.configure_recognition_options(prototype_quality_min="reject", search_quality_min="reject")
            service.index_paths([str(image_path)])
            self.assertEqual(1, detector.detect_calls)
            service.label_indexed_faces("Alice", [(str(image_path), 0)], similarity_threshold=0.5)
            service.accept_pending_face_labels([item.proposal_id for item in service.load_pending_face_labels(include_tiny_faces=True)])
            Path(f"{service.db_path}.faiss").write_bytes(b"ann")
            Path(f"{service.db_path}.faiss.json").write_text("{}", encoding="utf-8")
            model_cache = root / "runtime_models" / "human"
            model_cache.mkdir(parents=True)
            (model_cache / "model.bin").write_bytes(b"model")

            with patch("app.services.face_search.face_model_runtime_root_dir", return_value=root / "runtime_models"):
                report = service.prepare_face_data_purge()
                self.assertGreater(report["table_counts"]["face_index"], 0)
                self.assertTrue(report["ann_files"])
                self.assertEqual([str(model_cache)], report["model_cache_targets"])
                purge_result = service.purge_face_data(confirm=True)
            self.assertFalse(Path(f"{service.db_path}.faiss").exists())
            self.assertFalse(Path(f"{service.db_path}.faiss.json").exists())
            self.assertFalse(model_cache.exists())
            self.assertEqual([], service.load_indexed_faces(include_tiny_faces=True, include_hidden=True))
            self.assertTrue(purge_result["removed_ann_files"])

    def test_face_service_records_destructive_action_audit(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "face.jpg"
            Image.new("RGB", (64, 64), (20, 30, 40)).save(image_path)
            service = FaceIndexService(
                detection_service=FakeFaceDetectionService(),
                embedding_service=FakeFaceEmbeddingService(),
                db_path=root / "faces.sqlite3",
            )
            service.save_face_records(
                [
                    FaceIndexRecord(str(image_path), 0, (0, 0, 28, 36), 0.99, np.array([1.0, 0.0, 0.0], dtype=np.float32)),
                    FaceIndexRecord(str(image_path), 1, (32, 0, 60, 36), 0.98, np.array([0.0, 1.0, 0.0], dtype=np.float32)),
                ],
                mtime_ns=image_path.stat().st_mtime_ns,
                file_size=image_path.stat().st_size,
                assess_quality=False,
            )
            service.label_indexed_faces("Alice", [(str(image_path), 0)], similarity_threshold=0.5)
            service.accept_pending_face_labels([item.proposal_id for item in service.load_pending_face_labels(include_tiny_faces=True)])
            service.save_person_profile("Alice", notes="audit", favorite=True, hidden=False)
            service.hide_person("Alice")
            service.hide_face(str(image_path), 0)
            service.set_face_recognition_enabled(False)
            service.label_indexed_faces("Bob", [(str(image_path), 1)], similarity_threshold=0.5)
            pending = service.load_pending_face_labels(include_tiny_faces=True, include_hidden=True)
            service.reject_pending_face_labels([item.proposal_id for item in pending])
            service.purge_face_data(confirm=True)

            events = service.load_face_action_audit(limit=20)
            actions = [event.action for event in events]

            for expected in (
                "purge_face_data",
                "reject_pending_face_labels",
                "set_face_recognition_enabled",
                "hide_face",
                "hide_person",
                "save_person_profile",
                "accept_pending_face_labels",
            ):
                self.assertIn(expected, actions)
            accept_event = next(event for event in events if event.action == "accept_pending_face_labels")
            self.assertTrue(accept_event.reversible)
            purge_event = events[0]
            self.assertEqual("purge_face_data", purge_event.action)
            self.assertFalse(purge_event.reversible)
            self.assertIn("table_counts", purge_event.details)

    def test_face_labels_pending_and_prototypes_survive_safe_rescan_without_index_shift(self):
        with TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "group.jpg"
            Image.new("RGB", (96, 64), (20, 30, 40)).save(image_path)
            service = FaceIndexService(
                detection_service=FakeFaceDetectionService(),
                embedding_service=FakeFaceEmbeddingService(),
                db_path=Path(tmp) / "faces.sqlite3",
            )
            service.save_face_records(
                [
                    FaceIndexRecord(str(image_path), 0, (0, 0, 36, 36), 0.99, np.array([1.0, 0.0, 0.0], dtype=np.float32)),
                    FaceIndexRecord(str(image_path), 1, (48, 0, 84, 36), 0.98, np.array([0.0, 1.0, 0.0], dtype=np.float32)),
                ],
                mtime_ns=image_path.stat().st_mtime_ns,
                file_size=image_path.stat().st_size,
                assess_quality=False,
            )
            service.label_indexed_faces("Alice", [(str(image_path), 0)], similarity_threshold=0.5)
            service.accept_pending_face_labels([item.proposal_id for item in service.load_pending_face_labels(include_tiny_faces=True)])
            service._queue_face_label_assignments([
                FaceLabelAssignment("Bob", str(image_path), 1, (48, 0, 84, 36), 0.95, source="manual_review", pending=True)
            ])

            service.save_face_records(
                [
                    FaceIndexRecord(str(image_path), 0, (49, 1, 85, 37), 0.98, np.array([0.0, 1.0, 0.0], dtype=np.float32)),
                    FaceIndexRecord(str(image_path), 1, (1, 1, 37, 37), 0.99, np.array([1.0, 0.0, 0.0], dtype=np.float32)),
                ],
                mtime_ns=image_path.stat().st_mtime_ns,
                file_size=image_path.stat().st_size,
                assess_quality=False,
            )
            labels = service.list_face_labels(include_tiny_faces=True)
            self.assertEqual([("Alice", 1)], [(item.person_name, item.face_index) for item in labels])
            pending = service.load_pending_face_labels(include_tiny_faces=True)
            self.assertEqual([("Bob", 0)], [(item.person_name, item.face_index) for item in pending])
            prototypes = service.load_person_prototype_faces("Alice", include_tiny_faces=True)
            self.assertEqual([(str(image_path), 1)], [(item.image_path, item.face_index) for item in prototypes])

            service.save_face_records(
                [
                    FaceIndexRecord(str(image_path), 0, (49, 1, 85, 37), 0.98, np.array([0.0, 1.0, 0.0], dtype=np.float32)),
                ],
                mtime_ns=image_path.stat().st_mtime_ns,
                file_size=image_path.stat().st_size,
                assess_quality=False,
            )
            self.assertEqual([], service.list_face_labels(include_tiny_faces=True))
            self.assertEqual([], service.load_person_prototype_faces("Alice", include_tiny_faces=True))
            self.assertEqual([("Bob", 0)], [(item.person_name, item.face_index) for item in service.load_pending_face_labels(include_tiny_faces=True)])

    def test_load_person_prototype_faces_backfills_legacy_tiny_visibility(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            tiny_image = root / "tiny.jpg"
            visible_image = root / "visible.jpg"
            for path, color in (
                (tiny_image, (20, 30, 40)),
                (visible_image, (30, 40, 50)),
            ):
                Image.new("RGB", (64, 64), color).save(path)
            db_path = root / "faces.sqlite3"
            service = FaceIndexService(
                detection_service=FakeFaceDetectionService(),
                embedding_service=FakeFaceEmbeddingService(),
                db_path=db_path,
            )
            service.save_face_records(
                [
                    FaceIndexRecord(str(tiny_image), 0, (0, 0, 20, 20), 0.99, np.array([1.0, 0.0, 0.0], dtype=np.float32)),
                    FaceIndexRecord(str(visible_image), 0, (0, 0, 40, 40), 0.98, np.array([0.9, 0.1, 0.0], dtype=np.float32)),
                ],
                mtime_ns=1,
                file_size=1,
                assess_quality=False,
            )
            service.label_indexed_faces(
                "Alice",
                [(str(tiny_image), 0), (str(visible_image), 0)],
                similarity_threshold=0.55,
            )

            with sqlite3.connect(str(db_path)) as connection:
                connection.execute("UPDATE face_index SET is_tiny=0")
                connection.execute(
                    "INSERT OR REPLACE INTO face_settings(key, value) VALUES (?, ?)",
                    ("face_index_tiny_visibility_revision", "0"),
                )

            reloaded = FaceIndexService(
                detection_service=FakeFaceDetectionService(),
                embedding_service=FakeFaceEmbeddingService(),
                db_path=db_path,
            )

            all_faces = reloaded.load_person_prototype_faces("Alice", include_tiny_faces=True)
            visible_faces = reloaded.load_person_prototype_faces("Alice", include_tiny_faces=False)

            self.assertEqual(
                {str(tiny_image), str(visible_image)},
                {face.image_path for face in all_faces},
            )
            self.assertEqual([str(visible_image)], [face.image_path for face in visible_faces])

            with sqlite3.connect(str(db_path)) as connection:
                rows = connection.execute(
                    "SELECT image_path, is_tiny FROM face_index ORDER BY image_path"
                ).fetchall()
            self.assertEqual(
                {
                    str(tiny_image): 1,
                    str(visible_image): 0,
                },
                {str(image_path): int(is_tiny) for image_path, is_tiny in rows},
            )

    def test_load_face_metadata_many_skips_hidden_faces_by_default(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_a = root / "alice.jpg"
            image_b = root / "bob.jpg"
            for path, color in (
                (image_a, (20, 30, 40)),
                (image_b, (30, 40, 50)),
            ):
                Image.new("RGB", (64, 64), color).save(path)
            service = FaceIndexService(
                detection_service=FakeFaceDetectionService(),
                embedding_service=FakeFaceEmbeddingService(),
                db_path=root / "faces.sqlite3",
            )
            service.save_face_records(
                [
                    FaceIndexRecord(
                        str(image_a),
                        0,
                        (0, 0, 36, 36),
                        0.99,
                        np.array([1.0, 0.0, 0.0], dtype=np.float32),
                        quality_status="clean",
                    ),
                    FaceIndexRecord(
                        str(image_b),
                        0,
                        (0, 0, 36, 36),
                        0.98,
                        np.array([0.0, 1.0, 0.0], dtype=np.float32),
                        quality_status="clean",
                    ),
                ],
                mtime_ns=1,
                file_size=1,
                assess_quality=False,
            )
            service.label_indexed_faces("Alice", [(str(image_a), 0)], similarity_threshold=0.5)
            service.label_indexed_faces("Bob", [(str(image_b), 0)], similarity_threshold=0.5)
            service.accept_pending_face_labels(
                [item.proposal_id for item in service.load_pending_face_labels(include_tiny_faces=True)]
            )
            with sqlite3.connect(str(root / "faces.sqlite3")) as connection:
                connection.execute(
                    "UPDATE face_index SET quality_status=? WHERE image_path=? AND face_index=?",
                    ("review", str(image_b), 0),
                )
            service.hide_face(str(image_b), 0)

            refs = [(str(image_a), 0), (str(image_b), 0), (str(image_a), 0)]
            visible_only = service.load_face_metadata_many(refs)
            including_hidden = service.load_face_metadata_many(refs, include_hidden=True)

            self.assertEqual({(str(image_a), 0): ("Alice", "clean")}, visible_only)
            self.assertEqual(
                {
                    (str(image_a), 0): ("Alice", "clean"),
                    (str(image_b), 0): ("Bob", "review"),
                },
                including_hidden,
            )

    def test_search_similar_faces_combines_multiple_example_embeddings(self):
        with TemporaryDirectory() as tmp:
            image_a = Path(tmp) / "a.jpg"
            image_b = Path(tmp) / "b.jpg"
            for path, color in (
                (image_a, (20, 30, 40)),
                (image_b, (30, 40, 50)),
            ):
                Image.new("RGB", (64, 64), color).save(path)
            service = FaceIndexService(
                detection_service=FakeFaceDetectionService(),
                embedding_service=FakeFaceEmbeddingService(),
                db_path=Path(tmp) / "faces.sqlite3",
            )
            service.save_face_records(
                [
                    FaceIndexRecord(str(image_a), 0, (0, 0, 32, 32), 0.99, np.array([1.0, 0.0, 0.0], dtype=np.float32)),
                    FaceIndexRecord(str(image_b), 0, (0, 0, 32, 32), 0.98, np.array([0.9701425, 0.2425356, 0.0], dtype=np.float32)),
                ],
                mtime_ns=image_a.stat().st_mtime_ns,
                file_size=image_a.stat().st_size,
            )
            captured: dict[str, object] = {}

            def _fake_search(query_embedding, **kwargs):
                captured["query_embedding"] = np.asarray(query_embedding, dtype=np.float32)
                captured.update(kwargs)
                return []

            service._search_by_embedding = _fake_search  # type: ignore[method-assign]
            service.search_similar_faces([(str(image_a), 0), (str(image_b), 0)], top_k=5, min_score=0.0)
            expected = np.asarray([1.0, 0.0, 0.0], dtype=np.float32) + np.asarray([0.9701425, 0.2425356, 0.0], dtype=np.float32)
            expected = expected / np.linalg.norm(expected)
            self.assertTrue(np.allclose(captured["query_embedding"], expected, atol=1e-5))
            self.assertEqual({(str(image_a), 0), (str(image_b), 0)}, set(captured["exclude"]))

    def test_saved_identity_threshold_overrides_global_match_floor(self):
        with TemporaryDirectory() as tmp:
            image_a = Path(tmp) / "a.jpg"
            Image.new("RGB", (64, 64), (20, 30, 40)).save(image_a)
            service = FaceIndexService(
                detection_service=FakeFaceDetectionService(),
                embedding_service=FakeFaceEmbeddingService(),
                db_path=Path(tmp) / "faces.sqlite3",
            )
            service._save_person_prototype(
                "Alice",
                np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
                similarity_threshold=0.85,
            )
            captured: dict[str, object] = {}

            def _fake_search(query_embedding, **kwargs):
                captured["query_embedding"] = np.asarray(query_embedding, dtype=np.float32)
                captured.update(kwargs)
                return []

            service._search_by_embedding = _fake_search  # type: ignore[method-assign]
            service.search_by_person_name("Alice", include_tiny_faces=True)
            self.assertAlmostEqual(0.85, float(captured["min_face_score"]), places=5)

    def test_find_face_records_by_people_requires_all_requested_names(self):
        service = FaceIndexService(
            detection_service=FakeFaceDetectionService(),
            embedding_service=FakeFaceEmbeddingService(),
            db_path=Path("/tmp/faces-test.sqlite3"),
        )
        records = [
            IndexedFaceRecord("img1.jpg", 0, (0, 0, 20, 20), 0.9, np.zeros((3,), dtype=np.float32), person_name="Alice"),
            IndexedFaceRecord("img1.jpg", 1, (20, 0, 40, 20), 0.9, np.zeros((3,), dtype=np.float32), person_name="Bob"),
            IndexedFaceRecord("img2.jpg", 0, (0, 0, 20, 20), 0.9, np.zeros((3,), dtype=np.float32), person_name="Alice"),
        ]
        service.load_all_records = lambda **_kwargs: list(records)  # type: ignore[method-assign]
        results = service.find_face_records_by_people(["Alice", "Bob"], require_all=True)
        self.assertEqual(
            [("img1.jpg", 0), ("img1.jpg", 1)],
            [(record.image_path, int(record.face_index)) for record in results],
        )

    def test_find_face_records_queries_unknown_named_and_primary_people(self):
        service = FaceIndexService(
            detection_service=FakeFaceDetectionService(),
            embedding_service=FakeFaceEmbeddingService(),
            db_path=Path("/tmp/faces-test.sqlite3"),
        )
        records = [
            IndexedFaceRecord("img1.jpg", 0, (0, 0, 60, 60), 0.9, np.zeros((3,), dtype=np.float32), person_name="Alice"),
            IndexedFaceRecord("img1.jpg", 1, (65, 0, 85, 20), 0.9, np.zeros((3,), dtype=np.float32), person_name=""),
            IndexedFaceRecord("img2.jpg", 0, (0, 0, 80, 80), 0.9, np.zeros((3,), dtype=np.float32), person_name="Bob"),
            IndexedFaceRecord("img2.jpg", 1, (82, 0, 96, 14), 0.9, np.zeros((3,), dtype=np.float32), person_name="Carol"),
        ]
        service.load_all_records = lambda **_kwargs: list(records)  # type: ignore[method-assign]
        named = service.find_face_records_with_named_people()
        unknown = service.find_face_records_with_unknown_people()
        primary = service.find_face_records_with_primary_person("Bob")
        self.assertEqual(
            [("img1.jpg", 0), ("img2.jpg", 0), ("img2.jpg", 1)],
            [(record.image_path, int(record.face_index)) for record in named],
        )
        self.assertEqual([("img1.jpg", 1)], [(record.image_path, int(record.face_index)) for record in unknown])
        self.assertEqual([("img2.jpg", 0)], [(record.image_path, int(record.face_index)) for record in primary])

    def test_max_speed_profile_uses_more_workers_than_balanced(self):
        balanced = select_performance_profile("balanced")
        max_speed = select_performance_profile("max_speed")
        self.assertGreaterEqual(max_speed.thumbnail_workers, 1)
        self.assertGreaterEqual(max_speed.embedding_preprocess_workers, 1)
        self.assertIsNone(max_speed.backend_worker_cap)
        self.assertGreaterEqual(max_speed.embedding_memory_cache_size, balanced.embedding_memory_cache_size)

    def test_low_memory_profile_uses_less_cache_and_parallelism_than_balanced(self):
        balanced = select_performance_profile("balanced")
        low_memory = select_performance_profile("low_memory")
        self.assertLessEqual(low_memory.thumbnail_workers, balanced.thumbnail_workers)
        self.assertLessEqual(low_memory.embedding_preprocess_workers, balanced.embedding_preprocess_workers)
        self.assertLess(low_memory.embedding_memory_cache_size, balanced.embedding_memory_cache_size)
        self.assertEqual(1, low_memory.backend_workers_for(3))

    def test_max_speed_profile_uses_fast_kmeans_backend(self):
        service = ClusteringService()
        rng = np.random.default_rng(42)
        matrix = rng.normal(size=(600, 8)).astype(np.float32)
        matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)

        _clusters, metrics = service.cluster_prepared(
            matrix,
            8,
            backend="cosine-kmeans",
            performance_profile="max_speed",
        )

        self.assertEqual("cosine-kmeans-fast", metrics["backend"])

    def test_main_import_does_not_pull_face_search_stack(self):
        script = (
            "import sys; "
            "sys.path.insert(0, r'%s'); "
            "import main; "
            "print('app.services.face_search' in sys.modules)"
        ) % str(Path(__file__).resolve().parents[1] / "src")
        completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
        self.assertEqual("False", completed.stdout.strip())

    def test_embedding_service_can_skip_cache_reads_and_still_write_embeddings(self):
        class FakeDevice:
            type = "cpu"

            def __str__(self):
                return "cpu"

        class FakeCacheLookupService:
            def __init__(self):
                self.get_calls = 0
                self.put_rows = []

            @staticmethod
            def build_embedding_key(image_path, model_name, signature):
                return f"{image_path}|{model_name}|{signature}"

            def build_embedding_keys(self, image_paths, model_name, signature, *, path_fingerprints=None):
                _ = path_fingerprints
                return [self.build_embedding_key(image_path, model_name, signature) for image_path in image_paths]

            def get_embeddings_many(self, cache_keys):
                self.get_calls += 1
                return {}

            def put_embeddings_many(self, rows):
                self.put_rows.extend(rows)

        class FakeModelManager:
            def __init__(self):
                self.device = FakeDevice()
                self.execution_policy = SimpleNamespace(
                    effective_mode="cpu",
                    onnx_provider="CPUExecutionProvider",
                    reason="test",
                )

            @staticmethod
            def get_bundle(model_name, use_onnx=False):
                return SimpleNamespace(
                    model_name=model_name,
                    signature="sig",
                    input_mode="tensor",
                    onnx_session=None,
                    family="torchvision",
                    preprocess=None,
                )

            @staticmethod
            def suggested_batch_size(model_name):
                _ = model_name
                return 2

        cache = FakeCacheLookupService()
        service = EmbeddingService(
            model_manager=FakeModelManager(),
            cache_service=cache,
            performance_profile=select_performance_profile("balanced"),
        )
        with patch.object(service, "_prepare_batch", side_effect=lambda bundle, batch_items: (list(batch_items), list(batch_items), 0.0)):
            with patch.object(
                service,
                "_run_model_with_backoff",
                side_effect=lambda bundle, prepared_batch: (
                    [np.asarray([1.0, 0.0], dtype=np.float32) for _ in prepared_batch],
                    False,
                ),
            ):
                embeddings, metrics = service.embed_paths(["a.jpg", "b.jpg"], "fake", use_cache_lookup=False)

        self.assertEqual(0, cache.get_calls)
        self.assertEqual(2, len(cache.put_rows))
        self.assertEqual(2, len(embeddings))
        self.assertEqual("disabled", metrics["embedding_cache_lookup"])
        self.assertEqual(0.0, metrics["cache_lookup_time_s"])
        self.assertEqual(0, metrics["cache_hits"])
        self.assertEqual(0, metrics["cache_misses"])

    def test_embedding_service_enabled_lookup_uses_cache_reads_and_preserves_hit_miss_metrics(self):
        class FakeDevice:
            type = "cpu"

            def __str__(self):
                return "cpu"

        class FakeCacheLookupService:
            def __init__(self):
                self.get_calls = 0
                self.put_rows = []

            @staticmethod
            def build_embedding_key(image_path, model_name, signature):
                return f"{image_path}|{model_name}|{signature}"

            def build_embedding_keys(self, image_paths, model_name, signature, *, path_fingerprints=None):
                _ = path_fingerprints
                return [self.build_embedding_key(image_path, model_name, signature) for image_path in image_paths]

            def get_embeddings_many(self, cache_keys):
                self.get_calls += 1
                return {cache_keys[0]: np.asarray([9.0, 9.0], dtype=np.float32)}

            def put_embeddings_many(self, rows):
                self.put_rows.extend(rows)

        class FakeModelManager:
            def __init__(self):
                self.device = FakeDevice()
                self.execution_policy = SimpleNamespace(
                    effective_mode="cpu",
                    onnx_provider="CPUExecutionProvider",
                    reason="test",
                )

            @staticmethod
            def get_bundle(model_name, use_onnx=False):
                return SimpleNamespace(
                    model_name=model_name,
                    signature="sig",
                    input_mode="tensor",
                    onnx_session=None,
                    family="torchvision",
                    preprocess=None,
                )

            @staticmethod
            def suggested_batch_size(model_name):
                _ = model_name
                return 2

        cache = FakeCacheLookupService()
        service = EmbeddingService(
            model_manager=FakeModelManager(),
            cache_service=cache,
            performance_profile=select_performance_profile("balanced"),
        )
        with patch.object(service, "_prepare_batch", side_effect=lambda bundle, batch_items: (list(batch_items), list(batch_items), 0.0)):
            with patch.object(
                service,
                "_run_model_with_backoff",
                side_effect=lambda bundle, prepared_batch: (
                    [np.asarray([1.0, 0.0], dtype=np.float32) for _ in prepared_batch],
                    False,
                ),
            ):
                embeddings, metrics = service.embed_paths(["a.jpg", "b.jpg"], "fake", use_cache_lookup=True)

        self.assertEqual(1, cache.get_calls)
        self.assertEqual(1, len(cache.put_rows))
        self.assertEqual(2, len(embeddings))
        self.assertEqual("enabled", metrics["embedding_cache_lookup"])
        self.assertEqual(1, metrics["cache_hits"])
        self.assertEqual(1, metrics["cache_misses"])

    def test_embedding_service_full_static_cache_hit_skips_model_load(self):
        class FakeDevice:
            type = "cpu"

            def __str__(self):
                return "cpu"

        class FakeCacheLookupService:
            def __init__(self):
                self.get_calls = 0
                self.put_rows = []

            @staticmethod
            def build_embedding_key(image_path, model_name, signature):
                return f"{image_path}|{model_name}|{signature}"

            def build_embedding_keys(self, image_paths, model_name, signature, *, path_fingerprints=None):
                _ = path_fingerprints
                return [self.build_embedding_key(image_path, model_name, signature) for image_path in image_paths]

            def get_embeddings_many(self, cache_keys):
                self.get_calls += 1
                return {
                    cache_key: np.asarray([float(index), 1.0], dtype=np.float32)
                    for index, cache_key in enumerate(cache_keys)
                }

            def put_embeddings_many(self, rows):
                self.put_rows.extend(rows)

        class FakeModelManager:
            def __init__(self):
                self.device = FakeDevice()
                self.execution_policy = SimpleNamespace(
                    effective_mode="cpu",
                    onnx_provider="CPUExecutionProvider",
                    reason="test",
                )

            @staticmethod
            def static_signature(model_name, use_onnx=False):
                _ = (model_name, use_onnx)
                return "static-test-signature"

            @staticmethod
            def get_bundle(model_name, use_onnx=False):
                _ = (model_name, use_onnx)
                raise AssertionError("get_bundle should not run when every embedding is cached")

        cache = FakeCacheLookupService()
        service = EmbeddingService(
            model_manager=FakeModelManager(),
            cache_service=cache,
            performance_profile=select_performance_profile("balanced"),
        )

        embeddings, metrics = service.embed_paths(["a.jpg", "b.jpg"], "fake", use_cache_lookup=True)

        self.assertEqual(1, cache.get_calls)
        self.assertEqual([], cache.put_rows)
        self.assertEqual(2, len(embeddings))
        self.assertEqual("cache", metrics["embedding_backend"])
        self.assertEqual(0.0, metrics["model_load_time_s"])
        self.assertEqual(2, metrics["cache_hits"])
        self.assertEqual(0, metrics["cache_misses"])
        self.assertIn("Skipped model load", metrics["runtime_reason"])

    def test_pipeline_runs_multiple_backends_and_builds_membership(self):
        with TemporaryDirectory() as tmp:
            paths = []
            for index in range(4):
                image_path = Path(tmp) / f"{index}.png"
                Image.new("RGB", (16, 16), (index * 30, index * 30, index * 30)).save(image_path)
                paths.append(str(image_path))
            clustering_service = FakeClusteringService()
            embedding_service = FakeEmbeddingService()
            pipeline = ClusteringPipelineService(
                discovery_service=FakeDiscoveryService(paths),
                embedding_service=embedding_service,
                clustering_service=clustering_service,
                embedding_index_service=EmbeddingIndexService(),
                result_cache_service=ResultCacheService(),
            )
            result, metrics = pipeline.run(
                ClusteringRequest(
                    directory=str(tmp),
                    embedding_models=["fake"],
                    num_clusters=2,
                    clustering_backends=["cosine-kmeans", "graph"],
                    recursive=True,
                    similarity_mode="cosine",
                    outlier_policy="assign",
                    use_onnx=False,
                    reuse_result_cache=False,
                    use_embedding_cache_lookup=False,
                )
            )
            key_kmeans = "fake::cosine::cosine-kmeans"
            key_graph = "fake::cosine::graph"
            self.assertIn(key_kmeans, result.clusters_by_key)
            self.assertIn(key_graph, result.clusters_by_key)
            self.assertIn(key_kmeans, result.cluster_explanations_by_key)
            self.assertEqual(2, result.cluster_explanations_by_key[key_kmeans][0].cluster_size)
            self.assertEqual(0, result.membership_by_image[paths[0]][key_kmeans]["cluster_id"])
            self.assertEqual(10, result.membership_by_image[paths[0]][key_graph]["cluster_id"])
            self.assertIn("clustering_backends", metrics)
            self.assertEqual([False], embedding_service.use_cache_lookup_values)
            self.assertEqual("disabled", metrics["embedding_cache_lookup"])
            self.assertEqual(1, clustering_service.prepare_calls)

    def test_pipeline_full_result_cache_hit_skips_embedding_prepare_and_index_for_basic_runs(self):
        with TemporaryDirectory() as tmp:
            paths = []
            for index in range(4):
                image_path = Path(tmp) / f"cached_{index}.png"
                Image.new("RGB", (16, 16), (index * 20, index * 20, index * 20)).save(image_path)
                paths.append(str(image_path))
            clustering_service = FakeClusteringService()
            embedding_service = FakeEmbeddingService()
            result_cache_service = AlwaysHitResultCacheService(paths)
            pipeline = ClusteringPipelineService(
                discovery_service=FakeDiscoveryService(paths),
                embedding_service=embedding_service,
                clustering_service=clustering_service,
                embedding_index_service=EmbeddingIndexService(),
                result_cache_service=result_cache_service,
            )

            result, metrics = pipeline.run(
                ClusteringRequest(
                    directory=str(tmp),
                    embedding_models=["fake"],
                    num_clusters=2,
                    clustering_backends=["cosine-kmeans", "graph"],
                    recursive=True,
                    similarity_mode="semantic",
                    similarity_modes=["semantic", "cosine"],
                    outlier_policy="assign",
                    use_onnx=False,
                    reuse_result_cache=True,
                    use_embedding_cache_lookup=True,
                    generate_cluster_explanations=False,
                )
            )

            expected_keys = {
                "fake::semantic::cosine-kmeans",
                "fake::semantic::graph",
                "fake::cosine::cosine-kmeans",
                "fake::cosine::graph",
            }
            self.assertEqual(expected_keys, set(result.clusters_by_key))
            self.assertEqual({}, result.cluster_explanations_by_key)
            self.assertEqual([], embedding_service.use_cache_lookup_values)
            self.assertEqual(0, clustering_service.prepare_calls)
            self.assertEqual(4, len(result_cache_service.load_calls))
            self.assertEqual([], result_cache_service.save_calls)
            self.assertTrue(metrics["full_result_cache_hit"])
            self.assertEqual("result-cache-hit", metrics["embedding_stage_skipped"])
            self.assertEqual("skipped", metrics["embedding_backend"])
            self.assertEqual(0.0, metrics["embedding_time_s"])
            self.assertEqual(0.0, metrics["prepared_matrix_time_s"])
            self.assertEqual(0.0, metrics["clustering_time_s"])
            self.assertEqual(0.0, metrics["backend_wall_time_s"])
            self.assertEqual("skipped-result-cache-hit", metrics["index_reuse"])
            self.assertEqual(2, metrics["prepared_matrix_dim[fake::semantic]"])
            self.assertEqual(2, metrics["prepared_matrix_dim[fake::cosine]"])
            self.assertTrue(metrics["backend_cache_hit[fake::semantic::cosine-kmeans]"])
            self.assertEqual(0, result.membership_by_image[paths[0]]["fake::semantic::cosine-kmeans"]["cluster_id"])

    def test_pipeline_expands_similarity_modes_without_reembedding(self):
        with TemporaryDirectory() as tmp:
            paths = []
            for index in range(4):
                image_path = Path(tmp) / f"multi_{index}.png"
                Image.new("RGB", (16, 16), (index * 20, index * 20, index * 20)).save(image_path)
                paths.append(str(image_path))
            clustering_service = FakeClusteringService()
            embedding_service = FakeEmbeddingService()
            pipeline = ClusteringPipelineService(
                discovery_service=FakeDiscoveryService(paths),
                embedding_service=embedding_service,
                clustering_service=clustering_service,
                embedding_index_service=EmbeddingIndexService(),
                result_cache_service=ResultCacheService(),
            )

            result, metrics = pipeline.run(
                ClusteringRequest(
                    directory=str(tmp),
                    embedding_models=["fake"],
                    num_clusters=2,
                    clustering_backends=["cosine-kmeans"],
                    recursive=True,
                    similarity_mode="semantic",
                    similarity_modes=["semantic", "cosine"],
                    outlier_policy="assign",
                    use_onnx=False,
                    reuse_result_cache=False,
                    use_embedding_cache_lookup=False,
                )
            )

            self.assertIn("fake::semantic::cosine-kmeans", result.clusters_by_key)
            self.assertIn("fake::cosine::cosine-kmeans", result.clusters_by_key)
            self.assertIn("fake::semantic::cosine-kmeans", result.cluster_explanations_by_key)
            self.assertEqual([False], embedding_service.use_cache_lookup_values)
            self.assertEqual(2, clustering_service.prepare_calls)
            self.assertEqual("semantic, cosine", metrics["similarity_modes"])
            self.assertEqual(2, metrics["prepared_matrix_dim[fake::semantic]"])
            self.assertEqual(2, metrics["prepared_matrix_dim[fake::cosine]"])
            self.assertIn("semantic full vector", metrics["similarity_space[fake::semantic]"])
            self.assertIn("cosine full vector", metrics["similarity_space[fake::cosine]"])
            explanation = result.cluster_explanations_by_key["fake::semantic::cosine-kmeans"][0]
            self.assertEqual("semantic", explanation.similarity_mode)
            self.assertIn("semantic full vector", explanation.similarity_space)
            self.assertEqual((1.0, 1.0), explanation.member_cohesion_scores)

    def test_pipeline_emits_cluster_meanings_when_enabled(self):
        with TemporaryDirectory() as tmp:
            paths = []
            for index in range(4):
                image_path = Path(tmp) / f"meaning_{index}.png"
                Image.new("RGB", (16, 16), (index * 20, index * 20, index * 20)).save(image_path)
                paths.append(str(image_path))
            meaning_service = FakeClusterMeaningService()
            pipeline = ClusteringPipelineService(
                discovery_service=FakeDiscoveryService(paths),
                embedding_service=FakeEmbeddingService(),
                clustering_service=FakeClusteringService(),
                embedding_index_service=EmbeddingIndexService(),
                result_cache_service=ResultCacheService(),
                cluster_meaning_service=meaning_service,
            )

            result, metrics = pipeline.run(
                ClusteringRequest(
                    directory=str(tmp),
                    embedding_models=["fake"],
                    num_clusters=2,
                    clustering_backends=["cosine-kmeans"],
                    recursive=True,
                    similarity_mode="semantic",
                    outlier_policy="assign",
                    use_onnx=False,
                    reuse_result_cache=False,
                    use_embedding_cache_lookup=False,
                    generate_cluster_meanings=True,
                    cluster_meaning_model="auto",
                )
            )

            self.assertEqual(1, len(meaning_service.calls))
            self.assertIn("fake::semantic::cosine-kmeans", result.cluster_meanings_by_key)
            self.assertEqual("ok", metrics["cluster_meaning_status"])

    def test_pipeline_omits_cluster_meanings_when_disabled(self):
        with TemporaryDirectory() as tmp:
            paths = []
            for index in range(4):
                image_path = Path(tmp) / f"no_meaning_{index}.png"
                Image.new("RGB", (16, 16), (index * 20, index * 20, index * 20)).save(image_path)
                paths.append(str(image_path))
            meaning_service = FakeClusterMeaningService()
            pipeline = ClusteringPipelineService(
                discovery_service=FakeDiscoveryService(paths),
                embedding_service=FakeEmbeddingService(),
                clustering_service=FakeClusteringService(),
                embedding_index_service=EmbeddingIndexService(),
                result_cache_service=ResultCacheService(),
                cluster_meaning_service=meaning_service,
            )

            result, _metrics = pipeline.run(
                ClusteringRequest(
                    directory=str(tmp),
                    embedding_models=["fake"],
                    num_clusters=2,
                    clustering_backends=["cosine-kmeans"],
                    recursive=True,
                    similarity_mode="semantic",
                    outlier_policy="assign",
                    use_onnx=False,
                    reuse_result_cache=False,
                    use_embedding_cache_lookup=False,
                )
            )

            self.assertEqual([], meaning_service.calls)
            self.assertEqual({}, result.cluster_meanings_by_key)

    def test_pipeline_can_cluster_explicit_source_paths(self):
        with TemporaryDirectory() as tmp:
            paths = []
            for index in range(4):
                image_path = Path(tmp) / f"ws_{index}.png"
                Image.new("RGB", (16, 16), (index * 20, index * 20, index * 20)).save(image_path)
                paths.append(str(image_path))
            pipeline = ClusteringPipelineService(
                discovery_service=FakeDiscoveryService([]),
                embedding_service=FakeEmbeddingService(),
                clustering_service=FakeClusteringService(),
                embedding_index_service=EmbeddingIndexService(),
                result_cache_service=ResultCacheService(),
            )
            result, metrics = pipeline.run(
                ClusteringRequest(
                    directory="missing",
                    embedding_models=["fake"],
                    num_clusters=2,
                    clustering_backends=["cosine-kmeans"],
                    recursive=True,
                    similarity_mode="cosine",
                    outlier_policy="assign",
                    use_onnx=False,
                    reuse_result_cache=False,
                    source_paths=paths,
                )
            )
            self.assertEqual("working_set", metrics["source_scope"])
            self.assertEqual(set(paths), set(result.membership_by_image.keys()))


if __name__ == "__main__":
    unittest.main()
