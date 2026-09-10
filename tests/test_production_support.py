import json
import os
import sys
import time
import unittest
import logging
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
from PIL import Image
from PyQt6.QtCore import QSettings, QThread
from PyQt6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from apps.pyqt_production.icon_assets import production_icon_path
from apps.pyqt_production.identity import PRODUCTION_QSETTINGS_APP, PRODUCTION_QSETTINGS_ORG
from apps.pyqt_production.model_download_controller import ModelDownloadController
from apps.shared.benchmark_schema import BenchmarkStage, build_report, write_report
from apps.pyqt_production.session_controller import ClusteringSessionController
from apps.shared.runtime_migrations import RuntimeMigrationService
from apps.shared.runtime_support import activate_runtime_root
from apps.shared.support_bundle import export_support_bundle
from app.services.clustering_pipeline import ClusteringPipelineService, ClusteringRequest
from app.services.model_assets import sha256_file
from infra.runtime import RuntimeCapabilities, RuntimeCapabilityService
import infra.settings as settings_mod
from ml.clustering import ClusteringService


APP = QApplication.instance() or QApplication([])


class _FakeEmbeddingService:
    def __init__(self):
        self.last_fingerprints = None

    def embed_paths(
        self,
        image_paths,
        model_name,
        progress_callback=None,
        use_onnx=False,
        use_cache_lookup=True,
        path_fingerprints=None,
    ):
        _ = (progress_callback, use_onnx, use_cache_lookup)
        self.last_fingerprints = dict(path_fingerprints or {})
        vectors = [(path, np.asarray([1.0, 0.0], dtype=np.float32)) for path in image_paths]
        return vectors, {
            "cache_hits": 0,
            "cache_misses": len(vectors),
            "model_load_time_s": 0.0,
            "cache_lookup_time_s": 0.0,
            "preprocess_time_s": 0.0,
            "inference_time_s": 0.0,
            "model_device": "cpu",
            "embedding_backend": "torch",
            "amp_enabled": False,
            "effective_mode": "cpu",
            "onnx_provider": "CPUExecutionProvider",
            "runtime_reason": "test",
        }


class _FakeClusteringService:
    def __init__(self):
        self.prepare_calls = 0
        self._delegate = ClusteringService()

    def prepare_matrix(self, embeddings, pca_dim=None, similarity_mode="semantic"):
        _ = (pca_dim, similarity_mode)
        self.prepare_calls += 1
        return np.asarray(embeddings, dtype=np.float32)

    def cluster_prepared(self, embeddings, num_clusters, backend="cosine-kmeans", outlier_policy="assign"):
        _ = (embeddings, num_clusters, backend, outlier_policy)
        raise AssertionError("cluster_prepared should not run on a result-cache hit")

    def build_cluster_explanations(self, metric_matrix, clusters, *, cluster_quality_score=None, prepared_info=None):
        return self._delegate.build_cluster_explanations(
            metric_matrix,
            clusters,
            cluster_quality_score=cluster_quality_score,
            prepared_info=prepared_info,
        )


class _FakeEmbeddingIndexService:
    @staticmethod
    def build_snapshot_key_from_fingerprints(fingerprints):
        payload = "|".join(f"{path}:{mtime_ns}:{size}" for path, mtime_ns, size in fingerprints)
        return f"snapshot:{payload}"

    def ensure_index(self, snapshot_key, model_name, ordered_embeddings):
        _ = (snapshot_key, model_name, ordered_embeddings)
        raise AssertionError("ensure_index should not run on a result-cache hit")


class _FakeResultCacheService:
    def __init__(self, cached_paths: list[str] | None = None):
        self.cached_paths = list(cached_paths or [])

    def build_result_key(self, **kwargs):
        return "::".join(f"{key}={value}" for key, value in sorted(kwargs.items()))

    def load(self, result_key):
        _ = result_key
        return (
            {0: list(self.cached_paths)},
            {"clustering_time_s": 1.234, "cluster_quality_score": 0.9},
        )

    def save(self, result_key, clustered_images, metrics):
        _ = (result_key, clustered_images, metrics)


class ProductionSupportTests(unittest.TestCase):
    def setUp(self):
        store = QSettings(PRODUCTION_QSETTINGS_ORG, PRODUCTION_QSETTINGS_APP)
        self._production_settings_snapshot = {
            key: store.value(key)
            for key in store.allKeys()
        }
        store.setValue("setup/completed", True)
        store.setValue("safety/read_only_mode", False)
        store.setValue("workspace/default_view", "clustering")
        store.setValue("workspace/faces_mode", "basic")
        store.setValue("runtime/preferred_mode", "cpu")
        store.sync()

    @staticmethod
    def _wait_for(predicate, *, timeout_s: float = 5.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            APP.processEvents()
            if predicate():
                return
            time.sleep(0.01)
        APP.processEvents()
        if not predicate():
            raise AssertionError("Timed out waiting for condition")

    @classmethod
    def _wait_for_storage_idle(cls, window) -> None:
        cls._wait_for(
            lambda: window._storage_usage_thread is None and window._storage_clear_thread is None,
            timeout_s=5.0,
        )

    class _PollingAsyncQThread(QThread):
        def __init__(self):
            super().__init__()
            self.stop_event = Event()
            self.wait_calls: list[int] = []

        def wait(self, timeout_ms: int = 0):
            self.wait_calls.append(int(timeout_ms))
            return super().wait(timeout_ms)

        def run(self):
            while not self.stop_event.is_set():
                time.sleep(0.01)

    def tearDown(self):
        settings_mod._RUNTIME_BASE_DIR = None
        store = QSettings(PRODUCTION_QSETTINGS_ORG, PRODUCTION_QSETTINGS_APP)
        store.clear()
        for key, value in self._production_settings_snapshot.items():
            store.setValue(key, value)
        store.sync()

    def test_settings_uses_exact_image_clustering_app_dir_override(self):
        with TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
                settings_mod._RUNTIME_BASE_DIR = None
                resolved = settings_mod.get_runtime_base_dir()
                self.assertEqual(Path(tmp), resolved)

    def test_settings_uses_linux_data_home_without_a_runtime_override(self):
        with TemporaryDirectory() as tmp:
            environment = dict(os.environ)
            environment.pop("CLUSTERLENS_RUNTIME_ROOT", None)
            environment.pop("IMAGE_CLUSTERING_APP_DIR", None)
            environment["XDG_DATA_HOME"] = tmp
            with patch.dict(os.environ, environment, clear=True):
                settings_mod._RUNTIME_BASE_DIR = None
                self.assertEqual(Path(tmp) / "ClusterLens", settings_mod.get_runtime_base_dir())

    def test_workers_receive_the_canonical_runtime_root(self):
        with TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"CLUSTERLENS_RUNTIME_ROOT": tmp}, clear=False):
                layout = activate_runtime_root("ProductionWorkerEnvironmentTest")
                clustering_environment = ClusteringSessionController(layout)._worker_environment()
                download_environment = ModelDownloadController(layout)._worker_environment()
                for environment in (clustering_environment, download_environment):
                    self.assertEqual(str(layout.root), environment.value("CLUSTERLENS_RUNTIME_ROOT"))
                    self.assertEqual(str(layout.root), environment.value("IMAGE_CLUSTERING_APP_DIR"))
                    self.assertEqual(str(layout.model_assets_dir), environment.value("IMAGE_CLUSTERING_MODEL_ASSETS_DIR"))

    def test_production_wait_for_thread_shutdown_polls_python_qthreads(self):
        from apps.pyqt_production.ui.async_job import wait_for_thread_shutdown

        thread = self._PollingAsyncQThread()
        thread.start()
        self._wait_for(thread.isRunning, timeout_s=1.0)

        thread.stop_event.set()
        ready = wait_for_thread_shutdown(thread, timeout_ms=500)

        self.assertTrue(ready)
        self.assertEqual([], thread.wait_calls)
        self._wait_for(lambda: not thread.isRunning(), timeout_s=1.0)

    def test_production_icon_asset_resolves_from_repo(self):
        icon_path = production_icon_path()
        self.assertIsNotNone(icon_path)
        self.assertTrue(icon_path.exists())
        self.assertEqual("app_icon.png", icon_path.name)

    def test_production_window_loads_non_null_icon(self):
        from apps.pyqt_production.app import ProductionClusterApp, RUNTIME_LAYOUT

        window = ProductionClusterApp(RUNTIME_LAYOUT)
        self.assertFalse(window.windowIcon().isNull())
        window.close()

    def test_production_window_switches_to_faces_workspace(self):
        from apps.pyqt_production.app import ProductionClusterApp

        with TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
                settings_mod._RUNTIME_BASE_DIR = None
                layout = activate_runtime_root("ProductionFacesWorkspaceTest")
                window = ProductionClusterApp(layout)
                self._wait_for_storage_idle(window)
                self.assertIsNone(window.faces_pane)

                window.set_active_workspace("faces")
                self._wait_for(lambda: window.faces_pane is not None, timeout_s=5.0)

                self.assertEqual("faces", window._active_workspace)
                self.assertIs(window.workspace_stack.currentWidget(), window.faces_pane)
                self.assertEqual(["human"], list(window.face_services_global))
                self.assertEqual(1, window.faces_pane.face_mode_combo.count())
                self.assertTrue(window.faces_workspace_button.isChecked())
                self.assertFalse(window.clustering_workspace_button.isChecked())
                window.close()

    def test_production_window_switches_to_names_workspace_without_opening_faces(self):
        from apps.pyqt_production.app import ProductionClusterApp

        with TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
                settings_mod._RUNTIME_BASE_DIR = None
                layout = activate_runtime_root("ProductionNamesWorkspaceTest")
                window = ProductionClusterApp(layout)
                self._wait_for_storage_idle(window)
                window.show()
                APP.processEvents()

                self.assertTrue(window.names_workspace_button.isVisible())
                self.assertIsNone(window.faces_pane)
                window.set_active_workspace("names")
                self._wait_for(lambda: window.names_pane is not None, timeout_s=5.0)
                self._wait_for(
                    lambda: "saved name(s)" in window.names_pane.status_label.text(),
                    timeout_s=5.0,
                )

                self.assertEqual("names", window._active_workspace)
                self.assertIs(window.workspace_stack.currentWidget(), window.names_pane)
                self.assertTrue(window.names_workspace_button.isChecked())
                self.assertFalse(window.faces_workspace_button.isChecked())
                self.assertFalse(window._source_pane_visible())
                self.assertIsNone(window.faces_pane)
                self.assertTrue(callable(window.names_pane.gallery.context_menu_action_provider))
                window.close()

    def test_production_face_provider_never_instantiates_an_uninstalled_default_pack(self):
        from apps.pyqt_production.app import ProductionClusterApp
        from app.services import face_search
        from app.services.model_assets import ModelAssetService

        with TemporaryDirectory() as tmp:
            runtime_root = Path(tmp) / "runtime"
            managed_models = Path(tmp) / "managed-face-models"
            with (
                patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": str(runtime_root)}, clear=False),
                patch.object(face_search, "face_model_runtime_root_dir", return_value=managed_models),
                patch.object(ModelAssetService, "local_cache_present", return_value=True),
            ):
                settings_mod._RUNTIME_BASE_DIR = None
                layout = activate_runtime_root("ProductionFaceProviderFallbackTest")
                window = ProductionClusterApp(layout)
                self._wait_for_storage_idle(window)

                service = window._face_service_for_pipeline(
                    "global",
                    "human",
                    face_search.DEFAULT_HUMAN_FACE_DETECTOR_ID,
                    face_search.DEFAULT_HUMAN_FACE_EMBEDDER_ID,
                )

                self.assertEqual(face_search.BUILTIN_HUMAN_DETECTOR_ID, service.detector_id)
                self.assertEqual(face_search.BUILTIN_HUMAN_EMBEDDER_ID, service.embedder_id)
                self.assertEqual("face_search_index.db", Path(service.db_path).name)
                self.assertTrue(service.is_ready())
                self.assertNotIn("_internal", service.readiness_message())
                window.close()

    def test_production_startup_promotes_a_complete_interrupted_face_model_before_faces_ui(self):
        from apps.pyqt_production.app import ProductionClusterApp
        from app.services.face_model_installer import FaceModelInstaller

        with TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
                settings_mod._RUNTIME_BASE_DIR = None
                layout = activate_runtime_root("ProductionFaceRecoveryTest")
                installer = FaceModelInstaller()
                target = installer.bundle_dir("yunet_2026may")
                staging = target.parent / ".yunet_2026may.test.installing"
                staging.mkdir(parents=True)
                installer._copy_catalog_metadata("yunet_2026may", staging)
                payload = staging / "detector.onnx"
                payload.write_bytes(b"complete-staged-model")
                installer._write_install_record(payload, bundle_id="yunet_2026may")

                window = ProductionClusterApp(layout)
                self._wait_for_storage_idle(window)

                self.assertTrue((target / "detector.onnx").is_file())
                self.assertFalse(staging.exists())
                window.close()

    def test_production_face_results_open_in_main_gallery(self):
        from apps.pyqt_production.app import ProductionClusterApp

        with TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
                settings_mod._RUNTIME_BASE_DIR = None
                layout = activate_runtime_root("ProductionFaceResultsGalleryTest")
                window = ProductionClusterApp(layout)
                self._wait_for_storage_idle(window)
                window.set_active_workspace("faces")
                self._wait_for(lambda: window.faces_pane is not None, timeout_s=5.0)
                image_a = str(Path(tmp) / "a.jpg")
                image_b = str(Path(tmp) / "b.jpg")
                window.faces_pane.context_for_path = lambda path: {"origin": "faces", "path": path}

                window._open_face_results_in_main_gallery([image_a, image_b])
                self._wait_for(lambda: list(window.gallery_pane.images) == [image_a, image_b], timeout_s=2.0)

                self.assertEqual("clustering", window._active_workspace)
                self.assertEqual([image_a, image_b], list(window.gallery_pane.images))
                self.assertEqual("faces", window.gallery_pane.inspector_context_provider(image_a)["origin"])
                window.close()

    def test_benchmark_schema_writes_json_and_markdown(self):
        with TemporaryDirectory() as tmp:
            report = build_report(
                app_id="test_app",
                variant="production",
                scenario="cluster",
                runtime_root=tmp,
                stages=[
                    BenchmarkStage(
                        name="cold",
                        elapsed_s=1.25,
                        details={"pipeline_wall_time_s": 1.25, "hotspots": [{"function": "demo", "cumtime_s": 0.5}]},
                    )
                ],
                summary={"cold_pipeline_wall_time_s": 1.25},
                metadata={"folder": "demo"},
            )
            json_path, markdown_path = write_report(report, tmp, "report")
            self.assertTrue(json_path.exists())
            self.assertTrue(markdown_path.exists())
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual("test_app", payload["app_id"])
            markdown = markdown_path.read_text(encoding="utf-8")
            self.assertIn("Hotspots", markdown)
            self.assertIn("Stage Details", markdown)

    def test_support_bundle_exports_logs_and_crash_artifacts(self):
        with TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
                layout = activate_runtime_root("ProductionSupportBundleTest")
                layout.app_log.write_text(f"app-log {layout.root}", encoding="utf-8")
                layout.crash_log.write_text("crash-log", encoding="utf-8")
                layout.last_crash_json.write_text('{"kind":"test"}', encoding="utf-8")
                bundle_path = export_support_bundle(layout, bundle_name="bundle", metadata={"ok": True, "path": str(layout.root)})
                self.assertTrue(bundle_path.exists())
                with ZipFile(bundle_path) as archive:
                    names = set(archive.namelist())
                    self.assertIn("metadata.json", names)
                    self.assertIn("logs/app.log", names)
                    self.assertIn("crash/crash.log", names)
                    self.assertIn("crash/last_crash.json", names)
                    metadata = archive.read("metadata.json").decode("utf-8")
                    app_log = archive.read("logs/app.log").decode("utf-8")
                    self.assertIn("<runtime-root>", metadata)
                    self.assertIn("<runtime-root>", app_log)
                    self.assertNotIn(str(layout.root), metadata)

    def test_runtime_migration_writes_state_and_backs_up_databases(self):
        with TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
                layout = activate_runtime_root("ProductionMigrationTest")
                tag_db = layout.cache_dir / "image_tags.sqlite3"
                tag_db.write_bytes(b"tags")

                result = RuntimeMigrationService(layout).run()

                self.assertEqual(0, result.previous_version)
                self.assertEqual(1, result.current_version)
                self.assertTrue((layout.root / "runtime_migrations.json").exists())
                self.assertTrue((layout.root / "runtime_migrations.sqlite3").exists())
                self.assertTrue((layout.support_dir / "migration_backups" / "v1" / "image_tags.sqlite3").exists())

    def test_clustering_pipeline_cache_hit_reports_zero_live_backend_time(self):
        with TemporaryDirectory() as tmp:
            image_a = Path(tmp) / "a.jpg"
            image_b = Path(tmp) / "b.jpg"
            image_a.write_bytes(b"a")
            time.sleep(0.001)
            image_b.write_bytes(b"b")
            embedding_service = _FakeEmbeddingService()
            clustering_service = _FakeClusteringService()
            pipeline = ClusteringPipelineService(
                embedding_service=embedding_service,
                clustering_service=clustering_service,
                embedding_index_service=_FakeEmbeddingIndexService(),
                result_cache_service=_FakeResultCacheService([str(image_a), str(image_b)]),
            )

            result, metrics = pipeline.run(
                ClusteringRequest(
                    directory=tmp,
                    embedding_models=["dino"],
                    num_clusters=2,
                    clustering_backends=["cosine-kmeans"],
                    recursive=False,
                    similarity_mode="semantic",
                    outlier_policy="assign",
                    use_onnx=False,
                    reuse_result_cache=True,
                    use_embedding_cache_lookup=True,
                    source_paths=[str(image_a), str(image_b)],
                )
            )

            backend_metrics = result.metrics_by_key["dino::semantic::cosine-kmeans"]
            self.assertTrue(backend_metrics["result_cache_hit"])
            self.assertEqual(0.0, backend_metrics["clustering_time_s"])
            self.assertEqual(0.0, backend_metrics["backend_wall_time_s"])
            self.assertEqual(0.0, backend_metrics["backend_compute_time_s"])
            self.assertEqual(1.234, backend_metrics["cached_clustering_time_s"])
            self.assertEqual(0.0, metrics["clustering_time_s"])
            self.assertTrue(metrics["backend_cache_hit[dino::semantic::cosine-kmeans]"])
            self.assertEqual(1, clustering_service.prepare_calls)
            self.assertIn("dino::semantic::cosine-kmeans", result.cluster_explanations_by_key)
            self.assertEqual(2, result.cluster_explanations_by_key["dino::semantic::cosine-kmeans"][0].cluster_size)
            self.assertIn(str(image_a), embedding_service.last_fingerprints)

    def test_session_controller_classifies_optional_worker_notes_with_guidance(self):
        xet_level, xet_message = ClusteringSessionController._classify_stderr_line(
            "WARNING - huggingface_hub.file_download - Xet Storage is enabled for this repo, "
            "but the 'hf_xet' package is not installed. Falling back to regular HTTP download."
        )
        flash_level, flash_message = ClusteringSessionController._classify_stderr_line(
            "UserWarning: Torch was not compiled with flash attention."
        )
        info_level, info_message = ClusteringSessionController._classify_stderr_line(
            "2026-05-07 11:33:08,303 - INFO - ml.embeddings - Loaded model 'clip' on cuda (onnx=False)"
        )
        error_level, error_message = ClusteringSessionController._classify_stderr_line(
            "2026-05-07 11:33:08,303 - ERROR - apps.pyqt_production.worker - boom"
        )

        self.assertEqual(logging.WARNING, xet_level)
        self.assertIn("hf_xet", xet_message)
        self.assertIn("packaged runtime", xet_message)
        self.assertEqual(logging.WARNING, flash_level)
        self.assertIn("flash attention", flash_message.lower())
        self.assertIn("final exe", flash_message.lower())
        self.assertEqual(logging.INFO, info_level)
        self.assertIn("Loaded model", info_message)
        self.assertEqual(logging.ERROR, error_level)
        self.assertIn("boom", error_message)

    def test_session_controller_uses_frozen_worker_entrypoint_for_packaged_builds(self):
        with TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
                layout = activate_runtime_root("ProductionFrozenWorkerArgsTest")
                controller = ClusteringSessionController(layout)
                request_path = Path(tmp) / "request.json"

                with patch.object(sys, "frozen", False, create=True):
                    self.assertEqual(
                        ["-m", "apps.pyqt_production.worker", "--request-json", str(request_path)],
                        controller._worker_arguments(request_json=request_path),
                    )
                    self.assertEqual(
                        ["-m", "apps.pyqt_production.worker", "--daemon"],
                        controller._worker_arguments(daemon=True),
                    )

                with patch.object(sys, "frozen", True, create=True):
                    self.assertEqual(
                        ["--worker", "--request-json", str(request_path)],
                        controller._worker_arguments(request_json=request_path),
                    )
                    self.assertEqual(["--worker", "--daemon"], controller._worker_arguments(daemon=True))

    def test_session_controller_failed_start_is_terminal_and_removes_request(self):
        from apps.pyqt_production.worker_protocol import ProductionClusterRequest

        with TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
                layout = activate_runtime_root("ProductionFailedWorkerStartTest")
                controller = ClusteringSessionController(layout)
                controller.worker_program = str(Path(tmp) / "missing-worker-program")
                failures: list[str] = []
                running_states: list[bool] = []
                controller.failed.connect(failures.append)
                controller.running_changed.connect(running_states.append)
                request = ProductionClusterRequest(
                    directory=tmp,
                    embedding_models=["fast_preview"],
                    num_clusters=2,
                    clustering_backends=["cosine-kmeans"],
                    recursive=False,
                    similarity_mode="semantic",
                    outlier_policy="assign",
                    use_onnx=False,
                    reuse_result_cache=True,
                    use_embedding_cache_lookup=True,
                    preferred_execution_mode="cpu",
                )

                self.assertTrue(controller.start(request))
                self._wait_for(lambda: bool(failures), timeout_s=3.0)

                self.assertFalse(controller.is_running())
                self.assertIsNone(controller._request_file)
                self.assertEqual([True, False], running_states)
                self.assertFalse(list((layout.cache_dir / "tmp").glob("cluster_request_*.json")))

    def test_warm_session_controller_failed_start_emits_failure(self):
        from apps.pyqt_production.worker_protocol import ProductionClusterRequest

        with TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
                layout = activate_runtime_root("ProductionFailedWarmWorkerStartTest")
                controller = ClusteringSessionController(layout)
                controller.set_keep_worker_warm(True)
                controller.worker_program = str(Path(tmp) / "missing-warm-worker-program")
                failures: list[str] = []
                running_states: list[bool] = []
                controller.failed.connect(failures.append)
                controller.running_changed.connect(running_states.append)
                request = ProductionClusterRequest(
                    directory=tmp,
                    embedding_models=["fast_preview"],
                    num_clusters=2,
                    clustering_backends=["cosine-kmeans"],
                    recursive=False,
                    similarity_mode="semantic",
                    outlier_policy="assign",
                    use_onnx=False,
                    reuse_result_cache=True,
                    use_embedding_cache_lookup=True,
                    preferred_execution_mode="cpu",
                )

                self.assertTrue(controller.start(request))
                self._wait_for(lambda: bool(failures), timeout_s=3.0)

                self.assertFalse(controller.is_running())
                self.assertEqual([False], running_states)

    def test_model_download_failed_start_is_terminal_and_removes_request(self):
        from app.services.model_downloads import ModelDownloadItem
        from apps.pyqt_production.model_download_controller import ModelDownloadController

        with TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
                layout = activate_runtime_root("ProductionFailedModelDownloadStartTest")
                controller = ModelDownloadController(layout)
                controller.worker_program = str(Path(tmp) / "missing-model-worker-program")
                failures: list[str] = []
                running_states: list[bool] = []
                controller.failed.connect(failures.append)
                controller.running_changed.connect(running_states.append)

                self.assertTrue(controller.start([ModelDownloadItem("dino")]))
                self._wait_for(lambda: bool(failures), timeout_s=3.0)

                self.assertFalse(controller.is_running())
                self.assertIsNone(controller._request_file)
                self.assertEqual([True, False], running_states)
                self.assertFalse(list((layout.cache_dir / "tmp").glob("model_download_request_*.json")))

    def test_production_entrypoint_routes_worker_mode_without_starting_gui(self):
        from apps.pyqt_production import __main__ as production_main

        with patch("apps.pyqt_production.worker.main", return_value=12) as worker_main:
            self.assertEqual(12, production_main.main(["--worker", "--request-json", "request.json"]))

        worker_main.assert_called_once_with(["--request-json", "request.json"])

    def test_production_request_normalizes_legacy_similarity_mode(self):
        from apps.pyqt_production.worker_protocol import ProductionClusterRequest

        request = ProductionClusterRequest(
            directory="D:/images",
            embedding_models=["dino"],
            num_clusters=4,
            clustering_backends=["cosine-kmeans"],
            recursive=True,
            similarity_mode="near_duplicate",
            outlier_policy="assign",
            use_onnx=False,
            reuse_result_cache=True,
            use_embedding_cache_lookup=True,
        )
        self.assertEqual("semantic", request.similarity_mode)
        self.assertEqual(["semantic"], request.similarity_modes)
        self.assertFalse(request.generate_cluster_meanings)
        self.assertTrue(request.generate_cluster_explanations)
        self.assertEqual("auto", request.cluster_meaning_model)

        request = ProductionClusterRequest(
            directory="D:/images",
            embedding_models=["dino"],
            num_clusters=4,
            clustering_backends=["cosine-kmeans"],
            recursive=True,
            similarity_mode="semantic",
            similarity_modes=["semantic", "cosine"],
            outlier_policy="assign",
            use_onnx=False,
            reuse_result_cache=True,
            use_embedding_cache_lookup=True,
            generate_cluster_meanings=True,
            cluster_meaning_model="clip",
            batch_size_cpu=11,
            batch_size_gpu=37,
            preprocess_workers=6,
            vram_headroom_mb=1536,
        )
        self.assertEqual(["semantic", "cosine"], request.similarity_modes)
        self.assertTrue(request.generate_cluster_meanings)
        self.assertTrue(request.generate_cluster_explanations)
        self.assertEqual("clip", request.cluster_meaning_model)
        self.assertTrue(request.allow_model_downloads)
        self.assertEqual(11, request.as_dict()["batch_size_cpu"])
        self.assertEqual(37, request.as_dict()["batch_size_gpu"])
        self.assertEqual(6, request.as_dict()["preprocess_workers"])
        self.assertEqual(1536, request.as_dict()["vram_headroom_mb"])

    def test_session_controller_releases_completed_worker_payload(self):
        with TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
                layout = activate_runtime_root("ProductionPayloadReleaseTest")
                controller = ClusteringSessionController(layout)
                seen: list[dict] = []
                controller.completed.connect(lambda payload: seen.append(payload))
                controller._result_payload = {
                    "clusters_by_key": {
                        "fast_preview::semantic::cosine-kmeans": {
                            "0": [f"image_{index}.jpg" for index in range(1000)]
                        }
                    }
                }

                controller._emit_completed_payload()

                self.assertEqual(1, len(seen))
                self.assertIn("clusters_by_key", seen[0])
                self.assertIsNone(controller._result_payload)

    def test_cancelled_tag_context_refresh_releases_retained_thread_refs(self):
        from apps.pyqt_production.app import ProductionClusterApp

        with TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
                settings_mod._RUNTIME_BASE_DIR = None
                layout = activate_runtime_root("ProductionTagContextCancelTest")
                image_path = Path(tmp) / "image.jpg"
                image_path.write_bytes(b"fake")
                window = ProductionClusterApp(layout)
                self._wait_for_storage_idle(window)
                window.cluster_data = {"dino::semantic::hdbscan": {0: [str(image_path)]}}

                def _slow_load_tags(paths, *, progress_callback=None, cancel_check=None):
                    _ = (paths, progress_callback)
                    for _index in range(200):
                        if cancel_check is not None:
                            cancel_check()
                        time.sleep(0.005)
                    return {}

                with patch.object(window.image_tag_service, "load_tags_for_paths", side_effect=_slow_load_tags):
                    window._start_cluster_tag_context_refresh()
                    self._wait_for(lambda: window._tag_context_thread is not None, timeout_s=2.0)
                    window._cancel_cluster_tag_context_refresh()
                    self.assertIsNone(window._tag_context_thread)
                    self.assertTrue(window._retained_async_refs)
                    self._wait_for(lambda: not window._retained_async_refs, timeout_s=5.0)

                window.close()

    def test_production_download_decline_falls_back_to_bundled_model_asset(self):
        from apps.pyqt_production.app import ProductionClusterApp
        from apps.pyqt_production.worker_protocol import ProductionClusterRequest

        with TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
                settings_mod._RUNTIME_BASE_DIR = None
                layout = activate_runtime_root("ProductionDownloadFallbackTest")
                asset_dir = layout.model_assets_dir / "fast_preview"
                asset_dir.mkdir(parents=True, exist_ok=True)
                model_path = asset_dir / "model.onnx"
                model_path.write_bytes(b"onnx")
                (asset_dir / "metadata.json").write_text(
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
                window = ProductionClusterApp(layout)
                self._wait_for_storage_idle(window)
                window.settings_store.setValue("models/offline_mode", False)
                request = ProductionClusterRequest(
                    directory=tmp,
                    embedding_models=["dino"],
                    num_clusters=4,
                    clustering_backends=["cosine-kmeans"],
                    recursive=True,
                    similarity_mode="semantic",
                    similarity_modes=["semantic"],
                    outlier_policy="assign",
                    use_onnx=False,
                    reuse_result_cache=True,
                    use_embedding_cache_lookup=True,
                    generate_cluster_meanings=True,
                    cluster_meaning_model="auto",
                )

                with patch("apps.pyqt_production.app.confirmBox", return_value=False):
                    resolved = window._prepare_request_model_downloads(request)

                self.assertIsNotNone(resolved)
                self.assertEqual(["fast_preview"], resolved.embedding_models)
                self.assertTrue(resolved.use_onnx)
                self.assertFalse(resolved.allow_model_downloads)
                self.assertFalse(resolved.generate_cluster_meanings)
                window.close()

    def test_production_download_approval_preserves_selected_models(self):
        from apps.pyqt_production.app import ProductionClusterApp
        from apps.pyqt_production.worker_protocol import ProductionClusterRequest

        with TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
                settings_mod._RUNTIME_BASE_DIR = None
                layout = activate_runtime_root("ProductionDownloadApproveTest")
                window = ProductionClusterApp(layout)
                self._wait_for_storage_idle(window)
                window.settings_store.setValue("models/offline_mode", False)
                request = ProductionClusterRequest(
                    directory=tmp,
                    embedding_models=["dino"],
                    num_clusters=4,
                    clustering_backends=["cosine-kmeans"],
                    recursive=True,
                    similarity_mode="semantic",
                    similarity_modes=["semantic"],
                    outlier_policy="assign",
                    use_onnx=False,
                    reuse_result_cache=True,
                    use_embedding_cache_lookup=True,
                    generate_cluster_meanings=False,
                    cluster_meaning_model="auto",
                )

                with patch("apps.pyqt_production.app.confirmBox", return_value=True):
                    resolved = window._prepare_request_model_downloads(request)

                self.assertIsNotNone(resolved)
                self.assertEqual(["dino"], resolved.embedding_models)
                self.assertTrue(resolved.allow_model_downloads)
                window.close()

    def test_missing_facenet_scan_request_starts_visible_shared_download_after_confirmation(self):
        from apps.pyqt_production.app import ProductionClusterApp

        with TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
                settings_mod._RUNTIME_BASE_DIR = None
                layout = activate_runtime_root("ProductionFaceModelDownloadTest")
                window = ProductionClusterApp(layout)
                self._wait_for_storage_idle(window)

                with (
                    patch.object(window.model_asset_service, "model_available_without_download", return_value=False),
                    patch.object(window.model_download_controller, "is_running", return_value=False),
                    patch.object(window.model_download_controller, "start", return_value=True) as start_download,
                    patch("apps.pyqt_production.app.confirmBox", return_value=True) as confirm_download,
                ):
                    window._request_face_model_download("facenet")

                requested_items = start_download.call_args.args[0]
                self.assertEqual(["facenet"], [item.model_name for item in requested_items])
                self.assertEqual("facenet", window._active_post_install_model_name)
                prompt_text = str(confirm_download.call_args.args[1])
                self.assertIn("visible in Jobs", prompt_text)
                self.assertIn("cancellable", prompt_text)
                window._active_post_install_model_name = None
                window.close()

    def test_production_offline_model_mode_skips_download_prompt_and_falls_back(self):
        from apps.pyqt_production.app import ProductionClusterApp
        from apps.pyqt_production.worker_protocol import ProductionClusterRequest

        with TemporaryDirectory() as tmp:
            runtime_root = Path(tmp) / "runtime"
            with patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": str(runtime_root)}, clear=False):
                settings_mod._RUNTIME_BASE_DIR = None
                layout = activate_runtime_root("ProductionOfflineModelTest")
                asset_dir = layout.model_assets_dir / "fast_preview"
                asset_dir.mkdir(parents=True)
                model_path = asset_dir / "model.onnx"
                model_path.write_bytes(b"onnx")
                (asset_dir / "metadata.json").write_text(
                    json.dumps({"model_name": "fast_preview", "input_size": [224, 224], "sha256": sha256_file(model_path)}),
                    encoding="utf-8",
                )
                window = ProductionClusterApp(layout)
                self._wait_for_storage_idle(window)
                window.settings_store.setValue("models/offline_mode", True)
                request = ProductionClusterRequest(
                    directory=str(runtime_root),
                    embedding_models=["dino"],
                    num_clusters=4,
                    clustering_backends=["cosine-kmeans"],
                    recursive=True,
                    similarity_mode="semantic",
                    outlier_policy="assign",
                    use_onnx=False,
                    reuse_result_cache=True,
                    use_embedding_cache_lookup=True,
                    generate_cluster_meanings=False,
                )
                with patch("apps.pyqt_production.app.confirmBox", side_effect=AssertionError("prompt should not open")):
                    resolved = window._prepare_request_model_downloads(request)

                self.assertIsNotNone(resolved)
                self.assertEqual(["fast_preview"], resolved.embedding_models)
                self.assertFalse(resolved.allow_model_downloads)
                window.settings_store.setValue("models/offline_mode", False)
                window.close()

    def test_runtime_diagnostics_reports_optional_accelerator_guidance(self):
        service = RuntimeCapabilityService()
        capabilities = RuntimeCapabilities(
            torch_version="2.2.2+cu121",
            torch_cuda_build=True,
            torch_cuda_available=True,
            cuda_device_count=1,
            cuda_device_name="Test GPU",
            cuda_total_memory_mb=8192,
            onnx_available=True,
            onnx_version="1.18.0",
            onnx_providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
        )
        versions = {
            "torch": "2.2.2+cu121",
            "onnxruntime": "1.18.0",
            "onnx": "1.16.0",
            "onnxruntime-gpu": "1.18.0",
            "hf_xet": "",
            "hf-xet": "",
            "flash-attn": "",
            "flash_attn": "",
        }

        with patch.object(service, "detect", return_value=capabilities), patch(
            "infra.runtime._package_version", side_effect=lambda name: versions.get(name, "")
        ), patch("infra.runtime._module_available", return_value=False), patch(
            "infra.runtime._flash_sdp_enabled", return_value=False
        ):
            details = service.diagnostics("cuda")

        optional_details = details.get("optional_details") or {}
        remediation = "\n".join(details.get("remediation") or [])
        self.assertFalse(optional_details.get("hf_xet_installed"))
        self.assertFalse(optional_details.get("flash_sdp_enabled"))
        self.assertIn("hf_xet", remediation)
        self.assertIn("packaged exe", remediation)
        self.assertIn("Flash SDP", remediation)

    def test_production_gallery_uses_one_overflow_without_legacy_buttons(self):
        from apps.pyqt_production.app import ProductionClusterApp, RUNTIME_LAYOUT

        window = ProductionClusterApp(RUNTIME_LAYOUT)
        window.show()
        APP.processEvents()

        self.assertEqual("basic", window._clustering_mode)
        self.assertEqual("basic", window.gallery_pane.inspector_display_mode)
        self.assertTrue(window.gallery_pane.open_folder_button.isVisible())
        self.assertTrue(window.gallery_pane.actions_menu_button.isVisible())
        self.assertTrue(window.gallery_pane.file_ops_menu.menuAction().isVisible())
        self.assertFalse(window.gallery_pane.metadata_menu.menuAction().isVisible())
        self.assertFalse(window.gallery_pane.selected_tags_button.isVisible())
        for legacy_name in (
            "copy_paths_button", "export_paths_button", "copy_button", "move_button",
            "delete_button", "exif_button", "tags_button", "retry_failed_button",
            "metadata_menu_button", "file_ops_menu_button", "more_menu_button",
        ):
            self.assertFalse(hasattr(window.gallery_pane, legacy_name))
        self.assertFalse(window.tag_manager_button.isVisible())
        self.assertFalse(window.suggest_tags_button.isVisible())
        self.assertTrue(window.cluster_pane.details_scroll.isHidden())
        self.assertTrue(window.cluster_pane.meaning_label.isHidden())
        self.assertTrue(window.cluster_pane.shape_widget.isHidden())
        self.assertTrue(window.cluster_pane.basis_label.isHidden())

        window.set_clustering_mode("advanced")
        APP.processEvents()

        self.assertTrue(window.gallery_pane.selected_tags_button.isVisible())
        self.assertTrue(window.gallery_pane.actions_menu_button.isVisible())
        self.assertTrue(window.gallery_pane.metadata_menu.menuAction().isVisible())
        self.assertTrue(window.gallery_pane.file_ops_menu.menuAction().isVisible())
        self.assertEqual("advanced", window.gallery_pane.inspector_display_mode)
        self.assertFalse(window.tag_manager_button.isVisible())
        self.assertFalse(window.suggest_tags_button.isVisible())
        self.assertTrue(window.cluster_actions_menu_action.isVisible())
        self.assertFalse(window.cluster_pane.details_scroll.isHidden())
        self.assertFalse(window.cluster_pane.meaning_label.isHidden())
        self.assertFalse(window.cluster_pane.shape_widget.isHidden())
        self.assertFalse(window.cluster_pane.basis_label.isHidden())
        window.close()

    def test_production_clustering_controls_use_curated_model_and_backend_surface(self):
        from apps.pyqt_production.app import ProductionClusterApp, RUNTIME_LAYOUT

        window = ProductionClusterApp(RUNTIME_LAYOUT)
        models = set(window.clustering_pane.embedding_checkboxes)
        backends = set(window.clustering_pane.backend_checkboxes)

        self.assertEqual(
            {"fast_preview", "mobileclip", "dino", "dinov2_base", "clip", "openclip", "siglip", "resnet"},
            models,
        )
        self.assertNotIn("convnext", models)
        self.assertNotIn("phash_embedding", models)
        self.assertNotIn("facenet", models)
        self.assertNotIn("dino_large", models)
        self.assertEqual({"cosine-kmeans", "hdbscan", "graph"}, backends)
        self.assertNotIn("faiss", backends)
        self.assertNotIn("sklearn", backends)

        window.clustering_pane.apply_state(
            {
                "embedding_models": ["dino_large"],
                "clustering_backends": ["faiss"],
            }
        )
        self.assertEqual(["dinov2_base"], window.clustering_pane.selected_embedding_models())
        self.assertEqual(["cosine-kmeans"], window.clustering_pane.selected_clustering_backends())

        window.clustering_pane.apply_state(
            {
                "embedding_models": ["convnext"],
                "clustering_backends": ["faiss"],
            }
        )
        self.assertEqual(["resnet"], window.clustering_pane.selected_embedding_models())

        window.clustering_pane.apply_state(
            {
                "embedding_models": ["clip", "convnext"],
                "clustering_backends": ["graph", "faiss"],
            }
        )
        self.assertEqual(["clip"], window.clustering_pane.selected_embedding_models())
        self.assertEqual(["graph"], window.clustering_pane.selected_clustering_backends())

    def test_production_presets_keep_advanced_technical_controls_visible(self):
        from apps.pyqt_production.app import ProductionClusterApp, RUNTIME_LAYOUT

        window = ProductionClusterApp(RUNTIME_LAYOUT)
        window.show()
        window.set_clustering_mode("advanced")
        APP.processEvents()

        pane = window.clustering_pane
        self.assertEqual("balanced", pane.preset_combo.currentData())
        self.assertTrue(pane.technical_panel.isVisible())
        self.assertTrue(pane.backend_checkboxes["hdbscan"].isVisible())

        for preset in ("fast_preview", "high_quality"):
            pane.preset_combo.setCurrentIndex(pane.preset_combo.findData(preset))
            APP.processEvents()
            self.assertTrue(pane.technical_panel.isVisible())
            self.assertTrue(pane.backend_checkboxes["hdbscan"].isVisible())

        window.close()
        window.close()

    def test_production_shell_request_enforces_curated_scope(self):
        from apps.pyqt_production.app import ProductionClusterApp, RUNTIME_LAYOUT

        window = ProductionClusterApp(RUNTIME_LAYOUT)
        with patch.object(window.clustering_pane, "selected_embedding_models", return_value=["convnext"]), patch.object(
            window.clustering_pane,
            "selected_clustering_backends",
            return_value=["faiss"],
        ):
            request = window._build_request()

        self.assertEqual(["resnet"], request.embedding_models)
        self.assertEqual(["cosine-kmeans"], request.clustering_backends)
        self.assertTrue(request.generate_cluster_explanations)
        self.assertFalse(request.generate_cluster_meanings)
        window.close()

    def test_production_shell_uses_shared_authoritative_ui_modules(self):
        from apps.pyqt_production.app import ProductionClusterApp, RUNTIME_LAYOUT

        window = ProductionClusterApp(RUNTIME_LAYOUT)
        self.assertTrue(window.gallery_pane.__class__.__module__.startswith("ui."))
        self.assertTrue(window.cluster_pane.__class__.__module__.startswith("ui."))
        self.assertTrue(window.clustering_pane.__class__.__module__.startswith("ui."))
        self.assertTrue(window.footer_bar.__class__.__module__.startswith("ui."))
        window.close()

    def test_production_settings_dialog_uses_production_owned_copy(self):
        from apps.pyqt_production.app import RUNTIME_LAYOUT
        from apps.pyqt_production.settings_dialog import ProductionSettingsDialog
        from app.services.cache_maintenance import CacheClearResult, CacheUsageSummary

        store = QSettings("ClusterLensTests", "ProductionSettingsDialog")
        dialog = ProductionSettingsDialog(
            store,
            RuntimeCapabilityService(),
            describe_rebuildable_caches=lambda: CacheUsageSummary(
                cache_root=str(RUNTIME_LAYOUT.cache_dir),
                target_bytes={"embeddings.sqlite3": 0, "cluster_results/": 0},
                total_bytes=0,
            ),
            clear_rebuildable_caches=lambda: CacheClearResult((), 0, ()),
            can_clear_rebuildable_caches=lambda: True,
            runtime_layout=RUNTIME_LAYOUT,
            support_metadata_provider=lambda: {},
        )
        self._wait_for(lambda: dialog._cache_usage_thread is None, timeout_s=5.0)
        combined_text = "\n".join(
            [
                dialog.runtime_text.toPlainText(),
                dialog.cache_usage_text.toPlainText(),
                dialog.cache_status_label.text(),
            ]
        ).lower()
        self.assertEqual("apps.pyqt_production.settings_dialog", ProductionSettingsDialog.__module__)
        self.assertNotIn("face/search", combined_text)
        self.assertNotIn("convnext", combined_text)
        dialog.close()

    def test_production_pane_hide_buttons_and_toolbar_toggles_work(self):
        from apps.pyqt_production.app import ProductionClusterApp, RUNTIME_LAYOUT

        window = ProductionClusterApp(RUNTIME_LAYOUT)
        window.show()
        window.set_clustering_mode("advanced")
        APP.processEvents()

        window.source_pane.hide_requested.emit()
        APP.processEvents()
        self.assertFalse(window.source_pane.isVisible())
        self.assertFalse(window.source_toggle.isChecked())

        window.clustering_pane.hide_requested.emit()
        APP.processEvents()
        self.assertFalse(window.clustering_pane.isVisible())
        self.assertFalse(window.controls_toggle.isChecked())

        window.cluster_pane.hide_requested.emit()
        APP.processEvents()
        self.assertFalse(window.cluster_pane.isVisible())
        self.assertFalse(window.details_toggle.isChecked())

        window.source_toggle.setChecked(True)
        window.controls_toggle.setChecked(True)
        window.details_toggle.setChecked(True)
        APP.processEvents()

        self.assertTrue(window.source_pane.isVisible())
        self.assertTrue(window.clustering_pane.isVisible())
        self.assertTrue(window.cluster_pane.isVisible())
        window.close()

    def test_production_suggested_cluster_tags_write_to_app_tag_db(self):
        from apps.pyqt_production.app import ProductionClusterApp
        from app.services.cluster_meanings import ClusterMeaning, ClusterMeaningLabel

        with TemporaryDirectory() as tmp:
            runtime_root = Path(tmp) / "runtime"
            images_root = Path(tmp) / "images"
            images_root.mkdir(parents=True, exist_ok=True)
            image_paths = [images_root / "a.jpg", images_root / "b.jpg"]
            for index, image_path in enumerate(image_paths, start=1):
                Image.new("RGB", (24, 24), (index * 10, 30, 40)).save(image_path)

            with patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": str(runtime_root)}, clear=False):
                settings_mod._RUNTIME_BASE_DIR = None
                layout = activate_runtime_root("ProductionSuggestTagsTest")
                window = ProductionClusterApp(layout)
                self._wait_for_storage_idle(window)
                window.set_clustering_mode("advanced")
                key = "dino::semantic::hdbscan"
                paths = [str(path) for path in image_paths]
                window.cluster_data = {key: {3: paths}}
                window.cluster_meanings = {
                    key: {
                        3: ClusterMeaning(
                            cluster_id=3,
                            labels=(
                                ClusterMeaningLabel("food", "a photo of food", 0.31),
                                ClusterMeaningLabel("party", "a party photo", 0.27),
                            ),
                            confidence="Medium",
                            explanation_model="clip",
                            image_count_used=2,
                            status="ok",
                        )
                    }
                }
                window.cluster_pane.update_clusters(
                    window.cluster_data,
                    cluster_meanings=window.cluster_meanings,
                )
                window.cluster_pane.on_cluster_selected(window.cluster_pane._grid_model.index(0, 0))

                with patch("apps.pyqt_production.app.confirmBox", return_value=True), patch(
                    "apps.pyqt_production.app.infoBox"
                ), patch.object(window, "_on_gallery_metadata_changed") as metadata_changed:
                    window.apply_cluster_tag_suggestions()
                    self._wait_for(
                        lambda: "food" in window.image_tag_service.load_tags_for_paths(paths, import_missing_exif=False)[paths[0]],
                        timeout_s=5.0,
                    )
                    self._wait_for(lambda: metadata_changed.called, timeout_s=5.0)

                tags_by_path = window.image_tag_service.load_tags_for_paths(paths, import_missing_exif=False)
                self.assertEqual(("food", "party"), tags_by_path[paths[0]])
                self.assertEqual(("food", "party"), tags_by_path[paths[1]])
                window.close()

    def test_production_preflight_moves_tag_filter_discovery_off_gui_thread(self):
        from apps.pyqt_production.app import ProductionClusterApp
        from app.services.discovery import ImageDiscoveryService

        with TemporaryDirectory() as tmp:
            runtime_root = Path(tmp) / "runtime"
            images_root = Path(tmp) / "images"
            images_root.mkdir(parents=True, exist_ok=True)
            image_paths = [images_root / "a.jpg", images_root / "b.jpg"]
            for index, image_path in enumerate(image_paths, start=1):
                Image.new("RGB", (24, 24), (index * 10, 30, 40)).save(image_path)

            with patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": str(runtime_root)}, clear=False):
                settings_mod._RUNTIME_BASE_DIR = None
                layout = activate_runtime_root("ProductionPreflightTest")
                window = ProductionClusterApp(layout)
                self._wait_for_storage_idle(window)
                window.source_pane.set_selected_directory(str(images_root))
                window.image_tag_service.apply_tag_edit([str(path) for path in image_paths], add_tags=["selfie"], mirror_to_exif=False)
                window.clustering_pane.tag_filter_field.setText("selfie")
                window.clustering_pane.tag_match_combobox.setCurrentText("Any")

                discovery_thread_flags: list[bool] = []
                captured_requests: list[object] = []
                original_discover = ImageDiscoveryService.discover_result

                def _discover(service, directory, recursive=None, progress_callback=None, cancel_check=None):
                    discovery_thread_flags.append(QThread.currentThread() is APP.thread())
                    time.sleep(0.05)
                    return original_discover(
                        service,
                        directory,
                        recursive=recursive,
                        progress_callback=progress_callback,
                        cancel_check=cancel_check,
                    )

                def _start(request):
                    captured_requests.append(request)
                    return True

                with patch("apps.pyqt_production.app.confirmBox", return_value=True), patch.object(
                    ImageDiscoveryService, "discover_result", autospec=True, side_effect=_discover
                ), patch.object(
                    window,
                    "_prepare_request_model_downloads",
                    side_effect=lambda request: replace(request, allow_model_downloads=False),
                ), patch.object(window.session_controller, "start", side_effect=_start):
                    window.run_clustering()
                    APP.processEvents()
                    self.assertIsNotNone(window._preflight_thread)
                    self.assertFalse(window.footer_bar.clear_storage_button.isEnabled())
                    self._wait_for(lambda: bool(captured_requests), timeout_s=5.0)

                self.assertTrue(discovery_thread_flags)
                self.assertFalse(any(discovery_thread_flags))
                self.assertEqual([str(path) for path in image_paths], captured_requests[0].source_paths)
                window._set_running_state(False)
                window.close()

    def test_production_footer_storage_summary_and_clear_action(self):
        from apps.pyqt_production.app import ProductionClusterApp

        with TemporaryDirectory() as tmp:
            runtime_root = Path(tmp) / "runtime"
            with patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": str(runtime_root)}, clear=False):
                settings_mod._RUNTIME_BASE_DIR = None
                layout = activate_runtime_root("ProductionStorageFooterTest")
                (layout.cache_dir / "embeddings.sqlite3").write_bytes(b"embed")
                (layout.benchmarks_dir / "bench.json").write_text("{}", encoding="utf-8")
                (layout.support_dir / "bundle.zip").write_bytes(b"bundle")

                window = ProductionClusterApp(layout)
                self._wait_for_storage_idle(window)

                self.assertIn("Storage:", window.footer_bar.storage_label._full_text)
                self.assertIn("Tag database", window.footer_bar.storage_label.toolTip())
                self.assertIn("Runtime temp files", window.footer_bar.storage_label.toolTip())
                self.assertEqual("Clear rebuildable data", window.footer_bar.clear_storage_button.text())
                self.assertFalse(window.footer_bar.clear_storage_button.isVisible())
                self.assertTrue(window.footer_bar.clear_storage_button.isEnabled())

                with patch("apps.pyqt_production.app.confirmBox", return_value=True):
                    window._request_runtime_storage_clear()
                    self._wait_for_storage_idle(window)

                self.assertTrue(layout.cache_dir.exists())
                self.assertTrue((layout.cache_dir / "image_tags.sqlite3").exists())
                self.assertFalse((layout.benchmarks_dir / "bench.json").exists())
                self.assertFalse((layout.support_dir / "bundle.zip").exists())
                self.assertIn("clear", window.footer_bar.status_label.toolTip().lower())
                window.close()


if __name__ == "__main__":
    unittest.main()
