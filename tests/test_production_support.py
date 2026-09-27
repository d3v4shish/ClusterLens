import json
import os
import sys
import time
import unittest
import logging
import gc
import signal
import subprocess
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
from PIL import Image
from PyQt6.QtCore import QCoreApplication, QEvent, QProcess, QSettings, QThread, QTimer
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
from app.selection import SelectionTarget
from app.services.clustering_pipeline import ClusteringPipelineService, ClusteringRequest
from app.services.model_assets import ModelDownloadPlan, sha256_file
from infra.runtime import ExecutionPolicy, RuntimeCapabilities, RuntimeCapabilityService
from infra.qt_diagnostics import append_qt_diagnostic
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
        self._runtime_environment_snapshot = {
            key: os.environ.get(key)
            for key in ("CLUSTERLENS_RUNTIME_ROOT", "IMAGE_CLUSTERING_APP_DIR")
        }
        # ``activate_runtime_root`` deliberately sets both process variables.
        # Start each test clean so one temporary runtime cannot leak into the
        # next test's fixture or into a developer's local cache.
        os.environ.pop("CLUSTERLENS_RUNTIME_ROOT", None)
        os.environ.pop("IMAGE_CLUSTERING_APP_DIR", None)
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
            lambda: (
                window._storage_usage_job is None
                and window._storage_clear_job is None
                and window._startup_maintenance_job is None
                and window._model_storage_recovery_job is None
                and window._startup_readiness_job is None
                and bool(getattr(window, "_startup_maintenance_complete", True))
                and bool(getattr(window, "_model_storage_recovery_complete", True))
                and not window.job_manager.active_jobs()
                and not window.job_presentation._refresh_timer.isActive()
            ),
            timeout_s=10.0,
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
        for key, value in self._runtime_environment_snapshot.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        store = QSettings(PRODUCTION_QSETTINGS_ORG, PRODUCTION_QSETTINGS_APP)
        store.clear()
        for key, value in self._production_settings_snapshot.items():
            store.setValue(key, value)
        store.sync()
        # Production windows own large widget trees and use deferred Qt
        # deletion. Flush them between cases so later global stylesheet tests
        # do not repolish every window created earlier in this process.
        for widget in list(APP.topLevelWidgets()):
            try:
                if widget.close():
                    widget.deleteLater()
            except RuntimeError:
                continue
        for _ in range(4):
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            APP.processEvents()
        gc.collect()

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

    def test_production_close_shuts_down_modeless_jobs_monitor(self):
        from PyQt6 import sip

        from apps.pyqt_production.app import ProductionClusterApp, RUNTIME_LAYOUT

        window = ProductionClusterApp(RUNTIME_LAYOUT)
        window.show()
        window._open_jobs_dialog()
        APP.processEvents()
        dialog = window.jobs_widget._jobs_dialog
        self.assertIsNotNone(dialog)
        self.assertTrue(dialog.isVisible())

        window.close()
        APP.processEvents()

        self.assertTrue(window.jobs_widget._shutting_down)
        self.assertTrue(dialog._shutting_down)
        self.assertTrue(sip.isdeleted(dialog) or not dialog.isVisible())

    def test_interactive_close_retains_a_stuck_worker_without_blocking_and_reopens(self):
        from apps.pyqt_production import app as production_app
        from apps.pyqt_production.app import ProductionClusterApp
        from ui.async_job import AsyncJob, start_job_in_thread

        entered = Event()
        release = Event()

        def _blocked_work(_progress, _cancel_check):
            entered.set()
            if not release.wait(timeout=5.0):
                raise RuntimeError("close-under-load fixture was not released")
            return "drained"

        with TemporaryDirectory() as tmp:
            with patch.dict(
                os.environ,
                {
                    "IMAGE_CLUSTERING_APP_DIR": tmp,
                    "CLUSTERLENS_PACKAGED_LAUNCH_SMOKE": "close-under-load-test",
                },
                clear=False,
            ):
                settings_mod._RUNTIME_BASE_DIR = None
                window = ProductionClusterApp(activate_runtime_root("ProductionAsyncCloseTest"))
                replacement = None
                session_controller = window.session_controller
                model_download_controller = window.model_download_controller
                try:
                    window.show()
                    job = AsyncJob(_blocked_work)
                    thread = start_job_in_thread(job)
                    window._storage_usage_job = job
                    window._storage_usage_thread = thread
                    self._wait_for(entered.is_set)

                    qt_acknowledged: list[bool] = []
                    with patch.dict(os.environ, {"PYTEST_CURRENT_TEST": ""}, clear=False):
                        window.close()
                        QTimer.singleShot(0, lambda: qt_acknowledged.append(True))
                        self._wait_for(lambda: bool(qt_acknowledged))

                        self.assertIn(window, production_app._CLOSE_PENDING_WINDOWS)
                        self.assertTrue(
                            any(retained_thread is thread for _job, retained_thread in window._retained_async_refs)
                        )
                        self.assertTrue(thread.isRunning())
                        window.close()
                        window.close()
                        self.assertEqual(1, production_app._CLOSE_PENDING_WINDOWS.count(window))
                        self.assertTrue(window._close_retry_scheduled)

                        release.set()

                        def _thread_stopped() -> bool:
                            try:
                                return not thread.isRunning()
                            except RuntimeError:
                                return True

                        self._wait_for(
                            lambda: (
                                _thread_stopped()
                                and window not in production_app._CLOSE_PENDING_WINDOWS
                            )
                        )

                    self.assertIsNone(session_controller._process)
                    self.assertIsNone(model_download_controller._process)

                    replacement = ProductionClusterApp(
                        activate_runtime_root("ProductionAsyncCloseRestartTest")
                    )
                    replacement.show()
                    APP.processEvents()
                    self.assertTrue(replacement.isVisible())
                finally:
                    release.set()
                    if replacement is not None:
                        replacement.close()
                    try:
                        window.close()
                    except RuntimeError:
                        pass

    def test_close_under_load_matrix_drains_owned_work_and_preserves_commit_boundaries(self):
        from apps.pyqt_production import app as production_app
        from apps.pyqt_production.app import ProductionClusterApp
        from infra.cancel import Cancelled
        from ui.async_job import AsyncJob, start_job_in_thread
        from ui.search_pane import SearchPane

        class _BlockedDecodeThread(QThread):
            def __init__(self, entered: Event, release: Event):
                super().__init__()
                self._entered = entered
                self._release = release
                self.setObjectName("CloseMatrixDecodeThread")

            def run(self):
                self._entered.set()
                if not self._release.wait(timeout=10.0):
                    raise RuntimeError("decode close fixture was not released")

        entered = {name: Event() for name in ("probe", "decode", "index", "inference", "database", "journal")}
        release = {name: Event() for name in entered}

        with TemporaryDirectory() as tmp:
            environment = {
                "IMAGE_CLUSTERING_APP_DIR": tmp,
                "CLUSTERLENS_PACKAGED_LAUNCH_SMOKE": "close-matrix-test",
            }
            with patch.dict(os.environ, environment, clear=False):
                settings_mod._RUNTIME_BASE_DIR = None
                layout = activate_runtime_root("ProductionCloseMatrixTest")
                window = ProductionClusterApp(layout)
                replacement = None
                journal_marker = Path(tmp) / "journal-committed.json"
                database_marker = Path(tmp) / "database-uncommitted.json"
                download_partial = Path(tmp) / "model.partial"
                download_ready = Path(tmp) / "model.ready"

                def _run_for(name: str, *, commit_path: Path | None = None):
                    def _run(_progress, cancel_check):
                        entered[name].set()
                        if not release[name].wait(timeout=10.0):
                            raise RuntimeError(f"{name} close fixture was not released")
                        if commit_path is not None:
                            commit_path.write_text(name, encoding="utf-8")
                            return SimpleNamespace(completion_survives_cancellation=True)
                        if cancel_check():
                            raise Cancelled()
                        return name

                    return _run

                def _worker(name: str, *, commit_path: Path | None = None):
                    job = AsyncJob(_run_for(name, commit_path=commit_path))
                    return job, start_job_in_thread(job)

                try:
                    window.show()

                    probe_job, probe_thread = _worker("probe")
                    window._startup_readiness_job = probe_job
                    window._startup_readiness_thread = probe_thread

                    database_job, database_thread = _worker("database")
                    window._storage_clear_job = database_job
                    window._storage_clear_thread = database_thread

                    inference_job, inference_thread = _worker("inference")
                    window._tag_suggestion_job = inference_job
                    window._tag_suggestion_thread = inference_thread

                    people = SearchPane(window, enabled_tabs=[], external_results=True)
                    people._start_job("Indexing fixture", _run_for("index"), lambda _result: None)
                    window.faces_pane = people

                    journal_job, journal_thread = _worker("journal", commit_path=journal_marker)
                    window.gallery_pane._active_action_job = journal_job
                    window.gallery_pane._active_action_thread = journal_thread

                    decode_thread = _BlockedDecodeThread(entered["decode"], release["decode"])
                    window.gallery_pane.loader_threads = [decode_thread]
                    decode_thread.start()

                    process = QProcess(window.model_download_controller)
                    process.setProgram(sys.executable)
                    process.setArguments(
                        [
                            "-c",
                            (
                                "import pathlib, signal, sys, time; "
                                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                                "pathlib.Path(sys.argv[1]).write_text('partial', encoding='utf-8'); "
                                "time.sleep(30); "
                                "pathlib.Path(sys.argv[2]).write_text('ready', encoding='utf-8')"
                            ),
                            str(download_partial),
                            str(download_ready),
                        ]
                    )
                    process.finished.connect(
                        lambda exit_code, exit_status, process=process: window.model_download_controller._on_finished(
                            process, exit_code, exit_status
                        )
                    )
                    window.model_download_controller._process = process
                    process.start()

                    self._wait_for(lambda: all(event.is_set() for event in entered.values()))
                    self._wait_for(download_partial.exists)

                    qt_acknowledged: list[bool] = []
                    with patch.dict(os.environ, {"PYTEST_CURRENT_TEST": ""}, clear=False):
                        window.close()
                        window.close()
                        QTimer.singleShot(0, lambda: qt_acknowledged.append(True))
                        self._wait_for(lambda: bool(qt_acknowledged))
                        self.assertEqual(1, production_app._CLOSE_PENDING_WINDOWS.count(window))

                        for name in ("probe", "decode", "index", "inference", "database"):
                            release[name].set()
                        self._wait_for(lambda: window.model_download_controller._process is None)
                        APP.processEvents()
                        self.assertIn(window, production_app._CLOSE_PENDING_WINDOWS)
                        self.assertFalse(journal_marker.exists())
                        self.assertFalse(database_marker.exists())
                        self.assertTrue(download_partial.exists())
                        self.assertFalse(download_ready.exists())

                        release["journal"].set()
                        deadline = time.monotonic() + 5.0
                        while window in production_app._CLOSE_PENDING_WINDOWS and time.monotonic() < deadline:
                            APP.processEvents()
                            time.sleep(0.01)
                        if window in production_app._CLOSE_PENDING_WINDOWS:
                            def _running(thread) -> bool:
                                try:
                                    return bool(thread is not None and thread.isRunning())
                                except RuntimeError:
                                    return False

                            self.fail(
                                "close matrix did not drain: "
                                + repr(
                                    {
                                        "shell_retained": [
                                            _running(thread) for _job, thread in window._retained_async_refs
                                        ],
                                        "people_active": _running(people._active_thread),
                                        "gallery_action": _running(window.gallery_pane._active_action_thread),
                                        "gallery_loaders": [
                                            _running(thread)
                                            for thread in window.gallery_pane._retained_loader_threads
                                        ],
                                        "download": window.model_download_controller._process is not None,
                                        "close_retry": window._close_retry_scheduled,
                                        "shutdown_results": {
                                            "model": window.model_download_controller.shutdown(0),
                                            "session": window.session_controller.shutdown(0),
                                            "gallery": window.gallery_pane.shutdown_jobs(timeout_ms=0),
                                            "photo_gallery": window.photo_gallery.shutdown_jobs(timeout_ms=0),
                                            "faces": people.shutdown_jobs(timeout_ms=0),
                                            "cluster": window.cluster_pane.shutdown_jobs(timeout_ms=0),
                                        },
                                    }
                                )
                            )

                    self.assertTrue(journal_marker.exists())
                    self.assertEqual("journal", journal_marker.read_text(encoding="utf-8"))
                    self.assertFalse(database_marker.exists())
                    self.assertFalse(download_ready.exists())

                    replacement = ProductionClusterApp(
                        activate_runtime_root("ProductionCloseMatrixRestartTest")
                    )
                    replacement.show()
                    APP.processEvents()
                    self.assertTrue(replacement.isVisible())
                finally:
                    for event in release.values():
                        event.set()
                    if replacement is not None:
                        replacement.close()
                    try:
                        window.close()
                    except RuntimeError:
                        pass

    def test_production_library_tags_and_names_share_coordinator_and_queue_declared_work(self):
        from apps.pyqt_production.app import ProductionClusterApp
        from app.path_scope import PathScope
        from ui.job_manager import JobManager
        from ui.gallery_pane import GalleryPane
        from ui.library_pane import LibraryPane
        from ui.names_pane import NamesPane
        from ui.async_job import AsyncJob
        from ui.photo_inspector_dialog import PhotoInspectorDialog
        from ui.work_coordinator import JobSpec, WorkCoordinator

        with TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
                settings_mod._RUNTIME_BASE_DIR = None
                layout = activate_runtime_root("ProductionCoordinatedLibraryTest")
                window = ProductionClusterApp(layout)
                try:
                    self.assertIs(window.library_pane.work_coordinator, window.work_coordinator)
                    self.assertIs(window.tags_pane.work_coordinator, window.work_coordinator)
                    self.assertIs(window.gallery_pane.work_coordinator, window.work_coordinator)
                    self.assertIs(window.photo_gallery._actions.work_coordinator, window.work_coordinator)
                    self.assertIs(window.library_pane.gallery.work_coordinator, window.work_coordinator)
                    self.assertIs(window.library_pane.timeline_gallery._actions.work_coordinator, window.work_coordinator)
                    self.assertIs(window.tags_pane.gallery.work_coordinator, window.work_coordinator)
                    window._build_names_workspace_widget()
                    self.assertIs(window.names_pane.work_coordinator, window.work_coordinator)
                    self.assertIs(window.names_pane.gallery.work_coordinator, window.work_coordinator)
                finally:
                    window.close()

                manager = JobManager()
                coordinator = WorkCoordinator(manager)
                pane = LibraryPane(
                    lambda: PathScope.from_paths((tmp,)),
                    lambda: None,
                    job_manager=manager,
                    work_coordinator=coordinator,
                )
                try:
                    read_spec = pane._coordination_spec("Loading Library")
                    refresh_spec = pane._coordination_spec("Refreshing Library")
                    trash_spec = pane._coordination_spec("Trashing Hash duplicate Candidates")
                    self.assertEqual("Library", read_spec.origin)
                    self.assertTrue(read_spec.data_home_read)
                    self.assertTrue(refresh_spec.data_home_write)
                    self.assertTrue(refresh_spec.source_reads)
                    self.assertEqual("Tools", trash_spec.origin)
                    self.assertTrue(trash_spec.source_writes)

                    blocker = coordinator.submit(
                        JobSpec("Catalog writer", data_home_write=True),
                        lambda _job_id, _fallback: None,
                    )
                    completed: list[str] = []
                    worker = pane._start_job(
                        "Loading Library",
                        lambda _progress, _cancel: "loaded",
                        completed.append,
                    )
                    coordinated_id = pane._coordinated_job_ids[worker]
                    self.assertEqual("queued", manager.get(coordinated_id).status)
                    self.assertFalse(any(job is worker for job, _thread in pane._jobs))

                    coordinator.finish(blocker)
                    self._wait_for(lambda: completed == ["loaded"])
                    self._wait_for(lambda: manager.get(coordinated_id).status == "finished")
                    matching_rows = [state for state in manager.history(500) if state.job_id == coordinated_id]
                    self.assertEqual(1, len(matching_rows))
                finally:
                    pane.close()

                names = NamesPane(
                    lambda: None,
                    job_manager=manager,
                    work_coordinator=coordinator,
                    active_scope_provider=lambda: PathScope.from_paths((tmp,)),
                )
                try:
                    refresh_spec = names._coordination_spec("refresh")
                    photos_spec = names._coordination_spec("photos")
                    deep_spec = names._coordination_spec("deep_similar")
                    mutation_spec = names._coordination_spec(
                        "mutation",
                        source_writes=(str(Path(tmp) / "named.jpg"),),
                    )
                    self.assertEqual("People", refresh_spec.origin)
                    self.assertTrue(refresh_spec.data_home_write)
                    self.assertTrue(photos_spec.data_home_read)
                    self.assertFalse(photos_spec.source_reads)
                    self.assertTrue(deep_spec.data_home_read)
                    self.assertTrue(deep_spec.model_cache_read)
                    self.assertEqual((str(Path(tmp)),), tuple(scope.path for scope in deep_spec.normalized().source_reads))
                    self.assertTrue(mutation_spec.data_home_write)
                    self.assertEqual(
                        (str(Path(tmp) / "named.jpg"),),
                        tuple(scope.path for scope in mutation_spec.normalized().source_writes),
                    )

                    blocker = coordinator.submit(
                        JobSpec("Names database writer", data_home_write=True),
                        lambda _job_id, _fallback: None,
                    )
                    completed = []
                    names._start_job(
                        "photos",
                        lambda _progress, _cancel: "loaded",
                        completed.append,
                        self.fail,
                    )
                    worker = names._photos_job
                    self.assertIsNotNone(worker)
                    coordinated_id = names._coordinated_job_ids[worker]
                    self.assertEqual("queued", manager.get(coordinated_id).status)
                    self.assertIsNone(names._photos_thread)
                    self.assertFalse(any(job is worker for job, _thread in names._operation_threads))

                    coordinator.finish(blocker)
                    self._wait_for(lambda: completed == ["loaded"])
                    self._wait_for(lambda: manager.get(coordinated_id).status == "finished")
                    matching_rows = [state for state in manager.history(500) if state.job_id == coordinated_id]
                    self.assertEqual(1, len(matching_rows))
                finally:
                    names.shutdown_jobs(timeout_ms=500)
                    names.close()

                gallery = GalleryPane()
                gallery.configure_jobs(manager, coordinator, origin="Photos")
                photo_path = str(Path(tmp) / "photo.jpg")
                try:
                    blocker = coordinator.submit(
                        JobSpec("Photo writer", source_writes=(photo_path,)),
                        lambda _job_id, _fallback: None,
                    )
                    completed = []
                    gallery._start_action_job(
                        "Reading one photo",
                        lambda _progress, _cancel: "decoded",
                        completed.append,
                        source_reads=(photo_path,),
                    )
                    coordinated_id = gallery._active_action_job_id
                    self.assertIsNotNone(coordinated_id)
                    self.assertEqual("queued", manager.get(coordinated_id).status)
                    self.assertIsNone(gallery._active_action_thread)

                    coordinator.finish(blocker)
                    self._wait_for(lambda: completed == ["decoded"])
                    self._wait_for(lambda: manager.get(coordinated_id).status == "finished")
                    matching_rows = [state for state in manager.history(500) if state.job_id == coordinated_id]
                    self.assertEqual(1, len(matching_rows))

                    blocker = coordinator.submit(
                        JobSpec("Second photo writer", source_writes=(photo_path,)),
                        lambda _job_id, _fallback: None,
                    )
                    cancelled_work = []
                    gallery._start_action_job(
                        "Queued photo read",
                        lambda _progress, _cancel: cancelled_work.append("started"),
                        lambda _result: cancelled_work.append("completed"),
                        source_reads=(photo_path,),
                    )
                    cancelled_id = gallery._active_action_job_id
                    self.assertIsNotNone(cancelled_id)
                    self.assertEqual("queued", manager.get(cancelled_id).status)
                    self.assertTrue(gallery.shutdown_jobs(timeout_ms=500))
                    self.assertEqual("cancelled", manager.get(cancelled_id).status)
                    coordinator.finish(blocker)
                    APP.processEvents()
                    self.assertEqual([], cancelled_work)
                finally:
                    gallery.shutdown_jobs(timeout_ms=500)
                    gallery.close()

                blocker = coordinator.submit(
                    JobSpec("Inspector photo writer", source_writes=(photo_path,)),
                    lambda _job_id, _fallback: None,
                )
                inspector = PhotoInspectorDialog(
                    image_paths=[],
                    job_manager=manager,
                    work_coordinator=coordinator,
                )
                completed = []
                worker = AsyncJob(lambda _progress, _cancel: "inspected")
                try:
                    inspector._start_operation_worker(
                        worker,
                        "Reading Inspector photo",
                        foreground=False,
                        job_attribute="_active_job",
                        thread_attribute="_active_thread",
                        source_reads=(photo_path,),
                    )
                    coordinated_id = inspector._coordinated_job_ids[worker]
                    self.assertEqual("queued", manager.get(coordinated_id).status)
                    self.assertIsNone(inspector._active_thread)
                    worker.completed.connect(completed.append)

                    coordinator.finish(blocker)
                    self._wait_for(lambda: completed == ["inspected"])
                    self._wait_for(lambda: manager.get(coordinated_id).status == "finished")
                    state = manager.get(coordinated_id)
                    self.assertEqual("Photo Inspector", state.origin)
                    matching_rows = [row for row in manager.history(500) if row.job_id == coordinated_id]
                    self.assertEqual(1, len(matching_rows))
                finally:
                    inspector.shutdown_jobs(timeout_ms=500)
                    inspector.close()

    def test_production_settings_jobs_wait_for_coordinator_resources(self):
        from apps.pyqt_production.settings_dialog import ProductionSettingsDialog
        from ui.async_job import AsyncJob
        from ui.job_manager import JobManager
        from ui.work_coordinator import JobSpec, WorkCoordinator

        with TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
                settings_mod._RUNTIME_BASE_DIR = None
                layout = activate_runtime_root("ProductionCoordinatedSettingsTest")
                manager = JobManager()
                coordinator = WorkCoordinator(manager)
                blocker = coordinator.submit(
                    JobSpec("Data Home writer", data_home_write=True),
                    lambda _job_id, _fallback: None,
                )
                store = QSettings(str(Path(tmp) / "settings.ini"), QSettings.Format.IniFormat)
                dialog = ProductionSettingsDialog(
                    store,
                    RuntimeCapabilityService(),
                    runtime_layout=layout,
                    support_metadata_provider=lambda: {},
                    job_manager=manager,
                    work_coordinator=coordinator,
                    auto_refresh=False,
                )
                completed = []
                worker = AsyncJob(lambda _progress, _cancel: "scanned")
                dialog._cache_usage_job = worker
                worker.completed.connect(completed.append)
                try:
                    dialog._start_settings_job(
                        worker,
                        role="cache_usage",
                        thread_attribute="_cache_usage_thread",
                    )
                    coordinated_id = dialog._settings_job_ids[worker]
                    self.assertEqual("queued", manager.get(coordinated_id).status)
                    self.assertIsNone(dialog._cache_usage_thread)
                    self.assertTrue(dialog.cancel_settings_tasks_button.isEnabled())

                    coordinator.finish(blocker)
                    self._wait_for(lambda: completed == ["scanned"])
                    self._wait_for(lambda: manager.get(coordinated_id).status == "finished")
                    state = manager.get(coordinated_id)
                    self.assertEqual("Settings", state.origin)
                    matching_rows = [row for row in manager.history(500) if row.job_id == coordinated_id]
                    self.assertEqual(1, len(matching_rows))
                    self._wait_for(lambda: dialog._cache_usage_job is None)

                    blocker = coordinator.submit(
                        JobSpec("Second Data Home writer", data_home_write=True),
                        lambda _job_id, _fallback: None,
                    )
                    started = []
                    queued_worker = AsyncJob(lambda _progress, _cancel: started.append(True))
                    dialog._cache_usage_job = queued_worker
                    dialog._start_settings_job(
                        queued_worker,
                        role="cache_usage",
                        thread_attribute="_cache_usage_thread",
                    )
                    queued_id = dialog._settings_job_ids[queued_worker]
                    self.assertEqual("queued", manager.get(queued_id).status)
                    dialog._cancel_active_settings_tasks()
                    self.assertEqual("cancelled", manager.get(queued_id).status)
                    self.assertIsNone(dialog._cache_usage_thread)
                    self.assertIsNone(dialog._cache_usage_job)
                    self.assertFalse(dialog.cancel_settings_tasks_button.isEnabled())
                    coordinator.finish(blocker)
                    APP.processEvents()
                    self.assertEqual([], started)
                finally:
                    dialog.shutdown_jobs(timeout_ms=500)
                    dialog.close()

    def test_library_navigation_snapshot_initializes_and_reads_catalog_off_qt(self):
        from app.path_scope import PathScope
        from app.services.library_catalog import LibraryCatalogService
        from ui.library_pane import LibraryPane

        with TemporaryDirectory() as tmp:
            database = Path(tmp) / "catalog.sqlite3"
            catalog = LibraryCatalogService(db_path=database)
            self.assertFalse(database.exists())
            entered = Event()
            release = Event()
            read_threads: list[object] = []
            qt_acknowledged: list[bool] = []
            original_list_roots = catalog.list_roots

            def _blocked_list_roots(*args, **kwargs):
                read_threads.append(QThread.currentThread())
                entered.set()
                self.assertTrue(release.wait(timeout=3.0))
                return original_list_roots(*args, **kwargs)

            with patch.object(catalog, "list_roots", side_effect=_blocked_list_roots):
                pane = LibraryPane(
                    lambda: PathScope.from_paths((tmp,)),
                    lambda: None,
                    catalog=catalog,
                )
                try:
                    QTimer.singleShot(0, lambda: qt_acknowledged.append(True))
                    self._wait_for(entered.is_set, timeout_s=3.0)
                    self._wait_for(lambda: bool(qt_acknowledged), timeout_s=3.0)
                    self.assertTrue(read_threads)
                    self.assertIsNot(read_threads[0], APP.thread())
                    release.set()
                    self._wait_for(lambda: pane._catalog_snapshot_job is None, timeout_s=3.0)
                    self.assertTrue(database.exists())
                finally:
                    release.set()
                    pane.shutdown_jobs(timeout_ms=3000)
                    pane.close()

    def test_empty_active_scope_uses_one_shared_edit_roots_surface(self):
        from apps.pyqt_production.app import ProductionClusterApp

        with TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
                settings_mod._RUNTIME_BASE_DIR = None
                layout = activate_runtime_root("ProductionEmptyScopeTest")
                window = ProductionClusterApp(layout)
                try:
                    window.source_pane.set_active_roots(())
                    APP.processEvents()

                    self.assertIs(window.workspace_stack.currentWidget(), window.empty_scope_panel)
                    self.assertFalse(window.empty_scope_edit_roots_button.isHidden())
                    self.assertTrue(window.edit_roots_button.isHidden())
                    self.assertTrue(window.workspace_subnav.isHidden())
                    self.assertEqual("primary", window.empty_scope_edit_roots_button.property("kind"))
                    self.assertIn("same active roots", window.empty_scope_panel.findChild(type(window.scope_summary_label), "noActiveRootsDetail").text())
                    active_job_ids = {job.job_id for job in window.job_manager.active_jobs()}

                    for workspace_id in ("clustering", "faces", "tags", "library"):
                        window.set_active_workspace(workspace_id)
                        APP.processEvents()
                        self.assertIs(window.workspace_stack.currentWidget(), window.empty_scope_panel)
                        # A pre-existing asynchronous Library navigation snapshot may
                        # finish while routes change; switching an empty scope must not
                        # create any additional work.
                        self.assertTrue(
                            {job.job_id for job in window.job_manager.active_jobs()}.issubset(active_job_ids)
                        )
                    self.assertIsNone(window.faces_pane)
                finally:
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
                window.source_pane.set_active_roots((tmp,))

                window.set_active_workspace("faces")
                self._wait_for(lambda: window.faces_pane is not None, timeout_s=5.0)

                self.assertEqual("faces", window._active_workspace)
                self.assertIs(window.workspace_stack.currentWidget(), window.faces_pane)
                self.assertEqual(["human"], list(window.face_services_global))
                self.assertEqual(1, window.faces_pane.face_mode_combo.count())
                self.assertTrue(window.faces_workspace_button.isChecked())
                self.assertFalse(window.clustering_workspace_button.isChecked())
                window.close()

    def test_direct_photo_face_tools_are_lazy_retryable_and_read_only_safe(self):
        from apps.pyqt_production.app import ProductionClusterApp

        with TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
                settings_mod._RUNTIME_BASE_DIR = None
                layout = activate_runtime_root("ProductionDirectPhotoFaceToolsTest")
                window = ProductionClusterApp(layout)
                self._wait_for_storage_idle(window)
                # This isolated path is about Photos wiring, not model probing.
                window._startup_readiness_report = SimpleNamespace(face_ready=True)
                requested_loads: list[bool] = []
                ready: list[bool] = []
                failures: list[str] = []
                window._start_faces_workspace_load = lambda: requested_loads.append(True)  # type: ignore[method-assign]

                window._request_photo_face_tools("/photo.jpg", lambda: ready.append(True), failures.append)

                self.assertEqual([True], requested_loads)
                self.assertEqual([], ready)
                self.assertEqual([], failures)
                window._finish_pending_photo_face_tool_requests(error="Face model is unavailable")
                self.assertEqual(["Face model is unavailable"], failures)

                window._request_photo_face_tools("/photo.jpg", lambda: ready.append(True), failures.append)
                window._finish_pending_photo_face_tool_requests()
                self.assertEqual([True], ready)

                with patch.object(window, "_read_only_mode", return_value=True):
                    window._request_photo_face_tools("/photo.jpg", lambda: ready.append(True), failures.append)
                self.assertEqual("Face editing is disabled by read-only safety mode.", failures[-1])
                window.close()

    def test_workspace_switch_cancels_stale_faces_initialization(self):
        from apps.pyqt_production.app import ProductionClusterApp

        started = Event()
        release = Event()
        constructor_on_ui_thread: list[bool] = []

        class _SlowFaceIndexService:
            def __init__(self, **_kwargs):
                constructor_on_ui_thread.append(QThread.currentThread() is APP.thread())
                if not started.is_set():
                    started.set()
                    release.wait(timeout=5.0)

        fake_face_search = SimpleNamespace(
            DEFAULT_HUMAN_FACE_DETECTOR_ID="builtin-detector",
            DEFAULT_HUMAN_FACE_EMBEDDER_ID="builtin-embedder",
            BUILTIN_HUMAN_DETECTOR_ID="builtin-detector",
            BUILTIN_HUMAN_EMBEDDER_ID="builtin-embedder",
            resolve_ready_face_pipeline_ids=lambda *_args: ("builtin-detector", "builtin-embedder"),
            FaceIndexService=_SlowFaceIndexService,
        )
        policy = SimpleNamespace(cuda_required_unavailable=False)

        with TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
                settings_mod._RUNTIME_BASE_DIR = None
                layout = activate_runtime_root("ProductionFacesWorkspaceSwitchTest")
                window = ProductionClusterApp(layout)
                self._wait_for_storage_idle(window)
                window.source_pane.set_active_roots((tmp,))
                with (
                    patch("apps.pyqt_production.app._face_search_api", return_value=fake_face_search),
                    patch.object(window.runtime_service, "select_policy", return_value=policy),
                    patch.object(window.runtime_service, "detect", return_value=SimpleNamespace()),
                ):
                    window.set_active_workspace("faces")
                    self._wait_for(started.is_set, timeout_s=5.0)
                    init_job = window._faces_init_job

                    window.set_active_workspace("clustering")

                    self.assertIsNotNone(init_job)
                    self.assertTrue(constructor_on_ui_thread)
                    self.assertFalse(any(constructor_on_ui_thread))
                    self.assertTrue(init_job._cancel_requested)
                    self.assertEqual("clustering", window._active_workspace)
                    self.assertIs(window.workspace_stack.currentWidget(), window.clustering_workspace)
                    release.set()
                    self._wait_for(lambda: window._faces_init_thread is None, timeout_s=5.0)

                self.assertIsNone(window.faces_pane)
                self.assertEqual("clustering", window._active_workspace)
                self.assertIs(window.workspace_stack.currentWidget(), window.clustering_workspace)
                window.close()

    def test_settings_face_service_reload_constructs_services_off_qt(self):
        from apps.pyqt_production.app import ProductionClusterApp

        entered = Event()
        release = Event()
        constructor_on_ui_thread: list[bool] = []

        with TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
                settings_mod._RUNTIME_BASE_DIR = None
                window = ProductionClusterApp(activate_runtime_root("ProductionFaceReloadThreadTest"))
                try:
                    self._wait_for_storage_idle(window)
                    window.source_pane.set_active_roots((tmp,))
                    window.set_active_workspace("faces")
                    self._wait_for(lambda: window.faces_pane is not None, timeout_s=5.0)
                    original_global = window.face_service_global
                    original_session = window.face_service_session

                    def _service_factory(**kwargs):
                        constructor_on_ui_thread.append(QThread.currentThread() is APP.thread())
                        if not entered.is_set():
                            entered.set()
                            self.assertTrue(release.wait(timeout=5.0))
                        return original_session if "session" in str(kwargs.get("db_path", "")) else original_global

                    fake_face_search = SimpleNamespace(
                        DEFAULT_HUMAN_FACE_DETECTOR_ID="builtin-detector",
                        DEFAULT_HUMAN_FACE_EMBEDDER_ID="builtin-embedder",
                        BUILTIN_HUMAN_DETECTOR_ID="builtin-detector",
                        BUILTIN_HUMAN_EMBEDDER_ID="builtin-embedder",
                        resolve_ready_face_pipeline_ids=lambda *_args: ("builtin-detector", "builtin-embedder"),
                        FaceIndexService=_service_factory,
                    )
                    qt_acknowledged: list[bool] = []
                    with patch("apps.pyqt_production.app._face_search_api", return_value=fake_face_search):
                        window._reload_face_services_from_settings()
                        self._wait_for(entered.is_set, timeout_s=5.0)
                        QTimer.singleShot(0, lambda: qt_acknowledged.append(True))
                        self._wait_for(lambda: bool(qt_acknowledged), timeout_s=3.0)
                        self.assertFalse(any(constructor_on_ui_thread))
                        self.assertFalse(window.faces_pane.isEnabled())
                        release.set()
                        self._wait_for(lambda: window._faces_init_job is None, timeout_s=5.0)
                        self.assertTrue(window.faces_pane.isEnabled())
                finally:
                    release.set()
                    window.close()

    def test_gallery_result_partitioning_keeps_qt_responsive(self):
        from apps.pyqt_production.app import ProductionClusterApp
        from app.path_scope import PathScope
        from app.services.discovery import ImageDiscoveryService

        entered = Event()
        release = Event()
        partition_threads: list[object] = []

        with TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
                settings_mod._RUNTIME_BASE_DIR = None
                window = ProductionClusterApp(activate_runtime_root("ProductionGalleryPartitionThreadTest"))
                try:
                    self._wait_for_storage_idle(window)
                    root = str(Path(tmp) / "photos")
                    paths = tuple(f"{root}/{index:05d}.jpg" for index in range(5_000))
                    result = SimpleNamespace(paths=paths, snapshot_key="fixed", filtered_count=0)
                    original_contains = PathScope.contains

                    def _blocked_contains(scope, candidate):
                        partition_threads.append(QThread.currentThread())
                        if not entered.is_set():
                            entered.set()
                            self.assertTrue(release.wait(timeout=5.0))
                        return original_contains(scope, candidate)

                    qt_acknowledged: list[bool] = []
                    with (
                        patch.object(ImageDiscoveryService, "discover_roots_result", return_value=result),
                        patch.object(PathScope, "contains", _blocked_contains),
                    ):
                        window._load_gallery_scope(PathScope((root,)))
                        self._wait_for(entered.is_set, timeout_s=5.0)
                        QTimer.singleShot(0, lambda: qt_acknowledged.append(True))
                        self._wait_for(lambda: bool(qt_acknowledged), timeout_s=3.0)
                        self.assertTrue(partition_threads)
                        self.assertTrue(all(thread is not APP.thread() for thread in partition_threads))
                        release.set()
                        self._wait_for(lambda: window._gallery_discovery_job is None, timeout_s=5.0)
                    self.assertEqual(5_000, len(window._gallery_paths))
                finally:
                    release.set()
                    window.close()

    def test_startup_readiness_gate_controls_model_bound_actions(self):
        from apps.pyqt_production.app import ProductionClusterApp

        with TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
                settings_mod._RUNTIME_BASE_DIR = None
                layout = activate_runtime_root("ProductionStartupReadinessGateTest")
                window = ProductionClusterApp(layout)
                unavailable = SimpleNamespace(
                    clustering_ready=False,
                    clustering_message="Install the selected clustering model.",
                    face_ready=False,
                    face_message="Install FaceNet.",
                )
                window._startup_readiness_report = unavailable
                window._startup_readiness_error = ""
                window._apply_startup_workflow_gate()

                self.assertFalse(window.source_pane.basic_run_button.isEnabled())
                self.assertFalse(window.clustering_pane.cluster_button.isEnabled())
                self.assertIn("Install the selected clustering model", window.clustering_pane.cluster_button.toolTip())

                ready = SimpleNamespace(
                    clustering_ready=True,
                    clustering_message="CPU and clustering models are ready.",
                    face_ready=True,
                    face_message="Face pipeline ready.",
                )
                window._startup_readiness_report = ready
                window._apply_startup_workflow_gate()

                self.assertTrue(window.source_pane.basic_run_button.isEnabled())
                self.assertTrue(window.clustering_pane.cluster_button.isEnabled())
                window.close()

    def test_startup_readiness_replacement_discards_late_result_and_cancel_is_explicit(self):
        from app.services.startup_readiness import StartupReadinessReport
        from apps.pyqt_production.app import ProductionClusterApp

        first_started = Event()
        third_started = Event()
        release_first = Event()
        release_third = Event()
        calls = 0
        readiness_on_ui_thread: list[bool] = []
        report = StartupReadinessReport(
            execution_policy=ExecutionPolicy(
                preferred_mode="cpu",
                effective_mode="cpu",
                reason="Fixture CPU runtime is ready.",
            ),
            capabilities=RuntimeCapabilities(torch_version="fixture", onnx_version="fixture"),
            clustering_models=("fixture",),
            clustering_ready=True,
            clustering_message="Fixture clustering is ready.",
            face_detector_id="fixture-detector",
            face_embedder_id="fixture-embedder",
            face_ready=True,
            face_message="Fixture faces are ready.",
        )

        def _readiness_fixture(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            readiness_on_ui_thread.append(QThread.currentThread() is APP.thread())
            if calls == 1:
                first_started.set()
                release_first.wait(timeout=5.0)
            elif calls == 3:
                third_started.set()
                release_third.wait(timeout=5.0)
            return report

        with TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
                settings_mod._RUNTIME_BASE_DIR = None
                layout = activate_runtime_root("ProductionStartupReadinessReplacementTest")
                window = ProductionClusterApp(layout)
                try:
                    self._wait_for_storage_idle(window)
                    self._wait_for(
                        lambda: not window._thread_is_running(window._startup_readiness_thread),
                        timeout_s=5.0,
                    )
                    with patch("apps.pyqt_production.app.inspect_startup_readiness", side_effect=_readiness_fixture):
                        window._begin_startup_readiness_check()
                        self._wait_for(first_started.is_set, timeout_s=5.0)
                        first_thread = window._startup_readiness_thread
                        self.assertFalse(any(readiness_on_ui_thread))
                        self.assertEqual("Runtime checking… · Checking", window.runtime_badge.text())

                        # The second generation is current. Releasing the old
                        # worker after it completes must not overwrite Ready.
                        window._begin_startup_readiness_check()
                        self._wait_for(
                            lambda: window._startup_readiness_report is report
                            and window.runtime_badge.text().endswith("· Ready"),
                            timeout_s=5.0,
                        )
                        release_first.set()
                        self._wait_for(
                            lambda: first_thread is None or not first_thread.isRunning(),
                            timeout_s=5.0,
                        )
                        self.assertTrue(window.runtime_badge.text().endswith("· Ready"))

                        window._begin_startup_readiness_check()
                        self._wait_for(third_started.is_set, timeout_s=5.0)
                        current_job = window._startup_readiness_job
                        self.assertIsNotNone(current_job)
                        current_job.cancel()
                        release_third.set()
                        self._wait_for(
                            lambda: window.runtime_badge.text() == "Runtime failed · Failed",
                            timeout_s=5.0,
                        )
                        self.assertIn("Rescan GPU Resources", window.runtime_badge.toolTip())
                finally:
                    release_first.set()
                    release_third.set()
                    window.close()

    def test_production_window_switches_to_names_workspace_without_opening_faces(self):
        from apps.pyqt_production.app import ProductionClusterApp

        with TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
                settings_mod._RUNTIME_BASE_DIR = None
                layout = activate_runtime_root("ProductionNamesWorkspaceTest")
                window = ProductionClusterApp(layout)
                self._wait_for_storage_idle(window)
                window.source_pane.set_active_roots((str(Path(__file__).parent),))
                window.show()
                APP.processEvents()

                self.assertTrue(window.people_workspace_button.isVisible())
                self.assertIsNone(window.faces_pane)
                window.set_active_workspace("names")
                self._wait_for(lambda: window.names_pane is not None, timeout_s=5.0)
                self._wait_for(
                    lambda: "saved name(s)" in window.names_pane.status_label.text(),
                    timeout_s=5.0,
                )

                self.assertEqual("names", window._active_workspace)
                self.assertIs(window.workspace_stack.currentWidget(), window.names_pane)
                self.assertTrue(window.people_workspace_button.isChecked())
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
                window.source_pane.set_active_roots((tmp,))
                window.set_active_workspace("faces")
                self._wait_for(lambda: window.faces_pane is not None, timeout_s=5.0)

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
            with patch.dict(
                os.environ,
                {"CLUSTERLENS_RUNTIME_ROOT": tmp, "IMAGE_CLUSTERING_APP_DIR": tmp},
                clear=False,
            ):
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

    def test_production_ui_refreshes_recovery_history_after_actual_killed_move(self):
        with TemporaryDirectory() as tmp:
            fixture_root = Path(tmp) / "fixture"
            fixture_root.mkdir()
            source = fixture_root / "source.jpg"
            source.write_bytes(b"source-photo")
            destination = fixture_root / "destination"
            runtime_root = Path(tmp) / "runtime"
            with patch.dict(
                os.environ,
                {"CLUSTERLENS_RUNTIME_ROOT": str(runtime_root), "IMAGE_CLUSTERING_APP_DIR": str(runtime_root)},
                clear=False,
            ):
                settings_mod._RUNTIME_BASE_DIR = None
                layout = activate_runtime_root("ProductionKilledMoveRecoveryTest")
                from apps.pyqt_production.app import ProductionClusterApp
                from apps.pyqt_production.settings_dialog import ProductionSettingsDialog

                app_settings = settings_mod.get_settings()
                config = {
                    "audit_log_path": str(app_settings.log_dir / "file_operations.jsonl"),
                    "journal_path": str(app_settings.log_dir / "file_operations.sqlite3"),
                    "temp_dir": str(app_settings.cache_dir / "tmp" / "file_ops"),
                    "source_path": str(source),
                    "destination": str(destination),
                }
                (fixture_root / "gallery_paths.json").write_text(json.dumps(config), encoding="utf-8")
                environment = dict(os.environ)
                source_root = str(Path(__file__).resolve().parents[1] / "src")
                environment["PYTHONPATH"] = source_root + (
                    os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
                )
                child = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve().parent / "durable_kill_worker.py"),
                        "gallery_move",
                        "after_filesystem_mutation",
                        str(fixture_root),
                    ],
                    cwd=Path(__file__).resolve().parents[1],
                    env=environment,
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=30,
                )
                self.assertEqual(-signal.SIGKILL, child.returncode, child.stderr)
                self.assertFalse(source.exists())
                moved = destination / source.name
                self.assertEqual(b"source-photo", moved.read_bytes())

                window = ProductionClusterApp(layout)
                dialog = None
                try:
                    self._wait_for_storage_idle(window)
                    entries = window.gallery_pane.action_service.read_audit_entries(limit=10)
                    self.assertEqual(1, len(entries))
                    self.assertTrue(entries[0]["completed"])
                    self.assertEqual("restorable", entries[0]["recovery_status"])
                    self.assertEqual([[str(source), str(moved)]], entries[0]["changed_paths"])

                    dialog = ProductionSettingsDialog(
                        window.settings_store,
                        RuntimeCapabilityService(),
                        runtime_layout=layout,
                        support_metadata_provider=lambda: {},
                        auto_refresh=False,
                        parent=window,
                    )
                    dialog.refresh_operation_journal()
                    self._wait_for(
                        lambda: dialog._journal_refresh_job is None and dialog._journal_refresh_thread is None,
                        timeout_s=5.0,
                    )
                    self.assertEqual(1, dialog.operation_journal_model.rowCount())
                    shown = dialog.operation_journal_model.entry_at(0)
                    self.assertTrue(shown["completed"])
                    self.assertEqual("restorable", shown["recovery_status"])
                    self.assertIn("Showing 1 of 1 recovery operations", dialog.journal_status_label.text())
                finally:
                    if dialog is not None:
                        dialog.close()
                    window.close()

    def test_production_face_results_open_in_top_level_gallery_route(self):
        from apps.pyqt_production.app import ProductionClusterApp

        with TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
                settings_mod._RUNTIME_BASE_DIR = None
                layout = activate_runtime_root("ProductionFaceResultsGalleryTest")
                window = ProductionClusterApp(layout)
                self._wait_for_storage_idle(window)
                window.source_pane.set_active_roots((str(Path(__file__).parent),))
                window.set_active_workspace("faces")
                self._wait_for(lambda: window.faces_pane is not None, timeout_s=5.0)
                image_a = str(Path(tmp) / "a.jpg")
                image_b = str(Path(tmp) / "b.jpg")
                window.faces_pane.context_for_path = lambda path: {"origin": "faces", "path": path}

                window._open_face_results_in_main_gallery([image_a, image_b])
                self._wait_for(lambda: window._photo_set_route is not None, timeout_s=2.0)

                self.assertEqual("clustering", window._active_workspace)
                self.assertIs(window.clustering_gallery_stack.currentWidget(), window.photo_gallery)
                self.assertEqual([image_a, image_b], list(window.photo_gallery._model.all_paths()))
                self.assertEqual("faces", window._photo_gallery_context_for_path(image_a)["origin"])
                self.assertEqual("faces", window._photo_set_route.return_workspace)
                self.assertFalse(window.photo_gallery.review_faces_button.isHidden())
                self.assertFalse(window.photo_gallery.back_to_folder_button.isHidden())
                window.close()

    def test_production_migrates_saved_gallery_to_library_but_keeps_explicit_gallery_routes(self):
        from apps.pyqt_production.app import ProductionClusterApp

        with TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
                settings_mod._RUNTIME_BASE_DIR = None
                layout = activate_runtime_root("ProductionLibraryFirstMigrationTest")
                window = ProductionClusterApp(layout)
                try:
                    self._wait_for_storage_idle(window)
                    window.settings_store.setValue("workspace/default_view", "gallery")
                    self.assertEqual("library", window._preferred_workspace())
                    window.set_active_workspace("gallery")
                    self.assertEqual("clustering", window._active_workspace)
                    self.assertIs(window.clustering_gallery_stack.currentWidget(), window.photo_gallery)
                finally:
                    window.close()

    def test_production_clustering_job_records_exact_cpu_fallback_reason(self):
        from apps.pyqt_production.app import ProductionClusterApp

        with TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
                settings_mod._RUNTIME_BASE_DIR = None
                layout = activate_runtime_root("ProductionClusteringFallbackJobTest")
                window = ProductionClusterApp(layout)
                self._wait_for_storage_idle(window)
                window._start_cluster_tag_context_refresh = lambda: None

                window._on_clustering_started()
                job_id = window._active_job_id
                window._on_clustering_completed(
                    {
                        "clusters_by_key": {},
                        "membership_by_image": {},
                        "metrics_by_key": {
                            "dino::hdbscan": {
                                "compute_fallback_reason": "cuML HDBSCAN provider unavailable",
                            }
                        },
                    }
                )

                job = window.job_manager.get(job_id)
                self.assertIsNotNone(job)
                self.assertEqual("finished", job.status)
                self.assertEqual("CPU fallback: cuML HDBSCAN provider unavailable", job.cache_status)
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

    def test_worker_progress_protocol_preserves_zero_and_normalizes_invalid_values(self):
        cases = (
            ({"value": 0, "status": "Starting"}, (0, "Starting")),
            ({"value": 1, "status": "Preparing"}, (1, "Preparing")),
            ({"value": 100, "status": "Done"}, (100, "Done")),
            ({"value": -1, "status": "Discovering"}, (-1, "Discovering")),
            ({"value": None, "status": "Waiting"}, (-1, "Waiting")),
            ({"status": "Waiting"}, (-1, "Waiting")),
            ({"value": "bad", "status": "Waiting"}, (-1, "Waiting")),
            ({"value": float("nan"), "status": "Waiting"}, (-1, "Waiting")),
            ({"value": float("inf"), "status": "Waiting"}, (-1, "Waiting")),
            ({"value": True, "status": "Waiting"}, (-1, "Waiting")),
            ({"value": -2, "status": "Below range"}, (0, "Below range")),
            ({"value": 125, "status": "Finishing"}, (100, "Finishing")),
            ({"completed": 1, "total": 4, "phase": "Embedding", "unit": "photos"}, (25, "Embedding — 1/4 photos")),
            ({"completed": 0, "total": 0, "phase": "Empty"}, (-1, "Empty — 0/0")),
            ({"completed": 9, "total": 4, "phase": "Verify"}, (100, "Verify — 9/4")),
            ({"completed": 1.5, "total": 4, "phase": "Invalid count"}, (-1, "Invalid count")),
            ({"completed": 1, "total": 4, "unit": ["photos"]}, (25, "1/4")),
            (
                {"value": 80, "phase": "Import", "processed": 8, "skipped": 1, "failed": 1},
                (80, "Import — 8 processed, 1 skipped, 1 failed"),
            ),
            ({"value": 4, "status": {"bad": "status"}, "phase": ["bad"]}, (4, "")),
        )
        with TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
                layout = activate_runtime_root("ProductionProgressProtocolTest")
                clustering = ClusteringSessionController(layout)
                downloads = ModelDownloadController(layout)
                clustering_seen: list[tuple[int, str]] = []
                download_seen: list[tuple[int, str]] = []
                clustering.progress.connect(lambda value, status: clustering_seen.append((value, status)))
                downloads.progress.connect(lambda value, status: download_seen.append((value, status)))

                for body, expected in cases:
                    with self.subTest(body=body):
                        clustering_seen.clear()
                        download_seen.clear()
                        line = json.dumps({"type": "progress", "payload": body}) + "\n"
                        clustering._stdout_buffer = line
                        clustering._drain_stdout_lines()
                        downloads._handle_stdout_line(line)
                        self.assertEqual([expected], clustering_seen)
                        self.assertEqual([expected], download_seen)

    def test_worker_progress_protocol_allows_a_new_phase_to_restart_at_zero(self):
        with TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
                layout = activate_runtime_root("ProductionProgressPhaseTest")
                for controller in (ClusteringSessionController(layout), ModelDownloadController(layout)):
                    with self.subTest(controller=type(controller).__name__):
                        seen: list[tuple[int, str]] = []
                        controller.progress.connect(lambda value, status: seen.append((value, status)))
                        lines = (
                            json.dumps({"type": "progress", "payload": {"value": 100, "phase": "Download"}})
                            + "\n"
                            + json.dumps({"type": "progress", "payload": {"value": 0, "phase": "Verify"}})
                            + "\n"
                        )
                        if isinstance(controller, ClusteringSessionController):
                            controller._stdout_buffer = lines
                            controller._drain_stdout_lines()
                        else:
                            for line in lines.splitlines():
                                controller._handle_stdout_line(line)

                        self.assertEqual([(100, "Download"), (0, "Verify")], seen)
                        self.assertIsNone(controller._result_payload)

    def test_process_controllers_emit_one_terminal_event_and_no_late_progress(self):
        with TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
                layout = activate_runtime_root("ProductionTerminalProtocolTest")
                for controller_kind in ("clustering", "download"):
                    for first_terminal in ("completed", "failed", "cancelled"):
                        with self.subTest(controller=controller_kind, first=first_terminal):
                            controller = (
                                ClusteringSessionController(layout)
                                if controller_kind == "clustering"
                                else ModelDownloadController(layout)
                            )
                            events: list[tuple[str, object]] = []
                            progress: list[tuple[int, str]] = []
                            controller.completed.connect(lambda payload: events.append(("completed", dict(payload))))
                            controller.failed.connect(lambda message: events.append(("failed", str(message))))
                            controller.cancelled.connect(lambda: events.append(("cancelled", None)))
                            controller.progress.connect(lambda value, text: progress.append((value, text)))
                            controller._pending_progress = (37, "Preparing 3/8")

                            def _complete() -> None:
                                if controller_kind == "clustering":
                                    controller._result_payload = {"ok": True}
                                    controller._emit_completed_payload()
                                else:
                                    controller._emit_completed_once({"ok": True})

                            terminal_calls = {
                                "completed": _complete,
                                "failed": lambda: controller._emit_failed_once("fixture failure"),
                                "cancelled": controller._emit_cancelled_once,
                            }
                            terminal_calls[first_terminal]()
                            for name in ("completed", "failed", "cancelled"):
                                terminal_calls[name]()
                            controller._publish_progress(99, "late progress")

                            self.assertEqual([(37, "Preparing 3/8")], progress)
                            self.assertEqual(1, len(events))
                            self.assertEqual(first_terminal, events[0][0])

    def test_late_qt_diagnostic_does_not_recreate_a_removed_runtime(self):
        with TemporaryDirectory() as tmp:
            removed_runtime = Path(tmp) / "removed-runtime"
            log_file = removed_runtime / "logs" / "app.log"
            with patch(
                "infra.qt_diagnostics.get_settings",
                return_value=SimpleNamespace(log_file=str(log_file)),
            ):
                append_qt_diagnostic("[LateCallback] fixture")

            self.assertFalse(removed_runtime.exists())

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
                if controller._diagnostic_future is not None:
                    controller._diagnostic_future.result(timeout=3.0)

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
                if controller._diagnostic_future is not None:
                    controller._diagnostic_future.result(timeout=3.0)

    def test_disabling_warm_worker_stops_idle_process_and_blocks_restart_until_exit(self):
        class _FakeProcess:
            def __init__(self):
                self.writes: list[bytes] = []
                self.closed = False

            def state(self):
                return QProcess.ProcessState.Running

            def write(self, payload):
                self.writes.append(bytes(payload))

            def closeWriteChannel(self):
                self.closed = True

        with TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
                controller = ClusteringSessionController(
                    activate_runtime_root("ProductionWarmWorkerDisableTest")
                )
                process = _FakeProcess()
                controller._process = process
                controller._daemon_process = True
                controller._keep_worker_warm = True
                self.assertTrue(controller.is_worker_warm())

                controller.set_keep_worker_warm(False)

                self.assertFalse(controller.is_worker_warm())
                self.assertTrue(controller.is_running())
                self.assertEqual(
                    [{"type": "shutdown", "payload": {}}],
                    [json.loads(payload.decode("utf-8")) for payload in process.writes],
                )
                self.assertTrue(process.closed)

    def test_disabling_warm_worker_during_work_stops_it_after_completion(self):
        class _FakeProcess:
            def __init__(self):
                self.writes: list[bytes] = []
                self.closed = False

            def state(self):
                return QProcess.ProcessState.Running

            def write(self, payload):
                self.writes.append(bytes(payload))

            def closeWriteChannel(self):
                self.closed = True

        with TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
                controller = ClusteringSessionController(
                    activate_runtime_root("ProductionWarmWorkerBusyDisableTest")
                )
                process = _FakeProcess()
                controller._process = process
                controller._daemon_process = True
                controller._keep_worker_warm = True
                controller._running_request = True
                controller._result_payload = {"clusters_by_key": {}}

                controller.set_keep_worker_warm(False)
                self.assertFalse(process.writes)

                controller._complete_daemon_result()

                self.assertEqual(1, len(process.writes))
                self.assertTrue(process.closed)
                self.assertTrue(controller.is_running())

    def test_real_warm_worker_exits_restarts_and_leaves_no_child(self):
        from apps.pyqt_production.worker_protocol import ProductionClusterRequest

        with TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
                layout = activate_runtime_root("ProductionRealWarmWorkerLifecycleTest")
                empty_source = Path(tmp) / "empty-source"
                empty_source.mkdir()
                request = ProductionClusterRequest(
                    directory=str(empty_source),
                    embedding_models=["fast_preview"],
                    num_clusters=2,
                    clustering_backends=["cosine-kmeans"],
                    recursive=False,
                    similarity_mode="semantic",
                    outlier_policy="assign",
                    use_onnx=False,
                    reuse_result_cache=False,
                    use_embedding_cache_lookup=False,
                    preferred_execution_mode="cpu",
                    allow_model_downloads=False,
                )
                controller = ClusteringSessionController(layout)
                failures: list[str] = []
                controller.failed.connect(failures.append)

                def run_until_idle_failure() -> tuple[int, QProcess]:
                    failure_count = len(failures)
                    controller.set_keep_worker_warm(True)
                    self.assertTrue(controller.start(request))
                    self._wait_for(
                        lambda: len(failures) > failure_count and controller.is_worker_warm(),
                        timeout_s=10.0,
                    )
                    process = controller._process
                    self.assertIsNotNone(process)
                    pid = int(process.processId())
                    self.assertGreater(pid, 0)
                    self.assertTrue(Path(f"/proc/{pid}").exists())
                    if controller._diagnostic_future is not None:
                        controller._diagnostic_future.result(timeout=3.0)
                    return pid, process

                first_pid, first_process = run_until_idle_failure()
                controller.set_keep_worker_warm(False)
                self._wait_for(lambda: controller._process is None, timeout_s=5.0)
                self.assertFalse(Path(f"/proc/{first_pid}").exists())

                second_pid, second_process = run_until_idle_failure()
                self.assertNotEqual(first_pid, second_pid)
                controller.set_keep_worker_warm(False)
                self._wait_for(lambda: controller._process is None, timeout_s=5.0)
                self.assertFalse(Path(f"/proc/{second_pid}").exists())
                for process in (first_process, second_process):
                    try:
                        self.assertEqual(QProcess.ProcessState.NotRunning, process.state())
                    except RuntimeError:
                        pass

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
                if controller._diagnostic_future is not None:
                    controller._diagnostic_future.result(timeout=3.0)

    def test_production_entrypoint_routes_worker_mode_without_starting_gui(self):
        from apps.pyqt_production import __main__ as production_main

        with patch("apps.pyqt_production.worker.main", return_value=12) as worker_main:
            self.assertEqual(12, production_main.main(["--worker", "--request-json", "request.json"]))

        worker_main.assert_called_once_with(["--request-json", "request.json"])

    def test_production_entrypoint_allows_explicit_new_instance(self):
        from apps.pyqt_production import __main__ as production_main

        with patch("apps.pyqt_production.app.run", return_value=7) as run:
            self.assertEqual(7, production_main.main(["--new-instance"]))

        run.assert_called_once_with(new_instance=True)

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
            backend_options_by_backend={
                "hdbscan": {"min_cluster_size": 8, "min_samples": 3}
            },
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
        self.assertEqual(
            {"hdbscan": {"min_cluster_size": 8, "min_samples": 3}},
            request.as_dict()["backend_options_by_backend"],
        )

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

                missing_dino_plan = ModelDownloadPlan(
                    requested_models=("dino",),
                    runnable_without_download=(),
                    models_requiring_download=("dino",),
                    meaning_model="auto",
                    meaning_requires_download=True,
                    bundled_fallback_model="fast_preview",
                    bundled_models=("fast_preview",),
                )
                with (
                    patch("apps.pyqt_production.app.confirmBox", return_value=False),
                    patch.object(
                        window.model_asset_service,
                        "build_download_plan",
                        return_value=missing_dino_plan,
                    ),
                ):
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

                missing_dino_plan = ModelDownloadPlan(
                    requested_models=("dino",),
                    runnable_without_download=(),
                    models_requiring_download=("dino",),
                    meaning_model=None,
                    meaning_requires_download=False,
                    bundled_fallback_model="fast_preview",
                    bundled_models=("fast_preview",),
                )
                with (
                    patch("apps.pyqt_production.app.confirmBox", return_value=True),
                    patch.object(
                        window.model_asset_service,
                        "build_download_plan",
                        return_value=missing_dino_plan,
                    ),
                ):
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
            with patch.dict(
                os.environ,
                {
                    "CLUSTERLENS_RUNTIME_ROOT": str(runtime_root),
                    "IMAGE_CLUSTERING_APP_DIR": str(runtime_root),
                },
                clear=False,
            ):
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
        ), patch(
            "infra.runtime._cpu_runtime_details",
            return_value={
                "numpy_simd": ("AVX2",),
                "blas": "openblas",
                "blas_threads": 8,
                "libjpeg_turbo": True,
            },
        ):
            details = service.diagnostics("cuda")

        optional_details = details.get("optional_details") or {}
        remediation = "\n".join(details.get("remediation") or [])
        self.assertFalse(optional_details.get("hf_xet_installed"))
        self.assertFalse(optional_details.get("flash_sdp_enabled"))
        self.assertEqual(("AVX2",), details["cpu_details"]["numpy_simd"])
        self.assertEqual("openblas", details["cpu_details"]["blas"])
        self.assertTrue(details["cpu_details"]["libjpeg_turbo"])
        self.assertIn("hf_xet", remediation)
        self.assertIn("packaged exe", remediation)
        self.assertIn("Flash SDP", remediation)
        self.assertIn("cuML", remediation)

    def test_production_gallery_uses_one_overflow_without_legacy_buttons(self):
        from apps.pyqt_production.app import ProductionClusterApp, RUNTIME_LAYOUT

        window = ProductionClusterApp(RUNTIME_LAYOUT)
        window.source_pane.set_active_roots((str(Path(__file__).parent),))
        window.show()
        APP.processEvents()

        self.assertEqual("basic", window._clustering_mode)
        self.assertEqual("basic", window.gallery_pane.inspector_display_mode)
        self.assertFalse(hasattr(window, "gallery_workspace_button"))
        self.assertIs(window.clustering_gallery_stack.currentWidget(), window.photo_gallery)
        self.assertTrue(window.photo_gallery.isVisible())
        for legacy_name in (
            "copy_paths_button", "export_paths_button", "copy_button", "move_button",
            "delete_button", "exif_button", "tags_button", "retry_failed_button",
            "metadata_menu_button", "file_ops_menu_button", "more_menu_button",
        ):
            self.assertFalse(hasattr(window.gallery_pane, legacy_name))
        self.assertTrue(window.tags_workspace_button.isVisible())
        self.assertTrue(window.cluster_pane.details_scroll.isHidden())
        self.assertTrue(window.cluster_pane.meaning_label.isHidden())
        self.assertTrue(window.cluster_pane.shape_widget.isHidden())
        self.assertTrue(window.cluster_pane.basis_label.isHidden())

        window.set_clustering_mode("advanced")
        APP.processEvents()

        self.assertIs(window.clustering_gallery_stack.currentWidget(), window.gallery_pane)
        self.assertTrue(window.gallery_pane.selected_tags_button.isVisible())
        self.assertTrue(window.gallery_pane.actions_menu_button.isVisible())
        self.assertTrue(window.gallery_pane.metadata_menu.menuAction().isVisible())
        self.assertTrue(window.gallery_pane.file_ops_menu.menuAction().isVisible())
        self.assertEqual("advanced", window.gallery_pane.inspector_display_mode)
        self.assertTrue(window.tags_workspace_button.isVisible())
        self.assertFalse(window.cluster_pane.details_scroll.isHidden())
        self.assertFalse(window.cluster_pane.meaning_label.isHidden())
        self.assertFalse(window.cluster_pane.shape_widget.isHidden())
        self.assertFalse(window.cluster_pane.basis_label.isHidden())
        window.close()

    def test_production_library_workspace_tracks_cluster_selection(self):
        with TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
                settings_mod._RUNTIME_BASE_DIR = None
                layout = activate_runtime_root("ProductionLibraryWorkspaceTest")
                from apps.pyqt_production.app import ProductionClusterApp

                window = ProductionClusterApp(layout)
                window.show()
                window.source_pane.set_active_roots((tmp,))
                window.set_active_workspace("library")
                APP.processEvents()

                self.assertTrue(window.library_workspace_button.isVisible())
                self.assertIs(window.workspace_stack.currentWidget(), window.library_pane)
                self.assertTrue(window.source_pane.isHidden())
                self.assertTrue(window.mode_selector.isHidden())
                target = SelectionTarget(
                    paths=("/photos/first.jpg", "/photos/second.jpg"),
                    kind="cluster",
                    label="siglip | Cluster 3",
                    source_context={"comparison_key": "siglip", "cluster_id": 3},
                )
                window._on_cluster_selection_target_changed(target)
                self.assertEqual("siglip:3", window.library_pane._selected_cluster_key)
                self.assertEqual(target.paths, window.library_pane._selected_cluster_members)
                window.close()

    def test_library_compact_layout_keeps_actions_and_filters_readable(self):
        with TemporaryDirectory() as tmp:
            from app.services.library_catalog import LibraryCatalogService
            from ui.library_pane import LibraryPane

            pane = LibraryPane(
                lambda: "",
                lambda: None,
                catalog=LibraryCatalogService(db_path=Path(tmp) / "library.sqlite3"),
            )
            pane.resize(720, 900)
            pane.show()
            APP.processEvents()

            self.assertFalse(hasattr(pane, "start_here_label"))
            self.assertTrue(pane.library_help_button.isVisible())
            self.assertIn("explicitly registered roots", pane.library_help_button.toolTip())
            self.assertFalse(pane.status_label.isVisible())
            self.assertEqual(
                pane._timeline_layout_for_width(pane._timeline_usable_width()),
                pane._timeline_layout_mode,
            )
            self.assertEqual(pane._timeline_tab_height(), pane.tabs.height())
            for button in (
                pane.add_current_root_button,
                pane.add_root_button,
                pane.remove_root_button,
                pane.save_album_button,
                pane.delete_album_button,
                pane.timeline_reload_button,
                pane.timeline_refresh_dates_button,
            ):
                self.assertGreaterEqual(
                    button.contentsRect().width(),
                    button.fontMetrics().horizontalAdvance(button.text()),
                    button.text(),
                )
            self.assertGreaterEqual(pane.timeline_start.width(), pane.timeline_start.minimumWidth())
            self.assertGreaterEqual(pane.timeline_end.width(), pane.timeline_end.minimumWidth())
            self.assertGreaterEqual(pane.timeline_camera.width(), pane.timeline_camera.minimumWidth())

            pane.status_label.setText("Loading the full Library timeline…")
            self.assertTrue(pane.status_label.isVisible())
            pane.status_label.setText("Library timeline ready.")
            self.assertFalse(pane.status_label.isVisible())

            pane.tabs.setCurrentIndex(pane.tabs.indexOf(pane.duplicate_list.parentWidget()))
            APP.processEvents()
            self.assertGreaterEqual(pane.tabs.height(), pane._TAB_HEIGHTS["Cleanup"])
            pane.close()

    def test_library_timeline_controls_reflow_without_losing_values(self):
        with TemporaryDirectory() as tmp:
            from app.services.library_catalog import LibraryCatalogService
            from ui.library_pane import LibraryPane

            pane = LibraryPane(
                lambda: "",
                lambda: None,
                catalog=LibraryCatalogService(db_path=Path(tmp) / "library.sqlite3"),
            )
            pane.timeline_start.setText("2024-01-01")
            pane.timeline_end.setText("2024-12-31")
            pane.timeline_camera.setText("Test camera")
            pane.timeline_date_source.setCurrentIndex(pane.timeline_date_source.findData("filename_only"))
            pane.timeline_grouping.setCurrentIndex(pane.timeline_grouping.findData("year_month_day"))
            pane.timeline_filename_patterns.setPlainText("CAM_%Y%m%d_%H%M%S")
            pane.timeline_epoch_heuristic.setChecked(True)
            pane.show()

            def cell_position(key: str) -> tuple[int, int, int, int]:
                for index in range(pane.timeline_form_layout.count()):
                    if pane.timeline_form_layout.itemAt(index).widget() is pane._timeline_cells[key]:
                        return pane.timeline_form_layout.getItemPosition(index)
                self.fail(f"Timeline cell {key} is missing")

            primary_widths = pane._timeline_primary_widths()
            primary_spacing = pane.timeline_form_layout.horizontalSpacing()
            one_row_width = pane._layout_required_width(
                tuple(primary_widths[key] for key in pane._TIMELINE_PRIMARY_KEYS), primary_spacing
            )
            two_row_width = max(
                pane._layout_required_width(
                    tuple(primary_widths[key] for key in pane._TIMELINE_TWO_ROW_FILTER_KEYS), primary_spacing
                ),
                pane._layout_required_width(
                    tuple(primary_widths[key] for key in pane._TIMELINE_TWO_ROW_POLICY_KEYS), primary_spacing
                ),
            )

            # The breakpoints are derived from the visible controls, not a
            # hard-coded screen width. Keep the outer pane wide while giving
            # the scroll viewport an exact, deterministic allocation.
            for scroll_width, mode in (
                (one_row_width + 48, "wide"),
                (two_row_width + 24, "medium"),
            ):
                pane.resize(one_row_width + 200, 900)
                pane.timeline_scroll.setFixedWidth(scroll_width)
                APP.processEvents()
                pane._update_timeline_layout()
                APP.processEvents()
                self.assertEqual(mode, pane._timeline_layout_mode)
                usable_width = pane._timeline_usable_width()
                if mode == "wide":
                    self.assertGreaterEqual(usable_width, one_row_width)
                else:
                    self.assertGreaterEqual(usable_width, two_row_width)
                    self.assertLess(usable_width, one_row_width)
                self.assertEqual(pane._timeline_tab_height(), pane.tabs.height())
                expected_positions = (
                    {
                        "start": (0, 0, 1, 1),
                        "end": (0, 1, 1, 1),
                        "camera": (0, 2, 1, 1),
                        "date_source": (0, 3, 1, 1),
                        "grouping": (0, 4, 1, 1),
                        "show": (0, 5, 1, 1),
                        "advanced": (1, 0, 1, 6),
                    }
                    if mode == "wide"
                    else {
                        "start": (0, 0, 1, 1),
                        "end": (0, 1, 1, 1),
                        "camera": (0, 2, 1, 4),
                        "date_source": (1, 0, 1, 3),
                        "grouping": (1, 3, 1, 1),
                        "show": (1, 4, 1, 2),
                        "advanced": (2, 0, 1, 6),
                    }
                )
                for key, position in expected_positions.items():
                    self.assertEqual(position, cell_position(key), key)
                if mode == "wide":
                    self.assertEqual(0, pane.timeline_scroll.verticalScrollBar().maximum())
                    self.assertEqual(primary_widths["start"], pane._timeline_cells["start"].width())
                    self.assertEqual(primary_widths["end"], pane._timeline_cells["end"].width())
                    self.assertGreater(pane._timeline_cells["camera"].width(), primary_widths["camera"])
                    self.assertGreater(pane._timeline_cells["date_source"].width(), primary_widths["date_source"])
                self.assertEqual(7, pane.timeline_form_layout.count())
                self.assertTrue(pane.timeline_advanced_body.isHidden())
                self.assertEqual(3, pane.timeline_advanced_body.layout().count())
                for key in ("start", "end", "camera", "date_source", "grouping", "show", "advanced"):
                    self.assertTrue(pane._timeline_cells[key].isVisible())
                for key in ("patterns", "epoch", "refresh"):
                    self.assertFalse(pane._timeline_cells[key].isVisible())

            pane.timeline_scroll.setFixedWidth(max(1, two_row_width - 24))
            APP.processEvents()
            pane._update_timeline_layout()
            APP.processEvents()
            self.assertEqual("narrow", pane._timeline_layout_mode)
            self.assertLess(pane._timeline_usable_width(), two_row_width)
            self.assertEqual(
                {
                    "start": (0, 0, 1, 1),
                    "end": (1, 0, 1, 1),
                    "camera": (2, 0, 1, 1),
                    "date_source": (3, 0, 1, 1),
                    "grouping": (4, 0, 1, 1),
                    "show": (5, 0, 1, 1),
                    "advanced": (6, 0, 1, 1),
                },
                {key: cell_position(key) for key in ("start", "end", "camera", "date_source", "grouping", "show", "advanced")},
            )
            self.assertEqual(7, pane.timeline_form_layout.count())

            self.assertEqual("2024-01-01", pane.timeline_start.text())
            self.assertEqual("2024-12-31", pane.timeline_end.text())
            self.assertEqual("Test camera", pane.timeline_camera.text())
            self.assertEqual("filename_only", pane.timeline_date_source.currentData())
            self.assertEqual("year_month_day", pane.timeline_grouping.currentData())
            self.assertEqual("CAM_%Y%m%d_%H%M%S", pane.timeline_filename_patterns.toPlainText())
            self.assertTrue(pane.timeline_epoch_heuristic.isChecked())
            self.assertGreater(pane.timeline_scroll.verticalScrollBar().maximum(), 0)
            pane.close()

    def test_library_search_hides_empty_generated_context_panel(self):
        with TemporaryDirectory() as tmp:
            from PyQt6.QtWidgets import QListWidgetItem

            from app.services.library_catalog import LibraryCatalogService
            from ui.library_pane import LibraryPane

            pane = LibraryPane(
                lambda: "",
                lambda: None,
                catalog=LibraryCatalogService(db_path=Path(tmp) / "library.sqlite3"),
            )
            pane.resize(1200, 900)
            pane.show()
            pane.tabs.setCurrentIndex(pane.tabs.indexOf(pane.search_field.parentWidget()))
            APP.processEvents()

            self.assertFalse(pane.context_results_panel.isVisible())
            self.assertFalse(pane.open_context_button.isEnabled())
            self.assertEqual(pane._TAB_HEIGHTS["Search"], pane.tabs.height())

            pane.context_results.addItem(QListWidgetItem("A generated context"))
            pane._set_context_results_panel_visible(True)
            APP.processEvents()
            self.assertTrue(pane.context_results_panel.isVisible())
            self.assertEqual(pane._SEARCH_CONTEXT_TAB_HEIGHT, pane.tabs.height())
            pane.context_results.setCurrentRow(0)
            self.assertTrue(pane.open_context_button.isEnabled())

            pane.context_results.clear()
            pane._set_context_results_panel_visible(False)
            APP.processEvents()
            self.assertFalse(pane.context_results_panel.isVisible())
            self.assertFalse(pane.open_context_button.isEnabled())
            self.assertEqual(pane._TAB_HEIGHTS["Search"], pane.tabs.height())
            pane.close()

    def test_library_timeline_filename_patterns_are_visible_persisted_and_refreshable(self):
        with TemporaryDirectory() as tmp:
            from app.services.library_catalog import LibraryCatalogService
            from ui.library_pane import LibraryPane, TimelineDatePreferences

            saved: list[TimelineDatePreferences] = []
            pane = LibraryPane(
                lambda: "",
                lambda: None,
                catalog=LibraryCatalogService(db_path=Path(tmp) / "library.sqlite3"),
                timeline_date_preferences_provider=lambda: TimelineDatePreferences(
                    "filename_only", ("CAM_%Y%m%d_%H%M%S",), True, "year_month_day"
                ),
                timeline_date_preferences_changed=saved.append,
            )
            pane.show()
            self._wait_for(lambda: pane._catalog_snapshot_job is None)

            self.assertEqual("filename_only", pane.timeline_date_source.currentData())
            self.assertEqual("year_month_day", pane.timeline_grouping.currentData())
            self.assertEqual("CAM_%Y%m%d_%H%M%S", pane.timeline_filename_patterns.toPlainText())
            self.assertTrue(pane.timeline_epoch_heuristic.isChecked())
            self.assertTrue(pane.timeline_filename_patterns.tabChangesFocus())
            self.assertIn("named rule", pane.timeline_date_help_button.toolTip())

            pane.timeline_filename_patterns.setPlainText("ARCHIVE_%Y.%m.%d-%H.%M.%S")
            pane.timeline_epoch_heuristic.setChecked(False)
            pane.refresh_timeline_dates()
            self.assertEqual(("ARCHIVE_%Y.%m.%d-%H.%M.%S",), pane.catalog.filename_date_patterns)
            self.assertEqual(
                TimelineDatePreferences("filename_only", ("ARCHIVE_%Y.%m.%d-%H.%M.%S",), False, "year_month_day"),
                saved[-1],
            )
            self.assertIn("No registered active roots", pane.status_label.text())
            pane.close()

    def test_library_timeline_uses_nested_virtual_sections_while_search_progressively_pages(self):
        with TemporaryDirectory() as tmp:
            from app.services.library_catalog import CatalogTimeline, LibraryCatalogService, TimelineMonth, TimelineYear
            from ui.library_pane import LibraryPane

            pane = LibraryPane(
                lambda: "",
                lambda: None,
                catalog=LibraryCatalogService(db_path=Path(tmp) / "library.sqlite3"),
            )
            timeline = CatalogTimeline(
                years=(
                    TimelineYear(2026, (TimelineMonth(2026, 3, ("/march.jpg",)), TimelineMonth(2026, 2, ("/february.jpg",)))),
                    TimelineYear(2025, (TimelineMonth(2025, 12, ("/december.jpg",)),)),
                ),
                total_count=3,
            )
            sections, collapsed = pane._timeline_sections(timeline)
            pane.timeline_gallery.set_sections_with_collapsed(sections, collapsed_section_ids=collapsed)
            pane._timeline_paths = timeline.image_paths
            pane._update_controls()

            self.assertIs(pane.gallery_stack.currentWidget(), pane.timeline_gallery)
            self.assertFalse(pane.load_more_button.isVisible())
            self.assertTrue(pane.open_gallery_button.isEnabled())
            headers = [
                pane.timeline_gallery._model.data(pane.timeline_gallery._model.index(row, 0), pane.timeline_gallery._model.HeaderRole).section_id
                for row in range(pane.timeline_gallery._model.rowCount())
                if pane.timeline_gallery._model.data(pane.timeline_gallery._model.index(row, 0), pane.timeline_gallery._model.HeaderRole)
            ]
            self.assertEqual(["timeline:year:2026", "timeline:year:2026:month:03", "timeline:year:2026:month:02", "timeline:year:2025"], headers)

            pane.tabs.setCurrentIndex(pane.tabs.indexOf(pane.search_field.parentWidget()))
            APP.processEvents()
            self.assertIs(pane.gallery_stack.currentWidget(), pane.gallery)
            self.assertFalse(pane.load_more_button.isHidden())
            self.assertEqual(500, pane.PAGE_SIZE)
            pane.shutdown_jobs(timeout_ms=1_000)
            pane.close()

    def test_library_timeline_photo_total_stays_separate_from_thumbnail_tasks(self):
        with TemporaryDirectory() as tmp:
            from app.services.library_catalog import CatalogTimeline, LibraryCatalogService, TimelineMonth, TimelineYear
            from ui.job_manager import JobManager
            from ui.library_pane import LibraryPane

            paths = tuple(f"/fixture/{index:02d}.jpg" for index in range(36))
            timeline = CatalogTimeline(
                years=(TimelineYear(2026, (TimelineMonth(2026, 9, paths),)),),
                total_count=36,
            )
            pane = LibraryPane(
                lambda: "",
                lambda: None,
                catalog=LibraryCatalogService(db_path=Path(tmp) / "library.sqlite3"),
            )
            manager = JobManager()
            pane.timeline_gallery.set_job_manager(manager, origin="Library")
            try:
                with patch.object(pane.timeline_gallery, "_queue_visible_loads_after_layout"):
                    pane._publish_timeline(timeline)
                self.assertEqual("Showing 36 photos in 1 year", pane.timeline_count.text())

                for index in range(3):
                    pane.timeline_gallery._begin_viewport_task(1, paths[index], "thumbnail")
                self.assertEqual("Loading thumbnails 0/3 visible", pane.timeline_gallery.status_label.text())
                self.assertEqual("Showing 36 photos in 1 year", pane.timeline_count.text())
                job = manager.get(pane.timeline_gallery._viewport_job_id)
                self.assertIsNotNone(job)
                self.assertEqual("Library", job.origin)
            finally:
                pane.shutdown_jobs(timeout_ms=1_000)
                pane.close()

    def test_library_search_progressively_publishes_every_page_after_the_first_viewport_batch(self):
        with TemporaryDirectory() as tmp:
            from app.services.library_catalog import CatalogAsset, CatalogPage, LibraryCatalogService
            from ui.library_pane import LibraryPane

            pane = LibraryPane(
                lambda: "",
                lambda: None,
                catalog=LibraryCatalogService(db_path=Path(tmp) / "library.sqlite3"),
            )
            assets = tuple(
                CatalogAsset(f"/fixture/{index:04d}.jpg", "root", "", "unparsed", "", "", 1, 1, 1, ".jpg")
                for index in range(1_001)
            )
            offsets: list[int] = []

            def query(query):
                offsets.append(query.offset)
                start = int(query.offset)
                end = min(len(assets), start + int(query.limit))
                return CatalogPage(assets[start:end], len(assets), end if end < len(assets) else None)

            def publish(paths):
                pane.gallery.images = list(paths)

            pane.gallery.update_gallery = publish
            try:
                pane.tabs.setCurrentIndex(pane.tabs.indexOf(pane.search_field.parentWidget()))
                with patch.object(pane.catalog, "query_assets", side_effect=query):
                    pane.run_search()
                    self._wait_for(lambda: len(pane.gallery.images) == len(assets), timeout_s=5.0)
                self.assertEqual([0, 500, 1_000], offsets)
                self.assertFalse(pane._asset_auto_loading)
                self.assertEqual("Showing 1,001 of 1,001 matching photos", pane.timeline_count.text())
            finally:
                pane.shutdown_jobs(timeout_ms=1_000)
                pane.close()

    def test_library_duplicate_trash_preview_requires_explicit_candidate_checks(self):
        from PyQt6.QtWidgets import QDialogButtonBox

        from ui.library_pane import _DuplicateTrashPreviewDialog

        dialog = _DuplicateTrashPreviewDialog("Hash duplicate", ["/fixture/a.jpg", "/fixture/b.jpg"])
        try:
            accept = dialog.buttons.button(QDialogButtonBox.StandardButton.Ok)
            self.assertEqual([], dialog.selected_paths())
            self.assertFalse(accept.isEnabled())
            dialog._set_all_checked(True)
            self.assertEqual(["/fixture/a.jpg", "/fixture/b.jpg"], dialog.selected_paths())
            self.assertTrue(accept.isEnabled())
            dialog._set_all_checked(False)
            self.assertEqual([], dialog.selected_paths())
            self.assertFalse(accept.isEnabled())
        finally:
            dialog.close()

    def test_production_process_jobs_do_not_spawn_when_cancelled_while_queued(self):
        from app.services.model_downloads import ModelDownloadItem
        from apps.pyqt_production.app import ProductionClusterApp
        from apps.pyqt_production.worker_protocol import ProductionClusterRequest
        from ui.work_coordinator import JobSpec

        with TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
                settings_mod._RUNTIME_BASE_DIR = None
                layout = activate_runtime_root("ProductionQueuedProcessCoordinationTest")
                window = ProductionClusterApp(layout)
                try:
                    self._wait_for_storage_idle(window)

                    model_blocker = window.work_coordinator.submit(
                        JobSpec("Model cache writer", model_cache_write=True),
                        lambda _job_id, _fallback: None,
                    )
                    with patch.object(window.model_download_controller, "start", return_value=True) as start_download:
                        self.assertTrue(window._queue_model_download([ModelDownloadItem("fast_preview")]))
                        model_job_id = window._model_download_job_id
                        self.assertEqual("queued", window.job_manager.get(model_job_id).status)
                        start_download.assert_not_called()
                        window.work_coordinator.cancel(model_job_id)
                        self.assertIsNone(window._model_download_job_id)
                        window.work_coordinator.finish(model_blocker)
                        APP.processEvents()
                        start_download.assert_not_called()

                    source_blocker = window.work_coordinator.submit(
                        JobSpec("Source writer", source_writes=(tmp,)),
                        lambda _job_id, _fallback: None,
                    )
                    request = ProductionClusterRequest(
                        directory=tmp,
                        source_roots=[tmp],
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
                    with patch.object(window.session_controller, "start", return_value=True) as start_session:
                        self.assertTrue(window._start_request(request, run_origin="test"))
                        session_job_id = window._active_job_id
                        self.assertEqual("queued", window.job_manager.get(session_job_id).status)
                        start_session.assert_not_called()
                        window.cancel_clustering()
                        self.assertIsNone(window._active_job_id)
                        window.work_coordinator.finish(source_blocker)
                        APP.processEvents()
                        start_session.assert_not_called()
                finally:
                    window.close()

    def test_clustering_launch_uses_background_readiness_snapshot_without_qt_probe(self):
        from apps.pyqt_production.app import ProductionClusterApp
        from apps.pyqt_production.worker_protocol import ProductionClusterRequest

        with TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
                settings_mod._RUNTIME_BASE_DIR = None
                layout = activate_runtime_root("ProductionReadinessSnapshotLaunchTest")
                window = ProductionClusterApp(layout)
                try:
                    self._wait_for_storage_idle(window)
                    policy = ExecutionPolicy(
                        preferred_mode="cpu",
                        effective_mode="cpu",
                        reason="Fixture readiness snapshot.",
                    )
                    window._startup_readiness_report = SimpleNamespace(
                        execution_policy=policy,
                        clustering_ready=True,
                        clustering_message="Fixture clustering is ready.",
                        face_ready=False,
                        face_message="Not used by this test.",
                    )
                    request = ProductionClusterRequest(
                        directory=tmp,
                        source_roots=[tmp],
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
                    with (
                        patch.object(
                            window.runtime_service,
                            "select_policy",
                            side_effect=AssertionError("Qt launch probed runtime policy"),
                        ),
                        patch.object(
                            window.runtime_service,
                            "detect",
                            side_effect=AssertionError("Qt launch probed runtime capabilities"),
                        ),
                        patch.object(window.session_controller, "start", return_value=True) as start_session,
                    ):
                        self.assertTrue(window._start_request(request, run_origin="test"))
                        start_session.assert_called_once()
                    active_job_id = window._active_job_id
                    self.assertIsNotNone(active_job_id)
                    window.work_coordinator.finish(active_job_id, status="cancelled")
                    APP.processEvents()

                    window._startup_readiness_report = SimpleNamespace(
                        execution_policy=policy,
                        clustering_ready=True,
                    )
                    stale_request = replace(request, preferred_execution_mode="cuda")
                    with (
                        patch.object(window, "_begin_startup_readiness_check") as begin_readiness,
                        patch.object(window.session_controller, "start") as stale_start,
                    ):
                        self.assertFalse(window._start_request(stale_request, run_origin="test"))
                        begin_readiness.assert_called_once_with()
                        stale_start.assert_not_called()

                    unavailable_policy = ExecutionPolicy(
                        preferred_mode="cuda",
                        effective_mode="cpu",
                        reason="CUDA provider unavailable.",
                        error="CUDA provider unavailable.",
                    )
                    window._startup_readiness_report = SimpleNamespace(
                        execution_policy=unavailable_policy,
                        clustering_ready=False,
                    )
                    with (
                        patch("apps.pyqt_production.app.errorBox") as error_box,
                        patch.object(window.session_controller, "start") as unavailable_start,
                    ):
                        self.assertFalse(window._start_request(stale_request, run_origin="test"))
                        unavailable_start.assert_not_called()
                        error_box.assert_called_once_with(
                            "CUDA unavailable",
                            "CUDA provider unavailable.",
                        )
                finally:
                    window.close()

    def test_cuda_oom_is_terminal_and_never_silently_retries_on_cpu(self):
        from apps.pyqt_production.app import ProductionClusterApp
        from ui.work_coordinator import JobSpec

        with TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
                settings_mod._RUNTIME_BASE_DIR = None
                window = ProductionClusterApp(
                    activate_runtime_root("ProductionCudaOomTerminalTest")
                )
                try:
                    self._wait_for_storage_idle(window)
                    job_id = window.work_coordinator.submit(
                        JobSpec("Explicit CUDA run", uses_gpu=True),
                        lambda _job_id, _fallback: None,
                    )
                    window._coordinated_process_job_ids.add(job_id)
                    window._active_job_id = job_id
                    with (
                        patch("apps.pyqt_production.app.errorBox") as error_box,
                        patch.object(window.session_controller, "start") as restart,
                    ):
                        window._on_clustering_failed("CUDA out of memory while loading the model")
                    state = window.job_manager.get(job_id)
                    self.assertEqual("failed", state.status)
                    self.assertIn("out of memory", state.error.lower())
                    restart.assert_not_called()
                    error_box.assert_called_once()
                    self.assertEqual("Failed", window.footer_bar.status_label.text())
                finally:
                    window.close()

    def test_settings_runtime_apply_only_invalidates_for_background_readiness(self):
        from apps.pyqt_production.app import ProductionClusterApp

        with TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
                settings_mod._RUNTIME_BASE_DIR = None
                layout = activate_runtime_root("ProductionSettingsRuntimeInvalidationTest")
                window = ProductionClusterApp(layout)
                try:
                    self._wait_for_storage_idle(window)
                    window._startup_readiness_report = object()
                    with (
                        patch.object(
                            window.runtime_service,
                            "select_policy",
                            side_effect=AssertionError("Settings apply probed runtime policy"),
                        ),
                        patch.object(
                            window.runtime_service,
                            "detect",
                            side_effect=AssertionError("Settings apply probed runtime capabilities"),
                        ),
                    ):
                        window._apply_runtime_status()
                    self.assertIsNone(window._startup_readiness_report)
                    self.assertEqual("cpu", window.execution_policy.preferred_mode)
                    self.assertIn("background", window.execution_policy.reason.lower())
                    self.assertEqual("Runtime checking… · Checking", window.runtime_badge.text())
                finally:
                    window.close()

    def test_library_timeline_grouping_drop_down_builds_year_month_week_and_day_views(self):
        with TemporaryDirectory() as tmp:
            from app.services.library_catalog import CatalogTimeline, LibraryCatalogService, TimelineDay, TimelineMonth, TimelineYear
            from ui.library_pane import LibraryPane

            pane = LibraryPane(
                lambda: "",
                lambda: None,
                catalog=LibraryCatalogService(db_path=Path(tmp) / "library.sqlite3"),
            )
            timeline = CatalogTimeline(
                years=(
                    TimelineYear(
                        2024,
                        (
                            TimelineMonth(
                                2024,
                                5,
                                ("/may-20.jpg", "/may-19.jpg", "/may-01.jpg"),
                                (
                                    TimelineDay(2024, 5, 20, 21, ("/may-20.jpg",)),
                                    TimelineDay(2024, 5, 19, 20, ("/may-19.jpg",)),
                                    TimelineDay(2024, 5, 1, 18, ("/may-01.jpg",)),
                                ),
                            ),
                        ),
                    ),
                ),
                total_count=3,
            )

            pane.timeline_grouping.setCurrentIndex(pane.timeline_grouping.findData("year"))
            year_sections, _collapsed = pane._timeline_sections(timeline)
            self.assertEqual(("/may-20.jpg", "/may-19.jpg", "/may-01.jpg"), year_sections[0].paths)
            self.assertEqual((), year_sections[0].children)

            pane.timeline_grouping.setCurrentIndex(pane.timeline_grouping.findData("year_month_week"))
            week_sections, _collapsed = pane._timeline_sections(timeline)
            week_children = week_sections[0].children[0].children
            self.assertEqual(["Week 21", "Week 20", "Week 18"], [section.title for section in week_children])
            self.assertEqual(("/may-20.jpg",), week_children[0].paths)

            pane.timeline_grouping.setCurrentIndex(pane.timeline_grouping.findData("year_month_day"))
            day_sections, _collapsed = pane._timeline_sections(timeline)
            day_children = day_sections[0].children[0].children
            self.assertEqual(["May 20, 2024", "May 19, 2024", "May 1, 2024"], [section.title for section in day_children])
            pane.close()

    def test_library_timeline_labels_unparsed_filename_time_explicitly(self):
        with TemporaryDirectory() as tmp:
            from app.services.library_catalog import CatalogTimeline, LibraryCatalogService, TimelineMonth, TimelineYear
            from ui.library_pane import LibraryPane

            pane = LibraryPane(
                lambda: "",
                lambda: None,
                catalog=LibraryCatalogService(db_path=Path(tmp) / "library.sqlite3"),
            )
            sections, _collapsed = pane._timeline_sections(
                CatalogTimeline(
                    years=(TimelineYear(0, (TimelineMonth(0, 0, ("/unparsed.jpg",)),)),),
                    total_count=1,
                )
            )

            self.assertEqual("Unparsed", sections[0].title)
            self.assertEqual("Unparsed filename time", sections[0].children[0].title)
            pane.close()

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
        window.source_pane.set_active_roots((str(Path(__file__).parent),))
        window.show()
        window.set_clustering_mode("advanced")
        APP.processEvents()

        pane = window.clustering_pane
        self.assertEqual("balanced", pane.preset_combo.currentData())
        self.assertTrue(pane.technical_panel.isVisible())
        self.assertTrue(pane.backend_checkboxes["hdbscan"].isVisible())

        self.assertEqual(
            ["Quick groups", "Similar scenes", "Events", "Documents", "People-heavy", "Detailed groups", "Custom"],
            [pane.preset_combo.itemText(index) for index in range(pane.preset_combo.count())],
        )
        for preset in ("fast_preview", "events", "documents", "people_heavy", "high_quality"):
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

    def test_production_file_hdbscan_options_persist_and_enter_request(self):
        from apps.pyqt_production.app import ProductionClusterApp, RUNTIME_LAYOUT
        from ui.mode_panes import ClusteringOptionsPane

        window = ProductionClusterApp(RUNTIME_LAYOUT)
        pane = window.clustering_pane
        pane.preset_combo.setCurrentIndex(pane.preset_combo.findData("custom"))
        for checkbox in pane.backend_checkboxes.values():
            checkbox.setChecked(False)
        pane.backend_checkboxes["hdbscan"].setChecked(True)
        pane.hdbscan_min_cluster_size_spin.setValue(9)
        pane.hdbscan_min_samples_spin.setValue(4)
        pane.hdbscan_cluster_selection_epsilon_spin.setValue(0.175)
        pane.hdbscan_allow_single_cluster_checkbox.setChecked(True)
        APP.processEvents()

        expected = {
            "hdbscan": {
                "min_cluster_size": 9,
                "min_samples": 4,
                "cluster_selection_epsilon": 0.175,
                "allow_single_cluster": True,
            }
        }
        self.assertFalse(pane.hdbscan_options_group.isHidden())
        self.assertEqual(expected, window._build_request().backend_options_by_backend)
        state = pane.export_state()
        restored = ClusteringOptionsPane(option_scope="production")
        restored.show()
        restored.apply_state(state)
        APP.processEvents()
        self.assertEqual(expected, restored.backend_options_by_backend())
        self.assertTrue(restored.hdbscan_options_group.isVisible())
        restored.close()
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
        from apps.pyqt_production.app import RUNTIME_LAYOUT, _select_settings_storage_section
        from apps.pyqt_production.settings_dialog import ProductionSettingsDialog
        from app.services.cache_maintenance import CacheClearResult, CacheUsageSummary, GeneratedStorageSummary

        store = QSettings("ClusterLensTests", "ProductionSettingsDialog")
        store.clear()
        store.sync()
        dialog = ProductionSettingsDialog(
            store,
            RuntimeCapabilityService(),
            describe_rebuildable_caches=lambda: CacheUsageSummary(
                cache_root=str(RUNTIME_LAYOUT.cache_dir),
                target_bytes={"embeddings.sqlite3": 0, "cluster_results/": 0},
                total_bytes=0,
            ),
            describe_generated_storage=lambda: GeneratedStorageSummary(
                runtime_root=str(RUNTIME_LAYOUT.root),
                config_location="test-settings.ini",
                cache_root=str(RUNTIME_LAYOUT.cache_dir),
                target_bytes={
                    "logs": 0,
                    "thumbnails": 0,
                    "rebuildable_caches": 0,
                    "library_catalog": 0,
                    "face_databases": 0,
                    "ann_files": 0,
                    "model_caches": 0,
                    "temp_files": 0,
                    "crash_reports": 0,
                    "support_bundles": 0,
                    "benchmarks": 0,
                    "model_assets": 0,
                },
                target_paths={},
                total_bytes=0,
            ),
            clear_rebuildable_caches=lambda: CacheClearResult((), 0, ()),
            clear_library_catalog=lambda: CacheClearResult(("library_catalog",), 0, ()),
            clear_runtime_temp_files=lambda: CacheClearResult((), 0, ()),
            clear_face_storage=lambda: CacheClearResult((), 0, ()),
            clear_model_caches=lambda: CacheClearResult((), 0, ()),
            clear_logs=lambda: CacheClearResult((), 0, ()),
            clear_runtime_reports=lambda: CacheClearResult((), 0, ()),
            clear_model_assets=lambda: CacheClearResult((), 0, ()),
            can_clear_rebuildable_caches=lambda: True,
            runtime_layout=RUNTIME_LAYOUT,
            support_metadata_provider=lambda: {},
        )
        self._wait_for(lambda: dialog._cache_usage_thread is None, timeout_s=5.0)
        self.assertTrue(_select_settings_storage_section(dialog))
        combined_text = "\n".join(
            [
                dialog.runtime_text.toPlainText(),
                dialog.cache_usage_text.toPlainText(),
                dialog.cache_status_label.text(),
            ]
        ).lower()
        self.assertEqual("apps.pyqt_production.settings_dialog", ProductionSettingsDialog.__module__)
        self.assertEqual("Rescan GPU Resources", dialog.refresh_runtime_button.text())
        self.assertIn("GPU acceleration checklist", dialog.runtime_text.toPlainText())
        self.assertIn("no model downloads", dialog.runtime_text.toPlainText().lower())
        self.assertIn("ONNX face indexing:", dialog.runtime_text.toPlainText())
        self.assertIn("Cosine K-means:", dialog.runtime_text.toPlainText())
        self.assertIn("CPU SIMD dispatch:", dialog.runtime_text.toPlainText())
        self.assertIn("CPU BLAS:", dialog.runtime_text.toPlainText())
        self.assertIn("Face databases", dialog.generated_storage_text.toPlainText())
        self.assertIn("Library catalog", dialog.generated_storage_text.toPlainText())
        self.assertEqual("Clear Library Cache", dialog.clear_library_catalog_button.text())
        self.assertEqual("Clear Reports", dialog.clear_runtime_reports_button.text())
        self.assertEqual("Clear Installed Model Assets", dialog.clear_model_assets_button.text())
        self.assertTrue(dialog.clear_runtime_temp_button.isEnabled())
        self.assertTrue(dialog.clear_library_catalog_button.isEnabled())

        self.assertTrue(dialog.clear_runtime_reports_button.isEnabled())
        self.assertTrue(dialog.clear_model_assets_button.isEnabled())
        self.assertNotIn("face/search", combined_text)
        self.assertNotIn("convnext", combined_text)
        self.assertEqual("Disabled", dialog.minimum_image_width.text())
        self.assertIn("Thumbnail-name matching: off", dialog.source_filter_summary.text())
        dialog.ignore_thumbnail_like.setChecked(True)
        dialog.minimum_image_width.setValue(640)
        dialog.minimum_image_height.setValue(480)
        dialog.minimum_file_size_kib.setValue(128)
        values = dialog.values()
        self.assertTrue(values["source_filters/ignore_thumbnail_like"])
        self.assertEqual(640, values["source_filters/min_width"])
        self.assertEqual(480, values["source_filters/min_height"])
        self.assertEqual(128 * 1024, values["source_filters/min_file_size_bytes"])
        self.assertIn("Thumbnail-name matching: on", dialog.source_filter_summary.text())
        self.assertIn("width ≥ 640 px", dialog.source_filter_summary.text())
        from app.services.source_admission import SourceAdmissionPolicy

        for key in (
            "source_filters/ignore_thumbnail_like",
            "source_filters/min_width",
            "source_filters/min_height",
            "source_filters/min_file_size_bytes",
        ):
            dialog.settings_registry.set(store, key, values[key])
        restored_policy = SourceAdmissionPolicy.from_settings_store(store)
        self.assertEqual(
            SourceAdmissionPolicy(True, minimum_width=640, minimum_height=480, minimum_file_size_bytes=128 * 1024),
            restored_policy,
        )
        dialog.close()

    def test_production_cache_clear_reports_late_cancel_as_committed_retryable_state(self):
        from apps.pyqt_production.app import ProductionClusterApp
        from app.services.cache_maintenance import CacheUsageSummary
        from infra.cancel import Cancelled

        class FakeCacheMaintenance:
            def __init__(self) -> None:
                self.describe_calls = 0

            def describe_rebuildable_caches(self, *, cancel_check=None):
                self.describe_calls += 1
                total = 8192 if self.describe_calls == 1 else 0
                return CacheUsageSummary("/cache", {"embeddings.sqlite3": total}, total)

            def clear_rebuildable_disk_targets(self, **_kwargs):
                raise Cancelled()

            @staticmethod
            def pending_rebuildable_cleanup_targets():
                return ("cluster_results/",)

        fake = SimpleNamespace(
            cache_maintenance_service=FakeCacheMaintenance(),
            _clear_embedding_cache=lambda **_kwargs: (["embeddings.sqlite3"], []),
        )

        result = ProductionClusterApp._clear_rebuildable_caches(
            fake,
            cancel_check=lambda: False,
        )

        self.assertEqual(("embeddings.sqlite3",), result.cleared_targets)
        self.assertEqual(("cluster_results/",), result.retry_targets)
        self.assertTrue(result.completion_survives_cancellation)
        self.assertTrue(any("after a cache target committed" in failure for failure in result.failures))

    def test_production_settings_resource_rescan_forces_background_runtime_refresh(self):
        from apps.pyqt_production.app import RUNTIME_LAYOUT
        from apps.pyqt_production.settings_dialog import ProductionSettingsDialog

        store = QSettings("ClusterLensTests", "ProductionRuntimeRescan")
        service = RuntimeCapabilityService()
        dialog = ProductionSettingsDialog(
            store,
            service,
            runtime_layout=RUNTIME_LAYOUT,
            support_metadata_provider=lambda: {},
        )
        preferred_mode = str(dialog.execution_mode.currentData() or "auto")
        rescanned: list[bool] = []
        dialog.runtime_rescanned.connect(lambda: rescanned.append(True))
        with patch.object(service, "diagnostics", wraps=service.diagnostics) as diagnostics:
            dialog.refresh_runtime_button.click()
            self.assertFalse(dialog.refresh_runtime_button.isEnabled())
            self.assertFalse(dialog.verify_runtime_button.isEnabled())
            self._wait_for(lambda: dialog._verify_thread is None, timeout_s=5.0)

        diagnostics.assert_called_once_with(preferred_mode, refresh=False)
        self.assertTrue(dialog.refresh_runtime_button.isEnabled())
        self.assertTrue(dialog.verify_runtime_button.isEnabled())
        self.assertIn("face indexing:", dialog.runtime_text.toPlainText().lower())
        self.assertIn("gpu acceleration checklist", dialog.runtime_text.toPlainText().lower())
        self.assertEqual([True], rescanned)
        dialog.close()

    def test_production_settings_rescan_checks_cuda_even_when_cpu_mode_is_active(self):
        from apps.pyqt_production.app import RUNTIME_LAYOUT
        from apps.pyqt_production.settings_dialog import ProductionSettingsDialog

        store = QSettings("ClusterLensTests", "ProductionRuntimeRescanCpuMode")
        service = RuntimeCapabilityService()
        dialog = ProductionSettingsDialog(
            store,
            service,
            runtime_layout=RUNTIME_LAYOUT,
            support_metadata_provider=lambda: {},
        )
        dialog._set_combo_data(dialog.execution_mode, "cpu")
        with patch.object(service, "select_policy", wraps=service.select_policy) as select_policy:
            dialog.refresh_runtime_button.click()
            self._wait_for(lambda: dialog._verify_thread is None, timeout_s=5.0)

        self.assertTrue(
            any(
                call.kwargs == {"preferred_mode": "cuda", "refresh": True}
                for call in select_policy.call_args_list
            )
        )
        self.assertIn("cuda onnx", dialog.runtime_text.toPlainText().lower())
        dialog.close()

    def test_production_pane_hide_buttons_and_toolbar_toggles_work(self):
        from apps.pyqt_production.app import ProductionClusterApp, RUNTIME_LAYOUT

        window = ProductionClusterApp(RUNTIME_LAYOUT)
        window.source_pane.set_active_roots((str(Path(__file__).parent),))
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
                window.set_active_workspace("tags")
                self.assertTrue(window.tags_pane.apply_suggestions_button.isEnabled())
                self.assertIn("food", window.tags_pane.suggestion_label.text())

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

    def test_production_tags_generate_selected_cluster_suggestions_on_demand(self):
        from apps.pyqt_production.app import ProductionClusterApp
        from app.services.cluster_meanings import ClusterMeaning, ClusterMeaningLabel, ClusterMeaningService

        with TemporaryDirectory() as tmp:
            runtime_root = Path(tmp) / "runtime"
            images_root = Path(tmp) / "images"
            images_root.mkdir(parents=True, exist_ok=True)
            image_path = images_root / "a.jpg"
            Image.new("RGB", (24, 24), (20, 30, 40)).save(image_path)
            key = "dino::semantic::hdbscan"
            paths = [str(image_path)]
            meaning = ClusterMeaning(
                cluster_id=3,
                labels=(ClusterMeaningLabel("beach", "a photo of a beach", 0.9),),
                confidence="High",
                explanation_model="clip",
                image_count_used=1,
                status="ok",
            )

            with patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": str(runtime_root)}, clear=False):
                settings_mod._RUNTIME_BASE_DIR = None
                layout = activate_runtime_root("ProductionGenerateTagsTest")
                window = ProductionClusterApp(layout)
                self._wait_for_storage_idle(window)
                window.cluster_data = {key: {3: paths}}
                window.cluster_pane.update_clusters(window.cluster_data)
                window.cluster_pane.on_cluster_selected(window.cluster_pane._grid_model.index(0, 0))
                window.set_active_workspace("tags")
                self.assertTrue(window.tags_pane.generate_suggestions_button.isEnabled())

                with patch.object(
                    ClusterMeaningService,
                    "generate",
                    return_value=({key: {3: meaning}}, {"cluster_meaning_status": "ok"}),
                ) as generate:
                    window.generate_cluster_tag_suggestions()
                    self._wait_for(lambda: window._tag_suggestion_thread is None, timeout_s=5.0)

                generate.assert_called_once()
                self.assertEqual(meaning, window.cluster_meanings[key][3])
                self.assertTrue(window.tags_pane.apply_suggestions_button.isEnabled())
                self.assertIn("beach", window.tags_pane.suggestion_label.text())
                window.close()

    def test_production_tags_workspace_pages_current_folder_without_media_scan(self):
        from apps.pyqt_production.app import ProductionClusterApp

        with TemporaryDirectory() as tmp:
            runtime_root = Path(tmp) / "runtime"
            images_root = Path(tmp) / "images"
            other_root = Path(tmp) / "other"
            images_root.mkdir(parents=True, exist_ok=True)
            other_root.mkdir(parents=True, exist_ok=True)
            image_a = images_root / "a.jpg"
            image_b = images_root / "b.jpg"
            image_c = other_root / "c.jpg"
            for image_path in (image_a, image_b, image_c):
                Image.new("RGB", (24, 24), (20, 30, 40)).save(image_path)

            with patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": str(runtime_root)}, clear=False):
                settings_mod._RUNTIME_BASE_DIR = None
                layout = activate_runtime_root("ProductionTagsWorkspaceTest")
                window = ProductionClusterApp(layout)
                self._wait_for_storage_idle(window)
                window.source_pane.selected_directory = ""
                window._on_directory_changed("")
                window.set_active_workspace("tags")
                window.tags_pane.scope_combo.setCurrentIndex(1)
                window.tags_pane.refresh()
                self._wait_for(
                    lambda: (
                        window.tags_pane.inventory_model.rowCount() == 0
                        and not window.tags_pane._inventory_loading
                        and "Choose a folder" in window.tags_pane.status_label.text()
                    ),
                    timeout_s=5.0,
                )
                window.source_pane.set_selected_directory(str(images_root))
                window.image_tag_service.apply_tag_edit([str(image_a), str(image_b)], add_tags=["Beach"], mirror_to_exif=False)
                window.image_tag_service.apply_tag_edit([str(image_c)], add_tags=["City"], mirror_to_exif=False)
                window.tags_pane.refresh()
                self._wait_for(lambda: window.tags_pane.inventory_model.rowCount() == 1, timeout_s=5.0)
                entry = window.tags_pane.inventory_model.item_at(0)
                self.assertEqual("Beach", entry.title)
                window.tags_pane.inventory_list.setCurrentIndex(window.tags_pane.inventory_model.index(0, 0))
                self._wait_for(
                    lambda: window.tags_pane.gallery.images == [str(image_a), str(image_b)], timeout_s=5.0
                )
                self.assertEqual(2, window.tags_pane._photo_total)
                window.close()

    def test_production_tags_workspace_coalesces_rapid_page_and_selection_requests(self):
        from apps.pyqt_production.app import ProductionClusterApp

        with TemporaryDirectory() as tmp:
            runtime_root = Path(tmp) / "runtime"
            images_root = Path(tmp) / "images"
            images_root.mkdir(parents=True, exist_ok=True)
            image_a = images_root / "a.jpg"
            image_b = images_root / "b.jpg"
            for image_path in (image_a, image_b):
                Image.new("RGB", (24, 24), (20, 30, 40)).save(image_path)

            with patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": str(runtime_root)}, clear=False):
                settings_mod._RUNTIME_BASE_DIR = None
                layout = activate_runtime_root("ProductionTagsCoalesceTest")
                window = ProductionClusterApp(layout)
                self._wait_for_storage_idle(window)
                window.source_pane.set_selected_directory(str(images_root))
                window.image_tag_service.apply_tag_edit([str(image_a)], add_tags=["Beach"], mirror_to_exif=False)
                window.image_tag_service.apply_tag_edit([str(image_b)], add_tags=["Family"], mirror_to_exif=False)
                pane = window.tags_pane
                pane.INVENTORY_PAGE_SIZE = 1
                pane.PHOTO_PAGE_SIZE = 1
                pane.scope_combo.setCurrentIndex(1)
                window.set_active_workspace("tags")
                pane.refresh()
                self._wait_for(lambda: pane.inventory_model.rowCount() == 1 and pane._inventory_total == 2, timeout_s=5.0)

                pane._load_more_inventory()
                pane._load_more_inventory()
                self._wait_for(lambda: pane.inventory_model.rowCount() == 2 and not pane._inventory_loading, timeout_s=5.0)
                self.assertEqual(["Beach", "Family"], [pane.inventory_model.item_at(index).title for index in range(2)])

                pane.inventory_list.setCurrentIndex(pane.inventory_model.index(0, 0))
                pane.inventory_list.setCurrentIndex(pane.inventory_model.index(1, 0))
                self._wait_for(lambda: pane.gallery.images == [str(image_b)] and not pane._photo_loading, timeout_s=5.0)
                self.assertEqual("Family", pane._selected_tag)
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
                window.tags_pane.set_filter(["selfie"], "Any")
                window._startup_readiness_report = SimpleNamespace(
                    clustering_ready=True,
                    clustering_message="Clustering ready.",
                    face_ready=False,
                    face_message="Face setup is not part of this preflight test.",
                )

                discovery_thread_flags: list[bool] = []
                captured_requests: list[object] = []
                discovery_entered = Event()
                release_discovery = Event()
                original_discover = ImageDiscoveryService.discover_result

                def _discover(service, directory, recursive=None, progress_callback=None, cancel_check=None):
                    discovery_thread_flags.append(QThread.currentThread() is APP.thread())
                    discovery_entered.set()
                    if not release_discovery.wait(timeout=5.0):
                        raise AssertionError("test did not release blocked discovery")
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
                    try:
                        window.run_clustering()
                        self._wait_for(discovery_entered.is_set, timeout_s=5.0)
                        self.assertIsNotNone(window._preflight_thread)
                        self.assertFalse(window.footer_bar.clear_storage_button.isEnabled())
                        release_discovery.set()
                        self._wait_for(lambda: bool(captured_requests), timeout_s=5.0)
                    finally:
                        release_discovery.set()

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
