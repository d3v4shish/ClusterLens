import json
import os
import sys
import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PyQt6.QtCore import QProcess, QTimer
from PyQt6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from apps.pyqt_production.process_transport import (
    MAX_DIAGNOSTIC_CHARS,
    MAX_PROTOCOL_LINE_CHARS,
    TRUNCATION_MARKER,
    bounded_diagnostic_append,
    read_result_payload,
    write_process_logs_async,
    write_result_payload,
)
from apps.pyqt_production.session_controller import ClusteringSessionController
from apps.pyqt_production.model_download_controller import ModelDownloadController
from apps.pyqt_production.worker_protocol import ProductionClusterRequest
from apps.shared.runtime_support import activate_runtime_root
from app.services.model_downloads import ModelDownloadItem


APP = QApplication.instance() or QApplication([])


def _wait_for(predicate, timeout_s=3.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        APP.processEvents()
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition did not become true")


def _request(directory: str) -> ProductionClusterRequest:
    return ProductionClusterRequest(
        directory=directory,
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


def test_diagnostic_capture_retains_bounded_tail_with_visible_marker():
    captured = bounded_diagnostic_append("old\n", "x" * (MAX_DIAGNOSTIC_CHARS * 2))

    assert len(captured) == MAX_DIAGNOSTIC_CHARS
    assert captured.startswith(TRUNCATION_MARKER)
    assert captured.endswith("x" * 128)


def test_result_handoff_validates_scope_and_removes_no_source_data():
    with TemporaryDirectory() as tmp:
        allowed = Path(tmp) / "managed"
        outside = Path(tmp) / "outside"
        result_path = write_result_payload(allowed, "result", {"rows": [1, 2, 3]})
        outside_path = write_result_payload(outside, "result", {"rows": []})

        assert read_result_payload(result_path, allowed_directory=allowed) == {"rows": [1, 2, 3]}
        try:
            read_result_payload(outside_path, allowed_directory=allowed)
        except ValueError as exc:
            assert "outside" in str(exc)
        else:
            raise AssertionError("outside result reference was accepted")


def test_session_protocol_handles_partial_malformed_and_oversized_records():
    class FakeProcess:
        killed = False

        def kill(self):
            self.killed = True

    with TemporaryDirectory() as tmp:
        with patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
            controller = ClusteringSessionController(activate_runtime_root("TransportProtocolTest"))
            seen = []
            controller.progress.connect(lambda value, status: seen.append((value, status)))
            controller._stdout_buffer = '{"type":"progress","payload":{"value":4'
            controller._drain_stdout_lines()
            assert seen == []

            controller._stdout_buffer += ',"status":"Working"}}\nnot-json\n'
            controller._drain_stdout_lines()
            assert seen == [(4, "Working")]

            process = FakeProcess()
            controller._process = process
            controller._stdout_buffer = ("x" * (MAX_PROTOCOL_LINE_CHARS + 1)) + "\n"
            controller._drain_stdout_lines()
            assert process.killed
            assert "safety limit" in controller._error_message


def test_rapid_progress_is_coalesced_but_latest_value_is_delivered():
    with TemporaryDirectory() as tmp:
        with patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
            controller = ClusteringSessionController(activate_runtime_root("TransportProgressTest"))
            controller._process = object()
            seen = []
            controller.progress.connect(lambda value, status: seen.append((value, status)))

            for value in range(100):
                controller._publish_progress(value, f"step {value}")

            assert seen == []
            _wait_for(lambda: bool(seen))
            assert seen == [(99, "step 99")]


def test_request_serialization_and_result_parsing_run_off_qt_thread():
    with TemporaryDirectory() as tmp:
        with patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
            layout = activate_runtime_root("TransportThreadAffinityTest")
            controller = ClusteringSessionController(layout)
            controller.worker_program = str(Path(tmp) / "missing-worker")
            caller_thread = threading.get_ident()
            serialization_threads = []
            parse_threads = []
            failures = []
            completed = []
            original_serialize = __import__(
                "apps.pyqt_production.session_controller", fromlist=["prepare_protocol_line"]
            ).prepare_protocol_line
            original_read = __import__(
                "apps.pyqt_production.session_controller", fromlist=["read_result_payload"]
            ).read_result_payload

            def record_serialize(*args, **kwargs):
                serialization_threads.append(threading.get_ident())
                return original_serialize(*args, **kwargs)

            controller.failed.connect(failures.append)
            with patch("apps.pyqt_production.session_controller.prepare_protocol_line", side_effect=record_serialize):
                assert controller.start(_request(tmp))
                _wait_for(lambda: bool(failures))
            assert serialization_threads and serialization_threads[0] != caller_thread

            result_path = write_result_payload(layout.cache_dir / "tmp", "cluster_result", {"ok": True})
            controller._cancel_requested = False
            controller._generation += 1
            controller._terminal_emitted = False
            controller._running_request = True
            controller._pending_exit = (None, 0, QProcess.ExitStatus.NormalExit)
            controller._result_path = result_path
            controller.completed.connect(completed.append)

            def record_read(*args, **kwargs):
                parse_threads.append(threading.get_ident())
                return original_read(*args, **kwargs)

            with patch("apps.pyqt_production.session_controller.read_result_payload", side_effect=record_read):
                controller._start_result_parse()
                _wait_for(lambda: bool(completed))
            assert completed == [{"ok": True}]
            assert parse_threads and parse_threads[0] != caller_thread
            assert not result_path.exists()


def test_model_request_serialization_runs_off_qt_thread():
    with TemporaryDirectory() as tmp:
        with patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
            controller = ModelDownloadController(activate_runtime_root("ModelTransportThreadTest"))
            controller.worker_program = str(Path(tmp) / "missing-model-worker")
            caller_thread = threading.get_ident()
            serialization_threads = []
            failures = []
            from apps.pyqt_production import process_transport

            original = process_transport.prepare_protocol_line

            def record(*args, **kwargs):
                serialization_threads.append(threading.get_ident())
                return original(*args, **kwargs)

            controller.failed.connect(failures.append)
            with patch("apps.pyqt_production.process_transport.prepare_protocol_line", side_effect=record):
                assert controller.start([ModelDownloadItem("dino")])
                _wait_for(lambda: bool(failures))
            assert serialization_threads and serialization_threads[0] != caller_thread
            if controller._diagnostic_future is not None:
                controller._diagnostic_future.result(timeout=3.0)


def test_stale_result_parse_is_not_published():
    with TemporaryDirectory() as tmp:
        with patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
            layout = activate_runtime_root("StaleTransportResultTest")
            controller = ClusteringSessionController(layout)
            result_path = write_result_payload(layout.cache_dir / "tmp", "cluster_result", {"stale": True})
            entered = threading.Event()
            release = threading.Event()
            completed = []
            original_read = read_result_payload

            def blocked_read(*args, **kwargs):
                entered.set()
                assert release.wait(2.0)
                return original_read(*args, **kwargs)

            controller.completed.connect(completed.append)
            controller._running_request = True
            controller._pending_exit = (None, 0, QProcess.ExitStatus.NormalExit)
            controller._result_path = result_path
            with patch("apps.pyqt_production.session_controller.read_result_payload", side_effect=blocked_read):
                controller._start_result_parse()
                assert entered.wait(2.0)
                controller._generation += 1
                release.set()
                _wait_for(lambda: controller._result_job is None)
            assert completed == []


def test_large_result_preparation_keeps_qt_event_loop_responsive():
    with TemporaryDirectory() as tmp:
        with patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
            layout = activate_runtime_root("LargeTransportResultTest")
            controller = ClusteringSessionController(layout)
            membership = {
                f"/photos/{index:05d}.jpg": {
                    "fast_preview::semantic::cosine-kmeans": {
                        "cluster_id": index % 20,
                        "rank": index,
                    }
                }
                for index in range(5000)
            }
            result_path = write_result_payload(
                layout.cache_dir / "tmp",
                "cluster_result",
                {
                    "clusters_by_key": {
                        "fast_preview::semantic::cosine-kmeans": {
                            str(cluster_id): [
                                path
                                for offset, path in enumerate(membership)
                                if offset % 20 == cluster_id
                            ]
                            for cluster_id in range(20)
                        }
                    },
                    "membership_by_image": membership,
                },
            )
            entered = threading.Event()
            release = threading.Event()
            qt_acknowledged = []
            completed = []

            def transform(payload):
                entered.set()
                assert release.wait(2.0)
                return {
                    "_prepared": {
                        "membership_by_image": {
                            str(path): {str(key): dict(value) for key, value in dict(entries).items()}
                            for path, entries in dict(payload["membership_by_image"]).items()
                        }
                    }
                }

            controller.result_transform = transform
            controller._running_request = True
            controller._pending_exit = (None, 0, QProcess.ExitStatus.NormalExit)
            controller._result_path = result_path
            controller.completed.connect(completed.append)
            controller._start_result_parse()
            assert entered.wait(2.0)
            QTimer.singleShot(0, lambda: qt_acknowledged.append(True))
            _wait_for(lambda: bool(qt_acknowledged))
            release.set()
            _wait_for(lambda: bool(completed))

            prepared = completed[0]["_prepared"]
            assert len(prepared["membership_by_image"]) == 5000
            assert not result_path.exists()


def test_session_shutdown_terminates_then_escalates_without_waiting_on_qprocess():
    class FakeProcess:
        terminated = False
        killed = False

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

        def state(self):
            return QProcess.ProcessState.Running

        def waitForFinished(self, _timeout):
            raise AssertionError("shutdown blocked on QProcess")

    with TemporaryDirectory() as tmp:
        with patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
            controller = ClusteringSessionController(activate_runtime_root("TransportShutdownTest"))
            process = FakeProcess()
            controller._process = process

            assert not controller.shutdown(2500)
            assert process.terminated
            assert not process.killed

            controller._escalate_shutdown()
            assert process.killed


def test_session_shutdown_escalation_cannot_kill_a_replacement_process():
    class FakeProcess:
        def __init__(self):
            self.terminated = False
            self.killed = False

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

        def state(self):
            return QProcess.ProcessState.Running

    with TemporaryDirectory() as tmp:
        with patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
            controller = ClusteringSessionController(activate_runtime_root("TransportShutdownIdentityTest"))
            original = FakeProcess()
            replacement = FakeProcess()
            controller._process = original

            assert not controller.shutdown(2500)
            controller._process = replacement
            controller._escalate_shutdown()

            assert original.terminated
            assert not original.killed
            assert not replacement.killed
            assert controller._shutdown_target is None


def test_model_download_shutdown_terminates_then_escalates_only_its_owned_process():
    class FakeProcess:
        def __init__(self):
            self.terminated = False
            self.killed = False

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

        def state(self):
            return QProcess.ProcessState.Running

        def waitForFinished(self, _timeout):
            raise AssertionError("shutdown blocked on QProcess")

    with TemporaryDirectory() as tmp:
        with patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
            controller = ModelDownloadController(activate_runtime_root("ModelShutdownIdentityTest"))
            original = FakeProcess()
            replacement = FakeProcess()
            controller._process = original

            assert not controller.shutdown(2500)
            assert original.terminated
            assert not original.killed

            controller._process = replacement
            controller._escalate_shutdown()
            assert not original.killed
            assert not replacement.killed

            controller._process = original
            controller._shutdown_target = original
            controller._escalate_shutdown()
            assert original.killed


def test_real_stuck_owned_subprocess_is_killed_and_session_controller_restarts():
    with TemporaryDirectory() as tmp:
        with patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
            controller = ClusteringSessionController(activate_runtime_root("RealShutdownEscalationTest"))
            marker = Path(tmp) / "child-ready"
            process = QProcess(controller)
            process.setProperty("daemon_process", False)
            process.setProgram(sys.executable)
            process.setArguments(
                [
                    "-c",
                    (
                        "import pathlib, signal, sys, time; "
                        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                        "pathlib.Path(sys.argv[1]).write_text('ready', encoding='utf-8'); "
                        "time.sleep(30)"
                    ),
                    str(marker),
                ]
            )
            process.finished.connect(
                lambda exit_code, exit_status, process=process: controller._on_finished(
                    process, exit_code, exit_status
                )
            )
            controller._process = process
            controller._running_request = True
            process.start()
            _wait_for(marker.exists)

            assert not controller.shutdown(25)
            _wait_for(lambda: controller._process is None)
            assert process.state() == QProcess.ProcessState.NotRunning
            assert not controller.is_running()

            failures = []
            controller.failed.connect(failures.append)
            controller.worker_program = str(Path(tmp) / "missing-restart-worker")
            assert controller.start(_request(tmp))
            _wait_for(lambda: bool(failures))
            assert not controller.is_running()
            if controller._diagnostic_future is not None:
                controller._diagnostic_future.result(timeout=3.0)


def test_diagnostic_log_write_runs_off_qt_and_does_not_recreate_removed_runtime():
    with TemporaryDirectory() as tmp:
        logs = Path(tmp) / "logs"
        logs.mkdir()
        target = logs / "worker.log"
        caller_thread = threading.get_ident()
        writer_threads = []
        written = threading.Event()
        original_write = Path.write_text

        def record_write(path, *args, **kwargs):
            writer_threads.append(threading.get_ident())
            try:
                return original_write(path, *args, **kwargs)
            finally:
                written.set()

        with patch.object(Path, "write_text", record_write):
            write_process_logs_async(target, "bounded", None, "")
            assert written.wait(2.0)
        assert writer_threads[0] != caller_thread
        assert target.read_text(encoding="utf-8") == "bounded"

        target.unlink()
        logs.rmdir()
        future = write_process_logs_async(target, "late", None, "")
        future.result(timeout=2.0)
        assert not logs.exists()
