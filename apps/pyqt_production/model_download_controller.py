from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

from PyQt6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, pyqtSignal

from app.services.model_downloads import ModelDownloadItem, normalize_model_download_items
from apps.pyqt_production.worker_protocol import parse_progress_event
from apps.pyqt_production.process_transport import (
    MAX_PROTOCOL_LINE_CHARS,
    MAX_PROTOCOL_BUFFER_CHARS,
    bounded_diagnostic_append,
    cleanup_paths_async,
    read_result_payload,
    write_process_logs_async,
)
from apps.shared.runtime_support import RuntimeLayout
from src.ui.async_job import AsyncJob, defer_async_job_dispose, start_job_in_thread


LOGGER = logging.getLogger(__name__)
PROCESS_SHUTDOWN_GRACE_MS = 1500


class ModelDownloadController(QObject):
    """Runs all model acquisition in one visible, cancellable worker process."""

    started = pyqtSignal(tuple)
    progress = pyqtSignal(int, str)
    completed = pyqtSignal(dict)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()
    running_changed = pyqtSignal(bool)

    def __init__(self, runtime_layout: RuntimeLayout, parent=None) -> None:
        super().__init__(parent)
        self.runtime_layout = runtime_layout
        self.repo_root = Path(__file__).resolve().parents[2]
        # Preserve the active virtual environment instead of resolving through
        # its python symlink to an interpreter without project dependencies.
        self.worker_program = os.path.abspath(sys.executable)
        self._process: QProcess | None = None
        self._request_file: Path | None = None
        self._items: tuple[ModelDownloadItem, ...] = ()
        self._stdout_buffer = ""
        self._stdout_diagnostic = ""
        self._stderr_buffer = ""
        self._result_payload: dict[str, object] | None = None
        self._error_message = ""
        self._cancel_requested = False
        self._terminal_emitted = False
        self._preparing = False
        self._prepare_job: AsyncJob | None = None
        self._prepare_thread = None
        self._result_path: Path | None = None
        self._result_job: AsyncJob | None = None
        self._result_thread = None
        self._pending_exit: tuple[int, object] | None = None
        self._generation = 0
        self._stdout_log_path = self.runtime_layout.logs_dir / "model_worker_stdout.log"
        self._stderr_log_path = self.runtime_layout.logs_dir / "model_worker_stderr.log"
        self._pending_progress: tuple[int, str] | None = None
        self._progress_flush_scheduled = False
        self._diagnostic_future = None
        self._shutdown_target: QProcess | None = None
        self._shutdown_timer = QTimer(self)
        self._shutdown_timer.setSingleShot(True)
        self._shutdown_timer.timeout.connect(self._escalate_shutdown)

    def is_running(self) -> bool:
        return self._process is not None or self._preparing or self._result_job is not None

    @property
    def active_items(self) -> tuple[ModelDownloadItem, ...]:
        return self._items

    def start(self, items: list[ModelDownloadItem] | tuple[ModelDownloadItem, ...]) -> bool:
        if self.is_running():
            return False
        normalized = normalize_model_download_items(items)
        if not normalized:
            self.failed.emit("The model download request is empty.")
            return False

        self._items = normalized
        self._stdout_buffer = ""
        self._stdout_diagnostic = ""
        self._stderr_buffer = ""
        self._result_payload = None
        self._error_message = ""
        self._cancel_requested = False
        self._terminal_emitted = False
        self._result_path = None
        self._pending_exit = None
        self._pending_progress = None
        self._progress_flush_scheduled = False
        self._generation += 1
        payload = {
            "items": [
                {"model_name": item.model_name, "require_text": bool(item.require_text)}
                for item in normalized
            ]
        }
        self._prepare_request(payload, self._generation)
        return True

    def _prepare_request(self, payload: dict[str, object], generation: int) -> None:
        self._preparing = True

        def _run(_progress, cancel_check):
            from infra.cancel import Cancelled

            if cancel_check():
                raise Cancelled()
            from apps.pyqt_production.process_transport import prepare_protocol_line

            return prepare_protocol_line("request", payload)

        job = AsyncJob(_run)
        self._prepare_job = job

        def _cleanup() -> None:
            if self._prepare_job is job:
                self._prepare_job = None
            defer_async_job_dispose(job)

        def _completed(request_line) -> None:
            _cleanup()
            if generation != self._generation or self._cancel_requested:
                self._preparing = False
                self.running_changed.emit(False)
                self._emit_cancelled_once()
                return
            self._preparing = False
            self._start_process(bytes(request_line))

        def _failed(message: str) -> None:
            _cleanup()
            if generation != self._generation:
                return
            self._preparing = False
            self.running_changed.emit(False)
            self._emit_failed_once(f"Could not prepare model download request: {message}")

        def _cancelled() -> None:
            _cleanup()
            if generation != self._generation:
                return
            self._preparing = False
            self.running_changed.emit(False)
            self._emit_cancelled_once()

        job.completed.connect(_completed)
        job.failed.connect(_failed)
        job.cancelled.connect(_cancelled)
        thread = start_job_in_thread(job)
        self._prepare_thread = thread
        thread.finished.connect(lambda thread=thread: self._release_async_thread("_prepare_thread", thread))

    def _start_process(self, request_line: bytes) -> None:
        self._request_file = None

        process = QProcess(self)
        process.setProgram(self.worker_program)
        process.setArguments(self._worker_arguments(request_stdin=True))
        process.setWorkingDirectory(str(self.repo_root))
        process.setProcessEnvironment(self._worker_environment())
        process.readyReadStandardOutput.connect(lambda process=process: self._read_stdout(process))
        process.readyReadStandardError.connect(lambda process=process: self._read_stderr(process))
        process.finished.connect(
            lambda exit_code, exit_status, process=process: self._on_finished(process, exit_code, exit_status)
        )
        process.errorOccurred.connect(lambda error, process=process: self._on_error(process, error))
        process.started.connect(
            lambda process=process, line=bytes(request_line): self._write_one_shot_request(process, line)
        )
        self._process = process
        LOGGER.info(
            "Starting model download worker | program=%s args=%s items=%s",
            self.worker_program,
            process.arguments(),
            [item.key for item in self._items],
        )
        process.start()
        self.started.emit(tuple(item.key for item in self._items))
        self.running_changed.emit(True)

    def _write_one_shot_request(self, process: QProcess, line: bytes) -> None:
        if process is not self._process:
            return
        try:
            process.write(bytes(line))
            process.closeWriteChannel()
        except RuntimeError:
            return

    def cancel(self) -> None:
        process = self._process
        self._cancel_requested = True
        self.progress.emit(-1, "Cancelling model download; cached partial files will be reused on retry...")
        if self._prepare_job is not None:
            self._prepare_job.cancel()
        if self._result_job is not None:
            self._result_job.cancel()
        if process is None:
            return
        self._request_process_stop(process, PROCESS_SHUTDOWN_GRACE_MS)

    def shutdown(self, timeout_ms: int = 2500) -> bool:
        if self._prepare_job is not None:
            self._prepare_job.cancel()
        if self._result_job is not None:
            self._result_job.cancel()
        process = self._process
        if process is None:
            return not self.is_running()
        self._cancel_requested = True
        grace_ms = int(timeout_ms) if int(timeout_ms) > 0 else PROCESS_SHUTDOWN_GRACE_MS
        self._request_process_stop(process, grace_ms)
        try:
            return process.state() == QProcess.ProcessState.NotRunning and not self.is_running()
        except RuntimeError:
            return False

    def _request_process_stop(self, process: QProcess, grace_ms: int) -> None:
        if process is not self._process:
            return
        if self._shutdown_target is process:
            return
        self._shutdown_target = process
        try:
            process.terminate()
        except Exception:
            self._kill_shutdown_target(process)
            return
        try:
            if process.state() == QProcess.ProcessState.NotRunning:
                return
        except RuntimeError:
            return
        self._shutdown_timer.start(max(1, int(grace_ms)))

    def _escalate_shutdown(self) -> None:
        self._shutdown_timer.stop()
        process = self._shutdown_target
        if process is None or process is not self._process:
            self._shutdown_target = None
            return
        try:
            if process.state() == QProcess.ProcessState.NotRunning:
                self._shutdown_target = None
                return
        except RuntimeError:
            self._shutdown_target = None
            return
        self._kill_shutdown_target(process)

    def _kill_shutdown_target(self, process: QProcess) -> None:
        if process is not self._process or process is not self._shutdown_target:
            return
        try:
            process.kill()
        except Exception:
            self._shutdown_target = None

    def _clear_shutdown_escalation(self, process: QProcess | None) -> None:
        if process is not self._shutdown_target:
            return
        self._shutdown_timer.stop()
        self._shutdown_target = None

    def _worker_arguments(self, request_file: Path | None = None, *, request_stdin: bool = False) -> list[str]:
        if request_stdin:
            if bool(getattr(sys, "frozen", False)):
                return ["--worker", "--model-request-stdin"]
            return ["-m", "apps.pyqt_production.worker", "--model-request-stdin"]
        if request_file is None:
            raise ValueError("A model request file is required when stdin transport is disabled.")
        if bool(getattr(sys, "frozen", False)):
            return ["--worker", "--model-request-json", str(request_file)]
        return ["-m", "apps.pyqt_production.worker", "--model-request-json", str(request_file)]

    def _worker_environment(self) -> QProcessEnvironment:
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("CLUSTERLENS_RUNTIME_ROOT", str(self.runtime_layout.root))
        environment.insert("IMAGE_CLUSTERING_APP_DIR", str(self.runtime_layout.root))
        environment.insert("IMAGE_CLUSTERING_MODEL_ASSETS_DIR", str(self.runtime_layout.model_assets_dir))
        environment.insert("PYTHONUNBUFFERED", "1")
        for key, value in os.environ.items():
            if key.startswith("QT_") or key in {"PATH", "PYTHONPATH"}:
                environment.insert(key, value)
        return environment

    def _read_stdout(self, process: QProcess) -> None:
        try:
            text = bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace")
        except RuntimeError:
            return
        if process is not self._process:
            return
        self._stdout_diagnostic = bounded_diagnostic_append(self._stdout_diagnostic, text)
        self._stdout_buffer += text
        if len(self._stdout_buffer) > MAX_PROTOCOL_BUFFER_CHARS:
            self._error_message = (
                f"Model worker protocol buffer exceeded the {MAX_PROTOCOL_BUFFER_CHARS}-character safety limit."
            )
            self._stdout_buffer = ""
            try:
                process.kill()
            except RuntimeError:
                pass
            return
        if len(self._stdout_buffer) > MAX_PROTOCOL_LINE_CHARS and "\n" not in self._stdout_buffer:
            self._error_message = (
                f"Model worker protocol record exceeded the {MAX_PROTOCOL_LINE_CHARS}-character safety limit."
            )
            self._stdout_buffer = ""
            try:
                process.kill()
            except RuntimeError:
                pass
            return
        while "\n" in self._stdout_buffer:
            line, self._stdout_buffer = self._stdout_buffer.split("\n", 1)
            if len(line) > MAX_PROTOCOL_LINE_CHARS:
                self._error_message = (
                    f"Model worker protocol record exceeded the {MAX_PROTOCOL_LINE_CHARS}-character safety limit."
                )
                try:
                    process.kill()
                except RuntimeError:
                    pass
                return
            self._handle_stdout_line(line)

    def _handle_stdout_line(self, line: str) -> None:
        line = str(line or "").strip()
        if not line:
            return
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            LOGGER.warning("Ignoring non-JSON model worker output: %.512s", line)
            return
        event_type = str(event.get("type") or "")
        body = dict(event.get("payload") or {})
        if event_type == "progress":
            value, status = parse_progress_event(body)
            self._publish_progress(value, status)
        elif event_type == "result":
            self._result_payload = body
        elif event_type == "result_ref":
            raw_path = str(body.get("path") or "").strip()
            if raw_path:
                self._result_path = Path(raw_path)
            else:
                self._error_message = "Model worker returned an empty result reference."
        elif event_type == "error":
            self._error_message = str(body.get("message") or "Model download worker failed.")
            if body.get("traceback"):
                LOGGER.error("Model download worker traceback (truncated):\n%.4096s", body.get("traceback"))

    def _read_stderr(self, process: QProcess) -> None:
        try:
            text = bytes(process.readAllStandardError()).decode("utf-8", errors="replace")
        except RuntimeError:
            return
        if process is not self._process:
            return
        self._stderr_buffer = bounded_diagnostic_append(self._stderr_buffer, text)

    def _publish_progress(self, value: int, status: str) -> None:
        if self._terminal_emitted:
            return
        if self._process is None:
            self.progress.emit(int(value), str(status))
            return
        self._pending_progress = (int(value), str(status))
        if self._progress_flush_scheduled:
            return
        self._progress_flush_scheduled = True
        QTimer.singleShot(33, self._flush_progress)

    def _flush_progress(self) -> None:
        self._progress_flush_scheduled = False
        pending = self._pending_progress
        self._pending_progress = None
        if pending is not None and not self._terminal_emitted and self._qobject_alive():
            self.progress.emit(*pending)

    def _on_error(self, process: QProcess, error) -> None:  # noqa: ANN001
        if not self._qobject_alive():
            return
        if process is not self._process:
            return
        self._error_message = self._error_message or (
            f"Model download process error: {error} | program={self.worker_program} | cwd={self.repo_root}"
        )
        self._stderr_buffer = bounded_diagnostic_append(self._stderr_buffer, self._error_message + "\n")
        LOGGER.error(self._error_message)
        if error == QProcess.ProcessError.FailedToStart:
            # Qt does not guarantee a finished signal after FailedToStart.
            # Finalize on the event loop so Jobs never remains permanently
            # stuck in a running state.
            QTimer.singleShot(0, lambda process=process: self._finalize_failed_start(process))

    def _finalize_failed_start(self, process: QProcess) -> None:
        if process is not self._process:
            return
        self._on_finished(process, -1, QProcess.ExitStatus.CrashExit)

    def _on_finished(self, process: QProcess, exit_code: int, exit_status) -> None:  # noqa: ANN001
        if not self._qobject_alive():
            return
        self._clear_shutdown_escalation(process)
        if process is not self._process:
            try:
                process.deleteLater()
            except RuntimeError:
                pass
            return
        self._read_stdout(process)
        self._read_stderr(process)
        if self._stdout_buffer.strip():
            self._handle_stdout_line(self._stdout_buffer)
        self._stdout_buffer = ""
        self._process = None
        self._flush_progress()
        self._persist_diagnostics()
        self._remove_request_file()
        try:
            process.deleteLater()
        except RuntimeError:
            pass
        if self._terminal_emitted:
            return
        if self._cancel_requested:
            self._generation += 1
            if self._result_job is not None:
                self._result_job.cancel()
            cleanup_paths_async(self._result_path)
            self._result_path = None
            self.running_changed.emit(False)
            self._emit_cancelled_once()
        elif self._result_path is not None and int(exit_code) == 0:
            self._pending_exit = (int(exit_code), exit_status)
            self._start_result_parse()
        elif self._result_payload is not None and int(exit_code) == 0:
            self.running_changed.emit(False)
            self._emit_completed_once(dict(self._result_payload))
        else:
            message = self._error_message or self._stderr_buffer.strip()
            if not message:
                message = f"Model download worker exited with code {exit_code} ({exit_status})."
            self.running_changed.emit(False)
            self._emit_failed_once(message)
        self._items = ()

    def _start_result_parse(self) -> None:
        path = self._result_path
        if path is None or self._result_job is not None:
            return
        self._result_path = None
        generation = self._generation
        allowed_directory = self.runtime_layout.cache_dir / "tmp"

        def _run(_progress, cancel_check):
            from infra.cancel import Cancelled

            if cancel_check():
                raise Cancelled()
            try:
                payload = read_result_payload(path, allowed_directory=allowed_directory)
            finally:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    LOGGER.warning("Could not remove model result file: %s", path, exc_info=True)
            if cancel_check():
                raise Cancelled()
            return payload

        job = AsyncJob(_run)
        self._result_job = job

        def _cleanup() -> None:
            if self._result_job is job:
                self._result_job = None
            defer_async_job_dispose(job)

        def _completed(payload) -> None:
            _cleanup()
            if generation != self._generation or self._cancel_requested:
                self.running_changed.emit(False)
                self._emit_cancelled_once()
                return
            self._pending_exit = None
            self._result_payload = dict(payload)
            self.running_changed.emit(False)
            self._emit_completed_once(dict(self._result_payload))

        def _failed(message: str) -> None:
            _cleanup()
            if generation != self._generation:
                return
            self._pending_exit = None
            self.running_changed.emit(False)
            self._emit_failed_once(f"Could not read model worker result: {message}")

        def _cancelled() -> None:
            _cleanup()
            if generation == self._generation:
                self._pending_exit = None
                self.running_changed.emit(False)
                self._emit_cancelled_once()

        job.completed.connect(_completed)
        job.failed.connect(_failed)
        job.cancelled.connect(_cancelled)
        thread = start_job_in_thread(job)
        self._result_thread = thread
        thread.finished.connect(lambda thread=thread: self._release_async_thread("_result_thread", thread))

    def _emit_failed_once(self, message: str) -> None:
        if self._terminal_emitted:
            return
        self._flush_progress()
        self._terminal_emitted = True
        self.failed.emit(str(message))

    def _emit_cancelled_once(self) -> None:
        if self._terminal_emitted:
            return
        self._flush_progress()
        self._terminal_emitted = True
        self.cancelled.emit()

    def _emit_completed_once(self, payload: dict[str, object]) -> None:
        if self._terminal_emitted:
            return
        self._flush_progress()
        self._terminal_emitted = True
        self.completed.emit(dict(payload))

    def _persist_diagnostics(self) -> None:
        self._diagnostic_future = write_process_logs_async(
            self._stdout_log_path,
            self._stdout_diagnostic,
            self._stderr_log_path,
            self._stderr_buffer,
        )

    def _qobject_alive(self) -> bool:
        try:
            self.thread()
        except RuntimeError:
            return False
        return True

    def _release_async_thread(self, attribute: str, thread) -> None:
        if getattr(self, attribute, None) is thread:
            setattr(self, attribute, None)
        try:
            if getattr(thread, "_job", None) is not None:
                thread._job = None
        except (AttributeError, RuntimeError):
            pass

    def _remove_request_file(self) -> None:
        request_file = self._request_file
        self._request_file = None
        cleanup_paths_async(request_file)
