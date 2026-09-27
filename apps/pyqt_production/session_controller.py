from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path

from PyQt6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, pyqtSignal

from apps.pyqt_production.worker_protocol import ProductionClusterRequest, parse_progress_event
from apps.pyqt_production.process_transport import (
    MAX_PROTOCOL_LINE_CHARS,
    MAX_PROTOCOL_BUFFER_CHARS,
    bounded_diagnostic_append,
    cleanup_paths_async,
    prepare_protocol_line,
    read_result_payload,
    write_process_logs_async,
)
from apps.shared.runtime_support import RuntimeLayout
from src.ui.async_job import AsyncJob, defer_async_job_dispose, start_job_in_thread


LOGGER = logging.getLogger(__name__)
PROCESS_SHUTDOWN_GRACE_MS = 1500


class ClusteringSessionController(QObject):
    started = pyqtSignal()
    progress = pyqtSignal(int, str)
    completed = pyqtSignal(dict)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()
    running_changed = pyqtSignal(bool)

    def __init__(self, runtime_layout: RuntimeLayout, parent=None) -> None:
        super().__init__(parent)
        self.runtime_layout = runtime_layout
        self.repo_root = Path(__file__).resolve().parents[2]
        # Keep the virtual-environment entry point. Resolving its symlink can
        # select the base interpreter, which does not have project packages.
        self.worker_program = os.path.abspath(sys.executable)
        self._process: QProcess | None = None
        self._request: ProductionClusterRequest | None = None
        self._stdout_buffer = ""
        self._stdout_diagnostic = ""
        self._stderr_buffer = ""
        self._stderr_line_buffer = ""
        self._result_payload: dict[str, object] | None = None
        self._error_message = ""
        self._cancel_requested = False
        self._request_file: Path | None = None
        self._stdout_log_path: Path | None = None
        self._stderr_log_path: Path | None = None
        self._keep_worker_warm = False
        self._running_request = False
        self._daemon_process = False
        self._pending_daemon_request: ProductionClusterRequest | None = None
        self._pending_daemon_line: bytes | None = None
        self._stopping_idle_daemon = False
        self._preparing = False
        self._prepare_job: AsyncJob | None = None
        self._prepare_thread = None
        self._result_path: Path | None = None
        self._result_job: AsyncJob | None = None
        self._result_thread = None
        self._pending_exit: tuple[QProcess | None, int, object] | None = None
        self._generation = 0
        self._terminal_emitted = False
        self._pending_progress: tuple[int, str] | None = None
        self._progress_flush_scheduled = False
        self._diagnostic_future = None
        self._shutdown_target: QProcess | None = None
        self._shutdown_timer = QTimer(self)
        self._shutdown_timer.setSingleShot(True)
        self._shutdown_timer.timeout.connect(self._escalate_shutdown)
        self.result_transform: Callable[[dict[str, object]], dict[str, object]] | None = None

    def is_running(self) -> bool:
        return (
            self._running_request
            or self._preparing
            or self._result_job is not None
            or self._pending_daemon_request is not None
            or self._stopping_idle_daemon
        )

    def is_worker_warm(self) -> bool:
        if not self._keep_worker_warm:
            return False
        return self._worker_process_is_idle()

    def _worker_process_is_idle(self) -> bool:
        try:
            return (
                self._process is not None
                and self._process.state() != QProcess.ProcessState.NotRunning
                and not self._running_request
                and self._pending_daemon_request is None
                and not self._stopping_idle_daemon
            )
        except RuntimeError:
            return False

    def set_keep_worker_warm(self, enabled: bool) -> None:
        was_idle = self._worker_process_is_idle()
        self._keep_worker_warm = bool(enabled)
        if not self._keep_worker_warm and was_idle:
            self._stop_idle_daemon()

    def start(self, request: ProductionClusterRequest) -> bool:
        if self.is_running():
            return False
        self._request = request
        self._cancel_requested = False
        self._terminal_emitted = False
        self._generation += 1
        self._reset_run_buffers()
        self._prepare_worker_logs(truncate=True)
        self._prepare_request(request, self._generation)
        return True

    def cancel(self) -> None:
        self._cancel_requested = True
        if self._prepare_job is not None:
            self._prepare_job.cancel()
        if self._result_job is not None:
            self._result_job.cancel()
        process = self._process
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
            # Test doubles and already-deleted Qt wrappers cannot always offer
            # the cooperative stop. A hard stop remains non-blocking.
            self._kill_shutdown_target(process)
            return
        try:
            if process.state() == QProcess.ProcessState.NotRunning:
                return
        except RuntimeError:
            return
        self._shutdown_timer.start(max(1, int(grace_ms)))

    def _schedule_shutdown_escalation(self, process: QProcess, grace_ms: int) -> None:
        if process is not self._process or self._shutdown_target is process:
            return
        self._shutdown_target = process
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

    def _stop_idle_daemon(self) -> None:
        process = self._process
        if process is None or self._running_request or self._pending_daemon_request is not None:
            return
        self._stopping_idle_daemon = True
        try:
            process.write(json.dumps({"type": "shutdown", "payload": {}}).encode("utf-8") + b"\n")
            process.closeWriteChannel()
            self._schedule_shutdown_escalation(process, PROCESS_SHUTDOWN_GRACE_MS)
        except Exception:
            self._shutdown_target = process
            self._kill_shutdown_target(process)

    def _reset_run_buffers(self) -> None:
        self._result_payload = None
        self._result_path = None
        self._pending_exit = None
        self._error_message = ""
        self._stdout_buffer = ""
        self._stdout_diagnostic = ""
        self._stderr_buffer = ""
        self._stderr_line_buffer = ""
        self._pending_progress = None
        self._progress_flush_scheduled = False

    def _prepare_worker_logs(self, *, truncate: bool) -> None:
        self._stdout_log_path = self.runtime_layout.logs_dir / "worker_stdout.log"
        self._stderr_log_path = self.runtime_layout.logs_dir / "worker_stderr.log"

    def _prepare_request(self, request: ProductionClusterRequest, generation: int) -> None:
        self._preparing = True
        keep_warm = bool(self._keep_worker_warm)
        payload = request.as_dict()

        def _run(_progress, cancel_check):
            if cancel_check():
                from infra.cancel import Cancelled

                raise Cancelled()
            if keep_warm:
                return ("daemon", prepare_protocol_line("request", payload))
            return ("inline", prepare_protocol_line("request", payload))

        job = AsyncJob(_run)
        self._prepare_job = job

        def _cleanup() -> None:
            if self._prepare_job is job:
                self._prepare_job = None
            defer_async_job_dispose(job)

        def _completed(prepared) -> None:
            _cleanup()
            if generation != self._generation or self._cancel_requested:
                self._preparing = False
                self.running_changed.emit(False)
                self._emit_cancelled_once()
                return
            self._preparing = False
            mode, value = prepared
            if mode == "daemon":
                self._start_or_send_daemon(request, bytes(value))
            else:
                self._start_process(request, bytes(value))

        def _failed(message: str) -> None:
            _cleanup()
            if generation != self._generation:
                return
            self._preparing = False
            self.running_changed.emit(False)
            self._emit_failed_once(f"Could not prepare worker request: {message}")

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

    def _worker_arguments(
        self,
        *,
        request_json: Path | None = None,
        request_stdin: bool = False,
        daemon: bool = False,
    ) -> list[str]:
        if bool(getattr(sys, "frozen", False)):
            args = ["--worker"]
        else:
            args = ["-m", "apps.pyqt_production.worker"]
        if daemon:
            args.append("--daemon")
        elif request_stdin:
            args.append("--request-stdin")
        elif request_json is not None:
            args.extend(["--request-json", str(request_json)])
        return args

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

    def _start_process(self, request: ProductionClusterRequest, request_line: bytes) -> None:
        self._stopping_idle_daemon = False
        self._running_request = True
        self._daemon_process = False
        self._request_file = None

        process = QProcess(self)
        process.setProperty("daemon_process", False)
        process.setProgram(self.worker_program)
        process.setArguments(self._worker_arguments(request_stdin=True))
        process.setWorkingDirectory(str(self.repo_root))
        process.setProcessEnvironment(self._worker_environment())
        process.readyReadStandardOutput.connect(lambda process=process: self._read_stdout(process))
        process.readyReadStandardError.connect(lambda process=process: self._read_stderr(process))
        process.finished.connect(
            lambda exit_code, exit_status, process=process: self._on_finished(process, exit_code, exit_status)
        )
        process.errorOccurred.connect(lambda error, process=process: self._on_error(error, process))
        process.started.connect(
            lambda process=process, line=bytes(request_line): self._write_one_shot_request(process, line)
        )
        self._process = process
        LOGGER.info(
            "Starting production worker with program=%s args=%s cwd=%s runtime=%s request=%s",
            self.worker_program,
            process.arguments(),
            self.repo_root,
            self.runtime_layout.root,
            "stdin",
        )
        process.start()
        self.started.emit()
        self.running_changed.emit(True)

    def _write_one_shot_request(self, process: QProcess, line: bytes) -> None:
        if process is not self._process:
            return
        try:
            process.write(bytes(line))
            process.closeWriteChannel()
        except RuntimeError:
            return

    def _start_or_send_daemon(self, request: ProductionClusterRequest, line: bytes) -> None:
        self._stopping_idle_daemon = False
        self._request_file = None
        process = self._process
        if process is not None and process.state() != QProcess.ProcessState.NotRunning and self._daemon_process:
            self._send_daemon_request(request, line)
            return
        self._daemon_process = True
        self._pending_daemon_request = request
        self._pending_daemon_line = bytes(line)
        process = QProcess(self)
        process.setProperty("daemon_process", True)
        process.setProgram(self.worker_program)
        process.setArguments(self._worker_arguments(daemon=True))
        process.setWorkingDirectory(str(self.repo_root))
        process.setProcessEnvironment(self._worker_environment())
        process.readyReadStandardOutput.connect(lambda process=process: self._read_stdout(process))
        process.readyReadStandardError.connect(lambda process=process: self._read_stderr(process))
        process.finished.connect(
            lambda exit_code, exit_status, process=process: self._on_finished(process, exit_code, exit_status)
        )
        process.errorOccurred.connect(lambda error, process=process: self._on_error(error, process))
        process.started.connect(self._send_pending_daemon_request)
        self._process = process
        LOGGER.info(
            "Starting persistent production worker with program=%s args=%s cwd=%s runtime=%s",
            self.worker_program,
            process.arguments(),
            self.repo_root,
            self.runtime_layout.root,
        )
        process.start()

    def _send_pending_daemon_request(self) -> None:
        request = self._pending_daemon_request
        line = self._pending_daemon_line
        self._pending_daemon_request = None
        self._pending_daemon_line = None
        if request is not None and line is not None:
            self._send_daemon_request(request, line)

    def _send_daemon_request(self, request: ProductionClusterRequest, line: bytes) -> None:
        if self._process is None:
            self._emit_failed_once("Persistent worker is not available.")
            return
        self._request = request
        self._running_request = True
        self._process.write(bytes(line))
        LOGGER.info(
            "Sent request to persistent production worker | runtime=%s models=%s backends=%s profile=%s",
            self.runtime_layout.root,
            ",".join(request.embedding_models),
            ",".join(request.clustering_backends),
            request.performance_profile,
        )
        self.started.emit()
        self.running_changed.emit(True)

    def _read_stdout(self, process: QProcess | None = None) -> None:
        process = process or self._process
        if process is None or process is not self._process:
            return
        try:
            text = bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace")
        except RuntimeError:
            return
        self._stdout_diagnostic = bounded_diagnostic_append(self._stdout_diagnostic, text)
        self._stdout_buffer += text
        if len(self._stdout_buffer) > MAX_PROTOCOL_BUFFER_CHARS:
            self._fail_oversized_protocol_record(process)
            return
        if len(self._stdout_buffer) > MAX_PROTOCOL_LINE_CHARS and "\n" not in self._stdout_buffer:
            self._fail_oversized_protocol_record(process)
            return
        self._drain_stdout_lines()

    def _drain_stdout_lines(self) -> None:
        while "\n" in self._stdout_buffer:
            line, self._stdout_buffer = self._stdout_buffer.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            if len(line) > MAX_PROTOCOL_LINE_CHARS:
                process = self._process
                if process is not None:
                    self._fail_oversized_protocol_record(process)
                return
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                LOGGER.warning("Ignoring non-JSON worker line: %.512s", line)
                continue
            event_type = str(payload.get("type", ""))
            body = dict(payload.get("payload") or {})
            if event_type == "progress":
                value, status = parse_progress_event(body)
                self._publish_progress(value, status)
            elif event_type == "result":
                self._result_payload = body
                if self._daemon_process:
                    self._complete_daemon_result()
            elif event_type == "result_ref":
                self._accept_result_reference(body)
            elif event_type == "error":
                self._error_message = str(body.get("message", "Worker failed"))
                if body.get("traceback"):
                    LOGGER.error("Worker traceback (truncated):\n%.4096s", body.get("traceback"))
                if self._daemon_process:
                    self._handle_daemon_failure()
            elif event_type == "ready":
                LOGGER.info("Persistent production worker ready: %s", body.get("status", "ready"))

    def _read_stderr(self, process: QProcess | None = None) -> None:
        process = process or self._process
        if process is None or process is not self._process:
            return
        try:
            text = bytes(process.readAllStandardError()).decode("utf-8", errors="replace")
        except RuntimeError:
            return
        self._stderr_buffer = bounded_diagnostic_append(self._stderr_buffer, text)
        self._stderr_line_buffer = bounded_diagnostic_append(self._stderr_line_buffer, text)

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

    def _fail_oversized_protocol_record(self, process: QProcess) -> None:
        self._error_message = (
            f"Worker protocol record exceeded the {MAX_PROTOCOL_LINE_CHARS}-character safety limit."
        )
        self._stdout_buffer = ""
        try:
            process.kill()
        except RuntimeError:
            pass

    def _accept_result_reference(self, body: dict[str, object]) -> None:
        raw_path = str(body.get("path") or "").strip()
        if not raw_path:
            self._error_message = "Worker returned an empty result reference."
            return
        self._result_path = Path(raw_path)
        if self._daemon_process:
            self._start_result_parse()

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
                transform = self.result_transform
                if transform is not None:
                    payload = transform(payload)
            finally:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    LOGGER.warning("Could not remove worker result file: %s", path, exc_info=True)
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
                self._finish_result_cancelled()
                return
            self._result_payload = dict(payload)
            if self._daemon_process:
                self._complete_daemon_result()
            elif self._pending_exit is not None:
                self._pending_exit = None
                self._running_request = False
                self.running_changed.emit(False)
                self._emit_completed_payload()

        def _failed(message: str) -> None:
            _cleanup()
            if generation != self._generation:
                return
            self._error_message = f"Could not read worker result: {message}"
            if self._daemon_process:
                self._handle_daemon_failure()
            elif self._pending_exit is not None:
                self._pending_exit = None
                self._running_request = False
                self.running_changed.emit(False)
                self._emit_failed_once(self._error_message)

        def _cancelled() -> None:
            _cleanup()
            if generation == self._generation:
                self._finish_result_cancelled()

        job.completed.connect(_completed)
        job.failed.connect(_failed)
        job.cancelled.connect(_cancelled)
        thread = start_job_in_thread(job)
        self._result_thread = thread
        thread.finished.connect(lambda thread=thread: self._release_async_thread("_result_thread", thread))

    def _finish_result_cancelled(self) -> None:
        if not self._running_request and self._pending_exit is None:
            return
        self._pending_exit = None
        self._running_request = False
        self.running_changed.emit(False)
        self._emit_cancelled_once()

    def _drain_stderr_lines(self, *, flush: bool = False) -> None:
        while "\n" in self._stderr_line_buffer:
            line, self._stderr_line_buffer = self._stderr_line_buffer.split("\n", 1)
            self._log_stderr_line(line)
        if flush and self._stderr_line_buffer.strip():
            line = self._stderr_line_buffer
            self._stderr_line_buffer = ""
            self._log_stderr_line(line)

    def _log_stderr_line(self, line: str) -> None:
        level, message = self._classify_stderr_line(line)
        if not message:
            return
        LOGGER.log(level, "Worker stderr: %s", message)
        if level >= logging.ERROR and not self._error_message and not message.lower().startswith("traceback"):
            self._error_message = message

    @staticmethod
    def _stderr_summary(stderr: str) -> str:
        lines = [line.strip() for line in str(stderr or "").splitlines() if line.strip()]
        for line in reversed(lines):
            lowered = line.lower()
            if "error" in lowered or "exception" in lowered:
                return line
        return lines[-1] if lines else ""

    @staticmethod
    def _classify_stderr_line(line: str) -> tuple[int, str]:
        message = str(line or "").strip()
        if not message:
            return logging.DEBUG, ""
        lowered = message.lower()
        if "xet storage is enabled for this repo" in lowered and "hf_xet" in lowered:
            return (
                logging.WARNING,
                f"{message} | Install `hf_xet` (or `huggingface_hub[hf_xet]`) for faster downloads and include it in the packaged runtime if you ship a final exe.",
            )
        if "torch was not compiled with flash attention" in lowered:
            return (
                logging.WARNING,
                f"{message} | If you want this optimization, ship a Torch/CUDA runtime with flash-attention support and verify it from Settings before packaging the final exe.",
            )
        if "scaled_dot_product_attention(" in lowered:
            return logging.INFO, message
        if " - error - " in lowered or lowered.startswith("traceback"):
            return logging.ERROR, message
        if " - warning - " in lowered or "userwarning:" in lowered:
            return logging.WARNING, message
        if " - info - " in lowered:
            return logging.INFO, message
        return logging.WARNING, message

    def _on_error(self, error, process: QProcess | None = None) -> None:  # noqa: ANN001
        if not self._qobject_alive():
            return
        process = process or self._process
        if process is None or process is not self._process:
            return
        details = (
            f"Worker process error: {error} | program={self.worker_program} "
            f"| cwd={self.repo_root} | runtime={self.runtime_layout.root}"
        )
        LOGGER.error(details)
        self._stderr_buffer = bounded_diagnostic_append(self._stderr_buffer, details + "\n")
        if error == QProcess.ProcessError.FailedToStart:
            self._error_message = self._error_message or details
        if error == QProcess.ProcessError.FailedToStart:
            QTimer.singleShot(0, lambda process=process: self._finalize_failed_start(process))

    def _finalize_failed_start(self, process: QProcess) -> None:
        if process is not self._process or not self._qobject_alive():
            return
        self._on_finished(process, -1, QProcess.ExitStatus.CrashExit)

    def _on_finished(self, process: QProcess | None, exit_code: int, exit_status) -> None:  # noqa: ANN001
        if not self._qobject_alive():
            return
        if process is None:
            process = self._process
        self._clear_shutdown_escalation(process)
        if process is not self._process:
            LOGGER.info("Previous production worker exited: code=%s status=%s", exit_code, exit_status)
            self._delete_process_later(process)
            return
        self._read_stdout(process)
        self._read_stderr(process)
        self._drain_stdout_lines()
        self._flush_progress()
        self._persist_diagnostics()
        had_pending_daemon_request = self._pending_daemon_request is not None
        self._process = None
        try:
            daemon_process = bool(process.property("daemon_process")) if process is not None else self._daemon_process
        except RuntimeError:
            daemon_process = self._daemon_process
        self._daemon_process = False
        self._stopping_idle_daemon = False
        self._pending_daemon_request = None
        self._pending_daemon_line = None
        was_running = self._running_request
        self._remove_request_file()
        if self._cancel_requested:
            self._generation += 1
            if self._result_job is not None:
                self._result_job.cancel()
            cleanup_paths_async(self._result_path)
            self._result_path = None
            self._pending_exit = None
            self._running_request = False
            self.running_changed.emit(False)
            self._delete_process_later(process)
            self._emit_cancelled_once()
            return
        if daemon_process and not was_running:
            self._running_request = False
            self.running_changed.emit(False)
            if had_pending_daemon_request:
                message = self._error_message or self._stderr_summary(self._stderr_buffer) or (
                    f"Persistent worker exited before accepting the request with code {exit_code} ({exit_status})"
                )
                self._emit_failed_once(message)
                self._delete_process_later(process)
                return
            LOGGER.info("Persistent production worker exited while idle: code=%s status=%s", exit_code, exit_status)
            self._delete_process_later(process)
            return
        if self._result_path is not None and int(exit_code) == 0:
            self._pending_exit = (process, int(exit_code), exit_status)
            self._delete_process_later(process)
            self._start_result_parse()
            return
        if self._result_payload is not None and int(exit_code) == 0:
            self._running_request = False
            self.running_changed.emit(False)
            self._delete_process_later(process)
            self._emit_completed_payload()
            return
        self._running_request = False
        self.running_changed.emit(False)
        message = self._error_message or self._stderr_summary(self._stderr_buffer) or f"Worker exited with code {exit_code} ({exit_status})"
        self._emit_failed_once(message)
        self._delete_process_later(process)

    def _remove_request_file(self) -> None:
        request_file = self._request_file
        self._request_file = None
        cleanup_paths_async(request_file)

    def _persist_diagnostics(self) -> None:
        self._diagnostic_future = write_process_logs_async(
            self._stdout_log_path,
            self._stdout_diagnostic,
            self._stderr_log_path,
            self._stderr_buffer,
        )

    @staticmethod
    def _delete_process_later(process: QProcess | None) -> None:
        if process is None:
            return
        try:
            process.deleteLater()
        except RuntimeError:
            pass

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

    def _complete_daemon_result(self) -> None:
        if not self._running_request or self._result_payload is None:
            return
        self._running_request = False
        self._flush_progress()
        self._persist_diagnostics()
        self.running_changed.emit(False)
        self._emit_completed_payload()
        if not self._keep_worker_warm:
            self._stop_idle_daemon()

    def _emit_completed_payload(self) -> None:
        payload = dict(self._result_payload or {})
        self._result_payload = None
        if self._terminal_emitted:
            return
        self._flush_progress()
        self._terminal_emitted = True
        self.completed.emit(payload)

    def _handle_daemon_failure(self) -> None:
        if not self._running_request:
            return
        message = self._error_message or "Persistent worker failed"
        self._running_request = False
        self._flush_progress()
        self._persist_diagnostics()
        self.running_changed.emit(False)
        self._emit_failed_once(message)

    def _emit_failed_once(self, message: str) -> None:
        if self._terminal_emitted:
            return
        self._flush_progress()
        self._terminal_emitted = True
        self.failed.emit(self._explicit_failure_message(str(message)))

    def _emit_cancelled_once(self) -> None:
        if self._terminal_emitted:
            return
        self._flush_progress()
        self._terminal_emitted = True
        self.cancelled.emit()

    def _explicit_failure_message(self, message: str) -> str:
        request = self._request
        if request is None or request.preferred_execution_mode == "cpu":
            return str(message)
        return (
            f"{message}\n\n"
            "No CPU retry was started automatically, so the failed run was not duplicated. "
            "Choose CPU in Settings and retry if you want an explicit fallback run."
        )
