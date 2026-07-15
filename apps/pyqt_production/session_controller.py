from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import QObject, QProcess, QProcessEnvironment, pyqtSignal

from apps.pyqt_production.worker_protocol import ProductionClusterRequest
from apps.shared.runtime_support import RuntimeLayout


LOGGER = logging.getLogger(__name__)


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
        self.worker_program = str(Path(sys.executable).resolve())
        self._process: QProcess | None = None
        self._request: ProductionClusterRequest | None = None
        self._stdout_buffer = ""
        self._stderr_buffer = ""
        self._stderr_line_buffer = ""
        self._result_payload: dict[str, object] | None = None
        self._error_message = ""
        self._cancel_requested = False
        self._fallback_attempted = False
        self._request_file: Path | None = None
        self._stdout_log_path: Path | None = None
        self._stderr_log_path: Path | None = None
        self._keep_worker_warm = False
        self._running_request = False
        self._daemon_process = False
        self._pending_daemon_request: ProductionClusterRequest | None = None

    def is_running(self) -> bool:
        return self._running_request

    def is_worker_warm(self) -> bool:
        return (
            self._keep_worker_warm
            and self._process is not None
            and self._process.state() != QProcess.ProcessState.NotRunning
            and not self._running_request
        )

    def set_keep_worker_warm(self, enabled: bool) -> None:
        self._keep_worker_warm = bool(enabled)
        if not self._keep_worker_warm and self.is_worker_warm():
            self._stop_idle_daemon()

    def start(self, request: ProductionClusterRequest) -> bool:
        if self.is_running():
            return False
        self._request = request
        self._fallback_attempted = False
        self._cancel_requested = False
        if self._keep_worker_warm:
            self._start_or_send_daemon(request)
        else:
            self._start_process(request)
        return True

    def cancel(self) -> None:
        self._cancel_requested = True
        if self._process is None:
            return
        self._process.kill()

    def shutdown(self, timeout_ms: int = 2500) -> bool:
        process = self._process
        if process is None:
            return True
        self._cancel_requested = True
        try:
            process.kill()
        except Exception:
            return False
        try:
            return bool(process.waitForFinished(int(timeout_ms)))
        except Exception:
            return False

    def _stop_idle_daemon(self) -> None:
        process = self._process
        if process is None or self._running_request:
            return
        try:
            process.write(json.dumps({"type": "shutdown", "payload": {}}).encode("utf-8") + b"\n")
            process.closeWriteChannel()
        except Exception:
            try:
                process.kill()
            except Exception:
                return

    def _reset_run_buffers(self) -> None:
        self._result_payload = None
        self._error_message = ""
        self._stdout_buffer = ""
        self._stderr_buffer = ""
        self._stderr_line_buffer = ""

    def _prepare_worker_logs(self, *, truncate: bool) -> None:
        self._stdout_log_path = self.runtime_layout.logs_dir / "worker_stdout.log"
        self._stderr_log_path = self.runtime_layout.logs_dir / "worker_stderr.log"
        for log_path in (self._stdout_log_path, self._stderr_log_path):
            if log_path is None:
                continue
            log_path.parent.mkdir(parents=True, exist_ok=True)
            if truncate:
                log_path.write_text("", encoding="utf-8")

    def _worker_arguments(self, *, request_json: Path | None = None, daemon: bool = False) -> list[str]:
        if bool(getattr(sys, "frozen", False)):
            args = ["--worker"]
        else:
            args = ["-m", "apps.pyqt_production.worker"]
        if daemon:
            args.append("--daemon")
        elif request_json is not None:
            args.extend(["--request-json", str(request_json)])
        return args

    def _start_process(self, request: ProductionClusterRequest) -> None:
        self._reset_run_buffers()
        self._running_request = True
        self._daemon_process = False
        request_dir = self.runtime_layout.cache_dir / "tmp"
        request_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self._request_file = request_dir / f"cluster_request_{stamp}.json"
        self._request_file.write_text(json.dumps(request.as_dict(), indent=2), encoding="utf-8")
        self._prepare_worker_logs(truncate=True)

        process = QProcess(self)
        process.setProperty("daemon_process", False)
        process.setProgram(self.worker_program)
        process.setArguments(self._worker_arguments(request_json=self._request_file))
        process.setWorkingDirectory(str(self.repo_root))
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("IMAGE_CLUSTERING_APP_DIR", str(self.runtime_layout.root))
        environment.insert("IMAGE_CLUSTERING_MODEL_ASSETS_DIR", str(self.runtime_layout.model_assets_dir))
        environment.insert("PYTHONUNBUFFERED", "1")
        for key, value in os.environ.items():
            if key.startswith("QT_") or key in {"PATH", "PYTHONPATH"}:
                environment.insert(key, value)
        process.setProcessEnvironment(environment)
        process.readyReadStandardOutput.connect(self._read_stdout)
        process.readyReadStandardError.connect(self._read_stderr)
        process.finished.connect(
            lambda exit_code, exit_status, process=process: self._on_finished(process, exit_code, exit_status)
        )
        process.errorOccurred.connect(self._on_error)
        self._process = process
        LOGGER.info(
            "Starting production worker with program=%s args=%s cwd=%s runtime=%s request=%s",
            self.worker_program,
            process.arguments(),
            self.repo_root,
            self.runtime_layout.root,
            self._request_file,
        )
        process.start()
        self.started.emit()
        self.running_changed.emit(True)

    def _start_or_send_daemon(self, request: ProductionClusterRequest) -> None:
        self._reset_run_buffers()
        self._request_file = None
        self._prepare_worker_logs(truncate=True)
        process = self._process
        if process is not None and process.state() != QProcess.ProcessState.NotRunning and self._daemon_process:
            self._send_daemon_request(request)
            return
        self._daemon_process = True
        self._pending_daemon_request = request
        process = QProcess(self)
        process.setProperty("daemon_process", True)
        process.setProgram(self.worker_program)
        process.setArguments(self._worker_arguments(daemon=True))
        process.setWorkingDirectory(str(self.repo_root))
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("IMAGE_CLUSTERING_APP_DIR", str(self.runtime_layout.root))
        environment.insert("IMAGE_CLUSTERING_MODEL_ASSETS_DIR", str(self.runtime_layout.model_assets_dir))
        environment.insert("PYTHONUNBUFFERED", "1")
        for key, value in os.environ.items():
            if key.startswith("QT_") or key in {"PATH", "PYTHONPATH"}:
                environment.insert(key, value)
        process.setProcessEnvironment(environment)
        process.readyReadStandardOutput.connect(self._read_stdout)
        process.readyReadStandardError.connect(self._read_stderr)
        process.finished.connect(
            lambda exit_code, exit_status, process=process: self._on_finished(process, exit_code, exit_status)
        )
        process.errorOccurred.connect(self._on_error)
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
        self._pending_daemon_request = None
        if request is not None:
            self._send_daemon_request(request)

    def _send_daemon_request(self, request: ProductionClusterRequest) -> None:
        if self._process is None:
            self.failed.emit("Persistent worker is not available.")
            return
        self._request = request
        self._running_request = True
        line = json.dumps({"type": "request", "payload": request.as_dict()}) + "\n"
        self._process.write(line.encode("utf-8"))
        LOGGER.info(
            "Sent request to persistent production worker | runtime=%s models=%s backends=%s profile=%s",
            self.runtime_layout.root,
            ",".join(request.embedding_models),
            ",".join(request.clustering_backends),
            request.performance_profile,
        )
        self.started.emit()
        self.running_changed.emit(True)

    def _read_stdout(self) -> None:
        if self._process is None:
            return
        text = bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._stdout_buffer += text
        if text and self._stdout_log_path is not None:
            with self._stdout_log_path.open("a", encoding="utf-8") as handle:
                handle.write(text)
        self._drain_stdout_lines()

    def _drain_stdout_lines(self) -> None:
        while "\n" in self._stdout_buffer:
            line, self._stdout_buffer = self._stdout_buffer.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                LOGGER.warning("Ignoring non-JSON worker line: %s", line)
                continue
            event_type = str(payload.get("type", ""))
            body = dict(payload.get("payload") or {})
            if event_type == "progress":
                self.progress.emit(int(body.get("value", -1) or -1), str(body.get("status", "")))
            elif event_type == "result":
                self._result_payload = body
                if self._daemon_process:
                    self._complete_daemon_result()
            elif event_type == "error":
                self._error_message = str(body.get("message", "Worker failed"))
                if body.get("traceback"):
                    LOGGER.error("Worker traceback:\n%s", body.get("traceback"))
                if self._daemon_process:
                    self._handle_daemon_failure()
            elif event_type == "ready":
                LOGGER.info("Persistent production worker ready: %s", body.get("status", "ready"))

    def _read_stderr(self) -> None:
        if self._process is None:
            return
        text = bytes(self._process.readAllStandardError()).decode("utf-8", errors="replace")
        self._stderr_buffer += text
        self._stderr_line_buffer += text
        if text and self._stderr_log_path is not None:
            with self._stderr_log_path.open("a", encoding="utf-8") as handle:
                handle.write(text)
        self._drain_stderr_lines()

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
        if level >= logging.ERROR and not self._error_message:
            self._error_message = message

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

    def _on_error(self, error) -> None:  # noqa: ANN001
        details = (
            f"Worker process error: {error} | program={self.worker_program} "
            f"| cwd={self.repo_root} | runtime={self.runtime_layout.root}"
        )
        LOGGER.error(details)
        self._error_message = self._error_message or details
        if self._stderr_log_path is not None:
            with self._stderr_log_path.open("a", encoding="utf-8") as handle:
                handle.write(details + "\n")

    def _on_finished(self, process: QProcess | None, exit_code: int, exit_status) -> None:  # noqa: ANN001
        self._drain_stdout_lines()
        self._drain_stderr_lines(flush=True)
        if process is None:
            process = self._process
        if process is not self._process:
            LOGGER.info("Previous production worker exited: code=%s status=%s", exit_code, exit_status)
            if process is not None:
                process.deleteLater()
            return
        self._process = None
        daemon_process = bool(process.property("daemon_process")) if process is not None else self._daemon_process
        self._daemon_process = False
        was_running = self._running_request
        self._running_request = False
        self.running_changed.emit(False)
        if self._cancel_requested:
            if process is not None:
                process.deleteLater()
            self.cancelled.emit()
            return
        if daemon_process and not was_running:
            LOGGER.info("Persistent production worker exited while idle: code=%s status=%s", exit_code, exit_status)
            if process is not None:
                process.deleteLater()
            return
        if self._result_payload is not None and int(exit_code) == 0:
            if process is not None:
                process.deleteLater()
            self._emit_completed_payload()
            return
        if (
            not self._fallback_attempted
            and self._request is not None
            and self._request.preferred_execution_mode != "cpu"
        ):
            self._fallback_attempted = True
            fallback = ProductionClusterRequest(**{**self._request.as_dict(), "preferred_execution_mode": "cpu"})
            self.progress.emit(-1, "Worker failed; retrying on CPU fallback")
            if process is not None:
                process.deleteLater()
            if self._keep_worker_warm:
                self._start_or_send_daemon(fallback)
            else:
                self._start_process(fallback)
            return
        message = self._error_message or self._stderr_buffer.strip() or f"Worker exited with code {exit_code} ({exit_status})"
        self.failed.emit(message)
        if process is not None:
            process.deleteLater()

    def _complete_daemon_result(self) -> None:
        if not self._running_request or self._result_payload is None:
            return
        self._running_request = False
        self.running_changed.emit(False)
        self._emit_completed_payload()

    def _emit_completed_payload(self) -> None:
        payload = dict(self._result_payload or {})
        self._result_payload = None
        self.completed.emit(payload)

    def _handle_daemon_failure(self) -> None:
        if not self._running_request:
            return
        if (
            not self._fallback_attempted
            and self._request is not None
            and self._request.preferred_execution_mode != "cpu"
        ):
            self._fallback_attempted = True
            fallback = ProductionClusterRequest(**{**self._request.as_dict(), "preferred_execution_mode": "cpu"})
            self.progress.emit(-1, "Worker failed; retrying on CPU fallback")
            self._reset_run_buffers()
            self._send_daemon_request(fallback)
            return
        message = self._error_message or "Persistent worker failed"
        self._running_request = False
        self.running_changed.emit(False)
        self.failed.emit(message)
